#!/usr/bin/env python3
import json
import time
import array
import threading
import math
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.action import ActionClient

from vosk import Model, KaldiRecognizer
from ament_index_python.packages import get_package_share_directory

from audio_common_msgs.msg import AudioStamped
from audio_common_msgs.action import TTS

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


# ----------------- PARAMS -----------------
SAMPLE_RATE = 16000               # doit matcher la source audio

CONF_THRESH_FINAL = 0.85          # monte un peu pour réduire les faux positifs
DEBOUNCE_MS = 600                 # anti-répétitions sur final

ANSWER_TIMEOUT_S = 10.0           # délai max pour répondre à UNE question
SESSION_TIMEOUT_S = 50.0          # délai max total
MAX_RETRIES = 1                   # 1 retry => 2 chances au total

AUDIO_TOPIC = "/audio_mic"
TTS_ACTION = "/say"

# Désactive partial par défaut (beaucoup plus fiable en environnement bruité)
ENABLE_PARTIAL = False

# Anti-echo: ignore ASR pendant TTS + un tail après la fin
TTS_TAIL_S = 1.0

# Garde de bruit simple: ignore trames trop faibles
# Ajuste selon ton micro/bruit ambiant.
MIN_RMS = 250.0

# Sécurité médicale: "no" doit être confirmé 2 fois en FINAL
# (réduit fortement les urgences déclenchées par bruit)
NO_FINAL_CONFIRMATIONS_REQUIRED = 2
NO_CONFIRM_WINDOW_S = 2.5

# Cooldown après fin du dialogue (laisser le patient se relever)
POST_DIALOG_COOLDOWN_S = 10.0

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PLACES = ["home", "hospital"]
YESNO = ["yes", "no"]
# ------------------------------------------


def get_model_path() -> str:
    share_dir = get_package_share_directory("ai_pkg")
    return str(Path(share_dir) / "models" / "vosk-model-small-en-us-0.15")


def pcm_rms_int16(pcm_bytes: bytes) -> float:
    """RMS simple sur PCM int16 little-endian."""
    if not pcm_bytes:
        return 0.0
    a = array.array("h")
    a.frombytes(pcm_bytes)
    if len(a) == 0:
        return 0.0
    # RMS = sqrt(mean(x^2))
    acc = 0.0
    for x in a:
        acc += float(x) * float(x)
    return math.sqrt(acc / float(len(a)))


class DialogNode(Node):
    """
    Dialogue patient piloté par sf.dialog_active.

    Objectif:
    - Vérifier conscience/orientation avec questions simples en choix limité:
        1) yes/no
        2) day of week
        3) place (home/hospital)

    Sécurités anti faux positifs:
    - ASR ignoré pendant le TTS + petit tail
    - grammaire Vosk réduite par étape (recréation recognizer)
    - partial désactivé (option)
    - seuil RMS minimum
    - "no" => urgence seulement sur FINAL + double confirmation
    - debounce par mot détecté
    - Quand on passe à DONE, on attend POST_DIALOG_COOLDOWN_S avant de mettre
      sf.dialog_active=False. Ça bloque la FSM dans DIALOG pendant 10s pour laisser
      le patient se relever et éviter un re-trigger immédiat de la fall detection.
    """

    # ---- mini FSM interne ----
    STEP_ASK_OK = "ASK_OK"
    STEP_ASK_DAY = "ASK_DAY"
    STEP_ASK_PLACE = "ASK_PLACE"
    STEP_DONE = "DONE"

    def __init__(self):
        super().__init__("dialog_node")

        self._lock = threading.Lock()

        # Session state
        self._session_running = False
        self._prev_dialog_active = False

        self._step = self.STEP_ASK_OK
        self._step_started_t = 0.0
        self._session_started_t = 0.0
        self._retries = 0

        # Anti double trigger (final)
        self._last_emit_ms = {}

        # NO confirmation
        self._no_final_hits = 0
        self._no_first_hit_t = 0.0

        # Cooldown fin de dialogue
        self._done_cooldown_until = 0.0

        # ---- Vosk model ----
        model_path = get_model_path()
        log(f"[DIALOG] Loading Vosk model: {model_path}")
        self.model = Model(model_path)

        # recognizer (créé par étape)
        self.recognizer = None
        self._recreate_recognizer_for_step(self.STEP_ASK_OK)

        # ---- Audio sub ----
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.subscription = self.create_subscription(
            AudioStamped,
            AUDIO_TOPIC,
            self.audio_callback,
            qos,
        )

        # ---- TTS ----
        self.tts_client = ActionClient(self, TTS, TTS_ACTION)
        self._tts_ready = False
        self._tts_checked_once = False

        # TTS speaking gate (anti-echo)
        self._tts_speaking = False
        self._tts_speaking_until = 0.0

        # Timer: manage session start/stop + timeouts + cooldown DONE
        self.timer = self.create_timer(0.1, self.loop)

        log(f"[DIALOG] Ready. Listening on {AUDIO_TOPIC}")

    # ---------------- Step-specific grammar ----------------
    def _allowed_keywords_for_step(self, step: str):
        if step == self.STEP_ASK_OK:
            return YESNO
        if step == self.STEP_ASK_DAY:
            return DAYS
        if step == self.STEP_ASK_PLACE:
            return PLACES
        return []

    def _recreate_recognizer_for_step(self, step: str):
        """Recrée le recognizer avec une grammaire très restrictive selon l'étape."""
        allowed = self._allowed_keywords_for_step(step)
        grammar = json.dumps(allowed + ["[unk]"])
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE, grammar)
        try:
            self.recognizer.SetWords(True)
        except Exception:
            pass

        # reset debounces for allowed words
        self._last_emit_ms = {k: 0.0 for k in allowed}

    def _reset_recognizer(self):
        try:
            self.recognizer.Reset()
        except Exception:
            pass

    # ---------------- TTS ----------------
    def _ensure_tts_ready_once(self):
        if self._tts_checked_once:
            return self._tts_ready
        self._tts_checked_once = True
        self._tts_ready = self.tts_client.wait_for_server(timeout_sec=1.5)
        if not self._tts_ready:
            log("[DIALOG] ⚠️ TTS server /say not available (yet).")
        return self._tts_ready

    def say(self, text, lang="en", volume=1.0, rate=1.0):
        if not self._ensure_tts_ready_once():
            return

        goal = TTS.Goal()
        goal.text = text
        goal.language = lang
        goal.volume = float(volume)
        goal.rate = float(rate)

        log(f"[DIALOG] 🗣️ {text}")

        # Mark speaking immediately (best effort)
        self._tts_speaking = True

        future = self.tts_client.send_goal_async(goal)

        def _goal_response_cb(fut):
            try:
                goal_handle = fut.result()
            except Exception:
                self._tts_speaking = False
                self._tts_speaking_until = time.monotonic() + TTS_TAIL_S
                return

            if not goal_handle.accepted:
                self._tts_speaking = False
                self._tts_speaking_until = time.monotonic() + TTS_TAIL_S
                return

            result_future = goal_handle.get_result_async()

            def _result_cb(_):
                self._tts_speaking = False
                self._tts_speaking_until = time.monotonic() + TTS_TAIL_S

            result_future.add_done_callback(_result_cb)

        future.add_done_callback(_goal_response_cb)

    # ---------------- Session control ----------------
    def _start_session(self):
        with self._lock:
            self._session_running = True
            self._step = self.STEP_ASK_OK
            self._retries = 0
            self._session_started_t = time.monotonic()
            self._step_started_t = time.monotonic()

            self._no_final_hits = 0
            self._no_first_hit_t = 0.0

            # reset cooldown
            self._done_cooldown_until = 0.0

            self._tts_ready = False
            self._tts_checked_once = False

            self._recreate_recognizer_for_step(self.STEP_ASK_OK)
            self._reset_recognizer()

        log("[DIALOG] Session started")
        self.say("Can you hear me?")

        # IMPORTANT: On évite de prononcer "YES or NO" en TTS pour limiter l'écho lexical

    def _stop_session(self):
        # IMPORTANT: ne pas désactiver sf.dialog_active tout de suite
        with self._lock:
            self._session_running = False
            self._step = self.STEP_DONE
            self._done_cooldown_until = time.monotonic() + POST_DIALOG_COOLDOWN_S

        log(f"[DIALOG] Session finished -> cooldown {POST_DIALOG_COOLDOWN_S:.1f}s before dialog_active=False")

    def call_ambulance(self, reason=""):
        if reason:
            log(f"[DIALOG] 🚑 EMERGENCY ({reason})")
        else:
            log("[DIALOG] 🚑 EMERGENCY")
        self.say("Emergency. I am calling for help now.")
        self._stop_session()

    # ---------------- Helpers: questions / timeouts ----------------
    def _ask_current_step(self):
        if self._step == self.STEP_ASK_OK:
            self.say("Are you okay?")
        elif self._step == self.STEP_ASK_DAY:
            self.say("What day is it today?")
        elif self._step == self.STEP_ASK_PLACE:
            self.say("Where are you? Home or hospital.")

    def _advance_step(self):
        self._retries = 0
        self._step_started_t = time.monotonic()

        if self._step == self.STEP_ASK_OK:
            self._step = self.STEP_ASK_DAY
            self._recreate_recognizer_for_step(self._step)
            self._reset_recognizer()
            self.say("Thank you. What day is it today?")
        elif self._step == self.STEP_ASK_DAY:
            self._step = self.STEP_ASK_PLACE
            self._recreate_recognizer_for_step(self._step)
            self._reset_recognizer()
            self.say("Thank you. Where are you? Home or hospital?")
        elif self._step == self.STEP_ASK_PLACE:
            self._step = self.STEP_DONE
            self.say("Thank you. You seem conscious and oriented.")
            self._stop_session()

    def _handle_invalid_or_timeout(self, what=""):
        if self._retries < MAX_RETRIES:
            self._retries += 1
            self._step_started_t = time.monotonic()
            if what:
                self.say(f"I did not understand. {what}")
            else:
                self.say("I did not understand. Please repeat.")
            self._ask_current_step()
        else:
            self.call_ambulance("no valid answer")

    # ---------------- Main loop ----------------
    def loop(self):
        # start session on rising edge
        if sf.dialog_active and not self._prev_dialog_active:
            self._start_session()

        # if FSM forces dialog off, stop silently
        if (not sf.dialog_active) and self._prev_dialog_active:
            with self._lock:
                self._session_running = False
                self._step = self.STEP_DONE
                self._done_cooldown_until = 0.0
            log("[DIALOG] FSM disabled dialog_active -> stop session (silent)")

        self._prev_dialog_active = sf.dialog_active

        now = time.monotonic()

        # Cooldown post-DONE : on retarde la libération de la FSM
        with self._lock:
            step = self._step
            cooldown_until = self._done_cooldown_until
            session_running = self._session_running
            step_t = self._step_started_t
            sess_t = self._session_started_t

        if step == self.STEP_DONE and cooldown_until > 0.0:
            if now >= cooldown_until:
                log("[DIALOG] Cooldown done -> dialog_active=False (FSM can advance)")
                sf.dialog_active = False
                with self._lock:
                    self._done_cooldown_until = 0.0
            return  # pendant cooldown: on ne fait rien d'autre

        # manage timeouts
        if not session_running:
            return

        if now - sess_t > SESSION_TIMEOUT_S:
            self.call_ambulance("session timeout")
            return

        if step != self.STEP_DONE and (now - step_t > ANSWER_TIMEOUT_S):
            self._handle_invalid_or_timeout("Please answer now.")

    # ---------------- Recognition ----------------
    def _emit_word_final(self, word: str, conf: float):
        now_ms = time.time() * 1000.0

        # per-step debounce
        last = self._last_emit_ms.get(word, 0.0)
        if now_ms - last < DEBOUNCE_MS:
            return
        self._last_emit_ms[word] = now_ms

        self._on_word(word, conf, source="final")

    def _handle_no_final_confirm(self):
        """Double confirmation pour NO (FINAL only) afin d'éviter les faux positifs."""
        now = time.monotonic()
        if self._no_final_hits == 0:
            self._no_final_hits = 1
            self._no_first_hit_t = now
            self.say("Could you repeat: are you okay?")
            # reset timer for the step to give time to answer
            self._step_started_t = time.monotonic()
            return

        # if too late, reset and treat as first hit again
        if now - self._no_first_hit_t > NO_CONFIRM_WINDOW_S:
            self._no_final_hits = 1
            self._no_first_hit_t = now
            self.say("Please repeat clearly !")
            self._step_started_t = time.monotonic()
            return

        self._no_final_hits += 1
        if self._no_final_hits >= NO_FINAL_CONFIRMATIONS_REQUIRED:
            self.call_ambulance("patient said NO (confirmed)")

    def _on_word(self, word: str, conf: float, source=""):
        with self._lock:
            if not self._session_running:
                return
            step = self._step

        log(f"[DIALOG] DETECTED ({source}): {word.upper()} (conf={conf:.2f}) step={step}")

        # Sécurité: on ne déclenche jamais l'urgence sur un partial
        if word == "no":
            if source != "final":
                return
            # double confirmation
            self._handle_no_final_confirm()
            return

        if step == self.STEP_ASK_OK:
            if word == "yes":
                self._advance_step()
            else:
                self._handle_invalid_or_timeout("Please answer yes or no.")
            return

        if step == self.STEP_ASK_DAY:
            if word in DAYS:
                today = DAYS[datetime.now().weekday()]  # ex: "wednesday"

                if word == today:
                    self._advance_step()
                else:
                    # Mauvais jour → désorientation temporelle
                    self._handle_invalid_or_timeout(
                        f"That is not correct. Today is {today}. Please repeat."
                    )
            else:
                self._handle_invalid_or_timeout("Please say a day of the week.")
            return

        if step == self.STEP_ASK_PLACE:
            if word in PLACES:
                self._advance_step()
            else:
                self._handle_invalid_or_timeout("Please say home or hospital.")
            return

    def audio_callback(self, msg: AudioStamped):
        if not sf.dialog_active:
            return

        with self._lock:
            if not self._session_running:
                return

        # Anti-echo: ignore ASR while TTS is speaking (and short tail)
        now = time.monotonic()
        if self._tts_speaking or now < self._tts_speaking_until:
            return

        data = msg.audio.audio_data
        if not data.int16_data:
            return

        pcm = array.array("h", data.int16_data).tobytes()
        if not pcm:
            return

        # Noise gate
        if pcm_rms_int16(pcm) < MIN_RMS:
            return

        # Vosk: final OR partial
        is_final = self.recognizer.AcceptWaveform(pcm)

        if is_final:
            try:
                res = json.loads(self.recognizer.Result())
            except Exception:
                return

            text = (res.get("text") or "").strip().lower()
            if not text:
                return

            # mots pertinents (avec conf)
            words = res.get("result", [])

            # si "result" absent, fallback sur "text"
            if not words and text:
                words = [{"word": text, "conf": 0.0}]

            for w in words:
                word = (w.get("word") or "").strip().lower()
                conf = float(w.get("conf", 0.0))

                # grammaire déjà restrictive, mais on garde une barrière conf
                if conf < CONF_THRESH_FINAL:
                    continue

                # Comme le recognizer est par étape, si word sort, c'est normalement attendu.
                # On filtre quand même par allowed.
                allowed = set(self._allowed_keywords_for_step(self._step))
                if word in allowed:
                    self._emit_word_final(word, conf)

        else:
            if not ENABLE_PARTIAL:
                return

            # PartialResult (optionnel, déconseillé en milieu bruité)
            try:
                pres = json.loads(self.recognizer.PartialResult())
            except Exception:
                return
            ptxt = (pres.get("partial") or "").strip().lower()
            if not ptxt:
                return

            allowed = set(self._allowed_keywords_for_step(self._step))
            tokens = ptxt.split()
            # on prend le premier token autorisé
            detected = None
            for t in tokens:
                if t in allowed:
                    detected = t
                    break
            if not detected:
                return

            # sécurité: jamais "no" via partial
            if detected == "no":
                return

            # partial => on n'avance pas directement, on attend final (plus sûr)
            return


def main(args=None):
    rclpy.init(args=args)
    node = DialogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
