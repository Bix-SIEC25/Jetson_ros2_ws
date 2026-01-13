#!/usr/bin/env python3
import json
import time
import array
import threading
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
SAMPLE_RATE = 16000 # A VERIFIER !!!!!

CONF_THRESH_FINAL = 0.75        # pour les résultats "final" (avec conf)
DEBOUNCE_MS = 500               # anti-répétitions
PARTIAL_HITS_REQUIRED = 3       # stabilité minimale sur partial pour valider un mot
PARTIAL_DEBOUNCE_MS = 800       # encore plus strict sur partial

ANSWER_TIMEOUT_S = 8.0          # délai max pour répondre à UNE question
SESSION_TIMEOUT_S = 40.0        # délai max total
MAX_RETRIES = 1                 # 1 retry => 2 chances au total

AUDIO_TOPIC = "/audio_mic"
TTS_ACTION = "/say"

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PLACES = ["home", "hospital"]
KEYWORDS = ["yes", "no"] + DAYS + PLACES
# ------------------------------------------


def get_model_path() -> str:
    share_dir = get_package_share_directory("ai_pkg")
    return str(Path(share_dir) / "models" / "vosk-model-small-en-us-0.15")


class DialogNode(Node):
    """
    Dialogue patient piloté par sf.dialog_active.

    Objectif:
    - Vérifier conscience/orientation avec questions simples en choix limité:
        1) yes/no
        2) day of week
        3) place (home/hospital)
    - Si échec (no, timeouts, réponses invalides) => "urgence" (call_ambulance())
    - Sinon => ok et rend la main à la FSM (sf.dialog_active=False)

    Améliorations vs version précédente:
    - timeouts + retries
    - traite aussi PartialResult (plus réactif)
    - évite double TTS spam
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

        # anti double trigger
        self._last_emit_ms = {k: 0.0 for k in KEYWORDS}

        # partial stability
        self._partial_last_word = None
        self._partial_hits = 0
        self._last_partial_emit_ms = {k: 0.0 for k in KEYWORDS}

        # ---- Vosk ----
        model_path = get_model_path()
        log(f"[DIALOG] Loading Vosk model: {model_path}")
        model = Model(model_path)

        grammar = json.dumps(KEYWORDS + ["[unk]"])
        self.recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)

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

        # Timer: manage session start/stop + timeouts
        self.timer = self.create_timer(0.1, self.loop)

        log(f"[DIALOG] Ready. Listening on {AUDIO_TOPIC}")

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
        self.tts_client.send_goal_async(goal)

    # ---------------- Session control ----------------
    def _reset_recognizer(self):
        try:
            self.recognizer.Reset()
        except Exception:
            pass

    def _start_session(self):
        with self._lock:
            self._session_running = True
            self._step = self.STEP_ASK_OK
            self._retries = 0
            self._session_started_t = time.monotonic()
            self._step_started_t = time.monotonic()

            self._last_emit_ms = {k: 0.0 for k in KEYWORDS}
            self._partial_last_word = None
            self._partial_hits = 0
            self._last_partial_emit_ms = {k: 0.0 for k in KEYWORDS}

            self._tts_ready = False
            self._tts_checked_once = False
            self._reset_recognizer()

        log("[DIALOG] Session started")
        self.say("Hello. Can you hear me? Please say YES or NO.")

    def _stop_session(self):
        with self._lock:
            self._session_running = False
            self._step = self.STEP_DONE

        log("[DIALOG] Session finished -> dialog_active=False")
        sf.dialog_active = False

    def call_ambulance(self, reason=""):
        if reason:
            log(f"[DIALOG] 🚑 EMERGENCY ({reason})")
        else:
            log("[DIALOG] 🚑 EMERGENCY")
        self.say("Emergency. I am calling for help now.")
        self._stop_session()

    # ---------------- Helpers: questions / timeouts ----------------
    def _ask_current_step(self):
        """Re-ask the current question."""
        if self._step == self.STEP_ASK_OK:
            self.say("Please say YES or NO. Are you okay?")
        elif self._step == self.STEP_ASK_DAY:
            self.say("What day is it today? Say Monday, Tuesday, Wednesday, Thursday, Friday, Saturday or Sunday.")
        elif self._step == self.STEP_ASK_PLACE:
            self.say("Where are you? Say HOME or HOSPITAL.")

    def _advance_step(self):
        """Go to next step and ask question."""
        self._retries = 0
        self._step_started_t = time.monotonic()

        if self._step == self.STEP_ASK_OK:
            self._step = self.STEP_ASK_DAY
            self.say("Okay. What day is it today? Say Monday, Tuesday, Wednesday, Thursday, Friday, Saturday or Sunday.")
        elif self._step == self.STEP_ASK_DAY:
            self._step = self.STEP_ASK_PLACE
            self.say("Thank you. Where are you? Say HOME or HOSPITAL.")
        elif self._step == self.STEP_ASK_PLACE:
            self._step = self.STEP_DONE
            self.say("Thank you. You seem conscious and oriented.")
            self._stop_session()

    def _handle_invalid_or_timeout(self, what=""):
        """Retry once, then emergency."""
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
            log("[DIALOG] FSM disabled dialog_active -> stop session (silent)")

        self._prev_dialog_active = sf.dialog_active

        # manage timeouts
        with self._lock:
            if not self._session_running:
                return
            step = self._step
            step_t = self._step_started_t
            sess_t = self._session_started_t

        now = time.monotonic()
        if now - sess_t > SESSION_TIMEOUT_S:
            self.call_ambulance("session timeout")
            return

        if step != self.STEP_DONE and (now - step_t > ANSWER_TIMEOUT_S):
            self._handle_invalid_or_timeout("Please answer now.")

    # ---------------- Recognition ----------------
    def _emit_word_final(self, word: str, conf: float):
        """Final word with confidence from Vosk result."""
        now_ms = time.time() * 1000.0
        if conf < CONF_THRESH_FINAL:
            return
        if now_ms - self._last_emit_ms.get(word, 0.0) < DEBOUNCE_MS:
            return
        self._last_emit_ms[word] = now_ms
        self._on_word(word, conf, source="final")

    def _emit_word_partial_stable(self, word: str):
        """Partial word with stability heuristic."""
        now_ms = time.time() * 1000.0
        if now_ms - self._last_partial_emit_ms.get(word, 0.0) < PARTIAL_DEBOUNCE_MS:
            return
        self._last_partial_emit_ms[word] = now_ms
        # conf unknown on partial; treat as 1.0 but gated by stability & stricter debounce
        self._on_word(word, 1.0, source="partial")

    def _on_word(self, word: str, conf: float, source=""):
        with self._lock:
            if not self._session_running:
                return
            step = self._step

        log(f"[DIALOG] DETECTED ({source}): {word.upper()} (conf={conf:.2f}) step={step}")

        # "no" => emergency at any time
        if word == "no":
            self.call_ambulance("patient said NO")
            return

        # Step logic
        if step == self.STEP_ASK_OK:
            if word == "yes":
                self._advance_step()
            else:
                self._handle_invalid_or_timeout("Please say YES or NO.")
            return

        if step == self.STEP_ASK_DAY:
            if word in DAYS:
                # Optionnel: vérifier si c'est le bon jour (orientation temporelle réelle)
                today = DAYS[datetime.now().weekday()]  # e.g. "monday"
                if word != today:
                    # on tolère 1 retry : il peut mal prononcer / ASR erreur
                    self._handle_invalid_or_timeout("That does not seem correct. Please repeat the day.")
                else:
                    self._advance_step()
            else:
                self._handle_invalid_or_timeout("Please say a day of the week.")
            return

        if step == self.STEP_ASK_PLACE:
            if word in PLACES:
                self._advance_step()
            else:
                self._handle_invalid_or_timeout("Please say HOME or HOSPITAL.")
            return

    def audio_callback(self, msg: AudioStamped):
        if not sf.dialog_active:
            return

        with self._lock:
            if not self._session_running:
                return

        data = msg.audio.audio_data
        if not data.int16_data:
            return

        pcm = array.array("h", data.int16_data).tobytes()
        if not pcm:
            return

        # Vosk: final OR partial
        is_final = self.recognizer.AcceptWaveform(pcm)

        if is_final:
            try:
                res = json.loads(self.recognizer.Result())
            except Exception:
                return

            text = (res.get("text") or "").strip()
            if not text:
                return

            # mots pertinents (avec conf)
            words = [w for w in res.get("result", []) if w.get("word") in KEYWORDS]
            # fallback si Vosk ne donne pas "result"
            if not words and text in KEYWORDS:
                words = [{"word": text, "conf": 1.0}]

            for w in words:
                word = w.get("word")
                conf = float(w.get("conf", 0.0))
                if word in KEYWORDS:
                    self._emit_word_final(word, conf)

        else:
            # PartialResult: plus réactif
            try:
                pres = json.loads(self.recognizer.PartialResult())
            except Exception:
                return
            ptxt = (pres.get("partial") or "").strip().lower()
            if not ptxt:
                # reset stability slowly
                with self._lock:
                    self._partial_last_word = None
                    self._partial_hits = 0
                return

            # on cherche si un keyword apparaît dans le partial
            detected = None
            for k in KEYWORDS:
                if k in ptxt.split():
                    detected = k
                    break
            if not detected:
                return

            with self._lock:
                if detected == self._partial_last_word:
                    self._partial_hits += 1
                else:
                    self._partial_last_word = detected
                    self._partial_hits = 1

                hits = self._partial_hits

            if hits >= PARTIAL_HITS_REQUIRED:
                # emit once, then require new stability to emit again
                with self._lock:
                    self._partial_hits = 0
                    self._partial_last_word = None
                self._emit_word_partial_stable(detected)


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
