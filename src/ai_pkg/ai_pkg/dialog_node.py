#!/usr/bin/env python3
import json
import time
import threading
import math
import subprocess
from pathlib import Path
from datetime import datetime
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from vosk import Model, KaldiRecognizer
from ament_index_python.packages import get_package_share_directory

from audio_common_msgs.action import TTS

import webrtcvad

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


# ----------------- PARAMS -----------------
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # int16

# (3) un peu moins strict pour les mots très courts
CONF_THRESH_FINAL_DEFAULT = 0.60
CONF_THRESH_FINAL_YESNO = 0.55
# fallback si "text" final est exactement un mot de la grammaire (même si conf absente)
ACCEPT_EXACT_TEXT_FALLBACK = True

DEBOUNCE_MS = 600

ANSWER_TIMEOUT_S = 10.0
SESSION_TIMEOUT_S = 50.0
MAX_RETRIES = 1

TTS_ACTION = "/say"
ENABLE_PARTIAL = False

TTS_TAIL_S = 1.0

# (2) Recheck TTS server si pas prêt
TTS_RECHECK_PERIOD_S = 1.0          # retente toutes les 1s si /say pas prêt
TTS_WAIT_TIMEOUT_S = 0.25           # durée max d'un wait_for_server
TTS_QUEUE_MAX = 25                  # anti explosion de queue

NO_FINAL_CONFIRMATIONS_REQUIRED = 2
NO_CONFIRM_WINDOW_S = 2.5

POST_DIALOG_COOLDOWN_S = 10.0

ALSA_DEVICE = "hw:0,0"

CHUNK_MS = 20  # webrtcvad accepte 10/20/30 ms
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_BYTES = CHUNK_SAMPLES * CHANNELS * SAMPLE_WIDTH_BYTES

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PLACES = ["home", "hospital"]
YESNO = ["yes", "no"]

# VAD
VAD_MODE = 2                 # 0 permissif -> 3 strict
VAD_START_FRAMES = 2         # frames speech consécutives pour déclencher "speech start"
VAD_HANGOVER_S = 0.25        # maintien après dernier speech (anti-coupure)

# Pré-roll (début du mot)
PRE_ROLL_MS = 200
PRE_ROLL_FRAMES = max(1, PRE_ROLL_MS // CHUNK_MS)

# Post-roll (fin du mot)
POST_ROLL_MS = 150
POST_ROLL_FRAMES = max(1, POST_ROLL_MS // CHUNK_MS)

# Flush silence (force un final Vosk)
VAD_FLUSH_FRAMES = 6
# ------------------------------------------


def get_model_path() -> str:
    share_dir = get_package_share_directory("ai_pkg")
    return str(Path(share_dir) / "models" / "vosk-model-small-en-us-0.15")


class DialogNode(Node):
    # ---- mini FSM interne ----
    STEP_ASK_OK = "ASK_OK"
    STEP_ASK_DAY = "ASK_DAY"
    STEP_ASK_PLACE = "ASK_PLACE"
    STEP_DONE = "DONE"

    def __init__(self):
        super().__init__("dialog_node")

        # Lock unique : protège session + recognizer
        self._lock = threading.Lock()

        # Session state
        self._session_running = False
        self._prev_dialog_active = False

        self._step = self.STEP_ASK_OK
        self._step_started_t = 0.0
        self._session_started_t = 0.0
        self._retries = 0

        # Anti double trigger
        self._last_emit_ms = {}

        # NO confirmation
        self._no_final_hits = 0
        self._no_first_hit_t = 0.0

        # Cooldown
        self._done_cooldown_until = 0.0

        # (1) Pause timeouts pendant TTS (vrai "pause")
        self._timers_paused = False
        self._pause_t0 = 0.0

        # ---- Vosk model ----
        model_path = get_model_path()
        log(f"[DIALOG] Loading Vosk model: {model_path}")
        self.model = Model(model_path)

        self.recognizer = None
        self._recreate_recognizer_for_step(self.STEP_ASK_OK)

        # ---- VAD WebRTC ----
        self.vad = webrtcvad.Vad(VAD_MODE)
        self._vad_in_speech = False
        self._vad_speech_streak = 0
        self._vad_last_voice_t = 0.0

        # pré-roll buffer
        self._pre_roll = deque(maxlen=PRE_ROLL_FRAMES)

        # post-roll compteur (frames restantes)
        self._post_roll_left = 0

        # ---- TTS ----
        self.tts_client = ActionClient(self, TTS, TTS_ACTION)

        # (2) readiness + recheck automatique
        self._tts_ready = False
        self._tts_last_check_t = 0.0

        # Anti-echo TTS
        self._tts_speaking = False
        self._tts_speaking_until = 0.0

        # TTS queue: garantit qu'on n'envoie JAMAIS 2 goals en parallèle
        self._tts_q = deque(maxlen=TTS_QUEUE_MAX)  # file de messages à dire
        self._tts_q_lock = threading.Lock()        # protège la file + l'état inflight
        self._tts_inflight = False                 # True = un goal TTS est en cours

        # ---- ALSA thread ----
        self._audio_thread = None
        self._audio_stop_evt = threading.Event()
        self._arecord_proc = None

        # Timer principal
        self.timer = self.create_timer(0.1, self.loop)

        log(
            f"[DIALOG] Ready. ALSA({ALSA_DEVICE}) {SAMPLE_RATE}Hz | "
            f"VAD mode={VAD_MODE} | pre-roll={PRE_ROLL_MS}ms({PRE_ROLL_FRAMES}) | "
            f"post-roll={POST_ROLL_MS}ms({POST_ROLL_FRAMES})"
        )

    # ---------------- Step-specific grammar ----------------
    def _allowed_keywords_for_step(self, step: str):
        if step == self.STEP_ASK_OK:
            return YESNO
        if step == self.STEP_ASK_DAY:
            return DAYS
        if step == self.STEP_ASK_PLACE:
            return PLACES
        return []

    def _conf_thresh_for_step(self, step: str) -> float:
        # (3) seuil plus permissif pour yes/no
        if step == self.STEP_ASK_OK:
            return CONF_THRESH_FINAL_YESNO
        return CONF_THRESH_FINAL_DEFAULT

    def _recreate_recognizer_for_step(self, step: str):
        """Protégé : évite course avec thread audio."""
        with self._lock:
            allowed = self._allowed_keywords_for_step(step)
            grammar = json.dumps(allowed + ["[unk]"])
            self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE, grammar)
            try:
                self.recognizer.SetWords(True)
            except Exception:
                pass
            self._last_emit_ms = {k: 0.0 for k in allowed}

    def _reset_recognizer(self):
        """Protégé : évite Reset pendant AcceptWaveform."""
        with self._lock:
            try:
                if self.recognizer is not None:
                    self.recognizer.Reset()
            except Exception:
                pass

    # ---------------- TTS ----------------
    def _ensure_tts_ready(self) -> bool:
        """
        (2) version robuste:
        - si prêt: ok
        - sinon: retente périodiquement (TTS_RECHECK_PERIOD_S)
        - ne bloque pas longtemps
        """
        now = time.monotonic()

        if self._tts_ready:
            return True

        # pas trop de wait_for_server en boucle
        if now - self._tts_last_check_t < TTS_RECHECK_PERIOD_S:
            return False

        self._tts_last_check_t = now

        try:
            ready = self.tts_client.wait_for_server(timeout_sec=TTS_WAIT_TIMEOUT_S)
        except Exception:
            ready = False

        self._tts_ready = bool(ready)
        if not self._tts_ready:
            log("[DIALOG] ⚠️ TTS server /say not available (retrying).")
        return self._tts_ready

    def say(self, text, lang="en", volume=1.0, rate=1.0):
        """
        Version SAFE:
        - say() met en file
        - un dispatcher envoie le prochain goal UNIQUEMENT quand le précédent est fini
        => jamais 2 TTS back-to-back en parallèle (overlap)

        (2) IMPORTANT:
        - On ne drop PLUS le texte si /say n'est pas encore prêt
        - On le met en queue, et loop() relancera _tts_kick() quand /say sera dispo
        """
        text = str(text).strip()
        if not text:
            return

        log(f"[DIALOG] 🗣️ (queue) {text}")

        with self._tts_q_lock:
            self._tts_q.append((text, lang, float(volume), float(rate)))

        self._tts_kick()

    def _tts_kick(self):
        """
        Lance un TTS si:
        - serveur prêt
        - rien en flight
        - queue non vide
        """
        if not self._ensure_tts_ready():
            return

        with self._tts_q_lock:
            if self._tts_inflight:
                return
            if not self._tts_q:
                return

            text, lang, volume, rate = self._tts_q.popleft()
            self._tts_inflight = True

        # on envoie hors lock
        goal = TTS.Goal()
        goal.text = text
        goal.language = lang
        goal.volume = float(volume)
        goal.rate = float(rate)

        self._tts_speaking = True

        future = self.tts_client.send_goal_async(goal)

        def _goal_response_cb(fut):
            try:
                goal_handle = fut.result()
            except Exception:
                # échec -> libère et tente le suivant
                self._tts_speaking = False
                self._tts_speaking_until = time.monotonic() + TTS_TAIL_S
                with self._tts_q_lock:
                    self._tts_inflight = False
                self._tts_kick()
                return

            if not goal_handle.accepted:
                # rejet -> libère et tente le suivant
                self._tts_speaking = False
                self._tts_speaking_until = time.monotonic() + TTS_TAIL_S
                with self._tts_q_lock:
                    self._tts_inflight = False
                self._tts_kick()
                return

            result_future = goal_handle.get_result_async()

            def _result_cb(_):
                # fin -> libère + tail anti-echo + suivant
                self._tts_speaking = False
                self._tts_speaking_until = time.monotonic() + TTS_TAIL_S
                with self._tts_q_lock:
                    self._tts_inflight = False
                self._tts_kick()

            result_future.add_done_callback(_result_cb)

        future.add_done_callback(_goal_response_cb)

    # ---------------- ALSA capture ----------------
    def _start_arecord(self):
        cmd = [
            "arecord",
            "-D", ALSA_DEVICE,
            "-f", "S16_LE",
            "-c", str(CHANNELS),
            "-r", str(SAMPLE_RATE),
            "-t", "raw",
            "-q",
        ]
        log(f"[DIALOG] Starting capture: {' '.join(cmd)}")
        self._arecord_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _stop_arecord(self):
        p = self._arecord_proc
        self._arecord_proc = None
        if p is None:
            return
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=1.0)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def _vad_is_speech(self, pcm: bytes) -> bool:
        if len(pcm) != CHUNK_BYTES:
            return False
        try:
            return self.vad.is_speech(pcm, SAMPLE_RATE)
        except Exception:
            return False

    def _clear_vad_buffers(self):
        self._vad_in_speech = False
        self._vad_speech_streak = 0
        self._vad_last_voice_t = 0.0
        self._pre_roll.clear()
        self._post_roll_left = 0

    def _audio_loop(self):
        """
        Capture ALSA -> VAD -> Vosk
        Pré-roll : on bufferise N frames avant speech start.
        Post-roll : après speech end, on garde N frames supplémentaires.
        """
        try:
            self._start_arecord()
            p = self._arecord_proc
            if p is None or p.stdout is None:
                log("[DIALOG] ❌ arecord failed to start (no stdout).")
                return

            silence_frame = b"\x00" * CHUNK_BYTES

            while not self._audio_stop_evt.is_set():
                chunk = p.stdout.read(CHUNK_BYTES)
                if not chunk:
                    time.sleep(0.01)
                    continue

                # session gate
                with self._lock:
                    if not self._session_running:
                        self._clear_vad_buffers()
                        continue

                # anti-echo
                now = time.monotonic()
                if self._tts_speaking or now < self._tts_speaking_until:
                    self._clear_vad_buffers()
                    continue

                speech = self._vad_is_speech(chunk)

                # ==========================
                # 1) Hors speech : pré-roll
                # ==========================
                if not self._vad_in_speech:
                    self._pre_roll.append(chunk)

                    if speech:
                        self._vad_last_voice_t = now
                        self._vad_speech_streak += 1
                    else:
                        self._vad_speech_streak = 0

                    # speech start confirmé
                    if self._vad_speech_streak >= VAD_START_FRAMES:
                        self._vad_in_speech = True
                        self._post_roll_left = 0
                        log("[DIALOG] VAD: speech start -> send pre-roll")

                        frames = list(self._pre_roll)
                        self._pre_roll.clear()

                        # envoie pré-roll (contient début du mot)
                        for fr in frames:
                            self._process_pcm(fr)

                    continue

                # =================================
                # 2) Pendant speech (ou post-roll)
                # =================================
                if speech:
                    # voix détectée -> normal
                    self._vad_last_voice_t = now
                    self._post_roll_left = 0  # si on était en post-roll, on l'annule
                    self._process_pcm(chunk)
                    continue

                # speech=False ici
                # Hangover : on continue un peu (conserve les phonèmes faibles)
                if now - self._vad_last_voice_t <= VAD_HANGOVER_S:
                    self._process_pcm(chunk)
                    continue

                # Début de post-roll si pas déjà en cours
                if self._post_roll_left == 0:
                    self._post_roll_left = POST_ROLL_FRAMES
                    log(f"[DIALOG] VAD: speech end -> post-roll {POST_ROLL_FRAMES} frames")

                # Pendant post-roll : on envoie encore l'audio réel
                if self._post_roll_left > 0:
                    self._process_pcm(chunk)
                    self._post_roll_left -= 1

                    # si post-roll terminé -> flush silence + sortie speech
                    if self._post_roll_left == 0:
                        log("[DIALOG] VAD: post-roll done -> flush silence & stop speech")
                        for _ in range(VAD_FLUSH_FRAMES):
                            self._process_pcm(silence_frame)

                        self._vad_in_speech = False
                        self._vad_speech_streak = 0
                        self._pre_roll.clear()

                continue

        finally:
            self._stop_arecord()
            log("[DIALOG] Audio thread stopped")

    # ---------------- Vosk processing (NO DEADLOCK) ----------------
    def _process_pcm(self, pcm: bytes):
        """
        Lock uniquement autour de Vosk (AcceptWaveform/Result),
        puis analyse HORS lock.
        """
        step = None
        res_raw = None

        with self._lock:
            rec = self.recognizer
            step = self._step
            if rec is None:
                return

            is_final = rec.AcceptWaveform(pcm)

            if is_final:
                try:
                    res_raw = rec.Result()
                except Exception:
                    return
            else:
                if not ENABLE_PARTIAL:
                    return
                return

        # ---- HORS lock ----
        try:
            res = json.loads(res_raw)
        except Exception:
            return

        text = (res.get("text") or "").strip().lower()
        if not text:
            return

        words = res.get("result", [])
        if not words and text:
            words = [{"word": text, "conf": 0.0}]

        allowed = set(self._allowed_keywords_for_step(step))
        min_conf = self._conf_thresh_for_step(step)

        # 1 seul mot : meilleur conf parmi allowed
        best_word = None
        best_conf = -1.0

        for w in words:
            word = (w.get("word") or "").strip().lower()
            conf = float(w.get("conf", 0.0))

            if word not in allowed:
                continue

            if conf >= min_conf and conf > best_conf:
                best_conf = conf
                best_word = word

        # (3) fallback exact match (utile si Vosk donne text="yes" mais conf faible/absente)
        if best_word is None and ACCEPT_EXACT_TEXT_FALLBACK:
            if text in allowed and len(text.split()) == 1:
                best_word = text
                best_conf = max(0.0, best_conf)  # conf non fiable, mais on accepte
        if best_word is None:
            return

        self._emit_word_final(best_word, best_conf)

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

            self._done_cooldown_until = 0.0

            # reset pause state
            self._timers_paused = False
            self._pause_t0 = 0.0

            # reset TTS readiness (mais avec recheck auto)
            self._tts_ready = False
            self._tts_last_check_t = 0.0

        # reset queue TTS à l'entrée
        with self._tts_q_lock:
            self._tts_q.clear()
            self._tts_inflight = False

        self._recreate_recognizer_for_step(self.STEP_ASK_OK)
        self._reset_recognizer()
        self._clear_vad_buffers()

        if self._audio_thread is None or not self._audio_thread.is_alive():
            self._audio_stop_evt.clear()
            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._audio_thread.start()

        log("[DIALOG] Session started")
        self._ask_current_step()

    def _stop_session(self):
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
        with self._lock:
            step = self._step

        if step == self.STEP_ASK_OK:
            self.say("Are you okay?")
        elif step == self.STEP_ASK_DAY:
            self.say("What day is it today?")
        elif step == self.STEP_ASK_PLACE:
            self.say("Where are you? Home or hospital.")

    def _advance_step(self):
        with self._lock:
            self._retries = 0
            self._step_started_t = time.monotonic()
            step = self._step

        if step == self.STEP_ASK_OK:
            with self._lock:
                self._step = self.STEP_ASK_DAY
            self._recreate_recognizer_for_step(self.STEP_ASK_DAY)
            self._reset_recognizer()
            self.say("Thank you. What day is it today?")

        elif step == self.STEP_ASK_DAY:
            with self._lock:
                self._step = self.STEP_ASK_PLACE
            self._recreate_recognizer_for_step(self.STEP_ASK_PLACE)
            self._reset_recognizer()
            self.say("Thank you. Where are you? Home or hospital?")

        elif step == self.STEP_ASK_PLACE:
            with self._lock:
                self._step = self.STEP_DONE
            self.say("Thank you. You seem conscious and oriented.")
            self._stop_session()

    def _handle_invalid_or_timeout(self, what=""):
        """
        Ici on garde le comportement,
        mais même s'il y a plusieurs say(), la queue empêche tout overlap.
        """
        with self._lock:
            retries = self._retries

        if retries < MAX_RETRIES:
            with self._lock:
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
        # rising edge -> start
        if sf.dialog_active and not self._prev_dialog_active:
            self._start_session()

        # falling edge forced by FSM
        if (not sf.dialog_active) and self._prev_dialog_active:
            with self._lock:
                self._session_running = False
                self._step = self.STEP_DONE
                self._done_cooldown_until = 0.0
            log("[DIALOG] FSM disabled dialog_active -> stop session (silent)")

        self._prev_dialog_active = sf.dialog_active

        now = time.monotonic()

        # (2) permet de relancer un TTS queued dès que /say devient dispo
        self._tts_kick()

        with self._lock:
            step = self._step
            cooldown_until = self._done_cooldown_until
            session_running = self._session_running
            step_t = self._step_started_t
            sess_t = self._session_started_t

            speaking = self._tts_speaking
            speaking_until = self._tts_speaking_until

        # cooldown post-DONE
        if step == self.STEP_DONE and cooldown_until > 0.0:
            if now >= cooldown_until:
                log("[DIALOG] Cooldown done -> dialog_active=False (FSM can advance)")
                sf.dialog_active = False
                with self._lock:
                    self._done_cooldown_until = 0.0
            return

        if not session_running:
            return

        # ==========================================================
        # (1) Pause des timeouts pendant TTS + tail anti-echo
        # ==========================================================
        tts_block = speaking or (now < speaking_until)

        if tts_block and not self._timers_paused:
            # on démarre la pause
            with self._lock:
                self._timers_paused = True
                self._pause_t0 = now
            return

        if (not tts_block) and self._timers_paused:
            # on termine la pause et on décale les timestamps (vrai "pause")
            with self._lock:
                paused_dur = now - self._pause_t0
                self._step_started_t += paused_dur
                self._session_started_t += paused_dur

                # optionnel: pause aussi la fenêtre de confirmation "no"
                if self._no_first_hit_t > 0.0:
                    self._no_first_hit_t += paused_dur

                self._timers_paused = False
                self._pause_t0 = 0.0

            # refresh variables après modif
            with self._lock:
                step_t = self._step_started_t
                sess_t = self._session_started_t

        # si on est toujours en block -> rien à faire
        if tts_block:
            return

        # ==========================================================
        # Timeouts normaux (hors TTS)
        # ==========================================================
        if now - sess_t > SESSION_TIMEOUT_S:
            self.call_ambulance("session timeout")
            return

        if step != self.STEP_DONE and (now - step_t > ANSWER_TIMEOUT_S):
            self._handle_invalid_or_timeout("Please answer now.")

    # ---------------- Recognition ----------------
    def _emit_word_final(self, word: str, conf: float):
        now_ms = time.monotonic() * 1000.0
        with self._lock:
            last = self._last_emit_ms.get(word, 0.0)
            if now_ms - last < DEBOUNCE_MS:
                return
            self._last_emit_ms[word] = now_ms

        self._on_word(word, conf, source="final")

    def _handle_no_final_confirm(self):
        now = time.monotonic()

        with self._lock:
            hits = self._no_final_hits
            first_t = self._no_first_hit_t

        if hits == 0:
            with self._lock:
                self._no_final_hits = 1
                self._no_first_hit_t = now
                self._step_started_t = time.monotonic()
            self.say("Could you repeat: are you okay?")
            return

        if now - first_t > NO_CONFIRM_WINDOW_S:
            with self._lock:
                self._no_final_hits = 1
                self._no_first_hit_t = now
                self._step_started_t = time.monotonic()
            self.say("Please repeat clearly!")
            return

        with self._lock:
            self._no_final_hits += 1
            hits = self._no_final_hits

        if hits >= NO_FINAL_CONFIRMATIONS_REQUIRED:
            self.call_ambulance("patient said NO (confirmed)")

    def _on_word(self, word: str, conf: float, source=""):
        with self._lock:
            if not self._session_running:
                return
            step = self._step

        log(f"[DIALOG] DETECTED ({source}): {word.upper()} (conf={conf:.2f}) step={step}")

        if word == "no":
            if source != "final":
                return
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
                today = DAYS[datetime.now().weekday()]
                if word == today:
                    self._advance_step()
                else:
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

    def destroy_node(self):
        self._audio_stop_evt.set()
        self._stop_arecord()

        # stop queue
        with self._tts_q_lock:
            self._tts_q.clear()
            self._tts_inflight = False

        super().destroy_node()


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
