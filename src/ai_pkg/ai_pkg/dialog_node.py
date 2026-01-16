#!/usr/bin/env python3
import json
import time
import array
import threading
import math
import subprocess
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from vosk import Model, KaldiRecognizer
from ament_index_python.packages import get_package_share_directory

from audio_common_msgs.action import TTS

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


# ----------------- PARAMS -----------------
SAMPLE_RATE = 16000               # DOIT matcher la capture micro (ici: 16k)
CHANNELS = 1                      # mono
SAMPLE_WIDTH_BYTES = 2            # int16 = 2 bytes

CONF_THRESH_FINAL = 0.70
DEBOUNCE_MS = 600

ANSWER_TIMEOUT_S = 10.0
SESSION_TIMEOUT_S = 50.0
MAX_RETRIES = 1

TTS_ACTION = "/say"

ENABLE_PARTIAL = False

TTS_TAIL_S = 1.0
MIN_RMS = 250.0

NO_FINAL_CONFIRMATIONS_REQUIRED = 2
NO_CONFIRM_WINDOW_S = 2.5

POST_DIALOG_COOLDOWN_S = 10.0

# Capture ALSA: on utilise arecord directement
ALSA_DEVICE = "hw:0,0"            # webcam C270
CHUNK_MS = 20                     # 10-30ms typique; 20ms = bon compromis
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_BYTES = CHUNK_SAMPLES * CHANNELS * SAMPLE_WIDTH_BYTES

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
    acc = 0.0
    for x in a:
        acc += float(x) * float(x)
    return math.sqrt(acc / float(len(a)))


class DialogNode(Node):
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

        # ---- TTS ----
        self.tts_client = ActionClient(self, TTS, TTS_ACTION)
        self._tts_ready = False
        self._tts_checked_once = False

        # TTS speaking gate (anti-echo)
        self._tts_speaking = False
        self._tts_speaking_until = 0.0

        # ---- ALSA capture thread ----
        self._audio_thread = None
        self._audio_stop_evt = threading.Event()
        self._arecord_proc = None

        # Timer: manage session start/stop + timeouts + cooldown DONE
        self.timer = self.create_timer(0.1, self.loop)

        log(f"[DIALOG] Ready. Capturing mic directly via ALSA ({ALSA_DEVICE}) @ {SAMPLE_RATE}Hz mono")

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
        allowed = self._allowed_keywords_for_step(step)
        grammar = json.dumps(allowed + ["[unk]"])
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE, grammar)
        try:
            self.recognizer.SetWords(True)
        except Exception:
            pass
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

    # ---------------- ALSA capture ----------------
    def _start_arecord(self):
        """
        Lance arecord en sortie RAW (S16_LE) sur stdout.
        """
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
            stderr=subprocess.PIPE,
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

    def _audio_loop(self):
        """
        Boucle de capture: lit des chunks PCM int16 mono 16k,
        applique noise gate + anti-echo, puis feed Vosk.
        """
        try:
            self._start_arecord()
            p = self._arecord_proc
            if p is None or p.stdout is None:
                log("[DIALOG] ❌ arecord failed to start (no stdout).")
                return

            while not self._audio_stop_evt.is_set():
                chunk = p.stdout.read(CHUNK_BYTES)
                if not chunk:
                    # arecord ended or no data
                    time.sleep(0.01)
                    continue

                # N'avance pas si session off
                with self._lock:
                    if not self._session_running:
                        continue

                # Anti-echo: ignore ASR while TTS is speaking (and short tail)
                now = time.monotonic()
                if self._tts_speaking or now < self._tts_speaking_until:
                    continue

                # Noise gate
                if pcm_rms_int16(chunk) < MIN_RMS:
                    continue

                self._process_pcm(chunk)

        finally:
            self._stop_arecord()
            log("[DIALOG] Audio thread stopped")

    def _process_pcm(self, pcm: bytes):
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

            words = res.get("result", [])
            if not words and text:
                words = [{"word": text, "conf": 0.0}]

            for w in words:
                word = (w.get("word") or "").strip().lower()
                conf = float(w.get("conf", 0.0))

                if conf < CONF_THRESH_FINAL:
                    continue

                allowed = set(self._allowed_keywords_for_step(self._step))
                if word in allowed:
                    self._emit_word_final(word, conf)

        else:
            if not ENABLE_PARTIAL:
                return
            try:
                pres = json.loads(self.recognizer.PartialResult())
            except Exception:
                return
            ptxt = (pres.get("partial") or "").strip().lower()
            if not ptxt:
                return

            allowed = set(self._allowed_keywords_for_step(self._step))
            tokens = ptxt.split()
            detected = None
            for t in tokens:
                if t in allowed:
                    detected = t
                    break
            if not detected:
                return
            if detected == "no":
                return
            return

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

            self._tts_ready = False
            self._tts_checked_once = False

            self._recreate_recognizer_for_step(self.STEP_ASK_OK)
            self._reset_recognizer()

            # start audio thread if not running
            if self._audio_thread is None or not self._audio_thread.is_alive():
                self._audio_stop_evt.clear()
                self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
                self._audio_thread.start()

        log("[DIALOG] Session started")
        self.say("Can you hear me?")

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

        # Cooldown post-DONE
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
            return

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
        last = self._last_emit_ms.get(word, 0.0)
        if now_ms - last < DEBOUNCE_MS:
            return
        self._last_emit_ms[word] = now_ms
        self._on_word(word, conf, source="final")

    def _handle_no_final_confirm(self):
        now = time.monotonic()
        if self._no_final_hits == 0:
            self._no_final_hits = 1
            self._no_first_hit_t = now
            self.say("Could you repeat: are you okay?")
            self._step_started_t = time.monotonic()
            return

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
        # stop audio thread / arecord
        self._audio_stop_evt.set()
        self._stop_arecord()
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
