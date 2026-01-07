#!/usr/bin/env python3
import json
import time
import array
import threading
from pathlib import Path

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


# ---- paramètres ----
SAMPLE_RATE = 16000
CONF_THRESH = 0.75 # 0.80/0.85 pour être plus strict
DEBOUNCE_MS = 500 # 800/1000 si il y a des répétitions

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PRESIDENTS = ["trump", "obama"]
KEYWORDS = ["yes", "no"] + DAYS + PRESIDENTS

AUDIO_TOPIC = "/audio_mic"
# ---------------------


def get_model_path() -> str:
    share_dir = get_package_share_directory("ai_pkg")
    return str(Path(share_dir) / "models" / "vosk-model-small-en-us-0.15")


class DialogNode(Node):
    """
    DialogNode piloté par sf.dialog_active.

    - Quand sf.dialog_active passe à True: démarre une "session" de dialogue
    - Écoute /audio_mic via Vosk et réagit aux mots clés
    - Quand terminé (OK ou urgence): sf.dialog_active = False
    """

    def __init__(self):
        super().__init__("dialog_node")

        # --- état interne ---
        self._lock = threading.Lock()
        self._session_running = False
        self._prev_dialog_active = False

        self.waiting_for_day = False
        self.waiting_for_president = False

        # --- Chargement Vosk ---
        model_path = get_model_path()
        log(f"[DIALOG] Chargement Vosk: {model_path}")
        model = Model(model_path)

        grammar = json.dumps(KEYWORDS + ["[unk]"])
        self.recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)
        self.last_emit = {k: 0.0 for k in KEYWORDS}

        # --- QoS audio (BEST_EFFORT) ---
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Abonnement audio ---
        self.subscription = self.create_subscription(
            AudioStamped,
            AUDIO_TOPIC,
            self.audio_callback,
            qos,
        )

        # --- Client TTS ---
        self.tts_client = ActionClient(self, TTS, "/say")

        # Timer léger pour détecter activation/désactivation FSM
        self.timer = self.create_timer(0.1, self.loop)

        log(f"[DIALOG] Prêt, écoute sur {AUDIO_TOPIC}")

    # ---------- TTS ----------
    def say(self, text, lang="en", volume=1.0, rate=1.0):
        """Envoie du texte au TTS audio_common (non bloquant)."""
        if not self.tts_client.wait_for_server(timeout_sec=0.2):
            log("[DIALOG] ⚠️ Serveur TTS /say non dispo")
            return

        goal = TTS.Goal()
        goal.text = text
        goal.language = lang
        goal.volume = float(volume)
        goal.rate = float(rate)

        log(f"[DIALOG] 🗣️ {text}")
        self.tts_client.send_goal_async(goal)

    # ---------- gestion session ----------
    def _start_session(self):
        with self._lock:
            self._session_running = True
            self.waiting_for_day = False
            self.waiting_for_president = False
            self.last_emit = {k: 0.0 for k in KEYWORDS}
            try:
                self.recognizer.Reset()
            except Exception:
                pass

        log("[DIALOG] Session démarrée")
        # Prompt initial (tu peux changer le texte)
        self.say("Hello. Are you okay? Please say yes or no.")

    def _stop_session(self):
        with self._lock:
            self._session_running = False
            self.waiting_for_day = False
            self.waiting_for_president = False
        log("[DIALOG] Session terminée -> dialog_active=False")
        sf.dialog_active = False

    def call_ambulance(self):
        log("[DIALOG] 🚑 Calling an ambulance...")
        self.say("Emergency! Ambulance is on the way.")
        self._stop_session()

    # ---------- boucle FSM gating ----------
    def loop(self):
        # front montant: False -> True => start session
        if sf.dialog_active and not self._prev_dialog_active:
            self._start_session()

        # si FSM coupe l'état DIALOG, on arrête proprement
        if (not sf.dialog_active) and self._prev_dialog_active:
            with self._lock:
                self._session_running = False
                self.waiting_for_day = False
                self.waiting_for_president = False
            log("[DIALOG] FSM a désactivé dialog_active -> stop session (silent)")

        self._prev_dialog_active = sf.dialog_active

    # ---------- callback audio ----------
    def audio_callback(self, msg: AudioStamped):
        # Ne rien faire si pas en état DIALOG
        if not sf.dialog_active:
            return

        with self._lock:
            if not self._session_running:
                return

        audio = msg.audio
        data = audio.audio_data

        # format int16 attendu
        if not data.int16_data:
            return

        pcm = array.array("h", data.int16_data).tobytes()
        if not pcm:
            return

        # Reconnaissance Vosk
        final = self.recognizer.AcceptWaveform(pcm)
        if not final:
            return

        res = json.loads(self.recognizer.Result())
        text = res.get("text", "").strip()
        if not text:
            return

        # mots pertinents
        words = [w for w in res.get("result", []) if w.get("word") in KEYWORDS]
        if not words and text in KEYWORDS:
            words = [{"word": text, "conf": 1.0}]

        now = time.time() * 1000.0

        for w in words:
            word = w.get("word")
            conf = float(w.get("conf", 0.0))
            if word not in KEYWORDS:
                continue

            # debounce + seuil
            if conf < CONF_THRESH:
                continue
            if now - self.last_emit[word] < DEBOUNCE_MS:
                continue
            self.last_emit[word] = now

            log(f"[DIALOG] DETECTED: {word.upper()} (conf={conf:.2f})")

            # ----- logique dialogue -----
            with self._lock:
                if not self._session_running:
                    return

                if word == "no":
                    # Si la personne dit non -> urgence
                    self.say("Okay. I will call an ambulance now.")
                    self.call_ambulance()
                    return

                if word == "yes" and (not self.waiting_for_day) and (not self.waiting_for_president):
                    self.say("Okay, great! What day is it today?")
                    self.waiting_for_day = True
                    return

                if self.waiting_for_day:
                    if word in DAYS:
                        self.waiting_for_day = False
                        self.say(f"Is today {word}? Please tell me who the president is.")
                        self.waiting_for_president = True
                    else:
                        self.say("That is not a valid day. I will call an ambulance now.")
                        self.call_ambulance()
                    return

                if self.waiting_for_president:
                    if word == "trump":
                        self.say("You said Trump. Everything is good.")
                        self.waiting_for_president = False
                        # Dialogue terminé -> rendre la main à la FSM
                        self._stop_session()
                        return
                    elif word == "obama":
                        self.say("You said Obama. I will call an ambulance now.")
                        self.call_ambulance()
                        return
                    else:
                        self.say("That is not a valid president. I will call an ambulance now.")
                        self.call_ambulance()
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
