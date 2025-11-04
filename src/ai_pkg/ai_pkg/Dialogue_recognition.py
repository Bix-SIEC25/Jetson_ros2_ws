import json
import time
import array
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.action import ActionClient

from vosk import Model, KaldiRecognizer
from ament_index_python.packages import get_package_share_directory

from audio_common_msgs.msg import AudioStamped
from audio_common_msgs.action import TTS


# ---- paramètres ----
SAMPLE_RATE = 16000
CONF_THRESH = 0.75
DEBOUNCE_MS = 500
KEYWORDS = ["yes", "no", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "trump", "obama"]
AUDIO_TOPIC = "/audio_mic"
# ---------------------


def get_model_path() -> str:
    """Retourne le chemin du modèle Vosk dans le package ai_pkg."""
    share_dir = get_package_share_directory("ai_pkg")
    return str(Path(share_dir) / "models" / "vosk-model-small-en-us-0.15")


class VoskKeywordNode(Node):
    def __init__(self):
        super().__init__("vocal_recognition")

        # ---- Chargement modèle Vosk ----
        model_path = get_model_path()
        self.get_logger().info(f"Chargement du modèle Vosk depuis : {model_path}")
        model = Model(model_path)

        grammar = json.dumps(KEYWORDS + ["[unk]"])
        self.recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)
        self.last_emit = {k: 0.0 for k in KEYWORDS}

        # ---- QoS audio_common (BEST_EFFORT) ----
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- Abonnement audio ----
        self.subscription = self.create_subscription(
            AudioStamped,
            AUDIO_TOPIC,
            self.audio_callback,
            qos,
        )

        # ---- Client TTS ----
        self.tts_client = ActionClient(self, TTS, '/say')

        self.get_logger().info(f"🎙️  Vosk prêt, écoute sur {AUDIO_TOPIC}")

        # ---- Variables de gestion du dialogue ----
        self.waiting_for_day = False
        self.waiting_for_president = False

    # ---------- FONCTION POUR FAIRE PARLER LE ROBOT ----------
    def say(self, text, lang="en", volume=1.0, rate=1.0):
        """Envoie du texte au TTS audio_common."""
        if not self.tts_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("⚠️ Serveur TTS /say non disponible")
            return

        goal = TTS.Goal()
        goal.text = text
        goal.language = lang
        goal.volume = volume
        goal.rate = rate

        self.get_logger().info(f"🗣️ TTS → {text}")
        self.tts_client.send_goal_async(goal)

    # ---------- CALLBACK AUDIO ----------
    def audio_callback(self, msg: AudioStamped):
        audio = msg.audio
        data = audio.audio_data

        # audio_capturer_node format=8 → int16
        if not data.int16_data:
            return

        pcm = array.array('h', data.int16_data).tobytes()
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

        words = [w for w in res.get("result", []) if w.get("word") in KEYWORDS]
        if not words and text in KEYWORDS:
            words = [{"word": text, "conf": 1.0}]

        now = time.time() * 1000.0

        # Traitement des mots détectés
        for w in words:
            word = w.get("word")
            conf = float(w.get("conf", 0.0))

            if conf >= CONF_THRESH and now - self.last_emit[word] >= DEBOUNCE_MS:
                self.last_emit[word] = now
                self.get_logger().info(f"[DETECTED] {word.upper()} (conf={conf:.2f})")

                # ----- ACTIONS ET TTS -----
                if word == "yes":
                    self.say("Okay, great! What day is it today?")
                    self.waiting_for_day = True
                elif self.waiting_for_day:
                    if word in KEYWORDS[4:10]:  # Vérification si le jour est valide
                        self.get_logger().info(f"Day detected: {word}")
                        self.waiting_for_day = False
                        self.say(f"Is today {word}? Please tell me who the president is.")
                        self.waiting_for_president = True
                    else:
                        self.say("That’s not a valid day. I will call an ambulance now.")
                        self.call_ambulance()

                elif self.waiting_for_president:
                    if word == "trump":
                        self.say("You said Trump! Everything is good.")
                        self.waiting_for_president = False
                    elif word == "obama":
                        self.say("You said Obama! I will call an ambulance now.")
                        self.call_ambulance()
                    else:
                        self.say("That’s not a valid president. I will call an ambulance now.")
                        self.call_ambulance()

    def call_ambulance(self):
        """Simuler l'appel à une ambulance."""
        self.get_logger().info("🚑 Calling an ambulance...")
        self.say("Emergency! Ambulance is on the way.")

# ---------- MAIN ----------
def main(args=None):
    rclpy.init(args=args)
    node = VoskKeywordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
