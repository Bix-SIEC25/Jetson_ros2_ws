import json
import time
import array
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from vosk import Model, KaldiRecognizer
from ament_index_python.packages import get_package_share_directory
from audio_common_msgs.msg import AudioStamped

# ---- paramètres à adapter si besoin ----
SAMPLE_RATE = 16000
CONF_THRESH = 0.75        # 0.6 (tolérant) → 0.9 (strict)
DEBOUNCE_MS = 500         # anti-rebond en ms
KEYWORDS = ["yes", "no", "help", "please", "hi", "batiste"]
AUDIO_TOPIC = "/audio_mic"    # topic publié par audio_common
# ----------------------------------------


def get_model_path() -> str:
    """Retourne le chemin du modèle Vosk dans le package ai_pkg."""
    share_dir = get_package_share_directory("ai_pkg")
    model_dir = Path(share_dir) / "models" / "vosk-model-small-en-us-0.15"
    return str(model_dir)


class VoskKeywordNode(Node):
    def __init__(self):
        super().__init__("vocal_recognition")

        # Charger le modèle
        model_path = get_model_path()
        self.get_logger().info(f"Chargement du modèle Vosk depuis : {model_path}")
        model = Model(model_path)

        grammar = json.dumps(KEYWORDS + ["[unk]"])
        self.recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)

        self.last_emit = {k: 0.0 for k in KEYWORDS}

        # QoS alignée sur audio_capturer_node (BEST_EFFORT)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Abonnement au topic audio (audio_common)
        self.subscription = self.create_subscription(
            AudioStamped,
            AUDIO_TOPIC,
            self.audio_callback,
            qos,
        )

        self.get_logger().info(
            f"🎙️  Vosk prêt, écoute sur {AUDIO_TOPIC} (Ctrl+C pour arrêter)"
        )

    def audio_callback(self, msg: AudioStamped):
        # Récupération des données audio depuis AudioStamped
        audio = msg.audio          # Audio
        data = audio.audio_data    # AudioData

        # Avec format=8 (paInt16) → int16_data est utilisé
        if not data.int16_data:
            # Rien à traiter (ou format différent)
            return

        # Convertir la liste d'int16 en bytes (PCM 16-bit little-endian)
        arr = array.array('h', data.int16_data)
        data_bytes = arr.tobytes()

        if not data_bytes:
            return

        # Envoi du chunk à Vosk
        final = self.recognizer.AcceptWaveform(data_bytes)
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
        for w in words:
            word = w.get("word")
            conf = float(w.get("conf", 0.0))
            if conf >= CONF_THRESH and now - self.last_emit[word] >= DEBOUNCE_MS:
                self.last_emit[word] = now
                self.get_logger().info(f"[DETECTED] {word.upper()}  (conf={conf:.2f})")

                if word == "yes":
                    self.get_logger().info("Action : lui aussi")
                elif word == "help":
                    self.get_logger().info("Action : URGENCE")
                elif word == "hi":
                    self.get_logger().info("Action : bonjour")
                elif word == "no":
                    self.get_logger().info("🛑 Action : arrêter le robot")
                elif word == "batiste":
                    self.get_logger().info("🛑 Action : try")
                elif word == "please":
                    self.get_logger().info("Action : tu connais la politesse")


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
