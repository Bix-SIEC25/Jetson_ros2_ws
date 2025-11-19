import cv2 as cv
import numpy as np
from pyzbar import pyzbar as bar

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from interfaces.msg import LogEntry

class QRCodeNode(Node):
    def __init__(self):
        super().__init__('qrcode_node')
        qos = QoSProfile(depth=10)

        # Publisher pour envoyer le texte du QR code
        self.logger_publisher_ = self.create_publisher(LogEntry, '/logger')
        self.publisher_ = self.create_publisher(String, 'qrcode_data', 10)

        self.last_logged = None

        # Subscriber pour recevoir les images compressées ROS2
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',     # <--- ton topic ROS2
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('QRCode Node started! Subscribed to /image_raw/compressed')

        # 🔹 confiance basée sur la stabilité
        self.last_output = None
        self.stable_count = 0
        self.MAX_STABLE = 5   # nb de frames pour atteindre 100 % de stabilité

        # 🔹 dernier identifiant détecté (Resident ID)
        self.current_resident_id = None

        # Paramètres pour l'affichage limité
        self.frame_count = 0  # Compteur pour limiter l'affichage

    def image_callback(self, msg: CompressedImage):
        # --- 1) convertir le CompressedImage ROS2 -> image OpenCV (BGR) ---
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().warn(f'Failed to decode image: {e}')
            return

        if frame is None:
            return

        h, w = frame.shape[:2]
        frame_area = float(w * h)

        # Limiter la détection à toutes les 10 frames (amélioration de la performance)
        self.frame_count += 1
        if self.frame_count % 10 != 0:  # Ne traiter que 1 frame sur 10
            return

        result = bar.decode(frame)
        output = None
        confidence = 0.0

        for data in result:
            # --- 1) Texte du QR code ---
            output = data.data.decode('utf-8')

            # === A) STABILITÉ DANS LE TEMPS ===
            if output == self.last_output:
                self.stable_count += 1
            else:
                self.last_output = output
                self.stable_count = 1
            stability_score = min(1.0, self.stable_count / self.MAX_STABLE)

            # === B) TAILLE DU QR DANS L’IMAGE ===
            x, y, bw, bh = data.rect
            box_area = float(bw * bh)
            area_ratio = box_area / frame_area

            AREA_MIN = 0.003
            AREA_MAX = 0.03
            area_score = (area_ratio - AREA_MIN) / (AREA_MAX - AREA_MIN)
            area_score = max(0.0, min(1.0, area_score))

            # === C) NETTETÉ (FOCUS) ===
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                var_lap = cv.Laplacian(gray, cv.CV_64F).var()

                SHARP_MIN = 10.0
                SHARP_MAX = 120.0
                sharp_score = (var_lap - SHARP_MIN) / (SHARP_MAX - SHARP_MIN)
                sharp_score = max(0.0, min(1.0, sharp_score))
            else:
                sharp_score = 0.0

            # === D) COMBINAISON DES SCORES ===
            confidence_raw = (
                0.6 * stability_score +
                0.25 * area_score +
                0.15 * sharp_score
            )

            confidence = max(0.0, min(1.0, 0.1 + 0.9 * confidence_raw))

            self.current_resident_id = output

            CONF_THRESHOLD = 0.5
            if confidence >= CONF_THRESHOLD:
                msg_out = String()
                msg_out.data = output
                self.publisher_.publish(msg_out)

            if output != self.last_logged:
                self.last_logged = output
                self.get_logger().info(
                    f'current_resident_id={self.current_resident_id}'
                )

                # Envoi de l'ID détecté dans le LogEntry
                msg = LogEntry()
                msg.level = LogEntry.TRACE
                msg.sender = "QRNode"
                msg.message = f"{self.current_resident_id}"
                self.logger_publisher_.publish(msg)

        # Aucune opération d'affichage (cv.imshow et cv.putText ont été supprimées)

def main(args=None):
    rclpy.init(args=args)
    node = QRCodeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
