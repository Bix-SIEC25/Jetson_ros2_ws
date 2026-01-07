#!/usr/bin/env python3
import threading

import cv2 as cv
import numpy as np
from pyzbar import pyzbar as bar

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from interfaces.msg import LogEntry

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log
# from ai_pkg.utils.speaker import say


class QRNode(Node):
    """
    QR scanner headless piloté par sf.qr_active.
    - Traite /image_raw/compressed seulement quand sf.qr_active == True
    - Anti faux positifs:
        * stabilité N frames consécutives sur le même contenu
        * taille du QR dans l'image
        * netteté (variance Laplacien) sur la ROI
        * confirmation sur K frames où confidence >= threshold
    - Dès QR confirmé:
        * publish String sur 'qrcode_data'
        * publish LogEntry sur '/logger'
        * sf.qr_active = False (la FSM coupe l'autre parallèle)
    """

    def __init__(self):
        super().__init__("qr_node")

        # Publishers
        self.publisher_ = self.create_publisher(String, "qrcode_data", 10)
        self.logger_publisher_ = self.create_publisher(LogEntry, "/logger", 10)

        # Subscriber
        self.subscription = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.image_callback,
            qos_profile_sensor_data,
        )

        # --- Anti faux positifs / paramètres ---
        self.DETECT_EVERY_N_FRAMES = 2     # 1 image sur N (CPU)
        self.MAX_STABLE = 6               # stabilité "100%" à 6 hits consécutifs
        self.CONF_THRESHOLD = 0.65        # seuil global confiance
        self.MIN_CONFIRM_HITS = 3         # nombre de frames (consécutives) au-dessus du seuil

        # Aire QR (dans l'image) -> score
        self.AREA_MIN = 0.003
        self.AREA_MAX = 0.030

        # Netteté Laplacien -> score
        self.SHARP_MIN = 10.0
        self.SHARP_MAX = 120.0

        # --- Etat interne ---
        self._lock = threading.Lock()
        self._latest_msg = None
        self._latest_seq = 0
        self._processed_seq = 0

        self._frame_seen = 0

        self.last_output = None
        self.stable_count = 0
        self.confirm_hits = 0

        self.current_resident_id = None
        self.last_logged = None

        # Timer de travail
        self.timer = self.create_timer(0.05, self.loop)

        log("[QR] Node ready (headless). Subscribed to /image_raw/compressed")

    def image_callback(self, msg: CompressedImage):
        # On stocke juste la dernière image (pas de boulot ici)
        with self._lock:
            self._latest_msg = msg
            self._latest_seq += 1

    def _reset_scan(self):
        self.last_output = None
        self.stable_count = 0
        self.confirm_hits = 0
        self.current_resident_id = None

    def _decode_frame(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
            return frame
        except Exception as e:
            log(f"[QR] Failed to decode image: {e}")
            return None

    def _score_candidate(self, frame, qr_item):
        """
        Calcule (output, confidence) pour un QR code candidat.
        """
        h, w = frame.shape[:2]
        frame_area = float(w * h)

        output = qr_item.data.decode("utf-8", errors="ignore")

        # --- A) Stabilité temporelle (sera combinée ensuite) ---
        # (On ne met pas à jour ici, c'est fait après sélection du meilleur candidat)

        # --- B) Taille du QR dans l'image ---
        x, y, bw, bh = qr_item.rect
        box_area = float(bw * bh)
        area_ratio = box_area / frame_area if frame_area > 1 else 0.0

        area_score = (area_ratio - self.AREA_MIN) / (self.AREA_MAX - self.AREA_MIN)
        area_score = max(0.0, min(1.0, area_score))

        # --- C) Netteté ---
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            var_lap = cv.Laplacian(gray, cv.CV_64F).var()

            sharp_score = (var_lap - self.SHARP_MIN) / (self.SHARP_MAX - self.SHARP_MIN)
            sharp_score = max(0.0, min(1.0, sharp_score))
        else:
            sharp_score = 0.0

        # score partiel (sans stabilité)
        base = 0.65 * area_score + 0.35 * sharp_score
        base = max(0.0, min(1.0, base))

        return output, base

    def loop(self):
        # Si FSM coupe qr_active, on reset et on quitte
        if not sf.qr_active:
            self._reset_scan()
            return

        # Récupérer une nouvelle image
        with self._lock:
            if self._latest_msg is None:
                return
            if self._latest_seq == self._processed_seq:
                return
            msg = self._latest_msg
            seq = self._latest_seq

        self._processed_seq = seq

        # Throttle frames
        self._frame_seen += 1
        if (self._frame_seen % self.DETECT_EVERY_N_FRAMES) != 0:
            return

        frame = self._decode_frame(msg)
        if frame is None:
            return

        # Détection QR
        results = bar.decode(frame)
        if not results:
            # Si on ne voit rien, on “relâche” la stabilité
            if self.stable_count > 0:
                self.stable_count = max(0, self.stable_count - 1)
            self.confirm_hits = 0
            return

        # Choisir le meilleur QR candidat (par score base)
        best_output = None
        best_base = -1.0
        for item in results:
            out, base = self._score_candidate(frame, item)
            if out and base > best_base:
                best_output = out
                best_base = base

        if best_output is None:
            return

        # --- A) Stabilité (mise à jour) ---
        if best_output == self.last_output:
            self.stable_count += 1
        else:
            self.last_output = best_output
            self.stable_count = 1
            self.confirm_hits = 0  # reset confirmation si changement

        stability_score = min(1.0, self.stable_count / float(self.MAX_STABLE))

        # --- D) Combinaison finale ---
        # On mixe stabilité + qualité image (taille/netteté)
        confidence_raw = 0.60 * stability_score + 0.40 * best_base
        confidence = max(0.0, min(1.0, 0.05 + 0.95 * confidence_raw))

        self.current_resident_id = best_output

        # Confirmation multi-frames au-dessus du seuil
        if confidence >= self.CONF_THRESHOLD:
            self.confirm_hits += 1
        else:
            self.confirm_hits = 0

        log(
            f"[QR] candidate='{best_output}' "
            f"stable={self.stable_count}/{self.MAX_STABLE} "
            f"confirm_hits={self.confirm_hits}/{self.MIN_CONFIRM_HITS} "
            f"conf={confidence:.2f}"
        )

        # Si confirmé -> publish + log + stop self
        if self.confirm_hits >= self.MIN_CONFIRM_HITS:
            msg_out = String()
            msg_out.data = best_output
            self.publisher_.publish(msg_out)

            if best_output != self.last_logged:
                self.last_logged = best_output
                log_msg = LogEntry()
                log_msg.level = LogEntry.TRACE
                log_msg.sender = "QRNode"
                log_msg.message = best_output
                self.logger_publisher_.publish(log_msg)

            log("[QR] QR CONFIRMED -> qr_active=False")
            # say("QR code detected.")
            sf.qr_active = False
            self._reset_scan()


def main(args=None):
    rclpy.init(args=args)
    node = QRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
