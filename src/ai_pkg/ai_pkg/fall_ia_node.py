#!/usr/bin/env python3
import time
from io import BytesIO
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


class FallIANode(Node):
    """
    Détection de chute "IA" (headless) pilotée par la FSM via sf.fall_ia_active.

    - Ne traite des images QUE quand sf.fall_ia_active == True
    - Confirme une chute seulement après N frames consécutives -> évite les faux positifs "flash"
    - Dès chute confirmée:
        sf.return_val = sf.RET_FALL_AI
        sf.fall_ia_active = False
    - Si la FSM coupe sf.fall_ia_active = False, le node s'arrête immédiatement.
    """

    def __init__(self):
        super().__init__("fall_ia_node")

        # ===== Device =====
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log(f"[FALL_IA] Device: {self.device}")

        # ===== Model =====
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights).to(self.device)
        self.model.eval()

        # Pré-traitement recommandé par les weights (meilleure précision que ToTensor brut)
        self.preprocess = self.weights.transforms()

        # COCO "person" class id
        self.PERSON_CLASS_ID = 1

        # ===== Heuristiques / filtres =====
        self.conf_thresh = 0.75          # augmente => moins de faux positifs
        self.fall_ratio_thresh = 0.80    # h/w < 0.80 => "allongé"
        self.max_area_ratio = 0.90       # ignore si box couvre trop l'image (souvent faux)
        self.min_box_area = 0.02         # ignore si box trop petite (2% image) -> bruit

        # ===== Anti-faux-positifs temporels =====
        self.MIN_CONFIRM_FRAMES = 4      # chute confirmée si 4 inférences consécutives "fall"
        self.DETECT_EVERY_N_FRAMES = 3   # on ne fait l'inférence qu'1 image sur N reçues
        self._fall_streak = 0
        self._frame_seen = 0

        # ===== Flux images =====
        self._lock = threading.Lock()
        self._latest_msg = None
        self._latest_seq = 0
        self._processed_seq = 0
        self._processing = False

        self.sub = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.image_callback,
            qos_profile_sensor_data
        )

        # Timer "contrôle" (et lance l'inférence)
        self.timer = self.create_timer(0.05, self.loop)

        log("[FALL_IA] Ready")

    def image_callback(self, msg: CompressedImage):
        # Stocke la dernière image reçue (on traite dans loop)
        with self._lock:
            self._latest_msg = msg
            self._latest_seq += 1

    def _reset_scan(self):
        self._fall_streak = 0
        self._frame_seen = 0
        self._processing = False
        self._processed_seq = self._latest_seq

    def _decode_to_pil_rgb(self, msg: CompressedImage):
        # CompressedImage.data contient généralement du JPEG/PNG
        bio = BytesIO(bytes(msg.data))
        from PIL import Image  # import local pour éviter coût au démarrage si besoin
        img = Image.open(bio).convert("RGB")
        return img

    def _infer_fall(self, pil_img) -> bool:
        """
        Retourne True si on détecte au moins une personne "allongée" selon l'heuristique.
        """
        img_tensor = self.preprocess(pil_img).to(self.device)

        with torch.no_grad():
            out = self.model([img_tensor])[0]

        boxes = out["boxes"].detach().cpu()
        labels = out["labels"].detach().cpu()
        scores = out["scores"].detach().cpu()

        # Taille de l'image traitée par le modèle (après preprocess)
        _, H, W = img_tensor.shape
        img_area = float(W * H)

        fall_detected = False

        for box, label, score in zip(boxes, labels, scores):
            if float(score) < self.conf_thresh:
                continue
            if int(label) != self.PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 1.0 or h <= 1.0:
                continue

            box_area = w * h
            if box_area > self.max_area_ratio * img_area:
                continue
            if box_area < self.min_box_area * img_area:
                continue

            aspect = h / (w + 1e-6)  # h/w
            if aspect < self.fall_ratio_thresh:
                fall_detected = True
                break

        return fall_detected

    def loop(self):
        # Si la FSM coupe, on reset et on sort
        if not sf.fall_ia_active:
            if self._fall_streak != 0 or self._processing:
                self._reset_scan()
            return

        # Evite de lancer plusieurs inférences en parallèle
        if self._processing:
            return

        # Récupère une nouvelle image si dispo
        with self._lock:
            if self._latest_msg is None:
                return
            if self._latest_seq == self._processed_seq:
                return
            msg = self._latest_msg
            seq = self._latest_seq

        # Throttle: une image sur N
        self._frame_seen += 1
        if (self._frame_seen % self.DETECT_EVERY_N_FRAMES) != 0:
            self._processed_seq = seq
            return

        self._processing = True
        try:
            # Re-check : si la FSM coupe pendant qu'on s'apprête à traiter
            if not sf.fall_ia_active:
                self._reset_scan()
                return

            pil_img = self._decode_to_pil_rgb(msg)
            is_fall = self._infer_fall(pil_img)

            if not sf.fall_ia_active:
                # La FSM a peut-être coupé pendant l'inférence
                self._reset_scan()
                return

            if is_fall:
                self._fall_streak += 1
                log(f"[FALL_IA] fall detected (streak={self._fall_streak}/{self.MIN_CONFIRM_FRAMES})")
            else:
                if self._fall_streak != 0:
                    log("[FALL_IA] fall streak reset")
                self._fall_streak = 0

            # Confirmation
            if self._fall_streak >= self.MIN_CONFIRM_FRAMES:
                log("[FALL_IA] FALL CONFIRMED -> stop self, set return_val")
                sf.return_val = sf.RET_FALL_AI
                sf.fall_ia_active = False
                self._reset_scan()

        except Exception as e:
            log(f"[FALL_IA] ERROR: {e}")
            # En cas d'erreur, on reset le streak mais on laisse actif (la FSM décidera)
            self._fall_streak = 0

        finally:
            self._processed_seq = seq
            self._processing = False


def main(args=None):
    rclpy.init(args=args)
    node = FallIANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
