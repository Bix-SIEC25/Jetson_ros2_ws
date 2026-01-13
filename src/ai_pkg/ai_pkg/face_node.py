#!/usr/bin/env python3
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from interfaces.msg import LogEntry

from facenet_pytorch import MTCNN, InceptionResnetV1

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log
from ai_pkg.utils.speaker import say


class FaceNode(Node):
    """
    Node de reconnaissance faciale piloté par la FSM via sf.face_active.

    - Quand sf.face_active == False : ignore les images.
    - Quand sf.face_active passe à True : commence à traiter les images caméra.
    - Dès qu'un visage NON "UNKNOWN" est détecté :
        * publie le nom
        * log dans /logger
        * sf.face_active = False  -> signale à la FSM que l'étape FACE est terminée
    """

    def __init__(self):
        super().__init__("face_node")

        # --------- ÉTAT INTERNE ---------
        self._active = False           # POUR savoir si on est en phase de travail pour la FSM
        self.last_logged_name = None   # pour ne pas spammer le logger
        self.name = "UNKNOWN"          # nom courant détecté

        # --------- DEVICE (CPU / GPU) ---------
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        log(f'[FACE] Device: {self.device}')

        # --------- CHARGEMENT DES MODÈLES ---------
        share_dir = Path(get_package_share_directory('ai_pkg'))
        models_dir = share_dir / 'models'

        self.declare_parameter('embeddings_path', str(models_dir / 'embeddings.npy'))
        self.declare_parameter('names_path', str(models_dir / 'names.npy'))

        EMB_PATH = self.get_parameter('embeddings_path').get_parameter_value().string_value
        NAMES_PATH = self.get_parameter('names_path').get_parameter_value().string_value

        log(f'[FACE] Loading embeddings from: {EMB_PATH}')
        log(f'[FACE] Loading names from: {NAMES_PATH}')

        try:
            self.embeddings_db = np.load(EMB_PATH)
            self.names_db = np.load(NAMES_PATH)
            log(f'[FACE] Database loaded: {len(self.names_db)} identities')
        except Exception as e:
            log(f'[FACE] Error loading embeddings/names: {e}')
            self.embeddings_db = None
            self.names_db = None

        # Détection & embedding
        self.mtcnn = MTCNN(image_size=160, margin=20, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Publisher : nom reconnu (réutilise ton topic existant)
        self.publisher_ = self.create_publisher(String, 'qrcode_data', 10)

        # Logger centralisé (comme pour le QR)
        self.logger_publisher_ = self.create_publisher(LogEntry, '/logger', 10)

        # Subscriber : images caméra compressées
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data
        )

        # Timer pour surveiller les changements de sf.face_active (optionnel mais pratique pour les logs)
        self.state_timer = self.create_timer(0.2, self.state_monitor)

        log("[FACE] FaceNode started! Subscribed to /image_raw/compressed")

    # -------------------------------------------------------------------
    # MONITORING DU FLAG DE LA FSM
    # -------------------------------------------------------------------
    def state_monitor(self):
        """
        Surveille le booléen sf.face_active pour logguer les transitions
        et savoir quand on est "actif" ou pas.
        """
        if sf.face_active and not self._active:
            self._active = True
            log("[FACE] Activation par la FSM (sf.face_active = True)")
            say("Face recognition activated")
        elif not sf.face_active and self._active:
            self._active = False
            log("[FACE] Désactivation par la FSM (sf.face_active = False)")

    # -------------------------------------------------------------------
    # RECONNAISSANCE
    # -------------------------------------------------------------------
    def recognize(self, frame_rgb):
        if self.embeddings_db is None:
            return None, None

        img = Image.fromarray(frame_rgb)

        # Détection de visage
        face = self.mtcnn(img)
        if face is None:
            return None, None

        face = face.to(self.device)

        with torch.no_grad():
            emb = self.resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]

        # Comparaison avec la base
        dists = np.linalg.norm(self.embeddings_db - emb, axis=1)
        idx = np.argmin(dists)
        min_dist = dists[idx]

        name = str(self.names_db[idx]) if min_dist < 0.9 else "UNKNOWN"
        return name, float(min_dist)

    # -------------------------------------------------------------------
    # CALLBACK IMAGE
    # -------------------------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        # Si la FSM n'a pas activé l'étape FACE → on ignore les images
        if not sf.face_active:
            return

        # 1. décodage de l'image
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            log(f"[FACE] Failed to decode image: {e}")
            return

        if frame is None:
            return

        # 2. BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. Reconnaissance faciale
        name, dist = self.recognize(frame_rgb)

        if name is None:
            return

        # 4. Publication du résultat
        msg_out = String()
        msg_out.data = name
        self.publisher_.publish(msg_out)
        log(f"[FACE] Face: {name} ({dist:.2f})")

        # 5. Logger centralisé (comme QRCodeNode)
        if name != "UNKNOWN":
            self.last_logged_name = name

            log_msg = LogEntry()
            log_msg.level = LogEntry.TRACE
            log_msg.sender = "FaceRecognitionNode"
            log_msg.message = name

            self.logger_publisher_.publish(log_msg)
            log(f"[FACE] LOG ENTRY envoyé pour {name}")

        # 6. Si visage connu → signaler fin de travail à la FSM
        if name != "UNKNOWN":
            log("[FACE] Visage connu détecté, fin de l'étape FACE → sf.face_active = False")
            say(f"{name} has been detected")
            sf.face_active = False
            self.name = "UNKNOWN"
            # La FSM verra ce False et passera à l'état suivant


# -----------------------------------------------------------------------
# MAIN (utilisé seulement lancement de ce node tout seul)
# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = FaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
