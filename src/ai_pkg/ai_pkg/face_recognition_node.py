import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from interfaces.msg import LogEntry  # <-- ajout

from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition_node')

        # Choose device (CPU or GPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Device: {self.device}')

        # Models directory inside ROS2 share/
        share_dir = Path(get_package_share_directory('ai_pkg'))
        models_dir = share_dir / 'models'

        # Parameters for embeddings and names
        self.declare_parameter('embeddings_path', str(models_dir / 'embeddings.npy'))
        self.declare_parameter('names_path', str(models_dir / 'names.npy'))

        EMB_PATH = self.get_parameter('embeddings_path').get_parameter_value().string_value
        NAMES_PATH = self.get_parameter('names_path').get_parameter_value().string_value

        self.get_logger().info(f'Loading embeddings from: {EMB_PATH}')
        self.get_logger().info(f'Loading names from: {NAMES_PATH}')

        # Load embeddings and names database
        try:
            self.embeddings_db = np.load(EMB_PATH)
            self.names_db = np.load(NAMES_PATH)
            self.get_logger().info(f'Database loaded: {len(self.names_db)} identities')
        except Exception as e:
            self.get_logger().error(f'Error loading embeddings/names: {e}')
            self.embeddings_db = None
            self.names_db = None

        # Face detection model (MTCNN)
        self.mtcnn = MTCNN(image_size=160, margin=20, device=self.device)

        # Face embedding model (FaceNet)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Publisher: recognized face (String)
        self.publisher_ = self.create_publisher(String, 'qrcode_data', 10)

        # Publisher: logger (LogEntry), comme dans QRCodeNode
        self.logger_publisher_ = self.create_publisher(LogEntry, '/logger', 10)

        # Pour éviter de logger mille fois la même personne
        self.last_logged_name = None

        # Subscriber: receive camera compressed images
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("FaceRecognition Node started! Subscribed to /image_raw/compressed")

    # -------------------------------------------------------------------
    # RECOGNITION FUNCTION
    # -------------------------------------------------------------------
    def recognize(self, frame_rgb):
        if self.embeddings_db is None:
            return None, None

        img = Image.fromarray(frame_rgb)

        # Detect face
        face = self.mtcnn(img)
        if face is None:
            return None, None

        # Ensure face tensor is moved to GPU if model is on GPU
        face = face.to(self.device)

        # Compute embedding
        with torch.no_grad():
            emb = self.resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]

        # Compare with database
        dists = np.linalg.norm(self.embeddings_db - emb, axis=1)
        idx = np.argmin(dists)
        min_dist = dists[idx]

        name = str(self.names_db[idx]) if min_dist < 0.9 else "UNKNOWN"

        return name, float(min_dist)

    # -------------------------------------------------------------------
    # CALLBACK
    # -------------------------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        # 1. decode compressed image into cv2 frame
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode image: {e}")
            return

        if frame is None:
            return

        # 2. Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. Face recognition
        name, dist = self.recognize(frame_rgb)

        # 4. Publish result
        if name is not None:
            # Publication "classique" sur le topic face_recognition
            msg_out = String()
            msg_out.data = name
            self.publisher_.publish(msg_out)
            self.get_logger().info(f"Face: {name} ({dist:.2f}), old_face {self.last_logged_name}")

            # Publication dans le logger, comme pour le QR code
            if name != self.last_logged_name and name != "UNKNOWN":
                self.last_logged_name = name

                log_msg = LogEntry()
                log_msg.level = LogEntry.TRACE
                log_msg.sender = "FaceRecognitionNode"
                log_msg.message = name  # la personne reconnue

                self.logger_publisher_.publish(log_msg)
                self.get_logger().info(f"LOG ENTRY envoyé pour {name}")

            # Draw text on image
            cv2.putText(
                frame,
                f"{name} ({dist:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # 5. Show the video (useful on Jetson with a screen)
        try:
            cv2.imshow("FaceNet Recognition", frame)
            cv2.waitKey(1)
        except Exception:
            # headless mode
            pass


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
