# Check if the libraries are compatible with jetson environment before running the code.
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path  # <--- para construir caminhos relativos

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition_node')

        # Choose device (CPU or GPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Device: {self.device}')

        # Diretório deste arquivo (face_recognition_node.py)
        current_dir = Path(__file__).resolve().parent

        # Parâmetros ROS2 para permitir override, mas por padrão usa arquivos na mesma pasta do node
        self.declare_parameter(
            'embeddings_path',
            str(current_dir / 'embeddings.npy')
        )
        self.declare_parameter(
            'names_path',
            str(current_dir / 'names.npy')
        )

        EMB_PATH = self.get_parameter('embeddings_path').get_parameter_value().string_value
        NAMES_PATH = self.get_parameter('names_path').get_parameter_value().string_value

        self.get_logger().info(f'Loading embeddings from: {EMB_PATH}')
        self.get_logger().info(f'Loading names from: {NAMES_PATH}')

        # Load embeddings and names database
        try:
            self.embeddings_db = np.load(EMB_PATH)
            self.names_db = np.load(NAMES_PATH)
            self.get_logger().info(f'Database loaded: {len(self.names_db)} entries.')
        except Exception as e:
            self.get_logger().error(f'Error loading embeddings/names: {e}')
            self.embeddings_db = None
            self.names_db = None

        # Face models
        self.mtcnn = MTCNN(image_size=160, margin=20, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Publisher (who was recognized)
        self.publisher_ = self.create_publisher(String, 'face_recognition', 10)

        # Subscriber: receive camera images from ROS2 (like your QRCode node)
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',          # <--- mesma topic do seu QRCodeNode
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('FaceRecognition Node started! Subscribed to /image_raw/compressed')

    def recognize(self, frame_rgb):
        """Receives a RGB frame (numpy), returns (name, distance)."""
        if self.embeddings_db is None or self.names_db is None:
            return None, None

        img = Image.fromarray(frame_rgb)

        face = self.mtcnn(img)
        if face is None:
            return None, None

        with torch.no_grad():
            emb = self.resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]

        dists = np.linalg.norm(self.embeddings_db - emb, axis=1)
        idx = np.argmin(dists)
        min_dist = dists[idx]

        if min_dist < 0.9:   # threshold ajustável
            name = str(self.names_db[idx])
        else:
            name = "UNKNOWN"

        return name, float(min_dist)

    def image_callback(self, msg: CompressedImage):
        """Called every time a compressed image is received from the topic."""
        # 1) ROS2 CompressedImage -> OpenCV BGR frame
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().warn(f'Failed to decode image: {e}')
            return

        if frame is None:
            return

        # 2) BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3) Face recognition
        name, dist = self.recognize(frame_rgb)

        if name is not None:
            # Publish result
            msg_out = String()
            msg_out.data = f"{name};{dist:.3f}"
            self.publisher_.publish(msg_out)
            self.get_logger().info(f'Face: {name} ({dist:.2f})')

            # Draw on frame
            cv2.putText(frame, f"{name} ({dist:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 4) Show image (ok se a Jetson tiver interface gráfica)
        cv2.imshow('FaceNet Recognition', frame)
        cv2.waitKey(1)


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


if __name__ == '__main__':
    main()

