
# import cv2
# import torch
# from torchvision import transforms
# import torchvision
# from torchvision.models.detection import ssdlite320_mobilenet_v3_large
# from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights

# # =========================
# # 1. Modèle PyTorch
# # =========================

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"[INFO] Using device: {device}")

# weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
# model = ssdlite320_mobilenet_v3_large(weights=weights).to(device)
# model.eval()


# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# PERSON_CLASS_ID = 1

# # =========================
# # 2. Vidéo / webcam
# # =========================
# cap = cv2.VideoCapture(0)

# conf_thresh = 0.7
# fall_ratio_thresh = 0.8  # si h/w < 0.8 → allongé
# max_area_ratio = 0.90    # si box > 90% image → on ignore

# # --------- NEW : détection 1 frame sur 3 ----------
# DETECT_EVERY_N = 3
# frame_idx = 0
# last_detections = []  # (x1, y1, x2, y2, color, text)
# # --------------------------------------------------

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (1020, 600))
#     H, W = frame.shape[:2]

#     frame_idx += 1
#     run_detection = (frame_idx % DETECT_EVERY_N == 0)

#     if run_detection:
#         # ======================
#         #   INFÉRENCE MODELE
#         # ======================
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img_tensor = transform(img_rgb).to(device)

#         with torch.no_grad():
#             outputs = model([img_tensor])

#         out = outputs[0]
#         boxes = out["boxes"].detach().cpu().numpy()
#         labels = out["labels"].detach().cpu().numpy()
#         scores = out["scores"].detach().cpu().numpy()

#         new_detections = []

#         for box, label, score in zip(boxes, labels, scores):
#             if score < conf_thresh:
#                 continue
#             if label != PERSON_CLASS_ID:
#                 continue

#             x1, y1, x2, y2 = box.astype(int)
#             w = x2 - x1
#             h = y2 - y1

#             if w <= 0 or h <= 0:
#                 continue

#             # filtre taille
#             box_area = w * h
#             img_area = W * H
#             if box_area > max_area_ratio * img_area:
#                 continue

#             aspect = h / float(w)

#             if aspect < fall_ratio_thresh:
#                 color = (0, 0, 255)
#                 text = f"FALL ({aspect:.2f})"
#             else:
#                 color = (0, 255, 0)
#                 text = f"PERSON ({aspect:.2f})"

#             new_detections.append((x1, y1, x2, y2, color, text))

#         # on mémorise les detections de cette frame
#         last_detections = new_detections

#     # ======================
#     #   DESSIN (TOUJOURS)
#     # ======================
#     for x1, y1, x2, y2, color, text in last_detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, text, (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     cv2.imshow("Fall Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# Fall detection as ROS2 node – Jetson friendly
import cv2
import numpy as np
import torch
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights


class FallDetectionNode(Node):
    def __init__(self):
        super().__init__('fall_detection_node')

        # ====== Device (CPU / GPU) ======
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Device: {self.device}')

        # ====== Load detection model ======
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).to(self.device)
        self.model.eval()
        self.get_logger().info('SSDLite320_MobileNet_V3 model loaded.')

        # Transform (image -> tensor)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # COCO person ID
        self.PERSON_CLASS_ID = 1

        # Thresholds
        self.conf_thresh = 0.7
        self.fall_ratio_thresh = 0.8    # h/w < 0.8 => allongé
        self.max_area_ratio = 0.90      # box > 90% de l’image => on ignore

        # Optional : ne faire l’inférence qu’1 frame sur N
        self.DETECT_EVERY_N = 3
        self.frame_idx = 0
        self.last_detections = []   # (x1, y1, x2, y2, label_text, is_fall)

        # ====== Publisher ======
        self.publisher_ = self.create_publisher(String, 'fall_detection', 10)

        # ====== Subscriber ======
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',          # même topic que ta caméra
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('FallDetectionNode started! Subscribed to /image_raw/compressed')

    def image_callback(self, msg: CompressedImage):
        """Reçoit une image compressée ROS2, fait la détection et publie FALL / OK."""
        # 1) Decode ROS2 CompressedImage -> OpenCV BGR
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().warn(f'Failed to decode image: {e}')
            return

        if frame is None:
            return

        # Resize pour accélérer (adapter selon ton besoin)
        frame = cv2.resize(frame, (640, 360))
        H, W = frame.shape[:2]

        # Decide if we run detection this frame
        self.frame_idx += 1
        run_detection = (self.frame_idx % self.DETECT_EVERY_N == 0)

        if run_detection:
            # 2) BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(frame_rgb).to(self.device)

            with torch.no_grad():
                outputs = self.model([img_tensor])

            out = outputs[0]
            boxes = out["boxes"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()

            new_detections = []
            fall_detected = False

            for box, label, score in zip(boxes, labels, scores):
                if score < self.conf_thresh:
                    continue
                if label != self.PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue

                box_area = w * h
                img_area = W * H
                if box_area > self.max_area_ratio * img_area:
                    continue

                aspect = h / float(w)

                if aspect < self.fall_ratio_thresh:
                    color = (0, 0, 255)
                    text = f"FALL ({aspect:.2f})"
                    is_fall = True
                    fall_detected = True
                else:
                    color = (0, 255, 0)
                    text = f"PERSON ({aspect:.2f})"
                    is_fall = False

                new_detections.append((x1, y1, x2, y2, color, text, is_fall))

            self.last_detections = new_detections

            # 3) Publish result (simple string, tu peux faire plus propre après)
            msg_out = String()
            if fall_detected:
                msg_out.data = "FALL_DETECTED"
            else:
                msg_out.data = "NO_FALL"
            self.publisher_.publish(msg_out)
            self.get_logger().info(msg_out.data)
        # sinon : on garde last_detections

        # 4) Dessiner les dernières détections sur l’image
        for x1, y1, x2, y2, color, text, is_fall in self.last_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 5) Affichage local (debug)
        cv2.imshow('Fall Detection', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FallDetectionNode()
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

