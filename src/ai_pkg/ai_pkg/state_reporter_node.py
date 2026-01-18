#!/usr/bin/env python3
import math
import requests

import rclpy
from rclpy.node import Node

import tf2_ros
from tf2_ros import TransformException

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


SERVER_URL = "https://bix.ovh/add_state.php"
PERIOD_S = 3.0
TIMEOUT_S = 2.0

GLOBAL_FRAME = "map"
BASE_FRAME = "base_link"

SEND_IF_NO_TF = False  # True => envoie x=y=dir=0 si TF pas prêt


class StateReporterNode(Node):
    def __init__(self):
        super().__init__("state_reporter_node")

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._http = requests.Session()
        self._timer = self.create_timer(PERIOD_S, self._tick)

        log(f"[STATE_REPORT] Started: {SERVER_URL} every {PERIOD_S:.1f}s | TF {GLOBAL_FRAME}->{BASE_FRAME}")

    def _get_pose_xy_yaw(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                GLOBAL_FRAME,
                BASE_FRAME,
                rclpy.time.Time()
            )
        except TransformException:
            return None

        x = float(tf.transform.translation.x)
        y = float(tf.transform.translation.y)

        q = tf.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return x, y, yaw

    @staticmethod
    def _b(v) -> int:
        return 1 if bool(v) else 0

    def _tick(self):
        pose = self._get_pose_xy_yaw()
        if pose is None:
            if not SEND_IF_NO_TF:
                # log("[STATE_REPORT] TF not ready -> skip")
                return
            x, y, yaw = 0.0, 0.0, 0.0
        else:
            x, y, yaw = pose

        params = {
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "dir": f"{yaw:.3f}",
            "wait_car": self._b(sf.wait_car_active),
            "qr": self._b(sf.qr_active),
            "face": self._b(sf.face_active),
            "dialog": self._b(sf.dialog_active),
            "fall_ia": self._b(sf.fall_ia_active),
            "mov_car": self._b(sf.mov_car_active),
            "wait_image_verif": self._b(sf.wait_image_verif_active),
        }

        try:
            r = self._http.get(SERVER_URL, params=params, timeout=TIMEOUT_S)
            if r.status_code < 200 or r.status_code >= 300:
                log(f"[STATE_REPORT] HTTP {r.status_code} | params={params}")
                return
            log(f"[STATE_REPORT] ok | x={params['x']} y={params['y']} dir={params['dir']}")
        except requests.RequestException as e:
            log(f"[STATE_REPORT] send failed: {e}")

    def destroy_node(self):
        try:
            self._http.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StateReporterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
