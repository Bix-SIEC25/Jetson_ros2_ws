#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


class QRNode(Node):
    def __init__(self):
        super().__init__("qr_node")
        self._working = False
        self.timer = self.create_timer(0.2, self.loop)

    def loop(self):
        if not sf.qr_active:
            self._working = False
            return

        if self._working:
            return

        self._working = True
        log("[QR] Démarrage du travail (simulation scan QR)")

        # --- travail simulé ---
        time.sleep(2.0)
        # ----------------------

        log("[QR] Travail terminé → qr_active = False")
        sf.qr_active = False
        self._working = False


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
