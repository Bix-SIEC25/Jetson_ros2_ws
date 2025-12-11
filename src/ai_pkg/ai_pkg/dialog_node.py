#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


class DialogNode(Node):
    def __init__(self):
        super().__init__("dialog_node")
        self._working = False
        self.timer = self.create_timer(0.2, self.loop)

    def loop(self):
        if not sf.dialog_active:
            self._working = False
            return

        if self._working:
            return

        self._working = True
        log("[DIALOG] Démarrage du travail (simulation dialogue)")

        # --- travail simulé ---
        time.sleep(3.0)
        # ----------------------

        log("[DIALOG] Travail terminé → dialog_active = False")
        sf.dialog_active = False
        self._working = False


def main(args=None):
    rclpy.init(args=args)
    node = DialogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
