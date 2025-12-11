#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor

# Imports des nodes AI
from ai_pkg.fsm_node import FSMNode
from ai_pkg.wait_car_node import WaitCarNode
from ai_pkg.qr_node import QRNode
from ai_pkg.face_node import FaceNode
from ai_pkg.dialog_node import DialogNode


def main(args=None):
    rclpy.init(args=args)

    # Instanciation des nodes
    fsm     = FSMNode()
    waitcar = WaitCarNode()
    qr      = QRNode()
    face    = FaceNode()
    dialog  = DialogNode()

    # Executor multi-threads pour faire tourner tout ça ensemble
    executor = MultiThreadedExecutor()
    executor.add_node(fsm)
    executor.add_node(waitcar)
    executor.add_node(qr)
    executor.add_node(face)
    executor.add_node(dialog)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in [fsm, waitcar, qr, face, dialog]:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
