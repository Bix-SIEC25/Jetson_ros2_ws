# fsm_node.py
import rclpy
from rclpy.node import Node
from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


class FSMNode(Node):
    def __init__(self):
        super().__init__("scenario_fsm")
        log(f"[FSM] Start at state: {sf.current_state.name}")

        self.activate(sf.State.WAIT_CAR)

        self.timer = self.create_timer(2.0, self.loop_once)

    def activate(self, state):
        sf.current_state = state

        # La FSM ne met que à TRUE → jamais à FALSE
        if state == sf.State.WAIT_CAR:
            sf.wait_car_active = True

        elif state == sf.State.QR:
            sf.qr_active = True

        elif state == sf.State.FACE:
            sf.face_active = True

        elif state == sf.State.DIALOG:
            sf.dialog_active = True

        log(
            f"[FSM] {state.name} activé | "
            f"(wait_car={sf.wait_car_active}, qr={sf.qr_active}, "
            f"face={sf.face_active}, dialog={sf.dialog_active})"
        )

    def loop_once(self):
        # La FSM attend que le node ait remis son booléen à False
        if sf.current_state == sf.State.WAIT_CAR:
            if not sf.wait_car_active:      # ← node a fini
                self.activate(sf.State.QR)

        elif sf.current_state == sf.State.QR:
            if not sf.qr_active:
                self.activate(sf.State.FACE)

        elif sf.current_state == sf.State.FACE:
            if not sf.face_active:
                self.activate(sf.State.DIALOG)

        elif sf.current_state == sf.State.DIALOG:
            if not sf.dialog_active:
                self.activate(sf.State.WAIT_CAR)


def main(args=None):
    rclpy.init(args=args)
    node = FSMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
