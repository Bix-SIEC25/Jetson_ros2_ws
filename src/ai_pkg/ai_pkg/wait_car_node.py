#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log
from ai_pkg.utils.speaker import say


class WaitCarNode(Node):
    """
    Étape d'attente de voiture :
        - Le node s'active uniquement si sf.wait_car_active == True (FSM)
        - Il écoute le topic /car_arrived_to_fall (Bool)
        - Dès qu'il reçoit True → fin de l'étape → sf.wait_car_active = False
    """

    def __init__(self):
        super().__init__("wait_car_node")

        self._working = False
        self.car_arrived = False

        # Subscriber : la voiture est arrivée ?
        self.sub = self.create_subscription(
            Bool,
            '/car_arrived_to_fall',
            self.car_callback,
            10
        )

        # Timer pour vérifier régulièrement si on doit travailler
        self.timer = self.create_timer(0.2, self.loop)

        log("[WAIT_CAR] Node initialisé, en attente d'activation FSM.")

    # -------------------------------------------------------------------
    # Callback du topic : /car_arrived_to_fall
    # -------------------------------------------------------------------
    def car_callback(self, msg: Bool):
        # Ne mémorise que pendant l'étape active
        if not sf.wait_car_active:
            return
        self.car_arrived = bool(msg.data)
        if self.car_arrived and self._working:
            log("[WAIT_CAR] Signal reçu : car_arrived_to_fall = TRUE")

    # -------------------------------------------------------------------
    # Boucle principale pilotée par la FSM
    # -------------------------------------------------------------------
    def loop(self):
        # Si la FSM n'a pas activé cette étape → idle
        if not sf.wait_car_active:
            self._working = False
            self.car_arrived = False   # reset pour éviter le stale
            return
        
        if not self._working:
            self._working = True
            self.car_arrived = False   # reset à l'entrée d'état
            log("[WAIT_CAR] Activation par la FSM → attente du signal /car_arrived_to_fall")
            return

        if self.car_arrived:
            log("[WAIT_CAR] Voiture détectée -> Fin d'étape WAIT_CAR → wait_car_active = False")
            say("The car has arrived")
            sf.wait_car_active = False
            self._working = False
            self.car_arrived = False  # reset local
            return


# -----------------------------------------------------------------------
# MAIN (uniquement pour lancer le node seul)
# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = WaitCarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
