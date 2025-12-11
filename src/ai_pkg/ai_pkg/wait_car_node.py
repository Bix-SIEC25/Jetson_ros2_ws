#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


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
        """
        Ce callback est appelé dès qu'un message arrive sur /car_arrived_to_fall.
        On met juste à jour l'état interne.
        """
        self.car_arrived = msg.data

        if msg.data:
            log("[WAIT_CAR] Signal reçu : car_arrived_to_fall = TRUE")

    # -------------------------------------------------------------------
    # Boucle principale pilotée par la FSM
    # -------------------------------------------------------------------
    def loop(self):
        # Si la FSM n'a pas activé cette étape → idle
        if not sf.wait_car_active:
            self._working = False
            return

        # Si on a déjà commencé à attendre → on ne relance pas
        if self._working:
            # Si la voiture vient d'arriver → terminer l'étape
            if self.car_arrived:
                log("[WAIT_CAR] Voiture détectée -> Fin d'étape WAIT_CAR → wait_car_active = False")
                sf.wait_car_active = False
                self._working = False
                self.car_arrived = False  # reset local
            return

        # --- Activation de l'attente ---
        self._working = True
        log("[WAIT_CAR] Activation par la FSM → attente du signal /car_arrived_to_fall")


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
