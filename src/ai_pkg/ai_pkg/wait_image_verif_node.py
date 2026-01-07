#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log
from ai_pkg.utils.speaker import say


class WaitImageVerifNode(Node):
    """
    Étape d'attente de vérification d'image :
        - S'active uniquement si sf.wait_image_verif_active == True (FSM)
        - Écoute /image_verified (Bool)
        - Dès qu'il reçoit un message pendant l'étape :
            True  -> sf.return_val = sf.RET_VERIFIED
            False -> sf.return_val = sf.RET_NOT_VERIFIED
          puis fin d'étape -> sf.wait_image_verif_active = False
    """

    def __init__(self):
        super().__init__("wait_image_verif_node")

        self._working = False
        self._last_msg = None  # None = pas encore reçu de message pour cette étape

        self.sub = self.create_subscription(
            Bool,
            "/image_verified",
            self.image_callback,
            10
        )

        self.timer = self.create_timer(0.2, self.loop)

        log("[WAIT_IMAGE_VERIF] Node initialisé, en attente d'activation FSM.")

    # -------------------------------------------------------------------
    # Callback du topic : /image_verified
    # -------------------------------------------------------------------
    def image_callback(self, msg: Bool):
        # On mémorise le dernier message reçu
        self._last_msg = bool(msg.data)

        # Log uniquement si on est en phase active, sinon ça spam pour rien
        if self._working:
            log(f"[WAIT_IMAGE_VERIF] Reçu /image_verified = {self._last_msg}")

    # -------------------------------------------------------------------
    # Boucle principale pilotée par la FSM
    # -------------------------------------------------------------------
    def loop(self):
        # Si la FSM n'a pas activé cette étape → idle + reset local
        if not sf.wait_image_verif_active:
            self._working = False
            self._last_msg = None
            return

        # Première entrée dans l'étape
        if not self._working:
            self._working = True
            self._last_msg = None  # reset pour cette étape
            log("[WAIT_IMAGE_VERIF] Activation par la FSM → attente du signal /image_verified")
            return

        # Étape active : dès qu'on a un message, on tranche et on termine
        if self._last_msg is None:
            return

        if self._last_msg:
            log("[WAIT_IMAGE_VERIF] Image vérifiée -> VERIFIED")
            say("The image has been verified")
            sf.return_val = sf.RET_VERIFIED
        else:
            log("[WAIT_IMAGE_VERIF] Image non vérifiée -> NOT_VERIFIED")
            say("The image has not been verified")
            sf.return_val = sf.RET_NOT_VERIFIED

        # Fin d'étape
        sf.wait_image_verif_active = False
        self._working = False
        self._last_msg = None


def main(args=None):
    rclpy.init(args=args)
    node = WaitImageVerifNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
