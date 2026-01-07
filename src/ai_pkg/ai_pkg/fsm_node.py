import rclpy
from rclpy.node import Node

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


class FSMNode(Node):
    """
    FSM
    """

    def __init__(self):
        super().__init__("scenario_fsm")

        self._fall_winner = None   # "AI" ou "ACCEL" (qui finit en premier dans FALL_DETECTION)
        self._rr_winner = None     # "FACE" ou "QR" (optionnel, juste pour debug)

        log("[FSM] Start -> FALL_DETECTION")
        self.activate(sf.State.FALL_DETECTION)

        self.timer = self.create_timer(0.1, self.loop_once)

    # ---------- Helpers ----------
    def _log_flags(self, prefix=""):
        log(
            f"{prefix}[FSM] state={sf.current_state.name} | "
            f"fall_ia={sf.fall_ia_active}, mov_car={sf.mov_car_active}, "
            f"wait_car={sf.wait_car_active}, "
            f"face={sf.face_active}, qr={sf.qr_active}, "
            f"verif={sf.wait_image_verif_active}, dialog={sf.dialog_active}, "
            f"return_val={sf.return_val}"
        )

    def _reset_return_val(self):
        # Pour éviter d’utiliser une vieille valeur
        sf.return_val = 0

    def _all_off(self):
        sf.wait_car_active = False
        sf.qr_active = False
        sf.face_active = False
        sf.dialog_active = False
        sf.fall_ia_active = False
        sf.mov_car_active = False
        sf.wait_image_verif_active = False

    # ---------- Activation d'état ----------
    def activate(self, state: sf.State):
        sf.current_state = state
        self._reset_return_val()
        self._all_off()

        if state == sf.State.FALL_DETECTION:
            self._fall_winner = None
            sf.fall_ia_active = True
            sf.mov_car_active = True

        elif state == sf.State.WAIT_CAR:
            sf.wait_car_active = True

        elif state == sf.State.RESIDENT_RECOGNITION:
            self._rr_winner = None
            sf.face_active = True
            sf.qr_active = True

        elif state == sf.State.FALL_VERIFICATION:
            sf.wait_image_verif_active = True

        elif state == sf.State.DIALOG:
            sf.dialog_active = True

        self._log_flags(prefix=f"[FSM] {state.name} activé | ")

    # ---------- Boucle ----------
    def loop_once(self):
        st = sf.current_state

        # ===== FALL_DETECTION (parallèle) =====
        if st == sf.State.FALL_DETECTION:
            # Si un finit, il met SON flag à False. La FSM coupe l’autre.
            if (not sf.fall_ia_active) and sf.mov_car_active:
                # AI a fini en premier
                self._fall_winner = "AI"
                sf.mov_car_active = False
                log("[FSM] FALL_DETECTION: FALL_IA finished first -> stop MOV_CAR")

            elif (not sf.mov_car_active) and sf.fall_ia_active:
                # ACCEL/movement a fini en premier
                self._fall_winner = "ACCEL"
                sf.fall_ia_active = False
                log("[FSM] FALL_DETECTION: MOV_CAR finished first -> stop FALL_IA")

            # Quand les deux sont off -> on branche
            if (not sf.fall_ia_active) and (not sf.mov_car_active):
                # Décision : priorité au winner détecté, sinon fallback sur return_val
                if self._fall_winner == "ACCEL" or sf.return_val == sf.RET_FALL_ACCEL:
                    log("[FSM] FALL_DETECTION terminé -> branch=ACCEL -> WAIT_CAR")
                    self.activate(sf.State.WAIT_CAR)
                else:
                    # par défaut AI
                    log("[FSM] FALL_DETECTION terminé -> branch=AI -> RESIDENT_RECOGNITION")
                    self.activate(sf.State.RESIDENT_RECOGNITION)

        # ===== WAIT_CAR =====
        elif st == sf.State.WAIT_CAR:
            if not sf.wait_car_active:
                log("[FSM] WAIT_CAR terminé -> RESIDENT_RECOGNITION")
                self.activate(sf.State.RESIDENT_RECOGNITION)

        # ===== RESIDENT_RECOGNITION (parallèle) =====
        elif st == sf.State.RESIDENT_RECOGNITION:
            # Dès qu’un termine, la FSM coupe l’autre
            if (not sf.face_active) and sf.qr_active:
                self._rr_winner = "FACE"
                sf.qr_active = False
                log("[FSM] RESIDENT_RECOGNITION: FACE finished first -> stop QR")

            elif (not sf.qr_active) and sf.face_active:
                self._rr_winner = "QR"
                sf.face_active = False
                log("[FSM] RESIDENT_RECOGNITION: QR finished first -> stop FACE")

            # Quand les deux sont off -> next
            if (not sf.face_active) and (not sf.qr_active):
                log(f"[FSM] RESIDENT_RECOGNITION terminé (winner={self._rr_winner}) -> FALL_VERIFICATION")
                self.activate(sf.State.FALL_VERIFICATION)

        # ===== FALL_VERIFICATION =====
        elif st == sf.State.FALL_VERIFICATION:
            if not sf.wait_image_verif_active:
                if sf.return_val == sf.RET_VERIFIED:
                    log("[FSM] FALL_VERIFICATION -> VERIFIED -> DIALOG")
                    self.activate(sf.State.DIALOG)
                else:
                    log("[FSM] FALL_VERIFICATION -> NOT_VERIFIED -> FALL_DETECTION")
                    self.activate(sf.State.FALL_DETECTION)

        # ===== DIALOG =====
        elif st == sf.State.DIALOG:
            if not sf.dialog_active:
                log("[FSM] DIALOG terminé -> FALL_DETECTION")
                self.activate(sf.State.FALL_DETECTION)


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
