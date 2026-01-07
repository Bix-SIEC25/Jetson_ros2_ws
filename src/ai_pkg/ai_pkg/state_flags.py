from enum import Enum, auto

RET_NONE = 0
RET_FALL_ACCEL = 10
RET_FALL_AI = 11
RET_VERIFIED = 20
RET_NOT_VERIFIED = 21

class State(Enum):
    FALL_DETECTION = auto()
    WAIT_CAR = auto()
    RESIDENT_RECOGNITION = auto()
    FALL_VERIFICATION = auto()
    DIALOG = auto()

# État global
current_state = State.FALL_DETECTION

# Valeur de return
return_val = 0

# Ordres envoyés par la FSM (True = node doit travailler)
wait_car_active = False
qr_active = False
face_active = False
dialog_active = False
fall_ia_active = False
mov_car_active = False
wait_image_verif_active = False
