from enum import Enum, auto

class State(Enum):
    WAIT_CAR = auto()
    QR = auto()
    FACE = auto()
    DIALOG = auto()

# État global
current_state = State.WAIT_CAR

# Valeur de return
return_val = 0

# Ordres envoyés par la FSM (True = node doit travailler)
wait_car_active = False
qr_active = False
face_active = False
dialog_active = False
