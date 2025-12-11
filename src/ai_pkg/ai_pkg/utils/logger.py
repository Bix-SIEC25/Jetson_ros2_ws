from datetime import datetime
from pathlib import Path

# Chemin unique du fichier de log
LOG_FILE = Path("/home/jetson/ros2_ws/src/ai_pkg/fsm.log")

def log(msg: str):
    """Logger partagé pour tout ai_pkg."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
