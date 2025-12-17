from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from audio_common_msgs.action import TTS
from ai_pkg.utils.logger import log


class _TalkerNode(Node):
    def __init__(self):
        super().__init__("talker_helper")
        self._tts_client = ActionClient(self, TTS, "/say")
        log("TalkerNode initialisé")

    def say(self, text: str):
        if not self._tts_client.wait_for_server(timeout_sec=1.0):
            log("[TTS]: serveur /say indisponible (timeout)")
            return

        goal = TTS.Goal()
        goal.text = text

        log(f"[TTS]: envoi → '{text}'")
        send_future = self._tts_client.send_goal_async(goal)

        def _goal_response_cb(fut):
            try:
                goal_handle = fut.result()
            except Exception as e:
                log(f"[TTS]: échec envoi goal ({e})")
                return

            if not goal_handle.accepted:
                log("[TTS]: goal rejeté par le serveur")
                return

            log(f"[TTS]: goal accepté → '{text}'")

        send_future.add_done_callback(_goal_response_cb)


# ===============================
# Singleton + API publique
# ===============================

_talker_node: Optional[_TalkerNode] = None


def say(text: str):
    """
    Fait parler le robot via l'action /say.
    Non bloquant.
    pactl set-sink-volume @DEFAULT_SINK@ 50%
    
    """
    global _talker_node

    if not rclpy.ok():
        log("[TTS]: rclpy non initialisé, abandon")
        return

    if _talker_node is None:
        _talker_node = _TalkerNode()

    _talker_node.say(text)
