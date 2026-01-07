#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


class MovCarNode(Node):

    def __init__(self):
        super().__init__("mov_car_node")

        self.cb_group = ReentrantCallbackGroup()

        # ---------- Nav2 Action client ----------
        self._nav_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose",
            callback_group=self.cb_group
        )

        # ---------- Fall detection ----------
        self._sub_fall = self.create_subscription(
            Bool,
            "/someone_fell",
            self._on_someone_fell,
            10,
            callback_group=self.cb_group
        )

        # ---------- Waypoints (x, y, yaw) ----------
        self._waypoints = [
            (1.0, 2.0, 0.0),
            (2.0, 2.0, 0.0),
            (2.0, 1.0, -math.pi / 2),
            (1.0, 1.0, math.pi),
            (0.0, 1.0, math.pi / 2),
            (0.0, 2.0, 0.0),
        ]

        self._wp_idx = 0
        self._goal_handle = None
        self._in_flight = False

        self._timer = self.create_timer(0.2, self._tick, callback_group=self.cb_group)

        log("[MOV_CAR] Started")

    # =========================================================
    # Utils
    # =========================================================
    def _yaw_to_quaternion(self, yaw: float):
        return (
            0.0,
            0.0,
            math.sin(yaw / 2.0),
            math.cos(yaw / 2.0)
        )

    def _make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        qx, qy, qz, qw = self._yaw_to_quaternion(yaw)

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def _cancel_goal(self):
        if self._goal_handle:
            self._goal_handle.cancel_goal_async()
            log("[MOV_CAR] Goal canceled")
        self._goal_handle = None
        self._in_flight = False

    # =========================================================
    # ROS callbacks
    # =========================================================
    def _on_someone_fell(self, msg: Bool):
        if not msg.data or not sf.mov_car_active:
            return

        log("[MOV_CAR] someone_fell=True -> STOP + RET_FALL_ACCEL")

        sf.return_val = sf.RET_FALL_ACCEL
        sf.mov_car_active = False
        self._cancel_goal()

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            log("[MOV_CAR] Goal rejected")
            self._in_flight = False
            return

        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        self._goal_handle = None
        self._in_flight = False

        try:
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                log("[MOV_CAR] Goal reached")
            else:
                log(f"[MOV_CAR] Goal finished status={status}")
        except Exception as e:
            log(f"[MOV_CAR] Result error: {e}")

    # =========================================================
    # Main loop
    # =========================================================
    def _tick(self):
        # FSM coupe MOV_CAR
        if not sf.mov_car_active and False:
            if self._goal_handle:
                self._cancel_goal()
            return

        # Envoi du prochain waypoint
        if not self._in_flight and False:
            if not self._nav_client.wait_for_server(timeout_sec=0.1):
                return

            x, y, yaw = self._waypoints[self._wp_idx]
            self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)

            goal = NavigateToPose.Goal()
            goal.pose = self._make_pose(x, y, yaw)

            self._in_flight = True
            log(f"[MOV_CAR] Send goal ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")

            send_future = self._nav_client.send_goal_async(goal)
            send_future.add_done_callback(self._on_goal_response)


def main(args=None):
    rclpy.init(args=args)
    node = MovCarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
