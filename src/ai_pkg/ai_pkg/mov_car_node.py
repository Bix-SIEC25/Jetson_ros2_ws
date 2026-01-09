#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from time import sleep

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

import tf2_ros
from tf2_ros import TransformException

from ai_pkg import state_flags as sf
from ai_pkg.utils.logger import log


class MovCarNode(Node):

    def __init__(self):
        super().__init__("mov_car_node")

        self.cb_group = ReentrantCallbackGroup()

        # ---------- Params ----------
        self._global_frame = "map"
        self._base_frame = "base_link"
        self._arrive_dist_m = 0.30  # 30 cm

        # ---------- TF (robot pose in map) ----------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

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
            (25, 11.6, -2.508),
            (18, 6.5, -2.508),
            (12, 2, -2.508),
            (8, -1.3, -2.508),
            (3.2, -4.27, -2.508),
            (1.79, -5.41, -2.508),

            (-3.28, -2.08, 1.87),
            (-5.43, -4.97, -0.394),
            (2.74, -4.65, 0.67),
            
            (12.1, 2.1, 0.696),
            (17.5, 6.3, 0.696),
            (24.3, 11.2, 0.696)
        ]

        self._wp_idx = 0
        self._goal_handle = None
        self._in_flight = False
        self._pending_next = False

        # Target currently considered “active” for the euclidean check
        self._current_goal_xy = None  # (x, y)

        # Sequence id to ignore late callbacks from previous goals
        self._goal_seq = 0

        self._timer = self.create_timer(0.2, self._tick, callback_group=self.cb_group)

        log("[MOV_CAR] Started : set the pose now !")
        sleep(20.0)
        log("[MOV_CAR] Started : ready to move the car.")


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
        pose.header.frame_id = self._global_frame
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
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
                log("[MOV_CAR] Goal canceled")
            except Exception as e:
                log(f"[MOV_CAR] Cancel error: {e}")

        self._goal_handle = None
        self._in_flight = False
        self._current_goal_xy = None

    def _get_robot_xy_in_map(self):
        """
        Returns robot (x, y) in map frame using TF map -> base_link.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self._global_frame,
                self._base_frame,
                rclpy.time.Time()  # latest
            )
            x = float(tf.transform.translation.x)
            y = float(tf.transform.translation.y)
            return x, y
        except TransformException:
            return None

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

    def _on_goal_response(self, future, seq_id: int):
        # If this response belongs to an old goal, ignore (or cancel if accepted)
        if seq_id != self._goal_seq:
            try:
                gh = future.result()
                if gh and gh.accepted:
                    gh.cancel_goal_async()
            except Exception:
                pass
            return

        goal_handle = future.result()
        if not goal_handle.accepted:
            log("[MOV_CAR] Goal rejected")
            self._goal_handle = None
            self._in_flight = False
            self._current_goal_xy = None
            return

        self._goal_handle = goal_handle

        # Still listen to result (for logs / safety), but we do NOT use it to advance waypoint.
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut: self._on_result(fut, seq_id))

    def _on_result(self, future, seq_id: int):
        # Ignore old results
        if seq_id != self._goal_seq:
            return

        try:
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                log("[MOV_CAR] Nav2 reports SUCCEEDED")
            else:
                log(f"[MOV_CAR] Nav2 finished status={status}")
        except Exception as e:
            log(f"[MOV_CAR] Result error: {e}")

        # On considère le goal terminé (cancelled ou autre)
        self._goal_handle = None
        self._current_goal_xy = None

        if self._pending_next:
            # On libère l'envoi du prochain waypoint UNE SEULE FOIS
            self._pending_next = False
            self._in_flight = False
        else:
            # Cas normal: goal fini sans "arrivé distance" (ex: abort)
            self._in_flight = False


    # =========================================================
    # Main loop
    # =========================================================
    def _tick(self):
        if self._pending_next:
            return

        # FSM coupe MOV_CAR
        if not sf.mov_car_active:
            if self._goal_handle or self._in_flight:
                self._cancel_goal()
            return

        # If we have an active target, check euclidean distance robot<->waypoint
        if self._in_flight and self._current_goal_xy is not None:
            robot_xy = self._get_robot_xy_in_map()
            if robot_xy is not None:
                rx, ry = robot_xy
                gx, gy = self._current_goal_xy
                dist = math.hypot(gx - rx, gy - ry)

                if dist < self._arrive_dist_m:
                    log(f"[MOV_CAR] Arrived by distance: d={dist:.3f} m < {self._arrive_dist_m:.2f} m -> next WP")
                    self._pending_next = True
                    # On demande l'annulation, mais on NE met PAS _in_flight à False ici,
                    # sinon un autre tick peut envoyer un nouveau goal avant que le cancel soit traité.
                    if self._goal_handle is not None:
                        try:
                            self._goal_handle.cancel_goal_async()
                            log("[MOV_CAR] Goal canceled")
                        except Exception as e:
                            log(f"[MOV_CAR] Cancel error: {e}")
                    else:
                        # Si pas de goal_handle (rare), on peut passer direct au suivant
                        self._in_flight = False
                        self._current_goal_xy = None
                        self._pending_next = False
                    return


        # Envoi du prochain waypoint
        if not self._in_flight:
            if not self._nav_client.wait_for_server(timeout_sec=0.1):
                return

            x, y, yaw = self._waypoints[self._wp_idx]
            self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)

            goal = NavigateToPose.Goal()
            goal.pose = self._make_pose(x, y, yaw)

            self._in_flight = True
            self._current_goal_xy = (float(x), float(y))

            self._goal_seq += 1
            seq_id = self._goal_seq

            log(f"[MOV_CAR] Send goal ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")

            send_future = self._nav_client.send_goal_async(goal)
            send_future.add_done_callback(lambda fut, sid=seq_id: self._on_goal_response(fut, sid))


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
