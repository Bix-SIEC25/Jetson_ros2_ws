#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
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

        # ---------- TF ----------
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

        # ---------- Waypoints (A ... B) ----------
        self._waypoints = [
            (1.79, -5.41, -2.508),     # A
            (-3.28, -2.08, 1.87),
            (-5.43, -4.97, -0.394),
            (24.3, 11.2, 0.696)        # B
        ]

        self._wp_idx = 0
        self._direction = +1  # +1 vers B, -1 vers A (défini à la ré-entrée)
        self._goal_handle = None
        self._in_flight = False
        self._pending_next = False
        self._current_goal_xy = None  # (x, y)
        self._goal_seq = 0

        # yaws de référence pour discriminer A/B
        self._yaw_ref_A = -2.508
        self._yaw_ref_B = 0.696

        # Detect re-entry
        self._was_active = False
        self._need_route_init = False

        # Initial pose gate
        self._have_initialpose = False
        self._sub_initpose = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self._on_initialpose,
            10,
            callback_group=self.cb_group
        )

        self._timer = self.create_timer(0.2, self._tick, callback_group=self.cb_group)
        log("[MOV_CAR] Started : waiting for /initialpose (set pose in RViz2)")

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
        self._pending_next = False  # éviter blocage

    def _get_robot_xy_in_map(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                self._global_frame,
                self._base_frame,
                rclpy.time.Time()
            )
            return float(tf.transform.translation.x), float(tf.transform.translation.y)
        except TransformException:
            return None

    def _get_robot_yaw_in_map(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                self._global_frame,
                self._base_frame,
                rclpy.time.Time()
            )
            q = tf.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny_cosp, cosy_cosp)
        except TransformException:
            return None

    def _wrap_pi(self, a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def _ang_dist(self, a: float, b: float) -> float:
        return abs(self._wrap_pi(a - b))

    def _init_route_from_yaw(self) -> bool:
        """
        YAW-ONLY:
        - si yaw proche de yaw_ref_A => on vise A (idx 0), direction = -1
        - sinon => on vise B (idx last), direction = +1
        """
        yaw = self._get_robot_yaw_in_map()
        if yaw is None:
            return False

        dA = self._ang_dist(yaw, self._yaw_ref_A)
        dB = self._ang_dist(yaw, self._yaw_ref_B)

        if dA < dB:
            # vers A
            self._direction = -1
            self._wp_idx = 0
            target = "A"
        else:
            # vers B
            self._direction = +1
            self._wp_idx = len(self._waypoints) - 1
            target = "B"

        log(f"[MOV_CAR] Re-entry init (yaw-only): yaw={yaw:.2f}, dA={dA:.2f}, dB={dB:.2f} -> target {target}, wp_idx={self._wp_idx}, dir={self._direction}")
        return True

    def _advance_wp_pingpong(self):
        """
        Met à jour _wp_idx pour aller au waypoint suivant dans le sens self._direction
        avec rebond aux extrémités.
        """
        n = len(self._waypoints)
        if n <= 1:
            return

        nxt = self._wp_idx + self._direction

        # rebond côté A
        if nxt < 0:
            self._direction = +1
            nxt = 1

        # rebond côté B
        if nxt >= n:
            self._direction = -1
            nxt = n - 2

        self._wp_idx = nxt

    # =========================================================
    # ROS callbacks
    # =========================================================
    def _on_initialpose(self, msg: PoseWithCovarianceStamped):
        if not self._have_initialpose:
            self._have_initialpose = True
            log("[MOV_CAR] Initial pose received -> ready")

    def _on_someone_fell(self, msg: Bool):
        if not msg.data or not sf.mov_car_active:
            return

        log("[MOV_CAR] someone_fell=True -> STOP + RET_FALL_ACCEL")
        sf.return_val = sf.RET_FALL_ACCEL
        sf.mov_car_active = False
        self._cancel_goal()

    def _on_goal_response(self, future, seq_id: int):
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
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut: self._on_result(fut, seq_id))

    def _on_result(self, future, seq_id: int):
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

        self._goal_handle = None
        self._current_goal_xy = None

        # si on avait "arrivé par distance" -> on libère l'envoi du suivant
        if self._pending_next:
            self._pending_next = False
            self._in_flight = False
        else:
            self._in_flight = False

    # =========================================================
    # Main loop
    # =========================================================
    def _tick(self):
        # détecte ré-entrée (front montant)
        if sf.mov_car_active and not self._was_active:
            self._cancel_goal()
            if self._have_initialpose:
                self._need_route_init = True
            else:
                log("[MOV_CAR] Enter MOV_CAR but no /initialpose yet -> waiting")

        self._was_active = sf.mov_car_active

        # état désactivé
        if not sf.mov_car_active:
            if self._goal_handle or self._in_flight:
                self._cancel_goal()
            return

        # attente initial pose
        if not self._have_initialpose:
            return

        # init route yaw-only (on réessaie tant que TF pas prêt)
        if self._need_route_init:
            if not self._init_route_from_yaw():
                return
            self._need_route_init = False

        # évite double-send pendant cancel
        if self._pending_next:
            return

        # check arrivée par distance
        if self._in_flight and self._current_goal_xy is not None:
            robot_xy = self._get_robot_xy_in_map()
            if robot_xy is not None:
                rx, ry = robot_xy
                gx, gy = self._current_goal_xy
                dist = math.hypot(gx - rx, gy - ry)

                if dist < self._arrive_dist_m:
                    log(f"[MOV_CAR] Arrived by distance: d={dist:.3f} m < {self._arrive_dist_m:.2f} m -> next WP")
                    self._pending_next = True

                    if self._goal_handle is not None:
                        try:
                            self._goal_handle.cancel_goal_async()
                            log("[MOV_CAR] Goal canceled")
                        except Exception as e:
                            log(f"[MOV_CAR] Cancel error: {e}")
                    else:
                        # rare
                        self._in_flight = False
                        self._current_goal_xy = None
                        self._pending_next = False
                    return

        # envoi du prochain waypoint
        if not self._in_flight:
            if not self._nav_client.wait_for_server(timeout_sec=0.1):
                return

            x, y, yaw = self._waypoints[self._wp_idx]

            goal = NavigateToPose.Goal()
            goal.pose = self._make_pose(x, y, yaw)

            self._in_flight = True
            self._current_goal_xy = (float(x), float(y))

            self._goal_seq += 1
            seq_id = self._goal_seq

            log(f"[MOV_CAR] Send goal idx={self._wp_idx} ({x:.2f}, {y:.2f}, yaw={yaw:.2f}) dir={self._direction}")

            # IMPORTANT: prépare le waypoint suivant dans le bon sens (ping-pong)
            self._advance_wp_pingpong()

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
