#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
import yaml
import argparse

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Bool

from utils import clip_angle 

def _resolve_config_path() -> str:
    import argparse as _ap
    from pathlib import Path as _P
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--config-dir", type=str, default="deployment/config")
    _args, _ = _p.parse_known_args()
    cd = _P(_args.config_dir)
    if not cd.is_absolute():
        cd = _P.cwd() / cd
    return str(cd / "robot.yaml")

CONFIG_PATH = _resolve_config_path()
with open(CONFIG_PATH, "r") as f:
    robot_cfg = yaml.safe_load(f)

MAX_V: float = robot_cfg["max_v"]
MAX_W: float = robot_cfg["max_w"]

RATE: int = robot_cfg["frame_rate"]     
DT: float = 1.0 / RATE                 
EPS: float = 1e-8
WAYPOINT_TIMEOUT_DEFAULT: float = 1.0
WAYPOINT_TIMEOUT_VFH: float = 5.0  
DISTANCE_REPORT_INTERVAL: float = 0.1
MAX_TIME: float = 30000.0


class PDControllerNode(Node):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("pd_controller")
        self.controller_type = args.control
        self.waypoint_timeout = (
            WAYPOINT_TIMEOUT_VFH if args.control == "vfh" else WAYPOINT_TIMEOUT_DEFAULT
        )

        if  args.robot == "turtlebot4":
            waypoint_topic = "/robot2/waypoint"
            vel_topic = "/robot2/cmd_vel"
            odom_topic = "/odom"
        else:
            raise ValueError(f"Unknown robot type: {args.robot}")

        self.get_logger().info(f"Robot Type: {args.robot}, Topics: {waypoint_topic}, {vel_topic}")

        self.waypoint: Optional[np.ndarray] = None
        self._last_wp_time: float = 0.0
        self.reached_goal: bool = False
        self.total_distance: float = 0.0
        self._last_odom_x: Optional[float] = None
        self._last_odom_y: Optional[float] = None
        self.last_report_time: float = time.time()

        self.start_time: float = time.time()
        self.total_time: float = 0.0

        self.vel_pub = self.create_publisher(Twist, vel_topic, 1)
        self.create_subscription(
            Float32MultiArray, waypoint_topic, self._waypoint_cb, 1
        )
        self.create_subscription(Bool, "/topoplan/reached_goal", self._goal_cb, 1)

        from rclpy.qos import QoSProfile, ReliabilityPolicy
        odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, odom_qos)

        self.create_timer(1.0 / RATE, self._timer_cb)
        self.get_logger().info(
            "PD controller node initialised – waiting for waypoints…"
        )

    def _waypoint_cb(self, msg: Float32MultiArray) -> None:
        self.waypoint = np.asarray(msg.data, dtype=float)
        self._last_wp_time = time.time()
        self.get_logger().debug(f"Waypoint received: {self.waypoint.tolist()}")

    def _goal_cb(self, msg: Bool) -> None:
        self.reached_goal = msg.data
        if self.reached_goal:
            self.get_logger().info(f"Total distance: {self.total_distance:.3f} m")
            self.get_logger().info(f"Total time: {self.total_time:.3f} s")

    def _waypoint_valid(self) -> bool:
        return (
            self.waypoint is not None
            and (time.time() - self._last_wp_time) < self.waypoint_timeout
        )

    def _odom_cb(self, msg: Odometry) -> None:
        """Track actual distance traveled from odometry."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self._last_odom_x is not None:
            dx = x - self._last_odom_x
            dy = y - self._last_odom_y
            self.total_distance += math.sqrt(dx * dx + dy * dy)

            if self.total_distance > 100.0:
                self._goal_cb(Bool(data=True))

        self._last_odom_x = x
        self._last_odom_y = y

        current_time = time.time()
        if current_time - self.last_report_time >= DISTANCE_REPORT_INTERVAL:
            self.get_logger().info(f"Current distance (odom): {self.total_distance:.3f} m")
            self.last_report_time = current_time

    def _pd_control(self, wp: np.ndarray) -> Tuple[float, float]:
        """Compute (v, w) for 2‑D or 4‑D waypoint."""
        if wp.size == 2:
            dx, dy = wp
            use_heading = False
        elif wp.size == 4:
            dx, dy, hx, hy = wp
            use_heading = np.abs(dx) < EPS and np.abs(dy) < EPS
        else:
            raise ValueError("Waypoint must be 2‑D or 4‑D vector")

        if dx < -EPS and not use_heading:
            v = float(np.clip(dx / DT, -MAX_V, 0.0))
            return v, 0.0

        if use_heading:
            v = 0.0
            desired_yaw = np.arctan2(hy, hx) 
        elif abs(dx) < EPS:
            v = 0.0
            desired_yaw = np.sign(dy) * np.pi / 2
        else:
            v = dx / DT
            desired_yaw = np.arctan2(dy, dx)

        w = clip_angle(desired_yaw) / DT

        return float(np.clip(v, 0.0, MAX_V)), float(np.clip(w, -MAX_W, MAX_W))

    def _timer_cb(self) -> None:
        vel_msg = Twist()

        current_time = time.time()
        self.total_time = current_time - self.start_time

        print("Current time:", self.total_time)
        if self.total_time > MAX_TIME:
            self._goal_cb(Bool(data=True))

        if self.reached_goal:
            self.vel_pub.publish(vel_msg)
            self.get_logger().info("Reached goal – stopping controller.")
            rclpy.shutdown()
            return

        if self._waypoint_valid():
            v, w = self._pd_control(self.waypoint)
            vel_msg.linear.x = v
            vel_msg.angular.z = w

            self.get_logger().info(f"Publishing velocity: v={v:.3f}, w={w:.3f}")

        self.vel_pub.publish(vel_msg)


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control", type=str, default="nomad", help="control type (nomad, care)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="turtlebot4",
        choices=[ "turtlebot4"],
        help="Robot type",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="deployment/config",
        help="Path to config directory (default: deployment/config, use simulation/config for sim)",
    )
    args, unknown = parser.parse_known_args()

    node = PDControllerNode(args)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
