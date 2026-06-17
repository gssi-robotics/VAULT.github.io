from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import Deque
_DA2_METRIC = str(Path(__file__).resolve().parents[2] / "Depth-Anything-V2" / "metric_depth")
if _DA2_METRIC not in sys.path:
    sys.path.insert(0, _DA2_METRIC)

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action

from depth_anything_v2.dpt import DepthAnythingV2

from VfhPlus import defaults as D
from VfhPlus.vfh_star import VFHStar
from VfhPlus.depth_processing import compute_distance_vector, TemporalAggregator, pad_distance_vector
from VfhPlus.nomad_vector import waypoint_to_reference, bin_to_waypoint, generate_direction_waypoints
from VfhPlus.depth_markers import DepthMarkerPublisher
THIS_DIR = Path.cwd()
print(f"This is the Directory in explore_vfh {THIS_DIR}")

def _parse_config_dir() -> Path:
    """Extract --config-dir early so module-level config loading works."""
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--config-dir", type=str, default="deployment/config")
    _args, _ = _p.parse_known_args()
    cd = Path(_args.config_dir)
    if not cd.is_absolute():
        cd = THIS_DIR / cd
    return cd

_CONFIG_DIR = _parse_config_dir()

ROBOT_CONFIG_PATH = _CONFIG_DIR / "robot.yaml"
MODEL_CONFIG_PATH = THIS_DIR / "deployment/config/models.yaml" 
VFH_CONFIG_PATH = _CONFIG_DIR / "vfh.yaml"

print(f"Config directory: {_CONFIG_DIR}")

with open(ROBOT_CONFIG_PATH, "r") as f:
    ROBOT_CONF = yaml.safe_load(f)
MAX_V = ROBOT_CONF["max_v"]
MAX_W = ROBOT_CONF["max_w"]
RATE = ROBOT_CONF["frame_rate"]

with open(VFH_CONFIG_PATH, "r") as f:
    VFH_CONF = yaml.safe_load(f)


def _load_model(model_name: str, device: torch.device):
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    model_config_path = model_paths[model_name]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    ckpt_path = model_paths[model_name]["ckpt_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
    print(f"Loading model from {ckpt_path}")
    model = load_model(ckpt_path, model_params, device)
    return model.to(device).eval(), model_params


class VFHExplorationNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("vfh_exploration")
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.model, self.model_params = _load_model(args.model, self.device)
        self.context_size: int = self.model_params["context_size"]
        self.last_ctx_time = self.get_clock().now()
        self.ctx_dt = 0.1

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.context_queue: Deque[np.ndarray] = deque(maxlen=self.context_size + 1)
        self.bridge = CvBridge()

        if args.robot == "locobot" or args.robot == "locobot2":
            image_topic = "/robot1/camera/image" if args.robot == "locobot" else "/robot3/camera/image"
            waypoint_topic = "/robot1/waypoint" if args.robot == "locobot" else "/robot3/waypoint"
            sampled_actions_topic = ("/robot1" if args.robot == "locobot" else "/robot3") + "/sampled_actions"
            trajectory_viz_topic = ("/robot1" if args.robot == "locobot" else "/robot3") + "/trajectory_viz"
            self.DIM = (320, 240)
        elif args.robot == "robomaster":
            image_topic = "/camera/image_color"
            waypoint_topic = "/robot3/waypoint"
            sampled_actions_topic = "/robot3/sampled_actions"
            trajectory_viz_topic = "/robot3/trajectory_viz"
            self.DIM = (640, 360)
        elif args.robot == "turtlebot4":
            image_topic = "/robot2/oakd/rgb/preview/image_raw"
            waypoint_topic = "/robot2/waypoint"
            sampled_actions_topic = "/robot2/sampled_actions"
            trajectory_viz_topic = "/robot2/trajectory_viz"
            self.DIM = (320, 200)
        else:
            raise ValueError(f"Unknown robot type: {args.robot}")

        self.vfh_num_bins = VFH_CONF.get("num_bins", D.NUM_BINS)
        self.vfh_fov_deg = VFH_CONF.get("fov_deg", D.FOV_DEG)
        self.vfh_max_range = VFH_CONF.get("max_sensing_range", D.MAX_RANGE)
        self.vfh_vertical_margin = VFH_CONF.get("vertical_margin", D.VERTICAL_MARGIN)
        self.vfh_floor_margin = VFH_CONF.get("floor_margin", D.FLOOR_MARGIN)
        self.vfh_safety_margin = VFH_CONF.get("safety_margin", D.SAFETY_MARGIN)
        self.vfh_depth_scale = VFH_CONF.get("depth_scale", D.DEPTH_SCALE)
        self.vfh_speed_reduction = VFH_CONF.get("speed_reduction", D.SPEED_REDUCTION)
        self.vfh_num_waypoints = VFH_CONF.get("num_vfh_waypoints", D.NUM_VFH_WAYPOINTS)
        self.vfh_waypoint_index = VFH_CONF.get("vfh_waypoint_index", D.VFH_WAYPOINT_INDEX)

        self.fov_padding_bins = VFH_CONF.get("fov_padding_bins", D.FOV_PADDING_BINS)
        self.vfh_total_bins = self.vfh_num_bins + 2 * self.fov_padding_bins
        bin_width_deg = self.vfh_fov_deg / self.vfh_num_bins
        self.vfh_virtual_fov_deg = bin_width_deg * self.vfh_total_bins

        self.vfh = VFHStar(
            num_bins=self.vfh_total_bins,
            fov_deg=self.vfh_virtual_fov_deg,
            safety_threshold=VFH_CONF.get("safety_threshold", D.SAFETY_THRESHOLD),
            s_max=VFH_CONF.get("s_max", D.S_MAX),
            mu1=VFH_CONF.get("mu1", D.MU1),
            mu2=VFH_CONF.get("mu2", D.MU2),
            mu3=VFH_CONF.get("mu3", D.MU3),
            robot_radius=VFH_CONF.get("robot_radius", D.ROBOT_RADIUS),
            recovery_reverse_cycles=VFH_CONF.get("recovery_reverse_cycles", D.RECOVERY_REVERSE_CYCLES),
            recovery_turn_cycles=VFH_CONF.get("recovery_turn_cycles", D.RECOVERY_TURN_CYCLES),
            fov_padding_bins=self.fov_padding_bins,
        )

        intrinsics_path = self._get_intrinsics_path_from_config()
        self._init_depth_model(intrinsics_path)

        self.distance_vector = pad_distance_vector(
            np.full(self.vfh_num_bins, np.inf),
            padding_bins=self.fov_padding_bins,
        )
        self.temporal_agg = TemporalAggregator(
            num_bins=self.vfh_num_bins,
            window_size=VFH_CONF.get("temporal_window", D.TEMPORAL_WINDOW),
            danger_threshold=VFH_CONF.get("safety_threshold", D.SAFETY_THRESHOLD),
        )
        self.current_waypoint = np.zeros(2)
        self._new_depth_available = False  

        self.depth_marker_pub = DepthMarkerPublisher(
            node=self,
            topic="/vfh/depth_markers",
            num_bins=self.vfh_total_bins,
            fov_deg=self.vfh_virtual_fov_deg,
            max_range=self.vfh_max_range,
            safety_threshold=VFH_CONF.get("safety_threshold", D.SAFETY_THRESHOLD),
        )

        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0
        self._cmd_v = 0.0
        self._cmd_w = 0.0

        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, image_topic, self._image_cb, image_qos)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, waypoint_topic, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, sampled_actions_topic, 1
        )
        self.viz_pub = self.create_publisher(Image, trajectory_viz_topic, 1)

        from nav_msgs.msg import Odometry as OdomMsg
        self.create_subscription(OdomMsg, "/odom", self._odom_cb, 10)

        from geometry_msgs.msg import Twist
        vel_topic = {
            "locobot": "/robot1/cmd_vel",
            "locobot2": "/robot3/cmd_vel",
            "robomaster": "/cmd_vel",
            "turtlebot4": "/robot2/cmd_vel",
        }.get(args.robot, "/cmd_vel")
        self.create_subscription(Twist, vel_topic, self._cmd_vel_cb, 10)

        self.create_timer(1.0 / RATE, self._timer_cb)

        self._log_params(image_topic, waypoint_topic, sampled_actions_topic,
                         trajectory_viz_topic, intrinsics_path)

    def _get_intrinsics_path_from_config(self) -> str:
        if "intrinsics_path" in ROBOT_CONF:
            p = ROBOT_CONF["intrinsics_path"]
        else:
            robot_key = f"{self.args.robot}_intrinsics_path"
            defaults = {
                "turtlebot4": "intrinsic/turtlebot4/intrinsics.npy",
            }
            p = ROBOT_CONF.get(robot_key, defaults.get(self.args.robot, ""))
        if not os.path.isabs(p):
            p = str(THIS_DIR / p)
        return p

    def _init_depth_model(self, intrinsics_path: str):
        if not intrinsics_path or not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
        self.K = np.load(intrinsics_path)
        self.get_logger().info(f"Loaded camera intrinsics from: {intrinsics_path}")

        da2_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        encoder = VFH_CONF.get("depth_encoder", "vits")
        max_depth = VFH_CONF.get("depth_max_depth", 20)
        weights = VFH_CONF.get("depth_weights", "")
        if not os.path.isabs(weights):
            weights = str(THIS_DIR / weights)
        if not os.path.exists(weights):
            raise FileNotFoundError(f"DA2 metric weights not found: {weights}")

        self.depth_model = DepthAnythingV2(**{**da2_configs[encoder], "max_depth": max_depth})
        self.depth_model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)
        )
        self.depth_model = self.depth_model.to(self.device).eval()

    def _log_params(self, img_t, wp_t, sa_t, viz_t, intr):
        L = self.get_logger().info
        L("=" * 60)
        L("VFH* EXPLORATION NODE (DA2 metric + VFH*)")
        L("=" * 60)
        L(f"Robot: {self.args.robot}")
        L(f"Image topic: {img_t}")
        L("-" * 60)
        L("ROBOT CONFIGURATION:")
        L(f"  Max V: {MAX_V} m/s,  Max W: {MAX_W} rad/s,  Rate: {RATE} Hz")
        L("-" * 60)
        L("VFH* CONFIGURATION:")
        L(f"  Bins: {self.vfh_num_bins},  FOV: {self.vfh_fov_deg}°")
        L(f"  FOV padding: {self.fov_padding_bins} bins/side → total {self.vfh_total_bins} bins, virtual FOV {self.vfh_virtual_fov_deg:.1f}°")
        L(f"  Safety threshold: {VFH_CONF['safety_threshold']} m")
        L(f"  Max sensing range: {self.vfh_max_range} m")
        L(f"  s_max: {VFH_CONF['s_max']}")
        L(f"  Cost weights: μ₁={VFH_CONF['mu1']}, μ₂={VFH_CONF['mu2']}, μ₃={VFH_CONF['mu3']}")
        L(f"  Speed reduction: {self.vfh_speed_reduction}")
        L(f"  VFH waypoints: {self.vfh_num_waypoints}, select index: {self.vfh_waypoint_index}")
        L("-" * 60)
        L(f"DEPTH MODEL: DA2 metric ({VFH_CONF.get('depth_encoder', 'vits')})")
        L(f"  Intrinsics: {intr}")
        L("-" * 60)
        L("MODEL CONFIGURATION:")
        L(f"  Model: {self.args.model},  Device: {self.device}")
        L(f"  Context size: {self.context_size}")
        L(f"  Trajectory length: {self.model_params['len_traj_pred']}")
        L(f"  Diffusion iters: {self.model_params['num_diffusion_iters']}")
        L("-" * 60)
        L("ROS TOPICS:")
        L(f"  Subscribe: {img_t}")
        L(f"  Waypoint:  {wp_t}")
        L(f"  Actions:   {sa_t}")
        L(f"  Viz:       {viz_t}")
        L("=" * 60)

    def _odom_cb(self, msg):
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y
        import math as _m
        q = msg.pose.pose.orientation
        self._odom_theta = _m.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _cmd_vel_cb(self, msg):
        self._cmd_v = msg.linear.x
        self._cmd_w = msg.angular.z

    def _image_cb(self, msg: Image):
        now = self.get_clock().now()
        if (now - self.last_ctx_time).nanoseconds < self.ctx_dt * 1e9:
            return
        self.context_queue.append(msg_to_pil(msg))
        self.last_ctx_time = now

        cv2_img = self.bridge.imgmsg_to_cv2(msg)
        if self.args.robot in ("locobot", "locobot2"):
            frame = cv2.resize(cv2_img, self.DIM)
        else:
            frame = cv2_img

        with torch.no_grad():
            depth_map = self.depth_model.infer_image(frame)  
        raw_dv = compute_distance_vector(
            depth_map, self.K,
            num_bins=self.vfh_num_bins,
            fov_deg=self.vfh_fov_deg,
            max_range=self.vfh_max_range,
            vertical_margin=self.vfh_vertical_margin,
            floor_margin=self.vfh_floor_margin,
            safety_margin=self.vfh_safety_margin,
            depth_scale=self.vfh_depth_scale,
        )
        smoothed_dv = self.temporal_agg.update(raw_dv)
        self.distance_vector = pad_distance_vector(
            smoothed_dv, padding_bins=self.fov_padding_bins,
        )
        self._new_depth_available = True


    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            return np.pi
        return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

    def _select_closest_traj_angle(
        self, trajs: np.ndarray, default_idx: int = 0
    ) -> int:
        prev_wp = self.current_waypoint
        if np.linalg.norm(prev_wp) < 1e-3:
            return default_idx
        cand_wps = trajs[:, self.args.waypoint]
        angles = np.array([self._angle_between(prev_wp, wp) for wp in cand_wps])
        return int(np.argmin(angles))






    def _republish_waypoint(self):
        if np.linalg.norm(self.current_waypoint) > 1e-3:
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = [float(self.current_waypoint[0]),
                                 float(self.current_waypoint[1])]
            self.waypoint_pub.publish(waypoint_msg)

    def _timer_cb(self):
        self._republish_waypoint()

        if len(self.context_queue) <= self.context_size:
            return

        if not self._new_depth_available:
            return
        self._new_depth_available = False

        obs_imgs = transform_images(
            list(self.context_queue), self.model_params["image_size"], center_crop=False
        ).to(self.device)
        fake_goal = torch.randn(
            (1, 3, *self.model_params["image_size"]), device=self.device
        )
        mask = torch.ones(1, device=self.device, dtype=torch.long)

        with torch.no_grad():
            obs_cond = self.model(
                "vision_encoder",
                obs_img=obs_imgs,
                goal_img=fake_goal,
                input_goal_mask=mask,
            )
            rep_fn = (
                (lambda x: x.repeat(self.args.num_samples, 1))
                if obs_cond.ndim == 2
                else (lambda x: x.repeat(self.args.num_samples, 1, 1))
            )
            obs_cond = rep_fn(obs_cond)

            len_traj = self.model_params["len_traj_pred"]
            naction = torch.randn(
                (self.args.num_samples, len_traj, 2), device=self.device
            )
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(noise_pred, k, naction).prev_sample

        traj_batch = to_numpy(get_action(naction))  
        all_ref_bins = []
        all_ref_wps = []
        for i in range(len(traj_batch)):
            wp = traj_batch[i][self.args.waypoint]
            _, rb = waypoint_to_reference(
                float(wp[0]), float(wp[1]),
                num_bins=self.vfh_total_bins,
                fov_deg=self.vfh_virtual_fov_deg,
            )
            all_ref_bins.append(rb)
            all_ref_wps.append(wp)

        best_bin, best_angle, was_modified = self.vfh.compute(
            self.distance_vector, all_ref_bins
        )

        if not was_modified and best_bin in all_ref_bins:
            chosen_idx = all_ref_bins.index(best_bin)
        else:
            chosen_idx = int(np.argmin([abs(b - best_bin) for b in all_ref_bins]))
        chosen_wp = all_ref_wps[chosen_idx]
        ref_bin = all_ref_bins[chosen_idx]

        if self.vfh.recovery_just_completed:
            self.temporal_agg.reset()
            self.vfh.recovery_just_completed = False
            print("[VFH*] Flushed temporal aggregator after recovery")


        self.depth_marker_pub.publish(
            self.distance_vector,
            selected_bin=best_bin,
            reference_bin=ref_bin,
        )

        vfh_waypoint = None
        if was_modified:
            if self.vfh._recovery_phase == self.vfh._PHASE_TURN:
                import math as _math
                hx = _math.cos(best_angle)
                hy = _math.sin(best_angle)
                vfh_waypoint = np.array([0.0, 0.0, hx, hy])
                print(f"[VFH*] TURN phase: rotating in place toward {_math.degrees(best_angle):.1f}°")
            else:
                original_magnitude = np.linalg.norm(chosen_wp)
                vfh_waypoints = generate_direction_waypoints(
                    best_angle,
                    max_magnitude=original_magnitude * self.vfh_speed_reduction,
                    num_waypoints=self.vfh_num_waypoints,
                )
                wp_idx = min(self.vfh_waypoint_index, len(vfh_waypoints) - 1)
                vfh_waypoint = vfh_waypoints[wp_idx]
                print(f"VFH* selected waypoint index {wp_idx}: {vfh_waypoint}")

        recovery = getattr(self.vfh, '_recovery_phase', 0) != 0  # any non-NORMAL phase

        final_wp = vfh_waypoint if vfh_waypoint is not None else chosen_wp
        self.current_waypoint = final_wp
        self._publish_action_msgs(traj_batch, override_waypoint=final_wp)
        self._publish_viz_image(traj_batch, was_modified, selected_idx=chosen_idx,
                                final_wp=final_wp)


    def _publish_action_msgs(self, traj_batch: np.ndarray, override_waypoint: np.ndarray | None = None):
        sampled_actions_msg = Float32MultiArray()
        sampled_actions_msg.data = [0.0] + [float(x) for x in traj_batch.flatten()]
        self.sampled_actions_pub.publish(sampled_actions_msg)

        if override_waypoint is not None:
            chosen = override_waypoint
        else:
            chosen = traj_batch[0][self.args.waypoint]
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = [float(x) for x in chosen]
        self.waypoint_pub.publish(waypoint_msg)

    def _publish_viz_image(self, traj_batch: np.ndarray, vfh_active: bool,
                           selected_idx: int = 0, final_wp: np.ndarray | None = None):
        frame = np.array(self.context_queue[-1])
        img_h, img_w = frame.shape[:2]
        viz = frame.copy()

        cx = img_w // 2
        cy = int(img_h * 0.95)

        pixels_per_m = 3.0
        lateral_scale = 1.0
        robot_sym = 10

        def wp_to_px(wp):
            dx, dy = float(wp[0]), float(wp[1])
            return int(cx - dy * pixels_per_m * lateral_scale), int(cy - dx * pixels_per_m)

        cv2.line(viz, (cx - robot_sym, cy), (cx + robot_sym, cy), (255, 0, 0), 2)
        cv2.line(viz, (cx, cy - robot_sym), (cx, cy + robot_sym), (255, 0, 0), 2)

        selected_end = None
        for i, traj in enumerate(traj_batch):
            pts = [(cx, cy)]
            acc_x, acc_y = 0.0, 0.0
            for dx, dy in traj:
                acc_x += dx
                acc_y += dy
                px = int(cx - acc_y * pixels_per_m * lateral_scale)
                py = int(cy - acc_x * pixels_per_m)
                pts.append((px, py))
            if len(pts) >= 2:
                if i == selected_idx:
                    color, thick = (0, 255, 0), 3
                    selected_end = pts[-1]
                else:
                    color, thick = (140, 140, 140), 1
                cv2.polylines(viz, [np.array(pts, dtype=np.int32)], False, color, thick)

        if selected_end is not None:
            cv2.circle(viz, selected_end, 7, (0, 255, 0), -1)
            cv2.circle(viz, selected_end, 9, (0, 0, 0), 2)
            cv2.putText(viz, "NoMaD wp",
                        (selected_end[0] + 10, selected_end[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if vfh_active and final_wp is not None:
            fx, fy = wp_to_px(final_wp)
            cv2.arrowedLine(viz, (cx, cy), (fx, fy), (255, 0, 0), 3, tipLength=0.25)
            cv2.drawMarker(viz, (fx, fy), (255, 0, 0),
                           markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
            cv2.putText(viz, "VFH* wp", (fx + 10, fy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.rectangle(viz, (0, 0), (img_w, 26), (0, 0, 0), -1)
        cv2.putText(viz, "EXPLORE", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1)

        badge_text  = "VFH* OVERRIDE" if vfh_active else "NoMaD"
        badge_color = (255, 0, 0) if vfh_active else (0, 200, 0)
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bx = img_w - tw - 12
        cv2.rectangle(viz, (bx - 6, 3), (img_w - 3, 23), badge_color, -1)
        cv2.putText(viz, badge_text, (bx, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)

        ly = img_h - 8
        cv2.circle(viz, (10, ly - 4), 5, (0, 255, 0), -1)
        cv2.putText(viz, "NoMaD", (20, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0), 1)
        cv2.drawMarker(viz, (80, ly - 4), (255, 0, 0),
                       markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)
        cv2.putText(viz, "VFH*", (90, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 0, 0), 1)

        img_msg = self.bridge.cv2_to_imgmsg(viz, encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.viz_pub.publish(img_msg)


def main():
    parser = argparse.ArgumentParser("VFH* exploration (ROS 2)")
    parser.add_argument("--model", "-m", default="nomad")
    parser.add_argument("--waypoint", "-w", type=int, default=2)
    parser.add_argument("--num-samples", "-n", type=int, default=8)
    parser.add_argument(
        "--robot",
        type=str,
        default="turtlebot4",
        choices=["locobot", "locobot2", "robomaster", "turtlebot4"],
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="deployment/config",
        help="Path to config directory (default: deployment/config, use simulation/config for sim)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = VFHExplorationNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            out = node.metrics.save_all()
            node.metrics.print_summary()
            print(f"Metrics saved to: {out}")
        except Exception as e:
            print(f"Warning: could not save metrics: {e}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
