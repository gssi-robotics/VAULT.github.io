
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Deque, List, Optional, Tuple

_DA2_METRIC = str(Path(__file__).resolve().parents[3] / "Depth-Anything-V2" / "metric_depth")
if _DA2_METRIC not in sys.path:
    sys.path.insert(0, _DA2_METRIC)

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from depth_anything_v2.dpt import DepthAnythingV2

from VfhPlus import defaults as D
from vfhstar_nav import VFHStar
from VfhPlus.depth_processing import (
    compute_distance_vector, TemporalAggregator, pad_distance_vector,
)
from VfhPlus.nomad_vector import waypoint_to_reference, generate_direction_waypoints
from VfhPlus.depth_markers import DepthMarkerPublisher, BinRayMarkerPublisher

from Object_decetion.Object_detection import (
    load_yolo_model, detect_objects_with_confidence,
)

THIS_DIR = Path.cwd()


def _parse_config_dir() -> Path:
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--config-dir", type=str, default="deployment/config")
    _args, _ = _p.parse_known_args()
    cd = Path(_args.config_dir)
    if not cd.is_absolute():
        cd = THIS_DIR / cd
    return cd


_CONFIG_DIR       = _parse_config_dir()
ROBOT_CONFIG_PATH = _CONFIG_DIR / "robot.yaml"
MODEL_CONFIG_PATH = THIS_DIR / "deployment/config/models.yaml"
VFH_CONFIG_PATH   = _CONFIG_DIR / "vfh.yaml"
NAV_CONFIG_PATH   = _CONFIG_DIR / "vfh_navigation.yaml"

with open(ROBOT_CONFIG_PATH) as f:
    ROBOT_CONF = yaml.safe_load(f)
MAX_V = ROBOT_CONF["max_v"]
MAX_W = ROBOT_CONF["max_w"]
RATE  = ROBOT_CONF["frame_rate"]

with open(VFH_CONFIG_PATH) as f:
    VFH_CONF = yaml.safe_load(f)

if NAV_CONFIG_PATH.is_file():
    with open(NAV_CONFIG_PATH) as f:
        NAV_CONF = yaml.safe_load(f) or {}
else:
    NAV_CONF = {}

INFERENCE_HZ     = float(NAV_CONF.get("inference_rate_hz",   RATE))
CONTROL_HZ       = float(NAV_CONF.get("control_rate_hz",     RATE))
WATCHDOG_S       = float(NAV_CONF.get("watchdog_timeout_s",  1.0))
NUM_EXEC_THREADS = int(NAV_CONF.get("num_executor_threads",  4))



class NavState(Enum):
    EXPLORE  = "EXPLORE"   
    NAV_GOAL = "NAV_GOAL"  
    REACHED  = "REACHED"   



def _load_nomad_model(model_name: str, device: torch.device):
    with open(MODEL_CONFIG_PATH) as f:
        model_paths = yaml.safe_load(f)
    model_config_path = model_paths[model_name]["config_path"]
    with open(model_config_path) as f:
        model_params = yaml.safe_load(f)
    ckpt_path = model_paths[model_name]["ckpt_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found: {ckpt_path}")
    model = load_model(ckpt_path, model_params, device)
    return model.to(device).eval(), model_params


class NavigationVFHNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("navigation_vfh")
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Device: {self.device}")

        self.model, self.model_params = _load_nomad_model(args.model, self.device)
        self.context_size: int = self.model_params["context_size"]
        self.last_ctx_time = self.get_clock().now()
        self.ctx_dt = 0.1
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.context_queue: Deque = deque(maxlen=self.context_size + 1)
        self.bridge = CvBridge()

        if args.robot == "turtlebot4":
            image_topic           = "/robot2/oakd/rgb/preview/image_raw"
            waypoint_topic        = "/robot2/waypoint"
            sampled_actions_topic = "/robot2/sampled_actions"
            trajectory_viz_topic  = "/robot2/trajectory_viz"
            self.DIM = (320, 200)
        elif args.robot == "locobot":
            image_topic           = "/robot1/camera/image"
            waypoint_topic        = "/robot1/waypoint"
            sampled_actions_topic = "/robot1/sampled_actions"
            trajectory_viz_topic  = "/robot1/trajectory_viz"
            self.DIM = (320, 240)
        else:
            raise ValueError(f"Unknown robot: {args.robot}")

        self.vfh_num_bins      = VFH_CONF.get("num_bins",          D.NUM_BINS)
        self.vfh_fov_deg       = VFH_CONF.get("fov_deg",           D.FOV_DEG)
        self.vfh_max_range     = VFH_CONF.get("max_sensing_range",  D.MAX_RANGE)
        self.vfh_v_margin      = VFH_CONF.get("vertical_margin",    D.VERTICAL_MARGIN)
        self.vfh_floor_margin  = VFH_CONF.get("floor_margin",       D.FLOOR_MARGIN)
        self.vfh_safety_margin = VFH_CONF.get("safety_margin",      D.SAFETY_MARGIN)
        self.vfh_depth_scale   = VFH_CONF.get("depth_scale",        D.DEPTH_SCALE)
        self.vfh_speed_red     = VFH_CONF.get("speed_reduction",    D.SPEED_REDUCTION)
        self.vfh_num_wps       = VFH_CONF.get("num_vfh_waypoints",  D.NUM_VFH_WAYPOINTS)
        self.vfh_wp_idx        = VFH_CONF.get("vfh_waypoint_index", D.VFH_WAYPOINT_INDEX)
        self.fov_padding_bins  = VFH_CONF.get("fov_padding_bins",   D.FOV_PADDING_BINS)

        self.vfh_total_bins  = self.vfh_num_bins + 2 * self.fov_padding_bins
        bin_width_deg        = self.vfh_fov_deg / self.vfh_num_bins
        self.vfh_virtual_fov = bin_width_deg * self.vfh_total_bins

        self.vfh = VFHStar(
            num_bins                = self.vfh_total_bins,
            fov_deg                 = self.vfh_virtual_fov,
            safety_threshold        = VFH_CONF.get("safety_threshold",        D.SAFETY_THRESHOLD),
            s_max                   = VFH_CONF.get("s_max",                    D.S_MAX),
            mu1                     = VFH_CONF.get("mu1",                      D.MU1),
            mu2                     = VFH_CONF.get("mu2",                      D.MU2),
            mu3                     = VFH_CONF.get("mu3",                      D.MU3),
            robot_radius            = VFH_CONF.get("robot_radius",             D.ROBOT_RADIUS),
            recovery_reverse_cycles = VFH_CONF.get("recovery_reverse_cycles", D.RECOVERY_REVERSE_CYCLES),
            recovery_turn_cycles    = VFH_CONF.get("recovery_turn_cycles",     D.RECOVERY_TURN_CYCLES),
            fov_padding_bins        = self.fov_padding_bins,
        )

        intrinsics_path = self._intrinsics_path()
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")
        self.K = np.load(intrinsics_path)

        da2_cfg = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        enc     = VFH_CONF.get("depth_encoder", "vits")
        weights = VFH_CONF.get("depth_weights", "")
        if not os.path.isabs(weights):
            weights = str(THIS_DIR / weights)
        self.depth_model = DepthAnythingV2(
            **{**da2_cfg[enc], "max_depth": VFH_CONF.get("depth_max_depth", 20)}
        )
        self.depth_model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)
        )
        self.depth_model = self.depth_model.to(self.device).eval()

        yolo_weights = args.yolo_weights
        if not os.path.isabs(yolo_weights):
            yolo_weights = str(THIS_DIR / yolo_weights)
        self.yolo_model, _ = load_yolo_model(yolo_weights, device=str(self.device))
        self.yolo_conf_threshold = args.yolo_conf
        self.yolo_classes        = args.yolo_classes  

        self.distance_vector = pad_distance_vector(
            np.full(self.vfh_num_bins, np.inf),
            padding_bins=self.fov_padding_bins,
        )
        self.temporal_agg = TemporalAggregator(
            num_bins        = self.vfh_num_bins,
            window_size     = VFH_CONF.get("temporal_window",  D.TEMPORAL_WINDOW),
            danger_threshold= VFH_CONF.get("safety_threshold", D.SAFETY_THRESHOLD),
        )
        self.current_waypoint    = np.zeros(2)
        self._new_depth_available = False

        self._state_lock = threading.Lock()
        self._wp_lock    = threading.Lock()
        self._last_waypoint_time = None      
        self.watchdog_timeout_s  = WATCHDOG_S
        self._goal_bins:             List[int]             = []
        self._goal_confs:            List[float]           = []

        self._goal_bin_ranges:       List[Tuple[int, int]] = []
        self._goal_stale:            int  = 0
        self._goal_max_stale:        int  = args.goal_stale_frames
        self._goal_seen_last_image:  bool = False

        self.state: NavState = NavState.EXPLORE
        self.goal_timeout_frames: int  = args.goal_timeout_frames

        self.goal_reach_distance: float = args.goal_reach_distance

        self.depth_marker_pub = DepthMarkerPublisher(
            node             = self,
            topic            = "/vfh/depth_markers",
            num_bins         = self.vfh_total_bins,
            fov_deg          = self.vfh_virtual_fov,
            max_range        = self.vfh_max_range,
            safety_threshold = VFH_CONF.get("safety_threshold", D.SAFETY_THRESHOLD),
        )


        self.detected_ray_pub = BinRayMarkerPublisher(
            node      = self,
            topic     = "/vfh/detected_objects_ray",
            num_bins  = self.vfh_total_bins,
            fov_deg   = self.vfh_virtual_fov,
            color     = (1.0, 1.0, 1.0, 1.0),
            marker_ns = "detected_objects",
        )
        self.goal_ray_pub = BinRayMarkerPublisher(
            node      = self,
            topic     = "/vfh/goal_reference_bins",
            num_bins  = self.vfh_total_bins,
            fov_deg   = self.vfh_virtual_fov,
            color     = (1.0, 0.45, 0.0, 1.0),
            marker_ns = "goal_refs",
        )
        self.chosen_ray_pub = BinRayMarkerPublisher(
            node       = self,
            topic      = "/vfh/chosen_bin",
            num_bins   = self.vfh_total_bins,
            fov_deg    = self.vfh_virtual_fov,
            color      = (0.0, 0.2, 1.0 , 1.0),
            marker_ns  = "chosen_bin",
            point_size = 0.09,
        )





        self._image_group     = MutuallyExclusiveCallbackGroup()
        self._inference_group = MutuallyExclusiveCallbackGroup()
        self._control_group   = MutuallyExclusiveCallbackGroup()

        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Image, image_topic, self._image_cb, img_qos,
            callback_group=self._image_group,
        )
        self.waypoint_pub        = self.create_publisher(Float32MultiArray, waypoint_topic,        1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, sampled_actions_topic, 1)
        self.viz_pub             = self.create_publisher(Image, trajectory_viz_topic,              1)
        self.reached_goal_pub    = self.create_publisher(Bool, "/topoplan/reached_goal",           1)

        self.create_timer(
            1.0 / INFERENCE_HZ, self._inference_cb,
            callback_group=self._inference_group,
        )

        self.create_timer(
            1.0 / CONTROL_HZ, self._control_cb,
            callback_group=self._control_group,
        )
        self.get_logger().info(
            f"NavigationVFHNode ready — robot={args.robot}, "
            f"state={self.state.value}, "
            f"goal_timeout={self.goal_timeout_frames} frames, "
            f"inference={INFERENCE_HZ:.1f}Hz, control={CONTROL_HZ:.1f}Hz, "
            f"watchdog={self.watchdog_timeout_s:.2f}s"
        )

    def _intrinsics_path(self) -> str:
        p = ROBOT_CONF.get("intrinsics_path", "")
        if not p:
            p = f"intrinsic/{self.args.robot}/intrinsics.npy"
        if not os.path.isabs(p):
            p = str(THIS_DIR / p)
        return p

    def _publish_stop(self) -> None:
        wp_msg = Float32MultiArray()
        wp_msg.data = [0.0, 0.0]
        self.waypoint_pub.publish(wp_msg)
        with self._wp_lock:
            self.current_waypoint    = np.zeros(2)
            self._last_waypoint_time = self.get_clock().now()
        self.reached_goal_pub.publish(Bool(data=True))


    def _image_cb(self, msg: Image) -> None:
        now = self.get_clock().now()
        if (now - self.last_ctx_time).nanoseconds < self.ctx_dt * 1e9:
            return
        pil_frame = msg_to_pil(msg)

        frame = self.bridge.imgmsg_to_cv2(msg)
        enc = msg.encoding.lower().replace("-", "")
        if enc in ("rgb8", "rgb"):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif enc in ("bgr8", "bgr"):
            bgr = frame
        else:
            bgr = frame  # best-effort; DA2 and YOLO assume BGR

        with torch.no_grad():
            depth_map = self.depth_model.infer_image(bgr)

        raw_dv = compute_distance_vector(
            depth_map, self.K,
            num_bins        = self.vfh_num_bins,
            fov_deg         = self.vfh_fov_deg,
            max_range       = self.vfh_max_range,
            vertical_margin = self.vfh_v_margin,
            floor_margin    = self.vfh_floor_margin,
            safety_margin   = self.vfh_safety_margin,
            depth_scale     = self.vfh_depth_scale,
        )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_bins, goal_confs, raw_ranges = detect_objects_with_confidence(
            rgb,
            self.yolo_model,
            num_bins       = self.vfh_num_bins,
            fov_deg        = self.vfh_fov_deg,
            conf_threshold = self.yolo_conf_threshold,
            classes        = self.yolo_classes,
        )
        goal_bins = [b + self.fov_padding_bins for b in raw_bins]
        goal_bin_ranges = [
            (lo + self.fov_padding_bins, hi + self.fov_padding_bins)
            for (lo, hi) in raw_ranges
        ]
        if raw_bins:
            self.get_logger().info(
                f"[DBG/YOLO] raw_bins={raw_bins} (real-FOV space) "
                f"→ padded_bins={goal_bins}  "
                f"padded_ranges={goal_bin_ranges} "
                f"(shift=+{self.fov_padding_bins}, valid range "
                f"{self.fov_padding_bins}..{self.fov_padding_bins + self.vfh_num_bins - 1})"
            )


        with self._state_lock:
            self.context_queue.append(pil_frame)
            self.last_ctx_time = now

            smoothed = self.temporal_agg.update(raw_dv)
            self.distance_vector      = pad_distance_vector(smoothed, self.fov_padding_bins)
            self._new_depth_available = True

            if goal_bins:
                self._goal_bins             = goal_bins
                self._goal_confs            = goal_confs
                self._goal_bin_ranges       = goal_bin_ranges
                self._goal_stale            = 0
                self._goal_seen_last_image  = True
            else:
                self._goal_stale           += 1
                self._goal_seen_last_image  = False
                if self._goal_stale >= self._goal_max_stale and self.state != NavState.NAV_GOAL:
                    self._goal_bins        = []
                    self._goal_confs       = []
                    self._goal_bin_ranges  = []

        if goal_bins:
            self.get_logger().info(
                f"[YOLO] {len(goal_bins)} object(s) → "
                f"bins={goal_bins}, ranges={goal_bin_ranges}, "
                f"confs={[f'{c:.2f}' for c in goal_confs]}"
            )


    def _inference_cb(self) -> None:
        if self.state == NavState.REACHED:
            self._publish_stop()
            return


        with self._state_lock:
            if len(self.context_queue) <= self.context_size:
                return
            if not self._new_depth_available:
                return
            self._new_depth_available = False

            distance_vector = self.distance_vector.copy()
            ctx_snapshot    = list(self.context_queue)
            gbins_snap      = list(self._goal_bins)
            gconfs_snap     = list(self._goal_confs)
            granges_snap    = list(self._goal_bin_ranges)
            goal_stale_snap = self._goal_stale

        goal_bins  = gbins_snap  if gbins_snap else None
        goal_confs = gconfs_snap if gbins_snap else None

        prev_state = self.state

        if self.state == NavState.EXPLORE and goal_bins:
            self.state = NavState.NAV_GOAL
            self.get_logger().info(
                f"[NavSM] EXPLORE → NAV_GOAL  (detected bins={goal_bins})"
            )

        if self.state == NavState.NAV_GOAL:
            if goal_stale_snap >= self.goal_timeout_frames:
                self.state = NavState.EXPLORE
                with self._state_lock:
                    self._goal_bins       = []
                    self._goal_confs      = []
                    self._goal_bin_ranges = []
                    self._goal_stale      = 0
                goal_bins  = None
                goal_confs = None
                self.get_logger().info(
                    f"[NavSM] NAV_GOAL → EXPLORE  "
                    f"(goal lost for {self.goal_timeout_frames} frames)"
                )
            elif goal_bins:
                valid_lo = self.fov_padding_bins
                valid_hi = self.fov_padding_bins + self.vfh_num_bins - 1
                for gb in goal_bins:
                    if gb < valid_lo or gb > valid_hi:
                        self.get_logger().warn(
                            f"[NavSM] goal bin={gb} is padding "
                            f"(valid FOV: {valid_lo}–{valid_hi}) — skipping REACHED check"
                        )
                        continue
                    actual_dist = distance_vector[gb]
                    self.get_logger().info(
                        f"[NavSM] NAV_GOAL  bin={gb}  dist={actual_dist:.2f} m  "
                        f"goal_reach={self.goal_reach_distance:.2f} m"
                    )
                    if actual_dist < self.goal_reach_distance:
                        self.state = NavState.REACHED
                        self.get_logger().info(
                            f"[NavSM] NAV_GOAL → REACHED  "
                            f"(bin={gb}, dist={actual_dist:.2f} m "
                            f"< goal_reach={self.goal_reach_distance:.2f} m)"
                        )
                        self._publish_stop()
                        return

        if prev_state != self.state:
            with self._state_lock:
                self.temporal_agg.reset()

        obs_imgs = transform_images(
            ctx_snapshot, self.model_params["image_size"], center_crop=False
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
            rep = (
                (lambda x: x.repeat(self.args.num_samples, 1))
                if obs_cond.ndim == 2
                else (lambda x: x.repeat(self.args.num_samples, 1, 1))
            )
            obs_cond = rep(obs_cond)

            naction = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2),
                device=self.device,
            )
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(noise_pred, k, naction).prev_sample

        traj_batch = to_numpy(get_action(naction))   # (S, L, 2)

        all_ref_bins: List[int] = []
        all_ref_wps             = []
        for i in range(len(traj_batch)):
            wp = traj_batch[i][self.args.waypoint]
            _, rb = waypoint_to_reference(
                float(wp[0]), float(wp[1]),
                num_bins=self.vfh_total_bins,
                fov_deg=self.vfh_virtual_fov,
            )
            all_ref_bins.append(rb)
            all_ref_wps.append(wp)

        valid_lo = self.fov_padding_bins
        valid_hi = self.fov_padding_bins + self.vfh_num_bins - 1

        if self.state == NavState.NAV_GOAL and goal_bins:
            clamped_goal_bins = [max(valid_lo, min(gb, valid_hi)) for gb in goal_bins]
            reference_bins = clamped_goal_bins

            dist_for_vfh = distance_vector.copy()

            if (granges_snap
                    and len(granges_snap) == len(goal_bins)):
                span_src = list(zip(goal_bins, granges_snap))
            else:
                span_src = [(c, (c, c)) for c in goal_bins]

            mask_info = []
            for center, (lo_raw, hi_raw) in span_src:
                center_c = max(valid_lo, min(center, valid_hi))
                span_lo  = max(valid_lo, min(lo_raw, valid_hi))
                span_hi  = max(valid_lo, min(hi_raw, valid_hi))

                d = float(distance_vector[center_c])
                if not np.isfinite(d) or d <= 0.1:
                    d = max(self.vfh.safety_threshold, 0.5)
                margin_angle = math.atan2(self.vfh.robot_radius, d)
                margin_bins  = max(1, math.ceil(margin_angle / self.vfh.bin_width))
                k = margin_bins + self.args.goal_mask_extra

                mask_lo = max(valid_lo, span_lo - k)
                mask_hi = min(valid_hi, span_hi + k)
                dist_for_vfh[mask_lo:mask_hi + 1] = np.inf
                mask_info.append((center_c, span_lo, span_hi, mask_lo, mask_hi, k))

            self.get_logger().info(
                "[NavSM] NAV_GOAL mask: "
                + ", ".join(
                    f"ref={c} span=[{slo}..{shi}] mask=[{mlo}..{mhi}] (±{k})"
                    for c, slo, shi, mlo, mhi, k in mask_info
                )
                + f"  refs={reference_bins}"
            )
        else:
            reference_bins = all_ref_bins
            dist_for_vfh = distance_vector

        best_bin, best_angle, was_modified = self.vfh.compute(
            dist_for_vfh, reference_bins
        )

        if best_bin in all_ref_bins:
            chosen_idx = all_ref_bins.index(best_bin)
        else:
            chosen_idx = int(np.argmin([abs(b - best_bin) for b in all_ref_bins]))
        chosen_nomad_wp = (
            all_ref_wps[chosen_idx] if self.state == NavState.EXPLORE else None
        )

        if self.vfh.recovery_just_completed:
            self.temporal_agg.reset()
            self.vfh.recovery_just_completed = False
            self.get_logger().info("[NavVFH*] Flushed temporal aggregator after recovery")

        ref_for_viz = (
            reference_bins[0]
            if self.state == NavState.NAV_GOAL and goal_bins
            else all_ref_bins[chosen_idx]
        )
        self.depth_marker_pub.publish(
            distance_vector,
            selected_bin  = best_bin,
            reference_bin = ref_for_viz,
        )

        self.detected_ray_pub.publish(gbins_snap or [])
        if self.state == NavState.NAV_GOAL and goal_bins:
            self.goal_ray_pub.publish(reference_bins)
        else:
            self.goal_ray_pub.publish([])
        self.chosen_ray_pub.publish([best_bin])

        self.get_logger().info(
            f"[DBG] state={self.state.value}  "
            f"cached_goal_bins={gbins_snap}  "
            f"yolo_hit_last_frame={self._goal_seen_last_image}  "
            f"nomad_bins={all_ref_bins}  "
            f"ref_bins→VFH*={reference_bins}  "
            f"best_bin={best_bin}  was_modified={was_modified}  "
            f"goal_stale={goal_stale_snap}/"
            f"(mem_clear={self._goal_max_stale}, "
            f"timeout={self.goal_timeout_frames})"
        )

        final_wp = self._make_waypoint(
            best_bin, best_angle, was_modified, chosen_nomad_wp,
        )
        with self._wp_lock:
            self.current_waypoint    = final_wp
            self._last_waypoint_time = self.get_clock().now()

        self.get_logger().info(
            f"[NavVFH*] state={self.state.value}  modified={was_modified}  "
            f"bin={best_bin}  angle={math.degrees(best_angle):.1f}°  "
            f"wp={final_wp[:2]}"
        )

        self._publish(traj_batch, final_wp, was_modified, chosen_idx)


    def _control_cb(self) -> None:
        now = self.get_clock().now()
        with self._wp_lock:
            wp     = np.asarray(self.current_waypoint, dtype=float).copy()
            last_t = self._last_waypoint_time

        if last_t is None:
            return

        age_s = (now - last_t).nanoseconds * 1e-9
        if age_s > self.watchdog_timeout_s:
            msg = Float32MultiArray()
            msg.data = [0.0, 0.0]
            self.waypoint_pub.publish(msg)
            return

        if np.linalg.norm(wp[:2]) > 1e-3 or len(wp) == 4:
            msg = Float32MultiArray()
            msg.data = [float(x) for x in wp]
            self.waypoint_pub.publish(msg)


    def _make_waypoint(
        self,
        best_bin: int,
        best_angle: float,
        was_modified: bool,
        chosen_nomad_wp: Optional[np.ndarray],
    ) -> np.ndarray:
        if was_modified and self.vfh._recovery_phase == self.vfh._PHASE_TURN:
            hx = math.cos(best_angle)
            hy = math.sin(best_angle)
            self.get_logger().info(
                f"[NavVFH*] TURN phase: rotating toward {math.degrees(best_angle):.1f}°"
            )
            return np.array([0.0, 0.0, hx, hy])

        if self.state == NavState.EXPLORE and not was_modified:
            return chosen_nomad_wp
        if self.state == NavState.NAV_GOAL:
            magnitude = MAX_V
            wp_idx   = self.args.nav_goal_waypoint_index
        else:
            magnitude = np.linalg.norm(chosen_nomad_wp)
            if magnitude < 1e-3:
                magnitude = MAX_V
            wp_idx = self.vfh_wp_idx
        wps = generate_direction_waypoints(
            best_angle,
            max_magnitude = magnitude * self.vfh_speed_red,
            num_waypoints = self.vfh_num_wps,
        )
        chosen = wps[min(wp_idx, len(wps) - 1)]
        self.get_logger().info(
            f"[NavVFH*] wp_idx={wp_idx}/{len(wps)-1}  "
            f"magnitude={magnitude * self.vfh_speed_red:.4f} m  "
            f"chosen_mag={float(np.linalg.norm(chosen)):.4f} m"
        )
        return chosen


    def _publish(
        self,
        traj_batch: np.ndarray,
        final_wp: np.ndarray,
        vfh_active: bool,
        selected_idx: int,
    ) -> None:
        sa_msg = Float32MultiArray()
        sa_msg.data = [0.0] + [float(x) for x in traj_batch.flatten()]
        self.sampled_actions_pub.publish(sa_msg)

        wp_msg = Float32MultiArray()
        wp_msg.data = [float(x) for x in final_wp]
        self.waypoint_pub.publish(wp_msg)
        self.get_logger().info(
            f"[DBG/WP] PUBLISH state={self.state.value} "
            f"final_wp={[round(float(x), 3) for x in final_wp]} "
            f"(len={len(final_wp)})"
        )


        self._publish_viz(traj_batch, vfh_active, selected_idx, final_wp)

    def _publish_viz(
        self,
        traj_batch: np.ndarray,
        vfh_active: bool,
        selected_idx: int,
        final_wp: np.ndarray,
    ) -> None:
        frame        = np.array(self.context_queue[-1])
        img_h, img_w = frame.shape[:2]
        viz          = frame.copy()
        cx, cy       = img_w // 2, int(img_h * 0.95)
        ppm          = 3.0

        def wp_to_px(wp):
            dx, dy = float(wp[0]), float(wp[1])
            return int(cx - dy * ppm), int(cy - dx * ppm)

        cv2.line(viz, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 2)
        cv2.line(viz, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 2)

        selected_end = None
        for i, traj in enumerate(traj_batch):
            pts = [(cx, cy)]
            ax, ay = 0.0, 0.0
            for dx, dy in traj:
                ax += dx; ay += dy
                pts.append((int(cx - ay * ppm), int(cy - ax * ppm)))
            if len(pts) >= 2:
                if i == selected_idx:
                    color, thick = (0, 255, 0), 3
                    selected_end = pts[-1]
                else:
                    color, thick = (140, 140, 140), 1
                cv2.polylines(viz, [np.array(pts, np.int32)], False, color, thick)

        if selected_end is not None:
            cv2.circle(viz, selected_end, 7, (0, 255, 0), -1)
            cv2.circle(viz, selected_end, 9, (0, 0, 0), 2)
            cv2.putText(viz, "NoMaD wp",
                        (selected_end[0] + 10, selected_end[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if vfh_active:
            fx, fy = wp_to_px(final_wp)
            cv2.arrowedLine(viz, (cx, cy), (fx, fy), (255, 0, 0), 3, tipLength=0.25)
            cv2.drawMarker(viz, (fx, fy), (255, 0, 0),
                           markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
            cv2.putText(viz, "VFH* wp", (fx + 10, fy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        with self._state_lock:
            goal_bins_viz  = list(self._goal_bins)
            goal_confs_viz = list(self._goal_confs)
            goal_stale_viz = self._goal_stale
        if goal_bins_viz:
            fov_rad   = math.radians(self.vfh_virtual_fov)
            bin_width = fov_rad / self.vfh_total_bins
            for gb, gc in zip(goal_bins_viz, goal_confs_viz):
                angle  = fov_rad / 2 - (gb + 0.5) * bin_width
                length = int(60 * gc)
                ex = int(cx - math.sin(angle) * length)
                ey = int(cy - math.cos(angle) * length)
                cv2.arrowedLine(viz, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)

        cv2.rectangle(viz, (0, 0), (img_w, 26), (0, 0, 0), -1)
        state_colors = {
            NavState.EXPLORE:  (220, 220, 220),
            NavState.NAV_GOAL: (0, 255, 255),
            NavState.REACHED:  (0, 255, 0),
        }
        label = f"NAV:{self.state.value}"
        if self.state == NavState.NAV_GOAL and goal_stale_viz > 0:
            label += f" (lost {goal_stale_viz}/{self.goal_timeout_frames})"
        cv2.putText(viz, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    state_colors[self.state], 1)

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
        cv2.arrowedLine(viz, (140, ly - 4), (158, ly - 4), (0, 255, 255), 2, tipLength=0.4)
        cv2.putText(viz, "goal", (162, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1)

        img_msg = self.bridge.cv2_to_imgmsg(viz, encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.viz_pub.publish(img_msg)



def main():
    parser = argparse.ArgumentParser("NavigationVFH — goal-oriented object navigation")
    parser.add_argument("--model",       "-m", default="nomad")
    parser.add_argument("--waypoint",    "-w", type=int, default=2)
    parser.add_argument("--num-samples", "-n", type=int, default=8)
    parser.add_argument("--robot",       type=str, default="turtlebot4",
                        choices=["locobot", "turtlebot4"])
    parser.add_argument("--config-dir",  type=str, default="deployment/config")
    parser.add_argument("--yolo-weights",  type=str, required=True,
                        help="Path to YOLO .pt weights file")
    parser.add_argument("--yolo-conf",     type=float, default=0.6,
                        help="YOLO confidence threshold (default: 0.6)")
    parser.add_argument("--yolo-classes",  type=int, nargs="*", default=None,
                        help="YOLO class IDs to detect (default: all classes)")
    parser.add_argument("--goal-timeout-frames", type=int, default=7,
                        help="Timer ticks without detection before reverting to EXPLORE "
                             "(default: 3)")
    parser.add_argument("--goal-reach-distance", type=float, default=0.000001,
                        help="Depth (m) at which the goal object is considered reached "
                             "(default: 0.5); must be smaller than VFH* safety_threshold")
    parser.add_argument("--goal-stale-frames", type=int, default=5,
                        help="Consecutive image frames without detection before clearing "
                             "goal bins (default: 3); prevents race-condition blanking")
    parser.add_argument("--goal-mask-extra", type=int, default=4,
                        help="Extra bins (beyond the adaptive robot-body margin) masked "
                             "as clear around each goal bin to keep VFH* from rejecting "
                             "the goal direction on clearance checks (default: 2)")
    parser.add_argument("--nav-goal-waypoint-index", type=int, default=0,
                        help="Waypoint index used in NAV_GOAL state "
                             "(0 = closest, vfh_num_waypoints-1 = farthest). "
                             "Lower values give shorter lookahead so the robot "
                             "re-steers toward the object on each inference "
                             "instead of committing to a deeper point (default: 0)")
    args = parser.parse_args()

    rclpy.init()
    node = NavigationVFHNode(args)
    executor = MultiThreadedExecutor(num_threads=NUM_EXEC_THREADS)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
