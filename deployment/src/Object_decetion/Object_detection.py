
import math
import pathlib
from typing import Optional

import numpy as np
import yaml



def load_yolo_model(model_path: str, device: str | None = None, offline: bool = True):

    import os
    import torch

    if offline:
        os.environ["YOLO_OFFLINE"] = "true"
    model_path = "/home/<USER>/yolov8n.pt"
    if not pathlib.Path(model_path).is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: '{model_path}'\n"
        )

    from ultralytics import YOLO

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)
    model.to(device)
    return model, device


def detect_objects_per_bin(
    image: np.ndarray,
    model,
    n_bins: int,
    fov_deg: float,
    conf: float = 0.5,
    classes: list[int] | None = None,
) -> np.ndarray:

    W = image.shape[1]
    bin_width_px = W / n_bins
    detection_vec = np.zeros(n_bins, dtype=np.float32)

    # Ultralytics YOLO expects BGR (OpenCV convention); image arrives as RGB.
    bgr = image[:, :, ::-1]

    results = model(
        bgr,
        conf=conf,
        classes=classes if classes else None,
        verbose=False,
    )

    if results and results[0].boxes is not None and len(results[0].boxes):
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs      = results[0].boxes.conf.cpu().numpy()
        cls_ids    = results[0].boxes.cls.cpu().numpy().astype(int)
        for (x1, _y1, x2, _y2), score, cid in zip(boxes_xyxy, confs, cls_ids):
            bin_lo = max(0,          int(x1 / bin_width_px))
            bin_hi = min(n_bins - 1, int(x2 / bin_width_px))
            detection_vec[bin_lo : bin_hi + 1] = 1.0
            print(
                f"YOLO detection: class={cid} conf={score:.2f}  "
                f"x=[{x1:.0f},{x2:.0f}px] → bins [{bin_lo},{bin_hi}]"
            )
    else:
        print("YOLO: no detections this frame")

    return detection_vec


def detect_objects_with_confidence(
    image: np.ndarray,
    model,
    num_bins: int,
    fov_deg: float,
    conf_threshold: float = 0.25,
    classes: list[int] | None = None,
) -> tuple[list[int], list[float], list[tuple[int, int]]]:
    W = image.shape[1]
    bin_width_px = W / num_bins

    bgr = image[:, :, ::-1]
    results = model(bgr, conf=conf_threshold, classes=classes, verbose=False)

    goal_bins: list[int] = []
    goal_confidences: list[float] = []
    goal_bin_ranges: list[tuple[int, int]] = []

    if results and results[0].boxes is not None and len(results[0].boxes):
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs       = results[0].boxes.conf.cpu().numpy()
        cls_ids     = results[0].boxes.cls.cpu().numpy().astype(int)

        for (x1, _y1, x2, _y2), score, cid in zip(boxes_xyxy, confs, cls_ids):
            u_center = (x1 + x2) / 2.0
            bin_idx  = int(u_center / bin_width_px)
            bin_idx  = max(0, min(bin_idx, num_bins - 1))
            bin_lo   = max(0,            int(x1 / bin_width_px))
            bin_hi   = min(num_bins - 1, int(x2 / bin_width_px))
            goal_bins.append(bin_idx)
            goal_confidences.append(float(score))
            goal_bin_ranges.append((bin_lo, bin_hi))
            print(
                f"[YOLO] class={cid} conf={score:.2f}  "
                f"x=[{x1:.0f},{x2:.0f}px]  center_bin={bin_idx}  "
                f"bin_range=[{bin_lo}..{bin_hi}]"
            )
    else:
        print("[YOLO] no detections this frame")

    return goal_bins, goal_confidences, goal_bin_ranges



def _load_config() -> dict:
    cfg_path = pathlib.Path(__file__).parents[1] / "config" / "robot.yml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)





#------------------------------Rviz node

try:
    from rclpy.node import Node as _ROSNode
    from rclpy.time import Time
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point, Vector3
    from std_msgs.msg import ColorRGBA
    from builtin_interfaces.msg import Duration
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


class ObjectMarkerPublisher:

    def __init__(
        self,
        node,
        topic: str = "/vfh/detected_objects_markers",
        num_bins: int = 72,
        fov_deg: float = 90.0,
        frame_id: str = "base_link",
    ) -> None:
        self._node = node
        self._pub = node.create_publisher(MarkerArray, topic, 10)
        self.num_bins = num_bins
        self.fov_rad = math.radians(fov_deg)
        self.frame_id = frame_id

    def publish(
        self,
        yolo_vector: np.ndarray,
        stamp: Optional[Time] = None,
        selected_bin: Optional[int] = None,
        detected_bin: Optional[int] = None,
    ) -> None:
        if stamp is None:
            stamp = self._node.get_clock().now()

        ma = MarkerArray()
        ts = stamp.to_msg()

        half_fov = self.fov_rad / 2.0
        bin_width = self.fov_rad / self.num_bins
        lifetime = Duration(sec=0, nanosec=int(0.5e9))  # 500 ms

        for i in range(self.num_bins):
            angle = half_fov - (i + 0.5) * bin_width

            detec = yolo_vector[i]
            is_obj = detec < 0.5  
            plot_dist = 1

            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = ts
            m.ns = "detection_vector"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.lifetime = lifetime

            start = Point(x=0.0, y=0.0, z=0.1)  
            end = Point(
                x=plot_dist * math.cos(angle),
                y=plot_dist * math.sin(angle),
                z=0.1,
            )
            m.points = [start, end]


            m.scale = Vector3(x=0.02, y=0.04, z=0.04)

            if i == selected_bin:
                color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)  # cyan
            elif i == detected_bin:
                color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)  # magenta
            elif is_obj:
                color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.2)  # grey 
            else:
                color = self._distance_color()
            m.color = color

            ma.markers.append(m)
        self._pub.publish(ma)

    def _distance_color(self) -> ColorRGBA:
        s = 1
        r = 1.0 - s
        g = 1.0

        return ColorRGBA(r=r, g=g, b=0.0, a=0.9)
