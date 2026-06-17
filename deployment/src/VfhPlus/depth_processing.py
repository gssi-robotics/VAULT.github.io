
from __future__ import annotations

import math
from collections import deque

import numpy as np

from VfhPlus.defaults import (
    NUM_BINS, FOV_DEG, MAX_RANGE, VERTICAL_MARGIN,
    FLOOR_MARGIN, SAFETY_MARGIN, DEPTH_SCALE, TEMPORAL_WINDOW,
    SAFETY_THRESHOLD, FOV_PADDING_BINS,
)

def pad_distance_vector(
    distance_vector: np.ndarray,
    padding_bins: int = FOV_PADDING_BINS,
    fill_value: float = 0.0,
) -> np.ndarray:
    if padding_bins <= 0:
        return distance_vector
    pad = np.full(padding_bins, fill_value, dtype=distance_vector.dtype)
    print(f"Padding distance vector with {padding_bins} bins on each side, fill_value={fill_value}")
    return np.concatenate([pad, distance_vector, pad])


def compute_distance_vector(
    depth_map: np.ndarray,
    K: np.ndarray,
    *,
    num_bins: int = NUM_BINS,
    fov_deg: float = FOV_DEG,
    max_range: float = MAX_RANGE,
    vertical_margin: float = VERTICAL_MARGIN,
    floor_margin: float = FLOOR_MARGIN,
    safety_margin: float = SAFETY_MARGIN,
    depth_scale: float = DEPTH_SCALE,
) -> np.ndarray:
    fov_rad = math.radians(fov_deg)
    half_fov = fov_rad / 2.0
    bin_width = fov_rad / num_bins

    H, W = depth_map.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    scaled_depth = depth_map.astype(np.float64) * depth_scale

    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    Z = scaled_depth.ravel()
    X = ((uu.ravel() - cx) * Z) / fx   
    Y = ((vv.ravel() - cy) * Z) / fy  

    mask = (Z > 0) & (Z <= max_range) & (Y >= -vertical_margin) & (Y <= floor_margin)
    n_total = int((Z > 0).sum())
    n_kept = int(mask.sum())
    n_floor = int(((Z > 0) & (Y > floor_margin)).sum())
    print(f"[depth] points: total={n_total}  kept={n_kept}  floor_excluded={n_floor}")
    X_f = X[mask]
    Z_f = Z[mask]

    distance_vector = np.full(num_bins, np.inf, dtype=np.float64)

    if len(X_f) == 0:
        return distance_vector

    angles = np.arctan2(-X_f, Z_f)

    actual_ranges = np.sqrt(X_f ** 2 + Z_f ** 2)
    ranges = np.maximum(actual_ranges - safety_margin, 1e-3)

    bin_indices = ((half_fov - angles) / bin_width).astype(int)
    valid = (bin_indices >= 0) & (bin_indices < num_bins)
    bin_indices = bin_indices[valid]
    ranges = ranges[valid]

    np.minimum.at(distance_vector, bin_indices, ranges)
    print(f"This is distance vector {distance_vector}")

    return distance_vector


class TemporalAggregator:

    def __init__(
        self,
        num_bins: int = NUM_BINS,
        window_size: int = TEMPORAL_WINDOW,
        danger_threshold: float = SAFETY_THRESHOLD,
    ) -> None:
        self.num_bins = num_bins
        self.window_size = window_size
        self.danger_threshold = danger_threshold
        self._history: deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, distance_vector: np.ndarray) -> np.ndarray:
        self._history.append(distance_vector.copy())
        return self.get_filtered()

    def get_filtered(self) -> np.ndarray:
        if not self._history:
            return np.full(self.num_bins, np.inf, dtype=np.float64)

        stacked = np.array(list(self._history))  
        result = np.min(stacked, axis=0)  
        return result

    @property
    def certainty(self) -> np.ndarray:
        if not self._history:
            return np.zeros(self.num_bins, dtype=np.float64)
        stacked = np.array(list(self._history))
        blocked = stacked < self.danger_threshold  
        frac_blocked = blocked.min(axis=0)
        return np.abs(4.0 * frac_blocked - 1.0)

    @property
    def danger(self) -> np.ndarray:
        if not self._history:
            return np.zeros(self.num_bins, dtype=np.float64)
        stacked = np.array(list(self._history))
        return (stacked < self.danger_threshold).mean(axis=0)

    def reset(self) -> None:
        self._history.clear()
