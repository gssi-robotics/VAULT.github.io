
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from VfhPlus.defaults import NUM_BINS, FOV_DEG, NUM_VFH_WAYPOINTS


def waypoint_to_reference(
    dx: float,
    dy: float,
    *,
    num_bins: int = NUM_BINS,
    fov_deg: float = FOV_DEG,
) -> Tuple[np.ndarray, int]:
    fov_rad = math.radians(fov_deg)
    half_fov = fov_rad / 2.0
    bin_width = fov_rad / num_bins
    print(f"Waypoint to reference: dx={dx}, dy={dy}, angle={math.degrees(math.atan2(dy, dx)):.1f}°")
    print(f"FOV: {fov_deg}°, Bin width: {math.degrees(bin_width):.1f}°")
    angle = math.atan2(dy, dx)

    # Map angle → bin index (bin 0 = leftmost, bin N-1 = rightmost)
    bin_idx = int((half_fov - angle) / bin_width)
    print(f"Initial bin index (before clamping): {bin_idx}")
    bin_idx = max(0, min(bin_idx, num_bins - 1))
    print(f"Clamped bin index: {bin_idx}")
    nomad_vector = np.zeros(num_bins, dtype=np.float64)
    print(f"Nomad vector before setting bin: {nomad_vector}")
    nomad_vector[bin_idx] = 1.0
    print(f"Nomad vector after setting bin {bin_idx}: {nomad_vector}")

    return nomad_vector, bin_idx


def bin_to_waypoint(
    bin_idx: int,
    magnitude: float,
    *,
    num_bins: int = NUM_BINS,
    fov_deg: float = FOV_DEG,
) -> np.ndarray:
    fov_rad = math.radians(fov_deg)
    bin_width = fov_rad / num_bins
    angle = fov_rad / 2 - (bin_idx + 0.5) * bin_width
    bin_to_waypoint = np.array([magnitude * math.cos(angle), magnitude * math.sin(angle)])

    print(f"Bin to waypoint or Nomad Vector {bin_to_waypoint}")

    return bin_to_waypoint


def generate_direction_waypoints(
    angle: float,
    max_magnitude: float,
    *,
    num_waypoints: int = NUM_VFH_WAYPOINTS,
) -> np.ndarray:
    if num_waypoints < 1:
        num_waypoints = 1
    magnitudes = np.linspace(max_magnitude / num_waypoints, max_magnitude, num_waypoints)
    print(f"Generating {num_waypoints} waypoints with magnitudes: {magnitudes}")
    waypoints = np.column_stack([
        magnitudes * math.cos(angle),
        magnitudes * math.sin(angle),
    ])
    print(f"Generated {num_waypoints} waypoints along angle {math.degrees(angle):.1f}°: {waypoints}")
    # print(f"Waypoints shape: {waypoints.shape}")
    return waypoints