
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from VfhPlus.defaults import (
    NUM_BINS, FOV_DEG, SAFETY_THRESHOLD, S_MAX, MU1, MU2, MU3,
    ROBOT_RADIUS, RECOVERY_REVERSE_CYCLES, RECOVERY_TURN_CYCLES,
    FOV_PADDING_BINS,
)


class VFHStar:
    _PHASE_NORMAL = 0
    _PHASE_REVERSE = 1
    _PHASE_TURN = 2

    def __init__(
        self,
        num_bins: int = NUM_BINS,
        fov_deg: float = FOV_DEG,
        safety_threshold: float = SAFETY_THRESHOLD,
        s_max: int = S_MAX,
        mu1: float = MU1,
        mu2: float = MU2,
        mu3: float = MU3,
        robot_radius: float = ROBOT_RADIUS,
        recovery_reverse_cycles: int = RECOVERY_REVERSE_CYCLES,
        recovery_turn_cycles: int = RECOVERY_TURN_CYCLES,
        fov_padding_bins: int = FOV_PADDING_BINS,
    ) -> None:
        self.num_bins = num_bins
        self.fov_rad = math.radians(fov_deg)
        self.bin_width = self.fov_rad / num_bins
        self.safety_threshold = safety_threshold
        self.s_max = s_max
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.robot_radius = robot_radius
        self.fov_padding_bins = fov_padding_bins

        self.recovery_reverse_cycles = recovery_reverse_cycles
        self.recovery_turn_cycles = recovery_turn_cycles

        # Recovery state
        self._recovery_phase: int = self._PHASE_NORMAL
        self._recovery_reverse_count: int = 0
        self._recovery_turn_count: int = 0
        self._recovery_turn_angle: float = 0.0
        self.recovery_just_completed: bool = False

        self.prev_selected_bin: Optional[int] = None

    def compute(
        self,
        distance_vector: np.ndarray,
        reference_bins: List[int],
        robot_heading_bin: Optional[int] = None,
    ) -> Tuple[int, float, bool]:
        if robot_heading_bin is None:
            robot_heading_bin = self.num_bins // 2

        blocked = distance_vector < self.safety_threshold
        print(f"[VFH*] safety_threshold={self.safety_threshold}  robot_radius={self.robot_radius}")
        print(f"The binary polar distance vector is {blocked} which is directly from the thresholding, no temporal and no padding yet")

        if self.fov_padding_bins > 0:
            blocked[:self.fov_padding_bins] = True
            blocked[-self.fov_padding_bins:] = True
            print(f"[VFH*] Padded {self.fov_padding_bins} bins on each side → blocked: {blocked}")

        if self.fov_padding_bins > 0:
            lo = self.fov_padding_bins
            hi = self.num_bins - 1 - self.fov_padding_bins
            clamped = []
            for rb in reference_bins:
                if rb < lo:
                    print(f"[VFH*] reference_bin {rb} in left padding → clamped to {lo}")
                    clamped.append(lo)
                elif rb > hi:
                    print(f"[VFH*] reference_bin {rb} in right padding → clamped to {hi}")
                    clamped.append(hi)
                else:
                    clamped.append(rb)
            reference_bins = clamped

        primary_ref = int(np.median(reference_bins))

        safe_refs = [
            rb for rb in reference_bins
            if not blocked[rb] and self._bin_has_clearance(rb, blocked, distance_vector)
        ]
        print(f"[VFH*] reference_bins={reference_bins}, safe_refs={safe_refs}")

        if safe_refs:
            self._reset_recovery()
            prev_bin = self.prev_selected_bin if self.prev_selected_bin is not None else primary_ref
            best_bin = self._select_best(safe_refs, reference_bins, robot_heading_bin, prev_bin)
            self.prev_selected_bin = best_bin
            bin_to_angle = self._bin_to_angle(best_bin)
            print(f"[VFH*] Fast path: best_bin={best_bin}, angle={math.degrees(bin_to_angle):.1f}°")
            return best_bin, bin_to_angle, False

        print(f"[VFH*] No safe reference bin found — searching valleys")

        all_valleys = self._find_valleys(blocked)
        print(f"The reference idx was blocked and this is the vallyes: {all_valleys}")

        if not all_valleys:
            print(f"No free valleys at all – triggering recovery")
            return self._recover(distance_vector)

        wide_valleys = self._filter_narrow_valleys(all_valleys, distance_vector)
        if wide_valleys:
            self._reset_recovery()
            valleys = wide_valleys
            print(f"Valleys after robot-width filtering: {valleys}")
        else:
            return self._recover(distance_vector)

        candidates = self._generate_candidates(valleys, primary_ref)
        print(f"The candidates of the valleys are {candidates}")
        if not candidates:
            print(f"No candidate chosen from valleys! :(")
            return primary_ref, self._bin_to_angle(primary_ref), True

        safe_candidates = [
            c for c in candidates
            if self._bin_has_clearance(c, blocked, distance_vector)
        ]
        if safe_candidates:
            print(f"Candidates after clearance filtering: {safe_candidates} (from {candidates})")
            candidates = safe_candidates
        else:
            print(f"[VFH*] All candidates fail clearance check — triggering recovery")
            return self._recover(distance_vector)

        prev_bin = (
            self.prev_selected_bin
            if self.prev_selected_bin is not None
            else primary_ref
        )
        print(f"Previous selected bin: {prev_bin} (primary_ref if None)")

        best_bin = self._select_best(
            candidates, reference_bins, robot_heading_bin, prev_bin
        )
        print(f"prev_bin={prev_bin}, best_bin={best_bin}, angle={math.degrees(self._bin_to_angle(best_bin)):.1f}°")
        self.prev_selected_bin = best_bin
        return best_bin, self._bin_to_angle(best_bin), True

    def angle_to_bin(self, angle: float) -> int:
        idx = int((self.fov_rad / 2 - angle) / self.bin_width)
        return max(0, min(idx, self.num_bins - 1))

    def _bin_to_angle(self, bin_idx: int) -> float:
        return self.fov_rad / 2 - (bin_idx + 0.5) * self.bin_width

    def _find_valleys(self, blocked: np.ndarray) -> List[Tuple[int, int]]:
        valleys: List[Tuple[int, int]] = []
        n = len(blocked)
        i = 0
        while i < n:
            if not blocked[i]:
                start = i
                while i < n and not blocked[i]:
                    i += 1
                valleys.append((start, i - 1))
            else:
                i += 1
        return valleys

    def _is_padding_bin(self, idx: int) -> bool:
        if self.fov_padding_bins <= 0:
            return False
        return idx < self.fov_padding_bins or idx >= self.num_bins - self.fov_padding_bins

    def _gap_width(
        self, n_bins: int, distance_vector: np.ndarray, left_idx: int, right_idx: int
    ) -> float:
        angular_width = n_bins * self.bin_width
        print(f"Calculating gap width for valley ({left_idx}, {right_idx}) with angular width {math.degrees(angular_width):.1f}°")
        edge_dists: List[float] = []
        if left_idx > 0 and not self._is_padding_bin(left_idx - 1):
            edge_dists.append(float(distance_vector[left_idx - 1]))

        if right_idx < self.num_bins - 1 and not self._is_padding_bin(right_idx + 1):
            edge_dists.append(float(distance_vector[right_idx + 1]))

        valley_dists = distance_vector[left_idx : right_idx + 1]
        finite_valley = valley_dists[np.isfinite(valley_dists)]
        if len(finite_valley) > 0:
            d = float(np.mean(finite_valley))
        elif edge_dists:
            d = min(edge_dists)
        else:
            d = self.safety_threshold

        return d * 2.0 * math.tan(angular_width / 2.0)

    def _filter_narrow_valleys(
        self,
        valleys: List[Tuple[int, int]],
        distance_vector: np.ndarray,
    ) -> List[Tuple[int, int]]:
        robot_diameter = 2.0 * self.robot_radius
        kept: List[Tuple[int, int]] = []
        for start, end in valleys:
            n_bins = end - start + 1
            gap = self._gap_width(n_bins, distance_vector, start, end)
            if gap >= robot_diameter + 0.50:
                kept.append((start, end))
            else:
                print(f"  Valley ({start},{end}) rejected: gap={gap:.3f}m < robot_diameter={robot_diameter:.3f}m")
        return kept

    def _bin_has_clearance(
        self, bin_idx: int, blocked: np.ndarray, distance_vector: np.ndarray
    ) -> bool:
        start = bin_idx
        while start > 0 and not blocked[start - 1]:
            start -= 1
        end = bin_idx
        while end < self.num_bins - 1 and not blocked[end + 1]:
            end += 1

        n_bins = end - start + 1
        gap = self._gap_width(n_bins, distance_vector, start, end)
        robot_diameter = 2.0 * self.robot_radius
        if gap < robot_diameter + 0.5:
            return False

        neighbourhood = distance_vector[max(0, bin_idx - 1) : bin_idx + 2]
        finite_vals = neighbourhood[np.isfinite(neighbourhood)]
        d = float(np.mean(finite_vals)) if len(finite_vals) > 0 else self.safety_threshold
        margin_angle = math.atan2(self.robot_radius, d)
        margin_bins = max(1, math.ceil(margin_angle / self.bin_width))

        left_margin = bin_idx - start
        right_margin = end - bin_idx
        ok = left_margin >= margin_bins and right_margin >= margin_bins
        if not ok:
            print(f"[VFH*] Bin {bin_idx} clearance fail: "
                  f"valley=({start},{end}), L_margin={left_margin}, "
                  f"R_margin={right_margin}, need={margin_bins} "
                  f"(d={d:.2f}m, robot_r={self.robot_radius}m)")
        return ok

    def _reset_recovery(self) -> None:
        if self._recovery_phase != self._PHASE_NORMAL:
            print("[VFH* RECOVERY] Recovery complete — resuming normal operation.")
            self.recovery_just_completed = True
        self._recovery_phase = self._PHASE_NORMAL
        self._recovery_reverse_count = 0
        self._recovery_turn_count = 0
        self._recovery_turn_angle = 0.0

    def _pick_turn_direction(self, distance_vector: np.ndarray) -> float:
        mid = self.num_bins // 2
        if self.prev_selected_bin is not None and self.prev_selected_bin != mid:
            if self.prev_selected_bin < mid:
                print(
                    f"[VFH* RECOVERY] Turning LEFT to follow previous selected bin "
                    f"({self.prev_selected_bin} < {mid})"
                )
                return math.pi / 2.0
            print(
                f"[VFH* RECOVERY] Turning RIGHT to follow previous selected bin "
                f"({self.prev_selected_bin} >= {mid})"
            )
            return -math.pi / 2.0

        left = np.asarray(distance_vector[:mid], dtype=float)
        right = np.asarray(distance_vector[mid:], dtype=float)
        left_open_bins = int(np.count_nonzero((left > self.safety_threshold) | ~np.isfinite(left)))
        right_open_bins = int(np.count_nonzero((right > self.safety_threshold) | ~np.isfinite(right)))

        left_finite = left[np.isfinite(left)]
        right_finite = right[np.isfinite(right)]
        left_clearance = float(np.mean(left_finite)) if left_finite.size > 0 else 0.0
        right_clearance = float(np.mean(right_finite)) if right_finite.size > 0 else 0.0

        if left_open_bins > right_open_bins:
            print(
                f"[VFH* RECOVERY] Turning LEFT "
                f"(open_bins L={left_open_bins} > R={right_open_bins})"
            )
            return math.pi / 2.0
        if right_open_bins > left_open_bins:
            print(
                f"[VFH* RECOVERY] Turning RIGHT "
                f"(open_bins R={right_open_bins} > L={left_open_bins})"
            )
            return -math.pi / 2.0

        if left_clearance > right_clearance:
            print(
                f"[VFH* RECOVERY] Turning LEFT "
                f"(mean_clearance L={left_clearance:.2f} > R={right_clearance:.2f})"
            )
            return math.pi / 2.0
        if right_clearance > left_clearance:
            print(
                f"[VFH* RECOVERY] Turning RIGHT "
                f"(mean_clearance R={right_clearance:.2f} > L={left_clearance:.2f})"
            )
            return -math.pi / 2.0
        print(
            "[VFH* RECOVERY] Left/right tie in openness and clearance; "
            "defaulting RIGHT to avoid left bias"
        )
        return -math.pi / 2.0

    def _recover(
        self,
        distance_vector: np.ndarray,
    ) -> Tuple[int, float, bool]:
        if self._recovery_phase == self._PHASE_NORMAL:
            self._recovery_phase = self._PHASE_REVERSE
            self._recovery_reverse_count = 0
            print(f"[VFH* RECOVERY] No safe valley — backing up for "
                  f"{self.recovery_reverse_cycles} cycles, then turning")

        if self._recovery_phase == self._PHASE_REVERSE:
            self._recovery_reverse_count += 1
            if self._recovery_reverse_count > self.recovery_reverse_cycles:
                self._recovery_phase = self._PHASE_TURN
                self._recovery_turn_count = 0
                self._recovery_turn_angle = self._pick_turn_direction(distance_vector)
                print(f"The robot should rotate {self._recovery_turn_angle}  ")
                print(f"[VFH* RECOVERY] Reverse done — starting {self.recovery_turn_cycles}-cycle turn")
            else:
                print(f"[VFH* RECOVERY] Reversing — step "
                      f"{self._recovery_reverse_count}/{self.recovery_reverse_cycles}")
                return self.num_bins // 2, math.pi, True

        if self._recovery_phase == self._PHASE_TURN:
            self._recovery_turn_count += 1
            if self._recovery_turn_count > self.recovery_turn_cycles:
                print("[VFH* RECOVERY] Turn complete — resuming normal pipeline")
                self._reset_recovery()
                return self.num_bins // 2, 0.0, False
            else:
                print(f"[VFH* RECOVERY] Turning — step "
                      f"{self._recovery_turn_count}/{self.recovery_turn_cycles}")
                return self.num_bins // 2, self._recovery_turn_angle, True

    def _generate_candidates(
        self, valleys: List[Tuple[int, int]], reference_bin: int
    ) -> List[int]:
        candidates: List[int] = []
        half_s = self.s_max // 2

        for start, end in valleys:
            width = end - start + 1
            if width < self.s_max:
                candidates.append((start + end) // 2)
            else:
                if abs(start - reference_bin) <= abs(end - reference_bin):
                    near, far = start, end
                else:
                    near, far = end, start
                c_near = near + half_s if near == start else near - half_s
                c_far = far - half_s if far == end else far + half_s
                c_near = max(start, min(c_near, end))
                c_far = max(start, min(c_far, end))

                candidates.append(c_near)
                candidates.append(c_far)
                candidates.append((start + end) // 2)

        clamped = list({max(0, min(c, self.num_bins - 1)) for c in candidates})
        return clamped

    def _select_best(
        self,
        candidates: List[int],
        reference_bins: List[int],
        heading_bin: int,
        prev_bin: int,
    ) -> int:
        best_cost = float("inf")
        best = candidates[0]
        for c in candidates:
            target_cost = min(self._bin_distance(c, r) for r in reference_bins)
            cost = (
                self.mu1 * target_cost
                + self.mu2 * self._bin_distance(c, heading_bin)
                + self.mu3 * self._bin_distance(c, prev_bin)
            )
            if cost < best_cost:
                best_cost = cost
                best = c
        return best

    def _bin_distance(self, a: int, b: int) -> int:
        diff = abs(a - b)
        return min(diff, self.num_bins - diff)
