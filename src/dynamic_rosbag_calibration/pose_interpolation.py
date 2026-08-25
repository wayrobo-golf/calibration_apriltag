"""Strict image-time interpolation of INS poses on SE(3)."""

from __future__ import annotations

import bisect
import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .calibration_types import Transform
from .dynamic_experiment_types import (
    InterpolatedPose,
    PoseInterpolationConfig,
    TimedPose,
)


def interpolate_pose_se3(
    rows: Sequence[TimedPose],
    stamp_s: float,
    config: PoseInterpolationConfig,
) -> InterpolatedPose:
    """Interpolate ``ins_map <- ins_link`` at one strictly bracketed stamp."""
    values = tuple(rows)
    if not values:
        raise ValueError("pose interpolation requires at least one odometry sample")
    target = float(stamp_s)
    if not math.isfinite(target):
        raise ValueError("interpolation stamp must be finite")
    stamps = tuple(value.stamp_s for value in values)
    if any(right <= left for left, right in zip(stamps, stamps[1:])):
        raise ValueError("odometry stamps must be strictly increasing")
    if target < stamps[0] or target > stamps[-1]:
        raise ValueError("image stamp is outside odometry range; extrapolation forbidden")

    index = bisect.bisect_left(stamps, target)
    if index < len(values) and stamps[index] == target:
        selected = values[index]
        return InterpolatedPose(
            pose=Transform(selected.matrix, "ins_map", "ins_link"),
            left_stamp_s=target,
            right_stamp_s=target,
            alpha=0.0,
            bracket_gap_ms=0.0,
            left_dt_ms=0.0,
            right_dt_ms=0.0,
        )
    if index == 0 or index == len(values):
        raise ValueError("image stamp is outside odometry range; extrapolation forbidden")

    left = values[index - 1]
    right = values[index]
    gap_s = right.stamp_s - left.stamp_s
    left_dt_s = target - left.stamp_s
    right_dt_s = right.stamp_s - target
    gap_ms = gap_s * 1000.0
    left_ms = left_dt_s * 1000.0
    right_ms = right_dt_s * 1000.0
    tolerance_ms = 1.0e-9
    if gap_ms > config.maximum_bracket_gap_ms + tolerance_ms:
        raise ValueError(
            f"odometry bracket gap {gap_ms:.6f} ms exceeds "
            f"{config.maximum_bracket_gap_ms:.6f} ms"
        )
    if max(left_ms, right_ms) > (
        config.maximum_endpoint_distance_ms + tolerance_ms
    ):
        raise ValueError(
            "image-to-odometry endpoint distance exceeds "
            f"{config.maximum_endpoint_distance_ms:.6f} ms"
        )

    alpha = left_dt_s / gap_s
    translation = (
        (1.0 - alpha) * left.matrix[:3, 3]
        + alpha * right.matrix[:3, 3]
    )
    key_times = np.array([left.stamp_s, right.stamp_s], dtype=np.float64)
    rotations = Rotation.from_matrix(
        np.stack((left.matrix[:3, :3], right.matrix[:3, :3]))
    )
    rotation = Slerp(key_times, rotations)([target]).as_matrix()[0]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return InterpolatedPose(
        pose=Transform(matrix, "ins_map", "ins_link"),
        left_stamp_s=left.stamp_s,
        right_stamp_s=right.stamp_s,
        alpha=float(alpha),
        bracket_gap_ms=float(gap_ms),
        left_dt_ms=float(left_ms),
        right_dt_ms=float(right_ms),
    )

