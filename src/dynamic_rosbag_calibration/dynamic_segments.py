"""Structural-state classification for safe dynamic calibration frames."""

from __future__ import annotations

import bisect
import math
from typing import Sequence

import numpy as np

from .dynamic_experiment_types import (
    DynamicBagEvidence,
    DynamicFrameEvidence,
    HighpassMotionSample,
    StructuralGateConfig,
    StructuralState,
    TimedImu,
)


def _strict_stamps(values: Sequence[object], name: str) -> np.ndarray:
    stamps = np.asarray([float(getattr(item, "stamp_s")) for item in values])
    if stamps.size and np.any(np.diff(stamps) <= 0.0):
        raise ValueError(f"{name} source stamps must be strictly increasing")
    return stamps


def compute_highpass_motion(
    imu_samples: Sequence[TimedImu],
    window_s: float,
) -> tuple[HighpassMotionSample, ...]:
    """Subtract a centered time-window median from each six-axis IMU sample."""
    values = tuple(imu_samples)
    if not values:
        return ()
    window = float(window_s)
    if not math.isfinite(window) or window <= 0.0:
        raise ValueError("highpass window must be finite and positive")
    stamps = _strict_stamps(values, "IMU")
    acceleration = np.stack(
        [item.linear_acceleration_mps2 for item in values]
    )
    angular = np.stack([item.angular_velocity_radps for item in values])
    half = window / 2.0
    result: list[HighpassMotionSample] = []
    for index, stamp in enumerate(stamps):
        left = int(np.searchsorted(stamps, stamp - half, side="left"))
        right = int(np.searchsorted(stamps, stamp + half, side="right"))
        acceleration_residual = acceleration[index] - np.median(
            acceleration[left:right], axis=0
        )
        angular_residual = angular[index] - np.median(angular[left:right], axis=0)
        result.append(
            HighpassMotionSample(
                stamp_s=float(stamp),
                acceleration_norm_mps2=float(np.linalg.norm(acceleration_residual)),
                angular_rate_norm_degps=float(
                    np.degrees(np.linalg.norm(angular_residual))
                ),
            )
        )
    return tuple(result)


def _nearest_motion(
    samples: Sequence[HighpassMotionSample], stamp_s: float
) -> tuple[HighpassMotionSample | None, float | None]:
    if not samples:
        return None, None
    stamps = [item.stamp_s for item in samples]
    index = bisect.bisect_left(stamps, stamp_s)
    candidates = [value for value in (index - 1, index) if 0 <= value < len(samples)]
    selected = min(candidates, key=lambda value: abs(stamps[value] - stamp_s))
    delta_s = abs(stamps[selected] - stamp_s)
    if delta_s > 0.05:
        return None, delta_s * 1000.0
    return samples[selected], delta_s * 1000.0


def classify_dynamic_frames(
    bag: DynamicBagEvidence,
    config: StructuralGateConfig,
    *,
    bag_known_to_avoid_bridge: bool = False,
) -> tuple[DynamicFrameEvidence, ...]:
    """Classify every frame; only pre-contact, quiet frames become safe."""
    frames = tuple(
        sorted(bag.frames, key=lambda item: (item.source_stamp_s, item.frame_key))
    )
    motion = compute_highpass_motion(bag.imu, config.highpass_window_s)

    contact_index: int | None = None
    turnaround_index: int | None = None
    release_index: int | None = None
    if not bag_known_to_avoid_bridge:
        for index, frame in enumerate(frames):
            depth = frame.initial_tag0_depth_m
            if depth is not None and depth <= config.contact_candidate_depth_m:
                contact_index = index
                break
        if contact_index is not None:
            valid_depths = [
                (index, float(frame.initial_tag0_depth_m))
                for index, frame in enumerate(frames[contact_index:], contact_index)
                if frame.initial_tag0_depth_m is not None
            ]
            if valid_depths:
                turnaround_index = min(valid_depths, key=lambda item: item[1])[0]
                for index, frame in enumerate(
                    frames[turnaround_index + 1 :],
                    turnaround_index + 1,
                ):
                    depth = frame.initial_tag0_depth_m
                    if depth is not None and depth > config.contact_candidate_depth_m:
                        release_index = index
                        break

    result: list[DynamicFrameEvidence] = []
    for index, frame in enumerate(frames):
        reasons = list(frame.exclusion_reasons)
        sample, _motion_dt_ms = _nearest_motion(motion, frame.source_stamp_s)
        acceleration = None if sample is None else sample.acceleration_norm_mps2
        angular = None if sample is None else sample.angular_rate_norm_degps

        if contact_index is not None and index >= contact_index:
            reasons.append("CAL-E-DYNAMIC-POST-CONTACT")
            if index == contact_index:
                state = StructuralState.CONTACT_TRANSIENT
            elif release_index is not None and index >= release_index:
                state = StructuralState.RELEASE_TRANSIENT
            else:
                state = StructuralState.LOADED_STABLE
        else:
            state = StructuralState.UNKNOWN
            depth = frame.initial_tag0_depth_m
            if frame.observation is None or reasons:
                reasons.append("CAL-E-DYNAMIC-BASE-EVIDENCE")
            if not frame.ins_quality_good:
                reasons.append("CAL-E-DYNAMIC-INS-QUALITY")
            if depth is None or not (
                config.minimum_safe_depth_m <= depth <= config.maximum_depth_m
            ):
                reasons.append("CAL-E-DYNAMIC-DEPTH-RANGE")
            if sample is None:
                reasons.append("CAL-E-DYNAMIC-IMU-MISSING")
            else:
                if acceleration >= config.acceleration_highpass_limit_mps2:
                    reasons.append("CAL-E-DYNAMIC-ACCELERATION")
                if angular >= config.angular_rate_highpass_limit_degps:
                    reasons.append("CAL-E-DYNAMIC-ANGULAR-RATE")
            if not reasons:
                state = StructuralState.UNLOADED_SAFE

        result.append(
            DynamicFrameEvidence(
                source=frame,
                structural_state=state,
                acceleration_highpass_mps2=acceleration,
                angular_rate_highpass_degps=angular,
                exclusion_reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return tuple(result)

