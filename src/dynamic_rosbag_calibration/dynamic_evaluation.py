"""Independent dynamic localization evaluation in the INS_ODOM map frame."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .calibration_types import readonly_float_array
from .se3 import inverse


@dataclass(frozen=True)
class DynamicFrame:
    frame_key: str
    station_id: str
    stamp_s: float
    odom_dt_ms: float
    reference_map_ins: np.ndarray
    camera_tag0: np.ndarray
    pnp_rms_px: float
    odom_source_stamp_s: float | None = None

    def __post_init__(self) -> None:
        if not self.frame_key or not self.station_id:
            raise ValueError("dynamic frame identifiers must be non-empty")
        for name in ("stamp_s", "odom_dt_ms", "pnp_rms_px"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.odom_dt_ms < 0.0:
            raise ValueError("odom_dt_ms must be non-negative")
        if self.odom_source_stamp_s is not None:
            if not math.isfinite(float(self.odom_source_stamp_s)):
                raise ValueError("odom_source_stamp_s must be finite")
            measured_dt_ms = abs(self.stamp_s - self.odom_source_stamp_s) * 1000.0
            if not math.isclose(
                measured_dt_ms,
                self.odom_dt_ms,
                rel_tol=0.0,
                abs_tol=1.0e-3,
            ):
                raise ValueError(
                    "odom_dt_ms must match image and odometry source timestamps"
                )
        object.__setattr__(
            self,
            "reference_map_ins",
            readonly_float_array(self.reference_map_ins, (4, 4), "reference_map_ins"),
        )
        object.__setattr__(
            self,
            "camera_tag0",
            readonly_float_array(self.camera_tag0, (4, 4), "camera_tag0"),
        )


ERROR_METRICS: Mapping[str, str] = {
    "error_x_m": "m",
    "error_y_m": "m",
    "error_yaw_deg": "deg",
    "abs_error_x_m": "m",
    "abs_error_y_m": "m",
    "error_xy_m": "m",
    "abs_error_yaw_deg": "deg",
}

DYNAMIC_TRUE_AXIS_METRICS: Mapping[str, str] = {
    "error_x_m": "m",
    "error_y_m": "m",
    "error_yaw_deg": "deg",
}


@dataclass(frozen=True)
class StaticValidationResult:
    passed: bool
    xy_p80_m: float
    yaw_p80_deg: float
    reasons: tuple[str, ...]


def classify_static_validation(
    *,
    xy_p80_m: float,
    yaw_p80_deg: float,
    maximum_xy_p80_m: float = 0.03,
    maximum_yaw_p80_deg: float = 0.5,
) -> StaticValidationResult:
    values = (xy_p80_m, yaw_p80_deg, maximum_xy_p80_m, maximum_yaw_p80_deg)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("static validation values must be finite")
    if min(values) < 0.0:
        raise ValueError("static validation values must be non-negative")
    reasons: list[str] = []
    if xy_p80_m > maximum_xy_p80_m:
        reasons.append("STATIC_XY_P80_EXCEEDS_0_03_M")
    if yaw_p80_deg > maximum_yaw_p80_deg:
        reasons.append("STATIC_YAW_P80_EXCEEDS_0_5_DEG")
    return StaticValidationResult(
        passed=not reasons,
        xy_p80_m=float(xy_p80_m),
        yaw_p80_deg=float(yaw_p80_deg),
        reasons=tuple(reasons),
    )


def parse_true_dynamic_bags(
    readme_text: str, minimum_xy_span_m: float = 1.0
) -> tuple[str, ...]:
    """Return INS-qualified dynamic bags whose documented XY span is >= limit."""
    if not math.isfinite(minimum_xy_span_m) or minimum_xy_span_m <= 0.0:
        raise ValueError("minimum_xy_span_m must be positive and finite")
    selected: list[str] = []
    minimum_cm = minimum_xy_span_m * 100.0
    for line in readme_text.splitlines():
        if not line.lstrip().startswith("| bag_"):
            continue
        columns = [value.strip() for value in line.strip().strip("|").split("|")]
        if len(columns) < 5:
            continue
        name, availability, motion = columns[0], columns[2], columns[3]
        try:
            horizontal_span_cm = float(columns[4])
        except ValueError:
            continue
        if (
            availability == "可用"
            and motion == "动态"
            and horizontal_span_cm >= minimum_cm
        ):
            selected.append(name)
    return tuple(sorted(dict.fromkeys(selected)))


def _yaw_deg(transform: np.ndarray) -> float:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))


def wrapped_yaw_difference_deg(visual_deg: float, reference_deg: float) -> float:
    difference_rad = math.radians(float(visual_deg) - float(reference_deg))
    return math.degrees(math.atan2(math.sin(difference_rad), math.cos(difference_rad)))


def unwrap_yaw_for_display(
    reference_deg: Sequence[float], visual_deg: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_deg, dtype=np.float64).reshape(-1)
    visual = np.asarray(visual_deg, dtype=np.float64).reshape(-1)
    if reference.shape != visual.shape:
        raise ValueError("reference and visual yaw arrays must have the same shape")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(visual)):
        raise ValueError("yaw arrays must be finite")
    if reference.size == 0:
        return reference.copy(), visual.copy()
    reference_unwrapped = np.degrees(np.unwrap(np.radians(reference)))
    visual_unwrapped = np.degrees(np.unwrap(np.radians(visual)))
    branch_shift = 360.0 * round(
        float(reference_unwrapped[0] - visual_unwrapped[0]) / 360.0
    )
    visual_unwrapped = visual_unwrapped + branch_shift
    return reference_unwrapped, visual_unwrapped


def evaluate_dynamic_frames(
    frames: Iterable[DynamicFrame],
    board_pose: np.ndarray,
    effective_extrinsic: np.ndarray,
) -> list[dict[str, object]]:
    """Reconstruct visual map<-ins poses without fitting to dynamic INS data."""
    board = readonly_float_array(board_pose, (4, 4), "board_pose")
    extrinsic = readonly_float_array(
        effective_extrinsic, (4, 4), "effective_extrinsic"
    )
    by_station: dict[str, list[DynamicFrame]] = defaultdict(list)
    for frame in frames:
        by_station[frame.station_id].append(frame)

    rows: list[dict[str, object]] = []
    for station_id in sorted(by_station):
        station_frames = sorted(
            by_station[station_id], key=lambda value: (value.stamp_s, value.frame_key)
        )
        first_reference = station_frames[0].reference_map_ins[:3, 3]
        first_stamp = station_frames[0].stamp_s
        station_rows: list[dict[str, object]] = []
        for sample_index, frame in enumerate(station_frames, start=1):
            predicted = board @ inverse(frame.camera_tag0) @ inverse(extrinsic)
            reference = frame.reference_map_ins
            translation_error = predicted[:3, 3] - reference[:3, 3]
            reference_yaw = _yaw_deg(reference)
            visual_yaw = _yaw_deg(predicted)
            yaw_error = wrapped_yaw_difference_deg(visual_yaw, reference_yaw)
            row: dict[str, object] = {
                "frame_key": frame.frame_key,
                "station_id": station_id,
                "sample_index": sample_index,
                "stamp_s": float(frame.stamp_s),
                "image_source_stamp_s": float(frame.stamp_s),
                "odom_source_stamp_s": (
                    float(frame.odom_source_stamp_s)
                    if frame.odom_source_stamp_s is not None
                    else None
                ),
                "elapsed_s": float(frame.stamp_s - first_stamp),
                "odom_dt_ms": float(frame.odom_dt_ms),
                "pnp_rms_px": float(frame.pnp_rms_px),
                "camera_tag0_tz_m": float(frame.camera_tag0[2, 3]),
                "reference_x_m": float(reference[0, 3]),
                "reference_y_m": float(reference[1, 3]),
                "reference_z_m": float(reference[2, 3]),
                "reference_yaw_deg": reference_yaw,
                "visual_x_m": float(predicted[0, 3]),
                "visual_y_m": float(predicted[1, 3]),
                "visual_z_m": float(predicted[2, 3]),
                "visual_yaw_deg": visual_yaw,
                "reference_x_local_m": float(reference[0, 3] - first_reference[0]),
                "reference_y_local_m": float(reference[1, 3] - first_reference[1]),
                "visual_x_local_m": float(predicted[0, 3] - first_reference[0]),
                "visual_y_local_m": float(predicted[1, 3] - first_reference[1]),
                "error_x_m": float(translation_error[0]),
                "error_y_m": float(translation_error[1]),
                "error_z_m": float(translation_error[2]),
                "abs_error_x_m": float(abs(translation_error[0])),
                "abs_error_y_m": float(abs(translation_error[1])),
                "error_xy_m": float(np.linalg.norm(translation_error[:2])),
                "error_xyz_m": float(np.linalg.norm(translation_error)),
                "error_yaw_deg": yaw_error,
                "abs_error_yaw_deg": abs(yaw_error),
            }
            station_rows.append(row)
        reference_display, visual_display = unwrap_yaw_for_display(
            [float(row["reference_yaw_deg"]) for row in station_rows],
            [float(row["visual_yaw_deg"]) for row in station_rows],
        )
        for row, reference_yaw, visual_yaw in zip(
            station_rows, reference_display, visual_display
        ):
            row["reference_yaw_display_deg"] = float(reference_yaw)
            row["visual_yaw_display_deg"] = float(visual_yaw)
        rows.extend(station_rows)
    return rows


def summarize_errors(
    rows: Iterable[Mapping[str, object]], group_fields: Sequence[str]
) -> list[dict[str, object]]:
    values = list(rows)
    if not values:
        raise ValueError("cannot summarize an empty dynamic frame set")
    fields = tuple(group_fields)
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in values:
        grouped[tuple(row[field] for field in fields)].append(row)

    output: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        group = grouped[key]
        for metric, unit in ERROR_METRICS.items():
            array = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{metric} contains non-finite values")
            result = {field: value for field, value in zip(fields, key)}
            result.update(
                {
                    "metric": metric,
                    "unit": unit,
                    "count": int(array.size),
                    "mean": float(np.mean(array)),
                    "median": float(np.median(array)),
                    "p50": float(np.quantile(array, 0.50, method="linear")),
                    "p80": float(np.quantile(array, 0.80, method="linear")),
                    "p95": float(np.quantile(array, 0.95, method="linear")),
                    "p99": float(np.quantile(array, 0.99, method="linear")),
                    "max": float(np.max(array)),
                }
            )
            output.append(result)
    return output


def summarize_dynamic_true_errors(
    rows: Iterable[Mapping[str, object]],
    *,
    maximum_camera_tag0_tz_m: float = 8.0,
) -> list[dict[str, object]]:
    """Summarize signed axis bias and absolute tails for true-dynamic frames."""
    maximum_depth = float(maximum_camera_tag0_tz_m)
    if not math.isfinite(maximum_depth) or maximum_depth <= 0.0:
        raise ValueError("maximum_camera_tag0_tz_m must be positive and finite")
    dynamic_true = [row for row in rows if row.get("group") == "dynamic_true"]
    if not dynamic_true:
        raise ValueError("cannot summarize empty dynamic_true frame set")
    within_limit = [
        row
        for row in dynamic_true
        if float(row["camera_tag0_tz_m"]) <= maximum_depth
    ]
    if not within_limit:
        raise ValueError("dynamic_true contains no frames within the Tag0 depth limit")

    cohorts = (
        ("all", None, dynamic_true),
        ("camera_tag0_tz_le_8m", maximum_depth, within_limit),
    )
    output: list[dict[str, object]] = []
    for distance_scope, depth_limit, cohort in cohorts:
        for metric, unit in DYNAMIC_TRUE_AXIS_METRICS.items():
            signed = np.asarray(
                [float(row[metric]) for row in cohort],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(signed)):
                raise ValueError(f"{metric} contains non-finite values")
            absolute = np.abs(signed)
            output.append(
                {
                    "distance_scope": distance_scope,
                    "maximum_camera_tag0_tz_m": depth_limit,
                    "metric": metric,
                    "unit": unit,
                    "count": int(signed.size),
                    "signed_median": float(np.median(signed)),
                    "abs_p80": float(
                        np.quantile(absolute, 0.80, method="linear")
                    ),
                    "abs_p95": float(
                        np.quantile(absolute, 0.95, method="linear")
                    ),
                    "abs_pmax": float(np.max(absolute)),
                }
            )
    return output
