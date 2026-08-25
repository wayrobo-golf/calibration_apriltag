"""Common-frame independent validation for safe-dynamic candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration_types import CameraModel, Transform
from .dynamic_evaluation import (
    DynamicFrame,
    evaluate_dynamic_frames,
    wrapped_yaw_difference_deg,
)
from .dynamic_experiment_types import DynamicFrameEvidence, StructuralState
from .pnp_prepare import solve_frame_with_frozen_geometry
from .tag_geometry import FrozenTagGeometry


@dataclass(frozen=True)
class SafeDynamicValidationResult:
    rows: tuple[Mapping[str, object], ...]
    per_bag_metrics: tuple[Mapping[str, object], ...]
    depth_metrics: tuple[Mapping[str, object], ...]
    pooled_metrics: tuple[Mapping[str, object], ...]
    retention: tuple[Mapping[str, object], ...]
    common_frame_counts: tuple[Mapping[str, object], ...]


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("validation metric cohort must be finite and non-empty")
    return {
        "p50": float(np.quantile(array, 0.50, method="linear")),
        "p80": float(np.quantile(array, 0.80, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "pmax": float(np.max(array)),
    }


def _bag_metrics(
    candidate_id: str,
    dataset: str,
    bag_id: str,
    rows: Sequence[Mapping[str, object]],
    validation_scope: str,
) -> dict[str, object]:
    primary = [row for row in rows if bool(row["primary_scope"])]
    if not primary:
        raise ValueError(f"{candidate_id}/{bag_id} has no primary validation frame")
    xy = [float(row["error_xy_m"]) for row in primary]
    yaw = [float(row["abs_error_yaw_deg"]) for row in primary]
    result: dict[str, object] = {
        "candidate_id": candidate_id,
        "dataset": dataset,
        "bag_id": bag_id,
        "validation_scope": validation_scope,
        "frame_count": len(primary),
        "signed_median_x_m": float(
            np.median([float(row["error_x_m"]) for row in primary])
        ),
        "signed_median_y_m": float(
            np.median([float(row["error_y_m"]) for row in primary])
        ),
        "signed_median_yaw_deg": float(
            np.median([float(row["error_yaw_deg"]) for row in primary])
        ),
    }
    result.update({f"xy_{key}_m": value for key, value in _quantiles(xy).items()})
    result.update(
        {f"yaw_{key}_deg": value for key, value in _quantiles(yaw).items()}
    )
    return result


def _distance_bin(depth_m: float) -> str:
    if depth_m <= 4.5:
        return "near"
    if depth_m <= 6.5:
        return "middle"
    return "far"


def _yaw_deg(transform: np.ndarray) -> float:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))


def evaluate_safe_dynamic_candidates(
    candidates: Mapping[str, Mapping[str, Any]],
    bags: Mapping[str, Sequence[DynamicFrameEvidence]],
    *,
    dataset: str,
    camera_model: CameraModel,
    tag_size_m: float,
    primary_states: Sequence[StructuralState] = (StructuralState.UNLOADED_SAFE,),
    validation_scope: str = "UNLOADED_SAFE",
) -> SafeDynamicValidationResult:
    """Evaluate every candidate on the exact same retained keys per bag."""
    if not candidates or not bags:
        raise ValueError("safe dynamic validation requires candidates and bags")
    primary_state_set = frozenset(primary_states)
    if not primary_state_set:
        raise ValueError("validation primary states must be non-empty")
    scope = str(validation_scope)
    if not scope:
        raise ValueError("validation scope must be non-empty")
    all_rows: list[Mapping[str, object]] = []
    metrics: list[Mapping[str, object]] = []
    depth_metrics: list[Mapping[str, object]] = []
    retention_rows: list[Mapping[str, object]] = []
    common_counts: list[Mapping[str, object]] = []
    for bag_id in sorted(bags):
        evidence_by_key = {
            item.source.frame_key: item
            for item in bags[bag_id]
            if item.source.observation is not None
            and not item.source.exclusion_reasons
        }
        if not evidence_by_key:
            raise ValueError(f"validation bag {bag_id} has no base-valid frames")
        primary_keys = {
            frame_key
            for frame_key, evidence in evidence_by_key.items()
            if evidence.structural_state in primary_state_set
        }
        if not primary_keys:
            raise ValueError(f"validation bag {bag_id} has no primary base-valid frames")
        prepared_by_candidate: dict[str, dict[str, object]] = {}
        for candidate_id, candidate in sorted(candidates.items()):
            frozen = FrozenTagGeometry(
                Transform(
                    np.asarray(candidate["T_tag0_tag1_calibrated"], dtype=np.float64),
                    "tag0",
                    "tag1",
                ),
                (),
            )
            prepared: dict[str, object] = {}
            for frame_key, evidence in evidence_by_key.items():
                observation = evidence.source.observation
                assert observation is not None
                solved = solve_frame_with_frozen_geometry(
                    observation,
                    camera_model,
                    tag_size_m,
                    frozen,
                )
                if solved.passed and solved.camera_tag0 is not None:
                    prepared[frame_key] = solved
            prepared_by_candidate[candidate_id] = prepared
            retention_rows.append(
                {
                    "candidate_id": candidate_id,
                    "dataset": dataset,
                    "bag_id": bag_id,
                    "validation_scope": scope,
                    "base_valid_frame_count": len(evidence_by_key),
                    "retained_frame_count": len(prepared),
                    "retained_fraction": len(prepared) / len(evidence_by_key),
                    "base_primary_frame_count": len(primary_keys),
                    "retained_primary_frame_count": len(
                        primary_keys.intersection(prepared)
                    ),
                    "primary_retained_fraction": (
                        len(primary_keys.intersection(prepared)) / len(primary_keys)
                    ),
                }
            )
        common = set(evidence_by_key)
        for prepared in prepared_by_candidate.values():
            common.intersection_update(prepared)
        if not common:
            raise ValueError(f"validation bag {bag_id} has zero common candidate frames")
        common_counts.append(
            {
                "dataset": dataset,
                "bag_id": bag_id,
                "validation_scope": scope,
                "base_valid_frame_count": len(evidence_by_key),
                "common_frame_count": len(common),
                "common_fraction": len(common) / len(evidence_by_key),
                "base_primary_frame_count": len(primary_keys),
                "common_primary_frame_count": len(primary_keys.intersection(common)),
                "common_primary_fraction": (
                    len(primary_keys.intersection(common)) / len(primary_keys)
                ),
            }
        )
        for candidate_id, candidate in sorted(candidates.items()):
            board_pose = np.asarray(
                candidate["T_ins_map_tag0_calibrated"], dtype=np.float64
            )
            effective_extrinsic = np.asarray(
                candidate["T_ins_camera_calibrated"], dtype=np.float64
            )
            dynamic_frames: list[DynamicFrame] = []
            for frame_key in sorted(
                common,
                key=lambda key: (
                    evidence_by_key[key].source.source_stamp_s,
                    key,
                ),
            ):
                evidence = evidence_by_key[frame_key]
                observation = evidence.source.observation
                solved = prepared_by_candidate[candidate_id][frame_key]
                assert observation is not None
                dynamic_frames.append(
                    DynamicFrame(
                        frame_key=frame_key,
                        station_id=bag_id,
                        stamp_s=evidence.source.source_stamp_s,
                        odom_dt_ms=observation.odom_dt_ms,
                        reference_map_ins=observation.map_ins.matrix,
                        camera_tag0=solved.camera_tag0.matrix,
                        pnp_rms_px=float(solved.dual_reprojection_rms_px),
                    )
                )
            evaluated = evaluate_dynamic_frames(
                dynamic_frames,
                board_pose,
                effective_extrinsic,
            )
            observed_boards = {
                frame.frame_key: (
                    frame.reference_map_ins
                    @ effective_extrinsic
                    @ frame.camera_tag0
                )
                for frame in dynamic_frames
            }
            candidate_rows: list[Mapping[str, object]] = []
            for row in evaluated:
                evidence = evidence_by_key[str(row["frame_key"])]
                observed_board = observed_boards[str(row["frame_key"])]
                displacement = board_pose[:3, 3] - observed_board[:3, 3]
                value = dict(row)
                value.update(
                    candidate_id=candidate_id,
                    dataset=dataset,
                    bag_id=bag_id,
                    structural_state=evidence.structural_state.value,
                    validation_scope=scope,
                    initial_tag0_depth_m=evidence.source.initial_tag0_depth_m,
                    # The structural state was produced from candidate-independent
                    # Tag0-only PnP depth.  Do not let a candidate change its own
                    # primary validation cohort through frozen-geometry PnP depth.
                    primary_scope=(
                        evidence.structural_state in primary_state_set
                    ),
                    acceleration_highpass_mps2=(
                        evidence.acceleration_highpass_mps2
                    ),
                    angular_rate_highpass_degps=(
                        evidence.angular_rate_highpass_degps
                    ),
                    observed_tag0_x_m=float(observed_board[0, 3]),
                    observed_tag0_y_m=float(observed_board[1, 3]),
                    observed_tag0_z_m=float(observed_board[2, 3]),
                    observed_tag0_yaw_deg=_yaw_deg(observed_board),
                    board_displacement_x_m=float(displacement[0]),
                    board_displacement_y_m=float(displacement[1]),
                    board_displacement_z_m=float(displacement[2]),
                    board_displacement_m=float(np.linalg.norm(displacement)),
                    board_displacement_yaw_deg=wrapped_yaw_difference_deg(
                        _yaw_deg(board_pose),
                        _yaw_deg(observed_board),
                    ),
                )
                candidate_rows.append(value)
            all_rows.extend(candidate_rows)
            metrics.append(
                _bag_metrics(
                    candidate_id,
                    dataset,
                    bag_id,
                    candidate_rows,
                    scope,
                )
            )
            for distance_bin in ("near", "middle", "far"):
                cohort = [
                    row
                    for row in candidate_rows
                    if bool(row["primary_scope"])
                    and _distance_bin(float(row["initial_tag0_depth_m"]))
                    == distance_bin
                ]
                if not cohort:
                    continue
                value = _bag_metrics(candidate_id, dataset, bag_id, cohort, scope)
                value["distance_bin"] = distance_bin
                depth_metrics.append(value)
    pooled_metrics: list[Mapping[str, object]] = []
    for candidate_id in sorted(candidates):
        cohort = [
            row
            for row in all_rows
            if row["candidate_id"] == candidate_id and bool(row["primary_scope"])
        ]
        value = _bag_metrics(candidate_id, dataset, "__pooled__", cohort, scope)
        value["bag_count"] = len(bags)
        pooled_metrics.append(value)
    return SafeDynamicValidationResult(
        rows=tuple(all_rows),
        per_bag_metrics=tuple(metrics),
        depth_metrics=tuple(depth_metrics),
        pooled_metrics=tuple(pooled_metrics),
        retention=tuple(retention_rows),
        common_frame_counts=tuple(common_counts),
    )
