"""Offline recovery of a quarantined calibration candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from .b2_solver import B2SolverConfig, solve_calibration
from .calibration_types import CalibrationProblem, Transform, readonly_float_array
from .config import OnlineCalibrationConfig
from .covariance import CovarianceConfig
from .frozen_runtime import FrozenRuntimeSnapshot
from .io import atomic_write_json
from .models import StationRecord, StationState
from .pnp_prepare import solve_frame_with_frozen_geometry
from .tag_geometry import (
    FrozenTagGeometry,
    TagGeometryConfig,
    estimate_tag_geometry,
    select_balanced_views,
)
from .tag_geometry_initialization import initialize_tag_geometry


RECOVERED_STATUS = "RECOVERED_NON_INSTALLABLE"
PROHIBITED_OUTPUT_PARTS = frozenset({"approved", "rejected", "config"})


def load_session_station_records(
    session_dir: Path,
    *,
    evidence_root: Path,
) -> tuple[StationRecord, ...]:
    session_path = Path(session_dir).resolve()
    root = Path(evidence_root).resolve()
    manifest_path = session_path / "session_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("recovery requires a schema_version 2 session manifest")
    values = payload.get("stations")
    if not isinstance(values, list):
        raise ValueError("session manifest stations must be a list")
    records: list[StationRecord] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("session station entry must be a mapping")
        state = StationState(str(item.get("state", "")))
        if state not in (
            StationState.PROVISIONAL_ACCEPTED,
            StationState.ACCEPTED,
        ):
            continue
        bag_value = item.get("bag_path")
        if not isinstance(bag_value, str) or not bag_value:
            raise ValueError(f"{item.get('station_id')} has no bag_path")
        bag_path = Path(bag_value)
        if not bag_path.is_absolute():
            bag_path = root / bag_path
        bag_path = bag_path.resolve()
        if not bag_path.is_dir():
            raise ValueError(f"station rosbag does not exist: {bag_path}")
        started_at_s = item.get("started_at_s")
        ended_at_s = item.get("ended_at_s")
        if (
            isinstance(started_at_s, bool)
            or isinstance(ended_at_s, bool)
            or not isinstance(started_at_s, (int, float))
            or not isinstance(ended_at_s, (int, float))
            or not np.isfinite(started_at_s)
            or not np.isfinite(ended_at_s)
            or float(ended_at_s) <= float(started_at_s)
        ):
            raise ValueError(
                f"{item.get('station_id')} has an invalid recording window"
            )
        records.append(
            StationRecord(
                station_id=str(item.get("station_id", "")),
                board_instance_id=str(item.get("board_instance_id", "")),
                state=state,
                requested_at_s=float(item.get("requested_at_s", started_at_s)),
                started_at_s=float(started_at_s),
                ended_at_s=float(ended_at_s),
                bag_path=bag_path,
                raw_data_identity_sha256=(
                    str(item["raw_data_identity_sha256"])
                    if item.get("raw_data_identity_sha256") is not None
                    else None
                ),
                raw_data_size_bytes=(
                    int(item["raw_data_size_bytes"])
                    if item.get("raw_data_size_bytes") is not None
                    else None
                ),
                distance_bin=(
                    str(item["distance_bin"])
                    if item.get("distance_bin") is not None
                    else None
                ),
                lateral_bin=(
                    str(item["lateral_bin"])
                    if item.get("lateral_bin") is not None
                    else None
                ),
            )
        )
    records.sort(key=lambda item: item.station_id)
    if len(records) != 6:
        raise ValueError(
            f"recovery requires exactly six accepted station bags, got {len(records)}"
        )
    if len({item.station_id for item in records}) != len(records):
        raise ValueError("session manifest contains duplicate accepted station IDs")
    return tuple(records)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("recovery output contains a non-finite number")
    return value


@dataclass(frozen=True)
class RecoveredCalibrationCandidate:
    status: str
    calibration_dataset_id: str
    T_ins_camera_nominal: np.ndarray
    T_ins_camera_calibrated: np.ndarray
    T_ins_map_tag0_calibrated: np.ndarray
    T_tag0_tag1_calibrated: np.ndarray
    right_correction_rotvec_rad: tuple[float, float, float]
    best_start_name: str
    objective_cost: float
    observability: Mapping[str, Any]
    covariance: Mapping[str, Any]
    multistart_spread: Mapping[str, float]
    multistart_runs: tuple[Mapping[str, Any], ...]
    quality_gate_pass: bool
    quality_gate_reasons: tuple[str, ...]
    config_fingerprint: str
    software_fingerprint: str
    metrics: Mapping[str, Any]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status != RECOVERED_STATUS:
            raise ValueError(f"recovered candidate status must be {RECOVERED_STATUS}")
        if not self.calibration_dataset_id:
            raise ValueError("calibration_dataset_id must be non-empty")
        matrices = {
            "T_ins_camera_nominal": (
                self.T_ins_camera_nominal,
                "ins_link",
                "left_camera",
            ),
            "T_ins_camera_calibrated": (
                self.T_ins_camera_calibrated,
                "ins_link",
                "left_camera",
            ),
            "T_ins_map_tag0_calibrated": (
                self.T_ins_map_tag0_calibrated,
                "ins_map",
                "tag0",
            ),
            "T_tag0_tag1_calibrated": (
                self.T_tag0_tag1_calibrated,
                "tag0",
                "tag1",
            ),
        }
        for name, (matrix, parent, child) in matrices.items():
            validated = Transform(matrix, parent, child).matrix
            object.__setattr__(self, name, validated)
        correction = tuple(float(value) for value in self.right_correction_rotvec_rad)
        if len(correction) != 3 or not all(np.isfinite(correction)):
            raise ValueError("right_correction_rotvec_rad must contain three finite values")
        if not np.isfinite(self.objective_cost):
            raise ValueError("objective_cost must be finite")
        if not self.config_fingerprint or not self.software_fingerprint:
            raise ValueError("config and software fingerprints must be non-empty")
        object.__setattr__(self, "right_correction_rotvec_rad", correction)
        _json_value(self)

    def payload(self) -> dict[str, Any]:
        result = _json_value(asdict(self))
        result["schema_version"] = 1
        result["runtime_installable"] = False
        result["candidate_warning"] = (
            "RECOVERED OFFLINE CANDIDATE - DO NOT INSTALL OR WRITE RUNTIME CONFIG"
        )
        result["transform_semantics"] = {
            "T_ins_camera_nominal": "p_ins = T_ins_camera_nominal * p_camera",
            "T_ins_camera_calibrated": (
                "p_ins = T_ins_camera_calibrated * p_camera"
            ),
            "T_ins_map_tag0_calibrated": (
                "p_ins_map = T_ins_map_tag0_calibrated * p_tag0"
            ),
            "T_tag0_tag1_calibrated": (
                "p_tag0 = T_tag0_tag1_calibrated * p_tag1"
            ),
        }
        return result


def _workflow_configs(
    config: OnlineCalibrationConfig,
) -> tuple[TagGeometryConfig, B2SolverConfig, int]:
    geometry_config = TagGeometryConfig(
        **dict(config.calibration.tag_geometry.optimizer)
    )
    solver_values = dict(config.calibration.solver)
    deterministic_start_count = int(
        solver_values.pop("deterministic_start_count")
    )
    if deterministic_start_count != 21:
        raise ValueError("recovery requires exactly 21 deterministic starts")
    covariance = CovarianceConfig(**dict(solver_values.pop("covariance")))
    solver_config = B2SolverConfig(covariance=covariance, **solver_values)
    return geometry_config, solver_config, deterministic_start_count


def _observability_payload(value: Any) -> dict[str, Any]:
    serialized = _json_value(value)
    if not isinstance(serialized, dict):
        raise ValueError("observability result must serialize to a mapping")
    return serialized


def _multistart_payload(value: Any) -> dict[str, Any]:
    return {
        "name": str(value.name),
        "success": bool(value.success),
        "cost": float(value.cost),
        "nfev": int(getattr(value, "nfev", 0)),
        "message": str(getattr(value, "message", "")),
        "covariance_invalid_count": int(
            getattr(value, "covariance_invalid_count", 0)
        ),
    }


def recover_calibration_candidate(
    problem: CalibrationProblem,
    *,
    frozen_runtime: FrozenRuntimeSnapshot,
    config: OnlineCalibrationConfig,
    calibration_dataset_id: str,
    software_fingerprint: str,
    config_fingerprint: str | None = None,
    geometry_initializer: Callable[..., Any] = initialize_tag_geometry,
    geometry_estimator: Callable[..., Any] = estimate_tag_geometry,
    frame_preparer: Callable[..., Any] = solve_frame_with_frozen_geometry,
    solve_extrinsic: Callable[..., Any] = solve_calibration,
    progress_callback: Callable[[str, str], None] | None = None,
    allow_failed_tag_geometry: bool = False,
) -> RecoveredCalibrationCandidate:
    if not np.allclose(
        problem.camera_model.matrix,
        frozen_runtime.camera_matrix,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("problem camera matrix does not match frozen runtime")
    if not np.allclose(
        problem.camera_model.distortion,
        frozen_runtime.distortion_coefficients,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("problem camera distortion does not match frozen runtime")
    if not np.isclose(problem.tag_size_m, frozen_runtime.tag_size_m, atol=1.0e-12):
        raise ValueError("problem tag size does not match frozen runtime")
    if not calibration_dataset_id or not software_fingerprint:
        raise ValueError("dataset and software fingerprints must be non-empty")
    geometry_config, solver_config, start_count = _workflow_configs(config)
    timings: dict[str, dict[str, float]] = {}

    def stage(name: str, function: Callable[[], Any]) -> Any:
        started = time.monotonic()
        if progress_callback is not None:
            progress_callback(name, "started")
        try:
            return function()
        finally:
            ended = time.monotonic()
            timings[name] = {
                "started_monotonic_s": started,
                "ended_monotonic_s": ended,
                "duration_s": ended - started,
            }
            if progress_callback is not None:
                progress_callback(name, "finished")

    nominal = Transform(
        frozen_runtime.T_ins_camera,
        "ins_link",
        "left_camera",
    )
    frozen_nominal_problem = CalibrationProblem(
        frames=problem.frames,
        nominal_extrinsic=nominal,
        initial_tag0_tag1=problem.initial_tag0_tag1,
        tag_size_m=problem.tag_size_m,
        camera_model=problem.camera_model,
        config_fingerprint=problem.config_fingerprint,
        initial_tag0_tag1_source_path=problem.initial_tag0_tag1_source_path,
        initial_tag0_tag1_source_sha256=problem.initial_tag0_tag1_source_sha256,
    )
    reuse_mode = config.calibration.tag_geometry.initialization_mode == "reuse"
    initialization = stage(
        "tag_geometry_initialization",
        lambda: geometry_initializer(
            frozen_nominal_problem.frames,
            frozen_nominal_problem.camera_model,
            frozen_nominal_problem.tag_size_m,
            config.calibration.tag_geometry.initialization_mode,
            frozen_nominal_problem.initial_tag0_tag1 if reuse_mode else None,
            (
                frozen_nominal_problem.initial_tag0_tag1_source_path
                if reuse_mode
                else None
            ),
            (
                frozen_nominal_problem.initial_tag0_tag1_source_sha256
                if reuse_mode
                else None
            ),
            bootstrap_maximum_views_per_station=(
                geometry_config.maximum_views_per_station
            ),
            minimum_valid_stations=geometry_config.minimum_valid_stations,
        ),
    )
    geometry = stage(
        "tag_geometry_estimation",
        lambda: geometry_estimator(
            initialization,
            frozen_nominal_problem.camera_model,
            frozen_nominal_problem.tag_size_m,
            geometry_config,
        ),
    )
    diagnostic_geometry_override = False
    if not geometry.quality_gate_pass or geometry.frozen is None:
        if not allow_failed_tag_geometry:
            raise ValueError(
                "recovered tag geometry failed quality gate: "
                f"{tuple(geometry.reasons)}"
            )
        if not geometry.solver_success:
            raise ValueError(
                "diagnostic recovery cannot continue after G2 solver failure"
            )
        balanced = select_balanced_views(
            initialization.seeded_frames,
            geometry_config.maximum_views_per_station,
        )
        geometry = replace(
            geometry,
            frozen=FrozenTagGeometry(
                geometry.candidate,
                tuple(frame.frame_key for frame in balanced),
            ),
        )
        diagnostic_geometry_override = True
    prepared = stage(
        "frozen_geometry_recheck",
        lambda: tuple(
            frame_preparer(
                frame,
                frozen_nominal_problem.camera_model,
                frozen_nominal_problem.tag_size_m,
                geometry.frozen,
            )
            for frame in frozen_nominal_problem.frames
        ),
    )
    retained = tuple(
        replace(item.source, camera_tag0=item.camera_tag0)
        for item in prepared
        if item.passed and item.camera_tag0 is not None
    )
    source_counts: dict[str, int] = {}
    retained_counts: dict[str, int] = {}
    for frame in frozen_nominal_problem.frames:
        source_counts[frame.station_id] = source_counts.get(frame.station_id, 0) + 1
    for frame in retained:
        retained_counts[frame.station_id] = retained_counts.get(frame.station_id, 0) + 1
    failed_fraction = tuple(
        station_id
        for station_id, count in sorted(source_counts.items())
        if retained_counts.get(station_id, 0) / count
        < config.station_gate.minimum_retained_fraction
    )
    if failed_fraction:
        raise ValueError(
            "frozen geometry retained fraction failed for stations: "
            f"{failed_fraction}"
        )
    frozen_problem = CalibrationProblem(
        frames=retained,
        nominal_extrinsic=nominal,
        initial_tag0_tag1=geometry.frozen.transform,
        tag_size_m=frozen_nominal_problem.tag_size_m,
        camera_model=frozen_nominal_problem.camera_model,
        config_fingerprint=frozen_nominal_problem.config_fingerprint,
    )
    calibration = stage(
        "b2_multistart",
        lambda: solve_extrinsic(frozen_problem, solver_config),
    )
    board_id = config.calibration.board_instance_id
    if board_id not in calibration.board_poses:
        raise ValueError(f"B2 result has no board pose for {board_id}")
    quality_reasons = tuple(
        dict.fromkeys(tuple(geometry.reasons) + tuple(calibration.reasons))
    )
    multistart_runs = tuple(
        _multistart_payload(value) for value in calibration.multistart_results
    )
    audit = {
        "schema_version": 1,
        "status": RECOVERED_STATUS,
        "runtime_installable": False,
        "post_hoc_alignment_applied": False,
        "frozen_runtime_identity_sha256": frozen_runtime.identity_sha256,
        "deterministic_start_count": start_count,
        "bootstrap_frame_count": len(
            select_balanced_views(
                frozen_nominal_problem.frames,
                geometry_config.maximum_views_per_station,
            )
        ),
        "seed_frame_count": len(initialization.seeded_frames),
        "source_frame_count": len(frozen_nominal_problem.frames),
        "retained_frame_count": len(retained),
        "retained_frames_per_station": retained_counts,
        "stage_timings": timings,
        "tag_geometry": {
            "quality_gate_pass": bool(geometry.quality_gate_pass),
            "diagnostic_override": diagnostic_geometry_override,
            "reasons": list(geometry.reasons),
            "overall_reprojection_rms_px": float(
                geometry.overall_reprojection_rms_px
            ),
            "maximum_view_rms_px": float(geometry.maximum_view_rms_px),
            "balanced_view_count": int(geometry.balanced_view_count),
            "valid_station_count": int(geometry.valid_station_count),
            "loo_translation_std_m": float(geometry.loo_translation_std_m),
            "loo_rotation_std_deg": float(geometry.loo_rotation_std_deg),
            "loo_translation_difference_max_m": float(
                geometry.loo_translation_difference_max_m
            ),
            "loo_rotation_difference_max_deg": float(
                geometry.loo_rotation_difference_max_deg
            ),
        },
        "b2": {
            "quality_gate_pass": bool(calibration.quality_gate_pass),
            "reasons": list(calibration.reasons),
            "best_start_name": str(calibration.best_start_name),
            "objective_cost": float(calibration.objective_cost),
        },
    }
    covariance = {
        "config": asdict(solver_config.covariance),
        "invalid_frame_count": int(calibration.covariance_invalid_count),
        "note": (
            "B2 stores per-frame PnP covariance validity diagnostics; it does "
            "not currently expose a final parameter covariance matrix"
        ),
    }
    metrics = {
        "source_frame_count": len(frozen_nominal_problem.frames),
        "retained_frame_count": len(retained),
        "retained_fraction": len(retained) / len(frozen_nominal_problem.frames),
        "tag_geometry_overall_reprojection_rms_px": float(
            geometry.overall_reprojection_rms_px
        ),
        "tag_geometry_maximum_view_rms_px": float(
            geometry.maximum_view_rms_px
        ),
        "huber_cutoff": float(calibration.huber_cutoff),
    }
    return RecoveredCalibrationCandidate(
        status=RECOVERED_STATUS,
        calibration_dataset_id=calibration_dataset_id,
        T_ins_camera_nominal=nominal.matrix,
        T_ins_camera_calibrated=calibration.effective_extrinsic.matrix,
        T_ins_map_tag0_calibrated=calibration.board_poses[board_id].matrix,
        T_tag0_tag1_calibrated=geometry.frozen.transform.matrix,
        right_correction_rotvec_rad=tuple(
            calibration.right_correction_rotvec_rad
        ),
        best_start_name=str(calibration.best_start_name),
        objective_cost=float(calibration.objective_cost),
        observability=_observability_payload(calibration.observability),
        covariance=covariance,
        multistart_spread={
            "extrinsic_deg": float(calibration.extrinsic_multistart_spread_deg),
            "board_translation_m": float(
                calibration.board_translation_multistart_spread_m
            ),
            "board_rotation_deg": float(
                calibration.board_rotation_multistart_spread_deg
            ),
        },
        multistart_runs=multistart_runs,
        quality_gate_pass=bool(
            geometry.quality_gate_pass and calibration.quality_gate_pass
        ),
        quality_gate_reasons=quality_reasons,
        config_fingerprint=(config_fingerprint or config.source_fingerprint),
        software_fingerprint=software_fingerprint,
        metrics=metrics,
        audit=audit,
    )


def _atomic_write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    payload = yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        sort_keys=False,
    )
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_recovery_outputs(
    candidate: RecoveredCalibrationCandidate,
    output_dir: Path,
) -> tuple[Path, Path]:
    destination = Path(output_dir).resolve()
    if any(part.lower() in PROHIBITED_OUTPUT_PARTS for part in destination.parts):
        raise ValueError("recovery outputs must stay in a quarantine directory")
    if candidate.status != RECOVERED_STATUS:
        raise ValueError("only RECOVERED_NON_INSTALLABLE candidates may be written")
    candidate_path = destination / "recovered_candidate.yaml"
    audit_path = destination / "recovery_audit.json"
    _atomic_write_yaml(candidate_path, candidate.payload())
    atomic_write_json(audit_path, _json_value(candidate.audit))
    return candidate_path, audit_path
