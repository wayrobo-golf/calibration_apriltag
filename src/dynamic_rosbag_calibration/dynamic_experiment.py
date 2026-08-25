"""Safe-dynamic candidate orchestration over the existing calibration core."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections import Counter
from typing import Callable, Mapping, Sequence

import numpy as np

from .b2_solver import solve_calibration
from .calibration_types import CalibrationProblem, CameraModel, FrameObservation, Transform
from .config import OnlineCalibrationConfig
from .dynamic_experiment_types import DynamicSelection
from .frozen_runtime import FrozenRuntimeSnapshot
from .io import atomic_write_json, canonical_json_bytes, sha256_bytes
from .recovered_calibration import (
    RecoveredCalibrationCandidate,
    _workflow_configs,
    recover_calibration_candidate,
    write_recovery_outputs,
)
from .pnp_prepare import solve_frame_with_frozen_geometry
from .tag_geometry import (
    FrozenTagGeometry,
    TagGeometryConfig,
    TagGeometryResult,
    estimate_tag_geometry,
    select_balanced_views,
)
from .tag_geometry_initialization import (
    TagGeometryInitializationResult,
    initialize_tag_geometry,
)
from .rosbag_problem import _runtime_transform, _single_yaml


ALGORITHM_ID = "B2-SAFE-DYNAMIC-RXY"
NEGATIVE_CONTROL_ALGORITHM_ID = "B2-ALL-DYNAMIC-LE8M-RXY-NEGATIVE-CONTROL"


def conditional_ablation_tag_geometry_audit(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Make frozen-geometry audit semantics explicit and machine-readable."""
    result = dict(value)
    fixed_gate_pass = bool(result.get("quality_gate_pass"))
    result.update(
        estimation_status="G2_NOT_RUN_CONDITIONAL_ABLATION",
        six_station_gate_applicable=False,
        quality_gate_pass=None,
        fixed_geometry_reprojection_gate_pass=fixed_gate_pass,
    )
    return result


def initialize_frozen_reused_tag_geometry(
    frames: Sequence[FrameObservation],
    _camera: CameraModel,
    _tag_size_m: float,
    mode: str,
    reuse_initial: Transform | None,
    source_path: str | Path | None,
    source_sha256: str | None,
    *,
    bootstrap_maximum_views_per_station: int = 5,
    minimum_valid_stations: int = 6,
) -> TagGeometryInitializationResult:
    """Accept approved geometry without imposing the six-station G2 contract."""
    if mode != "reuse" or reuse_initial is None:
        raise ValueError("frozen Tag geometry requires reuse mode and an initial transform")
    if source_path is None or source_sha256 is None:
        raise ValueError("frozen Tag geometry requires source path and SHA-256")
    if bootstrap_maximum_views_per_station <= 0:
        raise ValueError("maximum views per station must be positive")
    if minimum_valid_stations <= 0:
        raise ValueError("minimum valid stations must be positive")
    seeded = tuple(
        frame
        for frame in sorted(frames, key=lambda item: item.frame_key)
        if frame.basic_valid and frame.station_gate_pass
    )
    if not seeded:
        raise ValueError("frozen Tag geometry has no valid input frame")
    return TagGeometryInitializationResult(
        mode="reuse_frozen",
        initial_transform=reuse_initial,
        seeded_frames=seeded,
        source_path=Path(source_path),
        source_sha256=str(source_sha256),
    )


def evaluate_frozen_reused_tag_geometry(
    initialization: TagGeometryInitializationResult,
    camera: CameraModel,
    tag_size_m: float,
    config: TagGeometryConfig,
) -> TagGeometryResult:
    """Audit PnP fit while leaving an approved Tag transform unchanged."""
    balanced = select_balanced_views(
        initialization.seeded_frames,
        config.maximum_views_per_station,
    )
    if not balanced:
        raise ValueError("frozen Tag geometry has no balanced audit view")
    frozen = FrozenTagGeometry(
        initialization.initial_transform,
        tuple(frame.frame_key for frame in balanced),
    )
    prepared = tuple(
        solve_frame_with_frozen_geometry(frame, camera, tag_size_m, frozen)
        for frame in balanced
    )
    retained = tuple(
        item
        for item in prepared
        if item.passed and item.camera_tag0 is not None
    )
    if not retained:
        raise ValueError("approved frozen Tag geometry retained no audit view")
    rms = np.asarray(
        [float(item.dual_reprojection_rms_px) for item in retained],
        dtype=np.float64,
    )
    overall_rms = float(np.sqrt(np.mean(np.square(rms))))
    maximum_view_rms = float(np.max(rms))
    reasons: list[str] = []
    if overall_rms > config.overall_reprojection_rms_max_px:
        reasons.append("CAL-E-TAG-GEOMETRY-OVERALL-RMS")
    if maximum_view_rms > config.maximum_view_rms_max_px:
        reasons.append("CAL-E-TAG-GEOMETRY-VIEW-RMS")
    return TagGeometryResult(
        candidate=initialization.initial_transform,
        frozen=frozen,
        quality_gate_pass=not reasons,
        reasons=tuple(reasons),
        overall_reprojection_rms_px=overall_rms,
        maximum_view_rms_px=maximum_view_rms,
        balanced_view_count=len(balanced),
        valid_station_count=len({frame.station_id for frame in balanced}),
        loo_translation_std_m=0.0,
        loo_rotation_std_deg=0.0,
        loo_translation_difference_max_m=0.0,
        loo_rotation_difference_max_deg=0.0,
        solver_success=True,
        solver_message="approved geometry frozen; no G2 estimation",
        initialization=initialization,
    )


def diagnose_dynamic_tag_geometry(
    problem: CalibrationProblem,
    online_config: OnlineCalibrationConfig,
) -> dict[str, object]:
    """Run G2 and explain frozen-geometry retention without starting B2."""
    geometry_config, _solver_config, _start_count = _workflow_configs(online_config)
    reuse_mode = online_config.calibration.tag_geometry.initialization_mode == "reuse"
    initialization = initialize_tag_geometry(
        problem.frames,
        problem.camera_model,
        problem.tag_size_m,
        online_config.calibration.tag_geometry.initialization_mode,
        problem.initial_tag0_tag1 if reuse_mode else None,
        problem.initial_tag0_tag1_source_path if reuse_mode else None,
        problem.initial_tag0_tag1_source_sha256 if reuse_mode else None,
        bootstrap_maximum_views_per_station=geometry_config.maximum_views_per_station,
        minimum_valid_stations=geometry_config.minimum_valid_stations,
    )
    geometry = estimate_tag_geometry(
        initialization,
        problem.camera_model,
        problem.tag_size_m,
        geometry_config,
    )
    if geometry.frozen is None:
        return {
            "quality_gate_pass": False,
            "reasons": list(geometry.reasons),
            "overall_reprojection_rms_px": geometry.overall_reprojection_rms_px,
            "maximum_view_rms_px": geometry.maximum_view_rms_px,
        }
    source_counts = Counter(frame.station_id for frame in problem.frames)
    retained_counts: Counter[str] = Counter()
    reason_counts: dict[str, Counter[str]] = {
        station_id: Counter() for station_id in source_counts
    }
    for frame in problem.frames:
        prepared = solve_frame_with_frozen_geometry(
            frame,
            problem.camera_model,
            problem.tag_size_m,
            geometry.frozen,
        )
        if prepared.passed and prepared.camera_tag0 is not None:
            retained_counts[frame.station_id] += 1
        else:
            reason_counts[frame.station_id].update(prepared.reasons)
    return {
        "quality_gate_pass": geometry.quality_gate_pass,
        "reasons": list(geometry.reasons),
        "overall_reprojection_rms_px": geometry.overall_reprojection_rms_px,
        "maximum_view_rms_px": geometry.maximum_view_rms_px,
        "T_tag0_tag1": geometry.frozen.transform.matrix.tolist(),
        "per_bag": [
            {
                "bag_id": station_id,
                "source_frame_count": source_counts[station_id],
                "retained_frame_count": retained_counts[station_id],
                "retained_fraction": retained_counts[station_id]
                / source_counts[station_id],
                "reasons": dict(sorted(reason_counts[station_id].items())),
            }
            for station_id in sorted(source_counts)
        ],
    }


def select_b2_problem(
    problem: CalibrationProblem,
    selected_frame_keys: frozenset[str],
) -> CalibrationProblem:
    frames = tuple(
        frame for frame in problem.frames if frame.frame_key in selected_frame_keys
    )
    if not frames:
        raise ValueError("frozen geometry retained no selected dynamic B2 frame")
    return CalibrationProblem(
        frames=frames,
        nominal_extrinsic=problem.nominal_extrinsic,
        initial_tag0_tag1=problem.initial_tag0_tag1,
        tag_size_m=problem.tag_size_m,
        camera_model=problem.camera_model,
        config_fingerprint=problem.config_fingerprint,
        initial_tag0_tag1_source_path=problem.initial_tag0_tag1_source_path,
        initial_tag0_tag1_source_sha256=problem.initial_tag0_tag1_source_sha256,
    )


def _runtime_snapshot(
    problem: CalibrationProblem,
    machine_config_dir: Path,
    identity_sha256: str,
) -> FrozenRuntimeSnapshot:
    payload = _single_yaml(Path(machine_config_dir) / "tf_static_golfC_param.yaml")
    params = payload["static_calibration"]["ros__parameters"]
    ins_lidar = _runtime_transform(params["lidar_to_ins_extrisric"])
    lidar_camera = _runtime_transform(params["left_camera_to_lidar_extrinsic"])
    return FrozenRuntimeSnapshot(
        camera_matrix=problem.camera_model.matrix,
        distortion_coefficients=problem.camera_model.distortion,
        T_ins_lidar=ins_lidar,
        T_lidar_camera=lidar_camera,
        T_ins_camera=problem.nominal_extrinsic.matrix,
        tag_size_m=problem.tag_size_m,
        sources=(),
        identity_sha256=identity_sha256,
    )


def solve_safe_dynamic_candidate(
    problem: CalibrationProblem,
    selection: DynamicSelection,
    *,
    online_config: OnlineCalibrationConfig,
    machine_config_dir: Path,
    output_dir: Path,
    evidence_identity: Mapping[str, object],
    progress_callback: Callable[[str, str], None] | None = None,
    algorithm_id: str = ALGORITHM_ID,
    freeze_reused_tag_geometry: bool = False,
    allow_failed_tag_geometry: bool = False,
    software_fingerprint: str = "fae-safe-dynamic-unspecified",
) -> RecoveredCalibrationCandidate:
    """Prepare Tag geometry, then solve Rx/Ry on balanced dynamic keys."""
    if not selection.coverage.passed:
        raise ValueError(
            f"safe dynamic coverage failed: {selection.coverage.reasons}"
        )
    selected_keys = frozenset(
        item.source.frame_key for item in selection.frames
    )
    evidence_hash = sha256_bytes(canonical_json_bytes(dict(evidence_identity)))
    runtime_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "machine_config_fingerprint": problem.config_fingerprint,
                "evidence_identity_sha256": evidence_hash,
                "selected_frame_keys_sha256": selection.selected_frame_keys_sha256,
            }
        )
    )
    runtime = _runtime_snapshot(problem, machine_config_dir, runtime_hash)
    dataset_id = sha256_bytes(
        canonical_json_bytes(
            {
                "algorithm_id": algorithm_id,
                "evidence_identity_sha256": evidence_hash,
                "selected_frame_keys_sha256": selection.selected_frame_keys_sha256,
            }
        )
    )

    def solve(filtered_problem: CalibrationProblem, solver_config: object):
        b2_problem = select_b2_problem(filtered_problem, selected_keys)
        return solve_calibration(b2_problem, solver_config)

    recovery_options = {}
    if freeze_reused_tag_geometry:
        if online_config.calibration.tag_geometry.initialization_mode != "reuse":
            raise ValueError("frozen Tag geometry is only valid in reuse mode")
        recovery_options = {
            "geometry_initializer": initialize_frozen_reused_tag_geometry,
            "geometry_estimator": evaluate_frozen_reused_tag_geometry,
        }
    candidate = recover_calibration_candidate(
        problem,
        frozen_runtime=runtime,
        config=online_config,
        calibration_dataset_id=dataset_id,
        software_fingerprint=software_fingerprint,
        solve_extrinsic=solve,
        progress_callback=progress_callback,
        allow_failed_tag_geometry=allow_failed_tag_geometry,
        **recovery_options,
    )
    if freeze_reused_tag_geometry:
        audit = dict(candidate.audit)
        audit["tag_geometry"] = conditional_ablation_tag_geometry_audit(
            audit["tag_geometry"]
        )
        candidate = replace(candidate, audit=audit)
    destination = Path(output_dir)
    written, audit_path = write_recovery_outputs(candidate, destination)
    atomic_write_json(
        destination / "experiment_metadata.json",
        {
            "schema_version": 1,
            "algorithm_id": algorithm_id,
            "execution_mode": "offline_experiment",
            "deliverable": False,
            "runtime_installable": False,
            "tag_geometry_mode": (
                "approved_frozen_reuse"
                if freeze_reused_tag_geometry
                else "reestimated_g2"
            ),
            "tag_geometry_estimation_status": (
                "G2_NOT_RUN_CONDITIONAL_ABLATION"
                if freeze_reused_tag_geometry
                else "G2_ESTIMATED"
            ),
            "g2_six_station_gate_applicable": (
                not freeze_reused_tag_geometry
            ),
            "failed_tag_geometry_diagnostic_override_allowed": bool(
                allow_failed_tag_geometry
            ),
            "candidate_path": written.name,
            "audit_path": audit_path.name,
            "evidence_identity_sha256": evidence_hash,
            "selected_frame_keys_sha256": selection.selected_frame_keys_sha256,
            "selected_frame_count_before_frozen_geometry": len(selection.frames),
            "coverage": {
                "passed": selection.coverage.passed,
                "reasons": list(selection.coverage.reasons),
                "bag_count": selection.coverage.bag_count,
                "total_frame_count": selection.coverage.total_frame_count,
                "bearing_span_deg": selection.coverage.bearing_span_deg,
                "distance_bin_bag_counts": [
                    list(item) for item in selection.coverage.distance_bin_bag_counts
                ],
            },
        },
    )
    return candidate
