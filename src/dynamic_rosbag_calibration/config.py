"""Configuration loading with the Draft 2.2 invariants enforced in code."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError
from .io import sha256_file


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    return float(value)


_SOLVER_INTEGER_FIELDS = frozenset(
    {
        "deterministic_start_count",
        "max_nfev_per_start",
        "irls_max_iterations",
        "minimum_successful_multistarts",
    }
)
_SOLVER_FLOAT_FIELDS = frozenset(
    {
        "xtol",
        "ftol",
        "gtol",
        "extrinsic_rotation_bound_deg",
        "irls_weight_tolerance",
        "mad_consistency_factor",
        "cutoff_mad_multiplier",
        "cutoff_min",
        "cutoff_max",
        "scaled_condition_number_max",
        "multistart_extrinsic_spread_max_deg",
        "multistart_board_translation_spread_max_m",
        "multistart_board_rotation_spread_max_deg",
    }
)
_COVARIANCE_FLOAT_FIELDS = frozenset(
    {
        "pixel_sigma_floor_px",
        "pixel_sigma_ceiling_px",
        "normal_matrix_rcond",
        "scaled_eigenvalue_min",
        "scaled_eigenvalue_max",
        "unmodelled_translation_floor_m",
        "unmodelled_rotation_floor_deg",
    }
)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str) -> int:
    number = _finite_float(value, name)
    if not number.is_integer() or number <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return int(number)


def _normalize_solver(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = _SOLVER_INTEGER_FIELDS | _SOLVER_FLOAT_FIELDS | {"covariance"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"calibration.solver has unknown fields: {unknown}")
    result: dict[str, Any] = {}
    for field in _SOLVER_INTEGER_FIELDS:
        if field in value:
            result[field] = _integer(
                value[field], f"calibration.solver.{field}"
            )
    for field in _SOLVER_FLOAT_FIELDS:
        if field in value:
            result[field] = _finite_float(
                value[field], f"calibration.solver.{field}"
            )
    covariance = _mapping(
        value.get("covariance"), "calibration.solver.covariance"
    )
    covariance_unknown = sorted(set(covariance) - _COVARIANCE_FLOAT_FIELDS)
    if covariance_unknown:
        raise ConfigError(
            "calibration.solver.covariance has unknown fields: "
            f"{covariance_unknown}"
        )
    normalized_covariance = {
        field: _finite_float(
            covariance[field], f"calibration.solver.covariance.{field}"
        )
        for field in _COVARIANCE_FLOAT_FIELDS
        if field in covariance
    }
    positive_solver_fields = _SOLVER_FLOAT_FIELDS - {
        "irls_weight_tolerance",
        "multistart_extrinsic_spread_max_deg",
        "multistart_board_translation_spread_max_m",
        "multistart_board_rotation_spread_max_deg",
    }
    for field in positive_solver_fields:
        if field in result and result[field] <= 0.0:
            raise ConfigError(f"calibration.solver.{field} must be positive")
    for field in _SOLVER_FLOAT_FIELDS - positive_solver_fields:
        if field in result and result[field] < 0.0:
            raise ConfigError(f"calibration.solver.{field} must be non-negative")
    if result.get("cutoff_max", float("inf")) < result.get("cutoff_min", 0.0):
        raise ConfigError("calibration.solver.cutoff_max must be >= cutoff_min")
    positive_covariance_fields = {
        "pixel_sigma_floor_px",
        "pixel_sigma_ceiling_px",
        "scaled_eigenvalue_min",
        "scaled_eigenvalue_max",
    }
    for field in positive_covariance_fields:
        if field in normalized_covariance and normalized_covariance[field] <= 0.0:
            raise ConfigError(
                f"calibration.solver.covariance.{field} must be positive"
            )
    for field in _COVARIANCE_FLOAT_FIELDS - positive_covariance_fields:
        if field in normalized_covariance and normalized_covariance[field] < 0.0:
            raise ConfigError(
                f"calibration.solver.covariance.{field} must be non-negative"
            )
    if normalized_covariance.get("pixel_sigma_ceiling_px", float("inf")) < (
        normalized_covariance.get("pixel_sigma_floor_px", 0.0)
    ):
        raise ConfigError(
            "calibration.solver.covariance.pixel_sigma_ceiling_px must be "
            ">= pixel_sigma_floor_px"
        )
    if normalized_covariance.get("scaled_eigenvalue_max", float("inf")) < (
        normalized_covariance.get("scaled_eigenvalue_min", 0.0)
    ):
        raise ConfigError(
            "calibration.solver.covariance.scaled_eigenvalue_max must be "
            ">= scaled_eigenvalue_min"
        )
    result["covariance"] = normalized_covariance
    return result


@dataclass(frozen=True)
class MachineConfig:
    serial: str
    camera_serial: str
    tag_rig_serial: str


@dataclass(frozen=True)
class TopicConfig:
    image: str
    odom: str
    inspvax: str
    vision_qc: str
    tf_static: str


@dataclass(frozen=True)
class SessionConfig:
    recording_duration_s: float
    live_status_period_ms: int
    minimum_free_disk_gb: float
    recorder_ready_timeout_s: float
    recorder_ready_poll_period_ms: int
    rosbag_stop_timeout_s: float
    rosbag_terminate_timeout_s: float


@dataclass(frozen=True)
class BagTimingConfig:
    image_maximum_source_gap_ms: float
    odom_maximum_source_gap_ms: float
    inspvax_maximum_source_gap_ms: float


@dataclass(frozen=True)
class LiveQcConfig:
    queue_capacity: int
    flush_timeout_s: float
    roi_xyxy: tuple[int, int, int, int]
    detector_nthreads: int
    quad_decimate: float
    quad_sigma: float
    refine_edges: int
    decode_sharpening: float
    maximum_translation_error_m: float
    maximum_rotation_error_deg: float


@dataclass(frozen=True)
class StationGateConfig:
    duration_tolerance_s: float
    required_ins_solution_status: str
    required_ins_position_type: str
    maximum_inspvax_sample_gap_ms: float
    maximum_ins_xy_span_m: float
    maximum_ins_yaw_span_deg: float
    maximum_speed_p95_mps: float
    maximum_time_association_p95_ms: float
    minimum_tag_edge_px: float
    minimum_image_or_roi_margin_px: float
    minimum_f0_valid_frames: int
    minimum_retained_fraction: float
    maximum_work_distance_m: float


@dataclass(frozen=True)
class CollectionGateConfig:
    minimum_qualified_stations: int
    preferred_qualified_stations: int
    start_calibration_when_preferred_ready: bool
    additional_station_policy: str
    distance_bin_edges_m: tuple[float, float]
    minimum_lateral_bearing_span_deg: float
    require_distance_bins: tuple[str, ...]


@dataclass(frozen=True)
class OutputConfig:
    session_root: Path
    approved_root: Path
    rejected_root: Path
    overwrite_existing: bool
    write_runtime_config: bool


@dataclass(frozen=True)
class TagGeometryRunConfig:
    initialization_mode: str
    reuse_initial_path: Path | None
    optimizer: Mapping[str, Any]


@dataclass(frozen=True)
class CalibrationRunConfig:
    machine_config_dir: Path
    board_instance_id: str
    tag_geometry: TagGeometryRunConfig
    solver: Mapping[str, Any]
    compliance: Mapping[str, Any]


@dataclass(frozen=True)
class OnlineCalibrationConfig:
    schema_version: int
    source_fingerprint: str
    machine: MachineConfig
    topics: TopicConfig
    session: SessionConfig
    bag_timing: BagTimingConfig
    live_qc: LiveQcConfig
    station_gate: StationGateConfig
    collection_gate: CollectionGateConfig
    output: OutputConfig
    calibration: CalibrationRunConfig
    record_topics: tuple[str, ...]

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ConfigError("only schema_version 2 is supported")
        if abs(self.session.recording_duration_s - 10.0) > 1e-12:
            raise ConfigError("online_session.recording_duration_s must be 10.0")
        if self.session.recorder_ready_timeout_s <= 0.0:
            raise ConfigError("online_session.recorder_ready_timeout_s must be positive")
        if self.session.recorder_ready_poll_period_ms <= 0:
            raise ConfigError(
                "online_session.recorder_ready_poll_period_ms must be positive"
            )
        timing_limits = (
            self.bag_timing.image_maximum_source_gap_ms,
            self.bag_timing.odom_maximum_source_gap_ms,
            self.bag_timing.inspvax_maximum_source_gap_ms,
        )
        if any(limit <= 0.0 for limit in timing_limits):
            raise ConfigError("bag_timing source gap limits must be positive")
        if self.station_gate.required_ins_solution_status != "INS_SOLUTION_GOOD":
            raise ConfigError("required INS solution status must be INS_SOLUTION_GOOD")
        if self.station_gate.required_ins_position_type != "INS_RTKFIXED":
            raise ConfigError("required INS position type must be INS_RTKFIXED")
        collection = self.collection_gate
        if collection.minimum_qualified_stations != 6:
            raise ConfigError("minimum_qualified_stations must be 6")
        if collection.preferred_qualified_stations != 6:
            raise ConfigError("preferred_qualified_stations must be 6")
        if not collection.start_calibration_when_preferred_ready:
            raise ConfigError("six qualified stations must enable calibration immediately")
        if collection.additional_station_policy != "only_after_coverage_or_quality_gate_failure":
            raise ConfigError("additional stations may only be requested after a gate failure")
        if collection.minimum_lateral_bearing_span_deg != 4.0:
            raise ConfigError("minimum_lateral_bearing_span_deg must be 4.0")
        if self.output.overwrite_existing:
            raise ConfigError("overwriting an existing result is forbidden")
        if self.output.write_runtime_config:
            raise ConfigError("the calibration process may not write the active runtime config")
        if not self.calibration.board_instance_id:
            raise ConfigError("calibration.board_instance_id must be non-empty")
        if not str(self.calibration.machine_config_dir):
            raise ConfigError("calibration.machine_config_dir must be non-empty")
        if _number(
            self.calibration.compliance.get("position_limit_m"),
            "calibration.compliance.position_limit_m",
        ) != 0.03:
            raise ConfigError("calibration.compliance.position_limit_m must be 0.03")
        if _number(
            self.calibration.compliance.get("yaw_limit_deg"),
            "calibration.compliance.yaw_limit_deg",
        ) != 0.5:
            raise ConfigError("calibration.compliance.yaw_limit_deg must be 0.5")
        geometry = self.calibration.tag_geometry
        if geometry.initialization_mode not in {"bootstrap", "reuse"}:
            raise ConfigError(
                "calibration.tag_geometry.initialization_mode must be bootstrap or reuse"
            )
        if geometry.initialization_mode == "reuse" and geometry.reuse_initial_path is None:
            raise ConfigError(
                "calibration.tag_geometry.reuse_initial_path is required in reuse mode"
            )
        if geometry.initialization_mode == "bootstrap" and geometry.reuse_initial_path is not None:
            raise ConfigError(
                "calibration.tag_geometry.reuse_initial_path is forbidden in bootstrap mode"
            )
        if int(geometry.optimizer.get("minimum_valid_stations", 0)) != 6:
            raise ConfigError("calibration.tag_geometry.minimum_valid_stations must be 6")
        if int(self.calibration.solver.get("deterministic_start_count", 0)) != 21:
            raise ConfigError("calibration.solver.deterministic_start_count must be 21")
        if int(self.calibration.solver.get("minimum_successful_multistarts", 0)) != 21:
            raise ConfigError(
                "calibration.solver.minimum_successful_multistarts must be 21"
            )
        if self.station_gate.maximum_inspvax_sample_gap_ms <= 0.0:
            raise ConfigError("maximum_inspvax_sample_gap_ms must be positive")
        if self.station_gate.maximum_ins_xy_span_m != 0.025:
            raise ConfigError("maximum_ins_xy_span_m must be 0.025")
        if self.station_gate.minimum_tag_edge_px != 15.0:
            raise ConfigError("minimum_tag_edge_px must be 15.0")
        if self.live_qc.queue_capacity <= 0:
            raise ConfigError("live_qc.queue_capacity must be positive")
        if self.live_qc.flush_timeout_s <= 0.0:
            raise ConfigError("live_qc.flush_timeout_s must be positive")
        x0, y0, x1, y1 = self.live_qc.roi_xyxy
        if min(x0, y0) < 0 or x0 >= x1 or y0 >= y1:
            raise ConfigError("live_qc.roi_xyxy must be a non-empty non-negative ROI")
        if self.live_qc.detector_nthreads <= 0:
            raise ConfigError("live_qc.detector_nthreads must be positive")
        if self.live_qc.quad_decimate <= 0.0 or self.live_qc.quad_sigma < 0.0:
            raise ConfigError("live_qc detector scale/sigma values are invalid")
        if self.live_qc.refine_edges not in (0, 1):
            raise ConfigError("live_qc.refine_edges must be 0 or 1")
        if self.live_qc.maximum_translation_error_m != 0.10:
            raise ConfigError("live_qc.maximum_translation_error_m must be 0.10")
        if self.live_qc.maximum_rotation_error_deg != 2.0:
            raise ConfigError("live_qc.maximum_rotation_error_deg must be 2.0")
        if not 0.0 < self.station_gate.minimum_retained_fraction <= 1.0:
            raise ConfigError("minimum_retained_fraction must be in (0, 1]")
        required_topics = {self.topics.image, self.topics.odom, self.topics.inspvax}
        if not required_topics.issubset(set(self.record_topics)):
            raise ConfigError("record_topics must contain image, odom, and inspvax topics")


def load_config(path: str | Path) -> OnlineCalibrationConfig:
    config_path = Path(path)
    payload = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "root")
    machine = _mapping(payload.get("machine"), "machine")
    topics = _mapping(payload.get("topics"), "topics")
    session = _mapping(payload.get("online_session"), "online_session")
    bag_timing = _mapping(payload.get("bag_timing"), "bag_timing")
    live_qc = _mapping(payload.get("live_qc"), "live_qc")
    station = _mapping(payload.get("station_gate"), "station_gate")
    collection = _mapping(payload.get("collection_gate"), "collection_gate")
    output = _mapping(payload.get("output"), "output")
    calibration = _mapping(payload.get("calibration"), "calibration")
    tag_geometry = _mapping(calibration.get("tag_geometry"), "calibration.tag_geometry")
    initialization_mode = tag_geometry.get("initialization_mode")
    if not isinstance(initialization_mode, str):
        raise ConfigError(
            "calibration.tag_geometry.initialization_mode must be bootstrap or reuse"
        )
    reuse_initial_value = tag_geometry.get("reuse_initial_path")
    reuse_initial_path = (
        Path(reuse_initial_value)
        if isinstance(reuse_initial_value, str) and reuse_initial_value
        else None
    )
    optimizer = dict(tag_geometry)
    optimizer.pop("initialization_mode", None)
    optimizer.pop("reuse_initial_path", None)
    solver = _normalize_solver(
        _mapping(calibration.get("solver"), "calibration.solver")
    )
    compliance = _mapping(calibration.get("compliance"), "calibration.compliance")
    edges = collection.get("distance_bin_edges_m")
    if not isinstance(edges, list) or len(edges) != 2:
        raise ConfigError("collection_gate.distance_bin_edges_m must contain two values")
    roi = live_qc.get("roi_xyxy")
    if not isinstance(roi, list) or len(roi) != 4:
        raise ConfigError("live_qc.roi_xyxy must contain four integers")
    record_topics = payload.get("record_topics")
    if not isinstance(record_topics, list) or not all(isinstance(item, str) for item in record_topics):
        raise ConfigError("record_topics must be a list of topic names")
    value = OnlineCalibrationConfig(
        schema_version=int(payload.get("schema_version", 0)),
        source_fingerprint=sha256_file(config_path),
        machine=MachineConfig(
            serial=str(machine.get("serial", "")),
            camera_serial=str(machine.get("camera_serial", "")),
            tag_rig_serial=str(machine.get("tag_rig_serial", "")),
        ),
        topics=TopicConfig(**{name: str(topics.get(name, "")) for name in TopicConfig.__annotations__}),
        session=SessionConfig(
            recording_duration_s=_number(session.get("recording_duration_s"), "recording_duration_s"),
            live_status_period_ms=int(session.get("live_status_period_ms", 500)),
            minimum_free_disk_gb=_number(session.get("minimum_free_disk_gb"), "minimum_free_disk_gb"),
            recorder_ready_timeout_s=_number(
                session.get("recorder_ready_timeout_s"),
                "online_session.recorder_ready_timeout_s",
            ),
            recorder_ready_poll_period_ms=int(
                session.get("recorder_ready_poll_period_ms", 0)
            ),
            rosbag_stop_timeout_s=_number(session.get("rosbag_stop_timeout_s"), "rosbag_stop_timeout_s"),
            rosbag_terminate_timeout_s=_number(session.get("rosbag_terminate_timeout_s"), "rosbag_terminate_timeout_s"),
        ),
        bag_timing=BagTimingConfig(
            image_maximum_source_gap_ms=_number(
                bag_timing.get("image_maximum_source_gap_ms"),
                "bag_timing.image_maximum_source_gap_ms",
            ),
            odom_maximum_source_gap_ms=_number(
                bag_timing.get("odom_maximum_source_gap_ms"),
                "bag_timing.odom_maximum_source_gap_ms",
            ),
            inspvax_maximum_source_gap_ms=_number(
                bag_timing.get("inspvax_maximum_source_gap_ms"),
                "bag_timing.inspvax_maximum_source_gap_ms",
            ),
        ),
        live_qc=LiveQcConfig(
            queue_capacity=int(live_qc.get("queue_capacity", 0)),
            flush_timeout_s=_number(live_qc.get("flush_timeout_s"), "live_qc.flush_timeout_s"),
            roi_xyxy=tuple(int(item) for item in roi),
            detector_nthreads=int(live_qc.get("detector_nthreads", 0)),
            quad_decimate=_number(live_qc.get("quad_decimate"), "live_qc.quad_decimate"),
            quad_sigma=_number(live_qc.get("quad_sigma"), "live_qc.quad_sigma"),
            refine_edges=int(live_qc.get("refine_edges", -1)),
            decode_sharpening=_number(live_qc.get("decode_sharpening"), "live_qc.decode_sharpening"),
            maximum_translation_error_m=_number(live_qc.get("maximum_translation_error_m"), "live_qc.maximum_translation_error_m"),
            maximum_rotation_error_deg=_number(live_qc.get("maximum_rotation_error_deg"), "live_qc.maximum_rotation_error_deg"),
        ),
        station_gate=StationGateConfig(
            duration_tolerance_s=_number(station.get("duration_tolerance_s"), "duration_tolerance_s"),
            required_ins_solution_status=str(station.get("required_ins_solution_status", "")),
            required_ins_position_type=str(station.get("required_ins_position_type", "")),
            maximum_inspvax_sample_gap_ms=_number(station.get("maximum_inspvax_sample_gap_ms"), "maximum_inspvax_sample_gap_ms"),
            maximum_ins_xy_span_m=_number(station.get("maximum_ins_xy_span_m"), "maximum_ins_xy_span_m"),
            maximum_ins_yaw_span_deg=_number(station.get("maximum_ins_yaw_span_deg"), "maximum_ins_yaw_span_deg"),
            maximum_speed_p95_mps=_number(station.get("maximum_speed_p95_mps"), "maximum_speed_p95_mps"),
            maximum_time_association_p95_ms=_number(station.get("maximum_time_association_p95_ms"), "maximum_time_association_p95_ms"),
            minimum_tag_edge_px=_number(station.get("minimum_tag_edge_px"), "minimum_tag_edge_px"),
            minimum_image_or_roi_margin_px=_number(station.get("minimum_image_or_roi_margin_px"), "minimum_image_or_roi_margin_px"),
            minimum_f0_valid_frames=int(station.get("minimum_f0_valid_frames", 0)),
            minimum_retained_fraction=_number(station.get("minimum_retained_fraction"), "minimum_retained_fraction"),
            maximum_work_distance_m=_number(station.get("maximum_work_distance_m"), "maximum_work_distance_m"),
        ),
        collection_gate=CollectionGateConfig(
            minimum_qualified_stations=int(collection.get("minimum_qualified_stations", 0)),
            preferred_qualified_stations=int(collection.get("preferred_qualified_stations", 0)),
            start_calibration_when_preferred_ready=bool(collection.get("start_calibration_when_preferred_ready", False)),
            additional_station_policy=str(collection.get("additional_station_policy", "")),
            distance_bin_edges_m=(float(edges[0]), float(edges[1])),
            minimum_lateral_bearing_span_deg=_number(
                collection.get("minimum_lateral_bearing_span_deg"),
                "minimum_lateral_bearing_span_deg",
            ),
            require_distance_bins=tuple(str(item) for item in collection.get("require_distance_bins", [])),
        ),
        output=OutputConfig(
            session_root=Path(str(output.get("session_root", ""))),
            approved_root=Path(str(output.get("approved_root", ""))),
            rejected_root=Path(str(output.get("rejected_root", ""))),
            overwrite_existing=bool(output.get("overwrite_existing", False)),
            write_runtime_config=bool(output.get("write_runtime_config", False)),
        ),
        calibration=CalibrationRunConfig(
            machine_config_dir=Path(str(calibration.get("machine_config_dir", ""))),
            board_instance_id=str(calibration.get("board_instance_id", "")),
            tag_geometry=TagGeometryRunConfig(
                initialization_mode=initialization_mode,
                reuse_initial_path=reuse_initial_path,
                optimizer=optimizer,
            ),
            solver=solver,
            compliance=dict(compliance),
        ),
        record_topics=tuple(record_topics),
    )
    value.validate()
    return value
