"""Single-file configuration contract for dynamic rosbag calibration."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping

import yaml

from .config import OnlineCalibrationConfig, load_config
from .dynamic_experiment_config import load_dynamic_experiment_config
from .dynamic_experiment_types import DynamicExperimentConfig
from .io import sha256_file


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _StrictLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise ValueError(f"{name} has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")


def _path(value: object, source_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (source_dir / path).resolve()


@dataclass(frozen=True)
class RosbagInput:
    bag_id: str
    path: Path
    role: str
    reason: str | None = None


@dataclass(frozen=True)
class PreparedRuntime:
    online: OnlineCalibrationConfig
    experiment: DynamicExperimentConfig
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class CalibrationJobConfig:
    source_path: Path
    source_fingerprint: str
    job_id: str
    dataset_date: str
    report_title: str
    novatel_message_dir: Path
    output_dir: Path
    inventory_only: bool
    skip_validation: bool
    bags: tuple[RosbagInput, ...]
    machine_config_dir: Path
    online_payload: Mapping[str, object]
    experiment_payload: Mapping[str, object]
    manifest: Mapping[str, object]

    @property
    def bag_paths(self) -> Mapping[str, Path]:
        return {item.bag_id: item.path for item in self.bags if item.role != "excluded"}

    @contextmanager
    def prepare_runtime(self) -> Iterator[PreparedRuntime]:
        """Build the legacy in-memory views consumed by the migrated algorithm."""
        with tempfile.TemporaryDirectory(prefix="dynamic-calibration-config-") as directory:
            root = Path(directory)
            online_path = root / "online.yaml"
            experiment_path = root / "experiment.yaml"
            loader_online = deepcopy(self.online_payload)
            loader_calibration = loader_online["calibration"]
            loader_calibration["tag_geometry"]["initialization_mode"] = "bootstrap"
            loader_calibration["tag_geometry"].pop("reuse_initial_path", None)
            loader_calibration["tag_geometry"]["minimum_valid_stations"] = 6
            loader_calibration["compliance"]["position_limit_m"] = 0.03
            loader_calibration["compliance"]["yaw_limit_deg"] = 0.5
            online_path.write_text(
                yaml.safe_dump(loader_online, sort_keys=False),
                encoding="utf-8",
            )
            experiment_path.write_text(
                yaml.safe_dump(dict(self.experiment_payload), sort_keys=False),
                encoding="utf-8",
            )
            online = load_config(online_path)
            online = replace(
                online,
                calibration=replace(
                    online.calibration,
                    compliance=deepcopy(self.online_payload["calibration"]["compliance"]),
                ),
            )
            yield PreparedRuntime(
                online=online,
                experiment=load_dynamic_experiment_config(experiment_path),
                manifest=deepcopy(self.manifest),
            )


def _load_bags(
    value: Mapping[str, Any],
    source_dir: Path,
) -> tuple[RosbagInput, ...]:
    _exact(
        value,
        {"calibration", "validation", "displacement_evaluation", "excluded"},
        "rosbags",
    )
    result: list[RosbagInput] = []
    for role in ("calibration", "validation", "displacement_evaluation", "excluded"):
        entries = value[role]
        if not isinstance(entries, list):
            raise ValueError(f"rosbags.{role} must be a list")
        for index, raw in enumerate(entries):
            entry = _mapping(raw, f"rosbags.{role}[{index}]")
            fields = {"id", "path", "reason"} if role == "excluded" else {"id", "path"}
            _exact(entry, fields, f"rosbags.{role}[{index}]")
            bag_id = str(entry["id"])
            if not bag_id or "/" in bag_id or "\\" in bag_id:
                raise ValueError(f"rosbags.{role}[{index}].id is invalid")
            path = _path(entry["path"], source_dir, f"rosbags.{role}[{index}].path")
            reason = str(entry["reason"]) if role == "excluded" else None
            if role == "excluded" and not reason:
                raise ValueError(f"rosbags.{role}[{index}].reason must be non-empty")
            result.append(RosbagInput(bag_id, path, role, reason))
    ids = [item.bag_id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("rosbag identifiers must be unique across all roles")
    if sum(item.role == "calibration" for item in result) < 2:
        raise ValueError("at least two calibration rosbags are required")
    if not any(item.role == "validation" for item in result):
        raise ValueError("at least one validation rosbag is required")
    for item in result:
        if not item.path.is_dir() or not (item.path / "metadata.yaml").is_file():
            raise ValueError(f"rosbag directory or metadata.yaml is missing: {item.path}")
    return tuple(result)


def load_job_config(path: str | Path) -> CalibrationJobConfig:
    """Load and validate the one public configuration file."""
    source = Path(path).expanduser().resolve()
    root = _mapping(
        yaml.load(source.read_text(encoding="utf-8"), Loader=_StrictLoader),
        "calibration job config",
    )
    _exact(
        root,
        {
            "schema_version",
            "job",
            "runtime",
            "machine",
            "rosbags",
            "topics",
            "detection",
            "interpolation",
            "structural_gate",
            "sampling",
            "coverage",
            "calibration",
        },
        "calibration job config",
    )
    if root["schema_version"] != 1:
        raise ValueError("only calibration job schema_version 1 is supported")

    source_dir = source.parent
    job = _mapping(root["job"], "job")
    _exact(job, {"id", "dataset_date", "report_title"}, "job")
    job_id = str(job["id"])
    dataset_date = str(job["dataset_date"])
    if not job_id or "/" in job_id or "\\" in job_id:
        raise ValueError("job.id must be a non-empty path-safe identifier")
    if len(dataset_date) != 8 or not dataset_date.isdigit():
        raise ValueError("job.dataset_date must use YYYYMMDD")

    runtime = _mapping(root["runtime"], "runtime")
    _exact(
        runtime,
        {"novatel_message_dir", "output_dir", "inventory_only", "skip_validation"},
        "runtime",
    )
    for field in ("inventory_only", "skip_validation"):
        if not isinstance(runtime[field], bool):
            raise ValueError(f"runtime.{field} must be boolean")

    machine = _mapping(root["machine"], "machine")
    _exact(
        machine,
        {
            "serial",
            "camera_serial",
            "tag_rig_serial",
            "config_dir",
            "board_instance_id",
        },
        "machine",
    )
    machine_config_dir = _path(machine["config_dir"], source_dir, "machine.config_dir")
    if not machine_config_dir.is_dir():
        raise ValueError(f"machine.config_dir does not exist: {machine_config_dir}")
    bags = _load_bags(_mapping(root["rosbags"], "rosbags"), source_dir)

    topics = dict(_mapping(root["topics"], "topics"))
    _exact(topics, {"image", "odom", "inspvax", "imu"}, "topics")
    detection = dict(_mapping(root["detection"], "detection"))
    _exact(
        detection,
        {
            "queue_capacity",
            "flush_timeout_s",
            "roi_xyxy",
            "detector_nthreads",
            "quad_decimate",
            "quad_sigma",
            "refine_edges",
            "decode_sharpening",
            "maximum_translation_error_m",
            "maximum_rotation_error_deg",
        },
        "detection",
    )
    interpolation = dict(_mapping(root["interpolation"], "interpolation"))
    _exact(
        interpolation,
        {"maximum_bracket_gap_ms", "maximum_endpoint_distance_ms"},
        "interpolation",
    )
    structural_gate = dict(_mapping(root["structural_gate"], "structural_gate"))
    _exact(
        structural_gate,
        {
            "minimum_safe_depth_m",
            "maximum_depth_m",
            "contact_candidate_depth_m",
            "acceleration_highpass_limit_mps2",
            "angular_rate_highpass_limit_degps",
            "highpass_window_s",
            "contact_confirmation_s",
        },
        "structural_gate",
    )
    sampling = dict(_mapping(root["sampling"], "sampling"))
    _exact(
        sampling,
        {
            "minimum_xy_increment_m",
            "minimum_yaw_increment_deg",
            "maximum_time_increment_s",
            "maximum_frames_per_cell",
            "maximum_frames_per_bag",
            "minimum_bag_count",
            "minimum_bags_per_distance_bin",
            "minimum_total_frames",
        },
        "sampling",
    )
    coverage = dict(_mapping(root["coverage"], "coverage"))
    _exact(
        coverage,
        {"distance_bin_edges_m", "minimum_bearing_span_deg"},
        "coverage",
    )
    calibration = dict(_mapping(root["calibration"], "calibration"))
    _exact(
        calibration,
        {
            "maximum_depth_m",
            "maximum_depth_inclusive",
            "maximum_combinations",
            "allow_failed_candidate_for_diagnostics",
            "tag_geometry",
            "solver",
            "quality_gate",
        },
        "calibration",
    )

    tag_geometry = dict(_mapping(calibration["tag_geometry"], "calibration.tag_geometry"))
    _exact(
        tag_geometry,
        {
            "initialization_mode",
            "maximum_views_per_station",
            "minimum_valid_stations",
            "minimum_balanced_views",
            "overall_reprojection_rms_max_px",
            "maximum_view_rms_max_px",
            "loo_translation_std_max_m",
            "loo_rotation_std_max_deg",
            "loo_translation_difference_max_m",
            "loo_rotation_difference_max_deg",
            "max_nfev",
        },
        "calibration.tag_geometry",
    )
    if tag_geometry["initialization_mode"] not in {"bootstrap", "frozen_reuse"}:
        raise ValueError(
            "calibration.tag_geometry.initialization_mode must be bootstrap or frozen_reuse"
        )
    solver = dict(_mapping(calibration["solver"], "calibration.solver"))
    quality_gate = dict(_mapping(calibration["quality_gate"], "calibration.quality_gate"))
    _exact(
        quality_gate,
        {
            "quantile",
            "quantile_method",
            "position_limit_m",
            "yaw_limit_deg",
            "all_stations_must_pass",
        },
        "calibration.quality_gate",
    )
    for field in (
        "maximum_depth_inclusive",
        "allow_failed_candidate_for_diagnostics",
    ):
        if not isinstance(calibration[field], bool):
            raise ValueError(f"calibration.{field} must be boolean")
    if int(sampling.get("minimum_bag_count", 0)) != 2:
        raise ValueError("sampling.minimum_bag_count must be 2")
    maximum_combinations = calibration["maximum_combinations"]
    if (
        isinstance(maximum_combinations, bool)
        or not isinstance(maximum_combinations, int)
        or maximum_combinations < 0
    ):
        raise ValueError("calibration.maximum_combinations must be a non-negative integer")
    novatel_message_dir = _path(
        runtime["novatel_message_dir"], source_dir, "runtime.novatel_message_dir"
    )
    if not novatel_message_dir.is_dir():
        raise ValueError(
            f"runtime.novatel_message_dir does not exist: {novatel_message_dir}"
        )
    calibration_ids = [item.bag_id for item in bags if item.role == "calibration"]
    validation_ids = [item.bag_id for item in bags if item.role == "validation"]
    displacement_ids = [
        item.bag_id for item in bags if item.role == "displacement_evaluation"
    ]
    excluded = {
        item.bag_id: item.reason for item in bags if item.role == "excluded"
    }
    all_active_ids = calibration_ids + validation_ids + displacement_ids

    online_payload: dict[str, object] = {
        "schema_version": 2,
        "machine": {
            "serial": str(machine["serial"]),
            "camera_serial": str(machine["camera_serial"]),
            "tag_rig_serial": str(machine["tag_rig_serial"]),
        },
        "topics": {
            "image": str(topics["image"]),
            "odom": str(topics["odom"]),
            "inspvax": str(topics["inspvax"]),
            "vision_qc": "/dynamic_rosbag_calibration/vision_qc",
            "tf_static": "/tf_static",
        },
        "online_session": {
            "recording_duration_s": 10.0,
            "live_status_period_ms": 500,
            "minimum_free_disk_gb": 1.0,
            "recorder_ready_timeout_s": 5.0,
            "recorder_ready_poll_period_ms": 50,
            "rosbag_stop_timeout_s": 15.0,
            "rosbag_terminate_timeout_s": 3.0,
        },
        "bag_timing": {
            "image_maximum_source_gap_ms": 500.0,
            "odom_maximum_source_gap_ms": 100.0,
            "inspvax_maximum_source_gap_ms": 100.0,
        },
        "live_qc": detection,
        "station_gate": {
            "duration_tolerance_s": 0.2,
            "required_ins_solution_status": "INS_SOLUTION_GOOD",
            "required_ins_position_type": "INS_RTKFIXED",
            "maximum_inspvax_sample_gap_ms": 100.0,
            "maximum_ins_xy_span_m": 0.025,
            "maximum_ins_yaw_span_deg": 0.10,
            "maximum_speed_p95_mps": 0.02,
            "maximum_time_association_p95_ms": 20.0,
            "minimum_tag_edge_px": 15.0,
            "minimum_image_or_roi_margin_px": 20.0,
            "minimum_f0_valid_frames": 30,
            "minimum_retained_fraction": 0.20,
            "maximum_work_distance_m": 8.0,
        },
        "collection_gate": {
            "minimum_qualified_stations": 6,
            "preferred_qualified_stations": 6,
            "start_calibration_when_preferred_ready": True,
            "additional_station_policy": "only_after_coverage_or_quality_gate_failure",
            "distance_bin_edges_m": coverage["distance_bin_edges_m"],
            "minimum_lateral_bearing_span_deg": 4.0,
            "require_distance_bins": ["near", "middle", "far"],
        },
        "output": {
            "session_root": "unused/sessions",
            "approved_root": "unused/approved",
            "rejected_root": "unused/rejected",
            "overwrite_existing": False,
            "write_runtime_config": False,
        },
        "calibration": {
            "machine_config_dir": str(machine_config_dir),
            "board_instance_id": str(machine["board_instance_id"]),
            "tag_geometry": tag_geometry,
            "solver": solver,
            "compliance": quality_gate,
        },
        "record_topics": [
            str(topics["image"]),
            str(topics["odom"]),
            str(topics["inspvax"]),
            str(topics["imu"]),
        ],
    }
    experiment_payload: dict[str, object] = {
        "schema_version": 1,
        "dataset_name": job_id,
        "topics": topics,
        "bags": {
            "training": calibration_ids,
            "holdout_no_bridge": validation_ids + displacement_ids,
            "excluded": excluded,
        },
        "interpolation": interpolation,
        "structural_gate": structural_gate,
        "sampling": sampling,
        "coverage": coverage,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "date_range": [dataset_date, dataset_date],
        "report_title": str(job["report_title"]),
        "datasets": {
            dataset_date: {
                "bags": all_active_ids,
                "calibration_bags": calibration_ids,
                "displacement_evaluation_bags": displacement_ids,
                "excluded": excluded,
                "maximum_combinations": maximum_combinations,
            }
        },
        "two_bag_sampling": {
            "minimum_bag_count": int(sampling["minimum_bag_count"]),
            "minimum_bags_per_distance_bin": int(
                sampling["minimum_bags_per_distance_bin"]
            ),
            "minimum_total_frames": int(sampling["minimum_total_frames"]),
        },
        "two_bag_coverage": {
            "minimum_bearing_span_deg": float(coverage["minimum_bearing_span_deg"]),
        },
        "calibration_policy": {
            "maximum_depth_m": float(calibration["maximum_depth_m"]),
            "maximum_depth_inclusive": calibration["maximum_depth_inclusive"],
            "tag_geometry_initialization_mode": str(
                tag_geometry["initialization_mode"]
            ),
            "tag_geometry_minimum_valid_stations": int(
                tag_geometry["minimum_valid_stations"]
            ),
            "tag_geometry_maximum_views_per_station": int(
                tag_geometry["maximum_views_per_station"]
            ),
            "tag_geometry_minimum_balanced_views": int(
                tag_geometry["minimum_balanced_views"]
            ),
            "allow_failed_tag_geometry_for_diagnostics": calibration[
                "allow_failed_candidate_for_diagnostics"
            ],
        },
    }
    return CalibrationJobConfig(
        source_path=source,
        source_fingerprint=sha256_file(source),
        job_id=job_id,
        dataset_date=dataset_date,
        report_title=str(job["report_title"]),
        novatel_message_dir=novatel_message_dir,
        output_dir=_path(runtime["output_dir"], source_dir, "runtime.output_dir"),
        inventory_only=runtime["inventory_only"],
        skip_validation=runtime["skip_validation"],
        bags=bags,
        machine_config_dir=machine_config_dir,
        online_payload=online_payload,
        experiment_payload=experiment_payload,
        manifest=manifest,
    )
