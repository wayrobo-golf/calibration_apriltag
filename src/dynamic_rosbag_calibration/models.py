"""Data models shared by ROS adapters and the deterministic session core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SessionState(str, Enum):
    CREATED = "SESSION_CREATED"
    COLLECTING = "COLLECTING_MORE_STATIONS"
    COLLECTION_READY = "COLLECTION_READY"
    CALIBRATION_RUNNING = "CALIBRATION_RUNNING"
    COMPLETE = "COMPLETE"


class StationState(str, Enum):
    WAITING_INS = "STATION_WAITING_INS"
    RECORDER_STARTING = "STATION_RECORDER_STARTING"
    RECORDER_READY = "STATION_RECORDER_READY"
    RECORDING = "STATION_RECORDING"
    FINALIZING = "STATION_FINALIZING"
    PROVISIONAL_ACCEPTED = "STATION_PROVISIONAL_ACCEPTED"
    ACCEPTED = "STATION_ACCEPTED"
    REJECTED = "STATION_REJECTED"
    INCOMPLETE = "STATION_INCOMPLETE"


@dataclass(frozen=True)
class InsQualitySample:
    source_stamp_s: float
    callback_entry_stamp_s: float
    solution_status: str
    position_type: str


@dataclass(frozen=True)
class OdomSample:
    stamp_s: float
    x_m: float
    y_m: float
    yaw_deg: float
    speed_mps: float


@dataclass(frozen=True)
class VisionQualitySample:
    stamp_s: float
    both_tags_detected: bool
    f0_valid: bool
    training_gate_pass: bool | None
    camera_tag0_tz_m: float
    bearing_x_deg: float
    minimum_tag_edge_px: float
    minimum_margin_px: float
    odom_association_dt_ms: float
    diagnostic_reasons: tuple[str, ...] = ()
    dual_reprojection_rms_px: float | None = None


@dataclass(frozen=True)
class RecorderStopResult:
    ok: bool
    output_path: Path
    message: str
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class StationGateReport:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass
class StationRecord:
    station_id: str
    board_instance_id: str
    state: StationState
    requested_at_s: float
    recorder_requested_at_s: float | None = None
    recorder_ready_at_s: float | None = None
    started_at_s: float | None = None
    ended_at_s: float | None = None
    bag_path: Path | None = None
    gate_report: StationGateReport | None = None
    provisional_gate_report: StationGateReport | None = None
    final_gate_report: StationGateReport | None = None
    bag_integrity: dict[str, Any] | None = None
    raw_data_identity_sha256: str | None = None
    raw_data_size_bytes: int | None = None
    distance_bin: str | None = None
    lateral_bin: str | None = None
    quality_samples: list[InsQualitySample] = field(default_factory=list, repr=False)
    odom_samples: list[OdomSample] = field(default_factory=list, repr=False)
    vision_samples: list[VisionQualitySample] = field(default_factory=list, repr=False)
    latched_reasons: list[str] = field(default_factory=list, repr=False)

    def manifest_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("quality_samples", None)
        value.pop("odom_samples", None)
        value.pop("vision_samples", None)
        value.pop("latched_reasons", None)
        value.pop("gate_report", None)
        value["schema_version"] = 2
        value["state"] = self.state.value
        value["bag_path"] = str(self.bag_path) if self.bag_path else None
        return value


def normalize_station_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Read schema v1 explicitly without inventing source-time evidence."""
    value = dict(payload)
    schema_version = int(value.get("schema_version", 1))
    if schema_version == 2:
        value.setdefault("provisional_gate_report", None)
        value.setdefault("final_gate_report", None)
        return value
    if schema_version != 1:
        raise ValueError(f"unsupported station manifest schema {schema_version}")
    legacy_gate = value.pop("gate_report", None)
    diagnostics = dict(value.get("diagnostics", {}))
    if isinstance(legacy_gate, dict):
        metrics = legacy_gate.get("metrics", {})
        if isinstance(metrics, dict) and "maximum_inspvax_gap_ms" in metrics:
            diagnostics["legacy_callback_gap_ms"] = metrics[
                "maximum_inspvax_gap_ms"
            ]
        diagnostics["legacy_gate_report"] = legacy_gate
    value.update(
        schema_version=2,
        provisional_gate_report=None,
        final_gate_report=None,
        diagnostics=diagnostics,
        migrated_from_schema_version=1,
    )
    return value


@dataclass(frozen=True)
class CollectionGateReport:
    ready: bool
    accepted_count: int
    reasons: tuple[str, ...]
    missing_distance_bins: tuple[str, ...]
    lateral_bearing_min_deg: float | None
    lateral_bearing_max_deg: float | None
    lateral_bearing_span_deg: float | None
    minimum_lateral_bearing_span_deg: float
    next_station_hint: str | None
