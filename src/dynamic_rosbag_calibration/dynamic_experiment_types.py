"""Immutable contracts for safe-dynamic calibration experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .calibration_types import FrameObservation, Transform, readonly_float_array


class StructuralState(str, Enum):
    UNLOADED_SAFE = "UNLOADED_SAFE"
    CONTACT_TRANSIENT = "CONTACT_TRANSIENT"
    LOADED_STABLE = "LOADED_STABLE"
    RELEASE_TRANSIENT = "RELEASE_TRANSIENT"
    UNKNOWN = "UNKNOWN"


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class PoseInterpolationConfig:
    maximum_bracket_gap_ms: float
    maximum_endpoint_distance_ms: float

    def __post_init__(self) -> None:
        bracket = _positive(self.maximum_bracket_gap_ms, "maximum_bracket_gap_ms")
        endpoint = _positive(
            self.maximum_endpoint_distance_ms,
            "maximum_endpoint_distance_ms",
        )
        object.__setattr__(self, "maximum_bracket_gap_ms", bracket)
        object.__setattr__(self, "maximum_endpoint_distance_ms", endpoint)


@dataclass(frozen=True)
class StructuralGateConfig:
    minimum_safe_depth_m: float
    maximum_depth_m: float
    contact_candidate_depth_m: float
    acceleration_highpass_limit_mps2: float
    angular_rate_highpass_limit_degps: float
    highpass_window_s: float
    contact_confirmation_s: float

    def __post_init__(self) -> None:
        minimum = _positive(self.minimum_safe_depth_m, "minimum_safe_depth_m")
        maximum = _positive(self.maximum_depth_m, "maximum_depth_m")
        contact = _positive(
            self.contact_candidate_depth_m,
            "contact_candidate_depth_m",
        )
        if maximum <= minimum:
            raise ValueError("safe depth range must be strictly increasing")
        if contact >= minimum:
            raise ValueError("contact candidate depth must be below safe depth")
        object.__setattr__(self, "minimum_safe_depth_m", minimum)
        object.__setattr__(self, "maximum_depth_m", maximum)
        object.__setattr__(self, "contact_candidate_depth_m", contact)
        object.__setattr__(
            self,
            "acceleration_highpass_limit_mps2",
            _positive(
                self.acceleration_highpass_limit_mps2,
                "acceleration_highpass_limit_mps2",
            ),
        )
        object.__setattr__(
            self,
            "angular_rate_highpass_limit_degps",
            _positive(
                self.angular_rate_highpass_limit_degps,
                "angular_rate_highpass_limit_degps",
            ),
        )
        object.__setattr__(
            self,
            "highpass_window_s",
            _positive(self.highpass_window_s, "highpass_window_s"),
        )
        object.__setattr__(
            self,
            "contact_confirmation_s",
            _positive(self.contact_confirmation_s, "contact_confirmation_s"),
        )


@dataclass(frozen=True)
class DynamicSamplingConfig:
    minimum_xy_increment_m: float
    minimum_yaw_increment_deg: float
    maximum_time_increment_s: float
    maximum_frames_per_cell: int
    maximum_frames_per_bag: int
    minimum_bag_count: int
    minimum_bags_per_distance_bin: int
    minimum_total_frames: int

    def __post_init__(self) -> None:
        for field in (
            "minimum_xy_increment_m",
            "minimum_yaw_increment_deg",
            "maximum_time_increment_s",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        for field in (
            "maximum_frames_per_cell",
            "maximum_frames_per_bag",
            "minimum_bag_count",
            "minimum_bags_per_distance_bin",
            "minimum_total_frames",
        ):
            object.__setattr__(
                self,
                field,
                _positive_integer(getattr(self, field), field),
            )
        if self.maximum_frames_per_cell > self.maximum_frames_per_bag:
            raise ValueError(
                "maximum_frames_per_cell cannot exceed maximum_frames_per_bag"
            )


@dataclass(frozen=True)
class DynamicCoverageConfig:
    distance_bin_edges_m: tuple[float, float]
    minimum_bearing_span_deg: float

    def __post_init__(self) -> None:
        edges = tuple(float(value) for value in self.distance_bin_edges_m)
        if (
            len(edges) != 2
            or not all(math.isfinite(value) and value > 0.0 for value in edges)
            or edges[0] >= edges[1]
        ):
            raise ValueError("distance bin edges must be finite, positive, and increasing")
        object.__setattr__(self, "distance_bin_edges_m", edges)
        object.__setattr__(
            self,
            "minimum_bearing_span_deg",
            _positive(self.minimum_bearing_span_deg, "minimum_bearing_span_deg"),
        )


@dataclass(frozen=True)
class DynamicExperimentTopics:
    image: str
    odom: str
    inspvax: str
    imu: str

    def __post_init__(self) -> None:
        values = (self.image, self.odom, self.inspvax, self.imu)
        if any(not value or not value.startswith("/") for value in values):
            raise ValueError("dynamic experiment topics must be absolute ROS names")
        if len(set(values)) != len(values):
            raise ValueError("dynamic experiment topics must be unique")


@dataclass(frozen=True)
class DynamicBagGroups:
    training: tuple[str, ...]
    holdout_no_bridge: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        training = tuple(str(value) for value in self.training)
        holdout = tuple(str(value) for value in self.holdout_no_bridge)
        excluded = tuple((str(key), str(reason)) for key, reason in self.excluded)
        all_ids = (*training, *holdout, *(key for key, _reason in excluded))
        if any(not value for value in all_ids):
            raise ValueError("bag identifiers must be non-empty")
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("training, holdout, and excluded bag groups must not overlap")
        if not training or not holdout:
            raise ValueError("training and holdout bag groups must be non-empty")
        if any(not reason for _key, reason in excluded):
            raise ValueError("excluded bag reasons must be non-empty")
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "holdout_no_bridge", holdout)
        object.__setattr__(self, "excluded", excluded)


@dataclass(frozen=True)
class DynamicExperimentConfig:
    schema_version: int
    dataset_name: str
    topics: DynamicExperimentTopics
    bags: DynamicBagGroups
    interpolation: PoseInterpolationConfig
    structural_gate: StructuralGateConfig
    sampling: DynamicSamplingConfig
    coverage: DynamicCoverageConfig
    source_fingerprint: str


@dataclass(frozen=True)
class TimedPose:
    stamp_s: float
    matrix: np.ndarray

    def __post_init__(self) -> None:
        stamp = float(self.stamp_s)
        if not math.isfinite(stamp):
            raise ValueError("pose stamp must be finite")
        matrix = readonly_float_array(self.matrix, (4, 4), "timed pose matrix")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
            raise ValueError("timed pose matrix must be homogeneous")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("timed pose rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("timed pose rotation determinant must be +1")
        object.__setattr__(self, "stamp_s", stamp)
        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class InterpolatedPose:
    pose: Transform
    left_stamp_s: float
    right_stamp_s: float
    alpha: float
    bracket_gap_ms: float
    left_dt_ms: float
    right_dt_ms: float


@dataclass(frozen=True)
class TimedImu:
    stamp_s: float
    linear_acceleration_mps2: np.ndarray
    angular_velocity_radps: np.ndarray

    def __post_init__(self) -> None:
        stamp = float(self.stamp_s)
        if not math.isfinite(stamp):
            raise ValueError("IMU stamp must be finite")
        object.__setattr__(self, "stamp_s", stamp)
        object.__setattr__(
            self,
            "linear_acceleration_mps2",
            readonly_float_array(
                self.linear_acceleration_mps2,
                (3,),
                "linear_acceleration_mps2",
            ),
        )
        object.__setattr__(
            self,
            "angular_velocity_radps",
            readonly_float_array(
                self.angular_velocity_radps,
                (3,),
                "angular_velocity_radps",
            ),
        )


@dataclass(frozen=True)
class TimedInsQuality:
    stamp_s: float
    solution_status: int
    position_type: int

    def __post_init__(self) -> None:
        stamp = float(self.stamp_s)
        if not math.isfinite(stamp):
            raise ValueError("INS quality stamp must be finite")
        object.__setattr__(self, "stamp_s", stamp)
        object.__setattr__(self, "solution_status", int(self.solution_status))
        object.__setattr__(self, "position_type", int(self.position_type))

    @property
    def good(self) -> bool:
        return self.solution_status == 3 and self.position_type == 56


@dataclass(frozen=True)
class HighpassMotionSample:
    stamp_s: float
    acceleration_norm_mps2: float
    angular_rate_norm_degps: float

    def __post_init__(self) -> None:
        for field in (
            "stamp_s",
            "acceleration_norm_mps2",
            "angular_rate_norm_degps",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or (field != "stamp_s" and value < 0.0):
                raise ValueError(f"{field} must be finite and non-negative")
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class RawDynamicFrame:
    frame_key: str
    bag_id: str
    sequence: int
    source_stamp_s: float
    observation: FrameObservation | None
    camera_tag0_initial: Transform | None
    initial_tag0_depth_m: float | None
    bearing_x_deg: float | None
    minimum_tag_edge_px: float
    minimum_margin_px: float
    interpolation: InterpolatedPose | None
    ins_quality_good: bool
    ins_quality_dt_ms: float | None
    exclusion_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.frame_key or not self.bag_id or self.sequence <= 0:
            raise ValueError("raw dynamic frame identifiers must be valid")
        stamp = float(self.source_stamp_s)
        if not math.isfinite(stamp):
            raise ValueError("raw dynamic frame source stamp must be finite")
        object.__setattr__(self, "source_stamp_s", stamp)
        for field in ("initial_tag0_depth_m", "bearing_x_deg", "ins_quality_dt_ms"):
            value = getattr(self, field)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite when present")
        for field in ("minimum_tag_edge_px", "minimum_margin_px"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
            object.__setattr__(self, field, value)
        object.__setattr__(
            self,
            "exclusion_reasons",
            tuple(dict.fromkeys(str(value) for value in self.exclusion_reasons)),
        )


@dataclass(frozen=True)
class DynamicBagEvidence:
    bag_id: str
    bag_path: Path
    raw_data_identity_sha256: str
    topic_message_counts: tuple[tuple[str, int], ...]
    odometry: tuple[TimedPose, ...]
    imu: tuple[TimedImu, ...]
    quality: tuple[TimedInsQuality, ...]
    frames: tuple[RawDynamicFrame, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.bag_id or not self.raw_data_identity_sha256:
            raise ValueError("dynamic bag evidence requires identifiers")
        if len(self.raw_data_identity_sha256) != 64:
            raise ValueError("raw data identity SHA256 must contain 64 characters")
        object.__setattr__(self, "bag_path", Path(self.bag_path))
        object.__setattr__(self, "topic_message_counts", tuple(self.topic_message_counts))
        object.__setattr__(self, "odometry", tuple(self.odometry))
        object.__setattr__(self, "imu", tuple(self.imu))
        object.__setattr__(self, "quality", tuple(self.quality))
        object.__setattr__(self, "frames", tuple(self.frames))


@dataclass(frozen=True)
class DynamicFrameEvidence:
    source: RawDynamicFrame
    structural_state: StructuralState
    acceleration_highpass_mps2: float | None
    angular_rate_highpass_degps: float | None
    exclusion_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "acceleration_highpass_mps2",
            "angular_rate_highpass_degps",
        ):
            value = getattr(self, field)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{field} must be finite and non-negative")
        object.__setattr__(
            self,
            "exclusion_reasons",
            tuple(dict.fromkeys(str(value) for value in self.exclusion_reasons)),
        )


@dataclass(frozen=True)
class DynamicCoverageReport:
    passed: bool
    reasons: tuple[str, ...]
    bag_count: int
    total_frame_count: int
    bearing_min_deg: float | None
    bearing_max_deg: float | None
    bearing_span_deg: float | None
    distance_bin_bag_counts: tuple[tuple[str, int], ...]
    distance_bin_frame_counts: tuple[tuple[str, int], ...]
    cell_counts: tuple[tuple[str, str, str, int], ...]


@dataclass(frozen=True)
class DynamicSelection:
    frames: tuple[DynamicFrameEvidence, ...]
    keyframe_keys: tuple[str, ...]
    selected_frame_keys_sha256: str
    coverage: DynamicCoverageReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "keyframe_keys", tuple(self.keyframe_keys))
        if len(self.selected_frame_keys_sha256) != 64:
            raise ValueError("selected frame keys SHA256 must contain 64 characters")
