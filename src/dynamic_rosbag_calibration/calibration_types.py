"""Immutable data contracts for the calibration mathematical core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def readonly_float_array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Transform:
    matrix: FloatArray
    parent_frame: str
    child_frame: str

    def __post_init__(self) -> None:
        matrix = readonly_float_array(self.matrix, (4, 4), "transform matrix")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
            raise ValueError("transform matrix must have homogeneous final row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("transform rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("transform rotation determinant must be +1")
        if not self.parent_frame or not self.child_frame:
            raise ValueError("transform frame names must be non-empty")
        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class CameraModel:
    matrix: FloatArray
    distortion: FloatArray

    def __post_init__(self) -> None:
        matrix = readonly_float_array(self.matrix, (3, 3), "camera matrix")
        distortion = np.array(self.distortion, dtype=np.float64, copy=True).reshape(-1)
        if not np.all(np.isfinite(distortion)):
            raise ValueError("camera distortion must contain only finite values")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        distortion.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "distortion", distortion)


@dataclass(frozen=True)
class FrameObservation:
    frame_key: str
    station_id: str
    board_instance_id: str
    map_ins: Transform
    camera_tag0: Transform | None
    corners_px: FloatArray
    odom_dt_ms: float
    basic_valid: bool
    station_gate_pass: bool

    def __post_init__(self) -> None:
        if not self.frame_key or not self.station_id or not self.board_instance_id:
            raise ValueError("frame, station, and board identifiers must be non-empty")
        if self.map_ins.parent_frame != "ins_map" or self.map_ins.child_frame != "ins_link":
            raise ValueError("map_ins must have direction ins_map <- ins_link")
        if self.camera_tag0 is not None and self.camera_tag0.child_frame != "tag0":
            raise ValueError("camera_tag0 child frame must be tag0")
        if not np.isfinite(self.odom_dt_ms) or self.odom_dt_ms < 0.0:
            raise ValueError("odom_dt_ms must be finite and non-negative")
        object.__setattr__(
            self,
            "corners_px",
            readonly_float_array(self.corners_px, (8, 2), "corners_px"),
        )


@dataclass(frozen=True)
class CalibrationProblem:
    frames: tuple[FrameObservation, ...]
    nominal_extrinsic: Transform
    initial_tag0_tag1: Transform | None
    tag_size_m: float
    camera_model: CameraModel
    config_fingerprint: str
    initial_tag0_tag1_source_path: Path | None = None
    initial_tag0_tag1_source_sha256: str | None = None

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("calibration problem must contain frames")
        if self.nominal_extrinsic.parent_frame != "ins_link":
            raise ValueError("nominal extrinsic parent frame must be ins_link")
        if self.nominal_extrinsic.child_frame != "left_camera":
            raise ValueError("nominal extrinsic child frame must be left_camera")
        if self.initial_tag0_tag1 is not None and (
            self.initial_tag0_tag1.parent_frame != "tag0"
            or self.initial_tag0_tag1.child_frame != "tag1"
        ):
            raise ValueError("initial tag geometry must have direction tag0 <- tag1")
        if not np.isfinite(self.tag_size_m) or self.tag_size_m <= 0.0:
            raise ValueError("tag_size_m must be finite and positive")
        if not self.config_fingerprint:
            raise ValueError("config_fingerprint must be non-empty")
        object.__setattr__(self, "frames", frames)
