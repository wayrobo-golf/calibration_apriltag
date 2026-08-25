"""Balanced dual-Tag corner bundle adjustment and geometry quality gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .calibration_types import CameraModel, FrameObservation, Transform
from .se3 import exp_se3, inverse, log_se3


@dataclass(frozen=True)
class TagGeometryConfig:
    maximum_views_per_station: int = 5
    minimum_valid_stations: int = 6
    minimum_balanced_views: int = 25
    overall_reprojection_rms_max_px: float = 0.5
    maximum_view_rms_max_px: float = 1.0
    loo_translation_std_max_m: float = 0.005
    loo_rotation_std_max_deg: float = 0.5
    loo_translation_difference_max_m: float = 0.01
    loo_rotation_difference_max_deg: float = 1.0
    max_nfev: int = 500


@dataclass(frozen=True)
class FrozenTagGeometry:
    transform: Transform
    source_frame_keys: tuple[str, ...]


@dataclass(frozen=True)
class TagGeometryResult:
    candidate: Transform
    frozen: FrozenTagGeometry | None
    quality_gate_pass: bool
    reasons: tuple[str, ...]
    overall_reprojection_rms_px: float
    maximum_view_rms_px: float
    balanced_view_count: int
    valid_station_count: int
    loo_translation_std_m: float
    loo_rotation_std_deg: float
    loo_translation_difference_max_m: float
    loo_rotation_difference_max_deg: float
    solver_success: bool
    solver_message: str
    initialization: object | None = None


def tag_square_points(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def select_balanced_views(
    frames: Sequence[FrameObservation], maximum_per_station: int
) -> tuple[FrameObservation, ...]:
    if maximum_per_station <= 0:
        raise ValueError("maximum_per_station must be positive")
    grouped: dict[str, list[FrameObservation]] = {}
    for frame in sorted(frames, key=lambda item: item.frame_key):
        if frame.basic_valid and frame.station_gate_pass:
            grouped.setdefault(frame.station_id, []).append(frame)
    selected: list[FrameObservation] = []
    for station in sorted(grouped):
        values = grouped[station]
        if len(values) <= maximum_per_station:
            selected.extend(values)
            continue
        indices = np.linspace(0, len(values) - 1, maximum_per_station, dtype=int)
        selected.extend(values[int(index)] for index in indices)
    return tuple(selected)


def _project(
    camera_tag: np.ndarray,
    object_points: np.ndarray,
    camera: CameraModel,
) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(camera_tag[:3, :3])
    pixels, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        camera_tag[:3, 3],
        camera.matrix,
        camera.distortion,
    )
    return pixels.reshape(-1, 2)


def _solve(
    frames: Sequence[FrameObservation],
    camera: CameraModel,
    tag_size_m: float,
    initial_geometry: Transform,
    max_nfev: int,
) -> tuple[np.ndarray, object, np.ndarray]:
    points = tag_square_points(tag_size_m)
    if any(frame.camera_tag0 is None for frame in frames):
        raise ValueError("tag geometry frames must have initialized camera_tag0 poses")
    initial_views = [frame.camera_tag0.matrix for frame in frames if frame.camera_tag0 is not None]
    parameter_count = 6 + 6 * len(frames)
    station_counts: dict[str, int] = {}
    for frame in frames:
        station_counts[frame.station_id] = station_counts.get(frame.station_id, 0) + 1
    weights = np.array(
        [1.0 / np.sqrt(station_counts[frame.station_id]) for frame in frames],
        dtype=np.float64,
    )

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        geometry = initial_geometry.matrix @ exp_se3(parameters[:6])
        views = [
            initial_views[index] @ exp_se3(parameters[6 + 6 * index : 12 + 6 * index])
            for index in range(len(frames))
        ]
        return geometry, views

    def raw_residual(parameters: np.ndarray) -> np.ndarray:
        geometry, views = unpack(parameters)
        result = np.empty((len(frames), 8, 2), dtype=np.float64)
        for index, frame in enumerate(frames):
            result[index, :4] = _project(views[index], points, camera) - frame.corners_px[:4]
            result[index, 4:] = (
                _project(views[index] @ geometry, points, camera) - frame.corners_px[4:]
            )
        return result

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (raw_residual(parameters) * weights[:, None, None]).reshape(-1)

    sparsity = lil_matrix((16 * len(frames), parameter_count), dtype=int)
    for index in range(len(frames)):
        row = slice(16 * index, 16 * (index + 1))
        sparsity[row, :6] = 1
        sparsity[row, 6 + 6 * index : 12 + 6 * index] = 1
    scale = np.tile(
        np.array([0.03, 0.03, 0.03, np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5)]),
        1 + len(frames),
    )
    fit = least_squares(
        residual,
        np.zeros(parameter_count, dtype=np.float64),
        method="trf",
        loss="huber",
        f_scale=1.0,
        jac_sparsity=sparsity,
        x_scale=scale,
        max_nfev=max_nfev,
        # The Jacobian is obtained by finite differences over pixel residuals.
        # Tolerances below roughly sqrt(machine epsilon) make noisy real data
        # hit max_nfev after the cost has already reached its numeric plateau.
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    geometry, _ = unpack(fit.x)
    return geometry, fit, raw_residual(fit.x)


def estimate_tag_geometry(
    frames_or_initialization: object,
    camera: CameraModel,
    tag_size_m: float,
    initial_or_config: Transform | TagGeometryConfig,
    config: TagGeometryConfig | None = None,
) -> TagGeometryResult:
    initialization = None
    if config is None:
        initialization = frames_or_initialization
        frames = tuple(initialization.seeded_frames)
        initial_geometry = initialization.initial_transform
        config = initial_or_config
        if not isinstance(config, TagGeometryConfig):
            raise TypeError("new estimate_tag_geometry form requires TagGeometryConfig")
    else:
        frames = frames_or_initialization
        initial_geometry = initial_or_config
        if not isinstance(initial_geometry, Transform):
            raise TypeError("legacy estimate_tag_geometry form requires Transform")
    balanced = select_balanced_views(frames, config.maximum_views_per_station)
    if not balanced:
        raise ValueError("no valid frames for tag geometry estimation")
    geometry, fit, residual = _solve(
        balanced, camera, tag_size_m, initial_geometry, config.max_nfev
    )
    per_view_rms = np.sqrt(np.mean(np.square(residual), axis=(1, 2)))
    overall_rms = float(np.sqrt(np.mean(np.square(residual))))
    maximum_view_rms = float(np.max(per_view_rms))
    stations = tuple(sorted({frame.station_id for frame in balanced}))

    loo_deltas: list[np.ndarray] = []
    for station in stations:
        subset = tuple(frame for frame in balanced if frame.station_id != station)
        if not subset:
            continue
        candidate, loo_fit, _ = _solve(
            subset,
            camera,
            tag_size_m,
            Transform(geometry, "tag0", "tag1"),
            config.max_nfev,
        )
        if bool(loo_fit.success) and np.all(np.isfinite(candidate)):
            loo_deltas.append(log_se3(inverse(geometry) @ candidate))
    if loo_deltas:
        values = np.asarray(loo_deltas, dtype=np.float64)
        translation_std = float(np.linalg.norm(np.std(values[:, :3], axis=0)))
        rotation_std_deg = float(np.degrees(np.linalg.norm(np.std(values[:, 3:], axis=0))))
        translation_max = float(np.max(np.linalg.norm(values[:, :3], axis=1)))
        rotation_max_deg = float(np.degrees(np.max(np.linalg.norm(values[:, 3:], axis=1))))
    else:
        translation_std = rotation_std_deg = translation_max = rotation_max_deg = float("inf")

    reasons: list[str] = []
    if len(stations) < config.minimum_valid_stations:
        reasons.append("CAL-E-TAG-GEOMETRY-STATION-COUNT")
    if len(balanced) < config.minimum_balanced_views:
        reasons.append("CAL-E-TAG-GEOMETRY-VIEW-COUNT")
    if not bool(fit.success) or not np.all(np.isfinite(geometry)):
        reasons.append("CAL-E-TAG-GEOMETRY-SOLVER")
    if len(loo_deltas) != len(stations):
        reasons.append("CAL-E-TAG-GEOMETRY-LOO-SOLVER")
    if overall_rms > config.overall_reprojection_rms_max_px:
        reasons.append("CAL-E-TAG-GEOMETRY-OVERALL-RMS")
    if maximum_view_rms > config.maximum_view_rms_max_px:
        reasons.append("CAL-E-TAG-GEOMETRY-VIEW-RMS")
    if translation_std > config.loo_translation_std_max_m:
        reasons.append("CAL-E-TAG-GEOMETRY-LOO-TRANSLATION-STD")
    if rotation_std_deg > config.loo_rotation_std_max_deg:
        reasons.append("CAL-E-TAG-GEOMETRY-LOO-ROTATION-STD")
    if translation_max > config.loo_translation_difference_max_m:
        reasons.append("CAL-E-TAG-GEOMETRY-LOO-TRANSLATION-MAX")
    if rotation_max_deg > config.loo_rotation_difference_max_deg:
        reasons.append("CAL-E-TAG-GEOMETRY-LOO-ROTATION-MAX")
    candidate = Transform(geometry, "tag0", "tag1")
    passed = not reasons
    frozen = (
        FrozenTagGeometry(candidate, tuple(frame.frame_key for frame in balanced))
        if passed
        else None
    )
    return TagGeometryResult(
        candidate=candidate,
        frozen=frozen,
        quality_gate_pass=passed,
        reasons=tuple(reasons),
        overall_reprojection_rms_px=overall_rms,
        maximum_view_rms_px=maximum_view_rms,
        balanced_view_count=len(balanced),
        valid_station_count=len(stations),
        loo_translation_std_m=translation_std,
        loo_rotation_std_deg=rotation_std_deg,
        loo_translation_difference_max_m=translation_max,
        loo_rotation_difference_max_deg=rotation_max_deg,
        solver_success=bool(fit.success),
        solver_message=str(fit.message),
        initialization=initialization,
    )
