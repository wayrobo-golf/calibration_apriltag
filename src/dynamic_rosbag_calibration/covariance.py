"""Scaled PnP covariance estimation and closure-residual propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .calibration_types import CameraModel
from .se3 import exp_se3


@dataclass(frozen=True)
class CovarianceConfig:
    pixel_sigma_floor_px: float = 0.10
    pixel_sigma_ceiling_px: float = 5.0
    normal_matrix_rcond: float = 1.0e-12
    scaled_eigenvalue_min: float = 1.0e-6
    scaled_eigenvalue_max: float = 1.0e6
    unmodelled_translation_floor_m: float = 0.001
    unmodelled_rotation_floor_deg: float = 0.01


@dataclass(frozen=True)
class CovarianceEstimate:
    valid: bool
    reason: str | None
    numerical_rank: int
    pixel_sigma_px: float | None
    pose_covariance: np.ndarray | None
    residual_covariance: np.ndarray | None
    whitener: np.ndarray | None


def _project(
    transform: np.ndarray,
    points: np.ndarray,
    camera: CameraModel,
) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
    pixels, _ = cv2.projectPoints(
        points,
        rotation_vector,
        transform[:3, 3],
        camera.matrix,
        camera.distortion,
    )
    return pixels.reshape(-1)


def _central_jacobian(function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    columns = []
    for index in range(6):
        step = 1.0e-6
        delta = np.zeros(6, dtype=np.float64)
        delta[index] = step
        columns.append((function(delta) - function(-delta)) / (2.0 * step))
    return np.column_stack(columns)


def _invalid(rank: int, reason: str) -> CovarianceEstimate:
    return CovarianceEstimate(False, reason, rank, None, None, None, None)


def estimate_pnp_covariance(
    camera_tag0: np.ndarray,
    observed_corners_px: np.ndarray,
    object_points_tag0_m: np.ndarray,
    camera: CameraModel,
    closure_residual: Callable[[np.ndarray], np.ndarray],
    config: CovarianceConfig,
) -> CovarianceEstimate:
    pose = np.asarray(camera_tag0, dtype=np.float64).reshape(4, 4)
    observed = np.asarray(observed_corners_px, dtype=np.float64).reshape(-1, 2)
    points = np.asarray(object_points_tag0_m, dtype=np.float64).reshape(-1, 3)
    if observed.shape[0] != points.shape[0]:
        raise ValueError("corner and object-point counts must match")
    try:
        predicted = _project(pose, points, camera)
        raw = observed.reshape(-1) - predicted
        degrees_of_freedom = max(1, raw.size - 6)
        sigma_squared = float(np.dot(raw, raw) / degrees_of_freedom)
        sigma_squared = float(
            np.clip(
                sigma_squared,
                config.pixel_sigma_floor_px**2,
                config.pixel_sigma_ceiling_px**2,
            )
        )
        pose_jacobian = _central_jacobian(
            lambda delta: _project(pose @ exp_se3(delta), points, camera)
        )
        parameter_scale = np.diag(
            [0.03, 0.03, 0.03, np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5)]
        )
        scaled_jacobian = pose_jacobian @ parameter_scale
        singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
        threshold = (
            np.finfo(np.float64).eps
            * max(scaled_jacobian.shape)
            * singular_values[0]
            if singular_values.size
            else float("inf")
        )
        rank = int(np.sum(singular_values > threshold))
        if rank < 6:
            return _invalid(rank, "PNP_COVARIANCE_INVALID_RANK")
        normal = (scaled_jacobian.T @ scaled_jacobian) / sigma_squared
        largest = float(np.max(np.linalg.eigvalsh(normal)))
        regularized = normal + config.normal_matrix_rcond * largest * np.eye(6)
        scaled_covariance = np.linalg.inv(regularized)
        pose_covariance = parameter_scale @ scaled_covariance @ parameter_scale

        residual_jacobian = _central_jacobian(
            lambda delta: np.asarray(
                closure_residual(pose @ exp_se3(delta)), dtype=np.float64
            ).reshape(6)
        )
        residual_covariance = residual_jacobian @ pose_covariance @ residual_jacobian.T
        floor = np.diag(
            [
                config.unmodelled_translation_floor_m**2,
                config.unmodelled_translation_floor_m**2,
                config.unmodelled_translation_floor_m**2,
                np.deg2rad(config.unmodelled_rotation_floor_deg) ** 2,
                np.deg2rad(config.unmodelled_rotation_floor_deg) ** 2,
                np.deg2rad(config.unmodelled_rotation_floor_deg) ** 2,
            ]
        )
        residual_covariance = 0.5 * (
            residual_covariance + residual_covariance.T
        ) + floor
        residual_scale = np.diag(
            [
                1.0 / 0.03,
                1.0 / 0.03,
                1.0 / 0.03,
                1.0 / np.deg2rad(0.5),
                1.0 / np.deg2rad(0.5),
                1.0 / np.deg2rad(0.5),
            ]
        )
        dimensionless = residual_scale @ residual_covariance @ residual_scale.T
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (dimensionless + dimensionless.T))
        eigenvalues = np.clip(
            eigenvalues,
            config.scaled_eigenvalue_min,
            config.scaled_eigenvalue_max,
        )
        dimensionless = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        inverse_scale = np.linalg.inv(residual_scale)
        residual_covariance = inverse_scale @ dimensionless @ inverse_scale.T
        cholesky = np.linalg.cholesky(residual_covariance)
        whitener = np.linalg.inv(cholesky)
        if not all(
            np.all(np.isfinite(value))
            for value in (pose_covariance, residual_covariance, whitener)
        ):
            return _invalid(rank, "PNP_COVARIANCE_INVALID_NONFINITE")
        return CovarianceEstimate(
            True,
            None,
            rank,
            float(np.sqrt(sigma_squared)),
            pose_covariance,
            residual_covariance,
            whitener,
        )
    except (cv2.error, ValueError, np.linalg.LinAlgError, FloatingPointError):
        return _invalid(0, "PNP_COVARIANCE_INVALID_NUMERICAL")
