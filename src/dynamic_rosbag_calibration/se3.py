"""SE(3) operations using tangent order [tx, ty, tz, rx, ry, rz]."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


FloatArray = NDArray[np.float64]


def _skew(value: FloatArray) -> FloatArray:
    x, y, z = np.asarray(value, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def exp_se3(twist: FloatArray) -> FloatArray:
    value = np.asarray(twist, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(value)):
        raise ValueError("SE(3) twist must be finite")
    rho = value[:3]
    omega = value[3:]
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < 1e-8:
        v_matrix = np.eye(3) + 0.5 * omega_hat + (1.0 / 6.0) * (omega_hat @ omega_hat)
    else:
        theta2 = theta * theta
        v_matrix = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta2 * omega_hat
            + (theta - np.sin(theta)) / (theta2 * theta) * (omega_hat @ omega_hat)
        )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(omega).as_matrix()
    result[:3, 3] = v_matrix @ rho
    return result


def log_se3(transform: FloatArray) -> FloatArray:
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(value)):
        raise ValueError("SE(3) transform must be finite")
    omega = Rotation.from_matrix(
        np.array(value[:3, :3], dtype=np.float64, copy=True)
    ).as_rotvec()
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < 1e-8:
        v_inverse = np.eye(3) - 0.5 * omega_hat + (1.0 / 12.0) * (omega_hat @ omega_hat)
    else:
        theta2 = theta * theta
        coefficient = (1.0 / theta2) - (
            (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        )
        v_inverse = np.eye(3) - 0.5 * omega_hat + coefficient * (omega_hat @ omega_hat)
    rho = v_inverse @ value[:3, 3]
    return np.concatenate((rho, omega)).astype(np.float64)


def inverse(transform: FloatArray) -> FloatArray:
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def compose(*transforms: FloatArray) -> FloatArray:
    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        result = result @ np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return result


def right_correct_rx_ry(
    nominal_extrinsic: FloatArray,
    rx_rad: float,
    ry_rad: float,
) -> FloatArray:
    correction = exp_se3(
        np.array([0.0, 0.0, 0.0, float(rx_rad), float(ry_rad), 0.0])
    )
    return compose(nominal_extrinsic, correction)


def right_correct_rx_ry_rz(
    nominal_extrinsic: FloatArray,
    rx_rad: float,
    ry_rad: float,
    rz_rad: float,
) -> FloatArray:
    """Apply an experimental full three-axis right rotation correction.

    Translation remains frozen.  This helper is intentionally separate from
    ``right_correct_rx_ry`` so production callers retain the existing two-axis
    calibration model unless they opt in explicitly.
    """
    correction = exp_se3(
        np.array(
            [
                0.0,
                0.0,
                0.0,
                float(rx_rad),
                float(ry_rad),
                float(rz_rad),
            ]
        )
    )
    return compose(nominal_extrinsic, correction)
