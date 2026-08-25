"""Scaled Jacobian rank and condition-number diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservabilityReport:
    passed: bool
    reasons: tuple[str, ...]
    numerical_rank: int
    column_count: int
    condition_number: float
    singular_values: tuple[float, ...]
    rank_threshold: float


def diagnose_observability(
    jacobian: np.ndarray,
    parameter_scale: np.ndarray,
    maximum_condition_number: float,
) -> ObservabilityReport:
    matrix = np.asarray(jacobian, dtype=np.float64)
    scale = np.asarray(parameter_scale, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != scale.size:
        raise ValueError("jacobian columns must match parameter scale")
    scaled = matrix * scale[None, :]
    singular = np.linalg.svd(scaled, compute_uv=False)
    threshold = (
        float(np.finfo(np.float64).eps * max(scaled.shape) * singular[0])
        if singular.size
        else float("inf")
    )
    observable = singular[singular > threshold]
    rank = int(observable.size)
    condition = (
        float(observable[0] / observable[-1]) if observable.size else float("inf")
    )
    reasons = []
    if rank != matrix.shape[1]:
        reasons.append("CAL-E-CALIBRATION-JACOBIAN-RANK")
    if not np.isfinite(condition) or condition > maximum_condition_number:
        reasons.append("CAL-E-CALIBRATION-CONDITION-NUMBER")
    return ObservabilityReport(
        passed=not reasons,
        reasons=tuple(reasons),
        numerical_rank=rank,
        column_count=matrix.shape[1],
        condition_number=condition,
        singular_values=tuple(float(value) for value in singular),
        rank_threshold=threshold,
    )
