"""Production B2 pose-domain rotation and nuisance-board optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from .calibration_types import CalibrationProblem, Transform
from .covariance import CovarianceConfig, CovarianceEstimate, estimate_pnp_covariance
from .observability import ObservabilityReport, diagnose_observability
from .se3 import (
    exp_se3,
    inverse,
    log_se3,
    right_correct_rx_ry,
    right_correct_rx_ry_rz,
)
from .tag_geometry import tag_square_points


TRANSLATION_SCALE_M = 0.03
ROTATION_SCALE_RAD = math.radians(0.5)


@dataclass(frozen=True)
class B2SolverConfig:
    max_nfev_per_start: int = 500
    xtol: float = 1.0e-11
    ftol: float = 1.0e-11
    gtol: float = 1.0e-11
    extrinsic_rotation_bound_deg: float = 5.0
    irls_max_iterations: int = 10
    irls_weight_tolerance: float = 1.0e-6
    mad_consistency_factor: float = 1.4826
    cutoff_mad_multiplier: float = 1.5
    cutoff_min: float = 1.0
    cutoff_max: float = 10.0
    scaled_condition_number_max: float = 1000.0
    multistart_extrinsic_spread_max_deg: float = 0.02
    multistart_board_translation_spread_max_m: float = 0.005
    multistart_board_rotation_spread_max_deg: float = 0.05
    minimum_successful_multistarts: int = 21
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
    extrinsic_rotation_parameter_count: int = 2

    def __post_init__(self) -> None:
        if self.extrinsic_rotation_parameter_count not in (2, 3):
            raise ValueError(
                "extrinsic_rotation_parameter_count must be either 2 or 3"
            )


@dataclass(frozen=True)
class MultiStart:
    name: str
    extrinsic_rotation_rad: tuple[float, ...]
    common_board_delta: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class MultiStartResult:
    name: str
    success: bool
    cost: float
    nfev: int
    message: str
    parameters: tuple[float, ...]
    huber_cutoff: float
    covariance_invalid_count: int
    observability: ObservabilityReport
    effective_extrinsic_matrix: np.ndarray
    board_matrices: dict[str, np.ndarray]


@dataclass(frozen=True)
class CalibrationResult:
    algorithm_id: str
    effective_extrinsic: Transform
    board_poses: dict[str, Transform]
    right_correction_rotvec_rad: tuple[float, float, float]
    map_origin_m: tuple[float, float, float]
    quality_gate_pass: bool
    reasons: tuple[str, ...]
    best_start_name: str
    objective_cost: float
    huber_cutoff: float
    observability: ObservabilityReport
    multistart_results: tuple[MultiStartResult, ...]
    extrinsic_multistart_spread_deg: float
    board_translation_multistart_spread_m: float
    board_rotation_multistart_spread_deg: float
    covariance_invalid_count: int


def station_sqrt_weights(station_ids: Sequence[str]) -> np.ndarray:
    values = np.asarray(station_ids, dtype=str)
    unique, counts = np.unique(values, return_counts=True)
    mapping = dict(zip(unique, counts))
    return np.asarray([1.0 / np.sqrt(mapping[value]) for value in values], dtype=np.float64)


def deterministic_multistarts(
    board_ids: Sequence[str],
    extrinsic_rotation_parameter_count: int = 2,
) -> tuple[MultiStart, ...]:
    del board_ids
    if extrinsic_rotation_parameter_count not in (2, 3):
        raise ValueError(
            "extrinsic_rotation_parameter_count must be either 2 or 3"
        )
    half_degree = math.radians(0.5)
    zero = (0.0,) * 6
    if extrinsic_rotation_parameter_count == 2:
        extrinsic_starts = (
            ("zero", (0.0, 0.0)),
            ("rx_plus", (half_degree, 0.0)),
            ("rx_minus", (-half_degree, 0.0)),
            ("ry_plus", (0.0, half_degree)),
            ("ry_minus", (0.0, -half_degree)),
            ("rx_plus_ry_plus", (half_degree, half_degree)),
            ("rx_plus_ry_minus", (half_degree, -half_degree)),
            ("rx_minus_ry_plus", (-half_degree, half_degree)),
            ("rx_minus_ry_minus", (-half_degree, -half_degree)),
        )
    else:
        extrinsic_starts = (
            ("zero", (0.0, 0.0, 0.0)),
            ("rx_plus", (half_degree, 0.0, 0.0)),
            ("rx_minus", (-half_degree, 0.0, 0.0)),
            ("ry_plus", (0.0, half_degree, 0.0)),
            ("ry_minus", (0.0, -half_degree, 0.0)),
            ("rz_plus", (0.0, 0.0, half_degree)),
            ("rz_minus", (0.0, 0.0, -half_degree)),
            ("rxyz_plus", (half_degree, half_degree, half_degree)),
            ("rxyz_minus", (-half_degree, -half_degree, -half_degree)),
        )
    starts = [MultiStart(name, correction, zero) for name, correction in extrinsic_starts]
    no_extrinsic = (0.0,) * extrinsic_rotation_parameter_count
    axes = ("x", "y", "z")
    for index, axis in enumerate(axes):
        for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
            delta = [0.0] * 6
            delta[index] = sign * 0.05
            starts.append(MultiStart(f"board_t{axis}_{sign_name}", no_extrinsic, tuple(delta)))
    for index, axis in enumerate(axes, start=3):
        for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
            delta = [0.0] * 6
            delta[index] = sign * half_degree
            starts.append(MultiStart(f"board_r{axis}_{sign_name}", no_extrinsic, tuple(delta)))
    return tuple(starts)


def _se3_center(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("cannot calculate center of empty transform sequence")
    center = np.array(values[0], copy=True)
    scale = np.array(
        [TRANSLATION_SCALE_M] * 3 + [ROTATION_SCALE_RAD] * 3,
        dtype=np.float64,
    )
    for _ in range(50):
        deltas = np.asarray([log_se3(inverse(center) @ value) for value in values])
        norms = np.linalg.norm(deltas / scale, axis=1)
        weights = np.minimum(1.0, 1.0 / np.maximum(norms, 1.0e-12))
        update = np.sum(deltas * weights[:, None], axis=0) / np.sum(weights)
        center = center @ exp_se3(update)
        if np.linalg.norm(update / scale) < 1.0e-10:
            break
    return center


def initialize_board_poses(
    problem: CalibrationProblem,
    map_ins_matrices: Sequence[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    maps = (
        [frame.map_ins.matrix for frame in problem.frames]
        if map_ins_matrices is None
        else list(map_ins_matrices)
    )
    station_centers: dict[tuple[str, str], np.ndarray] = {}
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for frame, map_ins in zip(problem.frames, maps):
        grouped.setdefault((frame.board_instance_id, frame.station_id), []).append(
            map_ins @ problem.nominal_extrinsic.matrix @ frame.camera_tag0.matrix
        )
    for key, values in grouped.items():
        station_centers[key] = _se3_center(values)
    result = {}
    for board in sorted({frame.board_instance_id for frame in problem.frames}):
        result[board] = _se3_center(
            [value for (board_id, _), value in station_centers.items() if board_id == board]
        )
    return result


def _huber_pseudo(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    result = values.copy()
    mask = magnitude > 1.0
    result[mask] = np.sign(values[mask]) * np.sqrt(2.0 * magnitude[mask] - 1.0)
    return result


@dataclass
class _RunContext:
    problem: CalibrationProblem
    config: B2SolverConfig
    map_origin: np.ndarray
    map_ins: list[np.ndarray]
    board_ids: tuple[str, ...]
    nominal_boards: dict[str, np.ndarray]
    station_weights: np.ndarray
    parameter_scale: np.ndarray
    object_points: np.ndarray

    def unpack(self, parameters: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        extrinsic_count = self.config.extrinsic_rotation_parameter_count
        if extrinsic_count == 2:
            extrinsic = right_correct_rx_ry(
                self.problem.nominal_extrinsic.matrix,
                parameters[0],
                parameters[1],
            )
        else:
            extrinsic = right_correct_rx_ry_rz(
                self.problem.nominal_extrinsic.matrix,
                parameters[0],
                parameters[1],
                parameters[2],
            )
        boards = {
            board: self.nominal_boards[board]
            @ exp_se3(
                parameters[
                    extrinsic_count + 6 * index : extrinsic_count + 6 * (index + 1)
                ]
            )
            for index, board in enumerate(self.board_ids)
        }
        return extrinsic, boards

    def closure_values(self, parameters: np.ndarray) -> np.ndarray:
        extrinsic, boards = self.unpack(parameters)
        values = np.empty((len(self.problem.frames), 6), dtype=np.float64)
        for index, frame in enumerate(self.problem.frames):
            values[index] = log_se3(
                inverse(boards[frame.board_instance_id])
                @ self.map_ins[index]
                @ extrinsic
                @ frame.camera_tag0.matrix
            )
        return values


def _bounds(context: _RunContext) -> tuple[np.ndarray, np.ndarray]:
    size = context.parameter_scale.size
    lower = np.full(size, -np.inf, dtype=np.float64)
    upper = np.full(size, np.inf, dtype=np.float64)
    bound = math.radians(context.config.extrinsic_rotation_bound_deg)
    count = context.config.extrinsic_rotation_parameter_count
    lower[:count] = -bound
    upper[:count] = bound
    return lower, upper


def _run_start(context: _RunContext, start: MultiStart) -> MultiStartResult:
    board_count = len(context.board_ids)
    extrinsic_count = context.config.extrinsic_rotation_parameter_count
    board_initial = np.tile(np.asarray(start.common_board_delta), board_count)
    fixed_scale = np.array(
        [TRANSLATION_SCALE_M] * 3 + [ROTATION_SCALE_RAD] * 3,
        dtype=np.float64,
    )

    def s0_residual(board_parameters: np.ndarray) -> np.ndarray:
        full = np.concatenate((np.asarray(start.extrinsic_rotation_rad), board_parameters))
        values = context.closure_values(full) / fixed_scale
        return (_huber_pseudo(values) * context.station_weights[:, None]).reshape(-1)

    s0 = least_squares(
        s0_residual,
        board_initial,
        method="trf",
        x_scale=np.tile(
            context.parameter_scale[extrinsic_count : extrinsic_count + 6],
            board_count,
        ),
        max_nfev=context.config.max_nfev_per_start,
        xtol=context.config.xtol,
        ftol=context.config.ftol,
        gtol=context.config.gtol,
    )
    initial = np.concatenate((np.asarray(start.extrinsic_rotation_rad), s0.x))

    def s1_residual(parameters: np.ndarray) -> np.ndarray:
        values = context.closure_values(parameters) / fixed_scale
        return (_huber_pseudo(values) * context.station_weights[:, None]).reshape(-1)

    s1 = least_squares(
        s1_residual,
        initial,
        method="trf",
        bounds=_bounds(context),
        x_scale=context.parameter_scale,
        max_nfev=context.config.max_nfev_per_start,
        xtol=context.config.xtol,
        ftol=context.config.ftol,
        gtol=context.config.gtol,
    )
    s1_extrinsic, s1_boards = context.unpack(s1.x)
    covariance: list[CovarianceEstimate] = []
    for index, frame in enumerate(context.problem.frames):
        board = s1_boards[frame.board_instance_id]
        map_ins = context.map_ins[index]
        covariance.append(
            estimate_pnp_covariance(
                frame.camera_tag0.matrix,
                frame.corners_px,
                context.object_points,
                context.problem.camera_model,
                lambda candidate, board=board, map_ins=map_ins: log_se3(
                    inverse(board) @ map_ins @ s1_extrinsic @ candidate
                ),
                context.config.covariance,
            )
        )
    invalid_count = sum(not item.valid for item in covariance)
    whiteners = [item.whitener if item.whitener is not None else np.eye(6) for item in covariance]

    def whitened_values(parameters: np.ndarray) -> np.ndarray:
        raw = context.closure_values(parameters)
        return np.asarray(
            [whiteners[index] @ raw[index] for index in range(len(raw))],
            dtype=np.float64,
        )

    q = np.linalg.norm(whitened_values(s1.x), axis=1)
    median = float(np.median(q))
    mad = context.config.mad_consistency_factor * float(np.median(np.abs(q - median)))
    cutoff = float(
        np.clip(
            median + context.config.cutoff_mad_multiplier * mad,
            context.config.cutoff_min,
            context.config.cutoff_max,
        )
    )
    parameters = s1.x
    huber_weights = np.ones(len(context.problem.frames), dtype=np.float64)
    fit = s1
    for _ in range(context.config.irls_max_iterations):
        previous = huber_weights.copy()
        norms = np.linalg.norm(whitened_values(parameters), axis=1)
        huber_weights = np.minimum(1.0, cutoff / np.maximum(norms, 1.0e-12))

        def residual(candidate: np.ndarray) -> np.ndarray:
            values = whitened_values(candidate)
            combined = (
                np.sqrt(huber_weights) * context.station_weights
            )[:, None]
            return (values * combined).reshape(-1)

        fit = least_squares(
            residual,
            parameters,
            method="trf",
            bounds=_bounds(context),
            x_scale=context.parameter_scale,
            max_nfev=context.config.max_nfev_per_start,
            xtol=context.config.xtol,
            ftol=context.config.ftol,
            gtol=context.config.gtol,
        )
        parameters = fit.x
        if float(np.max(np.abs(huber_weights - previous))) < context.config.irls_weight_tolerance:
            break
    extrinsic, boards = context.unpack(parameters)
    observability = diagnose_observability(
        fit.jac,
        context.parameter_scale,
        context.config.scaled_condition_number_max,
    )
    success = bool(fit.success) and np.all(np.isfinite(parameters)) and invalid_count == 0
    return MultiStartResult(
        name=start.name,
        success=success,
        cost=float(fit.cost),
        nfev=int(fit.nfev),
        message=str(fit.message),
        parameters=tuple(float(value) for value in parameters),
        huber_cutoff=cutoff,
        covariance_invalid_count=invalid_count,
        observability=observability,
        effective_extrinsic_matrix=extrinsic,
        board_matrices=boards,
    )


def solve_calibration(
    problem: CalibrationProblem,
    config: B2SolverConfig,
) -> CalibrationResult:
    frames = problem.frames
    map_origin = np.median(
        np.asarray([frame.map_ins.matrix[:3, 3] for frame in frames]), axis=0
    )
    local_maps = []
    for frame in frames:
        value = np.array(frame.map_ins.matrix, copy=True)
        value[:3, 3] -= map_origin
        local_maps.append(value)
    nominal_boards = initialize_board_poses(problem, local_maps)
    board_ids = tuple(sorted(nominal_boards))
    one_board_scale = np.array(
        [TRANSLATION_SCALE_M] * 3 + [ROTATION_SCALE_RAD] * 3,
        dtype=np.float64,
    )
    extrinsic_count = config.extrinsic_rotation_parameter_count
    parameter_scale = np.concatenate(
        (
            np.full(extrinsic_count, ROTATION_SCALE_RAD, dtype=np.float64),
            np.tile(one_board_scale, len(board_ids)),
        )
    )
    square = tag_square_points(problem.tag_size_m)
    geometry = problem.initial_tag0_tag1.matrix
    points = np.vstack(
        (square, (geometry[:3, :3] @ square.T).T + geometry[:3, 3])
    )
    context = _RunContext(
        problem=problem,
        config=config,
        map_origin=map_origin,
        map_ins=local_maps,
        board_ids=board_ids,
        nominal_boards=nominal_boards,
        station_weights=station_sqrt_weights([frame.station_id for frame in frames]),
        parameter_scale=parameter_scale,
        object_points=points,
    )
    runs = tuple(
        _run_start(context, start)
        for start in deterministic_multistarts(board_ids, extrinsic_count)
    )
    finite_runs = [run for run in runs if run.success and np.isfinite(run.cost)]
    candidates = finite_runs or [run for run in runs if np.isfinite(run.cost)]
    if not candidates:
        raise RuntimeError("all B2 multi-start solves produced non-finite objectives")
    best = min(candidates, key=lambda run: (run.cost, run.name))

    extrinsic_spread = max(
        float(
            np.degrees(
                np.linalg.norm(
                    log_se3(
                        inverse(best.effective_extrinsic_matrix)
                        @ run.effective_extrinsic_matrix
                    )[3:]
                )
            )
        )
        for run in candidates
    )
    board_translation_spread = 0.0
    board_rotation_spread = 0.0
    for run in candidates:
        for board in board_ids:
            difference = log_se3(
                inverse(best.board_matrices[board]) @ run.board_matrices[board]
            )
            board_translation_spread = max(
                board_translation_spread, float(np.linalg.norm(difference[:3]))
            )
            board_rotation_spread = max(
                board_rotation_spread,
                float(np.degrees(np.linalg.norm(difference[3:]))),
            )

    reasons: list[str] = []
    if len(finite_runs) < config.minimum_successful_multistarts:
        reasons.append("CAL-E-CALIBRATION-MULTISTART-SUCCESS-COUNT")
    if not best.success:
        reasons.append("CAL-E-CALIBRATION-SOLVER")
    reasons.extend(best.observability.reasons)
    if best.covariance_invalid_count:
        reasons.append("CAL-E-CALIBRATION-PNP-COVARIANCE")
    if extrinsic_spread > config.multistart_extrinsic_spread_max_deg:
        reasons.append("CAL-E-CALIBRATION-MULTISTART-EXTRINSIC")
    if board_translation_spread > config.multistart_board_translation_spread_max_m:
        reasons.append("CAL-E-CALIBRATION-MULTISTART-BOARD-TRANSLATION")
    if board_rotation_spread > config.multistart_board_rotation_spread_max_deg:
        reasons.append("CAL-E-CALIBRATION-MULTISTART-BOARD-ROTATION")
    bound = math.radians(config.extrinsic_rotation_bound_deg)
    if any(
        abs(abs(value) - bound) <= 1.0e-8
        for value in best.parameters[:extrinsic_count]
    ):
        reasons.append("CAL-E-CALIBRATION-PARAMETER-BOUND")
    if not all(np.all(np.isfinite(value)) for value in best.board_matrices.values()):
        reasons.append("CAL-E-CALIBRATION-NONFINITE")

    output_boards = {}
    for board, local in best.board_matrices.items():
        value = np.array(local, copy=True)
        value[:3, 3] += map_origin
        output_boards[board] = Transform(value, "ins_map", "tag0")
    local_correction = log_se3(
        inverse(problem.nominal_extrinsic.matrix) @ best.effective_extrinsic_matrix
    )
    return CalibrationResult(
        algorithm_id=("B2" if extrinsic_count == 2 else "B2-RXYZ-ABLATION"),
        effective_extrinsic=Transform(
            best.effective_extrinsic_matrix, "ins_link", "left_camera"
        ),
        board_poses=output_boards,
        right_correction_rotvec_rad=(
            float(local_correction[3]),
            float(local_correction[4]),
            float(local_correction[5]),
        ),
        map_origin_m=tuple(float(value) for value in map_origin),
        quality_gate_pass=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        best_start_name=best.name,
        objective_cost=best.cost,
        huber_cutoff=best.huber_cutoff,
        observability=best.observability,
        multistart_results=runs,
        extrinsic_multistart_spread_deg=extrinsic_spread,
        board_translation_multistart_spread_m=board_translation_spread,
        board_rotation_multistart_spread_deg=board_rotation_spread,
        covariance_invalid_count=best.covariance_invalid_count,
    )
