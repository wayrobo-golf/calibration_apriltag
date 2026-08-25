"""Recompute dual SQPnP and single-Tag IPPE after geometry freeze."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration_types import CameraModel, FrameObservation, Transform
from .pnp_gate import (
    DualTagPnpGateResult,
    IppeCandidateComparison,
    evaluate_dual_tag_frame,
)
from .se3 import inverse, log_se3
from .tag_geometry import FrozenTagGeometry, tag_square_points


@dataclass(frozen=True)
class SolvedPnPCorners:
    camera_tag0: Transform | None
    object_points_tag0_m: np.ndarray
    dual_reprojection_rms_px: float | None
    minimum_depth_m: float | None
    tag0_gate: object
    tag1_gate: object
    training_gate_pass: bool
    passed: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        points = np.array(self.object_points_tag0_m, dtype=np.float64, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "object_points_tag0_m", points)


@dataclass(frozen=True)
class PreparedPnPFrame(SolvedPnPCorners):
    source: FrameObservation


@dataclass(frozen=True)
class SingleTagPnPResult:
    candidates: tuple[Transform, ...]
    selected: Transform | None
    reprojection_rms_px: float | None


def _transform(rotation_vector: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation_vector, dtype=np.float64).reshape(3)
    return result


def _depths(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T[:, 2] + transform[2, 3]


def _project(
    transform: np.ndarray, points: np.ndarray, camera: CameraModel
) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
    pixels, _ = cv2.projectPoints(
        points,
        rotation_vector,
        transform[:3, 3],
        camera.matrix,
        camera.distortion,
    )
    return pixels.reshape(-1, 2)


def _ippe_candidates(
    points: np.ndarray,
    pixels: np.ndarray,
    camera: CameraModel,
) -> tuple[np.ndarray, ...]:
    result = cv2.solvePnPGeneric(
        points,
        pixels,
        camera.matrix,
        camera.distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not result or not bool(result[0]):
        return ()
    return tuple(_transform(rvec, tvec) for rvec, tvec in zip(result[1], result[2]))


def solve_single_tag_pnp(
    corners_px: np.ndarray,
    camera: CameraModel,
    tag_size_m: float,
) -> SingleTagPnPResult:
    points = tag_square_points(tag_size_m)
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    matrices = tuple(
        value
        for value in _ippe_candidates(points, corners, camera)
        if np.all(_depths(value, points) > 0.0)
    )
    candidates = tuple(
        Transform(value, "left_camera", "tag") for value in matrices
    )
    if not candidates:
        return SingleTagPnPResult((), None, None)

    def rms(candidate: Transform) -> float:
        residual = _project(candidate.matrix, points, camera) - corners
        return float(np.sqrt(np.mean(np.square(residual))))

    selected = min(candidates, key=rms)
    return SingleTagPnPResult(candidates, selected, rms(selected))


def _comparison(dual: np.ndarray, candidate: np.ndarray, positive: bool) -> IppeCandidateComparison:
    difference = log_se3(inverse(dual) @ candidate)
    return IppeCandidateComparison(
        positive_depth=positive,
        translation_error_m=float(np.linalg.norm(difference[:3])),
        rotation_error_deg=float(np.degrees(np.linalg.norm(difference[3:]))),
    )


def solve_corners_with_frozen_geometry(
    corners_px: np.ndarray,
    camera: CameraModel,
    tag_size_m: float,
    frozen_geometry: FrozenTagGeometry,
    *,
    maximum_translation_error_m: float = 0.10,
    maximum_rotation_error_deg: float = 2.0,
) -> SolvedPnPCorners:
    square = tag_square_points(tag_size_m)
    geometry = frozen_geometry.transform.matrix
    tag1_in_tag0 = (geometry[:3, :3] @ square.T).T + geometry[:3, 3]
    object_points = np.vstack((square, tag1_in_tag0))
    corners = np.asarray(corners_px, dtype=np.float64).reshape(8, 2)
    empty_gate = evaluate_dual_tag_frame((), ())
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            corners,
            camera.matrix,
            camera.distortion,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not success:
            raise ValueError("SQPnP returned failure")
        dual = _transform(rotation_vector, translation_vector)
        depth = _depths(dual, object_points)
        if not np.all(np.isfinite(dual)) or np.any(depth <= 0.0):
            raise ValueError("SQPnP returned non-positive depth")
        predicted = _project(dual, object_points, camera)
        rms = float(np.sqrt(np.mean(np.square(predicted - corners))))

        tag0_candidates = _ippe_candidates(square, corners[:4], camera)
        tag1_candidates = _ippe_candidates(square, corners[4:], camera)
        tag0_comparisons = tuple(
            _comparison(dual, candidate, bool(np.all(_depths(candidate, square) > 0.0)))
            for candidate in tag0_candidates
        )
        tag1_comparisons = tuple(
            _comparison(
                dual,
                candidate @ inverse(geometry),
                bool(np.all(_depths(candidate, square) > 0.0)),
            )
            for candidate in tag1_candidates
        )
        gate: DualTagPnpGateResult = evaluate_dual_tag_frame(
            tag0_comparisons,
            tag1_comparisons,
            maximum_translation_error_m,
            maximum_rotation_error_deg,
        )
        reasons: list[str] = []
        if not gate.tag0.passed:
            reasons.append("CAL-E-PNP-TAG0-CANDIDATE-GATE")
        if not gate.tag1.passed:
            reasons.append("CAL-E-PNP-TAG1-CANDIDATE-GATE")
        passed = not reasons
        return SolvedPnPCorners(
            camera_tag0=Transform(dual, "left_camera", "tag0"),
            object_points_tag0_m=object_points,
            dual_reprojection_rms_px=rms,
            minimum_depth_m=float(np.min(depth)),
            tag0_gate=gate.tag0,
            tag1_gate=gate.tag1,
            training_gate_pass=passed,
            passed=passed,
            reasons=tuple(reasons),
        )
    except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
        reason = (
            "CAL-E-PNP-NONPOSITIVE-DEPTH"
            if "depth" in str(error).lower()
            else "CAL-E-PNP-SOLVER"
        )
        return SolvedPnPCorners(
            camera_tag0=None,
            object_points_tag0_m=object_points,
            dual_reprojection_rms_px=None,
            minimum_depth_m=None,
            tag0_gate=empty_gate.tag0,
            tag1_gate=empty_gate.tag1,
            training_gate_pass=False,
            passed=False,
            reasons=(reason,),
        )


def solve_frame_with_frozen_geometry(
    frame: FrameObservation,
    camera: CameraModel,
    tag_size_m: float,
    frozen_geometry: FrozenTagGeometry,
    *,
    maximum_translation_error_m: float = 0.10,
    maximum_rotation_error_deg: float = 2.0,
) -> PreparedPnPFrame:
    solved = solve_corners_with_frozen_geometry(
        frame.corners_px,
        camera,
        tag_size_m,
        frozen_geometry,
        maximum_translation_error_m=maximum_translation_error_m,
        maximum_rotation_error_deg=maximum_rotation_error_deg,
    )
    return PreparedPnPFrame(
        camera_tag0=solved.camera_tag0,
        object_points_tag0_m=solved.object_points_tag0_m,
        dual_reprojection_rms_px=solved.dual_reprojection_rms_px,
        minimum_depth_m=solved.minimum_depth_m,
        tag0_gate=solved.tag0_gate,
        tag1_gate=solved.tag1_gate,
        training_gate_pass=solved.training_gate_pass,
        passed=solved.passed,
        reasons=solved.reasons,
        source=frame,
    )
