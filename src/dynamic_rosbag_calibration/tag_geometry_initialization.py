"""Deterministic bootstrap/reuse initialization for Tag0-to-Tag1 geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .calibration_types import CameraModel, FrameObservation, Transform
from .pnp_prepare import solve_corners_with_frozen_geometry, solve_single_tag_pnp
from .se3 import inverse, log_se3
from .tag_geometry import FrozenTagGeometry, select_balanced_views


class TagGeometryInitializationError(ValueError):
    pass


@dataclass(frozen=True)
class TagGeometryInitializationResult:
    mode: str
    initial_transform: Transform
    seeded_frames: tuple[FrameObservation, ...]
    source_path: Path | None
    source_sha256: str | None


@dataclass(frozen=True)
class _Candidate:
    transform: Transform
    station_id: str
    frame_key: str
    candidate_key: str


def _difference(left: Transform, right: Transform) -> tuple[float, float]:
    delta = log_se3(inverse(left.matrix) @ right.matrix)
    return (
        float(np.linalg.norm(delta[:3])),
        float(np.degrees(np.linalg.norm(delta[3:]))),
    )


def _components(candidates: Sequence[_Candidate]) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in candidates]
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            translation_m, rotation_deg = _difference(
                candidates[left].transform, candidates[right].transform
            )
            if translation_m <= 0.10 and rotation_deg <= 5.0:
                adjacency[left].add(right)
                adjacency[right].add(left)
    unseen = set(range(len(candidates)))
    result: list[tuple[int, ...]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component: list[int] = []
        unseen.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        result.append(tuple(sorted(component)))
    return tuple(result)


def _component_medoid(
    component: Sequence[int], candidates: Sequence[_Candidate]
) -> tuple[_Candidate, float]:
    scored: list[tuple[float, str, _Candidate]] = []
    for index in component:
        distances = []
        for other in component:
            translation_m, rotation_deg = _difference(
                candidates[index].transform, candidates[other].transform
            )
            distances.append(translation_m / 0.10 + rotation_deg / 5.0)
        scored.append(
            (
                float(np.median(distances)),
                candidates[index].candidate_key,
                candidates[index],
            )
        )
    score, _, medoid = min(scored, key=lambda item: (item[0], item[1]))
    return medoid, score


def _bootstrap_initial(
    frames: Sequence[FrameObservation],
    camera: CameraModel,
    tag_size_m: float,
    minimum_valid_stations: int,
) -> Transform:
    candidates: list[_Candidate] = []
    for frame in sorted(frames, key=lambda item: item.frame_key):
        if not frame.basic_valid or not frame.station_gate_pass:
            continue
        tag0 = solve_single_tag_pnp(frame.corners_px[:4], camera, tag_size_m)
        tag1 = solve_single_tag_pnp(frame.corners_px[4:], camera, tag_size_m)
        for tag0_index, camera_tag0 in enumerate(tag0.candidates):
            for tag1_index, camera_tag1 in enumerate(tag1.candidates):
                relative = Transform(
                    inverse(camera_tag0.matrix) @ camera_tag1.matrix,
                    "tag0",
                    "tag1",
                )
                key = f"{frame.frame_key}/{tag0_index}/{tag1_index}"
                candidates.append(
                    _Candidate(relative, frame.station_id, frame.frame_key, key)
                )
    if not candidates:
        raise TagGeometryInitializationError(
            "CAL-E-TAG-GEOMETRY-INITIALIZATION: no positive-depth candidates"
        )

    ranked = []
    for component in _components(candidates):
        stations = {candidates[index].station_id for index in component}
        frames_in_component = {candidates[index].frame_key for index in component}
        medoid, median_distance = _component_medoid(component, candidates)
        ranked.append(
            (
                -len(stations),
                -len(frames_in_component),
                median_distance,
                medoid.candidate_key,
                medoid,
            )
        )
    ranked.sort(key=lambda item: item[:4])
    best = ranked[0]
    if -best[0] < minimum_valid_stations:
        raise TagGeometryInitializationError(
            "CAL-E-TAG-GEOMETRY-INITIALIZATION-STATION-COUNT: "
            f"bootstrap requires {minimum_valid_stations} stations"
        )
    if len(ranked) > 1:
        second = ranked[1]
        if (
            best[0] == second[0]
            and best[1] == second[1]
            and abs(best[2] - second[2]) <= 1.0e-12
        ):
            raise TagGeometryInitializationError(
                "CAL-E-TAG-GEOMETRY-AMBIGUOUS: tied bootstrap clusters"
            )
    return best[4].transform


def _seed_frames(
    frames: Sequence[FrameObservation],
    camera: CameraModel,
    tag_size_m: float,
    initial: Transform,
    minimum_valid_stations: int,
) -> tuple[FrameObservation, ...]:
    frozen = FrozenTagGeometry(initial, ())
    seeded: list[FrameObservation] = []
    for frame in sorted(frames, key=lambda item: item.frame_key):
        if not frame.basic_valid or not frame.station_gate_pass:
            continue
        solved = solve_corners_with_frozen_geometry(
            frame.corners_px, camera, tag_size_m, frozen
        )
        if solved.camera_tag0 is not None:
            seeded.append(replace(frame, camera_tag0=solved.camera_tag0))
    if len({frame.station_id for frame in seeded}) < minimum_valid_stations:
        raise TagGeometryInitializationError(
            "CAL-E-TAG-GEOMETRY-INITIALIZATION-STATION-COUNT: "
            f"fewer than {minimum_valid_stations} stations could be seeded"
        )
    return tuple(seeded)


def initialize_tag_geometry(
    frames: Sequence[FrameObservation],
    camera: CameraModel,
    tag_size_m: float,
    mode: str,
    reuse_initial: Transform | None,
    source_path: str | Path | None,
    source_sha256: str | None,
    *,
    bootstrap_maximum_views_per_station: int = 5,
    minimum_valid_stations: int = 6,
) -> TagGeometryInitializationResult:
    if minimum_valid_stations <= 0:
        raise TagGeometryInitializationError(
            "minimum_valid_stations must be positive"
        )
    if mode == "bootstrap":
        if reuse_initial is not None or source_path is not None or source_sha256 is not None:
            raise TagGeometryInitializationError(
                "bootstrap must not receive reuse geometry provenance"
            )
        bootstrap_frames = select_balanced_views(
            frames, bootstrap_maximum_views_per_station
        )
        initial = _bootstrap_initial(
            bootstrap_frames,
            camera,
            tag_size_m,
            minimum_valid_stations,
        )
        resolved_path = None
        resolved_sha256 = None
    elif mode == "reuse":
        if reuse_initial is None or source_path is None or source_sha256 is None:
            raise TagGeometryInitializationError(
                "reuse requires an initial transform, source path, and SHA-256"
            )
        initial = reuse_initial
        resolved_path = Path(source_path)
        resolved_sha256 = str(source_sha256)
    else:
        raise TagGeometryInitializationError("mode must be bootstrap or reuse")
    return TagGeometryInitializationResult(
        mode=mode,
        initial_transform=initial,
        seeded_frames=_seed_frames(
            frames,
            camera,
            tag_size_m,
            initial,
            minimum_valid_stations,
        ),
        source_path=resolved_path,
        source_sha256=resolved_sha256,
    )
