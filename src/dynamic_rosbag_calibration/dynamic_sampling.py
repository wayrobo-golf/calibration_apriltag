"""Deterministic de-correlation, balancing, and coverage gates."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

import numpy as np

from .dynamic_experiment_types import (
    DynamicCoverageReport,
    DynamicFrameEvidence,
    DynamicSamplingConfig,
    DynamicSelection,
    StructuralState,
)
from .io import canonical_json_bytes, sha256_bytes


DISTANCE_BINS = ("near", "middle", "far")
BEARING_BINS = ("low", "middle", "high")


def _yaw_deg(matrix: np.ndarray) -> float:
    return math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))


def _wrapped_difference_deg(left: float, right: float) -> float:
    value = math.radians(left - right)
    return math.degrees(math.atan2(math.sin(value), math.cos(value)))


def _distance_bin(depth_m: float, edges: tuple[float, float]) -> str:
    if depth_m <= edges[0]:
        return "near"
    if depth_m <= edges[1]:
        return "middle"
    return "far"


def _bearing_bin(value: float, boundaries: tuple[float, float]) -> str:
    if value < boundaries[0]:
        return "low"
    if value > boundaries[1]:
        return "high"
    return "middle"


def _evenly_spaced(values: Sequence[DynamicFrameEvidence], count: int) -> list[DynamicFrameEvidence]:
    rows = list(values)
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(index)] for index in indices]


def _keyframes(
    values: Sequence[DynamicFrameEvidence],
    config: DynamicSamplingConfig,
) -> tuple[DynamicFrameEvidence, ...]:
    grouped: dict[str, list[DynamicFrameEvidence]] = defaultdict(list)
    for item in values:
        grouped[item.source.bag_id].append(item)
    selected: list[DynamicFrameEvidence] = []
    for bag_id in sorted(grouped):
        rows = sorted(
            grouped[bag_id],
            key=lambda item: (item.source.source_stamp_s, item.source.frame_key),
        )
        previous: DynamicFrameEvidence | None = None
        for item in rows:
            observation = item.source.observation
            if observation is None:
                continue
            if previous is None:
                selected.append(item)
                previous = item
                continue
            previous_observation = previous.source.observation
            assert previous_observation is not None
            xy = float(
                np.linalg.norm(
                    observation.map_ins.matrix[:2, 3]
                    - previous_observation.map_ins.matrix[:2, 3]
                )
            )
            yaw = abs(
                _wrapped_difference_deg(
                    _yaw_deg(observation.map_ins.matrix),
                    _yaw_deg(previous_observation.map_ins.matrix),
                )
            )
            elapsed = item.source.source_stamp_s - previous.source.source_stamp_s
            if (
                xy >= config.minimum_xy_increment_m
                or yaw >= config.minimum_yaw_increment_deg
                or elapsed >= config.maximum_time_increment_s
            ):
                selected.append(item)
                previous = item
    return tuple(selected)


def select_balanced_dynamic_frames(
    frames: Sequence[DynamicFrameEvidence],
    sampling: DynamicSamplingConfig,
    *,
    distance_edges_m: tuple[float, float],
    minimum_bearing_span_deg: float,
) -> DynamicSelection:
    """Select a stable frame panel and return explicit dynamic-B1 coverage."""
    safe = tuple(
        sorted(
            (
                item
                for item in frames
                if item.structural_state is StructuralState.UNLOADED_SAFE
                and item.source.observation is not None
                and item.source.initial_tag0_depth_m is not None
                and item.source.bearing_x_deg is not None
            ),
            key=lambda item: (
                item.source.bag_id,
                item.source.source_stamp_s,
                item.source.frame_key,
            ),
        )
    )
    keyframes = _keyframes(safe, sampling)
    bearings = np.asarray(
        [float(item.source.bearing_x_deg) for item in keyframes],
        dtype=np.float64,
    )
    boundaries = (
        tuple(np.quantile(bearings, [1.0 / 3.0, 2.0 / 3.0], method="linear"))
        if bearings.size
        else (0.0, 0.0)
    )
    cells: dict[tuple[str, str, str], list[DynamicFrameEvidence]] = defaultdict(list)
    for item in keyframes:
        depth = float(item.source.initial_tag0_depth_m)
        bearing = float(item.source.bearing_x_deg)
        key = (
            item.source.bag_id,
            _distance_bin(depth, distance_edges_m),
            _bearing_bin(bearing, boundaries),
        )
        cells[key].append(item)

    by_bag: dict[str, list[DynamicFrameEvidence]] = defaultdict(list)
    for key in sorted(cells):
        rows = sorted(
            cells[key],
            key=lambda item: (item.source.source_stamp_s, item.source.frame_key),
        )
        by_bag[key[0]].extend(
            _evenly_spaced(rows, sampling.maximum_frames_per_cell)
        )

    selected: list[DynamicFrameEvidence] = []
    for bag_id in sorted(by_bag):
        unique = {
            item.source.frame_key: item for item in by_bag[bag_id]
        }
        rows = sorted(
            unique.values(),
            key=lambda item: (item.source.source_stamp_s, item.source.frame_key),
        )
        selected.extend(_evenly_spaced(rows, sampling.maximum_frames_per_bag))
    selected.sort(
        key=lambda item: (
            item.source.bag_id,
            item.source.source_stamp_s,
            item.source.frame_key,
        )
    )

    bag_ids = {item.source.bag_id for item in selected}
    distance_frames: dict[str, list[DynamicFrameEvidence]] = {
        name: [] for name in DISTANCE_BINS
    }
    for item in selected:
        distance_frames[
            _distance_bin(float(item.source.initial_tag0_depth_m), distance_edges_m)
        ].append(item)
    distance_bag_counts = tuple(
        (
            name,
            len({item.source.bag_id for item in distance_frames[name]}),
        )
        for name in DISTANCE_BINS
    )
    distance_frame_counts = tuple(
        (name, len(distance_frames[name])) for name in DISTANCE_BINS
    )
    bearings_by_bag: dict[str, list[float]] = defaultdict(list)
    for item in selected:
        bearings_by_bag[item.source.bag_id].append(
            float(item.source.bearing_x_deg)
        )
    # Dynamic frames are not independent stations.  Use each bag's median as
    # one station-like view so a single ambiguous PnP frame cannot manufacture
    # lateral coverage.
    selected_bearings = [
        float(np.median(bearings_by_bag[bag_id]))
        for bag_id in sorted(bearings_by_bag)
    ]
    bearing_min = min(selected_bearings) if selected_bearings else None
    bearing_max = max(selected_bearings) if selected_bearings else None
    bearing_span = (
        bearing_max - bearing_min
        if bearing_min is not None and bearing_max is not None
        else None
    )

    reasons: list[str] = []
    if len(bag_ids) < sampling.minimum_bag_count:
        reasons.append("CAL-E-DYNAMIC-COVERAGE-BAG-COUNT")
    if any(
        count < sampling.minimum_bags_per_distance_bin
        for _name, count in distance_bag_counts
    ):
        reasons.append("CAL-E-DYNAMIC-COVERAGE-DISTANCE-BAGS")
    if bearing_span is None or bearing_span < minimum_bearing_span_deg:
        reasons.append("CAL-E-DYNAMIC-COVERAGE-BEARING-SPAN")
    if len(selected) < sampling.minimum_total_frames:
        reasons.append("CAL-E-DYNAMIC-COVERAGE-FRAME-COUNT")

    cell_counts = tuple(
        (bag, distance, bearing, len(rows))
        for (bag, distance, bearing), rows in sorted(cells.items())
    )
    keys = tuple(item.source.frame_key for item in selected)
    return DynamicSelection(
        frames=tuple(selected),
        keyframe_keys=tuple(item.source.frame_key for item in keyframes),
        selected_frame_keys_sha256=sha256_bytes(canonical_json_bytes(keys)),
        coverage=DynamicCoverageReport(
            passed=not reasons,
            reasons=tuple(reasons),
            bag_count=len(bag_ids),
            total_frame_count=len(selected),
            bearing_min_deg=bearing_min,
            bearing_max_deg=bearing_max,
            bearing_span_deg=bearing_span,
            distance_bin_bag_counts=distance_bag_counts,
            distance_bin_frame_counts=distance_frame_counts,
            cell_counts=cell_counts,
        ),
    )
