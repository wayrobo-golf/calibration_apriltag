"""Deterministic bag-subset planning and transform stability metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .dynamic_experiment_types import (
    DynamicFrameEvidence,
    DynamicSamplingConfig,
    DynamicSelection,
)
from .dynamic_sampling import select_balanced_dynamic_frames


@dataclass(frozen=True)
class DynamicBagCombination:
    bag_ids: tuple[str, ...]
    selection: DynamicSelection


@dataclass(frozen=True)
class TransformDifference:
    translation_m: float
    rotation_deg: float
    delta_x_m: float
    delta_y_m: float
    delta_z_m: float
    delta_roll_deg: float
    delta_pitch_deg: float
    delta_yaw_deg: float


def enumerate_feasible_dynamic_combinations(
    frames: Sequence[DynamicFrameEvidence],
    sampling: DynamicSamplingConfig,
    *,
    bag_count: int,
    distance_edges_m: tuple[float, float],
    minimum_bearing_span_deg: float,
) -> tuple[DynamicBagCombination, ...]:
    """Enumerate bag subsets that pass the unchanged dynamic coverage gate."""
    values = tuple(frames)
    bag_ids = tuple(sorted({item.source.bag_id for item in values}))
    if bag_count <= 0 or bag_count > len(bag_ids):
        raise ValueError("bag_count must fit the available dynamic bag count")
    result: list[DynamicBagCombination] = []
    for selected_ids in combinations(bag_ids, bag_count):
        selected_set = frozenset(selected_ids)
        selection = select_balanced_dynamic_frames(
            tuple(
                item for item in values if item.source.bag_id in selected_set
            ),
            sampling,
            distance_edges_m=distance_edges_m,
            minimum_bearing_span_deg=minimum_bearing_span_deg,
        )
        if selection.coverage.passed:
            result.append(DynamicBagCombination(selected_ids, selection))
    return tuple(result)


def select_balanced_dynamic_combinations(
    feasible: Sequence[DynamicBagCombination],
    all_bag_ids: Sequence[str],
    *,
    count: int,
) -> tuple[DynamicBagCombination, ...]:
    """Choose a reproducible panel with balanced bag and bag-pair incidence."""
    candidates = list(sorted(feasible, key=lambda item: item.bag_ids))
    bag_ids = tuple(sorted(str(value) for value in all_bag_ids))
    if len(set(bag_ids)) != len(bag_ids):
        raise ValueError("all_bag_ids must be unique")
    if count <= 0 or count > len(candidates):
        raise ValueError("count must fit the feasible combination count")
    covered = set().union(*(set(item.bag_ids) for item in candidates))
    if not set(bag_ids).issubset(covered):
        raise ValueError("feasible combinations cannot cover every bag")

    selected: list[DynamicBagCombination] = []
    bag_load: Counter[str] = Counter()
    pair_load: Counter[tuple[str, str]] = Counter()
    while len(selected) < count:

        def score(item: DynamicBagCombination) -> tuple[object, ...]:
            after = Counter(bag_load)
            after.update(item.bag_ids)
            uncovered = sum(after[bag_id] == 0 for bag_id in bag_ids)
            imbalance = max(after[bag_id] for bag_id in bag_ids) - min(
                after[bag_id] for bag_id in bag_ids
            )
            squared_load = sum(after[bag_id] ** 2 for bag_id in bag_ids)
            repeated_pairs = sum(
                pair_load[pair] for pair in combinations(item.bag_ids, 2)
            )
            coverage = item.selection.coverage
            return (
                uncovered,
                imbalance,
                squared_load,
                repeated_pairs,
                -float(coverage.bearing_span_deg or 0.0),
                -coverage.total_frame_count,
                item.bag_ids,
            )

        chosen = min(candidates, key=score)
        selected.append(chosen)
        candidates.remove(chosen)
        bag_load.update(chosen.bag_ids)
        pair_load.update(combinations(chosen.bag_ids, 2))
    if set().union(*(set(item.bag_ids) for item in selected)) != set(bag_ids):
        raise ValueError("selected panel does not cover every dynamic bag")
    return tuple(selected)


def transform_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> TransformDifference:
    reference_value = np.asarray(reference, dtype=np.float64).reshape(4, 4)
    candidate_value = np.asarray(candidate, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(reference_value)) or not np.all(
        np.isfinite(candidate_value)
    ):
        raise ValueError("transform matrices must be finite")
    relative = np.linalg.inv(reference_value) @ candidate_value
    translation = relative[:3, 3]
    roll_deg, pitch_deg, yaw_deg = Rotation.from_matrix(
        relative[:3, :3]
    ).as_euler("xyz", degrees=True)
    return TransformDifference(
        translation_m=float(np.linalg.norm(translation)),
        rotation_deg=float(
            np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude())
        ),
        delta_x_m=float(candidate_value[0, 3] - reference_value[0, 3]),
        delta_y_m=float(candidate_value[1, 3] - reference_value[1, 3]),
        delta_z_m=float(candidate_value[2, 3] - reference_value[2, 3]),
        delta_roll_deg=float(roll_deg),
        delta_pitch_deg=float(pitch_deg),
        delta_yaw_deg=float(yaw_deg),
    )


def candidate_differences(
    candidates: Mapping[str, Mapping[str, object]],
    *,
    reference_id: str,
    transform_fields: Sequence[str],
) -> tuple[dict[str, object], ...]:
    if reference_id not in candidates:
        raise ValueError("reference candidate is missing")
    reference = candidates[reference_id]
    rows: list[dict[str, object]] = []
    for candidate_id in sorted(candidates):
        for field in transform_fields:
            if field not in reference or field not in candidates[candidate_id]:
                raise ValueError(f"candidate transform is missing: {field}")
            difference = transform_difference(
                np.asarray(reference[field], dtype=np.float64),
                np.asarray(candidates[candidate_id][field], dtype=np.float64),
            )
            rows.append(
                {
                    "reference_id": reference_id,
                    "candidate_id": candidate_id,
                    "transform": field,
                    "translation_difference_m": difference.translation_m,
                    "rotation_difference_deg": difference.rotation_deg,
                    "delta_x_m": difference.delta_x_m,
                    "delta_y_m": difference.delta_y_m,
                    "delta_z_m": difference.delta_z_m,
                    "delta_roll_deg": difference.delta_roll_deg,
                    "delta_pitch_deg": difference.delta_pitch_deg,
                    "delta_yaw_deg": difference.delta_yaw_deg,
                }
            )
    return tuple(rows)


def pairwise_candidate_differences(
    candidates: Mapping[str, Mapping[str, object]],
    *,
    transform_fields: Sequence[str],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for left_id, right_id in combinations(sorted(candidates), 2):
        for field in transform_fields:
            difference = transform_difference(
                np.asarray(candidates[left_id][field], dtype=np.float64),
                np.asarray(candidates[right_id][field], dtype=np.float64),
            )
            rows.append(
                {
                    "left_candidate_id": left_id,
                    "right_candidate_id": right_id,
                    "transform": field,
                    "translation_difference_m": difference.translation_m,
                    "rotation_difference_deg": difference.rotation_deg,
                    "delta_x_m": difference.delta_x_m,
                    "delta_y_m": difference.delta_y_m,
                    "delta_z_m": difference.delta_z_m,
                    "delta_roll_deg": difference.delta_roll_deg,
                    "delta_pitch_deg": difference.delta_pitch_deg,
                    "delta_yaw_deg": difference.delta_yaw_deg,
                }
            )
    return tuple(rows)


def summarize_transform_stability(
    pairwise_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Summarize each transform independently over candidate pairs."""
    metric_fields = (
        "translation_difference_m",
        "rotation_difference_deg",
        "delta_x_m",
        "delta_y_m",
        "delta_z_m",
        "delta_roll_deg",
        "delta_pitch_deg",
        "delta_yaw_deg",
    )
    transforms = sorted({str(row["transform"]) for row in pairwise_rows})
    summaries: list[dict[str, object]] = []
    for transform in transforms:
        rows = [row for row in pairwise_rows if str(row["transform"]) == transform]
        summary: dict[str, object] = {
            "transform": transform,
            "pair_count": len(rows),
        }
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite transform stability metric: {field}")
            prefix = field.removesuffix("_difference_m").removesuffix(
                "_difference_deg"
            ).removesuffix("_m").removesuffix("_deg")
            unit = "m" if field.endswith("_m") else "deg"
            summary[f"{prefix}_mean_{unit}"] = float(np.mean(values))
            summary[f"{prefix}_std_{unit}"] = float(np.std(values))
            summary[f"{prefix}_p50_{unit}"] = float(np.percentile(values, 50))
            summary[f"{prefix}_p80_{unit}"] = float(np.percentile(values, 80))
            summary[f"{prefix}_p95_{unit}"] = float(np.percentile(values, 95))
            summary[f"{prefix}_max_{unit}"] = float(np.max(values))
        summaries.append(summary)
    return tuple(summaries)


def _normalized_ranks(values: Mapping[str, float]) -> dict[str, float]:
    unique = sorted(set(float(value) for value in values.values()))
    denominator = max(len(unique) - 1, 1)
    ranks = {value: index / denominator for index, value in enumerate(unique)}
    return {key: ranks[float(value)] for key, value in values.items()}


def rank_candidates_by_stability_and_holdout(
    pairwise_rows: Sequence[Mapping[str, object]],
    holdout_rows: Sequence[Mapping[str, object]],
    *,
    required_transforms: Sequence[str],
) -> tuple[dict[str, object], ...]:
    """Rank candidates equally by transform medoid proximity and holdout error."""
    holdouts = {str(row["candidate_id"]): row for row in holdout_rows}
    if not holdouts:
        raise ValueError("candidate ranking requires holdout metrics")
    candidate_ids = tuple(sorted(holdouts))
    stability_metrics: dict[str, dict[str, float]] = {}
    for transform in required_transforms:
        transform_rows = [
            row for row in pairwise_rows if str(row["transform"]) == str(transform)
        ]
        for metric in ("translation_difference_m", "rotation_difference_deg"):
            per_candidate: dict[str, list[float]] = {
                candidate_id: [] for candidate_id in candidate_ids
            }
            for row in transform_rows:
                value = float(row[metric])
                for field in ("left_candidate_id", "right_candidate_id"):
                    candidate_id = str(row[field])
                    if candidate_id in per_candidate:
                        per_candidate[candidate_id].append(value)
            if any(not values for values in per_candidate.values()):
                raise ValueError(
                    f"transform {transform} lacks pairwise coverage for every candidate"
                )
            stability_metrics[f"{transform}:{metric}"] = {
                candidate_id: float(np.mean(values))
                for candidate_id, values in per_candidate.items()
            }
    stability_ranks = {
        name: _normalized_ranks(values)
        for name, values in stability_metrics.items()
    }
    holdout_metrics = {
        "equal_bag_xy_p80_m": {
            candidate_id: float(holdouts[candidate_id]["equal_bag_xy_p80_m"])
            for candidate_id in candidate_ids
        },
        "equal_bag_yaw_p80_deg": {
            candidate_id: float(holdouts[candidate_id]["equal_bag_yaw_p80_deg"])
            for candidate_id in candidate_ids
        },
    }
    holdout_ranks = {
        name: _normalized_ranks(values) for name, values in holdout_metrics.items()
    }
    rows: list[dict[str, object]] = []
    for candidate_id in candidate_ids:
        stability_score = float(
            np.mean([ranks[candidate_id] for ranks in stability_ranks.values()])
        )
        holdout_score = float(
            np.mean([ranks[candidate_id] for ranks in holdout_ranks.values()])
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "stability_rank_score": stability_score,
                "holdout_rank_score": holdout_score,
                "combined_rank_score": 0.5 * (stability_score + holdout_score),
                "selected": False,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["combined_rank_score"]),
            float(row["holdout_rank_score"]),
            str(row["candidate_id"]),
        )
    )
    rows[0]["selected"] = True
    return tuple(rows)


def apply_quality_gate_preference(
    ranking: Sequence[Mapping[str, object]],
    quality_gate_pass_by_candidate: Mapping[str, bool],
) -> tuple[dict[str, object], ...]:
    """Prefer a passing candidate without excluding failed diagnostic rows."""
    rows = [dict(row) for row in ranking]
    candidate_ids = {str(row["candidate_id"]) for row in rows}
    if candidate_ids != set(quality_gate_pass_by_candidate):
        raise ValueError("quality-gate mapping must cover exactly the ranking")
    passing = [
        row
        for row in rows
        if bool(quality_gate_pass_by_candidate[str(row["candidate_id"])])
    ]
    pool = passing or rows
    selected = min(
        pool,
        key=lambda row: (
            float(row["combined_rank_score"]),
            float(row["holdout_rank_score"]),
            str(row["candidate_id"]),
        ),
    )
    selected_id = str(selected["candidate_id"])
    for row in rows:
        candidate_id = str(row["candidate_id"])
        row["quality_gate_pass"] = bool(
            quality_gate_pass_by_candidate[candidate_id]
        )
        row["selected"] = candidate_id == selected_id
    rows.sort(
        key=lambda row: (
            not bool(row["selected"]),
            float(row["combined_rank_score"]),
            str(row["candidate_id"]),
        )
    )
    return tuple(rows)
