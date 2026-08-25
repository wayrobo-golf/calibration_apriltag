"""Planning and leakage-safe summaries for same-day dynamic calibration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from .dynamic_combination_analysis import (
    DynamicBagCombination,
    select_balanced_dynamic_combinations,
)
from .dynamic_experiment_types import DynamicFrameEvidence, StructuralState


def retain_base_valid_frames_within_depth(
    frames: Sequence[DynamicFrameEvidence],
    *,
    maximum_depth_m: float,
    inclusive: bool = True,
) -> tuple[DynamicFrameEvidence, ...]:
    """Keep all base-valid frames in the depth scope, without structural gating.

    The ``UNLOADED_SAFE`` state below is only an adapter for the legacy balanced
    sampler, whose public contract selects that state.  It does not classify the
    retained frames as mechanically safe and callers should keep the original
    evidence for reporting and validation.
    """
    maximum = float(maximum_depth_m)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_depth_m must be finite and positive")
    retained: list[DynamicFrameEvidence] = []
    for item in frames:
        source = item.source
        depth = source.initial_tag0_depth_m
        within_depth = (
            depth is not None
            and 0.0 < float(depth)
            and (
                float(depth) <= maximum
                if inclusive
                else float(depth) < maximum
            )
        )
        if (
            source.observation is None
            or source.exclusion_reasons
            or not within_depth
            or source.bearing_x_deg is None
        ):
            continue
        retained.append(
            replace(
                item,
                structural_state=StructuralState.UNLOADED_SAFE,
                exclusion_reasons=(),
            )
        )
    return tuple(retained)


def select_representative_validation_bag(
    per_bag_metrics: Sequence[Mapping[str, object]],
    candidate_id: str,
) -> str:
    """Select the holdout closest to the candidate's median XY/yaw errors."""
    rows = [
        row
        for row in per_bag_metrics
        if str(row["candidate_id"]) == str(candidate_id)
    ]
    if not rows:
        raise ValueError(f"candidate {candidate_id} has no per-bag validation metric")
    xy_values = np.asarray([float(row["xy_p80_m"]) for row in rows])
    yaw_values = np.asarray([float(row["yaw_p80_deg"]) for row in rows])
    if not np.all(np.isfinite(xy_values)) or not np.all(np.isfinite(yaw_values)):
        raise ValueError("representative-bag metrics must be finite")
    xy_median = float(np.median(xy_values))
    yaw_median = float(np.median(yaw_values))
    xy_scale = max(abs(xy_median), 1.0e-12)
    yaw_scale = max(abs(yaw_median), 1.0e-12)
    selected = min(
        rows,
        key=lambda row: (
            abs(float(row["xy_p80_m"]) - xy_median) / xy_scale
            + abs(float(row["yaw_p80_deg"]) - yaw_median) / yaw_scale,
            str(row["bag_id"]),
        ),
    )
    return str(selected["bag_id"])


def select_same_day_combination_panel(
    feasible: Sequence[DynamicBagCombination],
    *,
    maximum_count: int,
) -> tuple[DynamicBagCombination, ...]:
    """Keep all small panels, otherwise select a deterministic balanced panel."""
    values = tuple(feasible)
    if maximum_count <= 0:
        raise ValueError("maximum_count must be positive")
    if not values:
        return ()
    if len(values) <= maximum_count:
        return values
    bag_ids = tuple(sorted(set().union(*(set(item.bag_ids) for item in values))))
    return select_balanced_dynamic_combinations(
        values,
        bag_ids,
        count=maximum_count,
    )


def candidate_holdout_bags(
    training_bags_by_candidate: Mapping[str, Sequence[str]],
    validation_bag_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Return same-day validation bags after excluding each candidate's training bags."""
    available = tuple(sorted(str(value) for value in validation_bag_ids))
    return {
        str(candidate_id): tuple(
            bag_id
            for bag_id in available
            if bag_id not in frozenset(str(value) for value in training_bags)
        )
        for candidate_id, training_bags in sorted(training_bags_by_candidate.items())
    }


def summarize_candidate_holdouts(
    per_bag_metrics: Sequence[Mapping[str, object]],
    training_bags_by_candidate: Mapping[str, Sequence[str]],
) -> tuple[dict[str, object], ...]:
    """Summarize candidate-specific holdouts with equal weight per bag."""
    training = {
        str(candidate_id): frozenset(str(value) for value in bag_ids)
        for candidate_id, bag_ids in training_bags_by_candidate.items()
    }
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in per_bag_metrics:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in training:
            raise ValueError(f"unknown candidate in validation metrics: {candidate_id}")
        bag_id = str(row["bag_id"])
        if bag_id in training[candidate_id]:
            raise ValueError(
                f"candidate {candidate_id} validation contains training bag {bag_id}"
            )
        grouped[candidate_id].append(row)

    output: list[dict[str, object]] = []
    for candidate_id in sorted(training):
        rows = grouped.get(candidate_id, [])
        if not rows:
            raise ValueError(f"candidate {candidate_id} has no same-day holdout metric")
        output.append(
            {
                "candidate_id": candidate_id,
                "holdout_bag_count": len(rows),
                "holdout_bag_ids": ";".join(
                    sorted(str(row["bag_id"]) for row in rows)
                ),
                "equal_bag_xy_p80_m": float(
                    np.mean([float(row["xy_p80_m"]) for row in rows])
                ),
                "equal_bag_xy_p95_m": float(
                    np.mean([float(row["xy_p95_m"]) for row in rows])
                ),
                "equal_bag_yaw_p80_deg": float(
                    np.mean([float(row["yaw_p80_deg"]) for row in rows])
                ),
                "equal_bag_yaw_p95_deg": float(
                    np.mean([float(row["yaw_p95_deg"]) for row in rows])
                ),
                "median_signed_x_m": float(
                    np.median([float(row["signed_median_x_m"]) for row in rows])
                ),
                "median_signed_y_m": float(
                    np.median([float(row["signed_median_y_m"]) for row in rows])
                ),
                "median_signed_yaw_deg": float(
                    np.median(
                        [float(row["signed_median_yaw_deg"]) for row in rows]
                    )
                ),
            }
        )
    return tuple(output)
