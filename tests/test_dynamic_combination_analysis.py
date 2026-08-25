import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import dynamic_rosbag_calibration.dynamic_combination_analysis as combination_analysis

from dynamic_rosbag_calibration.dynamic_combination_analysis import (
    DynamicBagCombination,
    candidate_differences,
    select_balanced_dynamic_combinations,
    transform_difference,
)
from dynamic_rosbag_calibration.dynamic_experiment_types import (
    DynamicCoverageReport,
    DynamicSelection,
)


def _combination(*bag_ids: str, span: float = 5.0) -> DynamicBagCombination:
    coverage = DynamicCoverageReport(
        passed=True,
        reasons=(),
        bag_count=len(bag_ids),
        total_frame_count=120,
        bearing_min_deg=-2.5,
        bearing_max_deg=span - 2.5,
        bearing_span_deg=span,
        distance_bin_bag_counts=(("near", 3), ("middle", 4), ("far", 3)),
        distance_bin_frame_counts=(("near", 40), ("middle", 40), ("far", 40)),
        cell_counts=(),
    )
    return DynamicBagCombination(
        tuple(bag_ids), DynamicSelection((), (), "a" * 64, coverage)
    )


def test_balanced_combination_panel_is_deterministic_and_covers_all_bags():
    feasible = (
        _combination("a", "b", "c"),
        _combination("a", "d", "e"),
        _combination("b", "d", "f"),
        _combination("c", "e", "f"),
        _combination("a", "e", "f"),
    )

    first = select_balanced_dynamic_combinations(
        feasible, tuple("abcdef"), count=3
    )
    second = select_balanced_dynamic_combinations(
        feasible, tuple("abcdef"), count=3
    )

    assert first == second
    assert set().union(*(set(item.bag_ids) for item in first)) == set("abcdef")


def test_transform_and_candidate_differences_use_relative_se3():
    reference = np.eye(4)
    candidate = np.eye(4)
    candidate[:3, :3] = Rotation.from_euler("z", 2.0, degrees=True).as_matrix()
    candidate[:3, 3] = [0.03, 0.04, 0.0]

    difference = transform_difference(reference, candidate)
    rows = candidate_differences(
        {
            "baseline": {"T": reference.tolist()},
            "candidate": {"T": candidate.tolist()},
        },
        reference_id="baseline",
        transform_fields=("T",),
    )

    assert difference.translation_m == pytest.approx(0.05)
    assert difference.rotation_deg == pytest.approx(2.0)
    row = next(value for value in rows if value["candidate_id"] == "candidate")
    assert row["delta_x_m"] == pytest.approx(0.03)
    assert row["delta_y_m"] == pytest.approx(0.04)


def test_pairwise_rows_and_summaries_keep_tag_and_world_transforms_separate():
    identity = np.eye(4)
    tag_shift = np.eye(4)
    tag_shift[0, 3] = 0.01
    world_shift = np.eye(4)
    world_shift[:3, :3] = Rotation.from_euler("y", 2.0, degrees=True).as_matrix()
    candidates = {
        "c01": {"T_tag0_tag1_calibrated": identity, "T_ins_map_tag0_calibrated": identity},
        "c02": {"T_tag0_tag1_calibrated": tag_shift, "T_ins_map_tag0_calibrated": world_shift},
    }

    rows = combination_analysis.pairwise_candidate_differences(
        candidates,
        transform_fields=("T_tag0_tag1_calibrated", "T_ins_map_tag0_calibrated"),
    )
    summaries = combination_analysis.summarize_transform_stability(rows)

    assert rows[0]["delta_x_m"] == pytest.approx(0.01)
    assert rows[1]["delta_pitch_deg"] == pytest.approx(2.0)
    assert [row["transform"] for row in summaries] == [
        "T_ins_map_tag0_calibrated",
        "T_tag0_tag1_calibrated",
    ]
    by_transform = {row["transform"]: row for row in summaries}
    assert by_transform["T_tag0_tag1_calibrated"]["translation_p95_m"] == pytest.approx(0.01)
    assert by_transform["T_ins_map_tag0_calibrated"]["rotation_p95_deg"] == pytest.approx(2.0)


def test_candidate_selection_combines_transform_stability_and_holdout_ranks():
    pairwise_rows = (
        {"left_candidate_id": "a", "right_candidate_id": "b", "transform": "tag", "translation_difference_m": 0.01, "rotation_difference_deg": 0.1},
        {"left_candidate_id": "a", "right_candidate_id": "c", "transform": "tag", "translation_difference_m": 0.02, "rotation_difference_deg": 0.2},
        {"left_candidate_id": "b", "right_candidate_id": "c", "transform": "tag", "translation_difference_m": 0.03, "rotation_difference_deg": 0.3},
        {"left_candidate_id": "a", "right_candidate_id": "b", "transform": "world", "translation_difference_m": 0.01, "rotation_difference_deg": 0.1},
        {"left_candidate_id": "a", "right_candidate_id": "c", "transform": "world", "translation_difference_m": 0.02, "rotation_difference_deg": 0.2},
        {"left_candidate_id": "b", "right_candidate_id": "c", "transform": "world", "translation_difference_m": 0.03, "rotation_difference_deg": 0.3},
    )
    holdouts = (
        {"candidate_id": "a", "equal_bag_xy_p80_m": 0.03, "equal_bag_yaw_p80_deg": 0.3},
        {"candidate_id": "b", "equal_bag_xy_p80_m": 0.01, "equal_bag_yaw_p80_deg": 0.1},
        {"candidate_id": "c", "equal_bag_xy_p80_m": 0.08, "equal_bag_yaw_p80_deg": 0.8},
    )

    ranking = combination_analysis.rank_candidates_by_stability_and_holdout(
        pairwise_rows,
        holdouts,
        required_transforms=("tag", "world"),
    )

    assert ranking[0]["candidate_id"] == "b"
    assert ranking[0]["selected"] is True


def test_quality_gate_passed_candidate_is_preferred_for_display():
    ranking = (
        {"candidate_id": "failed", "combined_rank_score": 0.1, "holdout_rank_score": 0.1, "selected": True},
        {"candidate_id": "passed", "combined_rank_score": 0.4, "holdout_rank_score": 0.3, "selected": False},
    )

    selected = combination_analysis.apply_quality_gate_preference(
        ranking,
        {"failed": False, "passed": True},
    )

    assert selected[0]["candidate_id"] == "passed"
    assert selected[0]["selected"] is True
    assert selected[1]["selected"] is False
