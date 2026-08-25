from dataclasses import replace

import cv2
import numpy as np
import pytest

import dynamic_rosbag_calibration.dynamic_experiment as dynamic_experiment
import dynamic_rosbag_calibration.same_day_dynamic_analysis as same_day_analysis
from dynamic_rosbag_calibration.calibration_types import CameraModel, FrameObservation, Transform
from dynamic_rosbag_calibration.dynamic_experiment import (
    evaluate_frozen_reused_tag_geometry,
    initialize_frozen_reused_tag_geometry,
)
from dynamic_rosbag_calibration.same_day_dynamic_analysis import (
    candidate_holdout_bags,
    retain_base_valid_frames_within_depth,
    select_same_day_combination_panel,
    summarize_candidate_holdouts,
)
from dynamic_rosbag_calibration.dynamic_combination_analysis import DynamicBagCombination
from dynamic_rosbag_calibration.dynamic_experiment_types import (
    DynamicFrameEvidence,
    DynamicCoverageReport,
    RawDynamicFrame,
    DynamicSelection,
    StructuralState,
)
from dynamic_rosbag_calibration.tag_geometry import TagGeometryConfig, tag_square_points


def _combination(*bag_ids: str, span: float = 5.0) -> DynamicBagCombination:
    coverage = DynamicCoverageReport(
        passed=True,
        reasons=(),
        bag_count=len(bag_ids),
        total_frame_count=60,
        bearing_min_deg=-2.5,
        bearing_max_deg=span - 2.5,
        bearing_span_deg=span,
        distance_bin_bag_counts=(("near", 2), ("middle", 2), ("far", 2)),
        distance_bin_frame_counts=(("near", 20), ("middle", 20), ("far", 20)),
        cell_counts=(),
    )
    return DynamicBagCombination(
        tuple(bag_ids), DynamicSelection((), (), "a" * 64, coverage)
    )


def test_small_same_day_panel_keeps_every_feasible_pair():
    feasible = (
        _combination("a", "b"),
        _combination("a", "c"),
        _combination("b", "c"),
    )

    assert select_same_day_combination_panel(feasible, maximum_count=12) == feasible


def test_large_same_day_panel_is_balanced_and_deterministic():
    bags = tuple("abcdef")
    feasible = tuple(
        _combination(left, right)
        for index, left in enumerate(bags)
        for right in bags[index + 1 :]
    )

    first = select_same_day_combination_panel(feasible, maximum_count=6)
    second = select_same_day_combination_panel(feasible, maximum_count=6)

    assert first == second
    assert set().union(*(set(item.bag_ids) for item in first)) == set(bags)


def test_candidate_holdout_bags_never_include_training_bags():
    plan = {
        "c01": ("a", "b"),
        "c02": ("a", "c"),
        "c03": ("b", "c"),
    }

    assert candidate_holdout_bags(plan, ("a", "b", "c"))["c01"] == ("c",)
    assert candidate_holdout_bags(plan, ("a", "b", "c"))["c02"] == ("b",)


def test_same_day_depth_scope_keeps_post_contact_frames_and_drops_over_8m():
    observation = FrameObservation(
        frame_key="bag/000000001",
        station_id="bag",
        board_instance_id="board_setup_01",
        map_ins=Transform(np.eye(4), "ins_map", "ins_link"),
        camera_tag0=Transform(np.eye(4), "left_camera", "tag0"),
        corners_px=np.zeros((8, 2)),
        odom_dt_ms=1.0,
        basic_valid=True,
        station_gate_pass=False,
    )

    def evidence(depth_m: float, sequence: int) -> DynamicFrameEvidence:
        raw = RawDynamicFrame(
            frame_key=f"bag/{sequence:09d}",
            bag_id="bag",
            sequence=sequence,
            source_stamp_s=float(sequence),
            observation=replace(observation, frame_key=f"bag/{sequence:09d}"),
            camera_tag0_initial=None,
            initial_tag0_depth_m=depth_m,
            bearing_x_deg=1.0,
            minimum_tag_edge_px=20.0,
            minimum_margin_px=30.0,
            interpolation=None,
            ins_quality_good=True,
            ins_quality_dt_ms=1.0,
            exclusion_reasons=(),
        )
        return DynamicFrameEvidence(
            raw,
            StructuralState.LOADED_STABLE,
            0.0,
            0.0,
            ("CAL-E-DYNAMIC-POST-CONTACT",),
        )

    retained = retain_base_valid_frames_within_depth(
        (evidence(3.0, 1), evidence(8.0, 2), evidence(8.01, 3)),
        maximum_depth_m=8.0,
    )

    assert [item.source.initial_tag0_depth_m for item in retained] == [3.0, 8.0]
    assert all(item.structural_state is StructuralState.UNLOADED_SAFE for item in retained)


def test_strict_depth_scope_excludes_the_8m_boundary():
    observation = FrameObservation(
        frame_key="bag/000000001",
        station_id="bag",
        board_instance_id="board_setup_01",
        map_ins=Transform(np.eye(4), "ins_map", "ins_link"),
        camera_tag0=Transform(np.eye(4), "left_camera", "tag0"),
        corners_px=np.zeros((8, 2)),
        odom_dt_ms=1.0,
        basic_valid=True,
        station_gate_pass=False,
    )

    def evidence(depth_m: float, sequence: int) -> DynamicFrameEvidence:
        raw = RawDynamicFrame(
            frame_key=f"bag/{sequence:09d}",
            bag_id="bag",
            sequence=sequence,
            source_stamp_s=float(sequence),
            observation=replace(observation, frame_key=f"bag/{sequence:09d}"),
            camera_tag0_initial=None,
            initial_tag0_depth_m=depth_m,
            bearing_x_deg=1.0,
            minimum_tag_edge_px=20.0,
            minimum_margin_px=30.0,
            interpolation=None,
            ins_quality_good=True,
            ins_quality_dt_ms=1.0,
            exclusion_reasons=(),
        )
        return DynamicFrameEvidence(
            raw,
            StructuralState.UNKNOWN,
            0.0,
            0.0,
            (),
        )

    retained = retain_base_valid_frames_within_depth(
        (evidence(7.999, 1), evidence(8.0, 2), evidence(8.001, 3)),
        maximum_depth_m=8.0,
        inclusive=False,
    )

    assert [item.source.initial_tag0_depth_m for item in retained] == [7.999]


def test_representative_bag_is_closest_to_candidate_median_errors():
    rows = (
        {"candidate_id": "c01", "bag_id": "a", "xy_p80_m": 0.01, "yaw_p80_deg": 0.1},
        {"candidate_id": "c01", "bag_id": "b", "xy_p80_m": 0.02, "yaw_p80_deg": 0.2},
        {"candidate_id": "c01", "bag_id": "c", "xy_p80_m": 0.08, "yaw_p80_deg": 0.8},
        {"candidate_id": "c02", "bag_id": "z", "xy_p80_m": 0.02, "yaw_p80_deg": 0.2},
    )

    selected = same_day_analysis.select_representative_validation_bag(rows, "c01")

    assert selected == "b"


def test_holdout_summary_is_equal_bag_and_rejects_training_leakage():
    plan = {"c01": ("a", "b"), "c02": ("a", "c")}
    rows = (
        {
            "candidate_id": "c01",
            "bag_id": "c",
            "xy_p80_m": 0.10,
            "xy_p95_m": 0.20,
            "yaw_p80_deg": 0.30,
            "yaw_p95_deg": 0.40,
            "signed_median_x_m": 0.01,
            "signed_median_y_m": -0.02,
            "signed_median_yaw_deg": 0.05,
        },
        {
            "candidate_id": "c02",
            "bag_id": "b",
            "xy_p80_m": 0.20,
            "xy_p95_m": 0.30,
            "yaw_p80_deg": 0.40,
            "yaw_p95_deg": 0.50,
            "signed_median_x_m": 0.02,
            "signed_median_y_m": -0.01,
            "signed_median_yaw_deg": 0.15,
        },
    )

    summary = summarize_candidate_holdouts(rows, plan)

    assert summary[0]["candidate_id"] == "c01"
    assert summary[0]["holdout_bag_count"] == 1
    assert summary[0]["equal_bag_xy_p80_m"] == pytest.approx(0.10)
    assert summary[1]["equal_bag_yaw_p80_deg"] == pytest.approx(0.40)

    leaked = rows + ({**rows[0], "bag_id": "a"},)
    with pytest.raises(ValueError, match="training bag"):
        summarize_candidate_holdouts(leaked, plan)


def test_frozen_reuse_geometry_allows_two_bags_without_reestimating_geometry():
    camera = CameraModel(
        np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
    )
    tag0_tag1 = np.eye(4)
    tag0_tag1[0, 3] = 0.6
    points = tag_square_points(0.5)
    frames = []
    for station in ("a", "b"):
        for index in range(5):
            camera_tag0 = np.eye(4)
            camera_tag0[:3, 3] = [0.1 * index, 0.0, 5.0]
            tag0_pixels, _ = cv2.projectPoints(
                points,
                np.zeros(3),
                camera_tag0[:3, 3],
                camera.matrix,
                camera.distortion,
            )
            camera_tag1 = camera_tag0 @ tag0_tag1
            tag1_pixels, _ = cv2.projectPoints(
                points,
                np.zeros(3),
                camera_tag1[:3, 3],
                camera.matrix,
                camera.distortion,
            )
            frames.append(
                FrameObservation(
                    frame_key=f"{station}/{index}",
                    station_id=station,
                    board_instance_id="board_setup_01",
                    map_ins=Transform(np.eye(4), "ins_map", "ins_link"),
                    camera_tag0=Transform(camera_tag0, "left_camera", "tag0"),
                    corners_px=np.vstack((tag0_pixels.reshape(4, 2), tag1_pixels.reshape(4, 2))),
                    odom_dt_ms=1.0,
                    basic_valid=True,
                    station_gate_pass=True,
                )
            )
    initial = Transform(tag0_tag1, "tag0", "tag1")

    initialization = initialize_frozen_reused_tag_geometry(
        frames,
        camera,
        0.5,
        "reuse",
        initial,
        "/tmp/tag.yaml",
        "a" * 64,
        bootstrap_maximum_views_per_station=5,
    )
    result = evaluate_frozen_reused_tag_geometry(
        initialization,
        camera,
        0.5,
        TagGeometryConfig(),
    )

    assert len({frame.station_id for frame in initialization.seeded_frames}) == 2
    assert result.quality_gate_pass
    assert result.valid_station_count == 2
    np.testing.assert_allclose(result.frozen.transform.matrix, tag0_tag1)
    assert result.solver_message == "approved geometry frozen; no G2 estimation"

    strict = evaluate_frozen_reused_tag_geometry(
        initialization,
        camera,
        0.5,
        replace(
            TagGeometryConfig(),
            overall_reprojection_rms_max_px=(
                result.overall_reprojection_rms_px * 0.5
            ),
        ),
    )
    assert not strict.quality_gate_pass
    assert "CAL-E-TAG-GEOMETRY-OVERALL-RMS" in strict.reasons

    strict_view = evaluate_frozen_reused_tag_geometry(
        initialization,
        camera,
        0.5,
        replace(
            TagGeometryConfig(),
            maximum_view_rms_max_px=result.maximum_view_rms_px * 0.5,
        ),
    )
    assert not strict_view.quality_gate_pass
    assert "CAL-E-TAG-GEOMETRY-VIEW-RMS" in strict_view.reasons


def test_frozen_geometry_audit_marks_g2_as_not_run_and_six_station_gate_na():
    assert hasattr(dynamic_experiment, "conditional_ablation_tag_geometry_audit")

    audit = dynamic_experiment.conditional_ablation_tag_geometry_audit(
        {
            "quality_gate_pass": True,
            "reasons": [],
            "overall_reprojection_rms_px": 0.25,
            "maximum_view_rms_px": 0.40,
            "balanced_view_count": 20,
            "valid_station_count": 2,
        }
    )

    assert audit["estimation_status"] == "G2_NOT_RUN_CONDITIONAL_ABLATION"
    assert audit["six_station_gate_applicable"] is False
    assert audit["quality_gate_pass"] is None
    assert audit["fixed_geometry_reprojection_gate_pass"] is True
