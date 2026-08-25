from types import SimpleNamespace

import numpy as np

from dynamic_rosbag_calibration.calibration_types import CameraModel, FrameObservation, Transform
from dynamic_rosbag_calibration.dynamic_experiment_types import (
    DynamicFrameEvidence,
    RawDynamicFrame,
    StructuralState,
)
from dynamic_rosbag_calibration.safe_dynamic_validation import evaluate_safe_dynamic_candidates


def _evidence(sequence: int, state: StructuralState) -> DynamicFrameEvidence:
    observation = FrameObservation(
        frame_key=f"104757/{sequence:09d}",
        station_id="104757",
        board_instance_id="board_setup_01",
        map_ins=Transform(np.eye(4), "ins_map", "ins_link"),
        camera_tag0=Transform(np.eye(4), "left_camera", "tag0"),
        corners_px=np.zeros((8, 2)),
        odom_dt_ms=2.0,
        basic_valid=True,
        station_gate_pass=False,
    )
    raw = RawDynamicFrame(
        frame_key=observation.frame_key,
        bag_id="104757",
        sequence=sequence,
        source_stamp_s=float(sequence),
        observation=observation,
        camera_tag0_initial=observation.camera_tag0,
        initial_tag0_depth_m=5.0 if state is StructuralState.UNLOADED_SAFE else 3.0,
        bearing_x_deg=-7.0,
        minimum_tag_edge_px=20.0,
        minimum_margin_px=30.0,
        interpolation=None,
        ins_quality_good=True,
        ins_quality_dt_ms=2.0,
        exclusion_reasons=(),
    )
    return DynamicFrameEvidence(raw, state, 0.0, 0.0, ())


def test_validation_uses_common_frames_and_candidate_independent_primary_scope(
    monkeypatch,
):
    def fake_solve(frame, _camera, _tag_size, _frozen):
        matrix = np.eye(4)
        matrix[2, 3] = 99.0  # Must not alter the precomputed primary cohort.
        return SimpleNamespace(
            passed=True,
            camera_tag0=Transform(matrix, "left_camera", "tag0"),
            dual_reprojection_rms_px=0.1,
        )

    monkeypatch.setattr(
        "dynamic_rosbag_calibration.safe_dynamic_validation.solve_frame_with_frozen_geometry",
        fake_solve,
    )
    candidates = {
        candidate_id: {
            "T_tag0_tag1_calibrated": np.eye(4).tolist(),
            "T_ins_map_tag0_calibrated": np.eye(4).tolist(),
            "T_ins_camera_calibrated": np.eye(4).tolist(),
        }
        for candidate_id in ("full_static_rxy", "safe_dynamic_rxy")
    }
    result = evaluate_safe_dynamic_candidates(
        candidates,
        {
            "104757": (
                _evidence(1, StructuralState.UNLOADED_SAFE),
                _evidence(2, StructuralState.UNKNOWN),
            )
        },
        dataset="onsite_20260819_holdout",
        camera_model=CameraModel(np.eye(3), np.zeros(5)),
        tag_size_m=0.5,
    )

    assert result.common_frame_counts[0]["common_frame_count"] == 2
    assert result.common_frame_counts[0]["common_primary_frame_count"] == 1
    assert len(result.rows) == 4
    assert len(result.pooled_metrics) == 2
    assert {row["distance_bin"] for row in result.depth_metrics} == {"middle"}
    for candidate_id in candidates:
        candidate_rows = [
            row for row in result.rows if row["candidate_id"] == candidate_id
        ]
        assert [row["primary_scope"] for row in candidate_rows] == [True, False]
        metric = next(
            row for row in result.per_bag_metrics if row["candidate_id"] == candidate_id
        )
        assert metric["frame_count"] == 1


def test_validation_reports_board_displacement_from_observed_preimpact_pose(
    monkeypatch,
):
    observed_camera_tag0 = np.eye(4)
    observed_camera_tag0[:3, 3] = [0.30, -0.40, 0.0]

    def fake_solve(_frame, _camera, _tag_size, _frozen):
        return SimpleNamespace(
            passed=True,
            camera_tag0=Transform(observed_camera_tag0, "left_camera", "tag0"),
            dual_reprojection_rms_px=0.1,
        )

    monkeypatch.setattr(
        "dynamic_rosbag_calibration.safe_dynamic_validation.solve_frame_with_frozen_geometry",
        fake_solve,
    )
    result = evaluate_safe_dynamic_candidates(
        {
            "postimpact": {
                "T_tag0_tag1_calibrated": np.eye(4).tolist(),
                "T_ins_map_tag0_calibrated": np.eye(4).tolist(),
                "T_ins_camera_calibrated": np.eye(4).tolist(),
            }
        },
        {"104757": (_evidence(1, StructuralState.UNLOADED_SAFE),)},
        dataset="preimpact_displacement",
        camera_model=CameraModel(np.eye(3), np.zeros(5)),
        tag_size_m=0.5,
    )

    row = result.rows[0]
    assert row["observed_tag0_x_m"] == 0.30
    assert row["observed_tag0_y_m"] == -0.40
    assert row["board_displacement_x_m"] == -0.30
    assert row["board_displacement_y_m"] == 0.40
    assert row["board_displacement_m"] == 0.50
    assert row["board_displacement_yaw_deg"] == 0.0


def test_validation_can_label_post_contact_structural_stress_scope(monkeypatch):
    def fake_solve(frame, _camera, _tag_size, _frozen):
        return SimpleNamespace(
            passed=True,
            camera_tag0=frame.camera_tag0,
            dual_reprojection_rms_px=0.1,
        )

    monkeypatch.setattr(
        "dynamic_rosbag_calibration.safe_dynamic_validation.solve_frame_with_frozen_geometry",
        fake_solve,
    )
    candidate = {
        "candidate": {
            "T_tag0_tag1_calibrated": np.eye(4).tolist(),
            "T_ins_map_tag0_calibrated": np.eye(4).tolist(),
            "T_ins_camera_calibrated": np.eye(4).tolist(),
        }
    }

    result = evaluate_safe_dynamic_candidates(
        candidate,
        {
            "104757": (
                _evidence(1, StructuralState.CONTACT_TRANSIENT),
                _evidence(2, StructuralState.LOADED_STABLE),
                _evidence(3, StructuralState.RELEASE_TRANSIENT),
            )
        },
        dataset="onsite_stress",
        camera_model=CameraModel(np.eye(3), np.zeros(5)),
        tag_size_m=0.5,
        primary_states=(
            StructuralState.CONTACT_TRANSIENT,
            StructuralState.LOADED_STABLE,
            StructuralState.RELEASE_TRANSIENT,
        ),
        validation_scope="POST_CONTACT_STRUCTURAL_STRESS",
    )

    assert result.per_bag_metrics[0]["validation_scope"] == "POST_CONTACT_STRUCTURAL_STRESS"
    assert result.per_bag_metrics[0]["frame_count"] == 3
    assert all(row["primary_scope"] for row in result.rows)
    assert all(
        row["validation_scope"] == "POST_CONTACT_STRUCTURAL_STRESS"
        for row in result.rows
    )
