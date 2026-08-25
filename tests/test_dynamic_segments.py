import numpy as np
import pytest

from dynamic_rosbag_calibration.calibration_types import FrameObservation, Transform
from dynamic_rosbag_calibration.dynamic_experiment_types import (
    DynamicBagEvidence,
    RawDynamicFrame,
    StructuralGateConfig,
    StructuralState,
    TimedImu,
)
from dynamic_rosbag_calibration.dynamic_segments import (
    classify_dynamic_frames,
    compute_highpass_motion,
)


CONFIG = StructuralGateConfig(
    minimum_safe_depth_m=4.25,
    maximum_depth_m=8.0,
    contact_candidate_depth_m=3.5,
    acceleration_highpass_limit_mps2=2.0,
    angular_rate_highpass_limit_degps=6.0,
    highpass_window_s=1.0,
    contact_confirmation_s=0.1,
)


def _imu(stamp, acceleration=(0.0, 0.0, 9.8), angular=(0.0, 0.0, 0.0)):
    return TimedImu(stamp, np.asarray(acceleration), np.asarray(angular))


def _frame(sequence, stamp, depth, *, valid=True):
    pose = np.eye(4)
    pose[0, 3] = 10.0 - depth
    observation = (
        FrameObservation(
            frame_key=f"bag/{sequence:09d}",
            station_id="bag",
            board_instance_id="board_setup_01",
            map_ins=Transform(pose, "ins_map", "ins_link"),
            camera_tag0=Transform(
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, depth],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                "left_camera",
                "tag0",
            ),
            corners_px=np.zeros((8, 2)),
            odom_dt_ms=5.0,
            basic_valid=valid,
            station_gate_pass=False,
        )
        if valid
        else None
    )
    return RawDynamicFrame(
        frame_key=f"bag/{sequence:09d}",
        bag_id="bag",
        sequence=sequence,
        source_stamp_s=stamp,
        observation=observation,
        camera_tag0_initial=(observation.camera_tag0 if observation else None),
        initial_tag0_depth_m=depth,
        bearing_x_deg=-7.0,
        minimum_tag_edge_px=20.0,
        minimum_margin_px=30.0,
        interpolation=None,
        ins_quality_good=valid,
        ins_quality_dt_ms=5.0,
        exclusion_reasons=() if valid else ("CAL-E-DYNAMIC-BASE",),
    )


def _bag(frames, imu):
    return DynamicBagEvidence(
        bag_id="bag",
        bag_path="/tmp/bag",
        raw_data_identity_sha256="a" * 64,
        topic_message_counts=(),
        odometry=(),
        imu=tuple(imu),
        quality=(),
        frames=tuple(frames),
        diagnostics={},
    )


def test_highpass_motion_rejects_impulse_but_removes_constant_gravity():
    samples = [_imu(index * 0.01) for index in range(201)]
    samples[100] = _imu(1.0, acceleration=(4.0, 0.0, 9.8))
    result = compute_highpass_motion(samples, window_s=1.0)
    assert result[20].acceleration_norm_mps2 == pytest.approx(0.0)
    assert result[100].acceleration_norm_mps2 == pytest.approx(4.0)


def test_first_contact_is_latched_and_outbound_does_not_become_safe_again():
    frames = [
        _frame(1, 0.0, 7.0),
        _frame(2, 1.0, 5.0),
        _frame(3, 2.0, 4.3),
        _frame(4, 3.0, 3.4),
        _frame(5, 4.0, 2.7),
        _frame(6, 5.0, 4.5),
        _frame(7, 6.0, 6.0),
    ]
    imu = [_imu(index * 0.01) for index in range(601)]
    result = classify_dynamic_frames(_bag(frames, imu), CONFIG)
    assert [item.structural_state for item in result[:3]] == [
        StructuralState.UNLOADED_SAFE,
        StructuralState.UNLOADED_SAFE,
        StructuralState.UNLOADED_SAFE,
    ]
    assert result[3].structural_state == StructuralState.CONTACT_TRANSIENT
    assert result[4].structural_state == StructuralState.LOADED_STABLE
    assert result[5].structural_state == StructuralState.RELEASE_TRANSIENT
    assert result[6].structural_state == StructuralState.RELEASE_TRANSIENT
    assert "CAL-E-DYNAMIC-POST-CONTACT" in result[6].exclusion_reasons


def test_known_no_bridge_bag_can_remain_safe_below_contact_depth():
    frames = [_frame(1, 0.0, 5.0), _frame(2, 1.0, 3.3), _frame(3, 2.0, 5.0)]
    imu = [_imu(index * 0.01) for index in range(201)]
    result = classify_dynamic_frames(
        _bag(frames, imu),
        CONFIG,
        bag_known_to_avoid_bridge=True,
    )
    assert result[0].structural_state == StructuralState.UNLOADED_SAFE
    assert result[1].structural_state == StructuralState.UNKNOWN
    assert result[2].structural_state == StructuralState.UNLOADED_SAFE
    assert "CAL-E-DYNAMIC-DEPTH-RANGE" in result[1].exclusion_reasons


def test_missing_imu_or_motion_limit_fails_closed():
    missing = classify_dynamic_frames(_bag([_frame(1, 1.0, 5.0)], []), CONFIG)
    assert missing[0].structural_state == StructuralState.UNKNOWN
    assert "CAL-E-DYNAMIC-IMU-MISSING" in missing[0].exclusion_reasons

    imu = [_imu(index * 0.01) for index in range(201)]
    imu[100] = _imu(1.0, acceleration=(3.0, 0.0, 9.8))
    motion = classify_dynamic_frames(_bag([_frame(1, 1.0, 5.0)], imu), CONFIG)
    assert motion[0].structural_state == StructuralState.UNKNOWN
    assert "CAL-E-DYNAMIC-ACCELERATION" in motion[0].exclusion_reasons

