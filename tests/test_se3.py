import math

import numpy as np
import pytest

from dynamic_rosbag_calibration.calibration_types import CameraModel, FrameObservation, Transform
from dynamic_rosbag_calibration.se3 import (
    compose,
    exp_se3,
    inverse,
    log_se3,
    right_correct_rx_ry,
    right_correct_rx_ry_rz,
)


def test_exp_log_round_trip_for_small_and_general_twists():
    values = (
        np.array([1e-9, -2e-9, 3e-9, 1e-10, -2e-10, 3e-10]),
        np.array([0.3, -0.2, 1.1, 0.2, -0.4, 0.1]),
        np.array([0.1, 0.2, -0.1, math.pi - 1e-5, 0.0, 0.0]),
    )
    for twist in values:
        recovered = log_se3(exp_se3(twist))
        assert np.allclose(recovered, twist, atol=2e-7)


def test_inverse_and_compose_follow_parent_child_direction():
    map_ins = exp_se3(np.array([2.0, -1.0, 0.3, 0.1, 0.2, -0.1]))
    ins_camera = exp_se3(np.array([0.4, 0.1, -0.2, -0.2, 0.1, 0.3]))
    camera_tag = exp_se3(np.array([0.2, -0.4, 5.0, 0.05, 0.1, -0.05]))
    map_tag = compose(map_ins, ins_camera, camera_tag)
    assert np.allclose(compose(map_tag, inverse(camera_tag), inverse(ins_camera)), map_ins)


def test_right_rx_ry_correction_preserves_translation_and_has_zero_local_rz():
    nominal = exp_se3(np.array([0.4, -0.1, 0.3, 0.2, -0.1, 0.4]))
    corrected = right_correct_rx_ry(nominal, math.radians(1.2), math.radians(-0.7))
    local = log_se3(inverse(nominal) @ corrected)
    assert np.array_equal(corrected[:3, 3], nominal[:3, 3])
    assert np.allclose(local[3:], [math.radians(1.2), math.radians(-0.7), 0.0], atol=1e-12)


def test_right_rx_ry_rz_correction_preserves_translation_and_uses_all_axes():
    nominal = exp_se3(np.array([0.4, -0.1, 0.3, 0.2, -0.1, 0.4]))
    correction = np.radians([1.2, -0.7, 0.35])
    corrected = right_correct_rx_ry_rz(nominal, *correction)
    local = log_se3(inverse(nominal) @ corrected)
    assert np.array_equal(corrected[:3, 3], nominal[:3, 3])
    assert np.allclose(local[3:], correction, atol=1e-12)


def test_calibration_arrays_are_copied_and_read_only():
    matrix = np.eye(4)
    transform = Transform(matrix, "ins_map", "ins_link")
    matrix[0, 3] = 99.0
    assert transform.matrix[0, 3] == 0.0
    with pytest.raises(ValueError):
        transform.matrix[0, 3] = 1.0

    corners = np.arange(16, dtype=float).reshape(8, 2)
    observation = FrameObservation(
        frame_key="s001/000001",
        station_id="s001",
        board_instance_id="board_setup_01",
        map_ins=transform,
        camera_tag0=Transform(np.eye(4), "left_camera", "tag0"),
        corners_px=corners,
        odom_dt_ms=3.0,
        basic_valid=True,
        station_gate_pass=True,
    )
    corners[0, 0] = 999.0
    assert observation.corners_px[0, 0] == 0.0
    with pytest.raises(ValueError):
        observation.corners_px[0, 0] = 1.0


def test_camera_model_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        CameraModel(np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]), np.zeros(5))
