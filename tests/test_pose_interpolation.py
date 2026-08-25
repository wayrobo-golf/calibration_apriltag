import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dynamic_rosbag_calibration.dynamic_experiment_types import PoseInterpolationConfig, TimedPose
from dynamic_rosbag_calibration.pose_interpolation import interpolate_pose_se3


CONFIG = PoseInterpolationConfig(
    maximum_bracket_gap_ms=50.0,
    maximum_endpoint_distance_ms=30.0,
)


def _pose(stamp_s: float, xyz=(0.0, 0.0, 0.0), yaw_deg=0.0) -> TimedPose:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    matrix[:3, 3] = xyz
    return TimedPose(stamp_s=stamp_s, matrix=matrix)


def test_interpolation_uses_linear_translation_and_rotation_slerp():
    result = interpolate_pose_se3(
        (_pose(10.0, (0.0, 0.0, 0.0), 0.0), _pose(10.04, (2.0, 4.0, 0.0), 90.0)),
        10.02,
        CONFIG,
    )

    assert result.pose.matrix[:3, 3] == pytest.approx([1.0, 2.0, 0.0])
    yaw = Rotation.from_matrix(
        np.array(result.pose.matrix[:3, :3], copy=True)
    ).as_euler("zyx", degrees=True)[0]
    assert yaw == pytest.approx(45.0)
    rotation = result.pose.matrix[:3, :3]
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert result.alpha == pytest.approx(0.5)
    assert result.bracket_gap_ms == pytest.approx(40.0)
    assert result.left_dt_ms == pytest.approx(20.0)
    assert result.right_dt_ms == pytest.approx(20.0)


def test_exact_timestamp_returns_exact_pose_without_division_by_zero():
    expected = _pose(10.0, (1.0, 2.0, 3.0), 30.0)
    result = interpolate_pose_se3((expected, _pose(10.02)), 10.0, CONFIG)
    assert result.pose.matrix == pytest.approx(expected.matrix)
    assert result.alpha == 0.0
    assert result.left_stamp_s == result.right_stamp_s == 10.0
    assert result.bracket_gap_ms == 0.0


@pytest.mark.parametrize("stamp_s", [9.99, 10.03])
def test_interpolation_forbids_extrapolation(stamp_s):
    with pytest.raises(ValueError, match="outside odometry range"):
        interpolate_pose_se3((_pose(10.0), _pose(10.02)), stamp_s, CONFIG)


def test_interpolation_rejects_large_bracket_gap():
    with pytest.raises(ValueError, match="bracket gap"):
        interpolate_pose_se3((_pose(10.0), _pose(10.06)), 10.03, CONFIG)


def test_interpolation_rejects_large_endpoint_distance():
    config = PoseInterpolationConfig(100.0, 20.0)
    with pytest.raises(ValueError, match="endpoint distance"):
        interpolate_pose_se3((_pose(10.0), _pose(10.05)), 10.025, config)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ((), "at least one"),
        ((_pose(10.0), _pose(10.0)), "strictly increasing"),
        ((_pose(10.01), _pose(10.0)), "strictly increasing"),
    ],
)
def test_interpolation_rejects_empty_duplicate_or_reversed_input(rows, message):
    with pytest.raises(ValueError, match=message):
        interpolate_pose_se3(rows, 10.0, CONFIG)


def test_timed_pose_rejects_nonfinite_or_invalid_homogeneous_matrix():
    with pytest.raises(ValueError, match="finite"):
        _pose(math.nan)
    matrix = np.eye(4)
    matrix[3, 3] = 2.0
    with pytest.raises(ValueError, match="homogeneous"):
        TimedPose(10.0, matrix)
