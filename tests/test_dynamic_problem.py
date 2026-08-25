from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from dynamic_rosbag_calibration.calibration_types import CameraModel
from dynamic_rosbag_calibration.dynamic_experiment_types import (
    PoseInterpolationConfig,
    TimedInsQuality,
    TimedPose,
)
from dynamic_rosbag_calibration.dynamic_problem import (
    associate_ins_quality,
    create_dynamic_apriltag_detector,
    prepare_dynamic_frame,
)
from dynamic_rosbag_calibration.tag_geometry import tag_square_points


@dataclass
class Detection:
    tag_id: int
    corners: np.ndarray
    decision_margin: float = 100.0


class Detector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, _image, estimate_tag_pose=False):
        assert not estimate_tag_pose
        return self.detections


def _project(points, translation, camera):
    pixels, _ = cv2.projectPoints(
        points,
        np.zeros(3),
        np.asarray(translation, dtype=np.float64),
        camera.matrix,
        camera.distortion,
    )
    return pixels.reshape(-1, 2)


def test_quality_association_prefers_earlier_sample_on_tie_and_applies_limit():
    rows = (
        TimedInsQuality(9.99, 3, 56),
        TimedInsQuality(10.01, 1, 0),
    )
    selected, delta_ms = associate_ins_quality(rows, 10.0, maximum_dt_ms=20.0)
    assert selected == rows[0]
    assert delta_ms == pytest.approx(10.0)

    selected, delta_ms = associate_ins_quality(rows, 10.2, maximum_dt_ms=20.0)
    assert selected is None
    assert delta_ms == pytest.approx(190.0)


def test_dynamic_detector_fails_closed_without_production_binding(monkeypatch):
    def unavailable(_config):
        raise RuntimeError("internal live QC requires pyapriltags")

    monkeypatch.setattr(
        "dynamic_rosbag_calibration.dynamic_problem.create_apriltag_detector", unavailable
    )
    with pytest.raises(RuntimeError, match="pyapriltags"):
        create_dynamic_apriltag_detector(object())


def test_prepare_dynamic_frame_uses_tag0_only_depth_and_se3_interpolation():
    camera = CameraModel(
        np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
    )
    square = tag_square_points(0.5)
    detector = Detector(
        [
            Detection(0, _project(square, (0.5, 0.0, 5.0), camera)),
            Detection(1, _project(square, (1.2, 0.0, 5.0), camera)),
        ]
    )
    left = np.eye(4)
    right = np.eye(4)
    right[0, 3] = 2.0
    result = prepare_dynamic_frame(
        bag_id="104757",
        sequence=1,
        source_stamp_s=10.02,
        gray=np.zeros((720, 1280), dtype=np.uint8),
        detector=detector,
        camera=camera,
        tag_size_m=0.5,
        roi_xyxy=(0, 0, 1280, 720),
        odometry=(TimedPose(10.0, left), TimedPose(10.04, right)),
        quality=(TimedInsQuality(10.02, 3, 56),),
        interpolation_config=PoseInterpolationConfig(50.0, 30.0),
        board_instance_id="board_setup_01",
        minimum_tag_edge_px=15.0,
        minimum_margin_px=20.0,
    )

    assert result.initial_tag0_depth_m == pytest.approx(5.0, abs=1e-6)
    assert result.bearing_x_deg == pytest.approx(
        np.degrees(np.arctan2(0.5, 5.0)), abs=1e-6
    )
    assert result.observation is not None
    assert result.observation.map_ins.matrix[0, 3] == pytest.approx(1.0)
    assert result.ins_quality_good
    assert not result.exclusion_reasons


def test_prepare_dynamic_frame_preserves_detection_but_fails_closed_on_timing():
    camera = CameraModel(
        np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
    )
    square = tag_square_points(0.5)
    detector = Detector(
        [
            Detection(0, _project(square, (0.0, 0.0, 5.0), camera)),
            Detection(1, _project(square, (1.0, 0.0, 5.0), camera)),
        ]
    )
    result = prepare_dynamic_frame(
        bag_id="bag",
        sequence=1,
        source_stamp_s=10.2,
        gray=np.zeros((720, 1280), dtype=np.uint8),
        detector=detector,
        camera=camera,
        tag_size_m=0.5,
        roi_xyxy=(0, 0, 1280, 720),
        odometry=(TimedPose(10.0, np.eye(4)), TimedPose(10.04, np.eye(4))),
        quality=(),
        interpolation_config=PoseInterpolationConfig(50.0, 30.0),
        board_instance_id="board_setup_01",
        minimum_tag_edge_px=15.0,
        minimum_margin_px=20.0,
    )
    assert result.camera_tag0_initial is not None
    assert result.observation is None
    assert "CAL-E-DYNAMIC-ODOM-INTERPOLATION" in result.exclusion_reasons
    assert "CAL-E-DYNAMIC-INSPVAX-ASSOCIATION" in result.exclusion_reasons
