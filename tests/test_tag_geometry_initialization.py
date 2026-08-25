from dataclasses import replace

import cv2
import numpy as np
import pytest

import dynamic_rosbag_calibration.tag_geometry_initialization as initialization_module

from dynamic_rosbag_calibration.calibration_types import CameraModel, FrameObservation, Transform
from dynamic_rosbag_calibration.se3 import exp_se3, inverse, log_se3
from dynamic_rosbag_calibration.tag_geometry import TagGeometryConfig, estimate_tag_geometry, tag_square_points
from dynamic_rosbag_calibration.tag_geometry_initialization import (
    TagGeometryInitializationError,
    initialize_tag_geometry,
)


def project(transform, points, camera):
    rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
    pixels, _ = cv2.projectPoints(
        points, rotation_vector, transform[:3, 3], camera.matrix, camera.distortion
    )
    return pixels.reshape(-1, 2)


def raw_frames(station_count=6, view_count=5):
    camera = CameraModel(
        np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 360.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
    )
    truth = exp_se3(np.array([0.35, 0.01, 0.04, 0.05, -0.18, 0.07]))
    square = tag_square_points(0.30)
    frames = []
    for station in range(station_count):
        for view in range(view_count):
            camera_tag0 = exp_se3(
                np.array(
                    [
                        -0.25 + 0.09 * station,
                        -0.08 + 0.04 * view,
                        3.0 + 0.35 * station,
                        -0.08 + 0.03 * view,
                        0.10 - 0.025 * station,
                        -0.06 + 0.02 * view,
                    ]
                )
            )
            corners = np.vstack(
                (
                    project(camera_tag0, square, camera),
                    project(camera_tag0 @ truth, square, camera),
                )
            )
            frames.append(
                FrameObservation(
                    f"s{station:03d}/{view:06d}",
                    f"s{station:03d}",
                    "board",
                    Transform(np.eye(4), "ins_map", "ins_link"),
                    None,
                    corners,
                    2.0,
                    True,
                    True,
                )
            )
    return tuple(frames), camera, Transform(truth, "tag0", "tag1")


def geometry_config():
    return TagGeometryConfig(max_nfev=300)


def test_bootstrap_limit_does_not_limit_all_frame_seeding(monkeypatch):
    frames, camera, truth = raw_frames(station_count=6, view_count=20)
    seen = {}

    def fake_bootstrap(values, camera_model, tag_size_m, minimum_valid_stations):
        seen["bootstrap"] = tuple(values)
        seen["minimum_valid_stations"] = minimum_valid_stations
        return truth

    def fake_seed(values, camera_model, tag_size_m, initial, minimum_valid_stations):
        seen["seed"] = tuple(values)
        seen["seed_minimum_valid_stations"] = minimum_valid_stations
        return tuple(values)

    monkeypatch.setattr(initialization_module, "_bootstrap_initial", fake_bootstrap)
    monkeypatch.setattr(initialization_module, "_seed_frames", fake_seed)

    result = initialize_tag_geometry(
        frames,
        camera,
        0.30,
        "bootstrap",
        None,
        None,
        None,
        bootstrap_maximum_views_per_station=5,
    )

    assert result.initial_transform is truth
    assert len(seen["bootstrap"]) == 30
    assert len(seen["seed"]) == 120
    assert seen["minimum_valid_stations"] == 6
    assert seen["seed_minimum_valid_stations"] == 6


def test_bootstrap_recovers_geometry_without_old_matrix():
    frames, camera, truth = raw_frames()
    result = initialize_tag_geometry(
        frames, camera, 0.30, "bootstrap", None, None, None
    )
    error = log_se3(inverse(truth.matrix) @ result.initial_transform.matrix)
    assert np.linalg.norm(error[:3]) < 0.10
    assert np.degrees(np.linalg.norm(error[3:])) < 5.0
    assert len({frame.station_id for frame in result.seeded_frames}) == 6
    assert result.source_path is None
    assert result.source_sha256 is None


def test_reuse_is_only_seed_and_geometry_is_still_optimized():
    frames, camera, truth = raw_frames()
    biased = Transform(
        truth.matrix @ exp_se3(np.array([0.03, 0.0, 0.0, 0.0, 0.03, 0.0])),
        "tag0",
        "tag1",
    )
    initialization = initialize_tag_geometry(
        frames,
        camera,
        0.30,
        "reuse",
        biased,
        "old.yaml",
        "a" * 64,
    )
    fitted = estimate_tag_geometry(initialization, camera, 0.30, geometry_config())
    assert fitted.initialization is initialization
    assert not np.allclose(fitted.candidate.matrix, biased.matrix)
    error = log_se3(inverse(truth.matrix) @ fitted.candidate.matrix)
    assert np.linalg.norm(error[:3]) < 1e-4
    assert np.degrees(np.linalg.norm(error[3:])) < 1e-3


def test_bootstrap_requires_support_from_six_distinct_stations():
    frames, camera, _ = raw_frames(station_count=5)
    with pytest.raises(
        TagGeometryInitializationError,
        match="CAL-E-TAG-GEOMETRY-INITIALIZATION-STATION-COUNT",
    ):
        initialize_tag_geometry(
            frames, camera, 0.30, "bootstrap", None, None, None
        )


def test_bootstrap_honors_an_explicit_two_station_offline_gate():
    frames, camera, truth = raw_frames(station_count=2, view_count=12)

    result = initialize_tag_geometry(
        frames,
        camera,
        0.30,
        "bootstrap",
        None,
        None,
        None,
        bootstrap_maximum_views_per_station=12,
        minimum_valid_stations=2,
    )

    error = log_se3(inverse(truth.matrix) @ result.initial_transform.matrix)
    assert np.linalg.norm(error[:3]) < 0.10
    assert np.degrees(np.linalg.norm(error[3:])) < 5.0
    assert len({frame.station_id for frame in result.seeded_frames}) == 2


def test_bootstrap_is_deterministic_under_input_order_changes():
    frames, camera, _ = raw_frames()
    first = initialize_tag_geometry(
        frames, camera, 0.30, "bootstrap", None, None, None
    )
    second = initialize_tag_geometry(
        tuple(reversed(frames)), camera, 0.30, "bootstrap", None, None, None
    )
    assert np.allclose(first.initial_transform.matrix, second.initial_transform.matrix)
    assert tuple(frame.frame_key for frame in first.seeded_frames) == tuple(
        frame.frame_key for frame in second.seeded_frames
    )
