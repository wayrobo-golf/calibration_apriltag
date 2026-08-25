import math

import cv2
import numpy as np

from dynamic_rosbag_calibration.b2_solver import (
    B2SolverConfig,
    deterministic_multistarts,
    solve_calibration,
    station_sqrt_weights,
)
from dynamic_rosbag_calibration.calibration_types import (
    CalibrationProblem,
    CameraModel,
    FrameObservation,
    Transform,
)
from dynamic_rosbag_calibration.observability import diagnose_observability
from dynamic_rosbag_calibration.se3 import exp_se3, inverse, log_se3
from dynamic_rosbag_calibration.tag_geometry import tag_square_points


def project(transform, points, camera):
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    pixels, _ = cv2.projectPoints(points, rvec, transform[:3, 3], camera.matrix, camera.distortion)
    return pixels.reshape(-1, 2)


def synthetic_problem(*, degenerate=False):
    camera = CameraModel(
        np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 360.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
    )
    nominal = exp_se3(np.array([0.25, -0.08, 0.12, -0.04, 0.08, 0.12]))
    correction = np.array([math.radians(0.65), math.radians(-0.45)])
    truth_extrinsic = nominal @ exp_se3(
        np.array([0.0, 0.0, 0.0, correction[0], correction[1], 0.0])
    )
    truth_board = exp_se3(np.array([8.0, 3.0, 1.4, 0.15, -0.1, 0.35]))
    geometry = exp_se3(np.array([0.35, 0.01, 0.04, 0.05, -0.18, 0.07]))
    square = tag_square_points(0.30)
    points = np.vstack(
        (square, (geometry[:3, :3] @ square.T).T + geometry[:3, 3])
    )
    frames = []
    for station in range(6):
        for sample in range(3):
            if degenerate:
                map_ins = exp_se3(np.array([2.0, 1.0, 0.3, 0.0, 0.0, 0.1]))
            else:
                map_ins = exp_se3(
                    np.array(
                        [
                            1.5 + 0.6 * station,
                            -0.5 + 0.25 * sample,
                            0.3 + 0.03 * station,
                            -0.04 + 0.015 * sample,
                            -0.10 + 0.04 * station,
                            -0.25 + 0.10 * station + 0.01 * sample,
                        ]
                    )
                )
            camera_tag0 = inverse(truth_extrinsic) @ inverse(map_ins) @ truth_board
            corners = project(camera_tag0, points, camera)
            frames.append(
                FrameObservation(
                    f"s{station:03d}/{sample:06d}",
                    f"s{station:03d}",
                    "board_setup_01",
                    Transform(map_ins, "ins_map", "ins_link"),
                    Transform(camera_tag0, "left_camera", "tag0"),
                    corners,
                    2.0,
                    True,
                    True,
                )
            )
    return (
        CalibrationProblem(
            tuple(frames),
            Transform(nominal, "ins_link", "left_camera"),
            Transform(geometry, "tag0", "tag1"),
            0.30,
            camera,
            "synthetic-b2",
        ),
        truth_extrinsic,
        truth_board,
        correction,
    )


def fast_config():
    return B2SolverConfig(
        max_nfev_per_start=150,
        irls_max_iterations=5,
        multistart_extrinsic_spread_max_deg=0.02,
        multistart_board_translation_spread_max_m=0.005,
        multistart_board_rotation_spread_max_deg=0.05,
        scaled_condition_number_max=1000.0,
    )


def test_deterministic_plan_has_exactly_twenty_one_unique_starts():
    starts = deterministic_multistarts(("board_setup_01",))
    assert len(starts) == 21
    assert len({start.name for start in starts}) == 21
    assert starts[0].name == "zero"

    rxyz_starts = deterministic_multistarts(
        ("board_setup_01",), extrinsic_rotation_parameter_count=3
    )
    assert len(rxyz_starts) == 21
    assert len({start.name for start in rxyz_starts}) == 21
    assert {"rz_plus", "rz_minus"}.issubset({start.name for start in rxyz_starts})


def test_station_weights_keep_each_station_total_weight_equal_after_duplication():
    stations = np.array(["a", "a", "b", "b", "b", "b"])
    weights = station_sqrt_weights(stations)
    assert np.isclose(np.sum(weights[stations == "a"] ** 2), 1.0)
    assert np.isclose(np.sum(weights[stations == "b"] ** 2), 1.0)


def test_b2_recovers_rx_ry_and_board_pose_without_changing_fixed_parameters():
    problem, truth_extrinsic, truth_board, correction = synthetic_problem()
    result = solve_calibration(problem, fast_config())
    extrinsic_error = log_se3(inverse(truth_extrinsic) @ result.effective_extrinsic.matrix)
    board_error = log_se3(inverse(truth_board) @ result.board_poses["board_setup_01"].matrix)
    nominal = problem.nominal_extrinsic.matrix
    local = log_se3(inverse(nominal) @ result.effective_extrinsic.matrix)
    assert result.quality_gate_pass, result.reasons
    assert len(result.multistart_results) == 21
    assert np.allclose(result.right_correction_rotvec_rad, [*correction, 0.0], atol=2e-5)
    assert np.linalg.norm(extrinsic_error) < 2e-5
    assert np.linalg.norm(board_error) < 2e-5
    assert np.array_equal(result.effective_extrinsic.matrix[:3, 3], nominal[:3, 3])
    assert abs(local[5]) < 1e-12


def test_b2_ablation_recovers_three_axis_rotation_with_fixed_translation():
    problem, _truth_extrinsic, truth_board, _correction = synthetic_problem()
    correction = np.radians([0.65, -0.45, 0.30])
    truth_extrinsic = problem.nominal_extrinsic.matrix @ exp_se3(
        np.array([0.0, 0.0, 0.0, *correction])
    )
    frames = []
    square = tag_square_points(problem.tag_size_m)
    geometry = problem.initial_tag0_tag1.matrix
    points = np.vstack(
        (square, (geometry[:3, :3] @ square.T).T + geometry[:3, 3])
    )
    for frame in problem.frames:
        camera_tag0 = inverse(truth_extrinsic) @ inverse(frame.map_ins.matrix) @ truth_board
        frames.append(
            FrameObservation(
                frame.frame_key,
                frame.station_id,
                frame.board_instance_id,
                frame.map_ins,
                Transform(camera_tag0, "left_camera", "tag0"),
                project(camera_tag0, points, problem.camera_model),
                frame.odom_dt_ms,
                frame.basic_valid,
                frame.station_gate_pass,
            )
        )
    rxyz_problem = CalibrationProblem(
        tuple(frames),
        problem.nominal_extrinsic,
        problem.initial_tag0_tag1,
        problem.tag_size_m,
        problem.camera_model,
        problem.config_fingerprint,
    )
    from dataclasses import replace

    result = solve_calibration(
        rxyz_problem,
        replace(fast_config(), extrinsic_rotation_parameter_count=3),
    )
    local = log_se3(
        inverse(problem.nominal_extrinsic.matrix) @ result.effective_extrinsic.matrix
    )
    assert result.quality_gate_pass, result.reasons
    assert result.algorithm_id == "B2-RXYZ-ABLATION"
    assert np.allclose(local[3:], correction, atol=2e-5)
    assert np.array_equal(
        result.effective_extrinsic.matrix[:3, 3],
        problem.nominal_extrinsic.matrix[:3, 3],
    )


def test_b2_rejects_unsupported_extrinsic_rotation_parameter_count():
    import pytest

    with pytest.raises(ValueError, match="either 2 or 3"):
        B2SolverConfig(extrinsic_rotation_parameter_count=4)


def test_quality_gate_requires_all_twenty_one_starts_to_succeed():
    from dataclasses import replace

    problem, _, _, _ = synthetic_problem()
    result = solve_calibration(
        problem,
        replace(fast_config(), minimum_successful_multistarts=22),
    )
    assert not result.quality_gate_pass
    assert "CAL-E-CALIBRATION-MULTISTART-SUCCESS-COUNT" in result.reasons


def test_observability_report_rejects_rank_deficient_problem():
    jacobian = np.ones((20, 8), dtype=float)
    report = diagnose_observability(jacobian, np.ones(8), 1000.0)
    assert not report.passed
    assert report.numerical_rank == 1
    assert "CAL-E-CALIBRATION-JACOBIAN-RANK" in report.reasons
