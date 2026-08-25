import numpy as np

from dynamic_rosbag_calibration.calibration_types import FrameObservation, Transform
from dynamic_rosbag_calibration.dynamic_experiment_types import (
    DynamicFrameEvidence,
    DynamicSamplingConfig,
    RawDynamicFrame,
    StructuralState,
)
from dynamic_rosbag_calibration.dynamic_sampling import select_balanced_dynamic_frames


SAMPLING = DynamicSamplingConfig(
    minimum_xy_increment_m=0.10,
    minimum_yaw_increment_deg=0.20,
    maximum_time_increment_s=0.50,
    maximum_frames_per_cell=8,
    maximum_frames_per_bag=48,
    minimum_bag_count=6,
    minimum_bags_per_distance_bin=3,
    minimum_total_frames=18,
)


def _evidence(bag_id, sequence, stamp, x, depth, bearing, state=StructuralState.UNLOADED_SAFE):
    map_ins = np.eye(4)
    map_ins[0, 3] = x
    observation = FrameObservation(
        frame_key=f"{bag_id}/{sequence:09d}",
        station_id=bag_id,
        board_instance_id="board_setup_01",
        map_ins=Transform(map_ins, "ins_map", "ins_link"),
        camera_tag0=Transform(np.eye(4), "left_camera", "tag0"),
        corners_px=np.zeros((8, 2)),
        odom_dt_ms=1.0,
        basic_valid=True,
        station_gate_pass=False,
    )
    raw = RawDynamicFrame(
        frame_key=observation.frame_key,
        bag_id=bag_id,
        sequence=sequence,
        source_stamp_s=stamp,
        observation=observation,
        camera_tag0_initial=None,
        initial_tag0_depth_m=depth,
        bearing_x_deg=bearing,
        minimum_tag_edge_px=20.0,
        minimum_margin_px=30.0,
        interpolation=None,
        ins_quality_good=True,
        ins_quality_dt_ms=1.0,
        exclusion_reasons=(),
    )
    return DynamicFrameEvidence(raw, state, 0.0, 0.0, ())


def _covered_frames():
    frames = []
    for bag_index in range(6):
        bag = f"b{bag_index}"
        bearing = -10.0 + bag_index
        for sequence, depth in enumerate((4.3, 5.5, 7.0), 1):
            frames.append(
                _evidence(
                    bag,
                    sequence,
                    sequence * 0.6,
                    sequence * 0.2,
                    depth,
                    bearing,
                )
            )
    return frames


def test_balanced_selection_passes_dynamic_b1_coverage_and_is_deterministic():
    frames = _covered_frames()
    first = select_balanced_dynamic_frames(
        frames,
        SAMPLING,
        distance_edges_m=(4.5, 6.5),
        minimum_bearing_span_deg=4.0,
    )
    second = select_balanced_dynamic_frames(
        list(reversed(frames)),
        SAMPLING,
        distance_edges_m=(4.5, 6.5),
        minimum_bearing_span_deg=4.0,
    )
    assert first.coverage.passed
    assert len(first.frames) == 18
    assert [item.source.frame_key for item in first.frames] == [
        item.source.frame_key for item in second.frames
    ]
    assert first.selected_frame_keys_sha256 == second.selected_frame_keys_sha256
    assert dict(first.coverage.distance_bin_bag_counts) == {
        "far": 6,
        "middle": 6,
        "near": 6,
    }


def test_selection_excludes_non_safe_frames():
    frames = _covered_frames()
    contaminated = _evidence(
        "b0", 99, 99.0, 99.0, 5.0, -7.0, StructuralState.LOADED_STABLE
    )
    result = select_balanced_dynamic_frames(
        [*frames, contaminated],
        SAMPLING,
        distance_edges_m=(4.5, 6.5),
        minimum_bearing_span_deg=4.0,
    )
    assert contaminated.source.frame_key not in {
        item.source.frame_key for item in result.frames
    }


def test_coverage_fails_when_bearing_span_or_bag_count_is_insufficient():
    frames = [
        _evidence(f"b{index}", sequence, sequence, sequence, depth, -7.0)
        for index in range(5)
        for sequence, depth in enumerate((4.3, 5.5, 7.0), 1)
    ]
    result = select_balanced_dynamic_frames(
        frames,
        SAMPLING,
        distance_edges_m=(4.5, 6.5),
        minimum_bearing_span_deg=4.0,
    )
    assert not result.coverage.passed
    assert "CAL-E-DYNAMIC-COVERAGE-BAG-COUNT" in result.coverage.reasons
    assert "CAL-E-DYNAMIC-COVERAGE-BEARING-SPAN" in result.coverage.reasons


def test_keyframe_rule_drops_highly_correlated_frames():
    frames = [
        _evidence("b0", index, index * 0.1, index * 0.01, 5.0, -7.0)
        for index in range(1, 11)
    ]
    result = select_balanced_dynamic_frames(
        frames,
        SAMPLING,
        distance_edges_m=(4.5, 6.5),
        minimum_bearing_span_deg=4.0,
    )
    assert len(result.keyframe_keys) < len(frames)
