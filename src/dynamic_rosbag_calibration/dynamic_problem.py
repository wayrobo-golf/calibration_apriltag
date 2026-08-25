"""Read-only ROS2 bag adapter for safe-dynamic calibration evidence."""

from __future__ import annotations

import bisect
from collections import Counter
import math
import os
from pathlib import Path
from dataclasses import replace
from typing import Callable, Sequence

import numpy as np

from .calibration_types import CalibrationProblem, CameraModel, FrameObservation, Transform
from .config import LiveQcConfig, TagGeometryRunConfig
from .dynamic_experiment_types import (
    DynamicBagEvidence,
    DynamicExperimentTopics,
    PoseInterpolationConfig,
    RawDynamicFrame,
    TimedImu,
    TimedInsQuality,
    TimedPose,
)
from .io import build_raw_data_identity
from .live_qc import (
    create_apriltag_detector,
    detect_dual_tag,
    image_message_to_gray,
    load_live_qc_profile,
)
from .pnp_prepare import solve_single_tag_pnp
from .pose_interpolation import interpolate_pose_se3
from .rosbag_problem import _message_stamp_s, _pose_message_transform
from .rosbag_problem import (
    _runtime_transform,
    _single_yaml,
    load_tag_geometry_input,
    machine_config_fingerprint,
)


INS_SOLUTION_GOOD = 3
INS_RTKFIXED = 56
DEFAULT_MAXIMUM_QUALITY_ASSOCIATION_MS = 100.0


def create_dynamic_apriltag_detector(config: LiveQcConfig) -> object:
    """Create the production detector; calibration must fail if it is unavailable."""
    detector = create_apriltag_detector(config)
    try:
        setattr(detector, "implementation_id", "pyapriltags_production_profile")
    except (AttributeError, TypeError):
        pass
    return detector


def _strictly_increasing(values: Sequence[object], name: str) -> None:
    stamps = [float(getattr(item, "stamp_s")) for item in values]
    if any(right <= left for left, right in zip(stamps, stamps[1:])):
        raise ValueError(f"{name} source stamps must be strictly increasing")


def _vector3(value: object) -> np.ndarray:
    result = np.asarray(
        [float(value.x), float(value.y), float(value.z)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("sensor vector must contain finite values")
    return result


def associate_ins_quality(
    rows: Sequence[TimedInsQuality],
    stamp_s: float,
    *,
    maximum_dt_ms: float = DEFAULT_MAXIMUM_QUALITY_ASSOCIATION_MS,
) -> tuple[TimedInsQuality | None, float | None]:
    """Associate the nearest quality sample, preferring the earlier sample on ties."""
    if not rows:
        return None, None
    stamps = [item.stamp_s for item in rows]
    index = bisect.bisect_left(stamps, stamp_s)
    candidates = [value for value in (index - 1, index) if 0 <= value < len(rows)]
    selected_index = min(
        candidates,
        key=lambda value: (abs(stamps[value] - stamp_s), stamps[value]),
    )
    delta_ms = abs(stamps[selected_index] - stamp_s) * 1000.0
    if delta_ms > maximum_dt_ms:
        return None, float(delta_ms)
    return rows[selected_index], float(delta_ms)


def _candidate_message_definition_dirs() -> tuple[Path, ...]:
    result: list[Path] = []
    for variable in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"):
        for prefix in os.environ.get(variable, "").split(os.pathsep):
            if not prefix:
                continue
            result.append(
                Path(prefix) / "share/novatel_oem7_msgs/msg"
            )
    return tuple(dict.fromkeys(result))


def build_dynamic_typestore(
    novatel_message_dir: Path | None = None,
):
    """Build a Humble typestore with all installed NovAtel message definitions."""
    try:
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as error:
        raise RuntimeError("dynamic bag loading requires rosbags") from error
    candidates = (
        (Path(novatel_message_dir),)
        if novatel_message_dir is not None
        else _candidate_message_definition_dirs()
    )
    roots = [path for path in candidates if path.is_dir()]
    if not roots:
        searched = ", ".join(str(path) for path in candidates) or "<none>"
        raise ValueError(
            "novatel_oem7_msgs .msg directory was not found; "
            f"searched: {searched}"
        )
    root = roots[0]
    definitions: dict[str, object] = {}
    for path in sorted(root.glob("*.msg")):
        definitions.update(
            get_types_from_msg(
                path.read_text(encoding="utf-8"),
                f"novatel_oem7_msgs/msg/{path.stem}",
            )
        )
    if "novatel_oem7_msgs/msg/INSPVAX" not in definitions:
        raise ValueError(f"INSPVAX.msg is missing from {root}")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(definitions)
    return typestore


def prepare_dynamic_frame(
    *,
    bag_id: str,
    sequence: int,
    source_stamp_s: float,
    gray: np.ndarray,
    detector: object,
    camera: CameraModel,
    tag_size_m: float,
    roi_xyxy: tuple[int, int, int, int],
    odometry: Sequence[TimedPose],
    quality: Sequence[TimedInsQuality],
    interpolation_config: PoseInterpolationConfig,
    board_instance_id: str,
    minimum_tag_edge_px: float,
    minimum_margin_px: float,
) -> RawDynamicFrame:
    """Create candidate-independent single-Tag depth and interpolated INS evidence."""
    reasons: list[str] = []
    interpolation = None
    try:
        interpolation = interpolate_pose_se3(
            odometry,
            source_stamp_s,
            interpolation_config,
        )
    except ValueError:
        reasons.append("CAL-E-DYNAMIC-ODOM-INTERPOLATION")

    quality_sample, quality_dt_ms = associate_ins_quality(quality, source_stamp_s)
    quality_good = bool(quality_sample is not None and quality_sample.good)
    if quality_sample is None:
        reasons.append("CAL-E-DYNAMIC-INSPVAX-ASSOCIATION")
    elif not quality_good:
        reasons.append("CAL-E-DYNAMIC-INS-QUALITY")

    detected = detect_dual_tag(gray, detector, roi_xyxy)
    camera_tag0 = None
    depth = None
    bearing = None
    corners = None
    minimum_edge = 0.0
    minimum_margin = 0.0
    if detected is None:
        reasons.append("CAL-E-DYNAMIC-DUAL-TAG")
    else:
        corners = detected.corners_px
        minimum_edge = detected.minimum_tag_edge_px
        minimum_margin = detected.minimum_margin_px
        if minimum_edge < minimum_tag_edge_px:
            reasons.append("CAL-E-DYNAMIC-TAG-SIZE")
        if minimum_margin < minimum_margin_px:
            reasons.append("CAL-E-DYNAMIC-TAG-MARGIN")
        single = solve_single_tag_pnp(corners[:4], camera, tag_size_m)
        if single.selected is None:
            reasons.append("CAL-E-DYNAMIC-TAG0-PNP")
        else:
            camera_tag0 = Transform(
                single.selected.matrix,
                "left_camera",
                "tag0",
            )
            translation = camera_tag0.matrix[:3, 3]
            depth = float(translation[2])
            bearing = math.degrees(math.atan2(float(translation[0]), depth))

    observation = None
    if interpolation is not None and corners is not None and camera_tag0 is not None:
        observation = FrameObservation(
            frame_key=f"{bag_id}/{sequence:09d}",
            station_id=bag_id,
            board_instance_id=board_instance_id,
            map_ins=interpolation.pose,
            camera_tag0=camera_tag0,
            corners_px=corners,
            odom_dt_ms=max(interpolation.left_dt_ms, interpolation.right_dt_ms),
            basic_valid=not reasons,
            station_gate_pass=False,
        )
    return RawDynamicFrame(
        frame_key=f"{bag_id}/{sequence:09d}",
        bag_id=bag_id,
        sequence=sequence,
        source_stamp_s=source_stamp_s,
        observation=observation,
        camera_tag0_initial=camera_tag0,
        initial_tag0_depth_m=depth,
        bearing_x_deg=bearing,
        minimum_tag_edge_px=minimum_edge,
        minimum_margin_px=minimum_margin,
        interpolation=interpolation,
        ins_quality_good=quality_good,
        ins_quality_dt_ms=quality_dt_ms,
        exclusion_reasons=tuple(dict.fromkeys(reasons)),
    )


def read_dynamic_bag_evidence(
    bag_path: Path,
    bag_id: str,
    machine_config_dir: Path,
    *,
    topics: DynamicExperimentTopics,
    interpolation_config: PoseInterpolationConfig,
    live_qc_config: LiveQcConfig,
    board_instance_id: str,
    novatel_message_dir: Path | None = None,
    typestore: object | None = None,
    detector: object | None = None,
    identity_builder: Callable[[Path], dict[str, object]] = build_raw_data_identity,
) -> DynamicBagEvidence:
    """Read one bag twice so images are detected once after time evidence is frozen."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise RuntimeError("dynamic bag loading requires rosbags") from error
    path = Path(bag_path).resolve()
    if not path.is_dir() or not (path / "metadata.yaml").is_file():
        raise ValueError(f"dynamic bag does not exist or lacks metadata.yaml: {path}")
    store = typestore or build_dynamic_typestore(novatel_message_dir)
    camera, tag_size_m = load_live_qc_profile(Path(machine_config_dir))
    tag_detector = detector or create_dynamic_apriltag_detector(live_qc_config)
    required_topics = {topics.image, topics.odom, topics.inspvax, topics.imu}

    odometry: list[TimedPose] = []
    imu: list[TimedImu] = []
    quality: list[TimedInsQuality] = []
    topic_counts: dict[str, int] = {}
    with AnyReader([path], default_typestore=store) as reader:
        for connection in reader.connections:
            topic_counts[connection.topic] = topic_counts.get(connection.topic, 0) + int(
                connection.msgcount
            )
        missing = sorted(topic for topic in required_topics if topic_counts.get(topic, 0) <= 0)
        if missing:
            raise ValueError(f"{path} is missing required dynamic topics: {missing}")
        connections = [
            item
            for item in reader.connections
            if item.topic in {topics.odom, topics.inspvax, topics.imu}
        ]
        for connection, _receive_ns, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            stamp_s = _message_stamp_s(message)
            if connection.topic == topics.odom:
                if (
                    str(message.header.frame_id) != "ins_map"
                    or str(message.child_frame_id) != "ins_link"
                ):
                    raise ValueError(
                        f"{bag_id} odometry must be ins_map <- ins_link"
                    )
                odometry.append(TimedPose(stamp_s, _pose_message_transform(message)))
            elif connection.topic == topics.imu:
                imu.append(
                    TimedImu(
                        stamp_s,
                        _vector3(message.linear_acceleration),
                        _vector3(message.angular_velocity),
                    )
                )
            else:
                quality.append(
                    TimedInsQuality(
                        stamp_s,
                        int(message.ins_status.status),
                        int(message.pos_type.type),
                    )
                )
    _strictly_increasing(odometry, "odometry")
    _strictly_increasing(imu, "IMU")
    _strictly_increasing(quality, "INSPVAX")

    frames: list[RawDynamicFrame] = []
    last_image_stamp: float | None = None
    with AnyReader([path], default_typestore=store) as reader:
        image_connections = [
            item for item in reader.connections if item.topic == topics.image
        ]
        for sequence, (connection, _receive_ns, raw) in enumerate(
            reader.messages(connections=image_connections),
            1,
        ):
            message = reader.deserialize(raw, connection.msgtype)
            stamp_s = _message_stamp_s(message)
            if last_image_stamp is not None and stamp_s <= last_image_stamp:
                raise ValueError("image source stamps must be strictly increasing")
            last_image_stamp = stamp_s
            frames.append(
                prepare_dynamic_frame(
                    bag_id=bag_id,
                    sequence=sequence,
                    source_stamp_s=stamp_s,
                    gray=image_message_to_gray(message),
                    detector=tag_detector,
                    camera=camera,
                    tag_size_m=tag_size_m,
                    roi_xyxy=live_qc_config.roi_xyxy,
                    odometry=odometry,
                    quality=quality,
                    interpolation_config=interpolation_config,
                    board_instance_id=board_instance_id,
                    minimum_tag_edge_px=15.0,
                    minimum_margin_px=20.0,
                )
            )
    if not frames:
        raise ValueError(f"dynamic bag {bag_id} contains no images")

    identity = identity_builder(path)
    reason_counts = Counter(
        reason for frame in frames for reason in frame.exclusion_reasons
    )
    return DynamicBagEvidence(
        bag_id=bag_id,
        bag_path=path,
        raw_data_identity_sha256=str(identity["identity_sha256"]),
        topic_message_counts=tuple(sorted(topic_counts.items())),
        odometry=tuple(odometry),
        imu=tuple(imu),
        quality=tuple(quality),
        frames=tuple(frames),
        diagnostics={
            "detector_implementation": str(
                getattr(tag_detector, "implementation_id", type(tag_detector).__name__)
            ),
            "image_count": len(frames),
            "interpolated_frame_count": sum(
                frame.interpolation is not None for frame in frames
            ),
            "dual_tag_frame_count": sum(
                frame.camera_tag0_initial is not None for frame in frames
            ),
            "base_valid_frame_count": sum(
                frame.observation is not None and not frame.exclusion_reasons
                for frame in frames
            ),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        },
    )


def build_calibration_problem_from_dynamic_frames(
    frames: Sequence[object],
    machine_config_dir: Path,
    *,
    tag_geometry_run_config: TagGeometryRunConfig,
) -> CalibrationProblem:
    """Project selected experiment evidence onto the existing math contract."""
    observations: list[FrameObservation] = []
    for item in frames:
        source = getattr(item, "source", item)
        observation = getattr(source, "observation", None)
        if observation is None:
            continue
        observations.append(
            replace(
                observation,
                basic_valid=True,
                station_gate_pass=True,
            )
        )
    if not observations:
        raise ValueError("dynamic calibration selection contains no observations")
    config_dir = Path(machine_config_dir)
    runtime = _single_yaml(config_dir / "tf_static_golfC_param.yaml")
    params = runtime["static_calibration"]["ros__parameters"]
    nominal = _runtime_transform(params["lidar_to_ins_extrisric"]) @ _runtime_transform(
        params["left_camera_to_lidar_extrinsic"]
    )
    camera, _profile_tag_size = load_live_qc_profile(config_dir)
    tag_size_m, geometry, geometry_source_sha256 = load_tag_geometry_input(
        config_dir / "tag0_tag1_extrinsic.yaml",
        tag_geometry_run_config.initialization_mode,
        tag_geometry_run_config.reuse_initial_path,
    )
    return CalibrationProblem(
        frames=tuple(observations),
        nominal_extrinsic=Transform(nominal, "ins_link", "left_camera"),
        initial_tag0_tag1=geometry,
        tag_size_m=tag_size_m,
        camera_model=camera,
        config_fingerprint=machine_config_fingerprint(config_dir),
        initial_tag0_tag1_source_path=(
            tag_geometry_run_config.reuse_initial_path
            if tag_geometry_run_config.initialization_mode == "reuse"
            else None
        ),
        initial_tag0_tag1_source_sha256=geometry_source_sha256,
    )
