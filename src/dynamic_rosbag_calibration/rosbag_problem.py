"""ROS2 bag adapter that builds immutable calibration observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .calibration_types import (
    CalibrationProblem,
    CameraModel,
    FrameObservation,
    Transform,
)
from .io import canonical_json_bytes, sha256_bytes, sha256_file
from .config import LiveQcConfig, TagGeometryRunConfig
from .live_qc import create_apriltag_detector, detect_dual_tag, image_message_to_gray
from .models import StationRecord
from .tag_geometry import tag_square_points


NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class TimedPose:
    stamp_s: float
    matrix: np.ndarray


def _single_yaml(path: Path) -> dict[str, Any]:
    documents = [
        item for item in yaml.safe_load_all(path.read_text(encoding="utf-8")) if item
    ]
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise ValueError(f"{path} must contain exactly one YAML mapping")
    return documents[0]


def _runtime_transform(item: dict[str, Any]) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_quat(
        [item["qx"], item["qy"], item["qz"], item["qw"]]
    ).as_matrix()
    value[:3, 3] = [item["tx"], item["ty"], item["tz"]]
    return value


def _pose_message_transform(message: Any) -> np.ndarray:
    pose = message.pose.pose
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_quat(
        [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ]
    ).as_matrix()
    value[:3, 3] = [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    ]
    return value


_image_message_to_gray = image_message_to_gray


def _nearest_pose(rows: Sequence[TimedPose], stamp_s: float) -> tuple[TimedPose, float]:
    if not rows:
        raise ValueError("cannot associate an image without odometry")
    stamps = np.fromiter((row.stamp_s for row in rows), dtype=np.float64)
    index = int(np.searchsorted(stamps, stamp_s))
    candidates = [value for value in (index - 1, index) if 0 <= value < len(rows)]
    selected = min(candidates, key=lambda value: abs(rows[value].stamp_s - stamp_s))
    return rows[selected], abs(rows[selected].stamp_s - stamp_s)


def _message_stamp_s(message: object) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        raise ValueError("bag message has no header.stamp")
    value = float(stamp.sec) + float(stamp.nanosec) / NANOSECONDS_PER_SECOND
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("bag message has invalid header.stamp")
    return value


def _in_station_window(stamp_s: float, record: StationRecord) -> bool:
    if record.started_at_s is None or record.ended_at_s is None:
        return False
    return record.started_at_s <= stamp_s <= record.ended_at_s


def load_tag_geometry_input(
    machine_geometry_path: Path,
    mode: str,
    reuse_path: Path | None,
) -> tuple[float, Transform | None, str | None]:
    machine_payload = _single_yaml(Path(machine_geometry_path))
    tag_size_m = float(machine_payload["tag_size_m"])
    if mode == "bootstrap":
        return tag_size_m, None, None
    if mode != "reuse" or reuse_path is None:
        raise ValueError("tag geometry mode must be bootstrap or configured reuse")
    reuse_payload = _single_yaml(Path(reuse_path))
    initial = Transform(
        np.asarray(reuse_payload["transform"]["matrix"], dtype=np.float64),
        "tag0",
        "tag1",
    )
    return tag_size_m, initial, sha256_file(Path(reuse_path))


def _initial_camera_tag0(
    corners_px: np.ndarray,
    camera: CameraModel,
    geometry: Transform,
    tag_size_m: float,
) -> np.ndarray:
    square = tag_square_points(tag_size_m)
    transform = geometry.matrix
    tag1 = (transform[:3, :3] @ square.T).T + transform[:3, 3]
    points = np.vstack((square, tag1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        points,
        np.asarray(corners_px, dtype=np.float64).reshape(8, 2),
        camera.matrix,
        camera.distortion,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not success:
        raise ValueError("initial dual-Tag SQPnP failed")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = np.asarray(translation_vector).reshape(3)
    camera_points = (rotation @ points.T).T + value[:3, 3]
    if np.any(camera_points[:, 2] <= 0.0):
        raise ValueError("initial dual-Tag SQPnP has non-positive depth")
    return value


def _detect_dual_tag(gray: np.ndarray, detector: Any) -> np.ndarray | None:
    result = detect_dual_tag(gray, detector, (560, 0, 1280, 650))
    return None if result is None else result.corners_px


def load_problem_from_rosbags(
    records: tuple[StationRecord, ...],
    machine_config_dir: Path,
    *,
    tag_geometry_run_config: TagGeometryRunConfig,
    image_topic: str = "/camera/left/image",
    odom_topic: str = "/novatel/oem7/ins_odom",
    live_qc_config: LiveQcConfig | None = None,
) -> CalibrationProblem:
    """Replay the six frozen station bags; this does not record new data."""
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise RuntimeError("bag calibration requires pyapriltags and rosbags") from error

    config_dir = Path(machine_config_dir)
    runtime_path = config_dir / "tf_static_golfC_param.yaml"
    camera_path = config_dir / "camera_info"
    geometry_path = config_dir / "tag0_tag1_extrinsic.yaml"
    runtime = _single_yaml(runtime_path)
    params = runtime["static_calibration"]["ros__parameters"]
    nominal = _runtime_transform(params["lidar_to_ins_extrisric"]) @ _runtime_transform(
        params["left_camera_to_lidar_extrinsic"]
    )
    camera_payload = _single_yaml(camera_path)
    camera = CameraModel(
        np.asarray(camera_payload["data"], dtype=np.float64).reshape(3, 3),
        np.asarray(
            camera_payload.get("distortion_coefficients", {}).get(
                "data", np.zeros(5)
            ),
            dtype=np.float64,
        ),
    )
    tag_size_m, geometry, geometry_source_sha256 = load_tag_geometry_input(
        geometry_path,
        tag_geometry_run_config.initialization_mode,
        tag_geometry_run_config.reuse_initial_path,
    )
    if live_qc_config is None:
        from pyapriltags import Detector

        detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        roi_xyxy = (560, 0, 1280, 650)
    else:
        detector = create_apriltag_detector(live_qc_config)
        roi_xyxy = live_qc_config.roi_xyxy
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    frames: list[FrameObservation] = []
    for record in records:
        if record.bag_path is None:
            raise ValueError(f"{record.station_id} has no frozen bag path")
        bag_path = Path(record.bag_path)
        odometry: list[TimedPose] = []
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            connections = [c for c in reader.connections if c.topic == odom_topic]
            for connection, _timestamp_ns, raw in reader.messages(
                connections=connections
            ):
                message = reader.deserialize(raw, connection.msgtype)
                stamp_s = _message_stamp_s(message)
                if not _in_station_window(stamp_s, record):
                    continue
                if (
                    str(message.header.frame_id) != "ins_map"
                    or str(message.child_frame_id) != "ins_link"
                ):
                    raise ValueError(
                        f"{record.station_id} odometry must be ins_map <- ins_link"
                    )
                odometry.append(
                    TimedPose(
                        stamp_s,
                        _pose_message_transform(message),
                    )
                )
        odometry.sort(key=lambda value: value.stamp_s)
        sequence = 0
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            connections = [c for c in reader.connections if c.topic == image_topic]
            for connection, _timestamp_ns, raw in reader.messages(
                connections=connections
            ):
                sequence += 1
                message = reader.deserialize(raw, connection.msgtype)
                stamp_s = _message_stamp_s(message)
                if not _in_station_window(stamp_s, record):
                    continue
                detection = detect_dual_tag(
                    _image_message_to_gray(message), detector, roi_xyxy
                )
                corners = None if detection is None else detection.corners_px
                if corners is None:
                    continue
                odom, delta_s = _nearest_pose(odometry, stamp_s)
                frames.append(
                    FrameObservation(
                        frame_key=f"{record.station_id}/{sequence:09d}",
                        station_id=record.station_id,
                        board_instance_id=record.board_instance_id,
                        map_ins=Transform(odom.matrix, "ins_map", "ins_link"),
                        camera_tag0=None,
                        corners_px=corners,
                        odom_dt_ms=delta_s * 1000.0,
                        basic_valid=True,
                        station_gate_pass=True,
                    )
                )
    if not frames:
        raise ValueError("six frozen station bags contain no usable dual-Tag frames")
    fingerprint = machine_config_fingerprint(config_dir)
    return CalibrationProblem(
        tuple(frames),
        Transform(nominal, "ins_link", "left_camera"),
        geometry,
        tag_size_m,
        camera,
        fingerprint,
        (
            tag_geometry_run_config.reuse_initial_path
            if tag_geometry_run_config.initialization_mode == "reuse"
            else None
        ),
        geometry_source_sha256,
    )


def machine_config_fingerprint(machine_config_dir: Path) -> str:
    config_dir = Path(machine_config_dir)
    runtime_path = config_dir / "tf_static_golfC_param.yaml"
    camera_path = config_dir / "camera_info"
    geometry_path = config_dir / "tag0_tag1_extrinsic.yaml"
    geometry_metadata = _single_yaml(geometry_path)
    return sha256_bytes(
        canonical_json_bytes(
            {
                runtime_path.name: sha256_file(runtime_path),
                camera_path.name: sha256_file(camera_path),
                "tag_geometry_metadata": {
                    "tag_size_m": float(geometry_metadata["tag_size_m"]),
                    "tag0_id": int(geometry_metadata.get("tag0_id", 0)),
                    "tag1_id": int(geometry_metadata.get("tag1_id", 1)),
                },
            }
        )
    )
