"""Freeze and audit camera/TF inputs recorded with calibration evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .calibration_types import CameraModel, Transform, readonly_float_array
from .io import build_raw_data_identity, canonical_json_bytes, sha256_bytes


SourceGroup = Literal["calibration", "static", "dynamic_micro", "dynamic_true"]
REQUIRED_SOURCE_GROUPS = frozenset(
    {"calibration", "static", "dynamic_micro", "dynamic_true"}
)
INS_LIDAR_EDGE = ("ins_link", "lidar_link")
LIDAR_CAMERA_EDGE = ("lidar_link", "camera_infra1_optical_frame")
REQUIRED_TF_EDGES = (INS_LIDAR_EDGE, LIDAR_CAMERA_EDGE)
VALIDATION_GROUPS: dict[str, SourceGroup] = {
    "135027": "static",
    "135110": "static",
    "135148": "static",
    "135220": "static",
    "135244": "static",
    "135334": "static",
    "135307": "dynamic_micro",
    "135505": "dynamic_true",
}


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    absolute_path: str
    relative_path: str
    group: SourceGroup
    size_bytes: int
    sha256: str
    start_time_s: float
    end_time_s: float

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if not Path(self.absolute_path).is_absolute():
            raise ValueError("absolute_path must be absolute")
        if Path(self.relative_path).is_absolute() or not self.relative_path:
            raise ValueError("relative_path must be non-empty and relative")
        if self.group not in REQUIRED_SOURCE_GROUPS:
            raise ValueError(f"unsupported source group: {self.group}")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if len(self.sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.sha256.lower()
        ):
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        if (
            not np.isfinite(self.start_time_s)
            or not np.isfinite(self.end_time_s)
            or self.end_time_s < self.start_time_s
        ):
            raise ValueError("source time range must be finite and ordered")


@dataclass(frozen=True)
class FrozenRuntimeSnapshot:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    T_ins_lidar: np.ndarray
    T_lidar_camera: np.ndarray
    T_ins_camera: np.ndarray
    tag_size_m: float
    sources: tuple[SourceRecord, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        camera = CameraModel(self.camera_matrix, self.distortion_coefficients)
        distortion = readonly_float_array(
            camera.distortion, (5,), "distortion_coefficients"
        )
        ins_lidar = Transform(self.T_ins_lidar, *INS_LIDAR_EDGE)
        lidar_camera = Transform(self.T_lidar_camera, *LIDAR_CAMERA_EDGE)
        ins_camera = Transform(
            self.T_ins_camera,
            "ins_link",
            "camera_infra1_optical_frame",
        )
        expected = ins_lidar.matrix @ lidar_camera.matrix
        if not np.allclose(ins_camera.matrix, expected, rtol=0.0, atol=1.0e-12):
            raise ValueError("T_ins_camera does not equal T_ins_lidar @ T_lidar_camera")
        if not np.isfinite(self.tag_size_m) or self.tag_size_m <= 0.0:
            raise ValueError("tag_size_m must be finite and positive")
        if len(self.identity_sha256) != 64:
            raise ValueError("identity_sha256 must contain 64 characters")
        object.__setattr__(self, "camera_matrix", camera.matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)
        object.__setattr__(self, "T_ins_lidar", ins_lidar.matrix)
        object.__setattr__(self, "T_lidar_camera", lidar_camera.matrix)
        object.__setattr__(self, "T_ins_camera", ins_camera.matrix)
        object.__setattr__(self, "sources", tuple(self.sources))


def _validated_tf_tree(
    value: Mapping[tuple[str, str], object], sample_index: int
) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for parent, child in REQUIRED_TF_EDGES:
        edge = (parent, child)
        if edge not in value:
            raise ValueError(
                f"tf_static sample {sample_index} is missing {parent} <- {child}"
            )
        result[edge] = Transform(value[edge], parent, child).matrix
    return result


def build_frozen_runtime(
    *,
    intrinsic_samples: Sequence[object],
    tf_static_samples: Sequence[Mapping[tuple[str, str], object]],
    source_records: Sequence[SourceRecord],
    tag_size_m: float,
) -> FrozenRuntimeSnapshot:
    if not intrinsic_samples:
        raise ValueError("no camera intrinsic samples were recorded")
    if not tf_static_samples:
        raise ValueError("no /tf_static samples were recorded")
    cameras = tuple(
        CameraModel(value, np.zeros(5, dtype=np.float64)).matrix
        for value in intrinsic_samples
    )
    camera = cameras[0]
    if any(
        not np.allclose(value, camera, rtol=0.0, atol=1.0e-12)
        for value in cameras[1:]
    ):
        raise ValueError("inconsistent camera intrinsics across rosbags")
    trees = tuple(
        _validated_tf_tree(value, index)
        for index, value in enumerate(tf_static_samples)
    )
    first_tree = trees[0]
    for tree in trees[1:]:
        for edge in REQUIRED_TF_EDGES:
            if not np.allclose(
                tree[edge], first_tree[edge], rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    "inconsistent /tf_static across rosbags for "
                    f"{edge[0]} <- {edge[1]}"
                )
    sources = tuple(sorted(source_records, key=lambda item: item.source_id))
    source_ids = [item.source_id for item in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source_id in frozen runtime inventory")
    groups = {item.group for item in sources}
    missing_groups = sorted(REQUIRED_SOURCE_GROUPS - groups)
    if missing_groups:
        raise ValueError(f"missing source groups: {', '.join(missing_groups)}")
    ins_lidar = first_tree[INS_LIDAR_EDGE]
    lidar_camera = first_tree[LIDAR_CAMERA_EDGE]
    ins_camera = ins_lidar @ lidar_camera
    identity_payload = {
        "schema_version": 1,
        "camera_matrix": camera.tolist(),
        "distortion_coefficients": [0.0] * 5,
        "T_ins_lidar": ins_lidar.tolist(),
        "T_lidar_camera": lidar_camera.tolist(),
        "T_ins_camera": ins_camera.tolist(),
        "tag_size_m": float(tag_size_m),
        "sources": [asdict(item) for item in sources],
    }
    identity_sha256 = sha256_bytes(canonical_json_bytes(identity_payload))
    return FrozenRuntimeSnapshot(
        camera_matrix=camera,
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        T_ins_lidar=ins_lidar,
        T_lidar_camera=lidar_camera,
        T_ins_camera=ins_camera,
        tag_size_m=float(tag_size_m),
        sources=sources,
        identity_sha256=identity_sha256,
    )


def _transform_message_matrix(transform: Any) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_quat(
        [
            float(transform.rotation.x),
            float(transform.rotation.y),
            float(transform.rotation.z),
            float(transform.rotation.w),
        ]
    ).as_matrix()
    value[:3, 3] = [
        float(transform.translation.x),
        float(transform.translation.y),
        float(transform.translation.z),
    ]
    return value


def _read_bag_runtime_samples(
    bag_path: Path,
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], float, float]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise RuntimeError("frozen runtime extraction requires rosbags") from error
    intrinsic_samples: list[np.ndarray] = []
    tf_tree: dict[tuple[str, str], np.ndarray] = {}
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        selected = [
            connection
            for connection in reader.connections
            if connection.topic in {"/camera/left/intrinsic_matrix", "/tf_static"}
        ]
        for connection, _timestamp_ns, raw in reader.messages(connections=selected):
            message = reader.deserialize(raw, connection.msgtype)
            if connection.topic == "/camera/left/intrinsic_matrix":
                intrinsic_samples.append(
                    np.asarray(message.data, dtype=np.float64).reshape(3, 3)
                )
                continue
            for stamped in message.transforms:
                edge = (str(stamped.header.frame_id), str(stamped.child_frame_id))
                if edge not in REQUIRED_TF_EDGES:
                    continue
                matrix = _transform_message_matrix(stamped.transform)
                if edge in tf_tree and not np.allclose(
                    tf_tree[edge], matrix, rtol=0.0, atol=1.0e-12
                ):
                    raise ValueError(
                        f"{bag_path} contains conflicting {edge[0]} <- {edge[1]}"
                    )
                tf_tree[edge] = matrix
        start_time_s = float(reader.start_time) / 1.0e9
        end_time_s = float(reader.end_time) / 1.0e9
    if len(intrinsic_samples) != 1:
        raise ValueError(
            f"{bag_path} must contain exactly one camera intrinsic matrix"
        )
    return intrinsic_samples[0], tf_tree, start_time_s, end_time_s


def _calibration_source_id(path: Path) -> str:
    source_id = path.name.removeprefix("rosbag2_")
    if not source_id.startswith("s"):
        raise ValueError(f"cannot classify calibration rosbag {path}")
    return source_id


def _validation_source_id(path: Path) -> str:
    source_id = path.name.rsplit("_", 1)[-1]
    if source_id not in VALIDATION_GROUPS:
        raise ValueError(f"cannot classify validation rosbag {path}")
    return source_id


def read_frozen_runtime(
    calibration_bags: Sequence[Path],
    validation_bags: Sequence[Path],
    *,
    tag_size_m: float = 0.25,
    evidence_root: Path | None = None,
    sample_reader: Callable[
        [Path], tuple[np.ndarray, dict[tuple[str, str], np.ndarray], float, float]
    ] = _read_bag_runtime_samples,
    identity_builder: Callable[[Path], dict[str, Any]] = build_raw_data_identity,
) -> tuple[FrozenRuntimeSnapshot, dict[str, Any]]:
    calibration_paths = tuple(Path(value).resolve() for value in calibration_bags)
    validation_paths = tuple(Path(value).resolve() for value in validation_bags)
    all_paths = calibration_paths + validation_paths
    if not calibration_paths or not validation_paths:
        raise ValueError("calibration and validation rosbag lists must be non-empty")
    if any(not value.is_dir() for value in all_paths):
        raise ValueError("every rosbag path must be an existing directory")
    root = (
        Path(evidence_root).resolve()
        if evidence_root is not None
        else Path(os.path.commonpath([str(value) for value in all_paths]))
    )
    intrinsic_samples: list[np.ndarray] = []
    tf_static_samples: list[dict[tuple[str, str], np.ndarray]] = []
    sources: list[SourceRecord] = []
    specifications = [
        (path, _calibration_source_id(path), "calibration")
        for path in calibration_paths
    ] + [
        (path, _validation_source_id(path), VALIDATION_GROUPS[_validation_source_id(path)])
        for path in validation_paths
    ]
    for path, source_id, group in specifications:
        intrinsic, tf_tree, start_time_s, end_time_s = sample_reader(path)
        intrinsic_samples.append(intrinsic)
        tf_static_samples.append(tf_tree)
        identity = identity_builder(path)
        sources.append(
            SourceRecord(
                source_id=source_id,
                absolute_path=str(path),
                relative_path=path.relative_to(root).as_posix(),
                group=group,
                size_bytes=sum(
                    int(item["size_bytes"]) for item in identity["files"]
                ),
                sha256=str(identity["identity_sha256"]),
                start_time_s=start_time_s,
                end_time_s=end_time_s,
            )
        )
    snapshot = build_frozen_runtime(
        intrinsic_samples=intrinsic_samples,
        tf_static_samples=tf_static_samples,
        source_records=sources,
        tag_size_m=tag_size_m,
    )
    audit = {
        "schema_version": 1,
        "identity_sha256": snapshot.identity_sha256,
        "evidence_root": str(root),
        "source_record_time_basis": "rosbag_record_timestamp",
        "camera_matrix": snapshot.camera_matrix.tolist(),
        "distortion_coefficients": snapshot.distortion_coefficients.tolist(),
        "distortion_provenance": (
            "historical camera_info contains only the 3x3 intrinsic matrix; "
            "the existing five-zero fallback is frozen explicitly"
        ),
        "T_ins_lidar": snapshot.T_ins_lidar.tolist(),
        "T_lidar_camera": snapshot.T_lidar_camera.tolist(),
        "T_ins_camera": snapshot.T_ins_camera.tolist(),
        "tag_size_m": snapshot.tag_size_m,
        "sources": [asdict(item) for item in snapshot.sources],
    }
    return snapshot, audit
