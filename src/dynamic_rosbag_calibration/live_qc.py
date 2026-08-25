"""Self-contained live image QC for the online calibration process.

The pure processor owns AprilTag detection and reuses the same SQPnP/IPPE gate
as frozen-bag replay.  The bounded worker deliberately drops stale QC jobs
under load; rosbag recording runs in a separate process and is never coupled to
this queue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Generic, TypeVar

import cv2
import numpy as np
import yaml

from .calibration_types import CameraModel, Transform
from .config import LiveQcConfig
from .models import VisionQualitySample
from .pnp_prepare import solve_single_tag_pnp


_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


@dataclass(frozen=True)
class LiveQcJob:
    station_id: str
    image_message: object
    image_stamp_s: float
    odom_stamp_s: float | None
    odom_valid: bool
    ins_quality_good: bool


@dataclass(frozen=True)
class LiveQcResult:
    station_id: str | None
    image_stamp_s: float
    sample: VisionQualitySample
    corners_px: np.ndarray | None
    camera_tag0: Transform | None
    dual_reprojection_rms_px: float | None
    reasons: tuple[str, ...]

    def mirror_dict(self) -> dict[str, object]:
        sample = self.sample
        return {
            "schema_version": 2,
            "station_id": self.station_id,
            "image_stamp_s": self.image_stamp_s,
            "both_tags_detected": sample.both_tags_detected,
            "f0_valid": sample.f0_valid,
            "training_gate_pass": sample.training_gate_pass,
            "camera_tag0_tz_m": sample.camera_tag0_tz_m,
            "bearing_x_deg": sample.bearing_x_deg,
            "minimum_tag_edge_px": sample.minimum_tag_edge_px,
            "minimum_margin_px": sample.minimum_margin_px,
            "odom_association_dt_ms": sample.odom_association_dt_ms,
            "dual_reprojection_rms_px": self.dual_reprojection_rms_px,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DetectedDualTag:
    corners_px: np.ndarray
    minimum_tag_edge_px: float
    minimum_margin_px: float


@dataclass(frozen=True)
class WorkerOutcome(Generic[_Output]):
    input_item: object
    value: _Output | None
    error: str | None


class LatestItemWorker(Generic[_Input, _Output]):
    """One background worker with a bounded latest-item pending queue."""

    def __init__(
        self,
        processor: Callable[[_Input], _Output],
        *,
        capacity: int,
        name: str = "fae-live-qc",
    ) -> None:
        if capacity <= 0:
            raise ValueError("worker capacity must be positive")
        self._processor = processor
        self._capacity = capacity
        self._condition = threading.Condition()
        self._pending: deque[_Input] = deque()
        self._results: deque[WorkerOutcome[_Output]] = deque()
        self._processing = False
        self._closing = False
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._failures = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=False)
        self._thread.start()

    def submit(self, item: _Input) -> bool:
        with self._condition:
            if self._closing:
                return False
            self._submitted += 1
            if len(self._pending) >= self._capacity:
                self._pending.popleft()
                self._dropped += 1
            self._pending.append(item)
            self._condition.notify_all()
            return True

    def drain(self) -> tuple[WorkerOutcome[_Output], ...]:
        with self._condition:
            values = tuple(self._results)
            self._results.clear()
            return values

    def flush(self, timeout_s: float) -> bool:
        if timeout_s <= 0.0:
            raise ValueError("flush timeout must be positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._pending or self._processing:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def status_dict(self) -> dict[str, object]:
        with self._condition:
            return {
                "qc_samples_submitted": self._submitted,
                "qc_samples_processed": self._processed,
                "qc_samples_dropped": self._dropped,
                "qc_processing_failures": self._failures,
                "qc_samples_pending": len(self._pending),
                "processing": self._processing,
            }

    def close(self, timeout_s: float) -> bool:
        flushed = self.flush(timeout_s)
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout_s)
        return flushed and not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closing:
                    self._condition.wait()
                if self._closing and not self._pending:
                    return
                item = self._pending.popleft()
                self._processing = True
            value: _Output | None = None
            error: str | None = None
            try:
                value = self._processor(item)
            except Exception as exception:  # fail closed and surface via status/log
                error = f"{type(exception).__name__}: {exception}"
            with self._condition:
                self._processed += 1
                if error is not None:
                    self._failures += 1
                self._results.append(WorkerOutcome(item, value, error))
                self._processing = False
                self._condition.notify_all()


class NearestStampBuffer:
    def __init__(self, capacity: int = 512) -> None:
        if capacity <= 0:
            raise ValueError("stamp buffer capacity must be positive")
        self._values: deque[tuple[float, bool]] = deque(maxlen=capacity)

    def append(self, stamp_s: float, valid: bool) -> None:
        value = float(stamp_s)
        if not math.isfinite(value):
            return
        self._values.append((value, bool(valid)))

    def nearest(self, stamp_s: float) -> tuple[float | None, bool]:
        if not self._values:
            return None, False
        target = float(stamp_s)
        stamp, valid = min(self._values, key=lambda item: abs(item[0] - target))
        return stamp, valid


def image_message_to_gray(message: Any) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    encoding = str(message.encoding).lower()
    channels = {
        "mono8": 1,
        "8uc1": 1,
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
    }.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding: {message.encoding}")
    packed_width = width * channels
    step = int(getattr(message, "step", packed_width))
    if height <= 0 or width <= 0 or step < packed_width:
        raise ValueError("invalid image dimensions or row step")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size != height * step:
        raise ValueError("image data length does not match height and step")
    rows = raw.reshape(height, step)
    packed = rows[:, :packed_width]
    if channels == 1:
        return np.ascontiguousarray(packed.reshape(height, width))
    image = np.ascontiguousarray(packed.reshape(height, width, channels))
    conversion = {
        "bgr8": cv2.COLOR_BGR2GRAY,
        "rgb8": cv2.COLOR_RGB2GRAY,
        "bgra8": cv2.COLOR_BGRA2GRAY,
        "rgba8": cv2.COLOR_RGBA2GRAY,
    }[encoding]
    return cv2.cvtColor(image, conversion)


def detect_dual_tag(
    gray: np.ndarray,
    detector: object,
    roi_xyxy: tuple[int, int, int, int],
) -> DetectedDualTag | None:
    """Select the highest-margin Tag 0 and Tag 1 inside one frozen ROI."""
    image = np.asarray(gray)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("dual-Tag detection requires a mono8 image")
    height, width = image.shape
    configured_x0, configured_y0, configured_x1, configured_y1 = roi_xyxy
    x0 = max(0, configured_x0)
    y0 = max(0, configured_y0)
    x1 = min(width, configured_x1)
    y1 = min(height, configured_y1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("approved AprilTag ROI does not overlap the image")
    best: dict[int, tuple[float, np.ndarray]] = {}
    for detection in detector.detect(
        image[y0:y1, x0:x1], estimate_tag_pose=False
    ):
        tag_id = int(detection.tag_id)
        if tag_id not in (0, 1):
            continue
        confidence = float(getattr(detection, "decision_margin", 1.0))
        if not math.isfinite(confidence):
            continue
        corners = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
        corners += np.array([x0, y0], dtype=np.float64)
        if not np.all(np.isfinite(corners)):
            continue
        if tag_id not in best or confidence > best[tag_id][0]:
            best[tag_id] = (confidence, corners)
    if 0 not in best or 1 not in best:
        return None
    per_tag = (best[0][1], best[1][1])
    edge_lengths: list[float] = []
    margins: list[float] = []
    for corners in per_tag:
        edge_lengths.extend(
            np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        )
        margins.extend(corners[:, 0] - x0)
        margins.extend(x1 - corners[:, 0])
        margins.extend(corners[:, 1] - y0)
        margins.extend(y1 - corners[:, 1])
    return DetectedDualTag(
        corners_px=np.vstack(per_tag),
        minimum_tag_edge_px=float(np.min(edge_lengths)),
        minimum_margin_px=float(np.min(margins)),
    )


def _single_yaml(path: Path) -> dict[str, Any]:
    documents = [
        item for item in yaml.safe_load_all(path.read_text(encoding="utf-8")) if item
    ]
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise ValueError(f"{path} must contain exactly one YAML mapping")
    return documents[0]


def load_live_qc_profile(
    machine_config_dir: Path,
) -> tuple[CameraModel, float]:
    config_dir = Path(machine_config_dir)
    camera_payload = _single_yaml(config_dir / "camera_info")
    camera = CameraModel(
        np.asarray(camera_payload["data"], dtype=np.float64).reshape(3, 3),
        np.asarray(
            camera_payload.get("distortion_coefficients", {}).get(
                "data", np.zeros(5)
            ),
            dtype=np.float64,
        ),
    )
    geometry_payload = _single_yaml(config_dir / "tag0_tag1_extrinsic.yaml")
    return camera, float(geometry_payload["tag_size_m"])


def create_apriltag_detector(config: LiveQcConfig) -> object:
    try:
        from pyapriltags import Detector
    except ImportError as error:
        raise RuntimeError("internal live QC requires pyapriltags") from error
    return Detector(
        families="tag36h11",
        nthreads=config.detector_nthreads,
        quad_decimate=config.quad_decimate,
        quad_sigma=config.quad_sigma,
        refine_edges=config.refine_edges,
        decode_sharpening=config.decode_sharpening,
        debug=0,
    )


class LiveQcProcessor:
    def __init__(
        self,
        camera: CameraModel,
        tag_size_m: float,
        detector: object,
        config: LiveQcConfig,
    ) -> None:
        self.camera = camera
        self.tag_size_m = float(tag_size_m)
        self.detector = detector
        self.config = config

    @classmethod
    def from_machine_config(
        cls, machine_config_dir: Path, config: LiveQcConfig
    ) -> "LiveQcProcessor":
        camera, tag_size_m = load_live_qc_profile(machine_config_dir)
        return cls(
            camera,
            tag_size_m,
            create_apriltag_detector(config),
            config,
        )

    def process_job(self, job: LiveQcJob) -> LiveQcResult:
        return self.process_gray(
            image_message_to_gray(job.image_message),
            station_id=job.station_id,
            image_stamp_s=job.image_stamp_s,
            odom_stamp_s=job.odom_stamp_s,
            odom_valid=job.odom_valid,
            ins_quality_good=job.ins_quality_good,
        )

    def process_gray(
        self,
        gray: np.ndarray,
        *,
        station_id: str | None = None,
        image_stamp_s: float,
        odom_stamp_s: float | None,
        odom_valid: bool,
        ins_quality_good: bool,
    ) -> LiveQcResult:
        image = np.asarray(gray)
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("live QC requires a mono8 image")
        detected = detect_dual_tag(
            image,
            self.detector,
            self.config.roi_xyxy,
        )
        odom_dt_ms = (
            abs(float(image_stamp_s) - float(odom_stamp_s)) * 1000.0
            if odom_stamp_s is not None
            else 1.0e9
        )
        if detected is None:
            return self._invalid_result(
                image_stamp_s,
                odom_dt_ms,
                ("CAL-E-LIVE-QC-DUAL-TAG",),
                station_id=station_id,
            )
        corners = detected.corners_px
        tag0 = solve_single_tag_pnp(corners[:4], self.camera, self.tag_size_m)
        tag1 = solve_single_tag_pnp(corners[4:], self.camera, self.tag_size_m)
        reasons: list[str] = []
        if tag0.selected is None:
            reasons.append("CAL-E-PNP-TAG0-SOLVER")
        if tag1.selected is None:
            reasons.append("CAL-E-PNP-TAG1-SOLVER")
        if odom_stamp_s is None or not odom_valid:
            reasons.append("CAL-E-LIVE-QC-ODOM")
        if not ins_quality_good:
            reasons.append("CAL-E-LIVE-QC-INS-QUALITY")
        f0_valid = (
            tag0.selected is not None
            and tag1.selected is not None
            and odom_stamp_s is not None
            and odom_valid
            and ins_quality_good
        )
        camera_tag0 = (
            None
            if tag0.selected is None
            else Transform(tag0.selected.matrix, "left_camera", "tag0")
        )
        if camera_tag0 is None:
            distance_m = 0.0
            bearing_deg = 0.0
        else:
            translation = camera_tag0.matrix[:3, 3]
            distance_m = float(translation[2])
            bearing_deg = math.degrees(math.atan2(float(translation[0]), distance_m))
        sample = VisionQualitySample(
            stamp_s=float(image_stamp_s),
            both_tags_detected=True,
            f0_valid=f0_valid,
            training_gate_pass=None,
            camera_tag0_tz_m=distance_m,
            bearing_x_deg=bearing_deg,
            minimum_tag_edge_px=detected.minimum_tag_edge_px,
            minimum_margin_px=detected.minimum_margin_px,
            odom_association_dt_ms=odom_dt_ms,
            diagnostic_reasons=tuple(dict.fromkeys(reasons)),
            dual_reprojection_rms_px=None,
        )
        return LiveQcResult(
            station_id,
            float(image_stamp_s),
            sample,
            corners,
            camera_tag0,
            None,
            tuple(dict.fromkeys(reasons)),
        )

    def _invalid_result(
        self,
        image_stamp_s: float,
        odom_dt_ms: float,
        reasons: tuple[str, ...],
        *,
        station_id: str | None,
    ) -> LiveQcResult:
        sample = VisionQualitySample(
            stamp_s=float(image_stamp_s),
            both_tags_detected=False,
            f0_valid=False,
            training_gate_pass=None,
            camera_tag0_tz_m=0.0,
            bearing_x_deg=0.0,
            minimum_tag_edge_px=0.0,
            minimum_margin_px=0.0,
            odom_association_dt_ms=float(odom_dt_ms),
            diagnostic_reasons=reasons,
            dual_reprojection_rms_px=None,
        )
        return LiveQcResult(
            station_id, float(image_stamp_s), sample, None, None, None, reasons
        )
