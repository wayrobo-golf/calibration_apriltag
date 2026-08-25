"""Standalone dynamic rosbag calibration library."""

from .job_config import CalibrationJobConfig, RosbagInput, load_job_config

__all__ = [
    "CalibrationJobConfig",
    "RosbagInput",
    "load_job_config",
    "run_calibration",
]

__version__ = "0.1.0"


def run_calibration(config_path: str) -> int:
    """Run a calibration job synchronously and return its process-style status."""
    from .pipeline import run_from_config

    return run_from_config(config_path)
