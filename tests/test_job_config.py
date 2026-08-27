from pathlib import Path

import pytest
import yaml

from dynamic_rosbag_calibration.job_config import load_job_config


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = PACKAGE_ROOT / "config/calibration_job.yaml"


def _write_test_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    machine_dir = tmp_path / "machine"
    message_dir = tmp_path / "novatel_msgs"
    machine_dir.mkdir()
    message_dir.mkdir()

    roles = {
        "calibration": ("cal_a", "cal_b"),
        "validation": ("val_a",),
        "displacement_evaluation": (),
        "excluded": (),
    }
    rosbags = {}
    for role, identifiers in roles.items():
        entries = []
        for bag_id in identifiers:
            bag_path = tmp_path / bag_id
            bag_path.mkdir()
            (bag_path / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
            entries.append({"id": bag_id, "path": str(bag_path)})
        rosbags[role] = entries

    payload["job"]["id"] = "unit_test"
    payload["job"]["dataset_date"] = "20260821"
    payload["runtime"]["novatel_message_dir"] = str(message_dir)
    payload["runtime"]["output_dir"] = "outputs/result"
    payload["machine"]["config_dir"] = str(machine_dir)
    payload["rosbags"] = rosbags
    payload["calibration"]["quality_gate"]["position_limit_m"] = 0.07
    payload["calibration"]["quality_gate"]["yaw_limit_deg"] = 1.2
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_loads_one_config_and_preserves_adjustable_quality_gates(tmp_path: Path) -> None:
    path = _write_test_config(tmp_path)

    job = load_job_config(path)

    assert tuple(job.bag_paths) == ("cal_a", "cal_b", "val_a")
    assert job.output_dir == tmp_path / "outputs/result"
    with job.prepare_runtime() as prepared:
        assert prepared.online.calibration.compliance["position_limit_m"] == 0.07
        assert prepared.online.calibration.compliance["yaw_limit_deg"] == 1.2
        assert prepared.experiment.sampling.minimum_bag_count == 2
        assert prepared.manifest["datasets"]["20260821"]["calibration_bags"] == [
            "cal_a",
            "cal_b",
        ]


def test_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_job_config(path)


def test_rejects_missing_rosbag_metadata(tmp_path: Path) -> None:
    path = _write_test_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["rosbags"]["validation"][0]["path"] = str(tmp_path / "missing")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata.yaml is missing"):
        load_job_config(path)


def test_rejects_unknown_public_fields(tmp_path: Path) -> None:
    path = _write_test_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["calibration"]["quality_gate"]["silent_fallback"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        load_job_config(path)
