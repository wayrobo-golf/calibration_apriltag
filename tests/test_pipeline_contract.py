import pytest

from dynamic_rosbag_calibration.pipeline import (
    _experiment_completion_status,
    _write_day_report,
    build_parser,
)


def test_public_cli_accepts_only_the_single_config_argument() -> None:
    args = build_parser().parse_args(["--config", "/tmp/job.yaml"])
    assert str(args.config) == "/tmp/job.yaml"

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--config", "/tmp/job.yaml", "--manifest", "/tmp/manifest.yaml"]
        )


@pytest.mark.parametrize(
    ("day_status", "expected"),
    [
        ("INVENTORY_ONLY", "INVENTORY_ONLY"),
        ("INVENTORY_WITH_ERRORS", "INVENTORY_WITH_ERRORS"),
        ("INVENTORY_FAILED", "INCOMPLETE"),
    ],
)
def test_inventory_failures_are_visible_in_root_status(
    day_status: str,
    expected: str,
) -> None:
    assert _experiment_completion_status(
        [{"status": day_status}],
        inventory_only=True,
    ) == expected


def test_day_report_renders_bag_and_candidate_failures(tmp_path) -> None:
    report_path = _write_day_report(
        tmp_path,
        date="20260827",
        source_present=True,
        inventory=(
            {
                "bag_id": "102016",
                "frame_count": 1,
                "base_valid_frame_count": 1,
                "within_8m_frame_count": 1,
                "within_8m_depth_min_m": 1.0,
                "within_8m_depth_max_m": 1.0,
                "within_8m_bearing_median_deg": 0.0,
            },
        ),
        plan=(),
        pairwise_summary=(),
        parameters=(),
        holdouts=(),
        failures=(
            {"bag_id": "102016", "error": "invalid margin"},
            {"candidate_id": "c04", "error": "G2 solver failure"},
        ),
        validation_failures=(),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- bag `102016`: invalid margin" in report
    assert "- candidate `c04`: G2 solver failure" in report
