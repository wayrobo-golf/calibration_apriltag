import pytest

from dynamic_rosbag_calibration.pipeline import (
    _experiment_completion_status,
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
