import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "src/dynamic_rosbag_calibration"


def test_standalone_package_does_not_import_original_tool() -> None:
    violations = []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "fae_calibration"
            ):
                violations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("fae_calibration"):
                        violations.append(f"{path.name}:{node.lineno}")
    assert violations == []
