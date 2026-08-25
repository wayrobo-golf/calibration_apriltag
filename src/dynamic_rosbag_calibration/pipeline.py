#!/usr/bin/env python3
"""Run two-bag dynamic calibration combinations with same-day-only holdouts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import yaml

from dynamic_rosbag_calibration.dynamic_combination_analysis import (
    apply_quality_gate_preference,
    enumerate_feasible_dynamic_combinations,
    pairwise_candidate_differences,
    rank_candidates_by_stability_and_holdout,
    summarize_transform_stability,
)
from dynamic_rosbag_calibration.dynamic_experiment import solve_safe_dynamic_candidate
from dynamic_rosbag_calibration.dynamic_experiment_types import (
    DynamicFrameEvidence,
    StructuralState,
)
from dynamic_rosbag_calibration.dynamic_problem import (
    build_calibration_problem_from_dynamic_frames,
    read_dynamic_bag_evidence,
)
from dynamic_rosbag_calibration.dynamic_segments import classify_dynamic_frames
from dynamic_rosbag_calibration.interactive_bag_report import (
    write_interactive_bag_index,
    write_interactive_bag_report,
)
from dynamic_rosbag_calibration.io import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from dynamic_rosbag_calibration.job_config import load_job_config
from dynamic_rosbag_calibration.live_qc import load_live_qc_profile
from dynamic_rosbag_calibration.safe_dynamic_validation import (
    evaluate_safe_dynamic_candidates,
)
from dynamic_rosbag_calibration.same_day_dynamic_analysis import (
    retain_base_valid_frames_within_depth,
    select_representative_validation_bag,
    select_same_day_combination_panel,
    summarize_candidate_holdouts,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
TRANSFORM_FIELDS = (
    "T_tag0_tag1_calibrated",
    "T_ins_map_tag0_calibrated",
    "T_ins_camera_calibrated",
)
CANDIDATE_ARTIFACTS = (
    "recovered_candidate.yaml",
    "recovery_audit.json",
    "experiment_metadata.json",
)


def _clear_previous_completion_artifacts(
    output_dir: Path,
    dates: Sequence[str],
) -> None:
    """Remove only pipeline-owned completion claims before starting a new run."""
    for name in ("FINDINGS.md", "REPORT.md", "summary.json"):
        path = output_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()
    for date in dates:
        day_dir = output_dir / str(date)
        report = day_dir / "REPORT.md"
        if report.is_file() or report.is_symlink():
            report.unlink()
        validation = day_dir / "validation_same_day"
        if validation.is_symlink() or validation.is_file():
            validation.unlink()
        elif validation.is_dir():
            shutil.rmtree(validation)


def _day_completion_status(
    *,
    selected_count: int,
    successful_count: int,
    validated_count: int,
    validation_skipped: bool,
    failure_count: int,
    validation_failure_count: int,
    exclusion_count: int,
) -> str:
    if successful_count != selected_count or failure_count:
        return "INCOMPLETE_SOLVE"
    if validation_skipped:
        return "CALIBRATION_ONLY"
    if validated_count != successful_count or validation_failure_count:
        return "INCOMPLETE_VALIDATION"
    return "COMPLETE_WITH_EXCLUSIONS" if exclusion_count else "COMPLETE"


def _can_write_completed_findings(
    summaries: Sequence[Mapping[str, object]],
    *,
    inventory_only: bool,
) -> bool:
    if inventory_only:
        return False
    statuses = {str(row["status"]) for row in summaries}
    allowed = {
        "COMPLETE",
        "COMPLETE_WITH_EXCLUSIONS",
        "SOURCE_DATA_MISSING",
        "NO_CONFIGURED_DYNAMIC_BAGS",
    }
    return bool(statuses.intersection({"COMPLETE", "COMPLETE_WITH_EXCLUSIONS"})) and statuses.issubset(allowed)


def _experiment_completion_status(
    summaries: Sequence[Mapping[str, object]],
    *,
    inventory_only: bool,
) -> str:
    if inventory_only:
        statuses = {str(row["status"]) for row in summaries}
        if "INVENTORY_FAILED" in statuses:
            return "INCOMPLETE"
        if "INVENTORY_WITH_ERRORS" in statuses:
            return "INVENTORY_WITH_ERRORS"
        return "INVENTORY_ONLY"
    if not _can_write_completed_findings(summaries, inventory_only=False):
        return "INCOMPLETE"
    if any(str(row["status"]) == "COMPLETE_WITH_EXCLUSIONS" for row in summaries):
        return "COMPLETE_WITH_EXCLUSIONS"
    return "COMPLETE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def _progress(stage: str, **details: object) -> None:
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)


def _load_manifest(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("same-day manifest must be a schema-v1 mapping")
    datasets = payload.get("datasets")
    sampling = payload.get("two_bag_sampling")
    coverage = payload.get("two_bag_coverage")
    if (
        not isinstance(datasets, dict)
        or not isinstance(sampling, dict)
        or not isinstance(coverage, dict)
    ):
        raise ValueError(
            "same-day manifest requires datasets, two_bag_sampling, and two_bag_coverage"
        )
    dates = tuple(sorted(str(value) for value in datasets))
    if not dates or any(len(date) != 8 or not date.isdigit() for date in dates):
        raise ValueError("manifest datasets must use YYYYMMDD date keys")
    for date in dates:
        spec = datasets[date]
        required_dataset_fields = {
            "bags",
            "excluded",
            "maximum_combinations",
        }
        optional_dataset_fields = {
            "calibration_bags",
            "displacement_evaluation_bags",
        }
        if (
            not isinstance(spec, dict)
            or not required_dataset_fields.issubset(spec)
            or not set(spec).issubset(required_dataset_fields | optional_dataset_fields)
        ):
            raise ValueError(f"invalid dataset manifest entry: {date}")
        bags = spec["bags"]
        if not isinstance(bags, list) or len(set(str(value) for value in bags)) != len(bags):
            raise ValueError(f"{date} bags must be a unique list")
        _dataset_bag_groups(spec)
        maximum = spec["maximum_combinations"]
        if not isinstance(maximum, int) or maximum < 0:
            raise ValueError(f"{date} maximum_combinations must be non-negative")
    required_sampling = {
        "minimum_bag_count",
        "minimum_bags_per_distance_bin",
        "minimum_total_frames",
    }
    if set(sampling) != required_sampling:
        raise ValueError("two_bag_sampling has an invalid schema")
    if int(sampling["minimum_bag_count"]) != 2:
        raise ValueError("same-day experiment must calibrate exactly two bags")
    if set(coverage) != {"minimum_bearing_span_deg"}:
        raise ValueError("two_bag_coverage has an invalid schema")
    bearing_span = float(coverage["minimum_bearing_span_deg"])
    if not np.isfinite(bearing_span) or bearing_span <= 0.0:
        raise ValueError("two-bag bearing span must be finite and positive")
    policy = payload.get(
        "calibration_policy",
        {
            "maximum_depth_m": 8.0,
            "maximum_depth_inclusive": True,
            "tag_geometry_initialization_mode": "frozen_reuse",
        },
    )
    if not isinstance(policy, dict):
        raise ValueError("calibration_policy must be a mapping")
    required_policy = {
        "maximum_depth_m",
        "maximum_depth_inclusive",
        "tag_geometry_initialization_mode",
    }
    if not required_policy.issubset(policy):
        raise ValueError(f"calibration_policy requires {sorted(required_policy)}")
    maximum_depth_m = float(policy["maximum_depth_m"])
    if not np.isfinite(maximum_depth_m) or maximum_depth_m <= 0.0:
        raise ValueError("calibration_policy.maximum_depth_m must be positive")
    if not isinstance(policy["maximum_depth_inclusive"], bool):
        raise ValueError("maximum_depth_inclusive must be boolean")
    mode = str(policy["tag_geometry_initialization_mode"])
    if mode not in {"bootstrap", "frozen_reuse"}:
        raise ValueError("tag geometry mode must be bootstrap or frozen_reuse")
    if mode == "bootstrap":
        required_bootstrap = {
            "tag_geometry_minimum_valid_stations",
            "tag_geometry_maximum_views_per_station",
            "tag_geometry_minimum_balanced_views",
        }
        if not required_bootstrap.issubset(policy):
            raise ValueError(
                f"bootstrap calibration_policy requires {sorted(required_bootstrap)}"
            )
        for field in required_bootstrap:
            value = policy[field]
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"calibration_policy.{field} must be positive integer")
    payload["calibration_policy"] = dict(policy)
    return payload


def _dataset_bag_groups(
    spec: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    all_bags = tuple(str(value) for value in spec["bags"])
    calibration_bags = tuple(
        str(value) for value in spec.get("calibration_bags", all_bags)
    )
    displacement_bags = tuple(
        str(value) for value in spec.get("displacement_evaluation_bags", ())
    )
    for name, values in (
        ("calibration_bags", calibration_bags),
        ("displacement_evaluation_bags", displacement_bags),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must be unique")
        unknown = set(values).difference(all_bags)
        if unknown:
            raise ValueError(f"{name} contains unconfigured bags: {sorted(unknown)}")
    overlap = set(calibration_bags).intersection(displacement_bags)
    if overlap:
        raise ValueError(
            "calibration and displacement-evaluation bags must be disjoint: "
            f"{sorted(overlap)}"
        )
    if len(calibration_bags) < 2:
        raise ValueError("calibration_bags must contain at least two bags")
    return all_bags, calibration_bags, displacement_bags


def _partition_validation_metrics(
    rows: Sequence[Mapping[str, object]],
    displacement_bag_ids: Sequence[str],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    displacement_ids = frozenset(str(value) for value in displacement_bag_ids)
    normal = [row for row in rows if str(row["bag_id"]) not in displacement_ids]
    displacement = [row for row in rows if str(row["bag_id"]) in displacement_ids]
    return normal, displacement


def _summarize_board_displacement(
    rows: Sequence[Mapping[str, object]],
    displacement_bag_ids: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    displacement_ids = frozenset(str(value) for value in displacement_bag_ids)
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        bag_id = str(row["bag_id"])
        if bag_id not in displacement_ids or not bool(row["primary_scope"]):
            continue
        key = (str(row["candidate_id"]), bag_id)
        grouped.setdefault(key, []).append(row)
    per_bag: list[dict[str, object]] = []
    for (candidate_id, bag_id), group in sorted(grouped.items()):
        translations = np.asarray(
            [float(row["board_displacement_m"]) for row in group],
            dtype=np.float64,
        )
        abs_yaws = np.abs(
            np.asarray(
                [float(row["board_displacement_yaw_deg"]) for row in group],
                dtype=np.float64,
            )
        )
        per_bag.append(
            {
                "candidate_id": candidate_id,
                "bag_id": bag_id,
                "frame_count": len(group),
                "signed_median_x_m": float(
                    np.median([float(row["board_displacement_x_m"]) for row in group])
                ),
                "signed_median_y_m": float(
                    np.median([float(row["board_displacement_y_m"]) for row in group])
                ),
                "signed_median_z_m": float(
                    np.median([float(row["board_displacement_z_m"]) for row in group])
                ),
                "signed_median_yaw_deg": float(
                    np.median(
                        [float(row["board_displacement_yaw_deg"]) for row in group]
                    )
                ),
                "translation_median_m": float(np.median(translations)),
                "translation_p80_m": float(np.quantile(translations, 0.80)),
                "translation_p95_m": float(np.quantile(translations, 0.95)),
                "translation_max_m": float(np.max(translations)),
                "abs_yaw_median_deg": float(np.median(abs_yaws)),
                "abs_yaw_p80_deg": float(np.quantile(abs_yaws, 0.80)),
                "abs_yaw_p95_deg": float(np.quantile(abs_yaws, 0.95)),
                "abs_yaw_max_deg": float(np.max(abs_yaws)),
            }
        )
    summaries: list[dict[str, object]] = []
    for candidate_id in sorted({str(row["candidate_id"]) for row in per_bag}):
        candidate_rows = [
            row for row in per_bag if str(row["candidate_id"]) == candidate_id
        ]
        translations = np.asarray(
            [float(row["translation_median_m"]) for row in candidate_rows]
        )
        abs_yaws = np.asarray(
            [float(row["abs_yaw_median_deg"]) for row in candidate_rows]
        )
        summaries.append(
            {
                "candidate_id": candidate_id,
                "displacement_bag_count": len(candidate_rows),
                "equal_bag_translation_median_m": float(np.median(translations)),
                "equal_bag_translation_max_m": float(np.max(translations)),
                "equal_bag_abs_yaw_median_deg": float(np.median(abs_yaws)),
                "equal_bag_abs_yaw_max_deg": float(np.max(abs_yaws)),
            }
        )
    return per_bag, summaries


def _bag_path(root: Path, date: str, bag_id: str) -> Path:
    candidates = (
        Path(root) / date / f"bag_{date}_{bag_id}",
        Path(root) / date / "2_rosbag" / f"bag_{date}_{bag_id}",
    )
    for path in candidates:
        if path.is_dir() and (path / "metadata.yaml").is_file():
            return path
    raise FileNotFoundError(f"same-day bag is missing: {candidates}")


def _read_candidate(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"candidate is not a YAML mapping: {path}")
    for field in TRANSFORM_FIELDS:
        matrix = np.asarray(payload.get(field), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"candidate has invalid {field}: {path}")
    return payload


def _read_reusable_candidate(
    candidate_dir: Path,
    solve_identity: Mapping[str, object],
) -> dict[str, object] | None:
    destination = Path(candidate_dir)
    marker_path = destination / "combination_input.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            not isinstance(marker, dict)
            or marker.get("schema_version") != 2
            or marker.get("solve_identity") != dict(solve_identity)
        ):
            return None
        hashes = marker.get("artifact_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(CANDIDATE_ARTIFACTS):
            return None
        for name in CANDIDATE_ARTIFACTS:
            path = destination / name
            if not path.is_file() or sha256_file(path) != hashes[name]:
                return None
        return _read_candidate(destination / "recovered_candidate.yaml")
    except (OSError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError):
        return None


def _publish_candidate_attempt(
    attempt_dir: Path,
    candidate_dir: Path,
    solve_identity: Mapping[str, object],
) -> None:
    source = Path(attempt_dir)
    destination = Path(candidate_dir)
    artifact_hashes: dict[str, str] = {}
    for name in CANDIDATE_ARTIFACTS:
        path = source / name
        if not path.is_file():
            raise ValueError(f"candidate attempt is missing {name}")
        artifact_hashes[name] = sha256_file(path)
    destination.mkdir(parents=True, exist_ok=True)
    for name in CANDIDATE_ARTIFACTS:
        os.replace(source / name, destination / name)
    atomic_write_json(
        destination / "combination_input.json",
        {
            "schema_version": 2,
            "solve_identity": dict(solve_identity),
            "artifact_sha256": artifact_hashes,
        },
    )
    (destination / "failure.json").unlink(missing_ok=True)


def _pipeline_software_fingerprint() -> str:
    sources = sorted(PACKAGE_ROOT.glob("*.py"))
    identity = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
        }
        for path in sources
    ]
    return sha256_bytes(canonical_json_bytes(identity))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(values[0])
    if any(set(row) != set(fields) for row in values):
        raise ValueError(f"CSV rows do not share one schema: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def _difference_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for transform in TRANSFORM_FIELDS:
        group = [row for row in rows if row["transform"] == transform]
        for metric in ("translation_difference_m", "rotation_difference_deg"):
            values = np.asarray(
                [float(row[metric]) for row in group], dtype=np.float64
            )
            if not values.size:
                continue
            output.append(
                {
                    "transform": transform,
                    "metric": metric,
                    "pair_count": int(values.size),
                    "p50": float(np.quantile(values, 0.50, method="linear")),
                    "p80": float(np.quantile(values, 0.80, method="linear")),
                    "p95": float(np.quantile(values, 0.95, method="linear")),
                    "maximum": float(np.max(values)),
                }
            )
    return output


def _candidate_parameters(
    candidates: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate_id, candidate in sorted(candidates.items()):
        correction = np.degrees(
            np.asarray(candidate["right_correction_rotvec_rad"], dtype=np.float64)
        )
        observability = candidate.get("observability", {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "rx_correction_deg": float(correction[0]),
                "ry_correction_deg": float(correction[1]),
                "rz_correction_deg": float(correction[2]),
                "objective_cost": float(candidate["objective_cost"]),
                "quality_gate_pass": bool(candidate["quality_gate_pass"]),
                "observability_condition_number": float(
                    observability.get("condition_number", float("nan"))
                ),
            }
        )
    return rows


def _bag_inventory_row(
    date: str,
    bag_id: str,
    classified: Sequence[DynamicFrameEvidence],
    within_8m: Sequence[DynamicFrameEvidence],
    identity_sha256: str,
) -> dict[str, object]:
    depths = [
        float(item.source.initial_tag0_depth_m)
        for item in within_8m
        if item.source.initial_tag0_depth_m is not None
    ]
    bearings = [
        float(item.source.bearing_x_deg)
        for item in within_8m
        if item.source.bearing_x_deg is not None
    ]
    return {
        "date": date,
        "bag_id": bag_id,
        "raw_data_identity_sha256": identity_sha256,
        "frame_count": len(classified),
        "base_valid_frame_count": sum(
            item.source.observation is not None and not item.source.exclusion_reasons
            for item in classified
        ),
        "within_8m_frame_count": len(within_8m),
        "within_8m_depth_min_m": min(depths) if depths else "",
        "within_8m_depth_max_m": max(depths) if depths else "",
        "within_8m_bearing_median_deg": (
            float(np.median(bearings)) if bearings else ""
        ),
    }


def _write_day_report(
    output_dir: Path,
    *,
    date: str,
    source_present: bool,
    inventory: Sequence[Mapping[str, object]],
    plan: Sequence[Mapping[str, object]],
    pairwise_summary: Sequence[Mapping[str, object]],
    parameters: Sequence[Mapping[str, object]],
    holdouts: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    validation_failures: Sequence[Mapping[str, object]],
    maximum_depth_m: float = 8.0,
    maximum_depth_inclusive: bool = True,
    tag_geometry_mode: str = "frozen_reuse",
    allow_failed_tag_geometry: bool = False,
    calibration_bag_ids: Sequence[str] = (),
    displacement_bag_ids: Sequence[str] = (),
    displacement_summary: Sequence[Mapping[str, object]] = (),
) -> Path:
    depth_operator = "<=" if maximum_depth_inclusive else "<"
    lines = [f"# {date} 两-bag 动态标定与同日验证", ""]
    if not source_present:
        lines.extend(
            [
                "状态：`SOURCE_DATA_MISSING`。",
                "",
                f"配置的 {date} rosbag 数据不存在，未执行标定或验证。",
            ]
        )
    elif not inventory:
        lines.append("状态：`NO_CONFIGURED_DYNAMIC_BAGS`。")
    else:
        calibration_ids = frozenset(str(value) for value in calibration_bag_ids)
        displacement_ids = frozenset(str(value) for value in displacement_bag_ids)
        lines.extend(
            [
                "所有候选仅使用本日 2 个动态 bag 求解；每个候选的验证集合明确排除它自己的 2 个训练 bag，未使用跨日数据。",
                (
                    f"标定仅允许使用：`{', '.join(calibration_bag_ids)}`；"
                    f"撞桥前位移评估仅使用：`{', '.join(displacement_bag_ids)}`，"
                    "这些 bag 不参与组合生成、候选排名或代表轨迹选择。"
                    if displacement_bag_ids
                    else ""
                ),
                "",
                f"不使用结构状态门；仅基础证据有效且 `0 < camera_tag0_tz_m {depth_operator} {maximum_depth_m:.1f}` 的帧纳入标定和验证。",
                (
                    "每个组合仅从自身两包 bootstrap 并重新估计 `T_tag0_tag1`，不读取旧标定矩阵。"
                    if tag_geometry_mode == "bootstrap"
                    else "本轮固定已批准 `T_tag0_tag1`，仅执行固定几何重投影检查和 B2 条件消融。"
                ),
                (
                    "G2 质量门失败的组合仅为计算跨组合偏差而继续 B2；其 `quality_gate_pass` 保持为 false，候选始终不可安装。"
                    if allow_failed_tag_geometry
                    else "G2 质量门失败时中止该候选，不继续 B2。"
                ),
                "",
                "## 8 m 数据清单",
                "",
                "| bag | role | raw frames | base valid | within 8 m | depth range (m) | median bearing (deg) |",
                "|---|---|---:|---:|---:|---|---:|",
            ]
        )
        for row in inventory:
            bag_id = str(row["bag_id"])
            role = (
                "displacement evaluation"
                if bag_id in displacement_ids
                else "calibration/holdout"
                if bag_id in calibration_ids
                else "holdout"
            )
            if int(row["within_8m_frame_count"]) > 0:
                depth_range = (
                    f"{float(row['within_8m_depth_min_m']):.3f}–"
                    f"{float(row['within_8m_depth_max_m']):.3f}"
                )
                bearing = f"{float(row['within_8m_bearing_median_deg']):.3f}"
            else:
                depth_range = "-"
                bearing = "-"
            lines.append(
                f"| {row['bag_id']} | {role} | {row['frame_count']} | {row['base_valid_frame_count']} | "
                f"{row['within_8m_frame_count']} | "
                f"{depth_range} | {bearing} |"
            )
        lines.extend(
            [
                "",
                "## 标定组合",
                "",
                "| candidate | training bags | selected frames | bearing span (deg) |",
                "|---|---|---:|---:|",
            ]
        )
        for row in plan:
            lines.append(
                f"| {row['candidate_id']} | {', '.join(row['bag_ids'])} | "
                f"{row['selected_frame_count']} | {float(row['bearing_span_deg']):.3f} |"
            )
        lines.extend(
            [
                "",
                "## 同日候选两两差异",
                "",
                "| transform | metric | pairs | P50 | P80 | P95 | maximum |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in pairwise_summary:
            lines.append(
                f"| {row['transform']} | {row['metric']} | {row['pair_count']} | "
                f"{float(row['p50']):.6f} | {float(row['p80']):.6f} | "
                f"{float(row['p95']):.6f} | {float(row['maximum']):.6f} |"
            )
        lines.extend(
            [
                "",
                "## INS–相机旋转修正",
                "",
                "| candidate | Rx (deg) | Ry (deg) | cost | candidate quality | condition |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for row in parameters:
            lines.append(
                f"| {row['candidate_id']} | {float(row['rx_correction_deg']):.6f} | "
                f"{float(row['ry_correction_deg']):.6f} | {float(row['objective_cost']):.3f} | "
                f"{row['quality_gate_pass']} | {float(row['observability_condition_number']):.3f} |"
            )
        if holdouts:
            lines.extend(
                [
                    "",
                    "## 严格同日留出验证（逐 bag 等权）",
                    "",
                    "| candidate | scope | holdout bags | XY P80 (m) | XY P95 (m) | yaw P80 (deg) | yaw P95 (deg) |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in holdouts:
                lines.append(
                    f"| {row['candidate_id']} | {row['validation_scope']} | {row['holdout_bag_count']} | "
                    f"{float(row['equal_bag_xy_p80_m']):.6f} | "
                    f"{float(row['equal_bag_xy_p95_m']):.6f} | "
                    f"{float(row['equal_bag_yaw_p80_deg']):.6f} | "
                    f"{float(row['equal_bag_yaw_p95_deg']):.6f} |"
                )
            lines.extend(
                [
                    "",
                    "逐 bag 交互网页入口：`validation_same_day/interactive_bags/index.html`。",
                ]
            )
        if displacement_summary:
            selection_path = output_dir / "selected_candidate.json"
            selected_candidate_id = None
            if selection_path.is_file():
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                selected_candidate_id = str(selection["selected_candidate_id"])
            selected_displacements = [
                row
                for row in displacement_summary
                if selected_candidate_id is None
                or str(row["candidate_id"]) == selected_candidate_id
            ]
            lines.extend(
                [
                    "",
                    "## 撞桥前码牌位移评估",
                    "",
                    "以下结果由撞桥后标定结果与前三组逐帧重建的撞桥前码牌位姿直接比较；不参与标定或候选排序。",
                    "",
                    "| candidate | bags | translation median (m) | translation max (m) | abs yaw median (deg) | abs yaw max (deg) |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in selected_displacements:
                lines.append(
                    f"| {row['candidate_id']} | {row['displacement_bag_count']} | "
                    f"{float(row['equal_bag_translation_median_m']):.6f} | "
                    f"{float(row['equal_bag_translation_max_m']):.6f} | "
                    f"{float(row['equal_bag_abs_yaw_median_deg']):.6f} | "
                    f"{float(row['equal_bag_abs_yaw_max_deg']):.6f} |"
                )
        if failures:
            lines.extend(["", "## 标定失败", ""])
            lines.extend(
                f"- `{row['candidate_id']}`: {row['error']}" for row in failures
            )
        if validation_failures:
            lines.extend(["", "## 验证排除/失败", ""])
            lines.extend(
                f"- `{row['bag_id']}`: {row['error']}" for row in validation_failures
            )
        selection_path = output_dir / "selected_candidate.json"
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            lines.extend(
                [
                    "",
                    "## 展示候选与轨迹",
                    "",
                    f"- 候选：`{selection['selected_candidate_id']}`；",
                    f"- 代表留出 bag：`{selection['representative_holdout_bag_id']}`；",
                    "- 交互轨迹：`SELECTED_INS_VISUAL_TRAJECTORY.html`；",
                    "- `T_ins_map_tag0_calibrated` 的语义为 `T_world_tag0`：`p_world = T_world_tag0 · p_tag0`。",
                ]
            )
    path = output_dir / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_date(
    *,
    date: str,
    spec: Mapping[str, object],
    args: argparse.Namespace,
    online: object,
    experiment: object,
    machine_config_dir: Path,
    run_identity: Mapping[str, object],
) -> dict[str, object]:
    output_dir = args.output_dir / date
    output_dir.mkdir(parents=True, exist_ok=True)
    bag_ids, calibration_bag_ids, displacement_bag_ids = _dataset_bag_groups(spec)
    if not bag_ids:
        report = _write_day_report(
            output_dir,
            date=date,
            source_present=True,
            inventory=(),
            plan=(),
            pairwise_summary=(),
            parameters=(),
            holdouts=(),
            failures=(),
            validation_failures=(),
        )
        return {"date": date, "status": "NO_CONFIGURED_DYNAMIC_BAGS", "report": str(report)}

    classified_by_bag: dict[str, tuple[DynamicFrameEvidence, ...]] = {}
    within_8m_by_bag: dict[str, tuple[DynamicFrameEvidence, ...]] = {}
    validation_by_bag: dict[str, tuple[DynamicFrameEvidence, ...]] = {}
    identities: dict[str, dict[str, object]] = {}
    inventory: list[dict[str, object]] = []
    read_failures: list[dict[str, object]] = []
    for index, bag_id in enumerate(bag_ids, 1):
        _progress(
            "same_day_bag_read_start",
            date=date,
            bag_id=bag_id,
            bag_index=index,
            bag_count=len(bag_ids),
        )
        try:
            evidence = read_dynamic_bag_evidence(
                Path(args.bag_paths[bag_id]),
                bag_id,
                machine_config_dir,
                topics=experiment.topics,
                interpolation_config=experiment.interpolation,
                live_qc_config=online.live_qc,
                board_instance_id=online.calibration.board_instance_id,
                novatel_message_dir=args.novatel_message_dir,
            )
            classified = classify_dynamic_frames(
                evidence,
                experiment.structural_gate,
            )
            within_8m = retain_base_valid_frames_within_depth(
                classified,
                maximum_depth_m=float(args.maximum_depth_m),
                inclusive=bool(args.maximum_depth_inclusive),
            )
            classified_by_bag[bag_id] = classified
            within_8m_by_bag[bag_id] = within_8m
            validation_by_bag[bag_id] = tuple(
                replace(item, structural_state=StructuralState.UNKNOWN)
                for item in within_8m
            )
            identities[bag_id] = {
                "date": date,
                "raw_data_identity_sha256": evidence.raw_data_identity_sha256,
                "detector_implementation": evidence.diagnostics[
                    "detector_implementation"
                ],
            }
            inventory.append(
                _bag_inventory_row(
                    date,
                    bag_id,
                    classified,
                    within_8m,
                    evidence.raw_data_identity_sha256,
                )
            )
            _progress(
                "same_day_bag_read_complete",
                date=date,
                bag_id=bag_id,
                within_8m_frame_count=inventory[-1]["within_8m_frame_count"],
            )
        except Exception as error:
            read_failures.append({"bag_id": bag_id, "error": str(error)})
            _progress(
                "same_day_bag_read_failed",
                date=date,
                bag_id=bag_id,
                error=str(error),
            )
    _write_csv(output_dir / "bag_8m_inventory.csv", inventory)
    atomic_write_json(output_dir / "bag_read_failures.json", read_failures)

    all_frames = tuple(
        item
        for bag_id in calibration_bag_ids
        if bag_id in within_8m_by_bag
        for item in within_8m_by_bag[bag_id]
    )
    feasible = (
        enumerate_feasible_dynamic_combinations(
            all_frames,
            experiment.sampling,
            bag_count=2,
            distance_edges_m=experiment.coverage.distance_bin_edges_m,
            minimum_bearing_span_deg=experiment.coverage.minimum_bearing_span_deg,
        )
        if len({item.source.bag_id for item in all_frames}) >= 2
        else ()
    )
    maximum_combinations = int(spec["maximum_combinations"])
    selected = select_same_day_combination_panel(
        feasible,
        maximum_count=maximum_combinations,
    ) if maximum_combinations else ()
    plan: list[dict[str, object]] = []
    for index, combination in enumerate(selected, 1):
        selected_depths = [
            float(item.source.initial_tag0_depth_m)
            for item in combination.selection.frames
            if item.source.initial_tag0_depth_m is not None
        ]
        if not selected_depths:
            raise ValueError("selected calibration combination has no depth evidence")
        maximum_selected_depth_m = max(selected_depths)
        depth_passed = (
            maximum_selected_depth_m <= float(args.maximum_depth_m)
            if args.maximum_depth_inclusive
            else maximum_selected_depth_m < float(args.maximum_depth_m)
        )
        if not depth_passed:
            raise ValueError(
                "selected calibration frame violates the configured depth limit"
            )
        plan.append(
            {
                "candidate_id": f"{date}_dyn2_c{index:02d}",
                "bag_ids": list(combination.bag_ids),
                "selected_frame_count": len(combination.selection.frames),
                "selected_frame_keys_sha256": combination.selection.selected_frame_keys_sha256,
                "bearing_span_deg": combination.selection.coverage.bearing_span_deg,
                "maximum_selected_depth_m": maximum_selected_depth_m,
                "maximum_depth_m": float(args.maximum_depth_m),
                "maximum_depth_inclusive": bool(args.maximum_depth_inclusive),
                "distance_bin_bag_counts": [
                    list(value)
                    for value in combination.selection.coverage.distance_bin_bag_counts
                ],
            }
        )
    atomic_write_json(
        output_dir / "combination_plan.json",
        {
            "schema_version": 1,
            "date": date,
            "cross_date_validation_prohibited": True,
            "configured_bag_ids": list(bag_ids),
            "calibration_bag_ids": list(calibration_bag_ids),
            "displacement_evaluation_bag_ids": list(displacement_bag_ids),
            "readable_bag_ids": sorted(classified_by_bag),
            "feasible_combination_count": len(feasible),
            "selected_combinations": plan,
        },
    )
    _progress(
        "same_day_combination_plan_complete",
        date=date,
        feasible_count=len(feasible),
        selected_count=len(selected),
    )
    if args.inventory_only or not selected:
        failures = read_failures + ([] if selected else [{"candidate_id": "__plan__", "error": "no feasible two-bag combination passed coverage"}])
        report = _write_day_report(
            output_dir,
            date=date,
            source_present=True,
            inventory=inventory,
            plan=plan,
            pairwise_summary=(),
            parameters=(),
            holdouts=(),
            failures=failures,
            validation_failures=(),
        )
        if args.inventory_only and not classified_by_bag:
            status = "INVENTORY_FAILED"
        elif args.inventory_only and read_failures:
            status = "INVENTORY_WITH_ERRORS"
        elif args.inventory_only:
            status = "INVENTORY_ONLY"
        else:
            status = "NO_FEASIBLE_COMBINATION"
        return {
            "date": date,
            "status": status,
            "configured_bag_count": len(bag_ids),
            "readable_bag_count": len(classified_by_bag),
            "feasible_combination_count": len(feasible),
            "selected_combination_count": len(selected),
            "report": str(report),
        }

    candidates: dict[str, dict[str, object]] = {}
    solve_failures: list[dict[str, object]] = []
    for index, (combination, plan_row) in enumerate(zip(selected, plan), 1):
        candidate_id = str(plan_row["candidate_id"])
        candidate_dir = output_dir / "candidates" / candidate_id
        candidate_path = candidate_dir / "recovered_candidate.yaml"
        selected_bags = frozenset(combination.bag_ids)
        calibration_frames = tuple(
            item
            for item in all_frames
            if item.source.bag_id in selected_bags
        )
        problem = build_calibration_problem_from_dynamic_frames(
            calibration_frames,
            machine_config_dir,
            tag_geometry_run_config=online.calibration.tag_geometry,
        )
        algorithm_id = f"B2-SAME-DAY-DYNAMIC-RXY-{candidate_id}"
        solve_identity = {
            "schema_version": 2,
            "date": date,
            "candidate_id": candidate_id,
            "algorithm_id": algorithm_id,
            "bag_ids": list(combination.bag_ids),
            "selected_frame_keys_sha256": (
                combination.selection.selected_frame_keys_sha256
            ),
            "tag_geometry_mode": str(args.tag_geometry_mode),
            "approved_tag_geometry_sha256": (
                problem.initial_tag0_tag1_source_sha256
            ),
            "machine_config_fingerprint": problem.config_fingerprint,
            "run_identity": dict(run_identity),
            "evidence_identities": {
                bag_id: identities[bag_id] for bag_id in combination.bag_ids
            },
        }
        reused = _read_reusable_candidate(candidate_dir, solve_identity)
        if reused is not None:
            candidates[candidate_id] = reused
            _progress("same_day_combination_reused", date=date, candidate_id=candidate_id)
            continue
        _progress(
            "same_day_combination_solve_start",
            date=date,
            candidate_id=candidate_id,
            combination_index=index,
            combination_count=len(selected),
            bag_ids=list(combination.bag_ids),
            selected_frame_count=len(combination.selection.frames),
        )
        try:
            candidate_parent = output_dir / "candidates"
            candidate_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f".{candidate_id}.attempt-",
                dir=candidate_parent,
            ) as attempt_name:
                attempt_dir = Path(attempt_name)
                solved = solve_safe_dynamic_candidate(
                    problem,
                    combination.selection,
                    online_config=online,
                    machine_config_dir=machine_config_dir,
                    output_dir=attempt_dir,
                    evidence_identity={
                        bag_id: identities[bag_id]
                        for bag_id in combination.bag_ids
                    },
                    progress_callback=(
                        lambda stage, status, cid=candidate_id: _progress(
                            "same_day_combination_stage",
                            date=date,
                            candidate_id=cid,
                            candidate_stage=stage,
                            status=status,
                        )
                    ),
                    algorithm_id=algorithm_id,
                    freeze_reused_tag_geometry=bool(args.freeze_reused_tag_geometry),
                    allow_failed_tag_geometry=bool(args.allow_failed_tag_geometry),
                    software_fingerprint=str(
                        run_identity["pipeline_software_sha256"]
                    ),
                )
                _read_candidate(attempt_dir / "recovered_candidate.yaml")
                _publish_candidate_attempt(
                    attempt_dir,
                    candidate_dir,
                    solve_identity,
                )
            candidates[candidate_id] = _read_candidate(candidate_path)
            _progress(
                "same_day_combination_solve_complete",
                date=date,
                candidate_id=candidate_id,
                quality_gate_pass=solved.quality_gate_pass,
                quality_gate_reasons=list(solved.quality_gate_reasons),
            )
        except Exception as error:
            failure = {"candidate_id": candidate_id, "error": str(error)}
            solve_failures.append(failure)
            atomic_write_json(candidate_dir / "failure.json", failure)
            _progress("same_day_combination_solve_failed", date=date, **failure)

    pairwise_rows = pairwise_candidate_differences(
        candidates,
        transform_fields=TRANSFORM_FIELDS,
    ) if len(candidates) >= 2 else ()
    pairwise_summary = _difference_summary(pairwise_rows)
    stability_summary = summarize_transform_stability(pairwise_rows)
    parameters = _candidate_parameters(candidates)
    _write_csv(output_dir / "pairwise_transform_differences.csv", pairwise_rows)
    _write_csv(output_dir / "pairwise_transform_summary.csv", pairwise_summary)
    _write_csv(output_dir / "transform_stability_summary.csv", stability_summary)
    _write_csv(
        output_dir / "T_tag0_tag1_pairwise_differences.csv",
        [
            row
            for row in pairwise_rows
            if row["transform"] == "T_tag0_tag1_calibrated"
        ],
    )
    _write_csv(
        output_dir / "T_world_tag0_pairwise_differences.csv",
        [
            row
            for row in pairwise_rows
            if row["transform"] == "T_ins_map_tag0_calibrated"
        ],
    )
    _write_csv(
        output_dir / "T_tag0_tag1_stability_summary.csv",
        [
            row
            for row in stability_summary
            if row["transform"] == "T_tag0_tag1_calibrated"
        ],
    )
    _write_csv(
        output_dir / "T_world_tag0_stability_summary.csv",
        [
            row
            for row in stability_summary
            if row["transform"] == "T_ins_map_tag0_calibrated"
        ],
    )
    _write_csv(output_dir / "candidate_parameters.csv", parameters)

    validation_rows: list[Mapping[str, object]] = []
    per_bag_metrics: list[Mapping[str, object]] = []
    depth_metrics: list[Mapping[str, object]] = []
    pooled_metrics: list[Mapping[str, object]] = []
    retention: list[Mapping[str, object]] = []
    common_counts: list[Mapping[str, object]] = []
    validation_failures: list[dict[str, object]] = []
    validation_exclusions: list[dict[str, object]] = []
    completed_validation_bags: set[str] = set()
    interactive_reports = []
    training_by_candidate = {
        str(row["candidate_id"]): tuple(str(value) for value in row["bag_ids"])
        for row in plan
        if str(row["candidate_id"]) in candidates
    }
    if not args.skip_validation:
        camera, tag_size_m = load_live_qc_profile(machine_config_dir)
        dataset_name = f"onsite_{date}_same_day_holdout"
        for bag_id in sorted(validation_by_bag):
            eligible = {
                candidate_id: candidates[candidate_id]
                for candidate_id, training_bags in training_by_candidate.items()
                if bag_id not in training_bags
            }
            if not eligible:
                validation_failures.append(
                    {"bag_id": bag_id, "error": "no candidate leaves this bag out"}
                )
                continue
            if not validation_by_bag[bag_id]:
                validation_exclusions.append(
                    {
                        "bag_id": bag_id,
                        "error": "no base-valid validation frame within 8 m",
                    }
                )
                continue
            primary_states = (StructuralState.UNKNOWN,)
            validation_scope = str(args.validation_scope)
            try:
                result = evaluate_safe_dynamic_candidates(
                    eligible,
                    {bag_id: validation_by_bag[bag_id]},
                    dataset=dataset_name,
                    camera_model=camera,
                    tag_size_m=tag_size_m,
                    primary_states=primary_states,
                    validation_scope=validation_scope,
                )
                validation_rows.extend(result.rows)
                per_bag_metrics.extend(result.per_bag_metrics)
                depth_metrics.extend(result.depth_metrics)
                pooled_metrics.extend(result.pooled_metrics)
                retention.extend(result.retention)
                common_counts.extend(result.common_frame_counts)
                completed_validation_bags.add(bag_id)
                bag_retention = {
                    str(row["candidate_id"]): float(row["primary_retained_fraction"])
                    for row in result.retention
                }
                origins = {
                    candidate_id: np.asarray(
                        candidate["T_ins_map_tag0_calibrated"], dtype=np.float64
                    )[:3, 3]
                    for candidate_id, candidate in eligible.items()
                }
                interactive_reports.append(
                    write_interactive_bag_report(
                        result.rows,
                        dataset=dataset_name,
                        bag_id=bag_id,
                        output_path=(
                            output_dir
                            / "validation_same_day/interactive_bags"
                            / dataset_name
                            / f"{bag_id}.html"
                        ),
                        bag_identity_sha256=str(
                            identities[bag_id]["raw_data_identity_sha256"]
                        ),
                        retention_by_candidate=bag_retention,
                        tag0_map_origin_by_candidate=origins,
                    )
                )
                _progress(
                    "same_day_validation_bag_complete",
                    date=date,
                    bag_id=bag_id,
                    candidate_count=len(eligible),
                    validation_scope=validation_scope,
                    common_frame_count=result.common_frame_counts[0]["common_frame_count"],
                )
            except Exception as error:
                validation_failures.append({"bag_id": bag_id, "error": str(error)})
                _progress(
                    "same_day_validation_bag_failed",
                    date=date,
                    bag_id=bag_id,
                    error=str(error),
                )
        if interactive_reports:
            write_interactive_bag_index(
                interactive_reports,
                output_dir / "validation_same_day/interactive_bags/index.html",
            )

    validation_dir = output_dir / "validation_same_day"
    _write_csv(validation_dir / "common_validation_rows.csv", validation_rows)
    _write_csv(validation_dir / "per_bag_metrics.csv", per_bag_metrics)
    _write_csv(validation_dir / "depth_metrics.csv", depth_metrics)
    _write_csv(validation_dir / "pooled_metrics_per_bag_run.csv", pooled_metrics)
    _write_csv(validation_dir / "candidate_retention.csv", retention)
    _write_csv(validation_dir / "common_frame_counts.csv", common_counts)
    displacement_per_bag, displacement_summary = _summarize_board_displacement(
        validation_rows,
        displacement_bag_ids,
    )
    _write_csv(
        validation_dir / "preimpact_board_displacement_per_bag.csv",
        displacement_per_bag,
    )
    _write_csv(
        validation_dir / "preimpact_board_displacement_summary.csv",
        displacement_summary,
    )
    atomic_write_json(validation_dir / "validation_failures.json", validation_failures)
    atomic_write_json(
        validation_dir / "validation_exclusions.json",
        validation_exclusions,
    )
    validation_scope = str(args.validation_scope)
    scope_metrics = [
        row
        for row in per_bag_metrics
        if row["validation_scope"] == validation_scope
    ]
    normal_scope_metrics, displacement_scope_metrics = _partition_validation_metrics(
        scope_metrics,
        displacement_bag_ids,
    )
    _write_csv(
        validation_dir / "preimpact_localization_metrics.csv",
        displacement_scope_metrics,
    )
    scope_candidates = {
        str(row["candidate_id"]) for row in normal_scope_metrics
    }
    scope_training = {
        candidate_id: bag_ids
        for candidate_id, bag_ids in training_by_candidate.items()
        if candidate_id in scope_candidates
    }
    holdout_summary = (
        [
            {**row, "validation_scope": validation_scope}
            for row in summarize_candidate_holdouts(
                normal_scope_metrics,
                scope_training,
            )
        ]
        if scope_training
        else []
    )
    _write_csv(
        validation_dir / "equal_bag_all_dynamic_le8m_holdout_summary.csv",
        holdout_summary,
    )
    _write_csv(validation_dir / "equal_bag_holdout_summary.csv", holdout_summary)

    analysis_candidates = {
        candidate_id: candidate
        for candidate_id, candidate in candidates.items()
        if candidate_id in scope_candidates
    }
    ranking: tuple[dict[str, object], ...] = ()
    selected_candidate_id: str | None = None
    representative_bag_id: str | None = None
    selected_trajectory_report: str | None = None
    if len(analysis_candidates) >= 2:
        qualified_pairwise = pairwise_candidate_differences(
            analysis_candidates,
            transform_fields=(
                "T_tag0_tag1_calibrated",
                "T_ins_map_tag0_calibrated",
            ),
        )
        qualified_holdouts = [
            row
            for row in holdout_summary
            if str(row["candidate_id"]) in analysis_candidates
        ]
        ranking = rank_candidates_by_stability_and_holdout(
            qualified_pairwise,
            qualified_holdouts,
            required_transforms=(
                "T_tag0_tag1_calibrated",
                "T_ins_map_tag0_calibrated",
            ),
        )
        ranking = apply_quality_gate_preference(
            ranking,
            {
                candidate_id: bool(candidate.get("quality_gate_pass"))
                for candidate_id, candidate in analysis_candidates.items()
            },
        )
        selected_candidate_id = str(ranking[0]["candidate_id"])
        representative_bag_id = select_representative_validation_bag(
            normal_scope_metrics,
            selected_candidate_id,
        )
        selected_rows = [
            row
            for row in validation_rows
            if str(row["candidate_id"]) == selected_candidate_id
            and str(row["bag_id"]) == representative_bag_id
        ]
        selected_retention = next(
            float(row["primary_retained_fraction"])
            for row in retention
            if str(row["candidate_id"]) == selected_candidate_id
            and str(row["bag_id"]) == representative_bag_id
        )
        selected_origin = np.asarray(
            analysis_candidates[selected_candidate_id][
                "T_ins_map_tag0_calibrated"
            ],
            dtype=np.float64,
        )[:3, 3]
        selected_path = output_dir / "SELECTED_INS_VISUAL_TRAJECTORY.html"
        write_interactive_bag_report(
            selected_rows,
            dataset=f"onsite_{date}_same_day_holdout",
            bag_id=representative_bag_id,
            output_path=selected_path,
            bag_identity_sha256=str(
                identities[representative_bag_id]["raw_data_identity_sha256"]
            ),
            retention_by_candidate={selected_candidate_id: selected_retention},
            tag0_map_origin_by_candidate={selected_candidate_id: selected_origin},
        )
        selected_trajectory_report = str(selected_path)
    _write_csv(output_dir / "candidate_ranking.csv", ranking)
    atomic_write_json(
        output_dir / "selected_candidate.json",
        {
            "schema_version": 1,
            "selected_candidate_id": selected_candidate_id,
            "representative_holdout_bag_id": representative_bag_id,
            "trajectory_report": selected_trajectory_report,
            "selection_policy": (
                "quality-gate pass, then equal-weight transform-stability and "
                "holdout-error rank; representative bag closest to median XY/yaw P80"
            ),
            "transform_semantics": {
                "T_tag0_tag1_calibrated": "p_tag0 = T_tag0_tag1 * p_tag1",
                "T_ins_map_tag0_calibrated": "p_world = T_world_tag0 * p_tag0",
            },
        },
    )

    failures = read_failures + solve_failures
    reported_validation_issues = validation_exclusions + validation_failures
    report = _write_day_report(
        output_dir,
        date=date,
        source_present=True,
        inventory=inventory,
        plan=plan,
        pairwise_summary=pairwise_summary,
        parameters=parameters,
        holdouts=holdout_summary,
        failures=failures,
        validation_failures=reported_validation_issues,
        maximum_depth_m=float(args.maximum_depth_m),
        maximum_depth_inclusive=bool(args.maximum_depth_inclusive),
        tag_geometry_mode=str(args.tag_geometry_mode),
        allow_failed_tag_geometry=bool(args.allow_failed_tag_geometry),
        calibration_bag_ids=calibration_bag_ids,
        displacement_bag_ids=displacement_bag_ids,
        displacement_summary=displacement_summary,
    )
    validated_candidate_count = len(
        {str(row["candidate_id"]) for row in holdout_summary}
    )
    status = _day_completion_status(
        selected_count=len(selected),
        successful_count=len(candidates),
        validated_count=validated_candidate_count,
        validation_skipped=bool(args.skip_validation),
        failure_count=len(failures),
        validation_failure_count=len(validation_failures),
        exclusion_count=len(validation_exclusions),
    )
    summary = {
        "date": date,
        "status": status,
        "configured_bag_count": len(bag_ids),
        "calibration_bag_count": len(calibration_bag_ids),
        "displacement_evaluation_bag_count": len(displacement_bag_ids),
        "readable_bag_count": len(classified_by_bag),
        "within_8m_bag_count": sum(
            int(row["within_8m_frame_count"]) > 0 for row in inventory
        ),
        "feasible_combination_count": len(feasible),
        "selected_combination_count": len(selected),
        "successful_candidate_count": len(candidates),
        "validated_candidate_count": validated_candidate_count,
        "validated_bag_count": len(completed_validation_bags),
        "validation_exclusion_count": len(validation_exclusions),
        "validation_failure_count": len(validation_failures),
        "selected_candidate_id": selected_candidate_id,
        "representative_holdout_bag_id": representative_bag_id,
        "selected_trajectory_report": selected_trajectory_report,
        "report": str(report),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _write_root_report(
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    *,
    title: str = "20260811–20260815 两-bag 动态标定与同日验证",
    maximum_depth_m: float = 8.0,
    maximum_depth_inclusive: bool = True,
    tag_geometry_mode: str = "frozen_reuse",
) -> Path:
    depth_operator = "<=" if maximum_depth_inclusive else "<"
    geometry_text = (
        "每个组合仅用自身两包从零初始化并重新估计双 Tag 几何，不读取旧标定矩阵。"
        if tag_geometry_mode == "bootstrap"
        else "每个组合固定已批准双 Tag 几何，仅评估外参条件消融。"
    )
    lines = [
        f"# {title}",
        "",
        "本实验禁止跨日验证：每个候选只用当天 2 个动态 bag 标定，并只在当天未参与该候选训练的 bag 上验证。",
        f"全程不使用 safe/contact/loaded/release 结构分组；仅基础证据有效且 `0 < camera_tag0_tz_m {depth_operator} {maximum_depth_m:.1f}` 的帧可用于标定与同口径验证。",
        geometry_text,
        "二-bag 覆盖门要求近/中/远三档各至少有 1 个 bag、合计至少 40 个均衡帧，且两个 bag 的采样中位 bearing 跨度至少 0.1°；不沿用 0819 safe 数据的 3° bearing 门。",
        "",
        "| date | status | configured bags | bags with <=8 m frames | feasible pairs | selected | solved | validated |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['date']} | {row['status']} | {row.get('configured_bag_count', 0)} | "
            f"{row.get('within_8m_bag_count', 0)} | {row.get('feasible_combination_count', 0)} | "
            f"{row.get('selected_combination_count', 0)} | {row.get('successful_candidate_count', 0)} | "
            f"{row.get('validated_candidate_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 日期说明",
            "",
            "- 0813、0815 若显示 `SOURCE_DATA_MISSING`，表示本机原始数据根目录中没有对应日期，未用其它日期代替。",
            "- 每日详细报告、候选、CSV 与交互网页均位于对应日期子目录。",
            "- 这些是自动生成分析产物，只保存在 ignored regression output，不进入 Git。",
        ]
    )
    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_completed_findings(output_dir: Path) -> Path:
    day_values: dict[str, dict[str, object]] = {}
    for date in ("20260811", "20260812", "20260814"):
        day_dir = output_dir / date
        inventory = _read_csv(day_dir / "bag_8m_inventory.csv")
        parameters = _read_csv(day_dir / "candidate_parameters.csv")
        pairwise = _read_csv(day_dir / "pairwise_transform_differences.csv")
        holdouts = _read_csv(
            day_dir / "validation_same_day/equal_bag_holdout_summary.csv"
        )
        camera_pairs = [
            row
            for row in pairwise
            if row["transform"] == "T_ins_camera_calibrated"
        ]
        map_pairs = [
            row
            for row in pairwise
            if row["transform"] == "T_ins_map_tag0_calibrated"
        ]
        tag_pairs = [
            row
            for row in pairwise
            if row["transform"] == "T_tag0_tag1_calibrated"
        ]
        day_values[date] = {
            "within_8m_bags": sum(
                int(row["within_8m_frame_count"]) > 0 for row in inventory
            ),
            "within_8m_frames": sum(
                int(row["within_8m_frame_count"]) for row in inventory
            ),
            "parameters": parameters,
            "camera_worst": (
                max(camera_pairs, key=lambda row: float(row["rotation_difference_deg"]))
                if camera_pairs
                else None
            ),
            "map_worst": (
                max(map_pairs, key=lambda row: float(row["rotation_difference_deg"]))
                if map_pairs
                else None
            ),
            "tag_rotation_max": (
                max(float(row["rotation_difference_deg"]) for row in tag_pairs)
                if tag_pairs
                else 0.0
            ),
            "holdouts": holdouts,
        }

    date = "20260814"
    day_dir = output_dir / date
    plan_payload = json.loads((day_dir / "combination_plan.json").read_text(encoding="utf-8"))
    training = {
        str(row["candidate_id"]): frozenset(str(value) for value in row["bag_ids"])
        for row in plan_payload["selected_combinations"]
    }
    parameters = {
        row["candidate_id"]: row
        for row in day_values[date]["parameters"]
    }
    influence_rows: list[dict[str, object]] = []
    for bag_id in sorted(set().union(*training.values())):
        included = [key for key, bags in training.items() if bag_id in bags]
        excluded = [key for key, bags in training.items() if bag_id not in bags]
        row: dict[str, object] = {
            "bag_id": bag_id,
            "included_combination_count": len(included),
            "excluded_combination_count": len(excluded),
        }
        sources = {
            "rx_correction_deg": (parameters, "rx_correction_deg"),
            "ry_correction_deg": (parameters, "ry_correction_deg"),
        }
        for label, (source, field) in sources.items():
            included_mean = float(
                np.mean([float(source[key][field]) for key in included])
            )
            excluded_mean = float(
                np.mean([float(source[key][field]) for key in excluded])
            )
            row[f"{label}_included_minus_excluded"] = included_mean - excluded_mean
        influence_rows.append(row)
    _write_csv(day_dir / "bag_influence.csv", influence_rows)

    def span(date_key: str, field: str) -> float:
        rows = day_values[date_key]["parameters"]
        values = [float(row[field]) for row in rows]
        return max(values) - min(values) if values else 0.0

    def holdout_mean(date_key: str, field: str) -> float | None:
        rows = day_values[date_key]["holdouts"]
        return float(np.mean([float(row[field]) for row in rows])) if rows else None

    worst_influence_rx = max(
        influence_rows,
        key=lambda row: abs(float(row["rx_correction_deg_included_minus_excluded"])),
    )
    worst_influence_ry = max(
        influence_rows,
        key=lambda row: abs(float(row["ry_correction_deg_included_minus_excluded"])),
    )
    priority_bag_ids = list(
        dict.fromkeys(
            [worst_influence_rx["bag_id"], worst_influence_ry["bag_id"]]
        )
    )
    priority_bags_text = "、".join(f"`{bag_id}`" for bag_id in priority_bag_ids)
    worst_camera = day_values["20260814"]["camera_worst"]
    worst_map = day_values["20260814"]["map_worst"]
    lines = [
        "# 0811–0815 同日动态标定：原因分析",
        "",
        "## 结论",
        "",
        "若 tag0 与 tag1 刚性连接，真实 `T_tag0_tag1` 应当恒定。原报告不同组合独立估计得到最大约 0.557° 的角差，说明数据与估计器未满足这一刚性不变量；这是需要优先解释的异常，而不是可以忽略的正常现象。每个子集重新估计 G2 只是暴露异常的测量方式，不是异常的根因。",
        "",
        "本轮二-bag实验无法评估既有 G2 至少六 station 的质量门，因此没有运行 G2，而是固定同一份已批准 Tag 几何并检查固定几何重投影质量。候选间 `T_tag0_tag1` 为数值零是约束导致的结果，只能回答“强制刚性不变量后外参是否仍不稳定”，不能证明结构没有变化，也不能解释原来的 0.557°。",
        "",
        "强制固定 Tag 几何后，0814 的二-bag候选之间，"
        f"`T_ins_camera` 最大角差仍为 {float(worst_camera['rotation_difference_deg']):.3f}° "
        f"（{worst_camera['left_candidate_id']} vs {worst_camera['right_candidate_id']}），"
        f"`T_ins_map_tag0` 最大角差为 {float(worst_map['rotation_difference_deg']):.3f}°；"
        f"Rx/Ry 跨度分别为 {span('20260814', 'rx_correction_deg'):.3f}°/"
        f"{span('20260814', 'ry_correction_deg'):.3f}°。这说明 0814 数据内部还存在明显的 bag/轨迹依赖。",
        "",
        "## 本轮数据口径",
        "",
        f"- 0811/0812/0814 分别有 {day_values['20260811']['within_8m_bags']}/"
        f"{day_values['20260812']['within_8m_bags']}/{day_values['20260814']['within_8m_bags']} 个 bag 含 8 m 内有效帧，"
        f"对应 {day_values['20260811']['within_8m_frames']}/"
        f"{day_values['20260812']['within_8m_frames']}/{day_values['20260814']['within_8m_frames']} 帧。",
        "- 这些日期全程不会导致码牌振动，因此不使用 `safe`、接触前后或结构应力分组；所有基础证据有效且 `0 < camera_tag0_tz_m <= 8.0` 的帧统一进入标定/验证候选池。",
        f"- 8 m 内同日留出的平均 XY/Yaw P80：0811 为 "
        f"{holdout_mean('20260811', 'equal_bag_xy_p80_m'):.3f} m/"
        f"{holdout_mean('20260811', 'equal_bag_yaw_p80_deg'):.3f}°，0812 为 "
        f"{holdout_mean('20260812', 'equal_bag_xy_p80_m'):.3f} m/"
        f"{holdout_mean('20260812', 'equal_bag_yaw_p80_deg'):.3f}°，0814 为 "
        f"{holdout_mean('20260814', 'equal_bag_xy_p80_m'):.3f} m/"
        f"{holdout_mean('20260814', 'equal_bag_yaw_p80_deg'):.3f}°。",
        "",
        "## Bag 依赖",
        "",
        f"0814 中，`{worst_influence_rx['bag_id']}` 对 Rx 的包含/不包含均值差绝对值最大："
        f"{float(worst_influence_rx['rx_correction_deg_included_minus_excluded']):+.3f}°；"
        f"`{worst_influence_ry['bag_id']}` 对 Ry 的包含/不包含均值差绝对值最大："
        f"{float(worst_influence_ry['ry_correction_deg_included_minus_excluded']):+.3f}°。"
        "这些是重叠组合上的描述性关联，不是单 bag 因果效应。",
        "",
        "## 原因判断",
        "",
        "1. **已确认的异常**：自由估计的 `T_tag0_tag1` 对数据组合敏感，最大角差约 0.557°，违反刚性连接所要求的真实几何不变量；根因仍未确认。",
        "2. **已确认的条件消融结果**：固定 `T_tag0_tag1` 后，0814 的 `T_ins_camera` 仍有明显组合差，说明原问题不只存在于 G2，但固定实验不能反向证明 G2 或机械结构正常。",
        "3. **仍需区分的原因**：实际连接松动/弹性形变、双 Tag 角点或 PnP 的视角相关系统误差、平面几何弱可观性，以及轨迹/时间同步偏差都可能造成估计不一致。0811～0815 不存在振动分区，不能再用所谓 safe/结构应力分组作为因果证据。",
        "",
        "## 下一步最小闭环",
        "",
        "- 直接用同一图像内 tag0/tag1 的角点，按 bag 独立估计相对位姿并做帧重采样置信区间；该检查不依赖 INS、地图码牌跨日位置或时间同步。",
        "- 对固定 `T_tag0_tag1` 计算逐帧双 Tag 重投影残差，并按匹配的距离、bearing 和行驶方向比较；若残差或自由估计的相对位姿随时间可重复跳变，再检查连接松动或弹性形变。",
        f"- 优先复查 0814 的 {priority_bags_text} 及其配对，核对图像 bearing、时间同步、车辆姿态和轨迹；不要用跨日码牌位置作验证。",
        f"- 二-bag候选可用于诊断，不建议直接部署；0814 的 "
        f"{float(worst_camera['rotation_difference_deg']):.3f}° 外参组合差已经超过原问题量级。",
        "",
        "详细描述性关联见 `20260814/bag_influence.csv`；逐 bag 轨迹见各日 `validation_same_day/interactive_bags/index.html`。",
    ]
    path = output_dir / "FINDINGS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["findings"] = str(path)
        atomic_write_json(summary_path, summary)
    return path


def _execute_job(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, object],
    online: object,
    experiment: object,
    job_config_sha256: str,
) -> int:
    sampling_values = manifest["two_bag_sampling"]
    experiment = replace(
        experiment,
        sampling=replace(
            experiment.sampling,
            minimum_bag_count=int(sampling_values["minimum_bag_count"]),
            minimum_bags_per_distance_bin=int(
                sampling_values["minimum_bags_per_distance_bin"]
            ),
            minimum_total_frames=int(sampling_values["minimum_total_frames"]),
        ),
        coverage=replace(
            experiment.coverage,
            minimum_bearing_span_deg=float(
                manifest["two_bag_coverage"]["minimum_bearing_span_deg"]
            ),
        ),
    )
    machine_config_dir = Path(online.calibration.machine_config_dir)
    if not machine_config_dir.is_absolute():
        raise ValueError("machine.config_dir must resolve to an absolute path")
    policy = manifest["calibration_policy"]
    args.maximum_depth_m = float(policy["maximum_depth_m"])
    args.maximum_depth_inclusive = bool(policy["maximum_depth_inclusive"])
    args.validation_scope = (
        "ALL_DYNAMIC_LE_8M"
        if args.maximum_depth_inclusive and args.maximum_depth_m == 8.0
        else "ALL_DYNAMIC_LT_8M"
    )
    args.tag_geometry_mode = str(policy["tag_geometry_initialization_mode"])
    args.allow_failed_tag_geometry = bool(
        policy.get("allow_failed_tag_geometry_for_diagnostics", False)
    )
    initializer_path = machine_config_dir / "tag0_tag1_extrinsic.yaml"
    if args.tag_geometry_mode == "bootstrap":
        optimizer = dict(online.calibration.tag_geometry.optimizer)
        optimizer.update(
            minimum_valid_stations=int(
                policy["tag_geometry_minimum_valid_stations"]
            ),
            maximum_views_per_station=int(
                policy["tag_geometry_maximum_views_per_station"]
            ),
            minimum_balanced_views=int(
                policy["tag_geometry_minimum_balanced_views"]
            ),
        )
        online = replace(
            online,
            calibration=replace(
                online.calibration,
                tag_geometry=replace(
                    online.calibration.tag_geometry,
                    initialization_mode="bootstrap",
                    reuse_initial_path=None,
                    optimizer=optimizer,
                ),
            ),
        )
        args.freeze_reused_tag_geometry = False
        approved_tag_geometry_sha256 = None
    else:
        online = replace(
            online,
            calibration=replace(
                online.calibration,
                tag_geometry=replace(
                    online.calibration.tag_geometry,
                    initialization_mode="reuse",
                    reuse_initial_path=initializer_path,
                ),
            ),
        )
        args.freeze_reused_tag_geometry = True
        approved_tag_geometry_sha256 = sha256_file(initializer_path)
    run_identity = {
        "pipeline_software_sha256": _pipeline_software_fingerprint(),
        "online_config_sha256": online.source_fingerprint,
        "base_experiment_config_sha256": experiment.source_fingerprint,
        "same_day_manifest_sha256": job_config_sha256,
        "job_config_sha256": job_config_sha256,
        "approved_tag_geometry_sha256": approved_tag_geometry_sha256,
        "calibration_policy": dict(policy),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_completion_artifacts(
        args.output_dir,
        tuple(str(date) for date in manifest["datasets"]),
    )
    summaries = []
    for date, spec in sorted(manifest["datasets"].items()):
        summaries.append(
            _run_date(
                date=str(date),
                spec=spec,
                args=args,
                online=online,
                experiment=experiment,
                machine_config_dir=machine_config_dir,
                run_identity=run_identity,
            )
        )
    report = _write_root_report(
        args.output_dir,
        summaries,
        title=str(
            manifest.get(
                "report_title",
                "20260811–20260815 两-bag 动态标定与同日验证",
            )
        ),
        maximum_depth_m=args.maximum_depth_m,
        maximum_depth_inclusive=args.maximum_depth_inclusive,
        tag_geometry_mode=args.tag_geometry_mode,
    )
    findings = None
    if set(str(date) for date in manifest["datasets"]) == {
        "20260811",
        "20260812",
        "20260813",
        "20260814",
        "20260815",
    } and _can_write_completed_findings(
        summaries,
        inventory_only=bool(args.inventory_only),
    ):
        findings = _write_completed_findings(args.output_dir)
    experiment_status = _experiment_completion_status(
        summaries,
        inventory_only=bool(args.inventory_only),
    )
    payload = {
        "schema_version": 1,
        "status": experiment_status,
        "cross_date_validation_prohibited": True,
        "dates": summaries,
        "report": str(report),
        "findings": None if findings is None else str(findings),
    }
    atomic_write_json(args.output_dir / "summary.json", payload)
    _progress("same_day_experiment_complete", **payload)
    return 0


def run_from_config(config_path: str | Path) -> int:
    """Run one calibration job from the public single-file configuration."""
    job = load_job_config(config_path)
    try:
        import pyapriltags  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "dynamic calibration requires the declared pyapriltags dependency"
        ) from error
    args = argparse.Namespace(
        output_dir=job.output_dir,
        inventory_only=job.inventory_only,
        skip_validation=job.skip_validation,
        novatel_message_dir=job.novatel_message_dir,
        bag_paths=job.bag_paths,
    )
    with job.prepare_runtime() as prepared:
        return _execute_job(
            args=args,
            manifest=prepared.manifest,
            online=prepared.online,
            experiment=prepared.experiment,
            job_config_sha256=job.source_fingerprint,
        )


def main() -> int:
    args = build_parser().parse_args()
    return run_from_config(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
