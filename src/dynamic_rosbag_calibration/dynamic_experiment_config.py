"""Strict configuration loader for offline safe-dynamic experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .dynamic_experiment_types import (
    DynamicBagGroups,
    DynamicCoverageConfig,
    DynamicExperimentConfig,
    DynamicExperimentTopics,
    DynamicSamplingConfig,
    PoseInterpolationConfig,
    StructuralGateConfig,
)
from .io import sha256_file


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _StrictLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ValueError(f"{name} has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique non-empty identifiers")
    return result


def load_dynamic_experiment_config(path: Path) -> DynamicExperimentConfig:
    """Load one strict schema-v1 safe-dynamic experiment configuration."""
    source = Path(path)
    payload = yaml.load(source.read_text(encoding="utf-8"), Loader=_StrictLoader)
    root = _mapping(payload, "dynamic experiment config")
    _exact(
        root,
        {
            "schema_version",
            "dataset_name",
            "topics",
            "bags",
            "interpolation",
            "structural_gate",
            "sampling",
            "coverage",
        },
        "dynamic experiment config",
    )
    if root["schema_version"] != 1:
        raise ValueError("only dynamic experiment schema_version 1 is supported")
    dataset_name = str(root["dataset_name"])
    if not dataset_name or "/" in dataset_name or "\\" in dataset_name:
        raise ValueError("dataset_name must be a non-empty path-safe identifier")

    topics = _mapping(root["topics"], "topics")
    _exact(topics, {"image", "odom", "inspvax", "imu"}, "topics")

    bags = _mapping(root["bags"], "bags")
    _exact(bags, {"training", "holdout_no_bridge", "excluded"}, "bags")
    excluded_value = _mapping(bags["excluded"], "bags.excluded")

    interpolation = _mapping(root["interpolation"], "interpolation")
    _exact(
        interpolation,
        {"maximum_bracket_gap_ms", "maximum_endpoint_distance_ms"},
        "interpolation",
    )

    structural = _mapping(root["structural_gate"], "structural_gate")
    structural_fields = {
        "minimum_safe_depth_m",
        "maximum_depth_m",
        "contact_candidate_depth_m",
        "acceleration_highpass_limit_mps2",
        "angular_rate_highpass_limit_degps",
        "highpass_window_s",
        "contact_confirmation_s",
    }
    _exact(structural, structural_fields, "structural_gate")

    sampling = _mapping(root["sampling"], "sampling")
    sampling_fields = {
        "minimum_xy_increment_m",
        "minimum_yaw_increment_deg",
        "maximum_time_increment_s",
        "maximum_frames_per_cell",
        "maximum_frames_per_bag",
        "minimum_bag_count",
        "minimum_bags_per_distance_bin",
        "minimum_total_frames",
    }
    _exact(sampling, sampling_fields, "sampling")

    coverage = _mapping(root["coverage"], "coverage")
    _exact(
        coverage,
        {"distance_bin_edges_m", "minimum_bearing_span_deg"},
        "coverage",
    )
    edges = coverage["distance_bin_edges_m"]
    if not isinstance(edges, list) or len(edges) != 2:
        raise ValueError("distance bin edges must contain exactly two values")

    return DynamicExperimentConfig(
        schema_version=1,
        dataset_name=dataset_name,
        topics=DynamicExperimentTopics(
            image=str(topics["image"]),
            odom=str(topics["odom"]),
            inspvax=str(topics["inspvax"]),
            imu=str(topics["imu"]),
        ),
        bags=DynamicBagGroups(
            training=_string_list(bags["training"], "bags.training"),
            holdout_no_bridge=_string_list(
                bags["holdout_no_bridge"],
                "bags.holdout_no_bridge",
            ),
            excluded=tuple(
                (str(key), str(reason)) for key, reason in excluded_value.items()
            ),
        ),
        interpolation=PoseInterpolationConfig(
            maximum_bracket_gap_ms=_number(
                interpolation["maximum_bracket_gap_ms"],
                "interpolation.maximum_bracket_gap_ms",
            ),
            maximum_endpoint_distance_ms=_number(
                interpolation["maximum_endpoint_distance_ms"],
                "interpolation.maximum_endpoint_distance_ms",
            ),
        ),
        structural_gate=StructuralGateConfig(
            **{
                field: _number(
                    structural[field],
                    f"structural_gate.{field}",
                )
                for field in structural_fields
            }
        ),
        sampling=DynamicSamplingConfig(
            minimum_xy_increment_m=_number(
                sampling["minimum_xy_increment_m"],
                "sampling.minimum_xy_increment_m",
            ),
            minimum_yaw_increment_deg=_number(
                sampling["minimum_yaw_increment_deg"],
                "sampling.minimum_yaw_increment_deg",
            ),
            maximum_time_increment_s=_number(
                sampling["maximum_time_increment_s"],
                "sampling.maximum_time_increment_s",
            ),
            maximum_frames_per_cell=_positive_integer(
                sampling["maximum_frames_per_cell"],
                "sampling.maximum_frames_per_cell",
            ),
            maximum_frames_per_bag=_positive_integer(
                sampling["maximum_frames_per_bag"],
                "sampling.maximum_frames_per_bag",
            ),
            minimum_bag_count=_positive_integer(
                sampling["minimum_bag_count"],
                "sampling.minimum_bag_count",
            ),
            minimum_bags_per_distance_bin=_positive_integer(
                sampling["minimum_bags_per_distance_bin"],
                "sampling.minimum_bags_per_distance_bin",
            ),
            minimum_total_frames=_positive_integer(
                sampling["minimum_total_frames"],
                "sampling.minimum_total_frames",
            ),
        ),
        coverage=DynamicCoverageConfig(
            distance_bin_edges_m=(
                _number(edges[0], "coverage.distance_bin_edges_m[0]"),
                _number(edges[1], "coverage.distance_bin_edges_m[1]"),
            ),
            minimum_bearing_span_deg=_number(
                coverage["minimum_bearing_span_deg"],
                "coverage.minimum_bearing_span_deg",
            ),
        ),
        source_fingerprint=sha256_file(source),
    )

