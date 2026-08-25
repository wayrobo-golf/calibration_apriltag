"""Pure FR-05 gate for comparing IPPE candidates with a dual-Tag pose.

The vision adapter remains responsible for running PnP and calculating the
translation/rotation differences.  This module owns the acceptance semantics:
do not force a preferred planar branch; any positive-depth IPPE solution may
pass, and both Tags must have a passing solution in the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IppeCandidateComparison:
    positive_depth: bool
    translation_error_m: float
    rotation_error_deg: float


@dataclass(frozen=True)
class TagPnpGateResult:
    passed: bool
    passing_candidate_indices: tuple[int, ...]
    positive_depth_candidate_count: int


@dataclass(frozen=True)
class DualTagPnpGateResult:
    passed: bool
    tag0: TagPnpGateResult
    tag1: TagPnpGateResult


def evaluate_tag_candidates(
    candidates: Iterable[IppeCandidateComparison],
    maximum_translation_error_m: float = 0.10,
    maximum_rotation_error_deg: float = 2.0,
) -> TagPnpGateResult:
    """Accept a Tag when any positive-depth candidate meets both thresholds."""

    values = tuple(candidates)
    positive_count = sum(candidate.positive_depth for candidate in values)
    passing = tuple(
        index
        for index, candidate in enumerate(values)
        if candidate.positive_depth
        and candidate.translation_error_m <= maximum_translation_error_m
        and candidate.rotation_error_deg <= maximum_rotation_error_deg
    )
    return TagPnpGateResult(bool(passing), passing, positive_count)


def evaluate_dual_tag_frame(
    tag0_candidates: Iterable[IppeCandidateComparison],
    tag1_candidates: Iterable[IppeCandidateComparison],
    maximum_translation_error_m: float = 0.10,
    maximum_rotation_error_deg: float = 2.0,
) -> DualTagPnpGateResult:
    """Apply FR-05 independently to Tag0 and Tag1, then require both Tags."""

    tag0 = evaluate_tag_candidates(
        tag0_candidates,
        maximum_translation_error_m,
        maximum_rotation_error_deg,
    )
    tag1 = evaluate_tag_candidates(
        tag1_candidates,
        maximum_translation_error_m,
        maximum_rotation_error_deg,
    )
    return DualTagPnpGateResult(tag0.passed and tag1.passed, tag0, tag1)
