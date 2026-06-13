from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ScoreDistribution:
    values: tuple[float, ...]
    leader: float
    runner_up: float
    upper_quartile: float
    median: float
    lower_quartile: float
    local_gap: float
    spread: float
    adaptive_clear_margin: float
    adaptive_support_margin: float
    support_cluster_size: int
    dominance_ratio: float


def percentile(values: Sequence[float], q: float) -> float:
    """计算分位数。"""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(float(q), 1.0)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_scores(
    values: Sequence[float],
    *,
    fallback_clear_margin: float,
    min_clear_margin: float,
    max_clear_margin: float,
    min_support_margin: float,
    max_support_margin: float,
) -> ScoreDistribution:
    """汇总一组分数的分布特征。"""
    ordered = sorted((float(value) for value in values), reverse=True)
    if not ordered:
        clear_margin = max(min_clear_margin, min(max_clear_margin, float(fallback_clear_margin)))
        support_margin = max(min_support_margin, min(max_support_margin, clear_margin * 0.75))
        return ScoreDistribution(
            values=(),
            leader=0.0,
            runner_up=0.0,
            upper_quartile=0.0,
            median=0.0,
            lower_quartile=0.0,
            local_gap=0.0,
            spread=0.0,
            adaptive_clear_margin=clear_margin,
            adaptive_support_margin=support_margin,
            support_cluster_size=0,
            dominance_ratio=0.0,
        )

    leader = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else ordered[0]
    upper_quartile = percentile(ordered, 0.75)
    median = percentile(ordered, 0.5)
    lower_quartile = percentile(ordered, 0.25)
    local_gap = max(leader - runner_up, 0.0)
    spread = max(leader - median, upper_quartile - lower_quartile, 0.0)

    clear_margin = min(
        max_clear_margin,
        max(min_clear_margin, float(fallback_clear_margin) * 0.5, spread * 0.75),
    )
    support_margin = min(
        max_support_margin,
        max(min_support_margin, clear_margin * 0.7, spread * 0.5),
    )
    support_cluster_size = sum(1 for value in ordered if leader - value <= support_margin + 1e-9)
    dominance_ratio = local_gap / clear_margin if clear_margin > 0 else 0.0

    return ScoreDistribution(
        values=tuple(ordered),
        leader=leader,
        runner_up=runner_up,
        upper_quartile=upper_quartile,
        median=median,
        lower_quartile=lower_quartile,
        local_gap=local_gap,
        spread=spread,
        adaptive_clear_margin=clear_margin,
        adaptive_support_margin=support_margin,
        support_cluster_size=support_cluster_size,
        dominance_ratio=dominance_ratio,
    )
