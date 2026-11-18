"""Tests that compare a numeric outcome across groups."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from stats_analyst.stats.results import TestResult
from stats_analyst.stats.validation import (
    StatisticalInputError,
    group_values,
    require_alpha,
    require_columns,
    require_numeric,
)

MeansMethod = Literal["auto", "welch", "mann_whitney"]

# Shapiro-Wilk becomes over-sensitive with large samples; past this size we rely on the CLT.
_NORMALITY_MAX_N = 5000
_CLT_MIN_N = 30


def compare_means(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    group_a: str,
    group_b: str,
    method: MeansMethod = "auto",
    alpha: float = 0.05,
) -> TestResult:
    """Compare a numeric outcome between two groups.

    With method="auto", Welch's t-test is used when both groups are plausibly normal
    or large enough for the central limit theorem; otherwise Mann-Whitney U is used.
    """
    require_alpha(alpha)
    require_columns(df, value_column, group_column)
    require_numeric(df, value_column)
    a = group_values(df, value_column, group_column, group_a)
    b = group_values(df, value_column, group_column, group_b)

    normality = {group_a: _normality_p_value(a), group_b: _normality_p_value(b)}
    warnings: list[str] = []
    chosen = method
    if method == "auto":
        chosen = (
            "welch"
            if _parametric_ok(a, normality[group_a], alpha)
            and _parametric_ok(b, normality[group_b], alpha)
            else "mann_whitney"
        )
    elif method == "welch" and not (
        _parametric_ok(a, normality[group_a], alpha)
        and _parametric_ok(b, normality[group_b], alpha)
    ):
        warnings.append("Normality is doubtful and samples are small; consider Mann-Whitney U.")

    details = {
        "group_a": _summary(group_a, a),
        "group_b": _summary(group_b, b),
        "normality_shapiro_p": normality,
        "method_selected": chosen,
    }

    if chosen == "welch":
        result = stats.ttest_ind(a, b, equal_var=False)
        ci = result.confidence_interval(confidence_level=1 - alpha)
        return TestResult(
            test="welch_t_test",
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            alpha=alpha,
            effect_size={
                "mean_difference": float(a.mean() - b.mean()),
                "cohens_d": _cohens_d(a, b),
            },
            confidence_interval={"low": float(ci.low), "high": float(ci.high)},
            details={**details, "degrees_of_freedom": float(result.df)},
            warnings=warnings,
        )

    if chosen == "mann_whitney":
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        u = float(result.statistic)
        # Rank-biserial correlation: P(A > B) - P(A < B).
        rank_biserial = 2 * u / (a.size * b.size) - 1
        return TestResult(
            test="mann_whitney_u",
            statistic=u,
            p_value=float(result.pvalue),
            alpha=alpha,
            effect_size={
                "median_difference": float(np.median(a) - np.median(b)),
                "rank_biserial": float(rank_biserial),
            },
            details=details,
            warnings=warnings,
        )

    raise StatisticalInputError(f"Unknown method '{method}'.")


def compare_many_groups(
    df: pd.DataFrame, value_column: str, group_column: str, alpha: float = 0.05
) -> TestResult:
    """One-way ANOVA (Welch-robust check via Levene) plus Kruskal-Wallis for 3+ groups."""
    require_alpha(alpha)
    require_columns(df, value_column, group_column)
    require_numeric(df, value_column)
    levels = sorted(df[group_column].dropna().astype(str).unique().tolist())
    if len(levels) < 2:
        raise StatisticalInputError(f"Column '{group_column}' needs at least 2 groups.")
    samples = [group_values(df, value_column, group_column, level) for level in levels]

    anova = stats.f_oneway(*samples)
    kruskal = stats.kruskal(*samples)
    levene = stats.levene(*samples)

    grand_mean = np.concatenate(samples).mean()
    ss_between = sum(s.size * (s.mean() - grand_mean) ** 2 for s in samples)
    ss_total = sum(((s - grand_mean) ** 2).sum() for s in samples)
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

    warnings = []
    if levene.pvalue < alpha:
        warnings.append("Levene's test rejects equal variances; prefer the Kruskal-Wallis result.")
    return TestResult(
        test="one_way_anova",
        statistic=float(anova.statistic),
        p_value=float(anova.pvalue),
        alpha=alpha,
        effect_size={"eta_squared": eta_squared},
        details={
            "groups": [_summary(level, s) for level, s in zip(levels, samples, strict=True)],
            "kruskal_wallis": {
                "statistic": float(kruskal.statistic),
                "p_value": float(kruskal.pvalue),
            },
            "levene_p_value": float(levene.pvalue),
        },
        warnings=warnings,
    )


def _normality_p_value(values: np.ndarray) -> float | None:
    if values.size < 3 or values.size > _NORMALITY_MAX_N or np.ptp(values) == 0:
        return None
    return float(stats.shapiro(values).pvalue)


def _parametric_ok(values: np.ndarray, normality_p: float | None, alpha: float) -> bool:
    if values.size >= _CLT_MIN_N:
        return True
    return normality_p is not None and normality_p >= alpha


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _summary(name: str, values: np.ndarray) -> dict[str, float | int | str]:
    return {
        "group": name,
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)),
    }
