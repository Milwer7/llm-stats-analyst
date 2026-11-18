"""A/B testing for binary (conversion-style) outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from stats_analyst.stats.results import TestResult
from stats_analyst.stats.validation import StatisticalInputError, require_alpha, require_columns

_TRUE_LABELS = {"1", "true", "yes", "y", "t", "converted"}
_FALSE_LABELS = {"0", "false", "no", "n", "f", "not_converted"}


def two_proportion_test(
    df: pd.DataFrame,
    outcome_column: str,
    group_column: str,
    control: str,
    treatment: str,
    alpha: float = 0.05,
) -> TestResult:
    """Two-sided two-proportion z-test of treatment vs control conversion rates."""
    require_alpha(alpha)
    require_columns(df, outcome_column, group_column)
    outcome = _as_binary(df[outcome_column], outcome_column)
    labels = df[group_column].astype(str)

    counts = {}
    for name in (control, treatment):
        mask = (labels == name) & outcome.notna()
        if not mask.any():
            levels = sorted(labels.unique().tolist())
            raise StatisticalInputError(
                f"Group '{name}' not found in '{group_column}'. Levels: {levels}"
            )
        counts[name] = (int(outcome[mask].sum()), int(mask.sum()))

    (x_c, n_c), (x_t, n_t) = counts[control], counts[treatment]
    p_c, p_t = x_c / n_c, x_t / n_t
    pooled = (x_c + x_t) / (n_c + n_t)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))

    # Unpooled standard error for the confidence interval of the difference.
    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c

    warnings = []
    if min(x_c, n_c - x_c, x_t, n_t - x_t) < 10:
        warnings.append(
            "Fewer than 10 successes or failures in a group; "
            "the normal approximation is unreliable."
        )

    return TestResult(
        test="two_proportion_z_test",
        statistic=float(z),
        p_value=p_value,
        alpha=alpha,
        effect_size={
            "absolute_lift": float(diff),
            "relative_lift": float(diff / p_c) if p_c > 0 else float("nan"),
            "cohens_h": float(2 * np.arcsin(np.sqrt(p_t)) - 2 * np.arcsin(np.sqrt(p_c))),
        },
        confidence_interval={
            "low": float(diff - z_crit * se_diff),
            "high": float(diff + z_crit * se_diff),
        },
        details={
            "control": {"group": control, "conversions": x_c, "n": n_c, "rate": p_c},
            "treatment": {"group": treatment, "conversions": x_t, "n": n_t, "rate": p_t},
        },
        warnings=warnings,
    )


def _as_binary(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique().tolist())
        if values <= {0, 1}:
            return series.astype(float)
        raise StatisticalInputError(
            f"Column '{name}' must be binary (0/1); found {sorted(values)[:5]}."
        )
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        lambda v: 1.0 if v in _TRUE_LABELS else 0.0 if v in _FALSE_LABELS else np.nan
    )
    if mapped.isna().all():
        raise StatisticalInputError(
            f"Column '{name}' could not be interpreted as a binary outcome."
        )
    return mapped
