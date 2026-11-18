"""Association between two variables."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from stats_analyst.stats.results import TestResult
from stats_analyst.stats.validation import (
    StatisticalInputError,
    require_alpha,
    require_columns,
    require_numeric,
)

CorrelationMethod = Literal["pearson", "spearman"]
_MAX_CATEGORIES = 50


def chi_square_independence(
    df: pd.DataFrame, column_a: str, column_b: str, alpha: float = 0.05
) -> TestResult:
    """Pearson chi-square test of independence between two categorical columns."""
    require_alpha(alpha)
    require_columns(df, column_a, column_b)
    table = pd.crosstab(df[column_a].astype(str), df[column_b].astype(str))
    if min(table.shape) < 2:
        raise StatisticalInputError("Both columns need at least 2 categories.")
    if max(table.shape) > _MAX_CATEGORIES:
        raise StatisticalInputError(
            f"Too many categories (>{_MAX_CATEGORIES}) for a chi-square test."
        )

    result = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(result.statistic / (n * (min(table.shape) - 1)))
    warnings = []
    if (result.expected_freq < 5).mean() > 0.2:
        warnings.append(
            "More than 20% of expected counts are below 5; "
            "the chi-square approximation is unreliable."
        )
    return TestResult(
        test="chi_square_independence",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alpha=alpha,
        effect_size={"cramers_v": float(cramers_v)},
        details={
            "degrees_of_freedom": int(result.dof),
            "contingency_table": {str(k): v for k, v in table.to_dict(orient="index").items()},
        },
        warnings=warnings,
    )


def correlation(
    df: pd.DataFrame,
    column_x: str,
    column_y: str,
    method: CorrelationMethod = "pearson",
    alpha: float = 0.05,
) -> TestResult:
    """Pearson or Spearman correlation with significance test."""
    require_alpha(alpha)
    require_columns(df, column_x, column_y)
    require_numeric(df, column_x)
    require_numeric(df, column_y)
    pair = df[[column_x, column_y]].dropna()
    if len(pair) < 3:
        raise StatisticalInputError("At least 3 complete pairs are required.")
    x, y = pair[column_x].to_numpy(dtype=float), pair[column_y].to_numpy(dtype=float)

    if method == "pearson":
        result = stats.pearsonr(x, y)
        ci = result.confidence_interval(confidence_level=1 - alpha)
        interval: dict[str, float] | None = {"low": float(ci.low), "high": float(ci.high)}
    elif method == "spearman":
        result = stats.spearmanr(x, y)
        interval = None
    else:
        raise StatisticalInputError("method must be 'pearson' or 'spearman'.")

    r = float(result.statistic)
    return TestResult(
        test=f"{method}_correlation",
        statistic=r,
        p_value=float(result.pvalue),
        alpha=alpha,
        effect_size={"r": r, "r_squared": r**2},
        confidence_interval=interval,
        details={"n_pairs": len(pair)},
        warnings=["Correlation does not imply causation."],
    )
