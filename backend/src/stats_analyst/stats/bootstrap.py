"""Distribution-free inference via resampling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from stats_analyst.stats.results import TestResult
from stats_analyst.stats.validation import (
    StatisticalInputError,
    group_values,
    require_alpha,
    require_columns,
    require_numeric,
)

Statistic = Literal["mean", "median"]
_MAX_RESAMPLES = 20_000


def bootstrap_difference(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    group_a: str,
    group_b: str,
    statistic: Statistic = "mean",
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Percentile bootstrap CI for statistic(A) - statistic(B), plus a permutation p-value."""
    require_alpha(alpha)
    require_columns(df, value_column, group_column)
    require_numeric(df, value_column)
    if not 100 <= n_resamples <= _MAX_RESAMPLES:
        raise StatisticalInputError(f"n_resamples must be between 100 and {_MAX_RESAMPLES}.")
    reducers: dict[str, Callable[..., Any]] = {"mean": np.mean, "median": np.median}
    func = reducers.get(statistic)
    if func is None:
        raise StatisticalInputError("statistic must be 'mean' or 'median'.")

    a = group_values(df, value_column, group_column, group_a)
    b = group_values(df, value_column, group_column, group_b)
    rng = np.random.default_rng(seed)
    observed = float(func(a) - func(b))

    boot_a = func(rng.choice(a, size=(n_resamples, a.size), replace=True), axis=1)
    boot_b = func(rng.choice(b, size=(n_resamples, b.size), replace=True), axis=1)
    diffs = boot_a - boot_b
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])

    # Permutation test under H0: both groups come from the same distribution.
    pooled = np.concatenate([a, b])
    permuted = np.array([rng.permutation(pooled) for _ in range(n_resamples)])
    perm_diffs = func(permuted[:, : a.size], axis=1) - func(permuted[:, a.size :], axis=1)
    p_value = float((np.sum(np.abs(perm_diffs) >= abs(observed)) + 1) / (n_resamples + 1))

    return TestResult(
        test=f"bootstrap_{statistic}_difference",
        statistic=observed,
        p_value=p_value,
        alpha=alpha,
        effect_size={f"{statistic}_difference": observed},
        confidence_interval={"low": float(low), "high": float(high)},
        details={
            "n_resamples": n_resamples,
            "seed": seed,
            "group_a": {"group": group_a, "n": int(a.size), statistic: float(func(a))},
            "group_b": {"group": group_b, "n": int(b.size), statistic: float(func(b))},
            "bootstrap_standard_error": float(diffs.std(ddof=1)),
            "p_value_method": "two-sided permutation test",
        },
    )
