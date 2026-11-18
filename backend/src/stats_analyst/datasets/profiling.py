"""Lightweight descriptive profiles that give the agent (and the user) a map of the data."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from stats_analyst.stats.results import json_safe
from stats_analyst.stats.validation import require_columns

_MAX_LEVELS = 15


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = json_safe(
        {
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "columns": [_column_overview(df[column]) for column in df.columns],
        }
    )
    return profile


def profile_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    require_columns(df, column)
    series = df[column]
    profile = _column_overview(series)
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        described = series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        profile["summary"] = {str(k): float(v) for k, v in described.astype(float).items()}
        profile["skewness"] = float(stats.skew(series.dropna().to_numpy(dtype=float)))
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        profile["n_outliers_iqr"] = int(
            ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        )
    else:
        counts = series.astype(str).value_counts().head(_MAX_LEVELS)
        profile["top_levels"] = {str(k): int(v) for k, v in counts.items()}
    cleaned: dict[str, Any] = json_safe(profile)
    return cleaned


def _column_overview(series: pd.Series) -> dict[str, Any]:
    n_unique = int(series.nunique(dropna=True))
    overview: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "kind": _kind(series, n_unique),
        "n_missing": int(series.isna().sum()),
        "n_unique": n_unique,
    }
    if overview["kind"] in {"categorical", "binary"} and n_unique <= _MAX_LEVELS:
        overview["levels"] = sorted(map(str, series.dropna().unique().tolist()))
    return overview


def _kind(series: pd.Series, n_unique: int) -> str:
    if n_unique == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"
