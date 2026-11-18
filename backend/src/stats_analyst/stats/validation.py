"""Input validation helpers shared by the statistical routines."""

from __future__ import annotations

import numpy as np
import pandas as pd


class StatisticalInputError(ValueError):
    """Raised when the data or the arguments cannot support the requested procedure."""


def require_columns(df: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise StatisticalInputError(f"Unknown column(s): {missing}. Available: {available}")


def require_numeric(df: pd.DataFrame, column: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise StatisticalInputError(f"Column '{column}' must be numeric (got {df[column].dtype}).")


def group_values(df: pd.DataFrame, value_column: str, group_column: str, group: str) -> np.ndarray:
    """Return the non-null values of `value_column` for one level of `group_column`."""
    labels = df[group_column].astype(str)
    if group not in set(labels):
        levels = sorted(labels.dropna().unique().tolist())
        raise StatisticalInputError(
            f"Group '{group}' not found in column '{group_column}'. Levels: {levels}"
        )
    values = df.loc[labels == group, value_column].dropna().to_numpy(dtype=float)
    if values.size < 2:
        raise StatisticalInputError(f"Group '{group}' has fewer than 2 non-null observations.")
    return values


def require_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise StatisticalInputError("alpha must be between 0 and 1.")
