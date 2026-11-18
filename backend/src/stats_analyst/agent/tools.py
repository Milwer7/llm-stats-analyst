"""Tools the agent can call. Each tool wraps a pure function bound to one dataset."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from stats_analyst import stats
from stats_analyst.datasets import profile_column, profile_dataset
from stats_analyst.llm import ToolSpec
from stats_analyst.stats.validation import StatisticalInputError

ToolHandler = Callable[[pd.DataFrame, dict[str, Any]], dict[str, Any]]

_ALPHA = {"type": "number", "description": "Significance level, default 0.05.", "default": 0.05}


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _str(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: ToolHandler


def _test(func: Callable[..., stats.TestResult]) -> ToolHandler:
    return lambda df, args: func(df, **args).to_dict()


TOOLS: dict[str, Tool] = {
    tool.spec.name: tool
    for tool in [
        Tool(
            ToolSpec(
                name="describe_dataset",
                description="List columns with their type, missing values and levels. Call first.",
                parameters=_schema({}, []),
            ),
            lambda df, _: profile_dataset(df),
        ),
        Tool(
            ToolSpec(
                name="describe_column",
                description="Detailed profile of one column: quantiles, skewness and outliers "
                "for numeric columns, level counts for categorical ones.",
                parameters=_schema({"column": _str("Column name.")}, ["column"]),
            ),
            lambda df, args: profile_column(df, **args),
        ),
        Tool(
            ToolSpec(
                name="compare_means",
                description="Compare a numeric outcome between two groups. method='auto' checks "
                "normality and picks Welch's t-test or Mann-Whitney U. Returns effect sizes.",
                parameters=_schema(
                    {
                        "value_column": _str("Numeric outcome column."),
                        "group_column": _str("Column that defines the groups."),
                        "group_a": _str("First group label."),
                        "group_b": _str("Second group label."),
                        "method": {"type": "string", "enum": ["auto", "welch", "mann_whitney"]},
                        "alpha": _ALPHA,
                    },
                    ["value_column", "group_column", "group_a", "group_b"],
                ),
            ),
            _test(stats.compare_means),
        ),
        Tool(
            ToolSpec(
                name="compare_many_groups",
                description="Compare a numeric outcome across 3+ groups (one-way ANOVA with "
                "Levene check, plus Kruskal-Wallis).",
                parameters=_schema(
                    {
                        "value_column": _str("Numeric outcome column."),
                        "group_column": _str("Column that defines the groups."),
                        "alpha": _ALPHA,
                    },
                    ["value_column", "group_column"],
                ),
            ),
            _test(stats.compare_many_groups),
        ),
        Tool(
            ToolSpec(
                name="ab_test_proportions",
                description="A/B test for a binary outcome (e.g. conversion): two-proportion "
                "z-test with absolute and relative lift and a CI for the difference.",
                parameters=_schema(
                    {
                        "outcome_column": _str("Binary outcome column (0/1, true/false, yes/no)."),
                        "group_column": _str("Column with the variant label."),
                        "control": _str("Control group label."),
                        "treatment": _str("Treatment group label."),
                        "alpha": _ALPHA,
                    },
                    ["outcome_column", "group_column", "control", "treatment"],
                ),
            ),
            _test(stats.two_proportion_test),
        ),
        Tool(
            ToolSpec(
                name="bootstrap_difference",
                description="Distribution-free comparison of two groups: bootstrap CI for the "
                "difference in mean or median, and a permutation-test p-value. Use when data "
                "is skewed or has outliers.",
                parameters=_schema(
                    {
                        "value_column": _str("Numeric outcome column."),
                        "group_column": _str("Column that defines the groups."),
                        "group_a": _str("First group label."),
                        "group_b": _str("Second group label."),
                        "statistic": {"type": "string", "enum": ["mean", "median"]},
                        "n_resamples": {"type": "integer", "minimum": 100, "maximum": 20000},
                        "alpha": _ALPHA,
                    },
                    ["value_column", "group_column", "group_a", "group_b"],
                ),
            ),
            _test(stats.bootstrap_difference),
        ),
        Tool(
            ToolSpec(
                name="chi_square_independence",
                description="Chi-square test of independence between two categorical columns, "
                "with Cramer's V.",
                parameters=_schema(
                    {
                        "column_a": _str("First categorical column."),
                        "column_b": _str("Second categorical column."),
                        "alpha": _ALPHA,
                    },
                    ["column_a", "column_b"],
                ),
            ),
            _test(stats.chi_square_independence),
        ),
        Tool(
            ToolSpec(
                name="correlation",
                description="Pearson (linear) or Spearman (monotonic, robust) correlation "
                "between two numeric columns.",
                parameters=_schema(
                    {
                        "column_x": _str("First numeric column."),
                        "column_y": _str("Second numeric column."),
                        "method": {"type": "string", "enum": ["pearson", "spearman"]},
                        "alpha": _ALPHA,
                    },
                    ["column_x", "column_y"],
                ),
            ),
            _test(stats.correlation),
        ),
    ]
}


def tool_specs() -> list[ToolSpec]:
    return [tool.spec for tool in TOOLS.values()]


@dataclass(frozen=True)
class ToolOutcome:
    output: dict[str, Any]
    is_error: bool

    def to_json(self) -> str:
        return json.dumps(self.output, default=str)


def run_tool(name: str, arguments: dict[str, Any], df: pd.DataFrame) -> ToolOutcome:
    """Execute a tool, turning bad arguments into an error the model can recover from."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolOutcome({"error": f"Unknown tool '{name}'. Available: {list(TOOLS)}"}, True)
    allowed = set(tool.spec.parameters["properties"])
    unexpected = set(arguments) - allowed
    if unexpected:
        return ToolOutcome(
            {"error": f"Unexpected arguments {sorted(unexpected)} for '{name}'."}, True
        )
    try:
        return ToolOutcome(tool.handler(df, arguments), False)
    except (StatisticalInputError, TypeError) as exc:
        return ToolOutcome({"error": str(exc)}, True)
