from __future__ import annotations

import pandas as pd

from stats_analyst.agent import StatisticalAnalyst
from stats_analyst.agent.tools import run_tool
from tests.fakes import ScriptedLLM, final_turn, tool_turn


def test_agent_runs_tools_and_returns_final_answer(experiment: pd.DataFrame) -> None:
    llm = ScriptedLLM(
        [
            tool_turn("describe_dataset"),
            tool_turn(
                "ab_test_proportions",
                outcome_column="converted",
                group_column="group",
                control="control",
                treatment="treatment",
            ),
            final_turn("**Answer**: treatment converts better."),
        ]
    )
    result = StatisticalAnalyst(llm).analyze(experiment, "Does treatment convert better?")

    assert result.completed
    assert result.answer.startswith("**Answer**")
    assert [step.tool for step in result.steps] == ["describe_dataset", "ab_test_proportions"]
    assert result.steps[1].output["significant"] is True
    # The tool result must be fed back to the model on the following turn.
    last_sent = llm.calls[-1][-1]
    assert last_sent.role == "tool" and last_sent.tool_name == "ab_test_proportions"


def test_tool_errors_are_returned_to_the_model(experiment: pd.DataFrame) -> None:
    llm = ScriptedLLM([tool_turn("describe_column", column="does_not_exist"), final_turn("Sorry.")])
    result = StatisticalAnalyst(llm).analyze(experiment, "Describe a column")
    assert result.steps[0].is_error
    assert "Unknown column" in result.steps[0].output["error"]


def test_step_limit_marks_result_incomplete(experiment: pd.DataFrame) -> None:
    llm = ScriptedLLM([tool_turn("describe_dataset") for _ in range(3)])
    result = StatisticalAnalyst(llm, max_steps=3).analyze(experiment, "Loop forever")
    assert not result.completed
    assert len(result.steps) == 3


def test_run_tool_rejects_unknown_tools_and_arguments(experiment: pd.DataFrame) -> None:
    assert run_tool("drop_table", {}, experiment).is_error
    outcome = run_tool("describe_dataset", {"sql": "x"}, experiment)
    assert outcome.is_error and "Unexpected" in outcome.output["error"]
