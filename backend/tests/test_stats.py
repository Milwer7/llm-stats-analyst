from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp

from stats_analyst import stats
from stats_analyst.stats.validation import StatisticalInputError


class TestCompareMeans:
    def test_detects_shift_with_welch_for_large_samples(self, experiment: pd.DataFrame) -> None:
        result = stats.compare_means(experiment, "value", "group", "treatment", "control")
        assert result.test == "welch_t_test"
        assert result.significant
        assert result.effect_size["mean_difference"] == pytest.approx(1.0, abs=0.4)
        assert result.confidence_interval is not None
        assert result.confidence_interval["low"] > 0

    def test_matches_scipy_welch(self, experiment: pd.DataFrame) -> None:
        result = stats.compare_means(experiment, "value", "group", "control", "treatment", "welch")
        a = experiment.loc[experiment.group == "control", "value"]
        b = experiment.loc[experiment.group == "treatment", "value"]
        expected = sp.ttest_ind(a, b, equal_var=False)
        assert result.p_value == pytest.approx(expected.pvalue)

    def test_auto_falls_back_to_mann_whitney_for_small_skewed_samples(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "g": ["a"] * 15 + ["b"] * 15,
                "v": np.concatenate([rng.lognormal(0, 1.5, 15), rng.lognormal(1, 1.5, 15)]),
            }
        )
        result = stats.compare_means(df, "v", "g", "a", "b")
        assert result.test == "mann_whitney_u"
        assert -1 <= result.effect_size["rank_biserial"] <= 1

    def test_unknown_group_lists_levels(self, experiment: pd.DataFrame) -> None:
        with pytest.raises(StatisticalInputError, match="Levels"):
            stats.compare_means(experiment, "value", "group", "control", "missing")

    def test_non_numeric_outcome_is_rejected(self, experiment: pd.DataFrame) -> None:
        with pytest.raises(StatisticalInputError, match="numeric"):
            stats.compare_means(experiment, "segment", "group", "control", "treatment")


def test_compare_many_groups_finds_no_effect_on_random_segments(experiment: pd.DataFrame) -> None:
    result = stats.compare_many_groups(experiment, "noise", "segment")
    assert result.test == "one_way_anova"
    assert result.p_value is not None and result.p_value > 0.01
    assert len(result.details["groups"]) == 3


class TestTwoProportions:
    def test_detects_lift(self, experiment: pd.DataFrame) -> None:
        result = stats.two_proportion_test(experiment, "converted", "group", "control", "treatment")
        assert result.significant
        assert result.effect_size["absolute_lift"] > 0
        assert result.details["control"]["n"] == 300

    def test_accepts_text_booleans(self) -> None:
        df = pd.DataFrame({"g": ["a", "a", "b", "b"] * 30, "y": ["yes", "no", "yes", "yes"] * 30})
        result = stats.two_proportion_test(df, "y", "g", "a", "b")
        assert result.details["control"]["rate"] == pytest.approx(0.5)
        assert result.details["treatment"]["rate"] == pytest.approx(1.0)

    def test_rejects_non_binary_numeric(self, experiment: pd.DataFrame) -> None:
        with pytest.raises(StatisticalInputError, match="binary"):
            stats.two_proportion_test(experiment, "value", "group", "control", "treatment")


class TestBootstrap:
    def test_ci_contains_observed_difference_and_is_reproducible(
        self, experiment: pd.DataFrame
    ) -> None:
        kwargs = dict(n_resamples=1000, seed=3)
        first = stats.bootstrap_difference(
            experiment, "value", "group", "treatment", "control", **kwargs
        )
        second = stats.bootstrap_difference(
            experiment, "value", "group", "treatment", "control", **kwargs
        )
        assert first.to_dict() == second.to_dict()
        ci = first.confidence_interval
        assert ci is not None and ci["low"] < first.statistic < ci["high"]  # type: ignore[operator]
        assert first.significant

    def test_rejects_too_many_resamples(self, experiment: pd.DataFrame) -> None:
        with pytest.raises(StatisticalInputError):
            stats.bootstrap_difference(experiment, "value", "group", "a", "b", n_resamples=10**6)


def test_chi_square_detects_dependence(experiment: pd.DataFrame) -> None:
    result = stats.chi_square_independence(experiment, "group", "converted")
    assert result.significant
    assert 0 < result.effect_size["cramers_v"] < 1


def test_correlation_pearson_and_spearman(experiment: pd.DataFrame) -> None:
    pearson = stats.correlation(experiment, "value", "value_x2")
    spearman = stats.correlation(experiment, "value", "noise", method="spearman")
    assert pearson.effect_size["r"] > 0.9
    assert pearson.confidence_interval is not None
    assert abs(spearman.effect_size["r"]) < 0.2


def test_result_serialization_has_no_nan() -> None:
    result = stats.TestResult(test="x", statistic=float("nan"), p_value=0.01, alpha=0.05)
    payload = result.to_dict()
    assert payload["statistic"] is None
    assert payload["significant"] is True
