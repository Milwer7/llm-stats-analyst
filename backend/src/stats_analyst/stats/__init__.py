"""Pure statistical routines. No LLM or web dependencies live here."""

from stats_analyst.stats.ab_testing import two_proportion_test
from stats_analyst.stats.association import chi_square_independence, correlation
from stats_analyst.stats.bootstrap import bootstrap_difference
from stats_analyst.stats.hypothesis import compare_many_groups, compare_means
from stats_analyst.stats.results import TestResult

__all__ = [
    "TestResult",
    "bootstrap_difference",
    "chi_square_independence",
    "compare_many_groups",
    "compare_means",
    "correlation",
    "two_proportion_test",
]
