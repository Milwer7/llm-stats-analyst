"""Shared result type returned by every statistical routine."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TestResult:
    """Outcome of a statistical procedure, ready to be serialized for the LLM or the UI."""

    __test__ = False  # keep pytest from collecting this class

    test: str
    statistic: float | None
    p_value: float | None
    alpha: float
    effect_size: dict[str, float] = field(default_factory=dict)
    confidence_interval: dict[str, float] | None = None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool | None:
        if self.p_value is None:
            return None
        return self.p_value < self.alpha

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["significant"] = self.significant
        cleaned: dict[str, Any] = json_safe(payload)
        return cleaned


def json_safe(value: Any) -> Any:
    """Round floats and replace NaN/inf so the payload is valid, compact JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if hasattr(value, "item"):  # numpy scalar
        return json_safe(value.item())
    return value
