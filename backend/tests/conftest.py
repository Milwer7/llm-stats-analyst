from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def experiment() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    group = np.repeat(["control", "treatment"], n // 2)
    value = np.concatenate([rng.normal(10, 2, n // 2), rng.normal(11, 2, n // 2)])
    converted = np.concatenate([rng.random(n // 2) < 0.10, rng.random(n // 2) < 0.20]).astype(int)
    segment = rng.choice(["a", "b", "c"], size=n)
    noise = rng.normal(size=n)
    return pd.DataFrame(
        {
            "group": group,
            "value": value,
            "converted": converted,
            "segment": segment,
            "value_x2": value * 2 + noise,
            "noise": noise,
        }
    )
