"""Generate reproducible synthetic datasets for demos and tests.

Usage: uv run python scripts/generate_sample_data.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7


def checkout_experiment(n: int = 4000, seed: int = SEED) -> pd.DataFrame:
    """Randomized A/B test of a new checkout flow with a small conversion lift."""
    rng = np.random.default_rng(seed)
    variant = rng.choice(["control", "new_checkout"], size=n)
    device = rng.choice(["mobile", "desktop", "tablet"], size=n, p=[0.6, 0.32, 0.08])
    base_rate = np.where(device == "desktop", 0.13, 0.09)
    converted = rng.random(n) < base_rate + np.where(variant == "new_checkout", 0.025, 0.0)
    # Order values are right-skewed (log-normal) and only exist for converted sessions.
    order_value = np.where(
        converted, np.round(rng.lognormal(mean=3.6, sigma=0.6, size=n), 2), np.nan
    )
    session_minutes = np.round(rng.gamma(shape=2.0, scale=3.0, size=n), 1)
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "variant": variant,
            "device": device,
            "session_minutes": session_minutes,
            "converted": converted.astype(int),
            "order_value": order_value,
        }
    )


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "checkout_experiment.csv"
    checkout_experiment().to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
