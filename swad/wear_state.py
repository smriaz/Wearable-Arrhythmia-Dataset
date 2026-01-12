from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_wear_state(
    rng: np.random.Generator,
    t_index_1hz: pd.DatetimeIndex,
    off_wrist_starts_per_day: float = 1.0,
    median_min: float = 20.0,
    logsigma: float = 1.0,
) -> pd.DataFrame:
    """Simulate on-wrist/off-wrist and contact quality on a 1Hz grid."""
    on = np.ones(len(t_index_1hz), dtype=int)

    # Convert starts/day to per-second probability on 1 Hz grid
    p_start = float(off_wrist_starts_per_day) / 86400.0

    i = 0
    while i < len(on):
        if rng.random() < p_start:
            # Lognormal duration with given median in minutes
            mu = np.log(max(1.0, median_min*60.0))
            dur = int(np.exp(rng.normal(mu, logsigma)))
            dur = max(60, min(dur, 6*3600))
            j = min(len(on), i+dur)
            on[i:j] = 0
            i = j
        else:
            i += 1

    # Contact quality: baseline high with small noise; can be refined downstream
    q = 0.85 + rng.normal(0, 0.05, size=len(on))
    q = np.clip(q, 0, 1)

    return pd.DataFrame({"timestamp": t_index_1hz, "on_wrist": on, "contact_quality": q})
