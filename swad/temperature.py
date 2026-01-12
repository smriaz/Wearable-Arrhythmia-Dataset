from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_temperature(rng: np.random.Generator, t_index_temp: pd.DatetimeIndex) -> pd.DataFrame:
    """Simulate skin temperature with circadian rhythm and optional mild illness bumps."""
    T0 = float(rng.normal(33.0, 0.5))
    A = float(rng.uniform(0.2, 0.8))
    phi = float(rng.uniform(0, 2*np.pi))

    # illness bouts
    illness = []
    if rng.random() < 0.15:
        start = t_index_temp[int(rng.integers(0, max(1, len(t_index_temp)-1)))]
        dur = int(np.exp(rng.normal(np.log(8*3600), 0.8)))
        end = start + pd.Timedelta(seconds=max(3600, min(dur, 3*86400)))
        bump = float(rng.uniform(0.5, 1.5))
        illness.append((start, end, bump))

    rows = []
    for t in t_index_temp:
        sec = int((t.value // 1_000_000_000) % 86400)
        val = T0 + A*np.sin(2*np.pi*sec/86400.0 + phi) + float(rng.normal(0, 0.1))
        for a,b,bump in illness:
            if a <= t < b:
                val += bump
        rows.append((t, float(val)))
    return pd.DataFrame(rows, columns=["timestamp","skin_temp_c"])
