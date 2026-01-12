from __future__ import annotations
import numpy as np
import pandas as pd

STAGES = ["wake","light","deep","rem"]

def simulate_sleep_epochs(
    rng: np.random.Generator,
    start_ts: pd.Timestamp,
    days: int,
    epoch_s: int
) -> pd.DataFrame:
    """Simulate wearable-grade sleep staging at fixed epoch size.

    Outside sleep window, stage is wake. Inside, stage follows a Markov chain.
    """
    rows = []
    # Transition matrix (rows sum to 1), tuned for plausible proportions
    P = np.array([
        [0.85,0.10,0.02,0.03],  # wake
        [0.08,0.70,0.12,0.10],  # light
        [0.03,0.20,0.65,0.12],  # deep
        [0.06,0.25,0.10,0.59],  # rem
    ])

    for d in range(days):
        # Bedtime around 23:00 and wake time around 07:00 next day
        bed = (start_ts + pd.Timedelta(days=d)).normalize() + pd.Timedelta(hours=23)               + pd.Timedelta(minutes=int(rng.normal(0,20)))
        wake = (start_ts + pd.Timedelta(days=d+1)).normalize() + pd.Timedelta(hours=7)                + pd.Timedelta(minutes=int(rng.normal(0,20)))

        day_start = (start_ts + pd.Timedelta(days=d)).normalize()
        day_end = day_start + pd.Timedelta(days=1)
        t = day_start
        state = 0  # wake
        while t < day_end:
            in_sleep = (t >= bed) and (t < wake)
            if not in_sleep:
                stage = "wake"
                state = 0
            else:
                state = int(rng.choice(len(STAGES), p=P[state]))
                stage = STAGES[state]
            rows.append((t, stage))
            t += pd.Timedelta(seconds=int(epoch_s))

    return pd.DataFrame(rows, columns=["epoch_start","sleep_stage"])
