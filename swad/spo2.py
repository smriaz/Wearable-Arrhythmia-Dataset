from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_spo2(
    rng: np.random.Generator,
    t_index_spo2: pd.DatetimeIndex,
    sleep_epochs: pd.DataFrame,
    wear_1hz: pd.DataFrame,
) -> pd.DataFrame:
    """Simulate SpO2 (night-focused), with optional desaturation events."""
    sleep_map = pd.Series(sleep_epochs.sleep_stage.values, index=sleep_epochs.epoch_start)
    on_series = pd.Series(wear_1hz.on_wrist.values, index=wear_1hz.timestamp)
    q_series  = pd.Series(wear_1hz.contact_quality.values, index=wear_1hz.timestamp)

    S0 = float(rng.normal(97.0, 1.0))
    beta_S = float(rng.normal(0.2, 0.05))  # small circadian drift

    # Generate desaturation events during sleep: Poisson rate per hour
    lam = 0.3 / 3600.0  # ~0.3 events/hour in sleep
    event_starts = []
    for t in t_index_spo2:
        st = str(sleep_map[:t].iloc[-1]) if len(sleep_map[:t]) else "wake"
        asleep = (st != "wake")
        if asleep and rng.random() < lam * (t_index_spo2.freq.delta.total_seconds() if t_index_spo2.freq is not None else 15):
            event_starts.append(t)

    # Precompute event contributions
    drops = []
    for t in t_index_spo2:
        sec = int((t.value // 1_000_000_000) % 86400)
        c = float(np.sin(2*np.pi*sec/86400.0))
        val = S0 + beta_S*c
        # event decay sum
        for es in event_starts:
            if t >= es:
                A = 5.0  # typical
                tau = 30.0
                dt = (t - es).total_seconds()
                val += -A * np.exp(-dt / tau)
        on = int(on_series.reindex([t], method="nearest").iloc[0])
        q  = float(q_series.reindex([t], method="nearest").iloc[0])
        if on == 0:
            drops.append((t, np.nan, 0.0))
        else:
            noise = float(rng.normal(0, 0.6))
            spo2 = float(np.clip(val + noise, 70, 100))
            quality = float(np.clip(q + rng.normal(0, 0.1), 0, 1))
            drops.append((t, spo2, quality))
    return pd.DataFrame(drops, columns=["timestamp","spo2_pct","spo2_quality"])
