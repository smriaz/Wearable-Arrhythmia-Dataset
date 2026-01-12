from __future__ import annotations
import numpy as np
import pandas as pd

def irregularity_stat(ibi_window_s: np.ndarray) -> float:
    x = ibi_window_s[~np.isnan(ibi_window_s)]
    if len(x) < 5:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(mad / (med + 1e-6))

def simulate_ecg_events(
    rng: np.random.Generator,
    start_ts: pd.Timestamp,
    days: int,
    cfg,
    ibi_obs: pd.DataFrame,
    beats_true: pd.DataFrame,
) -> pd.DataFrame:
    """Simulate intermittent ECG confirmations (scheduled + device-triggered)."""
    times = []

    # Scheduled
    K = max(1, int(cfg.ecg_scheduled_every_days))
    for d in range(0, int(days), K):
        t = (start_ts + pd.Timedelta(days=d)).normalize() + pd.Timedelta(hours=12)             + pd.Timedelta(minutes=int(rng.normal(0,15)))
        times.append(("scheduled", t))

    # Device-triggered: scan every 5s to reduce compute
    ibi_series = pd.Series(ibi_obs.ibi_s.values, index=ibi_obs.timestamp)
    sqi_series = pd.Series(ibi_obs.ppg_sqi.values, index=ibi_obs.timestamp)
    true_series = pd.Series(beats_true.rhythm_true.values, index=beats_true.timestamp)

    refractory = pd.Timedelta(minutes=int(cfg.ecg_refractory_min))
    last = None
    t_scan = pd.date_range(start_ts, start_ts + pd.Timedelta(days=int(days)), freq="5s", inclusive="left")

    for t in t_scan:
        if last is not None and (t - last) < refractory:
            continue
        win = ibi_series.loc[(ibi_series.index >= t - pd.Timedelta(seconds=60)) & (ibi_series.index <= t)]
        I = irregularity_stat(win.to_numpy())
        sqi = float(sqi_series.reindex([t], method="nearest").iloc[0]) if len(sqi_series) else 0.0
        if I > float(cfg.ecg_trigger_tau_irregularity) and sqi > float(cfg.ecg_trigger_tau_sqi):
            times.append(("device_triggered", t))
            last = t

    # Labels
    rows = []
    for trig, t in times:
        true = str(true_series.reindex([t], method="nearest").iloc[0]) if len(true_series) else "SR"
        if rng.random() < float(cfg.ecg_noise_prob):
            label = "noise"
            sqi = float(np.clip(rng.normal(0.3,0.2), 0, 1))
        else:
            if true == "AF":
                label = "AF" if rng.random() > float(cfg.ecg_fn) else "nonAF"
            else:
                label = "AF" if rng.random() < float(cfg.ecg_fp) else "nonAF"
            sqi = float(np.clip(rng.normal(0.85,0.08), 0, 1))
        rows.append((t, trig, label, sqi, true))
    return pd.DataFrame(rows, columns=["timestamp","trigger_type","ecg_label","ecg_sqi","rhythm_true_at_ecg"])
