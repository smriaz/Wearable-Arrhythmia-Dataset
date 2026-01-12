from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

Interval = Tuple[pd.Timestamp, pd.Timestamp]

def circadian_sin(ts: pd.DatetimeIndex) -> np.ndarray:
    sec = (ts.view("int64") // 1_000_000_000) % 86400
    return np.sin(2*np.pi*sec/86400.0)

def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a,b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb,b))
        else:
            out.append((a,b))
    return out

def simulate_af_episodes(
    rng: np.random.Generator,
    t_index_1hz: pd.DatetimeIndex,
    af_lambda0_per_day: float
) -> List[Interval]:
    """Approximate inhomogeneous Poisson episode start process on 1Hz grid."""
    c = circadian_sin(t_index_1hz)
    hour = t_index_1hz.hour.to_numpy()
    sleep = ((hour >= 23) | (hour <= 6)).astype(float)

    lam0 = float(af_lambda0_per_day) / 86400.0  # per second
    lam = lam0 * np.exp(0.4*c + 0.6*sleep)

    starts = []
    for i, t in enumerate(t_index_1hz):
        if rng.random() < lam[i]:
            starts.append(t)

    episodes: List[Interval] = []
    for st in starts:
        dur = float(np.exp(rng.normal(np.log(20*60), 1.0)))  # seconds
        dur = float(np.clip(dur, 60, 24*3600))
        episodes.append((st, st + pd.Timedelta(seconds=dur)))
    return merge_intervals(episodes)

def simulate_true_ibi_stream(
    rng: np.random.Generator,
    start_ts: pd.Timestamp,
    days: int,
    phenotype,
    activity_1hz: pd.Series,
    sleep_epochs: pd.DataFrame,
    af_intervals: List[Interval],
    af_hr_delta_bpm: float,
    ectopy_lambda_per_hour: float,
) -> pd.DataFrame:
    """Generate beat times with true IBIs and latent rhythm (SR/AF) plus ectopy events."""
    t_end = start_ts + pd.Timedelta(days=int(days))

    sleep_map = pd.Series(sleep_epochs.sleep_stage.values, index=sleep_epochs.epoch_start)
    def sleep_stage_at(t: pd.Timestamp) -> str:
        # assume sleep_map is dense; nearest backwards
        s = sleep_map[:t]
        return str(s.iloc[-1]) if len(s) else "wake"

    def in_af(t: pd.Timestamp) -> bool:
        for a,b in af_intervals:
            if a <= t < b:
                return True
        return False

    beats = []
    delta_prev = 0.0

    # Ectopy: schedule beat indices by time-based Poisson, then map to nearest beat
    # We'll mark ectopy online by sampling at each beat with small probability.
    p_ectopy_per_beat = float(ectopy_lambda_per_hour) / 3600.0 * 0.8  # approx for ~1 beat/sec

    b = start_ts
    while b < t_end:
        a = float(activity_1hz.reindex([b], method="nearest").iloc[0])
        stage = sleep_stage_at(b)
        asleep = 1.0 if stage != "wake" else 0.0

        sec = int((b.value // 1_000_000_000) % 86400)
        c = float(np.sin(2*np.pi*sec/86400.0))

        h = float(phenotype.baseline_hr_bpm)
        h += 25.0 * a
        if stage == "deep":
            h -= 8.0
        elif stage == "light":
            h -= 4.0
        elif stage == "rem":
            h += 2.0
        h += 3.0 * c
        if bool(phenotype.beta_blocker):
            h -= 4.0
        h = float(np.clip(h, 35, 200))

        af = (phenotype.af_type != "none") and in_af(b)
        ectopy = False

        if af:
            h_af = float(np.clip(h + float(af_hr_delta_bpm), 35, 220))
            mu = 60.0 / h_af
            eta = rng.laplace(0.0, 0.12*mu)
            x = float(np.clip(mu + eta, 0.3, 2.0))
            delta_prev = 0.0
        else:
            mu = 60.0 / h
            sigma0 = float(phenotype.baseline_hrv_s)
            sigma = float(sigma0 * np.exp(0.6*asleep - 0.8*a))
            rho = 0.85
            eps = float(rng.normal(0.0, sigma))
            delta = float(rho*delta_prev + eps)
            x = float(np.clip(mu + delta, 0.3, 2.0))
            delta_prev = delta

            # Ectopy only in SR
            if rng.random() < p_ectopy_per_beat * (1.0 + 0.5*a):
                ectopy = True
                r = float(rng.uniform(0.4, 0.8))
                x = float(np.clip(r*x, 0.25, 1.8))

        beats.append((b, x, "AF" if af else "SR", stage, a, int(ectopy)))
        b = b + pd.Timedelta(seconds=x)

    df = pd.DataFrame(beats, columns=["timestamp","ibi_true_s","rhythm_true","sleep_stage","activity","ectopy_flag"])
    df["ibi_true_ms"] = (df["ibi_true_s"]*1000.0).round(1)
    return df
