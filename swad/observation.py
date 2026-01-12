from __future__ import annotations
import numpy as np
import pandas as pd

def observe_ibi(
    rng: np.random.Generator,
    beats_true: pd.DataFrame,
    accel_df: pd.DataFrame,
    wear_1hz: pd.DataFrame,
    skin_tone_proxy: float,
    artifact_base: float,
    artifact_motion_weight: float,
    artifact_contact_weight: float,
    meas_jitter_base_s: float,
    sqi_motion_k: float,
):
    """Convert true beat stream to observed IBI with artifacts and PPG SQI.

    Returns:
      ibi_obs: timestamped IBI measurements (may be NaN)
      artifact_points: point labels ("artifact" or "no_contact") at beat timestamps
    """
    mag = np.sqrt(accel_df.ax**2 + accel_df.ay**2 + accel_df.az**2)
    mag_series = pd.Series(mag.values, index=accel_df.timestamp)

    on_series = pd.Series(wear_1hz.on_wrist.values, index=wear_1hz.timestamp)
    q_series  = pd.Series(wear_1hz.contact_quality.values, index=wear_1hz.timestamp)

    u = float(np.clip(1.0 - 0.5*float(skin_tone_proxy), 0.3, 1.0))

    out_rows = []
    art_rows = []

    for _, r in beats_true.iterrows():
        t = r["timestamp"]
        x = float(r["ibi_true_s"])

        on = int(on_series.reindex([t], method="nearest").iloc[0])
        q  = float(q_series.reindex([t], method="nearest").iloc[0])
        m  = float(mag_series.reindex([t], method="nearest").iloc[0])

        if on == 0:
            out_rows.append((t, np.nan, 0.0))
            art_rows.append((t, "no_contact"))
            continue

        sigma_meas = float(meas_jitter_base_s / u * (1.0/(q+1e-3)))
        x_meas = float(x + rng.normal(0, sigma_meas))

        logits = float(-3.5 + artifact_motion_weight*m - artifact_contact_weight*q + 0.8*(1-u))
        p_art = float(1.0/(1.0+np.exp(-logits)))
        p_art = float(np.clip(p_art + artifact_base, 0, 1))

        sqi = float(q*u*np.exp(-sqi_motion_k*m) + rng.normal(0, 0.03))
        sqi = float(np.clip(sqi, 0, 1))

        if rng.random() < p_art:
            typ = str(rng.choice(["drop","dup","merge","burst"], p=[0.35,0.25,0.20,0.20]))
            if typ == "drop":
                x_meas = np.nan
            elif typ == "dup":
                x_meas = float(np.clip(rng.uniform(0.2,0.6)*x, 0.25, 0.6))
            elif typ == "merge":
                x_meas = float(np.clip(rng.uniform(1.3,2.2)*x, 1.2, 2.5))
            else:
                x_meas = float(x + rng.normal(0, (0.08*(1+m)/(q+1e-3))))
            sqi = float(np.clip(sqi - rng.uniform(0.2,0.6), 0, 1))
            art_rows.append((t, "artifact"))

        out_rows.append((t, x_meas, sqi))

    ibi_obs = pd.DataFrame(out_rows, columns=["timestamp","ibi_s","ppg_sqi"])
    ibi_obs["ibi_ms"] = (ibi_obs["ibi_s"]*1000.0).round(1)
    artifact_points = pd.DataFrame(art_rows, columns=["timestamp","label"])
    return ibi_obs, artifact_points

def derive_hr_1hz(ibi_obs: pd.DataFrame, t_index_1hz: pd.DatetimeIndex, window_s: int = 10) -> pd.DataFrame:
    """Derive a 1Hz HR estimate from recent valid IBI values."""
    s = pd.Series(ibi_obs.ibi_s.values, index=ibi_obs.timestamp)
    hr = []
    for t in t_index_1hz:
        w = s.loc[(s.index >= t - pd.Timedelta(seconds=window_s)) & (s.index <= t)]
        x = w.dropna().to_numpy()
        if len(x) < 3:
            hr.append((t, np.nan))
        else:
            hr.append((t, float(60.0/np.median(x))))
    return pd.DataFrame(hr, columns=["timestamp","hr_bpm"])
