from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_activity_intensity(rng: np.random.Generator, t_index_1hz: pd.DatetimeIndex) -> pd.Series:
    """Semi-Markov activity intensity on a 1Hz grid."""
    states = [0,1,2]  # sedentary/light/moderate
    alpha = {0:0.05, 1:0.35, 2:0.70}
    # lognormal params for duration in seconds: exp(N(mu, sigma))
    mu = {0:6.0, 1:4.5, 2:4.0}
    sig= {0:0.8, 1:0.7, 2:0.6}
    trans = {
        0: [0.70,0.25,0.05],
        1: [0.50,0.40,0.10],
        2: [0.55,0.35,0.10]
    }

    out = np.zeros(len(t_index_1hz), dtype=float)
    i = 0
    s = 0
    while i < len(out):
        dur = int(np.exp(rng.normal(mu[s], sig[s])))
        dur = max(30, min(dur, 3*3600))
        a = alpha[s] + rng.normal(0, 0.03)
        a = float(np.clip(a, 0, 1))
        j = min(len(out), i+dur)
        out[i:j] = a
        s = int(rng.choice(states, p=trans[s]))
        i = j
    return pd.Series(out, index=t_index_1hz, name="activity")

def simulate_accel(
    rng: np.random.Generator,
    t_index_accel: pd.DatetimeIndex,
    activity_1hz: pd.Series
) -> pd.DataFrame:
    """Raw-ish 3-axis accelerometer correlated with activity intensity."""
    a = activity_1hz.reindex(t_index_accel, method="nearest").to_numpy()
    m = 0.05 + 0.6*a + rng.normal(0, 0.08, size=len(a))

    v = rng.normal(size=(len(a),3))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    xyz = (m[:,None]*v) + rng.normal(0, 0.02, size=(len(a),3))

    return pd.DataFrame({"timestamp": t_index_accel, "ax": xyz[:,0], "ay": xyz[:,1], "az": xyz[:,2]})
