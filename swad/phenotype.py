from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Phenotype:
    subject_id: str
    age_band: str
    sex: str
    fitness: str
    baseline_hr_bpm: float
    baseline_hrv_s: float
    beta_blocker: bool
    skin_tone_proxy: float
    af_type: str  # "none" | "paroxysmal" | "persistent"

def sample_phenotype(rng: np.random.Generator, subject_id: str) -> Phenotype:
    age_band = rng.choice(
        ["18-29","30-39","40-49","50-59","60-69","70-79"],
        p=[.12,.16,.18,.20,.20,.14]
    )
    sex = rng.choice(["female","male"], p=[0.5,0.5])
    fitness = rng.choice(["low","mid","high"], p=[.35,.45,.20])

    base_hr = rng.normal(72, 8)
    if fitness == "high":
        base_hr -= 8
    if fitness == "low":
        base_hr += 4
    base_hr = float(np.clip(base_hr, 45, 110))

    base_hrv = rng.normal(0.04, 0.015)  # seconds (40ms typical)
    if fitness == "high":
        base_hrv += 0.02
    base_hrv = float(np.clip(base_hrv, 0.01, 0.15))

    beta = bool(rng.random() < (0.25 if age_band in ["60-69","70-79"] else 0.10))
    skin = float(np.clip(rng.beta(2,2), 0, 1))

    return Phenotype(
        subject_id=subject_id,
        age_band=str(age_band),
        sex=str(sex),
        fitness=str(fitness),
        baseline_hr_bpm=base_hr,
        baseline_hrv_s=base_hrv,
        beta_blocker=beta,
        skin_tone_proxy=skin,
        af_type="none",
    )
