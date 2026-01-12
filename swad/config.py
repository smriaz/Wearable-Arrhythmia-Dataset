from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SimConfig:
    """Configuration for SWAD simulation.

    Notes:
      - Keep defaults conservative for Streamlit Cloud (file sizes).
      - Increase accel_hz/days/n_subjects locally if you have resources.
    """
    n_subjects: int = 50
    days: int = 14
    start_iso: str = "2025-01-01T00:00:00"

    # Sampling
    accel_hz: int = 25
    hr_hz: int = 1
    contact_hz: int = 1
    sleep_epoch_s: int = 30
    spo2_epoch_s: int = 15
    temp_epoch_s: int = 60

    # AF / ectopy
    af_prevalence: float = 0.25
    af_persistent_frac: float = 0.10
    af_lambda0_per_day: float = 0.4         # baseline episode starts/day for paroxysmal
    af_hr_delta_bpm: float = 8.0

    ectopy_lambda_per_hour: float = 0.5

    # Wear state / contact
    off_wrist_starts_per_day: float = 1.0
    off_wrist_median_min: float = 20.0
    off_wrist_logsigma: float = 1.0

    # Noise / artifacts
    artifact_base: float = 0.02
    artifact_motion_weight: float = 1.2
    artifact_contact_weight: float = 1.5
    meas_jitter_base_s: float = 0.01
    sqi_motion_k: float = 0.8

    # ECG confirmations
    ecg_scheduled_every_days: int = 7
    ecg_trigger_tau_irregularity: float = 0.12
    ecg_trigger_tau_sqi: float = 0.5
    ecg_refractory_min: int = 120
    ecg_noise_prob: float = 0.03
    ecg_fn: float = 0.05
    ecg_fp: float = 0.01

    # Optional sensors
    enable_spo2: bool = True
    enable_temperature: bool = True

    # Output
    include_raw_accel: bool = True
    include_waveform_ecg: bool = False  # keep False by default (size)

    # Export
    export_format: str = "parquet"  # "parquet" or "csv"
