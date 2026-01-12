import streamlit as st
import pandas as pd

from swad.config import SimConfig
from swad.rng import make_rng
from swad.phenotype import sample_phenotype
from swad.timegrid import make_time_index
from swad.sleep import simulate_sleep_epochs
from swad.activity import simulate_activity_intensity, simulate_accel
from swad.wear_state import simulate_wear_state
from swad.rhythm import simulate_af_episodes, simulate_true_ibi_stream
from swad.observation import observe_ibi, derive_hr_1hz
from swad.spo2 import simulate_spo2
from swad.temperature import simulate_temperature
from swad.ecg import simulate_ecg_events
from swad.labels import intervals_from_af, ectopy_intervals, point_labels_to_intervals, concat_and_sort, apply_priority
from swad.export import bundle_subject_to_zip

st.set_page_config(page_title="SWAD Generator", layout="wide")
st.title("SWAD — Synthetic Wearable Arrhythmia Dataset Generator")

st.markdown(
    """Generate fully synthetic wearable time series for arrhythmia research:
- PPG-derived IBI + SQI
- Accelerometer
- Wear/contact state
- Sleep staging
- Optional SpO₂ and skin temperature
- Intermittent ECG confirmations
- Ground-truth interval labels (hidden truth for evaluation)
"""
)

with st.sidebar:
    st.header("Cohort")
    n_subjects = st.number_input("Subjects (download = one subject preview; cohort generation is optional)", 1, 5000, 10)
    days = st.number_input("Days per subject", 1, 90, 14)
    seed = st.number_input("Seed", 0, 10_000_000, 42)

    st.header("Sensors")
    include_raw_accel = st.checkbox("Include raw accelerometer (bigger files)", value=True)
    enable_spo2 = st.checkbox("Enable SpO₂", value=True)
    enable_temp = st.checkbox("Enable skin temperature", value=True)

    st.header("AF parameters")
    af_prev = st.slider("AF prevalence", 0.0, 1.0, 0.25, 0.01)
    af_persistent = st.slider("Persistent fraction among AF", 0.0, 1.0, 0.10, 0.01)
    af_lambda0 = st.slider("AF episode starts/day (paroxysmal)", 0.0, 5.0, 0.4, 0.05)
    af_hr_delta = st.slider("AF HR delta (bpm)", 0.0, 40.0, 8.0, 1.0)

    st.header("Artifacts")
    art_base = st.slider("Artifact base rate", 0.0, 0.2, 0.02, 0.005)

cfg = SimConfig(
    n_subjects=int(n_subjects),
    days=int(days),
    enable_spo2=bool(enable_spo2),
    enable_temperature=bool(enable_temp),
    include_raw_accel=bool(include_raw_accel),
    af_prevalence=float(af_prev),
    af_persistent_frac=float(af_persistent),
    af_lambda0_per_day=float(af_lambda0),
    af_hr_delta_bpm=float(af_hr_delta),
    artifact_base=float(art_base),
)

st.write("### Generate (Preview)")
generate = st.button("Generate one example subject", type="primary")

if generate:
    rng = make_rng(int(seed))
    subject_id = "subject_00001"
    start_ts = pd.Timestamp(cfg.start_iso)

    ph = sample_phenotype(rng, subject_id=subject_id)

    # Assign AF type to meet prevalence (for preview we just sample)
    if rng.random() < cfg.af_prevalence:
        ph.af_type = "persistent" if rng.random() < cfg.af_persistent_frac else "paroxysmal"
    else:
        ph.af_type = "none"

    # Time grids
    t_1hz = make_time_index(cfg.start_iso, cfg.days, 1)

    # Simulate covariates
    sleep = simulate_sleep_epochs(rng, start_ts, cfg.days, cfg.sleep_epoch_s)
    activity = simulate_activity_intensity(rng, t_1hz)

    if cfg.include_raw_accel:
        t_acc = make_time_index(cfg.start_iso, cfg.days, cfg.accel_hz)
        accel = simulate_accel(rng, t_acc, activity)
    else:
        accel = pd.DataFrame({"timestamp": t_1hz, "ax": 0.0, "ay": 0.0, "az": 0.0})

    wear = simulate_wear_state(
        rng, t_1hz,
        off_wrist_starts_per_day=cfg.off_wrist_starts_per_day,
        median_min=cfg.off_wrist_median_min,
        logsigma=cfg.off_wrist_logsigma,
    )

    # AF intervals + beats
    if ph.af_type == "paroxysmal":
        af_intervals = simulate_af_episodes(rng, t_1hz, cfg.af_lambda0_per_day)
    elif ph.af_type == "persistent":
        # persistent AF starting day 0
        af_intervals = [(start_ts, start_ts + pd.Timedelta(days=cfg.days))]
    else:
        af_intervals = []

    beats_true = simulate_true_ibi_stream(
        rng, start_ts, cfg.days, ph, activity, sleep, af_intervals,
        af_hr_delta_bpm=cfg.af_hr_delta_bpm,
        ectopy_lambda_per_hour=cfg.ectopy_lambda_per_hour,
    )

    ibi_obs, artifact_points = observe_ibi(
        rng, beats_true, accel, wear,
        skin_tone_proxy=ph.skin_tone_proxy,
        artifact_base=cfg.artifact_base,
        artifact_motion_weight=cfg.artifact_motion_weight,
        artifact_contact_weight=cfg.artifact_contact_weight,
        meas_jitter_base_s=cfg.meas_jitter_base_s,
        sqi_motion_k=cfg.sqi_motion_k,
    )

    hr_1hz = derive_hr_1hz(ibi_obs, t_1hz)

    # Optional sensors
    spo2 = None
    if cfg.enable_spo2:
        t_spo2 = pd.date_range(start_ts, start_ts + pd.Timedelta(days=cfg.days), freq=f"{cfg.spo2_epoch_s}s", inclusive="left")
        spo2 = simulate_spo2(rng, t_spo2, sleep, wear)

    temp = None
    if cfg.enable_temperature:
        t_temp = pd.date_range(start_ts, start_ts + pd.Timedelta(days=cfg.days), freq=f"{cfg.temp_epoch_s}s", inclusive="left")
        temp = simulate_temperature(rng, t_temp)

    # ECG confirmations
    ecg = simulate_ecg_events(rng, start_ts, cfg.days, cfg, ibi_obs, beats_true)

    # Labels (intervals)
    labels_af = intervals_from_af(af_intervals)
    labels_ect = ectopy_intervals(beats_true)
    labels_art = point_labels_to_intervals(artifact_points, window_s=2.0)
    labels = apply_priority(concat_and_sort(labels_af, labels_ect, labels_art))

    st.success(f"Generated {subject_id} (AF type: {ph.af_type}).")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Observed IBI sample")
        st.dataframe(ibi_obs.head(25))
        st.write("Derived HR (1Hz) sample")
        st.dataframe(hr_1hz.head(25))
    with c2:
        st.write("ECG events sample")
        st.dataframe(ecg.head(25))
        st.write("Label intervals sample")
        st.dataframe(labels.head(25))

    files = {
        "ibi.parquet": ibi_obs,
        "hr.parquet": hr_1hz,
        "accel.parquet": accel,
        "wear_state.parquet": wear,
        "sleep.parquet": sleep,
        "ecg_events.parquet": ecg,
        "labels.parquet": labels,
    }
    if spo2 is not None:
        files["spo2.parquet"] = spo2
    if temp is not None:
        files["temperature.parquet"] = temp

    meta = {
        "subject_id": subject_id,
        "age_band": ph.age_band,
        "sex": ph.sex,
        "fitness": ph.fitness,
        "baseline_hr_bpm": ph.baseline_hr_bpm,
        "baseline_hrv_s": ph.baseline_hrv_s,
        "beta_blocker": ph.beta_blocker,
        "skin_tone_proxy": ph.skin_tone_proxy,
        "af_type": ph.af_type,
        "days": cfg.days,
        "start_iso": cfg.start_iso,
    }

    zip_bytes = bundle_subject_to_zip(subject_id, files, meta, fmt=cfg.export_format)
    st.download_button(
        "Download subject ZIP",
        data=zip_bytes,
        file_name=f"{subject_id}.zip",
        mime="application/zip"
    )
