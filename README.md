# SWAD Generator (Synthetic Wearable Arrhythmia Dataset)

This repo provides a **Streamlit app** that generates fully synthetic wearable time series suitable for
**methods research** and benchmarking:

- PPG-derived inter-beat intervals (IBI) + signal quality index (SQI)
- Accelerometer
- Wear state + contact quality
- Sleep staging (wake/light/deep/REM)
- Optional SpO₂ and skin temperature
- Intermittent ECG confirmations
- Ground-truth interval labels (AF, ectopy, artifact, no_contact)

> All data are synthetic (no real patient data).

## Run (Streamlit)
1. Install dependencies (Streamlit Cloud handles this automatically via `requirements.txt`).
2. Launch:
   - `streamlit run app.py`

## Outputs
Download button produces a ZIP with:
- `demographics.json`
- `.parquet` files for each stream (default)
- `labels.parquet` containing interval annotations and a `priority` column.

## Notes
- Raw accelerometer can be large; toggle off for smaller downloads.
- The simulator is intentionally simple but structured for research extensions.

## License
MIT
