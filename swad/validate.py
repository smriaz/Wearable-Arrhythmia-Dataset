from __future__ import annotations
import pandas as pd

def validate_subject_bundle(files: dict) -> list[str]:
    """Lightweight validation; returns a list of warnings/errors."""
    issues = []
    required = ["ibi", "wear_state", "sleep", "ecg_events", "labels"]
    for r in required:
        if r not in files:
            issues.append(f"Missing required stream: {r}")
    if "ibi" in files:
        df = files["ibi"]
        if "timestamp" not in df.columns:
            issues.append("ibi missing timestamp")
        if "ibi_s" not in df.columns:
            issues.append("ibi missing ibi_s")
    return issues
