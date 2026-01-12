from __future__ import annotations
import io, json, zipfile
from typing import Dict, Any
import pandas as pd

def _df_to_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "parquet":
        df.to_parquet(buf, index=False)
    elif fmt == "csv":
        buf.write(df.to_csv(index=False).encode("utf-8"))
    else:
        raise ValueError("fmt must be 'parquet' or 'csv'")
    return buf.getvalue()

def bundle_subject_to_zip(
    subject_id: str,
    files: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    fmt: str = "parquet"
) -> bytes:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{subject_id}/demographics.json", json.dumps(metadata, indent=2))
        for name, df in files.items():
            if name.endswith(".parquet"):
                z.writestr(f"{subject_id}/{name}", _df_to_bytes(df, "parquet"))
            elif name.endswith(".csv"):
                z.writestr(f"{subject_id}/{name}", _df_to_bytes(df, "csv"))
            else:
                # default to chosen fmt
                ext = "parquet" if fmt == "parquet" else "csv"
                z.writestr(f"{subject_id}/{name}.{ext}", _df_to_bytes(df, fmt))
    return zbuf.getvalue()

def bundle_cohort_to_zip(
    cohort_files: Dict[str, bytes]
) -> bytes:
    """Bundle already-zipped subjects + metadata into a single ZIP."""
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, content in cohort_files.items():
            z.writestr(path, content)
    return zbuf.getvalue()
