from __future__ import annotations
import pandas as pd
from typing import List, Tuple

def intervals_from_af(af_intervals: List[Tuple[pd.Timestamp,pd.Timestamp]]) -> pd.DataFrame:
    return pd.DataFrame([(a,b,"AF") for a,b in af_intervals], columns=["start_time","end_time","label"])

def ectopy_intervals(beats_true: pd.DataFrame, pre_s: float = 2.0, post_s: float = 4.0) -> pd.DataFrame:
    ect = beats_true[beats_true["ectopy_flag"] == 1][["timestamp"]].copy()
    rows = []
    for t in ect["timestamp"].tolist():
        rows.append((t - pd.Timedelta(seconds=pre_s), t + pd.Timedelta(seconds=post_s), "ectopy"))
    return pd.DataFrame(rows, columns=["start_time","end_time","label"])

def point_labels_to_intervals(points: pd.DataFrame, window_s: float = 2.0) -> pd.DataFrame:
    """Convert beat-level point labels (artifact/no_contact) to short intervals."""
    rows = []
    for t, lab in points[["timestamp","label"]].itertuples(index=False):
        rows.append((t - pd.Timedelta(seconds=window_s/2), t + pd.Timedelta(seconds=window_s/2), str(lab)))
    return pd.DataFrame(rows, columns=["start_time","end_time","label"])

def concat_and_sort(*dfs: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([d for d in dfs if d is not None and len(d)], ignore_index=True)
    if len(df) == 0:
        return pd.DataFrame(columns=["start_time","end_time","label"])
    df = df.sort_values(["start_time","end_time","label"]).reset_index(drop=True)
    return df

def apply_priority(labels: pd.DataFrame) -> pd.DataFrame:
    """Optional: enforce priority ordering by collapsing overlaps.

    Priority: no_contact > artifact > AF > ectopy
    This implementation keeps all intervals but adds 'priority' column to support downstream filtering.
    """
    pri = {"no_contact":4, "artifact":3, "AF":2, "ectopy":1}
    labels = labels.copy()
    labels["priority"] = labels["label"].map(lambda x: pri.get(str(x), 0)).astype(int)
    return labels
