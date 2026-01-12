import pandas as pd

def make_time_index(start_iso: str, days: int, hz: int) -> pd.DatetimeIndex:
    """Create a regularly-sampled time index."""
    if hz <= 0:
        raise ValueError("hz must be positive")
    freq_ms = int(1000 / hz)
    periods = int(days * 24 * 60 * 60 * hz)
    return pd.date_range(start=pd.Timestamp(start_iso), periods=periods, freq=f"{freq_ms}ms")
