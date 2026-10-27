"""Helpers to read SECOM-style sensor files and score lots with the saved model."""
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
N_SENSORS = 590
SENSOR_COLS = [f"s{i + 1:03d}" for i in range(N_SENSORS)]


def load_model():
    # the saved pipeline uses custom steps defined in model/transformers.py
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    return joblib.load(MODEL_DIR / "secom_failure_model.joblib")


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return token.strip().lower() == "nan"


def read_lots(file_like) -> pd.DataFrame:
    """Read either the raw UCI file (secom.data: space separated, no header)
    or a CSV with a header row containing s001 ... s590."""
    raw = file_like.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
    first = text.strip().splitlines()[0]
    comma = first.count(",") > 10
    first_token = (first.split(",")[0] if comma else first.split()[0]).strip()
    has_header = not _is_number(first_token)
    df = pd.read_csv(io.StringIO(text), sep="," if comma else r"\s+",
                     header=0 if has_header else None, na_values=["NaN", "nan", ""])
    if has_header:
        missing = [c for c in SENSOR_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Header found, but {len(missing)} sensor columns are missing (for example {missing[:3]}). "
                             f"Expected columns s001 ... s{N_SENSORS}.")
        return df[SENSOR_COLS].astype(float)
    if df.shape[1] != N_SENSORS:
        raise ValueError(f"Expected {N_SENSORS} sensor columns, found {df.shape[1]}.")
    df.columns = SENSOR_COLS
    return df.astype(float)


def score_lots(model, lots: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(lots[SENSOR_COLS])[:, 1]
    out = pd.DataFrame({"lot_number": np.arange(1, len(lots) + 1), "risk_score": proba})
    out["risk_rank"] = out["risk_score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("risk_rank").reset_index(drop=True)
