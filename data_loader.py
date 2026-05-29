"""
data_loader.py - Load and clean the primary corrosion dataset (corrosion_10000.xlsx).

Handles the title row above the header, renames French columns to clean keys,
coerces types, fixes the CO2>100% artifact, and derives the NACE risk class.
"""

from pathlib import Path
import pandas as pd

import schema

RAW_XLSX = Path(__file__).parent / "data" / "raw" / "corrosion_10000.xlsx"


def _find_header_row(path, sheet=0, key="ID", max_scan=8):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=max_scan)
    for i in range(len(raw)):
        if str(raw.iloc[i, 0]).strip() == key:
            return i
    return 0


def load_clean(path=RAW_XLSX) -> pd.DataFrame:
    """Return a tidy DataFrame with clean ASCII columns + derived risk_level."""
    path = Path(path)
    hdr = _find_header_row(path)
    df = pd.read_excel(path, sheet_name=0, header=hdr).dropna(how="all")
    df = df.rename(columns=schema.COLUMN_MAP)
    df = df[[c for c in schema.COLUMN_MAP.values() if c in df.columns]]

    numeric_cols = schema.NUMERIC_KEYS + ["corrosion_rate", "thickness_loss_rate"]
    for k in numeric_cols:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    for k in schema.CATEGORICAL_KEYS + ["mechanism"]:
        if k in df.columns:
            df[k] = df[k].astype(str).str.strip()

    # CO2 is a percentage; values above 100 are data artifacts -> clip.
    if "co2" in df.columns:
        df["co2"] = df["co2"].clip(upper=100)

    df = df.dropna(subset=["corrosion_rate"]).reset_index(drop=True)
    df["risk_level"] = df["corrosion_rate"].apply(schema.risk_from_rate)
    return df


def get_xy(df: pd.DataFrame):
    """Split into X (model inputs) and a dict of target Series."""
    X = df[[k for k in schema.FEATURE_KEYS if k in df.columns]].copy()
    y = {
        "corrosion_rate": df["corrosion_rate"],
        "thickness_loss_rate": df.get("thickness_loss_rate"),
        "risk_level": df["risk_level"],
        "mechanism": df.get("mechanism"),
    }
    return X, y


def categories(df: pd.DataFrame) -> dict:
    return {k: sorted(df[k].dropna().unique().tolist())
            for k in schema.CATEGORICAL_KEYS if k in df.columns}


def numeric_ranges(df: pd.DataFrame) -> dict:
    return {k: [float(df[k].min()), float(df[k].max())]
            for k in schema.NUMERIC_KEYS if k in df.columns}


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    df = load_clean()
    print("clean shape:", df.shape)
    print("features used:", [k for k in schema.FEATURE_KEYS if k in df.columns])
    print("\nrisk distribution (NACE):")
    print(df["risk_level"].value_counts().reindex([b[0] for b in schema.RISK_BINS]).to_string())
    print("\ncategory sizes:", {k: len(v) for k, v in categories(df).items()})
    cr = df["corrosion_rate"].describe()
    print("\ncorrosion_rate: min=%.3f median=%.3f mean=%.3f max=%.3f"
          % (cr["min"], cr["50%"], cr["mean"], cr["max"]))
