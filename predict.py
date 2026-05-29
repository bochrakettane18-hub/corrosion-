"""
predict.py - Inference layer for the corrosion-prediction system.

Loads the trained artifacts (corrosion_rate, thickness_loss_rate, risk models +
metadata) once, then turns raw inputs into the five engineering outputs:

  1. corrosion_rate        (mm/year)  - regression
  2. thickness_loss_rate   (mm/year)  - regression
  3. risk_level (NACE)                - from the corrosion rate via NACE RP0775 bins
  4. failure_probability              - P(high)+P(severe) from the risk classifier
  5. rul_years                        - usable wall thickness / corrosion rate

Both a single-segment dict (manual form) and a batch DataFrame (file upload)
are supported. All column handling routes through schema.py so inputs never
drift from what the models were trained on.
"""

from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

import schema

MODELS_DIR = Path(__file__).parent / "models"

# Classes the risk classifier emits that we treat as "needs intervention".
AGGRESSIVE_CLASSES = ("high", "severe")


@lru_cache(maxsize=1)
def load_artifacts():
    """Load models + metadata once and cache them for the process lifetime."""
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {MODELS_DIR}. Run train.py first."
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    risk_bundle = joblib.load(MODELS_DIR / "risk_model.joblib")
    return {
        "corrosion_rate": joblib.load(MODELS_DIR / "corrosion_rate_model.joblib"),
        "thickness_loss_rate": joblib.load(MODELS_DIR / "thickness_loss_rate_model.joblib"),
        "risk_pipeline": risk_bundle["pipeline"],
        "risk_classes": list(risk_bundle["classes"]),
        "meta": meta,
    }


def _frame_from_inputs(inputs) -> pd.DataFrame:
    """Coerce a dict (one segment) or DataFrame (batch) into a clean feature frame.

    Reindexes to schema.FEATURE_KEYS, numerics -> numeric (NaN ok, imputer fills),
    categoricals -> str (handle_unknown='ignore' tolerates unseen labels).
    """
    if isinstance(inputs, dict):
        df = pd.DataFrame([inputs])
    else:
        df = inputs.copy()
    X = df.reindex(columns=schema.FEATURE_KEYS)
    for k in schema.NUMERIC_KEYS:
        X[k] = pd.to_numeric(X[k], errors="coerce")
    for k in schema.CATEGORICAL_KEYS:
        X[k] = X[k].astype("object").where(X[k].notna(), None)
        X[k] = X[k].astype(str)
    return X


def _failure_probability(proba_row, classes) -> float:
    """Aggregate classifier probabilities into a single intervention-risk score.

    Defined as P(high) + P(severe): the probability the segment sits in an
    aggressive-corrosion regime. This is an interpretable risk score, not a
    time-to-leak probability (we have no failure-time labels to calibrate that).
    """
    idx = {c: i for i, c in enumerate(classes)}
    return float(sum(proba_row[idx[c]] for c in AGGRESSIVE_CLASSES if c in idx))


def predict_batch(df: pd.DataFrame, wall_thickness_mm=None, min_allowable_mm=0.0) -> pd.DataFrame:
    """Predict all outputs for every row of a DataFrame.

    wall_thickness_mm / min_allowable_mm may be scalars or per-row array-likes.
    If wall_thickness_mm is None, RUL is left as NaN (rate alone is meaningless
    for life without a wall to consume).
    """
    art = load_artifacts()
    X = _frame_from_inputs(df)
    n = len(X)

    rate = np.asarray(art["corrosion_rate"].predict(X), dtype=float)
    tloss = np.asarray(art["thickness_loss_rate"].predict(X), dtype=float)
    rate = np.clip(rate, 0.0, None)
    tloss = np.clip(tloss, 0.0, None)

    classes = art["risk_classes"]
    proba = art["risk_pipeline"].predict_proba(X)
    fail_prob = np.array([_failure_probability(proba[i], classes) for i in range(n)])
    risk_pred = [classes[i] for i in proba.argmax(axis=1)]

    # NACE risk straight from the predicted rate (physics-anchored, monotonic).
    risk_nace = [schema.risk_from_rate(r) for r in rate]

    out = pd.DataFrame({
        "corrosion_rate_mm_yr": rate,
        "thickness_loss_rate_mm_yr": tloss,
        "risk_level": risk_nace,
        "risk_class_pred": risk_pred,
        "failure_probability": fail_prob,
    }, index=X.index)

    if wall_thickness_mm is not None:
        wt = np.broadcast_to(np.asarray(wall_thickness_mm, dtype=float), (n,))
        ma = np.broadcast_to(np.asarray(min_allowable_mm, dtype=float), (n,))
        out["rul_years"] = [
            schema.rul_years(rate[i], wt[i], ma[i]) for i in range(n)
        ]
    else:
        out["rul_years"] = np.nan

    return out


def predict_one(inputs: dict, wall_thickness_mm=None, min_allowable_mm=0.0) -> dict:
    """Predict all outputs for a single pipeline segment (manual form path)."""
    row = predict_batch(_frame_from_inputs(inputs),
                        wall_thickness_mm=wall_thickness_mm,
                        min_allowable_mm=min_allowable_mm).iloc[0]
    return row.to_dict()


def _base_key(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to its original feature key.

    'num__temperature' -> 'temperature'; 'cat__material_Acier 13Cr' -> 'material'.
    """
    if transformed_name.startswith("num__"):
        return transformed_name[5:]
    if transformed_name.startswith("cat__"):
        rest = transformed_name[5:]
        for k in schema.CATEGORICAL_KEYS:
            if rest.startswith(k + "_"):
                return k
        return rest
    return transformed_name


@lru_cache(maxsize=1)
def _corrosion_explainer():
    """Build (and cache) a SHAP TreeExplainer for the corrosion-rate model.

    Explains the XGBoost stage on the post-preprocessing feature space; values
    are in the model's log1p-rate space (the target was log-transformed).
    """
    import shap
    art = load_artifacts()
    pipeline = art["corrosion_rate"].regressor_      # fitted Pipeline inside TTR
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    feat_names = list(prep.get_feature_names_out())
    return prep, explainer, feat_names


def explain_one(inputs, top_n: int = 8):
    """Top-N feature contributions to the corrosion-rate prediction.

    Returns a list of (feature_key, contribution) sorted by |contribution|,
    with one-hot columns summed back onto their parent categorical feature.
    Contribution sign: positive pushes the predicted rate up, negative down.
    """
    prep, explainer, feat_names = _corrosion_explainer()
    X = _frame_from_inputs(inputs)
    Xt = prep.transform(X)
    sv = np.asarray(explainer.shap_values(Xt))[0]
    agg = {}
    for name, val in zip(feat_names, sv):
        k = _base_key(name)
        agg[k] = agg.get(k, 0.0) + float(val)
    return sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import data_loader
    df = data_loader.load_clean()
    X, y = data_loader.get_xy(df)

    sample = X.iloc[0].to_dict()
    print("sample input:", {k: sample[k] for k in list(sample)[:6]}, "...")
    res = predict_one(sample, wall_thickness_mm=12.0, min_allowable_mm=3.0)
    print("\nsingle-segment prediction:")
    for k, v in res.items():
        print(f"  {k:28s} {v}")
    print(f"\n  actual corrosion_rate       {y['corrosion_rate'].iloc[0]:.3f}")

    print("\nbatch prediction on first 5 rows:")
    batch = predict_batch(X.head(5), wall_thickness_mm=12.0, min_allowable_mm=3.0)
    print(batch.to_string())
