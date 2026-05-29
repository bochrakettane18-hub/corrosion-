"""
train.py - Train corrosion models on the primary dataset and save artifacts.

Targets:
  - corrosion_rate (mm/yr)      regression  [RF + XGBoost, log-target] -> best saved
  - thickness_loss_rate (mm/yr) regression  [RF + XGBoost, log-target] -> best saved
  - risk_level (NACE)           classification [XGBoost] -> drives failure probability

Artifacts in models/:
  corrosion_rate_model.joblib, thickness_loss_rate_model.joblib, risk_model.joblib,
  metadata.json (feature lists, categories, ranges, metrics).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score)
from xgboost import XGBRegressor, XGBClassifier

import schema
import data_loader

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), schema.NUMERIC_KEYS),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         schema.CATEGORICAL_KEYS),
    ])


def eval_reg(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "R2": float(r2_score(y_true, y_pred))}


def train_regression_target(X_tr, X_te, y_tr, y_te, name):
    candidates = {
        "random_forest": RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                               random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                random_state=RANDOM_STATE),
    }
    results, best, best_model = {}, None, None
    for mname, reg in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", reg)])
        ttr = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
        ttr.fit(X_tr, y_tr)
        m = eval_reg(y_te, ttr.predict(X_te))
        results[mname] = m
        print(f"  [{name}] {mname}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} R2={m['R2']:.3f}")
        if best is None or m["RMSE"] < results[best]["RMSE"]:
            best, best_model = mname, ttr
    joblib.dump(best_model, MODELS_DIR / f"{name}_model.joblib")
    print(f"  -> saved best ({best}) for {name}")
    return {"best": best, "metrics": results}


def train_risk(X_tr, X_te, y_tr, y_te):
    le = LabelEncoder().fit(y_tr)
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("model", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                             random_state=RANDOM_STATE, eval_metric="mlogloss"))])
    pipe.fit(X_tr, le.transform(y_tr))
    pred = pipe.predict(X_te)
    yte = le.transform(y_te)
    acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
    print(f"  [risk] xgboost: accuracy={acc:.3f} macroF1={f1:.3f}")
    joblib.dump({"pipeline": pipe, "classes": list(le.classes_)}, MODELS_DIR / "risk_model.joblib")
    return {"accuracy": float(acc), "macro_f1": float(f1), "classes": list(le.classes_)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    df = data_loader.load_clean()
    X, y = data_loader.get_xy(df)
    print("data matrix:", X.shape)

    meta = {
        "feature_keys": schema.FEATURE_KEYS,
        "numeric_keys": schema.NUMERIC_KEYS,
        "categorical_keys": schema.CATEGORICAL_KEYS,
        "categories": data_loader.categories(df),
        "numeric_ranges": data_loader.numeric_ranges(df),
        "risk_bins": [list(b) for b in schema.RISK_BINS],
        "targets": {},
    }

    for tgt in ["corrosion_rate", "thickness_loss_rate"]:
        print(f"\nTraining {tgt} ...")
        Xtr, Xte, ytr, yte = train_test_split(X, y[tgt], test_size=0.2, random_state=RANDOM_STATE)
        meta["targets"][tgt] = train_regression_target(Xtr, Xte, ytr, yte, tgt)

    print("\nTraining risk classifier ...")
    Xtr, Xte, ytr, yte = train_test_split(X, y["risk_level"], test_size=0.2,
                                          random_state=RANDOM_STATE, stratify=y["risk_level"])
    meta["risk"] = train_risk(Xtr, Xte, ytr, yte)

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts in {MODELS_DIR}")


if __name__ == "__main__":
    main()
