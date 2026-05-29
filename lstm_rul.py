"""
lstm_rul.py - LSTM for Remaining Useful Life (RUL) from wall-thickness inspection series.

The primary dataset is cross-sectional (one row per segment), so there are no real
degradation sequences. We synthesize physically-motivated wall-thickness trajectories:
each segment loses metal at a base corrosion rate (drawn from the trained
corrosion-rate model's predictions on the real data) modulated by a degradation
regime (linear / accelerating pitting / decelerating passivation), seasonal
operating swings, and gauge measurement noise.

An LSTM then learns to map a window of recent (noisy) thickness readings to the
remaining time until the wall reaches its minimum allowable thickness.

Artifacts:
  models/lstm_rul.keras       - trained Keras model
  models/lstm_rul_meta.json   - window size, channel/target scaling, test metrics
"""

import json
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
RANDOM_STATE = 42

WINDOW = 12          # months of history fed to the LSTM (1 year of monthly readings)
DT = 1.0 / 12.0      # inspection interval in years (monthly)
MAX_YEARS = 30.0     # simulation horizon
MEAS_NOISE = 0.04    # gauge noise std (mm)
REGIMES = ("linear", "pitting", "passivation")


def _season(t, amp, phase):
    return 1.0 + amp * np.sin(2.0 * np.pi * t + phase)


def simulate_trajectory(T0, r0, min_allow, regime="linear", years=MAX_YEARS,
                        dt=DT, seasonal_amp=0.0, phase=0.0, rng=None):
    """Noise-free underlying wall-thickness trajectory for one segment.

    Returns (t, thickness) sampled every dt up to `years` (or until well past
    the min-allowable wall). thickness is monotonically non-increasing in the mean.
    """
    n = int(round(years / dt)) + 1
    t = np.arange(n) * dt
    if regime == "pitting":
        accel = 0.20 if rng is None else rng.uniform(0.10, 0.35)
        rate = r0 * (1.0 + accel * t)
    elif regime == "passivation":
        floor = 0.25 * r0
        decay = 0.5 if rng is None else rng.uniform(0.3, 0.9)
        rate = floor + (r0 - floor) * np.exp(-decay * t)
    else:  # linear
        rate = np.full_like(t, r0)
    rate = rate * _season(t, seasonal_amp, phase)
    thickness = T0 - np.cumsum(rate) * dt
    return t, thickness


def _failure_time(t, thickness, min_allow):
    """First time thickness drops to/below min_allow; inf if it never does."""
    below = np.where(thickness <= min_allow)[0]
    return t[below[0]] if len(below) else np.inf


def _build_windows(n_trajectories=4000, max_windows=150000, seed=RANDOM_STATE):
    """Generate synthetic trajectories and slice them into (window, RUL) samples.

    Channels per timestep:
      ch0 = remaining wall above min allowable (mm)
      ch1 = per-step thickness loss (mm)  -> recent corrosion-rate signal
    Target: RUL in years at the end of the window (from the noise-free truth).
    """
    rng = np.random.default_rng(seed)

    # Base corrosion rates: use the trained model's predictions on real data so the
    # synthetic rates follow a realistic distribution; fall back to a lognormal.
    try:
        import data_loader, predict
        X, _ = data_loader.get_xy(data_loader.load_clean())
        base_rates = np.asarray(
            predict.predict_batch(X.sample(min(2000, len(X)), random_state=seed))
            ["corrosion_rate_mm_yr"], dtype=float)
        base_rates = base_rates[base_rates > 0.02]
    except Exception:
        base_rates = rng.lognormal(mean=-0.7, sigma=0.9, size=2000)

    Xs, ys = [], []
    for _ in range(n_trajectories):
        r0 = float(rng.choice(base_rates)) * rng.uniform(0.7, 1.3)
        r0 = max(r0, 0.02)
        T0 = rng.uniform(8.0, 20.0)
        min_allow = T0 * rng.uniform(0.4, 0.6)
        regime = REGIMES[rng.integers(len(REGIMES))]
        amp = rng.uniform(0.0, 0.15)
        phase = rng.uniform(0.0, 2 * np.pi)

        t, truth = simulate_trajectory(T0, r0, min_allow, regime, MAX_YEARS, DT,
                                       seasonal_amp=amp, phase=phase, rng=rng)
        t_fail = _failure_time(t, truth, min_allow)
        if not np.isfinite(t_fail):
            continue
        measured = truth + rng.normal(0.0, MEAS_NOISE, size=truth.shape)

        rem = measured - min_allow
        loss = np.diff(measured, prepend=measured[0])
        last = int(np.searchsorted(t, t_fail))  # stop at failure
        for k in range(WINDOW - 1, last):
            rul = t_fail - t[k]
            if rul < 0:
                continue
            sl = slice(k - WINDOW + 1, k + 1)
            Xs.append(np.stack([rem[sl], loss[sl]], axis=-1))
            ys.append(rul)
        if len(Xs) >= max_windows:
            break

    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return X, y


def _standardize(X, y):
    ch_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    ch_std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-8)
    Xs = (X - ch_mean) / ch_std
    ys = (y - y_mean) / y_std
    scaling = {"ch_mean": ch_mean.tolist(), "ch_std": ch_std.tolist(),
               "y_mean": y_mean, "y_std": y_std}
    return Xs, ys, scaling


def build_model(window, n_ch):
    from tensorflow import keras
    from tensorflow.keras import layers
    m = keras.Sequential([
        keras.Input((window, n_ch)),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(RANDOM_STATE)

    print("Generating synthetic degradation trajectories ...")
    X, y = _build_windows()
    print(f"  windows: {X.shape}, RUL years range [{y.min():.2f}, {y.max():.2f}]")

    Xs, ys, scaling = _standardize(X, y)
    n = len(Xs)
    idx = np.random.default_rng(RANDOM_STATE).permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]

    model = build_model(WINDOW, X.shape[-1])
    es = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
    model.fit(Xs[tr], ys[tr], validation_split=0.2, epochs=40, batch_size=256,
              callbacks=[es], verbose=2)

    pred_te = model.predict(Xs[te], verbose=0).ravel() * scaling["y_std"] + scaling["y_mean"]
    true_te = y[te]
    mae = float(np.mean(np.abs(pred_te - true_te)))
    rmse = float(np.sqrt(np.mean((pred_te - true_te) ** 2)))
    print(f"\n  test MAE = {mae:.3f} years, RMSE = {rmse:.3f} years")

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "lstm_rul.keras")
    meta = {"window": WINDOW, "dt": DT, "channels": ["remaining_mm", "loss_per_step_mm"],
            "regimes": list(REGIMES), "scaling": scaling,
            "metrics": {"mae_years": mae, "rmse_years": rmse}}
    with open(MODELS_DIR / "lstm_rul_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  saved lstm_rul.keras + lstm_rul_meta.json to {MODELS_DIR}")


# --------------------------------------------------------------- inference
_MODEL = None
_META = None


def _load():
    global _MODEL, _META
    if _MODEL is None:
        from tensorflow import keras
        _MODEL = keras.models.load_model(MODELS_DIR / "lstm_rul.keras")
        with open(MODELS_DIR / "lstm_rul_meta.json", encoding="utf-8") as f:
            _META = json.load(f)
    return _MODEL, _META


def forecast_rul(measured_series, min_allowable):
    """RUL (years) from a sequence of recent wall-thickness measurements (mm).

    Uses the last `window` readings. Returns a non-negative float.
    """
    model, meta = _load()
    w = meta["window"]
    s = np.asarray(measured_series, dtype=float)
    if len(s) < w:
        s = np.pad(s, (w - len(s), 0), mode="edge")
    s = s[-w:]
    rem = s - float(min_allowable)
    loss = np.diff(s, prepend=s[0])
    X = np.stack([rem, loss], axis=-1)[None, ...]
    ch_mean = np.asarray(meta["scaling"]["ch_mean"])
    ch_std = np.asarray(meta["scaling"]["ch_std"])
    Xs = (X - ch_mean) / ch_std
    pred = float(model.predict(Xs, verbose=0).ravel()[0])
    pred = pred * meta["scaling"]["y_std"] + meta["scaling"]["y_mean"]
    return max(pred, 0.0)


if __name__ == "__main__":
    main()
