"""
app.py - Streamlit front-end for the corrosion-prediction system.

Two entry paths, both feeding the same trained models via predict.py:
  - Single segment: a grouped manual form (Asset / Fluid / Operating / Chemistry).
  - Batch: upload a CSV/XLSX export and score every row, with a CSV download.

Outputs per segment: corrosion rate, thickness-loss rate, NACE risk level,
failure-probability score, and remaining useful life (RUL).

Run:  streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import schema
import predict

st.set_page_config(page_title="Corrosion PdM", page_icon="🛢️", layout="wide")

# ---------------------------------------------------------------- i18n
T = {
    "en": {
        "title": "Pipeline Corrosion — Predictive Maintenance",
        "subtitle": "Predict corrosion rate, thickness loss, risk, failure probability and remaining life.",
        "single": "Single segment",
        "batch": "Batch upload",
        "wall": "Wall thickness",
        "minwall": "Min. allowable thickness",
        "predict": "Predict",
        "results": "Results",
        "corr_rate": "Corrosion rate",
        "tloss": "Thickness-loss rate",
        "risk": "Risk level (NACE)",
        "failprob": "Failure-risk score",
        "rul": "Remaining useful life",
        "years": "years",
        "upload_help": "Upload a CSV or Excel file. Columns are matched to model features by name; unknown columns are ignored and missing ones are imputed.",
        "run_batch": "Score file",
        "download": "Download results (CSV)",
        "rows_scored": "rows scored",
        "model_perf": "Model performance",
        "no_models": "Model artifacts not found. Run `python train.py` first.",
        "rul_note": "RUL = (wall − min allowable) / corrosion rate.",
        "failprob_note": "P(high)+P(severe) from the risk classifier — an intervention-priority score, not a calibrated time-to-leak probability.",
        "template": "Need a template? Download a sample input file.",
        "dl_template": "Download input template (CSV)",
        "inf": "no measurable loss",
        "explain_title": "Why this prediction? (SHAP)",
        "explain_note": "Contribution of each feature to the predicted corrosion rate (model's log scale). Red drives the rate up, blue brings it down.",
        "impact": "Impact on corrosion rate",
        "lstm_tab": "RUL forecast (LSTM)",
        "lstm_intro": "Forecast remaining useful life from a wall-thickness inspection history. The LSTM was trained on synthesized degradation trajectories; below, an inspection history is simulated from your inputs and fed to the model.",
        "init_thk": "Initial wall thickness",
        "age": "Service age so far",
        "regime": "Degradation regime",
        "base_rate": "Corrosion rate",
        "run_forecast": "Forecast RUL",
        "lstm_rul": "LSTM-estimated RUL",
        "analytic_rul": "Analytical RUL",
        "thk_axis": "Wall thickness (mm)",
        "time_axis": "Time (years)",
        "history": "Inspection history",
        "projection": "Projected degradation",
        "min_allow_line": "Min. allowable",
        "lstm_unavailable": "LSTM model not trained yet. Run `python lstm_rul.py` to enable this tab.",
        "yr": "yr",
    },
    "fr": {
        "title": "Corrosion des pipelines — Maintenance prédictive",
        "subtitle": "Prédire la vitesse de corrosion, la perte d'épaisseur, le risque, la probabilité de défaillance et la durée de vie restante.",
        "single": "Segment unique",
        "batch": "Import par fichier",
        "wall": "Épaisseur de paroi",
        "minwall": "Épaisseur minimale admissible",
        "predict": "Prédire",
        "results": "Résultats",
        "corr_rate": "Vitesse de corrosion",
        "tloss": "Taux de perte d'épaisseur",
        "risk": "Niveau de risque (NACE)",
        "failprob": "Score de risque de défaillance",
        "rul": "Durée de vie restante",
        "years": "ans",
        "upload_help": "Importez un fichier CSV ou Excel. Les colonnes sont associées aux variables du modèle par leur nom ; les colonnes inconnues sont ignorées et les manquantes sont imputées.",
        "run_batch": "Analyser le fichier",
        "download": "Télécharger les résultats (CSV)",
        "rows_scored": "lignes analysées",
        "model_perf": "Performance du modèle",
        "no_models": "Artefacts du modèle introuvables. Exécutez d'abord `python train.py`.",
        "rul_note": "DVR = (épaisseur − minimale admissible) / vitesse de corrosion.",
        "failprob_note": "P(élevé)+P(sévère) du classifieur de risque — un score de priorité d'intervention, pas une probabilité de fuite calibrée dans le temps.",
        "template": "Besoin d'un modèle ? Téléchargez un fichier d'exemple.",
        "dl_template": "Télécharger le modèle d'entrée (CSV)",
        "inf": "aucune perte mesurable",
        "explain_title": "Pourquoi cette prédiction ? (SHAP)",
        "explain_note": "Contribution de chaque variable à la vitesse de corrosion prédite (échelle logarithmique du modèle). Le rouge augmente la vitesse, le bleu la diminue.",
        "impact": "Impact sur la vitesse de corrosion",
        "lstm_tab": "Prévision DVR (LSTM)",
        "lstm_intro": "Prévoir la durée de vie restante à partir d'un historique d'inspection d'épaisseur. Le LSTM a été entraîné sur des trajectoires de dégradation synthétiques ; ci-dessous, un historique est simulé à partir de vos entrées puis fourni au modèle.",
        "init_thk": "Épaisseur initiale de paroi",
        "age": "Âge en service",
        "regime": "Régime de dégradation",
        "base_rate": "Vitesse de corrosion",
        "run_forecast": "Prévoir la DVR",
        "lstm_rul": "DVR estimée par LSTM",
        "analytic_rul": "DVR analytique",
        "thk_axis": "Épaisseur de paroi (mm)",
        "time_axis": "Temps (années)",
        "history": "Historique d'inspection",
        "projection": "Dégradation projetée",
        "min_allow_line": "Minimale admissible",
        "lstm_unavailable": "Modèle LSTM pas encore entraîné. Exécutez `python lstm_rul.py` pour activer cet onglet.",
        "yr": "ans",
    },
}

REGIME_LABEL = {
    "en": {"linear": "Linear (uniform)", "pitting": "Accelerating (pitting)", "passivation": "Decelerating (passivation)"},
    "fr": {"linear": "Linéaire (uniforme)", "pitting": "Accélérée (piqûration)", "passivation": "Décélérée (passivation)"},
}

# feature key -> human label, per language (built from the schema)
FEATURE_LABELS = {
    "en": {f.key: f.label_en for f in (schema.NUMERIC_FEATURES + schema.CATEGORICAL_FEATURES)},
    "fr": {f.key: f.label_fr for f in (schema.NUMERIC_FEATURES + schema.CATEGORICAL_FEATURES)},
}

RISK_COLORS = {"low": "#2e7d32", "moderate": "#f9a825", "high": "#ef6c00", "severe": "#c62828"}
RISK_LABEL = {
    "en": {"low": "Low", "moderate": "Moderate", "high": "High", "severe": "Severe"},
    "fr": {"low": "Faible", "moderate": "Modéré", "high": "Élevé", "severe": "Sévère"},
}


@st.cache_resource
def get_artifacts():
    return predict.load_artifacts()


def fmt_rul(years: float, t) -> str:
    if years is None or (isinstance(years, float) and (np.isinf(years) or np.isnan(years))):
        return t["inf"]
    return f"{years:.1f} {t['years']}"


def corrosion_gauge(rate: float) -> go.Figure:
    axis_max = max(0.5, float(rate) * 1.3)
    cap = lambda x: min(x, axis_max)
    steps = [
        {"range": [0, cap(0.025)], "color": "#c8e6c9"},
        {"range": [cap(0.025), cap(0.125)], "color": "#fff9c4"},
        {"range": [cap(0.125), cap(0.25)], "color": "#ffe0b2"},
        {"range": [cap(0.25), axis_max], "color": "#ffcdd2"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(rate),
        number={"suffix": " mm/yr", "font": {"size": 26}},
        gauge={
            "axis": {"range": [0, axis_max]},
            "bar": {"color": "#37474f"},
            "steps": steps,
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=10))
    return fig


def prob_gauge(p: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(p) * 100,
        number={"suffix": " %", "font": {"size": 26}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#37474f"},
            "steps": [
                {"range": [0, 33], "color": "#c8e6c9"},
                {"range": [33, 66], "color": "#fff9c4"},
                {"range": [66, 100], "color": "#ffcdd2"},
            ],
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=10))
    return fig


def render_explanation(inputs: dict, lang: str, t):
    """SHAP feature-contribution bar chart for the corrosion-rate prediction."""
    try:
        items = predict.explain_one(inputs, top_n=8)
    except Exception as e:
        st.caption(f"Explanation unavailable: {e}")
        return
    items = items[::-1]  # largest contribution on top of a horizontal bar
    labels = [FEATURE_LABELS[lang].get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#c62828" if v > 0 else "#1565c0" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=t["impact"])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(t["explain_note"])


def render_outputs(res: dict, lang: str, t):
    risk = res["risk_level"]
    color = RISK_COLORS.get(risk, "#555")
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{color};color:white;"
        f"font-size:1.3rem;font-weight:700;text-align:center;margin-bottom:8px;'>"
        f"{t['risk']}: {RISK_LABEL[lang].get(risk, risk).upper()}</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption(t["corr_rate"])
        st.plotly_chart(corrosion_gauge(res["corrosion_rate_mm_yr"]), use_container_width=True)
    with c2:
        st.caption(t["failprob"])
        st.plotly_chart(prob_gauge(res["failure_probability"]), use_container_width=True)
        st.caption(t["failprob_note"])

    m1, m2 = st.columns(2)
    m1.metric(t["tloss"], f"{res['thickness_loss_rate_mm_yr']:.3f} mm/yr")
    m2.metric(t["rul"], fmt_rul(res.get("rul_years"), t))
    st.caption(t["rul_note"])


def single_form(lang: str, t, meta):
    cats = meta["categories"]
    ranges = meta["numeric_ranges"]
    values = {}

    with st.form("single"):
        for group in schema.GROUPS:
            feats = schema.features_in_group(group)
            if not feats:
                continue
            st.subheader(group)
            cols = st.columns(2)
            for i, f in enumerate(feats):
                col = cols[i % 2]
                label = (f.label_en if lang == "en" else f.label_fr)
                unit = getattr(f, "unit", "")
                if unit:
                    label = f"{label} ({unit})"
                if f.kind == "numeric":
                    lo, hi = ranges.get(f.key, [f.min, f.max])
                    step = 0.1 if hi <= 20 else 1.0
                    values[f.key] = col.number_input(
                        label, min_value=float(lo), max_value=float(hi),
                        value=float(min(max(f.default, lo), hi)), step=step, key=f.key,
                    )
                else:
                    opts = cats.get(f.key, [])
                    values[f.key] = col.selectbox(label, opts, key=f.key)

        st.subheader("RUL")
        rc = st.columns(2)
        wall = rc[0].number_input(f"{t['wall']} (mm)", min_value=0.1, max_value=200.0,
                                  value=12.7, step=0.1)
        minwall = rc[1].number_input(f"{t['minwall']} (mm)", min_value=0.0, max_value=200.0,
                                     value=3.0, step=0.1)
        submitted = st.form_submit_button(t["predict"], type="primary", use_container_width=True)

    if submitted:
        res = predict.predict_one(values, wall_thickness_mm=wall, min_allowable_mm=minwall)
        st.divider()
        st.subheader(t["results"])
        render_outputs(res, lang, t)
        with st.expander(t["explain_title"], expanded=True):
            render_explanation(values, lang, t)
        report = {**values, "wall_thickness_mm": wall, "min_allowable_mm": minwall, **res}
        st.download_button(t["download"], pd.DataFrame([report]).to_csv(index=False).encode("utf-8"),
                           file_name="corrosion_report.csv", mime="text/csv")


def _template_df(meta) -> pd.DataFrame:
    cats = meta["categories"]
    ranges = meta["numeric_ranges"]
    row = {}
    for f in schema.NUMERIC_FEATURES:
        lo, hi = ranges.get(f.key, [f.min, f.max])
        row[f.key] = round(min(max(f.default, lo), hi), 3)
    for f in schema.CATEGORICAL_FEATURES:
        opts = cats.get(f.key, [])
        row[f.key] = opts[0] if opts else ""
    return pd.DataFrame([row, row])


def batch_upload(lang: str, t, meta):
    st.info(t["upload_help"])

    with st.expander(t["template"]):
        tpl = _template_df(meta)
        st.download_button(t["dl_template"], tpl.to_csv(index=False).encode("utf-8"),
                           file_name="corrosion_input_template.csv", mime="text/csv")

    rc = st.columns(2)
    wall = rc[0].number_input(f"{t['wall']} (mm)", min_value=0.1, max_value=200.0,
                              value=12.7, step=0.1, key="b_wall")
    minwall = rc[1].number_input(f"{t['minwall']} (mm)", min_value=0.0, max_value=200.0,
                                 value=3.0, step=0.1, key="b_minwall")

    up = st.file_uploader("CSV / XLSX", type=["csv", "xlsx", "xls"])
    if up is None:
        return

    if up.name.lower().endswith(("xlsx", "xls")):
        df = pd.read_excel(up)
    else:
        df = pd.read_csv(up)

    if st.button(t["run_batch"], type="primary"):
        wcol = df["wall_thickness_mm"] if "wall_thickness_mm" in df.columns else wall
        mcol = df["min_allowable_mm"] if "min_allowable_mm" in df.columns else minwall
        preds = predict.predict_batch(df, wall_thickness_mm=wcol, min_allowable_mm=mcol)
        merged = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

        st.success(f"{len(merged)} {t['rows_scored']}")
        counts = preds["risk_level"].value_counts()
        cols = st.columns(4)
        for i, name in enumerate(["low", "moderate", "high", "severe"]):
            cols[i].metric(RISK_LABEL[lang][name], int(counts.get(name, 0)))

        st.dataframe(merged, use_container_width=True, height=380)
        st.download_button(t["download"], merged.to_csv(index=False).encode("utf-8"),
                           file_name="corrosion_predictions.csv", mime="text/csv")


def lstm_forecast_tab(lang: str, t):
    import lstm_rul
    from pathlib import Path

    st.info(t["lstm_intro"])
    if not (Path(lstm_rul.MODELS_DIR) / "lstm_rul.keras").exists():
        st.warning(t["lstm_unavailable"])
        return

    c = st.columns(4)
    T0 = c[0].number_input(f"{t['init_thk']} (mm)", 4.0, 50.0, 12.7, 0.1)
    minw = c[1].number_input(f"{t['minwall']} (mm)", 0.0, 40.0, 6.0, 0.1)
    rate = c[2].number_input(f"{t['base_rate']} (mm/yr)", 0.01, 10.0, 0.40, 0.01)
    age = c[3].number_input(f"{t['age']} ({t['yr']})", 1.0, 25.0, 4.0, 0.5)
    regime = st.selectbox(t["regime"], lstm_rul.REGIMES,
                          format_func=lambda r: REGIME_LABEL[lang][r])

    if not st.button(t["run_forecast"], type="primary"):
        return

    rng = np.random.default_rng(0)
    t_hist, truth_hist = lstm_rul.simulate_trajectory(T0, rate, minw, regime, years=age, rng=rng)
    measured = truth_hist + rng.normal(0.0, lstm_rul.MEAS_NOISE, size=truth_hist.shape)
    cur_thk = float(measured[-1])

    lstm_years = lstm_rul.forecast_rul(measured, minw)
    analytic = (cur_thk - minw) / rate if rate > 0 else float("inf")

    horizon = min(age + max(lstm_years, analytic, 1.0) + 2.0, lstm_rul.MAX_YEARS)
    t_full, truth_full = lstm_rul.simulate_trajectory(T0, rate, minw, regime, years=horizon)
    future = t_full >= age

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_hist, y=measured, mode="markers+lines",
                             name=t["history"], marker=dict(size=5, color="#1565c0")))
    fig.add_trace(go.Scatter(x=t_full[future], y=truth_full[future], mode="lines",
                             name=t["projection"], line=dict(color="#ef6c00", dash="dash")))
    fig.add_hline(y=minw, line=dict(color="#c62828", dash="dot"),
                  annotation_text=t["min_allow_line"], annotation_position="bottom right")
    fig.add_vline(x=age + lstm_years, line=dict(color="#2e7d32"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=t["time_axis"], yaxis_title=t["thk_axis"],
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric(t["lstm_rul"], f"{lstm_years:.1f} {t['years']}")
    m2.metric(t["analytic_rul"], f"{analytic:.1f} {t['years']}")


def main():
    lang = st.sidebar.radio("Language / Langue", ["en", "fr"],
                            format_func=lambda x: "English" if x == "en" else "Français")
    t = T[lang]

    st.title(t["title"])
    st.caption(t["subtitle"])

    try:
        art = get_artifacts()
    except FileNotFoundError:
        st.error(t["no_models"])
        st.stop()
    meta = art["meta"]

    with st.sidebar:
        st.divider()
        st.subheader(t["model_perf"])
        cr = meta["targets"]["corrosion_rate"]
        crm = cr["metrics"][cr["best"]]
        st.metric(f"{t['corr_rate']} R²", f"{crm['R2']:.3f}")
        st.metric(f"{t['risk']} acc.", f"{meta['risk']['accuracy']:.1%}")
        st.caption(f"corrosion MAE {crm['MAE']:.3f} mm/yr · best: {cr['best']}")
        lstm_meta = Path(predict.MODELS_DIR) / "lstm_rul_meta.json"
        if lstm_meta.exists():
            with open(lstm_meta, encoding="utf-8") as f:
                lm = json.load(f)
            st.metric(f"{t['lstm_rul']} MAE", f"{lm['metrics']['mae_years']:.2f} {t['years']}")

    tab1, tab2, tab3 = st.tabs([t["single"], t["batch"], t["lstm_tab"]])
    with tab1:
        single_form(lang, t, meta)
    with tab2:
        batch_upload(lang, t, meta)
    with tab3:
        lstm_forecast_tab(lang, t)


if __name__ == "__main__":
    main()
