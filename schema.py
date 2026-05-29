"""
schema.py - Single source of truth for the corrosion-prediction system.

Built from the primary dataset (data/raw/corrosion_10000.xlsx, sheet
"Donnees_Corrosion"): 10,000 rows of physics-coherent synthetic data with
13 operating/chemistry features and a measured corrosion-rate target.

The data loader, the training pipeline, and the Streamlit app all import from
here, so column names, units, form fields, and model inputs never drift apart.
"""

from dataclasses import dataclass


# Raw French/Excel headers -> clean ASCII keys used everywhere in code.
COLUMN_MAP = {
    "ID": "id",
    "Matériau": "material",
    "Fluide": "fluid",
    "Phase": "phase",
    "Température (°C)": "temperature",
    "Pression (bar)": "pressure",
    "Densité (kg/m³)": "density",
    "Vitesse écoulement (m/s)": "flow_velocity",
    "Composition chimique": "chemistry",
    "pH": "ph",
    "Cl⁻ (mg/L)": "chloride",
    "CO₂ (%)": "co2",
    "H₂S (ppm)": "h2s",
    "O₂ (mg/L)": "o2",
    "Durée exposition (h)": "exposure_h",
    "Vitesse corrosion (mm/an)": "corrosion_rate",
    "Perte épaisseur (mm/an)": "thickness_loss_rate",
    "Remarques": "mechanism",
    "Norme / Référence": "reference",
}

GROUPS = ["Asset", "Fluid", "Operating conditions", "Fluid chemistry"]


@dataclass(frozen=True)
class NumericFeature:
    key: str
    label_en: str
    label_fr: str
    unit: str
    min: float
    max: float
    default: float
    group: str
    kind: str = "numeric"


@dataclass(frozen=True)
class CategoricalFeature:
    key: str
    label_en: str
    label_fr: str
    group: str
    # options are filled at runtime from models/metadata.json (derived from data)
    kind: str = "categorical"


# Ranges below are display hints for the form, taken from the dataset describe().
NUMERIC_FEATURES = [
    NumericFeature("temperature", "Temperature", "Température", "°C", 5, 210, 100, "Operating conditions"),
    NumericFeature("pressure", "Pressure", "Pression", "bar", 1, 350, 100, "Operating conditions"),
    NumericFeature("density", "Density", "Densité", "kg/m³", 0.7, 1236, 850, "Operating conditions"),
    NumericFeature("flow_velocity", "Flow velocity", "Vitesse d'écoulement", "m/s", 0.05, 12, 3.0, "Operating conditions"),
    NumericFeature("exposure_h", "Exposure duration", "Durée d'exposition", "h", 168, 17520, 1080, "Operating conditions"),
    NumericFeature("ph", "pH", "pH", "", 0.5, 14, 6.0, "Water chemistry"),
    NumericFeature("chloride", "Chloride (Cl⁻)", "Chlorures (Cl⁻)", "mg/L", 0, 180000, 300, "Water chemistry"),
    NumericFeature("co2", "CO₂", "CO₂", "%", 0, 100, 0.2, "Water chemistry"),
    NumericFeature("h2s", "H₂S", "H₂S", "ppm", 0, 6000, 0, "Water chemistry"),
    NumericFeature("o2", "Dissolved O₂", "O₂ dissous", "mg/L", 0, 14, 0.1, "Water chemistry"),
]

CATEGORICAL_FEATURES = [
    CategoricalFeature("material", "Material", "Matériau", "Asset"),
    CategoricalFeature("fluid", "Fluid", "Fluide", "Fluid"),
    CategoricalFeature("phase", "Phase", "Phase", "Fluid"),
    CategoricalFeature("chemistry", "Chemical composition", "Composition chimique", "Fluid"),
]

NUMERIC_KEYS = [f.key for f in NUMERIC_FEATURES]
CATEGORICAL_KEYS = [f.key for f in CATEGORICAL_FEATURES]
FEATURE_KEYS = NUMERIC_KEYS + CATEGORICAL_KEYS

# Identifiers / metadata / leakage / targets - never fed to the model as inputs.
DROP_FROM_FEATURES = ["id", "reference", "mechanism", "corrosion_rate", "thickness_loss_rate"]


@dataclass(frozen=True)
class Target:
    key: str
    label_en: str
    unit: str
    task: str


TARGETS = [
    Target("corrosion_rate", "Corrosion rate", "mm/year", "regression"),
    Target("thickness_loss_rate", "Thickness-loss rate", "mm/year", "regression"),
    Target("mechanism", "Likely corrosion mechanism", "", "classification"),
]

# NACE RP0775 corrosion-rate categories for carbon steel (mm/year).
RISK_BINS = [
    ("low", 0.0, 0.025),
    ("moderate", 0.025, 0.125),
    ("high", 0.125, 0.25),
    ("severe", 0.25, float("inf")),
]


def risk_from_rate(rate_mm_per_year: float) -> str:
    """Map a corrosion rate (mm/year) to a NACE RP0775 risk class."""
    for name, lo, hi in RISK_BINS:
        if lo <= rate_mm_per_year < hi:
            return name
    return "severe"


def rul_years(rate_mm_per_year: float, wall_thickness_mm: float, min_allowable_mm: float = 0.0) -> float:
    """Remaining useful life (years) = usable wall thickness / corrosion rate."""
    if not rate_mm_per_year or rate_mm_per_year <= 0:
        return float("inf")
    usable = max(wall_thickness_mm - min_allowable_mm, 0.0)
    return usable / rate_mm_per_year


def features_in_group(group: str):
    return [f for f in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if f.group == group]


if __name__ == "__main__":
    print(f"{len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical features; "
          f"{len(TARGETS)} targets; risk bins: {[b[0] for b in RISK_BINS]}")
