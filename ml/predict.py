"""Inference, explanation (occlusion-based factors), flags and advice."""
import json
import os

import joblib
import numpy as np
import pandas as pd

from ml.schema import CATALOG, DISEASES
from ml.train import clean_inputs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
_cache = {}


def models_ready():
    return all(os.path.exists(os.path.join(MODEL_DIR, f"{d}.pkl")) for d in DISEASES) and \
        os.path.exists(os.path.join(MODEL_DIR, "metrics.json"))


def load_model(disease):
    if disease not in _cache:
        _cache[disease] = joblib.load(os.path.join(MODEL_DIR, f"{disease}.pkl"))
    return _cache[disease]


def load_metrics():
    p = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def risk_level(p):
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Moderate"
    return "High"


ADVICE = {
    "diabetes": {
        "Low": ["Maintain a balanced diet rich in fibre, vegetables and whole grains.",
                "Keep up 150 minutes of moderate exercise per week.",
                "Get fasting blood sugar re-checked once every year."],
        "Moderate": ["Reduce refined sugar and processed carbohydrates; prefer low-GI foods.",
                     "Aim for weight loss of 5–7% if BMI is above 25.",
                     "Repeat fasting glucose / HbA1c test within 3 months and consult a physician."],
        "High": ["Consult a doctor / diabetologist soon for HbA1c and OGTT confirmation.",
                 "Strictly limit sugary drinks, sweets and refined flour.",
                 "Start daily walking (30–45 min) and monitor blood glucose regularly."],
    },
    "heart": {
        "Low": ["Continue heart-healthy habits: less salt, no tobacco, regular activity.",
                "Check blood pressure and lipid profile once a year.",
                "Maintain a healthy weight and manage stress and sleep."],
        "Moderate": ["Get a lipid profile and an ECG done and discuss results with a physician.",
                     "Cut down saturated fat, fried food and salt; quit smoking if applicable.",
                     "Monitor blood pressure weekly."],
        "High": ["Please consult a cardiologist soon for ECG / echo / stress test confirmation.",
                 "Do not ignore chest pain, breathlessness or dizziness — seek emergency care if severe.",
                 "Strictly control BP, cholesterol and blood sugar with medical guidance."],
    },
    "kidney": {
        "Low": ["Stay hydrated (2–3 L water/day unless advised otherwise).",
                "Avoid long-term, unsupervised use of painkillers (NSAIDs).",
                "Check creatinine and urine albumin yearly if you have diabetes or hypertension."],
        "Moderate": ["Repeat serum creatinine, eGFR and urine albumin-creatinine ratio tests.",
                     "Control BP and blood sugar; reduce salt intake.",
                     "Consult a physician to review medicines that may affect the kidneys."],
        "High": ["Consult a nephrologist soon for eGFR, ultrasound and urine tests.",
                 "Avoid NSAIDs and herbal supplements unless approved by your doctor.",
                 "Limit salt, and follow medical advice on protein, potassium and fluid intake."],
    },
}


def parse_value(key, raw):
    """Convert form/OCR input to float (or NaN when empty)."""
    if raw is None:
        return np.nan
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return np.nan
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{CATALOG[key]['label']}: '{raw}' is not a valid number")


def validate(disease, raw_inputs):
    """Return (clean_dict, errors). clean_dict has NaN for missing."""
    out, errors = {}, []
    for k in DISEASES[disease]["features"]:
        meta = CATALOG[k]
        try:
            v = parse_value(k, raw_inputs.get(k))
        except ValueError as e:
            errors.append(str(e))
            v = np.nan
        if not np.isnan(v):
            if meta.get("type") == "select":
                if v not in [o[0] for o in meta["options"]]:
                    errors.append(f"{meta['label']}: invalid choice")
            elif v < meta["min"] or v > meta["max"]:
                errors.append(f"{meta['label']} must be between {meta['min']} and {meta['max']} {meta.get('unit', '')}")
        out[k] = v
    return out, errors


def completeness(disease, values):
    feats = DISEASES[disease]["features"]
    given = [k for k in feats if not np.isnan(values.get(k, np.nan))]
    return len(given), len(feats)


def _prob(pipe, feats, row):
    return float(pipe.predict_proba(pd.DataFrame([row], columns=feats))[0, 1])


def predict(disease, values):
    """values: dict key->float/NaN. Returns result dict."""
    bundle = load_model(disease)
    pipe, feats = bundle["pipeline"], bundle["features"]
    metrics = load_metrics().get(disease, {})
    medians = metrics.get("medians", {})

    df = clean_inputs(disease, pd.DataFrame([{k: values.get(k, np.nan) for k in feats}]))
    row = df.iloc[0].to_dict()
    p = _prob(pipe, feats, row)

    # occlusion explanation: how much does risk change if the feature were "typical" (median)?
    def logit(q):
        q = min(max(q, 1e-9), 1 - 1e-9)
        return float(np.log(q / (1 - q)))

    raw = []
    for k in feats:
        if np.isnan(row[k]):
            continue
        alt = dict(row)
        alt[k] = medians.get(k, np.nan)
        raw.append((k, logit(p) - logit(_prob(pipe, feats, alt))))
    total = sum(abs(d) for _, d in raw) or 1.0
    # share of influence in % (log-odds scale: stays meaningful even when risk is ~0% or ~100%)
    factors = [dict(key=k, label=CATALOG[k]["label"], value=row[k], impact=round(100 * d / total, 1))
               for k, d in raw if abs(d) > 1e-9]
    factors.sort(key=lambda f: -abs(f["impact"]))

    flags = []
    for k in feats:
        meta, v = CATALOG[k], row[k]
        if np.isnan(v) or not meta.get("ref"):
            continue
        lo, hi = meta["ref"]
        status = "low" if v < lo else "high" if v > hi else "normal"
        flags.append(dict(key=k, label=meta["label"], value=v, unit=meta.get("unit", ""),
                          ref=f"{lo:g} – {hi:g}", status=status))

    lvl = risk_level(p)
    given, total = completeness(disease, values)
    return dict(
        disease=disease, probability=round(p, 4), percent=min(99.9, max(0.1, round(p * 100, 1))), level=lvl,
        label="Positive risk" if p >= 0.5 else "Negative risk",
        factors=factors[:6], flags=flags, advice=ADVICE[disease][lvl],
        model=bundle["meta"]["best_model"], given=given, total=total,
        completeness=round(100 * given / total),
    )
