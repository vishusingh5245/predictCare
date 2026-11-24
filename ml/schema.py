"""
Central schema: every clinical field, its label/unit/valid range/reference range,
and which fields each disease model uses.
All datasets (synthetic or real) are renamed to these canonical keys.
"""

# key -> metadata
# type: number | select ; ref = (low, high) healthy reference range (for flagging)
CATALOG = {
    # ---- demographics
    "age": dict(label="Age", unit="years", min=1, max=120, step=1, group="Demographics", ref=None, default_hint=45),
    "sex": dict(label="Sex", type="select", group="Demographics",
                options=[(1, "Male"), (0, "Female")]),
    "pregnancies": dict(label="Pregnancies", unit="count", min=0, max=20, step=1, group="Demographics", ref=None,
                        help="Number of times pregnant (0 for male)"),
    # ---- vitals
    "bp_sys": dict(label="Systolic BP", unit="mmHg", min=70, max=250, step=1, group="Vitals", ref=(90, 120)),
    "bp_dia": dict(label="Diastolic BP", unit="mmHg", min=40, max=160, step=1, group="Vitals", ref=(60, 80)),
    "bmi": dict(label="BMI", unit="kg/m²", min=10, max=70, step=0.1, group="Vitals", ref=(18.5, 24.9)),
    "skin_thickness": dict(label="Skin-fold Thickness", unit="mm", min=0, max=100, step=1, group="Vitals",
                           ref=(10, 30), help="Triceps skin-fold (optional)"),
    "heart_rate_max": dict(label="Max Heart Rate", unit="bpm", min=60, max=220, step=1, group="Cardiac",
                           ref=None, help="Max heart rate achieved in stress test / pulse"),
    # ---- blood tests
    "glucose": dict(label="Blood Glucose", unit="mg/dL", min=40, max=500, step=1, group="Blood Tests", ref=(70, 125)),
    "insulin": dict(label="Insulin (2-hr)", unit="µU/mL", min=0, max=900, step=1, group="Blood Tests", ref=(16, 166)),
    "cholesterol": dict(label="Total Cholesterol", unit="mg/dL", min=100, max=600, step=1, group="Blood Tests",
                        ref=(125, 200)),
    "urea": dict(label="Blood Urea", unit="mg/dL", min=5, max=250, step=1, group="Blood Tests", ref=(15, 45)),
    "creatinine": dict(label="Serum Creatinine", unit="mg/dL", min=0.2, max=20, step=0.1, group="Blood Tests",
                       ref=(0.6, 1.3)),
    "hemoglobin": dict(label="Hemoglobin", unit="g/dL", min=3, max=20, step=0.1, group="Blood Tests", ref=(12, 17.5)),
    "dpf": dict(label="Diabetes Pedigree Function", unit="", min=0.05, max=2.6, step=0.01, group="History",
                ref=None, help="Family-history score (0.08 – 2.4). Leave blank if unknown"),
    # ---- urine tests
    "sg": dict(label="Urine Specific Gravity", unit="", min=1.000, max=1.040, step=0.005, group="Urine Tests",
               ref=(1.005, 1.025)),
    "albumin": dict(label="Urine Albumin", unit="0–5 scale", min=0, max=5, step=1, group="Urine Tests", ref=(0, 0)),
    "sugar": dict(label="Urine Sugar", unit="0–5 scale", min=0, max=5, step=1, group="Urine Tests", ref=(0, 0)),
    # ---- cardiac / history
    "chest_pain": dict(label="Chest Pain Type", type="select", group="Cardiac",
                       options=[(1, "Typical angina"), (2, "Atypical angina"), (3, "Non-anginal pain"),
                                (4, "Asymptomatic")]),
    "fbs_high": dict(label="Fasting Sugar > 120 mg/dL", type="select", group="Cardiac",
                     options=[(0, "No"), (1, "Yes")]),
    "rest_ecg": dict(label="Resting ECG", type="select", group="Cardiac",
                     options=[(0, "Normal"), (1, "ST-T abnormality"), (2, "LV hypertrophy")]),
    "exercise_angina": dict(label="Exercise-induced Angina", type="select", group="Cardiac",
                            options=[(0, "No"), (1, "Yes")]),
    "st_depression": dict(label="ST Depression (Oldpeak)", unit="mm", min=0, max=7, step=0.1, group="Cardiac",
                          ref=(0, 1)),
    "hypertension": dict(label="Hypertension History", type="select", group="History",
                         options=[(0, "No"), (1, "Yes")]),
    "diabetes_hist": dict(label="Diabetes History", type="select", group="History",
                          options=[(0, "No"), (1, "Yes")]),
}

DISEASES = {
    "diabetes": dict(
        name="Diabetes", icon="🩸", color="#f43f5e",
        tagline="Metabolic risk from glucose, BMI, insulin & family history",
        dataset="Pima Indians Diabetes (schema)",
        features=["pregnancies", "glucose", "bp_dia", "skin_thickness", "insulin", "bmi", "dpf", "age"],
        required=["glucose", "age"],
    ),
    "heart": dict(
        name="Heart Disease", icon="❤️", color="#f97316",
        tagline="Cardiac risk from BP, cholesterol, ECG & exercise response",
        dataset="UCI Cleveland Heart (schema)",
        features=["age", "sex", "chest_pain", "bp_sys", "cholesterol", "fbs_high", "rest_ecg",
                  "heart_rate_max", "exercise_angina", "st_depression"],
        required=["age", "sex"],
    ),
    "kidney": dict(
        name="Kidney Disease", icon="🫘", color="#8b5cf6",
        tagline="Renal risk from creatinine, urea, albumin & hemoglobin",
        dataset="UCI Chronic Kidney Disease (schema)",
        features=["age", "bp_dia", "sg", "albumin", "sugar", "glucose", "urea", "creatinine", "hemoglobin",
                  "hypertension", "diabetes_hist"],
        required=["creatinine"],
    ),
}

# Order of groups on the combined (report review) form
GROUP_ORDER = ["Demographics", "Vitals", "Blood Tests", "Urine Tests", "Cardiac", "History"]


def fields_for(disease):
    return DISEASES[disease]["features"]


def all_fields():
    seen = []
    for d in DISEASES.values():
        for f in d["features"]:
            if f not in seen:
                seen.append(f)
    return seen


def grouped(keys):
    """Return [(group, [(key, meta), ...]), ...] preserving GROUP_ORDER."""
    out = {}
    for k in keys:
        out.setdefault(CATALOG[k]["group"], []).append((k, CATALOG[k]))
    return [(g, out[g]) for g in GROUP_ORDER if g in out]
