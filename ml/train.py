"""
Train + compare ML models for Diabetes, Heart Disease and Kidney Disease.

Pipeline (as in project proposal):
  Data cleaning -> median imputation -> standard scaling -> classifier
Models compared: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting
Best model per disease (by 5-fold CV ROC-AUC) is serialized to models/<disease>.pkl

Run:  python -m ml.train
Real data: put CSVs in data/real/{diabetes,heart,kidney}.csv  (see README for column names)
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.generate_data import GENERATORS  # noqa: E402
from ml.schema import DISEASES  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
REAL_DIR = os.path.join(DATA_DIR, "real")

# columns where a 0 is physiologically impossible -> treat as missing (proposal: "invalid zero entries")
INVALID_ZERO = {"diabetes": ["glucose", "bp_dia", "skin_thickness", "insulin", "bmi"]}

RENAME = {
    "diabetes": {"Pregnancies": "pregnancies", "Glucose": "glucose", "BloodPressure": "bp_dia",
                 "SkinThickness": "skin_thickness", "Insulin": "insulin", "BMI": "bmi",
                 "DiabetesPedigreeFunction": "dpf", "Age": "age", "Outcome": "target"},
    "heart": {"cp": "chest_pain", "trestbps": "bp_sys", "chol": "cholesterol", "fbs": "fbs_high",
              "restecg": "rest_ecg", "thalach": "heart_rate_max", "exang": "exercise_angina",
              "oldpeak": "st_depression"},
    "kidney": {"bp": "bp_dia", "al": "albumin", "su": "sugar", "bgr": "glucose", "bu": "urea",
               "sc": "creatinine", "hemo": "hemoglobin", "htn": "hypertension", "dm": "diabetes_hist",
               "classification": "target"},
}


def clean_inputs(disease, df):
    """Shared cleaning used for training AND live prediction."""
    df = df.copy()
    for c in INVALID_ZERO.get(disease, []):
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan
    return df


def load_dataset(disease):
    path = os.path.join(REAL_DIR, f"{disease}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns=RENAME.get(disease, {}))
        if disease == "kidney":  # UCI CKD is messy: tabs, '?', yes/no, ckd/notckd
            df = df.apply(lambda s: s.astype(str).str.strip().str.replace("\t", "") if s.dtype == object else s)
            df = df.replace({"yes": 1, "no": 0, "ckd": 1, "notckd": 0, "?": np.nan, "nan": np.nan})
        if disease == "heart" and df["chest_pain"].min() == 0:  # Kaggle 0-3 coding -> 1-4
            df["chest_pain"] += 1
        cols = DISEASES[disease]["features"] + ["target"]
        df = df[cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["target"])
        return df, "real"
    return GENERATORS[disease](), "synthetic"


def candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "SVM (RBF)": SVC(kernel="rbf", C=2.0, probability=True, random_state=7),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                                random_state=7, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.06,
                                                        random_state=7),
    }


def build_pipeline(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", clf)])


def train_one(disease):
    df, source = load_dataset(disease)
    raw_missing = int(clean_inputs(disease, df).drop(columns="target").isna().sum().sum())
    df = clean_inputs(disease, df)
    feats = DISEASES[disease]["features"]
    X, y = df[feats], df["target"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    results, fitted = {}, {}
    for name, clf in candidates().items():
        pipe = build_pipeline(clf)
        cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc")
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(yte, prob)
        idx = np.linspace(0, len(fpr) - 1, min(40, len(fpr))).astype(int)
        results[name] = dict(
            accuracy=round(accuracy_score(yte, pred), 4),
            precision=round(precision_score(yte, pred), 4),
            recall=round(recall_score(yte, pred), 4),
            f1=round(f1_score(yte, pred), 4),
            roc_auc=round(roc_auc_score(yte, prob), 4),
            cv_auc_mean=round(cvs.mean(), 4), cv_auc_std=round(cvs.std(), 4),
            roc=dict(fpr=[round(float(v), 4) for v in fpr[idx]], tpr=[round(float(v), 4) for v in tpr[idx]]),
        )
        fitted[name] = pipe
        print(f"  {disease:9s} {name:20s} acc={results[name]['accuracy']:.3f} "
              f"auc={results[name]['roc_auc']:.3f} cv_auc={cvs.mean():.3f}")

    best = max(results, key=lambda k: results[k]["cv_auc_mean"])
    pipe = fitted[best]
    pred = pipe.predict(Xte)
    cm = confusion_matrix(yte, pred).tolist()
    imp = permutation_importance(pipe, Xte, yte, n_repeats=8, random_state=1, scoring="roc_auc")
    imp_vals = np.clip(imp.importances_mean, 0, None)
    imp_vals = imp_vals / imp_vals.sum() if imp_vals.sum() > 0 else imp_vals
    importance = sorted(({"feature": f, "importance": round(float(v), 4)} for f, v in zip(feats, imp_vals)),
                        key=lambda d: -d["importance"])
    corr = {f: round(float(df[f].corr(df["target"])), 3) for f in feats}

    meta = dict(
        disease=disease, source=source, n_samples=int(len(df)), n_features=len(feats),
        class_counts={"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        missing_cells_after_cleaning=raw_missing,
        best_model=best, models=results, confusion_matrix=cm, feature_importance=importance,
        correlation_with_target=corr,
        medians={f: float(X[f].median()) for f in feats},
        train_size=int(len(Xtr)), test_size=int(len(Xte)),
    )
    joblib.dump({"pipeline": pipe, "features": feats, "meta": {"best_model": best}},
                os.path.join(MODEL_DIR, f"{disease}.pkl"))
    # also export the .sav pickle mentioned in the proposal (same artefact)
    joblib.dump(pipe, os.path.join(MODEL_DIR, f"{disease}_model.sav"))
    return meta


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    allmeta = {}
    for d in DISEASES:
        print(f"[train] {d}")
        allmeta[d] = train_one(d)
        # save the dataset used so it can be inspected / shown to examiners
        df, src = load_dataset(d)
        df.to_csv(os.path.join(DATA_DIR, f"{d}_{src}_dataset.csv"), index=False)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(allmeta, f, indent=2)
    print("[train] done ->", MODEL_DIR)


if __name__ == "__main__":
    main()
