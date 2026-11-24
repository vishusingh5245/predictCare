"""
Synthetic clinical datasets (used ONLY when real CSVs are not available in data/real/).

The generators follow the published class-conditional statistics of the public
Pima-Diabetes, UCI-Cleveland-Heart and UCI-CKD datasets (means/spreads per class)
including realistic data impurities (invalid zeros, missing values) so the
cleaning + imputation pipeline from the project proposal is actually exercised.

For the final submission, drop the real Kaggle/UCI CSVs into data/real/ (see README)
and re-run:  python -m ml.train
"""
import numpy as np
import pandas as pd


def _clip(a, lo, hi):
    return np.clip(a, lo, hi)


def make_diabetes(n=2200, seed=11):
    r = np.random.default_rng(seed)
    y = r.binomial(1, 0.36, n)
    p = lambda a, b: np.where(y == 1, a, b)
    age = _clip(r.normal(p(38, 31), 11), 21, 81).round()
    preg = _clip(r.poisson(p(4.6, 3.1)) * (age > 22), 0, 17)
    glucose = _clip(r.normal(p(142, 110), p(33, 27)), 44, 300).round()
    bp = _clip(r.normal(p(75, 70), 12), 40, 122).round()
    skin = _clip(r.normal(p(33, 27), 10), 7, 99).round()
    insulin = _clip(np.exp(r.normal(p(5.2, 4.55), 0.65)), 14, 846).round()
    bmi = _clip(r.normal(p(35.2, 30.4), 6.5), 18, 67).round(1)
    dpf = _clip(np.exp(r.normal(p(-0.65, -0.95), 0.6)), 0.08, 2.42).round(3)
    df = pd.DataFrame(dict(pregnancies=preg, glucose=glucose, bp_dia=bp, skin_thickness=skin,
                           insulin=insulin, bmi=bmi, dpf=dpf, age=age, target=y))
    # --- realistic impurities: invalid zeros exactly like the Pima dataset
    for col, frac in [("bp_dia", 0.045), ("skin_thickness", 0.29), ("insulin", 0.48), ("bmi", 0.014),
                      ("glucose", 0.006)]:
        df.loc[r.random(n) < frac, col] = 0
    return df


def make_heart(n=1600, seed=12):
    r = np.random.default_rng(seed)
    y = r.binomial(1, 0.5, n)
    p = lambda a, b: np.where(y == 1, a, b)
    age = _clip(r.normal(p(57, 52), 9), 29, 77).round()
    sex = r.binomial(1, p(0.82, 0.60))
    cp = np.array([r.choice([1, 2, 3, 4], p=(pr)) for pr in
                   np.where(y[:, None] == 1, [[.06, .10, .14, .70]], [[.10, .27, .42, .21]])])
    bps = _clip(r.normal(p(134, 129), 18), 94, 200).round()
    chol = _clip(r.normal(p(251, 243), 52), 126, 564).round()
    fbs = r.binomial(1, p(0.16, 0.14))
    ecg = np.array([r.choice([0, 1, 2], p=pr) for pr in
                    np.where(y[:, None] == 1, [[.42, .02, .56]], [[.55, .01, .44]])])
    hr = _clip(r.normal(p(139, 158), 22), 71, 202).round()
    exang = r.binomial(1, p(0.55, 0.14))
    old = _clip(r.gamma(p(1.7, 1.0), p(0.9, 0.55)) - 0.1, 0, 6.2).round(1)
    df = pd.DataFrame(dict(age=age, sex=sex, chest_pain=cp, bp_sys=bps, cholesterol=chol, fbs_high=fbs,
                           rest_ecg=ecg, heart_rate_max=hr, exercise_angina=exang, st_depression=old, target=y))
    for col in ["cholesterol", "heart_rate_max", "bp_sys"]:
        df.loc[r.random(n) < 0.02, col] = np.nan
    return df


def make_kidney(n=1500, seed=13):
    r = np.random.default_rng(seed)
    y = r.binomial(1, 0.40, n)
    p = lambda a, b: np.where(y == 1, a, b)
    age = _clip(r.normal(p(55, 46), 16), 4, 90).round()
    bp = _clip(r.normal(p(82, 74), 12), 50, 140).round()
    sg_vals = np.array([1.005, 1.010, 1.015, 1.020, 1.025])
    sg = np.array([r.choice(sg_vals, p=pr) for pr in
                   np.where(y[:, None] == 1, [[.10, .30, .30, .25, .05]], [[.00, .03, .10, .52, .35]])])
    al = np.array([r.choice(range(6), p=pr) for pr in
                   np.where(y[:, None] == 1, [[.12, .22, .26, .24, .13, .03]], [[.90, .06, .03, .01, .0, .0]])])
    su = np.array([r.choice(range(6), p=pr) for pr in
                   np.where(y[:, None] == 1, [[.62, .10, .10, .08, .07, .03]], [[.96, .02, .01, .01, .0, .0]])])
    glu = _clip(r.normal(p(150, 108), p(60, 22)), 50, 450).round()
    bu = _clip(np.exp(r.normal(p(4.0, 3.35), p(0.6, 0.28))), 10, 240).round(1)
    sc = _clip(np.exp(r.normal(p(1.0, -0.05), p(0.85, 0.22))), 0.4, 18).round(1)
    hemo = _clip(r.normal(p(10.6, 14.4), p(2.4, 1.4)), 3.5, 17.8).round(1)
    htn = r.binomial(1, p(0.62, 0.07))
    dm = r.binomial(1, p(0.48, 0.05))
    df = pd.DataFrame(dict(age=age, bp_dia=bp, sg=sg, albumin=al, sugar=su, glucose=glu, urea=bu,
                           creatinine=sc, hemoglobin=hemo, hypertension=htn, diabetes_hist=dm, target=y))
    for col in df.columns[:-1]:
        df.loc[r.random(n) < 0.04, col] = np.nan
    return df


GENERATORS = {"diabetes": make_diabetes, "heart": make_heart, "kidney": make_kidney}
