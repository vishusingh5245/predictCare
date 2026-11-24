# PredictCare — Predictive Healthcare & Disease Risk Prediction System
B.Tech CSE 7th Sem Major Project · Group 15 · SREC Mirzapur

AI/ML web portal: patient registration/login → upload a medical report (PDF/image) → OCR extracts lab values →
ML models estimate **Diabetes, Heart Disease and Kidney Disease** risk → explainable result, PDF report, history in database.

## Run (Windows / Linux / Mac)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python app.py                  # opens on http://127.0.0.1:5000  (models auto-train on first run)
```
Trained models are already included in `models/`. To retrain: `python -m ml.train`.

### Scanning IMAGES (JPG/PNG) needs Tesseract OCR
Text-based PDFs work without it. For photos / scanned PDFs install Tesseract:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki  (then add `C:\Program Files\Tesseract-OCR` to PATH)
- Ubuntu: `sudo apt install tesseract-ocr`   · Mac: `brew install tesseract`

### Demo tip (no internet on demo day?)
Charts use Chart.js. Download https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js and save it as
`static/js/chart.umd.min.js` — the app then uses the local copy automatically.

## Demo flow
1. Register → Login  2. New Assessment → upload `sample_reports/sample_report_high_risk.pdf`
3. Review extracted values → Analyse  4. Open detailed report → Download PDF  5. History, Model Insights pages.
(`python make_sample_reports.py` regenerates the sample reports.)

## Project structure
```
app.py            Flask routes (auth, upload, prediction, history, profile)
database.py       SQLite schema: users, reports, predictions, login_logs
pdf_report.py     Downloadable PDF health summary
ml/schema.py      All clinical fields + which fields each disease model uses
ml/generate_data.py  Demo datasets (used only if real CSVs are absent)
ml/train.py       Cleaning → median imputation → scaling → LR/SVM/RF/GB comparison → saves best model
ml/ocr_extract.py PDF/Image text extraction + regex parameter parser + unit conversion
ml/predict.py     Inference, risk level, explanation, reference-range flags, advice
models/           *.pkl, *_model.sav, metrics.json
templates/ static/  Website UI
```

## IMPORTANT — datasets
Internet was not available while building, so the bundled models are trained on **synthetic data statistically modelled on
Pima-Diabetes, UCI-Heart (Cleveland) and UCI-CKD** (includes invalid zeros & missing values to exercise the cleaning step).
Metrics shown in *Model Insights* therefore are for demo data, not for real patients.
**Before final submission, train on the real datasets** (as in your proposal): put the Kaggle/UCI CSVs here and run `python -m ml.train`
- `data/real/diabetes.csv` — Pima file (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome)
- `data/real/heart.csv` — Cleveland file (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, target)
- `data/real/kidney.csv` — UCI CKD file (age, bp, sg, al, su, bgr, bu, sc, hemo, htn, dm, classification)
The Model Insights page will then show "real dataset" automatically.

## Database tables
users · reports (file, OCR text, extracted values) · predictions (inputs, probability, risk level, explanation) · login_logs

## Security
Hashed passwords (Werkzeug), CSRF tokens, per-user data isolation, upload type/size validation, input range validation.
Set a real `SECRET_KEY` environment variable for deployment.

⚠ Academic prototype. Not a medical diagnosis.
