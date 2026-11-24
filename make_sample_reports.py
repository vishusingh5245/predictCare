"""Generate demo lab reports (PDF + PNG) so the scan feature can be demonstrated.
Run: python make_sample_reports.py
"""
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_reports")
os.makedirs(OUT, exist_ok=True)

REPORTS = {
    "sample_report_high_risk": dict(
        name="Ramesh Kumar", age=58, sex="Male", bp="152/96", bmi="31.4",
        tests=[("Fasting Blood Glucose", "168", "mg/dL", "70 - 100"), ("Insulin", "210", "uU/mL", "16 - 166"),
               ("Total Cholesterol", "262", "mg/dL", "< 200"), ("Hemoglobin", "10.2", "g/dL", "13 - 17"),
               ("Serum Creatinine", "2.8", "mg/dL", "0.6 - 1.3"), ("Blood Urea", "78", "mg/dL", "15 - 45"),
               ("Max Heart Rate", "128", "bpm", "-"),
               ("Specific Gravity (Urine)", "1.010", "", "1.005 - 1.025"),
               ("Albumin (Urine)", "3", "0-5 scale", "0"), ("Sugar (Urine)", "2", "0-5 scale", "0")]),
    "sample_report_low_risk": dict(
        name="Anita Singh", age=29, sex="Female", bp="112/72", bmi="22.1",
        tests=[("Fasting Blood Glucose", "88", "mg/dL", "70 - 100"), ("Insulin", "74", "uU/mL", "16 - 166"),
               ("Total Cholesterol", "168", "mg/dL", "< 200"), ("Hemoglobin", "13.4", "g/dL", "12 - 15.5"),
               ("Serum Creatinine", "0.8", "mg/dL", "0.6 - 1.3"), ("Blood Urea", "26", "mg/dL", "15 - 45"),
               ("Max Heart Rate", "176", "bpm", "-"),
               ("Specific Gravity (Urine)", "1.020", "", "1.005 - 1.025"),
               ("Albumin (Urine)", "0", "0-5 scale", "0"), ("Sugar (Urine)", "0", "0-5 scale", "0")]),
}


def make(fname, d):
    c = canvas.Canvas(os.path.join(OUT, fname + ".pdf"), pagesize=A4)
    W, H = A4
    c.setFillColor(colors.HexColor("#0f766e"))
    c.rect(0, H - 90, W, 90, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, H - 45, "City Diagnostic & Pathology Laboratory")
    c.setFont("Helvetica", 10)
    c.drawString(40, H - 65, "Mirzapur, Uttar Pradesh  |  SAMPLE REPORT FOR PROJECT DEMO - NOT A REAL PATIENT")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 11)
    y = H - 125
    c.drawString(40, y, f"Patient Name: {d['name']}")
    c.drawString(320, y, f"Age: {d['age']} Years")
    y -= 20
    c.drawString(40, y, f"Sex: {d['sex']}")
    c.drawString(320, y, f"BP: {d['bp']} mmHg")
    y -= 20
    c.drawString(40, y, f"BMI: {d['bmi']} kg/m2")
    c.drawString(320, y, "Sample: Blood / Urine")
    y -= 35
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.HexColor("#e2e8f0"))
    c.rect(35, y - 5, W - 70, 22, fill=1, stroke=0)
    c.setFillColor(colors.black)
    for x, t in [(40, "Test"), (270, "Result"), (340, "Unit"), (420, "Reference")]:
        c.drawString(x, y, t)
    c.setFont("Helvetica", 11)
    for t, v, u, r in d["tests"]:
        y -= 26
        c.drawString(40, y, t)
        c.drawString(270, y, v)
        c.drawString(340, y, u)
        c.drawString(420, y, r)
        c.setStrokeColor(colors.HexColor("#e2e8f0"))
        c.line(35, y - 8, W - 35, y - 8)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, 60, "Electronically generated sample report for academic demonstration purposes only.")
    c.save()


for k, v in REPORTS.items():
    make(k, v)
print("sample reports written to", OUT)
