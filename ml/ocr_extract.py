"""
Medical report scanner.
  PDF (text layer)  -> pdfplumber
  PDF (scanned)     -> page raster -> Tesseract OCR
  Image (jpg/png)   -> Pillow preprocessing -> Tesseract OCR
Then regex-based parameter extraction + unit normalisation.
"""
import os
import re

try:
    import pytesseract
    from PIL import Image, ImageOps, ImageFilter
    TESS = True
except Exception:  # pragma: no cover
    TESS = False

import pdfplumber


def tesseract_available():
    if not TESS:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_image(img):
    img = ImageOps.grayscale(img)
    if img.width < 1600:
        s = 1600 / img.width
        img = img.resize((int(img.width * s), int(img.height * s)))
    img = ImageOps.autocontrast(img).filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(img, config="--psm 6")


def extract_text(path):
    """Return (text, method). Raises RuntimeError with a friendly message if impossible."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:10]:
                text += (page.extract_text() or "") + "\n"
            if len(text.strip()) >= 40:
                return text, "PDF text layer"
            if not tesseract_available():
                raise RuntimeError("Scanned PDF detected but Tesseract OCR is not installed on the server.")
            for page in pdf.pages[:5]:
                text += _ocr_image(page.to_image(resolution=220).original) + "\n"
        return text, "OCR (scanned PDF)"
    if not tesseract_available():
        raise RuntimeError("Tesseract OCR is not installed. Install it (see README) or upload a text-based PDF.")
    return _ocr_image(Image.open(path)), "OCR (image)"


NUM = r"([0-9]{1,4}(?:[.,][0-9]{1,3})?)"
SEP = r"[\s:=\-–—|\.]*"

# canonical key -> list of label regexes
PATTERNS = {
    "age": [r"\bage\b" + SEP + NUM + r"\s*(?:y|yrs|years|yr)?"],
    "glucose": [r"(?:fasting|random|post\s*prandial|pp)?\s*(?:blood\s*)?(?:glucose|sugar|fbs|rbs|ppbs|fbg|rbg)\b(?!\s*\(?urine)" + r"[^0-9\n]{0,30}" + NUM],
    "insulin": [r"\binsulin\b[^0-9\n]{0,25}" + NUM],
    "bmi": [r"\bbmi\b|body\s*mass\s*index"],  # handled below with its own pattern
    "cholesterol": [r"(?:total\s*)?cholesterol(?!\s*[-–]?\s*(?:hdl|ldl|vldl))[^0-9\n]{0,25}" + NUM],
    "hemoglobin": [r"(?:hemoglobin|haemoglobin|\bhb\b|\bhgb\b)[^0-9\n]{0,25}" + NUM],
    "creatinine": [r"(?:serum\s*)?creatinine[^0-9\n]{0,25}" + NUM],
    "urea": [r"(?:blood\s*urea(?:\s*nitrogen)?|\bbun\b|\burea\b)[^0-9\n]{0,25}" + NUM],
    "heart_rate_max": [r"(?:max(?:imum)?\.?\s*heart\s*rate|max\s*hr|peak\s*hr|pulse(?:\s*rate)?|heart\s*rate)[^0-9\n]{0,25}" + NUM],
    "sg": [r"(?:specific\s*gravity|\bsg\b)[^0-9\n]{0,20}(1[.,][0-9]{2,3})"],
    "albumin": [r"(?:urine\s*)?(?:albumin|protein)\s*(?:\(urine\))?[^0-9A-Za-z\n]{0,20}(nil|absent|negative|trace|\+{1,4}|[0-5])\b"],
    "sugar": [r"(?:urine\s*)?(?:sugar|glucose)\s*\(urine\)[^0-9A-Za-z\n]{0,20}(nil|absent|negative|trace|\+{1,4}|[0-5])\b"],
}


def _num(s):
    return float(s.replace(",", "."))


def _grade(tok):
    tok = tok.strip().lower()
    if tok in ("nil", "absent", "negative", "0"):
        return 0
    if tok == "trace":
        return 1
    if tok.startswith("+"):
        return min(len(tok), 5)
    try:
        return int(tok)
    except ValueError:
        return None


def parse_report(text):
    """Return dict key -> value, plus 'notes' about conversions."""
    t = text.replace(" ", " ")
    low = t.lower()
    found, notes = {}, []

    m = re.search(r"\bage\b" + SEP + r"(\d{1,3})\s*(?:y|yrs|years|yr)?", low)
    if m:
        found["age"] = int(m.group(1))
    m = re.search(r"\b(?:sex|gender)\b" + SEP + r"(male|female|m|f)\b", low)
    if m:
        found["sex"] = 1 if m.group(1).startswith("m") else 0
    else:
        m = re.search(r"\b(\d{1,3})\s*(?:y|yrs|years)?\s*/\s*(male|female|m|f)\b", low)
        if m:
            found.setdefault("age", int(m.group(1)))
            found["sex"] = 1 if m.group(2).startswith("m") else 0

    m = re.search(r"\b(?:bp|blood\s*pressure)\b[^0-9\n]{0,25}(\d{2,3})\s*/\s*(\d{2,3})", low)
    if m:
        found["bp_sys"], found["bp_dia"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(?:bmi|body\s*mass\s*index)[^0-9\n]{0,25}" + NUM, low)
    if m:
        found["bmi"] = _num(m.group(1))
    m = re.search(r"pregnanc(?:y|ies)[^0-9\n]{0,20}(\d{1,2})", low)
    if m:
        found["pregnancies"] = int(m.group(1))
    m = re.search(r"skin\s*(?:fold)?\s*thickness[^0-9\n]{0,20}" + NUM, low)
    if m:
        found["skin_thickness"] = _num(m.group(1))
    m = re.search(r"(?:st\s*depression|oldpeak)[^0-9\n]{0,20}" + NUM, low)
    if m:
        found["st_depression"] = _num(m.group(1))

    for key in ("glucose", "insulin", "cholesterol", "hemoglobin", "creatinine", "urea", "heart_rate_max"):
        for pat in PATTERNS[key]:
            # glucose: prefer 'fasting' line if several
            ms = list(re.finditer(pat, low))
            if ms:
                if key == "glucose":
                    fasting = [x for x in ms if "fasting" in x.group(0) or "fbs" in x.group(0)]
                    ms = fasting or ms
                try:
                    found[key] = _num(ms[0].group(ms[0].lastindex))
                except Exception:
                    pass
                break

    m = re.search(PATTERNS["sg"][0], low)
    if m:
        found["sg"] = round(_num(m.group(1)), 3)
    m = re.search(PATTERNS["albumin"][0], low)
    if m and _grade(m.group(1)) is not None:
        found["albumin"] = _grade(m.group(1))
    m = re.search(PATTERNS["sugar"][0], low)
    if m and _grade(m.group(1)) is not None:
        found["sugar"] = _grade(m.group(1))

    # ---- unit normalisation
    if "glucose" in found and found["glucose"] < 30:
        found["glucose"] = round(found["glucose"] * 18.0, 0)
        notes.append("Glucose looked like mmol/L → converted to mg/dL (×18).")
    if "cholesterol" in found and found["cholesterol"] < 15:
        found["cholesterol"] = round(found["cholesterol"] * 38.67, 0)
        notes.append("Cholesterol looked like mmol/L → converted to mg/dL (×38.67).")
    if "hemoglobin" in found and found["hemoglobin"] > 30:  # g/L
        found["hemoglobin"] = round(found["hemoglobin"] / 10, 1)
    # derived fields
    if "glucose" in found:
        found["fbs_high"] = 1 if found["glucose"] > 120 else 0
    if found.get("sex") == 1 and "pregnancies" not in found:
        found["pregnancies"] = 0
    return found, notes


def scan_report(path):
    text, method = extract_text(path)
    values, notes = parse_report(text)
    return dict(text=text, method=method, values=values, notes=notes)
