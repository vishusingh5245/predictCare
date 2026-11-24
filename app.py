# Predictive Healthcare & Disease Risk Prediction System
# Flask web application  |  Group 15, SRECM Mirzapur
# Run:  python app.py      ->  http://127.0.0.1:5000
# """
import io
import json
import os
import re
import secrets
import uuid
from functools import wraps

import numpy as np
from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request, send_file,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from database import close_db, get_db, init_db
from ml import ocr_extract
from ml.predict import completeness, load_metrics, models_ready, predict, validate
from ml.schema import CATALOG, DISEASES, all_fields, grouped

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

app = Flask(__name__)
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://predict-care-topaz.vercel.app").rstrip("/")
app.jinja_env.globals["PUBLIC_BASE_URL"] = PUBLIC_BASE_URL
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or "dev-secret-change-me-" + "healthcare-g15",
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=bool(os.environ.get("VERCEL")),
    SESSION_COOKIE_SAMESITE="Lax",
)
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.teardown_appcontext(close_db)
init_db()

if not models_ready():  # first run convenience
    from ml.train import main as _train
    print("[startup] models not found – training now ...")
    _train()


# ------------------------------------------------------------------ helpers
def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token
app.jinja_env.globals["DISEASES"] = DISEASES
app.jinja_env.globals["CATALOG"] = CATALOG


@app.before_request
def _csrf_protect():
    if request.method == "POST":
        if request.form.get("_csrf") != session.get("_csrf"):
            abort(400, "Invalid or missing CSRF token. Please reload the page and try again.")


@app.before_request
def _load_user():
    g.user = None
    uid = session.get("uid")
    if uid:
        g.user = get_db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if g.user is None:
            session.clear()


def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if g.user is None:
            flash("Please log in to continue.", "info")
            return redirect(url_for("login", next=request.path))
        return f(*a, **kw)
    return wrapper


@app.template_filter("pretty")
def pretty(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return v


@app.template_filter("dt")
def fmt_dt(s):
    from datetime import datetime
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return s


def option_label(key, value):
    meta = CATALOG[key]
    if meta.get("type") == "select":
        for v, l in meta["options"]:
            if v == value:
                return l
    return None


app.jinja_env.globals["option_label"] = option_label


@app.context_processor
def _chartjs():
    """Use a local copy of Chart.js (works offline on demo day) if present, else the CDN."""
    local = os.path.join(BASE, "static", "js", "chart.umd.min.js")
    src = url_for("static", filename="js/chart.umd.min.js") if os.path.exists(local) \
        else "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
    return {"chartjs_src": src}


def clean_json(d):
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()}


def save_prediction(user_id, disease, values, result, source, report_id=None):
    db = get_db()
    cur = db.execute(
        "INSERT INTO predictions (user_id, report_id, disease, inputs_json, probability, risk_level, "
        "result_json, model_name, source) VALUES (?,?,?,?,?,?,?,?,?)",
        (user_id, report_id, disease, json.dumps(clean_json(values)), result["probability"], result["level"],
         json.dumps(result), result["model"], source))
    db.commit()
    return cur.lastrowid


def own_prediction(pid):
    row = get_db().execute("SELECT * FROM predictions WHERE id=? AND user_id=?", (pid, g.user["id"])).fetchone()
    if row is None:
        abort(404)
    return row


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ------------------------------------------------------------------ public pages
@app.route("/")
def index():
    metrics = load_metrics()
    stats = {d: metrics.get(d, {}).get("models", {}).get(metrics.get(d, {}).get("best_model", ""), {})
             for d in DISEASES}
    return render_template("index.html", stats=stats, metrics=metrics)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        f = request.form
        name, email = f.get("full_name", "").strip(), f.get("email", "").strip().lower()
        pw, pw2 = f.get("password", ""), f.get("confirm", "")
        age, gender, phone = f.get("age", "").strip(), f.get("gender", ""), f.get("phone", "").strip()
        errs = []
        if len(name) < 3:
            errs.append("Please enter your full name (min 3 characters).")
        if not EMAIL_RE.match(email):
            errs.append("Please enter a valid email address.")
        if len(pw) < 8 or not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
            errs.append("Password must be at least 8 characters and contain letters and numbers.")
        if pw != pw2:
            errs.append("Passwords do not match.")
        if age and not (age.isdigit() and 1 <= int(age) <= 120):
            errs.append("Age must be between 1 and 120.")
        if phone and not re.fullmatch(r"[+0-9\- ]{7,15}", phone):
            errs.append("Phone number looks invalid.")
        db = get_db()
        if not errs and db.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            errs.append("An account with this email already exists. Try logging in.")
        if errs:
            for e in errs:
                flash(e, "error")
            return render_template("register.html", form=f), 400
        cur = db.execute("INSERT INTO users (full_name,email,password_hash,age,gender,phone) VALUES (?,?,?,?,?,?)",
                         (name, email, generate_password_hash(pw), int(age) if age else None, gender or None,
                          phone or None))
        db.commit()
        session.clear()
        session["uid"] = cur.lastrowid
        flash(f"Welcome, {name.split()[0]}! Your account has been created.", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html", form={})


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email, pw = request.form.get("email", "").strip().lower(), request.form.get("password", "")
        db = get_db()
        u = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        ok = bool(u and check_password_hash(u["password_hash"], pw))
        db.execute("INSERT INTO login_logs (user_id,email,success,ip) VALUES (?,?,?,?)",
                   (u["id"] if u else None, email, int(ok), request.remote_addr))
        db.commit()
        if ok:
            nxt = request.args.get("next", "")
            session.clear()
            session["uid"] = u["id"]
            flash(f"Welcome back, {u['full_name'].split()[0]}!", "success")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
        flash("Invalid email or password.", "error")
        return render_template("login.html", email=email), 401
    return render_template("login.html", email="")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ------------------------------------------------------------------ dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    uid = g.user["id"]
    rows = db.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
    n_reports = db.execute("SELECT COUNT(*) c FROM reports WHERE user_id=?", (uid,)).fetchone()["c"]
    latest = {}
    for r in rows:
        latest.setdefault(r["disease"], r)
    high = sum(1 for r in latest.values() if r["risk_level"] == "High")
    timeline = [dict(date=r["created_at"][:10], disease=r["disease"], pct=round(r["probability"] * 100, 1))
                for r in reversed(rows[:30])]
    return render_template("dashboard.html", recent=rows[:6], total=len(rows), n_reports=n_reports,
                           latest=latest, high=high, timeline=timeline)


# ------------------------------------------------------------------ assessment
@app.route("/assess")
@login_required
def assess():
    return render_template("assess.html", ocr_ok=ocr_extract.tesseract_available())


@app.route("/assess/manual/<disease>", methods=["GET", "POST"])
@login_required
def assess_manual(disease):
    if disease not in DISEASES:
        abort(404)
    feats = DISEASES[disease]["features"]
    prefill = {}
    if g.user["age"] and "age" in feats:
        prefill["age"] = g.user["age"]
    if g.user["gender"] in ("Male", "Female") and "sex" in feats:
        prefill["sex"] = 1 if g.user["gender"] == "Male" else 0
    if request.method == "POST":
        values, errs = validate(disease, request.form)
        given, total = completeness(disease, values)
        miss_req = [CATALOG[k]["label"] for k in DISEASES[disease]["required"] if np.isnan(values[k])]
        if miss_req:
            errs.append("Required: " + ", ".join(miss_req))
        if errs:
            for e in errs:
                flash(e, "error")
            return render_template("manual.html", disease=disease, groups=grouped(feats),
                                   form=request.form, prefill={}), 400
        res = predict(disease, values)
        pid = save_prediction(g.user["id"], disease, values, res, "manual")
        return redirect(url_for("result", pid=pid))
    return render_template("manual.html", disease=disease, groups=grouped(feats), form={}, prefill=prefill)


@app.route("/assess/upload", methods=["POST"])
@login_required
def upload_report():
    f = request.files.get("report")
    if not f or not f.filename:
        flash("Please choose a medical report file to upload.", "error")
        return redirect(url_for("assess"))
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        flash("Unsupported file type. Upload a PDF or an image (PNG / JPG).", "error")
        return redirect(url_for("assess"))
    stored = f"{g.user['id']}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, stored)
    f.save(path)
    try:
        scan = ocr_extract.scan_report(path)
    except RuntimeError as e:
        os.remove(path)
        flash(str(e), "error")
        return redirect(url_for("assess"))
    except Exception:
        app.logger.exception("scan failed")
        os.remove(path)
        flash("Could not read this file. Please upload a clearer scan or a text-based PDF.", "error")
        return redirect(url_for("assess"))
    db = get_db()
    cur = db.execute(
        "INSERT INTO reports (user_id, original_name, stored_name, scan_method, ocr_text, extracted_json) "
        "VALUES (?,?,?,?,?,?)",
        (g.user["id"], secure_filename(f.filename) or "report", stored, scan["method"], scan["text"],
         json.dumps({"values": scan["values"], "notes": scan["notes"]})))
    db.commit()
    return redirect(url_for("review_report", rid=cur.lastrowid))


def own_report(rid):
    row = get_db().execute("SELECT * FROM reports WHERE id=? AND user_id=?", (rid, g.user["id"])).fetchone()
    if row is None:
        abort(404)
    return row


@app.route("/report/<int:rid>/review", methods=["GET", "POST"])
@login_required
def review_report(rid):
    rep = own_report(rid)
    data = json.loads(rep["extracted_json"] or "{}")
    extracted = data.get("values", {})
    keys = all_fields()
    if request.method == "POST":
        raw = {k: request.form.get(k) for k in keys}
        chosen = [d for d in DISEASES if request.form.get(f"run_{d}")]
        if not chosen:
            flash("Select at least one disease to analyse.", "error")
            return redirect(url_for("review_report", rid=rid))
        made, skipped, errs_all = [], [], []
        for d in chosen:
            values, errs = validate(d, raw)
            if errs:
                errs_all += [f"{DISEASES[d]['name']}: {e}" for e in errs]
                continue
            given, total = completeness(d, values)
            if any(np.isnan(values[k]) for k in DISEASES[d]["required"]) or given < max(3, total // 2):
                skipped.append(DISEASES[d]["name"])
                continue
            res = predict(d, values)
            made.append(save_prediction(g.user["id"], d, values, res, "report", rid))
        for e in errs_all:
            flash(e, "error")
        if skipped:
            flash("Not enough data to analyse: " + ", ".join(skipped) +
                  ". Add more values and try again.", "info")
        if not made:
            return redirect(url_for("review_report", rid=rid))
        return redirect(url_for("report_results", rid=rid))
    present = [k for k in keys if k in extracted]
    return render_template("review.html", rep=rep, extracted=extracted, notes=data.get("notes", []),
                           groups=grouped(keys), present=present,
                           ready={d: sum(1 for k in DISEASES[d]["features"] if k in extracted) for d in DISEASES})


@app.route("/report/<int:rid>/file")
@login_required
def report_file(rid):
    rep = own_report(rid)
    return send_file(os.path.join(UPLOAD_DIR, rep["stored_name"]), download_name=rep["original_name"])


@app.route("/report/<int:rid>/results")
@login_required
def report_results(rid):
    rep = own_report(rid)
    rows = get_db().execute("SELECT * FROM predictions WHERE report_id=? AND user_id=? ORDER BY id DESC",
                            (rid, g.user["id"])).fetchall()
    seen, latest = set(), []
    for r in rows:  # newest per disease
        if r["disease"] not in seen:
            seen.add(r["disease"])
            latest.append(r)
    latest.sort(key=lambda r: -r["probability"])
    return render_template("report_results.html", rep=rep,
                           items=[(r, json.loads(r["result_json"])) for r in latest])


@app.route("/result/<int:pid>")
@login_required
def result(pid):
    row = own_prediction(pid)
    res = json.loads(row["result_json"])
    inputs = json.loads(row["inputs_json"])
    return render_template("result.html", row=row, res=res, inputs=inputs, d=DISEASES[row["disease"]])


@app.route("/result/<int:pid>/pdf")
@login_required
def result_pdf(pid):
    from pdf_report import build_pdf
    row = own_prediction(pid)
    buf = build_pdf(g.user, row, json.loads(row["result_json"]), json.loads(row["inputs_json"]))
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"{row['disease']}_risk_report_{pid}.pdf")


@app.route("/history")
@login_required
def history():
    dis = request.args.get("disease", "")
    q = "SELECT * FROM predictions WHERE user_id=?"
    args = [g.user["id"]]
    if dis in DISEASES:
        q += " AND disease=?"
        args.append(dis)
    rows = get_db().execute(q + " ORDER BY id DESC", args).fetchall()
    return render_template("history.html", rows=rows, dis=dis)


@app.route("/history/<int:pid>/delete", methods=["POST"])
@login_required
def delete_prediction(pid):
    own_prediction(pid)
    db = get_db()
    db.execute("DELETE FROM predictions WHERE id=?", (pid,))
    db.commit()
    flash("Assessment deleted.", "info")
    return redirect(url_for("history"))


# ------------------------------------------------------------------ insights / profile
@app.route("/insights")
@login_required
def insights():
    labels = {k: {f: CATALOG[f]["label"] for f in d["features"]} for k, d in DISEASES.items()}
    return render_template("insights.html", metrics=load_metrics(), labels=labels)


@app.route("/api/metrics")
@login_required
def api_metrics():
    return jsonify(load_metrics())


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    db = get_db()
    if request.method == "POST":
        act = request.form.get("action")
        if act == "profile":
            name = request.form.get("full_name", "").strip()
            age = request.form.get("age", "").strip()
            if len(name) < 3 or (age and not (age.isdigit() and 1 <= int(age) <= 120)):
                flash("Please enter a valid name and age.", "error")
            else:
                db.execute("UPDATE users SET full_name=?, age=?, gender=?, phone=? WHERE id=?",
                           (name, int(age) if age else None, request.form.get("gender") or None,
                            request.form.get("phone", "").strip() or None, g.user["id"]))
                db.commit()
                flash("Profile updated.", "success")
        elif act == "password":
            cur, new = request.form.get("current", ""), request.form.get("new", "")
            if not check_password_hash(g.user["password_hash"], cur):
                flash("Current password is incorrect.", "error")
            elif len(new) < 8 or not re.search(r"[A-Za-z]", new) or not re.search(r"\d", new):
                flash("New password must be 8+ characters with letters and numbers.", "error")
            else:
                db.execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(new), g.user["id"]))
                db.commit()
                flash("Password changed successfully.", "success")
        elif act == "delete":
            if check_password_hash(g.user["password_hash"], request.form.get("current", "")):
                for r in db.execute("SELECT stored_name FROM reports WHERE user_id=?", (g.user["id"],)):
                    try:
                        os.remove(os.path.join(UPLOAD_DIR, r["stored_name"]))
                    except OSError:
                        pass
                db.execute("DELETE FROM users WHERE id=?", (g.user["id"],))
                db.commit()
                session.clear()
                flash("Your account and all data have been deleted.", "info")
                return redirect(url_for("index"))
            flash("Password incorrect – account not deleted.", "error")
        return redirect(url_for("profile"))
    logs = db.execute("SELECT * FROM login_logs WHERE user_id=? ORDER BY id DESC LIMIT 5", (g.user["id"],)).fetchall()
    return render_template("profile.html", logs=logs)


@app.errorhandler(404)
def e404(_e):
    return render_template("error.html", code=404, msg="This page could not be found."), 404


@app.errorhandler(400)
def e400(e):
    return render_template("error.html", code=400, msg=getattr(e, "description", "Bad request")), 400


@app.errorhandler(413)
def e413(_e):
    return render_template("error.html", code=413, msg="File too large (max 10 MB)."), 413


@app.errorhandler(500)
def e500(_e):
    return render_template("error.html", code=500, msg="Something went wrong on our side."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=os.environ.get("FLASK_DEBUG") == "1")
