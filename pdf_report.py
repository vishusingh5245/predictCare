"""Downloadable PDF health-risk summary card (future-scope item from the proposal)."""
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ml.schema import CATALOG, DISEASES

LEVEL_COLOR = {"Low": "#16a34a", "Moderate": "#d97706", "High": "#dc2626"}


def build_pdf(user, row, res, inputs):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Disease Risk Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor("#0f766e"), fontSize=20, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#0f172a"), fontSize=13,
                        spaceBefore=12)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#64748b"))
    d = DISEASES[row["disease"]]
    lc = colors.HexColor(LEVEL_COLOR[res["level"]])

    el = [Paragraph("Predictive Healthcare & Disease Risk Prediction System", h1),
          Paragraph(f"{d['name']} Risk Assessment Report", ss["Heading3"]), Spacer(1, 4)]
    info = [["Patient", user["full_name"], "Report ID", f"#{row['id']}"],
            ["Email", user["email"], "Date", row["created_at"]],
            ["Model", res["model"], "Data completeness", f"{res['completeness']}%"]]
    t = Table(info, colWidths=[24 * mm, 62 * mm, 36 * mm, 50 * mm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
                           ("TEXTCOLOR", (2, 0), (2, -1), colors.grey),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    el += [t, Spacer(1, 8)]

    box = Table([[Paragraph(f"<font size=26 color='{LEVEL_COLOR[res['level']]}'><b>{res['percent']}%</b></font>", body),
                  Paragraph(f"<font size=15 color='{LEVEL_COLOR[res['level']]}'><b>{res['level']} Risk</b></font><br/>"
                            f"<font size=9 color='#64748b'>Estimated probability of {d['name'].lower()} "
                            f"from the supplied parameters</font>", body)]],
                colWidths=[45 * mm, 125 * mm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, lc), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 10),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    el += [box, Paragraph("Input Parameters", h2)]

    rows = [["Parameter", "Value", "Reference", "Status"]]
    flags = {f["key"]: f for f in res["flags"]}
    for k in d["features"]:
        v = inputs.get(k)
        meta = CATALOG[k]
        if v is None:
            val = "not provided"
        elif meta.get("type") == "select":
            val = dict(meta["options"]).get(v, v)
        else:
            val = f"{v:g} {meta.get('unit', '')}".strip()
        f = flags.get(k)
        rows.append([meta["label"], val, f["ref"] if f else "-", f["status"].upper() if f else "-"])
    t = Table(rows, colWidths=[62 * mm, 40 * mm, 38 * mm, 30 * mm], repeatRows=1)
    st = [("FONTSIZE", (0, 0), (-1, -1), 9), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]
    for i, r in enumerate(rows[1:], 1):
        if r[3] in ("HIGH", "LOW"):
            st.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#dc2626")))
    t.setStyle(TableStyle(st))
    el.append(t)

    if res["factors"]:
        el.append(Paragraph("Key Contributing Factors", h2))
        for f in res["factors"][:5]:
            sign = "increases" if f["impact"] > 0 else "lowers"
            el.append(Paragraph(f"• <b>{f['label']}</b> ({f['value']:g}) {sign} risk ({abs(f['impact'])}% of total influence)", body))
    el.append(Paragraph("Recommendations", h2))
    for a in res["advice"]:
        el.append(Paragraph("• " + a, body))
    el += [Spacer(1, 14),
           Paragraph("<b>Disclaimer:</b> This is an academic prototype that gives a preliminary, machine-learning based "
                     "risk estimate. It is NOT a medical diagnosis and does not replace advice from a qualified doctor.",
                     small)]
    doc.build(el)
    buf.seek(0)
    return buf
