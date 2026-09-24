"""AeroTwin PDF report generator.

Creates a self-contained engineering-style PDF report from the processed
telemetry dataframe used by the Streamlit dashboard. Charts are rendered as
high-resolution static figures so the report is portable and does not depend
on Plotly/Kaleido being installed.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

PAGE_W, PAGE_H = A4
NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1976D2")
TEAL = colors.HexColor("#00897B")
RED = colors.HexColor("#C62828")
AMBER = colors.HexColor("#F9A825")
GREEN = colors.HexColor("#2E7D32")
LIGHT = colors.HexColor("#F4F7FA")
MID = colors.HexColor("#D9E2EC")
TEXT = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")


def _status_color(status: str):
    return {"HEALTHY": GREEN, "DEGRADING": AMBER, "WARNING": colors.HexColor("#EF6C00"), "CRITICAL": RED}.get(status, BLUE)


def _fmt(v: Any, digits=1, suffix=""):
    try:
        return f"{float(v):.{digits}f}{suffix}"
    except Exception:
        return "—"


def _chart_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _line_chart(d, cols, labels, title, ylabel, figsize=(7.1, 3.25), dashed=None):
    fig, ax = plt.subplots(figsize=figsize)
    for col, label in zip(cols, labels):
        style = "--" if dashed and col in dashed else "-"
        ax.plot(d["t"], d[col], linewidth=1.8, linestyle=style, label=label)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mission time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=.22)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    return _chart_png(fig)


def _bar_chart(labels, values, title, xlabel="Health (%)"):
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    vals = np.asarray(values, dtype=float) * 100
    bars = ax.barh(labels, vals)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=.22)
    for b, v in zip(bars, vals):
        ax.text(min(v + 1, 96), b.get_y() + b.get_height()/2, f"{v:.0f}%", va="center", fontsize=8)
    fig.tight_layout()
    return _chart_png(fig)


def _gauge(value, title):
    value = float(np.clip(value, 0, 1))
    fig, ax = plt.subplots(figsize=(3.1, 2.15), subplot_kw={"projection": "polar"})
    theta = np.linspace(np.pi, 0, 180)
    ax.plot(theta, np.ones_like(theta), linewidth=18, color="#E7EDF3", solid_capstyle="round")
    n = max(2, int(180 * value))
    ax.plot(theta[:n], np.ones(n), linewidth=18, color={"HEALTHY":"#2E7D32","DEGRADING":"#F9A825","WARNING":"#EF6C00","CRITICAL":"#C62828"}.get(title, "#1976D2"), solid_capstyle="round")
    ax.set_ylim(0, 1.25)
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.grid(False)
    ax.text(np.pi/2, .35, f"{value*100:.0f}%", ha="center", va="center", fontsize=21, fontweight="bold")
    ax.text(np.pi/2, .10, title, ha="center", va="center", fontsize=9)
    fig.tight_layout(pad=.3)
    return _chart_png(fig)


class _ReportDoc(BaseDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        frame = Frame(16*mm, 15*mm, PAGE_W-32*mm, PAGE_H-31*mm, id="normal")
        self.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=self._footer)])

    def _footer(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(MID)
        canvas.line(16*mm, 12*mm, PAGE_W-16*mm, 12*mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(16*mm, 7.5*mm, "AeroTwin • AI-assisted Digital Twin for MALE UAV Aero-Piston Engine Health")
        canvas.drawRightString(PAGE_W-16*mm, 7.5*mm, f"Page {doc.page}")
        canvas.restoreState()


def build_pdf_report(d: pd.DataFrame, source_name: str = "Unknown", scenario: str | None = None,
                     diagnosis=None, rul_result=None, rul_confidence=None,
                     threshold_result=None) -> bytes:
    """Return a polished PDF report as bytes."""
    d = d.copy()
    r = d.iloc[-1]
    if diagnosis is None:
        diagnosis = ("UNAVAILABLE", "UNKNOWN", 0.0, [])
    fault, severity, confidence, contributors = diagnosis
    if rul_result is None:
        rul_result = (0, 0)
    if threshold_result is None:
        threshold_result = (None, None, None, None, None)

    buf = BytesIO()
    doc = _ReportDoc(buf, pagesize=A4, title="AeroTwin Engine Health Report", author="AeroTwin")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=colors.white, spaceAfter=5)
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontSize=10.5, leading=14, textColor=TEXT)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=NAVY, spaceBefore=5, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=TEXT, spaceBefore=5, spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.2, leading=13, textColor=TEXT, spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10.5, textColor=MUTED)
    metric = ParagraphStyle("metric", parent=body, alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=15, leading=17)
    label = ParagraphStyle("label", parent=small, alignment=TA_CENTER, fontSize=7.2)
    callout = ParagraphStyle("callout", parent=body, fontSize=9, leading=12)

    story = []
    # Cover / executive summary
    cover = Table([[Paragraph("AeroTwin", title), Paragraph("ENGINE HEALTH REPORT", ParagraphStyle("ct", parent=title, alignment=TA_RIGHT, fontSize=10, leading=12))]], colWidths=[110*mm, 60*mm])
    cover.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), NAVY), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 12), ("RIGHTPADDING", (0,0), (-1,-1), 12), ("TOPPADDING", (0,0), (-1,-1), 12), ("BOTTOMPADDING", (0,0), (-1,-1), 12)]))
    story += [cover, Spacer(1, 8)]
    story.append(Paragraph("AI-assisted Digital Twin for MALE UAV Aero-Piston Engine Health", subtitle))
    story.append(Spacer(1, 12))

    status = str(getattr(r, "status", "UNKNOWN"))
    health_score = float(getattr(r, "overall_health", 0))
    source_text = source_name.replace("&", "&amp;")
    scenario_text = (scenario or (source_name.split(": ", 1)[1] if source_name.startswith("Simulated:") else "Telemetry upload")).replace("&", "&amp;")
    metadata = [
        [Paragraph("REPORT SOURCE", small), Paragraph("MISSION / SCENARIO", small), Paragraph("SAMPLES", small), Paragraph("FINAL STATUS", small)],
        [Paragraph(source_text, body), Paragraph(scenario_text, body), Paragraph(f"{len(d):,}", metric), Paragraph(status, ParagraphStyle("st", parent=metric, textColor=_status_color(status)))],
    ]
    mt = Table(metadata, colWidths=[50*mm, 48*mm, 25*mm, 47*mm])
    mt.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), LIGHT), ("BOX", (0,0), (-1,-1), .5, MID), ("INNERGRID", (0,0), (-1,-1), .4, MID), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
    story.append(mt)
    story.append(Spacer(1, 9))

    # Summary metrics with gauge
    gauge = Image(_gauge(health_score, status), width=62*mm, height=43*mm)
    summary_text = (
        f"<b>Current engine health:</b> {health_score*100:.0f}%. "
        f"The latest telemetry sample is classified as <b>{status}</b>. "
        f"The hybrid diagnostic layer reports <b>{fault}</b> at {confidence*100:.0f}% confidence. "
        f"This report is a prototype engineering assessment; it is not a certified aircraft maintenance release."
    )
    summary = Table([[gauge, Paragraph(summary_text, callout)]], colWidths=[70*mm, 100*mm])
    summary.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("BACKGROUND", (0,0), (-1,-1), colors.white), ("BOX", (0,0), (-1,-1), .5, MID), ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7), ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6)]))
    story += [summary, Spacer(1, 8), Paragraph("1. Executive Summary", h1)]

    # Key metrics row
    rr = rul_result
    metrics = [
        ("Overall Health", f"{health_score*100:.0f}%"),
        ("RPM", _fmt(r.rpm, 0)),
        ("CHT", _fmt(r.cht, 1, " °C")),
        ("EGT", _fmt(r.egt, 0, " °C")),
        ("Twin Fit", f"{float(getattr(r, "twin_fit", 0))*100:.0f}%"),
    ]
    row = [[Paragraph(v, metric) for _, v in metrics], [Paragraph(k, label) for k, _ in metrics]]
    kt = Table(row, colWidths=[34*mm]*5)
    kt.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), LIGHT), ("BOX", (0,0), (-1,-1), .4, MID), ("INNERGRID", (0,0), (-1,-1), .4, MID), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]))
    story.append(kt)
    story.append(Spacer(1, 8))

    # Diagnosis + RUL
    rul_lo, rul_hi = rr
    threshold_idx, ai_idx, threshold_time, ai_time, threshold_parameter = threshold_result
    diag_lines = [
        f"<b>Probable fault:</b> {fault}",
        f"<b>Severity:</b> {severity}",
        f"<b>Diagnostic confidence:</b> {confidence*100:.0f}%",
        f"<b>Top contributors:</b> {', '.join(str(x[0]) for x in contributors) if contributors else 'None identified'}",
    ]
    rul_text = (
        f"<b>Prototype RUL:</b> {float(rul_lo):.1f}–{float(rul_hi):.1f} equivalent operating hours<br/>"
        f"<b>RUL confidence:</b> {(float(rul_confidence)*100 if rul_confidence is not None else 0):.0f}%<br/>"
        f"The RUL model uses recent health-trend slope and an explicit prototype mapping of 60 simulation samples = 1 operating hour."
    )
    if threshold_time is not None or ai_time is not None:
        detection_text = f"<b>First confirmed threshold alarm:</b> {threshold_time if threshold_time is not None else 'Not detected'}<br/><b>First confirmed AI anomaly:</b> {ai_time if ai_time is not None else 'Not detected'}"
        if threshold_parameter:
            detection_text += f"<br/><b>Threshold parameter(s):</b> {threshold_parameter}"
    else:
        detection_text = "No confirmed threshold/AI detection timestamp was available."
    dt = Table([[Paragraph("<b>DIAGNOSTIC ASSESSMENT</b><br/>" + "<br/>".join(diag_lines), body), Paragraph("<b>RUL &amp; DETECTION</b><br/>" + rul_text + "<br/>" + detection_text, body)]], colWidths=[84*mm, 86*mm])
    dt.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), LIGHT), ("BOX", (0,0), (-1,-1), .5, MID), ("INNERGRID", (0,0), (-1,-1), .4, MID), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.append(dt)

    story.append(PageBreak())
    story.append(Paragraph("2. Health & Digital Twin Analysis", h1))
    subsystem_cols = [("thermal_health", "Thermal"), ("mechanical_health", "Mechanical"), ("lubrication_health", "Lubrication"), ("electrical_health", "Electrical"), ("fuel_health", "Fuel / Injection")]
    available = [(c,n) for c,n in subsystem_cols if c in d.columns]
    if available:
        story.append(Image(_bar_chart([n for c,n in available], [d.iloc[-1][c] for c,n in available], "Subsystem health at latest sample"), width=171*mm, height=72*mm))
        story.append(Spacer(1, 5))

    if all(c in d.columns for c in ["cht", "cht_expected", "egt", "egt_expected"]):
        story.append(Image(_line_chart(d, ["cht","cht_expected","egt","egt_expected"], ["Observed CHT","Expected CHT","Observed EGT","Expected EGT"], "Thermal telemetry vs Digital Twin expectation", "Temperature (°C)", dashed={"cht_expected","egt_expected"}), width=171*mm, height=78*mm))
    story.append(Spacer(1, 4))
    if all(c in d.columns for c in ["oil_pressure","oil_pressure_expected","vibration","vibration_expected"]):
        dd = d.copy(); dd["vibration_x100"] = dd["vibration"]*100; dd["vibration_expected_x100"] = dd["vibration_expected"]*100
        story.append(Image(_line_chart(dd, ["oil_pressure","oil_pressure_expected","vibration_x100","vibration_expected_x100"], ["Observed oil pressure","Expected oil pressure","Vibration ×100","Expected vibration ×100"], "Lubrication and mechanical state", "Value / scaled vibration", dashed={"oil_pressure_expected","vibration_expected_x100"}), width=171*mm, height=78*mm))

    story.append(PageBreak())
    story.append(Paragraph("3. AI Diagnostics & Anomaly Evidence", h1))
    if "anomaly_score" in d.columns:
        fig = plt.figure(figsize=(7.1, 3.0)); ax = fig.add_subplot(111)
        ax.plot(d.t, d.anomaly_score, linewidth=1.8, label="AI anomaly score")
        ax.axhline(.35, linestyle="--", linewidth=1, label="Prototype alert reference")
        ax.set_title("Multivariate anomaly score", loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mission time"); ax.set_ylabel("Score"); ax.grid(True, alpha=.22); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); story.append(Image(_chart_png(fig), width=171*mm, height=72*mm))
    story.append(Spacer(1, 7))
    contrib_rows = [[Paragraph("Sensor / signal", small), Paragraph("Absolute deviation", small), Paragraph("Interpretation", small)]]
    for name, val in contributors:
        contrib_rows.append([Paragraph(str(name), body), Paragraph(f"{float(val):.2f}", body), Paragraph("Higher deviation contributed more strongly to the hybrid diagnostic evidence.", body)])
    if len(contrib_rows) > 1:
        ct = Table(contrib_rows, colWidths=[52*mm, 35*mm, 83*mm])
        ct.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), .4, MID), ("VALIGN", (0,0), (-1,-1), "TOP"), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]), ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
        story.append(ct)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Maintenance advisory", h2))
    advisory = {
        "THERMAL DEGRADATION / OVERHEATING": "Inspect cooling airflow, CHT/EGT trend, thermal management and operating load before the next sortie.",
        "LUBRICATION DEGRADATION": "Inspect oil level/quality, oil pressure circuit, lubrication passages and filter condition before continued operation.",
        "MECHANICAL / VIBRATION ABNORMALITY": "Inspect rotating components, mounts, bearings and vibration sources; confirm with physical inspection before dispatch.",
        "INJECTOR / FUEL ABNORMALITY": "Inspect fuel pressure, injector condition, injection timing and fuel delivery before continued high-load operation.",
        "NORMAL / NO CONFIRMED FAULT": "Continue routine monitoring. No specific fault pattern was confirmed by the prototype diagnostic layer.",
    }.get(fault, "Review the indicated sensor deviations and perform engineering inspection before continued operation if the condition persists.")
    story.append(Table([[Paragraph(advisory, callout)]], colWidths=[170*mm], style=[("BACKGROUND", (0,0), (-1,-1), LIGHT), ("BOX", (0,0), (-1,-1), .5, MID), ("LEFTPADDING", (0,0), (-1,-1), 9), ("RIGHTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)]))

    story.append(PageBreak())
    story.append(Paragraph("4. Key Telemetry Snapshot", h1))
    cols = [c for c in ["t","mission_phase","throttle","altitude_ft","rpm","cht","egt","oil_pressure","oil_temp","fuel_flow","vibration","battery_voltage","alternator_health","injection_timing"] if c in d.columns]
    snap = d.iloc[np.linspace(0, len(d)-1, min(12, len(d))).astype(int)][cols].copy()
    header_map = {"t":"Time","mission_phase":"Phase","throttle":"Throttle","altitude_ft":"Alt (ft)","rpm":"RPM","cht":"CHT","egt":"EGT","oil_pressure":"Oil P","oil_temp":"Oil T","fuel_flow":"Fuel","vibration":"Vib","battery_voltage":"Batt V","alternator_health":"Alt H","injection_timing":"Inj. timing"}
    data = [[Paragraph(header_map.get(c,c), small) for c in cols]]
    for _, row in snap.iterrows():
        vals=[]
        for c in cols:
            v=row[c]
            if isinstance(v, (float, np.floating)): text=f"{float(v):.1f}"
            else: text=str(v)
            vals.append(Paragraph(text, ParagraphStyle("cell", parent=small, fontSize=6.6, leading=8)))
        data.append(vals)
    widths = [170*mm/len(cols)]*len(cols)
    table=Table(data,colWidths=widths,repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,MID),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Method & limitations", h2))
    story.append(Paragraph("The report is generated from the same processed dataframe used by the AeroTwin dashboard. The Digital Twin estimates a healthy expected state from throttle, altitude and ambient temperature, while the anomaly detector and rule-based engineering relationships provide diagnostic evidence. RUL is explicitly a prototype estimate and should be calibrated against historical engine degradation/run-to-failure data before operational use.", body))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')} • AeroTwin prototype • Source: {source_text}", small))

    doc.build(story)
    return buf.getvalue()
