"""Plotly charts and the PDF feedback report."""
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

METRICS = ["clarity", "relevance", "depth", "star"]


def _df(results: list) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, 1):
        rows.append({"Q": f"Q{i}", "type": r["question"].get("type", ""), **{m: r["score"][m] for m in METRICS},
                     "overall": r["score"]["overall"], "fillers": r["score"]["filler_total"]})
    return pd.DataFrame(rows)


def score_trend(results):
    df = _df(results)
    fig = go.Figure(go.Scatter(x=df["Q"], y=df["overall"], mode="lines+markers+text",
                               text=df["overall"], textposition="top center", line=dict(width=3)))
    fig.update_layout(title="Score per question", yaxis=dict(range=[0, 10.5]), height=350)
    return fig


def radar(results):
    df = _df(results)
    vals = [round(df[m].mean(), 1) for m in METRICS]
    labels = ["Clarity", "Relevance", "Depth", "STAR structure"]
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself"))
    fig.update_layout(title="Skill profile", polar=dict(radialaxis=dict(range=[0, 10])), height=350)
    return fig


def filler_chart(results):
    df = _df(results)
    fig = go.Figure(go.Bar(x=df["Q"], y=df["fillers"]))
    fig.update_layout(title="Filler words per answer", height=300)
    return fig


def skill_gap_chart(gap: dict):
    fig = go.Figure(go.Bar(x=["Matched", "Missing", "Extra"],
                           y=[len(gap["matched"]), len(gap["missing"]), len(gap["extra"])]))
    fig.update_layout(title=f"Resume vs JD skills (match {gap['match_pct']}%)", height=300)
    return fig


def overall_average(results) -> float:
    return round(sum(r["score"]["overall"] for r in results) / len(results), 1) if results else 0.0


def _clean(text) -> str:
    return str(text).encode("latin-1", "replace").decode("latin-1")


def build_pdf(role: str, gap: dict, results: list) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def line(text, size=10, style="", h=6):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, h, _clean(text), new_x="LMARGIN", new_y="NEXT")

    line("InterviewIQ - Feedback Report", 18, "B", 10)
    line(f"Role: {role}", 11)
    line(f"Overall score: {overall_average(results)} / 10", 12, "B")
    line(f"Skill match: {gap['match_pct']}%   |   Missing: {', '.join(gap['missing']) or 'none'}")
    pdf.ln(4)
    for i, r in enumerate(results, 1):
        s = r["score"]
        line(f"Q{i} [{r['question'].get('type', '')}] {r['question']['question']}", 11, "B")
        line(f"Answer: {r['answer']}")
        line(f"Clarity {s['clarity']} | Relevance {s['relevance']} | Depth {s['depth']} | "
             f"STAR {s['star']} | Overall {s['overall']} | Fillers {s['filler_total']}", 10, "B")
        line(f"Strengths: {s['strengths']}")
        line(f"Improve: {s['improvements']}")
        if s.get("model_answer_hint"):
            line(f"Stronger answer: {s['model_answer_hint']}")
        pdf.ln(4)
    return bytes(pdf.output())
