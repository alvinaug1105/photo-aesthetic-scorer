"""
src/visualizer.py — Plotly charts for Photo Scorer.
"""

import plotly.graph_objects as go
import numpy as np

DARK  = "#15151E"
CARD  = "#1e1e2e"
RED   = "#E8002D"
WHITE = "#FFFFFF"
GREY  = "#38383F"
TEAL  = "#00D2BE"
GOLD  = "#FFD700"

LABELS = {
    "brightness":  "Brightness",
    "sharpness":   "Sharpness",
    "composition": "Composition",
    "color":       "Color",
    "noise":       "Noise",
    "contrast":    "Contrast",
}


def radar_chart(features: dict, nima_score: float, score_color: str = TEAL) -> go.Figure:
    labs = [LABELS[k] for k in features] + [LABELS[list(features.keys())[0]]]
    vals = list(features.values()) + [list(features.values())[0]]
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=labs, fill="toself",
        fillcolor=f"rgba(0,210,190,0.15)",
        line=dict(color=score_color, width=2.5),
        name="Your Photo",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[7.0]*len(labs), theta=labs,
        line=dict(color=GREY, width=1, dash="dot"),
        name="Good Baseline",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,10], color=WHITE, gridcolor=GREY, tickfont=dict(size=10)),
            angularaxis=dict(color=WHITE, tickfont=dict(size=11)),
            bgcolor=CARD,
        ),
        showlegend=True,
        legend=dict(bgcolor=DARK, font=dict(color=WHITE, size=11), orientation="h", y=-0.15),
        paper_bgcolor=DARK,
        font=dict(color=WHITE, family="Arial"),
        title=dict(text=f"Photo Breakdown  ·  NIMA {nima_score}/10",
                   x=0.5, font=dict(size=15, color=WHITE)),
        margin=dict(l=55, r=55, t=70, b=55),
        height=400,
    )
    return fig


def distribution_chart(distribution: list[float]) -> go.Figure:
    scores = list(range(1, 11))
    peak   = int(np.argmax(distribution))
    colors = [RED if i == peak else TEAL for i in range(10)]
    fig    = go.Figure(go.Bar(
        x=scores, y=[round(d*100,1) for d in distribution],
        marker_color=colors,
        text=[f"{d*100:.1f}%" for d in distribution],
        textposition="outside",
        textfont=dict(color=WHITE, size=9),
    ))
    fig.update_layout(
        title=dict(text="Score Distribution", x=0.5, font=dict(size=14, color=WHITE)),
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font=dict(color=WHITE),
        xaxis=dict(title="Score (1–10)", gridcolor=GREY, tickvals=scores, color=WHITE),
        yaxis=dict(title="Probability %", gridcolor=GREY, color=WHITE),
        margin=dict(l=40, r=20, t=50, b=45),
        height=280,
    )
    return fig


def batch_chart(results: list[dict]) -> go.Figure:
    names  = [r.get("name", f"Photo {i+1}") for i, r in enumerate(results)]
    scores = [r["score"] for r in results]
    colors = [GOLD if s == max(scores) else TEAL for s in scores]
    fig    = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker_color=colors,
        text=[str(s) for s in scores],
        textposition="outside",
        textfont=dict(color=WHITE, size=11),
    ))
    fig.update_layout(
        title=dict(text="📸 Photo Ranking", x=0.5, font=dict(size=15, color=WHITE)),
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font=dict(color=WHITE),
        xaxis=dict(title="NIMA Score", range=[0,11], gridcolor=GREY, color=WHITE),
        yaxis=dict(gridcolor=GREY, color=WHITE),
        margin=dict(l=130, r=50, t=60, b=45),
        height=max(300, len(results)*50),
    )
    return fig


def gallery_bar(feat_cols: list, avg_vals: list) -> go.Figure:
    colors = [GOLD if v == max(avg_vals) else RED if v == min(avg_vals) else TEAL for v in avg_vals]
    fig    = go.Figure(go.Bar(
        x=feat_cols, y=avg_vals,
        marker_color=colors,
        text=[f"{v:.1f}" for v in avg_vals],
        textposition="outside",
        textfont=dict(color=WHITE, size=12),
    ))
    fig.add_hline(y=7.0, line_dash="dot", line_color=GREY,
                  annotation_text="Good Baseline (7.0)",
                  annotation_font_color=WHITE)
    fig.update_layout(
        title=dict(text="Your Strengths & Weaknesses", x=0.5, font=dict(size=15, color=WHITE)),
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font=dict(color=WHITE),
        xaxis=dict(gridcolor=GREY, color=WHITE),
        yaxis=dict(title="Avg Score", range=[0,11], gridcolor=GREY, color=WHITE),
        margin=dict(l=40, r=20, t=60, b=45),
        height=340,
    )
    return fig
