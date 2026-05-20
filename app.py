"""
app.py — Photo Aesthetic Scorer v3.0
Premium dark UI · Full-width · Glow animations · Pro Benchmark · Percentile rank
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="📸 Photo Aesthetic Scorer",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BG_DARK  = "#15151E"
BG_CARD  = "#1e1e2e"
BG_CARD2 = "#2a2a3e"
RED      = "#E8002D"
TEAL     = "#00D2BE"
GOLD     = "#FFD700"
ORANGE   = "#FF8C00"
GREY     = "#38383F"
WHITE    = "#FFFFFF"
SUBTEXT  = "#9a9a9a"

CV_DIM_INFO = {
    "sharpness":   {"label": "Sharpness",   "icon": "🔍", "desc": "Laplacian variance — higher = sharper",       "pro": 7.2},
    "composition": {"label": "Composition", "icon": "📐", "desc": "RoT / Symmetry / Negative Space analysis",    "pro": 6.8},
    "brightness":  {"label": "Brightness",  "icon": "☀️", "desc": "Mean brightness (0=black, 10=white)",         "pro": 6.5},
    "contrast":    {"label": "Contrast",    "icon": "🌗", "desc": "Light/dark standard deviation",               "pro": 7.0},
    "color":       {"label": "Color",       "icon": "🎨", "desc": "HSV saturation mean",                         "pro": 6.3},
    "noise":       {"label": "Noise",       "icon": "📡", "desc": "Higher = less noise",                         "pro": 6.9},
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""<style>
/* ── Full width ── */
.stApp {{
    background-color: {BG_DARK};
    color: {WHITE};
}}
.main .block-container {{
    background-color: {BG_DARK};
    color: {WHITE};
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1.5rem !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #1a1a2e 0%, {BG_DARK} 100%);
    border-right: 2px solid {RED};
    min-width: 220px !important;
    max-width: 260px !important;
}}
[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1px solid {GREY};
    border-radius: 10px;
    padding: 16px 20px;
    border-top: 3px solid {RED};
}}
[data-testid="stMetricValue"] {{
    color: {WHITE} !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{ color: {SUBTEXT} !important; font-size: 0.8rem !important; }}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {RED}, #c0001f);
    color: {WHITE};
    border: none;
    border-radius: 8px;
    font-weight: 700;
    width: 100%;
    padding: 0.65rem 1rem;
    font-size: 0.95rem;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(232,0,45,0.4);
}}

/* ── Headers ── */
h1 {{ color: {WHITE}; font-weight: 800; }}
h2 {{ color: {WHITE}; border-left: 4px solid {RED}; padding-left: 12px; font-weight: 700; }}
h3 {{ color: {RED}; font-weight: 600; }}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {BG_CARD};
    border: 2px dashed {GREY};
    border-radius: 10px;
    padding: 10px;
}}
[data-testid="stFileUploader"]:hover {{ border-color: {RED}; }}

/* ── Glow animations ── */
@keyframes glow-red  {{ 0%,100%{{box-shadow:0 0 20px {RED}44}} 50%{{box-shadow:0 0 50px {RED}99}} }}
@keyframes glow-gold {{ 0%,100%{{box-shadow:0 0 20px {GOLD}44}} 50%{{box-shadow:0 0 50px {GOLD}99}} }}
@keyframes glow-teal {{ 0%,100%{{box-shadow:0 0 20px {TEAL}44}} 50%{{box-shadow:0 0 50px {TEAL}99}} }}
@keyframes count-up  {{ from{{opacity:0;transform:scale(0.7)}} to{{opacity:1;transform:scale(1)}} }}
.glow-red  {{ animation: glow-red  2s infinite; border: 2px solid {RED}  !important; }}
.glow-gold {{ animation: glow-gold 2s infinite; border: 2px solid {GOLD} !important; }}
.glow-teal {{ animation: glow-teal 2s infinite; border: 2px solid {TEAL} !important; }}
.score-number {{ animation: count-up 0.6s ease-out; }}

/* ── Cards ── */
.score-badge {{
    background: linear-gradient(135deg, {BG_CARD}, {BG_CARD2});
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 10px 0;
}}
.photo-card {{
    background: linear-gradient(135deg, {BG_CARD}, #16213e);
    border-radius: 12px;
    padding: 14px;
    border: 1px solid {BG_CARD2};
    transition: border-color 0.3s ease, transform 0.2s ease;
    margin-bottom: 8px;
}}
.photo-card:hover {{ border-color: {RED}; transform: translateY(-2px); }}

/* ── Tips ── */
.tip-ok   {{ background:#0d2b1e; border-left:3px solid {TEAL};   padding:8px 12px; border-radius:6px; margin:4px 0; }}
.tip-warn {{ background:#2b1a0d; border-left:3px solid {ORANGE}; padding:8px 12px; border-radius:6px; margin:4px 0; }}

/* ── Misc ── */
hr {{ border-color: {GREY}; }}
[data-testid="stRadio"] label {{ color: {WHITE} !important; }}

/* ── Mobile ── */
@media (max-width: 768px) {{
    .score-number {{ font-size: 2.5rem !important; }}
    .main .block-container {{ padding: 0.5rem !important; }}
}}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_score_meta(score: float):
    percentile = scipy_stats.norm.cdf(score, loc=5.0, scale=0.9) * 100
    if   score >= 7.5: return "glow-teal", TEAL,   percentile
    elif score >= 6.5: return "glow-gold", GOLD,   percentile
    elif score >= 5.5: return "glow-gold", ORANGE, percentile
    else:              return "glow-red",  RED,    percentile


def render_score_badge(nima: dict):
    glow_cls, color, pct = get_score_meta(nima["score"])
    st.markdown(f"""
    <div class="score-badge {glow_cls}">
        <div class="score-number"
             style="font-size:4rem;font-weight:900;color:{color};line-height:1;">
            {nima['score']}
        </div>
        <div style="color:{SUBTEXT};font-size:0.9rem;margin:4px 0;">out of 10.0</div>
        <div style="font-size:1.3rem;margin-top:8px;">{nima['rating']}</div>
        <div style="color:{SUBTEXT};font-size:0.8rem;margin-top:4px;">
            ±{nima['std']} confidence
        </div>
        <hr style="border-color:{color}33;margin:12px 0;">
        <div style="color:{SUBTEXT};font-size:0.85rem;">
            📊 Better than
            <b style="color:{color}">{pct:.0f}%</b>
            of AVA dataset photos
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_cv_dimensions(cv_feat: dict):
    st.markdown("### 📊 Photo Breakdown")
    cols = st.columns(3)
    for i, (key, info) in enumerate(CV_DIM_INFO.items()):
        val = cv_feat.get(key, 0)
        pro = info["pro"]
        color = TEAL if val >= pro else GOLD if val >= 5.0 else RED
        with cols[i % 3]:
            st.markdown(f"""
            <div class="photo-card">
                <div style="font-size:1.4rem">{info['icon']}</div>
                <div style="font-weight:700;color:{color};font-size:1.4rem">{val}/10</div>
                <div style="font-size:0.9rem;font-weight:600">{info['label']}</div>
                <div style="font-size:0.72rem;color:{SUBTEXT};margin-top:3px">{info['desc']}</div>
                <div style="margin-top:8px;background:{BG_CARD2};border-radius:4px;height:4px;">
                    <div style="background:{color};width:{min(val/10*100,100):.0f}%;
                                height:4px;border-radius:4px;"></div>
                </div>
                <div style="font-size:0.7rem;color:{SUBTEXT};margin-top:3px">
                    Pro avg: {pro}
                </div>
            </div>
            """, unsafe_allow_html=True)


def radar_chart_enhanced(cv_feat: dict, nima_score: float, score_color: str):
    keys   = list(CV_DIM_INFO.keys())
    labels = [CV_DIM_INFO[k]["label"] for k in keys]
    vals   = [cv_feat.get(k, 0) for k in keys]
    pros   = [CV_DIM_INFO[k]["pro"]   for k in keys]

    labels_c = labels + [labels[0]]
    vals_c   = vals   + [vals[0]]
    pros_c   = pros   + [pros[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=labels_c, fill="toself", name="Your Photo",
        line=dict(color=score_color, width=2.5),
        fillcolor=f"rgba({_hex_to_rgb(score_color)},0.15)",
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=pros_c, theta=labels_c, fill="toself", name="Pro Benchmark",
        line=dict(color=GOLD, width=2, dash="dot"),
        fillcolor="rgba(255,215,0,0.06)",
        hovertemplate="%{theta} Pro avg: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG_CARD,
            radialaxis=dict(visible=True, range=[0, 10],
                            gridcolor=BG_CARD2, color=SUBTEXT,
                            tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=BG_CARD2, color=WHITE),
        ),
        paper_bgcolor=BG_DARK,
        font=dict(color=WHITE, family="Arial"),
        legend=dict(bgcolor=BG_CARD, bordercolor=GREY, x=0.82, y=1.1),
        title=dict(text=f"📡 Photo Breakdown · NIMA {nima_score}",
                   x=0.5, font=dict(size=15, color=WHITE)),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420,
    )
    return fig


def distribution_chart(distribution: list):
    scores = list(range(1, 11))
    colors = [RED if s <= 4 else ORANGE if s <= 6 else GOLD if s <= 7 else TEAL
              for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=[round(v * 100, 1) for v in distribution],
        marker=dict(color=colors, opacity=0.85),
        text=[f"{v*100:.1f}%" for v in distribution],
        textposition="outside",
        textfont=dict(color=WHITE, size=10),
        hovertemplate="Score %{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Score Distribution", x=0.5,
                   font=dict(size=14, color=WHITE)),
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font=dict(color=WHITE),
        xaxis=dict(title="Score (1–10)", gridcolor=GREY, tickvals=scores),
        yaxis=dict(title="Probability %", gridcolor=GREY,
                   range=[0, max(distribution) * 115]),
        margin=dict(l=30, r=20, t=50, b=40),
        height=320,
    )
    return fig


def generate_suggestions(score: float, cv_feat: dict) -> list[str]:
    tips = []
    if cv_feat.get("sharpness", 10) < CV_DIM_INFO["sharpness"]["pro"]:
        tips.append("🔍 **Sharpness** — Use f/8–f/11, ensure focus is on subject, avoid camera shake")
    if cv_feat.get("composition", 10) < CV_DIM_INFO["composition"]["pro"]:
        tips.append("📐 **Composition** — Try Rule of Thirds, symmetry, or deliberate negative space — no single rule is mandatory")
    if cv_feat.get("brightness", 0) < 3.5:
        tips.append("☀️ **Underexposed** — Lift Exposure +0.5EV or raise Shadows in Lightroom")
    elif cv_feat.get("brightness", 0) > 8.5:
        tips.append("☀️ **Overexposed** — Pull Highlights -20 to recover detail")
    if cv_feat.get("color", 10) < CV_DIM_INFO["color"]["pro"]:
        tips.append("🎨 **Color** — Add Vibrance +15 (more natural than Saturation)")
    if cv_feat.get("noise", 10) < CV_DIM_INFO["noise"]["pro"]:
        tips.append("📡 **Noise** — Keep ISO ≤ 800 or apply Noise Reduction +30 in post")
    if cv_feat.get("contrast", 10) < CV_DIM_INFO["contrast"]["pro"]:
        tips.append("🌗 **Contrast** — Add Contrast +10 or use an S-curve for depth")
    if score < 5.0:
        tips.append("💡 **General** — Shoot during Golden Hour (30 min after sunrise/before sunset) for the best natural light")
    return tips if tips else ["✅ Great shot! Composition and lighting are well controlled 📸"]


def batch_chart(results: list) -> go.Figure:
    names  = [r["name"][:20] + "…" if len(r["name"]) > 20 else r["name"]
              for r in results]
    scores = [r["score"] for r in results]
    colors = [TEAL if s >= 6.5 else GOLD if s >= 5.5 else RED for s in scores]
    fig = go.Figure(go.Bar(
        x=names, y=scores,
        marker=dict(color=colors, opacity=0.9),
        text=[f"{s}" for s in scores],
        textposition="outside",
        textfont=dict(color=WHITE, size=11),
        hovertemplate="%{x}<br>Score: %{y}<extra></extra>",
    ))
    fig.add_hline(y=5.0, line_dash="dot", line_color=GREY,
                  annotation_text="AVA Average (5.0)",
                  annotation_font_color=SUBTEXT)
    fig.update_layout(
        title=dict(text="📊 Batch Score Rankings", x=0.5,
                   font=dict(size=15, color=WHITE)),
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font=dict(color=WHITE),
        xaxis=dict(gridcolor=GREY, tickangle=-35),
        yaxis=dict(gridcolor=GREY, range=[0, 11], title="NIMA Score"),
        margin=dict(l=30, r=20, t=60, b=80),
        height=380,
    )
    return fig


def batch_trend_chart(results: list) -> go.Figure:
    scores  = [r["score"] for r in results]
    names   = [r["name"]  for r in results]
    rolling = pd.Series(scores).rolling(5, min_periods=1).mean().tolist()
    colors  = [TEAL if s >= 6.5 else GOLD if s >= 5.5 else RED for s in scores]
    avg     = sum(scores) / len(scores)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Score Distribution", "Score Trend (5-photo rolling avg)"],
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=15,
        marker=dict(color=RED, opacity=0.8, line=dict(color=BG_DARK, width=1)),
        name="Distribution",
        hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>",
    ), row=1, col=1)

    # ✅ add_shape instead of add_vline (subplot compatible)
    fig.add_shape(
        type="line", x0=avg, x1=avg, y0=0, y1=1,
        xref="x1", yref="paper",
        line=dict(color=GOLD, dash="dot", width=1.5),
    )
    fig.add_annotation(
        x=avg, y=1, xref="x1", yref="paper",
        text=f"Avg {avg:.2f}", showarrow=False,
        font=dict(color=GOLD, size=11), yanchor="bottom",
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(scores))), y=scores,
        mode="markers", name="Score",
        marker=dict(color=colors, size=8, opacity=0.85),
        customdata=names,
        hovertemplate="%{customdata}<br>Score: %{y:.2f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=list(range(len(scores))), y=rolling,
        mode="lines", name="Rolling Avg",
        line=dict(color=GOLD, width=2.5, dash="dot"),
        hovertemplate="Rolling avg: %{y:.2f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font=dict(color=WHITE), height=360,
        legend=dict(bgcolor=BG_CARD, bordercolor=GREY),
        margin=dict(l=30, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=GREY, linecolor=GREY)
    fig.update_yaxes(gridcolor=GREY, linecolor=GREY, range=[0, 11])
    return fig


def gallery_bar(labels: list, avg_vals: list) -> go.Figure:
    colors = [TEAL if v >= 6.5 else GOLD if v >= 5.0 else RED for v in avg_vals]
    fig = go.Figure(go.Bar(
        x=labels, y=avg_vals,
        marker=dict(color=colors, opacity=0.9),
        text=[f"{v:.1f}" for v in avg_vals],
        textposition="outside",
        textfont=dict(color=WHITE, size=12),
    ))
    fig.update_layout(
        title=dict(text="📊 Gallery Average by Dimension", x=0.5,
                   font=dict(size=15, color=WHITE)),
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font=dict(color=WHITE),
        xaxis=dict(gridcolor=GREY),
        yaxis=dict(gridcolor=GREY, range=[0, 11], title="Score /10"),
        margin=dict(l=30, r=20, t=60, b=40),
        height=320,
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nima():
    from nima_scorer import load_model
    return load_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📸 Photo Scorer")
    st.markdown(f"<p style='color:{SUBTEXT};font-size:0.85rem;'>Powered by NIMA · MobileNet</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "🖼️ Single Photo",
        "📊 Batch Analyzer",
        "🏆 Gallery Stats",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style='color:{SUBTEXT};font-size:0.8rem;line-height:1.8'>
    <b style='color:{WHITE}'>Model</b><br>NIMA MobileNet<br><br>
    <b style='color:{WHITE}'>Dataset</b><br>AVA (255k photos)<br><br>
    <b style='color:{WHITE}'>Features</b><br>6 CV dimensions<br><br>
    <b style='color:{WHITE}'>Auto-compress</b><br>✅ Images > 1200px
    </div>
    """, unsafe_allow_html=True)


# ── Model preload ─────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading NIMA model..."):
    model = load_nima()


# ╔══════════════════════════════════════════════════════╗
# ║  PAGE 1 — Single Photo                              ║
# ╚══════════════════════════════════════════════════════╝
if page == "🖼️ Single Photo":
    st.markdown("## 🖼️ Single Photo Scorer")
    st.markdown(f"<p style='color:{SUBTEXT}'>Upload a photo · AI scores aesthetic quality in seconds</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader(
        "Drop your photo here",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supports JPG, PNG, WebP. Large files auto-compressed.",
    )

    if uploaded:
        from nima_scorer import score_image
        from cv_features import extract_features
        from explainer   import generate_tips, overall_verdict

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Analysing..."):
            nima    = score_image(model, tmp_path)
            cv_feat = extract_features(tmp_path)

            # ✅ Pop comp_style BEFORE passing to explainer
            comp_style = cv_feat.pop("comp_style", "Unknown")

            tips             = generate_tips(cv_feat)
            verdict, v_color = overall_verdict(nima["score"], cv_feat)

        col_img, col_right = st.columns([1.1, 0.9], gap="large")

        with col_img:
            st.image(nima["preview"], caption=f"📁 {uploaded.name}",
                     use_container_width=True)

        with col_right:
            render_score_badge(nima)

            st.markdown(f"""
            <div style="background:{BG_CARD};border-radius:10px;padding:14px;
                        margin:10px 0;border-left:4px solid {v_color};">
                <span style="font-size:1rem">{verdict}</span>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Sharpness",   f"{cv_feat['sharpness']}/10")
            m2.metric("Composition", f"{cv_feat['composition']}/10")
            m3.metric("Brightness",  f"{cv_feat['brightness']}/10")

            # ✅ Composition style label
            st.markdown(f"""
            <div style="background:{BG_CARD};border-radius:8px;padding:10px 14px;
                        margin-top:8px;border-left:3px solid {TEAL};">
                <span style="color:{SUBTEXT};font-size:0.8rem;">Detected Style</span><br>
                <span style="font-size:1rem;font-weight:700;">{comp_style}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        col_radar, col_dist = st.columns([1.2, 0.8], gap="large")
        with col_radar:
            st.plotly_chart(
                radar_chart_enhanced(cv_feat, nima["score"], nima["color"]),
                use_container_width=True,
            )
        with col_dist:
            st.plotly_chart(
                distribution_chart(nima["distribution"]),
                use_container_width=True,
            )

        st.markdown("---")
        render_cv_dimensions(cv_feat)

        st.markdown("---")
        st.markdown("## 💡 AI Suggestions")
        ai_tips = generate_suggestions(nima["score"], cv_feat)
        for tip in ai_tips:
            cls = "tip-ok" if tip.startswith("✅") else "tip-warn"
            st.markdown(f'<div class="{cls}">{tip}</div>', unsafe_allow_html=True)

        if tips:
            st.markdown("### 🔧 Technical Tips")
            for tip in tips:
                cls = "tip-ok" if tip.startswith("✅") else "tip-warn"
                st.markdown(f'<div class="{cls}">{tip}</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════╗
# ║  PAGE 2 — Batch Analyzer                            ║
# ╚══════════════════════════════════════════════════════╝
elif page == "📊 Batch Analyzer":
    st.markdown("## 📊 Batch Analyzer")
    st.markdown(f"<p style='color:{SUBTEXT}'>Upload multiple photos · AI ranks them by aesthetic score</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    uploads = st.file_uploader(
        "Drop your photos here (up to 20)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploads:
        st.markdown(f"**{len(uploads)} photo(s) selected**")
        if st.button(f"🚀 Rank {len(uploads)} Photos"):
            from nima_scorer import score_image

            results  = []
            progress = st.progress(0, text="Analysing photos...")

            for i, f in enumerate(uploads):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                r         = score_image(model, tmp_path)
                r["name"] = f.name
                r["path"] = tmp_path
                results.append(r)
                progress.progress((i + 1) / len(uploads),
                                  text=f"Analysed {i+1}/{len(uploads)}: {f.name}")

            results = sorted(results, key=lambda x: x["score"], reverse=True)
            progress.empty()

            best = results[0]
            _, best_color, best_pct = get_score_meta(best["score"])
            st.markdown(f"""
            <div style="background:{BG_CARD};border:2px solid {GOLD};border-radius:12px;
                        padding:20px;margin:16px 0;text-align:center">
                🥇 <b>Best Photo:</b> {best['name']}
                <span style="color:{GOLD};font-size:1.4rem;font-weight:700;margin-left:12px">
                    {best['score']}/10
                </span>
                <span style="margin-left:8px">{best['rating']}</span>
                <div style="color:{SUBTEXT};font-size:0.8rem;margin-top:6px">
                    Better than {best_pct:.0f}% of AVA dataset photos
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_chart, col_prev = st.columns([1.2, 0.8], gap="large")
            with col_chart:
                st.plotly_chart(batch_chart(results), use_container_width=True)
            with col_prev:
                st.image(best["preview"], caption="🥇 Best Photo",
                         use_container_width=True)

            if len(results) >= 3:
                st.markdown("---")
                st.markdown("### 📈 Score Analytics")
                st.plotly_chart(batch_trend_chart(results), use_container_width=True)

                scores = [r["score"] for r in results]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📸 Total",   len(scores))
                c2.metric("⭐ Average", f"{sum(scores)/len(scores):.2f}")
                c3.metric("🏆 Best",    f"{max(scores):.2f}")
                c4.metric("📉 Lowest",  f"{min(scores):.2f}")

            st.markdown("### 📋 Full Results")
            df = pd.DataFrame([{
                "Rank":   i + 1,
                "Photo":  r["name"],
                "Score":  r["score"],
                "Rating": r["rating"],
            } for i, r in enumerate(results)])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════╗
# ║  PAGE 3 — Gallery Stats                             ║
# ╚══════════════════════════════════════════════════════╝
elif page == "🏆 Gallery Stats":
    st.markdown("## 🏆 Gallery Stats")
    st.markdown(f"<p style='color:{SUBTEXT}'>Upload your full collection · Discover your shooting strengths</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    uploads = st.file_uploader(
        "Upload your gallery",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploads:
        st.markdown(f"**{len(uploads)} photo(s) selected**")
        if st.button(f"📊 Analyse Gallery ({len(uploads)} photos)"):
            from nima_scorer import score_image
            from cv_features import extract_features

            records  = []
            progress = st.progress(0, text="Analysing gallery...")
            for i, f in enumerate(uploads):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                nima    = score_image(model, tmp_path)
                cv_feat = extract_features(tmp_path)

                # ✅ Pop comp_style to avoid non-numeric issues
                cv_feat.pop("comp_style", None)

                records.append({"name": f.name, "nima": nima["score"], **cv_feat})
                progress.progress((i + 1) / len(uploads),
                                  text=f"Analysed {i+1}/{len(uploads)}")

            progress.empty()
            df = pd.DataFrame(records)

            st.markdown("### 📊 Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📸 Photos",    len(df))
            c2.metric("⭐ Avg Score", f"{df['nima'].mean():.2f}")
            c3.metric("🌟 Best",      f"{df['nima'].max():.2f}")
            c4.metric("📉 Worst",     f"{df['nima'].min():.2f}")

            feat_cols = ["brightness", "sharpness", "composition",
                         "color", "noise", "contrast"]
            avg_vals  = [df[c].mean() for c in feat_cols]
            labels    = ["Brightness", "Sharpness", "Composition",
                         "Color", "Noise", "Contrast"]

            st.plotly_chart(gallery_bar(labels, avg_vals), use_container_width=True)

            best_i  = int(np.argmax(avg_vals))
            worst_i = int(np.argmin(avg_vals))
            st.success(f"✅ Strongest area: **{labels[best_i]}** ({avg_vals[best_i]:.1f}/10)")
            st.warning(f"⚠️ Most room to improve: **{labels[worst_i]}** ({avg_vals[worst_i]:.1f}/10)")

            st.markdown("### 📈 Score Distribution")
            import plotly.express as px
            fig_hist = px.histogram(
                df, x="nima", nbins=10,
                labels={"nima": "NIMA Score"},
                color_discrete_sequence=[RED],
            )
            fig_hist.update_layout(
                paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
                font=dict(color=WHITE),
                xaxis=dict(gridcolor=GREY),
                yaxis=dict(gridcolor=GREY),
                margin=dict(l=40, r=20, t=30, b=40),
                height=280,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("### 📋 All Photos")
            st.dataframe(
                df[["name", "nima"] + feat_cols].rename(
                    columns={"name": "Photo", "nima": "Score"}
                ),
                use_container_width=True, hide_index=True,
            )