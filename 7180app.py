"""
Mind Mirror — Depression Risk Screening Tool
CS 7180 Special Topics in AI — Spring 2026
Andrew Min & Mengyi Ma

A Streamlit app with three tabs:
  1. Screener — interactive depression risk prediction
  2. Global Insights — data visualizations
  3. Model Card — technical methodology
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mind Mirror",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,wght@0,400;0,500;0,700;1,400&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #1a1a2e;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    padding: 0 0 8px 0;
    border-bottom: 2px solid #e8e4df;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    padding: 10px 24px;
    border-radius: 8px 8px 0 0;
    color: #555;
}
.stTabs [aria-selected="true"] {
    color: #1a1a2e !important;
    border-bottom: 3px solid #5e4fa2 !important;
    font-weight: 700;
}

/* Result cards */
.risk-card {
    border-radius: 16px;
    padding: 32px;
    margin: 16px 0;
    text-align: center;
}
.risk-card.at-risk {
    background: linear-gradient(135deg, #fff0f0, #ffe6e6);
    border: 2px solid #e57373;
}
.risk-card.low-risk {
    background: linear-gradient(135deg, #f0f9f0, #e6f5e6);
    border: 2px solid #81c784;
}
.risk-card h2 {
    margin: 0 0 8px 0;
    font-size: 2rem;
}
.risk-card .score {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'DM Serif Display', serif;
}
.risk-card.at-risk .score { color: #c62828; }
.risk-card.low-risk .score { color: #2e7d32; }

/* Metric card */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}
.metric-box {
    flex: 1;
    background: #faf8f5;
    border: 1px solid #e8e4df;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-box .label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.metric-box .value {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'DM Serif Display', serif;
    color: #1a1a2e;
}

/* Driver chips */
.driver {
    background: #f5f3ff;
    border: 1px solid #d1c4e9;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #4a148c;
}

/* What-If Explorer */
.whatif-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.whatif-header h4 {
    margin: 0;
    font-size: 1.15rem;
}
.whatif-subtext {
    font-size: 0.88rem;
    color: #777;
    margin-bottom: 12px;
}
.whatif-result {
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0 0 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.whatif-result.at-risk {
    background: linear-gradient(135deg, #fff0f0, #ffe6e6);
    border: 1.5px solid #e57373;
}
.whatif-result.low-risk {
    background: linear-gradient(135deg, #f0f9f0, #e6f5e6);
    border: 1.5px solid #81c784;
}
.whatif-result .wi-label {
    font-size: 0.82rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.whatif-result .wi-score {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'DM Serif Display', serif;
}
.whatif-result.at-risk .wi-score { color: #c62828; }
.whatif-result.low-risk .wi-score { color: #2e7d32; }
.whatif-delta {
    font-size: 0.95rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
}
.whatif-delta.up { background: #ffebee; color: #c62828; }
.whatif-delta.down { background: #e8f5e9; color: #2e7d32; }
.whatif-delta.flat { background: #f5f5f5; color: #888; }

/* Recommendation cards */
.rec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin: 14px 0;
}
.rec-card {
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e8e4df;
    transition: box-shadow 0.2s ease;
}
.rec-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.rec-card .rec-icon {
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: #5e4fa2;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.rec-card .rec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    color: #1a1a2e;
    margin-bottom: 6px;
}
.rec-card .rec-body {
    font-size: 0.85rem;
    color: #555;
    line-height: 1.5;
}
.rec-card.emotional { background: linear-gradient(135deg, #fef9f0, #fdf3e3); border-color: #f0d9a8; }
.rec-card.stability { background: linear-gradient(135deg, #f0f4ff, #e8eeff); border-color: #b8c9f0; }
.rec-card.social { background: linear-gradient(135deg, #f5f0ff, #ede4ff); border-color: #c9b8f0; }
.rec-card.structure { background: linear-gradient(135deg, #f0faf5, #e4f5ec); border-color: #a8dfc0; }
.rec-card.general { background: linear-gradient(135deg, #faf8f5, #f3f0eb); border-color: #e0dbd3; }

/* Header */
.app-header {
    text-align: center;
    padding: 20px 0 10px 0;
}
.app-header h1 {
    font-size: 2.6rem;
    margin: 0;
    background: linear-gradient(135deg, #5e4fa2, #7e57c2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header .subtitle {
    font-size: 1rem;
    color: #777;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Assets ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

@st.cache_resource
def load_model():
    return joblib.load(DATA_DIR / "mental_health_model.pkl")

@st.cache_resource
def load_feature_cols():
    return joblib.load(DATA_DIR / "feature_cols.pkl")

@st.cache_data
def load_json(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)

@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA_DIR / name)


model = load_model()
feature_cols = load_feature_cols()
feat_imp = load_json("feature_importance.json")
country_risk = load_csv("country_risk.csv")
age_stats = load_csv("age_stats.csv")
tipi_profiles = load_csv("tipi_profiles.csv")
dataset_avg = load_json("dataset_avg.json")


# ─── TIPI Labels ────────────────────────────────────────────────────────────────

TIPI_LABELS = {
    "TIPI1":  "Extraverted, enthusiastic",
    "TIPI2":  "Critical, quarrelsome",
    "TIPI3":  "Dependable, self-disciplined",
    "TIPI4":  "Anxious, easily upset",
    "TIPI5":  "Open to new experiences, complex",
    "TIPI6":  "Reserved, quiet",
    "TIPI7":  "Sympathetic, warm",
    "TIPI8":  "Disorganized, careless",
    "TIPI9":  "Calm, emotionally stable",
    "TIPI10": "Conventional, uncreative",
}

GENDER_MAP = {0: "Missed", 1: "Male", 2: "Female", 3: "Other"}
GENDER_INV = {v: k for k, v in GENDER_MAP.items()}

EDUCATION_MAP = {
    0: "Missed", 1: "Less than high school", 2: "High school",
    3: "University degree", 4: "Graduate degree"
}
EDUCATION_INV = {v: k for k, v in EDUCATION_MAP.items()}

URBAN_MAP = {0: "Missed", 1: "Rural", 2: "Suburban", 3: "Urban"}
URBAN_INV = {v: k for k, v in URBAN_MAP.items()}

RELIGION_MAP = {
    0: "Missed", 1: "Agnostic", 2: "Atheist", 3: "Buddhist", 4: "Christian (Catholic)",
    5: "Christian (Mormon)", 6: "Christian (Protestant)", 7: "Christian (Other)",
    8: "Hindu", 9: "Jewish", 10: "Muslim", 11: "Sikh", 12: "Other"
}
RELIGION_INV = {v: k for k, v in RELIGION_MAP.items()}

ORIENTATION_MAP = {
    0: "Missed", 1: "Heterosexual", 2: "Bisexual",
    3: "Homosexual", 4: "Asexual", 5: "Other"
}
ORIENTATION_INV = {v: k for k, v in ORIENTATION_MAP.items()}

RACE_MAP = {
    10: "Asian", 20: "Arab", 30: "Black", 40: "Indigenous Australian",
    50: "Native American", 60: "White", 70: "Other"
}
RACE_INV = {v: k for k, v in RACE_MAP.items()}

MARRIED_MAP = {0: "Missed", 1: "Never married", 2: "Currently married", 3: "Previously married"}
MARRIED_INV = {v: k for k, v in MARRIED_MAP.items()}


# ─── Helper: per-user feature contribution ──────────────────────────────────────

# Dataset-average values for each feature (approximate medians/modes from training data)
FEATURE_BASELINES = {
    "TIPI1": 4, "TIPI2": 4, "TIPI3": 5, "TIPI4": 4, "TIPI5": 5,
    "TIPI6": 4, "TIPI7": 5, "TIPI8": 4, "TIPI9": 4, "TIPI10": 3,
    "education": 3, "urban": 2, "gender": 1, "age": 25,
    "religion": 1, "orientation": 1, "race": 60, "married": 1, "familysize": 3,
}


def compute_user_contributions(input_dict, model_obj, feat_cols):
    """
    Approximate per-feature contribution by measuring how much the risk
    probability changes when each feature is replaced with its baseline value.
    A larger positive delta means the user's value for that feature is pushing
    risk *higher* than baseline.
    """
    base_df = pd.DataFrame([input_dict])[feat_cols]
    base_prob = model_obj.predict_proba(base_df)[0][1]

    contributions = {}
    for feat in feat_cols:
        if feat in FEATURE_BASELINES:
            modified = dict(input_dict)
            modified[feat] = FEATURE_BASELINES[feat]
            mod_df = pd.DataFrame([modified])[feat_cols]
            mod_prob = model_obj.predict_proba(mod_df)[0][1]
            # positive = this feature pushes risk up vs baseline
            contributions[feat] = base_prob - mod_prob
        else:
            contributions[feat] = 0.0
    return contributions


# ─── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>Mind Mirror</h1>
    <div class="subtitle">Depression Risk Screening from Personality & Demographics</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Screener", "Global Insights"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCREENER
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Take the Screening")
    st.markdown("Rate each personality statement from **1** (disagree strongly) to **7** (agree strongly), then fill in your demographic information.")

    # --- Personality Sliders (2 columns) ---
    st.markdown("#### Personality Traits (TIPI)")
    tipi_keys = list(TIPI_LABELS.keys())
    col_left, col_right = st.columns(2)
    tipi_values = {}

    for i, key in enumerate(tipi_keys):
        col = col_left if i % 2 == 0 else col_right
        with col:
            tipi_values[key] = st.slider(
                f"{TIPI_LABELS[key]}",
                min_value=1, max_value=7, value=4, key=key
            )

    # --- Demographic Inputs ---
    st.markdown("#### Demographics")
    d1, d2, d3 = st.columns(3)

    with d1:
        age = st.number_input("Age", min_value=13, max_value=100, value=25)
        gender = st.selectbox("Gender", list(GENDER_INV.keys())[1:])
        education = st.selectbox("Education", list(EDUCATION_INV.keys())[1:])

    with d2:
        urban = st.selectbox("Living area", list(URBAN_INV.keys())[1:])
        religion = st.selectbox("Religion", list(RELIGION_INV.keys())[1:])
        orientation = st.selectbox("Orientation", list(ORIENTATION_INV.keys())[1:])

    with d3:
        race = st.selectbox("Race", list(RACE_INV.keys()))
        married = st.selectbox("Marital status", list(MARRIED_INV.keys())[1:])
        familysize = st.number_input("Family size", min_value=0, max_value=20, value=3)

    # --- Predict: save to session_state so results survive slider reruns ---
    if st.button("Get My Results", type="primary", use_container_width=True):
        input_dict = {
            **tipi_values,
            "education": EDUCATION_INV[education],
            "urban": URBAN_INV[urban],
            "gender": GENDER_INV[gender],
            "age": age,
            "religion": RELIGION_INV[religion],
            "orientation": ORIENTATION_INV[orientation],
            "race": RACE_INV[race],
            "married": MARRIED_INV[married],
            "familysize": familysize,
        }

        input_df = pd.DataFrame([input_dict])[feature_cols]
        prob = model.predict_proba(input_df)[0]
        pred = int(prob[1] >= 0.5)
        risk_pct = round(prob[1] * 100, 1)

        st.session_state["results"] = {
            "input_dict": input_dict,
            "prob": prob.tolist(),
            "pred": pred,
            "risk_pct": risk_pct,
        }

    # ─── Results display (persists across reruns) ────────────────────────────
    if "results" in st.session_state:
        res = st.session_state["results"]
        input_dict = res["input_dict"]
        prob = res["prob"]
        pred = res["pred"]
        risk_pct = res["risk_pct"]

        # Result card
        if pred == 1:
            st.markdown(f"""
            <div class="risk-card at-risk">
                <h2>At-Risk</h2>
                <div class="score">{risk_pct}%</div>
                <p style="color:#c62828; font-size:0.95rem;">risk score</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                "Based on your personality profile and demographics, the model estimates an **elevated risk** "
                "for depression. This means your responses are similar to individuals in our dataset who scored "
                "above the severe threshold (>=21) on the DASS-42 depression subscale."
            )
        else:
            st.markdown(f"""
            <div class="risk-card low-risk">
                <h2>Low Risk</h2>
                <div class="score">{risk_pct}%</div>
                <p style="color:#2e7d32; font-size:0.95rem;">risk score</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                "Based on your personality profile and demographics, the model estimates a **lower risk** "
                "for depression. Your responses are more similar to individuals who scored below the severe "
                "threshold on the DASS-42 depression subscale."
            )

        # Probability bar
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=["Prediction"], x=[prob[0] * 100], name="Low Risk",
            orientation="h", marker_color="#81c784", text=f"{prob[0]*100:.1f}%", textposition="inside"
        ))
        fig_bar.add_trace(go.Bar(
            y=["Prediction"], x=[prob[1] * 100], name="At-Risk",
            orientation="h", marker_color="#e57373", text=f"{prob[1]*100:.1f}%", textposition="inside"
        ))
        fig_bar.update_layout(
            barmode="stack", height=80, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False, range=[0, 100]),
            yaxis=dict(visible=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Top 3 Drivers (per-user, not static global importance) ---
        st.markdown("**Top 3 Factors Driving This Prediction:**")
        contributions = compute_user_contributions(input_dict, model, feature_cols)
        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = sorted_contribs[:3]

        for feat_name, contrib in top3:
            val = input_dict[feat_name]
            direction = "increases" if contrib > 0 else "decreases"
            contrib_display = f"{abs(contrib) * 100:.1f}pp"
            if feat_name.startswith("TIPI"):
                label = TIPI_LABELS[feat_name]
                st.markdown(
                    f'<div class="driver"><strong>{label}</strong> — '
                    f'Your score of <strong>{val}</strong> {direction} risk '
                    f'({contrib_display})</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="driver"><strong>{feat_name.capitalize()}</strong> = '
                    f'<strong>{val}</strong> {direction} risk '
                    f'({contrib_display})</div>',
                    unsafe_allow_html=True
                )

        # ─── What-If Explorer ────────────────────────────────────────────
        # Lives OUTSIDE the button block so slider changes rerun the page
        # while session_state keeps the original results visible.
        st.divider()
        st.markdown("""
        <div class="whatif-header">
            <h4>What-If Explorer</h4>
        </div>
        <div class="whatif-subtext">
            Adjust any of the 10 personality traits below to see how changes would shift your predicted risk score.
        </div>
        """, unsafe_allow_html=True)

        # All 10 TIPI features sorted by importance, shown in 2 rows of 5
        tipi_feats_sorted = sorted(
            [(k, v) for k, v in feat_imp.items() if k.startswith("TIPI")],
            key=lambda x: x[1], reverse=True
        )

        whatif_dict = dict(input_dict)

        # Row 1: traits 1–5
        wi_row1 = st.columns(5)
        for idx, (feat_key, imp_val) in enumerate(tipi_feats_sorted[:5]):
            with wi_row1[idx]:
                short_label = TIPI_LABELS[feat_key].split(",")[0]
                whatif_dict[feat_key] = st.slider(
                    f"{short_label}",
                    min_value=1, max_value=7,
                    value=int(input_dict[feat_key]),
                    key=f"wi_{feat_key}",
                    help=f"Importance: {imp_val:.1%}"
                )

        # Row 2: traits 6–10
        wi_row2 = st.columns(5)
        for idx, (feat_key, imp_val) in enumerate(tipi_feats_sorted[5:]):
            with wi_row2[idx]:
                short_label = TIPI_LABELS[feat_key].split(",")[0]
                whatif_dict[feat_key] = st.slider(
                    f"{short_label}",
                    min_value=1, max_value=7,
                    value=int(input_dict[feat_key]),
                    key=f"wi_{feat_key}",
                    help=f"Importance: {imp_val:.1%}"
                )

        # Compute what-if prediction
        whatif_df = pd.DataFrame([whatif_dict])[feature_cols]
        wi_prob = model.predict_proba(whatif_df)[0]
        wi_pred = int(wi_prob[1] >= 0.5)
        wi_risk_pct = round(wi_prob[1] * 100, 1)
        delta = round(wi_risk_pct - risk_pct, 1)

        if delta > 0.05:
            delta_cls = "up"
            delta_str = f"+{delta}pp"
        elif delta < -0.05:
            delta_cls = "down"
            delta_str = f"{delta}pp"
        else:
            delta_cls = "flat"
            delta_str = "no change"

        risk_cls = "at-risk" if wi_pred == 1 else "low-risk"
        status_label = "At-Risk" if wi_pred == 1 else "Low Risk"

        st.markdown(f"""
        <div class="whatif-result {risk_cls}">
            <div>
                <div class="wi-label">What-If Risk Score</div>
                <div class="wi-score">{wi_risk_pct}%</div>
                <div style="font-size:0.85rem; color:#555; margin-top:2px;">{status_label}</div>
            </div>
            <div style="text-align:right;">
                <div class="wi-label">Change from Original</div>
                <div class="whatif-delta {delta_cls}">{delta_str}</div>
                <div style="font-size:0.8rem; color:#aaa; margin-top:4px;">Original: {risk_pct}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─── Personalized Recommendations ────────────────────────────────
        st.divider()
        st.markdown("""
        <div class="whatif-header">
            <h4>Personalized Insights & Resources</h4>
        </div>
        <div class="whatif-subtext">
            Based on your profile, here are tailored areas for reflection and support.
        </div>
        """, unsafe_allow_html=True)

        recs = []

        if input_dict.get("TIPI4", 4) >= 5:
            recs.append({
                "tag": "Emotional Regulation", "cls": "emotional",
                "title": "Managing Emotional Reactivity",
                "body": (
                    "Your score on <em>Anxious, easily upset</em> is elevated — this trait is the "
                    "strongest predictor of risk in the model. Evidence-based approaches like "
                    "Cognitive Behavioral Therapy (CBT) and mindfulness-based stress reduction "
                    "can help build emotional regulation skills."
                ),
            })

        if input_dict.get("TIPI9", 4) <= 3:
            recs.append({
                "tag": "Resilience", "cls": "stability",
                "title": "Building Emotional Stability",
                "body": (
                    "A lower score on <em>Calm, emotionally stable</em> is strongly associated "
                    "with higher risk. Regular physical activity, structured sleep routines, and "
                    "breathing exercises have shown measurable improvements in emotional resilience "
                    "across clinical studies."
                ),
            })

        if input_dict.get("TIPI8", 4) >= 5:
            recs.append({
                "tag": "Structure", "cls": "structure",
                "title": "Creating Structure & Routine",
                "body": (
                    "Higher scores on <em>Disorganized, careless</em> can amplify stress. "
                    "Small habits — task lists, time-blocking, and consistent daily routines — "
                    "can reduce cognitive load and create a sense of control that supports "
                    "mental well-being."
                ),
            })

        if input_dict.get("TIPI6", 4) >= 5:
            recs.append({
                "tag": "Connection", "cls": "social",
                "title": "Social Connection at Your Pace",
                "body": (
                    "Being <em>Reserved, quiet</em> is not a risk factor on its own, but "
                    "social isolation can compound other stressors. Consider low-pressure "
                    "connection — online communities, one-on-one conversations, or "
                    "shared-interest groups that feel comfortable."
                ),
            })

        if input_dict.get("TIPI1", 4) <= 2:
            recs.append({
                "tag": "Social", "cls": "social",
                "title": "Introversion & Well-Being",
                "body": (
                    "A low extraversion score may mean you recharge in solitude — which is healthy. "
                    "The key is ensuring you still have <em>meaningful</em> social contact. Even "
                    "one trusted relationship is a significant protective factor against depression."
                ),
            })

        if input_dict.get("age", 25) <= 24:
            recs.append({
                "tag": "Youth", "cls": "general",
                "title": "Young Adult Support",
                "body": (
                    "The 18-24 age group shows the highest risk rate in the dataset. "
                    "Campus counseling centers, peer support programs, and crisis text lines "
                    "(text HOME to 741741) are free, confidential resources designed for "
                    "this life stage."
                ),
            })

        if pred == 1:
            recs.append({
                "tag": "Support", "cls": "general",
                "title": "Professional Support",
                "body": (
                    "Your screening suggests elevated risk. Speaking with a licensed therapist "
                    "or counselor is the single most effective next step. Many offer sliding-scale "
                    "fees or accept insurance. The SAMHSA helpline "
                    "<strong>(1-800-662-4357)</strong> provides free, 24/7 referrals."
                ),
            })
        else:
            recs.append({
                "tag": "Wellness", "cls": "general",
                "title": "Maintaining Well-Being",
                "body": (
                    "Your screening suggests lower risk — that's encouraging. Continuing "
                    "protective habits like regular exercise, sufficient sleep, social connection, "
                    "and stress management will help maintain your well-being over time."
                ),
            })

        rec_html = '<div class="rec-grid">'
        for rec in recs:
            rec_html += f"""
            <div class="rec-card {rec['cls']}">
                <div class="rec-icon">{rec['tag']}</div>
                <div class="rec-title">{rec['title']}</div>
                <div class="rec-body">{rec['body']}</div>
            </div>
            """
        rec_html += '</div>'
        st.markdown(rec_html, unsafe_allow_html=True)

        # Disclaimer
        st.markdown(
            "<div style='background:#faf8f5; border-left:4px solid #b0bec5; border-radius:0 8px 8px 0; "
            "padding:16px 20px; margin-top:24px; font-size:0.85rem; color:#666; line-height:1.5;'>"
            "<strong>Disclaimer:</strong> Mind Mirror is an educational screening tool trained on survey "
            "data and is not a substitute for professional clinical diagnosis."
            "</div>",
            unsafe_allow_html=True,
        )



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GLOBAL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Global Insights")
    st.markdown("Explore patterns in the DASS-42 dataset across countries, age groups, and personality profiles.")

    # --- Country Risk ---
    st.markdown("#### Depression Risk Rate by Country")
    st.caption("Top 20 countries with >=50 survey responses")

    avg_risk = dataset_avg["overall_avg_risk"]
    country_risk_sorted = country_risk.sort_values("risk_rate", ascending=True)
    colors = ["#e57373" if r >= avg_risk else "#81c784" for r in country_risk_sorted["risk_rate"]]

    fig_country = go.Figure()
    fig_country.add_trace(go.Bar(
        y=country_risk_sorted["country"],
        x=country_risk_sorted["risk_rate"],
        orientation="h",
        marker_color=colors,
        text=[f"{r}%" for r in country_risk_sorted["risk_rate"]],
        textposition="outside",
    ))
    fig_country.add_vline(x=avg_risk, line_dash="dash", line_color="#5e4fa2",
                          annotation_text=f"Dataset avg: {avg_risk}%", annotation_position="top right")
    fig_country.update_layout(
        height=550, margin=dict(l=0, r=40, t=20, b=20),
        xaxis_title="At-Risk Rate (%)", yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"),
        xaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig_country, use_container_width=True)

    st.divider()

    # --- Age Group ---
    st.markdown("#### Risk Rate by Age Group")
    col_age1, col_age2 = st.columns(2)

    with col_age1:
        fig_age = go.Figure()
        age_colors = ["#7e57c2" if ag == "18-24" else "#b0bec5" for ag in age_stats["age_group"]]
        fig_age.add_trace(go.Bar(
            x=age_stats["age_group"], y=age_stats["risk_rate"],
            marker_color=age_colors,
            text=[f"{r}%" for r in age_stats["risk_rate"]],
            textposition="outside",
        ))
        fig_age.update_layout(
            height=350, margin=dict(l=0, r=0, t=30, b=0),
            title="At-Risk Rate by Age Bracket",
            yaxis_title="Risk Rate (%)", xaxis_title="Age Group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"), yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col_age2:
        fig_dep = go.Figure()
        fig_dep.add_trace(go.Bar(
            x=age_stats["age_group"], y=age_stats["mean_dep"],
            marker_color=age_colors,
            text=[f"{d}" for d in age_stats["mean_dep"]],
            textposition="outside",
        ))
        fig_dep.update_layout(
            height=350, margin=dict(l=0, r=0, t=30, b=0),
            title="Mean Depression Score by Age Bracket",
            yaxis_title="Mean DASS Depression Score", xaxis_title="Age Group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"), yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_dep, use_container_width=True)

    st.markdown(
        "> **Key finding:** The **18-24** age group shows the highest risk rate in the dataset, "
        "consistent with clinical research on young adult mental health vulnerability."
    )

    st.divider()

    # --- TIPI Profiles ---
    st.markdown("#### Personality Profiles: At-Risk vs. Low-Risk")

    trait_labels = [TIPI_LABELS[t].split(",")[0] for t in tipi_profiles["trait"]]

    fig_tipi = go.Figure()
    fig_tipi.add_trace(go.Bar(
        name="At-Risk", x=trait_labels, y=tipi_profiles["at_risk"],
        marker_color="#e57373"
    ))
    fig_tipi.add_trace(go.Bar(
        name="Low-Risk", x=trait_labels, y=tipi_profiles["low_risk"],
        marker_color="#81c784"
    ))
    fig_tipi.update_layout(
        barmode="group", height=400, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Mean TIPI Score (1-7)", xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"), yaxis=dict(gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_tipi, use_container_width=True)

    # Difference chart
    fig_diff = go.Figure()
    diff_colors = ["#e57373" if d > 0 else "#81c784" for d in tipi_profiles["diff"]]
    fig_diff.add_trace(go.Bar(
        x=trait_labels, y=tipi_profiles["diff"],
        marker_color=diff_colors,
        text=[f"{d:+.2f}" for d in tipi_profiles["diff"]],
        textposition="outside",
    ))
    fig_diff.update_layout(
        height=300, margin=dict(l=0, r=0, t=30, b=0),
        title="Score Difference (At-Risk minus Low-Risk)",
        yaxis_title="Difference", xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"), yaxis=dict(gridcolor="#eee"),
    )
    fig_diff.add_hline(y=0, line_color="#555", line_width=1)
    st.plotly_chart(fig_diff, use_container_width=True)

    st.markdown(
        "> At-risk individuals score **significantly higher** on *Anxious, easily upset* (TIPI4) "
        "and **significantly lower** on *Calm, emotionally stable* (TIPI9) — the two most important "
        "features in the model."
    )


# ─── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "Mind Mirror · CS 7180 Special Topics in AI · Andrew Min & Mengyi Ma · Spring 2026"
    "</div>",
    unsafe_allow_html=True,
)
