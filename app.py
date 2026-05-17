import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="NeuroGuard — Stroke Risk AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0A0F1E;
    color: #E8EDF8;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(56,100,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(0,210,180,0.08) 0%, transparent 60%),
        #0A0F1E;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1300px !important; }

.ng-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.5rem 2.5rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; backdrop-filter: blur(20px);
    margin-bottom: 2.5rem;
}
.ng-logo { display: flex; align-items: center; gap: 14px; }
.ng-logo-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #3864FF, #00D2B4);
    border-radius: 14px; display: flex; align-items: center;
    justify-content: center; font-size: 24px;
    box-shadow: 0 8px 24px rgba(56,100,255,0.35);
}
.ng-logo-text {
    font-family: 'DM Serif Display', serif; font-size: 1.6rem;
    background: linear-gradient(135deg, #fff 30%, #00D2B4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.ng-logo-sub {
    font-size: 0.72rem; color: rgba(255,255,255,0.4);
    letter-spacing: 2px; text-transform: uppercase; margin-top: -4px;
}
.ng-badge {
    padding: 6px 16px; border-radius: 50px;
    background: rgba(0,210,180,0.1); border: 1px solid rgba(0,210,180,0.3);
    color: #00D2B4; font-size: 0.75rem; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase;
}
.ng-section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: #3864FF; margin-bottom: 1rem;
}
.ng-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 2rem;
}
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #E8EDF8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
label, [data-testid="stWidgetLabel"] p {
    color: rgba(255,255,255,0.6) !important; font-size: 0.78rem !important;
    font-weight: 500 !important; letter-spacing: 0.5px !important;
    text-transform: uppercase !important; margin-bottom: 4px !important;
}
.stButton > button {
    width: 100% !important; padding: 16px 32px !important;
    background: linear-gradient(135deg, #3864FF, #5B8BFF) !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 600 !important;
    box-shadow: 0 8px 32px rgba(56,100,255,0.35) !important;
    margin-top: 1.2rem !important;
}
.result-card {
    border-radius: 18px; padding: 2rem; text-align: center;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.result-high   { background: rgba(255,71,87,0.08);  border: 1px solid rgba(255,71,87,0.3); }
.result-medium { background: rgba(255,165,2,0.08);  border: 1px solid rgba(255,165,2,0.3); }
.result-low    { background: rgba(0,210,180,0.08);  border: 1px solid rgba(0,210,180,0.3); }
.result-label  {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 0.5rem; opacity: 0.7;
}
.result-pct    {
    font-family: 'DM Serif Display', serif; font-size: 4rem;
    line-height: 1; margin-bottom: 0.4rem;
}
.result-level  { font-size: 1.1rem; font-weight: 600; letter-spacing: 1px; }
.factor-chip {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 16px; border-radius: 50px; font-size: 0.82rem;
    font-weight: 500; margin: 4px; border: 1px solid;
}
.factor-red    { background: rgba(255,71,87,0.1);  border-color: rgba(255,71,87,0.3);  color: #FF8A94; }
.factor-orange { background: rgba(255,165,2,0.1);  border-color: rgba(255,165,2,0.3);  color: #FFD32A; }
.factor-green  { background: rgba(0,210,180,0.1);  border-color: rgba(0,210,180,0.3);  color: #00D2B4; }
.reco-box {
    border-radius: 14px; padding: 1.2rem 1.5rem;
    margin-top: 1.2rem; border: 1px solid;
    font-size: 0.88rem; line-height: 1.6;
}
.reco-high   { background: rgba(255,71,87,0.06);  border-color: rgba(255,71,87,0.2);  color: #FF8A94; }
.reco-medium { background: rgba(255,165,2,0.06);  border-color: rgba(255,165,2,0.2);  color: #FFD32A; }
.reco-low    { background: rgba(0,210,180,0.06);  border-color: rgba(0,210,180,0.2);  color: #00D2B4; }
.disclaimer {
    font-size: 0.72rem; color: rgba(255,255,255,0.25);
    text-align: center; margin-top: 2rem; padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05); line-height: 1.6;
}
.ng-divider { height: 1px; background: rgba(255,255,255,0.06); margin: 1.5rem 0; }
.idle-state { text-align: center; padding: 3rem 1rem; color: rgba(255,255,255,0.2); }
.idle-icon  { font-size: 4rem; margin-bottom: 1rem; opacity: 0.4; }
.idle-text  { font-size: 0.88rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Colonnes exactes du modèle ───────────────────────────────────
FEATURE_COLS = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Female', 'gender_Male',
    'ever_married_No', 'ever_married_Yes',
    'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children',
    'Residence_type_Rural', 'Residence_type_Urban',
    'smoking_status_Unknown', 'smoking_status_formerly smoked',
    'smoking_status_never smoked', 'smoking_status_smokes'
]

# ── Seuil médical ────────────────────────────────────────────────
THRESHOLD = 0.3

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base   = os.path.dirname(os.path.abspath(__file__))
    model  = joblib.load(os.path.join(base, 'gb_model.pkl'))
    scaler = joblib.load(os.path.join(base, 'gb_scaler.pkl'))
    return model, scaler

model, scaler = load_model()

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="ng-header">
    <div class="ng-logo">
        <div class="ng-logo-icon">🧠</div>
        <div>
            <div class="ng-logo-text">NeuroGuard</div>
            <div class="ng-logo-sub">Stroke Risk Intelligence</div>
        </div>
    </div>
    <div class="ng-badge">✦ Gradient Boosting Model</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 1], gap="large")

# ════════════════════════════════════════════════════════════════
#  FORMULAIRE
# ════════════════════════════════════════════════════════════════
with col_form:
    st.markdown('<div class="ng-section-label">◈ Patient Profile</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age     = st.number_input("Age", min_value=1, max_value=100,
                                   value=50, step=1)
        bmi     = st.number_input("BMI (kg/m²)", min_value=10.0,
                                   max_value=100.0, value=28.0,
                                   step=0.1, format="%.1f")
        glucose = st.number_input("Avg Glucose (mg/dL)", min_value=50.0,
                                   max_value=300.0, value=100.0,
                                   step=0.1, format="%.1f")
    with c2:
        hypertension  = st.selectbox("Hypertension", [0, 1],
                                      format_func=lambda x: "Yes" if x else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1],
                                      format_func=lambda x: "Yes" if x else "No")
        gender        = st.selectbox("Gender", ["Female", "Male"])

    c3, c4 = st.columns(2)
    with c3:
        ever_married   = st.selectbox("Ever Married", ["Yes", "No"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    with c4:
        work_type      = st.selectbox("Work Type",
                                       ["Private", "Self-employed",
                                        "Govt_job", "children", "Never_worked"])
        smoking_status = st.selectbox("Smoking Status",
                                       ["never smoked", "formerly smoked",
                                        "smokes", "Unknown"])

    predict_btn = st.button("⟡  Run Stroke Risk Analysis")

# ════════════════════════════════════════════════════════════════
#  RÉSULTATS
# ════════════════════════════════════════════════════════════════
with col_result:
    st.markdown('<div class="ng-section-label">◈ Risk Assessment</div>',
                unsafe_allow_html=True)

    if predict_btn:

        # ── Construire input ─────────────────────────────────────
        input_data = {
            'age':               age,
            'hypertension':      hypertension,
            'heart_disease':     heart_disease,
            'avg_glucose_level': glucose,
            'bmi':               bmi,
            'gender_Female':     1 if gender == "Female" else 0,
            'gender_Male':       1 if gender == "Male"   else 0,
            'ever_married_No':   1 if ever_married == "No"  else 0,
            'ever_married_Yes':  1 if ever_married == "Yes" else 0,
            'work_type_Govt_job':      1 if work_type == "Govt_job"      else 0,
            'work_type_Never_worked':  1 if work_type == "Never_worked"  else 0,
            'work_type_Private':       1 if work_type == "Private"       else 0,
            'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
            'work_type_children':      1 if work_type == "children"      else 0,
            'Residence_type_Rural':    1 if residence_type == "Rural"    else 0,
            'Residence_type_Urban':    1 if residence_type == "Urban"    else 0,
            'smoking_status_Unknown':         1 if smoking_status == "Unknown"         else 0,
            'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
            'smoking_status_never smoked':    1 if smoking_status == "never smoked"    else 0,
            'smoking_status_smokes':          1 if smoking_status == "smokes"          else 0,
        }

        # Aligner + scaler + prédiction
        input_df     = pd.DataFrame([input_data])[FEATURE_COLS]
        input_scaled = scaler.transform(input_df)
        proba        = model.predict_proba(input_scaled)[0][1]
        risk         = round(proba * 100, 1)

        # ── Niveau selon seuil médical 0.3 ───────────────────────
        if proba < THRESHOLD:
            lvl, cls, color, emoji = "LOW RISK",  "low",    "#00D2B4", "✓"
        elif proba < 0.6:
            lvl, cls, color, emoji = "MODERATE",  "medium", "#FFA502", "⚠"
        else:
            lvl, cls, color, emoji = "HIGH RISK", "high",   "#FF4757", "✕"

        # ── Result card ──────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card result-{cls}">
            <div class="result-label">Stroke Probability</div>
            <div class="result-pct" style="color:{color}">{risk}%</div>
            <div class="result-level" style="color:{color}">{emoji} {lvl}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauge ────────────────────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk,
            number = {"suffix": "%", "font": {"size": 28, "color": color,
                                               "family": "DM Sans"}},
            gauge = {
                "axis": {"range": [0, 100],
                         "tickcolor": "rgba(255,255,255,0.2)",
                         "tickfont":  {"color": "rgba(255,255,255,0.3)", "size": 11}},
                "bar":  {"color": color, "thickness": 0.25},
                "bgcolor":     "rgba(255,255,255,0.03)",
                "bordercolor": "rgba(255,255,255,0)",
                "steps": [
                    {"range": [0,  30], "color": "rgba(0,210,180,0.08)"},
                    {"range": [30, 60], "color": "rgba(255,165,2,0.08)"},
                    {"range": [60,100], "color": "rgba(255,71,87,0.08)"},
                ],
                "threshold": {"line":      {"color": color, "width": 3},
                              "thickness": 0.8,
                              "value":     risk}
            }
        ))
        fig.update_layout(
            height=200, margin=dict(t=20, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "DM Sans", "color": "rgba(255,255,255,0.5)"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Divider ──────────────────────────────────────────────
        st.markdown('<div class="ng-divider"></div>', unsafe_allow_html=True)

        # ── Risk factors ─────────────────────────────────────────
        st.markdown('<div class="ng-section-label">◈ Risk Factors Detected</div>',
                    unsafe_allow_html=True)

        factors_html = ""
        if age >= 60:
            factors_html += f'<span class="factor-chip factor-red">⬤ Age {age}y</span>'
        if hypertension:
            factors_html += '<span class="factor-chip factor-red">⬤ Hypertension</span>'
        if heart_disease:
            factors_html += '<span class="factor-chip factor-red">⬤ Heart Disease</span>'
        if glucose > 140:
            factors_html += f'<span class="factor-chip factor-red">⬤ Glucose {glucose:.1f}</span>'
        if bmi > 30:
            factors_html += f'<span class="factor-chip factor-orange">⬤ BMI {bmi:.1f}</span>'
        if smoking_status == "smokes":
            factors_html += '<span class="factor-chip factor-orange">⬤ Active Smoker</span>'
        if smoking_status == "formerly smoked":
            factors_html += '<span class="factor-chip factor-orange">⬤ Ex-Smoker</span>'

        if not factors_html:
            factors_html = '<span class="factor-chip factor-green">✓ No major risk factors</span>'

        st.markdown(f'<div style="margin-bottom:1rem">{factors_html}</div>',
                    unsafe_allow_html=True)

        # ── Recommandation ───────────────────────────────────────
        if proba >= 0.6:
            reco_cls  = "high"
            reco_text = "🏥 <strong>Urgent medical consultation recommended.</strong> The patient presents a high probability of stroke. Immediate clinical evaluation is advised."
        elif proba >= THRESHOLD:
            reco_cls  = "medium"
            reco_text = "⚕️ <strong>Regular medical follow-up recommended.</strong> Lifestyle adjustments and periodic check-ups are advised."
        else:
            reco_cls  = "low"
            reco_text = "✓ <strong>Low risk profile.</strong> Continue maintaining healthy lifestyle habits. Routine annual check-ups are sufficient."

        st.markdown(f'<div class="reco-box reco-{reco_cls}">{reco_text}</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            ⚠ This tool is an AI-assisted decision support system only.<br>
            It does not replace professional medical diagnosis or clinical judgment.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="ng-panel">
            <div class="idle-state">
                <div class="idle-icon">🧠</div>
                <div class="idle-text">
                    Fill in the patient profile on the left<br>
                    and run the analysis to see the stroke<br>
                    risk assessment here.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)