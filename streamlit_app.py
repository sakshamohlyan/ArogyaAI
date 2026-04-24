"""
ArogyaAI — Streamlit Frontend
==============================
Changes from v2:
  - Removed NeuroScan (brain) and RetinaScan (retinopathy) cards
  - Removed "Research" nav button
  - Renamed GlucoScan → Diabetes Analyser (numerical input, no scan)
  - Renamed CardioScan → Heart Risk Analyser (numerical input, no scan)
  - Added full About section: Saksham Ohlyan, UIET Punjab University
  - Added Project Details and Cautions section
  - Updated copyright to current year (2025)

Run:
  python app.py          (Terminal 1 — Flask API)
  streamlit run streamlit_app.py  (Terminal 2 — UI)
"""

import io
import os
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE     = os.environ.get("API_BASE", "http://localhost:5050")
CURRENT_YEAR = datetime.now().year   # auto-updates every year

st.set_page_config(
    page_title="ArogyaAI | Advanced Disease Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Cyberpunk CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700&family=Orbitron:wght@400;700&display=swap');

:root {
  --primary:     #00f7ff;
  --secondary:   #ff00d6;
  --accent:      #00ff66;
  --dark:        #0a0e17;
  --darker:      #050811;
  --light:       #ccd6f6;
  --card-bg:     rgba(15, 25, 45, 0.85);
  --card-border: rgba(0, 247, 255, 0.3);
}

.stApp {
  background-color: var(--darker);
  background-image:
    radial-gradient(circle at 15% 50%, rgba(0,247,255,0.05) 0%, transparent 20%),
    radial-gradient(circle at 85% 30%, rgba(255,0,214,0.05) 0%, transparent 20%),
    radial-gradient(circle at 50% 80%, rgba(0,255,102,0.05) 0%, transparent 20%),
    linear-gradient(to right,  rgba(0,247,255,0.04) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(0,247,255,0.04) 1px, transparent 1px);
  background-size: auto, auto, auto, 25px 25px, 25px 25px;
  font-family: 'Exo 2', sans-serif;
  color: var(--light);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
div[data-testid="stDecoration"]  { display: none; }

/* ── Nav ── */
.nav-bar {
  display: flex; justify-content: center; gap: 30px;
  padding: 12px 0; border-bottom: 1px solid var(--card-border);
  margin-bottom: 10px;
}
.nav-link {
  color: var(--light); text-decoration: none;
  font-weight: 600; font-size: 0.95rem; opacity: 0.85;
}

/* ── Hero ── */
.hero-header { text-align: center; padding: 50px 20px 20px; }
.logo-row {
  display: flex; align-items: center; justify-content: center;
  gap: 16px; margin-bottom: 12px;
}
.logo-icon {
  width: 48px; height: 48px; background: var(--primary);
  border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: 22px; font-weight: 700; color: #000;
  box-shadow: 0 0 15px #00f7ff, 0 0 30px #00f7ff;
  animation: pulse 3s infinite alternate;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 10px #00f7ff, 0 0 20px #00f7ff; }
  100% { box-shadow: 0 0 25px #00f7ff, 0 0 50px #00f7ff; }
}
.logo-text {
  font-family: 'Orbitron', sans-serif; font-size: 2.6rem; font-weight: 700;
  background: linear-gradient(45deg, var(--primary), var(--accent));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-title {
  font-family: 'Orbitron', sans-serif; font-size: 2.8rem; font-weight: 700;
  background: linear-gradient(45deg, var(--primary), var(--secondary));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 10px 0;
}
.hero-sub {
  font-size: 1.1rem; opacity: 0.8;
  max-width: 680px; margin: 0 auto 20px; line-height: 1.6;
}
.disclaimer-box {
  background: rgba(255,0,0,0.08); border: 1px solid rgba(255,80,80,0.4);
  border-radius: 10px; padding: 14px 20px;
  max-width: 780px; margin: 0 auto 30px;
  font-size: 0.9rem; color: #ffaaaa; text-align: center;
}

/* ── Section heading ── */
.section-heading {
  font-family: 'Orbitron', sans-serif; font-size: 1.2rem;
  color: var(--primary); text-align: center; letter-spacing: 3px;
  margin: 30px 0 20px; text-transform: uppercase;
}

/* ── Disease cards ── */
.disease-card {
  background: var(--card-bg); border: 1px solid var(--card-border);
  border-radius: 14px; padding: 28px 22px; text-align: center;
  position: relative; overflow: hidden; transition: all 0.3s ease;
}
.disease-card::before {
  content: ''; position: absolute;
  top: -2px; left: -2px; right: -2px; bottom: -2px; z-index: -1;
  background: linear-gradient(45deg, var(--primary), transparent, var(--secondary));
  border-radius: 16px; opacity: 0.3;
  animation: border-glow 3s infinite alternate;
}
@keyframes border-glow {
  0% { opacity: 0.2; } 100% { opacity: 0.6; }
}
.disease-card:hover { transform: translateY(-6px); border-color: var(--primary); }
.card-icon {
  font-size: 2.5rem; width: 72px; height: 72px;
  background: rgba(0,247,255,0.08); border: 1px solid var(--primary);
  border-radius: 50%; display: flex; align-items: center;
  justify-content: center; margin: 0 auto 14px;
  box-shadow: 0 0 12px rgba(0,247,255,0.3);
}
.card-title {
  font-family: 'Orbitron', sans-serif; font-size: 1.1rem;
  color: var(--primary); margin-bottom: 8px;
}
.card-badge {
  display: inline-block; padding: 2px 10px; border-radius: 20px;
  font-size: 0.72rem; font-weight: 600; margin-bottom: 8px;
}
.badge-image   { background: rgba(0,247,255,0.12); color: var(--primary); border: 1px solid var(--primary); }
.badge-numeric { background: rgba(255,0,214,0.12); color: #ff88ff; border: 1px solid #ff00d6; }
.card-desc { font-size: 0.85rem; opacity: 0.75; margin-bottom: 16px; }

/* ── Detection panel ── */
.detect-panel {
  background: var(--card-bg); border: 1px solid var(--card-border);
  border-radius: 14px; padding: 32px;
  max-width: 760px; margin: 0 auto 40px;
  position: relative; overflow: hidden;
}
.detect-panel::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, var(--primary), var(--secondary));
}
.panel-title {
  font-family: 'Orbitron', sans-serif; font-size: 1.5rem;
  color: var(--primary); margin-bottom: 22px;
}

/* ── Form fields ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stFileUploader > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: 8px !important; color: var(--light) !important;
}
.stNumberInput label, .stSelectbox label,
.stFileUploader label { color: var(--primary) !important; font-weight: 600; }

/* ── Result boxes ── */
.result-positive {
  background: rgba(255,80,80,0.1); border: 1px solid rgba(255,80,80,0.5);
  border-left: 4px solid #ff5050; border-radius: 10px;
  padding: 20px; margin-top: 20px;
}
.result-negative {
  background: rgba(0,255,102,0.06); border: 1px solid rgba(0,255,102,0.4);
  border-left: 4px solid var(--accent); border-radius: 10px;
  padding: 20px; margin-top: 20px;
}
.result-title { font-family: 'Orbitron', sans-serif; font-size: 1.2rem; margin-bottom: 10px; }
.result-positive .result-title { color: #ff7070; }
.result-negative .result-title { color: var(--accent); }
.risk-badge {
  display: inline-block; padding: 3px 14px; border-radius: 20px;
  font-size: 0.85rem; font-weight: 700; margin-bottom: 12px;
}
.risk-HIGH     { background: rgba(255,50,50,0.2);  color: #ff7070; border: 1px solid #ff5050; }
.risk-MODERATE { background: rgba(255,170,0,0.15); color: #ffcc44; border: 1px solid #ffaa00; }
.risk-LOW      { background: rgba(0,255,102,0.1);  color: var(--accent); border: 1px solid var(--accent); }
.conf-bar-wrap { height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; margin: 10px 0; overflow: hidden; }
.conf-bar { height: 100%; border-radius: 5px; background: linear-gradient(90deg, var(--accent), var(--primary)); }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #007bff, #0056b3) !important;
  color: white !important; border: none !important;
  border-radius: 30px !important; padding: 10px 28px !important;
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.9rem !important; font-weight: 600 !important;
  transition: all 0.3s !important; width: 100% !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(0,123,255,0.5) !important;
}

/* ── About / info sections ── */
.about-panel {
  background: var(--card-bg); border: 1px solid var(--card-border);
  border-radius: 14px; padding: 36px 40px;
  max-width: 900px; margin: 0 auto 40px; position: relative; overflow: hidden;
}
.about-panel::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, var(--secondary), var(--primary));
}
.about-title {
  font-family: 'Orbitron', sans-serif; font-size: 1.6rem;
  color: var(--secondary); margin-bottom: 20px;
}
.about-name {
  font-family: 'Orbitron', sans-serif; font-size: 1.3rem;
  color: var(--primary); margin: 0 0 4px;
}
.about-inst { font-size: 1rem; color: #aac4ff; margin-bottom: 20px; }
.about-text { font-size: 0.92rem; line-height: 1.8; color: var(--light); opacity: 0.88; }
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 20px 0; }
.info-card {
  background: rgba(0,247,255,0.04); border: 1px solid rgba(0,247,255,0.15);
  border-radius: 10px; padding: 14px 16px;
}
.info-card h4 { font-size: 0.85rem; color: var(--primary); font-weight: 600; margin-bottom: 6px; }
.info-card p  { font-size: 0.82rem; color: var(--light); opacity: 0.8; line-height: 1.6; margin: 0; }
.caution-box {
  background: rgba(255,170,0,0.07); border: 1px solid rgba(255,170,0,0.35);
  border-left: 4px solid #ffaa00; border-radius: 0 10px 10px 0;
  padding: 14px 18px; margin: 10px 0;
}
.caution-box h4 { color: #ffcc44; font-size: 0.88rem; font-weight: 600; margin-bottom: 4px; }
.caution-box p  { font-size: 0.82rem; color: var(--light); opacity: 0.82; margin: 0; line-height: 1.6; }
.tech-pill {
  display: inline-block; background: rgba(0,255,102,0.08);
  border: 1px solid rgba(0,255,102,0.3); border-radius: 20px;
  padding: 3px 12px; font-size: 0.78rem; color: var(--accent);
  margin: 3px;
}

/* ── Footer ── */
.footer {
  text-align: center; padding: 30px 20px;
  border-top: 1px solid rgba(255,255,255,0.08);
  margin-top: 20px; font-size: 0.85rem; opacity: 0.6;
}

/* ── Tabs ── */
div[data-baseweb="tab-list"] { background: transparent !important; }
div[data-baseweb="tab"] {
  background: var(--card-bg) !important; border: 1px solid var(--card-border) !important;
  border-radius: 8px 8px 0 0 !important; color: var(--light) !important;
  font-family: 'Exo 2', sans-serif !important;
}
div[aria-selected="true"] { border-bottom: 2px solid var(--primary) !important; color: var(--primary) !important; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def call_api(endpoint: str, json_data: dict = None, files=None):
    try:
        url = f"{API_BASE}{endpoint}"
        r   = requests.post(url, files=files, timeout=15) if files else \
              requests.post(url, json=json_data, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach Flask API. Make sure `python app.py` is running on port 5050.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def render_result(result: dict):
    if not result:
        return
    prob           = result.get("probability", 0)
    label          = result.get("label", "Unknown")
    level          = result.get("risk_level", "LOW")
    insights       = result.get("insights", [])
    recommendation = result.get("recommendation", "")
    engine         = result.get("engine", "")
    all_classes    = result.get("all_classes", {})

    is_positive = level in ("HIGH", "MODERATE")
    box_class   = "result-positive" if is_positive else "result-negative"
    icon        = "⚠️" if is_positive else "✅"
    pct         = int(prob * 100)

    st.markdown(f"""
    <div class="{box_class}">
      <div class="result-title">{icon} {label}</div>
      <span class="risk-badge risk-{level}">Risk: {level}</span>
      {"&nbsp;&nbsp;<small style='opacity:0.6;font-size:0.75rem'>Engine: " + engine + "</small>" if engine else ""}
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:{pct}%"></div>
      </div>
      <p style="font-size:0.9rem;margin-bottom:10px">
        Confidence: <strong>{pct}%</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)

    if insights:
        st.markdown("**Key Findings:**")
        for tip in insights:
            st.markdown(f"▶ {tip}")

    # Show per-class breakdown for image models
    if all_classes:
        with st.expander("View full class probabilities"):
            for cls, p in sorted(all_classes.items(), key=lambda x: -x[1]):
                st.markdown(f"`{cls}` — {p*100:.1f}%")

    if recommendation:
        st.info(f"**Recommendation:** {recommendation}")

    st.caption("⚠️ For research and educational purposes only. Not a substitute for professional medical advice.")


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
  <div class="nav-bar">
    <span class="nav-link">Home</span>
    <span class="nav-link">Detection</span>
    <span class="nav-link">About</span>
  </div>
  <div class="logo-row">
    <div class="logo-icon">+</div>
    <div class="logo-text">ArogyaAI</div>
  </div>
  <div class="hero-title">ADVANCED DISEASE DETECTION</div>
  <p class="hero-sub">
    AI-powered clinical decision support for pneumonia, skin cancer,
    diabetes risk and cardiac health — built for research and education.
  </p>
  <div class="disclaimer-box">
    ⚠️ This tool is for <strong>research and educational use only</strong>.
    It is <strong>not</strong> a certified medical device and must not replace
    diagnosis or treatment by a qualified healthcare professional.
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE CARDS  — 4 modules only (Brain + Retina removed)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-heading">Select Detection Module</div>', unsafe_allow_html=True)

# (icon, display_name, api_key, description, mode, badge_type, button_label)
DISEASES = [
    ("🫁", "PneumoScan",          "pneumonia", "Pneumonia detection from chest X-ray images",       "image",   "image",   "Upload X-Ray"),
    ("🔬", "DermaScan",           "skin",      "Skin cancer detection from dermatoscopic images",   "image",   "image",   "Upload Image"),
    ("🩸", "Diabetes Analyser",   "diabetes",  "Diabetes risk assessment using clinical parameters","tabular", "numeric", "Enter Values"),
    ("❤️", "Heart Risk Analyser", "heart",     "Cardiovascular disease risk from clinical values",  "tabular", "numeric", "Enter Values"),
]

col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]

for i, (icon, name, key, desc, mode, badge, btn_label) in enumerate(DISEASES):
    badge_class = "badge-image" if badge == "image" else "badge-numeric"
    badge_text  = "Image Upload" if badge == "image" else "Numeric Input"
    with cols[i]:
        st.markdown(f"""
        <div class="disease-card">
          <div class="card-icon">{icon}</div>
          <div class="card-title">{name}</div>
          <span class="card-badge {badge_class}">{badge_text}</span>
          <p class="card-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(btn_label, key=f"btn_{key}"):
            st.session_state["active_disease"] = key
            st.session_state["active_mode"]    = mode
            st.session_state["active_name"]    = name
            st.session_state["active_icon"]    = icon

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION PANEL
# ═══════════════════════════════════════════════════════════════════════════════

active = st.session_state.get("active_disease")

if active:
    mode = st.session_state.get("active_mode", "tabular")
    name = st.session_state.get("active_name", active)
    icon = st.session_state.get("active_icon", "🔬")

    st.markdown(f"""
    <div style="max-width:760px;margin:0 auto 10px;padding:0 20px">
      <div class="detect-panel">
        <div class="panel-title">{icon} {name}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 3, 1])

    with center:

        # ── PNEUMONIA ─────────────────────────────────────────────────────────
        if active == "pneumonia":
            st.markdown("**Upload a chest X-ray image for pneumonia analysis**")
            st.caption("Supported formats: JPG, JPEG, PNG, BMP, TIFF")
            uploaded = st.file_uploader("Choose chest X-ray image",
                                        type=["jpg","jpeg","png","bmp","tiff"],
                                        key="up_pneumonia")
            if uploaded:
                c1, c2 = st.columns([1, 1])
                with c1:
                    try:
                        st.image(Image.open(uploaded), caption="Uploaded X-ray",
                                 use_container_width=True)
                    except Exception:
                        st.info("Image preview unavailable")
                with c2:
                    st.markdown(f"""
                    **File:** `{uploaded.name}`
                    **Size:** `{uploaded.size // 1024} KB`
                    **Type:** `{uploaded.type}`
                    """)
                if st.button("Run PneumoScan Analysis", key="run_pneumonia"):
                    with st.spinner("Analysing X-ray…"):
                        uploaded.seek(0)
                        result = call_api("/api/predict/pneumonia",
                                         files={"file":(uploaded.name, uploaded.read(), uploaded.type)})
                    if result:
                        render_result(result)

        # ── SKIN CANCER ───────────────────────────────────────────────────────
        elif active == "skin":
            st.markdown("**Upload a dermatoscopic skin lesion image**")
            st.caption("Best results with close-up dermatoscope images. Supported: JPG, PNG, BMP, TIFF")
            uploaded = st.file_uploader("Choose skin lesion image",
                                        type=["jpg","jpeg","png","bmp","tiff"],
                                        key="up_skin")
            if uploaded:
                c1, c2 = st.columns([1, 1])
                with c1:
                    try:
                        st.image(Image.open(uploaded), caption="Uploaded lesion image",
                                 use_container_width=True)
                    except Exception:
                        st.info("Image preview unavailable")
                with c2:
                    st.markdown(f"""
                    **File:** `{uploaded.name}`
                    **Size:** `{uploaded.size // 1024} KB`
                    **Type:** `{uploaded.type}`
                    """)
                if st.button("Run DermaScan Analysis", key="run_skin"):
                    with st.spinner("Analysing skin lesion…"):
                        uploaded.seek(0)
                        result = call_api("/api/predict/skin",
                                         files={"file":(uploaded.name, uploaded.read(), uploaded.type)})
                    if result:
                        render_result(result)

        # ── DIABETES ANALYSER ─────────────────────────────────────────────────
        elif active == "diabetes":
            st.markdown("**Enter the patient's clinical measurements below**")
            st.caption("All values are numerical — no image required.")
            c1, c2 = st.columns(2)
            with c1:
                glucose     = st.number_input("Glucose Level (mg/dL)",
                                              min_value=0, max_value=500, value=120,
                                              help="Plasma glucose concentration (2-hour oral glucose tolerance test)",
                                              key="d_gluc")
                bp          = st.number_input("Blood Pressure (mmHg)",
                                              min_value=0, max_value=200, value=72,
                                              help="Diastolic blood pressure in mmHg",
                                              key="d_bp")
                insulin     = st.number_input("Insulin (μU/mL)",
                                              min_value=0, max_value=900, value=80,
                                              help="2-hour serum insulin in μU/mL",
                                              key="d_ins")
            with c2:
                bmi         = st.number_input("BMI (kg/m²)",
                                              min_value=10.0, max_value=70.0, value=28.0,
                                              step=0.1, format="%.1f",
                                              help="Body mass index",
                                              key="d_bmi")
                age         = st.number_input("Age (years)",
                                              min_value=1, max_value=120, value=35,
                                              key="d_age")
                pregnancies = st.number_input("Number of Pregnancies",
                                              min_value=0, max_value=20, value=0,
                                              help="Number of times pregnant (enter 0 if not applicable)",
                                              key="d_preg")

            if st.button("Analyse Diabetes Risk", key="run_diabetes"):
                with st.spinner("Calculating diabetes risk…"):
                    result = call_api("/api/predict/diabetes", {
                        "glucose": glucose, "bp": bp, "insulin": insulin,
                        "bmi": bmi, "age": age, "pregnancies": pregnancies,
                    })
                if result:
                    render_result(result)

        # ── HEART RISK ANALYSER ───────────────────────────────────────────────
        elif active == "heart":
            st.markdown("**Enter the patient's cardiac clinical values below**")
            st.caption("All values are numerical — no image required.")
            c1, c2 = st.columns(2)
            with c1:
                age      = st.number_input("Age (years)",
                                           min_value=29, max_value=100, value=50,
                                           key="h_age")
                sex      = st.selectbox("Biological Sex",
                                        ["Male", "Female"],
                                        key="h_sex")
                cp       = st.number_input("Chest Pain Type (0–3)",
                                           min_value=0, max_value=3, value=1,
                                           help="0=Typical angina  1=Atypical angina  2=Non-anginal  3=Asymptomatic",
                                           key="h_cp")
                trestbps = st.number_input("Resting Blood Pressure (mmHg)",
                                           min_value=80, max_value=220, value=130,
                                           key="h_tbp")
            with c2:
                chol    = st.number_input("Serum Cholesterol (mg/dL)",
                                          min_value=100, max_value=600, value=240,
                                          key="h_chol")
                thalach = st.number_input("Maximum Heart Rate Achieved",
                                          min_value=60, max_value=220, value=150,
                                          key="h_thal")
                exang   = st.selectbox("Exercise-Induced Angina",
                                       ["No", "Yes"],
                                       help="Chest pain or discomfort during physical activity",
                                       key="h_exang")

            if st.button("Analyse Heart Disease Risk", key="run_heart"):
                with st.spinner("Calculating cardiovascular risk…"):
                    result = call_api("/api/predict/heart", {
                        "age": age, "sex": sex, "cp": cp,
                        "trestbps": trestbps, "chol": chol,
                        "thalach": thalach, "exang": exang,
                    })
                if result:
                    render_result(result)

    # Close button
    _, c2, _ = st.columns([1, 3, 1])
    with c2:
        if st.button("✕ Close Panel", key="close"):
            for k in ["active_disease","active_mode","active_name","active_icon"]:
                st.session_state.pop(k, None)
            st.rerun()

else:
    # Welcome placeholder
    st.markdown("""
    <div style="max-width:900px;margin:0 auto;padding:0 20px">
      <div class="detect-panel" style="text-align:center;padding:48px">
        <div style="font-size:3.5rem;margin-bottom:18px">🧬</div>
        <div style="font-family:'Orbitron',sans-serif;font-size:1.3rem;
                    color:var(--primary);margin-bottom:14px">
          Select a Detection Module Above
        </div>
        <p style="opacity:0.7;max-width:520px;margin:0 auto;line-height:1.7;font-size:0.9rem">
          <strong style="color:var(--primary)">Image-based modules</strong> (PneumoScan, DermaScan)
          — upload a medical photograph for AI analysis.<br><br>
          <strong style="color:#ff88ff">Numerical modules</strong> (Diabetes Analyser, Heart Risk Analyser)
          — enter clinical measurements from a patient report or lab test.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT SECTION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-heading">About This Project</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="about-panel">
  <div class="about-title">Project Details</div>

  <div class="about-name">Saksham Ohlyan</div>
  <div class="about-inst">UIET — University Institute of Engineering &amp; Technology<br>
  Panjab University, Chandigarh</div>

  <div class="about-text">
    ArogyaAI is an end-to-end AI-powered clinical decision support system
    developed as part of a Machine Learning and MLOps project at UIET, Panjab University.
    It integrates four disease detection modules — pneumonia detection from chest X-rays,
    skin cancer classification from dermatoscopic images, diabetes risk prediction,
    and cardiovascular disease risk assessment — into a single unified web application.
  </div>

  <div class="info-grid">
    <div class="info-card">
      <h4>Tech Stack</h4>
      <p>
        <span class="tech-pill">Python 3.12</span>
        <span class="tech-pill">Flask 3.x</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">Scikit-learn</span>
        <span class="tech-pill">PyTorch</span>
        <span class="tech-pill">MobileNetV2</span>
        <span class="tech-pill">Pillow</span>
        <span class="tech-pill">NumPy / Pandas</span>
      </p>
    </div>
    <div class="info-card">
      <h4>Datasets Used</h4>
      <p>
        Chest X-Ray Images (Pneumonia) — Kaggle / Paul Mooney<br>
        Skin Cancer MNIST: HAM10000 — ISIC / Kaggle<br>
        Pima Indians Diabetes Database — UCI / Kaggle<br>
        UCI Heart Disease (Cleveland) — UCI / Kaggle
      </p>
    </div>
    <div class="info-card">
      <h4>Models</h4>
      <p>
        Image classification: MobileNetV2 (Transfer Learning) fine-tuned
        on domain-specific Kaggle datasets.<br>
        Tabular: Gradient Boosting (Diabetes) and
        Random Forest (Heart Disease) with Scikit-learn Pipelines.
      </p>
    </div>
    <div class="info-card">
      <h4>Architecture</h4>
      <p>
        Two-tier system: Flask REST API (port 5050) handles all ML inference.
        Streamlit frontend (port 8501) provides the interactive UI.
        CNN weights auto-load when available; falls back to
        feature-extraction + GBM when not trained.
      </p>
    </div>
  </div>

  <div class="about-title" style="font-size:1.2rem;margin-top:28px;margin-bottom:16px">
    ⚠️ Important Cautions
  </div>

  <div class="caution-box">
    <h4>Not a Medical Device</h4>
    <p>ArogyaAI is a research and educational prototype. It has not been validated
    clinically, has not received regulatory approval (CDSCO / FDA / CE), and must
    never be used as the sole basis for clinical decisions, diagnosis, or treatment.</p>
  </div>

  <div class="caution-box">
    <h4>Model Accuracy Limitations</h4>
    <p>The models are trained on publicly available Kaggle datasets which may not
    represent the full diversity of patient populations, imaging equipment, or
    clinical conditions. Accuracy on real-world data may differ significantly
    from reported training metrics.</p>
  </div>

  <div class="caution-box">
    <h4>Image Quality Dependency</h4>
    <p>Image-based modules (PneumoScan, DermaScan) are sensitive to image quality.
    Poor lighting, low resolution, incorrect orientation, or non-clinical photographs
    will produce unreliable results. Only use properly acquired medical images.</p>
  </div>

  <div class="caution-box">
    <h4>No Patient Data Storage</h4>
    <p>This application does not store, log, or transmit any patient data or uploaded
    images beyond the scope of a single analysis request. All data is processed
    in memory and discarded after the API response is returned.</p>
  </div>

  <div class="caution-box">
    <h4>Always Consult a Doctor</h4>
    <p>Any concerning result from this tool must be followed up with a licensed
    healthcare professional. Self-diagnosis based on AI predictions is dangerous.
    This tool is intended to assist, not replace, qualified medical judgment.</p>
  </div>

</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER — dynamic year, no hardcoded 2024
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="footer">
  <p>ArogyaAI | Advanced Disease Detection System</p>
  <p>Developed by <strong>Saksham Ohlyan</strong> &nbsp;|&nbsp;
     UIET, Panjab University, Chandigarh</p>
  <p>For research and educational purposes only &nbsp;&copy;&nbsp;{CURRENT_YEAR}</p>
</div>
""", unsafe_allow_html=True)