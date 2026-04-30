"""
Student Dropout Prediction System
HCS221 | Great Zimbabwe University
Clean minimal light theme — white & green, mobile-card layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GZU Dropout Risk",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── custom CSS — clean light / mobile-card aesthetic ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f0f4f0 !important;
    color: #1a2e1a !important;
    font-family: 'Inter', sans-serif !important;
}

/* centre and constrain like a phone screen */
[data-testid="stMainBlockContainer"] {
    max-width: 480px !important;
    margin: 0 auto !important;
    padding: 0 12px 40px 12px !important;
}

/* hide sidebar toggle */
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ── headings ── */
h1 { font-size: 20px !important; font-weight: 700 !important; color: #1a2e1a !important; margin-bottom: 2px !important; }
h2 { font-size: 14px !important; font-weight: 600 !important; color: #2e7d32 !important; letter-spacing: 0.5px; text-transform: uppercase; }
h3 { font-size: 13px !important; font-weight: 600 !important; color: #388e3c !important; }

/* ── card wrapper (use via st.markdown) ── */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}

.section-label {
    font-size: 11px;
    font-weight: 600;
    color: #81c784;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e8f5e9 !important;
    border-top: 3px solid #43a047 !important;
    border-radius: 12px !important;
    padding: 12px 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    font-weight: 600 !important;
    color: #81c784 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1b5e20 !important;
}

/* ── sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #43a047 !important;
}

/* ── inputs & selects ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    border: 1px solid #c8e6c9 !important;
    border-radius: 10px !important;
    background: #f9fdf9 !important;
    color: #1a2e1a !important;
    font-size: 14px !important;
}

/* ── primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #2e7d32 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(46,125,50,0.25) !important;
    transition: background 0.2s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1b5e20 !important;
}

/* ── secondary button ── */
[data-testid="stButton"] > button {
    background: #ffffff !important;
    border: 1.5px solid #a5d6a7 !important;
    border-radius: 10px !important;
    color: #2e7d32 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #a5d6a7 !important;
    border-radius: 12px !important;
    background: #f9fdf9 !important;
    padding: 10px !important;
}

/* ── alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    font-size: 14px !important;
}

/* ── divider ── */
hr { border-color: #e8f5e9 !important; margin: 12px 0 !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e8f5e9 !important;
    border-radius: 10px !important;
}

/* ── tabs ── */
[data-testid="stTabs"] button {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #81c784 !important;
    border-radius: 0 !important;
    flex: 1 !important;
    text-align: center !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2e7d32 !important;
    border-bottom: 2px solid #2e7d32 !important;
}

/* ── caption ── */
[data-testid="stCaptionContainer"] {
    font-size: 11px !important;
    color: #a5d6a7 !important;
}

/* ── code ── */
.stCode, code {
    background: #f1f8e9 !important;
    border: 1px solid #c8e6c9 !important;
    border-radius: 8px !important;
    color: #1b5e20 !important;
    font-size: 12px !important;
}

/* ── expand/collapse ── */
[data-testid="stExpander"] {
    border: 1px solid #e8f5e9 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ── load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("dropout_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_model()

# ── encoding maps ─────────────────────────────────────────────────────────────
INTERNET_MAP = {'Poor': 0, 'Moderate': 1, 'Good': 2}
ELEC_MAP     = {'Low': 0, 'Medium': 1, 'High': 2}
FEES_MAP     = {'Unknown': 0, 'Partial': 1, 'Full': 2}
GRADE_MAP    = {'F': 0, '1': 1, '2.1': 2, '2.2': 3, '3': 4}
INCOME_MIN, INCOME_MAX = 50.0, 800.0


def encode_and_scale(inputs: dict) -> pd.DataFrame:
    d = inputs.copy()
    d['gender']        = 1 if d['gender'] == 'Female' else 0
    d['location']      = 1 if d['location'] == 'Rural' else 0
    d['part_time_job'] = 1 if d['part_time_job'] == 'Yes' else 0
    d['internet_access']         = INTERNET_MAP[d['internet_access']]
    d['electricity_reliability'] = ELEC_MAP[d['electricity_reliability']]
    d['fees_paid']               = FEES_MAP[d['fees_paid']]
    d['previous_grade']          = GRADE_MAP[d['previous_grade']]
    df_raw = pd.DataFrame([d])
    scale_cols = ['age','family_income','transport_time','study_hours',
                  'attendance_rate','lms_logins','stress_level']
    df_raw[scale_cols] = scaler.transform(df_raw[scale_cols])
    df_raw['study_efficiency'] = df_raw['study_hours'] / (df_raw['attendance_rate'] + 0.01)
    df_raw['hardship_index']   = (
        (1 - df_raw['family_income']) +
        (2 - df_raw['electricity_reliability']) +
        (2 - df_raw['internet_access'])
    )
    return df_raw[features]


# ── top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:10px;padding:18px 0 8px 0'>
  <div style='background:#2e7d32;border-radius:12px;width:40px;height:40px;
              display:flex;align-items:center;justify-content:center;font-size:20px'>🎓</div>
  <div>
    <div style='font-size:17px;font-weight:700;color:#1a2e1a;line-height:1.2'>Dropout Risk System</div>
    <div style='font-size:11px;color:#81c784;font-weight:500'>GZU · HCS221</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── navigation tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Assess  ", "  Batch  ", "  Info  "])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("<div class='section-label'>Personal Details</div>", unsafe_allow_html=True)
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            age    = st.slider("Age", 18, 24, 20)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            location      = st.selectbox("Location", ["Urban", "Rural"])
            part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])

    st.divider()
    st.markdown("<div class='section-label'>Socio-Economic</div>", unsafe_allow_html=True)
    family_income           = st.slider("Family Income (USD/mo)", 50, 800, 250, 10)
    fees_paid               = st.selectbox("Fees Paid", ["Full", "Partial", "Unknown"])
    c3, c4 = st.columns(2)
    with c3:
        internet_access = st.selectbox("Internet", ["Good", "Moderate", "Poor"])
    with c4:
        electricity_reliability = st.selectbox("Electricity", ["High", "Medium", "Low"])
    transport_time = st.slider("Transport Time (hrs)", 0.0, 4.0, 1.0, 0.1)

    st.divider()
    st.markdown("<div class='section-label'>Academic</div>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        study_hours    = st.slider("Study Hours/day", 0.0, 10.0, 3.0, 0.5)
        attendance_rate = st.slider("Attendance (%)", 0.0, 100.0, 75.0, 1.0)
    with c6:
        lms_logins     = st.number_input("LMS Logins/month", 0, 100, 12)
        previous_grade = st.selectbox("Previous Grade", ["3", "2.2", "2.1", "1", "F"])
    stress_level = st.slider("Stress Level (1–10)", 1.0, 10.0, 5.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Assessment", type="primary", use_container_width=True)

    if run:
        inputs = dict(
            age=age, gender=gender, location=location, part_time_job=part_time_job,
            family_income=family_income, fees_paid=fees_paid,
            internet_access=internet_access, electricity_reliability=electricity_reliability,
            transport_time=transport_time, study_hours=study_hours,
            attendance_rate=attendance_rate, lms_logins=lms_logins,
            previous_grade=previous_grade, stress_level=stress_level
        )
        X_input = encode_and_scale(inputs)
        proba   = model.predict_proba(X_input)[0][1]
        pred    = int(proba >= 0.5)
        tier    = "High" if proba >= 0.65 else ("Medium" if proba >= 0.4 else "Low")
        tier_color = {"High": "#c62828", "Medium": "#e65100", "Low": "#2e7d32"}[tier]

        st.divider()

        # ── result card ──
        bg = "#ffebee" if pred == 1 else "#e8f5e9"
        border = "#ef9a9a" if pred == 1 else "#a5d6a7"
        icon   = "⚠️" if pred == 1 else "✅"
        label  = "Dropout Risk Detected" if pred == 1 else "Low Risk — Likely to Continue"
        st.markdown(f"""
        <div style='background:{bg};border:1.5px solid {border};border-radius:14px;
                    padding:16px;margin:8px 0 4px 0'>
          <div style='font-size:22px;margin-bottom:4px'>{icon}</div>
          <div style='font-size:16px;font-weight:700;color:{tier_color}'>{label}</div>
          <div style='font-size:28px;font-weight:800;color:{tier_color};margin:6px 0 2px 0'>{proba:.1%}</div>
          <div style='font-size:11px;color:#666;font-weight:500'>dropout probability · {tier} risk</div>
        </div>
        """, unsafe_allow_html=True)

        # ── probability bar ──
        fig, ax = plt.subplots(figsize=(5, 1.2))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        ax.barh(0, 1, height=0.5, color='#e8f5e9', edgecolor='#c8e6c9', linewidth=1)
        bar_c = '#c62828' if proba >= 0.65 else ('#e65100' if proba >= 0.4 else '#2e7d32')
        ax.barh(0, proba, height=0.5, color=bar_c, edgecolor='none')
        ax.axvline(0.5, color='#9e9e9e', linewidth=1, linestyle='--', alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%','25%','50%','75%','100%'], color='#9e9e9e', fontsize=8)
        ax.tick_params(axis='x', length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── metrics row ──
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability", f"{proba:.1%}")
        m2.metric("Risk Tier",   tier)
        m3.metric("Confidence",  f"{max(proba, 1-proba):.1%}")

        # ── risk flags ──
        flags = []
        if fees_paid == 'Unknown':    flags.append(("Fees not recorded", "💳"))
        if attendance_rate < 60:      flags.append(("Low attendance", "📉"))
        if previous_grade == 'F':     flags.append(("Failed previous grade", "📝"))
        if location == 'Rural':       flags.append(("Rural location", "🏡"))
        if internet_access == 'Poor': flags.append(("Poor internet access", "📶"))
        if family_income < 200:       flags.append(("Low family income", "💰"))
        if stress_level > 7:          flags.append(("High stress level", "😓"))
        if part_time_job == 'Yes':    flags.append(("Working part-time", "💼"))

        if flags:
            st.markdown("<div style='margin-top:10px;font-size:11px;font-weight:600;color:#81c784;text-transform:uppercase;letter-spacing:1px'>Risk Factors</div>", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, (flag, icon) in enumerate(flags):
                cols[i % 2].markdown(
                    f"<div style='background:#fff3e0;border-radius:8px;padding:6px 10px;"
                    f"margin-bottom:6px;font-size:12px;color:#bf360c'>{icon} {flag}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div style='background:#e8f5e9;border-radius:8px;padding:10px 14px;"
                "font-size:13px;color:#2e7d32;margin-top:8px'>✅ No major risk factors identified</div>",
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH SCAN
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("<div class='section-label'>Upload Student Records</div>", unsafe_allow_html=True)
    st.caption("Upload a CSV file with student records to run predictions at scale.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.markdown(
            f"<div style='font-size:13px;color:#2e7d32;font-weight:600;margin:6px 0'>"
            f"📂 {len(df_batch):,} records loaded</div>",
            unsafe_allow_html=True
        )
        st.dataframe(df_batch.head(5), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Run Batch Analysis", type="primary", use_container_width=True):
            try:
                db = df_batch.copy()
                db['fees_paid'] = db['fees_paid'].fillna('Unknown')
                for col in ['family_income', 'study_hours', 'attendance_rate']:
                    db[col] = db[col].fillna(db[col].median())
                db['gender']        = (db['gender'] == 'Female').astype(int)
                db['location']      = (db['location'] == 'Rural').astype(int)
                db['part_time_job'] = (db['part_time_job'] == 'Yes').astype(int)
                db['internet_access']         = db['internet_access'].map(INTERNET_MAP)
                db['electricity_reliability'] = db['electricity_reliability'].map(ELEC_MAP)
                db['fees_paid']               = db['fees_paid'].map(FEES_MAP)
                db['previous_grade']          = db['previous_grade'].map(GRADE_MAP)
                scale_cols = ['age','family_income','transport_time','study_hours',
                              'attendance_rate','lms_logins','stress_level']
                db[scale_cols] = scaler.transform(db[scale_cols])
                db['study_efficiency'] = db['study_hours'] / (db['attendance_rate'] + 0.01)
                db['hardship_index']   = (
                    (1 - db['family_income']) +
                    (2 - db['electricity_reliability']) +
                    (2 - db['internet_access'])
                )
                probas = model.predict_proba(db[features])[:, 1]
                preds  = (probas >= 0.5).astype(int)
                df_batch['dropout_probability'] = probas.round(4)
                df_batch['prediction']          = preds
                df_batch['risk_tier']           = pd.cut(
                    probas, bins=[0, 0.4, 0.65, 1.0],
                    labels=['LOW', 'MEDIUM', 'HIGH']
                )
                n_high = (df_batch['risk_tier'] == 'HIGH').sum()
                n_med  = (df_batch['risk_tier'] == 'MEDIUM').sum()
                n_low  = (df_batch['risk_tier'] == 'LOW').sum()

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",  f"{len(df_batch):,}")
                c2.metric("🔴 High",  f"{n_high:,}")
                c3.metric("🟠 Med",   f"{n_med:,}")
                c4.metric("🟢 Low",   f"{n_low:,}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(
                    df_batch[['dropout_probability','prediction','risk_tier']].head(20),
                    use_container_width=True
                )
                csv_out = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇  Download Full Results",
                    data=csv_out,
                    file_name="dropout_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Processing error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SYSTEM INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("<div class='section-label'>Model Configuration</div>", unsafe_allow_html=True)
    config = {
        "Algorithm":        "Decision Tree Classifier",
        "Max Depth":        "10",
        "Random State":     "7",
        "Train Split":      "80%",
        "Test Split":       "20%",
        "Stratified":       "Yes",
        "Scaler":           "MinMaxScaler",
        "Features":         "16 (incl. 2 engineered)",
        "Training Records": "418,859",
        "Test Records":     "104,715",
        "Dataset Total":    "523,574",
    }
    for k, v in config.items():
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:8px 0;border-bottom:1px solid #e8f5e9;font-size:13px'>"
            f"<span style='color:#555'>{k}</span>"
            f"<span style='font-weight:600;color:#2e7d32'>{v}</span></div>",
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("<div class='section-label'>Features</div>", unsafe_allow_html=True)
    feat_data = [
        ("age",                     "Numeric",     "Student age"),
        ("gender",                  "Binary",      "0=Male, 1=Female"),
        ("location",                "Binary",      "0=Urban, 1=Rural"),
        ("family_income",           "Numeric",     "Monthly income (USD)"),
        ("internet_access",         "Ordinal 0-2", "Poor/Moderate/Good"),
        ("electricity_reliability", "Ordinal 0-2", "Low/Medium/High"),
        ("transport_time",          "Numeric",     "Hours to campus"),
        ("study_hours",             "Numeric",     "Daily study hours"),
        ("attendance_rate",         "Numeric",     "Class attendance %"),
        ("lms_logins",              "Numeric",     "Logins per month"),
        ("previous_grade",          "Ordinal 0-4", "F/1/2.1/2.2/3"),
        ("fees_paid",               "Ordinal 0-2", "Unknown/Partial/Full"),
        ("part_time_job",           "Binary",      "0=No, 1=Yes"),
        ("stress_level",            "Numeric",     "Self-reported 1–10"),
        ("study_efficiency",        "✨ Engineered","study_hours / attendance"),
        ("hardship_index",          "✨ Engineered","Composite deprivation score"),
    ]
    for fname, ftype, fdesc in feat_data:
        eng = ftype.startswith("✨")
        tag_bg = "#e8f5e9" if eng else "#f5f5f5"
        tag_col = "#2e7d32" if eng else "#666"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:7px 0;"
            f"border-bottom:1px solid #f0f4f0;font-size:12px'>"
            f"<span style='font-weight:600;color:#1a2e1a;min-width:150px'>{fname}</span>"
            f"<span style='background:{tag_bg};color:{tag_col};padding:2px 7px;"
            f"border-radius:20px;font-size:10px;font-weight:600'>{ftype}</span>"
            f"<span style='color:#888;margin-left:auto'>{fdesc}</span></div>",
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("<div class='section-label'>Local Setup</div>", unsafe_allow_html=True)
    st.code("""pip install streamlit scikit-learn pandas numpy matplotlib joblib

# Files required in same folder:
# app_v3.py  dropout_model.pkl  scaler.pkl  features.pkl

streamlit run app_v3.py""", language="bash")


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:20px 0 8px 0;
            font-size:11px;color:#a5d6a7;font-weight:500'>
  HCS221 · Great Zimbabwe University · Dept. of Mathematics & Computer Science
</div>
""", unsafe_allow_html=True)
