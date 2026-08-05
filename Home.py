"""
Repo_11_Predictive_Maintenance — Home.py
Author : Mohamed · M3
"""
import pathlib
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance · M3", page_icon="🏭", layout="wide")
LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🏭 Predictive Maintenance")
    st.markdown("M3 · ML Engine · P11")
    st.divider()
    st.markdown("**Navigate:**")
    st.markdown("📊 EDA Dashboard → 13 tabs")
    st.markdown("🤖 ML Models     → 5 tabs")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f4f7fb;}
.hero{background:linear-gradient(135deg,#1a237e,#c62828);
      padding:48px 40px;border-radius:14px;margin-bottom:28px;}
.hero h1{color:#ffffff !important;font-size:2.4rem;font-weight:800;margin:0 0 8px 0;}
.hero p{color:#ffcdd2 !important;font-size:1.08rem;margin:0;}
.card{background:#ffffff;border-radius:10px;padding:22px 24px;
      box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:4px solid #c62828;}
.card h3{color:#c62828 !important;margin:0 0 8px 0;font-size:1.05rem;}
.card p{color:#37474f !important;font-size:0.92rem;margin:0;line-height:1.6;}
.stat-card{background:#ffffff;border-radius:10px;padding:18px;text-align:center;
           box-shadow:0 2px 10px rgba(0,0,0,0.07);border-bottom:3px solid #c62828;}
.stat-num{font-size:1.9rem;font-weight:800;color:#c62828 !important;}
.stat-lbl{font-size:0.82rem;color:#546e7a !important;margin-top:4px;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏭 Predictive Maintenance Analysis</h1>
  <p>End-to-end ML pipeline · 10,000 machine readings · AI4I 2020 Dataset · M3 Portfolio · Project 11 of 12</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
for col, (num, lbl) in zip([c1,c2,c3,c4,c5],[
    ("10,000","Readings"), ("3.4%","Failure Rate"),
    ("5","Failure Modes"), ("13","EDA Tabs"), ("12","ML Models")]):
    col.markdown(f"""<div class="stat-card">
      <div class="stat-num">{num}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📌 About This Project")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="card"><h3>🎯 Objective</h3>
    <p>Predict machine failures before they occur using process parameters:
    temperature, speed, torque, and tool wear. Identify which failure mode
    is most likely and optimise maintenance scheduling.</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card"><h3>📊 Dataset</h3>
    <p>AI4I 2020 Predictive Maintenance · 10,000 synthetic machine readings ·
    5 failure modes: TWF, HDF, PWF, OSF, RNF. Engineered:
    temp_diff, power_proxy, strain_index, wear_category.</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card"><h3>🔑 Key Signals</h3>
    <p>Tool wear progression · Temperature differential ·
    Mechanical strain (high torque + low speed) ·
    Power consumption proxy · Product type quality level.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    st.markdown("### 📈 EDA Dashboard — 13 Tabs")
    for num, name, desc in [
        ("1","Data Overview","Shape, types, stats, dictionary"),
        ("2","Failure Analysis ★","Overall rate + by type + failure modes"),
        ("3","Tool Wear Progression ★","Wear vs failure probability"),
        ("4","Process Parameters ★","Temp · Speed · Torque distributions"),
        ("5","Failure Mode Breakdown ★","TWF/HDF/PWF/OSF/RNF deep dive"),
        ("6","OEE Analysis ★","Overall Equipment Effectiveness proxy"),
        ("7","Multicollinearity","VIF analysis"),
        ("8","Correlation","Heatmap + top failure predictors"),
        ("9","Business KPIs ★","Downtime cost · maintenance ROI"),
        ("10","Category Deep-Dive ★","Type × wear × strain heatmaps"),
        ("11","Statistical Tests ★","T1-T4: failed vs non-failed parameters"),
        ("12","Feature Engineering","Engineered features + distributions"),
        ("13","Insights & Report","Findings + recommendations + download"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

with col5:
    st.markdown("### 🤖 ML Models — 5 Tabs")
    for num, name, desc in [
        ("1","Model Training","6 Reg + 6 Clf · individual buttons"),
        ("2","Regression Results","R², MAE, RMSE · predict Torque"),
        ("3","Classification Results","F1, Recall, ROC-AUC · predict failure"),
        ("4","Feature Importance","Top failure predictors"),
        ("5","Predict","Interactive failure risk scorer"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("**Failure Rate: 3.4%** — SEVERE imbalance.\n\n"
               "All classifiers use `class_weight='balanced'`.\n\n"
               "Evaluate with **F1, Recall, ROC-AUC**.\n\n"
               "Never use accuracy — 96.6% achieved by predicting all Normal!")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#90a4ae;font-size:0.85rem;'>"
            "Mohamed · M3 · ML Engine Portfolio · Project 11 of 12 · Predictive Maintenance</p>",
            unsafe_allow_html=True)
