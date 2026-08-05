"""
Repo_11_Predictive_Maintenance — EDA_dashboard.py  (13 Tabs)
Author : Mohamed · M3
Dataset: AI4I 2020 Predictive Maintenance · 10,000 machine readings
"""
import os, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

warnings.filterwarnings("ignore")
S = st.session_state

st.set_page_config(page_title="EDA · Predictive Maintenance · M3",
                   page_icon="🏭", layout="wide")

LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"

# ── PATH (Windows-safe) ──────────────────────────────────────
_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "maintenance_clean.csv"
)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🏭 EDA Dashboard")
    st.markdown("Predictive Maintenance · 13 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    st.success("✅ maintenance_clean.csv")
    st.caption("Loaded from data/ folder")

CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","light":"#e3f2fd","dark":"#1a237e",
       "purple":"#6a1b9a","amber":"#f57f17","grey":"#546e7a"}

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f4f7fb;}
div[data-testid="metric-container"]{background:#fce4ec;border-left:4px solid #c62828;border-radius:6px;padding:10px 14px;}
.sec-header{background:linear-gradient(90deg,#c62828,#1565c0);color:#ffffff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}
.insight-box{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.insight-box p{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}
.warn-box{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.warn-box p{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}
.info-box{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.info-box p{color:#0d2a4a !important;margin:0;font-size:0.93rem;line-height:1.6;}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)

# ── LOAD ─────────────────────────────────────────────────────
if not os.path.exists(_data_path):
    st.error(f"❌ File not found: {_data_path}")
    st.info("Run P11_clean_data.py in Jupyter → copy maintenance_clean.csv to data/ folder")
    st.stop()

try:
    df = pd.read_csv(_data_path, sep=",", decimal=".")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"❌ Load error: {e}")
    st.stop()

if df.empty:
    st.warning("⚠️ Dataset is empty.")
    st.stop()

S["df_work"] = df

TARGET  = "Machine failure"
REG_T   = "Torque [Nm]"
FAILURE_MODES = ["TWF","HDF","PWF","OSF","RNF"]
FAILURE_NAMES = {
    "TWF":"Tool Wear Failure",
    "HDF":"Heat Dissipation Failure",
    "PWF":"Power Deviation Failure",
    "OSF":"Overstrain Failure",
    "RNF":"Random Failure"
}

df_fail   = df[df[TARGET] == 1]
df_normal = df[df[TARGET] == 0]
fail_rate = df[TARGET].mean() * 100

NUM_COLS = [c for c in ["Air temperature [K]","Process temperature [K]",
                         "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]",
                         "temp_diff","power_proxy"] if c in df.columns]

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Data Overview",
    "2 · Failure Analysis ★",
    "3 · Tool Wear Progression ★",
    "4 · Process Parameters ★",
    "5 · Failure Mode Breakdown ★",
    "6 · OEE Analysis ★",
    "7 · Multicollinearity",
    "8 · Correlation",
    "9 · Business KPIs ★",
    "10 · Category Deep-Dive ★",
    "11 · Statistical Tests ★",
    "12 · Feature Engineering",
    "13 · Insights & Report",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("📋 Tab 1 — Data Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Readings", f"{len(df):,}")
    c2.metric("Failures",       f"{len(df_fail):,}")
    c3.metric("Normal",         f"{len(df_normal):,}")
    c4.metric("Failure Rate",   f"{fail_rate:.1f}%")
    c5.metric("Features",       f"{df.shape[1]}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📄 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        sec("📐 Column Info")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  df.dtypes.astype(str).values,
            "Nulls":  df.isnull().sum().values,
        })
        st.dataframe(info_df, use_container_width=True)

    st.markdown("---")
    sec("📊 Descriptive Statistics")
    st.dataframe(df[NUM_COLS].describe().round(4), use_container_width=True)

    st.markdown("---")
    sec("🗂 Data Dictionary")
    dd = pd.DataFrame({
        "Column":      ["Type","Air temperature [K]","Process temperature [K]",
                        "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]",
                        "Machine failure","TWF","HDF","PWF","OSF","RNF",
                        "Type_enc","temp_diff","power_proxy","wear_category",
                        "high_wear","high_torque","low_speed","strain_index",
                        "overheat","failure_mode_count"],
        "Type":        ["Categorical","Numeric","Numeric","Numeric","Numeric","Numeric",
                        "Target","Failure Mode","Failure Mode","Failure Mode",
                        "Failure Mode","Failure Mode",
                        "Engineered","Engineered","Engineered","Engineered",
                        "Engineered","Engineered","Engineered","Engineered",
                        "Engineered","Engineered"],
        "Description": [
            "Product quality variant: L=Low / M=Medium / H=High",
            "Ambient air temperature (Kelvin) — mean ~300K",
            "Process temperature (Kelvin) — mean ~310K",
            "Machine rotational speed (RPM) — mean ~1539 rpm",
            "Applied torque (Newton-meters) — REGRESSION TARGET",
            "Cumulative tool wear time (minutes) — 0 to 253 min",
            "1=failure occurred / 0=normal — CLASSIFICATION TARGET",
            "Tool Wear Failure (0.46%)",
            "Heat Dissipation Failure (1.15%) — most common",
            "Power Deviation Failure (0.95%)",
            "Overstrain Failure (0.98%)",
            "Random Failure (0.19%)",
            "L=0 / M=1 / H=2 — ordinal quality encoding",
            "Process temp − Air temp (K) — heat management signal",
            "Torque × RPM / 9550 — mechanical power in kW",
            "Tool wear bins: Fresh/Low/Medium/High/Critical",
            "1 if Tool wear > 200 min (critical threshold)",
            "1 if Torque > Q75 (46.8 N·m)",
            "1 if RPM < Q25 (1,423 rpm)",
            "1 if high_torque AND low_speed — mechanical strain",
            "1 if Process temp > Q90 — overheating flag",
            "Sum of all 5 failure mode columns (0–5)",
        ]
    })
    st.dataframe(dd, use_container_width=True)
    warn(f"Failure rate {fail_rate:.1f}% — SEVERE imbalance → class_weight='balanced' mandatory → evaluate with F1/Recall/AUC only")

# ══════════════════════════════════════════════════════════════
# TAB 2 — FAILURE ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("🚨 Tab 2 — Failure Analysis ★")
    info("3.4% failure rate — understanding WHERE failures concentrate is the key business question.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Overall Balance")
        bal = pd.DataFrame({"Label":["Normal","Failure"],
                            "Count":[len(df_normal), len(df_fail)]})
        bal["Pct"] = (bal["Count"]/len(df)*100).round(2)
        fig = px.bar(bal, x="Label", y="Count", color="Label",
                     color_discrete_map={"Normal":CLR["primary"],"Failure":CLR["danger"]},
                     text=bal["Pct"].apply(lambda x: f"{x:.1f}%"),
                     title="Normal vs Failure")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=370, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Failure Rate by Product Type")
        if "Type" in df.columns:
            type_fail = df.groupby("Type")[TARGET].agg(
                Total="count",Fail="sum").reset_index()
            type_fail["Rate%"] = (type_fail["Fail"]/type_fail["Total"]*100).round(2)
            type_fail = type_fail.sort_values("Rate%", ascending=False)
            fig2 = px.bar(type_fail, x="Type", y="Rate%",
                          color="Rate%",
                          color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                          title="Failure Rate % by Product Type",
                          text=type_fail["Rate%"].apply(lambda x: f"{x:.2f}%"))
            fig2.add_hline(y=fail_rate, line_dash="dash", line_color="blue",
                           annotation_text=f"Avg {fail_rate:.1f}%")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=370)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Failure Mode Breakdown")
    mode_data = pd.DataFrame({
        "Mode": list(FAILURE_NAMES.values()),
        "Code": list(FAILURE_NAMES.keys()),
        "Count": [df[c].sum() for c in FAILURE_NAMES.keys()],
        "Rate%": [round(df[c].mean()*100, 2) for c in FAILURE_NAMES.keys()]
    }).sort_values("Count", ascending=False)

    col3, col4 = st.columns(2)
    with col3:
        st.dataframe(mode_data, use_container_width=True)
    with col4:
        fig3 = px.bar(mode_data, x="Code", y="Rate%",
                      color="Rate%",
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Failure Rate % by Mode",
                      text=mode_data["Rate%"].apply(lambda x: f"{x:.2f}%"))
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    insight("HDF (Heat Dissipation Failure) is the most common failure mode — thermal management is the #1 priority.")
    insight("The 5 failure modes are INDEPENDENT — a machine can have multiple modes simultaneously.")
    warn("3.4% failure rate — never use accuracy metric. A model predicting all Normal gets 96.6% accuracy but catches ZERO failures.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — TOOL WEAR PROGRESSION ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("🔧 Tab 3 — Tool Wear Progression ★")
    info("Tool wear is the most actionable predictor — it's directly controllable through maintenance scheduling.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Failure Rate by Tool Wear Bin")
        if "wear_category" in df.columns:
            wc = df.groupby("wear_category", observed=True)[TARGET].agg(
                Total="count",Fail="sum").reset_index()
            wc["Rate%"] = (wc["Fail"]/wc["Total"]*100).round(2)
            fig = px.bar(wc, x="wear_category", y="Rate%",
                         color="Rate%",
                         color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                         title="Failure Rate % by Tool Wear Category",
                         text=wc["Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig.add_hline(y=fail_rate, line_dash="dash", line_color="blue",
                          annotation_text=f"Avg {fail_rate:.1f}%")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Tool Wear Distribution — Failed vs Normal")
        fig2, ax = plt.subplots(figsize=(7,4))
        ax.hist(df_normal["Tool wear [min]"], bins=40, alpha=0.6,
                color=CLR["primary"], density=True, label="Normal")
        ax.hist(df_fail["Tool wear [min]"],   bins=40, alpha=0.7,
                color=CLR["danger"],  density=True, label="Failure")
        ax.axvline(200, color=CLR["warning"], lw=2, ls="--", label="Critical=200min")
        ax.set_xlabel("Tool Wear (min)"); ax.set_ylabel("Density")
        ax.set_title("Tool Wear: Failed vs Normal"); ax.legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    c1.metric("Failed — Avg Wear",  f"{df_fail['Tool wear [min]'].mean():.0f} min")
    c2.metric("Normal — Avg Wear",  f"{df_normal['Tool wear [min]'].mean():.0f} min")
    c3.metric("Critical Threshold", "200 min → 15.5% failure rate")

    insight("Tool wear > 200 min shows 15.5% failure rate vs 2.4% for normal wear — 6.5× higher risk.")
    insight("Clear threshold at 200 min — replace tools before this point to prevent failures.")
    warn("Tool wear is directly actionable — schedule replacement at 180 min to maintain safety margin.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — PROCESS PARAMETERS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("⚙️ Tab 4 — Process Parameters ★")
    info("Understanding normal operating ranges and failure zones for each process parameter.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Air Temp",     f"{df['Air temperature [K]'].mean():.1f} K")
    c2.metric("Avg Process Temp", f"{df['Process temperature [K]'].mean():.1f} K")
    c3.metric("Avg Speed",        f"{df['Rotational speed [rpm]'].mean():.0f} rpm")
    c4.metric("Avg Torque",       f"{df['Torque [Nm]'].mean():.1f} N·m")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Torque Distribution — Failed vs Normal")
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(df_normal["Torque [Nm]"], bins=40, alpha=0.6,
                color=CLR["primary"], density=True, label="Normal")
        ax.hist(df_fail["Torque [Nm]"],   bins=40, alpha=0.7,
                color=CLR["danger"],  density=True, label="Failure")
        ax.axvline(df_normal["Torque [Nm]"].mean(), color=CLR["primary"], lw=2, ls="--")
        ax.axvline(df_fail["Torque [Nm]"].mean(),   color=CLR["danger"],  lw=2, ls="--")
        ax.set_xlabel("Torque (N·m)"); ax.set_ylabel("Density")
        ax.set_title("Torque: Failed vs Normal"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Speed Distribution — Failed vs Normal")
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.hist(df_normal["Rotational speed [rpm]"], bins=40, alpha=0.6,
                 color=CLR["primary"], density=True, label="Normal")
        ax2.hist(df_fail["Rotational speed [rpm]"],   bins=40, alpha=0.7,
                 color=CLR["danger"],  density=True, label="Failure")
        ax2.set_xlabel("Rotational Speed (rpm)"); ax2.set_ylabel("Density")
        ax2.set_title("Speed: Failed vs Normal"); ax2.legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Torque vs Speed — Scatter (Failure Highlighted)")
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig3 = px.scatter(sample, x="Rotational speed [rpm]", y="Torque [Nm]",
                      color=TARGET, opacity=0.5,
                      color_discrete_map={0:CLR["primary"], 1:CLR["danger"]},
                      labels={TARGET:"Failure"},
                      title="Speed vs Torque — Failure Zone Visible")
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

    insight(f"Failed machines have higher torque (mean={df_fail['Torque [Nm]'].mean():.1f}) vs normal (mean={df_normal['Torque [Nm]'].mean():.1f} N·m).")
    insight("Low speed + high torque zone = mechanical strain → highest failure concentration visible in scatter.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — FAILURE MODE BREAKDOWN ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("💥 Tab 5 — Failure Mode Breakdown ★")
    info("Each failure mode has a different root cause — understanding them drives targeted maintenance actions.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Failure Mode Counts")
        mode_df = pd.DataFrame({
            "Mode": [FAILURE_NAMES[c] for c in FAILURE_MODES],
            "Code": FAILURE_MODES,
            "Count": [df[c].sum() for c in FAILURE_MODES],
            "Rate%": [round(df[c].mean()*100,2) for c in FAILURE_MODES]
        }).sort_values("Count", ascending=True)
        fig = px.bar(mode_df, y="Code", x="Count", orientation="h",
                     color="Count",
                     color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                     title="Failure Count by Mode",
                     text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, yaxis_title="Failure Mode")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Failure Mode by Product Type")
        type_mode = df.groupby("Type")[FAILURE_MODES].sum().reset_index()
        fig2 = go.Figure()
        colors_m = [CLR["danger"],CLR["warning"],CLR["amber"],CLR["primary"],CLR["grey"]]
        for i, mode in enumerate(FAILURE_MODES):
            fig2.add_trace(go.Bar(name=FAILURE_NAMES[mode],
                                  x=type_mode["Type"],
                                  y=type_mode[mode],
                                  marker_color=colors_m[i]))
        fig2.update_layout(barmode="group", height=350,
                           title="Failure Modes by Product Type")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Failure Mode Descriptions & Root Causes")
    root_causes = {
        "TWF": ("Tool Wear Failure", "Tool wear exceeds threshold → tool breaks during operation",
                "Replace tools before wear limit · monitor wear rate"),
        "HDF": ("Heat Dissipation Failure", "Temperature difference (Process - Air) too low → overheating",
                "Improve cooling · monitor temp_diff · reduce load at high temps"),
        "PWF": ("Power Deviation Failure", "Mechanical power (Torque × Speed) outside safe range",
                "Monitor power_proxy · set safe operating envelope limits"),
        "OSF": ("Overstrain Failure", "Torque × Tool wear product exceeds limit by product type",
                "Reduce torque as tool wears · use strain_index early warning"),
        "RNF": ("Random Failure", "Random failure unrelated to process parameters",
                "Cannot be predicted from parameters — schedule preventive maintenance"),
    }
    for code, (name, cause, action) in root_causes.items():
        count = df[code].sum()
        st.markdown(f"**{code} — {name}** ({count} cases · {count/10:.2f}%)")
        st.markdown(f'<div class="info-box"><p>📍 Root cause: {cause}<br>🔧 Action: {action}</p></div>',
                    unsafe_allow_html=True)

    insight("HDF is most common — thermal management is the primary maintenance priority.")
    warn("RNF (Random Failure) cannot be predicted from process parameters — only preventive scheduling helps.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — OEE ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("📈 Tab 6 — OEE Analysis ★ (Overall Equipment Effectiveness)")
    info("OEE = Availability × Performance × Quality. Failures directly reduce Availability and Quality components.")

    # OEE proxy calculation
    TOTAL_READINGS  = len(df)
    FAILED_READINGS = df[TARGET].sum()
    NORMAL_READINGS = TOTAL_READINGS - FAILED_READINGS

    availability = NORMAL_READINGS / TOTAL_READINGS * 100
    performance  = (df["Rotational speed [rpm]"].mean() /
                    df["Rotational speed [rpm]"].max() * 100)
    quality      = availability
    oee          = (availability / 100) * (performance / 100) * (quality / 100) * 100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Availability",  f"{availability:.1f}%", help="% of time machine is operational")
    c2.metric("Performance",   f"{performance:.1f}%",  help="Actual vs max speed ratio")
    c3.metric("Quality",       f"{quality:.1f}%",      help="Non-defective output rate")
    c4.metric("OEE Score",     f"{oee:.1f}%",          help="World class OEE = 85%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 OEE Waterfall")
        fig, ax = plt.subplots(figsize=(7,4))
        components = ['Availability','Performance','Quality','OEE']
        values     = [availability, performance, quality, oee]
        colors_oee = [CLR["success"] if v >= 85 else CLR["warning"] if v >= 70
                      else CLR["danger"] for v in values]
        bars = ax.bar(components, values, color=colors_oee, edgecolor="white")
        ax.axhline(85, color=CLR["success"], lw=2, ls="--", label="World class (85%)")
        ax.set_ylabel("OEE Component (%)")
        ax.set_title("OEE Components", fontweight="bold")
        ax.set_ylim(0, 110)
        ax.legend()
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, val+1,
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Downtime Analysis by Failure Mode")
        downtime_data = pd.DataFrame({
            "Failure Mode": [FAILURE_NAMES[c] for c in FAILURE_MODES],
            "Incidents": [df[c].sum() for c in FAILURE_MODES],
        }).sort_values("Incidents", ascending=False)
        downtime_data["Downtime%"] = (downtime_data["Incidents"]/TOTAL_READINGS*100).round(2)
        fig2 = px.pie(downtime_data, names="Failure Mode", values="Incidents",
                      color_discrete_sequence=[CLR["danger"],CLR["warning"],
                                               CLR["amber"],CLR["primary"],CLR["grey"]],
                      title="Downtime Share by Failure Mode", hole=0.4)
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    insight(f"Current OEE: {oee:.1f}% — below world-class target of 85%. Primary cause: availability loss from failures.")
    warn("Every 1% improvement in availability = 100 additional operational readings per 10,000 total.")

# ══════════════════════════════════════════════════════════════
# TAB 7 — MULTICOLLINEARITY
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("🔁 Tab 7 — Multicollinearity / VIF")
    info("VIF > 10 = severe multicollinearity. Expected between temperature columns and derived features.")

    vif_cols = [c for c in ["Air temperature [K]","Process temperature [K]",
                             "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]",
                             "temp_diff","power_proxy","high_wear","strain_index","overheat"]
                if c in df.columns]
    vif_data = df[vif_cols].dropna()
    try:
        vif_df = pd.DataFrame({
            "Feature": vif_cols,
            "VIF": [round(variance_inflation_factor(vif_data.values,i),2)
                    for i in range(len(vif_cols))]
        }).sort_values("VIF", ascending=False)
        vif_df["Risk"] = vif_df["VIF"].apply(
            lambda v: "🔴 High" if v>10 else "🟡 Medium" if v>5 else "🟢 Low")

        col1, col2 = st.columns([1,1.5])
        with col1:
            st.dataframe(vif_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7,5))
            colors_vif = [CLR["danger"] if v>10 else CLR["warning"] if v>5
                          else CLR["success"] for v in vif_df["VIF"]]
            ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors_vif)
            ax.axvline(10, color=CLR["danger"],  lw=2, ls="--", label="VIF=10")
            ax.axvline(5,  color=CLR["warning"], lw=1.5, ls=":",  label="VIF=5")
            ax.set_xlabel("VIF"); ax.set_title("Multicollinearity Check")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()
    except Exception as e:
        warn(f"VIF error: {e}")

    warn("Air temp and Process temp are correlated — use temp_diff as single feature for linear models.")
    insight("Tree-based models (RF, GB) handle multicollinearity automatically — VIF only matters for Logistic Regression.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — CORRELATION
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("🔥 Tab 8 — Correlation Analysis")

    num_cols_corr = [c for c in df.select_dtypes(include=np.number).columns
                     if c not in FAILURE_MODES]
    corr = df[num_cols_corr].corr()

    fig, ax = plt.subplots(figsize=(13,9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size":7})
    ax.set_title("Correlation Matrix — Predictive Maintenance Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sec("🎯 Top Correlations with Machine Failure")
    tgt = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(12)
    fig2, ax2 = plt.subplots(figsize=(10,5))
    colors_bar = [CLR["danger"] if v>0 else CLR["success"] for v in tgt.values]
    ax2.barh(tgt.index, tgt.values, color=colors_bar)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Pearson r with Machine Failure")
    ax2.set_title("Feature Correlation with Machine Failure", fontsize=12, fontweight="bold")
    for i,(idx,val) in enumerate(tgt.items()):
        ax2.text(val+0.005 if val>=0 else val-0.005, i,
                 f"{val:.3f}", va="center",
                 ha="left" if val>=0 else "right", fontsize=9)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("Torque shows positive correlation — higher torque = more strain = more failures.")
    insight("Rotational speed shows negative correlation — lower speed = stalling = failure risk.")
    warn("Pearson r misses non-linear effects — use RF/GB feature importance for full picture.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — BUSINESS KPIs ★
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💼 Tab 9 — Business KPIs ★")
    info("Translate failure statistics into business metrics: downtime cost, maintenance ROI, savings potential.")

    DOWNTIME_COST_PER_FAILURE = 5000
    PREVENTIVE_COST           = 500
    CORRECTIVE_COST           = 8000

    total_failures    = df[TARGET].sum()
    current_cost      = total_failures * CORRECTIVE_COST
    preventive_cost   = total_failures * PREVENTIVE_COST
    savings           = current_cost - preventive_cost
    roi_maintenance   = savings / preventive_cost * 100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Failures",      f"{total_failures:,}")
    c2.metric("Corrective Cost",     f"${current_cost:,.0f}")
    c3.metric("Preventive Cost",     f"${preventive_cost:,.0f}")
    c4.metric("Potential Savings",   f"${savings:,.0f}")

    st.markdown("---")
    sec("📊 Cost Comparison: Corrective vs Preventive Maintenance")
    cost_data = pd.DataFrame({
        "Strategy":     ["Corrective\n(current)", "Preventive\n(proposed)"],
        "Total Cost":   [current_cost, preventive_cost],
        "Cost per Unit":[CORRECTIVE_COST, PREVENTIVE_COST]
    })
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(cost_data, x="Strategy", y="Total Cost",
                     color="Strategy",
                     color_discrete_map={"Corrective\n(current)":CLR["danger"],
                                         "Preventive\n(proposed)":CLR["success"]},
                     title="Total Maintenance Cost",
                     text=cost_data["Total Cost"].apply(lambda x: f"${x:,.0f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(pd.DataFrame({
            "Metric":      ["Total Failures","Corrective Cost","Preventive Cost",
                            "Savings","Maintenance ROI","Cost Reduction"],
            "Value":       [f"{total_failures:,}",f"${current_cost:,.0f}",
                            f"${preventive_cost:,.0f}",f"${savings:,.0f}",
                            f"{roi_maintenance:,.0f}%",
                            f"{(savings/current_cost*100):.0f}%"]
        }), use_container_width=True)

    insight(f"Switching to preventive maintenance saves ${savings:,.0f} — {roi_maintenance:,.0f}% ROI.")
    insight("Cost per failure: $8,000 corrective vs $500 preventive — 16× cheaper to prevent than repair.")
    warn("These are estimates — actual savings depend on downtime duration, production rates, and part costs.")

# ══════════════════════════════════════════════════════════════
# TAB 10 — CATEGORY DEEP-DIVE ★
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    sec("🔎 Tab 10 — Category Deep-Dive ★")
    info("Cross-tabulation: which combinations of conditions produce highest failure rates?")

    sec("📊 Tool Wear Category × Product Type — Failure Rate Heatmap")
    if "wear_category" in df.columns and "Type" in df.columns:
        heat = df.groupby(["Type","wear_category"], observed=True)[TARGET].mean().unstack() * 100
        fig, ax = plt.subplots(figsize=(10,4))
        sns.heatmap(heat.round(1), annot=True, fmt=".1f", cmap="RdYlGn_r",
                    ax=ax, linewidths=0.5, annot_kws={"size":10})
        ax.set_title("Failure Rate % — Product Type × Tool Wear Category",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Strain Index × Wear Category")
        if "strain_index" in df.columns and "wear_category" in df.columns:
            heat2 = df.groupby(["strain_index","wear_category"],
                                observed=True)[TARGET].mean().unstack() * 100
            heat2.index = ["Normal Strain","High Strain"]
            fig2, ax2 = plt.subplots(figsize=(7,3))
            sns.heatmap(heat2.round(1), annot=True, fmt=".1f", cmap="RdYlGn_r",
                        ax=ax2, linewidths=0.5, annot_kws={"size":10})
            ax2.set_title("Failure Rate % — Strain × Wear", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    with col2:
        sec("📊 Overheat × High Torque")
        if "overheat" in df.columns and "high_torque" in df.columns:
            heat3 = df.groupby(["overheat","high_torque"])[TARGET].mean().unstack() * 100
            heat3.index = ["Normal Temp","Overheat"]
            heat3.columns = ["Normal Torque","High Torque"]
            fig3, ax3 = plt.subplots(figsize=(6,3))
            sns.heatmap(heat3.round(1), annot=True, fmt=".1f", cmap="RdYlGn_r",
                        ax=ax3, linewidths=0.5, annot_kws={"size":12})
            ax3.set_title("Failure Rate % — Temp × Torque", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Overheat + High Torque + Critical Wear = worst combination → failure rate spikes dramatically.")
    insight("L-type products in critical wear zone show highest failure rate — quality matters under stress.")

# ══════════════════════════════════════════════════════════════
# TAB 11 — STATISTICAL TESTS ★
# ══════════════════════════════════════════════════════════════
with tabs[10]:
    sec("🧪 Tab 11 — Statistical Tests ★")
    info("4 tests validating the most important process-failure relationships.")

    def run_test(name, gA, gB, labelA, labelB):
        t_stat, p_val = stats.ttest_ind(gA.dropna(), gB.dropna(), equal_var=False)
        pooled   = np.sqrt((gA.std()**2 + gB.std()**2) / 2)
        cohens_d = (gA.mean() - gB.mean()) / (pooled + 1e-10)
        res = pd.DataFrame({
            "Metric": ["Test","H₀","Group A","Group B",
                       "A Mean","B Mean","t-stat","p-value",
                       "Significant","Cohen's d","Effect","Decision"],
            "Result": [
                "Welch T-Test", "No difference between failed/normal",
                f"{labelA} (n={len(gA):,})", f"{labelB} (n={len(gB):,})",
                f"{gA.mean():.4f}", f"{gB.mean():.4f}",
                f"{t_stat:.4f}", f"{p_val:.6f}",
                "✅ YES" if p_val < 0.05 else "❌ NO",
                f"{cohens_d:.4f}",
                "Large" if abs(cohens_d)>0.8 else "Medium" if abs(cohens_d)>0.5 else "Small",
                "✅ REJECT H₀" if p_val < 0.05 else "❌ FAIL to reject H₀"
            ]
        })
        return res, p_val, cohens_d

    # T1: Torque
    sec("T1 — Torque: Do failed machines have significantly different torque?")
    r1, p1, d1 = run_test("T1", df_fail["Torque [Nm]"], df_normal["Torque [Nm]"],
                           "Failed","Normal")
    col1, col2 = st.columns([1.2,1])
    with col1: st.dataframe(r1, use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(df_normal["Torque [Nm]"], bins=40, alpha=0.6,
                color=CLR["primary"], density=True, label="Normal")
        ax.hist(df_fail["Torque [Nm]"],   bins=40, alpha=0.7,
                color=CLR["danger"],  density=True, label="Failed")
        ax.set_title("T1: Torque Distribution"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
    if p1 < 0.05:
        insight(f"T1: Failed machines have significantly different torque (d={d1:.3f}) — torque is a strong failure signal.")

    # T2: Tool Wear
    st.markdown("---")
    sec("T2 — Tool Wear: Is tool wear significantly higher in failed machines?")
    r2, p2, d2 = run_test("T2", df_fail["Tool wear [min]"], df_normal["Tool wear [min]"],
                           "Failed","Normal")
    st.dataframe(r2, use_container_width=True)
    if p2 < 0.05:
        insight(f"T2: Tool wear is significantly higher in failed machines (d={d2:.3f}) — confirms wear→failure link.")

    # T3: Temperature Difference
    st.markdown("---")
    sec("T3 — Temp Difference: Is temp_diff significantly different in failed machines?")
    if "temp_diff" in df.columns:
        r3, p3, d3 = run_test("T3", df_fail["temp_diff"], df_normal["temp_diff"],
                               "Failed","Normal")
        st.dataframe(r3, use_container_width=True)
        if p3 < 0.05:
            insight(f"T3: Temperature differential differs in failed machines (d={d3:.3f}) — thermal signal confirmed.")

    # T4: Power Proxy
    st.markdown("---")
    sec("T4 — Power Proxy: Is mechanical power significantly higher in failed machines?")
    if "power_proxy" in df.columns:
        r4, p4, d4 = run_test("T4", df_fail["power_proxy"], df_normal["power_proxy"],
                               "Failed","Normal")
        st.dataframe(r4, use_container_width=True)
        if p4 < 0.05:
            insight(f"T4: Power proxy differs significantly in failed machines (d={d4:.3f}) — overloading confirmed.")

# ══════════════════════════════════════════════════════════════
# TAB 12 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
with tabs[11]:
    sec("⚙️ Tab 12 — Feature Engineering")

    fe = pd.DataFrame({
        "Feature":     ["Type_enc","temp_diff","power_proxy","wear_category",
                        "high_wear","high_torque","low_speed","strain_index",
                        "overheat","failure_mode_count"],
        "Formula":     ["L=0 / M=1 / H=2","Process_temp − Air_temp",
                        "Torque × RPM / 9550","pd.cut(wear, 5 bins)",
                        "wear > 200 min","Torque > Q75 (46.8 N·m)",
                        "RPM < Q25 (1423 rpm)","high_torque AND low_speed",
                        "Process_temp > Q90","TWF+HDF+PWF+OSF+RNF"],
        "Failure Rate":["H < L (quality helps)","Lower diff = overheating risk",
                        "Higher power = more strain","Critical > 15%",
                        "15.5% vs 2.4% normal","Part of strain composite",
                        "Part of strain composite","12.5% vs 1.4% normal",
                        "Higher temp = HDF risk","Multi-mode = highest risk"],
        "Reason":      ["Ordinal quality for ML","Single thermal signal",
                        "Combined mechanical load","Non-linear wear effect",
                        "Critical maintenance threshold","Overload detector",
                        "Stalling detector","9.3× failure multiplier ★",
                        "Thermal alarm flag","Failure severity indicator"],
    })
    st.dataframe(fe, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 strain_index vs Normal — Failure Rate")
        if "strain_index" in df.columns:
            si = df.groupby("strain_index")[TARGET].mean() * 100
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(["Normal Strain","High Strain"], si.values,
                   color=[CLR["success"], CLR["danger"]], edgecolor="white")
            ax.set_ylabel("Failure Rate %")
            ax.set_title("strain_index Impact", fontweight="bold")
            for i, v in enumerate(si.values):
                ax.text(i, v+0.3, f"{v:.1f}%", ha="center", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 power_proxy Distribution")
        if "power_proxy" in df.columns:
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.hist(df_normal["power_proxy"], bins=40, alpha=0.6,
                     color=CLR["primary"], density=True, label="Normal")
            ax2.hist(df_fail["power_proxy"], bins=40, alpha=0.7,
                     color=CLR["danger"], density=True, label="Failed")
            ax2.set_xlabel("Power Proxy (kW)")
            ax2.set_title("Power Proxy: Failed vs Normal")
            ax2.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("strain_index is the most powerful engineered feature — 9.3× failure multiplier.")
    insight("power_proxy = Torque × RPM / 9550 — standard mechanical engineering formula for shaft power.")

# ══════════════════════════════════════════════════════════════
# TAB 13 — INSIGHTS & REPORT
# ══════════════════════════════════════════════════════════════
with tabs[12]:
    sec("💡 Tab 13 — Insights & Recommendations")

    st.markdown(f"### 🏭 Predictive Maintenance — Final Report")
    st.markdown(f"**{len(df):,} readings · {fail_rate:.1f}% failure rate · AI4I 2020 · M3**")
    st.markdown("---")

    sec("1️⃣ Top Failure Drivers")
    insight("Tool wear > 200 min — 15.5% failure rate (6.5× normal). Replace at 180 min.")
    insight("Mechanical strain (high torque + low speed) — 9.3× higher failure rate.")
    insight("HDF most common — heat dissipation failures dominate (1.15% of all readings).")
    insight("High torque alone is insufficient signal — combined with low speed is the real danger zone.")

    sec("2️⃣ Failure Mode Summary")
    for code, (name, cause, action) in {
        "HDF": ("Heat Dissipation", "Temp differential too low → overheating","Improve cooling system"),
        "OSF": ("Overstrain", "Torque × wear exceeds type-specific limit","Monitor strain_index daily"),
        "PWF": ("Power Deviation", "Mechanical power outside safe envelope","Set power_proxy alerts"),
        "TWF": ("Tool Wear", "Tool reaches critical wear threshold","Replace at 180 min threshold"),
        "RNF": ("Random", "Unrelated to process parameters","Preventive schedule only"),
    }.items():
        st.markdown(f'<div class="info-box"><p><b>{code} — {name}:</b> {cause} → {action}</p></div>',
                    unsafe_allow_html=True)

    sec("3️⃣ Maintenance Recommendations")
    recs = [
        ("🔧 Tool Replacement Protocol", "Replace tools at 180 min wear — 20 min safety margin before critical threshold."),
        ("🌡 Thermal Monitoring", "Alert when temp_diff < 8.5K — heat dissipation degrading."),
        ("⚡ Power Envelope", "Alert when power_proxy > 8.5 kW — machine overloading."),
        ("🔄 Strain Detection", "Trigger inspection when strain_index=1 persists > 10 consecutive readings."),
        ("📊 ML Deployment", "Deploy Gradient Boosting model for real-time failure probability scoring."),
        ("📅 Preventive Schedule", "Schedule maintenance every 150 min of tool wear for L-type products."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""PREDICTIVE MAINTENANCE — FINAL REPORT
M3 · AI4I 2020 · {len(df):,} Machine Readings
Failure Rate: {fail_rate:.1f}% | class_weight='balanced' MANDATORY

TOP FAILURE DRIVERS:
1. Tool wear > 200 min → 15.5% failure rate (6.5x normal)
2. Mechanical strain (high torque + low speed) → 9.3x higher
3. HDF most common failure mode (1.15%)
4. High power_proxy → overloading condition

FAILURE MODES:
- TWF (0.46%): Tool Wear → replace at 180 min
- HDF (1.15%): Heat Dissipation → improve cooling
- PWF (0.95%): Power Deviation → monitor power_proxy
- OSF (0.98%): Overstrain → monitor strain_index
- RNF (0.19%): Random → preventive schedule only

RECOMMENDATIONS:
- Replace tools at 180 min (safety margin before 200 min critical)
- Alert when temp_diff < 8.5K
- Alert when power_proxy > 8.5 kW
- Inspect when strain_index=1 persists > 10 readings
- Deploy ML model for real-time failure scoring
- Preventive maintenance every 150 min tool wear (L-type)
"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="PredictiveMaintenance_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        # Machine failure summary by tool wear — value-added analysis output
        wear_summary = df.groupby("wear_category", observed=True).agg(
            Total_Readings  = ("Machine failure", "count"),
            Failures        = ("Machine failure", "sum"),
            Avg_Torque      = ("Torque [Nm]",     "mean"),
            Avg_Speed       = ("Rotational speed [rpm]", "mean"),
            Avg_Power_kW    = ("power_proxy",     "mean"),
        ).round(3).reset_index()
        wear_summary["Failure_Rate%"] = (
            wear_summary["Failures"] / wear_summary["Total_Readings"] * 100).round(2)
        st.download_button("📥 Wear vs Failure Analysis (.csv)",
                           wear_summary.to_csv(index=False),
                           file_name="WearFailure_Analysis_M3.csv",
                           mime="text/csv", use_container_width=True)
    with col3:
        # Failure mode breakdown — actionable maintenance KPI
        mode_summary = pd.DataFrame({
            "Failure_Mode":  ["TWF","HDF","PWF","OSF","RNF"],
            "Full_Name":     ["Tool Wear","Heat Dissipation","Power Deviation",
                              "Overstrain","Random"],
            "Count":         [df[m].sum() for m in ["TWF","HDF","PWF","OSF","RNF"]],
            "Rate%":         [round(df[m].mean()*100,2) for m in ["TWF","HDF","PWF","OSF","RNF"]],
            "Avg_Torque_Failed": [df[df[m]==1]["Torque [Nm]"].mean().round(2)
                                  for m in ["TWF","HDF","PWF","OSF","RNF"]],
        })
        st.download_button("📥 Failure Mode KPIs (.csv)",
                           mode_summary.to_csv(index=False),
                           file_name="FailureMode_KPIs_M3.csv",
                           mime="text/csv", use_container_width=True)

    st.markdown("---")
    sec("📊 Preview — Tool Wear vs Failure Analysis")
    st.dataframe(wear_summary.style
                 .background_gradient(subset=["Failure_Rate%"], cmap="RdYlGn_r")
                 .format({"Avg_Torque":"{:.2f}","Avg_Speed":"{:.0f}",
                          "Avg_Power_kW":"{:.3f}","Failure_Rate%":"{:.2f}%"}),
                 use_container_width=True)
