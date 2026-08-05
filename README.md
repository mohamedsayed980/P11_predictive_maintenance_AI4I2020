# 🏭 P11 — Predictive Maintenance Analysis & Failure Prediction
**M3 · ML Engine Portfolio · Project 11 of 12**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Source-UCI_AI4I_2020-0052CC)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
[![Domain](https://img.shields.io/badge/Domain-Manufacturing-FF6B35)](https://github.com)

---

## 📌 Project Overview

End-to-end predictive maintenance analysis and ML failure prediction on **10,000 synthetic machine readings** from the AI4I 2020 dataset. The goal is to predict machine failures before they occur, identify the dominant failure mode, and optimise maintenance scheduling.

**Core Questions:**
- Which process parameters best predict machine failure?
- At what tool wear threshold does failure risk spike?
- Is mechanical strain (high torque + low speed) a reliable early warning signal?
- What is the ROI of switching from corrective to preventive maintenance?

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository · AI4I 2020 |
| Raw Records | 10,000 machine readings |
| Original Features | 14 |
| After Engineering | 22 |
| Failure Rate | 3.4% (339 failures) |

### Failure Modes (Independent Binary Columns)

| Code | Name | Count | Rate |
|------|------|-------|------|
| HDF | Heat Dissipation Failure | 115 | 1.15% |
| OSF | Overstrain Failure | 98 | 0.98% |
| PWF | Power Deviation Failure | 95 | 0.95% |
| TWF | Tool Wear Failure | 46 | 0.46% |
| RNF | Random Failure | 19 | 0.19% |

> **Note:** Failure modes are INDEPENDENT — a machine can have multiple modes simultaneously. Total mode events (373) > total failures (339) because 34 cases had 2 simultaneous failure modes.

---

## 🎯 Targets

| Type | Column | Description |
|------|--------|-------------|
| **Regression** | `Torque [Nm]` | Applied process torque (3.8–76.6 N·m) |
| **Classification** | `Machine failure` | 1 if failure occurred, 0 if normal |

**Balance:** 96.6% Normal / 3.4% Failure → **SEVERE IMBALANCE**

`class_weight='balanced'` **MANDATORY** on ALL classifiers

Evaluate with **F1, Recall, ROC-AUC** — NEVER accuracy

---

## ⚙️ Feature Engineering

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `Type_enc` | L=0 / M=1 / H=2 | Ordinal quality encoding |
| `temp_diff` | Process_temp − Air_temp | Heat dissipation signal |
| `power_proxy` | Torque × RPM / 9550 | Mechanical shaft power (kW) |
| `wear_category` | pd.cut(wear, 5 bins) | Non-linear wear effect |
| `high_wear` | wear > 200 min | Critical maintenance threshold |
| `high_torque` | Torque > Q75 (46.8 N·m) | Overload detector |
| `low_speed` | RPM < Q25 (1,423 rpm) | Stalling detector |
| `strain_index` | high_torque AND low_speed | **9.3× failure multiplier** ★ |
| `overheat` | Process_temp > Q90 | Thermal alarm flag |
| `failure_mode_count` | TWF+HDF+PWF+OSF+RNF | Failure severity indicator |

### Engineering Insights

**power_proxy** = Torque × RPM / 9550
- Standard mechanical engineering formula: P(kW) = T × N / 9550
- Combines two separate sensors into one meaningful load signal

**strain_index** = high_torque AND low_speed
- Physics: if speed drops but load stays same → torque must rise → maximum mechanical stress
- Result: **12.5% failure rate** vs 1.35% normal → **9.3× multiplier**

---

## 📊 EDA Dashboard — 13 Tabs

| Tab | Title | Highlight |
|-----|-------|-----------|
| 1 | Data Overview | Shape, types, stats, dictionary |
| 2 | Failure Analysis ★ | Overall rate + by type + failure modes |
| 3 | Tool Wear Progression ★ | Wear vs failure probability — clear threshold at 200 min |
| 4 | Process Parameters ★ | Temp · Speed · Torque distributions |
| 5 | Failure Mode Breakdown ★ | TWF/HDF/PWF/OSF/RNF deep dive + root causes |
| 6 | OEE Analysis ★ | Overall Equipment Effectiveness proxy |
| 7 | Multicollinearity | VIF analysis |
| 8 | Correlation | Heatmap + top failure predictors |
| 9 | Business KPIs ★ | Downtime cost · corrective vs preventive ROI |
| 10 | Category Deep-Dive ★ | Type × wear × strain heatmaps |
| 11 | Statistical Tests ★ | T1-T4: torque, wear, temp, power vs failure |
| 12 | Feature Engineering | Engineered features + distributions |
| 13 | Insights & Report | Findings + recommendations + download |

---

## 🤖 ML Models — 5 Tabs

| Tab | Content |
|-----|---------|
| 1 | Training — 6 Reg + 6 Clf · individual buttons |
| 2 | Regression Results — R², MAE, RMSE · predict Torque |
| 3 | Classification Results — F1, Recall, ROC-AUC · predict failure |
| 4 | Feature Importance — top failure predictors |
| 5 | Interactive Predict — real-time failure risk scorer |

---

## 🔑 Key Findings

**1. Tool Wear is the Primary Actionable Signal**
Tool wear > 200 min shows 15.5% failure rate vs 2.4% normal — 6.5× higher risk. Clear replacement threshold.

**2. Mechanical Strain = Strongest Engineered Feature**
High torque + low speed simultaneously → 9.3× higher failure rate. Best early warning signal.

**3. HDF is the Most Common Failure Mode**
Heat Dissipation accounts for 1.15% of all readings — thermal management is priority #1.

**4. 34 Multi-Mode Failures Exist**
373 total failure mode events vs 339 failures → 34 cases had 2 failure modes simultaneously — compounding effects.

**5. Product Quality Matters Under Stress**
H-type (High quality) shows lower failure rates — quality components withstand process extremes better.

---

## 💡 Maintenance Recommendations

| Priority | Action |
|----------|--------|
| 🔴 Critical | Replace tools at **180 min** — 20 min safety margin before critical threshold |
| 🔴 Critical | Alert when **strain_index = 1** persists > 10 consecutive readings |
| 🟡 High | Alert when **temp_diff < 8.5K** — heat dissipation degrading |
| 🟡 High | Alert when **power_proxy > 8.5 kW** — machine overloading |
| 🟢 Medium | Deploy ML model for real-time failure probability scoring |
| 🟢 Medium | Preventive maintenance every 150 min tool wear for L-type products |

### Business Case
```
Corrective maintenance cost : $8,000 per failure × 339 failures = $2,712,000
Preventive maintenance cost : $500  per event  × 339 events    = $169,500
Potential savings            : $2,542,500
Maintenance ROI              : 1,501%
```

---

## 🗂 Project Structure

```
📁 Repo_11_Predictive_Maintenance/
├── Home.py
├── M3_logo.png
├── requirements.txt
├── README.md
├── data/
│   └── maintenance_clean.csv       ← from P11_clean_data.py (Jupyter)
└── pages/
    ├── EDA_dashboard.py             ← 13-tab analysis
    └── ML_Models.py                 ← 5-tab ML engine
```

---

## 🚀 How to Run

```bash
git clone https://github.com/YourUsername/Repo_11_Predictive_Maintenance.git
cd Repo_11_Predictive_Maintenance

pip install -r requirements.txt

# Step 1: Generate clean dataset in Jupyter
# Run P11_clean_data.py → saves maintenance_clean.csv
# Copy maintenance_clean.csv to data/ folder (do NOT open in Excel)

# Step 2: Launch app
streamlit run Home.py
```

> ⚠️ **Important:** Copy CSV directly via File Explorer — never open in Excel before copying.

---

## 🛠 Tech Stack

`Python 3.11` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Scikit-learn` · `SciPy` · `Statsmodels` · `Psutil`

---

**Mohamed · M3 · ML Engine Portfolio — 12 End-to-End Data Science Projects**
