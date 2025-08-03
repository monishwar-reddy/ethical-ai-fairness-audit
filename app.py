import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate
)

# Page config
st.set_page_config(page_title="Fairness Audit", layout="wide")
st.title("🧠 Ethical AI Fairness Dashboard")

st.markdown("Upload any HR-related dataset (CSV) and analyze **bias in salary/performance** based on gender or other sensitive attributes.")

# Upload Dataset
st.sidebar.header("Upload your HR CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    df = pd.read_csv("HRDataset_v14.csv")
    st.sidebar.info("Using default HRDataset_v14.csv")

# Preview data
st.markdown("### Data Preview")
st.dataframe(df.head())

# ========== SMART COLUMN MAPPING ==========
st.sidebar.header("Column Mapping")

col_options = df.columns.tolist()

salary_col = st.sidebar.selectbox("Select Salary Column", ["-- Select --"] + col_options)
gender_col = st.sidebar.selectbox("Select Gender Column", ["-- Select --"] + col_options)
perf_col = st.sidebar.selectbox("Select Performance Score Column", ["-- Select --"] + col_options)
absent_col = st.sidebar.selectbox("Select Absences Column", ["-- Select --"] + col_options)

if "-- Select --" in [salary_col, gender_col, perf_col, absent_col]:
    st.warning("⚠️ Please select all required columns to proceed.")
    st.stop()

# ========== PREPROCESSING ==========
# Normalize gender values if numeric
if df[gender_col].dtype in [int, float]:
    unique_vals = df[gender_col].unique()
    if len(unique_vals) == 2:
        mapping = {min(unique_vals): "Female", max(unique_vals): "Male"}
        df[gender_col] = df[gender_col].map(mapping)
    else:
        st.error("⚠️ Gender column must be binary (e.g., Male/Female or 0/1).")
        st.stop()

# Create high salary label based on median
median_salary = df[salary_col].median()
df["HighSalary"] = (df[salary_col] > median_salary).astype(int)

# ========== VISUALIZATION ==========
st.subheader("Visual Analysis")

# Salary by Gender
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.barplot(x=gender_col, y=salary_col, data=df, estimator=np.mean, ci=None, ax=ax1, palette="pastel")
ax1.set_title("Average Salary by Gender")
st.pyplot(fig1)

# Performance Score by Gender
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.countplot(x=perf_col, hue=gender_col, data=df, palette="muted", ax=ax2)
ax2.set_title("Performance Score by Gender")
st.pyplot(fig2)

# Absences by Gender
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.boxplot(x=gender_col, y=absent_col, data=df, ax=ax3)
ax3.set_title("Absences by Gender")
st.pyplot(fig3)

# ========== FAIRNESS METRICS ==========
st.subheader("Fairness Metric Calculation")

y_true = df["HighSalary"]
y_pred = df["HighSalary"]  # Auditing actual outcome
sensitive_feature = df[gender_col]

metric_frame = MetricFrame(
    metrics={
        "Selection Rate": selection_rate,
        "TPR": true_positive_rate,
        "FPR": false_positive_rate,
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)

st.markdown("#### Metric Summary by Gender")
st.dataframe(metric_frame.by_group)

dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)

st.markdown(f"**Demographic Parity Difference:** `{dpd:.3f}`")
st.markdown(f"**Equalized Odds Difference:** `{eod:.3f}`")

# ========== ETHICAL REPORT ==========
st.subheader("Auto-Generated Ethical Fairness Report")

if st.button("Generate Report"):
    report_md = f"""
# Ethical AI Fairness Audit Report

### Dataset Info
- File: `{uploaded_file.name if uploaded_file else 'Default HRDataset'}`
- Records: `{len(df)}`

### Selected Columns
- Salary: `{salary_col}`
- Gender: `{gender_col}`
- Performance: `{perf_col}`
- Absences: `{absent_col}`

### Fairness Metrics
- Demographic Parity Difference: `{dpd:.3f}`
- Equalized Odds Difference: `{eod:.3f}`

#### By Gender Group:
{metric_frame.by_group.to_markdown()}

---

*This audit was generated automatically.*
    """
    st.download_button("Download Report", report_md, file_name="fairness_audit_report.md")

# ========== DEBIASING HINTS ==========
st.subheader("Debiasing Suggestions")

if abs(dpd) > 0.1 or abs(eod) > 0.1:
    st.warning("⚠️ Significant bias detected. Consider:")
    st.markdown("""
- **Preprocessing**: Normalize or balance gender ratios.
- **Reweighing**: Adjust training weights.
- **Adversarial Debiasing**: Use debiasing networks.
- **Post-processing**: Use fairness calibration.
    """)
else:
    st.success("Metrics show acceptable fairness. No urgent debiasing needed.")

# Footer
st.markdown("---")
st.caption("Built by Monishwar Reddy Vardireddy | 2025")
