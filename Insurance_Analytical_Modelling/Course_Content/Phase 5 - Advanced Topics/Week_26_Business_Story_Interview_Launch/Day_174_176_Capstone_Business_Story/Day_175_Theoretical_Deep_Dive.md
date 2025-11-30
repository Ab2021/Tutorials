# Capstone Project: Business Story (Part 2) - Visuals & Dashboards - Theoretical Deep Dive

## Overview
"A picture is worth a thousand rows of data."
Executives don't read code. They read dashboards.
Day 175 focuses on **Data Visualization** and **Dashboard Design**.
We explore how to build interactive apps with **Streamlit**, design principles for **Tableau**, and how to visualize **Uncertainty** without confusing the stakeholder.

---

## 1. Conceptual Foundation

### 1.1 The 5-Second Rule

*   **Principle:** A user should understand the *primary message* of a dashboard within 5 seconds.
*   **Test:** Show the dashboard to a colleague. Take it away after 5 seconds. Ask: "Is the business doing well or poorly?"
*   **Failure:** "I don't know, I was trying to read the legend."

### 1.2 The "F" Pattern

*   **Eye Tracking:** Western readers scan screens in an "F" shape.
    *   **Top Left:** Most important KPI (e.g., Loss Ratio).
    *   **Top Right:** Filters (Date Range).
    *   **Bottom:** Detailed tables.
*   **Application:** Don't bury the lead in the bottom right corner.

---

## 2. Mathematical Framework

### 2.1 Visualizing Uncertainty

*   **Problem:** A point estimate ($\hat{y} = 500$) implies false precision.
*   **Solution:** **Confidence Intervals**.
    *   **Visual:** Shaded region around the line chart (e.g., `plt.fill_between`).
    *   **Interpretation:** "We are 95% sure the loss will be between \$4M and \$6M."

### 2.2 The Data-Ink Ratio (Tufte)

$$ Ratio = \frac{Data \ Ink}{Total \ Ink} $$
*   **Goal:** Maximize the ratio.
*   **Remove:** Gridlines, 3D effects, background colors, redundant labels.
*   **Keep:** The bars, the lines, the axes.

---

## 3. Theoretical Properties

### 3.1 Interactive vs. Static

*   **Static (PowerPoint):** Good for "Narrative" (guiding the audience through a specific story).
*   **Interactive (Streamlit):** Good for "Exploration" (letting the user answer their own questions).
*   **Capstone Strategy:** Use Static for the Presentation, Interactive for the Q&A.

### 3.2 Color Theory in Insurance

*   **Red:** Danger/Loss. (High Loss Ratio, Churn).
*   **Green:** Good/Profit. (Growth, Retention).
*   **Blue/Grey:** Neutral/Context.
*   **Accessibility:** Ensure colorblind-safe palettes (e.g., Viridis).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Streamlit Risk Dashboard (`dashboard.py`)

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Underwriting Portfolio Dashboard")

# Sidebar Filters
segment = st.sidebar.selectbox("Segment", ["Personal Auto", "Commercial"])
loss_threshold = st.sidebar.slider("Loss Ratio Threshold", 0.0, 2.0, 1.0)

# Data (Mock)
df = pd.DataFrame({'Region': ['North', 'South'], 'Loss_Ratio': [0.85, 1.2]})

# KPI Cards
col1, col2 = st.columns(2)
col1.metric("Total Premium", "$10M", "+5%")
col2.metric("Avg Loss Ratio", "92%", "-2%")

# Interactive Chart
fig = px.bar(df, x='Region', y='Loss_Ratio', 
             color='Loss_Ratio', 
             color_continuous_scale=['green', 'red'])
st.plotly_chart(fig)
```

### 4.2 Visualizing SHAP Values

*   **Tool:** `shap.plots.waterfall`.
*   **Context:** Explaining *one* prediction.
*   **Visual:** Red bars push risk up, Blue bars push risk down.
*   **Caption:** "This driver is high risk mainly because of 'Prior Accidents' (Red Bar)."

---

## 5. Evaluation & Validation

### 5.1 The "Ink" Audit

*   **Check:** Print your dashboard. Circle every element that *doesn't* convey data.
*   **Action:** Delete them.

### 5.2 Performance Profiling

*   **Issue:** Dashboard takes 10 seconds to load.
*   **Cause:** Calculating aggregations on the fly in Python.
*   **Fix:** Pre-aggregate data in SQL/Pandas. `df.groupby('Region').sum()`.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Spaghetti" Chart

*   **Mistake:** Plotting 50 lines on one chart (e.g., Loss Ratio by State).
*   **Result:** Unreadable mess.
*   **Fix:** **Small Multiples** (Facet Grid). 50 small charts, one for each state.

### 6.2 Truncated Y-Axis

*   **Mistake:** Starting the Y-axis at 90% to make a 1% change look huge.
*   **Ethics:** This is lying with data.
*   **Rule:** Bar charts must start at 0. Line charts can zoom in *if clearly labeled*.

---

## 7. Advanced Topics & Extensions

### 7.1 Geospatial Visualization (Kepler.gl)

*   **Use Case:** Hurricane Risk.
*   **Visual:** Hexbin map of exposure.
*   **Layer:** Overlay "Storm Path" on top of "Policy Locations".

### 7.2 Real-Time Streaming Dashboards

*   **Stack:** Kafka -> Streamlit.
*   **Use:** Monitoring "Quotes per Minute" during the Super Bowl ad.
*   **Implementation:** `st.empty()` placeholder that updates every second.

---

## 8. Regulatory & Governance Considerations

### 8.1 PII Masking

*   **Rule:** Never show raw PII (Name, SSN) on a dashboard unless necessary.
*   **Implementation:** Aggregate to Zip Code level.

---

## 9. Practical Example

### 9.1 The "Bad" vs. "Good" Dashboard

**Bad Dashboard:**
*   3D Pie Chart of "Policy Types". (Hard to compare angles).
*   Red/Green background colors everywhere. (Eye fatigue).
*   Table with 50 rows and no sorting.

**Good Dashboard:**
*   Bar Chart of "Policy Types" (Sorted).
*   Clean white background. Red used *only* for Loss Ratio > 100%.
*   "Top 5 Worst Regions" table.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Streamlit** enables rapid prototyping.
2.  **Tufte's Principles** ensure clarity.
3.  **Interactivity** empowers the user.

### 10.2 When to Use This Knowledge
*   **Capstone:** The "Demo" portion of your presentation.
*   **Work:** Building tools for Underwriters/Claims Adjusters.

### 10.3 Critical Success Factors
1.  **Speed:** If it's slow, they won't use it.
2.  **Relevance:** Does it answer the business question?

### 10.4 Further Reading
*   **Tufte:** "The Visual Display of Quantitative Information".
*   **Knaflic:** "Storytelling with Data".

---

## Appendix

### A. Glossary
*   **KPI:** Key Performance Indicator.
*   **Facet Grid:** Breaking a chart into multiple smaller charts by category.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Data-Ink** | $Ink_{data} / Ink_{total}$ | Design Efficiency |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
