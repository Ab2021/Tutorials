# Time Series Forecasting (Part 2) - Theoretical Deep Dive

## Overview
ARIMA is great, but it struggles with "Black Friday" spikes and non-linear trends. Today, we explore modern forecasting tools: **Facebook Prophet** and **Neural Time Series (LSTM, N-BEATS)**.

---

## 1. Conceptual Foundation

### 1.1 Facebook Prophet

*   **Philosophy:** Forecasting as a curve-fitting problem, not an AR/MA problem.
*   **Components:**
    *   **Trend:** Piecewise linear (can change direction at "changepoints").
    *   **Seasonality:** Fourier Series (smooth curves for weekly/yearly cycles).
    *   **Holidays:** Explicit list of dates (e.g., "Super Bowl", "Christmas").
*   **Pros:** Handles missing data and outliers gracefully. Very interpretable.

### 1.2 Neural Time Series (LSTM)

*   **Idea:** Treat time series like a sentence.
    *   Sentence: "The cat sat on the..." -> "mat".
    *   Series: "100, 110, 105, 120..." -> "125".
*   **Mechanism:** The LSTM cell remembers long-term dependencies (e.g., a cycle that repeats every 5 years).

### 1.3 N-BEATS (Neural Basis Expansion Analysis)

*   **State-of-the-Art:** Beat the M4 competition winner.
*   **Architecture:** A stack of fully connected layers.
    *   **Trend Block:** Learns polynomial curves.
    *   **Seasonality Block:** Learns periodic waves.
*   **Benefit:** Pure Deep Learning, but interpretable (you can plot the Trend block output separately).

---

## 2. Mathematical Framework

### 2.1 Prophet's Additive Model

$$ y(t) = g(t) + s(t) + h(t) + \epsilon_t $$
*   $g(t)$: Trend function (Logistic or Linear).
*   $s(t)$: Seasonality (Fourier Series: $\sum a_n \cos(2\pi nt/P) + b_n \sin(2\pi nt/P)$).
*   $h(t)$: Holiday effects (Indicator functions).

### 2.2 LSTM Cell State

$$ C_t = f_t \times C_{t-1} + i_t \times \tilde{C}_t $$
*   The new memory ($C_t$) is a mix of old memory ($C_{t-1}$) and new information ($\tilde{C}_t$).
*   The "Forget Gate" ($f_t$) decides what to throw away.

---

## 3. Theoretical Properties

### 3.1 Changepoints (Prophet)

*   Prophet automatically detects when the trend changes (e.g., COVID-19 hit).
*   **Regularization:** You can control how flexible the trend is. Too flexible = Overfitting.

### 3.2 Global vs. Local Models

*   **ARIMA:** Local (trained on one series).
*   **N-BEATS:** Global (can be trained on 10,000 different series at once to learn "general" time series patterns).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Prophet (Python)

```python
from prophet import Prophet
import pandas as pd

# 1. Prepare Data (Must be 'ds' and 'y')
df = pd.DataFrame({
    'ds': pd.to_datetime(['2023-01-01', '2023-01-02', ...]),
    'y': [100, 110, ...]
})

# 2. Add Holidays
holidays = pd.DataFrame({
    'holiday': 'super_bowl',
    'ds': pd.to_datetime(['2023-02-12', '2024-02-11']),
    'lower_window': 0,
    'upper_window': 1,
})

# 3. Fit Model
m = Prophet(holidays=holidays)
m.fit(df)

# 4. Forecast
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
m.plot(forecast)
```

### 4.2 LSTM for Time Series (PyTorch)

```python
# See Day 81 for LSTM code structure.
# Key difference: Input is (Batch, Sequence_Length, 1).
# Output is (Batch, 1).
```

---

## 5. Evaluation & Validation

### 5.1 Backtesting (Cross-Validation)

*   **Rolling Window:**
    *   Train on Jan-Mar, Test on Apr.
    *   Train on Jan-Apr, Test on May.
*   **Prophet Built-in:** `cross_validation(m, horizon='30 days')`.

### 5.2 Metrics for Spiky Data

*   **RMSE** is sensitive to outliers.
*   **MAE (Mean Absolute Error)** is more robust.
*   **WAPE (Weighted MAPE):** $\sum |y - \hat{y}| / \sum y$. Good for volume forecasting.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "COVID" Drop**
    *   If you train a model on 2019-2021, it learns that "March 2020" is a normal seasonal pattern.
    *   *Fix:* Mark COVID period as an outlier or use Prophet's changepoints to model the drop explicitly.

2.  **Trap: Prophet's Daily Seasonality**
    *   Prophet assumes daily patterns are constant.
    *   If "Monday" behavior changes in Winter vs. Summer, Prophet struggles.
    *   *Fix:* Add "Conditional Seasonality".

### 6.2 Implementation Challenges

1.  **Installation:**
    *   Prophet depends on `pystan`, which is notoriously hard to install on Windows.
    *   *Fix:* Use `conda install -c conda-forge prophet`.

---

## 7. Advanced Topics & Extensions

### 7.1 Neural Prophet

*   A hybrid library (PyTorch based).
*   Combines Prophet's interpretability with Neural Network's flexibility (AR-Net).

### 7.2 Probabilistic Forecasting

*   Don't just predict "100 calls". Predict "80 to 120 calls (95% CI)".
*   Prophet gives uncertainty intervals by simulating trend changes.
*   DeepAR (Amazon) gives probabilistic forecasts using RNNs.

---

## 8. Regulatory & Governance Considerations

### 8.1 Staffing Models

*   **Risk:** Under-forecasting call volume -> Long wait times -> Regulatory fine.
*   **Governance:** The forecast must be signed off. "Human-in-the-loop" adjustment (e.g., "Marketing is running a TV ad next week, add 10%").

---

## 9. Practical Example

### 9.1 Worked Example: The Call Center

**Scenario:**
*   Predict daily calls for next month.
*   **Challenge:** "Memorial Day" shifts every year. ARIMA fails.
*   **Prophet:**
    *   Added `country_holidays(country_name='US')`.
    *   Model learned that Memorial Day = -90% volume.
*   **Result:** Staffing accuracy improved from 80% to 95%.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Prophet** is best for business cycles + holidays.
2.  **LSTM/N-BEATS** are best for complex, high-frequency data.
3.  **Backtesting** is non-negotiable.

### 10.2 When to Use This Knowledge
*   **Operations:** Call center, Claims processing volume.
*   **Marketing:** Website traffic.

### 10.3 Critical Success Factors
1.  **Holiday List:** Maintain an accurate list of business holidays.
2.  **Changepoints:** Don't let the model over-react to noise.

### 10.4 Further Reading
*   **Taylor & Letham:** "Forecasting at Scale" (The Prophet Paper).
*   **Oreshkin et al.:** "N-BEATS: Neural basis expansion analysis".

---

## Appendix

### A. Glossary
*   **Changepoint:** A point in time where the trend slope changes.
*   **Fourier Series:** A sum of sine and cosine waves to approximate a periodic function.
*   **Horizon:** How far into the future we are predicting.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Prophet** | $y = g + s + h$ | Additive Model |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
