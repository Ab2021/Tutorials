# Time Series Forecasting (Part 1) - Theoretical Deep Dive

## Overview
Actuaries live in the future. We predict next year's claims, next decade's mortality, and next century's climate risk. This session covers the classical Time Series tools: **ARIMA** and **Holt-Winters**.

---

## 1. Conceptual Foundation

### 1.1 Components of a Time Series

1.  **Trend:** Long-term increase or decrease (e.g., Medical Inflation).
2.  **Seasonality:** Repeating patterns (e.g., More accidents in Winter, fewer in Summer).
3.  **Cyclicality:** Economic cycles (Recessions).
4.  **Noise (Residuals):** Random variation.

### 1.2 Stationarity

*   **Definition:** A time series is stationary if its Mean, Variance, and Autocorrelation are constant over time.
*   **Why it matters:** ARIMA assumes stationarity. You cannot model a trend directly with AR/MA terms; you must remove the trend first.
*   **Differencing:** Subtracting $y_t - y_{t-1}$ removes the trend.

### 1.3 Models

1.  **ARIMA (AutoRegressive Integrated Moving Average):**
    *   **AR (p):** Current value depends on past values ($y_{t-1}, y_{t-2}$).
    *   **I (d):** Differencing order (to make it stationary).
    *   **MA (q):** Current value depends on past *errors* ($\epsilon_{t-1}$).
2.  **Holt-Winters (Exponential Smoothing):**
    *   Weighted average of past observations, where recent observations get more weight.
    *   Explicitly models Level, Trend, and Seasonality.

---

## 2. Mathematical Framework

### 2.1 ARIMA(p,d,q) Equation

$$ y'_t = c + \phi_1 y'_{t-1} + \dots + \phi_p y'_{t-p} + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q} + \epsilon_t $$
*   $y'_t$: Differenced series (if $d=1$).
*   $\phi$: AR coefficients.
*   $\theta$: MA coefficients.

### 2.2 Augmented Dickey-Fuller (ADF) Test

*   **Null Hypothesis ($H_0$):** The series has a Unit Root (Non-Stationary).
*   **Alternative ($H_1$):** The series is Stationary.
*   *Rule:* If p-value < 0.05, reject $H_0$. The series is Stationary.

---

## 3. Theoretical Properties

### 3.1 Seasonality in ARIMA (SARIMA)

*   Standard ARIMA doesn't handle seasonality well.
*   **SARIMA(p,d,q)(P,D,Q)m:** Adds seasonal terms.
    *   $m$: Season length (e.g., 12 for monthly data).
    *   $D$: Seasonal differencing ($y_t - y_{t-12}$).

### 3.2 Additive vs. Multiplicative (Holt-Winters)

*   **Additive:** Seasonality is constant amount (e.g., +100 claims in Dec).
*   **Multiplicative:** Seasonality is a percentage (e.g., +10% claims in Dec).
*   *Actuarial Insight:* Claims are usually Multiplicative (as exposure grows, the seasonal swing grows).

---

## 4. Modeling Artifacts & Implementation

### 4.1 ARIMA with Statsmodels

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# 1. Load Data
data = pd.read_csv('monthly_claims.csv', index_col='Date', parse_dates=True)
y = data['ClaimCount']

# 2. Check Stationarity
result = adfuller(y)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# 3. Fit ARIMA (1, 1, 1)
model = ARIMA(y, order=(1, 1, 1))
model_fit = model.fit()

# 4. Forecast
forecast = model_fit.forecast(steps=12)
print(forecast)
```

### 4.2 Holt-Winters

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit Model (Multiplicative Seasonality)
hw_model = ExponentialSmoothing(
    y,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Forecast
hw_forecast = hw_model.forecast(12)
```

---

## 5. Evaluation & Validation

### 5.1 Residual Analysis

*   **White Noise:** The residuals of a good model should look like random noise (Mean 0, Constant Variance, No Autocorrelation).
*   **Ljung-Box Test:** Tests if residuals are independent. (p-value > 0.05 is good).

### 5.2 Metrics

*   **MAPE (Mean Absolute Percentage Error):** "We were off by 5%".
*   **RMSE:** Penalizes large errors.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Over-differencing**
    *   If you difference a stationary series, you introduce artificial correlations.
    *   *Fix:* Don't difference if ADF p-value < 0.05.

2.  **Trap: Ignoring Calendar Effects**
    *   "February has fewer claims".
    *   Is it because February is safer? No, it has fewer days (28).
    *   *Fix:* Always normalize by "Exposure Days" or "Business Days" before modeling.

### 6.2 Implementation Challenges

1.  **Grid Search:**
    *   Finding the best (p,d,q) is hard.
    *   *Fix:* Use `auto_arima` (from `pmdarima` library) which loops through combinations and picks the best AIC.

---

## 7. Advanced Topics & Extensions

### 7.1 Vector Autoregression (VAR)

*   Multivariate Time Series.
*   Model "Claim Count" and "Economic Inflation" together.
*   Allows Inflation to impact Claims, and (maybe) Claims to impact Inflation (unlikely).

### 7.2 GARCH (Volatility Modeling)

*   ARIMA models the Mean. GARCH models the Variance.
*   *Use:* Modeling Stock Market Volatility for Variable Annuity hedging.

---

## 8. Regulatory & Governance Considerations

### 8.1 "Why did the forecast change?"

*   Time Series models are sensitive to the most recent data point.
*   If Dec 31st had a huge spike, the Jan 1st forecast might jump.
*   *Mitigation:* Smoothing or Outlier detection.

---

## 9. Practical Example

### 9.1 Worked Example: The "Inflation" Projector

**Scenario:**
*   Predict Medical Inflation for the next 5 years.
*   **Data:** 20 years of CPI-Medical.
*   **Model:** ARIMA(1,1,0) with Drift.
*   **Result:** Predicted 4% annual increase.
*   **Impact:** Used this assumption in the Reserving Model (Chain Ladder + Inflation).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Stationarity** is key for ARIMA.
2.  **Seasonality** is key for Insurance.
3.  **Holt-Winters** is simpler and often better for seasonal data.

### 10.2 When to Use This Knowledge
*   **Reserving:** Predicting IBNR counts.
*   **Pricing:** Trending losses to future periods.

### 10.3 Critical Success Factors
1.  **Visual Inspection:** Always plot your data first.
2.  **Outlier Removal:** One hurricane can ruin an ARIMA model.

### 10.4 Further Reading
*   **Hyndman & Athanasopoulos:** "Forecasting: Principles and Practice" (The Bible of Forecasting).

---

## Appendix

### A. Glossary
*   **Lag:** A past time period ($t-1$).
*   **Autocorrelation:** Correlation of a series with itself at different lags.
*   **AIC:** Akaike Information Criterion (Lower is better).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Differencing** | $y_t - y_{t-1}$ | Remove Trend |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
