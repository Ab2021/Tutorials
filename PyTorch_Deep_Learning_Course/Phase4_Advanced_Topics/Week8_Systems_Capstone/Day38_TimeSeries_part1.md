# Day 38: Time Series - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: RevIN, TimesNet, and Temporal Fusion Transformer

## 1. Handling Non-Stationarity (RevIN)

Time series statistics (mean/std) change over time.
Normalization (StandardScaler) on the whole train set fails during inference if distribution shifts.
**RevIN (Reversible Instance Normalization)**:
1.  **Normalize**: $x' = (x - \mu) / \sigma$ per instance (window).
2.  **Model**: $y' = f(x')$.
3.  **Denormalize**: $y = y' \cdot \sigma + \mu$.
*   Restores the scale info *after* the model.

## 2. TimesNet

Transforms 1D time series into 2D tensors using FFT (Fast Fourier Transform).
*   Captures multi-periodicity.
*   Uses Inception Blocks (2D Conv) to extract features.
*   Unified model for Forecasting, Classification, and Anomaly Detection.

## 3. Temporal Fusion Transformer (TFT)

Google's model for interpretable forecasting.
*   Handles **Static Covariates** (Location, Store ID).
*   Handles **Known Future Inputs** (Holidays, Day of Week).
*   **Variable Selection Network**: Learns which features are important.
*   **Quantile Regression**: Predicts confidence intervals (P10, P50, P90).

## 4. Evaluation Metrics

*   **MSE/MAE**: Standard.
*   **MAPE**: Percentage error. Unstable if value is 0.
*   **CRPS (Continuous Ranked Probability Score)**: For probabilistic forecasts.
*   **DTW (Dynamic Time Warping)**: Similarity measure that handles temporal shifts.
