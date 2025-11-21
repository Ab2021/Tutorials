# Day 38: Time Series - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Forecasting, Transformers, and Statistics

### 1. Why is "Channel Independence" effective in PatchTST?
**Answer:**
*   Treating multivariate time series as multiple univariate series (sharing the same model).
*   Prevents overfitting to spurious correlations between channels.
*   Drastically increases the amount of training data (Num Channels $\times$ Num Samples).

### 2. What is "RevIN"?
**Answer:**
*   Reversible Instance Normalization.
*   Normalizes the input window by its mean/std, passes through model, then denormalizes output using the *same* mean/std.
*   Handles distribution shift (Non-stationarity).

### 3. What is the difference between "Univariate" and "Multivariate" forecasting?
**Answer:**
*   **Univariate**: Predict $y_{t+1}$ using only past $y$.
*   **Multivariate**: Predict $y_{t+1}$ using past $y$ and other features $x^1, x^2$.

### 4. What is "Autoregressive" vs "Direct" forecasting?
**Answer:**
*   **Autoregressive**: Predict 1 step, feed back as input, predict next. Error accumulates.
*   **Direct**: Predict all $H$ steps at once (Output dim = $H$). Stable but harder to learn dependencies.

### 5. What is "Informer"?
**Answer:**
*   Efficient Transformer for Time Series.
*   Uses **ProbSparse Attention** to reduce complexity to $O(L \log L)$.
*   Uses **Distillation** to compress sequence length in encoder.

### 6. What is "Seasonality" and "Trend"?
**Answer:**
*   **Trend**: Long-term direction (increasing/decreasing).
*   **Seasonality**: Repeating patterns (daily, weekly, yearly).
*   Decomposition: $Y = T + S + Residual$.

### 7. Why use Transformers instead of LSTM for Time Series?
**Answer:**
*   Parallel training (no BPTT).
*   Captures long-range dependencies better (Attention).
*   LSTMs struggle with sequences > 1000 steps.

### 8. What is "Look-ahead Bias"?
**Answer:**
*   Using information from the future to predict the past.
*   Common bug: Normalizing using global min/max of the whole dataset (including test set).

### 9. What is "Quantile Regression"?
**Answer:**
*   Predicting a specific percentile (e.g., 90th percentile) instead of the mean.
*   Loss: $\max(q \cdot e, (q-1) \cdot e)$.
*   Crucial for risk management (Inventory planning).

### 10. What is "DTW" (Dynamic Time Warping)?
**Answer:**
*   Distance metric between two sequences.
*   Allows non-linear alignment (e.g., one sequence is faster than the other).
*   Better than Euclidean distance for shape matching.

### 11. What is "Teacher Forcing" in Time Series?
**Answer:**
*   Same as NLP. Using ground truth past values during training instead of model predictions.
*   Standard for Transformer training.

### 12. What is "Stationarity"?
**Answer:**
*   Statistical properties (Mean, Variance, Autocorrelation) do not change over time.
*   Most models assume stationarity. Real world is non-stationary.

### 13. What is "N-BEATS"?
**Answer:**
*   Pure MLP architecture for forecasting.
*   Uses stacks of blocks with forward/backward residual links.
*   Interpretable (Trend/Seasonality blocks).

### 14. What is "Lag Features"?
**Answer:**
*   Creating features from past values: $y_{t-1}, y_{t-7}, y_{t-365}$.
*   Helps non-recurrent models (XGBoost/MLP) see history.

### 15. What is "Covariate"?
**Answer:**
*   Exogenous variables that influence the target.
*   **Static**: User ID, Location.
*   **Dynamic**: Weather, Price, Holiday.

### 16. What is "Patching" in Time Series?
**Answer:**
*   Grouping $P$ time steps into a single token.
*   Reduces sequence length by factor $P$.
*   Preserves local semantic information.

### 17. What is "Zero-Shot Forecasting"?
**Answer:**
*   Using a Foundation Model (e.g., TimeGPT, Lag-Llama) pre-trained on billions of time series.
*   Predict on new dataset without training.

### 18. How to handle missing values in Time Series?
**Answer:**
*   **Imputation**: Forward Fill, Mean, Interpolation.
*   **Masking**: Let the model ignore them (Masked Attention).

### 19. What is "Backtesting"?
**Answer:**
*   Validation strategy.
*   Train on $0 \to T$, Test on $T \to T+H$.
*   Slide window: Train $0 \to T+k$, Test $T+k \to T+k+H$.

### 20. What is "Fourier Transform" used for?
**Answer:**
*   Converting Time Domain to Frequency Domain.
*   Identifies dominant cycles (Seasonality).
*   Used in TimesNet and Autoformer.
