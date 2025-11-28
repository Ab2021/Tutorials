# ### Days 73–75 – Tree-based Models for Frequency & Severity

*   **Topic:** Non-linear Pricing
*   **Content:**
    *   **Decision Trees:** Visual, interpretable, but prone to overfitting.
    *   **Random Forests:** Bagging to reduce variance.
    *   **Gradient Boosting (XGBoost/LightGBM/CatBoost):** Boosting to reduce bias. The gold standard for tabular insurance data.
    *   **Objective Functions:** Poisson Loss, Gamma Loss, Tweedie Loss.
*   **Interview Pointers:**
    *   "How do you handle exposure in an XGBoost frequency model?" (Use `base_margin` or `ln(exposure)` as an offset).
*   **Tricky Parts:** Monotonicity constraints (Rate must increase with points/accidents).