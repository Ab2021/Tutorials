# ### Days 79–81 – Fraud Detection (Classification)

*   **Topic:** Finding the Needle
*   **Content:**
    *   **Types:** Hard Fraud (Staged accidents) vs. Soft Fraud (Padding claims).
    *   **Techniques:**
        *   Supervised: XGBoost on past confirmed fraud.
        *   Unsupervised: Isolation Forests, Autoencoders (Reconstruction error).
    *   **Social Network Analysis (SNA):** Graph features (Cycles, cliques).
*   **Interview Pointers:**
    *   "How do you evaluate a fraud model if you don't have many labeled fraud cases?" (Precision at Top K, feedback loops).
*   **Challenges:** Adversarial behavior (Fraudsters adapt).