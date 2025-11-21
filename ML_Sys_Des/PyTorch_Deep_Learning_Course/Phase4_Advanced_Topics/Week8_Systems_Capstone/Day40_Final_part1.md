# Day 40: Final Review - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: ML System Design, MLOps, and Ethics

## 1. ML System Design Framework

When asked to design a system (e.g., "Design a News Feed"):
1.  **Requirements**: Latency? Throughput? Personalization?
2.  **Data**: Sources? Labels? Features?
3.  **Model**: Two-Tower? XGBoost? LLM?
4.  **Training**: Offline vs Online. Frequency.
5.  **Serving**: Pre-compute vs Real-time. Caching.
6.  **Monitoring**: Drift detection. Logging.

## 2. MLOps: The Lifecycle

Code is only 5% of the system.
*   **Data Ops**: Feature Store (Feast), Versioning (DVC).
*   **Model Ops**: Registry, CI/CD for ML.
*   **Platform Ops**: GPU Clusters (Slurm/K8s).

## 3. Ethics & Safety

*   **Bias**: Training data reflects societal biases.
*   **Fairness**: Equal performance across demographics.
*   **Explainability**: SHAP, Integrated Gradients. Why did the model predict X?

## 4. How to Keep Learning

*   **Twitter/X**: Follow researchers (Karpathy, LeCun).
*   **Newsletters**: The Batch, Import AI.
*   **Conferences**: NeurIPS, ICML, CVPR, ICLR.
