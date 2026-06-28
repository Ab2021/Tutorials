# AI SYSTEM DESIGN SCENARIOS: THE RESUME MASTERCLASS (v1)
## Exhaustive Whiteboard Architectures Tailored to Chubb, Axtria, and EXL (No Code)

> **Critical Context:** The ultimate test of a Lead AI Engineer is the ability to deconstruct their own past work. When an interviewer asks for a system design, you must pivot the conversation to one of your massive successes. This document provides the absolute deepest architectural dive into the three core pillars of your resume. Memorize these 7-Block frameworks.

---

## SECTION 1: THE 7-BLOCK ARCHITECTURE FRAMEWORK

For ANY system design question, you must control the narrative by explicitly drawing these 7 blocks on the whiteboard. Do not skip straight to the model.

1.  **Block 1: Problem Scope & Constraints** → Scale, Latency SLA, Business KPI, Regulatory Constraints.
2.  **Block 2: Data Ingestion** → Streaming vs. Batch, Data Sources, PII Handling.
3.  **Block 3: Feature Engineering** → Feature Stores (Offline/Online), Temporal Leakage, Transformations.
4.  **Block 4: The Intelligence Core** → Model Selection, RAG pipelines, Ensembles, Hyperparameters.
5.  **Block 5: Serving Layer** → API Gateway, Async Queues (Celery), Kubernetes Pods, Edge deployments.
6.  **Block 6: Monitoring & Guardrails** → Data Drift (PSI), Calibration, Pydantic Validation.
7.  **Block 7: Telemetry & Feedback Loop** → Ground Truth updates, Retraining triggers, Business ROI.

---

## SCENARIO A: CHUBB INSURANCE — REAL-TIME HYBRID FRAUD SYSTEM

**The Interview Prompt:** "At Chubb, you built a fraud detection system that processed both structured data and unstructured notes. Walk me through the end-to-end architecture."

### Block 1: Problem Scope & Constraints
-   **Business Goal:** Increase the fraud referral rate to the Special Investigative Unit (SIU) from 12% to 25%, while strictly keeping precision (false positive rate) above 60% so we don't overwhelm investigators.
-   **Scale:** 10,000 incoming claims per day requiring real-time scoring. 1 million active claims requiring overnight batch re-scoring.
-   **Latency SLA:** Real-time predictions must return in under 200ms to avoid blocking the CRM UI.
-   **Regulatory Constraints:** Absolute explainability. We cannot deny a claim using a black-box neural network; we must provide traceable reason codes.

### Block 2: Data Ingestion
-   **Structured Path (Real-Time):** Claim submission triggers a Kafka event (`topic: claim-submitted`). A FastAPI consumer service reads the event payload.
-   **Unstructured Path (Async):** Adjusters type plain-text notes into the CRM. These trigger a separate asynchronous webhook, decoupling the slow NLP processing from the fast transactional system.
-   **Batch Path:** An Airflow CronJob runs at 2 AM, executing PySpark jobs on Databricks to extract all historical claims, policy holder details, and external ISO (Insurance Services Office) watchlists from Snowflake.

### Block 3: Feature Engineering & The Feature Store
-   **The Feature Store Architecture:** We implemented a dual-layer feature store.
    -   *Offline Store (Delta Lake):* Used for training. Stores point-in-time correct historical features (e.g., exactly what the claimant's history looked like on Jan 1st, 2024).
    -   *Online Store (Redis):* Used for real-time inference. Nightly batch jobs pre-compute heavy behavioral features (e.g., `claims_in_last_90_days`, `avg_claim_amount_vs_zipcode`) and load them into Redis as simple Key-Value pairs.
-   **The Leakage Trap:** Temporal leakage is the #1 cause of failed fraud models. We instituted a strict `AS_OF` timestamp cutoff. If an investigator flagged a claim as fraud on Day 5, we mathematically forbid the training pipeline from seeing any features or adjuster notes generated on Day 6.

### Block 4: The Intelligence Core (Hybrid ML + RAG)
This is a dual-pipeline architecture.
-   **Pipeline 1: The Structured Models (XGBoost & LightGBM)**
    -   *Real-Time (LightGBM):* Chosen purely for inference speed. Its leaf-wise tree growth and histogram-based splitting allow it to execute in < 15ms, easily hitting our 200ms API SLA.
    -   *Batch (XGBoost):* Chosen for probability calibration. Its level-wise growth creates more robust probability distributions. We use these precise probabilities (e.g., 0.82 vs 0.85) to accurately slot overnight claims into High, Medium, and Low risk queues.
-   **Pipeline 2: The Unstructured RAG Pipeline**
    -   Adjuster notes are highly variable. We cannot use simple regex.
    -   We built a vector database (FAISS) containing 500 confirmed, human-annotated historical fraud cases.
    -   *Execution:* When a new note arrives, we embed it using OpenAI `text-embedding-3-small`. We retrieve the top 5 most similar historical fraud cases.
    -   *Generation:* We pass the new note and the 5 historical examples to GPT-4o with a strict Prompt: "Extract binary flags for the following 30 fraud indicators. Output ONLY valid JSON."
    -   *Integration:* The resulting 30 binary flags (e.g., `inconsistent_timeline: True`) are saved to the database and fed as features into the next day's XGBoost run, merging NLP with tabular ML.

### Block 5: Serving Layer
-   **The API:** FastAPI deployed on an Azure Kubernetes Service (AKS) cluster.
-   **Resource Management:** Models are loaded into memory globally at pod startup (not per-request) to prevent Out-Of-Memory (OOM) crashes.
-   **Scaling:** Horizontal Pod Autoscaler (HPA) set to trigger at 70% CPU utilization, scaling from 3 to 20 pods during peak morning hours.

### Block 6: Monitoring & Guardrails
-   **Data Drift (Input):** We calculate the Population Stability Index (PSI) daily on our top 20 most important features. If a feature's distribution shifts massively (e.g., a bug in the CRM causes all claim amounts to drop by 10x), the pipeline halts before generating bad scores.
-   **Explainability Guardrail:** The LightGBM inference is wrapped in a TreeExplainer (SHAP). The API response forces the top 3 contributing features (e.g., "High Risk because: 1. Prior Claims = 4, 2. Amount > 3x Average") to be returned to the UI.

### Block 7: Telemetry & Feedback Loop
-   **The SIU Feedback Loop:** When investigators close a case, their final verdict (Fraud vs. Cleared) is written to the `labeled_claims` table.
-   **Automated Evaluation:** Every Monday, an automated Databricks notebook joins our predictions from 4 weeks ago with the final SIU verdicts. It calculates the Precision-Recall AUC (PR-AUC). If the PR-AUC drops by more than 3%, it triggers an alert for manual retraining.

---

## SCENARIO B: AXTRIA — OMNICHANNEL MARKETING MIX MODELING (MMM)

**The Interview Prompt:** "At Axtria, you built an MMM system for pharma clients. Standard MMM uses linear regression. Walk me through why and how you architected a more complex system."

### Block 1: Problem Scope & Constraints
-   **Business Goal:** Attribute sales correctly across multiple marketing channels (TV, Search, Social, Field Sales) to optimize next quarter's budget allocation.
-   **Scale:** Weekly data granularity across 3 years of history.
-   **Constraints:** Multicollinearity (channels are launched simultaneously). Diminishing Returns (spending $10M on TV doesn't yield 10x the sales of $1M).

### Block 2: Data Ingestion
-   **ETL Pipeline:** Python-based ETL pulls from Google Ads API, Nielsen (TV GRPs), and internal ERPs (Rx/Prescription sales).
-   **Data Integration:** Data is aggregated to a strictly weekly level. Missing data is imputed (forward-fill for delays, zero-fill for inactive campaigns).

### Block 3: Feature Engineering (The Mathematical Core)
-   **Adstock (Carryover Effect):** TV ads have memory. A commercial watched today influences a purchase next week. We engineer Adstock features using a geometric decay formula: `Adstock(t) = Spend(t) + decay_rate * Adstock(t-1)`.
-   **Saturation (Diminishing Returns):** We apply non-linear transformations (like the Negative Exponential function) to model the ceiling effect of marketing spend.
-   **External Regressors:** We engineer features for macroeconomic indices, seasonality, and competitor pricing to isolate the true baseline sales.

### Block 4: The Intelligence Core (XGBoost + Markov Chains)
-   **Why Not Just Linear Regression?** Linear regression assumes linear returns and channel independence. Pharma marketing is non-linear and synergistic (TV ads make Google Search ads cheaper and more effective).
-   **The XGBoost Model:** We train XGBoost on the engineered dataset. The tree-based splits naturally capture channel interactions (synergies) and the non-linear saturation curves.
-   **Markov Chain Attribution:** For digital multi-touch journeys, we don't just use Last-Click attribution. We build a Markov Chain transition matrix. We calculate the "Removal Effect" (If we completely remove the 'Email Campaign' node from the graph, how much does the overall probability of a prescription drop?). This distributes fractional credit mathematically.

### Block 5: Optimization & Serving
-   **The Optimization Engine:** The output of the MMM is not the final product. We feed the XGBoost model's predictions into a **SciPy SLSQP Optimizer**.
-   **The Objective Function:** Maximize total predicted sales.
-   **The Constraints:** Total budget = $X. Minimum TV spend = $Y. Maximum Digital spend = $Z.
-   **Delivery:** The optimizer generates the ideal budget allocation curve, which is delivered to stakeholders via a Streamlit dashboard allowing them to run "What-If" simulations in real-time.

### Block 6: Monitoring & Guardrails
-   **Holdout Validation:** The golden rule of MMM. We strictly hold out the last 12 weeks of data. We predict those 12 weeks and measure the Mean Absolute Percentage Error (MAPE). If MAPE > 10%, the model is rejected.
-   **Business Sanity Checks:** If the mathematical optimizer recommends dropping Field Sales to $0 (a political impossibility in Pharma), the constraints are mathematically tightened to force a realistic baseline.

### Block 7: Telemetry & Feedback
-   **Quarterly Refresh:** MMM models decay quickly as market dynamics shift. The entire Databricks pipeline is containerized and scheduled to re-run from scratch every 90 days, updating the adstock decay rates and saturation curves automatically based on the newest data.

---

## SCENARIO C: EXL HEALTH — PATIENT READMISSION & SURVIVAL ANALYSIS

**The Interview Prompt:** "At EXL Health, you built clinical models. Walk me through the architecture of the 30-day readmission prediction system and the survival analysis."

### Block 1: Problem Scope & Constraints
-   **Business Goal:** Predict which patients are highly likely to be readmitted to the hospital within 30 days of discharge so nurses can proactively intervene.
-   **Constraints:** HIPAA Data Privacy. Absolute requirement for interpretability (Clinicians reject black boxes). Extremely imbalanced dataset (~15% readmission rate).

### Block 2: Data Ingestion
-   **Nightly Batch Sync:** Data is pulled from Electronic Health Records (EHR), claims processing systems, and external Social Determinants of Health (SDOH) databases (housing stability, transport access).
-   **Data Quality Checks:** Extensive null checking on critical fields (Discharge Date, Primary Diagnosis).

### Block 3: Feature Engineering (Clinical Relevance)
-   **Temporal Leakage Defense:** The most critical block. If we include a diagnosis code that was updated on Day 3 *post-discharge*, the model is cheating. We enforced a strict SQL `AS_OF` join, guaranteeing that the model only sees data timestamped *before* the exact hour of discharge.
-   **Clinical Aggregations:** Engineered the Charlson Comorbidity Index (a weighted score of chronic conditions). Counted prior ED (Emergency Department) visits in the last 6 months. Mapped thousands of ICD-10 codes into broader clinical groupers (CCS).

### Block 4: The Intelligence Core (Classification & Survival)
-   **The Readmission Model:** Started with Logistic Regression (high trust). Migrated to Random Forest for a 5-point AUROC bump, while maintaining interpretability via Gini importance and SHAP values.
-   **The Survival Analysis (Kaplan-Meier):** To answer *when* patients drop out of care programs, we couldn't use standard classification because of "Right-Censored" data (patients who haven't dropped out yet by the end of the study). We used Kaplan-Meier estimation to build survival curves. This mathematically proved that month 2 was the critical drop-off window, allowing the business to time their interventions perfectly.

### Block 5: Serving Layer
-   **Batch Processing:** Clinicians don't need real-time scoring; they plan their outreach shifts in the morning. An Airflow DAG executes the scoring pipeline at 3 AM.
-   **Dashboard Integration:** The output is a secure CSV pushed to an SFTP server, which is ingested by the care management UI.

### Block 6: Monitoring & Guardrails
-   **Calibration (Platt Scaling):** A model outputting a score of "0.8" must correlate to an 80% real-world readmission rate. Random Forests are notoriously uncalibrated (they push scores toward the mean). We used Platt Scaling on a hold-out set to strictly calibrate the probabilities.
-   **Metric Selection:** We did not monitor Accuracy (useless on imbalanced data). We monitored PR-AUC (Precision-Recall) and specifically "Precision at Top Decile" (If a nurse calls the top 10% of our flagged patients, how many were actually going to be readmitted?).

### Block 7: Telemetry & Feedback
-   **Outcome Tracking:** 30 days after prediction, the actual outcomes materialize in the EHR. A feedback loop joins the predictions with the realities to calculate true model drift.
-   **Clinical Feedback:** A quarterly review board with the nursing staff. If they flag that a certain feature (e.g., "Distance to Hospital") is irrelevant for telehealth patients, we update the feature engineering pipeline.
