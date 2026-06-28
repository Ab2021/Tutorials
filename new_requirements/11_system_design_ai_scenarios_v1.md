# AI SYSTEM DESIGN SCENARIOS: THE MASTERCLASS (v1)
## Real-world whiteboard architectures tailored to YOUR Resume (No Code)

> **Critical Context:** When interviewing for a Lead AI Engineer role, you cannot rely on generic internet examples. You must use the 7-Block framework to deconstruct the exact projects on your resume: **Chubb Fraud Detection, Axtria MMM, and EXL Health Readmission.** If they ask for a system design, you pivot the whiteboard to one of these three architectures.

---

## SECTION 1: THE 7-BLOCK ARCHITECTURE FRAMEWORK

For ANY system design question, walk to the whiteboard and draw these 7 blocks in order. 

1.  **Block 1: Problem Scope & Constraints** → What is the scale? Latency? Business SLA?
2.  **Block 2: Data Ingestion** → How does messy data enter the system? (Kafka, Airflow, Batch).
3.  **Block 3: Feature Engineering** → The hardest part (Temporal leakage, Feature Stores).
4.  **Block 4: The Intelligence Core (Model Layer)** → RAG, XGBoost, Ensembles.
5.  **Block 5: Execution & UI (Serving)** → FastAPI, Kubernetes, Async Queues.
6.  **Block 6: Monitoring & Guardrails** → PSI, Data Drift, Faithfulness metrics.
7.  **Block 7: Telemetry & Feedback Loop** → Retraining triggers and human corrections.

---

## SCENARIO A: REAL-TIME FRAUD DETECTION (Based on Chubb)

**Prompt:** "Design a system to detect insurance fraud at the moment a claim is submitted."

### Block 1: Scope & Constraints
-   **Goal:** Increase fraud referral rate to Special Investigative Unit (SIU) while maintaining precision above 60%.
-   **Scale & Latency:** 10,000 claims/day real-time (SLA < 200ms) + 1M claims overnight batch (SLA: 6 AM).
-   **Constraints:** Regulatory explainability required (No black-box neural networks for the final decision).

### Block 2: Data Ingestion
-   **Real-Time Path:** Claim submission hits a Kafka topic (`claim-events`). FastAPI consumer reads the event.
-   **Batch Path:** Airflow CronJob at 2 AM triggers an ETL pipeline reading from Snowflake and ISO watchlists.

### Block 3: Feature Engineering & Storage
-   **Offline Store (Delta Lake):** PySpark calculates 90-day behavioral aggregates (e.g., `avg_claim_amount_90d`). We strictly enforce **Temporal Cutoffs** (only using data available *before* the claim) to prevent data leakage.
-   **Online Store (Redis):** Nightly batch jobs push these pre-calculated features to Redis. During real-time inference, the API fetches them in < 5ms.

### Block 4: The Intelligence Core (Hybrid ML + RAG)
-   **The Structured Model:** LightGBM for real-time (sub-15ms speed). XGBoost for overnight batch (better probability calibration).
-   **The Unstructured Pipeline (RAG):** For the adjuster's claim notes, we use an asynchronous Celery worker. It embeds the text, searches a FAISS vector DB for similar historical fraud cases, and uses GPT-4o to extract 30 binary fraud flags. These flags feed into the next day's XGBoost run.

### Block 5: Serving Layer
-   FastAPI deployed on Kubernetes. Configured with Horizontal Pod Autoscaling (HPA) targeting 70% CPU utilization.
-   Models are loaded into memory *once* at pod startup to prevent Out-Of-Memory (OOM) crashes.

### Block 6: Monitoring & Guardrails
-   **Input Monitoring:** Calculate Population Stability Index (PSI) daily on the top 20 features to detect data drift.
-   **Output Monitoring:** Track the predicted score distribution to ensure the risk tiering thresholds remain valid.

### Block 7: Feedback Loop
-   SIU investigators review flagged claims and confirm/deny fraud 4 weeks later.
-   These confirmed outcomes are written to a `labeled_claims` table, triggering an automated weekly PR-AUC evaluation to check for model degradation.

---

## SCENARIO B: OMNICHANNEL MARKETING MIX MODELING (Based on Axtria)

**Prompt:** "Design a system to attribute sales ROI to different marketing channels and optimize weekly budgets."

### Block 1: Scope & Constraints
-   **Goal:** Maximize total sales given a fixed budget.
-   **Scale:** Weekly granularity, 3 years of historical data per run.
-   **Constraints:** Highly correlated channels (multicollinearity). Diminishing returns on spend.

### Block 2: Data Ingestion
-   Extract data from ERP (Sales), Nielsen (TV spend), and Google/FB APIs (Digital spend).
-   Merge into a weekly time-series dataset using Databricks.

### Block 3: Feature Engineering (The Secret Sauce)
-   **Adstock Transformation:** Spend effect carries over to future weeks. `Adstock(t) = Spend(t) + decay * Adstock(t-1)`.
-   **Saturation:** Diminishing returns modeled mathematically (e.g., negative exponential).
-   **Calendar/Lag Features:** Holidays, seasonality indices, and 2-week lags for slow channels.

### Block 4: The Intelligence Core (XGBoost + Markov)
-   **Baseline:** Linear regression provides interpretable coefficients (baseline ROI).
-   **Advanced Model:** XGBoost naturally captures the S-curve saturation and channel synergy interactions (e.g., TV boosting Digital).
-   **Attribution:** For multi-touch digital journeys, we use Markov Chains to calculate the "removal effect" of a channel, distributing credit fairly across the sequence rather than just "last-click."

### Block 5: Serving & Optimization
-   The models do not run in real-time. They run in Databricks notebooks offline.
-   **The Optimizer:** We use SciPy constrained optimization. Objective: Maximize predicted sales from the XGBoost model. Constraints: Total budget < X, TV budget > Y.
-   Results are served via a Streamlit dashboard for stakeholders.

### Block 6: Monitoring & Guardrails
-   **Holdout Validation:** The last 3 months of data are strictly held out. We monitor the Mean Absolute Percentage Error (MAPE). If MAPE > 10%, the model requires investigation.
-   **Business Sanity Checks:** If the model suggests $0 TV spend (which violates business reality), we constrain the optimization space.

### Block 7: Feedback Loop
-   Quarterly retraining with the freshest data.
-   If business stakeholders dispute the attribution, we run a sensitivity analysis on the Adstock decay parameters to prove robustness.

---

## SCENARIO C: PATIENT READMISSION PREDICTION (Based on EXL Health)

**Prompt:** "Design a system to predict if a hospital patient will be readmitted within 30 days."

### Block 1: Scope & Constraints
-   **Goal:** Identify top 10% of high-risk patients for proactive nurse outreach.
-   **Constraints:** Severe regulatory compliance (HIPAA). Absolute necessity for clinical explainability (No black boxes).

### Block 2: Data Ingestion
-   Nightly batch extract from Electronic Health Records (EHR), claims databases, and Social Determinants of Health (SDOH) external feeds.

### Block 3: Feature Engineering (Leakage Prevention)
-   **The Trap:** Data Leakage. If we include a diagnosis code that was entered *after* the patient was discharged, the model learns the future.
-   **The Fix:** Strict temporal cutoffs. Features are joined using an `AS_OF` timestamp matching the exact hour of hospital discharge.
-   Features engineered: Charlson Comorbidity Index, prior admission counts, length of stay.

### Block 4: The Intelligence Core
-   Started with Logistic Regression as the interpretable baseline.
-   Moved to Random Forest / XGBoost for higher AUROC.
-   **Survival Analysis:** Used Kaplan-Meier curves to model *when* patients drop off care programs, identifying the 2-month mark as the critical intervention window.

### Block 5: Serving Layer
-   Batch scoring. A nightly Airflow DAG scores all patients discharged that day.
-   The scores are written to the Care Management Dashboard via a secure API.

### Block 6: Monitoring & Guardrails
-   **Calibration is Critical:** A risk score of 0.8 MUST mean an 80% real-world probability of readmission. We use Platt Scaling and monitor the Brier Score. If the model becomes overconfident, clinicians lose trust.
-   **Explainability:** Every prediction must be accompanied by SHAP values (e.g., "High Risk due to: 3 prior admissions, high comorbidity score").

### Block 7: Feedback Loop
-   Track actual 30-day outcomes. Compare against the predicted deciles (Capture Rate).
-   Gather qualitative feedback from the nurses: Was the outreach actually helpful, or was the patient fundamentally unpreventable (e.g., planned chemo)?
