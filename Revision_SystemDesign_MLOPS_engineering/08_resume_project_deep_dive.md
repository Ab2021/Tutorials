# RESUME PROJECT DEEP DIVE — Complete Technical Q&A
> For every project on your resume, this file has the full technical drill-down ready.

---

## PROJECT 1: CHUBB — INSURANCE FRAUD DETECTION (Current)

### Project Overview (30-second pitch)
> "At Chubb, I architected a production-grade fraud detection system combining XGBoost for structured claim features with a RAG pipeline using GPT-4o to extract fraud indicators from unstructured claim notes. The system improved fraud referral rates from 12% to 23% (capturing more fraudulent claims in the top risk tier) and won the Q1 2025 STAR Award."

---

### Q: Walk me through the full architecture
**Answer:**
```
Data Sources → Feature Engineering → Model Layer → Serving → Monitoring
    ↓                  ↓                  ↓            ↓          ↓
Snowflake          PySpark on         XGBoost       FastAPI    PSI checks
Synapse            Databricks       (batch)         K8s        SIU feedback
Claim notes        Delta tables     LightGBM        Airflow    MLflow tracking
ISO data           Feature store    (real-time)     CronJob    Ground truth eval
```

**Real-time path:**
1. Claim submitted → Kafka event → FastAPI consumer
2. Feature extraction from Redis (pre-computed behavioral features) + JSON payload
3. LightGBM inference (<15ms) → fraud_probability, risk_tier, top SHAP factors
4. Response to CRM: if score > 0.8 → SIU alert; 0.5-0.8 → enhanced review

**Batch path:**
1. Airflow CronJob at 2am daily
2. Read all active claims from Snowflake
3. PySpark feature engineering on Databricks (behavioral aggregates, joins with feature store)
4. pandas_udf for distributed XGBoost inference across cluster
5. Write scored claims to Snowflake fraud_scores table
6. Trigger SIU dashboard refresh

**NLP enrichment (async):**
1. New claim with notes → Celery task queue
2. Chunk notes → embed with OpenAI ada-002 → FAISS search for similar fraud cases
3. GPT-4o extracts 30-40 binary fraud flags (JSON structured output)
4. Flags stored in claim_nlp_features table
5. Next night's batch run includes NLP flags as features

---

### Q: Why did you use XGBoost and not LightGBM for batch?
**Answer:**
> "Both were evaluated. For batch overnight scoring, calibration quality matters more than raw speed — we use fraud probability scores for risk tiering (HIGH/MEDIUM/LOW), and if probabilities are miscalibrated, our tier thresholds don't work.
>
> XGBoost's level-wise tree growth produces better-calibrated probabilities than LightGBM's leaf-wise growth on our 300K row dataset. I measured calibration using Brier score and reliability diagrams: XGBoost Brier = 0.042 vs LightGBM = 0.058.
>
> LightGBM IS the right choice for real-time — it's 3x faster inference and the 30ms vs 90ms difference matters for 200ms SLA. So we use both: LightGBM for real-time, XGBoost for batch."

---

### Q: How did you handle the unstructured claim notes?
**Answer (full technical):**
> "Claim notes are adjuster commentaries, email threads, medical records — unstructured text with highly variable format and length.
>
> We couldn't train a supervised classifier on them directly because: (1) fraud labels come 4-6 weeks later, so we needed leading indicators, (2) fraud patterns change monthly, so a fine-tuned model would go stale, (3) we needed explainability for regulatory purposes.
>
> RAG solved this: we built a library of 500 confirmed fraud cases with expert-annotated fraud patterns. For each new claim note, we retrieve the top-5 most similar historical patterns, then prompt GPT-4o to extract whether those patterns appear in the new note. Output is 30-40 binary flags in JSON.
>
> Why not fine-tuning? Fine-tuning would require: labeled training data at scale, GPU infrastructure, months-long training cycles when fraud patterns shift. RAG lets us update the knowledge base by adding new confirmed cases — no training needed.
>
> Evaluation: weekly PR-AUC on 200-case human-annotated ground truth. Faithfulness checked via GPT-4.5 as judge. Recall@5 for retrieval quality."

---

### Q: How did you measure the 12% to 23% improvement?
**Answer:**
> "Referral rate is: (claims sent to SIU for investigation) / (total claims reviewed). At 12%, only 12 out of every 100 claims were flagged as suspicious and referred.
>
> Our target was 25% referral rate — we got to 23%.
>
> But raw referral rate isn't enough — precision matters too. If we referred 50% of claims but most were genuine, we'd overwhelm SIU with false positives.
>
> We measured Precision@Referral: of claims we referred, what % were confirmed fraudulent or suspicious by SIU. The constraint was: precision must stay above 60% as we increase referral rate.
>
> The XGBoost model without NLP flags achieved: 18% referral at 65% precision.
> Adding NLP flags from RAG pipeline: 23% referral at 63% precision — better recall, acceptable precision tradeoff."

---

### Q: What was your biggest technical challenge?
**Answer:**
> "Two challenges: first, feature leakage. Claims adjusters often add notes AFTER the claim is resolved — if we trained on notes from claims where fraud was already confirmed, the notes would contain information unavailable at decision time. We had to strict temporal cutoffs: only use notes written within the first 72 hours of claim creation.
>
> Second: prompt reliability across LLM versions. When OpenAI released GPT-4o-mini, our chain-of-thought prompts broke — the model ignored some reasoning steps. We added LLM-as-judge evaluation on a fixed 100-case test set and ran it before any production model version switch. Added 2 days to deployment but caught a 15% faithfulness degradation before it hit production."

---

## PROJECT 2: AXTRIA — MARKETING MIX MODELING AND OMNICHANNEL ANALYTICS

### Project Overview
> "At Axtria as a data science lead for pharma clients, I built MMM systems to attribute sales to marketing channels and optimize budget allocation. Key techniques: XGBoost with adstock features for non-linear saturation, Markov Chain attribution for multi-touch journeys, and survival analysis for customer engagement curves."

---

### Q: What does your MMM model output? (Interviewer kept asking this)
**Answer (crisp — one sentence first):**
> "The output is predicted sales in dollars for each time period, given the input marketing spend."

**Full explanation:**
> "The model takes weekly marketing spend per channel (TV, digital, print, promotions) as input, with adstock transformations applied to model carryover effects. The output for each week is: predicted total product sales.
>
> Channel attribution comes from: (1) in linear MMM, the coefficient for each channel = incremental sales per dollar spent; (2) in XGBoost MMM, SHAP values give the same attribution accounting for non-linear effects.
>
> ROI = channel_contribution / channel_spend. We use this to answer: 'For every dollar spent on TV, how much incremental revenue do we get?'
>
> The optimization layer takes these model outputs and finds the budget allocation B that maximizes predicted total sales subject to constraints (total budget, min spend per channel, channel availability)."

---

### Q: Why XGBoost for MMM? Why not linear regression?
**Answer:**
> "Linear regression is the right starting point and gives directly interpretable coefficients as channel contributions. But it has two limitations for pharma marketing data:
>
> First: diminishing returns. Doubling TV spend doesn't double sales — there's a saturation curve. Linear models assume linearity; XGBoost naturally captures the S-curve saturation.
>
> Second: channel interactions. TV spend amplifies the ROI of digital campaigns (synergy effect). Linear models assume independence; XGBoost tree interactions capture this.
>
> I validated with holdout: linear MMM MAPE = 14.2%. XGBoost MAPE = 8.7%. The 5.5-point improvement justified added complexity — and I used SHAP for channel attribution to maintain interpretability."

---

### Q: Why did you use Markov chains for attribution? (You were challenged on this)
**Answer (the HONEST one that acknowledges the challenge):**

> "Good challenge. You're right that Markov chains are not always the necessary choice for customer journey attribution.
>
> In the specific pharma context: we had multi-channel digital campaigns (email → search → social → website visit → prescription). The sequence of touchpoints mattered — different paths to conversion had different values. Linear attribution (last-click or equal credit) was wrong because early touchpoints were undervalued.
>
> Markov chains fitted because: customer journeys were explicitly sequential (we could observe the order), we needed to quantify what conversion probability drops if we remove a channel, and the memoryless assumption (next touchpoint depends only on current) was a reasonable approximation for weekly digital data.
>
> Alternatives I considered: Shapley values (game-theory, more complex), LSTM (requires much more data), data-driven attribution in Google Analytics (black box). Markov was the right balance of transparency, business interpretability, and technical accuracy.
>
> But I take your point — for simpler omnichannel problems with low data volume, last-touch or linear attribution may be sufficient and more maintainable."

---

### Q: Explain Kaplan-Meier for survival analysis (healthcare/EXL experience)
**Answer:**
> "I used KM at EXL Health to model patient engagement with a care management program.
>
> The business question: 'What is the probability a patient remains engaged at 3 months? 6 months?'
>
> This is a survival problem because: we can only observe patients until our study ends. Patients who haven't disengaged yet are 'right-censored' — we know they're still engaged as of today, but don't know their true disengagement date.
>
> KM estimates the survival function: S(t) = probability of remaining engaged past time t. The formula handles censored observations correctly: each event time updates S(t) = previous S * (1 - d_i/n_i), where d_i = disengagements at time t_i and n_i = still-engaged patients.
>
> Business output: the KM curve told us that 75% of patients remain engaged at month 1, but only 40% at month 6. This identified month 2-3 as the critical window for retention intervention. We built a separate model (Cox PH) to identify which patient characteristics predicted early dropout."

---

## PROJECT 3: EXL HEALTH — PATIENT READMISSION AND HEALTHCARE ANALYTICS

### Project Overview
> "At EXL, I worked as a data scientist for CVS Health and other healthcare clients. Key projects: patient readmission risk prediction, survival analysis for patient engagement, and healthcare claims analytics."

---

### Q: How did you build the patient readmission model?
**Answer:**
> "The business goal: identify high-risk patients for proactive outreach before hospital discharge to prevent 30-day readmission.
>
> Features: clinical features (diagnosis codes, procedure codes, length of stay, number of prior admissions, comorbidities via Charlson Comorbidity Index), demographic features, social determinants of health (SDOH — housing stability, transportation access), claims history.
>
> Key challenge: data leakage. Diagnosis codes entered AFTER discharge would be unavailable at decision time. We enforced temporal cutoffs: only features recorded BEFORE discharge.
>
> Model: started with logistic regression (interpretable, fast, regulatory-friendly). Achieved AUROC 0.72. Random Forest pushed to 0.78. Clinical team could accept 0.78 — interpretability trade-off was acceptable.
>
> Evaluation: AUROC (class-balanced here, ~20% readmission rate), precision at top decile, calibration (Brier score) — calibration critical for clinical decision thresholds."

---

### Q: What was your role and scope at EXL?
**Answer:**
> "At EXL I wore two hats: individual contributor for technical development AND project lead for a small team (2-3 analysts).
>
> On the technical side: feature engineering, model development, validation framework design, MLflow experiment tracking.
>
> On the leadership side: I ran standups, assigned tasks, reviewed code, managed timeline against client deliverables, and presented findings to client stakeholders at CVS Health.
>
> This is where I developed the lead skills I'm looking to apply in my next role — translating between business stakeholders who care about readmission rate and technical team members who care about AUC."

---

## FOLLOWUP QUESTION BANK — Complete List

### XGBoost Follow-ups
- "What is the learning rate and how does it interact with n_estimators?"
  > "learning_rate shrinks each tree's contribution by that factor. Lower rate = more trees needed for same fit, but better generalization. Rule of thumb: 0.05-0.1 with 500-1000 trees beats 0.3 with 100 trees. Always pair early_stopping_rounds with low learning_rate."

- "What is subsample vs colsample_bytree?"
  > "subsample = fraction of training ROWS used per tree (adds row-level randomness, reduces overfitting). colsample_bytree = fraction of FEATURES used per tree (adds feature-level randomness, reduces feature dominance). Both are between 0 and 1, typically 0.7-0.9."

- "What does scale_pos_weight do exactly?"
  > "It multiplies the gradient and hessian of positive class samples by scale_pos_weight. Effect: the model treats each positive (fraud) sample as if it were scale_pos_weight negative samples. This makes the model pay more attention to rare fraud cases. Set to: n_negatives/n_positives."

- "How does XGBoost handle missing values?"
  > "XGBoost learns the default direction for missing values during training. For each split, it tries routing missing values to left child vs right child, and picks whichever gives higher gain. This is built-in and doesn't require imputation."

### Feature Engineering Follow-ups
- "How did you select which features to include?"
  > "Three-stage process: (1) Information Value (IV > 0.1 to keep), (2) VIF < 10 to remove multicollinear, (3) SHAP-based permutation importance after first model run — remove features with near-zero SHAP. Started with 140 features, ended with 40 high-quality features."

- "How do you encode categorical variables with 1000 unique values?"
  > "Target encoding (mean fraud rate per category) with k-fold out-of-fold encoding to prevent leakage. Never target-encode without out-of-fold — training data leakage inflates importance of the feature. WoE encoding is an alternative with better interpretability for logistic regression."

### MLOps Follow-ups
- "How do you version control your training data?"
  > "Delta Lake time travel. The training run records the Delta table version (or timestamp) in MLflow metadata. Any run can be reproduced by: (1) checking out the model from MLflow registry by run_id, (2) reading the Delta table at the recorded timestamp. Full reproducibility."

- "How do you ensure no data leakage in production?"
  > "Two controls: (1) temporal cutoffs enforced at feature creation — we only join features available before the decision point, (2) schema validation at inference time — if a feature arrives late, it's nulled out using a feature availability map, not imputed with training data statistics."

- "What happens when the model fails in production?"
  > "We have: (1) readiness probe prevents traffic routing to broken pods, (2) if model returns error/timeout, fallback to rule-based system (not 'not fraud' — fallback to heuristics), (3) if error rate > 1% for 5 minutes, PagerDuty alert + automatic rollback to previous model version in MLflow registry."

---

## PROJECT STORYTELLING FRAMEWORK

### The SOAR Structure for Every Project

Use this for any resume project:

- **Situation:** What business problem existed?
- **Objective:** What was the specific target?
- **Action:** What did you build? What choices did you make?
- **Result:** What was the measurable impact?

### Example: Chubb Fraud Detection in SOAR

**Situation:** Fraud investigators were reviewing claims with low yield, missing fraud due to limited structured features.
**Objective:** Increase fraud referral rate to 25% while keeping precision above 60%.
**Action:** Built a real-time LightGBM scorer, a nightly XGBoost batch model, and an async RAG pipeline over claim notes.
**Result:** Referral rate improved from 12% to 23% at 63% precision; won Q1 2025 STAR Award.

### Pitfalls to Avoid in Resume Stories

- Vague impact: "improved model performance" → instead: "improved PR-AUC from 0.72 to 0.83"
- Hiding failure: mention what you learned
- Exaggerating scope: be clear about what you owned vs what the team owned
- Too much jargon before business context

---

## ADDITIONAL CHUBB FOLLOW-UPS

### "How do you prevent leakage in the RAG pipeline?"

> "We only used notes written within 72 hours of claim creation, before fraud confirmation. If a note was added after a SIU decision, we excluded it from training and inference. We also used temporal splits: train on older claims, validate on newer claims."

### "What if the LLM returns flags not supported by the text?"

> "We use an LLM-as-judge evaluation to score faithfulness. If average faithfulness drops below 4 out of 5, we pause the pipeline and update the prompt or retrieval strategy. We also log every extraction with retrieval context so investigators can verify the source."

### "How do you scale the RAG pipeline?"

> "Claim note processing is asynchronous via Celery workers. Embedding and LLM calls are batched where possible. FAISS indexes are rebuilt weekly with new confirmed cases. We cache embeddings for frequently retrieved documents."

### "What ground truth do you maintain?"

> "We maintain 200+ human-annotated confirmed fraud cases and 300+ genuine cases. Each week we run the RAG pipeline on this fixed set and measure flag precision, recall, and faithfulness. This gives a stable benchmark across prompt and model changes."

---

## ADDITIONAL AXTRIA FOLLOW-UPS

### "How do you handle missing channel data in MMM?"

> "We impute missing spend with zero when the channel was truly inactive, and use forward-fill or interpolation when data was delayed. We flag weeks with heavy imputation and exclude them from model validation if the proportion is high."

### "What if two channels are highly correlated?"

> "We check VIF and correlation matrices. If two channels are highly correlated, we either combine them into a single media channel or use regularization. In XGBoost, collinearity matters less than in linear regression, but we still monitor SHAP attribution to ensure credit is not double-counted."

### "How do you validate adstock and saturation parameters?"

> "We grid-search decay and saturation parameters using validation MAPE. We also compare against business priors: TV decay should be longer than digital, and saturation should be visible at high spend levels. If parameters violate business logic, we constrain the search space."

---

## ADDITIONAL EXL FOLLOW-UPS

### "Why is calibration important in readmission risk?"

> "A risk score of 0.8 should mean an 80% chance of readmission. If the model is overconfident, clinicians will trust the score too much. We calibrated with Platt scaling on a held-out set and tracked Brier score to ensure probabilities matched outcomes."

### "How did you work with clinicians?"

> "I presented results using SHAP explanations and decile charts rather than raw metrics. Clinicians care about which patients to target and why. I involved them in feature definition to ensure variables like Charlson Comorbidity Index and prior admissions were clinically meaningful."

---

## LEADERSHIP AND MENTORING STORIES

### "How do you mentor junior data scientists?"

> "I pair junior team members with clear ownership areas, such as feature engineering or model monitoring. I review their code and push them to justify every model choice with metrics and business impact. I also encourage them to build simple baselines first before adding complexity."

### "How do you manage stakeholder expectations?"

> "I set success metrics before starting a project. I run weekly status updates with concrete deliverables, and I translate technical progress into business language. When results differ from expectations, I explain why and propose a path forward rather than hiding bad news."

### "Tell me about a conflict with a stakeholder."

> "A client wanted a deep neural network for a small tabular dataset. I pushed back by building a logistic regression baseline first. It achieved 85% of the target performance with full interpretability. The client agreed to use it as the production model. This taught me to let the data and metrics make the case rather than argue from opinion."

---

## CHUBB FRAUD SYSTEM — DETAILED ARCHITECTURE AND OPS

### End-to-End Architecture Blocks

| Block | Components | Purpose |
|---|---|---|
| Ingestion | Snowflake, Synapse, CRM events, ISO watchlists | Collect structured and unstructured data |
| Feature Engineering | Databricks, PySpark, Delta Lake | Compute behavioral, network, temporal, NLP features |
| Online Feature Store | Redis | Serve pre-computed features with sub-5ms latency |
| Offline Feature Store | Delta Lake | Store point-in-time correct features for training |
| Real-Time Model | FastAPI, Kubernetes, LightGBM | Score claims synchronously at submission |
| Batch Model | Airflow, Databricks, XGBoost | Score all claims nightly with richer features |
| NLP Enrichment | Celery, OpenAI embeddings, FAISS, GPT-4o | Extract binary fraud flags from claim notes |
| Knowledge Graph | Neo4j / graph store | Detect fraud rings and provide network features |
| Model Registry | MLflow | Version models, stages, metadata |
| Monitoring | Prometheus/Grafana, PSI/KS checks, SIU feedback loop | Detect drift and degradation |
| Feedback | SIU confirmations, labeled claims | Drive retraining and prompt improvement |

### Operational Runbook for Chubb Fraud System

**Daily batch job failure:**
- Check Airflow DAG status and Databricks cluster logs
- Verify input data freshness and row counts
- If feature store write fails, hold previous day's features and alert
- If scoring fails, use previous day's scores with a note

**Real-time latency spike:**
- Check Redis latency and hit rate
- Check Kubernetes pod CPU/memory and HPA scaling
- Check downstream API dependencies
- If model latency exceeds SLA, activate fallback rule-based scoring

**RAG pipeline faithfulness drop:**
- Pause NLP flag ingestion
- Review recent prompt changes and model version changes
- Run LLM-as-judge on fixed ground truth set
- Update prompt or retrieval strategy before resuming

**Model drift alert:**
- Investigate whether shift is data pipeline issue or real pattern change
- If real and sustained, collect new labels and retrain challenger
- A/B test challenger before promotion

---

## AXTRIA MMM PIPELINE — DETAILED ARCHITECTURE AND OPS

### End-to-End Architecture Blocks

| Block | Components | Purpose |
|---|---|---|
| Ingestion | ERP, Nielsen TV, Google Ads, Facebook Ads, vendor invoices, weather/econ data | Collect weekly spend and sales data |
| Data Integration | Python ETL / Databricks | Merge channels into weekly granularity |
| Feature Engineering | Adstock, saturation, lag, calendar features | Capture carryover and diminishing returns |
| Model Layer | Linear regression baseline, XGBoost with SHAP | Predict sales and attribute channels |
| Optimization | SciPy constrained optimizer | Recommend budget allocation |
| Delivery | Streamlit/Excel/PowerPoint | Present results to stakeholders |
| Monitoring | Holdout MAPE, business sanity checks | Detect model degradation |
| Feedback | Business team input, holdout actuals | Refine adstock parameters and channel definitions |

### Operational Runbook for MMM

**Data source delay:**
- Flag weeks with imputed spend
- Exclude high-imputation weeks from validation if needed
- Communicate delay impact to stakeholders

**Channel correlation detected:**
- Review VIF and correlation matrix
- Combine channels or apply regularization
- Re-run attribution and compare with business priors

**MAPE degradation:**
- Check for structural breaks (new product launch, competitor action)
- Validate adstock and saturation parameters
- Consider retraining with more recent data or adding external regressors

**Stakeholder challenge on attribution:**
- Show comparison across attribution models
- Explain assumptions and limitations
- Offer sensitivity analysis on adstock parameters

---

## EXL HEALTH READMISSION — DETAILED ARCHITECTURE AND OPS

### End-to-End Architecture Blocks

| Block | Components | Purpose |
|---|---|---|
| Ingestion | EHR, claims data, SDOH sources | Collect clinical and demographic data |
| Feature Engineering | Python/PySpark | Build diagnosis, procedure, comorbidity, prior admission features |
| Leakage Prevention | Temporal cutoffs before discharge | Ensure only pre-discharge features are used |
| Model Layer | Logistic regression, Random Forest | Predict 30-day readmission risk |
| Calibration | Platt scaling, Brier score | Ensure risk scores match true probabilities |
| Serving | Batch scoring or API integration | Provide risk scores to care coordinators |
| Monitoring | AUROC, precision at top decile, calibration | Track model performance |
| Feedback | Readmission outcomes, clinician input | Drive retraining and feature updates |

### Operational Runbook for Readmission Model

**Data leakage alert:**
- Audit feature timestamps against discharge date
- Remove any feature that could not have been known at discharge
- Add tests to enforce temporal cutoffs

**Calibration drift:**
- Re-run reliability diagrams on recent predictions
- Re-apply Platt scaling on a fresh validation set
- Communicate score interpretation changes to clinicians

**Clinician trust issue:**
- Provide SHAP-based reason codes
- Show decile performance and case examples
- Involve clinicians in feature definition and threshold selection

**Regulatory audit:**
- Provide model card with features, performance, and limitations
- Show no protected attributes used as features
- Document validation methodology and temporal splits

---

## PROJECT-SPECIFIC INCIDENT AND OPS STORIES

### Chubb: Memory Leak in FastAPI Scorer

> "We had a memory leak causing pods to OOM every 72 hours. I used Grafana to spot linear memory growth, then added tracemalloc profiling. The root cause was a SHAP explainer initialized per request. I moved it to startup, tested in staging for 5 days, and added a memory alert. Pod uptime went from 72 hours to 30+ days."

### Axtria: Client Disputed Channel Attribution

> "A client thought TV was under-credited. I built a side-by-side comparison of last-touch, first-touch, and Markov attribution on the same dataset. The Markov model showed TV's removal effect was larger than last-touch suggested. The client accepted the model after seeing the concrete example and sensitivity analysis."

### EXL: Temporal Leakage Almost Shipped

> "During validation I noticed a feature based on post-discharge diagnosis codes had leaked into training. I caught it because I enforce feature-availability timestamps. I removed the feature, rebuilt the model, and added a CI check that rejects any feature whose latest timestamp is after the prediction point."

---

## ADDITIONAL PROJECT FOLLOW-UPS

### "How do you ensure the Chubb fraud model is reproducible?"

> "We record the code commit, Delta Lake table version or timestamp, feature pipeline version, and model artifact in MLflow for every training run. To reproduce, I check out the code, read the data at the recorded version, and load the model by run_id."

### "What is the SLA for the Axtria MMM refresh?"

> "We refresh the MMM model monthly with the latest 3 years of weekly data. The optimization and reporting deck are delivered within 5 business days of data cutoff. If a major channel has missing data, we flag it and may delay the refresh."

### "How is the EXL readmission model deployed?"

> "It is deployed as a batch scorer that runs nightly and writes risk scores to a care management dashboard. Some clients also consume it through an internal API. We chose batch because care coordinators plan outreach the next day; real-time scoring was unnecessary."

