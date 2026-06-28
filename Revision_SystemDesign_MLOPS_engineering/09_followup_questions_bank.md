# FOLLOW-UP QUESTIONS BANK — Every Pattern You'll Face
> This covers every follow-up question pattern seen across your 11 interviews.

---

## CATEGORY 1: MODEL INTERNALS (Most common drill-down)

### XGBoost Internals
**Q: "What is gradient boosting really doing at each step?"**
> "At each step, we fit a new tree to the PSEUDO-RESIDUALS — the negative gradient of the loss function with respect to the current prediction. For log loss (classification): pseudo-residual for sample i = actual_i - predicted_prob_i. The new tree maps features → this residual. We add it (scaled by learning_rate) to the ensemble. This is identical to gradient descent in function space."

**Q: "What happens if learning rate is too high?"**
> "Each tree's contribution is not shrunk enough. The model fits training data aggressively → high variance → overfitting. Rule: lower learning rate + more trees (with early stopping) almost always beats high learning rate + fewer trees. Typical: lr=0.05, n_estimators=1000 with early_stopping_rounds=30."

**Q: "What is min_child_weight in XGBoost?"**
> "The minimum SUM OF HESSIANS in any leaf. For log loss, hessian_i = predicted_prob_i * (1 - predicted_prob_i). A leaf with low-probability fraud cases will have small hessians. min_child_weight=5 means: a leaf must have enough samples with meaningful hessian values. This prevents overly specific splits for rare fraud patterns — it's a regularization parameter."

**Q: "What is gamma in XGBoost?"**
> "Minimum loss reduction required to make a further partition. If splitting a node only reduces loss by 0.001 and gamma=0.5, the split is REJECTED. Equivalent to pruning: after building the full tree, splits that gave gain < gamma are removed. Higher gamma = simpler, more conservative trees. Typical for fraud: gamma=0.1-1.0."

**Q: "How does XGBoost handle class imbalance internally?"**
> "scale_pos_weight multiplies the gradient and second derivative (hessian) of positive class samples by that value. So a scale_pos_weight=49 means positive samples contribute 49x more to the objective than negatives. The model sees positive samples as 'more important mistakes to correct' — equivalent to oversampling positives 49x, but without data duplication."

---

### LightGBM Internals
**Q: "Why is LightGBM faster than XGBoost?"**
> "Three reasons: (1) HISTOGRAM algorithm — bins continuous features into 256 bins, so instead of scanning all N unique values for a split, it scans 256. O(N) → O(256) effectively. (2) GOSS — only uses samples with large gradients (hard examples) for split finding, ignoring easy examples. (3) Leaf-wise growth — grows the most promising leaf first, avoids spending time on uninformative nodes."

**Q: "When does LightGBM overfit more than XGBoost?"**
> "Leaf-wise growth can create very deep, asymmetric trees. On small datasets (<50K rows), these deep branches memorize training patterns. Control with: min_data_in_leaf (minimum 20+ for small data), num_leaves (start at 31, don't go above 127 without validation). On large datasets (>100K rows), leaf-wise is safer and more accurate."

---

### Decision Tree Internals
**Q: "What is the difference between Gini and Entropy splits? When to prefer one?"**
> "Both measure node impurity. Gini = 1 - sum(p_i^2), Entropy = -sum(p_i * log2(p_i)). Entropy is slightly more sensitive to imbalanced splits (log function amplifies small differences). Gini is faster (no log computation). In practice: they give nearly identical trees and accuracy. Use Gini (default) unless you have theoretical reason to prefer information-theoretic interpretation."

**Q: "How does a decision tree handle continuous features vs categorical?"**
> "Continuous: sorts by feature value, tries every unique value as threshold. O(N log N) per feature per split. Categorical: scikit-learn converts to numeric first (you must encode). Native categorical support in LightGBM: tries all 2^(k-1) splits for k categories — efficient for low cardinality only."

---

## CATEGORY 2: EVALUATION AND STATISTICS

**Q: "Harmonic mean vs arithmetic mean — why does F1 use harmonic mean?"**
> "Harmonic mean penalizes extreme imbalance. Arithmetic(25, 75) = 50 = Arithmetic(50, 50). But Harmonic(25, 75) = 37.5 ≠ Harmonic(50, 50) = 50. The harmonic mean says: a model that is great at one thing but terrible at another is NOT a good model. F1 is high ONLY when both P and R are high."

**Q: "What is the difference between accuracy and balanced accuracy?"**
> "Standard accuracy: (TP + TN) / N. For 98% not-fraud: predicting always 'not fraud' gives 98% accuracy — misleading. Balanced accuracy: average of recall for each class = (Recall_fraud + Recall_not_fraud) / 2. For all-not-fraud predictor: Recall_fraud=0, Recall_not_fraud=1 → balanced accuracy = 50%. Much more honest for imbalanced data."

**Q: "When is ROC-AUC misleading for imbalanced data?"**
> "ROC-AUC plots TPR vs FPR. FPR = FP / (FP + TN). When TN is massive (98% of data), FPR is tiny even if FP is large in absolute terms. A model can have 0.85 ROC-AUC but still generate thousands of false positives per day. PR-AUC uses precision (TP / (TP + FP)) which doesn't normalize by TN — so it directly shows how many of your positives are correct."

**Q: "What is statistical power and why does it matter for A/B testing?"**
> "Power = P(reject null | alternative is true) = probability of detecting a real effect. Power depends on: sample size (more → higher power), effect size (larger → higher power), alpha (higher α → higher power but more false positives). Rule of thumb: target 80% power (β=0.2), meaning if the new model is truly better, we detect it 80% of the time. Formula: n = 2 * (z_α + z_β)^2 * σ^2 / Δ^2. For fraud detection, online A/B tests need 2-4 weeks to accumulate enough confirmed labels."

**Q: "What is bootstrapping and when do you use it?"**
> "Sampling with replacement from your dataset to create many 'bootstrap samples'. Fit model on each → get distribution of metrics. Use for: confidence intervals on AUC when test set is small, comparing two models (paired bootstrap), estimating variance of predictions. For fraud: boot_auc_95CI = [boot_aucs.percentile(2.5), boot_aucs.percentile(97.5)] gives confidence range for reported AUC."

---

## CATEGORY 3: FEATURE ENGINEERING DEEP-DIVES

**Q: "What is target encoding and why is it dangerous?"**
> "Target encoding replaces a categorical value with the mean target (fraud rate) for that category. Danger: if you compute mean fraud rate using the same training set you train on, the model sees 'the fraud rate of category X is Y%' during training and overfits to it. Solution: k-fold out-of-fold encoding — for each sample, compute target encoding using only OTHER folds. This prevents leakage."

**Q: "What is WoE encoding and when is it better than target encoding?"**
> "WoE = ln(Distribution_Events / Distribution_Non_Events) for each category bin. Better than target encoding when: (1) using logistic regression (WoE makes log-odds relationship linear), (2) need monotonic binning (WoE bins can be merged to create monotone patterns), (3) need IV for variable selection. Target encoding is simpler and works for tree models."

**Q: "How do you deal with high-cardinality categoricals (1000+ unique values)?"**
> "Four approaches: (1) Target encoding with out-of-fold (my default for tree models). (2) Frequency encoding (replace with count of how often the value appears — simple but loses fraud-rate signal). (3) Embedding: learn a dense representation via neural network (for very high cardinality like zip codes). (4) Hashing: deterministic hash to N buckets (loses interpretability, sometimes used for real-time systems). For insurance fraud: zip code (1000+ values) → target encoding with 5-fold OOF or geographic clustering."

**Q: "What is a feature store and why do you need one?"**
> "A feature store is a centralized system for computing, storing, and serving features. It solves the training-serving skew problem: features are computed with the SAME code and stored to ensure training features = serving features. Without a feature store: training pipeline computes feature A one way, serving pipeline computes it slightly differently → subtle bugs in production. Online store (Redis) serves real-time features. Offline store (Delta Lake) stores historical features for training."

---

## CATEGORY 4: MLOPS AND PRODUCTION

**Q: "What is blue-green deployment? When would you use it?"**
> "Blue = current production model. Green = new model. You deploy green alongside blue, route 100% traffic to blue, validate green in shadow mode or with internal test traffic. When validated: switch load balancer to route 100% to green. If green fails: switch back to blue instantly (no downtime). Use when: zero-downtime deployment is required, you have enough infrastructure to run two full production environments."

**Q: "What is the difference between data drift and concept drift?"**
> "Data drift: the distribution of INPUT features changes. Example: fraud amounts increasing as economy changes. The relationship between features and outcome may be unchanged — the model still works, but its predictions shift because inputs changed.
>
> Concept drift: the RELATIONSHIP between inputs and output changes. Example: fraudsters learn your model's signals and start submitting claims that avoid those signals. Same features → different fraud probability. Model becomes stale even if input distribution is stable.
>
> Data drift: detectable with PSI on features. Concept drift: detectable only with labeled outcome data (SIU feedback)."

**Q: "How do you do a model rollback in practice?"**
> "MLflow registry maintains all model versions. Rollback = two steps: (1) transition previous production version back to 'Production' stage in MLflow (takes seconds), (2) Kubernetes deployment has model_uri as configmap — update configmap to point to previous version, pods restart and load the old model. Total rollback time: 2-5 minutes. We test rollback procedure quarterly as part of disaster recovery exercises."

**Q: "What is the difference between a scheduler (Airflow) and an orchestrator (Kubernetes)?"**
> "Airflow schedules and orchestrates JOBS — DAGs define task dependencies, timing, and retry logic. It manages the workflow: 'run this script at 2am, wait for it to complete, then run that script'. Kubernetes orchestrates CONTAINERS — manages which containers run where, scales them up/down, restarts failed ones. They complement each other: Airflow triggers a Kubernetes Job to run the batch scoring script."

**Q: "How do you prevent training-serving skew?"**
> "Three controls: (1) Feature store — same computation code for both training and serving. (2) Schema validation at inference — reject inputs that don't match training schema. (3) Integration tests — run the serving endpoint on training data, compare predictions to offline training predictions. If offline and online predictions differ by more than epsilon → skew detected. We do this check as part of CD pipeline before each deployment."

---

## CATEGORY 5: SYSTEM DESIGN FOLLOW-UPS

**Q: "Why Kafka over a simple REST API for real-time events?"**
> "REST API is synchronous — if the fraud scorer is slow, the claim submission blocks waiting. Kafka decouples them: claim submission publishes to Kafka (fast, returns immediately), fraud scorer consumes at its own pace. Benefits: (1) at-least-once delivery guarantee (messages are persisted), (2) natural buffer during traffic spikes, (3) multiple consumers can read the same event (fraud scorer + analytics + audit logger). Use REST when: tight coupling is acceptable, very low volume, need synchronous response."

**Q: "How does Redis work as a feature store?"**
> "Redis is an in-memory key-value store. Key = claimant_id. Value = JSON with pre-computed features (claims_90d, avg_amount, risk_score). TTL = 24h (expires and gets refreshed daily by batch job). Reads: <1ms (in-memory). The batch job computes all behavioral features overnight using Spark, then writes to Redis using pipeline commands (batch write for efficiency). At inference time: Redis GET(claimant_id) → features in <1ms."

**Q: "What happens when Redis is down?"**
> "Fallback strategy: (1) circuit breaker detects Redis timeout, (2) real-time model falls back to reduced feature set (claim payload features only — no behavioral history), (3) fraud probability is less accurate but system keeps running, (4) PagerDuty alert, (5) after Redis recovers, batch job refreshes all features. We accept degraded performance over complete outage."

**Q: "How do you partition a Spark job to avoid data skew?"**
> "Data skew: one partition has 10M rows, others have 10K → one executor takes 100x longer. Solutions: (1) Repartition by a high-cardinality key (claim_id, not claim_type which may be imbalanced). (2) Salting: append random prefix to skewed keys (provider_id + random(0,10)) to spread data across partitions. (3) Use coalesce before write to combine small partitions. Monitor with Spark UI: if one task takes 10x longer than others → skew."

---

## CATEGORY 6: AGENTIC AI AND RAG

**Q: "What is the difference between RAG and fine-tuning? When to use each?"**
> "RAG: at inference time, retrieve relevant context from a knowledge base and inject it into the prompt. Dynamic — update knowledge base without retraining. Fine-tuning: train model on domain-specific data to bake in knowledge. Static — must retrain to update.
>
> Use RAG when: knowledge changes frequently (fraud patterns), need to cite sources, don't have GPU budget for fine-tuning, knowledge base is large (100K+ documents).
>
> Use fine-tuning when: need specific style/format, knowledge is stable, need very fast inference (no retrieval latency), domain is highly specialized (medical coding, legal)."

**Q: "What is hallucination in LLMs and how do you prevent it in production?"**
> "Hallucination: LLM generates plausible-sounding but factually incorrect information. In our RAG system: extracting fraud flags that aren't in the claim text.
>
> Prevention: (1) Structured output with JSON schema — LLM can't hallucinate outside defined format. (2) Faithfulness evaluation: LLM-as-judge checks if each extracted flag is grounded in retrieved text. (3) Human-in-the-loop: underwriters validate RAG outputs on a sample weekly. (4) Temperature=0 for extraction tasks — deterministic, less creative. (5) Explicit instruction: 'Only extract patterns you can cite directly from the text. If not present, return false.'"

**Q: "What is context window and why does chunking matter?"**
> "Context window = maximum tokens the LLM can process in one call (GPT-4o: 128K tokens). Claim notes can be 50K+ words across multiple documents. Options: (1) Stuff everything in context (expensive, hits limits), (2) Chunk and process separately (miss cross-document patterns), (3) RAG: chunk, embed, retrieve only relevant chunks (our approach). Chunk size = 500 tokens: large enough for context, small enough for precise retrieval. Overlap = 100 tokens: prevents losing patterns split across chunk boundaries."

---

## CATEGORY 7: BEHAVIORAL / LEADERSHIP

**Q: "Tell me about a time you disagreed with a stakeholder's technical decision"**
> "At Axtria, the business team wanted to use first-touch attribution for all marketing channels. I disagreed because it would systematically undervalue bottom-funnel channels like search.
>
> My approach: I built a comparison showing three attribution models (first-touch, last-touch, Markov) on the same dataset. Each model produced different channel ROIs and therefore different budget recommendations. I showed: if they followed first-touch attribution, they'd cut digital spend by 40% — but digital was the channel closest to conversion.
>
> Resolution: we agreed to use Markov chain attribution as the primary model, with first-touch as a supplementary view for brand-awareness decisions. The stakeholder got their first-touch view; we got a more accurate primary model."

**Q: "How do you mentor junior data scientists?"**
> "Two principles: (1) Give them ownership, not tasks. I assign junior team members full responsibility for a feature or evaluation metric — not just 'write this code'. They learn end-to-end.
>
> (2) Code review as teaching. When I review their code, I don't just fix — I explain WHY. 'This nested function makes testing impossible. Refactor to named methods. Here's why that matters for our CI pipeline.'
>
> I also set a rule: before using any complex algorithm, they must show me the simple baseline failed. This teaches the simplicity-first mindset early."

**Q: "Where do you want to be in 3 years?"**
> "In a Lead Data Scientist or Head of AI role where I'm setting technical direction for a team of 5-8 data scientists and engineers. Not just building models — designing the ML platform, defining evaluation frameworks, and mentoring the team to ship production-quality ML systems.
>
> I'm building toward that now: at Chubb I've led technical design decisions even in an IC role. My next step is formalizing that into a people leadership mandate alongside technical ownership.

---

## CATEGORY 8: STATISTICAL TESTS AND EXPERIMENT DESIGN

**Q: "When would you use a t-test vs Mann-Whitney U?"**
> "Use a t-test when the data is approximately normal and you want to compare means. Use Mann-Whitney U when the distribution is skewed or has outliers, because it tests differences in medians using ranks and does not assume normality. For insurance claim amounts, which are heavily right-skewed, I would use Mann-Whitney U."

**Q: "What is the difference between ANOVA and ANCOVA?"**
> "ANOVA compares means across three or more groups. ANCOVA does the same but adjusts for one or more continuous covariates. For example, comparing fraud rates across regions with ANCOVA would adjust for average claim amount so the comparison is fairer."

**Q: "When do you use non-parametric tests?"**
> "When data violates normality, when sample size is small, or when working with ordinal data. Common non-parametric tests: Mann-Whitney U for two groups, Kruskal-Wallis for three or more groups, Wilcoxon signed-rank for paired data."

**Q: "How do you test if a sample is normally distributed?"**
> "I use Shapiro-Wilk for small samples and Anderson-Darling or Kolmogorov-Smirnov for larger samples. I also visually inspect a Q-Q plot. If the distribution is skewed, I consider a Box-Cox or Yeo-Johnson transformation before parametric tests."

**Q: "What is the difference between statistical significance and business significance?"**
> "Statistical significance means an effect is unlikely due to chance. Business significance means the effect is large enough to matter. A 0.1% improvement in click-through rate can be statistically significant with millions of users but have no business impact. I always report both."

---

## CATEGORY 9: MULTI-LABEL AND MULTI-CLASS EVALUATION

**Q: "How do you evaluate a multi-label classifier?"**
> "For multi-label, each sample can have multiple labels. I use micro-averaged F1 if I care about overall performance across all labels, macro-averaged F1 if I want equal weight for each label, and weighted F1 if labels have different frequencies. I also report per-label precision and recall to identify weak classes."

**Q: "What is the difference between micro, macro, and weighted averages?"**
> "Micro averages aggregate contributions globally, so frequent labels dominate. Macro averages compute the metric for each label and then average, giving rare labels equal weight. Weighted averages compute per-label metrics and weight by support. Use macro when rare labels matter; use micro when overall error is most important."

**Q: "When do you use F-beta over F1?"**
> "Use F-beta when precision and recall are not equally important. F-beta generalizes F1 with a beta parameter. F-2 weights recall higher; F-0.5 weights precision higher. For fraud detection where missing fraud is costly, I might use F-2. For a review queue limited by investigator capacity, I might use F-0.5."

**Q: "How do you handle label hierarchy in multi-class NER?"**
> "I enforce a strict output schema and post-process predictions to respect hierarchy. For example, if the model predicts both B-PER and I-ORG for adjacent tokens, the decoder resolves the conflict using a validity map. I also evaluate using entity-level F1, not token-level, because partial entity matches are wrong in practice."

---

## CATEGORY 10: LLM EVALUATION AND GUARDRAILS

**Q: "What metrics beyond accuracy do you use for LLMs?"**
> "For extraction tasks, I use faithfulness, answer relevancy, and schema adherence. For generation tasks, I use BLEU/ROUGE for overlap and LLM-as-judge for quality. For RAG, I use retrieval Recall@K and context recall. The most important metric is usually downstream business impact: does the LLM output improve the primary model or user outcome?"

**Q: "How do you enforce structured output from an LLM?"**
> "I use function calling or JSON mode with a strict schema. I validate the output with Pydantic. If validation fails, I retry with a lower temperature or a stronger prompt. For critical tasks, I add a deterministic fallback when the LLM repeatedly fails."

**Q: "What guardrails do you put around agentic systems?"**> "Input guardrails reject toxic, off-topic, or PII-laden requests. Output guardrails enforce schema and factual constraints. Operational guardrails limit the number of tool calls, total cost, and execution time. Reflection layers verify outputs before they are returned."

**Q: "How do you measure hallucination?"**> "I measure hallucination by comparing generated claims to the retrieved or provided context. Metrics include faithfulness score, citation coverage, and human evaluation. For critical outputs, I require the model to cite source passages and flag any claim without a citation."

---

## CATEGORY 11: CLOUD AND MLOps FOLLOW-UPS

**Q: "When do you use Databricks vs managed services like SageMaker or Vertex AI?"**> "I use Databricks when I need a unified lakehouse with Spark, Delta Lake, and MLflow in one platform. I use managed services when the team wants to reduce operational overhead and use pre-built training and serving infrastructure. The choice depends on team size, existing cloud commitments, and need for control."

**Q: "How do you monitor data drift and concept drift?"**> "Data drift is monitored with PSI and KS tests on input features. Concept drift is monitored using lagged ground truth metrics such as precision, recall, and PR-AUC. Prediction drift is monitored by comparing the distribution of model scores over time. Each type of drift requires a different response."

**Q: "What is the difference between a feature store and a data warehouse?"**> "A feature store provides point-in-time correct features, training-serving consistency, and low-latency serving. A data warehouse stores historical data for analytics but does not guarantee consistency between training and online inference. A feature store sits on top of the data warehouse and adds ML-specific abstractions."

**Q: "How do you rollback a model in production?"**> "I keep the previous model version in the MLflow Production stage history. To rollback, I transition the previous version back to Production and update the serving configuration. Kubernetes pods load the new old version. The process takes 2 to 5 minutes if automated."

---

## CATEGORY 12: SYSTEM DESIGN TRADE-OFFS

**Q: "When do you use Kubernetes over serverless for ML serving?"**> "Kubernetes when latency is critical, traffic is sustained, and you need fine-grained deployment control like canary and blue-green. Serverless when traffic is sporadic, latency is flexible, and you want to minimize operational overhead. For fraud scoring with a 200ms SLA, I choose Kubernetes."

**Q: "How do you choose a vector database index?"**> "I benchmark Flat for recall baseline, IVF for balanced speed and recall at medium scale, and HNSW for high-recall low-latency at large scale. I also consider memory budget, metadata filtering needs, and whether hybrid search is required."

**Q: "SQL blocking vs graph for fraud rings — which first?"**> "SQL blocking first. It is O(N) with indexing and finds most rings that share attributes. Graph is added only when rings obfuscate direct links and the loss exposure justifies the engineering cost."

**Q: "Batch vs real-time scoring — how do you decide?"**> "If the business action requires an immediate decision, real-time is required. If the action can wait hours, batch is cheaper and can use richer features. Most systems use a hybrid: real-time for speed and batch for comprehensive accuracy."

---

## CATEGORY 13: BEHAVIORAL AND LEADERSHIP FOLLOW-UPS

**Q: "How do you handle a project that misses its target?"**> "I diagnose the gap: was it data quality, model performance, or business assumptions? I communicate early to stakeholders with a clear recovery plan. I also capture lessons learned and update project scoping for next time."

**Q: "How do you prioritize multiple ML projects?"**> "I prioritize by expected business value, data readiness, model feasibility, and strategic alignment. A project with high value and clean data should go first. A high-value project with poor data quality needs a data collection phase before modeling."

**Q: "What would your first 90 days look like in a lead role?"**> "First 30 days: understand current systems, team strengths, and key pain points. Days 30-60: deliver a quick win, often a monitoring dashboard or baseline model fix. Days 60-90: define roadmap, evaluation framework, and team rituals. I avoid changing everything at once."
"
