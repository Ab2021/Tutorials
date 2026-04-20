# 🧠 Key Interview Questions & Model Answers
### Tailored for: Adhish Prasoon | Research Director, DS @ Flipkart
### PhD (Machine Learning, University of Copenhagen) | M.Tech IIT Bombay

> **Key Intel on Adhish:**
> - PhD focused on **voxel/pixel classification for medical image segmentation** using deep learning + CNNs (MICCAI 2013, triplanar CNN paper)
> - Published: *"Portfolio Risk Management Model for EMI-based loan in E-Commerce"* — 300+ features, transaction-level delinquency prediction
> - Former SVP Data Science at Info Edge (Naukri, Jeevansathi, 99acres)
> - Academic + industry hybrid thinker → expect **mathematical depth + practical grounding**
> - Will push you on fundamentals, derivations, and the "WHY" behind every choice

---

## SECTION A: ML FUNDAMENTALS & MATHEMATICAL DEPTH

> ⚠️ Adhish will go deep. Don't just say "XGBoost works well here" — explain WHY mathematically.

---

### Q1: Derive the Logistic Regression loss function from first principles.

**Model Answer:**
Logistic Regression models the probability of class 1 as:
$$P(y=1|x) = \sigma(w^Tx) = \frac{1}{1+e^{-w^Tx}}$$

We assume labels y ∈ {0,1} are Bernoulli-distributed:
$$P(y|x; w) = \hat{y}^y (1-\hat{y})^{1-y}$$

Taking log-likelihood over N samples:
$$\ell(w) = \sum_{i=1}^N [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$

Maximizing log-likelihood = minimizing **Binary Cross-Entropy Loss:**
$$\mathcal{L}(w) = -\frac{1}{N} \sum_{i=1}^N [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$

Gradient w.r.t. w:
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{N} X^T (\hat{y} - y)$$

With L2 regularization: add λ||w||² → gradient becomes: X^T(ŷ - y)/N + 2λw

**Adhish follow-up:** *"Why is log-loss preferred over MSE for classification?"*
> MSE loss with sigmoid creates vanishing gradients — the gradient is σ(z)(1-σ(z))*(ŷ-y), which becomes tiny at extremes. Cross-entropy gradient is simply (ŷ - y), avoiding vanishing gradient problem. Also, cross-entropy is the proper probabilistic loss — it maximizes likelihood under Bernoulli assumption.

---

### Q2: Explain the Bias-Variance Tradeoff. How does it manifest in real fraud detection systems?

**Model Answer:**
$$\text{Expected Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **Bias:** Error from wrong assumptions (underfitting — model too simple)
- **Variance:** Error from sensitivity to training set fluctuations (overfitting — model too complex)
- **Irreducible Noise:** Inherent noise in data

**In Fraud Detection Context (Chubb):**
- **High Bias trap:** Simple logistic regression on aggregated features — misses subtle behavioral patterns
- **High Variance trap:** Deep neural network trained on 50K fraud claims — overfits to historical fraud patterns, misses novel fraud schemes
- **My solution:** XGBoost ensemble (moderate complexity) + regularization (L1 for feature selection, L2 for weight decay) + cross-validation to tune depth/n_estimators
- **At inference:** Calibrated probability scores (Platt scaling / isotonic regression) — critical in fraud where business needs well-calibrated risk scores for threshold setting

---

### Q3: Explain Gradient Boosting (XGBoost) mathematically. Why does it work well for tabular fraud data?

**Model Answer:**
Gradient Boosting builds additive models iteratively:
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

At each step, fit h_m to **negative gradient of loss** w.r.t. current predictions:
$$r_{im} = -\frac{\partial \mathcal{L}(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$$

For log-loss: residuals = (y_i - ŷ_i) = true label minus predicted probability

**XGBoost additions over vanilla GBM:**
- **2nd-order Taylor expansion** of loss (uses curvature) → better optimization
- **Regularization:** L1 (leaf weights) + L2 (sum of leaf weights) → prevents overfitting
- **Column subsampling:** Like Random Forest, reduces correlation between trees
- **Level-wise vs. leaf-wise growth:** XGBoost is level-wise (safer); LightGBM leaf-wise (faster)

**Why XGBoost for fraud tabular data:**
- Fraud features are inherently tabular (transaction amount, merchant category, time of day, velocity features)
- Gradient Boosting captures non-linear interactions automatically
- Built-in handling of missing values (learns default split direction)
- Feature importance (gain-based) aids explainability for regulatory requirements
- Outperforms deep learning on small-medium tabular datasets (typical fraud training sets)

---

### Q4: How do CNNs capture spatial hierarchies? (Adhish's PhD area — expect this connection)

**Model Answer:**
CNNs learn hierarchical spatial representations through:
1. **Local connectivity:** Filters cover receptive fields, not all pixels → translation invariance
2. **Parameter sharing:** Same filter across spatial locations → drastically reduces parameters
3. **Hierarchical composition:** Layer 1 detects edges → Layer 2 detects shapes → Layer 3 detects objects

**Adhish's triplanar CNN (MICCAI 2013):**
His PhD work used 3 separate 2D CNNs on the xy, yz, zx planes of a 3D medical volume to classify each voxel — this is a clever approximation of full 3D convolution at a fraction of the compute cost.

**How to connect this to your work:**
> "Your triplanar CNN approach for voxel classification was really elegant — using 2D CNNs as a proxy for 3D understanding. I see a parallel in how I approached fraud detection — rather than one monolithic model, I used specialized sub-models for different claim types (medical vs. liability vs. property), an ensemble that captures domain-specific patterns, similar to your plane-specific CNN specialization."

---

### Q5: Explain BERT's attention mechanism and why it's useful for NLP in fraud/claims.

**Model Answer:**
**Self-Attention (Scaled Dot-Product):**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where Q (Query), K (Key), V (Value) are linear projections of the input.

**Multi-Head Attention:** H parallel attention heads, each learning different aspects of relationships:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, ..., head_H) W^O$$

**BERT specifics:**
- **Bidirectional:** Unlike GPT (left-to-right), BERT sees full context in both directions
- **Pre-training:** Masked Language Modeling (predict masked tokens) + Next Sentence Prediction
- **Fine-tuning:** Add task-specific classification head; fine-tune on domain data

**Why critical for claims NLP:**
- Long-range dependencies: "injury occurred on DATE" → connected to "procedure performed on DATE +60 days" → inconsistency
- Bidirectionality: "patient was not injured" vs. "patient was injured" — full context matters for negation detection
- Domain fine-tuning: ClinicalBERT / LegalBERT variants capture industry-specific language

---

### Q6: What is the ROC-AUC vs. PR-AUC distinction? When does each matter in fraud?

**Model Answer:**

| Metric | Description | When To Use |
|---|---|---|
| **ROC-AUC** | Area under TPR vs. FPR curve | When negative class is NOT important (rare fraud, imbalanced data where you care about rank ordering) |
| **PR-AUC** | Area under Precision vs. Recall curve | When **positive class is rare** and false positives are costly — standard for fraud |
| **KS Statistic** | Max separation between fraud and non-fraud cumulative distributions | Risk model comparison, regulatory reporting |

**In fraud context:**
- Class imbalance: 1 fraud per 1000 transactions → ROC-AUC looks great even with poor precision
- PR-AUC exposes actual precision at meaningful recall thresholds
- **Business translation:** At 5% alert rate (recall 80%), what's precision? That's the actionable metric for investigation team sizing
- Also use: **F-beta with β < 1** when precision matters more than recall (too many false positives = investigator burnout)

---

## SECTION B: ML SYSTEM DESIGN

> Adhish's published work on EMI risk modeling shows he designs end-to-end production risk systems. Expect system design questions framed as business problems.

---

### Q7: DESIGN — Fraud Detection System for Flipkart at Scale

**Interviewer**: *"Design an end-to-end fraud detection system for Flipkart that handles both real-time transaction fraud and return fraud."*

**Framework to use: CIRCLE (Context, Identify, Rank, Cut, List, Evaluate)**

#### 1. Clarify & Scope
- Scale: 350M customers, 10M+ daily transactions, 500K+ sellers
- Two fraud types: Transaction fraud (payment) + Return/Seller fraud (behavioral)
- Latency: <100ms for real-time transaction scoring; hours OK for return fraud auditing
- Business cost of FP vs FN: FN (missed fraud) >> FP (false flag) for financial fraud; opposite for return fraud

#### 2. Data Sources & Features
```
Transaction Fraud Features:
├── User: account age, past fraud history, device fingerprint, IP geolocation
├── Transaction: amount, merchant, time of day, payment method, velocity (txns/hour)
├── Behavioral: session duration, page sequence, click patterns (mouse/touch)
└── Network: shared device IDs, shared addresses, graph-based ring detection

Return Fraud Features:
├── Return history: return rate, avg days to return, claimed reasons
├── Product: high-value category (phones, luxury), serial number linkage
├── Seller signals: seller rating trajectory, dispute frequency
├── Image analysis: AI X-ray scan similarity scores (Flipkart uses this in practice)
└── Network: linked accounts returning same products
```

#### 3. Model Architecture

```
REAL-TIME FRAUD (<100ms):
User Request → Feature Store (Redis) → 
    Lightweight GBM/XGBoost (pre-computed features) → 
    Risk Score → Threshold → Allow/Block/Step-up Auth

BEHAVIORAL FRAUD (Hours):
Daily batch → Graph Neural Network (seller ring detection) →
    Fraud Linkage Verification (device/address graph) →
    Human review queue prioritized by risk score

RETURN FRAUD:
Return event → Rules engine (first pass) →
    ML model (Random Forest + behavioral features) →
    Computer vision (product image validation) →
    Decision: approve/flag/reject
```

#### 4. Feature Store Design (Critical!)
- **Online feature store:** Redis/DynamoDB — pre-computed user features updated in real-time
- **Offline feature store:** BigQuery/Hive — historical aggregations, training data
- **Streaming:** Kafka → Flink/Spark Streaming → update velocity features in real-time

#### 5. Model Training & Deployment
- **Training:** Weekly retraining on rolling 90-day window (fraud patterns shift rapidly)
- **Champion-Challenger:** 90% traffic to champion model, 10% to challenger
- **Monitoring:** Distribution shift (PSI), performance drift (AUC drop >2% triggers alert)
- **Explainability:** SHAP values for each fraud flag → human reviewer sees top 3 reasons

#### 6. Operational Considerations
- **Human-in-the-loop:** High-value transactions (>₹50K) → mandatory human review
- **Feedback loop:** Investigator decisions → labeled data → model retraining pipeline
- **Adversarial robustness:** Fraudsters adapt → periodic red-team exercises, concept drift monitoring

---

### Q8: DESIGN — AI Agentic System for Automated Risk Investigation at Flipkart

**Interviewer:** *"How would you use LLMs/Agents to automate or augment the fraud investigation workflow?"*

#### Proposed Architecture: FraudInvestigatorAgent

```
TRIGGER: Fraud flag raised by detection system
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              FraudInvestigator ReAct Agent              │
│                                                         │
│  Available Tools:                                       │
│  ├── get_user_history(user_id) → transaction history    │
│  ├── get_network_graph(user_id) → linked accounts       │
│  ├── search_fraud_kb(pattern) → similar past cases (RAG)│
│  ├── query_returns_db(user_id) → return history         │
│  └── generate_investigation_report() → structured output│
│                                                         │
│  ReAct Loop:                                            │
│  Thought → Action (tool call) → Observation → Repeat   │
│                                                         │
│  Output: Structured Investigation Report                │
│  - Risk verdict: HIGH/MEDIUM/LOW                        │
│  - Evidence list (linked to source data)                │
│  - Recommended action                                   │
│  - Confidence score                                     │
└─────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **Guardrails:** Output schema enforcement, no hallucination of evidence
- **RAG knowledge base:** Past investigation reports → vectorized → similar case retrieval
- **Human-in-the-loop:** Agent produces report, human makes final decision
- **Audit trail:** Every tool call logged for regulatory compliance

---

### Q9: How would you handle class imbalance in Flipkart's fraud data (1 fraud in 1000 transactions)?

**Model Answer (structured):**

#### Data-level techniques:
- **SMOTE** (Synthetic Minority Oversampling Technique): Generate synthetic fraud samples by interpolating between k-nearest neighbors in feature space
- **Random Undersampling:** Remove majority class samples (risk: losing information)
- **Tomek Links:** Remove borderline majority samples near decision boundary

#### Algorithm-level techniques:
- **Class weights:** `class_weight='balanced'` in sklearn, or `scale_pos_weight=999` in XGBoost (N_neg/N_pos ratio)
- **Cost-sensitive learning:** Asymmetric loss function — penalize FN 10x more than FP

#### Threshold calibration:
- Never use default 0.5 threshold in imbalanced settings
- Use PR curve to find optimal threshold for business constraint (e.g., "flag top 1% of transactions")
- **F-beta score** to tune: β=0.5 (precision-heavy) or β=2 (recall-heavy)

#### Evaluation:
- **NEVER use accuracy** — 99.9% accuracy if you predict all non-fraud
- Primary: **PR-AUC + Recall at 95% Precision**
- Secondary: **KS statistic** (max separation between fraud/non-fraud score distributions)

**My experience at Chubb:**
> Class imbalance was severe in insurance fraud (< 2% claims are fraudulent). I used a combination of: XGBoost with `scale_pos_weight`, calibrated probabilities using Platt scaling, and tuned threshold to the business constraint of maximum 200 investigations/week — derived the threshold from the precision-recall curve.

---

## SECTION C: GENERATIVE AI & AGENTIC AI (HIGH RELEVANCE — Flipkart's 2025 AI focus)

---

### Q10: What is RAG (Retrieval-Augmented Generation)? When would you NOT use RAG?

**Model Answer:**
RAG = Retrieve relevant documents from a knowledge base → Augment LLM prompt with retrieved context → Generate grounded response

**Architecture:**
```
Query → Embedding → Vector DB Search → 
    Top-k Documents → Augmented Prompt → LLM → Response
```

**When RAG works well:**
- Knowledge is frequently updated (fraud patterns, policy documents)
- Need source attribution (compliance/regulatory needs)
- Proprietary knowledge not in LLM training data
- Hallucination risk is high (claims, legal, medical)

**When NOT to use RAG:**
- Simple classification/extraction tasks → fine-tune a smaller model
- Knowledge is static and compact → just put it in system prompt (context stuffing)
- Ultra-low latency required (<50ms) → RAG adds retrieval latency
- Knowledge fits in context window → no retrieval needed

**Evaluation framework — RAGAS:**
- **Faithfulness:** Generated answer supported by retrieved context
- **Answer Relevance:** Does generated answer address the query?
- **Context Recall:** Did retrieval find all relevant documents?
- **Context Precision:** Are retrieved documents actually relevant?

---

### Q11: Explain the difference between RAG, Fine-Tuning, and Prompt Engineering. When to use each?

| Approach | What Changes | When To Use | Cost |
|---|---|---|---|
| **Prompt Engineering** | System prompt, few-shot examples | Simple format changes, behavioral nudges | Low |
| **RAG** | External knowledge base | Factual, updatable knowledge | Medium |
| **Fine-Tuning** | Model weights | Domain style, format, behavior that can't be prompted | High |
| **RLHF/DPO** | Model preferences via reward | Alignment, safety, preference optimization | Very High |

**Key insight for Flipkart context:**
- For fraud pattern knowledge → **RAG** (fraud patterns evolve, need updates)
- For Flipkart-specific tone/format in seller communications → **Fine-tuning**
- For triksha (adversarial security framework) → **RLHF** concepts for robustness

---

### Q12: What are the main failure modes of LLM-based agents in production?

**Model Answer (from production experience at Chubb):**

1. **Hallucination / Confabulation:** LLM fabricates evidence not in source data
   - **Mitigation:** Strict grounding, output schema validation, source citation requirement

2. **Tool call loops:** Agent gets stuck in repetitive tool calls (infinite reasoning loop)
   - **Mitigation:** Max iteration limit, loop detection, fallback to default path

3. **Context window overflow:** Long investigations exhaust token budget
   - **Mitigation:** Hierarchical summarization, selective context pruning

4. **Latency:** Multi-step tool calls add latency; not suitable for real-time fraud scoring
   - **Mitigation:** Async agent for investigation; synchronous ML model for real-time blocking

5. **Adversarial prompt injection:** Malicious input hijacks agent behavior (Triksha addresses this at Flipkart)
   - **Mitigation:** Input sanitization, sandboxed tool execution, output validation

6. **Non-determinism:** Same input → different outputs; unpredictable in regulatory environments
   - **Mitigation:** Temperature=0, structured output schemas, deterministic tool calls

---

## SECTION D: STATISTICS & PROBABILITY

> Adhish has strong quantitative background — expect probability and Bayesian reasoning questions.

---

### Q13: A/B Testing — How would you design an experiment to measure the lift of a new fraud detection model?

**Model Answer:**

#### Experimental Design
- **Unit of randomization:** User-level (not transaction-level) to avoid cross-contamination
- **Traffic split:** 50/50 control (old model) vs. treatment (new model)
- **Duration:** Run for minimum 2 weeks (capture weekly seasonality, weekend vs. weekday)
- **Minimum detectable effect:** Pre-compute sample size using: n = 2σ²(z_α + z_β)² / δ²

#### Key Metrics
- **Primary:** Fraud catch rate (recall at fixed FPR) — business impact direct
- **Secondary:** False positive rate, precision, total financial loss
- **Guardrail:** False positive rate on good users — must not increase > 0.5%

#### Statistical Validity
- Use **Mann-Whitney U test** (non-parametric) if fraud rates are non-normal
- Multiple comparison correction (Bonferroni) if testing multiple segments
- **Sequential testing (optional):** Stop early if statistical significance reached, using spending functions (Pocock/O'Brien-Fleming boundaries)

#### Challenges in Fraud A/B Testing
- **Network effects:** Fraudsters share tactics → contamination between groups
- **Novelty effect:** New model catches new fraud types, old model improved by learning
- **Survivor bias:** Already-blocked fraudsters not in experiment

---

### Q14: Explain Bayesian vs. Frequentist approaches. When would you use each in DS at Flipkart?

**Model Answer:**

| | Frequentist | Bayesian |
|---|---|---|
| **Probability** | Long-run frequency | Degree of belief |
| **Parameters** | Fixed but unknown | Random variables with distributions |
| **Inference** | p-values, confidence intervals | Posterior distributions, credible intervals |
| **Prior info** | Ignored | Incorporated via prior |

**At Flipkart use cases:**

**Frequentist:** A/B testing (large samples, clear hypothesis) — classic hypothesis testing framework
**Bayesian:** 
- **Cold start problem:** New seller fraud risk — use prior from similar seller category, update with observed behavior
- **Multi-armed bandit for fraud intervention:** Thompson Sampling (Bayesian) to explore intervention strategies
- **Calibration:** Bayesian model averaging for ensemble uncertainty quantification
- **Small data:** Few frauds in new product category → Bayesian hierarchical model borrows strength across categories

---

### Q15: What is the difference between precision, recall, and F1? Design an operating point for Flipkart fraud.

**Model Answer:**

$$\text{Precision} = \frac{TP}{TP+FP}$$  → Of predicted fraud, how many are actually fraud?

$$\text{Recall} = \frac{TP}{TP+FN}$$  → Of actual fraud, how many did we catch?

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Business translation for Flipkart fraud operations:**

| Threshold Setting | Precision | Recall | Consequence |
|---|---|---|---|
| Very low (catch all) | Low | High | Investigation team overwhelmed with false positives |
| Very high (conservative) | High | Low | Massive financial loss from missed fraud |
| **Optimal** | ~80% | ~75% | Balanced: sustainable investigation queue + acceptable loss |

**My recommendation:**
- Run PR curve on validation set
- Define **business constraint:** "Investigation team can handle 500 cases/day"
- Find threshold that gives ≤500 flags/day at maximum recall
- This is a **precision-constrained recall optimization** problem
- Re-evaluate quarterly as investigation team capacity changes

---

## SECTION E: BEHAVIORAL & LEADERSHIP QUESTIONS

> Flipkart values: Customer-First | Audacity | Bias for Action | Ownership | Integrity

---

### Q16: Tell me about a time you built something that didn't work as expected. What did you learn?

**Structured Answer (STAR):**
> "At Axtria, I built a multi-touch attribution model for J&J that looked great on holdout data — PR-AUC of 0.82. But when deployed, marketing teams found the Markov Chain attribution was undervaluing digital channels compared to their intuition.
>
> The issue? I optimized for historical correlation, not causality. Digital touchpoints appeared early in the funnel (low conversion credit) but were causally driving brand awareness that converted weeks later.
>
> **What I did:** Introduced a time-decay weighting into the Markov Chain transition matrix, and validated with a small controlled budget experiment. Revised model showed +15% budget efficiency.
>
> **Learning:** Models that optimize for historical correlation without causal validation can mislead business decisions. Always close the loop with business experiments."

---

### Q17: How do you prioritize work when you have 3 DS projects, limited compute, and competing stakeholders?

**Answer:**
> "I use a **ROI × Effort matrix** — map each project on: Expected business value (revenue impact/risk reduction) vs. technical complexity + compute needs.
>
> I also distinguish between: 'urgent and important' (production model breaking), 'important not urgent' (new fraud detection capability), and 'urgent not important' (ad-hoc report requests).
>
> At Chubb, I was simultaneously running the fraud detection system, the Agentic BI tool POC, and claims summarization. I split my time: 60% fraud detection (production, highest business value), 25% claims summarization (executive visibility), 15% Agentic BI (innovation POC).
>
> I communicate trade-offs clearly to stakeholders — 'we can start X next sprint, here's the current project timeline' — rather than over-promising."

---

### Q18: Describe how you translate a vague business problem into an ML solution.

**Answer:**
> "My framework: **Problem → Metric → Data → Model → Deploy → Monitor**
>
> Business says: 'We're losing money to fraudulent returns.'
>
> I ask: How much? Per quarter? What's the current detection rate? What's the acceptable FP rate (we don't want to frustrate genuine customers)?
>
> This gives me: quantified business metric (reduce return fraud loss by 20%), evaluation metric (recall at 95% precision), constraints (data: 18 months of return history, compute: real-time scoring budget), success criteria.
>
> Then I work backward from the success metric to: feature engineering → model selection → training pipeline → A/B test → production deployment → business dashboard.
>
> The ML model is the middle step, not the first step."

---

## SECTION F: QUESTIONS TO ASK ADHISH

> Asking smart questions shows you've done your homework on HIS work. This is a differentiator.

1. **"I read your paper on Portfolio Risk Management for EMI-based loans at Flipkart — the 300+ feature approach for transaction-level delinquency prediction is fascinating. How did you handle feature staleness and model drift in that production setup? Did you end up using streaming features or batch-computed aggregates?"**

2. **"Given Flipkart's Triksha framework for adversarial LLM security — how does the team balance security robustness with the engineering cost of maintaining adversarial test suites as the underlying LLMs get updated?"**

3. **"The 'buy vs. build' philosophy for LLMs at Flipkart — where do you currently see the line for what's built in-house? Is there active work on domain-specific smaller models for e-commerce tasks?"**

4. **"What does the data science team structure look like for risk/fraud at Flipkart? Is it organized by product vertical, or is there a centralized risk data science function?"**

5. **"What would success look like in this role in the first 6 months? What's the most critical problem you'd want someone coming in to own?"**

---

## 📋 QUICK REFERENCE: FORMULAS TO HAVE READY

| Concept | Formula |
|---|---|
| Cross-Entropy Loss | -Σ yᵢ log(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ) |
| KL Divergence | Σ P(x) log(P(x)/Q(x)) |
| Bayes' Theorem | P(A|B) = P(B|A)·P(A) / P(B) |
| Information Gain | H(S) - Σ (|Sᵥ|/|S|)·H(Sᵥ) |
| SMOTE (conceptual) | x_new = xᵢ + λ·(x_neighbor - xᵢ), λ~U(0,1) |
| Attention | softmax(QKᵀ/√dₖ)·V |
| AUC-ROC | ∫₀¹ TPR(FPR) d(FPR) |
| F-beta | (1+β²)·P·R / (β²·P + R) |
| Gradient Boosting Step | F_m = F_{m-1} + η·h_m(x), h_m fits -∂L/∂F_{m-1} |
| XGBoost Leaf Score | -Σgᵢ/(Σhᵢ + λ) |

---

*End of Key Interview Questions Document*
