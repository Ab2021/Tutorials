# FRAUD ANALYTICS MASTERY — Specific Q&A for Lead Roles
> This covers every fraud-specific question pattern seen across your interviews with crisp answers.

---

## SECTION 1: FRAUD DETECTION FUNDAMENTALS

### Q: "Walk me through a fraud detection model from scratch"

**ANSWER TEMPLATE — Use this exact structure:**

> "Let me frame the problem first, then the solution."

**Problem framing:**
- Current state: [X]% fraud capture rate, want [Y]%
- Data we have: structured claims data + unstructured notes
- Business constraint: real-time SLA of 200ms + overnight batch for full coverage

**Simple first:**
> "I'd start by building a baseline using rule-based heuristics — red flags like 'claim filed within 7 days of policy start', 'claimant's third claim in 90 days'. Rules get you to maybe 40% capture rate with high precision."

**Why ML next:**
> "Rules miss complex patterns and novel fraud schemes. ML learns which combinations of signals are predictive. We went from 40% to 72% capture rate in the top 25% risk tier by adding XGBoost with 80+ features."

**Full flow:**
```
1. Data collection: structured features + claim notes
2. Feature engineering: behavioral, network, temporal, NLP-derived
3. Baseline: logistic regression + rule layer
4. Advanced: XGBoost (batch) + LightGBM (real-time)
5. RAG layer: BERT-extracted flags from claim notes
6. Evaluation: PR-AUC, Capture@25, Precision@70%Recall
7. Deployment: FastAPI/K8s (real-time) + Airflow/Databricks (batch)
8. Monitoring: PSI on features, SIU feedback loop, drift alerts
```

---

### Q: "How do you handle imbalanced data in fraud?"

**Full answer:**

> "For 2% fraud rate, I use three complementary approaches:"

**1. Algorithm-level correction:**
- XGBoost: `scale_pos_weight = 98/2 = 49` (tells model each fraud case counts as 49 non-fraud)
- LightGBM: `class_weight='balanced'` or `is_unbalance=True`
- This is the FIRST thing I try — no data modification needed

**2. Metric selection:**
- Use PR-AUC instead of ROC-AUC (ROC-AUC is misleading at 2% fraud)
- Use Capture@25 (what % of fraud is in top 25% risk tier) — this is the business metric
- Set decision threshold via PR curve, NOT 0.5 default

**3. Sampling (if needed):**
- SMOTE only if dataset < 100K rows and imbalance is > 20:1
- Avoid SMOTE for large datasets (creates noise, slower training)
- Never do random undersampling without careful validation

**Interview-ready sequence:**
> "I always try scale_pos_weight first. If PR-AUC is still poor, I investigate whether it's a feature engineering problem (need better signals) before resorting to SMOTE. Most of the time, good features + scale_pos_weight + threshold tuning is sufficient."

---

### Q: "Transaction-level vs cart-level fraud prediction — which do you choose?"

**Context:** E-commerce or insurance with multiple line items per claim.

**Answer:**
> "Both have their place, and the choice depends on latency and fraud pattern complexity."

**Transaction/item-level:**
- Individual scoring at time of submission (immediate decision)
- Latency: instant (<100ms)
- Catches: single fraudulent line items, velocity-based fraud
- Best for: real-time blocking, step-up authentication triggers
- Feature example: "this item category has 3x normal claim rate from this ZIP"

**Cart/claim-level:**
- Aggregate all line items before scoring
- Can't give immediate decision (must wait for full claim)
- Catches: coordinated fraud across items (unlikely combinations), pattern across the full claim
- Best for: batch review, comprehensive risk score

**Hybrid approach (recommended):**
> "Real-time: immediate transaction-level score to flag urgent cases. Overnight batch: full claim-level re-score with all context. This gives fast response AND comprehensive assessment."

**Testing approach:**
> "I'd A/B test both: route 50% of claims to transaction-level scorer and 50% to cart-level scorer. Measure fraud detection rate, false positive rate, and SIU investigator time per flagged claim. Results determine final architecture."

---

### Q: "How do you prioritize early fraud detection in the model?"

**The interviewer's answer (from your transcript):**
The question was about **cost-sensitive learning** — penalizing first-transaction fraud more than later transactions.

**Full answer:**
> "There are two model-level levers for prioritizing early detection:"

**1. Cost-sensitive learning:**
```python
# Give higher weight to fraud detected at transaction 1 vs transaction 5
sample_weights = np.where(
    df['is_first_transaction_fraud'],
    2.0,   # 2x weight for catching fraud early
    1.0    # Standard weight
)
model.fit(X_train, y_train, sample_weight=sample_weights)
```
> "By assigning 2x sample weight to first-transaction fraud, the model learns to be more sensitive to early fraud signals."

**2. Calibrated risk thresholds by transaction order:**
```python
thresholds = {
    1: 0.3,   # Lower threshold for transaction 1 (be more aggressive)
    2: 0.4,
    3: 0.5,   # Default threshold
    'n+': 0.6  # Higher threshold once pattern established
}
```
> "First transaction gets a lower decision threshold — we flag at 30% probability instead of 50%. We're willing to accept more false positives early because missing early fraud is costlier."

**3. Feature engineering for early signals:**
- Transaction 1 features: device fingerprint freshness, IP reputation, velocity in last 24h
- These early-stage signals are given high feature importance via SHAP analysis

---

### Q: "How do you handle the cold start problem for new cards/accounts?"

**Answer:**
> "Cold start in fraud means: new account has no transaction history → can't compute behavioral features → model has low confidence."

**Solutions in priority order:**

**1. Proxy features from account metadata:**
- Account age (even if 0 days, it's a feature)
- Risk score from application data (credit bureau)
- Device/IP used at registration — compare to known fraud device pool
- Geographic risk of account's home address

**2. External data to fill gap:**
- Credit bureau: payment history, derogatory marks
- Device fingerprinting: is this device associated with other accounts?
- IP geolocation risk: high-risk ISP or Tor exit node?

**3. Portfolio-level priors:**
- "New accounts in our portfolio with these characteristics have X% fraud rate"
- Bayesian approach: start with portfolio prior, update as transactions accumulate

**4. Progressive learning:**
- First transaction: use proxy + external features → conservative threshold
- After 5 transactions: behavioral features available → normal threshold
- After 30 transactions: full behavioral profile → relax to aggressive threshold

**Interview answer:**
> "Cold start is solved in layers: at account creation, I use external signals (credit risk, device fingerprint, IP reputation). As transactions accumulate, behavioral features kick in — I track this via account_age_days and transaction_count features. The model naturally learns that low transaction count should increase caution. For the first 5 transactions, I lower my fraud threshold by 20% as a conservative safety net."

---

## SECTION 2: FRAUD RINGS AND NETWORK FRAUD

### Q: "How would you identify fraud rings?"

**The SIMPLE answer first (what interviewers actually want):**

> "The simplest approach that scales is attribute blocking with SQL."

**Step 1: SQL blocking — O(N) complexity**
```sql
-- Find groups sharing suspicious attributes (fraud rings)
SELECT 
    phone_number,
    COUNT(DISTINCT claim_id) as claims_in_group,
    COUNT(DISTINCT claimant_id) as claimants_in_group,
    SUM(claim_amount) as total_exposure,
    ARRAY_AGG(claimant_id) as claimant_ids
FROM claims
WHERE claim_date >= DATEADD(year, -2, CURRENT_DATE)
GROUP BY phone_number
HAVING COUNT(DISTINCT claimant_id) >= 3  -- 3+ different claimants sharing a phone
ORDER BY total_exposure DESC;
```

Then expand to multi-attribute blocking:
```sql
-- Multi-attribute blocking: shared phone OR shared address OR shared device
SELECT 
    shared_attribute_type,
    shared_attribute_value,
    COUNT(DISTINCT claimant_id) as ring_size,
    SUM(fraud_amount) as ring_exposure
FROM (
    SELECT 'phone' as shared_attribute_type, phone_number as shared_attribute_value, claimant_id, claim_amount FROM claims
    UNION ALL
    SELECT 'address', address_hash, claimant_id, claim_amount FROM claims
    UNION ALL
    SELECT 'device', device_fingerprint, claimant_id, claim_amount FROM claims
) combined
GROUP BY shared_attribute_type, shared_attribute_value
HAVING COUNT(DISTINCT claimant_id) >= 2
```

**Step 2: Only then add graph (for complex cases)**
> "After SQL blocking identifies candidate rings, I build a graph only on those flagged entities (much smaller N — maybe 5K vs 1M). Community detection (Louvain algorithm) finds sub-clusters within the ring. This is O(N log N) on the small subset."

**Why NOT start with graph:**
> "Graph algorithms on 100K+ nodes with full pairwise comparison is O(N²) — at 100K nodes, that's 5 billion pairs. SQL blocking reduces the problem to manageable size first."

**Similarity search complexity (what the interviewer was testing):**
| Method | Complexity | Feasible at 100K? |
|--------|-----------|-------------------|
| Brute-force all pairs | O(N²) | No (5B comparisons) |
| SQL GROUP BY blocking | O(N) with index | Yes |
| ANN (FAISS/HNSW) | O(N log N) build | Yes |
| Graph BFS on full graph | O(V + E) but E can be O(V²) | Depends on edge density |

**The answer the interviewer wanted to hear:**
> "Start with O(N) SQL blocking. Don't start with graph — it's a different animal. Only add graph when you have specific evidence that fraud rings use non-obvious obfuscated connections that SQL can't catch."

---

### Q: "How do you detect fraud rings using attributes only (no graph)?"

**Answer:**
```
1. Define high-signal shared attributes: phone, address, email domain, device ID, IP
2. SQL: GROUP BY each attribute → find groups of size >= 2
3. Generate ring candidates: claims sharing 2+ attributes with same group
4. Compute ring-level features:
   - ring_size: number of unique claimants in ring
   - ring_amount: total claim amount
   - ring_velocity: claims per month
   - ring_fraud_history: prior confirmed frauds in ring
5. Score rings using XGBoost ring classifier
6. Escalate top-scoring rings to SIU for investigation
```

**What NOT to do:**
> "I would NOT compute pairwise Jaccard similarity for all 100K claims — that's O(N²) and infeasible. The SQL GROUP BY is O(N) with proper indexing and finds 90% of rings in seconds."

---

## SECTION 3: RAG AND LLM IN FRAUD (YOUR RESUME)

### Q: "How does your RAG system work for fraud detection?"

**Full technical answer:**

**What RAG does in your system:**
> "RAG (Retrieval-Augmented Generation) is used for extracting fraud indicator flags from unstructured claim notes — not for generating narrative text. The notes contain adjuster commentary, email threads, and medical records. We can't train a supervised classifier on these directly because the fraud patterns evolve quickly."

**The pipeline:**
```
1. Document ingestion: adjuster notes → PII removal → chunking (500 tokens, 100 overlap)
2. Historical fraud pattern library:
   - 500 confirmed fraud cases with expert-annotated patterns
   - These are stored as vector embeddings in FAISS
3. For new claim note:
   - Embed the claim → search FAISS for top-5 most similar historical cases
   - Construct prompt: fraud taxonomy + retrieved examples + claim text
   - GPT-4o extracts 30-40 binary flags in structured JSON
4. NLP flags become features in XGBoost fraud model
5. LLM-as-judge: GPT-4.5 evaluates faithfulness of extracted flags
6. Ground truth evaluation: weekly PR-AUC on 200-case human-annotated dataset
```

**Key evaluations:**
- Recall@K: of top-K retrieved chunks, how many are relevant to the fraud pattern?
- Faithfulness: does the extracted flag accurately reflect the claim text?
- PR-AUC on ground truth: model precision/recall vs human annotations
- Cadence: weekly evaluation on fixed ground truth set

**Interview answer for "how do you evaluate RAG?":**
> "Three levels of evaluation: retrieval quality (Recall@5 — are the retrieved historical cases relevant?), generation faithfulness (LLM-as-judge: GPT-4.5 scores whether the extracted flags are supported by the text), and downstream business impact (do NLP flags improve the XGBoost model's PR-AUC?). We maintain a ground truth of 200 human-annotated claims and evaluate weekly."

---

### Q: "What metrics do you use for LLM evaluation?"

**Answer:**

| Metric | What It Measures | Formula/Method |
|--------|-----------------|----------------|
| Faithfulness | Are generated flags supported by source text? | LLM-as-judge (GPT-4.5 scores 1-5) |
| Answer Relevancy | Are extracted flags relevant to fraud taxonomy? | Cosine similarity: answer embedding vs question |
| Context Recall | Did retrieval capture all needed information? | Compare retrieved context to reference answer |
| Recall@K | Top-K retrieved chunks: how many are relevant? | # relevant in top K / total relevant |
| BLEU/ROUGE | Text similarity vs reference (for text tasks) | N-gram overlap |
| Business PR-AUC | Do flags improve downstream classifier? | AUC-PR on confirmed fraud labels |

**Interview one-liner:**
> "For RAG evaluation, I use Ragas framework: Faithfulness (are outputs grounded in source?), Answer Relevancy (are outputs relevant to question?), and Context Recall (did retrieval capture needed info?). I also maintain a ground truth of human-annotated cases for weekly PR-AUC evaluation — because ultimately, the LLM output's value is whether it improves the downstream fraud model."

---

## SECTION 4: ADVANCED FRAUD CONCEPTS

### Q: "Precision vs Recall — which for real-time vs batch?"

**The complete answer with business reasoning:**

**Real-time model → PRECISION is primary:**
- Every flag = manual underwriter review (30-60 min of investigator time)
- Low precision = many false alarms = underwriter burnout + backlog
- Customer impact: flagged genuine claims cause payment delays = customer complaints
- Regulatory: wrongly held claims may violate claims-handling regulations
- Rule of thumb: aim for Precision > 70% at whatever Recall we can achieve

**Batch model → RECALL is primary:**
- Overnight run: investigators can handle larger queue (more resources available)
- Missing fraud (FN) = financial loss to company (no recovery possible after payment)
- Cost asymmetry: a missed $500K fraud >> cost of investigating 10 extra claims
- Rule of thumb: aim for Recall > 80% in batch; accept lower precision (50-60%)

**How to set thresholds:**
```python
# PR curve shows precision at every recall level
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

# Real-time: find threshold where precision = 70%
rt_threshold = thresholds[np.argmax(precision >= 0.70)]

# Batch: find threshold where recall = 80%
batch_threshold = thresholds[np.argmax(recall >= 0.80)]

print(f"Real-time threshold: {rt_threshold:.3f}, Recall: {recall[precision >= 0.70][0]:.3f}")
print(f"Batch threshold: {batch_threshold:.3f}, Precision: {precision[recall >= 0.80][0]:.3f}")
```

---

### Q: "How do you detect concept drift in fraud models?"

**Answer:**

**What is concept drift in fraud?**
> "Fraud patterns evolve. Fraudsters adapt to your model. What looked like fraud 2 years ago may not match today's schemes. Concept drift = the relationship between features and fraud outcome changes over time."

**Detection methods:**
1. **Performance monitoring (lagged ground truth):**
   - SIU confirms flags with 4-6 week lag
   - Weekly: compare predicted fraud rate vs confirmed fraud rate
   - If precision drops > 5% from baseline → suspect concept drift

2. **CUSUM (Cumulative Sum control chart):**
   - Accumulates sum of deviations from expected performance
   - Signals when cumulative drift exceeds threshold
   - Good for detecting gradual drift (not sudden shifts)

3. **Kolmogorov-Smirnov on predicted scores:**
   - Compare distribution of model outputs this month vs 3 months ago
   - Significant KS shift → either data drift or concept drift

4. **Business signals:**
   - SIU flags new fraud pattern → explicit concept drift signal
   - Regulatory change → distribution shift in valid claims
   - Economic event (pandemic, recession) → behavior change across all claimants

**Response:**
```
Drift detected
    → Investigate: data pipeline issue? or genuine pattern change?
    → If pattern change: collect new labeled examples of new fraud scheme
    → Add new features capturing new patterns
    → Retrain on combined dataset (old + new)
    → A/B test challenger vs champion
    → Deploy and monitor
```

---

### Q: "What features would you use for insurance fraud detection?"

**Complete feature list by category:**

**Behavioral Features:**
- Claims frequency: count_claims_30d, count_claims_90d, count_claims_365d
- Claim timing: days_since_last_claim, time_since_policy_start, day_of_week, hour_of_day
- Amount patterns: claim_amount, avg_amount_90d, ratio_amount_to_avg, amount_vs_policy_limit

**Claimant History Features:**
- prior_fraud_flags: number of historical SIU referrals
- prior_fraud_confirmed: 1 if previous fraud confirmed
- claim_type_diversity: entropy of claim types submitted (high diversity = suspicious)
- provider_switching: number of different providers used (sudden switch = suspicious)

**Network Features (shared attributes):**
- shared_phone_count: number of claimants sharing same phone number
- shared_address_count: number of claimants at same address
- provider_fraud_rate: historical fraud rate for the specific service provider
- is_first_appearance: is this claimant/provider new to our portfolio?

**Geographic Features:**
- zip_fraud_rate: historical fraud rate in claim's ZIP code
- distance_from_home: distance of incident from claimant's home address
- geo_clustering: does claim fit claimant's usual geographic pattern?

**Policy Features:**
- policy_tenure_days: how long policy has been active (new policies higher risk)
- coverage_type_risk: some coverage types have higher fraud rates
- policy_limit_utilization: claim amount as fraction of policy limit
- policy_recent_change: any coverage changes in last 30 days?

**NLP-Derived Features (from your RAG system):**
- inconsistent_timeline: claim timeline contradicts medical records
- exaggerated_damages: description inconsistent with repair estimates
- prior_claim_reference: note references a previous claim (suspicious pattern)
- staged_accident_indicators: keywords suggesting coordinated accident
- Multiple other binary flags (30-40 total)

---

## SECTION 5: INTERVIEW SCENARIOS — COMPLETE WORKED ANSWERS

### Scenario: "You're building fraud detection from scratch. Walk me through."

**Template response (4 minutes):**

> "Let me approach this in order: problem definition, data, features, model, evaluation, deployment.

**Problem:** We want to flag insurance claims likely to be fraudulent before payment, prioritizing precision for real-time and recall for batch.

**Simple baseline first:** Rule-based heuristics — new policy (< 30 days) + high claim amount (> $50K) + prior fraud flag. This gets us to maybe 30% capture in top tier with high precision. Good starting point.

**Why ML:** Rules miss interaction effects and novel schemes. When we add ML: baseline logistic regression gets us to 0.71 PR-AUC. XGBoost with behavioral features gets to 0.83 PR-AUC — a 12-point lift that justifies added complexity.

**Features I'd build:** Behavioral (claim frequency last 30/90/365 days), network (shared attributes with known fraudsters), geographic (ZIP fraud rate, distance from home), and NLP-derived flags from claim notes (RAG pipeline).

**Evaluation:** Primary metric is PR-AUC (2% fraud rate means ROC-AUC is misleading). Also track Capture@25 (% fraud in top quartile of risk scores) and Precision@70%Recall as operational SLA.

**Deployment:** Two paths — real-time FastAPI/LightGBM for immediate scoring, overnight Databricks/XGBoost for batch. MLflow registry for versioning. PSI monitoring on input features, SIU feedback for concept drift detection.

That's the end-to-end system. Which area would you like me to go deeper on?"

---

### Scenario: "Our fraud model has 95% accuracy but only catches 30% of fraud. What's wrong?"

**Answer:**
> "This is a classic class imbalance problem. With 2% fraud rate, a model predicting 'not fraud' for everything gets 98% accuracy. A model catching 30% of fraud at 95% accuracy is still better than random — but you're right that 30% recall means missing 70% of fraud."

**Diagnosis:**
1. Model is using accuracy as metric → switch to PR-AUC and Recall
2. Decision threshold is 0.5 (default) → too conservative for 2% fraud rate
3. May not have scale_pos_weight set → model hasn't been told fraud is important

**Fix:**
1. Set `scale_pos_weight = 49` (for 2% fraud: 98/2)
2. Retrain and evaluate using PR-AUC
3. Tune decision threshold on PR curve → optimize for business target (e.g., 80% recall)
4. If still poor: investigate feature engineering — need better fraud signals

**Expected outcome:**
> "After these fixes: PR-AUC from 0.45 to 0.72, Recall from 30% to 75% at acceptable Precision (65%). The 95% accuracy figure becomes irrelevant — we replace it with Capture@25 as the primary business metric."

---

## SECTION 6: QUICK REFERENCE CARD

| Topic | Key Answer |
|-------|-----------|
| Fraud rate | Typically 1-5% of claims; use PR-AUC not ROC-AUC |
| scale_pos_weight | = n_negatives / n_positives (e.g., 49 for 2% fraud) |
| Real-time metric | Precision (false positives = costly manual reviews) |
| Batch metric | Recall (false negatives = missed fraud = financial loss) |
| Fraud ring detection | SQL GROUP BY first (O(N)), graph only for complex cases (O(N log N)) |
| O(N²) | Brute-force pairwise similarity — does NOT scale at 100K+ nodes |
| Cold start | Proxy features (credit, device, IP) + portfolio priors + lower threshold |
| Concept drift | SIU feedback monitoring + CUSUM + KS test on predictions |
| PSI > 0.2 | Significant input drift — investigate and retrain |
| RAG evaluation | Recall@K + Faithfulness + PR-AUC on ground truth |
| Capture@25 | % of total fraud in top 25% risk tier — primary business metric |
| Transaction vs cart | Transaction for real-time blocking; cart for comprehensive assessment; hybrid best |

---

## SECTION 7: LIFT, GAIN, AND DECILE ANALYSIS

### What Are Lift and Gain Charts

**Gain chart:** Shows the percentage of total fraud captured by selecting the top X% of ranked claims.

**Lift chart:** Shows how much better the model is than random selection at each decile.

| Decile | Cumulative % Population | Cumulative % Fraud Captured (Gain) | Lift vs Random |
|---|---|---|---|
| Top 10% | 10% | 45% | 4.5x |
| Top 20% | 20% | 68% | 3.4x |
| Top 30% | 30% | 81% | 2.7x |
| Top 50% | 50% | 92% | 1.84x |
| 100% | 100% | 100% | 1.0x |

### Why They Matter in Fraud

Business stakeholders understand deciles more easily than PR-AUC.

**Interview one-liner:**
> "We don't need to review every claim. We score all claims, sort by fraud probability, and investigate the top two deciles. In our last model, the top decile captured 4.5x more fraud than random selection, which means investigators spend time on the highest-value cases."

### Decile Stability

- Track decile boundaries over time
- If top-decile scores drop, model may be degrading
- Use for threshold selection and capacity planning

---

## SECTION 8: EXPLAINING FRAUD METRICS TO NON-TECHNICAL STAKEHOLDERS

### Precision and Recall in Plain Language

**Precision:**
> "Out of all claims our model flags as suspicious, precision tells us how many are actually fraud. If precision is 70%, then 7 out of 10 flagged claims are real fraud. The other 3 are false alarms."

**Recall:**
> "Out of all fraud that actually exists in the data, recall tells us how many our model catches. If recall is 80%, we catch 8 out of 10 fraud cases but miss 2."

**F1 Score:**
> "F1 is the balance between precision and recall. It is useful when both false alarms and missed fraud are costly. We use it as a single number to compare models, but we still look at precision and recall separately for business decisions."

### Cost-Based Framing

Translate metrics into investigator time and money:

| Metric | Business Interpretation |
|---|---|
| Precision 70% | 30% of investigator time is wasted on false alarms |
| Recall 80% | We miss 20% of fraud, which equals $X million in exposure |
| False positive rate 5% | 5% of genuine customers experience delayed claims |
| Capture@25 65% | Two-thirds of all fraud is in the top quarter of our risk queue |

### Stakeholder Communication Template

> "The model ranks every claim by fraud probability. We focus investigators on the top 25% of ranked claims. In that group, we catch 65% of all fraud. We still have false alarms, but they are concentrated in a small queue that the SIU team can handle."

---

## SECTION 9: IMBALANCED DATA STRATEGIES BEYOND SMOTE

### Hierarchy of Techniques

**1. Algorithm-level weighting (always try first):**
- XGBoost scale_pos_weight
- LightGBM is_unbalance / class_weight
- sklearn class_weight='balanced'

**2. Evaluation metric correction:**
- PR-AUC instead of ROC-AUC
- F-beta with beta tuned to business cost
- Use business metrics like Capture@25 and expected savings

**3. Threshold tuning:**
- Pick threshold from PR curve, not 0.5
- Optimize for expected monetary value

**4. Sampling techniques (use only when needed):**
- SMOTE: oversample minority class with synthetic neighbors
- ADASYN: adaptive synthetic sampling focusing on hard examples
- Random undersampling: reduce majority class; risky with information loss
- Tomek links / Edited Nearest Neighbors: clean majority boundary

### When NOT to Use Sampling

- Dataset is large (> 100K rows)
- Fraud rate is above 5%
- You have strong discriminative features

**Interview one-liner:**
> "I try scale_pos_weight and threshold tuning first. I only use SMOTE on small datasets where the model genuinely lacks positive examples. On large fraud datasets, SMOTE adds noise and slows training without improving PR-AUC."

### Cost-Sensitive Learning

Instead of balancing classes, assign costs to errors:
- Cost of false negative = $amount of fraud
- Cost of false positive = investigator hourly cost
- Build custom loss or sample weights reflecting these costs

---

## SECTION 10: FRAUD-SPECIFIC EVALUATION METRICS

### Capture at Top K

- Capture@1%: fraud captured in top 1% of scores
- Capture@5%, Capture@10%, Capture@25%
- The business chooses K based on investigator capacity

### Expected Fraud Value

Expected fraud savings = (fraud captured * average fraud amount) - (false positives * investigation cost)

This is the most business-relevant metric.

### Precision at Fixed Recall

- Precision@80%Recall: how many flagged claims are fraud when we catch 80% of fraud
- Useful for operational SLA

### False Negative Rate by Fraud Amount

- High-value fraud should have lower false negative rate
- Weight recall by claim amount in evaluation

### Score Calibration

- Fraud scores should reflect true probability
- Use reliability diagrams, Brier score, Platt scaling, isotonic regression
- Well-calibrated scores enable risk-based pricing and reserves

---

## SECTION 11: ADVANCED SQL BLOCKING FOR FRAUD RINGS

### Why SQL Blocking First

SQL blocking is deterministic, fast, and explainable. It should be the first line of defense before graph analytics.

### Blocking Rules Examples

1. Same phone number used by 3+ claimants in 90 days
2. Same address with 5+ claims in 30 days
3. Same device fingerprint opening multiple accounts
4. Same bank account receiving payments from multiple claims
5. Same IP submitting claims for different policyholders

### Windowing and Velocity

Add time windows to avoid stale rings:
- Count only claims in the last 90 days
- Weight recent shared attributes more heavily
- Expire old connections

### Blocking Index Strategy

- Index phone_number, address_hash, device_fingerprint, bank_account
- Use composite indexes when filtering by date
- Partition by claim_date for historical queries

### Interview One-Liner

> "I detect fraud rings with SQL GROUP BY on shared attributes like phone, address, and device. I filter groups that span multiple claimants within a 90-day window. Only after SQL blocking identifies candidate rings do I use graph algorithms for deeper community detection."

---

## SECTION 12: CALIBRATION AND PROBABILITY QUALITY

### Why Calibration Matters in Fraud

A score of 0.80 should mean 80% of such claims are actually fraud. Poor calibration leads to:
- Bad reserve calculations
- Misleading risk tiers
- Broken A/B testing

### Calibration Checks

- Reliability diagram: plot predicted probability vs actual fraud rate
- Brier score: mean squared error of predicted probabilities
- Expected Calibration Error (ECE): weighted average of calibration gaps

### Calibration Methods

- Platt scaling: fit logistic regression on validation scores
- Isotonic regression: non-parametric calibration
- Beta calibration: for probabilities already bounded in [0,1]

### When Calibration Breaks

- Model retrained on different time period
- Feature distribution drifts
- Model moved from batch to real-time with different feature sources

**Interview one-liner:**> "I check calibration with reliability diagrams and Brier score. If scores are not calibrated, I apply Platt scaling on a held-out validation set. Calibrated scores are critical because investigators trust risk tiers and finance uses them for reserves."

---

## SECTION 13: LEAD-ROLE FRAUD INTERVIEW SCENARIOS

### "How would you build a fraud team and roadmap?"

> "I would start with a 0-90-180 day roadmap. First 30 days: audit current rules, SIU feedback loop, and data quality. Days 30-90: build baseline logistic regression, establish PR-AUC tracking, and deploy a lightweight real-time scorer. Days 90-180: add XGBoost batch, RAG/NLP flags, graph features, and A/B testing. I would staff a data scientist, an ML engineer, and a fraud analyst, with clear ownership of model, features, and operations."

### "How do you balance fraud detection with customer experience?"

> "Every false positive delays a genuine customer's claim. I measure customer complaints and claim cycle time as guardrail metrics. I tune the real-time model for high precision so only the most suspicious claims are held. For borderline cases, I use step-up verification rather than automatic denial. The batch model can afford more recall because it only prioritizes investigation, not immediate blocking."

### "Tell me about a time you improved fraud detection."

> "We had a rule-based system that reviewed claims randomly. I built a risk score that ranked claims by fraud probability. Investigators focused on the top decile. We improved Capture@25 from 35% to 65%, meaning we caught nearly twice as much fraud with the same investigator hours. We also reduced false positive complaints by 25% by tuning precision for the real-time hold queue."

### "What would you do if the model starts flagging too many genuine claims?"

> "I would first check the false positive rate and recent feature distributions. If a feature drifted, such as a new claim type with different amount patterns, I would retrain or exclude the affected population. If the threshold is too aggressive, I would raise it temporarily. I would also review the fallback rules and ensure the threshold optimization uses current business costs."

---

## SECTION 14: GRAPHRAG FOR FRAUD INVESTIGATION

### What Is GraphRAG

GraphRAG combines knowledge graphs with retrieval-augmented generation. Instead of retrieving isolated text chunks, it retrieves connected entities and relationships so the LLM can reason over multi-hop fraud evidence.

### Why Traditional RAG Is Not Enough for Fraud

Fraud evidence often lives in relationships:
- A claimant shares a phone with a known fraudster
- A provider works on multiple suspicious claims
- A vehicle appears in unrelated claims with different owners

Plain text RAG misses these relational patterns because chunks are processed in isolation.

### GraphRAG Architecture for Fraud

1. **Entity extraction:** parse claims, policies, and notes to extract entities (claimant, provider, address, vehicle, phone)
2. **Graph construction:** store entities and relationships in a property graph
3. **Community detection:** group related entities into communities
4. **Summarization:** generate community summaries that capture multi-hop patterns
5. **Retrieval:** given a new claim, retrieve relevant entities, neighbors, and community summaries
6. **Generation:** LLM reasons over the retrieved graph context to produce structured fraud indicators

### When to Use GraphRAG

- Fraud patterns are relational rather than purely textual
- Investigators ask multi-hop questions like "who else is connected to this provider?"
- You need explainable reasoning grounded in the graph

### Interview One-Liner

> "For fraud, I would use GraphRAG when the evidence is relational. I extract entities from claims and notes, build a knowledge graph, retrieve connected entities and community summaries, and use the LLM to reason over the graph context. This catches ring patterns that plain text RAG misses."

---

## SECTION 15: LABEL MATURATION AND SELECTION BIAS

### Label Maturation Problem

Fraud labels are not available immediately. A claim flagged today may not be confirmed as fraud for 4 to 6 weeks.

**Implications:**
- Recent claims have no confirmed labels
- Training data is always slightly stale
- A/B tests must wait for label maturation
- Performance metrics on recent data are unreliable

### Handling Label Maturation

- Define a label maturity window (e.g., 45 days) and only use labels after that window
- For recent claims, use proxy labels: SIU referral, amount paid, claim closure without recovery
- Clearly separate mature metrics from immature metrics in dashboards

### Selection Bias From Your Own Model

If your model blocks or holds suspicious claims, those claims get investigated more often. This creates a feedback loop:
- The model flags certain patterns
- Investigators review flagged claims and confirm some as fraud
- Training data over-represents flagged patterns
- The model becomes even more confident in those patterns

### Mitigating Selection Bias

- Include a random sample of unflagged claims in SIU review
- Use propensity scoring to weight samples
- Track model performance on a holdout set that bypasses the production filter
- Monitor for model-induced bias in label distribution

### Interview One-Liner

> "Fraud labels mature with a 4 to 6 week lag, so I only use confirmed labels after a maturity window. I also watch for selection bias: if we only label claims the model flags, we reinforce our own predictions. I mitigate this by randomly sampling unflagged claims for review."

---

## SECTION 16: ANOMALY DETECTION FOR COLD-START AND NOVEL FRAUD

### When Supervised Models Fail

- New fraud schemes have no labeled examples
- Cold-start entities have no history
- Fraudsters intentionally behave differently from historical patterns

### Anomaly Detection Approaches

| Approach | Best For | Example |
|---|---|---|
| Statistical outlier detection | Simple univariate anomalies | Claim amount 10x above normal |
| Isolation Forest | Multivariate outliers | Unusual combination of features |
| Autoencoder | High-dimensional data | Reconstruction error for new claim patterns |
| One-class SVM | Limited normal data | Profile of genuine claims |
| Clustering + distance | Group deviation | Claim far from normal claimant clusters |

### Combining Anomaly and Supervised Models

- Anomaly model scores new/cold-start cases
- Supervised model scores cases with sufficient history
- A gating model decides which score to trust based on entity history

### Interview One-Liner

> "For cold-start claims and novel fraud, I use anomaly detection as a parallel signal. An isolation forest or autoencoder flags patterns that deviate from normal behavior. I combine this with the supervised fraud score so new entities are not invisible to the system."

---

## SECTION 17: SEQUENCE AND BEHAVIORAL MODELING

### Why Sequences Matter

A single claim may look normal, but a sequence of actions can reveal fraud:
- Multiple small claims leading up to a large claim
- Rapid policy changes before a claim
- Claimant switches providers frequently

### Sequence Features

- Time between claims
- Order of claim types
- Changes in claim amounts over time
- Provider switching patterns
- Policy change timing

### Models for Sequences

- **RNN/LSTM/GRU:** capture temporal dependencies
- **Transformer:** capture long-range patterns
- **Markov models:** model transition probabilities between states

### When to Use Sequence Models

Use sequence models when the order and timing of events are predictive. For most fraud problems, well-engineered sequence features fed into XGBoost or LightGBM are sufficient and more interpretable.

### Interview One-Liner

> "I model sequences by extracting time-based and order-based features such as time-between-claims and provider-switching patterns. For most fraud use cases, gradient boosting with sequence features is enough. I would use an LSTM or Transformer only if long-range order is critical and labeled data is ample."

---

## SECTION 18: FAIRNESS AND DISPARATE IMPACT

### Why Fairness Matters in Fraud

A model that systematically flags claims from certain demographics, ZIP codes, or policy types creates regulatory and reputational risk.

### Fairness Metrics

- **Demographic parity:** flag rates are similar across groups
- **Equalized odds:** false positive and false negative rates are similar across groups
- **Calibration within groups:** predicted probabilities match actual rates within each group

### Detecting Disparate Impact

- Slice model performance by protected attributes where legally permitted
- Compare flag rates, precision, and recall across groups
- Check if certain features act as proxies for protected attributes

### Mitigation Strategies

- Remove or limit proxy features
- Apply fairness constraints during training
- Calibrate separately within groups
- Use human review for groups where the model is less accurate

### Interview One-Liner

> "I evaluate fraud models for disparate impact by slicing precision, recall, and flag rates across relevant groups. If I find a proxy feature creating bias, I remove or constrain it. I also calibrate within groups and route borderline cases to human review."

---

## SECTION 19: REASON CODES AND EXPLAINABILITY

### Why Reason Codes Matter

Investigators, regulators, and customers need to know why a claim was flagged.

### Generating Reason Codes

- SHAP top factors for each prediction
- Rule firing: which business rules triggered
- Graph-derived reasons: connected to known fraud ring
- NLP-derived reasons: specific fraud indicators from claim notes

### Format for Investigators

```
Flag reason: High claim amount relative to policy history
Supporting factor: Claim amount is 4x the claimant's 90-day average
Confidence: 0.91
Additional signals: Provider has 15% historical fraud rate; claimant filed 3 claims in 30 days
```

### Regulatory Explainability

- Provide top 3 to 5 factors per decision
- Ensure factors are business-meaningful, not just model internals
- Avoid using protected attributes as reason codes

### Interview One-Liner

> "Every fraud flag is accompanied by reason codes from SHAP, business rules, graph signals, and NLP indicators. I limit these to the top 3 to 5 business-meaningful factors so investigators and regulators can understand the decision."

---

## SECTION 20: ADVERSARIAL DRIFT AND FRAUDSTER ADAPTATION

### What Is Adversarial Drift

Fraudsters study your model's behavior and adjust to evade detection. This is concept drift driven by intelligent adversaries.

### Signs of Adversarial Drift

- Sudden drop in model performance on specific segments
- New patterns that exploit known model weaknesses
- Increase in claims that narrowly miss thresholds
- Feedback from SIU about novel schemes

### Defensive Strategies

- Monitor performance by segment and flag sudden drops
- Use ensemble models so no single weak point is exploitable
- Keep some rules and anomaly detection as non-learned signals
- Regularly rotate features and retrain
- Add randomization to thresholds and review queues so fraudsters cannot game the system

### Interview One-Liner

> "I assume fraudsters will adapt. I monitor segment-level performance, maintain ensembles and non-learned signals like rules and anomaly detection, and avoid deterministic thresholds that can be gamed. Regular retraining and feature rotation reduce exploitability."

---

## SECTION 21: ADDITIONAL FRAUD OPS SCENARIOS

### "Your offline PR-AUC improved, but online fraud loss did not drop. Why?"

> "Possible reasons: label maturation means recent labels are incomplete, the offline test set does not match production distribution, selection bias from our own flags inflates offline metrics, or the new model's calibration changed so thresholds no longer produce the intended action. I would verify by waiting for mature labels, running an A/B test, and checking calibration and threshold impact."

### "How do you handle a new fraud scheme with no labeled examples?"

> "First, I work with SIU to characterize the pattern. Then I engineer features that capture it. I use anomaly detection to surface similar cases for rapid labeling. Once I have enough confirmed examples, I retrain the supervised model and A/B test before full rollout."

### "How do you prevent the model from gaming the system?"

> "I avoid publishing exact thresholds or feature logic. I use ensemble models and maintain rule-based and anomaly-based signals that are harder to reverse-engineer. I monitor for edge-seeking behavior, where claims cluster just below thresholds, and I adjust thresholds or add features accordingly."

### "What is the right balance between automation and human review?"

> "High-confidence fraud can be auto-routed to investigation. Low-confidence cases go to human review. Anything involving large amounts, protected classes, or novel patterns should default to human review. The balance is set by precision, cost, and risk appetite."
