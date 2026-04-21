# 🌀 Edge Cases, Unusual Issues & Failure Modes
### DDS + DMM Round: The Questions That Trip Up 80% of Candidates

> These are the UNUSUAL questions you won't find in typical prep. 
> Master these and you'll stand out vs. all other Senior DS candidates.

---

## CATEGORY 1: RAG SYSTEM FAILURE MODES

**Issue 1: Retrieval retrieves the WRONG things confidently**

Scenario: A legitimate medical claim for a rare condition has unusual language → retrieves fraud cases as "similar" → LLM incorrectly flags as fraud.

Signs: High faithfulness score (LLM is faithful to retrieved docs) but low business precision.

Root cause: Embedding model conflates linguistic similarity with semantic fraud-relevance.

Solutions:
- Add a **re-ranking step** (cross-encoder: finer-grained relevance scoring than bi-encoder)
- Threshold on retrieval confidence: if max cosine < 0.65, don't let RAG influence the decision
- Add claim-type routing: use a classifier to route "rare condition" claims to a specialist model, not general RAG
- Human review for low-confidence retrievals

---

**Issue 2: Knowledge Base Poisoning**

Scenario: Fraudsters discover that if they submit claims that closely match legitimate patterns, the RAG system retrieves their "past good claims" as similar patterns, reducing their risk score.

This is an adversarial attack on the knowledge base itself.

Solutions:
- **Separate retrieval indexes**: Never use current pending claims as retrieval context. Only use confirmed ground-truth cases (investigator-verified fraud + legitimate).
- **Recency weighting**: Don't retrieve patterns from the same account — that's circular
- **Diversity enforcement**: Require retrieved patterns to come from diverse account IDs

---

**Issue 3: Embedding Model Staleness vs. LLM Upgrade Mismatch**

Scenario: You upgrade the LLM (GPT-3.5 → GPT-4) but the embedding model stays the same. The retrieval quality appears unchanged, but the LLM now over-relies on its parametric memory instead of retrieved context.

Signs: Faithfulness score drops (LLM outputs things not in retrieved context).

Solution:
- After any LLM upgrade: re-evaluate faithfulness score on benchmark set
- Consider re-tuning system prompt for new LLM's behavior
- Fine-tune embedding model jointly with LLM evaluation to ensure retrieval-generation alignment

---

## CATEGORY 2: GRADIENT BOOSTING FAILURE MODES

**Issue 1: XGBoost Learns from Suppressed Data (Feedback Loop)**

Scenario: XGBoost fraud model flags accounts → Flagged accounts are blocked → Blocked accounts don't appear in future training data → Model never learns from its errors.

Root cause: Training data is NOT a random sample — it's censored by the current model.

Solutions:
- **Counterfactual logging**: For 0.1% of transactions, override the block decision and log what actually happened (explore)
- **Inverse Propensity Scoring (IPS)**: During training, up-weight samples that had low probability of being unblocked (reweight for selection bias)
- **External labels**: Use chargeback data, investigator confirms — labels that don't depend on the model's previous decisions

---

**Issue 2: Feature Importance Instability Across CV Folds**

Scenario: In the readmission model, feature importance varies significantly fold-to-fold. Feature A is #1 in fold 1, #5 in fold 3.

Signs: SHAP values are inconsistent across subsets.

Causes:
- High feature correlation (two features carry same information, model picks one arbitrarily)
- Sparse features (only important in specific patient subgroups)

Solutions:
- **Permutation importance over CV**: Average permutation importance across folds for stability
- **Group correlated features**: Use clustering on feature correlation matrix, select one representative from each group
- **Report uncertainty**: "Feature A is consistently in top-5, but rank within top-5 varies (confidence interval: rank 1-4)"

---

**Issue 3: XGBoost Monotonicity Violation in Credit Scoring**

Scenario: In CLV/credit scoring, you expect "higher income → higher CLV" to be monotone. But XGBoost sometimes creates non-monotone relationships (income=high might score lower than income=medium for certain value ranges).

Business problem: Regulators and business stakeholders reject a credit model that says high-income applicants are riskier.

Solution:
```python
# XGBoost monotonicity constraints
# 1 = feature must be monotone increasing
# -1 = must be monotone decreasing
# 0 = no constraint

model = xgb.XGBClassifier(
    monotone_constraints={
        'income': 1,          # Higher income → lower risk (increase approval)
        'credit_age': 1,      # Older credit history → lower risk
        'num_derogatory': -1, # More derogatory marks → higher risk
    }
)
```

Cost: ~5-10% AUC reduction vs. unconstrained. Almost always worth it for regulatory compliance.

---

## CATEGORY 3: BERT/LLM UNUSUAL ISSUES

**Issue 1: BERT Negation Failure — "No Signs of Fraud"**

Scenario: Clinical note says "patient shows NO signs of cognitive impairment." BERT encodes this similarly to "patient shows signs of cognitive impairment" because it places high attention on "cognitive impairment."

Why: Standard BERT recognizes sentiment at document level but can miss fine-grained negation for specific entities — especially when negation spans multiple tokens.

Solution:
- Use **NegEx algorithm** as preprocessing: identify negation scope around key medical entities before passing to BERT
- Fine-tune BERT on negation-heavy examples: create contrastive pairs ("shows X" vs. "no signs of X")
- Add a negation detection head as an auxiliary task during BERT fine-tuning

---

**Issue 2: ClinicalBERT Domain Shift for Insurance Claims**

Scenario: ClinicalBERT was pre-trained on hospital notes (MIMIC-III dataset). Insurance claims have different language: adjuster jargon, legal terminology, "loss runs," "subrogation." Performance drops on insurance-specific claim documents.

Solutions (by compute budget):
- **Low budget**: Add domain-specific vocab to tokenizer; continue pre-training on claims corpus for 1 epoch with MLM
- **Medium budget**: Full fine-tuning on claims corpus with MLM objective (need ~100K+ claims documents)
- **Alternative**: Use domain-specific model (LegalBERT for legal claims, then layer insurance-specific fine-tuning)

---

**Issue 3: LLM Hallucination on Low-Context Claims**

Scenario: A new type of claim comes in (e.g., first cryptocurrency-related insurance claim ever processed) with zero similar cases in the knowledge base. RAG retrieves tangentially related cases. LLM fills the gap with hallucinated "evidence."

This is the most dangerous failure mode — confident wrong answers.

Detection:
- Monitor retrieval similarity scores: if max cosine < 0.60 → flag as "low confidence" retrieval
- LLM self-consistency check: run the same query 3x with temperature=0.7 — if outputs differ significantly, flag for human review

Prevention:
- **Uncertainty-aware output**: Force LLM to output a confidence score and explain retrieval basis
- **Hard threshold**: If retrieval similarity < 0.60, output "INSUFFICIENT CONTEXT — REQUIRES INVESTIGATOR REVIEW" instead of a risk score
- **Proactive KB expansion**: Monthly process — identify uncovered claim types, generate synthetic examples

---

## CATEGORY 4: DISTRIBUTED COMPUTING (PYSPARK) ISSUES

**Issue 1: Cartesian Explosion in Graph Features**

Scenario: Computing "users who share the same device fingerprint" requires a self-join on 2M users. Self-join of 2M × 2M = 4 trillion rows → cluster runs out of memory.

Solution:
- **Bloom filter pre-filter**: First check if fingerprints are definitely unique (fast) before doing expensive join
- **Bucketed join**: Bucket by hash(device_fingerprint) — only join within same bucket
- **Approximate distinct count**: Use HyperLogLog for "number of shared users" if exact count not needed
- **Partitioned join**: Process by device_fingerprint prefix blocks to control memory

---

**Issue 2: Data Skew in Fraud Datasets**

Scenario: 90% of fraud cases come from 5 merchant categories. Partitioning by category → 5 partitions have 90% of data → most executors idle.

Solutions:
- **Salting**: Add random salt to partition key to distribute: `partition_key = category + "_" + (random.randint(0, 9))`. Join key must be salted on both sides.
- **Custom partitioner**: Route by (category + user_id_bucket) for more even distribution
- **Broadcast join**: If merchant metadata is small (<200MB), broadcast to all executors to avoid shuffle join

---

**Issue 3: Floating Point Non-Determinism in Distributed Training**

Scenario: Same model trained twice on the same data gives different AUC (0.891 vs. 0.887). This fails reproducibility requirements.

Root cause: Distributed floating-point addition is not commutative — different worker execution orders → different accumulated gradients.

Solutions:
- Set `random_state=42` on all randomized operations
- Use `seed_everything()` (PyTorch Lightning) to set seeds for Python, NumPy, PyTorch, CUDA
- Accept tiny variance (<0.5%) as normal for distributed training; document acceptable variance range
- For critical production models: single-node training for reproducibility if feasible

---

## CATEGORY 5: AGENTIC AI EDGE CASES

**Issue 1: Tool Call Argument Hallucination**

Scenario: Agent calls `get_user_history(user_id='abc-123')` but the actual user_id is `ABC-123` (case-sensitive). Tool returns empty. Agent interprets "no history = new user." Makes wrong decision.

Solutions:
- **Schema validation**: Validate all tool arguments against expected format before execution
- **Tool response validation**: If response is unexpectedly empty, retry with fuzzy argument matching
- **Fallback**: If tool returns empty for first call, verify the argument format and retry once

---

**Issue 2: Agent Reasoning Gap (Intermediate Step is Wrong)**

Scenario: Agent correctly calls SQL to get total revenue ($1.2M). Then calculates fraud loss as "15%" of $1.2M. But 15% came from a prior (wrong) calculation, not from the database. Compound error.

Solution:
- **Scratchpad verification**: Agent must re-verify claimed numbers against tool outputs before using them in subsequent calculations
- **Self-consistency check**: Run final calculation two ways; if results differ > 5%, flag for human review
- **Citation requirement**: Every number in the final output must be traced to a specific tool call output

---

**Issue 3: Agent Scope Creep (Takes Unauthorized Actions)**

Scenario: User asks "identify and fix the data quality issues in the fraud table." Agent correctly identifies issues but then autonomously runs DELETE and UPDATE statements to "fix" them — unauthorized data modification.

Solution:
- **Read-only constraint**: All database tools are READ-ONLY by default. Separate "data modification" tools require explicit user approval at both the tool level and execution level
- **Intent confirmation**: Before any write operation, agent must output "I am about to [action]. Confirm? [Y/N]"
- **Audit trail**: Every tool call logged with timestamp, user session ID, and full parameters

---

## CATEGORY 6: MLOps UNUSUAL SCENARIOS

**Issue 1: Model Works Great Offline But Terrible in Production**

Classic training-serving skew checklist:
1. Feature computation: Is rolling window computed from current timestamp in serving vs. from label timestamp in training?
2. Missing value handling: Are null imputation strategies consistent?
3. Categorical encoding: Is the same encoder used? Are new categories handled (unknown labels)?
4. Feature ordering: Is the feature vector order guaranteed consistent?

Debugging approach:
- Log COMPLETE serving feature vectors for every prediction
- Periodically compare feature distributions: training distribution vs. serving distribution
- Alert if any feature PSI > 0.25 (significant distribution shift)

---

**Issue 2: Seasonal Concept Drift in Fraud**

Scenario: Chubb fraud model trained on Jan-Oct data performs well until festival season (Diwali, Christmas). Return fraud patterns shift dramatically → model precision drops 15%.

Why: Festival season creates legitimate purchase spikes that look like fraud from the model's perspective.

Solutions:
- **Time-aware features**: Add "day_to_festival" proximity features, time-of-year cyclical encoding
- **Seasonal retraining**: Monthly retrain window captures seasonal patterns
- **Ensemble with seasonally-tuned model**: During known seasonal periods, mix general model with season-specific model
- **Business rule override**: Temporarily raise fraud threshold during festival windows (business decision, not model)

---

*Master these and you are prepared for the 20% of questions that distinguish truly senior candidates.*
