# 🎯 Project Deep Dives & Behavioral Questions — Interview Prep
### Abhishek Bhardwaj | Solutions Architect ML/AI @ Huge

---

> [!IMPORTANT]
> This is where the interview gets **personal and specific**. Interviewers will drill every bullet on your resume. Have the full story — architecture, decisions, metrics, failures, and what you'd do differently — for every project. Your STAR answers should feel like natural conversation, not rehearsed scripts.

---

## SECTION 1: Insurance Fraud Detection (RAG + LLMs) — Chubb

---

### System Design: "Describe the complete architecture of your fraud detection system."

**Why they're asking**: This is an open-ended architecture question. They want to see systems thinking: how components fit together, what the data flow is, and what trade-offs you made.

**Model Answer — Full Architecture Narrative**:

The fraud detection system was designed to address a specific business need: Chubb's claims adjusters were spending 70% of their investigation time manually reviewing documents to identify red flags. We needed an AI system that could surface fraud signals from unstructured claims documents at claim intake, before the adjuster even opened the file.

**End-to-end architecture**:

```
INGESTION LAYER
Claims Documents (PDF, Word, TIFF scans)
    → S3 event trigger → Lambda → Document Processor
    → Text extraction (PDFMiner + Tesseract OCR for scans)
    → Table extraction (camelot for structured tables)
    → Metadata extraction: {claim_id, policy_num, doc_type, date, insured}
    → Canonical document JSON → S3 (raw store)

PROCESSING & INDEXING LAYER
    → Chunking service (ECS task):
      - Parent chunks: semantic sections (incident, medical, adjuster notes)
      - Child chunks: 256-token windows within sections
    → Embedding service: text-embedding-ada-002 (OpenAI API)
    → FAISS IVF index (2M vectors, nlist=500, nprobe=20)
    → Metadata store: PostgreSQL (claim_id, chunk_id, doc_type, dates, policy details)

RETRIEVAL & INFERENCE LAYER (real-time)
    New claim event → SageMaker Endpoint
    → Query construction: "Fraud indicators for [claim_type] claim with [key_facts]"
    → Hybrid retrieval: FAISS (dense) + Elasticsearch (sparse BM25) → RRF merge
    → Cross-encoder re-ranking: top-20 → top-5
    → GPT-4 inference with fraud-analysis system prompt
    → Structured output: {fraud_risk_score, fraud_indicators[], evidence_citations[], confidence}

OUTPUT & FEEDBACK LAYER
    → Fraud risk score stored in PostgreSQL claims DB
    → Adjuster dashboard: risk score + evidence panel (clickable citations)
    → Adjuster feedback (confirm/deny fraud indicators) → feedback table
    → Weekly feedback batch → retraining trigger (if Δ > threshold)
```

This architecture enabled adjusters to review fraud indicators in the first 5 minutes of opening a claim, rather than spending hours reading through documents. We measured a ~40% reduction in time-to-investigation-decision for high-risk claims.

---

### Q1.1: "How did you handle long insurance claim documents that exceed context windows?"

**Model Answer**:

Insurance claim files are notorious for being long — a single complex claim can have hundreds of pages: the original claim form, multiple medical records, police reports, repair estimates, adjuster field notes, and prior correspondence. The naive approach of dumping all of this into the context window is infeasible even with Claude's 200K context, because (a) it's expensive, (b) "lost in the middle" problem means the model ignores content in the middle of a very long context, and (c) for our multi-document corpus spanning thousands of claims, we need retrieval anyway.

Our strategy was **chunking + targeted retrieval + hierarchical synthesis**. Each document was split into semantic sections. At query time, the retrieval system fetched only the 5-7 most relevant chunks from the entire document corpus — so even a 500-page claim file contributed only 3-4 relevant chunks to the context. The trick was the **chunk retrieval strategy**: we retrieved at the child chunk level (256 tokens, high precision) but returned the parent section (up to 1,500 tokens, high coherence) to the LLM. This gave precise retrieval with readable context.

For **very long individual documents** where every section might be relevant (e.g., a 40-page medical record), we applied **hierarchical summarization**: first summarize each section (medical exam findings, treatment history, billing records) separately using GPT-3.5, then pass those structured summaries to GPT-4 for the fraud analysis. This kept the GPT-4 context tight while preserving all relevant information.

⚠️ **Trap**: Don't just say "we used a model with a long context window." That's the lazy answer. Show you thought about retrieval precision, coherence, and cost.

---

### Q1.2: "How did you measure fraud detection accuracy? What metrics? How did ground truth work?"

**Model Answer**:

Ground truth in fraud detection is the Achilles' heel — it arrives late and with noise. When a claim is submitted, it might take 6-18 months before it's formally classified as fraud (after SIU investigation, legal proceedings, or denial). This creates a significant labeled data lag.

We addressed this through a tiered evaluation strategy:

**Immediate proxy metrics** (available at inference time):
- **Retrieval quality**: RAGAS context precision and faithfulness on a curated evaluation set of 200 claims with manually annotated fraud indicators
- **Adjuster agreement rate**: When the model flagged fraud indicators and the adjuster reviewed, what % of indicators did the adjuster agree were valid? We maintained a live dashboard showing this at 78% initial agreement, improving to 84% over 6 months.

**Lagged ground truth metrics** (available 3-18 months later):
- **Precision**: Of claims flagged as high-risk by our system, what % were confirmed as fraud by SIU? Baseline industry rate ~3-5%, our high-risk flagged population confirmed at ~18%.
- **Recall**: Of claims that SIU later confirmed as fraud, what % did our system flag as high-risk? We maintained ~71% recall — meaning we surfaced 71% of eventual fraud cases early.
- **Lift**: At the top 10% of our fraud risk score, we saw 4.2x lift over baseline — meaning those claims were 4.2x more likely to be fraudulent than a random claim.

**Class imbalance handling**: Fraud is ~2-5% of claims. We addressed this by: (a) weighted loss functions during any ML training, (b) for the LLM-based system, adjusting the fraud risk score threshold (not 0.5 but lower) to increase recall while accepting some false positives at the adjuster review stage.

---

### Q1.3: "How did you make the LLM outputs explainable to insurance adjusters?"

**Model Answer**:

Explainability was non-negotiable. Adjusters are held legally accountable for their fraud determinations — they can't say "the AI told me." They need specific, citable evidence. Our LLM output format enforced this:

```python
# Output schema enforced via OpenAI structured outputs
class FraudAnalysisOutput(BaseModel):
    fraud_risk_score: float  # 0.0-1.0
    fraud_indicators: list[FraudIndicator]
    overall_assessment: str
    confidence_level: str  # "HIGH", "MEDIUM", "LOW"

class FraudIndicator(BaseModel):
    indicator_type: str  # e.g., "TIMING_ANOMALY", "INCONSISTENT_STATEMENT"
    description: str  # Plain English explanation
    supporting_evidence: str  # Direct quote from the claim document
    source_document: str  # Which document (medical record p.3, police report)
    source_chunk_id: str  # For traceability back to the original text
    confidence: float
```

The adjuster UI displayed each fraud indicator with a clickable "View Source" button that opened the exact section of the original document where the evidence was found. This design made the AI a **research assistant**, not a black box — the adjuster still made the final decision but saved hours of document review.

We also implemented **natural language explanations** for each indicator. Not "Anomaly detected in claim timeline" but "The claimant reported the accident occurred on March 15, but the towing company's invoice is dated March 13, creating a 2-day discrepancy that warrants investigation."

💡 **Key insight**: "The most important design decision for this project wasn't technical — it was framing. We positioned the system as 'surfacing evidence for adjusters to evaluate,' not 'detecting fraud.' This framing drove every design choice: citation links, confidence levels, natural language evidence summaries. It also drove adjuster adoption — they trusted a tool that showed its work."

---

### Q1.4: "How would you scale this to 1 million claims per day?"

**Model Answer**:

At 1M claims/day, the bottleneck shifts entirely from the ML model to the **ingestion, embedding, and inference throughput**. Let me address each:

**Ingestion at scale**: S3 → Lambda → Document Processor becomes a bottleneck. Replace with: S3 → SQS FIFO queue → ECS Fargate tasks (auto-scaled) for document processing. At 1M/day (~12 claims/second), 50 parallel ECS tasks with 10-thread parallelism each is sufficient.

**Embedding at scale**: OpenAI ada-002 API has rate limits. At 1M claims × 10 chunks average = 10M embedding API calls/day. Options: (a) batch embedding via OpenAI Batch API (50% cost reduction, 24-hour SLA acceptable for new claim indexing), (b) self-hosted BGE-large model on a T4 GPU cluster (20x cheaper at scale, ~300ms/batch of 32 chunks). I'd use (b) for cost efficiency at 1M/day scale.

**Vector search at scale**: FAISS on a single machine won't handle 1M+ claims × 10 chunks = 10M+ vectors being added daily to a 100M+ vector corpus. Solution: Vertex AI Vector Search (managed, handles petabyte scale, supports streaming index updates) or Qdrant distributed deployment with sharding.

**LLM inference at scale**: GPT-4 at $10/1M tokens is prohibitive for all claims. Apply a **cascaded architecture**: first pass a lightweight classifier (fine-tuned BERT, cost ~$0.001/claim) to score fraud probability. Only the top 20% high-risk claims proceed to GPT-4 inference. This gives 80% cost reduction while applying sophisticated analysis where it matters most. The BERT classifier acts as a triage gate.

---

## SECTION 2: BERT Fine-Tuning for Insurance NLP — Chubb

---

### Q2.1: "Walk me through the LoRA/PEFT fine-tuning process step by step."

**Model Answer**:

Full fine-tuning of BERT on our insurance domain was computationally expensive (training all 110M parameters) and risked catastrophic forgetting — the model losing its general language understanding in favor of narrow domain adaptation. LoRA (Low-Rank Adaptation) solved this elegantly.

**LoRA's core insight**: The weight updates during fine-tuning tend to have low intrinsic dimensionality — they can be decomposed as a product of two low-rank matrices. Instead of adding a full-rank ΔW update to each weight matrix W, LoRA adds `BA` where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k), and r << min(d,k).

**My step-by-step process**:

**Step 1 — Data preparation**: We had 15,000 labeled insurance claim sentences for NER (entities: MEDICAL_CODE, POLICY_NUMBER, CLAIMANT_NAME, COVERAGE_TYPE, INCIDENT_LOCATION) and 8,000 labeled claim-intent pairs (intents: BODILY_INJURY_CLAIM, PROPERTY_DAMAGE_CLAIM, FRAUD_INVESTIGATION, etc.).

**Step 2 — Tokenization strategy**: Standard BERT tokenization splits unknown tokens into subwords, which is bad for domain terms like "ICD-10 codes" (M54.5 → ['M', '##54', '.', '5']), policy formats, and insurance jargon. I added these to the tokenizer vocabulary and extended the embedding matrix accordingly — only the new embeddings were randomly initialized, preserving the pretrained embeddings for existing tokens.

**Step 3 — LoRA configuration**:
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,           # Rank: we tested 4, 8, 16, 32; r=16 gave best accuracy/efficiency trade-off
    lora_alpha=32,  # Scaling factor (lora_alpha/r = effective learning rate for LoRA weights)
    target_modules=["query", "value"],  # Apply LoRA to Q and V matrices in attention
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.TOKEN_CLASSIFICATION  # NER task
)

model = AutoModelForTokenClassification.from_pretrained(
    "roberta-base",  # RoBERTa > BERT for our use case (see below)
    num_labels=len(label_list)
)
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
# → Trainable params: 1,179,648 || All params: 125,483,010 || Trainable%: 0.94%
```

Only 0.94% of parameters are trainable with LoRA r=16 — this meant training in 2 hours on a single V100 GPU vs ~18 hours for full fine-tuning, with only ~18% improvement margin lost.

**Step 4 — Training**: Used HuggingFace Trainer with `SeqEvalMetric` for NER evaluation (entity-level F1, not token-level). Batch size 32, 3 epochs, cosine learning rate schedule with 6% warmup.

**Step 5 — Evaluation harness**: Built automated evaluation in MLflow — after each fine-tuning run, the model was automatically evaluated on a held-out test set, with F1 by entity type (medical codes proved hardest), precision, and recall logged. Compared across different LoRA ranks, alpha values, and target modules.

**Q: "Why RoBERTa over BERT?"**  
Answer: RoBERTa removes the Next Sentence Prediction (NSP) pre-training objective from BERT and trains longer with more data and dynamic masking. For NER and classification tasks (not NSP-dependent), RoBERTa consistently outperforms BERT by 1-3% F1. In our evaluation, RoBERTa-base outperformed BERT-base by 2.1% on NER F1, which mattered at our scale.

**Q: "What would you change if you used an LLM instead of fine-tuned BERT today?"**  
Answer: Modern LLMs (GPT-4, Claude, Gemini) can do NER via structured output prompting with zero-shot or few-shot examples — no training required. The trade-off: LLM NER costs ~100x more per document than a fine-tuned BERT inference, has 5-10x higher latency, and requires careful prompt engineering. For a high-volume insurance claims processing system doing NER on millions of documents, fine-tuned BERT remains the right call — it's cheaper, faster, and more controllable. I'd use LLMs for NER only on the high-complexity long-tail cases where BERT's confidence is low.

---

## SECTION 3: Agentic AI Data Scientist — LangGraph (Chubb)

---

### Q3.1: "How did you handle the SQL executor tool safely? SQL injection? Schema hallucination?"

**Model Answer**:

The SQL executor was the most dangerous tool in our agent because it had direct database write access potential and could be exploited to expose other users' data or execute destructive queries. We applied defense in depth:

**Defense 1 — Read-only database user**: The SQL tool connected with a service account that had SELECT-only permissions on a read-only replica. Even if the LLM generated a DROP TABLE query, it would fail at the database level. This is the most important control — least-privilege at the infrastructure layer.

**Defense 2 — SQL AST validation**: Before executing any SQL, we parsed it with `sqlparse` to build an AST and validate: (a) only SELECT statements allowed (no DML: INSERT/UPDATE/DELETE/DROP), (b) no UNION queries that could exfiltrate data across tables, (c) table names validated against a whitelist of allowed tables for this user's role, (d) LIMIT clause enforced (injected if missing, capped at 10,000 rows).

```python
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import DML

def validate_sql(sql: str, allowed_tables: list[str]) -> tuple[bool, str]:
    statements = sqlparse.parse(sql)
    if len(statements) != 1:
        return False, "Multiple statements not allowed"
    
    stmt = statements[0]
    
    # Check it's a SELECT statement
    first_token = stmt.token_first(skip_cm=True)
    if first_token.ttype is not DML or first_token.value.upper() != 'SELECT':
        return False, f"Only SELECT statements allowed, got: {first_token.value}"
    
    # Extract and validate table names
    tables = extract_table_names(stmt)
    unauthorized = [t for t in tables if t not in allowed_tables]
    if unauthorized:
        return False, f"Unauthorized tables: {unauthorized}"
    
    return True, "Valid"
```

**Defense 3 — Schema hallucination prevention**: LLMs confidently hallucinate column and table names. We injected the actual schema (table names, column names with types) into the system prompt for the SQL agent node, and validated generated SQL column names against the schema before execution. If the LLM generated `SELECT fraud_probability FROM claims` but the column is actually `fraud_score`, validation caught it, returned the error to the LLM, and the LLM self-corrected using the schema.

**Defense 4 — Parameterized queries for user-provided values**: Any user-provided values (e.g., "show me claims for policy number {user_input}") were never interpolated into SQL strings — they were passed as parameters to the database driver, preventing SQL injection at the application layer.

---

### Q3.2: "How would you redesign this as a multi-agent system today?"

*(See Section 3.2 in the Agentic AI deep dive document — cross-reference for the full answer.)*

---

## SECTION 4: Marketing Mix Modeling (MMM) — Axtria (J&J, Stemline)

---

### Q4.1: "Explain your MMM methodology. What is adstock? What is saturation?"

**Model Answer**:

Marketing Mix Modeling is a statistical technique that quantifies the contribution of each marketing channel to sales or revenue outcomes, enabling budget allocation optimization. The core challenge is that marketing effects are **not instantaneous** (a TV ad seen today drives purchases over the next 3 weeks) and **not linear** (doubling spend doesn't double sales — there's diminishing returns).

**Adstock (Carryover Effect)**: Adstock models the lagged and decaying effect of advertising. A TV spot seen today influences purchasing behavior for weeks afterward, but with diminishing impact over time. Geometric adstock is the simplest:

`Adstock(t) = Spend(t) + λ × Adstock(t-1)`

where λ (the decay rate, 0-1) controls how quickly the effect fades. λ=0.7 means 70% of last week's advertising impact carries over to this week. For J&J's immunology products, we found media channels had longer carryover effects than digital (TV λ≈0.75, paid search λ≈0.15) because brand-building effects of TV persist longer than immediate intent-capture from search.

For more flexibility, I used **Weibull CDF adstock** which can model both concave decay (like geometric) and S-shaped carryover patterns (advertising that builds awareness before converting):

`Adstock(t) = Σ spend(t-k) × Weibull_PDF(k; shape, scale)` for k in lag window

**Saturation (Diminishing Returns)**: Beyond a certain spend level, additional marketing investment produces decreasing marginal returns. We modeled this with the **Hill function**:

`Saturated_spend = α × spend^n / (spend^n + K^n)`

where `n` controls the steepness of the saturation curve and `K` is the inflection point (spend level where you get 50% of maximum response). For Stemline's niche oncology product with a small, well-defined prescriber audience, the Hill function saturated very quickly — small increases in detail rep visits delivered rapidly diminishing returns beyond ~3 visits/month per physician.

**Full model structure** (for J&J immunology):

```python
import pymc as pm
import numpy as np

with pm.Model() as mmm:
    # Priors
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    
    # TV channel
    tv_adstock_rate = pm.Beta("tv_adstock_rate", alpha=3, beta=1)  # Prior: likely high carryover
    tv_saturation_k = pm.Gamma("tv_sat_k", alpha=2, beta=1)
    tv_beta = pm.Normal("tv_beta", mu=0, sigma=1)
    
    # Digital channel
    digital_adstock_rate = pm.Beta("digital_adstock_rate", alpha=1, beta=3)  # Prior: low carryover
    digital_beta = pm.Normal("digital_beta", mu=0, sigma=1)
    
    # Apply adstock transformations
    tv_adstocked = geometric_adstock(tv_spend, tv_adstock_rate)
    tv_saturated = hill_saturation(tv_adstocked, tv_saturation_k, n=2)
    
    digital_adstocked = geometric_adstock(digital_spend, digital_adstock_rate)
    
    # Sales equation
    mu = (intercept 
          + tv_beta * tv_saturated
          + digital_beta * digital_adstocked
          + trend_component
          + seasonality_component)
    
    sigma = pm.HalfNormal("sigma", sigma=0.5)
    sales = pm.Normal("sales", mu=mu, sigma=sigma, observed=actual_sales)
    
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True)
```

**Bayesian vs Frequentist MMM**: Bayesian MMM (using PyMC as above) is superior for marketing because: (a) we can encode domain knowledge as priors (TV typically has longer carryover than digital), (b) we get full posterior distributions over parameters rather than point estimates — this gives uncertainty quantification for budget recommendations, (c) it handles multicollinearity between channels more gracefully through regularizing priors.

---

### Q4.2: "What is the genetic algorithm you designed for budget optimization?"

**Model Answer**:

After fitting the MMM and understanding the contribution of each channel, the business question is: given a fixed budget of $X, how should we allocate it across channels to maximize sales? This is a **constrained optimization problem** that's non-linear and non-convex (because of saturation curves), so gradient-based optimization often gets stuck in local optima.

I used a **multi-objective genetic algorithm (NSGA-II)** to optimize for two competing objectives simultaneously: maximize expected sales AND minimize budget variance (risk). The Pareto front of the NSGA-II gives a set of optimal budget allocations ranging from "high expected sales, high risk" to "lower expected sales, lower risk" — the marketing team can then choose their preferred risk-return tradeoff.

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

class MarketingBudgetOptimization(Problem):
    def __init__(self, mmm_params, total_budget, n_channels):
        super().__init__(
            n_var=n_channels,
            n_obj=2,  # Maximize sales, minimize variance
            n_ieq_constr=1,  # Budget constraint: sum(allocation) <= total_budget
            xl=np.zeros(n_channels),  # Lower bounds: 0 spend per channel
            xu=np.array([total_budget * 0.6] * n_channels)  # Upper bounds: no channel > 60% of budget
        )
        self.mmm_params = mmm_params
        self.total_budget = total_budget
    
    def _evaluate(self, X, out, *args, **kwargs):
        # X is a matrix of candidate budget allocations (n_population × n_channels)
        
        # Predict sales for each allocation using the fitted MMM
        predicted_sales = []
        for allocation in X:
            sales = self.predict_sales(allocation, self.mmm_params)
            predicted_sales.append(sales)
        
        predicted_sales = np.array(predicted_sales)
        
        out["F"] = np.column_stack([
            -predicted_sales.mean(axis=1),  # Objective 1: maximize expected sales (negate for minimization)
            predicted_sales.std(axis=1)     # Objective 2: minimize variance
        ])
        
        # Budget constraint: total spend must not exceed budget
        out["G"] = X.sum(axis=1) - self.total_budget

algorithm = NSGA2(pop_size=200, eliminate_duplicates=True)
result = minimize(
    MarketingBudgetOptimization(mmm_params, total_budget=10_000_000, n_channels=6),
    algorithm,
    ("n_gen", 300),
    verbose=True
)

# result.X contains the Pareto-optimal budget allocations
# result.F contains their (expected_sales, variance) values
```

The J&J marketing team received a Pareto front visualization and could select their preferred risk tolerance point. We validated by comparing our recommended allocation to the actual spend in a subsequent quarter — the model-recommended mix outperformed the business-as-usual allocation by ~10% in revenue, which was our headline result.

---

## SECTION 5: Omnichannel Attribution — Axtria

---

### Q5.1: "Explain Markov Chain attribution. Why did you add attention mechanisms?"

**Model Answer**:

**Markov Chain attribution** models the customer journey as a Markov process — a sequence of touchpoints where each state (channel) has a probability of transitioning to the next. The key innovation over rule-based attribution (first-touch, last-touch) is the **removal effect**: to measure the value of channel X, we remove X from the graph and measure how much conversion probability drops. The channel's attribution weight is proportional to this drop.

```
Customer journey: Email → Social → Paid Search → Conversion

Markov chain states: {Email, Social, PaidSearch, Conversion, NonConversion}

Transition matrix P[i][j] = probability of moving from channel i to channel j

Removal effect of PaidSearch:
  - Baseline conversion probability with all channels: 0.25
  - Conversion probability without PaidSearch: 0.18
  - PaidSearch removal effect: (0.25 - 0.18) / 0.25 = 28% attribution weight
```

**The limitation of Markov chains**: They're **memoryless** — the transition probability from Social → PaidSearch is the same regardless of whether the customer came from Email or Facebook first. But in reality, the effect of seeing a paid search ad is very different if you've just seen a TV commercial vs if this is your first brand exposure. Journey history matters.

**Why attention mechanisms helped**: I modeled the customer journey as a sequence problem, where each touchpoint should be weighted based on its **relevance to the final conversion given the full journey context**. This is exactly what self-attention captures. Using a Transformer encoder on the touchpoint sequence:

- Each touchpoint is embedded (channel type, position, time delta to next touchpoint)
- Multi-head self-attention learns which touchpoints in the journey most strongly influence each other
- The attention weights, averaged across heads, provide **interpretable attribution weights**

The result: 7% AUC improvement over Markov chains on held-out data, and more importantly, the attention weights revealed non-obvious patterns — for certain immunology drugs, early-funnel Medical Congress touchpoints were getting very high attention weights even though they weren't the last touch. This insight changed how J&J valued medical education events in their budget.

---

## SECTION 6: Pharma Rep Communication System — GenAI + Neo4j (Axtria)

---

### Q6.1: "Why Neo4j? What does the knowledge graph store?"

**Model Answer**:

Pharmaceutical rep-to-doctor communication is fundamentally a **graph problem**. A pharma rep needs to know: which doctors prescribe our drug, what their specialty is, what studies they've authored, what clinical trial data is relevant to their patient population, and what key opinion leaders in their network have said about the drug. These are all relationships — Neo4j's native graph storage and Cypher query language are purpose-built for this.

**Knowledge graph schema**:
```
Nodes:
- (Doctor {id, name, specialty, institution, prescriber_tier, location})
- (Drug {name, MOA, indication, dosing, side_effects})
- (Study {title, phase, results, journal, year})
- (KOL {name, specialty, influence_score})
- (TherapeuticArea {name, disease})
- (RepresentativeVisit {date, outcome, message_delivered})

Relationships:
- (Doctor)-[:PRESCRIBES {volume, recency}]->(Drug)
- (Doctor)-[:SPECIALIZES_IN]->(TherapeuticArea)
- (Doctor)-[:CO_AUTHORED]->(Study)
- (KOL)-[:ADVOCATES_FOR]->(Drug)
- (Doctor)-[:INFLUENCED_BY]->(KOL)
- (RepresentativeVisit)-[:TARGETED]->(Doctor)
```

**How GPT-4 used the knowledge graph**: The system first ran a Cypher query to retrieve structured facts about the target doctor:

```cypher
MATCH (d:Doctor {id: $doctor_id})-[:SPECIALIZES_IN]->(ta:TherapeuticArea)
MATCH (d)-[:PRESCRIBES]->(drug:Drug)
OPTIONAL MATCH (d)-[:CO_AUTHORED]->(study:Study)
OPTIONAL MATCH (d)-[:INFLUENCED_BY]->(kol:KOL)
RETURN d, ta, collect(drug) as drugs, collect(study) as studies, collect(kol) as kols
```

This structured data was then combined with retrieved chunks from the drug's clinical documentation (via RAG on a vector store of medical literature) and passed to GPT-4 to generate a personalized communication plan: what studies to reference, which KOLs to name-drop, what clinical aspects to emphasize based on the doctor's specialty and patient population.

**Guardrails for regulatory compliance**: FDA regulations prohibit off-label drug promotion. I implemented a guardrails layer that: (a) maintained a curated "approved claims" document in the vector store (only approved indication language), (b) had GPT-4 generate content only from the approved claims retrieval context, (c) ran a compliance checker (another LLM call with a strict compliance prompt) on every generated communication before it was shown to the rep, (d) logged all generated content for regulatory audit trails.

---

## SECTION 7: CVS Health / Aetna Projects — EXL (6 years)

---

### Q7.1: "Explain your Cox Proportional Hazards model for survival analysis. What are the assumptions? What if they're violated?"

**Model Answer**:

The Cox PH model is a semi-parametric model for time-to-event data. It models the hazard function (instantaneous risk of the event at time t, given survival to t) as:

`h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₙXₙ)`

The `h₀(t)` is the baseline hazard (non-parametric, left unspecified), and `exp(β'X)` is the covariate effect. The **proportional hazards assumption** states that the hazard ratio between any two subjects is constant over time — their "survival curves" are parallel on the log scale.

**At CVS Health/Aetna**, I used Cox PH for two applications:
1. **Patient readmission risk**: Time to hospital readmission with covariates: age, comorbidity score, discharge medication adherence, prior readmission history
2. **Customer churn**: Time to plan disenrollment with covariates: plan satisfaction scores, claim denial rate, premium changes, competitor offerings

**Key assumptions**:
1. **Proportional hazards**: The hazard ratio is constant over time. Test with Schoenfeld residuals test (`cox.zph()` in R, or `CoxPHFitter.check_assumptions()` in lifelines Python).
2. **Log-linearity**: The log hazard is linear in the covariates. Check with martingale residuals.
3. **Independence**: Survival times for different subjects are independent. Violated with clustered data (multiple patients from same hospital).

**When violated**:
- If PH assumption fails: use **time-varying coefficients** (`tt()` in R's coxme) or stratify by the violating variable
- If log-linearity fails: use splines for the offending covariate
- If independence fails: use **frailty models** (mixed effects Cox, `coxme` package) or clustered standard errors

In our readmission model, the age variable violated PH (older patients' hazard ratio vs younger patients changed over the follow-up period). We resolved this by including an `age × time` interaction term.

---

### Q7.2: "Why did you use TF-IDF + cosine similarity for NLP entity matching rather than learned embeddings?"

**Model Answer**:

This was a deliberate, context-appropriate choice. The entity matching problem was matching 20,000 plan sponsor names from one database against 40,000 company name records from another — essentially string matching with variations: "IBM Corporation" vs "International Business Machines Corp" vs "IBM Corp."

**Why TF-IDF + cosine similarity over embeddings at the time (2019)**:
1. **Interpretability**: TF-IDF similarity is fully explainable — a match on "International Business Machines" has a clear explanation. Embedding similarity is a black box number.
2. **The character n-gram trick**: I used **character-level TF-IDF** (n-grams of characters, not words), which handles abbreviations, truncations, and misspellings better than word-level TF-IDF.
3. **Scale and speed**: Spark MLlib's TF-IDF + cosine similarity runs across 2M pairs efficiently. Embedding similarity at scale (100K × 40K = 4B pairs) would require FAISS ANN, which was more complex to set up on Spark at the time.
4. **Cost**: Self-contained in Spark MLlib, no API costs.

**What I'd use today**: Sentence-BERT embeddings with FAISS ANN search would likely outperform TF-IDF for complex entity matching, especially for longer company names with semantic variations. But for this specific use case (mostly abbreviation and truncation variations), character n-gram TF-IDF was surprisingly competitive and achieved 75% accuracy, which the business found acceptable given the manual baseline of hours of work.

---

## SECTION 8: Behavioral & Leadership Questions

---

### Q8.1: "Tell me about a time you convinced a non-technical stakeholder to adopt AI."

**STAR Model Answer**:

**Situation**: When I joined Chubb's claims analytics team, the underwriters were deeply skeptical of any AI-based assistance. They'd seen a prior AI project fail — the model had recommended claim settlements that turned out to be wrong, and one underwriter faced regulatory scrutiny because of it. There was genuine fear that AI would either make wrong decisions or create legal liability.

**Task**: I was tasked with driving adoption of the Underwriters Assistance Tool — a claims summarization and coverage matching system I'd built. Without underwriter buy-in, the system would never be used regardless of how good the model was.

**Action**: Rather than presenting the system as "AI that helps you decide," I reframed it completely as "AI that does the reading, you do the deciding." I spent 3 one-on-ones with senior underwriters before any demo, genuinely listening to what parts of their workflow were tedious vs what required their expert judgment. They consistently said: reading through long claim documents and policy documents to identify coverage applicability was tedious. Making the actual coverage decision was their expert value-add.

I then tailored the demo to show exactly that: "The AI read the 47-page claim file and flagged these 3 potential coverage issues on pages 12, 28, and 41. Here are the relevant policy clauses. You decide whether each applies." The AI did the reading; the underwriter did the deciding. I also showed that the system had a "I'm uncertain" mode — when confidence was low, it explicitly said "I'm not confident about this coverage interpretation — flagging for senior review."

**Result**: The lead underwriter who was most skeptical became the biggest internal advocate. She co-presented the tool at Chubb's North America analytics showcase. Adoption went from 0% to 78% of eligible underwriters using it weekly within 6 months. The Q3 2025 North America Analytics Team Recognition award was partially for this project.

💡 **Key insight**: "The technical quality of the AI mattered less than the interaction design. The moment I stopped trying to make the AI seem more capable and started making it seem more honest about its limitations, trust went up dramatically."

---

### Q8.2: "Describe a production ML failure. How did you handle it?"

**STAR Model Answer**:

**Situation**: Three months after deploying the fraud risk scoring system at Chubb, I started receiving reports from claims managers that the model was generating high fraud risk scores for a specific category of legitimate medical claims — specifically, claims from claimants who had recently relocated from Florida. The false positive rate for this group had jumped from ~5% to ~31%.

**Task**: Diagnose the root cause, fix it, and restore trust with the claims managers who were now questioning the entire system.

**Action**: I first confirmed the issue wasn't a false alarm by pulling the last 30 days of predictions for Florida-relocation claims and computing the false positive rate against ground truth (cases that adjusters had reviewed and cleared). Confirmed: 31% FPR vs 5% baseline.

Root cause analysis using SHAP values: the model was placing very high weight on the feature "state_change_30_days" (whether the claimant had changed their registered state in the last 30 days). During the training period (pre-2024), Florida had a particularly high fraud rate, and state changes often preceded fraudulent activity. But Hurricane Ian recovery had just driven a massive legitimate relocation wave — hundreds of thousands of Floridians who lost homes legitimately moved to other states and filed legitimate insurance claims for their new residences.

**Fix**: (1) Immediate: added a "Hurricane Ian relocation exclusion" rule — claims where the address change was correlated with known disaster relief areas were excluded from the `state_change_30_days` feature computation. This was a hard rule, not ML. (2) Medium-term: retrained the model with recent data that included post-hurricane legitimate relocation claims to update the model's prior on state changes. (3) Process: added monitoring on false positive rates by claim subcategory, not just overall, so we'd catch subcategory-specific issues faster in future.

**Result**: FPR for the affected group returned to 6% within 2 weeks. I proactively briefed the claims managers on what happened, what we'd done to fix it, and what monitoring we'd added. They appreciated the transparency — it actually increased their trust in the system because they saw we caught and fixed issues quickly.

⚠️ **Common mistake**: Candidates say "we fixed the model." The right answer shows: (a) disciplined debugging with SHAP/interpretability, (b) a short-term fix (rule) while the model fix is built, (c) process improvement to prevent recurrence, (d) transparent stakeholder communication.

---

### Q8.3: "How would you approach the first 90 days as Solutions Architect at Huge?"

**Model Answer**:

I think about the first 90 days in three phases:

**Days 1-30 — Listen and map**: My first priority is understanding what's already working and what's not. I'd have 1:1s with every team I'll interact with: data scientists, ML engineers, client services, product managers. I'd audit the current technical landscape: what ML systems are deployed for clients, what the CI/CD setup looks like, what observability tools are in place. I'd also spend time with the client-facing teams to understand what clients are asking for that isn't being delivered — that's where the biggest leverage is.

**Days 31-60 — Deliver a quick win**: Based on what I learned in the first month, I'd identify one meaningful improvement I can implement independently that demonstrates value. This might be setting up LangSmith observability on an existing agentic system, improving a RAG retrieval pipeline's recall by adding re-ranking, or creating a Terraform module that makes it faster to provision ML infrastructure for new client projects. Quick wins build credibility.

**Days 61-90 — Propose the 6-month roadmap**: With a clear picture of the technical landscape and one delivered improvement under my belt, I'd present a prioritized technical roadmap: what platform investments would most accelerate client delivery? This might be building a shared Vector Infrastructure layer (Vertex AI Vector Search) that multiple client RAG applications share, establishing a standard MLOps pipeline template on Vertex AI Pipelines, or standing up a semantic routing layer that handles multi-client query routing intelligently.

💡 **Key Huge-specific angle**: "One thing I'd do in the first 30 days is specifically understand Huge's competitive differentiation as a design & technology company. How is 'Intelligent Experiences' positioned to clients? What's the narrative? As Solutions Architect, I'm not just building ML systems — I'm helping Huge tell a compelling AI story to clients like Google, McDonald's, and Nike. Understanding that story is as important as understanding the technical stack."

---

### Q8.4: "A Fortune 500 client asks for a GenAI chatbot in 6 weeks. What do you say?"

**Model Answer**:

I'd say yes — with a specific, scoped definition of what we can deliver in 6 weeks vs what the full roadmap looks like.

First, I'd ask clarifying questions: What is the chatbot supposed to do? Who are the users? What data does it have access to? What does "done" look like? — because "chatbot" can mean anything from a FAQ bot to a full agentic assistant.

If they want a **RAG-based customer-facing assistant** over a specific knowledge base (e.g., Nike's product catalog + brand guidelines), 6 weeks is achievable: Week 1-2: data ingestion and vector index setup; Week 3-4: RAG pipeline + LLM integration + basic UI; Week 5: evaluation, guardrails, load testing; Week 6: production deployment on Cloud Run with monitoring.

If they want an **agentic assistant** that can take actions (place orders, update CRM, personalize recommendations), 6 weeks is not enough for production quality. I'd propose: a scoped MVP (3 weeks) that demonstrates the core value with 2-3 hardcoded tool integrations, followed by a 3-month roadmap for the full production system with proper security, monitoring, and multi-tool orchestration.

My framework for scoping client AI requests: (1) clarify the definition of done, (2) identify the irreducible complexity (what can't be rushed — security, data quality, evaluation), (3) define what you CAN do in 6 weeks that demonstrates genuine value and builds toward the full vision, (4) be explicit about what you're deferring and why.

"The fastest way to lose a client's trust is to overpromise and underdeliver. The second fastest is to underscope and show them something that doesn't feel like real value. The sweet spot is an honest, ambitious scope that ships what matters in 6 weeks and has a credible path to the full vision."

---

*End of Project Deep Dives & Behavioral Document*

---

> [!TIP]
> For every project, memorize these 4 numbers: the **problem scale** (how many documents, users, transactions), the **business impact** ($ saved, % improvement, time reduced), the **key technical decision** (why X over Y), and the **lesson learned** (what you'd do differently). These four anchors make every project answer compelling.
