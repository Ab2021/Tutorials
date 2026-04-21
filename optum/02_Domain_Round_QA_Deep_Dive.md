# 🎯 Domain Round — Deep Q&A & Answering Frameworks
### Optum Sr. AI/ML Engineer — Expected Questions + Model Answers

---

## PART 1: GENAI & LLM ENGINEERING QUESTIONS

---

### Q1: Walk me through how you'd design a production RAG pipeline for healthcare claims analysis.

**Framework: BUILD-MEASURE-LEARN (always lead with business problem)**

> "The business problem first: Healthcare claims are unstructured and voluminous — an adjudicator reviewing a claim needs to quickly understand its clinical context, compare it to similar past claims, and make a coverage or fraud decision. A RAG pipeline augments that process.

> **Stage 1 — Ingestion & Indexing:**
> - Source: Claim documents, clinical notes, policy documents, historical investigation reports
> - Chunking: Semantic chunking (not fixed-size) — a claim section boundary matters more than 512 tokens
> - Embedding: Domain fine-tuned model (e.g., ClinicalBERT embeddings or fine-tuned E5/BGE on claims pairs) — generic embeddings miss clinical terminology
> - Vector store: FAISS for offline, Pinecone or AWS OpenSearch for production with real-time updates
> - Hybrid search: Dense (semantic) + sparse (BM25 keyword) — critical for medical codes like ICD-10 which have exact-match importance

> **Stage 2 — Retrieval:**
> - Query: Claim text or investigator question
> - Retrieve top-k similar historical claims (k=5-10)
> - Re-ranking: Cross-encoder re-ranker for precision (expensive but worth it for healthcare)
> - Context window management: Truncate retrieved docs to fit LLM context, prioritize by relevance score

> **Stage 3 — Generation:**
> - Structured prompt with retrieved context + system role (clinical fraud analyst)
> - Structured output schema: Force JSON with risk_level, indicators, confidence, recommended_action
> - Temperature: Low (0.1-0.2) for consistency and auditability
> - Guardrails: Faithfulness check (does output reference retrieved context?), PHI redaction, injection prevention

> **Stage 4 — Evaluation:**
> - RAGAS framework: Faithfulness >0.85, Context Recall >0.70, Answer Relevance >0.80
> - Shadow mode: Compare system flags to investigator verdicts for 60 days before production
> - Monitoring: Weekly faithfulness score trends, retrieval quality drift, PSI on claim text distribution"

---

### Q2: What is the difference between LangChain and LangGraph? When would you use each?

> "LangChain is great for building linear or simple DAG-style pipelines — you chain prompts, LLMs, and tools together sequentially. The LCEL (LangChain Expression Language) makes composition elegant. I've used it in production for our fraud RAG pipeline and the Agentic BI tool.

> LangGraph extends LangChain with a **stateful graph model** — think of it as a state machine where nodes are agent actions and edges can be conditional. This becomes essential when:
> - You need **cycles**: an agent needs to retry, self-correct, or iterate
> - You need **conditional routing**: based on output confidence, route to different agents or human review
> - You need **multi-agent coordination**: a planner agent orchestrates specialist agents
> - You need **persistent state**: the conversation state survives across multiple tool calls

> **Healthcare use case where LangGraph wins:** A Prior Authorization agent that needs to (1) extract clinical criteria → (2) retrieve payer policy rules → (3) match criteria to rules → (4) generate recommendation → (5) IF low confidence, route to human reviewer → (6) log decision with full audit trail. That conditional branch + state persistence = LangGraph territory.

> **Where LangChain is fine:** Simple RAG Q&A, single-step summarization, document classification pipelines without branching."

---

### Q3: How do you fine-tune an LLM? Walk me through the spectrum of approaches.

**Use the Depth-Ladder technique: Start accessible, go deeper when probed**

> "There's a spectrum from no-training to full fine-tuning:

> **Level 1 — Prompt Engineering / In-Context Learning:** No weight updates. Craft system prompts + few-shot examples. Fastest, cheapest, reversible. Works well when the base model already knows the task and you're steering behavior.

> **Level 2 — RAG:** Augment the model's 'knowledge' without touching weights. Best when you need domain-specific information that changes frequently (e.g., updated clinical guidelines, new fraud patterns).

> **Level 3 — Prompt Tuning / Prefix Tuning:** Add trainable 'soft prompt' tokens to the input — only those tokens update. Extremely parameter-efficient.

> **Level 4 — LoRA / QLoRA (most common for production):** Add rank-r decomposition matrices (A and B) to attention weight layers: `W' = W + AB`. Only A and B are trained — typically 0.1-1% of total parameters. QLoRA quantizes the base model to 4-bit for memory efficiency. This is the sweet spot for most production fine-tuning.

> **Level 5 — Full Fine-Tuning:** Update all weights. Requires the most compute and data; risk of catastrophic forgetting. Use only when LoRA isn't achieving target performance and you have large supervised dataset.

> **When I'd use each for Optum:**
> - Clinical summarization: Start with RAG + prompt engineering (guidelines change frequently)
> - Fraud classification: LoRA fine-tune on labeled claims pairs (stable task, need domain adaptation)
> - Structured output (ICD coding): LoRA fine-tune with strong output format supervision
> - New medical entity type recognition: Could justify full fine-tune if large labeled dataset exists"

---

### Q4: What is the difference between RAG and fine-tuning? How do you choose?

| Dimension | RAG | Fine-Tuning |
|---|---|---|
| **Knowledge** | External, updatable at retrieval time | Baked into weights at training time |
| **Update cost** | Add docs to vector store (minutes) | Re-training (hours to days) |
| **Hallucination risk** | Lower — grounded in retrieved context | Higher — model may confabulate |
| **Latency** | Higher — retrieval + generation | Lower — just generation |
| **Best for** | Dynamic knowledge, document Q&A, factual lookup | Style/format/task adaptation, new task types |

> "In healthcare: I'd use **RAG for policy-dependent tasks** (prior auth criteria change quarterly) and **fine-tuning for structural tasks** (ICD-10 code assignment has a stable label space and benefits from output format training). The production answer is often both: fine-tune the model on format + domain style, then RAG for factual grounding."

---

### Q5: How would you implement guardrails for an LLM in a HIPAA-compliant healthcare system?

> "I think about guardrails in three layers:

> **Input Layer:**
> - Prompt injection detection: Pattern matching + a small binary classifier trained to detect injection attempts
> - PHI detection: Regex + NER model to flag if user is sending raw PHI (they shouldn't — it should be de-identified before hitting the LLM)
> - Topic scope enforcement: Use a classifier to ensure the query is within the system's intended domain

> **Model/Inference Layer:**
> - System prompt pinning: The system prompt is immutable — user cannot override it via instruction
> - Temperature control: Low temperature (0.1-0.2) for factual/clinical tasks reduces creative hallucination
> - Structured output schemas: Pydantic or JSON schema enforcement — the model must produce valid structured output, not free text. This reduces hallucination surface area significantly.

> **Output Layer:**
> - PHI/PII redaction: Before the output returns to the user, run a redaction pass (regex + NER)
> - Faithfulness check: LLM-as-judge verifies that each claim in the output is grounded in the retrieved context
> - Toxicity/safety filter: A moderation classifier (e.g., AWS Bedrock Guardrails or custom)
> - Confidence thresholding: If model confidence is below threshold, flag for human review instead of auto-acting

> **Audit Layer:**
> - Log every prompt + response with user ID, timestamp, retrieved context — critical for HIPAA audit trails
> - Retention policy: 6 years minimum per HIPAA requirements

> In AWS Bedrock, much of this is natively available through **Bedrock Guardrails** — content filtering, PII redaction, grounding check, denied topics. I'd use that as the baseline and add custom input validation on top."

---

### Q6: Explain how multi-head attention works. Why multiple heads?

> "At the core, self-attention computes:
> `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

> Q, K, V are linear projections of the input. The dot product QK^T measures similarity between query tokens and key tokens; we scale by √d_k to prevent vanishing gradients in softmax for large d_k; the result is a weighted average of value vectors.

> **Why multiple heads?** Each head uses different learned Q, K, V projection matrices, so it can attend to different aspects of the input simultaneously:
> - Head 1 might learn syntactic relationships (subject ↔ verb)
> - Head 2 might learn semantic co-reference (pronoun ↔ entity)
> - Head 3 might learn positional proximity

> In healthcare NLP, this matters: one head might attend to the medication name, another to the dosage, another to the temporal qualifier ('was prescribed 3 months ago'). Multi-head lets the model capture all simultaneously.

> The outputs of all heads are concatenated and projected back: `MultiHead(Q,K,V) = Concat(head_1...head_h)W^O`

> **Computational cost:** O(n²·d) in sequence length — this is why long clinical notes are expensive and why techniques like sliding window attention or sparse attention become relevant for document-length inputs."

---

## PART 2: ML FOUNDATIONS (DON'T IGNORE THESE)

---

### Q7: You have severe class imbalance in healthcare fraud (1% fraud rate). How do you handle it?

> "I address it at three levels:

> **Data level:**
> - Oversampling: SMOTE to generate synthetic fraud examples in feature space
> - Undersampling: Random or informed undersampling of majority class
> - Class weights: `class_weight='balanced'` in sklearn — equivalent to reweighting the loss function

> **Model level:**
> - `scale_pos_weight` in XGBoost: Set to `(# negative) / (# positive)` = ~99
> - Focal Loss for neural models: Downweights easy negatives, focuses training on hard examples
> - Threshold calibration: Don't use 0.5 as decision boundary — optimize threshold on validation set for target Precision/Recall trade-off

> **Evaluation level (most important):**
> - Never use accuracy (99% accuracy by predicting all non-fraud)
> - Use **PR-AUC** (Precision-Recall AUC) — ROC-AUC is misleading with imbalance
> - Primary metric: **Recall at X% precision** — business decides how many false positives investigators can handle
> - At Chubb: We operated at 78% recall / 22% FPR — the business found that acceptable given investigator capacity

> **Real production insight:** The threshold you deploy is a business decision, not a model decision. Present the full PR curve to stakeholders and let them choose the operating point."

---

### Q8: How would you evaluate a machine learning model in production at Optum's scale?

> "I think about 4 layers:

> **1. Offline Evaluation (before deployment):**
> - Time-based split (never random for healthcare — it causes temporal leakage)
> - Primary metric: PR-AUC for classification, RMSE/Spearman for regression
> - Calibration: Reliability diagram + Brier score — critical in healthcare where clinicians act on probabilities
> - Subgroup analysis: AUC must not degrade >5% for protected groups (age, gender, race)

> **2. Shadow Mode (first production deployment):**
> - New model runs but doesn't influence decisions — outputs logged alongside existing system
> - Compare against human decisions or existing rules after 30-60 days
> - Success: Shadow model performs as well or better with acceptable FPR

> **3. A/B / Champion-Challenger (gradual rollout):**
> - Champion gets 90% traffic, challenger gets 10%
> - Monitor: Primary KPI + guardrail metrics (FPR must not increase)
> - Duration: Long enough to reach statistical significance (use sample size formula)

> **4. Production Monitoring (ongoing):**
> - Data drift: PSI on input feature distributions — >0.25 triggers investigation
> - Concept drift: Tracking model performance metrics against delayed labels
> - LLM-specific: Weekly faithfulness score sampling, retrieval quality tracking
> - Alert thresholds: Auto-page when PSI >0.25 or metric drops >10% from baseline"

---

## PART 3: SYSTEM DESIGN — HEALTHCARE AI

---

### Q9: SYSTEM DESIGN: Design a Prior Authorization AI system using LLMs and LangGraph.

**Framework: Clarify → Architecture → Evaluation → Safety → Monitoring**

> **Clarification questions first:**
> "Is this fully automated or human-in-the-loop? What's the latency requirement — real-time at point-of-care or batch overnight? How many prior auth requests per day? What payer types — commercial, Medicare, Medicaid? What's the tolerance for false denials vs. false approvals?"

> **Architecture:**

```
PRIOR AUTH AI SYSTEM

Inputs:
├── Clinical Documentation (unstructured: notes, lab results, imaging reports)
├── Payer Policy Rules (structured + semi-structured PDFs)
└── Patient History (EHR claims data)

LangGraph Agent Flow:
Node 1: Document Parser
├── Extract: Clinical diagnosis, requested procedure, supporting evidence
└── Tool: BERT NER + clinical IE pipeline

Node 2: Policy Retrieval (RAG)
├── Query: Payer-specific coverage criteria for requested procedure
└── Tool: Vector DB over payer policy corpus (hybrid BM25 + dense)

Node 3: Criteria Matching
├── For each coverage criterion: Does clinical documentation support it?
├── Tool: LLM structured evaluation (JSON output per criterion)
└── Confidence score per criterion

Node 4: Decision Synthesis
├── IF all criteria met + confidence >0.90 → Draft APPROVAL
├── IF criteria not met + confidence >0.90 → Draft DENIAL with reason
└── IF confidence <0.90 OR edge case → ROUTE TO HUMAN REVIEW

Node 5: Compliance Check
├── PHI redaction from output
├── Regulatory language validation
└── Audit log: Full trace of evidence → decision

Output:
├── Structured decision: APPROVE / DENY / PENDING_REVIEW
├── Supporting evidence citations from clinical docs
└── Audit trail for appeals process
```

> **Evaluation:**
> - Accuracy vs. gold-standard human decisions on 500-case test set
> - Faithfulness: All decision rationale traced to clinical documentation (no hallucination)
> - Consistency: Same clinical facts → same decision across 10 runs
> - Overturn rate: How often do human reviewers reverse the AI recommendation?

> **Safety / Responsible AI:**
> - Human review gate for all edge cases + low-confidence cases
> - No PHI in LLM prompts that leave VPC (use Bedrock within AWS with BAA)
> - Full audit trail per HIPAA requirements
> - Fairness: Monitor approval rates by demographic group — disparate impact detection"

---

## PART 4: BEHAVIORAL QUESTIONS (STAR+ FORMAT)

---

### Q10: Tell me about a time you built something innovative that didn't have a clear playbook.

**Story: Agentic BI Tool at Chubb**

> **Situation:** Our data science team was spending 40% of analyst time answering ad-hoc reporting questions from business stakeholders — same types of questions, repeated weekly, requiring custom SQL + Python scripts each time.

> **Task:** I proposed building an autonomous agent that could understand natural language analytical questions and execute the full analytical pipeline — SQL generation, data retrieval, Python analysis, and visualization.

> **Action:** I designed and built a LangChain-based agent with three custom tools: (1) a SQL query generator that validated against our schema and ran against BigQuery, (2) a Python executor for statistical analysis, and (3) a chart generator. The hardest part was safety — the agent needed to handle ambiguous queries gracefully, never execute destructive operations, and always cite data sources. I added a dry-run validation step before any SQL execution and a scope-enforcement classifier that blocked out-of-domain queries.

> **Result:** The tool achieved 87% task completion rate on our benchmark of 100 diverse queries. Analyst time on recurring reports dropped by approximately 60%. It was recognized in our Q3 2025 North America Analytics Team award.

> **Reflection:** What I learned: Evaluating agents is fundamentally different from evaluating models — there's no single accuracy score. I developed a 4-dimensional framework (task completion, tool accuracy, answer correctness, reliability/safety) that I'd apply to any agent system. For Optum, that same framework applies to any clinical decision-support agent."

---

### Q11: Describe a situation where you had to push back on a technical decision.

**Story: Fraud System Deployment Pressure**

> **Situation:** At Chubb, after building the RAG fraud detection system, there was business pressure to go live immediately. The model was showing promising shadow-mode results after only 3 weeks.

> **Task:** I needed to decide whether to approve production rollout or hold for more validation — even with stakeholder pressure to ship.

> **Action:** I presented a risk matrix: the cost of a false positive (wrongly flagging a legitimate claim → claimant hardship + regulatory risk) vs. the cost of insufficient validation (undiscovered failure modes in production). I recommended extending shadow mode to 60 days and implementing a mandatory human review gate for all high-risk flags regardless of model confidence. I framed this not as slowing down but as protecting the company from regulatory exposure.

> **Result:** Stakeholders agreed. In the additional 30 days, we discovered a specific claim type (coordination-of-benefits claims) where the model was producing systematically high false-positive rates — a failure mode we would have shipped into production. Fixing that before launch saved significant investigator resources and avoided claimant complaints.

> **Reflection:** In healthcare AI, deployment velocity is never the primary KPI. I'd apply the same principle at Optum — especially for any clinical decision-support system where an error affects patient access to care."

---

### Q12: How do you stay current with the fast-moving GenAI space while still delivering production systems?

> "I use a 3-tier system:

> **Tier 1 — Signal Monitoring (daily, 15 min):** Hugging Face Papers, Twitter/X AI accounts (Andrej Karpathy, Sebastian Raschka), and ArXiv's cs.CL/cs.LG daily digest. I filter for papers that change what's possible in production, not just academic benchmarks.

> **Tier 2 — Applied Learning (weekly, 2-3 hours):** I implement at least one new technique per week at a POC level. Currently: exploring LangGraph's streaming support and AWS Bedrock's new Agents features. Last quarter I ran a structured benchmark comparing LoRA vs. QLoRA vs. unsloth for a domain fine-tuning task.

> **Tier 3 — Production Readiness Assessment:** Before adopting anything new, I ask: Does it have a production-ready API? What's the failure mode when it breaks? What's the rollback plan? That filter keeps POC experimentation from becoming production tech debt.

> Specifically for this role: I've been closely following AWS Bedrock's roadmap — the Guardrails additions and the new Agents for Bedrock features directly apply to what I'd build at Optum."

---

*End of Q&A Deep Dive — See companion documents for System Design and Resume Deep Dive*
