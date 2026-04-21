# 🏆 Quick Reference Cheat Sheets
### All 4 Rounds: The Most Critical Facts to Have at Fingertips

> Print this. Study this on interview day morning.

---

## CHEAT SHEET 1: YOUR KEY NUMBERS (Know These Cold)

| Project | Metric | Number |
|---|---|---|
| Fraud Detection (Chubb) | Recall achieved in shadow mode | 78% |
| Fraud Detection (Chubb) | False positive rate | 22% |
| Fraud Detection (Chubb) | Mean time to flag improvement | >30 days → <2 days |
| Readmission Risk (EXL) | AUC journey | 0.82 → 0.89 (+8.5% relative) |
| Readmission Risk (EXL) | Clinical notes dataset size | 50K BERT fine-tuning runs |
| CLV Model (EXL) | Records processed | 2M+ prospects |
| CLV Model (EXL) | Processing time reduction | 70% (6h → 1.8h) |
| CLV Model (EXL) | Top decile lift | 4.2x average |
| Entity Matching (EXL) | Accuracy | 75% |
| Entity Matching (EXL) | Scale | 20K plans, 40K company names |
| Pharma RecommSys (Axtria) | Email open rate lift | +23% |
| Pharma RecommSys (Axtria) | Meeting acceptance lift | +17% |
| MMM (Axtria) | Revenue improvement | ~10% |
| MMM (Axtria) | Hyperparameter tuning improvement | 15% accuracy improvement |
| Fraud Embedding (Chubb) | Precision@5 improvement vs. ada-002 | +18% |
| Fraud RAGAS (Chubb) | Faithfulness score target | >0.85 (achieved 0.88) |
| Fraud RAGAS (Chubb) | Context recall | 0.71 (challenge area) |

---

## CHEAT SHEET 2: FORMULAS TO DERIVE ON THE SPOT

```
1. Logistic Regression Gradient:
   ∂L/∂w = (1/N) × X^T × (ŷ - y)
   
2. XGBoost Leaf Score:
   w* = -Σgi / (Σhi + λ)
   
3. XGBoost Split Gain:
   Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
   
4. Attention:
   Attention(Q,K,V) = softmax(QK^T/√d_k)V
   
5. Cross-Entropy Loss:
   L = -(1/N) Σ [y_i log(ŷ_i) + (1-y_i)log(1-ŷ_i)]
   
6. Softmax Gradient:
   ∂L/∂z_i = ŷ_i - y_i
   
7. KL Divergence:
   KL(P||Q) = Σ P(x) log(P(x)/Q(x))
   
8. F-beta:
   F_β = (1+β²)PR / (β²P + R)
   
9. Precision-Recall tradeoff marker:
   Precision = TP/(TP+FP), Recall = TP/(TP+FN)
   
10. Contrastive Loss:
    L = -log exp(sim(z_i,z_j)/τ) / Σ_{k≠i} exp(sim(z_i,z_k)/τ)
    
11. Bayes' Theorem:
    P(A|B) = P(B|A)P(A) / P(B)
    
12. Cox PH Hazard:
    h(t|x) = h_0(t) exp(β^T x)
    
13. PSI:
    PSI = Σ (P_new - P_base) × ln(P_new/P_base)
    — < 0.10: stable | 0.10-0.25: monitor | >0.25: retrain
    
14. A/B Sample Size:
    n = (z_α + z_β)² × 2p(1-p) / δ²
```

---

## CHEAT SHEET 3: SYSTEM DESIGN ONE-LINERS

| "Design a ___" | First 3 things to say |
|---|---|
| Fraud detection system | Clarify: real-time vs. batch latency requirement. Two paths: <100ms (features store + GBM) + async (RAG + GNN). Feature store: Redis online, BigQuery offline. |
| Credit risk model | Cold start: behavioral e-commerce signals as proxy. Warm: bureau + behavioral hybrid. Model: LightGBM scorecard. Monitoring: vintage analysis. |
| Recommendation system | Two-tower neural model for retrieval + cross-encoder for re-ranking. Cold start: popularity + exploration. Evaluation: NDCG@k offline, CTR online. |
| Return fraud system | Multi-modal: text NLP (return reason) + CV (product images) + behavioral (return velocity) + graph (linked accounts). Cascade: rules → ML → human. |
| LLM seller insights | RAG over seller performance data. Structured output schema. Hallucination guards. Feedback: investigator corrections → KB update. |

---

## CHEAT SHEET 4: MODEL SELECTION DECISION TREE

```
DATA TYPE?
├── Tabular data → XGBoost/LightGBM (first choice for fraud, CLV)
│   ├── <1M rows → XGBoost with full CV
│   └── >1M rows → LightGBM (faster, lower memory)
│
├── Text data → BERT family
│   ├── Classification → BERT + classification head
│   ├── Generation → GPT / T5 / LLaMA
│   └── Embeddings → Sentence-BERT / E5 / domain fine-tuned
│
├── Sequential data → 
│   ├── Short sequences → Transformer
│   └── Very long sequences → State Space Models (Mamba)
│
├── Graph data → GNN family
│   ├── Node classification → GraphSAGE / GAT
│   └── Link prediction → Node2Vec + MLP
│
└── Mixed (tabular + text) → Hybrid approach
    ├── Extract text embeddings → treat as features → XGBoost
    └── Multi-modal transformer (more complex, needs more data)

ALWAYS CONSIDER:
├── Interpretability required? → Tree models + SHAP | NOT deep NN end-to-end
├── Calibration required? → Add Platt/isotonic calibration
├── Latency <100ms? → Avoid deep NN, use GBM + feature store
└── Cold start? → Explicit handling required — don't assume historical data exists
```

---

## CHEAT SHEET 5: EVALUATION METRICS SELECTION

| Problem | Primary Metric | Why | Business Connection |
|---|---|---|---|
| Fraud detection (1% fraud) | PR-AUC | ROC-AUC misleading with class imbalance | "What % precision can investigators tolerate?" |
| Credit scoring | KS Statistic + Gini | Industry standard, regulatory reporting | "How well does model separate goods from bads?" |
| Readmission risk | AUC-ROC + Brier score | Rank ordering + calibration both matter | "Physician acts on probability, not just rank" |
| Recommendation | NDCG@k | Position-weighted relevance | "Top-1 recommendation matters more than top-5" |
| NER/Extraction | F1 per entity type | Precision AND recall matter for IE | "Missing an amount = bad data; hallucinating one = worse" |
| CLV prediction | Spearman rank + Decile Lift | Ranking for targeting, not absolute value | "Who do we call first?" |
| A/B test | Primary KPI + Guardrail metric | Don't optimize one at expense of other | "Fraud_recall must improve without hurting good user FPR" |

---

## CHEAT SHEET 6: PRODUCTION TERMS TO USE NATURALLY

| Term | One-Line Definition | When to Use |
|---|---|---|
| **Champion-Challenger** | Champion model (90% traffic) vs. challenger (10%) for continuous evaluation | Any model serving question |
| **Shadow mode** | New model runs but doesn't influence decisions — outputs logged only | First deployment question |
| **Feature store** | Unified online (Redis) + offline (BigQuery) feature computation layer | Any feature engineering design |
| **PSI** | Population Stability Index — metric for input distribution drift | Monitoring question |
| **Point-in-time correctness** | Join features using only info available at that historical moment — prevents leakage | Feature engineering, training data |
| **Training-serving skew** | Features computed differently at train vs. serve time → model degrades | Any performance-in-production discussion |
| **Calibration** | Model's predicted probability should match actual event rate at that probability | Always when discussing probabilities |
| **RAGAS** | Framework for evaluating RAG systems: faithfulness, relevance, recall, precision | Any RAG system question |
| **Hard negatives** | Training examples that are "close" to positives but are actually negatives | Embedding model training |
| **Contrastive loss** | Loss that brings similar embeddings together and pushes dissimilar apart | Embedding model question |
| **In-batch negatives** | Using other samples in the same training batch as negatives — scalable contrastive learning | SimCLR, your fraud embedding |
| **Temperature (τ)** | Controls sharpness of contrastive loss distribution — lower = harder training | Contrastive learning |
| **HNSW** | Hierarchical Navigable Small World — ANN algorithm in FAISS/vector DBs | Vector DB scaling question |
| **vLLM** | Framework for LLM serving with paged attention for high throughput | LLM inference question |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning using rank-r decomposition | Fine-tuning question |

---

## CHEAT SHEET 7: 60-SECOND ANSWERS TO COMMON OPENERS

**"Tell me about yourself" (60 seconds):**
> "9+ years of end-to-end ML across healthcare, insurance, and pharma. Currently at Chubb architecting fraud detection systems using RAG + LLMs — recognized with multiple awards. Before that, Axtria (MMM + GenAI pharma) and EXL/Aetna (healthcare ML at scale). My edge: I've built agentic AI systems and RAG pipelines in production, not just POCs. I'm here because Flipkart's fraud + risk scale is the next hard problem I want to solve."

**"Why Flipkart?" (45 seconds):**
> "The scale is genuinely different — 350M users, 10M daily transactions, adversarial fraud at e-commerce speed. My RAG + agentic fraud work gave me the methodology. I want to feel what changes at that scale. Also: Flipkart's AI philosophy — Triksha for LLM security, human-in-the-loop, responsible AI deployment — aligns with how I think about production AI. This isn't a pivot; it's an acceleration."

**"Walk me through your fraud system" (90-second opener):**
> "Business problem: Insurance fraud in long-tail claims was going undetected for 30+ days. Traditional rules caught 30% at 45% precision. I built a 4-layer system: BERT-based information extraction → domain fine-tuned embedding model → vector store for RAG retrieval → GPT-4 structured risk scoring. Dual-speed serving: real-time Kafka pipeline for new claims, batch for historical backfill. Shadow-mode validated for 60 days before production — achieved 78% recall at 22% FPR. Happy to go deep on any layer."

**"What's your biggest technical challenge?" (30 seconds):**
> "Evaluation without ground truth — fraud labels arrive 60+ days after the claim. I solved this with shadow-mode validation: compare system flags to investigator verdicts from 60 days prior. It's not perfect, but it's rigorous and mirrors what happens in production deployment."

---

*This is your "interview day morning" review. Internalize these patterns and numbers. You've got this.*
