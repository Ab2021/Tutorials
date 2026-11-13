# Module 18 — Deep Learning, Foundation Models, LLM Agents

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 34 hours (13 lecture, 12 lab, 9 project)
> **Track fit:** AI & Data Science core; required for the Applied AI track
> **Prerequisites:** Module 05 (forecasting baselines); Module 08 (hierarchical forecasting); Module 04 (evaluation and leakage); Python and PyTorch basics
> **Tools:** Python 3.12, PyTorch, NumPy, pandas, scikit-learn; an inference endpoint for a hosted foundation model; a vector store (pgvector, FAISS, or a managed equivalent)
> **Industry question:** "The vendor says their foundation model beats our tuned model zero-shot. What would you have to see to believe it, and what would it cost to be wrong?"

---

## 1. Why industry cares

Two things happened to supply-chain analytics between 2020 and 2026. Deep sequence models stopped being research artefacts and became the default architecture in commercial forecasting platforms. And large language models turned unstructured supply-chain text — contracts, supplier emails, deviation reports, audit findings, shelf photographs — into something a system can read.

The first change is an accuracy story with a governance problem. Neural forecasters can beat classical baselines on many series, but the gain is conditional: it depends on hierarchy, on series length, on how many related series exist to learn from, and on whether the evaluation was done honestly. The published gains are real and the published comparisons are frequently not comparable, because benchmarks leak into training data and because "zero-shot" is often "zero-shot on a dataset the model's authors had already seen".

The second change is a reliability story with an accuracy problem. An LLM that extracts payment terms from a contract at 94% accuracy is not 94% useful — it is useful only if the 6% failure is detectable, because a wrong payment term propagates into cash-flow forecasts and a wrong incoterm propagates into landed cost. The engineering problem is not the model; it is the verification layer around it.

The characteristic failure in 2026 is not choosing the wrong architecture. It is **shipping a system whose errors are invisible**. A classical forecast that is 12% wrong is visibly wrong on the accuracy report. An agent that confidently cites a clause that does not exist produces a fluent, plausible, wrong answer that passes every review that does not check the source. Everything in this module that is not about modelling is about making failure visible.

## 2. Learning outcomes

By the end of this module, you can:

- Explain the mechanics of a feed-forward network, an RNN, an LSTM, a temporal convolution, and a Transformer attention block, and say what each was designed to fix.
- Choose between a global and a local model for a given hierarchy, and justify the choice with the number of related series and their length.
- Place DeepAR, N-BEATS, N-HiTS, TFT, and PatchTST in the architecture space and state the inductive bias each contributes.
- Evaluate a time-series foundation model zero-shot, and design the fine-tuning comparison that would justify adopting it.
- Audit a benchmark for contamination and leakage, and state what a reported zero-shot result actually demonstrates.
- Build a retrieval-augmented generation pipeline over contracts and policies with citation verification.
- Specify an agent with tools, memory, planning, guardrails, and a human approval gate, and test each of those components.
- Evaluate a multimodal model on shelf images or damage detection against a defined operational metric.
- Compute the total cost of ownership for a build-versus-buy decision including inference, evaluation, and maintenance.
- Write an evaluation harness that reports accuracy, grounding, hallucination rate, latency, and cost per decision.

## 3. Mental model

### 3.1 Where the models sit

```
                         decision layer
                    (buy, build, ship, price)
                              ▲
              ┌───────────────┴───────────────┐
        structured signals              unstructured signals
        (demand, inventory,              (contracts, emails,
         prices, sensors)                 images, reports)
              ▲                                ▲
   ┌──────────┴──────────┐          ┌──────────┴──────────┐
   local statistical     global deep   classical NLP   LLM + RAG
   models (ETS, ARIMA)   models        (regex, BERT)   (extract, cite)
                         (DeepAR, TFT,
                          PatchTST, TSFM)
```

The two branches meet at the decision layer and share one requirement: the output must carry an uncertainty statement and a provenance trail. A point forecast with no interval and an extracted clause with no citation are the same failure in different modalities.

### 3.2 The three questions before any model choice

| Question | If yes | If no |
|---|---|---|
| Are there many related series with shared structure? | Global model is worth trying | Local model or simple baseline |
| Is the evaluation set provably unseen by the model? | Zero-shot claim is meaningful | Claim is a benchmark artefact |
| Can a wrong output be detected cheaply? | Automate with sampling audit | Require human review or a verification step |

The third question is the one that decides whether a system can be deployed at all, and it is asked last in almost every project.

### 3.3 The compounding-error principle

An LLM extraction step at 95% per-field accuracy over 12 fields yields `0.95^12 = 0.54` fully-correct records. An agentic workflow with five sequential tool calls at 95% each succeeds `0.95^5 = 0.77` of the time. Reliability is multiplicative, so the design question is never "what is the accuracy of the model" but "how many independent chances does the pipeline give it to be wrong, and what catches the failure".

## 4. Core concepts

### 4.1 Neural network fundamentals

A feed-forward layer computes `h = σ(Wx + b)`. Stacking layers composes nonlinear transformations, and training minimises a loss by gradient descent through backpropagation.

| Component | Purpose | Supply-chain relevance |
|---|---|---|
| Activation (ReLU, GELU) | Nonlinearity | Without it, stacked layers collapse to one linear map |
| Loss function | Defines what "good" means | Quantile loss for inventory; MAE for a median plan; asymmetric loss for service |
| Regularisation (dropout, weight decay) | Prevents memorisation | Critical with few series and short history |
| Normalisation (batch, layer) | Stabilises training | Series with different scales |
| Embeddings | Learned categorical representations | SKU, store, product hierarchy |

**The loss function is the business decision.** Training on MSE produces a mean forecast, which is correct for a symmetric cost and wrong for inventory, where the cost of being short exceeds the cost of being long. Quantile loss trains the model to produce the quantile the inventory policy actually needs, and it is the single highest-leverage modelling choice in this module.

### 4.2 Sequence architectures

| Architecture | Mechanism | Strength | Limitation |
|---|---|---|---|
| RNN | Hidden state carried forward | Arbitrary length in principle | Vanishing gradients; sequential, so slow |
| LSTM / GRU | Gated memory cells | Captures longer dependencies | Still sequential; struggles beyond a few hundred steps |
| TCN | Dilated causal convolutions | Parallel training, long receptive field | Fixed receptive field; memory grows with dilation |
| Transformer | Self-attention over all positions | Global context, parallel, scales | Quadratic in sequence length; needs data |

Self-attention computes, for each position, a weighted sum over all positions where the weights come from query-key similarity:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

For forecasting, the attention must be **causal** — position `t` may attend only to positions `≤ t` — or the model reads the future and reports an excellent loss that means nothing.

**PatchTST** applies the idea that made vision Transformers work: split the series into patches (subsequences), treat each patch as a token, and attend across patches. This shortens the effective sequence, reduces compute, and retains a long history — the architectural insight is that neighbouring time points are redundant, so tokenising them individually wastes capacity.

### 4.3 Global versus local models

| Property | Local model | Global model |
|---|---|---|
| Fit on | One series | All series in a panel |
| Learns | That series' dynamics | Shared patterns plus covariates |
| Data need | 2–3 seasonal cycles | Many series, even short ones |
| Cold start | Cannot forecast a new item | Forecasts a new item from covariates |
| Interpretability | High | Lower, but attention and feature importance help |
| Failure mode | Overfits short history | Averages away genuinely idiosyncratic behaviour |

The practical rule: **global models win when series are numerous, short, and related; local models win when series are few, long, and structurally different.** A retailer with 50,000 SKUs across 800 stores is the archetypal global case. A chemical plant forecasting six product lines with 15 years of history each is the archetypal local case, and a neural global model there is usually a way to spend three months to lose to exponential smoothing.

The strongest modern result in most public benchmarks is not "deep beats classical" but "a global model with good covariates beats a local model without them, and a well-tuned local model with good covariates is competitive". That is a statement about information, not architecture.

### 4.4 The neural forecasting zoo

| Model | Architecture | Inductive bias | When it earns its place |
|---|---|---|---|
| DeepAR | Autoregressive RNN with negative-binomial or Student-t likelihood | Probabilistic, handles count and intermittent data | Many related series, need predictive intervals |
| N-BEATS | Stacked fully connected blocks with basis expansion | No recurrence or attention; pure MLP with trend/seasonality bases | Strong univariate accuracy, simple to train |
| N-HiTS | Hierarchical multi-rate sampling plus multi-scale synthesis | Different frequencies for different horizons | Long horizons with multi-scale structure |
| TFT | LSTM encoder-decoder with variable selection and interpretable attention | Explicit feature selection and temporal importance | Covariate-rich forecasting with an explainability requirement |
| PatchTST | Patch-tokenised Transformer, channel-independent | Local semantic patches, long lookback | Long history, many series, efficient compute |

Two design distinctions matter more than the names. **Channel-independent** models treat each series separately and share weights; **channel-mixing** models attend across series at each time step. Channel-independent avoids spurious cross-series correlation but cannot exploit genuine cross-learning at inference. **Direct** models predict the whole horizon at once; **recursive** models predict one step and feed it back, so errors compound.

### 4.5 Time-series foundation models

A time-series foundation model (TSFM) is pretrained on a large corpus of time series and applied zero-shot, or lightly fine-tuned, to a new series.

| Model | Developer | Approach | Notes as of 2026 |
|---|---|---|---|
| Chronos | Amazon | Tokenises scaled values into a fixed vocabulary; a language-model-style architecture over tokens | Probabilistic; multiple sizes |
| TimesFM | Google | Decoder-only patched Transformer trained on a large time-series corpus | Designed for zero-shot point forecasting |
| Moirai | Salesforce | Any-variate attention; handles multivariate series with varying dimensions | Pretrained across multiple domains and frequencies |
| TTM (Tiny Time Mixers) | IBM | Small mixer architecture; light enough for edge or high-volume inference | Cost-oriented, fewer parameters |

The 2025–2026 evidence is more nuanced than the launch claims. TSFM-Bench (arXiv:2410.11802) evaluates foundation models across forecasting tasks and finds them competitive on many, but not uniformly superior to tuned task-specific models, with performance depending strongly on domain similarity between pretraining and target. A 2025 audit of time-series benchmark contamination (arXiv:2510.13654) found evidence that reported zero-shot results can be inflated when evaluation windows overlap pretraining corpora `[VERIFY: exact scope and magnitude of the contamination finding before quoting numbers]`.

| Claim | What it means | What to ask |
|---|---|---|
| "Zero-shot beats your model" | On their benchmark, with their preprocessing | Was the test window in pretraining data? Same horizon? Same covariates? |
| "Fine-tuning adds 8%" | Their fine-tune versus their zero-shot | Versus your tuned baseline, on your data? |
| "Handles any frequency" | Trained on many frequencies | At your frequency, with your intermittency? |
| "Cheaper than training" | No training cost | Inference cost at your volume? |

### 4.6 Evaluating zero-shot and fine-tuned models

| Dimension | Zero-shot question | Fine-tuned question |
|---|---|---|
| Data | Was the target series in pretraining? | Is the fine-tune set disjoint from the test set? |
| Horizon | Same horizon as the benchmark? | Same horizon as deployment? |
| Covariates | Does it accept yours? | Were they used in fine-tuning? |
| Baseline | Which classical baseline? | Same baseline, tuned? |
| Metric | Aggregate or per-series? | Weighted by business volume? |
| Uncertainty | Are intervals calibrated? | Are they calibrated after fine-tuning? |

The minimum defensible comparison is: same splits, same horizon, same covariates, at least two strong baselines (a seasonal naive and a tuned gradient-boosted or ETS model), and a per-series breakdown, not a single aggregate number. An aggregate win driven by a handful of series is a loss in practice.

**Calibration** is the metric most often skipped. A model that reports an 80% interval covering the truth 60% of the time will systematically under-stock. Measure empirical coverage by quantile, not just the point error.

### 4.7 Benchmark contamination and leakage

| Leakage type | Mechanism | Detection |
|---|---|---|
| Temporal leakage | Test period precedes or overlaps training | Audit split dates against pretraining corpus dates |
| Series leakage | Same series in pretraining and test | Series fingerprinting; hash matching on values |
| Label leakage | Future covariates used as inputs | Feature-timing audit |
| Preprocessing leakage | Scalers fit on the full series including test | Refit scalers inside the split |
| Hyperparameter leakage | Tuning on the test set | Nested validation |
| Benchmark overfitting | Model selected because it scores well on the benchmark | Held-out private evaluation set |

The practical defence is a **private holdout that has never been published** and a **post-cutoff window**: evaluate only on data generated after the model's training cutoff. If a vendor cannot state its cutoff, the zero-shot claim is unfalsifiable, which is not the same as true.

### 4.8 LLMs for extraction and supplier communication

| Task | Output | Failure that matters | Control |
|---|---|---|---|
| Contract extraction | Parties, term, incoterm, payment terms, penalties, renewal | A missed auto-renewal clause | Span citation; clause-level schema; human review of low-confidence fields |
| Invoice and packing-list extraction | Line items, quantities, amounts | Silent quantity mismatch | Three-way match against PO and receipt |
| Supplier email triage | Intent, urgency, affected SKUs | Missed supply disruption | Escalation rule for disruption intents |
| Supplier communication drafting | Email or negotiation position | Committing to terms outside authority | Template constraints; approval gate; no autonomous send |
| Deviation and audit reports | Structured findings | Fabricated finding or citation | Retrieval grounding; citation verification |
| Planning copilot | Explanation and options | Confident wrong recommendation | Tool outputs are the only source of numbers |

**Structured extraction is the highest-value LLM application in supply chain**, because the output is verifiable: a schema plus a source span can be checked mechanically. Free-text generation is the lowest-value, because the output is not verifiable without a human reading it.

The design rule: **the model may propose, but only a deterministic check may confirm.** Every extracted field carries a character span into the source document, and a validator confirms the span exists and contains the claimed value type. A field that fails the check is routed to a human, not silently accepted.

### 4.9 RAG over contracts and policies

Retrieval-augmented generation (Lewis et al., 2020) conditions a language model's answer on retrieved documents, so the answer can be grounded in a specific source rather than recalled from parameters.

```
query → embed → retrieve top-k chunks → rerank → prompt with chunks → answer + citations
                                                        │
                                          verify each citation against the chunk
```

| Design choice | Options | Failure if wrong |
|---|---|---|
| Chunking | Fixed tokens, semantic, clause-level | Clause split across chunks; retrieval returns half a rule |
| Embedding | General, domain-tuned | Misses domain vocabulary ("incoterm", "demurrage") |
| Retrieval | Dense, sparse (BM25), hybrid | Dense misses exact identifiers; sparse misses paraphrase |
| Reranking | Cross-encoder, LLM | Poor precision at `k`; context window filled with near-misses |
| Prompt | Answer-only, answer-with-citations, refusal | Model answers from memory when retrieval fails |
| Verification | Span match, entailment check | Citation exists but does not support the claim |

**Hybrid retrieval is the default in supply chain**, because queries mix natural language with exact identifiers — a PO number, a SKU code, a regulation reference — and dense embeddings are poor at exact matching while BM25 is poor at paraphrase.

**Abstention must be a designed behaviour.** The prompt must permit "the retrieved documents do not answer this", and the evaluation must score abstention as correct when the answer is genuinely absent. A system that never abstains has a hallucination rate of 100% on unanswerable questions.

### 4.10 Agents: tools, planning, memory, guardrails, human approval

An agent is a language model in a loop with tools, state, and a stopping condition.

| Component | Function | Design requirement |
|---|---|---|
| Tools | Deterministic functions: query the ERP, compute a plan, send an email | Typed schema; validation of arguments; least privilege |
| Planning | Decompose the goal into steps | Bounded step count; explicit plan object |
| Memory | Short-term context and long-term store | Scoped, expiring, auditable |
| Guardrails | Limits on actions and outputs | Allow-list of tools; spend and volume limits |
| Human approval | Gate before irreversible actions | Explicit approval object; timeout to safe default |
| Observability | Trace of every step | Full transcript with tool inputs and outputs |

**Irreversible actions require a human gate.** Sending a supplier email, releasing a purchase order, changing a price, or cancelling a shipment are all irreversible in the sense that the counterparty has acted. The agent may prepare the action; a human approves it; the approval is logged with the identity of the approver.

**Tool calls are the only source of numbers.** If the model states a demand figure, it must come from a tool output in the trace. This is checkable automatically and eliminates the most common class of plausible-but-wrong agent output.

| Guardrail | Test | Failure signal |
|---|---|---|
| Tool allow-list | Attempt a disallowed tool | Agent executes it |
| Argument validation | Pass a malformed SKU | Tool called with garbage |
| Spend limit | Request an action above the cap | Action proceeds |
| Approval gate | Request an irreversible action | Action executes without approval |
| Loop bound | Provide an unsolvable goal | Agent runs past the step limit |
| Injection resistance | Embed instructions in a retrieved document | Agent follows them |

**Prompt injection is the agent-specific threat**: a retrieved document or a supplier email can contain text instructing the agent to take an action. Treat all retrieved content as data, never as instructions, and test it explicitly with an adversarial document in the evaluation set.

### 4.11 Multimodal: shelf images, damage, video

| Task | Input | Output | Operational metric |
|---|---|---|---|
| Planogram compliance | Shelf photograph | Facing counts, out-of-stock, position | Compliance rate per store visit |
| Damage detection | Photo of carton or product | Damaged / not, severity, location | Precision at the damage class; false-alarm cost |
| Quality inspection | Line camera | Defect / pass | Escape rate and false-reject rate |
| Loading and unloading | Video | Counts, sequence, exceptions | Count accuracy; exception detection |
| Safety compliance | Video | PPE and zone violations | Recall on violations; alert fatigue rate |

Two rules apply to every one of these. **The operational metric is not the model metric.** A 96% accurate damage classifier that fires on 15% of undamaged cartons creates a manual review queue that costs more than the damage it prevents. Report precision at the operating threshold that the process can absorb, not at the threshold that maximises F1.

**Domain shift is the norm.** A model trained on one store's lighting, camera angle, and shelf fixtures degrades on another's. Validate on held-out stores, not held-out images from the same stores, and monitor for drift when fixtures or cameras change.

### 4.12 Evaluation: accuracy, grounding, hallucination, latency, cost

| Dimension | Metric | Target setting |
|---|---|---|
| Task accuracy | Exact match, F1, or field-level accuracy | Per field, not per document |
| Retrieval quality | Recall@k, MRR, nDCG | Recall@5 for the clause that answers the question |
| Grounding | Share of claims with a verified supporting span | 100% for factual claims |
| Hallucination | Unsupported-claim rate on a golden set | Near zero for regulated content; measured, not asserted |
| Abstention | Correct refusal on unanswerable questions | Scored as a first-class metric |
| Latency | p50 and p95 end-to-end | p95 against the process cadence |
| Cost | Cost per decision, cost per 1,000 documents | Include retries and human review |
| Robustness | Performance under injection, paraphrase, and truncation | Adversarial set |

A golden set of 100–300 examples, hand-labelled by a domain expert, is worth more than 10,000 unlabelled documents. Build it before the system, and freeze it — an evaluation set that is edited after seeing results is not an evaluation set.

### 4.13 Build versus buy and TCO

| Cost element | Build | Buy |
|---|---|---|
| Licence or subscription | None | Per-seat or per-call |
| Inference | Direct, variable with volume | Bundled or metered |
| Engineering | Model, pipeline, evaluation, on-call | Integration and configuration |
| Evaluation and red-teaming | Entirely yours | Partly vendor, but you still need your own |
| Maintenance | Model drift, prompt drift, dependency upgrades | Vendor releases; migration risk |
| Compliance | Data residency, audit, DPIA | Contract-dependent; vendor audit reports |
| Exit cost | Low, you own the pipeline | High if prompts and data live in the vendor |

**TCO for an LLM system is dominated by inference at volume and by human review at low accuracy.** A useful formula:

```
annual cost = (documents × cost per document) + (hours of review × loaded rate)
            + engineering FTE + platform fixed cost
```

**Worked example.** 500,000 contract-clause extractions per year. Model cost $0.004 per extraction; 8% of outputs need human review at 3 minutes and a $45/hour loaded rate; engineering 1.5 FTE at $180,000 fully loaded; platform $60,000.

```
Inference   = 500,000 × 0.004                     = $2,000
Review      = 500,000 × 0.08 × 0.05 h × $45       = $90,000
Engineering = 1.5 × 180,000                        = $270,000
Platform                                           = $60,000
Annual total                                       = $422,000
```

Inference is 0.5% of the cost. The engineering and the review queue are 85%. Any business case built on API pricing alone is describing a rounding error and ignoring the actual expense — which is why the accuracy improvement that halves the review rate is worth far more than a 30% cheaper model.

### 4.14 A runnable RAG harness with citation checks

The harness below is deliberately dependency-free so it runs anywhere, and it demonstrates the two checks that matter: retrieval recall and citation grounding, including a negative control that must fail.

```python
"""Offline RAG evaluation: TF-IDF retrieval, citation grounding, recall@k.
Run: python rag_eval.py    (standard library only)
Replace DOCS and GOLDEN with your own corpus and a hand-labelled golden set."""
import math
import re
from collections import Counter

DOCS = {
    "S1": "Supplier shall deliver within 30 days of purchase order acknowledgement. "
          "Late delivery incurs a service credit of 1% of order value per week.",
    "S2": "Buyer may terminate for convenience on 90 days written notice. "
          "Termination does not relieve either party of accrued payment obligations.",
    "S3": "Supplier shall maintain product liability insurance of not less than "
          "5 million USD and provide certificates annually.",
    "S4": "Force majeure excludes supplier financial distress and subcontractor failure. "
          "Notice must be given within 10 business days.",
}

# (question, gold document, the answer span a correct system must cite)
GOLDEN = [
    ("How much notice is required to terminate for convenience?", "S2", "90 days written notice"),
    ("What is the late delivery penalty?", "S1", "service credit of 1% of order value per week"),
    ("What insurance limit applies?", "S3", "5 million USD"),
]

TOKEN = re.compile(r"[a-z0-9]+")

def toks(text):
    return TOKEN.findall(text.lower())

def normalise(text):
    return " ".join(toks(text))

def build_index(docs):
    df = Counter()
    for d in docs.values():
        df.update(set(toks(d)))
    n = len(docs)
    idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
    vecs = {}
    for key, d in docs.items():
        tf = Counter(toks(d))
        v = {w: (1 + math.log(c)) * idf[w] for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs[key] = {w: x / norm for w, x in v.items()}
    return idf, vecs

IDF, VECS = build_index(DOCS)

def retrieve(query, k=2):
    tf = Counter(toks(query))
    q = {w: (1 + math.log(c)) * IDF.get(w, 0.0) for w, c in tf.items()}
    norm = math.sqrt(sum(x * x for x in q.values())) or 1.0
    q = {w: x / norm for w, x in q.items()}
    scores = {d: sum(q.get(w, 0.0) * v.get(w, 0.0) for w in q) for d, v in VECS.items()}
    return sorted(scores.items(), key=lambda kv: -kv[1])[:k]

def cited_span_supported(span, doc_key):
    """Grounding check: the quoted span must actually occur in the cited document."""
    return normalise(span) in normalise(DOCS[doc_key])

hits = 0
for question, gold_doc, span in GOLDEN:
    ranked = retrieve(question, k=2)
    keys = [k for k, _ in ranked]
    hits += gold_doc in keys
    grounded = cited_span_supported(span, keys[0]) if keys else False
    print(f"{question[:50]:52s} -> {keys[0]} ({ranked[0][1]:.2f}) "
          f"retrieved_ok={gold_doc == keys[0]} grounded={grounded}")

print(f"\nrecall@2 = {hits / len(GOLDEN):.2f}   n = {len(GOLDEN)}")
print("negative control (must be False):",
      cited_span_supported("Supplier shall provide a 99-year warranty.", "S1"))
```

The negative control matters as much as the positive cases. A grounding check that returns `True` for a fabricated clause is not a grounding check, and the only way to know is to feed it a fabrication and assert that it fails.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Series panel | Series × timestamp | Planning system | Consistent hierarchy; no mixed time zones |
| Covariates | Series × timestamp | ERP, weather, calendar | Known at forecast time; future values available |
| Forecast and actual | Series × timestamp × version | Planning system | Versioned; retain every published forecast |
| Document corpus | Document × chunk | Contract repository, policy library | Version, effective date, access control |
| Golden set | Question × gold answer × source span | Human labelling | Frozen before evaluation; expert-labelled |
| Model registry entry | Model × version | MLOps | Training data hash, cutoff date, hyperparameters |
| Inference log | Request × timestamp | Application | Prompt version, retrieved doc IDs, latency, cost |

**Standards and frameworks.** ISO/IEC 42001 (AI management systems) for governance; NIST AI Risk Management Framework for risk identification and testing; ISO/IEC 23894 for AI risk management guidance; the EU AI Act (Regulation 2024/1689) for obligations by risk class; ISO/IEC 5259 for data quality in analytics and ML. For model cards and documentation, the Model Cards and Datasheets for Datasets conventions are the de facto standard.

**Workflow.** Version every model and every prompt, and log the retrieved document IDs with each answer so a wrong answer can be reconstructed. Freeze the golden set. Report per-series and per-field results, never a single aggregate. Record the training cutoff date of every foundation model you use, because that date is what makes a zero-shot claim testable.

## 6. Industry case studies

### 6.1 Amazon Chronos and the zero-shot tokenisation approach (2024–2026)

**Context.** Time-series forecasting had produced many specialised architectures, each tuned to a domain, with no shared pretrained backbone comparable to what vision and language had.

**Intervention.** Amazon's Chronos treats scaled and quantised time-series values as tokens in a vocabulary and trains a language-model-style architecture over them, so forecasting becomes sequence-to-sequence prediction over tokens, and the model can be applied to a new series without gradient updates.

**Evidence.** The Chronos paper reports competitive zero-shot results across public forecasting benchmarks and multiple model sizes. These are author-reported benchmark results; the later contamination audit (arXiv:2510.13654) raises the general question of how much of such zero-shot performance reflects genuinely unseen data `[VERIFY: whether Chronos specifically was in scope].`

**Outcome.** Chronos made the "forecasting as language modelling" framing concrete and pushed several planning vendors to add zero-shot TSFM options to their platforms through 2025–2026. Vendor adoption is a market signal, not evidence of superiority.

**What transferred.** The abstraction: quantise and tokenise, and a general sequence model can consume series without a bespoke architecture. **What did not.** The claim that domain-specific preprocessing is obsolete. Scaling, seasonality handling, and covariate design still determine most of the accuracy on a real hierarchy.

### 6.2 Google TimesFM and decoder-only patched forecasting (2024–2026)

**Context.** Most neural forecasters were encoder-decoder or recurrent; decoder-only Transformers had become dominant in language modelling and suggested a simpler path.

**Intervention.** TimesFM uses a decoder-only Transformer over patches of the input series, pretrained on a large corpus of real and synthetic time series, and marketed for zero-shot point forecasting.

**Evidence.** The model paper reports zero-shot results competitive with supervised models on several benchmarks. Independent replication on domain-specific data is thinner, and results vary by whether covariates are available — which they often are in supply chain and usually are not in the benchmarks.

**Outcome.** TimesFM became one of the standard open TSFM baselines, and its patched decoder design influenced subsequent models.

**What did not transfer cleanly.** Point-forecast zero-shot performance to a production inventory system, which needs calibrated quantiles and covariates. The gap between "good zero-shot point forecast" and "usable inventory policy input" is where most TSFM pilots stall.

### 6.3 Salesforce Moirai and any-variate attention (2024–2026)

**Context.** Multivariate forecasting models typically require a fixed set of channels, so a model trained on one dataset cannot accept a different number of related series at inference.

**Intervention.** Moirai introduced any-variate attention, allowing the model to attend across a variable number of related series, and pretrained across multiple domains and frequencies so that one model can be applied to panels of differing width.

**Evidence.** Author-reported results across public benchmarks and a universality claim across frequencies and domains. As with the other TSFMs, the comparison is against benchmarks rather than against a customer's tuned production baseline.

**Outcome.** The any-variate design became a common reference point for handling heterogeneous panels, and Moirai is widely used as an open baseline.

**What transferred.** The idea that a foundation model must handle a variable-width context, which is the normal situation in supply chain — a SKU has different numbers of related SKUs in different categories. **What did not.** The assumption that relatedness in the panel is genuine. Attending across series that share an attribute but not a demand driver adds noise, and channel-independent models sometimes win for that reason.

### 6.4 Air Canada and chatbot liability (2024)

**Context.** Air Canada deployed a website chatbot that answered customer questions about fares and policies.

**Intervention.** The chatbot told a customer that bereavement fares could be claimed retroactively after travel, contrary to the airline's actual policy. The customer relied on the statement, was refused, and brought a claim to the British Columbia Civil Resolution Tribunal.

**Evidence.** *Moffatt v. Air Canada*, 2024 BCCRT 149. The tribunal held the airline responsible for the chatbot's statements, rejecting the argument that the chatbot was a separate entity.

**Outcome.** Air Canada was ordered to pay damages. The decision is a small monetary case with a large design consequence: the organisation is liable for what its automated agent says, and "the chatbot said it" is not a defence.

**What transferred.** The governance requirement: a customer-facing or supplier-facing agent needs grounding, a refusal behaviour, and a path to a human, because the firm owns the output. **What did not.** The inference that chatbots are too risky to deploy. The failure was an ungrounded policy answer with no retrieval and no verification, which is a design defect, not an inherent property of conversational interfaces.

## 7. Labs

### Lab 18.1 — Global versus local forecasting

**Objective.** Quantify when a global neural model beats a tuned local baseline.

**Data.** A panel of at least 500 related series with at least three seasonal cycles, plus covariates.

**Steps.** (1) Establish seasonal-naive and tuned ETS baselines per series. (2) Train a global gradient-boosted model and a global neural model (DeepAR or PatchTST). (3) Evaluate MASE per series and weighted by volume. (4) Slice results by series length and volume decile. (5) Identify which series the global model loses on and why. (6) State a routing rule: which series get which model.

**Deliverable.** A per-series comparison, a slice analysis, and a routing rule.

**Acceptance criteria.** Two baselines; per-series not aggregate reporting; slices by length and volume; loss cases explained. **Stretch.** Add hierarchical reconciliation and measure the effect on the aggregate.

### Lab 18.2 — Zero-shot TSFM evaluation with a contamination audit

**Objective.** Test a TSFM claim the way a sceptical reviewer would.

**Data.** A TSFM endpoint or open weights, a private holdout series, and a published benchmark series.

**Steps.** (1) Record the model's training cutoff. (2) Evaluate on the private holdout and on the benchmark. (3) Restrict the private evaluation to a post-cutoff window. (4) Compare with your tuned baseline on identical splits. (5) Check interval calibration by quantile. (6) Report the gap between benchmark and private performance.

**Deliverable.** An evaluation memo with the cutoff, both results, and calibration.

**Acceptance criteria.** Cutoff documented; post-cutoff window used; identical splits; calibration measured; benchmark-versus-private gap quantified. **Stretch.** Fine-tune and re-run the same protocol.

### Lab 18.3 — RAG over contracts with citation verification

**Objective.** Build retrieval with grounding checks and a measured hallucination rate.

**Data.** 50–200 contract or policy documents, chunked, plus a 100-question golden set with gold clauses.

**Steps.** (1) Chunk by clause, not by token count. (2) Implement sparse and dense retrieval and compare. (3) Add reranking and measure recall@5. (4) Require a citation for every factual claim. (5) Verify citations against the retrieved chunk. (6) Score abstention on unanswerable questions and report the unsupported-claim rate.

**Deliverable.** A working pipeline, an evaluation report, and an error taxonomy.

**Acceptance criteria.** Clause-level chunking; recall@5 reported; every claim cited; citations verified mechanically; abstention scored; negative control passes. **Stretch.** Add an injection test document and confirm the system ignores embedded instructions.

### Lab 18.4 — Agent guardrails and human approval

**Objective.** Build an agent with tools and prove the guardrails hold.

**Data.** A sandboxed ERP mock with read and write endpoints, an email sandbox, and an adversarial test set.

**Steps.** (1) Define three to five typed tools with least privilege. (2) Add a plan object and a step bound. (3) Add a spend limit and an allow-list. (4) Require human approval for every write action. (5) Run the adversarial set: disallowed tool, malformed argument, over-limit action, injection in a retrieved document, unsolvable goal. (6) Log the full trace and verify each guardrail fired.

**Deliverable.** An agent implementation, a guardrail test report, and traces.

**Acceptance criteria.** Every guardrail has a test that fails without it; no irreversible action executes without approval; injection test passes; full trace available. **Stretch.** Add a memory store with expiry and audit.

## 8. Assignment — Foundation-model evaluation

Produce a **foundation-model evaluation** for one supply-chain use case, at the standard a technical review board would accept.

| Section | Content |
|---|---|
| 1. Use case | Task, decision, volume, and cost of error |
| 2. Baseline | The current method and its measured performance |
| 3. Model options | At least three candidates, including one non-LLM option |
| 4. Evaluation design | Golden set construction, splits, frozen protocol |
| 5. Contamination | Cutoff dates, post-cutoff window, private holdout |
| 6. Results | Accuracy, grounding, hallucination, abstention, latency, cost |
| 7. Error analysis | Taxonomy of failures with examples and frequency |
| 8. Guardrails | Tools, limits, approval gates, injection tests |
| 9. TCO | Inference, review, engineering, platform, exit cost |
| 10. Recommendation | Adopt, pilot, or reject, with the reversal condition |
| 11. Reproduction | Code, prompts, golden set, one-command rerun |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Baseline discipline | 20% | Compares to nothing | Compares to one strong baseline | Two baselines, tuned, on identical splits |
| Contamination audit | 20% | Accepts the vendor's claim | Notes cutoff dates | Post-cutoff window plus private holdout |
| Evaluation quality | 20% | Single aggregate metric | Per-field and per-series | Calibration, abstention, and error taxonomy |
| Grounding | 15% | Cites documents | Cites spans | Verifies spans mechanically, with a negative control |
| Guardrails | 15% | Describes them | Implements allow-list and limits | Tests each guardrail and passes injection |
| TCO and decision | 10% | API cost only | Cost per document | Full annual cost with review and engineering |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| EU AI Act (Regulation 2024/1689) | Obligations by risk class; transparency for certain systems | Classify the use case; document; human oversight where required |
| GDPR / CCPA-CPRA | Personal data in documents, emails, and logs | Minimisation, retention limits, DPIA, processor terms |
| ISO/IEC 42001 | AI management system | Governance process, roles, documented risk assessment |
| NIST AI RMF | Risk identification and testing | Map, measure, manage, govern |
| Liability for automated statements | The firm owns its agent's output (*Moffatt v. Air Canada*, 2024 BCCRT 149) | Grounding, refusal, escalation path |
| Contract and IP | Copyright and confidentiality in training and RAG corpora | Licence review; access control per document |
| Trade secrets | Supplier pricing and terms in a model's context | Tenant isolation; no cross-customer context |
| Model provenance | Undocumented training data and cutoff | Vendor disclosure requirement; model card |
| Human oversight | Automation bias in approval gates | Reviewers see the evidence, not just the recommendation |

The ethical risk concentrates in two places. **Opacity**: an agent that recommends a supplier action without exposing its evidence cannot be overseen, and an approval gate over an unexplainable recommendation is theatre. **Automation bias**: a reviewer who approves 98% of recommendations is not reviewing; the design must sample-audit approvals and measure agreement independently.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Benchmark contamination | Zero-shot result above published supervised results | Test window in pretraining corpus | Wrong adoption decision | Post-cutoff window; private holdout |
| Temporal leakage | Validation loss better than production | Future information in features | Silent production failure | Feature-timing audit |
| Aggregate-only reporting | One MASE number, no slices | A few series drive the mean | Losses hidden in key segments | Per-series, volume-weighted |
| Point forecast for inventory | No intervals in the output | Mean forecast used for safety stock | Stockouts or excess | Quantile loss; calibration check |
| Local model on a cold-start item | New SKU has no forecast | No history to fit | Manual override culture | Global model with covariates |
| Ungrounded LLM answer | Fluent answer, no source | Model answers from parameters | Wrong policy applied | Retrieval plus span citation |
| Citation exists, does not support claim | Citation points to a real clause | Verification checks existence only | False confidence | Entailment or span check |
| No abstention | System always answers | Prompt forbids "I don't know" | Hallucination on unanswerable questions | Score abstention; permit refusal |
| Prompt injection | Retrieved doc contains instructions | Content treated as instruction | Unauthorised action | Data-only context; adversarial test |
| Agent without approval gate | Write tool callable directly | No irreversible-action control | Unauthorised purchase or email | Approval object; least privilege |
| Review queue unmeasured | No cost for human review | TCO excludes labour | Business case overstated | Model review volume explicitly |
| Multimodal domain shift | Accuracy drops on a new store | Different lighting or fixtures | Silent degradation | Held-out stores; drift monitoring |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | Why is MSE the wrong loss for inventory forecasting? | It targets the conditional mean; inventory costs are asymmetric, so train on the quantile the policy needs. |
| 2 | When does a global model beat a local one? | Many related, short series with shared structure and covariates; cold-start items are the clearest case. |
| 3 | What does patch tokenisation buy a Transformer? | Shorter effective sequence, long history retained, less redundant compute. |
| 4 | A vendor claims zero-shot beats your tuned model. What do you ask? | Cutoff date, benchmark versus private holdout, identical splits, covariates, horizon, calibration, per-series results. |
| 5 | How would you detect benchmark contamination? | Post-cutoff evaluation window, series fingerprinting, and a private holdout never published. |
| 6 | Explain the difference between retrieval quality and grounding. | Retrieval asks whether the right document was found; grounding asks whether the claim is supported by the cited span. |
| 7 | Why must a RAG system be allowed to abstain? | Unanswerable questions exist; a system that never refuses has a 100% hallucination rate on them. |
| 8 | What is prompt injection in a supply-chain agent? | Instructions embedded in retrieved content that redirect the agent; treat retrieved text as data only. |
| 9 | Which agent actions need human approval, and why? | Irreversible or counterparty-affecting actions: purchase orders, supplier emails, price changes, cancellations. |
| 10 | Your extraction is 95% per field across 12 fields. What is the record accuracy? | About 0.95^12 ≈ 54%; reliability is multiplicative, so design verification per field. |
| 11 | What is the biggest cost in an LLM business case? | Usually engineering and human review, not inference; compute all four components. |
| 12 | How do you evaluate a shelf-image model? | Out-of-stock and facing accuracy at the operating threshold, validated on held-out stores, with the cost of false alarms. |
| 13 | What would make you reject a vendor's TSFM? | No stated cutoff, no private-holdout evaluation, no calibrated intervals, no covariate support. |

## 13. Key terms

| Term | Definition |
|---|---|
| Backpropagation | Gradient computation through the computational graph used to train neural networks |
| Global model | One model trained across many series with shared parameters |
| Local model | A model fitted to a single series |
| RNN | Recurrent network carrying a hidden state across time steps |
| LSTM | Gated recurrent architecture designed to retain long-range dependencies |
| TCN | Temporal convolutional network using dilated causal convolutions |
| Transformer | Architecture built on self-attention over sequence positions |
| Causal attention | Attention restricted to positions at or before the current step |
| Patch tokenisation | Splitting a series into subsequences treated as tokens |
| Channel independence | Modelling each series separately while sharing weights |
| Quantile loss | Loss that trains a model to predict a specified quantile |
| DeepAR | Autoregressive probabilistic RNN forecaster for panels of series |
| N-BEATS | MLP-based forecaster with trend and seasonality basis expansion |
| N-HiTS | Multi-rate hierarchical forecaster for long horizons |
| TFT | Temporal Fusion Transformer with variable selection and interpretable attention |
| PatchTST | Patch-tokenised channel-independent Transformer forecaster |
| TSFM | Time-series foundation model pretrained for zero-shot or few-shot forecasting |
| Zero-shot | Applying a model to a task without task-specific training |
| Fine-tuning | Further training of a pretrained model on task data |
| Contamination | Evaluation data present in a model's training data |
| Temporal leakage | Use of future information in training or features |
| RAG | Retrieval-augmented generation: conditioning an answer on retrieved sources |
| Chunking | Splitting documents into retrievable units |
| Recall@k | Share of queries whose gold document appears in the top k results |
| Grounding | The degree to which claims are supported by cited sources |
| Hallucination | A confident output unsupported by any source |
| Abstention | Declining to answer when the evidence is absent |
| Agent | A language model in a loop with tools, state, and a stopping condition |
| Prompt injection | Malicious or accidental instructions embedded in retrieved content |
| Guardrail | A hard limit on agent actions or outputs |
| Human-in-the-loop | A required human approval before an action executes |
| Model card | Structured documentation of a model's training, evaluation, and limits |

## 14. Sources and further reading

**Standards and regulation**

- Regulation (EU) 2024/1689 (Artificial Intelligence Act). EUR-Lex.
- ISO/IEC 42001:2023, *Information technology — Artificial intelligence — Management system*. ISO.
- ISO/IEC 23894:2023, *Artificial intelligence — Guidance on risk management*. ISO.
- ISO/IEC 5259, *Data quality for analytics and machine learning*, Parts 1–4. ISO.
- NIST, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, January 2023. https://www.nist.gov/itl/ai-risk-management-framework — accessed 27 October 2026.
- Regulation (EU) 2016/679 (GDPR). EUR-Lex.

**Peer-reviewed and foundational**

- Hochreiter, S., and Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 9(8), 1997.
- Vaswani, A., et al. "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 30, 2017.
- Bai, S., Kolter, J. Z., and Koltun, V. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv:1803.01271, 2018.
- Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks." *International Journal of Forecasting*, 36(3), 2020.
- Oreshkin, B. N., Carpov, D., Chapados, N., and Bengio, Y. "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting." *ICLR*, 2020.
- Challu, C., et al. "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting." *AAAI*, 2023.
- Lim, B., Arık, S. Ö., Loeff, N., and Pfister, T. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." *International Journal of Forecasting*, 37(4), 2021.
- Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. "A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR*, 2023.
- Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, 2020.
- Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*, 2023.
- Ansari, A. F., et al. "Chronos: Learning the Language of Time Series." arXiv:2403.07815, 2024. `[VERIFY: venue and final version before citing as published.]`
- Das, A., Kong, W., Sen, R., and Zhou, Y. "A Decoder-only Foundation Model for Time-Series Forecasting." arXiv:2310.10688, 2023 (TimesFM).
- Woo, G., et al. "Unified Training of Universal Time Series Forecasting Transformers." arXiv:2402.02592, 2024 (Moirai).
- TSFM-Bench: benchmark evaluation of time-series foundation models. arXiv:2410.11802, 2024. `[VERIFY: exact title and author list before quoting results.]`
- Time-series benchmark contamination audit. arXiv:2510.13654, 2025. `[VERIFY: scope and findings before quoting.]`

**Industry**

- *Moffatt v. Air Canada*, 2024 BCCRT 149, British Columbia Civil Resolution Tribunal, 14 February 2024. https://decisions.civilresolutionbc.ca/ — accessed 27 October 2026. `[VERIFY: permalink to the decision.]`
- Gartner, "Gartner Predicts 70% of Organisations Will…", press release, 16 September 2025. Analyst claim; verify the exact scope of the 70% figure before quoting. https://www.gartner.com/en/newsroom — accessed 27 October 2026.
- Gartner, "Gartner Identifies the Top Supply Chain Technology Trends for 2026," press release, 30 June 2026. Analyst view on AI adoption in planning.
- IBM Research, Tiny Time Mixers (TTM) model family documentation. Vendor materials on parameter count and deployment cost.
- Amazon Science, Chronos model release notes and evaluation reports. Vendor-reported results.

**Tools and datasets**

- PyTorch. https://pytorch.org — accessed 27 October 2026.
- Hugging Face `transformers` and `datasets`. https://huggingface.co/docs — accessed 27 October 2026.
- `gluonts` probabilistic forecasting library (DeepAR reference implementation). https://ts.gluon.ai — accessed 27 October 2026.
- FAISS similarity search library. https://github.com/facebookresearch/faiss — accessed 27 October 2026.
- pgvector for vector search in PostgreSQL. https://github.com/pgvector/pgvector — accessed 27 October 2026.
- M5 and Favorita datasets as public panel baselines for Labs 18.1 and 18.2. Accessed 27 October 2026.

> **Note:** A foundation model that is 3% more accurate and unverifiable is worth less than a simpler model whose errors you can see. Build the evaluation before the system, and the verification before the deployment.

---

> **Next:** [Module 19](19_control_towers_visibility_events.md) — from model outputs to the operating picture: control towers, event streams, and the visibility layer that turns a forecast into a decision the organisation can act on.