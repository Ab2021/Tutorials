# Agentic / LLM / RAG Evaluation — Knowledge Base

A complete, source-grounded knowledge base on evaluating LLM systems, built from **38 video
transcripts** across nine source repositories. It is written to be *used* — to prepare for an
interview, to design an eval suite on Monday, or to answer "which model should we pick and what will
it cost."

**Scope:** evaluation foundations → methods and judges → benchmarks and leaderboards → RAG evaluation
→ agentic and trajectory evaluation → production, online and operational evaluation.

**Size:** 48 documents, ~325,000 words.

---

## How to use this KB

There are four artifact types, and they answer four different questions. Pick by the question you
have, not by the folder you land in.

| Artifact | Prefix | Answers | Use when |
|---|---|---|---|
| **Case study** | `CS-01`…`CS-24` | *"What does this source actually teach, in full?"* | You want to understand a topic properly — every mechanism, pros/cons, exceptions and when-to-use |
| **Cheat sheet** | `CHEAT-*` | *"What do I need at the moment of decision?"* | You are in a design review, an incident, or an interview and need the rule, the formula or the threshold |
| **Interview pack** | `IQ-*` | *"How will this be tested, and what does a strong answer sound like?"* | You are preparing for, or running, a hiring loop |
| **Codebase dossier** | `CODE-01`…`CODE-07` | *"What does the real implementation look like?"* | You are choosing or reading an actual eval framework |

**Every artifact is traceable.** Case studies carry `[mm:ss]` anchors into the source transcript for
nearly every claim. Where a claim is not from the source, it is marked `**Analyst note (outside
source):**` — so you can always tell transcript from supplementation.

---

## Reading paths

Six routes through the KB. Each is a real sequence, not a list.

### 1. "I am new to LLM evals" — the foundations arc
`CS-01` → `CS-02` → `CS-03` → `CS-04` → `CS-06` → `CS-07`
Then `CHEAT-eval-workflow.md`, then `IQ-eval-foundations.md` to test yourself.
**You will be able to:** distinguish a model eval from an application eval, name the 12-step eval
loop, and explain why one eval pipeline is never enough.

### 2. "I need to build an eval suite this week" — the practitioner arc
`CS-04` (the workflow) → `CS-07` (graders) → `CS-08` (G-Eval) → `CS-13`/`CS-14` (if RAG) → `CS-20`
(if an agent) → `CODE-03` (the process as agent skills) → `CHEAT-eval-workflow.md`
**You will be able to:** define success criteria, build a golden dataset, pick a grader, and wire a
regression gate.

### 3. "Should we use an LLM judge?" — the judge arc
`CS-07` → `CS-08` → `CS-22` §(the precision/recall audit) → `CHEAT-judge-design.md` → `IQ-judges.md`
**The load-bearing fact:** one team that actually measured its judge got **below 40% precision** —
"a flip of a coin." Never deploy an unvalidated judge.

### 4. "Which model should we buy?" — the selection arc
`CS-09` → `CS-10` → `CS-11` → `CS-12` → `CHEAT-benchmarks.md`
**The load-bearing fact:** a leaderboard is a **filtering** tool, not a decision tool. In the worked
example, the week's headline model came **last** on the team's own task.

### 5. "Our agent is in production" — the operations arc
`CS-20` → `CS-21` → `CS-22` → `CS-23` → `CS-24` → `CHEAT-production-evals.md` → `IQ-production.md`
**The load-bearing fact:** you are not evaluating a model — you are evaluating a **harness**, and
quality, cost and latency are one measurement.

### 6. "I am interviewing / hiring" — the loop arc
`IQ-eval-foundations` → `IQ-judges` → `IQ-rag` → `IQ-benchmarks` → `IQ-agentic` → `IQ-production`
Each pack has five tiers: fundamentals → applied trade-offs → senior/staff → debug-this-scenario →
trap questions, plus live-coding prompts, a take-home brief and a scoring rubric.

---

## Case studies — 24 documents, six domains

Each is a full treatment of one source: executive summary, definitions, decomposed content, frameworks,
a worked end-to-end example, pros/cons/exceptions, failure modes, implementation notes, interview-ready
Q&A, a cheat-sheet block and a glossary.

### 01 · Foundations

| # | Document | What it covers |
|---|---|---|
| CS-01 | [Model evals vs application evals](case_studies/01-foundations/CS-01-model-evals-vs-application-evals.md) | What "LLM evals" actually means — a whole testing setup, not a metric — and the split between *model capability* evals and *application* evals |
| CS-02 | [The eval curriculum map](case_studies/01-foundations/CS-02-llm-evals-playlist-and-curriculum-map.md) | Why shipping an unevaluated LLM app is dangerous, the two structural reasons LLM apps are harder to test than software, and the ten-topic roadmap |
| CS-03 | [Why multiple eval pipelines](case_studies/01-foundations/CS-03-why-multiple-eval-pipelines.md) | One escalating RAG example proving an LLM app has multiple failure points and risk categories — so it needs **three** levels of pipeline |
| CS-04 | [The complete eval workflow](case_studies/01-foundations/CS-04-complete-eval-workflow.md) | The canonical **12-step loop**, demonstrated end to end on a Zomato email-router |
| CS-05 | [Model evals and capabilities](case_studies/01-foundations/CS-05-model-evals-and-capabilities.md) | The four-step anatomy of a model eval, the benchmark-vs-custom-eval fork, and the eight core capabilities benchmarks target |

### 02 · Methods

| # | Document | What it covers |
|---|---|---|
| CS-06 | [Offline vs online evals](case_studies/02-methods/CS-06-offline-vs-online-evals.md) | Offline proves *correct*; online proves *normal* — and how they form one self-improving loop |
| CS-07 | [LLM-as-a-judge: reference-based vs reference-free](case_studies/02-methods/CS-07-llm-as-a-judge-reference-based-vs-reference-free.md) | Every pipeline is executed by one of **three graders** (program, human, LLM) and every case is either reference-based or reference-free |
| CS-08 | [G-Eval: the deterministic judge](case_studies/02-methods/CS-08-g-eval-deterministic-judge.md) | How to remove the two sources of judge variance — a one-line criteria and integer token sampling |

### 03 · Benchmarks & model selection

| # | Document | What it covers |
|---|---|---|
| CS-09 | [Reading a leaderboard](case_studies/03-benchmarks/CS-09-reading-llm-leaderboards.md) | A leaderboard is a **filtering** tool: shortlist 3–5, then run your own eval |
| CS-10 | [The evolution of knowledge benchmarks](case_studies/03-benchmarks/CS-10-evolution-of-ai-knowledge-benchmarks.md) | Seven benchmarks as one story — MMLU → TruthfulQA → AGIEval → GPQA → MMLU-Pro → SimpleQA → HLE — and why each saturates |
| CS-11 | [Saturation vs contamination](case_studies/03-benchmarks/CS-11-benchmark-saturation-vs-contamination.md) | The benchmark's **four components**, and the four independent reasons a number is untrustworthy |
| CS-12 | [Selecting the right LLM](case_studies/03-benchmarks/CS-12-selecting-the-right-llm-custom-model-evals.md) | A live build: requirements → cost filter (**146 models → 5**) → weighted ranking → execution-based custom eval |

### 04 · RAG

| # | Document | What it covers |
|---|---|---|
| CS-13 | [Testing RAG retrievers, hands-on](case_studies/04-rag/CS-13-testing-rag-retrievers-hands-on.md) | Why `Recall@K` over chunk IDs is the wrong golden dataset, and the iteration loop chunk size → reranker → embedding → k |
| CS-14 | [RAG evaluation: interview framing](case_studies/04-rag/CS-14-rag-evaluation-interview-framing.md) | An eight-step roadmap, a three-level suite, `run_evals.py`, and a five-part spoken answer |
| CS-15 | [Operational evals: faster, cheaper](case_studies/04-rag/CS-15-rag-operational-evals-faster-cheaper.md) | Latency, cost and reliability — measured with telemetry instead of judges — and used to decide deployability |
| CS-16 | [Securing RAG: toxicity, leakage, scope drift](case_studies/04-rag/CS-16-securing-rag-toxicity-leakage-scope-drift.md) | Building the safety suite: an eval policy as ground truth, plus three DeepEval suites with real scores and fixes |

### 05 · Agentic

| # | Document | What it covers |
|---|---|---|
| CS-17 | [Agentic evaluations workshop](case_studies/05-agentic/CS-17-agentic-evaluations-workshop.md) | Honest reporting, reliability as a second axis, simulated environments (GAIA 2 / ARE), environment-first practice, living benchmarks |
| CS-18 | [RL for agents workshop](case_studies/05-agentic/CS-18-rl-for-agents-workshop.md) | Why the rollout loop changed, benchmark construction, recursive language models, and *evals and environments are the same object* |
| CS-19 | [Fine-tuning a coding agent](case_studies/05-agentic/CS-19-fine-tuning-a-coding-agent-for-continual-learning.md) | A live SFT parameter sweep driven by a coding agent, and the masking mechanics underneath |

### 06 · Production, online & operational

| # | Document | What it covers |
|---|---|---|
| CS-20 | [Evals for agents that thrive in production](case_studies/06-production/CS-20-building-evals-for-agents-that-thrive-in-prod.md) | HHH+R, the **harness** as the evaluation target, context/tool/task/cost metric families, and evals costing ~10× serving |
| CS-21 | [Observability: traces, evals, alerts, red teaming](case_studies/06-production/CS-21-observability-traces-evals-alerts-red-teaming.md) | Four pillars on OpenTelemetry, the $250-per-contract human-eval arithmetic, alert thresholds, and the PM/engineering split |
| CS-22 | [Setting up agent evals and scaling them](case_studies/06-production/CS-22-setting-up-agent-evals-and-scaling-them.md) | Ground truth from lawyers, precision/recall arithmetic, subjectifying subjective criteria, and the judge audit you run before deploying |
| CS-23 | [Pricing AI agents and ROI](case_studies/06-production/CS-23-pricing-ai-agents-and-roi.md) | Value-first pricing, attribution × autonomy, **ceiling cost**, credits, the 80% rule, and the eval bill as a cost-of-goods line |
| CS-24 | [The harness shift](case_studies/06-production/CS-24-n8n-limitation-and-claude-code.md) | From *build your own agent* to *adopt a harness* — file-system search over embeddings, connectors, and the in-loop "evaluate" that is not an eval suite |

---

## Cheat sheets — 8 documents

Compact, decision-shaped reference. Every one has a 60-second version, core concepts, formulas or
thresholds, decision rules, top mistakes, and "if you only remember three things."

| Document | Use when |
|---|---|
| [Evaluation workflow, end to end](cheat_sheets/CHEAT-eval-workflow.md) | You are running an eval project and need the process |
| [LLM-as-a-judge: design & validation](cheat_sheets/CHEAT-judge-design.md) | You are writing or validating a judge |
| [Metric formulas & statistics](cheat_sheets/CHEAT-metrics.md) | You need the exact formula or the right statistic |
| [Benchmarks & leaderboards](cheat_sheets/CHEAT-benchmarks.md) | Someone shows you a score and expects you to act on it |
| [RAG evaluation](cheat_sheets/CHEAT-rag-eval.md) | You have a retriever + generator pipeline |
| [Agentic & trajectory evaluation](cheat_sheets/CHEAT-agentic-evals.md) | You are evaluating a multi-step agent |
| [Production & online evaluation](cheat_sheets/CHEAT-production-evals.md) | Your system is live and the question is cost, drift and price |
| [Choosing eval tooling](cheat_sheets/CHEAT-tools.md) | Someone asks "which eval framework should we use?" |

---

## Interview packs — 6 documents

Five tiers each: **fundamentals → applied trade-offs → senior/staff → debug-this-scenario → trap
questions**, plus live-coding prompts, a take-home brief, and a scoring rubric that states what
separates a hire from a no-hire.

| Pack | Covers | Level |
|---|---|---|
| [LLM evaluation foundations](interview_questions/IQ-eval-foundations.md) | CS-01…CS-05 | junior → mid |
| [LLM-as-a-judge & evaluation methods](interview_questions/IQ-judges.md) | CS-06, CS-07, CS-08 | mid → senior |
| [RAG evaluation](interview_questions/IQ-rag.md) | CS-13…CS-16 | mid → senior |
| [Benchmarks, leaderboards & model selection](interview_questions/IQ-benchmarks.md) | CS-09…CS-12 | mid → senior |
| [Agentic & trajectory evaluation](interview_questions/IQ-agentic.md) | CS-17, CS-18, CS-19 | senior → staff |
| [Production, online & operational evaluation](interview_questions/IQ-production.md) | CS-20…CS-24 | senior → staff |

---

## Codebase dossiers — 7 documents

Each dossier follows the same ten-section shape: what it is / what it is not → architecture → repository
map → core abstractions → **how an evaluation actually runs** → extending → configuration → strengths
and weaknesses → a minimal example → a reading order.

| Dossier | Repository | The one idea worth taking |
|---|---|---|
| [CODE-01](codebases/CODE-01-awesome-evals.md) | `awesome-evals` (BenchFlow) | The catalogue — `PATTERNS.md` as the index of eval patterns |
| [CODE-02](codebases/CODE-02-openai-evals.md) | OpenAI `evals` | An eval is **YAML plus optional code** — the registry-driven runner |
| [CODE-03](codebases/CODE-03-evals-skills.md) | `evals-skills` (AI Evals Course) | The eval **process itself**, packaged as agent skills |
| [CODE-04](codebases/CODE-04-evalscope.md) | EvalScope (ModelScope) | **205 benchmark adapters** behind one CLI, plus arena and stress testing |
| [CODE-05](codebases/CODE-05-frontier-evals.md) | `frontier-evals` | Frontier-model evaluation as infrastructure |
| [CODE-06](codebases/CODE-06-langfuse.md) | Langfuse | Traces, scores, experiments and prompts across **four storage planes** |
| [CODE-07](codebases/CODE-07-search-evals.md) | `search_evals` (Perplexity) | **Decimal cost accounting** with an explicit agent/grader cost split |

See also [`codebases/README.md`](codebases/README.md) for the cross-cutting comparison.

---

## Reference material

| Document | Contents |
|---|---|
| [`_meta/SOURCE_MAP.md`](_meta/SOURCE_MAP.md) | Every source transcript → the case study it became, plus the code-repository table |
| [`_meta/STYLE_GUIDE.md`](_meta/STYLE_GUIDE.md) | The conventions every document in this KB follows |

---

## How this KB was built

- **38 source transcripts** were mapped to **24 case studies**, and each case study was written from
  its own transcript rather than from general knowledge. Numbers in the case studies are the source's
  numbers.
- **Anchors.** `[mm:ss]` marks the point in the transcript where the claim appears, so any figure can
  be checked against the recording.
- **The analyst-note convention.** `**Analyst note:**` flags the author's judgement about the source
  (an internal inconsistency, an undelivered promise, a transcription artefact).
  `**Analyst note (outside source):**` flags anything that is *not* in the transcript. Nothing invented
  enters the KB unmarked.
- **Preserved inconsistencies.** Where a source contradicts itself, both values are kept and the
  contradiction is flagged rather than silently resolved. Notable examples: the **80%-vs-70%** judge
  deployment bar in CS-22, the **$786-vs-$1,000** bill in CS-21, and CS-12's unreconciled budgets and
  volumes. These are teaching material, not defects.

---

## Known limitations — read this before quoting

These are stated plainly so nothing here is mistaken for more than it is.

1. **Case studies and interview packs exceed the style guide's length target** (1,800–3,500 words and
   ≤9,000 words respectively). This is deliberate: every document was written to cover each minute
   aspect, with pros, cons, exceptions and when-to-use. The longest are CS-17/18/19 (workshop
   sessions) and CS-20/21/22 (production talks). The *calibration* is visible in CS-23 and CS-24,
   which are shorter and denser.

2. **Five case studies are one practitioner's series.** CS-20, CS-21, CS-22, CS-23 and CS-24 share a
   host, a running example (a contract-reading agent), a launch gate and a platform preference. They
   are **mutually corroborating but not independent observations**, and each of those files says so.
   Treat their tool rankings as one considered opinion, not as five data points.

3. **Some source figures are illustrative, not measured.** CS-12's cost arithmetic is a worked
   teaching example with internally inconsistent inputs; CS-20's "10× serving cost" and "16 minutes"
   come without a stated test-set size, token count or model. The *methods* are sound and transferable;
   re-derive the numbers for your own system.

4. **CS-24 is a nine-minute session, not a lecture.** It teaches no evaluation method — it documents
   where agent-building work moved. Its scope limit is flagged in the file itself, and its filename
   promises an "n8n limitation" the transcript never actually states.

5. **Transcription noise is present in the sources** and is flagged where it matters rather than
   silently corrected. Vendor and participant names are reproduced as the transcripts give them.

6. **Two warnings on `CODE-01` and `CODE-03`** — both are catalogues / process repositories with no
   single runnable code path, so some of the ten standard dossier sections are necessarily thin.

---

## Quality checks

```bash
python work/qa_kb.py evals-kb
```

The harness verifies, for every file: required section headings are present, relative markdown links
resolve to real files, word-count targets, and code-fence balance. Current state: **0 errors.**

---

## Contents at a glance

```
evals-kb/
├── README.md                     ← you are here
├── case_studies/                 24 documents
│   ├── 01-foundations/           CS-01 … CS-05
│   ├── 02-methods/               CS-06 … CS-08
│   ├── 03-benchmarks/            CS-09 … CS-12
│   ├── 04-rag/                   CS-13 … CS-16
│   ├── 05-agentic/               CS-17 … CS-19
│   └── 06-production/            CS-20 … CS-24
├── cheat_sheets/                 8 documents
├── interview_questions/          6 documents
├── codebases/                    7 dossiers + README
└── _meta/                        SOURCE_MAP.md, STYLE_GUIDE.md
```
