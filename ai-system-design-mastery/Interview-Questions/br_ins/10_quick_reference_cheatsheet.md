# Day-Of Cheat Sheet — Brit Insurance Lead AI Engineer Interview

> **This is your last-minute, high-density reference. Read it the night before. Skim it the morning of. Know it cold.**

---

## 1. Brit Insurance Quick Facts (Know Cold)

| Fact | Detail |
|------|--------|
| **Market position** | Lloyd's of London market leader in specialty insurance |
| **AI initiatives** | Broker APIs, Underwriting Transformation, catastrophe ML (satellite imagery), virtual claims adjusting |
| **India GCC** | Global Capability Centre working closely with UK AI team — expect collaboration questions |
| **Tech stack** | Azure AI, Copilot Studio, Power Platform, Cytora |
| **Strategy** | Smarter Underwriting + Operational Simplification |
| **Regulatory** | Lloyd's, FCA, PRA compliance — regulated AI is a core concern |
| **Ki Insurance** | Brit-incubated algorithmic underwriting spin-off (now independent) — shows appetite for AI-first products |

> [!IMPORTANT]
> Mention **Smarter Underwriting** and **Lloyd's regulated environment** naturally in your answers. These signal you've done your homework.

---

## 2. The 8 Questions They Asked (Previous Candidate — Memorize Triggers)

| # | Question | One-Line Memory Trigger |
|---|----------|------------------------|
| 1 | Azure AI components? | OpenAI + AI Search (hybrid) + AI Foundry + Document Intelligence + Managed Identity |
| 2 | RAG architecture with code? | Ingestion pipeline → query pipeline → hybrid search → reranker → generation |
| 3 | Team size & composition? | [Your number] + seniority mix + leadership structure + team outcomes |
| 4 | Agent evaluation? | Offline (task success, tool accuracy, trajectory) + Online (completion rate, escalation rate, cost) + LLM-judge |
| 5 | Git linear history? | Rebase local + Squash Merge on PR + branch protection rules |
| 6 | Image anomaly system design? | Blob → Queue → ML inference → Results DB → Review API → Audit log |
| 7 | RAG code review? | Explain → Walk through → 5 improvements (no error handling, no hybrid search, no reranking…) |
| 8 | Project deep-dive? | STAR format, lead-level, architecture-first, quantified outcomes |

> [!TIP]
> For Question 7 (code review), always narrate **what the code does first**, then pivot to improvements. Never lead with criticism.

---

## 3. Key Frameworks (One-Liners — Reach for These Under Pressure)

### 🎯 Answer Structure Frameworks

| Framework | When to Use | Template |
|-----------|-------------|----------|
| **Headlining** | Always, every answer | Start with the conclusion sentence, then justify |
| **STAR** | Behavioural / project questions | Situation (1 sentence) → Task (my role) → Action (60%, use 'I') → Result (quantified) |
| **Problem-Solution-Tradeoff** | Technical design questions | What problem → what solution → what you gave up |
| **4-Dial** | Optimisation / trade-off questions | Accuracy / Latency / Cost / Reliability — name the dial you're turning |
| **5-Phase AI** | End-to-end AI system questions | Diagnose → Retrieve → Generate → Evaluate → Monitor |

### 🌉 Bridge Statement (Use When Stuck)
> *"That's an interesting edge case. Let me think through it with a framework — when I've faced this before, the key tension is usually between [X] and [Y]. I'd approach it by..."*

**Never say:** *"I cannot think of it currently now."* — use the bridge every time.

---

## 4. RAG One-Page Summary

### Ingestion Pipeline
```
Raw Documents
    → Parse (PDF/HTML/DOCX via Document Intelligence)
    → Clean (strip noise, normalize)
    → Chunk (semantic chunking or parent-child hierarchy)
    → Embed (text-embedding-3-large, 3072 dims)
    → Index (Azure AI Search — vector + keyword fields)
```

### Query Pipeline
```
User Query
    → Embed query (same model as ingestion)
    → Hybrid Search (BM25 keyword + cosine vector similarity)
    → Rerank (cross-encoder, e.g. ms-marco-MiniLM)
    → Build context window (top-k chunks + metadata)
    → Generate (Azure OpenAI GPT-4o)
    → Verify (faithfulness check, source citation)
    → Return to user with citations
```

### Key RAG Metrics (RAGAS Framework)
| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Answer grounded in context? | > 0.85 |
| **Answer Relevancy** | Answer addresses the question? | > 0.80 |
| **Context Precision** | Retrieved chunks are relevant? | > 0.75 |
| **Context Recall** | All relevant chunks retrieved? | > 0.70 |

### Top 5 Improvements to Weak RAG Code
1. ❌ No hybrid search → ✅ Add BM25 + vector (semantic + keyword coverage)
2. ❌ No reranker → ✅ Add cross-encoder reranking for precision
3. ❌ No error handling → ✅ Wrap calls in try/except with fallback
4. ❌ No metadata → ✅ Store source, date, confidence for auditability
5. ❌ No eval harness → ✅ Add RAGAS or custom eval before pushing to prod

> [!NOTE]
> In insurance, **auditability = metadata on every retrieved chunk** (policy ID, version, effective date). Always mention this.

---

## 5. Agentic AI One-Page Summary

### Architecture Stack
```
User Intent
    → Planning Loop (ReAct: Reason → Act → Observe)
    → Tool Calling (structured JSON schemas, strict validation)
    → State Management (LangGraph StateGraph or custom dict)
    → Memory (4 layers: sensory buffer, working, episodic, semantic)
    → HITL Checkpoints (human-in-the-loop for high-stakes decisions)
    → Response
```

### Evaluation Matrix
| Level | Metric | How Measured |
|-------|--------|-------------|
| **Offline** | Task success rate | Automated test suite against golden set |
| **Offline** | Tool call accuracy | Correct tool + correct params |
| **Offline** | Trajectory quality | Optimal steps vs. actual steps taken |
| **Online** | Task completion rate | % of sessions fully resolved |
| **Online** | Escalation rate | % handed to human (target: lower over time) |
| **Online** | Cost per session | Token spend + compute per conversation |
| **Quality** | LLM-as-judge | GPT-4o rating coherence, groundedness, helpfulness |

### Failure Modes & Mitigations
| Failure | Symptom | Mitigation |
|---------|---------|------------|
| **Infinite loop** | Agent stuck repeating same tool call | `max_iterations` cap + fallback to human |
| **Prompt injection** | Retrieved doc hijacks agent instructions | Scan retrieved content; separate system/user trust levels |
| **Duplicate writes** | Idempotency violations on retries | Idempotency keys on all write operations |
| **Context exhaustion** | Token limit exceeded mid-task | Summarization + checkpointing; sliding window |
| **Hallucinated tool args** | Agent invents parameter values | Strict JSON schema validation; structured outputs |

### Orchestration Choice
| Framework | Best For | Avoid When |
|-----------|----------|------------|
| **LangGraph** | Stateful, multi-step, complex branching | Prototype / simple Q&A |
| **LangChain** | Simpler chains, fast prototyping | Complex state management needed |
| **Custom state machine** | Max control, regulated environment | Time-to-market pressure |

> [!IMPORTANT]
> In a **regulated insurance** context, always lean towards **Custom state machine or LangGraph** — explainability and auditability over convenience.

---

## 6. Azure AI Services Quick Reference

| Service | Core Function | Insurance Use Case | Key Detail |
|---------|--------------|-------------------|------------|
| **Azure OpenAI** | LLM inference (GPT-4o, GPT-4 Turbo) | Policy Q&A, submission parsing | PTU for cost predictability |
| **Azure AI Search** | Hybrid RAG retrieval (BM25 + vector) | Policy document search, appetite matching | Semantic ranker built-in |
| **AI Foundry** | LLM eval, prompt management, deployment | Model versioning, A/B eval | Successor to Azure ML Studio for LLMs |
| **Document Intelligence** | OCR + form parsing | Submission extraction, claims docs | Prebuilt + custom models |
| **Content Safety** | Prompt shields, jailbreak detection | Compliance guardrails, PII filtering | Required in FCA-regulated systems |
| **Copilot Studio** | Low-code agent builder | Internal tools, quick wins, IT helpdesk | Good for non-engineering sponsors |
| **Azure ML** | MLOps, classical ML pipelines | Risk scoring, cat model deployment | Use for non-LLM workloads |
| **Managed Identity** | Passwordless auth | Secure service-to-service comms | Always mention — no secrets in code |
| **Service Bus** | Async message queue | Surge handling for claims/catastrophe events | Dead-letter queue for error handling |

---

## 7. Git Linear History — 60-Second Answer (Memorize Verbatim)

> *"We used a **rebase + squash merge** strategy. Developers rebase on main locally and never merge. When the PR is approved, we use **Squash and Merge** in GitHub or Azure DevOps — so every feature lands as a single atomic commit on main. The three rules: never rebase a shared branch, always use `--force-with-lease` instead of `--force`, and enforce squash-only merges via branch protection policies. The result is a perfectly straight `git log --oneline` on main, which makes `git bisect` and release notes trivially easy."*

### Cheat Card
```bash
# Developer workflow
git fetch origin
git rebase origin/main          # stay current, never merge
git push --force-with-lease     # safe force push

# PR merge strategy (enforced via branch protection)
# GitHub: Squash and Merge only
# Azure DevOps: Squash commit policy

# Result on main
git log --oneline main
# a1b2c3d feat: add catastrophe risk scoring endpoint
# e4f5g6h feat: add RAG pipeline for policy search
# i7j8k9l fix: handle null submission fields
```

---

## 8. Image Anomaly System Design — 30-Second Summary

> *"I'd design it with: **Azure Blob Storage** for ingestion, **Service Bus queue** for decoupling and surge handling, **Azure ML inference endpoint** with auto-scaling GPU clusters, a **results database** storing structured output (damage score, confidence, bounding boxes), a **review API** for adjuster workflow with full audit trail, and a **monitoring layer** for model drift and data skew. For catastrophe events causing 10× surge, the queue acts as a buffer — the GPU cluster auto-scales based on queue depth with a ceiling to manage cost."*

### Architecture Diagram (Verbal)
```
[Drone/Satellite Image Upload]
        ↓
[Azure Blob Storage]  ← cold storage, lifecycle policies
        ↓
[Azure Service Bus Queue]  ← decouple ingestion from processing
        ↓
[Azure ML Inference Endpoint]  ← GPU cluster, auto-scales on queue depth
        ↓
[Azure SQL / Cosmos DB]  ← damage score, confidence, bounding boxes, model version
        ↓
[Adjuster Review API]  ← human-in-the-loop, approve/reject/escalate
        ↓
[Audit Log]  ← every decision, timestamp, model version, human override (FCA/Lloyd's req)
        ↓
[Azure Monitor + Evidently AI]  ← drift detection, data quality, alerting
```

### Key Design Decisions to Mention
| Decision | Choice | Reason |
|----------|--------|--------|
| Queue vs. direct call | Service Bus queue | Absorbs catastrophe surge spikes |
| GPU scaling trigger | Queue depth metric | More responsive than time-based scaling |
| Output storage | Structured DB with model version | Enables audit + model comparison |
| Human override | Mandatory in audit log | FCA/Lloyd's explainability requirement |
| Drift detection | Monitor input distribution | Seasonal, geographic, camera model shifts |

---

## 9. People Leadership — 30-Second Summary

> *"I've led teams of **[X]** across AI engineering and data science. My approach: set clear goals tied to business outcomes and individual capability plans, weekly 1:1s with structured feedback, pair programming for technical depth-building, and creating reusable standards — code review checklists, eval harnesses, runbooks. I also work cross-functionally with product, actuaries, and underwriters to translate vague requirements into scoped technical deliverables. I measure my own success by the team's output quality and individual growth trajectories, not just my personal delivery."*

### Leadership Evidence Points (Slot In Your Real Stories)
| Area | What They Want to Hear |
|------|------------------------|
| **Mentoring** | Specific example of upskilling a junior (pair programming, design reviews) |
| **Standards** | Created something reusable: eval harness, code review checklist, runbook |
| **Cross-functional** | Translated business/actuarial ask into AI system spec |
| **Conflict** | Navigated disagreement on technical direction with evidence/data |
| **Delivery** | Shipped production AI system with measurable business impact |

---

## 10. The 4 Panels They Are Testing

```
┌─────────────────────────────────────────────────────────────────┐
│  PANEL 1: TECHNICAL                                             │
│  RAG architecture · Agentic AI · Azure AI services             │
│  System design · Code quality · LLMOps/MLOps                   │
├─────────────────────────────────────────────────────────────────┤
│  PANEL 2: PRODUCT MINDSET                                       │
│  Ambiguous → scoped → shipped                                   │
│  Business impact · Stakeholder management · Prioritisation      │
├─────────────────────────────────────────────────────────────────┤
│  PANEL 3: PEOPLE LEADERSHIP                                     │
│  Mentoring · Managing upward · Cross-functional collaboration   │
│  Building team capability · Engineering culture                 │
├─────────────────────────────────────────────────────────────────┤
│  PANEL 4: REGULATED ENVIRONMENT                                 │
│  Auditability · Governance · Explainability                     │
│  Human oversight · FCA/Lloyd's compliance · Risk awareness      │
└─────────────────────────────────────────────────────────────────┘
```

> [!TIP]
> **Every technical answer** should have a regulated-environment angle. End with: *"…and in a Lloyd's-regulated context, we'd also add an audit log / explainability layer / human-in-the-loop checkpoint."*

---

## 11. Phrases to AVOID ❌

| Don't Say | Say Instead |
|-----------|------------|
| *"I cannot think of it currently now"* | Use the Bridge Statement — always |
| *"Uh", "Like", "Um"* | Pause. Breathe. Think. Then speak. |
| *"We did X..."* without your role | *"I specifically owned X, while the team..."* |
| Starting with a definition | Start with architecture or conclusion, define terms inline |
| *"It depends"* (without follow-up) | *"It depends on [X] vs [Y] — in an insurance context I'd default to..."* |
| Being purely theoretical | Ground every answer in a real (or realistic) system design |
| Badmouthing previous employer | Reframe as learning: *"It taught me that..."* |

---

## 12. Opening Gambit — Memorize This

> *"I'm [Name], a Lead AI Engineer with [X] years building production AI systems. Most recently I [one sentence on biggest project at right scale — e.g., 'led the design and delivery of an end-to-end RAG platform processing 50,000 insurance documents per day for underwriter Q&A']. I've managed teams of [X] across AI engineering and data science. What excites me about Brit specifically is the combination of catastrophe ML at Lloyd's scale, the Smarter Underwriting strategy, and the real challenge of deploying AI responsibly in a heavily regulated environment. I'm here to talk about how I'd help Brit accelerate that vision."*

### Personalization Slots
- `[X] years` → your actual number
- `[one sentence on biggest project]` → pick your most impressive, closest to insurance
- `[X] across AI engineering and data science]` → your actual team size + disciplines
- Memorize the Brit specifics: **catastrophe ML + Smarter Underwriting + Lloyd's regulated environment**

---

## 13. Closing Questions — Ask 2–3 of These

> [!NOTE]
> Questions signal curiosity and strategic thinking. Pick the ones most relevant to your conversation.

| # | Question | Why It's Good |
|---|----------|---------------|
| 1 | *"What does the India-UK AI team collaboration model look like day-to-day?"* | Shows awareness of GCC structure, practical curiosity |
| 2 | *"What are the top 2 AI initiatives the team will be focused on in the next 12 months?"* | Strategic, forward-looking, helps you assess fit |
| 3 | *"How mature is Brit's MLOps/LLMOps infrastructure, and where do you see the biggest opportunity to improve it?"* | Technical depth, leadership signal, opens discussion |
| 4 | *"What does 'good' look like for this Lead AI Engineer role at 6 months?"* | Outcome-oriented, shows you plan to deliver fast |
| 5 | *"How does Brit think about the build-vs-buy decision for AI tools — e.g., Copilot Studio vs. custom agents?"* | Shows product + commercial awareness |

---

## 14. Key Insurance & Lloyd's Vocabulary

| Term | Definition (Know This Cold) |
|------|-----------------------------|
| **Underwriting** | Process of evaluating and assuming risk in exchange for a premium |
| **Submission** | Broker's request for insurance coverage on behalf of a client — the AI input signal |
| **Appetite** | Insurer's defined willingness to take on specific risk types |
| **Lloyd's of London** | The insurance marketplace where syndicates underwrite risk — not a company |
| **Syndicate** | Group of underwriters at Lloyd's that pool capacity for a line of business |
| **FCA** | Financial Conduct Authority — regulates conduct in UK financial services |
| **PRA** | Prudential Regulation Authority — regulates financial soundness / systemic risk |
| **Broker API** | Digital interface connecting Brit's pricing/appetite to broker platforms (AI-powered) |
| **Catastrophe modeling** | Statistical modeling of large-scale insured losses (hurricanes, earthquakes, floods) |
| **Claims adjuster** | Person who assesses and settles insurance claims after a loss event |
| **Virtual claims adjusting** | Remote AI-powered damage assessment using aerial imagery / satellite / drone data |
| **Premium** | Amount charged to the insured in exchange for coverage |
| **Reinsurance** | Insurance for insurers — Brit cedes risk to reinsurers |
| **Quota share** | A reinsurance agreement where a percentage of all premiums/losses is shared |
| **Loss ratio** | Claims paid ÷ premiums earned — key performance metric |

---

## 15. The Day-Of Mental Checklist

```
Morning of interview:
□ Read sections 1, 2, 12 (facts + questions + opening)
□ Recite your Opening Gambit out loud — time it (aim for 45-60 seconds)
□ Check your biggest STAR story is ready (your strongest project)
□ Remember: architecture first, then implementation, then trade-offs

In the interview:
□ Headlining: conclusion first on EVERY answer
□ Pause before speaking — 3 seconds is fine, it looks thoughtful
□ Say "I" not "we" — own your contributions
□ End technical answers with the regulated-environment angle
□ If stuck → Bridge Statement, never "I don't know"

Closing:
□ Ask 2-3 questions from Section 13
□ Thank each interviewer by name
□ Reiterate your excitement for the specific Brit opportunity
```

---

## 16. Quick-Fire Answer Templates

### When asked about a technology you're less sure about:
> *"I've worked with [adjacent tech], which shares [core concept]. My approach would be [logical extension]. The key considerations in this context would be [1], [2], [3]. Is there a specific aspect you'd like me to go deeper on?"*

### When asked to critique code you've never seen:
> *"Let me first describe what this code is doing… [narrate]. Now, from a production-readiness standpoint, I'd focus on five areas: error handling, observability, scalability, security, and maintainability. Looking at this code specifically…"*

### When asked about failure:
> *"The project that challenged me most was [X]. The root cause was [honest assessment]. What I did was [action]. The outcome was [result]. What I took away — and still apply — is [specific lesson]."*

### When asked about your biggest technical strength:
> *"I'd say it's translating ambiguous business problems into rigorous AI system designs that actually ship. I combine [technical depth area] with the ability to communicate trade-offs clearly to non-technical stakeholders, which I've found is the rarest skill at Lead level."*

---

> [!CAUTION]
> **Do not memorize word-for-word scripts** (except Opening Gambit and Git answer). Know the **structure and key terms** — natural delivery beats robotic recitation every time.

---

*Last updated for Brit Insurance Lead AI Engineer interview. Good luck — you've got this.* 🎯
