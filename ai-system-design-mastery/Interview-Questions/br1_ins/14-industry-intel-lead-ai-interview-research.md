# 14 — INDUSTRY INTEL: LEAD-LEVEL AI INTERVIEWS & REAL QUESTIONS (WEB RESEARCH)

> **TL;DR.** This file is the *evidence base* — what Lead/Staff/Principal Applied-AI loops actually look like in 2025–2026, the real questions being asked, the documented failure modes, the serving/quantisation numbers, and the sources. It backs every claim in the rest of the pack. **Read it once on day 2 to calibrate, and use §2 as extra drill material on day 6.**

**Caveat on sourcing (read this):** the strongest sources are Microsoft Learn docs, company newsrooms, LMA/Lloyd's publications, the Bank of England, and arXiv/EMNLP papers. Several benchmark-comparison and prep-site pages carry future-dated or synthesised numbers — those are marked **[indicative]** and should never be quoted to an interviewer as fact. Where research found *no* evidence, it says so rather than filling the gap.

---

## 1. The Lead-level loop, and what "Lead" actually changes

### 1.1 Typical shape (5–7 stages, 4–8 weeks)

Recruiter screen (30 min) → hiring manager (45) → coding/take-home (LLM-flavoured, applied) → **system design (60–90, often sat twice)** → project deep-dive (45, probed hard) → behavioural/leadership (45) → final/exec (30).

**Two things to take from this:**
1. **System design carries the most weight and is the differentiating round.** Grades come from *stated assumptions, justified trade-offs, and cost/latency/failure-mode/eval reasoning* — not framework syntax.
2. **Behavioural/leadership carries ~30–40% of the weight at senior level.** That is the panel-feedback gap, quantified.

### 1.2 What changes from Senior → Lead/Staff (scope, not coding skill)

| Dimension | Senior | Lead | Staff/Principal |
|---|---|---|---|
| Scope | one team/feature | the team's tech direction | 2–5 teams / org-wide |
| Coding time | 60–80% | 30–70% | 30–50% |
| Decision authority | suggests, needs approval | team-level architecture | org-wide standards |
| Time horizon | this quarter | project roadmap | 6–12 months+ |
| Problem source | given scoped problems | breaks down projects | **finds** unassigned problems |
| Mentorship | juniors | the team | grows seniors into leads |

**Promotion dimensions senior→staff:** scope · technical influence (decisions → **codified standards**) · ambiguity tolerance · mentorship · technical vision.

**Staff-level questions to expect**, which you should be able to answer from files `05`–`12`:

- "Design an AI platform serving 50 engineers across 5 product teams." (→ standardisation, golden paths, an internal eval harness, a gateway)
- "Tell me about a time you changed technical direction another team owned." (→ **influence without authority**)
- "How would you establish eval standards for a 50-person org with none?" (→ start with one gate on one product, prove it, then codify — file `12` §5)
- "Build or buy a fine-tuning pipeline, you have two weeks." (→ **reversibility analysis**: buy, because it's reversible and time-boxed)

### 1.3 Documented failure modes (the #1 is worth repeating)

| Rank | Failure mode | Your defence |
|---|---|---|
| **1** | **No evals** in the take-home or design | File `07` — three layers + CI gate. Say "eval" unprompted. |
| 2 | **Over-engineering** — five frameworks where one suffices | The escalation ladder; 0.9³ = 0.73 |
| 3 | Choosing frameworks **without justifying trade-offs** | "X over Y because Z, cost W" every time |
| 4 | Thin **production/deployment/monitoring** (MLOps) depth | Files `05` §5, `06` §6 |
| 5 | Treating rounds as isolated (panels evaluate holistically) | Consistent positioning across all four parts |

---

## 2. Real questions actually asked (2024–2026) — extra drill bank

These are real or representative reported questions, company-tagged where known. **Add these to your day-6 drill.** The ⭐ ones overlap the previous Brit round.

### 2.1 RAG
1. ⭐ Design a production RAG system end to end. *(OpenAI, xAI)*
2. Design a **low-latency RAG** customer-support assistant: **p99 ≤ 1.5 s, 10M docs, 50 QPS, multi-tenant**. *(OpenAI screen)* — note how close this is to the mock prompt's challenge.
3. Design **and optimise** RAG: ingestion → layout-aware parsing → chunking → **hybrid BM25+vector with RRF** → cross-encoder rerank → **forced citations** → **abstention ladder** → **ACL enforcement** → stage-by-stage eval plan. *(OpenAI, hard)*
4. Handle documents with **tables, charts and images**.
5. RAG vs fine-tuning at scale.
6. Multi-tenant RAG isolation; vector-DB indexing algorithm choice.

**The answer patterns interviewers reward (from the research):** two-plane decomposition (**offline indexing plane vs online query plane**) · hybrid retrieval + RRF → cross-encoder rerank · **filter before retrieval** for multi-tenancy, never after · grounded generation with **forced citations and an abstention ladder** · stage-by-stage eval (recall@k, nDCG, faithfulness, answer relevance) · explicit latency budget · **graceful degradation — abstain rather than hallucinate**.

> **Add "abstention ladder" to your vocabulary.** It names exactly what file `06` §2 calls the empty-retrieval guard: *retrieve → if nothing clears the threshold, say you don't know → if confidence is low, route to a human.*

### 2.2 Agents, orchestration & production
7. Agent vs workflow — where is the line? *(the "flow engineering" framing)*
8. **Explain MCP and why it matters for production agents.**
9. Design a multi-agent system; long-running agent tasks.
10. Design a scalable **LLM Gateway** (rate limiting, token management, intelligent failover).
11. Architect **real-time prompt-injection detection and mitigation**.
12. Implement **observability for agentic workflows** (LangGraph).
13. Design a **resumable/durable agent** with checkpointing and preemption.
14. Build a **Mistral-powered RAG agent** *(Meta onsite, hard)*: CLI (`index`/`ask`/`serve`), HTTP `/chat`, **streaming with top-k and citations**, **exponential backoff on 429/timeouts**, env config, tests, and *explain how you'd evaluate the answers*.

### 2.3 Serving, optimisation & cost
15. Why KV cache matters; when to use speculative decoding; **compare batching strategies**.
16. Optimise inference cost; optimise latency for real-time apps; quantisation for deployment.
17. Choose between frontier models for a workload; **SLM vs frontier**; when reasoning models are worth the cost.
18. **Maintain low latency while running security classifiers on high-volume traffic.**
19. Scale an AI chat feature to **1M daily users**; cost for 1M queries/day.

### 2.4 Evaluation, guardrails & safety
20. **Evaluate LLM outputs with no ground truth.**
21. Explain RAGAS; compare two RAG systems objectively.
22. Detect and handle hallucinations.
23. Design an eval system comparing two LLMs on open-ended tasks.
24. **Self-consistency vs best-of-N; prevent reward hacking in best-of-N.**
25. **Multi-agent debate vs mixture of agents.**
26. Defend against prompt injection; design AI content moderation.

### 2.5 Production/MLOps & classical ML
27. Observability and CI/CD for LLM apps; rate limits and quotas; **provider model updates and deprecations**; token counting; feedback loops; context-window management for long conversations.
28. Design a fraud-detection system; design a recommendation/ranking system with real-time serving, monitoring and drift detection.

### 2.6 Scenario sets in the wild
From a published AI-system-design guide: five scenarios × 35 min — **customer-support chatbot · document-processing pipeline · enterprise-search RAG · code assistant · content-moderation system**. Note that the first two are almost certainly your Part 3.

### 2.7 Lead-specific behavioural questions surfaced
- "Tell me about a time you changed technical direction another team owned."
- "How would you establish eval standards for a 50-person org with none?"
- "Build vs buy — you have two weeks."
- Audience adaptation: **explain the same architecture to a director, then to a CEO** (practise the three translations in file `12` §6.2).

---

## 3. What the coding round has become

**The shift, stated well:** interviewers stopped asking "can you call the API?" and now ask **"how do you know it works, and how does it fail?"** Five pillars: LLM mechanism (prefill/decode, KV cache, attention) · RAG as an information-retrieval problem (embeddings + ANN, recall@k, chunking, hybrid + rerank) · agents/tools/MCP · **evals & reliability — "eval is the new system design"** · AI system design + LLMOps.

**Frontier-lab flavour for contrast:** "build a small working system live" in 60 minutes with escalating requirements (versioned KV store, in-memory SQL-like DB, spreadsheet formula engine with cycle detection, resumable iterator, multithreaded crawler with dedup) — and **your own tests are graded**. That is not your round, but it explains why *testing discipline* is now scored everywhere.

**Building blocks to have in your fingers:** Pydantic v2 models with strict structured outputs; streaming client handling content deltas, tool-call argument completion, and final completion; async bounded concurrency (`asyncio.Semaphore`); retry/backoff; idempotency keys; pytest-based eval harness. (File `06` §5 has the code.)

---

## 4. Brit-specific intel (verified from public sources)

> This is the part most candidates will not have done. Use two or three of these facts and stop — reciting research reads as stalking.

### 4.1 Structure & ownership

- **100%-owned by Fairfax Financial Holdings** (TSX-listed, Canada).
- **From 1 Jan 2025, Brit and Ki separated** into independent Fairfax operations: **Brit Group Holdings** (lead underwriting) and **Ki Financial Ltd** — the Lloyd's digital **follow** syndicate (syndicate 1618, >$1bn GWP in 2024). Brit still provides admin services to Ki.
- Brit is a **nominated lead across all classes**; **Brit Re (Bermuda)** expanding.
- Context numbers (FY2025): PBT **$716.7m**; IPW **$3.09bn**; undiscounted **combined ratio 89.3%**; capital ratio 175.2%. H1 2026: PBT $326.8m; combined ratio 89.5%; capital ratio 197.9%.

### 4.2 The AI agenda — the facts to quote

| Fact | The line to use |
|---|---|
| Stated strategy: **"leveraging AI for smarter underwriting and simpler operations"** — literally the JD's *Smarter Underwriting* and *Operational Simplification* | *"The strategy is already public — my job is executing it, not inventing it."* |
| Four pillars: **Focus, Capability, Simplification, Culture** | *"Capability and Culture are the pillars the panel feedback maps to — mentoring and community."* |
| An **AI Lab** for smarter underwriting; a company-wide **AI literacy programme**; **AI Champions** appointed; **20 AI proposals** selected for development | *"Twenty proposals tells me the constraint isn't ideas — it's prioritisation and delivery capacity. That's a Lead problem."* |
| H1 2026: **AI automation and submission digitisation expanded across five classes — Cyber, Property Worldwide, Cargo, Terrorism, Energy** | *"Five classes live is a real portfolio, not a pilot."* |
| A **hub-and-spoke AI operating model**; an **AI risk/governance framework** | *"Hub-and-spoke means central standards with embedded delivery — exactly how I'd set architecture and eval standards."* |
| Three-pronged approach: **internal data-science team + off-the-shelf products + external AI vendor partnerships** | *"That's the build/buy/partner question, and the JD asks for build-or-buy judgement explicitly."* |
| CEO **Martin Thompson**: will "continue to invest heavily – not just in our tools, but in the **skills of our people**" | *"Skills investment is why the people-leadership half of this JD is not an afterthought."* |
| **Common/future underwriting platform**; a **new pricing and rating engine** extended across classes; the **cloud-native modern data platform** as the foundation | *"The platform layer is done; the AI layer sits on top."* |
| **Acturis** launch (~Mar 2026) for the SME cyber product **C360** — digital broker distribution | *"Their broker-integration story is Acturis-based, which tells me the ingestion surface is digital, not just email."* |

### 4.3 Claims AI — the showpiece, with details

- ML over **ultra-high-resolution aerial imagery** that **pinpoints and colour-codes properties by damage class within days**, enabling **proactive triage before claims are reported** — "virtual claims adjusting".
- Imagery partner: **Geospatial Insurance Consortium (GIC), since April 2019**.
- Timeline: Sept 2021 Hurricane Ida POC → 2021 US tornadoes → **Oct 2022 Hurricane Ian** (earliest payment **8 Oct 2022**, ~a week after landfall, adjusted from London).

**Why this matters for Part 3:** your image-anomaly design in file `05` §6.1 is this system. Say "*days*", say "*before claims are reported*", and say "*human adjuster still decides*" — those three phrases show you read their material.

### 4.4 The India GCC — the actual hiring hub

- **Brit India GCC, Bengaluru**, led by **Ravi Verma Vunikili, Head of India GCC since April 2025** (previously Head of Engineering India from Jan 2023; ex-Litera, Thomson Reuters, Oracle; M.Tech IIIT-B, B.E. BITS Pilani; publicly focused on **"AI-native SDLC transformation"** and scaling the India organisation from 0 to 200+).
- Team spans data engineering, platform, full-stack, data management/testing, **data science / AI / ML**, and actuarial analytics.
- Concurrent AI hiring: AI Solutions Engineer (Dec 2025), Data & AI Architect (Nov 2025), **Applied AI Scientist** (Aug 2026), Lead Product Manager (data & AI, London market).

**Use this:** *"The GCC mandate looks like it's moving from delivery capacity to capability-building — 'AI-native SDLC transformation' is a standards-and-culture job as much as a headcount one. That's the version of this role I want."*

### 4.5 ⚠️ One correction to make

The JD lists **Cytora** among "comparable products". Research found **no evidence of a Brit–Cytora deployment** — Brit's public broker-platform news is **Acturis**, and Cytora's public partnership news is with **Polaris (2019)**. **Do not claim Brit uses Cytora.** Say instead: *"the JD names Cytora alongside Copilot Studio as the low-code/comparable class of tooling, so I'd read that as the buy-side option to be evaluated."* Being precise about this is itself a credibility signal.

---

## 5. Insurance-AI market reality (LMA data — use these numbers)

**Adoption (LMA survey, 81 firms / 45 Lloyd's managing agents):**

- ~47% experimenting, ~40% live — **but only ~14% have deployed agentic/generative AI in underwriting**; 65% are not doing so in underwriting/claims; 12% have no plans.
- **Top use cases: data extraction from unstructured documents (74%), submission preparation (54%), claims triage / simple-task automation (~33%).**
- **Barriers: data quality/availability (49%), integration with existing systems (46%), cost / uncertain ROI (48%).**
- A second LMA survey (~60% of market capacity): **72% have an AI framework; 62% enforce mandatory human oversight.**
- LMA/Oxbow "enhanced underwriting" framing: **"augmented underwriting"** — the algorithm triages the submission, scores risk and gives insights while **the human underwriter stays central**.

**What to do with this:** *"So 74% of the market is chasing unstructured-data extraction and 46% are blocked on integration — which is the JD's 'business integration and service design' gap, stated as market data. My background is exactly that problem. And the 14% agentic-in-underwriting figure tells me the winning move is augmented underwriting with strong human oversight, not autonomy."*

**Also note the 2025 consolidation:** **Cytora was acquired by Applied Systems in 2025**, and **hyperexponential launched "Triage"** (Sept 2025), extending from pricing into the full submission-to-portfolio journey. The vendor landscape is consolidating around front-of-workflow submission handling — a useful Lead-level observation.

---

## 6. The regulatory frame, precisely (name the rules)

**There are no AI-specific FCA/PRA rules.** AI is governed through existing regimes — which is exactly what a good answer sounds like.

| Instrument | What it requires | Your line |
|---|---|---|
| **PRA SS1/23** (effective 17 May 2024; updated Apr 2026) | **Model risk management, five principles:** model identification/risk classification · governance · development/implementation/use · **independent validation** · risk mitigants. Formally for banks/IM firms, **applied by analogy to insurers**. Consultation feedback flags **AI/ML explainability difficulty and fairness/bias risk**. | *"I'd treat the model as a controlled artefact with independent validation and a named owner — that's SS1/23's structure applied by analogy."* |
| **FCA Consumer Duty (PS22/9)** | Fully applies to automated decisions — **"the machine did it" is not a defence.** Explainability must be **accurate, actionable and understandable to the customer** ("reason why" obligation, distinct from GDPR Art. 22). **Critically: SHAP/LIME/feature-importance do NOT satisfy regulatory explainability.** | *"Feature importance is not an explanation. A 'reason why' is: in plain language, which facts drove the outcome and what the customer can do about it."* |
| **SM&CR** | Named accountability for the decision | *"There's a named human accountable, even when the model proposes."* |
| **SS2/21** | Outsourcing / third-party risk — covers cloud AI providers | *"A third-party model provider is an outsourced dependency and needs the same due diligence."* |
| **LMA AI Adoption Toolkit** (with Barnett Waddingham) | Five principles: governance & accountability · **risk tiering** · data protection/security/IP · training & awareness · pragmatic adoption. Its **AI Risk Tiering Template classifies "automated underwriting decisions" as HIGH risk** → independent validation, fairness/bias assessment, formal sign-off | *"The market toolkit already classifies automated underwriting as high risk, which tells me the default design is human-in-the-loop with independent validation."* |
| **Lloyd's Futureset (Mar 2024)** | Warns that firms using LLMs for algorithmic underwriting "may **inadvertently correlate** themselves due to inherent architectural inductive bias" | *"Model monoculture is a systemic risk — if everyone uses the same model on the same data, the market correlates. That's an argument for diversity and for monitoring."* |
| **Solvency II** | Capital and model governance | *"If a model touches pricing or reserving, Solvency II governance applies."* |
| **UK GDPR / DPA 2018** | Lawful basis, minimisation; **Art. 22** restricts solely-automated decisions with significant effect | *"Solely-automated decision-making with legal effect needs a legal basis and safeguards — which is another reason for the human gate."* |

**Two premium lines from this section:**
1. *"SHAP and LIME are not regulatory explainability — they're diagnostic tools. A regulated explanation is a plain-language 'reason why' with the facts that drove it and the customer's remedy."*
2. *"Lloyd's own warning is that if the whole market uses the same LLM architecture, the market correlates — so model monitoring isn't just about accuracy, it's about systemic concentration."*

---

## 7. London Market vocabulary — the deep layer

Beyond file `11` §1 (which covers the base terms), these are the platform and process names that signal genuine market literacy. **Use at most two.**

| Term | What it is |
|---|---|
| **Blueprint Two** | Lloyd's market modernisation programme |
| **CDR (Core Data Record)** — v3.2 | The market's data standard; CDR API allows broker direct-submit or extract-from-iMRC |
| **ACORD 2016.10** | The message standard: **EBOT** (electronic placing) and **ECOT** (electronic claims) |
| **Digital Gateway / IROS / IPOS / ICOS** | The market's replacing platforms (IPOS replaces POSH/LIDS; ICOS replaces ECF/CLASS) |
| **PPL / Whitespace / Touchstone / Acturis** | Placement platforms — PPL (Lloyd's), Whitespace (London market), Touchstone (Lloyd's), Acturis (broker distribution) |
| **MRC v3** | The Market Reform Contract — the modern slip |
| **Line slip** | A facility allowing a risk to be placed with multiple syndicates; **bulking vs non-bulking** |
| **Binding authority** | Written authority from a managing agent to a **coverholder** to bind risk |
| **ATLAS / BAR** | Lloyd's coverholder registration systems |
| **Bordereau** | Risk or premium schedule reported up from a coverholder (monthly/quarterly, or block) |
| **ECF / BDX** | Claims file / claims data exchange |
| **LPAN / FDO** | London Premium Advice Note / Firm Order |
| **"Stamp" capacity** | Lloyd's capacity, expressed in market share terms |
| **Cat modelling: RMS / AIR (Moody's) / Verisk** | The catastrophe models underpinning pricing and accumulation |
| **Moratorium** | An insurer's temporary halt on binding in an affected area post-catastrophe — **CAPE Analytics' post-cat imagery exists specifically to avoid blanket moratoria** |

**The one-liner if it fits:** *"Brit's Broker API work sits inside Blueprint Two and the Core Data Record — so the bulk-ingestion problem is a market-standard data problem, not just a PDF problem."*

> ⚠️ **Flagged gap:** the research could not confirm Blueprint Two / CDR 2025–2026 go-live status from official docs (they are weighted to 2022–2024 planning). Treat CDR v3.2 as the current spec and don't assert rollout dates.

---

## 8. Vendor landscape — corrected

| Vendor | What it does | Note |
|---|---|---|
| **Cytora** | Submission triage + data enrichment (LLM extraction + external data: firmographics, property, climate). Customers: Beazley, Allianz, Markel, Ecclesiastical | **Acquired by Applied Systems in 2025** |
| **hyperexponential (hx)** | Pricing/underwriting platform for commercial P&C — carriers process $45–50bn+ GWP; **launched "Triage" (Sept 2025)** plus Portfolio Intelligence and AI Agents | Aviva expanded partnership (Dec 2025) — 23 live models |
| **Send** | London InsurTech **underwriting workbench** (SaaS for insurers/MGAs); integrated with hx Renew (2024) | — |
| **Federato** | AI-native insurance decisioning platform; customers incl. Nationwide, QBE | — |
| **Others worth naming once** | **Kalepa** (underwriting decisioning) · **Tractable** (AI damage appraisal) · **CAPE Analytics / Arturo / Verisk / EagleView** (property imagery) · **Sixfold, Root** (Lloyd's Lab alumni) | See §9 for the imagery pattern |

---

## 9. The post-catastrophe imagery pattern (industry-wide — so you design it right)

Brit is not alone here, which is useful: it means there is an established shape.

| Player | What they shipped | The design lesson |
|---|---|---|
| **Verisk XactAI** (Sept 2025) | Note summarisation, auto photo labelling, receipt categorisation — **all human-oversight-gated** | Even the vendor ships HITL, not autonomy |
| **Verisk + EagleView** (Feb 2025) | Drone/AI damage detection into Xactimate/XactAnalysis; 3bn+ imagery library; 94% US coverage | Imagery at scale is a *stitching and georeferencing* problem |
| **CAPE Analytics (Moody's)** Event Damage for Underwriting (Oct 2025) | Post-cat "**gray sky**" imagery, **4-level damage severity**, **reason codes**, pre/post side-by-side | **Severity levels + reason codes** is exactly the explainability pattern you should propose |
| **Arturo** | Roof-condition models, pre/post-storm triage, flood footprint, fraud via prior-condition comparison | The **fraud** use case (comparing prior condition) is a clever one to mention |

**Two phrases to lift for your design answer:** **"gray sky vs blue sky imagery"** (post-event vs pristine baseline) and **"severity class + reason code per property"**. Both make the design sound informed rather than invented.

---

## 10. Azure stack — the 2025/26 specifics worth knowing

| Service | The detail that matters |
|---|---|
| **Azure AI Foundry** | Unified model catalogue, **agent service**, evaluations, project management. The enterprise pattern: **Foundry = AI engine, Copilot Studio = delivery layer** |
| **Foundry IQ** (new) | Managed knowledge layer producing **permission-aware knowledge bases** for agents, built on Azure AI Search; **ACL sync + Purview sensitivity-label enforcement**; sources include Blob/SharePoint/OneLake |
| **Azure AI Search — agentic retrieval** | LLM **query planning** → parallel subqueries → semantic rerank → merged grounding with citations. Ranking levels: L1 (BM25+HNSW) → Fused L1 (RRF) → **L2 semantic ranker** → L3 iterative agentic search. **Semantic ranker is now required for agentic retrieval**; set `k=50` for max semantic-ranker input. **Billing shifts from per-query to per-token for agentic retrieval** |
| **Document Intelligence** | `prebuilt-layout` v4.0 (API `2024-11-30`): up to **2,000 pages/file** (PDF+TIFF), selective `pages` param, paragraph **roles** (title/sectionHeading/pageHeader/pageFooter/pageNumber), tables, selection marks, figures, hierarchical sections, **Markdown output for RAG**. Plus **custom classifier** for document-type + page-range splitting, and custom neural/template models. Best-practice pipeline: **ingest → classify+split → layout parse → extract → store with source links + HITL fallback** |
| **Copilot Studio** | Fastest time-to-value, native Teams/SharePoint/Dynamics, days not weeks — but **locked to Microsoft-managed OpenAI models**, opaque/non-configurable RAG, and expensive at scale | 
| **Azure ML** | Full MLOps: pipelines, registries, managed online endpoints, responsible-AI dashboard, **model monitoring / drift** |
| **Content Safety** | Harm classifiers, **prompt-shield for indirect prompt injection**, groundedness detection; **Azure AI Evaluation SDK** for RAG/agent metrics |
| **Security plumbing** | **Private endpoints + VNet integration**, **managed identity** (zero stored credentials — Azure's genuine advantage over AWS IAM/STS and GCP WIF), Entra ID RBAC, Key Vault, customer-managed keys, Purview, Defender for AI |

**Azure's honest weaknesses (say one, it shows judgement):** thinner non-OpenAI model catalogue than Bedrock; agent framework less mature than Bedrock AgentCore; a "compliance premium" at scale on GPT pricing; complex consumption-based billing; lock-in.

**The mitigation line:** *"In a multi-provider world I'd put an **AI gateway** in front — LiteLLM, Kong or Portkey — on day one, so provider choice is a config change and not a rewrite. That also gives me per-tenant token accounting and failover in one place."*

---

## 11. LLM serving & quantisation numbers (the ones worth quoting)

**KV cache formula — know this cold:**

```
KV_bytes/token = 2 × layers × KV_heads × head_dim × dtype_bytes
GPU = weights + activations + KV cache + ~1–2 GB framework overhead
```

| Model (70B class) | KV per token | Factor |
|---|---|---|
| MHA BF16 | 2.5 MB | 1× |
| **GQA BF16** | **0.320 MB** | **8×** |
| **GQA + FP8 KV** | **0.160 MB** | **16×** (<0.3% quality loss) |
| MLA (DeepSeek-V3 style) | 0.00113 MB | 57× |

- **H100 capacity (FP8 weights, TP=2, ~40 GB KV headroom):** 4K ctx → ~62 users; 8K → ~31; 32K → ~8; 128K → **~1–2**.
- **Decode is memory-bandwidth-bound:** a 70B FP8 on 2×H100 ≈ **10.4 ms/token → ~96 tok/s single-stream ceiling**. Batching to 64 raises aggregate throughput to ~3,855 tok/s but worsens ITL.
- **PagedAttention (vLLM default block=16)** reclaims the **60–80%** HBM wasted by contiguous pre-allocation. **Prefix/RadixAttention at 80% hit → ~71% fewer prefill tokens → ~3.5× better TTFT.**

**Quantisation accuracy (EMNLP 2025; 5 methods × 5 models, avg drop vs BF16):**

| Method | Avg quality drop |
|---|---|
| **FP8 W8A8** | **−0.2%** |
| GPTQ int8 | −0.8% |
| **AWQ int4** | **−1.8%** |
| GPTQ int4 | −2.7% |
| **BNB-nf4** (HF/vLLM 4-bit default) | **−6.9%** |

- Degradation **worsens with context**: 4-bit up to **−23% at 128K**; worst case **−59%** (Llama-3.1-70B, BNB-nf4, long-context). Non-English degrades up to **5× more**.
- **AWQ generally beats GPTQ** on accuracy. A 4-bit *larger* model can beat a smaller FP16 model. **Activation quantisation (W4A4) costs ~30% on multi-step reasoning** vs ~4% weight-only — *reasoning is far more quant-sensitive than perplexity suggests.* That is the line to quote.

**Embeddings:**

| Model | Dims | Max tokens | Note |
|---|---|---|---|
| text-embedding-3-large | 3,072 (truncatable to 256/1,536) | 8,191 | Matryoshka — ~92–95% quality retained at 256 dims |
| text-embedding-3-small | 1,536 | 8,191 | cheap default |
| Cohere embed-v4 | 1,024 | 128K | multimodal, 100+ languages |
| **BAAI BGE-M3** | 1,024 | 8,192 | **hybrid dense + sparse + ColBERT, self-hostable on one A10G** |
| all-MiniLM-L6-v2 | 384 | 256 | English, CPU/edge, prototype |

- **MTEB retrieval scores cluster tightly** (OpenAI 0.685, Cohere v4 0.672, BGE-M3 0.661) — *"a 2-point MTEB gap rarely matters on your own domain; I'd benchmark on our data."*
- **Dimensions sweet spot 768–1,024.** 384→1,024 gains ~5–10%; 1,024→3,072 only ~1–3%. Storage: 1M vectors @1,024 float32 ≈ **4.1 GB**.
- Never mix query/document prefixes; normalising open-source embeddings is required; switching models requires a **full re-embed** and a dual-index migration.

**Chunking (empirical benchmarking):** quality **plateaus by ~32 tokens** and **collapses below ~16**; semantic chunking beats fixed-size by ~0.09 nDCG at small sizes. Practical default **256–512 tokens**.

**Retrieval:** two-stage (fast ANN → cross-encoder rerank) improves **precision@1 by 5–15%** for ~100 ms. **Add a reranker when P@1 < 0.85.**

**Semantic caching (2025 research):**
- Redis + HNSW, cosine threshold **0.8**: **up to 68.8% fewer API calls**, hit rate 61.6–68.8%, positive-hit accuracy >97%.
- **vCache (Berkeley):** per-embedding **learned** thresholds — up to **12.5× higher hit rate and 26× lower error** than static-threshold GPTCache, ~1.5 ms overhead.
- **~31% of ChatGPT queries are semantically similar to a prior one.**
- **The trend to state:** *"static cosine thresholds are unreliable — the state of the art is a learned, per-embedding threshold, and always verify the cached answer against the source before serving it."*

**Serving engine comparison:** vLLM = peak throughput, tail stability, structured outputs (wins single-request TTFT on 15/16 models). **SGLang = low-concurrency TTFT and prefix reuse (RadixAttention)** — one benchmark reports **583 ms vs vLLM's 2,141 ms TTFT at concurrency 1** on an 8B model, but vLLM wins peak throughput (+11%) and 8K-context latency. **TensorRT-LLM = raw throughput and long-context prefill.** TGI = deployment simplicity. *[Benchmark figures are indicative — different harnesses disagree.]*

---

## 12. Evaluation tooling — the names to use

| Tool | What it gives | The specific thing to mention |
|---|---|---|
| **LangSmith** | `agentevals` package — `create_trajectory_match_evaluator` with **four modes: strict (exact order), unordered, subset (no extra tools), superset (at least the reference tools)**; plus an LLM-as-judge trajectory evaluator | *"Four trajectory-matching modes — I'd use `subset` for 'must not call anything extra' and `strict` for regulated flows."* |
| **Arize Phoenix** | Built-in **`ToolSelectionEvaluator`** and **`ToolInvocationEvaluator`**; a **convergence/path evaluator scoring `optimal_path_length / actual_path_length`**; agent-as-a-judge for multi-step rubrics | *"Path-efficiency as a number — optimal steps over actual steps."* |
| **Langfuse** | LLM-as-judge on **observations** (recommended for live prod) vs **experiments** (offline datasets); CI/CD regression gating via a GitHub Action that raises on threshold breach; trace-level evaluators deprecated in v4 → observations-first | *"Judge the observation span in production; keep the dataset experiment for the gate."* |
| **RAGAS** | faithfulness, answer relevancy, context precision, context recall, answer correctness | The default retrieval/generation metric suite |
| **DeepEval** | RAGAS wrappers plus native metrics, **pytest integration (`assert_test`)**, `@observe` tracing | Eval assertions inside your test suite |
| **Judge validation** | Split human-labelled traces into dev/test; require **TPR and TNR > 90%** before trusting the judge | *"I don't ship a judge until it hits 90% true-positive and true-negative rate on held-out human labels."* |

**The failure taxonomy worth quoting (tau-bench, 60 tasks / 2,320 annotations):** naive agents scored **0% exact match** (over-calling lookup tools); a trained trajectory agent hit **80%**. Failures clustered as **missing tool (19), redundant call (15), tool error mishandled (15)**; annotations were **27% catastrophic, 27% suboptimal, 47% clean**.

**And the portability warning (a genuinely senior observation):** "faithfulness" is **not portable between frameworks — on neutral claims DeepEval scored F1 0.98 vs RAGAS 0.83 because their definitions differ (doesn't-contradict vs directly-inferrable). *"So I validate any judge against my own labelled data rather than trusting a metric's name."*

---

## 13. Sources

**Loops & lead-level expectations**
- interviewnode.com — "The AI Hiring Loop: How companies evaluate you across multiple rounds"
- myengineeringpath.dev — Staff AI Engineer guide · growthalgorithm.dev — tech lead vs EM
- github.com/AdilShamim8/GenAI-Roadmap-with-Notes-and-Projects — `field-guide/03-interview-loop-anatomy.md`
- interviewlegend.com — OpenAI guide

**Real questions & question banks**
- prachub.com — "Design a low-latency RAG system", "Design and optimize a RAG system", "Build a Mistral-powered RAG agent"
- github.com/ombharatiya/ai-system-design-guide — `00-interview-prep/01-question-bank.md` (128-question bank)
- github.com/alexeygrigorev/ai-engineering-field-guide — `interview/questions/04-ai-system-design.md`
- github.com/landedjobs/awesome-ai-engineer-interview
- dataford.io — Reddit GenAI Engineer interview guide

**Evaluation & observability**
- docs.langchain.com/langsmith/trajectory-evals
- arize.com/docs — agent trajectory evaluations · arize commit PR #12369 (tau-bench, 60 tasks / 2,320 annotations)
- langfuse.com/blog/2025-09-05-automated-evaluations · langfuse.com/docs/evaluation/overview
- thecodexandthecompass.substack.com — "Faithfulness Metrics Under the Microscope"

**Insurance / Lloyd's / regulation**
- lmalloyds.com — AI Adoption Toolkit · "The Growth of Enhanced Underwriting in the Lloyd's Market"
- bankofengland.co.uk — PRA SS1/23 Model Risk Management Principles
- tlt.com — "Automated Decisions, Human Consequences: the AI governance imperative for insurance"
- lloyds.com — Core Data Record · Delegated Authorities market knowledge
- hyperexponential.com — Triage launch announcement

**Brit**
- britinsurance.com — 2025 annual results · H1 2026 interim results · "ML innovation in claims" · "Brit uses aerial imagery to expedite Hurricane Ian claims payments" · brit-group/brit-india
- linkedin.com — Ravi Verma Vunikili, Head of Brit India

**Azure**
- learn.microsoft.com — Azure AI Search agentic retrieval overview · Foundry IQ · Document Intelligence `prebuilt-layout` (doc-intel-4.0.0)

**Serving, quantisation, embeddings, caching**
- rawlinson.ca — vLLM vs SGLang H100 benchmark
- aclanthology.org/2025.emnlp-main.479.pdf — quantisation accuracy study
- arxiv.org/html/2502.03771 — vCache · arxiv.org/html/2607.23507 — embedding framework
- vvinjamu.github.io/silicon-to-scale — KV cache & sizing · primer.edge.bond — LLM systems design

**Git**
- git-automation.com — merge-vs-rebase decision matrix · moderngitacademy.com — rebase vs merge

### 13.1 What the research could *not* confirm (do not assert these)
1. **No evidence of a Brit–Cytora deployment** (see §4.5).
2. **Blueprint Two / CDR 2025–2026 go-live status** — official docs are 2022–2024-weighted.
3. **Absolute engine throughput/latency and Azure-vs-AWS cost comparisons** — several sources are future-dated or synthesised; treat as **[indicative]** only.
4. **Raw Glassdoor/Blind verbatim reports** were thin — most retrievable content is aggregated prep sites. The strongest sources are the GitHub field guides and Reddit-derived summaries.
5. **No dedicated Brit tech blog** — tech detail comes from annual/interim reports and newsroom posts.

---

## 14. Cheat-sheet summary

- **Loops are 5–7 stages; system design carries the most weight; behavioural is 30–40% at senior level.**
- **The #1 documented rejection reason is *no evals*.** Say "eval" unprompted, in every round.
- **Answer patterns rewarded:** offline/online plane split · hybrid + RRF + rerank · **filter before retrieval** · forced citations + **abstention ladder** · stage-by-stage eval · graceful degradation.
- **Brit intel (use three):** "smarter underwriting and simpler operations" · **AI Lab + 20 proposals + AI Champions + AI literacy** · **five classes live (Cyber, Property Worldwide, Cargo, Terrorism, Energy)** · **hub-and-spoke with an AI risk framework** · claims ML over **GIC aerial imagery**, damage-class colour-coding **within days, before claims are reported** · GCC in **Bengaluru under Ravi Verma Vunikili**, "AI-native SDLC transformation".
- **Market data:** 74% of the market is chasing unstructured-data extraction; 46% blocked on integration; only ~14% agentic in underwriting; **62% enforce mandatory human oversight**.
- **Regulation:** SS1/23 model risk (5 principles) · Consumer Duty — **"the machine did it" is not a defence**, and **SHAP/LIME ≠ regulatory explainability** · LMA toolkit tiers **automated underwriting as HIGH risk** · Lloyd's correlation warning.
- **Serving:** GQA = 8× KV reduction, GQA+FP8 = 16×; FP8 −0.2%, AWQ-int4 −1.8%, BNB-nf4 −6.9%; **reasoning is far more quant-sensitive than perplexity suggests**.
- **Eval tooling:** LangSmith trajectory modes · Phoenix tool-selection + path-length · Langfuse observations · **TPR/TNR > 90% before trusting a judge**.
- **Never assert:** Brit–Cytora, CDR go-live dates, or exact benchmark figures.

## 15. Cross-links

- Domain vocabulary and the base primer: [`11-insurance-lloyds-domain-primer.md`](11-insurance-lloyds-domain-primer.md)
- Where these facts go in the interview: [`01`](01-brit-jd-decode-and-role-fit.md) · [`02`](02-stage1-format-60-minute-battle-plan.md) · [`05`](05-system-design-round-playbook.md) §6
- Evaluation tooling in context: [`07-agent-evaluation-and-guardrails.md`](07-agent-evaluation-and-guardrails.md)
- Index: [`00-README-and-7-day-plan.md`](00-README-and-7-day-plan.md)
