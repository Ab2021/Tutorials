# 11 — INSURANCE & LLOYD'S DOMAIN PRIMER (plus the Azure stack)

> **TL;DR.** You will be interviewed by people who live in the Lloyd's market. Using retail/tech vocabulary ("customer ticket", "users") instead of market vocabulary ("submission", "binding", "slip") reads as "will need six months of ramp". This file gives you: Lloyd's in ten minutes, the underwriting workflow end to end, where AI actually helps, the regulatory frame (FCA/PRA/Lloyd's, Consumer Duty, model risk), the vendor landscape, and the Azure + low-code stack the JD names.

**60-second interview answer (if asked "what do you know about Brit's business?"):**

> *"Brit is a specialty insurer and reinsurer at Lloyd's, part of Fairfax. Lloyd's is a market, not a company — risk is placed with syndicates through brokers. Brit's own agenda is three things: a cloud-native data platform they've already built, a headless broker API that takes bulk submissions in, and an Underwriting Transformation that puts a new platform and workbench in front of underwriters. Their applied-AI showpiece is computer vision for post-catastrophe property damage — the first virtual claims adjusting enabler in the London market. So the AI role sits on top of an existing data platform, and the problems are document-heavy, extraction-heavy, and heavily regulated."*

---

## 1. Lloyd's in ten minutes

| Term | What it means | Why it matters to you |
|---|---|---|
| **Lloyd's of London** | A **marketplace**, not an insurer. It provides the building, capital rules and licences;**syndicates** underwrite risk inside it | Never say "Lloyd's customers" — the market has *members*, and business flows through brokers |
| **Syndicate** | A group of capital providers underwriting via a managing agent | Brit operates as a syndicate (and an MGA/company market arm) |
| **Managing agent** | The firm that runs a syndicate day to day | Where the underwriters and ops teams sit |
| **Broker** | Intermediary who places risk with syndicates — the customer *interface* | Submissions arrive **from brokers**, which is why Broker APIs matter |
| **MGA / coverholder** | A firm with **delegated authority** to underwrite on behalf of capacity | Delegated authority means rules, bordereaux and audit |
| **Submission** | The package a broker sends: the risk details plus supporting documents | The atomic unit of the ingestion problem |
| **Slip** | The document recording the terms of the placed risk | Document AI target |
| **Quote → Bind** | Offer the terms → accept them, creating a contract | "Binding risk" is the irreversible moment |
| **Endorsement** | A mid-term change to an in-force policy | A frequent, structured-ish document task |
| **Claim** | Loss notification and the path to payment | Where Brit's computer-vision damage detection lives |
| **Bordereau / bordereaux** | A schedule of risks or premiums reported up from a coverholder | A data-ingestion problem, often ugly |
| **Appetite** | The classes of business and the characteristics a syndicate wants | The first question for every submission: in or out of appetite? |
| **Line / capacity** | How much of a risk the syndicate takes, and how much capital backs it | Sizing and pricing context |
| **Catastrophe ("cat") event** | A large loss event — hurricane, flood, wildfire | The trigger for Brit's imagery pipeline |
| **Reinsurance** | Insurance for insurers: **treaty** (portfolio-wide) or **facultative** (single risk) | Brit is also a reinsurer |
| **Combined ratio** | Loss ratio + expense ratio; under 100% = underwriting profit | The single business metric an insurer cares most about |
| **Loss ratio / expense ratio** | Claims paid ÷ premium; operating cost ÷ premium | Where AI savings and better risk selection show up |

**The one-sentence model of the business:** *"Specialty insurance is a judgement business operating on thin margins, so anything that speeds up the judgement or improves risk selection shows up directly in the combined ratio."*

### 1.1 The deep layer (use at most two of these — they signal real market literacy)

| Term | What it is |
|---|---|
| **Blueprint Two** | Lloyd's market modernisation programme |
| **CDR (Core Data Record)** — v3.2 | The market's data standard; the CDR API allows broker direct-submit or extract-from-iMRC |
| **ACORD 2016.10** | The message standard: **EBOT** (electronic placing) and **ECOT** (electronic claims) |
| **Digital Gateway / IROS / IPOS / ICOS** | The market's replacing platforms (IPOS replaces POSH/LIDS; ICOS replaces ECF/CLASS) |
| **PPL / Whitespace / Touchstone** | Placement platforms — PPL (Lloyd's), Whitespace (London market), Touchstone (Lloyd's) |
| **MRC v3** | The Market Reform Contract — the modern slip |
| **Line slip** | A facility placing a risk across multiple syndicates; **bulking vs non-bulking** |
| **Binding authority** | Written authority from a managing agent to a **coverholder** to bind risk |
| **ATLAS / BAR** | Lloyd's coverholder registration systems |
| **Bordereau** | Risk or premium schedule reported up from a coverholder (monthly/quarterly, or block) |
| **ECF / BDX** | Claims file / claims data exchange |
| **LPAN / FDO** | London Premium Advice Note / Firm Order |
| **Cat models: RMS / AIR (Moody's) / Verisk** | The catastrophe models underpinning pricing and accumulation |
| **Moratorium** | An insurer's temporary halt on binding in an affected area post-catastrophe. **CAPE's post-cat imagery exists specifically to avoid blanket moratoria** — a neat way to link the claims pipeline to underwriting |

**The one-liner if it fits:** *"Brit's Broker API work sits inside Blueprint Two and the Core Data Record — so bulk ingestion is a market-standard data problem, not just a PDF problem."*

> ⚠️ **Do not assert Blueprint Two / CDR go-live dates** — official docs are weighted to 2022–2024 planning. Treat CDR v3.2 as the current spec and leave dates alone.

---

## 2. The underwriting workflow — and where AI touches it

```
Broker → Submission (email + PDFs + spreadsheets)
   → INGESTION        ← document AI, bulk broker-API ingestion
   → TRIAGE           ← is this in appetite? routing
   → DATA EXTRACTION  ← schedules, sums insured, addresses, prior losses
   → ENRICHMENT       ← geocoding, sanctions/PEP checks, cat modelling, prior-year data
   → PRICING          ← models + underwriter judgement
   → QUOTE → negotiation → BIND
   → ENDORSEMENTS     ← mid-term changes
   → CLAIMS           ← notification, adjustment, payment
```

**The AI opportunities, in the order a Lead would rank them:**

| Stage | The AI task | The hard part |
|---|---|---|
| **Ingestion** | Multi-page PDF and email extraction to structured fields | Layout variance, tables, handwriting; every field must be traceable to a page |
| **Triage** | In-appetite classification and routing | Cost asymmetry (a wrong "out of appetite" loses the broker relationship) |
| **Extraction** | Schedules, sums insured, locations, prior loss history | Deterministic validation; a wrong number is worse than a missing one |
| **Enrichment** | Geocoding, sanctions/PEP screening assistance | Sanctions are a hard control — assist, never automate |
| **Pricing support** | Comparables retrieval, similar-risk lookup | Explainability; an underwriter must justify the number |
| **Claims** | Post-cat damage detection from imagery | Bursty volume, imbalanced classes, false-positive cost |
| **Operational simplification** | Email triage, document chasing, summarisation, drafting | Adoption — the underwriter must trust it |

**The line that lands:** *"The model is maybe 10% of it. The value is in the ingestion-to-triage path, where the underwriter currently spends minutes per submission deciding things a well-grounded extraction could pre-populate."*

---

## 3. Where Brit specifically is

### 3.1 The JD-facing facts

| Brit fact (from the JD and public positioning) | What you should say about it |
|---|---|
| **Cloud-native data platform** — "central, trusted, cost-efficient, flexible, scalable" | *"They've done the plumbing. My job would be the AI layer on top, not re-platforming."* |
| **Headless broker API** — multi-class-of-business, connects pricing and document generation to broker platforms and market hubs, enables **bulk submission ingestion** | *"Bulk ingestion changes the design — it's a queue and backpressure problem, not a synchronous API problem."* |
| **Underwriting Transformation** — pricing + **underwriter workbench**, "engineering, upskilling in python" | *"Upskilling an underwriting organisation in Python means the AI team also has a teaching mandate — mentoring is genuinely part of this role."* |
| **Post-catastrophe property damage ML** from ultra-high-resolution imagery — "first enabler for virtual claims adjusting in the London market" | *"That's a production computer-vision pipeline with a human adjuster in the loop. Expect design questions in that shape."* |
| **Azure AI, Copilot Studio, Power Platform/Power Apps, Cytora** named in the JD | *"Azure-native is the default; low-code is preferred where the capability isn't a differentiator."* — **but see the Cytora correction in §7.1** |

### 3.2 Structure and ownership (verified)

- **100%-owned by Fairfax Financial Holdings** (TSX-listed, Canada).
- **From 1 Jan 2025 Brit and Ki separated** into independent Fairfax operations: **Brit Group Holdings** (lead underwriting) and **Ki Financial Ltd** — the Lloyd's **digital follow syndicate** (syndicate 1618, >$1bn GWP in 2024). Brit still provides admin services to Ki.
- Brit is a **nominated lead across all classes**; **Brit Re (Bermuda)** is expanding.
- Scale context (FY2025): PBT **$716.7m**; IPW **$3.09bn**; undiscounted **combined ratio 89.3%**; capital ratio 175.2%. H1 2026: PBT $326.8m; combined ratio 89.5%; capital ratio 197.9%.

**Why it matters:** *"Brit and Ki are genuinely separate businesses now, so 'how does AI serve both, and where does the shared platform stop?' is a real architecture question, and a fair one to ask them."*

### 3.3 The AI agenda — the facts worth quoting

| Fact | The line to use |
|---|---|
| Stated strategy: **"leveraging AI for smarter underwriting and simpler operations"** — literally the JD's *Smarter Underwriting* and *Operational Simplification* | *"The strategy is already public — my job is executing it, not inventing it."* |
| Four pillars: **Focus · Capability · Simplification · Culture** | *"Capability and Culture are the pillars the panel feedback maps to — mentoring and community."* |
| An **AI Lab** for smarter underwriting; a company-wide **AI literacy programme**; **AI Champions** appointed; **20 AI proposals** selected for development | *"Twenty proposals tells me the constraint isn't ideas — it's prioritisation and delivery capacity. That's the Lead problem."* |
| H1 2026: **AI automation and submission digitisation expanded across five classes — Cyber, Property Worldwide, Cargo, Terrorism, Energy** | *"Five classes live is a real portfolio, not a pilot."* |
| A **hub-and-spoke AI operating model**, plus an **AI risk/governance framework** | *"Hub-and-spoke means central standards with embedded delivery — exactly how I'd set architecture and eval standards."* |
| Three-pronged approach: **internal data-science team + off-the-shelf products + external AI vendor partnerships** | *"That's the build/buy/partner question, and the JD asks for that judgement explicitly."* |
| CEO **Martin Thompson**: will "continue to invest heavily – not just in our tools, but in the **skills of our people**" | *"Skills investment is why the people-leadership half of this JD is not an afterthought."* |
| A **common/future underwriting platform**; a **new pricing and rating engine** extended across classes; the cloud-native modern data platform as the foundation | *"The platform layer is done; the AI layer sits on top."* |
| **Acturis** launch (~Mar 2026) for the SME cyber product **C360** — digital broker distribution | *"Their broker-integration story is Acturis-based, so the ingestion surface is digital, not just email."* |

### 3.4 The claims-AI showpiece, with detail

- ML over **ultra-high-resolution aerial imagery** that **pinpoints and colour-codes properties by damage class within days**, enabling **proactive triage before claims are reported** — "virtual claims adjusting".
- Imagery partner: **Geospatial Insurance Consortium (GIC), since April 2019**.
- Timeline: Sept 2021 Hurricane Ida POC → 2021 US tornadoes → **Oct 2022 Hurricane Ian** — earliest payment **8 Oct 2022**, roughly a week after landfall, adjusted from London.

**The three phrases to use:** "*within days*", "*before claims are reported*", and "*the adjuster still decides*". Those signal you have read their material — and they are exactly what the design in [`05`](05-system-design-round-playbook.md) §6.1 reproduces.

### 3.5 The India GCC — likely where this role sits

- **Brit India GCC, Bengaluru**, led by **Ravi Verma Vunikili, Head of India GCC since April 2025** (Head of Engineering India from Jan 2023; ex-Litera / Thomson Reuters / Oracle; M.Tech IIIT-B, B.E. BITS Pilani; publicly focused on **"AI-native SDLC transformation"**, scaling the India organisation from 0 to 200+).
- Team spans data engineering, platform, full-stack, data management/testing, **data science / AI / ML**, and actuarial analytics.
- Concurrent AI hiring: AI Solutions Engineer (Dec 2025), Data & AI Architect (Nov 2025), **Applied AI Scientist** (Aug 2026), Lead Product Manager (data & AI, London market).

**The line to use:** *"The GCC mandate reads as a move from delivery capacity to capability-building — 'AI-native SDLC transformation' is a standards-and-culture job as much as a headcount one. That's the version of this role I want."* — and ask your question 4 in [`13`](13-final-48-hours-cheat-sheet.md) §12 off the back of it.

---

## 4. The regulatory frame (name these confidently)

| Regime / body | What it means for AI |
|---|---|
| **FCA** (Financial Conduct Authority) | Conduct regulator. **Consumer Duty** = act to deliver good outcomes for retail customers — applies to any AI touching a customer outcome. |
| **PRA** (Prudential Regulation Authority) | Prudential regulator. **SS1/23** sets expectations for **model risk management** — model identification, governance, validation, monitoring, documentation. |
| **Lloyd's** | Market oversight: delegated authority rules, conduct, and performance management of syndicates. |
| **UK GDPR / DPA 2018** | Lawful basis, data minimisation, subject rights. **Automated decision-making** has specific rules where decisions are solely automated. |
| **Solvency II** | Capital adequacy; data quality and model governance obligations that flow into models used for pricing/reserving. |
| **Sanctions / AML / KYC** | Screening is a hard control; AI can assist triage but a human owns the screening decision. |
| **Data residency** | UK/EU data may not be shipped to arbitrary third-party providers without a legal basis; this drives private endpoints and sometimes self-hosting. |
| **LMA AI Adoption Toolkit** (with Barnett Waddingham) | Five principles: governance & accountability · **risk tiering** · data protection/security/IP · training & awareness · pragmatic adoption. Its **AI Risk Tiering Template classifies "automated underwriting decisions" as HIGH risk** — requiring independent validation, fairness/bias assessment and formal sign-off. |
| **Lloyd's Futureset (Mar 2024)** | Warns that firms using LLMs for algorithmic underwriting "may **inadvertently correlate** themselves due to inherent architectural inductive bias" — model monoculture is a systemic market risk. |

**There are no AI-specific FCA/PRA rules.** AI is governed through the existing regimes above — which is exactly what a good answer sounds like.

### 4.1 The two most quotable regulatory points

1. **Consumer Duty and explainability.** Consumer Duty **fully applies to automated decisions — "the machine did it" is not a defence**, and the explanation must be *accurate, actionable and understandable to the customer* (the **"reason why"** obligation). Critically: **SHAP, LIME and feature-importance do NOT satisfy regulatory explainability.**
   > *"Feature importance is a diagnostic tool, not an explanation. A regulated 'reason why' is plain language — which facts drove the outcome, and what the customer can do about it."*
2. **The correlation warning.** Because the whole market tends to reach for the same models, architecture-induced bias can correlate underwriters.
   > *"Lloyd's own point is that if everyone uses the same LLM architecture on the same data, the market correlates — so model monitoring is a concentration-risk control, not just an accuracy one."*

### 4.2 The market-adoption picture (LMA survey — use one number)

- **~47% experimenting, ~40% live — but only ~14% have deployed agentic/generative AI in underwriting.** 65% are not doing so in underwriting or claims; 12% have no plans.
- **Top use cases: data extraction from unstructured documents (74%), submission preparation (54%), claims triage (~33%).**
- **Barriers: data quality/availability (49%), integration with existing systems (46%), cost / uncertain ROI (48%).**
- A second survey (~60% of market capacity): **72% have an AI framework; 62% enforce mandatory human oversight.**
- The LMA/Oxbow framing for the winning pattern: **"augmented underwriting"** — the algorithm triages, scores and advises while **the human underwriter stays central**.

> *"74% of the market is chasing unstructured-data extraction and 46% are blocked on integration — that is the JD's 'business integration and service design' gap stated as market data. And 14% agentic-in-underwriting tells me the right target is augmented underwriting with audited human oversight, not autonomy."*

### 4.3 The model-risk answer, in interview language

> *"For anything that informs underwriting or claims, I'd treat the model as a controlled artefact: a documented purpose and explicit limitations, validation against a held-out set before use, post-deployment monitoring, a named accountable owner, change control with evaluation evidence, and human oversight for the decision itself. The evidence pack is the model card, the decision log, and the request trace. Structurally that is PRA SS1/23's five principles — model identification, governance, development/use, independent validation, and risk mitigants — applied by analogy to insurance."*

### 4.4 The vocabulary that signals you understand regulated AI

| Say | Instead of |
|---|---|
| "human-in-the-loop for the binding decision" | "the user approves it" |
| "traceable to the source document, page and span" | "we log the answer" |
| "the trace is the audit evidence" | "we can re-run it" |
| "model limitations register / model card" | "docs" |
| "change control with evaluation evidence" | "we tested it" |
| "the cost of error is asymmetric" | "we optimise accuracy" |

---

## 5. The Azure stack — components and *what each is for*

Answer "which Azure AI components have you worked on?" with **components + purpose**, never a bare list.

| Component | What it gives you | Where I'd use it | The limitation to name |
|---|---|---|---|
| **Azure OpenAI / Azure AI Foundry** | Managed access to frontier models with enterprise identity, private networking, content filtering; **Foundry is the AI engine (models, agent service, evals), Copilot Studio is the delivery layer** | The generation layer | Quota and regional model availability; latency depends on region |
| **Foundry IQ** (new) | Managed knowledge layer producing **permission-aware knowledge bases** for agents, built on Azure AI Search; **ACL sync + Purview sensitivity-label enforcement**; sources include Blob/SharePoint/OneLake | Grounding for internal agents where entitlement must be inherited | Still tied to the AI Search index model |
| **Azure AI Search** | Hybrid retrieval (BM25 + HNSW vector), **semantic ranker**, metadata filters, integrated vectorisation. Ranking levels: L1 (BM25+HNSW) → Fused L1 (RRF) → **L2 semantic ranker** → L3 **agentic retrieval** (LLM query planning → parallel subqueries → merge with citations) | The RAG retrieval layer | Semantic ranker is now **required** for agentic retrieval; set `k=50` for max semantic-ranker input; **agentic retrieval billing shifts from per-query to per-token** |
| **Azure AI Document Intelligence** | Layout-aware extraction. **`prebuilt-layout` v4.0 (API 2024-11-30): up to 2,000 pages/file, selective `pages`, paragraph roles, tables, selection marks, figures, hierarchical sections, Markdown output for RAG**; plus a **custom classifier** for type+page-range splitting | Submission / slip / wording ingestion | Custom models need labelled data; accuracy varies by document family. Pipeline: ingest → classify+split → layout parse → extract with source links → HITL fallback |
| **Azure AI Content Safety** | Harm classifiers, **prompt shield for indirect injection**, groundedness detection; **Azure AI Evaluation SDK** for RAG/agent metrics | Guardrail layer | A classifier is a *signal*, not the control; policy checks in code remain the enforcement |
| **Azure Machine Learning** | Classical model lifecycle: training, registry, endpoints, pipelines, **model monitoring / drift**, responsible-AI dashboard | Classical ML half of the role (pricing support, triage classifiers, damage models) | Governance still needs a process around it |
| **Copilot Studio** | Low-code conversational agents over enterprise data; native Teams/SharePoint/Dynamics | Simple internal assistants — days to value | **Locked to Microsoft-managed OpenAI models**; opaque/non-configurable RAG; expensive at scale |
| **Power Platform / Power Apps** | Workflow and app layer around a model's output | Where the underwriter actually sees the result | Not for the AI core; it's the delivery surface |
| **Microsoft Fabric / Purview** | Data platform and governance/cataloguing | Where the data already lives; lineage and classification | Lineage coverage depends on adoption |
| **Entra ID + managed identity + private endpoints + VNet** | The security boundary: no public access, no stored keys. **Managed identity is Azure's genuine advantage over AWS IAM/STS and GCP WIF** | Every production AI resource | Adds setup complexity; do it from day one, not later |

**The Azure judgement line:** *"In an Azure shop the default is the native service unless it costs me something I can't accept — index control, retrieval quality, or a data-residency case. The trade-off is integration and governance versus ceiling."*

**If asked for honest weaknesses** (say one — it shows judgement): thinner non-OpenAI model catalogue than Bedrock; agent framework less mature than Bedrock AgentCore; a compliance premium on GPT pricing at scale; complex consumption billing; lock-in.

**The mitigation line:** *"In a multi-provider world I'd put an **AI gateway** in front on day one — LiteLLM, Kong or Portkey — so the provider is a config change, not a rewrite, and I get per-tenant token accounting and failover in one place."*

---

## 6. Low-code vs pro-code — the JD's explicit ask

The JD names **"pro-code or low-code decisions"** and lists **Copilot Studio, Power Platform/Power Apps, Cytora**. Prepare a rule, not an opinion.

| Dimension | Favours low-code | Favours pro-code |
|---|---|---|
| Is the capability a differentiator? | No | Yes |
| Process complexity | Simple, well-defined | Branching, multi-step, agentic |
| Time to value | Weeks | Months |
| Retrieval / eval control needed | Basic | Deep (chunking, reranking, trajectory eval) |
| Cost visibility | Acceptable to hide | Must be per-unit and attributable |
| Integration depth | Standard connectors | Custom APIs, system of record write-back |
| Testing / version control | Light | CI gate, golden set, prompt versioning |
| Maintenance owner | Business/ops | Engineering |

**Your script:**

> *"Rule of thumb: low-code when the capability is not a differentiator, the process is simple, and the business needs it in weeks. Pro-code when I need control over retrieval, evaluation, cost or integration. And either way I document the decision — 'why we didn't build it' is part of the architecture, and it's the question a model-risk reviewer will ask."*

**Where low-code bites (know these):** retrieval-quality ceiling, shallow evaluation, hidden cost, weak testing/versioning, and lock-in if you later need to move.

---

## 7. Vendor landscape (say two or three, correctly, and stop)

| Vendor | What it does | Note |
|---|---|---|
| **Cytora** | Digital risk processing for commercial insurance — turns submissions into structured, decision-ready risk data. Customers include Beazley, Allianz, Markel, Ecclesiastical | **Acquired by Applied Systems in 2025** (named in the JD — see §7.1) |
| **hyperexponential (hx)** | Pricing and rating platform for specialty insurers (carriers process $45–50bn+ GWP). **Launched "Triage" (Sept 2025)**, extending from pricing into the submission-to-portfolio journey | — |
| **Send** | Underwriting workbench (SaaS) for commercial/specialty insurers; integrated with hx Renew (2024) | — |
| **Federato** | AI-native insurance decisioning; customers incl. Nationwide, QBE | — |
| **Sixfold / Root** (Lloyd's Lab alumni) | AI for underwriting risk assessment and submission triage | — |
| **Kalepa** | Underwriting decisioning | — |
| **Property-imagery vendors** | **Tractable** (damage appraisal) · **CAPE Analytics (Moody's)** / **Arturo** / **Verisk** / **EagleView** (aerial imagery) | See §7.2 |
| **Microsoft** | Copilot Studio, Power Platform, Azure AI — the low-code and model layer | — |
| **Open-source / self-hosted** | vLLM, LangGraph/LangChain, pgvector/Qdrant/OpenSearch — for control, residency, or cost | — |

**The Lead framing:** *"I'd buy the capability that isn't our edge — submission intake, workbench, rating — and build the parts that are: our appetite logic, our evaluation, our integration to the system of record, and the model-risk evidence around all of it."*

### 7.1 ⚠️ The Cytora correction (read this before you speak)

The JD lists **Cytora** among "comparable products". Web research found **no evidence of a Brit–Cytora deployment** — Brit's public broker-platform news is **Acturis**, and Cytora's public partnership news is with **Polaris (2019)**, before its acquisition by Applied Systems.

**Do not claim Brit uses Cytora.** Say instead:

> *"The JD names Cytora alongside Copilot Studio as the comparable, buy-side class of tooling, so I'd read that as the option to be evaluated rather than a current deployment."*

Being precise about an unverified claim is itself a credibility signal — and it is the same instinct as the "the trace is the audit evidence" discipline.

### 7.2 The post-catastrophe imagery pattern (so you design it right)

Brit is not alone here, which is useful — an established shape exists to design against:

| Player | What they shipped | The design lesson to lift |
|---|---|---|
| **Verisk XactAI** (Sept 2025) | Note summarisation, auto photo labelling, receipt categorisation — **all human-oversight-gated** | Even the vendors ship HITL, not autonomy |
| **Verisk + EagleView** (Feb 2025) | Drone/AI damage detection into Xactimate/XactAnalysis; 3bn+ imagery library | Imagery at scale is a *stitching and georeferencing* problem |
| **CAPE Analytics (Moody's)** Event Damage for Underwriting (Oct 2025) | Post-cat **"gray sky"** imagery, **4-level damage severity**, **reason codes**, pre/post side-by-side | **Severity class + reason code per property** — exactly the explainability pattern you should propose |
| **Arturo** | Roof-condition models, pre/post-storm triage, flood footprint, **fraud via prior-condition comparison** | Prior-condition comparison for fraud is a clever angle to mention |

**Two phrases to lift for §6.1's design answer:** **"gray sky vs blue sky imagery"** (post-event vs pristine baseline) and **"a severity class plus a reason code per property"**.

---

## 8. Vocabulary translation table (use this in Part 1 and Part 2)

| Tech term | Insurance-English |
|---|---|
| "user" | underwriter, broker, claims adjuster |
| "ticket" | submission, referral, endorsement |
| "customer request" | broker submission |
| "document ingestion" | submission intake / bulk ingestion |
| "output validation" | pre-bind checks; field-level traceability |
| "human-in-the-loop" | underwriter sign-off before binding |
| "false positive" | out-of-appetite risk accepted, or unnecessary adjuster dispatch |
| "false negative" | valid risk dropped, or genuine damage missed |
| "adoption" | underwriters actually using the workbench without double-checking |
| "business value" | underwriting time per submission; combined ratio impact |
| "SLA" | broker response time expectation |
| "drift" | wording changes, new LOBs, new regions |

---

## 9. Questions they may ask, with headlines

**"Why insurance? You could work anywhere."** → *"Because the constraint is real. In specialty insurance the model has to be explainable to an underwriter and defensible to a regulator, and that's exactly where applied AI stops being a demo and starts being engineering."*

**"What's the difference between what we do and retail insurance?"** → *"Specialty is judgement-heavy and document-heavy, placed through brokers on a thin margin. It's not a mass-market pricing problem — it's a per-risk expertise problem, which is why extraction, retrieval and triage matter more than a single big model."*

**"How would you explain model limitations to an underwriter?"** → *"Plainly, in their language: 'it's good at pulling sums insured off a schedule, and it's not reliable on handwritten endorsements — those go to you.' Overstating it once destroys adoption permanently."*

**"What would you do in the first 90 days?"** → Month 1 listen and map what's in production vs pilot; month 2 take one use case end-to-end with a named owner and a measurable outcome; month 3 install one standard — starting with an evaluation gate in CI.

**"Anything in our stack you'd push back on?"** → *"Not the stack — the sequencing. I'd want to know which use case has an owner who will still care in six months, because that's the variable that predicts success more than the model choice."*

---

## 10. Cheat-sheet summary

- **Lloyd's = market, not company.** Brokers place risk with syndicates. Brit is a syndicate + reinsurer under Fairfax, **100%-owned**; **Brit and Ki separated 1 Jan 2025**.
- **Workflow:** submission → ingestion → triage → extraction → enrichment → pricing → quote → **bind** → endorsements → claims.
- **Their strategy is public:** *"smarter underwriting and simpler operations"* · **AI Lab + 20 AI proposals + AI Champions + AI literacy** · **five classes live (Cyber, Property Worldwide, Cargo, Terrorism, Energy)** · **hub-and-spoke with an AI risk framework**.
- **Their showpiece = post-cat imagery** (GIC aerial imagery since 2019; Hurricane Ian Oct 2022) → expect a vision design question. Use *"within days"*, *"before claims are reported"*, *"the adjuster still decides"*.
- **The role likely sits in the Bengaluru GCC** (led by Ravi Verma Vunikili; *"AI-native SDLC transformation"*).
- **Regulators:** FCA (**Consumer Duty** — *"the machine did it" is not a defence*; **SHAP/LIME ≠ regulatory explainability**) · PRA (**SS1/23** model risk, five principles) · Lloyd's (**correlation warning**) · **LMA AI Adoption Toolkit** (tiers automated underwriting as **HIGH risk**) · UK GDPR · Solvency II · sanctions.
- **Market data:** 74% chasing unstructured-data extraction · 46% blocked on integration · only ~14% agentic in underwriting · **62% enforce mandatory human oversight**. Target **augmented underwriting**.
- **Model risk = model card + validation + monitoring + named owner + change control + human oversight.**
- **Azure default:** Azure OpenAI/Foundry (+**Foundry IQ**) · AI Search (**agentic retrieval**) · Document Intelligence (**layout v4.0**) · Content Safety · Azure ML · Copilot Studio · Power Platform — with private endpoints and managed identity. Put an **AI gateway** in front for provider portability.
- **Low-code rule:** use it where the capability isn't a differentiator; pro-code when you need retrieval, eval, cost or integration control. Document the decision either way.
- **⚠️ Never claim Brit uses Cytora.** The JD names it as a comparable tool; there is no public evidence of a deployment (and Cytora was acquired by Applied Systems in 2025).
- **Translate everything:** submission, not ticket; underwriter, not user; bind, not approve.

## 11. Cross-links

- The evidence and sources behind §3, §4, §7 and §5: [`14-industry-intel-lead-ai-interview-research.md`](14-industry-intel-lead-ai-interview-research.md)
- Where to use this language: [`01`](01-brit-jd-decode-and-role-fit.md) · [`02`](02-stage1-format-60-minute-battle-plan.md) · [`12`](12-people-leadership-and-stakeholder-round.md)
- Design questions in this domain: [`05`](05-system-design-round-playbook.md) §6
- Governance answers: [`07`](07-agent-evaluation-and-guardrails.md) §8
