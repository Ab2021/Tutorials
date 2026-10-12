# Mock Interview Sessions — Full Practice Scenarios

> [!NOTE]
> **Where this file sits.** The **scripted-rehearsal** companion to `08-applied-ai-mock-interview-prompt.md` and `09-mock-interview-strategies-and-drills.md`. `08` is the prompt you give the mock interviewer; `09` is the drill methodology; this file is **two complete worked sessions** — the questions, the scripted model answers with real numbers, the follow-up probes the panel is likely to fire, and an 8-row scoring rubric. Harvested in full from the earlier `br_ins/` set.

> [!WARNING]
> The Brit facts quoted inside the scripted answers here (syndicate numbers, Ki's status, leadership names, PML/catastrophe framing) are the **candidate's pitch material, not verified facts**. Ki in particular: it is **not** Brit's own algorithmic syndicate — Brit and Ki **separated on 1 Jan 2025**, and Ki Financial Ltd (syndicate 1618) is a *follow* syndicate that Brit merely administers. Check anything you intend to assert against `11` §3.2 and `14`.

> [!IMPORTANT]
> Based on the actual Brit Insurance interview structure: **4 parts, 60 minutes total**.
> Use these sessions as timed simulations — set a stopwatch and hold yourself to the time boundaries.

---

## Interview Structure at a Glance

| Part | Topic | Duration | Weight |
|---|---|---|---|
| 1 | Introductions | ~10 min | Low — sets tone |
| 2 | Deep-dive: agentic AI project | ~25 min | **High — this is the core** |
| 3 | System design (image anomaly) | ~15 min | **High — hands-on thinking** |
| 4 | Coding exploration | ~10 min | Medium — breadth over depth |

---

## MOCK SESSION 1: Full 60-Minute Interview Simulation

---

### Pre-Interview Checklist

Complete these at least 24 hours before the interview. Check each box in your mind before you start a practice session.

**Research (30 min)**
- [ ] Read Brit Insurance's last two annual reports — note catastrophe loss ratio, key geographies, tech mentions
- [ ] Read the Brit "smarter underwriting" press releases and any AI/ML blog posts
- [ ] Understand Lloyd's of London syndicates structure (Brit is syndicate 2987)
- [ ] Know who Ki Insurance is (Brit's algorithmic syndicate — fully automated underwriting)
- [ ] Look up Brit's key leadership names (CEO, CTO/CDO) for conversational reference
- [ ] Identify 2–3 recent major catastrophe events Brit would have been exposed to (hurricanes, wildfires, flooding)

**Technical Preparation (45 min)**
- [ ] Re-read your most complex agentic AI project end-to-end — can you draw it from memory?
- [ ] Practise the image anomaly system design on a whiteboard or paper without notes
- [ ] Refresh Azure AI Foundry, Azure OpenAI, and Azure AI Search feature list
- [ ] Review the sample RAG code (see Part 4) and memorise 5 improvements cold
- [ ] Know your evaluation metrics: RAGAS, faithfulness, answer relevancy, groundedness

**Logistics**
- [ ] Water bottle on the desk
- [ ] Quiet room, headphones off, phone silent
- [ ] LinkedIn profile open in a tab (to reference quickly if asked about tenure/dates)
- [ ] Pen and paper for system design (or a digital whiteboard if remote)
- [ ] GitHub repo / portfolio link ready to share

---

### Part 1: Introduction (10 minutes)

---

#### Opening Question 1

> **Interviewer**: *"Tell us about yourself and your experience with AI."*

**Objective**: Deliver a crisp 90–120 second headline that signals seniority, domain relevance, and readiness for this specific role. Lead with outcome, then proof.

---

**✅ Model Answer**

> "Sure. I'm a Lead AI Engineer with around eight years of experience, the last four of which have been focused almost entirely on building production AI and machine learning systems. In my current role I lead a team of five engineers — two ML specialists, two full-stack engineers, and a data engineer — and I'm personally accountable for the design, delivery, and quality of the AI layer across our platforms.
>
> The work that's most relevant to Brit is probably the agentic AI platform I architected over the last two years. It's an enterprise-grade multi-agent system that handles complex knowledge retrieval and decision support at scale — processing hundreds of thousands of queries per day across regulated document repositories. We built the entire thing on Azure: Azure OpenAI for the LLM layer, Azure AI Search with hybrid vector and keyword retrieval, Azure AI Foundry for evaluation, and a microservices orchestration layer using Semantic Kernel.
>
> What I think makes my background particularly relevant here is that I've done this in a regulated context — stringent audit trails, explainability requirements, model risk governance, the works. Insurance is arguably the most demanding regulated environment for AI, and I'm genuinely excited to apply that experience at Lloyd's of London scale.
>
> I'm also a strong technical leader. I run architecture reviews, define engineering standards, mentor engineers, and work closely with product and data science to translate business problems into deployable AI systems.
>
> To summarise: agentic AI at production scale, Azure-native, regulated environments, and team leadership — that's the core of what I bring."

**Why this works**:
- Opens with a clear seniority signal ("Lead AI Engineer", "team of five")
- Quantifies: hundreds of thousands of queries, 8 years, 2 years of agentic work
- Azure-specific without being a product catalogue — names tools with *why*
- Pivots naturally to Brit's context (regulated, Lloyd's)
- Ends with a clean summary headline — interviewers remember the last thing you say

---

#### Opening Question 2

> **Interviewer**: *"What attracted you to Brit Insurance specifically?"*

**Objective**: Show you did your research. Generic "I want to work in AI for insurance" answers are immediately spotted. Be specific — name products, strategies, and market positions.

---

**✅ Model Answer**

> "A few things, and I'll be direct about all of them.
>
> First, Ki Insurance. When I learned that Brit had built a fully algorithmic syndicate — one that underwrites risks autonomously without a human underwriter touching a file — that told me something important about the organisation's ambition and its appetite for real AI innovation. That's not a chatbot pilot. That's a fundamental redesign of the underwriting process using AI as the core decision engine. I want to be part of that kind of work.
>
> Second, the Broker API and distribution technology strategy. Insurance distribution has historically been extremely manual and fragmented. The fact that Brit is investing in API-driven connectivity with the broking community signals that they understand that better data pipelines will compound into better models, and better models mean better pricing and risk selection. That's an AI strategy, even if it doesn't always get framed that way.
>
> Third, the catastrophe analytics capability. Building ML models that assess PML — probable maximum loss — across complex correlated exposures is a genuinely hard problem. I've done related work on spatial data and anomaly detection, and I'm keen to go deeper into the catastrophe science domain.
>
> And honestly — Lloyd's of London as a market is unique. The data richness of the slip data, historical loss triangles, and class of business diversity you'd be modelling here is extraordinary compared to personal lines or even standard commercial. The ML problems are harder, which means the solutions you build are more valuable.
>
> So it's not just that I want to work in insurance AI — it's specifically that Brit seems to be doing the most interesting things in London market AI right now."

**Why this works**:
- Names Ki Insurance specifically — very few candidates will know this
- Connects Broker API investment to an AI data pipeline thesis — shows systems thinking
- Catastrophe analytics → bridges to your own background
- Compliments the market with substance, not flattery
- Ends with a confident, specific statement of intent

---

### Part 2: Project Deep-Dive (25 minutes)

> [!NOTE]
> This is the longest and most important section. The interviewers will probe, push back, and ask follow-ups. Stay calm, stay structured, and slow down when you get to technical depth — they want to see you think, not just recite.

---

#### Main Question

> **Interviewer**: *"Walk us through your most complex agentic AI project. What was the problem you were trying to solve?"*

**Objective**: Full STAR format. Set context in 60 seconds, describe action with technical depth over 3–4 minutes, land on quantified outcomes.

---

**✅ Model Answer — Full STAR**

> **Situation**:
> "We were operating in a large regulated enterprise with thousands of internal knowledge documents — policy documents, regulatory guidance notes, compliance frameworks, technical standards. Knowledge workers — think risk analysts, compliance officers, underwriting support — were spending hours each week manually searching through these to answer questions they needed to make decisions. Search was keyword-based and completely unaware of intent or context. We estimated we were burning roughly 15,000 person-hours per quarter on avoidable document lookup.
>
> **Task**:
> I was brought in to design and own the AI solution end-to-end. My mandate was: cut that lookup time by at least 60%, without sacrificing accuracy or auditability — because in a regulated environment, every answer the system gives needs to be traceable to a source document.
>
> **Action**:
> I designed a multi-agent knowledge retrieval system on Azure. Let me walk you through the architecture at a high level, then I'll go deeper on the parts that were hardest.
>
> At the top level we had an orchestrator agent — built with Semantic Kernel — whose job was to decompose an incoming user query into sub-tasks and route them to specialist agents. We had three specialist agents: a retrieval agent, a synthesis agent, and a validation agent.
>
> The retrieval agent used Azure AI Search with hybrid retrieval — a combination of dense vector search using Ada-002 embeddings and BM25 keyword search. We used Reciprocal Rank Fusion to merge the result sets. This was important because some queries were highly semantic — 'what's our exposure to business interruption in flood zones?' — while others were highly literal — 'what does clause 4.3.2 say?' Hybrid search handled both well. Documents were chunked at 512 tokens with 10% overlap and stored in Azure AI Search with metadata fields for document type, version, and classification.
>
> The synthesis agent took retrieved chunks, ran them through GPT-4o via Azure OpenAI, and generated a grounded answer with inline citations. We enforced grounded-only responses — the model was instructed to refuse to answer if the evidence wasn't in the retrieved context.
>
> The validation agent ran a faithfulness check using a secondary LLM call — essentially LLM-as-judge — scoring each response before it was returned to the user. Responses below a faithfulness threshold of 0.75 were either re-tried with a broader retrieval window or escalated to a human reviewer queue.
>
> Audit was non-negotiable. Every query, every retrieved chunk, every generated response, and every validation score was logged to an append-only audit store — Cosmos DB with a time-to-live policy matched to our document retention policy. Regulators could request a full trace of any decision, and we could replay the exact retrieval context.
>
> **Result**:
> We went live to 200 users in a controlled rollout, then scaled to 1,200 users over six months. Average query resolution time dropped from 47 minutes to under 4 minutes. User satisfaction score in our quarterly survey went from 2.8/5 to 4.4/5. We ran the system through a model risk review and it passed with one minor finding — around bias in document coverage — which we addressed with a metadata-weighted retrieval boost for recently updated documents. 
>
> That project taught me more about the operational realities of agentic systems than any prior work — specifically around failure modes, evaluation pipelines, and the importance of designing for observability from day one."

---

#### Follow-up 1

> **Interviewer**: *"How did you handle failures in your agent?"*

**✅ Model Answer**

> "Great question — this is something I learned the hard way on early iterations.
>
> I categorise failures into three types: **transient failures**, **logic failures**, and **safety failures** — and each needs a different strategy.
>
> **Transient failures** are things like Azure OpenAI returning a 429 rate limit or a timeout. For those, we implemented exponential backoff with jitter. We set a retry budget — three retries maximum per agent step, with full circuit breaker logic at the orchestrator level. If a specialist agent failed three times, the orchestrator would log the failure, return a graceful degradation response to the user ('I wasn't able to retrieve this right now — please try again in a moment'), and page our on-call engineer via Azure Monitor alert. Critically, every step was **idempotent** — retrying the retrieval agent wouldn't create duplicate writes or double-process anything.
>
> **Logic failures** are more subtle — things like the retrieval agent returning chunks from the wrong document version, or the synthesis agent hallucinating a citation. These don't throw exceptions; they silently produce wrong answers. This is why the validation agent was essential. We also ran offline regression tests weekly: a curated test set of 200 question-answer pairs from domain experts, scored for groundedness and correctness. Any model or index change had to pass this regression suite before deployment.
>
> **Safety failures** are the scariest — when the agent takes an action it shouldn't. In our read-only retrieval system this was less of a risk, but in the subsequent version where agents could initiate draft document workflows, we introduced a **human-in-the-loop gate** for any write action. The agent would present a proposed action with its reasoning, a confidence score, and the retrieved evidence, and a human had to approve before it committed anything. Low-confidence actions — below 0.7 on our internal scoring — were escalated immediately.
>
> The meta-principle is: **design for failure, not just for the happy path**. Every agent step should assume it might be called multiple times, should produce observable output for every execution, and should have a defined fallback."

---

#### Follow-up 2

> **Interviewer**: *"Which components of Azure AI did you work with directly?"*

**✅ Model Answer**

> "I'll go through the stack layer by layer, because there are several components and I want to be precise rather than namedropping.
>
> **Azure OpenAI Service**: GPT-4o for synthesis and validation agents. GPT-4o-mini for lower-latency classification tasks — query intent classification and entity extraction. We used the Responses API and structured output mode to force the model to emit JSON-schema-validated tool calls. We also used embedding models — Ada-002 initially, then tested text-embedding-3-large which gave us roughly 6% improvement on our retrieval recall metric.
>
> **Azure AI Search**: This was our retrieval backbone. We used hybrid search — vector field for semantic similarity, and full-text BM25 for keyword precision. We enabled semantic ranker (the cross-encoder reranking layer) for the top 10 results. We also used the integrated vectorisation feature to auto-embed on indexer runs, so when a document was updated in SharePoint it would flow through the indexer pipeline, get chunked, embedded, and indexed automatically. We used managed identity to authenticate the indexer to the blob source — no secrets in config.
>
> **Azure AI Foundry**: I used this for evaluation pipelines. We set up automated eval runs using the built-in groundedness, coherence, and relevancy metrics, plus a custom faithfulness evaluator that compared the response against the retrieved chunks. Foundry's experiment tracking meant we could compare eval results across model versions and prompt changes in a structured way — essential for governance.
>
> **Semantic Kernel**: This was our agent orchestration framework. We used the Python SDK. It gave us a clean abstraction for defining agent functions (tools), managing conversation state, and coordinating multi-step plans. We evaluated LangGraph and AutoGen as well but chose Semantic Kernel for its tighter Azure integration and better support for enterprise auth patterns.
>
> **Azure Service Bus**: Queue layer between the user-facing API and the agent backend. This decoupled the latency of agent execution from the UI response, and gave us a durable buffer during load spikes.
>
> **Azure Monitor + Application Insights**: Full observability — distributed traces per query, agent step timing, token usage, error rates, latency percentiles. We set alerts on p95 latency breaching 8 seconds and error rate exceeding 2%.
>
> **Managed Identity**: Used throughout — no secrets, no key rotation headaches. The API app's managed identity had RBAC roles scoped precisely: Cognitive Services OpenAI User on the OpenAI resource, Search Index Data Reader on the AI Search resource, and Cosmos DB data plane roles for audit writes."

---

#### Follow-up 3

> **Interviewer**: *"How did you evaluate your agents? How did you know they were performing well?"*

**✅ Model Answer**

> "Evaluation was a major investment — probably 20% of the total engineering effort — and I'd argue it was the most valuable 20%. We had a three-layer evaluation framework.
>
> **Layer 1 — Offline component metrics** (run before every deployment):
> - *Retrieval recall@K*: Given a ground-truth question-document pair, does the correct chunk appear in the top K retrieved results? We targeted recall@5 > 0.88. Below that, we'd investigate indexing or chunking issues.
> - *Faithfulness*: For every generated answer, what fraction of factual claims are grounded in the retrieved context? Scored by LLM-as-judge. Target: > 0.85.
> - *Answer relevancy*: Does the answer actually address the question asked? Scored by embedding cosine similarity between question and answer. Target: > 0.80.
> - *Groundedness* (from Azure AI Foundry): Azure's built-in metric — cross-references each sentence in the response against the retrieved documents.
> - *Latency*: p50, p95, p99 end-to-end from query to response. Target p95 < 8 seconds.
>
> **Layer 2 — Online metrics** (production monitoring):
> - *Thumbs up/down feedback* on every response. Users could give instant feedback. We tracked a weekly rolling net approval rate — target > 80%.
> - *Escalation rate*: What fraction of queries triggered human escalation (low-confidence validation score)? Target < 10%. Rising escalation rate = signal of retrieval quality degradation or index staleness.
> - *Query abandonment*: If a user submitted a follow-up query immediately after a response, that was a signal the first answer was insufficient.
>
> **Layer 3 — Periodic human evaluation** (monthly):
> - A sample of 100 queries reviewed by domain experts (compliance, underwriting).
> - They scored on a 1–5 scale: relevance, correctness, completeness, citation accuracy.
> - Results fed back into our ground-truth test set.
>
> The most important lesson: don't only measure what's easy to measure. Early on we only tracked thumbs up/down and called it good. When we added faithfulness scoring, we discovered that roughly 12% of 'thumbs up' responses had at least one unfaithful claim — users were happy with confident-sounding answers even when they were subtly wrong. That's dangerous in a regulated context, so we added faithfulness as a hard gate on the validation agent."

---

#### Follow-up 4

> **Interviewer**: *"How big of a team did you work on this with?"*

**✅ Model Answer**

> "I led a core team of five engineers. Let me describe the composition and how I structured the work.
>
> I had two ML/AI engineers who owned the retrieval and evaluation pipelines — they were deep on the vector search configuration, chunking strategies, embedding model selection, and the eval framework in AI Foundry. They reported to me technically, though formally they sat in a data science org — I was the engineering lead they were aligned to for this project.
>
> Two full-stack engineers owned the API layer and the user-facing application — a React front-end with streaming responses using Server-Sent Events, backed by FastAPI. They also owned the auth and RBAC layer.
>
> One data engineer owned the indexing pipeline — the document ingestion, chunking, metadata tagging, and the SharePoint-to-Blob-to-Search pipeline. Critical role, often underestimated.
>
> Then there was a broader set of stakeholders: a product manager who ran user research and prioritisation, a security architect who reviewed our managed identity and data classification design, and a data governance lead who owned our model risk documentation.
>
> My job was to: set the overall architecture, run weekly technical design reviews, unblock engineers when they hit ambiguity, make the hard trade-off calls (like switching from LangChain to Semantic Kernel mid-project — a painful but correct decision), and be the face of the technical delivery to senior stakeholders.
>
> I also mentored one of the junior ML engineers through their first production ML deployment — pair-programming on the evaluation framework and giving structured feedback on their design documents. They're now running that evaluation pipeline independently.
>
> The team dynamic that I'm most proud of: we shipped to 1,200 users with zero Sev-1 incidents. That doesn't happen by accident — it happens because you've invested in testing, monitoring, and runbooks."

---

#### Follow-up 5

> **Interviewer**: *"What were the regulatory and compliance considerations you had to work through?"*

**✅ Model Answer**

> "Several, and each required a different engineering response. Let me go through the main ones.
>
> **Auditability**: Every query and every answer had to be fully reproducible and explainable. We logged: the raw user query, the top-K retrieved chunks with their source document IDs and chunk IDs, the exact prompt sent to the LLM, the raw LLM response, the validation score, and the final response delivered. All in a structured, queryable format in Cosmos DB, with immutable writes (append-only, no deletes). Retention policy matched our document retention standard — 7 years.
>
> **Model Risk Management**: Our AI systems were subject to model risk governance — similar in spirit to SR 11-7 in banking, though adapted for our regulatory context. This meant producing a model card for the RAG system: what data was used for training the embedding model (we used a pre-trained Microsoft model, so we documented the upstream training data), validation results, known limitations, and an owner accountable for ongoing monitoring. This was reviewed and signed off by our Chief Risk Officer's team.
>
> **Data Classification and Privacy**: Documents in the index were classified at different sensitivity levels. We implemented metadata-based access control — when a user queried, the retrieval agent filtered the search results to only include documents the user's RBAC group was authorised to see. A compliance officer couldn't accidentally retrieve a document classified for the executive team only.
>
> **Explainability**: In a regulated environment, 'the model said so' is never an acceptable answer. Every response included inline citations — specific document names, section references, and page numbers. Users could click through to the source. The system was designed so a regulator asking 'why did your system give this answer?' could get a full, traceable explanation in under five minutes.
>
> **Bias and Coverage Monitoring**: We identified a risk that our index was biased toward older, more frequently accessed documents, which got more engagement signals. We mitigated this with a recency boost in the scoring profile — more recently updated documents got a small uplift in ranking. We also ran quarterly coverage audits: what percentage of the document corpus was being retrieved and cited? Dark spots in the coverage map indicated indexing or chunking issues.
>
> The meta-lesson: regulatory compliance isn't an afterthought you bolt on at the end. It has to be a first-class architectural constraint from day one, or you end up re-engineering fundamental parts of the system to retrofit it."

---

### Part 3: System Design (15 minutes)

> [!TIP]
> In a system design question, **the interviewer is watching how you think**, not just what you draw. Vocalise your reasoning at every step. Ask clarifying questions before drawing a single box.

---

#### Design Question

> **Interviewer**: *"I want you to design a system where you have images and need to run them through a model to check for anomalies. Draw the architecture."*

---

#### ⏱ Minute 1–2: Clarifying Questions

Before touching a whiteboard or drawing tool, say the following out loud:

> "Before I start designing, I want to make sure I'm solving the right problem. Can I ask a few clarifying questions?
>
> 1. **Scale**: Roughly how many images are we processing — per day under normal conditions, and what does peak look like? I'm thinking about a catastrophe event where volume might spike 10x overnight.
>
> 2. **Image source**: Where do these images come from? Broker uploads via a portal? Drone or satellite feeds post-catastrophe? This affects ingestion design significantly.
>
> 3. **Anomaly type**: What counts as an anomaly? Physical damage to property — structural, flood, fire? Or are we also looking at fraudulent document manipulation — altered images?
>
> 4. **Latency requirement**: Does an adjuster need a result in real-time while they're on a call with a broker, or is batch processing within a few hours acceptable?
>
> 5. **Cloud platform**: I'll assume Azure given the context — is that right?
>
> 6. **Output consumer**: Who acts on the anomaly detection result — an automated decision, a human adjuster, or both?
>
> Okay, so based on your answers I'll design for: **property damage detection post-catastrophe event**, **Azure-based**, **batch processing** (results within 2 hours of image upload), **broker and field adjuster uploads** as the source, and a **human adjuster review workflow** as the consumer of results. I'll also assume we need full auditability — this is going into underwriting and claims decisions."

---

#### ⏱ Minute 2–5: Architecture Drawing and Walkthrough

*[Narrate as you draw — talk through each component as you add it to the whiteboard]*

> "I'll draw this in layers — ingestion, processing, inference, results, and review.
>
> **Layer 1 — Ingestion**:
> Starting at the left. Images arrive via a broker portal or a mobile field app. Both channels upload to **Azure Blob Storage** — this is our durable raw image store. Blob Storage is ideal here: it handles any image format, scales infinitely, and integrates natively with the rest of the Azure ML stack.
>
> When a new blob lands, it fires an **Event Grid** trigger. Event Grid is low-latency and reliable — it picks up blob-created events and pushes a message to an **Azure Service Bus** queue. I use Service Bus rather than direct Event Grid fanout to the processor because Service Bus gives me durable queuing, dead-letter support, and back-pressure handling during surge events.
>
> **Layer 2 — Preprocessing**:
> A preprocessing worker (Azure Container App, auto-scaled) picks messages off the queue. Its job: validate the image (check format, file size, corruption), extract metadata (geolocation if EXIF data exists, claim reference number from file naming convention, upload timestamp), and perform **image tiling** for ultra-high-resolution imagery. Satellite or drone images can be 100+ megapixels — you can't feed that directly to a model. We tile them into 512×512 or 1024×1024 overlapping patches and process each patch independently, then aggregate results. Preprocessed tiles are written back to Blob Storage in a processed container; the metadata is written to a SQL metadata table.
>
> **Layer 3 — Inference**:
> An inference worker (Azure ML Online Endpoint or an AKS-hosted model server with GPU nodes) picks up preprocessed image paths from a second Service Bus queue and runs inference. 
>
> The model outputs per-tile: a damage classification label, a confidence score, and bounding boxes around detected anomalies. I'll talk about model selection in a moment.
>
> Results are written to a **results store** — Azure SQL or Cosmos DB depending on whether we need relational joins with claim data. I'll go Cosmos DB for the flexibility, with a well-defined document schema per image.
>
> **Layer 4 — Results and Review**:
> A **Results API** (Azure API Management + FastAPI backend) exposes the results to the adjuster review application. Adjusters see: the original image, the anomaly overlay (bounding boxes rendered on the image), confidence score, severity classification, and the claim reference it's linked to.
>
> High-confidence results (above 0.85) can be surfaced directly in the workflow with a recommendation. Low-confidence results (below 0.60) are flagged for mandatory human review. Results between 0.60 and 0.85 are surfaced with a 'review recommended' flag.
>
> **Audit Layer** (cuts across everything):
> Every step — upload, preprocessing, inference, result write, adjuster review action — is logged to an immutable audit table. This is non-negotiable for claims and underwriting decisions."

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IMAGE ANOMALY DETECTION SYSTEM                   │
└─────────────────────────────────────────────────────────────────────────┘

  [Broker Portal]                          [Field Adjuster App]
       │                                          │
       └──────────────┬───────────────────────────┘
                      ▼
          ┌─────────────────────┐
          │  Azure Blob Storage  │  ← Raw image store
          │  (raw-images/)       │
          └──────────┬──────────┘
                     │ Event Grid trigger (blob created)
                     ▼
          ┌─────────────────────┐
          │  Azure Service Bus   │  ← Ingestion queue (durable)
          │  Queue: ingest       │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Preprocessing       │  ← Container App (auto-scale)
          │  Worker              │    - Validate, tile, extract EXIF
          │                      │    - Write metadata to SQL
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Blob Storage        │  ← Processed tiles
          │  (processed-tiles/)  │
          └──────────┬──────────┘
                     │ → Service Bus Queue: inference
                     ▼
          ┌─────────────────────┐
          │  Inference Worker    │  ← AKS + GPU nodes / Azure ML EP
          │                      │    - Damage classification
          │  [Azure Computer     │    - Confidence score
          │   Vision / Custom    │    - Bounding boxes
          │   Fine-tuned Model]  │    - Severity label
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Results Store       │  ← Cosmos DB
          │  (per-image docs)    │    - Claim ref, scores, labels
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Results API         │  ← FastAPI + APIM
          │  (APIM Gateway)      │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Adjuster Review UI  │  ← Next.js / React
          │                      │    - Image + overlay
          │  High conf → auto    │    - Approve / reject / escalate
          │  Low conf  → review  │
          └─────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Audit Log           │  ← Cosmos DB (append-only)
          │  (immutable events)  │    - Full trace per image decision
          └─────────────────────┘
```

---

#### ⏱ Minute 5–10: Component Deep-Dive

> "Let me go deeper on the two components that are most technically interesting: image preprocessing and model selection.
>
> **Image preprocessing — tiling**:
> Catastrophe imagery, especially satellite data, can be gigapixel scale. You absolutely cannot send 100 megapixels through a CNN in one shot. We tile with overlap — typically 10–20% overlap between tiles — so that damage on a tile boundary isn't missed. After inference, we aggregate tile results back to the original image coordinate space. If 3 out of 5 tiles covering a roof come back with a 'severe damage' classification above 0.80 confidence, the overall roof assessment is 'severe damage'. This aggregation logic needs to be carefully designed and tested.
>
> We also apply preprocessing normalisation: colour normalisation (satellite images vary by sun angle, season, sensor), contrast enhancement, and format standardisation to PNG or JPEG at consistent DPI.
>
> **Model selection**:
> Two options. First: **Azure Computer Vision** — specifically the Image Analysis 4.0 API with custom model training via Azure Custom Vision. Fast to deploy, no GPU infrastructure to manage, good for standard damage categories. The downside: you're limited to the training data you can label, and Azure Custom Vision has limits on model complexity.
>
> Second: **Custom fine-tuned model** — for example, a YOLOv8 or a vision transformer (ViT or Swin Transformer) fine-tuned on a proprietary dataset of labelled loss imagery. This gives you much higher accuracy on domain-specific damage types — distinguishing flood waterline marks from structural cracks, for example — but requires a labelled training dataset, ML engineering to maintain the training pipeline, and GPU infrastructure.
>
> My recommendation: **start with Azure Computer Vision** for velocity, and invest in a custom fine-tuned model as a parallel workstream. Run them in A/B shadow mode — both score every image, but only Custom Vision results are used in production initially. Once the custom model beats Custom Vision on your evaluation benchmark, you flip the switch.
>
> **GPU cluster auto-scaling**:
> AKS with KEDA (Kubernetes Event-Driven Autoscaling) connected to Service Bus queue depth. When queue depth exceeds N messages, KEDA scales up GPU node replicas. When queue drains, nodes scale back down. This is cost-critical — GPU VMs are expensive. You don't want to be running 20 GPU nodes overnight when there's no catastrophe event in progress."

---

#### ⏱ Minute 10–13: Scaling and Reliability

> "A catastrophe event creates a step-change in volume. Hurricane hits Florida, and within 6 hours you have 50,000 images uploaded by field adjusters. Normal processing volume is maybe 2,000 images per day. So you need to handle a 25x surge gracefully.
>
> **Queue buffering**: Service Bus is your shock absorber. It doesn't care if the inference workers are temporarily overwhelmed — it will hold messages durably until workers can process them. Set queue TTL to 48 hours so no image is lost even if the processing layer has an extended outage.
>
> **Dead letter queue**: Any message that fails preprocessing or inference three times goes to a dead-letter queue. Ops team is alerted. A human reviews the original image to determine if it's corrupt, malformed, or a genuine model failure. This ensures nothing silently disappears.
>
> **Idempotency**: Every processing step is idempotent. If the preprocessing worker crashes and a message is redelivered, it checks whether a processed blob already exists for that image ID before re-processing. Same for inference — check if a result document already exists in Cosmos DB for that image ID before running inference again. This prevents duplicate results and wasted GPU time.
>
> **Retry strategy**: Exponential backoff with jitter for transient failures (GPU OOM, model endpoint timeouts). Max 3 retries per message, then dead-letter.
>
> **Circuit breaker**: If the model endpoint error rate exceeds 10% over a 5-minute window, the inference worker switches to a degraded mode: queues images without inference, alerts the ops team, and the results API shows adjusters that AI scoring is temporarily unavailable. Better to be honest about the outage than to serve silently wrong results."

---

#### ⏱ Minute 13–15: Monitoring and Governance

> "Four things I'd monitor day-to-day:
>
> **1. Processing pipeline health**: Queue depth by stage (ingest, preprocessing, inference), worker throughput (images per minute), error rate by failure type. Azure Monitor dashboards, with PagerDuty alerts on queue depth exceeding 1-hour processing backlog.
>
> **2. Model performance monitoring**: Weekly offline evaluation against a labelled holdout set. If accuracy on holdout drops more than 2 percentage points week-over-week, that's a model drift alert — investigate whether new image types or conditions have entered the distribution.
>
> **3. Human-in-the-loop for low confidence**: The adjuster review UI surfaces all images below 0.60 confidence for mandatory human review. Adjuster decisions feed back into our labelled dataset (with appropriate quality controls) — over time this creates a virtuous cycle where human corrections improve the training data, which improves the model.
>
> **4. Explainability for adjusters**: Bounding boxes are necessary but not sufficient. Adjusters need to understand *why* the model flagged something. We use GradCAM (Gradient-weighted Class Activation Mapping) to produce a heatmap overlay showing which pixels drove the classification. Rendered in the review UI as a colour overlay on the image. This gives adjusters the context they need to make an informed accept/override decision, and it satisfies the regulatory requirement that AI-assisted decisions be explainable.
>
> **Audit trail**: Every image decision — AI score, adjuster review action, final classification, timestamp, adjuster ID — is recorded in the audit log. Fully queryable. A claims dispute or regulatory review can pull the complete decision history for any image in under 30 seconds."

---

#### Potential Follow-Up Questions and Answers

> **"What model would you use?"**
>
> "I'd start with Azure Computer Vision (Image Analysis 4.0) with custom-trained categories for speed to market. In parallel I'd build a fine-tuning pipeline for a Swin Transformer or YOLOv8 model on proprietary loss imagery. The custom model would run in shadow mode against Custom Vision until it demonstrably outperforms it on our internal benchmark — then we promote it to production. I'd never bet the production system on a custom model that hasn't been validated against a labelled holdout set."

---

> **"How do you handle a 10x surge during a catastrophe?"**
>
> "Service Bus absorbs the surge — it's durable by design. KEDA auto-scales the inference workers from, say, 2 nodes to 20 nodes within a few minutes, triggered by queue depth. We also have pre-scaled node pools ready to go — Azure AKS node pool with spot instances for burst capacity, triggered by a catastrophe event flag that the operations team can set manually. We'd communicate to adjusters that processing time may increase to 4–6 hours during a major event rather than our normal 2-hour SLA. Honesty about degraded SLAs is better than silent failure."

---

> **"How do you ensure the model's decision is explainable to adjusters?"**
>
> "GradCAM heatmaps overlaid on the original image — this is the visual explanation layer. Plus structured output: the model returns not just a label but a confidence score, the specific damage categories detected (roof damage, window breakage, flooding, fire), and a severity level (minor / moderate / severe / total loss). Adjusters see the full picture: what the model detected, where it detected it, how confident it was, and what category of damage. They can agree, override, or escalate. All of their actions are logged."

---

> **"What if the model is wrong 15% of the time?"**
>
> "Fifteen percent error rate is too high for a production system making financial decisions — but the right response depends on the error type. False positives (flagging undamaged property as damaged) and false negatives (missing actual damage) have very different business costs. A false negative on a severe structural failure is catastrophically worse than a false positive.
>
> Immediate response: any result below 0.85 confidence goes to mandatory human review. At 15% error rate, that's probably 40–60% of results requiring human review — which is expensive but acceptable as a short-term quality gate while we improve the model.
>
> Medium-term: analyse the errors. Are they concentrated in specific image types? Specific damage categories? Specific geographies? Build a labelled error set and fine-tune the model specifically on the error distribution. Run shadow A/B tests between the current model and the improved model. Once the improved model beats the current on precision/recall, promote it.
>
> Long-term: invest in a continuous human feedback loop — adjuster overrides become training labels — and establish a monthly model retraining cadence."

---

### Part 4: Coding Exploration (10 minutes)

> [!NOTE]
> You are not being asked to write perfect code from scratch. The interviewer wants to see how you *read*, *analyse*, and *improve* existing code. Slow down, structure your answer, and prioritise improvements by impact.

---

#### Coding Question

> **Interviewer presents the following RAG code and asks**: *"Walk us through this code and suggest improvements."*

```python
import openai
import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Setup
search_client = SearchClient(
    endpoint=os.environ["SEARCH_ENDPOINT"],
    index_name="documents",
    credential=AzureKeyCredential(os.environ["SEARCH_API_KEY"])
)

openai.api_key = os.environ["OPENAI_API_KEY"]

def search_documents(query: str) -> list:
    results = search_client.search(query)
    docs = []
    for r in results:
        docs.append(r["content"])
    return docs

def generate_answer(query: str, context: list) -> str:
    context_str = "\n".join(context)
    prompt = f"""Answer the question using the context below.
Context: {context_str}
Question: {query}
Answer:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def rag_pipeline(query: str) -> str:
    docs = search_documents(query)
    answer = generate_answer(query, docs)
    return answer

# Usage
if __name__ == "__main__":
    result = rag_pipeline("What is the flood coverage limit?")
    print(result)
```

---

**✅ Model Answer — 3-Step Approach**

**Step 1 — Summary (30 seconds)**:

> "This is a minimal RAG pipeline — it takes a user query, retrieves relevant documents from Azure AI Search using basic keyword search, and passes them to GPT-4 to generate an answer. It's a reasonable skeleton, but there are several significant issues across security, retrieval quality, robustness, and production readiness. Let me walk through the code and then give you my prioritised list of improvements."

**Step 2 — Walk Through Components**:

> "The `search_documents` function calls Azure AI Search with a simple text query. It returns a flat list of content strings — no metadata, no source tracking, no top-K limit specified. The `generate_answer` function concatenates all retrieved docs into a single context string — no length management — and sends them as a user message to GPT-4. There's no system message, no instruction to stay grounded, and the prompt is basic. The `rag_pipeline` function chains them without any error handling. The credentials are loaded from environment variables — that part is correct for secrets — but the API key pattern for Azure AI Search should ideally use Managed Identity."

**Step 3 — 5 Improvements Prioritised by Impact**:

> **Priority 1 — Security: Replace API Key Auth with Managed Identity**

```python
# BEFORE (problematic)
credential=AzureKeyCredential(os.environ["SEARCH_API_KEY"])

# AFTER (production-safe)
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
search_client = SearchClient(
    endpoint=os.environ["SEARCH_ENDPOINT"],
    index_name="documents",
    credential=credential
)
# Also replace openai.api_key with Azure OpenAI client using Managed Identity
from openai import AzureOpenAI
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    ),
    api_version="2024-05-01-preview"
)
```

> **Priority 2 — Retrieval Quality: Add Hybrid Search + Top-K + Source Tracking**

```python
from azure.search.documents.models import VectorizedQuery

def search_documents(query: str, top_k: int = 5) -> list[dict]:
    # Get query embedding
    embedding_response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_vector = embedding_response.data[0].embedding

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )

    results = search_client.search(
        search_text=query,           # BM25 keyword search
        vector_queries=[vector_query], # Dense vector search
        select=["content", "source_document", "section", "page_number"],
        top=top_k,
        query_type="semantic",
        semantic_configuration_name="my-semantic-config"
    )

    docs = []
    for r in results:
        docs.append({
            "content": r["content"],
            "source": r.get("source_document", "Unknown"),
            "section": r.get("section", ""),
            "score": r["@search.score"]
        })
    return docs
```

> **Priority 3 — Grounded Prompt + System Message + Token Budget**

```python
def generate_answer(query: str, context: list[dict]) -> dict:
    # Format context with source attribution
    context_parts = []
    for i, doc in enumerate(context):
        context_parts.append(
            f"[Source {i+1}: {doc['source']}, {doc['section']}]\n{doc['content']}"
        )
    context_str = "\n\n".join(context_parts)

    # Token budget guard — truncate if needed
    MAX_CONTEXT_CHARS = 6000
    if len(context_str) > MAX_CONTEXT_CHARS:
        context_str = context_str[:MAX_CONTEXT_CHARS] + "...[truncated]"

    system_message = (
        "You are a precise insurance knowledge assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, say 'I cannot find this information in the provided documents.' "
        "Always cite the source document for each factual claim."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ],
        temperature=0.0,  # Deterministic for factual Q&A
        max_tokens=500
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [d["source"] for d in context],
        "usage": response.usage.model_dump()
    }
```

> **Priority 4 — Error Handling and Retry Logic**

```python
import time
from openai import RateLimitError, APITimeoutError

def rag_pipeline(query: str, max_retries: int = 3) -> dict:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    for attempt in range(max_retries):
        try:
            docs = search_documents(query)
            if not docs:
                return {"answer": "No relevant documents found.", "sources": []}
            answer_data = generate_answer(query, docs)
            return answer_data
        except RateLimitError:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + (0.1 * attempt)  # exponential backoff with jitter
                time.sleep(wait)
            else:
                raise
        except APITimeoutError:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
```

> **Priority 5 — Observability: Structured Logging and Tracing**

```python
import logging
import time
import uuid

logger = logging.getLogger(__name__)

def rag_pipeline(query: str) -> dict:
    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info("rag_pipeline_start", extra={
        "trace_id": trace_id,
        "query_length": len(query)
    })

    try:
        docs = search_documents(query)
        answer_data = generate_answer(query, docs)

        elapsed = time.perf_counter() - start_time
        logger.info("rag_pipeline_complete", extra={
            "trace_id": trace_id,
            "elapsed_ms": round(elapsed * 1000),
            "docs_retrieved": len(docs),
            "tokens_used": answer_data.get("usage", {}).get("total_tokens")
        })

        answer_data["trace_id"] = trace_id
        return answer_data

    except Exception as e:
        logger.error("rag_pipeline_error", extra={
            "trace_id": trace_id,
            "error": str(e)
        })
        raise
```

---

#### Git Workflow Follow-Up

> **Interviewer**: *"How would you achieve a linear commit history in Git with a team of 6 engineers?"*

**✅ Model Answer**

> "Linear history is important in a team of this size because it makes `git bisect` reliable and makes the project history readable as a narrative rather than a tangle of merge commits. There are two complementary strategies: **rebase workflow** and **squash merges** at the PR level.
>
> Here's the team workflow I'd implement:
>
> **Branch strategy**: Every engineer works on a short-lived feature branch off `main`. Branch naming convention: `feat/TICKET-123-description`, `fix/TICKET-456-description`. Main is protected — no direct pushes.
>
> **Local rebase discipline**: Before raising a PR, the engineer rebases their branch onto the latest `main`:
> ```bash
> git fetch origin
> git rebase origin/main
> # resolve any conflicts, then:
> git push --force-with-lease origin feat/my-feature
> ```
> We use `--force-with-lease` rather than `--force` because it's safe — it fails if someone else has pushed to the branch since your last fetch.
>
> **PR merge strategy — Squash and Merge**: When a PR is approved, we merge it using GitHub's 'Squash and Merge' option. This takes all commits on the feature branch — even if the engineer made 12 WIP commits during development — and collapses them into a single, well-described commit on `main`. The commit message follows Conventional Commits format: `feat(underwriting): add hybrid search to retrieval agent (#234)`.
>
> **Why not regular merge commits?** Merge commits produce a non-linear history — you see two parents per commit, and `git log --oneline` becomes hard to follow. With squash merges, `main` is always a clean, linear sequence of meaningful commits.
>
> **Branch protection rules on GitHub/Azure DevOps**:
> - Require PR reviews (minimum 1 approval)
> - Require status checks to pass (CI pipeline: lint, tests)
> - Disallow merge commits — only squash or rebase merge allowed
> - Require branches to be up-to-date before merging
>
> **Team rule**: If your branch falls more than 2 working days behind `main`, you rebase before raising the PR. This keeps rebase conflicts small and tractable.
>
> The result: `git log --oneline main` reads like a project changelog — one line per feature or fix, ordered chronologically, always mergeable with `git cherry-pick`, and perfectly bisectable."

---

## MOCK SESSION 2: Challenging Variant

> [!WARNING]
> This session is deliberately harder. The interviewers are more sceptical, the problems are more ambiguous, and the follow-ups are more adversarial. This is the session to use when you feel comfortable with Session 1 and want to stress-test your answers.

---

### Harder Version of Each Part

---

#### Part 1 Challenge — Adversarial Opener

> **Interviewer**: *"AI Lead Engineer is a senior role. What specifically makes you qualified to lead — not just contribute to — an AI team?"*

**✅ Model Answer**

> "Leadership for me is about three things: technical direction, people development, and organisational navigation.
>
> On technical direction: I've made consequential architecture decisions that the team had to live with. The decision to switch from LangChain to Semantic Kernel mid-project cost us two weeks of refactoring but gave us an architecture that scaled from 200 to 1,200 users without a rewrite. I owned that call, I argued for it with evidence, and I stood behind it when it was painful. That's what technical leadership looks like.
>
> On people development: I've grown engineers. The junior ML engineer I mentioned earlier was writing notebook experiments with no production discipline when they joined the team. I worked with them on structured code review, design document habits, and test coverage standards. Eighteen months later they're leading the evaluation framework independently. I can point to specific people who are better engineers because of time they spent on my team.
>
> On organisational navigation: AI projects fail more often because of stakeholder misalignment than because of technical problems. I've learned to invest heavily upfront in agreeing on success metrics, escalation paths, and scope with product, risk, and leadership before writing a line of code. That's what gets AI systems into production rather than POC purgatory.
>
> I'm not perfect — I've made mistakes. I under-estimated the compliance timeline on a model risk review and it delayed our go-live by three weeks. I've since built regulatory review into every project plan as a first-class workstream rather than a final gate. Learning from that is also part of leadership."

---

#### Part 2 Challenge — Hard Failure Mode Scenario

> **Interviewer**: *"Your agent called the wrong tool three times in a row and issued a duplicate refund of £24,000. How did this happen, and how do you fix it?"*

**✅ Model Answer — Incident Analysis + Fix**

> "This is a serious incident. Let me do a structured root cause analysis, then give you the fix.
>
> **What likely happened — root cause chain**:
>
> The agent called the wrong tool because the tool schemas were ambiguous — two tools with similar descriptions and overlapping parameter names. The LLM routing logic (deciding which tool to call) confused them in the context of the query. This is a **tool schema validation failure**.
>
> The wrong tool was called, failed, and the agent retried — without checking whether the first call had partially succeeded. This is a **missing idempotency check**.
>
> The retry amplified the failure: same wrong tool, called again. On the third call, it appears the system interpreted a partial state as a 'refund not yet issued' and issued it again. This is **retry amplification without idempotency keys**.
>
> The result: £24,000 refund issued 2–3 times depending on exact retry count.
>
> **Immediate response (incident management)**:
> 1. Alert: notify finance to hold the duplicate refund processing — most payment systems have a short window to reverse
> 2. Disable the refund tool in the agent until the fix is deployed
> 3. Pull the full audit log for the incident — reconstruct the exact sequence of tool calls, arguments, and LLM reasoning trace
> 4. Communicate to stakeholders: what happened, what we did immediately, and when the fix will be deployed
>
> **Technical fixes**:
>
> *Fix 1 — Idempotency keys on all write operations*:
> ```python
> import uuid
>
> def issue_refund(claim_id: str, amount: float, idempotency_key: str = None) -> dict:
>     if idempotency_key is None:
>         idempotency_key = str(uuid.uuid4())
>     
>     # Check if this idempotency key has already been processed
>     existing = refund_store.get(idempotency_key)
>     if existing:
>         return {"status": "already_processed", "result": existing}
>     
>     # Process refund
>     result = payment_api.refund(claim_id=claim_id, amount=amount)
>     
>     # Store result against idempotency key (TTL: 7 days)
>     refund_store.set(idempotency_key, result, ttl=604800)
>     return {"status": "processed", "result": result}
> ```
>
> *Fix 2 — Tool schema disambiguation*:
> ```python
> # BEFORE (ambiguous)
> tools = [
>     {"name": "process_payment", "description": "Process a payment or refund"},
>     {"name": "issue_credit", "description": "Issue a credit to a customer"}
> ]
>
> # AFTER (unambiguous, with strict parameter schemas)
> tools = [
>     {
>         "name": "issue_refund",
>         "description": "Issue a refund for an APPROVED claim. Use ONLY when claim status is 'approved' and refund has NOT been issued. Do NOT use for credits, adjustments, or payments.",
>         "parameters": {
>             "type": "object",
>             "properties": {
>                 "claim_id": {"type": "string", "description": "The unique claim identifier"},
>                 "amount_gbp": {"type": "number", "description": "Refund amount in GBP"},
>                 "idempotency_key": {"type": "string", "description": "Unique key to prevent duplicate processing"}
>             },
>             "required": ["claim_id", "amount_gbp", "idempotency_key"]
>         }
>     }
> ]
> ```
>
> *Fix 3 — HITL gate for all financial write operations*:
> ```python
> FINANCIAL_TOOLS = {"issue_refund", "process_payment", "adjust_claim_value"}
>
> def execute_tool_call(tool_name: str, args: dict, confidence: float) -> dict:
>     if tool_name in FINANCIAL_TOOLS:
>         # Always require human approval for financial actions
>         approval = human_review_queue.submit({
>             "tool": tool_name,
>             "args": args,
>             "confidence": confidence,
>             "requires_approval": True
>         })
>         if not approval.approved:
>             return {"status": "rejected", "reason": approval.reason}
>     
>     return tool_registry[tool_name](**args)
> ```
>
> *Fix 4 — Retry budget with tool-level circuit breaker*:
> ```python
> MAX_RETRIES_PER_TOOL = 1  # For financial tools, max 1 retry
> FINANCIAL_RETRY_BUDGET = 0  # Actually, zero retries for financial tools — fail fast
>
> def agent_step(tool_name: str, args: dict) -> dict:
>     max_retries = 0 if tool_name in FINANCIAL_TOOLS else 3
>     for attempt in range(max_retries + 1):
>         try:
>             return execute_tool_call(tool_name, args)
>         except TransientError:
>             if attempt < max_retries:
>                 continue
>             raise
>         except Exception:
>             raise  # Never retry non-transient errors
> ```
>
> **Process fixes**:
> - Mandatory integration tests for every tool that writes state, including duplicate-call scenarios
> - Monthly chaos engineering exercises: deliberately inject tool failures to verify idempotency holds
> - Incident review published to engineering team — blameless post-mortem culture, shared learnings"

---

#### Part 3 Challenge — Broker Submission Routing System Design

> **Interviewer**: *"Design a system that processes unstructured insurance broker submissions — emails with attachments — and automatically routes them to the right underwriter with a risk score."*

**✅ Model Answer Walkthrough**

> **Clarifying questions**:
> "A few quick clarifications: What's the volume — how many submissions per day? What are the routing dimensions — class of business, geography, line size? What CRM or underwriting management system are we routing into? And what's the latency requirement — does the broker expect a response email within minutes or hours?"

> **Architecture** (assume: 500 submissions/day, route by CoB and line size, connect to a UMS like Sequel, response within 30 minutes):

```
[Broker Email] → [Microsoft 365 / Exchange] → [Logic App / Email Trigger]
                                                         │
                                                         ▼
                                            [Azure Service Bus: submissions]
                                                         │
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │   Extraction Agent           │
                                          │   (Azure Document Intel.)    │
                                          │   - Parse email body         │
                                          │   - Extract attachments      │
                                          │   - Classify: slip, MRC,     │
                                          │     spreadsheet, photo       │
                                          └──────────────┬───────────────┘
                                                         │
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │   Entity Extraction Agent    │
                                          │   (GPT-4o structured output) │
                                          │   - Insured name, broker     │
                                          │   - Class of business        │
                                          │   - Sum insured / TSI        │
                                          │   - Inception date           │
                                          │   - Geography / territory    │
                                          └──────────────┬───────────────┘
                                                         │
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │   Risk Scoring Agent         │
                                          │   - ML model: risk score     │
                                          │   - Catastrophe exposure     │
                                          │   - Limits / attachment pts  │
                                          │   - Historical loss data     │
                                          └──────────────┬───────────────┘
                                                         │
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │   Routing Agent              │
                                          │   - Match CoB + geography    │
                                          │     to underwriter capacity  │
                                          │   - Check workload balance   │
                                          │   - Route to UMS (Sequel)    │
                                          └──────────────┬───────────────┘
                                                         │
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │   Broker Acknowledgement     │
                                          │   (automated email reply)    │
                                          │   - Submission ref number    │
                                          │   - Assigned underwriter     │
                                          │   - Expected review timeline │
                                          └──────────────────────────────┘
```

> "Key design decisions:
>
> **Document Intelligence**: Azure Document Intelligence handles PDF slip extraction, table parsing from MRC submissions, and form recognition for standard broker formats. It returns structured key-value pairs which feed the entity extraction agent.
>
> **Entity extraction**: GPT-4o with structured output mode (JSON schema) extracts the key risk attributes. The schema is precisely defined — class of business must match a controlled vocabulary, TSI must be a number with currency, territory must match an ISO country list. Invalid extractions trigger a human review queue.
>
> **Risk scoring**: A gradient boosting model (XGBoost or LightGBM) trained on historical submission data, with features: class of business, TSI, territory, broker channel, attachment point relative to TSI. Output: a preliminary risk score (0–100) and a complexity flag (simple / referral needed). This is not a binding underwriting decision — it's a triage score that helps the underwriter prioritise their queue.
>
> **Routing**: Rules-based with ML tiebreaking. Rules: CoB → eligible underwriter pool. Geography → licensed underwriter. Line size → underwriter authority level. ML tiebreaker: workload balance model that predicts which underwriter in the eligible pool has capacity. Route goes to the UMS via API.
>
> **Fallback**: Any submission where extraction confidence is below threshold, or where no eligible underwriter is found, goes to a triage queue for a senior underwriter to manually assign."

---

#### Part 4 Challenge — Buggy Agent Loop

> **Interviewer**: *"Here's a simplified agent loop. What's wrong with it and how would you fix it?"*

```python
def agent_loop(user_query: str, tools: dict) -> str:
    messages = [{"role": "user", "content": user_query}]
    
    while True:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            functions=list(tools.values())
        )
        
        message = response.choices[0].message
        
        if message.get("function_call"):
            tool_name = message["function_call"]["name"]
            tool_args = json.loads(message["function_call"]["arguments"])
            
            tool_result = tools[tool_name](**tool_args)
            
            messages.append(message)
            messages.append({
                "role": "function",
                "name": tool_name,
                "content": str(tool_result)
            })
        else:
            return message["content"]
```

**✅ Model Answer**

> "I can see **six problems** here. Let me go through them in order of severity.
>
> **Problem 1 — Infinite loop with no exit condition (Critical)**:
> The `while True` loop has only one exit: the `else` branch when no function call is made. If the model gets stuck in a tool-calling loop — which happens — this runs forever. Fix: add a `max_iterations` counter and break with a graceful error message if exceeded.
>
> **Problem 2 — No error handling on tool execution (Critical)**:
> `tools[tool_name](**tool_args)` will raise a `KeyError` if the model hallucinates a tool name that doesn't exist in the `tools` dict. It will raise any exception the tool raises. None of these are caught. Fix: wrap in try/except, append an error result to messages so the agent can recover.
>
> **Problem 3 — No JSON parse error handling (High)**:
> `json.loads(message["function_call"]["arguments"])` will raise `JSONDecodeError` if the model emits malformed JSON — which happens, especially with long or complex argument structures. Fix: wrap in try/except JSONDecodeError.
>
> **Problem 4 — Deprecated API usage (Medium)**:
> `openai.ChatCompletion.create` with `functions=` is the legacy function-calling API. The current API uses `tools=` with `tool_choice` and `tool_calls` (plural) in the response. The new API also supports parallel tool calls — multiple tools in a single response. This code will break or behave unexpectedly on the latest SDK version.
>
> **Problem 5 — Context window growth with no management (Medium)**:
> Every tool call appends to `messages`. In a long agentic session with many tool calls, the context window fills up and you get a `ContextLengthExceededError`. Fix: implement context management — either truncate old messages, summarise the conversation history, or use a sliding window.
>
> **Problem 6 — No observability (Low)**:
> No logging, no trace ID, no timing. If this fails in production, you have nothing to debug with. Fix: add structured logging around every LLM call and tool execution.
>
> **Fixed version**:

```python
import json
import logging
import time
import uuid
from openai import AzureOpenAI, BadRequestError

logger = logging.getLogger(__name__)

def agent_loop(
    user_query: str,
    tools: dict,
    tool_schemas: list,
    max_iterations: int = 10
) -> str:
    trace_id = str(uuid.uuid4())
    messages = [{"role": "user", "content": user_query}]
    
    logger.info("agent_loop_start", extra={"trace_id": trace_id, "query": user_query[:100]})
    
    for iteration in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tool_schemas,  # Current API format
                tool_choice="auto"
            )
        except BadRequestError as e:
            logger.error("llm_call_failed", extra={"trace_id": trace_id, "error": str(e)})
            return "I encountered an error processing your request. Please try again."
        
        message = response.choices[0].message
        messages.append(message.model_dump())  # append assistant message
        
        # Check for tool calls (new API: tool_calls, not function_call)
        if message.tool_calls:
            for tool_call in message.tool_calls:  # supports parallel tool calls
                tool_name = tool_call.function.name
                
                # Parse arguments safely
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    tool_result = f"Error: could not parse tool arguments: {e}"
                    logger.warning("tool_arg_parse_error", extra={"trace_id": trace_id, "tool": tool_name})
                else:
                    # Execute tool safely
                    if tool_name not in tools:
                        tool_result = f"Error: unknown tool '{tool_name}'"
                        logger.warning("unknown_tool_called", extra={"trace_id": trace_id, "tool": tool_name})
                    else:
                        try:
                            tool_result = tools[tool_name](**tool_args)
                        except Exception as e:
                            tool_result = f"Tool execution error: {e}"
                            logger.error("tool_execution_error", extra={"trace_id": trace_id, "tool": tool_name, "error": str(e)})
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
        else:
            # No tool call — final answer
            logger.info("agent_loop_complete", extra={"trace_id": trace_id, "iterations": iteration + 1})
            return message.content
    
    # Max iterations exceeded
    logger.warning("agent_loop_max_iterations", extra={"trace_id": trace_id})
    return "I wasn't able to complete this task within the allowed steps. Please rephrase or break it into smaller questions."
```

---

## Mock Interview Self-Assessment Rubric

After each practice session, score yourself honestly on every dimension. Target ≥ 4 on all dimensions before the real interview.

| Dimension | 1 — Poor | 2 | 3 — Acceptable | 4 | 5 — Excellent |
|---|---|---|---|---|---|
| **Opening headline** | Dived straight into chronological CV | Mixed — some summary | Clear conclusion first, brief proof | Summary + key proof points | Crisp 90-second narrative: role → scale → relevance → Brit |
| **Technical depth** | Surface-level: "we used GPT-4" | Named tools without rationale | Architecture described, some trade-offs | Architecture + trade-offs + alternatives considered | Full system design with explicit trade-off reasoning and quantified decisions |
| **Leadership signals** | "I" did everything, no team mention | Mentioned "the team" vaguely | Some team composition mentioned | Clear role differentiation, 1–2 people development examples | Specific people grown, decisions owned, conflicts navigated, outcomes attributed to team |
| **Communication** | Filler words, stream-of-consciousness, scattered | Some structure, occasional rambling | Mostly clear, STAR used, some filler | Clear structure, minimal filler, good pacing | Structured, confident, no filler, calibrated pace — slows for complexity, crisp for summaries |
| **Quantified outcomes** | No numbers anywhere | 1–2 vague numbers | Some numbers, not all impactful | Key outcomes quantified (time saved, users, error rate) | Specific, credible metrics throughout: users, latency, accuracy, cost, business impact |
| **Insurance domain awareness** | No insurance knowledge demonstrated | General "financial services" awareness | Basic insurance concepts (claims, underwriting) | Lloyd's, specific line classes, regulatory names (FCA, PRA) | Ki Insurance, catastrophe analytics, PML, Lloyd's syndicate structure, slip market, Broker API strategy |
| **System design structure** | Drew boxes randomly, no method | Clarifying questions skipped, partial design | Clarifying questions asked, major components covered | Full architecture with scaling and monitoring | Clarifying questions + phased design + deep-dive + failure modes + monitoring + quantified decisions |
| **Coding analysis** | Read code out loud, no insight | Spotted 1–2 issues | 3–4 issues with some fixes | 5 issues, prioritised, with code fixes | 6 issues, prioritised by severity/impact, clean fix code, explains *why* each matters in production |

---

## Practice Drills — 10-Minute Exercises

Use these daily in the week before the interview. Set a timer. Do each one out loud.

---

### Drill 1 — Explain RAG to a Non-Technical Person (60 seconds)

**Prompt**: *Your interviewer's assistant asks what RAG is. Explain it simply.*

**Target answer (adapt this)**:
> "Imagine you have a very smart assistant who knows a lot in general, but doesn't have access to your company's specific documents. RAG — Retrieval-Augmented Generation — gives that assistant a library card. When you ask a question, the system first searches your specific documents for the most relevant pages, then gives those pages to the AI assistant along with your question, so it can give you an answer based on your actual content rather than just its general knowledge. The key benefit is that the answer is grounded in real documents — so you can check its sources, and it won't make things up."

**Self-assessment**: Did you use a concrete analogy? Did you avoid jargon? Could a non-technical person repeat the explanation to someone else?

---

### Drill 2 — Technical Failure in STAR Format (2 minutes)

**Prompt**: *"Tell me about a significant technical mistake you made and what you learned."*

**Structure to follow**:
- **Situation** (15 sec): What was the system, what was the stakes?
- **Task** (10 sec): What were you trying to do?
- **Action** (45 sec): What did you decide? Why? What went wrong?
- **Result** (30 sec): What was the impact? What was the fix?
- **Learning** (20 sec): What changed in your approach after this?

**Example scenario to adapt**:
> Deployed a new embedding model version without running full regression tests first. Retrieval quality degraded silently — faithfulness dropped from 0.87 to 0.71 over 3 days. Users noticed via thumbs-down rate increase. Rolled back the model. Root cause: we had automated eval running weekly but not on every deployment. Fix: added eval as a mandatory CI gate — every model or index change must pass eval suite before merge. Zero silent degradations since.

**Self-assessment**: Did you take clear ownership without being self-flagellating? Did you end on the learning, not the failure?

---

### Drill 3 — Solo Image Anomaly System Design (15 minutes)

**No notes. Paper and pen only.**

1. Spend 1 minute writing clarifying questions you'd ask
2. Spend 5 minutes drawing the architecture from memory with component labels
3. Spend 3 minutes adding scaling and failure handling annotations
4. Spend 3 minutes writing the monitoring components
5. Spend 3 minutes preparing your answer to: "What if the model is wrong 15% of the time?"

**Check your diagram contains**:
- [ ] Blob Storage for raw images
- [ ] Event Grid trigger
- [ ] Service Bus queue
- [ ] Preprocessing worker with tiling
- [ ] Inference layer with GPU auto-scaling
- [ ] Results store
- [ ] Results API
- [ ] Adjuster review UI
- [ ] Audit log
- [ ] Dead letter queue
- [ ] GradCAM explainability layer

---

### Drill 4 — RAG Code Review (10 minutes)

Read the sample RAG code in Part 4 (without looking at the model answer). Set a 10-minute timer and write down:
- Your 30-second summary
- Your walk-through of each function
- Your top 5 prioritised improvements with brief code sketches

Then compare your answer to the model answer. **Score yourself**: How many of the 5 improvements did you identify? Did you prioritise security first?

**Common misses**:
- Forgetting to mention the missing system message (grounds the model)
- Not mentioning `temperature=0.0` for factual Q&A
- Missing the lack of token budget management (context overflow)
- Not catching the deprecated `ChatCompletion.create` API

---

### Drill 5 — Team Leadership Story (2 minutes)

**Prompt**: *"Tell me about a team member you mentored and the outcome."*

**Structure**:
- Who was the person (role, experience level)? (15 sec)
- What was the gap you identified? (20 sec)
- What did you specifically do to develop them? (40 sec)
- What was the measurable outcome? (25 sec)
- What did you learn about mentoring? (20 sec)

**Example to adapt**:
> Junior ML engineer, 1 year experience, strong Python skills but no production ML discipline — notebooks everywhere, no tests, no logging. I paired with them on the evaluation framework: we built it together, I led the first session, they drove the second, I reviewed and gave structured written feedback on their code. I also introduced them to design document culture — they wrote their first design doc for the chunking strategy, I reviewed it in writing and gave specific feedback on what was missing (failure modes, alternatives considered). Six months later they ran a production deployment independently with zero issues and wrote a runbook that I now use as the team standard. I learned that the best mentoring is specific and technical, not generic advice.

**Self-assessment**: Was there a specific, observable behaviour change? Did you describe *what you did*, not just what happened? Did you end with a learning?

---

> [!TIP]
> Run Mock Session 1 timed at least twice before the interview. Run Mock Session 2 once. Use the rubric after every session. Focus your final 48 hours on the two dimensions where you score lowest — usually quantified outcomes and insurance domain specificity.

---

> [!IMPORTANT]
> **The one thing that separates good from great in this interview**: In every answer, ask yourself — *"Have I said what I built, why I made that decision, what the trade-offs were, and what the measurable outcome was?"* If all four are in the answer, you're in excellent shape. If any one is missing, the answer is incomplete.
