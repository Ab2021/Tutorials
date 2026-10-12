# Brit Insurance Stage 1 — Interview Playbook: Minute-by-Minute Strategy

> [!IMPORTANT]
> This is a **1-hour structured interview**. Every minute is allocated. Know the format cold — your pacing IS part of the signal you send about seniority.

---

## 📋 Interview Format at a Glance

| Part | Duration | Topic | Your Goal |
|------|----------|-------|-----------|
| Part 1 | 10 min | Introductions | Establish credibility & narrative hook |
| Part 2 | 25 min | Agentic AI Project Deep-Dive | Demonstrate technical depth + leadership |
| Part 3 | 15 min | System Design | Show architectural thinking + insurance awareness |
| Part 4 | 10 min | Coding Exploration (RAG) | Demonstrate code literacy + critical thinking |

> [!NOTE]
> The interviewer controls the clock, not you. But *your density of signal per minute* determines whether they remember you as "strong yes" or "maybe". Every part has its own win condition — don't bleed time from one part into the next.

---

## PART 1 — Introduction Strategy (10 Minutes)

### The Core Goal

In 10 minutes you must land **three things** in the interviewer's mind:
1. **Leader** — You have managed teams, made decisions, been accountable for outcomes.
2. **Builder** — You have shipped production AI systems, not just prototypes.
3. **Domain Expert** — You understand the constraints of regulated, high-stakes domains (insurance adjacent or directly relevant).

If all three are registered by minute 10, the remaining 50 minutes are about depth, not credibility.

---

### The 2-Minute Introduction: Template Script

> [!TIP]
> Deliver this in **exactly 90–120 seconds**. Practice it until you can say it without thinking. The goal is to use the remaining 8 minutes for **their questions**, not your monologue.

```
"I'm [NAME] — currently a Lead AI Engineer at [COMPANY], where I lead 
a team of [N] engineers building [SHORT DESCRIPTION OF SYSTEM].

Over the past [X] years I've focused specifically on [agentic AI / 
multi-model systems / LLM orchestration] — the discipline of getting 
language models to take reliable, auditable actions in production.

Before that, at [PREV COMPANY], I built [RELEVANT SYSTEM] which handled 
[SCALE METRIC — e.g., "50k decisions per day" / "$2M in daily transactions"].

What drew me to Brit Insurance specifically is [ONE SPECIFIC THING — 
underwriting automation / claims AI / Lloyds market context]. I believe 
AI in insurance has to be explainable and reliable above all else — 
and that's the lens I've been working from for [X] years.

I'm excited to go deep on the agentic project we'll discuss — it 
captures a lot of what I think is relevant here."
```

**Placeholders to fill before the interview:**

| Placeholder | Your Answer |
|-------------|-------------|
| `[N]` engineers | Exact number — don't round up |
| `[SHORT DESCRIPTION]` | ≤ 10 words describing your system's purpose |
| `[SCALE METRIC]` | One hard number — requests/day, $, users, claims |
| `[ONE SPECIFIC THING]` | Research Brit's AI initiatives — pick one real thing |
| `[X] years` | Be precise — "4 years" not "several years" |

---

### How to Set Up the Narrative: Position Yourself as Leader + Builder + Domain Expert

**The Narrative Arc (not a script — a shape):**

```
Past ──────────────────► Present ──────────► Future at Brit
  │                          │                     │
  ▼                          ▼                     ▼
Technical         Leadership +              Insurance AI
Foundation     Production Scale            at Enterprise Scale
```

- **Don't start with education.** Start with current impact.
- **Don't list technologies.** Connect technologies to business outcomes.
- **Do mention team size.** "Lead" without a team is just a title.
- **Do mention a failure you learned from.** It signals maturity. Save details for Part 2.

---

### What to Mention (Checklist)

- [ ] Team size you've directly managed (engineers, not total org)
- [ ] Scale of AI systems (RPM, daily users, data volume, $ at stake)
- [ ] Any work in regulated domains: insurance, fintech, healthcare, legal
- [ ] That you care about **reliability and auditability** — these are Brit's language
- [ ] A forward-looking hook: one sentence on what you'd bring to their specific challenge

---

### What NOT to Say (Common Mistakes)

> [!WARNING]
> These mistakes signal junior/mid-level thinking in the first 2 minutes — very hard to recover from.

| ❌ Don't Say | ✅ Say Instead |
|-------------|--------------|
| "I've worked with a lot of LLMs..." | "I've run production multi-agent systems at scale..." |
| "I'm really passionate about AI..." | "I've been building [specific thing] for X years..." |
| "I'm familiar with insurance..." | "In regulated domains I've focused on explainability because..." |
| "My team and I tried to build..." | "I led the team that shipped [outcome]..." |
| "I don't have direct insurance experience but..." | *(Never open with a deficit — lead with what you have)* |
| Listing tools: "I use LangChain, Pinecone, OpenAI..." | "I chose X over Y because at our scale..." |

---

### The 8 Minutes After Your Intro

They will ask follow-up questions. Common ones and how to handle them:

| Their Question | Your Framework |
|----------------|---------------|
| "Tell me more about [your current role]..." | 30 sec more detail + bridge to Part 2 project |
| "What do you know about Brit Insurance?" | Lead with Lloyds of London market, mention their AI investment thesis |
| "Why are you leaving [current company]?" | Growth/impact framing, never negative |
| "What's your experience with insurance specifically?" | Map your regulated domain experience directly — don't apologize |

---

## PART 2 — Project Deep-Dive Strategy (25 Minutes)

### The Lead-Level STAR Framework

At a Lead level, STAR is not enough. Use **STARLA**:

| Letter | Stands For | What to Include |
|--------|-----------|-----------------|
| **S** | Situation | Business context, why it mattered, scale |
| **T** | Task | Your specific mandate, not the team's |
| **A** | Action | What *you* decided and why, alternatives considered |
| **R** | Result | Hard metrics, before vs. after |
| **L** | Leadership | How you grew/enabled the team |
| **A** | After | What you'd do differently — this signals growth mindset |

> [!NOTE]
> Interviewers at Lead/Principal level grade you on whether you can explain **why** you made decisions — not just what you did. Every architectural or process choice needs a rationale.

---

### Time Budget for 25 Minutes

```
[0:00 – 2:00]  ──  Overview: What the project was, why it mattered
[2:00 – 7:00]  ──  Architecture walkthrough (your choices + tradeoffs)
[7:00 – 12:00] ──  Team leadership moments (how you managed)
[12:00 – 17:00]──  Evaluation + production challenges
[17:00 – 22:00]──  Business impact (quantifiable)
[22:00 – 25:00]──  Q&A follow-ups, go deeper on their focus area
```

---

### The 5 Drill Areas — Preparation Templates

#### Drill Area 1: Architecture Decisions

**What they're asking:** *Did you make real architectural choices or just implement what someone else decided?*

**Framework:** **Choice → Alternatives Rejected → Tradeoff Owned**

```
"We chose [ARCHITECTURE PATTERN] over [ALTERNATIVE] because 
at our scale of [METRIC], [SPECIFIC REASON]. The tradeoff 
was [COST/DOWNSIDE], which we accepted because [BUSINESS REASON].

In hindsight I'd [CHANGE or KEEP] because [LEARNED SOMETHING]."
```

**Example (Agentic System):**
```
"We chose a supervisor-worker agent pattern over a flat ReAct loop 
because we had 12 distinct tool domains and a flat agent was 
generating tool-selection hallucinations 18% of the time. 

The tradeoff was added orchestration latency — about 400ms per 
supervisor round-trip. We accepted that because accuracy was 
the primary KPI, not speed.

In hindsight I'd also have implemented circuit breakers at the 
tool level from day one — we added them reactively after a 
third-party API outage cascaded into the supervisor."
```

---

#### Drill Area 2: Team Leadership

**What they're asking:** *Are you actually a lead or a senior who got a title?*

**Framework:** **Context → Decision Made for the Team → Outcome for Each Person**

Questions to prepare for:

| Question | Your Template Answer |
|----------|---------------------|
| "How did you manage disagreements about technical direction?" | Describe one real disagreement. Outcome: team aligned, person grew. |
| "How did you handle a low performer?" | Specific: what you observed → what you said → how it resolved |
| "How did you grow someone junior?" | Pair programming, code review culture, 1:1 structure |
| "How did you prioritize when everything was urgent?" | Explain your triage framework + stakeholder communication |

**Bridge for team leadership questions:**
```
"That's actually a great example of something I learned during 
this project. Let me walk you through a specific moment..."
```

---

#### Drill Area 3: Evaluation Methodology

**What they're asking:** *Do you actually know if your AI system is working?*

> [!IMPORTANT]
> This is the most commonly fumbled area for ML engineers. You must have a **layered evaluation story** — not just accuracy.

**The Three Layers of AI Evaluation (memorize this):**

```
Layer 1: Component-Level
  └── Unit tests on prompts, tools, retrieval pipelines
  └── Regression suites on known-bad inputs

Layer 2: System-Level  
  └── End-to-end task completion rate
  └── Human eval on 5% of production traffic
  └── Latency p50/p95/p99

Layer 3: Business-Level
  └── User adoption rate
  └── Escalation rate (how often humans had to override)
  └── Dollar impact: cost saved, revenue enabled
```

**Template Answer:**
```
"We had evaluation at three levels. At the component level, we 
maintained a regression suite of 400 'golden' input-output pairs 
that ran on every deployment — this caught prompt regressions from 
model updates three times.

At the system level, we sampled 5% of production agent runs for 
async human review. Our quality team graded on task completion, 
hallucination rate, and citation accuracy. We ran this weekly.

At the business level, the KPI was time-to-resolution on [DOMAIN] 
cases — we moved from [X] hours to [Y] hours, which translated 
to [$ or FTE impact]."
```

---

#### Drill Area 4: Production Challenges

**What they're asking:** *Have you actually run AI in production or just in notebooks?*

**The Production Failure Story Framework:**

```
1. What broke (specific, honest)
2. How you detected it (monitoring/alerting — not a user complaint if possible)
3. Immediate mitigation (what you did in the first hour)
4. Root cause (what actually caused it)
5. Prevention (what system change ensures it never happens that way again)
```

**Example Story:**
```
"The worst production incident we had was a prompt injection attack 
through user-uploaded documents. An adversarial user embedded 
instructions in a PDF that caused our extraction agent to invoke 
the data deletion tool.

We detected it via our action audit log — we had an alert on 
any invocation of destructive tools during a read-only session.
 
Immediate mitigation: we rate-limited document-triggered tool 
invocations within 20 minutes and pushed a hotfix to sanitize 
PDF text before it entered the prompt context.

Root cause: we hadn't sandboxed the document content from the 
instruction channel in our prompt architecture.

Prevention: we refactored all prompts to use XML tags with strict 
delimiters and added a pre-processing step that strips instruction-
like patterns from external content. We also added this scenario 
to our red-teaming suite."
```

---

#### Drill Area 5: Business Impact

**What they're asking:** *Are you a business-connected engineer or an academic?*

> [!TIP]
> Always have **three numbers** ready: a volume metric, an efficiency metric, and a cost/revenue metric.

**Template:**
```
"The project delivered three measurable outcomes:

1. Volume: We processed [X] [decisions/documents/claims] per day 
   that previously required manual handling.

2. Efficiency: Average handling time dropped from [X] to [Y] — 
   a [Z%] reduction.

3. Financial: Conservative estimate from finance was [$X] in 
   annual savings / [Y%] increase in throughput without headcount 
   growth.

And one qualitative outcome that mattered: [stakeholder name] 
used the system as a reference case for [larger initiative]."
```

---

### Handling Follow-Up Questions: The "Thread + Depth" Framework

When they ask a follow-up, don't panic and don't pad. Use this three-step structure:

```
1. HEADLINE (5 sec):   Answer the question in one sentence first.
2. EVIDENCE (30 sec):  Give one specific example or data point.
3. OFFER MORE (5 sec): "I can go deeper on [X] if useful."
```

This keeps you in control and signals you have more to give without dumping everything.

---

### Bridge Statements: Buy Time Gracefully

> [!NOTE]
> Bridge statements are not filler — they are **professional pacing tools**. Senior engineers use them deliberately.

| Situation | Bridge Statement |
|-----------|-----------------|
| You need 3 seconds to think | "That's a good question — let me make sure I give you the precise answer..." |
| You don't know the exact answer | "I don't have that exact number in my head, but the order of magnitude was... and I can follow up with the precise figure." |
| You disagree with their premise | "That's an interesting framing. My experience was slightly different — here's what I observed..." |
| You're going off-topic | "To stay focused on what's most relevant here — the key thing was..." |
| You want to redirect to a strength | "That specific scenario didn't come up, but an analogous challenge we faced was..." |

---

### Common Project Deep-Dive Questions & Model Answers

**Q: "What would you do differently if you started over?"**
```
"Three things. First, I'd instrument the observability layer before 
the first prototype — we added logging reactively and it cost us 
two sprints. Second, I'd define evaluation criteria with stakeholders 
before writing a line of model code — alignment on what 'good' means 
prevents scope creep. Third, I'd have a formal model-card process 
from the start — in a regulated context, documentation debt is 
technical debt."
```

**Q: "How did you handle model updates from your provider breaking your system?"**
```
"We implemented a model pinning strategy — our production system 
ran on a specific model version with an explicit upgrade review 
gate. Any provider model update triggered our full regression suite 
before we'd cut over. We also maintained a shadow deployment 
running the new version for 2 weeks before promotion. This caught 
a behavioral shift in [specific scenario] that would have caused 
a 12% degradation in our extraction accuracy."
```

**Q: "How did you ensure the system was fair/unbiased?"**
```
"We defined fairness slices early — demographic groups or case types 
where performance had to be within X% of the population baseline. 
We ran our eval suite disaggregated by these slices every sprint. 
We also conducted a red-team exercise specifically looking for 
disparate impact — [RESULT]. In a regulated domain like insurance, 
this isn't optional — it's a compliance requirement, and we built 
it into the development process, not as a post-launch audit."
```

---

## PART 3 — System Design Strategy (15 Minutes)

### The Actual Question Asked: Image Anomaly Detection System

**Full question (reconstructed):**
> *"Design a system that can detect anomalies in images submitted as part of insurance claims — for example, identifying AI-generated images, staging, or inconsistencies that suggest fraud."*

---

### 15-Minute Structure (Strict)

```
[0:00 – 2:00]  Clarify requirements (ask 3 targeted questions)
[2:00 – 5:00]  High-level architecture (draw/describe components)
[5:00 – 10:00] Component deep-dive (the meaty part)
[10:00 – 13:00] Scaling & reliability
[13:00 – 15:00] Monitoring & governance
```

> [!IMPORTANT]
> Start with requirements clarification. Jumping straight to a solution without clarifying is a **junior signal**. Senior engineers define the problem before solving it.

---

### Step 1: Requirements Clarification (2 min)

Ask exactly these three questions:

```
1. "What's the expected volume? Images per day, and peak burst rate?"
   → This determines sync vs. async processing, batch vs. streaming.

2. "What's the latency requirement — do claims adjusters need 
   real-time results during upload, or is async batch acceptable?"
   → This determines architecture shape entirely.

3. "What are the sensitivity/specificity tradeoffs? Is a false 
   positive (flagging a legitimate claim) or false negative 
   (missing fraud) more costly?"
   → This determines your threshold calibration and escalation design.
```

**Implied answers to assume (if they say "you decide"):**
- Volume: 50,000 images/day, peak 5,000/hour
- Latency: Async is fine, result within 5 minutes of upload
- Tolerance: False negatives are more costly (insurance context — fraud is expensive)

---

### Step 2: High-Level Architecture (3 min)

```
                    ┌─────────────────────────────────┐
                    │         INGESTION LAYER          │
  Upload API ──────►│  API Gateway → Queue (SQS/Kafka) │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │      PREPROCESSING LAYER         │
                    │  Resize / EXIF strip / Hash-dedup│
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       DETECTION LAYER            │
                    │  ┌──────────┐  ┌──────────────┐ │
                    │  │ AI-Gen   │  │ Consistency  │ │
                    │  │ Detector │  │ Analyzer     │ │
                    │  └──────────┘  └──────────────┘ │
                    │  ┌──────────┐  ┌──────────────┐ │
                    │  │ Metadata │  │ Duplicate    │ │
                    │  │ Analyzer │  │ Detector     │ │
                    │  └──────────┘  └──────────────┘ │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       SCORING & DECISION LAYER   │
                    │  Ensemble scorer → Threshold →   │
                    │  [Auto-Clear | Flag | Escalate]  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       AUDIT & GOVERNANCE LAYER   │
                    │  Decision log / Explainability / │
                    │  Human review queue              │
                    └─────────────────────────────────┘
```

**Say out loud while sketching:**
```
"I'm thinking async pipeline — images come in through an API 
gateway, get queued, then processed by parallel detection 
modules. The output is an ensemble risk score that feeds 
into a three-way decision: auto-clear, flag for review, 
or immediate escalation. Everything is logged for audit."
```

---

### Step 3: Component Deep-Dive (5 min)

Talk through each detection module and WHY it exists:

#### 3a. AI-Generated Image Detector
```
Approach: Fine-tuned binary classifier (e.g., EfficientNet or 
          ViT) on dataset of real vs. GAN/diffusion-generated images.

Why not off-the-shelf API? 
  - Data sovereignty: claims images are PII, can't send to external APIs
  - Model drift: AI generation techniques evolve fast — need retraining 
    pipeline
  - Cost at scale: 50K images/day at API pricing is prohibitive

Training data: Mix of LAION subsets + domain-specific real claim photos
Update cadence: Retrain monthly with flagged false negatives from 
                human review
```

#### 3b. Metadata / EXIF Consistency Analyzer
```
What it checks:
  - EXIF timestamp vs. claimed incident date/time
  - GPS coordinates vs. claimed incident location
  - Camera make/model consistency across multi-image claims
  - File creation date vs. upload date
  - Software tag (Photoshop, DALL-E signatures in EXIF)

Implementation: Rule-based + heuristic (fast, interpretable)

Insurance relevance: This is a major real fraud vector — 
                     stock photos have EXIF from years ago.
```

#### 3c. Image Consistency Analyzer (Forgery / Staging)
```
What it checks:
  - ELA (Error Level Analysis): detects JPEG re-compression from editing
  - Clone detection: patches copy-pasted within an image
  - Illumination inconsistency: lighting direction differs between 
    objects (suggests compositing)
  - Physical plausibility: damage patterns that don't match claimed 
    incident type

Implementation: CNN-based segmentation + classical CV (ELA, frequency 
                analysis) — ensemble for robustness
```

#### 3d. Cross-Claim Duplicate Detector
```
What it checks:
  - Perceptual hash (pHash) comparison against historical claim images
  - Catches: same damage photo submitted across multiple claims 
             (by same or different policyholders)

Implementation: pHash → approximate nearest neighbor search 
                (FAISS or Pinecone) with 0.95 similarity threshold

Insurance relevance: This is a known fraud ring pattern — 
                     one "stock" damage photo used for many claims.
```

#### 3e. Ensemble Scorer & Decision Engine
```
Inputs: Probability scores from each detection module
Output: Composite risk score [0.0 – 1.0]

Weighting: Not equal — AI-gen detection weighted 2x during 
           high-AI-fraud periods (adjustable)

Thresholds:
  [0.0 – 0.3]  → Auto-clear, no flag
  [0.3 – 0.7]  → Soft flag, queue for human review within 48h
  [0.7 – 1.0]  → Hard flag, escalate to SIU (Special Investigations Unit)

Threshold calibration: Reset quarterly using ROC curve analysis 
                       on human-reviewed cases
```

---

### Step 4: Scaling & Reliability (3 min)

```
Scaling:
  - Detection workers are stateless containers → horizontal autoscaling
  - Queue depth is the scaling signal (not CPU) — more images = 
    more workers
  - pHash index (FAISS) is read-heavy → cache in memory, 
    write-through to persistent store

Reliability:
  - Dead letter queue for images that fail processing > 3 times
  - Fallback: if detection layer fails, image goes to mandatory 
    human review (fail-safe, not fail-open)
  - Regional redundancy for GDPR/data residency (UK images stay in UK)

SLA:
  - 99.9% of images processed within 5 minutes of upload
  - Human review queue capped at 48-hour SLA
```

---

### Step 5: Monitoring & Governance (2 min)

> [!IMPORTANT]
> This is where you demonstrate **insurance domain awareness**. Governance is not optional in regulated environments — build it in from the start.

```
Monitoring:
  - False positive rate tracked weekly against human review outcomes
  - Model drift dashboard: distribution shift in input images 
    (catches when fraud patterns change)
  - Throughput / queue depth / processing latency dashboards
  - Alert: if FPR exceeds 5%, auto-review of threshold calibration

Governance:
  - Every decision is logged with: image ID, scores per module, 
    ensemble score, decision, model version
  - Explainability: heatmaps (Grad-CAM) stored alongside each 
    flagged image — regulators/SIU need to know WHY it was flagged
  - Audit trail immutable — write-once storage (S3 Object Lock or 
    equivalent)
  - Model card maintained for each model version: training data, 
    known failure modes, fairness evaluation
  - Quarterly bias audit: does the system flag claims differently 
    based on claim type, geography, or policyholder demographics?
```

---

### Insurance-Specific Considerations to Always Include

> [!NOTE]
> Drop these into any insurance AI design — they signal domain fluency immediately.

- **Data residency**: UK/EU insurance data often cannot leave the region — mention this
- **FCA/PRA compliance**: Financial Conduct Authority requires explainability for adverse decisions
- **SIU integration**: Special Investigations Unit is the human escalation path for fraud
- **Audit immutability**: Claims decisions must be reproducible years later for litigation
- **Adverse action explainability**: If a claim is denied/flagged, the reason must be explainable to the policyholder
- **Model governance cadence**: Quarterly model review is common in regulated insurers
- **Fail-safe over fail-open**: When AI is uncertain, default to human review — never auto-deny

---

## PART 4 — Coding Exploration Strategy (10 Minutes)

### The Format

They will likely show you a RAG (Retrieval-Augmented Generation) code snippet and ask you to:
1. Explain what it does
2. Identify issues or improvements
3. Possibly discuss how you'd test it

---

### Your Explanation Structure: What → How → What I'd Improve

```
"Let me walk through this in three passes:

First pass: What this code does at a high level.
  → [1-2 sentences on purpose, not implementation]

Second pass: How it does it, component by component.
  → [Walk through each function/block, name the pattern]

Third pass: What I'd improve.
  → [Structured critique — see below]
```

This structure signals that you're a **code reviewer, not just a coder** — which is the Lead-level expectation.

---

### The 5 Things to Look for in Any AI Code

Memorize this checklist. Scan the code against it:

| # | Check | What to Look For |
|---|-------|-----------------|
| 1 | **Error handling** | Are API calls wrapped? What happens on timeout/failure? |
| 2 | **Prompt hygiene** | Is user input sanitized before it enters the prompt? Injection risk? |
| 3 | **Observability** | Is anything logged? Can you trace a request end-to-end? |
| 4 | **Scalability** | Is there synchronous blocking where async would work? N+1 retrieval? |
| 5 | **Evaluation hooks** | Is there any way to measure quality of outputs? Ground truth? |

---

### How to Give Improvements Without Being Negative

> [!TIP]
> Frame all critique as **collaborative improvement**, not judgment. You're reviewing a colleague's PR, not grading a student.

**Critique Framework:**
```
"This is solid for [specific thing]. One thing I'd add is [improvement], 
because in production [specific scenario] can cause [specific problem]. 
The fix would be [concrete suggestion]."
```

**Example (RAG code with no error handling on retrieval):**
```
"The retrieval logic looks clean and the chunking strategy is reasonable. 
One thing I'd strengthen is the error handling around the vector store 
call — if the retrieval service is unavailable, right now we'd surface 
a raw exception to the user. I'd wrap this in a try/except with a 
fallback: either return a 'context unavailable' response that the 
LLM can gracefully handle, or retry with exponential backoff. In 
production, retrieval latency spikes are common and you don't want 
that to cascade into a user-facing error."
```

---

### Common RAG Code Issues to Know Cold

| Issue | What It Looks Like | Your Suggestion |
|-------|-------------------|-----------------|
| No chunking strategy | Entire docs in context | Semantic chunking + overlap parameter |
| Hardcoded top-k retrieval | `top_k=5` always | Dynamic k based on query complexity |
| No reranking | Raw vector search results fed to LLM | Add cross-encoder reranker pass |
| Context stuffing | All retrieved chunks concatenated blindly | Deduplication + relevance filtering |
| No citation/provenance | LLM can hallucinate sources | Attach source metadata, verify in output |
| Synchronous retrieval | Blocking call in main thread | Async retrieval with timeout |
| No eval | No way to know if RAG helps | Add retrieval recall metric, measure answer faithfulness |

---

## GENERAL ANSWERING FRAMEWORKS

### 1. The Problem-Solution-Tradeoff Framework

Use for any technical or architectural question:

```
Problem:   "The challenge was [SPECIFIC PROBLEM] which was causing 
            [MEASURABLE IMPACT]."

Solution:  "We addressed it by [SOLUTION], specifically by 
            [ONE KEY DECISION that distinguishes your approach]."

Tradeoff:  "The tradeoff was [COST/DOWNSIDE], which we accepted 
            because [BUSINESS OR TECHNICAL REASON]."
```

---

### 2. The Headlining Technique

> [!IMPORTANT]
> **Lead with the conclusion, then provide evidence.** This is the single biggest communication habit separating senior from mid-level engineers.

❌ **Don't:**
```
"So we started looking at the problem, and first we tried X, 
but that didn't work, then we tried Y, and eventually after 
some debugging we realized..."
```

✅ **Do:**
```
"The root cause was a race condition in the retrieval cache. 
Here's how we found it..."
```

Practice this: before answering any question, take 2 seconds to formulate the headline. Then speak.

---

### 3. Bridge Statements

Quick reference card — pick one per situation:

```
Buying time:       "Let me think about that for one moment..."
Don't know:        "I don't have that specific data point, but the 
                    range I'd expect is... and I can confirm after."
Redirecting:       "More relevant to your context here would be..."
Going deeper:      "There's an interesting nuance here — want me to 
                    go into that?"
Disagreeing:       "I've seen this handled differently — here's what 
                    worked in my context..."
Wrapping up:       "To summarize the key point: [ONE SENTENCE]."
```

---

### 4. The 4-Dial Framework

When designing any AI system, every decision lives on four dials. Make the tradeoffs explicit:

```
┌─────────────────────────────────────────────────────────────┐
│                    THE 4-DIAL FRAMEWORK                      │
│                                                              │
│   ACCURACY ◄────────────────────────────► COST              │
│   (precision, recall, F1)                 (compute, storage) │
│                                                              │
│   LATENCY  ◄────────────────────────────► RELIABILITY       │
│   (p50/p95/p99)                           (uptime, fallback) │
│                                                              │
│   Every architectural decision moves these dials.            │
│   State explicitly which dials you're optimizing for         │
│   and which you're accepting tradeoffs on.                   │
└─────────────────────────────────────────────────────────────┘
```

**How to use it in an answer:**
```
"For this system, we're optimizing primarily for accuracy and 
reliability — false negatives in fraud detection are expensive. 
We're accepting higher latency (5-min async pipeline vs. real-time) 
and higher compute cost to achieve that. This was an explicit 
decision made with the product and finance stakeholders."
```

---

## PREVIOUS FAILURE ANALYSIS — Learning from Cohere Health (Grade: B-)

> [!WARNING]
> This section is your pre-flight checklist. The issues that led to a B- at Cohere Health are patterns that WILL reappear under pressure unless you have specific countermeasures prepared.

---

### The Issues Identified

#### Issue 1: Verbal Filler
- **What happened:** "Um," "like," "you know," "sort of" appeared under pressure
- **Why it matters:** Filler words signal uncertainty — even when you know the answer
- **Countermeasure:** **The Pause Protocol**

```
PAUSE PROTOCOL:
Before answering any question:
  1. Inhale once (takes ~1 second)
  2. Form your headline in your head
  3. Then speak

A 2-second pause before speaking sounds confident.
"Um um um" for 2 seconds sounds uncertain.
Same duration — completely different signal.
```

#### Issue 2: Theoretical vs. Architectural Answers
- **What happened:** Answers described *what* the technology does rather than *why* you chose it
- **Example of theoretical:** "LangChain provides agent orchestration with tool calling..."
- **Example of architectural:** "We chose LangChain's agent loop over building our own because at the time, custom orchestration would have cost us 3 engineer-weeks we didn't have — but we've since abstracted behind our own interface so we could swap it out."
- **Countermeasure:** For every technology mentioned, immediately follow with *"...which we chose over [alternative] because..."*

#### Issue 3: Hesitation on Follow-Ups
- **What happened:** Follow-up questions caused loss of thread and searching for new content
- **Why it happens:** Follow-ups feel like traps — they're not. They're invitations to go deeper.
- **Countermeasure:** **The PREP Reframe**

```
PREP REFRAME:
When a follow-up lands, think: "They liked what I said and 
want more of it." Then use:

"Building on what I mentioned — [GO ONE LEVEL DEEPER]."

This maintains thread and shows you have depth.
```

---

### The 'Seniority in Communication' Concept

> [!NOTE]
> The difference between a senior and a lead isn't just what they know — it's how they communicate under pressure. These are the signals interviewers are calibrating on.

| Signal | Junior/Mid | Senior/Lead |
|--------|-----------|-------------|
| **Answer shape** | Builds up to the answer | Opens with the answer |
| **Uncertainty** | "I'm not sure, but maybe..." | "I don't have that exact figure — my estimate is X and here's why..." |
| **Scope handling** | Answers the question asked only | Answers + surfaces related considerations |
| **Tradeoffs** | Describes the solution chosen | Describes the solution AND the alternatives rejected |
| **Failure** | Avoids or minimizes | Describes, owns, and derives learning |
| **Silence** | Fills with filler | Uses deliberately as a thinking tool |

---

### Pre-Interview Communication Drill (Do this morning of)

Record yourself answering these three questions out loud. Listen back:

1. *"Tell me about yourself."* — Aim for exactly 90 seconds. No filler.
2. *"What was the hardest technical decision you made in your last project?"* — Headline first. Evidence second.
3. *"What would you do differently?"* — Three specific things. No hedging.

Grade yourself on: filler frequency, whether you headlined, whether you gave specific evidence.

---

## MINDSET & PREPARATION

### Day-Of Mindset: Peer Consultation, Not Being Tested

> [!TIP]
> The most important reframe: **You are not a candidate being evaluated. You are a senior practitioner being consulted.**

This is not a performance anxiety trick — it's structurally accurate. At Lead level, companies are not checking if you can do the job. They are determining if your approach to the job aligns with theirs. That's a conversation between peers.

**Behavioral cues of this mindset:**
- Ask clarifying questions early (peers do this; candidates don't dare)
- Push back on premises when you have a different view ("In my experience the opposite was true — want me to walk through why?")
- Show curiosity about their problems ("How do you currently handle X? I'm curious about the scale.")
- Speak about your work with ownership, not permission ("I decided..." not "We were told to...")

---

### Handling 'I Don't Know' Gracefully

The worst response to an unknown: *Silence + visible panic + guess*.
The best response: *Honest acknowledgment + structured thinking + offer to verify*.

**The Three-Part Unknown Response:**
```
Part 1 — Acknowledge:
  "I don't have that specific [number/detail/spec] in front of me."

Part 2 — Bound it:
  "What I can tell you is that the order of magnitude was [X], 
  because [REASONING based on what you do know]."

Part 3 — Close the loop:
  "I'd want to verify that exact figure — I can follow up by email 
  if it's important to your decision."
```

This response is *more impressive* than a confident but wrong answer.

---

### Asking Clarifying Questions as a Leadership Signal

Most candidates avoid clarifying questions because they fear looking like they don't understand. In reality, the opposite is true.

**Why clarifying questions signal leadership:**
- Leaders define scope before committing resources
- Leaders surface ambiguity before it becomes a bug
- Leaders ask "what does success look like?" before building

**Questions to ask in Part 3 (System Design) — pre-scripted:**
```
1. "Before I sketch the architecture — what's the volume we're 
   designing for? And is real-time processing a hard requirement 
   or is async acceptable?"

2. "Is the primary risk false positives or false negatives? 
   That will drive my threshold design significantly."

3. "Do we have any data residency or regulatory constraints 
   I should build around?"
```

Asking these three questions in the first 2 minutes of the system design sends a stronger signal than any architecture you could draw without asking them.

---

## PRE-INTERVIEW CHECKLIST

### Day Before

- [ ] Re-read this document top to bottom
- [ ] Write out your STARLA for your main agentic project (longhand)
- [ ] Practice the intro script 5 times out loud — time yourself
- [ ] Research Brit Insurance: current AI initiatives, Lloyds market context, recent news
- [ ] Prepare your three quantitative impact numbers (volume, efficiency, financial)
- [ ] Review your image anomaly detection architecture — be able to draw it in 3 minutes

### Morning Of

- [ ] Record and review the 3-question communication drill
- [ ] Reread the Seniority in Communication table
- [ ] Reread the bridge statements card
- [ ] Set your mindset: *peer consultation, not evaluation*

### 10 Minutes Before the Call

- [ ] Have a blank piece of paper and pen ready (system design sketching)
- [ ] Have a notepad open for their names and questions
- [ ] Silence all notifications
- [ ] Posture check: sit straight — it affects vocal quality

---

## QUICK REFERENCE CARDS

### At-a-Glance Answer Templates

```
Architecture question:
"We chose [X] over [Y] because [REASON]. Tradeoff was [COST]."

Leadership question:
"Specifically, I [ACTION] which led to [OUTCOME FOR PERSON/TEAM]."

Evaluation question:
"We measured at three levels: component, system, and business."

Production incident:
"It broke because [ROOT CAUSE]. We detected it via [MONITOR]. 
 Fixed by [MITIGATION]. Prevented by [SYSTEM CHANGE]."

Unknown answer:
"I don't have that exact number. Estimate: [X]. I'll verify."

System design opening:
"Before I design — three questions: volume? latency? 
 false positive vs. false negative tolerance?"
```

---

### The Final Reminder

> [!IMPORTANT]
> You are not being tested. You are showing what leading looks like from the inside. Every hesitation you let pass with a deliberate pause, every "we chose X because..." instead of "we used X", every clarifying question you ask instead of assuming — these are the signals that separate the hire from the close second.
>
> **Own the room. You've built these systems. Speak like you have.**

---

*Document last updated: Pre-interview preparation*
*Target role: Lead AI Engineer — Brit Insurance*
*Interview format: 1-hour Stage 1 (Virtual)*
