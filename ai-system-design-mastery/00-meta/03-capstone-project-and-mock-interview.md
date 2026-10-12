# The Capstone Project and the Mock Interview

> **Source note.** This file is grounded in the roadmap video (T1), which defines the course as five phases and describes phase 5 as a capstone project plus a one-to-one mock interview. The source's own words are quoted and translated in §1. Everything after that — the project-selection criteria, the deliverable structure, the rubric, the self-mock protocol — is **my elaboration**, because the source names the phase without specifying how to run it.

---

## 1. What the source actually says

The roadmap video closes with the final phase. Quoting the transcript (T1, [11:14]–[12:20]), with my translation:

> **"और फाइनली जब आपको ये आ जाएगा तब हम रेडी हो जाएंगे। एक फाइनल उसके लिए अभी तो सब करेंगे। एक हम डिजाइन एक क्रिएट करेंगे... जैसे रियल लाइफ से रिलेटेड कुछ हम इन्वेंशन करेंगे। कुछ लाइक एक ऐसा क्रिएट करेंगे प्रोजेक्ट जो कि अभी तक लाया ही नहीं करता ये सारी नॉलेज को यूज़ कर कर। तो ये उसके अंदर एलएलडी हो गया, एचएलडी हो गया — हर जितनी भी चीजें हमने अभी तक सीखी होगी, अप्लाई की होगी, सब हम चीजें इसके अंदर अप्लाई करेंगे।"**

> *"And finally, once you've got all this, we'll be ready. There's one final thing we'll do: we'll create a complete design. Something related to real life that we'll invent — a project that doesn't exist yet, using all this knowledge. So in it, LLD and HLD — everything we've learned so far, everything we've applied, we'll apply all of it inside this."*

> **"और फाइनली मैं लास्ट में अपने आपके सबके साथ एक मॉक सेशन रेडी करूंगा... अगर कोई भी इंटरेस्टेड होगा इसे इंप्रूव करना होगा, या फिर जिसे कॉन्फिडेंस नहीं आ रहा होगा, तो हम क्या करेंगे — वन टू वन सेशन करेंगे उसके साथ, मैं प्रॉपर मॉक इंटरव्यू लूंगा आपका और बताऊंगा कि आप अभी भी कहां पे लैक कर रहे हो।"**

> *"And finally, at the end I'll prepare a mock session with all of you... if anyone is interested in improving, or isn't feeling confident, we'll do a one-to-one session — I'll take a proper mock interview and tell you where you're still lacking."*

**What this tells you about the phase's purpose.** It is not revision. It is **synthesis under invented constraints**: take a system that does not exist, and design both levels of it — the components and their connections (HLD), and the classes and their methods (LLD) — using everything from phases 1–4. The invented-project constraint is deliberate: a system you have never seen cannot be recalled from a case study, so it tests whether the *method* transferred or only the examples.

**Also relevant from the same video**, because it describes what the mock is measuring:

> *"ये चेक करेंगे कि कितनी आपको नॉलेज है बेसिकली"* — the frontier labs "check how much knowledge you basically have."
>
> *"OpenAI, DeepMind, Anthropic evaluate how you reason, not just a diagram. पूरा रीज़निंग थिंकिंग आपकी चेक करेंगे"* — they evaluate **your reasoning, not just a diagram**.
>
> *"अगर उसकी सिस्टम डिज़ाइनिंग परफॉर्मेंस बेकार है तो वही मेन डिसाइडर फैक्टर होता है आपकी सैलरी का, आपके बैंड का"* — if system design performance is poor, **that is the main determinant of your salary band**.

And from the Day 1 Foundations video (T2), the interviewer's four criteria, which the mock should score against:

> Whether you can **think at scale** — "not just for 10 users but 10 million users"; whether you can **make trade-offs with proper reasoning**; whether you can **communicate clearly under pressure**; and whether you know **the right tools for the right job**.

**One practical caveat.** The source says he is building a platform to code and practise system design, and offers the one-to-one mock himself. Neither is available to you as a reader of this repository. So §4–§6 below are the substitute: a protocol you can run on your own or with a peer, designed to apply the same pressure the source's mock would.

---

## 2. Why a capstone, and why an invented project

**The transfer problem.** You can read all 100 files in this repository and still fail, because reading produces *recognition* and interviews test *application under pressure*. There is a specific failure mode that a capstone is designed to expose: **you can explain every component and still not be able to assemble them into a coherent system for a prompt you have not seen.**

| Level | Test | What produces it |
|---|---|---|
| **Recognise** | You can say what it is | Reading |
| **Explain** | You can say why it exists and what it costs | Reading + recall |
| **Apply** | You can say when to use it and when not to | Whiteboard drills |
| **Defend** ⚡ | You can name its failure modes and what changes if a requirement changes | **Capstone + mock** |

**Why invent rather than reuse.** If you design "the Twitter feed", you will reproduce the case study — and you will not notice that you are reproducing it. An invented system has no file to fall back on, so every component must be *derived* from a requirement or a number. That is the skill being tested, and it is the one the source's phrase *"a project that doesn't exist yet"* is engineered to isolate.

**Why HLD *and* LLD in one artefact.** The source's framing is that the capstone applies both. That is also how a real system is built: the components and their connections constrain the classes, and the class boundaries often force a reconsideration of the component boundary. Doing both on one system surfaces that interaction, which is exactly what a senior interview probes.

---

## 3. Running the capstone

### 3.1 Choosing the project

A good capstone project has five properties:

| Property | Why it matters |
|---|---|
| **Invents nothing you've already designed** | Otherwise you're recalling, not deriving |
| **Has a real-time or tight-latency path** | Forces queueing, caching, back-pressure decisions |
| **Has a data-intensive component** | Forces storage choice, indexing, sharding |
| **Has an AI component** | Forces the 2026 material: evals, cost per query, guardrails |
| **Has an operational dimension** | Forces SLOs, failure modes, cost — the part most candidates skip |

### 3.2 Ten candidate capstone projects

None of these duplicate the case studies in this repository, so all of them test derivation.

| # | Project | The hard part it exercises |
|---|---|---|
| 1 | **Real-time collaborative document editor with AI assist** | Conflict resolution (OT vs CRDT), presence, AI cost control, per-keystroke fan-out |
| 2 | **Multi-tenant AI agent platform** | Tenant isolation, per-tenant cost attribution, noisy-neighbour, tool sandboxing |
| 3 | **AI code-review bot for a large monorepo** | Repo-scale retrieval, incremental indexing, false-positive tolerance, CI integration |
| 4 | **Real-time voice translation** | Streaming ASR → MT → TTS pipeline, latency budget decomposition, barge-in |
| 5 | **Model evaluation as a service** | Reproducibility, eval-set versioning, statistical significance, cost per eval |
| 6 | **Medical imaging triage pipeline** | Regulatory constraints, audit trails, human-in-the-loop, precision/recall asymmetry |
| 7 | **Real-time multiplayer game backend** | Authoritative server, tick rate, matchmaking, lag compensation, regional placement |
| 8 | **Autonomous fleet telemetry and control** | Edge/cloud split, intermittent connectivity, safety interlocks, time-series volume |
| 9 | **Personal finance assistant with bank integrations** | Consent management, schema heterogeneity, idempotency, PII handling, explainability |
| 10 | **Adaptive AI tutor** | Learner modelling, curriculum sequencing, content generation vs curation, long-horizon evals |

**How to choose.** Pick the one where you can *feel* the requirement tension — where you already sense two goals pulling against each other. That tension is the raw material of a good design document, and the absence of it is why some projects feel flat.

### 3.3 The deliverable — a document outline

Write it as a design document, not an essay. Target 15–25 pages.

| § | Section | Contents |
|---|---|---|
| 1 | **Problem statement** | What system, for whom, solving what |
| 2 | **Functional requirements** | The 5–8 things it must do. Explicitly say what it will *not* do |
| 3 | **Non-functional requirements** | Latency, throughput, availability, consistency, cost, compliance — each with a number |
| 4 | **Capacity estimation** | DAU → QPS → storage → bandwidth → cost. **Show the arithmetic** |
| 5 | **API design** | The 3–6 endpoints that matter, with request/response shapes |
| 6 | **Data model** | Entities, relationships, and the *storage choice per entity* with a reason |
| 7 | **HLD** | The component diagram with labelled arrows. Every box traces to a requirement or a number |
| 8 | **Deep dives** | 2–3 components at depth, including at least one AI component |
| 9 | **LLD** | Class diagram for the core domain; key interfaces; the concurrency story |
| 10 | **Trade-offs** | 5+ in the four-part form (below), including the ones you did *not* choose |
| 11 | **Failure modes** | What breaks, what the user sees, what the system does |
| 12 | **Operationalisation** | SLIs/SLOs, alerts, deploy strategy, DR, cost model |
| 13 | **What I'd change** | If requirement X changed, what changes — the highest-signal section |

**The four-part trade-off form** — use it verbatim, it is the single most portable sentence pattern in this field:

> "I'll use **[X]** because **[number]**. It gives me **[benefit]**, at the cost of **[sacrifice]**."

### 3.4 A compact schedule

If you have two weeks:

| Days | Work |
|---|---|
| 1 | Choose the project. Write §1–§3. Do **not** start drawing |
| 2 | Capacity estimation (§4). This takes longer than you expect; that is normal |
| 3 | API and data model (§5–§6). Justify every storage choice |
| 4–5 | HLD (§7). Draw it three times; throw away the first two |
| 6–7 | Deep dives (§8), including the AI component |
| 8 | LLD (§9). Diagram the core domain, then write real interfaces |
| 9 | Trade-offs (§10) and failure modes (§11) |
| 10 | Operationalisation (§12). SLIs, SLOs, cost |
| 11 | §13, then a full read-through for internal consistency |
| 12 | **Self-mock on your own project** (§5). Record it |
| 13 | Fix what the recording exposed |
| 14 | Mock on an *unseen* project (§6). This is the real test |

**Day 14 is the one that matters.** A capstone proves you can design *your* system. A mock on an unseen prompt proves the method transferred — which is what the interview measures.

---

## 4. What interviewers actually score

Five dimensions, drawn from the source's criteria and from what the interview rounds in this repository reward.

### 4.1 The rubric

Score yourself 1–5 on each. Be harsh; a generous self-score is a useless self-score.

| # | Dimension | 1 — Failing | 3 — Adequate | 5 — Strong |
|---|---|---|---|---|
| 1 | **Requirement clarification** | Started drawing immediately | Asked functional questions, skipped non-functional | Elicited functional *and* non-functional, and stated what was out of scope |
| 2 | **Quantitative reasoning** | No numbers anywhere | Some numbers, loosely applied | Every architectural choice traces to a derived number, arithmetic shown |
| 3 | **Trade-off reasoning** | Declared components as good practice | Named trade-offs when asked | Volunteered trade-offs unprompted, in the four-part form, including rejected options |
| 4 | **Communication under pressure** | Rambled; lost the thread when interrupted | Clear but needed prompting to structure | Signposted structure, stayed on the thread through interruptions, managed time |
| 5 | **Depth and failure awareness** | Described the happy path only | Named some failure modes when asked | Named failure modes of *their own* design before being asked; knew when *not* to use a tool |
| 6 | **Right tool for the right job** | Reached for a favourite technology | Chose reasonably | Chose and *justified* against alternatives, and said when a simpler tool would do |

### 4.2 Reading your score

| Pattern | Diagnosis | Fix |
|---|---|---|
| High 1, low 2 | You talk well but do not quantify | Do 20 capacity estimations cold, alone, on paper |
| High 2, low 3 | You compute but do not reason | Practise stating every choice in the four-part form until automatic |
| High 3, low 4 | You know it but cannot deliver it | Mock under time pressure with interruptions |
| High 4, low 5 | You present well but are shallow | Go one level deeper on two components; study failure modes |
| Uniformly 3 | The classic plateau | Force every answer to include a number and a failure mode |

**The most common profile is (4, 2, 2, 4, 2)** — articulate, unquantified, unaware of failure modes. It reads as "knows the vocabulary, has not built anything."

---

## 5. Running a self-mock

### 5.1 The protocol (60 minutes, alone)

```
 0:00  Pick a prompt you have NOT designed        (use §3.2 list, or invert a case study)
 0:00  Start a timer. Start a screen recording.
 0:05  Requirements written down, functional + non-functional. STOP DRAWING until now
 0:10  Capacity arithmetic on paper. Show every step
 0:20  API + data model
 0:30  HLD diagram. Label every arrow
 0:45  Deep dive: pick the component you're least sure about
 0:55  Three trade-offs aloud, in the four-part form
 0:58  The close: "if requirement X changed, I'd change Y"
 1:00  Stop. Do not finish the diagram. The timer is the point
```

### 5.2 The review (45 minutes — this is where the learning is)

Watch the recording with the rubric in §4.1 and score each dimension. Then answer, in writing:

1. Did I draw before I clarified? (If yes, this is the single highest-value fix.)
2. How many distinct numbers did I say aloud? (Target: 15+. Below 5 is a failing performance.)
3. Did I state a trade-off before I was asked?
4. Did I name a failure mode of my own design before being asked?
5. Where did I go silent for more than ten seconds?
6. Did I finish with "what I'd change if the requirement changed"?

**Why recording is non-negotiable.** You cannot hear your own filler, your own hedging, or the moment you lost the thread while you are in it. The recording makes the gap between what you *know* and what you *delivered* visible, and that gap is precisely what the interview measures.

### 5.3 The pushback drill

The self-mock tests delivery. This tests defence. For any design, answer these five aloud, in under 60 seconds each:

| Pushback | What a strong answer does |
|---|---|
| "Why not just use Postgres for everything?" | Names the specific access pattern that breaks, and the number that breaks it |
| "What happens when this component fails?" | Degradation behaviour, what the user sees, what the alert is |
| "This is over-engineered. Simplify it." | Concedes the specific part that is, and says what requirement justified the rest |
| "Your estimate is wrong — traffic is 10×." | Says which component breaks *first*, and what changes |
| "How do you know it works?" | Names a metric, a target, and how it would detect the failure |

**The last one is the most commonly fumbled.** "How do you know it works?" is the question that separates candidates who have operated something from those who have only read about it.

---

## 6. Running a mock with a partner

Better than a self-mock, because a partner will not let you skip the hard part.

### 6.1 The interviewer's script

Give your partner this. The instruction that matters is the last one.

> **Format:** 60 minutes. You are a senior engineer at [company]. Be friendly but do not fill silences — let them struggle.
>
> **The prompt:** [pick from §3.2]
>
> **Your job:**
> 1. Give the prompt. Say nothing else for the first 5 minutes. **Let them ask.**
> 2. If they start drawing before clarifying requirements, ask: *"Before you draw — what are the requirements?"*
> 3. Answer their questions, but under-specify. If they ask "how many users?", say *"what would you assume?"* Make them commit to a number.
> 4. Interrupt once at ~20 minutes with a scope change: *"Actually, add [feature]."* Watch how they adapt.
> 5. At ~40 minutes, push back on their weakest choice: *"Why that, and not [alternative]?"*
> 6. At 55 minutes, ask: *"What would you change if [requirement] were 10×?"*
> 7. **Score them on the rubric in §4.1 and tell them the two lowest dimensions.** Be specific and be honest — the point of the exercise is lost if you are kind.

### 6.2 Using an LLM as the interviewer

Practical, always available, and genuinely useful — with one important caveat.

**Why it helps:** unlimited reps, no scheduling, and it will ask follow-ups you would not think to ask yourself.

**The caveat:** LLMs are sycophantic by default. Left unconstrained, an LLM interviewer will praise a mediocre design, which is worse than no practice — it calibrates you to a standard that does not exist. **You must explicitly instruct it to be adversarial**, and you must give it the rubric, or it will grade on vibes.

A prompt that works:

```
You are a senior staff engineer conducting a system design interview at a
frontier AI lab. Be rigorous and adversarial. Do not praise unless the answer
genuinely earns it. Do not fill silences with hints.

Rules:
- Give me ONE prompt and then stay quiet until I ask clarifying questions.
- If I start designing before clarifying requirements, stop me.
- Under-specify deliberately. When I ask for a number, say "what would you
  assume?" and make me commit.
- After 20 minutes of conversation, introduce a scope change.
- Push back at least twice on my weakest decision. Do not accept "it depends."
- At the end, score me 1-5 on: requirement clarification, quantitative
  reasoning, trade-off reasoning, communication, depth/failure awareness,
  right-tool-for-the-job. For each, quote the specific thing I said that
  justified the score. Then tell me the two dimensions I most need to fix.

Start by asking me which company and role I'm targeting.
```

**Score its feedback sceptically.** The scores are unreliable; the *quoted evidence* is not. Read the quotes it pulls, and judge for yourself whether they support the score.

---

## 7. Failure patterns to hunt for

The most common ways a prepared candidate fails. Find these in your recording.

| # | Pattern | What it looks like | Fix |
|---|---|---|---|
| 7.1 | **Drawing before clarifying** | Diagram at minute two | Hard rule: no drawing before requirements are written |
| 7.2 | **Unquantified design** | No numbers in the whole session | Force ≥15 numbers aloud; do capacity estimation cold |
| 7.3 | **Component recitation** | Boxes named because they are "best practice" | Every box must trace to a requirement or a number |
| 7.4 | **Silent deep-dive** | Long pause, then a shallow answer | Say the structure of your thinking aloud while you think |
| 7.5 | **Happy-path only** | No failure modes until asked | Volunteer two before being asked |
| 7.6 | **Tool-first thinking** | "We'll use Kafka" before knowing the throughput | Choose the tool *after* the number |
| 7.7 | **Defensive under pushback** | Repeats the original choice more loudly | Concede the specific part that is weak, defend the rest |
| 7.8 | **Time collapse** | Detailed on §1–4, rushed on trade-offs | Protect the last five minutes; trade-offs are the highest-signal content |
| 7.9 | **No close** | Ends mid-diagram | Always close with "what I'd change if..." |
| 7.10 | **Scope drift** | Design grows to cover everything | State what is out of scope, and defend the boundary |
| 7.11 | **AI hand-waving** | "Then an LLM handles it" with no evals, cost, or guardrails | AI components need the same rigour: latency, cost per query, eval, failure behaviour |
| 7.12 | **Ignoring the operational half** | Nothing about SLOs, alerts, cost, DR | Reserve a section; "how would you operate this?" is increasingly asked |

**7.11 deserves emphasis in 2026.** The most common AI-era failure is treating the model as a magic box that resolves the hard part. It does not — it *introduces* hard parts: latency variance, cost per query, non-determinism, evaluation, and a new class of failure where the system is "up" but wrong.

---

## 8. Q&A

**Q1. Why a capstone project rather than more reading?**
Because reading produces recognition and interviews test application. There is a specific gap a capstone exposes: you can explain every component and still fail to assemble them for a prompt you have not seen. The capstone is designed to find that gap while it is still cheap to fix, rather than in the interview.

**Q2. Why does the source insist the project be invented rather than a known system?**
Because a known system has a file to fall back on. If you design "the Twitter feed", you will reproduce the case study and not notice you are reproducing it. An invented system forces every component to be derived from a requirement or a number, which is precisely the skill being tested. The source's phrase is *"a project that doesn't exist yet"*, and the constraint is deliberate.

**Q3. How do I choose a capstone project?**
Pick one where you can already feel two requirements pulling against each other. That tension is the raw material of a real design document. If a project feels flat and uncontested, the design will be flat too — because there will be nothing to trade off, and trade-offs are what the document is actually for. Also check it exercises all five things in §3.1 — latency, data, AI, operations, and derivation.

**Q4. Should I do the capstone before or after reading the case studies?**
After. The case studies are the worked examples that give you the vocabulary and the sense of what a defended design looks like. The capstone is where you stop reading and derive. Doing the capstone first is possible but produces a weaker document, because you will not yet know what a strong trade-off statement sounds like.

**Q5. How long should the capstone take?**
About two weeks at a reasonable pace, per the schedule in §3.4. The capacity estimation (§4) reliably takes longer than expected — that is normal and it is not a sign you are slow. The HLD section should take two attempts, because the first diagram is always the one you thought you wanted rather than the one the numbers support.

**Q6. What is the most important section of the document?**
§13, "what I'd change if requirement X changed." It is the highest-signal section in the entire artefact, because it demonstrates that your design is a *response to specific constraints* rather than a memorised architecture. If you can say exactly what changes when a requirement moves, you have understood your own design. If you cannot, you have described one.

**Q7. How do I know if my capstone is good enough?**
Run the mock on your own project, then run a mock on a project you have never designed. If the second one goes nearly as well as the first, the method transferred. If it collapses, your capstone succeeded as an artefact and failed as training — which is a real and common outcome, and the fix is more unseen-prompt practice, not more document polish.

**Q8. Is a self-mock actually useful, or do I need a partner?**
Both, in order. The self-mock is available always and the recording is the part that matters — you cannot hear your own filler or notice the moment you lost the thread while you are in it. A partner is strictly better because they will not let you skip the hard part, and the interruption test in §6.1 is something you cannot honestly do to yourself.

**Q9. Can I use an LLM as an interviewer?**
Yes, and it is genuinely useful for reps. But the caveat is important: LLMs are sycophantic by default and will praise a mediocre design, which calibrates you to a standard that does not exist. You must explicitly instruct it to be adversarial and give it the rubric, or it grades on vibes. And treat its *scores* sceptically — the quoted evidence it produces is the part worth reading.

**Q10. What if I don't have time for a full capstone?**
Compress to the highest-value sections: requirements, capacity estimation, HLD, and trade-offs. Those four are what the interview actually assesses. LLD and operationalisation can be a page each. What you must not skip is the arithmetic and the trade-offs, because those are the two dimensions with the widest spread between candidates.

**Q11. How do I practise defence specifically?**
The pushback drill in §5.3 — five questions, 60 seconds each, aloud. The one people fumble most is "how do you know it works?", because it demands a metric and a target rather than a description. If you can answer all five crisply, you are ready for the follow-up half of the interview, which is where most candidates lose the round after a decent diagram.

**Q12. What is the biggest change in capstone design since AI systems became standard?**
The AI component has to be designed with the same rigour as everything else, and it introduces new constraints rather than removing old ones: latency variance, cost per query, non-determinism, evaluation, and a failure class where the system is "up" but producing wrong output. The most common 2026 failure pattern is treating the model as a magic box that resolves the hard part. It does not; it relocates it.

**Q13. How many mock interviews should I do before the real one?**
Enough that the *structure* is automatic and you can spend your attention on the content. For most people that is 3–5 full mocks, with at least two on prompts they have never seen. After that, additional mocks with diminishing returns are better replaced by targeted work on the two lowest rubric dimensions.

**Q14. How do I know which dimension to fix?**
Take the two lowest from your rubric. The mapping in §4.2 is diagnostic: high communication with low quantification means you talk well but do not compute, and the fix is twenty cold capacity estimations rather than more mocks. Fixing the lowest dimension is almost always higher-yield than another full run.

**Q15. ⚡ Pushback: "Isn't 15–25 pages excessive for interview prep?"**
The document is not the deliverable — the *derivation* is. You write 20 pages because that is the amount of thinking required to make every component trace to a number, and you cannot shortcut the thinking by shortening the document. In the interview you deliver none of it; you deliver the 45-minute version. But that 45-minute version is only available to someone who has done the 20-page version once, because that is what makes the derivation automatic.

**Q16. ⚡ Pushback: "Why not just memorise 20 case studies instead?"**
Because the interview prompt is usually one you have not seen, and memorised designs collapse the moment a requirement moves. Memorisation gives you a library of answers; the capstone gives you a method. The strongest evidence is the mock-on-an-unseen-prompt test in §7 — if you have only memorised, that test fails immediately, and it is the test the interview actually administers.

---

## 9. Cheat sheet

| | |
|---|---|
| **Phase 5 is** | A capstone (invented project, HLD **and** LLD applied) + a one-to-one mock interview |
| **Why invented** | A known system lets you recall instead of derive. The constraint is the point |
| **Deliverable** | 15–25 page design doc, 13 sections, §13 ("what I'd change") being the highest-signal |
| **Four-part trade-off** | "I'll use **[X]** because **[number]**. Benefit **[B]**, at the cost of **[C]**" |
| **Rubric (1–5 each)** | Requirement clarification · quantitative reasoning · trade-off reasoning · communication under pressure · depth/failure awareness · right tool for the right job |
| **Most common profile** | (4, 2, 2, 4, 2) — articulate, unquantified, unaware of failure modes |
| **The interviewer's four criteria** (source) | Think **at scale** · trade-offs **with proper reasoning** · communicate **clearly under pressure** · right **tool for the job** |
| **What frontier labs test** (source) | **"How you reason — not just a diagram"** |
| **Hard rule** | **Never draw before requirements are written down** |
| **Self-mock** | 60 min, timed, **recorded**. Review with the rubric — the recording is the point |
| **Pushback drill** | Postgres-for-everything · component failure · simplify it · traffic ×10 · **how do you know it works?** |
| **Partner mock** | Under-specify · make them commit to numbers · interrupt with a scope change at 20 min · push back at 40 · "what if ×10?" at 55 · **score honestly** |
| **LLM interviewer** | Useful for reps — but **must be told to be adversarial**, and give it the rubric. Trust the quotes, not the scores |
| **Biggest 2026 failure** | Treating the model as a magic box. AI adds constraints, it does not remove them |
| **The real test** | A mock on a prompt you have **never** designed. If that goes as well as your capstone, the method transferred |
| **Budget** | ~2 weeks, or 4 sections if compressed: requirements · estimation · HLD · trade-offs |

---

## 10. Cross-links

- [How to use this knowledge base (and the file template)](01-how-to-use-and-template.md) — the four-pass method and the whiteboard drill this file extends
- [Source & topic map](02-source-topic-map.md) — which transcript produced which file, and the five-phase mapping
- [The five-step interview framework](../HLD/01-foundations/04-five-step-interview-framework.md) — the method the capstone applies end to end
- [Six goals and trade-offs](../HLD/01-foundations/02-six-goals-and-tradeoffs.md) — the tensions your capstone must resolve
- [Capacity estimation playbook](../HLD/01-foundations/05-capacity-estimation-playbook.md) — the arithmetic the rubric's second dimension scores
- [Availability numbers and SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md) — the numbers behind your non-functional requirements
- [LLD: the machine-coding interview framework](../LLD/06-machine-coding-interview-framework.md) — the LLD half of the capstone
- [Behavioural and project questions](../Interview-Questions/07-behavioral-and-project-questions.md) — how the capstone becomes interview stories
- [Agentic coding assistant](../Case-Studies/HLD/11-agentic-coding-assistant.md) — the richest worked example of an AI system designed end to end
- [Netflix — streaming and recommendation](../Case-Studies/HLD/12-netflix-recommendation-and-streaming.md) — a model design document to measure your own against
- [SLOs, error budgets & incident response](../Operations/03-slos-error-budgets-and-incident-response.md) — §12 of the capstone document
- [Cost & FinOps](../Operations/06-cost-and-finops.md) — the cost model the rubric rewards
- [Operationalisation cheat sheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md) — the compressed ops revision for the operational dimension
