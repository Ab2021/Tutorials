# How to Use This Knowledge Base (and the File Template)

> **Read this first.** It explains what this repository is, how it's organised, the mandatory structure every file follows, and how to use it in the two weeks before an interview.

---

## 1. What this is

A complete, self-contained study system for **AI system design interviews** — built entirely from a collated set of video transcripts and reference repositories on AI system design, classical system design, LLM/RAG engineering and LLMOps.

It is designed to be **read in order once** and **used as a reference forever after**.

### What it covers

**The top level is the interview round**, not the topic. The source roadmap defines the course in five phases, and three of them are literally High-Level Design, Low-Level Design and a capstone — so the repository is organised the same way.

| Layer | Where | Content |
|---|---|---|
| **HLD** — the high-level design round | `HLD/` | Foundations (six goals, the five-step framework, capacity estimation, availability arithmetic) · networking (DNS, CDN, proxies, API gateway, rate limiting) · scaling (load balancers, caching, Redis, invalidation, sharding, replication) · data (SQL, NoSQL, vector DB, blob storage, selection) · AI systems (RAG, agents, LLM serving, guardrails, evals, model-weight distribution) |
| **LLD** — the low-level design round | `LLD/` | OOP and SOLID · design patterns · concurrency and threading · UML and class diagrams · the machine-coding framework · **16 class-level problems** |
| **Case studies** | `Case-Studies/` | **14 HLD** end-to-end designs (URL shortener, WhatsApp, notifications, Instagram, YouTube, Uber, BookMyShow, Twitter/X, RAG, support agent, coding assistant, Netflix, fraud detection, autocomplete) · **5 LLD** designs taking those same systems down to classes |
| **Operations** | `Operations/` | Readiness review, observability, SLOs and incident response, deployment and CI/CD, capacity planning, FinOps, reliability patterns, security and compliance, DR, LLMOps |
| **Interview questions** | `Interview-Questions/` | Eight question banks with model answers |
| **Cheat sheets** | `Cheat-Sheets/` | Seven dense revision sheets |

---

## 2. The one-paragraph summary of the whole field

System design is the discipline of **deciding how to build a large software system that works for millions of users without breaking or slowing down.** You are the architect, not the builder. Four problems make it hard: servers crash under load, databases slow under millions of queries, data is lost when a machine fails, and two users update the same record at once. Six goals pull against each other — scalability, availability, reliability, performance, consistency and cost — so every design is a *selection* justified by numbers. The interview has five steps: clarify requirements, estimate capacity, define the API, draw the high-level design, then deep-dive and state trade-offs. In 2026, the same discipline now extends to LLM systems, where the primitives are retrieval, evaluation, guardrails, inference economics and agent orchestration — and where *cost per query* and *output quality* are first-class design constraints alongside latency and availability.

---

## 3. How the repository is organised

```
00-meta/                 ← you are here (how to use it · source map · capstone + mock)
HLD/                     ← the high-level design round
  01-foundations/        ← the framework and the arithmetic  ← START HERE
  02-networking/         ← DNS, CDN, proxy, API gateway
  03-scaling/            ← load balancing, caching, Redis, sharding
  04-data/               ← SQL, NoSQL, vector DB, blob storage, selection
  05-ai-systems/         ← RAG, agents, LLM serving, guardrails, evals
LLD/                     ← the low-level design round
  01–06                  ← OOP/SOLID, patterns, concurrency, UML, framework
  problems/              ← 16 class-level problems
Case-Studies/            ← worked designs at both levels
  HLD/                   ← 14 end-to-end designs
  LLD/                   ← 5 systems taken down to classes
Operations/              ← running it in production
Interview-Questions/     ← 8 banks of model answers
Cheat-Sheets/            ← 7 dense revision sheets
```

### Reading paths

| If you have… | Read |
|---|---|
| **2 days** | HLD/01-foundations (all) → Cheat-Sheets (all) → Case-Studies/HLD/01-url-shortener |
| **1 week** | HLD/01-foundations → 02-networking → 03-scaling → Cheat-Sheets → Interview-Questions |
| **2 weeks** | Everything, in order, with the case studies read last each day |
| **An AI-specific interview tomorrow** | HLD/01-foundations/04 + 05 → HLD/05-ai-systems (all) → Cheat-Sheets/04 + 05 |
| **An LLD / machine-coding round** | LLD/01 → LLD/06 → LLD/problems/01–04 (the source-named four) → Cheat-Sheets/07 |
| **An operational role** | Operations (all) → Cheat-Sheets/06 → Interview-Questions/06 |
| **A classical infra interview** | HLD/02-networking → 03-scaling → 04-data → Case-Studies/HLD |
| **Not ready yet** | `00-meta/03-capstone-project-and-mock-interview.md` — invent a system, design it at both levels, mock it |
| **20 minutes before the call** | Cheat-Sheets/01 |

---

## 4. The file template (13 sections)

Every concept file in this repository follows the same shape, so you always know where to find a given kind of information.

| # | Section | What it contains |
|---|---|---|
| 1 | **Title + TL;DR + 60-second interview answer** | The whole file compressed into one paragraph you could say out loud |
| 2 | **Definition & mental model** | Precise definition, a memorable analogy, what it is *not* |
| 3 | **Why it exists** | The problem it solves; what breaks without it |
| 4 | **Internal mechanics** | How it actually works, with a diagram |
| 5 | **Variants & taxonomy** | The variants table — every flavour and when each applies |
| 6 | **Pros / Cons** | Honest two-column tables, per variant where relevant |
| 7 | **Exceptions, edge cases & failure modes** | The long table. Where the simple model breaks |
| 8 | **When to use / When NOT to use** | Decision rules, not just descriptions |
| 9 | **Numbers, defaults & tuning** | Concrete figures you can quote |
| 10 | **Interview Q&A** | ≥ 12 questions with spoken-style model answers |
| 11 | **Common mistakes** | What candidates get wrong |
| 12 | **Cheat-sheet summary** | The dense, printable version |
| 13 | **Cross-links** | Related files, relative paths |

### Operationalisation files use a playbook shape

| Section | Contents |
|---|---|
| Purpose | What this playbook is for |
| Scope | In scope / out of scope |
| SLIs & SLOs | What you measure and target |
| Architecture | Diagram of the system being run |
| SOPs & runbooks | The step-by-step procedures |
| Failure modes | What breaks and what to do |
| Capacity & cost | Sizing and economics |
| Metrics & alerts | Signals with actual thresholds |
| Change management | How changes are made safely |
| Audit & compliance | Evidence and obligations |
| Q&A | Interview-grade questions |
| Cheat sheet | The dense version |
| Cross-links | Related files |

### LLD problem files use a code-first shape

The 16 problems in [`LLD/problems/`](../LLD/problems/) follow their own shape, because an LLD answer is a different artefact from an HLD one — it is a class design, not an architecture.

| # | Section | What it contains |
|---|---|---|
| 1 | **Title + TL;DR + 60-second answer** | The design compressed into one paragraph you could say out loud |
| 2 | **Requirement clarification** | The questions you ask *before* designing — scope, actors, core use cases, and what is out of scope |
| 3 | **Core entities** | The nouns: the classes that will exist, and why each earns its place |
| 4 | **Class diagram** | The boxes-and-lines view — inheritance, composition, cardinality |
| 5 | **Interfaces & abstract classes** | The extension seams: which types are abstract, which are interfaces, and what that buys |
| 6 | **Code** | A real, readable implementation of the core — not pseudocode where real code is feasible |
| 7 | **Concurrency & thread-safety** | The race conditions, the locks, the atomicity boundary — where LLD actually gets hard |
| 8 | **Extensibility — "add feature X"** | The follow-up every interviewer asks. Answers it by pointing at the seam that already exists |
| 9 | **Failure modes** | What breaks: deadlock, starvation, memory growth, null handling, invalid state transitions |
| 10 | **Trade-offs** | In the same four-part form: choice, reason, benefit, sacrifice |
| 11 | **Q&A** | ≥ 12 questions, including ≥ 3 pushbacks — either inline, marked ⚡, or gathered into a closing *"Pushback round"* section. Both forms are in use; the round form reads better when the objections are long. |
| 12 | **Cheat sheet + cross-links** | The dense version, and where the HLD view of the same system lives |

**The HLD↔LLD rule.** Where a problem exists at both levels (rate limiter, LRU cache, pub-sub, BookMyShow), the LLD file owns the *class and code* treatment and cross-links to the HLD file for the distributed view. **Never duplicate** — a reader should be able to tell from the cross-link which file answers which question.

---

## 5. How to actually study with this

### The four-pass method

| Pass | Method | Output |
|---|---|---|
| **1. Read** | Read the file top to bottom, no notes | Familiarity |
| **2. Recall** | Cover the answers. Read each Q&A question and answer aloud | Gaps identified |
| **3. Reconstruct** | From the TL;DR alone, redraw the diagram on a blank page | Active recall |
| **4. Defend** | Have someone challenge the "cons" and "exceptions" sections | Trade-off fluency |

**Pass 3 is the one that matters most.** If you can't redraw it from memory, you don't know it — and interviews are whiteboard-shaped.

### The whiteboard drill

For every case study, spend 20 minutes doing this:

```
0–6 min    Write out functional + non-functional requirements from memory
6–11 min   Do the capacity arithmetic on paper
11–16 min  Draw the architecture with labelled arrows
16–20 min  Write three trade-off statements in the four-part form
```

Then compare against the case study file and mark what you missed. Repeat until the misses stop.

### What "knowing" a topic looks like

| Level | Test |
|---|---|
| **Recognise** | You can say what it is |
| **Explain** | You can say why it exists and what it costs |
| **Apply** | You can say when to use it and when not to |
| **Defend** ⚡ | You can name its failure modes and what changes if a requirement changes |

Interviews test levels 3 and 4. Reading gets you to 2. Only recall and defence get you to 4.

---

## 6. Ten rules that apply to every question

1. **Never start drawing.** Clarify requirements first — functional *and* non-functional.
2. **Estimate before you design.** Numbers justify architecture; without them, every component invites "why?"
3. **Derive components; don't declare them.** Each box traces to a requirement or a number.
4. **Label every arrow** with its protocol or purpose.
5. **State trade-offs in four parts:** the choice, the numeric reason, the benefit, the cost.
6. **Name the failure modes of your own design** before you're asked.
7. **Distinguish availability from reliability** — "is it up?" versus "is it working?"
8. **Remember the multipliers:** replication ×3, index overhead ×1.5, fan-out, peak ×3.
9. **Protect the last five minutes** for trade-offs, even if the diagram is unfinished.
10. **Always close with what you'd change if the requirement changed** — the highest-signal sentence in the interview.

---

## 7. The 2026 context — why AI system design matters now

| Fact | Implication |
|---|---|
| 60–70% of modern AI interviews include GenAI design questions | Classical system design alone is no longer enough |
| System design performance is the main determinant of the salary band | This is the highest-leverage preparation you can do |
| Frontier labs, FAANG and AI startups all ask it | The framework is portable across all three |
| LLM systems add two new first-class constraints | **Cost per query** and **output quality** — alongside latency and availability |
| "Is it up?" no longer implies "is it working?" | Quality must be an SLI, not an afterthought |

**The unifying insight of this whole repository:** the classical primitives — cache, queue, replica, shard, circuit breaker — still apply to AI systems. What's new is that the *value* being moved is generated text, the *cost* varies per request, and the *correctness* is probabilistic. Every AI file here is the classical pattern plus those three differences.

---

## 8. Q&A

**Q1. How is this different from a textbook?**
It's built from interview-oriented video transcripts, so it mirrors what interviewers actually ask and the order they ask it in — and every file is structured for recall (TL;DR, Q&A, cheat sheet) rather than for a linear read.

**Q2. Should I read it in order?**
In order once, yes — foundations before infrastructure before AI, because the AI files assume the classical vocabulary. After that, use the reading paths in §3.

**Q3. What if I only have one day?**
Foundations (04 and 05 for the framework and estimation), cheat sheet 01, and the URL shortener case study. That covers the structure of every answer you'll give.

**Q4. What's the single highest-value file?**
`01-foundations/04-five-step-interview-framework.md`. It's the operating procedure for every other file — the rest is knowledge, that one is method.

**Q5. How do the case studies relate to the concept files?**
Concept files teach a component; case studies assemble several into a defended design. Read the concepts first, then watch them used under time pressure.

**Q6. Are the operationalisation files relevant if I'm not applying for an SRE role?**
Yes — modern AI interviews increasingly ask "how would you operate this?" and "what would you monitor?", because production AI failures are quality failures, not uptime failures. They're also the source of the strongest behavioural-interview stories.

**Q7. How should I use the cheat sheets?**
As a *revision* tool, not a learning tool. They assume you've read the full file; their job is to make everything retrievable in 20 minutes before an interview.

**Q8. What should I do if a topic feels shallow after reading?**
Do the whiteboard drill in §5 and then read the interview Q&A again. Depth in interviews comes from recall under pressure, not from more reading.

**Q9. How do I know when I'm ready?**
When you can take a prompt you've never seen, run the five steps without notes, and defend three trade-offs against pushback. Test it on a case study you *haven't* read yet.

**Q10. Where are the sources?**
Every file cites its source at the top. See [Source & topic map](02-source-topic-map.md) for the full transcript-to-file mapping.

---

## 9. Cheat sheet

| | |
|---|---|
| **Structure** | Top level = **the interview round** (HLD · LLD · Case-Studies) · 13-section template for concepts · code-first shape for LLD · playbook shape for ops |
| **The two rounds** | **HLD** = which components exist and how they connect. **LLD** = the classes, methods and concurrency inside one component |
| **The framework** | Clarify → Estimate → API → HLD → Deep dive + trade-offs |
| **The LLD framework** | Clarify → Entities → Diagram → Interfaces → Code → Extend → Test |
| **The rule** | Never start drawing. Ask first |
| **The arithmetic** | DAU × req/user/day ÷ 86,400 × 3 peak |
| **The anchors** | 10⁵ s/day · 1M req/day ≈ 12 QPS · 1 KB × 1M = 1 GB · DB ≈ 5,000 QPS |
| **The nines** | 99% = 3.65 d · 99.9% = 8.77 h · 99.99% = 52.6 min |
| **The trade-off form** | "I'll use [X] because [number]; it gives me [benefit], at the cost of [sacrifice]" |
| **The close** | "If the requirement changed to [Y], I'd change [Z]" |
| **Study method** | Read → Recall → Reconstruct → Defend (pass 3 is the one that matters) |
| **Ready test** | Five steps, unseen prompt, no notes, three defended trade-offs |

---

## 10. Cross-links

- [Source & topic map](02-source-topic-map.md) — which transcript produced which file, and the five-phase mapping
- [The capstone project and the mock interview](03-capstone-project-and-mock-interview.md) — how to test all of this under pressure
- [README](../README.md) — the master index
- HLD: start at [What is system design](../HLD/01-foundations/01-what-is-system-design-and-why.md) · [Five-step framework](../HLD/01-foundations/04-five-step-interview-framework.md)
- LLD: start at [What is LLD and how to answer it](../LLD/01-what-is-lld-and-how-to-answer.md) · [Machine-coding framework](../LLD/06-machine-coding-interview-framework.md)
- Revise here: [Foundations cheat sheet](../Cheat-Sheets/01-foundations-and-estimation-cheatsheet.md) · [LLD cheat sheet](../Cheat-Sheets/07-lld-cheatsheet.md)
- Practise here: [Foundations questions](../Interview-Questions/01-foundations-and-capacity-questions.md) · [LLD questions](../Interview-Questions/08-lld-questions.md) · [URL shortener](../Case-Studies/HLD/01-url-shortener.md)
