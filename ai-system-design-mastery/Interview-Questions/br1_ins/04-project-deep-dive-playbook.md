# 04 — PROJECT DEEP-DIVE PLAYBOOK

> **TL;DR.** Part 2 is where the round is usually decided. They will take *your latest agentic AI project* and drill four or five levels: what, why, how, proof, production. This file gives you (a) the 90-second summary to rehearse, (b) the one-page architecture you must be able to draw from memory, (c) a decision log so no "why did you choose X" catches you cold, (d) the ~35 questions they will ask, grouped by rung, and (e) the six numbers you must know without looking anything up.

**60-second interview answer (the whole file, if asked "tell me about your project"):** Problem → Constraint → Architecture → the one decision → Result → Limitation. See §2.

---

## 1. Why this part is different from a "tell me about your work" question

The source funnel names the failure mode: *"No metrics / results from your own work."* And `prev_learning` adds the deeper one: the interviewer *"had to prompt you several times to 'get deeper'"* — i.e. the candidate stayed at the level of concepts instead of implementation. So the bar is:

| Level | Weak answer | Lead-level answer |
|---|---|---|
| L1 – What | "We built a RAG-based assistant." | "We built a submission-triage service that extracts 14 fields from a broker PDF and routes it." |
| L2 – Why | "Because LLMs are good at text." | "Because the alternative was 4 minutes of underwriter time per submission, and 60% of the fields were already present verbatim in the document." |
| L3 – How | "We used LangChain and a vector DB." | "Hybrid retrieval: BM25 plus dense, pre-filtered by line-of-business metadata, then a cross-encoder rerank to top-5; extraction via a constrained JSON schema with a Pydantic validator." |
| L4 – Proof | "It worked well." | "Golden set of 120 cases, per-field F1 on extraction, trajectory eval on the agent, and 94% agreement with the underwriter's own triage in shadow mode." |
| L5 – Production | "It's in production." | "It runs on Azure, costs £X per 1,000 submissions, P95 8.4s, one incident — a schema drift that the validator caught and routed to human review." |

**Your goal: be fluent at L5.** Anything you cannot describe at L5, do not lead with.

---

## 2. The 90-second project summary (rehearse to the second)

Fill this in *before the interview* and record yourself. Six beats, ~15 seconds each.

| Beat | Template | What it must contain |
|---|---|---|
| **Problem** | "The business problem was [pain], which cost [unit] per [period]." | A cost or time number attached to the pain |
| **Constraint** | "The constraint that shaped the design was [latency / accuracy / regulation / data sensitivity]." | *One* dominant constraint |
| **Architecture** | "The system was [box] → [box] → [box] → [box]." | 3–5 named boxes, in flow order |
| **Decision** | "The decision I'd point to is [X] over [Y], because [reason]." | A real trade-off, not a default choice |
| **Result** | "It delivered [metric 1] and [metric 2]." | Two numbers, at least one business-facing |
| **Limitation** | "What it still doesn't do is [honest gap]." | Volunteered, not extracted |

**Worked example (adapt to your truth):**

> *"The problem was broker submission triage — every submission arrived as a multi-page PDF plus email, and an underwriter spent about four minutes a submission just to decide whether it was in appetite. The constraint was trust: nothing could be auto-declined, and every field we extracted had to be traceable to a page and a box in the source document. The architecture was ingestion → Document Intelligence for layout → a hybrid retrieval layer over our appetite and wording corpus → a constrained extraction step → a deterministic validator → a triage queue for the underwriter. The decision I'd point to is keeping the LLM inside a JSON schema with server-side Pydantic validation rather than letting it free-text and parsing afterwards — it cost us some recall but it eliminated malformed output reaching the workflow. The result was triage time down about 40% and 30% of submissions pre-populated end-to-end. What it still doesn't do is handle handwritten endorsements reliably — those still route to a human."*

---

## 3. The one-page architecture you must draw from memory

Before the interview, draw your project **twice, from memory, on blank paper**, and check against the real thing. Label:

- [ ] Every box, with the technology named
- [ ] The **data flow direction** (arrows)
- [ ] Where the **human** sits (HITL gate)
- [ ] Where **validation** happens (deterministic checks)
- [ ] Where the **trace/eval** hook is
- [ ] Where it can **fail**, and what happens then
- [ ] The **system of record** it writes back to

Then, from the same memory, be able to answer: *"What happens if the vector DB is down?", "What if the LLM returns invalid JSON?", "What if the same submission arrives twice?"*

---

## 4. The decision log — pre-answer these six, or they will find them

Write your real answers. Each should be **"X over Y because Z, and the cost was W."**

| # | The decision they will probe | Why they probe it |
|---|---|---|
| 1 | **Why an agent at all?** Why not a fixed pipeline / RAG-only / prompt-only? | Testing whether you over-engineer. The escalation ladder: prompt → RAG → tools/agents → fine-tune. |
| 2 | **Orchestration:** framework (LangGraph/LangChain) vs custom state machine vs event-driven? | Testing judgement, not fashion. |
| 3 | **Model choice and tiering:** which model for which step, and why? | Testing cost awareness. |
| 4 | **Retrieval:** chunking strategy, hybrid vs dense, reranker, metadata filters? | Testing RAG depth. |
| 5 | **Output control:** structured outputs, schema, deterministic validation, retries? | Testing the "LLMs are non-deterministic" maturity. |
| 6 | **Evaluation & guardrails:** how you know it works, and what stops a bad output. | The single most discriminating question — file `07`. |

### 4.1 Model answers for the two hardest of the six

**"Why an agent instead of a fixed pipeline?"**

> *"Because the task had genuine branching — if the document is a renewal we need prior-year data, if it's a new binder we need the broker's authority check, and the number of steps depends on what the first extraction returns. A fixed DAG would have meant encoding every branch, and the branches changed quarterly. So: an agent with a small, well-specified tool set. But I kept the agent on a short leash — maximum four tool calls, every tool validated server-side, and if it exceeded the budget it fell back to a deterministic path. The trade-off is debuggability: a state machine is easier to trace, which is why I logged every step with LangFuse."*

**"Why structured outputs rather than parsing free text?"**

> *"Determinism. If the model free-texts and I regex afterwards, every prompt change can silently change the parse. Constraining it to a JSON schema moves the failure from 'subtly wrong value' to 'explicit schema violation', which the validator catches and routes to a human. The cost is a small recall hit on fields where the model would have expressed the value in a way the schema couldn't represent — we handled those with an explicit `unparsed` escape hatch rather than by widening the schema."*

---

## 5. The drill-down ladder — ~35 questions, grouped by rung

Prepare an answer for every one. Mark the ones you cannot answer at L5 and spend tonight on those.

### Rung 1 — What (they always start here)
1. Walk me through the project end to end.
2. What was your specific role?
3. How big was the team, and who did what?
4. How long did it take, and what was the hardest part?
5. What was the business case — who funded it and why?

### Rung 2 — Why (design rationale)
6. Why this architecture and not something simpler?
7. Why did you need an agent / RAG / tools at all?
8. What were the two or three alternatives you rejected?
9. Why that vector database?
10. Why that orchestration approach?
11. Why not fine-tune?
12. What would have broken if you'd removed the human-in-the-loop?

### Rung 3 — How (implementation depth)
13. Explain the RAG architecture in detail — with code.
14. What's your chunking strategy and why those numbers?
15. Hybrid or dense retrieval? What weighting? How do you tune it?
16. Do you rerank? Which model? What's the latency cost?
17. How are embeddings cached?
18. How do you handle a document longer than the context window?
19. Show me how the tool definitions are declared.
20. How do you validate the model's output before it's used?
21. What happens when the model returns invalid JSON?
22. How do you handle the same request arriving twice?
23. How do you handle a timeout mid-workflow?
24. What does your prompt look like — how do you separate instructions from retrieved content?
25. How do you defend against prompt injection from a malicious document?

### Rung 4 — Proof (evaluation — file `07`)
26. How did you evaluate your agents?
27. What was your eval set, and who wrote the ground truth?
28. What metrics? Why those and not accuracy?
29. How do you evaluate a *trajectory*, not just the final answer?
30. What did the last evaluation change about the system?
31. How do you detect drift after launch?

### Rung 5 — Production
32. What broke in production, and what did you change structurally?
33. What does it cost per unit, and how did you get it there?
34. Who owns it now, and what's the on-call story?
35. What would you do differently if you started again?
36. How did adoption actually happen — how did you get underwriters to use it?

### 5.1 The three that most often end a candidate

| Question | The trap | The Lead answer |
|---|---|---|
| "How did you evaluate your agents?" | Answering with output accuracy only | **Three layers: component, trajectory, online.** Trajectory eval = did it call the right tools in the right order, not just did it get the right answer. |
| "What would you do differently?" | "Nothing, it went well" or a vague answer | One *specific* technical and one *specific* process change, with the reason |
| "What was your contribution vs the team's?" | Over-claiming, or false modesty | Precise: "I designed X and wrote Y; Z was [person]'s; I owned the evaluation standard." |

---

## 6. The numbers sheet — six numbers, memorised

Write yours here. If you cannot fill all six, estimate honestly and *say* it is an estimate.

| # | Number | Yours | Why they ask |
|---|---|---|---|
| 1 | **Scale** — documents/day, requests/day, users | | Proves the system was real |
| 2 | **Latency** — P50 and P95, and what the user tolerates | | Proves you know your own system |
| 3 | **Quality** — the primary metric and its value | | Proves the work landed |
| 4 | **Cost** — per document / per conversation / per month | | Proves commercial thinking; **Brit cares about cost** |
| 5 | **Team** — size, disciplines, your span | | Proves Lead scope |
| 6 | **Outcome** — the business number (£, hours, %, adoption) | | The only thing the panel remembers |

---

## 7. Handling the honesty questions (these win points)

| Situation | Say this |
|---|---|
| You don't remember a number | *"I don't have that precisely in front of me — it was in the region of [X], and I'd rather check than guess."* Honesty beats a fabricated figure that collapses on the next question. |
| You didn't own a piece | *"That piece was led by [role]; I reviewed it / depended on it. What I owned was [X]."* |
| The project failed | *"It was sunset because [reason]. The thing I took from it was [lesson]."* |
| The result was modest | Never inflate. Say the true number and the *reason* it was the right call: *"It was a 6% improvement, but it was on our highest-volume queue, so it was worth more than a 20% gain elsewhere."* |

---

## 8. Common mistakes in this part

| Mistake | Fix |
|---|---|
| Describing the system as "we" throughout | Use "I" for your decisions; "we" for the team effort |
| Staying conceptual when pressed | Go to parameters, code, and failure modes |
| No evaluation story | File `07` — this is the most-asked question |
| Claiming perfection | Volunteer one limitation unprompted |
| Not knowing the cost | Know at least one unit-cost figure |
| Not mentioning the human | In insurance, always name where the human decides |
| Ignoring adoption | "How did you get people to actually use it?" is a Lead question — have an answer |

---

## 9. Cheat-sheet summary

- **90 seconds: Problem → Constraint → Architecture → Decision → Result → Limitation.**
- **Be fluent at L5 (production), not L1 (what it is).**
- **Six numbers, memorised:** scale, latency, quality, cost, team, outcome.
- **Six decisions pre-answered:** agent-vs-pipeline, orchestration, model tiering, retrieval, output control, evaluation.
- **~35 questions across five rungs** — answer all of them at least once out loud.
- **Volunteer a limitation.** Then name the human gate. Then name the cost.

## 10. Cross-links

- The evaluation answers: [`07-agent-evaluation-and-guardrails.md`](07-agent-evaluation-and-guardrails.md)
- RAG/orchestration depth: [`05-system-design-round-playbook.md`](05-system-design-round-playbook.md) · [`10-question-bank-with-model-answers.md`](10-question-bank-with-model-answers.md)
- Team/ownership stories: [`12-people-leadership-and-stakeholder-round.md`](12-people-leadership-and-stakeholder-round.md)
- Rehearsal harness: [`08-applied-ai-mock-interview-prompt.md`](08-applied-ai-mock-interview-prompt.md)
