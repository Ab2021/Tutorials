# 02 — STAGE-1 FORMAT & THE 60-MINUTE BATTLE PLAN

> **TL;DR.** The hour is four parts: **Introductions → Project drill-down → System design → Coding exploration**. Each part scores a different competency. Your job is not to be brilliant in all four — it is to *not lose any of the four*. This file gives you a minute-by-minute run-sheet, the opening and closing scripts, the transition lines between parts, and the failure modes to avoid in each.

**60-second interview answer (if asked "how do you want to use this hour?"):** *"I'd suggest we spend it the way you've structured it — a short intro, then I'd rather you go deep on a real project than stay broad, then a system design where I'll clarify before I draw, and finally whatever code or repo you'd like to look at. I'm happy to be interrupted — the drill-down is where you'll learn the most about how I actually work."*

---

## 1. The run-sheet

| Part | Duration | What they score | Your objective | Prepared in |
|---|---|---|---|---|
| **1. Introductions** | ~5–8 min | Communication, seniority, fit, whether you're "Lead" | Land the positioning sentence + one proof point. Get *them* talking. | `03` |
| **2. Project drill-down** | ~15–20 min | Ownership, production maturity, evaluation rigour, honesty | Be the person who *built* it, not the person who *attended* it. Have numbers. | `04`, `07` |
| **3. System design** | ~15–20 min | Requirements-first thinking, trade-offs, cost/latency/safety reasoning | Clarify → estimate → draw → deep-dive → self-critique. Name the dials. | `05` |
| **4. Coding exploration** | ~10–15 min | Real hands-on ability, team engineering practice | Read code aloud, critique constructively, answer the Git question cleanly. | `06` |
| **Wrap** | ~3 min | Interest, judgement | Ask 2–3 of the questions from `01` §7. | `01` |

> **Time control is yours.** If Part 2 overruns because you are rambling, Part 4 gets cut and you lose the coding signal. When you finish an answer, *stop talking* — do not fill.

---

## 2. Part 1 — Introductions (the script)

### 2.1 The 60-second intro (memorise this shape, not the words)

> **[Present]** *"I'm a [N]-year engineer, currently [role] at [company], where I lead [team size] people building [domain] AI."*
> **[Credibility]** *"Most recently [one-line project] — [one number]."*
> **[Relevance to Brit]** *"The reason I applied is the Smarter Underwriting agenda — [one domain sentence]."*
> **[Hand-back]** *"That's the short version. Where would you like me to go deeper — the team, or the most recent project?"*

**Why it works:** it is four sentences, it contains a number, it ends with a *hand-back question* (you are supposed to be a Lead — lead the conversation), and it invites the drill-down you have prepared hardest for.

### 2.2 The 2-minute version (only if invited)

Add, in this order: **(a)** the arc — how you moved from classical ML into GenAI; **(b)** your team — who was in it, what they owned; **(c)** one standard you introduced; **(d)** why insurance now. Then hand back.

### 2.3 What NOT to do in Part 1

| Don't | Because |
|---|---|
| Recite your CV chronologically | They have it. You are wasting the only minutes where you control the agenda. |
| Say "I'm passionate about AI" | Zero information. Replace with a domain-specific reason. |
| Talk for 5 minutes without a question back | Seniority = conversation control, not monologue length. |
| Under-sell impact ("we helped with…") | Explicit failure mode in the source funnel: *"Under-selling project impact."* |

---

## 3. Part 2 — Project drill-down (the make-or-break part)

This is where the round is usually won. They will pick *your latest agentic AI project* and go down four or five levels. Full preparation is in [`04-project-deep-dive-playbook.md`](04-project-deep-dive-playbook.md); the run-sheet version:

### 3.1 The 90-second project summary (rehearse to the second)

Structure it as **Problem → Constraint → Architecture → Decision → Result → Limitation**:

> *"The problem was [business pain, with a number attached]. The constraint that shaped everything was [latency / accuracy / regulation / data sensitivity]. The architecture was [3 boxes, named]. The one decision I'd point to is [the non-obvious trade-off you made] — we chose X over Y because [reason]. The result was [number one] and [number two]. And the limitation, honestly, is [what it still can't do]."*

**The limitation sentence is a seniority signal.** Candidates who volunteer a limitation are trusted; candidates who claim perfection are not.

### 3.2 The drill-down ladder — expect all five rungs

| Rung | Their question | Have ready |
|---|---|---|
| 1 — What | "Walk me through it." | The 90-second summary |
| 2 — Why | "Why that architecture and not [simpler]?" | The escalation ladder: prompting → RAG → tools/agents → fine-tune. Justify each hop. |
| 3 — How | "Show me the retrieval / orchestration code." | Chunking params, reranker, prompt assembly, tool schemas |
| 4 — Proof | "How did you know it worked? How did you evaluate the *agents*?" | File `07` — trajectory eval, golden sets, online metrics |
| 5 — Production | "What broke in production? What does it cost? Who owns it?" | One real incident, one cost figure, one ownership answer |

### 3.3 Three questions you *must* be able to answer instantly

1. **"How did you evaluate your agents?"** → headline: *"Three layers — component eval, trajectory eval, and online business metrics. Let me start with trajectory."* (file `07`)
2. **"What would you do differently?"** → have one *specific* answer, not "better testing".
3. **"What was your personal contribution versus the team's?"** → be precise and generous; name what others did.

---

## 4. Part 3 — System design (the run-sheet)

Full method in [`05-system-design-round-playbook.md`](05-system-design-round-playbook.md).

### 4.1 The five phases and their time budget (scale to the ~15–20 min you get)

| Phase | Time | You are doing | The sentence that proves it |
|---|---|---|---|
| 1. Clarify | 3 min | Ask one question per dial | *"Before I draw, four questions: accuracy bar, latency target, per-decision cost budget, and whether the output is regulated."* |
| 2. Estimate | 2 min | QPS, peak, tokens, storage | *"At [N] docs/day that's ~[x] QPS median, ~[y] peak."* |
| 3. Draw | 6 min | Boxes left-to-right | *"Client → gateway → router → [your stage] → validator → HITL."* |
| 4. Deep-dive | 6 min | The subsystem they pick | *"The lever here is the reranker / the cache / the queue."* |
| 5. Self-critique | 2 min | Name what you left out | *"What I haven't covered: failure modes, cost ceiling, and drift monitoring."* |

### 4.2 The line that makes you sound like a Lead

> *"Every box I add has to be justified by a constraint you gave me. If there's no constraint, I'd delete it — in production, each extra sequential component multiplies the error rate, not adds to it: 0.9 × 0.9 × 0.9 = 0.73."*

### 4.3 The most likely prompt (from candidate feedback)

> *"Given you have images, design a system to run them through a model for checking anomalies."*

This is their post-catastrophe property-damage use case. Have this design **drawn twice from memory** before the interview. The worked answer is in `05` §6.

---

## 5. Part 4 — Coding exploration (the run-sheet)

Full method in [`06-coding-exploration-playbook.md`](06-coding-exploration-playbook.md).

| Likely task | Your 30-second approach script |
|---|---|
| "Explain this RAG code and what you'd improve" | *"Let me read it in three passes: first the data flow, then the retrieval parameters, then the failure handling. I'll narrate as I go."* Then walk it and name **two** concrete improvements. |
| "Write a function to …" | Restate the requirement, state the interface, say the complexity, then code. |
| "How do you have linear commit history with multiple people?" | See `06` §4 — this is a *policy* question, and it has a clean, confident answer. |
| "How would you test this?" | Name unit → contract → eval regression → canary. |

**Rule for Part 4:** narrate your thinking out loud and *do not go silent*. Silence reads as "can't code". If you do not know an API, say *"I'd look up the exact signature — the shape of the call is…"* and continue.

---

## 6. The transitions — pre-written bridge lines

Say these to move between parts cleanly. They make you sound organised and they buy you 3 seconds.

| Transition | Line |
|---|---|
| Into Part 2 | *"Happy to go deep on the project — do you want the business context first, or the architecture?"* |
| Out of Part 2 | *"That's the system as built. I can go deeper on evaluation or on the retrieval layer, but I suspect you'd rather move to the design question."* |
| Into Part 3 | *"Before I draw anything, can I ask four scoping questions? Accuracy bar, latency target, cost budget, and whether the output is regulated."* |
| Out of Part 3 | *"I'd call that the sketch. The parts I'd want to firm up next are failure handling and the cost model."* |
| Into Part 4 | *"I'm comfortable reading code aloud — share whatever's most useful and I'll narrate my thinking."* |
| Wrap | *"Before we finish, can I ask you two things about the team?"* |

---

## 7. Materials and logistics checklist

- [ ] **Whiteboard/diagram tool tested** 10 minutes before: Excalidraw, Miro, or paper + camera. Draw *legibly*; label every box.
- [ ] **Your project architecture on one page** (digital + memory).
- [ ] **Your resume open** with the project bullets highlighted, so you can point at specifics.
- [ ] **Numbers sheet** — 6 numbers about your project (scale, latency, accuracy, cost, team size, outcome).
- [ ] **Water, notebook, pen.** Write the interviewer's questions down as they ask — it buys thinking time and signals care.
- [ ] **Camera framing, lighting, headset.** The last interview failed partly on *speech clarity*; a headset directly fixes that.
- [ ] **Phone silent, notifications off, browser tabs closed.**

---

## 8. Failure modes for this exact round

| Failure | Where it kills you | Fix |
|---|---|---|
| Rambling past the time budget | Part 2 eats Part 4 | Stop after 90 seconds; hand back with a question |
| Drawing before clarifying | Part 3 | The 4 dials, always |
| Listing tools instead of integrating them | Parts 2 & 3 | "It solves X because Y" for every tool named |
| No numbers anywhere | Parts 2 & 3 | Memorise 6 numbers |
| Ignoring the people side | Parts 1 & wrap | Files `01` §4 and `12` |
| Filler words when thinking | all | Pause. Silence is free; "uh" is expensive. |
| Saying "I can't think of it" | Parts 3 & 4 | Bridge statement, then a framework |
| Claiming perfection | Part 2 | Volunteer the limitation |

---

## 9. Cheat-sheet summary

- **Four parts, four competencies.** Intro = fit; Project = ownership; Design = trade-offs; Coding = hands-on.
- **Part 2 is the decider.** 90-second summary, five-rung drill, six numbers, one honest limitation.
- **Part 3 is process, not answer.** Clarify → estimate → draw → deep-dive → self-critique. Name the dials.
- **Part 4 needs narration.** Read code in three passes; never go silent; the Git question is a policy question.
- **Hand back with a question** at the end of every part. You are the Lead.
- **Never overrun.** Protect Part 4's time by stopping Part 2 early.

## 10. Cross-links

- Intro scripts: [`03-introduction-and-story-bank.md`](03-introduction-and-story-bank.md)
- Project drill: [`04-project-deep-dive-playbook.md`](04-project-deep-dive-playbook.md) · [`07-agent-evaluation-and-guardrails.md`](07-agent-evaluation-and-guardrails.md)
- Design round: [`05-system-design-round-playbook.md`](05-system-design-round-playbook.md)
- Coding round: [`06-coding-exploration-playbook.md`](06-coding-exploration-playbook.md)
- Rehearsal: [`08-applied-ai-mock-interview-prompt.md`](08-applied-ai-mock-interview-prompt.md) · [`09-mock-interview-strategies-and-drills.md`](09-mock-interview-strategies-and-drills.md)
