# 09 — MOCK INTERVIEW STRATEGIES & DRILLS

> **TL;DR.** Rehearsal only works if it is *structured*: a specific format, a specific interviewer persona, a recorded transcript, and a rubric scored the same way every time. This file gives you four mock formats, three interviewer persona cards, the drill schedule, and the 30-minute analysis loop that turns a mock into a fix. The whole point is to make the real hour feel like the fifth time, not the first.

**60-second interview answer:** *"I rehearse in four formats — a full 60-minute loop, a Part-2-only drill, a Part-3-only drill, and a delivery drill with no content at all — and I record every one, transcribe it, and score it against the same eight criteria. The recordings are where the improvement actually happens."*

---

## 1. The four mock formats

| Format | Length | What it isolates | How often |
|---|---|---|---|
| **F1 — Full loop** | 60 min | Endurance, time control, transitions | 2× before the interview |
| **F2 — Project drill only** | 20 min | Depth on your own work; the make-or-break part | 4× |
| **F3 — Design only** | 20 min | Clarify → estimate → draw → deep-dive → critique | 4× |
| **F4 — Delivery drill** | 15 min | Filler words, pacing, headlining — *content deliberately unimportant* | daily, 10 min |

### 1.1 How to run F1 (the full loop)

1. **Paste the prompt** from [`08`](08-applied-ai-mock-interview-prompt.md) §1 **+ Variant B** into a fresh LLM chat.
2. **Speak your answers aloud** into a recorder. Type nothing except a single line to tell the model you've finished.
3. **Set a visible timer.** Enforce the budget: intro 5, project 18, design 20, coding 12, wrap 5.
4. **Do not stop to look things up.** If you don't know, use a bridge statement and keep going. That is the skill being trained.
5. **Transcribe and score** with §4. Write the two fixes at the top of the next session.

### 1.2 How to run F2–F4

- **F2:** open your own project's README/diagram, ask an LLM to *"interview me about this project, four levels deep, and interrupt when I'm vague."* Paste in a one-paragraph project description first so the questions are grounded.
- **F3:** draw on a whiteboard while you talk. Redraw the **image-anomaly** design twice from memory ([`05`](05-system-design-round-playbook.md) §6.1).
- **F4:** take any 10 questions from [`10`](10-question-bank-with-model-answers.md), answer each in **under 60 seconds**, and count filler words in the transcript. Zero fillers is the goal.

---

## 2. Interviewer persona cards

Ask your LLM to adopt one persona per session. Different personas expose different weaknesses.

### Persona A — "The Practitioner" (most likely at Brit)

> *You are a hands-on principal engineer. You do not care about frameworks or buzzwords; you care whether the candidate has actually run the thing in production. Interrupt any answer that stays abstract and ask for the parameter, the number, or the failure that taught them that. Ask at least once: "what broke in production?" and at least once: "what does it cost per unit?"*

**Exposes:** theoretical answers, missing numbers, no production scar tissue.

### Persona B — "The Business Partner"

> *You are a head of data/AI who owns the budget and reports to the board. You don't understand tokens or vectors, and you stop the candidate whenever they use jargon without explaining it. You ask: "what does this save us, what does it risk, and who is accountable when it's wrong?" You push back on cost.*

**Exposes:** jargon leakage, no business framing, no risk ownership. **This is the persona that tests the panel-feedback gap.**

### Persona C — "The Skeptic"

> *You are openly doubtful. You assume the candidate is exaggerating their numbers, that the project was smaller than described, and that the design has an obvious hole. You say things like "that sounds like it would be too slow", "who actually reviewed this?", "what if the source document is wrong?" and "I don't believe that accuracy figure — how did you measure it?"*

**Exposes:** over-claiming, thin evaluation stories, defensiveness. **Practise staying warm and specific.**

### Persona D — "The People Manager"

> *You are the hiring manager for a team of twelve. You spend most of the interview on the people side: how they set goals, handle an underperformer, develop someone from science into engineering, resolve conflict, and build a community of practice. You are unimpressed by pure technical depth.*

**Exposes:** the mentoring gap. Run this one at least once.

---

## 3. Drill schedule (7 days, ~2 hours/day)

| Day | Drill | Duration | Success criterion |
|---|---|---|---|
| 1 | Write + record the 60-second intro; count fillers | 30 min | Zero fillers; ≤ 60 s |
| 1 | Read `01`, `02`; write the positioning sentence | 30 min | Can say it without notes |
| 2 | Learn 25 insurance terms (`11`); translate 10 technical terms (`03` §3) | 45 min | Each in < 15 s, jargon-free |
| 2 | **F4** delivery drill on 10 questions | 20 min | Zero fillers, ≤ 60 s each |
| 3 | Rebuild your project on one page from memory; fill the six numbers (`04` §6) | 45 min | Diagram matches reality; 6 numbers known |
| 3 | **F2** project drill, Persona A | 25 min | Answers at L5 (production), not L1 |
| 4 | **F3** design ×2 — the image-anomaly prompt | 50 min | Clarifies before drawing, both times |
| 4 | Write the evaluation narrative (`07` §9) aloud ×3 | 20 min | ≤ 90 s, four numbers, one trade-off |
| 5 | **F1** full loop, Persona A + Variant B | 75 min | Score ≥ 4 on criteria 1–4, 6 |
| 6 | Re-run whichever part scored lowest, with a different persona | 45 min | That criterion moves to ≥ 4 |
| 6 | **F1** full loop, Persona C (skeptic) | 60 min | No defensiveness; stays specific |
| 7 | Re-read `02` §3 and `13` only. Run one **F4** warm-up. | 30 min | Calm, not cramming |

---

## 4. The scoring rubric (use every time — consistency is the point)

Score 1–5. **Track two sessions in a row**; improvement only counts if a weak score moves and stays moved.

| # | Criterion | 1 | 3 | 5 | Weight |
|---|---|---|---|---|---|
| 1 | **Clarify first** | Drew/answered immediately | Asked some questions | One per dial, wrote them down | ★★★ |
| 2 | **Numbers** | None | One or two | QPS, tokens, cost, latency decomposed | ★★★ |
| 3 | **Headline first** | Started mid-answer | Conclusion present but buried | One-sentence conclusion, then Why/How | ★★★ |
| 4 | **Trade-off named** | No alternative mentioned | Mentioned, not costed | "X over Y because Z; the cost is W" | ★★★ |
| 5 | **Production reality** | Purely theoretical | Some production detail | Named an incident, an owner, a cost | ★★★ |
| 6 | **Self-critique** | Claimed complete | Vague gap named | Specific gaps + what I'd monitor | ★★ |
| 7 | **People/business** | Absent | Mentioned | Named the human, the stakeholder, the metric | ★★★ |
| 8 | **Delivery** | Fillers, rushing | Occasional fillers | Paused, measured, zero fillers | ★★ |

**Rule:** any ★★★ criterion below 4 on the *day-6* mock is the single thing you fix on day 7 morning.

---

## 5. The 30-minute analysis loop (turns a mock into a fix)

| Step | Time | Do |
|---|---|---|
| **Transcribe** | 5 min | Use a speech-to-text tool, or type the three worst answers |
| **Count fillers** | 3 min | Ctrl-F "uh", "like", "um", "so yeah", "basically". Write the count. |
| **Mark the vague spots** | 5 min | Highlight every sentence with no number, no name, no mechanism |
| **Score the rubric** | 5 min | 8 criteria, honestly |
| **Write two fixes** | 5 min | Not five. Two. Write them as *sentences to say*, not topics to study. |
| **Re-record only those two** | 7 min | Rehearse the fix in isolation, then re-run the affected part |

**Example fix format (this is what a good fix looks like):**

> ❌ "Revise evaluation."
> ✅ *"When asked how I evaluated the agent, open with: 'Three layers — component, trajectory, online — and a frozen golden set of about 120 cases in CI.' Then give one number per layer."*

---

## 6. Delivery drills (the `prev_learning` fixes, practised)

### 6.1 The pause drill
Ask yourself a question, then **count to two in your head before speaking**. Record. The silence feels enormous to you and composed to the listener. Repeat 10×.

### 6.2 The headlining drill
Take 10 answers. Write **only the first sentence** of each — the headline. Say them from the card. If a headline is longer than 20 words, it's not a headline.

### 6.3 The bridge drill
Rehearse these until automatic, so "I don't know" never leaves your mouth:
- *"That's an interesting edge case — the first thing I'd pin down is…"*
- *"I haven't hit that exact case; here's how I'd reason about it from first principles…"*
- *"I don't have that number precisely — it was in the region of X — and I'd rather check than guess."*
- *"Let me restate it to make sure I've understood: you're asking whether…"*

### 6.4 The jargon drill
Explain one technical term ([`03`](03-introduction-and-story-bank.md) §3) in one underwriter-friendly sentence, in under 15 seconds. 10 rounds. Persona B will fail you otherwise.

### 6.5 The slowdown drill
Speak at ~140 wpm. A Lead sounds unhurried. Record a fixed passage three times, timing it; aim for the same duration each time rather than fast.

---

## 7. The day-of warm-up (20 minutes, no new material)

| Minute | Do |
|---|---|
| 0–3 | Re-read `13` (cheat sheet) |
| 3–6 | Re-read `02` §3 (the run-sheet) |
| 6–9 | Say your 60-second intro out loud twice |
| 9–12 | Say the evaluation narrative (`07` §9) out loud once |
| 12–15 | Say the Git one-liner (`06` §4.5) out loud once |
| 15–18 | Run one **F4** drill on three random questions |
| 18–20 | Water, posture, mic check, phone silent. **Stop reading.** |

---

## 8. Common mistakes in rehearsal

| Mistake | Fix |
|---|---|
| Reading instead of speaking | Every drill is out loud, recorded |
| Practising only the parts you're good at | Start with the rubric's lowest score |
| Running mocks without scoring | Use the same 8 criteria every time |
| Fixing ten things at once | Two fixes per session, written as sentences |
| Not rehearsing the *transitions* | Say the bridge lines from `02` §6 |
| Rehearsing alone only | One mock with a real human, ideally a colleague, as Persona B |
| Cramming new material on day 7 | Day 7 is consolidation only |

---

## 9. Cheat-sheet summary

- **Four formats:** full loop (F1) · project drill (F2) · design drill (F3) · delivery drill (F4, daily).
- **Four personas:** Practitioner · Business Partner · Skeptic · People Manager. Run all four at least once.
- **Score the same 8 criteria every time.** ★★★ criteria below 4 on day 6 = your day-7 fix.
- **Two fixes per session, written as sentences to say.**
- **Delivery drills:** pause · headline · bridge · jargon · slowdown.
- **Day 7 = consolidate, don't cram.** Warm-up only, then stop reading.

## 10. Cross-links

- The prompt to drive F1: [`08-applied-ai-mock-interview-prompt.md`](08-applied-ai-mock-interview-prompt.md)
- Drill material: [`10-question-bank-with-model-answers.md`](10-question-bank-with-model-answers.md)
- Delivery fixes in context: [`03-introduction-and-story-bank.md`](03-introduction-and-story-bank.md) §4
- Final revision: [`13-final-48-hours-cheat-sheet.md`](13-final-48-hours-cheat-sheet.md)
