# 13 — FINAL 48 HOURS CHEAT SHEET

> **TL;DR.** Everything in this pack compressed to one printable sheet. **Read only this in the last 48 hours.** Nothing new — consolidation only. If a line here doesn't trigger the full answer in your head, that's the one thing to re-read.

---

## 1. The run-sheet (memorise)

| Part | Time | Objective | Opener |
|---|---|---|---|
| **1. Introductions** | 5–8 min | 4 sentences, 1 number, hand back | "I'm a [N]-year engineer leading [N] people… would you like the team, or the project?" |
| **2. Project drill-down** | 15–20 min | Be fluent at L5, not L1 | "Problem → Constraint → Architecture → Decision → Result → **Limitation**" |
| **3. System design** | 15–20 min | Process, not answer | "Before I draw, four questions — one per dial." |
| **4. Coding exploration** | 10–15 min | Narrate, critique, Git | "I'll read in three passes: data flow, parameters, failure handling." |
| **Wrap** | 3 min | Ask 2–3 questions | "Before we finish, can I ask you two things about the team?" |

**Never overrun Part 2.** Protect Part 4's time by stopping at 90 seconds per story.

---

## 2. The five rules (break these and nothing else matters)

1. **Headline first** → one-sentence conclusion, then Why, then How.
2. **Pause 2–3 seconds** before answering. Silence is free; "uh" is expensive.
3. **Never say "I can't think of it."** Bridge: *"That's an interesting edge case — the first thing I'd pin down is…"*
4. **Problem → Solution → Trade-off.** X over Y because Z, and the cost is W.
5. **Every metric ties to cost of error.** Precision at auto-approve; recall in triage.

**Bonus:** say **"deterministic validation layer"** — the model proposes, the validator disposes.

---

## 3. The four dials

Ask one question per dial before drawing anything: **Quality** (accuracy bar, cost of a wrong answer) · **Cost** (per-decision budget) · **Latency** (P95 target) · **Safety** (regulated? who's accountable?).

---

## 4. Part 1 — the intro

> *"I'm a [N]-year engineer, currently [role] at [company], where I lead [N] people across [disciplines]. Most recently, [one-line project] — [one number]. The reason I applied is [Smarter Underwriting / Operational Simplification] — [one domain sentence]. That's the short version — would you like the team, or the project?"*

**First-90-days:** listen & map → one use case end-to-end with a named owner → install one standard (eval gate in CI).

---

## 5. Part 2 — the project

**The 90-second summary:** Problem (with a number) → Constraint (one dominant) → Architecture (3–5 named boxes) → **the one decision** (X over Y because Z) → Result (two numbers) → **Limitation** (volunteered).

**Six numbers, memorised:** scale · latency P50/P95 · quality metric · **cost per unit** · team size · business outcome.

**Six decisions pre-answered:** why an agent at all · orchestration choice · model tiering · retrieval design · output control · evaluation.

**The three questions that end candidates:**
- *"How did you evaluate your agents?"* → **three layers** (§7)
- *"What would you do differently?"* → one specific technical + one process change
- *"Your contribution vs the team's?"* → precise and generous

**Honesty lines:** *"I don't have that number precisely — I'd rather check than guess."* · *"That piece was led by [role]; what I owned was X."*

---

## 6. Part 3 — system design

**Phases:** Clarify (3) → Estimate (2) → Draw (6) → Deep-dive (6) → Self-critique (2).

**The sketch:** Client → Load Balancer → API Gateway (auth + **RPS and token** limits) → Intent Router → {RAG + Memory + Tools} → LLM (streamed) → Output Validator → **HITL gate** → Tracing/Eval → Response.

**Retrieval funnel:** chunk → embed → ANN → hybrid (**RRF, k=60**) → metadata **pre-filter** → rerank **top-50→5** → most-relevant-first.

**The numbers:**

| Number | Value |
|---|---|
| QPS | daily / 86,400; peak ×2–3 |
| In-flight | QPS × duration (Little's Law) |
| KV cache | ~0.5 MB per 1k tokens (7B fp16) |
| Output vs input cost | **~3×** |
| Buffered 380 tok @ 25 tok/s | **16.1 s → 38% abandonment** |
| RAG TTFT | retrieval 200–500 + rerank 100–200 + prefill 200–500 = **0.5–1.2 s** |
| Chunk sizes | FAQ 100–300 · general 300–500 · legal 500–1,000 · overlap 10–20% |
| Semantic cache | cosine ≥ 0.95, 20–40% hit |
| Model tiering | 75% small / 20% mid / 5% frontier |
| Error compounding | 0.9³ = **0.73** |
| Fan-out | n=100, p=1% → **63.4%** slow |
| Agent chain | 0.97²⁰ ≈ **54%** |

**Reliability:** deadline propagation (downstream < upstream) · **one layer retries** · exponential backoff + **full jitter** · retry budget ~10% (token bucket) · **idempotency keyed on intent**.

**Streaming:** SSE, not WebSockets. Un-buffer at service + gateway + proxy (`proxy_buffering off`) + **gzip OFF**.

**Self-critique, always:** cost ceiling · failure modes · drift monitoring · **adoption**.

**The image-anomaly design (asked last time):** queue (bursts) → tile with overlap (store tile→image map) → batched GPU inference → stitch & dedupe → **threshold in code, not in the prompt** → auto-flag / human queue → **manual fallback if the model is down** → evidence store (image, tiles, boxes, model version, confidence, reviewer decision). Precision at auto-flag; recall in the human queue.

---

## 7. Part 2/3 — the evaluation answer (the money answer)

> *"Three layers, and one gate. **Component** — retrieval and generation measured separately, so I know which stage failed: Recall@K, MRR, context precision; faithfulness and citation correctness. **Trajectory** — tool selection, arguments, order, step count versus expected, and recovery, because a right answer by an unsafe path is still a failure. **Online** — task completion, escalation rate, human-override rate, grounded-answer rate, cost per resolved task. **The gate** — a frozen golden set of ~120 expert-labelled cases in CI; nothing merges without passing. And I calibrate any LLM judge against human labels first — a judge with 10% false positives reports a true 85% as about 80%."*

**Extras:** 50 examples ≈ ±8 pts; a 2-pt effect needs ~865; a 1-pt gate needs ~2,828. *"Availability SLOs tell you whether the service answered, not whether the answers were good."*

---

## 8. Part 4 — coding

**Three passes, out loud:** data flow → the parameters that matter → failure handling.
**Improvement order:** correctness → reliability → cost → latency → **style last**.

**Top RAG fixes:** similarity threshold + empty-retrieval guard · hybrid + rerank · schema validation + server-side checks · timeouts/retries/idempotency · context delimiting against injection.

**The Git answer (asked verbatim):**
> *"Protected main · short-lived branches · `git pull --rebase` to stay current · **squash-merge** to land · `--force-with-lease` on my own branch only · never merge into a feature branch · never force-push main. A linear history comes from one rule: only one thing lands on main, and it lands as a squash, never a merge commit."*

**Test layers:** unit → contract → eval regression (**CI gate**) → trajectory → canary → adversarial.

---

## 9. The two panel gaps — say these lines

- **Service design:** *"The model is maybe 10% of the work. The rest is: who acts on the output, which system of record it writes to, what happens when it's wrong, how the human's day changes, and what we measure six months after go-live."*
- **Mentoring:** *"My rule is that nobody stays in notebooks. Within a quarter, every engineer has shipped something with a monitoring dashboard and an owner's name on it."*

**Standards (name three):** eval gate in CI · every service has a metric + named owner · cost budget per unit with an alert. *"A standard only counts if it has a consequence."*

---

## 10. Domain vocabulary (use these, not tech words)

**Market:** syndicate · managing agent · broker · MGA/coverholder · delegated authority · bordereau
**Flow:** **submission** → triage → quote → **bind** → endorsement → **claim**
**Business:** appetite · line/capacity · **combined ratio** · loss ratio · treaty vs facultative · cat event
**Regulatory:** FCA **Consumer Duty** · PRA **SS1/23 model risk** · Lloyd's · UK GDPR · Solvency II · sanctions
**Evidence:** model card · limitations register · decision log · **the trace is the audit evidence**

**Never say:** "ticket", "users", "customer request", "we re-ran it".

---

## 11. Delivery fixes

Pause instead of filling · headline card for 20 answers · 5 bridge statements · 3 jargon translations · ~140 wpm · headset on, notifications off.

**Bridge statements:** *"That's an interesting edge case — the first thing I'd pin down is…"* · *"I haven't hit that exact case; here's how I'd reason from first principles…"* · *"I don't have that number precisely…"* · *"Let me restate it to make sure I've understood…"*

---

## 12. Your questions to ask them

1. Where does AI sit relative to the underwriters — shared roadmap or requirements-taking?
2. Which Smarter Underwriting use case is furthest into production, and what's been hardest?
3. How is model risk signed off for AI-assisted underwriting decisions?
4. For the India GCC team — delivery capacity for the UK, or capability-building too?

---

## 13. The last 20 minutes before the call

Read §1 → §5 → §7 → §8. Say the intro twice, the evaluation answer once, the Git line once. Then **stop reading**, drink water, check the mic, and walk around.

**You know the technology. What's being bought is framing, decisions and leadership. Lead the conversation.**
