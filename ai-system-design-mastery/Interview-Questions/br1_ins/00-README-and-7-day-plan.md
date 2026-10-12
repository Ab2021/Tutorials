# BRIT INSURANCE — LEAD AI ENGINEER · STAGE 1 PREP PACK

> **Read this file first.** It is the index, the plan, and the five rules that decide whether you pass.

**Role:** Lead AI Engineer, Data — Brit Ltd (Lloyd's of London specialty insurer, Fairfax-owned)
**Line manager:** Head of India GCC Enablement
**Round:** Stage 1, 1 hour, **four parts** (Introductions → Project drill-down → System Design → Coding Exploration)
**Output of this pack:** everything you say, draw, and write in that hour, pre-decided.

---

## 1. Why this pack is different from a generic prep guide

Three facts, all from your own material, shape everything here:

1. **The round has a known shape.** The candidate feedback file says the hour is split four ways. That is not a guess — it is the format. So we prepare four distinct performances, not one.
2. **Two named gaps sank the last profile.** The panel feedback says candidates are "lacking": **(a) broader service design considerations, business integration, and working with stakeholders and teams**, and **(b) technical and people mentoring for a team of data scientists, AI engineers, or ML engineers**. Those are now *scored competencies*. Roughly a third of your prep time should go to them, because they are the cheapest points on the board — most candidates ignore them.
3. **The last failed interview was lost on delivery, not knowledge.** `prev_learning` is explicit: *"You aren't being rejected for lack of knowledge, but for a lack of 'seniority in communication.'"* Filler words, tool-listing instead of architecture, and "I can't think of it right now." Every doc in this pack therefore contains **spoken-style scripts**, not just content.

> **The thesis of this pack:** you already know the technology; what is being bought at Lead level is the ability to *frame, decide, and lead*. So the docs are weighted toward framing, decisions, trade-offs, and people.

---

## 2. The file map

| # | File | Part of the interview it serves | Read it when |
|---|---|---|---|
| 00 | **This file** — index + 7-day plan | all | now |
| 01 | **JD decode & role fit** | all — decides your positioning | day 1 |
| 02 | **Stage-1 format & 60-minute battle plan** | all — the run-sheet | day 1, re-read on the day |
| 03 | **Introduction & story bank** | Part 1 — Introductions | day 2 |
| 04 | **Project deep-dive playbook** | Part 2 — Drill-down on your agentic project | day 2–3 |
| 05 | **System design playbook** | Part 3 — System Design | day 3–4 |
| 06 | **Coding exploration playbook** | Part 4 — Coding Exploration (incl. the Git question) | day 4 |
| 07 | **Agent evaluation, guardrails & metrics** | Part 2 & 3 — "How did you evaluate your agents?" | day 3 |
| 08 | **Applied AI mock-interview prompt** | rehearsal harness (the DeepSeek prompt) | day 5 |
| 09 | **Mock interview strategies & drills** | all — how to rehearse | day 4–6 |
| 10 | **Question bank with model answers** | all — drill material | day 2–6 |
| 11 | **Insurance & Lloyd's domain primer** | all — vocabulary and business context | day 2, then daily |
| 12 | **People leadership & stakeholder round** | the panel-feedback gap | day 4–5 |
| 13 | **Final 48 hours cheat sheet** | the night before | day 6–7 |
| 14 | **Industry intel: lead-AI loops, real questions & verified Brit facts** | all — the evidence base | day 2 (calibrate), day 6 (extra drills) |

---

## 3. The five rules (from `prev_learning` — break these and nothing else in the pack matters)

| # | Rule | The failure it prevents |
|---|---|---|
| 1 | **Headline first, then Why, then How.** Open every answer with the one-sentence conclusion. *"I'd use a tiered approach: regex first, then a small model, then a frontier model for the residue."* | Wandering, theoretical answers |
| 2 | **Pause 2–3 seconds before answering.** A composed pause reads as authority; fillers read as anxiety. | "Uh", "like", fragmented sentences |
| 3 | **Never say "I can't think of it."** Use a bridge: *"That's an interesting edge case — if I were designing it from scratch, the first thing I'd pin down is X. Let me walk you through that."* | Freezing under pressure |
| 4 | **Problem → Solution → Trade-off**, every time. Never list tools; explain how they integrate to solve a named pain. | "Tool-heavy" answers |
| 5 | **Connect every metric to cost-of-error.** *"I optimise Precision at the auto-approve threshold to protect the insurer, and Recall in the human-triage queue so nothing valid is silently denied."* | Choosing the wrong metric, or none |

**Bonus rule for this role:** an LLM output is a *proposal*, not a fact. Say the words **"deterministic validation layer"** at least three times in the hour.

---

## 4. The four dials — the spine of every design answer

Name the dial you are **optimising** and the dial you are **paying** with. If you say this sentence in Part 3, you have passed the seniority check.

| Dial | Question to ask | Levers |
|---|---|---|
| **Quality** | What is the cost of one wrong answer? | reranking, bigger model, better chunking, more context |
| **Cost** | What is the per-decision budget? | model tiering, caching, token limits, smaller models |
| **Latency** | What is the P95 target? | streaming, caching, smaller models, batching |
| **Safety** | Is the output regulated? Who signs it? | grounding, validators, HITL, confidence thresholds |

---

## 5. The 7-day plan

Assumes ~2–3 focused hours a day. Day 7 is the interview.

| Day | Focus | Deliverable of the day |
|---|---|---|
| **1** | 01 + 02. Understand the role and the run-sheet. | A one-paragraph positioning statement you can say in 20 seconds |
| **2** | 03 + 11 + 14 §1–§4. Write your intro scripts; learn 25 insurance terms cold; read the verified Brit facts twice. | 60-second and 2-minute intros recorded on your phone |
| **3** | 04 + 07. Rebuild your agentic project on one page; write the evaluation narrative. | A single-page architecture of your project, drawn from memory |
| **4** | 05 + 06. Run two system designs and the Git answer out loud. | The image-anomaly design drawn twice, under 15 min each |
| **5** | 08 + 12. Run the mock (DeepSeek prompt). Write your leadership stories. | One full mock transcript + its critique |
| **6** | 09 + 10 + 13 + 14 §2 (extra drill bank). Drill the weak spots the mock exposed; memorise the cheat sheet. | Cheat sheet re-written in your own words |
| **7** | **Interview.** Re-read 02 and 13 only. Nothing new. | — |

---

## 6. How to use this pack in the last hour before the call

Read **only** these, in this order: `13` (cheat sheet) → `02` §3 (the run-sheet) → `03` §2 (your intro) → `01` §6 (your positioning). Twenty minutes. Then stop reading and walk around.

---

## 7. Source-coverage map (proof that nothing was left out)

Every theme in your seven source files is placed somewhere in the pack. If you want to check a source is fully harvested, read down its row.

| Your source | Its themes | Where they live in the pack |
|---|---|---|
| **`brit_ins/jd`** | accountabilities, technical requirements, people-leadership block, Azure/low-code stack | `01` §2 (teardown + self-audit) · `11` (stack, low-code rule) · `12` (people) |
| **`brit_ins/candidate_feedback`** | the 4-part format, the 8 asked questions, the 2 panel gaps | `01` §3–§4 · `02` §3 (design phases) · `06` §4 (Git) · `07` (agents) |
| **`prev_learning`** | delivery coaching — headlining, filler words, bridges, P/S/T, cost-of-error | `03` §4 · `09` §6 (delivery drills) · `13` §11 |
| **AI Interview — Technical Concepts Deep Dive** | LLM internals, tokens/BPE, prefill/decode, KV cache, sampling, speculative decoding, batching, streaming/SSE | `10` §A, §B, **§L** · `05` §5, §8.5 · `14` §11 |
| **AI Interview Strategies & Answering Frameworks** | interview funnel + failure modes, what interviewers score, escalation ladder, reference architecture, protocol table, RAG funnel | `05` §2–§4 (funnel, architecture) · `07` · `10` §C · `14` §1 |
| **Ai sus design course** | latency myth, streaming, retry amplification, duplicate side effects, tail latency & fan-out, state/stateless scaling, rate limiting & token costing, silent failures, I/O-bound vs CPU-bound & the GIL, incident post-mortems | `05` §3, §5 (capacity, reliability) · **`06` §5.7 (concurrency/GIL)** · `07` §6 (silent failure) |
| **questions_akhilag** | 20×5 operational principles — streaming, semantic cache, tool blast radius, loop detection, reranking, freshness, action reversibility, provenance/audit, multi-tenancy, indexing | `06` §2 (critique checklist) · `07` §8 (provenance/audit) · `10` §C–§F |
| **questions_thinkinmodels** | 30+ scenario questions with worked analyses — judge calibration, 0.98²⁰, chunk-size debates, multi-tenant security, model upgrades, capacity planning | `10` (scenario answers) · `07` §4–§9 (eval, judge) · `05` §6 |
| **`zip_reference_file/nitmonk/archive/`** | *(excluded by your instruction — intentionally not read)* | — |
| **Web research (new, 2026-10-12)** | lead/staff loops, real questions, verified Brit facts, regulation, vendors, serving/quant numbers | `14` (all), folded into `05`, `06`, `07`, `11` |

**Two genuinely new gaps closed on 2026-10-12** (both were source-named and previously uncovered):
- **Python concurrency — I/O-bound vs CPU-bound and the GIL** → now `06` §5.7. This is Brit's exact split: chat/agent is I/O-bound, imagery inference is CPU-bound.
- **LLM internals depth — BPE, sampling parameters, attention, RoPE** → now `10` §L.

---

## 8. Cross-links

- Repo-wide conventions and the 13-section content template: [`../00-meta/01-how-to-use-and-template.md`](../../00-meta/01-how-to-use-and-template.md)
- The general HLD round this pack specialises: [`../HLD/01-foundations/`](../../HLD/01-foundations/)
- AI-systems deep dives (RAG, agents, serving, guardrails, evals): [`../HLD/05-ai-systems/`](../../HLD/05-ai-systems/)
- The operations playbooks (LLMOps, observability, FinOps): [`../Operations/`](../../Operations/)
- The general question banks (8 of them): [`../`](./)
