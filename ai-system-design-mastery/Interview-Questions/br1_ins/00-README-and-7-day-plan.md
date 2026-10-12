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

### 2b. The deep-dive library — harvested from your own source folders (2026-10-12)

Files **`15`–`26`** are the long-form reference shelf, preserved **in full** so that none of that work is lost. Files `00`–`14` are the curated, interview-shaped pack — **read those first**; come here when you want the substrate underneath an answer, or when the panel drills twice past your opening answer.

They come from two harvests, both on **2026-10-12**:
- **`15`–`24`** — the earlier `br_ins/` research set, harvested in full.
- **`25`–`26`** — production-architecture and caching material from `_extracted_txt/`.

Three of the `br_ins/` files carry a **correction banner** (marked ⚠️) because they predate the fact-check now recorded in `11` §7.1 and `14`. Their *frameworks* are sound; their *Brit-specific facts* are not. The authoritative sources for anything you assert about Brit are **`11` §3–§7** and **`14`** — that rule applies to every file in this table.

| # | File | Companion it deepens | Read it when |
|---|---|---|---|
| 15 | **RAG architecture deep dive** | `05` · `10` §L | you want the production code, chunking strategies, HNSW, RRF, reranking |
| 16 | **Agentic AI systems deep dive** | `07` | you want tool schemas, side-effect classes, loop detection, checkpointing, LangGraph |
| 17 | **Azure AI services deep dive** ⚠️ | `14` (§Azure) · `05` | you want Azure component-by-component depth — **§8 rewritten** |
| 18 | **System design exercise bank** | `05` | rehearsal: three worked exercises, incl. **the actual question** |
| 19 | **Coding exploration deep dive** | `06` | you want the 10 RAG improvements with before/after code + the full Git answer |
| 20 | **People leadership deep dive** | `12` | you want 6 complete STAR stories and 25 model answers |
| 21 | **Mock interview sessions** ⚠️ | `08` · `09` | you want two fully scripted mocks plus the 8-row rubric |
| 22 | **Quick reference cheat sheet** | `13` | day-of density: frameworks, RAG/agentic one-pagers, Azure table |
| 23 | **Brit JD analysis deep dive** ⚠️ | `01` · `11` · `14` | the delivery-model material: RACI, discovery workshop, SS1/23, 30-60-90 |
| 24 | **Stage-1 strategy deep dive** | `02` | minute-by-minute detail, STARLA, the dial reconciliation |
| 25 | **Production AI system design blueprint** | `05` · `18` | the whole production architecture — 11 building blocks, 4 planes, 13-item pre-launch checklist, 14 scenario Qs |
| 26 | **Caching layers deep dive** | `15` · `22` | KV vs prefix vs semantic cache: thresholds, cache keys, staleness, privacy, poison amplification |

> **⚠️ legend:** the file leads with a banner listing exactly which claims were unverified or wrong and what replaced them. Read the banner before quoting the file.

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

> **One vocabulary note.** The harvested deep-dives (`15`–`26`) name the same four dials as **Accuracy · Latency · Cost · Reliability**. They are the *same four* — only the fourth is named from a different angle. The reconciliation table lives in `24` § banner. **The line to use in the room:** *"quality, latency, cost — and the fourth dial is reliability; in a Lloyd's-regulated context that means safety and auditability, not just uptime."*

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

**Optional depth, if a day finishes early:** `15`–`26` (the harvested deep-dive library, §2b). Use them as a *reference shelf* — pull one when the mock exposes a thin spot, not as reading to get through. Days 3–5 benefit most: `15` for RAG depth, `16` for agents, `19` for the coding round, `20` for the leadership round, **`25` for the system-design round** and **`26` for any cost/latency question**.

---

## 6. How to use this pack in the last hour before the call

Read **only** these, in this order: `13` (cheat sheet) → `02` §3 (the run-sheet) → `03` §2 (your intro) → `01` §6 (your positioning). Twenty minutes. Then stop reading and walk around.

---

## 7. Source-coverage map (proof that nothing was left out)

Every theme in your source files is placed somewhere in the pack. If you want to check a source is fully harvested, read down its row.

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
| **`caching`** (new, 2026-10-12) | KV cache vs prefix/prompt cache vs semantic cache; similarity thresholds; cache-key composition; staleness & invalidation; privacy & tenant isolation; poison amplification; separate measurement | **`26`** (all) · `15` · `22` |
| **`aisyses des`** + **`ai-system-design-tutorial.zip`** (new, 2026-10-12) | the 11 production building blocks, back-of-envelope estimation, HLD vs LLD, the 4-plane architecture, pre-launch checklist, 14 scenario Qs + 6 quick-fire, minimum sufficient intelligence | **`25`** (all) · `05` · `18` |

**Two genuinely new gaps closed on 2026-10-12** (both were source-named and previously uncovered):
- **Python concurrency — I/O-bound vs CPU-bound and the GIL** → now `06` §5.7. This is Brit's exact split: chat/agent is I/O-bound, imagery inference is CPU-bound.
- **LLM internals depth — BPE, sampling parameters, attention, RoPE** → now `10` §L.

### The `br_ins/` harvest (2026-10-12)

The earlier `br_ins/` research set was treated as a **source to be harvested in full**, not as a superseded draft. All ten of its files now live in this pack as `15`–`24`. Nothing was dropped; where a claim conflicted with the verified research in `11`/`14`, the **claim** was corrected rather than the file discarded.

| `br_ins/` file | Now at | Correction applied |
|---|---|---|
| `00_INDEX.md` | *(superseded by this README)* | — |
| `01_company_jd_analysis.md` | **`23`** | ⚠️ banner + inline: Cytora (unverified), Nearmap/Maxar → **GIC**, Aon/Marsh/WTW (indicative only), GCC **Pune → Bengaluru**, FCA **PS22/3 → PS22/9**. Includes §10 Blueprint Two |
| `02_interview_structure_strategy.md` | **`24`** | four-dial reconciliation added; includes the full **§9–§15** appendix (rehearsal drills, crisis-pivot scripts, presence checklist, **scoring rubric**, **40 fire drills**, pre-interview checklist, post-interview debrief) |
| `03_rag_architecture_mastery.md` | **`15`** | — |
| `04_agentic_ai_systems.md` | **`16`** | role line: Senior/Lead → Lead |
| `05_azure_ai_services.md` | **`17`** | ⚠️ §8 rewritten — the "Brit uses Cytora" claim removed |
| `06_system_design_exercises.md` | **`18`** | — |
| `07_coding_exploration_prep.md` | **`19`** | — |
| `08_people_leadership_behavioral.md` | **`20`** | — |
| `09_mock_interview_sessions.md` | **`21`** | ⚠️ note added on the Ki/syndicate framing inside the scripted answers |
| `10_quick_reference_cheatsheet.md` | **`22`** | ⚠️ Brit-facts table corrected (Cytora, Ki status); four-dial pointer |

### The `_extracted_txt/` additions (2026-10-12)

Three further sources were handed over from `extracted/questions_rough/_extracted_txt/`. **They became two documents, not three**, because two of them are the same video: a transcript-derived set of notes and the video's written companion guide, in the same order, with the same eleven building blocks. Merging them avoided a near-duplicate file; every transcript-only detail was folded into `25` instead of dropped.

| Source | Now at | Notes |
|---|---|---|
| `aisyses des` **+** `ai-system-design-tutorial.zip` → `AI_System_Design_Guide_Codebasics.pdf` | **`25`** | The transcript's timestamps map one-for-one onto the PDF's blocks (05:58 auth → 09:34 gateway → 10:53 guardrails → 13:59 cache → 17:15 freshness → 23:15 router → 30:03 fallback → 35:14 observability; 27:18 minimum sufficient intelligence). Transcript-only specifics folded in: Keycloak/OIDC/SSO, Cloudflare/Nginx, LiteLLM, Datadog/MLflow, XGBoost/Random Forest, the ~200× cost gap, the GPT-4o-mini/Qwen/Mistral small tier. `BlipKart` is kept but explicitly framed as the source's fictional teaching example, **not** a stand-in for Brit. |
| `caching` | **`26`** | KV / prefix / semantic cache. The semantic-cache **risk** case is the heart of it, not the mechanics. |

> **One known omission, deliberate.** The transcript names a small classifier model as "JAV" — almost certainly a speech-to-text artefact. `25` flags it as such and tells you to say **"the small-model tier"** in the room rather than repeating a name you cannot verify.

---

## 8. Cross-links

- Repo-wide conventions and the 13-section content template: [`../00-meta/01-how-to-use-and-template.md`](../../00-meta/01-how-to-use-and-template.md)
- The general HLD round this pack specialises: [`../HLD/01-foundations/`](../../HLD/01-foundations/)
- AI-systems deep dives (RAG, agents, serving, guardrails, evals): [`../HLD/05-ai-systems/`](../../HLD/05-ai-systems/)
- The operations playbooks (LLMOps, observability, FinOps): [`../Operations/`](../../Operations/)
- The general question banks (8 of them): [`../`](./)
