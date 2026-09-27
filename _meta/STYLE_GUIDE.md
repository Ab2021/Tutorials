# Style Guide & Output Contract

Every document in this knowledge base MUST follow this contract. It exists so that
24 case studies written by different passes read as one coherent body of work.

---

## 1. Principles

1. **Source-faithful first, then opinionated.** Every claim that comes from a video
   transcript must be traceable. Use the timestamp anchors that already exist in the
   source (`[12:34]`). When *you* add judgement (pros/cons, when-to-use), mark it
   explicitly with `**Analyst note:**`.
2. **No hand-waving.** "Use LLM-as-judge for quality" is banned. Write *which* judge,
   *which* rubric, *what* threshold, *what* it costs, and *what* it gets wrong.
3. **Every technique gets the full treatment**: definition → mechanism → worked example
   → numbers → pros → cons → failure modes → exceptions → when to use / when NOT to use.
4. **Numbers are load-bearing.** Prices, token counts, latency, sample sizes, win rates,
   correlation coefficients, thresholds — copy them from the source verbatim. If the
   source is vague, say "source does not give a number".
5. **Vendor-neutral.** Name tools (RAGAS, DeepEval, Langfuse, Evalscope, promptfoo, G-Eval,
   TruLens…) but never assume the reader has one.
6. **English only.** Source transcripts are Hinglish/Banglish and were romanized; write the
   output in English, but quote key source phrases where the exact wording matters.

---

## 2. Case-study template (mandatory section order)

```markdown
# CS-NN · <Title>

> **Source transcript:** `<exact filename>` (<duration / line count>)
> **Domain:** <foundations | methods | benchmarks | rag | agentic | production>
> **One-liner:** <what this case study is about, in one sentence>
> **Prerequisites:** CS-XX, CS-YY  (or "none")

---

## 0. Executive summary
Ten bullets max. Each bullet must be a *fact a practitioner can act on*, not a topic label.

## 1. The problem this lecture solves
Why does this exist? What breaks if you ignore it? What was the pre-LLM world like?

## 2. Definitions & mental models
Table of every term the source defines: `Term | Definition (source's words) | Why it matters`.
Include the mental models/diagrams the speaker draws (e.g. "the three-box diagram").

## 3. Core content, decomposed
The bulk. One `###` per concept, in the order the source teaches it. Each subsection:

### 3.n <Concept name>  `[mm:ss]`
- **What the source says** (faithful, specific)
- **Mechanism** — how it actually works under the hood
- **Worked example** — the example the speaker used, or a better one
- **Numbers / thresholds** if any
- **Analyst note:** nuance, caveats, or corrections

Group these into a small number of `##` bands by theme; do not create 40 flat subsections.

## 4. Frameworks & decision procedures
Any checklist, triage tree, scoring rubric, or "if X then Y" procedure the source gives.
Render as a table, a Mermaid diagram, or a numbered procedure — not prose.

## 5. Worked end-to-end example
One realistic scenario carried all the way through (dataset → metric → judge → threshold →
decision). Reuse the source's own running example when it has one.

## 6. Pros, cons, exceptions
| Approach | Pros | Cons | Works when | Fails when | Cost |
Three tables if needed: one for techniques, one for tools, one for metrics.

## 7. Failure modes & anti-patterns
Numbered list. Each: *symptom → root cause → detection → fix*.

## 8. Implementation notes
Concrete: libraries, code shapes, API calls, file layouts, CI wiring. Code blocks allowed
and encouraged. Do **not** invent library APIs — if unsure, describe the shape in prose.

## 9. Interview-ready Q&A
8–12 questions an interviewer would actually ask on this topic, each with a **model answer**
of 3–8 sentences that a strong candidate would give. Include at least 2 "trap" questions
where the naive answer is wrong.

## 10. Cheat sheet
A compact, self-contained block (~40–70 lines) that could be printed on one page:
formulas, thresholds, tool commands, decision rules.

## 11. Glossary
Table: `Term | Meaning`, for every jargon term introduced.

## 12. Cross-references
- **Builds on:** CS-XX
- **Leads to:** CS-YY
- **External:** links that are *named in the source*, plus canonical references
```

**Length target:** 1,800–3,500 words. Go longer only when the source is unusually dense
(e.g. hands-on sessions). Never pad; every paragraph must carry information.

---

## 3. Cheat-sheet template

```markdown
# Cheat Sheet · <Topic>
> **Covers:** CS-XX, CS-YY · **Use when:** <trigger>

## The 60-second version
## Core concepts (table)
## Formulas & metrics (with definitions of every symbol)
## Decision rules (numbered)
## Thresholds & defaults worth memorising
## Tool commands (copy-pasteable)
## Top 10 mistakes
## If you only remember three things
```

Target 700–1,400 words. Dense, table-heavy, minimal prose.

---

## 4. Interview-question pack template

```markdown
# Interview Pack · <Domain>
> **Covers:** CS-XX..YY · **Role level:** <junior | mid | senior | staff>
> **Format:** <screen | onsite | system-design | take-home>

## How this domain is assessed
## Tier 1 — Fundamentals (10 Q)
## Tier 2 — Applied / trade-off (10 Q)
## Tier 3 — Senior / staff (8 Q)
## Tier 4 — Debug-this-scenario (5 Q)
## Tier 5 — Trap questions & the naive-answer trap (5 Q)
## Live-coding / whiteboard prompts (3)
## Take-home / case-study prompt (1 full brief)
## Scoring rubric (what separates a hire from a no-hire)
```

Each question: **Q**, then **What they're really testing**, then **Model answer** (bulleted,
3–8 points), then **Red flags** (what a weak answer sounds like).
Target 2,000–3,500 words.

---

## 5. Codebase dossier template

```markdown
# Codebase · <name>
> **Repo path:** `<...>` · **Language:** · **License:** · **Role in the stack:**

## 1. What it is / what it is not
## 2. Architecture (Mermaid diagram + module walkthrough)
## 3. The 10 files that matter (path → why → what to read in it)
## 4. Core abstractions (class/function level, with signatures)
## 5. How an evaluation actually runs (end-to-end trace through the code)
## 6. Extending it: add a new metric / dataset / model adapter (step-by-step)
## 7. Configuration & CLI surface (real flags, real config keys)
## 8. Strengths, weaknesses, and when to choose it over the alternatives
## 9. Minimal runnable example (real code you could paste)
## 10. Reading order for a newcomer (numbered, with time estimates)
```

Target 1,500–2,500 words.

---

## 6. Formatting rules

- Heading hierarchy: `#` document title only, `##` template sections, `###` concepts.
- Tables over prose for anything comparative.
- Mermaid for flows: ` ```mermaid ` fences.
- Timestamp anchors: `[mm:ss]` immediately after the claim they support.
- Cross-links: relative markdown links (`../02-methods/CS-07-...md`).
- No emoji in headings. No horizontal-rule spam.
- File names: `CS-NN-kebab-case-title.md`, cheat sheets `CHEAT-<topic>.md`,
  interview packs `IQ-<domain>.md`, codebases `CODE-<name>.md`.
