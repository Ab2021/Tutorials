# Module 34 â€” Conversational AI, Voice Agents and Customer Operations

> **Part:** [Part 06 â€” LLM Agents, Knowledge Systems and AI-Augmented Development](../INDEX.md)  
> **Prerequisites:** 18,20,30,31  
> **Purpose:** Design reliable, measurable customer-facing conversations with latency control, escalation, CRM integration, and QA.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | dialogue design | planned |
| 2 | real-time voice stack | planned |
| 3 | handoff to humans | planned |
| 4 | conversation analytics | planned |
| 5 | compliance and quality | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR015 | Product/date eligibility; grounded generation; native-language review; source dependencies; supersession triggers | Which generated scripts are fit for an approved, versioned coaching registry? | Unsupported product claims, expired evidence, read-aloud time, reviewer agreement and agent usability |
| PLAN-2026-IR016 | Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation | Which customer objections can be reported with reliable speaker attribution and evidence? | Human-gold WER, domain-term WER, role error, objection precision/recall and end-to-end report coverage |
| PLAN-2026-IR017 | Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment | Should a speech adapter be promoted, and to which evaluated product and language domains? | Paired WER with uncertainty by domain and code-switch slice, real-time factor and supported-route regression gates |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Streaming, real-time output and interruption](../../00_SOURCE_COVERAGE_AGENTS.md#lang-streaming) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
