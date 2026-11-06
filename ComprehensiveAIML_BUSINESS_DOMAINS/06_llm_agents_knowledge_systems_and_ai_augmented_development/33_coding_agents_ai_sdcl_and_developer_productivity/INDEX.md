# Module 33 â€” Coding Agents, AI SDLC and Developer Productivity

> **Part:** [Part 06 â€” LLM Agents, Knowledge Systems and AI-Augmented Development](../INDEX.md)  
> **Prerequisites:** 02,03,04,30,31  
> **Purpose:** Use AI to plan, write, test, review, and operate software while retaining verification and engineering discipline.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | repository context | planned |
| 2 | agentic coding workflows | planned |
| 3 | test generation and review | planned |
| 4 | CI/CD integration | planned |
| 5 | developer metrics and risks | planned |

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
| PLAN-2026-IR010 | Fill-in-the-middle; repository/time splits; authorized code retrieval; multilingual sampling; language adapters | When do repository-aware completion or language adapters justify their added complexity? | Sandboxed functional correctness and latency for code; native-reviewed grounding and worst-language regressions for support |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Source-specific Claude Code harness philosophy, Go-native engine, five-verb skills, PLANS.md, parallelism and integration](../../00_SOURCE_COVERAGE_AGENTS.md#agent-claude-harness) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [CLAUDE.md hierarchy, commands/skills, path rules, plan versus direct mode, iterative refinement, CI/CD and practice](../../00_SOURCE_COVERAGE_AGENTS.md#claude-configuration) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Support, code generation, multi-agent research, developer productivity, CI/CD and structured extraction scenarios](../../00_SOURCE_COVERAGE_AGENTS.md#claude-scenarios) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Hands-on multi-tool agent, code configuration, extraction and multi-agent pipeline exercises](../../00_SOURCE_COVERAGE_AGENTS.md#claude-practice) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
