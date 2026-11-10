# Module 02 â€” Python, SQL, Linux and Engineering Basics — Cheat Sheets

## 1. Overview

This collection carries 6 reference card(s) and 1 troubleshooting artifact(s) for `Module 02 â€” Python, SQL, Linux and Engineering Basics`. Cards teach a reusable method, calculation, or selection rule; the Common Issues file turns those rules into diagnostics. Status is measured, not asserted: a card is `authored` only when `Tools/manuscript_lint.py` exits 0 for it.

## 2. Topic ladder

| Level | Cards | Purpose |
|---|---|---|
| core | SQL Window Frames and Aggregate Semantics, 25 Common Issues — diagnostic catalog for async Python, HTTP/WebSocket I/O, LLM inference, containers, and SQL analytics | Build from prerequisites to production judgement |

## 3. Quick lookup

| If you need to… | Read |
|---|---|
| work on python async and concurrency | [Python Async and Concurrency](./01_Python_Async_and_Concurrency.md) |
| work on rest apis and websockets | [REST APIs and WebSockets](./02_REST_APIs_and_WebSockets.md) |
| work on llm fundamentals | [LLM Fundamentals](./03_LLM_Fundamentals.md) |
| work on docker and containers | [Docker and Containers](./04_Docker_and_Containers.md) |
| work on sql analytics patterns: grain, joins, windows, and metric defence | [SQL Analytics Patterns: Grain, Joins, Windows, and Metric Defence](./05_SQL_Analytics_Patterns.md) |
| work on sql window frames and aggregate semantics | [SQL Window Frames and Aggregate Semantics](./06_SQL_Window_Frames_and_Aggregate_Semantics.md) |
| diagnose a failure that keeps recurring | [25 Common Issues — diagnostic catalog for async Python, HTTP/WebSocket I/O, LLM inference, containers, and SQL analytics](./99_25_Common_Issues.md) |

## 4. Reading paths

| Path | Sequence |
|---|---|
| Beginner | foundation cards first, then the Quick Reference Card (§12) of each core card |
| Practitioner | core cards in ladder order, then one case in `../Case_Studies/` that cites them |
| Advanced | advanced cards, then the Common Issues artifact and its discriminating checks |

## 5. Prerequisite skills

| Skill | Why it is needed | Where it is available |
|---|---|---|
| Unit topic map | Defines what this collection is responsible for and what it excludes | `../INDEX.md` |
| Decision framing | Cards assume a stated decision, population, and cut-off | sibling case collection |
| Arithmetic discipline | Every card carries checked numbers with units | the cards themselves |

## 6. Exclusions and handoffs

| Excluded here | Handed to |
|---|---|
| Decision-context narrative, baselines, and outcomes | `../Case_Studies/` |
| Organisational policy and legal determination | governance modules, not this collection |
| Vendor-specific or unverified current API behaviour | treat as OPEN until sourced at authoring time |

## 7. Master symbol glossary

| Symbol / term | Meaning | Units | Shape / type | Domain | First use |
|---|---|---|---|---|---|
| — | no card defines a symbol yet | — | — | — | — |

## 8. Card directory

| # | Card | Level | Evidence | Status |
|---|---|---|---|---|
| 1 | [Python Async and Concurrency](./01_Python_Async_and_Concurrency.md) | 2 | DERIVED | authored |
| 2 | [REST APIs and WebSockets](./02_REST_APIs_and_WebSockets.md) | 2 | DERIVED | authored |
| 3 | [LLM Fundamentals](./03_LLM_Fundamentals.md) | 2 | DERIVED | authored |
| 4 | [Docker and Containers](./04_Docker_and_Containers.md) | 2 | DERIVED | authored |
| 5 | [SQL Analytics Patterns: Grain, Joins, Windows, and Metric Defence](./05_SQL_Analytics_Patterns.md) | 1 | DERIVED | authored |
| 6 | [SQL Window Frames and Aggregate Semantics](./06_SQL_Window_Frames_and_Aggregate_Semantics.md) | core | DERIVED | authored |
| 7 | [25 Common Issues — diagnostic catalog for async Python, HTTP/WebSocket I/O, LLM inference, containers, and SQL analytics](./99_25_Common_Issues.md) | core | ILLUSTRATIVE | authored |

## 9. Common Issues locator

- [25 Common Issues — diagnostic catalog for async Python, HTTP/WebSocket I/O, LLM inference, containers, and SQL analytics](./99_25_Common_Issues.md) — the unit's final troubleshooting artifact; entries use stable `CI-` ids.

## 10. Sources, versions, maintenance

| Field | Value |
|---|---|
| Evidence vocabulary | DERIVED / VERIFIED / SOURCED / ILLUSTRATIVE / HEURISTIC / OPEN |
| Version policy | Content is dated by its own frontmatter; unstable API, price, and regulatory facts are rechecked at edit time |
| Maintenance trigger | A card's validity condition, a source body, or a tool's semantics changes |
| Review cadence | Quarterly, plus on any material change to a cited source |

## 11. Acceptance checklist

- [x] Every linked card exists on disk.
- [x] Status column is measured by the lint, not asserted.
- [x] Planned commitments remain listed without links.
- [x] A final Common Issues artifact exists with at least 25 distinct issues.
- [ ] Independent technical review of each card.
