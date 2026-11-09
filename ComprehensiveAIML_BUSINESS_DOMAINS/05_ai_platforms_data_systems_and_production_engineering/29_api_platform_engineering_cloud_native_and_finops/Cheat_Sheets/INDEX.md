# Module 29 Cheat Sheets — API Platform Engineering, Cloud-Native Systems and FinOps

**Status:** planned collection; no individual sheet exists. Use the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [cheat-sheet master](../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). A completed sheet has 12 sections, >=1,000 substantive lines, checked arithmetic/trace, four diagnostics, answered interview pack, and final card.

## 1. Overview

This collection teaches platform engineers to make API, orchestration, tenancy, platform-product, and spend decisions with explicit contracts and bounded failure behavior. Readiness means stating ownership and units, choosing a disqualifiable alternative, and proving the relevant invariant or capacity calculation.

## 2. Topic ladder

| Level | File / planned topic | Testable outcome | Prerequisite | Status |
|---|---|---|---|---|
| Foundations | API design and contracts | define a compatible request, response, error, and idempotency contract | HTTP, schemas | planned |
| Core | Kubernetes and serverless AI workloads | select a deployment/control boundary | containers, scheduling | planned |
| Core | Multi-tenancy and platform product | isolate data, state, quota, and ownership | identity, authorization | planned |
| Advanced | FinOps and cost allocation | reconcile successful-work cost and budget | units, accounting | planned |
| Advanced | Global admission / token leases (PLAN-2026-IR022) | trace atomic debit, expiry, and recovery | concurrency | planned |
| Final | Common Issues | diagnose 25+ distinct issues | all above | not written |

## 3. Quick topic lookup

| Question | Destination | Use condition / failure warning |
|---|---|---|
| Can a retry create a duplicate side effect? | API contract sheet, Section 4 | A request ID without an ownership and retention rule is insufficient. |
| Which workload fits serverless versus long-lived serving? | workload patterns, Section 7 | Cold start, state, accelerator placement, and tail deadline can disqualify serverless. |
| How do tenants avoid cross-charge and cross-data access? | tenancy sheet, Section 8 | Namespace isolation is not automatically data-plane authorization. |
| Why did a budget permit too much traffic? | token lease sheet, Section 4 | Periodic usage reporting is not globally atomic admission. |

## 4. Reading paths — three options

- **Beginner:** API design and contracts → Kubernetes/serverless patterns → tenancy concepts; assumes HTTP and container basics.
- **Practitioner:** API design → workload patterns → tenancy/platform product → FinOps allocation; practice a failed retry and budget reconciliation.
- **Advanced:** all files → token leases → Common Issues; add fault injection and compatibility/recovery traces.

## 5. Prerequisite skills

Readers need HTTP status/error semantics, schema evolution, identity/authorization vocabulary, container scheduling concepts, rates versus counts, basic concurrency, and currency/unit arithmetic. Provider-specific API or pricing behavior must be checked from current primary documentation before implementation.

## 6. What this collection does NOT cover

It does not certify a provider's current price/capability, replace organization-specific security review, define every orchestration framework, or establish that a quota system is legally/commercially sufficient. Model inference algorithms belong in Module 25; release bundles belong in Module 26.

## 7. Master symbol glossary

| Symbol / term | Meaning | Units / type | Domain / warning | First planned use |
|---|---|---|---|---|
| `r` | request arrival rate | requests/s | interval average, not a burst guarantee | workload patterns |
| `b` | budget tokens | integer credits | must name global versus local scope | token leases |
| `c_s` | successful-work cost | currency/success | exclude/reveal failed work explicitly | FinOps |
| idempotency key | client operation identity | opaque string | not an authorization credential | API contracts |
| lease | temporary budget claim | token count + expiry | clock/failure policy required | token leases |

## 8. Quick Reference Card directory

| Planned card | Core rule | Use condition | Avoid when |
|---|---|---|---|
| API contract | retry only with an idempotency and response-replay contract | side effects are externally visible | identity/retention is unknown |
| Workload placement | choose a control boundary from state, deadline, and placement constraints | workload is measured | only average utilization is known |
| Tenancy | authorize every data/state boundary independently | tenant identity is propagated | namespace is used as the sole control |
| FinOps | allocate cost to completed useful work | meter is reconciled | price is stale or failed work hidden |
| Token leases | debit/refill under one global invariant or bounded failure policy | coordination semantics are explicit | periodic reports substitute for admission |

## 9. Common Issues locator

`06_25_Common_Issues.md` is required as the final artifact but **does not yet exist**. The collection claims zero completed issues; it must contain at least 25 distinct complete `VIOLATED_ASSUMPTION`, `NUMERICAL_EDGE`, `SCALE_FAILURE`, or `CONCEPT_CONFUSION` entries before the filename can claim 25.

## 10. Sources, versions and collection maintenance

| Source ID | Primary source / role | Version/date | Limitation / trigger |
|---|---|---|---|
| SRC-M29-01 | [RFC 9110 HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110) — request/response semantics | 2022 | does not specify an application's idempotency design |
| SRC-M29-02 | [Kubernetes documentation](https://kubernetes.io/docs/) — workload concepts | version must be pinned when authoring | behavior/configuration differs by release/provider |
| PLAN-2026-IR002 | reference-informed routing brief | 2026-11-06 planning | revalidate provider capability/economics |
| PLAN-2026-IR022 | reference-informed token lease brief | 2026-11-06 planning | source is not proof of a deployed global limiter |

## 11. Collection acceptance checklist

- [ ] Every planned topic has a real 12-section, >=1,000-line sheet with checked topic-specific calculation or trace.
- [ ] Sheet paths, prerequisites, glossary, cards, and reading paths link to real artifacts.
- [ ] Common Issues has 25 distinct complete entries and is final in the numbered sequence.
- [ ] Current provider/API/cost claims are primary-sourced, dated, and scoped; planned evidence remains planned.
