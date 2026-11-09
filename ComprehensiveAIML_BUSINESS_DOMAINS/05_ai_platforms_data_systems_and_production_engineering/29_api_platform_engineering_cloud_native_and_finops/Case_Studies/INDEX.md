# Module 29 Case Studies — API Platform Engineering, Cloud-Native Systems and FinOps

**Status:** planning catalog; no authored `CASE-*` files. Future cases must follow the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md).

## 1. Collection metadata

| Field | Value |
|---|---|
| Domain | API contracts, cloud-native operations, tenancy, platform product, and FinOps |
| Evidence boundary | illustrative/proposed; prices, capabilities, and tenancy claims require dated primary evidence |
| Review trigger | API version, tenancy policy, control plane, capacity, chargeback rule, or provider contract change |

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-29-01 | Internal AI platform design | platform | hybrid | select boundaries and product contract | authorization-scope | resource-admission | not-measured | proposed | — | needs customer/workload discovery |
| PLAN-29-02 | Cloud-cost anomaly response | FinOps | incident | contain spend safely | cost-latency-tradeoff | resource-admission | not-measured | proposed | — | needs billing/reconciliation evidence |
| PLAN-2026-IR002 | Select model routes within hard customer constraints | platform | llm-agentic | choose eligible fallback | eligibility-before-ranking | cost-latency-tradeoff | not-measured | proposed | 2026-11-06 (planning) | provider facts need revalidation |
| PLAN-2026-IR022 | Bound global API admission with token leases | platform | hybrid | keep within global budget | resource-admission | correctness-contract; bounded-work | not-measured | proposed | 2026-11-06 (planning) | token bucket differs from exact rolling window |

## 3. Canonical reusable patterns

- **Authorization scope:** bind identity, tenant, capabilities, data, cache, and billing ownership at the API boundary.
- **Resource admission:** debit a globally bounded budget atomically or use a clearly bounded lease/failure policy.
- **Cost–latency tradeoff:** allocate spend using successful work, tail risk, and service objectives rather than request count alone.

## 4. Similarity / contrast map

| Cluster | Shared reasoning | Critical distinction |
|---|---|---|
| platform and routing | eligibility, policy, and observable contracts | provider health is not tenancy authorization |
| FinOps and rate limiting | bounded admission and accounting | periodic reporting is not coordinated allocation |

## 5. Questions and future-case backlog

1. What API contract remains compatible across model and policy upgrades?
2. How are tenant isolation and charge ownership tested under retries?
3. Which lease failure mode may over-admit or over-reject work?
4. What cost metric excludes failed or abandoned work, and why?

## 6. Review / retirement log

| Date | Item | Decision | Reason |
|---|---|---|---|
| 2026-11-09 | catalog | retained as planned | no authored case, billing data, or primary provider evidence |

## 7. Pre-publish checks

- [ ] Link a 15-section `CASE-*` with tenancy, admission, recovery, capacity, and cost arithmetic.
- [ ] Reconcile catalog evidence/stage/status; verify dated provider and pricing claims before publication.
