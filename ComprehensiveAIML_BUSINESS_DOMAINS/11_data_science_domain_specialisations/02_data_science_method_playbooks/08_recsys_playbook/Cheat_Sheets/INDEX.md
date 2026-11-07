# Recommendation Systems Playbook - Cheat Sheets

## 1. One-paragraph overview

This collection supports recommendation engineers, applied scientists and product analysts choosing retrieval, ranking and slate controls. Readers need set arithmetic, vectors, timestamp semantics and basic experiments. The first draft supports matched eligibility-aware candidate comparisons and production diagnosis; readiness requires review, evidence and the unfinished sheets and issue catalog, not merely reading a card.

## 2. Topic ladder

| Level | File / topic | Outcome and prerequisite | Skill / example / failure theme | Status |
|---|---|---|---|---|
| Foundations to production | [01 Candidate generation, eligibility and ranking](01_Candidate_Generation_Eligibility_and_Ranking.md) | Compare eligible retrieval systems; prerequisites are sets, vectors and experiments | Eligible recall, temporal trace, cold start and rights; interview skills: matched comparison and diagnosis | draft, unreviewed |
| Core methods | Ranking metric card | Select a ranking metric given the first sheet | Derive position-sensitive metrics, compare a worked slate, diagnose rank and label mismatch | planned |
| Advanced | Exploration and diversity controls | Select exposure and diversity controls after retrieval and ranking basics | Work through controlled exposure, compare alternatives, diagnose bias and guardrail violations | planned |
| Final | Common Issues artifact | Diagnose at least 25 distinct fully specified failures | Cross-system issue records, mitigations and verification | planned; zero completed issues |

The former planned candidate-generation guide is represented by the draft first file; its draft status is not a publication claim.

## 3. Quick topic lookup

| Question | File and section | Use condition | Warning |
|---|---|---|---|
| What is the authorized universe? | [01 §3](01_Candidate_Generation_Eligibility_and_Ranking.md#3-foundational-definitions) | As-of rights and availability known | Do not substitute today's catalog. |
| How is recall computed? | [01 §4](01_Candidate_Generation_Eligibility_and_Ranking.md#4-core-formulas-with-full-derivation) | Matched nonempty relevance sets | Unobserved is not irrelevant. |
| What went wrong? | [01 §8](01_Candidate_Generation_Eligibility_and_Ranking.md#8-failure-modes--edge-cases) | Investigating hypothetical failure patterns | Scenarios are not incident evidence. |
| How big is vector payload? | [01 §9](01_Candidate_Generation_Eligibility_and_Ranking.md#9-complexity--cost-analysis) | Assumptions match sizing model | Payload excludes overhead. |

## 4. Reading paths — three options

- **Beginner:** Read 01 sections 1–5, then 6–8; objective is to compute eligible recall and understand a time boundary. Prerequisites: sets and elementary arithmetic. Follow the ranking-metric card when authored.
- **Practitioner:** Read all of 01, emphasizing sections 7–9 and the card; objective is to select a safe retrieval pipeline. Prerequisites: vectors, logs and experiments. Extend with the planned ranking and exploration sheets.
- **Advanced:** Read all of 01, including the assumptions, complexity and interview cases; objective is to defend matched offline and online evaluations. Extend to every planned sheet and final issues file as authored.

The paths honestly use the sole available draft rather than linking nonexistent files.

## 5. Prerequisite skills

- Mathematics: set intersection, ratios, vectors, dot products and simple complexity; [Stanford Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) is an accessible primary textbook.
- Domain: catalog rights, availability, user exposure, implicit outcomes and the meaning of a business guardrail.
- Systems: timestamps, point-in-time joins, index versions, cache invalidation and access-controlled event logging.
- Experimentation: assignment, population, outcome maturity and uncertainty; [Online Controlled Experiments](https://doi.org/10.1017/9781108653985) is a primary reference.

## 6. What this collection does NOT cover

- Detailed ranking-metric derivations and slate metrics are planned, not completed.
- Bandit algorithms and a full exploration policy are planned, not completed.
- Legal advice, rights contracts, personally identifying data policy and deployment-specific infrastructure require responsible domain owners.
- Crosswalk titles do not establish the contents of unavailable original source files.

## 7. Master symbol glossary

| Term | Meaning | Units / shape | Domain | First use |
|---|---|---|---|---|
| q | A request and its context | event | One timestamped opportunity | [01 §3](01_Candidate_Generation_Eligibility_and_Ranking.md#3-foundational-definitions) |
| I(t) | As-of catalog | item-ID set | At timestamp t | [01 §3](01_Candidate_Generation_Eligibility_and_Ranking.md#3-foundational-definitions) |
| E(q,t) | Authorized eligible catalog | item-ID set | Subset of I(t) | [01 §3](01_Candidate_Generation_Eligibility_and_Ranking.md#3-foundational-definitions) |
| R(q,t,H) | Observed relevant eligible items for horizon H | item-ID set | Subset of E; nonempty for per-request recall | [01 §3](01_Candidate_Generation_Eligibility_and_Ranking.md#3-foundational-definitions) |
| C_k | Candidate top-k | distinct item-ID set | At most k eligible IDs | [01 §4](01_Candidate_Generation_Eligibility_and_Ranking.md#4-core-formulas-with-full-derivation) |
| X_k, A_k | Exact and approximate scored top-k | item-ID sets | Same score and eligible universe | [01 §4](01_Candidate_Generation_Eligibility_and_Ranking.md#4-core-formulas-with-full-derivation) |
| N, d, k | Eligible count, vector dimension, cutoff | nonnegative integer counts | Defined workload; d positive for vectors | [01 §9](01_Candidate_Generation_Eligibility_and_Ranking.md#9-complexity--cost-analysis) |

The term recall is overloaded: ANN oracle overlap differs from labeled relevance recall.

## 8. Quick Reference Card directory

| Card | Core rule | Use condition | Critical assumption / avoid |
|---|---|---|---|
| [01 card](01_Candidate_Generation_Eligibility_and_Ranking.md#12-quick-reference-card) | Eligible relevant hits divided by eligible relevant count | Matched request and label population | Nonempty relevant set and as-of authorized catalog; avoid unmatched or immature comparisons. |

## 9. Common Issues locator

Final Common Issues file: **planned, not created**. Actual completed collection-level issue count: **0**. The hypothetical failure exercises in [01 §8](01_Candidate_Generation_Eligibility_and_Ranking.md#8-failure-modes--edge-cases) provide local diagnostics, not completed entries in the planned artifact. No issue IDs or issue anchors are asserted for an unwritten file.

## 10. Empirical sources, versions and collection maintenance

| File / status | Claim and primary source | Version / review | Limitation and trigger |
|---|---|---|---|
| 01 / draft | Retrieval and ranking context: [Covington et al.](https://doi.org/10.1145/2959100.2959190) | Historical paper; no substantive review | Not evidence of a present production architecture; revisit source interpretation and index changes. |
| 01 / draft | Implicit interactions: [Hu et al.](https://doi.org/10.1109/ICDM.2008.22) | Historical paper; no substantive review | A model for implicit data, not proof all nonactions are negatives. |
| 01 / draft | Position-biased feedback: [Joachims et al.](https://doi.org/10.1145/3018661.3018699) | Historical paper; no substantive review | Correction assumptions and support must be checked. |
| 01 / draft | Similarity search: [Faiss](https://faiss.ai/) | Version not pinned; no substantive review | Index-specific behavior and performance remain unverified. |

Numerical examples are illustrative rather than deployment results; actual benchmark, workload, cost and outcome evidence are not measured. Re-review when permissions, catalog behavior, model or index versions, workload or source claims change.

## 11. Collection acceptance checklist

- [x] One canonical INDEX and the existing planned topic names are retained.
- [x] Available sheet is marked draft, with an actual link and a card directory.
- [x] Local glossary, three reading paths and lookup identify the available file.
- [ ] Draft receives substantive technical and editorial review.
- [ ] Ranking and exploration planned files are authored and reviewed.
- [ ] Final Common Issues artifact contains at least 25 distinct complete issues.
- [ ] Empirical workload, versioned implementation, measurements and applicable governance are independently reviewed before production use.
