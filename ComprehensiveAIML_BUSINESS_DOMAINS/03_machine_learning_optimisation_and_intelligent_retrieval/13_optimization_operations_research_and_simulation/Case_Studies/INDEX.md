# Module 13 Case Studies — Optimisation, Operations Research and Simulation

## 1. Collection Metadata

**Scope:** bounded operational decisions involving routes, staffing, and scheduling objectives. **Editorial state:** catalog only; no case below is authored. **Evidence rule:** each eventual case must be explicitly illustrative, published, reproduced, or first-hand; a planning title is not outcome evidence. **Required structure:** the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md) require all 15 sections, 5a–5e, worked arithmetic and sensitivity, 12–18 answered interview prompts, and Section 15 last.

## 2. Case Catalog

| Case ID | Linked title if authored | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-2026-OR001 | `01_last_mile_routing_optimisation.md` (registered) | logistics | analytical | choose feasible routes with capacity/windows | constrained optimization | simulation | not-measured | proposed | null | no case artifact or measured routing result |
| CASE-2026-OR002 | `02_workforce_scheduling_under_uncertainty.md` (registered) | operations | analytical | select staffing under shortage scenarios | robust scheduling | simulation | not-measured | proposed | null | no case artifact or measured workforce result |
| CASE-2026-OR003 | `03_scheduling_objective_and_constraints.md` (registered) | scheduling | analytical | choose greedy versus weighted scheduling model | correctness-contract | none | not-measured | proposed | null | supporting IR025 teaching case; not authored |

**Editorial state for every row:** registered, not draft/reviewed/published.

## 3. Canonical Reusable Patterns

| Pattern | Decision contract | Required evidence | Disqualifier |
|---|---|---|---|
| constrained optimization | variables, objective, hard constraints, owner and fallback | feasibility replay and objective comparator | hidden constraints or no authorized action |
| simulation | event semantics, horizon, warm-up, replications and uncertainty | reproducible trace/configuration | using one stochastic run as a guarantee |
| correctness-contract | exact objective, interval semantics, tie rule and oracle | exhaustive small-instance or proof boundary | claiming weighted/multi-resource optimality from cardinality greedy |

## 4. Similarity / Contrast Map

| Comparison | Shared reasoning | Critical distinction |
|---|---|---|
| Routing vs workforce staffing | constrained allocation and scenario sensitivity | route sequence/vehicle feasibility differs from shift/labor constraints |
| Workforce staffing vs interval scheduling | coverage/resource capacity | uncertain demand and protected rules couple choices beyond one-resource intervals |
| Interval scheduling vs simulation | schedule feasibility | deterministic objective choice differs from stochastic outcome estimation |

## 5. Questions / Future-Case Backlog

A case may be authored only after it names a decision owner, action unit, baseline, constraint source or illustrative assumption, input cutoff, numerical oracle, validation plan, and failure/recovery path. Open questions include: what capacity constraint actually binds; whether missed service and labor costs are commensurable; whether durations are deterministic enough for an exact model; and which manual override is authorized when no feasible action exists.

## 6. Review / Retirement Log

| Date | Artifact | Action | Reason |
|---|---|---|---|
| 2026-11-09 | catalog | normalized to seven required index sections | preserves original rows while separating registration from authorship |
| 2026-11-09 | IR025 row | assigned stable case destination | source plan is scope only; no case completion claimed |

## 7. Pre-Publish Checks

- [x] Original routing and workforce decisions remain distinct.
- [x] IR025 scheduling decision has a stable non-duplicative case destination.
- [x] Evidence, intervention stage, and editorial state are separated.
- [ ] Every row has a 15-section case with 5a–5e and Section 15 card.
- [ ] Each case records an owner/action/unit/time/baseline and successful/failed recovery paths.
- [ ] Each quantitative conclusion has a checked arithmetic/oracle trace and sensitivity.
- [ ] Each source claim, navigation link, result, and interview pack is independently reviewed.
