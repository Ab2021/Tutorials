---
title: "Field Test Checklist"
topic: "Bounded field-testing of store interventions before rollout"
collection: "retail-01"
file_id: "03"
level: "core"
evidence: "DERIVED"
created: "2026-11-09"
last_reviewed: null
content_status: "draft"
version: "1.1"
notation_style: "plain-text"
prerequisites: ["basic rates, timestamps, and cohort definitions", "one confidence interval for a difference of means"]
pattern_tags: ["retail-operations", "point-in-time", "field-experiment", "rollout-gate"]
source_status: "OPEN"
owner_role: "retail analytics practitioner"
source_index: "INDEX.md"
---

# Field Test Checklist

> **Evidence disclaimer.** Every number in this sheet is derived arithmetic from stated inputs. Inputs that describe a plausible retailer rather than a sourced one are labelled `ILLUSTRATIVE`. No company deployment, vendor result, or measured financial outcome is claimed. Where a claim depends on a published method (design effect, complier-average effect, alpha spending) the method is named and the assumption is stated; the applied magnitude is still `ILLUSTRATIVE`.

## 1. File Header

**What this sheet decides.** A field test is a bounded trial of a changed way of working, run in a small number of real stores, whose only purpose is to justify *ship*, *iterate and retest*, or *stop*. This sheet is the checklist for that decision. It is written for the person who signs the rollout note: a store-operations lead, a merchandising manager, or an analytics partner who has to defend the number in a room where labour and capital are allocated.

**The compressed statement of the problem.** A field test earns trust only when it fixes the comparison, the exposure, the window, and the stopping rule *before* the outcome data arrives. Every failure mode in Section 8 is a violation of one of those four fixings.

### 1.1 The five conditions the checklist enforces

A test is admissible only if all five hold. They are conjunctive, not additive: a strong result on price of hardware never compensates for a contaminated control arm.

| # | Condition | Question it answers | Typical failing artefact |
|---|---|---|---|
| C1 | Comparable comparison | Would the control stores have moved the same way absent the change? | Control chosen because it was convenient |
| C2 | Instrumented exposure | Do we know which store-weeks actually received the treatment, and when? | A rollout date recorded at month grain |
| C3 | Fixed window and stopping rule | Was the analysis horizon written down before the first read-out? | "We looked weekly and stopped when it looked good" |
| C4 | Operational safety | Does any guardrail metric (queue, shrink, safety, overtime) breach during the test? | Guardrails absent from the dashboard |
| C5 | Reversible economics | Is the decision still positive under the pessimistic end of the interval? | Payback quoted at the point estimate only |

### 1.2 What this sheet does not cover

- Vendor contracts, data-processing agreements, and legal review of camera-based measurement. Those are governance artefacts with their own owners.
- The design of the store KPI tree itself. That is `01_Store_KPI_Tree.md`; this sheet consumes its definitions.
- The availability metric's own denominator rules. That is `02_Shelf_Availability_Metric_Card.md`.
- Any claim of measured impact. The arithmetic here is derived and illustrative, and is meant to be re-run on the reader's own inputs.

### 1.3 Evidence labels used throughout

- `DERIVED` — follows from algebra or from a named statistical result, and is re-checked in this file.
- `ILLUSTRATIVE` — a constructed input chosen to make the arithmetic visible. Values are plausible, not sourced.
- `OPEN` — depends on vendor behaviour, local law, or a retailer's own system, and cannot be settled from this document.

When a table mixes labels, the label attaches to the input, never to the arithmetic. A payback computed from an illustrative wage is illustrative even though the division is exact.

### 1.4 Notation conventions for this file

| Symbol | Meaning | Units | Type and valid range |
|---|---|---|---|
| `Y` | primary outcome for one store-week | depends on metric | real number |
| `n_c`, `n_t` | number of clusters (stores) per arm | count | integer `>= 2` |
| `m` | observations per cluster inside the window | count | integer `>= 1` |
| `rho` | intra-cluster correlation | dimensionless | `[0, 1)` |
| `DEFF` | design effect from clustering | dimensionless | `>= 1` |
| `sigma` | within-cluster standard deviation of `Y` | same as `Y` | `> 0` |
| `pi_c` | share of treated units that actually received treatment | fraction | `[0, 1]` |
| `MDE` | minimum detectable effect at stated power | same as `Y` | `> 0` |

Angles, ratios, and money are always given with units. A percentage-point difference is written `pp` and never `%`; a relative change is written `%`.

### 1.5 The thirty-minute protocol

Use the sheet in this order when a test result lands on your desk and a decision is due today.

1. Section 3 — write the six definitional sentences (grain, exposure, window, comparator, guardrail, decision rule) on one page. If any sentence needs a meeting, stop: the test was not pre-registered.
2. Section 7 — walk the eighteen gates and mark each `pass`, `fail`, or `unknown`. Two `unknown` marks in the safety family forces *iterate*, not *ship*.
3. Section 4 and Section 5 — recompute the effect, its interval, and the compliance-adjusted version. Do not accept a point estimate without an interval.
4. Section 9 — recompute payback and break-even. A statistically clean result with a 200-week payback is a *stop*.
5. Section 8 — name the one failure mode you cannot exclude, and state what evidence would exclude it.

### 1.6 Version and review trigger

Version 1.1 replaces a 1.0 draft that repeated the same six-sentence definitional block through the whole file; the definitions are now one table, and the space is spent on derivation and diagnosis. Review this sheet when any of these change: the primary metric definition, the store sampling frame, the guardrail set, or the cost of the hardware under test.

## 2. Concept Map / Mental Model

```text
 S0 FRAME        S1 INSTRUMENT      S2 PRE-REGISTER     S3 EXECUTE       S4 MEASURE      S5 DECIDE
 ----------      --------------     ---------------     -----------      ----------      ---------
 pick stores --> define exposure -> fix window and ---> run with  ----> recompute ---> ship /
 and weeks       and guardrails     stopping rule       fidelity log    effect + CI     iterate / stop
      |                |                   |                  |               |              |
      v                v                   v                  v               v              v
  [G1 balance]    [G2 instrumentation] [G3 freeze]      [G4 fidelity]  [G5 economics]  [signed note]
```

Read the diagram as a pipeline of five gate checks, not as five phases of a project plan. The gates are ordered by cost of repair: a balance failure found at S0 costs a spreadsheet, the same failure found at S4 costs the whole test.

### 2.1 Stage S0 — frame the comparison

The frame stage answers one question: *compared with what?* There are four legitimate comparators and they are not interchangeable.

- **Concurrent control stores** — a set of stores running the old process in the same weeks. Cheapest to defend, hardest to balance.
- **The same stores before the change** — a pre/post comparison. Vulnerable to seasonality, assortment resets, and anything else that moved in the same weeks.
- **The same stores, alternating weeks** — a switchback. Strong against slow drift, weak against carry-over between periods.
- **A constructed counterfactual** — a synthetic control built from untreated stores weighted to match history. Useful when only one or two stores can be treated; its credibility rests entirely on pre-period fit.

A field test that mixes two comparators inside one read-out ("we used control stores but also compared to last year") has two estimands and therefore no single answer. Pick one primary comparator and present the other as a sensitivity, labelled as such.

### 2.2 Stage S1 — instrument the exposure

Exposure is the *fact* of treatment, not the *intent* of treatment. The distinction matters because the two produce different estimands and different decisions.

- Intent-to-treat (ITT) compares stores assigned to the new process with stores assigned to the old one, whether or not anyone complied. It answers "what happens if we roll this out as a policy?".
- Complier-average effect (CACE) compares the store-weeks that actually used the new process with what those same weeks would have done under the old one. It answers "what does the process do when it is used?". Section 4.3 derives the adjustment and its assumptions.

Both numbers should appear in the decision note, because they answer different questions and they can point in different directions. A process that works well but is used in 40% of weeks is a training problem; a process used in 95% of weeks that does nothing is a design problem.

### 2.3 Stage S2 — pre-register the window and the stopping rule

Pre-registration for a store test is one page, and it is not bureaucracy: it is the only defence against the most common defect in retail experimentation, which is reading the dashboard every Monday until a pleasant week appears.

The page states: start date and end date at day grain; the primary metric; the guardrail metrics with thresholds; the analysis population; the minimum effect worth acting on; and the rule for an early stop (either a safety stop or a pre-declared interim look with an alpha-spending plan).

If the window cannot be fixed in advance because the rollout is demand-driven, replace the fixed window with a fixed *rule*: "the analysis uses the first four complete weeks after each store's go-live date, and stores are only included once four weeks have elapsed". That is still a pre-registration, because the rule, not the calendar, determines inclusion.

### 2.4 Stage S3 — execute with a fidelity log

Fidelity is the degree to which the delivered process matched the designed process. It is measured, not assumed, and it needs three cheap artefacts.

- A dated go-live record per store, at day grain, with the version of the procedure deployed.
- A weekly adherence count: how many of the expected task instances used the new procedure.
- A deviation log: every local workaround, with the store, the week, and the reason.

The deviation log is the most valuable and the most neglected artefact. When a test fails, the deviation log usually explains why; when a test succeeds, the deviation log tells you what the rollout will actually look like.

### 2.5 Stage S4 — measure with the interval, not the point

The measurement stage produces four numbers, in this order of importance: the effect with its interval; the compliance-adjusted effect; the guardrail deltas; and the cost per store-week. Reporting order matters because readers anchor on the first number they see, and a point estimate without an interval anchors the decision on precision the data does not have.

Two arithmetic habits prevent most errors here. First, compute the standard error at the level of randomisation, not the level of rows: if stores were randomised, the store is the unit. Second, recompute the interval after any change to the analysis population, because dropping stores changes `n` and therefore the interval width.

### 2.6 Stage S5 — decide, and write down what would change the decision

The decision is one of three, and the note must say which.

- **Ship** — the lower end of the interval still clears the cost of rollout, no guardrail breached, and the failure modes you cannot exclude are ones you can monitor.
- **Iterate and retest** — the effect is plausible but the interval includes zero, or fidelity was too low to interpret, or one guardrail is `unknown`.
- **Stop** — the interval excludes the effect size needed for payback, or a safety guardrail breached, or the comparison itself is not credible.

A decision note that does not state its own reversal condition cannot be reviewed later. Write the sentence: "I would reverse this if the four-week read-out shows X, or if adherence falls below Y".

### 2.7 The structural trap: compensatory scoring

Most home-made checklists fail at exactly one point: they turn the gates into a weighted score. A weighted score lets a store pass with a failed safety gate if enough other items score well. Section 4.4 derives why that is not a cosmetic issue, and Section 7 gives the conjunctive replacement.

The rule to remember: safety gates are ANDed, never averaged. Quality gates may be scored, because a weak score on one item is genuinely a partial result.
