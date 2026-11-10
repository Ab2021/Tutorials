---
case_id: "CASE-15-01-001"
case_title: "Probabilistic load forecasting as a reserve-procurement decision, not a curve-fitting exercise"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Energy / distribution grid operations and load forecasting"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["distribution-shift", "human-in-the-loop"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Grid operations analytics lead (proposed owner), with balancing-area dispatch authority retained by the control-room shift manager"
source_index: "./INDEX.md"
review_trigger: "Any change to reserve cost, value-of-lost-load assumption, weather-ensemble supplier, settlement revision policy, or the operating-day calendar contract"
---

# Probabilistic load forecasting as a reserve-procurement decision, not a curve-fitting exercise

**Evidence declaration.** Every number in this case is a constructed teaching value. Nothing here is a measurement of a named utility, a published benchmark, or evidence of realized savings. The case is written as an analysis and design exercise whose arithmetic is independently checkable; its operational claims remain `not-measured`.

## 1. The One-Line Framing

> Given **hourly area-level electricity demand in MW plus a weather-ensemble and calendar snapshot frozen at a 17:00 local issuance cut-off**, decide **which quantiles of next-operating-day demand to publish, and what reserve capacity to procure for each hour**, for **the control-room shift manager and the commercial desk**, to improve **the joint outcome of avoided involuntary load shedding and avoided excess reserve cost**, subject to **a hard prohibition on using post-cutoff information, a 30-minute issuance deadline, and the requirement that published quantiles be monotone across probability levels**, compared with **the current operator-adjusted persistence baseline (same weekday last week, manually nudged by the shift engineer)**.

The task category is **forecasting followed by a constrained allocation decision**, executed once per day at 17:00 with an intraday re-issue at 09:00 for the remaining hours. It is not a pure regression problem: the deliverable is a *set* of quantiles whose joint behaviour drives a monetary decision, and the decision cost is wildly asymmetric.

**Primary reusable pattern:** `evaluation-mismatch`. A model can win on mean absolute error and still lose money, because reserve procurement is driven by a far-tail quantile that contributes almost nothing to average error. **Secondary patterns:** `distribution-shift` (weather regimes and holiday calendars move the conditional distribution), `human-in-the-loop` (the shift manager may override, and the override itself is a measured decision that can destroy probabilistic information).

**The decisive tradeoff, stated up front.** Reserve cost and shortfall cost differ by roughly three orders of magnitude in the illustrative ledger below (about $12 per reserved MW-hour against $9,000 per unserved MWh). The cost-optimal reserve level is therefore an extreme quantile — around the 99.87th percentile — which an eight-year hourly sample can barely resolve by direct counting and cannot resolve at all within a single season. Meanwhile the *published* 90% interval is a communication and screening artefact. Confusing the two produces a system that looks well calibrated at 90% and still sheds load several times a year.

**Why the obvious framing fails.** The naive framing — "minimize forecast error" — treats all hours as exchangeable and all MW errors as equally costly. It silently optimises the middle of the distribution, hides the timing error that matters for the evening ramp, and gives no basis for choosing a reserve number at all. A second tempting framing — "publish the 90% interval because that is the standard" — imports a convention from interval reporting into a decision whose loss ratio implies a different quantile. Both framings are rejected here.

**Interview cue.** A reader should be able to state, in thirty seconds: input, cut-off, output set, owner, baseline, binding constraints, and the reason the loss ratio changes the answer.

**Acceptance for this section.** Input (hourly MW + frozen weather/calendar snapshot), output (monotone quantile vector + reserve MW per hour), beneficiary (shift manager, commercial desk), baseline (adjusted persistence), constraint (cut-off discipline, latency, monotonicity), and evidence basis (illustrative) are all identifiable above.

**Takeaway:** The framing must name the decision and its loss ratio, because that ratio — not the error metric — determines which quantile is the product.

## 2. Context & Background

### 2.1 The system being operated

The setting is a regional distribution utility serving about 1.2 million metered customers through 84 primary substations and roughly 3,400 km of medium-voltage feeder. It does not own most generation. It does own the obligation to serve, which means it must contract for reserve capacity and, in extremis, execute involuntary load shedding under a published safety protocol.

Demand on the area is the aggregate of five zones. Each zone has its own load composition, and that composition determines how it responds to weather:

| Zone | Dominant load | Weather sensitivity | Typical share of area peak |
|---|---|---|---|
| Z1 Riverside | Residential, electric resistance heating | High cold sensitivity | 24% |
| Z2 Uptown | Commercial offices, HVAC | High heat sensitivity, weekday pattern | 21% |
| Z3 Industrial Belt | Process load, two large consumers | Low weather sensitivity, step changes | 26% |
| Z4 Hills | Residential, heat pumps, poorer insulation | Very high both directions | 15% |
| Z5 Coastal | Mixed, tourism seasonality | Moderate, strong humidity interaction | 14% |

These shares sum to 100%. The peak-hour area load in the illustrative year is 1,050 MW, so the implied zone peaks are 252, 220.5, 273, 157.5 and 147 MW. Those are constructed values used consistently throughout this case.

### 2.2 The workflow around the decision

The daily cycle has fixed hand-off points:

1. **06:00** — Numerical weather prediction (NWP) cycle A is issued upstream; ensemble members become available to the utility between 06:20 and 06:50 after transfer and decoding.
2. **12:00** — Cycle B is issued; a partial decode may be available by 12:40.
3. **17:00** — The day-ahead forecast is frozen and published. This is the *issuance cut-off*. Reserve nominations for the next operating day are submitted shortly afterwards by the commercial desk.
4. **17:00–24:00** — The forecast may be re-issued only through the intraday channel, which has a separate version identifier and does not retroactively change the day-ahead record.
5. **Next day 00:00–24:00** — Operation. The shift manager may override a zone-level reserve instruction at any hour.
6. **D+1 to D+5** — Metered load arrives, is revised, and eventually settles. Settlement is the truth used for scoring.

The cut-off is not a technical nicety. If a forecast issued at 17:00 uses a weather cycle that arrived at 17:40, the published record becomes unreproducible and the evaluation becomes self-fulfilling: the model appears better because it saw the future. Section 5d and Section 6 Step 1 make this audit explicit.

### 2.3 Stakeholders and their actual interests

| Stakeholder | Interest | Failure they experience personally |
|---|---|---|
| Shift manager (decision owner for shedding) | No unserved energy, no surprise ramp | Shedding a feeder at 19:10 while the published 90% band said "normal" |
| Commercial desk (reserve buyer) | Buy enough, not too much | Over-procurement charged to the operating budget line, scrutinised quarterly |
| Network planning engineer | Headroom adequacy months ahead | Using a forecast product whose tail was never validated for planning |
| Regulatory reporting analyst | Defensible published statements | Having to explain why a "90% interval" contained 87% of outcomes |
| Field operations | Warning before switching | Being dispatched without notice during a heat event |
| Public-safety coordinator | Heat-vulnerability outreach | Outreach triggered by a deterministic threshold that is crossed erratically |

### 2.4 Baseline and its documented weaknesses

The incumbent baseline is *adjusted persistence*: take the same weekday of the previous week, add the operator's manual adjustment for expected weather and known outages. It has three documented weaknesses, none of which are model-quality complaints:

- **No uncertainty representation.** It produces a single number. Reserve is then set by a fixed heuristic (peak + 5%), which is neither derived from costs nor from the actual spread.
- **Operator dependence.** Two engineers produce materially different adjustments on similar days. The adjustment is not recorded in a structured field, so it cannot be evaluated.
- **Calendar brittleness.** Public holidays, daylight-saving transitions, and the two-week industrial shutdown are handled by memory and a paper note.

### 2.5 Scale, timing, and geography

- **Entities:** 5 zones, 84 substations, 1 area aggregate.
- **Decision instances:** 1 day-ahead issuance per day (365 per year) × 24 hours × 9 published quantiles = 78,840 numbers per year; plus one intraday re-issue per day for the remaining hours.
- **Data volume:** 8 years of hourly settlement data = 70,080 hours per series; 5 zone series + 1 area series.
- **Weather:** 51-member ensemble, 3 cycles per day, 10 variables retained, at 9 grid points → 51 × 3 × 10 × 9 = 13,770 values per day before interpolation.
- **Peak conditions:** summer heat events and winter cold snaps; the two regimes have different error structures and must be evaluated separately.
- **Boundaries:** one synchronous balancing area, one currency, one settlement revision policy.

### 2.6 Prior attempts

Two prior attempts are relevant and both were *not* carried forward, for reasons that constrain the design:

- A **point-forecast neural network** (2023 vintage) improved mean absolute percentage error on mild days but was withdrawn after operators lost trust during a heat event where it under-predicted the evening ramp by more than 90 MW. No uncertainty was available to explain the miss. *Status: historical description supplied as the organisation's narrative in this exercise; not independently verified.*
- A **fixed-quantile lookup table** built from three years of residuals. It produced intervals that were acceptable on average and badly wrong in the tail, and it could not incorporate the current weather ensemble. *Status: same caveat.*

Neither attempt produced a reproducible evaluation pipeline, which is the deeper reason both failed: nobody could say afterwards whether the model or the weather forecast had been at fault.

**Takeaway:** The baseline's weakness is structural (no uncertainty, no recorded adjustment), so a better model that also has no reproducible record would repeat the failure.

## 3. The Business Question

### 3.1 The question in the owner's words

> "At 17:00 I need to know how much reserve to carry for tomorrow's evening ramp, and I need to know how often we are likely to be short. If you give me a band that says '90%', I will ask you what the other 10% costs me — and if you cannot answer that, I will keep using peak plus five percent."

This is an **illustrative stakeholder question**, reconstructed for the exercise. It is not a quotation from a real person.

### 3.2 Decision contract

| Decision contract field | Content |
|---|---|
| Question | For each hour of the next operating day, how much reserve capacity should be procured, and what demand quantiles should be published to support that choice? |
| Decision owner | Control-room shift manager for the operational shedding decision; commercial desk for the reserve nomination, acting on the published quantile set |
| Trigger and deadline | Daily at 17:00 local; hard issuance deadline 17:30; reserve nominations due 18:00 |
| Available actions | (a) Procure the cost-optimal quantile level; (b) procure a policy-fixed reserve (peak + 5%); (c) procure an intermediate level with an explicit shortfall-probability ceiling; (d) no change, keep adjusted persistence |
| Value and downside | Avoided unserved energy at an assumed value of lost load; against reserve capacity cost and the credibility cost of alarms that do not materialise |
| Objective | Minimise expected total cost per operating day, in currency units, summed over 24 hours: reserve cost + expected shortfall cost + a stated penalty for interval miscoverage in published material |
| Hard constraints | No post-17:00 information; monotone quantiles; issuance within 30 minutes; every published number traceable to a frozen input manifest; no automated shedding action |
| Scope | Area aggregate and five zones; hours 00–23 of the next operating day; one settlement revision policy |
| Out of scope | Unit commitment of generation the utility does not own; intra-day market bidding strategy; long-run network reinforcement planning; individual customer-level forecasting |

### 3.3 Objective versus proxy

The business objective is expected cost. The convenient proxy is pinball loss, averaged over a quantile grid. These agree only when the quantile grid's weights match the decision's cost ratio, which in this case they do not: the reserve decision is dominated by a quantile near 0.9987, while a nine-point grid at 0.1…0.9 devotes no weight at all above 0.9.

The second proxy conflict is coverage. A published 90% interval that achieves 87.4% coverage is *not* automatically a failure of the reserve decision, because the reserve decision does not use the 90% interval. Treating one number as the scoreboard for both products is the central evaluation-mismatch of this case. Section 9 therefore defines two distinct metric families with separate gates and separate owners.

**Interview cue.** Ask which objective is optimised, what "no intervention" costs, and what an expensive false negative looks like in MW and currency.

**Acceptance for this section.** The question is answerable by a decision (procure at a stated quantile level), a deferral (gather tail evidence), or a rejection (keep baseline), and the proxy conflict is named rather than hidden.

**Takeaway:** One decision, two metric families: publication calibration and reserve-cost regret must be evaluated separately or the scoreboard will reward the wrong thing.

## 4. Assumptions & Constraints

Assumptions are conditions believed true and testable. Constraints are externally imposed limits. Choices under team control are recorded in Section 8c, not here.

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Hourly metered load is available at D+1 with revisions, and the *revised* value is the correct scoring truth | Data | Compare D+1 and settled values on 12 months of data; quantify revision distribution | unverified | Scoring against unreconciled truth corrupts every calibration claim | Score against settled values only; freeze a revision log | Settlement policy change / data owner |
| A-02 | NWP cycle A is fully available by 16:30 for every variable required | Data/infra | Ingest completeness audit over 24 months by cycle and variable | unverified | Mixed-vintage bundles silently enter the record | Explicit fallback to prior complete cycle with a manifest flag | Supplier or transfer change / platform owner |
| A-03 | Load response to temperature is monotone within the observed range and does not invert at extreme heat | Model | Segmented regression on 8 years; test slope sign above the 99th temperature percentile | unverified | Extrapolated tail is directionally wrong exactly when it matters | Saturating or piecewise model with documented breakpoints | New extreme event / modelling owner |
| A-04 | Zone loads are additive to the area without material unaccounted-for energy | Data | Reconcile zone telemetry against area settlement metering; quantify residual | unverified | Hierarchical reconciliation is anchored to a wrong total | Add a reconciliation residual term, published explicitly | Metering change / operations |
| A-05 | Reserve capacity can be procured in the stated quantity at the stated unit cost, and the cost is linear in MW | Business | Contract review with procurement; check block/step pricing | unverified | The cost-optimal quantile is mis-stated | Use the actual step-cost curve and solve numerically | Contract renewal / commercial desk |
| A-06 | Value of lost load is a defensible scalar for this decision | Business | Regulatory or internal policy reference; sensitivity over a plausible range | unverified | Optimal quantile moves by orders of magnitude in probability | Present a range and a policy-chosen ceiling on shortfall probability | Policy revision / regulatory analyst |
| A-07 | Weather ensemble members are exchangeable draws from a calibrated predictive distribution | Model | Rank histogram and spread-error diagnostics on historical cycles | unverified | Ensemble spread is misread as a probability statement | Recalibrate spread or use a residual model on top | NWP version change / modelling owner |
| A-08 | Public holidays and the industrial shutdown are known in advance and encoded | Business/data | Calendar table review with operations each quarter | verified in principle, not in this exercise | Systematic level errors on 12–18 days per year | Manual calendar override with recorded reason | Calendar publication / operations |
| A-09 | The 17:00 cut-off is technically enforceable in the pipeline, not merely in policy | Infra | Timestamp audit of every feature's availability instant | unverified | Leakage; unreproducible published record | Cut-off enforcement at the feature-store layer with rejection | Pipeline redesign / platform owner |
| A-10 | Operator overrides are recorded with reason codes and time | Human | Audit of override logging | unverified | Override effects cannot be separated from model effects | Structured override form; no override without a code | Interface change / shift manager |
| C-01 | Published quantiles must be monotone in probability level | Constraint | Enforced in the publication schema validation | confirmed as policy | Non-monotone output is unpublishable and confuses downstream consumers | Rearrangement or constrained fitting | Never relaxed |
| C-02 | Issuance must complete within 30 minutes of the cut-off | Constraint | Timed dry runs with the full ensemble | confirmed as policy | Late publication forces the baseline | Pre-computed features plus a fast scoring path | Peak-load growth / platform owner |
| C-03 | No automated load shedding; every shedding action is human-authorised | Constraint | Design review and runbook | confirmed as policy | Legal and safety exposure | Human-only action path | Regulatory change |
| C-04 | Forecast must not use settlement-revised load for hours after the cut-off | Constraint | Feature-availability audit | confirmed as policy | Leakage | Point-in-time feature store | Never relaxed |
| C-05 | Data residency: all load and customer data stay in the national region | Constraint | Infrastructure review | confirmed as policy | Product cannot be deployed | Regional deployment only | Contract change |
| C-06 | Reserve nomination granularity is hourly, in whole MW | Constraint | Market rules extract (illustrative) | confirmed as scenario assumption | Sub-MW optimisation is meaningless | Round with a documented rule | Market rule change |

The three assumptions most likely to reverse the design are **A-01** (truth definition), **A-03** (tail monotonicity), and **A-06** (value of lost load). Each is examined quantitatively in Section 6 and revisited in Section 12.

**Interview cue.** Change A-06 to a much lower value of lost load and explain what happens to the recommended quantile, the model, and the publication product.

**Acceptance for this section.** Every critical assumption has a check and a consequence; every constraint has a policy source or is labelled a scenario assumption.

**Takeaway:** The cut-off, the truth definition, and the loss ratio are the three assumptions that decide whether this product is honest, scoreable, and economically meaningful.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Detail |
|---|---|
| Unit of analysis | One row per (series, timestamp) where series ∈ {area, Z1…Z5} and timestamp is an hourly interval start in UTC, with a local-calendar mapping for operating-day assembly |
| Volume and window | 8 years hourly = 70,080 intervals per series; 6 series; 420,480 rows before feature expansion. Weather ensemble adds 13,770 values per day |
| Inputs | Metered load (MW, interval average), SCADA feeder telemetry (MW, 1-minute, used for nowcasting only), NWP ensemble (temperature, humidity, wind speed, cloud cover, solar irradiance, at 9 grid points, 51 members, 3 cycles), calendar (weekday, holiday flag, operating-day index), known outage schedule (planned switching, MW by zone-hour) |
| Outcome/label | Realized area or zone demand for the same interval, in MW, from the settlement series at its currently accepted revision; the *mature* label is the value after the settlement window closes |
| Population and exclusions | All hours of the 8-year window for training; the two years following are the backtest. Excluded: hours with an unmetered-zone telemetry failure that exceeds 15% of zone load, and the six hours surrounding a major black-start exercise, both with recorded reasons |
| Join semantics | Zone and area series join on the UTC interval start; weather joins on the nearest grid point and the cycle's valid time; each join is many-to-one on weather and one-to-one on load. Duplicate interval starts (from DST fall-back) are keyed by local offset, not by naive local clock |
| Splits | Train: years 1–6. Validation: year 7 (used for hyperparameters and the calibration layer). Test: year 8 plus a rolling-origin backtest across years 7–8 with daily refits. No random split anywhere: adjacent hours are near-duplicates and a random split is leakage by construction |

**Grain warning.** The most common defect in this domain is treating a "day" as 24 rows. Twice a year the local operating day has 23 or 25 hours. The dataset contract must carry the operating-day index and the expected hour count, and every downstream shape must be derived from that contract rather than hard-coded to 24.

### 5b. Tools & Stack

| Component | Role | Notes |
|---|---|---|
| Python 3.11, pandas/polars | Tabular assembly and point-in-time joins | Polars for the wide ensemble expansion |
| scikit-learn-compatible gradient boosting with quantile or multi-quantile objective | Candidate model | Must support monotone constraints or post-hoc rearrangement |
| Custom pinball/coverage scorer | Evaluation | Written as a pure function with unit tests on hand-computed vectors |
| Feature store with event-time semantics | Cut-off enforcement | Every feature carries `available_at`; the assembler rejects `available_at > cut-off` |
| Object storage + versioned manifests | Reproducibility | Manifest lists cycle IDs, file hashes, code revision |
| Workflow scheduler | Daily issuance and daily refit | Idempotent by (operating_day, cut-off) key |
| Dashboard / publication API | Consumption by commercial desk and control room | Schema-validated; monotonicity enforced at write time |
| Notebook environment | Analysis only | Never in the serving path |

No deep learning stack is proposed. With 70,080 hourly rows per series and strong calendar structure, a well-featured boosted quantile model plus a calibration layer is the credible choice; a sequence model is a *considered, untested* alternative recorded in 5e.

### 5c. Method Summary

Pipeline: `frozen snapshot -> point-in-time feature matrix -> multi-quantile model -> monotonic rearrangement -> hierarchical reconciliation -> calibration layer -> publication + reserve rule -> decision`.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Adjusted persistence (incumbent) | Operational description | Zero infrastructure, transparent | No uncertainty; operator-dependent; brittle calendar | Baseline, retained for comparison | Never removed; it is the comparator |
| Gradient-boosted quantile regression on weather-ensemble summaries | Method literature and the structure of the data | Handles calendar and weather interactions; multi-quantile in one model | Cannot represent the joint tail well; ensemble summaries lose member-level information | Chosen (proposed) | If tail calibration fails after the conformal layer |
| Quantile regression neural network with ensemble-member input channels | Plausible, untested here | Could use full ensemble spread | Needs far more data per regime; harder to audit | Considered, not tested | If ≥15 years of data and a dedicated MLOps owner exist |
| Residual bootstrap on a point model | Standard practice | Simple, explainable | Assumes homoscedastic residual structure across regimes | Considered, not tested | If interpretability outweighs tail accuracy |
| Fixed residual-quantile lookup | Prior attempt | Trivial to operate | Ignores current weather; tail wrong | Rejected (historical) | — |

The chosen model's objective is the sum of pinball losses over the quantile grid. Hyperparameters are selected on year 7 by average pinball across the 0.1–0.9 grid, but the *gate* is tail coverage after calibration, not the grid average. This split is deliberate: tuning on the grid average while gating on the tail prevents the common failure where a model is tuned to the metric it is not judged on.

### 5d. Data provenance, quality, exclusions and leakage controls

**Provenance.** Load data: settlement system extract, revision-tagged. Weather: named NWP supplier, cycle ID retained. Calendar: operations-published table. Outage schedule: switching log, entered by field operations, with a known 1–3 day lag for retrospective corrections.

**Quality controls and observed-or-not status.**

| Check | Rule | Action on failure | Status in this exercise |
|---|---|---|---|
| Interval completeness | Every expected interval present for the operating day (23/24/25) | Block publication for that day; do not silently fill | defined, not executed |
| Duplicate timestamps | Exactly one row per (series, UTC interval) | Reject the batch | defined, not executed |
| Implausible jumps | Zone load change > 35% hour-over-hour without a logged switching event | Flag for review; retain but mark | defined, not executed |
| Weather completeness | All required variables present for the chosen cycle | Fallback cycle with manifest flag | defined, not executed |
| Negative load | Any negative zone load | Quarantine; investigate reverse flow or meter error | defined, not executed |
| Revision monotonicity | D+1 value vs settled value differ beyond 2% | Record; use settled for scoring only | defined, not executed |

**Leakage controls.** Four specific leaks were designed out:

1. **Revised-load feature leak.** Using the current revision of yesterday's load as a lag feature is legitimate; using the *settled* value for an hour whose settled revision arrived after the cut-off is not, because the settled value incorporates corrections that were unknowable at 17:00. The feature assembler therefore serves lag features from the revision snapshot taken at the cut-off.
2. **Weather-observation leak.** Actual observed temperature for the target hour is not available at 17:00 the previous day. Only forecast cycles are eligible. A pipeline that joins "temperature" without a source-type column will silently join observations.
3. **Outage-log retro-edit leak.** Switching logs corrected after the event can explain away errors. Only the cut-off snapshot of the outage schedule is eligible as a feature; later corrections are evaluation annotations, never features.
4. **Target-day calendar leak.** The operating-day index must be computed from the calendar, not from any realised load, so that a holiday cannot be inferred from unusual consumption.

**Preprocessing discipline.** Scaling and encoding are fit inside the training window of each rolling origin. The calibration layer (Section 6, Step 6) is fit on a strictly earlier window than the window it is applied to.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | Forecast-verification literature on pinball loss and quantile scoring | conceptual | Loss definition and propriety | public | Not a data source for this case |
| SRC-02 | NWP supplier ensemble documentation | conceptual | Member count and cycle timing | restricted | Cycle latencies here are scenario values |
| SRC-03 | Internal settlement revision policy | unavailable in this exercise | Truth definition | restricted | Revision distribution assumed, not measured |
| SRC-04 | Reserve contract rate card | unavailable | Cost coefficients | restricted | $12/MW-h is illustrative |
| SRC-05 | Value-of-lost-load policy reference | unavailable | Shortfall penalty | restricted | $9,000/MWh is illustrative |
| RUN-01 | Rolling-origin backtest | not executed | Would establish calibration and regret | — | No run exists; all results below are hand-computed teaching arithmetic |

**What is reproducible.** Every formula and arithmetic step in Section 6 can be reproduced with a calculator. **What is not.** No model was trained, no backtest was run, and no published utility data was used. The case therefore teaches method and decision logic; it does not report performance.

**Interview cue.** Defend the split, the label revision rule, and the single most dangerous join in this pipeline.

**Acceptance for this section.** Another practitioner can reproduce the reasoning and the arithmetic, can see exactly which evidence is missing, and can distinguish tested decisions from proposals.

**Takeaway:** The point-in-time feature store is not infrastructure polish; it is the only mechanism that makes the published record falsifiable.

## 6. Analysis Walkthrough

### Step 1 — Freeze a forecast origin and prove what was knowable

Take an illustrative issuance at 17:00 local on a Tuesday, for the following operating day. Cycle A was issued at 06:00, decoded by 06:40. Cycle B was issued at 12:00, decoded by 12:45. An ensemble member set from cycle B is therefore eligible; a *partial* cycle B decode missing the humidity variable is not eligible as a whole, and mixing cycle A humidity with cycle B temperature creates a bundle that no single supplier version produced.

The manifest must record: cycle IDs per variable, retrieval timestamps, the code revision, the feature-store snapshot identifier, and the cut-off instant. Without the last item, the record cannot be audited.

A useful negative trace: an analyst joins `temperature_2m` from a table that contains both forecast and observed rows. For the target day, observed rows exist in the table (they arrived the day after). The join succeeds, the forecast improves dramatically, and the improvement is pure hindsight. The discriminator is a source-type column plus an `available_at` check; the symptom is a suspiciously precise evening ramp. This trace is a constructed diagnostic, not an observed incident.

### Step 2 — Score a single hour with pinball loss

Pinball loss is the standard proper scoring rule for a quantile. Derivation: for quantile level τ ∈ (0,1), the loss is

```text
FORMULA [M1]: L_tau(y, q) = tau * (y - q)      if y >= q
                            (1 - tau) * (q - y) if y < q
WHERE: y realized demand [MW]; q published tau-quantile [MW];
       tau dimensionless in (0,1); L in MW.
CONSTRAINT: q and y refer to the same series, interval and revision.
NUMERICAL EXAMPLE: tau = 0.9, y = 842 MW, q = 812 MW;
       L = 0.9 * (842 - 812) = 0.9 * 30 = 27 MW.
INTERPRETATION: the same 30 MW miss in the other direction costs 3 MW.
```

Why this is the right loss: it is the negative of the check function whose expectation is minimised by the true conditional quantile. Differentiating the expected loss with respect to q gives `P(Y < q) - tau` up to scale, so the minimiser satisfies `P(Y ≤ q) = tau`. That derivation is what licenses "publish a quantile, score with pinball" as a coherent pair.

Check the asymmetry: at τ = 0.9, an under-forecast of 30 MW costs 27 MW of loss, an over-forecast of 30 MW costs 3 MW. Ratio 9:1. At τ = 0.5 both branches cost 15 MW, so pinball is exactly half the absolute error. That identity is the quickest sanity check on any implementation: `L_0.5 = 0.5 * |y - q|`.

### Step 3 — Aggregate a quantile grid into one number

With nine quantiles at τ = 0.1…0.9 for the same hour, the following hand-computed vector applies:

| τ | q (MW) | y − q (MW) | Branch | L (MW) |
|---:|---:|---:|---|---:|
| 0.1 | 771 | +71 | under | 7.1 |
| 0.2 | 790 | +52 | under | 10.4 |
| 0.3 | 806 | +36 | under | 10.8 |
| 0.4 | 818 | +24 | under | 9.6 |
| 0.5 | 830 | +12 | under | 6.0 |
| 0.6 | 848 | −6 | over | 2.4 |
| 0.7 | 862 | −20 | over | 6.0 |
| 0.8 | 878 | −36 | over | 7.2 |
| 0.9 | 906 | −64 | over | 6.4 |
| — | — | — | sum | 65.9 |

Mean pinball over the grid is `65.9 / 9 = 7.3222 MW`. The quantile-decomposition approximation to CRPS is twice the integral of pinball over τ; with a trapezoidal reading of these nine interior points this gives `2 × 7.3222 = 14.6444 MW`. That number is an *approximation with unrepresented tail mass*, not CRPS: the decomposition integrates from 0 to 1, and the grid stops at 0.9 on one side and 0.1 on the other.

Check: `9 × 7.3222 = 65.9`; doubling is the correct factor because CRPS = 2∫QL dτ. Both checks pass, and both are necessary before quoting the number.

### Step 4 — Coverage is not calibration, and the sample is not independent

Suppose the published 90% interval covered 629 of 720 held-out hours over 30 operating days: `629 / 720 = 0.873611`, i.e. 87.4% against a nominal 90%.

A naive binomial standard error treats the 720 hours as independent:

```text
FORMULA [M2]: SE_binom = sqrt(p * (1 - p) / n)
WHERE: p = observed coverage proportion; n = number of intervals.
NUMERICAL EXAMPLE: p = 0.873611, n = 720;
       SE = sqrt(0.873611 * 0.126389 / 720) = sqrt(0.110411 / 720)
          = sqrt(0.000153349) = 0.012384.
INTERPRETATION: z = (0.873611 - 0.9) / 0.012384 = -2.13, a two-sided p near 0.033.
```

Hourly load errors are strongly serially correlated; a hot afternoon produces a run of misses, not independent coin flips. If the effective number of independent days is 30 and the daily coverage proportions vary with standard deviation 0.105, the day-level standard error of the mean coverage is `0.105 / sqrt(30) = 0.01917`, giving `z = -1.38` and p ≈ 0.17. The conclusion flips from "significant undercoverage" to "inconclusive at this sample size".

Lesson: the nominal 90% target needs a *cluster-aware* uncertainty statement, and 30 days is too few to police a 10% tail. A defensible gate needs at least a full seasonal cycle and a block-bootstrap interval, with the block length justified by the autocorrelation of the error process.

### Step 5 — The decision quantile is not the published quantile

Now the economics. Let reserve capacity cost `c_r = 12` currency units per MW-hour, and let unserved energy cost `c_s = 9000` currency units per MWh (an illustrative value of lost load). Capacity `C` is the procured amount; demand is `Y`. Shortfall is `(Y − C)^+`, so expected cost conditional on a forecast distribution is

```text
FORMULA [M3]: E[Cost | C] = c_r * C + c_s * E[(Y - C)^+]
WHERE: C [MW]; c_r [currency/MW-h]; c_s [currency/MWh]; E[(Y-C)^+] [MW].
CONSTRAINT: one hour, risk-neutral, linear costs, no intertemporal coupling.
```

Differentiate with respect to C: `d/dC = c_r − c_s * P(Y > C)`. Setting to zero,

```text
FORMULA [M4]: P(Y > C*) = c_r / c_s   =>   C* = F^{-1}(1 - c_r / c_s)
NUMERICAL EXAMPLE: 12 / 9000 = 0.0013333; 1 - 0.0013333 = 0.9986667.
INTERPRETATION: procuring the 99.8667th percentile is cost-optimal under M3.
```

Now the boundary. With 8 years of hourly data, 70,080 observations, the highest resolvable empirical probability by direct counting is `1 - 1/70080 = 0.9999857`; but within one season (about 2,190 summer hours) it is `1 - 1/2190 = 0.999543`, and within the 720-hour backtest it is `1 - 1/720 = 0.998611`. The required 0.998667 sits *just beyond* what 720 hours can resolve, and far beyond what 30 days can resolve. The empirical quantile at that level is an order statistic with enormous sampling variability: its standard error scales like `sqrt(τ(1−τ)/n) / f(q)`, and for a demand density `f` near the upper tail the denominator is small.

Consequence: the reserve rule cannot be a plain empirical quantile of the backtest. It must come from a fitted tail model (a parametric tail, an extreme-value tail, or a residual model with a calibrated parametric form), and its uncertainty must be reported. Reporting a number and calling it "the 99.87th percentile" without a tail model is the most common false precision in this domain.

A second consequence follows from the ratio: if the shortfall cost is revised downward to $2,000/MWh, the threshold becomes `12 / 2000 = 0.006`, i.e. the 99.4th percentile. If reserve cost rises to $30/MW-h, the threshold becomes `30 / 9000 = 0.003333`, i.e. the 99.6667th percentile. The product's target moves materially with contract terms, which is why C-05 and A-05 are recheck triggers.

### Step 6 — Calibration layer: conformal adjustment and its limits

A split-conformal-style correction uses held-out residuals to widen or narrow the interval so that empirical coverage matches nominal on the calibration window. For a quantile product, the practical version is:

```text
FORMULA [M5]: q_hat_tau(x) = q_model,tau(x) + delta_tau
WHERE: delta_tau = empirical tau-quantile of the calibration residuals
       r_i = y_i - q_model,tau(x_i) on a strictly earlier window.
CONSTRAINT: exchangeability between calibration and target windows;
            homoscedastic residual behaviour across the target regime.
NUMERICAL EXAMPLE: calibration residual 0.9-quantile = +18 MW;
       an uncalibrated q_0.9 of 812 MW becomes 830 MW.
INTERPRETATION: additive shift corrects a level bias, not a scale bias.
```

Two failure boundaries are worth stating precisely. First, if the residual distribution is wider in the target regime (a heat dome), an additive shift under-corrects: the interval remains too narrow even though its coverage on the calibration window was nominal. Second, if the calibration window is a mild season and the target is extreme, exchangeability is violated and the correction can make coverage *worse*. The diagnostic is coverage computed by regime, not overall: an overall 0.90 with 0.78 in heat hours is a failing product.

A scale-aware variant multiplies rather than shifts: `q_hat = mu(x) + s * (q_model − mu(x))`, with `s` fit to match coverage. That variant is preferable when the model's spread is directionally right but too small. It is not adopted here without evidence; it is recorded as a variant to test.

### Step 7 — Monotonicity: crossing quantiles and the rearrangement fix

Independent quantile fits can cross: for the same hour, `q_0.7 = 860` and `q_0.8 = 852`. Publishing that vector violates C-01 and produces nonsensical reserve numbers (the "80% case" is below the "70% case").

The rearrangement operation sorts the fitted values at each x, which yields a monotone vector. The known result (Chernozhukov, Fernández-Val and Galichon, 2010) is that rearrangement weakly reduces the pinball loss for every τ and therefore cannot hurt the quantile score. That is a theorem, and it is why sorting is a legitimate fix rather than a cosmetic hack.

But note what sorting does *not* do: it does not restore the intended probability levels when crossing is severe. If five of nine quantiles cross, the sorted vector is a monotone approximation whose τ-assignment is wrong, and the honest response is to refit with a multi-quantile objective that enforces monotonicity during estimation.

Check the arithmetic of the violation: with `q_0.7 = 860` and `q_0.8 = 852`, the implied 10% mass between τ = 0.7 and 0.8 is negative, `852 − 860 = −8 MW`. Sorting swaps the two values, restoring a non-negative 8 MW gap. The reserve rule reading q at 0.8 would have procured 8 MW less than the 0.7 reading — an inconsistency a downstream consumer cannot detect without the monotonicity check.

### Step 8 — Hierarchical coherence: quantiles do not add

The area forecast must be consistent with the five zone forecasts, because reserve is bought at area level while operational instructions are issued at zone level. The naive approach — sum the zone 90th percentiles and call it the area 90th percentile — is wrong whenever zones are not comonotonic.

A minimal counterexample, constructed:

```text
Two zones; each adds +100 MW with probability 0.10, else 0.
Zone 90th percentile (each): 0 MW, because P(0) = 0.90 >= 0.90.
Area 90th percentile: P(area = 0) = 0.9 * 0.9 = 0.81 < 0.90, so the
90th percentile of the area is 100 MW (P(area <= 0) = 0.81, P(<=100) = 1).
Sum of zone 90th percentiles = 0 MW; area 90th percentile = 100 MW.
Gap = 100 MW, from dependence structure alone.
```

With fully comonotonic zones (the same weather shock hitting all of them), the two coincide. Real zones sit between these extremes. Therefore the coherence step must either (a) model the joint distribution and derive area quantiles from simulated joint paths, or (b) apply a reconciliation rule and *document that the area quantiles are reconciled, not independent*.

An illustrative magnitude for the actual data shape: if the five zone 90th percentiles are 268, 236, 279, 176 and 158 MW, they sum to 1,117 MW. If a joint simulation gives an area 90th percentile of 1,064 MW, the naive sum overstates the requirement by 53 MW. At $12/MW-h that is `53 × 12 = $636` per hour of unnecessary reserve — small per hour, `$636 × 4 = $2,544` over a four-hour evening block, but the sign of the error depends on the dependence assumption, and the point is that the number is an artefact of a method choice that nobody documented.

### Step 9 — The timing-versus-magnitude split at the evening ramp

Reserve adequacy depends on the peak hour; public-safety messaging depends on the *time* the peak occurs. These are different functionals of the forecast distribution, and a model can be good at one and bad at the other.

Illustrative three hours with realised load `[880, 960, 1010]` MW: the realised peak is 1010 MW at hour 3.

- Candidate point forecasts `[875, 955, 990]` → peak magnitude error `1010 − 990 = 20` MW low; peak-hour identification correct.
- Baseline point forecasts `[900, 930, 995]` → peak magnitude error 15 MW low; peak hour also correct.
- Candidate wins on hourly MAE: `(5 + 5 + 20)/3 = 10` MW versus baseline `(20 + 30 + 15)/3 = 21.67` MW.
- Baseline wins on peak magnitude error: 15 MW versus 20 MW.

Both facts can be true simultaneously. If the reserve rule is "area capacity must cover the daily maximum", the baseline is the better input despite the worse MAE. If the rule is "hourly energy balance", the candidate is better. The evaluation must state which functional is scored; a single MAE number cannot arbitrate.

A subtle trap: taking the maximum of separate hourly quantiles is not the quantile of the daily maximum. If each of two hours independently exceeds threshold `m` with probability 0.08, each hourly 90th percentile is below `m`; the probability that the daily maximum exceeds `m` is `1 − 0.92² = 0.1536`, so the 90th percentile of the daily maximum exceeds `m`. Coherent path simulation, not per-hour quantile stacking, is required for peak-functionals.

### Step 10 — Sensitivity: which specification changes the answer?

| Stress/test | Original condition | Alternative | Change in conclusion | Uncertainty / limitation | Decision consequence |
|---|---|---|---|---|---|
| Loss ratio | c_s = 9,000, c_r = 12 → τ* = 0.99867 | c_s = 2,000 → τ* = 0.994 | Target quantile moves from extreme to merely high; empirical estimation becomes feasible | VoLL policy unknown | Reserve nomination falls; shortfall frequency rises by design |
| Reserve price | 12/MW-h | 30/MW-h → τ* = 0.99667 | Slightly lower reserve, still tail-driven | Contract not reviewed | Commercial desk must confirm banding |
| Coverage window | 30 days, naive binomial | 30 days, day-block bootstrap | "Significant undercoverage" becomes "inconclusive" | Dependence structure estimated, not known | Do not retune the model on this evidence |
| Dependence in reconciliation | Independent zones | Comonotonic zones | Area 90th percentile rises from 1,064 to 1,117 MW in the illustrative numbers | True dependence unknown | Reserve may be systematically under- or over-procured by tens of MW |
| Calibration regime | All-season calibration | Summer-only calibration | Heat-hour coverage improves; mild-hour intervals widen | Fewer calibration points | Regime-specific calibration layer required |
| Weather cycle | Cycle B fully decoded | Cycle B partial, cycle A used | Forecast loses recent information; error rises modestly | Not quantified here | Fallback flag must be published |

The highest-leverage sensitivity is the loss ratio, because it changes *which quantile is the product*, not merely how accurate the estimate is. A product team that fixes τ = 0.9 because "90% is standard" has chosen a policy, not a cost optimum, and should say so.

### Step 11 — Workload and capacity sizing for the scoring pipeline

Sizing is part of the analysis because a rolling-origin backtest that cannot finish inside the review cycle will be shortened, and a shortened backtest cannot resolve the tail.

Assume 730 daily refits, each fitting a multi-quantile model on 6 years of hourly data for 6 series. Measured or assumed cost per fit: 54 s single-threaded (assumed; no benchmark was run).

```text
FORMULA [M6]: W_serial = n_fits * t_fit ;  W_parallel = W_serial / w
WHERE: n_fits [fits]; t_fit [s/fit]; w = workers [dimensionless].
NUMERICAL EXAMPLE: W_serial = 730 * 54 = 39,420 s = 10.95 h.
       With w = 8: 39,420 / 8 = 4,927.5 s = 1.369 h.
INTERPRETATION: an 8-worker pool fits the backtest inside one working day.
```

Storage for the published record: 730 days × 24 h × 9 quantiles × 6 series = 946,080 numbers. As 8-byte floats that is `946,080 × 8 = 7,568,640 bytes`, i.e. 7.56864 MB decimal or `7,568,640 / 1,048,576 = 7.2180 MiB` binary. With a 20× overhead for manifests, versions and indexes, provision about 150 MiB. This is trivial; the constraint is not storage but the *auditability* of the manifest chain.

Latency budget for daily issuance: 30 minutes from cut-off to publication. Decompose as 8 min for ensemble ingestion and feature assembly, 6 min for model scoring across 9 quantiles × 6 series × 24 hours, 3 min for reconciliation, 2 min for schema validation and monotonicity checks, 5 min for human review, leaving 6 min of headroom. Those are allocations, not measurements. The critical path is ingestion; a partially decoded cycle must be detected within the first two minutes or the fallback cannot be exercised in time.

### Step 12 — What would falsify the recommendation

The recommendation in Section 7 is falsified if any of the following is observed on a prospective, pre-registered evaluation: (i) day-block-bootstrap 90% interval coverage below 0.85 in heat hours over a full summer; (ii) reserve-cost regret worse than the peak-plus-5% policy on the same days; (iii) hierarchical incoherence exceeding 25 MW between reconciled area quantiles and the sum of zone quantiles in more than 5% of hours; (iv) more than two issuance failures per month caused by ingestion. None of these has been observed, because no evaluation has been run.

**Takeaway:** The arithmetic shows that the decision quantile is near 0.9987, that no 720-hour sample can resolve it, and that coherence and timing functionals need their own methods — three requirements that a single MAE number hides.

## 7. The Finding

The recommended decision is to publish a nine-level quantile set from a calibrated multi-quantile model, but **to derive the reserve nomination from a separately fitted tail model evaluated against an explicit loss ratio**, and to gate the product on regime-specific coverage and on reserve-cost regret against the peak-plus-5% policy — not on average pinball loss. Under the illustrative ledger, the cost-optimal reserve level is the 99.87th percentile (with `12/9000 = 0.001333` shortfall probability), which lies beyond the resolvable range of a 720-hour backtest; the required tail extrapolation is therefore a stated model assumption with reported uncertainty, not an empirical fact. The evidence for any performance claim is absent: no model was trained and no backtest was run, so the recommendation is a design proposal whose acceptance depends on the prospective gates in Section 9. The main boundary is that the entire economic ranking depends on the assumed value of lost load, which is a policy input the analytics team must not choose by itself. The next justified action is a twelve-month, pre-registered backtest with regime-stratified coverage, a day-block bootstrap, and a parallel cost ledger for the incumbent policy.

## 8. System/Solution Design

### 8a. Offline / training / batch path

```text
[Settlement load extract, revision-tagged]
        |
        v
[Interval completeness + duplicate + jump checks] --fail--> [Quarantine batch]
        |
        v
[Point-in-time feature build: lags, calendar, weather-cycle summaries]
        |
        v
[Train/validation/test split by year and operating day]
        |
        v
[Multi-quantile fit -> monotone rearrangement -> calibration layer]
        |
        v
[Evaluation gates: regime coverage, regret vs policy baseline]
        |
        v
[Versioned model + calibration + manifest -> registry]
```

The offline path runs nightly for the daily refit and monthly for the full re-fit with recalibrated tail model. Every artifact is keyed by (model version, calibration window, data snapshot, code revision). The registry holds the exact feature list, because a feature-set change that is not versioned is a silent model change.

### 8b. Online / serving / real-time path

```text
[17:00 cut-off fires]
        |
        v
[Snapshot resolver: select latest complete NWP cycle per variable]
        |                                    |
        | incomplete                          | complete
        v                                    v
[Flag fallback, use prior cycle]      [Assemble features]
        |                                    |
        +----------------+-------------------+
                         v
                [Load model version from registry]
                         v
                [Score 9 quantiles x 6 series x 24 h]
                         v
                [Monotone check] --crossing--> [Rearrange; log count]
                         v
                [Hierarchical reconciliation (joint simulation)]
                         v
                [Schema + monotonicity + coherence validation]
                         |fail
                         v
                [Block publication, alert, fall back to baseline with label]
                         |pass
                         v
                [Publish vN with manifest] -> [Reserve rule] -> [Human review]
                         v
                [Override?] --yes--> [Record reason code + value]
                         v
                [Archive immutable record]
```

Failure behaviour, stated explicitly:

- **Cycle incomplete:** publish with a `fallback_cycle` flag; the reserve rule widens by a documented margin rather than pretending precision.
- **Scoring timeout (>30 min total):** publish the baseline with an explicit `degraded_baseline` label. Never publish a partial quantile set.
- **Coherence failure:** block, do not reconcile silently. Silent reconciliation destroys the audit trail.
- **Model registry unavailable:** fall back to the last-known-good model version pinned in the deployment manifest, and label it.
- **Human override:** allowed, but the override is a first-class record with a reason code; the published record retains both the model value and the override value so that override quality can be evaluated.

No automated shedding exists in this path. The output is advisory to a human with authority.

### 8c. Key design decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| DEC-01 | Single multi-quantile model over nine levels | Nine independent models | Cheaper, coherent by construction | Less flexibility per level | Severe crossing indicates misspecification | Crossing rate above 2% of hours |
| DEC-02 | Reserve from a separate tail model | Reserve from the published 0.9 quantile | Loss ratio 1:750 makes 0.9 indefensible | Extra component to validate | Tail model unvalidated → policy ceiling | Loss-ratio revision |
| DEC-03 | Joint simulation for area coherence | Naive sum of zone quantiles | Quantiles do not add | Simulation cost | Dependence model wrong | Coherence metric breach |
| DEC-04 | Regime-specific calibration | Single all-season calibration | Heat-hour coverage is the gate | More parameters, fewer points | Sparse regime → unstable shift | New regime appears |
| DEC-05 | Human override with reason code | No override; or free-text override | Operators need authority; quality needs measurement | Interface effort | Overrides become the de facto model | Override rate above 15% |
| DEC-06 | Baseline published alongside | Model only | Comparison must be live, not retrospective | Two products to explain | Confusion about which to use | Stakeholder feedback |

### 8d. Architecture artefacts and operational readiness

| Area | Answer for this design |
|---|---|
| Interfaces | Publication schema: operating_day, cut_off_utc, series_id, hour_index, tau, value_mw, model_version, manifest_id, flags[]. Idempotency key = (operating_day, cut_off_utc, model_version) |
| Data/state | Immutable published records; corrections create a new version, never an in-place edit |
| Reliability | SLI: fraction of operating days with a complete published set by 17:30. Target 99.5% over a rolling 90 days, owner: platform lead |
| Resilience | Two ingestion attempts with backoff; cycle fallback; hard publication deadline with baseline degradation |
| Recovery | Replay from manifest for any historical day; replay test quarterly |
| Observability | Ingest completeness per variable, crossing count, coherence gap MW, override rate, publication latency, regime coverage |
| Deployment | Model + calibration + feature schema versioned together; canary on shadow-only days before switching the published model |
| Security | Load data is commercially sensitive; access scoped to named roles; manifests retained for the settlement audit window |
| Economics | Reserve cost and shortfall cost dominate; compute cost is negligible by comparison (7.2 MiB of records; hours of CPU per month) |
| Human workflow | Shift manager may override with a code; commercial desk may deviate from the recommended reserve only with recorded justification |

**Sizing summary with units.** Nine quantiles × six series × 24 hours = 1,296 numbers per issuance; at 8 bytes each, 10,368 bytes of payload, so a year of issuances is `365 × 10,368 = 3,784,320 bytes ≈ 3.61 MiB` binary — trivially small. The binding resource is the tail model's fitting window and the human review minutes, not storage or compute.

**Interview cue.** Ask what happens if the weather supplier delivers a partial cycle at 16:55, if the model registry is down at 17:02, and if the shift manager overrides every hour for a week.

**Acceptance for this section.** A reader can follow both a successful and a failed issuance and can say who owns recovery.

**Takeaway:** Every failure path has a labelled degradation; unlabelled degradation is how a forecast product loses its audit trail.

## 9. Evaluation & Validation

### 9.1 Primary metric

| Evaluation field | Content |
|---|---|
| Primary metric (publication) | Mean pinball loss over τ ∈ {0.1…0.9} in MW, with regime-stratified coverage as a gate |
| Primary metric (decision) | Reserve-cost regret per operating day in currency: `Cost(model policy) − Cost(best fixed policy in the comparison set)`, computed on realised outcomes |
| Numerator / denominator | Pinball: sum of losses / number of scored intervals. Regret: difference of daily costs / one day |
| Population | All hours of the held-out year, plus a summer-heat and winter-cold subset defined by a temperature rule fixed before evaluation |
| Window | One full year for coverage; per-day for regret; 90-day rolling for the operational SLI |
| Comparator | Adjusted persistence point forecast with peak-plus-5% reserve policy; and the same model with a fixed 0.9 quantile policy |
| Guardrails | Publication by 17:30; monotone output; coherence gap ≤ 25 MW; override rate ≤ 15%; no use of post-cut-off data |
| Test design | Rolling-origin backtest with daily refits, pre-registered before the held-out year is touched |
| Sample | 8,760 hours per series; 6 series; block length for bootstrap chosen from error autocorrelation |
| Uncertainty | Day-block bootstrap, 1,000 resamples, 90% interval; regime-level intervals reported separately |
| Validity threat | Serial dependence inflating apparent precision; regime shift between calibration and test; truth revision differences |
| Artifact | Not executed; no run ID exists |
| Decision | **Not executed** — this is a proposal |

### 9.2 Pass/fail rule

Proposed, with the explicit caveat that thresholds were chosen by the author for teaching purposes and were not pre-registered with stakeholders:

- Publication coverage: 90% nominal interval must achieve ≥ 0.85 in each of the summer and winter regimes over a full season, with the lower bootstrap bound ≥ 0.82.
- Decision regret: mean daily regret versus the better of {peak+5%, fixed 0.9 policy} must be ≤ 0 on the held-out year, with the bootstrap interval not excluding zero in the wrong direction.
- Coherence: gap ≤ 25 MW in ≥ 95% of hours.
- Operational: ≥ 99.5% of days published by 17:30.

The gate is deliberately two-sided in a specific sense: a model that beats the policy on regret but fails regime coverage is *not* passed, because coverage failure means the published product is misleading even if the reserve number happened to be right. That combination is common and is exactly the evaluation-mismatch this case is about.

### 9.3 What could make a bad solution look good

1. **Scoring against revised load that was itself influenced by the forecast.** If settlement corrections are entered by staff who consult the published forecast, the truth is contaminated.
2. **Evaluating only on mild days.** Mild days dominate the sample and are easy; the tail is the product.
3. **Using naive binomial intervals.** They overstate precision by roughly a factor of `sqrt(n_days)` when errors are day-clustered; the earlier calculation shifted p from 0.033 to about 0.17.
4. **Tuning on the same year used for the gate.** Hyperparameters chosen on the test year convert the gate into a training metric.
5. **Reporting average regret over a period that excludes the two shortfall events.** The mean is dominated by ordinary days; the tail events are the entire economic story.

### 9.4 Uncertainty and identification limits

No causal claim is made anywhere in this case. The comparison between the model policy and the incumbent policy is a *decision* comparison under a forecast, not an estimate of the effect of adopting the model: operator behaviour, market prices and weather would all differ in a real adoption. Even a clean retrospective regret comparison is a conditional calculation, not a causal effect.

**Interview cue.** Ask which test could falsify the preferred solution, and how the evaluation could make a bad solution look good.

**Acceptance for this section.** An independent reader can state what would fail the case, what was actually executed (nothing), and what remains unvalidated.

**Takeaway:** Two gates, not one: calibration protects the published claim, regret protects the money, and neither substitutes for the other.

## 10. Action Taken

| Action field | Record |
|---|---|
| Recommendation | Adopt the calibrated quantile publication plus a separately validated tail-based reserve rule, gated as in Section 9 |
| Approval/status | Not recorded; proposed only. No approval body was consulted in this exercise |
| Implementer and handoff | Proposed: grid analytics team builds, platform team owns the pipeline, commercial desk owns reserve nomination |
| Effective scope | None; nothing deployed |
| Release strategy | Proposed shadow mode for one full season, then a canary on a single zone pair for one month, then staged area-wide |
| Stop/rollback | Trigger: regime coverage below 0.82 for two consecutive months, or coherence breach in more than 10% of hours. Authority: shift manager plus analytics lead. Rollback is a publication switch to the baseline with a recorded reason |
| Excluded actions | Automated shedding; automated reserve purchase; intraday market bidding; customer-level forecasting |

**No action taken as of 2026-11-10.** This is a design and analysis exercise. The organisation-level decision to fund the twelve-month backtest is the next required step, and it is outside the analytics team's authority.

**Interview cue.** Explain what needed approval and how operations would take ownership; distinguish a pilot from a proposal.

**Acceptance for this section.** Implementation claims have an action record; nothing here is presented as deployed.

**Takeaway:** The honest status of this case is "proposed with an unexecuted validation plan", and the case says so in every section that could drift.

## 11. Outcome & Results

**Outcome not yet measured.** No baseline or treatment value exists. The table below is a measurement plan, not a results table.

| Metric | Baseline: value/window/N | Treatment/after: value/window/N | Absolute change | Relative change | Uncertainty | Evidence/source | Limitation |
|---|---|---|---|---|---|---|---|
| Mean pinball, τ 0.1–0.9 (MW) | Not measured | Not measured | — | — | Day-block bootstrap, planned | None | Requires held-out year |
| 90% interval coverage, summer | Not measured | Not measured | — | — | Regime-stratified bootstrap, planned | None | Tail events rare |
| 90% interval coverage, winter | Not measured | Not measured | — | — | as above | None | as above |
| Reserve-cost regret (currency/day) | Peak+5% policy cost, not measured | Model policy cost, not measured | — | Undefined until both are measured | Paired day-level bootstrap, planned | None | Depends on assumed VoLL |
| Shortfall events per season | Not measured | Not measured | — | — | Poisson-style interval not appropriate if clustered | None | Events are dependent on weather regimes |
| Publication-by-17:30 rate | Not measured | Not measured | — | — | 90-day rolling proportion | None | Operational, not modelling |

**Source declaration:** all values in this section are unavailable by design. The case contains no observed, published, or simulated performance figures, and the illustrative arithmetic in Section 6 must not be read as a forecast of improvement.

**Interview cue.** Ask what improved, what worsened, and whether any reported change could be attributed to the intervention. Here the correct answer is that nothing has been measured and attribution is not even well defined yet.

**Acceptance for this section.** Every headline claim is scoped and consistent with `evidence_level: not-measured`.

**Takeaway:** An empty results table with a named owner and window is more useful than a filled one with invented numbers.

## 12. Failure Modes & What Went Wrong

These are designed failure modes with constructed examples. None is an observed incident. Each has a distinct mechanism, so that a reader can discriminate between them rather than lumping them as "bad forecast".

| Failure ID | Trigger and symptom | Detection / discriminating test | Confirmed cause or hypothesis | Impact/blast radius | Containment | Prevention / owner | Verification / limitation |
|---|---|---|---|---|---|---|---|
| F-01 | Evening forecast precise and biased high; error collapses when the observed-weather table is removed | Feature-availability audit: recompute with `available_at ≤ cut-off` filter | Post-cut-off observed temperature joined as "temperature" | Whole published record invalid; reserve over-procured | Withdraw affected versions; recompute | Source-type column and cut-off rejection at the store | Replay with filter; constructed |
| F-02 | Published quantiles cross; 0.8 below 0.7 | Monotonicity validation at schema write | Independent per-level fitting with unstable tail | Downstream misreads; reserve inconsistent | Rearrange; flag; refit | Multi-quantile objective or post-hoc sort with crossing metric | Crossing count per issuance; constructed |
| F-03 | Overall coverage 0.90 but heat-hour coverage 0.78 | Regime-stratified coverage on held-out summer | Calibration window lacked a heat regime; additive shift under-corrects | Public-safety messaging unreliable exactly when needed | Widen heat regime by documented factor; re-calibrate | Regime-specific calibration; tail model | Coverage by regime each season; constructed |
| F-04 | Area quantiles 40–60 MW below the sum of zone quantiles | Coherence check: |Σ zone q − area q| | Naive per-zone independence in the joint model | Reserve under-procured for correlated heat | Joint simulation; document dependence assumption | Coherence gap distribution; constructed |
| F-05 | Same day scores differently on consecutive runs | Truth version diff: D+1 vs settled | Settlement revision changed the label after publication | Apparent model drift that is a data artefact | Score both, publish both, never mix | Freeze the truth version per evaluation window | Revision log audit; constructed |
| F-06 | Large level error on 12–18 days a year | Calendar join audit against published holiday table | Holiday table not updated; operating-day index wrong | Systematic reserve error on those days | Manual override | Quarterly calendar review with recorded owner | Compare holiday-day error to ordinary days; constructed |
| F-07 | Model outputs exactly 0 MW for a zone for several hours | Input schema diff: null count per feature | Missing value filled with 0, which is a physically meaningful MW value | Severe under-procurement for that zone | Block publication; fill with explicit null path | Null-aware model or rejection; never fill MW with 0 | Null-injection test; constructed |
| F-08 | Override rate 40%; model accuracy unmeasurable | Override record analysis: count, reason codes, delta MW | Operators compensating for a systematic model bias, unlogged in free text | Published vs used forecast diverge | Introduce structured codes; compare overridden vs not | Override dashboard with reason taxonomy | Overridden-hour error vs clean-hour error; constructed |
| F-09 | Good hourly MAE, poor daily-maximum error | Score both functionals on the same days | Evaluation scored the wrong functional | Reserve sizing wrong despite a good scoreboard | Add peak-functional metric | Two-function gate in evaluation | Recompute both from the same run; constructed |
| F-10 | 23- and 25-hour days produce shape errors and offset misalignment | Interval-count check against operating-day contract | Hard-coded 24-hour vectors; naive local timestamps | Two days a year with a one-hour offset in the whole vector | Rebuild from the operating-day contract; UTC keys | Contract-driven shapes; DST test in CI | Synthetic 23/25-hour day test; constructed |

### 12.1 Deeper look at F-03, the most consequential mode

The mechanism deserves its own trace because it is the failure most likely to survive a superficially good evaluation. Suppose the calibration window is a mild spring: 60 days × 24 h = 1,440 hours, and the additive 0.9-quantile shift is +18 MW. In a heat dome, the model's *level* error is small (the weather features carry the mean shift) but its *spread* is far too narrow because the training window never contained a comparable temperature. The additive shift applies the same +18 MW, so the interval misses in most of the afternoon. Overall annual coverage can still land near 0.90 if the heat dome occupies only 40 hours out of 8,760: `(8,760 × 0.905 − 40 × 0.5 + 40 × 0.22)/8,760` — the construction shows how a 40-hour catastrophe is arithmetically invisible in an annual average.

Numbers for the construction: 8,720 ordinary hours at 0.905 coverage = 7,891.6 covered; 40 heat hours at 0.22 coverage = 8.8 covered; total covered 7,900.4 out of 8,760 → `7,900.4 / 8,760 = 0.9019`. The annual number looks *excellent* while the product fails completely in the only hours that matter for safety. This is the single clearest instance of evaluation-mismatch in the case, and it is the reason the gate is regime-stratified.

### 12.2 Deeper look at F-07, the silent one

A null-filled MW feature is worse than a missing value because zero is a legal load. A null-aware model can route the missingness into a separate indicator; a naive fill makes the model believe a zone is de-energised. The discriminating test is a null-injection test in the offline pipeline: replace 5% of one weather variable with nulls and check that the publication either flags or blocks. If the output simply shifts by a plausible amount, the fill is silent and will eventually be triggered by a real supplier outage.

**Interview cue.** Present F-01 or F-03 as a symptom only, and require competing hypotheses, a discriminating diagnostic, a safe mitigation, and a prevention test.

**Acceptance for this section.** A reader can distinguish at least three mechanisms that all present as "the forecast was wrong" and knows which check separates them.

**Takeaway:** Four different failures — vintage leakage, crossing, regime miscalibration, and coherence — all look like "bad forecast" to an operator; only instrumented diagnostics separate them.

## 13. Transferable Lesson

| Pattern field | Content |
|---|---|
| Canonical pattern | `evaluation-mismatch` — the scored quantity is not the decision quantity |
| General principle | Choose the scoring rule from the decision's loss structure; publish separately any product whose target probability is set by convention rather than by cost |
| Preconditions | A decision with an explicit, roughly linear cost of under- and over-provision; access to a defensible loss ratio; a sample large enough for the chosen tail or a stated tail model |
| Mechanism | Under a linear asymmetric loss, the optimal provision level is the quantile at `1 − c_over/c_under`; the score that selects that quantile is pinball loss at that level, not average error |
| Supporting evidence | The derivation in M3–M4 and the arithmetic showing the optimum near 0.9987; no empirical validation exists |
| Avoid when | The loss is non-linear or discontinuous (step contracts, legal thresholds), when the decision is intertemporal (multi-hour ramp constraints), or when the loss ratio is politically contested and the team has no authority to set it |
| Portability test | Ask: "If I shift provision by one unit, what does the decision owner gain and lose, and can I express both in the same unit?" If not, do not pretend a quantile solves it |
| Adaptation | In water-resource planning, the same structure appears as reservoir hedging with asymmetric flood/drought costs; in staffing, as over/under-staffing cost with a convex penalty; in inventory, as newsvendor critical ratio |

The lesson is not "use quantile regression". It is that **the probability level is a business parameter**, derived from costs, and a team that accepts a probability level from a template has silently accepted someone else's cost model. The corollary is a discipline: publish the loss ratio and its owner next to the chosen level, and re-derive it when contracts change.

**Interview cue.** Transfer the approach to a contrasting domain and identify the first assumption that breaks.

**Acceptance for this section.** The lesson remains useful without the utility context and names a condition under which following it is harmful.

**Takeaway:** The probability level is a business parameter with an owner, not a modelling default.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

| Relationship | Actual target | Mechanism shared or changed | Boundary learned |
|---|---|---|---|
| `contrasts-with` | [Energy load forecasting, time-series family](../../../01_production_ml_case_library/10_time_series_forecasting/Case_Studies/03_Energy_Load_Forecasting.md) | Both forecast hourly load; that case centres hourly point/quantile accuracy and peak pitfalls; this case centres the reserve decision, loss ratio, and hierarchical coherence | Accuracy evaluation does not determine procurement policy |
| `uses-method` | Energy forecast metrics (planned card; not yet authored) | Pinball, coverage, PINAW, peak functionals | Metric validity requires the same population and revision |
| `uses-method` | Grid data governance (planned card; not yet authored) | Point-in-time provenance, cut-off enforcement | Governance is a precondition for a falsifiable record |
| `extends` | Asset AI safety gate (planned card; not yet authored) | Human authority over automated recommendation | Forecast advice is not an operational command |
| `uses-method` | [Course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) | Formula anatomy and evidence labels | Teaching arithmetic is not a benchmark |

### 14b. What this case leaves open

| Open question | Why current evidence is insufficient | Data/test that would resolve it | Priority / trigger | Owner role |
|---|---|---|---|---|
| What is the defensible value of lost load for this balancing area? | It is a policy input; no source is available in this exercise | Regulatory or board-approved reference; sensitivity study | High, before any reserve change | Regulatory analyst |
| Does additive conformal calibration hold across regimes, or is a scale-aware variant required? | No residual data was examined | Regime-stratified calibration experiment on 3 years | High | Modelling owner |
| What is the true dependence structure between zones during heat events? | Requires joint observations of simultaneous extremes | Copula or empirical joint-tail study over 8 years | Medium | Analytics lead |
| Is the daily maximum or the hourly energy balance the binding physical constraint? | Depends on network configuration and contracts | Operations review; constraint study | Medium | Network planning |

### 14c. Intense interview pack — questions with checked answers

**Q-01 / Level 1 (framing) / skill: decision framing**

*Candidate prompt:* "We want to forecast load better. Where do you start?"

*Facts given:* daily 17:00 issuance, reserve procurement, human shedding authority.

*Should clarify:* who consumes the forecast, what action it changes, what the loss is for under- and over-provision, and what the baseline is.

*Strong answer:* Start from the decision, not the model. Name the action (reserve MW per hour), the owner (shift manager/commercial desk), the cost asymmetry, and the baseline (adjusted persistence with peak+5%). Then choose the target functional — a high quantile for reserve, an interval for publication — and only then the model. Reasoning: the loss ratio sets which quantile is the product, so model choice is downstream of cost structure.

*Acceptable alternatives:* Starting from data availability is acceptable if the candidate then forces the decision framing before choosing a metric.

*Follow-up:* "The commercial desk says 90% is the standard interval. What do you tell them?"

*Red flags:* Jumping to LSTM/Transformer architecture; proposing MAPE as the objective; no mention of asymmetric cost.

*Scoring:* 0 = model-first with no decision; 1 = names the decision but no cost asymmetry; 2 = derives the target quantile from costs; 3 = also states the policy-vs-cost distinction and who owns the loss ratio.

*Answer support:* Sections 1, 3, 6 Step 5.

**Q-02 / Level 2 (calculation) / skill: pinball loss**

*Candidate prompt:* "At τ = 0.9, actual 842 MW, forecast 812 MW. Compute the pinball loss and explain the asymmetry."

*Facts given:* y = 842, q = 812, τ = 0.9.

*Should clarify:* which series and revision, and that the loss is in MW.

*Strong answer:* Under-forecast branch: `0.9 × (842 − 812) = 27 MW`. If the same 30 MW error were an over-forecast, loss would be `0.1 × 30 = 3 MW`. Reason: the check function weights the tail probability τ on under-prediction and 1−τ on over-prediction, so at high τ under-prediction is punished nine times more. At τ = 0.5 the loss equals half the absolute error, which is the implementation sanity check.

*Acceptable alternatives:* Expressing the result as 27 "MW of loss units" is fine; calling it energy is wrong.

*Follow-up:* "Why is pinball a proper scoring rule?" Answer: its expectation is minimised at the true conditional quantile, since `d/dq E[L] ∝ P(Y < q) − τ`.

*Red flags:* Using `0.1 × 30` for an under-forecast; reporting 30 MW as the loss; multiplying by hours without a stated duration.

*Scoring:* 0 = wrong branch; 1 = correct number, no reasoning; 2 = correct with derivation; 3 = also connects propriety to the quantile definition and the τ = 0.5 check.

*Answer support:* Section 6 Step 2, M1.

**Q-03 / Level 2 (calculation) / skill: optimal provision under asymmetric cost**

*Candidate prompt:* "Reserve costs $12 per MW-hour and unserved energy costs $9,000 per MWh. What probability level should the reserve cover?"

*Facts given:* c_r = 12, c_s = 9,000, one hour, linear costs.

*Should clarify:* risk neutrality, single period, no intertemporal coupling, and that the reserve is capacity not energy.

*Strong answer:* Differentiate `c_r C + c_s E[(Y−C)^+]`: `c_r − c_s P(Y > C) = 0`, so `P(Y > C*) = 12/9000 = 0.001333` and the level is `1 − 0.001333 = 0.998667`. Check: at a lower C the marginal shortfall probability exceeds the cost ratio, so increasing C pays. State that a step-cost contract or a risk-averse owner changes the answer.

*Acceptable alternatives:* A policy ceiling such as "no more than 0.5% shortfall probability" is acceptable if labelled as policy, not derived optimum.

*Follow-up:* "If VoLL falls to $2,000/MWh?" → `12/2000 = 0.006` → 99.4th percentile.

*Red flags:* Answering 90%; answering "the 95th percentile because that is conventional"; ignoring the derivative and guessing.

*Scoring:* 0 = 90% or a guess; 1 = recognises asymmetry but wrong algebra; 2 = correct derivation and number; 3 = also states feasibility limits of empirical estimation at that level.

*Answer support:* Section 6 Step 5, M3–M4.

**Q-04 / Level 3 (judgement) / skill: sample-size and tail feasibility**

*Candidate prompt:* "You have 720 held-out hours. Can you empirically estimate the 99.87th percentile?"

*Facts given:* n = 720, target level 0.998667.

*Should clarify:* whether the hours are independent and whether a parametric tail is permitted.

*Strong answer:* No. The largest resolvable empirical probability is `1 − 1/720 = 0.998611`, which is *below* the target, so the required quantile lies beyond the sample's last order statistic. Even at 0.9986 the estimate is a single order statistic with large variance, roughly `sqrt(τ(1−τ)/n)/f(q)` in standard error. Use a fitted tail model (parametric or extreme-value) with reported uncertainty, or restate the decision as a policy ceiling that the data can actually police.

*Acceptable alternatives:* Bootstrapping the last order statistics gives an honest interval but cannot create information beyond the sample.

*Follow-up:* "What if you pool eight years?" → 70,080 hours makes 0.99986 resolvable, but seasonality means the pooled tail mixes regimes; the relevant question becomes conditional tail behaviour in heat hours.

*Red flags:* "Just take the max"; "use the 99th percentile as a proxy without saying so"; claiming a confidence interval from a single extreme point.

*Scoring:* 0 = yes; 1 = recognises rarity but not the resolvability bound; 2 = computes the bound and proposes a tail model; 3 = also discusses regime conditioning and uncertainty reporting.

*Answer support:* Section 6 Step 5.

**Q-05 / Level 2 (calculation) / skill: coverage uncertainty with dependence**

*Candidate prompt:* "Coverage was 629/720 = 87.4% against 90%. Is the model miscalibrated?"

*Facts given:* 30 days of hourly intervals.

*Should clarify:* dependence structure, whether the window is representative, and the pre-declared threshold.

*Strong answer:* Compute both intervals. Naive binomial: `p = 0.873611`, `SE = sqrt(0.873611 × 0.126389 / 720) = 0.012384`, `z = −2.13`, p ≈ 0.033 — apparently significant. Day-clustered: with 30 day-level proportions and SD 0.105, `SE = 0.105/sqrt(30) = 0.01917`, `z = −1.38`, p ≈ 0.17 — inconclusive. Hours within a hot afternoon are not independent, so the clustered interval is the defensible one. Conclusion: insufficient evidence; extend the window and stratify by regime.

*Acceptable alternatives:* A block bootstrap with a data-chosen block length is equivalent in spirit.

*Follow-up:* "How many independent days would you need to detect a 2.6 percentage-point deficit at 80% power?" → roughly `(2.8 × 0.105/0.026)² ≈ 128` days; state the approximation.

*Red flags:* Quoting the binomial p-value as decisive; concluding "model is fine" from a wide interval; ignoring regime.

*Scoring:* 0 = yes/no with no interval; 1 = binomial only; 2 = both intervals and the correct conclusion; 3 = also discusses power and the pre-registration problem.

*Answer support:* Section 6 Step 4, M2.

**Q-06 / Level 3 (design) / skill: leakage audit**

*Candidate prompt:* "Your new model beats the baseline by 40% on MAE. What is the first thing you check?"

*Facts given:* pipeline joins load, weather, calendar, outage logs.

*Should clarify:* the cut-off definition and the availability timestamp of every feature.

*Strong answer:* Recompute with a hard `available_at ≤ cut-off` filter and diff the feature matrix. The four likely leaks here are revised-load lags, observed weather joined as forecast weather, retro-corrected outage logs, and target-day calendar inference. A 40% jump in a domain with strong persistence is implausible; persistence baselines are hard to beat by more than single-digit percentages on hourly load. The discriminator is the ablated run, not intuition.

*Acceptable alternatives:* Starting with a target-shuffle test also works, but it catches different leaks (it will not catch weather-observation leakage because that is a legitimate predictor of the target, just unavailable at decision time).

*Follow-up:* "The weather table has both forecast and observed rows. How do you make the join safe?" → source-type column, availability timestamp, and a rejection path at the feature store.

*Red flags:* "Check the learning rate"; "add more data"; accepting the improvement.

*Scoring:* 0 = accepts the result; 1 = mentions leakage generically; 2 = names the cut-off filter and at least two concrete leaks; 3 = explains why the magnitude is implausible and designs the rejection.

*Answer support:* Section 5d, Section 12 F-01.

**Q-07 / Level 3 (design) / skill: hierarchical coherence**

*Candidate prompt:* "Should the area 90th percentile equal the sum of the zone 90th percentiles?"

*Facts given:* five zones, additive load, correlated weather.

*Should clarify:* dependence between zones and whether the area series is modelled directly.

*Strong answer:* No, except under comonotonicity. With two zones each at +100 MW with probability 0.10, each zone 90th percentile is 0 and the area 90th percentile is 100 MW, since `P(area = 0) = 0.81 < 0.90`. Quantiles are not additive. Either simulate joint paths and take the area quantile of the simulated area series, or reconcile explicitly and label the result as reconciled. Naive summation over- or under-states depending on dependence; the illustration gave 1,117 MW summed against 1,064 MW joint.

*Acceptable alternatives:* Bottom-up with an empirical copula is acceptable; top-down disaggregation is acceptable if zone-level instructions tolerate it.

*Follow-up:* "The zones are in the same weather system. What happens to the gap?" → dependence rises toward comonotonicity, so the gap shrinks toward zero.

*Red flags:* "Quantiles add"; "just average the zones"; ignoring the reconciliation question entirely.

*Scoring:* 0 = yes; 1 = says no without mechanism; 2 = counterexample plus a reconciliation approach; 3 = also quantifies the gap and states the dependence assumption's owner.

*Answer support:* Section 6 Step 8.

**Q-08 / Level 3 (diagnosis) / skill: ambiguous symptom**

*Candidate prompt:* "Overall coverage is exactly 90%, but the control room says the band is useless. What is happening?"

*Facts given:* annual aggregate coverage nominal; complaints concentrated in heat events.

*Should clarify:* the regime composition, the coverage by regime, and whether the band's width is actionable.

*Strong answer:* Aggregate coverage can be nominal while regime coverage is far off, because ordinary hours dominate. With 8,720 ordinary hours at 0.905 and 40 heat hours at 0.22, annual coverage is `7,900.4/8,760 = 0.9019` — apparently perfect, catastrophic in the only hours that matter. Also check width: a band that is nominally calibrated but 300 MW wide conveys nothing; PINAW or width relative to load is the second diagnostic. The fix is regime-specific calibration, not more data.

*Acceptable alternatives:* Checking for crossing or for a stale model version are reasonable first probes, but the given symptom (aggregate fine, regime bad) points to composition.

*Follow-up:* "How would you report this to the shift manager?" → coverage by regime with event counts, plus the width distribution, not the annual average.

*Red flags:* "Coverage is 90%, so it is calibrated"; blaming operators; proposing a wider band everywhere without measuring the cost.

*Scoring:* 0 = denies the complaint; 1 = suspects drift only; 2 = identifies regime composition with arithmetic; 3 = also distinguishes calibration from width and proposes the reporting change.

*Answer support:* Section 12.1.

**Q-09 / Level 2 (application) / skill: calendar and DST correctness**

*Candidate prompt:* "Twice a year your forecast vector is off by an hour. Why, and what is the fix?"

*Facts given:* hourly publication, local operating day, UTC storage.

*Should clarify:* how the operating day is defined and whether timestamps are naive local.

*Strong answer:* The operating day has 23 or 25 hours at DST transitions. A pipeline that builds 24-slot vectors from a naive local clock either duplicates or drops an hour, and everything after the transition shifts by one slot. Fix: key on UTC interval start, carry an operating-day index and expected hour count, and derive shapes from the contract. Add a synthetic 23/25-hour day test. Check: `23 + 25 = 48` hours across the two transition days, not 48 either way; each day must be validated independently.

*Acceptable alternatives:* Storing local time with explicit offsets is acceptable if the offset is part of the key.

*Follow-up:* "The market settles in local time. Does that change your answer?" → no; it changes the mapping layer, not the storage key.

*Red flags:* "Use 24 and pad"; "drop the extra hour"; ignoring the settlement mismatch.

*Scoring:* 0 = no idea; 1 = mentions DST but not the mechanism; 2 = mechanism plus fix; 3 = also the test and the settlement-mapping consequence.

*Answer support:* Section 5a grain warning, Section 12 F-10.

**Q-10 / Level 3 (evaluation) / skill: designing the gate**

*Candidate prompt:* "Design an acceptance gate for this forecast product."

*Facts given:* one-year held-out period, two regimes, human override, commercial consumer.

*Should clarify:* what the product is used for, who signs off, and which metric is primary for which consumer.

*Strong answer:* Two gates. Publication gate: regime-stratified coverage with a lower bootstrap bound, plus a width measure, plus monotonicity and coherence checks. Decision gate: paired daily reserve-cost regret against the better of the incumbent policy and a fixed-level policy, with the interval not excluding zero in the wrong direction. Operational gate: publication-by-deadline rate. Pre-register thresholds and evaluation window before touching the held-out year. A model that passes decision regret but fails regime coverage is rejected, because the published claim would be misleading.

*Acceptable alternatives:* Adding an override-rate guardrail is reasonable; dropping the width measure is acceptable only if the band is never shown to non-experts.

*Follow-up:* "Who owns the gate?" → analytics lead for the computation, shift manager for the operational criteria, commercial desk for the cost model.

*Red flags:* Single MAE threshold; tuning on the gate year; no uncertainty; no owner.

*Scoring:* 0 = one metric; 1 = multiple metrics, no uncertainty; 2 = two gates with bootstrap and pre-registration; 3 = also the conflict rule and named owners.

*Answer support:* Section 9.

**Q-11 / Level 3 (debugging) / skill: competing causes**

*Candidate prompt:* "Reserve shortfalls doubled last month while pinball loss was flat. Diagnose."

*Facts given:* flat average pinball, doubled shortfall events.

*Should clarify:* what changed — weather regime, contract, load composition, reconciliation, reserve policy, or the tail model version.

*Strong answer:* Flat average loss with more shortfalls means the change is in the tail or in the policy, not in the bulk. Rank hypotheses: (1) regime shift into a temperature range with poor tail calibration — check coverage by regime; (2) reserve policy or contract change — check the nomination record against the recommended level; (3) hierarchical reconciliation change altering area quantiles — check the coherence gap series; (4) tail model version change — check the registry; (5) demand composition change (a new industrial load) — check zone-level residuals with the step-change detector. The discriminating test is regime-stratified coverage plus a nomination-versus-recommendation diff, which separates model from process in one query.

*Acceptable alternatives:* Starting with the nomination record is also defensible because it is the cheapest check and shortfalls require both a forecast error and a procurement choice.

*Follow-up:* "Coverage by regime is fine and nominations match recommendations. Now what?" → the tail model's conditional shape is wrong despite correct marginal coverage; check the joint behaviour during the ramp and the dependence between hours.

*Red flags:* Retraining immediately; blaming weather without checking; no separation of model from process.

*Scoring:* 0 = retrain; 1 = one hypothesis; 2 = ranked hypotheses with discriminating checks; 3 = also the process-versus-model separation and the joint-tail follow-up.

*Answer support:* Sections 6, 9, 12.

**Q-12 / Level 2 (calculation) / skill: capacity planning for the backtest**

*Candidate prompt:* "730 daily refits at 54 seconds each, 8 workers. How long, and what would you cut?"

*Facts given:* serial fit time 54 s; 8 workers.

*Strong answer:* `730 × 54 = 39,420 s = 10.95 h` serial; `39,420 / 8 = 4,927.5 s = 1.37 h` parallel. Check: `8 × 4,927.5 = 39,420`. If the review window is shorter, reduce the refit frequency to weekly (104 refits, `104 × 54 = 5,616 s` serial, 11.7 min at 8 workers) but then the backtest cannot detect day-to-day recalibration effects; that trade must be stated rather than hidden.

*Acceptable alternatives:* Fitting once and reusing across the window is acceptable only if the evaluation explicitly studies a fixed model, not a refit policy.

*Follow-up:* "Storage for the published record?" → `730 × 24 × 9 × 6 = 946,080` values × 8 bytes = 7,568,640 bytes ≈ 7.22 MiB, so storage is not the constraint; auditability is.

*Red flags:* Unit errors (treating 39,420 s as minutes); ignoring the evaluation-validity cost of fewer refits.

*Scoring:* 0 = wrong units; 1 = serial only; 2 = parallel plus a check; 3 = also the validity trade of reducing refits.

*Answer support:* Section 6 Step 11, M6.

**Q-13 / Level 2 (application) / skill: monotonicity repair**

*Candidate prompt:* "q_0.7 = 860 and q_0.8 = 852 for the same hour. What do you do, and is sorting legitimate?"

*Facts given:* two crossing quantiles from independent fits.

*Strong answer:* The implied mass between the levels is `852 − 860 = −8 MW`, which is impossible. Sorting (rearrangement) yields a monotone vector and is legitimate because rearrangement weakly reduces pinball loss at every level — a known result, not a heuristic. But severe crossing means the τ-assignment is unreliable, so the right response is to refit with a monotone/multi-quantile objective and track a crossing-rate metric. Sorting is a safety net, not a fix.

*Acceptable alternatives:* Constrained fitting from the start is better if the library supports it.

*Follow-up:* "Does sorting change the reserve number?" → yes, it can: reading the 0.8 level before sorting would have procured 8 MW less than reading the 0.7 level, an inconsistency the consumer cannot see without validation.

*Red flags:* Averaging the two; dropping one; claiming sorting changes the fitted distribution's information content.

*Scoring:* 0 = ignore; 1 = sort without justification; 2 = sort with the loss argument and a crossing metric; 3 = also the downstream reserve consequence and the refit recommendation.

*Answer support:* Section 6 Step 7.

**Q-14 / Level 3 (judgement) / skill: changed constraint**

*Candidate prompt:* "Regulation now requires that the utility demonstrate a shortfall probability no greater than 0.1% for any operating day, regardless of cost. What changes?"

*Facts given:* a hard probability ceiling; existing forecast and reserve process.

*Should clarify:* the aggregation window (per hour or per day), and whether the ceiling applies to the forecast's own stated probability or to a demonstrated outcome frequency.

*Strong answer:* A hard ceiling changes the objective from cost minimisation to constrained minimisation: procure at least the 99.9th percentile of the *day-level maximum*, not the hourly demand, because a day-level constraint binds on the maximum. That requires coherent joint paths (per-hour quantiles cannot be stacked), a tail model validated at 0.999, and an explicit treatment of the fact that a 720-hour sample cannot estimate 0.999. The cost consequence may be large and must be quantified, not hidden. Also, "demonstrate" implies a monitoring protocol: report the realised frequency of exceedance by season with an interval, and define what happens when the ceiling is breached (a reportable event, not a silent retune).

*Acceptable alternatives:* A ceiling on hourly probability with a stated union bound is acceptable if the bound is computed and its conservatism acknowledged.

*Follow-up:* "The day-level maximum distribution is what binds. How do you estimate it?" → simulate joint hourly paths from the model and its dependence structure, take each path's maximum, and estimate the tail of that derived variable.

*Red flags:* Reading the 99.9th percentile of hourly demand; ignoring the union-over-hours effect; treating the requirement as a reporting formality.

*Scoring:* 0 = keeps the cost optimum; 1 = raises the quantile level only; 2 = distinguishes hourly from daily maximum and names the tail problem; 3 = also the monitoring protocol and cost consequence.

*Answer support:* Sections 6 Steps 5 and 9, Section 9.

**Takeaway:** The interview pack tests whether a candidate can separate the publication product from the decision product, and whether they know when the data cannot support the number they were asked for.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Energy / distribution grid operations; day-ahead probabilistic load forecasting for reserve procurement |
| Problem type | Probabilistic forecasting feeding a constrained hourly allocation decision (reserve MW), issued daily at 17:00 |
| Method | Multi-quantile boosted model on frozen weather-ensemble and calendar snapshot, monotone rearrangement, joint-path reconciliation, regime-specific calibration, separate tail model for the decision quantile |
| Primary result | No performance measured. Derived: cost-optimal reserve level ≈ 99.87th percentile at a 12:9,000 cost ratio, beyond the resolvability of a 720-hour backtest |
| Key lesson | The probability level is a business parameter derived from the loss ratio; publication calibration and reserve-cost regret are different gates and neither substitutes for the other |
| Tags | `evaluation-mismatch` (primary); `distribution-shift`, `human-in-the-loop` (secondary) |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-15-01-001 / v1.0 / 2026-11-10 |
| Evidence level | `not-measured`; all arithmetic is illustrative teaching values, independently checked |
| Implementation stage | Proposed design; shadow evaluation not scheduled; no deployment |
| Baseline / primary metric | Adjusted persistence + peak-plus-5% reserve; pinball loss over τ 0.1–0.9 (MW) for publication, reserve-cost regret (currency/day) for the decision |
| Key limitation | The economic ranking depends on a value of lost load that is a policy input, unavailable here; no tail model has been fitted or validated |
| Review trigger / owner | Loss-ratio, reserve contract, NWP supplier, settlement revision, or calendar contract change / grid operations analytics lead |
