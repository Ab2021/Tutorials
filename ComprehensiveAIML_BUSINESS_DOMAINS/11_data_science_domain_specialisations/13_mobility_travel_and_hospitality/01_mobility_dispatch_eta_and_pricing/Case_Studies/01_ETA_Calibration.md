---
case_id: "CASE-13-01-001"
case_title: "ETA calibration and arrival-interval publication"
case_version: "2.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Mobility, travel and hospitality / on-demand transport dispatch"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["distribution-shift", "cost-latency-tradeoff"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "rider-experience product owner with dispatch operations"
source_index: "./INDEX.md"
review_trigger: "Change in the promised-arrival product contract, traffic regime, city fleet mix, calibration population, or credit policy for breach"
claim_ids: ["CLM-01", "CLM-02", "CLM-03", "CLM-04", "CLM-05"]
---

# ETA calibration and arrival-interval publication

## 1. The One-Line Framing

> Given a trip request captured at request time `t_r` with pickup, dropoff, a candidate driver, and the traffic snapshot in force, publish a promised arrival interval `[L, U]` (minutes from `t_r`) for the rider-experience owner, so that the quoted upper bound `U` is met at least 90% of the time without widening the quote so far that riders abandon, subject to a 120 ms p99 quote-path latency budget, compared with today's point estimate plus a fixed five-minute pad.

The decision recurs per request, roughly 40,000 times per day in the illustrative city used throughout this case. The action is not "predict the arrival time"; it is "choose what number to promise, and whether to promise an interval at all for this request". Those are different decisions with different losses, and conflating them is the central error the case exists to expose.

This is an illustrative design exercise. Every count, minute, currency amount, and rate below is constructed to make the arithmetic and the decision logic checkable. Nothing here is a measurement from a named company, a deployment, or a production experiment. Where a real product would need a measured input — for example the sensitivity of rider conversion to quoted wait — the text says so and treats the number as an unvalidated assumption in a sensitivity band rather than a fact.

The obvious simpler framing — "fit a regression, report its output" — misses three things that determine whether the quote is useful. First, the loss is asymmetric: quoting two minutes too low produces a broken promise and a support contact, while quoting two minutes too high produces a quieter but equally real abandonment cost. A symmetric error metric such as MAE treats both identically and therefore cannot rank policies that trade them differently. Second, the quantity the rider reacts to is the upper bound of what they were told, not the model's conditional mean, so the evaluation must score the published number. Third, arrival truth is only observed for trips that actually complete, and the quote itself changes the probability of completion; the label-generating process is entangled with the intervention.

The primary reusable pattern is `evaluation-mismatch`: the offline metric and the operational objective disagree because they score a different quantity, on a different population, with a different loss. Two secondary patterns apply. `distribution-shift` is the reason a calibrated interval stops covering when the traffic regime moves. `cost-latency-tradeoff` is the reason the quote path cannot simply run a heavier model per request: the rider sees the quote after the matching step, and a slow quote path delays the whole request funnel.

A decisive tradeoff, stated up front so the reader knows what to watch: in the illustrative ledger of Section 6, the interval policy removes 1,578 breach events per day relative to the fixed-pad baseline but widens the mean quote by 2.8 minutes, and the net daily contribution swings from about +$4,400 to about −$7,200 depending on a single unmeasured elasticity. The case therefore does not recommend switching the product on; it recommends a measurement that can settle the sign of that ledger.

**Framing check.** A reader should be able to restate the decision in 30 seconds: request in, promise out, for the rider-experience owner, constrained by a 120 ms quote budget, against a fixed-pad baseline, with illustrative evidence only.

## 2. Context & Background

The workflow in scope is the request-to-pickup segment of an on-demand transport marketplace. A rider opens the app, sets a pickup and dropoff, and the platform performs three coupled operations: it estimates how long until a driver reaches the pickup, it selects and offers the trip to a driver, and it shows the rider a wait estimate that sets expectations. The quote appears after matching in the product flow used here, but it is also previewed pre-match as a range in some surfaces, so the same calibration question appears twice with different information available.

Upstream of the decision sits the map and routing layer. A routing service returns a route and a free-flow travel time for the driver's approach to pickup. A traffic snapshot, refreshed on a schedule, modifies edge speeds. The ETA model consumes route features (distance, turn count, signal density, historical speed by road class and hour), live conditions (current congestion index, weather band, event calendar), and driver state (recent movement, time since last trip, vehicle class). Downstream of the decision sit the offer-acceptance step, the pickup wait itself, the trip, and the post-trip feedback and support processes. A breach of the quoted upper bound can generate a support contact, a credit, a cancellation, or nothing at all; only the first two are observable in the platform's own systems.

The decision owner in this exercise is the rider-experience product owner, who owns the promised-arrival contract and the credit policy, working with dispatch operations, which owns the matching policy, and with the data platform team, which owns the traffic snapshot and the trip ledger. These three owners have different objectives. Dispatch operations is rewarded for completed trips per driver-hour and for low rider cancellation after match. The product owner is rewarded for quote accuracy and rider trust signals. The data platform team is rewarded for freshness and availability of its feeds. A quote policy that improves one of these can degrade another, and the case keeps that tension visible rather than collapsing it into a single score.

The baseline is a point estimate plus a fixed pad. The point estimate comes from a gradient-boosted regressor trained on historical trips, predicting pickup arrival time in minutes from the features listed above, with a log transform on the target. The published quote is that estimate plus five minutes, clipped at a floor of two minutes. This baseline is not a strawman: it is fast, it is explainable in a support conversation ("we quoted the estimate plus our standard buffer"), it has no per-segment state to maintain, and it degrades gracefully when a feature feed is missing. Its documented weakness is that a single pad is wrong in both directions across segments: in dense downtown cores at rush hour it under-covers, and on long suburban approaches at night it over-quotes by a wide margin, which suppresses conversion on exactly the trips that are most profitable per minute of driver time.

Scale and timing matter to the design but are only partly known here. The illustrative city has 40,000 requests per day, a peak-hour request rate near 3,000 per hour, 6,000 active drivers across three vehicle classes, and 30 operating zones grouped into 6 hour-buckets for calibration purposes. Seasonality includes rain bands, public holidays, and stadium events. The case does not assert a real fleet size, real demand, or real operating rules; these figures exist to make capacity and storage arithmetic concrete.

Prior attempts are part of the honest history and are marked as illustrative. The team first tried to improve the point estimate by adding features (driver heading, building-entrance complexity, elevator wait at large venues), which reduced mean absolute error by a small amount but did not change breach rates materially, because the residual tail — not the mean — drives breaches. It then tried a global multiplicative correction factor fitted to minimize MAE, which moved the estimate but left the tail shape intact. Both attempts are described in Section 6 as rejected approaches with reasons, not as measured project history.

Applicable requirements are limited and are not determined here. Consumer-protection rules about the accuracy of advertised wait times, rules about compensation for missed service commitments, and local transport regulations vary by jurisdiction and are not assessed by this case. The case treats the 90% coverage target as a product decision chosen by the owner, not as a legal standard, and it flags that any external commitment to a maximum wait would need review against the actual rules in force before publication.

What makes the problem persist despite the baseline's known weakness is a specific organizational fact: the fixed pad is owned by nobody. Changing it requires agreement between the product owner, who sees conversion data, and support operations, who see breach complaints, and the two groups measure different windows. A per-segment calibrated interval creates a single artifact with a named owner and a measurable target, which is the real reason to prefer it — not that conformal prediction is more fashionable than a regression.

**Interview cue.** Explain who experiences the failure and why the current process persists: the rider experiences a broken promise, support absorbs the cost, and the pad persists because no single team owns its value.

## 3. The Business Question

The stakeholder question, reconstructed as an illustrative paraphrase rather than a quotation: "Riders tell us our wait estimate is optimistic at rush hour and pessimistic at night. Can we quote a range that we actually meet nine times out of ten, without scaring people off with a wide range, and can we tell per zone whether it is working?"

| Decision contract | Value |
|---|---|
| Question | For each eligible request, should the platform publish a calibrated arrival interval, publish the legacy point-plus-pad quote, or publish nothing until the request is re-quoted after matching? |
| Decision owner | Rider-experience product owner, with dispatch operations named as co-owner for anything that changes matching inputs. |
| Trigger and deadline | A quote must be produced inside the request funnel; the policy choice itself is reviewed when breach rate or conversion moves outside guardrails, and at each traffic-regime change. |
| Available actions | (a) legacy point + fixed pad; (b) calibrated symmetric interval from absolute-residual conformal prediction; (c) two-sided quantile regression interval; (d) per-segment interval with a fallback to (a) where calibration evidence is thin; (e) manual re-quote via support, not scalable. |
| Value and downside | Value: fewer breached promises, fewer credits, better trust signals, and a quote that reflects the actual segment. Downside: wider quotes reduce conversion; a miscalibrated interval that under-covers is worse than the pad because it is presented with more authority. |
| Objective | Maximize expected daily contribution per request subject to promised-bound coverage ≥ 90% per reporting segment and quote-path p99 latency ≤ 120 ms. |
| Hard constraints | Quote path must degrade to the baseline when the traffic snapshot or model artifact is unavailable; no quote may be published from a calibration group with fewer than 200 matured residuals; every published quote must be reproducible from a logged artifact version. |
| Scope | Requests in the illustrative city, all vehicle classes, all zones, calendar days in the evaluation window. |
| Out of scope | Pricing and surge, matching policy, fare estimation, accessibility guarantees, and legal interpretation of wait-time commitments. |

The objective and the proxy metric disagree in a way that must be named. Coverage is a proxy for trust; conversion is a proxy for revenue; neither alone determines contribution. A policy can hit 90% coverage by quoting 40-minute intervals everywhere, which would satisfy the calibration target and destroy the product. A policy can maximize conversion by quoting the optimistic mode of the arrival distribution, which would look excellent for a week and then generate a support shock. The guardrail structure therefore has two sides: coverage must be at least the target per segment, and mean interval width must stay within a declared cap that is set from measured abandonment behavior, not from intuition.

A second disagreement is between segment-level and city-level reporting. City-level coverage can sit comfortably at 91% while two airport zones run at 78%, because airport trips are a minority of volume and their residuals are heavy-tailed. The objective is therefore defined per reporting segment with a city roll-up that uses request-volume weights, and the owner is required to see both. This is not statistical fastidiousness: the airport zones are where a missed promise is most expensive, because the rider is time-bound and the support cost per incident is highest.

The expensive false decision has a concrete shape. A false positive — publishing a calibrated interval that under-covers — spends trust and generates credits at a rate the team has not budgeted. A false negative — keeping the fixed pad — leaves a known, bounded inefficiency in place. The asymmetry means the burden of proof sits on the new policy: it must show coverage at or above target on a frozen replay, and it must show the width cost is inside the cap, before any live exposure.

**Interview cue.** Which objective is being optimized, what happens with no intervention, and what does an expensive false decision look like? Answer: contribution subject to coverage and width guardrails; the pad persists and continues to mis-cover at both ends; the expensive false decision is an authoritative-looking interval that breaks its promise.

## 4. Assumptions & Constraints

| ID | Assumption/constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Trip arrival truth is recorded with event time at pickup completion, and the ledger is immutable per trip | Data | Point-in-time replay against the trip ledger schema; duplicate-trip check on `trip_id` | unverified | Residuals are computed against a moving or duplicated truth, so calibration is meaningless | Freeze a versioned snapshot and recompute all residuals | Ledger schema change / data platform |
| A-02 | The residual distribution within a reporting segment is exchangeable between calibration and serving windows | Model | Rolling coverage by segment, compared with a predeclared band; drift alarm on 7-day coverage | unverified | Coverage falls below target without any model change; promises break | Recalibrate on a rolling window; widen to the fallback pad | Traffic regime change / ML owner |
| A-03 | Quoted wait affects rider cancellation and therefore which trips produce outcome labels | Business | Compare completion rates across quote-width buckets; audit cancellations for quote exposure | unverified | Calibration is estimated on a selected population of completers only | Model censoring explicitly or report coverage on the all-requests population | Product change / analytics |
| A-04 | The quote path can afford one model call plus one quantile-table lookup within 120 ms p99 | Infrastructure | Load test at 3,000 requests/hour with injected dependency latency | unverified | Quotes arrive after riders have already decided; the decision window is missed | Serve the baseline quote and log the miss | Fleet growth or feature-service change / platform |
| A-05 | Calibration groups with at least 200 matured residuals produce a stable conformal quantile | Statistical | Bootstrap the quantile by group; flag groups whose 95% interval spans more than 1.5 minutes | unverified | Small groups get erratic intervals that over- or under-cover unpredictably | Fall back to the parent group, then to the baseline | Group definition change / ML owner |
| C-01 | Promised-bound coverage must be at least 90% per reporting segment | Business | Product contract; measured on a frozen replay and then on live windows | confirmed as a target, not as a legal standard | The product publishes promises it does not keep | Narrow scope to segments that meet target | Product review / product owner |
| C-02 | Quote-path p99 latency must stay at or below 120 ms | Infrastructure | Load test plus live latency SLI | confirmed as an internal budget | Quote arrives too late in the funnel to matter | Baseline-only mode | Platform SLO change / platform |
| C-03 | No quote may be published from a group with fewer than 200 matured residuals | Statistical | Group coverage report with counts | confirmed as policy | Small-sample intervals produce arbitrary width and coverage | Parent-group or baseline fallback | Owner decision / ML owner |
| C-04 | Every quote must be reproducible from logged inputs and artifact versions | Governance | Audit sample: recompute 50 quotes from logs | confirmed as policy | Breach disputes and model review cannot be resolved | Freeze promotion until logging is fixed | Audit finding / governance |

The two assumptions most likely to reverse the recommendation are A-02 and A-03. If the residual distribution is not exchangeable within a segment — for example because the traffic snapshot's refresh cadence changes, or because road works persist for weeks — then conformal coverage guarantees do not hold, and the interval becomes a heuristic with a misleading reputation. If quote width changes completion behavior strongly, then the evaluation population is selected by the very policy under test, and coverage measured on completers is not the coverage experienced by all requesting riders. Both are testable with data the platform already holds, which is why the case treats them as checks rather than as blockers.

One assumption is deliberately absent. The case does not assume that a better point estimate produces a better promise. The two are related but not identical: a model can improve MAE while leaving the 90th percentile of the residual unchanged, and the promise lives at that percentile. This is the mechanism behind the first rejected attempt in Section 6, and it is worth stating as a constraint on how the team allocates modelling effort.

**Interview cue.** Change one invalidating assumption and explain how the method changes. If A-03 fails, the evaluation population becomes all requests with a censoring model for canceled trips, and the decision rule gains a completion-rate guardrail that can veto a coverage gain.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Detail |
|---|---|
| Unit of analysis | One trip request, keyed by immutable `request_id`; one row per request, with a nullable trip outcome for canceled or unmatched requests. |
| Volume and window | Illustrative: 40,000 requests/day, 30 days of development data (1.2 million requests), a 7-day frozen replay window (280,000 requests), and a 240-trip calibration set per reporting group drawn from the development window. |
| Inputs | `t_r` request timestamp (UTC, second resolution); pickup and dropoff coordinates snapped to road segments; route distance (km) and free-flow time (s); traffic snapshot version and its effective timestamp; congestion index; weather band; event flag; driver candidate id, vehicle class, driver recent-speed statistics; zone id; hour bucket. |
| Outcome/label | `t_pickup` = timestamp at which the driver marks arrival at pickup; `y = (t_pickup − t_r)/60` in minutes; matured only when the trip reaches pickup with a recorded event. Canceled requests have no `y` and are excluded from residual computation but retained in the all-requests population for guardrails. |
| Population and exclusions | Include requests with a validated `request_id`, a completed matching step, and a recorded pickup event. Exclude requests with `t_pickup < t_r` or `> 180` minutes (clock or event defects), duplicate `request_id`s, and requests whose quote was never displayed (experiment control branches). Exclusions are counted by reason. |
| Join semantics | `request_id` to trip ledger is 1:1 after deduplication; traffic snapshot joins on `snapshot_effective_at ≤ t_r` with the latest version, never on ingestion time; driver statistics join on the driver's last completed event before `t_r`. |
| Splits | Development window (days 1–23) for model fitting and calibration-set construction; validation window (days 24–30) for policy selection; frozen replay window (days 31–37) held out entirely and opened once. Splits are temporal, never random across requests, because traffic and demand are serially dependent. |

The grain deserves emphasis because it is where most ETA evaluations go wrong. If the unit is "trip" rather than "request", canceled and unmatched requests vanish, and the measured coverage applies only to the subpopulation that survived matching. If the unit is "quote event" rather than "request", a request that is re-quoted three times contributes three rows, and coverage is weighted toward requests that were already uncertain. The case fixes the unit at request and keeps re-quotes as a separate counter.

### 5b. Tools & Stack

The proposed stack is deliberately modest and is described as a design, not as a deployed system. A batch feature pipeline materializes point-in-time features into a versioned table keyed by `request_id` and feature-set version. A gradient-boosted regressor (any mature implementation) produces the point estimate `f`, trained on `log(y)` to respect positivity and multiplicative error structure, with monotone constraints where the domain justifies them (longer distance cannot reduce expected time at fixed conditions). A calibration job reads matured residuals from the frozen development window, computes the per-group conformal quantile, and writes a signed quantile table. A serving path validates the request, fetches features, reads the quantile table by group key, computes `[L, U]`, applies floor and cap rules, and emits the quote with artifact versions. A ledger stores `request_id`, features hash, model version, quantile-table version, published `L` and `U`, and later the observed `y` when it arrives. A monitoring job computes rolling coverage, width, and breach counts by segment.

The stack choices that matter are not the library names. They are: (1) serving reads a precomputed quantile table rather than running conformal machinery online, which keeps the latency budget; (2) the table is versioned and signed, so a quote can be reproduced; (3) the ledger records the published interval, not just the model output, because the decision under evaluation is the published number; (4) calibration sets are stored as residual lists per group, so a recalibration does not require re-running the model over history.

### 5c. Method Summary

The pipeline is: `request → validate and authorize → point estimate f → group key → conformal quantile q̂ → interval [max(floor, f − q̂), f + q̂] → width cap check → publish or fall back`.

Candidates considered:

| Candidate | Evidence considered | Strength | Limitation / avoid when | Chosen/rejected/not tested | Revisit if |
|---|---|---|---|---|---|
| Legacy point + 5-minute pad | Existing product behavior; illustrative breach rate 13.8% in replay | Trivial to serve; explainable; no state | One pad cannot fit heterogeneous segments; under-covers tails and over-quotes long trips | Retained as fallback baseline | Never fully retired; needed for degraded mode |
| Feature-enriched point model (same publication rule) | Illustrative development comparison | Small MAE improvement | MAE is not the decision loss; tail unchanged | Rejected for this decision, useful elsewhere | If a tail-aware loss is adopted |
| Global multiplicative bias correction | Illustrative residual analysis | Removes systematic offset | Corrects the center only; does not shape the tail | Rejected | If bias is the only defect |
| Split conformal on absolute residuals | Finite-sample coverage under exchangeability; cheap at serve time | Coverage target is explicit; per-group adaptation; auditable | Requires exchangeability and minimum group counts; symmetric by construction | Chosen for the replay evaluation | If residual skew makes symmetric intervals unacceptable |
| Quantile regression (two models, τ = 0.05 and 0.95) | Literature method; not fitted here | Captures skew; can be asymmetric | Two models to train and monitor; no finite-sample coverage guarantee without further work | Considered, not tested | If skew is demonstrated and width matters economically |
| Asymmetric conformal (signed-residual, two-sided) | Not fitted here | Skew-aware with coverage machinery | More complex group bookkeeping | Considered, not tested | If symmetric width triggers the cap |

The chosen method's mechanism, stated precisely: given a calibration set of `n` matured requests in a group with residuals `r_i = |y_i − f_i|`, sort them ascending. Let `k = ceil((n + 1)(1 − α))` with `α = 0.10`. Set `q̂ = r_(k)`, the `k`-th smallest residual, indexed from 1. Publish `[max(0, f − q̂), f + q̂]`. Under exchangeability of the calibration and serving draws, the probability that a new `y` falls inside is at least `1 − α`, and the exact finite-sample statement is at least `1 − α − 1/(n + 1)` in the conservative direction. With `n = 240` and `α = 0.10`, the guarantee floor is `0.9 − 1/241 = 0.89585`, i.e. 89.6%.

Two details are easy to get wrong and are called out here. First, `k` must be computed with the `n + 1` correction; using `ceil(n(1 − α))` under-covers slightly and the error grows as `n` shrinks. Second, the residual must be computed against the same point-estimate artifact that will serve, not against a retrained model, or the calibration and the serving distribution diverge before any drift occurs.

The simplest credible baseline is not "no model" but "the model with a pad", because the point estimate is already good and the pad is the only calibration mechanism. Keeping that baseline in the comparison forces the new policy to earn its complexity against a real alternative rather than against nothing.

### 5d. Data provenance, quality, exclusions and leakage controls

Every feature carries an availability timestamp. The traffic snapshot is joined by its effective time, not its ingestion time; a snapshot that became effective at 08:00 but was ingested at 08:07 must not inform a quote issued at 08:03. Driver recent-speed features are computed from events strictly before `t_r`. Zone and hour-bucket keys are derived deterministically from the request, so they cannot leak future demand.

The most dangerous leak in this problem is not in features but in the label population. If the calibration set is built from trips that completed, and completed trips are disproportionately those with generous quotes, then the residuals are computed on a sample selected by the policy being evaluated. The diagnostic is a completion-rate comparison across quote-width buckets; if completion rises with width beyond what matching explains, censoring is material. The case keeps canceled requests in the all-requests population and reports them as a separate denominator, so a coverage gain that arrives with a completion drop is visible.

A second leak is retrospective truth. The pickup event may be edited by support after a dispute, or backfilled when a driver's device reconnects. Residuals must be computed on a snapshot taken at a declared maturity cut-off, for example 48 hours after the request, and the snapshot version recorded. Otherwise a recalibration run silently uses different truth than the one before it.

Quality checks, stated as gates rather than aspirations: duplicate `request_id` rate below 0.1%; missing pickup event rate below 0.5% within the maturity window; clock-skew trips (negative or implausibly large durations) excluded and counted; traffic snapshot coverage of at least 99% of requests with a snapshot no older than the declared freshness bound; zone assignment unambiguous for at least 99.5% of requests. A failing gate blocks promotion of a new quantile table rather than being noted in a report.

Exclusions are logged with reason codes, and the evaluator reports the excluded fraction by segment. A segment whose exclusion rate exceeds 2% is examined before its coverage number is trusted, because exclusions are rarely random: they concentrate in exactly the anomalous conditions where the promise matters most.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | Illustrative trip-request and pickup-event schema, this case | v2.0, 2026-11-10 | Defines unit, keys, timestamps | Public in this document | Not a real schema; field semantics simplified |
| SRC-02 | Split-conformal coverage statement (Vovk et al. line of work; standard result) | Method reference, rechecked 2026-11-10 | Justifies the `n + 1` quantile and finite-sample floor | Public literature | Requires exchangeability; not a guarantee under drift |
| SRC-03 | Product contract: 90% promised-bound coverage per segment, 120 ms p99 quote budget | Illustrative owner decision | Decision constraints | Internal-by-construction | Values are constructed, not negotiated facts |
| RUN-01 | Frozen replay computation of coverage, width, and breach counts | Illustrative arithmetic in Section 6 | Worked decision evidence | Reproducible by hand from the stated counts | No dataset is shipped; counts are constructed |
| CFG-01 | Group key definition (city × zone × hour bucket × vehicle class) | v1, 2026-11-10 | Determines calibration granularity | In this document | Real deployments must test granularity against sample size |
| CFG-02 | Fallback rules: minimum 200 residuals, width cap, floor 2 minutes | v1, 2026-11-10 | Bounds the policy's authority | In this document | Cap value is illustrative; requires measured abandonment |

Reproduction status: the arithmetic in Section 6 is fully reproducible from the stated counts and formulas; the model, the calibration residuals, and the replay population are not shipped, so no reader can reproduce the numbers from data. That distinction is stated because a worked example with checked arithmetic is not an empirical result.

**Takeaway for Section 5.** The decision's integrity rests on three artifacts — a point-in-time feature table, a versioned residual set per group, and a ledger of published intervals — and on two populations — matured completions for calibration, all requests for guardrails.

## 6. Analysis Walkthrough

### 6.1 The decisive result

The calibrated interval beats the fixed pad on the headline coverage number and beats it on width in the segments where the pad was most wasteful, but it **fails the per-segment coverage guardrail in one segment** and its economic sign depends on an elasticity the team has not measured. That combination — aggregate win, local regression, and an unresolved economic sign — is the finding, and it is why the recommendation in Section 7 is a bounded pilot rather than a switch-over.

The rest of this section shows the work: how the population was frozen, how the conformal quantile is computed, what the interval does per segment, what the ledger says under the stated cost assumptions, and where it breaks.

### 6.2 Step ledger

| Step | Hypothesis or decision | Input snapshot | Method/parameters | Intermediate output | Interpretation | Evidence/status |
|---|---|---|---|---|---|---|
| 1 | The request grain, not the trip grain, is the right denominator | Replay window, 7 days | Deduplicate on `request_id`; retain canceled and unmatched requests in the guardrail population | 280,000 requests, 1,120 excluded (0.40%), 278,880 eligible | Exclusions are low enough that the eligible population represents the window | ILLUSTRATIVE counts, constructed |
| 2 | The point model is already good enough that only the tail is at issue | Development window | MAE and residual quantiles by segment | MAE 2.31 min; residual p90 6.8 min; p99 14.2 min | The pad of 5 min sits far below p90, which is why it breaches | ILLUSTRATIVE arithmetic on constructed residuals |
| 3 | Feature enrichment changes the tail | Development window | Add heading, entrance complexity, venue elevator features | MAE 2.19 min (−5.2%); residual p90 6.7 min (−0.1 min) | Mean improves; the quantity that drives breaches does not | REJECTED for this decision |
| 4 | A global multiplicative correction changes the tail | Development window | Fit scalar `c` minimizing MAE, publish `c·f` | MAE 2.24 min; breach rate 13.6% vs 13.8% | Corrects center, leaves shape; not worth the recalibration burden | REJECTED |
| 5 | Per-group conformal on absolute residuals reaches the coverage target | Calibration sets, 240 residuals per eligible group | `k = ceil((n+1)(1−α))`, `α = 0.10` | `q̂` per group; city breach 9.855% | Coverage target met at city level | ILLUSTRATIVE arithmetic |
| 6 | Segment coverage is uniform | Same | Per-segment breach rates | Five segments at or below 14%, one at 12.0% vs 10.0% baseline | One segment regresses; guardrail must fire | REGRESSION FOUND |
| 7 | The economic ledger is positive | Cost assumptions in 6.6 | Savings minus width cost | +$6,785 − (e × $4,704) per day | Sign depends on `e`; break-even at 1.44 | OPEN, unmeasured elasticity |
| 8 | The quote path fits the latency budget | Design, not executed | Sum of p99 stage budgets | 90 ms of 120 ms p99 | Fits with 30 ms headroom; feature fetch is the risk | PROPOSED, not load-tested |

### 6.3 The conformal quantile, computed

The publication rule is: publish the upper bound `U = f + q̂` and the lower bound `L = max(2, f − q̂)`, where `f` is the served point estimate in minutes and `q̂` is a group-level quantile of absolute residuals from a calibration set that was produced by the *same* artifact version as `f`.

For a group with `n` calibration residuals `r_1 ≤ r_2 ≤ … ≤ r_n` and target miscoverage `α`, the split-conformal choice is the `k`-th smallest residual with

```text
FORMULA [ETA-01]: k = ceil( (n + 1) * (1 - alpha) )
WHERE: k is a 1-based index into the sorted residual list; n is the calibration
       count [residuals]; alpha is the target miscoverage [dimensionless, 0<alpha<1].
CONSTRAINT: calibration residuals and the served request are exchangeable;
            residuals are computed from the same model artifact that serves;
            n >= 1 and k <= n, which requires alpha >= 1/(n+1).
NUMERICAL EXAMPLE: n = 240, alpha = 0.10;
                   (n+1)(1-alpha) = 241 * 0.90 = 216.9; ceil = 217;
                   q_hat = r_(217) = 9.2 minutes.
INTERPRETATION: publish U = f + 9.2 and L = max(2, f - 9.2); the 217th of 240
                residuals leaves 23 residuals above it, so calibration coverage
                is 1 - 23/240 = 90.42%.
```

The finite-sample coverage statement has a floor that is easy to forget:

```text
FORMULA [ETA-02]: coverage_floor = (1 - alpha) - 1/(n + 1)
WHERE: coverage_floor is the conservative lower bound on P(y in [L,U])
       [dimensionless]; other symbols as in ETA-01.
CONSTRAINT: exchangeability holds; the interval is the symmetric conformal
            interval; the bound is one-sided-conservative, not exact.
NUMERICAL EXAMPLE: alpha = 0.10, n = 240;
                   floor = 0.90 - 1/241 = 0.90 - 0.004149 = 0.895851 = 89.59%.
INTERPRETATION: the 90% target is technically "at least 89.6%" with 240
                residuals; to claim a true 90% floor at 95% confidence the
                group needs materially more data or a larger alpha margin.
```

The boundary case is instructive and is the reason for the 200-residual rule:

| `n` | `alpha` | `k = ceil((n+1)(1−alpha))` | `q̂` position | coverage floor `(1−alpha) − 1/(n+1)` |
|---:|---:|---:|---|---:|
| 9 | 0.10 | `ceil(10 × 0.90) = 9` | maximum residual | `0.90 − 0.100 = 0.800` (80.0%) |
| 60 | 0.10 | `ceil(61 × 0.90) = 55` | 55th of 60 | `0.90 − 0.01639 = 0.8836` (88.36%) |
| 200 | 0.10 | `ceil(201 × 0.90) = 181` | 181st of 200 | `0.90 − 0.004975 = 0.8950` (89.50%) |
| 240 | 0.10 | `ceil(241 × 0.90) = 217` | 217th of 240 | `0.90 − 0.004149 = 0.8959` (89.59%) |
| 2,000 | 0.10 | `ceil(2001 × 0.90) = 1801` | 1801st of 2000 | `0.90 − 0.000500 = 0.8995` (89.95%) |

Reading the table: at `n = 9` the method degenerates to "publish the worst residual ever seen", which is both unstable and, in a heavy-tailed group, absurdly wide. The floor rises slowly — 240 to 2,000 residuals buys about 0.36 percentage points of guarantee — so the 200-residual rule is a floor on sanity, not a route to precision. Precision in the coverage claim must come from a margin above the target, e.g. targeting `alpha = 0.08` when the product contract is 90%, so that the floor sits above 0.90. At `n = 240`, `alpha = 0.08` gives `k = ceil(241 × 0.92) = ceil(221.72) = 222`, `q̂ = r_(222)`, floor `= 0.92 − 0.004149 = 0.9159`.

### 6.4 Interval behaviour by segment

Six reporting segments carry the city. Shares are of the 40,000 eligible daily requests. `f` is the mean served point estimate, `q̂` the calibrated half-width, `U_legacy = f + 5`, `U_new = f + q̂`.

| Segment | Share | Daily requests | `f` (min) | `q̂` (min) | `U_legacy` (min) | `U_new` (min) | Δ upper bound (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 downtown rush | 0.18 | 7,200 | 9.0 | 9.2 | 14.0 | 18.2 | +4.2 |
| S2 downtown off-peak | 0.14 | 5,600 | 6.5 | 4.2 | 11.5 | 10.7 | −0.8 |
| S3 airport approach | 0.07 | 2,800 | 12.0 | 15.0 | 17.0 | 27.0 | +10.0 |
| S4 outer suburban night | 0.12 | 4,800 | 22.0 | 3.2 | 27.0 | 25.2 | −1.8 |
| S5 midtown daytime | 0.29 | 11,600 | 8.0 | 7.4 | 13.0 | 15.4 | +2.4 |
| S6 outer zone daytime | 0.20 | 8,000 | 15.0 | 9.8 | 20.0 | 24.8 | +4.8 |
| **City weighted** | **1.00** | **40,000** | — | — | — | — | **+2.8** |

The weighted mean change is the arithmetic that matters for the economic ledger, and it is worth checking by hand:

```text
FORMULA [ETA-03]: mean_delta_U = sum_s w_s * delta_U_s
WHERE: w_s is the request share of segment s [dimensionless, sum=1];
       delta_U_s is the change in published upper bound in segment s [min].
CONSTRAINT: shares are of eligible requests in the same window; a policy that
            changes eligibility invalidates the weights.
NUMERICAL EXAMPLE:
  0.18*4.2 + 0.14*(-0.8) + 0.07*10.0 + 0.12*(-1.8) + 0.29*2.4 + 0.20*4.8
  = 0.756 - 0.112 + 0.700 - 0.216 + 0.696 + 0.960
  = 2.784 minutes.
INTERPRETATION: the headline "2.8 minutes wider" is a weighted average over a
                distribution with real narrowing in two segments; quoting the
                average without the table hides the S4 win.
```

Two segments narrow (S2 by 0.8 min, S4 by 1.8 min) because the pad was too generous there. S4 is the clearest justification for the whole exercise: at night, on a long suburban approach, the pad quoted 27.0 minutes when a 25.2-minute upper bound would have been met 96% of the time. The pad was not "safe"; it was costing conversion on trips that were never at risk.

### 6.5 Breach counts and the guardrail failure

| Segment | Baseline breaches/day | Baseline rate | New breaches/day | New rate | Change (pp) |
|---|---:|---:|---:|---:|---:|
| S1 downtown rush | 1,584 | 22.00% | 756 | 10.50% | −11.50 |
| S2 downtown off-peak | 336 | 6.00% | 308 | 5.50% | −0.50 |
| S3 airport approach | 924 | 33.00% | 392 | 14.00% | −19.00 |
| S4 outer suburban night | 240 | 5.00% | 192 | 4.00% | −1.00 |
| S5 midtown daytime | 1,636 | 14.10% | 1,334 | 11.50% | −2.60 |
| S6 outer zone daytime | 800 | 10.00% | 960 | 12.00% | **+2.00** |
| **City total** | **5,520** | **13.80%** | **3,942** | **9.855%** | **−3.945** |

S6 is the failure the aggregate hides. Its breach rate *rises* by two percentage points under the new policy, so its coverage (88.0%) sits below the 90% guardrail even though the city total (90.145%) clears it. The mechanism is specific: S6 is a long, mixed-density outer zone whose residual distribution is right-skewed, and a symmetric conformal interval built on absolute residuals spends half its width on a lower bound that almost never binds. The 95th-percentile overrun in S6 is driven by a small number of long detours, so the absolute-residual quantile at `α = 0.10` lands short of the one-sided upper tail that the promise actually needs.

The corrected action for S6 is not "widen everywhere": it is either a one-sided upper quantile for skewed segments, or exclusion of S6 from the rollout until a skew-aware method is validated. The case recommends exclusion first, because it is reversible and because widening S6 to fix a 2 pp breach would add roughly 3 more minutes to 8,000 requests per day and cost more than the breach saves.

**Independent check of the city total.** `1,584 + 336 + 924 + 240 + 1,636 + 800 = 5,520` and `756 + 308 + 392 + 192 + 1,334 + 960 = 3,942`. `5,520 / 40,000 = 0.138`. `3,942 / 40,000 = 0.09855`. Coverage `= 1 − 0.09855 = 0.90145`. The reduction is `1,578` events per day. All four figures reconcile.

### 6.6 Economic ledger

The ledger compares the two policies on one day of eligible requests. Every unit price is an illustrative assumption, not a contracted rate.

| Line | Baseline | New policy (S1–S5) | Difference |
|---|---:|---:|---:|
| Eligible requests | 40,000 | 40,000 | 0 |
| Breach events | 5,520 | 3,942 | −1,578 |
| Cost per breach: credit | $2.50 | $2.50 | — |
| Cost per breach: expected support | $1.80 | $1.80 | — |
| Fully loaded breach cost | $4.30 | $4.30 | — |
| Breach cost total | $23,736 | $16,951 | −$6,785 |
| Mean published upper bound | — | +2.8 min | +2.8 min |
| Completion rate (assumed 62.0%) | 24,800 trips | 24,800 − Δ trips | −Δ |
| Contribution margin per completed trip | $4.20 | $4.20 | — |

The width cost is the only line that requires an elasticity. Under the assumption that each additional minute of published upper bound reduces the request-to-completion rate by `e` percentage points, the daily lost contribution is

```text
FORMULA [ETA-04]: C_width = e * mean_delta_U * N * m
WHERE: e is the completion loss per minute of published upper bound expressed
       as a FRACTION (0.01 means one percentage point per minute);
       mean_delta_U is FORMULA ETA-03 output [min]; N is eligible requests per
       day [requests/day]; m is contribution margin per completed trip
       [currency/trip].
CONSTRAINT: e is locally linear over the quoted range and is not confounded by
            price, matching, or supply changes; it is an ASSUMPTION, not a
            measurement, in this case. Keep e in fractional units: writing
            "1.45 percentage points" into this formula overstates cost 100x.
NUMERICAL EXAMPLE: e = 0.01, mean_delta_U = 2.784 min, N = 40,000, m = 4.20;
                   fractional hit = 0.01 * 2.784 = 0.02784 (2.784 pp);
                   lost trips = 0.02784 * 40,000 = 1,113.6 trips/day;
                   C_width = 1,113.6 * 4.20 = 4,677.12 currency units/day.
INTERPRETATION: the sensitivity coefficient is mean_delta_U * N * m = 467,712
                currency units/day per unit of e, so net = 6,785 - e * 467,712
                and break-even is e* = 6,785 / 467,712 = 0.014507.
```

The unit discipline in that block is the point of it. Writing "1.45 percentage points" into a formula that expects a fraction would overstate the cost by a factor of 100 and turn a near-break-even policy into an apparently catastrophic one, which is why the sensitivity table keeps percentage points in its label and applies the factor explicitly.

Table check: at `e = 0.5` pp/min the fraction is 0.005, so `0.005 × 467,712 = $2,338.56` and `6,785.00 − 2,338.56 = $4,446.44`. At `e = 2.0` pp/min the fraction is 0.02, so `0.02 × 467,712 = $9,354.24` and `6,785.00 − 9,354.24 = −$2,569.24`. The savings term itself checks as `1,578 × 4.30 = $6,785.40`, rounded to $6,785 in the table.

| `e` (pp of completion lost per minute) | Width cost/day | Net contribution/day |
|---:|---:|---:|
| 0.5 | $2,338.56 | **+$4,446** |
| 1.0 | $4,677.12 | **+$2,108** |
| 1.45 | $6,781.82 | **+$3** (break-even) |
| 2.0 | $9,354.24 | **−$2,569** |
| 3.0 | $14,031.36 | **−$7,246** |

### 6.7 Sensitivity and boundary conditions

| Stress/test | Original condition | Alternative | Change in conclusion | Uncertainty / limitation | Decision consequence |
|---|---|---|---|---|---|
| Completion elasticity | `e = 1.45` pp/min (break-even) | `e = 0.5` pp/min | Net becomes +$4,446/day | Elasticity is unmeasured; published estimates of wait sensitivity vary widely by market | Positive case requires a measured `e` below 1.45 |
| Completion elasticity | `e = 1.45` | `e = 3.0` pp/min | Net becomes −$7,246/day | Same | Rollout would destroy value; a pilot is the only defensible step |
| Breach cost | $4.30 fully loaded | $2.00 (credit only, no support) | Savings fall to $3,156/day; break-even `e = 0.675` | Support cost is a finance estimate | Recommendation becomes much weaker |
| Breach cost | $4.30 | $9.00 (credit + retention loss) | Savings rise to $14,202/day; break-even `e = 3.04` | Retention effect not measured | Recommendation becomes robust |
| Traffic regime | Stationary residuals | Snapshot cadence changes from 5 min to 20 min | Residuals widen and shift; `q̂` under-covers until recalibrated | Exchangeability (A-02) fails | Coverage alarm must trigger recalibration, not silent degradation |
| Group granularity | 540 groups | 180 groups (drop vehicle class) | Fewer small-sample fallbacks; wider intervals per group | Coarser groups hide class-specific tails | Trade coverage against width; test both |
| Coverage target | `alpha = 0.10` | `alpha = 0.08` | `q̂` rises by roughly 1.1 min on S1; breaches fall to about 3,100/day but width cost rises | Coverage floor now clears 90% honestly | Cost rises by about $2,680/day at `e = 1.45` |
| Small-group handling | Minimum 200 residuals | Minimum 500 | 96 groups become 214 groups on fallback, covering about 11% of requests | Fallback rate rises; coverage becomes baseline-like in those areas | Prefer parent-group pooling over baseline where possible |

The highest-leverage unknown is the completion elasticity `e`, by a wide margin: it moves the answer from a $4,400/day gain to a $7,200/day loss. Nothing in the modelling work can settle it. It requires a designed exposure — a randomized or switchback test in which comparable requests see different quote widths — and that test is the actual recommendation of this case. Stating a precise interval and a precise ledger while leaving `e` unmeasured would be the professional failure the case is designed to prevent.

### 6.8 What was tried first, and why it did not work

The first attempt enriched the point model. Driver heading, building-entrance complexity, and venue elevator-wait features together reduced MAE from 2.31 to 2.19 minutes, a 5.2% improvement. The breach rate moved from 13.80% to 13.60% — 5,520 to 5,440 events per day. The reason is structural, not a tuning problem: MAE is dominated by the bulk of the residual distribution, and the breach sits at the 86th percentile. A feature that sharpens the bulk can leave the tail almost untouched, so effort spent on mean accuracy buys almost nothing for the promise.

The second attempt fitted a single multiplicative correction `c` to the point estimate, minimizing MAE on the development window. It produced `c = 1.04`, moved mean error toward zero, and moved breach rate to 13.6% — the same place as the feature work. The diagnosis is the same: a location correction cannot reshape a tail, and the tail is what the promise depends on.

Neither attempt was wasted in the sense of being wrong; both were correct answers to the wrong question. That is the transferable distinction the case is built around: the modelling objective was inherited from the metric (MAE) rather than derived from the decision (a coverage constraint at a published upper bound).

### 6.9 Intermediate-output check

A reader given only the segment table should be able to reject a plausible wrong conclusion. Suppose a colleague reports "the new policy improves coverage by 3.9 points and therefore should ship". Checking against the table: the 3.9-point figure is the city aggregate `13.80% − 9.855%`; the same table shows S6 at 12.00% against a 10.00% baseline. The conclusion "should ship" is therefore unsupported for S6, and the correct statement is "should ship for S1–S5 pending a measured elasticity, with S6 excluded". This is the specific reasoning the interview pack tests.

**Takeaway for Section 6.** The interval policy meets the city coverage target, fails one segment, and has an economic sign that one unmeasured elasticity decides; a rigorous case states all three rather than reporting the first.

## 7. The Finding

For requests in segments S1–S5, a per-group split-conformal interval published as `[max(2, f − q̂), f + q̂]` reduces breached promises from 13.80% to 9.90% of eligible requests — 1,578 fewer breach events per day on the illustrative 40,000-request population — while narrowing the published upper bound in the two segments where the fixed pad was most wasteful. In segment S6 the same rule raises the breach rate from 10.00% to 12.00%, so the policy does not satisfy the per-segment 90% coverage guardrail there and must not be applied to S6 as specified.

The economic case is unresolved rather than positive. Under the stated cost assumptions the policy saves $6,785 per day in credits and expected support and costs `e × $467,712` per day in lost contribution, where `e` is the completion loss per minute of quoted upper bound; the ledger breaks even at `e = 1.45` percentage points per minute, a quantity this case does not measure. The recommendation is therefore conditional and reversible: run a randomized width exposure to measure `e`, ship the calibrated interval for S1–S5 in shadow mode immediately, and keep the fixed pad as the serving fallback everywhere until both the elasticity and the two coverage gates (city and segment) are satisfied on mature live data.

The dominant uncertainty is not statistical. The conformal coverage arithmetic is checked and the segment counts reconcile; what is unknown is a behavioural parameter and the exchangeability assumption under traffic-regime change. A wider calibration set cannot resolve either, and no amount of model refinement changes the segment finding in S6, because the defect there is the symmetry of the interval, not the quality of the point estimate.

**Takeaway for Section 7.** Coverage improves where the pad was wrong in one direction, regresses where the residual is skewed, and the money depends on a behavioural elasticity that must be measured, not assumed.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

The offline path has one job per stage and one artifact per stage, so that a failure can be attributed instead of guessed at.

```text
[trip + pickup event ledger] -> [maturity snapshot @ t+48h]
   -> [point-in-time feature build, feature-set version] -> [feature table]
   -> [point model fit on log(y), monotone constraints]
   -> [residual extraction against the SERVING artifact]
   -> [per-group calibration sets, >=200 matured residuals]
   -> [conformal quantile job: k = ceil((n+1)(1-alpha))]
   -> [quantile table, signed, versioned]
   -> [frozen replay evaluation: coverage, width, breach, per segment]
   -> [promotion gate: city and segment coverage + width cap + latency budget]
   -> [signed policy bundle {model_version, table_version, group_keys, caps}]
```

Stage contracts:

| Stage | Input | Output | Failure behaviour |
|---|---|---|---|
| Maturity snapshot | Raw events | Immutable snapshot at `t+48h` with version id | Snapshot job failing aborts the pipeline; it never silently falls back to live truth |
| Feature build | Snapshot | Feature table keyed `(request_id, feature_version)` | Rows with unavailable-at-cutoff fields are quarantined with reason codes, not imputed |
| Model fit | Development feature table | Model artifact + training report | Fit fails if monotone constraints cannot be satisfied; no automatic relaxation |
| Residual extraction | Model artifact + matured completions | Residual list per group | Aborts if the model artifact hash differs from the one recorded at fit time |
| Calibration | Residual lists + `alpha` | Quantile table | Groups under 200 residuals are excluded and routed to parent-group fallback |
| Replay evaluation | Quantile table + frozen window | Coverage/width report per segment | A missing segment blocks promotion for that segment only |
| Promotion gate | Report + latency test | Signed bundle | Any failed gate leaves the previous bundle active |

The gate is expressed as code-level predicates, not prose: `city_coverage >= 0.90`, `min_segment_coverage >= 0.90`, `mean_width_delta <= cap`, `p99_quote_latency <= 120ms`, `groups_published >= 0.85 of eligible request volume`. The last predicate exists because a policy that falls back to the baseline for 40% of requests has not shipped anything, even if its coverage report looks excellent on the covered remainder.

### 8b. Online / Serving / Real-Time Path

```text
[rider requests trip]
   -> [auth + idempotency key check]
   -> [feature fetch: route, traffic snapshot (latest effective), driver stats]
   -> [point model: f]  (80 ms p99 budget)
   -> [group key: city x zone x hour bucket x vehicle class]
   -> [quantile lookup q_hat] (5 ms budget)
   -> [interval: L=max(2, f-q_hat), U=f+q_hat]
   -> [width cap check] -> publish | fallback to legacy pad
   -> [ledger write: request_id, versions, L, U, fallback_reason]
   -> [rider sees quote]
```

Failure path, stated per dependency:

- **Traffic snapshot stale or absent.** Publish the legacy pad, record `fallback_reason=snapshot_missing`, emit a counter. Do not publish a calibrated interval computed from an older snapshot, because the residual distribution that justified `q̂` assumed the snapshot regime.
- **Model timeout.** Publish the legacy pad. The 120 ms budget is a product constraint; a correct-but-late quote that arrives after the rider has canceled is worth nothing.
- **Quantile table read failure.** Publish the legacy pad and page the on-call. The table is small and cached, so a read failure means a broader outage.
- **Group not present in the table.** Use the parent-group quantile (zone × hour bucket, ignoring vehicle class) and record `group_fallback=parent`. If the parent is also absent, use the legacy pad.
- **Ledger write failure.** Fail the quote closed to the legacy pad and log locally; an unlogged calibrated quote cannot be defended in a breach dispute, which violates C-04.

Retry policy: one retry per dependency with a 20 ms cap, idempotent on `request_id`, and no retry chain longer than two stages. The reason is arithmetic, not taste: three sequential 20 ms retries plus two 40 ms stage calls exceeds the whole 120 ms budget before the model runs.

Sizing the quote path, so the latency claim is not hand-waving:

| Stage | p50 | p99 budget | Notes |
|---|---:|---:|---|
| Auth + idempotency | 2 ms | 8 ms | Cached token validation |
| Feature fetch (route + snapshot + driver) | 12 ms | 45 ms | Three parallel reads; p99 dominated by snapshot cache miss |
| Point model | 15 ms | 35 ms | Single-row inference; batch size 1 |
| Quantile lookup + interval + cap | 2 ms | 5 ms | In-process cache of a few thousand rows |
| Ledger write (async acknowledged) | 3 ms | 12 ms | Write-ahead, acknowledged before response |
| Serialization + network | 5 ms | 15 ms | Includes client-side render budget share |
| **Sum of stage p99s** | — | **120 ms** | Budget fits exactly; no headroom if all stages hit p99 together |

That last row is the honest reading: summing p99s assumes perfect anti-correlation across stages, which is optimistic. The design therefore sets a 90 ms target for the model-plus-feature path and treats 120 ms as an alarm threshold rather than a target, with a monthly error-budget review. A reader who checks only the sum will conclude the design fits; a reader who checks the correlation will conclude it fits *conditionally*, which is the correct conclusion.

### 8c. Key Design Decisions

| ID | Decision | Alternative considered | Why rejected or deferred | Revisit condition |
|---|---|---|---|---|
| DEC-01 | Publish `[L, U]` rather than a point plus pad | Point estimate only with a trained offset | A point cannot express asymmetric loss; the rider needs a promise, and support needs a bound to adjudicate against | If product research shows riders ignore the upper bound |
| DEC-02 | Symmetric conformal interval for launch | One-sided upper quantile | Symmetric is simpler, has a clean finite-sample statement, and over-covers the lower side harmlessly in five of six segments | S6 evidence, or any segment where skew makes the lower bound bind |
| DEC-03 | Per-group calibration with parent fallback | Global calibration | Global reproduces the pad's core defect: uniform width across heterogeneous segments | Group sample sizes fall below the floor across many groups |
| DEC-04 | Precomputed quantile table at serve time | Online conformal computation | Online computation cannot meet 120 ms p99 with per-request residual access | If the table grows beyond memory or updates must be intraday |
| DEC-05 | Exclude S6 from the launch rather than widen it | Add a fixed 3-minute skew allowance for S6 | A skew allowance is an unvalidated fudge that hides the modelling gap; exclusion is reversible and keeps the guardrail honest | A skew-aware method passes the same gate |
| DEC-06 | Fallback to legacy pad, never to a stale calibrated quote | Serve last known calibrated interval | A stale interval looks calibrated and is not; the pad's error is bounded and understood | Never — this is a safety property, not a performance choice |
| DEC-07 | Log published `L` and `U`, not just `f` | Log model output only | The decision under audit is the published number; disputes hinge on what the rider saw | Never |

### 8d. Operational readiness

Ownership is split deliberately and each piece has a named accountable role in the design: the ML owner owns the model artifact, residual extraction, and quantile table; the platform owner owns the serving path, latency SLI, and fallback behaviour; the product owner owns the coverage target, the width cap, and the credit policy; support operations owns breach adjudication and the reason taxonomy. An incident that changes the published interval requires the ML owner and the product owner to agree on the rollback, which is why the bundle is signed and the previous version is kept warm.

Proposed SLIs, with targets to be set by the owners after a shadow period rather than invented here:

| SLI | Definition | Proposed guard |
|---|---|---|
| Quote coverage (city) | `1 − breaches / eligible requests`, 7-day rolling | ≥ 0.90 |
| Quote coverage (segment) | Same, per reporting segment | ≥ 0.90 per segment |
| Published width | Mean `(U − L)` by segment | Within declared cap per segment |
| Fallback rate | Quotes served by legacy pad / all quotes | Alarm above 5% sustained |
| Quote-path latency | p99 of end-to-end quote | ≤ 120 ms, alarm at 100 ms |
| Ledger completeness | Quotes with all version fields / all quotes | ≥ 0.999 |
| Recalibration staleness | Age of active quantile table | Alarm above 14 days |

Runbooks, stated as decisions rather than as a list of steps: (1) coverage alarm in one segment → switch that segment to the legacy pad by bundle edit, not by code deploy; (2) fallback rate alarm → check snapshot freshness first, model timeout second, table cache third; (3) width alarm → do not widen; investigate whether the point model or the residual set changed, because widening converts a modelling defect into a product cost; (4) latency alarm → shed the calibrated path for the highest-volume zone and keep it for the segments where the pad is worst, which preserves most of the benefit under load.

**Takeaway for Section 8.** The architecture's central safety property is that a calibrated quote is never served from stale inputs: the system degrades to a bounded, understood baseline rather than to an authoritative-looking guess.

## 9. Evaluation & Validation

### 9.1 Metric definitions

All metrics below are defined on one population, one window, and one comparator, because the most common evaluation failure in this domain is comparing a coverage number computed on completions with a completion rate computed on all requests.

| Metric | Numerator | Denominator | Population | Window | Comparator |
|---|---|---|---|---|---|
| Promised-bound coverage | Requests with `y ≤ U` | Eligible requests with matured `y` | Requests in the reporting segment | Calendar day, event time | Legacy pad on the same requests |
| Breach rate | Requests with `y > U` | Same | Same | Same | Same |
| Mean published width | Sum of `(U − L)` | Eligible requests quoted | All quoted requests (including fallbacks, reported separately) | Same | Same |
| Fallback rate | Quotes served by legacy pad | All quotes | All requests | Same | n/a |
| Completion rate | Requests reaching trip start | All eligible requests | All requests | Same | Same |
| Quote-path latency | n/a | n/a | All quote calls | p50/p99 over the window | Budget 120 ms |
| Segment disparity | Max segment breach rate | Min segment breach rate | Segments with ≥ 200 requests | Same | Baseline disparity |

The coverage denominator deserves one more sentence. Matured `y` exists only for requests that reached pickup, so a strict reading of "coverage" is conditional on reaching pickup. The case reports both the conditional coverage (on matured requests, comparable to the conformal guarantee) and the all-requests breach proxy (which counts a canceled-before-pickup request as not breached, because no promise was broken in the rider's experience of waiting). Mixing the two produces the classic error of reporting a coverage gain that is partly a completion decline.

### 9.2 Validation protocol

The protocol is a temporal replay on a frozen window, followed by a shadow comparison, followed by a designed exposure. Each stage answers a different question and none substitutes for another.

| Stage | Question answered | Design | Pass condition | Status |
|---|---|---|---|---|
| V1 Offline replay | Does the interval meet coverage on held-out history? | Apply the bundle to the frozen 7-day window; score published `U` against matured `y` | City ≥ 0.90 and every segment ≥ 0.90 | NOT EXECUTED (illustrative arithmetic only) |
| V2 Residual exchangeability | Do rolling windows behave like the calibration window? | Compute coverage per day for 30 days; test for trend and for step changes at snapshot-cadence events | No day below 0.88; no significant downward trend | NOT EXECUTED |
| V3 Shadow comparison | Does the new bundle behave like the replay in live traffic without exposure? | Compute both quotes for every request; publish only the legacy pad; log both | Live-vs-replay coverage gap ≤ 1 pp per segment | NOT EXECUTED |
| V4 Width exposure | What is the completion elasticity `e`? | Randomized or switchback assignment of quote width within segment, stratified by zone and hour; predeclared minimum detectable effect | Elasticity estimated with a confidence interval that excludes the break-even value | NOT EXECUTED — this is the decisive missing measurement |
| V5 Load and failure injection | Does the path fit 120 ms p99 and degrade correctly? | Replay 3,000 requests/hour with injected snapshot staleness, model timeout, table miss | p99 ≤ 120 ms; fallback behaviour matches Section 8b exactly | NOT EXECUTED |
| V6 Fairness and segment audit | Is any group systematically worse? | Breach rate by zone, income band proxy if available, vehicle class, time of day | No segment above 0.90 breach-inverse without a documented remediation | NOT EXECUTED |

### 9.3 Sample-size reasoning for V4

The elasticity is the decision-critical parameter, so its measurement deserves an explicit power calculation rather than a hand-wave.

```text
FORMULA [ETA-05]: n_per_arm = 2 * (z_(1-alpha/2) + z_(1-beta))^2 * sigma^2 / delta^2
WHERE: n_per_arm is requests per arm [requests]; z_(1-alpha/2)=1.96 for 95%
       two-sided confidence; z_(1-beta)=0.84 for 80% power; sigma is the
       standard deviation of the per-request completion indicator
       [dimensionless]; delta is the target difference in completion
       probability [dimensionless].
CONSTRAINT: two independent arms, one request per unit, no clustering; with
            zone-hour clustering the effective sample is smaller and the
            design must be analysed at the cluster level or inflated.
NUMERICAL EXAMPLE: p=0.62, sigma=sqrt(0.62*0.38)=0.4854; target delta=0.01
                   (one percentage point);
                   n_per_arm = 2*(1.96+0.84)^2*0.2356/0.01^2
                             = 2*7.84*0.2356/0.0001
                             = 2*7.84*2356.0
                             = 36,942 requests per arm.
INTERPRETATION: detecting a one-point completion change needs about 74,000
                requests total, i.e. roughly two days at 40,000 requests/day,
                IF assignment is truly per-request; with zone-hour switchback
                and ICC=0.01 the design effect is 1+(m-1)*ICC and the required
                window grows with cluster size m.
```

Check the arithmetic: `sigma = sqrt(0.62 × 0.38) = sqrt(0.2356) = 0.48539`; `7.84 × 0.2356 = 1.84710`; `1.84710 / 0.0001 = 18,471.0`; doubled is `36,942.1`, rounded to 36,942. At 40,000 requests per day split evenly, 20,000 per arm per day, two days gives 40,000 per arm. So a two-day exposure is barely sufficient before clustering, which is exactly the kind of uncomfortable result a power calculation exists to surface.

The decision relevance is direct: if the team wants to distinguish `e = 1.0` from `e = 1.45` (a 0.45 pp/min difference over a 2.8-minute width change, i.e. about 1.25 pp of completion), the required sample is larger still: `2 × 7.84 × 0.2356 / 0.0125^2 = 23,643` per arm, about 47,286 total — roughly 1.2 days at full volume, but with clustering and with the need to observe mature completions, a one-week exposure is the realistic minimum.

### 9.4 What the validation would have to show

For the policy to be recommended without qualification, all of the following must hold on the same frozen population: city coverage at least 0.90; every reporting segment at least 0.90; mean width delta within the cap; live-vs-replay gap under 1 pp; measured `e` with an upper confidence bound below 1.45 pp/min; fallback rate under 5%; and p99 latency under 120 ms with failure injection. Any single failure narrows scope rather than triggering a compensating change elsewhere — the pattern this case is teaching is that guardrails are not tradeable against each other.

### 9.5 Validity threats

| Threat | Mechanism | Mitigation | Residual risk |
|---|---|---|---|
| Selection by quote width | Wider quotes change completion, so matured labels are not a random sample | Report all-requests guardrails; model censoring if material | Elasticity itself is the confounder; only exposure resolves it |
| Non-exchangeability over time | Traffic regime shift invalidates the calibration quantile | Rolling coverage monitoring; recalibration trigger at 14 days or on regime change | Slow drift between reviews |
| Snapshot version mismatch | Serving uses a different snapshot vintage than calibration | Version the snapshot in the feature table and in the ledger | Partial coverage during snapshot outages |
| Metric gaming | Optimizing coverage by widening | Width cap as a co-equal guardrail with its own owner | Cap may be set too loosely |
| Survivorship in segments | Small segments produce unstable quantiles | 200-residual floor and parent fallback | Fallback rate can hide a coverage hole |
| Feedback loop | Publishing wider quotes reduces demand in a zone, which changes future residuals | Monitor demand and residual distribution jointly | Loop is slow and hard to attribute |

**Takeaway for Section 9.** The evaluation is designed to fail loudly on the segment where it should fail, and the decisive test (V4) measures a behavioural parameter that no offline metric can supply.

## 10. Action Taken

**As of 2026-11-10 no product action has been taken.** This is an illustrative design exercise; there is no pilot, no exposure, and no measured outcome. What follows is the proposed action sequence with its authority boundaries, written so a reader can tell a decision from a plan.

| Step | Action | Owner | Authority required | Reversible? | Stop condition |
|---|---|---|---|---|---|
| 1 | Freeze the replay population and publish the coverage/width report | ML owner | None (internal analysis) | Yes | Exclusion rate above 2% in any segment |
| 2 | Build the serving path with the legacy pad as the only published quote | Platform owner | Engineering change review | Yes | Latency budget breached under load |
| 3 | Run shadow mode (V3) for 14 days, computing both quotes and publishing the pad | Platform owner + product owner | Product sign-off | Yes | Live-vs-replay gap above 1 pp |
| 4 | Recalibrate for S6 with a one-sided upper quantile and re-run V1 for S6 only | ML owner | ML review | Yes | S6 coverage still below 0.90 |
| 5 | Run the width exposure (V4) in S1–S5 with predeclared stopping rules | Product owner + experimentation owner | Ethics/privacy review of the exposure design | Yes, by design | Any segment breach rate above 0.95 inverse-coverage during exposure |
| 6 | Decide switch-over using the measured `e` and the full gate | Product owner | Product + finance review | Partly | `e` upper bound at or above 1.45 pp/min |

The reason step 3 precedes step 5 is that shadow mode can be stopped instantly with no rider impact, while an exposure deliberately shows some riders a different promise. Ordering the reversible, zero-exposure work first is the operational expression of the case's evidence discipline.

**Stop authority** rests with the product owner for publication and with the platform owner for serving safety. Either can revert to the legacy pad by switching the active bundle; neither needs the other's agreement to revert, and both must document the reason. The asymmetry is intentional: reverting to a bounded, understood baseline should always be cheaper than continuing with a policy whose coverage is unproven.

**Not done, and deliberately so:** no change to the credit policy, no change to matching, no change to surge. Those decisions are adjacent and would confound the exposure if changed at the same time; the case keeps them fixed so the measured elasticity can be attributed to width.

## 11. Outcome & Results

**Outcome not measured.** There is no deployment, pilot, or experiment behind any number in this case. Every figure in Sections 6 and 9 is constructed arithmetic on a stated population, and the reader should treat it as a worked example of the reasoning, not as evidence about any real marketplace.

What *is* established by the work in this document, and what is not:

| Claim ID | Claim | Evidence type | Established? | What it does not establish |
|---|---|---|---|---|
| CLM-01 | Split conformal with `n = 240`, `alpha = 0.10` yields a calibration coverage of 90.42% and a finite-sample floor of 89.59% | DERIVED from the conformal construction and checked arithmetic | Yes, as mathematics | That real residuals are exchangeable |
| CLM-02 | The illustrative policy reduces breach events from 5,520 to 3,942 per day on the stated population | ILLUSTRATIVE arithmetic | Only as arithmetic | Any real breach reduction |
| CLM-03 | Segment S6 regresses from 10.00% to 12.00% breach rate under the symmetric interval | ILLUSTRATIVE arithmetic, mechanism argued | Arithmetic: yes; mechanism: plausible, not measured | That real skewed segments behave this way by this magnitude |
| CLM-04 | The economic ledger breaks even at `e = 1.45` pp/min under the stated costs | DERIVED from ETA-04 | Yes, conditional on the cost assumptions | The value of `e`, or that the cost assumptions hold |
| CLM-05 | Detecting a one-point completion change requires about 36,944 requests per arm absent clustering | DERIVED from the power formula | Yes, as a design calculation | That the exposure is feasible in one region |

The measurement plan that would replace these with observations has four artifacts, each with an owner and a deadline set by the owners rather than by this document: a frozen replay report (coverage and width per segment), a shadow-mode divergence report (live vs replay), an exposure readout (elasticity with confidence interval, by segment), and a failure-injection report (latency and fallback behaviour). Until at least the first three exist, the honest status of this decision is "designed, not evaluated".

A specific caution about interpreting the illustrative numbers: the 3.9-point city coverage improvement is large enough to look persuasive, and that is precisely why it is dangerous. In a real evaluation the improvement would probably be smaller, the segment regressions would be more numerous, and the elasticity would likely sit nearer the break-even value than either extreme in the sensitivity table. The case deliberately shows a favourable aggregate so that the reader has to do the work of finding the reasons it is not sufficient.

## 12. Failure Modes & What Went Wrong

The failures below are the ones this design is built to survive. Each is stated with the mechanism, the discriminating diagnostic that separates it from its nearest competitor, and the recovery path, because a failure catalogue without differential diagnosis is just a list of worries.

### F-01 Calibration-set and serving-artifact divergence

**Symptom.** Coverage degrades gradually over one to two weeks with no traffic change. The quantile table looks stable; the point model's residual spread does not.

**Mechanism.** The residual set was computed against model version A, but serving was promoted to version B (a retrain, a feature change, or a library upgrade that altered float behaviour). The residuals describe A's error distribution, and B's may be wider or narrower. The interval is calibrated for the wrong estimator.

**Diagnostic.** Compare a hash of the serving model artifact with the hash recorded in the calibration job's metadata, per group. A second check: recompute residuals for a 1,000-request sample against the *serving* artifact and compare the `q̂` implied by that sample with the active table's value; a gap above 0.5 minutes means divergence.

**Nearest competitor.** Genuine drift (F-03). Discrimination: divergence is a step change at the promotion timestamp; drift is a trend aligned with an external regime.

**Recovery.** Roll back the serving model to the version the table was calibrated against, or block the promotion until recalibration completes. Do not widen the interval as a stopgap.

**Prevention.** Make the artifact hash a hard join key between the model registry and the calibration job; a promotion that changes one without the other fails a CI check.

### F-02 Stale traffic snapshot served as current

**Symptom.** Coverage falls sharply in specific windows, concentrated at the minutes around a snapshot refresh. Breaches cluster by clock minute.

**Mechanism.** The serving path joins the most recently *ingested* snapshot rather than the most recently *effective* one, or the cache serves a snapshot past its freshness bound. The quote is computed for road conditions that have already changed.

**Diagnostic.** For breached requests, compare snapshot `effective_at` and `ingested_at` with `t_r`. A concentration of breaches where `effective_at` is older than the declared bound is conclusive. The competing explanation — a genuinely sudden congestion event — would show breaches concentrated in time but with fresh snapshots.

**Recovery.** Fall back to the legacy pad for the affected zone until snapshots are fresh; the pad is bounded error, whereas a stale-snapshot interval looks calibrated.

**Prevention.** Enforce the freshness bound in the serving path (reject and fall back), alarm on snapshot age, and include a freshness dimension in the coverage dashboard so a snapshot outage is visible as a coverage dip rather than as a mysterious model regression.

### F-03 Distribution shift after a traffic-regime change

**Symptom.** Coverage drifts below target over days, with the largest deterioration in the segments whose routes were most affected. Width appears adequate; residuals have a heavier tail.

**Mechanism.** A new road layout, a sustained construction closure, a seasonal pattern (school terms, holiday traffic), or a change in the routing engine's speed model changes the residual distribution. Exchangeability between calibration and serving fails, so the conformal guarantee no longer applies — not because the arithmetic is wrong, but because its premise is.

**Diagnostic.** Rolling 7-day coverage by segment against a predeclared band; a CUSUM or simple step test on daily coverage; correlation of the change point with an external event log. The discriminating check against F-01 is that residuals computed against the current artifact are themselves shifted, i.e. the model is fine but the world moved.

**Recovery.** Recalibrate on a rolling window that includes post-change data, and widen only if the recalibrated `q̂` widens. If the change is a one-off closure with a known end date, prefer a temporary segment split over a permanent widening.

**Prevention.** Scheduled recalibration (14-day cadence in this design), an event calendar that triggers recalibration, and a coverage alarm that pages rather than merely records.

### F-04 Small-group quantile instability

**Symptom.** A handful of groups publish intervals that swing wildly week to week — 2 minutes one week, 14 the next — with coverage alternating between 0.99 and 0.78. These are usually low-volume zones or rare vehicle classes.

**Mechanism.** With `n` near the floor, the `k`-th order statistic is sensitive to a few residuals. At `n = 40`, `k = ceil(41 × 0.9) = 37`, so `q̂` is the 4th largest residual: adding or removing one extreme trip moves it substantially.

**Diagnostic.** Track the week-over-week change in `q̂` by group and the group's residual count. A group whose `q̂` moves more than 30% with `n < 100` is exhibiting this failure, not genuine drift. Bootstrap the quantile: if the 95% interval for `q̂` spans more than 1.5 minutes, the group cannot support its own calibration.

**Recovery.** Route the group to its parent (zone × hour bucket) until it accumulates residuals, and record the fallback so the coverage report does not silently mix calibration levels.

**Prevention.** The 200-residual floor, plus a bootstrap stability gate at table-build time, plus a minimum-volume predicate in the promotion gate so a policy cannot claim coverage on 40% of traffic.

### F-05 Elasticity sign error in the ledger

**Symptom.** After a rollout, contribution falls even though coverage improved and the pre-launch ledger was positive. The coverage dashboard is green. Finance is unhappy.

**Mechanism.** The pre-launch ledger used an assumed `e` from a different market, a different product surface, or a different vehicle class. The true elasticity in this market is above the break-even value. Because the policy was justified on coverage, no one rechecked the behavioural assumption.

**Diagnostic.** Compare actual completion rate by width bucket before and after, adjusting for matching and supply. The giveaway is a completion decline concentrated in the segments whose width increased most, with no corresponding change in supply or price.

**Recovery.** Narrow the width in the affected segments (raise `alpha` toward 0.12–0.15, i.e. accept slightly worse coverage for a tighter bound) or revert to the pad. This is a genuine trade, not a free fix: coverage will fall below 0.90 in exchange for contribution.

**Prevention.** Make `e` a pre-registered parameter of the launch, with a measured value and a confidence interval, and require the exposure (V4) before any switch-over. A ledger with an assumed elasticity is a scenario, not a forecast.

### F-06 Ledger incompleteness breaks breach adjudication

**Symptom.** Support cannot adjudicate a breach dispute because the published `U` was not recorded, or was recorded without the artifact versions. Credits are issued inconsistently.

**Mechanism.** The serving path logs the model output `f` but not the published interval, or logs the interval without versions, or the async ledger write is dropped under load. The quote the rider saw is not reconstructible.

**Diagnostic.** Sample 50 disputed requests and attempt reconstruction from logs alone. Any request that cannot be reconstructed exactly — same `L`, same `U`, same versions — demonstrates the failure. The competing explanation (rider misremembering) is excluded by the fact that the reconstruction attempt fails on the platform side.

**Recovery.** Reconstruct from whatever partial logs exist, adjudicate on the balance of evidence, and treat the systemic gap as the real incident. Do not issue blanket credits without understanding the rate.

**Prevention.** Acknowledged write-ahead logging before the response is sent, a ledger-completeness SLI at 0.999, and a synthetic canary that quotes a known request hourly and verifies the ledger row.

### F-07 Fallback storm masking a coverage hole

**Symptom.** Coverage looks excellent during an incident — sometimes better than normal — while a large share of quotes are being served by the legacy pad. The dashboard shows 92% coverage and nobody notices that 30% of it is the baseline.

**Mechanism.** The fallback rate is a separate metric from coverage, and during a dependency outage the fallback path dominates. If the fallback's own coverage happens to be high (for example because the outage coincides with light traffic where the pad over-covers), the blended number improves.

**Diagnostic.** Always report coverage split by publishing path (calibrated vs fallback), never only blended. A fallback rate above 5% sustained is the alarm; a coverage improvement accompanied by a rising fallback rate is the tell.

**Recovery.** Fix the upstream dependency; keep the split reporting in place. Do not "fix" coverage by making the fallback the default.

## 13. Transferable Lesson

**The pattern.** When a metric's loss is asymmetric and the operational quantity is a published bound rather than a conditional mean, calibration must be evaluated at the bound, on the population that will experience it, with the cost of over- and under-quoting priced separately. An estimate that is excellent in mean absolute error can be uniformly poor at the promise it must keep.

**Why the lesson transfers.** The mechanism is not specific to arrival times. Four conditions make it apply, and all four are checkable before starting:

1. The decision publishes a number that a customer or operator acts on, so the published quantity — not the model output — is the decision variable.
2. The loss is asymmetric, so a symmetric accuracy metric cannot rank policies.
3. The relevant statistic is a tail quantile, so improving the bulk of the residual distribution buys little.
4. Truth is observed conditionally — only for units that survive an intermediate step — so the evaluation population is selected by the policy itself.

When condition 1 fails (the output is an internal score), the lesson collapses to ordinary regression evaluation. When condition 2 fails (the loss really is symmetric), a point-plus-pad can be optimal and the added machinery is waste. When condition 3 fails (the loss is driven by the mean, as in capacity planning over many trips), MAE improvements do transfer. When condition 4 fails (truth is observed for every unit), conformal coverage can be taken at face value.

**Preconditions that must hold for this specific solution.** Exchangeability of residuals within a reporting group; a calibration set produced by the serving artifact; enough residuals per group for the order statistic to be stable; a serving path fast enough to read a precomputed table; and a behavioural elasticity that has been measured rather than assumed. Each precondition has a named owner and a check in Section 9; none is a formality.

**Counterexample that breaks the lesson.** Consider a fleet-utilization decision: how many drivers to schedule per hour. There the loss is not a per-request bound but a sum over thousands of trips, idiosyncratic tail errors average out, and the mean bias of the forecast dominates. In that setting, the MAE-focused work that failed here would be the correct work, and a per-group conformal interval would add complexity without improving the scheduling decision. The same model artifact can be right for one decision and wrong for another; the lesson is about matching the method to the loss, not about the method being better.

**A second counterexample, closer to home.** Suppose the product team decides to publish a single point estimate but to train the model on the pinball loss at `τ = 0.9` instead of squared error. That approach attacks the same target (the 90th percentile) with a different mechanism. It is not obviously worse; it lacks the finite-sample coverage statement but can capture skew naturally. The honest position is that the case did not test it, and that a team with a well-tuned quantile model and a coverage-monitoring harness could reasonably prefer it. What the case *does* establish is that whichever mechanism is chosen, the evaluation must be at the published bound and per segment.

**Changed constraint.** If the product owner lowers the coverage target to 80% because research shows riders only notice gross breaches, the whole design loosens: `alpha = 0.20`, `q̂` shrinks materially, width cost falls, and the ledger's break-even elasticity moves up. The method does not change; the numbers do. That is the sign of a well-posed decision problem — the answer moves with the constraints rather than being baked into the tooling.

## 14. Connections & Cross-References

**Related methods and where they live in this collection.**

| Topic | Why it connects | Where it is covered |
|---|---|---|
| ETA evaluation | Defines the metrics used here: coverage, width, breach, segment disparity, and the conditional-vs-all-requests denominator distinction | [02_ETA_Evaluation.md](../Cheat_Sheets/02_ETA_Evaluation.md) |
| Mobility metric tree | Supplies the request-to-completion denominator structure that this case's guardrails depend on | [01_Mobility_Metric_Tree.md](../Cheat_Sheets/01_Mobility_Metric_Tree.md) |
| Dynamic-pricing guardrails | Shares the width-vs-volume economic structure; the elasticity logic is the same shape as price elasticity with a different lever | [03_Dynamic_Pricing_Guardrails.md](../Cheat_Sheets/03_Dynamic_Pricing_Guardrails.md) |
| Driver-supply forecast | The supply side of the same marketplace; a supply forecast error changes arrival times and therefore residuals | [02_Driver_Supply_Forecast.md](./02_Driver_Supply_Forecast.md) |
| Dispatch failure recovery | The operational counterpart: what happens when the quote path itself fails | [04_Dispatch_Failure_Recovery.md](./04_Dispatch_Failure_Recovery.md) |

**Cross-domain contrasts.** In travel and hospitality, the analogous problem is a hotel quoting a room-ready time or an airline quoting a delay; the same asymmetric-loss structure applies but the label is observed only for passengers who do not rebook, which is a stronger form of the selection problem in A-03. In food delivery, the promised delivery time has a much tighter tail and a contractual penalty in some markets, which raises the breach cost and pushes the optimal `alpha` down.

**Open questions, each with an owner.**

1. What is the measured completion elasticity by segment, and does it differ between new and returning riders? Owner: product analytics, answered by V4.
2. Does the residual distribution remain exchangeable across a snapshot-cadence change, and how quickly does coverage recover after recalibration? Owner: ML owner, answered by V2 and a planned cadence-change replay.
3. Is a one-sided upper quantile materially better than symmetric conformal for skewed segments, on both coverage and width? Owner: ML owner, untested here.
4. Does publishing an interval (rather than a point) change rider behavior independent of its width — an "interval framing" effect? Owner: product research, not measured.
5. What is the true fully loaded cost of a breach, including retention effects? Owner: finance, currently a \$4.30 placeholder.

### Interview Pack

**Q01 — Framing.** *You are handed this problem with no further context. State the decision, the unit of analysis, and the baseline in under a minute.*

**Strong answer.** The unit is the request, not the trip, because canceled requests still experience a promise. The decision is which quote to publish — legacy pad, calibrated interval, or defer — owned by the rider-experience product owner. The baseline is the existing point estimate plus a five-minute pad, and the binding constraints are 90% coverage per segment and a 120 ms p99 quote budget. The evidence is illustrative throughout.

**Reasoning.** Fixing the unit first prevents the most common downstream error: scoring coverage on completions while claiming it applies to all riders. Naming the baseline as a real competitor rather than a strawman forces the new policy to earn its complexity.

**Follow-up.** *What changes if the unit were the trip?* Coverage becomes conditional on completion, canceled requests vanish from the denominator, and a policy that suppresses completion could appear to improve coverage. The fix is either to keep requests as the guardrail population or to model censoring explicitly.

**Acceptable alternatives.** A candidate who says "the unit is the request/promise event" and explains why is fine; one who says "the prediction" and cannot name a denominator is not.

**Red flags.** Optimizing MAE; treating the model output as the decision; claiming a measured improvement; ignoring the fallback path.

**Scoring.** 3: names unit, owner, baseline, constraints, and evidence boundary. 2: names most, misses the evidence boundary. 1: describes a modelling task rather than a decision. 0: proposes a model without a decision.

**Support.** Sections 1, 3, 5a.

**Q02 — Leakage.** *Where is the most dangerous leakage in this pipeline, and how would you detect it?*

**Strong answer.** Not in the features — those have availability timestamps and join on effective time — but in the label population. Matured outcomes exist only for trips that reached pickup, and the quote width itself influences completion, so residuals are computed on a sample selected by the policy under test. Detection: compare completion rate across quote-width buckets, stratified by segment and hour; a monotone relationship beyond what matching explains indicates material selection.

**Reasoning.** Feature leakage inflates offline scores and is usually caught by a point-in-time audit. Population leakage is subtler: it does not inflate the score, it changes what the score means, and it can make a policy look calibrated while it is systematically mis-covering the riders who abandoned.

**Follow-up.** *How would you correct for it if it is material?* Either report coverage on the all-requests population with a censoring model for canceled trips, or design the exposure so that width is randomized and completion is measured as an outcome rather than a filter. The second is cleaner and is what V4 does.

**Acceptable alternatives.** A candidate who proposes inverse-probability weighting by observed completion propensity is also correct, though harder to defend than randomization.

**Red flags.** Answering "train/test split prevents leakage"; ignoring that truth is conditional; proposing to drop canceled requests without discussion.

**Scoring.** 3: identifies population selection, gives a discriminating check, proposes a correction. 2: identifies it but no check. 1: names feature leakage only. 0: says there is no leakage risk.

**Support.** Sections 4 (A-03), 5d, 9.5.

**Q03 — Calculation.** *With 240 calibration residuals and a 10% target miscoverage, which residual index do you publish, what is the implied calibration coverage, and what is the finite-sample floor?*

**Strong answer.** `k = ceil((241)(0.90)) = ceil(216.9) = 217`. Publishing the 217th smallest of 240 leaves 23 residuals above it, so calibration coverage is `1 − 23/240 = 0.90417`. The finite-sample floor is `0.90 − 1/241 = 0.89585`, i.e. 89.59%.

**Reasoning.** The `n+1` correction is what makes the guarantee hold; using `ceil(n(1−alpha)) = 216` would leave 24 above and give 0.90 in-calibration but a worse finite-sample bound. The floor matters because the product contract says 90%, and at `n = 240` the method honestly delivers "at least 89.6%", not 90%.

**Follow-up.** *What would you change to claim a true 90% floor?* Target `alpha = 0.08`, which at `n = 240` gives `k = ceil(241 × 0.92) = 222` and floor `0.92 − 0.00415 = 0.9159`, comfortably above 0.90, at the cost of roughly a minute more width.

**Acceptable alternatives.** Targeting `alpha` such that the floor clears the contract is the right shape of answer; exact index arithmetic is required.

**Red flags.** `k = 216`; claiming exactly 90% coverage; confusing `n` with the number of groups.

**Scoring.** 3: correct index, correct coverage, correct floor, correct remedy. 2: correct index and coverage, no floor. 1: right idea, arithmetic slip. 0: cannot compute.

**Support.** Section 6.3, FORMULA ETA-01 and ETA-02.

**Q04 — Diagnosis.** *Coverage drops from 90.4% to 86% over ten days. The model artifact hash is unchanged. Give your differential and the single most informative check.*

**Strong answer.** Two leading candidates: distribution shift (traffic regime, road works, season) and stale snapshot serving. A third is small-group instability if the drop is concentrated in low-volume groups. The most informative single check is residuals computed against the *current* artifact on the most recent 1,000 matured requests, compared with the calibration window: if the residual distribution itself has shifted, it is drift; if residuals are unchanged but serving uses older snapshots, the freshness audit on breached requests will show it.

**Reasoning.** Hash equality rules out F-01 (artifact divergence), so the remaining candidates differ in *where* the distribution changed: in the world, in the serving join, or in the estimation. Computing fresh residuals separates the world from the serving path.

**Follow-up.** *If it is drift, what do you do in the next hour versus the next month?* Next hour: recalibrate on a rolling 30-day window including post-change data and re-run the gate for affected segments; if the recomputed `q̂` widens beyond the cap, fall back to the pad for those segments. Next month: add an event-calendar trigger and shorten the recalibration cadence.

**Acceptable alternatives.** Checking snapshot freshness first is also defensible if the drop coincides with a known refresh incident.

**Red flags.** Immediately widening the interval; retraining the point model without checking residuals; blaming the data pipeline without a check.

**Scoring.** 3: three candidates, discriminating check, time-boxed actions. 2: two candidates with a check. 1: one guess. 0: widens the quote.

**Support.** Sections 12 F-01/F-02/F-03, 9.2 V2.

**Q05 — Economics.** *The policy saves \$6,785/day in breach costs and widens quotes by 2.8 minutes. What else do you need before recommending it?*

**Strong answer.** The completion elasticity with respect to published width, measured in this market and segment structure. The ledger is `net = 6,785 − e × 467,712`, so it breaks even at `e = 1.45` percentage points of completion per minute. Without a measured `e` and its confidence interval, the recommendation is a scenario, not a forecast. I would also want the breach cost confirmed by finance, because a credit-only cost of \$2.00 rather than \$4.30 cuts savings to \$3,156 and moves break-even to `e = 0.675`.

**Reasoning.** The sign of the decision is controlled by one unmeasured parameter whose plausible range straddles the break-even point. Reporting the savings alone would be the classic error of presenting the favourable half of a ledger.

**Follow-up.** *How would you measure `e` quickly without a long experiment?* Switchback assignment by zone-hour with a two-week window, analyzed at cluster level; the power calculation says ~36,942 requests per arm for a one-point effect, which is roughly two days of full volume before clustering inflation.

**Acceptable alternatives.** Proposing a shadow-mode proxy (e.g. historical quote-width variation instrumented post hoc) is acceptable if the candidate names its confounding risk.

**Red flags.** Recommending rollout on savings alone; using an elasticity from a different market without saying so; ignoring that wider quotes also reduce matching efficiency.

**Scoring.** 3: names `e`, computes break-even, proposes a measurement, checks sensitivity. 2: names `e` and break-even. 1: says "measure conversion" vaguely. 0: recommends on savings.

**Support.** Sections 6.6, 6.7, 9.3, FORMULA ETA-04/ETA-05.

**Q06 — Segment regression.** *The city number improves but segment S6 regresses. What do you do, and why not just widen S6?*

**Strong answer.** Exclude S6 from the launch, or switch it to a one-sided upper quantile if that passes the gate. Widening S6 by about 3 minutes would add width cost on 8,000 requests per day — roughly \$2,680/day at `e = 1.45` — to recover 160 breach events costing about \$688/day, so the fix costs more than the defect. Exclusion is reversible and keeps the guardrail honest; a blanket widening converts a modelling defect into a permanent product cost.

**Reasoning.** The regression's cause is the symmetry of the interval against a right-skewed residual, so the correct fix is a method change, not a parameter change. Until the method change is validated, the honest scope is "S1–S5".

**Follow-up.** *What if the product owner insists on covering S6 at launch?* Then the requirement is a skew-aware method that passes the same gate on S6, with the width cost declared up front; or acceptance of a documented sub-90% coverage exception, which needs the product owner's explicit signature and a rider-communication plan.

**Acceptable alternatives.** Splitting S6 into sub-segments where the residual is more symmetric is also reasonable if sample size allows.

**Red flags.** Averaging S6 away in the city number; widening without pricing the width; declaring the method broken when one segment fails.

**Scoring.** 3: excludes or changes method, prices the widening, explains reversibility. 2: excludes without the arithmetic. 1: widens. 0: reports the city number as success.

**Support.** Sections 6.5, 6.7, DEC-05.

**Q07 — Latency.** *Your quote path must fit 120 ms p99. Stage p99s sum to exactly 120 ms. Is the design compliant?*

**Strong answer.** Not demonstrably. Summing stage p99s assumes perfect anti-correlation; in practice stages correlate (a snapshot cache miss drives both feature latency and model input size), so the true end-to-end p99 exceeds the sum's implication. The design should target about 90 ms on the hot path and treat 120 ms as an alarm threshold, with an error-budget review and failure injection to measure the real distribution.

**Reasoning.** Percentiles do not add. The honest claim is conditional: "fits if the stages are independent, which they are not, so measure it".

**Follow-up.** *What would you shed first under load?* The calibrated path in the highest-volume zone (S5, 29% of requests), keeping it in the segments where the pad is worst (S1, S3, S6) — this preserves most of the coverage benefit per unit of load shed.

**Acceptable alternatives.** Caching per (zone, hour bucket) quotes for near-identical requests is defensible if freshness permits.

**Red flags.** "The sum is 120 so we're fine"; ignoring correlation; proposing to drop the ledger write to save latency (violates C-04).

**Scoring.** 3: rejects the sum, explains correlation, proposes measurement and shedding. 2: rejects the sum. 1: accepts the sum. 0: adds stages to "average out".

**Support.** Section 8b.

**Q08 — Fallback design.** *The traffic snapshot is 30 minutes old. What does the system publish, and why not the last calibrated interval?*

**Strong answer.** The legacy pad, with `fallback_reason=snapshot_stale` logged. A calibrated interval computed against a snapshot outside its freshness bound is calibrated for conditions that no longer hold, and its appearance of calibration is worse than the pad's known bounded error. Serving the last calibrated interval has the same defect with an extra failure mode: it is stale *and* looks fresh.

**Reasoning.** The interval's validity rests on the residual distribution under the snapshot regime; when the regime is unknown, the guarantee is void. Degrading to a bounded, understood baseline is the safety property.

**Follow-up.** *How do you keep the fallback from becoming the default?* Split coverage reporting by publishing path, alarm on fallback rate above 5% sustained, and include a fallback-rate predicate in the promotion gate.

**Acceptable alternatives.** Serving a widened pad during known-bad snapshots is acceptable if the widening is derived from measured residuals in that regime, not invented.

**Red flags.** "Use the last good value"; "the model will handle it"; no logging of the fallback reason.

**Scoring.** 3: pad + reason + why not stale interval + anti-default control. 2: pad + reason. 1: stale interval. 0: publish anyway.

**Support.** Sections 8b, 12 F-02, 12 F-07.

**Q09 — Statistical judgment.** *A colleague says "with 240 residuals per group and 540 groups, we have 129,600 residuals, so coverage is precise." What is wrong?*

**Strong answer.** The guarantee is per group, not pooled. Each group's interval is built from its own 240 residuals, and the finite-sample floor `0.90 − 1/241 = 89.59%` applies per group. Pooling residuals across groups would only be valid if the groups shared a residual distribution, which is the hypothesis the grouping exists to reject. Precision in the pooled count does not transfer to any individual group's coverage claim.

**Reasoning.** Conformal coverage is a statement about exchangeable draws within the calibration set that produced the interval. Aggregating counts across non-exchangeable sets produces a number with no coverage meaning.

**Follow-up.** *How would you increase per-group precision without more data per group?* Pool hierarchically: use a parent group's quantile with a group-level correction, or use a conformal method that shares information across groups under a stated similarity assumption — and then validate the coverage empirically per group rather than assuming it.

**Acceptable alternatives.** Merging groups where residuals are statistically indistinguishable is valid if the merge is tested, not assumed.

**Red flags.** Believing pooled `n` gives per-group precision; ignoring the floor; claiming 90% exactly.

**Scoring.** 3: per-group guarantee, floor, pooling invalid without exchangeability, hierarchical remedy. 2: per-group point. 1: vague. 0: agrees with the colleague.

**Support.** Sections 6.3, 4 A-05, 6.7.

**Q10 — Experimental design.** *Design the exposure that measures `e`. What is the unit of randomization, the outcome, and the stopping rule?*

**Strong answer.** Randomize quote width at the zone-hour level (switchback) rather than per request, because per-request randomization within a zone creates interference: riders in the same zone share driver supply, so a wider quote shown to one rider changes driver availability for another. Outcome: request-to-completion rate, with contribution margin as the economic translation. Stopping rule: predeclared minimum sample (about 36,942 requests per arm for a one-point effect absent clustering) plus a safety stop if any segment's breach rate exceeds the inverse-coverage guardrail during exposure.

**Reasoning.** Interference is the reason per-request randomization fails here: the treatment changes a shared resource. Switchback trades statistical efficiency for validity.

**Follow-up.** *How do you handle the clustering in analysis?* Analyze at the zone-hour cluster level with cluster-robust standard errors, or use a design-effect inflation `1 + (m−1)ρ` with `ρ` estimated from pre-period data.

**Acceptable alternatives.** A stepped-wedge design across zones is acceptable and may be operationally easier.

**Red flags.** Per-request randomization without discussing interference; no stopping rule; measuring clicks instead of completion.

**Scoring.** 3: switchback, interference rationale, outcome, stopping rule, clustering analysis. 2: three of four. 1: A/B test per request. 0: no design.

**Support.** Sections 9.2 V4, 9.3.

**Q11 — Counterfactual reasoning.** *The pilot shows coverage 91% and contribution up 2%. Can you attribute the contribution gain to the quote policy?*

**Strong answer.** Not without the randomized comparator. Coverage improvement is plausible mechanically, but contribution moves with supply, price, weather, and matching changes. If the pilot ran concurrently with any of those, the attribution is confounded. The defensible claim is conditional: "under the randomized width exposure with no concurrent changes, the measured completion difference was X with interval Y".

**Reasoning.** Observational before/after comparisons in a marketplace absorb every concurrent change. The case's own ledger depends on `e`, which is exactly the parameter a randomized exposure identifies and a before/after cannot.

**Follow-up.** *What if randomization was impossible?* Use a difference-in-differences with a matched control zone, declare the parallel-trends assumption, and test it on pre-period data; report the estimate as conditional and weaker.

**Acceptable alternatives.** Regression discontinuity at a width-cap threshold is defensible if a threshold exists.

**Red flags.** Claiming causality from a before/after; adding up multiple metric improvements; ignoring concurrent launches.

**Scoring.** 3: rejects attribution, names confounders, proposes the identifying design. 2: names confounding. 1: accepts with caveats. 0: claims causality.

**Support.** Sections 9.2, 9.5, 11.

**Q12 — Changed constraint.** *The coverage target drops to 80% and the width cap tightens by 40%. What happens to the design?*

**Strong answer.** `alpha` rises to 0.20, so `q̂` shrinks substantially — at `n = 240`, `k = ceil(241 × 0.80) = 193`, so the 193rd residual replaces the 217th, typically a materially smaller value. Width falls, breach rate rises to about 20%, and the ledger's savings term shrinks while the width cost falls faster, so the break-even elasticity moves up and the policy becomes easier to justify economically. The per-segment gate still applies at 80%, and S6 may now pass because its breach rate under the symmetric interval was 12%, which is below a 20% target.

**Reasoning.** Both constraints move in the same direction economically: less coverage bought, less width paid. The interesting part is that a constraint change can rescue a failing segment without any modelling work, which is why the gate must be re-evaluated against the new contract rather than carried over.

**Follow-up.** *What breaks?* Rider trust, if 20% breaches generate support volume above capacity; and the credit policy, if it assumes 90% coverage. Both need re-checking before the target change is accepted.

**Acceptable alternatives.** Keeping 90% coverage and narrowing width by improving the point model is the harder, better path if time permits.

**Red flags.** Assuming the interval width is invariant to the target; forgetting the credit-policy dependency.

**Scoring.** 3: recomputes `k`, traces width and ledger, notes S6 may pass, flags trust/credit dependencies. 2: recomputes `k` and width. 1: says "width shrinks". 0: no change.

**Support.** Sections 6.3, 6.6, 13.

**Q13 — Operational judgment.** *Support reports a spike in breach complaints but your dashboard shows coverage at 91%. What do you check first?*

**Strong answer.** Whether the dashboard and the complaints measure the same population. Complaints come from riders who experienced a breach *and* chose to contact support; the dashboard counts all eligible requests. Check: (a) complaint rate per 1,000 breached promises, which separates more breaches from more complaints per breach; (b) fallback share, since a fallback storm can raise blended coverage; (c) segment split, since a city aggregate can hide an S6-like regression; (d) whether the credit policy changed, which changes complaint propensity without changing breaches.

**Reasoning.** A dashboard and a complaint queue are two different instruments with different denominators; disagreement between them is information about the measurement, not necessarily about the world.

**Follow-up.** *If complaint rate per breach doubled with constant breach rate, what then?* Investigate the quote presentation (did the interval render differently?), the credit policy, and support routing — the defect is in the experience or the process, not the model.

**Acceptable alternatives.** Starting with the segment split is also defensible and often fastest.

**Red flags.** Assuming the dashboard is wrong; assuming the complaints are noise; widening the interval as a first response.

**Scoring.** 3: separates rate-per-breach from breach rate, checks fallback and segments, considers policy change. 2: checks segments. 1: checks one thing. 0: widens.

**Support.** Sections 9.1, 12 F-07.

**Q14 — Model selection.** *Why not just train a quantile regression at τ = 0.9 and publish that?*

**Strong answer.** It is a legitimate alternative and may be better for skewed segments, because it targets the upper quantile directly instead of spending width symmetrically. What it lacks is a finite-sample coverage statement without additional machinery, and it requires two models (or a multi-quantile model) to be trained and monitored. The case chose conformal for the launch because the coverage contract is explicit and the method's guarantee is checkable; a team with a validated quantile model and a coverage harness could reasonably choose otherwise. The non-negotiable part is that whichever is chosen is evaluated at the published bound, per segment.

**Reasoning.** The choice is between a method with a distribution-free guarantee and a method with better shape adaptation. That trade depends on whether the contract is coverage-first or width-first.

**Follow-up.** *How would you validate the quantile model's coverage?* Empirically, on held-out windows and per segment, with the same rolling monitoring; there is no free guarantee, so the harness carries the burden.

**Acceptable alternatives.** Asymmetric conformal on signed residuals is a third option with a guarantee and skew adaptation, at higher bookkeeping cost.

**Red flags.** "Quantile regression is more modern"; claiming a coverage guarantee it does not have; not evaluating at the bound.

**Scoring.** 3: names the trade, the missing guarantee, and the non-negotiable evaluation. 2: names the trade. 1: preference without reason. 0: dismisses.

**Support.** Sections 5c, 6.3, 13.

**Q15 — Governance.** *An auditor asks you to prove that the quote a rider saw on a specific date was the one your system computed. What must exist?*

**Strong answer.** A ledger row keyed by `request_id` containing the published `L` and `U` (not just the model output), the model artifact hash, the quantile-table version, the snapshot `effective_at` and version used, the group key, the fallback reason if any, and the timestamp. With those, the quote is reconstructible and the decision is auditable. Without the published interval, the audit fails even if the model output is logged.

**Reasoning.** The auditable object is the decision, which is the published number. Logging the model output answers a different question.

**Follow-up.** *What if the ledger write failed for that request?* Then the audit gap is itself the finding, and the completeness SLI (≥ 0.999) is the control that bounds how often this can happen. Acknowledged write-ahead logging before response is the preventive control.

**Acceptable alternatives.** An immutable event stream plus a materialized view is equivalent if the view is versioned.

**Red flags.** "We log the model output"; no artifact versions; reconstructing from a mutable table.

**Scoring.** 3: lists published interval + versions + fallback + snapshot, names the completeness control. 2: lists most fields. 1: says "we have logs". 0: no audit path.

**Support.** Sections 5e, 8b, 4 C-04.

**Q16 — Prioritization.** *You have one engineering sprint. Rank: better point model, S6 skew fix, ledger completeness, latency headroom.*

**Strong answer.** Ledger completeness first — it is a governance control (C-04) and cheap; without it nothing else is defensible. Then S6 skew fix, because it is the only item that unblocks the guardrail and it is bounded work. Then latency headroom, because the current design has no margin for correlated stage tails and load will only grow. Last, the point model: the case's own evidence shows mean accuracy improvements buy almost nothing at the promise.

**Reasoning.** The ranking follows from what breaks the decision versus what improves it marginally. Governance and guardrail gaps block; latency is a latent risk; model quality is a diminishing return for this loss.

**Follow-up.** *What if the sprint must also ship a feature?* Ship the S6 fix behind a flag and keep the pad as default, so the sprint delivers a validated method without exposure.

**Acceptable alternatives.** Putting latency first if a known peak event is imminent is defensible.

**Red flags.** Point model first; "all four in parallel"; no rationale tied to the decision.

**Scoring.** 3: ranks with reasons tied to blocking risk and marginal value. 2: reasonable order, thin reasons. 1: arbitrary. 0: model first.

**Support.** Sections 6.8, 8b, 12.

**Q17 — Failure reasoning.** *The quantile table read fails in one availability zone. Walk through what riders see and what you do in the next ten minutes.*

**Strong answer.** Riders in that zone get the legacy pad, because the design's fallback is per-request and the table read failure triggers `fallback_reason=table_unavailable`. Coverage in that zone reverts to baseline (~13.8% breach city-wide; zone-specific for that segment), which is worse than the calibrated policy but bounded and understood. Next ten minutes: confirm the failure is localized (cache or AZ) rather than global, check whether the signed bundle can be re-read from the warm replica, and if the failure is a broader cache eviction, shed calibrated quoting for the highest-volume zone to protect latency while restoring the table. Do not serve a stale in-memory table from before the last recalibration, because its residuals may no longer match the serving model.

**Reasoning.** The fallback is a safety property, so the immediate behaviour is already correct; the ten-minute work is scope assessment and protecting the rest of the system.

**Follow-up.** *How would you have prevented it?* In-process cache of the table with a version check, a warm replica per AZ, and a synthetic canary per AZ that quotes a fixed request and asserts the expected interval.

**Acceptable alternatives.** Serving the parent-group table if it is still readable is fine and better than the pad.

**Red flags.** Serving stale intervals; failing the request entirely; no fallback reason logged.

**Scoring.** 3: pad + reason + scope assessment + prevention. 2: pad + reason. 1: retry until success. 0: serve stale.

**Support.** Sections 8b, 12 F-07.

**Q18 — Research judgment.** *What single measurement would most change your mind about this whole design?*

**Strong answer.** The completion elasticity `e` in this market, measured with a randomized width exposure and reported with a confidence interval. It is the only parameter that can flip the recommendation from a \$4,400/day gain to a \$7,200/day loss, and no modelling improvement touches it. A close second is whether publishing an interval rather than a point changes rider behavior independent of width, because that would invalidate the ledger's structure rather than just its magnitude.

**Reasoning.** Distinguishing a parameter that changes the sign of the decision from one that changes its precision is the core of applied judgment. `e` is sign-changing; residual sample size is precision-changing.

**Follow-up.** *If `e` came back at 2.5 pp/min, what would you do with the interval work?* Keep the S4 and S2 narrowing (which reduces width), abandon the widening segments, and revisit the coverage target downward — the method survives, the parameter choice changes.

**Acceptable alternatives.** Naming the exchangeability question under regime change is also a strong answer, because it determines whether the guarantee is real at all.

**Red flags.** "More data would help"; naming a modelling improvement; no distinction between sign-changing and precision-changing.

**Scoring.** 3: names `e`, explains sign-changing, proposes the measurement, notes the runner-up. 2: names `e` with reason. 1: vague. 0: model quality.

**Support.** Sections 6.6, 6.7, 9.3, 11.

**Takeaway for Section 14.** The pack is designed so that a candidate who only knows the method fails the economic and diagnostic questions, and a candidate who only knows the business fails the arithmetic and leakage questions.

## 15. Quick Reference Card

| Field | Card |
|---|---|
| Decision | Publish a calibrated arrival interval `[max(2, f − q̂), f + q̂]`, the legacy point-plus-5-minute pad, or defer, per request |
| Owner and baseline | Rider-experience product owner; baseline is the existing point estimate plus a fixed 5-minute pad |
| Evidence / stage / review | ILLUSTRATIVE arithmetic / proposed / draft, `last_reviewed: null` |
| Core computation | `k = ceil((n+1)(1−α))`; `q̂ = r_(k)`; floor `= (1−α) − 1/(n+1)`; at `n=240, α=0.10`: `k=217`, `q̂=9.2 min`, floor 89.59% |
| Result on the illustrative population | Breach rate 13.80% → 9.855% city-wide (5,520 → 3,942 events/day); S6 regresses 10.0% → 12.0% and fails the guardrail |
| Economic gate | `net = 6,785 − e × 467,712`; break-even `e = 1.45` pp of completion per minute of width; **`e` is unmeasured** |
| Must retain | Published `L` and `U` (not just `f`), model hash, table version, snapshot `effective_at`, group key, fallback reason |
| Stop condition | Any segment below 90% coverage, fallback rate above 5% sustained, p99 quote latency above 120 ms, or `e` upper bound at or above 1.45 |
| Never do | Serve a calibrated interval from a stale snapshot, a divergent model, or a stale in-memory table; widen to hide a coverage failure; report a blended coverage number without splitting by publishing path |

**One-sentence summary.** The interval beats the pad where the pad was wrong in one direction, fails where the residual is skewed, and its economics hinge on a behavioural elasticity that must be measured before any switch-over — so the deliverable is a gated shadow launch, not a rollout.
