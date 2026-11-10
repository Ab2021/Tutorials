---
title: "Dashboard Design and Operational Alerts"
topic: "Dashboard design, signal hierarchy, and paging thresholds"
collection: "module-07-analytics-metrics-bi-product-sense"
file_id: "03"
level: "foundation"
status: "draft"
version: "2.0"
created: "2026-11-09"
last_reviewed: null
content_status: "draft"
reviewed: null
review_trigger: "Alert threshold, cadence, SLO target, error-budget policy, data cut-off, or paging owner changes"
notation_style: "plain-text"
prerequisites: ["ratio arithmetic", "reading a time series", "basic probability (normal and Poisson tails)"]
pattern_tags: ["business-domain-analytics"]
evidence: "DERIVED"
source_status: "OPEN"
owner_role: "analytics lead"
source_index: "INDEX.md"
---

# Dashboard Design and Operational Alerts

## 1. File Header

> **Purpose:** Design a dashboard that changes a named decision, and an alert that pages a named person for a reason worth waking them, with both quantities derived from arithmetic rather than taste.
> **Prerequisites:** ratio arithmetic; comfort reading a time series with a baseline; the normal and Poisson tails well enough to look up a z-score; the distinction between event time and processing time.
> **Key outcome:** Given a decision, a cadence, and a harm rate, you can state the alert's threshold, its expected false-alarm rate, its detection latency, its cost per page, and the person who acts on it — and you can show that all five are consistent with each other.
> **This file does NOT cover:** building the charts in a particular BI product, causal attribution of a movement to a release, experiment design, or incident command procedure beyond the paging decision.
> **Level:** Foundation, with production-grade alerting arithmetic in Sections 4 and 5.
> **Evidence:** All mechanisms are DERIVED; every number below is ILLUSTRATIVE input with VERIFIED arithmetic; no vendor limit, benchmark, or company incident is claimed.
> **Retrieval cue:** If someone asks "why did this page fire" or "what should be on this dashboard", start at Section 2 (the signal ladder) and Section 4.5 (breakeven latency).

A dashboard is not a picture of the business; it is a machine that converts a repeated question into a repeated decision. Everything in it either serves that conversion or it is decoration that costs attention every morning. The useful design question is therefore never "what should we show" but "who will look at this, on what cadence, and what will they do differently depending on what they see". If the second half of that sentence has no answer, the first half does not matter.

The same logic applied to alerting produces a stricter standard, because an alert spends a scarcer resource than attention: it interrupts a person who may be asleep, and it teaches that person what to believe about the system's noise level. A pager that fires ten times a week for conditions nobody acts on does not merely waste four hours; it trains its recipients to dismiss the next page, which is the failure that turns a detectable incident into an outage with a long duration. This sheet therefore treats the false-alarm rate, the detection latency, and the cost per page as design outputs to be computed, not as qualities to be hoped for.

Three scope boundaries are worth stating immediately. First, this is about operational surfaces — dashboards and alerts read on a cadence to make an operating decision — and not about exploratory analysis, whose value depends on freedom rather than on discipline. Second, the arithmetic assumes the metric definition is settled; if the numerator or denominator is still being argued about, the correct next step is the metric contract in File 01, not a threshold. Third, the numbers in the worked example are constructed to exercise the formulas and describe no real service; only the algebra transfers.

The characteristic failure this sheet exists to prevent is a surface that is arithmetically impeccable and operationally inert: correct totals, correct rates, no threshold calibrated to noise, no cadence matched to the decision, and no owner. Such a surface usually looks like success in a design review, because every panel can be justified individually. The discipline that prevents it is to design from the back — decide the action and the latency the action tolerates, then derive the cadence, the window, and the threshold that fit inside that latency, and only then choose what to draw.

**Takeaway:** a dashboard is a decision machine and an alert is an interruption with a price; both are designed backward from the action and its tolerable latency, never forward from the available data.

## 2. Concept Map / Mental Model

Three audiences read operational surfaces, and they need different artifacts with different refresh rates, different tolerances for error, and different notions of "now". Conflating them produces the classic artifact that nobody opens: a page dense enough for an engineer investigating, aggregated enough for an executive scanning, and therefore useless to both.

```text
AUDIENCE            QUESTION                       CADENCE      TOLERABLE LATENCY
on-call engineer    is the service healthy now     seconds      seconds to a minute
service owner       is the trend inside policy      daily        hours
business owner      is the outcome moving           weekly       days to weeks
```

The right-hand column is the constraint that determines everything else. If the business owner's decision is "renew the vendor contract next quarter", a real-time refresh adds nothing and costs a pipeline. If the engineer's decision is "roll back the deploy now", a nightly batch is worthless no matter how accurate it is. Cadence and latency are properties of the decision, so they are specified before the data model.

The second structural idea is the signal ladder: a strict ordering of metrics by how directly they indicate harm, with a rule that only the top rungs may page. The ladder exists because the alternative — paging on anything that looks alarming — produces an unbounded number of pages as the metric catalogue grows, and the page volume grows faster than the team's ability to triage it.

```text
TIER 0  customer-visible harm        error rate, latency at the SLO boundary,
                                     failed checkouts, auth failures
                                     -> page immediately, one page per symptom class

TIER 1  imminent harm               saturation of a dependency (queue depth,
                                     connection pool, replica lag, disk), certificate
                                     expiry inside the reaction window
                                     -> page if the predicted time-to-harm is
                                        shorter than the reaction window

TIER 2  leading indicator           conversion drift, retry rate, cache hit ratio,
                                     deploy-freeze violations
                                     -> ticket, reviewed on a daily cadence

TIER 3  explanation                 per-segment breakdowns, cohort cuts, funnels
                                     -> read on demand while diagnosing T0/T1

TIER 4  exploration                ad-hoc cuts, correlations, forecasts
                                     -> no notification ever
```

The rule that gives the ladder force is that a Tier 2 metric may never page, even when it is badly out of range, because paging on a leading indicator is a bet that the indicator predicts harm better than the harm metric does. When that bet is right, it is right only for a specific, measured lead time; without that measurement the page is a hypothesis wearing an alarm. The permitted exception is a Tier 2 metric with a *measured* lead time and a pre-registered prediction interval, promoted deliberately and reviewed after every page it generates.

A third idea is the difference between a *threshold* and a *trigger*. A threshold is a level on one metric. A trigger is a rule over one or more metrics, a window, and a persistence requirement, and it is the trigger — not the threshold — that determines the false-alarm rate. Two designs with identical thresholds but different windows have false-alarm rates that differ by an order of magnitude, which is why thresholds quoted without windows are not specifications.

The fourth idea is the funnel every alert passes through before it becomes work, and the funnel's shape explains most alerting pathologies.

```text
evaluated conditions   ~ 3,000 per day on a modest surface
detected excursions    ~ 30 per day at a 1% per-check rate
survived persistence   ~ 8 per day after debounce and multi-window rules
paged                  ~ 5 per day after tiering and routing
actionable             ~ 1.5 per day
found a real defect    ~ 1 per day
```

Each stage multiplies by a factor, so a change at the top changes the bottom much more than it appears to. Raising the per-check false-alarm rate from one percent to five percent on three thousand checks turns eight routed signals into forty, and the team's response to forty is to stop reading them. This is the arithmetic reason that "we will just triage faster" fails as an alerting strategy: triage capacity is roughly constant while the queue is not.

The fifth idea is that a dashboard's cost is dominated by its *dimension space*, not its panel count. A panel showing six metrics across four dimensions is cheap; a panel whose filter lets a reader select any of six dimensions, each with hundreds of values, is expensive in a way that is invisible until the storage bill arrives. The concept map below makes the two kinds of growth visible side by side.

```text
number of panels            grows linearly with questions asked
number of series            grows as product of dimension cardinalities
points stored per day       series * 86400 / sample_interval_seconds
bytes stored                points * bytes_per_point
scan time for one view      bytes / effective scan bandwidth
```

A dashboard's refresh policy, its query cost, and its retention are all functions of the third line, so the third line should be estimated before the panels are drawn. Sections 4 and 9 carry the arithmetic; the mental model needs only the shape: cardinality is multiplicative, panels are additive, and multiplicative growth wins.

The sixth idea concerns what a surface must *not* contain, and the list is short enough to memorise. It must not contain a number without its denominator, a rate without its population, a total without its window, a comparison without its comparator, or a trend without its cutoff. Each omission is a licence for a reader to fill in the missing piece from memory, and readers fill gaps with the interpretation they already held.

Finally, a dashboard needs a stated refresh contract with the same three fields an alert needs: when the data is complete, what the surface shows before completion, and who is responsible when the two disagree. A surface that silently shows partial data is worse than a surface that shows nothing, because partial data is indistinguishable from a drop. The standard remedy is to render an explicit "incomplete through 14:05, 92 percent of expected volume arrived" banner, derived from a volume expectation rather than from a clock.

**Takeaway:** decide audience and latency first, place every metric on the ladder so only Tier 0 and Tier 1 can page, and treat cardinality rather than panel count as the cost driver.

## 3. Foundational Definitions

| Term | Symbol | Meaning | Units / domain | Nonexample | Check |
|---|---|---|---|---|---|
| Decision | `D` | The action a named role takes after reading the surface | role + action + deadline | "visibility" | replay last month's readings and count actions |
| Cadence | `c` | Interval between evaluations of a trigger | seconds | refresh rate of the chart | compare to tolerable latency |
| Sample interval | `dt` | Spacing of stored points | seconds | number of panels | `points/day = series·86400/dt` |
| Baseline | `b_t` | Expected value of the metric at the same seasonal position | metric units | last period's value | seasonal index recomputation |
| Robust spread | `σ̂` | Scale estimate resistant to contamination by the event it detects | metric units | standard deviation of the whole series | median absolute deviation |
| Threshold | `T` | Level at which a single evaluation is flagged | metric units | a trigger | per-check false-alarm rate |
| Trigger | `R` | Rule over metric, window, persistence, and routing | rule | a threshold | simulated alarm rate |
| Persistence | `k` | Consecutive evaluations that must breach before notifying | count | a hysteresis band | measured page count |
| Hysteresis | `h` | Band between the raise and clear levels | metric units | a single level | time-to-clear distribution |
| Error budget | `B` | Allowed bad-event count or bad time inside the SLO window | events or minutes | the SLO target itself | `B = (1-S)·N` |
| Burn rate | `x` | Actual bad-event rate divided by the budgeted rate | dimensionless | error rate | `x = r/(1-S)` |
| Window | `w` | Interval over which the statistic is computed | seconds | retention period | recompute at two `w` |
| Detection latency | `L` | Time from harm onset to the notification | seconds | time to resolve | onset-to-page trace |
| Breakeven latency | `L*` | Latency at which the cost of waiting equals the cost of a page | seconds | the page budget | `L* = C_page / (harm rate·unit cost)` |
| Actionable page | — | A page whose recipient takes a different action than they would have without it | boolean | a page that was read | weekly audit of pages acted on |
| Page precision | `P_p` | Actionable pages divided by all pages | proportion | recall | `P_p = A/P` |
| Alert coverage | `C` | Share of real incidents that produced a page before a customer report | proportion | precision | incident review, both directions |
| Cardinality | `K` | Number of distinct series after all dimension combinations | count | number of panels | distinct-series count per interval |
| Comparator | — | The value the shown number is compared against | metric units | an arrow | recompute comparison from raw rows |
| Cutoff | `t_c` | Instant at which the reported population is frozen | UTC instant | query run time | equality across panels |

**Decision.** The definition that matters is operational: a decision names a role, an action, and a deadline. The test is retrospective and cheap — read the surface's own history for the last month and count how many times someone did something because of it. A surface that survives that test is a decision surface; one that does not is a monitoring artifact, which is fine as long as it is labelled that way, because unlabelled monitoring is mistaken for decision support and then judged by the wrong standard.

**Cadence and sample interval are not the same quantity.** Cadence is how often a human or a rule evaluates; sample interval is how finely the underlying series is stored. A trigger evaluated every five minutes over a series sampled every minute has five points per evaluation and a cadence of 300 seconds. Conflating them produces two errors: setting the sample interval to match a weekly review (destroying the ability to compute a percentile), or setting the evaluation cadence to match the sample interval (multiplying the number of statistical tests without improving latency).

**Baseline.** The baseline is not "the previous period". It is the expected value at the same seasonal position, and the difference matters whenever the cycle is strong. In the worked example below, Monday runs 24.7 percent above the weekly mean, so a baseline of "yesterday" makes every Monday look like a spike and every Sunday look like a recovery. The baseline must also be robust: if it is computed as a mean over a window that includes an incident, the incident raises its own baseline and the next occurrence is less visible.

**Robust spread.** The standard deviation of a series that contains the event you are trying to detect is inflated by that event, which raises the threshold and delays detection of the next one. The median absolute deviation is the standard defence: with a normal underlying distribution, `σ̂ = 1.4826 × MAD`, and a single large excursion moves the median and the MAD far less than it moves the mean and the standard deviation. The cost is efficiency — the MAD is less precise than the standard deviation when the data really is clean — and that cost is worth paying on any series that must detect its own anomalies.

**Threshold versus trigger.** A threshold with no window and no persistence requirement has no defined false-alarm rate, so it cannot be reviewed. The reviewable object is the trigger: metric, window, statistic, level, persistence, routing, and the person who owns the level. Reviewing a threshold without the window is like reviewing a speed limit without saying over what distance the average is taken.

**Persistence and hysteresis.** Persistence requires `k` consecutive breaches before notifying, which cuts the false-alarm rate approximately by `α^k` for independent checks and costs `k-1` sample intervals of latency. Hysteresis sets a separate clearing level below the raising level, which prevents a metric hovering at the threshold from generating a page every few minutes. The two are complements: persistence suppresses transient excursions on the way up, hysteresis suppresses flapping at the boundary on the way down.

**Error budget and burn rate.** The error budget converts a service-level objective into an allowance, and the burn rate expresses consumption against that allowance in units that are comparable across objectives. An SLO of 99.5 percent over 30 days allows 216 minutes of failure time, or 150,000 failed attempts out of 30 million. A burn rate of 1 consumes the whole budget in exactly the window; a rate of 14.4 consumes one fiftieth of the budget in one hour, which is why 14.4 appears in multi-window paging policies.

**Detection latency and breakeven latency.** Detection latency is a property of the trigger; breakeven latency is a property of the economics. Comparing them is the whole design question: if the trigger's latency exceeds the breakeven latency, the alert is arriving after the point at which paging was worth it, and the correct response is either a faster trigger or no page at all. The breakeven latency formula in Section 4.6 is short enough to compute during a design review, which is the only time it can influence a design.

**Page precision and coverage.** Precision answers "of the pages we sent, how many mattered"; coverage answers "of the things that mattered, how many paged". They trade off against each other, and an alerting programme that optimises only one fails in a predictable direction: high precision with low coverage means customers report outages, and high coverage with low precision means nobody reads the pager. Both numbers require a post-hoc incident review, which is why alerting quality cannot be assessed from the alert configuration alone.

**Cardinality.** Cardinality is the number of distinct series after all dimensions are combined, and it multiplies: five regions, twelve plans, and six client versions give 360 series for a single metric, before any per-customer cut. Everything downstream — storage, scan cost, the number of statistical tests, and therefore the false-alarm rate — scales with that product, which makes dimension selection the highest-leverage decision in dashboard design.

Three definitional traps deserve separate treatment because they cause errors that look like arithmetic.

The first is treating a rate as if it were a count when sizing a threshold. A count has variance equal to its mean under a Poisson model, so a count threshold can be set from a Poisson tail. A rate's variance depends on the denominator as well, and a rate computed on 34 attempts has a much wider distribution than the same rate computed on 34,000. The threshold must therefore be set on the quantity that is actually being evaluated, and the quantity's denominator must be large enough for the threshold to exist at all.

The second is confusing the SLO target with the SLO measurement. The target is a policy (99.5 percent over 30 days); the measurement is a ratio computed over a window with a stated population and a stated definition of a good event. A team that argues about a burn-rate page is usually arguing about the measurement, and the argument is unresolvable until the good-event definition is written down, including which failures are excluded and on whose authority.

The third is assuming that a silent dashboard means a healthy system. Silence has at least four causes: the system is healthy; the pipeline stopped; the series is being evaluated but every value is null; and the trigger's condition can never be true because of a units error. Only the first is good news, and the check that distinguishes them is a heartbeat — an always-true condition that pages when it stops being true, evaluated by the same machinery as the real triggers.

**Takeaway:** define decision, cadence, trigger, budget, and breakeven latency before drawing anything; a threshold without a window is not a specification and a dashboard without an owner is not a decision surface.

## 4. Core Formulas with Full Derivation

Every result below answers a design question that must be answered before a surface is built. Each is stated with its validity condition, and each is followed by the failure boundary where the formula stops being usable.

### 4.1 Base case: the standardized deviation and a robust scale

**Validity before use.** The base case applies to a metric whose value at a seasonal position is approximately normal around a stable baseline, evaluated at a cadence whose interval is short relative to the baseline's own drift. It does not apply to counts near zero (use 4.4), to ratios whose denominator is small (Section 8, F-03), or to a series whose level changes by design.

**Derivation.** Let `x_t` be the metric and `b_t` the seasonal baseline. Define the deviation `d_t = x_t - b_t`. If `d_t` has mean zero and scale `σ`, then `z_t = d_t/σ` has mean zero and unit scale, and the probability that a single evaluation exceeds a two-sided level `L` is `α = 2(1 - Φ(L))`. Because the scale is estimated rather than known, use the median absolute deviation: `MAD = median(|d_t - median(d)|)` and `σ̂ = 1.4826·MAD`, the constant being the reciprocal of the 75th percentile of the half-normal.

```text
FORMULA [DB-01]: z_t = (x_t - b_t) / σ̂ ,   σ̂ = 1.4826 * MAD(d)
WHERE: x_t = metric at evaluation t [metric units]; b_t = seasonal baseline at the same
       position [same units]; d = deviations from a trailing window [same units];
       z_t = dimensionless; MAD = median absolute deviation [same units as d].
CONSTRAINT: deviations approximately symmetric and centred at zero in the baseline
       window; the window must exclude known incidents; at least ~30 deviations for
       the MAD to be stable; σ̂ > 0.
NUMERICAL EXAMPLE: median deviation 0 at a Monday position, MAD = 40 failures
       -> σ̂ = 1.4826 * 40 = 59.304 failures. An observation of 1,300 against a
       baseline of 1,180 gives z = 120/59.304 = 2.024.
INTERPRETATION: the observation is about two robust standard deviations above the
       Monday baseline, which is not unusual for a single evaluation (Section 4.2
       gives the exact per-check probability) and must not be paged on its own.
```

The boundary that matters is contamination of the scale estimate itself. If the trailing window contains an incident that produced ten large deviations, the MAD is barely moved — that is its purpose — but the *mean* used for centring is moved, which shifts every subsequent z-score. The fix is to centre on the median as well, and to compute both statistics over a window chosen so that no incident contributes more than a small fraction of the points.

### 4.2 General case: the false-alarm rate of a trigger over many evaluations

**Validity before use.** This result converts a per-evaluation level into a page rate, and it assumes evaluations are approximately independent. It is exact for the per-evaluation probability and approximate for the daily rate, because real series are serially correlated and correlated across series.

**Derivation.** With `N` independent evaluations per day and per-evaluation false-alarm probability `α`, the number of false alarms is binomial, so the expected count is `Nα` and the probability of at least one is `1 - (1-α)^N`. The per-evaluation probability for a two-sided normal level is `2(1-Φ(L))`. The two lines together are the entire arithmetic of alert fatigue, and they explain why "three sigma" is a folk remedy that fails at high cadence: three sigma is a one-in-370 event per evaluation, which is fine at one evaluation per day and catastrophic at one per five minutes.

```text
FORMULA [DB-02]: E[false alarms/day] = N * α ,  P(>=1) = 1 - (1-α)^N ,
                 α = 2 * (1 - Φ(L))
WHERE: N = evaluations per day [count] = 86400 / cadence; α = per-evaluation
       two-sided false-alarm probability [proportion]; L = threshold in units of σ;
       Φ = standard normal CDF.
CONSTRAINT: evaluations independent; deviations normal and stationary; one series
       only (correlated series need a family-wise correction, Section 4.7).
NUMERICAL EXAMPLE: cadence 5 min -> N = 288. At L = 3, α = 0.0027, so
       E = 288 * 0.0027 = 0.778 pages/day. At L = 4, α = 6.3e-5, so
       E = 288 * 6.3e-5 = 0.0181 pages/day.
INTERPRETATION: four sigma looks only one third larger than three sigma but reduces
       the false-page rate by a factor of about 43, because the normal tail is
       exponential in the square of the level.
```

The boundary is cadence above roughly 300 evaluations per day per series. At a 30-second cadence, `N = 2,880` and even a four-sigma level yields `0.181` false pages per day per series, so a surface with fifty such series is paging about nine times a day under the null. The correct response at that point is not a higher level — the level needed becomes absurd — but a longer evaluation window, which increases the signal-to-noise ratio without increasing the test count (Section 4.3).

### 4.3 Variants: smoothing, persistence, and multi-window evaluation

Smoothing and longer windows are the two legitimate ways to buy detection power without buying false alarms. They differ in what they cost: smoothing costs latency on the leading edge, and a longer window costs latency at both edges.

**Exponentially weighted moving average.** The EWMA statistic `z_t = λx_t + (1-λ)z_{t-1}` has, under an unchanged mean, a steady-state standard deviation of `σ·sqrt(λ/(2-λ))`, so a symmetric control limit at `L` in *original* sigma units is `L·σ·sqrt(λ/(2-λ))` wide in EWMA units. The interpretation is convenient: at `λ = 0.2` and `L = 3`, the limit width is exactly `3σ·sqrt(0.1111) = 1.0σ`, and the resulting in-control average run length is on the order of several hundred evaluations rather than 370.

```text
FORMULA [DB-03]: z_t = λ x_t + (1-λ) z_{t-1} ;
                 limit = L * σ * sqrt(λ/(2-λ))
WHERE: x_t = deviation from baseline [σ units]; λ = smoothing constant (0,1];
       z_t = smoothed deviation [σ units]; L = design level in original sigma units.
CONSTRAINT: x_t has constant variance; the process mean is stable in control; λ and L
       fixed in advance; ARL0 read from a table or simulation, not assumed equal to
       the unsmoothed case.
NUMERICAL EXAMPLE: σ = 90 failures, λ = 0.2, L = 3 -> width = 90 * 3 * 0.3333 = 90.0
       failures. A simulation of 4,000 in-control runs of standard-normal input gives
       an average run length to a false alarm of about 548 evaluations, i.e. 45.7
       hours at a 5-minute cadence.
INTERPRETATION: the smoothed rule tolerates a narrower band for the same page rate,
       which buys sensitivity to sustained shifts at the cost of latency on a spike.
```

The counterexample is a one-sample spike. EWMA with `λ = 0.2` moves the statistic by only 20 percent of the spike on the first evaluation, so a single-interval outage that is fully recovered by the next evaluation may never breach the limit. That is the intended trade for a metric whose excursions are transient, and it is exactly wrong for a metric whose single-interval failure is the harm being detected.

**Persistence.** Requiring `k` consecutive breaches multiplies the per-check false-alarm probability approximately `k` times under independence — `α^k` — while adding `k-1` intervals of latency. At `α = 0.01` and `k = 3`, the false-alarm probability per opportunity falls to `1e-6` while latency rises by 10 minutes at a 5-minute cadence. Persistence is the cheapest available improvement to precision, and it fails when breaches are not independent, which is the normal case for a metric that is drifting rather than spiking.

```text
FORMULA [DB-04]: α_persist = α^k ;  added latency = (k-1) * cadence
WHERE: α = single-evaluation false-alarm probability; k = required consecutive
       breaches [count >= 1]; cadence in seconds.
CONSTRAINT: breaches independent; the true signal persists at least k intervals,
       otherwise detection probability falls by the same power law.
NUMERICAL EXAMPLE: α = 0.01, k = 3, cadence 300 s -> α_persist = 1e-6 and added
       latency = 600 s. A 2-interval real excursion is detected with probability
       roughly 0.01 of the single-check rate, i.e. effectively never.
INTERPRETATION: persistence trades precision for latency and for sensitivity to short
       incidents; choose k from the shortest incident the owner must catch, not from
       the page rate alone.
```

**Multi-window burn rate.** For SLO-based alerting, evaluate several windows at once and page on the combination that is both fast and unlikely to fire on noise: a long window at a low burn multiple paired with a short window at a high multiple. The long window suppresses noise because it contains many events; the short window provides the latency, and the requirement that both agree prevents the long window from staying breached after the incident ends.

```text
FORMULA [DB-05]: burn(x, w) = (bad events in w) / (B * w / W)
WHERE: B = error budget for the whole SLO window [events]; W = SLO window length
       [seconds]; w = evaluation window [seconds]; x = burn rate [dimensionless].
       A policy pages when burn(x_long, w_long) >= X_long AND
       burn(x_short, w_short) >= X_short.
CONSTRAINT: the good-event definition is stable across both windows; the budget is
       expressed in the same unit as the counter; windows are nested so that the
       short window is inside the long one.
NUMERICAL EXAMPLE: SLO 99.5 percent over 30 days = 216 minutes of budget. At a
       2.5 percent failure rate the burn is 0.025/0.005 = 5.0. A 1-hour window at
       burn 14.4 consumes 14.4/720 = 2.0 percent of the budget.
INTERPRETATION: the pair (14.4 over 1 hour, 6 over 6 hours) pages within an hour on a
       severe incident and within six hours on a moderate one, while ignoring the
       isolated bad minute that a single short window would flag.
```

**Seasonal index.** When the cycle is strong, the baseline must be indexed by position rather than smoothed. Compute the index as the ratio of the position's mean to the overall mean, then divide the observation by the index before standardizing. The variant is not cosmetic: a flat threshold applied to a series whose Monday is 24.7 percent above its weekly mean will fire every Monday, and the team's response will be to raise the threshold until a real 39 percent drop on Sunday no longer reaches it.

**Robust scale on a seasonal series.** Combine the two variants: compute deviations against the indexed baseline, then take the MAD over several weeks of the same weekday. This costs at least four to six weeks of history before the threshold exists at all, which is the honest price of a seasonal baseline and the reason new series should be monitored rather than paged.

The validity condition shared by all four variants is that the quantity being standardised must be the quantity whose tail you care about. Standardising a rate with a count-derived sigma, or a count with a rate-derived sigma, produces thresholds that are wrong by the square root of the denominator, and the error is invisible in a chart because the line still crosses the band.

### 4.4 Count metrics: the Poisson threshold

**Validity before use.** Use this when the metric is a count of independent rare events in a fixed window, and when the baseline rate is stable within the window. It fails when events cluster (one bad host producing thousands of failures), when the window straddles a deploy, and when the count is capped by capacity rather than generated by demand.

**Derivation.** If events arrive independently at rate `λ` per window, the count in a window is Poisson with mean and variance `λ`. The upper threshold at tail probability `α` is the smallest `T` with `P(X >= T) = 1 - Σ_{i<T} e^{-λ}λ^i/i! <= α`. For large `λ` the Poisson is approximately normal with standard deviation `sqrt(λ)`, so the threshold is `λ + z_{1-α}·sqrt(λ)`; for small `λ` the normal approximation is unusable and the exact tail is required.

```text
FORMULA [DB-06]: T = min{ k : P(X >= k | λ) <= α } ;  approx T = λ + z_{1-α} sqrt(λ)
WHERE: λ = baseline expected count per window [events]; α = per-window false-alarm
       probability; T = integer threshold [events]; z_{1-α} = normal quantile.
CONSTRAINT: independent arrivals; stable λ inside the window; window boundaries not
       aligned with a scheduled change; clustering absent.
NUMERICAL EXAMPLE: baseline 0.25 percent failure on 1,000,000 attempts/day gives
       λ = 8.68 failures per 5-minute window. Exact tail: P(X>=15) = 0.032,
       P(X>=12) = 0.167, P(X>=20) = 0.0016. A 1-hour window has λ = 104.2, and a
       doubling to 208.3 is z = (208.3-104.2)/sqrt(104.2) = 10.2.
INTERPRETATION: a 5-minute count threshold at 15 gives a 3.2 percent chance of a
       false page per window, i.e. about 9 per day, while a 1-hour threshold detects a
       doubling at ten sigma - the short window is the wrong instrument for a
       doubling, even though it looks more responsive.
```

The boundary is the small-`λ` regime, where a single extra event is a large relative change. At `λ = 0.7` events per window, `P(X>=2 | 0.7) = 0.156` and `P(X>=3) = 0.034`, so any threshold that fires on two events has a per-window false-alarm rate near 16 percent. Counts that low cannot be alerted on per window; they must be accumulated into a longer window whose `λ` is large enough for the tail to have resolution, and the resulting latency must then be checked against the breakeven latency.

### 4.5 Alert economics: precision, cost, and the breakeven latency

**Validity before use.** This result prices a page against the harm it prevents, and it assumes the harm rate is roughly constant while the incident is unmitigated. It does not model the case where harm accelerates, and it does not include the second-order cost of a false page (eroded trust), which must be added as a separate term when pages are frequent.

**Derivation.** Let `C_page` be the cost of one page in engineer-minutes including context switch, `h` the harm per unit time while the incident is undetected in cost units per minute, and `L` the detection-plus-response latency in minutes. Waiting costs `h·L`; paging costs `C_page`. The two are equal at `L* = C_page/h`. If the trigger's latency is below `L*`, paging is cheaper than waiting; if it is above, the page arrives after the point at which it could have paid for itself.

```text
FORMULA [DB-07]: L* = C_page / h ;  h = (r_inc - r_base) * volume * cost_per_harm
WHERE: C_page = cost of one page [cost units, e.g. engineer-minutes]; r_inc, r_base =
       failure rates during and before the incident [proportions]; volume = events per
       minute [count/min]; cost_per_harm = cost per failed event [cost units];
       h = harm rate [cost units/min]; L* = breakeven latency [min].
CONSTRAINT: harm rate constant over the interval; the page is acted on within
       L*; the baseline is the true counterfactual, not the pre-incident average of a
       series that already contains the incident.
NUMERICAL EXAMPLE: page cost 20 engineer-minutes at 1 cost unit per minute = 20.
       Baseline failure 0.25 percent, incident 2.5 percent, volume 694.4 attempts/min.
       Extra failures = 0.0225 * 694.4 = 15.63/min; at 12 units per failure,
       h = 187.5 units/min; L* = 20/187.5 = 0.107 min = 6.4 seconds.
INTERPRETATION: against a payment failure costing 12 units, only a trigger that pages
       within seconds pays for itself; if the fastest available window gives 60 seconds
       of latency, the page costs more than the harm it prevents and the correct design
       is automated mitigation, not a faster pager.
```

The same formula at a lower harm rate reverses the conclusion, and the reversal is the useful part. At a 0.5 percent incident rate the breakeven latency is 57.6 seconds; at 0.3 percent it is 288 seconds, or nearly five minutes. That is why one service should page on a one-minute window and another should open a ticket on a fifteen-minute aggregation, even when their dashboards look identical. The formula also explains a common pathology: a page whose breakeven latency is six seconds is almost certainly an automated-mitigation trigger wearing a human's name.

```text
ILLUSTRATIVE breakeven latency by incident rate (0.25 percent baseline, 694.4/min,
12 units per failure, 20 units per page)
  incident 0.30 percent -> 0.35 extra failures/min -> h = 4.2  -> L* = 288 s
  incident 0.50 percent -> 1.74 extra failures/min -> h = 20.8 -> L* = 57.6 s
  incident 1.00 percent -> 5.21 extra failures/min -> h = 62.5 -> L* = 19.2 s
  incident 2.50 percent -> 15.63 extra failures/min -> h = 187.5 -> L* = 6.4 s
  incident 5.00 percent -> 32.99 extra failures/min -> h = 395.8 -> L* = 3.0 s
```

**Page precision.** Precision is `A/P` where `A` is actionable pages and `P` all pages. Its complement is the false-page load, and the load in engineer-minutes is `P·C_page`. A surface sending five pages a day at 30 percent precision delivers 10.5 actionable pages per week and 24.5 non-actionable ones, which at 20 minutes each is 490 minutes of non-actionable interruption per week, or about one fifth of a working week. That number, not the page count, is the argument that wins a review.

```text
FORMULA [DB-08]: toil/week = P_week * (1 - P_p) * C_page
WHERE: P_week = pages per week [count]; P_p = page precision [proportion];
       C_page = minutes per page [min]; result in minutes per week.
CONSTRAINT: precision measured by audit, not estimated by the alert owner; C_page
       includes recovery time, typically 15-25 minutes for a context switch.
NUMERICAL EXAMPLE: 25 pages/week at 40 percent precision and 20 minutes each gives
       25 * 0.60 * 20 = 300 minutes/week, i.e. 12.5 percent of a 2,400-minute week.
       At 6 pages/week and 80 percent precision the same formula gives 24 minutes.
INTERPRETATION: halving the page count halves the toil exactly, while doubling
       precision halves it too - both levers are linear, so the cheaper one is the
       one the team can actually execute this quarter.
```

The validity condition for the whole economics block is that the page has an owner who can act. If the recipient cannot mitigate the condition, the harm continues regardless of latency and the page's only effect is to transfer anxiety; the correct design in that case is an automated response or an explicit acceptance, both of which cost less than a notification nobody can use.

### 4.6 Multi-series evaluation and family-wise control

A dashboard is never one series. With `S` series evaluated at the same cadence at level `α`, the expected number of false pages per evaluation is `Sα` even when every series is healthy, and the probability of at least one is `1 - (1-α)^S`. The correction is to set the per-series level from the family-wise budget, which for a Bonferroni rule means `α/S` per series, or equivalently a level of `z_{1-α/(2S)}`.

```text
FORMULA [DB-09]: α_series = α_family / S ;  L = z_{1 - α_series/2}
WHERE: S = series evaluated per evaluation cycle [count]; α_family = acceptable
       probability of any false page per cycle; L = per-series level in sigma.
CONSTRAINT: series independent (conservative when correlated); thresholds are the only
       comparison being made; the family is declared, not chosen after the fact.
NUMERICAL EXAMPLE: S = 300, α_family = 0.05 -> α_series = 1.667e-4 -> L = 3.765.
       At S = 40 the same budget gives L = 3.227.
INTERPRETATION: the correction is cheap in threshold units - a third of a sigma at
       300 series - which is why it should be applied by default rather than argued
       about per alert.
```

The counterexample is worth stating because it is the common case: `S` series that share an upstream dependency are positively correlated, so a single pipeline fault flags all of them at once. The Bonferroni level then over-corrects for the shared fault while under-correcting for nothing, and the practical result is that a real outage produces 300 simultaneous pages. The remedy is grouping: page once per dependency, and let the group's members be listed inside the single page, which is a design decision about the trigger rather than about the threshold.

### 4.7 Limits and counterexamples

Six limits bound everything above, and each is a check that costs less than the incident it prevents.

The first is non-stationarity. Every formula in this section assumes a baseline that would have held without the incident; a series with a weekly trend, a product launch, or a pricing change does not have one. The check is a plot of the residual after removing the seasonal index, inspected for a slope; if the residual has a slope, the threshold must be recomputed on a shorter baseline or the metric must be differenced.

The second is contamination of the scale. A scale estimated over a window that contains a large incident is inflated, so the threshold is too high and the next incident is detected late. The check is to recompute `σ̂` with and without the largest 1 percent of deviations; a difference above roughly 20 percent means the estimate was contaminated and the threshold needs a robust replacement.

The third is denominator instability in a ratio. A rate alert on a ratio whose denominator swings by an order of magnitude has a variance that swings with it, so a fixed rate threshold has a false-alarm rate that varies by window. The check is to plot the denominator beside the rate; if the denominator moves by more than about a factor of two between adjacent windows, alert on the numerator count with the Poisson threshold instead.

The fourth is the multi-window conjunction's blind spot. A policy that requires both a long and a short window to breach will miss an incident that began just before the short window and ended just after the long window's evaluation, which is a narrow but real interval. The check is to simulate a two-interval incident at every offset within the long window and confirm that at least one evaluation catches it; if not, add a third window or shorten the cadence.

The fifth is the units error that cannot fire. A threshold expressed in percent against a series expressed as a fraction is off by a factor of 100, and the alert simply never fires. The check is a deliberate test injection: push a synthetic observation through the evaluation path and confirm that the notification arrives. Any trigger that has never been observed to fire in a test is a hypothesis, not a control.

The sixth is the heartbeat. Absence of data produces no evaluations, and no evaluations produce no pages, so a dead pipeline is indistinguishable from a healthy service on the pager. The check is an always-true condition evaluated on the same path with a stated maximum staleness; if the staleness exceeds it, page. Every surface needs one, and the one that is missing is usually the one belonging to the pipeline that would have paged.

| Result | Validity condition | Diagnostic when it fails | Typical consequence of ignoring it |
|---|---|---|---|
| DB-01 robust deviation | symmetric, incident-free baseline window | recompute `σ̂` excluding the top 1 percent | threshold drifts with each incident |
| DB-02 page rate | independent evaluations, stationary normal noise | compare observed page count to `Nα` | pages scale with cadence, not with harm |
| DB-03 EWMA limits | constant variance, fixed `λ` and `L` | simulate ARL0 rather than assuming 370 | band too tight, chronic paging |
| DB-04 persistence | independent breaches, signal longer than `k` | replay incidents at each offset | short incidents never detected |
| DB-05 burn policy | stable good-event definition, nested windows | replay a one-minute spike | pages on transients or misses slow burns |
| DB-06 Poisson threshold | independent arrivals, stable `λ` | compare variance to mean in baseline | clustered faults page hundreds of series |
| DB-07 breakeven latency | constant harm rate, actionable page | measure time to mitigation | paging after the harm has been paid |
| DB-08 page toil | precision measured by audit | count pages acted on | toil estimated optimistically |
| DB-09 family-wise level | declared family, independence | count simultaneous flags | one fault becomes many pages |

**Takeaway:** every threshold in this section is a tail probability multiplied by a number of opportunities; design the opportunity count and the tail together, then check the result against the economics of the harm.

## 5. Worked Numerical Example

The scenario is a checkout path that processes card payments. All inputs are ILLUSTRATIVE and describe no real service; every arithmetic step is recomputed in the table and re-checked at the end of the section.

**Decision and owner.** The on-call payments engineer decides whether to fail over to the secondary processor. The decision deadline is ten minutes from harm onset, because the processor contract caps the failover window and the support team's promise to customers is fifteen minutes. The business owner separately decides weekly whether the SLO target still reflects the customer promise.

**Population, grain, and window.** The counted unit is one payment attempt. The population is attempts received by the primary processor, excluding attempts rejected by the client before transmission (those are a separate, client-side metric with a different owner). The SLO window is a rolling 30 days; the evaluation windows are 5 minutes and 1 hour. The cutoff is 60 seconds after each minute closes, and every panel in the example uses that single cutoff.

**What is not available at decision time.** The processor's own incident status is available only when the processor chooses to publish it, which lags the harm. Chargeback outcomes arrive weeks later and cannot be used. The root cause is unknown at the ten-minute horizon, so the decision must be made on symptom evidence alone.

| Step | Rule | Inputs with units | Intermediate (full precision) | Reported | Independent check |
|---|---|---|---|---|---|
| 1 | volume | 1,000,000 attempts/day, uniform | 1,000,000/1,440 | 694.44 attempts/min | 694.44 × 1,440 = 1,000,000 |
| 2 | 5-min expectation | 694.44 × 5 | 3,472.22 attempts/5 min | 3,472 | 3,472.22/5 = 694.44 |
| 3 | baseline failures | 0.25 percent of step 2 | 3,472.22 × 0.0025 | 8.68 failures | 8.68/3,472.22 = 0.0025 |
| 4 | budget in events | 30 days × 1,000,000 × 0.005 | 30,000,000 × 0.005 | 150,000 failures | 150,000/30,000,000 = 0.005 |
| 5 | budget in minutes | 43,200 × 0.005 | 216.0 | 216 min | 216/(30×1,440) = 0.005 |
| 6 | baseline burn | 0.0025/0.005 | 0.5 | 0.5× budget rate | 0.5 × 216 = 108 min/month |
| 7 | incident burn at 2.5 percent | 0.025/0.005 | 5.0 | 5.0× | 5.0 × 216 = 1,080 min/month |
| 8 | 1-hour consumption at 14.4× | 14.4 × (60/43,200) | 14.4/720 | 2.0 percent of budget | 0.02 × 216 = 4.32 min |
| 9 | 6-hour consumption at 6× | 6 × 360/43,200 | 6/120 | 5.0 percent of budget | 0.05 × 216 = 10.8 min |
| 10 | 5-min threshold at α = 0.032 | smallest `k` with tail ≤ 0.032 | 15 failures | 15 | exact Poisson tail 0.0320 |
| 11 | per-day false pages at that level | 288 × 0.032 | 9.22 | about 9/day | 9.22/288 = 0.032 |
| 12 | 1-hour threshold for doubling | `λ = 104.17`, `2λ = 208.33` | (208.33−104.17)/10.21 | z = 10.2 | 104.17 + 3 × 10.21 = 134.8 |
| 13 | extra failures at 2.5 percent | (0.025−0.0025) × 694.44 | 15.63/min | 15.63/min | 0.0225 × 694.44 |
| 14 | harm rate | 15.63 × 12 | 187.5 units/min | 187.5 | 15.63 × 12 |
| 15 | breakeven latency | 20/187.5 | 0.1067 min | 6.4 s | 6.4 × 187.5/60 = 20 |
| 16 | Sunday index | 0.90/1.0429 | 0.863 | 0.863 | 0.863 × 1.0429 = 0.90 |
| 17 | flat threshold at 1.2× mean | 1.0429 × 1.2 | 1.2514 | 1.251 | 1.251/1.0429 = 1.2 |
| 18 | Monday breach margin | 1.30/1.2514 | 1.0388 | 3.9 percent over | 1.2514 × 1.0388 = 1.30 |

The table is ordered so that each row can be checked from the row before it, and the two rows that are most often skipped are steps 4 and 5, which convert the SLO target into an allowance in two different units. Doing both is not redundant: the event count is what the burn-rate policy consumes, and the minute count is what the customer promise is written in, and a team that only tracks one of them discovers during an incident that the two disagree about how much budget is left.

Step 8 deserves a note because the number 14.4 looks arbitrary until it is derived. One hour is 1/720 of a 30-day window, so a burn rate of `X` over one hour consumes `X/720` of the budget; setting that equal to 2 percent gives `X = 14.4`. Choosing 2 percent as the target is a policy decision — how much budget one hour of severe incident may consume — and it belongs next to the SLO target rather than inside the alert rule.

The weekly seasonality is the second half of the example, and it is what makes a flat threshold indefensible. The ILLUSTRATIVE index by weekday is 1.30 on Monday, 1.05 Tuesday, 1.00 Wednesday, 1.05 Thursday, 1.15 Friday, 0.85 Saturday and 0.90 Sunday, with mean 1.0429. A flat threshold at 1.2 times the mean — 1.2514 — sits 3.9 percent *below* the ordinary Monday level, so it fires every Monday, while Sunday sits at 0.863 of the mean and therefore at 0.719 of the threshold: a genuine 39 percent drop in Sunday volume would not reach it. Both errors come from one design choice, and both are invisible in a chart that shows the threshold as a horizontal line.

```text
weekday   index   ratio to mean   flat threshold 1.2514
Monday     1.30      1.247        breaches by 3.9 percent
Tuesday    1.05      1.007        inside
Wednesday  1.00      0.959        inside
Thursday   1.05      1.007        inside
Friday     1.15      1.103        inside
Saturday   0.85      0.815        inside, 28 percent of headroom unused
Sunday     0.90      0.863        inside, needs 39 percent rise to fire
```

The repaired version divides the observation by the weekday index before evaluating, and recomputes the scale from the indexed residuals. That repair is not free: the index must be estimated from at least four to six weeks per weekday, the estimate must be refreshed on a schedule, and a holiday week must be excluded or indexed separately. The cost is real and the alternative is an alert that pages weekly for weather.

The 5-minute versus 1-hour comparison is the third lesson, and it is the one that most surprises reviewers. Step 10 sets a 5-minute count threshold at 15 failures, because 15 is the smallest integer whose Poisson tail at `λ = 8.68` is at or below 0.032, and step 11 shows that this level pages about nine times a day under the null. It looks like an acceptable trade until the sensitivity is computed: a doubling of the failure rate over 5 minutes gives `λ = 17.4`, and `P(X >= 15 | 17.4)` is only about 0.30, so the trigger misses roughly seven of every ten windows during a doubling. The 1-hour window detects the same doubling at `z = 10.2`, which is to say it cannot miss it, at the cost of up to an hour of latency.

That trade is resolved by the economics of step 15, not by statistics. With a breakeven latency of 6.4 seconds, neither window is fast enough for the human page to pay for itself, which means the payments path needs an automated mitigation — a circuit breaker or a processor failover — and the human alert should be a *notification* of an action already taken rather than a request for one. Recognising this changes the design from "make the pager faster" to "automate the mitigation and alert on the automation's decision", which is a different project with a different owner.

**Independent rechecks.** Every reported number is re-derived here without reference to the table's intermediate column. Volume: 1,000,000/1,440 = 694.444, and 694.444 × 5 = 3,472.22. Failures: 3,472.22 × 0.0025 = 8.6806. Budget: 0.005 × 43,200 = 216 minutes, and 216/43,200 = 0.005. Burn: 0.0025/0.005 = 0.5 and 0.025/0.005 = 5.0. Consumption: 14.4/720 = 0.02 and 6/120 = 0.05. Breakeven: 20/(0.0225 × 694.444 × 12) = 20/187.5 = 0.10667 minutes = 6.4 seconds. Seasonality: 1.30/1.042857 = 1.2466, and 1.042857 × 1.2 = 1.25143, so 1.30/1.25143 = 1.0388.

**Sensitivity.** Three inputs dominate the conclusion, and they move it in different directions. If the harm per failed payment were 3 units rather than 12, the breakeven latency would be 25.6 seconds, which is inside the reach of a 1-minute window and reverses the recommendation from automation to paging. If the page cost were 60 engineer-minutes rather than 20 — a page at 03:00 that requires a full context reload — the breakeven latency would triple to 19.2 seconds at 12 units per harm. If the baseline failure rate were 0.5 percent instead of 0.25, the burn at a 2.5 percent incident would be 5.0 against a budget rate of twice as much, halving the burn multiple and doubling the time before a burn-rate policy fires.

The combination of the three sensitivities gives the honest statement of the example: the arithmetic is exact, the design conclusion is robust to the page cost and fragile with respect to the harm per failure, and the harm estimate is the number most worth measuring before the alert is written. That ordering — measure the harm first, then price the page, then choose the trigger — is the reverse of how alerting is usually designed, which is why so many triggers are calibrated to nothing in particular.

**Production-scale change.** At ten times the daily volume the arithmetic scales but three things change qualitatively. The Poisson approximation improves, so thresholds become relatively tighter and more subtler incidents become detectable; the number of simultaneous evaluations grows, so the family-wise correction matters more and grouping becomes mandatory; and the 5-minute window's `λ` grows to 86.8, at which point the window is long enough that detection latency rather than statistics becomes the binding constraint.

```text
FORMULA [DB-10]: minimum sample n for a quantile lower bound at level q
                 n >= ln(alpha) / ln(q)
WHERE: q = quantile level [proportion]; alpha = desired confidence that the sample's
       maximum is above the true q-quantile; n = required observations [count].
CONSTRAINT: independent observations from the same distribution; the quantile is
       lower-bounded by the sample maximum, not estimated by interpolation.
NUMERICAL EXAMPLE: q = 0.99, alpha = 0.05 -> n >= ln(0.05)/ln(0.99) = 298.1, so 299
       observations. With n = 90 the probability that the maximum exceeds the true
       99th percentile is 1 - 0.99^90 = 0.596.
INTERPRETATION: a latency alert on the 99th percentile needs about 300 observations
       per evaluation window; a 90-request window cannot bound the percentile at all,
       which is why low-traffic services must alert on counts or on means.
```

The quantile result is the one that most often goes missing from a dashboard design, because percentiles are drawn by default and their sample requirements are invisible in the chart. At 90 requests per window the 99th percentile is the maximum of the sample, and the maximum of 90 draws carries essentially no information about the tail; the chart will still render a line, and the line will move by a factor of two between adjacent windows on identical traffic.

**Remaining uncertainty.** The example does not establish that the processor is the cause of the failures, that the failover improves outcomes, or that the harm estimate of 12 units per failure is correct. It establishes the arithmetic connecting a volume, a rate, a budget, a burn multiple, a threshold, a page rate, and a latency bound, and it establishes that two of those quantities — the page rate at the 5-minute threshold and the breakeven latency — disagree about whether a human should be involved at all.

**Takeaway:** the worked example converts one volume into a budget, a burn multiple, a threshold, a false-page rate, a toil estimate, and a latency bound, and the last of those decides whether the alert should page a human or drive an automated action.

## 6. Visual / Geometric Intuition

Six drawings carry the geometry of this sheet. Each is schematic, each is labelled with the quantity that makes it readable, and each corresponds to an algebra already stated.

The first is the alert funnel from Section 2, drawn with a real evaluation count so the multiplication is visible rather than asserted.

```text
3,000 conditions evaluated per day
   |
   | per-check false rate 1 percent
   v
30 detected excursions per day          <-- this is the number that becomes pages
   |
   | persistence k = 2 removes isolated points (about 70 percent of them)
   v
9 survived persistence
   |
   | tiering removes Tier 2 and Tier 3 from paging
   v
5 paged
   |
   | 30 percent precision
   v
1.5 actionable
```

The funnel's shape is the argument: the top line is a property of the surface's size, the second line is a property of the threshold, and only the fourth line is a property of the decision. A team that wants fewer pages can shrink the surface, loosen the threshold, add persistence, or re-tier — and those four levers change different lines of the drawing, which is why "make the alerts quieter" is not a specification.

The second drawing is the threshold plane: two metrics on the axes, one trigger as a region rather than a line. Most useful operational alerts are conjunctions, and a conjunction's false-alarm rate is not the sum of its parts.

```text
                error rate
                 ^
   5% |..........|##########|
      |          |##########|
   2% |..........+----------+   <- page region: error >= 2% AND latency >= 800 ms
      |          |          |      (both must hold in the same 5-min window)
 0.5% |..........|..........|
      +----------------------+------> p95 latency
              400ms      800ms
```

The dotted region is where one condition holds and the other does not: a latency rise with a healthy error rate, or an error rise with healthy latency. Those are the two most common shapes of a *partial* degradation, and a conjunction deliberately ignores them. Whether that is right depends on the decision: if the mitigation is "fail over", a partial degradation may not justify it, and the conjunction is correct. If the mitigation is "shed load", a latency-only degradation is exactly the trigger, and the conjunction is a bug.

The third drawing is the burn cone: cumulative bad time against elapsed time, with the budget line and the trajectories that a multi-window policy distinguishes.

```text
cumulative bad minutes
216 |*                                        budget exhausted at 30 days
    | \                                      /
    |  \                                   /
    |   \                                /   <- steady 2x burn: budget gone in 15 days
    |    \                             /
108 |     \                          /
    |      \                      /
    |       \                  /
    |        \              /
    |         \. . . . . ./                  <- healthy trajectory (0.5x burn)
  0 +--------------------+------------------> days
    0        7        14       21       30
```

The slope is the burn rate, so the visual question "will this line hit the ceiling before the window ends" is exactly the arithmetic question `burn × elapsed > budget`. A 2x burn looks unremarkable on day 3 and is unrecoverable by day 15, which is why the policy evaluates a long window as well as a short one: the long window is the slope, the short window is the wiggle.

The fourth drawing is the latency-harm plane, which is the geometry behind the breakeven result.

```text
cost
 ^
 |            harm = h * L  (rises linearly with latency)
 |          /
 |        /                    page cost C_page (flat)
 |      /
 |    /            * breakeven at L* = C_page / h
 |  /
 |/_____________________________> latency L
      6.4s   20s   60s   300s
```

Below `L*` the page is cheaper than waiting, so latency reduction is worth engineering effort. Above `L*` the page costs more than the harm, and every additional second of latency reduction buys nothing; the line that matters becomes the harm rate `h`, which is reduced by automated mitigation rather than by faster notification. The same drawing explains why two services with identical latency profiles deserve different alerting: their `h` differs, so their `L*` differs, and the horizontal position of the crossing point is a business fact rather than an engineering preference.

The fifth drawing is the seasonality grid, which is where flat thresholds die.

```text
            Mon  Tue  Wed  Thu  Fri  Sat  Sun
index      1.30 1.05 1.00 1.05 1.15 0.85 0.90
flat 1.25   ^^^  ok   ok   ok   ok   ok   ok     <-- fires every Monday
indexed     ok   ok   ok   ok   ok   ok   ok     <-- fires on deviation within weekday
                                    |
                                    +-- Sunday at 0.86 of mean: a 39 percent fall
                                        is needed before a flat rule notices
```

The grid makes two errors visible at once: the false positive on the high day and the missed detection on the low day. Neither is visible in a time series with a horizontal threshold drawn across it, because the threshold looks reasonable on average and the average is exactly the thing seasonality breaks.

The sixth drawing is the cardinality grid, which is the cost model in picture form.

```text
                      plan (12)
                   /   /   |   \    \
        region (5) -- + -- + -- + -- +     5 * 12 = 60 series per metric
                   \   \   |   /    /
                      client (6)
                      |
              60 * 6 = 360 series per metric
                      |
              * 4 metrics = 1,440 series
                      |
              * 86400/60 s = 2,073,600 points per day
```

The multiplication runs downward, and every box in it is a design decision made by someone who was not thinking about the bottom line. Restricting the client dimension to the two versions currently in the field cuts the series count by two thirds; that is a larger saving than any query optimisation available afterwards, and it is available before the first panel is drawn.

**Takeaway:** every drawing here is a multiplication or a crossing point, and the design questions are which multiplications to authorise and where the crossing point sits.

### 6.1 Reading a dashboard as a reviewer

A reviewer opening an unfamiliar operational dashboard should read it in a fixed order, because the failure modes that matter are structural and the individual panels will look fine in any order. The order below takes about six minutes and catches most design defects before they are built into a habit.

Start with the header: owner, decision, cadence, cutoff, and freshness. An ownerless dashboard fails immediately, and so does one whose cutoff is unstated, because the reader cannot tell a drop from an incompleteness. Then read the top-left panel, which is where the root number lives, and ask three questions: what is the denominator, what is the comparator, and what would the reader do if this number halved.

Then walk the panels in the reading order the layout implies. If the layout has no reading order — a grid of equally weighted tiles — the designer has pushed the prioritisation onto the reader, who will resolve it by looking at whichever tile is largest or reddest. A deliberate layout puts the decision number first, the guardrail second, the decomposition third, and the explanation last.

Finally, check the freshness banner against a known event. Pick a panel whose data you can independently verify for the last hour and confirm that the panel agrees with the source, including its lag. This is the exercise that catches the most expensive class of defect: a dashboard that has been quietly showing a stale table for a week, because every panel was plausible and nobody compared it to anything.

The corresponding anti-pattern is a review that discusses colour, sorting, and chart type. Those choices matter far less than cutoff consistency, denominator presence, and owner assignment, and a review that spends its time on the first three has not examined the artifact's actual risks.

### 6.2 Why the same numbers look different on two surfaces

Two surfaces showing the same metric can disagree without either being wrong, and the reasons are finite enough to enumerate. This matters operationally because a disagreement during an incident consumes the incident's most expensive minutes.

The first cause is window alignment. A rolling 5-minute window and a calendar 5-minute bucket differ at every boundary, and a spike that straddles a boundary is split in one and whole in the other. The second is aggregation order: a mean of per-minute rates and a rate computed from summed counts differ whenever the denominators differ across minutes, and the difference can be large when traffic is bursty. The third is population filters, particularly the treatment of retries, bot traffic, and internal accounts, which different surfaces include inconsistently. The fourth is the cutoff lag, which shifts every recent point by the pipeline delay. The fifth is the definition of the denominator's population at the boundary — an attempt that starts before the window and completes inside it belongs to one window by event time and the other by completion time.

The practical discipline is a single reconciliation panel that shows both surfaces' series side by side with their stated windows, and a written rule that during an incident, one surface is authoritative and the other is diagnostic. The rule matters more than the reconciliation, because the reconciliation takes an hour and the decision does not have an hour.

### 6.3 The freshness banner as a first-class panel

Freshness is the only panel that tells the reader how to interpret every other panel, so it deserves a primary position rather than a footnote. A useful freshness banner states the last complete minute, the expected volume for that minute, the actual volume, and the pipeline's current lag — four numbers, all cheap.

The expected volume is the interesting one, because it converts "is data arriving" into "is the expected fraction arriving". A pipeline that has delivered 92 percent of the expected volume for the last ten minutes is not down; it is late, and its numbers are low by a factor the reader can apply. A pipeline that has delivered 92 percent for an hour is either broken or the traffic really did fall, and the way to distinguish those is to compare an independent volume source such as a load balancer's request count against the warehouse's row count.

The banner also has a second job: it makes the cutoff visible to people who did not build the pipeline. Most disputes about "the number is wrong" are disputes about which minute the number covers, and a banner that states the cutoff in the same time zone as the reader's clock removes the dispute without an investigation.

### 6.4 Where the eye goes first, and why it matters

Layout determines what a reader checks first, and under time pressure, first is often all. A dashboard that puts the root number at the top left and the guardrail at the top right encodes a decision procedure; a dashboard that puts twelve equally sized tiles in a grid encodes nothing and leaves the procedure to habit.

Two layout rules follow from the reading order. The first is that any panel whose movement would change the decision must be within the first screenful, not below a scroll. The second is that explanatory panels must be visually subordinate — smaller, or collapsed — because an explanation that is as prominent as the symptom invites the reader to skip the symptom.

A useful test is to cover the dashboard except for the top-left quadrant and ask an on-call engineer what they would do. If the answer is "I do not know", the layout has failed regardless of the data quality, and the fix is to move the decision's inputs into that quadrant rather than to add another panel.

**Takeaway:** layout is a decision procedure made visible; put the root, the guardrail, and the freshness in the first screenful, and make explanations subordinate to symptoms.

## 7. Comparison / Selection Table

Four selection problems recur: which artifact a decision needs, which statistic a trigger should use, which window to aggregate over, and which channel should carry the signal. Each has criteria that can be written down, and each is usually decided by imitation of whatever the team saw last.

### 7.1 Choosing the artifact for the decision

| Artifact | Refresh | Decision it supports | Cost driver | Fails when | Choose when |
|---|---|---|---|---|---|
| Real-time status board | seconds | roll back, fail over, shed load | evaluation count, series count | cadence exceeds the decision's latency | the mitigation is automated or the on-call can act in minutes |
| Daily operational dashboard | hours | reallocate capacity, open an investigation | scan of one day's partition | the day's data is still incomplete | the decision is a shift-level staffing or routing choice |
| Weekly business review | days | change a target, change an owner, change a policy | rebuild of the aggregate layer | the metric is dominated by one unusual day | the decision is a policy rather than an action |
| Cohort or funnel view | daily to weekly | fix a stage, change onboarding | join cost across events | cohorts are immature at the cutoff | the question is where a multi-step process loses entities |
| Alert with a runbook | seconds to minutes | execute a specific remediation | false pages, toil | the runbook does not exist | the condition has a known, bounded remedy |
| SLO burn panel | minutes | declare an incident, freeze changes | budget computation | the good-event definition drifts | the promise is to customers and the budget is finite |
| Ad-hoc exploration | none | form a hypothesis | analyst time | it is mistaken for a decision surface | the question is not yet specified |

The ordering in the table is roughly by decision latency, and the point of the ordering is that artifacts are not interchangeable upgrades. A team that replaces a weekly review with a real-time board has not improved its decisions; it has moved the same decision to a cadence at which it cannot be made, and it has added a pipeline whose failure mode is now a business risk.

The column that decides most cases is the last one. A real-time board is justified by an automated or fast mitigation, and if no such mitigation exists, the board's only function is to let people watch harm accumulate in higher resolution. The useful question is therefore "what changes in the next fifteen minutes because of this" and not "how fresh is this data".

### 7.2 Choosing the trigger statistic

| Statistic | Detects | Requires | False-alarm behaviour | Blind to | Prefer when |
|---|---|---|---|---|---|
| Single-point z against a seasonal baseline | sudden level shifts | stable baseline, robust scale | `Nα` pages per day, grows with cadence | slow drift, small sustained shifts | volume is high and the incident is abrupt |
| EWMA | small sustained shifts | choice of `λ` fixed in advance | ARL of several hundred evaluations | isolated single-point spikes | the harm accumulates slowly |
| Count threshold against Poisson | rare-event bursts | independent arrivals, stable rate | exact tail, well behaved at low counts | correlated bursts from one source | the metric is a count of failures |
| Rate threshold with a fixed denominator floor | proportional degradation | denominators above the floor | depends on denominator variability | changes in the denominator | traffic is stable and the rate is the customer-visible quantity |
| Multi-window burn rate | budget consumption | stable good-event definition | very low by construction | short severe spikes below the long window | there is an explicit error budget |
| Percentile with a sample bound | tail latency | hundreds of observations per window | high at small `n` | means and totals | latency is the customer-visible harm and volume is sufficient |
| Conjunction of two metrics | partial degradations of a specific shape | both metrics on the same cutoff | lower than either alone | single-metric degradations | the mitigation is expensive and should be reserved |
| Change-point detector | structural breaks | a model of the pre-change process | model-dependent | gradual drift | the series has regimes and nobody documented them |

Three of these are commonly chosen for the wrong reason. EWMA is chosen because it looks smooth, which is a presentation preference rather than a detection one. Percentiles are chosen because they are already on the dashboard, which is a data-availability argument rather than a detection argument. Conjunctions are chosen because they reduce pages, which is true and also reduces coverage, and the coverage loss is rarely quantified.

The selection rule that survives review is to pick the statistic from the harm's shape. If harm arrives as a step, a single-point or count rule is right. If harm accumulates, an EWMA or a burn rate is right. If harm is a tail phenomenon, a percentile is right provided the sample supports it. And if the mitigation is expensive, a conjunction is justified only with a measured statement of what it will miss.

### 7.3 Choosing the window

| Window | Latency cost | Noise suppression | Good for | Bad for | Check before adopting |
|---|---|---|---|---|---|
| 1 minute | minimal | almost none | step changes above 3x baseline | anything below a factor of three | count per window is large enough for a tail |
| 5 minutes | up to 5 min | moderate | most operational incidents | small proportional shifts | `λ >= 5` per window at baseline |
| 1 hour | up to 1 hour | strong | proportional shifts of 20 percent or more | fast-moving customer harm | the harm's breakeven latency exceeds an hour |
| 6 hours | up to 6 hours | very strong | slow budget consumption | nearly all incidents | the policy is deliberately slow (multi-window) |
| Rolling 30 days | one incident | n/a | budget accounting, policy review | any operational decision | the good-event definition is frozen |

The check in the last column is the one that is skipped. A 1-minute window on a service with 40 requests per minute has a baseline count of well under one failure, so its tail has no resolution and its rate has a standard error of roughly 8 percentage points. Choosing that window because it is the most responsive produces a trigger that fires on single events and cannot distinguish a doubling from noise, which is the worst of both properties.

### 7.4 Choosing the channel

| Channel | Interrupts | Suitable for | Unsuitable for | Cost per delivery |
|---|---|---|---|---|
| Pager with a runbook | yes, sleeps | Tier 0 harm with a bounded remedy | anything without a remedy | high, 15-25 minutes of context |
| Pager without a runbook | yes | nothing | everything | high and destructive |
| Team chat channel | somewhat | Tier 1 signals, deploy notices | conditions needing action in minutes | low, minutes |
| Ticket queue | no | Tier 2 tickets, threshold reviews | anything time-sensitive | low, but decays if never closed |
| Email digest | no | weekly reviews, policy changes | operational awareness | negligible, often unread |
| Dashboard banner | no | freshness, partial data, degraded mode | anything the reader may not open | negligible |

The rule that matters is that a channel's interrupt level must match the condition's actionability, and the mismatch in both directions is common. A Tier 0 condition sent to chat is an outage that nobody owns. A Tier 2 condition sent to a pager is a toil generator that also trains the team to dismiss pages, which is the second-order cost that never appears in a page-count review.

### 7.5 Symptom-based versus cause-based alerting

| Approach | Fires on | False pages | Detects novel failures | Runbook quality | Maintenance |
|---|---|---|---|---|---|
| Symptom-based (SLO, error rate, latency) | what customers experience | low if thresholds are calibrated | yes, including unknown causes | generic triage | low; the definition is stable |
| Cause-based (CPU, queue depth, disk) | a hypothesised mechanism | high, since the mechanism often fails to matter | no, only known mechanisms | specific and effective | high; every change to the system changes the signal |
| Hybrid (symptom pages, cause explains) | customer harm pages, causes annotate | low | yes | specific once triaged | moderate |

Symptom-based alerting is the default for a reason that is arithmetic rather than philosophical: the number of possible causes grows with the system's complexity, while the number of customer-visible symptoms is small and stable. A team that pages on causes needs a trigger per mechanism and will accumulate triggers faster than it can calibrate them, whereas a team paging on symptoms has a fixed budget of triggers and can spend its calibration effort on those.

The hybrid is the practical answer, and the division of labour is exact: symptoms decide *whether* to page, causes decide *whom* to page and *what to try first*. A cause-based signal that pages on its own is asserting a mechanism-to-harm link that should be measured rather than assumed, and the measurement is a lead-time study whose output is a number in minutes.

### 7.6 Threshold, target, and floor

| Concept | Answers | Set by | Changes when | Danger |
|---|---|---|---|---|
| Threshold | when to notify | derived from the noise model | the noise model changes | tuned after an incident to silence it |
| Target | what the team aims for | product and customer promise | the promise changes | treated as a floor |
| Floor or guardrail | what is unacceptable | pre-registered before the window | almost never, and only with a record | relaxed when it binds |
| Budget | how much failure is allowed | derived from the target | the target or window changes | spent silently |

The four are routinely collapsed into one number, and the collapse produces two opposite pathologies. When a floor is treated as a target, the team optimises to sit exactly on the boundary and every small variation breaches it. When a target is treated as a floor, the team treats a normal operating level as an incident and pages on it. Both are cured by writing the four numbers down separately, which takes about ten minutes and prevents a quarter of argument.

### 7.7 Dashboard archetypes and their required panels

| Archetype | Root panel | Guardrail | Decomposition | Explanation | Freshness |
|---|---|---|---|---|---|
| Service health | error rate against SLO | latency at the boundary | by endpoint and by dependency | recent deploys, dependency status | required, per minute |
| Business outcome | north-star outcome rate | customer-protection metric | by segment and channel | campaign calendar, pricing changes | required, daily |
| Funnel | stage conversion | downstream retention | by stage and by entry source | recent releases, instrumented event changes | required, with cohort maturity |
| Capacity | utilisation against provisioned limit | error rate under load | by pool and by region | traffic forecast, scheduled changes | required, per minute |
| Financial | margin per transaction | refund or dispute rate | by product and by channel | fee schedule, processor mix | required, daily with finance cutoff |

The archetype table is useful in review because it converts "is this dashboard complete" into a checklist with a small number of rows. A service-health board without a dependency breakdown cannot support triage; a funnel board without a maturity indicator will be read as a decline on every recent cohort. Neither omission is a stylistic choice.

**Takeaway:** pick the artifact from the decision's latency, the statistic from the harm's shape, the window from the count's resolution, and the channel from the remedy's existence.

## 8. Failure Modes & Edge Cases

Twelve mechanisms produce nearly all operational-surface failures. Each entry states the symptom, the competing causes that are commonly confused with it, the check that separates them, the bounded correction, and the check that confirms the correction.

### FAIL-01 — Three-sigma at high cadence

**Symptom.** The pager is busy in proportion to the number of series rather than to the number of incidents, and the busy periods are spread uniformly across the day.

**Competing causes.** A genuinely noisy service; a threshold set too tight; too many series evaluated; correlated series flagging together. The last two are distinguishable: correlated series flag within one evaluation cycle of each other, whereas a noisy service produces scattered flags with no temporal clustering.

**Discriminating check.** Compute `NαS` for the surface and compare it to the observed page count over a quiet week. At 288 evaluations per series per day, three sigma, and 30 series, the expected false pages are `288 × 0.0027 × 30 = 23.3` per day; an observed 20 on a week with no incidents confirms the threshold, not the service, is the cause.

**Correction.** Raise the level to four sigma, which at the same surface gives `288 × 6.3e-5 × 30 = 0.54` per day, or lengthen the evaluation window so the effective test count falls. Do not both at once, because the resulting sensitivity cannot be attributed to either change.

**Verification.** Over the next quiet week, the page count should be within a factor of two of the predicted rate, and the detection latency for a deliberately injected test excursion should still be inside the breakeven latency.

### FAIL-02 — The alert that cannot fire

**Symptom.** A component fails, customers notice, and no page is generated. The alert exists, appears configured, and has never paged.

**Competing causes.** A units mismatch (percent against a fraction); a filter that excludes the affected population; an evaluation schedule that does not run; a threshold on the wrong side of the comparison; a metric that stopped reporting so there are no evaluations at all. The units case and the filtering case are separated by injecting a synthetic value: a test observation that should breach will breach under a units error of the opposite direction but not under a filter error.

**Discriminating check.** Run the trigger's own query for the incident window and confirm it returns the value the incident produced, then push that value through the evaluation path with its timestamp and confirm a notification is emitted and routed to the right person.

**Correction.** Add a test-injection job per trigger, run on a schedule, that asserts a notification is delivered; treat a failure of the injection job as a Tier 1 alert in its own right.

**Verification.** Every trigger in the registry has a passing injection record inside the last review period, and the registry's count matches the configuration's count — a mismatch means a trigger exists in one and not the other.

### FAIL-03 — The missing heartbeat

**Symptom.** The pipeline dies at 02:00 and the first notification is a customer report at 06:00. Every dashboard panel shows a flat line and nobody reads a flat line as an outage.

**Competing causes.** A genuinely quiet period; a paused evaluation schedule; a dead extractor; a null-filled series. The quiet-period case is excluded by comparing against the seasonal baseline, which for a 02:00 window is a known non-zero volume; the other three are separated by looking at the extractor's own completion records.

**Discriminating check.** Count the rows the extractor wrote per minute against the expected count for that minute. Zero rows for two consecutive minutes with a non-zero expectation is a pipeline fault, and it is visible without any statistical model.

**Correction.** Add an always-true condition per surface, evaluated on the same path, with a maximum staleness derived from the decision's latency plus the pipeline's normal lag; page when staleness exceeds it.

**Verification.** Kill the extractor in a staging environment and confirm the heartbeat pages inside the stated maximum staleness, and that the page names the surface rather than the extractor — the recipient needs to know what is unmonitored, not which job died.

### FAIL-04 — The flat threshold over a seasonal series

**Symptom.** The alert fires every Monday morning and never fires on a Sunday collapse. The team raises the threshold, and then a real Monday incident is missed.

**Competing causes.** A genuine Monday peak in traffic; a batch job that runs at the week boundary and inflates counts; an instrumentation change that went live on a Monday; the seasonal pattern itself. The batch-job case is identified by the event time of the inflated events, which will cluster in a sub-minute interval; the instrumentation case by a client-version discontinuity.

**Discriminating check.** Compute the weekday index over eight weeks and compare each day's ratio to the overall mean. Indices of roughly 1.25 for one weekday and 0.86 for another, with stable values across weeks, identify seasonality; a single anomalous week identifies an event.

**Correction.** Divide the observation by the weekday index before standardising, recompute the scale from the indexed residuals, and exclude holiday weeks from the index estimation rather than letting them distort it.

**Verification.** Over four weeks the false-page count should fall to the predicted rate and the detection latency for a 20 percent drop on the low day should be finite — that is, the trigger should be able to fire at all on the day it previously could not.

### FAIL-05 — Persistence longer than the incident

**Symptom.** Short, severe incidents are never paged, while the pager is quiet. Post-incident reviews find that the condition was true for two evaluations and the requirement was three.

**Competing causes.** A threshold that was never breached; an incident shorter than the window; a persistence rule set from a page-rate target rather than from an incident-duration distribution; correlated evaluations making persistence look more effective than it is.

**Discriminating check.** Build the distribution of incident durations from the incident log, and replay each incident through the trigger at every possible alignment offset. An incident whose duration is below `k × cadence` is invisible by construction, so the check is arithmetic rather than statistical.

**Correction.** Set `k` from the shortest incident the owner must catch, and accept the resulting page rate; if the page rate is unacceptable, reduce the surface or raise the level instead of extending `k`.

**Verification.** The replay shows detection for at least 90 percent of historical incidents whose duration exceeds the owner's stated minimum, and the false-page rate remains inside the budget.

### FAIL-06 — Cardinality that silently multiplies

**Symptom.** The dashboard is slow, the storage bill grows faster than the business, and the number of alerts rises even though nobody added any.

**Competing causes.** A new dimension value (a new plan, a new region); a label whose values are unbounded (request id, user id, error message); a join that fans out rows; retention that grew without a policy. Unbounded labels are identified by the distinct-value count growing linearly with traffic; a new dimension value by a step change in the distinct count.

**Discriminating check.** Compute distinct series per metric per day and regress it against traffic volume. A slope near zero with steps indicates dimension changes; a positive slope indicates an unbounded label, which is the expensive case.

**Correction.** Cap the dimension set at design time, replace unbounded labels with bucketed ones, and enforce a series-count budget per dashboard that fails the build when exceeded.

**Verification.** The series count per metric is stable across a month with traffic growth, and the storage projection from the actual series count matches the invoice within a stated tolerance.

### FAIL-07 — Aggregating percentages

**Symptom.** The dashboard's total conversion rate disagrees with the sum of the segments, and both look defensible.

**Competing causes.** An unweighted mean of segment rates; a segment that appears in two rows; a denominator computed from a different population; rounding at the segment level. The overlap case is identified by an entity-level intersection; the population case by reconciling denominators to the source count.

**Discriminating check.** Recompute the total as a ratio of sums from the segment counts and compare it to the mean of the segment rates. A difference that scales with the spread of segment sizes confirms the unweighted mean.

**Correction.** Compute every aggregate as a ratio of sums, publish the denominator beside each rate, and remove any panel that shows a mean of rates.

**Verification.** The segment counts sum to the total count, and the total rate equals the ratio of the summed numerator to the summed denominator to the displayed precision.

### FAIL-08 — Two surfaces, two answers

**Symptom.** During an incident, two dashboards disagree by 15 percent and the responders spend the first twenty minutes reconciling them.

**Competing causes.** Different windows; different aggregation order; different population filters; different cutoff lags; different definitions of the denominator at the boundary. Window and cutoff causes are separated by comparing a closed historical minute, where both should agree exactly; persistence of the disagreement in closed minutes points to filters or definitions.

**Discriminating check.** Pick a closed minute an hour old, compute both surfaces' values from raw rows with their stated definitions, and diff the row sets. The diff's composition identifies the cause: a filter difference shows as whole entities present in one set only; an aggregation difference shows as identical entity sets with different numerators.

**Correction.** Declare one surface authoritative for the incident, publish the other's definitions beside it, and add a reconciliation panel that states both windows explicitly.

**Verification.** A quarterly reconciliation of the two surfaces on closed minutes shows agreement within rounding, with any residual documented as a definition difference.

### FAIL-09 — The window that reports partial data

**Symptom.** Every morning the first reading of the daily dashboard shows a drop, which recovers by mid-morning without any code change.

**Competing causes.** Genuine overnight decline; an extraction that runs before the day's late events arrive; a time-zone boundary that splits the day; a backfill that lands after the first read. Late arrivals are identified by the arrival-time histogram; the time-zone case by a shift in the intraday shape.

**Discriminating check.** Plot the 99th percentile of arrival lag against the extract schedule. If the 99th percentile lag exceeds the extract delay, the morning reading is systematically premature and the drop is an artifact.

**Correction.** Publish a freshness banner stating the expected and actual volume, and make the daily authority a fixed-lag snapshot whose delay exceeds the 99th percentile lag; keep the live view for monitoring only.

**Verification.** The daily value computed at the first read and at the fixed lag converges once the lag exceeds the arrival percentile, and the banner's completeness figure reaches 100 percent before the value is quoted in a review.

### FAIL-10 — The page nobody can act on

**Symptom.** A page fires, the recipient acknowledges it, and no action follows. The incident, if there is one, resolves on its own or is found by a customer.

**Competing causes.** No remediation exists; the recipient lacks permission; the runbook is stale; the condition is real but harmless. The harmless case is identified by the absence of any customer-visible symptom during the page's window; the permission case by the recipient's own account of what they tried.

**Discriminating check.** Audit the last fifty pages: for each, record whether the recipient took a different action than they would have taken without the page. Anything above roughly 30 percent non-actionable means the surface is generating toil, not control.

**Correction.** Either write the runbook and grant the permission, or demote the condition to a ticket, or delete it. All three are acceptable; keeping a pager with no remedy is not.

**Verification.** The next audit shows a precision above the agreed floor, and every remaining paging condition maps to a named remediation and a named role that can execute it.

### FAIL-11 — Correlated series page together

**Symptom.** One dependency fails and forty alerts fire within the same minute, burying the one that names the cause.

**Competing causes.** A shared upstream dependency; a shared metric pipeline; a shared threshold that is too tight for all of them; a genuine simultaneous multi-service failure. The shared-dependency case is identified by the timing: all forty fire in the same evaluation cycle and recover together. A genuine multi-service failure produces staggered onsets.

**Discriminating check.** Correlate the page timestamps across series over a month. A cluster of simultaneous pages recurring with the same membership is a shared dependency, and its membership list is the grouping key.

**Correction.** Group the series under a single page keyed by the dependency, list members inside the page, and let the dependency's own health be the paging condition.

**Verification.** A simulated dependency failure produces one page containing all affected members, and the individual members are still visible on the dashboard for diagnosis.

### FAIL-12 — The guardrail that is relaxed when it binds

**Symptom.** A guardrail breaches, and within a day the floor is revised to accommodate the result. The next quarter the same thing happens.

**Competing causes.** The floor was mis-calibrated from the start; the population changed so the floor no longer applies; the improvement genuinely justifies a trade; the breach is caused by a different change. Mis-calibration is identified by the historical distribution: a floor that the healthy population breaches more than the stated fraction of the time was never calibrated.

**Discriminating check.** Compare the floor to the guardrail's own historical distribution over the preceding year. A floor below the 10th percentile of normal operation is not a guardrail; a floor above the 90th is an obstruction.

**Correction.** Calibrate the floor at a stated quantile of the historical distribution, record the calibration window and date in the contract, and require a written decision with a named owner to change it — not a silent edit.

**Verification.** The contract's change log shows every revision with its date and reason, and no revision occurs inside a measurement window in which the guardrail is being evaluated.

### 8.1 A triage order for a busy pager

When the pager is busy, the order in which pages are examined determines how much of the incident is spent on the pager rather than on the system. The order below is sorted by information gained per minute of attention.

First, look for simultaneous pages. A cluster means one cause, and the cluster's membership names the dependency; investigating any member individually wastes the cluster's information. Second, check the freshness banner, because a pipeline fault presents as a page storm across every series at once and is cured by fixing one job. Third, check whether any page is on a Tier 0 symptom; if so, that one owns the response and the rest are diagnostics.

Fourth, check the error budget's remaining balance. A page during a month in which the budget is exhausted has a different urgency from the same page with 80 percent of the budget intact, and the difference is a decision about whether to freeze changes rather than about the specific condition. Fifth, and only then, look at individual cause-based signals, because those are the ones most likely to be noise and least likely to name the harm.

The discipline is not that individual signals are unimportant; it is that their information value is conditional on the first four checks. A team that starts at the fifth check will, under load, spend the incident on the least informative rung of the ladder.

### 8.2 Provenance and severity

No failure in this catalogue is claimed as an observation from a specific organisation; the mechanisms are derived from the arithmetic in Section 4, and the pattern names are descriptive rather than attributed. The table records the damage each causes and the cheapest permanent countermeasure.

| ID | Provenance | Typical damage | Detection cost | Permanent countermeasure |
|---|---|---|---|---|
| FAIL-01 | derived from `Nα` scaling | chronic toil, eroded trust in pages | one quiet-week count | derive the level from cadence and series count |
| FAIL-02 | derived from tail logic | silent coverage loss on a critical path | one injection job | scheduled test injection per trigger |
| FAIL-03 | derived from the null-evaluation case | outage detected by customers | one row-count query | heartbeat with a derived staleness bound |
| FAIL-04 | derived from the seasonal-index failure | weekly false page plus missed low-day detection | one weekday-index computation | index the baseline before standardising |
| FAIL-05 | derived from `k × cadence` latency | missed short incidents | replay over offsets | set `k` from the shortest incident required |
| FAIL-06 | derived from the cardinality product | cost growth and alert-volume growth | one distinct-series count | declared series budget enforced at build |
| FAIL-07 | derived from T-04 of File 01 | wrong totals, wrong segment conclusions | one ratio-of-sums recompute | ratio-of-sums everywhere |
| FAIL-08 | derived from window and filter differences | incident response delayed | one closed-minute diff | one authoritative surface during incidents |
| FAIL-09 | derived from the arrival-lag distribution | daily false declines | one arrival histogram | fixed-lag authoritative snapshot |
| FAIL-10 | organisational, not arithmetic | toil without control | a fifty-page audit | remediation and role per paging condition |
| FAIL-11 | derived from shared dependencies | cause buried under symptoms | one timestamp correlation | page once per dependency group |
| FAIL-12 | governance, not arithmetic | guardrails that never bind | one distribution comparison | calibrated floor with a change log |

The ordering by cost suggests a build order for countermeasures that differs from the failure numbering. FAIL-02 and FAIL-03 come first because they are coverage failures: everything else assumes the page arrives. FAIL-07 and FAIL-08 come next because they corrupt the numbers every other check depends on. FAIL-01 and FAIL-04 are calibration and can be fixed with arithmetic already in this sheet. FAIL-10 and FAIL-12 are organisational and cannot be fixed by tooling, which is why they are listed last and cause the most cumulative damage.

**Takeaway:** coverage failures outrank calibration failures, because a perfectly calibrated alert that cannot fire is worse than a noisy one that can.

## 9. Complexity & Cost Analysis

The cost of an operational surface has four components: storage, scan, evaluation, and human attention. They scale differently, and only two of them appear on an infrastructure invoice.

### 9.1 Algorithmic shape

Let `S` be the number of series, `dt` the sample interval in seconds, `K` the number of distinct dimension combinations behind a series count, `E` the evaluations per day, and `R_d` the raw rows per day at the finest grain.

| Operation | Inputs | Time | Space | Dominant term |
|---|---|---|---|---|
| Point storage for one day | `S`, `dt` | `O(S·86400/dt)` writes | same | `S/dt` |
| Roll-up to a coarser interval | `S`, `dt`, `dt'` | `O(S·86400/dt)` | `O(S·86400/dt')` | same as raw |
| One panel query over a day | `S_pan`, `dt` | `O(S_pan·86400/dt)` | `O(S_pan)` | `S_pan/dt` |
| Full-surface scan | `S`, `dt`, retention | `O(S·86400·D/dt)` | `O(S)` | retention `D` |
| Trigger evaluation per cycle | `S_eval` | `O(S_eval)` | `O(S_eval)` | series evaluated, not points |
| Anomaly model fit | `S`, history | `O(S·H)` | `O(S·H)` | history length |
| Cardinality recount | raw rows | `O(R_d)` | `O(K)` | raw volume |
| Backfill for a definition change | `R_d`, history | `O(R_d·D)` | `O(K)` | history × volume |

The third and fourth rows are the ones that generate cost surprises. A panel query is proportional to the series it draws and the samples per series, so a panel with 200 series at a 10-second interval reads 1.73 million points per day, while the same panel at a 5-minute interval reads 57,600. The full-surface scan multiplies that by retention, which is why retention policy is a cost decision made months before the invoice arrives.

Trigger evaluation is cheap by comparison — it touches one aggregate per series per cycle — and this asymmetry is the reason that adding an alert is easy and adding a panel is expensive. A team that reasons about alert count as the cost driver has identified the wrong term.

### 9.2 Back-of-envelope sizing

**Workload (ILLUSTRATIVE).** 4,000 series, sampled every 60 seconds, 16 bytes per stored point after compression (delta-of-delta with a varint, a common shape for numeric series), retained 90 days.

```text
series                 = 4,000
samples/series/day     = 86,400 / 60       = 1,440
points/day             = 4,000 * 1,440     = 5,760,000
bytes/day              = 5,760,000 * 16    = 92,160,000 B = 92.16 MB
bytes/90 days          = 92.16 MB * 90     = 8,294.4 MB = 8.29 GB
scan of 90 days at 400 MB/s                = 8,294.4 / 400 = 20.7 s
```

At a 10-second interval the same series set produces 34,560,000 points per day, 552.96 MB per day, and 49.77 GB over 90 days — a factor of six in every quantity from one parameter. That factor is the entire argument for choosing the sample interval from the decision rather than from the tool's default.

**Query cost.** A dashboard loaded 200 times per business day, each load scanning the full 90 days, costs `20.7 s × 200 = 4,140 s = 1.15` compute-hours per day. At an ILLUSTRATIVE 3.00 currency units per compute-hour that is 3.45 units per day. With a 5-minute roll-up for the long-range panels, the same loads read a twentieth of the points and the cost falls proportionally; the roll-up's own build is one pass per interval, which is negligible beside 200 scans.

Those rates are constructed for arithmetic and are not taken from any vendor's price list. The structural conclusion is what transfers: scan cost is proportional to loads times points per load, and the roll-up converts a multiplicative cost into an additive one.

**Evaluation cost.** A trigger evaluated every five minutes across 4,000 series performs `288 × 4,000 = 1,152,000` evaluations per day. Under the null with a three-sigma two-sided level, the expected flags are `1,152,000 × 0.0027 = 3,110` per day, which is the number that explains why a large surface cannot use a three-sigma level: the flags are not rare at that scale. At four sigma the same surface produces `1,152,000 × 6.3e-5 = 72.6` flags per day, and at the Bonferroni level for `S = 4,000` the expected count falls to single digits.

**Human cost.** At 5 pages per day, 30 percent precision, and 20 minutes per page, the toil is `5 × 0.7 × 20 = 70` minutes per day, or 490 minutes per week — 20.4 percent of a 2,400-minute working week. Halving the page count halves it; doubling precision halves it as well, which is why precision and volume are the two levers worth arguing about and the threshold level is usually the one people argue about.

**Cost per decision.** The useful unit economics is total surface cost divided by decisions changed. A surface costing 3.45 units per day and 490 minutes per week of toil, which supports two decisions a quarter, costs far more per decision than a surface costing ten times as much compute and supporting a decision a week. This is the same argument as in File 01, applied to a different artifact: the cost of a surface is dominated by the decisions it does not support.

### 9.3 Where the budget actually goes

The nightly build of an aggregate layer is proportional to raw volume `R_d`, which grows with the business. Interactive panel loads are proportional to traffic to the dashboard, which grows with headcount and with habit. The backfill after a definition change is proportional to `R_d × D`, which is incurred unpredictably and is the largest single line in any year in which the definition changed.

The standard mitigations are structural rather than clever. Retain raw events only as long as the finest definition requires, and retain aggregates longer. Version the definitions so a backfill is scoped to the changed series rather than the whole surface. And accept a documented break in a series when a backfill is infeasible, rather than silently splicing two definitions and calling it continuity.

A budget rule that follows from the sizing above: the surface should be small enough that a full scan fits inside the time a human will wait, on the order of seconds. At 400 MB/s that bounds the stored volume at a few gigabytes, which at 16 bytes per point is a few hundred million points — a few thousand series at a one-minute interval over a quarter. Beyond that, an aggregate layer or a partitioning scheme is mandatory, and the choice between them should be made before the growth rather than after.

### 9.4 The cost of the wrong signal

Two errors have prices, and both are computable. Paging on noise costs the toil computed above plus the erosion of the pager's credibility, which shows up later as a slow response to a real page and is therefore invisible in the page-cost accounting. Failing to page costs the harm during the undetected interval, which is `h × L` from Section 4.5.

In the worked example the undetected interval at a 2.5 percent incident costs `187.5 × 30 = 5,625` units for a thirty-minute delay against a page cost of 20, a ratio of 281 to 1. That ratio is why the correct response to a low-precision pager on a high-harm path is not to raise the threshold but to add automated detection and mitigation: the asymmetry is too large to be managed by tuning the human notification.

A related and frequently missed cost is the maintenance of the runbook. Every paging condition needs a document that stays true, and a stale runbook converts a page into an investigation, which is the difference between a five-minute mitigation and a forty-minute one. The maintenance cost is small per condition and grows with the number of conditions, which is another reason to keep the paging set small deliberately.

### 9.5 Retention, aggregation, and the shape of the bill

Three knobs determine the storage bill, and they interact. Retention sets the multiplier; the sample interval sets the divisor; aggregation sets the compression achieved by the roll-up. Doubling retention doubles the bill, halving the sample interval doubles it, and rolling up to a coarser interval for long-range panels cuts the read cost without changing the stored volume.

The interaction that surprises teams is that a long retention on a fine interval is rarely needed: the panels that read a year of history almost always plot a daily aggregate, and the panels that read fine data almost always read the last day. Splitting the two — fine data for a fortnight, coarse data for two years — costs a fraction of a uniform policy and loses nothing that any panel actually reads.

**Takeaway:** storage scales with series count times samples per day times retention, evaluation scales with series count alone, and human cost scales with page count times imprecision — so the cheap levers are the series budget and the page budget.

## 10. Key Theorems / Results Reference

| Result | Statement | Hypotheses | Use | Does not imply |
|---|---|---|---|---|
| R-01 Normal tail | `α = 2(1-Φ(L))` for a two-sided level `L` | symmetric, stationary deviations | per-evaluation false-alarm rate | that the deviations are normal in the tail |
| R-02 Poisson tail | `P(X>=k) = 1 - Σ_{i<k} e^{-λ}λ^i/i!` | independent arrivals, stable `λ` | count thresholds at low volume | that clustered events are Poisson |
| R-03 Run length | `ARL0 = 1/α` for independent evaluations | independence | page rate from a level | ARL under correlation or smoothing |
| R-04 EWMA variance | `Var(z) = σ²λ/(2-λ)` in control | constant variance, fixed `λ` | control limit width | the ARL, which requires simulation |
| R-05 Bonferroni | `P(any of S) <= Sα` | any dependence structure | family-wise threshold | that `Sα` is tight |
| R-06 Quantile bound | `P(max > q-quantile) = 1 - q^n` | independent draws | minimum sample for a percentile | a point estimate of the quantile |
| R-07 Burn identity | budget consumed in `w` is `x·w/W` | stable good-event definition | multi-window policy design | that the budget is the right policy |
| R-08 Breakeven latency | `L* = C_page/h` | constant harm rate, actionable page | whether to page at all | anything about accelerating harm |
| R-09 Toil linearity | `toil = P(1-P_p)C_page` | measured precision | page-budget arguments | the second-order trust cost |
| R-10 Cardinality product | series `= Π` dimension cardinalities | dimensions independent by construction | storage and test-count projection | that all combinations occur |

**R-01 and R-03 together.** The run-length result is the bridge between a statistical level and an operational rate, and it holds only for independent evaluations. A series with a weekly cycle violates independence in the most consequential way: evaluations an hour apart are highly correlated, so the effective number of independent tests per day is closer to a handful than to 288. The practical consequence is that a nominal `ARL0` of 370 can behave like an `ARL0` of 30 on a seasonal series, which is precisely the case that the seasonal index in 4.3 repairs.

**R-02 and the clustering boundary.** The Poisson threshold is exact and cheap, and it fails in one specific and common way: when the events being counted come from a small number of sources, the count's variance exceeds its mean, and the true tail is heavier than the Poisson's. A count that doubles because one host doubled is not evidence of a fleet-wide fault, and the discriminating statistic is the number of distinct sources contributing to the count. When the source count is small, alert on the proportion of sources affected rather than on the event count.

**R-04 and the unstated ARL.** The EWMA variance formula gives the control limit but not the run length, and the run length is what determines the page rate. Quoting a limit without a run length is quoting a threshold without a false-alarm rate, which is the defect this sheet exists to prevent. The run length must be simulated for the chosen `λ` and `L`, and the simulation is a few lines of code with a fixed seed so that two reviewers can reproduce the number.

**R-05 and the grouping correction.** Bonferroni is valid under any dependence, which is why it is safe, and it is loose under positive dependence, which is why it is conservative. Its practical role in this sheet is to make the family explicit: the threshold changes little (3.23 to 3.76 between 40 and 300 series), while the act of declaring the family changes the design, because it forces the designer to say which series count as one test.

**R-06 and the percentile illusion.** The quantile bound is the cleanest available argument that a chart can display a statistic the data cannot support. At `n = 90` and `q = 0.99`, the sample maximum exceeds the true 99th percentile with probability `1 - 0.99^90 = 0.596`; the plotted line is therefore the maximum with a coin-flip relationship to the quantity it is labelled as. The bound `n >= ln(α)/ln(q)` gives 299 for `α = 0.05`, and any percentile panel below that count should be labelled as a maximum rather than as a percentile.

**R-07 and the choice of 14.4.** The burn identity makes the familiar constants derivable rather than memorised: one hour is `1/720` of a 30-day window, so a policy that allows 2 percent of the budget to be consumed in one hour corresponds to a burn rate of `0.02 × 720 = 14.4`. The same derivation at 5 percent in six hours gives `0.05 × 120 = 6`. A team that derives its constants can defend them; a team that copies them cannot say what policy they encode.

**R-08 and its boundary at automation.** The breakeven result is a comparison between two costs, and it has an interpretation that is easy to miss: when `L*` falls below the fastest achievable notification latency, the correct conclusion is not a faster pager but a different mitigation. The formula's assumption of constant harm is what makes it a lower bound on the value of speed; accelerating harm makes the case for automation stronger, never weaker.

**R-09 and the missing term.** Toil as computed is linear in page count and in imprecision, and it omits the trust cost, which is the reason the linear model understates the damage of a noisy pager. The omission is deliberate and should be stated whenever the linear model is used to justify a change: the model supports the direction of the argument, not the magnitude.

**R-10 and the projection.** The cardinality product is exact when every combination occurs and an overestimate when it does not. Its practical use is as a ceiling: if the product of the dimension cardinalities exceeds the series budget, the dashboard cannot be built as specified regardless of whether the combinations are populated today, because the first new customer in a sparsely populated region will populate them.

### 10.1 What these results do not establish

The results above establish thresholds, rates, latencies, and costs. They do not establish five things that reviews routinely assume.

They do not establish that the metric is the right one. A perfectly calibrated alert on a metric that no customer experiences is a well-engineered irrelevance, and no amount of tail arithmetic repairs a bad choice of quantity.

They do not establish that the owner can act. The breakeven latency assumes an actionable page; if the recipient cannot mitigate, `h` continues and the page's only effect is notification. This is the assumption most often violated, and it is organisational rather than statistical.

They do not establish that the harm estimate is correct. The economics of Section 4.5 are only as good as the cost per failed event, which is usually the least-measured input in the whole design and the one with the largest effect on the conclusion.

They do not establish that the baseline is the counterfactual. Every threshold compares an observation to a baseline, and the baseline is a description of the past rather than an estimate of what would have happened. When the comparison is used to justify an intervention's effect, a design is required; the surface can only report.

They do not establish that the surface will be read. A dashboard's value depends on a habit, and habits are formed by the first few weeks of use. A surface that is accurate, well-calibrated, and unused has a return of zero, which is why the owner and the cadence belong in the design rather than in the deployment checklist.

**Takeaway:** these ten results make the arithmetic checkable, and every one of them is silent on whether the measured quantity, the owner, the harm estimate, or the habit is right.

## 11. Interview / Exam Questions

Fifteen questions mixing recall, calculation, and diagnosis. Each has its own answer, and each answer states the step that would withdraw it.

**Q1. A service has 30 series, each evaluated every 5 minutes at a three-sigma two-sided level. How many false pages per day do you expect, and what would you change?**
*Answer:* `288 × 0.0027 × 30 = 23.3` per day, which is more than the team can triage. The cheapest correct change is to derive the level from the family budget: at `α_family = 0.05` over 30 series, the per-series level is `1.667e-3`, giving `L = 3.23`, and the expected count falls to `288 × 0.00167 × 30 = 14.4` per day — still too many. The real fix is fewer tests: lengthen the window to 15 minutes, which cuts `N` to 96 and, at the same level, gives 4.8 per day, or group the series by dependency so a shared fault is one test. *Withdrawal:* if the surface exists to detect a 10 percent shift rather than a step, none of these levels is sensitive enough and the statistic must change to an EWMA or a burn rate.

**Q2. Compute the breakeven latency for a page costing 20 engineer-minutes against a harm rate of 4.2 cost units per minute.**
*Answer:* `L* = 20/4.2 = 4.76` minutes, or 286 seconds. A trigger whose detection-plus-response latency exceeds 4.76 minutes costs more than it saves, so either the window must be shorter or the mitigation must be automated. *Withdrawal:* if the harm accelerates — a retry storm that doubles every minute — the constant-rate assumption fails and the true breakeven is shorter, which strengthens rather than weakens the conclusion.

**Q3. Why is three sigma acceptable at one evaluation per day and unacceptable at one per 30 seconds?**
*Answer:* because the page rate is `Nα`, not `α`. Three sigma gives `α = 0.0027`; at `N = 1` the expected false pages are `0.0027` per day, and at `N = 2,880` they are `7.78` per day per series. The level has not changed; the number of opportunities has, and the opportunities are what the pager experiences. *Withdrawal:* if evaluations are strongly autocorrelated, the effective `N` is much smaller than the nominal one, so the 30-second case is less bad than the arithmetic suggests — but it is still worse than the daily case by more than an order of magnitude.

**Q4. A dashboard's 99th-percentile latency panel shows a jump from 900 ms to 1,400 ms. The service handles 90 requests per 5-minute window. What do you say?**
*Answer:* the panel cannot support a 99th percentile. At `n = 90`, `P(max > true 99th percentile) = 1 - 0.99^90 = 0.596`, so the plotted value is the sample maximum, and its movement between windows is dominated by which single request was slowest. The minimum sample for a 95 percent-confidence lower bound at the 99th percentile is `ln(0.05)/ln(0.99) = 299` observations, so the panel needs a window of at least 17 five-minute periods at this volume, i.e. roughly 85 minutes. *Withdrawal:* if the metric were the mean or a count above a latency threshold, 90 observations would be usable, and the correct change would be to switch statistics rather than to widen the window.

**Q5. An SLO of 99.5 percent over 30 days — how much budget is there in minutes and in failed attempts at 1,000,000 attempts per day, and what does a 14.4 burn rate mean?**
*Answer:* `0.005 × 43,200 = 216` minutes, and `0.005 × 30,000,000 = 150,000` failed attempts. A burn rate of 14.4 means the budget is consumed 14.4 times faster than allowed, so the full budget would be gone in `720/14.4 = 50` hours, and one hour at that rate consumes `14.4/720 = 2` percent of it. *Withdrawal:* if the good-event definition excludes a class of failures that occurs during the incident, the consumed budget is understated and every one of these numbers is wrong in the same direction.

**Q6. Your pager fires 25 times a week at an estimated 40 percent precision, with 20 minutes per page. Compute the weekly toil and name the two levers.**
*Answer:* `25 × 0.6 × 20 = 300` minutes, or 12.5 percent of a 2,400-minute week. The levers are page volume and precision, and the formula is linear in both, so halving either halves the toil. Threshold level is a third lever but it acts through both, which makes its effect harder to predict and its review harder to defend. *Withdrawal:* if precision is estimated by the alert's owner rather than audited, the number is optimistic, and the audit is the first action rather than the calculation.

**Q7. A count metric's baseline is 0.7 events per 5-minute window. Can you alert on it per window?**
*Answer:* no. At `λ = 0.7`, `P(X >= 2) = 0.156` and `P(X >= 3) = 0.034`, so a threshold of two fires falsely in 15.6 percent of windows — about 45 times a day — and a threshold of three still fires about 10 times a day while detecting a doubling only weakly. The correct design accumulates into a window whose `λ` is large enough for resolution, and then checks the added latency against the breakeven latency; at `λ = 8.68` a window of roughly an hour is needed for a usable tail. *Withdrawal:* if the harm of a single event is large enough that a 15 percent false-page rate is acceptable, per-window alerting is defensible, and the number to state is the accepted page rate rather than the tail.

**Q8. Two dashboards disagree by 15 percent during an incident. What is your first check and why?**
*Answer:* a closed minute at least an hour old, computed from raw rows under both surfaces' stated definitions. Open minutes differ by cutoff and by partial arrival, so they cannot distinguish a definitional difference from a timing one; a closed minute removes the timing explanation and leaves the definitional one. If the closed minute agrees, the disagreement is a window or cutoff artifact and no definition changed. *Withdrawal:* if both surfaces read from different source systems with different event identity, the closed-minute comparison still differs legitimately and the reconciliation must start at identity rather than at aggregation.

**Q9. What does a heartbeat alert do that a threshold alert cannot?**
*Answer:* it detects the absence of evaluation. A threshold alert is a predicate over values, so when there are no values it is silent, which is indistinguishable from healthy. A heartbeat is a predicate over the arrival of evaluations, with a maximum staleness derived from the decision's latency plus the pipeline's normal lag; when it fails, the surface is unmonitored rather than healthy. *Withdrawal:* if the heartbeat is evaluated by the same job that produces the data, its own failure is also silent, which is why the heartbeat must run on a separate path with its own liveness check.

**Q10. Explain why a flat threshold on a series with a weekday index of 1.30 on Monday and 0.90 on Sunday is wrong in both directions.**
*Answer:* with a mean index of 1.043, a flat threshold at 1.2 times the mean is 1.251, which the ordinary Monday level of 1.30 exceeds by 3.9 percent — a false page every week — while Sunday at 0.90 sits at 0.719 of the threshold, requiring a 39 percent rise before it fires, so a 39 percent fall goes unnoticed. The fix is to divide by the weekday index before standardising and to recompute the scale from indexed residuals. *Withdrawal:* if the weekday pattern is unstable across weeks, indexing is not defensible and the correct change is a longer window or a different metric, not a better index.

**Q11. When is a cause-based alert better than a symptom-based one?**
*Answer:* when the cause has a measured lead time shorter than the symptom's breakeven latency and a specific, bounded remedy. A disk-filling alert is the canonical case: the lead time is measurable in hours, and the remedy is additive and known. Absent both properties, the cause-based alert is a hypothesis about mechanism, and paging on it asserts a mechanism-to-harm link that has not been measured. *Withdrawal:* if the mechanism's lead time is short and its harm is certain, the cause-based alert dominates because it fires before the symptom exists, which is the only case in which it is strictly better.

**Q12. A dependency failure produced 40 simultaneous pages. What design change prevents that, and what does it cost?**
*Answer:* page once per dependency group, listing members inside the page, and let the dependency's own health be the paging condition. The cost is a coverage gap: a single member failing independently of its group no longer pages, so the group must be paired with a per-member condition at a level high enough not to fire during group faults — typically a much higher level — or with a coverage review that accepts the gap explicitly. *Withdrawal:* if the forty series do not share a dependency, the simultaneity indicates a shared metric pipeline instead, and the grouping key must be the pipeline rather than the service.

**Q13. What is the difference between a threshold, a target, and a floor, and what goes wrong when they are collapsed?**
*Answer:* a threshold is when to notify and is derived from the noise model; a target is what the team aims for and is set by the customer promise; a floor is what is unacceptable and is pre-registered. Collapsing a floor into a target makes the team optimise to sit on the boundary, so normal variation breaches it; collapsing a target into a floor makes a normal operating level an incident, so the pager fires on ordinary days. Both are cured by writing all three down with their dates. *Withdrawal:* if the guardrail has no historical distribution to calibrate against, it is not yet a floor, and the first action is measurement rather than threshold-setting.

**Q14. A team wants to move from a daily dashboard to real-time. What must be true for that to improve decisions?**
*Answer:* the mitigation must be executable at the new cadence, the breakeven latency must be shorter than the new detection latency, and the volume must be large enough that the shorter window's tail has resolution. If the mitigation takes an hour, real-time detection buys an hour of watching; if the breakeven latency is six seconds, real-time detection at 30 seconds still misses the economics and automation is the answer. *Withdrawal:* if the decision is about a policy rather than an action, real-time is strictly worse because it invites reaction to noise, and the correct change is to keep the cadence and improve the metric's definition.

**Q15. How do you decide the sample interval for a new metric surface?**
*Answer:* from the shortest window any trigger will evaluate, with at least five samples per window so that the window's statistic has resolution, and no finer than the decision requires. A 5-minute trigger needs a 60-second interval; a 1-hour trigger does not need 10 seconds, and at 4,000 series the difference between 10 and 60 seconds is 34.56 million versus 5.76 million points per day, a factor of six in storage and scan. *Withdrawal:* if the surface must support retrospective analysis of a sub-minute phenomenon, the finest interval is set by that analysis rather than by any trigger, and the cost must be justified by the analysis's decision rather than by the alerting.

**Takeaway:** strong answers compute the tail or the economics, state the assumption that makes the conclusion valid, and name the observation that would withdraw it.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Decision first | Name the owner, the action, and the deadline before choosing a panel; an ownerless surface is a monitoring artifact, not a control. |
| Ladder | Only Tier 0 (customer harm) and Tier 1 (imminent harm) may page; Tier 2 opens a ticket; Tier 3 and 4 never notify. |
| Page rate | `E[pages/day] = (86400/cadence) × α × S`; three sigma at 5-minute cadence across 30 series is 23 pages/day under the null. |
| Level from family | For `S` series use `α/S` per series: 3.23 sigma at 40 series, 3.76 at 300 — the correction is cheap, so apply it by default. |
| Window | At least five baseline events per window; counts below `λ ≈ 5` have no usable tail, and percentiles need `ln(α)/ln(q)` samples (299 for the 99th at 95 percent confidence). |
| Seasonality | Divide by the weekday index before standardising; a flat threshold on a 1.30 Monday fires weekly and misses a 39 percent Sunday fall. |
| Budget | `B = (1-S) × N`: 99.5 percent over 30 days is 216 minutes or 150,000 of 30 million attempts; one hour at burn `x` consumes `x/720` of it. |
| Breakeven latency | `L* = C_page / h`; at 187.5 units/min of harm a 20-unit page pays only inside 6.4 seconds — below that, automate rather than page. |
| Toil | `P × (1-precision) × 20 min`: 25 pages/week at 40 percent precision is 300 minutes, 12.5 percent of a working week. |
| Cost | Points/day `= S × 86400/dt`; 4,000 series at 60 s is 5.76 M points/day, 92.16 MB/day, 8.29 GB over 90 days at 16 bytes. |
| Coverage | Every surface needs a heartbeat and every paging condition needs a test injection; a trigger that has never fired in a test is a hypothesis. |
| Evidence | Arithmetic is DERIVED and VERIFIED on ILLUSTRATIVE inputs; no vendor limit, benchmark, or company incident is claimed. |
| Go deeper | File 01 for the metric contracts these thresholds consume, File 02 for funnel and cohort populations, File 06 for the issue catalogue. |

