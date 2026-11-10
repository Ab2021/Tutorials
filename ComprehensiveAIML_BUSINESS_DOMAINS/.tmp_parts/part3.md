## 8. Failure Modes & Edge Cases

Each entry below follows the same anatomy: what you observe, the competing causes that produce that symptom, the check that discriminates between them, the smallest fix that resolves it, and the verification that confirms the fix worked. The competing causes are the important part — most experiment disasters are diagnosed by jumping to the first plausible explanation.

### 8.1 Failure: the arms are the wrong size

**Symptom.** Assigned counts differ from the design: 13,234 versus 12,766 rather than 13,300 and 13,300. Everyone notices the number but nobody agrees whether it matters.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Pure chance | Binomial fluctuation around the design | $\chi^2$ small, stable across days |
| Splitter bug | Hash modulo implemented with an off-by-one or a biased salt | Persistent imbalance in the same direction, independent of population |
| Differential eligibility | Assignment happens before a filter that fails more often in one arm | Imbalance concentrated in one entry surface or device class |
| Bot / crawler traffic | Non-human sessions routed deterministically | Imbalance concentrated in sessions with no downstream events |
| Delayed assignment logging | Log records arrive late for one arm | Imbalance shrinks over time as logs settle |

**Discriminating check.** Compute $\chi^2$ on *assignment* counts, not outcome counts, and recompute it on each of three consecutive days. Chance produces a wandering statistic that stays in the bulk; a splitter bug produces a persistent statistic with the same sign every day; differential eligibility produces imbalance that appears only after stratifying by surface; logging delay produces a statistic that converges toward zero as the table settles. The persistence test — same sign, three days, growing with $\sqrt n$ — is what separates bug from noise, and it costs one query.

**Bounded fix.** Void the readout, do not attempt to reweight. Rewriting the analysis to correct proportions is a well-known way to manufacture a result; the assignment is either the declared mechanism or it is not. Reproduce the hash offline on the assignment key to find the failing predicate, fix it, and rerun. If the traffic cannot be repeated, treat the experiment as a pilot and label it non-randomized in the record.

**Verification.** Rerun for one day with the fixed splitter and confirm $\chi^2<3.84$ at the planned size, then confirm that the same key replayed through the offline reproduction produces identical arm assignments for 100% of a 10,000-row sample. The second check is the one that catches "we fixed it and it looks better," which is not evidence.

**Why this is Section 8.1.** Every other failure in this section is diagnosed from outcome data, which may be confounded. This one is diagnosed from assignment metadata, which is not. It should run first, automatically, and block the readout.

### 8.2 Failure: the test was peeked and the boundary moved

**Symptom.** The reported effect is significant, but the analysis was run five times during the experiment and only the final report survives. The dashboard showed $p=0.03$ on day 4 and $p=0.07$ on day 9.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Genuine effect, lucky early estimate | True effect exists; sampling path crossed early | Effect persists and narrows; final interval excludes zero comfortably |
| Peeking-selected fluke | Multiple looks, reported the crossing | Reported p is near the boundary; effect shrinks toward zero with more data |
| Segment mining | Subgroup discovered post hoc | Effect concentrated in an unpredicted slice |
| Novelty decay | Real early effect that fades | Declining treatment-control gap over time (needs the 5.7 interaction test) |

**Discriminating check.** Plot the running estimate with its interval at each look against the information fraction. Under a genuine effect the estimate converges and the interval narrows monotonically. Under peeking, the *reported* estimate is the maximum of a noisy path, so it systematically shrinks when extended. The quantitative version: compute the probability that a null path would have crossed the observed boundary at least once in the number of looks actually performed — for five looks at 1.96 that probability is about 0.143, from the Section 4.3.7 simulation. If the reported p-value is 0.03 and the design allowed five looks, the honest statement is that the family-wise evidence is roughly $0.03\times$ the ratio of naive to adjusted critical values, not 0.03.

**Bounded fix.** Freeze the boundary now and re-read only at the pre-declared final size, or apply the correct group-sequential boundary retroactively — which requires that the look schedule was recorded, and is not possible if looks were triggered by the data. If neither is available, downgrade the claim: report the point estimate with an interval and explicitly state that the stopping rule was data-dependent, so the interval is a lower bound on uncertainty.

**Verification.** Show the audit log of analysis runs, with timestamps, alongside the final estimate. A readout that cannot produce that log is not reproducible; the log is also the artefact that makes the next experiment's pre-registration credible, because the team knows the schedule is recorded.

### 8.3 Failure: nine metrics were tested and the winner was chosen afterwards

**Symptom.** The primary metric is flat ($p=0.31$), but a secondary metric moved ($p=0.02$), and the deck leads with the secondary.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real secondary effect | Treatment genuinely affects it | Secondary effect replicates in the follow-up test |
| Multiplicity | One of nine tests crosses by chance | Effect size small; no mechanism story; disappears on replication |
| Metric correlation | Secondary is a component of the primary, so it moves with noise | Both move in the same direction, and the primary's CI barely includes the secondary's effect |
| Composition shift | Treatment changes who is in the population | Denominator changes; per-unit metric moves but total does not |

**Discriminating check.** Count the tests actually run and compute the expected number of false positives: nine independent tests at 0.05 give $1-0.95^9=0.370$, so a single 0.02 is unremarkable. Apply BH to the declared family — with $m=9$ and the smallest p at 0.02, the threshold at $k=1$ is $0.05\times1/9=0.0056$, so 0.02 fails. Then check whether the metric was declared pre-exposure; if it was not, it belongs in the exploratory section regardless of its p-value.

**Bounded fix.** Reorder the readout so the primary metric and its interval come first, and secondary metrics are reported as exploratory with explicit "not corrected, not confirmatory" labels. If the secondary is strategically important, register it as the primary of a follow-up test rather than promoting it in this one. Do not apply a correction to nine metrics and then report the smallest as if it were the only one — that is the same error with extra arithmetic.

**Verification.** The follow-up test, pre-registered with the secondary as primary, either reproduces the effect or does not. Reproducing a 0.02 finding at 80% power requires the effect to be at least as large as the MDE, which is exactly why promoting a secondary usually leads to a failed replication rather than a confirmation.

### 8.4 Failure: the guardrail was never powered, so "no harm" means "no evidence"

**Symptom.** The readout says "crash rate +37%, not significant, no guardrail concerns." The team ships. Three weeks later the crash rate is up and support volume is elevated.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Genuinely no harm | True effect zero | Bound is tight, not just non-significant |
| Underpowered guardrail | Rare event, small n, wide interval | Interval spans "no change" to "large increase" |
| Denominator shift | Crash rate per session fell because sessions fell | Absolute crash count rose while the rate fell |
| Reporting lag | Crashes recorded by a client SDK that batches | Under-count in the arm where the client crashes before flushing |

**Discriminating check.** Report the *bound*, not the p-value: a +37% point estimate with a +76% upper bound is not "no harm," it is "harm up to 76% is compatible with this data." Then recompute the guardrail on absolute counts as well as rates, and reconcile the crash count against an independent server-side error log. The reporting-lag cause is the nastiest, because it produces a *downward* bias in the arm that crashes more — the SDK dies before sending the crash. A server-side join catches it; the client-side metric alone cannot.

**Bounded fix.** For this readout, apply the bound-based tolerance rule from Section 7.4, which converts an imprecise estimate into a decision. For future tests, either size the guardrail for its own MDE (Section 5.6 shows 81,224 per arm for a 25% relative increase at $p\approx0.003$) or state in the pre-registration that the guardrail is directional and cannot gate shipment — which at least stops the false reassurance.

**Verification.** After the fix is deployed, re-read the guardrail for seven days at full traffic. Full traffic often gives more guardrail events per day than the capped experiment did in two weeks, which is the strongest argument for the "ship small, watch the guardrail" pattern.

### 8.5 Failure: the metric definition moved mid-flight

**Symptom.** The dashboard shows a 6% lift, but the underlying query was edited on day 6 to exclude a fraudulent payment provider. Nobody recorded the change.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real lift | Treatment changes behaviour | Lift present in both halves of the window |
| Definition change | Query, filter, or join altered | Step change at a specific timestamp, affecting both arms equally |
| Upstream schema change | Source table gained a column or changed a type | Row counts shift; nulls appear in a previously complete field |
| Duplicate events | Retry logic doubled confirmations after a client release | Numerator inflated in one arm, often the one with the new flow |

**Discriminating check.** Recompute the metric from immutable raw events with the pre-registered definition, then diff against the dashboard series. A step change at a timestamp that aligns with a deployment, present in both arms, is a definition or schema change; a step change in one arm only is a product or instrumentation effect. Row-count and null-count series, plotted by day and arm, discriminate the two in a single chart and should be a standing artefact rather than a forensic exercise.

**Bounded fix.** Recompute the entire readout on the frozen definition, using the raw event table and the commit hash from the pre-registration. Report the dashboard series only as a diagnostic. If the frozen definition cannot be reproduced because raw events were not retained, the readout is not recoverable and must be recorded as such.

**Verification.** The recomputed series and the frozen script's output must agree to the last reported digit on a re-run, and the script's hash must match the value recorded at pre-registration. Both are mechanical checks, and both are routinely skipped.

### 8.6 Failure: interference between units

**Symptom.** A delivery-fee test shows a large win in the treatment cities, but the control cities' delivery times also worsened during the test period. The launch produces no benefit.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real effect with contamination | Treatment pulled couriers from control cities | Control metrics move in the same direction as treatment |
| Seasonality | Weather or holiday shifted both | Effect is uniform across geography; compare to prior-year same weeks |
| Capacity constraint | The market cannot absorb the change at scale | Effect shrinks as the treated share grows |
| Measurement spillover | Shared accounts or devices span regions | Overlap in the identity graph across arms |

**Discriminating check.** Compare the control arm's metric *during the test* to its own pre-period. Under no interference the control should be flat relative to trend; under contamination it moves in the direction of treatment. A useful quantitative version: estimate the effect as a function of treated share $s$ and check whether it declines with $s$ — a negative slope is the signature of a market-level constraint rather than a unit-level effect.

**Bounded fix.** Redesign as a switchback: alternate treatment by time period in the same geography, so supply is shared by both arms within each period. Then estimate with period fixed effects. The cost is precision — the design effect from [E-06] applies with $m$ = units per period — and the benefit is that the estimand becomes interpretable.

**Verification.** In a switchback, the period-level randomization should produce balanced pre-period covariates across arms, and the within-period contrast should be stable across periods. If the effect flips sign between periods, the design is picking up period-level shocks and the estimate is not yet usable.

### 8.7 Failure: attrition that looks like an effect

**Symptom.** Purchase rate per tracked session is up 14%, but total purchases per assigned session are flat. The new module fails to load for a subset of sessions, and those sessions generate no tracked events.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real per-session effect | Behaviour genuinely changed | Denominator stable; absolute purchases rise proportionally |
| Differential attrition | Failed loads drop out of the tracked population | Tracked-session count lower in treatment; absolute total flat |
| Bot filtering | Filter applied after assignment, differentially | Imbalance in the filter rate by arm |
| Identity stitching change | New flow changes when the session ID is minted | Session counts differ; orders per identity unchanged |

**Discriminating check.** Always report the *assigned* denominator alongside the tracked one. If assigned counts are balanced and tracked counts are not, the difference is attrition, and the per-tracked-session metric is conditioned on a post-treatment variable. A second check reconciles total orders against a payment-provider source: if the provider total is flat while the per-session rate is up, the rate is an artefact.

**Bounded fix.** Use ITT on the assigned population as the headline, and treat the tracked-session analysis as a diagnostic of the load-failure rate. Then fix the load failure, because a 0.5% differential failure rate is a product defect regardless of what it does to the metric.

**Verification.** After the fix, the tracked and assigned denominators should agree within the logging loss rate, and the per-assigned-session metric should reconcile with the provider total to within the refund and chargeback rates.

### 8.8 Failure: the sample size was computed for the wrong variance

**Symptom.** The experiment was sized for 80% power, ran to the planned size, and came back at $p=0.19$ with a point estimate of exactly the planning $\delta$. Everyone concludes the effect "isn't there."

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Genuinely smaller effect | True effect below $\delta$ | Point estimate well below $\delta$; interval narrow |
| Variance underestimated | Planning used a cleaner population than the test | Realized variance exceeds plan; interval wider than predicted |
| Clustered analysis forgotten | Design effect applied to the mean but not to the SE | SE too small, or n too small by the design effect |
| Triggered population | Only a slice is affected | ITT diluted by $q$; effect on the triggered slice is larger |

**Discriminating check.** Compare the *realized* standard error to the planned one. If realized $\mathrm{SE}$ exceeds $\sqrt{\text{planned variance-sum}/n}$ by more than about 10%, the variance assumption was wrong and the test was underpowered by construction — the point estimate is then uninformative about whether a real effect exists. If the realized SE matches but the point estimate is exactly at $\delta$ with $p=0.19$, check the arithmetic of the original sizing, because an effect exactly at $\delta$ should reject with 50% probability, not 19%.

**Bounded fix.** Report the achieved MDE alongside the result, and state plainly that the interval is compatible with effects from zero to the planning $\delta$. Do not extend the test without a pre-declared extension rule; do register a new test sized on the realized variance, which is now known.

**Verification.** The new test should be sized with the realized variance plus a 10% cushion, and its achieved SE should fall within 5% of the plan at the target $n$. That agreement is the check that the variance estimate has stabilized.

### 8.9 Failure: the ratio metric's denominator moved

**Symptom.** Revenue per session is up 3.1% and the team celebrates, but revenue per user is flat and total revenue is up only 0.4%.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real revenue gain | Treatment increases spend | Total revenue and per-user revenue both rise |
| Session inflation | Treatment creates more sessions per user with the same spend | Sessions per user up; revenue per session up; per-user flat |
| Denominator deflation | Fewer, longer sessions | Sessions down; per-session value up |
| Mix shift | Treatment changes which sessions are counted | Composition of the denominator shifts by device or entry point |

**Discriminating check.** Always report numerator total, denominator total, and the ratio, with the ratio's delta-method SE from [E-04]. If the numerator is flat and the denominator fell, the ratio gain is arithmetic, not behavioural. A quick quantitative test: compute the ratio with the denominator held at its control-arm value; if the lift disappears, the denominator is the driver.

**Bounded fix.** Report both the ratio and its components, and make the decision on the component that matches the business question. For a revenue decision that is per user, per-user revenue is the metric; per-session revenue answers a different question about session quality.

**Verification.** The components should sum consistently: total revenue equals the sum over users, and per-user revenue should reconcile with per-session revenue times sessions per user. Any divergence larger than rounding indicates a join or deduplication defect.

### 8.10 Edge case: zero events in one arm

**Symptom.** The guardrail metric has 0 events in control and 3 in treatment. The ratio is infinite; the standard test is undefined.

**Competing causes.** A genuinely rare harm (the expected count is below 1 in both arms), a logging gap in the control arm, or a treatment-specific error path that produces events control cannot produce.

**Discriminating check.** Compute the expected count under the control rate: if $n p_C<1$, the zero is expected and the appropriate statement is a bound. Use the rule of three — with zero events in $n$ observations, the 95% upper bound on the rate is approximately $3/n$ — and apply it to the difference via an exact or score-based method rather than a Wald interval.

**Bounded fix.** Report an exact confidence bound, state that the design cannot resolve the guardrail, and gate the decision on absolute counts plus a manual review of the three events. Bounded verification: the review should classify each event and confirm whether the mechanism could occur in control.

### 8.11 Edge case: the outcome requires a horizon longer than the test

**Symptom.** The primary metric is 14-day retention, but the test ran 14 days, so the last cohort has only 1 day of exposure.

**Competing causes.** Unequal exposure across cohorts, censoring that differs by arm, or a genuine horizon effect.

**Discriminating check.** Plot the metric by days-since-first-exposure cohort. If the effect differs systematically by cohort, the aggregate is a weighted average of different exposure lengths and is not measuring the 14-day quantity. The check is a heat map of cohort by day, and it takes one query.

**Bounded fix.** Either restrict to cohorts with complete follow-up — which shrinks the sample and must be reflected in the power calculation — or use a survival or cumulative-incidence estimator that handles censoring. Never compare a partially observed cohort to a fully observed one.

**Verification.** The complete-follow-up sample size must still meet the requirement from [E-02]; if it does not, the readout is state 3, not a null result.

> **Takeaway:** every failure in this section produces a plausible-looking number, and the discriminator is always a *different* data cut — assignment counts, pre-period control trend, component decomposition, or an independent source — never a re-reading of the same estimate.

## 9. Complexity & Cost Analysis

### 9.1 Analytical cost of the readout itself

The statistics are cheap; the joins are not. Sizing the analysis honestly means counting the rows the query touches, not the rows the answer contains.

**Notation for this section.** $U$ = units per arm in the assignment table; $E$ = events per unit that must be joined (page views, orders, errors); $k$ = number of joins; $B$ = bytes per joined row; $R$ = bootstrap resamples. All costs are order-of-magnitude, not benchmarks.

```text
FORMULA [E-11]: N_fact = 2 * U * E            (rows scanned for a two-arm readout)
                Data = N_fact * B * k          (bytes read before aggregation)
                T_scan = Data / V              (V = sequential scan throughput, bytes/s)
                T_boot = R * N_fact / P        (P = resample-ops per second)
WHERE: U units/arm; E events/unit; B bytes/row; k join multiplicity;
       V storage throughput; R resamples; P processing rate.
CONSTRAINT: the final GROUP BY operates on N_fact rows; any per-unit metric needs one
            row per unit per event, so E is usually > 1; no early filter is assumed.
NUMERICAL EXAMPLE (small test): U=13,300, E=3.5, B=420, k=1, V=150 MB/s:
            N_fact=93,100 rows; Data=39.1 MB; T_scan=0.26 s.
NUMERICAL EXAMPLE (large test): U=2,000,000, E=4.2, B=380, V=2 GB/s:
            N_fact=16,800,000 rows; Data=6.38 GB; T_scan=3.2 s.
INTERPRETATION: the arithmetic is free; the scan dominates, and it is linear in units,
                so doubling the test doubles it — which is why analyses are cheap
                relative to the traffic they consume.
```

The small-test numbers show why the analysis cost is not the constraint: 0.26 seconds of scanning versus 14 days of traffic. The large-test numbers show the same at scale: 3.2 seconds against a 444-day run at 9,000 sessions per day. The real cost of an experiment is the traffic, not the query.

**The bootstrap is the exception.** A cluster bootstrap with $R=2{,}000$ resamples over 16.8M rows is $3.36\times10^{10}$ row-operations; at an illustrative $5\times10^7$ operations per second that is 672 seconds, over eleven minutes, and it grows linearly in both $R$ and $U$. For the small test, 5,000 resamples over 26,600 units is $1.33\times10^8$ operations, about 2.7 seconds. The practical rule: use the closed form where it exists, bootstrap only for quantiles and for ratio metrics whose delta-method assumptions are strained, and pre-compute the bootstrap in the same run as the primary so the cost is paid once.

### 9.2 Traffic cost of the experiment

Traffic cost is the dominant term and is measured in foregone margin.

```text
FORMULA [E-12]: Cost_traffic = D * S * c * m
WHERE: D days of test; S eligible sessions per day; c baseline conversion rate;
       m margin per order; product is currency.
CONSTRAINT: exposure is a share of traffic and the change has no effect during the test;
            the cost is an opportunity cost of the split, not a cash outflow.
NUMERICAL EXAMPLE: D=14, S=1900, c=0.12, m=46:
            Cost=14*1900*0.12*46 = 146,832 currency units of margin exposed.
INTERPRETATION: a two-week test at this volume puts about 147k of margin through a
                split; that is the budget being risked, and it belongs in the business
                case next to the build cost.
```

The figure is not a loss — half the traffic is in the new experience and may earn more — but it is the scale of what is being randomised, and it is the right basis for comparing "run a test" against "ship and monitor." If the expected gain is 33,203 per month (from Section 5 at 60,000 monthly sessions), a 14-day test costs about 0.3 months of expected gain in calendar time. That ratio, not a p-value calculation, is what a product lead should be weighing.

**Cost of a wrong decision.** Two errors, both priced:

| Error | Mechanism | Illustrative cost |
|---|---|---|
| Ship a null change | Implementation and maintenance, plus opportunity cost of the next change | Build cost, e.g. 36,000, plus 1,500/month maintenance |
| Kill a real win | Foregone margin for the period until the next attempt | 33,203/month times the delay |

With a 10.03% lift on 60,000 monthly sessions and 46 per order, the incremental margin is $0.0120301\times60{,}000\times46=33{,}203$ per month against a build of 36,000 — payback in 1.14 months. The lower confidence bound, +0.40556pp, gives $0.0040556\times60{,}000\times46=11{,}193$ per month and payback in 3.72 months. That spread — 1.1 to 3.7 months — is the honest business case, and it is derived from the interval rather than the point estimate, which is the only defensible way to present it.

### 9.3 Cost of guardrails

Guardrail cost has three components, and only the first is usually counted.

| Component | Unit | Illustrative magnitude | Why it is missed |
|---|---|---|---|
| Metric computation | extra query per guardrail | seconds to minutes | Trivial, so nobody counts it |
| Traffic required for power | units per arm | 81,224 per arm for a 25% relative effect at $p\approx0.003$ | The requirement exceeds the primary metric's, so it is silently dropped |
| Decision latency | days | 86 days at 1,900 sessions/day for the powered version | Calendar cost is paid by the team that wants to ship, not the team that wants the guardrail |

The middle row is the arithmetic that makes "fully powered guardrails" impossible for rare events inside a normal product cycle. The honest responses are: use bound-based tolerances (Section 7.4), measure the guardrail at full traffic after a capped launch (Section 8.4), or run a dedicated longer holdback for the rare harm. Each is a real choice; what is not acceptable is running an unpowered guardrail test and reporting "no significant harm."

```text
FORMULA [E-13]: n_guardrail = (z_{1-alpha/2} + z_{1-beta})^2 * 2 * p_g * (1 - p_g) / (r * p_g)^2
WHERE: p_g baseline guardrail rate; r relative increase to detect; result in units/arm.
CONSTRAINT: binary guardrail event; one-sided or two-sided alpha fixed in advance;
            baseline rate measured on the same population.
NUMERICAL EXAMPLE: p_g=0.0030827, r=0.25, alpha=0.05 two-sided, power 0.80:
            n = 7.84887*2*0.0030827*0.9969173/(0.00077068)^2
              = 0.048242/5.9394e-7 = 81,224 per arm.
INTERPRETATION: note that the answer is independent of p_g when r is fixed — n ~ 2*7.85/r^2
                times a near-1 factor — so the guardrail cost is set by the relative
                effect you refuse to miss, not by how rare the event is.
```

Simplify [E-13] before interpreting it. Substituting $r p_g$ for the absolute effect and cancelling one factor of $p_g$,
$$n=\frac{(z_{\alpha}+z_{\beta})^2\cdot2\,p_g(1-p_g)}{(r p_g)^2}=\frac{2(z_\alpha+z_\beta)^2(1-p_g)}{r^2\,p_g}.$$
Two consequences follow, and the second is counter-intuitive.

**Rarity cuts both ways, and the net effect is unfavourable.** The numerator factor $(1-p_g)$ barely moves as $p_g$ falls, but the denominator carries $p_g$ itself. So $n\propto1/p_g$ at fixed relative $r$: a guardrail at $p_g=0.0005$ needs $2\times7.84887\times0.9995/(0.0625\times0.0005)=502{,}077$ per arm, versus 81,224 at $p_g=0.0031$. A sixfold rarer harm costs roughly sixfold the traffic. The intuition "rare events are hard to measure" is correct, and it is stronger than most teams assume: at fixed *relative* precision, cost scales inversely with the base rate.

**Relative-effect targets are the expensive part.** Setting $r=0.25$ requires about $2\times7.84887\times0.997/(0.0625\times0.0031)=81{,}224$; relaxing to $r=0.50$ divides that by four, to about 20,306. Teams that ask for "detect a small guardrail regression" are specifying a fourfold traffic increase over asking for a moderate one, and the request should be priced before it is accepted.

### 9.4 Cost of the design alternatives

| Design | Traffic multiplier | Additional cost | When the multiplier is justified |
|---|---:|---|---|
| 50/50 unit-randomized | 1.00× | none | Default |
| 80/20 allocation | 1.56× | 56% more traffic for 20% exposure | Risk to users is real but the change is promising |
| 90/10 allocation | 2.78× | 178% more traffic | Early kill-switch testing, not a decision test |
| User-clustered, $m=800$, $\rho=0.02$ | 16.98× | 17× traffic | Interference or carryover makes session splits invalid |
| CUPED at $\rho=0.55$ | 0.70× | one pre-period covariate join | A stable pre-period metric exists |
| Switchback, 14 periods | 1.00×–2.00× | period fixed effects, more analysis | Marketplace interference |
| Sequential with 5 looks | 0.85×–1.00× | boundary discipline, more process | Early stopping has real option value |

The sequential row is the only one that can *save* traffic. Under a true effect, a group-sequential test stops early on average, with the expected saving depending on the effect size relative to the boundary; the honest framing is that it costs 0–15% extra in the worst case (if the effect is small and the test runs to the maximum) and can save a large fraction when the effect is large. For a 5-look Pocock design the maximum information is inflated slightly, and the expected saving under a strong effect is substantial — but the process cost of running boundaries correctly is real.

```text
FORMULA [E-14]: Total_units = 2 * n_per_arm * Mult_design
                Calendar_days = Total_units / S
WHERE: Mult_design from the table (allocation factor [E-05], design effect [E-06],
       or variance factor [E-07]); S eligible units per day.
CONSTRAINT: S is the rate at which *eligible* units arrive, not total traffic; the
            allocation multiplier changes the total, not the daily arrival rate.
NUMERICAL EXAMPLE: n=11,512, Mult=16.98 (clustered): Total=2*11512*16.98=390,947 units;
            at S=1900/day -> 205.8 days. With CUPED at 0.6975 and no clustering:
            Total=2*11512*0.6975=16,060 units -> 8.5 days.
INTERPRETATION: design choices move the calendar by a factor of 24 here, which is why
                the randomization unit is the highest-leverage design decision.
```

### 9.5 Organisational cost

The cheapest part of an experiment to compute and the largest to underestimate.

| Activity | Engineer-hours | Notes |
|---|---|---|
| Pre-registration document | 4–8 | Estimand, metric definitions, guardrails, decision rule |
| Instrumentation and exposure logging | 8–24 | Includes a shadow-run validation of the splitter |
| Dashboard and metric build | 6–16 | Frozen script, hash recorded |
| Guardrail wiring and alerting | 4–12 | Bound-based rule, not a p-value alert |
| Readout analysis and write-up | 8–16 | Including the diagnostics in Section 8 |
| Decision meeting | 1–2 | Should be short if the pre-registration is complete |

At an illustrative loaded rate of 100 per hour, the 31–78 hours above is 3,100–7,800 per experiment — comparable to the compute and far below the traffic cost. The organisational failure mode is different: an experiment without a pre-registration costs 60% less to run and produces results that are re-litigated for weeks, which is where the real expense lives.

### 9.6 Sizing summary: the three-term budget

For any proposed experiment, write three numbers before launch: traffic in units and calendar days, engineering in hours, and the margin exposed. Compare them to the expected value from the interval — not the point estimate — and to the cost of shipping without a test. If the traffic cost exceeds the expected annual gain, the experiment is not worth running; if the engineering cost exceeds the expected gain but the traffic cost does not, the design is too elaborate.

> **Takeaway:** the analysis costs seconds, the traffic costs weeks of margin, and the design choice can move the calendar by a factor of 20 — so optimize the unit and the allocation long before optimizing the query.
