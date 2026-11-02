# Switchback and Operational Pilot Designs — time-based alternation in stores, DCs, and lanes

> Module 05 — Experimentation, Pilots, and Evidence
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 04 topics 04–08; Module 05 topics 01–04
> Version 2026.5

---

## 1. Why this topic exists

Every previous topic in this module assumes you can put two versions of the world side by side. A
customer test splits customers. A cluster test splits sites. A geo test splits markets. Each needs a
boundary across which the treatment can be withheld and on either side of which a counterfactual can
be observed.

Some levers have no such boundary. A pick-path algorithm is loaded into the warehouse management
system (WMS) for the building. A wave-release rule governs every picker on every shift in a
distribution centre (DC). A slotting policy decides where every item sits. A dispatch batching rule
governs a depot's whole van fleet in an hour. A replenishment parameter set applies to a store, and
the store's staff work the whole store. A carrier change applies to a lane, and the lane carries every
order routed across it.

In each case, splitting the unit means splitting the operating model, and the operating model does
not split. You cannot run half a warehouse on one pick-path algorithm, because pickers walk the same
aisles. You cannot run half a lane on one carrier, because the volume on that lane is the volume on
that lane.

So the design moves along the other axis: **time**. The unit takes the treatment for a while, then
does not, then does again. The counterfactual is the same unit's untreated periods. This is a
**switchback design**, and it is the workhorse for operational change where neither the customer nor
the geography is the right assignment unit.

Four reasons this matters, and they are practical rather than methodological.

**The pilot is the last cheap moment to be wrong.** A slotting change that looks like a 9% win in one
DC and is rolled to forty costs the difference between 9% and 1% multiplied by forty sites, for as
long as the policy survives. Reading the pilot correctly costs four months. Reading it incorrectly
costs the rollout, the reversal, and the credibility.

**The naive analysis is the default analysis.** Most operational pilots are read as "the four weeks
after go-live versus the four weeks before". That comparison embeds a trend, a season, a novelty
response, and often a concurrent initiative, and it routinely produces two to three times the
defensible effect. Nobody is lying. The arithmetic is correct and the comparison is wrong.

**The vocabulary is missing.** Teams know what an A/B test is. They do not know what a washout is, why
strict alternation is dangerous, why period fixed effects cannot be estimated from one site, or why a
randomisation test over schedules can disagree sharply with a regression t-test on the same data.
Without that vocabulary the design defaults to whatever the site lead or the vendor proposes.

**The failure is silent.** A switchback with no washout produces a plausible number. A schedule
perfectly correlated with the fortnightly order cycle produces a plausible number. A single-site
pilot with a four-week window produces a plausible number. Broken and sound designs look identical on
a slide. Only the design memo distinguishes them, and only if someone reads it.

### 1.1 The four situations that force a time-based design

| Situation | Example | Why no cross-sectional split exists |
|---|---|---|
| Shared physical resource | Pick-path, slotting, wave release in a DC | Every picker and every aisle is exposed |
| Shared control system | WMS configuration, transport management system (TMS) routing rule | One configuration governs the installation |
| Shared workforce | Shift pattern, task rotation, incentive scheme in a store | The same people work every hour |
| Shared commercial instrument | Carrier on a lane, freight rate, delivery promise in a depot | The instrument is contracted per lane or depot |

The test to apply: **ask what a picker, driver, or store assistant would experience if you randomised
half the unit.** If the answer is "both versions in the same shift", the cross-section is closed.

> **Note:** This topic does not argue that every operational change needs a switchback. It argues
> that the choice between a parallel cluster design, a switchback, a stepped wedge, and a before-after
> comparison should be made deliberately and written down before launch. Module 05 Topic 03 covers
> the parallel case; this topic covers the temporal one.

## 2. Learning outcomes

By the end of this topic you can:

1. **Decide** whether a proposed operational pilot requires time-based assignment, using the
   shared-resource test, and **state** which of switchback, stepped wedge, before-after-with-control,
   or parallel cluster design applies and why.
2. **Construct** a switchback schedule for a named unit, specifying block length, washout length,
   balance, and randomisation of the order, and **justify** each choice against the operational cycle
   and the expected carryover.
3. **Diagnose** aliasing between a proposed schedule and a known calendar cycle, and **redesign** the
   schedule so that treatment is not collinear with the cycle.
4. **Analyse** a multi-unit switchback with unit fixed effects, period fixed effects, and standard
   errors clustered at the unit, and **explain** why the same specification is not identified in a
   single-unit switchback.
5. **Compute** the difference between a naive pre/post comparison, a naive treated-versus-control
   period comparison, and a trend-adjusted estimate from the same block-level data, and **attribute**
   the gap to a specific bias.
6. **Quantify** the resolution of a single-unit switchback from the number of admissible schedules,
   and **state** the smallest attainable p-value before the pilot runs.
7. **Identify** carryover, anticipation, and the pilot effect in a block-level series, and **propose**
   a measurement that separates each from the treatment.
8. **Write** a read-out that separates what was measured from what was extrapolated, including a
   defensible reading of a headline that reports a short terminal window as the result.

## 3. The core mental model

Hold three pictures: what a switchback looks like in time, what it buys relative to the alternatives,
and how the previous block leaks into the current one.

### 3.1 One unit, alternating

```text
   A SWITCHBACK IN TIME — ONE UNIT, TWO CONDITIONS

   block        1     2     3     4     5     6     7     8
   condition    C     T     T     C     C     T     C     T
                |-----|-----|-----|-----|-----|-----|-----|
   measured     ####  ####  ####  ####  ####  ####  ####  ####
   washout      ..    ..    ..    ..    ..    ..    ..    ..
                ^     ^     ^     ^     ^     ^     ^     ^
                switch points — the order of conditions is randomised

   The unit is its own control. Blocks 1, 4, 5 and 7 are the
   counterfactual for blocks 2, 3, 6 and 8.

   The dots at the start of each block are discarded: they are
   the washout, where the previous block's tail is still running.
```

Three properties carry most of the topic.

**Property 1 — the comparison is within-unit, so between-unit variance disappears.** Across 24 stores,
most of the variance in an effect estimate comes from the stores being different from each other. In a
switchback the store is compared with itself, and that component is removed by construction. This is
why switchbacks are attractive when units are few and heterogeneous.

**Property 2 — the variance you removed is replaced by one you created.** Time is now the assignment
axis, so anything that varies over time and correlates with the schedule becomes a confounder.
Seasonality, promotions, pay cycles, weather, labour availability, system releases, and the trend of
the metric itself all move from "nuisance absorbed by randomisation" to "threat that must be designed
away or modelled". You have traded between-unit confounding for within-unit temporal confounding, and
the trade is favourable only if you manage the temporal side explicitly.

**Property 3 — the unit of analysis is the block, not the transaction.** A DC picking 180,000 lines a
week for sixteen weeks produces roughly 2.9 million line records. The design has eight blocks. The
standard error is governed by eight numbers. Treating 2.9 million records as independent is wrong by
roughly the square root of 360,000.

### 3.2 What the design buys, against the alternatives

```text
   THREE WAYS TO ANSWER "WHAT WOULD HAVE HAPPENED?"

   1. BEFORE-AFTER              2. PARALLEL CONTROL        3. SWITCHBACK
      one unit                     two or more units          one unit, over time

      .......|.......              A  .......|.......          C ~~ T ~~ C ~~ T
             |                     B  .......|.......               ^    ^
          pre   post                  pre    post            each block is compared
                                                            with the same unit's
      confounded by:              confounded by:            other blocks
      - trend                     - unit differences
      - season                    - selection of B          confounded by:
      - concurrent change         - shared shocks            - trend in the window
      - regression to mean                                   - carryover between blocks
                                                             - calendar aliasing
```

The three columns are not ranked. They fail differently, and the right choice depends on how many
units you have and how fast the lever can be switched.

| Design | Units needed | Handles irreversible change | Main threat | Typical use |
|---|---|---|---|---|
| Before-after, no control | 1 | Yes | Secular trend, season | Last resort; a case study, not a test |
| Before-after with control site | 2 or more | Yes | Parallel trends | Capital projects, one-site pilots |
| Parallel cluster | 8 or more | Yes | Between-unit variance, selection | Store and DC rollouts with many sites |
| Switchback | 1 or more | No | Carryover, calendar aliasing | WMS rules, dispatch rules, pricing algorithms |
| Stepped wedge | 6 or more | Yes | Time confounding, few clusters | Phased rollouts everyone eventually gets |
| On-off-on reversal | 1 | No | Reversal may not occur | Testing that an effect tracks the switch |

### 3.3 How the previous block leaks into this one

```text
   CARRYOVER: THE EFFECT OF THE LAST BLOCK LEAKS INTO THIS ONE

   effect
     |         ,--.
     |        /    \
     |       /      \
     |      /        \_________
     |     /                   \________
     +----|------|------|------|------|------> days since switch
          0      5     10     15     20
          |<-- transient -->|<-- steady state -->|
          |<-- discard ---->|
              washout

   If the washout is shorter than the transient, the next block
   inherits part of the previous block's effect, and the estimate
   is a weighted average of the effect you wanted and the effect
   you were trying to wash out.
```

Carryover has three sources in operations, with different decay times.

- **Work-in-process (WIP) tail.** Orders released under the old rule are still being completed after
  the switch. Hours to a day in a DC; a shift or more in a plant.
- **Learning and adaptation.** Operators need time to reach steady-state performance under a new rule.
  This is a ramp-up, not a decay: hours for a pick-path change, weeks for a slotting change that
  requires relearning item locations.
- **State dependence.** A dispatch rule that emptied a backlog changes the state the next block starts
  from. A replenishment change alters the inventory position the following block inherits. This
  persists far longer than either of the others.

> **Note:** The washout absorbs the first and part of the second. It does not absorb the third. State
> dependence must be handled by modelling the state — backlog, inventory position, queue length — as
> a covariate, or by switching at a point where the state resets. This is the most under-appreciated
> threat in operational switchbacks.

### 3.4 The estimator follows from the design

| Design | Units | Estimator | Where identification comes from |
|---|---|---|---|
| Single-unit switchback | 1 | Treatment plus a smooth time trend, or a design-based test over schedules | Within-unit alternation, net of trend |
| Single-unit switchback with block fixed effects | 1 | **Not identified** — treatment is collinear with the block dummies | Nowhere |
| Multi-unit switchback, independent schedules | J | Unit fixed effects, period fixed effects, treatment; cluster by unit | Different units are in different conditions in the same period |
| Stepped wedge | J | Unit fixed effects, period fixed effects; cluster by unit | Staggered adoption times |
| Before-after with control site | 2 or a few | Difference-in-differences; cluster at site | Parallel trends between sites |
| On-off-on reversal | 1 | Phase contrasts plus a reversal check | The sign of the change follows the switch |

The second row is the one to memorise. **Period fixed effects are identified only when the schedule
varies across units within a period.** In a single site every unit is in the same condition at the
same time, so a period dummy and the treatment indicator are the same variable. Software will often
return a coefficient anyway, because the collinearity is not exact in floating point, and the
coefficient will be arbitrary.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Switchback | A design in which one unit alternates between conditions over time, so each condition serves as the other's counterfactual | Used for any before-after comparison | Condition per block, per unit |
| Time-based randomisation | Assigning treatment by randomly ordering time blocks rather than units | Called randomisation when the operator chose the order | Realised schedule plus seed |
| Epoch | One contiguous period in which a unit holds a single condition | Confused with a block when washout is inside the block | Start and end timestamp |
| Block | The unit of analysis: one epoch's measured, post-washout observations | Used to mean a calendar month | Block identifier, unit identifier |
| Block length | Duration of one epoch, including its washout | Chosen for convenience rather than for steady state | Days or hours |
| Washout | The discarded interval at the start of a block, during which the previous condition's effect persists | Omitted because "the switch is immediate" | Days or hours discarded |
| Wash-in | The discarded interval during which the new condition's effect is still developing | Confused with washout | Days or hours discarded |
| Carryover | The part of one block's effect that persists into the next block | Treated as measurement lag | Days, or a decay parameter |
| Carryover order (m) | The number of periods over which a treatment's effect persists in the optimal-design literature | Ignored in schedule construction | Integer number of periods |
| Anticipation | Behaviour change before the switch takes effect | Missed because it looks like a pre-period improvement | Weeks before switch |
| Transient | The non-steady portion of a block immediately after a switch | Averaged into the block mean | Days |
| Steady state | The portion of a block in which performance is stable under the condition | Assumed rather than demonstrated | Days |
| Period effect | A shift in the outcome common to all units in a given period | Confused with the treatment effect | Fixed effect, one per period |
| Unit fixed effect | A unit-specific level shift, removing between-unit differences | Omitted when the analysis runs at transaction level | Fixed effect, one per unit |
| Cluster-robust standard error | A standard error allowing correlation within a cluster | Computed at transaction level when the cluster is the unit | Clustered by unit identifier |
| Intracluster correlation (ICC) | Share of total variance lying between clusters | Assumed small in operational data | Proportion, 0 to 1 |
| Autocorrelation | Correlation of a series with its own lagged values | Ignored in pre/post t-tests | AR(1) coefficient, −1 to 1 |
| Aliasing | A schedule collinear with a known cycle, so the two cannot be separated | Invisible when the schedule is strict alternation | Correlation between schedule and cycle |
| Count balance | Equal numbers of treated and control blocks | Confused with the whole of balance | Count per condition |
| Position balance | Treated and control blocks spread across the window rather than clustered at one end | Not computed at all | Mean block index per condition |
| Deterministic alternation | Strict on-off switching at a fixed interval | Assumed equivalent to randomisation | Schedule pattern |
| Counterbalancing | Ordering conditions so each appears equally often in each position | Omitted in short pilots | Position counts |
| Stepped wedge | A design in which units adopt in a randomly ordered sequence and all eventually adopt | Used when the treatment cannot be withdrawn | Adoption time per unit |
| On-off-on reversal | An A-B-A-B design using the return to control as evidence | Confused with a two-block switchback | Phase sequence |
| Before-after with control | Change in a treated site against change in an untreated site | Reported as a randomised result | Two sites, two periods |
| Interrupted time series | A regression on a long series with a level and slope change at the intervention | Used with too few pre-intervention points | Segments before and after |
| Pilot effect | The bundle of selection, novelty, attention, slack, and measurement that makes a pilot outperform its own rollout | Used as a synonym for the Hawthorne effect | Composite; decompose it |
| Hawthorne effect | Behaviour change caused by awareness of being observed | Invoked as a universal explanation | Effect size, where measurable |
| Novelty effect | A temporary improvement caused by newness | Confused with durable process improvement | Decay time |
| Research participation effect | The measured effect of taking part in a study, independent of the intervention | Called the Hawthorne effect | Odds ratio in a meta-analysis |
| Compliance | The degree to which the assigned condition was actually in force | Assumed total | Share of blocks or shifts |
| Intent-to-treat (ITT) | The effect of assigning a condition, regardless of delivery | Reported as the effect of the intervention itself | Effect per assigned block |
| Guardrail metric | A secondary outcome that must not degrade | Omitted when the primary metric improves | Named metric with a threshold |
| Minimum detectable effect (MDE) | The smallest true effect a design detects at stated power and significance | Computed after the pilot | Outcome units and per cent |
| Secular trend | A smooth change in the outcome unrelated to treatment | Absorbed into the treatment estimate | Units per block |
| Resolution floor | The smallest p-value a schedule space can produce | Reported as a failure to reach significance | `2 / number of schedules` |
| Randomisation inference | Inference from the distribution of the estimate over all admissible schedules | Replaced by a t-test on block means | Exact or Monte Carlo p-value |
| Run-in period | A preliminary period used to reach steady state before measurement | Confused with washout | Days before block 1 |

Four deserve expansion, because the errors they cause are expensive.

**Washout versus wash-in.** Washout removes the residue of the previous condition; wash-in removes the
ramp-up of the current one. They usually overlap and are usually both needed. Discarding two days
after each switch does both at once, which is fine if two days covers both. Discarding nothing is a
claim that both are zero — a claim about the process that should be written down and tested.

**Carryover versus period effect.** A period effect is a shift common to all units in a period: a
heatwave, a system outage, a public holiday. Carryover depends on what each unit was doing in the
previous period. Period effects can be estimated and removed when you have multiple units. Carryover
cannot, because it is not common across units — it follows each unit's own history. That is why
carryover is the central threat and period effects are a nuisance.

**Pilot effect versus Hawthorne effect.** The Hawthorne effect is one component of the pilot effect,
and the evidence suggests it is smaller and less general than the folklore (section 9.6). The pilot
effect also contains site selection, management attention, extra resourcing, and the temporary
suspension of normal constraints. Treating "the Hawthorne effect" as the whole explanation lets the
other four go unmeasured.

**Resolution floor versus statistical significance.** A design with 70 admissible schedules cannot
produce a two-sided p-value below `2/70 = 0.029`, whatever the data say. Reporting `p = 0.06, not
significant` without the floor describes a design limitation as an empirical finding. It is the
temporal analogue of the donor-pool floor in Topic 04.

## 5. The mechanics, step by step

### Phase A — Decide that time is the assignment axis

**Step 1. Name the decision, the switch, and the switcher.** Write three sentences: what changes, who
physically or technically changes it, and how long the change takes to take effect. "The wave-release
rule changes; the WMS configuration team changes it; it takes effect at the next wave boundary." If
the second sentence has no answer, the lever is not operational and no design will help. If the third
is measured in weeks, a switchback is probably unavailable.

**Step 2. Apply the shared-resource test.** Ask what a picker, driver, or store assistant would
experience if half the unit were randomised. "Both versions in the same shift" closes the
cross-section. "One version, because the aisles are physically separate" means a zone-level cluster
design may be available, and that is usually more powerful.

**Step 3. Check whether a parallel design is already sufficient.** The Gap Inc. scheduling study
randomised 28 stores and detected a 5.1% productivity change. The Alibaba/Cainiao packing study
randomised 757 workers inside warehouses and measured 782,360 packages. Both are parallel designs and
both are stronger than a switchback, because they had units to spare. A switchback is the design of
last resort on the cross-section, not the first choice.

**Step 4. Check reversibility, switching cost, and latency.** A switchback requires a return to
control. A WMS rule reverts; a physical re-layout does not, at any sane cost. Every switch also
consumes capacity: retraining, reconfiguration, a period of lower throughput. If a switch costs half
a day of output, a sixteen-switch design has paid eight days for its identification. That cost is
legitimate and should be budgeted; it is also why blocks cannot be arbitrarily short.

### Phase B — Define the unit, the block, and the outcome

**Step 5. Define the unit as the smallest set of resources that share the lever.** Where pickers work
any aisle, the unit is the building. Where zone pickers are dedicated, it may be the zone. On a lane,
it is the lane. The test: can two candidate units be in different conditions at the same time without
one affecting the other? If yes, you have a multi-unit switchback, which is much stronger.

**Step 6. Choose block length from the operational cycle, not the calendar.** The block must cover
switch latency plus transient plus enough steady-state days to measure. It should also align with the
cycle over which the outcome is defined: weekly for DC labour productivity, daily for depot dispatch,
one shift for an algorithmic dispatch rule. Two weeks is a common DC and store compromise.

**Step 7. Set the washout as a stated number of days, justified from the three sources.** The WIP tail
is measurable from the WMS: how long after a rule change do orders released under the old rule finish?
The learning ramp is measurable from previous changes. State dependence needs a model. One to three
days is typical for a WMS rule change; zero is a claim that all three are zero.

**Step 8. Fix the primary outcome, the metric contract, and the guardrails.** State metric name,
numerator, denominator, population, time bucket, treatment of rework, and the aggregation rule. For a
DC the natural primary is units per labour hour (UPH): units handled divided by paid labour hours over
the same block. Paid hours, not productive hours, because the treatment may change the labour mix.
Guardrails: mispicks per 1,000 lines, recordable safety incidents, overtime hours per full-time
equivalent (FTE), and voluntary turnover. A pilot reporting UPH and not mispicks has reported the
favourable half.

### Phase C — Build the schedule

This is the step that separates a competent switchback from a decorative one, and it is the one most
often skipped.

**Step 9. Fix the number of blocks from power, then round to an even number.** Eight blocks (four
treated, four control) is the practical minimum for one unit and is thin. Twelve is materially
better. Sixteen is better still if the operation bears the switching cost.

**Step 10. Specify balance in two senses.** Count balance means equal numbers per condition. Position
balance means the conditions are spread across the window rather than clustered at one end. Count
balance alone is not enough: four treated blocks all in the second half is perfectly count-balanced
and completely confounded with the trend. Report the mean block index per condition; a difference
above about one block is a warning.

**Step 11. Check the schedule against every known calendar cycle.** List the cycles that move the
outcome — day of week, week of month, pay fortnight, promotional calendar, month-end close,
quarter-end, holidays, contract review dates, system release weekends — then correlate the treatment
indicator with each over the schedule. A correlation near ±1 means the design cannot separate
treatment from that cycle and the schedule must be redrawn.

**Step 12. Never use strict alternation.** `C, T, C, T, C, T, C, T` is the most natural thing to write
and the most dangerous. It is perfectly correlated with any two-block cycle, which in retail and
distribution includes the promotional fortnight, the pay fortnight, and the delivery rotation. It also
inflates the variance of a trend-adjusted estimate. Alternation feels balanced and is the opposite.

**Step 13. Randomise the order and record the seed.** Draw from the set of balanced schedules before
launch, with a recorded seed and timestamp. Two refinements are worth using: constrain the draw to
schedules with position balance inside a stated tolerance, and record the constraint and the number of
rejected draws, because the constraint changes the null distribution used later.

**Step 14. For multiple units, draw independent schedules.** Independence across units is what
identifies period fixed effects: in any week some units are treated and some are not, so a week effect
can be separated from a treatment effect. If all units switch together, the design collapses to a
single-unit switchback with more data and the same weakness. For a stepped wedge, the adoption order
is the randomisation and should be recorded the same way.

### Phase D — Power, before launch

**Step 15. Estimate block-level noise from history.** Take the same unit, a window of the planned
length, and compute the standard deviation of block-level values after removing trend. Use several
historical windows. The relevant quantity is the residual standard deviation at block level, not the
day-level standard deviation.

**Step 16. Compute the MDE.**

```text
For a single-unit switchback with K blocks, treatment plus linear trend:
    residual df = K - 3
    SE(effect)  ~ s / sqrt( sum over blocks of (d_i - mean d)^2 , adjusted for the trend )
    MDE         = ( t(0.975, K-3) + t(0.80, K-3) ) x SE
```

The trend adjustment matters. If the schedule is well balanced on position the variance inflation is
small; if not, it can double the MDE. That is the quantitative reason Step 10 exists.

**Step 17. Compare the MDE with the hoped-for effect, and stop if it is larger.** If the hope is below
the MDE, the pilot cannot succeed even if the lever works. The honest options are more blocks, more
units, a coarser and less noisy outcome, a bolder change, or no pilot. Running an underpowered pilot
and reading a null as evidence of no effect is the most common way to kill a good idea.

### Phase E — Run

**Step 18. Instrument the switch itself.** Log every switch: timestamp, executor, previous and new
configuration, and whether it happened on time. Compliance below about 90% turns the pilot into a
diluted version of itself. The usual failure is not defiance but drift: the old rule is quietly
reinstated for a difficult shift, or the new rule is applied only to some order types.

**Step 19. Keep a concurrent-change log.** Any other initiative, system change, layout change,
promotion, or staffing change inside the window goes in with a date. The log is the only defence
against the post-hoc explanation that arrives three months later.

**Step 20. Collect the state variables.** Backlog at block start, inventory position, queue length,
headcount, absence rate, order mix. These let you test state dependence, and they are what will be
used to argue the blocks were incomparable if you do not have them.

**Step 21. Do not change the schedule.** The temptation arrives in week 6, when the result looks flat.
Changing block length, dropping a block, or extending the window after seeing the data converts a
randomised design into a convenience sample with a randomisation story attached.

### Phase F — Analyse

**Step 22. Aggregate to the unit-by-block table first.** One row per unit per block. For a single unit
with eight blocks, eight rows. Everything downstream operates on that table. If the row count is in
the millions, the aggregation step was skipped and every standard error afterwards will be wrong.

**Step 23. Choose the estimator that matches the design, using section 3.4.** Single unit: treatment
plus a smooth trend, or a design-based test over schedules, or both. Multiple units with independent
schedules: unit and period fixed effects plus treatment, clustered at the unit. Stepped wedge: the
same, with period fixed effects mandatory. One control site: difference-in-differences, with the
caveat that two units cannot support a cluster-robust standard error.

**Step 24. Report the block-by-block path, not only the mean.** A mean over eight blocks hides a
decay, a reversal, or one anomalous block. Plot block means against block index with condition
marked. If treated blocks start high and converge to the control level, the mean is a misleading
summary of a novelty response.

**Step 25. Run the anticipation and carryover checks.** Compare the last control block before a switch
with the control trend: if it is elevated, operators were preparing and the control period is
contaminated at its end. Compute the outcome separately for the first measured days of each block and
the remainder: if the first days sit in the direction of the previous condition, the washout is too
short. The fix is a longer washout and a re-analysis, not a lagged treatment variable.

**Step 26. Run the design-based test.** For a single unit, enumerate or sample the admissible
schedules, recompute under each, and count those at least as extreme as the observed estimate. Report
the p-value and the resolution floor. For multiple units, use a randomisation test over unit-level
assignment, a cluster-robust interval, or both.

**Step 27. Decompose the mechanism.** UPH is a ratio. Decompose into units per picking hour, hours per
order, orders per block, and order-size mix. A change that raises UPH by reducing small orders has not
raised productivity; it has changed the work. This is where an operational pilot differs from a
marketing test: the headline is a ratio of two operational quantities and both can move.

### Phase G — Read out

**Step 28. Apply the reporting standard.** In order: the decision and its owner; the design and the
realised schedule; unit and block definitions; the washout and what it was meant to absorb; the
primary outcome and metric contract; the block path; the effect in absolute units and as a percentage
of the control level; the interval and the method that produced it; the design-based p-value and its
floor; compliance and the concurrent-change log; the guardrails; the decomposition; and what the pilot
cannot see.

**Step 29. State the extrapolation separately from the measurement.** The pilot measured one unit over
sixteen weeks; the rollout applies to forty units over years. Between those statements sit five
distinct generalisation gaps: site representativeness, season, novelty decay, scale-dependent
operations, and the difference between a temporary exception and a permanent policy. Name them. Do
not average them into a discount factor, which is what a "haircut" does and what makes the number
unfalsifiable.

### 5.1 Why a naive pre/post t-test fails, specifically

The default analysis is a two-sample t-test of before against after. It fails for four separable
reasons, and being able to name them is most of the defence.

**Reason 1 — the trend is in both numbers.** If the metric was already improving at one unit per
block, the post-period average exceeds the pre-period average with no treatment effect at all. In the
worked example in section 6, the trend accounts for roughly half the reported headline.

**Reason 2 — the standard error assumes independence that does not exist.** Daily observations within
a block are autocorrelated. For an AR(1) process with coefficient ρ, the variance of a mean of n
observations is inflated by approximately `(1+ρ)/(1−ρ)`. At ρ = 0.6 the inflation is a factor of 4, so
the standard error is understated by half, and the test is roughly twice as confident as it should be.

**Reason 3 — the windows are chosen after the fact.** "Four weeks before, four weeks after" is chosen
because it is round and the data was available, not because four weeks is the right horizon. Different
windows give different answers, and the reported window is sometimes the best-looking one.

**Reason 4 — regression to the mean selects the site.** Sites are often chosen because they were
underperforming and someone wanted a fix, or overperforming and someone wanted to learn. Either way
the pre-period contains an extreme value, and the post-period regresses toward the site's own mean
with no treatment at all.

> **Note:** Reasons 1 and 2 are what a switchback fixes by construction. Reason 3 is fixed by
> pre-registration. Reason 4 is fixed by randomising among eligible sites rather than accepting
> volunteers, which is why the pilot effect is a selection problem before it is a measurement problem.

## 6. Worked example

A grocery retailer — call it Ashfield — operates a regional DC serving 180 stores. All figures are
illustrative and internally consistent; the arithmetic is reproducible from the block table.

### 6.1 The decision

The DC picks in discrete waves: each order is released to a picker, picked in full, packed. The
operations team proposes **batch picking with sortation**: pickers take a batch of six orders, pick
the combined requirement in one pass, and sort at a consolidation station. The hypothesis is that
this reduces travel per unit and raises throughput per labour hour without raising the mispick rate.

The decision owner is the DC general manager, on a recommendation from the regional operations
director. The decision is whether to adopt batch picking across the region's eight DCs. The lever is a
WMS configuration applying to every picker in the building. A zone-level split was rejected because
pickers rotate across zones within a shift and the consolidation station is shared.

### 6.2 The design

- **Unit:** the DC. One unit.
- **Blocks:** eight blocks of two weeks, sixteen weeks. Each block has ten working days; the first two
  are washout, leaving eight measured days. Total measured: 64 days.
- **Washout:** two working days, intended to absorb the WIP tail and the initial learning ramp.
- **Schedule:** drawn at random from the 70 balanced eight-block schedules, constrained so the mean
  block index of the two conditions differ by at most one. Realised: `C, T, T, C, C, T, C, T`.
- **Primary outcome:** UPH, units handled divided by paid labour hours, aggregated to the block.
- **Guardrails:** mispicks per 1,000 lines, recordable safety incidents, overtime hours per FTE.

> **Note:** The realised schedule is a favourable draw. Its treatment indicator is nearly uncorrelated
> with block index, which is why the trend-adjusted estimate is as precise as it is; a random draw
> would often be worse. Section 6.7 reports a design-based p-value alongside the regression interval,
> and the two disagree for exactly this reason.

### 6.3 The block table

| Block | Weeks | Condition | Measured days | UPH |
|---:|---|---|---:|---:|
| 1 | 1–2 | Control | 8 | 104.9 |
| 2 | 3–4 | Treated | 8 | 109.7 |
| 3 | 5–6 | Treated | 8 | 113.1 |
| 4 | 7–8 | Control | 8 | 106.3 |
| 5 | 9–10 | Control | 8 | 108.2 |
| 6 | 11–12 | Treated | 8 | 115.8 |
| 7 | 13–14 | Control | 8 | 109.0 |
| 8 | 15–16 | Treated | 8 | 117.6 |

The four weeks before block 1 averaged **103.6 UPH**. That is the number the sponsor used.

### 6.4 Three estimates from the same eight numbers

**Estimate 1 — the sponsor's pre/post.** Final four weeks against the pre-pilot baseline:
`117.6 / 103.6 = 1.1351`, so **+13.5%**.

**Estimate 2 — all treated blocks against all control blocks.**

```text
control mean = (104.9 + 106.3 + 108.2 + 109.0) / 4 = 428.4 / 4 = 107.10
treated mean = (109.7 + 113.1 + 115.8 + 117.6) / 4 = 456.2 / 4 = 114.05
difference   = +6.95 UPH, or 6.95 / 107.10 = +6.5%
```

**Estimate 3 — treatment with a linear time trend.** Regress block mean on a treatment indicator and
block index:

```text
n = 8   mean block index = 4.5   mean treatment = 0.5
Sxx = 42.0   Sdd = 2.0   Sxd = 1.0   Sxy = 47.7   Sdy = 13.9
det = Sxx*Sdd - Sxd^2 = 83.0
trend     = (Sdd*Sxy - Sxd*Sdy) / det = (2.0*47.7 - 1.0*13.9) / 83.0 = 0.982 UPH per block
treatment = (Sxx*Sdy - Sxd*Sxy) / det = (42.0*13.9 - 1.0*47.7) / 83.0 = 6.459 UPH
s = 1.075 on 5 df
SE = s * sqrt(Sxx / det) = 1.075 * 0.7114 = 0.765
t  = 8.45      95% CI = 6.459 +/- 2.571 * 0.765 = (4.49, 8.42) UPH
relative = 6.459 / 107.35 = +6.0%   (fitted control level at the mean block)
```

So: **+13.5%, +6.5%, +6.0%** from the same sixteen weeks. Only the third is a defensible estimate.

### 6.5 Where the gap comes from

The gap between estimates 1 and 2 is a window choice: the baseline sits at block index 0 and the final
block at index 8.

```text
fitted control at block 8 = 107.35 + 0.982 * (8 - 4.5) = 110.79
trend contribution to the headline = 110.79 - 103.6 = 7.19 UPH
headline = 117.6 - 103.6 = 14.0 UPH
treatment = 6.46      residual noise = 14.0 - 7.19 - 6.46 = 0.35
```

Roughly half the headline is a trend already running before anyone proposed batch picking. The trend
is not mysterious: a goods-to-person station was commissioned the previous quarter, the pick faces
were re-slotted twice, and the labour agency conversion reduced turnover. None of it is in the
pilot's story and all of it is in the pilot's number.

The gap between estimates 2 and 3 is half a block of imbalance:

```text
mean block index, treated = (2+3+6+8)/4 = 4.75;  control = (1+4+5+7)/4 = 4.25
difference = 0.50 blocks;  bias = 0.50 * 0.982 = 0.49 UPH
naive 6.95 - adjusted 6.459 = 0.49   (matches exactly)
```

Half a block of imbalance, half a unit per hour of bias, about 7% of the effect. It is worth
understanding precisely because it is small: the naive contrast is defensible only when the schedule
is position-balanced, and you cannot know that without computing it.

### 6.6 Reading the 4-week win

The sponsor's slide has four elements, each individually correct: "the final four weeks ran at 117.6
UPH", "the baseline was 103.6", "that is a 13.5% improvement", "at regional scale this is worth £2.1m
a year". All true, and the last one is built on a wrong input.

Five questions convert the slide into a decision.

**What is the baseline?** Not "four weeks before launch" but "what would UPH have been in weeks 15–16
without batch picking". The trend-adjusted answer is 110.8, so the effect in that block is 6.8 UPH —
consistent with the average of 6.46 and inside its interval.

**Is the last block the right block to quote?** It has the most learning and the most favourable trend
position. Quoting it is quoting the maximum. The average across the four treated blocks is the
steady-state statistic.

**Is the effect still there?** The adjusted estimate is stable across treated blocks, but a pilot that
ends while the effect is still growing has not shown a steady state. The honest statement is "present
and roughly constant from block 3 onward", not "6% and will remain 6%".

**What did the guardrails do?** Section 6.9.

**What will the trend do next?** The trend belongs to the DC, not the treatment. If the goods-to-person
station is still ramping, the trend continues and the effect sits on top of it. If the trend was a
one-off reorganisation, it flattens and the effect stands alone. The two cases have different rollout
economics and sixteen weeks of data do not distinguish them.

### 6.7 The design-based test, and why it disagrees

Under the sharp null of no effect in any block, all 70 balanced schedules are equally plausible, so
the distribution of the trend-adjusted estimate over those schedules is the correct null.

```text
admissible schedules = C(8,4) = 70
SD of the estimate over schedules = 3.01 UPH
observed = 6.459 UPH;  schedules with |estimate| >= 6.459  = 4
two-sided randomisation p = 4/70 = 0.057;  floor = 2/70 = 0.029
```

The regression gave `t = 8.45` and an interval of (4.49, 8.42). The randomisation test gives 0.057.
Same eight numbers, wide disagreement.

The reason is the schedule space. Some balanced schedules put treated blocks at both ends
(`T,T,T,C,C,T,C,C`), aligning treatment with the curvature of the trend and producing trend-adjusted
estimates of ±7.0 UPH from the trend fit alone. Those inflate the null SD to 3.01, four times the
regression's 0.765. The regression interval is conditional on the drawn schedule and on the linear
trend being right; the randomisation p-value is conditional on neither.

> **Note:** Neither number is wrong. The interval answers "given this schedule and this trend model,
> how precise is the estimate". The randomisation p answers "how surprising is this estimate among the
> schedules I could have drawn". When the schedule was randomised, the second is what the design was
> built to answer, and it is the conservative one. Report both; let the design-based one govern.

The uncomfortable and correct consequence: **a sixteen-week single-site switchback cannot reliably
certify a 6% effect.** Lowering the floor requires more blocks — 12 gives `C(12,6) = 924` and 0.0022,
16 gives `C(16,8) = 12,870` and 0.0002 — or more units, since four DCs with independent schedules give
four independent switchbacks and a far tighter cluster-robust interval.

### 6.8 The block path and the carryover check

| Block | Condition | UPH | First 2 measured days | Remaining 6 days | Gap |
|---:|---|---:|---:|---:|---:|
| 1 | C | 104.9 | 104.2 | 105.1 | −0.9 |
| 2 | T | 109.7 | 108.4 | 110.1 | −1.7 |
| 3 | T | 113.1 | 112.6 | 113.3 | −0.7 |
| 4 | C | 106.3 | 107.4 | 105.9 | +1.5 |
| 5 | C | 108.2 | 108.9 | 108.0 | +0.9 |
| 6 | T | 115.8 | 114.1 | 116.4 | −2.3 |
| 7 | C | 109.0 | 110.2 | 108.6 | +1.6 |
| 8 | T | 117.6 | 116.0 | 118.1 | −2.1 |

Read the gap column against the previous block's condition. Every block following a switch into
treatment has a negative gap (−1.7, −0.7, −2.3, −2.1). Every block following a switch into control has
a positive gap (+1.5, +0.9, +1.6). That is the signature of an insufficient washout: the first two
measured days of each block still carry a third to a half of the previous condition's effect. Two days
absorbs the WIP tail but not the learning ramp; pickers need about four days to reach steady-state
speed under a new rule.

The correction is not a lagged variable. It is to re-cut the blocks with a four-day washout, which
leaves six measured days per block and 48 in total. The point estimate rises to 6.71 UPH with a
standard error of 0.94 and an interval of (4.29, 9.13). The estimate moves up because the dilution is
removed; the interval widens because a quarter of the data is discarded. The conclusion does not
change. The precision does.

### 6.9 Guardrails and the decomposition

| Metric | Control blocks | Treated blocks | Change | Verdict |
|---|---:|---:|---:|---|
| UPH | 107.10 | 114.05 | +6.5% | Primary outcome |
| Mispicks per 1,000 lines | 0.71 | 0.94 | +32% | **Guardrail breach** |
| Recordable safety incidents | 2 | 3 | +50% on a base of 2 | Inconclusive |
| Overtime hours per FTE | 4.2 | 3.1 | −26% | Favourable |
| Units per paid hour, excluding rework | 107.10 | 112.90 | +5.4% | After rework |

The mispick rate did not merely fail to improve; it rose by a third. The decomposition shows why:
batch picking requires a sortation step, and sortation errors concentrate on look-alike items. The
rework cost — return, credit, replacement delivery — is not in the UPH denominator, because rework is
handled by a separate team in a separate cost centre.

```text
gross UPH gain                                  +6.95 units/hour
rework-adjusted gain                            +5.80 units/hour
incremental mispicks per week at current volume   212
rework cost per 1,000 lines                       GBP 41
weekly rework cost                              GBP 8,692
weekly labour hours saved at 5.80 UPH              1,340
weekly labour saving at GBP 14.20/hour          GBP 19,028
net weekly benefit                              GBP 10,336
```

Still net positive at this volume. That is not the point. The point is that the primary outcome
reported a 6.5% gain while the operation experienced a 5.4% gain and a 32% quality regression, and a
read-out reporting only the first number has misdescribed the result in the direction that favours the
proposal.

> **Note:** The safety row is deliberately inconclusive. Three against two on a base of two is a
> count, not a signal; reporting it as "+50%" would abuse a small denominator. The correct treatment
> is to report the counts and state that the pilot was not sized to detect a change in safety
> outcomes. Guardrails with low base rates need their own surveillance, not a percentage.

### 6.10 The recommendation

"The pilot is consistent with a 6.0% throughput gain — 6.5 UPH, model-based 95% interval 4.5 to 8.4 —
with a design-based p of 0.057 against a floor of 0.029, so it is not conclusive at 5% on its own.
Mispicks rose from 0.71 to 0.94 per 1,000 lines, removing roughly 45% of the gross benefit; the
rework-adjusted gain is 5.4%. The first two measured days of every block still carry the previous
condition's effect, so the true effect is probably near the upper end of the interval. Recommend a
second phase with a four-day washout across four DCs with independent schedules, plus a separate test
of sortation error reduction, before a regional standard is set."

That paragraph is longer than the sponsor's slide and it is the one that survives the rollout.

## 7. Code, query, or calculation

The calculation below reproduces the adjusted estimate and the design-based p-value from the eight
block means. It uses only the Python standard library, and it takes eight numbers rather than eight
million.

```python
import itertools, statistics as st
y = [104.9, 109.7, 113.1, 106.3, 108.2, 115.8, 109.0, 117.6]   # UPH by block
x = list(range(1, 9))                                          # block index
obs = [0, 1, 1, 0, 0, 1, 0, 1]                                 # C,T,T,C,C,T,C,T

def eff(y, d):
    """Treatment coefficient from: block mean ~ block index + treatment."""
    mx, md, my = st.mean(x), st.mean(d), st.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    sdd = sum((b - md) ** 2 for b in d)
    sxd = sum((a - mx) * (b - md) for a, b in zip(x, d))
    sxy = sum((a - mx) * (c - my) for a, c in zip(x, y))
    sdy = sum((b - md) * (c - my) for b, c in zip(d, y))
    return (sxx * sdy - sxd * sxy) / (sxx * sdd - sxd ** 2)

est = eff(y, obs)
paths = [[1 if i in c else 0 for i in range(8)]
         for c in itertools.combinations(range(8), 4)]
p = sum(abs(eff(y, d)) >= abs(est) for d in paths) / len(paths)
print(f"effect={est:.2f} UPH  paths={len(paths)}  p_abs={p:.4f}")
```

Expected output, one line: `effect=6.46 UPH  paths=70  p_abs=0.0571`. The printed value is one-sided
in absolute value; section 6.7 quotes `4/70 = 0.057` because the null distribution over schedules is
symmetric by construction (swapping every condition negates the estimate), so the count already spans
both tails.

**What the code assumes, and what each assumption costs if it is wrong.**

1. **Blocks are independent given the trend.** If consecutive blocks share a state variable — backlog,
   inventory position — the residual SD is understated and the interval too narrow. Test by adding the
   lagged outcome; if significant, independence is false.
2. **The trend is linear over sixteen weeks.** A curved trend is absorbed into the treatment estimate.
   Add a quadratic and compare; with five residual degrees of freedom the test has almost no power,
   which is itself information about the design.
3. **The schedule was drawn uniformly from the 70.** If the team drew several and kept the best-looking
   one, the reference distribution is not the 70 and the p-value is optimistic. Record the draw.
4. **The sharp null is the hypothesis of interest.** The test asks whether the treatment had an effect
   in any block; a treatment that helps early blocks and hurts late ones equally will not be rejected.
5. **Eight blocks are enough to say something.** They are enough to produce a number, not enough to
   certify a 6% effect. The code will not tell you this; the resolution floor will.

## 8. Data, standards, and regulatory context

A switchback is a data-alignment problem before it is a statistical one. Assignment, outcome, and
operational state must be recorded at the same block boundary, in the same time zone, under the same
metric definition, for the whole window. Most failed pilots fail here, not in the model.

### 8.1 The data a switchback needs

| Object | What it must record | Where it usually lives | Failure if missing |
|---|---|---|---|
| Schedule | Realised condition per block per unit, with timestamps, seed, constraint | A document nobody maintains | Design cannot be reconstructed; analysis becomes observational |
| Switch log | When the switch executed, by whom, whether on time | Change-management tickets, if anywhere | Compliance unknown; a null result is uninterpretable |
| Outcome events | Units, orders, lines, or transactions with operation timestamps | WMS, TMS, point of sale (POS), ERP | Block aggregation impossible without reconstruction |
| Labour hours | Paid and productive hours by block, including agency and overtime | Time and attendance, payroll | The UPH denominator is wrong, usually flattering the pilot |
| Quality events | Mispicks, damages, returns with reason codes, rework hours | WMS exception log, quality system | Guardrails unevaluable; rework cost invisible |
| State variables | Backlog, queue length, inventory position at block start | WMS, planning system | State dependence untestable, and it is the main long-range confounder |
| Concurrent-change log | Other initiatives with dates and scope | Project management office, or nothing | Every later anomaly gets attributed to "noise" |
| Calendar | Promotions, holidays, pay cycles, releases, contract dates | Commercial and IT calendars | Aliasing cannot be checked |

The schedule and the switch log are the two almost always missing and almost always decisive. A
schedule reconstructed from a chat thread is not a randomisation record. A switch log saying "rolled
out in week 9" without a timestamp cannot tell you whether Monday or Wednesday was the first treated
shift, which is exactly what the washout depends on.

### 8.2 Time and metric standards

| Standard | Governs | Why it matters here | Grade |
|---|---|---|---|
| ISO 8601 | Date, time, duration, interval representation | Block boundaries must be unambiguous; `2026-W14` and `2026-04-06/2026-04-19` are not the same object | B |
| ISO 8601 week numbering | Week-of-year definition, Monday start | A differently defined "week 1" shifts every block boundary by up to six days | B |
| IANA time zone database | Zone identifiers and daylight-saving rules | A block spanning a clock change contains 167 or 169 hours, so a wall-clock UPH denominator is wrong twice a year | B |
| ISO 3534-1 | Statistical vocabulary and symbols | The read-out should not redefine "confidence interval", "significance", or "power" | B |
| ASCM SCOR Digital Standard (SCOR DS) | Reference processes and metric definitions | Shared vocabulary for perfect order and fulfilment cycle time; does not fix a local productivity denominator | B |
| ISO 22400 | Key performance indicators for manufacturing operations | Fixes formulas for availability, performance, and quality where the outcome is an equipment metric such as overall equipment effectiveness (OEE) | B |
| 4-4-5 retail calendar (National Retail Federation convention) | Retail period structure | Retail reporting often uses periods that do not align with ISO weeks; declare the calendar system | C |

Two rules follow. **Store timestamps in UTC and convert at analysis time**, because conversion rules
change when a jurisdiction changes its daylight-saving regime. And **declare the calendar system in
the pre-analysis plan**, because a 4-4-5 calendar and an ISO week calendar disagree about which days
belong to which period roughly four times a year.

The binding definition in an operational pilot is almost never the international standard. It is the
company's own metric dictionary, which is usually implicit. The most valuable artefact you can produce
before launch is a one-page metric contract: numerator, denominator, population, time bucket, and the
treatment of exceptions. Section 6.9 shows why — the UPH denominator excluded rework handled by a
separate team, and that exclusion is the difference between a 6.5% gain and a 5.4% gain.

### 8.3 Employment and monitoring constraints

A switchback in a DC measures the productivity of named people, which is a different legal object
from a test that measures orders.

- Regulation (EU) 2016/679 (the General Data Protection Regulation, GDPR) Article 88 permits Member
  States to provide more specific rules for personal data in the employment context. National
  implementations differ, and several require consultation with employee representatives before
  monitoring measures are introduced.
- In Germany the Works Constitution Act (*Betriebsverfassungsgesetz*) gives works councils
  co-determination rights over technical devices capable of monitoring employee behaviour or
  performance. A pick-rate pilot in a German DC may require consultation before launch, not after.
- In several jurisdictions, productivity monitoring of individuals is lawful only where necessary,
  proportionate, and disclosed.

This is educational content, not legal advice, and the current text and any sector rules should be
verified before relying on it. The design consequence is concrete: **decide before launch whether the
analysis needs individual-level data at all.** For most switchbacks it does not. The unit of analysis
is the block, and block-level aggregates with a minimum cell size satisfy both the statistics and most
of the privacy concern.

### 8.4 What is convention rather than requirement

- **"A two-week washout."** No standard says two weeks. The length is a property of the process and
  should be measured from the WIP tail and the learning ramp.
- **"Pilots run four to six weeks."** A duration chosen to fit a quarter, not to reach steady state or
  give enough blocks.
- **"Alternate weekly."** The most common schedule in practice and the worst available, for the
  aliasing reason in Step 12.
- **"The Hawthorne effect explains it."** The evidence (section 9.6) does not support a general, large
  effect. Invoking it usually substitutes for measuring selection, attention, and slack.
- **"p below 0.05 means it worked."** With a resolution floor of 0.029 that is achievable but not
  comfortably, and the floor belongs in the report.

An analyst who can separate a requirement from a convention can challenge a convention. One who cannot
will treat an inherited parameter as a physical constant.

### 8.5 The clinical ancestor, and why it does not bind

Switchbacks descend from the clinical crossover trial, where the methodology is older and stricter.
Two documents are worth knowing, and neither is binding on a supply chain pilot.

**ICH E9, *Statistical Principles for Clinical Trials***, and its E9(R1) addendum on estimands, set
the regulatory expectation for design, analysis, and pre-specified estimands. The concepts transfer —
define the estimand before you analyse. The obligations do not.

**The CONSORT extension for stepped-wedge cluster randomised trials** (Hemming and colleagues, *BMJ*,
2018) sets reporting expectations including the adoption schedule, period structure, and handling of
time effects. It is the best available checklist for a stepped-wedge supply chain pilot, and adopting
it voluntarily costs nothing.

> **Note:** The clinical crossover literature is also the best source of evidence on how badly
> carryover can go wrong, and section 9.4 uses it for that. Transfer the methodology and the warnings;
> do not transfer the regulatory framework or the sample sizes.

## 9. Industry practice and case evidence

The public evidence splits into a rigorous design literature, a substantial industry practice
literature from delivery marketplaces, a set of parallel-design operations studies showing when a
switchback is unnecessary, and a clinical literature on carryover that supplies the warnings.

### 9.1 The design literature

**Bojinov, Simchi-Levi, and Zhao**, "Design and Analysis of Switchback Experiments," *Management
Science* 69(7), 2023 (online 2022). The paper formalises the switchback as a minimax design problem
over randomisation points and probabilities under an m-order carryover, and proves three results that
matter operationally. Grade A.

**Fair coin flipping is optimal.** The optimal randomisation probability is q = 1/2 everywhere, which
is reassuring: the design does not need a clever allocation ratio.

**Randomise roughly once every m periods.** When carryover lasts m periods, the optimal randomisation
points occur once every m periods, with first and last epochs of length 2m, because a boundary epoch
carries carryover from one side only and therefore carries less information. This is the formal
version of Step 7's instruction to measure the washout rather than assume it.

**The gain over naive per-period randomisation is real and modest.** The authors report roughly
28–32% reduction in risk in their simulations at T = 120 periods and m = 2. That is meaningful and it
is smaller than the gain from getting the block structure right in the first place. Schedule
optimisation is a second-order refinement on top of a correct design, not a substitute for one.

### 9.2 Industry practice: delivery marketplaces

DoorDash published the most detailed public account, in engineering posts between 2018 and 2022. The
content is company-authored and its quantitative claims come from the company's own simulations, so
it is Grade C. It remains the most useful public description of running these designs at scale.

| Practice | What the company reports | What transfers |
|---|---|---|
| Unit definition | Randomises over region-by-time-window cells, roughly 30-minute windows at city level | The unit is the smallest cell containing the shared resource |
| Why not order-level | Treatment and control share the same courier fleet, so an order-level test measures interference | The shared-resource test in Step 2 |
| Window length | Tuned by running A/A tests at several granularities and taking the shortest window with no spurious effect | Empirical calibration of block length is a real technique |
| Analysis | Averages to the time-region cell with cluster-robust variance; a later post reports delivery-level t-tests at roughly 62% false positives in simulation against about 0.03 for a multilevel model | Transaction-level analysis of a cell-level design is the largest single source of false positives |
| Design trilemma | Switchback has low interference and cannot capture learning; unit-level A/B has high power, captures learning, and severe interference; market-level A/B has low interference and low power | The choice is a trade-off among failure modes, not a search for the best design |

Two caveats. The 62% figure is a simulation result under the company's own data-generating process and
should not be quoted as a property of delivery data generally. And a 30-minute window suits a dispatch
algorithm whose carryover dissipates in minutes; the same window applied to a slotting change would be
nonsense. The transferable content is the reasoning, not the parameters.

### 9.3 Parallel-design comparators: when not to switchback

**Gap Inc. scheduling study.** Kesavan, Lambert, Williams, and Pendem, "Doing Well by Doing Good:
Improving Retail Store Performance with Responsible Scheduling Practices at the Gap, Inc.",
*Management Science* 68(11), 2022. A randomised field experiment across 28 stores in San Francisco and
Chicago over nine months, November 2015 to August 2016. The intervention changed scheduling practices;
the study reports a 5.1% increase in store productivity, decomposed as roughly +3.3% sales and −1.8%
labour. Grade A. The relevance: 28 stores is enough for a parallel cluster design, so a switchback
would have cost power and disrupted operations for nothing.

**Alibaba/Cainiao packing study.** Sun, Van Mieghem, Zhang, and Hu, "Predicting Human Discretion to
Adjust Algorithmic Prescription: A Large-Scale Field Experiment in Warehouse Operations," *Management
Science*, 2022. A randomised experiment with 757 workers and 782,360 packages between 27 August and 9
September 2018, testing a bin-packing algorithm built around workers' tendency to switch to larger
bins. Deviation probability fell from 30.1% to 24.5%, with packing time improving by roughly 4.5% on
targeted packages. Grade A. The relevance is different: it shows that **inside one warehouse,
worker-level randomisation may be available.** Where the algorithm is applied per task rather than per
building, the shared-resource argument fails and a parallel design at worker level beats a switchback.

**Zoned storage thesis.** A 2025 MIT thesis reports an A/B quasi-experiment that split a warehouse into
control and treatment sections during peak season to test class-based scattered storage, and found no
improvement in picking productivity. Grade C: institutional research, single site, not peer-reviewed.
Included because negative operational results are rarely published.

### 9.4 Carryover: what the clinical literature establishes

| Finding | Source | Implication for a switchback |
|---|---|---|
| Differential carryover is aliased with the treatment-by-period interaction and cannot be separated from it in a two-period design | Ludwig, Willett, and Putt, *BMJ*, 2025 | A two-block on-off comparison cannot distinguish a treatment effect from a carryover effect |
| Statistical tests for carryover have low power, so a non-significant test does not rule it out | Ludwig et al., 2025; Wellek and Blettner, *Deutsches Ärzteblatt International*, 2012 | Do not "test for carryover and proceed if p > 0.05"; prevent it by design |
| Of 40 recent dietary crossover trials, 24 were at high risk of bias and 9 omitted washout entirely | Ludwig et al., 2025 | The base rate of this error in a mature literature is high; assume it is higher in operations |
| The standard two-period design cannot estimate the treatment contrast under unequal carryover; designs with three or more periods can | Laska, Meisner, and Kushner, *Biometrics*, 1983 | More blocks is the structural fix, not a better estimator |
| Washout length should be justified mechanistically; for drugs a common heuristic is three to four elimination half-lives | Lim and In, *Korean Journal of Anesthesiology*, 2021 | The operations analogue is to measure the WIP tail and learning ramp rather than adopt a fixed number of days |

The transferable conclusion is uncomfortable: **carryover cannot be tested away, only designed away.**
A pilot that runs a carryover test, finds p > 0.05, and proceeds has learned almost nothing, because
the test has low power precisely where carryover is moderate and damaging.

### 9.5 Stepped wedges: what the methodological reviews find

Stepped-wedge designs confound treatment with calendar time by construction, so period effects must be
modelled (Hughes, Granston, and Heagerty, *Contemporary Clinical Trials*, 2015). In a review of
published trials, roughly a quarter did not account for time effects and roughly a fifth did not
account for intracluster correlation (Nevins and colleagues, *Clinical Trials*, 2023/24, covering
trials published 2016–2022). With fewer than about 40 clusters, standard mixed-model and generalised
estimating equation intervals are anti-conservative without small-sample corrections, and only around
6% of reviewed trials with fewer than 40 clusters used appropriate methods (Nevins et al.; Varghese
and colleagues, *Journal of Clinical Epidemiology*, 2024). Stepped wedges are also frequently chosen
for logistical or ethical reasons rather than statistical ones, and the stated rationale is often weak
(Varghese et al., 2024). All Grade A.

The practical reading: a stepped wedge is right when the change is irreversible and the organisation
will not accept a permanent control group, and wrong when a parallel cluster trial is feasible,
because it gives up power for a political benefit the analysis does not recover.

### 9.6 The Hawthorne evidence, honestly stated

**McCambridge, Witton, and Elbourne**, "Systematic review of the Hawthorne effect: New concepts are
needed to study research participation effects," *Journal of Clinical Epidemiology* 67(3), 2014.
Nineteen studies with quantitative behavioural outcomes met inclusion: eight randomised trials, five
quasi-experimental, six observational. Twelve found a significant effect, but heterogeneity was extreme
(I² = 93%). Pooled odds ratios were 1.06 (95% CI 0.98–1.14) for randomised trials, 1.07 (0.99–1.17)
for quasi-experimental studies, 1.21 (1.03–1.41) for studies of being observed, and 1.17 (1.06–1.30)
overall. The authors conclude that research participation effects exist, are highly contingent on task
and context, and that there is "no single Hawthorne effect"; they recommend abandoning the unqualified
term. Grade A.

**Levitt and List**, "Was There Really a Hawthorne Effect at the Hawthorne Plant? An Analysis of the
Original Illumination Experiments," *American Economic Journal: Applied Economics* 3(1), 2011.
Recovering the original illumination data, the authors find the classic story — productivity rising
with every lighting change, including dimming — to be "entirely fictional". They report only weak,
subtle evidence: output roughly 3–4% higher during experimentation, and greater responsiveness to
artificial than to natural light. Grade A.

Three consequences. **Do not attribute an unexplained uplift to "the Hawthorne effect"** — the
evidence supports a small, context-dependent effect, not a large general one, and an unexplained 9% is
more likely selection, attention, or slack. **The relevant order of magnitude is 3–4%, not 30%.** And
**observation effects decay, and the decay is measurable** — a pilot showing a large effect in weeks
1–2 and a small one in weeks 7–8 has a decay curve, and the curve is the evidence, not the label.

### 9.7 Pilot-to-scale attenuation, and what is not evidenced

A 2026 CEP/LSE discussion paper (Calle, Gortazar, Hernandez-de-Benito, Hupkau, Molina-Millán, and
Roldán-Monés, "From pilot to policy: experimental evidence from scaling online tutoring", Discussion
Paper 2176) reports a pilot effect of +0.26 standard deviations and a scale-up effect at 3.6 times the
size of +0.11, roughly 30–42% of the pilot effect, attributed to tutor selection and loss of founder
oversight. Grade B: a working paper, in education, not supply chain. The transferable lesson is the
*structure* of the attenuation, not the magnitude — selection of the delivery agent, loss of the
leader's attention, and the difference between an exception granted to a pilot and a rule applied to
everyone all exist in a DC rollout.

There is very little peer-reviewed public evidence on switchbacks for supply chain levers
specifically: slotting, pick-path, wave release, dock scheduling, replenishment cadence, carrier
allocation, delivery-promise configuration. The methods transfer because the statistical structure is
the same; the variance magnitudes do not, because operational outcomes are driven by different
mechanisms and worse data. There is no public benchmark of bias for operational switchbacks and almost
no published long-run effects — which is why section 6.8 reports the carryover path rather than a
single number.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Strict alternation | The schedule reads C, T, C, T, C, T, C, T | Treatment collinear with every two-block calendar cycle | Effect inseparable from the promotional or pay cycle | Randomise the order; correlate against every known cycle |
| Block fixed effects in a single-unit design | "Week fixed effects" in the methods, one site | Treatment constant within block, so dummies absorb it | Coefficient arbitrary; model not identified | Treatment plus a smooth trend, or a design-based test |
| Washout omitted | No gap in the series at switch dates | Previous condition's WIP and learning tail carried forward | Effect diluted or reversed; variance understated | Discard a stated number of days at the start of each block |
| Washout too short | First two measured days differ in the previous condition's direction | Learning ramp longer than the discarded window | Estimate is a weighted average of the effect and its residue | Compare first days with the rest; lengthen and re-analyse |
| The 4-week win | Headline compares final weeks with the pre-launch baseline | Trend and season inside both numbers | Effect overstated, often by a factor of two | Trend-adjusted counterfactual; quote the block average |
| Position-unbalanced schedule | All treated blocks in the second half, counts still equal | Treatment correlated with block index | Naive contrast biased by trend times index gap | Report mean block index per condition; require gap below one |
| Analysis at transaction level | Millions of rows, eight blocks, one site | Rows treated as independent | Standard error too small by the root of rows per block | Aggregate to unit-by-block first; state the row count |
| Enthusiastic site selection | The site volunteered; it is the best or the worst performer | Selection on outcome or on motivation | Pilot effect is mostly selection | Randomise among eligible sites; define eligibility first |
| Concurrent change in the window | A layout, release, or promotion appears in the dates | Two causes, one window | Effect attributed to the wrong cause, either direction | Concurrent-change log from day one |
| Anticipation | The last control block before a switch is unusually good | Operators prepare for a known change | Control period contaminated at its end | Randomise switch timing where possible; check pre-switch blocks |
| Switch not executed | Switch log empty or reconstructed | Drift, workarounds, partial application | Dilution; a null result is uninterpretable | Log every switch; measure compliance |
| Guardrail omitted | The read-out reports throughput and not quality | The primary metric is a ratio that can improve while quality falls | Net benefit overstated; rollout imports a quality problem | Pre-register guardrails; report the rework-adjusted outcome |
| Season confound | The pilot runs through peak or a holiday period | Season confounded with the window | Effect is a seasonal artefact | Check the calendar before launch |
| Mean reported, path omitted | One number for the whole pilot | Decay, reversal, or one anomalous block averaged away | A temporary effect booked as durable | Report the block-by-block path |
| Resolution floor undeclared | "p = 0.06, not significant" with eight blocks | The schedule space cannot produce a smaller p | A design limitation reported as an empirical finding | Compute `2/C(2K,K)` before launch and report it |

Two of these deserve a note. **Strict alternation** is the failure that looks most like competence: the
team has clearly thought about balance, and the schedule is balanced in count. What destroys it is the
correlation with calendar cycles, and that is invisible unless someone lists the cycles and computes
it. **Block fixed effects in a single-unit design** is the failure a statistician is most likely to
commit, because it is the correct specification in the multi-unit case and the software does not
complain; the tell is the degrees of freedom, and any output from a model with eight block dummies and
one treatment indicator should be discarded.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Carryover is absorbed by the washout | The block mean must reflect only the current condition | Estimate is a weighted average of the effect and its residue | Compare first measured days with the rest |
| No state dependence between blocks | Blocks must be comparable draws | A block inherits the previous block's backlog or inventory position | Add the lagged outcome or state variable as a covariate |
| The trend is smooth over the window | A trend term must capture secular drift | A step, break, or curve is absorbed into the treatment estimate | Plot the series; add a quadratic; look for breaks |
| No calendar cycle correlates with the schedule | Treatment must be separable from season and cycle | Effect inseparable from the cycle | Correlate the treatment indicator with every known cycle |
| The schedule was drawn as recorded | Inference over schedules requires the draw actually used | The reference distribution is wrong; p-values optimistic | Keep the seed, constraint, and timestamp |
| Switches were executed as assigned | The estimate must be of the intervention, not the intent | Dilution; a null result is uninterpretable | Switch log with timestamps and compliance share |
| The metric definition is constant across blocks | Block means must be comparable | A definition change looks like an effect, either direction | Freeze the metric contract; audit mid-window |
| The block is long enough to reach steady state | The measured portion must represent the condition | The mean is dominated by the transient | Compare first days with the rest; lengthen blocks |
| The unit represents rollout units | The estimate must inform a rollout decision | Effect does not transfer; the pilot was a case study | Compare the pilot unit with the rollout population |
| No anticipation before switches | Control blocks must be untreated | The control period is contaminated at its end | Check pre-switch blocks against the control trend |

Two limits sit outside the table. **A switchback measures a temporary state, and rollouts create
permanent ones.** The treated blocks were temporary, publicly time-limited, and staffed with the
knowledge that they would end. A permanent policy changes hiring, training, capital investment, and
the organisation's beliefs. There is no statistical correction for this, only a phased rollout with a
holdout.

**A single-site switchback cannot resolve small effects, and most operational effects are small.**
Section 6.7 shows sixteen weeks resolving a 6% effect at p = 0.057 against a floor of 0.029. A 2%
effect would be undetectable at any block count that fits inside a year. The honest response is to run
the design where the effect is large — a bold change, a coarse outcome, a high-variance process — or
across many units, not to run it at one site and interpret the null.

## 12. Exercises

**E1. The shared-resource test.** A retailer wants to test a new shelf-replenishment cadence. Customers cross stores; store staff do not cross stores; three stores share a single delivery route.
- Deliverable: one page stating the unit, the shared resources, and whether a switchback is forced.
- Acceptance: names at least three shared resources, states for each whether it crosses the candidate unit boundary, and concludes with a design recommendation and one rejected alternative.
- Time: 30 minutes.

**E2. Schedule construction.** Build an eight-block schedule for a pick-path change in one DC, with a five-day washout, under the constraint that no more than three treated blocks appear consecutively.
- Deliverable: the block table with dates, condition, washout span, and the seed or method used to draw the order.
- Acceptance: exactly four treated and four control blocks; mean block index in each condition differs by under one; the constraint is stated and satisfied; washout is marked on every block, including blocks following a control period.
- Time: 45 minutes.

**E3. Recompute the example.** Using the data in section 7, compute the naive contrast, the trend-adjusted estimate, and the standard error by hand.
- Deliverable: the arithmetic, to two decimal places, with the formula written out.
- Acceptance: naive 5.53 UPH, adjusted 4.92 UPH, trend 0.612 UPH per block, residual SD 0.297, SE 0.253, t 19.4 (UPH = units per hour).
- Time: 40 minutes.

**E4. Read a pilot that won.** A vendor reports that a client achieved a 14% throughput gain in a four-week single-site pilot with no control group. Write the four questions you ask before believing it.
- Deliverable: four questions, each with the specific number or artefact that would answer it.
- Acceptance: covers baseline construction, concurrent change, metric denominator, and carryover or decay; each question names the document to request.
- Time: 30 minutes.

**E5. Carryover measurement.** You have daily output for a ten-week switchback with weekly blocks and no washout.
- Deliverable: a short analysis plan for estimating the decay of carryover, and the revised design it implies.
- Acceptance: states the regression with day-since-switch terms, explains how to separate the ramp from the effect using within-block day position, and proposes a washout length in days with a stated basis.
- Time: 50 minutes.

**E6. Resolution floor.** You are offered eight, ten, or twelve blocks by three different planners.
- Deliverable: a table of the smallest achievable two-sided p for each, and a recommendation.
- Acceptance: 0.029, 0.008, 0.0022 with the combinatorics shown; the recommendation states the block cost in calendar weeks and the operational disruption.
- Time: 25 minutes.

**E7. Critique a stepped wedge.** A supply chain team proposes a stepped-wedge rollout across six DCs over six months to avoid having a permanent control group.
- Deliverable: a one-page critique.
- Acceptance: identifies the six-cluster small-sample problem, states whether period effects are modelled, notes the direction of the bias if they are not, and gives a concrete alternative with its cost.
- Time: 45 minutes.

**E8. Design an on-off-on reversal.** A carrier surcharge changes the allocation of orders between two lanes.
- Deliverable: a full design document following the Phase A to Phase G structure in section 5.
- Acceptance: includes the estimand, the schedule with dates, the washout, the blocking variables, the primary and two guardrail metrics, the pre-registered decision rule, and the resolution floor.
- Time: 90 minutes.

## 13. Self-check answers

**A1.** The forced design arises when the lever is applied at the level of a resource that both treated and control observations draw on. Three shared resources: the delivery route (shared across three stores, so a store-level arm is contaminated), store staff and their learning, and the replenishment system configuration if it is set per region. If staff and route are shared, neither customer-level nor store-level assignment is clean, and a switchback over time inside each store is the remaining design. The rejected alternative should be a parallel store-level design with the shared route held constant across arms or the route randomised.

**A2.** A valid schedule needs four properties: equal or near-equal counts, no treatment-calendar cycle correlation, mean block index balance, and execution feasibility. A draw satisfying the constraint might be C, T, T, C, C, C, T, T (mean index treated 5.0, control 4.0, gap 1.0, which is at the boundary) or better T, C, C, T, T, C, T, C (treated mean 4.5, control 4.5, gap 0). The second is preferable. The constraint matters because long runs make the design fragile to a single concurrent event: if a machine fails in week 6, three treated blocks are contaminated at once.

**A3.** Naive: treated mean 111.53 minus control mean 106.00 equals 5.53 UPH over a mean index gap of 3.5 blocks. The trend term is 0.612 UPH per block, so the naive contrast absorbs 3.5 × 0.612 = 2.14 UPH of trend and understates the effect. Adjusted: coefficient 4.92 UPH, SE 0.253, t = 19.4, 95% CI 4.11 to 5.73 UPH. Note the direction: here the naive estimate is *lower* than the adjusted one because treated blocks sit late. In section 6 the opposite occurred, because the treated blocks sat early relative to control. The sign of the bias is set by the schedule, not by the design type.

**A4.** (i) What is the baseline, and is it the pre-launch period or a matched control period? Request the weekly series, not the headline. (ii) What else changed in those four weeks? Request the change-management log. (iii) What is the denominator — paid hours, productive hours, or scheduled hours — and does it include rework? Request the metric definition. (iv) What does the week-by-week path look like, and does the effect hold in weeks 3 and 4? Request the block table. A vendor who can answer all four has done the work; one who cannot has a number.

**A5.** Regress daily output on the treatment indicator, day-since-switch (capped at the block length), and an interaction, with day-of-week controls. The treatment coefficient estimates the effect at the switch; the day-since-switch coefficient estimates the ramp. A significant interaction means the effect is not stable across the block and the block mean is a weighted average of a transient and a steady-state value. If the ramp takes four days and blocks are seven, roughly a third of each block is unrepresentative; the fix is either a four-day washout with eleven-day blocks, or blocks long enough that the ramp is a small share.

**A6.** With K blocks and K/2 treated, the number of balanced schedules is C(2K, K): 70, 252, and 924. The smallest achievable two-sided p is 2 divided by that count: 0.0286, 0.0079, 0.0022. Eight blocks cannot produce a p below 0.0286, so a "significant at 0.05" claim from eight blocks is achievable but leaves almost no margin. Ten blocks is the practical minimum for a claim you expect to defend, and twelve is comfortable. The cost is calendar time: at two weeks per block, twelve blocks is twenty-four weeks, which crosses at least one seasonal boundary and must be checked against the promotional calendar.

**A7.** Six clusters is below the threshold at which standard cluster-robust and mixed-model intervals are reliable; small-sample corrections (Kenward–Roger, or bias-corrected cluster bootstrap) are required and are rarely applied. If period effects are not modelled, the treatment estimate absorbs whatever secular trend the rollout period contains, and because a stepped wedge gives treatment to more units over time, a rising trend biases the estimate upward and a falling trend biases it downward. The concrete alternative is a parallel cluster trial across six DCs with three per arm, which needs more units than exist, or a switchback within each DC if the change is reversible. If irreversibility is the true constraint, the honest option is a stepped wedge with a pre-specified trend model, small-sample corrected intervals, and a statement that six clusters cannot detect small effects.

**A8.** The rubric, not a single answer: the document passes if (1) the estimand names the population, the treatment contrast, and the outcome with its denominator; (2) the schedule is drawn, not alternated, with a seed and the constraint recorded; (3) washout is justified by the carrier's allocation cycle length, not by a default; (4) blocking variables are listed with the direction of expected gain; (5) two guardrails are named with thresholds; (6) the decision rule is stated before the data arrive, including what happens on a null; (7) the resolution floor is computed and the MDE stated. It fails if any of the seven is missing, and fails outright if the schedule is strictly alternating.

## 14. Interview questions

**Q1. When is a switchback the right design?**
- *Strong:* names a shared resource that both arms draw on — staff, equipment, a fleet, a dispatch algorithm — and adds that the design is chosen when neither customer nor geography can be randomised, with the cost of added variance stated.
- *Weak:* says "when you cannot do an A/B test" without naming the constraint, which suggests the design is a fallback rather than a response to a specific interference structure.

**Q2. Why not analyse at the transaction level?**
- *Strong:* because assignment is at the unit-by-block level, so transactions within a block are correlated; treats them as independent and the standard error shrinks by roughly the root of transactions per block, turning a 2-sigma result into a 20-sigma one.
- *Weak:* says "it inflates significance" without being able to say by how much or why, which usually means the person has the phrase but not the mechanism.

**Q3. Can you put period fixed effects in a single-unit switchback?**
- *Strong:* no — the treatment indicator is constant within block, so block dummies are perfectly collinear with it and the coefficient is arbitrary. In a multi-unit switchback with staggered schedules, period fixed effects are identified because treatment varies within a period across units, and then they are essential.
- *Weak:* says yes because "you always control for time", which is the most common statistical error in this design.

**Q4. How long should the washout be?**
- *Strong:* as long as the slower of the work-in-progress tail and the learning ramp, measured from a prior run or from the first days of the blocks in the data themselves, and reported with the basis.
- *Weak:* says "one week" or "two weeks" as a convention, without a mechanism. The convention answer is common and it is how carryover gets buried.

**Q5. A pilot shows a 9% gain in four weeks. What do you say?**
- *Strong:* asks for the path by week, the baseline construction, and the concurrent-change log; notes the plausible decomposition into novelty, attention, slack, and selection; and proposes a trend-adjusted estimate with a stated range.
- *Weak:* accepts or rejects the number outright. Both are wrong: the number may be real and temporary, or an artefact, and only the path distinguishes them.

**Q6. What is the resolution floor and how do you compute it?**
- *Strong:* with K blocks there are C(2K, K) balanced schedules; the smallest two-sided p is 2 divided by that count, so 8 blocks gives 0.029 and 12 gives 0.0022; it is a property of the design and must be declared before launch.
- *Weak:* has never heard of it, or believes the p-value can be arbitrarily small if the effect is large enough. It cannot: only the number of admissible schedules sets the floor.

**Q7. What is the Hawthorne effect, and how large is it?**
- *Strong:* reports that the systematic review finds no single effect, pooled odds ratios around 1.17 with high heterogeneity, and that the original illumination study's classic story does not survive re-analysis of its own data; treats a 3–4% order of magnitude as the reference.
- *Weak:* attributes any unexplained uplift to it, which makes it unfalsifiable and stops the search for selection, attention, and slack.

**Q8. Why is a stepped wedge risky with six clusters?**
- *Strong:* six is below the threshold where standard interval estimates are reliable; small-sample corrections are needed and rarely used; period effects must be modelled or treatment absorbs the trend; and the design gives up power relative to a parallel trial.
- *Weak:* treats the stepped wedge as strictly better because everyone eventually gets the treatment, which is a logistical advantage, not a statistical one.

**Q9. How do you design away carryover rather than test for it?**
- *Strong:* randomise the switch points rather than maximising them — roughly once every m periods when carryover lasts m — lengthen the boundary epochs, discard a washout, and use more blocks; notes that carryover tests have low power and a non-significant result means nothing.
- *Weak:* proposes testing for carryover and proceeding if p > 0.05, which is the standard error in the clinical literature and the reason washout is often omitted.

**Q10. What would make you abandon a switchback for a parallel design?**
- *Strong:* evidence that the lever is applied per task or per worker rather than per shared resource — the Alibaba packing case is the example — because then worker-level randomisation is available and the switchback's variance penalty and operational disruption buy nothing.
- *Weak:* says a parallel design is always better because it is simpler, which ignores the interference that forced the switchback in the first place.

## 15. Key terms

- **Switchback design** — a design in which treatment alternates over time within the same unit, so each unit is its own control.
- **Block** — one contiguous time period during which the unit holds a single condition.
- **Epoch** — a block, in the randomisation-design literature; the two words are used interchangeably here.
- **Washout** — time discarded at the start of a block so the previous condition's residue can clear.
- **Wash-in** — time discarded at the start of a block while the operator learns the new condition.
- **Carryover** — the effect of a previous condition persisting into the current block.
- **Carryover order (m)** — the number of periods over which a condition's effect persists.
- **Anticipation** — behaviour in a control block changed by knowledge of the coming switch.
- **Transient** — the short-lived response immediately after a switch.
- **Steady state** — the condition in which the response is stable within a block.
- **On-off-on (ABAB) reversal** — a switchback with a return to the original condition, used to demonstrate that the effect follows the condition.
- **Stepped wedge** — a design in which units switch from control to treatment in sequence, so treatment is confounded with calendar time.
- **Before-after with control** — a comparison of a treated unit's change against a concurrent untreated unit's change.
- **Interrupted time series (ITS)** — a before-after analysis modelling level and slope change at the intervention date.
- **Difference-in-differences (DiD)** — the difference of changes across treated and control units.
- **Period fixed effect** — a dummy for a time period; identified only when treatment varies within the period across units.
- **Cluster-robust standard error** — a standard error allowing correlation within a cluster, here the unit.
- **Intraclass correlation (ICC)** — the share of total variance that lies between clusters.
- **Autocorrelation** — correlation of a series with its own lagged values.
- **AR(1)** — a first-order autoregressive process; inflates variance by (1+ρ)/(1−ρ).
- **Randomisation inference** — inference from the distribution of the estimator over the schedules the design could have produced.
- **Resolution floor** — the smallest p-value the design can produce, `2/C(2K,K)` for a balanced schedule.
- **Minimum detectable effect (MDE)** — the smallest true effect the design can detect at a stated power.
- **Pilot effect** — the difference between the pilot-period response and the response after a permanent rollout.
- **Hawthorne effect** — a behavioural response to being observed; empirically small and highly context-dependent.
- **Guardrail metric** — a secondary outcome that must not degrade for a rollout to proceed.
- **Compliance** — the share of blocks in which the assigned condition was actually executed.

## 16. Sources

**Standards, regulators, and official bodies (Grade B)**

- International Organization for Standardization. ISO 8601, *Date and time — Representations for information interchange*. Defines date, time, and interval formats.
- International Organization for Standardization. ISO 3534-1, *Statistics — Vocabulary and symbols — Part 1: General statistical terms and terms used in probability*.
- International Organization for Standardization. ISO 22400, *Automation systems and integration — Key performance indicators (KPIs) for manufacturing operations management*.
- Association for Supply Chain Management. *SCOR Digital Standard (SCOR DS)*. Reference processes and metric definitions.
- Regulation (EU) 2016/679 (General Data Protection Regulation), Article 88, and national implementing provisions.
- *Betriebsverfassungsgesetz* (Works Constitution Act, Germany), co-determination provisions on monitoring devices.
- International Conference on Harmonisation. ICH E9, *Statistical Principles for Clinical Trials*, and E9(R1) addendum on estimands. Non-binding on supply chain pilots.

**Peer-reviewed methods and evidence (Grade A)**

- Bojinov, I., Simchi-Levi, D., and Zhao, J. "Design and Analysis of Switchback Experiments." *Management Science* 69(7), 2023. DOI 10.1287/mnsc.2022.4583.
- Kesavan, S., Lambert, S., Williams, J., and Pendem, P. "Doing Well by Doing Good: Improving Retail Store Performance with Responsible Scheduling Practices at the Gap, Inc." *Management Science* 68(11), 2022.
- Sun, J., Van Mieghem, J., Zhang, L., and Hu, Y. "Predicting Human Discretion to Adjust Algorithmic Prescription: A Large-Scale Field Experiment in Warehouse Operations." *Management Science*, 2022.
- Hemming, K., et al. "The revised CONSORT statement for reporting stepped-wedge cluster randomised trials: extension and elaboration." *BMJ*, 2018.
- Hughes, J., Granston, T., and Heagerty, P. "Current issues in the design and analysis of stepped wedge trials." *Contemporary Clinical Trials*, 2015.
- Nevins, P., et al. Reviews of time-effect and intracluster-correlation handling in stepped-wedge trials. *Clinical Trials*, 2023 and 2024.
- Varghese, C., et al. Review of methodological practice and stated rationale in stepped-wedge trials. *Journal of Clinical Epidemiology*, 2024.
- McCambridge, J., Witton, J., and Elbourne, D. "Systematic review of the Hawthorne effect: New concepts are needed to study research participation effects." *Journal of Clinical Epidemiology* 67(3), 2014.
- Levitt, S., and List, J. "Was There Really a Hawthorne Effect at the Hawthorne Plant? An Analysis of the Original Illumination Experiments." *American Economic Journal: Applied Economics* 3(1), 2011.
- Ludwig, V., Willett, W., and Putt, M. Analysis of carryover and washout in two-period crossover trials. *BMJ*, 2025.
- Wellek, S., and Blettner, M. "On the proper use of the crossover design in clinical trials." *Deutsches Ärzteblatt International*, 2012.
- Laska, E., Meisner, M., and Kushner, H. "Optimal crossover designs in the presence of carryover effects." *Biometrics*, 1983.
- Lim, C., and In, J. "Considerations for crossover design in clinical study." *Korean Journal of Anesthesiology*, 2021.

**Company cases and industry practice (Grade C)**

- DoorDash Engineering. Posts on switchback experimentation, 2018–2022. Company-authored; quantitative claims from company simulations.
- Massachusetts Institute of Technology. Thesis on zoned scattered storage, 2025. Single-site, not peer-reviewed.

**Working papers (Grade B)**

- Calle, E., Gortazar, L., Hernandez-de-Benito, M., Hupkau, C., Molina-Millán, T., and Roldán-Monés, A. "From pilot to policy: experimental evidence from scaling online tutoring." CEP Discussion Paper 2176, 2026. Education, not supply chain.

> **Note:** Titles, venues, and years are recorded as returned by retrieval and desk-checked. Where an article number or page range is not reproduced above, the title and venue are sufficient to locate it. No DOI is given unless verified.

## 17. Links

**This module**

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_geo-experiments.md`](04_geo-experiments.md)
- Next topic: [`06_power-mde.md`](06_power-mde.md)
- Prerequisites in this module: [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md), [`02_randomisation-units.md`](02_randomisation-units.md), [`03_ab-cluster-designs.md`](03_ab-cluster-designs.md)

**Elsewhere in Part I**

- Part index: [`../_index.md`](../_index.md)
- Experiments and quasi-experiments: [`../04_statistics_causal_inference/07_experiments-quasi.md`](../04_statistics_causal_inference/07_experiments-quasi.md)
- Difference-in-differences: [`../04_statistics_causal_inference/08_difference-in-differences.md`](../04_statistics_causal_inference/08_difference-in-differences.md)
- Autocorrelation and robust errors: [`../04_statistics_causal_inference/05_autocorrelation-robust-errors.md`](../04_statistics_causal_inference/05_autocorrelation-robust-errors.md)
- The three flows: [`../01_supply_chain_foundations/01_three-flows-one-network.md`](../01_supply_chain_foundations/01_three-flows-one-network.md)

**Course level**

- Module survey: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Pricing and promotions: [`../../../course/16_pricing_promotions_revenue_management.md`](../../../course/16_pricing_promotions_revenue_management.md)
- Domain case library: [`../../../course/28_domain_case_library.md`](../../../course/28_domain_case_library.md)
- Treasury and financial supply chain: [`../../../course/42_treasury_credit_financial_supply_chain.md`](../../../course/42_treasury_credit_financial_supply_chain.md)
