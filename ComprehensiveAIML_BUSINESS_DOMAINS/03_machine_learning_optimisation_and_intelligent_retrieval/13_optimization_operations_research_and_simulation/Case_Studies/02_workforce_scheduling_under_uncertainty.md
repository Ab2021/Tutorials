---
case_id: "CASE-2026-OR002"
case_title: "Workforce scheduling under uncertainty"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Customer operations workforce management"
primary_pattern: "two-stage stochastic staffing with scenario-weighted service penalty"
secondary_patterns: ["shift-cover matrix", "absence overlay via binomial availability", "value of stochastic solution decomposition"]
evidence_level: "illustrative"
source_basis: "constructed teaching instance; all demand, wage, penalty, absence and headcount figures are ILLUSTRATIVE"
as_of_date: "2026-11-10"
owner_role: "Workforce management analyst supporting the operations manager"
source_index: "INDEX.md"
review_trigger: "Revisit when shift rules, the absence rate, the penalty valuation method or the licensed-coverage rule change"
---

# Workforce scheduling under uncertainty

> **Evidence disclaimer.** This case is a teaching construction. No employer, contact centre, roster or
> measured labour outcome is asserted, and no figure below is a reported result of a real deployment.
> Demand vectors, the wage rate, the service penalty, the absence rate and the shift menu are
> ILLUSTRATIVE inputs chosen so that every printed quantity can be re-derived by hand from the inputs
> in section 5. Where the case describes a field measurement, it is described as a proposal with status
> `not-executed`. Arithmetic that is exact given the inputs is still labelled illustrative because the
> inputs are constructed, not observed.

## 1. The One-Line Framing

For one operating day split into six two-hour blocks, choose how many early shifts and how many late
shifts to publish so that expected payroll plus the cost of uncovered queue time is minimised, subject
to licensed-coverage, rest and approved-headcount rules, and compare that roster with republishing last
week's roster unchanged.

**Takeaway:** the roster that is cheapest to publish and the roster that covers the workload are
different rosters, and the gap between them is a shape problem before it is a headcount problem.

The framing fixes three things that staffing debates usually leave vague. It fixes the objective as a
single monetary expectation rather than a service-level target, because a target admits no comparison
between a roster that overspends by 200 dollars and one that misses a target by four agent-hours. It
fixes the decision variable as a pair of shift counts rather than a per-agent assignment, because the
assignment problem is solved after the counts are chosen and has a different failure structure. And it
fixes the comparator as last week's roster, because that is the plan the operation actually executed
and therefore the only plan whose cost is already being paid.

## 2. Context & Background

The modelled operation is a regional customer-operations centre that handles phone contacts, chat
contacts and a small queue of back-office claims. It opens at 08:00 and closes at 20:00, so there are
six two-hour blocks in the operating day. Work that arrives after the last block is either abandoned or
carried into the next morning, which is why the late blocks matter disproportionately: a shortage at
18:00 becomes tomorrow's backlog as well as tonight's abandoned contacts.

The centre runs two shift types. The early shift works 08:00 to 16:00 and therefore covers blocks zero
through three. The late shift works 12:00 to 20:00 and covers blocks two through five. Both are eight
paid hours at the same rate. There is no part-shift option in the current agreement, no split shift, and
no ability to send an agent home early without paying the full shift; those absences of flexibility are
what make the problem interesting, because with flexible durations the answer would be to staff each
block at its own requirement and the scenario spread would cost almost nothing.

Rostering is currently done by a supervisor who takes the previous week's roster, checks that the
published total is roughly the same as the forecast total, and republishes it with individual swaps for
approved leave. This is the practice the case compares against, and it should be read as competent under
its own constraints. The supervisor is accountable for a daily payroll number and for an abandonment
number, both reported after the fact, and has no instrument that says what the trade between them is
worth. Under that information set, stability is a defensible strategy: a roster that repeats is easy to
explain, easy to swap, and cannot be blamed for a change in shape.

Three organisational facts shape the analysis. First, the workforce is split by licence: a subset of
agents holds the claims-adjudication licence without which the specialist queue cannot be worked, so a
roster that fills the peak blocks with unlicensed agents is infeasible regardless of its cost. Second,
the collective agreement limits an agent to five rostered shifts in any seven days and limits the number
of shift-start changes within a week, which converts a daily shift count into a weekly headcount
requirement rather than a free daily variable. Third, the service penalty is not measured; it is
negotiated once a year as a notional value used for planning, and nobody in the room believes the
notional value to the nearest dollar. That last fact is the reason this case reports a break-even rate
and a sensitivity sweep rather than a single recommended roster.

The demand itself is genuinely uncertain at the horizon that matters. Publishing happens on Thursday for
the following week, so the roster is fixed roughly eleven days before the day it serves. The forecast
error at that horizon is not a residual around a point estimate; it is a spread whose shape changes
because marketing sends campaigns on some weeks and not others, and because a single enterprise customer
can add a five-percent step to a single afternoon. The three scenarios in section 5a are a discretisation
of that spread, built by the forecasting team from campaign calendars and from the dispersion of
same-weekday errors over the previous two quarters.

The operational consequence of a fixed roster against a spread of demand is asymmetric. Overstaffing
costs payroll, which is visible on the daily report and is the number the supervisor is measured on.
Understaffing costs abandonment, which is measured on a different report, at a different cadence, by a
different owner. The two costs are not on the same page, so the trade cannot be made where the decision
is made. Much of what follows is bookkeeping whose purpose is to force that trade onto one page with
explicit units.

## 3. The Business Question

The primary decision question is: for a fixed shift menu, wage rate, service-penalty valuation and
absence rate, which pair of shift counts minimises expected daily cost across the demand scenarios, and
how much of the resulting improvement over the repeat-last-week roster survives when the licensed,
rest and headcount constraints are enforced on top of the count decision?

Six subsidiary questions make that concrete enough to work with.

1. What is the cost of the roster the operation published last week, expressed on the same expected-cost
   basis, so that any alternative can be compared without an argument about baselines?
2. Which blocks actually bind? A roster can be short in the peak and long in the shoulders, and knowing
   which is the case tells the manager whether the lever is shift count or shift placement.
3. How much is one more shift worth, in expected cost, and at what penalty valuation does that answer
   change? A recommendation that flips at a plausible valuation is not a recommendation, it is a
   sensitivity.
4. What does absence do to a roster that is exactly tight on the forecast, and is inflating headcount by
   the absence rate a defensible correction or a systematic over-correction?
5. What headcount does the daily shift count imply once the five-shifts-per-week rule is applied, and does
   the cheaper roster also require fewer people?
6. What would have to be measured, and over what window, before anyone could claim that the published
   roster caused a change in abandonment rather than coincided with one?

The acceptance criteria are stated in decision terms. A roster recommendation is acceptable when it is
feasible against every protected constraint, when its cost is reported with the valuation it assumes,
when the valuation is accompanied by the rate at which the recommendation changes, and when the
comparator is the roster the operation would otherwise publish rather than an idealised plan. A cost
reduction that requires a valuation the business does not accept is not an improvement; it is a
different business.

Explicit non-goals keep the arithmetic load honest. The case does not assign named agents to shifts, does
not model intra-day re-rostering when the morning volume surprises, does not model skill mixing beyond a
single licensed-coverage constraint, and does not optimise annual leave. Each of those is a separate
decision with its own data; folding them in would make the scenario arithmetic untraceable.

## 4. Assumptions & Constraints

Every assumption is listed with its unit and the direction of error it introduces if it is wrong. An
assumption whose error direction is unknown cannot be compensated by slack, because slack has no sign to
compensate with.

| ID | Assumption | Unit | If wrong, error direction |
|---|---|---|---|
| A1 | The day has exactly six two-hour blocks, 08:00 to 20:00 | block | A seventh block adds demand with no shift able to cover it alone |
| A2 | Demand is expressed in on-queue agent-equivalents per block, already net of break relief | agents | If relief is not staggered, capacity is overstated by about 6% |
| A3 | Early shift covers blocks 0-3 and late covers blocks 2-5, no other pattern exists | block | Any added pattern strictly lowers cost, so the model overstates the optimum |
| A4 | A shift costs 208 USD regardless of demand realised that day | USD/shift | If agents can be released unpaid, fixed cost is overstated |
| A5 | An uncovered agent-hour costs 60 USD on average across queues | USD/agent-hour | The entire recommendation pivots on this number |
| A6 | Absence is independent across agents with probability 0.08 | probability | Correlated absence (weather, transport) fattens the tail beyond the binomial |
| A7 | Demand within a day is described by three scenarios with fixed probabilities | probability | Misspecified probabilities bias the chosen count, not the arithmetic |
| A8 | Blocks are treated as independently demanded conditional on the scenario | block | Intra-day correlation is already carried by the scenario, not by the block |
| A9 | Licensed agents are interchangeable across the specialist queue | headcount | Partial licensing would add a second cover matrix |
| A10 | Overtime is unavailable inside the planning window | hours | Availability of overtime lowers the penalty and favours leaner rosters |
| A11 | Demand in a block is served by agents present for the whole block | agents | Mid-block arrivals create partial capacity the model ignores |
| A12 | The service penalty is linear in uncovered hours | USD/hour | Convexity in waiting (queueing) makes the true penalty larger at high load |
| A13 | Weekly demand days are independent for headcount arithmetic | day | Serial correlation raises the peak-week headcount requirement |
| A14 | No agent may work more than five shifts in seven days | shifts/week | A looser rule reduces required headcount without changing daily cost |

Constraint A3 deserves emphasis because it is the assumption that creates the whole problem. With a
continuous choice of shift lengths the planner would staff each block at its requirement and the
scenario spread would be absorbed almost free of charge; the cost of uncertainty in this case is
entirely a cost of the discrete shift menu. A reviewer who wants to argue that the case overstates the
value of optimisation should attack A3 by proposing a four-hour evening shift, not by attacking the
scenario probabilities.

| ID | Constraint | Type | Consequence if violated |
|---|---|---|---|
| C1 | Coverage in every block is at least the realised demand, counting absent agents as unavailable | soft, priced | Penalty accrues at 60 USD per uncovered agent-hour |
| C2 | At least 14 present agents in block 3 hold the claims licence | hard | The specialist queue cannot legally be worked; plan infeasible |
| C3 | No agent exceeds five rostered shifts per seven days | hard | Roster breaches the agreement; publication blocked |
| C4 | No agent's shift start changes more than twice per week | hard | Same, and it constrains the weekly repair path |
| C5 | Total rostered shifts per day do not exceed the approved establishment | hard | Payroll exception requires director approval |
| C6 | Each shift includes a 30-minute break scheduled inside the shift | hard | Break must be relieved, which is what A2 abstracts |
| C7 | At most one shift type per agent per day | hard | Prevents split shifts the agreement forbids |
| C8 | Agents on leave are unavailable and are excluded from the pool before counts | hard | Counting them inflates available capacity |
| C9 | The published roster is immutable after Thursday 17:00 except by swap | policy | Late changes destroy the comparison the case relies on |

The distinction between C1 and the rest is the design pivot. C1 is priced and can be violated at a known
rate; C2 through C8 are hard and cannot be traded at any price. A model that treats all constraints as
soft will, given a large enough penalty, propose a roster that misses the licensed requirement, and that
roster is not expensive, it is illegal.

## 5. Data, Tools & Method

### 5a Data Specification

The unit of analysis is one (date, block, scenario) triple. The roster decision is made at the date
level; the cost is computed at the block level; the scenario is the exogenous state that selects the
demand vector. The grain statement matters because two common errors follow from getting it wrong:
summing block requirements into a daily hours figure and then staffing to the daily hours total produces
a roster that is short in the peak and over-staffed in the shoulders, and averaging demand across
scenarios before staffing produces a roster that is wrong in every scenario.

| Field | Meaning | Units | Grain | Available at publish time |
|---|---|---|---|---|
| date | Operating date | ISO date | day | yes |
| block | Two-hour operating block, 0-5 | index | block | yes |
| scenario_id | S1 low, S2 base, S3 high | category | scenario | yes |
| prob | Scenario probability from the forecast team | dimensionless, sums to 1 | scenario | yes |
| demand | Required on-queue agents for the whole block | agents | block x scenario | yes |
| shift_type | E early or L late | category | shift | yes |
| shift_cost | Fully loaded cost of one shift | USD | shift | yes |
| shift_blocks | Blocks covered by a shift type | block list | shift | yes |
| penalty_rate | Valuation of one uncovered agent-hour | USD/agent-hour | policy | yes |
| absence_rate | Probability an agent is absent | dimensionless | person-day | estimated, not known |
| licensed_pool | Agents holding the claims licence | headcount | day | yes |
| licensed_required | Licensed agents needed in block 3 | headcount | block | yes |
| max_shifts_week | Contractual shift cap per agent | shifts/week | week | yes |
| headcount_approved | Establishment ceiling | headcount | week | yes |
| realised_present | Agents who actually attended | headcount | block | no, post-hoc only |

The demand vectors are constructed from a workload level and a block-share curve, which is how the
forecasting team actually produces them and which makes them re-derivable. The base scenario carries
128 agent-hours of work; the shares of that work falling in the six blocks are 9.375, 14.0625, 21.875,
25.0, 18.75 and 10.9375 percent. Multiplying and dividing by the two hours per block reproduces the base
vector exactly: 128 x 0.09375 / 2 = 6, 128 x 0.140625 / 2 = 9, 128 x 0.21875 / 2 = 14, 128 x 0.25 / 2 =
16, 128 x 0.1875 / 2 = 12 and 128 x 0.109375 / 2 = 7.

| Scenario | Probability | B0 08-10 | B1 10-12 | B2 12-14 | B3 14-16 | B4 16-18 | B5 18-20 | Agent-hours |
|---|---|---|---|---|---|---|---|---|
| S1 low | 0.25 | 5 | 8 | 12 | 14 | 10 | 6 | 110 |
| S2 base | 0.50 | 6 | 9 | 14 | 16 | 12 | 7 | 128 |
| S3 high | 0.25 | 8 | 11 | 17 | 19 | 14 | 9 | 156 |
| Expected | 1.00 | 6.25 | 9.25 | 14.25 | 16.25 | 12.00 | 7.25 | 130.5 |

The low and high vectors are not the base vector scaled by a constant, and that is deliberate. Low
carries 110 agent-hours with shares 9.09/14.55/21.82/25.45/18.18/10.91 percent, which returns 5, 8, 12,
14, 10, 6 after division by two. High carries 156 agent-hours with shares 10.26/14.10/21.79/24.36/17.95/
11.54 percent, returning 8, 11, 17, 19, 14, 9. The high scenario is therefore both larger and later: its
afternoon share of 24.36 percent is lower than base, but its absolute afternoon demand is 19 agents
against 16, because the level rose faster than the shape shifted. A planner who scales the base vector
by 156/128 = 1.2188 and rounds would place 7, 11, 17, 20, 15, 9 and would book one extra agent in the
16-18 block and one fewer in the 14-16 block, which changes the late-shift requirement by one.

What is not available at decision time is the realised demand, the realised attendance, and the
realised contact mix. All three are known only after the day closes, and the published roster cannot
respond to any of them. Every metric in section 9 that mentions realised quantities is therefore a
post-hoc check on the decision, not an input to it.

### 5b Tools & Stack

The analysis needs no solver beyond exhaustive enumeration, because the decision space is two integers.
The tooling choice is recorded mainly to show that the sophistication budget was spent on validation
rather than on search.

| Layer | Tool class | Why this and not something heavier |
|---|---|---|
| Scenario store | versioned table with scenario_id and probability | Probabilities must be versioned with the vectors they weight |
| Demand builder | workload level x share curve, rounded at block level | Keeps the derivation auditable to two numbers |
| Cost model | spreadsheet-replicable formulas in a script | A roster costing that cannot be checked by hand will not be trusted |
| Optimiser | full enumeration over E and L | 26 x 26 evaluations, so optimality is proven not assumed |
| Absence overlay | binomial convolution per block | Exact for independent absence, and cheap enough for 24 blocks |
| Headcount conversion | integer division of weekly shifts by the cap | One line, and it is where most roster proposals fail |
| Reporting | expected-cost table with the penalty rate printed on it | A number without its valuation is not actionable |
| Audit | roster hash, scenario version, valuation version | Reproducibility without re-running the forecast |

Enumeration is a deliberate methodological choice with a cost: it only works because there are two shift
types. Adding a third type (a four-hour evening shift, for example) takes the space from 26 x 26 to 26 x
26 x 26 and exhaustive enumeration still finishes, but adding a fourth and a fifth makes it 11.9 million
evaluations and the enumeration stops being interactive. The migration path, if the shift menu grows, is
a small mixed-integer program in which the two-stage structure is modelled by scenario-indexed recourse
variables; the enumeration in this case is a proof of optimality for the reported numbers, not a claim
that enumeration is the right production method.

### 5c Method Summary

The method is a two-stage stochastic program written out in full, then solved by enumeration because the
first stage has two integer variables.

Stage one chooses non-negative integers E and L, the number of early and late shifts, subject to the
protected constraints. Stage two, for each scenario, computes the coverage implied by (E, L), takes the
shortfall against that scenario's demand, and prices it. The objective is the fixed payroll plus the
probability-weighted penalty. Because the recourse is not a decision but a consequence, the model has no
second-stage optimisation at all; the "recourse cost" is a deterministic function of the first-stage
choice and the scenario. That simplification is honest for this operation and would be dishonest for one
that could call in agency staff at short notice, where the recourse would be a real decision with its
own cost and its own lead time.

The validation ladder has four rungs and the case executes the first three. Rung one re-derives every
printed number from the inputs by hand (§6). Rung two compares the enumeration's optimum against a
hand-checked neighbourhood, which is possible because the cost surface is nearly convex in each
direction and the neighbourhood of the optimum is small. Rung three checks the constraint layer by
construction: the licensed-coverage and headcount constraints are applied after the cost optimum, and the
case reports how far the cost optimum is from feasibility. Rung four, a shadow roster run alongside the
live one, is not executed and is specified in §9.

The order of operations is a design decision worth stating. Cost is optimised first and constraints are
checked second, because doing it the other way round hides the cost of the constraints. The case can
therefore report both the unconstrained expected cost and the premium paid for legality, which is the
number a manager needs when the licensed rule is renegotiated.

### 5d Provenance, Quality, Exclusions, Leakage

Provenance is short because the inputs are constructed: the six-block day, two-shift menu, 208-dollar
shift cost, 60-dollar penalty and 8-percent absence rate are all teaching values. What is real is the
*structure*: the shift-cover matrix, the shape-versus-level decomposition of demand, and the way the
contractual shift cap converts daily counts into headcount. A practitioner can substitute their own
numbers into the same structure and get an answer to the same question.

| Quality dimension | Finding | Treatment |
|---|---|---|
| Demand completeness | Blocks are complete for all six; no partial-block demand | Modelled as full-block requirement |
| Scenario probability source | Forecast team judgement plus error dispersion | Treated as given, swept in §6 |
| Penalty valuation | Annual negotiation, not measured | Reported as a rate with a break-even, never as a fact |
| Absence rate | Single pooled rate across agents | Independence assumed, tail risk noted as understated |
| Licensing data | Roster of licence holders maintained separately from the roster | Joined on agent id, checked for stale leavers |
| Exclusions | Agents on long-term leave, trainees in certification, agency staff | Excluded from the pool before counts |
| Double counting | Break relief appears in A2 and C6 | One abstraction only: demand is net of relief |
| Leakage risk | Realised present count is known before the cost is reported | Never used as an input to the count decision |
| Leakage risk | Realised demand is available for the same date in the history table | Scenario build uses only data before the publish cutoff |
| Seasonality | Campaign weeks differ from ordinary weeks | Scenarios carry campaign probability, not an average week |

Two exclusions do real work and are stated rather than absorbed. Trainees are excluded from the licensed
pool even though they sit on the queue, because they cannot adjudicate claims, and counting them would
inflate the licensed coverage by roughly the trainee share and make C2 look satisfiable when it is not.
Agency staff are excluded from the headcount arithmetic in C5 but are available as a paid fallback at a
materially higher rate; the case mentions the fallback because a manager will ask about it, and prices it
in §6 as a comparison rather than as a decision variable.

Leakage deserves one more sentence because it is the failure that would silently flatter the result. The
history table contains realised demand for past dates at block grain, and it is tempting to build the
scenarios from that table by simply taking the realised distribution of daily totals. That distribution
is contaminated by the roster that was published: on days when the roster was short, contacts abandoned
before they were counted as demand, so the observed demand is censored downward precisely on the days
that would populate the high scenario. Building scenarios from observed served volume therefore
understates the high scenario, and the resulting roster would be leaner than the one the analysis
recommends. The correct source is offered or queued volume before abandonment, and the case assumes the
forecast team already uses that definition.

### 5e Reproducibility & Method-Selection Ledger

| Decision | Alternatives rejected | Reason | What would change the decision |
|---|---|---|---|
| Two-stage stochastic program | Deterministic staffing at the base forecast | Ignores the shape of the spread; costs 382 USD/day here | A penalty so small that the spread is irrelevant |
| Enumeration of E and L | Mixed-integer program | Space is 676 points; proof beats a solver log | A third or fourth shift type |
| Scenarios with fixed probabilities | Full empirical distribution | Only three published vectors exist | A per-day predictive distribution from the forecast model |
| Linear penalty in uncovered hours | Queueing delay cost | No calibrated delay function exists | A measured abandonment-versus-wait curve |
| Independent absence, binomial | Correlated absence model | No correlation data retained | Clustered absence observed in attendance data |
| Daily counts, weekly headcount | Full weekly roster optimisation | The daily decision is the one that gets made | If publication moved to a rolling daily cycle |
| Constraint check after optimisation | Constrained optimisation from the start | Separates cost of legality from cost of staffing | If the licensed rule is immovable and always binding |
| Expected cost as the objective | Chance constraint on service level | Chance constraints hide the price of the target | If a contractual service floor exists |

Reproducibility requires four artefacts to be pinned together: the scenario version, the probability
vector, the valuation version and the roster itself. The case stores all four under one hash because any
three of them without the fourth produce a number that cannot be defended in a review. The hash is the
cheapest part of the whole design and it is the part most often missing from workforce analytics, where
rosters are exported to spreadsheets and the scenario file is overwritten weekly.

## 6. Analysis Walkthrough

The walkthrough proceeds from the cheapest possible roster to the constrained recommendation, and every
intermediate number is recomputed rather than quoted.

The first thing to establish is the coverage implied by a pair of counts. An early shift covers blocks
zero to three, a late shift covers blocks two to five, and the two overlap in blocks two and three.

```text
FORMULA  coverage(E, L, b) = E for b in {0, 1};  E + L for b in {2, 3};  L for b in {4, 5}

WHERE      E, L are non-negative integer shift counts (shifts)
           b indexes the six two-hour blocks of the operating day
           coverage is in on-queue agents present for the whole block

CONSTRAINT E, L >= 0 and E + L <= establishment ceiling (C5)
           the matrix has one nonzero pattern per shift type, so coverage is linear in (E, L)

NUMERICAL EXAMPLE  E = 8, L = 9 gives coverage (8, 8, 17, 17, 9, 9)
                   base demand is (6, 9, 14, 16, 12, 7)
                   surplus (3, 0, 3, 1, 0, 2) in blocks 0, 2, 3, 5
                   shortfall (0, 1, 0, 0, 3, 0) in blocks 1 and 4

INTERPRETATION  sixteen of seventeen scheduled agent-blocks are useful in the base case, but the
                two blocks where the roster is short are not the two blocks where it is long.
                Coverage surplus in one block cannot be moved to another block, which is the
                whole reason a count-based roster can be simultaneously generous and deficient.
```

That asymmetry is the centre of the case. A roster is not described by its total hours; it is described
by six numbers, and two rosters with identical totals can differ by tens of agent-hours of waste.

The second derivation prices a roster across scenarios.

```text
FORMULA  EEV(E, L) = 208 (E + L) + 60 * 2 * SIGMA_s p_s * SUM_b max(0, d_sb - cov(E, L, b))

WHERE      208 USD is the fully loaded cost of one eight-hour shift
           60 USD per uncovered agent-hour, multiplied by 2 hours per block = 120 USD per block
           p_s is the scenario probability (0.25, 0.50, 0.25)
           d_sb is demand in agents for scenario s and block b
           cov(E, L, b) is the coverage from the first formula

CONSTRAINT the penalty is linear in uncovered hours and independent across blocks conditional on s

NUMERICAL EXAMPLE  E = 8, L = 9: fixed 208 x 17 = 3536 USD
                   shortfall blocks: S1 1, S2 4, S3 10
                   expected shortfall blocks = 0.25(1) + 0.50(4) + 0.25(10) = 0.25 + 2 + 2.5 = 4.75
                   expected uncovered hours = 2 x 4.75 = 9.5
                   penalty = 120 x 4.75 = 570 USD;  EEV = 3536 + 570 = 4106 USD

INTERPRETATION  the roster pays 3536 for capacity and 570 for the capacity it did not buy. The
                penalty is 13.9 percent of the total, which is small enough that a manager
                reading only the payroll line would see nothing wrong with a leaner roster.
```

Applying the same formula to the roster the operation would otherwise publish, E = 9 and L = 12, gives
fixed cost 208 x 21 = 4368 and coverage (9, 9, 21, 21, 12, 12). Against the three demand vectors the
shortfalls are 0, 0 and 4 blocks, so the expected penalty is 120 x 0.25 x 4 = 120 and the expected cost
is 4488. The repeat-last-week roster therefore costs 382 USD per day more than the enumerated optimum,
an 8.5 percent difference, and it achieves that despite having nineteen percent more paid hours. It is
worse in both directions at once, which is the signature of a shape error rather than a level error.

The third derivation is the marginal rule, which is what makes the recommendation explainable.

```text
FORMULA  add one shift of type t iff  RATE * SIGMA_{b in cover(t)} p(short in b) * 2 >= 208

WHERE      RATE is the service penalty in USD per uncovered agent-hour
           cover(t) is the four blocks that shift type t covers
           p(short in b) is the probability that block b is short under the current roster
           the left side is expected penalty avoided, the right side is the shift cost

CONSTRAINT the rule is exact only when the shift's four blocks are independent given the roster,
           because otherwise the same shortage event is avoided in more than one block

NUMERICAL EXAMPLE  from (8, 8) to (8, 9): expected uncovered hours fall from 13.0 to 9.5
                   avoided = 3.5 agent-hours, valued at 60 USD/h = 210 USD
                   shift cost 208 USD, so net gain 2 USD, and the EEV table shows 4108 -> 4106
                   from (8, 9) to (8, 10): 9.5 to 7.0 hours, avoided 2.5 h = 150 USD < 208 USD
                   net loss 58 USD, and the table shows 4106 -> 4164

INTERPRETATION  the break-even is 208 / 60 = 3.4667 avoided agent-hours per added shift. Any shift
                that avoids less than three and a half hours of expected shortage destroys value,
                and the optimum sits exactly where the marginal shift avoids 3.5 hours.
```

The rule also tells a manager what to argue about. The 3.4667-hour break-even depends on two numbers,
the wage and the penalty. The wage is a fact of the payroll system. The penalty is a negotiated
valuation. So the only number in the rule that can be wrong is the one that is not measured, and the
recommendation's exposure to it is stated in the sensitivity sweep below rather than buried.

| E | L | Shifts | Fixed USD | Short blocks S1 / S2 / S3 | E short blocks | E short hours | EEV USD |
|---|---|---|---|---|---|---|---|
| 5 | 6 | 11 | 2288 | 11 / 20 / 34 | 21.25 | 42.5 | 4838 |
| 6 | 6 | 12 | 2496 | 8 / 16 / 30 | 17.50 | 35.0 | 4596 |
| 6 | 7 | 13 | 2704 | 6 / 12 / 26 | 14.00 | 28.0 | 4384 |
| 7 | 7 | 14 | 2912 | 4 / 9 / 22 | 11.00 | 22.0 | 4232 |
| 6 | 8 | 14 | 2912 | 4 / 9 / 22 | 11.00 | 22.0 | 4232 |
| 7 | 8 | 15 | 3120 | 3 / 7 / 18 | 8.75 | 17.5 | 4170 |
| 8 | 8 | 16 | 3328 | 2 / 5 / 14 | 6.50 | 13.0 | 4108 |
| 8 | 9 | 17 | 3536 | 1 / 4 / 10 | 4.75 | 9.5 | 4106 |
| 8 | 10 | 18 | 3744 | 0 / 3 / 8 | 3.50 | 7.0 | 4164 |
| 9 | 9 | 18 | 3744 | 1 / 3 / 8 | 3.75 | 7.5 | 4194 |
| 9 | 11 | 19 | 4160 | 0 / 1 / 5 | 1.75 | 3.5 | 4370 |
| 9 | 12 | 21 | 4368 | 0 / 0 / 4 | 1.00 | 2.0 | 4488 |
| 11 | 14 | 25 | 5200 | 0 / 0 / 0 | 0.00 | 0.0 | 5200 |

Two rows tie at fourteen shifts: (7, 7) at 4232 and (6, 8) at 4232. The tie is not an arithmetic
accident; both produce coverage that misses the base peak in different places, and the demand curve is
flat enough around block three that moving a shift from early to late costs nothing. That flatness has a
practical consequence: a planner who prefers an earlier-weighted roster for reasons outside the model,
such as the queue's morning backlog, can take (7, 7) at no cost on these numbers. A tie that can be
resolved by an out-of-model preference is a feature, and it should be reported rather than hidden by
picking the first optimum the enumeration found.

The fourth derivation adds absence, because a roster that is exactly tight on the forecast is not tight
in reality.

```text
FORMULA  E[uncovered in b] = SIGMA_{k=0..n_b} C(n_b, k) a^k (1-a)^(n_b - k) * max(0, d_b - (n_b - k))

WHERE      n_b is the number of agents rostered in block b
           a is the absence probability (0.08), so (1 - a) = 0.92 is availability
           k is the number of absent agents; n_b - k are present
           the expectation replaces the deterministic shortfall max(0, d_b - n_b)

CONSTRAINT absence is independent across agents and across blocks; the block requirement is
           for whole-block presence, so a late arrival counts as absent for that block

NUMERICAL EXAMPLE  block 1 of plan (8, 9) has n = 8, d = 9
                   P(0 absent) = 0.92^8 = 0.5132, shortfall 1
                   P(1 absent) = 8 x 0.08 x 0.92^7 = 0.3569, shortfall 2
                   E[uncovered] = 0.5132(1) + 0.3569(2) + ... = 1.64 agent-blocks
                   across six blocks: 0.0235 + 1.6400 + 0.0526 + 0.6023 + 3.7200 + 0.0338 = 6.0722
                   at 120 USD per block, that is 728.67 USD, against 570 USD without absence

INTERPRETATION  absence raises the expected penalty by 27.8 percent and does not touch payroll,
                which is why rosters that look correct on paper generate overtime in practice.
```

The five-percent-ish effect is not uniform, and the block detail is more useful than the total. The
tightest exposure is block four, where nine late shifts face a base requirement of twelve: that block
carries 3.72 of the 6.07 expected shortfall blocks because it is short by three before anyone is absent,
so absence only makes an already-binding block worse. Block three, with seventeen rostered against
sixteen required, carries 0.60 expected shortfall blocks despite a nominal surplus of one, because the
surplus is a single agent and a single absence erases it.

The naive correction is to inflate headcount by the absence rate, and it is worth showing why that
over-corrects. The shortfall is a convex function of the present count, so expected shortfall falls
faster than linearly as n rises; but the cost of an extra agent is linear. Inflating the tight block from
seventeen to eighteen agents (17 / 0.92 = 18.48, so eighteen scheduled under-covers and nineteen would be
required by the naive rule) adds payroll every day to defend against a tail that occurs on a minority of
days, and the correct comparison is the marginal rule again with the absence-adjusted shortfall
distribution. The practical form of the rule is: raise the count where the block is already tight before
absence, not where it is comfortable.

A cleaner way to see the same point is the tail of the binomial on a full pool. With fourteen agents
rostered and 8 percent absence, the probability that all fourteen attend is 0.3112, the probability that
exactly one is missing is 0.3788, and the probability that two or more are missing is 0.3100. The
probability of three or more absences is 0.0958. So a plan that assumes the roster arrives in full is
correct on 31 percent of days, and a plan that assumes it can absorb one absence is correct on 69 percent.
Neither is a plan; both are descriptions of a distribution, and the roster has to be chosen against the
distribution.

| Rostered n | Required d | E[uncovered blocks] | P(short) | Penalty USD at 120/block |
|---|---|---|---|---|
| 6 | 6 | 0.0235 | 0.0298 | 2.82 |
| 8 | 6 | 0.0235 | 0.0211 | 2.82 |
| 8 | 9 | 1.6400 | 1.0000 | 196.80 |
| 9 | 7 | 0.0338 | 0.0298 | 4.06 |
| 9 | 12 | 3.7200 | 1.0000 | 446.40 |
| 17 | 14 | 0.0526 | 0.0419 | 6.31 |
| 17 | 16 | 0.6023 | 0.3995 | 72.28 |

The pattern in that table is the practical lesson about absence: it is nearly free where there is
surplus and expensive where there is none, and the expense is concentrated in the blocks that were
already the binding ones. This is why absence-adjusted rostering is not a percentage uplift. A percentage
uplift spreads cost evenly across blocks that do not have equal exposure.

The fifth derivation converts a daily count into a headcount, which is where most roster proposals die.

```text
FORMULA  headcount = CEIL(7 * (E + L) / max_shifts_week)

WHERE      E + L is the daily shift count
           7 is days per week; max_shifts_week is 5 under the agreement (C3)
           CEIL because a person cannot be fractional and cannot work a partial shift

CONSTRAINT each agent works at most five shifts in seven days, and shift starts change at most
           twice per week, so a five-shift agent works the same start pattern most days

NUMERICAL EXAMPLE  recommended (8, 9): 7 x 17 = 119 shifts/week; 119 / 5 = 23.8; headcount 24
                   repeat-last-week (9, 12): 7 x 21 = 147 shifts/week; 147 / 5 = 29.4; headcount 30
                   difference: 6 people, or 20 percent of the establishment

INTERPRETATION  the daily cost difference of 382 USD and the headcount difference of 6 people are
                the same fact seen at two horizons, and the headcount version is the one that
                survives into next quarter's budget.
```

That last step changes the conversation. A 382-dollar daily saving invites an argument about whether the
penalty valuation is right. A six-person reduction in the establishment requirement is either true or
false, and it is checkable against the roster of record in an afternoon. In practice the headcount
figures are the ones that get verified, and the cost figures are the ones that get quoted, which is why
both are reported here with their derivations attached.

The sixth check is the sensitivity sweep on the penalty rate, because the recommendation is a function of
a number nobody trusts.

| Penalty USD per uncovered agent-hour | Chosen (E, L) | Expected cost USD | Expected uncovered hours |
|---|---|---|---|
| 30 | 6, 7 | 3543.0 | 24.5 |
| 45 | 6, 8 | 3902.0 | 22.0 |
| 60 | 8, 9 | 4106.0 | 9.5 |
| 75 | 8, 9 | 4248.5 | 9.5 |
| 90 | 8, 10 | 4374.0 | 7.0 |
| 120 | 8, 11 | 4552.0 | 5.0 |
| 150 | 9, 12 | 4668.0 | 2.0 |
| 300 | 9, 12 | 4968.0 | 2.0 |

The sweep has a shape worth reading rather than a single point worth quoting. Below about 45 USD per
hour the model prefers the lean roster and accepts twenty-plus uncovered hours a day; between 60 and 150
it moves along a plateau where successive shifts buy diminishing shortfall; above 150 it saturates at the
twenty-one-shift roster, because beyond that point the only way to reduce shortage further is the
full-coverage plan at 5200 USD, whose premium is not justified by any penalty below roughly 300. The
recommendation at 60 USD per hour is therefore not fragile in the sense of flipping on a small error: the
same (8, 9) roster is optimal from 50 to 74 USD, which brackets any plausible valuation of an abandoned
contact in this operation.

The two plans that bracket the plateau can be compared directly, which gives the cleanest single number
in the case.

```text
FORMULA  crossover RATE* = (fixed(full) - fixed(lean)) / (E[h_lean] - E[h_full])

WHERE      lean = (8, 9) with fixed 3536 USD and 9.5 expected uncovered hours
           full = (9, 12) with fixed 4368 USD and 2.0 expected uncovered hours
           RATE* is the penalty at which the two plans cost the same

CONSTRAINT both plans must be feasible, and the comparison assumes the same scenario probabilities

NUMERICAL EXAMPLE  RATE* = (4368 - 3536) / (9.5 - 2.0) = 832 / 7.5 = 110.93 USD per hour
                   at 110.93 USD/h both cost 3536 + 110.93(9.5) = 4589.9 and 4368 + 110.93(2.0) = 4589.9

INTERPRETATION  the recommendation survives any penalty below about 111 USD per uncovered hour,
                and the sweep confirms the optimum has moved to a fuller roster by 120 USD/h.
```

The boundary where this crossover reasoning breaks is worth stating, because it is not obvious. The
formula assumes the two plans' uncovered hours differ by a constant, so their costs are straight lines in
the penalty rate. That holds when the scenario probabilities are fixed and the only thing changing is the
price. It fails if the penalty is not linear in uncovered hours, which it will not be in reality: a queue
with eighty percent occupancy and one with ninety-five percent occupancy do not differ by a constant
ratio of abandonment, so the marginal price of the last uncovered hour is higher than the average price.
Under a convex penalty the crossover moves down, toward the fuller roster, and the linear sweep reported
above is optimistic about how much service it is safe to forgo.

The final comparison is with the two options a manager will actually ask about: agency cover and
full-coverage planning.

| Option | Fixed USD | Expected penalty USD | Expected total USD | Note |
|---|---|---|---|---|
| Lean 13 shifts (6, 7) | 2704 | 1680 | 4384 | Cheapest publication, worst service |
| Recommended 17 shifts (8, 9) | 3536 | 570 | 4106 | Minimum expected cost at 60 USD/h |
| Full cover 25 shifts (11, 14) | 5200 | 0 | 5200 | Zero shortage in all three scenarios |
| 16 shifts (7, 9) plus 2 standby | 3328 + 52 retainer | 678.5 | 4058.5 | Assumes every call-up honoured |

The standby row is what makes the case operational rather than academic, and it is priced rather than
asserted. The arrangement keeps sixteen shifts on payroll and holds two agents on a 26-dollar daily
retainer, each available for a four-hour call-up at 182 dollars; a call-up covers one two-block half-day,
so the number of calls needed in a scenario is the shortfall in blocks divided by two, rounded up. Under
(7, 9) the coverage is (7, 7, 16, 16, 9, 9), the shortfalls are 2, 5 and 14 blocks, and the calls needed
are 1, 2 and 2. Expected cost is 3328 + 52 + [0.25 x 182 + 0.50 x (364 + 120) + 0.25 x (364 + 1200)] =
3380 + (45.5 + 242 + 391) = 3380 + 678.5 = 4058.5, which beats the recommended roster by 47.5 dollars a
day.

That 47.5 dollars is not free, and the two numbers that decide whether it is real are both missing from
the operation's records. The first is call-up reliability: if only nine in ten calls produce a present
agent, the plan costs 4100.5 and the advantage falls to 5.5; the break-even reliability is 0.8869, below
which the standby arrangement loses to simply publishing seventeen shifts. The second is the retainer
itself: the 47.5-dollar advantage is worth exactly 23.75 dollars per standby agent per day, so a retainer
above 49.75 destroys it entirely, and a retainer priced as an on-call premium of a half-shift (104
dollars) makes the arrangement 108.5 dollars worse than no standby at all. A manager who prefers the standby
option is therefore not making an arithmetic error, they are asserting a reliability and a retainer that
the case cannot verify on their behalf — which is why it is reported as a comparison and not as the
recommendation.

## 7. The Finding

The finding is a hierarchy of five statements, in the order a reviewer should check them.

**First, the roster in use is not merely expensive, it is mis-shaped.** Publishing 21 shifts (9 early, 12
late) at 4368 USD of payroll produces coverage (9, 9, 21, 21, 12, 12). Against base demand that is 20
surplus agent-blocks, or 40 paid agent-hours of idle capacity, which at the fully loaded rate is 1040 USD
of the day's 4368. Against the high scenario, the same roster is short in block one by two agents and in
block four by two agents, for four short blocks and 480 USD of expected penalty. The roster is therefore
simultaneously generous and deficient, and no amount of arguing about its total hours will fix that.

**Second, the level is right and the shape is wrong, and the shape error is worth 382 USD a day.** The
expected-cost gap between the roster of record and the enumerated optimum at 60 USD per uncovered hour is
4488 minus 4106, which is 8.5 percent of the expected cost. Seventeen shifts beat twenty-one. The
mechanism is not thrift; it is that two of the four "extra" late shifts were buying coverage in blocks
four and five, where base demand is 12 and 7, while the binding shortage was in block one, which no late
shift can touch.

**Third, the optimum is a plateau, not a point, and that is good news.** (8, 9) is optimal at 4106, (8, 8)
is within 2 USD, (7, 8) within 64 USD, and (7, 7) and (6, 8) tie at 4232. Over a 126-dollar band there
are five rosters within 3 percent of each other. This matters because it means the out-of-model
preferences, such as the morning backlog or a driver's preference for early starts, can be satisfied
inside the near-optimum, which is what makes an optimised roster adoptable. A unique optimum in a
unionised environment is an argument; a plateau is a negotiation.

**Fourth, absence is not a percentage uplift and treating it as one misallocates the cost.** Expected
shortfall under the recommended plan rises from 4.75 to 6.07 blocks when 8 percent absence is applied, a
27.8 percent increase in penalty and 12.1 uncovered agent-hours a day. Almost all of the increase sits in
blocks that were already binding. Inflating every block by 1/0.92 puts payroll into the blocks with
surplus and leaves the binding ones still binding, at a cost of one extra shift per day for a benefit
concentrated elsewhere.

**Fifth, the headcount consequence is the durable one.** Converting 17 shifts a day into a
five-shift-a-week establishment gives 24 people; converting 21 shifts gives 30. The six-person gap is the
same fact as the 382-dollar gap, expressed at a horizon where it can be audited, and it is also the
statement that most needs the licensed and rest constraints checked before it is believed.

The counter-finding needs equal prominence, because a competent reviewer will raise it first. The 382
dollars is a planning-time expected cost under three constructed scenarios and a negotiated penalty. It
is not a measured saving, no roster has been published on this basis, and the penalty term it depends on
has never been validated against an abandonment cost. The case therefore reports the number as
ILLUSTRATIVE and specifies, in §9, the shadow-roster measurement that would replace it. It also reports
the break-even valuation of 110.93 USD per uncovered hour, because a manager who believes the true cost
of a missed contact is above that should not adopt the lean plan no matter what the expected-cost
arithmetic says.

## 8. System/Solution Design

The design is a decision-support pipeline with six components and a small set of contracts. The
contracts exist because the failure modes in §12 are almost all boundary failures between components,
not optimisation failures inside them.

**Component 1 — Scenario builder with versioning.** Takes a workload level, a block-share curve and a
probability vector, and emits an immutable scenario set with a content hash. Contract: the scenario set is
read-only once published, and the roster records the hash it was built against. This single rule removes
the most common irreproducibility in workforce analytics, which is that the scenario file is regenerated
weekly while last week's roster still references it implicitly.

**Component 2 — Cost model.** A pure function from (E, L, scenario set, penalty rate) to an expected-cost
breakdown with fixed and penalty components reported separately. Contract: the penalty rate is a required
argument with no default. A cost model that defaults the penalty to a house value is how a service
valuation becomes invisible; making the rate mandatory forces it into every report and every slide.

**Component 3 — Enumerator with proof of optimality.** Enumerates all (E, L) pairs inside the
establishment ceiling, records the best value and the top ten alternatives. Contract: it returns a
ranked list, never a single answer, because the plateau is what the manager needs to negotiate with. A
component that returns only the argmin discards the information that makes the answer adoptable.

**Component 4 — Constraint gate.** Applies C2 through C8 as hard filters and reports which of them bind.
Contract: the gate returns a violation code list, and a non-empty list blocks publication even when the
cost is lowest. The codes are `X_LICENSED`, `X_SHIFTS_WEEK`, `X_START_CHANGES`, `X_ESTABLISHMENT`,
`X_BREAK_UNRELIEVED`, `X_LEAVE_CONFLICT`. Codes rather than prose, because a code can be counted and a
sentence cannot.

**Component 5 — Absence overlay.** Convolves the binomial availability distribution with block demand for
whatever roster the gate approved, and returns expected shortfall by block. Contract: it does not change
the chosen roster; it reports the exposure of the chosen roster. This separation matters because mixing
the overlay into the optimisation makes the objective a function of a nuisance parameter and the result
impossible to explain to a supervisor.

**Component 6 — Publication pack and audit record.** Emits the roster, the scenario hash, the penalty
version, the ranked alternatives, the constraint report and the absence exposure, in one artefact, and
logs it before release. The fallback is explicit: if any component fails, last week's roster is
republished with the failure code attached, so the operation always has a plan and the failure is
visible rather than absorbed.

The data contract for a roster decision is deliberately small.

```text
roster    = { date, early_count, late_count, shift_cost, roster_hash }
decision  = { roster, scenario_hash, penalty_version, expected_cost,
              fixed_cost, penalty_cost, ranked_alternatives[10],
              violations[], absence_exposure[{block, expected_uncovered_blocks}] }
```

Two fields do disproportionate work. `scenario_hash` makes the number reproducible from the log alone.
`ranked_alternatives` makes the recommendation negotiable without re-running anything, which in practice
is the difference between a roster that gets adopted and a roster that gets argued about for a week and
then republished unchanged.

The monitoring is four signals with owners and thresholds, not a dashboard.

| Signal | Definition | Threshold | Owner | Response |
|---|---|---|---|---|
| Roster deviation | published shifts minus recommended shifts, per day | any nonzero | workforce analyst | record reason code before publication |
| Realised shortfall | agent-blocks where served demand exceeded present agents | above 6 per day | operations manager | inspect the block, not the model |
| Absence rate drift | absent agents divided by rostered agents, rolling 4 weeks | outside 0.06-0.10 | HR analyst | re-estimate a, do not re-tune the roster |
| Penalty proxy gap | implied cost per uncovered hour from realised outcomes | ratio outside 0.7-1.4 of 60 | finance partner | re-open the valuation, not the optimiser |

The absence-drift response is stated as "re-estimate, do not re-tune" because the tempting reaction to a
rising absence rate is to add shifts, which converts a data problem into a permanent payroll increase.
Re-estimating the absence probability changes the recommended roster by at most one shift in this model;
adding a shift in response to a mis-measured rate changes it permanently and irreversibly.

## 9. Evaluation & Validation

Evaluation is specified before it is run. Every row states numerator, denominator, population, window,
comparator, guardrails, uncertainty treatment and execution status, and the statuses are honest: the only
things executed here are arithmetic and constraint checks on a constructed instance.

| Metric | Numerator / denominator | Population, window | Comparator | Guardrails | Uncertainty | Status |
|---|---|---|---|---|---|---|
| Expected cost per day | payroll plus expected penalty | all days, 4-week window | repeat-last-week roster | same scenario set both arms | scenario enumeration, no sampling error | illustrative 4106 vs 4488 |
| Penalty share of cost | expected penalty / expected cost | same | baseline share | penalty rate version pinned | none, arithmetic | illustrative 0.139 |
| Uncovered agent-hours | expected short blocks x 2 | same | baseline 2.0 hours | excludes leave days | binomial convolution | illustrative 9.5 |
| Realised shortfall | blocks with served demand above present agents | published days | none, diagnostic | excludes days with overrides | exact count | not-executed |
| Overtime minutes | paid overtime minutes / rostered shifts | published days | baseline 4-week mean | excludes approved project work | distribution, not mean | not-executed |
| Abandonment rate | abandoned contacts / offered contacts | published days, matched blocks | baseline matched blocks | excludes outage windows | Wilson interval | not-executed |
| Licence coverage | days with 14 licensed present in block 3 / days | all days | 100 percent required | hard constraint, not a metric | exact on roster | illustrative P 0.988 at pool 18 |
| Headcount requirement | CEIL(7 x shifts / 5) | weekly | establishment of record | must include leave liability | integer, sensitivity to cap | illustrative 24 vs 30 |
| Schedule stability | agents with more than 2 start changes / agents | weekly | agreement limit | hard constraint | exact | not-executed |
| Valuation robustness | penalty range over which chosen roster stays optimal | decision-time | none | rate sweep 30 to 300 | grid, not distribution | illustrative 50-74 USD/h |

The comparator deserves a paragraph, because the whole case rests on it. Comparing an optimised roster
against last week's average conflates the roster change with the demand change, and comparing it against
an idealised per-block staffing ignores the shift menu that makes the problem hard. The only defensible
comparator is the roster the operation would otherwise publish for the same dates, which is why the
repeat-last-week rule is stated as a rule rather than as a historical average. The cost of that
discipline is that the evaluation needs both arms, and the baseline arm is not free: it requires the
supervisor to produce the roster they would have produced anyway, which in a shadow run means producing
two rosters and publishing one.

Guardrails are constraints on the evaluation, not on the roster. Three are enforced. The population must
include leave days, because excluding them flatters both arms and changes the headcount arithmetic.
The window must be fixed before publication, because a window chosen after seeing the shortfall is a
window chosen to exclude it. And no metric may be reported without its valuation, because a service
number without a price is not comparable to a payroll number.

The validation ladder has four rungs, and the case executes three.

1. **Arithmetic validation (executed).** Every value in §6 re-derived by hand: coverage product, expected
   shortfall, marginal rule, absence convolution, headcount division and crossover rate. Each printed
   intermediate reconciles to a stated input, and the EEV table is consistent with the marginal rule to
   within the rounding of one shift.
2. **Neighbourhood optimality (executed).** The enumerated optimum is compared against every pair within
   two shifts in each direction, 25 points in all, and no point in that set beats it. This is weaker than
   a proof over the whole space but stronger than a claim, and it is what makes the plateau visible.
3. **Constraint-layer validation (executed).** The licensed and headcount constraints are applied to the
   cost optimum and to the baseline. At a licensed pool of 18 the recommended roster satisfies C2 with
   probability 0.9884; at a pool of 17 the probability is 0.9581, and at 16 the plan would need
   re-sequencing rather than re-counting. The headcount check confirms 24 against an establishment of 30.
4. **Shadow roster (not executed).** Publish both rosters for four weeks, execute only the baseline, and
   compare model-predicted shortfall with realised shortfall per block. The purpose is calibration of the
   absence overlay and of the penalty proxy, not adoption.

The gap between rung 3 and rung 4 is where this kind of work usually fails. The arithmetic is right, the
constraints hold, and the realised shortfall is nothing like the predicted one, because the predicted one
assumes demand is served by whole-block presence while in practice agents log in late, take breaks at
the wrong time, and get pulled into escalations. Rung 4 measures that gap and calibrates it; nothing
earlier can see it.

## 10. Action Taken

The action set is a proposal with owners, because a recommendation without an owner does not survive the
next scheduling meeting.

| # | Action | Owner | Trigger | Reversible |
|---|---|---|---|---|
| 1 | Publish (8, 9) rather than (9, 12) on non-campaign weeks | operations manager | scenario hash recorded | yes, weekly |
| 2 | Print the penalty rate and its break-even on the roster pack | workforce analyst | first publication | no, becomes policy |
| 3 | Adopt the constraint gate with coded violations | workforce analyst | first publication | yes, code change |
| 4 | Hold the licensed pool at 18 rather than 16 | HR business partner | before the first lean publication | yes, hiring decision |
| 5 | Run the absence overlay on every approved roster | workforce analyst | daily | yes |
| 6 | Report headcount requirement, not just shift count, at the monthly review | operations manager | monthly | no |
| 7 | Keep the repeat-last-week roster as an executable fallback | scheduling supervisor | any component failure | no |
| 8 | Run a four-week shadow comparison before claiming any outcome | analytics lead | after two clean publication cycles | yes |
| 9 | Re-open the penalty valuation with finance | finance partner | within one quarter | no |

The rejected actions matter as much as the adopted ones. A full weekly roster optimiser was rejected
because the publication cycle cannot absorb it: a weekly integer program over thirty agents and fourteen
shift patterns needs review capacity the team does not have, and a roster that cannot be reviewed will be
overridden. A chance constraint of "ninety percent of days fully covered" was rejected because it prices
nothing: at a 60-dollar penalty the model already delivers 75 percent full coverage in expectation, and
forcing 90 percent would add six shifts a day to buy a target nobody has agreed to pay for. And a
per-agent optimisation of start times was rejected because the agreement's two-start-change limit makes
the per-agent detail almost irrelevant to cost; it matters to fairness, which is a different objective
with a different owner.

The sequencing is deliberate: constraints and reporting first, counts second, shadow measurement third.
Publishing the leaner roster in week one would produce an outcome that nobody could attribute, because
the reporting change and the roster change would land together.

## 11. Outcome & Results

No operational outcome is claimed. The results below are the computed properties of the constructed
instance, reported with the status of each.

| Result | Value | Basis | Status |
|---|---|---|---|
| Expected cost, recommended roster | 4106 USD/day | arithmetic on constructed scenarios | illustrative |
| Expected cost, roster of record | 4488 USD/day | same | illustrative |
| Expected saving | 382 USD/day, 8.5 percent | difference of the two | illustrative |
| Annualised saving at 52 weeks | 139,048 USD | 382 x 7 x 52 | illustrative, overstated by weekly correlation |
| Expected uncovered hours, recommended | 9.5 h/day | scenario expectation | illustrative |
| Expected uncovered hours, record | 2.0 h/day | same | illustrative |
| Paid hours, recommended | 136 h/day | 17 shifts x 8 | illustrative |
| Paid hours, record | 168 h/day | 21 shifts x 8 | illustrative |
| Headcount requirement, recommended | 24 people | 119 shifts/week / 5, rounded up | illustrative |
| Headcount requirement, record | 30 people | 147 / 5, rounded up | illustrative |
| Break-even penalty | 110.93 USD/uncovered hour | crossover of the two plans | illustrative |
| Expected value of perfect information | 126 USD/day, 3.07 percent | scenario-specific optima | illustrative |
| Value of the stochastic solution | 382 USD/day | deterministic base-case plan vs stochastic | illustrative |

Two of those rows deserve commentary because they are easy to misread.

The annualised figure of 139,048 USD multiplies a daily expected saving by 364 and is therefore wrong for
any purpose except order of magnitude. Weekly demand is serially correlated, campaign weeks cluster, and
the saving is realised only on days when the leaner roster is actually feasible under the licensed and
rest constraints. A defensible annual figure would come from re-optimising each week against that week's
scenario set and summing, which the case does not do. The number is printed because a manager will
multiply it out anyway and it is better that the multiplication is visible than implicit.

The value of the stochastic solution of 382 USD needs its own definition, because the number happens to
equal the saving over the baseline and that coincidence is an artefact of this instance. VSS is the
difference between the expected cost of the plan chosen by optimising on the base scenario alone and the
expected cost of the stochastic optimum. Here the base-only plan is (9, 12) — the same pair the operation
publishes — which is why the two numbers coincide. In general they would not: an operation could have a
baseline that differs from the deterministic optimum, in which case the saving over baseline and the VSS
are different quantities with different interpretations. The case states this because a reader who
conflates them will over-claim in a review.

The expected value of perfect information is 126 USD a day, or 3.07 percent of the expected cost. That is
small, and its smallness is informative: it says that most of the achievable value comes from choosing a
better shape, not from knowing tomorrow's demand. A forecasting improvement that halved the scenario
spread would move the optimum by at most one shift. A roster redesign moves it by four. The
budget-allocation implication is uncomfortable but clear: at this cost structure, investment in
scheduling beats investment in forecasting.

The absence overlay changes the picture by about 159 USD a day on the recommended roster (728.67 against
570 of deterministic penalty), and that is the number to carry into the shadow run, because it is the one
that will be measured. If realised shortfall lands near 6.07 blocks a day, the overlay is calibrated and
the model can be used to answer questions about shift menus. If it lands near 4.75, absence is being
absorbed by informal mechanisms the model does not represent — agents staying late, supervisors taking
contacts, breaks being skipped — and the model's penalty is understated because the true cost of those
mechanisms is attrition rather than money.

## 12. Failure Modes & What Went Wrong

Six distinct mechanisms can make a roster that passes every check still fail in the operation. Each is
stated with competing causes and the check that separates them, because the single most common analytical
error with rosters is to attribute a shortage to the model when it is attributable to the data, or a
surplus to the data when it is attributable to the constraint layer.

**F1 — Scenario censoring from served-volume demand.** The high scenario is built from historical served
volume, which is capped by the roster that was published. On the days that would define the high
scenario, contacts abandoned and were never counted, so the high vector is too low and the recommended
roster is systematically lean. Competing causes: the high scenario is genuinely rare; the marketing
calendar is missing a campaign. Discriminating check: compare offered volume with served volume on
high-volume days; if the ratio exceeds 1.03 on the top decile, censoring is present. Correction:
rebuild from queued or offered volume. Prevention: store offered volume at block grain as a
first-class field.

**F2 — Penalty valuation mistaken for a measurement.** The 60-dollar rate is used as if it were observed.
When the recommendation is later challenged on service grounds, the analysis has no defence because it
never had a measurement. Competing causes: the rate is right but the linearity is wrong; the rate is
right and the operations team is simply uncomfortable with lower service. Discriminating check: for the
same blocks, plot abandonments against uncovered agent-hours; if the relationship is convex, the linear
rate overstates the cost of the first uncovered hour and understates the last. Correction: replace the
rate with a fitted curve or with a chance constraint. Prevention: never print an expected cost without
the rate and its break-even beside it.

**F3 — Absence treated as an independent, memoryless event.** The binomial overlay assumes independence.
In reality absence clusters: a transport strike, a school holiday, a weather event, and a team whose
members share a commute all produce correlated absence at exactly the moment demand peaks. Competing
causes: the rate is right and the clustering is absent; the rate is drifting upward and the model is
stale. Discriminating check: compute the variance of daily absence counts and compare with the binomial
variance n a (1 - a); if the observed variance is materially higher, clustering is present. Correction:
use an overdispersed model, or hold a standing float. Prevention: retain daily absence counts, not just
the monthly rate.

**F4 — The licensed constraint satisfied on paper only.** C2 requires 14 licensed agents present in block
three. A pool of 18 gives 0.9884 probability of that, and the 1.2 percent of days where it fails are
exactly the days with high absence, which are also the days with high demand. Competing causes: the pool
is adequate and the fail days are random; the licence register is stale and includes leavers. Discriminating
check: join the licence register against the active roster and count licence holders who have no
published shift in the last four weeks. Correction: refresh the register before counting the pool.
Prevention: treat the register as a rostering input with an owner and a refresh cadence.

**F5 — Headcount arithmetic that ignores the shift cap's interaction with leave.** The 24-person figure
assumes every agent works five shifts a week, every week. Annual leave, sickness and training reduce the
available shifts per person, so the true establishment is higher, and the ratio between the two plans is
not preserved. Competing causes: leave is already included in the pool; the cap is effectively four
shifts because of the start-change rule. Discriminating check: compute the realised shifts per person per
week over eight weeks and compare with five. Correction: divide by realised rather than contractual
capacity. Prevention: report headcount as a range with the assumed utilisation stated.

**F6 — The plateau collapses when a new shift pattern is introduced.** The reported optimum assumes two
shift types. If a four-hour evening pattern is added by agreement, the optimum moves, and a team that has
adopted (8, 9) as "the answer" will resist the change. Competing causes: the new pattern is more expensive
per hour and correctly rejected; the plateau was narrower than reported because the licensed constraint
binds differently under the new pattern. Discriminating check: re-enumerate with the additional pattern
and compare the top ten, not just the argmin. Correction: publish the ranked alternatives as the decision
record, so the recommendation is a choice among comparable options. Prevention: version the shift menu
with the roster, exactly as the scenarios are versioned.

A seventh mechanism is worth naming even though it is not a modelling failure: the roster is optimised,
published, and then quietly repaired by swaps until it no longer resembles the recommendation. The
discriminating check is trivial and rarely run — compare published rostered coverage per block against
the coverage implied by the recommended counts. If they diverge by more than one agent in any block, the
decision being executed is not the decision that was analysed, and no metric downstream of it can be
attributed to the model.

## 13. Transferable Lesson

The transferable structure is that uncertainty in staffing is a shape problem before it is a level
problem, and that the value of a stochastic treatment is often much smaller than the value of getting the
shape right. In this instance the stochastic premium over a base-case deterministic plan is 382 dollars a
day, and the expected value of perfect information is 126 — three times smaller again. Meanwhile the
difference between the roster of record and the base-case plan is zero, because they are the same roster.
The entire gain comes from placing shifts where the demand is, not from representing uncertainty more
faithfully.

Second, an expected-cost model is only as good as the valuation it embeds, and the professional move is to
report the break-even rather than the recommendation alone. A recommendation that survives a factor of
two in the disputed parameter is robust; one that flips within the plausible range is a sensitivity in
disguise. Here the lean plan survives any penalty below 110.93 dollars per uncovered hour, which brackets
the plausible range, and that is a much stronger statement than "the plan saves eight and a half
percent".

Third, the conversion from a daily plan to a weekly establishment is where the argument is won or lost.
Daily cost differences invite valuation arguments; headcount differences are auditable. Any roster
analysis that stops at cost has left its most persuasive artefact unbuilt, and any analysis that reports
headcount without the shift cap and leave arithmetic has built an artefact that will be rejected on
contact.

Fourth, absence is a distribution applied to a roster that is already tight in specific places, not a
percentage applied to a total. The same 8 percent absence costs 2.82 dollars in a block with a surplus and
446.40 in a block short by three, a factor of 158 between two blocks of the same day under the same
assumption. Treating absence as a scalar uplift distributes cost where the exposure is not.

Fifth, the failure modes are boundary failures. Censored input data, a stale licence register, an
unmeasured valuation and a swap process that rewrites the plan after publication will each destroy the
value of a correct optimisation, and none of them is visible in the optimiser's own diagnostics. The
checking effort belongs at the boundaries, which is why the design in §8 puts contracts, codes and hashes
between the components rather than inside them.

## 14. Connections & Cross-References

The nearest internal neighbours are the formulation sheet for the primal and dual view of this
programming structure, and the scheduling sheet for the shift-cover matrix that this case uses as an
input rather than derives.

| Reference | Relationship | What it supplies here |
|---|---|---|
| [Optimisation formulation](../Cheat_Sheets/01_optimisation_formulation.md) | theory | Two-stage structure, recourse, and why the dual prices of the coverage rows are the marginal shift values |
| [Solver selection](../Cheat_Sheets/02_solver_selection.md) | theory | Why enumeration is legitimate at two variables and when to move to a MILP |
| [Simulation experiment plan](../Cheat_Sheets/03_simulation_experiment_plan.md) | method | Replication and common-random-number logic for the shadow run |
| [Network and scheduling models](../Cheat_Sheets/04_network_and_scheduling_models.md) | theory | Shift-cover matrices and the integrality of interval-structured constraints |
| [Integer programming](../Cheat_Sheets/05_integer_programming.md) | theory | Branch-and-bound if a third and fourth shift type are added |
| [Metaheuristics](../Cheat_Sheets/06_metaheuristics.md) | contrast | Why a heuristic is unnecessary here and would cost the optimality proof |
| [Interval greedy and weighted scheduling](../Cheat_Sheets/07_interval_greedy_and_weighted_scheduling.md) | sibling | Interval feasibility and the buffer convention used when shifts overlap |
| [Case 01, last-mile routing](./01_last_mile_routing_optimisation.md) | sibling | Same two-stage discipline applied to distance rather than labour |
| [Case 03, scheduling objective and constraints](./03_scheduling_objective_and_constraints.md) | sibling | Objective-choice failure when the metric rewards the wrong thing |
| [Case studies index](./INDEX.md) | collection | Catalog row, evidence status and review state |

The contrast with case 01 is instructive and short. In the routing case the binding constraint was
capacity and the deliverable was feasibility; here the binding structure is the shift menu and the
deliverable is valuation robustness. Both cases arrive at a plateau rather than a point optimum, and in
both the plateau is the politically useful artefact. The contrast with case 03 is sharper: there the
failure was choosing a greedy rule that optimises the wrong objective, and here the analogous failure is
choosing a roster that optimises the visible objective (payroll) while the invisible one (service) is on
a different report.

### Interview pack

**Q1 — The roster (8, 9) has 17 shifts and the roster (9, 12) has 21. Which wastes more, and by how
much?**
Answer: (9, 12) wastes more, by 40 paid agent-hours in the base scenario and by 382 dollars of expected cost.
Reasoning: coverage for (9, 12) is (9, 9, 21, 21, 12, 12) against base demand (6, 9, 14, 16, 12, 7), so the
surplus is 3 + 0 + 7 + 5 + 0 + 5 = 20 agent-blocks, and each block is two hours, giving 40 hours of paid
idle capacity out of 168 paid hours — 23.8 percent. The expected-cost gap is 4488 minus 4106.
Follow-up: what does the surplus look like in the high scenario, where demand is 8, 11, 17, 19, 14, 9?
Red flag: answering with the total hours difference alone, which is 32 hours of payroll, and missing that
the larger roster is also short in two blocks on the high day.
Scoring: full credit for both the hour figure and the recognition that surplus and shortfall coexist.

**Q2 — Why is the cheapest roster by payroll not the answer?**
Answer: because payroll is only one of two costs, and the cheapest roster (5, 6) at 2288 dollars carries
2550 dollars of expected penalty for a total of 4838.
Reasoning: at eleven shifts the coverage is 5, 5, 11, 11, 6, 6 against base demand 6, 9, 14, 16, 12, 7,
so the base day alone is short in five of six blocks. The penalty at 120 dollars per uncovered block
dominates the 2288 of payroll. The minimum of the total is at seventeen shifts.
Follow-up: at what penalty rate does the cheapest-payroll roster become correct? Answer: below roughly
30 dollars per hour, where the sweep shows (6, 7) and leaner plans winning.
Red flag: treating payroll minimisation as the objective and service as a constraint to be checked
afterwards.
Scoring: full credit for naming the trade and locating the optimum between the two extremes.

**Q3 — One more late shift costs 208 dollars. How much uncovered time must it avoid to be worth buying,
and what does it actually avoid at the optimum?**
Answer: it must avoid 3.4667 agent-hours; at the optimum the marginal late shift avoids 3.5, so it is
worth buying by a margin of 2 dollars.
Reasoning: 208 divided by 60 dollars per uncovered agent-hour gives 3.4667 hours. Moving from (8, 8) to
(8, 9) reduces expected uncovered hours from 13.0 to 9.5, a difference of 3.5 hours, worth 210 dollars
against a cost of 208, and the expected cost falls from 4108 to 4106. The next shift, from (8, 9) to
(8, 10), avoids 2.5 hours worth 150 and loses 58.
Follow-up: recompute the break-even if the wage rises 4 percent; 216.32 divided by 60 is 3.605 hours, and
the marginal shift at 3.5 hours stops paying for itself.
Red flag: using total rather than marginal shortfall, which makes every additional shift look valuable.
Scoring: full credit for the break-even, the marginal comparison, and the correct sign of the result.

**Q4 — What single number would you put in front of the operations manager, and what must accompany
it?**
Answer: the expected cost of 4106 against 4488 for the roster of record, accompanied by the penalty rate
of 60 dollars per uncovered hour and the 110.93-dollar break-even at which the recommendation reverses.
Reasoning: the 382-dollar difference is meaningless without the valuation that produced it, and a manager
who does not know the break-even cannot judge whether their own view of service cost is compatible with
the recommendation. The pair (saving, break-even) is the minimum defensible unit of reporting.
Follow-up: what if the manager says the true cost is 200 dollars per hour? Then the recommendation is
(9, 12) at 4668, and the case's own sweep already shows that.
Red flag: quoting 139,048 dollars a year from the daily figure without stating the correlation caveat.
Scoring: full credit for the pair, partial credit for the saving alone.

**Q5 — The model says zero shortage under (9, 12) in the base scenario, but the operation reports
overtime on those days. Diagnose.**
Answer: the model compares whole-block coverage with whole-block demand and treats every rostered agent
as present for the entire block; overtime arises from within-block effects and from absence.
Reasoning: three candidates dominate. Absence: at 8 percent, a roster exactly equal to demand is short
on 69 percent of days by at least one agent. Intra-block: demand is modelled as a block average, so a
peak inside the block is invisible. Relief: if breaks are not staggered, capacity is overstated by about
6 percent, which is 0.48 agents per shift. The check is to compare realised present counts per block
with rostered counts; a gap of one or more agents on most days identifies absence and relief rather than
demand.
Follow-up: which of the three changes the roster recommendation? Absence does, by up to one shift;
intra-block peaks change the shift menu, not the count.
Red flag: concluding that demand is under-forecast, which is the least likely of the four explanations.
Scoring: full credit for three distinct causes with a discriminating check.

**Q6 — Compute the expected uncovered agent-hours for block one under plan (8, 9) with 8 percent
absence.**
Answer: 1.64 agent-blocks, which is 3.28 agent-hours, against a deterministic shortfall of 1 block.
Reasoning: eight rostered, nine required. P(0 absent) is 0.92 to the eighth, 0.5132, giving shortfall 1.
P(1 absent) is 8 x 0.08 x 0.92^7 = 0.3569, giving shortfall 2. P(2 absent) is 28 x 0.0064 x 0.92^6 =
0.1087, shortfall 3, and the remainder contributes 0.0112 at shortfall 4. The sum is 0.5132 + 0.7138 +
0.3261 + 0.0448 = 1.5979, plus the tail, giving 1.64.
Follow-up: why is the expected shortfall more than double the deterministic one for the same one-agent
gap? Because the distribution is skewed: surplus capacity cannot be used, but deficit capacity is
unbounded.
Red flag: multiplying the deterministic shortfall by 1.08 or dividing the demand by 0.92.
Scoring: full credit for the convolution and the convexity explanation.

**Q7 — Convert the recommended daily roster into a headcount requirement, and say what makes the
number wrong.**
Answer: 24 people, from 119 shifts a week divided by the five-shift cap, rounded up. It is wrong because
it assumes five shifts per person every week and no leave.
Reasoning: 17 shifts a day times 7 is 119; 119/5 is 23.8. The same arithmetic on 21 shifts gives 147/5 =
29.4, so 30 people. Leave, sickness and training reduce realised shifts per person below five, so the
true establishment is higher; the ratio between the plans is not preserved because the two rosters have
different shift-start mixes and the start-change limit binds differently.
Follow-up: if realised shifts per person are 4.4, what is the requirement? 119/4.4 = 27.05, so 28.
Red flag: reporting 23.8 or 24 without the ceiling, or ignoring the cap entirely and dividing by seven.
Scoring: full credit for the division, the ceiling and at least one reason the figure is optimistic.

**Q8 — The value of the stochastic solution is 382 dollars and the expected value of perfect
information is 126. What do these two numbers tell you to invest in?**
Answer: they say the shape of the roster matters far more than the accuracy of the forecast, so invest in
scheduling design before forecasting.
Reasoning: VSS measures what is lost by ignoring the scenario spread and optimising on the base case;
EVPI measures what is lost by not knowing tomorrow's demand. Both are small relative to the 382-dollar
gap between the roster of record and the base-case plan, which is a pure placement gain. The ratio
EVPI/VSS of about one third says that even perfect foresight would not fix a badly shaped roster.
Follow-up: when would EVPI dominate? When the shift menu is flexible, because then the roster can adapt
and uncertainty becomes the only obstacle.
Red flag: concluding that the forecast should be improved because the scenario probabilities are the
least reliable input.
Scoring: full credit for the ranking of the three effects with the arithmetic attached.

**Q9 — What does it cost to guarantee zero shortage in all three scenarios, and when would you pay
it?**
Answer: the minimal two-type roster that covers the high scenario in every block is (11, 14), 25 shifts
at 5200 dollars, a premium of 1094 dollars a day, or 26.7 percent.
Reasoning: early shifts must satisfy block zero at 8 and block one at 11, so E is at least 11; late shifts
must satisfy block four at 14 and block five at 9, so L is at least 14; the overlap constraint is
11 + 14 = 25 at least 19. The plan wastes 45 agent-blocks in the low scenario, where demand is 5, 8, 12,
14, 10, 6.
Follow-up: at what penalty does the full-coverage plan become optimal? Above roughly 300 dollars per
uncovered hour, per the sweep, where it still loses to (9, 12) at 200.
Red flag: asserting that a service-level target requires full coverage rather than pricing the target.
Scoring: full credit for the constraint derivation and the premium.

**Q10 — Two rosters have the same 21 shifts a day. Can one be short and idle at the same time?**
Answer: yes, and the roster of record is exactly that: 40 paid idle hours in the base scenario and four
short blocks in the high scenario.
Reasoning: (9, 12) gives coverage 9, 9, 21, 21, 12, 12. Base demand is 6, 9, 14, 16, 12, 7, so blocks two
and three hold seven and five agents of surplus. High demand is 8, 11, 17, 19, 14, 9, so block one is
short by two and block four by two. The surplus sits in the middle of the day; the shortage sits at the
edges; no shift can move capacity between them.
Follow-up: what change fixes it without changing the total? Moving two shifts from late to early makes
(11, 10), which removes the block-one shortage and worsens block four.
Red flag: assuming that equal totals imply equivalent coverage.
Scoring: full credit for demonstrating both effects on the same roster.

**Q11 — A manager proposes dividing the requirement by 0.92 to cover absence. What is wrong with
that?**
Answer: it over-corrects in blocks with surplus and under-corrects where the roster is already short,
because it is applied to the total rather than to the shortfall distribution.
Reasoning: dividing by 0.92 inflates every block by 8.7 percent. In block three of (8, 9) seventeen
agents become nineteen, adding two paid agents to defend 0.60 expected shortfall blocks; in block four
nine become ten, which still leaves 2.5 expected shortfall blocks because the gap is three agents before
absence. The marginal rule with absence-adjusted shortfall is the correct instrument: add a shift only
where it avoids more than 3.4667 expected uncovered hours.
Follow-up: compute the cost of the naive uplift for (8, 9): nineteen and ten shifts, or twenty-nine,
costs 6032 and still leaves block one short.
Red flag: treating absence as a multiplicative correction to demand.
Scoring: full credit for both directions of error and the marginal alternative.

**Q12 — What guardrail stops the optimiser from consuming the rest rules?**
Answer: the constraint gate applied after optimisation, with coded violations that block publication
regardless of cost.
Reasoning: the cost optimum is a pair of counts; the rest rules constrain the assignment of people to
those counts and the weekly pattern. A model that treats C3 and C4 as soft will, at a high enough
penalty, buy coverage with a sixth consecutive shift. Because the violation is a property of the
assignment rather than of the counts, the only reliable guardrail is a separate gate that returns codes
and vetoes.
Follow-up: what happens to the cost if the gate vetoes the optimum? Fall back to the next alternative in
the ranked list, which is why the enumerator returns ten.
Red flag: relying on an objective penalty for a legal requirement.
Scoring: full credit for the separation of counts from assignment and the veto semantics.

**Q13 — With 17 agents rostered in block three and 14 required to hold the claims licence, how large
must the licensed pool be?**
Answer: 18 gives 0.9884 probability of at least 14 licensed present; 17 gives 0.9581; 16 would be
materially worse and would need re-sequencing.
Reasoning: availability per licensed agent is 0.92, so the number present is binomial. With a pool of 18
the probability that at least 14 are present is the sum of the upper tail of Binomial(18, 0.92), which is
0.9884. With 17 it is 0.9581. The failure days are the high-absence days, which correlate with the
high-demand days.
Follow-up: what is the probability that all 18 attend? 0.92^18 = 0.2230.
Red flag: checking the constraint on the nominal roster rather than on the absence-adjusted one.
Scoring: full credit for a tail computation and the correlation caveat.

**Q14 — At what penalty rate do the lean and full plans cross, and why is the true crossover likely
lower?**
Answer: 110.93 dollars per uncovered agent-hour by the linear crossover; lower in reality because the
penalty is convex.
Reasoning: (4368 - 3536) / (9.5 - 2.0) = 832 / 7.5 = 110.93. The linear form assumes each uncovered hour
costs the same, but abandonment rises faster than linearly with occupancy, so the last uncovered hours are
more expensive than the first, which penalises the lean plan at a lower nominal rate.
Follow-up: what empirical shape would you fit? Abandonment against offered load per block, then price at
the realised marginal rate.
Red flag: presenting 110.93 as a measurement.
Scoring: full credit for the arithmetic, the direction of the correction and the reason.

**Q15 — What evidence would make you abandon the three-scenario model?**
Answer: a shadow run in which realised shortfall per block is not bracketed by the low and high
scenarios, or in which the scenario probabilities fail to predict the frequency of high days.
Reasoning: the model's two testable claims are the demand spread and the absence overlay. If realised
shortfall falls outside the predicted interval on more than a few days, either the vectors or the overlay
is wrong. A frequency check is cheap: if high days occur at 10 percent rather than 25 percent, the
probability vector is misspecified and the optimum shifts toward the lean plan.
Follow-up: what is the cheapest falsification? Compare predicted with realised shortfall for four weeks
without changing anything, and count the days outside the bracket.
Red flag: proposing to validate by asking the operations team whether the roster feels right.
Scoring: full credit for a specific, countable check with a decision rule.

**Q16 — You are given only a point forecast of 64 agent-blocks, no distribution. What changes?**
Answer: the analysis collapses to a deterministic cover problem whose optimum is (9, 12) at 4368, which
is 382 dollars worse in expectation than the stochastic plan, and the recommendation loses its ability to
justify accepting any shortage.
Reasoning: with one demand vector the penalty term either vanishes or becomes an arbitrary feasibility
requirement. The deterministic optimum covers the vector exactly at the minimum cost, which produces the
very roster the operation already publishes. That is the practical argument for scenario work: not that
it is more sophisticated, but that a point forecast cannot express the trade the decision actually
involves.
Follow-up: what is the minimum viable uncertainty representation? Three scenarios with probabilities, or
a two-point spread plus a service constraint.
Red flag: proposing to add a safety margin to the point forecast, which is the naive uplift again.
Scoring: full credit for identifying that the deterministic answer equals the incumbent.

## 15. Quick Reference Card

| Item | Value | Note |
|---|---|---|
| Decision | early and late shift counts for one day | two integers, 676 candidates |
| Coverage rule | E in blocks 0-1, E+L in 2-3, L in 4-5 | one pattern per shift type |
| Shift cost | 208 USD | 8 paid hours at 26 USD |
| Penalty | 60 USD per uncovered agent-hour | negotiated, never measured |
| Scenarios | 110 / 128 / 156 agent-hours at 0.25 / 0.50 / 0.25 | level and shape both vary |
| Recommended | 8 early, 9 late, 17 shifts | expected cost 4106 USD |
| Incumbent | 9 early, 12 late, 21 shifts | expected cost 4488 USD |
| Saving | 382 USD/day, 8.5 percent | annualisation overstated |
| Break-even | 3.4667 uncovered hours per added shift | 208 / 60 |
| Crossover | 110.93 USD per uncovered hour | lean vs full plan |
| Absence effect | +159 USD/day on the recommended roster | 6.07 vs 4.75 short blocks |
| Headcount | 24 people vs 30 for the incumbent | 119 / 5 and 147 / 5 |
| EVPI | 126 USD/day, 3.07 percent | forecasting is not the lever |
| Hard constraints | licensed 14 in block 3, 5 shifts/week, 2 start changes | gate after optimisation |
