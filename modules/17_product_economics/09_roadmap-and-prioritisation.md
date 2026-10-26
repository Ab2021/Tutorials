# Roadmap and Prioritisation

> Module 17 — Product, Economics, and Leadership
> Topic 09 of 09
> Estimated study time: 4 hours
> Prerequisites: 02 (Value Quantification), 08 (Leadership)
> Version 2026.3

## 1. Why this topic exists

A roadmap is not a list of features with dates. It is a sequence of commitments and bets, made
under uncertainty, against a fixed capacity, in an order that determines how much you learn and
how much you risk before you learn it. Most roadmaps are lists, which is why most roadmaps are
wrong within a quarter.

The two errors are symmetric. A roadmap that commits to everything commits to nothing and
overruns. In the worked example, the planned work totals 62 team-weeks against 52 weeks of
capacity, a 119 percent utilisation that guarantees either a slip or an unplanned cut. A roadmap
that commits to nothing preserves optionality and delivers nothing, which is the failure mode of
teams that have been burned by the first error.

Prioritisation is the discipline of choosing what not to do, in public, with a reason. It is the
most visible expression of a product leader's judgement, and it is the part of the job most often
performed badly because the tools encourage false precision.

### 1.1 Why prioritisation methods disappoint

Ranking methods such as weighted shortest job first produce a number and a rank. The rank is
usually treated as a decision, and it is not. It is an input that encodes assumptions about value,
time sensitivity, risk, and size, most of which are guesses. A method that hides its assumptions
in a score produces confident-looking decisions that nobody can audit.

The methods are still useful, for two reasons. They force explicit comparison on defined
dimensions, and they make disagreement specific. A team that argues about a WSJF score is arguing
about value, time criticality, or size, which is a productive argument. A team that argues about
"priority" is arguing about status.

### 1.2 Sequencing versus scheduling

Scheduling asks when each item will be done. Sequencing asks what order reduces the cost of being
wrong. The second question matters more when uncertainty is high, because the order determines
how much capital is committed before the key uncertainty is resolved.

The worked example makes the point numerically. Building the scale-up immediately has an expected
value of 3.20 million United States dollars (USD). Running a 1.20 million USD evidence study first
raises it to 4.03 million USD. The evidence step is worth 0.83 million USD in expectation, and it
is worth more when the probability of success is lower, not higher.

### 1.3 What this topic covers

It covers how to build a roadmap that is honest about uncertainty, how to prioritise with methods
that expose their assumptions, how to sequence work to buy information cheaply, and how to keep a
roadmap connected to capacity and to the commercial calendar. It does not cover agile delivery
mechanics, sprint planning, or engineering estimation technique.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Separate committed work from bets and state the commitment level of each | Exercise 1; self-check 1 |
| 2 | Size capacity and reserve an explicit buffer | Exercise 2; self-check 2 |
| 3 | Quantify cost of delay and cost of duration for a roadmap item | Exercise 3; self-check 4 |
| 4 | Rank items with a stated method and audit the ranking's assumptions | Exercise 4; interview question 4 |
| 5 | Test a ranking against the portfolio rather than item by item | Exercise 5; self-check 6 |
| 6 | Sequence evidence-generating work to reduce capital at risk | Exercise 6; interview question 6 |
| 7 | Set decision gates and kill criteria for each bet | Exercise 7; self-check 8 |
| 8 | Communicate a roadmap as a set of decisions with confidence levels | Exercise 8; interview question 10 |

## 3. The core mental model

A roadmap has three layers: the horizon, the uncertainty, and the capacity. The horizon says how
far ahead a statement is meaningful. The uncertainty says what kind of work it is. The capacity
says how much of it can be true at once.

### 3.1 The three horizons

```text
  NOW          committed, resourced, dated
    |          confidence: high        typical error: +-15%
    v
  NEXT         scoped, not resourced, sequenced
    |          confidence: medium      typical error: +-40%
    v
  LATER        directional, not scoped
               confidence: low         typical error: not estimable
```

The error is to publish LATER items with NOW dates. That converts a direction into a promise the
team cannot keep, and it trains the organisation to discount the roadmap entirely after the first
miss.

A useful rule: publish dates only for NOW, sequence only for NEXT, and themes only for LATER. A
roadmap that violates this rule is either dishonest or naive, and the organisation will discover
which.

### 3.2 The uncertainty funnel

```text
  EXPLORE  ------------------------------------------------
    cheap experiments, evidence studies, prototypes
    goal: reduce uncertainty about value or feasibility
    kill criteria: explicit
         |
         v
  EXPLOIT  -----------------------------------------------
    build, scale, integrate, sell
    goal: capture value from a validated opportunity
    commitment: capital, headcount, contracts
```

The sequencing principle is to spend little in EXPLORE and concentrate in EXPLOIT. A portfolio that
spends heavily in EXPLORE never captures value; one that spends heavily in EXPLOIT without
exploring commits capital before the key uncertainty is resolved.

The ratio depends on the business. A late-stage product with a validated market might spend 10
percent on exploration. A new product area might spend 40 percent. A product whose core claim is
unvalidated should spend most of its budget on the experiment.

### 3.3 Cost of delay and cost of duration

Two quantities drive sequencing, and they are different.

- **Cost of delay** is what you lose per unit of time until the item ships. It is a property of
  the item's value and its time sensitivity.
- **Cost of duration** is what the item costs per unit of time while it is being built. It is a
  property of the item's size and the opportunity cost of the capacity it consumes.

A large, valuable item can have a high cost of delay and a high cost of duration. Sequencing it
first may be right or wrong depending on which dominates. A small item with high cost of delay
should almost always go first, which is why the weighted shortest job first heuristic works when
it works.

### 3.4 Real options in a roadmap

A roadmap is a portfolio of options. An option has value because it lets you decide later with
better information. Four properties make an option valuable:

1. **Uncertainty.** If the outcome is known, there is nothing to learn.
2. **Time to decide.** If the decision cannot wait, the option expires.
3. **Cost of the experiment.** Cheap experiments create more option value per dollar.
4. **Reversibility of the commitment.** If the scale-up cannot be stopped, the option to stop is
   worth more.

The research on real options logic in product development is more nuanced than the practitioner
literature suggests. A study of resource allocation regimes found that sequencing alone predicted
higher new product sales, while low initial commitment and reallocation mattered only as a matched
pair (Academy of Management Journal, 2014). The practical reading is that staging decisions is
valuable, but staging without the willingness to reallocate capital after the stage is a ritual.

### 3.5 The capacity constraint is the plan

A roadmap that does not reconcile to capacity is a wish list. The arithmetic is simple and
ruthless:

```text
available_capacity = headcount x weeks x focus_factor
committed_capacity = sum of item estimates
buffer             = available_capacity - committed_capacity
```

A buffer below 15 percent is a plan to miss. A buffer above 40 percent is a plan to underuse. The
worked example's 119 percent utilisation is not ambitious; it is arithmetically impossible without
overtime, descoping, or both.

### 3.6 Redraw test

Draw the three horizons, place every roadmap item in one, and check that no LATER item carries a
date. Then sum the NOW and NEXT estimates and compare with capacity. If the sum exceeds capacity,
the roadmap is a list of intentions, not a plan.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| Roadmap | Sequenced plan of commitments and bets under uncertainty | Used to mean a feature list | Artifact |
| Prioritisation | Choosing what to do and what not to do | Confused with ranking | Process |
| Sequencing | Ordering work to manage risk and learning | Confused with scheduling | Process |
| Horizon | Time band within which a statement is meaningful | Used to imply dates | Now, next, later |
| Commitment | A promise with resources and a date | Applied to all items | Level |
| Bet | An investment with uncertain outcome and a kill criterion | Confused with a commitment | Level |
| Explore | Work whose purpose is to reduce uncertainty | Confused with research | Mode |
| Exploit | Work whose purpose is to capture validated value | Confused with delivery | Mode |
| Cost of delay | Value lost per unit of time until delivery | Confused with urgency | Currency per week |
| Cost of duration | Cost of occupying capacity per unit of time | Ignored | Currency per week |
| WSJF | Weighted shortest job first: cost of delay divided by job size | Treated as objective truth | Ratio |
| RICE | Reach, impact, confidence, effort scoring | Used without calibration | Score |
| Cost of delay | Also used as the numerator in CD3 | Confused with WSJF | Currency |
| CD3 | Cost of delay divided by duration | Confused with WSJF | Ratio |
| Time criticality | How much value decays with delay | Confused with a deadline | Score |
| Risk reduction | Value of an item in reducing uncertainty | Confused with technical debt | Score |
| Opportunity enablement | Value of an item in enabling future work | Ignored in scoring | Score |
| Job size | Duration or effort of an item | Underestimated | Weeks |
| Buffer | Capacity reserved for the unexpected | Treated as slack | Percentage |
| Utilisation | Committed capacity divided by available capacity | Maximised as a goal | Percentage |
| Throughput | Work completed per period | Confused with utilisation | Items per period |
| Cycle time | Time from start to completion | Confused with lead time | Days |
| Little's law | Throughput equals work in progress divided by cycle time | Ignored | Relationship |
| Work in progress | Items started and not finished | Maximised accidentally | Count |
| Stage gate | A decision point between phases | Confused with a status review | Gate |
| Kill criterion | Pre-agreed condition that stops a bet | Written after the result | Decision rule |
| Option value | Value of the ability to decide later | Ignored in fixed plans | Currency |
| Sunk cost | Cost already incurred and unrecoverable | Used to justify continuing | Currency |
| Escalation of commitment | Continuing a failing course because of prior investment | Confused with perseverance | Bias |
| Planning fallacy | Systematic underestimation of duration | Treated as optimism | Bias |
| Reference class forecasting | Estimating from similar past projects | Replaced by bottom-up estimates | Method |
| Confidence level | Stated probability that a date will be met | Omitted | Percentage |
| Dependency | Work that must precede another item | Under-mapped | Relationship |
| Critical path | Longest sequence of dependent work | Confused with the busiest person | Sequence |
| Platform work | Investment that enables multiple items | Deferred indefinitely | Category |
| Technical debt | Cost of rework deferred | Used to justify any refactor | Category |
| Theme | A direction without a scope | Given a date | Category |
| Now, next, later | Horizon vocabulary for honest roadmaps | Used to hide a lack of dates | Vocabulary |

## 5. The mechanics, step by step

The procedure builds a roadmap from a set of candidate items, a capacity constraint, and a set of
uncertainties. It assumes the commercial calendar from Topics 04 and 05 is known.

### Step 1. Separate commitments from bets

**Input:** the candidate list.
**Output:** two lists, with a commitment level and a reason for each.
**Rationale:** commitments and bets need different governance. A commitment is resourced and dated;
a bet has a hypothesis, a test, and a kill criterion. Mixing them produces either missed
commitments or bets that are never killed.

| Level | Evidence required | Governance | Example |
|---|---|---|---|
| Commitment | Contract, regulatory obligation, or executive approval | Progress review | Security remediation required by a customer contract |
| Planned | Validated need and a scoped design | Gate review | Feature with signed-off requirements |
| Bet | Hypothesis and a test | Kill criterion | New product area with unproven demand |
| Direction | A thesis without a scope | None | Platform strategy for the next two years |

A useful discipline: every item in the NOW horizon must be a commitment or a planned item, and
every bet must be in NEXT with a gate. A bet in NOW is a commitment the team cannot keep.

### Step 2. Size capacity and reserve a buffer

**Input:** headcount, calendar, and historical throughput.
**Output:** available capacity in person-weeks, and a reserved buffer.
**Rationale:** the capacity number is the denominator of every prioritisation decision. Without it,
the roadmap is unconstrained and therefore meaningless.

```text
available = headcount x working_weeks x focus_factor
```

Focus factor is the share of time available for roadmap work after meetings, support, and
interruptions. A value of 0.6 to 0.75 is typical for a product team; teams that assume 1.0 are
planning for a week that does not exist.

Reserve a buffer of 15 to 25 percent for unplanned work: production incidents, security findings,
customer escalations, and regulatory requests. A roadmap without a buffer does not eliminate
unplanned work; it displaces planned work, usually the platform investment that would have reduced
future unplanned work.

### Step 3. Classify each item by uncertainty

**Input:** the candidate list.
**Output:** each item labelled explore or exploit.
**Rationale:** the two modes need different estimates and different gates. An explore item cannot
be estimated in weeks with confidence, because its duration depends on what it finds.

| Signal | Mode | Estimate type | Gate |
|---|---|---|---|
| The outcome is unknown | Explore | Time-boxed, not duration-estimated | Kill criterion |
| The design is unknown | Explore | Spike with a fixed cap | Design review |
| The need is validated and the design is known | Exploit | Duration with a range | Delivery review |
| The obligation is external | Exploit | Fixed date driven by the obligation | Compliance review |

An explore item should be time-boxed, not estimated. "Six weeks to determine whether the data are
accessible and the effect is real" is a valid explore plan. "Eight weeks to build the model" is not,
because the model's feasibility is the unknown.

### Step 4. Quantify cost of delay for each item

**Input:** the value model for each item.
**Output:** a cost of delay in currency per week, with a range.
**Rationale:** cost of delay converts urgency from a feeling into a number, and it makes trade-offs
between items of different types possible.

Three patterns of cost of delay:

```text
  linear       value accrues steadily with time; delay costs the same each week
  urgent       value decays sharply after a date; delay is catastrophic after the date
  deferrable   value is roughly constant; delay costs little
```

For a linear item, cost of delay is the annual value divided by 52. For an urgent item, it is the
value at risk if the date is missed, concentrated in the weeks around the date. For a deferrable
item, it is small until a trigger, then large.

The distinction matters because a ranking method that sums "time criticality" as a single score
collapses these patterns and hides the date-dependent item that must not slip.

For an urgent item, estimate the cost of missing the date directly rather than deriving it from an
annual value. A contract penalty, a lost eligibility window, or a regulatory submission deadline
has a step change in value that no per-week average represents. Record the step change separately
from the linear cost of delay, and flag the item as date-bound in the ranking. A date-bound item
that ranks third is still the first item that must be protected, and the roadmap should say so
explicitly rather than relying on the score.

### Step 5. Estimate duration with a range, not a point

**Input:** the item description.
**Output:** a p50 and p90 duration.
**Rationale:** a point estimate is a false precision that makes the plan brittle. A range allows the
roadmap to state a confidence level, which is what a stakeholder actually needs.

Use reference class forecasting where possible: take the last five comparable items and use their
actual durations as the distribution. Teams systematically underestimate by 30 to 50 percent when
they estimate bottom-up, and the reference class corrects for that without requiring anyone to be
less optimistic.

A practical format:

| Item | p50 | p90 | Basis |
|---|---:|---:|---|
| Integration with the claims platform | 8 weeks | 14 weeks | Three comparable integrations |
| Outcome study | 10 weeks | 16 weeks | Two prior studies, plus data access risk |
| Security remediation | 4 weeks | 6 weeks | Vendor estimate plus internal review |

The p90 is the number to use for a commitment. The p50 is the number to use for sequencing.

### Step 6. Rank with a stated method

**Input:** cost of delay, size, and any hard constraints.
**Output:** a ranked list with the method and its assumptions stated.
**Rationale:** the method matters less than the transparency. A ranking that can be audited can be
argued with productively.

Weighted shortest job first, as used in the Scaled Agile Framework, is the most common method:

```text
WSJF = (user-business value + time criticality + risk reduction and opportunity enablement)
       / job size
```

Scores are relative, usually on a modified Fibonacci scale of 1, 2, 3, 5, 8, 13, 20. The method's
virtues are that it includes time sensitivity and risk reduction, and that it favours smaller
items, which reduces batch size. Its defects are that relative scores are not additive, that it
penalises large strategic bets, and that it says nothing about dependencies.

A worked example of the defect: an outcome study that unlocks a 14 million USD decision may score
low because its job size is large, while an audit logging change scores high because it is small
and time-critical. Both are legitimate; the ranking is not a decision.

RICE is the other common method, scoring reach, impact, and confidence divided by effort. It
handles confidence explicitly, which WSJF does not, and it is easier to explain to a commercial
audience. It handles time sensitivity poorly, which WSJF does explicitly. The choice between them
matters less than the discipline of stating which dimensions are excluded: RICE ignores urgency,
WSJF ignores confidence, and neither accounts for dependencies or regulatory dates. Whichever is
used, the omitted dimensions should be listed in the roadmap document so that the reader knows
what the ranking did not consider.

### Step 7. Test the ranking against the portfolio

**Input:** the ranked list.
**Output:** a portfolio that maximises total value subject to capacity and dependencies.
**Rationale:** item-by-item ranking does not optimise a portfolio. Greedy selection by ratio can
leave capacity unusable and produce a worse total than a different combination.

In the worked example from Topic 08, ranking by expected value per engineer-day selects B then A
for 735,000 USD, while the optimal portfolio is B plus C for 870,000 USD. The difference is 135,000
USD, or 18 percent, and it comes entirely from considering combinations rather than items.

For a roadmap, the same logic applies with dependencies added: an item that enables two others has
a portfolio value higher than its standalone value.

### Step 8. Sequence to buy information cheaply

**Input:** the portfolio and the uncertainties.
**Output:** a sequence in which the cheapest, most informative work comes first.
**Rationale:** the order of work determines how much capital is committed before the key
uncertainty is resolved. A sequence that builds first and tests later exposes the full investment
to the risk of being wrong.

The sequencing rules:

1. Resolve the uncertainty that would change the decision, not the one that is easiest to study.
2. Spend the minimum to resolve it; a study that costs as much as the decision is not an option.
3. Place the evidence step before the irreversible commitment, not after.
4. Parallelise exploration only when the scale-up capital is constrained, so that only the best
   option is funded.
5. Keep the option to stop open until the evidence is in.

Rule 4 is counter-intuitive and is the core of real-options sequencing. Running two cheap
explorations in parallel and committing to only one scale-up can be worth more than running one
exploration and scaling it, because the probability of at least one success rises faster than the
exploration cost.

Rule 5 is what most teams violate. The option to stop is destroyed the moment the build starts,
and it is destroyed silently, because starting the build feels like progress. A sequence that
places a reversible preparation step before the irreversible commitment preserves the option
without delaying the commitment by much. Preparing the deployment environment, drafting the
contract, and pre-selling the launch are reversible; signing the contract and buying the hardware
are not.

### Step 9. Set gates and kill criteria

**Input:** each bet.
**Output:** a gate with a date, a metric, a threshold, and a decider.
**Rationale:** a bet without a kill criterion becomes a commitment through inertia. The gate is the
mechanism that converts the option into a decision.

The gate specification:

| Element | Question | Example |
|---|---|---|
| Date | When is the decision? | End of Q2 |
| Metric | What is measured? | Data access confirmed and effect size at least 8 percent |
| Threshold | What result continues? | Both conditions met |
| Decider | Who decides? | Chief product officer |
| Consequence | What happens on failure? | Stop; reallocate the team to the next bet |
| Cost of the gate | What does the evidence cost? | 1.2 million USD |

A gate with no consequence is a review. A gate whose decider is the person who proposed the bet is
a ratification.

### Step 10. Reserve capacity for the unexpected and for platform work

**Input:** the portfolio.
**Output:** a roadmap with explicit reserves.
**Rationale:** unplanned work is certain, and platform work is always deferrable, which means it is
always deferred. Both need a protected allocation.

A workable split for a product team:

| Allocation | Share | Purpose |
|---|---:|---|
| Committed delivery | 50 to 60 percent | Contracted and planned items |
| Bets and exploration | 15 to 25 percent | Evidence-generating work |
| Platform and debt | 15 to 20 percent | Enablement and remediation |
| Buffer | 10 to 15 percent | Unplanned work |

The exact numbers depend on context, but the principle does not: every category needs a protected
share, or the loudest category will consume all of it.

### Step 11. Publish the roadmap as decisions, not promises

**Input:** the portfolio and sequence.
**Output:** a communication with horizons, confidence levels, and the reasoning.
**Rationale:** a roadmap is read by people who will plan their own work around it. Publishing
dates for uncertain items transfers risk to them without warning.

The communication states: what is committed and when, with a confidence level; what is next and in
what order, without dates; what is a direction, without scope; what is explicitly not being done,
with a reason; and what would change the plan.

The "not being done" list is the most important section and the one most often omitted. A roadmap
that does not say what it is not doing invites every stakeholder to assume their item is included.

### Step 12. Review on a cadence and re-plan against evidence

**Input:** progress and new information.
**Output:** a revised roadmap with a record of what changed and why.
**Rationale:** a roadmap that is not revised is either perfect or ignored. The review cadence
should match the rate at which new information arrives, typically monthly for a delivery roadmap
and quarterly for a strategic one.

The review asks four questions: what did we learn that changes the plan; what slipped and why; what
should be killed; and what capacity has been released or consumed. The output is a diff against the
previous roadmap, not a new document.

### 5.13 Branch points and edge cases

**No reimbursement pathway.** Without a payment route, the roadmap's commercial milestones are
budget approvals and contract signatures rather than code or coverage decisions. The sequencing
should put the evidence that unlocks a budget owner ahead of the build, because the build has no
value without a payer.

**Single-payer versus multi-payer.** In a single-payer system, one national assessment can
determine the market, and the roadmap must align with the assessment calendar, which the team
cannot influence. In a multi-payer system, the roadmap can sequence payer by payer, learning from
the first contract and applying it to the next.

**Capital versus operating budget.** A capital purchase has a fixed approval window and a long
lead time, so the roadmap must include the procurement and finance milestones, not only the
engineering work. An operating subscription can be sequenced more flexibly.

**Pilot-to-contract transition.** The roadmap should place the contract decision before the
expansion work, not after. A team that expands a pilot before the contract is signed is building
on an unfunded assumption.

**Procurement security review.** Security remediation has a hard external date and a dependency on
a third party's review queue. It belongs in the committed band, not the buffer, because it is
predictable and non-negotiable.

**Champion turnover.** A roadmap that depends on a champion's continued sponsorship should include
a milestone for institutionalising the requirement, such as a committee approval or a budget line,
before the champion is needed again.

**Regulatory milestones.** A device submission, a clinical trial start, or a certification
deadline is a fixed point that dominates sequencing. The roadmap should be built backwards from
those dates, and any item that cannot fit before them should be moved, not compressed.

### 5.14 Decision table for sequencing an uncertain bet

| Uncertainty | Cost of the experiment | Cost of being wrong | Delay cost | Sequence |
|---|---|---|---|---|
| High | Low | High | Low | Experiment first, then commit |
| High | Low | High | High | Experiment in parallel with reversible prep |
| High | High | High | Low | Narrow the experiment; test the cheapest assumption |
| High | High | High | High | Do not proceed; the option is too expensive |
| Low | Any | High | Low | Commit, but stage the commitment |
| Low | Any | Low | High | Commit now; speed dominates |
| Low | Any | Low | Low | Defer; it is not a priority |
| Any | Any | Low | Any | Bundle with related work or drop |

The fourth row is the one teams ignore. When the uncertainty is high, the experiment is expensive,
the downside is large, and the delay is costly, the correct decision is often not to play. A
roadmap that has no "do not proceed" outcome is not a decision tool.

## 6. Worked example

> **Note:** the company, the product, and all figures are a composite constructed for teaching.
> They are illustrative 2026 USD and not reported results.

### 6.1 The situation

A diagnostics software company has one product team and four quarters. Two opportunities compete
for the team, and one of them requires a large, irreversible commitment.

The product is an algorithm that predicts progression of a chronic condition. A pilot suggests it
works, with an estimated 55 percent probability that a definitive study will confirm an effect
large enough to support a commercial claim. If confirmed, the company can invest 4.50 million USD
in a scaled deployment that has a present value of 14.00 million USD at launch.

The study costs 1.20 million USD and takes two quarters. The scale-up takes two quarters and
requires the study's result.

### 6.2 The sequencing options

| Option | Description | Expected value (USD million) |
|---|---|---:|
| 1 | Build the scale-up now, without the study | 3.200 |
| 2 | Run the study, then scale if positive | 4.025 |
| 3 | Run the study, but accept a 1.50 million delay cost | 3.200 |
| 4 | Do nothing | 0.000 |

The arithmetic:

```text
Option 1: 0.55 x 14.00 - 4.50          = 3.20
Option 2: -1.20 + 0.55 x (14.00 - 4.50) = 4.025
Option 3: -1.20 + 0.55 x (14.00 - 1.50 - 4.50) = 3.20
```

The value of the evidence step is 0.825 million USD in expectation. That is less than the 1.20
million USD the study costs, which looks like a bad trade until you notice the asymmetry: if the
study is negative, the company avoids 4.50 million USD of scale-up spend, and that happens 45
percent of the time.

### 6.3 The delay tolerance

Sequencing costs time, and time costs value. The study is worth running only if the delay it
imposes destroys less than 1.50 million USD of the scale-up's present value.

| Delay cost (USD million) | Evidence first (USD million) | Big bet now (USD million) | Better |
|---:|---:|---:|---|
| 0.00 | 4.025 | 3.200 | Evidence |
| 0.50 | 3.750 | 3.200 | Evidence |
| 1.00 | 3.475 | 3.200 | Evidence |
| 1.50 | 3.200 | 3.200 | Indifferent |
| 2.00 | 2.925 | 3.200 | Big bet |
| 2.50 | 2.650 | 3.200 | Big bet |
| 3.00 | 2.375 | 3.200 | Big bet |

The break-even is 1.50 million USD, which is 10.7 percent of the scale-up's present value. That is
the number the leadership team should argue about: is two quarters of delay worth more or less than
10.7 percent of the launch value? That is a market question, not a modelling question, and it can
be answered with competitive intelligence rather than with more analysis.

### 6.4 Two diagnostics that reveal a weak roadmap

**Diagnostic 1: the sunk-cost exposure.** Building in parallel with the study exposes the full
5.70 million USD before the result is known, and the expected value falls to 2.00 million USD,
below both the sequential plan and the big bet. If the study is negative, the loss is 5.70 million
USD against 1.20 million USD for the sequential plan.

| Plan | Expected value (USD million) | Loss if negative (USD million) |
|---|---:|---:|
| Sequential | 4.025 | 1.200 |
| Parallel | 2.000 | 5.700 |
| Big bet | 3.200 | 4.500 |

The parallel plan is the worst of the three on both measures. It is also the plan that teams
default to when the roadmap is organised by function rather than by decision: engineering starts
building while the clinical team runs the study, because both are "on the roadmap," and nobody
sequences them.

**Diagnostic 2: the capacity arithmetic.** The roadmap contains five items totalling 62
team-weeks against 52 weeks of capacity, a 119 percent utilisation. The items are the outcome
study at 26 weeks, the scale-up build at 18, an interoperability fix at 8, audit logging at 6, and
a user interface refresh at 4.

| Item | Team-weeks |
|---|---:|
| Outcome study | 26 |
| Scale-up build | 18 |
| Interoperability fix | 8 |
| Audit logging | 6 |
| User interface refresh | 4 |
| **Total** | **62** |
| Capacity | 52 |
| Utilisation | 119 percent |

A roadmap at 119 percent is not a stretch plan; it is a plan that will fail in one of three ways.
The team will work overtime and burn out. The lowest-status item, which is usually the
interoperability fix, will be cut. Or the scale-up will start before the study concludes, which
converts the sequential plan into the parallel plan and destroys 2.03 million USD of expected
value.

That last mechanism is the dangerous one, because it happens without a decision. The roadmap is
over-committed, the study slips, and the build starts to "protect the date." The sequencing
discipline is destroyed by the capacity arithmetic, not by a bad argument.

### 6.5 Varying one assumption

The base case assumes a 55 percent probability that the study confirms the effect. Vary it.

| Probability | Evidence first (USD million) | Big bet now (USD million) | Advantage of evidence |
|---:|---:|---:|---:|
| 0.20 | 0.700 | -1.700 | 2.400 |
| 0.30 | 1.650 | -0.300 | 1.950 |
| 0.40 | 2.600 | 1.100 | 1.500 |
| 0.50 | 3.550 | 2.500 | 1.050 |
| 0.55 | 4.025 | 3.200 | 0.825 |
| 0.70 | 5.450 | 5.300 | 0.150 |
| 0.85 | 6.875 | 7.400 | -0.525 |

The advantage of the evidence step falls as the probability rises, and turns negative above about
0.80. That is the correct intuition: if you are almost certain the product works, testing is a
delay with little information value. The counter-intuitive part is how quickly the advantage
disappears: at a 70 percent probability it is only 0.15 million USD, which is within the noise of
the estimate.

The practical implication is that the sequencing decision should be made on the confidence range,
not the point estimate. If the team believes the probability is between 0.45 and 0.65, the evidence
step is worth running. If the team believes it is above 0.75, the case for testing is weak, and the
case for a staged commitment is stronger.

### 6.6 Parallel exploration with a constrained scale-up budget

The company can afford only one scale-up, at 4.50 million USD, and has 6.00 million USD of capital
for scale-ups, so a second success cannot be funded. It can, however, afford two studies at 1.20
million USD each.

| Plan | Studies | Probability of at least one success | Expected value (USD million) |
|---|---:|---:|---:|
| One study, one scale-up | 1 | 0.550 | 4.025 |
| Two studies, one scale-up | 2 | 0.798 | 5.176 |

Two studies and one scale-up is worth 1.15 million USD more than one study, even though the second
study has no funded scale-up behind it. The reason is that the second study is an option on being
able to choose the better of two programmes, and the probability of at least one success rises from
55 to 80 percent for an incremental cost of 1.20 million USD.

This is the real-options argument in its practical form: explore in parallel when exploration is
cheap and commitment is expensive, and concentrate the commitment on the best result. The
condition is that the scale-up capital is genuinely constrained; if both successes could be
funded, the calculation changes.

### 6.7 Ranked backlog for the quarter

The team also has a set of smaller items, scored with weighted shortest job first on a relative
1 to 20 scale.

| Item | Value | Time criticality | Risk reduction | Cost of delay | Size | WSJF |
|---|---:|---:|---:|---:|---:|---:|
| Audit logging | 3 | 13 | 8 | 24 | 5 | 4.80 |
| Interoperability fix | 8 | 3 | 5 | 16 | 5 | 3.20 |
| User interface refresh | 5 | 2 | 1 | 8 | 3 | 2.67 |
| Outcome study | 13 | 8 | 13 | 34 | 13 | 2.62 |

The outcome study ranks last, despite having the highest cost of delay, because its size is large.
That is the method working as designed and producing a bad answer for this portfolio: the study is
the item that unlocks the largest decision, and ranking it last would sequence the cheap work
ahead of the work that resolves the company's biggest uncertainty.

The lesson is not that WSJF is wrong. It is that a ratio method optimises flow, not portfolio
value, and a roadmap needs both. The resolution is to treat the study as a bet with its own
protected allocation, and to use WSJF to order the work within the committed band.

### 6.8 The roadmap

| Horizon | Item | Commitment | Confidence | Notes |
|---|---|---|---|---|
| Now, Q1 | Outcome study, first half | Committed, 1.20M USD | 80 percent on scope, 50 percent on result | Gate at end of Q2 |
| Now, Q1 | Audit logging | Committed, contract requirement | 90 percent | Hard external date |
| Now, Q2 | Outcome study, second half | Committed | 70 percent | Data access is the risk |
| Next, Q3 | Scale-up build if positive | Planned, conditional | 55 percent | 18 team-weeks |
| Next, Q3 | Interoperability fix | Planned | 70 percent | Depends on vendor |
| Later | Second programme study | Bet, 1.20M USD | Not estimated | Subject to capital |
| Later | User interface refresh | Direction | Not estimated | No date |
| Not doing | On-premises deployment option | — | — | Reason: no customer requirement; would consume 14 team-weeks |

The "not doing" row is the one that prevents the roadmap from being renegotiated every week. It
states a decision, with a reason, that stakeholders can argue with once rather than repeatedly.

### 6.9 The recommendation

The team should run the study first, hold the scale-up until the gate, and fund a second study in
parallel only if the capital for scale-ups remains constrained. It should publish the 1.50 million
USD delay tolerance so that the leadership team can debate the market assumption rather than the
model.

The roadmap should also state that the capacity plan is 52 team-weeks against 50 weeks of committed
and planned work, leaving a 4 percent buffer, which is below the 15 percent minimum. The gap should
be closed by deferring the interoperability fix to Q3, not by assuming the team will absorb it.

That final point is the most important output of the exercise. The arithmetic does not just rank
the items; it reveals that the plan is infeasible, and it names the item that must move.

## 7. Code, query, or calculation

The script computes sequencing expected values, the delay tolerance, the sensitivity to
probability, the parallel exploration case, WSJF rankings, and the capacity check.

```python
"""Sequencing under uncertainty, real-option value, and roadmap capacity."""

P = 0.55
STUDY_COST = 1.2
SCALE_COST = 4.5
SCALE_PV = 14.0


def ev_evidence_first(p=P, pv=SCALE_PV, study=STUDY_COST, scale=SCALE_COST):
    return -study + p * (pv - scale)


def ev_big_bet(p=P, pv=SCALE_PV, scale=SCALE_COST):
    return p * pv - scale


def ev_evidence_delay(delay, p=P):
    return -STUDY_COST + p * (SCALE_PV - delay - SCALE_COST)


print("SEQUENCING OPTIONS (USD millions)")
print(f"  big bet now          {ev_big_bet():7.3f}")
print(f"  evidence first       {ev_evidence_first():7.3f}")
print(f"  evidence, 1.5 delay  {ev_evidence_delay(1.5):7.3f}")
print(f"  do nothing            0.000")
print(f"  value of evidence    {ev_evidence_first() - ev_big_bet():7.3f}")

print("\nDELAY TOLERANCE")
break_even = None
for i in range(0, 401):
    d = i / 100
    if ev_evidence_delay(d) <= ev_big_bet():
        break_even = d
        break
print(f"  break-even delay     {break_even:.2f} USD million")
print(f"  as share of launch PV {break_even / SCALE_PV:.1%}")

print("\nSENSITIVITY: probability of success")
print("  p     evidence   big bet   advantage")
for p in (0.20, 0.30, 0.40, 0.50, 0.55, 0.70, 0.85):
    a, b = ev_evidence_first(p=p), ev_big_bet(p=p)
    print(f"  {p:4.2f}  {a:8.3f}  {b:8.3f}  {a - b:9.3f}")

print("\nPARALLEL EXPLORATION, ONE SCALE-UP")
n = 2
p_any = 1 - (1 - P) ** n
ev_two = -n * STUDY_COST + p_any * (SCALE_PV - SCALE_COST)
print(f"  probability of at least one success  {p_any:.4f}")
print(f"  EV two studies, one scale-up         {ev_two:7.3f}")
print(f"  EV one study, one scale-up           {ev_evidence_first():7.3f}")
print(f"  advantage                            {ev_two - ev_evidence_first():7.3f}")

print("\nSUNK-COST EXPOSURE")
parallel = -STUDY_COST - SCALE_COST + P * SCALE_PV
print(f"  parallel EV                {parallel:7.3f}")
print(f"  loss if negative, parallel {STUDY_COST + SCALE_COST:7.3f}")
print(f"  loss if negative, sequential {STUDY_COST:7.3f}")

print("\nWSJF (relative scores 1-20)")
items = {
    "Interoperability fix": (8, 3, 5, 5),
    "Outcome study": (13, 8, 13, 13),
    "UI refresh": (5, 2, 1, 3),
    "Audit logging": (3, 13, 8, 5),
}
rows = []
for name, (v, t, r, size) in items.items():
    cod = v + t + r
    rows.append((name, cod, size, cod / size))
print("  item                   CoD  size   WSJF")
for name, cod, size, w in sorted(rows, key=lambda r: -r[3]):
    print(f"  {name:22s} {cod:4d} {size:5d}  {w:6.2f}")

print("\nCAPACITY CHECK")
capacity = 4 * 13
plan = {"Outcome study": 26, "Scale-up build": 18, "Interoperability fix": 8,
        "Audit logging": 6, "UI refresh": 4}
used = sum(plan.values())
print(f"  capacity      {capacity} team-weeks")
print(f"  planned       {used} team-weeks")
print(f"  utilisation   {used / capacity:.1%}")
print(f"  buffer        {capacity - used} team-weeks")
for k, v in sorted(plan.items(), key=lambda kv: -kv[1]):
    print(f"  {k:22s} {v:3d}")
```

Expected output:

```text
SEQUENCING OPTIONS (USD millions)
  big bet now           3.200
  evidence first        4.025
  evidence, 1.5 delay   3.200
  do nothing            0.000
  value of evidence     0.825

DELAY TOLERANCE
  break-even delay     1.50 USD million
  as share of launch PV 10.7%

SENSITIVITY: probability of success
  p     evidence   big bet   advantage
  0.20     0.700    -1.700      2.400
  0.30     1.650    -0.300      1.950
  0.40     2.600     1.100      1.500
  0.50     3.550     2.500      1.050
  0.55     4.025     3.200      0.825
  0.70     5.450     5.300      0.150
  0.85     6.875     7.400     -0.525

PARALLEL EXPLORATION, ONE SCALE-UP
  probability of at least one success  0.7975
  EV two studies, one scale-up         5.176
  EV one study, one scale-up           4.025
  advantage                            1.151

SUNK-COST EXPOSURE
  parallel EV                2.000
  loss if negative, parallel 5.700
  loss if negative, sequential 1.200

WSJF (relative scores 1-20)
  item                   CoD  size   WSJF
  Audit logging           24     5    4.80
  Interoperability fix    16     5    3.20
  UI refresh               8     3    2.67
  Outcome study           34    13    2.62

CAPACITY CHECK
  capacity      52 team-weeks
  planned       62 team-weeks
  utilisation   119.2%
  buffer        -10 team-weeks
```

Two properties of the implementation are worth keeping. The delay tolerance is found by search
rather than by algebra, which makes it easy to change the model and rerun. And the capacity check
is a separate function of the plan, so an over-committed roadmap is visible before the sequencing
analysis rather than after.

Limitations: the model treats the study's cost and duration as certain, which they are not; a study
that overruns consumes the capacity that the scale-up needs. It also assumes the probability of
success is independent of the decision to run the study, which is false if the study is designed
to be confirmatory rather than exploratory.

## 8. Data, standards, and regulatory context

### 8.1 Regulatory milestones are fixed constraints

In a regulated product, the roadmap is anchored to external dates that cannot be moved: a
premarket submission, a clinical trial start, a quality audit, or a certification deadline. These
dates are not estimates and should not be treated as such.

| Milestone | Typical lead time | Consequence of missing |
|---|---|---|
| Premarket submission (510(k)) | 3 to 12 months including preparation | Launch delay |
| De novo or premarket approval | 12 to 36 months | Market access delay |
| Clinical trial start | 6 to 18 months including approvals | Evidence delay |
| Notified body review under EU MDR | 6 to 18 months | CE mark delay |
| ONC certification | 3 to 9 months | Eligibility for incentive programmes |
| SOC 2 Type II audit | 3 to 12 months including observation period | Procurement exclusion |

The roadmap should be built backwards from these dates. An engineering item that cannot fit before
a regulatory deadline should be moved, not compressed, because compression in a regulated process
produces a quality finding rather than a faster launch.

### 8.2 Design controls and lifecycle standards

If the product is a medical device, the design and development process is governed by design
controls in the Quality Management System Regulation at 21 CFR Part 820, and software lifecycle
processes by IEC 62304. Both require planned stages with reviews and documented outputs.

That structure maps onto the roadmap more directly than most agile frameworks admit. A design
review is a stage gate. A design input is a requirement. A design output is a specification. A
verification activity is a test. The roadmap's gates can satisfy these requirements if they are
documented in the right form, and a team that plans the roadmap without them will discover the
overhead at the audit.

### 8.3 Clinical governance and evidence planning

A product that changes clinical practice needs evidence, and evidence takes time that the roadmap
must include. The evidence plan should specify the study design, the endpoint, the sample size,
the analysis, and the decision it informs. A roadmap that places the study after the build has
reversed the order of value creation.

### 8.4 Data access and information blocking

In the United States, the information blocking rules at 45 CFR Part 171 affect how quickly data can
be obtained from a partner. A roadmap that assumes a data feed will be available in week two is
assuming away a regulatory and commercial process that can take months. The access milestone
belongs in the plan as a dependency, with an owner and a date.

### 8.5 Portfolio governance and stage gates

Institutions that fund a portfolio, rather than a single product, impose their own gates. A
hospital innovation fund, a payer's pilot programme, and a government grant scheme all have
application dates, reporting requirements, and continuation reviews. Those dates are external
constraints on the roadmap in the same way that regulatory milestones are.

The practical rule is to map the funder's calendar onto the roadmap before committing to a
sequence. A twelve-month grant with a six-month continuation review constrains the evidence step
to the first two quarters, whether or not that is the best scientific sequence. A roadmap that
ignores the funder's calendar will produce a strong application and a weak continuation.

### 8.6 What the standards do not settle

No standard tells you which opportunity to pursue. Regulatory and quality frameworks constrain the
sequence and the documentation, not the strategy. The strategic choice remains a judgement, made
with incomplete information, and reviewed when the information changes.

## 9. Industry practice and case evidence

### 9.1 Documented: cost of delay and weighted shortest job first

The scaled agile literature defines weighted shortest job first as cost of delay divided by job
size, with cost of delay composed of user-business value, time criticality, and risk reduction and
opportunity enablement, scored on relative scales. The method originates in Reinertsen's work on
product development flow, which argued that if you quantify only one thing, you should quantify
the cost of delay.

The method's documented weaknesses are consistent with the worked example: relative scores are not
additive, large items are penalised, and specialised resources and dependencies are ignored. The
practitioner literature acknowledges that the value is in the alignment conversation rather than
in the decimal precision of the score.

### 9.2 Documented: sequencing has measurable performance effects

Klingebiel and Adner examined alternative resource allocation regimes in product innovation and
identified three dimensions that distinguish real options logic: sequencing, low initial
commitment, and reallocation. They found that sequencing alone predicted higher new product sales,
while low initial commitment and reallocation mattered only as a matched pair. After controlling
for fit and sequencing, real options logic showed no performance advantage over other regimes
(Academy of Management Journal, 2014).

The finding is more precise than the popular version. It says that staging decisions is valuable,
and that staging without the willingness to reallocate is not. A roadmap with gates but no
consequence is exactly the case the research says does not work.

The corollary for roadmap design is that the gate must have an owner with the authority to move
people and money, not merely to approve continuation. If the gate decider cannot reallocate, the
organisation has adopted the vocabulary of real options without the mechanism, and the research
predicts no advantage from it.

### 9.3 Documented: parallel versus sequential testing

The operations literature on new product development compares sequential and parallel testing and
finds that the choice depends on the cost of testing, the cost of delay, and the correlation
between tests. Parallel testing reduces time to market and increases the cost of learning; the
optimal policy is not always sequential.

The worked example's result, that two parallel studies with one scale-up beat one study, is a
special case: it holds because the scale-up capital is constrained, so the second study buys an
option rather than a commitment.

### 9.4 Documented: the planning fallacy is systematic

The planning fallacy, the systematic tendency to underestimate the duration and cost of projects
even when similar projects have overrun, is one of the most replicated findings in judgement
research. Reference class forecasting, which estimates from the distribution of similar past
projects rather than from the specifics of the current one, is the standard corrective.

For a roadmap, the practical implication is that every estimate should be checked against a
reference class and adjusted upward, typically by 30 to 50 percent for software work.

### 9.5 Inference: what this implies for roadmap practice

The following is inference from the sources, not a documented finding.

First, the roadmap's most valuable section is the list of what is not being done. It is the only
part that prevents the plan from expanding to fill the available ambition.

Second, the capacity arithmetic should be checked before the sequencing analysis, because an
infeasible plan will be executed by compressing the sequence, which is the mechanism that destroys
the most value.

Third, gates need teeth. The research on real options logic suggests that sequencing without
reallocation does not produce a performance advantage, and a gate whose failure does not reallocate
capital is a status meeting.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Roadmap as a list | No capacity arithmetic | No constraint | Overcommitment | Reconcile to capacity |
| Dates on LATER items | Gantt chart four quarters out | False precision | Credibility loss after first miss | Horizon vocabulary |
| Over-utilisation | Buffer below 15 percent | Arithmetic infeasible | Slip or burnout | Reserve a buffer |
| Sequencing destroyed by capacity | Build starts before study ends | Overcommitment forces parallel work | Capital exposed to unresolved risk | Protect the gate |
| Ranking without portfolio | Items ranked one by one | Greedy selection | Suboptimal total | Test combinations |
| WSJF treated as truth | Score debated to one decimal | Hidden assumptions | Wrong item first | Audit the score |
| No kill criteria | Bets have no gates | Escalation of commitment | Sunk cost grows | Pre-agreed gate |
| Gate without consequence | Decision is "continue" regardless | Ratification | No learning | Name the reallocation |
| Platform deferred indefinitely | Debt items always last | Loudest work wins | Slowing delivery | Protected allocation |
| No "not doing" list | Every stakeholder assumes inclusion | Scope inflation | Renegotiation | Publish exclusions |
| Estimates without ranges | Single date per item | False confidence | Brittle plan | p50 and p90 |
| Regulatory date compressed | Engineering item squeezed before submission | Quality process cannot compress | Finding or delay | Move the item |

Two red flags are diagnostic. The first is a roadmap at more than 100 percent utilisation; it is
not ambitious, it is infeasible. The second is a bet with no stated kill criterion; it will become
a commitment through inertia, and the team will not notice the transition.

A third pattern is worth adding because it is specific to data products: the roadmap that treats
model performance as a deliverable. "Improve AUROC from 0.78 to 0.82" is a research objective, not
a roadmap item, because it does not specify whose decision changes or what the improvement is
worth. A roadmap item should name the decision and the metric that the buyer cares about, with the
model performance as an intermediate step. Otherwise the team can hit every technical milestone
and deliver nothing anyone can use.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | Capacity is the binding constraint | The plan is built on it | Money or approvals bind instead | Check the constraint |
| 2 | Focus factor is realistic | Capacity arithmetic depends on it | Meetings and support consume more | Measure actual focus time |
| 3 | Estimates can be referenced | Ranges need a basis | No comparable history | Build a reference class |
| 4 | Cost of delay can be estimated | Sequencing needs it | Value is speculative | Use a range and a threshold |
| 5 | The probability is estimable | Option value depends on it | No base rate | Use a range, test sensitivity |
| 6 | The gate will be honoured | Sequencing depends on it | Sponsor overrides | Name the decider and consequence |
| 7 | Dependencies are known | Sequence needs them | Undiscovered integration work | Map dependencies explicitly |
| 8 | Regulatory dates are fixed | Planning backwards requires them | Date moves | Confirm with the regulator |
| 9 | Reallocation is possible | Real options require it | Headcount is ring-fenced | Confirm authority to move people |
| 10 | The plan will be revised | A roadmap is a living document | It is published once and ignored | Set a review cadence |

The deepest limit is that a roadmap is a communication device as much as a plan. Its readers use it
to make their own commitments, and they will remember what it said. That is why the horizon
vocabulary matters more than the precision of the dates: a roadmap that is honest about confidence
survives being wrong, and a roadmap that is precise does not.

A second limit is that the analysis assumes the option set is known. In practice the best option
is often one that nobody proposed, and a roadmap process that only ranks submitted items will
systematically miss it. The remedy is a standing allocation for unrequested work, and a review
question that asks what is missing rather than only what should be cut.

## 12. Exercises

### Exercise 1. Commitments and bets

**Deliverable:** classify ten roadmap items as commitment, planned, bet, or direction, with the
evidence that justifies each.
**Criteria:** at least two bets identified; no bet in the NOW horizon; each commitment has an
external justification. **Time:** 30 minutes.

### Exercise 2. Capacity arithmetic

**Deliverable:** compute available capacity for a six-person team over a quarter at a 0.7 focus
factor, then reconcile a ten-item plan against it.
**Criteria:** focus factor applied; buffer stated; the item to defer named if the plan exceeds
capacity. **Time:** 45 minutes.

### Exercise 3. Cost of delay

**Deliverable:** compute cost of delay per week for three items with different decay patterns:
linear, urgent, and deferrable.
**Criteria:** each pattern represented; the urgent item's cost concentrated around its date; the
assumption about value stated. **Time:** 45 minutes.

### Exercise 4. Ranking audit

**Deliverable:** rank five items with WSJF, then write a one-page audit of what the ranking hides.
**Criteria:** scores reproducible; at least two hidden assumptions identified; a recommendation for
how to treat the top-ranked and bottom-ranked items. **Time:** 60 minutes.

### Exercise 5. Portfolio test

**Deliverable:** for four projects with costs, benefits, probabilities, and durations, find the
optimal portfolio under a capacity constraint and compare with the ratio-greedy result.
**Criteria:** enumeration or equivalent; the shortfall quantified; the reason for the difference
explained. **Time:** 60 minutes.

### Exercise 6. Sequencing memo

**Deliverable:** a two-page memo recommending a sequence for a bet with an evidence step and a
scale-up, including the delay tolerance.
**Criteria:** expected values computed; the delay tolerance stated; the conditions under which the
recommendation reverses. **Time:** 90 minutes.

### Exercise 7. Gate design

**Deliverable:** a gate specification for three bets, with date, metric, threshold, decider, and
consequence.
**Criteria:** each gate has a consequence that reallocates capacity; the decider is not the
proposer; the metric is measurable. **Time:** 45 minutes.

### Exercise 8. Roadmap communication

**Deliverable:** a one-page roadmap in now, next, later format with a "not doing" list and
confidence levels.
**Criteria:** no dates on later items; confidence stated for now items; at least three exclusions
with reasons. **Time:** 60 minutes.

## 13. Self-check answers

**1. What is the difference between a commitment and a bet?**
A commitment is resourced, dated, and expected to be delivered. A bet has a hypothesis, a test, a
cost cap, and a kill criterion. Commitments are governed by progress reviews; bets are governed by
gates.

**2. Why is a buffer necessary?**
Because unplanned work is certain: incidents, security findings, escalations, and regulatory
requests. Without a buffer, unplanned work displaces planned work, and the item displaced is
usually the platform investment that would have reduced future unplanned work.

**3. What is the difference between cost of delay and cost of duration?**
Cost of delay is the value lost per unit of time until delivery. Cost of duration is the cost of
occupying capacity per unit of time while the item is being built. A large valuable item can have
both high.

**4. Compute the cost of delay for an item worth 2.6 million USD per year with linear decay.**
2,600,000 / 52 = 50,000 USD per week.

**5. Why does WSJF rank large strategic items low?**
Because it divides cost of delay by job size, and a large item has a large denominator. The method
optimises flow, which favours small batches, and it does not account for the option value of an
item that unlocks a larger decision.

**6. Why can a portfolio outperform a greedy ranking?**
Because greedy selection by ratio can leave capacity unusable. In the Topic 08 example, greedy
selection produced 735,000 USD against an optimum of 870,000 USD, because a large item consumed
capacity that a better combination would have used.

**7. What are the four conditions that make an option valuable?**
Uncertainty, time to decide, a cheap experiment, and reversibility of the commitment.

**8. Compute the exercise gate threshold.** If a study costs 1.2 million USD and a scale-up costs
4.5 million USD with a present value of 14 million USD, the break-even probability is where
-1.2 + p x 9.5 = 0, so p = 0.126. The gate should be set above that, typically at a probability
that also covers the delay cost.

**9. Why is a 119 percent utilisation plan infeasible?**
Because the team cannot deliver more than its capacity without overtime or descoping. In practice
the plan is executed by compressing the sequence, which is how a sequential plan silently becomes a
parallel one.

**10. What is the "not doing" list and why does it matter?**
A published list of items excluded from the roadmap, with reasons. It prevents every stakeholder
from assuming their item is included and reduces repeated renegotiation.

**11. Why should regulatory dates be treated differently from estimates?**
Because they are external constraints that cannot be compressed by working harder. A quality
process that is compressed produces a finding, not a faster approval.

**12. Rubric for Exercise 6.** Full marks require computed expected values for at least three
options, a delay tolerance with its derivation, a stated reversal condition, and a recommendation
that names the cost of the evidence step. Half marks for a recommendation without the delay
arithmetic.

## 14. Interview questions

**1. How do you build a roadmap?**
Strong: starts from capacity and constraints, separates commitments from bets, sequences to buy
information, and publishes horizons with confidence. Weak: lists features by priority.

**2. How do you prioritise?**
Strong: names a method, states its assumptions, tests the result against the portfolio, and
describes what the method hides. Weak: says they use impact and effort.

**3. What is cost of delay?**
Strong: the value lost per unit of time, with linear, urgent, and deferrable patterns, and a worked
number. Weak: says it is urgency.

**4. What are the weaknesses of WSJF?**
Strong: relative scores are not additive, large items are penalised, dependencies are ignored, and
the score hides its assumptions. Weak: says it is subjective.

**5. How do you decide between building now and testing first?**
Strong: compares expected values, computes the delay tolerance, and states the probability at which
the answer reverses. Weak: says it depends on confidence.

**6. When would you run two experiments in parallel?**
Strong: when exploration is cheap, commitment is expensive or constrained, and only the best
option can be funded. Weak: says parallel is always faster.

**7. How do you handle an over-committed roadmap?**
Strong: recomputes capacity, names the item to defer, and explains the sequencing consequence of
not deferring. Weak: says they ask the team to work harder.

**8. What is a stage gate?**
Strong: a dated decision with a metric, a threshold, a decider, and a consequence that reallocates
capacity. Weak: says it is a review.

**9. How do you communicate uncertainty in a roadmap?**
Strong: horizons, confidence levels, and ranges; no dates on later items. Weak: says they add
caveats.

**10. How do regulatory milestones change sequencing?**
Strong: they are fixed external constraints, planned backwards from, and never compressed. Weak:
says they add time.

**11. How do you kill a project?**
Strong: pre-agreed kill criteria, a gate, a named decider, and a reallocation plan. Weak: says they
stop when it is clearly failing.

**12. How do you avoid the planning fallacy?**
Strong: reference class forecasting from comparable past projects, p50 and p90 ranges, and a
buffer. Weak: says they add 20 percent.

## 15. Key terms

- **Bet** — investment with uncertain outcome and a kill criterion.
- **Buffer** — capacity reserved for unplanned work.
- **CD3** — cost of delay divided by duration.
- **Commitment** — promise with resources and a date.
- **Cost of delay** — value lost per unit of time until delivery.
- **Cost of duration** — cost of occupying capacity per unit of time.
- **Critical path** — longest sequence of dependent work.
- **Cycle time** — time from start to completion.
- **Dependency** — work that must precede another item.
- **Escalation of commitment** — continuing a failing course because of prior investment.
- **Exploit** — work to capture validated value.
- **Explore** — work to reduce uncertainty.
- **Gate** — dated decision point with a metric and a consequence.
- **Horizon** — time band within which a statement is meaningful.
- **Job size** — duration or effort of an item.
- **Kill criterion** — pre-agreed condition that stops a bet.
- **Little's law** — throughput equals work in progress divided by cycle time.
- **Now, next, later** — horizon vocabulary for honest roadmaps.
- **Opportunity enablement** — value of an item in enabling future work.
- **Option value** — value of the ability to decide later.
- **Planning fallacy** — systematic underestimation of duration.
- **Platform work** — investment that enables multiple items.
- **Portfolio** — set of items competing for capacity.
- **Prioritisation** — choosing what to do and not do.
- **Reference class forecasting** — estimating from similar past projects.
- **RICE** — reach, impact, confidence, effort scoring.
- **Risk reduction** — value of an item in reducing uncertainty.
- **Roadmap** — sequenced plan of commitments and bets.
- **Sequencing** — ordering work to manage risk and learning.
- **Sunk cost** — cost already incurred and unrecoverable.
- **Technical debt** — cost of deferred rework.
- **Theme** — a direction without a scope.
- **Throughput** — work completed per period.
- **Time criticality** — how much value decays with delay.
- **Utilisation** — committed capacity divided by available capacity.
- **Work in progress** — items started and not finished.
- **WSJF** — weighted shortest job first.

## 16. Sources

All URLs accessed 26 October 2026.

1. Scaled Agile Framework. *WSJF — Extended Guidance*.
   https://framework.scaledagile.com/wsjf
2. Reinertsen DG. *The Principles of Product Development Flow: Second Generation Lean Product
   Development*. Celeritas, 2009.
3. Klingebiel K, Adner R. Real options logic revisited: the performance effects of alternative
   resource allocation regimes. *Academy of Management Journal*. 2014.
   https://doi.org/10.5465/amj.2012.0703
4. Keisler J, Mang C. Applying value of information and real options in R&D and new product
   development. *Wiley Encyclopedia of Operations Research and Management Science*. 2011.
   https://doi.org/10.1002/9780470400531.eorms0040
5. Wadeson N. Directed search with real options. *Economics of Innovation and New Technology*.
   2010. https://centaur.reading.ac.uk/16235/2/16235innovation_search.pdf
6. Ford DN, Sobek DK. Adapting real options to new product development by modeling the second
   Toyota paradox. *IEEE Transactions on Engineering Management*. 2005.
   https://doi.org/10.1109/tem.2005.844466
7. Koussis N, Martzoukos SH, Trigeorgis L. Multi-stage product development with exploration,
   value-enhancing, preemptive and innovation options. *Journal of Banking and Finance*.
   2013;37(1):174-190. https://ideas.repec.org/a/eee/jbfina/v37y2013i1p174-190.html
8. Highberg. *WSJF for Backlog Prioritization*. December 2023.
   https://highberg.com/insights/wsjf-for-backlog-prioritization/
9. US Food and Drug Administration. *Quality Management System Regulation*, 21 CFR Part 820.
   https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
10. International Electrotechnical Commission. *IEC 62304: Medical Device Software — Software Life
    Cycle Processes*. https://webstore.iec.ch/publication/6793
11. US Department of Health and Human Services. *Information Blocking*, 45 CFR Part 171.
    https://www.healthit.gov/topic/information-blocking
12. Office of the National Coordinator for Health Information Technology. *HTI-1 Final Rule*,
    2024. https://www.healthit.gov/topic/laws-regulation-and-policy/health-data-technology-and-interoperability-certification-program

Figures in Section 6 are illustrative and constructed for teaching. The real options findings are
summarised from the cited study; no company outcome is asserted.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_leadership.md`](08_leadership.md)
- First topic: [`01_product-discovery.md`](01_product-discovery.md)
- Related topic: [`03_health-economics.md`](03_health-economics.md)
- Module survey: [`../../course/17_product_economics_leadership.md`](../../course/17_product_economics_leadership.md)
- Labs: [`../../labs/`](../../labs/)
