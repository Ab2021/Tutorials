# Scenario and Sensitivity Analysis — which assumptions change the answer, and by how much

> Module 11 — Network Design and Optimization
> Topic 07 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 topics 04 and 05
> Version 2026.5

---

## 1. Why this topic exists

A network model returns one design. A committee has to commit capital to it. Between those two
facts sits every question the model cannot answer, and this topic is the discipline for
answering them in public rather than discovering them in year two.

The model from topic 04 is a deterministic object: it minimises fixed plus flow cost for one
demand level, one set of rates, one capacity per site, over one period. The output is a point.
The decision is not a point. It is a commitment that will be lived with while demand, wages,
freight, tariffs, and capacity all move. The gap between the point and the commitment is where
network studies fail, and the failures have a recognisable shape.

**The saving that evaporated.** A distribution study recommends consolidating from five sites
to three and reports a 9% cost reduction. Eighteen months later the company has three sites,
cost is 2% below the old baseline, and the operations director calls the study wrong. The study
was not wrong: its demand assumption was 12% above what materialised, and the three-site design
had been loaded to 96% of capacity, so the shortfall forced expediting, spot capacity, and a
service penalty the model never priced. A single sensitivity run at 0.85 times demand, reporting
that the design becomes infeasible below 0.94, would have changed the recommendation to four
sites at a cost of about 1.5%.

**The recommendation that reversed between meetings.** A European manufacturer runs the same
model twice, six months apart, and gets two different open sets. Nothing about the business
changed: the first run used a freight rate card at 2024 price levels and a labour cost at 2026;
the second used a 2026 rate card and a 2024 wage index. The transport-to-labour balance moved by
enough to reverse the ranking, and neither run reported the price year of each input, so nobody
could see the inconsistency. This is a reporting failure rather than a modelling failure, cured
by the parameter ledger that section 5 builds.

**The scenario set that agreed with itself.** A sponsor asks for scenarios. The analyst produces
a base case, an optimistic case, and a pessimistic case, all built by scaling demand up and
down by 10%. All three favour the sponsor's preferred site, and the study is used to justify a
decision made before it started. The failure is structural rather than dishonest: the set
varied one dimension, in one direction, around one anchor, and never varied the parameters that
would have questioned the answer. Section 5.3 gives the test that catches this.

Three things follow from these cases, and they organise the topic.

First, **the question the committee asks is not "what is the optimum" but "how wrong can you be
and still be right"**. That is a different question with different machinery: ranges, thresholds,
scenario sets, regret, and stability. The optimiser is necessary to answer it, and insufficient.

Second, **the useful output is a conditional statement, not a number**. "Open Q and R, at a cost
of 1,840 thousand per year, unless Q's avoidable fixed cost exceeds 1,020 thousand, in which
case open P and R at 1,950" is a recommendation. "The optimum is Q and R at 1,840" is a claim
that will be tested by events and will sometimes fail.

Third, **the discipline is about what you choose to vary, and what you refuse to vary**. A
sensitivity table with thirty rows is not more rigorous than one with six. It is usually less,
because thirty rows means nobody looked at which six matter.

> **Note:** this topic diagnoses the problem informally. It does not teach robust optimisation
> or stochastic programming. Those are the formal answers, and topic 08 owns them. The division
> of labour is deliberate: you cannot specify a robust model well until you know which
> parameters are worth hedging, and you cannot know that until you have done the analysis this
> topic describes.

**Relationship to the rest of the module.** Topic 04 owns the model being analysed and the
meaning of its coefficients; without that vocabulary the diagnostics in section 5 are opaque.
Topic 05 owns the multi-commodity and multi-period extensions, which multiply the parameter
count and make the ranked list in section 5, Step 3 more, not less, important. Topic 08 owns the
formal treatment of uncertainty. Topic 10 owns solver behaviour, including why a dual value
from a mixed-integer programme (MIP) is not the dual value of its linear relaxation. Module 20
owns simulation: simulation asks what a design does under a detailed operating model; this topic
asks which design, under which futures. A serious study uses both.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** sensitivity analysis from scenario analysis, and **explain** why a set of
   one-at-a-time runs systematically overstates confidence in a recommendation.
2. **Rank** the inputs of a network model by how much they actually move the answer, and
   **justify** the ranking with the structure of the model rather than with intuition.
3. **Read** dual values and reduced costs from the linear programming (LP) relaxation,
   **state** what they mean, and **name** the conditions under which the reading is unsafe.
4. **Explain** why right-hand side (RHS) ranging intervals are often so wide as to be useless,
   and **construct** a case where the reported range is wide while the true switch point is near.
5. **Design** a scenario set of three to six coherent futures, **name** the correlations between
   parameters in each, and **test** the set for the failure mode in which every scenario supports
   the preferred answer.
6. **Compute** and **present** a tornado chart, and **state** in one sentence what it does not
   show.
7. **Assess** decision stability — whether the same facility set is optimal across most
   scenarios — and **choose** between recommending a stable design and a cheaper unstable one.
8. **Compute** minimax regret for a small design/scenario matrix and **explain** when it should
   displace expected cost as the decision criterion.
9. **Write** a scenario and sensitivity section to the reporting standard in section 5.8,
   including the explicit statement of which scenario would flip the answer.

Each outcome maps to an exercise in section 12 and to a self-check answer in section 13.

## 3. The core mental model

Hold two pictures. The first is a tornado: a horizontal bar chart in which each parameter's
plausible range produces a range of outcomes, and the bars are sorted by length. The vertical
line down the middle is the base case. The second is a two-by-two that classifies what the
tornado and the scenario matrix together tell you about whether the recommendation is safe.

```text
                     total annual cost, thousand currency units per year
                       1,500   1,650   1,800   1,950   2,100   2,250
                          |       |       |       |       |       |
   fixed cost, all sites  |<=============|=====================>|
   demand volume          |    <========|===========>|          |
   labour rate            |      <======|======>|    |          |
   transport rate         |        <====|===>|    |    |          |
   capacity of Q          |            <=|=>|     |    |          |
                          |       |       |       |       |       |
                                      BASE 1,840

   each bar is the range of total cost when that parameter alone moves ±20 per cent
   and the model is re-solved; bars are sorted by length, longest at the top
   the vertical line is the base case; a bar that crosses the line changes the cost
   a bar that changes the OPEN SET is marked with a filled block, not just a length
```

```text
                          open set identical across the scenario set?
                                    YES                      NO
                    +---------------------------+---------------------------+
   the spread is    | SAME ANSWER EVERYWHERE    | OSCILLATION               |
   spread across    | recommend, and quote the  | two sets trade places as  |
   many scenarios   | tightest threshold        | parameters move; report   |
                    |                           | both and find the switch  |
                    +---------------------------+---------------------------+
   one scenario     | SINGLE DOMINANT SCENARIO  | NO STABLE ANSWER          |
   accounts for     | the set holds but the     | the answer is a bet on    |
   most of the      | cost is a bet on one      | one future; buy optional- |
   spread           | future; test its          | ity, phase the build, or  |
                    | credibility               | change the question       |
                    +---------------------------+---------------------------+
```

Six properties carry the rest of the topic.

**Property 1 — Sensitivity is a derivative; a scenario is a displacement.** A one-at-a-time
sensitivity answers "if this parameter moves by one unit, how much does the answer move, holding
everything else fixed". A scenario answers "if the world is like this, what is the answer". The
first is local and marginal. The second is global and joint. They are not two grades of the same
thing; they answer different questions, and a study needs both.

**Property 2 — One-at-a-time analysis overstates confidence, for two independent reasons.**
The first is arithmetic: real futures move several parameters at once, and the joint effect is
usually larger than any single effect. The second is structural: parameters are correlated, and
the correlation is often what matters. Demand and freight rates rise together in a boom; demand
falls and capacity becomes cheap in a recession; tariffs and transport costs move together when
trade policy tightens. A set of one-at-a-time runs samples the axes of the parameter space and
never samples its corners, which is where the decisions actually break.

**Property 3 — Cost is continuous and the open set is discrete, so a small cost change can
produce a large design change.** A 2% change in a fixed cost either does nothing or replaces a
site. The distribution of outcomes is therefore not smooth: it is a set of plateaus separated by
cliffs. Reporting a mean and a standard deviation of cost across scenarios describes the
plateaus and ignores the cliffs, which are the part the committee cares about.

**Property 4 — A threshold is more useful than an elasticity.** "A 10% increase in the transport
rate raises cost by 4.7%" is a fact about arithmetic. "Q stops being justified when its fixed
cost rises 12%, which is within the range the landlord is asking for" is a fact about a decision.
Always convert a sensitivity into the parameter value at which the answer changes, and then ask
whether that value is plausible.

**Property 5 — Probability is usually absent, and pretending otherwise is the most common
dishonesty in the genre.** A demand distribution estimated from three years of history is not a
distribution over the next ten years. A tariff scenario has no frequency interpretation at all.
When there are no defensible probabilities, expected cost is not a decision criterion; it is an
arbitrary weighted average wearing the costume of one. Use regret or robustness instead, and say
why.

**Property 6 — The answer has a shape, and the shape is the finding.** Across a parameter range,
the optimal cost typically looks like one of three things: a plateau, a threshold, or a cliff.
Naming the shape is more informative than any single number, and section 3.1 shows why.

### 3.1 Why the shape matters more than the number

Take the same network and sweep demand from 70 to 130 thousand units per year, re-solving at each
point. Three patterns are possible, and each implies a different recommendation.

| Shape | What the cost curve looks like | What it means for the decision | What to report |
|---|---|---|---|
| Plateau | Flat, then a step, then flat again | The design is robust over a wide band; the step locates the switch | The band and the step |
| Threshold | Smoothly rising, with a single crossing | Two designs trade places at a computable parameter value | The crossing value and its plausibility |
| Cliff | Flat, then a jump, then infeasible | The design is loaded to capacity; the failure mode is infeasibility, not cost | The infeasibility point, prominently |

The cliff is the one that destroys studies, because it does not appear in a cost table. A design
that becomes infeasible at 94% of forecast demand has a cost of "undefined" below that point,
and a report that quotes only the cost at forecast has hidden the most important fact about the
design. Section 6 shows a design of exactly this kind.

### 3.2 The correlation argument, in one small table

Suppose two parameters, demand volume and transport rate, each move over a low and a high value.
A one-at-a-time design tests three points: base, low demand, high transport. A joint design tests
the corners, and the corners are not equally likely.

| State of the world | Demand | Transport rate | Sampled by one-at-a-time? | Plausible? |
|---|---|---|---|---|
| Base | 1.00 | 1.00 | Yes | Yes |
| Slow growth, cheap fuel | 0.90 | 0.90 | No | Yes |
| Boom, expensive fuel | 1.10 | 1.15 | No | Yes |
| Stagflation | 0.90 | 1.15 | Partly | Yes |

Three of the four states are never visited by the one-at-a-time design, and two of those three
are the states in which a transport-heavy network and a fixed-heavy network diverge most. The
one-at-a-time design did not lie about any single parameter. It simply never asked the question
that the decision turns on.

> **Note:** correlation does not have to be estimated statistically to be used in scenario
> design. It has to be named. "In this scenario, demand falls and freight rates fall together
> because both are driven by the same industrial cycle" is a defensible sentence. It can be
> argued with. A correlation coefficient invented to two decimal places cannot.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Sensitivity analysis | The study of how the model output responds to a change in one input, all else held fixed | Used to mean any testing of assumptions | Method |
| Scenario analysis | Evaluation of the model under a coherent, jointly specified set of inputs representing a plausible future | Used as a synonym for sensitivity | Method |
| One-at-a-time (OAT) analysis | Varying one parameter at a time through a range, returning to the base between runs | Treated as a complete uncertainty analysis | Design |
| Local sensitivity | The derivative of the output with respect to an input at the base point | Reported as a global statement | Currency per unit of input |
| Global sensitivity | The distribution of output variation attributable to each input over its full range, including interactions | Confused with a wide OAT sweep | Method |
| Elasticity | The percentage change in output for a one per cent change in an input | Reported as a cost per unit change | Dimensionless |
| Base case | The single input set treated as most likely, against which changes are measured | Treated as a forecast | Parameter set |
| Tornado diagram | A sorted horizontal bar chart of output ranges produced by varying each input over its plausible range | Read as a probability statement | Chart |
| Scenario | One coherent future, specified as a complete input set | Used for a single parameter value | Parameter set |
| Scenario set | The collection of scenarios evaluated together, chosen to span the decision-relevant futures | Assembled ad hoc | Set |
| Coherent scenario | A scenario in which the parameter values are mutually consistent and jointly plausible | A scenario built by scaling one variable | Scenario property |
| Correlated parameters | Inputs that tend to move together, so that varying them independently misrepresents the joint distribution | Ignored in OAT designs | Correlation, or named narrative |
| Stress test | Evaluation under a specified adverse scenario, chosen to probe a particular vulnerability | Confused with a sensitivity run | Method |
| Stress scenario | A named, deliberately severe future used to test whether a decision survives | Chosen to be catastrophic rather than plausible | Parameter set |
| Decision stability | The degree to which the same open set remains optimal across the scenario set | Reported as a single number without a definition | Proportion of scenarios, or verdict |
| Oscillation | The phenomenon in which two or more designs alternate as optimal as parameters move | Treated as a modelling error | Pattern |
| Switch point | The parameter value at which two designs exchange rank | Confused with a break-even in accounting | Units of the parameter |
| Threshold | A parameter value beyond which the recommendation changes, including by becoming infeasible | Reported as an elasticity instead | Units of the parameter |
| Break-even | The parameter value at which two options cost the same | Used loosely for any threshold | Units of the parameter |
| Regret | The cost of a design in a scenario minus the cost of the best design in that scenario | Confused with loss or with variance | Currency per period |
| Minimax regret | The criterion that chooses the design whose worst-case regret is smallest | Confused with minimising worst-case cost | Decision rule |
| Maximin | The criterion that chooses the design with the best worst-case outcome | Confused with minimax regret | Decision rule |
| Expected cost | The probability-weighted average of a design's cost across scenarios | Used when the weights are not defensible | Currency per period |
| Robustness | The property of performing acceptably across a wide range of scenarios without a probability model | Used as a synonym for "good" | Property |
| Robust deviation | The maximum relative or absolute shortfall from the scenario optimum | A formal robustness measure | Currency or proportion |
| p-robust | A design whose relative regret is at most `p` in every scenario | Treated as an optimality claim | Proportion |
| Dual value | The rate of change of the LP optimal objective with respect to a constraint's right-hand side, locally | Read from a MIP as if it were an LP dual | Currency per unit of RHS |
| Shadow price | The economic interpretation of a dual value: the marginal value of one more unit of a resource | Used interchangeably with dual value without stating the caveat | Currency per unit |
| Reduced cost | The amount by which a variable's objective coefficient must improve before it enters the LP basis | Misread as a fixed cost | Currency per unit |
| RHS ranging | The interval over which one right-hand side may move without changing the LP basis | Treated as the interval over which the answer is unchanged | Interval in RHS units |
| Allowable increase and decrease | The upper and lower bounds of the RHS ranging interval | Reported without the basis caveat | Units of the RHS |
| Degeneracy | A condition in which a basic variable is at zero, so the dual solution is not unique and dual values are not unique shadow prices | Ignored when reading duals | Model state |
| Binding constraint | A constraint satisfied with equality at the solution | Assumed to imply a positive shadow price | Constraint state |
| Slack | The difference between the right-hand side and the left-hand side at the solution | Confused with spare capacity that can be sold | Units of the constraint |
| LP relaxation | The MIP with its integer variables allowed to take fractional values | Treated as a solution rather than a bound | Model |
| Scenario probability | A weight attached to a scenario for expected-value calculations | Asserted without evidence | Proportion |
| Plausibility | The judgement that a scenario could occur, independent of its probability | Confused with likelihood | Judgement |
| Decision rule | The stated criterion by which a design is chosen from the scenario matrix | Left implicit | Rule |
| Sample average approximation (SAA) | Estimating an expected-value objective by averaging over a sampled scenario set | Treated as exact | Method |

Three of these deserve expansion, because they are the ones most often used loosely in a meeting.

**Sensitivity versus scenario.** Sensitivity holds the world still and moves one dial. Scenario
moves the world. The practical test: if the parameter values in a run could not plausibly coexist,
it is a sensitivity, not a scenario. A run with demand down 20% and freight rates up 20% might be
a scenario (stagflation) or might be nonsense, depending on the industry. The analyst has to say
which, and why.

**Regret versus risk.** Regret is measured against the best you could have done in that state,
not against your expectation. It is therefore a statement about opportunity cost, and it is the
natural currency of a capital committee that will be asked "why did we not build the other one".
Minimax regret is the criterion that bounds the worst such question. It is not the same as
minimising worst-case cost, which is a different and usually more conservative rule.

**Dual value versus shadow price.** A dual value is a number the solver returns. A shadow price
is an economic claim that one more unit of the constrained resource is worth that number. The
claim is valid under conditions that section 5, Step 5 states and that degeneracy often violates.
The distinction is not pedantry: a capacity shadow price is routinely used to justify a capital
request for expansion, and if it is wrong the request is wrong.

## 5. The mechanics, step by step

This is the procedure. It takes a solved network model and produces a recommendation with
stated conditions. Eleven steps, each with an input, an output, and the branch that sends the
work back.

### Step 1 — Fix the decision and the metric before touching a parameter

**Input and output:** the decision sentence from topic 01 becomes one sentence naming the
design choice, the cost metric, the period, and the criterion by which designs will be compared.

**Why it matters:** a sensitivity analysis without a fixed metric will report five numbers that
move in different directions, and the meeting will pick the one that supports the preferred site.
Decide in advance whether the metric is annual total cost, net present value (NPV) over the
horizon, cost per unit shipped, or cost plus a service penalty. They rank designs differently,
and the difference is not a detail. NPV in particular discounts distant costs, which
systematically favours designs with high fixed cost and low variable cost, because the fixed
cost is incurred early and the variable saving accrues late.

**Branch:** if the metric is cost per unit, note that it falls automatically when volume rises.
A design can look better in a high-demand scenario for arithmetic reasons alone. Report totals.

### Step 2 — Build the parameter ledger

**Input and output:** every number that entered the model becomes a row in a ledger with value,
unit, source, price year, owner, and plausible range.

**Why it matters:** the ledger is the artefact that makes the rest of the analysis auditable.
Without it, a reviewer cannot tell whether a range came from a contract, an index, or a guess,
and the ranges are where the conclusions live. A minimal ledger has eight columns: parameter,
symbol, value, unit, source document, price year, owner, and low/high range with the basis for
each bound. If a range has no basis beyond "it felt right", mark it as a judgement and state who
made it; an unlabelled one is not legitimate.

### Step 3 — Rank the inputs by whether they can move the answer

Not all parameters matter, and the ones that matter are determined by the structure of the
model rather than by the industry. The table below is the working rank order. It is about
network design specifically; other model classes rank differently.

| Rank | Input | Why it moves the answer | Typical plausible range | Usually changes the open set? |
|---:|---|---|---|---|
| 1 | Demand volume | Enters every demand constraint; drives capacity feasibility | −20% to +30% over a strategic horizon | Yes, especially near a capacity ceiling |
| 2 | Demand geography | Changes the cost matrix and which sites are central | Relocation of 5% to 15% of volume | Yes, if the shift is directional |
| 3 | Fixed cost estimates | Traded directly against transport; sets the number of sites | −25% to +25%, mostly downward after allocation is stripped | Yes, often |
| 4 | Transport rates | The main variable cost in distribution networks | −15% to +40%, wider for ocean | Sometimes, when fixed and flow cost are comparable |
| 5 | Capacity ceilings | Decide feasibility, not just cost | −10% to −25% against nameplate assumptions | Yes, by making designs infeasible |
| 6 | Duty and tariff structure | Can remove a cross-border option entirely | Step changes, not percentages | Yes, and discontinuously |
| 7 | Labour rates | Affect both fixed site cost and variable handling | −10% to +30% by region | Sometimes |
| 8 | Energy prices | Affect handling, refrigeration, and line-haul | −30% to +100% | Rarely alone; matters with carbon pricing |
| 9 | Lead times and service windows | Restrict arcs rather than move cost | Days to weeks | Yes, by deleting arcs |
| 10 | Inventory carrying rate | Scales the inventory term if modelled | 15% to 30% per year | Rarely |
| 11 | Product mix within a family | Changes cube and handling, not usually cost per unit | Mix shifts of a few points | Rarely at family level |
| 12 | Corporate overhead allocation | Does not disappear on closure | Accounting policy | No, and should never be in `f_j` |
| 13 | Discount rate | Re-weights fixed against variable | 6% to 12% | Rarely, but changes the ranking |
| 14 | Site land and construction cost | Part of the fixed cost | Wide, but already inside rank 3 | Only as part of fixed cost |
| 15 | Fleet and asset residual values | Transition cost, not steady-state | Wide | No, in a steady-state model |

Two readings of this table matter. **The top six are where the study's time should go.** A
sensitivity analysis that spends equal effort on all fifteen has not understood the model. **The
bottom four are where the arguments happen**: finance and operations own them, and they are
usually the least consequential.

### Step 4 — Take the diagnostics from the LP relaxation

**Input and output:** the mixed-integer linear programme (MILP) and its LP relaxation, solved
separately, yield dual values, reduced costs, slacks, and a bound.

**Why it matters:** the relaxation is the only place where marginal values exist. A MIP has no
dual values, because its feasible region is not convex. The relaxation's duals tell you which
constraints are scarce in the continuous world, and that is a useful first pass at where the
design is fragile.

Three diagnostics to read:

- **Dual value on a capacity constraint.** Currency per unit of capacity per period. It says
  what one more unit of throughput at that site would be worth if capacity were divisible and
  the site were open.
- **Reduced cost on a closed site.** A negative reduced cost means opening that site would
  improve the relaxation by that amount per unit of its open variable. It is a screening device
  for "which site did the relaxation nearly choose", not a recommendation.
- **Slack on a capacity constraint.** Zero slack with a positive dual means the constraint binds.
  Zero slack with a zero dual means something is degenerate, and the next step explains why that
  matters.

### Step 5 — Interpret the duals without over-reading them

**Why it matters:** a dual value is routinely quoted as "capacity at Q is worth 4 per unit", and
that sentence is used to justify a capital request. It is true under conditions that a network
model frequently violates.

The reading is safe when the LP optimum is unique and non-degenerate. It is unsafe in three
situations.

1. **Degeneracy.** If a basic variable sits at zero, the dual solution is not unique, and
   different solvers, or different pivoting orders in the same solver, return different duals
   for the same problem. The number you see is one member of a set of optimal duals. The true
   one-sided shadow prices are the minimum and maximum over that set, and a dual of zero can
   coexist with a genuinely valuable constraint. The classical treatment is Gál (1986); Ho
   (2000) gives the practical test: if the allowable increase on the right-hand side is zero,
   the reported dual understates the value of an increase, and if the allowable decrease is
   zero, it understates the value of a decrease.
2. **Integrality.** The duals belong to the relaxation, in which a site can be 60% open. A
   capacity dual from the relaxation is not the value of capacity in the design you will build.
   It is an upper bound on what you should pay for extra capacity, and often a loose one.
3. **Non-binding but structural constraints.** A service window that is satisfied with slack has
   a zero dual, and the zero does not mean the constraint is unimportant. It means it is not
   binding at this solution. Remove it and the design may change completely, because the
   constraint defines the feasible set rather than pricing a marginal unit.

**The discipline:** quote a dual with its constraint, its units, its basis, and a sentence on
whether the problem is degenerate. If you cannot state the basis, do not quote the dual.

### Step 6 — Run RHS ranging, then ask whether the range is useful

**Input and output:** for each binding constraint, the allowable increase and decrease on its
right-hand side, converted into the units a manager thinks in.

**Why it matters:** ranging is the oldest form of sensitivity analysis in linear programming,
and it is the one most often reported and least often useful. The interval tells you how far one
number can move before the optimal basis changes. Two things routinely go wrong.

**The range is often absurdly wide.** In the worked example below, the transport rate would have
to rise by 137% before the recommended design stops being optimal. A range of that width is not
reassurance; it is evidence that the transport rate is not the parameter that decides this
design. Report the width as a finding, not as a certificate of robustness.

**The range is about the basis, not the design.** A basis can stay optimal while the integer
open set changes, and an open set can stay the same while the basis changes many times. The two
intervals are different objects. Ranging is a statement about a continuous relaxation's
optimal basis, and the decision is a statement about a set of binary variables.

> **Note:** in a degenerate LP, one side of the ranging interval is zero. A zero allowable
> increase does not mean the constraint cannot be relaxed; it means the basis changes
> immediately, and the dual you were given does not price the change. This is the single most
> common source of a wrong capacity valuation in a network study.

### Step 7 — Run the one-at-a-time sweep, and record the open set as well as the cost

**Input and output:** each parameter in ranks 1 to 6 moves over its plausible range, alone, with
the model re-solved at each point. The output is a table of parameter, value, total cost, open
set, and whether the open set changed.

**Why it matters:** this is the tornado data, and the discipline is in what gets recorded. A
sweep that records only cost will show a smooth curve and hide the discontinuity. Record the
open set as a categorical field, and mark every point at which it changes. Those markers are the
thresholds, and they are the output the committee will use.

### Step 8 — Design the scenario set

**Input and output:** the ledger, the ranked inputs, and a named narrative for each future
become three to six scenarios, each a complete parameter set with its correlations stated.

**Why it matters:** this is where the analysis is won or lost, and it is a design problem, not a
sampling problem. Four rules make a set defensible.

1. **Build scenarios from narratives, not from percentages.** "A tariff regime that prices
   carbon at the border, with demand flat and freight elevated" is a scenario. "Base plus 10%"
   is a dial position.
2. **State the correlations explicitly.** Which parameters move together, and why. A scenario
   without a stated correlation structure is a set of independent draws wearing a costume.
3. **Include one scenario that hurts the preferred design.** If the preferred design is the
   comfortable one, the set must contain the future in which it fails. This is a rule about
   intellectual honesty, and it is also the only way to find the threshold.
4. **Include the current network as a design, in every scenario.** It is the option of doing
   nothing, and its regret is the number the committee will compare against.

| Scenario archetype | What it varies | What it tests | Typical correlation |
|---|---|---|---|
| Base case | Nothing; the central estimate | Whether the model reproduces the expected answer | None |
| Demand-led growth | Volume, geography, mix | Whether capacity is adequate | Volume and freight rise together |
| Demand-led contraction | Volume, geography, fixed cost flexibility | Whether fixed cost can be taken out | Volume and wages fall together |
| Cost shock | Transport, energy, labour | Whether the design is too transport-heavy or too fixed-heavy | Energy and freight rise together |
| Policy shock | Tariff, duty, carbon, compliance | Whether cross-border options survive | Tariff and compliance cost rise together |
| Capacity shock | Capacity ceiling, lead time, disruption | Whether the design is feasible at all | Capacity falls and lead time rises |
| Structural break | Candidate set, technology, channel mix | Whether the model's structure is still right | Not a correlation; a different model |

### Step 9 — Run the matrix, and compute regret rather than only cost

**Input and output:** designs down the rows, scenarios across the columns, cost in each cell,
then a regret column and a worst-case column.

**Why it matters:** the raw cost matrix invites a comparison of each design to the base case.
The regret matrix invites the comparison that matters: within each scenario, how much worse is
this design than the best available. Regret is measured in the same units as cost and has a
direct interpretation — "if this future happens, we will have spent 158 thousand per year more
than we needed to".

```
regret(d, s) = cost(d, s) - min over designs d' of cost(d', s)
```

A cell with an infeasible design has no finite regret. Do not fill it with a large number unless
you state the penalty and defend it; a penalty is a probability judgement in disguise.

### Step 10 — Test decision stability and name the flip

**Input and output:** the open set per scenario becomes a stability verdict and a named switch
scenario.

**Why it matters:** stability is often worth more than optimality. A design that is optimal in
the base case and third in two other scenarios will be reversed within two years, and the
reversal costs more than the 2% of annual cost that the optimum bought. Count how many scenarios
each candidate design wins, how many it is feasible in, and what its worst rank is. Then name
the single scenario that would flip the recommendation, and ask whether that scenario is
credible. Section 5.7 gives the test.

### Step 11 — Choose the criterion and write to the standard

**Input and output:** the matrix, the regret column, and the probability judgement become a
decision rule and a written recommendation with conditions.

**Why it matters:** the criterion is a management choice, not an analytical one, and it should
be visible. Minimising expected cost, minimising worst-case regret, and maximising the number of
scenarios won are three different rules that can select three different designs from the same
matrix. State which one you used and why.

### 5.1 Why one-at-a-time systematically overstates confidence

The overstatement has three sources, and only the first is widely recognised.

**Source 1 — Additivity is assumed and rarely true.** If cost were linear in the parameters, the
joint effect of two changes would be the sum of the separate effects and an OAT study would be
complete. Network cost is piecewise linear at best and discontinuous at the open set. Near a
threshold, two moderate changes can combine to cross it when neither would alone; the OAT
analysis reports two small effects and misses the crossing.

**Source 2 — The corners are never sampled.** In `k` parameters an OAT design visits `2k + 1`
points, while the corners of the box number `2^k`. For `k = 10`, that is 21 points against 1,024.
The unsampled region is not uniformly unimportant; it is exactly where several parameters
combine, which is where real futures live.

**Source 3 — The range is chosen around the base, so it cannot reveal a structural break.** A
±20% band assumes the base is roughly right and asks how far off it can be. Some consequential
futures are not within ±20% of anything: a tariff that goes from zero to a positive rate, a site
that becomes unavailable, a channel that moves from 10% to 40% of volume. A percentage band
cannot represent a regime change, and a study built only from percentage bands will report a
comfortable answer to the wrong question.

### 5.2 The range that tells you nothing

In the section 6 instance the recommended design has a flow cost of 430 and a fixed cost of
1,410; the alternative has a flow cost of 350 and a fixed cost of 1,600, so the recommended design
wins by `(1,600 - 1,410) - (430 - 350) = 110`. Raising all transport costs by a factor `k` changes
both flow terms proportionally, and the ranking changes only when `1,410 + 430k = 1,600 + 350k`,
which gives `k = 2.375`, a rise of 137.5%.

The ranging report prints that width, and a reader concludes that transport is not a risk. The
narrower conclusion is that transport is not the switching parameter, because fixed cost is
roughly three times flow cost; reverse the ratio and the same percentage move flips the answer.
**The width of the range is a statement about the cost structure, not about the world.**

### 5.3 How many scenarios are enough

There is no correct number, and the literature does not converge on one. What it does converge
on is a design principle: enough scenarios to span the decision-relevant futures, few enough
that a person can hold them in mind and argue with each one.

| Source | Guidance | Basis |
|---|---|---|
| McKinsey scenario practice (2009) | At least four | With three, people choose the middle one; four forces a view |
| McKinsey crisis planning (2021) | At least three to four for deterministic work | CFO survey practice |
| Scenario planning review literature (Amer, Daim and Jetter, 2013) | No precise optimum; commonly three to six | Review of methods and practice |
| Practitioner futures work | Three to five | Cognitive load; distinctness |

The useful distinction is between a **designed set** and a **sample**. A designed set of four to
six scenarios, each built from a narrative, is the right tool for a capital decision, because
each scenario can be argued with and each has a name that survives into the minutes. A sample of
hundreds or thousands, generated by Monte Carlo or by sample average approximation, belongs with
topic 08. Mixing the two produces the worst of both: too many scenarios to discuss, too few to
estimate anything.

> **Note:** the failure mode to test for is not "too few scenarios". It is "every scenario
> supports the answer". Six scenarios all on the same side of the switch point have the
> appearance of rigour and the content of a press release. The test: identify the scenario in
> which the recommended design is worst, and check whether it is in the set. If it is not, the
> set is incomplete by construction.

### 5.4 The probability question, answered honestly

Most network studies have no defensible probabilities, and the honest response is to say so and
change the criterion rather than to invent weights. There are three positions, and each is
legitimate in the right setting.

1. **Probabilities exist and are defensible.** Demand for a mature product with a stable
   history, in a stable market, over a two-year horizon. Then expected cost is meaningful and
   sample average approximation is the right estimator. Even here the distribution is over the
   *near* future, and the strategic horizon usually outruns it.
2. **Probabilities do not exist, but plausibility judgements do.** A tariff regime, a technology
   shift, a competitor's network decision. Then use minimax regret or a robustness criterion,
   and state the judgements that shaped the scenario set.
3. **Neither exists, and the decision is a bet.** Then the honest output is conditional
   recommendations and a portfolio logic: which design is cheap to reverse, which keeps optional
   capacity, which can be phased.

The failure to avoid is the middle one dressed as the first: three scenarios, equal weights
because equal weights feel neutral, an expected cost computed to the currency unit, and a
recommendation inheriting the authority of arithmetic it does not have. Equal weighting is not
neutral; it is a strong claim that all three futures are equally likely.

### 5.5 Reading a tornado chart honestly

A tornado chart shows the size of the effect of each parameter, one at a time, over a stated
range. It does not show probability, it does not show interaction, and its bar lengths depend
entirely on the ranges chosen. Three rules keep it honest.

- **State the range basis on the chart.** "±20% around the 2026 plan" is a range basis.
  "Plausible range" is not, unless the plausibility argument is in the appendix.
- **Mark the base case and the switch points.** A bar that crosses the base line changes the
  cost; a bar that changes the open set is a different kind of event and should be marked so.
- **Do not sort by bar length and call the top bar the biggest risk.** The top bar is the
  parameter with the largest cost effect over the range you chose. Risk is effect times
  plausibility, and plausibility is not in the chart.

### 5.6 The stability test, stated precisely

For each scenario `s` and each candidate design `d`, you have a cost `C(d, s)` or a marker that
`d` is infeasible in `s`. Then:

- **Win count** — the number of scenarios in which `d` is the cheapest feasible design.
- **Feasibility count** — the number of scenarios in which `d` is feasible at all.
- **Worst rank** — the worst position `d` occupies among feasible designs.
- **Regret column** — `C(d, s)` minus the cheapest feasible cost in `s`, by scenario.
- **Max regret** — the largest finite regret. Infinite if `d` is infeasible anywhere.

A design with a high win count, full feasibility, and a small max regret is stable. A design
with a high win count but an infeasibility in one plausible scenario is not stable; it is lucky.
The distinction is the whole point of the test, and it is why feasibility must be recorded
separately from cost.

### 5.7 The stress test, stated precisely

A stress test is not a scenario and not a sensitivity. It is a question of the form: **what would
have to be true for this recommendation to be wrong, and is that thing credible**.

The procedure has four steps.

1. **Name the flip.** From the threshold analysis, identify the parameter value at which the
   recommendation changes, and the scenario in which it occurs.
2. **State the magnitude in the units the business uses.** "Q's fixed cost rises 12%, from 910
   to 1,020 thousand per year" is testable against a lease quote. "Fixed costs are uncertain"
   is not.
3. **Test the credibility.** Is 1,020 within the range of quotes the landlord has given? Is a
   5% demand increase inside the forecast error the business already experiences? Is a 9%
   capacity derating consistent with the site's demonstrated throughput in a bad month?
4. **Report the result in one of three forms.** The flip is not credible, and the
   recommendation stands with the threshold quoted. The flip is credible, and the recommendation
   changes. The flip is credible but insurance costs more than being wrong, and the
   recommendation stands with the insurance priced.

The third form is the one most often missed, and the most useful. A design that is 2% more
expensive and survives the credible downside is usually the right recommendation, and the 2% is
the premium.

### 5.8 What the reporting standard requires

The section of the report that carries this analysis has eight parts. A study that omits more
than one of them cannot be reviewed.

| # | Item | What must be stated |
|---:|---|---|
| 1 | Parameter set | Every parameter varied, with unit, source, and price year |
| 2 | Ranges | The low and high value for each, and the basis for each bound |
| 3 | Scenario definitions | Each scenario as a complete parameter set, with correlations named |
| 4 | Scenario count and rationale | Why this many, and what the set is designed to span |
| 5 | Stability finding | Win count, feasibility count, worst rank, by design |
| 6 | Regret table | Regret by design and scenario, with the criterion stated |
| 7 | The flip | The single scenario or threshold that would change the recommendation |
| 8 | The criterion | Expected cost, minimax regret, or robustness, and why |

Item 7 is the one that separates a study from a sales document. A recommendation that cannot
name the condition under which it is wrong has not been analysed; it has been asserted.

## 6. Worked example

Meridian Industrial distributes one product family from three candidate distribution centres
(DCs) to four regional markets. The instance is small enough to solve by hand and is built so
that the interesting phenomena — a near-threshold design, a parameter that flips at 12%, and a
parameter whose range is uselessly wide — are all visible. All figures are illustrative and
internally consistent. Costs are in thousands of currency units per year at 2026 prices.

### 6.1 The instance

| Market | Demand, thousand units/yr | Share |
|---|---:|---:|
| M1 | 35 | 35% |
| M2 | 30 | 30% |
| M3 | 20 | 20% |
| M4 | 15 | 15% |
| **Total** | **100** | **100%** |

| Site | Capacity, thousand units/yr | Fixed cost, thousand/yr | Fixed cost per unit of capacity |
|---|---:|---:|---:|
| P | 80 | 1,100 | 13.75 |
| Q | 55 | 910 | 16.55 |
| R | 50 | 500 | 10.00 |
| **Total** | **185** | **2,510** | |

| Unit cost, currency per unit | P | Q | R |
|---|---:|---:|---:|
| M1 | 3 | 5 | 9 |
| M2 | 4 | 3 | 8 |
| M3 | 8 | 6 | 4 |
| M4 | 9 | 7 | 3 |

Each unit cost includes line-haul, delivery, and variable handling. The fixed cost is the
avoidable annual cost and excludes allocated overhead.

### 6.2 The five feasible open sets

Three sites give seven non-empty subsets; the three single-site subsets are infeasible because
no site carries 100 thousand units.

| Open set | Capacity | Fixed | Constrained allocation | Flow | Total |
|---|---:|---:|---|---:|---:|
| {P, Q} | 135 | 2,010 | M1→P 35; M2→P 10, Q 20; M3→Q 20; M4→Q 15 | 430 | 2,440 |
| {P, R} | 130 | 1,600 | M1→P 35; M2→P 30; M3→R 20; M4→R 15 | 350 | 1,950 |
| {Q, R} | 105 | 1,410 | M1→Q 25, R 10; M2→Q 30; M3→R 20; M4→R 15 | 430 | **1,840** |
| {P, Q, R} | 185 | 2,510 | M1→P; M2→Q; M3→R; M4→R | 320 | 2,830 |
| {P} / {Q} / {R} | 80 / 55 / 50 | — | Infeasible: capacity below 100 | — | — |

The winner is {Q, R} at 1,840, a margin of 110 over {P, R}, or 6.0%. The margin is small, and
that is deliberate: a design with a 6% margin and no capacity slack is exactly the kind of
answer a committee is asked to commit to, and exactly the kind that sensitivity analysis exists
to interrogate.

Note the shape of {Q, R}. Its capacity is 105 against demand of 100, so it is 95.2% utilised in
aggregate, and Q itself is at 55 out of 55, which is 100%. Q's capacity binds, and R's does not.
The design has five thousand units of aggregate slack, all of it at R, and none of it where the
flow wants to go. This fact drives everything in sections 6.4 and 6.5.

A diagnostic from the relaxation of the fixed-open problem: with Q and R forced open, the dual
on Q's capacity is 4 currency units per unit of annual throughput. One more unit of Q capacity
moves one unit of M1 demand from R (cost 9) to Q (cost 5), saving 4. That dual is the price above
which expanding Q is not worth it, and it is also the number that explains the capacity
sensitivity in section 6.4.

### 6.3 The four-parameter one-at-a-time sweep

Each parameter moves ±20% alone, with the model re-solved at every point. "Cost" is the total of
the re-optimised design, and "open set" is the design that achieves it.

| Parameter | −20% | Cost at −20% | +20% | Cost at +20% | Swing | Open set changes? |
|---|---:|---:|---:|---:|---:|---|
| All fixed costs | 0.80× | 1,558 | 1.20× | 2,122 | 564 | No |
| Demand volume | 80 k units | 1,722 | 120 k units | 2,020 | 298 | Yes, at +20% |
| Labour rate | 0.80× | 1,692 | 1.20× | 1,988 | 297 | No |
| Transport rate | 0.80× | 1,754 | 1.20× | 1,926 | 172 | No |
| Q capacity | 44 k units | 1,950 | 66 k units | 1,800 | 150 | Yes, at −20% |

The arithmetic, so that it can be checked.

**Fixed costs, all sites.** Scaling all three fixed costs by `s` scales both candidates' fixed
terms: at `s = 0.8`, {Q, R} costs `0.8 × 1,410 + 430 = 1,558`; at `s = 1.2`,
`1.2 × 1,410 + 430 = 2,122`. The alternative {P, R} costs 1,630 and 2,270. No flip, because the
two sets differ by only 190 in fixed cost and scaling both changes that difference by 38 at most.

**Demand volume.** At 0.8×, demand is 80 and the unconstrained allocation fits inside both
capacities: Q takes 28 + 24 = 52 of its 55, R takes 16 + 12 = 28 of 50. Flow is
`28×5 + 24×3 + 16×4 + 12×3 = 312`, and total is `1,410 + 312 = 1,722`. At 1.2×, demand is 120,
which exceeds {Q, R}'s 105 capacity, so the design becomes infeasible rather than expensive. The
best feasible design is {P, R}: P carries 78 of 80, R carries 42 of 50, flow is 420, total 2,020.

**Labour rate.** Labour is 45% of fixed cost and 25% of flow cost in this instance, so a 20%
increase raises fixed costs by 9% and flow costs by 5%. {Q, R} goes to
`1,410 × 1.09 + 430 × 1.05 = 1,988.4`; the alternative goes to `1,600 × 1.09 + 350 × 1.05 =
2,111.5`. No flip: the gap widens, because labour is a larger share of P's fixed cost than of Q's.

**Transport rate.** Scaling all unit costs by 1.2 scales both flow terms and leaves the
allocation unchanged, because relative costs are unchanged: {Q, R} becomes 1,926, {P, R} 2,020.
Scaling by 0.8 gives 1,754 and 1,880. No flip at any plausible rate, as section 6.4 shows.

**Q capacity.** At 66 thousand units, M1 moves entirely to Q and flow falls to
`35×5 + 30×3 + 20×4 + 15×3 = 390`, so total is 1,800, a 40 saving. At 44 thousand units, {Q, R}
capacity falls to 94, below demand, and the design is infeasible; {P, R} at 1,950 is best.

### 6.4 The thresholds, which are not ranked like the bars

The tornado ranks fixed costs first, at a swing of 564. That ranking is misleading, and the
threshold table shows why.

| Parameter | Threshold value | Move from base | What happens at the threshold |
|---|---:|---:|---|
| Demand volume | 105 k units | +5.0% | {Q, R} becomes infeasible; {P, R} takes over |
| Q capacity | 50 k units | −9.1% | {Q, R} becomes infeasible; {P, R} takes over |
| Q fixed cost alone | 1,020 | +12.1% | {P, R} becomes cheaper; a cost flip, not a feasibility flip |
| Transport rate | 2.375× | +137.5% | {P, R} becomes cheaper; a cost flip |
| Labour rate | no positive threshold | — | {Q, R}'s advantage widens as labour rises |

The fixed-cost parameter flips the answer at 12.1% when Q's fixed cost moves alone, even though
the all-sites bar is the longest in the tornado and does not flip anything at ±20%. This is the
central lesson of the worked example, and it generalises: **a tornado bar that scales every site
together is nearly a uniform change to the objective, and uniform changes cancel out of a
ranking.** What moves a ranking is a change that is *differential* between the designs. Q's fixed
cost is differential; all fixed costs scaled together are not.

The 12 per cent figure is the one to carry into the report, because it is the number a lease
negotiation can move. A landlord asking 12% more than the current rent turns {Q, R} from the
answer into the runner-up, at a cost of 110 thousand per year.

### 6.5 The scenario matrix and the regret table

Five scenarios, each a complete parameter set. The correlations are named in the definition
column because they are the part that a percentage band cannot represent.

| ID | Scenario | Definition and named correlations | Demand | Transport | Fixed | Q capacity |
|---|---|---|---:|---:|---:|---:|
| A | Base | The 2026 plan; no correlation asserted | 100 | 1.00 | 1.00 | 55 |
| B | Growth | Volume +20%; geography unchanged; freight rises with volume | 120 | 1.05 | 1.00 | 55 |
| C | Cost shock | Freight and energy +20%; wages +20% (fixed +9%, flow +5%) | 100 | 1.20 | 1.09 | 55 |
| D | Capacity shock | Q's demonstrated throughput 12% below the planning basis | 100 | 1.00 | 1.00 | 48 |
| E | Contraction | Volume −20%; wages soften 20%; no site closures | 80 | 1.00 | 0.91 | 55 |

Costs by design and scenario, with each design re-optimised within its open set:

| Design | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| D1 = {Q, R} | 1,840 | infeasible | 2,053 | infeasible | 1,722 |
| D2 = {P, R} | 1,950 | 2,020 | 2,164 | 1,950 | 1,880 |
| D3 = {P, Q} | 2,440 | 2,537 | 2,719 | 2,447 | 2,346 |
| D4 = all three | 2,830 | 2,894 | 3,120 | 2,830 | 2,766 |

Regret, in thousand currency units per year, measured against the best feasible design in each
scenario:

| Design | A | B | C | D | E | Max regret | Feasible in |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 = {Q, R} | 0 | n/a | 0 | n/a | 0 | n/a | 3 of 5 |
| D2 = {P, R} | 110 | 0 | 111 | 0 | 158 | **158** | 5 of 5 |
| D3 = {P, Q} | 600 | 517 | 666 | 497 | 624 | 666 | 5 of 5 |
| D4 = all three | 990 | 874 | 1,067 | 880 | 1,044 | 1,067 | 5 of 5 |

Three findings, in order of importance.

**The base optimum is infeasible in two of five scenarios.** D1 wins everywhere it is feasible
— A, C, and E — and cannot serve demand in B or D, because it is loaded to 105 of 105 capacity.
A design that wins three scenarios and cannot operate in two is not a robust design. It is the
best design for the world the model was told about.

**Minimax regret selects D2.** D1 has no finite maximum regret because it is infeasible in B and
D. Among designs that are feasible everywhere, D2's worst regret is 158, against 666 and 1,067.
D2 is the design you can defend in every future in the set, at a base-case premium of 110 per
year, or 6.0%.

**Expected cost selects D1, if you let it.** With equal weights and a penalty of 4,000 per
infeasible scenario, expected costs are D1 1,600, D2 1,993, D3 2,509, D4 2,891. D1 wins. The
result depends entirely on the penalty: D2 only beats D1 on expected cost if the infeasibility
penalty exceeds roughly 4,980 per year. Choosing that penalty is choosing a probability
judgement — "infeasibility is worth about five thousand a year" — and it is the point at which
an expected-cost analysis stops being neutral. This is the honest reason minimax regret is the
better criterion here: it does not require the analyst to price an event that has no price.

### 6.6 What the analysis says, and what it does not

The recommendation, written in the form the reporting standard requires:

> Open Q and R, at a total annual cost of 1,840 thousand per year at 2026 prices. The design is
> optimal in the base case and in the cost-shock and contraction scenarios, with a margin of 6.0%
> over the runner-up. It is infeasible if total demand exceeds 105 thousand units (+5.0%) or if
> Q's demonstrated capacity falls 9.1% below the planning basis (48 thousand units). Q's fixed
> cost rising 12.1%, to 1,020 thousand per year, makes P and R cheaper by 110 thousand. If the
> capacity or demand thresholds are judged credible, open P and R instead, at a base-case
> premium of 110 thousand per year, and accept that the premium is the price of remaining
> feasible in every scenario tested.

What the analysis does not say: whether a 5% demand increase is credible (that requires the
forecast error history from Module 07), whether Q's demonstrated throughput in a bad month is 48
or 55 (that requires the site's operating data), and whether the organisation can execute a
two-site transition (that is topic 01's executability question, using topic 06's machinery). The
analysis has converted those questions into thresholds that someone else can answer. That is its job.

## 7. Code, query, or calculation

The calculation below reproduces every number in section 6 with a pencil. It is the preferred
form for a small instance because it shows the two things a solver call hides: which parameter
flips the open set, and how differently the parameters rank when sorted by threshold rather
than by swing.

```text
Meridian base:  f = [1100, 910, 500]   cap = [80, 55, 50]   d = 100
best open set {Q, R}: fixed 1,410 + flow 430 = 1,840

parameter             -20%    +20%   swing   open set at -20%   at +20%
all fixed costs       1,558   2,122     564   {Q,R}              {Q,R}
demand volume         1,722   2,020     298   {Q,R}              {P,R}
labour rate           1,692   1,988     297   {Q,R}              {Q,R}
transport rate        1,754   1,926     172   {Q,R}              {Q,R}
Q capacity            1,950   1,800     150   {P,R} (cap 94)     {Q,R}

thresholds, and what happens there
  demand      105 / 100 - 1              = +5.0%    {Q,R} infeasible
  Q capacity   50 / 55 - 1               = -9.1%    {Q,R} infeasible
  f_Q         1,950 - 500 - 430          = 1,020    +12.1%  -> {P,R}
  transport   (1,600 - 1,410)/(430 - 350) = 2.375   +137.5% -> {P,R}
  labour      diff = -110 - 65.5 x (m-1)             no positive root

rank by swing : fixed 564 > demand 298 > labour 297 > transport 172 > capacity 150
rank by flip  : demand 5.0% > capacity 9.1% > f_Q 12.1% > transport 137.5%
```

Two readings. **The rankings disagree, and the second one is the decision-relevant ranking.**
A parameter with a large cost swing that never changes the design is a parameter that changes
the size of the prize, not the choice. A parameter with a small swing that flips the design at
5% is the one that should be in the first paragraph of the report. **The all-fixed-cost row is
not a single parameter.** Scaling every fixed cost together is a near-uniform change to the
objective, and near-uniform changes cancel in a ranking; Q's fixed cost alone is differential
and flips at 12.1%. If the report had carried only the five-row tornado, the 12% threshold would
never have been found.

> **Note:** the labour row has no positive threshold in this instance, and that is a finding
> rather than a gap in the analysis. Not every parameter has a switch point within a plausible
> range. Reporting "no threshold in range" is more informative than reporting an elasticity that
> implies one exists somewhere.

## 8. Data, standards, and regulatory context

Scenario and sensitivity work is governed less by regulation than by documentation standards.
What is governed, and what is not, is worth stating because committees sometimes ask for a
standard that does not exist.

| Source | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 31000:2018 | Risk management principles and process | The requirement to identify, analyse, and evaluate risk, not the method | B |
| ISO/IEC 31010:2019 | Risk assessment techniques | Lists sensitivity analysis among recognised techniques; does not prescribe ranges | B |
| World Customs Organization HS nomenclature | Product classification | Tariff scenarios must key on HS codes or the duty change cannot be modelled | B |
| Regulation (EU) 2023/956 (CBAM) | Carbon border adjustment | A dated, non-probabilistic cost change that belongs in the policy scenario | B |
| ISO 14083:2023 | Transport chain emissions quantification | The emission factors used if carbon is a scenario dimension | B |
| IFRS 16 / ASC 842 | Lease accounting | The annual cash cost that belongs in `f_j`; the P&L charge is a different number | B |
| National statistical indices (Eurostat, BLS, ILO) | Labour and price deflation | Converting parameters to a common price year; the basis for cost ranges | B |
| Supervisory guidance on model risk management (US Federal Reserve SR 11-7) | Model documentation and validation | A documented parameter set, out-of-sample testing, and named limitations, by analogy | B |

**What is not governed.** No standard prescribes how many scenarios a network study should use,
how wide a range should be, or which decision criterion applies. ISO/IEC 31010 lists techniques
without ranking them. The scenario-count question is answered by communication and coverage
arguments, not by compliance. An analyst who claims a regulatory basis for a four-scenario set
is overstating the guidance.

**The price year is the one hard discipline.** Mixing a 2024 freight index with a 2026 wage index
produces a ranking that neither year supports, and the error is invisible in the output. Deflate
every cost to one stated year, record the deflator and its source in the ledger, and state the
year on every table.

> **Note:** tariff and carbon scenarios are step changes, not percentage ranges. A duty that
> moves from zero to a positive rate cannot be represented by a ±20% band, and a study built
> only from percentage bands will systematically miss the policy scenario that removes a
> cross-border option entirely.

## 9. Industry practice and case evidence

**What the review literature reports.** Klibi, Martel, and Guitouni's critical review of robust
supply chain network design (2010) observes that the dominant practice has been a deterministic
model followed by "appropriate sensitivity analysis and scenario analysis", and argues that this
is insufficient for long-horizon decisions under deep uncertainty. The observation is about
method prevalence, not outcomes, and it is the most useful framing for this topic. **A**

**What the factor-screening literature reports.** Lalwani, Disney, and Naim (2006) applied
Taguchi orthogonal arrays and analysis of variance to a distribution network simulation and
report that inventory holding cost and delivery frequency influenced the design more than demand
shifts or transport tariffs. If that ordering transfers, it inverts the intuition that demand is
always the first thing to vary; the finding is instance-specific and should prompt screening
rather than become a rule. **A**

**What the sensitivity-method literature reports.** Saltelli and colleagues (2019) argue that
one-at-a-time designs are invalid for non-linear models and leave most of a multi-dimensional
input space unexplored; in ten dimensions they put the unexplored fraction above 99%. Taylor and
Davis (2024) review one-way sensitivity reporting in health technology assessment and find that
tornado diagrams typically show only the extremes of the range — with a 95% interval, outcomes
at each end have about a 2.5% likelihood — and recommend showing intermediate values and
thresholds. Vreman and colleagues (2021) report that arbitrary rather than evidence-based ranges
were common in the appraisals they sampled. The domain is health economics, not logistics; the
methodological point transfers directly. **A**

| Practice | What is observed | Grade | How to read it |
|---|---|---|---|
| Deterministic model plus a sensitivity table | The norm in published and practitioner network studies | C | Description of prevalence, not of quality |
| Scenario software features | Vendors present scenario management as a product capability | D | A feature is not a method |
| Four-scenario strategy sets | Consultancy guidance recommends at least four | C | A communication heuristic, not an analytical result |
| Regret-bounded design | Formalised as p-robustness in the location literature | A | The rigorous version of "do not be too wrong anywhere" |
| Range choice by judgement | Common, and rarely documented | C | The main source of unverifiable conclusions |

**The gap between the literature and practice.** The formal machinery — p-robustness, minimax
regret, sample average approximation — is well developed and thinly used. Practitioner studies
overwhelmingly report a base case, a tornado chart, and three scenarios, with ranges chosen by
judgement and not documented. That is not necessarily wrong; it is cheap, understandable, and
adequate for many decisions. It becomes wrong when the ranges are chosen after the answer is
known, the failure mode section 10 opens with. **C**

## 10. Failure modes and red flags

| Failure | Red flag you can see | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Scenarios built to support the preferred answer | Every scenario's preferred site is the sponsor's site | Ranges set after the answer is known | Study ratifies a decision already made | Include one scenario that hurts the preferred design; state it in advance |
| One-at-a-time only | The deck has a tornado and no scenario matrix | OAT is cheaper and feels rigorous | Joint effects and correlations never tested | Add a designed set of three to five coherent scenarios |
| Range with no basis | The bar chart says "plausible range" and nothing else | Ranges chosen for visual balance | Bar lengths are arbitrary; ranking is meaningless | Source every bound in the ledger, or label it a judgement and name the judge |
| Tornado read as a risk ranking | "Transport is our biggest risk" from the longest bar | Effect size confused with probability | Effort and hedging aimed at the wrong parameter | Add plausibility and threshold columns; rank by flip, not by bar |
| RHS range reported as robustness | "The answer holds across a wide range" | Range is about the LP basis, not the design | False confidence in an unstable open set | Report open-set stability separately |
| Dual value quoted from a degenerate LP | A capacity shadow price with no basis stated | Duals are not unique under degeneracy | Capital request justified by a number the solver could have reported differently | State the basis; check allowable increase and decrease for zero |
| Infeasibility hidden as high cost | A scenario row with a very large cost and no note | Solver returned infeasible and the analyst substituted a penalty | The most important finding is buried | Record feasibility as a separate field; never impute silently |
| Expected cost with invented weights | Equal weights with no justification | Weights assert a probability judgement | Recommendation inherits false authority | Use regret or robustness when probabilities are not defensible |
| Uniform parameter scaling | Every cost moves by the same percentage in every scenario | Convenience | Uniform changes cancel; no scenario differentiates | Make at least one parameter move differentially per scenario |
| Price years mixed | Freight at one year, wages at another | Inputs pulled from systems with different vintages | Ranking reversed by a deflation artefact | Deflate to one stated year; record the deflator |
| Scenario count inflated | Twenty scenarios, none named, no narrative | Software makes runs cheap | Nobody can argue with any of them | Three to six named scenarios; a sample belongs in topic 08 |
| No stated criterion | The matrix has no decision rule | Criterion left implicit | The reader supplies their own and reaches a different design | State expected cost, minimax regret, or robustness, and why |

One of these deserves expansion.

**Infeasibility hidden as high cost.** A solver returns "infeasible", and the analyst needs a
number for the table, so a large penalty is substituted, averaged with the other scenarios, and
a design is selected. This is the most consequential silent error in the genre: it converts a
structural finding — the design cannot operate in this future — into a cost that can be traded
off. Section 6.5 showed the mechanism, and the penalty has no evidential basis.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails |
|---|---|---|
| The scenario set spans the decision-relevant futures | Regret and stability are computed over the set, not the world | A design that is worst in an unlisted future looks robust |
| Each scenario is internally coherent | Otherwise it is a sensitivity, and its probability is undefined | Regret comparisons across incoherent scenarios are meaningless |
| Parameter ranges are stated and sourced | Bar lengths and thresholds depend entirely on them | The analysis measures the analyst's range choice, not the business |
| The model structure is right in every scenario | One model is re-solved, not rebuilt | A structural break — a new channel, a new technology — is invisible |
| Costs are comparable across scenarios at one price year | Regret is a difference of costs | Mixed vintages produce spurious rankings |
| Feasibility is recorded separately from cost | An infeasible design has no finite cost | A structural failure is traded off as if it were expensive |
| The decision criterion is stated | Different criteria select different designs | The reader substitutes their own rule and disagrees with the conclusion |
| Scenario probabilities, if used, are defensible | Expected cost is a weighted average | A precise-looking number carries no information |
| Interactions are not dominant | The scenario matrix is a sample, not an exhaustive design | A joint effect outside the set is missed |
| The open set is the decision unit | Stability is counted over sets | Partial openings and phasing are invisible |

Two limits matter more than the rest. **The analysis is only as good as the scenario set**, and
the set is a judgement; no arithmetic repairs a set that omits the future in which the
recommendation fails. **The analysis says nothing about transition**: it compares steady-state
designs, not the cost and risk of getting from here to there, which is topic 06's subject.

## 12. Exercises

### Exercise 1 — Classify the runs (recall and application, 30 minutes)

Five runs: (i) demand +15% alone; (ii) demand +15% with freight +18% because both follow the
same cycle; (iii) all fixed costs +10%; (iv) a tariff removing one cross-border lane; (v) the
current network held fixed.

**Deliverable:** a table classifying each as sensitivity, scenario, or reference case, with one
sentence of justification.
**Acceptance criteria:** classification uses the joint-movement test, not the size of the
change; (ii) is a scenario with its correlation named; (iii) is noted as a near-uniform change
that may not move the ranking.

### Exercise 2 — Rank the inputs (analysis, 40 minutes)

Take a network study you know, or the section 6 instance, and produce the top six rows of the
ranked input list from section 5, Step 3.

**Deliverable:** six rows with parameter, plausible range, basis for the range, expected effect
on the open set.
**Acceptance criteria:** every range is sourced or labelled a judgement with a named owner; one
parameter is ranked low with a written reason; the ranking is justified by cost structure.

### Exercise 3 — Read the duals (calculation, 45 minutes)

A three-constraint LP has an optimal basis in which one basic variable is at zero, and the
solver reports a dual of 0 on a capacity constraint whose slack is zero.

**Deliverable:** why the reported dual may be wrong, both one-sided shadow prices, and a
one-paragraph instruction to a colleague on what to quote.
**Acceptance criteria:** degeneracy identified from the zero basic variable and zero slack; both
prices given with units; the instruction states the basis and the caveat.

### Exercise 4 — The useless range (critique, 30 minutes)

Using the section 6 numbers, compute the transport multiplier at which the ranking changes and
write the paragraph that would accompany it.

**Deliverable:** the algebra and a paragraph of no more than 120 words.
**Acceptance criteria:** the multiplier is 2.375; the paragraph attributes the width to the
fixed-to-flow ratio, not to an absence of transport risk; no robustness claim is made.

### Exercise 5 — Design a scenario set (design, 60 minutes)

For a consumer goods network with a 2030 horizon, design four scenarios.

**Deliverable:** four named scenarios, each a complete parameter set with correlations, plus the
coverage test from section 5.3.
**Acceptance criteria:** at least one scenario hurts the preferred design; no two lie on the same
side of every switch point; each correlation has a causal story; the count is justified.

### Exercise 6 — Convert the tornado into thresholds (calculation, 45 minutes)

Take the five rows of section 6.3 and compute, for each, the parameter value at which the open
set changes, or state that none exists in a plausible range.

**Deliverable:** a threshold table with the algebra for each row.
**Acceptance criteria:** demand +5.0%, Q capacity −9.1%, Q fixed cost +12.1%, transport +137.5%,
labour with no positive root and an explanation; the two rankings shown side by side.

### Exercise 7 — Regret against expectation (calculation, 45 minutes)

Using the cost matrix in section 6.5, compute the regret table and the expected cost of each
design under equal weights and under weights of 0.5, 0.2, 0.1, 0.1, 0.1 for A to E.

**Deliverable:** both tables and a statement of which design each criterion selects.
**Acceptance criteria:** infeasibility handled explicitly with the penalty stated; the
sensitivity of the answer to the penalty shown; the recommendation names the criterion.

### Exercise 8 — Write the stability memo (design, 60 minutes)

Write a one-page recommendation for Meridian to the standard in section 5.8.

**Deliverable:** the memo with the eight required items as headings.
**Acceptance criteria:** all eight items present; the flip named in one sentence; the criterion
stated with a reason; the recommendation conditional rather than absolute; every number with its
unit.

## 13. Self-check answers

**Exercise 1.** (i) Sensitivity: one parameter, everything else fixed. (ii) Scenario: two
parameters move together with a causal link, so the joint state is coherent. (iii) Sensitivity,
and a weak one: scaling all fixed costs is near-uniform and may not alter the ranking. (iv)
Scenario, and a structural one: the lane is removed, so the parameter space changes shape. (v)
Reference case: the do-nothing option, which belongs in every matrix as a design.

**Exercise 2.** A defensible six: demand volume (−20% to +30%), demand geography (5% to 15% of
volume relocated), fixed cost estimates (−25% to +25% after stripping allocation), capacity
ceilings (−10% to −25% against nameplate), transport rates (−15% to +40%), duty and tariff (step
changes). A low-ranked parameter with a reason might be the inventory carrying rate: it scales a
minority term and rarely changes the open set.

**Exercise 3.** The zero basic variable means the optimal dual set is not a singleton, and the
reported dual of 0 is one extreme point. The true one-sided shadow prices are the minimum and
maximum of the dual over that set: the value of an extra unit is the minimum, and the value of
removing a unit is the maximum, up to sign convention. Quote both, with the constraint, the
units, and the sentence "the LP is degenerate, so this is a range rather than a single price".

**Exercise 4.** `1,410 + 430k = 1,600 + 350k` gives `80k = 190`, so `k = 2.375`. The paragraph
should say transport would have to rise 137.5% before the ranking changes, that this reflects a
cost structure in which fixed cost is roughly three times flow cost, and that the correct
conclusion is that transport is not the switching parameter — not that transport is risk-free.

**Exercise 5.** A strong set: base; growth with freight rising alongside volume; a cost shock in
which energy, freight, and wages rise together; a policy shock in which a carbon border cost and
a tariff arrive together; a capacity shock at the most loaded site. Each correlation has a
causal story, and at least one scenario makes the preferred design infeasible or third. A weak
set varies demand only, in three sizes, and calls it three scenarios.

**Exercise 6.** Demand: `105/100 − 1 = +5.0%`, infeasibility. Q capacity: `50/55 − 1 = −9.1%`,
infeasibility. Q fixed cost: `1,950 − 500 − 430 = 1,020`, or `+12.1%`, a cost flip. Transport:
`(1,600 − 1,410)/(430 − 350) = 2.375`, or `+137.5%`, a cost flip. Labour: the difference is
`−110 − 65.5(m − 1)`, which never reaches zero for positive `m`, so there is no threshold.

**Exercise 7.** With equal weights and a penalty of 4,000, expected costs are D1 1,600, D2 1,993,
D3 2,509, D4 2,891, and D1 wins. With weights 0.5, 0.2, 0.1, 0.1, 0.1, D1 is
`920 + 800 + 205 + 400 + 172 = 2,497` and D2 is `975 + 404 + 216 + 195 + 188 = 1,978`, so D2
wins. The ranking changes with the weights, which is the point: expected cost is only as
defensible as the weights, and minimax regret selects D2 without needing them.

**Exercise 8.** A strong memo states the five parameters and their sourced ranges; five named
scenarios with correlations; why five rather than three (both feasibility shocks covered); the
stability finding that D1 wins three of five and is infeasible in two; the regret table with
D2's maximum of 158; the flip as "a 5.0% demand increase or a 9.1% capacity derating makes the
base optimum infeasible"; and minimax regret as the criterion because no defensible
probabilities exist. A weak memo gives the base optimum and a tornado chart.

## 14. Interview questions

**1. "What is the difference between sensitivity analysis and scenario analysis?"**
Strong: one parameter against a coherent joint future; the first is a derivative, the second a
displacement, and the second is where correlations live. Weak: "scenarios are more detailed
sensitivities", which misses the joint-movement point.

**2. "Why does one-at-a-time analysis overstate confidence?"**
Strong: additivity fails near thresholds, the corners of the parameter space are never sampled,
and percentage bands cannot represent regime changes. Weak: "because it does not test
everything", with no mechanism.

**3. "You are given a capacity shadow price. When should you not believe it?"**
Strong: when the LP is degenerate, when the value comes from a MIP, or when the constraint is
structural rather than marginal; quote the basis and the one-sided prices. Weak: quotes the
number and adds it to the capital request.

**4. "The ranging report says the transport rate can rise 140% before the answer changes. Good news?"**
Strong: it says transport is not the switching parameter in this cost structure; the width
reflects the fixed-to-flow ratio, not the absence of risk. Weak: "yes, the design is robust",
which is the sentence that gets repeated in the steering committee.

**5. "How many scenarios should a network study use?"**
Strong: three to six named, coherent, designed for coverage; a larger sample belongs to a
stochastic formulation, and the test is whether the set contains the future in which the answer
fails. Weak: a number with no rationale, or "as many as the software allows".

**6. "How do you stop a scenario set from justifying the answer someone already wanted?"**
Strong: fix the ranges and the set before the base solve; include a scenario that hurts the
preferred design; have the set reviewed by someone who disagrees. Weak: "we were objective",
which is not a control.

**7. "What is minimax regret, and when would you use it instead of expected cost?"**
Strong: choose the design whose worst shortfall against the scenario optimum is smallest; use it
when probabilities are not defensible, and note it is not minimising worst-case cost. Weak:
confuses it with maximin or variance minimisation.

**8. "What is decision stability and why might you prefer it to optimality?"**
Strong: the same open set winning across most scenarios; a stable design avoids a reversal that
costs more than the optimality premium, counted over sets and feasibility, not just cost. Weak:
"the answer does not change much", with no definition of change.

**9. "Your tornado chart's longest bar is transport. What do you tell the room?"**
Strong: the bar shows the cost effect over the range chosen, not the risk; check the threshold
and the plausibility before calling it the biggest exposure. Weak: "transport is our biggest
risk", which converts an effect size into a probability claim.

**10. "A design is optimal in the base case and infeasible in one of five scenarios. What do you recommend?"**
Strong: do not average the infeasibility into a cost; report it, price the alternative design,
and let the business judge the scenario's credibility. Weak: assigns a large penalty and reports
an expected cost, hiding the structural finding.

## 15. Key terms

- **Sensitivity analysis** — output response to one input, all else fixed.
- **Scenario analysis** — output under a coherent, jointly specified future.
- **One-at-a-time (OAT)** — varying one parameter through a range, returning to base.
- **Tornado diagram** — sorted bar chart of output ranges by parameter; effect, not probability.
- **Scenario set** — the named futures evaluated together; a design choice, not a sample.
- **Coherent scenario** — parameter values that could plausibly coexist.
- **Correlated parameters** — inputs that move together, so joint variation is not additive.
- **Stress test** — evaluation under a named adverse future chosen to probe a vulnerability.
- **Switch point** — the parameter value at which two designs exchange rank.
- **Threshold** — the value beyond which the recommendation changes, including by infeasibility.
- **Decision stability** — the same open set winning across most scenarios.
- **Oscillation** — two or more designs alternating as optimal as parameters move.
- **Regret** — a design's cost in a scenario minus the best feasible cost in that scenario.
- **Minimax regret** — choose the design with the smallest worst-case regret.
- **Expected cost** — probability-weighted average cost; only as good as the weights.
- **Robustness** — acceptable performance across scenarios without a probability model.
- **Dual value** — local rate of change of the LP objective with respect to a right-hand side.
- **Shadow price** — the economic reading of a dual; valid under stated conditions.
- **Reduced cost** — the improvement needed before a variable enters the LP basis.
- **RHS ranging** — the interval over which one right-hand side may move without a basis change.
- **Degeneracy** — a zero basic variable, which makes duals non-unique.
- **Feasibility count** — the number of scenarios in which a design can operate at all.
- **Coverage test** — whether the scenario set contains the future in which the answer fails.

## 16. Sources

Grading: A peer-reviewed or audited; B regulatory, official, or standard-setting; C company case,
consultancy, or institutional analysis; D vendor; E press or anecdote.

**Peer-reviewed and methodological**

- Gál, T. (1986). "Shadow prices and sensitivity analysis in linear programming under degeneracy." *OR Spectrum*, 8, 59–71. DOI: 10.1007/BF01719736. **A**
- Ho (2000). "Computing true shadow prices in linear programming." *Informatica*. One-sided shadow prices under degeneracy. **A**
- Koltai, T., and Tatay, T. "A practical approach to sensitivity analysis in linear programming under degeneracy for management decision making." *International Journal of Production Economics*. DOI: 10.1016/j.ijpe.2010.04.037. **A**
- Saltelli, A., and colleagues (2019). "Why so many published sensitivity analyses are false." *Environmental Modelling & Software*. DOI: 10.1016/j.envsoft.2019.01.012. **A**
- Taylor, M., and Davis, S. (2024). "Tornado diagrams only tell us 5% of the story." *Value in Health*. DOI: 10.1016/j.jval.2024.10.741. Abstract as indexed; verify before quoting. **A**
- Vreman, R. A., and colleagues (2021). "The application and implications of novel deterministic sensitivity analysis methods." *PharmacoEconomics*. PMC7790801. **A**
- Gandjour, A. (2021). "Deterministic sensitivity analysis under ignorance." *PharmacoEconomics*. PMC8440146. **A**
- Savage, L. J. (1951). "The theory of statistical decision." *JASA*, 46, 55–67. **A**
- Stoye, J. "Axioms for minimax regret choice correspondences." Working paper, Cornell University. Menu dependence and ambiguity aversion. **C**
- Klibi, W., Martel, A., and Guitouni, A. (2010). "The design of robust value-creating supply chain networks: a critical review." *EJOR*, 203(2), 283–293. DOI: 10.1016/j.ejor.2009.06.011. **A**
- Govindan, K., Fattahi, M., and Keyvanshokooh, E. (2017). "Supply chain network design under uncertainty: a comprehensive review." *EJOR*. DOI: 10.1016/j.ejor.2017.04.009. **A**
- Snyder, L. V., and Daskin, M. S. (2006). "Stochastic p-robust location problems." *IIE Transactions*. **A**
- Daskin, M. S., Hesse, S. M., and ReVelle, C. S. (1997). "α-reliable p-minimax regret." *Location Science*. DOI: 10.1016/S0966-8349(98)00036-9. **A**
- Lalwani, C. S., Disney, S. M., and Naim, M. M. (2006). "On assessing the sensitivity to uncertainty in distribution network design." *IJPDLM*. Taguchi screening. **A**
- Birge, J. R., and Louveaux, F. (2011). *Introduction to Stochastic Programming*, 2nd ed. Springer. **A**

**Standards, official, and regulatory**

- ISO 31000:2018, risk management guidelines; ISO/IEC 31010:2019, risk assessment techniques. **B**
- ISO 14083:2023, transport chain greenhouse gas quantification. **B**
- Regulation (EU) 2023/956, carbon border adjustment mechanism. **B**
- IFRS 16 *Leases*; ASC 842. **B**
- World Customs Organization, Harmonized System nomenclature. **B**
- US Federal Reserve, SR 11-7, guidance on model risk management. **B**
- Eurostat, U.S. Bureau of Labor Statistics, and ILO labour cost and price indices. **B**

**Company, consultancy, and institutional**

- Roxburgh, C. (2009). "The use and abuse of scenarios." McKinsey Quarterly. At least four scenarios; dependence collapses combinations. **C**
- McKinsey (2021). "Scenario-based cash planning in a crisis." Three to four deterministic scenarios in CFO practice. **C**
- Amer, M., Daim, T. U., and Jetter, A. J. (2013). "A review of scenario planning." *Futures*. No precise optimum scenario count; commonly three to six. **A**

**Vendor**

- Coupa, "Sensitivity analysis in supply chain design and planning"; ICRON, "From static assumptions to scenario-based network design"; Lambda SCS, supply chain risk measurement. Scenario management presented as a software capability. **D**

> **Note on citation discipline.** The scenario-count guidance is a communication heuristic
> repeated across sources, not a finding, and the Taylor and Davis figure is a conference
> abstract quoted as indexed. No range or threshold here transfers to another network without
> rebuilding the instance.

## 17. Links

**This module**

- Module map: [`_index.md`](_index.md)
- The model being analysed: [`04_facility-location.md`](04_facility-location.md)
- Multi-commodity and multi-period extensions: [`05_multi-commodity-period.md`](05_multi-commodity-period.md)
- MILP anatomy and why MIPs have no duals: [`03_milp-anatomy.md`](03_milp-anatomy.md)
- Transition and brownfield analysis: [`06_greenfield-brownfield.md`](06_greenfield-brownfield.md)
- The graph and cost-to-serve basis: [`02_graph-cost-to-serve.md`](02_graph-cost-to-serve.md)
- The decision ladder this analysis serves: [`01_network-decision-ladder.md`](01_network-decision-ladder.md)
- Next, the formal treatment of uncertainty: `08_robust-stochastic.md` (planned)
- Later, carbon as a scenario dimension: `09_carbon-objective.md` (planned)
- Later, solver behaviour and gap reporting: `10_solvers-heuristics.md` (planned)

**Adjacent modules**

- Forecast error, which is the input to the demand range: [`../07_time_series_forecasting/_index.md`](../07_time_series_forecasting/_index.md)
- Intermittent and probabilistic demand: [`../08_hierarchical_intermittent_probabilistic_forecasting/_index.md`](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md)

**Programme navigation**

- Part II methods map: [`../_index.md`](../_index.md)
- Part I foundations: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)

> **Note:** topics 08 to 10 are listed in `_index.md` but do not exist on disk yet, so they are
> shown as plain file names marked planned. Every markdown link on this page resolves.
