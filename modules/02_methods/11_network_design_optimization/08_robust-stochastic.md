# Robust and Stochastic Optimization — hedging against demand, cost, and disruption uncertainty

> Module 11 — Network Design and Optimization
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 topics 04, 05, and 07
> Version 2026.5

---

## 1. Why this topic exists

Topic 07 ends with a matrix. Designs down the rows, futures across the columns, regret in the
cells, and a sentence naming the future in which the recommendation fails. That is a diagnosis.
It is not yet a decision rule, and a capital committee cannot sign a matrix.

The gap is not analytical laziness. It is genuine: the matrix says "this design wins in three of
five futures and cannot operate in two", and the committee asks a question the analyst has no
principled way to answer. How much is being feasible in the other two worth. What probability to
put on the capacity shock. Whether the worst case is a real risk or a thought experiment. Topic
07 shows how to expose those questions; this topic shows the three formal ways of answering
them, what each one assumes, and when none of them is worth the cost.

**The design that was right on average and wrong in the average year.** A distributor replaces a
deterministic study with a stochastic one. The deterministic model, run at mean demand, opens
four distribution centres (DCs) and reports 8.4 million per year. The stochastic model opens
three, accepts a contractual expedite arrangement for the fourth region, and reports 8.1 million
expected cost. The saving is real but modest. The surprise is elsewhere: the stochastic design's
worst-case cost across the same scenarios is 9.6 million, while the deterministic design's is
9.5. The stochastic answer is better in expectation and slightly worse in the tail. The
committee had assumed those two facts could not diverge. They diverge because the stochastic
model optimises a weighted average, and a weighted average is not a risk statement.

**The disruption that ranked fourteenth.** A risk register scores a port closure as
low-probability and medium-impact, which puts it fourteenth on the list. The facility that
depends on that port is the cheapest in the network and carries 40 per cent of volume. A
probability-times-impact ranking is an expected-cost calculation with the units hidden, and it
systematically buries events whose impact is structural. Framed as a robust question — what
design still functions if any one port is unavailable for eight weeks — the answer changes, and
the change costs 1.9 per cent of annual network cost. The register was not wrong. It was
answering a different question from the one the design decision required.

**The study that ran for six weeks and changed nothing.** A team builds a full multi-stage
stochastic programme with 500 scenarios and a nine-hour solve. The first-stage recommendation
matches the three-scenario analysis from the first week, and the value of the stochastic
solution (VSS) comes out at 0.4 per cent of cost. The conclusion was available in three days;
knowing when to stop is part of the method.

Three things follow, and they organise the topic.

First, **uncertainty enters a network model in three different ways that are not
interchangeable.** A weighted average of futures, a probability-weighted programme with recourse,
and a worst case over a set are three different claims about what is known; choosing among them
is a claim about the evidence, not a preference about technique.

Second, **each family has a diagnostic that tells you whether it was worth doing.** For the
stochastic family, the VSS and the expected value of perfect information (EVPI); for the robust
family, the price of robustness. A study that reports neither has produced an answer with no
measure of what the answer is worth.

Third, **the machinery is expensive in four currencies: model size, solve time, data, and
organisational agreement.** The fourth is the one that kills projects. A scenario set that the
commercial team will not sign is not a scenario set; it is a negotiating position.

> **Note:** this topic builds directly on topic 07 and does not repeat its diagnostics. The
> scenario design rules, the regret table, the stability test, and the reporting discipline all
> live there. What is new here is the formal optimisation layer that sits on top of them, and
> the judgement about when that layer earns its cost.

**Relationship to the rest of the programme.** Topic 04 owns the facility-location model whose
first-stage variables this topic reuses; topic 05 owns the multi-period structure that makes a
scenario tree deep rather than wide; topic 07 owns the scenario diagnostics; topic 10 owns solver
behaviour, since a stochastic mixed-integer linear programme (MILP) needs decomposition to be
tractable. Module 08 supplies the demand distributions, Module 20 the out-of-sample test, and
Module 14A the disruption evidence, whose modelling pattern this topic borrows without
reproducing.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** deterministic-with-scenarios, two-stage stochastic programming, and robust
   optimisation, and **state** the assumption each makes about what is known.
2. **Write** a two-stage recourse formulation, **separating** first-stage variables from
   second-stage variables, and **explain** why non-anticipativity is a modelling constraint.
3. **Size** a scenario tree, **compute** leaves and nodes for a given branching factor and depth,
   and **explain** why scenario reduction is a modelling choice.
4. **Compute** VSS and EVPI from a solved instance, **interpret** each in currency per period, and
   **explain** why a small EVPI does not make uncertainty unimportant.
5. **Define** box, budget, and ellipsoidal uncertainty sets, **compute** the price of robustness,
   and **choose** the budget parameter with a stated justification.
6. **Critique** a robust model for over-conservatism and a stochastic model for false precision,
   and **place** distributionally robust optimisation (DRO) between them.
7. **Decide** whether a decision warrants a full stochastic or robust programme, or whether a
   designed scenario set with regret and stability analysis is sufficient, and **defend** it.
8. **Write** the uncertainty section of a network study to the reporting standard in section 5.8,
   including representation, count, variable lists, diagnostics, and first-stage stability.

Each outcome maps to an exercise in section 12 and to a self-check answer in section 13.

## 3. The core mental model

Hold two pictures. The first is a timeline: a decision fixed on the left, an uncertainty fork in
the middle, and a set of responses on the right. The second is a pair of shapes: a box and a
diamond, representing two ways of saying "the world can be anywhere in here".

```text
 time --------------------------------------------------------------------->

  FIRST STAGE (here-and-now)          SECOND STAGE (wait-and-see recourse)
  fixed before uncertainty            chosen after the scenario is known

  +---------------------+
  | open facilities x   |            +----------------------------------+
  | capacity, contracts |----------> | scenario 1: flow, expedite, buy  |
  | one design for all  |     |      | cost = Q(x, omega_1)             |
  +---------------------+     |      +----------------------------------+
                              |
                              |      +----------------------------------+
                              +----> | scenario 2: flow, expedite, buy  |
                              |      | cost = Q(x, omega_2)             |
                              |      +----------------------------------+
                              |
                              |      +----------------------------------+
                              +----> | scenario S: flow, expedite, buy  |
                                     | cost = Q(x, omega_S)             |
                                     +----------------------------------+

  objective = c'x + sum over s of pi_s * Q(x, omega_s)
  no component of x may carry an s subscript: that is non-anticipativity
```

```text
   BOX SET  |u1| <= 1, |u2| <= 1        BUDGET SET  |u1| + |u2| <= Gamma
   worst case at a corner               worst case on the sloping face

        u2                                   u2
        ^   +-------+                      ^      /\
        |   |       |                      |     /  \
     +1 |---|-------|------>           +1 |---/----\------>  Gamma = 1
        |   |   *   |                      |  /  *   \         (diamond)
        |   |       |                      | /        \
     -1 +---+-------+------> u1         -1 +/----------\----> u1
       -1        +1                     -1             +1

   includes the corner (+1,+1):           excludes (+1,+1) when Gamma < 2:
   both deviations at maximum            at most Gamma deviations at maximum
   at once, often implausible            jointly; a tunable conservatism
```

**Property 1 — The first-stage decision is the only one the model is really making.** Everything
on the right of the fork is a conditional plan, and a conditional plan is cheap to write and
easy to change. The design is the expensive, slow, irreversible part. A stochastic programme
that returns a beautiful recourse policy and a first-stage decision indistinguishable from the
deterministic one has produced an expensive way of saying "no change".

**Property 2 — A weighted average of futures is not a scenario analysis.** Replace demand by its
mean and you have built the expected-value problem. It is a legitimate model, and it is the one
almost every deterministic network study actually solves. Its weakness is not that the mean is
wrong. It is that a discrete design cannot be half-built: if the mean falls just above a
capacity threshold, the mean model builds the large facility, and the average of the two worlds
would have preferred the small one plus a contractual fallback. The averaging has quietly
assumed that you can buy capacity at its average price and in fractional amounts.

**Property 3 — Recourse is the formal name for operational flexibility, and it must be real.**
The second-stage variables are only worth modelling if someone can actually exercise them after
the uncertainty resolves: expedite, reallocate, buy spot, substitute, defer. If the operation is
rigid, the recourse problem collapses to the deterministic one with extra notation, and the VSS
goes to zero.

**Property 4 — Robust optimisation replaces probability with geometry.** Instead of asking how
likely a future is, it asks which futures are possible and chooses the design that is best under
the worst of them. The shape of "possible" is the uncertainty set, and the shape is a modelling
choice with as much consequence as any probability. A box says every combination of extremes is
possible; a budget set says only a limited number of things go wrong at once. The second is
usually truer and always less pessimistic.

**Property 5 — The two critiques are both correct, and they point in opposite directions.**
Robust optimisation can be too conservative: it defends against a worst case that may be
implausible, and it pays for that defence in every scenario. Stochastic programming can be too
optimistic: it optimises against the futures you thought of, using probabilities you asserted,
and returns a precise number that hides both choices. Neither critique defeats the other. They
describe different failure modes, and the practical question is which one your decision is more
exposed to.

### 3.1 The three families as three answers to one question

All three families answer the same question: given a set of possible futures, which design do I
commit to now. They differ in what they do with the futures.

| Family | What it does with the futures | What it assumes | What it returns | Diagnostic |
|---|---|---|---|---|
| Deterministic with scenarios | Averages the parameters, or averages the scenario optima | A single representative future exists, or the average design is buildable | One design, one cost number per scenario | None; topic 07's stability and regret tables substitute |
| Two-stage stochastic | Weights them by probability and optimises the expected cost with recourse | Probabilities are defensible; recourse is available and modellable | One first-stage design plus a recourse policy per scenario | VSS and EVPI, in currency per period |
| Robust | Takes the worst case over a defined set | The set is right; the worst case is worth defending against | One design that is feasible and bounded over the set | Price of robustness, and the set definition |
| Distributionally robust | Takes the worst case over a set of distributions | The ambiguity set contains the truth; the worst distribution is credible | One design, bounded over the ambiguity set | Radius of the ambiguity set; price of ambiguity |

The table is the topic in miniature. Read down the "assumes" column: each family buys its answer
with a different claim, and the claim is the thing to interrogate in a review.

### 3.2 The worst case as a dial, not a doctrine

Treat robust optimisation as a dial rather than a philosophy. The budget parameter, Gamma,
controls how many uncertain parameters may deviate simultaneously: at zero the model is
deterministic, at the full count it is the conservative box, and between them the cost rises
monotonically. The slope is the price of robustness per unit of protection. You can therefore
ask what one more unit of protection costs and stop where the marginal cost exceeds what the
business will pay. That is a conversation a committee can have.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Uncertainty representation | The formal object standing in for what is not known: a scenario set, a probability distribution, or an uncertainty set | Used to mean "the list of risks" | Method |
| Deterministic model | An optimisation model in which every parameter takes one value | Described as a model "without uncertainty" rather than one with a point assumption | Model class |
| Expected-value problem (EV) | The deterministic model in which each random parameter is replaced by its expectation | Treated as the stochastic solution | Currency per period |
| Expected result of using the EV solution (EEV) | The expected cost of applying the EV-optimal first-stage decision across the scenarios | Confused with the EV objective | Currency per period |
| Recourse problem (RP) | The stochastic programme: minimise first-stage cost plus expected optimal recourse | Called "the stochastic model" without saying which stage is random | Currency per period |
| Wait-and-see solution (WS) | The expected value of the per-scenario optima, as if the decision-maker could wait for each outcome | Treated as achievable | Currency per period |
| Here-and-now decision | A decision that must be fixed before the uncertainty resolves | Used for any early decision, including reversible ones | Decision timing |
| First-stage variable | A variable whose value cannot depend on the scenario | Includes variables that could be deferred | Variable class |
| Second-stage variable | A variable indexed by scenario, chosen after the uncertainty resolves | Used for variables that are physically fixed early | Variable class |
| Fixed recourse | A recourse matrix that does not depend on the scenario | Assumed without checking; it fails when the response itself is disrupted | Model property |
| Relatively complete recourse | The property that a feasible recourse exists for every feasible first-stage decision | Assumed when the model can in fact become infeasible | Model property |
| Non-anticipativity | The constraint that first-stage decisions cannot anticipate which scenario occurs | Treated as a solver detail | Constraint class |
| Scenario | One complete, internally consistent realisation of all uncertain parameters | Used for a single parameter value | Parameter set |
| Scenario tree | A rooted tree whose nodes are decision points and whose branches are realisations | Drawn as a fan of independent paths | Graph object |
| Scenario reduction | Replacing a large scenario set with a smaller one that is close in a probability metric | Treated as lossless compression | Method |
| Deterministic equivalent | The single large optimisation problem obtained by writing one copy of the second stage per scenario | Assumed to be small | Model |
| Sample average approximation (SAA) | Estimating the expected-cost objective by averaging over a Monte Carlo sample of scenarios | Reported as the true optimum | Method |
| Out-of-sample validation | Evaluating a computed first-stage decision on scenarios not used to compute it | Omitted when the sample is large | Test |
| Value of the stochastic solution (VSS) | EEV minus RP: the expected gain from using the stochastic model instead of the deterministic mean model | Reported without stating the deterministic comparator | Currency per period |
| Expected value of perfect information (EVPI) | RP minus WS: the most a decision-maker should pay to learn the future before committing | Confused with the value of forecasting | Currency per period |
| Uncertainty set | The set of parameter realisations against which a robust model must remain feasible and bounded | Chosen for mathematical convenience | Set |
| Box uncertainty set | An independent interval for each uncertain parameter, with the worst case at a corner | Used when extremes cannot co-occur | Set |
| Budget uncertainty set | A set in which at most a stated amount of total deviation is allowed across parameters | Confused with a probability statement about how many failures occur | Set, with parameter Gamma |
| Ellipsoidal uncertainty set | A set defined by a quadratic norm bound on the deviation vector | Assumed to keep the problem linear | Set |
| Robust counterpart | The reformulated optimisation problem that enforces feasibility for every element of the uncertainty set | Treated as an approximation | Model |
| Price of robustness | The increase in objective value caused by protecting against uncertainty, usually relative to the nominal optimum | Quoted without stating the protection level | Currency per period, or proportion |
| Minimax cost | The criterion that minimises the worst-case cost over the uncertainty set | Confused with minimax regret | Decision rule |
| Minimax regret | The criterion that minimises the worst-case shortfall against the best design in each scenario | Confused with minimax cost | Decision rule |
| Distributionally robust optimisation (DRO) | Optimisation against the worst probability distribution within an ambiguity set | Described as "robust optimisation with probabilities" without the set | Method |
| Ambiguity set | The set of probability distributions a DRO model treats as possible | Left unspecified in reported results | Set of distributions |
| Conditional value at risk (CVaR) | The expected cost in the worst tail of the distribution, at a stated confidence level | Used interchangeably with worst case | Currency per period |
| Disruption scenario | A scenario in which a facility, lane, or supplier is unavailable for a stated period | Modelled as a cost increase rather than a capacity removal | Scenario definition |
| First-stage stability | The degree to which the recommended design is unchanged across samples, scenario reductions, or ambiguity radii | Reported as a single number without a definition | Proportion, or verdict |

Four of these deserve a caution. **VSS versus EVPI:** they answer different questions, and a
study can have a large VSS and a small EVPI; section 6 is exactly that case, so never report one
as "the value of uncertainty". **Non-anticipativity:** trivially satisfied if no first-stage
variable is indexed by scenario, but it must be imposed explicitly in multi-stage models and in
decomposition, where subproblems are solved separately. **Budget versus probability:** Gamma
equal to two means at most two deviations, not that two are likely; the Bertsimas–Sim bound is a
bound, not a confidence level. **Price of robustness:** meaningful only with the protection
level, the baseline, and the units attached.

## 5. The mechanics, step by step

The procedure below takes a deterministic network model and a set of futures and produces
either a stochastic or a robust recommendation with its diagnostics. It has eleven steps and a
branch that sends the work back. Steps 1 to 3 are common to both families; the road forks at
step 4.

### Step 1 — Name the uncertainty and the moment it resolves

List every uncertain quantity with its unit, horizon, and the date it becomes known. The date
determines the model structure. Demand resolves continuously, which makes it a scenario
dimension or a multi-stage object, never a single second-stage parameter. A port closure
resolves when it happens; a freight rate resolves at contract signature. If every quantity
resolves after the same commitment, the problem is two-stage. If some resolve earlier, it is
multi-stage, and the tree in step 5 becomes the dominant cost.

### Step 2 — Separate first-stage from second-stage variables by asking who can act

Assign every variable to a stage using the physical or contractual commitment date. A lease is
first-stage: signed, non-refundable, unchanged by any scenario. A customer assignment is
second-stage: a planner can reallocate next month. A take-or-pay minimum is first-stage even
though the flow is second-stage, because the obligation is fixed. Ask **who signs, and when**;
if the same person signs before the uncertainty resolves, the variable is first-stage. Errors in
either direction destroy the model's value: too much in the first stage produces a rigid design,
too much in the second assumes flexibility nobody has.

### Step 3 — Choose the uncertainty representation, and write down what it claims

Choose one representation and write the sentence it asserts about the world.

| Representation | What you must be able to defend | What you must not claim |
|---|---|---|
| Scenario set with probabilities | That the scenarios are coherent and the weights are better than equal | That the weights are exact, or that the set is exhaustive |
| Scenario set without probabilities | That the set spans the decision-relevant futures | That the design is optimal for anything |
| Uncertainty set | That every point in the set is possible, and the worst point is worth defending | That points outside the set are impossible, or that the worst point is likely |
| Ambiguity set of distributions | That the true distribution is inside, at the stated radius | That the worst distribution is plausible |

The representation is the assumption a reviewer will attack, and writing it in one sentence
makes the attack possible, which is the point. "Demand in 2030 is uniform over 90 to 150 thousand
units, independently across regions" is attackable: the independence is probably false and the
shape almost certainly is. If you cannot defend a probability, use an uncertainty set at step
10; if you can defend only a mean and a covariance, the ambiguity set is the honest object.

### Step 4 — Write the recourse problem

Write three objects and the link between them. **First stage:** minimise `c'x` plus expected
recourse, subject to `A x = b` and integrality, where `x` is facility openings and capacity
commitments, `c` is their cost in currency per period, and `A x = b` holds budget and policy
constraints. **Second stage, conditional on scenario `omega`:** minimise `q(omega)' y` subject
to `T(omega) x + W(omega) y = h(omega)` and `y >= 0`, where `y` is flows, allocation, and
expediting, `h(omega)` is realised demand, and `W(omega)` maps a response into feasibility.
**Objective:** `min c'x + E[ Q(x, omega) ]`, where `Q` is the optimal second-stage value for a
fixed first-stage decision.

Writing the stages separately prevents the genre's most common error: indexing a first-stage
variable by scenario, which gives the model foresight it does not have. The matrix form makes
the separation visible, because `x` carries no `omega` subscript. Two assumptions travel with
the formulation. **Fixed recourse** means `W` does not depend on the scenario, so the standard
algorithms and convexity results apply; if the response itself can be disrupted, the problem is
harder and the VSS value is overstated. **Relatively complete recourse** means a feasible `y`
exists for every feasible `x`; if not, add an artificial unmet-demand variable with a penalty
chosen only when nothing else works, and report that penalty because it is a price on failure.

### Step 5 — Build the deterministic equivalent, and count it before you solve it

The deterministic equivalent has `|x| + S |y|` variables and roughly `|A| + S |W|` constraints,
where `S` is the number of scenarios and `|y|` is the size of one second-stage problem. For a
network with 20 candidate sites, 200 demand nodes, and 5 products, one second-stage problem might
have 20,000 flow variables. At 500 scenarios that is 10 million columns before any integrality.
The estimate is not a formality: it tells you whether you need decomposition (topic 10) or a
smaller scenario set. Size grows linearly in scenarios for a two-stage problem in
deterministic-equivalent form and multiplicatively in a multi-stage tree, as section 5.1 shows.
Most failed stochastic projects failed at this step, by building a model that could be written
and not solved.

### Step 6 — Generate or design the scenarios

There are two routes, and they are not interchangeable. A **designed set** is three to six
coherent futures, built from narratives, with named correlations; this is topic 07's method and
the right route for a capital decision, because each scenario can be argued with and each
survives into the minutes. A **sampled set** is hundreds or thousands of scenarios drawn from a
probability model, or reduced from a larger sample; this is the route for a stochastic
programme, and it requires a distribution, which requires a forecasting model (Module 08) and a
defensible dependence structure.

The two routes answer different questions and mixing them produces the worst of both. A designed
set of five scenarios fed into a stochastic programme produces an expected cost with five
weights and no statistical meaning. A sample of a thousand scenarios presented to a committee
produces a number nobody can argue with and nobody believes. **Branch:** if you cannot state a
distribution, stop. Use the designed set with topic 07's regret criterion, or go to step 10 and
use a robust set.

### Step 7 — Reduce the scenario set, and report the reduction

A reduced set of `N` scenarios with reassigned probabilities is reported with the distance
between the original and reduced measures and the algorithm used. A full tree explodes, and the
explosion is a property of the problem rather than a technical inconvenience. The reduction is a
modelling choice because it changes the objective: the reduced problem's optimum is optimal for
the reduced measure, not for the original one. Stability theory (Dupačová, Gröwe-Kuska, and
Römisch, 2003) bounds the change in optimal value by a distance between the measures, which is
why the distance is worth reporting.

### Step 8 — Solve, then test the first-stage decision out of sample

The in-sample objective is optimistic. The design was chosen to minimise the average over
exactly these scenarios, so it has been fitted to them. Evaluating it on a fresh sample gives an
unbiased estimate of its expected performance under the assumed distribution, and the gap
between in-sample and out-of-sample is a direct measure of how much the design was tuned to the
sample. Report both. A design whose out-of-sample cost is materially worse than its in-sample
cost has learned the sample, not the problem.

For a sample average approximation, repeat the whole procedure on `M` independent samples and
report the mean and spread of the first-stage designs and of the objective. Section 5.4 states
what the resulting confidence interval does and does not mean.

### Step 9 — Compute VSS and EVPI, and say what each one means

Four numbers and two differences:

```text
EV   = min over x of cost(x, expected parameters)
EEV  = expected cost of the EV-optimal design across the scenarios
RP   = min over x of expected cost across the scenarios, with recourse
WS   = expected value of the per-scenario optima

VSS  = EEV - RP      the value of using the stochastic model
EVPI = RP  - WS      the value of knowing the future before committing
```

These are the only two numbers that tell a reader whether the stochastic machinery earned its
cost, and their interpretation is asymmetric.

- A **large VSS** means the deterministic mean model would have chosen a materially worse
  design, so the stochastic model was worth building. A **small VSS** means the mean model was
  already close to right, and the extra machinery bought precision rather than a decision.
- A **large EVPI** means information about the future is valuable, which usually means the
  design is exposed to a scenario it handles badly. A **small EVPI** means even perfect foresight
  would not change the decision much, which is a statement about the decision, not about the
  importance of uncertainty.

**Sign conventions.** Everything above is written for minimisation; for maximisation the
differences reverse. State the direction in the report, because a reader who assumes the other
convention will read a negative VSS as a failure of the method. **What VSS is not:** it is not
the value of the software, the forecast, or information. It is the value of replacing one model
with another, and it depends entirely on which deterministic model you would otherwise have
used. Report the comparator, and read the pair as a stopping rule: large VSS with small EVPI means
adopt the stochastic design and stop buying forecast precision; small VSS with large EVPI means
buy information or a hedge, not a new design; small with small means the designed set was enough.

### Step 10 — Build the robust alternative and price the protection

The robust problem is `min over x of max over u in U of cost(x, u)`. The set shape is the model's
claim about how bad the world can be, and the price of robustness is what that claim costs; the
three shapes are tabulated in section 5.4. A box set says every combination of extremes is
possible, so for demand across four regions it means all four peak simultaneously, which is
usually false. A budget set with Gamma equal to two says at most two regions peak, a much better
description of a correlated world, and it costs much less to defend.

There is no correct Gamma, and the choice should be reported. Compute the robust design at Gamma
equal to 1, 2, and the full count, plot cost against Gamma, and identify the point at which
additional protection stops buying anything the business values. Bertsimas and Sim (2004) show
that the marginal cost of protection is the product of the protection variable and the dual of
the constraint, which is a useful way to see where the curve flattens.

### Step 11 — Choose the family, state the criterion, and write to the standard

The choice among families is a management decision about what evidence exists, not an analytical
preference. A team that uses a stochastic model because it is more sophisticated, when the
probabilities are invented, has produced a more precise version of a guess. A team that uses a
robust model because it is easier, when the probabilities are genuinely well estimated, has paid
a conservatism premium for nothing.

The decision rule, stated plainly:

1. If probabilities are defensible and recourse is real, use the stochastic family, and report
   VSS and EVPI.
2. If probabilities are not defensible but the possible set is, use the robust family, and report
   the price of robustness at the chosen protection level.
3. If neither is defensible, use the designed scenario set with regret and stability (topic 07),
   and report the flip.
4. If the first-stage decision is cheap to reverse, use none of them; run the deterministic model
   and revisit.

### 5.1 The scenario tree and the arithmetic of its size

A scenario tree with branching factor `b` and `T` stages of uncertainty (so `T` transitions after
the first decision) has `b^T` leaves and `(b^(T+1) - 1) / (b - 1)` nodes in total.

| Branching factor | Depth in stages | Leaves | Total nodes | Comment |
|---:|---:|---:|---:|---|
| 2 | 4 | 16 | 31 | Small enough for a spreadsheet |
| 3 | 4 | 81 | 121 | A typical four-period study |
| 3 | 6 | 729 | 1,093 | Getting large for a MILP |
| 3 | 10 | 59,049 | 88,573 | Beyond a direct deterministic equivalent |
| 5 | 10 | 9,765,625 | 12,207,031 | Not solvable as written |
| 10 | 6 | 1,000,000 | 1,111,111 | The same explosion with wider branches |

The lesson is that the growth is exponential in depth, so the number of periods is the dominant
design decision, and most practical studies keep the tree shallow — two or three stages — even
when the physical horizon is ten years. A three-stage model with a first decision, a mid-horizon
expansion, and an operational response is often the right compromise: deep enough to capture the
option value of waiting, shallow enough to solve.

```text
   depth 0            depth 1            depth 2            depth 3
     x                 node               node               leaf
     |                / | \              / | \              / | \
     +---------------o  o  o------------o  o  o------------o  o  o
                      \ | /              \ | /              \ | /
                       ...                ...                ...
   one first-stage     b nodes            b^2 nodes          b^3 leaves
   decision            after one          after two          after three
                       realisation        realisations       realisations
```

> **Note:** the deterministic equivalent of a tree with 59,049 leaves and 2,000 second-stage
> variables per leaf has about 118 million columns. That is not a model you tune. It is a model
> you decompose, reduce, or refuse to build. Topic 10 covers the decomposition; this topic's
> contribution is knowing the number before the solver does.

### 5.2 Scenario reduction is a modelling choice with consequences

Scenario reduction selects a subset of `N` scenarios from a larger set and reassigns
probabilities so that the reduced measure is close to the original in a probability metric.
Dupačová, Gröwe-Kuska, and Römisch (2003) formalise this: the optimal reduction minimises a
Fortet–Mourier-type distance, the problem decomposes into choosing which scenarios to delete and
how to redistribute their probabilities, and the deletion problem is NP-hard, so fast greedy
heuristics are used in practice.

Three consequences follow, and each belongs in the report. **The objective changes:** the
reduced problem's optimum is optimal for the reduced measure, so if the reduction deleted the
scenario that made a design expensive, the design looks better than it is. **The first-stage
decision can change:** a reduction that preserves the mean but loses the tails can reverse a
ranking between a flexible design and a cheap one, because the flexible design's advantage lives
in the tails. **The choice of N is a judgement:** compute the first-stage decision at several
values of `N`, report whether it is stable, and stop increasing `N` when the decision stops
changing — the same stability logic as topic 07, applied to sample size rather than scenario set.

### 5.3 Sample average approximation, and the honest statement

Sample average approximation replaces the expectation with an average over `N` independently
sampled scenarios and solves the resulting deterministic equivalent. The optimal value of the
sampled problem is a random variable. Its expectation is a lower bound on the true optimal value
for a minimisation problem, and the gap shrinks as `N` grows. Repeating the procedure on `M`
independent samples gives `M` optimal values and `M` first-stage designs; the standard error of
the mean objective is estimated from the spread across replications, and a confidence interval
follows.

The interval is an interval for the expected optimal value of the *sampled* problem at sample
size `N`. It is not an interval for the true optimum, and it does not account for the
possibility that the sampling distribution is wrong. Kleywegt, Shapiro, and Homem-de-Mello
(2002) give the convergence theory and the practical recipe.

What the interval does not mean, and this is the sentence to put in the report: **an SAA
solution is optimal for the sampled scenarios, not for the world.** The sample is a finite draw,
the distribution is an assumption, and the optimisation will exploit both. The three defences
are out-of-sample evaluation, replication across samples, and a stability report on the
first-stage design. If the design changes across replications, the sample is too small or the
problem is genuinely ambiguous, and both findings are more useful than the point estimate.

### 5.4 Robust optimisation: the uncertainty set and the price of robustness

| Set | Definition | Worst case | Counterpart | Conservatism |
|---|---|---|---|---|
| Box | `abs(u_i) <= 1` independently | A corner: all deviate at once | Linear | Highest |
| Budget | `sum abs(u_i) <= Gamma` | A face: Gamma deviations | Linear | Tunable |
| Ellipsoidal | `norm(u, 2) <= Omega` | A point on the ellipsoid | Second-order cone | Lower |

Box suits parameters driven by a common cause and misfits independent ones, because its corner
defends against a joint extreme that cannot occur. Budget is the practical default for integer
models: the counterpart stays linear (Bertsimas and Sim, 2004), `Gamma = 0` is nominal and
`Gamma = n` is the box, and the cost curve against `Gamma` is the deliverable. Ellipsoidal buys
less conservatism at the cost of linearity. The **price of robustness** is `(robust cost -
baseline) / baseline`. State the baseline — the stochastic optimum is more honest than the
infeasible nominal one — the protection level, and the units. The premium is not a welfare
loss: it buys a guarantee the objective does not measure.

### 5.5 The two critiques, stated at full strength

**The critique of robust optimisation: the worst case may be implausible.** A box set contains
the corner at which every demand is high, every cost is high, and every facility is disrupted.
If that corner is physically impossible, the model defends against a fiction and the premium
buys nothing. Ellipsoidal and budget sets are both responses: they shrink the set to the part
worth defending. The critique is answered by showing the worst point is credible, or by
shrinking the set until it is — not by saying "we are prudent".

**The counter-critique of stochastic programming: the scenario set may be too narrow.** A
stochastic model optimises against the futures it was given. If the set omits the disruption,
the model will not hedge it, and the design will look robust because the test was easy.
Confident weights on a narrow set produce a precise expected cost that is precise about the
wrong distribution. The answer is not more scenarios but out-of-sample testing, a reported
generation method, and the question "what does this design do in the future we did not model".

**The synthesis.** Robust optimisation is exposed to a wrong *set*; stochastic programming to a
wrong *measure*. A study that checks both is doing the right thing regardless of family.

### 5.6 Distributionally robust optimisation, at a usable level

DRO takes a set of distributions — the ambiguity set — and minimises the worst expected cost
over it: `min over x of max over P in P of E_P[cost(x, xi)]`. A singleton set makes it stochastic
programming; every distribution on the support makes it robust optimisation. The practical
value is in between: enough data to rule out some distributions, not enough to commit to one.

Two families are common. **Moment-based** sets fix the mean and covariance within confidence
regions estimated from data; Delage and Ye (2010) give a tractable formulation and a data-driven
region for the moments. **Distance-based** sets, usually on the Wasserstein distance, contain
every distribution within a stated distance of the empirical one; the radius is the dial and can
be calibrated to contain the truth at a stated confidence under assumptions. DRO is worth
considering with a few hundred observations. Report the ambiguity set: a DRO answer without its
set is as uninterpretable as a robust answer without its Gamma.

### 5.7 The practical cost, in four currencies

| Cost | What drives it | Typical scale | How to reduce it |
|---|---|---|---|
| Model size | Scenarios times second-stage variables | Linear in scenarios for two-stage; exponential in depth for multi-stage | Scenario reduction; shorter horizon; aggregation |
| Solve time | Integrality, scenarios, and the LP relaxation gap | Hours for a large deterministic equivalent; minutes with decomposition | Decomposition (topic 10); looser gap; heuristic first-stage |
| Data | Scenario generation, probabilities, dependence structure | A distribution per uncertain parameter, plus correlations | Designed scenarios; moment-based ambiguity sets |
| Organisational agreement | Getting commercial, finance, and operations to sign the scenario set | Weeks, and the usual cause of project failure | Pre-register the set; give each function a veto and a reason to use it |

The fourth row is the one always underestimated: a scenario set is a claim about the future that
several functions will be judged against, so the technical work takes days and the agreement
takes weeks.

### 5.8 The reporting standard

A study that uses any of these methods reports six things. A study that omits more than one
cannot be reviewed.

| # | Item | What must be stated |
|---:|---|---|
| 1 | Uncertainty representation | Scenario set, distribution, uncertainty set, or ambiguity set, with the claim each makes |
| 2 | Scenario count or set definition | The number and how it was chosen, or the set's bounds and parameters |
| 3 | First-stage variables | The complete list, with the commitment date that makes each first-stage |
| 4 | Second-stage variables | The complete list, with the response time that makes each available |
| 5 | Diagnostics | VSS and EVPI where the stochastic family is used; the price of robustness and Gamma where the robust family is used |
| 6 | First-stage stability | Whether the design changes across samples, reductions, or protection levels, and by how much |

Item 6 is the one most often missing and most often decisive. A design that is stable across
every sample and every reduction is a design the committee can commit to. A design that
oscillates is a signal that the problem is genuinely ambiguous, and the honest response is to
buy optionality or phase the investment, not to report the modal answer as if it were certain.

## 6. Worked example

A regional distributor serves one product family from one of two candidate DCs. The instance has
one structural decision, two demand scenarios, and numbers small enough to check by hand. The
structure is the lesson; the magnitudes are chosen for arithmetic, not for realism. Costs are in
thousand currency units per year at 2026 prices.

### 6.1 The instance

| Item | Small DC (S) | Large DC (L) |
|---|---:|---:|
| Capacity, thousand units per year | 90 | 130 |
| Fixed cost, thousand per year | 300 | 500 |
| Regular flow cost, currency per unit | 4 | 4 |
| Expedite cost above capacity, currency per unit | 12 | 12 |

Demand is 90 thousand units in the low scenario with probability 0.5, and 150 thousand units in
the high scenario with probability 0.5. The expected demand is 120 thousand units. Expedite is
an all-in cost that replaces the regular flow cost for the units above capacity; it represents
spot capacity and premium freight, and it is available in both scenarios because the distributor
holds a framework agreement with a third-party logistics provider.

The cost of a design in a scenario is therefore:

```text
cost(design, d) = fixed + 4 * min(d, capacity) + 12 * max(0, d - capacity)
```

### 6.2 The four numbers, computed by hand

| Design | Low scenario, d = 90 | High scenario, d = 150 | Expected cost |
|---|---:|---:|---:|
| S | 300 + 360 = 660 | 300 + 360 + 720 = 1,380 | 0.5 × 660 + 0.5 × 1,380 = 1,020 |
| L | 500 + 360 = 860 | 500 + 520 + 240 = 1,260 | 0.5 × 860 + 0.5 × 1,260 = 1,060 |

The stochastic programme chooses S, at an expected cost of 1,020. It accepts 60 thousand units
of expediting in the high scenario, at a cost of 720, rather than paying 200 more in fixed cost
in every scenario. That is the recourse decision, and it is the whole content of the model.

The deterministic mean model sees demand of 120 thousand units. At that demand, S costs
`300 + 360 + 360 = 1,020` and L costs `500 + 480 = 980`, so the mean model chooses L. The
expected cost of that choice is 1,060, not 980. The mean model's own objective, 980, is not
achievable in either scenario; it is the cost of a world that does not exist.

| Quantity | Value | Formula |
|---|---:|---|
| EV objective | 980 | cost of L at mean demand |
| EEV | 1,060 | expected cost of L across the two scenarios |
| RP | 1,020 | expected cost of S, the stochastic optimum |
| WS | 960 | 0.5 × 660 + 0.5 × 1,260, the per-scenario optima |
| VSS | 40 | EEV − RP = 1,060 − 1,020 |
| EVPI | 60 | RP − WS = 1,020 − 960 |

Four readings, in order of importance.

**The deterministic model chose the wrong design, and the VSS measures that.** Forty thousand
per year on a base of about one million is 4.0 per cent; the model was worth building because the
structural choice changed, not because the expected cost fell much.

**The EVPI is larger than the VSS, and that is not a contradiction.** With S in place, knowing the
future is worth 60 because in the high scenario the distributor would build L and save
`1,380 − 1,260 = 120`, weighted at 0.5; with the deterministic model, switching to the stochastic
design is worth 40. One diagnostic is about the model, the other about information.

**The probability barely matters.** Let `p` be the probability of the high scenario. S costs
`660 + 720p` and L costs `860 + 400p`, so S is optimal whenever `p <= 0.625`. The recommendation
holds across most of the plausible range: the model needed to know that the high scenario was not
near-certain, not a precise probability.

**The mean model's objective was unattainable.** The 980 is the cost of L at demand 120, and
demand is never 120. A deterministic study reporting 980 and a stochastic study reporting 1,020
look as if the stochastic model is worse, when the first number is a fiction.

### 6.3 The robust comparison

The robust version minimises the worst-case cost over the two scenarios.

| Design | Worst-case cost | Maximum regret | Criterion selects |
|---|---:|---:|---|
| S | 1,380 | 120 (high scenario, against L) | Minimax regret |
| L | 1,260 | 200 (low scenario, against S) | Minimax cost |

Minimax cost chooses L at 1,260. The price of robustness, measured against the stochastic
optimum, is `1,260 − 1,020 = 240`, or 23.5 per cent. Measured against the nominal mean-model
objective of 980, it would be 28.6 per cent, which is the misleading comparison: the 980 is not
achievable.

Minimax regret chooses S, because S's worst regret is 120 against L's 200. So the two robust
criteria disagree, and the disagreement is informative. Minimax cost asks "what if the worst
happens"; minimax regret asks "how badly would I wish I had chosen otherwise". In this instance
the worst case is the high scenario, which is not catastrophic for S — it costs 120 more than L
— so the regret criterion is comfortable with S, while the cost criterion is not.

The robust answer is also the answer that ignores the probability entirely. It chooses L, which
is optimal only when `p > 0.625`. In other words, minimax cost behaves as if the high scenario
were certain, and it pays 240 for that assumption. This is the over-conservatism critique in
one line, and it is why the budget set exists: with a set that allows only one of the two
scenarios to be adverse at a time, the worst case for S becomes 1,020 rather than 1,380, and S
is selected.

> **Note:** the example has two scenarios, so a budget set with Gamma equal to one is the same
> as the nominal set and Gamma equal to two is the same as the box. That degeneracy is an
> artefact of the size, not a property of budget sets. In a problem with ten uncertain
> parameters the intermediate values of Gamma are where the useful answers live.

### 6.4 What the analysis says, and what it does not

The recommendation, written in the form the reporting standard requires:

> Open the small DC, at an expected cost of 1,020 thousand per year at 2026 prices, with a
> contractual expedite arrangement covering up to 60 thousand units per year. The design is
> optimal for any high-scenario probability below 0.625. Using the deterministic mean model
> instead would open the large DC at an expected cost of 1,060, a loss of 40 thousand per year
> (the VSS). Perfect information about the demand scenario would be worth 60 thousand per year
> (the EVPI). A minimax-cost robust design would open the large DC at a worst-case cost of
> 1,260, a premium of 240 thousand per year over the stochastic optimum. The first-stage
> decision is stable across the probability range 0 to 0.625 and across the ambiguity set
> `p in [0.4, 0.6]`; it changes only if the high scenario is judged more than 62.5 per cent
> likely.

What the analysis does not say: whether the expedite agreement will actually be available at
12 per unit in a high-demand year, which is a procurement question and the single assumption
most likely to break the recommendation. If expedite costs 20 rather than 12, the high-scenario
cost of S rises to `300 + 360 + 1,200 = 1,860`, and the expedite advantage narrows sharply.
The analysis has converted that question into a threshold, which is its job. Section 7 does the
sensitivity in one line.

## 7. Code, query, or calculation

The calculation reproduces every number in section 6 with a pencil. It is the preferred form
for a two-scenario instance because it shows the three objects a solver call hides: which
design each criterion selects, how the diagnostics are formed, and where the probability
threshold sits.

```text
S: cap 90, fixed 300;  L: cap 130, fixed 500;  regular 4;  expedite 12
demand 90 (p) or 150 (1-p);  mean 120

cost(design,d) = fixed + 4*min(d,cap) + 12*max(0,d-cap)

S:  90 -> 300 + 360          =   660
   150 -> 300 + 360 + 720    = 1,380      E = 1,020 at p = 0.5
L:  90 -> 500 + 360          =   860
   150 -> 500 + 520 + 240    = 1,260      E = 1,060 at p = 0.5

RP   = min(1,020, 1,060)                  = 1,020   design S
EV   : d = 120 -> S 1,020, L 980          =   980   design L
EEV  = E[cost(L, .)]                      = 1,060
VSS  = 1,060 - 1,020                      =    40
WS   = 0.5*660 + 0.5*1,260                =   960
EVPI = 1,020 - 960                        =    60

minimax cost: max(660,1,380)=1,380 vs max(860,1,260)=1,260 -> L
price of robustness = 1,260 - 1,020       =   240   (23.5 per cent)
max regret: S max(0,120)=120; L max(200,0)=200 -> S

threshold: 660 + 720p = 860 + 400p  ->  p = 0.625
expedite at 20: S(150) = 300+360+1,200 = 1,860; E = 1,260 -> L wins at p = 0.5
```

Two checks before trusting the table. **Every cost is in the same units and the same price
year.** The fixed costs are annual, the flow costs are per unit, and the scenario demands are
annual; mixing a monthly fixed cost with an annual demand would scale some terms and not
others. **The expectation uses the same probabilities as the decision.** The 0.5 appears in the
RP, the EEV, and the WS; using different weights in different rows is the most common arithmetic
error in a VSS calculation, and it produces a sign that looks like a finding.

The last line is the sensitivity that matters. Raising the expedite cost from 12 to 20 raises
S's high-scenario cost to 1,860 and its expected cost to 1,260, at which point L wins. The
recommendation therefore has a procurement threshold — expedite must stay below about 13.3 per
unit for S to remain optimal at `p = 0.5` — and that threshold is a question someone can take to
the third-party logistics provider.

## 8. Data, standards, and regulatory context

No standard prescribes stochastic programming, robust optimisation, or a scenario count. That
is worth stating at the start, because committees sometimes ask which regulation requires a
particular method, and the honest answer is none. What the standards do govern is the risk
process around the model, the accounting of the costs inside it, and the reporting of the
disruption scenarios that the model evaluates.

| Source | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 31000:2018 | Risk management principles and process | Requires risk to be identified, analysed, and evaluated; does not prescribe the quantification | B |
| ISO/IEC 31010:2019 | Risk assessment techniques | Lists scenario analysis and sensitivity analysis among recognised techniques; does not rank them or set ranges | B |
| ISO 22301:2019 | Business continuity management systems | Requires disruption scenarios and recovery objectives; supplies the vocabulary for the availability scenarios in section 5.8 | B |
| ISO 28000:2022 | Security and resilience in the supply chain | Requires a security management system across the chain; does not prescribe a network model | B |
| ISO 14083:2023 | Quantification of greenhouse gas emissions from transport chains | Fixes the emission factors if carbon is an uncertain parameter in the model | B |
| IFRS 16 / ASC 842 | Lease accounting | Determines the annual cash cost that belongs in the fixed-charge term; the profit-and-loss charge is a different number | B |
| US Federal Reserve SR 11-7 | Model risk management | Documentation, validation, and named limitations for models used in decisions; applies by analogy outside banking | B |
| World Customs Organization HS nomenclature | Product classification | Tariff scenarios must key on HS codes, or the duty change cannot be represented | B |

**What is governed, and what that means for the model.** ISO 22301 requires an organisation to
identify disruptive incidents and to set recovery objectives, which is the organisational
justification for disruption scenarios. It does not tell you how many facilities may fail
simultaneously. ISO 31000 requires risk to be evaluated, which is the justification for
reporting the price of robustness; it does not tell you what protection level to buy. The
standards create the obligation to analyse and to document. The method remains a professional
judgement, and the report has to defend it.

**The data requirements.** A stochastic or robust model consumes four kinds of data, each with
its own quality standard. **Demand history or a model:** if the source is a probabilistic
forecast, Module 08 owns its calibration; if it is judgement, the judgement is the data and must
be named. **Costs with a price year:** mixing vintages inside a scenario produces a ranking that
neither year supports (topic 07). **Response capability:** expedite, reallocation, and spot
purchase are data — capacity, cost, and lead time — not assumptions. **Disruption history:**
frequencies, durations, and correlations from incident logs or industry sources; where the record
is too thin, report scenarios with stated durations and say that frequencies are not estimated.

> **Note:** ISO/IEC 31010 lists scenario analysis and sensitivity analysis as techniques and
> explicitly declines to rank them. A claim that a standard "requires" a stochastic model is
> overstating the guidance. The correct sentence is that the standard requires a documented
> treatment of uncertainty, and the method is the analyst's choice.

## 9. Industry practice and case evidence

**What the review literature reports.** Snyder's review of facility location under uncertainty
(2006) states the standard division of labour: locations are generally first-stage decisions,
while customer assignments are second-stage, or recourse, decisions. It covers stochastic
formulations (minimum expected cost, mean-variance, chance-constrained) and robust ones (minimax
cost, minimax regret, p-robust), and reports that the two streams developed largely in parallel
with limited cross-referencing — which explains why a practitioner often meets one tradition and
not the other. **A**

**What the reliability literature reports.** Snyder and Daskin (2005) formulate facility location
models that account for expected transportation cost after facility failures, allowing multiple
simultaneous failures with equal probabilities, and solve them by Lagrangian relaxation. They
report trade-off curves between operating cost and expected failure cost, and find that
substantial reliability improvements are often achievable for small increases in operating cost.
The finding is about a model class and a set of test instances, not a universal law, but it is
the most-cited quantitative statement of the resilience premium. **A**

**What the network-design literature reports.** Klibi, Martel, and Guitouni (2010) review robust
supply chain network design and observe that the dominant practice is a deterministic model
followed by sensitivity and scenario analysis. They argue this is insufficient for long-horizon
decisions under deep uncertainty, and they organise the robust approaches by the kind of
uncertainty they address. Govindan, Fattahi, and Keyvanshokooh (2017) review supply chain network
design under uncertainty more broadly and reach a similar conclusion about the gap between
method and practice. **A**

**What one worked stochastic study reports.** A 2026 preprint applies two-stage stochastic
optimisation to a capacitated facility-location instance on a public benchmark, with opening as
the first stage and allocation plus shortage as recourse, across three demand scenarios. It
reports a VSS of about 82,933 currency units against an EVPI of about 5,149, and states that the
VSS remains between roughly 10 and 17 times the EVPI under alternative scenario structures. The
direction — VSS much larger than EVPI — matches section 6. The magnitudes do not transfer, and
the paper is a preprint rather than a peer-reviewed article. **C**

| Practice | What is observed | Grade | How to read it |
|---|---|---|---|
| Deterministic model plus scenario table | The norm in published and practitioner network studies | C | Prevalence, not quality |
| Stochastic modules in design software | Vendors present scenario and stochastic optimisation as product features | D | A feature is not a validated method |
| Reliability and p-robust models | Well developed in the academic literature, thinly used in practice | A | The rigorous version of "do not be too wrong anywhere" |
| Disruption scenarios in continuity plans | Required by ISO 22301, usually qualitative | B | The organisational hook for quantified disruption scenarios |
| Published company cases of two-stage network design | Scarce, and rarely with auditable cost data | C | The evidence base is methodological, not commercial |

**The honest gap.** The public record contains few named companies reporting a two-stage
stochastic network design with audited costs; most evidence is methodological, a vendor claim
(grade D), or a consultancy claim without a data appendix (grade C). Present the diagnostics
from your own instance rather than cite someone else's percentage. Practice lags because
probabilities are often unestimable, the deterministic equivalent needs specialist decomposition
(topic 10), and agreeing a scenario set is slow (section 5.7).

**The common finding about designed sets.** In our reading of practice, and consistent with
topic 07's observation that three to six named scenarios are the norm, a well-constructed designed
set with a regret table and a stability test captures most of the decision value of a full
stochastic programme at a fraction of the cost. The mechanism is that most network decisions are
dominated by one or two structural questions — is demand above or below a capacity threshold, is
a site available or not — and a designed set that spans those questions answers them directly.
This is a professional judgement rather than a measured result, and it should be tested in each
instance by computing the VSS: if the VSS is small, the designed set was enough. **C**

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| First-stage variable indexed by scenario | A facility-opening variable carries a scenario subscript | The formulation gave the model foresight | The design is infeasible in practice and the cost is understated | Check that no first-stage variable depends on the scenario; impose non-anticipativity explicitly in multi-stage models |
| Probabilities invented to make the model run | Scenario weights are equal "for neutrality", or sum to one with no source | Expected cost requires weights, and weights require a judgement | A precise number with no evidential basis | Report the weights' source; if none, switch to a robust or regret criterion |
| Scenario set too narrow | Every scenario is within plus or minus 20 per cent of the base | The set was built by scaling one variable | The design is optimised for futures that were easy to imagine | Include a structural break and a disruption scenario; run out-of-sample |
| VSS computed against the wrong comparator | The report gives a VSS but not the deterministic design it beat | The comparator was a different model or a different objective | The number is uninterpretable and often flattering | State the deterministic model, its design, and its expected cost |
| EVPI read as the value of forecasting | "Perfect information is worth 60, so buy a better forecast" | EVPI is an upper bound on what information is worth, before its cost | Forecast investment sized to an unattainable ceiling | Compare EVPI with the cost of the forecast improvement, not with zero |
| Scenario reduction unreported | The model uses 100 scenarios, and the source set had 10,000 | Reduction is treated as preprocessing | The objective is optimal for a measure nobody examined | Report N, the algorithm, and the distance; test first-stage stability across reductions |
| SAA reported as the true optimum | A single sample, a single solve, a point estimate with three decimals | The sample optimum is a random variable | False precision; the design may not replicate | Replicate across samples; report the spread; evaluate out of sample |
| Box uncertainty set with independent parameters | Every parameter is allowed to hit its extreme simultaneously | The set was chosen for linearity | The worst case is physically impossible and the premium is wasted | Use a budget or ellipsoidal set; justify Gamma |
| Gamma reported as a probability | "Gamma equals two, so we are protected with 95 per cent confidence" | A budget set is not a probability statement | The committee believes a guarantee that does not exist | Report Gamma as a protection level; if a probability is needed, use the Bertsimas–Sim bound and label it a bound |
| Robust design compared with the nominal optimum | The price of robustness is measured against an unachievable cost | The nominal optimum is often infeasible under uncertainty | The premium is overstated and the method looks worse than it is | Compare against the stochastic optimum or a feasible deterministic design |
| Recourse assumed available when it is not | The second stage expedites, but no framework agreement exists | The model invented operational flexibility | The expected cost is unachievable and the VSS is fictitious | List the second-stage actions and their contracts; if an action cannot be executed, remove it |
| Disruption modelled as a cost increase | A port closure appears as a 30 per cent freight surcharge | A cost surcharge keeps the lane open | The structural consequence, loss of the lane, is missed | Model unavailability as a capacity or arc removal, not a price change |
| One scenario dominates the expectation | One scenario carries 90 per cent of the weight | The weights were set to make the base case dominant | The model is a deterministic study with extra notation | Check the weight distribution; report the effective number of scenarios |

**First-stage variable indexed by scenario.** This is the failure that looks like competence.
The model is larger, the output is richer, and the cost is lower, because the model is allowed
to choose a different network for each future and then report the average. It is not a
stochastic programme. It is a set of deterministic studies with a shared spreadsheet, and its
answer cannot be built. The check is mechanical: search the formulation for any first-stage
symbol carrying a scenario index, and remove it.

**Disruption modelled as a cost increase.** A facility that is unavailable is not a facility
with higher operating cost. It cannot serve demand at any price, and the network must reroute.
Modelling the disruption as a surcharge leaves the capacity in place and understates the
consequence, often by an order of magnitude. The correct pattern is to set the facility's
capacity to zero for the duration, or to remove the arc, and to let the recourse variables carry
the rerouting cost.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails |
|---|---|---|
| The first-stage and second-stage split matches the commitment dates | The model's value depends on the split being real | A variable placed in the second stage that is physically fixed produces an unachievable cost; the reverse produces an unnecessarily rigid design |
| Recourse actions are available at the modelled cost | The second stage is a claim about operational flexibility | The expected cost is optimistic and the VSS is fictitious |
| Probabilities are defensible, if the stochastic family is used | Expected cost is a probability-weighted average | The objective is an arbitrary weighted average wearing the costume of a measurement |
| The scenario set spans the decision-relevant futures | The optimum is optimal for the set, not the world | A design that fails in an unmodelled future looks robust |
| The uncertainty set contains the plausible worst case, if the robust family is used | The guarantee is only as good as the set | The design is protected against the wrong adversary |
| The worst point in the uncertainty set is credible | Conservatism has a cost | The premium buys protection against an impossible combination |
| Fixed recourse, or the departure from it is modelled | Standard algorithms and convexity results assume it | If the response itself can be disrupted, the recourse value is overstated |
| The distribution or ambiguity set is stable over the horizon | The model is solved once for a long commitment | A regime change invalidates the estimated distribution and the design with it |
| The objective captures the decision-maker's preferences | The model optimises what it is given | A design that is optimal for expected cost and unacceptable for tail risk is recommended without warning |
| The cost data share a price year and a unit basis | Differences of costs are the decision variable | Mixed vintages reverse rankings for reasons that have nothing to do with the business |

Two limits matter more than the rest.

**The model is only as good as its futures.** Neither family creates information. A stochastic
programme with a narrow scenario set and a robust programme with a small uncertainty set are
both confident answers to questions that were posed too narrowly. The diagnostics measure the
model's internal coherence, not its correspondence to the world. Out-of-sample testing and
scenario-set review are the only defences, and both are organisational.

**The methods compare designs, not transitions.** A stochastic model says which network to hold
in each future. It does not say how to get from the current network to that one, what the
transition costs, or what happens if the transition is interrupted. Topic 06 owns that question,
and a recommendation that ignores it is a recommendation about a company that does not exist
yet.

## 12. Exercises

### Exercise 1 — Classify the three families (recall, 25 minutes)

For each of the following, name the family and the assumption it makes: (i) a model that solves
the network for mean demand and reports the cost; (ii) a model that minimises expected cost with
a per-scenario reallocation variable; (iii) a model that minimises the maximum cost over a set of
demand and freight realisations; (iv) a model that minimises the worst expected cost over a set
of demand distributions whose means lie in a stated interval.

**Deliverable:** a four-row table with family, assumption, and the diagnostic that would tell you
whether the method was worth using. **Acceptance criteria:** each assumption stated in one
sentence; diagnostics named with units; (iv) correctly identified as distributionally robust.

### Exercise 2 — Compute the diagnostics (calculation, 45 minutes)

Using the section 6 instance, change the expedite cost from 12 to 18 per unit and recompute EV,
EEV, RP, WS, VSS, and EVPI at `p = 0.5`. Then find the expedite cost at which the stochastic
model switches from S to L.

**Deliverable:** the six numbers with formulas, and the threshold expedite cost. **Acceptance
criteria:** all arithmetic shown; the direction of each change explained; the threshold derived
algebraically, not by trial and error.

### Exercise 3 — Size the tree (calculation, 25 minutes)

A study has three uncertain parameters per period, four periods, and a branching factor of three.
Compute the number of leaves and the total number of nodes. Then compute the number of columns in
the deterministic equivalent if each second-stage problem has 1,500 variables and 100 first-stage
variables. State whether you would solve it directly.

**Deliverable:** the arithmetic and a one-paragraph recommendation. **Acceptance criteria:**
leaves and nodes correct; columns correct; the recommendation names decomposition or reduction,
or refuses the formulation with a reason.

### Exercise 4 — Write the formulation (application, 45 minutes)

A manufacturer must choose which of five plants to keep open for five years. Demand by region is
uncertain. After demand is known, the manufacturer can reallocate production among the open
plants, and can buy from a spot supplier at a premium. Write the two-stage formulation.

**Deliverable:** first-stage variables, second-stage variables, objective, and constraints.
**Acceptance criteria:** no first-stage variable carries a scenario index; the spot purchase is
in the second stage with a stated premium; the relatively-complete-recourse condition is checked
and the artificial variable, if used, is named and penalised.

### Exercise 5 — Critique the robust model (critique, 40 minutes)

A consultant proposes a box uncertainty set for demand in six regions, with each region allowed
to deviate by plus or minus 25 per cent independently. The price of robustness is 31 per cent.
Write a two-hundred-word critique.

**Deliverable:** the critique, with one alternative set and its justification. **Acceptance
criteria:** the independence assumption is challenged with a named common cause; the price is
compared with a feasible baseline; a budget or ellipsoidal alternative is proposed with a stated
Gamma or radius.

### Exercise 6 — Design a disruption scenario (design, 45 minutes)

A single plant supplies three regional DCs. The plant can be unavailable for four, eight, or
twelve weeks after a flood. Design the disruption modelling for both a stochastic and a robust
version.

**Deliverable:** the scenario definitions for the stochastic version, and the uncertainty set for
the robust version, with the first-stage and second-stage variables listed. **Acceptance
criteria:** unavailability is modelled as capacity removal, not a cost increase; the duration is
a parameter, not a single number; the robust set states how many facilities may fail at once.

### Exercise 7 — The SAA honesty memo (critique, 40 minutes)

A colleague reports a stochastic optimisation on 200 sampled scenarios, a single solve, and an
expected cost of 8.42 million with three decimal places. Write a one-page memo on what can and
cannot be concluded.

**Deliverable:** the memo, with the three defences and the specific tests you would run.
**Acceptance criteria:** the confidence interval is correctly described as applying to the
sampled problem; out-of-sample evaluation is specified; replication and first-stage stability are
named; no claim is made about the true optimum.

### Exercise 8 — Place DRO (application, 45 minutes)

For the section 6 instance, compute the DRO solution for ambiguity sets `p in [0.4, 0.6]` and
`p in [0.2, 0.8]`. Compare with the stochastic and minimax-cost answers.

**Deliverable:** a table of design, worst-case expected cost, and price of ambiguity for each set,
plus the stochastic and robust benchmarks. **Acceptance criteria:** the worst case is taken over
the stated interval, not over the endpoints of the demand range; the price of ambiguity is
measured against the stochastic optimum; widening the set is shown to move the answer toward the
robust one.

## 13. Self-check answers

**Exercise 1.** (i) Deterministic with scenarios, specifically the expected-value problem; it
assumes a single representative future, and has no diagnostic of its own — topic 07's stability
and regret tables substitute. (ii) Two-stage stochastic programming; it assumes probabilities are
defensible and recourse is available; the diagnostics are VSS and EVPI, in currency per period.
(iii) Robust optimisation; it assumes the uncertainty set is right and its worst point worth
defending; the diagnostic is the price of robustness at the stated protection level. (iv)
Distributionally robust optimisation; it assumes the true distribution lies in the ambiguity set
at the stated radius; the diagnostic is the price of ambiguity, and the set must be named.

**Exercise 2.** With expedite at 18: S in the high scenario costs `300 + 360 + 60 × 18 = 1,740`;
S in the low scenario is unchanged at 660. RP at `p = 0.5` is `0.5 × 660 + 0.5 × 1,740 = 1,200`.
L is unchanged at 1,060 expected, so the stochastic model now chooses L. The EV model at mean
demand 120 still chooses L at 980, so EEV = 1,060 and VSS = 0. WS is `0.5 × 660 + 0.5 × 1,260 =
960`, so EVPI = `1,060 − 960 = 100`. The threshold comes from `660 + 30e = 1,060`, giving
`e = 13.33`; above that, L is optimal at `p = 0.5`. The VSS falls to zero precisely when the
stochastic and deterministic models agree, which is the definition working, not a failure.

**Exercise 3.** Leaves are `3^4 = 81`; nodes are `(3^5 − 1) / (3 − 1) = 121`. Columns are `100 + 81
× 1,500 = 121,600`. The model is large but not impossible: a direct solve is plausible for a pure
linear programme, and decomposition or reduction to about 30 scenarios is advisable if any
first-stage variable is integer. State which and why, rather than quoting the size as a verdict.

**Exercise 4.** First stage: binary `x_j` for keeping plant `j` open, with fixed cost `f_j` and
capacity `cap_j`; no scenario index. Second stage, for scenario `s`: flow `y_ijs` from plant `j`
to region `i`, spot purchase `z_is`, and unmet demand `u_is`. Objective:
`min sum_j f_j x_j + sum_s pi_s [ sum_ij c_ij y_ijs + sum_i (p_spot z_is + p_pen u_is) ]`.
Constraints: `sum_i y_ijs <= cap_j x_j` for every `j, s`; `sum_j y_ijs + z_is + u_is >= d_is` for
every `i, s`; all variables non-negative. Relatively complete recourse holds because `u_is` can
always absorb the shortfall; `p_pen` must be large enough that it is chosen only when nothing
else is possible, and its value must be reported because it is a price on failure.

**Exercise 5.** A strong critique notes that six independent regional deviations imply a corner
in which all six regions peak at once, inconsistent with any common business cycle and with the
historical correlation between regions. That corner drives the worst case and the 31 per cent
premium. The premium is also measured against an unachievable nominal optimum, so the true cost
of protection is smaller than reported. A budget set with Gamma around 2, or an ellipsoidal set
with a radius calibrated to the observed covariance, would defend against the plausible joint
movement at a fraction of the cost. Ask what common cause could produce six simultaneous peaks,
and accept the box only if one exists.

**Exercise 6.** Stochastic version: scenarios are flood at the plant with duration 4, 8, or 12
weeks, with probabilities if they can be defended; plant capacity is zero for the duration, and
the DCs' recourse variables are reallocation, expedite, and unmet demand. First stage: plant and
DC capacity commitments and any pre-positioned inventory. Second stage: rerouting, spot
purchase, and demand allocation by week. Robust version: an uncertainty set in which at most one
facility is unavailable at a time, with a duration interval of 4 to 12 weeks; if the business
judges two simultaneous failures possible, the cardinality budget is two, and the price of that
decision is computed. Duration matters more than probability here, which is why the robust
framing is natural.

**Exercise 7.** The memo should say: 8.42 million is the optimal value of the problem defined by
the 200 sampled scenarios; it is a random variable, and a single solve gives no standard error;
the confidence interval from replication applies to the expected value of the sampled problem,
not to the true optimum; the design may not replicate across samples; and the three defences are
replication across independent samples, out-of-sample evaluation, and a stability report on the
first-stage design. The three decimal places should be removed. If the design is stable across
ten replications and the out-of-sample cost is close to the in-sample cost, the result is credible
under the sampling distribution; if not, the sample is too small or the problem is ambiguous.

**Exercise 8.** For `p in [0.4, 0.6]`: S's worst case is at `p = 0.6`, `660 + 432 = 1,092`; L's
worst case is also at `p = 0.6`, `860 + 240 = 1,100`; DRO selects S at 1,092, a price of ambiguity
of 72 against the stochastic 1,020. For `p in [0.2, 0.8]`: S's worst case is at `p = 0.8`,
`660 + 576 = 1,236`; L's is at `p = 0.8`, `860 + 320 = 1,180`; DRO selects L at 1,180, a price of
ambiguity of 160. Widening the set moves the answer from the stochastic choice to the robust one:
the ambiguity set is a dial between confidence and pessimism, and the wider it is, the closer the
answer comes to minimax cost.

## 14. Interview questions

**1. "What is the difference between stochastic programming and robust optimisation?"**
Strong: stochastic programming weights futures by probability and minimises expected cost with
recourse; robust optimisation takes the worst case over a set and uses no probabilities. The first
is exposed to a wrong measure, the second to a wrong set, and both return a first-stage design plus
a diagnostic. Weak: "robust is more conservative", with no account of when that is justified.

**2. "Why can the deterministic mean model be worse than it looks?"**
Strong: the mean is not a realisable scenario for a discrete design; a design cannot be
half-built, so the mean model may choose a design no averaged world would prefer; its reported
objective is unattainable; and the VSS measures the gap. Weak: "because it ignores uncertainty",
which does not explain why the error is structural rather than a matter of precision.

**3. "What is VSS and what is EVPI, and why report both?"**
Strong: VSS is EEV minus RP, the value of replacing the deterministic mean model with the
stochastic one; EVPI is RP minus WS, the most worth paying to learn the future before committing.
They answer different questions and differ by orders of magnitude in either direction. Weak:
treating them as two names for "the value of uncertainty", or reporting a VSS with no comparator.

**4. "A model has 500 scenarios and a nine-hour solve. What do you ask first?"**
Strong: the tree depth and the first-stage stability across reductions, then the out-of-sample
cost, then whether the 500 were sampled or designed. The nine hours is a symptom. Weak: asking
about the solver or the hardware before the scenario-generation method.

**5. "What is non-anticipativity, and where does it break?"**
Strong: the requirement that a first-stage decision cannot depend on which scenario occurs. It is
automatic in a two-stage formulation without scenario-indexed first-stage variables, and must be
imposed explicitly in multi-stage models and in decomposition, where subproblems are solved
separately. Weak: calling it a solver setting, or being unable to name the constraint.

**6. "When is a robust model too conservative?"**
Strong: when the uncertainty set contains combinations that cannot co-occur — every region
peaking, every facility failing, every cost rising at once — and the worst point drives the
design. The remedy is a budget or ellipsoidal set, with the protection level reported and the
price computed at two or three levels. Weak: "always", which is a slogan, or "never", which
ignores the geometry of the set.

**7. "When is a stochastic model too optimistic?"**
Strong: when the scenario set is narrow, the probabilities are asserted rather than estimated, and
the recourse assumes operational flexibility that does not exist. The symptoms are an unmeasured
out-of-sample gap and a VSS computed against a deliberately weak comparator. Weak: "when the
forecast is wrong", which describes every model.

**8. "What is distributionally robust optimisation, in one minute?"**
Strong: minimise the worst expected cost over a set of distributions rather than over a set of
parameter values. It nests stochastic programming when the set is a singleton and robust
optimisation when the set is everything on the support; the ambiguity set is the modelling
choice, and its radius is the dial. Weak: "robust with probabilities", without naming the set or
the worst-case distribution.

**9. "How would you decide whether to build the full stochastic model?"**
Strong: compute the VSS on a small designed scenario set first. If the deterministic and
stochastic designs agree, the full model will not change the decision and the designed set was
enough. If they disagree, or the answer oscillates, escalate — expensive irreversible first
stage, material uncertainty relative to the margin, a defensible probability model, real
recourse. Weak: "if we have the data", which is a condition and not a decision rule.

**10. "How do you model a facility that is unavailable for eight weeks?"**
Strong: as a scenario in which its capacity is zero for the duration, with the network rerouting
through recourse — not as a cost increase, which leaves the capacity in place. In the robust
version, as a cardinality-constrained set: at most `k` facilities unavailable at once, with
durations drawn from an interval. Weak: adding a percentage to the freight rate and calling it a
disruption scenario.

**11. "What is the price of robustness, and what must accompany it?"**
Strong: the increase in objective from protecting against uncertainty, against a stated baseline
— preferably the stochastic optimum rather than the unachievable nominal one — at a stated
protection level, with units and a percentage. Weak: a percentage with no Gamma, no baseline, and
no statement of what protection was bought.

## 15. Key terms

- **Uncertainty representation** — the formal object standing in for the unknown: scenarios, a
  distribution, or a set.
- **Expected-value problem (EV)** — the deterministic model solved at mean parameters.
- **EEV** — the expected cost of applying the EV-optimal design across the scenarios.
- **Recourse problem (RP)** — first-stage cost plus expected optimal recourse.
- **Wait-and-see solution (WS)** — the expected value of the per-scenario optima.
- **Here-and-now** — a decision fixed before the uncertainty resolves; **wait-and-see** — one
  taken after it resolves.
- **First-stage variable** — a variable that cannot depend on the scenario; **second-stage
  variable** — one indexed by scenario and chosen after resolution.
- **Recourse** — the second-stage response to a realised scenario; **fixed recourse** — a recourse matrix independent of the scenario.
- **Relatively complete recourse** — a feasible recourse exists for every feasible first-stage
  decision.
- **Non-anticipativity** — the constraint that first-stage decisions cannot anticipate the
  scenario.
- **Scenario** — one complete, internally consistent realisation of the uncertain parameters;
  **scenario tree** — a rooted tree of decision nodes and outcome branches.
- **Deterministic equivalent** — the single programme with one second-stage copy per scenario.
- **Scenario reduction** — replacing a large scenario set with a smaller one close in a
  probability metric.
- **Sample average approximation (SAA)** — estimating the expected objective by averaging over a
  Monte Carlo sample; **out-of-sample validation** tests the design on scenarios not used to
  compute it.
- **VSS** — value of the stochastic solution, `EEV − RP`.
- **EVPI** — expected value of perfect information, `RP − WS`.
- **Uncertainty set** — the realisations a robust model must survive: **box** (independent intervals, worst case at a corner), **budget** (total deviation bounded by Gamma, linear), **ellipsoidal** (quadratic norm bound, second-order cone).
- **Robust counterpart** — the reformulation enforcing feasibility over the set.
- **Price of robustness** — increase in objective caused by protection, relative to a stated
  baseline.
- **Minimax cost** — minimise the worst-case cost; **minimax regret** — minimise the worst-case
  shortfall against the per-scenario best.
- **Distributionally robust optimisation (DRO)** — worst expected cost over an ambiguity set.
- **Ambiguity set** — the set of distributions a DRO model treats as possible.
- **Conditional value at risk (CVaR)** — expected cost in the worst tail at a stated confidence
  level.
- **Disruption scenario** — a scenario in which a facility, lane, or supplier is unavailable.
- **First-stage stability** — whether the recommended design survives changes in sample,
  reduction, or protection level.

## 16. Sources

Grading: A peer-reviewed or audited; B regulatory, official, or standard-setting; C company
case, consultancy, preprint, or institutional analysis; D vendor; E press or anecdote.

**Foundations of stochastic programming**

- Dantzig, G. B. (1955). "Linear Programming under Uncertainty." *Management Science*, 1(3–4),
  197–206. DOI: 10.1287/mnsc.1.3-4.197. The founding paper of two-stage programming with recourse. **A**
- Beale, E. M. L. (1955). "On Minimizing a Convex Function Subject to Linear Inequalities."
  *Journal of the Royal Statistical Society B*, 17(2), 173–184. Independent co-origin. **A**
- Birge, J. R. (1982). "The Value of the Stochastic Solution in Stochastic Linear Programs with
  Fixed Recourse." *Mathematical Programming*, 24, 314–325. DOI: 10.1007/BF01585113. **A**
- Birge, J. R., and Louveaux, F. (2011). *Introduction to Stochastic Programming*, 2nd edition.
  Springer. The standard textbook for formulation, algorithms, and diagnostics. **A**
- Kleywegt, A. J., Shapiro, A., and Homem-de-Mello, T. (2002). "The Sample Average Approximation
  Method for Stochastic Discrete Optimization." *SIAM Journal on Optimization*, 12(2), 479–502. **A**
- Dupačová, J., Gröwe-Kuska, N., and Römisch, W. (2003). "Scenario Reduction in Stochastic
  Programming: An Approach Using Probability Metrics." *Mathematical Programming A*, 95(3),
  493–511. DOI: 10.1007/s10107-002-0331-0. Reduction and its stability bounds. **A**

**Robust and distributionally robust optimisation**

- Soyster, A. L. (1973). "Convex Programming with Set-Inclusive Constraints and Applications to
  Inexact Linear Programming." *Operations Research*, 21(5), 1154–1157. The box formulation. **A**
- Bertsimas, D., and Sim, M. (2004). "The Price of Robustness." *Operations Research*, 52(1),
  35–53. DOI: 10.1287/opre.1030.0065. The budget set and the linear counterpart. **A**
- Delage, E., and Ye, Y. (2010). "Distributionally Robust Optimization Under Moment Uncertainty
  with Application to Data-Driven Problems." *Operations Research*, 58(3), 595–612.
  DOI: 10.1287/opre.1090.0741. **A**

**Network design under uncertainty**

- Snyder, L. V. (2006). "Facility Location Under Uncertainty: A Review." *IIE Transactions*,
  38(7), 537–554. States the first-stage location, second-stage assignment split. **A**
- Snyder, L. V., and Daskin, M. S. (2005). "Reliability Models for Facility Location: The
  Expected Failure Cost Case." *Transportation Science*, 39(3), 400–416.
  DOI: 10.1287/trsc.1040.0107. **A**
- Snyder, L. V., and Daskin, M. S. (2006). "Stochastic p-Robust Location Problems." *IIE
  Transactions*, 38(11), 971–985. **A**
- Klibi, W., Martel, A., and Guitouni, A. (2010). "The Design of Robust Value-Creating Supply
  Chain Networks: A Critical Review." *European Journal of Operational Research*, 203(2),
  283–293. DOI: 10.1016/j.ejor.2009.06.011. **A**
- Govindan, K., Fattahi, M., and Keyvanshokooh, E. (2017). "Supply Chain Network Design under
  Uncertainty: A Comprehensive Review and Future Research Directions." *European Journal of
  Operational Research*, 263(1), 108–141. DOI: 10.1016/j.ejor.2017.04.009. **A**
- Ajide, O. (2026). "Two-Stage Stochastic Optimization for Capacitated Facility Location Under
  Demand Uncertainty." Optimization Online preprint, September 2026. Not peer-reviewed; cited for
  the direction of the finding, not its magnitudes. **C**

**Standards and official guidance**

- ISO 31000:2018, *Risk management — Guidelines*; ISO/IEC 31010:2019, *Risk assessment
  techniques*; ISO 22301:2019, *Business continuity management systems*; ISO 28000:2022,
  *Security and resilience — Security management systems for the supply chain*. **B**
- ISO 14083:2023, *Greenhouse gases — Quantification and reporting of greenhouse gas emissions
  arising from transport chain operations*. **B**
- IFRS 16 *Leases*; ASC 842 *Leases*; US Federal Reserve SR 11-7, *Guidance on Model Risk
  Management*. **B**

> **Note on citation discipline.** Vendor suites that present stochastic and robust optimisation
> as product modules make a capability claim, not an evidential one (grade D). No citation here
> is offered as evidence that a method produces a particular saving in a particular industry. The VSS and EVPI figures in section 6
> are computed from an illustrative instance. Do not transfer a percentage from any source
> without rebuilding the model.

## 17. Links

**This module**

- Module map: [`_index.md`](_index.md)
- The diagnostic this topic formalises: [`07_scenario-sensitivity.md`](07_scenario-sensitivity.md)
- The baseline model whose first stage this topic reuses: [`04_facility-location.md`](04_facility-location.md)
- Multi-period structure and the depth of a tree: [`05_multi-commodity-period.md`](05_multi-commodity-period.md)
- MILP anatomy and formulation discipline: [`03_milp-anatomy.md`](03_milp-anatomy.md)
- Transition and phasing, which this topic does not model: [`06_greenfield-brownfield.md`](06_greenfield-brownfield.md)
- The graph and cost-to-serve basis: [`02_graph-cost-to-serve.md`](02_graph-cost-to-serve.md)
- The decision ladder this analysis serves: [`01_network-decision-ladder.md`](01_network-decision-ladder.md)
- Next, carbon as part of the objective: [`09_carbon-objective.md`](09_carbon-objective.md)
- Later, decomposition and solver behaviour for large stochastic programmes: [`10_solvers-heuristics.md`](10_solvers-heuristics.md)

**Adjacent modules**

- Probabilistic forecasting, which supplies the demand distributions: [`../08_hierarchical_intermittent_probabilistic_forecasting/_index.md`](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md)
- Simulation, the natural out-of-sample validation: [`../20_digital_twins_simulation_optimization_uncertainty/_index.md`](../20_digital_twins_simulation_optimization_uncertainty/_index.md)
- Disruption and resilience in depth: [`../14A_resilience_disruption_analytics/_index.md`](../14A_resilience_disruption_analytics/_index.md)

**Programme navigation**

- Part II methods map: [`../_index.md`](../_index.md)
- Part I foundations: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)

> **Note:** all ten topic files in this module are now written. Every markdown link on this page
> resolves to a file that exists on disk.
