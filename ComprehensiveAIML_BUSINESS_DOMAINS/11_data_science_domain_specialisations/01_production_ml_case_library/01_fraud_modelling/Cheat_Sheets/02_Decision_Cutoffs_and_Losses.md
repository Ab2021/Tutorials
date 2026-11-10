---
collection: "fraud-modelling"
file_id: "02"
topic: "Decision cut-offs and losses"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
revised: "2026-11-10"
reviewed: null
review_trigger: "Change to action set, loss schedule, review capacity, calibration, or maturity definition"
notation_style: "plain-text"
prerequisites: ["01_Fraud_Taxonomy_and_Decision_Grain.md"]
pattern_tags: [threshold-policy, expected-loss, capacity, calibration]
source_status: "OPEN"
owner_role: "Fraud curriculum maintainer"
source_index: "INDEX.md"
---

# Decision Cut-offs and Losses

> **Purpose:** Turn a calibrated risk estimate into an action (allow, challenge, block, refer, abstain) using an explicit loss schedule, a capacity limit, and a stated population, and know when the resulting cut-off stops being valid.
> **Prerequisites:** [Fraud taxonomy and decision grain](01_Fraud_Taxonomy_and_Decision_Grain.md), conditional probability, and a named decision owner who can commit an action set.
> **Key outcome:** Given scores, three action costs, and a review budget, compute the two boundaries that partition the risk axis, the expected cost of the resulting policy, the shadow price of one more review slot, and the sensitivity of both boundaries to each input.
> **This file does NOT cover:** feature construction, label adjudication design, legal approval thresholds, or any measured deployment result.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 2.0 / not reviewed.
> **Evidence:** DERIVED loss arithmetic with ILLUSTRATIVE inputs; every currency figure below is constructed for teaching and is not a market benchmark.
> **Retrieval cue:** If anyone proposes "use 0.5" or "use the top 1 percent" without naming the action, the loss schedule, and the capacity, start at Section 2.

## 1. File Header

**The problem this sheet solves.** A fraud model produces a probability. A business needs an action. Between them sit four objects that are routinely left undefined: the action set with its costs, the loss schedule with its units, the capacity that limits how many items can be handled, and the population on which the probability was calibrated. When any one of the four is missing, the chosen cut-off is arbitrary, and arbitrary cut-offs are the reason two teams with identical models report opposite conclusions and both are honest.

**What a cut-off actually is.** It is a rule that maps an estimated risk and a set of row-level attributes to one action. It is not a property of the model, not a universal constant, and not comparable across populations. Two models with the same AUC can have different optimal cut-offs, because the cut-off depends on the calibration of the score while AUC does not. Two models with the same calibration can have different optimal cut-offs, because the action costs differ by channel. The only invariant statement is conditional: given this population, this schedule, this capacity, and this calibration, the boundary is this number.

**Why "probability above 0.5" is worse than arbitrary.** A 0.5 boundary corresponds to a loss schedule in which the fraud loss equals the friction cost exactly. In every retail fraud context the fraud loss is one to three orders of magnitude larger than the friction, so 0.5 is not merely inexact, it is roughly thirty times too high for the portfolio in Section 5. The rule survives because it is easy to explain, and it costs real money every month it is used.

**The three clocks that make cut-offs age.** A cut-off is invalidated by a change in the loss schedule (a new chargeback rule), a change in the population (a new market or channel), or a change in calibration (a model refresh or a rebalanced training set). All three happen without any code change to the serving path, which is why cut-offs need their own version, owner, and change log rather than living inside a model binary.

**Scope of the arithmetic.** Every quantity below is either a definition or an exact algebraic consequence of stated assumptions. The currency values, counts, and recovery rates are constructed examples marked ILLUSTRATIVE. The formulas will hold under substitution; the numbers will not, and should not be quoted as industry values.

**How to read a derivation block.** Each block has five parts. FORMULA states the relation. WHERE defines each symbol with its units and range. CONSTRAINT lists the assumptions that make the algebra valid. NUMERICAL EXAMPLE substitutes numbers and shows intermediate values. INTERPRETATION states the decision consequence. Each block ends with a validity condition, which is the clause a reviewer should attack first, because a formula is rarely wrong and its assumptions frequently are.

**Notation.** Costs and losses are in US dollars, per row unless stated. `p` is a calibrated probability in [0,1]. Friction is written `f` with an action subscript. Residual loss after a partial intervention is written `a_c` for challenge. Capacity is `K` in completed items per period, and review effort is `h` in hours. Shadow prices are `lambda` in dollars per hour. The odds form `p/(1-p)` appears throughout because the derivations are linear in odds.

**The one test that catches most errors.** For any proposed cut-off, compute the expected cost of the policy at that cut-off and compare it with the expected cost of the two neighbouring policies (one action cheaper, one action harsher). If the chosen policy is not the cheapest of the three on the actual population, the cut-off is wrong. This test needs no theory and no tooling, and it fails often enough that it is worth running on every threshold change review.

**A warning about single-number summaries.** No single number replaces a cut-off review. AUC is prevalence-blind, KS is cost-blind, precision at a fixed volume is calibration-blind, and expected loss at a fixed schedule is population-blind. The remainder of this sheet is arranged so that each of those blind spots is covered explicitly by a section rather than assumed away.

**Who owns the boundary.** Three roles must agree before a boundary goes live, and each owns a different input. The fraud analytics owner owns the model and the calibration. The operations owner owns the action set, the friction estimates, and the capacity. The finance owner owns the loss schedule and the attrition assumption. A boundary issued by only one of the three is a guess with a decimal point, and the characteristic failure is a boundary that is arithmetically sound, operationally infeasible, and financially unrecognised — all three at once, because nobody was required to sign the inputs they did not own.

**The review question that matters.** When a boundary is proposed, the useful question is not "is this the right number" but "which of the four inputs would have to be wrong, and by how much, for this boundary to be wrong in a way that matters". A boundary is robust when the answer is "a factor of five in any single input", and fragile when the answer is "a 10 percent change in attrition". Robustness is a property that can be computed before deployment and is almost never computed.

**What happens after deployment.** A live boundary generates three streams that must be joined: the decision record (score, boundary, schedule version, action), the outcome record (mature label, amount, recovery), and the cost record (support contact, reissue, complaint). Most teams have the first two and approximate the third. The consequence is that the realised cost can be computed but the realised friction cannot, so the schedule never gets validated against reality and drifts until someone re-derives it from scratch.

**Two audit questions with different answers.** "Was the decision consistent with the policy in force?" is answerable from the decision record alone and is a compliance question. "Was the policy the right one?" is answerable only from outcome and cost data and is an economics question. Conflating them produces the situation where a review certifies that everything worked as designed while the design itself was unprofitable — a common and expensive confusion in regulated fraud functions.

**Reading order for a practitioner under time pressure.** Sections 4.1 and 4.2 give the two formulas that cover ninety percent of real decisions. Section 5 shows them applied with all intermediate arithmetic. Section 8's first six entries cover the failures that actually occur. Section 12 is the card to keep. The remaining material exists so that a reviewer can find the assumption they want to challenge, and so that a practitioner facing an unusual schedule has the variant already derived rather than having to redo it under deadline.

## 2. Concept Map / Mental Model

The pipeline has five stages and two feedback loops. Reading the diagram left to right gives the forward computation; the loops are the reasons a static cut-off decays.

```text
[features at t_d] -> [model score s] -> [calibration map] -> [risk p]
                                                                |
                                                                v
                        [loss schedule: a, a_c, f_c, f_b]  ->  [policy]
                                                                |
                              +---------------------------------+
                              v                                 v
                    [immediate action]              [capacity-limited queue]
                              |                                 |
                              v                                 v
                    [observed outcomes]             [mature dispositions]
                              |                                 |
                              +---------> [refit schedule and calibration]
```

**The forward path in words.** Features available at the decision deadline produce a score. The score is mapped to a probability by a calibration fitted on a mature population. The probability meets a loss schedule to produce an action, and the action set is restricted by capacity for the actions that consume human time. Nothing in that chain is a modelling step except the score itself; the rest is arithmetic and governance.

**First feedback loop: the schedule.** Observed outcomes change the realised loss per fraud, the realised friction per false block, and the recovery rate of the challenge. A schedule copied from a two-year-old analysis is a schedule for a different business. Section 8.3 lists the checks that detect a stale schedule before it silently moves the operating point.

**Second feedback loop: the calibration.** The score distribution drifts, and even a drift-free score re-calibrates when prevalence moves. A boundary kept fixed in probability space is stable by construction and volatile in volume; a boundary kept fixed in quantile space is stable in volume and volatile in probability. Choosing which to fix is a business decision about which variable the operation can absorb, and it must be recorded, because the two choices produce different losses under the same shift.

**Why the loss schedule has four numbers, not two.** Allow has one cost (the fraud loss `a`). Challenge has two (`a_c`, the residual loss when the fraud succeeds anyway, and `f_c`, the friction when the customer is legitimate). Block has one (`f_b`, the friction plus any attrition). That is four numbers for three actions, and each one is measurable in principle. Teams that collapse them into "cost of a false positive" and "cost of a false negative" lose the ability to distinguish a challenge from a block, which is exactly the distinction that makes step-up authentication worth deploying.

**The capacity gate is not a cost.** Capacity differs from a cost in kind. A cost can be exceeded at a price; a capacity cannot be exceeded at any price within the period. That is why capacity is handled with a constraint and a shadow price rather than by adding a large penalty to the loss schedule. Section 4.3 derives the shadow price, and Section 9 shows how it converts into a staffing decision.

**Two ways the same policy can be expressed, with different consequences.** A probability boundary says "act when risk exceeds 1.41 percent". A volume boundary says "act on the top 2.9 percent of rows". On a stable population they coincide. On a shifting population the first keeps the loss ratio constant and lets volume move, while the second keeps workload constant and lets the loss ratio move. Neither is universally right; the choice must match whether operations or finance is the binding constraint.

**What the model contributes and what it cannot.** The model contributes the ordering and the calibration. The ordering determines how much loss is avoidable at any given volume. The calibration determines where the boundary sits in probability space. A model with excellent ordering and poor calibration still supports an optimal policy if the boundary is fitted on the same population, which is why "we only use the ranking" is a legitimate position — as long as the ranking is refitted when the population shifts, and the refit is done on outcomes rather than on score quantiles.

**The complete decision statement.** A defensible operating point reads as one sentence: "On population P, with schedule S at version v, and capacity K, we allow below 1.41 percent, challenge between 1.41 and 7.22 percent, and block above 7.22 percent; expected cost is C per month, the shadow price of a review slot is lambda, and the policy is invalidated by a schedule change or a population change." Every clause in that sentence is checkable, which is the entire point.

**Where the loss actually originates.** Trace a single fraudulent authorisation through a policy and the cost decomposition explains why the middle band dominates the economics. A fraud in the block band costs nothing because it is stopped. A fraud in the allow band costs the full `a`. A fraud in the challenge band costs `a_c`, which is `a` minus the recovery. So the policy's performance is the sum of two quantities: how many frauds fall in the allow band, and how much residual loss the challenge band leaves. The block band contributes only friction. Optimising "recall" therefore optimises something the cost function does not directly contain, because a fraud moved from allow to challenge barely changes recall and changes cost by 126 dollars.

**Why the schedule is a policy, not a measurement.** Two teams with identical models and identical data can legitimately disagree about the boundary if they hold different views about what a wrong block costs. That is not a data dispute, it is a value dispute, and it belongs to the business owner rather than to the analytics team. The analytics contribution is to make the disagreement explicit and computable: "you are choosing between 121,800 dollars of block friction and 198,000 dollars of fraud loss, and here is the exchange rate implied by your boundary".

**The exchange rate reading of a boundary.** Any boundary implies a rate at which the business trades friction for loss. At `p* = 0.032258`, the implied exchange is one dollar of friction accepted per 30 dollars of expected loss avoided. Stating a boundary in those terms is far more useful in a governance discussion than stating the probability, because the probability is not comparable across portfolios while the exchange rate is.

**What changes when the action is irreversible.** An irreversible action carries a third cost that does not appear in the schedule: the cost of the complaint or remediation when the action is wrong and the customer complains. That cost is not `f_b` again; it is a separate term that applies only to the irreversible action and only when the customer contests. Folding it in raises the block boundary specifically, which is why institutions with high complaint-handling costs run visibly higher block boundaries than their friction numbers alone would suggest.

**Why capacity must be stated in completed items.** A queue can accept unlimited items and complete a limited number. If capacity is stated as arrivals, every capacity calculation silently assumes the backlog is zero, and the assumption is false exactly when the queue matters. The distinction is not pedantic: an alert queue with 6,000 arrivals and 4,000 completions has a capacity of 4,000 and a growth rate of 2,000 per month, and a boundary tuned to arrivals will produce an unbounded backlog that the boundary calculation never noticed.

**The stability test worth running before deployment.** Perturb each schedule input by a factor of two in turn, recompute the boundary, and record the range. A boundary that moves by less than 20 percent under all four perturbations is robust and can be deployed with a stated tolerance. A boundary that moves by a factor of three under one perturbation is not a decision, it is an artefact of that input, and deploying it transfers an unexamined assumption into the production path. This test costs twenty minutes and is the highest-value activity in the entire threshold-setting process.

## 3. Foundational Definitions

### 3a Probability and score objects

| Term | Meaning | Unit / type | Boundary (it is not) |
|---|---|---|---|
| Score `s` | Model output before calibration, any monotone scale | real | Not a probability |
| Risk `p` | Calibrated probability of the declared outcome | [0,1] | Not the action-conditional risk |
| Outcome | The declared label: confirmed fraud on this row | boolean | Not a dispute, not a suspicion |
| Calibration map | Monotone function from `s` to `p` fitted on mature data | function | Not a decision rule |
| Base rate `pi` | Prevalence of the outcome on the scored population | [0,1] | Not the reviewed-row rate |
| Discrimination | Ability to order rows by outcome; measured by AUC or KS | [0,1] | Not calibration |
| Resolution | Ability to separate rows into fine risk bands | count of bands | Not discrimination |
| Operating point | The tuple (action set, boundaries, population, schedule) | declaration | Not a model parameter |

### 3b Loss schedule objects

| Term | Meaning | Unit / type | Boundary (it is not) |
|---|---|---|---|
| `a` | Mean loss when a fraudulent row is allowed to complete | currency/row | Not the transaction amount |
| `a_c` | Residual loss when a challenge fails to stop a fraud | currency/row | Not zero, not `a` |
| `f_c` | Friction when a legitimate row is challenged | currency/row | Not the cost of the challenge technology |
| `f_b` | Friction when a legitimate row is blocked | currency/row | Not only the support call |
| Attrition `q` | Share of wrongly blocked customers who leave | [0,1] | Not churn in general |
| `L` | Contribution margin lost per departed customer | currency/customer | Not revenue |
| Recovery | Share of challenged fraud that is prevented | [0,1] | Not the share of disputes won |
| Direct cost | Cash spent per action, such as SMS or analyst minutes | currency/row | Not the loss avoided |
| Opportunity cost | Value forgone because an action was taken | currency/row | Not the direct cost |

### 3c Capacity and effort objects

| Term | Meaning | Unit / type | Boundary (it is not) |
|---|---|---|---|
| `K` | Completed items per period | items/period | Not items queued |
| `h_i` | Effort to complete item `i` | hours/item | Not arrival spacing |
| `H` | Available effort in the period | hours | Not headcount |
| Shadow price `lambda` | Marginal value of one more unit of capacity | currency/hour | Not an average cost |
| Backlog | Items awaiting completion at period end | items | Not the queue length at an instant |
| Utilisation | Completed effort divided by available effort | [0,1] | Not service level |
| Service level | Share of items completed within a stated time | [0,1] | Not precision |

### 3d Population and evaluation objects

| Term | Meaning | Unit / type | Boundary (it is not) |
|---|---|---|---|
| Scored population | Rows the model actually evaluated | row set | Not all rows in the business |
| Eligible population | Rows on which the action set is permitted | row set | Not the scored set |
| Comparison cohort | Rows held under the previous policy | row set | Not a random sample unless randomised |
| Guardrail | A secondary metric that must not degrade beyond a limit | constraint | Not an objective |
| Expected cost | Schedule-weighted cost of a policy on a population | currency/period | Not realised cost |
| Realised cost | Cost that actually occurred | currency/period | Not comparable across policies without a holdout |

### 3e The four objects a cut-off cannot exist without

The definitions above can be compressed into a four-part test that takes thirty seconds and prevents most threshold arguments. A cut-off is only meaningful when all four are present.

First, an **action set**: the finite list of things the system may do. If "flag for review" and "block" are used interchangeably in conversation, the cut-off is undefined because the friction and the loss differ between them.

Second, a **loss schedule**: one number per action-outcome pair, in one currency, for one period. If the schedule mixes per-event and per-customer costs, the comparison is dimensionally invalid and no threshold can be optimal for it.

Third, a **capacity**: the number of items per period that can be handled for each human-dependent action. Without it the optimum is unbounded in the direction of more review, which is never achievable and is therefore not a recommendation.

Fourth, a **calibration population**: the exact row set on which the probabilities were fitted, with its window and grain. Without it the boundary cannot be transferred to production, and the transfer is the step where most operating-point failures originate.

The order matters. Action set first, because it determines which costs are needed. Then schedule, then capacity, then calibration. Teams that start with calibration and work backwards almost always end up with a boundary that is mathematically defensible and operationally meaningless.

### 3f Distinctions that change the number

*Expected versus realised.* Expected cost is computed from probabilities and a schedule; realised cost is measured. Comparing the realised cost of two policies requires a holdout, because the policies see different populations after deployment. Reporting an expected-cost improvement as a realised one is the most common overstatement in fraud business cases.

*Friction per event versus per customer.* A policy that blocks 3 percent of events on 16 events per customer contacts roughly 39 percent of customers per month at least once if the events are independent within customer, but far fewer if fraud and false positives cluster on the same customers. The two interpretations differ by an order of magnitude in customer-impact reporting, and clustering is the norm rather than the exception.

*Loss avoided versus loss displaced.* An avoided loss is one that never occurs. A displaced loss is one that occurs elsewhere, often on a channel with weaker controls. Only a holdout distinguishes them, and the distinction changes the sign of the business case when the displacement share is high.

*Threshold versus operating point.* The threshold is one number. The operating point is the whole tuple including population and schedule. Changing the population without changing the threshold is still a policy change, and it belongs in the change log even though no parameter moved.

*Fraud loss versus fraud volume.* A policy can reduce the count of successful frauds and increase the total loss, if the blocked events were small and the displaced events were large. Optimising on counts and reporting on currency produces exactly this mismatch, and it is detectable by recomputing the policy on amount-weighted cost.

*Recovery versus prevention.* A blocked fraud is prevented; a challenged fraud that is then abandoned is also prevented, but a challenged fraud that completes after verification is neither prevented nor recovered. The share of challenged fraud that completes is the complement of the recovery rate, and it is the number that determines whether the challenge is worth its friction. In the worked example a 30 percent completion share means `a_c = 0.30a`, which is the entire justification for the 5.81-point band.

*Direct friction versus total friction.* Direct friction is the cash cost of the action: an SMS, a reissue, four minutes of support. Total friction includes the customer's time valued at their own rate, the chance of abandonment, and the chance of departure. Only direct friction is measurable from the general ledger; the rest requires either a survey or a holdout. Reporting only direct friction is not dishonest, but it is incomplete in a direction that always makes the policy look cheaper than it is.

*Calibration population versus deployment population.* A calibration fitted on all cardholders applied to a newly onboarded cohort with three weeks of history is an extrapolation, because the model's feature distribution on that cohort lies outside the fitting range. The extrapolation is not visible in the calibration plot because the plot covers only the fitting population. This is the mechanism behind the recurring observation that new customers are over- or under-scored for their first quarter.

*Threshold stability versus sensitivity.* A boundary can be stable in one input and highly sensitive in another, and the two properties are independent. Stability in `a` follows from `a >> f_b`, which makes the ratio insensitive to `a`. Sensitivity to `q` follows from `qL` competing with `f_direct` at the same order of magnitude. So a boundary is typically robust to the largest number in the schedule and fragile to the smallest, which is the opposite of where review effort usually goes.

*Expected cost versus budget.* Expected cost is a probability-weighted number and can be planned for. A budget is a hard limit in a period, and the probability that realised cost exceeds the budget is a different question from its expectation. A policy with a lower expectation and a heavier tail can breach the budget more often than a policy with a higher expectation and a thin tail. Risk-averse selection therefore requires the distribution, not just the mean, which is the practical content of the uncertainty variant in Section 4.3.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: single-action break-even

The simplest case has two actions and one fraud outcome. It is the only case many practitioners ever need, and it already contains the entire logic of the sheet.

```text
FORMULA
  C_allow(p)   = p * a
  C_block(p)   = (1 - p) * f_b
  block  <=>  C_block < C_allow
  p*  =  f_b / (f_b + a)

WHERE
  p      calibrated fraud probability for this row            [0,1]
  a      mean loss if allowed and fraudulent                  currency/row
  f_b    friction if blocked and legitimate                   currency/row
  p*     break-even probability                               [0,1]

CONSTRAINT
  linear costs; one loss owner; no capacity limit; f_b independent of p.

NUMERICAL EXAMPLE
  a = 180.00, f_b = 6.00
  p* = 6.00 / (6.00 + 180.00) = 6.00 / 186.00 = 0.032258

INTERPRETATION
  Blocking is the cheaper action above 3.23 percent risk. At 0.09 percent
  population prevalence almost no row reaches that level, which is why a
  well-calibrated card model blocks a small share of authorisations and why
  a 0.5 boundary would block almost nothing at all.

VALIDITY CONDITION
  Fails when a or f_b vary across rows by more than roughly a factor of three,
  because then one boundary cannot represent the schedule.
```

Two structural facts follow immediately and are worth stating separately, because they are what reviewers actually need. First, the boundary is a ratio: it depends only on the relative size of the two costs. Doubling both leaves it unchanged. Second, the boundary in probability space is invariant to prevalence, while the volume above it is not. That asymmetry is the reason a fixed probability boundary is a stable policy and a fixed quantile boundary is a stable workload.

The odds form makes the invariance obvious and is the form used for the rest of the derivations.

```text
FORMULA
  C_block < C_allow  <=>  p/(1-p) < f_b/a
  odds* = f_b / a

WHERE
  odds*   break-even odds, dimensionless                       >0
  p/(1-p) odds of fraud for this row                           >0

CONSTRAINT
  same as the probability form; algebraically identical.

NUMERICAL EXAMPLE
  f_b = 6.00, a = 180.00 -> odds* = 0.033333
  check: p* = odds*/(1+odds*) = 0.033333/1.033333 = 0.032258   (agrees)

INTERPRETATION
  The boundary is the cost ratio expressed as odds. A row is blocked when its
  odds of fraud exceed one part in thirty, which is a sentence a reviewer can
  check without doing the division.

VALIDITY CONDITION
  Odds form overflows in floating point above about p = 0.9999; use the
  probability form in code near the extremes.
```

**Attrition belongs inside the friction.** The direct friction of a block is a support call, a reissued credential, and a few minutes of customer annoyance. The real cost also includes the chance the customer leaves. Folding attrition into `f_b` is the mechanism by which a technically small friction becomes a large one.

```text
FORMULA
  f_b_eff = f_direct + q * L
  p*_eff  = f_b_eff / (f_b_eff + a)

WHERE
  f_direct  cash and direct cost of one block        currency/row
  q         share of wrongly blocked who leave       [0,1]
  L         margin lost per departure                currency/customer
  f_b_eff   effective friction                       currency/row

CONSTRAINT
  departure is attributed to this block only; no offsetting retention action.

NUMERICAL EXAMPLE
  f_direct = 6.00, q = 0.005, L = 400.00
  f_b_eff = 6.00 + 0.005*400.00 = 6.00 + 2.00 = 8.00
  p*_eff = 8.00 / 188.00 = 0.042553
  relative shift = 0.042553 / 0.032258 = 1.319149

INTERPRETATION
  A half-percent attrition rate raises the block boundary by 32 percent,
  from 3.23 to 4.26 percent. This is the single largest unmeasured term in
  most threshold analyses and it always pushes in the same direction.

VALIDITY CONDITION
  Fails when departures are partly caused by fraud losses rather than blocks,
  which makes the attribution non-causal and the estimate an upper bound.
```

### 4.2 General case: three actions and two boundaries

With allow, challenge, and block available, the optimal policy is a partition of the odds axis into three intervals rather than a single cut. Both boundaries are ratios of cost differences, derived by pairwise dominance.

**Step one: allow versus challenge.** A challenge is worth inserting when the expected benefit of reducing the loss exceeds the friction of annoying a legitimate row.

```text
FORMULA
  C_allow(p)     = p * a
  C_challenge(p) = p * a_c + (1 - p) * f_c
  C_challenge < C_allow
    <=> p * (a - a_c) > (1 - p) * f_c
    <=> p / (1 - p) > f_c / (a - a_c)
    <=> p > f_c / (a - a_c + f_c)  =  tau_low

WHERE
  a_c     residual loss when the challenge still fails on fraud   currency/row
  f_c     friction when a legitimate row is challenged            currency/row
  tau_low lower boundary of the challenge band                    [0,1]

CONSTRAINT
  a_c < a; the challenge costs the same whether the row is fraud or not;
  a challenged fraud that is stopped costs a_c rather than a.

NUMERICAL EXAMPLE
  a = 180.00, a_c = 54.00, f_c = 1.80
  a - a_c = 126.00
  tau_low = 1.80 / (126.00 + 1.80) = 1.80 / 127.80 = 0.014085

INTERPRETATION
  Above 1.41 percent risk the challenge is cheaper than allowing. Note how far
  below the block boundary this sits: a challenge is justified at less than
  half the risk that justifies a block, because its friction is a third.

VALIDITY CONDITION
  Fails when the challenge has a completion cost of its own, for example when
  a failed challenge triggers a support call that is as expensive as a block.
```

**Step two: challenge versus block.** Above some risk, the block's higher friction is worth paying because it prevents the residual loss outright.

```text
FORMULA
  C_block(p) = (1 - p) * f_b
  C_block < C_challenge
    <=> (1 - p) * f_b < p * a_c + (1 - p) * f_c
    <=> (1 - p) * (f_b - f_c) < p * a_c
    <=> p / (1 - p) > (f_b - f_c) / a_c
    <=> p > (f_b - f_c) / (a_c + f_b - f_c)  =  tau_high

WHERE
  f_b       friction of a block on a legitimate row    currency/row
  tau_high  upper boundary of the challenge band        [0,1]

CONSTRAINT
  f_b > f_c, i.e. blocking costs more friction than challenging.

NUMERICAL EXAMPLE
  f_b = 6.00, f_c = 1.80, a_c = 54.00
  numerator = 4.20; denominator = 54.00 + 4.20 = 58.20
  tau_high = 4.20 / 58.20 = 0.072165
  band width = 0.072165 - 0.014085 = 0.058080

INTERPRETATION
  The block boundary at 7.22 percent is 2.24 times the single-action boundary
  of 3.23 percent, because the block now competes with a cheaper action that
  already removes 70 percent of the loss. The band is 5.81 points wide, so on
  a well-spread score it carries far more rows than either tail.

VALIDITY CONDITION
  Fails when the challenge is unresolved at decision time, because an
  unresolved challenge behaves like an allow with extra cost.
```

**Step three: the ordering constraint.** The two boundaries are only meaningful when `tau_low < tau_high`. That inequality fails in one important regime, and the failure is diagnostic rather than pathological.

```text
FORMULA
  tau_low < tau_high
  <=> f_c * (a_c + f_b - f_c) < (f_b - f_c) * (a - a_c + f_c)

WHERE
  all symbols as above

CONSTRAINT
  expansion is exact; no approximation used.

NUMERICAL EXAMPLE
  lhs = 1.80 * 58.20 = 104.76
  rhs = 4.20 * 127.80 = 536.76
  104.76 < 536.76  ->  ordering holds, band exists

INTERPRETATION
  The gap of 432.00 is the total value the challenge band creates per unit of
  risk at the margin. When the gap approaches zero the band collapses and the
  three-action policy degenerates to two actions; a negative gap means the
  challenge is dominated and should be removed from the action set rather
  than tuned.

VALIDITY CONDITION
  The inequality is necessary, not sufficient: it assumes the schedule applies
  uniformly across the score range, which Section 7 challenges.
```

### 4.3 Variants: capacity, quantiles, uncertainty, amount dependence

**Variant A: capacity-constrained selection.** When one action consumes a scarce resource, the boundary is not chosen from the schedule alone. It is chosen jointly with an allocation, and the multiplier on the constraint is the price of the resource.

```text
FORMULA
  maximise  sum_i v_i x_i     subject to  sum_i h_i x_i <= H
  v_i = p_i * a - (1 - p_i) * f_c          (net benefit of handling item i)
  Lagrangian condition: take item i iff  v_i / h_i > lambda
  lambda = marginal value of one additional hour

WHERE
  v_i     net benefit of completing item i            currency
  h_i     hours to complete item i                    hours
  H       hours available in the period                hours
  x_i     decision variable in {0,1}                  binary
  lambda  shadow price                                currency/hour

CONSTRAINT
  items independent; hours divisible; no learning between items.

NUMERICAL EXAMPLE
  two item types, H = 1,000 hours
  type X: p = 0.35, a = 1,200.00, f_c = 1.80, h = 0.5
          v = 0.35*1,200 - 0.65*1.80 = 420.00 - 1.17 = 418.83
          v/h = 837.66 per hour
  type Y: p = 0.08, a = 1,200.00, f_c = 1.80, h = 0.5
          v = 96.00 - 1.66 = 94.34
          v/h = 188.69 per hour
  capacity: X exhausts 400 items in 200 hours, leaving 800 hours
            800 / 0.5 = 1,600 items of type Y
  total benefit = 400*418.83 + 1,600*94.34 = 167,532.00 + 150,944.00
                = 318,476.00
  lambda sits between 188.69 and the next item's density (below 188.69)

INTERPRETATION
  A schedule-only rule that reviews everything above 10 percent risk takes all
  400 type-X items and stops, using 200 of 1,000 hours and capturing 167,532.
  The capacity-aware rule captures 318,476, which is 1.90 times as much, using
  the same staff. The improvement comes from filling idle capacity, not from a
  better model.

VALIDITY CONDITION
  Fails when items interact: reviewing one alert on an actor often resolves
  several, which makes the joint value less than the sum and biases the rule
  toward over-selecting clustered items.
```

**Variant B: fixed-quantile policy.** Many operations teams work to a volume target rather than a risk target. The policy is then "act on the top `v` share", and the effective boundary is whatever risk corresponds to that quantile.

```text
FORMULA
  q_v = quantile of the score distribution at volume share v
  effective_risk = calibration(q_v)
  expected loss ratio = a * (1 - F_fraud(q_v)) ... see block for the counting form

  counting form:
    flagged   = N * v
    caught    = pi * N * TPR(q_v)
    precision = caught / flagged
    loss_left = pi * N * a * (1 - TPR(q_v))
    friction  = (1 - pi) * N * FPR(q_v) * f_b

WHERE
  v        volume share acted upon                        [0,1]
  N        rows scored in the period                      count
  TPR, FPR operating-point rates on the population        [0,1]

CONSTRAINT
  TPR and FPR estimated on a population matching the production mix.

NUMERICAL EXAMPLE
  N = 4,000,000, pi = 0.000900, a = 180.00, f_b = 6.00
  v = 0.009 with TPR = 0.62, FPR = 0.005
  flagged   = 4,000,000 * 0.009 = 36,000
  caught    = 0.0009 * 4,000,000 * 0.62 = 2,232
  precision = 2,232 / 36,000 = 0.062000
  loss_left = 3,600 * 180.00 * 0.38 = 246,240.00
  friction  = 3,996,400 * 0.005 * 6.00 = 119,892.00
  total     = 366,132.00

INTERPRETATION
  The quantile policy is expressible as a loss, so it can be compared with any
  other policy on the same population. That is the only defensible way to
  compare a volume rule against a risk rule; comparing 0.9 percent with
  1.41 percent compares a volume with a probability and is meaningless.

VALIDITY CONDITION
  Fails when the score distribution shifts while the volume target is held
  fixed, because the implied risk boundary moves without anyone deciding it.
```

**Variant C: uncertainty in the loss schedule.** When `a` is estimated rather than known, a risk-neutral rule uses its mean and a conservative rule uses a lower bound. The difference between the two boundaries is the price of the ambiguity.

```text
FORMULA
  risk neutral:      p* = f_b / (f_b + E[a])
  conservative:      p* = f_b / (f_b + a_low)     with a_low = E[a] - z * sd
  shift factor:      (f_b + E[a]) / (f_b + a_low)

WHERE
  E[a]    mean loss per fraudulent row        currency/row
  sd      standard deviation of that loss     currency/row
  a_low   lower confidence bound              currency/row
  z       one-sided normal multiplier         dimensionless

CONSTRAINT
  loss is approximately normal on the relevant tail; f_b known precisely.

NUMERICAL EXAMPLE
  E[a] = 180.00, sd = 90.00, z = 1.645, f_b = 6.00
  a_low = 180.00 - 1.645*90.00 = 180.00 - 148.05 = 31.95
  neutral:  6.00 / 186.00 = 0.032258
  conserv.: 6.00 / 37.95 = 0.158102
  shift = 4.90

INTERPRETATION
  A one-sided 95 percent conservative boundary sits 4.9 times higher than the
  risk-neutral one. That is a genuine policy choice, not a statistical nicety:
  the conservative rule blocks far less and accepts more loss, and it should
  be chosen deliberately when the downside of a wrong block is irreversible.

VALIDITY CONDITION
  Fails when the loss distribution is heavy tailed, which it is: use a quantile
  of the empirical loss distribution rather than a normal approximation.
```

**Variant D: amount-dependent boundaries.** If the loss scales with the transaction amount, the boundary scales inversely. The correct rule uses each row's own amount, and a single boundary is the special case where all amounts are equal.

```text
FORMULA
  p*_i = f_b / (f_b + a_i)
  a_i  proportional to amount: a_i = r * A_i
  p*_i = f_b / (f_b + r * A_i)

WHERE
  A_i   amount of row i                          currency
  r     loss rate per unit amount on fraud        [0,1]
  p*_i  row-specific boundary                     [0,1]

CONSTRAINT
  loss is proportional to amount; friction is not.

NUMERICAL EXAMPLE
  f_b = 6.00, r = 0.90
  A = 20.00  -> a = 18.00,  p* = 6/24    = 0.250000
  A = 200.00 -> a = 180.00, p* = 6/186   = 0.032258
  A = 900.00 -> a = 810.00, p* = 6/816   = 0.007353
  ratio small to large = 0.250000 / 0.007353 = 34.0

INTERPRETATION
  A constant boundary at 3.23 percent under-blocks the 20-dollar case by a
  factor of 7.8 and over-blocks the 900-dollar case by a factor of 4.4. Both
  errors are produced by the policy's shape, not by the model.

VALIDITY CONDITION
  Fails when the loss is capped by a liability limit, because above the cap
  the boundary stops falling and becomes flat again.
```

### 4.4 Limits and counterexamples

**Counterexample one: a challenge that costs more than it saves.** Take `a = 180.00`, `f_c = 40.00`, `a_c = 100.00`. Then `tau_low = 40.00 / (80.00 + 40.00) = 0.333333` and `tau_high = (f_b − f_c)/(a_c + f_b − f_c)`, which with `f_b = 6.00` gives `(−34.00)/(100.00 − 34.00) = −0.515152`. A negative upper boundary means the challenge is never preferred to a block, and the ordering constraint from 4.2 fails. The correct conclusion is not "tune the boundaries" but "remove the challenge action", because it is dominated across the entire risk range. Any optimiser that returns boundaries here is solving the wrong problem.

**Counterexample two: aggregation hides a schedule change.** Two segments, each with a different friction. Segment A has `f_b = 2.00`, segment B has `f_b = 20.00`, both with `a = 180.00`. The pooled optimum computed from the average friction of 11.00 is `11.00/191.00 = 0.057592`. But the true optima are `0.010989` and `0.100000` respectively. Applying the pooled boundary to A under-blocks (it acts too rarely) and to B over-blocks (it acts too often), and the two errors do not cancel in cost even though they look like they should.

**Counterexample three: a boundary is meaningless without calibration.** A model outputs a score in [0, 12]. Applying `p* = 0.032258` to that score blocks nothing at all, because a score of 0.032 on that scale is nearly the minimum. The same threshold applied after a logistic calibration map blocks 0.9 percent of rows. Identical number, opposite policy: the boundary only means something in the units of `p`, and `p` only exists after a calibration fit on a named population.

**Counterexample four: the loss is not linear in the number of blocked rows.** When a merchant is blocked, the loss is not one transaction but the merchant's entire relationship, and the relationship value is not proportional to the blocked amount. A boundary derived from per-row costs therefore systematically under-blocks merchant-level risk, because the true `f_b` for a merchant is two orders of magnitude above the per-row friction. The repair is a separate boundary at merchant grain, which is the grain lesson of the companion sheet applied to the schedule.

**Counterexample five: friction is not constant in time.** A friction of 6.00 dollars per false block in a month with 36,000 blocks is 216,000 dollars, which support can absorb. The same 6.00 per block in a peak week with 180,000 blocks overwhelms support, and the marginal friction rises because the queue itself becomes the cost. Any schedule that assumes a constant `f_b` at high volume understates the friction of an aggressive policy exactly when the aggressive policy is most tempting.

**The one limit case everyone should compute.** Set `p = 1` and compare: allowing costs `a`, blocking costs 0. Set `p = 0`: allowing costs 0, blocking costs `f_b`. Both extreme rows confirm the direction of the boundary and catch sign errors in an implementation. A boundary that is not between those two behaviours is a coding defect, and the check costs one line.

**Counterexample six: the friction of a challenge that nobody completes.** A challenge's friction is not the cost of sending it; it is the cost of sending it plus the cost of the customers who abandon mid-flow. If 6 percent of challenged legitimate customers abandon the transaction entirely, the challenge has destroyed a legitimate purchase. Valued at the average order margin, that abandonment is a real loss that the schedule's `f_c` must include. With 40,000 challenged rows, 39,200 legitimate, a 6 percent abandonment rate destroys 2,352 purchases; at 9 dollars of margin each that is 21,168 dollars, against a nominal challenge cost of 70,560. Abandonment raises effective challenge friction from 1.80 to 2.34 and narrows the band by roughly 2 percentage points.

**Counterexample seven: the boundary that is optimal for the mean and disastrous for the tail.** Suppose loss per fraud is 180 dollars in the mean but 4,000 dollars for 2 percent of cases, which is a realistic shape in card-not-present fraud. A boundary computed from the mean blocks the same row regardless of which case it is, so it under-protects precisely the rows that will be expensive. The repair is a two-stage rule: a lower boundary for rows whose amount or merchant category puts them in the expensive tail, combined with a portfolio-level cap on the total friction the policy may spend. Both changes are schedule-level, not model-level, and both are invisible to any metric computed on counts.

**Counterexample eight: correlated rows make the per-row optimum collectively suboptimal.** A per-row rule is optimal row by row, but if the rows in a period are correlated — a single attack producing 400 events in twenty minutes — then the policy blocks 400 rows and pays friction 400 times while the attack is stopped once. The per-row calculation therefore overstates the cost of blocking in exactly the episode where blocking matters most. The practical correction is to compute expected cost per attack episode rather than per row when episodes are identifiable, and to note that the boundary in episode terms is materially lower than in row terms.

**Counterexample nine: the shadow price is not the salary.** When one analyst's work resolves five linked alerts, the effective shadow price per completed item is a fifth of the per-alert density, because the analyst was paid once. Treating alerts as independent overstates the cost of capacity by the clustering factor and therefore under-invests in review. The clustering factor is measurable as items-per-investigation and should be computed before any capacity business case is accepted.

## 5. Worked Numerical Example

The trace below is a constructed card-authorisation portfolio. All counts, amounts, and recovery rates are ILLUSTRATIVE; every product, ratio, and total is recomputed here and independently re-checked in 5e.

### 5a Portfolio and schedule

| Quantity | Symbol | Value | Source of the value in this trace |
|---|---|---|---|
| Authorisations per month | `N` | 4,000,000 | constructed log count |
| Fraudulent authorisations | — | 3,600 | constructed adjudication ledger |
| Population prevalence | `pi` | 0.000900 | 3,600 / 4,000,000 |
| Mean loss per allowed fraud | `a` | $180.00 | constructed ledger average |
| Gross monthly fraud loss | — | $648,000 | 3,600 × 180.00 |
| Friction per false block | `f_b` | $6.00 | support + reissue + annoyance |
| Friction per false challenge | `f_c` | $1.80 | SMS plus step-up completion |
| Residual loss per failed challenge | `a_c` | $54.00 | 30 percent of `a` survives |

The schedule has four numbers and each is separately arguable, which is the point: a reviewer can challenge `a_c = 54.00` without reopening the whole analysis. That is what a schedule is for.

### 5b Boundaries and the resulting bands

```text
tau_low  = 1.80 / (180.00 - 54.00 + 1.80) = 1.80 / 127.80  = 0.014085
tau_high = (6.00 - 1.80) / (54.00 + 6.00 - 1.80) = 4.20 / 58.20 = 0.072165
p*_block_only = 6.00 / 186.00 = 0.032258
```

Applying those boundaries to the constructed score distribution produces three bands. The counts are constructed to be internally consistent with 3,600 frauds.

| Band | Risk range | Rows | Fraud rows | Precision | Action |
|---|---|---|---|---|---|
| High | above 0.072165 | 22,000 | 1,700 | 0.077273 | block |
| Middle | 0.014085 to 0.072165 | 40,000 | 800 | 0.020000 | challenge |
| Low | below 0.014085 | 3,938,000 | 1,100 | 0.000279 | allow |
| Total | — | 4,000,000 | 3,600 | 0.000900 | — |

The band precisions are 7.7 percent, 2.0 percent, and 0.03 percent. Those three numbers describe one policy, and anyone who quotes a single "precision" for the system is quoting whichever band suits the argument. The middle band is the largest in row count and the lowest in precision, which is exactly why it needs a cheap action rather than a decision.

### 5c Expected cost of the three-action policy

Cost is measured against a hypothetical perfect policy that prevents every fraud at no friction, so the numbers below are the cost of imperfection. All currency in dollars per month.

| Line | Arithmetic | Result |
|---|---|---|
| Block band: friction | (22,000 − 1,700) × 6.00 | $121,800 |
| Block band: fraud loss | 0 (all 1,700 prevented) | $0 |
| Challenge band: friction | (40,000 − 800) × 1.80 | $70,560 |
| Challenge band: residual loss | 800 × 54.00 | $43,200 |
| Allow band: fraud loss | 1,100 × 180.00 | $198,000 |
| Total cost | 121,800 + 70,560 + 43,200 + 198,000 | $433,560 |
| Loss avoided | 648,000 − 433,560 | $214,440 |
| Cost as share of gross loss | 433,560 / 648,000 | 0.669074 |

The policy recovers 33.1 percent of the gross loss. That headline is unremarkable on its own; what makes it worth reporting is the comparison with the alternatives in 5d, because the absolute number depends entirely on the constructed band counts.

### 5d Alternative policies on the same population

**Policy S, single boundary at `p* = 0.032258`.** Only block is available, so everything above 3.23 percent is blocked and everything below is allowed. The block band now contains the high band plus part of the middle band.

| Line | Arithmetic | Result |
|---|---|---|
| Blocked rows above 0.032258 | 22,000 + 40,000 (constructed) | 62,000 |
| Fraud in blocked set | 1,700 + 800 | 2,500 |
| Friction | (62,000 − 2,500) × 6.00 | $357,000 |
| Fraud loss on allowed | 1,100 × 180.00 | $198,000 |
| Total cost | 357,000 + 198,000 | $555,000 |
| Comparison with three-action | 555,000 − 433,560 | +$121,440 worse |

That single-threshold policy is catastrophically expensive because it charges 6.00 dollars to stop rows that a 1.80-dollar challenge would have handled. The alternative single-threshold policy, at `tau_high = 0.072165`, blocks only the high band.

| Line | Arithmetic | Result |
|---|---|---|
| Friction | (22,000 − 1,700) × 6.00 | $121,800 |
| Fraud loss on all other rows | (3,600 − 1,700) × 180.00 | $342,000 |
| Total cost | 121,800 + 342,000 | $463,800 |
| Comparison with three-action | 463,800 − 433,560 | +$30,240 worse |

So the challenge band is worth 30,240 dollars a month, and it is worth it for a precise reason: it covers 40,000 rows at 1.80 dollars instead of either blocking them at 6.00 or allowing them outright at an expected 800 × 180 = 144,000 dollars of loss. The challenge costs 113,760 and returns 144,000, a net 30,240. Every input above is constructed, but the structure of the argument is the reusable part.

### 5e Independent re-check

Each load-bearing figure is recomputed by a second route. Agreement to six decimals is the acceptance criterion.

```text
1) Middle-band net benefit
   route A: loss avoided 800*180.00 = 144,000.00
            challenge cost 39,200*1.80 + 800*54.00 = 70,560 + 43,200 = 113,760.00
            net = 30,240.00
   route B: per-row expectation over the band
            p_bar = 800/40,000 = 0.020000
            gain per row = p*(a - a_c) - (1-p)*f_c
                         = 0.02*126.00 - 0.98*1.80
                         = 2.52 - 1.764 = 0.756
            total = 40,000 * 0.756 = 30,240.00                    OK

2) Boundaries
   tau_low via odds: f_c/(a-a_c) = 1.80/126.00 = 0.014286
                     p = 0.014286/1.014286 = 0.014085             OK
   tau_high via odds: (f_b-f_c)/a_c = 4.20/54.00 = 0.077778
                      p = 0.077778/1.077778 = 0.072165            OK

3) Total cost
   direct sum: 121,800 + 70,560 + 43,200 + 198,000 = 433,560.00  OK
   alternative: gross 648,000 - avoided 214,440 = 433,560.00      OK

4) Band counts
   22,000 + 40,000 + 3,938,000 = 4,000,000                        OK
   1,700 + 800 + 1,100 = 3,600                                    OK

5) Policy S cost
   friction (62,000-2,500)*6.00 = 59,500*6.00 = 357,000.00        OK
   total 357,000 + 198,000 = 555,000.00                           OK
```

The odds route in check 2 is the one worth internalising. Converting `f_c/(a − a_c)` into a probability by `x/(1+x)` is faster and less error-prone than substituting into the probability formula directly, and it makes the cost ratio visible as a single number.

### 5f Sensitivity

**Recovery rate.** The challenge band exists because `a_c < a`. If the challenge stops only 40 percent of challenged fraud rather than 70 percent, then `a_c = 108.00`, and both boundaries move inward: `tau_low = 1.80/73.80 = 0.024390` and `tau_high = 4.20/112.20 = 0.037433`. The band shrinks from 5.81 to 1.30 percentage points, a 78 percent reduction. A team that assumes 70 percent recovery without measuring it may be running a band that is four times too wide, which means challenging tens of thousands of rows that should have been allowed.

**Attrition.** With `q = 0.005` and `L = 400.00`, effective block friction rises from 6.00 to 8.00, `p*` rises from 0.032258 to 0.042553, and `tau_high` rises to `(8.00 − 1.80)/(54.00 + 8.00 − 1.80) = 6.20/60.20 = 0.102990`. The block band narrows by 22,000 rows in the constructed distribution, roughly 30 percent of blocks, which is a large operational change produced by a single unmeasured parameter.

**Prevalence.** Doubling prevalence while holding the score distribution's shape leaves `p*` unchanged, because the boundary is a cost ratio. What changes is volume: the block band roughly doubles and the challenge band roughly doubles. In the constructed portfolio, doubling prevalence to 7,200 frauds with proportional band growth would take blocked rows from 22,000 to about 44,000 and consumed review capacity from 40,000 to about 80,000. The boundary is stable; the workload is not.

**Loss amount.** If `a` falls to 90.00 because liability rules change, `p*_block` rises to `6.00/96.00 = 0.062500` and `tau_high` rises to `4.20/58.20` unchanged only if `a_c` scales too; with `a_c` unchanged at 54.00 the block boundary becomes `4.20/(54.00 + 4.20) = 0.072165` and the band's logic breaks down because a challenge now saves only 36.00 of a 90.00 loss. The lesson is that the schedule must be rescaled as a whole, not one term at a time.

## 6. Visual / Geometric Intuition

**The cost-crossing picture.** Plot cost against risk for each action. Allow is a line through the origin with slope `a`. Block is a line falling from `f_b` at zero risk to zero at certainty. Challenge is a line from `f_c` at zero to `a_c` at one. The optimum at any risk is the lowest of the three lines, and the crossings of the lines are the two boundaries.

```text
cost
 180 |*                          allow (slope 180)
     | *
     |          *
  54 |                    *      challenge (slope 126, intercept 1.8)
   6 |#-------#                   block (slope -6, intercept 6)
     +------+------+-----------+------> risk p
       0.0141  0.0722         1.0
```

The picture makes three things obvious that the algebra hides. First, the allow line is much steeper than the others, so the low-risk region is wide and the high-risk region is narrow. Second, the block line falls steeply in cost as risk rises, which is why it wins at the top. Third, and most usefully, the challenge line is almost flat over the middle region, which means the exact position of the challenge boundary matters much less than the exact position of the block boundary: misplacing `tau_low` by 20 percent costs almost nothing, while misplacing `tau_high` by 20 percent changes thousands of blocks.

**Marginal-benefit density.** In the capacity variant, plot `v_i / h_i` against rank. The curve typically falls by two orders of magnitude from the top item to the 1,000th, and the operating point is the horizontal line at `lambda` that cuts the cumulative hours at `H`.

```text
density ($/hour)
  837 |*
      | *
  189 |  * *
      |      *   *  *
    9 |                *   *   *   *   *
      +-----------------------------------> cumulative hours
        200      1,000    3,000    8,850
                        ^
                     lambda for H = 1,000
```

The visual point is that the curve is extremely convex. Doubling the review budget from 1,000 to 2,000 hours moves the boundary from about 189 to some much lower density, and adds far less value per hour than the first 200 hours did. Any capacity proposal should be accompanied by this curve, because the curve is what converts "we need more analysts" into "the next 200 hours are worth X".

**The threshold band as a strip.** Draw the risk axis as a strip and shade three regions. The challenge region is a wide strip in the middle, and its width is the policy's tolerance. Two failure patterns are visible immediately: a strip so narrow it is effectively a line (the challenge adds nothing), and a strip so wide that a large share of the population receives friction (the policy is expensive).

```text
allow ................|========== challenge ==========|######## block
                    0.0141      5.81 points wide   0.0722
     3,938,000 rows        40,000 rows            22,000 rows
```

The row counts under the strip are the operational consequence. A 5.81-point band covering 40,000 rows is a middle band with 1 percent of volume, but on a differently shaped score distribution the same band could cover 5 percent of volume. Band width is a property of the schedule and the score distribution jointly, never of the schedule alone.

**Precision-volume curve during a prevalence shift.** Fix the operating point in TPR/FPR space and move prevalence. At `TPR = 0.62` and `FPR = 0.005` on 4,000,000 rows, a prevalence of 0.000900 gives 2,232 true positives against 19,982 false positives and a precision of 10.0 percent. Raising prevalence to 0.003000 gives 7,440 true positives against 19,940 false positives and a precision of 27.2 percent.

```text
prevalence   TP       FP       flagged   precision
0.000900   2,232   19,982     22,214     0.100477
0.003000   7,440   19,940     27,380     0.271731
0.010000  24,800   19,800     44,600     0.556054
```

This is the most counter-intuitive picture in the sheet. Flagged volume rises by only 23 percent while precision rises by 171 percent, because the false-positive term is almost fixed at these prevalence levels. Analysts therefore experience a fraud surge as "the same workload, suddenly much more productive", and a team that reacts by tightening the volume target will destroy exactly the signal the surge created.

**Amount-conditioned boundary as a hyperbola.** The boundary `p*(A) = f_b/(f_b + rA)` is a hyperbola in amount. It is steep at small amounts, nearly flat above ten times the friction, and flat by construction above any liability cap. The knee of the curve is at `A = f_b / r`, which in the 5f example is `6.00/0.90 = 6.67`. Every row below the knee needs a very high risk before blocking is worth it, and every row above it needs very little.

**Where the knee sits in dollar terms.** The knee is the amount at which the loss term equals the friction term. Below it, friction dominates the expected cost and a block is almost never justified; above it, loss dominates and the boundary falls quickly. With `f_b = 6.00` and a 90 percent loss rate the knee is 6.67 dollars, so a policy that uses one boundary for a book spanning 5-dollar coffee purchases and 900-dollar electronics is mis-specified at both ends. The useful presentation to a business audience is a small table of boundaries by amount decile, because deciles make the hyperbola concrete and prevent the argument from becoming abstract.

```text
amount decile   mean amount   loss (0.90x)   boundary p*
1                     8.00          7.20       0.454545
2                    22.00         19.80       0.232558
3                    41.00         36.90       0.139861
5                    96.00         86.40       0.064935
7                   210.00        189.00       0.030769
9                   640.00        576.00       0.010309
10                1,450.00      1,305.00       0.004577
```

The ratio from the first decile to the last is 99.4 to one. No single boundary can be within a factor of two of correct across that range, which is the strongest available argument for amount-conditioned policy — stronger than any model-quality argument, because it holds regardless of how good the model is.

**The two-boundary picture with a shifted schedule.** Redraw the three-line cost chart after attrition raises `f_b` from 6.00 to 8.00. The block line now starts higher and falls steeper, so it crosses the challenge line further right: `tau_high` moves from 0.072165 to 0.102990. The visual consequence is that the block region shrinks and the challenge region grows, meaning an unmeasured attrition term mechanically converts blocks into challenges. Operations will observe this as "the model got more cautious" when in fact the schedule changed and the model did not.

**Why the block line is the steepest lever.** In the cost chart, the slope of each line is the marginal change in cost per unit of risk. Allow's slope is `a` (180), challenge's is `a − a_c` plus the `f_c` intercept (126), and block's is `−f_b` (−6). The block line is nearly flat compared with the other two, which means block cost is almost independent of risk and therefore block is chosen on the basis of the schedule rather than the score. That is the geometric reason a block boundary is so sensitive to `f_b` and so insensitive to model quality: at the top of the risk range the score is already decisive, and the only remaining question is whether the friction is affordable.

**Reading the marginal-density curve as a budget conversation.** The convexity in the earlier density chart is the entire content of a capacity business case. Two points on the curve tell the story: the first 200 hours return 167,532 dollars and the next 800 hours return 150,944. Doubling capacity from 200 to 1,000 hours therefore increases value by 90 percent, not 400 percent. A manager who wants four times the value needs roughly four times the hours at the flat part of the curve, where the value per hour has already fallen below the item cost. Presenting the curve rather than the totals is what makes the conversation about marginal value instead of about ambition.

## 7. Comparison / Selection Table

### 7a The action ladder with cost structure

| Action | Cost when legitimate | Cost when fraudulent | Reversible | Typical latency | Fails how |
|---|---|---|---|---|---|
| Allow | 0 | `a` | Not applicable | None | Silent loss |
| Monitor only | negligible | `a` | Yes | None | No protection, only evidence |
| Challenge | `f_c` | `a_c` | Yes | seconds to minutes | Abandonment, false completion |
| Refer | `f_c + h·c_h` | `a_c + h·c_h` | Yes, until queue clears | hours to days | Queue timeout |
| Decline event | `f_b` | 0 | No | milliseconds | Customer friction, retry |
| Block entity | `f_b + q·L` | 0 | Manual | minutes | Account closure, complaint |
| Abstain | default path cost | default path cost | Yes | None | Unowned error |

Reading down the reversibility column splits the ladder into two halves: challenge and refer can be undone, while decline and block cannot. That split, not the cost, is what usually decides the action set in regulated contexts, because an irreversible wrong action generates a complaint that costs more than the friction number in the schedule.

### 7b Calibration methods and what each buys

| Method | Data needed | Repairs | Does not repair | Cost |
|---|---|---|---|---|
| None (raw score) | none | nothing | any scale error | zero, and the boundary is arbitrary |
| Platt scaling | mature holdout, ~1,000 positives | monotone distortion | non-monotone distortion | minutes |
| Isotonic | mature holdout, ~10,000 positives | any monotone distortion | extrapolation beyond range | minutes, overfits in sparse bands |
| Prior correction | sampling rates only | rebalancing shift | feature or population drift | negligible |
| Band-wise recalibration | large holdout per band | band-specific error | small-sample bands | days of analysis |
| Refit the model | full retrain | everything | nothing, if data is unbiased | weeks |

The practical hierarchy is prior correction first, then Platt, then isotonic, and a refit only when the discrimination itself has degraded. The order matters because a refit is expensive and is frequently proposed to fix what is actually a scale error that Platt scaling would repair in an afternoon.

### 7c Boundary strategy selection

| Strategy | Fixed quantity | Moves under prevalence shift | Use when |
|---|---|---|---|
| Probability boundary | loss ratio | volume | finance owns the loss target |
| Quantile boundary | volume | loss ratio | operations is capacity bound |
| Fixed expected-cost boundary | cost per period | both | the business case is quoted in currency |
| Fixed precision | precision | volume and recall | a quality promise is contractual |
| Fixed recall | recall | volume and precision | regulatory or contractual coverage |
| Shadow-price boundary | budget | boundary and volume | review is the scarce input |

No row in this table is universally correct, but exactly one is defensible for a given business, and the choice must be written down. The most common institutional failure is that finance manages a probability boundary while operations manages a quantile boundary, and the two boundaries diverge silently until a month when both teams report a problem that neither can explain.

### 7d Loss-schedule elements and how each is estimated

| Element | Symbol | Estimation route | Main bias | Direction of bias |
|---|---|---|---|---|
| Fraud loss per row | `a` | mature ledger, mean over confirmed | unreported fraud omitted | downward |
| Residual after challenge | `a_c` | challenged cohort, matched | selection into challenge | ambiguous |
| Challenge friction | `f_c` | cost of the step plus abandonment | abandonment rarely priced | downward |
| Block friction | `f_b` | support cost plus reissue | attrition omitted | strongly downward |
| Attrition | `q` | holdout comparison | confounded with fraud | ambiguous |
| Margin per customer | `L` | finance model | short horizon | downward |
| Review cost | `c_h` | loaded salary plus tooling | queue delay omitted | downward |

Six of the seven biases run downward, which means the computed boundary is systematically too low and the policy systematically too aggressive. That asymmetry is worth stating in any threshold review: the errors in the schedule do not cancel, they compound in one direction.

### 7e Operating-point documentation minimum

| Field | Example form | Why it is required | Consequence if missing |
|---|---|---|---|
| Population | issuer authorisations, all channels, monthly | Fixes the denominator | Boundary transferred to another population mis-prices |
| Grain | event | Fixes the friction multiplier | Friction understated by the events-per-entity factor |
| Schedule version | v7, dated | Fixes the cost inputs | Cannot attribute a loss shift to costs or to model |
| Action set | allow, challenge, block | Fixes what the boundaries mean | Middle boundary silently unused |
| Boundaries | 0.014085, 0.072165 | Fixes the policy | Not reproducible from the record |
| Calibration fit | Platt, fitted on 2026-06 cohort | Fixes the score-to-risk map | Boundary applied to raw score |
| Capacity | 6,000 cases/month completed | Fixes feasibility | Infeasible queue accepted as policy |
| Shadow price | 188.69 per hour | Fixes the capacity argument | No basis for staffing decisions |

Every row in this table corresponds to a failure in Section 8. That is not a coincidence: the documentation minimum is essentially the list of inputs whose absence produces a named failure mode, rewritten as a record-keeping requirement. A team that fills this table for each policy version has already implemented most of the controls that audits check for.

### 7f Choosing between schedule-first and capacity-first design

| Situation | Start from | Reason | First number to compute |
|---|---|---|---|
| New programme, no queue yet | Schedule | Capacity can be built to fit the optimum | `f_b/a` and `tau_high` |
| Existing queue, fixed headcount | Capacity | The constraint is not negotiable within the year | `v_i/h_i` distribution and `lambda` |
| Regulatory coverage mandate | Recall | A floor on coverage may dominate cost | Recall at fixed volume and its loss |
| Contractual precision promise | Precision | The promise is measured, not modelled | Volume implied by the precision target |
| Loss budget set by finance | Expected cost | The budget is the binding constraint | Cost per 10,000 decisions |
| Model refresh decision only | Calibration | The boundary should not move | Share of rows above the boundary before and after |

The right-hand column is what makes this table operational rather than descriptive. Whatever the situation, there is exactly one number to compute first, and computing it before the others prevents the common pattern of building a full cost model for a decision that is actually constrained by something the cost model cannot change.

## 8. Failure Modes & Edge Cases

**8.1 Boundary applied to an uncalibrated score.** *Symptom:* a threshold change has no effect on volume, or a tiny change flips the entire population. *Competing causes:* (a) the score is not a probability and its scale is unrelated to [0,1]; (b) the calibration map was fitted on a different population; (c) the score is a rank and the boundary is above the maximum value. *Discriminating check:* compute the share of rows above the boundary and compare it with the intended share. If the share is 0 or 1, the scale is wrong; if it is plausible but the loss is wrong, the calibration is wrong. *Repair:* fit a calibration map on a mature holdout before touching the boundary.

**8.2 Challenge band collapsed by a weak challenge.** *Symptom:* the middle band challenges thousands of rows and the fraud rate in that band does not fall. *Competing causes:* (a) the challenge is easily defeated and `a_c` is close to `a`; (b) challenged frauds move to a channel where the outcome is unobserved; (c) the band is dominated by rows that would have been legitimate. *Discriminating check:* measure the fraud rate among challenged rows against the rate among a small held-out allowed share. If the two are equal, `a_c = a` and the challenge is pure cost.

**8.3 Stale loss schedule.** *Symptom:* expected cost computed from the schedule diverges from the finance ledger over three consecutive months. *Competing causes:* (a) the recovery or liability rules changed; (b) the ledger now nets recoveries differently; (c) fraud mix shifted toward larger amounts. *Discriminating check:* recompute `a` from the ledger by month and compare with the schedule value. A monotone drift means the schedule is stale; a step change means a rule changed; a stable `a` with a drifting total means the mix changed.

**8.4 Capacity treated as a cost.** *Symptom:* the optimiser recommends referring 40,000 rows a month when analysts can complete 6,000. *Competing causes:* (a) capacity was not modelled as a constraint; (b) hours per item were assumed uniform and the queue is dominated by expensive items; (c) arrived items were confused with completed items. *Discriminating check:* compute total required hours `sum h_i` for the recommended set and divide by available hours. A ratio above 1.0 means the recommendation is infeasible; a ratio below 0.7 means capacity is idle and the boundary should loosen.

**8.5 Fixed quantile hiding a schedule change.** *Symptom:* the top-1-percent policy produces 60 percent more friction this quarter with the same volume. *Competing causes:* (a) score distribution shifted so the top 1 percent sits at lower risk and therefore holds more legitimate rows; (b) friction per block rose; (c) the definition of a block changed to include challenges. *Discriminating check:* compute the implied risk at the quantile cut-off for both quarters. If it fell, the quantile rule silently loosened; recompute expected cost at the new implied risk and compare.

**8.6 Aggregation across segments with different schedules.** *Symptom:* each segment's own analysis says its boundary is right, and the pooled boundary loses money on both. *Competing causes:* (a) the pooled schedule is an average of dissimilar costs; (b) one segment dominates the population and its optimum was applied to all; (c) the segments have different calibrations so identical scores mean different risks. *Discriminating check:* compute the boundary per segment and compare with the pooled boundary. If the per-segment boundaries differ by more than a factor of two, pooling is invalid and the pooled number should be withdrawn rather than defended.

**8.7 Action feedback contaminating the evaluation.** *Symptom:* a policy evaluation shows a large loss reduction that does not appear in the ledger. *Competing causes:* (a) displaced fraud went to an unmonitored channel; (b) the decline suppressed the only channel through which the fraud would have been labelled; (c) the comparison period had unusually low fraud. *Discriminating check:* hold a random 5 percent share under the previous policy and compare total loss across the two groups including all channels. Only that comparison has a causal interpretation, and the share must be large enough to detect the effect.

**8.8 Threshold churn without versioning.** *Symptom:* the loss per decision drifts for reasons nobody can attribute, and two analysts report different boundaries for the same week. *Competing causes:* (a) the boundary lives in the serving code and was hot-patched; (b) different environments hold different values; (c) a shadow policy is running for a subset. *Discriminating check:* log the boundary value and the schedule version with every decision, then group decisions by that pair. Any group with a value not in the change log identifies the leak.

**8.9 Loss measured per event but friction measured per customer.** *Symptom:* the block boundary is set from a schedule in which `a` is per transaction and `f_b` is per customer contact, and the resulting policy over-blocks high-frequency customers. *Competing causes:* (a) genuinely higher risk on frequent customers; (b) dimensional mismatch favouring blocks on customers with many events; (c) the friction was amortised incorrectly. *Discriminating check:* recompute friction per event by dividing customer-level friction by the mean events per customer in the period. If the recomputed boundary moves by more than 20 percent, the original mixed units.

**8.10 Shadow price mistaken for average cost.** *Symptom:* a proposal justifies new analysts by comparing benefit against the average cost per review rather than the marginal cost. *Competing causes:* (a) the benefit curve is flat at the margin so added capacity is worth little; (b) average and marginal costs coincide because the team is at constant returns; (c) the benefit estimate double-counts items already handled. *Discriminating check:* compute the density of the 1,000th and 1,001st item and compare with the loaded hourly cost. If density is below cost, the extra capacity does not pay for itself no matter how impressive the total benefit looks.

**8.11 Uncertainty ignored in the schedule.** *Symptom:* a boundary chosen from a point estimate of `a` performs badly in months when losses are high. *Competing causes:* (a) `a` is heavy tailed and its mean is unstable; (b) the boundary is optimal only near the mean; (c) fraud mix changed. *Discriminating check:* recompute the policy's cost at the 5th and 95th percentile of `a`. If the cost ranking of the candidate policies flips between those quantiles, the choice is not robust and should be made on a stated risk preference rather than on the point estimate.

**8.12 Precision reported at a fixed threshold across a prevalence shift.** *Symptom:* precision falls from 27 to 10 percent with no change to the model, and the team proposes a retrain. *Competing causes:* (a) prevalence fell, so the same TPR/FPR produces worse precision; (b) calibration shifted, moving the effective boundary; (c) genuine degradation. *Discriminating check:* recompute TPR and FPR at the same score and compare across the two periods. If both are unchanged, the model is fine and the metric moved because prevalence did; report precision at both prevalences rather than the ratio.

## 9. Complexity & Cost Analysis

**Computational cost of applying a boundary.** Sorting is not required to apply a fixed boundary: one pass over `n` rows with a comparison is `O(n)` time and `O(1)` extra space, which for 4,000,000 rows is a few tens of milliseconds in a vectorised implementation. Choosing a quantile boundary requires either a sort at `O(n log n)` or a histogram approximation at `O(n)` with a resolution of `n/b` rows per bucket; for a 1,000-bucket histogram over 4,000,000 rows each bucket holds 4,000 rows, so the quantile is located to within 0.1 percent of the population. That precision is adequate for a volume target and is two orders of magnitude cheaper than a sort.

**Cost of fitting a schedule.** The four numbers in the schedule come from four different analyses, and their combined cost is dominated by the one that is usually skipped: attrition. Estimating `q` requires a controlled comparison between blocked-legitimate customers and similar unblocked ones, which in a reasonable design needs a few thousand treated customers and a quarter of observation. The arithmetic is trivial; the design is not, and that is why `q` is frequently replaced by a guess of zero, which biases the boundary downward by roughly 20 to 30 percent in the Section 5 portfolio.

**Cost of review capacity.** A completed review at 20 reviews per hour and a loaded cost of 25 dollars per hour costs 1.25 dollars. At 6 cases per analyst-day and 150 analyst-days per month the team competes 900 cases a month, which is 1,125 dollars per case if the team's fully loaded monthly cost is 1,012,500 dollars. The shadow price is not that average: it is the density of the last item admitted, which in the Section 4.3 trace sits just below 188.69 dollars per hour and is therefore far above the 25-dollar direct cost, because the items are valuable. Confusing the two numbers is how teams under-invest in review.

**Cost of calibration maintenance.** A Platt fit on 50,000 rows with 450 positives takes minutes to compute but requires a mature window of at least 120 days. With monthly refreshes the pipeline must therefore reach back four months, which conflicts with a quarterly model refit and forces either a shared feature store with point-in-time correctness or duplicated pipeline work. The engineering cost of that overlap is the real reason calibration is not refreshed as often as it should be.

**Cost of threshold churn.** Each boundary change requires re-running the evaluation, updating the monitoring baselines, notifying operations, and restating any trend charts. In a well-run team that is one day; in a team where the boundary is embedded in three services it is two weeks and a coordination meeting. The cost is not the change itself but the loss of comparability: a series with twelve changes in a year cannot be used to detect a slow trend, and slow trends are exactly what fraud control needs to detect.

**Cost of a missed challenge band.** In the Section 5 portfolio the challenge band is worth 30,240 dollars a month, or 362,880 a year, from a schedule decision that requires no model work. The cost of not having the band is not a modelling cost, it is a policy-design cost, and it is the single largest return available in most fraud programmes that already have a decent model. That claim is structural rather than measured: it follows from `f_c < f_b` and `a_c < a` in any workflow where a challenge exists.

**Where the costs are not additive.** The schedule treats friction, attrition, and review cost as separate lines, but they interact. A block that generates a support call consumes review capacity, so friction and capacity are the same resource at high volume. Treating them as separate lines lets a policy look affordable line by line while overrunning the combined budget, which is the analytical form of the operational complaint that "the model is fine but the contact centre is drowning".

**Cost of a wider challenge band, in one table.** Widening the band from 5.81 to, say, 11 points is not free even though the per-row friction is small, because the band captures a long tail of rows. The table below computes the marginal cost and benefit of extending the band downward by one percentage point of the constructed population, holding the recovery rate fixed.

| Extension | Extra rows | Extra frauds | Mean `p` in band | Friction cost | Loss avoided | Net |
|---|---|---|---|---|---|---|
| 1.4085 to 0.4 percent | 18,000 | 120 | 0.006667 | 32,184 | 15,120 | −17,064 |
| 0.4 to 0.2 percent | 26,000 | 60 | 0.002308 | 46,692 | 7,560 | −39,132 |
| 0.2 to 0.1 percent | 34,000 | 30 | 0.000882 | 61,146 | 3,780 | −57,366 |

Each row is computed with the same per-row test from Section 5e, `p·126 − (1−p)·1.80`, evaluated at the band's mean `p`. At the 1.4085 percent edge the term is zero by construction, and it goes monotonically negative below it: at `p = 0.006667` it is `0.840 − 1.788 = −0.948`, which times 18,000 rows gives −17,064. The third extension destroys three times as much value as the first while recovering a quarter of the loss, which is the arithmetic reason the lower boundary is a boundary at all rather than a matter of taste.

**Cost of measuring the schedule.** The four schedule inputs cost roughly: `a` two days from the ledger, `a_c` one week from a challenged cohort, `f_c` two days from the messaging provider plus an abandonment estimate, `f_b` one week from support records, `q` one quarter with a designed comparison, and `L` from the finance model. Summed, the cheap inputs take two weeks and the expensive one takes a quarter. Teams that skip `q` save a quarter and misprice the boundary by 20 to 30 percent for as long as the policy runs; at the scale of the Section 5 portfolio that is roughly 121,800 dollars of friction spent per month on a boundary that should have been 32 percent higher.

**Cost of running two policies in parallel during a transition.** A shadow policy that scores without acting costs almost nothing computationally, but a holdout that continues an old policy costs the difference between the two policies' losses on the holdout share. At a 5 percent holdout on a 648,000-dollar gross loss, if the new policy is expected to save 214,440, the holdout gives up about 10,722 dollars a month of that saving in exchange for a causal estimate. That is the price of the only evidence that will settle the displacement question, and it should be budgeted explicitly rather than discovered when finance asks for the number.

## 10. Key Theorems / Results Reference

**R1 — Two-action break-even.** `block` beats `allow` exactly when `p > f_b/(f_b + a)`. *Condition:* linear costs, single loss owner, no capacity limit. *Use:* every threshold argument reduces to this and a claim about what `f_b` and `a` are.

**R2 — Odds form of the same result.** The boundary is `f_b/a` in odds, so it depends only on the cost ratio. *Condition:* identical to R1. *Use:* it makes the invariance to prevalence evident and makes hand checks trivial.

**R3 — Three-action partition.** With allow, challenge, and block, the optimum is two boundaries `tau_low = f_c/(a − a_c + f_c)` and `tau_high = (f_b − f_c)/(a_c + f_b − f_c)`, and the challenge band exists exactly when `tau_low < tau_high`. *Condition:* `a_c < a` and `f_c < f_b`. *Use:* it shows that adding an action widens the friction-spending region and raises the block boundary.

**R4 — Challenge band value.** The per-row value of the middle band is `p(a − a_c) − (1 − p) f_c`, which is positive exactly inside the band and maximised where the two terms balance. *Condition:* linear costs. *Use:* it converts the band into a currency figure that finance can compare with alternatives.

**R5 — Capacity shadow price.** Under a binding hour constraint with independent items, the optimal allocation takes items in descending `v_i/h_i` and the shadow price equals the density of the last item admitted. *Condition:* independence, divisible effort. *Use:* it prices additional analyst capacity and shows why the average review cost is the wrong number.

**R6 — Boundary invariance in probability space.** A calibrated boundary does not move when prevalence moves; only the volume above it moves. *Condition:* calibration holds on the new population, schedule unchanged. *Use:* it is the reason a probability boundary is a stable policy and the reason a fraud surge appears as a workload event rather than a policy event.

**R7 — Non-invariance in quantile space.** A fixed quantile maps to a prevalence-dependent risk and therefore to a prevalence-dependent loss ratio. *Condition:* score distribution shifts with prevalence. *Use:* it explains why "top 1 percent" policies quietly change their economics.

**R8 — Prevalence sensitivity of precision at fixed rates.** At fixed TPR and FPR, precision is `pi·TPR / (pi·TPR + (1−pi)·FPR)`, which is concave and increases with `pi`. *Condition:* rates stable across populations. *Use:* it separates a metric move caused by prevalence from one caused by model degradation.

**R9 — Volume near-invariance at low prevalence.** When `pi` is small compared with `FPR`, flagged volume is approximately `N·FPR` and barely depends on prevalence. *Condition:* `pi·TPR << (1−pi)·FPR`. *Use:* it predicts the observed pattern where workload is flat during a surge and precision rises sharply.

**R10 — Attrition-adjusted friction.** `f_b_eff = f_direct + qL` and the boundary scales as `f_b_eff/(f_b_eff + a)`. *Condition:* attrition attributable to the block. *Use:* it quantifies why unmeasured attrition makes every policy too aggressive.

**R11 — Amount-conditioned boundary.** With loss proportional to amount, the correct boundary is `f_b/(f_b + rA_i)`, which is decreasing and convex in `A_i`. *Condition:* proportionality, no liability cap. *Use:* it shows a constant boundary is a specification error whenever amounts vary.

**R12 — Aggregation failure.** If segments differ in schedule, the pooled boundary is optimal for neither and the cost penalty is bounded below by the sum of per-segment deviations weighted by mass. *Condition:* segments identifiable. *Use:* it justifies segment-specific boundaries and explains why per-segment analyses can each be right while the pooled one is wrong.

**R13 — Robust boundary under schedule uncertainty.** Using a lower confidence bound on `a` raises the boundary by `(f_b + E[a])/(f_b + a_low)` and is the risk-averse optimum for a linear loss with a single-sided penalty. *Condition:* distribution of `a` known or approximated. *Use:* it makes the risk appetite explicit instead of hiding it inside a point estimate.

**R14 — Value of a holdout.** Without a randomised holdout the realised cost difference between two policies is confounded by selection, so only the expected-cost difference is estimable; with a holdout of share `g` the realised difference is estimable with standard error scaling as `1/sqrt(gN)`. *Condition:* random assignment independent of outcomes. *Use:* it sizes the holdout and settles arguments about whether a policy "actually worked".

**What these results do not establish.** None of R1 to R14 says anything about an adversary who responds to the policy. They are all single-agent optimisations. In fraud, a policy that is optimal against a stationary population is a policy that invites the population to move, and the resulting equilibrium is worse than the optimum computed on the pre-policy data. Treating the outputs of these formulas as final answers rather than as one-shot best responses is the most consequential misuse of the sheet.

**R15 — Friction-exchange rate of a boundary.** A boundary at `p*` implies that the business accepts one unit of friction per `(1 − p*)/p*` units of expected loss avoided. *Condition:* two-action case with linear costs. *Use:* it converts a probability into a governance statement that is comparable across portfolios, because the exchange rate is dimensionless while the probability is not.

**R16 — Abandonment-adjusted challenge friction.** With abandonment share `b` of challenged legitimate rows and margin `M` per completed purchase, `f_c_eff = f_c + b·M` and the band narrows by the ratio of the two friction terms. *Condition:* abandonment is attributable to the challenge. *Use:* it corrects the systematic understatement of challenge cost and explains why step-up flows lose money when abandonment is high.

**R17 — Episode-level boundary.** When `n_e` correlated rows form one episode, the effective friction per fraudulent episode is `f_b/n_e` at the margin, so the episode-level boundary is lower than the row-level boundary whenever `n_e > 1`. *Condition:* episodes identifiable and rows within an episode share an outcome. *Use:* it justifies looser row boundaries inside known attack episodes and prevents over-blocking during them.

**R18 — Holdout cost of a causal estimate.** A holdout of share `g` forgoes expected benefit `g·B` per period to obtain a realised estimate with standard error scaling as `1/sqrt(gN)`. *Condition:* benefit `B` constant and holdout randomly assigned. *Use:* it sets `g` by trading the cost of the experiment against the precision required, and it makes the trade explicit rather than ideological.

**R19 — Two-boundary sensitivity asymmetry.** Because `tau_low` depends on `f_c/(a − a_c)` while `tau_high` depends on `(f_b − f_c)/a_c`, a 20 percent error in `a_c` moves `tau_high` by roughly 20 percent and `tau_low` by less than 10 percent, while a 20 percent error in `f_c` moves `tau_low` by about 20 percent and `tau_high` by a smaller amount. *Condition:* the cost terms are of the sizes in the worked example. *Use:* it tells a reviewer which input to audit for which boundary.

**R20 — Band-width collapse threshold.** The challenge band disappears exactly when `a_c >= a·(f_b − f_c)/f_b`, which for the worked schedule gives `a_c >= 126.00`, i.e. a challenge must recover more than `f_c/f_b` of the loss — 30 percent here — for any band to exist at all. *Condition:* `f_c < f_b`; when friction is inverted the upper boundary is negative and the action is dominated everywhere. *Use:* it converts a debate about the recovery rate into a single pass-or-fail test, and at `a_c = 130.00` the boundaries cross (`tau_low = 0.034749 > tau_high = 0.031297`), confirming the collapse.

## 11. Interview / Exam Questions

**Q1. Compute the block boundary when friction is 8 dollars and mean fraud loss is 180 dollars, and say what changed if it used to be 3.23 percent.** `p* = 8/188 = 0.042553`. The boundary rose because friction rose from 6 to 8, which is a 33 percent increase in the numerator against a 1 percent increase in the denominator. The likely cause is an unmeasured attrition term of 2 dollars, so the underlying change is not the block cost but the customer value.

**Q2. Why is a 0.5 threshold equivalent to claiming fraud loss equals friction, and how wrong is that in the Section 5 portfolio?** At `p = 0.5` the indifference condition implies `f_b = a`. In the portfolio `a/f_b = 180/6 = 30`, so the correct boundary is 30 times lower; 0.5 is about 15.5 times the correct 3.23 percent boundary. The consequence is that almost nothing gets blocked and the loss is essentially the gross loss of 648,000 dollars a month.

**Q3. Derive the lower edge of the challenge band and explain why it is below the block-only boundary.** Challenge beats allow when `p(a − a_c) > (1−p)f_c`, giving `p > f_c/(a − a_c + f_c) = 1.80/127.80 = 0.014085`. It is below 3.23 percent because the challenge's friction is a third of the block's while it still removes 70 percent of the loss, so it dominates allowing over an interval where blocking would be too expensive.

**Q4. A colleague reports that precision halved after a prevalence drop and wants a retrain. What do you compute first?** TPR and FPR at the same score in both periods plus the two prevalences. Plug into `pi·TPR/(pi·TPR + (1−pi)FPR)`. If the predicted precisions match the observed ones, the model is unchanged and the metric moved for a reason a retrain cannot address; the correct action is to state the prevalence alongside the metric in future reports.

**Q5. Given 1,000 review hours, 400 high-value items at 0.5 hours each and 5,000 medium items at 0.5 hours each, which do you review and what is the shadow price?** All 400 high items use 200 hours and yield 167,532 dollars. The remaining 800 hours take 1,600 medium items worth 150,944 dollars. Total 318,476 dollars. The shadow price is just under the medium density of 188.69 dollars per hour, because the last admitted item sets the marginal value.

**Q6. Under what condition does the challenge band disappear, and what should you do then?** It disappears when `tau_low >= tau_high`. Substituting, that happens when `f_c` is large relative to `a − a_c` or when `f_b − f_c` is small relative to `a_c`. The response is to remove the challenge from the action set rather than to tune the boundaries, because an empty band means the challenge is dominated everywhere.

**Q7. Explain why the block boundary rises when a challenge action is added.** The block now competes against a cheaper action that already recovers most of the loss. The block must therefore prevent enough additional loss to justify its extra friction of `f_b − f_c`, and that requires higher risk. In the worked example the boundary moves from 3.23 to 7.22 percent, a factor of 2.24.

**Q8. Your ledger shows a monthly loss of 500,000 dollars and your schedule-based expected cost is 433,560 dollars. Which is right?** Both can be right if they measure different things. The ledger is realised cost including displaced fraud and unreported losses; the expected cost is a model on a constructed population. Compute the realised cost on the same population and period, then compare. A persistent gap of 15 percent is normal; a factor of two means the schedule's `a` is wrong or the population differs.

**Q9. Why does doubling the review budget not double the value of review?** Because the marginal density curve falls steeply. The first items are worth hundreds of dollars per hour and the thousandth a few tens. The gain from extra capacity is the integral of the curve between the two budgets, which is much less than proportional, and at some point falls below the loaded hourly cost of an analyst.

**Q10. A team wants to hold the top 1 percent fixed through a fraud surge. What breaks?** The loss ratio. Holding volume fixed means the implied risk boundary falls as prevalence rises, so the policy admits lower-risk rows at the same friction and the cost per avoided dollar worsens. The correct response to a surge with fixed capacity is to accept a higher boundary, state the loss consequence, and not to pretend the volume target is unchanged in meaning.

**Q11. Compute the boundary for a 20-dollar transaction and a 900-dollar transaction given a 90 percent loss rate and 6 dollars of friction.** At 20 dollars the loss is 18.00 and the boundary is `6/24 = 0.25`. At 900 dollars the loss is 810.00 and the boundary is `6/816 = 0.007353`. A single 3.23 percent boundary under-blocks the small case by a factor of 7.8 and over-blocks the large case by a factor of 4.4.

**Q12. What is the difference between a threshold and an operating point, and why does the distinction matter in a change log?** A threshold is one number; an operating point is the tuple of population, schedule version, action set, boundaries, and capacity. Changing the population while keeping the threshold still changes the policy's cost, so it must be logged as a policy change. Teams that log only the threshold cannot explain level shifts in their loss series.

**Q13. A policy review claims a 40 percent loss reduction. What single design would you ask for before believing it?** A randomised holdout that continues the previous policy on a small share, with total loss measured across all channels for both groups. Without it, the difference confounds prevention with displacement and with the policy's effect on which outcomes become observable.

**Q14. How do you price one additional analyst hour?** Compute `v_i/h_i` for the items at the current capacity boundary; the shadow price is the density of the last item admitted. If it exceeds the loaded hourly cost by a wide margin, capacity should expand; if it is below, the extra hour does not pay for itself and the boundary should tighten instead.

**Q15. What does the robust boundary formula tell a risk committee that the risk-neutral formula hides?** It tells them the cost of their risk appetite in boundary units. With a 95 percent lower bound on the loss the boundary moves by a factor of 4.9 in the worked example, which is a much larger effect than any modelling improvement they are likely to fund. That number belongs in the decision record, not in an appendix.

## 12. Quick Reference Card

| Item | Rule | Check |
|---|---|---|
| Block boundary | `p* = f_b/(f_b + a)`, odds `f_b/a` | Recompute with effective, not direct, friction |
| Challenge band | `tau_low = f_c/(a − a_c + f_c)` | Must be below `tau_high` or drop the action |
| Block upper edge | `tau_high = (f_b − f_c)/(a_c + f_b − f_c)` | Compare with the block-only boundary |
| Friction | Fold attrition in: `f_direct + q·L` | A miss here lowers the boundary 20–30 percent |
| Capacity | Take items by `v_i/h_i`; shadow price is the last item's density | Required hours ÷ available must be ≤ 1.0 |
| Calibration | Prior correction, then Platt, then isotonic | Check share of rows above the boundary is plausible |
| Prevalence shift | Boundary fixed in probability, volume fixed in quantile | State which one the business owns |
| Amount | `p*_i = f_b/(f_b + r·A_i)` | Flat boundary is a specification error |
| Evaluation | Random holdout, or report expected cost only | Without it, no causal claim |
