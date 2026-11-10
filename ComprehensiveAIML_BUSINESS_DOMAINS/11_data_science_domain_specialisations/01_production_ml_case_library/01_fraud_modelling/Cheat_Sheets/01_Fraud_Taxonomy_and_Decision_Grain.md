---
collection: "fraud-modelling"
file_id: "01"
topic: "Fraud taxonomy and decision grain"
level: 1
status: "draft"
version: "2.0"
created: "2026-11-09"
revised: "2026-11-10"
reviewed: null
review_trigger: "Material change to product actions, label sources, entity policy, or source availability"
notation_style: "plain-text"
prerequisites: []
pattern_tags: [decision-grain, label-delay, taxonomy, point-in-time]
source_status: "OPEN"
owner_role: "Fraud curriculum maintainer"
source_index: "INDEX.md"
---

# Fraud Taxonomy and Decision Grain

> **Purpose:** Classify a suspected-fraud workflow by product, actor, event, action, loss owner, label path, and decision deadline before anyone selects a model, a threshold, or a metric.
> **Prerequisites:** Basic probability, a timestamped event table, and the ability to distinguish an event from the entity that produced it.
> **Key outcome:** Given a described fraud workflow, name its family, its decision grain, its permissible actions, which labels are observed and which are structurally missing, and the metric that must not be borrowed from a neighbouring family.
> **This file does NOT cover:** threshold optimisation against a loss schedule, feature engineering, graph-model training, or jurisdiction-specific legal determinations.
> **Level:** 1 Foundations.
> **Version / reviewed:** 2.0 / not reviewed.
> **Evidence:** DERIVED identities with ILLUSTRATIVE inputs; no named organization, deployment, prevalence, or realised saving is measured here.
> **Retrieval cue:** If a team says "fraud is fraud" or proposes one label and one metric for several workflows, start at Section 2.

## 1. File Header

**Why this file exists.** Most failed fraud projects are not modelling failures. They are grain failures: the team trained on authorisation events, the business funds a card-level portfolio, the operations queue reviews cases, the regulator counts suspicious-activity reports, and the finance ledger books losses per claim. Five units of analysis, one model output, and every stakeholder computes a different number from the same predictions. This sheet gives a naming discipline that forces the unit, the clock, and the label path into the open before any modelling work starts.

**Scope of the teaching.** The sheet is written for payment authorisation, insurance claim, account-opening and account-takeover, anti-money-laundering alerting, and marketplace-abuse workflows. Those five families are used because they differ on the three axes that matter: how long the decision takes, what the system is permitted to do, and how the truth arrives. A taxonomist only needs those three axes plus the loss owner to place a new workflow correctly.

**What a taxonomy is not.** It is not a list of crime labels. "Phishing", "bust-out", "friendly fraud", and "synthetic identity" are *mechanism* words. They describe how an attacker operates, not what the model scores. Two workflows can share the same mechanism and still require different grains, different actions, and different metrics. Conversely one workflow can contain several mechanisms and still be a single decision unit. The taxonomy below is keyed to decisions, and mechanism vocabulary sits inside the mechanism column rather than at the top.

**Source status.** The course planning tree lists a taxonomy reference and five production fraud case topics. The underlying source bodies were not available for technical review, so every definition here is DERIVED from stated premises rather than quoted, and every numeric input is ILLUSTRATIVE. Numbers are constructed to make an arithmetic relation visible; they are not industry benchmarks and must not be cited as prevalence estimates.

**How to read the arithmetic.** All display formulas appear in fenced `text` blocks using the five-part anatomy: FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION. Each block closes with a validity condition stating when the relation stops holding. Where a relation is a definition rather than an empirical claim, the block says so explicitly, because a definition cannot be falsified by data and should not be presented as evidence.

**Notation conventions.** Rates are proportions in [0,1] unless a percent sign appears. Currency is US dollars and the symbol `$` always means the same unit within one block; no block mixes currencies. Counts are per stated calendar month. `t_d` is decision time, `t_e` is event time, and `t_m` is label-maturity time. Time is expressed in days from `t_e` unless stated. Subscript `e` marks an event-grain quantity and subscript `E` marks an entity-grain quantity.

**Maintenance.** The triggers are: a change to what an action is allowed to block, a change to the entity-definition policy (for example when a shared device starts linking cards), a change to the label source or its latency, or the appearance of a sourced measurement that contradicts an illustrative number. Any of those invalidates Sections 4, 5, and 7 together, so they must be revised as a set rather than patched individually.

**Reader contract.** If after reading Section 3 you cannot state, in one sentence each, the decision unit, the deadline, the action set, the label source, and the loss owner of the workflow in front of you, the workflow is not ready for a model. That sentence test is the actual deliverable of this sheet; everything else supports it.

## 2. Concept Map / Mental Model

The mental model is a ladder with three clocks attached. Reading down the ladder changes the denominator; reading across the clocks changes which rows are even visible at training time. Almost every metric dispute in fraud work is a disagreement about which rung and which clock the speaker is using.

```text
 rung 5  network / ring        one decision per cluster of linked entities
 rung 4  entity                one decision per card, account, policy, or merchant
 rung 3  case / claim          one decision per submitted claim or opened case
 rung 2  session / attempt     one decision per login or checkout attempt
 rung 1  event                 one decision per transaction, listing, or alert

 clocks:  t_e  event happened  ->  t_d  decision forced  ->  t_m  truth known
```

**Ladder rungs are not interchangeable.** Moving up a rung aggregates rows and therefore raises the base rate, because an entity with many chances to be fraudulent is more likely to be fraudulent at least once than any single one of its events. Moving down a rung multiplies friction: a policy that blocks entities charges one unit of customer friction, while the same policy expressed per event charges once per event on that entity. The base-rate effect and the friction effect push in opposite directions, and Section 4.2 turns that trade into an inequality.

**Three clocks, three different censoring patterns.** At `t_d` the system must act with whatever arrived before that instant. At `t_m` the label becomes trustworthy. Between them sits a window in which the outcome exists but is unrecorded, and a training set cut inside that window silently understates the fraud rate. Section 4.3 gives the correction and Section 8 lists the discriminating checks that separate censoring from genuine prevalence decline.

**The one-row-one-decision invariant.** A table is ready to model when every row corresponds to exactly one decision the business is allowed to take, taken at one instant, with one loss owner for the resulting error. Any table that violates one of those three conditions will produce a model whose offline metric cannot be reproduced in production, and no amount of hyperparameter work repairs it.

**Where the adversary sits in the model.** Fraud is one of the few prediction problems where the label distribution is partly produced by the model's own past actions. A blocked attempt never becomes a loss, so it never becomes a chargeback, so the training set learns that blocked-looking traffic is legitimate. The taxonomy must therefore record, for every row, whether the row was *scored and allowed*, *scored and blocked*, or *never scored*. That field is called the selection path in Section 3 and it is the single most commonly omitted column in fraud feature stores.

**Two failure shapes to keep in view.** First, metric transfer: an event-level recall number quoted for an entity-level policy overstates how many customers are actually protected, or understates it, depending on correlation. Second, action transfer: a rule proven on the decline action is not proven on the challenge action, because the challenge changes who completes the transaction and therefore what the outcome means. Both shapes appear repeatedly in Section 8.

**A compact way to remember it.** Grain decides the denominator; the clock decides the numerator; the action decides what the numerator will ever contain. Write those three clauses at the top of every fraud analysis document and most reviewer objections dissolve before they are raised.

## 3. Foundational Definitions

Definitions are grouped by the axis they belong to. The boundary column is the load-bearing part: each entry names the neighbouring concept the term is routinely confused with.

### 3a Units and grains

| Term | Meaning | Unit / type | Boundary (it is not) |
|---|---|---|---|
| Event | One timestamped occurrence the system can score: authorisation, claim line, login, listing, alert | row | Not a customer |
| Entity | The durable thing a loss is ultimately booked against: card, account, policy, merchant, device cluster | key | Not a transaction |
| Case | A human-opened investigation container that may reference several events and entities | work item | Not an outcome |
| Decision grain | The unit in which an action is taken and an error is charged | declaration | Not the training-table grain by default |
| Multiplicity `m` | Mean number of fraudulent events per fraudulent entity | events/entity | Not the mean event count per entity |
| Exposure `n̄` | Mean number of events per entity in the window, fraud or not | events/entity | Not the review count |
| Ring / cluster | A set of entities linked by shared devices, beneficiaries, or addresses | set | Not a legal finding |
| Entity resolution | The join policy that decides when two rows belong to one entity | procedure | Not a model feature |

### 3b Time and maturity

| Term | Meaning | Unit / type | Boundary (it is not) |
|---|---|---|---|
| Event time `t_e` | When the underlying act happened | timestamp | Not the log-write time |
| Decision time `t_d` | Deadline by which the action must be committed | timestamp | Not the batch score time |
| Label maturity `t_m` | When the outcome is final enough to train on | timestamp | Not the customer complaint date |
| Label lag | `t_m - t_e` | days | Not the investigation duration |
| Censoring | Outcome exists but is unobserved at the cut-off | condition | Not missing data in the random sense |
| Coverage | Share of eventual labels visible at a given cut-off | proportion | Not label accuracy |
| Chargeback window | Contractual period during which a dispute can be raised | days | Not the fraud-detection window |

### 3c Labels and truth

| Term | Meaning | Unit / type | Boundary (it is not) |
|---|---|---|---|
| Confirmed fraud | Adjudicated outcome with a retained decision record | boolean + evidence | Not suspicion |
| Dispute | Cardholder assertion, adjudication pending | boolean | Not confirmed fraud |
| Chargeback | Completed dispute with funds moved | boolean | A subset of disputes |
| Label-path sensitivity `κ` | Share of true frauds that ever generate a label | proportion | Not model recall |
| Weak label | Proxy used when the true label is unavailable | boolean | Not ground truth |
| Silence-as-legitimate | Treating an unlabelled row as a negative | assumption | Not a measurement |
| Adjudication drift | Change over time in what counts as confirmed | process | Not concept drift in features |
| Hard negative | Confirmed legitimate row that resembles fraud | row | Not an unlabelled row |

### 3d Actions and losses

| Term | Meaning | Unit / type | Boundary (it is not) |
|---|---|---|---|
| Allow | Commit the event with no added friction | action | Not absence of monitoring |
| Challenge | Insert a verification step before commit | action | Not a soft decline |
| Decline | Refuse the event outright | action | Not account closure |
| Refer | Send to a human queue without blocking | action | Not a decline with a review flag |
| Abstain | Score refused; route to default policy | action | Not a random action |
| Friction | Cost charged when a legitimate item is inconvenienced | currency/event | Not the fraud loss avoided |
| False decline | Legitimate item declined | event | Not a false alert |
| Loss owner | The budget line that absorbs the error | role | Not the model owner |
| Loss cap | Contractual limit on what one incident can cost | currency | Not expected loss |

### 3e Population and selection

| Term | Meaning | Unit / type | Boundary (it is not) |
|---|---|---|---|
| Prevalence `π` | Share of rows in the scored population that are fraudulent | proportion | Not the share of reviewed rows |
| Base rate | Prevalence at the grain being discussed | proportion | Not a constant across rungs |
| Selection path | Whether a row was allowed, blocked, or never scored | categorical | Not a feature |
| Review sampling | How human-reviewed rows were chosen | procedure | Not random sampling |
| Alert | A rule or model output that requests human attention | item | Not a case |
| Duplicate alert | A further alert on an entity with an open case | item | Not a new fraud |
| Disposition | The recorded outcome of an alert or case | categorical | Not a financial outcome |
| Typology drift | Change in attacker mechanism over time | process | Not population drift |

## 4. Core Formulas with Full Derivation

### 4.1 Base case: break-even probability at one grain

The first result is the only one every practitioner must be able to derive on a whiteboard, because every threshold argument reduces to it.

```text
FORMULA
  EL_allow(p) = p * a
  EL_block(p) = (1 - p) * f
  block  <=>  p * a  >  (1 - p) * f
  p* = f / (f + a)

WHERE
  p  calibrated probability that this row is fraudulent          [0,1]
  a  mean loss if a fraudulent row is allowed                    currency/row
  f  mean friction if a legitimate row is blocked                currency/row
  p* break-even probability, same units as p                     [0,1]

CONSTRAINT
  losses are linear in the outcome; no capacity limit; one loss owner;
  f does not depend on p; a does not depend on which other rows exist.

NUMERICAL EXAMPLE
  a = 180.00, f = 6.00  ->  p* = 6.00 / 186.00 = 0.032258

INTERPRETATION
  Blocking is worth it only above 3.23 percent posterior risk for this
  loss ratio. The number is a ratio of costs, not a property of the model.

VALIDITY CONDITION
  Holds only while a and f are constants across rows. Section 4.4 shows the
  counterexample in which a varies by a factor of ten.
```

The derivation is a dominance argument rather than a statistical one: for a single row there is no variance term, so the lower expected cost wins outright. That is why the result is stable and why it is so often misapplied — it is exact under assumptions that real portfolios violate.

### 4.2 General case: moving between grains

Two identities connect event grain and entity grain. The first is a prevalence transform; the second bounds how much recall can improve when the decision unit grows.

```text
FORMULA
  pi_E = pi_e * n_bar / m

WHERE
  pi_e   event-grain fraud prevalence                     [0,1]
  pi_E   entity-grain fraud prevalence (>=1 fraud event)  [0,1]
  n_bar  mean events per entity in the window             events/entity
  m      mean fraudulent events per fraudulent entity     events/entity

CONSTRAINT
  the window is the same for both counts; m > 0; events are counted once.

NUMERICAL EXAMPLE
  pi_e = 0.0009, n_bar = 16.0, m = 1.24138
  pi_E = 0.0009 * 16.0 / 1.24138 = 0.0144 / 1.24138 = 0.011600

INTERPRETATION
  The entity-grain base rate is 12.9 times the event-grain base rate here.
  A metric moved between the two grains without this factor is off by an
  order of magnitude, not by a rounding error.

VALIDITY CONDITION
  Requires that the counted events belong to entities inside the same window;
  breaks when entities straddle the boundary or when m is estimated on
  confirmed-only labels, which understate m.
```

The second identity is the recall bound. If an entity is flagged when any of its events is flagged, entity recall can never be worse than event recall and is at most the complement of the probability that every event is missed.

```text
FORMULA
  d_E(rho=0) = 1 - (1 - d_e)^m
  d_e <= d_E <= 1 - (1 - d_e)^m          for rho >= 0

WHERE
  d_e    event-grain detection rate for fraudulent events      [0,1]
  d_E    entity-grain detection rate for fraudulent entities   [0,1]
  rho    within-entity correlation of detection outcomes       [-1,1]

CONSTRAINT
  m >= 1; detection of one event does not change the entity's definition;
  the lower bound holds because a flagged event flags its entity.

NUMERICAL EXAMPLE
  d_e = 0.60, m = 1.24138
  ln(0.40) = -0.916291
  -0.916291 * 1.24138 = -1.137473
  exp(-1.137473) = 0.320540
  d_E upper bound = 1 - 0.320540 = 0.679460

INTERPRETATION
  Independent misses would lift detection from 60.0 to 68.0 percent purely by
  giving each entity more chances. A measured 65.0 percent therefore implies
  positive correlation, not a worse model.

VALIDITY CONDITION
  Fails when events on one entity are near-duplicates: with rho = 1 the bound
  collapses to d_e and the gain is exactly zero.
```

### 4.3 Variants: alert grain, label-path sensitivity, censoring

Alert-based workflows add an intermediate rung. Precision computed at the alert rung can differ by two orders of magnitude from precision at the case rung, and both statements can be arithmetically correct.

```text
FORMULA
  precision_alert = confirmed / alerts
  precision_case  = confirmed / cases
  alerts_per_case = alerts / cases
  precision_dedup = confirmed / (alerts * (1 - dup_share))

WHERE
  confirmed   adjudicated fraudulent units                       count
  alerts      rule or model outputs sent to a queue              count
  cases       human-opened investigations that reference alerts  count
  dup_share   share of alerts on an entity with an open case     [0,1]

CONSTRAINT
  every confirmed unit traces to at least one alert; dedup applies per entity.

NUMERICAL EXAMPLE
  60,000 alerts, 1,200 cases, 260 confirmed, dup_share = 0.34
  precision_alert = 260 / 60,000 = 0.004333
  precision_case  = 260 / 1,200  = 0.216667
  alerts_per_case = 60,000 / 1,200 = 50.0
  precision_dedup = 260 / (60,000 * 0.66) = 260 / 39,600 = 0.006566

INTERPRETATION
  0.43 percent, 21.7 percent, and 0.66 percent describe one system. The unit
  must be attached to every precision number or the comparison is meaningless.

VALIDITY CONDITION
  Holds while cases are opened only from alerts; a case opened from a tip
  breaks the denominator and must be counted separately.
```

The label-path correction inflates observed prevalence by the share of true frauds that the dispute process never records.

```text
FORMULA
  pi_true = pi_obs / kappa

WHERE
  pi_obs  share of rows carrying a confirmed-fraud label       [0,1]
  kappa   share of true frauds that ever produce that label    [0,1]
  pi_true prevalence among all true frauds                     [0,1]

CONSTRAINT
  label generation is independent of the features used to predict it; kappa>0.

NUMERICAL EXAMPLE
  pi_obs = 0.000900, kappa = 0.78
  pi_true = 0.000900 / 0.78 = 0.0011538

INTERPRETATION
  The true rate is 28 percent higher than the observed rate. The break-even
  threshold is unchanged because it depends on costs, but the expected loss of
  every allowing action rises by 28 percent.

VALIDITY CONDITION
  Breaks when kappa itself depends on amount: small frauds are systematically
  less likely to be disputed, which is the common case in card portfolios.
```

Censoring is the third variant, and it is the one that most often corrupts a training window without any error appearing in the logs.

```text
FORMULA
  coverage(t) = P(label_mature by age t)
  pi_apparent(window w) = pi_true * coverage(w)

WHERE
  w        age of the newest rows included in the window        days
  coverage share of eventual labels already visible at age w    [0,1]

CONSTRAINT
  coverage is measured on historical cohorts, not assumed linear in w.

NUMERICAL EXAMPLE
  illustrative curve: 0.55 at 30 days, 0.82 at 60, 0.96 at 90, 1.00 at 120
  window ending 30 days before the cut: pi_apparent = 0.0009 * 0.55 = 0.000495

INTERPRETATION
  The last month of a training set looks 45 percent cleaner than it is, so any
  trend fitted across the boundary picks up an artefact, not a change in attack.

VALIDITY CONDITION
  Requires a stable chargeback window; a mid-window change to the dispute
  process invalidates every cohort measured across it.
```

### 4.4 Limits and counterexamples

**Amount heterogeneity breaks the constant-threshold rule.** If `a` varies by row, the correct rule uses each row's own amount: `p > f / (f + a_i)`. A single threshold is equivalent to assuming a common amount, and it systematically under-blocks large-value rows and over-blocks small-value rows. The counterexample below shows the size of the distortion without any change in model quality.

```text
FORMULA
  p*(a_i) = f / (f + a_i)
  distortion = p*(a_small) / p*(a_large)

WHERE
  a_small = 20.00, a_large = 900.00, f = 6.00

CONSTRAINT
  both rows share one score and one constant threshold chosen at mean amount.

NUMERICAL EXAMPLE
  p*(20)  = 6 / 26  = 0.230769
  p*(900) = 6 / 906 = 0.006623
  distortion = 0.230769 / 0.006623 = 34.8

INTERPRETATION
  A constant cut-off at the mean-amount value of 0.032258 allows rows that are
  4.9 times the break-even risk when the amount is small, and blocks rows that
  are 4.9 times safer than needed when the amount is large. Both errors are
  produced by the threshold, not by the model.

VALIDITY CONDITION
  The formula itself is exact; what fails is the assumption that one threshold
  can represent a heterogeneous amount distribution.
```

**The counterexample that matters most.** Suppose an event-grain model has 60 percent recall and an entity-grain model has 62 percent recall. It is tempting to conclude the entity model is better. But if the event model blocks 0.9 percent of events while the entity model blocks 3.0 percent of entities, the entity model has purchased its two extra points with roughly 3.3 times the friction per protected unit. Section 5 works the full comparison and Section 8.4 gives the discriminating check that separates a genuinely better model from a more aggressive policy wearing the same metric.

## 5. Worked Numerical Example

The trace below is a constructed authorisation portfolio. Every count is ILLUSTRATIVE; every ratio, product, and identity check is recomputed in this section and re-checked independently at the end.

### 5a The portfolio

| Quantity | Symbol | Value | Derivation |
|---|---|---|---|
| Active cards (entities) | `E` | 250,000 | portfolio statement |
| Authorisation events per month | `N` | 4,000,000 | authorisation log count |
| Mean events per entity | `n̄` | 16.0 | 4,000,000 / 250,000 |
| Fraudulent events | `F_e` | 3,600 | adjudicated ledger |
| Event-grain prevalence | `π_e` | 0.000900 | 3,600 / 4,000,000 |
| Entities with at least one fraud | `F_E` | 2,900 | entity roll-up |
| Entity-grain prevalence | `π_E` | 0.011600 | 2,900 / 250,000 |
| Fraud events per fraudulent entity | `m` | 1.241379 | 3,600 / 2,900 |
| Mean loss per fraudulent event | `a` | $180.00 | ledger average |
| Gross monthly fraud loss | `G` | $648,000 | 3,600 × 180 |
| Mean loss per fraudulent entity | `a_E` | $223.448 | 1.241379 × 180 |

The row that matters is `a_E`. A card that is defrauded once in a month loses 223 dollars on average, because a fraction of defrauded cards are hit repeatedly. Reporting a monthly loss of 180 dollars per compromised card understates the portfolio by 24 percent, and that understatement compounds once the number is annualised.

### 5b Grain transform, computed twice

```text
ROUTE 1 (definition of m)
  pi_E = F_E / E = 2,900 / 250,000 = 0.011600

ROUTE 2 (identity)
  pi_e * n_bar = 0.000900 * 16.0 = 0.014400
  0.014400 / 1.241379 = 0.011599

AGREEMENT
  |0.011600 - 0.011599| = 0.000001  (rounding at the sixth decimal only)
```

The two routes agreeing is not a coincidence; the first route is the definition and the second is an algebraic rearrangement. What the check buys is a guarantee that the four quantities `π_e`, `n̄`, `m`, `π_E` were not each estimated from a different window. When the two routes disagree by more than rounding, one of the four inputs is from the wrong population, and Section 8.1 tells you which check isolates it.

### 5c Two policies compared on the same population

**Policy A, event grain.** Block events at a 0.9 percent volume rate; the model catches 60 percent of fraudulent events; friction is 6 dollars per blocked event.

| Line | Arithmetic | Result |
|---|---|---|
| Blocked events | 0.009 × 4,000,000 | 36,000 |
| Fraud events prevented | 0.60 × 3,600 | 2,160 |
| Precision of blocked set | 2,160 / 36,000 | 0.060000 |
| Loss avoided | 2,160 × 180 | $388,800 |
| Friction incurred | 36,000 × 6 | $216,000 |
| Net benefit | 388,800 − 216,000 | $172,800 |

**Policy B, entity grain.** Block the whole card at a 3.0 percent volume rate; the model catches 65 percent of fraudulent cards; friction is 6 dollars per blocked card.

| Line | Arithmetic | Result |
|---|---|---|
| Blocked cards | 0.030 × 250,000 | 7,500 |
| Fraud cards caught | 0.65 × 2,900 | 1,885 |
| Precision of blocked set | 1,885 / 7,500 | 0.251333 |
| Events prevented | 1,885 × 1.241379 | 2,340 |
| Loss avoided | 2,340 × 180 | $421,200 |
| Friction incurred | 7,500 × 6 | $45,000 |
| Net benefit | 421,200 − 45,000 | $376,200 |

Policy B wins by 203,400 dollars a month in this trace, and the reason is arithmetic rather than modelling: the same 6-dollar friction is charged once per card instead of once per blocked event. Entity recall of 65 percent also sits exactly where Section 4.2 predicts, between the lower bound of 60.0 and the independent-miss upper bound of 67.946 percent.

### 5d Sensitivity: when does the ranking flip?

Entity-grain blocking charges friction on 7,500 legitimate cards, and some of those customers leave. Let `q` be the share of blocked legitimate customers who attrite and let lifetime margin per lost customer be $95.00. Then entity net benefit falls by `7,500 × 95 × q`, and the two policies tie when:

```text
SETUP
  entity_net(q) = 421,200 - 7,500 * (6 + 95 * q)
  event_net     = 172,800

SOLVE
  421,200 - 45,000 - 712,500 * q = 172,800
  376,200 - 712,500 * q = 172,800
  712,500 * q = 203,400
  q = 0.285474

CHECK
  entity_net(0.285474) = 376,200 - 712,500 * 0.285474
                       = 376,200 - 203,400 = 172,800   (matches event_net)
```

The crossover is at a 28.5 percent attrition rate among blocked legitimate customers. Below that, entity grain wins; above it, event grain wins. Two lessons follow. First, the grain decision is a customer-value question, not a model question. Second, a team that never measured `q` cannot know which policy it should be running, and no offline AUC will tell it.

The second sensitivity is on `m`. If the same portfolio had `m = 3.0` instead of 1.241, prevalence at entity grain would be `0.0144 / 3.0 = 0.0048`, the independent-miss recall bound would rise to `1 − 0.4³ = 0.936`, and the friction advantage of entity grain would grow because each blocked card now prevents three events rather than 1.24. Grain choice is therefore sensitive to a quantity that most teams never estimate.

### 5e What the example does not prove

The trace uses a constant loss per fraud event. Section 4.4 shows that a constant threshold is wrong when amounts vary, and this portfolio certainly has varying amounts, so the real optimum would use amount-conditional boundaries. The trace also assumes the two policies see the same population, which they do not: once policy B blocks a card, its later events never reach the authorisation stream, so the observed event rate under policy B falls for reasons unrelated to fraud. Both caveats are deliberate teaching points, and Section 8 lists the checks that detect them.

## 6. Visual / Geometric Intuition

**Two clocks on one axis.** Draw a horizontal line for one event. The event sits at `t_e`. A vertical bar at `t_d` marks the deadline: everything to the right of the bar is invisible to the decision. A second bar at `t_m` marks maturity. The region between `t_d` and `t_m` is where the label exists but is unrecorded, and the area of that region is the censoring exposure.

```text
t_e            t_d                    t_m
 |--------------|----------------------|
    features        censored window     label final
    available       (outcome unknown)   (outcome known)
```

The decision deadline is often much closer to `t_e` than practitioners assume. In authorisation it is measured in hundreds of milliseconds; in claims it is days; in AML alerting it is weeks. The same feature that is legitimate at a 5-day deadline may be unavailable at a 400-millisecond deadline, which is why the grain declaration must be paired with a latency budget.

**Rung aggregation as a rectangle.** Picture 16 event cells in a row, one per card-month. Event-grain policies colour individual cells; entity-grain policies colour all 16 cells at once. If four cells are fraudulent, the entity policy is right to colour all 16 only because it cannot see which four. The picture makes the trade obvious: entity grain buys fewer colouring events (friction) and loses finer control (over-blocking).

```text
entity grain:  [######## ########]   1 friction charge, 16 cells affected
event grain:   [.#..#....#......]    3 friction charges, 3 cells blocked
               cells: 16  fraud: 3  entity recall: 1/1  event recall: 3/3
```

**Base-rate ladder.** Each rung multiplies the base rate by roughly the number of chances per unit. Plotting prevalence against rung gives a steep line, and the steepness is the reason a 99.9 percent accurate classifier at event grain can still be useless: at 0.09 percent prevalence, accuracy is dominated by the legitimate majority.

```text
rung            prevalence    rows per 10,000
event             0.0900%           9
entity            1.1600%         116
ring (4 cards)    ~4.5%           ~450
```

**The correlation fan.** The recall bound `1 − (1 − d_e)^m` is an upper bound reached only when misses are independent within an entity. Draw `d_E` as a function of within-entity correlation `ρ` from 0 to 1: it falls monotonically from 68.0 percent to 60.0 percent. A measured value above the curve means the misses are anti-correlated, which usually signals that the model is keyed on an entity-level feature and is therefore not really making event-level decisions at all.

## 7. Comparison / Selection Table

### 7a The five families on the axes that matter

| Family | Decision unit | Deadline | Permitted actions | Label source | Maturity | Loss owner | Cardinality per entity |
|---|---|---|---|---|---|---|---|
| Card authorisation | Authorisation event | < 1 second | Allow, step-up, decline | Chargeback, fraud report | 90–120 days | Issuer P&L | 16–40 events/month |
| Insurance claim | Claim, with line items | 1–10 days | Approve, request evidence, refer to SIU, deny | Adjudication + recovery | 30–180 days | Claims reserve | 1–3 claims/year |
| Account takeover | Login or session | < 1 second to 1 day | Allow, challenge, lock, force reset | Customer report, chargeback | 30–90 days | Deposit or card P&L | 1–8 sessions/day |
| Application / onboarding | Application | 1 minute to 3 days | Approve, manual review, decline, limit | First-party default, bust-out | 90–365 days | Credit loss | 1 |
| AML monitoring | Alert, then case | 1–30 days | Close, escalate, file report | Investigator disposition, later enforcement | 180+ days | Compliance budget | Many transactions, few cases |
| Marketplace abuse | Listing or seller | Hours to days | Delist, restrict, hold funds, ban | Buyer dispute, chargeback | 30–90 days | Marketplace goodwill | Many listings per seller |

Reading across a row tells you what the metric must be. Reading down a column shows which families can share plumbing: the deadline column splits the table into sub-second, day-scale, and week-scale systems, and no feature store design spans all three comfortably.

### 7b Metric transfers that always break

| Metric borrowed | From | Why it fails at the destination | Use instead |
|---|---|---|---|
| Event recall | Authorisation | A card with 16 events is protected when any one is caught; event recall understates customer protection | Entity recall with the bound from 4.2 |
| Alert precision | AML | Precision at alert grain is diluted by duplicate alerts on one entity | Deduplicated precision per entity |
| AUC | Any | Invariant to prevalence and to the loss ratio, so it cannot rank two policies with different friction | Expected loss per 10,000 decisions |
| Approval rate | Onboarding | Higher approval looks better until the bust-out cohort matures a year later | Vintage-level loss per approved account |
| Case closure rate | AML | Closing cases fast is compatible with closing them wrongly | Confirmed-fraud yield per investigator hour |
| Chargeback rate | Card | Only reflects fraud above the dispute threshold; small fraud is invisible | Fraud rate joined to the unreported-loss estimate |

### 7c Choosing a grain when two are defensible

| Condition | Prefer event grain | Prefer entity grain |
|---|---|---|
| Friction charged per action | Low relative to fraud loss | High, or charged per customer contact |
| Multiplicity `m` | Near 1.0 | Well above 1.0 |
| Attrition from a wrong block | High | Low |
| Latency budget | Sub-second, event-level features only | Batch, entity history available |
| Loss owner | Books per event | Books per customer or account |
| Regulatory visibility | Per-transaction reporting required | Per-customer risk rating required |

The table is not a scoring rubric; it is a list of the six facts you must establish before defending either choice. Two of them (`m` and attrition) are usually unmeasured, which is why the honest answer is often "we do not yet know, and here is the experiment that would tell us".

## 8. Failure Modes & Edge Cases

**8.1 Prevalence mismatch across grains.** *Symptom:* the finance loss rate and the model's training prevalence differ by an order of magnitude and nobody can reconcile them. *Competing causes:* (a) one number is entity grain and the other event grain; (b) one counts fraud events, the other counts fraudulent currency; (c) one includes recovered funds. *Discriminating check:* recompute the ratio `π_E / π_e` and compare it with `n̄ / m` from the same window. If the ratio matches, the disagreement is grain and is not an error. If it does not, compute both numbers on a 30-day cohort and compare again.

**8.2 Recall quoted at the wrong rung.** *Symptom:* a vendor claims 60 percent detection, operations observes that most compromised customers are still hit. *Competing causes:* (a) event-grain recall presented as customer protection; (b) recall measured on reviewed rows only; (c) recall measured before block-induced selection removed the easy cases. *Discriminating check:* recompute recall on an entity roll-up and compare against the 4.2 bounds. A value below the lower bound means the metric was not event recall at all; a value above the upper bound means the entity key is leaking label information.

**8.3 Silence treated as legitimate.** *Symptom:* a new rule shows spectacular precision in backtest and mediocre precision in production. *Competing causes:* (a) unlabelled fraud was scored as negative; (b) the rule targets a segment with low dispute propensity; (c) the rule was trained on rows that had already been reviewed. *Discriminating check:* restrict evaluation to rows with a mature, adjudicated disposition and compare. If precision falls by more than the label coverage gap, cause (a) dominates.

**8.4 Aggressiveness mistaken for skill.** *Symptom:* a new policy raises recall and raises total loss. *Competing causes:* (a) friction outran the prevented loss; (b) the policy shifted fraud to an unmonitored channel; (c) true prevalence rose independently. *Discriminating check:* compute net benefit per 10,000 decisions at fixed volume for both policies, then re-run at fixed precision. A policy that wins at fixed volume and loses at fixed precision is buying recall with volume.

**8.5 Censoring at the training boundary.** *Symptom:* the last two weeks of data look unusually clean and the model under-predicts recent fraud. *Competing causes:* (a) label immaturity; (b) a genuine decline in attack volume; (c) a change in the dispute process. *Discriminating check:* plot label-arrival curves for two cohorts 90 days apart. A parallel shift means process change, a change in level only in the recent cohort means censoring, and a drop across all cohorts means a real decline.

**8.6 Entity resolution merging distinct actors.** *Symptom:* blocking one card also freezes an unrelated customer's account. *Competing causes:* (a) shared-device linkage merged a household; (b) a merchant terminal ID was used as an entity key; (c) deliberate shared infrastructure by an attacker to poison the graph. *Discriminating check:* audit the largest clusters by fan-out. Clusters of size 2–5 with distinct surnames and distinct payment instruments are household merges; single large clusters with fresh devices are attack-driven poisoning.

**8.7 Action changes the observed outcome.** *Symptom:* declining a segment improves its observed fraud rate, and the improvement is cited as evidence the decline worked. *Competing causes:* (a) genuinely prevented fraud; (b) displaced fraud that reappears elsewhere; (c) the decline removed the only channel through which the fraud was observable. *Discriminating check:* track the segment's fraud rate across a control holdout that continues to allow a small share. Only the holdout separates prevention from displacement.

**8.8 Duplicate alerts inflating workload metrics.** *Symptom:* analyst throughput looks adequate while the backlog grows. *Competing causes:* (a) duplicate alerts on open cases; (b) alerts requiring evidence not yet available; (c) genuinely rising attack volume. *Discriminating check:* compute the deduplicated alert count by entity and the median alerts per open case. A dedup share above one third means the queue metric is measuring generation, not work.

**8.9 Maturity assumption inside the label.** *Symptom:* precision looks excellent on claims decided within 48 hours. *Competing causes:* (a) fast decisions are easy decisions; (b) the fast-decision cohort has a different fraud mix; (c) investigators close thin cases early with a benign disposition. *Discriminating check:* compare the 48-hour cohort's later reopening rate with the slow cohort's. A reopening rate above 5 percent means the fast label is not the label.

**8.10 Grain drift after a product change.** *Symptom:* a stable model degrades after a wallet or tokenisation launch. *Competing causes:* (a) the event count per entity changed, so prevalence at entity grain moved; (b) the fraud mechanism moved to a channel with no labels; (c) a feature lost availability at the new deadline. *Discriminating check:* recompute `n̄` before and after. If `n̄` moved by more than 20 percent, grain drift alone explains part of the degradation and the metric requires rebasing before any model change.

## 9. Complexity & Cost Analysis

**Cost of the grain decision.** The dominant term is friction multiplicity. Event grain charges `f` once per blocked event; entity grain charges `f` once per blocked entity; ring grain charges once per cluster. If `n̄ = 16`, event grain can incur up to sixteen times the friction of entity grain for the same blocked population, which is exactly the 216,000 versus 45,000 difference in Section 5c.

**Cost of entity resolution.** Exact-key joins (card number, policy number) are linear in the event count: 4,000,000 rows sorted and grouped once. Linkage-based resolution is quadratic in the worst case: full pairwise comparison over 250,000 entities is `250,000 × 249,999 / 2 = 31,249,875,000` pairs, which is not computable in a monthly batch at reasonable cost. Blocking on a shared device key reduces the candidate set to the sum of squared block sizes:

```text
pairs = sum over keys k of c(k) * (c(k) - 1) / 2
      = 5,000 * (6 * 5 / 2) + 85,000 * 0
      = 5,000 * 15 = 75,000 candidate pairs per month
```

The ratio between the two figures is 416,665 to one. That single computation is why every practical fraud graph starts with a blocking key and why the choice of blocking key is a modelling decision with direct cost consequences.

**Cost of label latency.** A label that arrives at day 120 cannot supervise a model deployed at day 1. The usable training window ends at `cut − 120 days`, so a team that wants a monthly refresh with a full 12-month history is really training on months 13 to 24 before the deployment month. Shortening the wait requires either a faster label path or an explicit weak-label model with its own error budget, and neither is free.

**Cost of review capacity.** Manual review is the most expensive action in the table and the one most often sized by headcount rather than by marginal value. Section 4 of the companion sheet derives the shadow price of a review slot; at a 25 dollar hourly cost and 20 reviews per hour, one review costs 1.25 dollars, so any queue item whose expected net benefit exceeds that is worth staffing.

**Cost of metric churn.** A metric definition change forces a full historical restatement before the trend line is usable. Changing an entity definition after 18 months of reporting means either restating 18 months or living with a discontinuity, and teams routinely choose the discontinuity without recording it. The cost is not computational; it is the loss of the ability to detect a real change against a consistent baseline.

**Where the cost is hidden.** Friction is booked in customer-service budgets, while fraud loss is booked in the fraud budget. A policy that moves cost from one to the other looks free to each owner separately. Any net-benefit calculation that reads only one ledger will therefore prefer the policy that exports its cost to the other ledger, which is the single most common way a fraud business case overstates its value.

## 10. Key Theorems / Results Reference

**R1 — Break-even identity.** For one row with linear losses, blocking has lower expected cost than allowing exactly when `p > f / (f + a)`. *Condition:* single loss owner, no capacity limit, constant `a` and `f`. *Why it matters:* every threshold argument is this identity plus an argument about what `a` and `f` are.

**R2 — Prevalence transform.** `π_E = π_e · n̄ / m` when both counts use the same window and `m > 0`. *Condition:* one entity per event in the roll-up. *Why it matters:* it is the only defensible bridge between the two most reported fraud rates.

**R3 — Recall bounds under aggregation.** `d_e ≤ d_E ≤ 1 − (1 − d_e)^m`. *Condition:* entity is flagged if any event is flagged; `m ≥ 1`. *Why it matters:* it tells you whether an observed lift in recall is arithmetic or model improvement.

**R4 — Threshold invariance in probability space.** Under a fixed loss ratio `f / a`, the optimal threshold `p*` does not depend on prevalence. *Condition:* `p` is calibrated and the action set is unchanged. *Why it matters:* under a prevalence shift the threshold should not move, but the flagged volume will, which is why capacity binds exactly when fraud rises.

**R5 — Threshold non-invariance in quantile space.** A fixed score quantile maps to a prevalence-dependent threshold. *Condition:* score distribution shifts with the population. *Why it matters:* "we always take the top 1 percent" is a capacity policy, not a risk policy, and it silently changes the loss ratio when prevalence moves.

**R6 — Censoring bias direction.** For any window ending closer than `t_m` to the cut, the apparent fraud rate is biased downward by the uncovered share. *Condition:* labels arrive after the event, never before. *Why it matters:* the bias has a known sign, so it can be corrected rather than merely suspected.

**R7 — Label-path inflation.** `π_true = π_obs / κ` when label generation is independent of features. *Condition:* constant `κ`. *Why it matters:* it converts a "confirmed cases only" rate into a population rate and quantifies how much the confirmed-only view understates.

**R8 — Selection-path bias in precision.** Precision measured on reviewed rows equals population precision only when review selection is independent of the outcome given the score. *Condition:* conditional independence. *Why it matters:* it explains why backtest precision on reviewed cases exceeds production precision, always in the same direction.

**R9 — Friction multiplicity.** The friction ratio between event and entity grain equals the mean number of events per blocked entity, `n̄`. *Condition:* one friction charge per action. *Why it matters:* it is the entire quantitative content of the grain decision on the cost side.

**R10 — Observable-share Goodhart.** If only a share `s` of losses is observable and the policy optimises the observable part, effort shifts toward the observable mechanisms until marginal returns equalise, leaving total loss possibly unchanged. *Condition:* attacker or staff response to policy. *Why it matters:* it predicts that a successful optimisation can improve the metric and not the loss.

## 11. Interview / Exam Questions

**Q1. A team reports 0.09 percent fraud prevalence and a 1.2 percent chargeback rate for the same portfolio. Which is wrong?** Neither, necessarily. The 0.09 percent is event grain and the 1.2 percent is a value-weighted or entity-grain rate. Convert with `π_E = π_e n̄ / m`; at `n̄ = 16` and `m = 1.24` the entity rate is 11.6 per thousand events-equivalent. Then check whether the chargeback rate counts disputes or confirmed frauds, because disputes run higher than confirmations by the reversal share.

**Q2. Given `d_e = 0.55` and `m = 2`, what is the best possible entity-level detection rate and what assumption makes it unattainable?** `1 − 0.45² = 0.7975`. It requires detection to be independent across the entity's events. Real events on one entity share device, merchant, and amount features, so misses correlate positively and the achieved rate lands between 0.55 and 0.7975.

**Q3. Your precision is 22 percent at case grain and 0.4 percent at alert grain. Which number goes in the business case?** Both, labelled. The case-grain number describes analyst output; the alert-grain number describes queue load and the cost of generating work. A business case that quotes only the flattering number is unable to size the analyst team.

**Q4. Why can an entity-grain policy that blocks 3 percent of customers be cheaper than an event-grain policy that blocks 0.9 percent of transactions?** Because friction is charged per action. At `n̄ = 16`, the event policy pays friction on roughly 0.9 percent of 16 events per customer, about 14 percent of customers contacted per month, versus 3 percent under entity grain. The comparison is only valid if attrition from over-blocking is included, which is the 28.5 percent crossover in Section 5d.

**Q5. A rule shows 80 percent precision on investigated cases and 12 percent precision in production. Diagnose.** Investigated cases were selected by the previous rule, so the row set is conditioned on a high score. Recompute on all rows in the population and compare. If the drop matches the review selection rate, this is R8 and not a modelling defect.

**Q6. What single field must every fraud training row carry, and why?** The selection path: scored-allowed, scored-blocked, or never scored. Without it, blocked rows enter training as legitimate, so the model learns that the pattern it just blocked is benign, and the resulting scores drift downward on exactly the traffic the policy is trying to catch.

**Q7. You are asked to detect fraud on a payments app with a 400-millisecond deadline. Which rung is available?** Event or session grain only. Entity history can be served from a low-latency store, but any feature requiring a join to claims or case data is unavailable at that deadline. Choosing a higher rung changes the deadline, so the rung and the latency budget must be decided together.

**Q8. Compute the entity prevalence if 900 fraud events are spread over 700 cards in a base of 120,000 cards, and state what is missing.** `m = 900/700 = 1.2857`; the entity rate is `700/120,000 = 0.005833`. What is missing is `n̄`, the mean events per card, which is needed only for the transform check, not for the direct computation. The direct computation is why the definition route should be preferred whenever the roll-up exists.

**Q9. An AML unit reports that 96 percent of alerts are closed as false positives and claims its model is therefore poor. What is wrong with the claim?** Alert-grain precision is structurally low because one genuine case generates dozens of alerts. The meaningful figure is confirmed cases per 100 cases opened, or currency confirmed per analyst hour. A 4 percent alert precision alongside 20 percent case precision is a normal ratio, not a broken model.

**Q10. Distinguish censoring from a real decline in fraud using two numbers.** Plot the label-arrival curve for two cohorts 90 days apart. Censoring shifts only the recent portion of the curve while the mature portion is unchanged; a real decline lowers the mature plateau. Report both the plateau and the age at which it is reached.

**Q11. Why does an optimal probability threshold not move when prevalence doubles?** Because the threshold compares the cost of two errors on the same row, and both costs are unconditional on prevalence. What does move is the number of rows above the threshold. This is why a fixed capacity queue and a fixed probability threshold cannot both be held constant during a prevalence shift.

**Q12. A stakeholder asks for "the fraud rate". Give the four answers you must offer.** Event prevalence, entity prevalence, value-weighted loss rate, and confirmed-versus-suspected basis. State the window and the label maturity for each. If pressed for one number, give loss per 10,000 decisions because it is the only one that carries the action's cost.

**Q13. Your entity-graph links 40 cards through one device and the analyst wants all 40 frozen. What do you check first?** Cluster composition: distinct surnames, distinct funding instruments, account ages, and whether the device appeared before or after the accounts. A household shares a device for years; an attack cluster shares one for weeks and links to accounts opened in a burst.

**Q14. What makes a metric "transferable" between two fraud families?** Same grain, same action set, same label path, and same loss owner. In practice that means almost nothing transfers except the loss arithmetic. The taxonomy in Section 7 exists so that a borrowed metric can be named, checked against those four conditions, and rejected explicitly rather than silently.

## 12. Quick Reference Card

| Item | Rule | Check |
|---|---|---|
| Grain | Declare event, entity, case, or ring before any metric | One row equals one decision |
| Base rate | `π_E = π_e n̄ / m` | Compute both routes; agree to rounding |
| Recall | `d_e ≤ d_E ≤ 1 − (1 − d_e)^m` | Measured value must sit inside the bounds |
| Break-even | `p* = f / (f + a)` | Recompute with actual friction, not a guess |
| Friction | Charged per action, so grain sets the multiplier | Compare `n̄` before comparing policies |
| Labels | `π_true = π_obs / κ` | Estimate `κ` on a labelled cohort |
| Censoring | Apparent rate falls by uncovered share | Plot label-arrival curves by cohort |
| Selection | Record allowed, blocked, never scored | Recompute precision on all rows |
| Unit | Attach grain to every rate in every document | A number without a grain is not a metric |
| Transfer | Same grain, actions, labels, loss owner | Otherwise name the mismatch and refuse |
