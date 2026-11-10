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

**How the sections fit together.** Sections 2 and 3 establish vocabulary and the ladder. Section 4 supplies the arithmetic that converts the vocabulary into numbers, and its four subsections move from a single row (4.1) through grain conversion (4.2) to alert-grain and label variants (4.3) before closing on the cases where the arithmetic stops applying (4.4). Section 5 runs two complete traces so the arithmetic can be seen operating on realistic counts, and its final subsection re-derives the load-bearing numbers by an independent route. Sections 6 and 7 turn the numbers into pictures and selection rules. Section 8 collects the failure mechanisms with discriminating checks, Section 9 costs them, and Section 10 states the results compactly enough to cite in review.

**What "declared" means in practice.** A declaration is a written artefact, not a conversation. It names the entity key, the window, the deadline in milliseconds or days, the action vocabulary, the label source with its maturity, and the loss owner with a budget line. It is versioned, and a change to any field is a change of decision contract rather than a model refresh. Teams that keep this artefact discover that most "model bugs" were actually unannounced contract changes, and they can prove it by comparing versions.

**A note on what counts as evidence here.** Every formula in Section 4 is an identity or a dominance argument that can be checked with arithmetic alone. Every input is constructed. Nothing in this sheet is a measurement of a real portfolio, and no number here should be used as a benchmark. The value of the sheet is the machinery, not the values: substitute your own counts into the same blocks and the checks will still hold or still fail for the same reasons.

**One sentence to take to a design review.** A fraud model's output is a number; the grain decides what that number is about; the clock decides which rows could have contributed to it; the action decides what the business can do with it; and the label decides whether anyone will ever know if it was right.

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

**Why five rungs rather than two.** A two-rung model (event and entity) is enough for card fraud but breaks on two other families. Insurance claims need a case rung because one claim can carry twelve line items with independent decisions, so the case is the unit that a handler works and the line is the unit that a rule scores. AML needs an alert rung above the case rung because one case legitimately generates dozens of alerts, and reporting precision at alert grain makes a sound system look broken. Marketplace abuse needs a ring rung because a seller network can be fraudulent as a network while every individual listing looks ordinary. Naming the rung at the start prevents the most common cross-team argument, which is two teams reporting different precision numbers for the same system and both being right.

**The counterfactual column that most tables omit.** For each rung, write down what would have happened had the system not acted. At event grain the counterfactual is usually "the transaction would have completed", which is checkable against chargeback data. At entity grain the counterfactual is "the customer would have kept transacting normally", which is not checkable, because blocking changes the customer's subsequent behaviour. The taxonomy therefore distinguishes rungs where the counterfactual is observable from rungs where it must be assumed, and Section 8.7 gives the holdout design that recovers an estimate.

**Reading an unfamiliar workflow in ninety seconds.** Ask three questions in order. How long does the system have between the event and the forced action? What is the strongest action available, and who is harmed when it is wrong? When does the outcome become trustworthy, and through which process? The answers place the workflow on the ladder, in the action table, and on the clock axis respectively. If the three answers do not point to the same rung, the workflow is genuinely multi-stage and must be modelled as two decisions rather than one.

**Multi-stage workflows are the normal case, not the exception.** An onboarding decision and a first-transaction decision are different rungs with different labels, and a system that merges them produces a model whose score means different things for different rows. The correct pattern is a chain: the onboarding decision uses application evidence and its own label, the early-transaction decision uses onboarding output plus behaviour, and each stage carries its own metric contract. Merging stages is the second-most-common grain failure after event-versus-entity confusion, and unlike that one it is invisible in aggregate metrics because the merged model still ranks acceptably.

**What an entity actually is depends on policy, not physics.** A device fingerprint that links a household of four produces one entity where the business believes it has four customers. A shared payment instrument in a marketplace links a legitimate dropshipper to fifty unrelated sellers. Entity resolution is therefore a governance decision with a documented owner, and the taxonomy entry for it belongs next to the action set rather than in a feature dictionary. Section 8.6 lists the checks that separate benign over-merging from adversarial graph poisoning.

**Two clocks are not enough for claims.** A claim has an event time (the incident), a submission time (the form arrived), and a decision time (the handler must decide). These can be weeks apart, and a model that uses submission-time features while being evaluated on incident-time cohorts will appear to work and then fail on a seasonality boundary. The rule is mechanical: every feature carries the timestamp at which it became true, and every label carries the timestamp at which it became trustworthy. A dataset is only usable when those two columns exist for every row.

**The attacker is inside the loop.** Ordinary supervised learning assumes the input distribution is independent of the deployed model. Fraud violates this in the strongest possible way: a successful model changes what the attacker does next, and the change is directed at the model's weaknesses rather than at the population's average. Two practical consequences follow. First, every metric computed on historical data describes a policy that is no longer running, so the gap between backtest and production is structural rather than accidental. Second, the taxonomy must record the *observability* of each mechanism, because a mechanism that has moved out of the monitored channel disappears from the label set while the loss continues.

**Why the ladder has a ring rung at all.** Below the ring rung, the unit of loss is an entity and the unit of decision is an entity. A ring is a set whose members coordinate, so the loss of the set exceeds the sum of the losses its members would produce independently, because coordination converts independent small legitimate-looking events into a single large extraction. The ring rung exists to capture that super-additivity, and it is the only rung where the loss is not the sum of per-member losses.

**What "declared population" prevents.** Without a declared population, precision can be improved indefinitely by narrowing the denominator to rows the model already handles well. Every fraud metric in existence can be gamed this way, and the gaming is usually not malicious: teams naturally report on the segment they understand. Declaring the population in the same sentence as the number removes the option, because the reviewer can then ask whether the segment is the one the business cares about.

**A note on entity-level labels.** An entity-grain label is usually "at least one confirmed fraud event on this entity in the window". That label is a disjunction, so an entity with one confirmed event among forty is labelled identically to an entity with forty confirmed events. Any model trained on the disjunction will predict "will be defrauded at all", which is a different question from "how much will be lost". The two questions justify different actions: the first justifies a block, the second justifies a limit. Modelling a disjunction and acting as though a magnitude were predicted is a category error that shows up as over-blocking of low-value entities.

**The maintenance consequence of all this.** Because the grain, the clock, and the action are all mutable, the taxonomy is a living document with a version, not a memo. Each version should change at most one axis, so that a downstream metric change can be attributed. A taxonomy that changes entity definition and action vocabulary in the same release makes every subsequent trend comparison uninterpretable, and the cost lands months later when someone tries to explain a level shift to a risk committee.

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

**Notes on the definitions that cause the most argument.**

*Decision grain versus training-table grain.* The training table can be at any grain as long as the label, the features, and the decision all respect the declared decision grain. A common and defensible pattern is to train at event grain and score at entity grain by aggregating event scores. That pattern is only valid if the aggregation rule is declared: maximum, sum, or share-above-cut. The three rules produce materially different entity scores from identical event scores, and switching between them silently changes the operating point.

*Case versus alert.* An alert is a request for attention; a case is a container that a person opens. In AML a single case may absorb fifty alerts, so the alert-to-case ratio is a workload metric and the case-to-confirmation ratio is a quality metric. Reporting one as the other is the standard error behind the "96 percent of alerts are false" complaint in Q9 of Section 11.

*Chargeback versus confirmed fraud.* A chargeback is a completed dispute. Confirmed fraud is an adjudicated outcome. Their ratio, the confirmation rate, is itself a measured quantity and it drifts whenever the issuer changes its dispute handling. A team that trains on chargebacks alone inherits dispute-handling policy as if it were ground truth.

*Prevalence versus incidence.* Prevalence is a share of rows in a period; incidence is a rate of new fraudulent entities per unit time. Prevalence is what a classifier sees; incidence is what the risk committee manages. Confusing them makes a portfolio look stable while incidence rises.

*Friction versus loss.* Friction is charged whenever a legitimate item is inconvenienced. Some firms book it as a customer-service cost, some as churn risk, some not at all. If it is not booked, the break-even threshold collapses to zero and the policy becomes "block everything suspicious", which is a governance failure expressed as an arithmetic one.

*Selection path versus label.* Selection path records what the system did; the label records what the outcome was. A blocked row can still acquire a fraud label if the actor tries elsewhere and is caught, and an allowed row can acquire a legitimate label even if the customer was in fact defrauded and never noticed. The two columns are independent and both are required.

*Weak label versus negative.* A weak label is a proxy with a stated error profile, such as "no dispute within 120 days". A negative is a claim that the row was legitimate. Treating a weak label as a negative is the assumption in Section 3c labelled silence-as-legitimate, and it is the mechanism behind the backtest-to-production precision gap.

*Entity resolution versus graph feature.* Resolution decides identity; the graph feature measures proximity. Building a proximity feature on an unresolved key produces clusters that look like fraud rings because they are artefacts of a shared corporate card or a merchant terminal. Resolution must be settled first, and its version must be stored with the feature values.

*Loss owner versus model owner.* They differ almost always. The model owner is measured on offline metrics, the loss owner on the P&L. Any business case must name both and must resolve conflicting incentives explicitly, because a model owner rewarded on recall will prefer a policy the loss owner pays for.

*Entity identity as a versioned decision.* The set of rows that share an entity key changes whenever the resolution policy changes, so an entity key is not a property of the data but an output of a process. Store the policy version next to the key and refuse to compare two periods that used different versions without restating one of them. The failure this prevents is subtle: a model retrained after a resolution change appears to have improved, when in fact its training labels were reorganised into fewer, larger entities.

*Case state as a label hazard.* A case has a state machine (open, pending evidence, closed-fraud, closed-legitimate, reopened). If labels are derived from case state at extraction time, then cases still open are neither positive nor negative and are usually dropped, which biases the training set toward fast-resolving cases. The bias is systematic and points in a fixed direction: fast-resolving cases are the easy ones, so the model is trained on an easier problem than it faces.

*Threshold as a policy object.* A threshold belongs with the action set, the loss schedule, and the capacity limit, not with the model artefact. When a threshold is versioned inside the model binary, any loss-schedule change requires a model release, and teams respond by hard-coding the threshold in the serving layer, where it becomes invisible to audit. Keep the threshold in the policy layer with its own version and change log.

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

**Three actions, two boundaries.** With allow, challenge, and block available, the optimum is a partition of the score axis into three regions rather than a threshold. The lower boundary separates allow from challenge, the upper boundary separates challenge from block, and both are ratios of cost differences. The middle region exists because a challenge is cheaper than a block when the row is legitimate and cheaper than an allow when the row is fraudulent, so it dominates both neighbours in an interval around moderate risk.

```text
FORMULA
  c  = loss if a fraudulent row passes a challenge (p * a_c)
  k  = friction if a legitimate row is challenged (1 - p) * f_c
  challenge beats allow when  p * a_c + (1-p) * f_c  <  p * a
  challenge beats block   when  p * a_c + (1-p) * f_c  <  (1-p) * f_b
  -> tau_low  = f_c / (a - a_c + f_c)
     tau_high = (f_b - f_c) / (a_c + f_b - f_c)

WHERE
  a    loss on an allowed fraudulent row         currency/row
  a_c  residual loss after a challenge on fraud   currency/row, a_c < a
  f_c  friction when a legitimate row is challenged  currency/row
  f_b  friction when a legitimate row is blocked  currency/row
  tau_low, tau_high boundaries of the challenge band  [0,1]

CONSTRAINT
  tau_low < tau_high; a_c < a; f_c < f_b; challenges resolve instantly.

NUMERICAL EXAMPLE
  a = 180.00, a_c = 54.00, f_c = 1.80, f_b = 6.00
  tau_low  = 1.80 / (180.00 - 54.00 + 1.80) = 1.80 / 127.80 = 0.014085
  tau_high = (6.00 - 1.80) / (54.00 + 6.00 - 1.80) = 4.20 / 58.20 = 0.072165

INTERPRETATION
  The challenge band spans 1.41 to 7.22 percent posterior risk, which is
  5.1 times wider than the single block boundary at 3.23 percent. Adding an
  action widens the region where friction is spent proportionally.

VALIDITY CONDITION
  Fails when the challenge is not resolved, because an unresolved challenge is
  an allow with extra steps and a_c rises toward a while the band narrows.
```

The comparison with the single-boundary case is instructive. The block boundary alone was 3.23 percent; the challenge band's lower edge is 1.41 percent, so the challenge absorbs rows that a single-threshold policy would have allowed. That is the arithmetic behind step-up authentication: it is not a weaker version of a decline, it is a distinct region of the risk axis that a one-threshold policy cannot express.

**Why the block boundary moved.** With an intervening challenge available, blocking must clear a higher bar, because the block now competes against a cheaper action that already captures most of the prevention. The 7.22 percent upper boundary is roughly 2.2 times the single-action 3.23 percent, and the multiple is set by how much loss a challenge recovers. If `a_c` were 90.00 instead of 54.00, the upper boundary would rise to `4.20 / (90.00 + 4.20) = 0.044586`, still above 3.23 percent but lower, because a weak challenge leaves more loss for the block to prevent.

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

**Capacity turns a threshold into a knapsack.** When review slots are scarce, the correct rule is not "refer everything above `p*`" but "rank by expected net benefit per unit of review time and take items until the slots run out". The two rules coincide only when every item costs the same review time. In claims, a suspicious large claim may consume twelve handler hours while a small one consumes one, so the per-item ranking is wrong by an order of magnitude.

```text
FORMULA
  value_i = p_i * a_i - (1 - p_i) * f
  rank by value_i / h_i , accept while cumulative h <= H

WHERE
  value_i  expected net benefit of reviewing item i        currency
  h_i      handler hours to complete item i                hours
  H        total handler hours available in the period     hours

CONSTRAINT
  hours are the binding resource; items are independent; no learning across items.

NUMERICAL EXAMPLE
  item A: p=0.40, a=4,000.00, f=20.00, h=12
    value = 0.40*4,000 - 0.60*20 = 1,600.00 - 12.00 = 1,588.00
    density = 1,588.00 / 12 = 132.33 per hour
  item B: p=0.30, a=400.00, f=20.00, h=1
    value = 0.30*400 - 0.70*20 = 120.00 - 14.00 = 106.00
    density = 106.00 / 1 = 106.00 per hour

INTERPRETATION
  A ranks first on absolute value but second on density. With H = 13 hours the
  optimal allocation is B then A only if 1 + 12 <= 13, which holds, so both fit;
  with H = 12 the choice is A alone (1,588) versus B twelve times (1,272), and
  A wins. The ranking rule matters only when the budget cannot hold every item,
  which is the normal case for queues.

VALIDITY CONDITION
  Fails when items interact, for example when two alerts concern one actor and
  reviewing either reveals both, in which case the joint value is not additive.
```

**Calibration failure masquerades as a threshold error.** When predicted probabilities are systematically high by a factor of two, the threshold appears to be set too low because too many rows cross it. The correct repair is recalibration, not a threshold move, and the diagnostic is a reliability plot on a mature cohort. A useful single number is the ratio of mean predicted to observed rate: 1.9 means the model over-predicts by 90 percent, and every threshold in probability space is then wrong by that factor.

```text
FORMULA
  miscal = mean(p_hat) / observed_rate
  p_corrected = p_hat / miscal

WHERE
  p_hat           mean predicted probability over the cohort   [0,1]
  observed_rate   mature outcome frequency over the same rows   [0,1]
  miscal          multiplicative miscalibration, dimensionless  >0

CONSTRAINT
  cohort is mature and representative of the scored population; the error is
  modelled as multiplicative rather than additive.

NUMERICAL EXAMPLE
  mean(p_hat) = 0.0210, observed_rate = 0.0110
  miscal = 0.0210 / 0.0110 = 1.909091
  p_corrected at p_hat = 0.05  ->  0.05 / 1.909091 = 0.026190
  p* = 6/186 = 0.032258  ->  raw 0.05 crosses, corrected 0.0262 does not

INTERPRETATION
  A row scored 5.0 percent is really 2.6 percent and is below the 3.23 percent
  break-even. The threshold was never wrong; the probability scale was.

VALIDITY CONDITION
  Multiplicative correction fails when miscalibration is concentrated in one
  score band, which is the usual shape after a prevalence shift.
```

**Class-balanced training destroys the probability scale but not the ranking.** Training on 50/50 rebalanced data multiplies the predicted odds by the inverse of the sampling ratio. The model then ranks correctly and calibrates badly. The repair is an explicit prior correction rather than reweighting at scoring time, because reweighting at scoring time cannot recover the original base rate when the sampler dropped rows without a record.

```text
FORMULA
  logit(p_true) = logit(p_sampled) - log(w1 / w0)

WHERE
  w1  sampling rate applied to fraudulent rows          [0,1]
  w0  sampling rate applied to legitimate rows          [0,1]

CONSTRAINT
  sampling is independent of features given the label; both rates are recorded.

NUMERICAL EXAMPLE
  w1 = 0.40, w0 = 0.004
  log(w1/w0) = log(100.0) = 4.605170
  p_sampled = 0.500  ->  logit = 0
  logit_true = 0 - 4.605170 = -4.605170
  p_true = 1 / (1 + exp(4.605170)) = 1/101 = 0.009901

INTERPRETATION
  A row the model calls a coin flip is really 0.99 percent. Without the
  correction every downstream threshold is applied at roughly fifty times
  the intended risk.

VALIDITY CONDITION
  Fails when the sampling rates were not stored; then only ranking transfers
  and thresholds must be re-fitted from a fresh unbiased slice.
```

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

### 5f A second trace at alert grain

The card trace hides the alert-rung problem because one authorisation generates at most one alert. The AML trace below makes the rung distinction unavoidable. All counts are ILLUSTRATIVE.

| Quantity | Symbol | Value |
|---|---|---|
| Monitoring rules in production | — | 24 |
| Transaction rows screened per month | — | 9,000,000 |
| Alerts raised | `A` | 60,000 |
| Alert-to-case conversion | — | 2.0 percent |
| Cases opened | `C` | 1,200 |
| Confirmed cases | — | 240 |
| Confirmed case precision | — | 0.200000 |
| Currency confirmed per case | — | $14,500 |
| Currency confirmed total | — | $3,480,000 |
| Alerts per case | `A/C` | 50.0 |
| Duplicate share | — | 0.34 |
| Analyst hours per case | — | 6.0 |
| Total analyst hours | — | 7,200 |
| Loaded analyst cost per hour | — | $55.00 |
| Review cost | — | $396,000 |
| Cost per confirmed case | — | $1,650.00 |
| Recovery ratio | — | 3,480,000 / 396,000 = 8.788 |

The recovery ratio of 8.79 is the only number that decides whether the programme is worth running, and it is a ratio between a currency figure and a labour figure. That comparison is legitimate only because both are in the same currency and the same month; if the programme were judged on alert precision (0.4 percent) it would look catastrophic, and if judged on case precision (20 percent) it would look respectable. Neither figure alone supports a decision.

**Where the hours actually go.** Of the 6.0 hours per case, a constructed split is 2.5 hours on evidence assembly, 1.5 on disposition writing, 1.2 on escalation, and 0.8 on waiting for counterparties. Only 0.8 hours is decision-making. This matters because the standard productivity lever — a better model — cannot touch 4.4 of those hours. A ranked queue helps only to the extent that it changes which cases are opened, which is a case-selection lever, not a model lever.

```text
HOURS ACCOUNTING
  per case: 2.5 + 1.5 + 1.2 + 0.8 = 6.0   (checks with the table)
  decision share = 0.8 / 6.0 = 0.133333
  hours recoverable by ranking = 0.8 per case at best
  ceiling on throughput gain = 6.0 / (6.0 - 0.8) = 1.153846
```

A perfect ranking system yields at most a 15.4 percent throughput gain on this cost structure. Any proposal promising to double capacity by improving ranking is arithmetically impossible unless it also removes evidence-assembly or reporting work.

**Deduplication as a workload lever, not a model lever.** With `A/C = 50` and a duplicate share of 0.34, roughly 20,400 alerts per month are repeat hits on an entity already under review. Collapsing them into the open case changes no model output and removes a third of the queue generation, but it does change the alert-precision number from 0.4 to 0.66 percent, which is why the metric must be reported deduplicated or the improvement will be invisible.

```text
dedup_alerts = 60,000 * (1 - 0.34) = 39,600
reduction    = 60,000 - 39,600 = 20,400 alerts, i.e. 34.0 percent
new precision= 240 / 39,600 = 0.006061  (case count unchanged)
```

The precision change from 0.004000 to 0.006061 is a 51.5 percent relative improvement produced by suppressing repeats. Reporting the undeduplicated figure while claiming model benefit is the specific error that makes AML business cases untrustworthy.

### 5g Independent arithmetic re-check

Recompute the three load-bearing numbers a different way and confirm agreement.

```text
1) Case precision
   direct:  240 / 1,200 = 0.200000
   via alerts: (60,000 / 50) = 1,200 cases; 240/1,200 = 0.200000   OK

2) Alerts per case from conversion
   conversion 2.0 percent -> 1/0.02 = 50.0 alerts per case
   table value 50.0                                               OK

3) Recovery ratio
   direct: 3,480,000 / 396,000 = 8.787879
   via unit costs: 14,500 / (6.0 * 55.00) = 14,500/330 = 43.939
   NOTE: the unit-cost route gives 43.9 because it applies the confirmed
   currency to every case, not just confirmed ones. Correcting by case
   precision: 43.939 * (240/1,200) = 8.787879                     OK

4) Hours
   1,200 * 6.0 = 7,200;  7,200 * 55.00 = 396,000                   OK
```

The third check is the interesting one. Two routes give 43.9 and 8.79, and the gap is exactly the case precision. Whenever two cost routes disagree by a factor that equals a conversion rate, the likely cause is that one route applied a per-success quantity to all attempts. That diagnostic generalises well beyond fraud work.

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

```text
rho     d_E      comment
0.00   0.6795    independent misses, bound attained
0.25   0.6512    mild clustering, typical of shared device features
0.50   0.6300    shared merchant and amount features
0.75   0.6135    near-duplicate events
1.00   0.6000    entity-level scoring: no gain from aggregation
```

**The friction ledger as a stacked bar.** For every 10,000 decisions, draw a bar whose positive portion is loss avoided and negative portion is friction plus expected attrition. Policy A's bar is thin on both sides because friction is charged often; policy B's bar is thicker on the positive side and thinner on the negative. The point of the picture is that the two bars are not drawn on the same axis unless attrition is included, and attrition is the only term in the chart that requires a measurement nobody has.

**Selection path as a mask.** Draw the full event table as a grid, then mask the rows that were blocked. The masked region is not random: it is concentrated where the previous model scored high. Training on the unmasked grid is training on a population that has been conditioned on the old policy, and the conditioning is strongest exactly where the new model is expected to be most confident.

```text
full table:      [  scored, allowed  ][ scored, blocked ][ never scored ]
                 ~96% of rows           ~3.5%              ~0.5%
label content:   weak labels dominate   suppressed frauds    unknown
usable for
threshold fit:   yes, with correction    no                   no
```

**The metric ladder as a funnel.** Alert to case to confirmation to recovery is a funnel, and each transition loses a known share. Draw it with counts at every level and the two ratios that matter on the sides: conversion and confirmation. Funnels that are drawn without counts invite the mistake of comparing a precision computed at the top with one computed at the bottom.

```text
9,000,000 screened
      |
  60,000 alerts      conversion 0.667% of screened
      |
   1,200 cases       case rate 2.0% of alerts
      |
     240 confirmed    confirmation 20.0% of cases
```

**Base-rate ladder, continued.** The ring rung is where the picture becomes misleading, because a ring of four cards has a prevalence near 4.5 percent only if rings are rare. If rings are common, ring prevalence falls while ring *loss share* rises, because a ring with four cards loses four times as much. Prevalence and loss share therefore move in opposite directions across the top two rungs, and no single number can represent both.

**Where time goes in a sub-second decision.** A 400-millisecond budget typically divides into 40 ms for network and protocol, 90 ms for risk-feature retrieval, 60 ms for model inference, 20 ms for policy evaluation, and 190 ms of headroom. The headroom is not waste; it absorbs tail latency. Any grain choice that requires an extra join costs 40 to 80 ms and therefore consumes most of the headroom, which is the mechanical reason entity-grain features are often served from a precomputed cache rather than joined at decision time.

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

### 7d Same mechanism, different grain

The reason mechanism words cannot drive taxonomy is that identical mechanisms land on different rungs. The table below takes three mechanisms and places them.

| Mechanism | Usual decision unit | Why not the adjacent rung | Characteristic label path |
|---|---|---|---|
| Stolen card number used online | Authorisation event | Entity grain would block the card after the first bad event, too late for that event | Chargeback at 45–90 days |
| Bust-out on a new account | Entity, over a 6–18 month horizon | Event grain sees only small, plausible transactions | First-party default after months of silence |
| Account takeover then immediate transfer | Session, then entity lock | Event grain misses the fact that the session began with a reset | Customer report plus chargeback |
| Collusive claims ring | Ring, above case grain | Case grain decides each claim separately and approves each | Cross-claim anomaly, then enforcement |
| Refund abuse by a serial returner | Entity | Event grain sees each return as ordinary | Merchant-initiated block list |

The middle column is a negative claim — "not the adjacent rung" — and it is the column reviewers should challenge. A taxonomy that cannot say why the neighbouring rung is wrong is a label list, not a decision contract.

### 7e Signals available by rung and deadline

| Rung / deadline | Safe signals | Unsafe signals | Failure if mixed |
|---|---|---|---|
| Event, sub-second | Device, velocity counters, BIN, amount, merchant category | Case dispositions, chargebacks, graph depth > 2 | Label leakage into production |
| Session, sub-second | Event signals plus session age and reset history | Entity lifetime value | Cost-blind decisions that over-block new customers |
| Case, days | Documents, prior claims, investigator notes | Future claims | Hindsight contamination |
| Entity, batch | Aggregates over 30–365 days | Anything computed after the decision | Target leakage with a plausible-looking feature |
| Ring, batch | Linkage graph at the decision date | Later-confirmed links | Graph built with knowledge of the answer |

The right-hand column is the useful one. Every leakage incident observed in practice is a row of this table being violated, and the violation is always detectable by removing the feature and checking whether performance collapses to a level consistent with the population base rate.

### 7f Metric selection by decision, not by tradition

| Decision being made | Primary metric | Secondary guardrail | Metric that will mislead |
|---|---|---|---|
| Set a threshold | Expected net benefit per 10,000 decisions | Blocked volume at fixed precision | AUC, KS |
| Size an analyst team | Confirmed cases per analyst hour | Reopen rate within 30 days | Alerts per analyst |
| Choose a model version | Precision at fixed recall, plus calibration error | Latency p99 | Accuracy |
| Report to a risk committee | Loss per 10,000 decisions, trended | Incidence per 100,000 accounts | Chargeback rate alone |
| Justify an entity resolution change | Cluster audit sample correctness | Number of customer complaints about freezing | Recall on known rings |
| Evaluate a new alert rule | Marginal confirmed cases, net of duplicate load | Analyst time per confirmed case | Alert count |

The last column is the point of the table. Every metric listed there is defensible in a different context, which is why it gets used; it becomes misleading only when it is attached to the wrong decision.

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

**8.11 Entity count drift from inactive keys.** *Symptom:* entity-grain prevalence falls by 30 percent with no change in fraud counts. *Competing causes:* (a) dormant cards were reactivated into the denominator; (b) the entity definition widened to include supplementary cards; (c) genuine fraud reduction. *Discriminating check:* recompute the rate on the intersection of the two entity sets, using the same numerator. If the intersection rate is stable, the change is entirely denominator-side and the trend should be restated rather than reported as improvement.

**8.12 Label maturation inside one reporting month.** *Symptom:* month-over-month fraud counts show a saw-tooth pattern that repeats annually. *Competing causes:* (a) chargeback windows interacting with holiday volume; (b) investigator vacation capacity; (c) seasonality of the attack itself. *Discriminating check:* plot the count by label age rather than by event month for the same cohort. A saw-tooth that disappears when plotted by label age is a maturation artefact; one that persists is genuine seasonality.

**8.13 Alert deduplication hiding a new actor.** *Symptom:* suppressing duplicates reduces alerts by a third and the confirmed count falls alongside it. *Competing causes:* (a) the suppressed repeats carried separate information; (b) the suppression key merged two actors; (c) the dedup window was longer than the case lifetime. *Discriminating check:* re-run one month with suppression on and off, and compare confirmed cases per actor rather than per alert. A loss concentrated in actors whose first alert preceded an open case means the key was too coarse.

**8.14 Metric definition changed under pressure.** *Symptom:* a team's reported detection rate jumps 9 points in the month a new lead arrives. *Competing causes:* (a) the denominator moved from events to entities; (b) the population lost its hardest segment through a filter; (c) a genuine model improvement. *Discriminating check:* recompute the previous month's number under the new definition. If the jump survives, the definition changed the number; if the jump vanishes, the model changed the number. Report both figures and the restated series.

### 8b Two discriminating traces

The checks above are stated as procedures. Two short traces show what the output looks like when a check is actually run, because the failure mode of a check is usually that nobody computes the comparison number.

```text
CHECK 8.1  prevalence reconciliation
  reported loss rate (finance)        0.0410 percent of volume, value weighted
  reported event prevalence (model)   0.0900 percent of rows, count weighted
  mean fraud amount / mean all amount = 720 / 62 = 11.6129
  implied value-weighted rate = 0.0900 * 11.6129 = 1.0452 percent
  observed ratio 0.0410 / 1.0452 = 0.0392   <-- not reconciled
  next step: one of the two uses a different window or net-of-recovery basis
```

A ratio of 0.039 rather than 1.000 is the signal that the weighting explanation is insufficient. The correct follow-up is not more arithmetic but a check of the two source windows, because a value-weighted rate computed net of recoveries over 12 months will not match a count-weighted rate over 3 months no matter how the weights are applied.

```text
CHECK 8.5  censoring versus decline, two cohorts
  cohort A (90 days older): plateau 0.0115 reached at age 96 days
  cohort B (recent):        plateau 0.0098 reached at age 94 days
  ratio B/A at plateau = 0.852174
  age to 90 percent of plateau: A = 71 days, B = 69 days
  reading: both plateaus are observed, so censoring is excluded;
           the 14.8 percent drop is a real change in the population.
```

The diagnostic value comes from waiting for both plateaus. A team that compares the two cohorts at 30 days would see 0.55 versus 0.53 and conclude no change, while at 96 days the difference is 14.8 percent. Any statement about fraud trend must state the age at which it was measured.

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

**Cost of a wrong entity key.** The cost is asymmetric in a way that is rarely modelled. Over-merging produces a false block on a legitimate customer, which costs friction plus attrition and is visible. Under-merging produces a miss on a genuine ring, which costs the fraud loss and is invisible because the fraud appears as ordinary independent events. The ratio between the two costs is typically between ten and fifty to one in favour of under-merging being worse, and yet most resolution thresholds are tuned on the visible side because that is the side with data.

```text
FORMULA
  res_cost(threshold) = N_pair * [FPR(t) * c_over + FNR(t) * c_under * k]

WHERE
  N_pair     candidate pairs scored per month                 count
  FPR, FNR   pair-level error rates at threshold t            [0,1]
  c_over     cost of one wrongly merged pair                  currency
  c_under    cost of one missed merge, per pair on the ring   currency
  k          mean ring size amplification, missed pairs -> loss  pairs

CONSTRAINT
  pair errors are treated as independent; ring loss scales with size.

NUMERICAL EXAMPLE
  N_pair = 75,000; c_over = 90.00; c_under = 400.00; k = 3.0
  t1: FPR 0.002, FNR 0.35 -> 75,000*(0.002*90 + 0.35*400*3)
                          = 75,000*(0.18 + 420) = 75,000*420.18 = 31,513,500
  t2: FPR 0.010, FNR 0.18 -> 75,000*(0.90 + 216) = 75,000*216.90 = 16,267,500

INTERPRETATION
  Lowering the threshold (t2, more merges) nearly halves the cost despite
  five times the over-merge rate, because the under-merge term dominates.
  The 400/90 cost ratio plus the ring amplification is what drives it.

VALIDITY CONDITION
  Holds while c_over excludes litigation or reputational cost; if a wrongly
  frozen account produces a regulatory complaint, the ratio can invert.
```

**Cost of latency headroom.** Reserving 190 ms of a 400 ms budget for tail latency is not overhead to be reclaimed. At a p99 target of 380 ms, adding one 80 ms join moves p99 to roughly 460 ms and breaches the contract, and the breach is not uniformly distributed: it lands on the slowest 1 percent of requests, which are disproportionately mobile-network users. The latency cost therefore converts directly into a fairness cost, and that is why feature platforms for fraud decisions are usually precomputed rather than joined.

**Cost of stale features.** A precomputed velocity counter refreshed hourly is 0 to 60 minutes stale. Staleness costs accuracy on exactly the burst patterns that matter: a card tested five times in four minutes looks like one test to an hourly counter. The mitigation is a hybrid, streaming short-window counters plus batch long-window counters, and its cost is two systems to maintain plus a reconciliation job to keep them consistent. The reconciliation failure mode is silent: when the two disagree, the model reads whichever is queried first.

**Cost of audit evidence.** Each decision must retain the score, the threshold version, the feature vector summary, and the action. At 4,000,000 decisions per month and roughly 400 bytes of retained evidence per decision, that is 1.6 GB per month, 19.2 GB per year, which is trivial. The real cost is that the evidence must remain interpretable after three feature-schema changes, so the retention design must include a schema version and the ability to replay an old decision. Teams that omit the schema version discover the gap during their first dispute arbitration and cannot reconstruct why a decision was taken.

**Cost of a model refresh cadence.** Monthly refreshes cost roughly three engineer-days each in validation and sign-off, so twelve refreshes is 36 days of engineering per year. If the model's decay curve halves performance every nine months, quarterly refresh is adequate and monthly is waste. Decay rates are measurable: track precision at fixed volume by month against a fixed reference cohort, and refresh only when the drop exceeds the friction of a release.

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

**R11 — Rung monotonicity of prevalence.** Entity prevalence is at least event prevalence whenever each fraud event sits on a distinct fraudulent entity and `n̄ ≥ m`. *Condition:* both counted in the same window. *Why it matters:* it makes the direction of a grain correction predictable, so a reviewer can reject a claimed reconciliation that moves the wrong way.

**R12 — Review-slot shadow price.** With `K` slots per period and marginal items ranked by net benefit density, the shadow price of a slot equals the density of the last item admitted. *Condition:* items independent, hours divisible. *Why it matters:* it converts "we need more analysts" into a number that finance can compare against the benefit curve.

**R13 — Calibration-preserving monotone transforms.** Any strictly increasing transform of a calibrated score preserves ranking and destroys calibration, so thresholds must be re-derived after any rescaling. *Condition:* transform is monotone. *Why it matters:* it explains why a model refresh that improves rank metrics can silently move the operating point, and why the operating point must be versioned with the score transform.

**R14 — Duplicate-alert dilution.** If alert precision is `P` and the mean alerts per case is `c`, case precision equals `P · c` under uniform alert quality. *Condition:* alerts generated independently of case identity. *Why it matters:* it gives the exact ratio in Section 5f and shows that a 0.4 percent alert precision with 50 alerts per case is consistent with 20 percent case precision.

**What these results do not cover.** None of R1 to R14 addresses adversarial adaptation, which is the defining property of fraud rather than a nuisance. A model whose threshold is public, or inferable from repeated probing, has its effective threshold moved by the attacker. The nearest formal treatment is a Stackelberg game in which the defender commits to a policy and the attacker best-responds, and the practical consequence is that any published or probed threshold must be randomised within a band. Randomisation costs a little precision and buys a lot of durability, and no offline metric reveals the trade because offline data contains no attacker response.

**Reading the results as a checklist.** R1, R4, and R13 govern thresholds. R2, R3, R11 govern grain. R6, R7, R8 govern labels and evaluation. R5, R12 govern capacity. R9, R10, R14 govern cost and gaming. A reviewer facing an unfamiliar fraud proposal can walk the groups in that order and will usually find the weak assumption in the second group, because grain assumptions are the ones nobody writes down.

**R15 — Sampling-ratio prior correction.** Rebalancing the training set by sampling rates `w1` and `w0` shifts the log-odds by `log(w1/w0)`, so a model trained on rebalanced data reproduces the population probability only after subtracting that constant. *Condition:* the sampling rates were recorded and sampling was independent of features given the label. *Why it matters:* it is the mechanism behind Section 4.4's fifty-fold error and the reason a rebalancing pipeline must persist its ratios alongside the model artefact.

**R16 — Multiplicative recalibration.** If predicted probabilities exceed the observed rate by a constant factor `miscal`, dividing all scores by `miscal` restores calibration and leaves ranking untouched. *Condition:* the miscalibration is multiplicative rather than band-specific. *Why it matters:* it separates a calibration repair from a threshold change, which are frequently conflated when an operating point misbehaves.

**R17 — Three-action partition.** With allow, challenge, and block available and linear losses, the optimal policy is a two-boundary partition with `tau_low = f_c / (a - a_c + f_c)` and `tau_high = (f_b - f_c) / (a_c + f_b - f_c)`, and `tau_high > f_b/(f_b + a)` whenever `a_c < a`. *Condition:* challenges resolve and their residual loss is strictly below the allow loss. *Why it matters:* it proves that the block boundary is not an independent policy choice; it is determined by how much the challenge recovers.

**R18 — Knapsack dominance of density ranking.** When review hours are the binding resource and items are independent, ranking by net benefit per hour and taking items greedily is optimal for the fractional problem and within one item's value for the integral one. *Condition:* independence, divisible hours. *Why it matters:* it justifies the per-hour ranking in Section 4.4 and shows why absolute-value ranking is wrong under a hard budget.

**R19 — Additivity failure under coordination.** For a coordinated ring, loss of the set exceeds the sum of member losses computed independently, so any metric that sums per-entity expected losses understates ring risk by the coordination premium. *Condition:* members condition their behaviour on each other. *Why it matters:* it is the formal reason a ring rung exists and why entity-grain aggregation cannot substitute for it.

**R20 — Threshold probing.** If an attacker can query a deterministic policy, the number of queries needed to locate a threshold to within `epsilon` in score space is about `1/epsilon` in the one-dimensional case, so any deterministic cut-off is recoverable at low cost. *Condition:* queries are answered and observable by the attacker. *Why it matters:* it converts threshold secrecy from a security preference into an arithmetic constraint, and it is why production policies randomise inside a band.

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

**Q15. A challenge action recovers 70 percent of the loss. Recompute the block boundary given `a = 180`, `f_c = 1.80`, `f_b = 6.00`.** A 70 percent recovery means `a_c = 0.30 × 180 = 54.00`, so `tau_low = 1.80 / (180.00 − 54.00 + 1.80) = 1.80 / 127.80 = 0.014085` and `tau_high = (6.00 − 1.80) / (54.00 + 6.00 − 1.80) = 4.20 / 58.20 = 0.072165`; the band is 5.81 percentage points wide. If the challenge recovers only 40 percent, `a_c = 108.00` and both edges move inward: `tau_low = 1.80 / 73.80 = 0.024390`, `tau_high = 4.20 / 112.20 = 0.037433`, a band of 1.30 points. The recovery rate, not the threshold designer, controls how much friction the policy can spend.

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
