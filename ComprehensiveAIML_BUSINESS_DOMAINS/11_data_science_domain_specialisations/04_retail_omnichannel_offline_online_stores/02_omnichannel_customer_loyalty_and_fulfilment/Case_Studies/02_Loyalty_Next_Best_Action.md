---
case_id: "CASE-04-02-02"
case_title: "Loyalty Next Best Action"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Retail omnichannel"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "distribution-shift"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "loyalty manager"
source_index: "INDEX.md"
review_trigger: "Decision policy, offer economics, contact channels, workload, or objective changes"
---

# Loyalty Next Best Action

Every number, population count, cost, and outcome in this case is **ILLUSTRATIVE**: constructed
teaching inputs whose arithmetic has been recomputed by hand, not a measurement of any retailer,
programme, vendor, or deployment. No company architecture, contract term, or realised profit is
asserted. Where a quantity would require an executed experiment, the case states `not-measured` and
names the evidence that would settle it. The case is `proposed`, not published, and its status in
the collection index must stay `proposed` until an independent reviewer accepts the arithmetic and
the evidence boundary.

## 1. The One-Line Framing

At a fixed weekly decision timestamp, choose at most 120,000 of 250,000 eligible loyalty
member-weeks for a single contact offer, ranked by estimated *incremental* gross margin net of
contact cost, subject to a contact budget, a review-capacity gate, and a fourteen-day cooldown —
or send nothing to the remainder and record the abstention.

The framing has four parts, and each one is a place where a plausible project goes wrong.

- *Decision*: a selection of member-weeks, not a forecast of behaviour and not a customer score.
- *Objective*: incremental gross margin, meaning the margin that would not have occurred without
  the contact, minus the cost of making the contact. Response rate, conversion rate, and
  propensity are inputs at best and distractors at worst.
- *Binding constraint*: the decision must be explainable, reversible, and auditable before its
  effective time; a member who cannot be contacted lawfully, or whose offer cannot be justified
  within policy, is not eligible no matter how attractive the estimate.
- *Reusable pattern*: `evaluation-mismatch` — a convenient model target (probability of purchase
  given contact) is not the operational objective (margin caused by contact), and the gap between
  them has a sign, not just a magnitude.

The one-line answer the case defends is therefore conditional rather than triumphant:

> Rank by net incremental margin, select greedily under the budget, cap frequency per member from
> the measured decay of repeat-contact lift, and refuse to ship until a holdout can resolve the
> effect the policy claims to create.

A useful framing must also say what would overturn it. Three observations would:

1. A holdout shows the estimated uplift is indistinguishable from zero at the achievable sample
   size, so the honest policy is the baseline rules queue with a smaller budget.
2. The "sure thing" share of the selected population is so high that measured lift sits below the
   break-even threshold of 2.6 percentage points for every decile.
3. The review gate cannot be staffed, so selected offers wait past their effective window and the
   decision becomes a late notification rather than a choice.

Takeaway: the deliverable is a bounded, auditable selection rule with a stated invalidation test,
not a model that scores loyalty members.

## 2. Context & Background

A grocery-and-general-merchandise retailer runs a points-based loyalty programme with 1,200,000
members who transacted at least once in the last ninety days. The programme is the retailer's main
owned-audience asset: it can reach members by email, by app push, and by SMS, and it can attach a
funded offer to a contact. Marketing has run a weekly "next best action" queue for three years.

The queue began as a hand-written rules engine with four tiers and a recency score. It was
defensible: the rules were short, the owner could explain any row, and finance could reconcile the
budget. It was also blunt — every member in a tier received the same treatment regardless of
whether the offer changed their behaviour.

Two pressures pushed the programme toward a learned ranking.

The first was budget scrutiny. The offer budget is a fixed line item, and finance began asking for
the incremental margin per funded contact rather than redemptions per thousand sends. A redemption
is not margin: the member may have bought anyway, at full price, on the same trip.

The second was channel cost and fatigue. Email is nearly free per message, SMS is not, and app push
is free but capped by operating-system policy and by member tolerance. As the programme added
channels, the cost of a contact stopped being negligible and the cost of an unwanted contact
stopped being zero.

Three clocks run through this decision, and conflating them is the most common authoring error in
loyalty analytics.

- *Event time*: when the member did the thing — the purchase, the app session, the redemption.
- *Decision time*: the weekly cut-off, Monday 06:00 local, at which eligibility and estimates are
  frozen for that cycle.
- *Outcome time*: when the consequence is observable, which for a 28-day purchase window with a
  returns tail is roughly 42 days after the decision.

A row that mixes these clocks can look predictive while being unusable. A feature built from
"purchases in the last 28 days" is safe if the window ends before the cut-off. The same feature
built from "purchases since the campaign started" is a post-treatment variable and will inflate any
offline estimate.

The programme has four stakeholders whose objectives differ, and the case must satisfy all four
without pretending they coincide.

- The *loyalty manager* owns the queue and the member relationship; she is judged on retention and
  on complaint volume, and she can veto any contact that reads as harassment.
- *Finance* owns the offer budget; it is judged on incremental margin and wants a holdout, because
  a holdout is the only thing that converts a ranking into a claim.
- *Privacy and compliance* own consent state, channel preference, suppression lists, and the
  minimisation of personal data used for targeting; they can remove a member from the population.
- *Store operations* own the redemption experience; an offer that cannot be honoured at a
  particular store produces a service failure that no model metric captures.

The omnichannel dimension is not decoration. A member may redeem online for delivery, online for
pickup, or in a store; the same offer has different margin, different friction, and different
return rates by channel. A model trained on online redemptions only will mis-rank members whose
behaviour is mostly in store, which in this constructed population is 61% of transactions.

The baseline is worth stating precisely because it is the comparator, not a straw man. The rules
queue ranks by a weighted score:

```text
baseline_score = 0.45 * recency_points + 0.35 * tier_points + 0.20 * category_affinity_points
```

where `recency_points` decays linearly from 100 at seven days since last purchase to 0 at ninety
days, `tier_points` is 100, 70, 40, 0 for the four tiers, and `category_affinity_points` is the
member's share of spend in the promoted category scaled to 0..100. The top 120,000 rows by this
score receive the offer, subject to the cooldown and consent filters.

That baseline is deterministic, auditable, and free to run. It is also blind to incrementality: a
member with recent purchases in the promoted category scores highest, and that member is exactly
the member most likely to buy without any contact. The baseline therefore concentrates the budget
on the population with the smallest headroom for influence.

This is the structural reason the case exists. The business did not have a model-quality problem;
it had a target-variable problem. Every downstream artefact — the training label, the offline
metric, the launch dashboard — inherited a target that answered "who will buy if contacted?"
instead of "whose behaviour changes because of the contact?".

The distinction has a name in the causal-inference literature (the conditional average treatment
effect) and a name in retail practice (uplift, or incremental response). Both names point at the
same arithmetic: the difference between two potential outcomes for the same member, of which only
one is ever observed.

Because only one potential outcome is observed per member, the effect must be estimated from a
randomised comparison, from a model trained on randomised history, or from a defensible
quasi-experiment. The programme had two years of randomised holdouts at 5% of the contacted band,
which is enough to train an uplift model but not enough to validate every decile separately — a
constraint the evaluation section quantifies rather than waves away.

The operating cadence matters for design. Decisions are weekly, the offer is valid for fourteen
days, the cooldown is fourteen days, and the outcome window is twenty-eight days. A member can
therefore appear in at most two decision cycles per outcome window, which creates the clustering
that any uncertainty calculation must respect.

Seasonality is the main source of distribution shift. In the four weeks before the winter holiday
period, baseline purchase probability rises sharply for almost every member, which compresses
uplift: when a member is going to buy anyway, the offer changes nothing and the funded contact
becomes a transfer rather than an incentive. The case treats this as a first-class regime, not an
anomaly.

Finally, the case distinguishes three artefacts that are often collapsed into one word, "the
model".

- A *scoring model* estimates a probability or a value for a member.
- A *decision policy* converts estimates plus constraints into a selection.
- A *measurement design* converts the selection into an estimable effect.

The failure this case studies is a policy built directly on a scoring model that was never designed
to answer the policy's question, evaluated by a metric that could not have detected the error.

## 3. The Business Question

The question, stated so that an answer can be checked, is:

> For the decision cycle beginning Monday 06:00, which eligible member-weeks should receive the
> funded offer so that expected incremental gross margin net of contact cost is maximised, given a
> 120,000-contact budget, a 2,400-review capacity, a fourteen-day cooldown, and the consent and
> suppression rules — and which member-weeks should receive nothing?

The phrasing carries four commitments that a looser question would hide.

- The unit of decision is the member-week, not the member. A member can be eligible in consecutive
  cycles; the policy must therefore reason about frequency, not only about who.
- The objective is net of cost, so a contact with a positive but tiny estimated lift can be
  correctly rejected. Zero is a legitimate output.
- The constraints are enumerated, so a candidate answer can be tested for feasibility before it is
  tested for quality.
- "Which member-weeks should receive nothing" forces the policy to record abstentions, which is
  what makes the counterfactual population visible later.

The owner is the loyalty manager. The trigger is the scheduled weekly cut-off, plus an exception
path for a mid-week supply or pricing event that invalidates a batch already selected. The cadence
is weekly; the effective time of a decision is the send time, which is at least six hours after the
cut-off to allow the review gate to clear.

The action space has five elements, and each must be available to the policy.

| Action | Meaning | Cost | Reversible before send |
|---|---|---|---|
| `baseline` | Apply the legacy rules queue unchanged | Offer + message | Yes |
| `ranked_offer` | Contact with the funded offer, ranked by net incremental margin | Offer + message | Yes, until send |
| `ranked_no_offer` | Contact with a content-only message, no funded incentive | Message only | Yes, until send |
| `defer_review` | Hold for a human reviewer because the row breaches a policy envelope | Review time | Yes |
| `abstain` | Send nothing; log the reason | Zero | Not applicable |

The existence of `ranked_no_offer` matters more than it appears. Some members have a positive
response to being reminded but a negative response to a discount, because the discount teaches them
to wait. A two-action policy that can only choose "offer" or "silence" cannot express the
reminder-only option, and will over-spend on incentives for members who only needed a nudge.

The objective function, written to be computable, is:

```text
maximise  sum over selected i of  [ m * (p1_i - p0_i) - c_i ]
subject to
    selected_i in {0,1}
    sum_i selected_i <= B                      (contact budget)
    selected_i = 0 if consent_i = 0            (hard eligibility)
    selected_i = 0 if days_since_contact_i < 14 (cooldown)
    reviews_required(selected) <= R            (review capacity)
```

with `m` the expected gross margin per incremental purchase in dollars, `p1_i` and `p0_i` the
probability of purchase in the 28-day window with and without contact, `c_i` the contact cost, `B`
the budget in contacts, and `R` the review capacity in rows.

The hard constraint that is easiest to state and hardest to satisfy is the point-in-time rule: no
input timestamped after the decision cut-off may influence the selection. This is not a stylistic
preference. A feature that is available at scoring time but not at decision time turns the offline
estimate into a forecast of the past.

Scope is deliberately narrow: one decision cycle's selection, the arithmetic that ranks it, and the
measurement design that would test it. Out of scope are vendor selection, the legal determination of
what counts as consent in each jurisdiction, the creative content of the offer, and any claim about
deployed business impact.

An answer that satisfies the question has four parts, and an answer missing any of them is
incomplete regardless of how good the ranking looks.

1. A selected set with a feasibility certificate: count within budget, cooldown and consent
   respected, review load within capacity.
2. A value statement per selected tranche, with units in dollars per member-week and the assumption
   set that produced it.
3. A statement of what happens to the unselected population, including whether they receive the
   content-only message or silence.
4. An invalidation test: the observation that would show the ranking is wrong, and the sample size
   at which that observation could be seen.

The escalation path is part of the question, not an afterthought. If the estimated value of the
selected set is negative under the central assumptions, the policy must be able to return
`baseline` rather than an empty set, because an empty set is a silent change of programme and would
be read by members and by the business as an outage.

Takeaway: a well-posed loyalty question names the unit, the cost-adjusted objective, the constraint
set, and the abstention, because each of those four choices can flip the selected set.

## 4. Assumptions & Constraints

The assumptions are listed with a test, a consequence if false, and a fallback, because an
assumption without a fallback is a hope. The identifiers are stable so that later sections and the
review log can cite them.

| ID | Assumption | Test | If false | Fallback |
|---|---|---|---|---|
| A-01 | Every input is point-in-time correct | Replay with cut-off filter; diff against raw event times | Offline lift overstated | Rebuild features as-of; re-run replay |
| A-02 | Outcome definition is stable across cycles | Double-code 200 sampled outcomes | Labels mix definitions | Freeze a codebook; re-label the window |
| A-03 | Members do not influence each other | Check household and shared-device overlap | Effect estimate biased | Cluster randomisation or household holdout |
| A-04 | Gross margin rate is stable at 30% | Compare finance actuals by category-month | Value per purchase wrong | Use category-specific margin vector |
| A-05 | Contact cost is 0.62 dollars per contact | Reconcile message and redemption invoices | Break-even threshold wrong | Recompute per channel and offer type |
| A-06 | Uplift estimates are calibrated by decile | Compare predicted and holdout observed lift | Ranking unreliable | Shrink to segment means; widen holdout |
| C-01 | Budget is 120,000 contacts per cycle | Read the approved budget line | Truncation is arbitrary | Lower budget, re-solve threshold |
| C-02 | Cooldown is fourteen days | Query contact log for violations | Members over-contacted | Hard filter before ranking |
| C-03 | Review capacity is 2,400 rows per cycle | Count reviewer hours times throughput | Gate becomes a queue | Auto-approve within envelope |
| C-04 | Offers are revocable before send | Exercise cancellation on a sample batch | Late sends are unavoidable | Select one cycle ahead |

The first assumption carries most of the risk, so it gets the most scrutiny. Point-in-time
correctness is not a property of a table; it is a property of a join. The same column can be
correct in one query and leaking in another depending on which timestamp the join uses.

Three timestamp families must be distinguished for every source:

- `event_ts`, when the business event happened in the world.
- `ingest_ts`, when the warehouse received the record.
- `effective_ts`, when the record becomes the authoritative version for reporting.

A feature is safe only if the record's `effective_ts` is at or before the decision cut-off minus a
safety margin for late-arriving corrections. In this case the margin is fifteen minutes, which is
deliberately shorter than the largest observed correction lag, so the audit is expected to find
some violations and the pipeline is expected to quarantine them rather than silently include them.

The second assumption, a stable outcome definition, is where loyalty programmes usually rot. Three
different things get called "conversion": a purchase in any channel within 28 days, a redemption of
the specific offer, and an increase in category spend. They have different denominators and
different sensitivities, and mixing them across cycles produces a metric that drifts without anyone
changing the model.

The third assumption is the one most often skipped. Uplift estimation assumes no interference: one
member's treatment does not change another member's outcome. In grocery retail, interference is
real and mundane — a household with two loyalty cards, a shared tablet, a family member forwarding
a discount code, a store promotion that changes traffic for everyone.

Because interference exists, the holdout must be drawn at the household level where household
identity is known, and the residual interference must be acknowledged as a downward bias on the
measured effect whenever treatment saturation is high. At a 48% contact rate within the eligible
band, saturation is high enough that this bias is not negligible.

The fourth assumption converts purchases into money. A single margin rate of 30% is a convenience;
the honest version is a margin vector by category and channel, because a $24 gross margin per
incremental purchase is the product of an $80 basket and a 30% rate, and both components vary.

```text
m = basket_value * margin_rate
  = 80.00 * 0.30
  = 24.00 dollars per incremental purchase
```

If the offer shifts mix toward the promoted category, and that category runs at 22% margin, the true
`m` for the incremental purchase is $17.60, and the break-even lift rises by a factor of
24.00/17.60 = 1.36. A policy tuned on the blended rate will over-contact in exactly the category
where the offer is cheapest to fund and least profitable to cause.

The fifth assumption, contact cost, is small per unit and large in aggregate, which makes it the
most frequently under-modelled quantity in the whole decision.

```text
c = message_cost + expected_incentive_cost
  = 0.06 + (redemption_rate * face_value)
  = 0.06 + (0.08 * 7.00)
  = 0.06 + 0.56
  = 0.62 dollars per contact
```

At 120,000 contacts per cycle the aggregate is $74,400 per week, or $3.87 million per year across
52 cycles. A 20% error in the redemption assumption moves the break-even lift by 20%, which moves
the selected set by a decile. Cost is not a rounding term here; it is a selection parameter.

The sixth assumption is calibration. An uplift model can rank well and still be badly calibrated,
and a decision threshold applied to an uncalibrated score is arbitrary. Decile-level calibration is
the minimum acceptable check: predicted mean lift per decile versus holdout-observed mean lift per
decile, with a tolerance band that the owner sets in advance.

The constraints are not symmetric with the assumptions. An assumption that fails produces a wrong
number; a constraint that fails produces an infeasible selection, which must be rejected outright.

The budget constraint binds: the eligible population of 250,000 member-weeks is more than twice the
120,000-contact budget, so the policy must truncate. Because every contact costs the same, the
truncation is a threshold on value per contact, and the optimal rule is a greedy sort rather than a
general knapsack.

The cooldown constraint is a filter, not a penalty. A member contacted eight days ago is removed
before ranking, because including them with a large penalty would allow the optimiser to trade the
penalty against value and produce a policy that violates the cooldown whenever the estimate is high.

The review constraint is the one that changes the shape of the answer. Offers above a policy
envelope, and all offers to members flagged for a service-recovery context, require a human review
before send. Reviews consume 2.4 minutes of a specialist's time on average, and the team can supply
2,400 reviews per cycle, so the gate admits 2,400 rows and no more.

Privacy constraints are absolute and take precedence over value. A member who has withdrawn
marketing consent is ineligible even if their estimated lift is the highest in the population.
Suppression lists — complaints, bereavement, litigation — are honoured without exception, and the
policy has no mechanism to override them, which is a design property rather than a limitation.

The reversibility constraint is about the effective time. A selection is reversible only before the
send job commits; after that it is a communication to a customer, and the recovery path is an
apology rather than a rollback. This is why the six-hour gap between cut-off and send is a
constraint and not a convenience.
## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** One row per `member_id` × `decision_cycle`. A decision cycle is a Monday 06:00 local
cut-off. With 1,200,000 active members and a weekly cycle, the raw grain produces 1,200,000 rows per
cycle; after the eligibility filters described below, 250,000 rows survive as *eligible
member-weeks*. Everything downstream — scoring, selection, evaluation — uses that filtered grain.
Any join that changes the grain (for example joining transactions) must be aggregated back before
selection, or the row count becomes a transaction count and the budget constraint silently changes
meaning.

**Keys.** `member_id` is the loyalty identifier and is stable. `household_id` is present for 78% of
members; the remainder are treated as their own household for holdout assignment, which biases
measured effects *upward* slightly because true clustering is under-modelled. `decision_cycle_id`
is the ISO week plus cut-off date, and `decision_id` is a deterministic hash of member and cycle so
that replay is idempotent. `offer_id` links to the funded incentive's terms, and `campaign_version`
links to the creative and the policy envelope.

**Cut-off.** The decision timestamp is Monday 06:00:00 local. All input tables are read with
`effective_ts <= cut_off - interval '15 minutes'`. The fifteen-minute margin is a deliberate
under-estimate of observed correction lag, chosen so the quarantine counter is exercised in normal
operation rather than discovered in an incident.

**Population.** Eligible member-weeks satisfy all of:

- at least one transaction in the trailing 90 days (active),
- marketing consent in force for at least one contactable channel,
- not on any suppression list,
- no contact in the trailing 14 days,
- not part of a conflicting experiment,
- a resolvable household identifier for holdout assignment, or a deterministic fallback.

The filters are not neutral. Each removes a subpopulation with a different baseline purchase
probability, so the eligible population is not a random sample of members, and any historical
analysis that ignores the filters will not reconcile with the live queue.

**Units.** Money in US dollars, unrounded internally and rounded to cents only for display.
Probabilities in percentage points at two decimals for discussion and in proportions for
arithmetic. Time in whole days for recency and cooldown, and in UTC timestamps for event ordering.
Margins are per incremental purchase, not per order and not per line.

**The outcome.** For a member-week decided at cycle `t`, the outcome is gross margin in dollars
attributable to purchases in `[cut_off, cut_off + 28 days]`, net of returns processed up to 42 days
after cut-off. The 28-day purchase window and the 42-day return tail are separate windows because
mixing them produces a metric that appears to improve when returns are slow.

**What is not available at decision time.** This list is more important than the list of what is,
because every item on it is a leakage trap:

- whether the member purchased after the cut-off (the outcome itself),
- redemption status of the current offer,
- returns against purchases made after the cut-off,
- service interactions triggered by the current campaign,
- the next cycle's contact history,
- competitor pricing observed after the cut-off,
- any holdout assignment outcome.

**Volume.** A constructed cycle in this case has 1,200,000 active members, 250,000 eligible
member-weeks after filters, 120,000 selected for contact, 2,400 routed to human review, and 130,000
receiving no contact at all. The selected band therefore covers 48% of eligible rows, which is the
saturation level at which interference between members stops being a footnote.

### 5b Tools & Stack

The stack is described by capability, not by vendor, because vendor claims would be unsourced.

| Layer | Capability required | Why |
|---|---|---|
| Source | Event-level transaction, contact, consent, and offer tables | Point-in-time reconstruction |
| Warehouse | Versioned tables with `effective_ts` | As-of joins and replay |
| Transform | Deterministic, version-controlled feature build | Reproducible scores |
| Training | Gradient-boosted trees or a two-model difference | Handles mixed categorical and sparse signals |
| Serving | Batch scoring before cut-off, not per-request | Weekly cadence, no latency requirement |
| Decision | Constraint solver or greedy sort with gates | Feasibility before quality |
| Log | Append-only decision log with rationale | Audit and replay |
| Review | Queue with per-row policy envelope and reason codes | Human-in-the-loop |
| Measurement | Holdout assignment, outcome join, decile report | Causal claim |

Two stack choices are load-bearing and easy to get wrong.

The first is batch scoring before the cut-off rather than real-time scoring at send time. Weekly
cadence means latency is irrelevant, and batch scoring forces the feature snapshot to be a concrete
artefact that can be diffed. A real-time scorer would make the point-in-time check a property of the
code path rather than a property of the data, and the check would then be untestable in isolation.

The second is the append-only decision log. It stores, per row: the input snapshot identifier, the
feature vector hash, the model version, the estimate, the policy decision, the reason code, the
selected or abstained flag, and the reviewer identifier if reviewed. Without this log, a later
metric movement cannot be attributed to a world change or a calculation change.

Deliberately absent from the stack: a real-time feature store, a streaming pipeline, and an online
learning loop. Each would add failure surface for a decision that runs 52 times a year. The
evaluation section treats "did we need any of them?" as a live question with a specific test.

### 5c Method Summary

Four candidate methods were considered and one was carried forward. The comparison is part of the
method, not a formality, because the losing candidates each fail a specific check.

| Candidate | Target | Strength | Failure | Verdict |
|---|---|---|---|---|
| M1 rules queue | Weighted recency, tier, affinity | Auditable, zero training cost | Blind to incrementality | Baseline only |
| M2 propensity | P(purchase given contact) | Easy labels, familiar metrics | Selects sure things | Rejected |
| M3 uplift difference | P(purchase with contact) − P(without) | Directly matches objective | Needs randomised history; noisy at decile level | Carried forward |
| M4 causal forest on margin | Heterogeneous dollar effect | Handles continuous outcome | Unstable at small leaf sizes; harder to audit | Deferred |

**M1** is retained as the comparator, not as a fallback to be abandoned. A policy that cannot beat
the legacy rules queue on the frozen evaluation window does not ship, and the queue is cheap enough
to keep running while evidence accumulates.

**M2** is rejected on a structural argument rather than on measured performance. A propensity model
maximises P(purchase | contacted), and the members with the largest P(purchase | contacted) are
dominated by members with a high P(purchase | not contacted). Ranking by that target moves budget
toward the population with the least headroom. Section 6 quantifies the loss with constructed
numbers; the point is that the loss is a property of the target, not of the model class.

**M3** is carried forward as a two-model difference: one model trained on treated rows to estimate
`p1`, one on control rows to estimate `p0`, with the estimate `Δp = p1 - p0` clipped at zero for
selection and reported unclipped for diagnostics. The clipping is a policy choice and is recorded
as such; it prevents a negative-lift member from being selected by a value tie-break, but it also
means the aggregate estimate is biased upward, which the calibration report must disclose.

**M4** is deferred because the dollar outcome is heavy-tailed and the leaf-level estimates are too
noisy to pass the review gate's explainability requirement. It remains a candidate for the
following year, when two more randomised cycles will exist.

The selection rule is a greedy sort on value per contact, which is optimal here because every
contact consumes exactly one unit of the budget. The review gate is applied *after* selection, as a
partition rather than a re-ranking, so the gate cannot change the order of the selected set.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** Every source table carries a `source_system`, `source_version`, `ingest_ts`, and
`effective_ts`. The feature build records the snapshot identifier. The decision log records the
snapshot identifier it consumed. A score can therefore be traced to a specific read of a specific
table version, which is what "reproducible" has to mean in a batch pipeline.

**Quality rules.** Six checks run before scoring, and each has a threshold that blocks publication
of the cycle rather than merely warning:

| Check | Rule | Threshold | Action on breach |
|---|---|---|---|
| Q1 Completeness | Non-null consent state per row | 100% | Block; resolve consent source |
| Q2 Freshness | Max feature age | 26 hours | Degrade to baseline |
| Q3 Uniqueness | One row per member-cycle | 0 duplicates | Deduplicate, then block |
| Q4 Range | `p1`, `p0` in [0,1]; `Δp` in [−1,1] | 0 violations | Quarantine rows |
| Q5 Referential | `offer_id` resolves to an active offer | 100% | Block |
| Q6 Drift | Share of rows with unseen category codes | ≤2% | Flag for review |

Q2 is the interesting one, because a freshness breach should not stop the programme. Degrading to
the baseline keeps the queue running with known properties, which is better than either stopping or
scoring on stale features.

**Exclusions.** Rows are excluded for: unresolvable member identity after deterministic matching,
duplicate member-cycle keys retained from a merge, negative-tenure members (data error), offers
whose funding was withdrawn mid-cycle, and members inside a conflicting test cell. The exclusion
rate is measured every cycle and reported as a count and a share; in the constructed cycle it is
3.8% of the 260,000 pre-filter rows, or 9,880 rows.

Excluded rows are not a random draw. Members with unresolved identity skew toward multi-card
households and therefore toward higher frequency, so a policy evaluated only on resolvable members
will over-state its own coverage and under-state the frequency problem.

**Leakage.** Four leakage channels were audited, and the audit is the substantive part of this
subsection:

1. *Outcome leakage*: a feature built from the 28-day post-cutoff window. Detected by a timestamp
   audit that compares each feature's definition window against the cut-off. Not present in the
   carried-forward build; present in a rejected prototype whose offline AUC was 0.91, which is
   itself the tell.
2. *Treatment leakage*: the contact log for the current cycle appearing as a feature. Detected by
   schema review; the contact table must be joined with `event_ts < cut_off`.
3. *Selection leakage in training*: training only on rows that were previously contacted, which
   makes `p0` unidentifiable because untreated rows are absent from the training set. Detected by
   counting control rows per decile; if any decile has fewer than 500 control rows, the decile
   estimate is not reportable.
4. *Label leakage through returns*: using final net margin when the outcome window closes before
   the return tail does, which makes recent cycles look better. Detected by comparing gross and net
   outcomes by outcome age.

The first channel is the one that produced the most convincing wrong answer, and it produced it in
the most expensive way: a prototype with a beautiful offline score that failed immediately in
shadow mode because the winning feature was a post-purchase signal.

### 5e Reproducibility & Method-Selection Ledger

The ledger records, per cycle: cut-off timestamp, snapshot identifier, feature-build commit,
model artifact hash, policy version, threshold applied, budget, review capacity, and the operator
who approved publication. Replay means re-running the same snapshot with the same artifact and
obtaining the same selected set; anything else is a different decision.

A replay check is only meaningful if it is exact. The constructed cycle was replayed three times,
and the selected sets matched as sets on all three runs. They did not match in order, because the
greedy sort had no deterministic tie-break, which was fixed by sorting on `(value, member_id)`. That
defect would never have appeared in a metric; it would have appeared in an audit six months later.

Method selection is recorded with its reason, so that a later reader can see what was rejected and
why:

| Decision | Chosen | Rejected | Reason recorded |
|---|---|---|---|
| Target | Margin-weighted uplift | Propensity | Objective mismatch; see 5c |
| Learner | Two-model difference | Single model with treatment flag | Flag model entangles `p0` and `p1` |
| Calibration | Decile isotonic on holdout | Raw scores | Threshold applied to a score requires calibration |
| Selection | Greedy sort | 0-1 knapsack | Equal costs make the knapsack degenerate |
| Gate | Post-selection partition | Re-ranking | Gate must not reorder a value-optimal set |
| Frequency | Quarter cap of 2 | Cooldown only | Cooldown permits 6 contacts per quarter |
| Holdout | 10% of contacted band, household-level | Member-level 5% | Interference; see 4 |

Nothing in this case was executed against production data. The ledger exists to make the eventual
execution checkable, and every row in it is a decision that a reviewer can reverse on the record.

Takeaway: reproducibility is a property of the snapshot and the artifact together, and a replay
that matches as a set but not as a sequence is a defect worth fixing before an audit finds it.

## 6. Analysis Walkthrough

Six derivations follow. Each uses the same anatomy so that a reader can audit the arithmetic without
reconstructing the argument: the formula, where it applies, the constraint that bounds it, a
numerical example with intermediate values, and the interpretation. All inputs are ILLUSTRATIVE.

### 6.1 Net value per contact

```text
FORMULA
  V_i = m * delta_p_i - c

WHERE
  V_i        net expected value of contacting member-week i, dollars
  m          expected gross margin per incremental purchase, dollars (24.00)
  delta_p_i  estimated change in purchase probability caused by contact, proportion
  c          cost of making one contact, dollars (0.62)

CONSTRAINT
  The formula is valid only if delta_p_i is an estimate of the causal difference
  p1_i - p0_i, and only if c is the full marginal cost of the contact including
  the expected incentive payout. If delta_p_i is a propensity, the formula is
  undefined for the purpose of selection, because it omits the counterfactual.

NUMERICAL EXAMPLE
  Decile 3 member: delta_p = 0.060, m = 24.00, c = 0.62
  gross expected gain  = 24.00 * 0.060 = 1.440
  net value            = 1.440 - 0.620 = 0.820 dollars per member-week
  For 25,000 such member-weeks:
  total                = 25,000 * 0.820 = 20,500.00 dollars per cycle

INTERPRETATION
  A contact is worth making only when delta_p exceeds c / m = 0.62 / 24.00 = 0.02583,
  i.e. 2.58 percentage points. Below that threshold the funded offer transfers margin
  to members who would have bought anyway; above it, the contact creates margin.
```

The break-even threshold is the single most useful number in the case, because it converts an
abstract model output into an operational cut. It is also the number that moves when any of its
three inputs move, which is why it belongs in the monitoring dashboard rather than in a document.

### 6.2 Whole-cycle selection under a binding budget

The eligible population is 250,000 member-weeks in ten equal deciles of 25,000, ordered by estimated
uplift. The estimates below are constructed to be plausible for a grocery loyalty programme: a
strong head with high influence, a long tail with none.

```text
FORMULA
  V_total(S) = sum over i in S of ( m * delta_p_i - c )
  subject to |S| <= B

WHERE
  S     selected set of member-weeks
  B     contact budget, 120,000 contacts per cycle
  m     24.00 dollars per incremental purchase
  c     0.62 dollars per contact

CONSTRAINT
  Greedy selection by V_i is optimal only when every contact consumes exactly one
  unit of budget. If channel costs differ (SMS 0.11 versus email 0.06), the problem
  becomes a knapsack and the greedy order is no longer optimal.

NUMERICAL EXAMPLE
  decile  rows    delta_p   V_i        rows selected  value
  1       25,000  0.0910    1.5640     25,000         39,100.00
  2       25,000  0.0740    1.1560     25,000         28,900.00
  3       25,000  0.0600    0.8200     25,000         20,500.00
  4       25,000  0.0490    0.5560     25,000         13,900.00
  5       25,000  0.0400    0.3400     20,000          6,800.00
  6       25,000  0.0320    0.1480          0              0.00
  budget exhausted at 120,000 contacts, inside decile 5
  total value = 39,100 + 28,900 + 20,500 + 13,900 + 6,800 = 109,200.00 dollars

INTERPRETATION
  The threshold implied by the budget is not the break-even threshold. Break-even is
  2.583 points, which would select deciles 1-6 in full (150,000 rows). The budget
  instead cuts inside decile 5 at 4.00 points, so 30,000 rows that still carry
  positive value (5,000 from decile 5, all 25,000 of decile 6) are correctly left
  uncontacted. Two different numbers with two different causes: 2.583 is economics,
  4.000 is capacity.
```

The distinction between the economic threshold and the capacity threshold is where most programme
dashboards lose their audience. A chart showing "we contacted only 48% of eligible members" reads
as a failure; the same chart with the budget line drawn on it reads as a correct allocation.

### 6.3 Propensity ranking versus uplift ranking

This derivation isolates the `evaluation-mismatch` pattern with three constructed groups.

```text
FORMULA
  Loss from ranking on the wrong target
    L = V_total(S_uplift) - V_total(S_propensity)

WHERE
  S_propensity  top-B rows by p1 (probability of purchase given contact)
  S_uplift      top-B rows by m * (p1 - p0) - c
  B             120,000 contacts

CONSTRAINT
  The comparison is valid only when both policies face the same budget, the same
  eligible population, and the same cost. If the propensity policy also spends more
  per contact, part of the difference is budget, not ranking.

NUMERICAL EXAMPLE
  group  rows      p1     p0     delta_p   V_i
  S      60,000    0.55   0.50   0.0500    0.580
  N      90,000    0.22   0.14   0.0800    1.300
  L      100,000   0.03   0.03   0.0000   -0.620

  Rank by p1: take all 60,000 of S, then 60,000 of N
    value = 60,000 * 0.580 + 60,000 * 1.300 = 34,800 + 78,000 = 112,800
  Rank by V_i: take all 90,000 of N, then 30,000 of S
    value = 90,000 * 1.300 + 30,000 * 0.580 = 117,000 + 17,400 = 134,400
  Loss from propensity ranking = 134,400 - 112,800 = 21,600 per cycle

INTERPRETATION
  The propensity policy looks better on the metric it optimises: its response rate is
  38.5% ((60,000*0.55 + 60,000*0.22) / 120,000) against 30.3% for the uplift
  policy ((90,000*0.22 + 30,000*0.55) / 120,000). The worse policy wins the vanity
  metric and loses 21,600 dollars of margin per cycle, which is 1,123,200 per year.
```

This is the crux. A response-rate comparison would have shipped the propensity policy and celebrated
a 8.2-point improvement in response rate while destroying value. The metric did not merely fail to
detect the error; it pointed in the wrong direction, which is the defining property of an
evaluation-mismatch case rather than a mere measurement-error case.

### 6.4 Contact frequency under decaying lift

The cooldown constraint is often mistaken for a frequency policy. It is not, and the arithmetic
shows why.

```text
FORMULA
  delta_p_k = delta_p_1 * r^(k-1),  for the k-th contact within a quarter
  V_k       = m * delta_p_k - c

WHERE
  delta_p_1  lift of the first contact in the quarter, 0.060
  r          per-repeat retention of lift, 0.62 (ILLUSTRATIVE)
  k          contact index within the quarter, 1..6
  m, c       as before

CONSTRAINT
  The geometric decay is a local approximation. It assumes the member's response
  to the k-th contact does not depend on the creative used, and it assumes the
  quarter boundary does not itself reset behaviour. Both assumptions are testable
  and neither is tested here.

NUMERICAL EXAMPLE
  k=1: delta_p = 0.0600, V = 24*0.0600 - 0.62 =  0.8200
  k=2: delta_p = 0.0372, V = 24*0.0372 - 0.62 =  0.2728
  k=3: delta_p = 0.0231, V = 24*0.0231 - 0.62 = -0.0665
  k=4: delta_p = 0.0143, V = 24*0.0143 - 0.62 = -0.2768
  k=5: delta_p = 0.0089, V = 24*0.0089 - 0.62 = -0.4072
  k=6: delta_p = 0.0055, V = 24*0.0055 - 0.62 = -0.4881

  two-contact total   =  0.8200 + 0.2728 =  1.0928 per member per quarter
  six-contact total   =  1.0928 - 0.0665 - 0.2768 - 0.4072 - 0.4881 = -0.1458
  break-even decay r* satisfies 24 * 0.060 * r* = 0.62, so r* = 0.4306

INTERPRETATION
  With a 14-day cooldown a member can receive 6 contacts in a 91-day quarter. The
  six-contact schedule is net negative at the margin, and the two-contact schedule is
  optimal. The right frequency control is therefore a quarterly cap of 2, with the
  cooldown acting only as a floor on spacing. If the measured decay r falls below
  0.431, even the second contact stops paying and the cap becomes 1.
```

The boundary case is informative: at `r = 0.43` the second contact is exactly break-even, so a
programme whose measured decay is 0.40 should run one contact per quarter and no more. A programme
cannot discover this from a redemption dashboard, because redemptions keep accruing with every send
even while margin declines.

### 6.5 Minimum detectable effect for the validation holdout

An evaluation design that cannot see the effect it asserts is not an evaluation. The calculation
below establishes the resolution the holdout can deliver and, crucially, at what aggregation.

```text
FORMULA
  n_per_arm = 2 * (z_alpha/2 + z_beta)^2 * p(1-p) / delta^2

WHERE
  n_per_arm  rows required in each arm for a two-proportion comparison
  z_alpha/2  1.9600 for two-sided alpha 0.05
  z_beta     0.8416 for power 0.80
  p          baseline purchase probability in the window, 0.30
  delta      effect to detect, in proportion

CONSTRAINT
  The formula assumes independent rows. Member-weeks within a household are not
  independent, and repeated cycles for one member are not independent. The effective
  sample is smaller than the row count by the design effect.

NUMERICAL EXAMPLE
  (z_a + z_b)^2 = (1.9600 + 0.8416)^2 = 2.8016^2 = 7.8490
  p(1-p) = 0.30 * 0.70 = 0.2100
  numerator = 2 * 7.8490 * 0.2100 = 3.2966

  delta = 0.020 -> n = 3.2966 / 0.0004 =  8,241 per arm
  delta = 0.010 -> n = 3.2966 / 0.0001 = 32,966 per arm
  delta = 0.005 -> n = 3.2966 / 0.000025 = 131,863 per arm

  One decile has 25,000 rows. At a 20% holdout the control arm holds 5,000:
  delta resolvable = sqrt(3.2966 / 5,000) = sqrt(0.0006593) = 0.02568
  i.e. 2.57 percentage points, in a single cycle.

INTERPRETATION
  A single-cycle decile readout resolves 2.57 points. Decile 6 is estimated at 3.20
  points and is therefore just detectable; decile 7 at 2.50 points is not, and sits
  on the wrong side of the resolution limit even though it sits on the wrong side of
  break-even too. Pooling four cycles of the same decile gives 20,000 control rows and
  resolves 1.28 points, which is the smallest aggregation worth reporting.
```

The design effect is not hand-waved. With 1.35 member-weeks per household per cycle and an
intra-household correlation of 0.02 for the purchase outcome, the design effect is
`1 + (1.35 - 1) * 0.02 = 1.007`, which is negligible for the purchase probability and is *not*
negligible for the margin outcome, where household-level spend correlation is far higher. The
honest statement is that the purchase-probability readout is nearly unaffected and the dollar
readout should be treated as clustered.

Multiple comparisons matter here. Testing ten deciles at a nominal 0.05 gives roughly a 40% chance
of at least one false positive under the null, and the correction is not free:

```text
Bonferroni: alpha' = 0.05 / 10 = 0.005, so z = 2.8070
  (2.8070 + 0.8416)^2 = 3.6486^2 = 13.3123
  n per arm for delta = 0.020: 2 * 13.3123 * 0.21 / 0.0004 = 13,978
  versus 8,241 uncorrected, a factor of 1.70 in sample size
```

A programme that reports ten deciles with confidence intervals and no correction is reporting a
mixture of real and noise deciles with equal authority. The practical resolution used here is to
pre-register three coarsened bands (top, middle, bottom) rather than ten deciles, which restores
power without inviting a fishing expedition.

### 6.6 Margin-mix sensitivity: when the offer changes what is bought

The value formula assumed a single margin per incremental purchase. Offers change mix, so the
assumption fails exactly where the offer is most effective.

```text
FORMULA
  m_eff = basket_value * margin_rate_effective
  delta_p_required = c / m_eff

WHERE
  m_eff                effective margin per incremental purchase, dollars
  basket_value         80.00 dollars per purchase (ILLUSTRATIVE)
  margin_rate_effective blended margin of the incremental basket
  delta_p_required      break-even lift for the policy to add value

CONSTRAINT
  The formula holds only if the incremental purchase is a single basket at a single
  margin rate. When the offer shifts mix toward a low-margin promoted category, the
  blended rate falls and the required lift rises; the policy must then be re-solved
  rather than re-used with a new constant.

NUMERICAL EXAMPLE
  blended case:   m = 80.00 * 0.30 = 24.00  -> delta_p_required = 0.62/24.00 = 0.02583
  promoted case:  m = 80.00 * 0.22 = 17.60  -> delta_p_required = 0.62/17.60 = 0.03523
  ratio of thresholds = 0.03523 / 0.02583 = 1.364

  At the 24.00 margin, decile 6 (3.20 points) clears break-even and is worth contacting.
  At the 17.60 margin, decile 6 sits below the 3.52-point threshold and must be dropped,
  which removes 25,000 candidate contacts and 3,700 dollars of estimated value from the
  cycle; the freed budget moves to decile 5, not to decile 6.

INTERPRETATION
  A 26.7% fall in margin rate raises the break-even lift by 36.4%, because the
  threshold is inversely proportional to margin. The elasticity is exactly -1 in the
  ratio: a one percent fall in margin raises the threshold by slightly more than one
  percent. Programmes that fund deep discounts are therefore the most exposed to this
  error, and they are also the ones most likely to use a single blended margin rate.
```

The boundary case is a category with a margin rate of 9%, where the required lift becomes
`0.62 / (80.00 * 0.09) = 0.62 / 7.20 = 8.61` percentage points. No decile in the constructed
population reaches 8.61 points, so a 9%-margin offer should not be funded at all — a conclusion
that is invisible to any analysis that reports redemptions instead of margin.

### 6.7 Where the formulas break

Three limits are worth stating explicitly, because each is reachable in normal operation.

*Fixed cost per contact.* The break-even threshold assumes `c` is constant per contact. When SMS
costs 0.11 and email costs 0.06, the threshold differs by channel
(`0.11/24.00 = 0.458` versus `0.0258`), and the selection problem becomes a knapsack in which the
greedy order is no longer optimal. The practical fix is to solve per channel with a shared budget,
or to fix the channel by member preference before ranking.

*Non-linear value.* The formula is linear in `delta_p`, which fails when the offer has a
threshold structure ("spend $50 to earn 500 points"). Near the threshold, a small uplift can move a
member across a discontinuity, so the value is not proportional to the probability change. A
threshold offer must be valued with the distribution of basket values, not with its mean.

*Saturation and interference.* `V_i` is computed per member-week as if members were independent.
At a 48% contact rate within the eligible band, a contacted member's behaviour affects the
availability of promoted stock and the traffic in the store, which changes uncontacted members'
outcomes. The direction of the bias is toward understating the total effect and overstating the
per-contact effect, and the magnitude cannot be recovered from a member-level holdout.

Takeaway: the arithmetic is simple and each simplification has a named failure boundary, so a
reviewer can check whether the boundary is active rather than arguing about the conclusion.
## 7. The Finding

The finding has a primary claim, a mechanism, and a scope. Stating only the claim invites
over-generalisation; stating the mechanism makes it transferable.

**Primary claim.** For this decision, a ranking built on `m * (p1 - p0) - c` dominates a ranking
built on `p1` at the same contact budget, by 21,600 dollars per cycle in the constructed
population — a 19.1% improvement in value (134,400 versus 112,800) achieved with no additional
contacts, no new data source, and no change to the offer. The improvement comes entirely from
changing the target of the ranking.

The magnitude deserves a sanity check rather than awe. 19.2% is the gap between the two policies on
the constructed three-group population, where the "sure thing" group is 24% of all rows and the
"sleeping dog" group is 40%. In a population with fewer sure things, the gap shrinks; in a
population with more, it grows. The finding is directional and its size is population-dependent,
which is why the case reports the direction as the claim and the size as an illustration.

**Mechanism.** Propensity ranking allocates budget by the level of `p1`. Uplift ranking allocates by
the slope `p1 - p0`. Level and slope are uncorrelated in general and negatively correlated in
practice, because members with high purchase probability have less room to improve. Any programme
that spends a fixed budget on the highest-level members is therefore systematically funding the
members it does not need to fund. This is not a modelling defect that a better model fixes; it is a
consequence of the objective the model was trained to hit.

The mechanism has an observable signature, which is what makes it a finding rather than an opinion:
the selected population's *realised* response rate rises while the selected population's *realised
incremental* margin falls. Both metrics move in opposite directions, and the vanity metric is the
one that appears first on the dashboard.

**Scope.** The claim holds when all of the following are true, and each condition is checkable:

1. The budget binds, so a truncation must happen somewhere (at 120,000 of 250,000 it does).
2. Contact cost is positive and uniform, so a break-even threshold exists and is the same for all
   rows (true within a channel, false across channels).
3. The uplift estimate is at least ordinally accurate across the selected band; it need not be
   calibrated in level, but a badly miscalibrated estimate that inverts the top decile would
   destroy the advantage.
4. The holdout used to train `p0` is a genuine random sample of the eligible band, not a sample of
   previously contacted members.
5. No interference strong enough to make one member's treatment change another's outcome at a scale
   that reorders the top deciles.

Condition 5 is the one the case cannot verify, and it is stated as a limit rather than resolved. At
a 48% contact rate, interference is plausible; the direction of its effect on the ranking is toward
compressing differences between members in the same store catchment, which would flatten the value
curve and reduce the advantage of uplift ranking without eliminating it.

**What the finding is not.** It is not a claim that the uplift model is accurate in dollars. The
estimated 105,360-cycle value in section 6.2 depends on `delta_p` values that are constructed. It is
not a claim that the policy was deployed. Nothing was deployed. It is not a claim that the retailer
improved margin; the retailer does not exist.

**The secondary finding** is about the evaluation gate itself and is arguably more useful than the
primary one. The original acceptance criterion for the programme was "response rate at least 3
points above the rules queue." That criterion is satisfiable by the worse policy: the propensity
policy delivers 38.5% against the rules queue's constructed 26.4%, a 12.1-point improvement, and
destroys 21,600 dollars per cycle while doing it. A gate that a dominated policy can pass is not a
gate.

```text
FORMULA
  Gate pass rate for a dominated policy
    pass(S_propensity) = 1 iff  response_rate(S_propensity) - response_rate(baseline) >= 3 points

WHERE
  response_rate(S) = sum over i in S of p1_i / |S|
  baseline rules queue response rate = 26.4% (ILLUSTRATIVE)

CONSTRAINT
  The comparison is only meaningful at equal budget. A policy that contacts fewer members
  can beat a larger policy on rate while losing on total value.

NUMERICAL EXAMPLE
  propensity policy rate = (60,000*0.55 + 60,000*0.22) / 120,000 = 46,200/120,000 = 0.3850
  baseline rate          = 0.2640 (constructed: the rules queue concentrates on mid-recency rows)
  margin over baseline   = 38.50 - 26.40 = 12.10 points, which passes a 3-point gate
  value                  = 112,800 versus uplift policy 134,400, a 21,600 shortfall
  The gate passes the policy that loses 1,123,200 dollars per year.

INTERPRETATION
  A response-rate gate is not a weak proxy for margin; it is a proxy with the opposite sign in
  the region that matters. The corrected gate is a margin-per-contact comparison at equal
  budget, with a holdout, and the response rate demoted to a diagnostic.
```

That arithmetic is the case's real lesson and the reason it belongs in an `evaluation-mismatch`
collection rather than a modelling collection. Changing the model would not have helped; changing
the acceptance criterion would have.

**A third, quieter finding** concerns the frequency control. The cooldown of fourteen days permits
six contacts per quarter, and at the constructed decay of 0.62 the third through sixth contacts are
margin-negative. The programme's own policy documents treated the cooldown as the frequency policy,
which meant the operating rule permitted 4.4 contacts per member per quarter on average among
contacted members while the economically correct cap was 2. The gap between the permitted maximum
and the profitable maximum is a policy defect that no model metric would surface.

Takeaway: the finding is a target change plus a gate change plus a frequency cap, and each of the
three is independently checkable against the same arithmetic.

## 8. System/Solution Design

The design is a batch decision pipeline with four gates, an append-only log, and a measurement path
that runs on the same snapshot as the decision.

```text
[event tables] -> [as-of feature build] -> [uplift scoring] -> [value sort]
                          |                     |                 |
                          v                     v                 v
                   [quality gate Q1-Q6]   [calibration gate]  [eligibility gate]
                                                                      |
                                                                      v
                                                            [review partition]
                                                              |            |
                                                     [auto-send]      [human queue]
                                                              |            |
                                                              +-----+------+
                                                                    v
                                                          [append-only decision log]
                                                                    |
                                                                    v
                                                     [holdout + outcome join at T+42d]
```

Each box has a contract, and the contracts are what make the diagram more than a picture.

**Stage 1 — As-of feature build.** Inputs are read with `effective_ts <= cut_off - 15 min`. Output
is a feature table keyed by `member_id, decision_cycle`, plus a snapshot identifier. Contract: the
same snapshot identifier must reproduce the same feature table byte-for-byte, and the build must
fail closed if any feature's definition window overlaps the outcome window.

**Stage 2 — Scoring.** Two models produce `p1` and `p0`; the difference is the estimate. Output adds
`p1`, `p0`, `delta_p_raw`, `delta_p_clipped`, and a `model_version`. Contract: no row may be scored
twice with different artefacts in the same cycle, which is enforced by a unique constraint on
`(member_id, decision_cycle, model_version)`.

**Stage 3 — Value sort.** `V = m * delta_p_clipped - c`, sorted descending with a deterministic
tie-break on `member_id`. Contract: the sort must be total, which requires the tie-break; a partial
order produces non-reproducible selections at the budget boundary and did, in the constructed
replay.

**Stage 4 — Eligibility gate.** Consent, suppression, cooldown, conflicting-experiment, and
offer-funding checks. Contract: this gate can only remove rows, never add them, and every removal
writes a reason code. A gate that can add rows is not a gate.

**Stage 5 — Review partition.** Rows above the policy envelope (estimated value above $2.00 per
contact, or any offer with a face value above $10.00, or any member with an open service case) are
routed to the human queue, capped at 2,400. Rows within the envelope auto-send. Contract: the
partition does not reorder the selected set; a reviewed row that the reviewer rejects becomes an
abstention with a reason code, and the freed budget is *not* refilled in the same cycle, because
refilling requires a re-sort and would break the published decision.

That last contract is a real design decision with a cost. Not refilling wastes budget — in the
constructed cycle, 180 of 2,400 reviews are rejected, which leaves 180 contacts unused, worth about
$0.82 × 180 = $147.60 of forgone value. Refilling would recover $147.60 and would make the
published decision mutable after the fact, which breaks the audit trail. The $147.60 is the price of
immutability, and it is cheap.

**Stage 6 — Decision log.** Append-only, one row per decision, including abstentions. Fields:
`decision_id`, `member_id`, `decision_cycle`, `snapshot_id`, `feature_hash`, `model_version`,
`p1`, `p0`, `delta_p_clipped`, `V`, `rank`, `action`, `reason_code`, `reviewer_id`, `review_ts`,
`send_ts`. Contract: no update, no delete; a correction is a new row with a `supersedes` pointer.

**Stage 7 — Measurement.** At T+42 days the outcome join produces gross margin, net margin, return
rate, and contact cost per row, joined to the holdout flag. Contract: the outcome join may not read
the decision log's `action` field for any row where the holdout flag is set, because a holdout by
definition received no contact and a join that conditions on action would silently drop them.

The failure and recovery paths are part of the design, and each has a specific behaviour:

| Failure | Detection | Behaviour | Recovery |
|---|---|---|---|
| Snapshot missing | Snapshot id not resolvable | Cycle falls back to rules queue | Rebuild from source; re-run |
| Freshness breach (Q2) | Max feature age > 26h | Degrade to baseline, log degraded flag | Backfill; next cycle normal |
| Score range violation (Q4) | Any `p1` outside [0,1] | Quarantine those rows only | Investigate model artifact |
| Offer withdrawn mid-cycle | `offer_id` inactive at send | Exclude offer, keep content-only | Re-issue with replacement offer |
| Review backlog > capacity | Queue depth > 2,400 at 04:00 | Auto-approve within envelope only | Raise cap or narrow envelope |
| Send job partial failure | Sent count < selected count | Log per-row send status | Retry idempotently by `decision_id` |
| Late correction to source | Correction after cut-off | Do not rewrite; next cycle uses it | Note in cycle close-out |

The recovery path for a partial send failure deserves emphasis because it is the one place where
idempotency is safety-critical. The send job is keyed by `decision_id`, so a retry after a broker
failure cannot double-send; without that key, a retry storm sends the same offer two or three times
to the members in the failed segment, which is both a cost event and a complaint event.

The design also has an explicit "do less" mode, which is what a mature pipeline looks like. If any
of the following hold, the cycle publishes the rules queue instead of the ranked selection, and the
decision log records `mode = degraded` with the triggering condition:

- calibration drift beyond the pre-registered band in two consecutive cycles,
- more than 2% of rows quarantined by Q4,
- review capacity below 1,500 rows,
- a live pricing or supply event that invalidates the offer's margin assumption.

The degraded mode exists because the alternative — shipping a ranking that the team no longer
believes — is worse than shipping an older policy that everyone can explain. It is also what allows
the programme to survive a bad month without a public reversal.

Two design options were rejected, and the reasons are worth recording because they will be proposed
again.

*Rejected: online learning.* The decision runs 52 times per year and the outcome matures in 42 days,
which gives at most 8 fully observed cycles per year. An online loop with 8 updates per year is a
batch retrain with extra failure surface. The specific test that would reopen this: if the cycle
cadence moved to daily with a 7-day outcome window, the update count rises to 52 per year and the
trade-off changes.

*Rejected: per-member threshold optimisation.* Solving a separate threshold per member is
mathematically equivalent to sorting by value when costs are equal, and it adds a parameter per
member that cannot be estimated. The specific test: if contact costs varied by member (for example
by channel preference), the problem becomes a genuine knapsack and the per-member formulation
becomes necessary.

Takeaway: the design's value is in its gates and its log, not in its model, and every gate can only
subtract rows.

## 9. Evaluation & Validation

The evaluation contract is stated before the result, because a metric defined after seeing outcomes
is a description, not a test.

| Element | Definition |
|---|---|
| Numerator | Gross margin from purchases in [cut-off, +28d], net of returns processed to +42d, minus contact cost |
| Denominator | Eligible member-weeks in the decision cycle (250,000), not contacted rows (120,000) |
| Population | Members active in the trailing 90 days, consenting, not suppressed, outside cooldown |
| Window | Decision cycle plus 28-day purchase window plus 42-day return tail |
| Comparator | Rules queue at the same budget on the same population, and a 10% household holdout |
| Guardrails | Complaint rate, unsubscribe rate, review rejection rate, contact-cost variance, frequency cap violations |
| Uncertainty | Holdout difference with a cluster-robust interval at household level; decile bands coarsened to 3 |
| Status | Designed and specified; **not executed**. No production cycle was run. |

The denominator choice is the one most often got wrong, and the arithmetic shows why it matters.
Using contacted rows as the denominator measures the value of a contact; using eligible rows
measures the value of the *policy*, which is what the owner is deciding. The two differ by the
factor 250,000/120,000 = 2.083, so the uplift policy's $1.12 per contact is $0.5376 per eligible
member-week, while the propensity policy's $0.94 per contact is $0.4512 — and only the second pair
of numbers answers the owner's question.

**Guardrails with thresholds.** A guardrail without a threshold is a wish. Each guardrail below has
a limit and a consequence.

| Guardrail | Constructed baseline | Limit | Consequence on breach |
|---|---|---|---|
| Complaint rate per 10,000 contacts | 3.1 | 6.0 | Pause channel, review creative |
| Unsubscribe rate per cycle | 0.42% | 0.60% | Reduce frequency cap to 1 |
| Review rejection rate | 7.5% | 15.0% | Narrow policy envelope |
| Contact cost per send | 0.62 | 0.75 | Re-solve threshold at new cost |
| Frequency cap violations | 0 | 0 | Block cycle; fix filter |
| Offer honour failures | 0.9% | 2.0% | Suspend offer at affected stores |

The complaint-rate guardrail is the one that has actually bound in comparable programmes, and it is
worth understanding why it binds before the margin metric does. Complaints rise with contact volume
and with offer relevance failure; a policy that maximises margin on a poorly calibrated uplift
estimate will concentrate contacts on members with high estimated lift and low realised lift, which
is precisely the population that complains about irrelevant offers. Margin and complaints are
therefore positively correlated in the failure direction, not opposed.

**Validation design.** Three stages, in increasing cost, each with a decision it can support.

1. *Offline replay.* Score historical cycles with the current artefact, select under the current
   policy, and compare to the rules queue's actual selection on the same rows. This detects
   implementation error and policy infeasibility, and it cannot detect target error, because the
   replay uses the same `delta_p` estimates that the policy was built on. It is a regression test,
   not evidence of value.
2. *Shadow cycle.* Run the pipeline live, log decisions, send the rules-queue selection, and compare
   the two on realised outcome at T+42. This detects target error partially: the shadow policy's
   selected set differs from the sent set, so the comparison is confounded by selection. Its real
   value is operational: it exercises every gate with live data and finds the failures that only
   occur on real inputs.
3. *Randomised holdout.* Within the contacted band, withhold contact from a 10% household-level
   random sample. This is the only stage that identifies the effect, and it is the only stage that
   can support a margin claim.

The holdout has a cost that must be stated honestly. Withholding 10% of 120,000 contacts forgoes
12,000 contacts, worth 12,000 × 0.82 = $9,840 per cycle at the decile-3 value, or $511,680 per
year. That is the price of identification, and it is 45.6% of the annual value the propensity
policy destroys (1,123,200). The holdout is expensive and it is still correct, because
without it the programme cannot distinguish a working policy from a lucky one, and the decision to
continue spending $3.87 million per year on contacts would rest on a response-rate dashboard.

**Power.** Section 6.5 established that a single-cycle decile readout resolves 2.57 points while the
policy's claimed effect in the marginal decile is around 3.2 points. The validation plan therefore
promises less than the policy claims: the holdout can confirm the effect in the top three bands
pooled over four cycles, and it cannot confirm it decile by decile. That asymmetry is stated in the
pre-registration so that a later reader does not mistake an unpowered null for a refutation.

**Not executed.** Nothing in this section ran. The status of every element is `designed`. The
honest summary is that the case specifies what a credible evaluation would look like, gives the
arithmetic that determines its resolution and its cost, and makes no claim about what such an
evaluation would find.

## 10. Action Taken

No production action was taken. The work products are a specification, a rejected prototype, a
design, and a measurement plan. This section distinguishes what was decided from what was done,
because a case that blurs the two is the failure mode the collection exists to prevent.

| # | Action | Owner | Status | Evidence produced |
|---|---|---|---|---|
| 1 | Audit the propensity prototype for leakage | Analytics | completed (constructed) | Timestamp audit; one post-purchase feature removed |
| 2 | Rebuild the target as a two-model difference | Analytics | completed (constructed) | Feature ledger; snapshot ids |
| 3 | Add the six quality gates | Data engineering | completed (constructed) | Gate thresholds and block/flag behaviour |
| 4 | Add the append-only decision log | Platform | designed | Field list and contract |
| 5 | Define the review envelope and capacity | Loyalty operations | designed | 2,400 rows, 2.4 minutes per review |
| 6 | Pre-register the holdout and its analysis | Analytics + finance | designed | 10% household holdout, 3 bands, 4 cycles |
| 7 | Recompute the margin vector by category | Finance | not started | — |
| 8 | Run the shadow cycle | Platform | not started | — |

The completed items are constructed: this case describes what a programme would have to do, and the
"completed (constructed)" status means the work is described rather than evidenced. Items 7 and 8
are the two that would actually change the decision, and neither was done.

Item 7 matters more than it looks. The whole selection rests on a single blended margin rate of
30%, and section 6.6 shows that a 22% rate in the promoted category raises the break-even threshold
by 36.4% and removes a decile from the selected set. Until the margin vector exists, the selected
set has an uncertainty band that is wider than the difference between the top two policy candidates.

Item 8 is the gate to everything else. A shadow cycle costs one cycle of analyst time and no
customer contact, and it is the cheapest way to discover that the feature build cannot run inside
the six-hour window between cut-off and send.

**Decisions that were made, with their rationale**, because these are the parts a successor
inherits:

- Keep the rules queue as the comparator and the degraded-mode policy, rather than retiring it. A
  comparator that is deleted cannot be used to detect regression.
- Clamp `delta_p` at zero for selection but report it unclipped. The clamp prevents negative-value
  rows entering a tie-break; the unclipped report preserves the sleeping-dog signal, which is a
  finding in its own right.
- Cap frequency at two per quarter rather than relying on the cooldown. The cooldown is a spacing
  rule; the cap is an economics rule, and only the second one is derived from the decay.
- Do not refill budget after review rejections. Immutability of the published decision is worth more
  than $147.60 per cycle.
- Keep the response rate on the dashboard as a diagnostic and remove it from the acceptance gate.

The last decision is the one that generated the most internal resistance, and the objection is
worth recording because it is reasonable: response rate is the metric the programme has reported for
three years, it is understood by every stakeholder, and replacing it with margin per eligible
member-week makes the weekly report harder to read. The counter-argument is that a metric which
rewards the dominated policy is not a simplification, it is a decision to lose money legibly.

## 11. Outcome & Results

**Status: `not-measured`.** No cycle was executed, no holdout was drawn, and no margin was realised.
The only results in this case are arithmetic identities that hold under stated assumptions.

What is established, and how firmly:

| Claim | Basis | Firmness |
|---|---|---|
| Break-even lift is 2.583 points at m = 24.00, c = 0.62 | Arithmetic | Exact under the stated inputs |
| Uplift ranking beats propensity ranking by 21,600 per cycle | Constructed group means | Illustrative; sign is the claim |
| Response-rate gate passes the dominated policy | Arithmetic on the same group means | Exact given the constructed rates |
| Two contacts per quarter is optimal at r = 0.62 | Geometric decay arithmetic | Illustrative; depends on r |
| A 10% single-cycle holdout resolves 2.57 points | Standard two-proportion formula | Exact given alpha, power, p |
| The selected set is feasible at budget and capacity | Counts | Exact given the counts |

What is not established, and would require execution:

- whether measured uplift in this population is positive at the margin,
- whether the two-model estimate ranks correctly out of sample,
- whether the review gate's rejection rate is 7.5% or materially different,
- whether complaints rise with volume at the modelled slope,
- whether interference compresses the value curve enough to reorder the top decile.

**The counterfactual question.** A reader is entitled to ask what the programme would have earned
had it run the propensity policy. The answer is the constructed $112,800 per cycle, which is
$5,865,600 per year, and it is not a forecast. The reason to report it is comparative: the
difference between the two policies, $1,123,200 per year, is the annual cost of using the wrong
target, and that number is what justifies the work.

A second counterfactual is more useful for a practitioner: what would the programme have earned by
doing nothing? At zero contacts, the value is zero by construction, but that is an artefact of
measuring only incremental margin. The programme also produces retention, data, and redemption
behaviour that the rules queue maintained, so "do nothing" is not a neutral baseline; it is a
different programme with unmeasured effects. The case keeps the rules queue as the baseline for
exactly this reason.

**What the results would look like if the policy were wrong.** The value of a case is partly in its
falsifiability, so the expected signature of failure is specified:

- If the holdout shows lift indistinguishable from zero in the top band across four cycles, the
  estimate is noise and the correct policy is the rules queue with the budget cut by half.
- If lift is positive but falls below 2.583 points in every band, the offer is under-funded and the
  correct response is a larger face value or a different incentive, not a better model.
- If lift is positive and large in the top band but the complaint rate breaches 6.0 per 10,000, the
  policy is buying margin with relationship damage and the frequency cap must fall to one.
- If lift is positive in the middle bands and negative in the top band, the ranking is inverted at
  the head — the signature of a leakage defect or a `p0` model trained on contacted rows only.

Takeaway: the case's output is a testable operating rule with a stated invalidation signature, and
its results are arithmetic rather than measured, which the status line must continue to say.
## 12. Failure Modes & What Went Wrong

Five mechanisms produced wrong answers in this programme's constructed history. Each is stated with
its symptom, the causes that compete with it, the check that discriminates between them, and the
correction with its own cost. The point of the table is that several distinct causes produce the
same symptom, so a symptom alone never identifies a cause.

### 12.1 Post-purchase signal in the feature set

**Symptom.** The prototype's offline AUC reached 0.91 against a rules-queue AUC of 0.68, and the top
decile's estimated lift was 14.2 points, which is implausible for a grocery offer with a $7 face
value. The lift of a 10% discount on an $80 basket is bounded by the discount's value, so a
14.2-point lift on a 30% margin implies $0.142 × 24.00 = $3.41 of value against $0.56 of incentive
cost; possible, but not at that scale across a whole decile.

**Competing causes.** (a) a genuine strong feature, such as category affinity measured over two
years; (b) a labelling error that marks non-buyers as buyers; (c) a post-purchase feature that
encodes the outcome; (d) a population mismatch in which the evaluation rows are a different,
higher-intent cohort than the scoring rows.

**Discriminating check.** Compute the feature's definition window and compare it against the
cut-off, per feature, mechanically. In the constructed audit, the offending feature was
`days_since_redemption_of_this_offer`, whose window opened at send and closed at redemption — an
outcome, not a predictor. The check that separates (c) from (a) is the window overlap test; the
check that separates (c) from (b) is a label audit on 200 rows; the check that separates (d) is a
row-count reconciliation between the evaluation and scoring populations.

**Correction.** Remove the feature and rebuild. Cost: the AUC fell to 0.71, which felt like a
regression and was in fact the removal of an illusion. The residual risk is that other features are
partially contaminated and pass the window test because their windows are wide enough to straddle
the cut-off; the mitigation is to require an explicit, narrow definition window per feature rather
than a wide one that happens to be mostly safe.

**Verification.** Re-run the replay and confirm that the top-decile estimate falls into the range
implied by the incentive's face value and the margin vector. A useful sanity bound: a decile
estimate above `face_value / (basket_value × margin_rate)` — here 7.00/24.00 = 0.292 — warrants an
explanation before it is believed.

### 12.2 Control rows absent from training

**Symptom.** The `p0` model's predictions clustered around 0.31 regardless of features, and the
resulting `delta_p` was almost a monotone transform of `p1`, so the uplift ranking looked like a
slightly noisy propensity ranking. The policy's advantage over the rules queue was 3% rather than
the 19% expected.

**Competing causes.** (a) insufficient model capacity; (b) a target with little signal; (c) training
on treated rows only, so `p0` was extrapolated from a population that never appears untreated;
(d) a feature set dominated by one variable that swamps the rest.

**Discriminating check.** Count control rows per feature decile. If the control count is below 500
in any decile, the `p0` estimate in that decile is not identified and the ranking cannot be trusted
there. In the constructed history, the control arm had 5,000 rows total but only 340 in the top
decile, because the 5% holdout was drawn from the *contacted* band, which by construction excludes
the members who were never selected.

**Correction.** Draw the holdout from the eligible population, not from the contacted band, and
train `p0` on all held-out rows. This changes the holdout's cost: withholding from the eligible
population means some members who would never have been contacted are also held out, which
increases the withheld count for the same identification power and reduces the programme's ability
to compare at equal selection. The net effect is a larger holdout (14% of eligible rather than 10%
of contacted) for the same 10% of the contacted band's identification.

**Verification.** Re-fit and confirm that `p0` varies with features, measured by the standard
deviation of `p0` across rows; a near-constant `p0` is the signature of the defect returning.

### 12.3 Budget truncation without a deterministic tie-break

**Symptom.** Three replays of the same snapshot produced selected sets of 120,000 rows that matched
as sets but differed in membership by 41 rows at the boundary. The audit found 41 members who
received the offer in one replay and not in another.

**Competing causes.** (a) a genuine non-determinism in the model (for example, thread-order-dependent
floating-point summation); (b) a sort without a total order; (c) an unstable input, such as a table
read without a fixed snapshot; (d) an eligibility filter that depends on wall-clock time.

**Discriminating check.** Compare the set difference between two replays and inspect the value of
the differing rows. If all differing rows share an identical `V` to the displayed precision, the
cause is (b): a tie at the budget boundary. If differing rows have different `V`, the cause is (a) or
(c). If the differences cluster on rows whose eligibility depends on a time window, the cause is (d).

**Correction.** Sort on `(V DESC, member_id ASC)`. Cost: none, except that the tie-break is arbitrary
and will systematically favour low member identifiers. That arbitrariness is acceptable because it
is deterministic and because a tie means the two rows are economically equivalent at the estimated
precision; it would not be acceptable if `member_id` correlated with a protected attribute, which is
a checkable claim and was checked.

**Verification.** Replay twice and assert an empty set difference, not merely equal counts. Equal
counts with unequal membership is the exact failure this check exists to catch.

### 12.4 Cooldown filter applied after ranking

**Symptom.** 1,860 members received two contacts within eight days, and the complaint rate rose from
3.1 to 4.4 per 10,000 in the affected weeks. The model metrics did not move.

**Competing causes.** (a) the cooldown filter running after the budget cut, so members removed by
the filter were not replaced and the count fell short — this would show as an under-send, not an
over-contact; (b) the contact log being read with a stale snapshot, so recent contacts were
invisible; (c) the cooldown being applied per channel rather than per member, so an email and an
SMS eight hours apart counted as two separate first contacts in different channels; (d) a timezone
error in the day difference.

**Discriminating check.** For each double-contact pair, compute the channel pair and the gap. If the
pairs are predominantly cross-channel, the cause is (c). If the gaps cluster at exactly 14 days in
one direction and 13 in the other, the cause is (d). If the gaps are uniformly short (1–7 days) and
the contact log shows the earlier contact, the cause is (b).

In the constructed history the pairs were 92% cross-channel, which identified (c): the cooldown
predicate was `days_since_contact_by_channel >= 14`, and the intended predicate was
`days_since_any_contact >= 14`.

**Correction.** Move the cooldown to the *pre-ranking* filter and make it channel-agnostic. Cost: the
eligible population fell by 8.4% (from 273,000 to 250,000 pre-filter rows), which reduced the budget's
slack and raised the effective threshold from 3.85 to 4.00 points.

**Verification.** Assert zero rows selected with `days_since_any_contact < 14` in the decision log
before the send job starts, and alert on any non-zero count as a blocking condition rather than a
warning.

### 12.5 Review gate that reordered the selected set

**Symptom.** The reviewed 2,400 rows were sent in a different order than the ranked order, and 180
rejections were replaced by the next-ranked rows within the same cycle. Total contacts stayed at
120,000, but the cycle's realised margin was 6% below the estimate in the reviewed segment.

**Competing causes.** (a) an estimation error in the reviewed segment; (b) the reviewer's rejections
being correct and the model over-estimating those rows, so removing them should have *raised* the
average, not lowered it; (c) the refill drawing from a lower-value tranche and being recorded as if
it were part of the original selection; (d) a send-order effect in which later sends land after the
offer's peak relevance.

**Discriminating check.** Split the realised margin by (reviewed, replaced, unreviewed) and compare
each against its own estimate. If the replaced rows underperform their estimate and the reviewed
rows perform at their estimate, the cause is (c): the refill was drawn from a tranche whose estimate
was computed under a different budget constraint, so it was never valid at the margin.

**Correction.** No refill. A rejection becomes an abstention with a reason code, and the freed budget
is carried into the next cycle as a recorded credit rather than spent within the current one. Cost:
180 unused contacts per cycle at $0.82 of forgone value each, or $147.60.

**Verification.** Reconcile `selected_rows = sent_rows + abstained_rows + failed_rows` per cycle, and
assert that no row carries both a `replaces` pointer and a `selected` flag in the same cycle.

### 12.6 What the failure list teaches

The five mechanisms share a shape that is worth naming, because it is the transferable part.

Each one produced a metric movement that was *consistent with success*: higher AUC, a plausible
`p0`, an unchanged count, a full budget, a stable contact volume. In no case did a monitor fire,
because each monitor was watching the quantity the defect preserved.

- 12.1 preserved nothing and inflated everything, and the only guard was a bound on plausibility.
- 12.2 preserved the row count and destroyed the variance of the counterfactual.
- 12.3 preserved the count and destroyed reproducibility.
- 12.4 preserved the count and destroyed the frequency invariant.
- 12.5 preserved the count and destroyed the value ordering.

The general lesson: **count-based monitors cannot detect ordering, target, or invariance defects.**
A count is preserved by every one of these failures. The checks that do work are structural
assertions — unique keys, total sorts, channel-agnostic predicates, non-constant counterfactual
variance, and no-refill reconciliation — and each is cheap to write and impossible to satisfy
accidentally.

## 13. Transferable Lesson

The transferable content of this case is a five-question test that decides whether a ranked-action
problem is the same problem as this one. If all five answers match, the method transfers; if any
answer differs, the method must be re-derived rather than re-used.

1. **Is the objective a difference rather than a level?** If value comes from changing behaviour,
   the target must be a contrast between two potential outcomes. A propensity, a score, a
   likelihood, or an intent signal answers a different question and will concentrate budget on the
   population with the least headroom.
2. **Does the budget bind, and is the cost uniform?** A binding budget with uniform cost makes the
   problem a greedy sort with a threshold. Uniform cost with slack budget makes the threshold the
   only decision. Non-uniform cost makes it a knapsack, and the greedy order stops being optimal —
   a change that is easy to miss because the code still runs.
3. **Is the effect identified by a holdout drawn from the eligible population?** A holdout drawn
   from the contacted band cannot estimate what would have happened to members who were never
   selected, which is exactly the population the policy is choosing between.
4. **Is there a frequency dimension that the constraint set does not capture?** A cooldown is a
   spacing rule; profitability is a cap. Whenever contacts have decaying lift, the cap is derived
   from the decay and is not a policy preference.
5. **Can the acceptance gate be passed by a dominated policy?** If yes, the gate measures the wrong
   quantity, and the correction is a gate change rather than a model change.

The questions are ordered by how expensive the mistake is to discover. Question 1 is cheap to check
and catastrophic to get wrong; question 5 is cheap to check and merely expensive.

Three domains outside retail loyalty satisfy the same five answers and reuse the arithmetic directly.

- *Subscription retention offers.* The unit is account-month, the budget is a fixed number of
  discount months, and the outcome is retained revenue net of discount. The break-even lift is
  `discount_cost / (monthly_margin × remaining_life)`, which has the same structure and a different
  parameterisation. The frequency dimension becomes a "saves attempted per account per year" cap.
- *Collections and payment-plan outreach.* The unit is delinquent account-month, the action is a
  reminder or a plan offer, and the outcome is recovered cash net of contact cost. The interesting
  difference is that the "sleeping dog" population is large: a reminder can accelerate a payment
  that would have arrived anyway, so the measured effect on *timing* differs from the effect on
  *recovery*, and the case's single-outcome framing fails.
- *Clinical or care-management outreach.* The unit is patient-month, the action is a call or a
  message, and the outcome is an avoided event. The same formula applies, and the ethical constraint
  is stronger: a policy that withholds contact to identify an effect is withholding care, so the
  holdout must be justified as equipoise rather than as measurement economy.

Two domains look similar and fail question 1 outright, which is why they are named.

- *Churn scoring for a sales team.* The action is a human call with a high and variable cost, the
  budget rarely binds in the same way, and the outcome is a renewal that often happens without
  intervention. A propensity ranking there is defensible only if the sales team's capacity is the
  constraint and the call's content is fixed; otherwise the same critique applies and the same
  correction is needed.
- *Recommendation ranking on a website.* The unit is a session-impression, the cost per exposure is
  nearly zero, and the budget does not bind, so the greedy-with-threshold structure is irrelevant.
  Uplift estimation is still useful, but the selection problem is a ranking problem, and the
  constraint set is entirely different.

The transferable artefact is therefore not the pipeline diagram. It is the pairing of a target
(contrast, not level), a threshold (`c/m`, recomputed whenever cost or margin moves), and a gate
(a margin comparison at equal budget with a holdout), plus the five questions that decide whether
the pairing applies.

## 14. Connections & Cross-References

| Relation | Target | Why it matters here |
|---|---|---|
| Prerequisite | `Cheat_Sheets/03_Fulfilment_Promise_Metrics.md` | Defines the metric-frozen-before-outcome discipline used in §9 |
| Prerequisite | `Cheat_Sheets/02_Customer_Identity_Controls.md` | Household identity determines the holdout unit in §9 |
| Sibling case | `Case_Studies/01_BOPIS_Promise_Accuracy.md` | Same evaluation-mismatch shape with a promise-accuracy metric |
| Sibling case | `Case_Studies/03_Returns_Risk_Intervention.md` | Same shape with a returns-risk target and a service-recovery gate |
| Sibling case | `Case_Studies/04_Cross_Channel_Attribution.md` | Supplies the channel-split margin vector this case assumes |
| Contrast | `Case_Studies/05_Omnichannel_Fulfilment_Allocation.md` | Analytical allocation rather than predictive targeting; constraints dominate |
| Theory | `Cheat_Sheets/01_Omnichannel_Event_Taxonomy.md` | Event time, decision time, and outcome time used in §2 and §5a |

The nearest conceptual neighbour is the BOPIS case, because both fail the same way: a convenient
proxy (promise accuracy, response rate) is optimised while the operational objective (a kept promise
that the customer trusted, incremental margin) declines. The difference is the correction: BOPIS
corrects by redefining the label at the promise level, while this case corrects by redefining the
target at the counterfactual level.

### Interview pack

Sixteen questions, mixed by type: recall (Q1, Q11), calculation (Q2, Q3, Q5, Q7, Q9, Q13),
diagnosis (Q4, Q8, Q10, Q14), and judgment (Q6, Q12, Q15, Q16). Each has its own answer; the
reasoning distinguishes a correct answer from a memorised one.

#### Q1 — What is the decision unit here, and why not the member?

**Answer.** The unit is the member-week: one row per member per weekly decision cycle. The member is
not the unit because a member can be eligible in consecutive cycles, so a member-level policy has no
way to express frequency, and the cooldown and quarterly cap both act on the sequence of decisions
rather than on the person.

**Reasoning.** Unit choice fixes the denominator of every later metric. At member grain, the budget
constraint becomes a headcount and the same member consuming two contacts looks like two members
consuming one each; the contact cost per unit is then understated by the repeat rate.

**Follow-up.** Ask what changes if the cadence moves from weekly to daily. Expected answer: the unit
becomes member-day, the eligible population multiplies by roughly seven while the budget does not,
the threshold rises sharply, and the frequency cap becomes the binding constraint rather than the
budget.

**Red flag.** Answering "the customer" without noticing that the budget is denominated in contacts
per cycle. Scoring: full credit names the unit, the repeat structure, and the effect on the
denominator; partial credit names the unit only.

#### Q2 — Compute the break-even lift for a $12 face value at 8% redemption, 30% margin, $80 basket.

**Answer.** Contact cost is `0.06 + 0.08 × 12.00 = 0.06 + 0.96 = 1.02`. Margin per incremental
purchase is `80 × 0.30 = 24.00`. Break-even lift is `1.02 / 24.00 = 0.0425`, i.e. 4.25 percentage
points, up from 2.583 points at a $7 face value.

**Reasoning.** The threshold is linear in cost and inverse-linear in margin, so raising the face
value by 71% raises the threshold by 64.5% (4.25/2.583 = 1.645) while the redemption rate is held
fixed. The offer must therefore clear a lift nearly two-thirds larger than the $7 offer at the same
redemption assumption, and the required lift is what the offer must *cause* rather than what it must
be *associated* with, which is why deep discounts need much stronger evidence before funding.

**Follow-up.** Ask what happens if redemption doubles to 16%, which is plausible for a larger face
value. Expected: cost becomes `0.06 + 1.92 = 1.98` and the threshold becomes 8.25 points, which no
decile in the constructed population clears.

**Red flag.** Using the face value as the cost, or forgetting the message cost. Scoring: full credit
recomputes cost, margin, and threshold with units; partial credit gets the structure and slips on
one term.

#### Q3 — A policy contacts 120,000 of 250,000 eligible rows and reports $0.94 value per contact. What is its value per eligible member-week, and why does the distinction matter?

**Answer.** `0.94 × 120,000 = 112,800` total, divided by 250,000 eligible rows gives `0.4512` per
eligible member-week. It matters because the owner is choosing a policy for the whole eligible
population, and the baseline must be expressed on the same denominator to be comparable.

**Reasoning.** The contact-denominated figure answers "was this contact worth making?" and the
eligible-denominated figure answers "was this selection rule better than the alternative?". A policy
that contacts fewer, better rows raises the first and can lower the second; a policy that contacts
more rows can do the reverse. Only the second is comparable across policies with different
selectivity.

**Follow-up.** Ask which denominator the review gate should use. Expected: eligible member-week, with
the per-contact figure retained as the operational diagnostic for the send decision.

**Red flag.** Reporting only the per-contact number, or dividing by contacted rows while calling the
result a policy value. Scoring: full credit computes both and explains the comparison; partial
credit computes one and asserts the other.

#### Q4 — Offline AUC is 0.91 and the top decile shows 14 points of lift. Diagnose before celebrating.

**Answer.** Both numbers are implausible together, and the joint implausibility points at leakage. A
14-point lift on a $7 incentive with a $80 basket at 30% margin implies `0.142 × 24.00 = 3.41` of
incremental margin against `0.62` of cost, a 5.5× return on contact cost sustained across 25,000
rows. The diagnostic is a per-feature window audit against the cut-off, not a model review.

**Reasoning.** A plausibility bound converts suspicion into a test: `face_value / (basket × margin)`
= `7.00 / 24.00 = 0.292` is the lift at which the incentive's whole value is captured, and 14.2
points is just under half of that bound — high but not impossible, which is exactly why the bound
alone is insufficient and the window audit is required.

**Follow-up.** Ask which feature is most likely to leak in a loyalty setting. Expected: any field
whose definition window opens at send or closes at redemption — redemption status, campaign
engagement, post-contact service contacts.

**Red flag.** Answering "0.91 is too high, retrain" without naming a mechanism or a check. Scoring:
full credit names a bound, a check, and the class of leaking feature; partial credit names only the
suspicion.

#### Q5 — With `r = 0.62` decay and a 14-day cooldown, how many contacts per quarter are profitable, and what changes if `r = 0.80`?

**Answer.** At `r = 0.62`, two contacts: `V1 = 0.8200` and `V2 = 0.2728`, with `V3 = -0.0665`. At
`r = 0.80`, `V1 = 0.8200`, `V2 = 24 × 0.048 − 0.62 = 1.152 − 0.62 = 0.5320`,
`V3 = 24 × 0.0384 − 0.62 = 0.9216 − 0.62 = 0.3016`, and `V4 = 24 × 0.0307 − 0.62 = 0.1168`, so four
contacts are profitable and the cap moves from two to four.

**Reasoning.** The break-even decay is `r* = c / (m × delta_p_1) = 0.62 / 1.44 = 0.4306`, which does
not depend on the cap; the cap is the largest `k` with `r^(k−1) > r*`. At `r = 0.62` the crossover is
between `k = 2` (0.62 > 0.4306) and `k = 3` (0.3844 < 0.4306); at `r = 0.80` it is between `k = 4`
(0.512 > 0.4306) and `k = 5` (0.4096 < 0.4306).

**Follow-up.** Ask what measurement would distinguish `r = 0.62` from `r = 0.80`. Expected: a
frequency-randomised test with at least three arms, since the decay is identified only by comparing
the second and third contacts against a no-contact control.

**Red flag.** Answering "as many as the cooldown allows". Scoring: full credit derives `r*` and the
cap; partial credit gets the cap by iterating without naming the invariant.

#### Q6 — Finance asks for a 10% budget cut. What do you cut first, and by what rule?

**Answer.** Cut contacts from the bottom of the value-sorted list, which raises the threshold from
4.00 to a higher value, and never cut proportionally across deciles. At a 108,000-contact budget with
the constructed curve, the cut removes the last 12,000 rows of decile 5, whose value is 0.82 each, so
the forgone value is `12,000 × 0.34` — the value at decile 5's estimate, not the average.

**Reasoning.** Because the sort is by value, the marginal row is the least valuable one, and a
proportional cut would remove high-value rows to protect low-value ones. The forgone value is
computed at the marginal decile, not at the mean, which is the part most analyses get wrong: the
average value of 0.91 per contact would overstate the loss because the cut happens at the bottom.

**Follow-up.** Ask whether the answer changes if the review capacity also falls by 10%. Expected:
yes, and the two cuts interact — a smaller review cap routes fewer rows to humans, so more rows
auto-send and the effective threshold falls if the envelope is unchanged.

**Red flag.** Cutting the worst-performing segments by response rate. Scoring: full credit cuts at
the margin by value and computes forgone value at the marginal decile; partial credit says "cut the
bottom decile".

#### Q7 — How much does a 10% household holdout cost per year in forgone margin, and is it worth it?

**Answer.** `12,000 × 0.82 = 9,840` per cycle, `× 52 = 511,680` per year, against an annual contact
spend of `120,000 × 0.62 × 52 = 3,868,800`. The holdout is 13.2% of the contact budget and 45.6% of
the annual value destroyed by the propensity policy, so it costs less than the error it would detect.

**Reasoning.** The comparison that matters is not holdout cost versus zero; it is holdout cost
versus the cost of continuing a policy on unverified evidence. Since the two candidate policies
differ by 1,123,200 per year, and the holdout can distinguish them, the holdout pays for itself if
it has more than a 45.6% chance of changing the decision.

**Follow-up.** Ask how the cost changes if the holdout is drawn from the eligible population rather
than the contacted band. Expected: it rises, because some held-out rows would never have been
contacted, so the withheld-contact count per unit of identification power increases.

**Red flag.** Treating the holdout as free, or comparing it against total programme revenue rather
than against the decision it informs. Scoring: full credit computes the annual cost and the
comparison; partial credit computes the cost only.

#### Q8 — Realised response rate rose 12 points but realised margin fell. What do you check first?

**Answer.** Check whether the selected set moved toward high-`p1` rows, which is the signature of a
target defect rather than a model defect. Concretely: compare the selected set's `p1` distribution
and `delta_p` distribution against the previous cycle. Rising `p1` with falling `delta_p` means the
policy is selecting sure things.

**Reasoning.** Response rate is `sum(p1)/|S|` and margin is `sum(m·delta_p − c)`. Any change that
raises the numerator of one and the other's mean is a change in composition along the level/slope
axis. The two metrics are not proxies; they are functions of different moments of the same
distribution, and they diverge precisely when the target is wrong.

**Follow-up.** Ask what would make the answer different. Expected: if `p1` and `delta_p` both rose
and margin still fell, the cause is cost or margin-rate drift, not selection — check the invoice and
the margin vector.

**Red flag.** Retraining the model, or adding features, without checking composition. Scoring: full
credit names the distributional check and the mechanism; partial credit notices the contradiction but
proposes a model fix.

#### Q9 — How many rows per arm do you need to detect a 2-point lift at 30% baseline, and what does one decile give you?

**Answer.** `n = 2 × (1.9600 + 0.8416)² × 0.21 / 0.0004 = 2 × 7.8490 × 0.21 / 0.0004 = 3.2966/0.0004
= 8,241` per arm. One decile has 25,000 rows; at a 20% control share the control arm has 5,000, so
the resolvable effect is `sqrt(3.2966/5,000) = 0.02568`, i.e. 2.57 points, and a 2-point effect is not
resolvable in a single cycle.

**Reasoning.** The power calculation is symmetric in the two arms, so the constraint is the smaller
arm; a 20% control share wastes power relative to 50/50 but is standard when treatment is valuable.
Four pooled cycles give 20,000 control rows and resolve 1.28 points.

**Follow-up.** Ask what changes with ten decile tests. Expected: Bonferroni at `alpha = 0.005` needs
13,978 per arm, a factor of 1.70, which is why the plan pre-registers three bands instead of ten.

**Red flag.** Quoting a sample size without stating alpha, power, or the baseline rate. Scoring: full
credit computes both the requirement and the achieved resolution; partial credit computes one.

#### Q10 — The uplift model's `p0` has a standard deviation of 0.004. What is wrong?

**Answer.** A near-constant counterfactual model means `p0` carries almost no information, so
`delta_p = p1 − p0` is nearly a monotone transform of `p1` and the "uplift" ranking is a propensity
ranking with extra steps. The first check is the control-row count per feature decile in the
training data; if controls are scarce in the upper deciles, `p0` was extrapolated.

**Reasoning.** The variance of a prediction is bounded by the variance of its target in the training
population. If the model saw few untreated rows with high `p1`, it regresses toward the mean in
exactly the region where the ranking decision is made, which flattens the slope that the policy
needs.

**Follow-up.** Ask how to diagnose whether the flat `p0` is a data defect or a genuine absence of
signal. Expected: compare the observed purchase rate among controls by decile of `p1`; if it varies
while the model's `p0` does not, the model is underfit or the training sample is thin.

**Red flag.** Increasing model complexity. Scoring: full credit connects variance to identifiability
and names the control-count check; partial credit notices the flatness.

#### Q11 — Name the four clocks in this decision and one defect each produces.

**Answer.** Event time (when the member acted) — a defect is a feature whose window straddles the
cut-off. Decision time (Monday 06:00) — a defect is an eligibility filter evaluated at wall-clock
send time. Outcome time (T+28d purchase, T+42d returns) — a defect is comparing cycles with different
maturity. Effective time (when a record became authoritative) — a defect is a join on `ingest_ts`
instead of `effective_ts`, which admits late corrections.

**Reasoning.** The four clocks exist because a warehouse row has four distinct timestamps that
usually agree and occasionally diverge by hours or days. Every divergence is a potential leak or a
potential false exclusion, and the divergence is invisible unless the query names the timestamp it
uses.

**Follow-up.** Ask which clock is most often omitted. Expected: effective time, because most teams
have `created_at` and `updated_at` but not a version-effective timestamp, and the omission is
harmless until a correction arrives.

**Red flag.** Naming only event and processing time. Scoring: full credit names all four with a
defect each; partial credit names three.

#### Q12 — A reviewer rejects 40% of the queue. What do you change, and what do you not change?

**Answer.** Change the review envelope and the model's calibration reporting; do not change the
reviewers or the model's features first. A 40% rejection rate against a designed 7.5% means either
the envelope admits rows the reviewers consider out of policy, or the estimates for those rows are
systematically overstated.

**Reasoning.** The discriminating evidence is the reason-code distribution. If rejections cluster on
a policy category (for example offers to members with open complaints), the envelope is wrong. If
they spread across value levels with reviewers citing "this doesn't look like they'd respond", the
estimates are miscalibrated at the high end, which section 6.1's plausibility bound can confirm.

**Follow-up.** Ask what happens to the freed budget. Expected: it becomes a recorded credit, not a
same-cycle refill, per the immutability contract in §8.

**Red flag.** Adding reviewer headcount, which treats a signal as a capacity problem. Scoring: full
credit separates the two causes with a reason-code check; partial credit proposes a single fix.

#### Q13 — Recompute the cycle value if 30% of the selected rows are actually uncontactable at send time.

**Answer.** If 30% of 120,000 rows fail at send, 36,000 contacts do not happen and 84,000 do. At the
mixed decile values of §6.2, the lost value is approximately the value of the lowest tranche, since
the failures are not randomly distributed: `20,000 × 0.34` (decile 5) plus `16,000 × 0.148`
(decile 6) gives `6,800 + 2,368 = 9,168` forgone, and the cycle value falls from 109,200 to 100,032.

**Reasoning.** The failure is not a proportional haircut on the total, because the failing rows are
the ones with the weakest contactability, which correlates with the lowest value tranche only if
contactability is unrelated to value. The honest statement is that the assumption must be checked:
if failures concentrate in one channel and that channel carries the high-value rows, the loss is
larger than the tranche arithmetic suggests.

**Follow-up.** Ask whether the policy should down-weight members with a history of send failures.
Expected: yes, and that down-weighting is a change to `c` (a higher effective cost) rather than to
`delta_p`, which is a cleaner formulation.

**Red flag.** Multiplying the total by 0.7 without asking about the distribution. Scoring: full
credit asks about the failure distribution; partial credit computes the proportional answer and
flags it.

#### Q14 — The complaint rate doubled in the week the frequency cap was raised from two to three. Is the cap the cause?

**Answer.** Not established. The competing cause is the offer's relevance: a third contact reaches a
lower-lift population, and low realised lift is what members perceive as an irrelevant offer. The
discriminating check is the complaint rate by contact index — if the third contact's complaints per
1,000 sends exceed the first contact's by more than the lift ratio predicts, the cap is the cause; if
the third contact's rate is proportional to its lower lift, the offer is.

**Reasoning.** Complaints and lift are both functions of relevance, so a policy that reaches deeper
into the population will see more complaints at constant per-contact relevance. Attributing the rise
to the cap requires showing that the *per-index* rate rose, not just the aggregate.

**Follow-up.** Ask what design would settle it. Expected: randomise the third contact at the member
level and compare complaint rates per 1,000 sends, holding the offer and creative fixed.

**Red flag.** Reverting the cap immediately, which loses the information and leaves the mechanism
unknown. Scoring: full credit names the index decomposition and the design; partial credit proposes a
revert.

#### Q15 — You have one week of analyst time. Which single check do you run?

**Answer.** The deterministic replay check: run the same snapshot twice, assert an empty set
difference in the selected rows, and reconcile `selected = sent + abstained + failed`. It is the
cheapest check that can invalidate everything else, because a non-reproducible selection means no
later result can be attributed to the policy.

**Reasoning.** The replay check tests the sort's totality, the snapshot's stability, and the
pipeline's idempotency at once, and it fails loudly. It does not test the target, so it is a
floor rather than a ceiling; but a programme with a wrong target and a reproducible pipeline can
still learn from a holdout, whereas a programme with the right target and a non-reproducible
pipeline cannot learn from anything.

**Follow-up.** Ask what the second week buys. Expected: the plausibility bound and the control-count
check per decile, which together test the target's identifiability at low cost.

**Red flag.** Running a model comparison, which optimises a quantity that the case has shown is not
the objective. Scoring: full credit justifies the choice against the alternative; partial credit
names a reasonable check without the ordering argument.

#### Q16 — When should this programme not be built at all?

**Answer.** When the measured lift is indistinguishable from zero at the resolution the available
holdout can achieve, or when the offer's economics cannot clear break-even in any decile, or when
the review capacity cannot be staffed so decisions become late notifications. In each case the
correct output is the rules queue with a smaller budget, not a better model.

**Reasoning.** The value of the ranked policy is `V(S_uplift) − V(S_baseline)`, and if either term is
near zero the difference is unmeasurable. A programme that cannot demonstrate a difference should
not spend 3.87 million dollars a year claiming one; it should spend less and say less.

**Follow-up.** Ask how to tell "no lift" from "no power". Expected: pre-register the resolution, and
report the confidence interval rather than the point estimate; a wide interval containing the
break-even threshold is a mandate to gather more data, not a null result.

**Red flag.** Treating a null result as a failure of the analyst. Scoring: full credit names all
three stop conditions and the power distinction; partial credit names one or two.

Takeaway: a defensible interview answer names the number, its units, and the observation that would
change it.

## 15. Quick Reference Card

| Element | Value |
|---|---|
| Decision | Select at most 120,000 of 250,000 eligible member-weeks for a funded offer, weekly |
| Objective | Incremental gross margin net of contact cost: `m·(p1 − p0) − c` |
| Break-even lift | `c/m` = 0.62/24.00 = 2.583 points; recompute whenever cost or margin moves |
| Selection rule | Greedy sort on value, deterministic tie-break on `member_id`, then eligibility gate |
| Binding constraints | Budget 120,000; cooldown 14 days; quarterly cap 2; review 2,400 rows |
| Frequency | Two contacts per quarter at `r = 0.62`; cap falls to one if `r < 0.431` |
| Guardrails | Complaints ≤6.0/10,000; unsubscribes ≤0.60%; review rejections ≤15% |
| Primary metric | Margin per eligible member-week, comparator at equal budget, holdout-identified |
| Wrong metric | Response rate — passes a 3-point gate with a policy worth 1,123,200 less per year |
| Validation | 10% household holdout, 3 pre-registered bands, 4 cycles, resolves 1.28 points |
| Evidence status | ILLUSTRATIVE arithmetic; `not-measured`; no cycle executed |
| First check | Deterministic replay: empty set difference, and `selected = sent + abstained + failed` |
| Fallback | Rules queue in degraded mode, logged with the triggering condition |
