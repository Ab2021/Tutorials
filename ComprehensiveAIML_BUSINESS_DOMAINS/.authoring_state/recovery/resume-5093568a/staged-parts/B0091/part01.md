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
