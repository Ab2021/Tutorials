---
case_id: "CASE-M12-02"
case_title: "Customer segmentation for campaign selection"
case_version: "1.0"

created: "2026-11-11"
last_reviewed: null
status: "proposed"
content_status: "draft"

case_type: "analytical"
domain: "lifecycle marketing and customer relationship management"
primary_pattern: "decision-ready customer segmentation"
secondary_patterns: ["time-safe feature construction", "randomized campaign evaluation"]

evidence_level: "illustrative"
source_basis: "constructed teaching case; no company result asserted"
as_of_date: "2026-11-11"
owner_role: "Lifecycle Marketing Lead"

source_index: "INDEX.md"
review_trigger: "Material change in identity resolution, consent policy, channel capacity, offer economics, or assignment stability."
--- Specific decision context: current record

# Customer segmentation for campaigns

Every record, amount, rate, and outcome in this case is ILLUSTRATIVE. The case describes a reproducible decision design, not a deployed company's performance or a claim that clustering caused a commercial result.


## 1. The One-Line Framing

At the weekly campaign cut, choose one permitted message, offer, or suppression for each eligible household, using a segment that describes observed behaviour without pretending to predict individual response.
The decision owner is the lifecycle-marketing lead, who must spend a fixed outbound capacity without contacting people who withdrew permission or already received a recent competing message.
The comparator is a single broad campaign sent to the same eligible population under the same channel and budget limits.
A segment is useful only when its definition, assignment time, and action rule are frozen before campaign selection.

The objective is expected incremental contribution after discount cost, message cost, and known operational constraints rather than response rate alone.
The unit is a resolved customer-household at the decision cut; duplicate identifiers must never create two offers for one person.
This case uses three behavioral groups and an explicit unassigned route, because a forced cluster can create an unjustified action.
The intended decision is modest: choose a testable policy for a limited campaign, then retain or retire it based on randomized evidence.

A high silhouette score is not a success condition, because geometric separation does not establish an offer's incremental value.
The policy must leave a customer untouched when consent, suppression, frequency, or identity confidence is insufficient.
Campaign capacity is treated as a scarce resource, so segment priority is recalculated after exclusions rather than on the raw cluster size.
The case distinguishes descriptive segmentation from uplift modeling; a group average is not an individual's causal treatment effect.

A cohort's name is a communication aid, not a permission to infer motive, need, protected status, or lifetime value.
The implementation is batch-oriented: features close on Sunday 23:59 UTC, assignments publish Monday, and delivery happens only after a final consent check.
The evidence limit is explicit: all numerical examples below are constructed and are not a measured outcome from a real campaign.
## 2. Context & Background

**Decision cadence.** The retailer plans one email campaign each Monday and one optional in-app reminder later in the week.

Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe.
**Commercial setting.** The catalogue has replenishable essentials, seasonal goods, and higher-margin accessories whose offers cannot be treated as interchangeable.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Commercial setting.** The catalogue has replenishable essentials, seasonal goods, and higher-margin accessories whose offers cannot be treated as interchangeable.

The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Commercial setting.** The catalogue has replenishable essentials, seasonal goods, and higher-margin accessories whose offers cannot be treated as interchangeable.
**Customer relationship.** A household can purchase through web, app, and store-linked accounts, creating a many-identifier identity problem before any clustering occurs.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Customer relationship.** A household can purchase through web, app, and store-linked accounts, creating a many-identifier identity problem before any clustering occurs.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Customer relationship.** A household can purchase through web, app, and store-linked accounts, creating a many-identifier identity problem before any clustering occurs.

**Marketing constraint.** Only customers with current channel permission may enter a send file; historical permission is not a valid substitute for the cut-time record.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Marketing constraint.** Only customers with current channel permission may enter a send file; historical permission is not a valid substitute for the cut-time record.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Marketing constraint.** Only customers with current channel permission may enter a send file; historical permission is not a valid substitute for the cut-time record.
**Frequency policy.** The contact governor excludes anyone who received two promotional messages in the prior seven days, even if a segment rule would otherwise select them.

Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Frequency policy.** The contact governor excludes anyone who received two promotional messages in the prior seven days, even if a segment rule would otherwise select them.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Frequency policy.** The contact governor excludes anyone who received two promotional messages in the prior seven days, even if a segment rule would otherwise select them.
**Offer constraint.** A discount code is single-use, expires after fourteen calendar days, and cannot be combined with a contractually funded promotion.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Offer constraint.** A discount code is single-use, expires after fourteen calendar days, and cannot be combined with a contractually funded promotion.

The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Offer constraint.** A discount code is single-use, expires after fourteen calendar days, and cannot be combined with a contractually funded promotion.
**Fulfilment constraint.** Inventory availability is evaluated at campaign assembly, because showing a depleted item converts a segmentation error into avoidable customer friction.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Fulfilment constraint.** Inventory availability is evaluated at campaign assembly, because showing a depleted item converts a segmentation error into avoidable customer friction.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Fulfilment constraint.** Inventory availability is evaluated at campaign assembly, because showing a depleted item converts a segmentation error into avoidable customer friction.

**Measurement context.** Purchase events settle at different times from order placement, so the case uses event time for features and a later accounting window for contribution.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Measurement context.** Purchase events settle at different times from order placement, so the case uses event time for features and a later accounting window for contribution.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Measurement context.** Purchase events settle at different times from order placement, so the case uses event time for features and a later accounting window for contribution.
**Identity context.** An email hash may map to multiple customer records after a household split, and the system therefore preserves an identity-confidence field.

Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Identity context.** An email hash may map to multiple customer records after a household split, and the system therefore preserves an identity-confidence field.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Identity context.** An email hash may map to multiple customer records after a household split, and the system therefore preserves an identity-confidence field.
**Privacy context.** Postal region is used only for delivery feasibility checks, not as an input to cluster fitting or a proxy for demographic targeting.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Privacy context.** Postal region is used only for delivery feasibility checks, not as an input to cluster fitting or a proxy for demographic targeting.

The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Privacy context.** Postal region is used only for delivery feasibility checks, not as an input to cluster fitting or a proxy for demographic targeting.
**Seasonality context.** A holiday week can increase both purchase recency and message volume, making a recent-behaviour cluster look larger without any durable preference change.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Seasonality context.** A holiday week can increase both purchase recency and message volume, making a recent-behaviour cluster look larger without any durable preference change.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Seasonality context.** A holiday week can increase both purchase recency and message volume, making a recent-behaviour cluster look larger without any durable preference change.

**Channel context.** Email opens are treated as noisy delivery signals because privacy features and image blocking make them unsuitable as a primary behavioural feature.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Channel context.** Email opens are treated as noisy delivery signals because privacy features and image blocking make them unsuitable as a primary behavioural feature.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Channel context.** Email opens are treated as noisy delivery signals because privacy features and image blocking make them unsuitable as a primary behavioural feature.
**Catalog context.** Category affinity is calculated from completed purchases rather than clicks, because browsing may reflect gift research or stock checking.

Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Catalog context.** Category affinity is calculated from completed purchases rather than clicks, because browsing may reflect gift research or stock checking.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Catalog context.** Category affinity is calculated from completed purchases rather than clicks, because browsing may reflect gift research or stock checking.
**Service context.** Customers with an unresolved support case are held out from promotion so service recovery is not confused with a revenue experiment.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Service context.** Customers with an unresolved support case are held out from promotion so service recovery is not confused with a revenue experiment.

The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Service context.** Customers with an unresolved support case are held out from promotion so service recovery is not confused with a revenue experiment.
**Governance context.** The marketing lead owns the action policy, while data stewardship owns the identity and consent eligibility checks.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Governance context.** The marketing lead owns the action policy, while data stewardship owns the identity and consent eligibility checks.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Governance context.** The marketing lead owns the action policy, while data stewardship owns the identity and consent eligibility checks.

**Analytical context.** Clustering is refit monthly but campaign assignments use a frozen approved version for the entire weekly decision cycle.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Analytical context.** Clustering is refit monthly but campaign assignments use a frozen approved version for the entire weekly decision cycle.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Analytical context.** Clustering is refit monthly but campaign assignments use a frozen approved version for the entire weekly decision cycle.
**Economic context.** Contribution credits net product margin and discount funding but excludes speculative lifetime value from the immediate campaign decision.

Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Economic context.** Contribution credits net product margin and discount funding but excludes speculative lifetime value from the immediate campaign decision.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Economic context.** Contribution credits net product margin and discount funding but excludes speculative lifetime value from the immediate campaign decision.
**Audit context.** Each send record stores feature cut, identity link version, cluster version, eligibility decision, policy rule, and offer identifier.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Audit context.** Each send record stores feature cut, identity link version, cluster version, eligibility decision, policy rule, and offer identifier.

The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Audit context.** Each send record stores feature cut, identity link version, cluster version, eligibility decision, policy rule, and offer identifier.
**Human context.** Merchandising may veto an offer for a segment when a supplier restriction changes after the cluster was produced.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Human context.** Merchandising may veto an offer for a segment when a supplier restriction changes after the cluster was produced.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Human context.** Merchandising may veto an offer for a segment when a supplier restriction changes after the cluster was produced.

**Fallback context.** When the segmentation pipeline is unavailable, the approved fallback is a consented random holdout-compatible broad campaign, not stale assignments.
Its operational implication is recorded before campaign launch so a later analyst can separate a data condition from a marketing choice. Specific decision context: **Fallback context.** When the segmentation pipeline is unavailable, the approved fallback is a consented random holdout-compatible broad campaign, not stale assignments.
The case treats this as a binding design fact rather than as background colour, because changing it can change who is eligible or what action is safe. Specific decision context: **Fallback context.** When the segmentation pipeline is unavailable, the approved fallback is a consented random holdout-compatible broad campaign, not stale assignments.
## 3. The Business Question

**Decision question.** Which consented, identity-resolved customers should be eligible for campaign assignment at the Monday cut?

The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart.
A valid answer also names the comparator, decision window, and condition under which the policy stops applying.
**Decision question.** How many stable behavioural segments can be explained to campaign operators without hiding material within-group differences?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** How many stable behavioural segments can be explained to campaign operators without hiding material within-group differences?

A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** How many stable behavioural segments can be explained to campaign operators without hiding material within-group differences?
**Decision question.** Which feature windows describe pre-decision behaviour without using a purchase, return, or consent event that arrives after the cut?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** Which feature windows describe pre-decision behaviour without using a purchase, return, or consent event that arrives after the cut?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** Which feature windows describe pre-decision behaviour without using a purchase, return, or consent event that arrives after the cut?

**Decision question.** When should a customer be placed in the unassigned route instead of being forced into the nearest centroid?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** When should a customer be placed in the unassigned route instead of being forced into the nearest centroid?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** When should a customer be placed in the unassigned route instead of being forced into the nearest centroid?
**Decision question.** Which offer family is economically feasible for each segment after discount funding and inventory constraints are applied?

The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** Which offer family is economically feasible for each segment after discount funding and inventory constraints are applied?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** Which offer family is economically feasible for each segment after discount funding and inventory constraints are applied?
**Decision question.** How should the policy allocate a 12,000-email capacity when more eligible customers qualify than the channel can contact?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** How should the policy allocate a 12,000-email capacity when more eligible customers qualify than the channel can contact?

A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** How should the policy allocate a 12,000-email capacity when more eligible customers qualify than the channel can contact?
**Decision question.** What randomized comparator isolates the incremental effect of the segment policy from the effect of merely sending more messages?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** What randomized comparator isolates the incremental effect of the segment policy from the effect of merely sending more messages?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** What randomized comparator isolates the incremental effect of the segment policy from the effect of merely sending more messages?

**Decision question.** Which guardrails stop a seemingly profitable campaign from increasing complaints, unsubscribes, or inequitable access?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** Which guardrails stop a seemingly profitable campaign from increasing complaints, unsubscribes, or inequitable access?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** Which guardrails stop a seemingly profitable campaign from increasing complaints, unsubscribes, or inequitable access?
**Decision question.** How will the team distinguish a shift in customer behaviour from a broken identity join or delayed event feed?

The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** How will the team distinguish a shift in customer behaviour from a broken identity join or delayed event feed?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** How will the team distinguish a shift in customer behaviour from a broken identity join or delayed event feed?
**Decision question.** What evidence is sufficient to keep, change, or retire a segment-to-offer rule after the test window closes?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** What evidence is sufficient to keep, change, or retire a segment-to-offer rule after the test window closes?

A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** What evidence is sufficient to keep, change, or retire a segment-to-offer rule after the test window closes?
**Decision question.** How will the system prevent two linked accounts from receiving incompatible discounts in the same campaign?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** How will the system prevent two linked accounts from receiving incompatible discounts in the same campaign?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** How will the system prevent two linked accounts from receiving incompatible discounts in the same campaign?

**Decision question.** What should happen when an offer is unavailable after assignment but before delivery?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** What should happen when an offer is unavailable after assignment but before delivery?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** What should happen when an offer is unavailable after assignment but before delivery?
**Decision question.** How does the policy avoid treating protected or sensitive characteristics as targets or using proxies intentionally?

The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** How does the policy avoid treating protected or sensitive characteristics as targets or using proxies intentionally?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** How does the policy avoid treating protected or sensitive characteristics as targets or using proxies intentionally?
**Decision question.** Which artifacts let an auditor reconstruct why one customer was contacted and another was suppressed?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** Which artifacts let an auditor reconstruct why one customer was contacted and another was suppressed?

A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** Which artifacts let an auditor reconstruct why one customer was contacted and another was suppressed?
**Decision question.** What outcome remains unknown because the case is illustrative and has not run a real campaign?
The answer must be a rule that could be applied to a frozen send file, not a retrospective story about a chart. Specific decision context: **Decision question.** What outcome remains unknown because the case is illustrative and has not run a real campaign?
A valid answer also names the comparator, decision window, and condition under which the policy stops applying. Specific decision context: **Decision question.** What outcome remains unknown because the case is illustrative and has not run a real campaign?

## 4. Assumptions & Constraints

**Population boundary.** The analysis covers resolved adult customer records with an active relationship and at least one transaction in the past 365 days.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision.
It is not safe to repair the condition by changing the population after outcomes have been observed.
**Decision timestamp.** The weekly decision timestamp is Monday 08:00 UTC, and no event with a later event time may enter features or eligibility.

This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Decision timestamp.** The weekly decision timestamp is Monday 08:00 UTC, and no event with a later event time may enter features or eligibility.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Decision timestamp.** The weekly decision timestamp is Monday 08:00 UTC, and no event with a later event time may enter features or eligibility.
**Feature history.** Recency, frequency, and monetary summaries use the 180 days ending at the cut, while trend features compare the latest 30 days with the preceding 30.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Feature history.** Recency, frequency, and monetary summaries use the 180 days ending at the cut, while trend features compare the latest 30 days with the preceding 30.

It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Feature history.** Recency, frequency, and monetary summaries use the 180 days ending at the cut, while trend features compare the latest 30 days with the preceding 30.
**Identity threshold.** Records with a deterministic customer identifier are eligible; probabilistic household links below 0.98 confidence are excluded from automated assignment.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Identity threshold.** Records with a deterministic customer identifier are eligible; probabilistic household links below 0.98 confidence are excluded from automated assignment.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Identity threshold.** Records with a deterministic customer identifier are eligible; probabilistic household links below 0.98 confidence are excluded from automated assignment.

**Consent rule.** Email permission must be active at both assembly and send time, since an intervening withdrawal overrides the original assignment.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Consent rule.** Email permission must be active at both assembly and send time, since an intervening withdrawal overrides the original assignment.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Consent rule.** Email permission must be active at both assembly and send time, since an intervening withdrawal overrides the original assignment.
**Channel capacity.** The constructed weekly email limit is 12,000 distinct households, with no capacity borrowing from the in-app channel.

This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Channel capacity.** The constructed weekly email limit is 12,000 distinct households, with no capacity borrowing from the in-app channel.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Channel capacity.** The constructed weekly email limit is 12,000 distinct households, with no capacity borrowing from the in-app channel.
**Frequency cap.** No household can receive more than two marketing messages in seven days or more than six in thirty days.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Frequency cap.** No household can receive more than two marketing messages in seven days or more than six in thirty days.

It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Frequency cap.** No household can receive more than two marketing messages in seven days or more than six in thirty days.
**Suppression rule.** Fraud review, deceased flags, legal opt-outs, unresolved severe complaints, and active returns disputes block promotional contact.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Suppression rule.** Fraud review, deceased flags, legal opt-outs, unresolved severe complaints, and active returns disputes block promotional contact.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Suppression rule.** Fraud review, deceased flags, legal opt-outs, unresolved severe complaints, and active returns disputes block promotional contact.

**Offer limit.** Each selected household receives at most one offer family in the campaign to avoid attribution ambiguity and discount stacking.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Offer limit.** Each selected household receives at most one offer family in the campaign to avoid attribution ambiguity and discount stacking.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Offer limit.** Each selected household receives at most one offer family in the campaign to avoid attribution ambiguity and discount stacking.
**Cluster range.** Candidate counts from two through six are considered, because one cluster makes no allocation distinction and larger counts may not be operable.

This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Cluster range.** Candidate counts from two through six are considered, because one cluster makes no allocation distinction and larger counts may not be operable.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Cluster range.** Candidate counts from two through six are considered, because one cluster makes no allocation distinction and larger counts may not be operable.
**Distance rule.** K-means uses standardized numeric features and a predeclared winsorization treatment; a raw currency feature cannot dominate Euclidean distance.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Distance rule.** K-means uses standardized numeric features and a predeclared winsorization treatment; a raw currency feature cannot dominate Euclidean distance.

It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Distance rule.** K-means uses standardized numeric features and a predeclared winsorization treatment; a raw currency feature cannot dominate Euclidean distance.
**Assignment uncertainty.** Customers near a centroid boundary are marked uncertain when their nearest-versus-second-nearest distance margin is below the policy threshold.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Assignment uncertainty.** Customers near a centroid boundary are marked uncertain when their nearest-versus-second-nearest distance margin is below the policy threshold.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Assignment uncertainty.** Customers near a centroid boundary are marked uncertain when their nearest-versus-second-nearest distance margin is below the policy threshold.

**Fairness scope.** Protected attributes are not required for assignment, but lawful audit attributes may be analyzed in a segregated review environment where available.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Fairness scope.** Protected attributes are not required for assignment, but lawful audit attributes may be analyzed in a segregated review environment where available.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Fairness scope.** Protected attributes are not required for assignment, but lawful audit attributes may be analyzed in a segregated review environment where available.
**Experiment scope.** Randomization happens after all eligibility and capacity gates, so treatment and control have the same opportunity to receive a message.

This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Experiment scope.** Randomization happens after all eligibility and capacity gates, so treatment and control have the same opportunity to receive a message.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Experiment scope.** Randomization happens after all eligibility and capacity gates, so treatment and control have the same opportunity to receive a message.
**Attribution window.** The primary conversion window is fourteen days after send, with returns observed for an additional twenty-eight days before contribution is final.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Attribution window.** The primary conversion window is fourteen days after send, with returns observed for an additional twenty-eight days before contribution is final.

It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Attribution window.** The primary conversion window is fourteen days after send, with returns observed for an additional twenty-eight days before contribution is final.
**No causal shortcut.** Segment averages may guide hypotheses but never replace a randomized estimate of the policy's incremental effect.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **No causal shortcut.** Segment averages may guide hypotheses but never replace a randomized estimate of the policy's incremental effect.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **No causal shortcut.** Segment averages may guide hypotheses but never replace a randomized estimate of the policy's incremental effect.

**Data latency.** Events later than 36 hours after occurrence are excluded from the weekly cut and monitored as a feed-quality issue.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Data latency.** Events later than 36 hours after occurrence are excluded from the weekly cut and monitored as a feed-quality issue.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Data latency.** Events later than 36 hours after occurrence are excluded from the weekly cut and monitored as a feed-quality issue.
**Versioning.** Feature code, reference population, scaler, centroids, action matrix, and suppression policy each have immutable version identifiers.

This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Versioning.** Feature code, reference population, scaler, centroids, action matrix, and suppression policy each have immutable version identifiers.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Versioning.** Feature code, reference population, scaler, centroids, action matrix, and suppression policy each have immutable version identifiers.
**Human override.** A documented merchandising or safety override can suppress an action but cannot silently alter a customer's cluster label.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Human override.** A documented merchandising or safety override can suppress an action but cannot silently alter a customer's cluster label.

It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Human override.** A documented merchandising or safety override can suppress an action but cannot silently alter a customer's cluster label.
**Stopping rule.** The policy is paused if a mandatory guardrail breaches its limit, even if the provisional contribution estimate appears positive.
This assumption is recorded as a testable condition; an observed violation requires either a rerun from a valid snapshot or a documented no-send decision. Specific decision context: **Stopping rule.** The policy is paused if a mandatory guardrail breaches its limit, even if the provisional contribution estimate appears positive.
It is not safe to repair the condition by changing the population after outcomes have been observed. Specific decision context: **Stopping rule.** The policy is paused if a mandatory guardrail breaches its limit, even if the provisional contribution estimate appears positive.

## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** One row represents one resolved customer-household at the exact weekly decision cut; raw order lines are aggregated before modelling.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete.
**Primary key.** The durable key is `customer_household_id`; source-specific account IDs remain as lineage fields and are never used to count eligible recipients twice.

The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Primary key.** The durable key is `customer_household_id`; source-specific account IDs remain as lineage fields and are never used to count eligible recipients twice.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Primary key.** The durable key is `customer_household_id`; source-specific account IDs remain as lineage fields and are never used to count eligible recipients twice.
**Feature cutoff.** Every feature has event_time less than or equal to Sunday 23:59:59 UTC, while processing_time is retained only to diagnose late arrival.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Feature cutoff.** Every feature has event_time less than or equal to Sunday 23:59:59 UTC, while processing_time is retained only to diagnose late arrival.

A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Feature cutoff.** Every feature has event_time less than or equal to Sunday 23:59:59 UTC, while processing_time is retained only to diagnose late arrival.
**Population.** The starting population is all active customer records with a transaction in 365 days, before consent, identity, frequency, and safety exclusions.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Population.** The starting population is all active customer records with a transaction in 365 days, before consent, identity, frequency, and safety exclusions.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Population.** The starting population is all active customer records with a transaction in 365 days, before consent, identity, frequency, and safety exclusions.

**Transaction amount.** Net spend is measured in currency units after cancellations and recorded returns, using the currency conversion rule fixed for the cut.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Transaction amount.** Net spend is measured in currency units after cancellations and recorded returns, using the currency conversion rule fixed for the cut.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Transaction amount.** Net spend is measured in currency units after cancellations and recorded returns, using the currency conversion rule fixed for the cut.
**Recency.** Recency is calendar days from the most recent completed purchase to the cutoff and is bounded below by zero.

The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Recency.** Recency is calendar days from the most recent completed purchase to the cutoff and is bounded below by zero.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Recency.** Recency is calendar days from the most recent completed purchase to the cutoff and is bounded below by zero.
**Frequency.** Purchase frequency is the count of completed orders in the last 180 days, not the count of page views or line items.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Frequency.** Purchase frequency is the count of completed orders in the last 180 days, not the count of page views or line items.

A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Frequency.** Purchase frequency is the count of completed orders in the last 180 days, not the count of page views or line items.
**Margin proxy.** Category margin band is an approved catalog lookup, used only after modelling to choose feasible offers rather than to represent individual profitability.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Margin proxy.** Category margin band is an approved catalog lookup, used only after modelling to choose feasible offers rather than to represent individual profitability.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Margin proxy.** Category margin band is an approved catalog lookup, used only after modelling to choose feasible offers rather than to represent individual profitability.

**Category mix.** Each category feature is the share of completed order value in the 180-day window, and shares are zero when the denominator is zero.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Category mix.** Each category feature is the share of completed order value in the 180-day window, and shares are zero when the denominator is zero.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Category mix.** Each category feature is the share of completed order value in the 180-day window, and shares are zero when the denominator is zero.
**Return behaviour.** Return rate is returned value divided by fulfilled value over 180 days; it is absent when there was no fulfilled value and is not imputed as zero.

The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Return behaviour.** Return rate is returned value divided by fulfilled value over 180 days; it is absent when there was no fulfilled value and is not imputed as zero.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Return behaviour.** Return rate is returned value divided by fulfilled value over 180 days; it is absent when there was no fulfilled value and is not imputed as zero.
**Engagement.** Recent email delivery status may define a channel eligibility diagnostic, but opens and clicks are excluded from clustering due to measurement artifacts.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Engagement.** Recent email delivery status may define a channel eligibility diagnostic, but opens and clicks are excluded from clustering due to measurement artifacts.

A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Engagement.** Recent email delivery status may define a channel eligibility diagnostic, but opens and clicks are excluded from clustering due to measurement artifacts.
**Consent.** Consent status and effective timestamp come from the preference service and are evaluated at send time as well as at assembly.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Consent.** Consent status and effective timestamp come from the preference service and are evaluated at send time as well as at assembly.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Consent.** Consent status and effective timestamp come from the preference service and are evaluated at send time as well as at assembly.

**Identity confidence.** The match service provides deterministic, high-confidence, and ambiguous classes; ambiguous class records cannot receive an automated marketing action.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Identity confidence.** The match service provides deterministic, high-confidence, and ambiguous classes; ambiguous class records cannot receive an automated marketing action.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Identity confidence.** The match service provides deterministic, high-confidence, and ambiguous classes; ambiguous class records cannot receive an automated marketing action.
**Inventory.** Available-to-promise inventory is a campaign-time action constraint and is deliberately not a clustering feature because it changes independently of customer behaviour.

The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Inventory.** Available-to-promise inventory is a campaign-time action constraint and is deliberately not a clustering feature because it changes independently of customer behaviour.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Inventory.** Available-to-promise inventory is a campaign-time action constraint and is deliberately not a clustering feature because it changes independently of customer behaviour.
**Support status.** An unresolved high-severity case is an exclusion flag; ticket text and sensitive complaint details are unavailable to the segmentation workflow.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Support status.** An unresolved high-severity case is an exclusion flag; ticket text and sensitive complaint details are unavailable to the segmentation workflow.

A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Support status.** An unresolved high-severity case is an exclusion flag; ticket text and sensitive complaint details are unavailable to the segmentation workflow.
**Protected information.** No protected class, inferred demographic label, health information, precise location history, or free-text customer note is available to the fit.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Protected information.** No protected class, inferred demographic label, health information, precise location history, or free-text customer note is available to the fit.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Protected information.** No protected class, inferred demographic label, health information, precise location history, or free-text customer note is available to the fit.

**Future information.** Purchases, returns, consent changes, redemptions, and campaign exposures after the cutoff are unavailable at assignment time.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Future information.** Purchases, returns, consent changes, redemptions, and campaign exposures after the cutoff are unavailable at assignment time.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Future information.** Purchases, returns, consent changes, redemptions, and campaign exposures after the cutoff are unavailable at assignment time.
**Outcome fields.** Test-period orders, net contribution, unsubscribe events, complaints, and refunds are held in a later evaluation table rather than the fitting table.

The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Outcome fields.** Test-period orders, net contribution, unsubscribe events, complaints, and refunds are held in a later evaluation table rather than the fitting table.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Outcome fields.** Test-period orders, net contribution, unsubscribe events, complaints, and refunds are held in a later evaluation table rather than the fitting table.
**Retention.** The feature snapshot is retained according to the organization policy; the case assumes only a versioned aggregate snapshot enters model artifacts.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Retention.** The feature snapshot is retained according to the organization policy; the case assumes only a versioned aggregate snapshot enters model artifacts.

A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Retention.** The feature snapshot is retained according to the organization policy; the case assumes only a versioned aggregate snapshot enters model artifacts.
**Quality flags.** Each numeric feature has missingness, source freshness, and valid-range flags so an aggregate can be distinguished from an unavailable observation.
The data contract records owner, source table, event-time rule, unit, permissible range, and the consequence of a missing value. Specific decision context: **Quality flags.** Each numeric feature has missingness, source freshness, and valid-range flags so an aggregate can be distinguished from an unavailable observation.
A value that is unavailable at the decision cut is not backfilled from a later extract, even when that would make a dashboard look complete. Specific decision context: **Quality flags.** Each numeric feature has missingness, source freshness, and valid-range flags so an aggregate can be distinguished from an unavailable observation.

### 5b Tools & Stack

**Snapshot builder.** A scheduled SQL transformation produces a time-stamped customer feature table from orders, identities, preferences, and exclusions.
The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup.
All production-like names are illustrative roles, not assertions about a particular vendor or company architecture.
**Validation layer.** Schema checks enforce one row per durable household key, nonnegative monetary fields, and a documented maximum event lag.

The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup. Specific decision context: **Validation layer.** Schema checks enforce one row per durable household key, nonnegative monetary fields, and a documented maximum event lag.
All production-like names are illustrative roles, not assertions about a particular vendor or company architecture. Specific decision context: **Validation layer.** Schema checks enforce one row per durable household key, nonnegative monetary fields, and a documented maximum event lag.
**Model environment.** A versioned Python environment fits scalers and clustering candidates from a frozen training snapshot, with fixed random seeds recorded.
The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup. Specific decision context: **Model environment.** A versioned Python environment fits scalers and clustering candidates from a frozen training snapshot, with fixed random seeds recorded.

All production-like names are illustrative roles, not assertions about a particular vendor or company architecture. Specific decision context: **Model environment.** A versioned Python environment fits scalers and clustering candidates from a frozen training snapshot, with fixed random seeds recorded.
**Artifact registry.** The registry stores feature schema hash, reference cohort hash, preprocessing parameters, centroid matrix, fit date, and approval status.
The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup. Specific decision context: **Artifact registry.** The registry stores feature schema hash, reference cohort hash, preprocessing parameters, centroid matrix, fit date, and approval status.
All production-like names are illustrative roles, not assertions about a particular vendor or company architecture. Specific decision context: **Artifact registry.** The registry stores feature schema hash, reference cohort hash, preprocessing parameters, centroid matrix, fit date, and approval status.

**Assignment job.** A batch job applies the approved scaler and centroids to the current eligible snapshot, retaining nearest and second-nearest distances.
The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup. Specific decision context: **Assignment job.** A batch job applies the approved scaler and centroids to the current eligible snapshot, retaining nearest and second-nearest distances.
All production-like names are illustrative roles, not assertions about a particular vendor or company architecture. Specific decision context: **Assignment job.** A batch job applies the approved scaler and centroids to the current eligible snapshot, retaining nearest and second-nearest distances.
**Policy service.** A deterministic rules layer joins assignments to consent, frequency, inventory, safety suppressions, and offer constraints before a send list exists.

The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup. Specific decision context: **Policy service.** A deterministic rules layer joins assignments to consent, frequency, inventory, safety suppressions, and offer constraints before a send list exists.
All production-like names are illustrative roles, not assertions about a particular vendor or company architecture. Specific decision context: **Policy service.** A deterministic rules layer joins assignments to consent, frequency, inventory, safety suppressions, and offer constraints before a send list exists.
**Experiment service.** A seeded randomizer assigns eligible policy-qualified records to treatment or control and writes assignment before campaign delivery.
The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup. Specific decision context: **Experiment service.** A seeded randomizer assigns eligible policy-qualified records to treatment or control and writes assignment before campaign delivery.

All production-like names are illustrative roles, not assertions about a particular vendor or company architecture. Specific decision context: **Experiment service.** A seeded randomizer assigns eligible policy-qualified records to treatment or control and writes assignment before campaign delivery.
**Delivery platform.** The channel platform receives only a minimized send payload, offer code, expiry, and campaign identifier rather than the feature vector.
The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup. Specific decision context: **Delivery platform.** The channel platform receives only a minimized send payload, offer code, expiry, and campaign identifier rather than the feature vector.
All production-like names are illustrative roles, not assertions about a particular vendor or company architecture. Specific decision context: **Delivery platform.** The channel platform receives only a minimized send payload, offer code, expiry, and campaign identifier rather than the feature vector.

**Measurement store.** Exposure, delivery, conversion, refunds, and opt-out events are joined through campaign and household keys after the attribution window.
The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup. Specific decision context: **Measurement store.** Exposure, delivery, conversion, refunds, and opt-out events are joined through campaign and household keys after the attribution window.
All production-like names are illustrative roles, not assertions about a particular vendor or company architecture. Specific decision context: **Measurement store.** Exposure, delivery, conversion, refunds, and opt-out events are joined through campaign and household keys after the attribution window.
**Audit store.** An append-only decision log stores the rule path and versions required to reconstruct each inclusion, exclusion, or fallback action.

The component boundary exists to make errors attributable: a bad cluster fit should not be indistinguishable from a stale consent lookup. Specific decision context: **Audit store.** An append-only decision log stores the rule path and versions required to reconstruct each inclusion, exclusion, or fallback action.
All production-like names are illustrative roles, not assertions about a particular vendor or company architecture. Specific decision context: **Audit store.** An append-only decision log stores the rule path and versions required to reconstruct each inclusion, exclusion, or fallback action.
### 5c Method Summary

**Method step.** Create a feature snapshot from records that satisfy the training population rule and are complete before the reference cut.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage.

**Method step.** Inspect distributions, missingness, duplicate keys, and event lag before any scaling or clustering candidate is fit.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Inspect distributions, missingness, duplicate keys, and event lag before any scaling or clustering candidate is fit.
**Method step.** Winsorize extreme spend only according to a predeclared training quantile, then apply the same bound to future assignments.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Winsorize extreme spend only according to a predeclared training quantile, then apply the same bound to future assignments.

**Method step.** Transform skewed nonnegative frequency and spend features with log one plus x when the transformation is defined and recorded.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Transform skewed nonnegative frequency and spend features with log one plus x when the transformation is defined and recorded.
**Method step.** Standardize selected numeric features using training-set mean and standard deviation, never statistics recalculated from the campaign cohort.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Standardize selected numeric features using training-set mean and standard deviation, never statistics recalculated from the campaign cohort.

**Method step.** Fit candidate k-means solutions for k from two through six with multiple deterministic-seed restarts and record inertia, silhouette, and cluster sizes.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Fit candidate k-means solutions for k from two through six with multiple deterministic-seed restarts and record inertia, silhouette, and cluster sizes.
**Method step.** Reject a candidate that has attractive internal metrics but contains a tiny cluster that cannot support a distinguishable action or stable evaluation.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Reject a candidate that has attractive internal metrics but contains a tiny cluster that cannot support a distinguishable action or stable evaluation.

**Method step.** Choose the smallest candidate whose groups have coherent pre-decision behaviour, acceptable resample stability, and usable action differentiation.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Choose the smallest candidate whose groups have coherent pre-decision behaviour, acceptable resample stability, and usable action differentiation.
**Method step.** Name clusters only after reviewing aggregate feature profiles; names must describe observed patterns rather than speculate about motivation or identity.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Name clusters only after reviewing aggregate feature profiles; names must describe observed patterns rather than speculate about motivation or identity.

**Method step.** Assign each new eligible record to the nearest approved centroid using the frozen preprocessing parameters.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Assign each new eligible record to the nearest approved centroid using the frozen preprocessing parameters.
**Method step.** Route records with missing required features, ambiguous identity, or a small distance margin to an unassigned policy rather than fabricate precision.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Route records with missing required features, ambiguous identity, or a small distance margin to an unassigned policy rather than fabricate precision.

**Method step.** Apply action eligibility after assignment, because a stable segment does not waive consent, inventory, legal, or frequency limits.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Apply action eligibility after assignment, because a stable segment does not waive consent, inventory, legal, or frequency limits.
**Method step.** Randomize the final policy-qualified set to an action and an untreated or broad-message comparator before delivery.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Randomize the final policy-qualified set to an action and an untreated or broad-message comparator before delivery.

**Method step.** Evaluate incremental contribution and guardrails over prespecified windows, then decide whether the policy version remains eligible for future tests.
Its output is persisted with a version and a row count so the next stage can be reconciled to the prior stage. Specific decision context: **Method step.** Evaluate incremental contribution and guardrails over prespecified windows, then decide whether the policy version remains eligible for future tests.
### 5d Provenance, Quality, Exclusions, Leakage

**Late orders.** An order posted after the feature cutoff is excluded from the current run even if its purchase event occurred before cutoff, and its lag is logged.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels.

**Returns timing.** A return initiated after cutoff cannot revise the historical feature snapshot; it appears in the next valid snapshot and in outcome accounting when relevant.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Returns timing.** A return initiated after cutoff cannot revise the historical feature snapshot; it appears in the next valid snapshot and in outcome accounting when relevant.
**Consent withdrawal.** A withdrawal after assignment but before delivery removes the customer from the send file and is counted as a pre-send exclusion, not a nonresponse.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Consent withdrawal.** A withdrawal after assignment but before delivery removes the customer from the send file and is counted as a pre-send exclusion, not a nonresponse.

**Account merge.** A merge event after the cutoff triggers reassignment only in the next run; the prior decision log retains the pre-merge identity state.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Account merge.** A merge event after the cutoff triggers reassignment only in the next run; the prior decision log retains the pre-merge identity state.
**Duplicate ingestion.** Repeated order events are deduplicated by immutable transaction event ID before customer aggregation, with duplicate counts monitored by source.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Duplicate ingestion.** Repeated order events are deduplicated by immutable transaction event ID before customer aggregation, with duplicate counts monitored by source.

**Currency correction.** A correction feed is versioned with its effective timestamp; reprocessing cannot silently replace the rate used in an earlier audit snapshot.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Currency correction.** A correction feed is versioned with its effective timestamp; reprocessing cannot silently replace the rate used in an earlier audit snapshot.
**Campaign leakage.** Prior campaign response and current campaign exposure are excluded from clustering features to prevent a past targeting rule from defining the new groups.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Campaign leakage.** Prior campaign response and current campaign exposure are excluded from clustering features to prevent a past targeting rule from defining the new groups.

**Selection leakage.** Outcome data from any customer excluded by consent or frequency policy cannot be used to claim the action policy would have worked for that excluded group.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Selection leakage.** Outcome data from any customer excluded by consent or frequency policy cannot be used to claim the action policy would have worked for that excluded group.
**Join loss.** Records that fail identity resolution are quantified by channel and source rather than dropped without a denominator.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Join loss.** Records that fail identity resolution are quantified by channel and source rather than dropped without a denominator.

**Missingness.** A missing order history has a different meaning from a zero-spend history, so a separate availability flag accompanies imputed numeric values.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Missingness.** A missing order history has a different meaning from a zero-spend history, so a separate availability flag accompanies imputed numeric values.
**Reference drift.** The scaler's reference mean and standard deviation remain fixed for a model version; a new reference distribution requires a newly approved version.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Reference drift.** The scaler's reference mean and standard deviation remain fixed for a model version; a new reference distribution requires a newly approved version.

**Manual edits.** Analyst overrides are prohibited in the fitting dataset; approved campaign suppressions are applied downstream and logged separately.
The discriminating check is a dated source-to-snapshot reconciliation rather than a visual inspection of cluster labels. Specific decision context: **Manual edits.** Analyst overrides are prohibited in the fitting dataset; approved campaign suppressions are applied downstream and logged separately.
### 5e Reproducibility & Method-Selection Ledger

**Reference cohort.** Retain Eligible training records from a defined historical cut, with a saved row count and ordered household-key hash.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables.

**Feature recipe.** Retain Feature names, SQL revision, units, clipping thresholds, transformations, and missing-value policy.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables. Specific decision context: **Feature recipe.** Retain Feature names, SQL revision, units, clipping thresholds, transformations, and missing-value policy.
**Preprocessing.** Retain Training means, standard deviations, category vocabulary, and the exact order of columns passed to the estimator.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables. Specific decision context: **Preprocessing.** Retain Training means, standard deviations, category vocabulary, and the exact order of columns passed to the estimator.

**Candidate search.** Retain Candidate k values, initialization count, random seeds, convergence tolerance, and maximum iterations.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables. Specific decision context: **Candidate search.** Retain Candidate k values, initialization count, random seeds, convergence tolerance, and maximum iterations.
**Selection rationale.** Retain The smallest usable k selected after inspecting stability, action separation, size balance, and reviewability rather than one metric alone.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables. Specific decision context: **Selection rationale.** Retain The smallest usable k selected after inspecting stability, action separation, size balance, and reviewability rather than one metric alone.

**Centroid artifact.** Retain A matrix of standardized centroid coordinates with model version, fit timestamp, and artifact checksum.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables. Specific decision context: **Centroid artifact.** Retain A matrix of standardized centroid coordinates with model version, fit timestamp, and artifact checksum.
**Assignment record.** Retain Per-row nearest centroid, nearest distance, second-nearest distance, margin, assignment state, and feature-quality flags.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables. Specific decision context: **Assignment record.** Retain Per-row nearest centroid, nearest distance, second-nearest distance, margin, assignment state, and feature-quality flags.

**Policy matrix.** Retain Segment-to-action eligibility, priority score definition, cap rule, suppressions, fallback, and named approving owner.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables. Specific decision context: **Policy matrix.** Retain Segment-to-action eligibility, priority score definition, cap rule, suppressions, fallback, and named approving owner.
**Experiment protocol.** Retain Randomization seed, allocation ratio, eligibility population, primary metric, guardrails, and evaluation close date.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables. Specific decision context: **Experiment protocol.** Retain Randomization seed, allocation ratio, eligibility population, primary metric, guardrails, and evaluation close date.

**Decision record.** Retain Whether the version was retained, changed, paused, or retired, with the evidence boundary clearly marked as illustrative here.
Reproducibility means a reviewer can regenerate the same decision from the same snapshot, not merely rerun code against today's changing tables. Specific decision context: **Decision record.** Retain Whether the version was retained, changed, paused, or retired, with the evidence boundary clearly marked as illustrative here.
## 6. Analysis Walkthrough

The walkthrough uses a constructed reference cohort of 24,000 identity-resolved and consent-eligible households before frequency and campaign-specific exclusions.
The examples demonstrate arithmetic and decision logic only; they do not estimate a real retailer's response or profit.

First, construct time-safe features; second, assign clusters using frozen centroids; third, rank policy-qualified customers under capacity; finally, evaluate a randomized campaign.
A boundary condition accompanies each formula because an otherwise correct calculation can be invalid when its denominator, time cut, or decision population changes.
The chosen illustrative three-group solution is labelled replenishment-oriented, occasion-led, and infrequent-value-seeking solely from aggregate behavioural profiles.
Labels do not imply that every member wants that category or that a person will respond to the corresponding offer.

```text
FORMULA → z_ij = (x_ij - mu_j) / sigma_j
WHERE → x_ij is household i's 180-day net spend in currency units; mu_j and sigma_j are the training-cohort mean and standard deviation for spend; z_ij is unitless.
CONSTRAINT → sigma_j must be positive and mu_j, sigma_j must be frozen from the valid training cut; do not estimate them from the current campaign send file.
NUMERICAL EXAMPLE → ILLUSTRATIVE: x = 180 currency units, mu = 120 currency units, sigma = 40 currency units.
NUMERICAL EXAMPLE → z = (180 - 120) / 40 = 60 / 40 = 1.50 unitless.
INTERPRETATION → This household is 1.50 training standard deviations above mean spend; it is not automatically a high-value customer or an offer mandate.
BOUNDARY → If every reference household has identical spend then sigma = 0, the standardized value is undefined; remove or redesign that feature rather than divide by zero.
```
```text
FORMULA → d_ik^2 = sum over j of (z_ij - c_kj)^2
WHERE → z_i is household i's standardized feature vector; c_k is centroid k in the same feature order and scale; d_ik is a unitless Euclidean distance.
CONSTRAINT → Compare only vectors with the identical frozen feature schema and imputation treatment; distance has no meaning across unmatched columns or scales.
NUMERICAL EXAMPLE → ILLUSTRATIVE: for two retained features, z_i = (1.50, -0.50) and centroid A = (1.00, 0.00).
NUMERICAL EXAMPLE → d_iA^2 = (1.50 - 1.00)^2 + (-0.50 - 0.00)^2 = 0.25 + 0.25 = 0.50.
NUMERICAL EXAMPLE → d_iA = square root of 0.50 = 0.707 unitless; the assignment compares this result against every approved centroid.
INTERPRETATION → The nearest centroid yields a descriptive assignment, while the gap to the second nearest centroid communicates assignment certainty.
BOUNDARY → A centroid distance is unreliable when missing features were imputed from a distribution unlike the household's actual history; route severe missingness to unassigned.
```
```text
FORMULA → m_i = d_i(second) - d_i(first)
WHERE → d_i(first) is the nearest-centroid distance and d_i(second) is the second-nearest distance for household i; m_i is a unitless assignment margin.
CONSTRAINT → The nearest and second-nearest distances must arise from the same frozen centroid set; m_i is not comparable across unrelated model versions without calibration.
NUMERICAL EXAMPLE → ILLUSTRATIVE: nearest distance = 0.707 and second-nearest distance = 0.860.
NUMERICAL EXAMPLE → m_i = 0.860 - 0.707 = 0.153 unitless.
NUMERICAL EXAMPLE → If the policy threshold is m_i >= 0.120, then 0.153 >= 0.120 and the record is assignment-eligible, subject to all downstream gates.
INTERPRETATION → A larger margin means the closest available description is comparatively clearer, not that an offer will perform better.
BOUNDARY → When two centroids are almost identical, high assignment counts can coexist with tiny margins; refit or reduce k instead of pretending the labels are distinct.
```
```text
FORMULA → E[IC_s] = p_s * (M_s - D_s) - C_send
WHERE → p_s is an illustrative incremental conversion probability for policy cell s; M_s is contribution before discount; D_s is discount funding; C_send is per-message cost, all currency units except p_s.
CONSTRAINT → p_s must come from a randomized comparison for the same eligible population and offer; a cluster's observed purchase rate is not a substitute.
NUMERICAL EXAMPLE → ILLUSTRATIVE: p_s = 0.018, M_s = 32.00 currency units, D_s = 6.00 currency units, and C_send = 0.08 currency units.
NUMERICAL EXAMPLE → E[IC_s] = 0.018 * (32.00 - 6.00) - 0.08 = 0.018 * 26.00 - 0.08 = 0.468 - 0.08 = 0.388 currency units per eligible send.
INTERPRETATION → At this constructed estimate the cell has positive expected incremental contribution before uncertainty and guardrails are considered.
BOUNDARY → If stock is constrained, incremental purchases can displace full-price purchases; the simple expression then overstates incremental value and requires inventory-aware evaluation.
```
**Walkthrough control.** Candidate k equals two merged households with different category patterns into one broad group, which made the action matrix too blunt for the stated objective.
The result is accepted only if its inputs and decision boundary are documented in the run record.
**Walkthrough control.** Candidate k equals four separated a small high-return subgroup whose profile could not support a distinct ethical or economical offer, so it failed actionability review.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** Candidate k equals four separated a small high-return subgroup whose profile could not support a distinct ethical or economical offer, so it failed actionability review.

**Walkthrough control.** The constructed selection of k equals three is therefore a governance decision supported by internal diagnostics, not a discovery of three natural human types.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** The constructed selection of k equals three is therefore a governance decision supported by internal diagnostics, not a discovery of three natural human types.
**Walkthrough control.** Replenishment-oriented households have shorter recency and repeat purchases in a limited set of consumable categories in the illustrative profile.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** Replenishment-oriented households have shorter recency and repeat purchases in a limited set of consumable categories in the illustrative profile.

**Walkthrough control.** Occasion-led households show concentrated seasonal purchase dates and broader category baskets, which may reflect events rather than a persistent personal trait.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** Occasion-led households show concentrated seasonal purchase dates and broader category baskets, which may reflect events rather than a persistent personal trait.
**Walkthrough control.** Infrequent-value-seeking households have longer recency and lower order count but a higher share of discount-eligible orders in the constructed snapshot.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** Infrequent-value-seeking households have longer recency and lower order count but a higher share of discount-eligible orders in the constructed snapshot.

**Walkthrough control.** The first policy maps replenishment-oriented households to a reminder without a discount when permitted inventory supports their usual category family.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** The first policy maps replenishment-oriented households to a reminder without a discount when permitted inventory supports their usual category family.
**Walkthrough control.** The second policy maps occasion-led households to a curated category message only when the current catalog has a nonrestricted seasonal assortment.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** The second policy maps occasion-led households to a curated category message only when the current catalog has a nonrestricted seasonal assortment.

**Walkthrough control.** The third policy proposes a bounded discount test for infrequent-value-seeking households, but only within the funded offer cap and randomized evaluation design.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** The third policy proposes a bounded discount test for infrequent-value-seeking households, but only within the funded offer cap and randomized evaluation design.
**Walkthrough control.** The unassigned policy gives no promotional message by default, preventing low-confidence assignment from becoming a hidden catch-all treatment.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** The unassigned policy gives no promotional message by default, preventing low-confidence assignment from becoming a hidden catch-all treatment.

**Walkthrough control.** Assignment counts are reconciled across eligible, assigned, uncertain, suppressed, and capacity-excluded states so the denominator never disappears.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** Assignment counts are reconciled across eligible, assigned, uncertain, suppressed, and capacity-excluded states so the denominator never disappears.
**Walkthrough control.** A cluster profile table is reviewed with data stewardship and marketing to ensure names reflect aggregates and do not make claims about individuals.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** A cluster profile table is reviewed with data stewardship and marketing to ensure names reflect aggregates and do not make claims about individuals.

**Walkthrough control.** The action matrix is deliberately downstream of the centroids, allowing an offer change without refitting a descriptive model.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** The action matrix is deliberately downstream of the centroids, allowing an offer change without refitting a descriptive model.
**Walkthrough control.** Every calculation is re-runnable from stored features and parameters; manual spreadsheet centroid edits are outside the approved method.
The result is accepted only if its inputs and decision boundary are documented in the run record. Specific decision context: **Walkthrough control.** Every calculation is re-runnable from stored features and parameters; manual spreadsheet centroid edits are outside the approved method.

## 7. The Finding

**Finding.** The constructed analysis supports a three-segment descriptive view only when assignment uncertainty and downstream eligibility are retained.
This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed.
The relevant owner records whether the condition remains true at the next approved refresh.
**Finding.** No segment should receive an automatic campaign solely because it is closest to a centroid; consent, contact pressure, inventory, and safety checks remain independent gates.

This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed. Specific decision context: **Finding.** No segment should receive an automatic campaign solely because it is closest to a centroid; consent, contact pressure, inventory, and safety checks remain independent gates.
The relevant owner records whether the condition remains true at the next approved refresh. Specific decision context: **Finding.** No segment should receive an automatic campaign solely because it is closest to a centroid; consent, contact pressure, inventory, and safety checks remain independent gates.
**Finding.** The smallest action-ready structure is preferable to a more granular solution when extra clusters do not change a permissible campaign decision.
This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed. Specific decision context: **Finding.** The smallest action-ready structure is preferable to a more granular solution when extra clusters do not change a permissible campaign decision.

The relevant owner records whether the condition remains true at the next approved refresh. Specific decision context: **Finding.** The smallest action-ready structure is preferable to a more granular solution when extra clusters do not change a permissible campaign decision.
**Finding.** A stable segment profile can justify a hypothesis about message relevance, but a randomized experiment is still required before claiming incremental contribution.
This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed. Specific decision context: **Finding.** A stable segment profile can justify a hypothesis about message relevance, but a randomized experiment is still required before claiming incremental contribution.
The relevant owner records whether the condition remains true at the next approved refresh. Specific decision context: **Finding.** A stable segment profile can justify a hypothesis about message relevance, but a randomized experiment is still required before claiming incremental contribution.

**Finding.** The capacity-constrained policy should rank eligible cells by predeclared expected incremental contribution only after its inputs are supported by valid experiment evidence.
This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed. Specific decision context: **Finding.** The capacity-constrained policy should rank eligible cells by predeclared expected incremental contribution only after its inputs are supported by valid experiment evidence.
The relevant owner records whether the condition remains true at the next approved refresh. Specific decision context: **Finding.** The capacity-constrained policy should rank eligible cells by predeclared expected incremental contribution only after its inputs are supported by valid experiment evidence.
**Finding.** Until such evidence exists, this teaching case uses the ranking rule as a proposed decision design rather than a deployed optimization.

This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed. Specific decision context: **Finding.** Until such evidence exists, this teaching case uses the ranking rule as a proposed decision design rather than a deployed optimization.
The relevant owner records whether the condition remains true at the next approved refresh. Specific decision context: **Finding.** Until such evidence exists, this teaching case uses the ranking rule as a proposed decision design rather than a deployed optimization.
**Finding.** The unassigned route is a finding in its own right: uncertain or incomplete records must not be treated as harmless noise.
This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed. Specific decision context: **Finding.** The unassigned route is a finding in its own right: uncertain or incomplete records must not be treated as harmless noise.

The relevant owner records whether the condition remains true at the next approved refresh. Specific decision context: **Finding.** The unassigned route is a finding in its own right: uncertain or incomplete records must not be treated as harmless noise.
**Finding.** Identity confidence is more consequential than a small change in silhouette when a bad merge can send two different offers to the same household.
This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed. Specific decision context: **Finding.** Identity confidence is more consequential than a small change in silhouette when a bad merge can send two different offers to the same household.
The relevant owner records whether the condition remains true at the next approved refresh. Specific decision context: **Finding.** Identity confidence is more consequential than a small change in silhouette when a bad merge can send two different offers to the same household.

**Finding.** A segment's size is not an entitlement to budget; a large group can be deprioritized if its action has weak evidence or a guardrail risk.
This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed. Specific decision context: **Finding.** A segment's size is not an entitlement to budget; a large group can be deprioritized if its action has weak evidence or a guardrail risk.
The relevant owner records whether the condition remains true at the next approved refresh. Specific decision context: **Finding.** A segment's size is not an entitlement to budget; a large group can be deprioritized if its action has weak evidence or a guardrail risk.
**Finding.** The evaluation unit must remain the eligible household, not the cluster alone, because treatment eligibility changes the population that can be observed.

This statement is limited to the illustrated design and must be replaced by observed evidence before a production decision is claimed. Specific decision context: **Finding.** The evaluation unit must remain the eligible household, not the cluster alone, because treatment eligibility changes the population that can be observed.
The relevant owner records whether the condition remains true at the next approved refresh. Specific decision context: **Finding.** The evaluation unit must remain the eligible household, not the cluster alone, because treatment eligibility changes the population that can be observed.
## 8. System/Solution Design

**Snapshot freeze.** Close the feature snapshot at the declared event-time cut and attach the source freshness report before fitting or assignment.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label.

A reconciliation count is emitted at this boundary to make silent record loss detectable.
**Eligibility gate.** Evaluate durable identity, active consent, legal suppression, service protection, and frequency cap before the policy can select a household.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Eligibility gate.** Evaluate durable identity, active consent, legal suppression, service protection, and frequency cap before the policy can select a household.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Eligibility gate.** Evaluate durable identity, active consent, legal suppression, service protection, and frequency cap before the policy can select a household.

**Feature validator.** Reject a run when a required column is absent, its unit changes, its valid range is breached materially, or duplicate household keys remain.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Feature validator.** Reject a run when a required column is absent, its unit changes, its valid range is breached materially, or duplicate household keys remain.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Feature validator.** Reject a run when a required column is absent, its unit changes, its valid range is breached materially, or duplicate household keys remain.
**Model artifact.** Load only a signed-off centroid and preprocessing artifact whose schema hash matches the feature snapshot.

The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Model artifact.** Load only a signed-off centroid and preprocessing artifact whose schema hash matches the feature snapshot.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Model artifact.** Load only a signed-off centroid and preprocessing artifact whose schema hash matches the feature snapshot.
**Uncertainty gate.** Compare first and second centroid distances and route low-margin records to unassigned rather than assigning by a barely smaller value.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Uncertainty gate.** Compare first and second centroid distances and route low-margin records to unassigned rather than assigning by a barely smaller value.

A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Uncertainty gate.** Compare first and second centroid distances and route low-margin records to unassigned rather than assigning by a barely smaller value.
**Action matrix.** Join an approved segment-action matrix that contains allowed offer families, exclusion conditions, expiry, and the accountable marketing owner.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Action matrix.** Join an approved segment-action matrix that contains allowed offer families, exclusion conditions, expiry, and the accountable marketing owner.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Action matrix.** Join an approved segment-action matrix that contains allowed offer families, exclusion conditions, expiry, and the accountable marketing owner.

**Inventory gate.** Recheck available-to-promise quantity and supplier restrictions at campaign build time because they can change after segment assignment.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Inventory gate.** Recheck available-to-promise quantity and supplier restrictions at campaign build time because they can change after segment assignment.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Inventory gate.** Recheck available-to-promise quantity and supplier restrictions at campaign build time because they can change after segment assignment.
**Capacity allocator.** Sort policy-qualified records by an approved priority expression and take no more than the current channel capacity after deduplication.

The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Capacity allocator.** Sort policy-qualified records by an approved priority expression and take no more than the current channel capacity after deduplication.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Capacity allocator.** Sort policy-qualified records by an approved priority expression and take no more than the current channel capacity after deduplication.
**Household deduplicator.** Resolve competing assignments across linked accounts so one household receives one compatible action in the campaign period.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Household deduplicator.** Resolve competing assignments across linked accounts so one household receives one compatible action in the campaign period.

A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Household deduplicator.** Resolve competing assignments across linked accounts so one household receives one compatible action in the campaign period.
**Randomizer.** Use a stored seed and stratified allocation by action cell when a comparison is required, preserving a treatment and control denominator.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Randomizer.** Use a stored seed and stratified allocation by action cell when a comparison is required, preserving a treatment and control denominator.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Randomizer.** Use a stored seed and stratified allocation by action cell when a comparison is required, preserving a treatment and control denominator.

**Delivery handoff.** Transmit a minimized payload with recipient token, campaign ID, action code, and expiry; do not expose cluster features to the channel vendor.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Delivery handoff.** Transmit a minimized payload with recipient token, campaign ID, action code, and expiry; do not expose cluster features to the channel vendor.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Delivery handoff.** Transmit a minimized payload with recipient token, campaign ID, action code, and expiry; do not expose cluster features to the channel vendor.
**Pre-send recheck.** Re-evaluate consent, suppression, and frequency immediately before delivery because these conditions can change after file assembly.

The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Pre-send recheck.** Re-evaluate consent, suppression, and frequency immediately before delivery because these conditions can change after file assembly.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Pre-send recheck.** Re-evaluate consent, suppression, and frequency immediately before delivery because these conditions can change after file assembly.
**Exposure logger.** Write planned, accepted, delivered, bounced, and suppressed states separately so non-delivery is not mislabelled as no response.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Exposure logger.** Write planned, accepted, delivered, bounced, and suppressed states separately so non-delivery is not mislabelled as no response.

A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Exposure logger.** Write planned, accepted, delivered, bounced, and suppressed states separately so non-delivery is not mislabelled as no response.
**Outcome linker.** Join conversion and contribution outcomes to the frozen experiment assignment, while keeping returns and refunds in a later maturation view.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Outcome linker.** Join conversion and contribution outcomes to the frozen experiment assignment, while keeping returns and refunds in a later maturation view.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Outcome linker.** Join conversion and contribution outcomes to the frozen experiment assignment, while keeping returns and refunds in a later maturation view.

**Decision log.** Persist feature cut, identity version, policy version, cluster ID, distance margin, rule path, and operator override reason per candidate.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Decision log.** Persist feature cut, identity version, policy version, cluster ID, distance margin, rule path, and operator override reason per candidate.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Decision log.** Persist feature cut, identity version, policy version, cluster ID, distance margin, rule path, and operator override reason per candidate.
**Access control.** Limit raw identity links and audit attributes to authorized roles; campaign operators need action outputs, not unnecessary personal data.

The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Access control.** Limit raw identity links and audit attributes to authorized roles; campaign operators need action outputs, not unnecessary personal data.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Access control.** Limit raw identity links and audit attributes to authorized roles; campaign operators need action outputs, not unnecessary personal data.
**Version rollback.** Support rollback to the last approved policy artifact while retaining the failed run for investigation and audit.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Version rollback.** Support rollback to the last approved policy artifact while retaining the failed run for investigation and audit.

A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Version rollback.** Support rollback to the last approved policy artifact while retaining the failed run for investigation and audit.
**Monitoring board.** Show funnel counts, data freshness, assignment distribution, uncertainty share, cap pressure, and guardrail events for each campaign run.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Monitoring board.** Show funnel counts, data freshness, assignment distribution, uncertainty share, cap pressure, and guardrail events for each campaign run.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Monitoring board.** Show funnel counts, data freshness, assignment distribution, uncertainty share, cap pressure, and guardrail events for each campaign run.

**Approval gate.** Require named approval for a new action matrix even when the clustering artifact has not changed, because policy changes alter customer treatment.
The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Approval gate.** Require named approval for a new action matrix even when the clustering artifact has not changed, because policy changes alter customer treatment.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Approval gate.** Require named approval for a new action matrix even when the clustering artifact has not changed, because policy changes alter customer treatment.
**Retention boundary.** Purge or archive decision artifacts according to retention policy while preserving the minimum audit record needed to explain a send decision.

The interface has a single explicit failure state so downstream systems do not interpret an unavailable result as a valid negative label. Specific decision context: **Retention boundary.** Purge or archive decision artifacts according to retention policy while preserving the minimum audit record needed to explain a send decision.
A reconciliation count is emitted at this boundary to make silent record loss detectable. Specific decision context: **Retention boundary.** Purge or archive decision artifacts according to retention policy while preserving the minimum audit record needed to explain a send decision.
## 9. Evaluation & Validation

**Primary estimand.** Incremental net contribution per eligible randomized household over fourteen days, defined as treatment mean net contribution minus comparator mean net contribution.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning.

A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail.
**Numerator.** For each arm, sum completed-order contribution less funded discount, message cost, and attributed returns for households in the frozen eligible population.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Numerator.** For each arm, sum completed-order contribution less funded discount, message cost, and attributed returns for households in the frozen eligible population.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Numerator.** For each arm, sum completed-order contribution less funded discount, message cost, and attributed returns for households in the frozen eligible population.

**Denominator.** Use all randomized eligible households in the assigned arm, including undelivered records when the estimand is intent-to-treat.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Denominator.** Use all randomized eligible households in the assigned arm, including undelivered records when the estimand is intent-to-treat.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Denominator.** Use all randomized eligible households in the assigned arm, including undelivered records when the estimand is intent-to-treat.
**Population.** The analysis population is policy-qualified, consented, identity-resolved households that survived all pre-randomization exclusions at the run cut.

The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Population.** The analysis population is policy-qualified, consented, identity-resolved households that survived all pre-randomization exclusions at the run cut.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Population.** The analysis population is policy-qualified, consented, identity-resolved households that survived all pre-randomization exclusions at the run cut.
**Window.** Measure conversion and contribution for fourteen days after scheduled send, then mature returns for twenty-eight additional days before the final readout.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Window.** Measure conversion and contribution for fourteen days after scheduled send, then mature returns for twenty-eight additional days before the final readout.

A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Window.** Measure conversion and contribution for fourteen days after scheduled send, then mature returns for twenty-eight additional days before the final readout.
**Comparator.** Use randomized no-message or approved broad-message control with the same eligibility and measurement rules, not last month's different audience.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Comparator.** Use randomized no-message or approved broad-message control with the same eligibility and measurement rules, not last month's different audience.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Comparator.** Use randomized no-message or approved broad-message control with the same eligibility and measurement rules, not last month's different audience.

**Randomization.** Seeded assignment occurs within action cell and broad recency band so balance is inspectable without conditioning on post-treatment outcomes.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Randomization.** Seeded assignment occurs within action cell and broad recency band so balance is inspectable without conditioning on post-treatment outcomes.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Randomization.** Seeded assignment occurs within action cell and broad recency band so balance is inspectable without conditioning on post-treatment outcomes.
**Uncertainty.** Report a confidence interval or randomization-compatible interval around the arm difference and state the chosen confidence level before analysis.

The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Uncertainty.** Report a confidence interval or randomization-compatible interval around the arm difference and state the chosen confidence level before analysis.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Uncertainty.** Report a confidence interval or randomization-compatible interval around the arm difference and state the chosen confidence level before analysis.
**Capacity fairness.** Compare the share selected after ranking with the share eligible by lawful audit groups where data collection and use are permitted.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Capacity fairness.** Compare the share selected after ranking with the share eligible by lawful audit groups where data collection and use are permitted.

A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Capacity fairness.** Compare the share selected after ranking with the share eligible by lawful audit groups where data collection and use are permitted.
**Contact guardrail.** Track unsubscribe rate per delivered message and compare treatment with control over the same fourteen-day window.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Contact guardrail.** Track unsubscribe rate per delivered message and compare treatment with control over the same fourteen-day window.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Contact guardrail.** Track unsubscribe rate per delivered message and compare treatment with control over the same fourteen-day window.

**Complaint guardrail.** Track verified campaign-related complaints per thousand delivered messages and investigate any material treatment-control difference.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Complaint guardrail.** Track verified campaign-related complaints per thousand delivered messages and investigate any material treatment-control difference.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Complaint guardrail.** Track verified campaign-related complaints per thousand delivered messages and investigate any material treatment-control difference.
**Frequency guardrail.** Verify zero messages beyond the stated seven-day and thirty-day caps; this is a hard compliance count, not a significance test.

The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Frequency guardrail.** Verify zero messages beyond the stated seven-day and thirty-day caps; this is a hard compliance count, not a significance test.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Frequency guardrail.** Verify zero messages beyond the stated seven-day and thirty-day caps; this is a hard compliance count, not a significance test.
**Identity guardrail.** Measure duplicate-household deliveries and incompatible-offer collisions per campaign; the target is zero within the deterministic linkage scope.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Identity guardrail.** Measure duplicate-household deliveries and incompatible-offer collisions per campaign; the target is zero within the deterministic linkage scope.

A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Identity guardrail.** Measure duplicate-household deliveries and incompatible-offer collisions per campaign; the target is zero within the deterministic linkage scope.
**Economic guardrail.** Track discount redemption, gross margin, stockout substitutions, and return rate so an apparent conversion increase is not mistaken for contribution.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Economic guardrail.** Track discount redemption, gross margin, stockout substitutions, and return rate so an apparent conversion increase is not mistaken for contribution.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Economic guardrail.** Track discount redemption, gross margin, stockout substitutions, and return rate so an apparent conversion increase is not mistaken for contribution.

**Stability validation.** Compare resampled cluster solutions using adjusted agreement after centroid matching, and inspect whether action cells remain interpretable.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Stability validation.** Compare resampled cluster solutions using adjusted agreement after centroid matching, and inspect whether action cells remain interpretable.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Stability validation.** Compare resampled cluster solutions using adjusted agreement after centroid matching, and inspect whether action cells remain interpretable.
**Temporal validation.** Assign a later holdout snapshot with the frozen artifact and compare feature distributions, assignment shares, and margin distributions.

The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Temporal validation.** Assign a later holdout snapshot with the frozen artifact and compare feature distributions, assignment shares, and margin distributions.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Temporal validation.** Assign a later holdout snapshot with the frozen artifact and compare feature distributions, assignment shares, and margin distributions.
**Sensitivity check.** Vary predeclared winsorization and missingness thresholds to see whether the action matrix changes for a material share of eligible customers.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Sensitivity check.** Vary predeclared winsorization and missingness thresholds to see whether the action matrix changes for a material share of eligible customers.

A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Sensitivity check.** Vary predeclared winsorization and missingness thresholds to see whether the action matrix changes for a material share of eligible customers.
**Negative check.** Confirm that post-send events do not appear in the assignment table and that a future-dated record cannot alter an earlier decision log.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Negative check.** Confirm that post-send events do not appear in the assignment table and that a future-dated record cannot alter an earlier decision log.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Negative check.** Confirm that post-send events do not appear in the assignment table and that a future-dated record cannot alter an earlier decision log.

**Execution status.** These checks are specified for the illustrative design and were not executed on a real campaign dataset.
The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Execution status.** These checks are specified for the illustrative design and were not executed on a real campaign dataset.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Execution status.** These checks are specified for the illustrative design and were not executed on a real campaign dataset.
**Decision rule.** Keep a policy only if the primary estimate, uncertainty, mandatory guardrails, data validity, and operational feasibility jointly meet the predeclared gate.

The metric definition includes its numerator, denominator, time window, and population so a dashboard cannot silently change its meaning. Specific decision context: **Decision rule.** Keep a policy only if the primary estimate, uncertainty, mandatory guardrails, data validity, and operational feasibility jointly meet the predeclared gate.
A favorable descriptive cluster statistic cannot override a failed customer-safety or consent guardrail. Specific decision context: **Decision rule.** Keep a policy only if the primary estimate, uncertainty, mandatory guardrails, data validity, and operational feasibility jointly meet the predeclared gate.
## 10. Action Taken

**Operational action.** Freeze the weekly feature snapshot before any campaign eligibility query is run.
The responsible role signs the run record or records why the campaign did not proceed.

No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception.
**Operational action.** Run source freshness, duplicate-key, valid-range, and event-time checks; fail closed when a required check fails.
The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Run source freshness, duplicate-key, valid-range, and event-time checks; fail closed when a required check fails.
No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Run source freshness, duplicate-key, valid-range, and event-time checks; fail closed when a required check fails.

**Operational action.** Apply the approved preprocessing and centroid artifact, recording distance margin and assignment state for every input household.
The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Apply the approved preprocessing and centroid artifact, recording distance margin and assignment state for every input household.
No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Apply the approved preprocessing and centroid artifact, recording distance margin and assignment state for every input household.
**Operational action.** Exclude ambiguous identities, insufficient feature records, low-margin assignments, and all policy suppressions from automated promotional action.

The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Exclude ambiguous identities, insufficient feature records, low-margin assignments, and all policy suppressions from automated promotional action.
No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Exclude ambiguous identities, insufficient feature records, low-margin assignments, and all policy suppressions from automated promotional action.
**Operational action.** Join the approved segment-action matrix and remove actions that conflict with inventory, supplier, or current business restrictions.
The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Join the approved segment-action matrix and remove actions that conflict with inventory, supplier, or current business restrictions.

No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Join the approved segment-action matrix and remove actions that conflict with inventory, supplier, or current business restrictions.
**Operational action.** Deduplicate at household level before capacity allocation so related accounts cannot consume multiple contact slots.
The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Deduplicate at household level before capacity allocation so related accounts cannot consume multiple contact slots.
No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Deduplicate at household level before capacity allocation so related accounts cannot consume multiple contact slots.

**Operational action.** Rank only policy-qualified candidates using the approved, evidence-limited priority rule and apply the fixed 12,000-email cap.
The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Rank only policy-qualified candidates using the approved, evidence-limited priority rule and apply the fixed 12,000-email cap.
No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Rank only policy-qualified candidates using the approved, evidence-limited priority rule and apply the fixed 12,000-email cap.
**Operational action.** Randomize the final eligible cohort to the treatment action and its declared comparator before exporting any channel payload.

The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Randomize the final eligible cohort to the treatment action and its declared comparator before exporting any channel payload.
No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Randomize the final eligible cohort to the treatment action and its declared comparator before exporting any channel payload.
**Operational action.** Perform a final send-time consent and frequency recheck, recording every removal with a precise reason code.
The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Perform a final send-time consent and frequency recheck, recording every removal with a precise reason code.

No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Perform a final send-time consent and frequency recheck, recording every removal with a precise reason code.
**Operational action.** Send the minimum operational payload and retain the immutable decision log outside the delivery platform.
The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Send the minimum operational payload and retain the immutable decision log outside the delivery platform.
No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Send the minimum operational payload and retain the immutable decision log outside the delivery platform.

**Operational action.** Collect exposure and outcome events into a separate measurement view without feeding them back into the current assignment artifact.
The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Collect exposure and outcome events into a separate measurement view without feeding them back into the current assignment artifact.
No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Collect exposure and outcome events into a separate measurement view without feeding them back into the current assignment artifact.
**Operational action.** Review the planned evaluation after outcome maturity and publish a decision that distinguishes observed evidence from this illustrative specification.

The responsible role signs the run record or records why the campaign did not proceed. Specific decision context: **Operational action.** Review the planned evaluation after outcome maturity and publish a decision that distinguishes observed evidence from this illustrative specification.
No operator may replace an automated exclusion with a manual inclusion without an auditable approved exception. Specific decision context: **Operational action.** Review the planned evaluation after outcome maturity and publish a decision that distinguishes observed evidence from this illustrative specification.
**Run-completion action.** Reconcile the count of assigned records with the count entering eligibility so a missing join cannot masquerade as a marketing exclusion.
This operational evidence allows the evaluation team to reconstruct the campaign population without querying mutable live tables.

**Run-completion action.** Document the exact channel payload schema and prohibit raw feature columns from crossing the campaign-delivery boundary.
This operational evidence allows the evaluation team to reconstruct the campaign population without querying mutable live tables. Specific decision context: **Run-completion action.** Document the exact channel payload schema and prohibit raw feature columns from crossing the campaign-delivery boundary.
**Run-completion action.** Store a no-send reason for every capacity-excluded household so later analysis does not confuse lack of contact with a negative outcome.
This operational evidence allows the evaluation team to reconstruct the campaign population without querying mutable live tables. Specific decision context: **Run-completion action.** Store a no-send reason for every capacity-excluded household so later analysis does not confuse lack of contact with a negative outcome.

**Run-completion action.** Publish an end-of-run artifact manifest that names the snapshot, model, policy, randomization, and delivery files used in the campaign.
This operational evidence allows the evaluation team to reconstruct the campaign population without querying mutable live tables. Specific decision context: **Run-completion action.** Publish an end-of-run artifact manifest that names the snapshot, model, policy, randomization, and delivery files used in the campaign.

## 11. Outcome & Results

**Result boundary.** No real campaign was executed for this illustrative case, so there is no measured lift, revenue, complaint rate, or retention result to report.
The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion.
A reader should be able to tell whether a statement is a requirement, an example, or a measured fact.
**Result boundary.** The constructed arithmetic shows how an action cell could be evaluated, not that it should be deployed or that its expected value will materialize.

The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion. Specific decision context: **Result boundary.** The constructed arithmetic shows how an action cell could be evaluated, not that it should be deployed or that its expected value will materialize.
A reader should be able to tell whether a statement is a requirement, an example, or a measured fact. Specific decision context: **Result boundary.** The constructed arithmetic shows how an action cell could be evaluated, not that it should be deployed or that its expected value will materialize.
**Result boundary.** The design outcome is a reproducible policy path from source data to a constrained send file with explicit no-send routes.
The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion. Specific decision context: **Result boundary.** The design outcome is a reproducible policy path from source data to a constrained send file with explicit no-send routes.

A reader should be able to tell whether a statement is a requirement, an example, or a measured fact. Specific decision context: **Result boundary.** The design outcome is a reproducible policy path from source data to a constrained send file with explicit no-send routes.
**Result boundary.** A completed real evaluation would report arm sizes, delivery counts, numerator components, denominator definition, uncertainty interval, and each guardrail result.
The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion. Specific decision context: **Result boundary.** A completed real evaluation would report arm sizes, delivery counts, numerator components, denominator definition, uncertainty interval, and each guardrail result.
A reader should be able to tell whether a statement is a requirement, an example, or a measured fact. Specific decision context: **Result boundary.** A completed real evaluation would report arm sizes, delivery counts, numerator components, denominator definition, uncertainty interval, and each guardrail result.

**Result boundary.** A positive conversion difference without matured returns, discount funding, and message costs would be preliminary rather than a contribution result.
The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion. Specific decision context: **Result boundary.** A positive conversion difference without matured returns, discount funding, and message costs would be preliminary rather than a contribution result.
A reader should be able to tell whether a statement is a requirement, an example, or a measured fact. Specific decision context: **Result boundary.** A positive conversion difference without matured returns, discount funding, and message costs would be preliminary rather than a contribution result.
**Result boundary.** A stable clustering diagnostic without a randomized campaign comparator would remain descriptive evidence only.

The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion. Specific decision context: **Result boundary.** A stable clustering diagnostic without a randomized campaign comparator would remain descriptive evidence only.
A reader should be able to tell whether a statement is a requirement, an example, or a measured fact. Specific decision context: **Result boundary.** A stable clustering diagnostic without a randomized campaign comparator would remain descriptive evidence only.
**Result boundary.** A valid null result would still be valuable because it could retire an unnecessary segmentation policy before it increases contact pressure.
The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion. Specific decision context: **Result boundary.** A valid null result would still be valuable because it could retire an unnecessary segmentation policy before it increases contact pressure.

A reader should be able to tell whether a statement is a requirement, an example, or a measured fact. Specific decision context: **Result boundary.** A valid null result would still be valuable because it could retire an unnecessary segmentation policy before it increases contact pressure.
**Result boundary.** A guardrail breach would outweigh a favorable economic point estimate until the mechanism is understood and corrected.
The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion. Specific decision context: **Result boundary.** A guardrail breach would outweigh a favorable economic point estimate until the mechanism is understood and corrected.
A reader should be able to tell whether a statement is a requirement, an example, or a measured fact. Specific decision context: **Result boundary.** A guardrail breach would outweigh a favorable economic point estimate until the mechanism is understood and corrected.

**Result boundary.** Any future result must identify the exact artifact, population, date range, and action matrix used so it cannot be generalized across versions casually.
The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion. Specific decision context: **Result boundary.** Any future result must identify the exact artifact, population, date range, and action matrix used so it cannot be generalized across versions casually.
A reader should be able to tell whether a statement is a requirement, an example, or a measured fact. Specific decision context: **Result boundary.** Any future result must identify the exact artifact, population, date range, and action matrix used so it cannot be generalized across versions casually.
**Result boundary.** The present status is proposed analytical design with illustrative inputs, not an observed production outcome.

The report must preserve this evidence boundary when the manuscript is reused in planning or interview discussion. Specific decision context: **Result boundary.** The present status is proposed analytical design with illustrative inputs, not an observed production outcome.
A reader should be able to tell whether a statement is a requirement, an example, or a measured fact. Specific decision context: **Result boundary.** The present status is proposed analytical design with illustrative inputs, not an observed production outcome.
## 12. Failure Modes & What Went Wrong

**Failure mechanism — Late event leakage.** A completed order arrives after the cut but is backfilled into the current snapshot, making features look more current than the decision could have known.
**Discriminating check.** Compare event time, ingestion time, and snapshot manifest; reproduce the run from the manifest and confirm no later processing partition was read.

**Recovery.** Exclude the late event from that decision version, quantify affected records, and repair the extraction boundary before the next run.
**Failure mechanism — Identity merge collision.** Two accounts are later resolved as one household after both received different offers, creating duplicate contact and ambiguous attribution.
**Discriminating check.** Inspect household-link version, deterministic identifiers, and delivery logs for multiple offers under the merged identity.
**Recovery.** Suppress future conflicting actions, reconcile the decision log, and raise the identity-confidence threshold if collisions exceed tolerance.

**Failure mechanism — Consent race.** A customer withdraws email permission between campaign assembly and send, while the delivery list uses a stale preference extract.
**Discriminating check.** Compare send-time preference timestamp with delivery timestamp and count sends where withdrawal preceded handoff.
**Recovery.** Add a final preference lookup, remove affected records, and treat any bypass as a compliance incident rather than a model metric.
**Failure mechanism — Feature unit drift.** Spend changes from currency units to cents in one source feed, causing scaled distance and cluster shares to shift abruptly.

**Discriminating check.** Check unit metadata, distribution quantiles, and the ratio of current to reference medians before assignment.
**Recovery.** Fail the run on unit validation, correct the transform, and never compensate by moving centroids manually.
**Failure mechanism — Missingness masquerade.** A source outage produces missing purchase histories that are imputed as zero, making active customers appear inactive.
**Discriminating check.** Compare availability flags and source row counts with zero-value rates; true zero histories should not rise with feed failures.

**Recovery.** Route affected records to unassigned, restore the source, and rerun from a complete snapshot when possible.
**Failure mechanism — Centroid-version mismatch.** Current features are assigned with a scaler from one version and centroids from another, so distances no longer have the intended geometry.
**Discriminating check.** Verify schema hash, artifact checksum, feature ordering, and preprocessing version in the assignment log.
**Recovery.** Block mismatched artifacts at load time and roll back to the last coherent approved bundle.

**Failure mechanism — Seasonal composition shift.** A holiday period increases gift purchases, making normally infrequent households resemble occasion-led customers without a durable preference change.
**Discriminating check.** Compare assignment share and feature distributions against seasonal reference weeks rather than only against the immediately prior week.
**Recovery.** Use seasonal monitoring, defer action changes, or fit a separately approved seasonal model if the pattern recurs.
**Failure mechanism — Campaign-history contamination.** Past campaign exposures enter a feature table and cause clusters to encode prior targeting rather than underlying purchase behaviour.

**Discriminating check.** Trace feature lineage for campaign IDs, exposure timestamps, and post-campaign engagement columns.
**Recovery.** Remove contaminated features, refit from a clean historical snapshot, and document the invalidated versions.
**Failure mechanism — Capacity-induced selection bias.** Only top-ranked households are contacted, then their response rate is presented as proof that the segment itself is responsive.
**Discriminating check.** Check whether the analysis conditions on rank, send status, or a post-policy variable when comparing cluster outcomes.

**Recovery.** Use randomized comparison within the eligible cohort and report the policy's effect at the stated capacity.
**Failure mechanism — Discount cannibalization.** A discount accelerates purchases that would have happened at full price, so gross order rate rises while incremental contribution falls.
**Discriminating check.** Compare treatment and control net margin, timing of repeat purchases, redemption cost, and stock substitution rather than conversions alone.
**Recovery.** Constrain the offer, extend the observation window, and retain contribution as the primary economic metric.

**Failure mechanism — Stockout substitution.** Promoted products run out, pushing customers to lower-margin substitutes or creating complaint-prone broken links.
**Discriminating check.** Join inventory snapshots and substitution orders to exposure time, then compare stockout prevalence by arm.
**Recovery.** Suppress depleted offers, apply real-time inventory gates, and rerun allocation with feasible alternatives.
**Failure mechanism — Unsubscribe concentration.** A segment receives a more frequent or more aggressive message and produces a localized opt-out increase hidden by an overall average.

**Discriminating check.** Break down delivered-message unsubscribe rate by action cell and recency band with treatment-control comparison.
**Recovery.** Pause the affected action, lower frequency or revise creative, and require a new approval before resuming.
**Failure mechanism — Proxy discrimination.** A behavioral feature or offer policy creates disparate access or burden for a lawful audit group even though protected attributes were not fitted.
**Discriminating check.** Where lawful audit data exist, compare eligibility, assignment uncertainty, selection, offer value, and guardrail rates across groups.

**Recovery.** Escalate material disparity for review, test feature and policy alternatives, and do not use absence of intent as a dismissal.
**Failure mechanism — Tiny unstable cluster.** A candidate solution isolates a small group that appears distinctive in one sample but changes membership under resampling.
**Discriminating check.** Measure matched-cluster agreement across bootstrap samples and inspect cell size after eligibility exclusions.
**Recovery.** Merge or retire the group; operational labels need stability and an actionable denominator, not novelty.

**Failure mechanism — Silent row loss.** An inner join to preferences removes records without a consent row, shrinking the denominator before exclusion reasons are logged.
**Discriminating check.** Reconcile raw population, joined population, and exclusion-reason counts; inspect null-key and unmatched-key rates.
**Recovery.** Use explicit left joins and reason codes, then decide whether unmatched records are safely suppressible or indicate data breakage.
**Failure mechanism — Control contamination.** A control household receives a similar campaign through another channel or an overlapping business campaign during the attribution window.

**Discriminating check.** Check cross-campaign exposure logs and channel message history for both randomized arms.
**Recovery.** Exclude or model the contamination only under the protocol, improve contact governance, and rerun the test if the estimand is no longer credible.
**Failure mechanism — Return-window truncation.** The campaign readout closes before elevated return behaviour is known, overstating net contribution.
**Discriminating check.** Compare provisional and matured refund rates by treatment arm and verify the stated return maturation window.

**Recovery.** Mark early results provisional, delay the decision, or estimate with a prespecified maturity model that is later audited.
**Failure mechanism — Manual suppression opacity.** An operator removes customers from a send list without a reason, making both capacity allocation and experiment denominators irreproducible.
**Discriminating check.** Compare assignment output, policy-qualified count, and handoff count; require reason codes for every variance.
**Recovery.** Disallow unlogged edits, restore the source-of-truth workflow, and invalidate affected evaluation slices if randomization changed.

**Failure mechanism — Metric denominator switch.** A dashboard changes from eligible households to delivered messages, making a rate appear to improve without the policy improving.
**Discriminating check.** Read the metric contract and recompute from row-level exposure data using the original denominator.
**Recovery.** Version dashboard definitions, annotate the switch, and use the prespecified estimand for the decision.
**Failure mechanism — Offer-rule drift.** Merchandising changes discount eligibility after assignment, so the delivered treatment differs from the randomized policy cell.

**Discriminating check.** Compare action matrix version at randomization, handoff, and delivery; list records whose action code changed.
**Recovery.** Freeze offer rules for the experiment or classify post-randomization changes as deviations in the analysis.
## 13. Transferable Lesson

**Lesson.** Segmentation is not an action policy: it becomes useful only when a permitted action, capacity, comparator, and no-action route are specified.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation.

**Lesson.** Time safety is a product requirement, not a modelling nicety, because one post-cut event can make an apparently reproducible campaign impossible to execute honestly.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation. Specific decision context: **Lesson.** Time safety is a product requirement, not a modelling nicety, because one post-cut event can make an apparently reproducible campaign impossible to execute honestly.
**Lesson.** Identity resolution belongs in the decision contract because duplicate or merged records change both customer experience and experiment denominators.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation. Specific decision context: **Lesson.** Identity resolution belongs in the decision contract because duplicate or merged records change both customer experience and experiment denominators.

**Lesson.** Internal clustering scores are diagnostics for geometry; they do not demonstrate incremental customer or business value.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation. Specific decision context: **Lesson.** Internal clustering scores are diagnostics for geometry; they do not demonstrate incremental customer or business value.
**Lesson.** The smallest stable, explainable, action-distinguishing segmentation is generally safer than a high-count taxonomy that cannot be governed.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation. Specific decision context: **Lesson.** The smallest stable, explainable, action-distinguishing segmentation is generally safer than a high-count taxonomy that cannot be governed.

**Lesson.** A distance margin is useful operational uncertainty information, but it is not a probability of response or an estimate of causal effect.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation. Specific decision context: **Lesson.** A distance margin is useful operational uncertainty information, but it is not a probability of response or an estimate of causal effect.
**Lesson.** Consent and frequency controls must run after every analytic step and again before delivery, because eligibility can change while a campaign is being assembled.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation. Specific decision context: **Lesson.** Consent and frequency controls must run after every analytic step and again before delivery, because eligibility can change while a campaign is being assembled.

**Lesson.** Randomize the policy-qualified population to learn whether the action helps; do not use the cluster label as proof that an action worked.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation. Specific decision context: **Lesson.** Randomize the policy-qualified population to learn whether the action helps; do not use the cluster label as proof that an action worked.
**Lesson.** Guardrails need hard denominators and named owners so customer harm cannot disappear behind a favorable average contribution figure.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation. Specific decision context: **Lesson.** Guardrails need hard denominators and named owners so customer harm cannot disappear behind a favorable average contribution figure.

**Lesson.** Versioned artifacts and immutable decision logs make recovery possible when the cause is a data pipeline, policy, or model artifact rather than customer behaviour.
This principle transfers to retention, recommendation, fraud outreach, and service prioritization whenever an unsupervised description becomes a customer-facing allocation. Specific decision context: **Lesson.** Versioned artifacts and immutable decision logs make recovery possible when the cause is a data pipeline, policy, or model artifact rather than customer behaviour.
## 14. Connections & Cross-References

**Connection.** Clustering and embedding methods provide the distance and representation concepts used in the assignment stage.
The connection is operational: each discipline provides a control that prevents an attractive clustering result from becoming an unsafe campaign decision.

**Connection.** Segmentation evaluation supplies internal validity, stability, and actionability checks that precede any campaign outcome claim.
The connection is operational: each discipline provides a control that prevents an attractive clustering result from becoming an unsafe campaign decision. Specific decision context: **Connection.** Segmentation evaluation supplies internal validity, stability, and actionability checks that precede any campaign outcome claim.
**Connection.** Human review loops are relevant when policy owners examine unassigned records, fairness findings, or proposed action matrices.
The connection is operational: each discipline provides a control that prevents an attractive clustering result from becoming an unsafe campaign decision. Specific decision context: **Connection.** Human review loops are relevant when policy owners examine unassigned records, fairness findings, or proposed action matrices.

**Connection.** Experiment design connects the descriptive model to a causal estimate of campaign policy effect.
The connection is operational: each discipline provides a control that prevents an attractive clustering result from becoming an unsafe campaign decision. Specific decision context: **Connection.** Experiment design connects the descriptive model to a causal estimate of campaign policy effect.
**Connection.** Data engineering practices provide event-time cuts, lineage, schema control, and reconciliation needed to keep features decision-safe.
The connection is operational: each discipline provides a control that prevents an attractive clustering result from becoming an unsafe campaign decision. Specific decision context: **Connection.** Data engineering practices provide event-time cuts, lineage, schema control, and reconciliation needed to keep features decision-safe.

**Connection.** Privacy and responsible-use principles constrain identity linkage, consent handling, audit analysis, and the communication of segment labels.
The connection is operational: each discipline provides a control that prevents an attractive clustering result from becoming an unsafe campaign decision. Specific decision context: **Connection.** Privacy and responsible-use principles constrain identity linkage, consent handling, audit analysis, and the communication of segment labels.
### Interview pack

**Question 1.** Why is a customer segment not sufficient to authorize a campaign?
**Answer.** Because a segment is descriptive. The campaign still needs consent, identity, frequency, inventory, action-policy, and capacity gates.

**Reasoning.** A nearest-centroid label says nothing about permission or incremental benefit, so downstream rules remain mandatory.
**Follow-up.** Name two final checks that must be evaluated at send time.
**Red flag.** Saying that a high silhouette score alone proves a message should be sent.
**Scoring.** Full credit identifies the descriptive-versus-policy distinction and at least two operational gates.

**Question 2.** Calculate a standardized spend value when spend is 180, reference mean is 120, and reference standard deviation is 40 currency units.
**Answer.** The value is (180 minus 120) divided by 40, which equals 1.50 unitless.
**Reasoning.** The numerator is 60 currency units and division by 40 currency units removes the unit.
**Follow-up.** What happens if the reference standard deviation is zero?

**Red flag.** Reporting 60 as the z-score or recomputing the reference statistics from the campaign audience.
**Scoring.** Full credit shows intermediate arithmetic and states the zero-variance boundary.
**Question 3.** A household has nearest distance 0.707 and second-nearest distance 0.860. Is it eligible under a margin threshold of 0.120?
**Answer.** Its margin is 0.860 minus 0.707, or 0.153, so it clears the stated threshold.

**Reasoning.** The comparison is valid only because both distances use the same frozen centroids and preprocessing.
**Follow-up.** What is the action if the margin is 0.090?
**Red flag.** Calling the margin a probability that the customer will buy.
**Scoring.** Full credit computes 0.153 and routes the smaller-margin example to unassigned.

**Question 4.** Why can a three-cluster solution be preferable to a six-cluster solution with a slightly better internal score?
**Answer.** Three may be preferable when it is stable, explainable, and supports distinct permissible actions, whereas six may create tiny or unstable groups.
**Reasoning.** The selection objective includes actionability and governance, not geometric fit alone.
**Follow-up.** Which resampling diagnostic would you inspect?

**Red flag.** Choosing the largest k because more labels appear more personalized.
**Scoring.** Full credit names a stability or matched-agreement check and a policy consequence.
**Question 5.** How would you detect post-cut leakage in a weekly feature snapshot?
**Answer.** Compare each feature source event time and processing time to the declared cutoff and reproduce from the snapshot manifest.

**Reasoning.** A transaction processed later may be legitimate historical data, but it cannot revise a decision already made without a documented rerun.
**Follow-up.** Give one event that commonly leaks into campaign features.
**Red flag.** Checking only the current table without a dated manifest.
**Scoring.** Full credit distinguishes event time from processing time and specifies a reconstruction check.

**Question 6.** A treatment arm has 600 eligible households with 54 conversions, and control has 600 with 42. What is the raw conversion-rate difference?
**Answer.** Treatment rate is 54 divided by 600 equals 0.090; control is 42 divided by 600 equals 0.070; the difference is 0.020 or 2.0 percentage points.
**Reasoning.** This is a descriptive arm difference before uncertainty, contribution, returns, and guardrails are assessed.
**Follow-up.** Why should the denominator be eligible households for intent-to-treat?

**Red flag.** Using 54 divided by 42 or replacing eligible households with delivered messages silently.
**Scoring.** Full credit shows both rates and labels the result as a percentage-point difference.
**Question 7.** What failure can occur if a consent withdrawal happens after file assembly?
**Answer.** A stale delivery list can send a message after permission was withdrawn, creating a compliance and customer-trust failure.

**Reasoning.** The decisive comparison is withdrawal effective time against send handoff or delivery time.
**Follow-up.** What system control prevents this race?
**Red flag.** Treating the event as ordinary nonresponse or relying on the weekly snapshot alone.
**Scoring.** Full credit requires a final preference recheck and an auditable suppression reason.

**Question 8.** How would you test whether a small cluster is stable enough for an action policy?
**Answer.** Refit or resample the training data, match clusters across fits, measure membership agreement, and inspect post-exclusion cell size.
**Reasoning.** A small group can look distinctive in one sample yet move substantially when records or seeds change.
**Follow-up.** What policy decision follows poor agreement?

**Red flag.** Using only one random seed or declaring stability from a cluster name.
**Scoring.** Full credit includes both a resampling method and a merge or retire response.
**Question 9.** Why can conversion rate increase while incremental contribution decreases?
**Answer.** Discounts can pull forward full-price purchases, fund low-margin orders, or increase returns, so more orders need not create more net value.

**Reasoning.** Contribution therefore subtracts discount funding, message cost, and matured return impact rather than treating every conversion equally.
**Follow-up.** Which comparator makes the estimate causal?
**Red flag.** Equating gross revenue with incremental contribution.
**Scoring.** Full credit identifies cannibalization or margin mechanisms and names randomized control.

**Question 10.** A source outage makes completed purchase history missing for many records. Why is imputing zero dangerous?
**Answer.** Zero means known absence of purchases, while missing can mean the source failed; treating them alike can misassign active customers.
**Reasoning.** Availability flags and source row counts discriminate an outage from true zero behaviour.
**Follow-up.** What is the safe immediate action?

**Red flag.** Silently filling zeros because the clustering code requires a number.
**Scoring.** Full credit routes affected records to unassigned or fails closed pending a valid snapshot.
**Question 11.** What fairness review is possible if protected attributes are not used in model fitting?
**Answer.** Where lawful audit attributes are available in a segregated review, compare eligibility, uncertainty, selection, offer value, and harms across groups.

**Reasoning.** Excluding protected attributes from fitting does not prove that behavioural features and policy rules have no disparate impact.
**Follow-up.** Name one outcome guardrail to compare across groups.
**Red flag.** Claiming fairness is guaranteed because the model does not contain a protected column.
**Scoring.** Full credit separates fitting restrictions from outcome auditing and names a disparity check.

**Question 12.** How do you distinguish a true seasonal shift from a broken feature feed when segment shares change sharply?
**Answer.** Compare source freshness, missingness, unit checks, and event lag first; then compare valid features against analogous seasonal weeks and later holdouts.
**Reasoning.** A broken feed often changes availability or distributions abruptly, while a seasonal shift should appear in complete data and recur in comparable periods.
**Follow-up.** What should happen before changing centroids?

**Red flag.** Immediately retraining without verifying the pipeline.
**Scoring.** Full credit gives competing causes, discriminating checks, and a pause or validated rerun decision.
## 15. Quick Reference Card

**Decision unit.** One identity-resolved customer-household at a declared campaign cut.
Use this line as a review prompt before approving a segmentation-driven campaign run.

**Primary purpose.** Allocate a constrained, permitted campaign policy rather than assign a permanent customer identity.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Primary purpose.** Allocate a constrained, permitted campaign policy rather than assign a permanent customer identity.
**Feature cut.** Event time at or before Sunday 23:59:59 UTC; retain processing time only for quality diagnosis.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Feature cut.** Event time at or before Sunday 23:59:59 UTC; retain processing time only for quality diagnosis.

**Forbidden future data.** Post-cut purchases, returns, consent changes, campaign exposures, and outcome events.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Forbidden future data.** Post-cut purchases, returns, consent changes, campaign exposures, and outcome events.
**Identity rule.** Use durable household key; suppress ambiguous or low-confidence links from automated action.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Identity rule.** Use durable household key; suppress ambiguous or low-confidence links from automated action.

**Consent rule.** Active permission is required at assembly and rechecked immediately before delivery.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Consent rule.** Active permission is required at assembly and rechecked immediately before delivery.
**Frequency rule.** Respect the stated seven-day and thirty-day caps before capacity allocation.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Frequency rule.** Respect the stated seven-day and thirty-day caps before capacity allocation.

**Fit candidates.** Compare k from two through six using internal diagnostics, stability, actionability, and governance.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Fit candidates.** Compare k from two through six using internal diagnostics, stability, actionability, and governance.
**Scaling.** Use frozen training means and standard deviations; never recompute them from the current send file.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Scaling.** Use frozen training means and standard deviations; never recompute them from the current send file.

**Assignment.** Nearest frozen centroid plus stored nearest and second-nearest distances.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Assignment.** Nearest frozen centroid plus stored nearest and second-nearest distances.
**Uncertainty route.** Low margin, missing required features, or invalid artifact means unassigned, not forced assignment.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Uncertainty route.** Low margin, missing required features, or invalid artifact means unassigned, not forced assignment.

**Action matrix.** Keep segment-to-offer policy versioned separately from the clustering artifact.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Action matrix.** Keep segment-to-offer policy versioned separately from the clustering artifact.
**Capacity.** Rank only policy-qualified households and take no more than the channel's approved limit.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Capacity.** Rank only policy-qualified households and take no more than the channel's approved limit.

**Comparator.** Randomized no-message or broad-message control with the same pre-randomization eligibility.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Comparator.** Randomized no-message or broad-message control with the same pre-randomization eligibility.
**Primary metric.** Incremental net contribution per eligible randomized household over the declared attribution window.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Primary metric.** Incremental net contribution per eligible randomized household over the declared attribution window.

**Economic components.** Contribution less discount funding, message cost, and matured returns; label all constructed values ILLUSTRATIVE.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Economic components.** Contribution less discount funding, message cost, and matured returns; label all constructed values ILLUSTRATIVE.
**Guardrails.** Unsubscribe, complaint, frequency-cap breach, duplicate delivery, stockout, and lawful fairness-audit checks.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Guardrails.** Unsubscribe, complaint, frequency-cap breach, duplicate delivery, stockout, and lawful fairness-audit checks.

**Stability.** Resample, match clusters, inspect agreement, size, and whether action cells remain interpretable.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Stability.** Resample, match clusters, inspect agreement, size, and whether action cells remain interpretable.
**Key log fields.** Feature cut, identity version, artifact version, assignment, margin, policy rule, action, and exclusion reason.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Key log fields.** Feature cut, identity version, artifact version, assignment, margin, policy rule, action, and exclusion reason.

**Hard stop.** Pause on invalid data, consent breach, artifact mismatch, or mandatory guardrail failure.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Hard stop.** Pause on invalid data, consent breach, artifact mismatch, or mandatory guardrail failure.
**Fallback.** Use an approved broad or no-send policy; do not use stale segmentation outputs.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Fallback.** Use an approved broad or no-send policy; do not use stale segmentation outputs.

**Evidence boundary.** This case is illustrative and proposes checks; it reports no measured company campaign result.
Use this line as a review prompt before approving a segmentation-driven campaign run. Specific decision context: **Evidence boundary.** This case is illustrative and proposes checks; it reports no measured company campaign result.

