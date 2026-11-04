# Productivity Metrics

> Module 13 — Warehousing, Fulfillment, and Labor Analytics
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 13, topics 01 and 02
> Version 2026.5

*Units, lines, orders per hour and the denominators that make them comparable.*

## 1. Why this topic exists

A warehouse manager must decide how many people to schedule, where to investigate delay,
and whether a process change actually produced more customer-ready work per hour.
A single throughput number answers none of those questions unless its numerator and denominator
refer to the same process, population, and interval.

An associate may pick 200 eaches in one hour for one order line, or 60 eaches across 60 lines.
The first result has more units per hour; the second has more lines per hour.
Neither alone establishes which associate performed better: travel, slotting, packaging,
order composition, equipment, and upstream availability differ.
A shift may also appear more productive when its unrecorded waiting time disappears from its denominator.

The three rates in this topic measure different outputs:
units per labor hour, order lines per labor hour, and completed orders per labor hour.
A **labor hour** is one person's hour, not one elapsed clock hour for an entire team.
Two people working simultaneously for one elapsed hour contribute two labor hours.
Machine hours, scheduled hours, paid hours, and active task hours are different denominators.

For staffing, use a rate whose labor scope matches the staff to be planned.
For improvement, compare equivalent work under comparable conditions and include quality and safety.
For finance, include the support labor and costs required by the costing question.
Never use a faster-looking but narrower rate as proof that total warehouse cost has fallen.

This lesson covers measurement and interpretation, not the construction of engineered labor standards.
See the [module survey](../../../course/13_warehousing_fulfillment_labor.md) for labor standards context.
All operating data and example dates below are illustrative unless explicitly attributed.
The example is not evidence about an actual site, workforce, company, or target.

## 2. Learning outcomes

By the end of this topic you can:

1. Define units, lines, orders, and the completion event for each rate.
2. Reconcile labor hours to a timekeeping source and choose a denominator for a decision.
3. Compute the three rates from matching numerator and denominator populations.
4. Aggregate rates as total output divided by total hours rather than an unweighted average.
5. Adjust an observed comparison for work mix and explain what the adjustment cannot prove.
6. Construct reproducible event and labor data, with a runnable calculation.
7. Assess rework, safety, privacy, and legal considerations before deploying a measure.
8. Design a dashboard and critique a performance claim using an explicit metric contract.

Exercises 1–8 map to these outcomes in order; section 13 supplies calculations or rubrics.

## 3. The core mental model

```text
Customer demand -> release -> pick -> pack -> ship / order complete
                         |        |       |
                     task events  |       +-> completed orders
                         |        +----------> packed lines and units
                         +-------------------> picked lines and units

Timekeeping -> person-time intervals -> classify -> allocate -> reconcile
                                                |
Valid output / matching labor hours ------------+-> rate and coverage
                                 +-> mix, quality, safety, uncertainty
                                 +-> staffing or process decision
```

The top line describes physical flow; a transaction is evidence of a state transition,
not automatically a unique physical action.
The bottom line describes people paid or assigned during the same scope.
The numerator and denominator meet only after both are classified and reconciled.

Choose the output event first. A pick confirmation answers a pick question;
a shipped order answers a customer-completion question.
A partially shipped order is not a completed order under an all-lines-shipped rule.
If the business accepts partial shipment as completion, write that rule explicitly.

For a period, a rate is output quantity divided by labor hours.
It is not the mean of individual rates unless the individuals have equal exposure hours.
A rate describes work delivered under observed conditions, not innate worker ability.
Mix, process, availability, support, and data recording all shape the result.

The denominator is a boundary around costs and activities.
Direct pick hours exclude pack time; end-to-end hours include both if defined that way.
Comparing those rates without showing the boundary can reverse the apparent conclusion.
Track volume and hours alongside each rate to expose small samples and missing coverage.

### Three layers of comparison

1. **Arithmetic:** Are output quantity and labor hours correctly calculated?
2. **Population:** Are the work, time, location, event, and quality boundaries aligned?
3. **Interpretation:** Are mix and operational context sufficiently similar for the decision?

Passing arithmetic does not establish causal impact or fair individual evaluation.
A comparison is valid only for a stated use, not for every possible use.

## 4. Definitions and vocabulary

The following dictionary is a starting contract. A site must adopt one precise version per metric.

| Term | Definition and grain | Common misuse or unit |
|---|---|---|
| Unit | Count of physical base items confirmed at the selected process event | Confusing eaches, cases, and pallets; eaches |
| Order line | One distinct order ID and line ID under the stated completion rule | Counting task fragments as lines; lines |
| Order | One distinct customer order satisfying the chosen completion rule | Counting cartons or shipments as orders; orders |
| Pick task | Assigned or executed work instruction, possibly covering a fragment of a line | Treating task count as order line count; tasks |
| Confirmation | Recorded claim that a task or quantity was completed | Assuming every record is unique and irreversible; events |
| Reversal | Event that negates an earlier event or quantity | Dropping the reversal but keeping the original; signed units |
| Rework | Additional work to correct or repeat an earlier process | Counting correction as new saleable output; labor hours |
| Labor hour | Sum of elapsed work time across people in scope | Substituting elapsed shift hours; person-hours |
| Paid hour | Time compensated according to the payroll record | Assuming every paid hour is active task time; person-hours |
| Direct hour | Time classified to the defined operating process | Assuming all sites use the same activity codes; person-hours |
| Active task hour | Time assigned to recorded execution intervals | Treating it as total paid capacity; person-hours |
| Indirect hour | Support time not classified as direct under a local policy | Assuming indirect means unnecessary; person-hours |
| Scheduled hour | Planned staffed time before actual attendance and changes | Reporting it as actual paid time; person-hours |
| Utilization | Ratio of a specified productive-time subset to a specified available-time base | Claiming a universally defined utilization; percent |
| Throughput | Completed output over an interval | Calling a count a rate without its interval; units per hour |
| Productivity | Output divided by a defined input | Calling machine throughput labor productivity; output per labor hour |
| Weighted rate | Sum of output divided by sum of hours | Averaging rates without hour weights; output per labor hour |
| Mix | Distribution of work attributes such as line size and travel zone | Treating the distribution as a performance result; share |
| Stratum | Comparable subset defined before examining outcomes | Creating tiny groups after seeing results; category |
| Standardization | Applying common weights to stratum-specific rates | Claiming standardization removes all confounding; weighted rate |
| Coverage | Fraction of relevant records or hours successfully included | Treating unlinked records as zero labor; percent |
| Quality yield | Accepted output divided by output submitted for acceptance | Treating gross scans as good output; percent |
| Period | Specified time interval, usually half-open [start, end) | Mixing local and UTC boundaries; timestamps |
| Cohort | Set of orders defined by a common entry or release event | Assuming period completions are the same population; orders |
| Work in process | Work started but not yet completed at a boundary | Ignoring inventory of incomplete orders; orders or lines |
| Metric contract | Versioned specification of the full measurement definition | Relying on a dashboard label alone; specification |

## 5. The mechanics, step by step

The sequence below is a measurement design procedure, not an instruction to set individual quotas.
For each step, preserve the input, transformation, output, and exception decisions.
### 5.1 State the decision

Input: the staffing, improvement, costing, or service decision and its owner.
Ask which controllable process boundary the decision concerns.
A pick supervisor needs pick output and pick labor; a fulfillment director may need all warehouse labor.
Output: one sentence naming the decision, decision date, and acceptable measurement delay.
Branch: if the decision is individual discipline, stop and involve human resources, legal, and worker representatives.
A team-level descriptive rate does not establish an individual performance cause.
### 5.2 Identify the population

Input: site, operation, channel, shift, time interval, and included employment categories.
Specify whether agency, temporary, maintenance, supervisory, and automation support workers are included.
A comparison of two sites requires the same inclusion policy or a documented bridge.
Output: an inclusion list and a documented owner for each category.
Branch: if a relevant labor category cannot be captured, report incomplete coverage and avoid a total-site claim.
### 5.3 Choose the numerator

Input: the decision and the relevant state transition.
Choose picked eaches, fully picked order lines, or fully shipped orders; do not call these interchangeable.
State whether completed work is accepted, gross, or net of reversals.
Output: event type, output entity key, quantity unit, and completion rule.
Branch: when only fragments are observable, report fragments rather than silently labeling them completed lines.
### 5.4 Establish unit conversion

Input: product master and transaction units of measure at the event date.
Convert case or pallet quantities to eaches using the valid item-specific factor.
A case of six items and a case of twelve items are not equal quantities of eaches.
Output: converted quantity plus the original unit and conversion version.
Branch: if conversion is missing or conflicts with the transaction, quarantine it and report affected volume.
### 5.5 Define the line lifecycle

Input: order line key, original required quantity, amendments, cancellations, and all valid pick events.
Define whether a line is complete when net picked quantity meets the final uncancelled requirement.
Repeated confirmations can be needed for one line; a partial pick is not a completed line.
Output: at most one completion per eligible order line and a completion timestamp.
Branch: if a previously complete line is reopened, apply a versioned correction policy and restate affected periods.
### 5.6 Define the order lifecycle

Input: order identifier, eligible uncancelled lines, and shipment confirmations.
Choose complete when every eligible line is shipped, not when the first carton leaves.
Keep split shipments attached to the same order and retain the final completion timestamp.
Output: at most one completed order per order lifecycle.
Branch: if all lines are cancelled, exclude the order from completed shipped orders and disclose cancellations.
### 5.7 Bound the clock

Input: site timezone, period start, period end, and event timestamps.
Use [start, end) so an event at the boundary appears in one period only.
Store timestamps with timezone or convert consistently and account for daylight-saving transitions.
Output: a period calendar used by both the event and labor pipelines.
Branch: if shifts cross midnight, split intervals at the reporting boundary rather than assigning all time to one day.
### 5.8 Deduplicate and net events

Input: immutable event ID, business key, event type, quantity, and reversal reference.
Remove repeated deliveries of the same event ID; preserve legitimate separate task fragments.
Apply signed reversal or correction events under a documented effective-time policy.
Output: a traceable net event ledger and a log of rejected duplicates.
Branch: if corrections arrive after publication, restate the affected period or publish a separate adjustment.
### 5.9 Collect time intervals

Input: clock punches, activity assignments, breaks, absences, and employment records.
Convert person-level intervals into person-hours, not just hours on the shift clock.
Distinguish paid from unpaid breaks according to the authoritative timekeeping source.
Output: nonoverlapping paid intervals by person, site, period, and activity.
Branch: missing punches or conflicting intervals require reconciliation, not automatic zero hours.
### 5.10 Classify labor

Input: local activity codes and the agreed metric boundary.
Map direct pick, pack, replenish, supervision, training, maintenance, and other support.
Keep a separate unclassified bucket and preserve the mapping version.
Output: hours by category that sum back to the paid-hour ledger.
Branch: if the mapping changes, bridge prior periods or mark a break in the series.
### 5.11 Handle shared labor

Input: time intervals for people supporting multiple processes and an approved allocation driver.
Prefer observed activity intervals where available; otherwise use an explicit shared-hours allocation.
For example, allocate shared supervision by direct hours only if that is reasonable for the decision.
Output: allocated hours plus the raw shared bucket and allocation method.
Branch: if allocation is arbitrary, publish both direct and inclusive rates and test sensitivity.
### 5.12 Align attribution

Input: output event ledger and labor ledger at the same site, operation, and period.
Use shift or site as the primary attribution when tasks and individuals cannot be reliably matched.
Orders may be completed by one shift using work done on another; that is a period-flow rate.
Output: numerator and denominator with consistent boundaries and explicit attribution rules.
Branch: for a cohort cost study, link all labor used by the cohort rather than just completion-period labor.
### 5.13 Calculate each rate

Input: eligible net output and positive included person-hours.
Units per hour = accepted eaches / included person-hours.
Lines per hour = completed eligible lines / included person-hours.
Orders per hour = completed eligible orders / included person-hours.
Output: three rates with raw numerators, hours, period, scope, and units.
Branch: a zero denominator yields not defined, not zero or infinity.
### 5.14 Aggregate correctly

Input: output and hours for the included shifts or sites.
Add eligible outputs and add corresponding hours before dividing.
This is the hours-weighted mean of the underlying rates when each denominator is positive.
Output: pooled rate and the component counts needed to reproduce it.
Branch: do not combine incompatible process scopes merely because both columns say units per hour.
### 5.15 Diagnose work mix

Input: comparable work attributes and counts for both groups.
Consider zone, handling type, units per line, lines per order, travel, channel, and service.
Compare rates inside stable strata before interpreting the pooled difference.
Output: stratum-specific rates and the share of exposure hours in each stratum.
Branch: if the groups lack common support, do not extrapolate an adjusted result to unseen work.
### 5.16 Standardize cautiously

Input: valid stratum rates for each group and common weights that sum to one.
Compute each group standardized rate as the sum of weight times stratum rate.
Use a fixed, decision-relevant reference distribution, such as a pooled hour distribution.
Output: adjusted rates, weights, unadjusted rates, and stratum sample sizes.
Branch: if workload changes within strata or relevant confounders are missing, label residual differences descriptive.
### 5.17 Pair output with quality

Input: confirmed errors, returns attributable to the process, damage, and correction labor.
Count good output under a fixed acceptance rule, while retaining the gross quantity and lag.
Show quality metrics beside gross rate; include rework hours in the relevant full-scope denominator.
Output: gross and accepted output, error rate, rework hours, and maturity of quality observations.
Branch: if defects are observed after the period closes, mark the recent quality figure provisional.
### 5.18 Pair output with safety

Input: safety incidents, reported hazards, and ergonomic observations at an appropriate aggregate grain.
Do not treat absence of reported injuries as proof that a speed increase is safe.
Measure safety independently of the productivity goal and protect hazard reporting.
Output: a decision record with safety review and stop conditions.
Branch: a suspected unsafe practice stops the change pending qualified review.
### 5.19 Validate and reconcile

Input: published source totals, event ledger totals, and timekeeping totals.
Reconcile included, excluded, duplicate, reversed, missing, and late records.
Trace samples across system boundaries and investigate large unexplained differences.
Output: a reconciliation table, unresolved issue list, and coverage ratios.
Branch: below an agreed coverage threshold, withhold comparisons rather than filling missing labor with zero.
### 5.20 Publish with context

Input: approved metric contract, computed results, coverage, mix, quality, and source versions.
Publish the counts and denominators with each rate, plus period and refresh time.
Use access controls and avoid ranking workers by a metric they cannot meaningfully control.
Output: a reproducible dashboard and named owner for exceptions and changes.
Branch: a changed definition starts a new version; never splice incomparable series without a bridge.
### 5.21 Use rates in staffing

Input: expected workload, planned operating conditions, a comparable historical rate, and planned paid hours per person.
Forecast workload and translate it into hours using a rate that includes the labor to be scheduled.
Add only capacity or contingencies not already embodied in that inclusive rate.
Output: required person-hours and a rounded-up staffing plan, with sensitivity to mix and availability.
Branch: where workload or capacity is uncertain, show a range rather than claiming the estimate is a commitment.
### 5.22 Evaluate change

Input: intervention dates, control groups where feasible, mix, quality, and comparable time windows.
Predefine the outcome and check for coincident layout, staffing, or demand changes.
Compare within strata and, if possible, with an unaffected comparison area.
Output: descriptive differences and a candid assessment of alternative explanations.
Branch: without a credible comparison design, describe association and avoid attributing the entire difference to the intervention.
## 6. Worked example

### 6.1 Decision and boundary

A fictional distribution center must plan a 40-hour paid week of outbound fulfillment.
The planning team compares two illustrative shifts over the same Monday–Friday period in May 2026.
Both shifts handle the same two broad work types: easy and difficult.
This is a synthetic teaching case, not a company benchmark.

The unit is one accepted physical each confirmed at pick.
A completed line is an uncancelled order line fully picked during the period.
A completed order is an order whose last eligible line ships in the period.
Those counts are distinct event populations; no assertion is made that all belong to the same orders.
All denominators below are paid fulfillment person-hours including pick, pack, and support.

### 6.2 Shift totals

| Shift | Accepted picked eaches | Completed lines | Completed orders | Paid fulfillment hours |
|---|---:|---:|---:|---:|
| A | 12,600 eaches | 2,520 lines | 840 orders | 120 person-hours |
| B | 11,400 eaches | 2,280 lines | 760 orders | 120 person-hours |
| Both | 24,000 eaches | 4,800 lines | 1,600 orders | 240 person-hours |

For A, 12,600 / 120 = 105 eaches per paid fulfillment person-hour.
For B, 11,400 / 120 = 95 eaches per paid fulfillment person-hour.
A appears 10 eaches per person-hour faster, or 10 / 95 = 10.53% relative to B.
That is a descriptive difference, not a causal effect or a worker ranking.

For A, 2,520 / 120 = 21 completed lines per person-hour.
For B, 2,280 / 120 = 19 completed lines per person-hour.
For A, 840 / 120 = 7 completed orders per person-hour.
For B, 760 / 120 = 6.333 completed orders per person-hour.
The aggregate rates are 100 eaches, 20 lines, and 6.667 orders per person-hour.
The matching time boundary makes the arithmetic legitimate; event alignment still limits interpretation.

### 6.3 Reconcile paid hours

| Shift | Pick | Pack | Support | Total |
|---|---:|---:|---:|---:|
| A | 70 h | 30 h | 20 h | 120 h |
| B | 68 h | 32 h | 20 h | 120 h |

These are person-hours, not elapsed shift hours.
The source payroll total is 120 paid person-hours per shift; no hours are unclassified in this example.
If A reports 12,600 / 70 = 180 picked eaches per direct pick hour, that is a different metric.
It excludes 50 paid person-hours of pack and support and cannot replace 105 eaches per paid fulfillment hour.
The direct-pick rate can describe picking only if the 70-hour classification and output scope are appropriate.
It is not sufficient to staff the entire fulfillment operation.

### 6.4 Test the mix explanation

Suppose each accepted unit and all paid fulfillment hours can be assigned to one work type.
The hours include allocated support; allocation is assumed accurate for this example.

| Shift | Work type | Hours | Accepted eaches | Eaches per person-hour |
|---|---|---:|---:|---:|
| A | Easy | 90 h | 10,800 | 120 |
| A | Difficult | 30 h | 1,800 | 60 |
| B | Easy | 70 h | 8,400 | 120 |
| B | Difficult | 50 h | 3,000 | 60 |

A's easy share is 90 / 120 = 75%; B's is 70 / 120 = 58.33%.
Both shifts have the same within-type rates in these illustrative data.
The pooled easy weight is (90 + 70) / 240 = 2/3.
The pooled difficult weight is (30 + 50) / 240 = 1/3.
A standardized rate is (2/3 × 120) + (1/3 × 60) = 100 eaches per person-hour.
B's standardized rate is also 100 eaches per person-hour.
Here, the 10-each raw gap is entirely accounted for by the two-category mix in the constructed data.
In real operations, this coarse adjustment cannot establish that mix is the only cause.

### 6.5 Plan, forecast, and sensitivity

The next-week workload forecast is 26,400 accepted eaches under the same process and mix.
The pooled 100-each-per-paid-hour rate implies 26,400 / 100 = 264 paid fulfillment person-hours.
At 40 planned paid hours per person, 264 / 40 = 6.6 people, rounded up to 7 scheduled people.
This is a plan, not a staffing commitment; it assumes all 40 paid hours are available to this scope.
If a person contributes only 36 in-scope paid hours, 264 / 36 = 7.33, rounded up to 8 people.
Do not add an extra allowance for paid support time already included in the 100-each rate.
If the actual work mix changes, recompute hours by type rather than assuming the pooled rate persists.

### 6.6 Quality and service limits

Suppose the 24,000 eaches are accepted using a fixed audit rule; late defects can restate the figure.
A measured quality yield requires gross audited output and accepted output, not just accepted counts.
The example does not provide those gross counts and therefore does not claim a numerical quality rate.
The example also does not claim that the 1,600 orders are a cohort costed by the 240 hours.
Order completion can lag picking, and work in process can rise or fall across the week.
The plan uses an each-based period-flow rate; a shift in units per line can invalidate it.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the worked example, checks reconciliations, and computes standardized rates.
Copy the complete block into `productivity.py` and run `python productivity.py`.
It is a calculation demonstration using already reconciled aggregates, not a production event pipeline.
The production pipeline must implement the event, quality, and timekeeping rules in section 5.

```python
from math import ceil

shifts = {
    'A': {'units': 12600, 'lines': 2520, 'orders': 840,
          'pick': 70, 'pack': 30, 'support': 20},
    'B': {'units': 11400, 'lines': 2280, 'orders': 760,
          'pick': 68, 'pack': 32, 'support': 20},
}
strata = {
    'A': {'easy': (10800, 90), 'difficult': (1800, 30)},
    'B': {'easy': (8400, 70), 'difficult': (3000, 50)},
}

def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None

def hours(row):
    return row['pick'] + row['pack'] + row['support']

for name, row in shifts.items():
    h = hours(row)
    assert sum(v[0] for v in strata[name].values()) == row['units']
    assert sum(v[1] for v in strata[name].values()) == h
    print(f"{name}: {ratio(row['units'], h):.3f} eaches/h; "
          f"{ratio(row['lines'], h):.3f} lines/h; "
          f"{ratio(row['orders'], h):.3f} orders/h")

h_total = sum(hours(row) for row in shifts.values())
u_total = sum(row['units'] for row in shifts.values())
print(f'Pooled: {ratio(u_total, h_total):.3f} eaches/h')

weights = {
    group: sum(strata[name][group][1] for name in strata) / h_total
    for group in ('easy', 'difficult')
}
for name in shifts:
    standardized = sum(
        weights[group] * ratio(*strata[name][group])
        for group in weights
    )
    print(f'{name} standardized: {standardized:.3f} eaches/h')

required_hours = 26400 / ratio(u_total, h_total)
print(f'Plan: {required_hours:.1f} h; {ceil(required_hours / 40)} people')
```

Expected output:

```text
A: 105.000 eaches/h; 21.000 lines/h; 7.000 orders/h
B: 95.000 eaches/h; 19.000 lines/h; 6.333 orders/h
Pooled: 100.000 eaches/h
A standardized: 100.000 eaches/h
B standardized: 100.000 eaches/h
Plan: 264.0 h; 7 people
```

The `assert` checks are example safeguards, not a complete data-quality system.
They fail if the stratified units or hours do not reconcile to the shift totals.
The `ratio` function returns `None` for a nonpositive denominator; callers must handle that result.
The supplied example has positive denominators, so formatting `None` is not encountered.
The standardized calculation requires observed, positive hours in every included stratum.
No individual records are required for this demonstration.

## 8. Data, standards, and regulatory context

Rules depend on jurisdiction, employment status, date, and the actual operational practice.
The following is educational context as of November 2026, not legal advice.
A metric definition does not itself determine whether a particular quota or employment practice is lawful.

### 8.1 Source system roles

A warehouse management system (WMS) records tasks, locations, inventory states, and confirmations.
An order management system supplies order and line identity and subsequent amendments.
A timekeeping or payroll system supplies compensable time subject to correction and local policy.
A transportation or shipping system may establish the final shipment completion event.
A quality system may record defects only after a delay.
A shared identifier and versioned event policy are needed before joining these sources.

### 8.2 GS1 identifiers

[GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications)
define a system of identifiers and barcode applications.
A Global Trade Item Number (GTIN) identifies a trade item, not inherently a customer order or a labor hour.
Packaging levels and conversion factors still need to be understood at the transaction date.
A barcode scan is an event observation; it does not by itself prove good completion or unique labor time.
GS1 conventions do not impose the site-specific definition of an accepted line or an hourly denominator.

### 8.3 Event semantics

[GS1 EPCIS](https://www.gs1.org/standards/epcis) describes a standard for sharing visibility event data.
Its event model can help express what, when, where, why, and how objects are involved.
It does not automatically deduplicate the warehouse's local scans or allocate employee time.
Use source-specific event IDs, correction semantics, and an auditable business-key strategy.
A late-arriving correction should be visible in the report's as-of timestamp and revision policy.

### 8.4 U.S. pay and safety

The U.S. Department of Labor explains the Fair Labor Standards Act (FLSA) and hours worked in
[Fact Sheet #22](https://www.dol.gov/agencies/whd/fact-sheets/22-flsa-hours-worked).
An activity code or dashboard denominator cannot by itself decide what time is compensable.
The applicable law and facts govern; do not omit compensable waiting or required work from payroll.
Consult qualified counsel and the authoritative wage-and-hour records for actual decisions.

The U.S. Occupational Safety and Health Administration (OSHA) describes employer obligations under
[the Occupational Safety and Health Act](https://www.osha.gov/laws-regs/oshact/section5).
A productivity metric does not displace applicable safety duties.
Safety and health reporting, ergonomics, and safe-work procedures require independent review.

### 8.5 Warehouse quota rules

California's [Labor Code sections 2100–2111](https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?lawCode=LAB&division=2.&title=&part=7.&chapter=12.&article=)
set out warehouse-distribution-center quota provisions for covered employers and employees.
Coverage, notice, disclosures, and other obligations depend on the statute and facts.
Do not assume that an informal dashboard is exempt or that every warehouse is covered.
Check current text and local legal advice before implementing or revising a quota.

### 8.6 Privacy and retention

Individual event and timestamp histories can reveal sensitive behavioral patterns.
Define a legitimate purpose, restrict access, limit retention, and consider aggregation.
Applicable privacy, labor, collective bargaining, and consultation requirements vary.
A pseudonymous employee ID is not necessarily anonymous when it can be linked back to a person.
Separate improvement analysis from disciplinary use and make governance explicit.

## 9. Industry practice and case evidence

External cases show why measurement design matters; they do not validate this lesson's synthetic rates.
Company disclosures and case studies are not controlled estimates of labor productivity.
Read each source's stated scope, date, measurement definition, and incentives before transferring a claim.

### 9.1 Amazon and OSHA inspections

OSHA's [January 2023 announcement](https://www.osha.gov/news/newsreleases/national/01182023)
describes citations at three Amazon warehouse facilities after inspections, including ergonomic hazards.
The release concerns OSHA's allegations and enforcement actions, not a generic rate benchmark.
It is evidence that safety and work design require attention alongside throughput.
[INFERENCE] A dashboard that rewards speed without checking ergonomic outcomes can conceal a material risk.
The release does not establish a universal causal effect of any specific productivity metric.
Enforcement outcomes and later developments should be checked separately before relying on the case.

### 9.2 Ocado and automated fulfillment

[Ocado Group's annual reports](https://www.ocadogroup.com/investors/results-centre) discuss its
technology and fulfillment operations in company publications.
Company-reported operating indicators are affected by site maturity, order mix, automation,
and the boundaries chosen for labor and costs.
[INFERENCE] Comparing an automated site with a manual site on direct pick hours alone may omit
maintenance, engineering, or upstream preparation needed to deliver the output.
This is a measurement lesson, not a claim about a specific Ocado productivity improvement.

### 9.3 U.S. Bureau of Labor Statistics productivity

The [Bureau of Labor Statistics productivity overview](https://www.bls.gov/productivity/)
explains that labor productivity relates output to labor input.
Official productivity statistics use defined output and hours concepts appropriate to their statistical scope.
A warehouse eaches-per-paid-hour metric is a local operational measure, not automatically comparable
with national value-added productivity or total factor productivity.
[INFERENCE] A metric label is insufficient to establish comparability between a national series and a facility dashboard.

### 9.4 What transfers and what does not

A useful transfer is the discipline of pairing a defined output with a defined input.
A hazardous transfer is to adopt another company's reported rate as a quota without harmonizing scope.
Compare process design, output definitions, hours, staffing model, geography, and quality criteria.
Evidence strength varies: regulatory inspection statements document an agency action;
a company report is a company disclosure; neither replaces a local causal evaluation.

## 10. Failure modes and red flags

The table lists minimum controls. No single control guarantees a fair or safe result.

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mixed units | Case and each scans rise together | Different package sizes summed | Inflated output | Version conversions; quarantine missing factors |
| Duplicate events | Repeated event IDs | Retransmission counted twice | Inflated numerator | Idempotent event ingestion |
| Reversal ignored | Corrections arrive after close | Original remains counted | False completion | Signed ledger and restatement |
| Partial line counted | Multiple scans per order line | Fragments treated as finished lines | Inflated lines | Lifecycle completion rule |
| Split order counted twice | Several shipments per order | Shipment used as order key | Inflated orders | Order-level completion state |
| Missing time | Unclassified or absent punches | Hours disappear | Inflated rate | Reconcile to payroll and disclose coverage |
| Shift boundary mismatch | Output at midnight jumps | Different calendars used | Spurious daily volatility | Shared half-open periods |
| Double-counted labor | Overlapping activity intervals | Person-time allocated twice | Depressed rate | Enforce interval reconciliation |
| Hidden support | Direct hours shrink after outsourcing | Support moves out of scope | False improvement | Inclusive labor view and cost bridge |
| Mix reversal | Pooled gap conflicts with strata | Different workload shares | Misleading comparison | Standardize with common weights |
| Rework rewarded | Gross scans rise with defects | Repeat work counts as new output | More errors and apparent speed | Net good output; include correction hours |
| Late quality | Recent periods look unusually good | Defects not yet observed | Premature claim | Maturity window and revisions |
| Small sample | High rates from few hours | Random variation dominates | Unstable ranking | Show counts and wider periods |
| Unsafe incentives | Hazard reports fall as rates rise | Workers fear reporting | Unrecognized harm | Independent safety review and protections |
| Clock versus person-hours | Staffing changes but denominator does not | Elapsed time used as labor input | Invalid staffing comparison | Sum person-time |
| Zero denominator | Blank hour fields with output | Missing data interpreted as zero | Infinite or fabricated rate | Mark undefined and investigate |
| Cross-shift work | One shift completes another's work | Completion-period attribution | Unfair shift comparison | Report WIP and use cohort study if needed |
| Changing policy | Rate jumps at mapping change | Classification redefined | False trend | Version and bridge definitions |
| Incentive gaming | Easy tasks selected first | Output measure rewards selection | Delayed difficult work | Monitor mix, backlog and service |
| Unapproved individual surveillance | Detailed traces repurposed | Purpose and access expand | Trust, privacy and legal risk | Governance, access controls and consultation |

## 11. Assumptions and limits

Every row should be reviewed before interpreting an observed gap or building a plan.

| Assumption | Why needed | What breaks it | Practical test |
|---|---|---|---|
| Event IDs are stable | Deduplication depends on identity | Reused or absent IDs | Trace samples to source logs |
| Completion rule is consistent | Output must mean the same thing | Cancellations treated differently | Audit lifecycle examples |
| Unit factors are correct | Eaches require valid conversion | Packaging changes | Compare transaction to item master at event date |
| Paid hours are complete | Denominator must include scoped labor | Missing agency workers | Reconcile to payroll and agency invoices |
| Activity codes are comparable | Direct-hour boundaries must match | Different local code maps | Review mappings with both supervisors |
| Intervals do not overlap | Person-time cannot count twice | Concurrent assignments | Detect overlapping time spans |
| Support allocation is reasonable | Work-type rates require comparable cost | Unequal maintenance burden | Sensitivity-test alternative allocations |
| Common support exists | Standardization needs observed strata | A site never handles difficult work | Inspect hours and volumes by stratum |
| Strata capture important mix | Adjustment should reduce mix bias | Travel or item size varies within type | Add observable attributes and inspect residuals |
| Recent quality is mature | Good output requires defect capture | Returns arrive weeks later | Compare historical defect arrival curves |
| Process remains stable | Historical rate informs future planning | Automation or layout changes | Segment pre/post and validate assumptions |
| Time buckets align | Rate compares simultaneous scope | Different timezones or overnight shifts | Trace boundary events and split intervals |
| Period flow is informative | Completions and hours are related over time | Large change in work in process | Show opening and closing WIP |
| Observed work is representative | Plan uses historical performance | Peak mix differs from history | Compare forecast attributes with observed distribution |
| Reporting does not distort work | Behavior may change under incentives | Task cherry-picking | Monitor backlog, quality and difficult-work share |
| Definition is legally reviewed | Measurement can affect workers | Local quota or privacy rules apply | Seek current jurisdiction-specific review |

No set of assumptions establishes an individual causal effect by itself.
Unmeasured process differences remain possible even when all listed tests pass.
### 11.1 Decision-specific interpretation checks

These short scenarios show why the same label can support different conclusions.
Each describes a fact pattern, the immediate analytical risk, and a decision rule.
They are teaching scenarios, not statements about particular employers.
**Bulk order surge.**

- Situation: A shift receives fewer but much larger lines.
- Risk: Units per hour can rise while lines per hour stays flat.
- Response: Show units per line and both rates before changing staffing.
**Small-order surge.**

- Situation: The same number of eaches arrives across more orders.
- Risk: Order-handling work rises even when unit volume is unchanged.
- Response: Forecast orders and lines as well as eaches.
**New pick zone.**

- Situation: A distant zone opens halfway through the week.
- Risk: Travel time changes within a supposedly stable work category.
- Response: Split the period or add a zone stratum.
**Carton conversion.**

- Situation: A product changes from six eaches per case to eight.
- Risk: A current master conversion misstates older transactions.
- Response: Use the factor effective at the transaction time.
**Duplicate messaging.**

- Situation: A network retry delivers an identical event twice.
- Risk: A naive sum counts the same confirmation twice.
- Response: Deduplicate by event identity before summing quantities.
**Legitimate partial picks.**

- Situation: One line requires two tasks from different locations.
- Risk: Deduplicating on line ID discards genuine work.
- Response: Keep distinct task events and compute one lifecycle completion.
**Reverse after close.**

- Situation: A recorded pick is reversed on the following day.
- Risk: A frozen report retains an output that no longer exists.
- Response: Apply the correction and disclose the restatement policy.
**Cancelled remainder.**

- Situation: An order line is partially picked before the balance is cancelled.
- Risk: The completion requirement changes during its lifecycle.
- Response: Use the documented final eligible requirement and retain amendment history.
**Order split shipment.**

- Situation: Two cartons for one order leave on different days.
- Risk: Counting cartons as orders overstates completed orders.
- Response: Count one order at its last required shipment.
**Shorted order.**

- Situation: An order is short-picked and remains open.
- Risk: Counting release or first shipment as complete masks a service failure.
- Response: Apply the explicit full-completion rule and track the short separately.
**Overnight shift.**

- Situation: A shift starts before midnight and ends after midnight.
- Risk: Assigning every hour to the start date distorts daily productivity.
- Response: Split person-time at the local reporting boundary.
**Daylight-saving change.**

- Situation: A shift crosses a clock transition.
- Risk: Naive local subtraction may produce the wrong elapsed hours.
- Response: Use timezone-aware timestamps and actual elapsed intervals.
**Agency hours absent.**

- Situation: Temporary workers appear in task logs but not payroll extracts.
- Risk: Output is captured while some person-hours are missing.
- Response: Reconcile agency hours before publishing the combined rate.
**Unpaid meal.**

- Situation: A meal interval is unpaid under the applicable timekeeping record.
- Risk: Counting scheduled presence as paid hours changes the denominator.
- Response: Use the correct paid interval and keep scheduled hours separate.
**Paid waiting.**

- Situation: Workers wait for replenishment during a paid shift.
- Risk: Restricting the denominator to scanner activity hides the wait.
- Response: Include paid waiting in an inclusive staffing measure.
**Training day.**

- Situation: New staff spend paid time in required training.
- Risk: An active-task metric excludes a real staffing cost.
- Response: Publish both operational and inclusive rates with training hours.
**Indirect support.**

- Situation: A mechanic supports several automated areas.
- Risk: Direct pick hours omit labor needed to operate the equipment.
- Response: Report inclusive site hours or an explicit support allocation.
**Outsourced maintenance.**

- Situation: A contractor replaces employees without changing physical work.
- Risk: A payroll-only series improves while total resource use may not.
- Response: Bridge employment categories and report contractor cost or hours.
**Replenishment outage.**

- Situation: Pickers wait because forward locations are empty.
- Risk: Low pick rate is attributed to pickers rather than a system constraint.
- Response: Pair rates with availability and replenishment indicators.
**Late trailer arrival.**

- Situation: Outbound workers have a paid idle interval awaiting transport.
- Risk: Throughput falls for reasons outside the packing process.
- Response: Separate delay diagnostics from paid-hour staffing totals.
**Break coding change.**

- Situation: A site changes the activity code assigned to paid breaks.
- Risk: Direct hours move even if actual work does not.
- Response: Bridge the old and new coding before comparing trends.
**Task interleaving.**

- Situation: One person handles pick and replenish concurrently.
- Risk: Overlapping activity assignments count one person twice.
- Response: Split or allocate the interval so assigned hours sum to elapsed paid time.
**Shared supervision.**

- Situation: A supervisor covers two areas with different headcounts.
- Risk: Adding all supervision to both areas double counts labor.
- Response: Choose an allocation driver and reconcile to the original total.
**Unlinked person.**

- Situation: A task has no reliable employee identifier.
- Risk: Individual attribution fails even if team output is known.
- Response: Report team-level results and quantify unmatched tasks.
**Weekend close.**

- Situation: One shift ships work picked earlier by another shift.
- Risk: Period order completions do not equal that shift’s self-contained output.
- Response: Expose WIP and use a linked cohort for cohort costing.
**Growing backlog.**

- Situation: Output remains stable while released orders increase.
- Risk: A stable completion rate hides worsening flow and service.
- Response: Show release, completion, opening WIP, and closing WIP.
**Easy-work selection.**

- Situation: Workers choose high-volume easy tasks before difficult tasks.
- Risk: A rising unit rate may reflect selection rather than improvement.
- Response: Monitor difficult-work backlog and the mix of completed work.
**Double handling.**

- Situation: A damaged item is picked again.
- Risk: Both picks may be counted as customer-ready output.
- Response: Count net accepted output and retain rework labor.
**Late customer returns.**

- Situation: Defects emerge several weeks after shipment.
- Risk: Recent quality-adjusted rates look better because of incomplete observation.
- Response: Compare cohorts with equal maturity and revise as returns arrive.
**Unmatched exclusions.**

- Situation: One site excludes damaged inventory and the other includes it.
- Risk: The compared numerators have different eligibility rules.
- Response: Harmonize exclusions or show a quantified bridge.
**Zero activity interval.**

- Situation: A paid interval has no completed order.
- Risk: A zero numerator is meaningful if hours are valid.
- Response: Report zero orders per hour for that interval and retain the hours.
**Missing denominator.**

- Situation: Output is recorded but the labor extract has not arrived.
- Risk: Replacing missing hours with zero invents a rate.
- Response: Mark the rate unavailable until hours reconcile.
**Low volume.**

- Situation: A new zone has one hour of observed work.
- Risk: A large rate may be driven by a few events.
- Response: Publish sample size and avoid a strong comparison.
**Definition revision.**

- Situation: A dashboard adds support hours in June.
- Risk: A drop in rate can be entirely definitional.
- Response: Version the metric and recalculate history when feasible.
**Common-weight choice.**

- Situation: One group rarely handles difficult work.
- Risk: A reference distribution heavily weights a sparsely observed stratum.
- Response: Disclose stratum sample sizes and test sensitivity to weights.
**No common support.**

- Situation: One group has no observations for a refrigerated zone.
- Risk: Its refrigerated rate cannot be measured from these data.
- Response: Limit the adjusted comparison to overlap and state the excluded scope.
**Many workers at once.**

- Situation: Three people each work a two-hour interval.
- Risk: Two elapsed hours are mistaken for two labor hours.
- Response: Use six person-hours before calculating the rate.
**Maintenance shutdown.**

- Situation: An automation outage reduces throughput for a paid shift.
- Risk: Direct task time hides the paid downtime.
- Response: Show inclusive hours and an outage annotation.
**New automation.**

- Situation: A robot increases units but adds engineering and maintenance work.
- Risk: A pick-only rate overstates total labor savings.
- Response: Compare total in-scope labor and costs over equivalent periods.
**Metric as quota.**

- Situation: A descriptive historical rate becomes a mandatory individual threshold.
- Risk: The new use introduces safety, fairness, and legal risks.
- Response: Obtain worker, safety, and legal review before any quota decision.
**Plan versus commitment.**

- Situation: A forecast implies seven people on average.
- Risk: Treating a rounded point estimate as guaranteed capacity ignores variation.
- Response: Show low and high scenarios and distinguish plan from commitment.
**Accuracy versus speed.**

- Situation: A shift increases scans while error reports also increase.
- Risk: The gross rate rewards speed even when usable output falls.
- Response: Evaluate accepted output, correction hours, and safety together.
**Automation ramp.**

- Situation: A new site has low early throughput and extra commissioning staff.
- Risk: Comparing the ramp week to a mature site confounds maturity and technology.
- Response: Compare like-for-like maturity and retain total launch costs.
**Same total hours.**

- Situation: Two sites both report 100 paid hours.
- Risk: The same denominator may contain different job categories.
- Response: Compare classified hour components before comparing rates.
**Mean of means.**

- Situation: A 1-hour shift has a much higher rate than a 99-hour shift.
- Risk: Equal weighting gives the short shift half the influence.
- Response: Sum outputs and divide by 100 total hours.
**Incomplete quality audit.**

- Situation: Only high-risk orders are inspected.
- Risk: Audit yield is not necessarily the yield of all orders.
- Response: Report the sampling design and avoid generalizing without adjustment.
**Changed order policy.**

- Situation: A new process permits partial shipments to count as finished.
- Risk: Order counts rise without equivalent service completion.
- Response: Preserve both definitions during a bridge period.
**Compensability question.**

- Situation: An employer proposes to exclude required preparatory work.
- Risk: A performance denominator cannot override wage law.
- Response: Consult authoritative time records and qualified legal advice.
**Return-to-stock work.**

- Situation: A cancelled item is put back in inventory.
- Risk: Return work consumes labor but adds no shipped order.
- Response: Include its hours in the relevant inclusive scope and track volume.
**Small common stratum.**

- Situation: Both shifts handle fragile goods, but one has only two lines.
- Risk: A standardized estimate may be very unstable.
- Response: Report the small sample and avoid a confident adjusted ranking.
**Mixed channels.**

- Situation: Retail replenishment and consumer orders share a building.
- Risk: Pallet-heavy retail work and small parcel work require different handling.
- Response: Separate channels and state what common labor is allocated.

**Return to work.**

- Situation: A worker returns on a temporary accommodation.
- Risk: A team average cannot establish an appropriate individual expectation.
- Response: Apply the applicable accommodation process, not a generic productivity target.

**Equipment sharing.**

- Situation: Two packers wait for one label printer.
- Risk: Low output reflects a shared capacity constraint as well as labor.
- Response: Record the equipment constraint and compare feasible operating conditions.

**Inventory discrepancy.**

- Situation: The WMS asks a picker to retrieve nonexistent stock.
- Risk: Search and exception resolution consume paid hours without confirmed output.
- Response: Retain those hours and monitor inventory accuracy separately.

**Quality threshold.**

- Situation: Two sites use different acceptance thresholds.
- Risk: A site can raise accepted output merely by relaxing its inspection.
- Response: Harmonize quality definitions before comparing good-output rates.

**Calendar coverage.**

- Situation: One report includes holidays and another excludes them.
- Risk: Different operating days distort comparability and staffing assumptions.
- Response: Align calendars or explicitly adjust the reporting populations.

**Shipment correction.**

- Situation: A shipping transaction is voided and later recreated.
- Risk: Counting both transactions as distinct orders inflates output.
- Response: Use the order lifecycle and record the void and replacement.

**Late payroll adjustment.**

- Situation: A missed punch is corrected after the first report.
- Risk: The originally published denominator was incomplete.
- Response: Version and restate the rate when the authoritative hours change.

**Fragmented time.**

- Situation: A worker alternates five-minute tasks across processes.
- Risk: Rounding each task interval upward creates extra labor hours.
- Response: Preserve precise intervals and round only at the reporting stage.

**Unrepresentative reference.**

- Situation: A pooled work mix is dominated by the larger site.
- Risk: Standardization may emphasize that site’s operating conditions.
- Response: State the reference weights and compare a decision-relevant alternative.

**Task assignment.**

- Situation: The most experienced team receives unusually difficult work.
- Risk: Raw rates combine selection and operating differences.
- Response: Compare matched work while avoiding claims about personal ability.

**Forecast error.**

- Situation: Actual orders differ materially from the plan.
- Risk: A staffing miss may reflect demand error rather than the selected labor rate.
- Response: Separate forecast error from rate and attendance variance in the review.

## 12. Exercises

Use a calculator or a short program. Show the numerator, denominator, unit, and scope.
All figures in these exercises are fictional and refer to one stated reporting period.

### Exercise 1 — Define the outputs

A customer order contains two uncancelled lines.
One line has three eaches and is picked in two tasks; the other has one each and is picked in one task.
All four eaches are accepted and the full order is shipped in the reporting period.

- Deliverable: count accepted eaches, completed lines, completed orders, and tasks.
- Acceptance: distinguish the three entity grains and explain why tasks do not equal lines.
- Time: 8 minutes.

### Exercise 2 — Reconcile the denominator

Six workers each receive eight paid hours in a day.
Across the team, direct picking is 30 person-hours, packing is 8 person-hours,
and support is 6 person-hours; four paid hours remain unclassified.

- Deliverable: calculate paid hours and classified-hour coverage.
- Acceptance: state whether an inclusive rate can exclude the unclassified hours and why.
- Time: 10 minutes.

### Exercise 3 — Calculate three rates

During a period the site records 9,600 accepted eaches, 2,400 completed lines,
and 600 completed orders against 120 paid fulfillment person-hours.

- Deliverable: compute units, lines, and orders per paid fulfillment person-hour.
- Acceptance: show each division, state units, and identify the common time scope.
- Time: 8 minutes.

### Exercise 4 — Pool unequal exposures

Shift X completes 600 eaches in 5 person-hours; shift Y completes 4,500 eaches in 50 person-hours.
Their event definitions and labor scope match.

- Deliverable: compute each rate, the pooled rate, and the unweighted average of shift rates.
- Acceptance: explain the weighting difference and identify the correct combined rate.
- Time: 10 minutes.

### Exercise 5 — Standardize mix

Group A works 8 hours on easy tasks at 120 eaches/hour and 2 hours on hard tasks at 60 eaches/hour.
Group B works 2 hours on easy tasks at 120 eaches/hour and 8 hours on hard tasks at 60 eaches/hour.
Use fixed weights of 50% easy and 50% hard for both groups.

- Deliverable: calculate both raw and standardized rates.
- Acceptance: explain the difference between workload mix and within-type performance.
- Time: 12 minutes.

### Exercise 6 — Build a data contract

A WMS sends pick confirmations and reversals; payroll sends paid intervals.
Some order lines span multiple tasks and some workers change activities during a shift.

- Deliverable: sketch keys, timestamps, data transformations, and reconciliation checks for a team-level rate.
- Acceptance: address corrections, conversion, interval overlap, missing labor, and period boundary alignment.
- Time: 20 minutes.

### Exercise 7 — Critique a claim

A manager reports a 20% increase in pick scans per active scanner hour after automation.
The same period has more large single-line orders, newly outsourced maintenance,
and a quality audit that has not yet matured.

- Deliverable: write a decision memo that distinguishes the observed metric from a total-site gain.
- Acceptance: identify at least four alternative explanations or limitations and a safe next step.
- Time: 15 minutes.

### Exercise 8 — Design a planning dashboard

A director needs next-week staffing for mixed retail and e-commerce fulfillment.
Some workers split time across picking, packing, and replenishment.
Customer returns may be recorded several weeks after shipment.

- Deliverable: propose a metric contract, dashboard views, plan calculation, and governance controls.
- Acceptance: include all three rates, paid-hour reconciliation, mix, quality lag, safety,
  a forecast-versus-plan distinction, and a policy for missing or zero denominators.
- Time: 25 minutes.

## 13. Self-check answers

### Answer 1 — Entity grains

The output is four accepted eaches, two completed order lines, and one completed order.
The three pick tasks are execution instructions and can split a line.
Counting three tasks as three lines would overstate the completed lines by one.
The answer assumes the stated acceptance and shipment events meet the definitions.

### Answer 2 — Person-time and coverage

Six people × eight paid hours = 48 paid person-hours.
Classified hours are 30 + 8 + 6 = 44 person-hours.
Coverage is 44 / 48 = 91.67% of paid person-hours.
The remaining four hours are not zero hours and must be reconciled or disclosed.
An inclusive paid-hour rate uses all 48 hours if all belong to the defined scope.
A direct-pick rate uses 30 hours only when its label and numerator support that narrower question.

### Answer 3 — Three rates

9,600 / 120 = 80 accepted eaches per paid fulfillment person-hour.
2,400 / 120 = 20 completed lines per paid fulfillment person-hour.
600 / 120 = 5 completed orders per paid fulfillment person-hour.
These measure different output grains; one is not a substitute for another.
The orders and lines need not be the same cohort as the units for a period-flow report.

### Answer 4 — Pool using hours

X's rate is 600 / 5 = 120 eaches per person-hour.
Y's rate is 4,500 / 50 = 90 eaches per person-hour.
The unweighted average is (120 + 90) / 2 = 105 eaches per person-hour.
The pooled rate is (600 + 4,500) / (5 + 50) = 5,100 / 55 = 92.727 eaches per person-hour.
The pooled rate gives Y ten times the weight of X because Y supplies ten times the hours.
The unweighted average describes an average of two shift rates, not output per total person-hour.

### Answer 5 — Work-mix adjustment

A completes 8 × 120 + 2 × 60 = 1,080 eaches in 10 hours, or 108 eaches/hour.
B completes 2 × 120 + 8 × 60 = 720 eaches in 10 hours, or 72 eaches/hour.
Using the common mix, each standardized rate is 0.5 × 120 + 0.5 × 60 = 90 eaches/hour.
The raw 36-each-per-hour gap is caused by the constructed mix difference in this example.
Real categories may hide residual variation; equality of standardized rates is not proof of equal effort.

### Answer 6 — Data contract rubric

A strong response defines a stable event ID and a separate order-line business key.
It retains event time, source time, event type, quantity, unit, reversal linkage, and effective conversion.
It specifies a final-state rule for line completion and an as-of policy for corrections.
It defines a person, site, start, end, activity, paid status, and source for each labor interval.
It prevents overlapping person-time from being counted twice and splits at reporting boundaries.
It reconciles included plus excluded plus unclassified hours to payroll and agency sources.
It discloses unmatched work and prohibits an individual claim when attribution is unreliable.
A weak response simply joins scans to payroll by date and counts rows.

### Answer 7 — Critique rubric

A strong response preserves the 20% figure as a reported change in scans per active scanner hour.
It explains that scans are not necessarily accepted eaches, completed lines, or completed orders.
It names the narrower denominator, order-mix change, outsourced maintenance,
immature quality observations, and any concurrent process changes as limitations.
It proposes an inclusive labor and cost bridge, comparable mix analysis,
a mature quality cohort, and a separate safety review before a decision.
It does not claim that automation caused a 20% total-site productivity gain.

### Answer 8 — Dashboard rubric

A strong contract names site, interval, event definitions, units, population, and source versions.
It shows accepted units, completed lines, and completed orders with their matching paid hours.
It displays direct and inclusive labor separately, reconciles shared hours, and shows coverage.
It stratifies by channel and relevant workload attributes without hiding small or absent strata.
It separates provisional from mature quality, shows WIP and safety context,
and displays counts and rates rather than rankings alone.
It turns an explicit workload forecast into an hours plan under stated rate and availability assumptions.
It labels staffing as a plan rather than a promise, and includes sensitivity to mix and hours per person.
It specifies access, retention, worker consultation, correction, and jurisdiction-specific legal review.
It marks a zero denominator undefined and missing time unavailable pending reconciliation.

## 14. Interview questions

### 1. What is the difference between units, lines, and orders per hour?

A strong answer defines each output entity and its completion event, and names the labor denominator.
It explains why changes in line size or lines per order make the rates move differently.
A weak answer treats every scan as both a unit and an order.

### 2. Why are person-hours different from elapsed hours?

A strong answer sums the time contributed by all people and prevents overlapping assignments.
It gives the example that three people working two hours contribute six person-hours.
A weak answer divides team output by two elapsed hours regardless of staffing.

### 3. How would you compare two sites with different work mix?

A strong answer harmonizes definitions, reports raw rates, and compares within supported strata.
It applies a common reference distribution while disclosing missing overlap and residual confounding.
A weak answer declares the higher pooled rate the superior workforce.

### 4. Why not average hourly rates equally?

A strong answer uses total output divided by total hours and explains hour weights.
It distinguishes that pooled operational rate from a mean of equally weighted shift statistics.
A weak answer uses an unweighted average without examining exposure.

### 5. How should corrections be handled?

A strong answer retains event identity, reversals, effective-time logic, and a restatement policy.
It distinguishes duplicate messages from legitimate fragments of the same line.
A weak answer drops all records with a repeated order-line identifier.

### 6. What changes when a metric is used for staffing?

A strong answer includes the support and paid time required by the staffing boundary.
It tests future mix, planned available hours per person, and sensitivity of the resulting plan.
A weak answer converts active scanner time directly into headcount without an hours bridge.

### 7. What if orders are completed by a later shift?

A strong answer distinguishes period flow from a release cohort and reports work in process.
It avoids attributing the completed order solely to the final shift for individual assessment.
A weak answer assumes the shift that ships an order performed all its work.

### 8. How do you prevent quality gaming?

A strong answer reports accepted output, defect maturity, and correction labor beside gross scans.
It acknowledges that targeted audits and delayed returns complicate yield estimates.
A weak answer counts every repeat scan as a new successful unit.

### 9. What is the role of safety and labor law?

A strong answer treats safety review and legal obligations as independent of a productivity target.
It identifies local quota rules, compensable time, and privacy as questions for qualified review.
A weak answer assumes that a dashboard definition determines legal compliance.

### 10. What would make you withhold a comparison?

A strong answer names missing labor, inconsistent boundaries, no common support, or immature quality.
It specifies the evidence needed to repair the comparison and preserves raw coverage information.
A weak answer fills missing hours with zero and publishes an infinite rate.

## 15. Key terms

- **Accepted output:** output passing the stated quality and completion rule; see [mechanics](#5-the-mechanics-step-by-step).
- **Activity code:** classification of person-time; see [labor classification](#510-classify-labor).
- **Active task hour:** recorded execution time, not necessarily all paid time; see [definitions](#4-definitions-and-vocabulary).
- **Allocation:** distribution of shared labor using a documented driver; see [shared labor](#511-handle-shared-labor).
- **Attribution:** assignment of events or hours to a reporting scope; see [alignment](#512-align-attribution).
- **Base unit:** comparable physical item unit after conversion; see [unit conversion](#54-establish-unit-conversion).
- **Cohort:** orders grouped by entry or release; see [mental model](#3-the-core-mental-model).
- **Completion event:** chosen lifecycle state transition; see [numerator](#53-choose-the-numerator).
- **Coverage:** fraction of relevant records or hours included; see [validation](#519-validate-and-reconcile).
- **Direct hour:** time in a defined operating process; see [worked reconciliation](#63-reconcile-paid-hours).
- **Each:** one physical base item; see [worked example](#6-worked-example).
- **Event ID:** identifier used to distinguish deliveries and events; see [deduplication](#58-deduplicate-and-net-events).
- **Forecast:** estimated future demand, not a staffing commitment; see [planning](#65-plan-forecast-and-sensitivity).
- **Good output:** accepted work under a defined quality rule; see [quality](#517-pair-output-with-quality).
- **Indirect hour:** support time outside a specified direct category; see [labor classification](#510-classify-labor).
- **Labor hour:** one person's hour of included work; see [definitions](#4-definitions-and-vocabulary).
- **Line:** an order-line entity under its completion rule; see [line lifecycle](#55-define-the-line-lifecycle).
- **Mix:** distribution of work attributes; see [mix diagnosis](#515-diagnose-work-mix).
- **Order:** customer order under the chosen completion rule; see [order lifecycle](#56-define-the-order-lifecycle).
- **Paid hour:** compensated time in the authoritative source; see [regulatory context](#84-us-pay-and-safety).
- **Person-hour:** sum of one-hour intervals across people; see [time intervals](#59-collect-time-intervals).
- **Plan:** proposed action based on assumptions, not a guarantee; see [planning](#521-use-rates-in-staffing).
- **Productivity:** output per specified input; see [mental model](#3-the-core-mental-model).
- **Quality yield:** accepted output divided by output submitted for acceptance; see [quality limits](#66-quality-and-service-limits).
- **Reversal:** event negating earlier recorded work; see [event ledger](#58-deduplicate-and-net-events).
- **Rework:** work repeated to correct a defect; see [failure modes](#10-failure-modes-and-red-flags).
- **Standardization:** application of common weights to stratum rates; see [mix adjustment](#516-standardize-cautiously).
- **Stratum:** subset intended to make work more comparable; see [worked mix](#64-test-the-mix-explanation).
- **Throughput:** completed quantity over a specified interval; see [definitions](#4-definitions-and-vocabulary).
- **Work in process:** work started and not yet completed; see [limits](#11-assumptions-and-limits).

## 16. Sources

Sources are checkable starting points; applicability and current versions should be verified for a real deployment.
The worked calculations and exercises are synthetic, not source-derived performance claims.

1. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Authoritative identifier and barcode specifications; does not define warehouse labor productivity.
2. GS1. [EPCIS standard](https://www.gs1.org/standards/epcis).
   Visibility event framework; local completion and correction rules still require specification.
3. U.S. Department of Labor, Wage and Hour Division. [Fact Sheet #22: Hours Worked Under the FLSA](https://www.dol.gov/agencies/whd/fact-sheets/22-flsa-hours-worked).
   Federal hours-worked guidance; applicability depends on facts and current law.
4. OSHA. [Occupational Safety and Health Act, section 5](https://www.osha.gov/laws-regs/oshact/section5).
   Statutory employer and employee duties; not a site-specific productivity standard.
5. California Legislature. [Labor Code sections 2100–2111](https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?lawCode=LAB&division=2.&title=&part=7.&chapter=12.&article=).
   Warehouse distribution center quota provisions; assess coverage and current text.
6. OSHA. [U.S. Department of Labor cites Amazon for unsafe conditions at three warehouses](https://www.osha.gov/news/newsreleases/national/01182023), January 18, 2023.
   Agency announcement of inspection citations; not a universal productivity benchmark.
7. U.S. Bureau of Labor Statistics. [Productivity](https://www.bls.gov/productivity/).
   Official overview of productivity statistics and their output/input scope.
8. Ocado Group. [Results centre](https://www.ocadogroup.com/investors/results-centre).
   Company publications and annual reports; company disclosures are not independent causal estimates.

## 17. Links

- [Module 13 index](_index.md): scope, sequence, and assessment.
- [Module 13 survey](../../../course/13_warehousing_fulfillment_labor.md): complete module overview.
- [Warehouse flow system](01_warehouse-flow-system.md): physical process and buffers.
- [WMS data model](02_wms-data-model.md): task, event, location, and state context.
- [Simulation and optimization labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md): related production-lab context.

Sibling topics may be planned in the module index; follow their status before assigning them.
