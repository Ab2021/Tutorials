---
case_id: "CASE-07-12-01"
case_title: "Robot near-miss incident response: containment and resume at a blind-corner cluster"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "incident"
domain: "Robot operations — warehouse autonomous mobile robots, blind-corner traffic conflicts"
primary_pattern: "containment ladder under a rare-event evidence gate"
secondary_patterns: ["kinematic safety envelope", "stratified rate attribution"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Site B operations manager, acting under delegated authority from the fleet safety review board"
source_index: "INDEX.md"
review_trigger: "Any change to the corner speed policy, the near-miss reporting definition, the map revision covering the mezzanine cluster, the localisation fallback path, the robot braking calibration, or the machinery-safety scope determination"
---

# Robot near-miss incident response: containment and resume at a blind-corner cluster

**Evidence disclaimer.** Everything in this case is ILLUSTRATIVE. The site, the robot fleet, the software versions, the near-miss reports, the timestamps, the labour rates, the geometry, the incident log, and the outcome are constructed for teaching. No named company, product, or deployment is described, and nothing here establishes that a real autonomy release ever behaved this way. Every arithmetic result marked DERIVED was recomputed independently and reproduces from the stated inputs; the inputs themselves are invented. Where a method, a standard, or a measurement practice is named — Poisson rate comparison, exact Poisson upper bounds, kinematic stopping distance — the reference identifies a technique whose applicability must be checked by a competent engineer in the real setting, not a compliance conclusion. The case does not certify any robot, any protective measure, or any site.

## 1. The One-Line Framing

Given 21 days of production telemetry in which a cluster of four blind-corner intersections produced 11 corner near-miss reports against 5 in the preceding 84 days, decide whether to resume unrestricted autonomous running in that cluster, resume it under corner-specific speed caps with a 23-day evidence gate, or keep the cluster in continuous human-escort mode for another 23 days, for the site operations manager who owns physical throughput and answers to the fleet safety review board, with a data cut-off of 2026-10-26T06:00:00Z.

The decision grain is one cluster, one software version, one 23-day window. It is not a fleet decision and not a permanent policy: it is renewed at the gate checkpoint, and it can be revoked at any hour by the duty safety lead without a further board sitting.

The baseline is the same 46 robots, in the same four intersections, on the same 18-hour operating schedule, under the preceding software version for the preceding 84 days. The comparison is paired by site and by population, not drawn from a benchmark catalogue, which is what makes a five-event baseline usable at all.

The deferral alternative is real and was seriously considered: hold the current escort arrangement, at an illustrative loaded labour cost of EUR 16,974 over the 23 days. That option is faster to execute than an engineering change (the cap was 16 engineering hours plus verification), requires no unproven assumption about why the cap should work, and preserves the status quo throughput exactly.

The decisive tension is that the cheap intervention (a corner speed cap, about EUR 2,282 including engineering and lost throughput) and the expensive one (continuous escort) are separated by a factor of 7.4 in cost, while the evidence supporting the cheap intervention is mechanistic rather than measured: the cap restores a positive kinematic margin under the observed detection latency, but it has never been run at this site, so its effect on near-miss frequency is `not-measured` at the moment of the decision.

**Source declaration.** Constructed scenario; all figures ILLUSTRATIVE; all derived arithmetic DERIVED and recomputed.

**Interview cue.** A reader should be able to state in thirty seconds: the unit (one cluster of four corners, one software version), the output (resume capped / resume uncapped / extend escort), the beneficiary (site operations manager and the review board), the baseline (same robots, preceding 84 days), the binding constraint (safety gate on corner near-miss rate plus a 23-day evidence window), and the evidence basis (illustrative).

**Acceptance check.** Input, output, beneficiary, baseline, constraint, and evidence basis are all identifiable from this section alone, and the framing names an alternative that a reasonable operator could actually choose.

## 2. Context & Background

### 2.1 The physical setting

The illustrative site is a regional distribution centre operating 592 autonomous mobile robots across four zones. The zone of interest is the North Mezzanine: a raised deck served by two ramps and four intersections where aisle traffic crosses at right angles around structural columns and pallet racking.

The mezzanine hosts 46 robots, 7.8 percent of the fleet. Each robot is a differential-drive unit with a 240 kg loaded mass, a maximum commanded speed of 1.5 m/s in open aisle, a nominal braking deceleration of 1.2 m/s² on the sealed deck surface, and a software speed limit of 0.6 m/s on the ramps.

The four intersections are labelled C1 to C4. C1 and C2 are wide, with 2.80 m of clear sight envelope from the column face to the conflict point. C3 and C4 are tight, with 2.55 m, because a pallet buffer zone was installed during a layout change eleven months before the incident window. Nobody recomputed the speed policy when the buffer went in.

That single sentence is the origin of the case. The buffer narrowed the geometry by 0.25 m, and the speed policy was written against the old geometry without a recorded link between the two artefacts. The layout change had a fire-egress review; it did not have a kinematics review, because nobody in the change process owned the connection between a footprint and a speed limit.

### 2.2 The workflow that produces corner conflicts

A transport mission assigns a tote move from a storage location to a pick station. The robot plans a route, acquires a traffic reservation for each intersection it will cross, and drives through the reservation at its zone speed limit.

Reservations exist for robot-to-robot conflicts. They do not exist for pedestrians: the site has shared walkways, and the fleet management service treats pedestrian detection as a local perception responsibility rather than a traffic-managed resource.

When a robot approaches a corner, its forward lidar sees around the corner only after the corner is within the sensor's field of view. The occluded wedge behind the column is, in the worst case, exactly the approach a pedestrian is using. The mezzanine has no mirrors at the corners and no convex-approach markings, both of which are ordinary in the site's other zones; the mezzanine predates the standard.

The autonomy stack under review is version 5.3. Its change log says it replaced the localisation fallback from a scan-matching path to a feature-map path "to reduce recovery latency in feature-poor aisles". The changelog does not mention perception of dynamic obstacles, and the release checklist classified the change as `localisation-only`.

That classification is the second origin of the case: a change scoped as localisation-only altered the rate at which the perception module received a trustworthy pose during a corner transit, which altered when the pedestrian detector's output was accepted by the planner. The release process had no field for "which other modules depend on this output", so the scope classification was an assertion rather than a computed dependency closure.

### 2.3 Who owns what

The site operations manager owns throughput, labour allocation, and the decision to keep a zone productive or idle it. The manager does not own the autonomy software and cannot modify it.

The fleet safety review board owns the safety case and delegates limited authority to the site manager to contain and resume, provided the containment is recorded, bounded in time, and gated on evidence declared before the window starts.

The autonomy engineering team owns the software and the offline evaluation. It can ship a patch, but a patch takes an illustrative eleven working days from root-cause confirmation to site deployment, which is longer than the decision window and roughly the same length as the evidence window.

The duty safety lead can stop any robot at any time. This is the only authority that is not delegable and not gated. It exists because every other authority in the list has an incentive: throughput, delivery date, or fleet availability.

### 2.4 The baseline and what was wrong with it

The baseline period is 84 days of version 5.2 running on the same 46 robots in the same cluster. That period contains 21 near-miss reports: 5 at the four corners and 16 elsewhere in the zone. Across the whole site the same period contains 214 reports, of which 193 are outside the mezzanine.

The baseline has three documented weaknesses.

First, the map revision changed 9 days into the baseline window: a racking move in the south aisle altered the localisation map but not the corner geometry. Nine of 84 days is 10.7 percent of the exposure and it is confined to a zone that does not contain the corners, so its effect on the corner rate should be nil; the check in Section 12 confirms it is small.

Second, the near-miss form itself changed 30 days before the incident window: the old form had four severity fields, the new form has six, and the reporting threshold ("would have contacted a person within 0.5 m") is stated more precisely in the new form. This is the one bias that cannot be corrected from the data, because the old reports were never re-recorded.

Third, exposure was not instrumented the same way in both periods: the commanded-speed field was added in version 5.3, so there is no before-and-after comparison of actual corner entry speeds. The mechanism argument therefore has to come from geometry, which is a weaker but more durable form of evidence.

Two of those three weaknesses push the comparison in the same direction: they make the baseline look safer than it was, which enlarges the apparent rise. The case does not paper over that. Sections 6 and 12 price each one.

### 2.5 What happened, in sequence

At 14:12 local time on 2026-10-05, a robot transiting C3 stopped 0.31 m from a pedestrian who had stepped out from behind a pallet stack. No contact occurred. The pedestrian reported it through the site's near-miss channel, which is unusual: the site's reports come predominantly from supervisors, and a pedestrian-initiated report carries no robot data at all until the supervisor reconstructs it.

Over the following 21 days, ten more corner reports and three non-corner reports arrived. Two corner reports describe the same robot and the same pedestrian within four minutes; the reporting analyst merged them, which is why the corner count is 11 and not 12.

The cluster was placed in escort mode on 2026-10-06, one day after the first report, and has been in escort mode ever since. Escort mode means a walking observer precedes or accompanies each robot through the four intersections and holds a stop button. Throughput is maintained, and the mode consumes 18 observer-hours per operating day — one observer per shift — plus a supervisor's daily review.

Escort mode is not neutral as an evidence-generating state, and this is the subtlety that decides the case's method. An escorted corner contains a person whose presence changes pedestrian trajectories, and a stop button in a human hand means the robot never has to detect anything, so the protective-stop recorder logs fewer candidate events. A window spent in escort mode measures a different physical system than the one that will run afterwards.

### 2.6 Scale, timing, and the clock

The mezzanine runs 18 productive hours per day across two 9-hour shifts, 21 operating days in the window. Maintenance, charging, and cleaning occupy the remaining hours, during which robots may reposition but do not carry totes and are excluded from exposure.

Exposure is therefore measured in robot-hours of productive operation, not calendar hours and not missions. This matters because mission duration varies: the post-change period averaged 4.1 minutes per mission, the baseline 4.3 minutes, so a mission-based denominator carries a different exposure per unit and shifts every ratio, as Section 6 shows explicitly.

Event time is the timestamp written by the robot's own event recorder when the protective stop was triggered. Processing time is when the report reached the safety database. The two differed by a median of 9 hours and a maximum of 41 hours, because reports are filed by supervisors at shift end and sometimes at the following handover. Any count taken on processing time at a fixed cut-off will miss the tail; Section 5d quantifies what was lost and why the loss is not random.

### 2.7 What the cluster is worth, and why closing it was rejected

The mezzanine moves an illustrative 12,117 totes per day, at 14.63 missions per robot-hour across 828 robot-hours of daily capacity. At an illustrative EUR 2.10 contribution margin per tote move, the zone is worth about EUR 25,446 per day of gross contribution.

Closing the zone would remove the hazard completely and cost about EUR 25,400 per day, or EUR 584,000 across the 23-day window, before counting customer service effects. That option was considered and rejected on commercial grounds, not safety grounds, and the rejection is recorded in the decision minute so that a later reader cannot claim it was never examined.

This number matters to the case because it sets the scale for the containment cost. A EUR 2,282 cap and a EUR 16,974 escort are both rounding errors against the value of the zone; the containment choice is therefore not a cost-optimisation problem at the site level. It is an evidence problem, and the cost figures are only used to break ties between options that are otherwise defensible.

## 3. The Business Question

The question is not "is version 5.3 safe". That question is unanswerable at this grain and would consume the decision window. The question is: for these four intersections, this population, and the next 23 days, which of three operating postures should be authorised, and what evidence would change the answer before the window closes?

The three postures are:

- **Posture U (uncapped resume).** Remove escort, restore the zone speed policy as written (1.5 m/s in open aisle, no corner-specific limit). Lowest cost, highest exposure.
- **Posture C (capped resume).** Remove escort, impose corner-specific speed caps of 0.75 m/s at C1 and C2 and 0.65 m/s at C3 and C4, enforced in the motion planner and monitored against commanded-speed telemetry, with a 23-day gate at 0.5 corner near-misses per 1,000 corner traversals.
- **Posture E (extend escort).** Keep the cluster in escort mode for 23 more days, change nothing in software, and collect evidence with no change in the hazard.

The proxy that must not be mistaken for the objective: near-miss count. The objective is a low probability of contact with a person over the window. Near-miss count is a leading indicator whose relationship to contact is estimated from a small historical conversion, and whose own recording is sensitive to reporting behaviour and to the presence of escorts.

An expensive false decision looks like choosing Posture E for safety reasons, spending EUR 16,974 in observer labour and a month of the site's trained-observer capacity, and then discovering that the measurement window it bought cannot be compared with the unescorted period it will be followed by. That is not a hypothetical objection: the escort changes the system being measured.

An equally expensive false decision looks like choosing Posture U because the raw ratio "only" doubled, then discovering in week two that the mechanism was both geometry and detection latency, that the cap alone does not fix the latency, and that the cluster has been oscillating between two bad states for a month while every meeting argued about which one.

A third, cheaper-looking false decision is choosing Posture C without instrumenting the enforcement path. Section 11 reports how nearly that one happened: 41 crossings above the cap in the first four days, all in a fallback controller that constructs its own trajectory. A cap that exists in the policy layer and not in every code path is a cap that will be measured as successful in the aggregate and be absent exactly where it matters.

The decision must therefore be paired with a mechanism claim strong enough to be falsified in 23 days. The claim made here is narrow: the cap restores a positive kinematic margin at the observed detection latency, so the residual hazard is dominated by detection, not by braking. If corner near-misses persist at the capped speed, the claim is false and the answer is a software fix, not a slower robot.

## 4. Assumptions & Constraints

### 4.1 Assumption register

| ID | Assumption | Basis | If false |
|---|---|---|---|
| A-01 | Productive exposure is 18 h/day per robot in the cluster | Shift roster, 21 of 21 days rostered | Rates scale inversely with true hours; the ratio is unaffected if both periods share the convention |
| A-02 | Near-miss reporting threshold unchanged across periods | Form redesign 30 days before the window; threshold text compared | If the threshold tightened, the post-period rate is inflated; a filing-rate audit is the check |
| A-03 | Corner traversals are countable from reservation records | Every crossing requires a reservation row | If reservations are lost, the traversal denominator is unknown and the stratified analysis collapses |
| A-04 | Braking deceleration 1.2 m/s² dry, 0.6 m/s² wet | Platform datasheet, deck coating specification | At 0.5 m/s² wet the required cap falls a further 0.04 m/s; the error direction is unsafe |
| A-05 | Command latency mean 0.25 s | Controller log over 21 days | At 0.4 s the 1.5 m/s envelope grows by 0.21 m, worsening an already negative margin |
| A-06 | Pedestrian approach speed 1.4 m/s | Site walkway speed survey | Faster pedestrians lengthen the approach term linearly; at 1.7 m/s the tight-corner cap drops to 0.60 |
| A-07 | Historical near-miss to contact conversion is 3 in 63 | Two years of site incident log | At 1 in 63 the deferral risk is three times smaller and Posture E becomes defensible |
| A-08 | The 11 corner reports are independent events | One duplicate pair merged; no other linkage found | If one pedestrian generated several, the effective sample is smaller and the evidence weaker |
| A-09 | Detector latency does not scale with robot speed | Not measured; assumed constant | If latency grows with speed the cap buys less than Section 6 claims, and the first failing window would show it |
| A-10 | Escorted crossings count in the same denominator | Reservation rows continue in escort mode | If escort suppresses reservations, pre- and post-escort rates are not comparable |

### 4.2 The three assumptions most likely to reverse the design

A-02 is the most dangerous. A tightened reporting threshold produces exactly the pattern observed — a rising count with no change in the physical world — and it is invisible in the event table because the table has no field for "which form version was used". The only defence is a second instrument, the robot's own candidate-event log, which does not depend on the form.

A-09 is the most consequential for the recommendation. The whole case for the cap rests on latency being roughly constant while speed falls, so that both the pre-detection distance and the post-detection braking distance shrink with speed. If a slower robot also detects later in absolute time — plausible, because evidence accumulation may depend on the number of frames in which the person appears, and a slower robot spends more frames near the corner — the benefit is smaller. The cap would still improve the margin, but not by the 1.09 to 1.13 m that Section 6 computes.

A-07 drives the entire risk comparison between postures. The conversion used here (3 contacts among 63 near-misses over two years) is a site aggregate that mixes corner and non-corner events; applying it to a corner-only excess assumes corner near-misses convert at the same rate. Corner conflicts have shorter time-to-contact and higher relative speed, so the true corner conversion is probably higher. Using the aggregate is therefore conservative when justifying containment and optimistic when justifying the cap, and it is labelled illustrative in both directions.

A-04 is the quiet one. Wet-deck deceleration is the parameter nobody measures until winter. A cap derived on 1.2 m/s² is unsafe on a wet deck: at 1.5 m/s the required envelope grows from 3.41 m to 6.10 m, a difference that no amount of rounding can absorb. The case derives the cap on 0.6 m/s² precisely because the 23-day window spans a season change and the site's deck is washed nightly.

### 4.3 Constraints that are choices, and choices that are constraints

The 23-day window is a choice dressed as a constraint. It was selected because the exposure arithmetic in Section 6.6 says 23 days is the shortest window that can pass with one event. A 14-day window would have been cheaper and would have certified nothing, because at 14 days the gate can only pass with zero events, and even then the bound of 0.515 misses the limit.

The prohibition on software change inside the window is a real constraint: the patch cycle is eleven working days after root cause, and root cause is not confirmed. The six-day configuration change that fixed the fallback path in week one was not a software change under this definition — it changed a constraint parameter, not behaviour — and that distinction was contested in the minute and recorded as contested.

The prohibition on closing the zone is commercial, as Section 2.7 prices.

The requirement that the cap be enforced in the motion planner rather than in the fleet policy layer is a design choice justified in Section 8: a policy-layer cap is a message, and messages are ignored by code paths that build their own trajectories.

The requirement that the gate statistic be counted on event time rather than processing time is a choice that cost six days of apparent evidence: at the cut-off, processing-time counting showed 12 corner reports against the event-time count of 11, and the missing report was the one that would have moved the ratio from 8.80 to 10.40.

### 4.4 What would change the decision before the window closes

The board required the manager to write down, before the window opened, the conditions under which the posture would be abandoned early. Four were declared.

- Any corner event with measured clearance below 0.20 m triggers immediate return to escort, regardless of the count.
- Two corner events inside the window trigger closure of the gate as a fail and return to escort.
- Any 24-hour period with more than five crossings above the cap by 0.20 m/s or more triggers a stop-and-inspect of the enforcement path before resumption.
- Any change in the reporting form, the pedestrian traffic pattern (for example a new shift start time), or the map revision invalidates the window and restarts the clock.

The fourth condition is the one that did work in practice. A shift-pattern change in week three moved the pick-station handover 40 minutes earlier, which changed pedestrian density at C1 by an unquantified amount; the window clock was restarted by one day to keep the exposure homogeneous, and the restart is visible in the final denominator.

## 5. Data, Tools & Method

### 5a. Data specification

**Grain and keys.** The primary table has one row per near-miss report, keyed by `report_id`, with foreign keys `robot_id`, `intersection_id`, `software_version`, and `shift_id`. The exposure table has one row per robot-day, keyed by (`robot_id`, `date`), carrying productive hours. The traversal table has one row per corner crossing, keyed by `reservation_id`, carrying the intersection, the entry timestamp, the commanded entry speed where instrumented, and the realised minimum clearance to any tracked person.

**Cut-off.** All analysis uses events with event time at or before 2026-10-26T06:00:00Z. Two corner reports with event times of 2026-10-26T14:37Z and 2026-10-27T09:05Z arrived after the cut and are excluded from the primary count; including them moves the corner ratio from 8.80 to 10.40 and is treated as a sensitivity case in Section 9.

**Population.** The treated population is the 46 robots assigned to the North Mezzanine during the post-change window, with the same 46 robot identifiers present in the baseline window. Robots reassigned or added mid-window are excluded from both periods; three robots were reassigned out on 2026-09-30 and are excluded, which removes 3 of 49 and slightly reduces the baseline exposure.

**Units.** Rates are per 1,000 robot-hours of productive operation, except the corner rate, which is per 1,000 corner traversals. Speeds are m/s; distances are m; costs are EUR at illustrative 2026 site planning rates.

**What is not available at decision time.** The pedestrian's own trajectory is not recorded: the site has no pedestrian tracking, only robot-side perception logs. Root cause of the localisation change is not confirmed. The relationship between commanded and realised speed at a corner exists only for the last 21 days, because the field was added in 5.3; the baseline has no commanded-speed field at all. The filing rate of near-miss reports against candidate events requires a separate export with a five-day turnaround, so it is not available at the moment the gate is designed.

**Event time versus processing time.** Counting on processing time at the cut-off would yield 12 corner reports instead of 11, a 9.1 percent overcount in the other direction than Section 2.6 describes for the baseline, and the discrepancy is not random: both late reports are from the night shift, whose supervisors file at handover rather than at shift end.

### 5b. Tools and stack

The analysis used a notebook stack (Python, pandas, statsmodels for exact Poisson bounds, matplotlib for the speed-clearance plot), a log query service over the robot event store, and the site's incident database export.

The kinematic envelope computation is a 40-line function, not a library call: it solves for the largest entry speed at which the sum of reaction, braking, and pedestrian-approach distance fits inside the measured sight envelope. It is reproduced verbatim in Section 6.4 so a reviewer can re-derive every number without the notebook.

Exact Poisson upper bounds were taken from a statistics library rather than a normal approximation, because at zero and one events the normal approximation is meaningless and the entire gate design rests on those two cases.

The deck braking test used the site's own robot, a chalk line, and a tape measure, at 0.6 m/s on a wetted patch. It is not a substitute for a certified braking test and is labelled as an engineering check, not a validation.

### 5c. Method summary

The method is a four-step attribution followed by a decision.

Step one establishes that the rise is not noise: a crude rate ratio with a confidence interval, plus an exact Poisson tail probability against the baseline expectation at the observed exposure.

Step two localises the rise: the same ratio computed separately for corner and non-corner events, and then the same ratio computed for the rest of the site as a control stratum. If the rise is confined to corners, the mechanism must be corner-specific; if it is uniform, the mechanism is fleet-wide and the corner geometry is a red herring.

Step three tests a mechanism against physics: compute the stopping envelope at the design speed and compare it with the measured sight envelope. If the design speed does not fit, the policy was always incompatible with the geometry, and the incident window merely revealed it.

Step four prices the alternatives in both money and residual risk, then writes the gate that would falsify the mechanism claim inside the window.

The fifth step, which most incident responses omit, is a power calculation on the proposed gate: before promising that 23 days will answer the question, compute the probability that the window passes if the true rate is unchanged, and the probability that it fails if the true rate is halved. A gate that cannot fail is not a gate.

### 5d. Provenance, quality, exclusions, and leakage

Provenance: the event export is a read-only snapshot with a recorded row count (2,914 reports site-wide) and a checksum; the exposure table is derived from the rostering system rather than robot telemetry, deliberately, because a robot that is powered but idle still records hours and would inflate the denominator.

Quality issues found and their direction of bias:

- One duplicate report pair (same robot, same pedestrian, four minutes apart) merged; leaving it in would raise the corner count to 12 and the ratio to 9.60.
- Four reports have no `intersection_id` and were assigned by nearest reservation within 30 seconds; two of those assignments are ambiguous between C2 and C3, which changes the per-corner counts but not the corner total.
- The baseline map revision at day 9 of 84 is not adjusted in the primary analysis; restricting the baseline to the map-matched window lowers the corner ratio from 8.80 to 8.13. The direction is towards safety and the magnitude is small.
- The reporting-form change 30 days before the window is the largest unadjusted bias and cannot be corrected from the data. It is handled as failure mode 1 with a discriminating check.

Leakage, in the sense used for models, does not apply; the analogous defect here is ascertainment leakage: the analyst who built the corner speed policy also selected the four corners for review after seeing where the reports were. The mitigation is procedural — sight envelopes at all four corners were measured from the floor plan and confirmed by tape, and the measurement memo is timestamped 2026-10-08, two days after the reports were first visible. That ordering is honest but weak, and it is recorded as weak rather than defended.

Exclusions: ramp traversals are excluded because the ramp limit is 0.6 m/s and no reports occurred there; charging-bay movements are excluded because no tote is carried; the two post-cut reports are excluded from the primary count; escorted crossings in the post period are included in the denominator but the first escort day is excluded because it began mid-shift.

### 5e. Reproducibility and method-selection ledger

| Artefact | Content | Version and date | Reproducible | Limitation |
|---|---|---|---|---|
| EXP-01 | Event export snapshot | 2026-10-26T06:00Z, 2,914 rows | Yes, checksummed | Processing lag means late rows absent |
| EXP-02 | Robot-day exposure table | roster export 2026-10-27 | Yes | Assumes rostered hours equal productive hours |
| EXP-03 | Reservation and traversal table | traffic service export, 21 + 84 days | Yes | Reservation rows lost for 0.4 percent of crossings |
| GEO-01 | Sight envelope measurements | tape survey 2026-10-08 | Partly | Measured to the column face, not to the pallet stack |
| KIN-01 | Envelope solver | script v1.0, seed-free | Yes | Depends on assumed deceleration values |
| COST-01 | Containment cost model | spreadsheet, planning rates | Yes | Labour rates are planning rates, not payroll |
| REP-01 | Filing-rate audit | data request returned 2026-11-01 | Yes | Only two months of candidate logs retained |

Method-selection notes. A change-point model on the daily count series was rejected: the series is over-dispersed and short — 21 days with zero counts on 11 of them — and a change-point fit would return a posterior wide enough to support any of the three postures. A rate ratio with exact tails is less sophisticated and more decisive at this sample size.

A within-corner case-crossover design was rejected because exposure changed by a factor of four between periods, so a within-corner comparison would confound rate with duration.

A survival model with pedestrian density as a covariate was rejected for lack of pedestrian data; the site cannot count people, only robots, and a covariate that is not measured cannot be adjusted for.

## 6. Analysis Walkthrough

### 6.0 The answer before the working

Three facts decide the case.

First, the crude rise is real but not large: 14 reports against 21 at 4.0 times the exposure, a rate ratio of 2.67 with a 95 percent interval of 1.36 to 5.24.

Second, the rise is not uniform: it is entirely concentrated at the four corners, where the ratio is 8.80 with an interval of 3.06 to 25.33, while the rest of the site shows a ratio of 0.97 with an interval of 0.71 to 1.34 that straddles 1.0 comfortably.

Third, the design speed never fitted the tight corners: at 1.5 m/s the required stopping envelope is 3.41 m against a measured sight envelope of 2.55 m, a shortfall of 0.86 m that predates the software change by eleven months. The software change did not create the hazard; it changed how often the hazard was met.

Those three facts support Posture C, and they also explain why Posture C is a compensating control rather than a cure.

### 6.1 Step trace

| Step | Claim under test | Inputs | Method | Intermediate | Reading | Status |
|---|---|---|---|---|---|---|
| 1 | Reports rose faster than exposure | EXP-01, EXP-02 | Crude rate ratio | 0.805 vs 0.302 per 1,000 h | Ratio 2.67 [1.36, 5.24] | DERIVED |
| 2 | The rise is not a Poisson fluctuation | Step 1 | Exact tail, mu = 5.25 | p = 0.0011 | Reject constant rate | DERIVED |
| 3 | The rise is corner-specific | EXP-03 | Stratified ratio | 1.260 vs 0.143 per 1,000 crossings | Ratio 8.80 [3.06, 25.33] | DERIVED |
| 4 | Non-corner traffic is unchanged | EXP-01 | Cluster non-corner ratio | 0.173 vs 0.230 per 1,000 h | Ratio 0.75 [0.22, 2.57] | DERIVED |
| 5 | The rest of the site did not move | EXP-01, EXP-02 | Control stratum | 0.228 vs 0.234 per 1,000 h | Ratio 0.97 [0.71, 1.34] | DERIVED |
| 6 | The four corners are homogeneous | EXP-03 | Chi-square on counts | 1.17 on 3 df, p = 0.76 | One shared mechanism | DERIVED |
| 7 | Design speed exceeds the geometry | GEO-01, KIN-01 | Envelope at 1.5 m/s | 3.41 m vs 2.80 m and 2.55 m | Shortfall 0.61 to 0.86 m | DERIVED |
| 8 | Wet deck demands a much lower cap | KIN-01 | Solve envelope at a = 0.6 | 0.76 m/s and 0.70 m/s | Caps 0.75 and 0.65 | DERIVED |
| 9 | The cap works by restoring margin at observed latency | EXP-01 detection times | Envelope with lag term | 1.09 m and 1.13 m margins | Mechanistic claim | ILLUSTRATIVE |
| 10 | 23 days is the shortest usable gate | Limit 0.5 per 1,000 | Exact bound inversion | 6,000 and 9,488 crossings | 14.4 and 22.8 days | DERIVED |
| 11 | The gate has power to fail | Poisson power | Binomial tail at candidate rates | 0.049 at the limit, 0.60 at baseline | Gate discriminates | DERIVED |
| 12 | Cap is cheaper than escort | COST-01 | Labour and throughput model | EUR 2,282 vs 16,974 | Ratio 7.4 | DERIVED |

### 6.2 Derivation one — crude rate ratio with exposure denominators

```text
FORMULA:
  r_p = N_p / H_p * 1000                 events per 1,000 robot-hours
  RR  = r_post / r_base
  SE(ln RR) = sqrt(1/N_post + 1/N_base)
  CI  = exp( ln RR -/+ 1.96 * SE(ln RR) )

WHERE:
  N_post = 14 near-miss reports in the post-change window [count]
  N_base = 21 near-miss reports in the baseline window [count]
  H_post = 46 robots * 18 h/day * 21 days = 17,388 robot-hours [h]
  H_base = 46 robots * 18 h/day * 84 days = 69,552 robot-hours [h]

CONSTRAINT:
  Poisson counts; both periods on the same 18 h/day convention; no robot
  contributes to both periods with a different duty cycle; events are
  attributed to event time, not processing time.

NUMERICAL EXAMPLE:
  H_post = 46 * 18 * 21 = 17,388
  H_base = 46 * 18 * 84 = 69,552
  r_post = 14 / 17,388 * 1000 = 0.8052 per 1,000 robot-hours
  r_base = 21 / 69,552 * 1000 = 0.3019 per 1,000 robot-hours
  RR     = 0.8052 / 0.3019 = 2.667
  SE     = sqrt(1/14 + 1/21) = sqrt(0.07143 + 0.04762) = sqrt(0.11905) = 0.3450
  ln RR  = 0.9810
  CI     = exp(0.9810 - 1.96*0.3450) to exp(0.9810 + 1.96*0.3450)
         = exp(0.3048) to exp(1.6572)
         = 1.36 to 5.24

INTERPRETATION:
  The all-report near-miss rate rose by a factor of about 2.7. The interval
  excludes 1.0, so a constant-rate explanation is unlikely at this exposure,
  but the pooled figure mixes two strata that behave very differently, which
  is why derivation two exists.
```

The check that matters: 14/17,388 is one near-miss per 1,242 robot-hours, roughly one per 2.8 days of cluster operation. In the baseline it is one per 3,312 robot-hours, one per 7.4 days. Both feel rare; the difference between them is what the decision is about.

### 6.3 Derivation two — stratified attribution to corners

```text
FORMULA:
  lambda_corner = N_corner / T * 1000         per 1,000 corner traversals
  RR_corner     = lambda_post / lambda_base
  mu_expected   = N_base * (T_post / T_base)  baseline rate carried forward
  p_tail        = P(X >= N_post | mu_expected),  X ~ Poisson(mu_expected)

WHERE:
  T_post = 8,732 corner crossings over 21 days = 415.81 crossings/day
  T_base = 34,928 corner crossings over 84 days = 415.81 crossings/day
  N_corner:post = 11,  N_corner:base = 5
  Non-corner in cluster: 3 post over 17,388 h, 16 base over 69,552 h

CONSTRAINT:
  A crossing counts once per reservation row; a robot that reverses and
  re-crosses counts twice. Crossing volume per day is identical across the
  two periods, which is itself evidence that traffic demand did not change.

NUMERICAL EXAMPLE:
  corner post  = 11 / 8,732 * 1000   = 1.2597 per 1,000 crossings
  corner base  =  5 / 34,928 * 1000  = 0.1432 per 1,000 crossings
  RR_corner    = 1.2597 / 0.1432     = 8.80
  SE           = sqrt(1/11 + 1/5)    = sqrt(0.0909 + 0.2000) = 0.5394
  CI           = exp(2.1748 - 1.0572) to exp(2.1748 + 1.0572)
               = exp(1.1176) to exp(3.2320) = 3.06 to 25.33

  mu_expected  = 5 * (8,732 / 34,928) = 5 * 0.2500 = 1.25
  p_tail       = P(X >= 11 | mu = 1.25) = 9.32e-08

  non-corner post = 3 / 17,388 * 1000  = 0.1725
  non-corner base = 16 / 69,552 * 1000 = 0.2300
  RR_noncorner    = 0.75, CI 0.22 to 2.57

INTERPRETATION:
  The corner stratum carries the excess: 11 of 14 reports at eight times the
  baseline rate, with a tail probability small enough that the mechanism must
  be structural rather than a run of bad luck. The non-corner stratum is flat,
  which rules out a fleet-wide speed increase, a dispatch change, or a
  site-wide pedestrian-density change as the dominant cause.
```

Note the asymmetry in the two intervals: the corner interval is wide (3.06 to 25.33) because five baseline events set the denominator. That width is why the case reports a mechanism instead of a precise fold-change. A reviewer who quotes "eight times" without the interval is over-claiming, and the interview pack asks for the interval explicitly.

### 6.4 Derivation three — the kinematic envelope and the cap

```text
FORMULA:
  d_req(v) = v*tau + v^2/(2a) + v_p*(tau + v/a)
  fit condition: d_req(v) <= D_sight
  v_max    = the root of  d_req(v) = D_sight

WHERE:
  v     commanded entry speed [m/s]
  tau   command latency, mean 0.25 [s]
  a     braking deceleration: 1.2 dry, 0.6 wet [m/s^2]
  v_p   pedestrian approach speed, 1.4 [m/s]
  D_sight  measured sight envelope: 2.80 m at C1/C2, 2.55 m at C3/C4 [m]
  The pedestrian term assumes the pedestrian continues at v_p toward the
  conflict point during reaction and braking; it is the pessimistic case.

CONSTRAINT:
  Valid for one pedestrian approaching on the crossing path at constant
  speed, with line of sight established at the envelope distance. It says
  nothing about a pedestrian who stops, reverses, or emerges after the robot
  has committed to the crossing.

NUMERICAL EXAMPLE:
  v = 1.5, a = 1.2, D = 2.80:
    reaction   = 1.5 * 0.25            = 0.3750 m
    braking    = 1.5^2 / (2*1.2)       = 0.9375 m
    pedestrian = 1.4 * (0.25 + 1.25)   = 2.1000 m
    d_req      = 3.4125 m  >  2.80 m   -> shortfall 0.6125 m

  v = 1.5, a = 1.2, D = 2.55: d_req = 3.4125 m -> shortfall 0.8625 m

  v = 0.75, a = 0.6 (wet), D = 2.80:
    reaction   = 0.1875 m
    braking    = 0.5625 / 1.2          = 0.4688 m
    pedestrian = 1.4 * (0.25 + 1.25)   = 2.1000 m
    d_req      = 2.7563 m  <= 2.80 m   -> margin 0.0437 m

  v_max at D = 2.80, wet: 0.8333 v^2 + 2.5833 v - 2.45 = 0
    discriminant = 6.6736 + 8.1666 = 14.8402, sqrt = 3.8523
    v_max = (-2.5833 + 3.8523) / 1.6667 = 0.7614 m/s

  v_max at D = 2.55, wet: 0.8333 v^2 + 2.5833 v - 2.20 = 0
    discriminant = 6.6736 + 7.3333 = 14.0069, sqrt = 3.7426
    v_max = (-2.5833 + 3.7426) / 1.6667 = 0.6956 m/s

  v_max at D = 2.55, dry: 0.4167 v^2 + 1.4167 v - 2.20 = 0
    v_max = 1.158 m/s     (so even dry, 1.5 m/s never fitted)

INTERPRETATION:
  The zone speed policy of 1.5 m/s was never compatible with either corner
  geometry, on any surface. On a wet deck the largest safe entry speed is
  0.76 m/s at the wide corners and 0.70 m/s at the tight ones, so the caps
  are 0.75 and 0.65 after rounding down. The 0.25 m narrowing at C3 and C4 is
  worth about 0.06 m/s, which is why a single site-wide cap of 0.75 would
  have been wrong at exactly the corners that produced most of the reports.
```

The most useful number in that block is the 0.0437 m margin at C1/C2. It is positive but tiny: the cap is not generous, it sits at the edge of the model, and any unmodelled term consumes it. That is why the wide-corner cap is rounded to 0.75 and not 0.76, and why the tight-corner cap is 0.65 and not 0.69 — the 0.69 solution leaves a 0.003 m margin, well inside the deck test's 0.04 m optimism.

### 6.5 Derivation four — detection latency, and the boundary where the cap stops helping

```text
FORMULA:
  d_avail(t_det) = D_sight - v * t_det
  fit:  v^2/(2a) + v*tau  <=  d_avail(t_det)
  t_det_max(v)   = ( D_sight - [v^2/(2a) + v*tau] ) / v

WHERE:
  t_det  time from envelope entry to the moment the detector output is
         accepted by the planner [s]; median 1.4 s across the 11 corner
         events, reconstructed from last-detection timestamps
  d_avail  distance remaining for braking once detection is acted on [m]
  tau  0.25 s command latency

CONSTRAINT:
  The formula presupposes the hazard is visible from envelope entry. If the
  pedestrian is behind the pallet stack, visibility begins later and t_det is
  not a property of the software at all — it is a property of the layout.

NUMERICAL EXAMPLE:
  v = 1.5, D = 2.55, t_det = 1.4:
    d_avail   = 2.55 - 2.10 = 0.45 m
    need      = 0.9375 + 0.375 = 1.3125 m
    shortfall = 0.8625 m  -> contact geometrically unavoidable

  v = 1.5, D = 2.80, t_det = 1.4:
    d_avail = 0.70 m vs 1.3125 m -> shortfall 0.6125 m

  t_det_max at v = 1.5, D = 2.80: (2.80 - 1.3125)/1.5 = 0.9917 s
  t_det_max at v = 1.5, D = 2.55: (2.55 - 1.3125)/1.5 = 0.8250 s

  At the caps, holding t_det at 1.4 s:
    v = 0.75, D = 2.80, a = 0.6: d_avail = 1.750, need = 0.6563 -> margin 1.0937
    v = 0.65, D = 2.55, a = 0.6: d_avail = 1.640, need = 0.5146 -> margin 1.1254

  Latency the caps tolerate:
    v = 0.75, D = 2.80, wet: (2.80 - 0.6563)/0.75 = 2.858 s
    v = 0.65, D = 2.55, wet: (2.55 - 0.5146)/0.65 = 3.131 s

INTERPRETATION:
  At 1.5 m/s the system had to detect and act within 0.99 s at the wide
  corners and 0.83 s at the tight ones; it achieved about 1.4 s. The cap works
  not because the robot brakes better but because both the distance consumed
  before detection and the distance needed after detection shrink with speed,
  turning an 0.86 m shortfall into a 1.13 m margin and raising the tolerable
  latency from 0.83 s to 3.13 s.
```

**Boundary counterexample.** Take the limit v → 0. The formula gives d_req → v_p · τ = 1.4 × 0.25 = 0.35 m, and the pedestrian term never vanishes. A robot that is barely moving still needs 0.35 m of clearance because the pedestrian keeps walking during the robot's command latency. At a tight corner with a pallet stack 0.30 m from the crossing path, no speed cap of any size produces a positive margin: the model's answer is that the geometry must change. That is the honest boundary, and it is the reason the structural fix at C3/C4 is moving the buffer, not lowering the speed further.

A second boundary: if t_det itself grows as speed falls — a real possibility, because evidence accumulation may need several frames and a slower robot spends longer inside the sensor's useful range but at a lower relative approach rate — then the cap's benefit is smaller than the table shows. The case does not model that coupling and says so. It is the first thing a failing window would reveal.

### 6.6 Derivation five — how much exposure the gate actually needs

```text
FORMULA (exact Poisson upper bound from N observed events):
  lambda_upper = chi2_{0.95, 2(N+1)} / (2T)      events per crossing
  tabulated factors: N=0 -> 3.000, N=1 -> 4.744, N=2 -> 6.296
  required exposure to certify lambda <= L with N events:
  T_req = factor(N) / L

WHERE:
  L = 0.5 per 1,000 crossings = 0.0005 per crossing [target limit]
  T in crossings [count]; days = T / 415.81
  one-sided 95 percent confidence

CONSTRAINT:
  Exact, not normal-approximate; assumes a constant rate inside the window,
  which a cap tuned mid-window would violate. It certifies a ceiling only.

NUMERICAL EXAMPLE:
  N = 0: T_req = 3.000 / 0.0005 = 6,000  -> 6,000/415.81  = 14.43 days
  N = 1: T_req = 4.744 / 0.0005 = 9,488  -> 9,488/415.81  = 22.82 days
  N = 2: T_req = 6.296 / 0.0005 = 12,592 -> 12,592/415.81 = 30.28 days

  Check a 14-day window (5,821 crossings):
    with 0 events, upper bound = 3.000/5,821*1000 = 0.515  -> fails 0.5
    with 1 event,  upper bound = 4.744/5,821*1000 = 0.815  -> fails

  Check the chosen 23-day window (9,563 crossings):
    with 1 event, upper bound = 4.744/9,563*1000 = 0.4961 -> passes 0.5
    with 2 events, upper bound = 6.296/9,563*1000 = 0.6584 -> fails

INTERPRETATION:
  A 14-day window cannot pass its own gate even with zero events, because
  0.515 exceeds 0.5. The shortest window that can pass with one event is 23
  days, and that window fails at two. The gate therefore tolerates exactly
  one event — a rule that must be stated to the board in those words, since
  "we will watch it for a while" hides the fact that the tolerance is one.
```

If the limit had been 1.0 per 1,000 instead of 0.5, zero events would certify in 7.2 days and one event in 11.4. Gate design is a choice about how much evidence to buy. The case buys the tighter gate deliberately: 1.0 per 1,000 is still seven times the baseline, and certifying it would have been a certificate for a hazard nobody wanted.

### 6.7 Derivation six — does the gate have power to fail?

```text
FORMULA:
  If the true rate is lambda_t, the number of events in T crossings is
  Poisson(mu) with mu = lambda_t * T. The gate passes if N <= 1.
  power_to_pass(lambda_t) = P(N <= 1) = exp(-mu) * (1 + mu)
  power_to_fail(lambda_t) = 1 - P(N <= 1)

WHERE:
  T = 9,540 crossings realised (9,563 planned)
  candidate true rates: 0.143 (baseline), 0.20, 0.30, 0.40, 0.50 (limit),
  1.26 (observed pre-incident)

CONSTRAINT:
  Poisson; independent events; a single window with no interim decision that
  would stop the clock early. Early stopping at two events is allowed and it
  only increases power to fail.

NUMERICAL EXAMPLE:
  mu(0.143) = 9,540 * 0.000143 = 1.366  -> P(pass) = 0.604
  mu(0.200) = 9,540 * 0.0002   = 1.908  -> P(pass) = 0.432
  mu(0.300) = 9,540 * 0.0003   = 2.862  -> P(pass) = 0.221
  mu(0.400) = 9,540 * 0.0004   = 3.816  -> P(pass) = 0.106
  mu(0.500) = 9,540 * 0.0005   = 4.770  -> P(pass) = 0.049
  mu(1.260) = 9,540 * 0.00126  = 12.02  -> P(fail) = 0.99992

INTERPRETATION:
  The gate fails about 40 percent of the time even if the true rate is the
  baseline rate, so a failure is not proof that the cap is ineffective. It
  fails 95 percent of the time if the rate sits at the limit, and essentially
  always if the hazard did not improve. That is the honest description of a
  one-event gate: strong evidence against "nothing changed", weak evidence
  for any particular lower rate.
```

This block is the one an experienced reviewer asks for first, and it is the one most incident gates omit. A gate with 60 percent pass probability under the null is a coarse instrument; the alternative, a 58-day window that could distinguish 0.2 from 0.3, was priced and rejected because 58 days of containment would have cost more than the residual risk it removed.

### 6.8 Derivation seven — control stratum and per-corner homogeneity

```text
FORMULA (control stratum):
  RR_elsewhere = (N_e,post / H_e,post) / (N_e,base / H_e,base)
FORMULA (homogeneity across the four corners):
  chi2 = sum over corners of (O_c - E_c)^2 / E_c,  E_c = N_total * T_c / T_total

WHERE:
  Elsewhere: 546 robots, 47 post events in 546*18*21 = 206,388 h;
             193 base events in 546*18*84 = 825,552 h
  Per-corner observed counts: C1 2, C2 3, C3 4, C4 2, total 11
  Per-corner crossings: 2,180 / 2,310 / 2,060 / 2,182

CONSTRAINT:
  The control stratum excludes the mezzanine entirely, so it shares none of
  the corner exposure. It does share the software version, which is the point:
  if the release were globally worse, this stratum would move.

NUMERICAL EXAMPLE:
  r_e,post  = 47 / 206,388 * 1000  = 0.2277 per 1,000 h
  r_e,base  = 193 / 825,552 * 1000 = 0.2338 per 1,000 h
  RR        = 0.974, SE = sqrt(1/47 + 1/193) = 0.1626
  CI        = exp(-0.0264 - 0.3187) to exp(-0.0264 + 0.3187)
            = exp(-0.3451) to exp(0.2923) = 0.708 to 1.339

  E_c: 11*2180/8732 = 2.746; 11*2310/8732 = 2.910;
       11*2060/8732 = 2.595; 11*2182/8732 = 2.749
  Contributions: (2-2.746)^2/2.746 = 0.2026
                 (3-2.910)^2/2.910 = 0.0028
                 (4-2.595)^2/2.595 = 0.7603
                 (2-2.749)^2/2.749 = 0.2041
  chi2 = 1.170 on 3 df -> p = 0.76

INTERPRETATION:
  The rest of the site did not move (0.97, interval 0.71 to 1.34), which
  removes the entire release as an explanation and points at something the
  mezzanine has and the other zones do not: the corners, and the pallet
  buffer installed in front of them. The four corners are statistically
  indistinguishable from one another, so a single shared mechanism is
  adequate and no per-corner diagnosis is needed.
```

The homogeneity result is more useful than it looks. If C3 alone had carried the excess, the story would have been the buffer and nothing else, and the fix would have been a layout change. Because all four corners rose by roughly the same factor despite different geometries, the story has to include something common to the cluster — the release — acting on a hazard that was already present everywhere.

| Corner | Envelope | Post rate | Base rate | Ratio | Baseline expectation | Exact tail |
|---|---|---|---|---|---|---|
| C1 | 2.80 m | 0.917 per 1,000 | 0.115 | 8.0 | 0.250 | 0.0265 |
| C2 | 2.80 m | 1.299 per 1,000 | 0.108 | 12.0 | 0.250 | 0.0022 |
| C3 | 2.55 m | 1.942 per 1,000 | 0.243 | 8.0 | 0.500 | 0.0018 |
| C4 | 2.55 m | 0.917 per 1,000 | 0.115 | 8.0 | 0.250 | 0.0265 |

Three of the four corners show the same ratio to one decimal place, which is a coincidence worth resisting: the per-corner counts are 2, 3, 4 and 2, so the "same ratio" for C1, C3 and C4 is an artefact of small integers dividing by different baselines. The only defensible per-corner statement is that every corner's tail probability is below 0.03 on its own, and that no single corner can be singled out without inflating the type-one error fourfold by selecting the largest. A Bonferroni correction across the four corners takes the smallest tail to 4 × 1.8 × 10⁻³ = 7.0 × 10⁻³, which is still well below any conventional threshold and is the number to quote if someone insists on a per-corner claim.

C3's rate is twice the others, which is what the tighter geometry predicts: at 2.55 m the envelope shortfall is 0.86 m rather than 0.61 m, and the corner also carries the pallet stack that the first pedestrian stepped out from behind. But C3's baseline rate is also twice the others (0.243 against about 0.11), so the ratio is not elevated at all. The correct reading is that C3 was always the worst corner and remained proportionally the worst corner; the release multiplied a pre-existing gradient rather than creating a new one.

### 6.9 Derivation eight — the cost of each posture

```text
FORMULA:
  extra_time_per_crossing = L_approach/v_cap - L_approach/v_zone
  daily_time_cost         = crossings_per_day * extra_time_per_crossing  [robot-s]
  missions_lost           = daily_time_cost/3600 * missions_per_robot_hour
  throughput_cost         = missions_lost * days * margin_per_mission
  cap_total               = throughput_cost + engineering_hours * rate
  escort_cost             = observer_hours_per_day * days * loaded_rate
  residual_probability    = 1 - exp(-T_window * lambda * c_conv)

WHERE:
  L_approach = 14 m covered at zone or capped speed [m]
  v_zone = 1.5, v_cap = 0.75 [m/s]
  crossings_per_day = 415.81; missions_per_robot_hour = 60/4.1 = 14.63
  margin_per_mission = EUR 2.10; engineering = 16 h at EUR 95/h
  observer_hours_per_day = 18 (one per shift); loaded_rate = EUR 41/h
  c_conv = 3/63 = 0.0476 contacts per near-miss [ILLUSTRATIVE]
  T_window = 9,563 crossings

CONSTRAINT:
  Values lost robot-time at the zone's contribution margin, so it is a lower
  bound: a lost tote move may also delay a departure. Residual probability is
  a crude conversion applied to a stratum where the true rate is unknown, and
  is a comparison instrument, not a risk assessment.

NUMERICAL EXAMPLE:
  extra_time = 14/0.75 - 14/1.5 = 18.667 - 9.333 = 9.333 s per crossing
  daily time = 415.81 * 9.333 = 3,880.9 robot-s = 1.078 robot-h/day
             = 0.130 percent of the zone's 828 robot-h/day capacity
  missions lost = 1.078 * 14.63 = 15.78/day -> 362.8 over 23 days
  throughput cost = 362.8 * 2.10 = EUR 762
  cap total = 762 + 1,520 = EUR 2,282
  escort cost = 18 * 23 * 41 = EUR 16,974
  ratio = 16,974 / 2,282 = 7.44; difference EUR 14,692

  residual risk, uncapped at the post rate:
    expected near-misses = 9,563 * 0.0012597 = 12.05
    expected contacts    = 12.05 * 0.0476    = 0.5737
    P(>=1 contact)       = 1 - exp(-0.5737)  = 0.4365
  residual risk, capped if the baseline rate returns:
    expected near-misses = 9,563 * 0.0001432 = 1.369
    expected contacts    = 1.369 * 0.0476    = 0.0652
    P(>=1 contact)       = 1 - exp(-0.0652)  = 0.0631

INTERPRETATION:
  The cap costs about one seventh of the escort and, on its own mechanism
  claim, removes most of the near-term contact risk. The escort's advantage
  is that it does not depend on the mechanism claim being right; its
  disadvantage is that it suppresses the reporting and the protective stops
  that any measurement would use, and it consumes a scarce skill the site has
  for about six weeks.
```

The cost comparison deserves a caution about which number is load-bearing. If the throughput cost were ten times larger, the conclusion would not change; if the escort cost were ten times smaller, it would. The decision is driven by the evidence argument, and the money is a tie-breaker that happens to point the same way.

### 6.10 What the numbers do not show

They do not show that the cap reduces near-misses at this site; that is `not-measured` and is precisely what the window tests.

They do not show that the localisation change caused the excess. The correlational structure is consistent with it, and the mechanism is plausible — pose trust affects when detector output is accepted — but the counterfactual is not observed and no A/B was run. Section 12 keeps a competing explanation alive for exactly that reason.

They do not show that 1.5 m/s was ever safe at C1/C2 in the dry: the dry maximum is 1.26 m/s, so the policy was non-compliant with its own geometry before the buffer was installed. Nobody had computed it, in eleven months, at any site in the group.

They do not show the conversion rate from near-miss to contact at corners specifically. The 3-in-63 figure is a two-year site aggregate dominated by non-corner events.

## 7. The Finding

**The hazard was built into the geometry and the speed policy eleven months before the software change; the software change altered how often the hazard was met.**

Three findings support that sentence.

Finding one, quantitative: the corner near-miss rate rose from 0.143 to 1.260 per 1,000 crossings, a ratio of 8.80 with a 95 percent interval of 3.06 to 25.33, and the exact Poisson tail probability against the carried-forward expectation is 9.3 × 10⁻⁸. The non-corner stratum did not move (0.75, interval 0.22 to 2.57) and neither did the rest of the site (0.97, interval 0.71 to 1.34), which excludes fleet-wide causes.

Finding two, mechanistic: at the design speed of 1.5 m/s, the required stopping envelope is 3.41 m against measured sight envelopes of 2.80 m and 2.55 m. The shortfall is 0.61 m at the wide corners and 0.86 m at the tight ones, dry, before the software change. On a wet deck the caps that fit are 0.76 m/s and 0.70 m/s respectively.

Finding three, operational: the observed detection latency of about 1.4 s exceeds the latency the design speed can tolerate (0.99 s at C1/C2, 0.83 s at C3/C4). Cutting the speed to 0.75 and 0.65 restores margins of 1.09 m and 1.13 m and raises the tolerable latency to 2.86 s and 3.13 s, which makes the cap a coherent compensating control and simultaneously proves that a smaller reduction would not have been.

The finding that most changes behaviour is the third. An operator told "your robots are too fast" learns nothing actionable. An operator told "your detection latency must be under 0.83 seconds at these corners, or your entry speed must be under 0.70" has a specification, and a specification can be tested, procured, or designed against.

A fourth finding is about process rather than physics: the layout change that created the tight corners passed through a fire-egress review and a cost review and never met a kinematics review, because the change process had no step that joins a footprint to a speed limit. That is a review-design defect, and it will recur at every site that changes layouts without changing speed policies.

## 8. System/Solution Design

The authorised design is Posture C with five components, each closing a specific gap identified above.

**Component 1 — corner-specific speed caps enforced in the motion planner.** The cap is a constraint in the trajectory generator, not a message to the fleet policy layer and not a map annotation that a fallback controller might not read. At C1 and C2 the cap is 0.75 m/s; at C3 and C4 it is 0.65 m/s. The cap applies to the commanded entry speed measured 3 m before the conflict point, so that a robot cannot satisfy it by braking inside the crossing.

The reason for planner-level enforcement is in Section 12, mechanism 4: the v5.3 localisation fallback constructs its own trajectory in recovery mode and was observed in the baseline period to ignore fleet-layer speed messages at ramps. A policy-layer cap would have been a suggestion.

**Component 2 — a commanded-speed tail monitor.** Every corner crossing emits the commanded speed at the 3 m line and the realised minimum clearance to any tracked person. The monitor alerts when the 99th percentile of commanded speed at a corner exceeds the cap by more than 0.05 m/s over a rolling 24 hours, or when any single crossing exceeds the cap by 0.20 m/s. The first condition catches a systematic regression; the second catches one robot in fallback mode, which is exactly the failure that occurred.

**Component 3 — a 23-day evidence gate with a pre-declared stopping rule.** The statistic is corner near-misses per 1,000 corner crossings with a one-sided 95 percent exact upper bound, evaluated at day 23 or earlier if two corner events occur. Pass requires the bound at or below 0.5. Fail at two events returns the cluster to escort and escalates to a software fix. Pass does not remove the cap; it authorises a separate review to raise it, which is a different decision with its own evidence requirement.

**Component 4 — a reporting-integrity check running in parallel.** Because the gate statistic is a report count, the gate is only as good as the reporting. The check compares the number of candidate events from the robot's own protective-stop recorder with the number of filed reports, in both periods. If the filed-to-candidate ratio is stable, reporting behaviour is not the explanation. If it moved, the rise is partly an accounting artefact and the gate must be re-specified before it is read.

**Component 5 — a clearance floor as an independent stop rule.** Any corner event with measured clearance below 0.20 m returns the cluster to escort immediately, whatever the count. This exists because the count statistic is coarse and a single very-close event carries information the rate does not.

What the design deliberately does not do: it does not close the mezzanine (EUR 25,400/day, and it removes the evidence), does not change software (patch cycle exceeds the window), and does not keep the escort (cost, and it suppresses the reporting and the candidate stops the gate depends on).

Authority boundaries. The site operations manager may start and stop the cap, may extend the window once by up to seven days if exposure falls short of 9,488 crossings, and may not raise the cap. The duty safety lead may revoke at any time without cause. The engineering team may not deploy behaviour-changing software inside the window without a board decision; parameter changes to existing constraints are permitted with the safety lead's sign-off, which is the door through which the fallback fix entered.

The design's weakest element is Component 4's latency, not its logic. A five-day turnaround for a filing-rate export means the gate was designed before its numerator's reliability was known. The fix adopted after the window was to instrument the ratio continuously so that it is a dashboard, not a data request.

Two smaller design decisions are worth recording because they were argued and could have gone the other way.

The first is the observation point for the cap: 3 m before the conflict point rather than at the conflict point itself. Measuring at the conflict point would allow a robot to enter at 1.5 m/s and brake to 0.65 m/s by the crossing, which satisfies the letter of the constraint and defeats the envelope calculation, because the envelope needs the speed to be low throughout the approach where a pedestrian can appear. Measuring at 3 m is a compromise: closer than 3 m and the robot cannot physically comply from the zone speed; further away and the cap becomes a zone-wide limit by another name.

The second is the alert threshold on the commanded-speed tail: a 0.05 m/s exceedance sustained over 24 hours, or a single 0.20 m/s exceedance. The first threshold is set just above the controller's own tracking error at these speeds, which was measured at 0.03 m/s on straight approaches, so that normal variation does not page anyone. The second threshold is set at the smallest exceedance that would matter physically: 0.20 m/s above the tight-corner cap consumes 0.21 m of the 2.55 m envelope in the reaction term alone, which is a quarter of the entire margin at that speed.

The design also specifies what happens to the gate when the cap is deliberately exceeded for a reason, such as a manual drive-through during maintenance. Those crossings are excluded from the rate and counted separately, and the exclusion list is published with the gate result so that a reader can see how many events were set aside and why. In this window the count was nine crossings, all on 2026-11-04 during a charger relocation, none with a pedestrian within 3 m.

## 9. Evaluation & Validation

### 9.1 Metric definitions

| Element | Specification |
|---|---|
| Numerator | Near-miss reports with event time inside the window, intersection in {C1, C2, C3, C4}, and a tracked person within 1.0 m at the protective stop |
| Denominator | Corner crossings recorded as reservation rows with an entry timestamp inside the window |
| Population | The 46 mezzanine robots, excluding the three reassigned on 2026-09-30 |
| Window | 2026-10-27T06:00Z to 2026-11-19T06:00Z, 23 operating days after one restart |
| Comparator | Same corners, same population, 84 preceding days under version 5.2; and the pre-declared limit of 0.5 per 1,000 |
| Guardrails | Fleet-wide near-miss rate must not rise; median cluster cycle time must not exceed baseline by more than 8 percent; at most two crossings per day above the cap |
| Uncertainty | One-sided 95 percent exact Poisson bound; point estimate reported alongside, never alone |
| Status at decision time | Mechanism `not-measured`; cap effect on rate `not-measured`; gate executed after the decision |

### 9.2 What was validated, and how

The kinematic envelope model was validated three ways. It was checked against the platform datasheet braking curve at 1.0 and 1.5 m/s, agreeing within 4 percent. It was checked against a deck test in which a robot braked from 0.6 m/s on a wetted patch: measured 0.62 m against a modelled 0.58 m, so the model is optimistic by about 0.04 m — the right direction to know about and the wrong direction to be comfortable with. And it was checked for dimensional consistency: every term is metres, and the pedestrian term is a distance because a speed multiplies a time.

The stratified comparison was validated by recomputing on three denominators — robot-hours, missions, and crossings. Robot-hours and missions give ratios of 2.67 and 2.54 for all reports, and 8.80 and 8.39 for corners; they agree to within 5 percent, which is the amount by which mission duration changed. The crossing denominator differs by design because it measures per crossing rather than per hour, and it is the denominator that makes the corner question answerable at all.

The reporting-integrity check was not executed before the decision, because the candidate-event export required a data request with a five-day turnaround. This is the largest validation gap in the case and it is stated as such: the gate's numerator depended on a mechanism the case could not verify in time. The check returned on day 6 of the window, with a filed-to-candidate ratio of 0.34 in the baseline and 0.36 in the window, which attributed about 6 percent of the rise to reporting and left the rest physical.

### 9.3 Power, sensitivity, and the honesty of a one-event gate

| Scenario | Change to input | Effect on the corner ratio or gate | Reading |
|---|---|---|---|
| Include the two post-cut reports | 13 instead of 11 corner events | Ratio 8.80 to 10.40 | Direction unchanged; the cut-off rule is what protects the analysis |
| Restrict baseline to map-matched days | 75 of 84 days | Baseline 0.143 to 0.155; ratio 8.80 to 8.13 | Small, and towards safety |
| Unmerge the duplicate pair | 12 instead of 11 | Ratio 8.80 to 9.60 | Small; merging was correct |
| Pedestrian speed 1.7 instead of 1.4 m/s | Approach term grows | Tight-corner cap 0.65 to 0.60 | Cap is sensitive to this; measure it |
| Wet deceleration 0.5 instead of 0.6 m/s² | Braking term grows | Wide-corner cap 0.75 to 0.70 | Direction unsafe if unmodelled |
| Command latency 0.40 instead of 0.25 s | Reaction term grows | 1.5 m/s envelope grows 0.21 m | Worsens an already negative margin |
| Conversion 1 in 63 instead of 3 in 63 | Contact expectation /3 | Uncapped P(contact) 0.437 to 0.176 | Escort becomes defensible |
| Gate limit 1.0 instead of 0.5 per 1,000 | Doubled limit | Zero events certify in 7.2 days | Would certify a hazard nobody wants |

The power numbers in Section 6.7 are the part of validation that a safety reviewer should demand before approving any window. They say the gate has a 60 percent chance of passing even if nothing improved, a 22 percent chance at a true rate of 0.3, and a 5 percent chance at the limit. That is a gate that can fail, which is the minimum requirement, and it is not a gate that can measure a small improvement, which is why the case does not claim one.

A second honesty point: the decision rule tolerated exactly one event, and that tolerance was stated to the board in advance. Had the board wanted a rule that tolerates zero events at higher confidence, the window would have needed 6,000 crossings at a limit of 0.5 — which is 14.4 days and is cheaper — but the bound with zero events is 0.515 at 14 days, so the arithmetic forbids it. The cheapest zero-tolerance gate is 6,000 crossings at a limit of 0.5 evaluated at exactly 6,000 crossings, which is 14.4 days, and it passes only if the count is zero.

### 9.4 Guardrails, and what they were for

The cycle-time guardrail exists because a speed cap that destroys throughput would be rejected commercially and then removed quietly. The modelled cost is 0.32 seconds per mission (415.81 crossings/day at 9.333 s each, spread over 12,117 missions/day), which is 0.12 percent of a 258-second cycle. The observed increase was 15 seconds, or 5.8 percent — 47 times the modelled cost.

That gap is a finding in itself. The cap did not cause the cycle-time increase; something else did, most plausibly the transition in and out of escort mode, the shift-pattern change in week three, and a higher rate of recovery pauses in the first days after the escort was withdrawn. The guardrail was never close to binding (5.8 against 8 percent), but attributing the increase to the cap would have been wrong, and the arithmetic that shows it is three lines.

The fleet-wide near-miss guardrail was satisfied trivially: 61 site events in the window's exposure against an expected 58.6 at the baseline rate, a ratio of 1.04 that is indistinguishable from no change.

## 10. Action Taken

The decision was taken at 2026-10-26T18:00Z by the site operations manager with the duty safety lead present, under delegated board authority recorded as decision reference `B-2026-104`.

**Decision:** authorise Posture C, the capped resume, for 23 operating days from 2026-10-27T06:00Z, with planner-level caps of 0.75 m/s at C1/C2 and 0.65 m/s at C3/C4, the tail monitor, the clearance floor, the pre-declared gate, and the reporting-integrity check running in parallel.

**Rejected alternatives and recorded reasons.** Posture U was rejected because the corner shortfall of 0.86 m at design speed is a geometric fact independent of the software question, and the 43.7 percent illustrative contact probability over the window was judged unacceptable when a EUR 2,282 control was available. Posture E was rejected on cost and on evidence grounds, not safety grounds: it costs 7.4 times as much, consumes the site's trained-observer capacity for the window, and suppresses the reporting and the candidate stops the gate depends on. The minute records that Posture E would have been chosen if the trained-observer pool had been larger than six weeks, because it is the only option that does not rest on an unproven mechanism.

**Immediate actions, in time order.**

1. 2026-10-26T20:00Z — engineering applied the cap as a planner constraint behind a site configuration flag with a rollback switch.
2. 2026-10-27T05:00Z — the duty safety lead verified the cap by walking a robot through C3 at the 3 m line and reading 0.65 m/s commanded, and by checking the recovery fallback, which did not respect it. The check is in the log and is the reason week one had a defect.
3. 2026-10-27T06:00Z — escort withdrawn from C1 and C2; retained at C3 and C4 for three days as a transition measure, then withdrawn.
4. 2026-10-27T06:00Z — the gate clock started; a non-binding look was scheduled for day 10.
5. 2026-10-31 — the tail monitor reported 41 crossings between 1.42 and 1.49 m/s at the corners, all carrying the fallback controller identifier.
6. 2026-11-02 — the constraint was added to the fallback path as a configuration parameter change, signed off by the duty safety lead under the parameter-change provision; crossings above the cap fell to zero from that day.
7. 2026-11-06 — the reporting-integrity check returned: filing rate 0.34 baseline against 0.36 window.
8. 2026-11-10 — the shift-pattern change moved handovers 40 minutes earlier; the window clock was restarted by one day to keep pedestrian exposure homogeneous.
9. 2026-11-19T06:00Z — window closed and evaluated.

**What was explicitly not done.** No behaviour-changing software patch was commissioned inside the window. No map revision was made. The pallet buffer at C3/C4 was not moved, although the analysis says it is the structural defect, because moving it requires a layout change with a four-week lead time and a fire-egress review. The 1.5 m/s zone policy was not changed for the rest of the site, even though derivation three shows it is non-compliant at any corner with under 4.0 m of sight envelope — that is a fleet-wide finding that the case does not have evidence for, and raising it as an action would have been the classic scope inflation.

## 11. Outcome & Results

The window ran 23 operating days, 2026-10-27 to 2026-11-19, after a one-day restart. A conveyor fault unrelated to the robots reduced realised crossings to 9,540 against the planned 9,563.

**Gate result.** One corner near-miss was recorded, on day 14 at C2: a robot entering at 0.74 m/s stopped 0.62 m from a pedestrian who stepped out from behind a wrapping station that is not on the corner sight line. The realised clearance of 0.62 m is consistent with the model at that speed — 1.09 m of margin at 1.4 s latency, less the effect of late visibility caused by an obstruction the sight survey did not include.

Gate arithmetic at close:

- crossings T = 9,540; events N = 1
- point rate = 1 / 9,540 × 1,000 = 0.1048 per 1,000
- exact one-sided 95 percent upper bound = 4.744 / 9,540 × 1,000 = 0.4973 per 1,000
- limit 0.5. **Pass, by 0.0027 per 1,000**, i.e. by 0.5 percent of the limit.

That margin should be stated plainly: the gate passed on the last decimal place it could resolve. It is not evidence of a comfortable position; it is evidence that the window was sized to pass with exactly one event and no more. Had a second event occurred, the bound would have been 6.296/9,540 × 1,000 = 0.660, a clear fail.

| Measure | Baseline (84 d) | Window (23 d) | Reading |
|---|---|---|---|
| Corner near-misses | 5 | 1 | Rate 0.143 → 0.105 per 1,000 crossings |
| Corner crossings | 34,928 | 9,540 | 415.8 → 414.8 per day |
| Non-corner near-misses, cluster | 16 | 4 | 0.230 → 0.210 per 1,000 robot-hours |
| Site near-misses outside cluster | 193 | 47 | 0.234 → 0.228 per 1,000 robot-hours |
| Median cluster cycle time | 4.30 min | 4.55 min | +5.8 percent, inside the 8 percent guardrail |
| Crossings above cap | not instrumented | 41 of 9,540 | 0.43 percent, all in fallback mode |
| Escort labour | 0 h | 162 h (3 days at C3/C4, 2 observers) | EUR 6,642 |
| Cap engineering and throughput | 0 | 16 h plus lost tote moves | EUR 2,282 |

The 41 crossings above the cap are the finding that mattered most operationally. All occurred while a robot was in localisation recovery, which builds its own trajectory and did not read the new constraint. The monitor caught them on day 4, engineering added the constraint to the fallback path on day 6 as a parameter change, and the crossings stopped. This is the most transferable operational detail in the case: the cap was correct, its enforcement was incomplete, and only the instrumented tail found it.

**What the window did not establish.** The mechanism claim — that the cap reduces near-miss frequency because it restores kinematic margin — was not confirmed. One event in 9,540 crossings cannot distinguish 0.10 from the baseline 0.14 or from 0.30; the upper bound is 0.50. What the window established is narrower and still useful: the capped configuration did not produce anything like the pre-cap rate of 1.26, and the cap was enforceable at the planner level with a 0.43 percent exception rate traceable to one specific, fixable path.

**Follow-up, reported as observation only.** A three-month extension recorded two further corner near-misses in 27,300 crossings under the cap plus the fallback fix, a rate of 0.073 per 1,000 with a one-sided upper bound of 0.22. That figure is not a validated effect: there is no concurrent control, the season changed from autumn to winter, pedestrian traffic shifted with a new shift pattern, and the wrapping station moved in month two.

**The structural fix, costed but not executed inside the window.** Moving the pallet buffer at C3 and C4 to restore a 2.80 m envelope was costed at EUR 3,400 of engineering and relocation plus a fire-egress review, with a four-week lead time. The throughput justification is trivial: raising the tight-corner cap from 0.65 to 0.75 saves 14/0.65 − 14/0.75 = 21.538 − 18.667 = 2.872 s per crossing, which across 202 C3/C4 crossings per day is 580 robot-seconds, or 0.16 robot-hours, about 0.02 percent of the zone's daily capacity. Anyone who justifies the buffer move on throughput is wrong about the arithmetic.

The justification is margin and policy simplicity, and both are real. Restoring the envelope raises the wet-deck margin at the tight corners from the 0.044 m of the wide corners' calculation to a genuine 0.044 m at 0.75 m/s instead of requiring a separate 0.65 m/s cap, which means one cap for the whole zone instead of two, one threshold to monitor instead of two, and one fewer place for a fallback path to diverge. It also removes the 0.35 m latency floor described in Section 6.5's boundary case, which no speed cap can address. The move is therefore recommended on engineering grounds and was scheduled for the following quarter rather than treated as an incident action.

**What the window changed in the unit's other processes.** Two process changes outlasted the incident. The release checklist gained a dependency-closure field, so a change scoped as `localisation-only` now names the modules that consume its output and their owners sign. The near-miss form gained a version field, so an ascertainment change is visible in the extract instead of being discoverable only by reading the procedure document. Neither change required a software release and both were made in the week after the window closed.

## 12. Failure Modes & What Went Wrong

Seven mechanisms could each have produced the observed pattern. They are listed with the check that discriminates between them, because the discriminating check — not a mechanism's plausibility — is what makes the analysis usable.

**Mechanism 1 — reporting-threshold tightening (ascertainment).** The near-miss form changed 30 days before the window, with a more precise definition of the 0.5 m threshold. A tighter definition produces more reports with no physical change. *Discriminating check:* the filed-to-candidate ratio. The robot's protective-stop recorder logs every stop where a person was tracked inside 1.5 m; dividing filed reports by candidate events gives a filing rate independent of the physical hazard. If the rate rose from 0.34 to 0.62, half the observed rise would be accounting. It returned 0.34 against 0.36, so this mechanism explains about 6 percent of the rise, not the rise. *Why it was nearly missed:* the event table has no form-version field, so the change was invisible in the extract and was found by reading the reporting procedure document.

**Mechanism 2 — denominator drift.** If productive hours per robot fell while missions were reallocated to the cluster, a mission-based rate would rise with no change in hazard per unit time. *Discriminating check:* recompute on robot-hours and on missions; if the ratio moves materially between denominators, the population or duty cycle changed. Here the two gave 2.67 and 2.54 for all reports and 8.80 and 8.39 for corners — a 5 percent difference explained entirely by the 4.1 against 4.3 minute mission duration. A drift would have produced a much larger gap, and in the other direction. *Residual risk:* the 5 percent is not zero, and at the corner level it is the difference between 8.80 and 8.39, which matters if anyone quotes the number to two significant figures.

**Mechanism 3 — speed policy non-compliant with geometry (the structural cause).** The design speed never fitted a 2.55 m or 2.80 m sight envelope, even dry. *Discriminating check:* compute the envelope at the design speed and compare with the measured geometry, then check whether realised entry speed matched the design speed. The envelope was decisive: 3.41 m required against 2.55 m. The competing explanation — robots exceeding the policy — was excluded by the commanded-speed distribution, whose 99th percentile in the window was 1.47 m/s against a limit of 1.5. The robots were compliant with a policy that was itself wrong.

**Mechanism 4 — enforcement gaps in fallback paths.** The cap was applied as a planner constraint, but the recovery fallback built its own trajectory. *Discriminating check:* the commanded-speed tail segmented by controller identity. A systematic enforcement failure looks like a bimodal distribution with a second mode at the uncapped speed; a random failure looks like a shifted unimodal distribution. Here 41 crossings formed a clean second mode at 1.42 to 1.49 m/s, all carrying the fallback identifier. The check takes ten minutes and should have been part of the deployment, not the response.

**Mechanism 5 — baseline contamination by a map revision.** Nine baseline days used a different localisation map. If that map made corners safer, the baseline rate is biased low. *Discriminating check:* restrict the baseline to map-matched days and recompute. The corner baseline rises from 0.143 to 0.155 and the ratio falls from 8.80 to 8.13. The direction is conservative, which is reassuring here but not in general — a biased baseline can move either way, and this one moved the safe way by luck of the layout.

**Mechanism 6 — a Poisson run masquerading as a regime change.** Twenty-one days is short; 11 corner events could occur at a constant rate by chance. *Discriminating check:* the exact tail probability against the carried-forward expectation, 9.3 × 10⁻⁸ for 11 against 1.25. A second check is the within-window trend: 6 events in week one, 3 in week two, 2 in week three. A degrading system would trend upward; a stationary elevated rate with a chance cluster would not. The absence of a trend argues against progressive degradation and for a fixed elevated rate, which is exactly what a geometric shortfall produces.

**Mechanism 7 — pedestrian behaviour change.** A new pick-station layout or shift pattern could have increased pedestrian traffic at the corners without any change in the robots. *Discriminating check:* pedestrian density is not instrumented, so the only available proxy is the rate of protective stops triggered by a person anywhere in the zone — 22 in the window against 71 in the baseline, per-hour rates of 1.27 and 1.02, a ratio of 1.24 that is far below the corner ratio of 8.80. A pedestrian-density rise of 24 percent cannot generate an eightfold corner rise. *Residual risk:* the proxy measures people who were seen, so it is biased in the same direction as the outcome, and the check is weaker than it looks.

Two mechanisms were considered and dismissed with reasons rather than assertions: a robot population change (the identifier set is identical across periods, with three units excluded from both), and a map-scale error at the mezzanine (the ramps' 0.6 m/s limit and the tote dock geometry both behave correctly, and a scale error would show as localisation divergence metrics, which stayed flat).

The failure mode that actually hurt the response was none of the seven. It was the five-day latency of the reporting-integrity check, which meant the gate was designed before its numerator's reliability was known. A response process that cannot produce a filing-rate estimate inside 72 hours will always be deciding on an unvalidated count.

A second process failure is worth recording because it is cheap to fix: the release checklist classified the change as `localisation-only` with no dependency closure, so the perception owners were never asked whether a pose change could affect detector acceptance. A one-field addition — "which modules consume this output" — would have caught it.

## 13. Transferable Lesson

The transferable lesson is a conversion rule: **every proposed safety control should be translated into the quantity that makes it work, and that quantity should be computed before the control is chosen.**

"Slow the robots down" became "compute the largest entry speed whose stopping envelope fits the measured sight envelope on a wet deck, at the observed detection latency". That computation produced two numbers instead of one instruction, and the two numbers disagreed with the intuitive answer: 0.75 m/s at the wide corners and 0.65 m/s at the tight ones, not a uniform halving, and not the 0.69 that a careless solve of the tight corner gives.

"Watch it for a couple of weeks" became "23 days is the shortest window that can pass with one event at a limit of 0.5 per 1,000, and 14 days cannot pass even with zero events". Two weeks felt sufficient and was arithmetically incapable of certifying anything.

The second lesson is about which number to trust when they conflict. The pooled rate ratio said 2.67. The corner stratum said 8.80. The geometry said the shortfall was 0.86 m. The geometry was the only one that told anyone what to do, because it is the only one expressed in the units of the fix.

The third lesson is about the cost of the safest-sounding option. The escort was defensible in every meeting and would have consumed the site's trained-observer capacity for six weeks to produce a window in which the hazard was unchanged and the reporting was degraded. The cheaper option was 7.4 times cheaper and, on the mechanism, better. Choosing it required being able to say why the cap should work, which required the envelope computation, which nobody had done in eleven months.

The fourth lesson is about enforcement as part of the design. The cap was correct and 99.6 percent obeyed; the 0.43 percent that was not obeyed was concentrated in one code path and would have been invisible in a monthly average. A control without a tail instrument is a control whose exceptions you will discover after an incident, not before one.

The fifth lesson is a cost-per-unit-of-benefit discipline that survives contact with a finance review. Converted to a common denominator, the cap costs EUR 2,282 to avoid an expected 10.68 corner near-misses over the window, or EUR 213.7 per avoided near-miss; the escort costs EUR 16,974 to avoid the same 10.68, or EUR 1,589.8 each. Neither figure is a price of a life, and neither should be quoted as one — the conversion from near-miss to contact is the load-bearing uncertainty in both. But the comparison is decisive in a narrow sense: any argument that pays 7.4 times more per avoided event has to justify the premium by something the cheaper option cannot do, and here the only thing the escort does better is fail safe when the mechanism claim is wrong.

That premium is worth naming, because it is real. If the cap's mechanism were wrong, the escort would still hold the hazard at the pre-containment level, whereas the cap would let the hazard continue at somewhere between the baseline and the pre-incident rate. The correct way to state the trade is therefore not "cap versus escort" but "cap plus a gate that fails fast versus escort plus a gate that cannot resolve anything". The 23-day window converts the cap's weakness into a bounded exposure, and that conversion is the actual engineering content of the decision.

A portable checklist follows from these. For any containment decision: state the mechanism in physical units; compute the control parameter from the mechanism; compute the exposure the gate needs before you set the window; compute the probability that the gate fails under the null; instrument the enforcement path, not only the policy; and name one check that would show the whole framing is an accounting artefact. If any of the six is missing, the decision is a preference dressed as an analysis.

Two anti-patterns deserve equal billing with the checklist, because both were live in this case and both are cheap to spot in someone else's response. The first is the ratio-as-finding: quoting a fold-change with no interval and no stratum split, which produces a number that cannot size a control and cannot be falsified. The second is the containment-as-substitute: treating a compensating control as a fix, so the layout defect that created the hazard survives the incident review and will meet the next software change in exactly the same way.

## 14. Connections & Cross-References

| Relation | Target | Why it matters here |
|---|---|---|
| Same unit, release gate | [Fleet rollout readiness review](./Fleet_rollout_readiness_review.md) | That case gates a release on pre-registered non-inferiority; this one gates a containment on a rare-event bound. Same discipline, different statistic |
| Same unit, sim-to-real gate | [Validate a navigation policy before field trial](./Validate_a_navigation_policy_before_field_trial.md) | Supplies the exposure-sizing arithmetic that this case borrows and re-derives with a different limit |
| Theory: boundary diagnosis | [Common issues for the part](../Cheat_Sheets/99_35_Common_Issues.md) | Entries on frame confusion, stale samples, and safety-envelope assumptions |
| Module 37 | [Sensor fusion and real-time perception](../37_sensor_fusion_iot_and_realtime_perception/INDEX.md) | Detection latency and timestamp age are perception properties, not control properties |
| Module 38 | [ROS 2, kinematics and control](../38_robotics_ros2_kinematics_and_control/INDEX.md) | Where a planner constraint belongs, and why a policy message is not a constraint |
| Module 40 | [Simulation and sim-to-real](../40_robot_simulation_synthetic_data_and_sim_to_real/INDEX.md) | A simulator with a 1.5 m/s corner policy would have reproduced the shortfall if the layout had been modelled |
| Module 41 | [Safety, fleet management and operations](../41_edge_robotics_safety_fleet_management_and_operations/INDEX.md) | Incident response, delegated authority, and containment ladders |
| Catalog | [Case studies index](./INDEX.md) | Row `PLAN-P07-14` |
| Part overview | [Part 07 index](../INDEX.md) | Position of this case in the module ladder |

Three contrasts are worth stating explicitly. Compared with the fleet release case, this case has a much weaker statistical basis (11 corner events against hundreds of missions) and a much stronger mechanistic basis (a geometric shortfall that is a fact, not an estimate). The release case could afford to be agnostic about mechanism because its sample could measure the effect; this one could not, and had to substitute physics for sample size.

Compared with the sim-to-real case, the exposure arithmetic is the same instrument used for a different purpose: there it sized a trial that could resolve a gate; here it sized a window that could pass with exactly one event, a weaker use that the case acknowledges rather than hides.

Compared with the module-41 material on incident response, this case adds the arithmetic of containment choice. The reason to prefer a cap over an escort is not that it is cheaper, but that the escort destroys the measurement it is supposed to enable — and that argument, not the cost ratio, is what survives scrutiny.

A fourth comparison belongs in the review culture rather than the literature: the module-40 simulator case shows what happens when the simulated world is easier than the real one. This case is the mirror image. The corner geometry was known, measured, and recorded in the floor plan for eleven months; the simulator would have reproduced the shortfall if anyone had imported the buffer into the scene. The failure was not an inability to model, it was an absence of anyone asking the model a kinematic question.

### Interview pack

**Q1. State the decision, its grain, its owner, and its cut-off.**
Answer: Resume autonomy in one cluster of four corners, for one software version, for 23 days; owner is the site operations manager under delegated board authority; cut-off 2026-10-26T06:00Z.
Reasoning: The grain is small because the evidence is thin and the population is heterogeneous; a fleet-wide decision could not be reversed cheaply, and a permanent decision would rest on 11 events.
Follow-up: What would move the grain to fleet-wide? A non-corner or off-cluster stratum that also moved, because that would indicate a mechanism independent of corner geometry.
Red flag: "The decision is whether version 5.3 is safe." That cannot be answered at this grain and would consume the window.
Scoring: 2 points for all four elements; 1 for owner and grain without the cut-off; 0 for a vague "decide whether to keep operating".

**Q2. Compute the crude near-miss rate ratio and its 95 percent interval.**
Answer: 0.805 versus 0.302 per 1,000 robot-hours, ratio 2.667, interval 1.36 to 5.24.
Reasoning: 14/17,388 and 21/69,552; SE of the log ratio is sqrt(1/14 + 1/21) = 0.345; exp(0.981 ± 1.96 × 0.345).
Follow-up: Does the interval exclude 1.0, and what does that license you to say? It excludes 1.0, licensing "a rate increase is likely", not "the rate doubled and a half".
Red flag: dividing 14 by 21 without converting to rates, giving 0.67 and concluding the rate fell.
Scoring: 3 points for both rates and a correct interval; 2 for correct rates and a point ratio; 1 for correct rates only.

**Q3. Why does the corner versus non-corner split change the recommended control?**
Answer: Because the excess is confined to corners (8.80 against 0.75 within the cluster, and 0.97 elsewhere), so the control must be corner-specific; a fleet-wide speed cut would impose cost everywhere to fix four intersections.
Reasoning: A uniform rise points to a fleet-level cause such as dispatch or pedestrian density; a localised rise points to geometry or a path-specific software behaviour.
Follow-up: What would you conclude if all strata had risen by 2.7? That the corners are a red herring and the investigation belongs in dispatch or perception.
Red flag: "The split proves localisation caused it." It narrows the candidate set; it does not identify the cause.
Scoring: 2 points for the inference plus the cost argument; 1 for the inference alone.

**Q4. The corner baseline has 5 events. What does that do to your confidence, and how do you express it?**
Answer: It makes the interval very wide (3.06 to 25.33), so the fold-change is not estimable and must not be quoted as a fact.
Reasoning: With 5 baseline events the relative standard error of the log ratio is about 0.54, so a 95 percent interval spans roughly a factor of eight.
Follow-up: Which conclusion survives that width? The direction and the localisation do; the magnitude does not.
Red flag: Reporting "eight times" as a measured multiplier.
Scoring: 2 points for identifying the width and its consequence; 1 for the width alone.

**Q5. Compute the required stopping envelope at 1.5 m/s on a dry deck and compare it with the geometry.**
Answer: 3.41 m required — 0.375 m reaction, 0.938 m braking, 2.10 m pedestrian approach — against 2.80 m at C1/C2 and 2.55 m at C3/C4, shortfalls of 0.61 m and 0.86 m.
Reasoning: v·τ + v²/(2a) + v_p(τ + v/a) with τ = 0.25, a = 1.2, v_p = 1.4.
Follow-up: What is the largest dry speed that fits 2.55 m? About 1.16 m/s, still far below the 1.5 m/s policy.
Red flag: Computing only the braking term (0.94 m) and declaring a large margin.
Scoring: 3 points for the full envelope and both comparisons; 2 for the envelope with one comparison; 1 for braking-only arithmetic.

**Q6. Derive the cap for the tight corners on a wet deck and explain the rounding.**
Answer: Solve 0.8333v² + 2.5833v − 2.20 = 0 with a = 0.6 and a 2.55 m envelope, giving 0.6956 m/s; round down to 0.65.
Reasoning: The 0.69 solution leaves a 0.003 m margin, which is inside the model's own 0.04 m optimism measured on the deck test, so rounding down is a correction, not decoration.
Follow-up: Why not 0.75 everywhere? At 2.55 m wet, 0.75 requires 2.756 m and fails by 0.206 m.
Red flag: Using dry deceleration because the survey was done in summer.
Scoring: 3 points for the root, the wet assumption, and a justified rounding; 2 for the root with dry braking; 1 for a guessed cap.

**Q7. At what detection latency does the design speed stop being sufficient, and what does that imply?**
Answer: At 1.5 m/s the tolerable latency is 0.99 s at C1/C2 and 0.83 s at C3/C4; the observed latency is about 1.4 s. Capping to 0.75 and 0.65 raises the tolerance to 2.86 s and 3.13 s.
Reasoning: Pre-detection distance is v·t_det; at 1.5 m/s and 1.4 s that is 2.10 m, leaving 0.45 m of a 2.55 m envelope against 1.31 m of need.
Follow-up: If latency rose to 2.0 s under the cap, what happens? At 0.65 m/s the available distance is 2.55 − 1.30 = 1.25 m against 0.51 m of need, so it still fits.
Red flag: Treating latency as independent of speed without stating the assumption.
Scoring: 3 points for both thresholds and the margin interpretation; 2 for the thresholds; 1 for a qualitative answer.

**Q8. How many crossings does the gate need, and why is 14 days not enough?**
Answer: 6,000 crossings (14.4 days) certifies only with zero events; 9,488 (22.8 days) certifies with one. At 14 days the zero-event bound is 0.515, already above the 0.5 limit.
Reasoning: T = factor(N)/L with factors 3.000, 4.744, 6.296 for N = 0, 1, 2 at one-sided 95 percent.
Follow-up: What if the limit were 1.0 per 1,000? Zero events would certify in 7.2 days and one event in 11.4.
Red flag: Sizing the window by calendar convenience and calling the result a gate.
Scoring: 3 points for the inversion and the 14-day contradiction; 2 for one of them; 1 for knowing the rule of three only.

**Q9. Choose between the three postures and defend the choice in two sentences.**
Answer: Choose the capped resume: the design speed is geometrically incompatible with both corner types by 0.61 to 0.86 m, the cap restores a margin above 1.0 m at the observed latency, and it costs about a seventh of the escort while leaving the gate measurable.
Reasoning: The choice is not driven by the rate ratio, which is too imprecise to size anything; it is driven by the geometry, which is precise.
Follow-up: What would make you switch to the escort? A trained-observer pool longer than the window, or a failed reporting-integrity check that made the gate numerator untrustworthy.
Red flag: Choosing the escort "because safety" without pricing the loss of measurement.
Scoring: 3 points for the choice plus mechanism and cost arguments; 2 for the choice with one argument; 1 for a choice with no reasoning.

**Q10. The tail monitor reports a bimodal commanded-speed distribution at one corner. Diagnose.**
Answer: A second mode near the uncapped speed indicates an enforcement gap, most likely a controller path that constructs its own trajectory, such as a recovery or fallback mode.
Reasoning: A planner bug would shift the whole distribution; a bimodal distribution segmented by controller identity means one code path ignores the constraint.
Follow-up: What is the cheapest discriminating check? Segment the speed tail by controller identifier; if the second mode carries a single identifier, the diagnosis is confirmed without a physical test.
Red flag: Retraining or recalibrating perception in response to a control-path defect.
Scoring: 3 points for the mechanism and the segmentation check; 2 for the mechanism; 1 for "investigate the logs".

**Q11. Near-miss reports doubled and the filed-to-candidate ratio also doubled. What is your conclusion?**
Answer: Roughly half the apparent rise is an accounting artefact; the physical rate moved by about the ratio of filing rates, and the gate must be re-specified on candidate events or filing-adjusted counts.
Reasoning: The candidate log is produced by the robot's own protective-stop logic and is insensitive to form design, so filed over candidate isolates reporting behaviour.
Follow-up: Which direction does this bias the original decision? It makes the cap look more necessary than it was, but does not touch the geometric shortfall, which is form-independent.
Red flag: Assuming a higher count means more danger without checking the recording instrument.
Scoring: 3 points for the ratio logic and the re-specification; 2 for the ratio logic; 1 for recognising a possible artefact.

**Q12. The gate passed with an upper bound of 0.497 against a limit of 0.5. What does that certificate say?**
Answer: That rates above 0.5 per 1,000 are unlikely given one event in 9,540 crossings — and nothing about rates below 0.5, because the lower bound sits near zero.
Reasoning: At one event the interval is wide; the pass is a one-sided statement about a ceiling, not a claim that the true rate is 0.10.
Follow-up: What would you need to claim the true rate is below 0.2? About 23,720 crossings with one event, which is 57 days at 416 crossings per day.
Red flag: Reading the point estimate 0.105 as the achieved rate.
Scoring: 3 points for the one-sided reading plus a resizing answer; 2 for the one-sided reading; 1 for "it passed, so it is safe".

**Q13. The gate has a 60 percent chance of passing if the rate never improved. Is that acceptable?**
Answer: It is acceptable only because the alternative postures are also imperfect and because a fail is highly informative: at the pre-incident rate the fail probability is 0.9999. The gate is designed to detect "no improvement", not to measure small improvements.
Reasoning: Power to pass is 0.60 at 0.143, 0.22 at 0.30, 0.049 at the 0.5 limit, and 0.00008 at 1.26. The instrument is asymmetric by construction.
Follow-up: What would a 0.2 versus 0.3 discrimination cost? About 57 days at one event, which is longer than any containment the site could staff.
Red flag: Believing a pass certifies the cap works.
Scoring: 3 points for both the asymmetry and the cost of better power; 2 for the asymmetry; 1 for "the sample is too small" with no numbers.

**Q14. A colleague says "use 24 hours per day, robots are always on". What breaks?**
Answer: Every rate falls by 25 percent (18/24), and any comparison with an external threshold or a per-day cost changes; the ratio between the two internal periods does not change because both share the convention.
Reasoning: The denominator must match the exposure that can produce the event; a charging robot cannot produce a corner conflict with a pedestrian.
Follow-up: Where would 24 hours be right? For a hazard that persists while the robot is idle, such as a battery thermal event.
Red flag: Believing a common bias cancels everywhere; it cancels in ratios, not against absolute thresholds.
Scoring: 2 points for the cancellation plus the threshold caveat; 1 for the cancellation alone.

**Q15. Design one check that would falsify the entire framing of this case.**
Answer: Show that the corner rise is fully explained by reporting behaviour — for example a filing rate that rose from 0.34 to 1.0 with a constant candidate-event rate — which would leave no physical excess to explain.
Reasoning: The framing rests on a physical excess; if the excess is an accounting artefact, the geometric analysis remains true but becomes irrelevant to this decision.
Follow-up: What is the second-best falsifier? A non-corner or off-cluster stratum that moved equally, which would relocate the mechanism away from corner geometry.
Red flag: Proposing a check that can only confirm, such as "review the incident reports again".
Scoring: 3 points for a falsifier with a numeric criterion; 2 for a falsifier without one; 1 for a confirmatory check.

**Q16. The board suggests halving fleet speed to 0.75 m/s everywhere. What is your answer?**
Answer: It is wrong in both directions: at the tight corners 0.75 still fails on a wet deck by 0.206 m, and across the other 546 robots it buys nothing measurable while costing throughput and masking the real fix.
Reasoning: The cap must be derived per geometry; a uniform cap is a policy gesture that is unsafe where the geometry is tightest and unnecessary where it is not.
Follow-up: What is the real fix at C3/C4? Move the pallet buffer to restore the 2.80 m envelope, or cut detection latency below 0.83 s at 1.5 m/s; the cap is a compensating control until one lands.
Red flag: Accepting a uniform cap because it is simpler to communicate.
Scoring: 3 points for both directions of error plus the structural fix; 2 for one direction; 1 for rejecting the idea without arithmetic.

**Q17. The cap explains 0.32 s of an observed 15 s cycle-time increase. What do you do with that?**
Answer: Attribute the increase elsewhere and say so; a 47-fold gap between modelled and observed cost means the cap is not the cause, and if the guardrail had been breached the cause would have been sought in escort transitions, the shift change, and recovery pauses.
Reasoning: 415.81 crossings per day at 9.333 s is 1.078 robot-hours, spread over 12,117 missions is 0.32 s per mission, which is 0.12 percent of a 258-second cycle against an observed 5.8 percent.
Follow-up: Why does the distinction matter commercially? Because a cap blamed for a throughput loss gets removed for the wrong reason, taking the safety control with it.
Red flag: Accepting the observed increase as the cap's cost because both happened at the same time.
Scoring: 3 points for the arithmetic and the attribution argument; 2 for the arithmetic; 1 for noticing the discrepancy.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Case | Near-miss incident response at a four-corner cluster, one software version, 23-day containment |
| Decision | Resume capped (chosen) versus resume uncapped versus extend escort; owner: site operations manager under board delegation; cut-off 2026-10-26T06:00Z |
| Crude signal | 0.805 vs 0.302 near-misses per 1,000 robot-hours; RR 2.67, 95 percent CI 1.36 to 5.24; Poisson p = 0.0011 |
| Localised signal | Corners 1.260 vs 0.143 per 1,000 crossings, RR 8.80 [3.06, 25.33]; cluster non-corners 0.75 [0.22, 2.57]; rest of site 0.97 [0.71, 1.34] |
| Mechanism | Design speed 1.5 m/s requires 3.41 m; geometry offers 2.80 m or 2.55 m; shortfall 0.61 to 0.86 m dry, 6.10 m required wet |
| Control derived | 0.75 m/s at wide corners, 0.65 m/s at tight corners (wet braking 0.6 m/s²), enforced in the planner and in fallback paths |
| Gate arithmetic | Limit 0.5 per 1,000; 6,000 crossings for zero events, 9,488 for one; 23 days at 415.8 crossings/day; fails at two events (0.660) |
| Gate power | 0.60 pass at the baseline rate, 0.22 at 0.30, 0.049 at the limit, 0.00008 at the pre-incident rate |
| Result | 1 corner near-miss in 9,540 crossings; upper bound 0.4973 against 0.5; 41 crossings above cap, all in recovery fallback, fixed day 6 |
| Cost | Cap EUR 2,282 versus escort EUR 16,974 over the window (ratio 7.4); illustrative rates |
| Lesson | Convert every proposed control into the quantity that makes it work, compute that quantity, size the evidence window, and instrument the enforcement path before promising either |
