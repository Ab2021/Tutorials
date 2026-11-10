---
case_id: "CASE-2026-0713"
case_title: "Fleet rollout readiness review for an autonomous mobile robot software release"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
source_status: "OPEN"
case_type: "analytical"
domain: "Robot operations — warehouse autonomous mobile robot (AMR) fleet release management"
primary_pattern: "release evidence gate"
secondary_patterns: ["freshness gate", "bounded authority"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "Fleet operations director, with release-board authority"
source_index: "INDEX.md"
review_trigger: "Any change to the release gate thresholds, the telemetry schema used for the metrics, the robot platform major version, the site topology, or the applicable safety-standard scope determination"
---

# Fleet rollout readiness review

## 1. The One-Line Framing

Given 28 days of v4.1 production telemetry from 240 robots and 28 days of a 24-robot v4.2 pilot, decide whether to approve, hold, or stage the v4.2 navigation and task-execution release for the 240-robot fleet, for the operations director who owns physical throughput and the safety case, subject to a pre-registered release gate that forbids a mission-completion regression beyond 0.5 percentage points and any uncertified increase in near-miss frequency, compared with the v4.1 fleet baseline.

The task is a release-authorisation decision, made once per release cycle and again at each staged promotion checkpoint; the decision frequency is low but the blast radius is fleet-wide and partially irreversible.

The primary success condition in plain language: the release may proceed to a wider population only when the measured evidence, at the stated uncertainty, satisfies every gate condition written before the pilot data were examined.

The primary pattern is `release evidence gate`. The two secondary patterns are `freshness gate` — telemetry that arrives after the decision window is inadmissible as evidence — and `bounded authority` — no individual engineer, vendor representative, or site manager can widen a cohort past the authorised population.

**Source declaration.** This case is an illustrative teaching exercise. The operator, the sites, the robots, the release content, the telemetry values, the pilot, the gate thresholds, the staffing arithmetic, the cost figures, and the decision record are constructed. They are not measurements of any named company, product, or deployment, and no claim here establishes that a real v4.2 release behaved in this way. Arithmetic that follows from the stated inputs is marked DERIVED and was independently recomputed; the inputs themselves are ILLUSTRATIVE. Where a real-world standard, method, or measurement practice is named, the reference identifies a scope or a technique to verify, not a compliance conclusion.

The decisive tradeoff is uncomfortable and specific: v4.2 improved nothing that the business asked for, and its most visible regression was cheap to fix while its most expensive regression was invisible at the pilot's sample size. A reviewer who read only the headline completion number would have both over-reacted (rejecting a release whose dominant defect is a tunable parameter) and under-reacted (approving a release whose intervention load would have required two additional staffed teleoperation positions at full scale).

The obvious simpler framing — "does the new version complete more missions?" — misses three parts of the decision. It treats one pooled completion ratio as if it were equally informative in every operating segment, when 0.5 of the 0.8-percentage-point deficit sits in a quarter of the mission population. It treats absence of near-miss reports as evidence of safety, when the pilot's exposure cannot distinguish a 2.5-fold increase from no change. And it treats the pilot cohort as representative of the fleet, when the cohort was one aisle cluster at one site with a healthy-robot selection rule.

A second naive framing — "the pilot sample is too small, so collect more data" — is also wrong as stated, because the required exposure differs by three orders of magnitude across the three gate metrics, and an undirected "collect more data" instruction would have spent a year to answer the cheapest question.

**Interview cue.** A reader should be able to state, in thirty seconds: the unit (one release, one fleet), the output (approve/hold/stage), the beneficiary (operations director), the baseline (v4.1 on the same 240 robots), the binding constraint (pre-registered non-inferiority on completion plus certified near-miss bound), and the evidence basis (illustrative).

**Acceptance check.** Input, output, beneficiary, baseline, constraint, and evidence basis are all identifiable from this section alone, and the framing names a decision that a specific role could actually sign.

## 2. Context & Background

### 2.1 The operating environment

The illustrative setting is a regional distribution operation running 240 autonomous mobile robots that move totes between storage aisles, pick stations, and dispatch lanes. Robots are grouped into aisle clusters; a fleet management service assigns missions, resolves traffic conflicts at intersections and doorways, and dispatches a human teleoperator when a robot cannot resolve a situation by itself.

Three sites make up the fleet. Site A operates 96 robots across four aisle clusters and is the largest and most modern building, with wide cross-aisles and powered doors. Site B operates 84 robots in a older building with narrower doorways, a single charging mezzanine, and shared pedestrian crossings. Site C operates 60 robots in a building with mixed pallet and tote flows and a legacy door controller that the integrator has never fully replaced.

Productive operation is scheduled for 18 hours per day at every site, with 6 hours reserved for charging, cleaning, and maintenance. That schedule is the denominator convention for every rate in this case. Using 24 hours would understate every per-hour hazard and intervention rate by 25 percent, and the error would be invisible because it would apply to both baseline and pilot.

The fleet is not a monolith of identical units. Roughly one in seven robots carries a different lidar model from the rest because of a supply interruption two hardware revisions ago. That heterogeneity matters later: the release under review changes the localisation fallback path, and the fallback path is exactly the code that behaves differently across lidar models.

### 2.2 The workflow the release touches

A mission begins when the fleet management service allocates a transport task to an idle robot. The robot plans a route against the current map and traffic reservation state, drives to the source, docks or waits for a tote transfer, drives to the destination, and reports completion or a failure code. A failed mission is recovered by one of two routes: the robot retries autonomously, or a teleoperator takes control, resolves the situation, and either completes or aborts the mission.

Two classes of event sit outside the mission record and must be joined to it deliberately:

- **Remote interventions.** A teleoperator session opened against a specific robot at a specific time, with a reason code chosen from a controlled vocabulary, a handle time, and an outcome.
- **Near-miss reports.** A human-entered record, filed by a site safety observer or by a teleoperator, describing a physical proximity event that did not produce contact. Reports are filed after the fact, are adjudicated by a safety coordinator, and are therefore available with a lag that varies from hours to several days.

The release under review, designated v4.2, changes three things. It replaces the local planner's clearance model with a learned cost surface. It changes the localisation fallback from "stop and request assistance" to "re-localise against a coarse map and continue at reduced speed". And it changes the door-interaction state machine to poll the door controller rather than wait on its event stream, because the legacy controller at Site C emits duplicate events.

Only the first change was presented to the release board as a performance change. The second change was presented as an availability improvement. The third change was presented as a bug fix. In practice the three interact: continued-at-reduced-speed behaviour after re-localisation is what allows a robot to reach a doorway it would previously have refused, and the polling change is what lets it wait there.

### 2.3 Stakeholders and authority

| Role | Interest | Authority in this decision |
|---|---|---|
| Fleet operations director | Throughput, labour cost, service level to the fulfilment network | Owns the approve/hold/stage decision; signs the rollout record |
| Site safety coordinator | Physical harm prevention; near-miss reporting integrity | Veto over any cohort expansion; owns near-miss adjudication |
| Release board (engineering, QA, security) | Defect escape, rollback feasibility, evidence quality | Recommends; can block on technical grounds |
| Teleoperations manager | Staffing, queue time, operator fatigue | Must confirm capacity for the target population |
| Platform integration owner | Robot OS image, OTA pipeline, config service | Executes the rollout and the rollback |
| Vendor engineering | Feature delivery, defect triage | Advises; holds no release authority |

The separation of roles matters because the vendor's incentive is to ship, the operations director's incentive is to hit a service level, and the safety coordinator's incentive is to avoid an event that ends the programme. A release gate is only meaningful if the party who benefits from shipping cannot reinterpret the gate after seeing the data. Hence the pre-registration requirement in Section 9.

### 2.4 The baseline and its documented weaknesses

The baseline is v4.1 running on all 240 robots for the 28 days preceding the pilot. Over that window the fleet accumulated 120,960 productive robot-hours, attempted 322,500 missions, completed 316,050 of them, recorded 1,512 remote interventions, and logged 42 adjudicated near-miss reports.

The baseline is not a clean control. Its weaknesses, stated now because they constrain every inference later:

1. **No concurrent control.** v4.2 was piloted after v4.1's window, not alongside it. Seasonal demand, a map update, and an aisle re-slotting project all occurred between the two windows. Any difference is therefore a difference-in-time as well as a difference-in-version.
2. **Cohort selection.** The 24 pilot robots were chosen by the site's own engineers as "healthy and centrally located". That selection favours robots with good sensor calibration and favours aisles with high mission density, both of which bias completion upward relative to a random cohort.
3. **Reporting-dependent safety metric.** Near-misses are only counted when someone files a report. A release that changes how often robots stop in doorways may change how often humans notice reportable situations, independent of the physical hazard.
4. **Mixed lidar population.** The baseline pools two sensor configurations. A pooled rate hides a split that becomes important when the fallback path changes.

None of these weaknesses is fatal, and all four are quantifiable to some degree. The mistake to avoid is treating the pilot as a randomised controlled trial and then attributing the whole difference to the release.

### 2.5 Prior attempts and organisational history

Two prior release attempts are relevant context, and both are part of the constructed scenario rather than observed history.

The v4.0 release fourteen months earlier had been pushed to all sites in a single overnight window. Two robots failed mid-update and required manual recovery with a service laptop; the fleet ran mixed versions for nineteen hours because the configuration service would not accept a second update while the first was in flight. No mission-completion metric had been defined at the time, so the incident was reconstructed afterwards from operator recollection. The outcome was a written commitment to staged rollout with a defined rollback window, which is the policy under which this case's decision is made.

The v4.1 release eight months earlier was staged and passed its gate, but the gate had no near-miss condition, because the safety coordinator had not yet been given a veto. The v4.2 gate is the first to include one. That institutional change is the reason the case's hardest question is not "is completion worse" but "what would count as evidence about safety".

### 2.6 Scale, timing, and boundaries

| Context field | Value in this scenario |
|---|---|
| Fleet size | 240 robots: 96 at Site A, 84 at Site B, 60 at Site C |
| Productive schedule | 18 h/day, 7 days/week, all sites |
| Baseline window | 28 consecutive days, immediately before the pilot |
| Pilot window | 28 consecutive days, 24 robots, Site A aisle cluster 2 |
| Mission volume | 11,520 attempts per fleet-day at the baseline rate |
| Release content | Learned clearance cost surface; re-localise-and-continue fallback; door-controller polling |
| Out-of-scope systems | Charging scheduler, warehouse management system, pick-station conveyors, perimeter security |
| Geography | Three buildings in one metropolitan area; single regulatory jurisdiction; single language for reporting |
| Data boundary | Telemetry retained 400 days in the operations data platform; near-miss reports retained in the safety system with restricted access |

The time-zone convention is that every timestamp in this case is recorded in UTC at the robot and in local time at the human record, and joins are performed on UTC after conversion. That convention sounds bureaucratic until a join silently misaligns by one hour across a daylight-saving boundary and assigns night-shift interventions to the wrong shift.

**Interview cue.** Explain who experiences the failure and why the current process persists despite its limitations.

The answer: the people who experience the failure are the teleoperators (who absorb every unresolved robot) and the pickers who share doorway space with the robots; the process persists because a pooled completion ratio is easy to report, because the pilot was cheap, and because the gate's safety condition was new and therefore had no precedent for what counted as passing.

**Acceptance check.** A reader can draw the mission lifecycle, mark the three points where v4.2 changes behaviour, and identify that near-miss evidence arrives on a different clock from mission evidence.

## 3. The Business Question

**Illustrative stakeholder question (constructed, not a quotation):** "Engineering says the new planner is faster and the fallback logic will cut interventions. My numbers say completion is down and my teleoperations lead says her queue is worse. Do I ship v4.2 to all 240 robots on Monday, hold at the pilot, or stage it — and what exactly would have to be true for me to widen it later?"

That question is a paraphrase of a role, not a record of anything a person said.

| Decision contract | Content |
|---|---|
| Question | Approve, hold, or stage the v4.2 release for the 240-robot fleet, given a pre-registered gate and 28 days of pilot evidence |
| Decision owner | Fleet operations director |
| Trigger and deadline | Vendor support for v4.1 ends at the next platform major version; a network-level demand increase begins in nine weeks. Decision needed within five working days of the pilot close |
| Available actions | (a) approve full fleet; (b) hold entirely at v4.1; (c) stage to a defined larger cohort with conditions; (d) approve with a subset of the release (config-only parameter change, no new binary) |
| Value and downside | Value: claimed planner speed and reduced intervention load. Downside: mission-completion loss, teleoperation overload, a contact event that suspends the programme, and an unrecoverable robot |
| Objective | Maximise completed missions per productive robot-hour subject to non-inferiority on completion, a certified near-miss bound, a rollback window under 90 minutes, and staffing feasibility at target scale |
| Hard constraints | No cohort expansion without a passing gate; no unreviewed safety-condition waiver; no mixed-version fleet beyond 24 hours; no rollback path that leaves the map schema incompatible |
| Scope | One release, three sites, 240 robots, 18 h/day schedule, 28-day evaluation windows |
| Out of scope | Pricing or contract changes with the vendor; redesign of the building; changes to the warehouse management system; individual operator performance management |

### 3.1 Why the objective is not the same as the proxy

Completed missions per robot-hour is a throughput proxy. The business cares about totes delivered on time to the pick face and outbound lanes. The proxy can rise while the business outcome falls if the release shifts the mix toward short missions, or if missions complete but deposit totes in the wrong lane and generate downstream rework that the mission record never sees.

The gate therefore includes a mix guardrail: the doorway-mission share must remain within two percentage points of baseline, because a mix shift toward aisle missions would inflate the pooled completion ratio without any real improvement. In the pilot the share was identical at 25.0 percent in both windows, so the guardrail holds and the comparison is not mix-driven. That check is cheap and it removes the single most common way a fleet metric lies.

The second proxy conflict is staffing. A release that reduces completed missions but also reduces interventions is a genuine trade — fewer missions, less human load. A release that reduces completed missions and increases interventions is a pure loss. The pilot showed the second pattern, which is why the decision is not a judgement call about acceptable trade-offs: at full scale it is dominated on both axes.

### 3.2 What an expensive false decision looks like

Three false decisions are available, and they are not symmetric.

**False approval.** The fleet runs v4.2 at 240 robots. Completion falls 0.8 percentage points, which at 11,520 attempts per day is roughly 92 additional failed missions per day, each needing about eleven minutes of manual recovery, and interventions rise by 31 percent, which at scale requires two additional staffed teleoperation positions. If a contact event follows, the programme is suspended and the release is rolled back under pressure, which is the worst possible time to discover that the rollback path crosses a map-schema migration.

**False rejection.** The release is refused outright. The dominant defect — a doorway clearance parameter that regressed 2 percentage points in the doorway segment — is a configuration value, not a code defect. Rejecting the binary throws away the parts of the release that did work and forces a re-pilot cycle that costs another 28 days, while the vendor's v4.1 support window closes.

**False staging.** The release is staged without a defined expansion condition, which in practice becomes a permanent 72-robot island. The fleet then carries two behaviour profiles indefinitely, doubling the regression-test surface and making every later diagnosis ambiguous.

The case is built so that only the third variant — staging with pre-registered expansion conditions and a rollback that does not cross the schema migration — survives contact with the arithmetic.

**Interview cue.** Ask which objective is being optimised, what happens with no intervention, and what an expensive false decision looks like.

With no intervention the fleet stays at v4.1, which is safe and supportable for one more platform cycle but blocks the door-controller fix that Site C has been waiting for. That asymmetry — the release contains a genuine fix bundled with a genuine regression — is the reason "hold" is not the same as "do nothing".

**Acceptance check.** The question can be answered by an action (stage to 72 with conditions) or by a justified decision to gather more evidence (which the gate arithmetic in Section 9 makes specific rather than vague).

## 4. Assumptions & Constraints

### 4.1 Assumption and constraint register

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Telemetry events carry robot-local timestamps with a known offset to UTC and are joined on UTC | Data | Schema inspection plus a join audit against a shift roster | Verified in scenario; audit result constructed | Shift-level rates are misassigned; intervention spikes appear at wrong hours | Re-derive with local-time joins and compare | Schema version change / data platform owner |
| A-02 | A mission attempt is counted once, and autonomous retries are recorded as attempts rather than as new missions | Data | Compare orchestrator mission IDs against robot-side attempt counters for 2,000 missions | Unverified — the two systems disagree by 0.3 percent in a spot check | Completion ratio is inflated or deflated depending on which system is authoritative | Use orchestrator ID as the unit and publish the reconciliation gap | Any orchestrator release / platform owner |
| A-03 | Near-miss reports are filed without regard to the software version running on the robot | Data / human | Compare report rates by version in blinded observer sampling | Violated in scenario — observers knew which robots were on the pilot | Safety comparison is confounded by reporting behaviour | Restrict safety claims to injury-free operation and physical proximity sensor data | Any change to reporting workflow / safety coordinator |
| A-04 | The 24-robot pilot cohort is representative of the 240-robot fleet | Statistical | Compare baseline completion and sensor calibration age across cohort and fleet | Violated in scenario — cohort selected as "healthy and central" | Pilot results are optimistic in level, though the direction of the delta may survive | Stage before generalising; stratify next cohort | Cohort selection rule change / QA lead |
| A-05 | The pilot window and baseline window have comparable demand mix | Business | Compare doorway share, mission type mix, and hourly arrival profile | Verified in scenario — doorway share 25.0 percent in both | Pooled comparison is mix-driven and meaningless | Report by segment with mix standardisation | Demand pattern change / operations analyst |
| A-06 | The v4.2 image is behaviourally identical across the two lidar configurations | Technical | Run the localisation fallback regression suite on both sensor variants in the hardware lab | Unverified — lab suite covers only the majority variant | The pilot's 24 robots do not represent the ~34 robots on the minority sensor | Restrict first cohort to majority-sensor robots or extend the lab suite | Sensor bill-of-materials change / platform owner |
| A-07 | Rollback from v4.2 to v4.1 does not require reverting the map schema | Technical | Attempt a rollback in a hardware-in-the-loop rig with the new map, then with the old map | Unverified — the map migration is one-way in the release notes | Rollback takes a robot out of service for a rebuild rather than six minutes | Freeze map migration; ship v4.2 with dual-schema support | Map service release / integration owner |
| A-08 | Intervention rate scales linearly with fleet size at fixed mission mix | Operational | Compare per-robot intervention rate across sites at baseline | Unverified — Site B runs 1.4× the Site A rate at the same version | Staffing plan under-provisions the hardest site | Provision to the highest site rate, not the fleet mean | Site topology change / teleoperations manager |
| C-01 | No cohort expansion without a passing gate recorded before the data were examined | Governance | Release-board minute from before pilot start | Confirmed in scenario | Gate becomes a negotiation and the evidence loses its meaning | Freeze expansion; re-register the gate | Any gate amendment / release board |
| C-02 | Completion non-inferiority margin is 0.5 percentage points absolute | Business | Service-level agreement between operations and the fulfilment network | Confirmed in scenario (scenario-level, not a legal claim) | The decision flips: −0.8 pp either fails or passes depending on the margin | Re-derive margin from the network's cost of a failed mission | Contract renegotiation / operations director |
| C-03 | Near-miss upper 95 percent bound must not exceed 1.5× the baseline rate | Safety | Safety coordinator's written condition attached to the pilot approval | Confirmed in scenario | An uncertified increase is treated as acceptable, which no one has authority to do | Hold; extend exposure or restrict speed | Safety review / safety coordinator |
| C-04 | Rollback to the prior version must complete within 90 minutes for a cohort of up to 72 robots | Operational | Measured rollback in the rig, then in a staged cohort | Confirmed in scenario | A failed rollout cannot be contained inside one shift | Reduce cohort size or add update lanes | OTA pipeline change / integration owner |
| C-05 | Fleet must never run more than two software versions, and never for more than 24 hours | Operational | Configuration-service capability and enforcement test | Confirmed in scenario | Mixed-version traffic logic produces conflict deadlocks at intersections | Pause rollout; drain cohort to one version | Orchestrator protocol change / platform owner |
| C-06 | Physical safety certification scope is determined by the site's competent authority, not by this review | Regulatory | Written scope determination from the authority | OPEN — not obtained in scenario | Any claim that the release is "certified" would be unsupported | Make no certification claim; treat as engineering evidence only | Any change of robot, task, or layout / safety coordinator |

### 4.2 The three assumptions most likely to reverse the design

**A-07, rollback crossing the map schema.** If reverting the binary also requires reverting the map, then "rollback in 72 minutes" is false for any robot that has already migrated, and the containment story collapses. This single assumption determines whether staging is safe at all. It is listed first among equals because it is the only one whose failure converts a recoverable incident into a fleet-wide service interruption.

**A-04 plus A-06, cohort representativeness.** Both are violated in the scenario, and together they bound how far the pilot's *level* can be generalised. They do not necessarily reverse the *direction* of the completion delta, because the defect mechanism is concentrated in doorway missions and the pilot cohort had the same doorway share as the fleet. Distinguishing "the level is optimistic" from "the delta is wrong" is the analytical crux of Section 6.

**A-02, what counts as a mission attempt.** If retries are counted as new attempts, a release that retries more aggressively appears to attempt more missions and can appear to complete more of them while delivering the same number of totes. If retries are excluded from attempts, a release that fails and retries silently looks better than one that fails and stops. The denominator definition is not a detail; it is the metric.

### 4.3 Constraints that are choices, and choices that are constraints

The distinction is worth stating because release boards routinely blur it. The 0.5-percentage-point margin and the 1.5× near-miss bound are *business and safety choices* made by identifiable roles; they could be different, and Section 6 shows what happens when the margin moves to 1.0 point. The 18-hour schedule, the three-site topology, and the two-version limit are *structural constraints* of this operation; changing them is a different project. Treating a choice as a constraint produces fatalism ("the margin is what it is"), and treating a constraint as a choice produces fantasy ("we will just run three versions for a week").

**Interview cue.** Change one invalidating assumption and explain how the method, architecture, or decision changes.

Take A-07. If rollback cannot cross the map migration, the staged cohort must be capped at robots whose maps are migrated only after the gate passes, which means the update sequence becomes binary-first, map-second, with a hold point between them, and the containment window doubles because two artefacts must be reverted in order.

**Acceptance check.** Each critical assumption has a check and a consequence; each constraint has a source or an explicit scenario assumption; and the register distinguishes the two.
