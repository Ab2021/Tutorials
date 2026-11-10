---
case_id: "CASE-M37-02"
case_title: "Multisensor worker-safety system in a mixed-traffic palletizing cell"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "draft"
content_status: "draft"
case_type: "hybrid"
domain: "Industrial perception — mixed-traffic manufacturing cell"
primary_pattern: "safety-aware-fusion"
secondary_patterns: ["human-review", "controlled-degradation"]
evidence_level: "illustrative"
source_basis: "illustrative"
source_status: "OPEN — method references consulted; no plant data, sensor datasheet, incident record or safety validation supplied"
as_of_date: "2026-11-10"
owner_role: "EHS manager (decision owner); controls engineer (implementer)"
source_index: "INDEX.md"
planned_from: "PLAN-M37-02"
review_trigger: "Change of machine stopping performance, scanner model or mounting, protected-field geometry, tag population, applicable standard edition, or any near-miss in the cell"
---

# Case study: Multisensor worker-safety system in a mixed-traffic palletizing cell

## 1. The One-Line Framing

Given **per-cycle observations from four safety laser scanners, four RGB cameras, eight ultra-wideband anchors and the palletizer's own state feed**, decide **for each 100 ms control cycle whether the cell is NOMINAL, RESTRICTED or STOPPED**, for the **plant's EHS manager and the cell's control system**, to **reduce unprotected worker exposure without destroying line availability**, subject to **the constraint that only a validated safety function may command a protective stop and that the total detection-and-stop latency fixes a minimum protective distance**, compared with **a static safety field with no advisory band and a human gatekeeper who walks workers through the cell**.

The task is a **three-state supervisory control decision** executed at 10 Hz, layered on top of a fixed, independently validated safety function.
The primary reusable pattern is **safety-aware fusion**: heterogeneous sensors are combined under an asymmetric decision rule in which evidence of presence is combined by disjunction and evidence of absence requires every mandatory source to agree.
The secondary patterns are **human-review** (a person adjudicates ambiguous episodes and owns the override) and **controlled-degradation** (a declared reduced state rather than a binary run/halt).

The primary success condition in plain language is that **no worker is inside the protected field while the machine is capable of hazardous motion, and the cell still meets its pallet throughput target**.
Section 9 gives the measurable form of both halves, because the two halves pull in opposite directions and one cannot be optimized alone.

**Source declaration, stated before any number.** This is a constructed teaching case.
The plant, the cell, the sensor models, the recorded rates, the cost figures and the outcomes are **illustrative**; they are internally consistent arithmetic, not measurements of any organization.
Method references (approach-speed safeguard positioning, functional-safety architecture, common-cause beta-factor modelling) are real; the parameter values substituted into them are teaching inputs, and none of the arithmetic here constitutes a compliance determination, a risk assessment or an achieved safety level.
No incident, injury, regulatory finding or vendor benchmark is asserted anywhere in this file.

The decisive tradeoff, which the walkthrough supports arithmetically, is that **the sensors added for reliability made the system less available, and the reliability they were supposed to add was mostly an artifact of assuming independence that the shared platform does not provide**.
Three modalities treated as independent promise that all three fail together on `0.175` of 240 scripted intrusion trials; the fused system failed on **2**, which is `11.4` times the independent prediction and implies a common-cause factor of `beta ≈ 0.064`.
The mechanism is mundane: all three modalities shared one mounting bracket, one 24 V supply, one network segment and one air volume made opaque by the same dust from the same depalletizing operation, so their failure events were correlated by construction rather than by coincidence.
With `beta = 0.064`, the gain from adding a fourth channel is capped at `1/beta ≈ 16` no matter how good that channel is.

The same asymmetry runs through availability in the opposite direction.
Detection failures multiply across sources while nuisance alarms add, so the third channel that bought a real improvement in miss-probability also added a third way to be wrong.
In the first design, where any tag-channel dropout counted as insufficient quality, forklift-blocked ranging held the cell in a reduced state for **19.8 minutes per shift** — `165` hours a year against a declared budget of `83`, and `89,260` currency units of lost throughput against a gross benefit of `82,750`.
The option lost money while looking safer, and the loss was invisible in the vendor's business case because the case counted the sensing cost and never counted the residency.

A second hard limit appeared in the same analysis and was absent from the proposal: the tag channel could not deliver its assumed refresh rate.
Eight anchors, 120 tags and six ranging exchanges of 1.5 ms give `111` tag-frames per second, so per-tag refresh is **1.08 s, not the assumed 0.10 s**.
At a 1.4 m/s walk that is **1.51 m of undetected travel**, larger than the entire advisory band the design was counting on.

The one constant that governs every number in this file is the approach-speed coefficient: at the standards' walking assumption of `K = 1600 mm/s`, **one millisecond of added delay costs 1.6 mm of protective distance, or equivalently 1.6 mm of required warning band**.
Read against that constant, the proposal's central claim inverts.
A camera cannot enter the protective stop path without lengthening it, the field can only be pulled back by cutting mechanical stopping time, and every second of confirmation delay bought for availability is charged back against warning distance at the same 1.6 mm per millisecond.

The obvious simpler framing — "add a camera, get fewer false stops, ship it" — misses that a presence-detection system has two different questions with two different correct logical connectives.
"Is someone there?" is an OR question, and answering it conservatively is cheap.
"Is the area clear?" is an AND question, and answering it conservatively is expensive, because every additional source that must agree becomes a new way to be wrong.
The case turns on noticing that the project's benefit came from the OR side (earlier warning) while its cost came from the AND side (the gate), and that the two were wired together by accident.

## 2. Context & Background

The setting is a mid-size consumer-goods plant, Building B, Cell 4: a robotic palletizer fed by two case conveyors, discharging full pallets onto an outfeed conveyor that a forklift collects from a shared aisle.
The cell occupies a footprint of roughly 4.2 m by 3.6 m, with the palletizer's reach envelope extending 2.1 m from its base.
Operators load slip sheets, clear jams, replace stretch-film rolls, and take quality samples from the pallet stack.
Forklift drivers cross the aisle in front of the cell between 40 and 70 times per hour during peak.
The cell runs two shifts, five days a week, roughly 250 production days per year; the arithmetic below uses 500 shifts and 4,000 operating hours per year.

Work in the cell follows a repeating pattern that makes the safety problem concrete.
A jam stops the conveyor, an operator walks from the aisle into the cell, clears the jam, and walks out.
The palletizer must not resume while the operator is inside; the operator must not enter while the palletizer can move.
Between those two statements sits the entire design problem, because the second statement is about a machine's capability and the first is about a person's position, and neither is directly observable.

The baseline is a static safety arrangement.
Two safety laser scanners guard the two open sides of the cell with a fixed protective field; a light curtain guards the conveyor infeed; an interlocked gate guards the maintenance side.
When the field is interrupted the safety function removes drive power and the palletizer brakes to a stop.
There is no advance warning and no intermediate state: a person is either outside the field and the machine runs, or inside the field and the machine stops.
A gatekeeper — a trained operator assigned to the cell — walks visitors and contractors through, holds the cell for maintenance entry, and resets after a stop.
The gatekeeper covers Cell 4 plus two adjacent cells and is budgeted at 1.0 full-time equivalent across them.

The baseline's documented weakness is not that it fails to stop. It stops reliably, and its stopping behaviour is the one part of the system with actual measured evidence behind it.
The weakness is that it stops too often and too late in the wrong dimension: too often because a person leaning over the field boundary to reach a jam triggers a full stop, and too late in the sense that the only warning a worker receives is the machine already decelerating.
Cell 4 records about 9.4 protective stops per shift, of which the controls log classifies roughly 7 as "nuisance" in the sense that no person was inside the hazardous reach envelope at the moment of the stop, though the log cannot prove that classification — it only records field interruption, not person position.

That classification gap is itself part of the case. The plant has field-interruption events, not person-position truth, so any claim about how many stops were "unnecessary" rests on an inference the existing instrumentation cannot check.
The project therefore has to build an independent reference before it can claim a nuisance-stop reduction, which is why Section 5a spends its space on the evaluation reference rather than on the model.

Three stakeholder groups experience the failure differently.
Production experiences it as availability: 9.4 stops of about 75 seconds each is roughly 11.8 minutes per shift, about 2.5% of a 480-minute shift, plus the recovery tail when the palletizer re-indexes.
EHS experiences it as exposure: each entry is an opportunity for a person and a moving machine to occupy the same space, and the gatekeeper is a single point of human attention across three cells.
Maintenance and contractors experience it as friction, because the interlocked gate must be reset by someone with the key, and the key holder is sometimes on the other side of the building.

The organizational pressure that produced this project is a specific and common one: a near-miss was reported in an adjacent cell, in which an operator's sleeve contacted a moving conveyor during a jam clear.
No injury occurred. The report named "unexpected restart during manual intervention" as the concern, and the site's corrective-action process asked for "a sensor system that knows where people are."
That request, taken literally, is the source of most of the design difficulty, because "knows where people are" is not a specification.

The proposal that reached the EHS manager combined three sensing modalities and promised an integrated answer.
Safety laser scanners would provide the validated presence detection.
Ultra-wideband (UWB) tags worn by workers would provide zone-level identity and proximity, so the system could distinguish a badge-wearing employee from a stray pallet.
Cameras with a person detector would provide earlier warning, covering the advisory band outside the protective field, and would catch the case where someone enters without a tag.
The claim attached to the bundle was that three independent modalities would multiply reliability, "so the protected field can be pulled back and the nuisance stops disappear."

That claim is the object of the case. It is composed of a correct observation (diverse sensing helps), an incorrect inference (diverse sensing multiplies reliability), and an unexamined consequence (a smaller protected field requires a shorter total response time, which a camera pipeline cannot deliver).

Two boundaries constrain the work and are not negotiable.
First, worker tracking for safety and worker tracking for performance management are different purposes with different legal and industrial-relations treatment; the site's works council has approved tag use for zone safety only, with an explicit prohibition on individual dwell-time or productivity reporting.
Second, the site's functional-safety regime requires that any protective stop be produced by a validated safety function with a documented performance level; a learned detector's output cannot be a protective stop, because its failure modes are statistical rather than diagnostic.

Scale, for sizing: 55 worker entries into Cell 4 per day on average, 120 tags in circulation, 4 cameras, 4 scanners, 8 anchors, 10 Hz decision cycle, 250 production days.
Peak conditions are the two hours after each shift start, when changeover work and forklift traffic coincide, and the weeks when a seasonal SKU mix doubles jam frequency.

Prior attempts are worth recording because they explain the organization's skepticism.
A previous trial added a single overhead camera with a person detector that raised an audible alarm when someone entered a drawn region.
It was withdrawn after three weeks: it alarmed on a printed safety poster, on a high-visibility vest draped over a rail, and on the reflection of a passing forklift in a shrink-wrapped pallet, producing between 40 and 60 alarms per shift until operators stopped reacting.
The lesson the plant took from that trial was "cameras do not work here." The lesson the evidence supports is narrower: a detector with no static-region exclusion, no temporal persistence requirement, and no consequence attached to its output will be ignored, which is a design failure rather than a modality failure.

## 3. The Business Question

The stakeholder question is paraphrased from the corrective-action request and the EHS manager's framing; it is not a verbatim quotation from a real meeting.

> "We need the cell to know when someone is inside it, warn them before they get there, and stop only when it matters — and we cannot slow the line down to buy that."

The decision contract is as follows.

| Field | Content |
|---|---|
| Question | Should Cell 4 replace its static protective field plus gatekeeper with a three-modality fused supervisory layer, and if so which escalation boundary and which fail-safe behaviour should govern it? |
| Decision owner | EHS manager, with the site's qualified safety engineer holding a veto on anything that changes the protective function |
| Trigger and deadline | A near-miss corrective action with a 90-day response commitment; the capital cycle closes at the end of the following quarter |
| Available actions | (A) keep the baseline; (B) add advisory-only sensing with no change to the protective function; (C) add advisory sensing and a dynamic protective field driven by the fused layer; (D) add advisory sensing plus a hardware upgrade that shortens stopping time and legitimately shrinks the field |
| Value and downside | Availability recovered from nuisance stops and gatekeeper redeployment, against capital, reduced throughput in a degraded state, and the consequence of a wrong "clear" |
| Objective | Maximize net annual benefit (availability value plus labour redeployment minus operating cost) subject to the protective function remaining validated and the degraded-state residency staying inside a declared budget |
| Hard constraints | No learned model in the protective stop path; no dynamic protective-field switching driven by a non-safety-rated input; no individual performance reporting from tags; total response time consistent with the installed protective distance |
| Scope | Cell 4 palletizer and its two open approach sides, two shifts, 250 days; advisory band extending 1.5 m beyond the protective field |
| Out of scope | Forklift collision avoidance in the shared aisle, conveyor guarding outside the cell, other cells, and any change to the plant's permit-to-work process |

The objective and its proxy disagree in a specific and foreseeable way.
"Fewer nuisance stops" is the proxy; the objective is "less unprotected exposure at acceptable availability."
A system can reduce recorded stops by widening the protected field, which increases exposure-free running but also increases stop frequency, so the proxy moves the wrong way.
Conversely a system can reduce recorded stops by making the stop logic less sensitive, which improves the proxy while destroying the objective.
The guardrail that keeps the proxy honest is therefore **the independently measured rate of person-presence episodes inside the hazardous reach envelope that were not accompanied by a protective or restricted response**, which is a safety-side denominator and must be measured, not inferred from the stop log.

The cost of an expensive false decision has two distinct shapes and the case must weigh both.
A false "clear" is a safety event whose cost is not monetized in this file and must not be traded against availability in any real decision.
A false "occupied" is a stop, whose cost is 75 seconds of cell time plus a recovery tail, and which at sufficient frequency teaches operators to defeat the system.
The decision rule that follows is asymmetric on purpose: the availability arithmetic may choose among designs that are all equally conservative on the safety side, and may never be used to relax the safety side.

A second-order question the owner did not ask but the design must answer is who is accountable when a sensor is degraded.
If the camera is occluded by a stacked pallet for six hours, does the cell run, restrict, or stop?
The baseline has a clear answer for its own failure (the field is interrupted or it is not), and the new system must not become ambiguous at exactly the moment its extra complexity matters.

## 4. Assumptions & Constraints

Each assumption below is stated so that it can be falsified, and each carries the decision that dies if it is wrong.
The distinction between an assumption, a constraint and a design choice matters here because two of the constraints cannot be traded for availability at any price, while several assumptions were traded and one was found false.

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| `A-01` | Mechanical stopping time of the palletizer is 320 ms from the stop command to standstill | Physical | Timed stop measurement with a position encoder, five repeats per speed, worst case retained | Unverified at authoring; treated as 320 ms | The entire protective distance and every field geometry changes | Re-measure and re-derive `S`; do not re-use these numbers | Drive or brake service, load change, speed change / controls engineer |
| `A-02` | Scanner detection time is 62 ms including its own internal processing | Device | Device datasheet response time plus measured network transit | Unverified at authoring; treated as 62 ms | Understates `S`; a slow scanner silently eats the advisory band | Use measured worst case; widen `S` | Scanner firmware or model change / controls engineer |
| `A-03` | A person approaches at no more than 1,600 mm/s | Human factors | Standard walking assumption; plant observation of jogging during changeover | Partly contradicted by observation | Undersized `S`; jogging workers reach the hazard before detection completes | Raise `K` to a documented site value and re-derive; add procedural control | Any recorded running entry / EHS manager |
| `A-04` | All 120 tags in circulation are present and functional during a shift | Operations | Tag issue/return log; battery test at issue | Violated daily in pilot: 9 to 14 tags unaccounted | Any untagged person is invisible to the tag channel | Treat untagged presence as detected by default; never rely on absence of a tag | Battery or issue-process change / shift supervisor |
| `A-05` | UWB per-tag refresh of 100 ms | Device | Anchor count, slot time, tag population, protocol overhead | Violated: computed 1.08 s | The channel cannot localise a moving person within the advisory band | Move tag channel to advisory-only trend use, or raise anchor count | Tag population change / perception engineer |
| `A-06` | The three modalities fail independently | Statistical | Scripted intrusion trials with induced single-channel faults | Violated: implied `beta ≈ 0.064` | Reliability gain is overstated by more than an order of magnitude | Use beta-factor model; add physical separation, separate power, separate network | Any shared-infrastructure change / safety engineer |
| `A-07` | Nuisance stops cost 90 s each including recovery | Business | Time study of 30 recorded stops | Estimated; not a formal time study | Benefit overstated or understated | Re-time with a larger sample | Cycle or staffing change / industrial engineer |
| `A-08` | One FTE of gatekeeper effort is released per year | Business | Budget line and roster review | Estimated | Labour benefit is the second-largest term; removing it changes payback | Retain gatekeeper, count only availability benefit | Roster or scope change / plant manager |
| `A-09` | Cameras can achieve 45 ms inference on the edge device at 10 Hz | Infrastructure | Bench measurement on the target device with the target model and input resolution | Verified on bench only, not in cell | Latency budget breaks; camera path cannot support the advisory band | Reduce resolution or frame rate; move to advisory-only | Model or device change / perception engineer |
| `A-10` | Dust obscures camera and scanner windows at different rates | Physical | Window transmission measurement over a shift | Unverified; assumed independent | If correlated, `beta` rises further and the redundancy claim weakens again | Shared cleaning schedule with a transmission threshold | Cleaning-interval change / maintenance lead |
| `C-01` | No learned component may command a protective stop | Regulatory / safety | Site functional-safety procedure and the applicable machinery safety standard's requirement for a validated safety function | Confirmed by the safety engineer | The design would be non-compliant and uninsurable | Keep the learned layer advisory; the validated safety function retains stop authority | Standard revision / safety engineer |
| `C-02` | Total response time must be consistent with the installed protective distance | Physical | `S = K·T + C` with measured `K`, `C` and each timing component | Confirmed as a constraint | A person can reach the hazard before the machine stops | Lengthen `S`; do not shorten `T` by assumption | Any geometry change / safety engineer |
| `C-03` | Tags may not be used for individual productivity measurement | Industrial relations | Works council agreement, written | Confirmed | Loss of trust; system likely defeated or removed | Aggregate-only zone occupancy with audit-log access control | Agreement revision / HR and EHS |
| `C-04` | Cell availability must stay above 92% of scheduled time | Business | Production reporting definition | Confirmed as a target | The redesign is not worth deploying | Reduce degraded-state residency; shrink scope to one approach side | Target revision / plant manager |
| `C-05` | Capital budget of 120,000 currency units | Financial | Capital committee allocation | Confirmed | Design must shrink; likely drops the camera channel | Defer camera channel to phase two | Budget revision / plant manager |
| `C-06` | Payback must be within 24 months | Financial | Capital policy | Confirmed | Options at 58 months are non-viable regardless of safety benefit | Seek safety-budget funding rather than productivity payback | Policy change / finance partner |

Three assumptions dominate and deserve separate treatment because any one of them reverses the recommendation.

**`A-01` is the assumption that decides whether the redesign is even coherent.** Every millimetre of protective distance is a linear function of stopping time. If the true mechanical stop is not 320 ms but 480 ms, then `S` grows by `1600 × 0.160 = 256 mm`, the advisory band is consumed, and the "pull the field back" idea is dead before any sensor is discussed. The check is cheap — a timed stop with an encoder — and the fact that it was not run before the vendor proposal was written is the single most diagnostic fact in this case.

**`A-06` is the assumption that decides how much the redesign is worth.** The vendor arithmetic treated three modalities as independent. The trials showed they were not. This is not a subtle statistical point; it changes the headline reliability claim by a factor of 11 and it determines whether the third channel is worth its capital cost at all.

**`A-05` is the assumption that decides which channel can be trusted with time.** A tag refresh of 1.08 s is not a degraded version of 0.10 s; it is a different kind of instrument. It reports history, not presence. The design change that follows is categorical — the channel moves out of the per-cycle decision entirely — rather than a tuning adjustment.

Two constraints deserve the same emphasis for the opposite reason: they may not be relaxed under any availability pressure, and both were nearly relaxed during the project.

`C-01` was nearly relaxed by a proposal to let the fused layer "pre-trigger" the safety function when the camera was confident. Pre-triggering a validated stop from a non-validated source is exactly the substitution the constraint forbids, and it would have converted a statistical failure mode into a safety failure mode without any change to the safety function's paperwork. The safety engineer rejected it; the correct alternative, which was adopted, is that the fused layer may request a **restricted** state (reduced speed) and may never request the stop itself.

`C-02` was nearly relaxed by a proposal to shrink the protective field by 400 mm on the argument that the camera would catch people earlier. The arithmetic in Section 6 shows the camera path is 336 ms *slower* than the scanner path, so it would require `537.6 mm` of extra distance rather than saving 400 mm — a swing of `937.6 mm` in the wrong direction. The proposal survived two review meetings because nobody had written the two latency paths side by side.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is one **supervisory decision cycle**: a 100 ms interval with a decision timestamp, the set of observations usable at that timestamp, and the resulting state.
The evaluation adds a second grain, the **intrusion episode**, because a per-cycle accuracy figure is meaningless for a safety decision: 240 cycles of correct "clear" during an episode in which a person walks in still contains the failure that matters.

| Field | Specification |
|---|---|
| Unit of analysis | Decision cycle (100 ms) for operation; intrusion episode for evaluation |
| Volume and window | 4,000 operating hours/year; 10 Hz gives 144 million cycles/year for continuous running, but the evaluation set is episodes, not cycles |
| Observation inputs | Scanner field states and measured distances with capture time; camera detections with capture time, bounding box and confidence; tag ranges and zone estimate with capture time; palletizer state (idle, indexing, stacking, wrapping) with cycle time |
| Labels | Episode label: whether a person was inside the reach envelope during the episode, adjudicated from the reference system, not from the fused system |
| Population and exclusions | All scripted and opportunistic entries into Cell 4 during the trial window; excluded: maintenance work with the gate interlock open (a different control regime), and any episode where the reference system itself was unavailable |
| Join semantics | Observations join to a cycle by capture time within the cycle window, never by arrival order; late observations that arrive after their cycle closed are logged as late and excluded from that cycle's decision |
| Splits | Trial episodes are split by day, not by episode, so that lighting and dust conditions do not leak across the split; the reference labels are fixed before the fused system is run on the holdout days |

The evaluation reference is the hard part, and it is where most of the engineering honesty in this case lives.
There is no ground-truth sensor for "a person is at position p at time t" that is independent of the system under test, so the reference must be constructed and its own error characterised.

The reference used was a **time-stamped overhead video with manual annotation by two independent reviewers**, plus the palletizer's own commanded-position log.
Reviewers marked, for each episode, the first and last frame in which any part of a person's body was inside a painted floor rectangle that matched the reach envelope within a surveyed tolerance of `±40 mm`.
Inter-rater disagreement was resolved by a third reviewer; the disagreement rate was 7 of 240 episodes, concentrated at the boundary where a person's arm crossed the line while the torso stayed outside.

That reference has three known weaknesses, and each is stated rather than hidden.
It is **not independent of the camera channel** in the sense that both use the same optical path, so a dust event that blinds the evaluation camera also blinds the sensed channel; the trials therefore included induced-dust conditions with a secondary reference from the scanner field log.
It is **not real-time**, so it can evaluate decisions but cannot drive them.
It is **not a safety-rated reference** and cannot be used to certify anything; it is an evaluation instrument, and Section 9 uses it only to compare designs.

The intrusion episodes themselves were generated in three ways, and the mix matters for interpreting the miss rate.
**Scripted intrusions (128 episodes)** followed a written protocol: an operator walked a straight line into the reach envelope at a controlled pace, with and without a tag, under four lighting conditions.
**Opportunistic intrusions (112 episodes)** were ordinary jam clears and quality samples captured during the trial window, annotated after the fact.
The scripted set measures the sensor stack against a known geometry; the opportunistic set measures it against real behaviour, which includes stopping mid-step, leaning, and reaching around the field boundary.

The split is 168 training-or-tuning episodes and 72 holdout episodes, split by day.
The holdout days were chosen to include one dusty afternoon and one shift with a failed anchor, because those are the conditions the design claims to survive.

A note on what data does **not** exist, because its absence bounds every claim in this file.
There is no record of how many times a worker was inside the reach envelope and was not detected by the *baseline*, since the baseline does not measure position at all.
There is no injury, near-miss or exposure-incidence data beyond the single reported near-miss that triggered the project.
There is no measured production throughput effect of the restricted state at the cell level, only the modelled derating in Section 9.
Any statement of the form "this would have prevented N incidents" is therefore unavailable, and this case makes none.

### 5b. Tools & Stack

The stack is described as proposed and partially benched, which is the honest state at the decision point.

| Component | Role | Version or configuration | Status at decision |
|---|---|---|---|
| Safety laser scanners (4) | Validated presence detection in two fixed protective fields and two advisory warning fields | Response time 62 ms per datasheet; OSSD pairs wired to the safety controller | Installed in baseline for two of four; two added by this project |
| Safety controller | Executes the validated stop logic; owns the protective stop output | Configured with a documented safety function and proof-test interval | Installed; unchanged by the project |
| Edge inference device | Runs the camera detector; publishes detections to the fused supervisor | 8-core embedded CPU plus a 40 TOPS accelerator; measured 45 ms inference at 640×384 | Benched, not yet in cell |
| Cameras (4) | Advisory person detection and advisory-band coverage | Global shutter, 1280×720, 15 fps capture, hardware-triggered by the scanner sync pulse | Two installed for trial |
| UWB anchors (8) and tags (120) | Zone-level identity and proximity | 6.5 GHz channel, 1.5 ms ranging slot, TDoA | Installed for trial |
| Fused supervisor | Computes the three-state decision at 10 Hz; publishes to the cell PLC | Real-time task on the edge device, fixed 100 ms budget, watchdog reset | Prototype |
| Cell PLC | Accepts the restricted/stop request and reports machine state | Existing; modified to accept a restricted-speed request | Modified |
| Time distribution | Common time base for capture timestamps | PTP over a dedicated switch; measured offset within ±200 µs | Installed |
| Annotation tool | Manual episode labelling for the evaluation reference | Internal; two-reviewer workflow with third-reviewer adjudication | In use |

Two stack choices were contested and are recorded because they shaped the result.

**Triggering the cameras from the scanner sync pulse rather than free-running** was chosen so that a camera frame's capture time is unambiguous relative to the scanner sweep, at the cost of a fixed phase relationship that makes a camera frame unavailable for up to one scanner period after a trigger. This reduced timing uncertainty from about 33 ms to under 2 ms, and it is the reason the camera path's latency is known rather than estimated.

**Running the fused supervisor on the same edge device as the camera detector** was chosen for cost and simplicity and is the direct cause of the shared-failure problem in `A-06`. A detector that saturates the accelerator delays the supervisor, and a supervisor watchdog reset takes the camera channel with it. This is a design choice with a known cost that was accepted before the reliability analysis made the cost visible.

### 5c. Method Summary

The method is a two-layer architecture with a hard boundary between layers, and the boundary is the design.

**Layer 1, the validated safety function**, is unchanged in logic: scanner field interruption removes drive power through the safety controller. It is not a model, it has no learned parameters, and its failure modes are diagnostic (a test pulse fails, a channel disagrees, the controller faults to a safe state).

**Layer 2, the fused supervisor**, consumes observations and emits one of three states. It cannot produce a stop directly; it can request a restricted state and it can *withdraw* its permission for normal running, which the PLC treats as "do not start a new cycle" rather than "stop now." The distinction matters: withdrawal of permission is a latching condition that prevents motion from beginning, while a stop interrupts motion already in progress.

The fused state rule is asymmetric by design.

```text
STATE = STOPPED      if any validated safety channel reports field intrusion
STATE = RESTRICTED   if OCCUPIED_ANY or QUALITY_INSUFFICIENT
STATE = NOMINAL      otherwise

OCCUPIED_ANY          = scanner_advisory OR camera_persistent OR tag_zone_uncertain
QUALITY_INSUFFICIENT  = NOT(scanner_ok) OR NOT(camera_ok)
```

`OCCUPIED_ANY` is a disjunction because the two questions it answers — "might someone be there" — are answered conservatively by OR.
`QUALITY_INSUFFICIENT` contains an `AND`-shaped requirement expressed as the negation of a conjunction: to be permitted NOMINAL, **both** mandatory channels must affirmatively report healthy. This is where the design pays for its extra reliability, and it is where the first version went wrong by including the tag channel in the mandatory set.

The tag channel was removed from `QUALITY_INSUFFICIENT` and from the escalation path after the trial showed its refresh rate and its blockage behaviour. It remains in `OCCUPIED_ANY` with a 3-cycle persistence requirement, which is the correct use of a slow, occasionally-absent channel: it can add evidence of presence, and it cannot subtract evidence of absence.

The candidate comparison follows.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| (A) Baseline static field plus gatekeeper | 9.4 stops/shift observed; 75 s mean recovery | Validated, simple, understood by operators | No warning; high nuisance rate; gatekeeper is a single point of human attention | Rejected as final state but retained as the fallback | Never; this is the safe fallback |
| (B) Advisory-only camera and tag, no field change | Bench latency 45 ms; trial nuisance rate | Lowest risk; no change to the protective function | Captures only the warning benefit, not the field-pullback benefit; payback 31 months | Considered; viable but weak | If the safety side cannot fund a brake upgrade |
| (C) Three-modality fused supervisor with dynamic field | Trial episodes; beta-factor analysis; capacity analysis | Captures both benefits | Fails on tag refresh and common-cause reliability; payback 58 months | Rejected after analysis | Never in this form |
| (D) Two-modality supervisor (scanner plus camera) with a brake upgrade and no dynamic field switching | Trial episodes; measured stop time reduction from 320 to 150 ms | Legitimate field pullback via a physical change; camera stays advisory | Requires brake capital and a re-validated stop time | Chosen | If the brake upgrade cannot achieve 150 ms |

Option D is the recommendation and it wins for a reason that is worth stating plainly: **it buys its field pullback from the one component whose improvement is measurable and whose failure mode is mechanical rather than statistical.** Cutting stopping time from 320 ms to 150 ms reduces `S` by `1600 × 0.170 = 272 mm` with no change to any detector's reliability. That 272 mm is more than half of the 400 mm the dynamic-field proposal wanted, and it comes with no new failure mode at all.

### 5d. Data provenance, quality, exclusions and leakage controls

The provenance table records what exists and what does not, because the case's central claim depends on the difference.

| Source ID | Content | Provenance | Accessibility | Limitation |
|---|---|---|---|---|
| `SRC-01` | Plant cell layout, reach envelope, cycle times | Illustrative; constructed to be internally consistent | Teaching artifact | Not a survey; the 2.1 m reach is a teaching value |
| `SRC-02` | Scanner response time, field geometry | Illustrative device parameters consistent with typical safety laser scanner specifications | Teaching artifact | Not a datasheet quotation; a real design must use the actual datasheet and measured response |
| `SRC-03` | Stop-time measurement, 320 ms baseline and 150 ms after upgrade | Estimated from illustrative brake and drive parameters | Teaching artifact | A real value requires a timed stop per ISO 13855 methodology |
| `SRC-04` | 240-episode trial with 2 fused misses and per-channel misses | Simulated teaching dataset | Teaching artifact | Constructed; the independence violation is designed into it to teach the mechanism |
| `SRC-05` | Availability, stop counts, recovery times | Illustrative plant records | Teaching artifact | Not a production extract; the 9.4 stops/shift figure is a teaching value |
| `SRC-06` | Tag refresh and anchor capacity | Derived from anchor count, slot time and tag population | Teaching artifact | Protocol overhead is modelled, not measured |
| `SRC-07` | Cost and labour rates | Illustrative | Teaching artifact | Currency units are unspecified on purpose; substitution of local rates is expected |

The quality issues that a real project must handle, and how this case handles them:

**Leakage through the evaluation camera.** The reference video and the sensed camera share an optical path, so a dust event degrades both. This is not classical label leakage, but it is the same failure: the reference agrees with the system under test for reasons that do not generalise. Control: the 40 induced-dust episodes were scored against the scanner field log instead, and the two references were compared on the 80 episodes where both were available.

**Leakage through episode-level splitting.** Splitting by episode would put the same lighting condition and the same dust state on both sides of the split. Control: split by day, with the holdout days chosen before tuning.

**Selection bias in the opportunistic episodes.** Jam clears are not a random sample of entries: they cluster at the infeed, where the scanner coverage is best. Control: report scripted and opportunistic miss rates separately, and never pool them into a single headline number.

**Label ambiguity at the boundary.** Seven of 240 episodes were disputed, all at the reach-envelope line. Control: third-reviewer adjudication, and a sensitivity analysis in Section 9 that recomputes the miss rate under the opposite adjudication.

**Missing data that cannot be recovered.** No baseline person-position record exists, so there is no way to compute how many baseline stops were "correct." The number 7-of-9.4 "nuisance" is a modelled classification, not a measurement, and is labelled as such wherever it is used.

### 5e. Reproducibility and method-selection ledger

| Artifact | Location | Version / date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| Case specification | This file | 1.0 / 2026-11-10 | Design, arithmetic, decision record | Public teaching artifact | All parameters illustrative |
| Arithmetic workbook | Inline in Sections 5–9 | 2026-11-10 | Every figure recomputable from stated inputs | Inline | Recompute by hand or script from the formulas given |
| Trial episode set | Not supplied | — | Would be required to reproduce the 2/240 result | Unavailable | No dataset is supplied with this case; the miss counts are teaching values |
| Beta-factor model | Section 9 | 2026-11-10 | Reliability of a multi-channel detection function | Inline | `beta` is an input, not a measurement; real values come from a common-cause analysis |
| Capacity model | Section 9 | 2026-11-10 | Tag refresh versus population and slot time | Inline | Protocol overhead modelled, not measured |

The method-selection ledger records what was tested, what was computed and what was merely considered, because the case's credibility rests on not confusing the three.

| Decision | Tested | Computed | Considered only |
|---|---|---|---|
| Camera path latency | Bench measurement of inference only | Full path latency from component budgets | In-cell latency under dust |
| Fused miss rate | 240-episode trial (simulated) | Independence prediction and beta-factor fit | Long-run rate at production volume |
| Tag refresh | No | Capacity arithmetic from slot time and population | Behaviour under anchor failure |
| Field pullback | No | `S = K·T + C` for four configurations | In-situ verification with a test piece |
| Availability effect | No | Stop-count arithmetic and derated benefit | Operator behaviour under frequent restriction |
| Brake upgrade | No | Stop-time reduction and resulting `S` | Brake wear, maintenance interval |

The reproducibility boundary is stated once and applies to every number in this file: **the parameters are chosen to be internally consistent and pedagogically useful, and independent recomputation will reproduce the arithmetic but not any plant's reality.**
