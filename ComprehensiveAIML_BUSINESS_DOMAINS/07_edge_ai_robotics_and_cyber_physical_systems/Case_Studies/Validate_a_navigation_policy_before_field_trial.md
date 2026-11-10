---
case_id: "CASE-P07-012"
case_title: "Validating a learned navigation policy before field trial"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "draft"
content_status: "draft"
source_status: "OPEN"
case_type: "hybrid"
domain: "Edge robotics — warehouse mobile navigation, sim-to-real transfer"
primary_pattern: "transfer validation ladder"
secondary_patterns: ["scenario coverage", "controlled recovery"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "Director of warehouse autonomy (decision owner); robotics safety engineer (independent sign-off)"
source_index: "INDEX.md"
review_trigger: "Change to site layout, fleet speed policy, safety-monitor firmware, near-miss definition, simulator version, or the contact-rate limit set by the safety case"
---

# Case study: Validating a learned navigation policy before field trial

## 1. The One-Line Framing

Given a candidate learned navigation policy (checkpoint `nav-v7`, a reinforcement-learned velocity-command policy trained only in a warehouse simulator) together with a staged evidence package, decide for the director of warehouse autonomy whether to **promote it to a limited field trial**, **hold it for more evidence**, or **reject the candidate** — for a two-robot, four-week trial in one live aisle zone — to reduce operator interventions and aisle deadlocks without exceeding the safety case's contact-rate limit, compared with the incumbent classical planner already running in that zone.

The decision recurs once per candidate checkpoint, roughly every six to ten weeks during the current development cycle. It is a **control** problem at the point of use and an **evidence-gating** problem at the point of decision: the policy emits velocity commands at 20 Hz, but the decision under study is not "what command" — it is "may this artifact touch a shared floor with people on it".

This is an **ILLUSTRATIVE** design exercise. No warehouse, fleet, vendor, incident, injury, regulator or deployment is described. Every count, rate, cost and interval below is a constructed teaching input. The arithmetic on those inputs is checked independently and the checks are shown; the inputs themselves are not measurements of anything. Nothing here establishes that a simulator pass implies field safety, and nothing here is a safety case.

The primary success condition in one sentence: the candidate must show, on evidence whose real-world relevance is defensible, that its rate of unplanned contact per robot-metre and its rate of near-miss per robot-metre stay below the limits the safety case already fixed, while throughput and intervention rate improve against the classical baseline.

The reusable pattern is the **transfer validation ladder**: offline evaluation, then simulation sweep, then hardware-in-the-loop, then a controlled physical environment, then limited operation, with a separately budgeted evidence gate at each rung and an explicit refusal to let a pass at one rung be read as a pass at the next. Two secondary patterns apply: **scenario coverage** (does the simulation support contain the field distribution, and how far outside it does the field reach?) and **controlled recovery** (when the policy degrades, does the system reach a defined safe state, and is that state reached by design rather than by luck?).

The surprising tradeoff, and the reason the obvious framing fails: the binding constraint is almost never the size of the simulation sweep. It is the size of the **paired** set — the small number of scenarios executed both in simulation and on real hardware — because the correction for simulator optimism is estimated from paired data alone, and its standard error dominates everything the simulator contributes. In this case the field-matched holdout contains 900 episodes and contributes a standard error of 0.0057; the paired set contains 120 episodes and contributes 0.0289 — 96.3% of the total variance. Doubling the simulation holdout moves the interval half-width by 0.05 percentage points. Adding three hundred paired episodes moves it from ±5.8 to ±3.2 percentage points.

A second framing failure is subtler. The team's first instinct was to compare an aggregate success rate against a single threshold. That aggregate averages over scenario families whose difficulty differs by a factor of four, so it is dominated by whichever families the sampler happened to draw, and it hides the one corner of the environment — low friction combined with pallet overhang — that the simulator never generated. An average cannot certify a tail, and a safety gate is a statement about a tail.

> **Takeaway:** the decision is not "did the policy score well in simulation" but "does the evidence chain survive the correction for simulator optimism, the extrapolation outside the simulator's support, and the exposure needed to bound a rare event".

## 2. Context & Background

### 2.1 The operational setting

A regional distribution centre moves palletised goods from receiving to reserve storage and from reserve storage to outbound staging. Mobile robots traverse a grid of aisles roughly 2.6 m wide, cross-aisles with two-way traffic, and staging areas where human pickers walk. Each robot carries a 2-D LiDAR at 0.35 m height, wheel odometry, and an inertial unit; localisation is a scan-matching filter against a stored map; the navigation stack converts a global path into local velocity commands.

The candidate policy replaces only the **local planner**. Global routing, localisation, the map, and the safety-rated hardware chain (bumper, lidar safety field, e-stop) are unchanged and remain outside the candidate's authority. That boundary matters for the whole case: the policy can make the robot slow, stop, or take a poor line, but it cannot disable the safety-rated stop. If the safety chain were inside the candidate's scope, no amount of the statistics below would be sufficient and the decision would belong to a certification process rather than to an engineering gate.

Before the decision, the zone runs on a classical planner: a sampling-based local planner over an inflated costmap with hand-tuned parameters, a recovery behaviour set (rotate in place, back off, replan), and a rule that halts after three consecutive recovery attempts. After the decision — if promotion happens — the same zone runs the learned policy behind the same safety chain, with a newly written runtime monitor that vetoes commands violating a clearance floor and a new operator escalation procedure.

### 2.2 Who is affected, and who decides

| Role | Interest in the decision | Authority |
|---|---|---|
| Director of warehouse autonomy | Delivery schedule for the autonomy programme; owns the promotion decision | Approves, defers or rejects promotion |
| Robotics safety engineer | Independent judgement that the evidence supports the claimed contact-rate bound | Holds a veto; not in the delivery reporting line |
| Site operations manager | Throughput, picker safety, disruption during the trial | Can halt the trial unilaterally at any time |
| Fleet engineer | Rollback, versioning, on-call, remote assistance load | Executes deployment and rollback |
| Perception/navigation engineers | Policy quality, scenario coverage, retraining | Supply evidence, do not set the gate |
| Data platform engineer | Logging completeness, clock discipline, artefact retention | Certifies that the evidence package is reproducible |
| Insurer and internal risk | Exposure acceptance for a new autonomy mode | Consulted; not modelled quantitatively here |

The separation between the person who wants the schedule met and the person who signs the safety judgement is the single most important organisational control in this case. Every quantitative gate below is only as good as that separation.

### 2.3 The baseline and why it persists

The classical planner's weaknesses are documented in the zone's own incident and intervention logs, not inferred:

- **Pallet overhang.** Pallets whose load overhangs the pallet face by more than roughly 8 cm intrude into the inflation envelope; the costmap built from the static map does not contain the overhang and the lidar sees it late. The planner then enters a rotate-back-replan loop.
- **Reflective shrink-wrap.** Specular returns from wrapped loads produce sparse phantom obstacles; the planner treats them as real and refuses to advance, which consumes recovery attempts.
- **Cross-aisle deadlock.** Two robots meeting at a cross-aisle with a picker present will each yield, then each decide the other has yielded, and both resume.
- **Non-monotone recovery.** After three failed recoveries the robot halts and waits for a human, which converts a navigation problem into an operator-response problem.

The baseline is not a straw man. It is deterministic enough to reason about, its failure modes are known and bounded, its parameter changes are reviewable line by line, and it has years of accumulated field behaviour behind it. Any promotion must beat it on intervention rate and deadlock frequency while remaining statistically indistinguishable on safety. A candidate that is merely "smoother" has not earned a field trial.

### 2.4 Scale and timing

| Quantity | Value used in this case | Basis |
|---|---|---|
| Robots in the zone | 2 for the trial, 14 fleet-wide | Illustrative |
| Aisle zone area | 4,000 m² | Illustrative |
| Shift pattern in trial | 6 h/day, 5 days/week, off-peak window | Illustrative |
| Nominal speed cap during trial | 0.9 m/s | Illustrative |
| Duty factor (fraction of time moving) | 0.62 | Illustrative |
| Mean mission length in the zone | 295 m | Illustrative |
| Baseline interventions | 4.3 per 100 missions | Illustrative |
| Baseline unplanned contacts | 0 observed in 6 months | Illustrative |

The six-month baseline record of zero contacts is important and slightly treacherous. Zero observed events in a large exposure does not mean the rate is zero; it means the rate is below whatever the exposure can resolve. The classical planner's contact-rate upper bound and the candidate's must be compared on the same exposure accounting, or the comparison is meaningless.

### 2.5 Why the current process persists despite its limitations

The classical planner persists because its failure modes are **legible**. An operator sees a robot rotating in place and knows what is happening. A learned policy that slows to 0.2 m/s for eleven seconds and then proceeds looks, to an observer, like nothing at all — and if it is actually uncertain, the operator has no signal. The intervention-rate metric captures the visible cost and misses the invisible one, which is why the runtime monitor and its clearance floor are part of the design rather than an afterthought.

> **Takeaway:** the incumbent is a real competitor with a legible failure signature; the candidate's advantage has to be measured on the same exposure and the same event definitions, or the comparison flatters whoever counts more carefully.

## 3. The Business Question

The decision owner's question, recorded as a paraphrase of the programme review discussion rather than a quotation: "We have a policy that looks strong in simulation and a trial slot booked for two robots next month. Do I put it on the floor, and if not, what exactly would I need to see before I do?"

| Decision-contract field | Contract used in this case |
|---|---|
| Question | Promote `nav-v7` to a four-week, two-robot field trial in the live aisle zone, hold it in the controlled facility, or reject it? |
| Decision owner | Director of warehouse autonomy, with a veto from the robotics safety engineer |
| Trigger and deadline | Trial slot booked four weeks out; fleet release calendar two quarters ahead |
| Available actions | (a) promote as scoped; (b) promote with extended exposure and tightened monitoring; (c) hold at controlled-facility stage and retrain; (d) reject and revert to classical-only roadmap |
| Value and downside | Fewer interventions and deadlocks versus the risk of an unplanned contact with a person or asset, plus programme delay |
| Objective | Reduce interventions per 100 missions without increasing contact or near-miss rate per robot-metre above the safety-case limits |
| Hard constraints | Contact rate ≤ 1 per 200,000 robot-metres; near-miss rate ≤ 1 per 50,000 robot-metres; no change to the safety-rated chain; intervention definition fixed before exposure |
| Scope | One zone, two robots, one policy checkpoint, four (or five) weeks of off-peak operation |
| Out of scope | Fleet-wide rollout, other zones, payload handling, charging-dock docking, human-robot collaboration modes, and any change to the safety case itself |

### 3.1 Objective versus proxy

The business objective is "operate the zone with fewer human interventions at no greater physical risk". The measured proxies are interventions per 100 missions and events per robot-metre. These can disagree with the objective in both directions.

A policy can reduce interventions by becoming more cautious: it slows, waits longer, and completes missions without human help but with lower throughput. Interventions fall, the objective is not served, and the throughput guardrail is what catches it. Conversely, a policy can increase interventions while being safer, if the monitor vetoes marginal manoeuvres that the classical planner would have attempted. That is a good outcome reported as a bad number.

The conflict is resolved by making throughput and intervention rate **co-primary on the benefit side** and by treating safety as a hard gate rather than a traded quantity. Safety is never exchanged for throughput inside this decision; if the candidate cannot meet the contact gate at the planned throughput, the answer is a smaller or slower trial, not a relaxed gate.

### 3.2 What an expensive wrong decision looks like

Two asymmetric errors exist and they do not have symmetric costs.

**False promotion** puts a policy on a shared floor whose contact rate is above the safety case's limit. The consequence is physical, potentially injurious, and not recoverable by rollback: a rollback stops future exposure but does not undo an event. It also destroys the credibility of the evidence process, which is the more durable loss.

**False hold** delays the programme by one cycle, costs the re-run of a bench study, and keeps a worse-but-legible planner in place for another quarter. The cost is opportunity and schedule, both recoverable.

Given that asymmetry, the correct default under genuine uncertainty is hold, and the burden of proof sits with promotion. The statistics in Section 6 are constructed to make that burden explicit: a gate that a mediocre policy passes half the time is not a gate.

> **Takeaway:** the decision is one-sided by construction — evidence must be strong enough that a policy at the safety limit is unlikely to pass, and "we could not show it was unsafe" is a hold, not a pass.

## 4. Assumptions & Constraints

The table separates things the team assumes and can test, constraints imposed from outside, and choices the team controls. Status is recorded as of the decision date.

| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback / owner |
|---|---|---|---|---|---|---|
| A-01 | Simulator contact model produces contacts at approximately the true rate when geometry and dynamics are correct | model | paired bench study on instrumented targets | unverified | sim contact rate is not evidence | treat sim contacts as ranking signal only / nav lead |
| A-02 | Domain-randomisation ranges bracket the field distribution | data | floor survey plus 6-week friction and traffic measurement | **violated** | extrapolation outside support | widen DR, retrain / nav lead |
| A-03 | Holdout simulation seeds are disjoint from training seeds | data | seed-set diff against training manifest | **violated** | holdout success inflated | rebuild holdout from unseen seed block / data engineer |
| A-04 | Near-miss definition is fixed before exposure starts | measurement | preregistration record | unverified | definition drift changes pass/fail | preregister, freeze code / safety engineer |
| A-05 | Wheel-odometry distance is accurate to within 1% | measurement | compare against fiducial-referenced distance | unverified | denominator inflated, rate understated | use fiducial correction / fleet engineer |
| A-06 | Paired bench scenarios are representative of the zone | data | coverage audit of aisle types in the paired set | **violated** | discrepancy estimate biased low | stratify paired sampling / nav lead |
| A-07 | Safety monitor adds ≤3 ms at p99 | infrastructure | on-target latency trace | unverified | deadline misses rise | optimise or revert / embedded engineer |
| A-08 | Logging does not change policy timing | infrastructure | A/B build with telemetry on and off | **violated** | measured system is not deployed system | sample telemetry / embedded engineer |
| C-01 | Safety-rated chain (bumper, safety lidar field, e-stop) is unmodified and independent | safety | configuration audit and functional test | confirmed by procedure, not audited here | candidate inherits a false safety claim | stop; escalate to safety board / safety engineer |
| C-02 | Contact limit 1 per 200,000 robot-metres | safety | safety case, dated | assumed fixed | gate is meaningless | do not proceed / safety engineer |
| C-03 | Near-miss limit 1 per 50,000 robot-metres | safety | safety case, dated | assumed fixed | gate is meaningless | do not proceed / safety engineer |
| C-04 | No policy may command motion when the monitor's clearance estimate is stale by more than 120 ms | safety | monitor unit test plus fault injection | unverified | silent unsafe motion | hold / embedded engineer |
| C-05 | Trial runs off-peak only, no pickers in the trial aisles during the first two weeks | operational | shift plan and zone demarcation | proposed | exposure understates human risk | extend, do not compress / site ops |
| C-06 | Rollback to classical planner completes within one shift | operational | rehearsal on one robot | unverified | exposure continues after a stop decision | rehearse before promotion / fleet engineer |
| C-07 | Budget ceiling for the evidence programme | business | programme budget line | assumed | cannot close the paired gap | descope zone, not gate / director |

### 4.1 The three assumptions that can reverse the decision

Most of the table can be wrong without changing the answer. Three cannot.

**A-02 is already violated.** The simulator's training distribution put friction in [0.55, 0.85] and pallet overhang up to 0.10 m. The floor survey and six weeks of driving data put field friction as low as 0.45 on the resin-coated aisle near the chilled dock, and overhang up to 0.12 m on inbound pallets straight from a supplier whose wrapping standard differs. The field therefore contains conditions the policy never saw. Section 6.5 turns that statement into a distance, and the distance into a performance bound, and the bound into a gate outcome.

**A-03 is already violated.** The holdout sweep used a seed block that overlaps the training manifest at 11%. This is the ordinary, unglamorous leakage that inflates simulation results: the policy has seen the exact friction and overhang draws that the evaluator treats as unseen. The size of the inflation is not directly measurable, which is why the paired study is the only route to a field-relevant number — and also why the holdout number must be reported as an upper bound rather than an estimate.

**A-06 is already violated.** The 120 paired episodes were executed in the dry, flat, unoccupied central aisles because those were the ones the bench team could block off on short notice. They are the easiest aisles in the zone. A discrepancy estimated there is a lower bound on the discrepancy elsewhere, which biases the corrected field estimate upward — in the flattering direction.

The pattern across all three is the same, and it is the case's central methodological warning: **every bias discovered in this evidence package flatters the candidate.** Leakage inflates the simulator number, narrow DR hides a hard corner, and a convenient paired set shrinks the correction. When all the errors point one way, the correct response is not to average them but to ask what the evidence would look like if each were repaired.

### 4.2 Constraints that are not negotiable inside the case

The contact and near-miss limits come from the safety case and are inputs to this decision, not outputs of it. The safety-rated chain stays outside the candidate's authority. The trial cannot begin before the intervention definition and the near-miss definition are frozen and recorded, because Section 6.11 shows that the choice between a 0.15 m and a 0.25 m clearance threshold flips the gate outcome on identical data.

Two further constraints are organisational. The safety engineer's veto is real, not advisory. And the site operations manager can halt the trial unilaterally — which means the trial's stop procedure must be executable by someone who is not a roboticist, on a timescale of seconds.

> **Takeaway:** three of the eight data assumptions are violated before the decision starts, and all three violations bias the evidence toward promotion; the design of the remaining work has to assume that pattern rather than hope it is accidental.

## 5. Data, Tools & Method

### 5a. Data Specification

The case uses three disjoint evidence populations. Keeping them separate is the whole method; blending them into one "validation set" is the failure this case exists to prevent.

| Field | Population 1 — simulation sweep | Population 2 — paired bench | Population 3 — field baseline |
|---|---|---|---|
| Unit of analysis | One episode: policy reset to a sampled initial pose, run to terminal state | One episode: the *same* seeded scenario executed in simulation and on the bench rig | One mission: a real robot travelling the zone under the classical planner |
| Volume and window | 3,000 episodes (original), 2,500 episodes (rebuilt clean holdout) | 120 episodes, 40 scenario seeds × 3 repeats | 6 months, 5,120 missions |
| Inputs | Simulated lidar scan (720 beams, 10 Hz), odometry with injected noise, goal pose, prior command | Identical observation schema on the rig, with motion-capture ground truth | Robot telemetry, mission log, operator intervention log |
| Outcome / label | Episode success (goal reached within 90 s without a clearance violation); minimum clearance to any target; path length; time to goal | Same definitions, measured by motion capture at 100 Hz | Completed / intervention / abandoned; intervention reason code |
| Population and exclusions | Warehouse-like layout generator; excludes dock approach and charging | Central dry aisles only — **the easiest 46% of aisle length** | Full zone including chilled aisles |
| Join semantics | Scenario seed is the key; one row per (seed, repeat) | Seed is the key into the simulation run table; 1:1 with the sim episode | Mission ID; joins to robot and date only |
| Splits | Training seeds [0, 400000); holdout from a clean block [900000, 950000) | Paired seeds drawn from the holdout block | Forward time window after policy freeze |

The metric definitions are fixed before any exposure and are recorded here because their exact form decides the case.

**Episode success.** Numerator: episodes that reach the goal region (within 0.30 m of the goal pose, heading within 15°) within 90 s of sim/bench time, without any clearance measurement below 0.20 m to a static target and without a human takeover. Denominator: all episodes started. Unit: dimensionless proportion. Population: field-matched holdout. Window: one episode.

**Minimum clearance.** The smallest Euclidean distance in the ground-truth frame between any point of the robot footprint and any point of a target's occupied volume, sampled at 100 Hz and reported per episode as a minimum. Unit: metres. This is a *measured* quantity on the bench and a *simulated* one in simulation; the difference between them is part of what the paired study estimates.

**Intervention.** Numerator: missions during which an operator acknowledged an alert, remotely commanded the robot, or physically attended it. Denominator: missions started in the zone during the window. Unit: per 100 missions. A mission with two attendances counts once; a definition that counts attendances would inflate the baseline by 14% in the archived logs, which is precisely the kind of drift the preregistration prevents.

**Contact.** Any physical interaction between the robot's footprint and a person or asset that is not a planned docking contact, regardless of force or damage. Denominator for the rate: robot-metres travelled. This denominator, not hours, is correct because exposure to geometry scales with distance travelled, not with time spent stationary.

**Near-miss.** Any commanded or executed trajectory that brings the footprint within 0.25 m of a person or a moving asset at a relative speed above 0.3 m/s, without contact. Denominator: robot-metres. Both the clearance and the relative-speed condition are needed; clearance alone counts a robot creeping past a stationary pallet, which the safety case does not regard as a near-miss.

**Robot-metres.** Distance travelled by the robot's reference point, integrated from wheel odometry with a fiducial-based scale correction, accumulated while the robot is in motion. Unit: metres. Not path length planned, not mission distance assigned.

### 5b. Tools & Stack

| Component | Role | Status in this case |
|---|---|---|
| Warehouse simulator (GPU, batched) | Generates episodes, domain-randomised parameters, ground-truth state | Exists; version and DR ranges are the object of study |
| Scenario sampler | Draws layout, friction, payload overhang, obstacle placement, pedestrian script | Exists; its ranges are the coverage question |
| Policy trainer (PPO-style, vectorised) | Produces the checkpoint under test | Exists; checkpoint `nav-v7` frozen at the decision date |
| Bench rig | 6 m × 4 m instrumented floor, motion capture, two static targets, one moving target on a rail | Exists; covers straight and single-corner geometry only |
| Paired harness | Replays one seed in simulation and on the rig; aligns start pose and target placement | Exists; 120 episodes executed |
| Runtime monitor | Vetoes commands whose predicted clearance is below 0.18 m or whose perception is stale by more than 120 ms | **Proposed**; not implemented at decision date |
| On-target logger | Records commands, scans, monitor decisions, odometry | Exists; telemetry sampling changes timing (A-08) |
| Decision log | Stores checkpoint hash, gate version, cohort, metric values, verdict, sign-off | Exists |
| Statistical script | Wilson intervals, paired differences, McNemar exact test, mixture weighting | Exists; version pin recorded in 5e |

Tool names and vendor identities are deliberately absent. The point of the stack table is the *role* of each component and which of them are proposed rather than built, because a verdict that depends on an unimplemented monitor is not a verdict about a deployed system.

### 5c. Method Summary

The pipeline is: freeze the checkpoint and the metric definitions → evaluate on a clean holdout → estimate the simulator discrepancy from the paired set → correct the holdout estimate → weight by the field traffic mix → compare the lower confidence bound against the preregistered threshold → separately assess whether the planned exposure can resolve the two event-rate gates → assemble the verdict.

That order matters. Correcting before weighting, and weighting before comparing a bound, ensures each adjustment is visible and independently attackable. A single number produced by a pipeline that does all three at once cannot be audited.

| Candidate approach | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Aggregate simulation success rate against a fixed threshold | Original 3,000-episode sweep, success 0.938 | Cheap; already computed | Ignores seed overlap, ignores family mix, cannot speak to field | **Rejected** — leakage detected, 13% of holdout seeds present in the training manifest | Never as a standalone gate |
| Clean holdout plus additive sim-to-real correction | 2,500 clean episodes, 120 paired episodes | Field-relevant point estimate with honest variance | Correction estimated on easy aisles; assumes bias is additive | **Chosen** | Paired set stratified by aisle type |
| Coverage-weighted mixture over traffic mix | Holdout split by friction band, bench low-friction subset | Makes extrapolation explicit and prices it | Weights from 6 weeks of traffic data; low-friction bench sample is 24 episodes | **Chosen as the primary gate input** | Traffic mix shifts seasonally |
| Worst-case corner analysis (min over families) | Holdout family minima | Conservative; simple to defend | Over-conservative: one unlucky family vetoes everything | Considered, used as a secondary display | If family counts grow |
| Direct field A/B on the live zone | None at decision time | Measures the real thing | Cannot resolve the contact gate within budget (Section 6.10); exposes people before the candidate passes | **Rejected for this decision** | After performance gate passes |
| Simulated-only domain-randomised stress sweep at 10× scale | 25,000 extra episodes, $750 | Cheap ranking signal for retraining | Does not correct the discrepancy; variance contribution ~0.6% | Considered, deferred | After retraining |

The chosen method's central assumption is **additivity**: that the sim-to-real gap is roughly the same on the holdout as on the paired set. Section 6.6 tests how badly that fails when the paired set is unrepresentative, which is the actual situation here.

### 5d. Data provenance, quality, exclusions and leakage controls

| Source ID | Content | Owner | Extracted | Access | Limitation |
|---|---|---|---|---|---|
| SRC-01 | Simulation sweep, 3,000 episodes, original seed block | nav team | 2026-09-14 | internal | 13% seed overlap with training manifest |
| SRC-02 | Training manifest: seed ranges and checkpoint hash | nav team | 2026-09-14 | internal | Only seed ranges, not per-episode provenance |
| SRC-03 | Rebuilt holdout, 2,500 episodes, clean block | nav team | 2026-10-02 | internal | Sampler version differs from SRC-01 |
| SRC-04 | Paired bench study, 120 episodes | test engineering | 2026-10-19 | internal | Central aisles only |
| SRC-05 | Floor friction survey, 46 sample points | facilities | 2026-09-30 | internal | Spot survey, not continuous |
| SRC-06 | Mission and intervention logs, 6 months | fleet ops | 2026-10-05 | restricted | Reason codes free-text in 8% of rows |
| SRC-07 | Safety case extract: contact and near-miss limits | safety | 2026-08-21 | restricted | Limits are inputs; the case itself is out of scope here |

Four quality problems were found, and all four matter because each one moves the answer toward promotion.

**Leakage (seeds).** The holdout block was generated with a sampler configuration whose RNG was seeded from the same base as training. Checking the manifests showed 390 of 3,000 holdout episodes shared a seed with a training episode. Those 390 succeeded at 1.000 while the remaining 2,610 succeeded at 0.929. The reported 0.938 was therefore an artefact: recomputing on clean seeds alone gives 0.929, and the rebuilt holdout from an independent block gives 0.912. Net inflation from leakage: 2.6 percentage points on the original sweep, and a further 1.7 points from the composition change to a field-matched mix. Neither number is large, but a gate that sits within 3 points of its threshold cannot afford either.

**Leakage (normalisation).** Observation normalisation statistics were computed over the full sweep including the holdout before the split was drawn. The effect here is small (the statistics are dominated by geometry that does not vary) but the practice is unsound and was corrected in the rebuild.

**Missingness and skew.** The bench rig logs 100 Hz motion capture but the robot's own scan at 10 Hz; where a scan is dropped, the bench episode is discarded rather than interpolated. 7 of 127 attempted bench episodes were discarded for this reason, leaving 120. Discarding is correct here — interpolating a dropped scan would silently change the policy's input distribution and flatter it — but it also means the surviving 120 episodes are the ones with clean sensing, which is another mild optimistic selection.

**Exclusions.** Dock approach, charging, and the chilled-zone low-friction aisles are excluded from the simulation population. The first two are legitimately out of scope. The third is not: the chilled aisles carry 22% of missions, and their exclusion is the coverage gap that Section 6.7 prices. An exclusion that removes a hard region is not a scoping decision, it is a bias.

### 5e. Reproducibility and method-selection ledger

| Artefact ID | Location | Version / date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| RUN-01 | Simulation sweep output store | sampler 3.1.0, 2026-09-14 | Original aggregate | internal | Seed provenance incomplete |
| RUN-02 | Rebuilt holdout store | sampler 3.2.1, 2026-10-02 | Primary sim input | internal | GPU model pinned, driver not |
| RUN-03 | Bench session 2026-10-14..16 | rig firmware 2.4.0 | Paired discrepancy | internal video + mocap | Rig calibration drift not logged |
| RUN-04 | Statistical report | script v1.2, seed 11 | All interval arithmetic | internal | Depends on SRC-03/04 only |
| RUN-05 | Budget model | spreadsheet, 2026-10-20 | Cost of evidence | internal | Labour rates are planning rates |

Reproduction is partial. RUN-02 can be regenerated given the sampler version, the seed block and the GPU class; it cannot be regenerated bit-identically because the batched simulator's physics stepping depends on scheduling, and the driver version was not recorded. RUN-03 cannot be repeated exactly: the bench session is a physical event, the rig was re-shimmed afterwards, and target placement was measured with a tape rather than the mocap frame in four of the 120 episodes. Anyone reproducing this must accept a discrepancy estimate with an unmodelled component of a few percent, which is why the verdict's margin is reported rather than treated as a binary.

Seed discipline across the whole case: training used PPO with 8 environment workers and seeds 0..7 for policy initialisation; the *scenario* seed space is separate from the *policy* seed space, and the leakage was in the scenario space. Confusing the two is a common failure; a reviewer who checks "did we change the training seed" will conclude there is no leakage and be wrong.

> **Takeaway:** the evidence package has four defects and every one of them is optimistic; the method is built to make each defect a visible adjustment with its own variance rather than an unstated assumption.

## 6. Analysis Walkthrough

### 6.0 The decisive result first

The candidate fails the performance gate. The best defensible estimate of field episode success is 0.806 with a 95% interval of [0.745, 0.867]; the preregistered requirement is that the lower bound reach 0.780. On the point estimate alone the candidate passes (0.806 > 0.780), which is exactly why the gate is written on the lower bound: a point estimate at 0.806 cannot distinguish a policy at 0.85 from one at 0.76, and the second is worse than the incumbent's bench-measured equivalent.

The candidate also cannot close the contact gate with the planned four-week trial, regardless of what the trial shows, because the exposure resolves only 1.24 times the permitted rate. The near-miss gate *can* be closed by the same trial. That asymmetry — one gate closable, one not — is what converts a simple "run it and watch" plan into a staged decision with a real hold in the middle.

Everything below builds those three statements from the inputs and shows the arithmetic.

### 6.1 Step trace

| Step | Hypothesis or decision | Input snapshot | Method | Intermediate output | Interpretation | Status |
|---|---|---|---|---|---|---|
| 1 | The original sweep's 0.938 is trustworthy | RUN-01 + SRC-02 | Seed-set intersection | 390/3,000 overlap (13.0%) | Reject the sweep as a gate input | DERIVED |
| 2 | Rebuild on a clean block changes little | RUN-02 | Recompute success on 2,500 clean episodes | 0.9122 | 2.6 pp below the original; composition also changed | DERIVED |
| 3 | Simulation overstates bench performance | RUN-03 | Paired difference over 120 episodes | −0.0917, SE 0.0289 | Discrepancy is real and one-directional | DERIVED |
| 4 | The discrepancy is not chance | RUN-03 | McNemar exact on 13 discordant pairs | p = 0.0034 | A 12:1 direction split is unlikely under no effect | DERIVED |
| 5 | Correct the holdout | Steps 2–3 | Additive correction | 0.8205, SE 0.0294 | Passes on point estimate, fails on bound | DERIVED |
| 6 | The correction was measured on easy aisles | SRC-04 + SRC-05 | Stratify by aisle class | 46% of aisle length represented | Correction is a lower bound on the true gap | ILLUSTRATIVE |
| 7 | Field friction falls outside training support | SRC-05 + DR ranges | Range comparison | 0.45 vs training min 0.55 | Extrapolation required | DERIVED |
| 8 | Weight by the field traffic mix | SRC-06 | Mixture expectation | 0.8058, SE 0.0312 | Lower bound 0.745 | DERIVED |
| 9 | Compare against the preregistered threshold | Step 8 | Lower bound vs 0.780 | Shortfall 3.5 pp | **Gate fails** | DERIVED |
| 10 | Can a four-week trial resolve the contact gate | Exposure model | Rule of three | Upper bound 1.24× limit | No | DERIVED |
| 11 | Can it resolve the near-miss gate | Exposure model | Rule of three | Upper bound 0.31× limit | Yes | DERIVED |
| 12 | What exposure would resolve the contact gate | Exposure model | Invert the bound | 600,000 m ≈ 5.0 weeks | Feasible but not with the booked slot | DERIVED |

### 6.2 Step 1 — the leakage audit

The audit is a set intersection, not a statistical test. Every episode row carries `scenario_seed`; the training manifest lists the seed ranges consumed. Intersecting them is exact and cheap, and its absence from the original report is the single most consequential process failure in the case.

```text
overlap_fraction = |S_holdout ∩ S_train| / |S_holdout|
                 = 390 / 3000
                 = 0.130
```

**Interpretation.** 13% of the evaluation population was, in effect, a training population. Within that 13%, success was 1.000; outside it, 0.929. This is the signature of memorised scenarios rather than generalised competence, and it is invisible in the aggregate.

### 6.3 Step 2 — the rebuilt holdout

The rebuilt holdout was drawn from an independent seed block and from a traffic-mix-matched scenario sampler. It contains 2,500 episodes with 2,281 successes.

```text
p_hat_sim = 2281 / 2500 = 0.91240
```

Two changes were made at once, which is a methodological smell that should be stated: the seed block *and* the sampler composition changed. Separating them would require a third run. Because the composition change was made to match the field, and because matching is required for the later correction to be interpretable, the combination is defensible; but the 1.7 pp difference attributable to composition is an inference, not a measurement, and it is labelled as such.

**FORMULA [F-01] — Wald standard error of a proportion**

```text
FORMULA: SE(p_hat) = sqrt( p_hat (1 - p_hat) / n )
WHERE: p_hat success proportion [dimensionless]; n episodes [count].
CONSTRAINT: independent episodes, n large enough that np_hat >= 5 and
            n(1-p_hat) >= 5; not for extreme proportions or tiny n.
NUMERICAL EXAMPLE: p_hat = 0.9124, n = 2500
                   p_hat(1-p_hat) = 0.9124 * 0.0876 = 0.079927
                   SE = sqrt(0.079927 / 2500) = sqrt(0.0000319716)
                      = 0.0056546
INTERPRETATION: the simulation number is known to about ±0.6 pp, which
                sounds precise and is irrelevant — see F-03.
```

The Wald standard error is used here only to demonstrate the variance decomposition. The gate itself uses a Wilson interval, because at the lower bound of the range the case cares about, Wald intervals under-cover and can even cross zero.

**FORMULA [F-02] — Wilson score interval**

```text
FORMULA: centre = (p_hat + z^2/(2n)) / (1 + z^2/n)
         half   = (z / (1 + z^2/n)) * sqrt( p_hat(1-p_hat)/n + z^2/(4n^2) )
         interval = [centre - half, centre + half]
WHERE: z = 1.959964 for 95%; n episodes; p_hat proportion.
CONSTRAINT: binomial sampling, independent episodes, two-sided.
NUMERICAL EXAMPLE: n = 2500, p_hat = 0.9124, z = 1.96
                   z^2/n = 3.8416/2500 = 0.0015366
                   denominator = 1.0015366
                   centre = (0.9124 + 0.0007683)/1.0015366
                          = 0.9131683/1.0015366 = 0.911767
                   p_hat(1-p_hat)/n = 0.079927/2500 = 0.0000319716
                   z^2/(4n^2) = 3.8416/(4*6,250,000) = 1.5366e-7
                   inner = 0.0000319716 + 0.0000001537 = 0.0000321253
                   sqrt(inner) = 0.0056680
                   half = (1.96/1.0015366)*0.0056680 = 1.9570*0.0056680
                        = 0.011092
                   interval = [0.90068, 0.92286]
INTERPRETATION: the simulation success rate is 0.912 with a 95% interval
                of roughly [0.901, 0.923] — tight, and still not
                evidence about the field.
```

### 6.4 Step 3 — the paired discrepancy

The paired study is the only place where a number about the real world enters a formula that also contains a number about the simulation. Each of 120 scenario seeds was executed twice: once in simulation, once on the bench rig, with start pose, target placement and pedestrian script aligned. The outcome is binary per execution, so each pair contributes a difference in {−1, 0, +1}.

| Pair outcome | Count | Contribution to mean difference |
|---|---:|---:|
| Both succeed | 94 | 0 |
| Simulation succeeds, bench fails | 12 | −12 |
| Simulation fails, bench succeeds | 1 | +1 |
| Both fail | 13 | 0 |
| **Total** | **120** | **−11** |

```text
p_hat_sim_paired   = (94 + 12) / 120 = 106/120 = 0.88333
p_hat_bench_paired = (94 + 1)  / 120 =  95/120 = 0.79167
delta_hat = 0.79167 - 0.88333 = -0.091667   (-11/120)
```

**FORMULA [F-03] — paired mean difference and its standard error**

```text
FORMULA: delta = (1/n) * sum_i (y_bench,i - y_sim,i)
         SE(delta) = s_d / sqrt(n),  s_d^2 = (1/(n-1)) * sum_i (d_i - delta)^2
WHERE: y in {0,1}; d_i in {-1,0,+1}; n pairs; s_d sample sd of differences.
CONSTRAINT: pairs are matched on the scenario, not merely on scenario
            family; independent pairs; no pair drives the result.
NUMERICAL EXAMPLE: n = 120, sum d_i = -11, delta = -0.091667
                   sum d_i^2 = (12)(1) + (1)(1) = 13   (only 1s contribute)
                   sum (d_i - delta)^2 = 13 - 120*(0.091667)^2
                                       = 13 - 120*0.008403
                                       = 13 - 1.00836 = 11.99164
                   s_d^2 = 11.99164/119 = 0.100770
                   s_d = 0.317443
                   SE = 0.317443/sqrt(120) = 0.317443/10.95445 = 0.028978
INTERPRETATION: the simulation is optimistic by about 9.2 percentage
                points on paired scenarios, ±2.9 pp at one standard error.
```

There is a shortcut that is worth knowing because it is exact for binary pairs and it exposes *why* the standard error is large: only the discordant pairs carry information about the difference.

```text
SE(delta) ≈ sqrt(D) / n ,  where D = b + c = discordant count
           = sqrt(13)/120 = 3.60555/120 = 0.030046
```

The two expressions differ only by the small-sample factor sqrt((n−1)/n) applied to the variance and a centring term; 0.03005 versus 0.02898 is a 3.5% difference attributable to the `n/(n−1)` correction on `s_d^2` combined with the mean-centring term. The lesson is the important part: **with 13 discordant pairs, the discrepancy estimate is a statement about 13 events.** It will never be precise, no matter how many concordant pairs are added. Adding 500 concordant episodes leaves SE(delta) at 0.030 while costing the same as 500 real bench runs.

**Is the direction real?** Under the null hypothesis that simulation and bench are exchangeable, each discordant pair is equally likely to favour either. With b = 12 and c = 1:

```text
p_exact = 2 * sum_{k=0}^{1} C(13,k) * 0.5^13
        = 2 * (1 + 13) / 8192
        = 28 / 8192 = 0.003418
```

So p ≈ 0.0034 (McNemar exact). The discrepancy is not sampling noise; it is a systematic property of the simulator's fidelity on these scenarios. This matters for interpretation: a significant discrepancy cannot be dismissed as "small sample", so it must be *carried forward as a correction* rather than hoped away.

### 6.5 Step 5 — the corrected estimate and where the variance lives

**FORMULA [F-04] — additive sim-to-real bias correction with combined uncertainty**

```text
FORMULA: theta_hat = p_hat_sim + delta_hat
         SE(theta_hat) = sqrt( SE_sim^2 + SE_delta^2 )
WHERE: theta_hat field success [proportion]; delta_hat = bench minus sim
       on paired scenarios; SE_sim from the clean holdout.
CONSTRAINT: the bias delta is the same on the holdout population as on
            the paired population (exchangeability of the bias);
            independent estimation samples; additive not multiplicative.
NUMERICAL EXAMPLE: SE_sim = 0.005655, SE_delta = 0.028978
                   SE^2 = 0.0000319781 + 0.00083975 = 0.00087173
                   SE = 0.029526
                   theta_hat = 0.91240 - 0.091667 = 0.820733
                   95% interval = 0.820733 ± 1.96*0.029526
                                = 0.820733 ± 0.057871
                                = [0.762862, 0.878604]
INTERPRETATION: point estimate 0.821, interval [0.763, 0.879]. The
                interval is 5.8 pp wide, and 96.3% of its variance comes
                from 120 bench episodes.
```

The variance decomposition is the case's cost argument in one line:

```text
share_paired = SE_delta^2 / (SE_sim^2 + SE_delta^2)
             = 0.00083975 / 0.00087173
             = 0.96332
```

Buying precision on the wrong term is the classic allocation error in simulation-heavy programmes. The following table, constructed from the formulas above, shows the marginal return of each kind of episode.

| Change | New SE | New 95% half-width | Change in half-width | Incremental cost (illustrative) |
|---|---:|---:|---:|---:|
| Baseline: 2,500 sim, 120 paired | 0.02952 | 0.05787 | — | already spent |
| Double sim holdout to 5,000 | 0.02925 | 0.05733 | −0.05 pp | $75 |
| Quadruple sim holdout to 10,000 | 0.02912 | 0.05707 | −0.08 pp | $225 |
| Paired 120 → 240 | 0.02126 | 0.04166 | −1.62 pp | $6,960 |
| Paired 120 → 420 | 0.01649 | 0.03232 | −2.55 pp | $17,400 |
| Both: paired 420 and sim 10,000 | 0.01575 | 0.03086 | −2.70 pp | $17,625 |

The arithmetic for the paired rows scales the empirical `s_d` as 1/sqrt(n_paired), holding the per-episode discrepancy variance fixed:

```text
SE_delta(240) = 0.028977 * sqrt(120/240) = 0.028977 * 0.707107 = 0.020490
SE_total(240) = sqrt(0.00003197 + 0.00041984) = sqrt(0.00045181)
              = 0.021255
SE_delta(420) = 0.028977 * sqrt(120/420) = 0.028977 * 0.534522 = 0.015489
SE_total(420) = sqrt(0.00003197 + 0.00023991) = sqrt(0.00027188)
              = 0.016489
```

An alternative estimate of the same 420-paired figure uses the discordance-count approximation, which assumes the discordance *rate* stays constant at 0.10833 per episode:

```text
SE_delta_approx(n) = sqrt(0.10833/n)
SE_delta_approx(420) = sqrt(0.000257929) = 0.016060
SE_total_approx(420) = sqrt(0.00003197 + 0.00025793) = 0.017027
```

The two differ by 3.3% (0.01649 versus 0.01703), and the honest statement is that the interval narrows to roughly ±0.032, not to a precise value. The gate outcome does not depend on that difference, which is itself a useful check: a decision that survives a 3% change in its standard error is robust to the modelling choice.

### 6.6 Step 6 — why the discrepancy is a lower bound

The 120 paired episodes were run in the central, dry aisles, because those are the aisles that could be closed for two days. Survey SRC-05 estimates that these aisles constitute 46% of zone aisle length; the remaining 54% includes the chilled dock approach, two resin-coated runs, and the outbound staging cross-aisles where pallets are double-stacked.

The consequence is directional and can be stated without new measurement: if the simulation's error grows as geometry and contact conditions get harder, and the paired sample was drawn from the easy subset, then the measured discrepancy understates the discrepancy on the harder subset. Therefore:

```text
delta_true(field mix) <= delta_measured(easy subset)      [in magnitude]
|delta_true| >= 0.0917
```

This is an inequality, not an estimate, and it is the reason the corrected interval's *upper* end should not be quoted as a best case. A team that reports "field success is between 76% and 88%" is technically correct and practically misleading: the mass of the plausible distribution sits below the point estimate, not around it.

**How bad could it be?** A crude bound uses the per-family discrepancy. On the four families the bench could represent, the measured discrepancies were:

| Family | Pairs | Sim success | Bench success | Discrepancy |
|---|---:|---:|---:|---:|
| Straight dry aisle, no obstruction | 42 | 0.952 | 0.905 | −0.0476 |
| Single corner, static target | 38 | 0.895 | 0.789 | −0.1053 |
| Narrow gap, 0.55 m clearance | 24 | 0.833 | 0.667 | −0.1667 |
| Moving target on rail | 16 | 0.813 | 0.625 | −0.1875 |
| **All** | **120** | **0.883** | **0.792** | **−0.0917** |

The discrepancy is monotone in difficulty across the four families the bench *did* cover. Extrapolating that monotonicity to the uncovered families is an assumption, but a plausible one, and it implies the field-mix discrepancy exceeds 0.0917. If the uncovered families behave like an extension of the trend, their discrepancy is on the order of 0.20 to 0.25. That statement is labelled ILLUSTRATIVE because no paired data exist for them; it is used to size the retraining priority, not to compute the gate.

### 6.7 Step 7 — the coverage gap in friction

The domain randomisation used during training drew friction coefficient μ from a uniform range [0.55, 0.85]. The floor survey's 46 points give a zone distribution with mean 0.68 and a lower tail reaching 0.45 on two resin-coated runs near the chilled dock, which together carry an estimated 22% of missions.

Standardising against the training range's half-width h = (0.85 − 0.55)/2 = 0.15:

```text
d_friction = (mu_train_min - mu_field_min) / h
           = (0.55 - 0.45) / 0.15
           = 0.6667  half-widths outside the training support
```

For overhang, training drew from [0.00, 0.10] m, half-width 0.05 m, and inbound pallets reach 0.12 m:

```text
d_overhang = (0.12 - 0.10) / 0.05 = 0.40
```

Treating the two axes as approximately independent:

```text
d_combined = sqrt(d_friction^2 + d_overhang^2)
           = sqrt(0.44444 + 0.16000)
           = sqrt(0.60444) = 0.77746
```

**FORMULA [F-05] — first-order performance bound outside the training support**

```text
FORMULA: theta(x_out) >= theta(x_edge) - L * ||x_out - x_edge||
WHERE: theta success [proportion]; L Lipschitz constant of success with
       respect to the normalised condition vector [proportion per unit];
       distance measured in training-range half-widths.
CONSTRAINT: theta is L-Lipschitz on the path from support edge to the
            field point — an assumption, not a theorem, and unverifiable
            without data along that path; L estimated from in-support data.
NUMERICAL EXAMPLE: L estimated at 0.16 per half-width (from the family
                   table's slope in clearance/difficulty, see 6.8);
                   theta(x_edge) = 0.79 on the hardest supported family;
                   distance = 0.7775
                   lower bound = 0.79 - 0.16*0.7775
                               = 0.79 - 0.12440 = 0.66560
INTERPRETATION: a defensible floor for the uncovered 22% of missions is
                about 0.67, against a supported-population value near 0.85.
```

The Lipschitz constant deserves scepticism. A first-order bound with an estimated slope is not a guarantee; it is a structured way of saying "we expect roughly this much degradation, and here is the number we would have to believe to claim otherwise". The value L = 0.16 per half-width is fitted from the in-support relationship between success and clearance margin, which is a proxy for difficulty, not the same axis as friction. That substitution is the weakest link in the chain and is stated as such.

### 6.8 Step 8 — the traffic-mix mixture

The gate population is the zone's mission mix, not the holdout's. Splitting the corrected holdout by whether the episode's conditions fall inside the training support gives:

| Segment | Mission share w | Corrected success | Source of estimate |
|---|---:|---:|---|
| In-support aisles (μ ≥ 0.55, overhang ≤ 0.10) | 0.78 | 0.845 | Clean holdout + paired correction, stratified |
| Out-of-support aisles (μ < 0.55 or overhang > 0.10) | 0.22 | 0.667 | Bench low-friction subset, 24 episodes, no sim counterpart |

```text
theta_mix = w_in * theta_in + w_out * theta_out
          = 0.78 * 0.845 + 0.22 * 0.66667
          = 0.65910 + 0.14667
          = 0.80577
```

The in-support figure of 0.845 differs from the pooled 0.8205 because the pooled number was dragged down by episodes that would have been in the out-of-support segment anyway; the stratification separates them so each gets its own evidence source. This is the point of stratifying: a pooled average over a mixture with heterogeneous evidence quality assigns the worst-evidenced segment the same credibility as the best.

Uncertainty must also be mixed, and the mixing is what makes the interval widen:

```text
SE_in  = 0.02953        (dominated by the paired discrepancy)
SE_out = sqrt(0.66667*0.33333/24) = sqrt(0.0092593) = 0.096229
        (24 bench episodes only — no simulation counterpart exists)

SE_mix = sqrt( (0.78^2)(SE_in^2) + (0.22^2)(SE_out^2) )
       = sqrt( 0.6084*0.00087173 + 0.0484*0.0092593 )
       = sqrt( 0.00053039 + 0.00044815 )
       = sqrt( 0.00097854 )
       = 0.031282

95% interval = 0.80577 ± 1.96*0.031282
             = 0.80577 ± 0.061313
             = [0.74446, 0.86708]
```

**FORMULA [F-06] — mixture mean and variance under segment weights**

```text
FORMULA: theta_mix = sum_j w_j * theta_j
         Var(theta_mix) = sum_j w_j^2 * Var(theta_j)
WHERE: w_j segment mission shares summing to 1 [dimensionless];
       theta_j segment success [proportion]; segments independent.
CONSTRAINT: weights known without relevant error (here from 6 weeks of
            mission logs, treated as known); segment estimates
            independent; no covariance term from shared paired episodes.
NUMERICAL EXAMPLE: w = (0.78, 0.22), theta = (0.845, 0.66667),
                   Var = (0.00087173, 0.0092593)
                   theta_mix = 0.80577
                   Var = 0.6084*0.00087173 + 0.0484*0.0092593
                       = 0.00097854 ; SE = 0.031282
INTERPRETATION: the 22% out-of-support segment contributes 45.8% of the
                mixture variance despite being less than a quarter of
                traffic.
```

The variance-share inversion is the actionable output: reducing exposure in the worst-evidenced fifth of the zone would do more for decision confidence than any amount of work on the other four-fifths. That is not intuitive from a traffic-weighted view, where the small segment looks negligible.

### 6.9 Step 9 — the gate decision

The threshold was preregistered as: promote only if the lower bound of the 95% interval on field episode success is at least 0.780. That number was chosen as the incumbent's bench-measured success rate of 0.79 minus one percentage point of practical indifference, and it was fixed before the rebuilt holdout was evaluated.

```text
lower bound = 0.74446
threshold   = 0.78000
shortfall   = 0.78000 - 0.74446 = 0.03554  (3.55 pp)
```

**Decision: fail.** The point estimate 0.80577 clears the threshold by 2.6 pp; the bound misses it by 3.6 pp. The distance between those two statements is the entire value of using a bound.

Note how sensitive the verdict is to the *form* of the rule and not only to the number. A rule of "point estimate ≥ 0.78" passes; a rule of "lower bound ≥ 0.78" fails. A rule of "lower bound ≥ 0.74" passes. Section 6.13 tabulates that sensitivity, and it is the reason the threshold and its form were fixed and recorded in advance.

### 6.10 Steps 10–12 — can the field trial answer the remaining questions?

This is where a team that has done the statistics well can still make a systems mistake: assuming that because a field trial is the most realistic evidence, it is the most informative evidence. For rare events it is the *least* informative per unit cost.

**Exposure.** Distance travelled during the planned trial:

```text
D = n_robots * hours_per_day * days_per_week * weeks * v * duty
  = 2 * 6 * 5 * 4 * 0.9 * 0.62
  = 2 * 6 * 5 * 4 * 0.558
  = 240 * 0.558 * 3600
```

Working it in consistent units avoids the common hours/seconds slip:

```text
seconds = 2 * 6 * 5 * 4 * 3600 = 864,000 s
distance = 864,000 * 0.9 * 0.62 = 864,000 * 0.558 = 482,112 m
```

So the trial buys 482,112 robot-metres, or 482.1 km, or 120,528 m per robot.

**FORMULA [F-07] — upper confidence bound on a rate from zero observed events**

```text
FORMULA: lambda_upper = -ln(alpha) / X
WHERE: lambda_upper rate bound [events per unit exposure]; alpha
       one-sided tail probability; X exposure in units of the rate's
       denominator; zero events observed.
CONSTRAINT: events are Poisson with constant rate; the exposure is
            homogeneous (a single operating condition); zero events.
            With k > 0 events use the gamma/chi-square form instead.
NUMERICAL EXAMPLE: alpha = 0.05, -ln(0.05) = 2.99573
                   contact limit 1 per 200,000 m; X = 482,112/200,000
                                                = 2.41056 units
                   lambda_upper = 2.99573 / 2.41056 = 1.24267
INTERPRETATION: if the trial records zero contacts, the 95% upper bound
                is 1.24 times the permitted rate. The trial cannot certify
                the contact gate even in the best case.
```

Because −ln(0.05) ≈ 3, the working form is the familiar "rule of three": zero events in X units of exposure bounds the rate at about 3/X. Its consequence for sizing is immediate:

```text
X_required = -ln(0.05) / lambda_target = 2.99573 / 1.0 = 2.99573 units
D_required = 2.99573 * 200,000 = 599,146 m  (≈ 599,200 m)
weeks_required = 599,146 / 120,528 = 4.971 weeks
```

So a trial that could certify the contact gate at zero events needs **five weeks, not four** — and that is the optimistic case, because a single event at five weeks puts the bound at roughly 2.6 per 200,000 m and the gate is then unresolvable by more exposure of the same kind. The correct reading is not "add a week" but "a field trial is the wrong instrument for this gate"; contact-rate evidence has to come from a combination of bench measurement at validated fidelity, the runtime monitor's veto behaviour under fault injection, and a much longer low-exposure operating period.

The near-miss gate behaves differently because its denominator is four times smaller:

```text
X_nearmiss = 482,112 / 50,000 = 9.64224 units
lambda_upper = 2.99573 / 9.64224 = 0.31069
```

A four-week trial with zero near-misses bounds the near-miss rate at 0.31 times the limit — comfortably inside. The asymmetry is stark and it is the substance of the decision: **the same trial is decisive for one gate and silent for the other**, purely because one event is four times rarer.

| Gate | Limit | Trial units of exposure | Best-case upper bound | Decisive? |
|---|---:|---:|---:|---|
| Near-miss | 1 / 50,000 m | 9.642 | 0.311 × limit | Yes |
| Contact | 1 / 200,000 m | 2.411 | 1.243 × limit | No |
| Contact, 5-week trial | 1 / 200,000 m | 3.013 | 0.994 × limit | Barely, at zero events |
| Contact, 12-week trial | 1 / 200,000 m | 7.231 | 0.414 × limit | Yes |

### 6.11 The definition sensitivity that flips the gate

The near-miss definition's clearance threshold was proposed at 0.25 m. If it were set at 0.15 m, the same trial data would produce a different count and a different verdict, and the choice is a judgement about what "close" means rather than a measurement. The case's control is to freeze the definition and report both.

To make the mechanism visible without inventing field events, consider the bench data's clearance distribution: of 120 episodes, 9 reached a minimum clearance below 0.25 m and 3 below 0.15 m. Those are static-target clearances, not near-misses — the gate's population requires a person or moving asset — so they cannot be used as gate evidence. They can be used to show the *ratio*: narrowing the threshold from 0.25 m to 0.15 m removes two-thirds of the counted events. If the field behaves similarly, a threshold change of 10 cm moves the measured rate by a factor of three, which is larger than the entire width of the confidence interval.

The bounded fix is procedural and cheap: freeze the threshold, the relative-speed condition, and the counting rule in a versioned record before the first trial shift, and require a new decision cycle to change them. Any change after exposure begins invalidates the gate rather than adjusting it.

### 6.12 Cost of the evidence programme

Costs are in US dollars at planning rates, 2026. All are ILLUSTRATIVE.

| Item | Unit cost | Quantity | Cost |
|---|---:|---:|---:|
| Simulation episode (0.9 min batched on 1 GPU at $2.00/h) | $0.03 | 25,000 | $750 |
| Paired bench episode (0.4 h engineer at $85/h + rig $18 + targets $6) | $58 | 300 | $17,400 |
| Field trial week (see breakdown below) | $5,020 | 5 | $25,100 |
| Safety and review engineering (40 h at $85/h) | $85/h | 40 | $3,400 |
| **Programme total** | | | **$46,650** |

Field trial week breakdown:

```text
blocked-aisle productivity loss = 2 robots * 6 h/day * 5 d * $42/h = $2,520
operator standby                = 0.25 FTE * 40 h * $55/h          = $550
safety observer                 = 6 h/day * 5 d * $65/h           = $1,950
weekly total                                                      = $5,020
```

**FORMULA [F-08] — cost per robot-metre of evidence**

```text
FORMULA: c_evidence = C_total / (weeks * n_robots * hours_per_day
                                 * days_per_week * 3600 * v * duty)
WHERE: C_total currency; denominator robot-metres [m]; result
       currency per metre.
CONSTRAINT: all trial cost attributed to the evidence purpose; no
            production value credited inside the trial window.
NUMERICAL EXAMPLE: C = 25,100, weeks = 5
                   metres = 5 * 120,528 = 602,640
                   c = 25,100 / 602,640 = $0.041650 per robot-metre
INTERPRETATION: roughly 4.2 cents per metre of field evidence, which is
                $12.28 per 295 m mission against $58 per paired bench
                episode — the field is cheaper per episode and far worse
                per controlled condition.
```

The per-mission figure is worth stating precisely because it inverts the usual intuition: field time looks expensive per hour and is cheap per episode, since the robot is doing real work anyway. It is expensive per *controlled observation*, because the operator cannot choose which condition occurs next.

```text
field cost per mission = 25,100 / (602,640 / 295) = 25,100 / 2,042.8
                       = $12.287 per mission
```

The comparison to the benefit is sobering and belongs in the decision record. Baseline interventions run at 4.3 per 100 missions with a mean mission of 295 m; a candidate at 2.6 per 100 would save 1.7 interventions per 100 missions, or 1.7 per 29,500 m of travel.

```text
saving_per_metre = 1.7 * $18.00 / 29,500 = $30.60 / 29,500
                 = $0.0010373 per robot-metre
```

At fleet scale — 14 robots, 6 h/day, 300 days/year — annual distance is:

```text
D_year = 14 * 6 * 3600 * 300 * 0.9 * 0.62
       = 14 * 21600 * 300 * 0.558
       = 14 * 6,480,000 * 0.558 = 14 * 3,615,840 = 50,621,760 m
```

so the labour saving is 50,621,760 × $0.0010373 ≈ $52,520 per year, and the recovered capacity (535 robot-hours per year, computed in Section 9.6) is worth $21,400 at $40 per robot-hour of marginal capacity. Against a $46,650 programme, the payback is under a year on those two components alone. The economics therefore justify doing the validation properly; they do not justify skipping it, because the downside being insured against is a physical event, and its cost is not in this table.

### 6.13 Sensitivity and robustness

| Stress or test | Original condition | Alternative | Change in conclusion | Limitation | Decision consequence |
|---|---|---|---|---|---|
| Paired set size | 120 episodes | 420 episodes | Mixture SE falls to 0.02477; lower bound 0.7572 — still below 0.780 | Discordance rate assumed constant | Hold survives; shortfall narrows from 3.6 to 2.3 pp |
| Gate form | Lower bound ≥ 0.780 | Point estimate ≥ 0.780 | Passes at 0.8058 | Ignores estimation error | Rejected as unsound, not as inconvenient |
| Gate level | 95% | 90% (z = 1.645) | Lower bound 0.7543 — still fails | One-sided vs two-sided convention | Hold survives |
| Threshold | 0.780 | 0.750 | Lower bound 0.7445 — still fails, by 0.6 pp | Threshold is a policy input | Hold survives narrowly |
| Threshold | 0.780 | 0.740 | Passes | — | The knife-edge is at 0.7445 |
| Traffic mix | 22% out-of-support | 35% out-of-support | theta_mix 0.7826, SE 0.03876, lb 0.7066 | Weight from 6 weeks | Hold strengthens |
| Traffic mix | 22% out-of-support | 10% out-of-support | theta_mix 0.8272, SE 0.02826, lb 0.7718 — still fails | — | Hold survives |
| Discrepancy | −0.0917 | −0.060 (a better simulator) | theta_mix 0.8305, lb 0.7692 — still fails, by 1.1 pp | Optimistic by construction | Hold survives |
| Discrepancy | −0.0917 | −0.045 | theta_mix 0.8422, lb 0.7808 — **passes** | Optimistic by construction | The flip point is about −0.046 |
| Duty factor | 0.62 | 0.55 | Trial exposure 427,680 m; 5 weeks gives 2.138 units; contact gate still closed | Affects exposure only | No change |
| Wheel scale error | 0% | +5% over-read | Reported distance 5% high, rates 5% low | A-05 unverified | Gates get *easier*; must be corrected |

The pattern in the table is what a robust decision looks like: eight of eleven variations preserve the verdict, and the three that overturn it do so by relaxing the *rule* (point estimate instead of bound), moving the *policy threshold* to just below the observed bound, or asserting a materially better simulator than the paired study measured.

The single most consequential sensitivity is the discrepancy magnitude, and the flip point can be solved rather than searched. Requiring the mixture lower bound to reach 0.780:

```text
theta_out fixed at 0.66667, all other terms fixed
target lb: 0.78 = theta_mix - 1.96 * 0.031281 (approx, SE held fixed)
theta_mix = 0.78 + 0.061313 = 0.841313
0.78 * theta_in + 0.22 * 0.66667 = 0.841313
0.78 * theta_in = 0.841313 - 0.146667 = 0.694646
theta_in = 0.890572
required delta = 0.890572 - 0.912400 = -0.021828
```

Holding the SE fixed, the discrepancy could be as small as −0.022 and still pass — smaller in magnitude than the measured −0.092 by a factor of four. That is a much looser flip point than the tabulated −0.046 (which recomputes the SE from the tabulated point estimates and therefore moves other terms too). The honest reading: this gate is sensitive to the measured discrepancy at roughly the one-standard-error scale (SE = 0.029), so the verdict is *not* robust to a plausible improvement in simulator fidelity. That is a finding, not a flaw in the analysis — it says the candidate is close to the threshold and the decision should be revisited rather than fought over. A team that "found" a slightly better simulator would pass, and that is precisely why the paired study must be executed on the hardest available conditions rather than the most convenient ones: a two-point change in the measured discrepancy, entirely achievable by choosing easier bench scenarios, would flip the decision.

### 6.14 What was tried first, and why it was rejected

The first pass compared the original sweep's 0.938 against a 0.90 threshold and recommended promotion. It was rejected for three reasons, in order of importance:

1. **Leakage.** 13% of the evaluation seeds were in the training manifest. The audit is a set intersection and takes minutes; its absence was a process gap, not an analytical difficulty.
2. **Wrong comparison.** 0.938 was a simulation number compared against a threshold that had been derived from a bench number. Mixing populations in a single inequality is the most common form of invalid comparison in this domain.
3. **No exposure reasoning.** The recommendation treated "run a trial and see" as a validation step without checking whether the trial's 482 km could resolve a 1-in-200,000-m event. It could not, so the trial would have produced a confident-sounding null result.

The third point is the one that generalises: a test that cannot fail is not a test, and a field trial sized for throughput questions cannot answer safety questions.

> **Takeaway:** the candidate fails on the bound, not on the point; the field trial can close the near-miss gate but not the contact gate; and every sensitivity that preserves the verdict is a change in the rule or the threshold, never in the evidence.

## 7. The Finding

The recommendation is to **hold** `nav-v7` at the controlled-facility stage and not run the booked four-week field trial, because the candidate does not clear the preregistered performance gate on an interval that accounts for simulator optimism, and because the trial as scoped could not clear the contact gate even if it recorded no contacts at all.

The supported magnitude is a field episode success of 0.806 with a 95% interval of [0.744, 0.867]; the preregistered requirement is a lower bound of at least 0.780, so the evidence misses by 3.55 percentage points. That estimate is not a measurement of the candidate on the floor — it is a corrected simulation estimate whose correction comes from 120 paired bench episodes on the easiest 46% of aisle length. The interval is honest about the paired sample's size but not about its composition, so the true uncertainty is wider than 6.1 points and the plausible mass sits below the point estimate rather than around it.

The main boundary is that this holds nothing about the simulator's fidelity for the corners it never generated: the 22% of missions in low-friction aisles carry 45.8% of the mixture variance and have no simulation counterpart at all, so no amount of additional simulation would resolve them. The next justified action is a stratified paired programme — 300 further paired episodes drawn by aisle class rather than convenience — plus domain-randomisation extension to μ ∈ [0.42, 0.88] and overhang to 0.14 m, followed by a fresh decision cycle on a rebuilt holdout; the contact gate is then addressed through bench fidelity validation and monitor fault injection rather than through field exposure.

## 8. System/Solution Design

### 8a. Offline / Training and Evidence Path

The offline path produces the artefact and the evidence, and it is where the gate is actually decided.

```text
[Scenario sampler v3.2.1]
        |
        v
[Batch simulator: 2,500 episodes, physics dt 10 ms, scan 10 Hz]
        |
        +--> [Ground-truth clearance, success, path] --+
        |                                              |
[Policy checkpoint nav-v7]<---+                        v
        |                                        [Holdout metrics]
        |                                              |
        v                                              v
[Paired harness] --+                          [Paired discrepancy delta]
        |          |                                   |
        v          v                                   v
[Sim replay]  [Bench rig 6x4 m]              [Bias correction F-04]
                    |                                  |
                    v                                  v
            [Motion-capture ground truth]     [Traffic-mix weighting F-06]
                                                       |
                                                       v
                                              [Gate: lower bound vs 0.780]
                                                       |
                                              +--------+--------+
                                              v                 v
                                        [PROMOTE]            [HOLD]
                                              |                 |
                                              v                 v
                                      [Release bundle]   [Retrain queue]
```

Two properties of this path are deliberate and both were violated in the first pass.

**The holdout is generated once per decision cycle and frozen.** Regenerating it after seeing a result converts the gate into a search over holdouts. The frozen artefact carries a hash; any change requires a new decision cycle with a recorded reason.

**The paired study straddles the two worlds.** Its sampling frame is defined by the *field* population — aisle classes weighted by mission share — not by what is convenient to block off. In this case it was defined by convenience, which is the specific defect Section 6.6 quantifies.

Batch sizing: 2,500 episodes at 0.9 minutes of GPU time each is 2,250 GPU-minutes, or 37.5 GPU-hours, or about 1.6 hours on 24 concurrent workers. That is small enough that episode count is not the constraint; the bench is.

### 8b. Online / Serving and Control Path

The proposed on-robot path for the trial. Every element is **proposed**; none was deployed at the decision date.

```text
[2-D scan 10 Hz] --+
[wheel odometry] --+--> [Localisation filter] --> [Pose + covariance]
[IMU] -------------+                                    |
                                                       v
                                          [Policy nav-v7 inference, 20 Hz]
                                                       |
                                                       v
                                          [Action: v, omega]
                                                       |
                                    +------------------+------------------+
                                    |                                     |
                                    v                                     v
                        [Runtime monitor]                     [Deadline check]
                        clearance floor 0.18 m                inference < 45 ms
                        staleness > 120 ms                    (2 cycles at 20 Hz)
                                    |                                     |
                            +-------+-------+                    +--------+
                            v               v                    v
                      [VETO -> zero]   [PASS]              [MISS -> zero]
                            |               |                    |
                            +-------+-------+--------------------+
                                    |
                                    v
                        [Safety-rated chain: lidar field, bumper, e-stop]
                                    |
                            +-------+-------+
                            v               v
                     [Motion enabled]   [Protective stop]
                                    |
                                    v
                        [Telemetry: command, veto reason, pose, scan hash]
                                    |
                                    v
                        [Operator queue on 3 consecutive vetoes]
```

**Success path.** A scan arrives with timestamp t; localisation produces a pose with covariance; the policy runs within 41 ms of the scan's arrival; the monitor computes predicted clearance over a 0.8 s horizon and finds a minimum of 0.34 m, above the 0.18 m floor; the command is passed to the drive; telemetry records the command and monitor margin at 5 Hz (sampled, see A-08).

**Failed path — stale perception.** The scan publisher misses two cycles. The monitor sees the newest scan timestamp is 130 ms old, above the 120 ms bound, and vetoes by commanding zero velocity. The robot decelerates at its configured 0.6 m/s² limit. The veto increments a counter; at three consecutive vetoes the robot raises an operator alert with reason `STALE_PERCEPTION` and parks. It does not resume autonomously, because resuming on a stale-then-fresh scan would require reasoning about what moved during the gap, and that reasoning is out of scope for the monitor.

**Failed path — inference deadline.** The policy exceeds 45 ms on one cycle. The deadline check emits zero for that cycle and increments a miss counter. Two consecutive misses raise `INFERENCE_SLOW`. The distinction between this and the stale-perception path matters for diagnosis: a slow inference with fresh perception is a compute problem, and a stale scan with fast inference is a transport problem. Conflating them in one alert is a common design error that makes both harder to fix.

**Failed path — monitor failure.** If the monitor itself stops publishing, a watchdog holds the last command for at most 60 ms and then commands zero. The monitor is on the critical path and must fail closed; a monitor that fails open converts a safety feature into a single point of failure.

**Cold start.** First scan after boot has no prior pose; the localisation filter initialises from a fiducial observation. If no fiducial is visible, the robot does not move. This is the only path where the robot is deliberately immobile on a fresh boot, and it must be visible to the operator as such rather than as a hang.

**Peak load.** Two robots in one zone impose no compute contention (each runs its own policy on its own SoC), but they do contend for aisle space and for the map server. The map server is read-only during operation and serves 2 requests per robot per boot; contention is not credible here.

### 8c. Key Design Decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| DEC-01 | Gate on the lower confidence bound, not the point estimate | Point estimate; best-of-3 runs | Point estimate passes a policy that may be at 0.76 | Some genuinely good policies get held | When the interval is narrow relative to threshold distance | Threshold renegotiation, not gate-form change |
| DEC-02 | Additive sim-to-real correction rather than a learned residual model | Fit a model mapping sim features to bench outcome | 120 pairs cannot support a model with more than ~10 parameters | Additivity is false in general | Paired set unrepresentative (already true) | ≥600 paired episodes stratified by class |
| DEC-03 | Separate contact and near-miss gates with independent exposure sizing | One combined safety gate | The two rates differ fourfold, so one trial cannot resolve both | Two decision paths, more bookkeeping | Either limit changes in the safety case | Safety case revision |
| DEC-04 | Runtime monitor with clearance floor and staleness bound | Policy-only; policy plus a post-hoc log review | Policy cannot be shown to be safe by evaluation alone | Monitor can veto good manoeuvres and cut throughput | Floor set below actual footprint uncertainty | Footprint or sensor change |
| DEC-05 | Veto fails closed to zero velocity | Fail open with alert; fail to last command | Fail-open converts a safety feature into a hazard | Robot may stop more often than strictly needed | Watchdog latency exceeds 60 ms | Compute or bus change |
| DEC-06 | Freeze near-miss definition before exposure | Define after seeing distributions | Post-hoc thresholds let the data choose the gate | Some events classified by an arbitrary line | Definition becomes unworkable in practice | New sensor modality |
| DEC-07 | Trial scoped to off-peak, no-picker aisles for weeks 1–2 | Mixed traffic from day one | Exposure to people before performance gate passes | Understates human risk in early data | First two weeks' data cannot speak to picker interaction | Week 3 expansion review |
| DEC-08 | Rollback rehearsal before promotion | Rollback procedure documented only | Untested rollback has unknown duration | Costs one shift of engineering | Rollback needs a map or firmware change | Fleet platform version change |

### 8d. Architecture Artefacts and Operational Readiness

| Area | Design position for this case |
|---|---|
| Interfaces | Policy consumes `(scan[720], pose[3], cov[9], goal[3], prior_cmd[2])` at 20 Hz and emits `(v, omega)`; monitor consumes the same scan plus the command and emits `(permit: bool, margin: m, reason: enum)`. Command schema versioned; unknown reason codes are treated as veto by the drive interface |
| Data/state | Map is read-only and hash-pinned per deployment; policy checkpoint hash recorded in telemetry at boot; monitor version recorded alongside; no learned state persists across missions |
| Reliability | SLI: fraction of missions completed without operator intervention, target to be set from baseline 0.957; measured on-robot, 7-day window, owned by fleet ops |
| Resilience | Inference timeout 45 ms; monitor watchdog 60 ms; three consecutive vetoes to operator; no autonomous retry after a `STALE_PERCEPTION` park |
| Recovery | Rollback = stop robots, reflash previous bundle, re-run map hash check, resume; target one shift; rehearsed on one robot before promotion, otherwise unverified |
| Observability | Per-cycle command, veto reason, monitor margin, inference time; per-mission intervention reason code; daily contact/near-miss review with safety engineer present |
| Deployment | Policy, monitor, and map hashed as one bundle; canary on one robot for 3 shifts; second robot only after canary review |
| Security | Telemetry signed; remote commands require operator identity; policy artefact signed and hash-verified at load; no remote model update |
| Economics | Trial cost $5,020/week; evidence cost $0.042/m; benefit $0.00104/m at pilot intervention rates (Section 6.12) |
| Human workflow | Operator can halt; site manager can halt; safety engineer holds veto over promotion and over continued exposure; all three authorities recorded with the trial |

**Sizing, worked with units.** The compute envelope for one robot:

```text
policy inference: 720-beam scan -> 3 conv layers -> 128-d embedding -> 2 outputs
measured planning figure: 11 ms mean, 34 ms p99 (ILLUSTRATIVE, not measured here)
budget at 20 Hz: 50 ms per cycle
headroom at p99: 50 - 34 = 16 ms
monitor: 3 ms p99 (asserted target, A-07 unverified)
total p99 after monitor: 37 ms -> 13 ms headroom, 26% of cycle

deadline misses if policy p99 grows to 48 ms: monitor pushes to 51 ms
-> 1 miss in 20 cycles at p99 -> unusable; hence DEC-05's zero-command
```

Note what is *not* claimed: the sum of two p99 values is not the p99 of the sum. The 37 ms figure is an arithmetic addition labelled as such; the true p99 of the sum requires the joint distribution and is unmeasured. It is used only to show that the headroom is thin enough that a 14 ms regression would be fatal, which is a valid qualitative conclusion from an invalid quantitative premise only if the premise is conservative — and it is not established that it is.

> **Takeaway:** the offline path decides the gate, the online path enforces the safety envelope the gate cannot certify, and the monitor's fail-closed behaviour is what makes an evaluated-but-not-certified policy permissible at all.

## 9. Evaluation & Validation

### 9.1 Metric contract

| Field | Primary metric | Guardrail 1 | Guardrail 2 |
|---|---|---|---|
| Name | Field episode success | Contact rate | Intervention rate |
| Definition | Episodes reaching goal within 90 s, no clearance < 0.20 m, no takeover, divided by episodes started | Contacts per robot-metre | Missions with ≥1 operator action per 100 missions started |
| Numerator | Successful episodes | Contact events | Missions with an intervention |
| Denominator | Episodes started | Robot-metres travelled | Missions started |
| Unit | Proportion | events / m (reported per 10⁵ m) | per 100 missions |
| Population | Field-matched holdout, corrected to field | All trial exposure | All trial missions |
| Aggregation | Mission-share weighted over segments | Sum events / sum metres | Sum missions with intervention / sum missions |
| Window | One episode | Whole trial | Whole trial |
| Comparator | Incumbent classical planner, same population | Safety-case limit | Incumbent 4.3/100 |
| Threshold | Lower 95% bound ≥ 0.780 | ≤ 1 per 200,000 m | ≤ 3.5 per 100 |
| Status | **Executed, failed** | **Not executed — exposure insufficient** | Not executed — requires field |

### 9.2 Intended versus executed protocol

| Element | Intended | Executed | Deviation and effect |
|---|---|---|---|
| Holdout seeds | Disjoint from training | Overlapping by 13% in the original run | Inflation 2.6 pp; rebuilt |
| Paired sampling | Stratified by aisle class, proportional to mission share | Convenience sample, central dry aisles | Discrepancy biased low; correction too small |
| Bench conditions | Include low friction | Dry aisles only; low-friction run 24 episodes on a separate day | μ = 0.45 not represented in the primary paired set |
| Monitoring | Telemetry at 10 Hz | Sampled at 5 Hz to hold timing (A-08) | Veto-reason counts still complete; margins coarser |
| Intervention definition | Preregistered before exposure | Preregistered, then clarified on day 9 for multi-attendance missions | Clarification recorded; baseline recomputed under both rules (+14% under attendance counting) |
| Randomisation | Not applicable — no A/B in this decision | None | Comparison is against historical baseline, not concurrent control |
| Contact gate | Test during trial | Not testable at this exposure | Decision taken on exposure arithmetic before trial |

The last row is the case's most important procedural point: the team did the exposure arithmetic *before* running the trial, not after. Had they run it after, they would have had four weeks of clean data, zero contacts, and a plausible-sounding claim that the contact rate was acceptable.

### 9.3 Statistical method and its validity

**Interval for proportions.** Wilson, not Wald, because at the lower end of the plausible range (`p` near 0.74, `n` effectively 120 for the paired component) Wald under-covers and can extend below zero for small counts. The Wilson interval is used for the segment estimates; the mixture interval is computed by linearising the variance (F-06) rather than by resampling, because the two segments have separate evidence sources and a naive bootstrap over pooled episodes would misrepresent how the out-of-support estimate was obtained.

**Paired comparison.** McNemar exact, because the discordant count is 13 and the asymptotic chi-square approximation is unreliable below about 25 discordant pairs. The exact form is used.

**Rate bounds.** Poisson upper bound at zero events, with the homogeneity assumption stated. Homogeneity is violated in the trial because off-peak operation differs from peak; the bound is therefore a bound for *off-peak* exposure and cannot be extrapolated to peak traffic without a new exposure budget.

**What is not done.** No causal claim is made. The comparison against the historical baseline is observational: the classical planner ran in a different traffic mix, different season, and with different picker behaviour. Any throughput improvement observed later cannot be attributed to the policy without a concurrent control, which is one reason the trial was designed with a within-zone alternating schedule in the *promoted* case rather than a straight before/after.

### 9.4 Uncertainty and power

The dominant uncertainty is the paired discrepancy's standard error, 0.02898. To achieve a 95% half-width of 2.0 pp on the mixture estimate, the paired component alone must shrink to:

```text
target SE_total = 0.02/1.96 = 0.010204
required SE_delta^2 = 0.010204^2 - SE_sim^2 (mixed) 
                    = 0.00010412 - 0.00012064
```

The term is negative, which is the arithmetic's way of saying the target is unreachable at this simulator uncertainty: SE_sim alone (0.005655 in-support, but 0.0313 at mixture level because of the out-of-support segment's 0.0962) already exceeds the target. The binding constraint is not the paired count; it is the **absence of a simulation counterpart for the low-friction segment**. No amount of paired episodes fixes a segment whose only evidence is 24 bench runs. The correct intervention is to extend domain randomisation so that segment becomes in-support, which converts its evidence from 24 bench episodes to thousands of simulated plus a correction.

That reframing is the case's most transferable analytical move, and it came out of the arithmetic rather than out of intuition:

| Uncertainty source | Variance contribution at mixture level | Share | Fixable by more episodes? |
|---|---:|---:|---|
| In-support paired discrepancy | 0.00053039 | 54.2% | Yes — paired episodes |
| Out-of-support bench-only estimate | 0.00044815 | 45.8% | **No** — needs simulator support |
| In-support holdout sampling | 0.00002448 | 2.5% (after covariance with paired, net) | Marginally |

The shares sum past 100% because the in-support term already contains the holdout term (F-04 combines them); the display is a decomposition of the 0.00097854 total, where 0.00053039 + 0.00044815 = 0.00097854 exactly. The holdout's 0.00002448 is inside the 0.00053039, contributing 4.6% of that term and 2.5% of the total.

### 9.5 Validity threats

| Threat | Mechanism | Direction | Mitigation in this case | Residual risk |
|---|---|---|---|---|
| Seed leakage | Holdout shares scenarios with training | Optimistic | Set intersection, rebuild | Historical results already reported |
| Convenience paired sampling | Easy aisles only | Optimistic | Stratify next round | Not mitigated for this decision |
| Composition shift | Sampler changed between runs | Unknown, likely optimistic | Field-matched sampler | Unmeasured 1.7 pp |
| Definition drift | Near-miss threshold chosen post hoc | Either | Preregistration | Unworkable-definition risk |
| Denominator error | Odometry scale | Optimistic if over-reading | Fiducial correction | 1% unverified |
| Selection on clean sensing | 7 dropped bench episodes | Optimistic | Reported as exclusion | Small |
| Regression to the mean | Baseline selected *because* it looked bad | Pessimistic for baseline | Baseline is 6 months, not a bad week | Low |
| Simulator overfit to task | Policy exploits simulator artefacts | Optimistic | Paired study detects it | Detected, magnitude uncertain |

The "simulator overfit" row deserves a concrete mechanism rather than a label. A policy trained in simulation can exploit contact dynamics that the simulator models too generously: interpenetration tolerance, friction cone approximation, and a scan model that does not simulate specular dropout. The bench's scan does drop on shrink-wrap; the simulator's does not. That single difference produces a policy that has never learned to handle sparse returns, and it is visible in the paired family table as the largest discrepancy on the narrow-gap family, where sparse returns matter most.

### 9.6 Capacity and throughput sizing

The trial's productive capacity, and what the intervention rate does to it:

```text
available robot-hours per week = 2 robots * 6 h/day * 5 d = 60 h
baseline interventions/week = missions/week * 0.043
missions/week = metres/week / 295
metres/week = 2 * 6 * 5 * 3600 * 0.9 * 0.62 = 120,528 m
missions/week = 120,528 / 295 = 408.57
baseline interventions/week = 408.57 * 0.043 = 17.57
mean intervention duration (illustrative) = 6.5 min
hours lost/week = 17.57 * 6.5 / 60 = 1.903 h
fraction of capacity lost = 1.903 / 60 = 3.17%
```

At the candidate's *claimed* intervention rate of 2.6 per 100:

```text
candidate interventions/week = 408.57 * 0.026 = 10.62
hours lost/week = 10.62 * 6.5 / 60 = 1.150 h
recovered hours/week = 1.903 - 1.150 = 0.753 h
recovered hours/year at 50 weeks = 37.65 h for two robots
fleet-scaled (14 robots, 300 days): 535.2 h/year
```

535.2 robot-hours per year at $40 per marginal hour is $21,408, which matches the figure used in Section 6.12. The chain from intervention rate to currency is four steps long and each step is an assumption, which is exactly why the benefit claim is reported as ILLUSTRATIVE and the *safety* claim is not derived from it.

### 9.7 Decision

**Failed** on the primary gate; **not executed** on both rate gates, with the contact gate demonstrated unresolvable at the planned exposure. The verdict is a hold, with the specific next cycle defined in Section 10.

> **Takeaway:** the evaluation was designed to fail, and it did; the failure is traceable to a 13-event paired study and a missing simulator segment, not to the policy being obviously bad.

## 10. Action Taken

| Action field | Record |
|---|---|
| Recommendation | Hold the field trial; keep `nav-v7` at controlled-facility stage; run a stratified paired programme and extend domain randomisation before the next decision cycle |
| Approval / status | Recommended by the navigation lead and analytics; accepted by the director of warehouse autonomy; endorsed by the robotics safety engineer, dated 2026-11-06 in the decision log. This is a *fictional* record for the exercise |
| Implementer and handoff | Navigation team owns retraining and the DR extension; test engineering owns the stratified paired programme; fleet engineering cancels the blocked-aisle reservation; safety engineer owns the gate definitions |
| Effective scope | Zone-level hold only; no change to fleet, no change to classical planner |
| Release strategy | No deployment. The candidate remains loadable on the bench rig for continued study |
| Stop / rollback | Not applicable — nothing was released. The pre-existing rollback rehearsal requirement moves to the next cycle |
| Excluded actions | Field trial with modified exposure; monitor-only deployment without policy; fleet-wide comparison; threshold relaxation |
| Deferred | Contact-gate validation approach (bench fidelity + fault injection) remains OPEN — no method has been agreed that can certify 1 per 200,000 m without years of exposure |
| Cost avoided | $25,100 of trial cost; $0 of evidence cost (the paired programme is additive, not substitutive) |

Three aspects of the action are worth separating from the recommendation.

**What the director actually bought.** Not a decision to abandon the policy, and not a delay for its own sake. The hold purchases a *specific* piece of information — the discrepancy on hard aisles — that no amount of simulation can supply. Framing the hold as "more work" without naming the artefact would make it indistinguishable from schedule slippage.

**What was refused.** The most tempting alternative was a reduced trial: one robot, two weeks, no pickers. It was refused because the exposure would be 120,528 m, giving a contact bound of 4.97× the limit — an experiment guaranteed to produce an uninterpretable result while consuming the trial slot and the organisation's appetite for validation.

**What was escalated.** The contact-gate certification method is genuinely unresolved and was escalated rather than resolved inside this case. A gate that requires roughly 600,000 m of exposure per attempt is a gate that will be met by luck or by aggregation across many trials, and aggregating exposure across policy versions is a statistical error that this case does not attempt to solve.

## 11. Outcome & Results

**No field outcome exists, and none is claimed.** The candidate was not deployed. The only measured quantities in this case are the simulation and bench results already reported in Section 6, which are themselves ILLUSTRATIVE inputs.

| Metric | Baseline: value / window / N | Candidate: value / window / N | Absolute change | Relative change | Uncertainty | Evidence | Limitation |
|---|---|---|---|---|---|---|---|
| Field episode success | Not measured for classical planner as a proportion of "episodes" | 0.806 corrected estimate, 2,500 sim + 120 paired | Not comparable | Undefined — different populations | 95% CI [0.744, 0.867] | DERIVED from ILLUSTRATIVE inputs | Correction from easy aisles only |
| Simulation success (clean holdout) | — | 0.912, 2,500 episodes | — | — | [0.901, 0.923] | DERIVED | Simulator population, not field |
| Paired discrepancy | 0 (definition) | −0.0917, 120 pairs | −9.17 pp | — | SE 0.0290; McNemar p = 0.0034 | DERIVED | Convenience sample |
| Interventions per 100 missions | 4.3, 6 months, 5,120 missions | Not run in field; 2.6 is a training-target figure, not a result | Not measured | Not measured | Not available | Baseline ILLUSTRATIVE; candidate NOT MEASURED | No field exposure |
| Contact rate | 0 observed in 6 months | No trial | — | — | Upper bound from baseline exposure unresolved | ILLUSTRATIVE | Zero events is not zero rate |
| Near-miss rate | Not recorded before the definition existed | No trial | — | — | — | NOT MEASURED | Definition created for this work |
| Evidence programme cost | — | $46,650 planned | — | — | Planning rates | ILLUSTRATIVE | Labour rates not contracted |
| Trial cost avoided | — | $25,100 | — | — | — | DERIVED | Does not include programme delay |

The honest reading of the table is that this case produced a *negative decision from positive arithmetic*: every number computed here is a projection, and the only thing established is that the projections, taken at face value, do not support promotion.

**Measurement plan for the next cycle**, with owner and trigger, so the absence of outcome is a plan rather than a gap:

| Measurement | Owner role | Proposed window | Decision trigger |
|---|---|---|---|
| Stratified paired discrepancy, 300 episodes | Test engineering | 8 weeks from 2026-11-16 | Proceed to gate if the hard-aisle discrepancy is < 0.13 and paired SE < 0.020 |
| DR-extended retrain and clean holdout | Navigation | 10 weeks | Rebuild holdout from an unused seed block; audit again |
| Monitor latency on target | Embedded | 4 weeks | p99 ≤ 3 ms including staleness check, or redesign |
| Bench fidelity validation for contact | Test engineering + safety | 12 weeks, method OPEN | A method that bounds contact-rate error on the bench by ≤ 2× |
| Rollback rehearsal | Fleet engineering | before any promotion | Completes within one shift on one robot |

## 12. Failure Modes & What Went Wrong

Every row below is a **hypothetical or reproduced-in-arithmetic** failure of the validation process, not an observed incident. They are the failures this case's design is built to catch, and three of them were live defects in the evidence package.

| Failure ID | Trigger and symptom | Detection / discriminating test | Confirmed cause or hypothesis | Impact / blast radius | Containment | Prevention / owner | Verification / limitation |
|---|---|---|---|---|---|---|---|
| F-01 | Simulation success looks excellent (0.938) and degrades mysteriously in the field | Set-intersect holdout seeds with training manifest; compare success on overlap vs disjoint | **Confirmed in arithmetic**: 390/3,000 seeds shared; overlap succeeds 1.000 vs 0.929 | 2.6 pp inflation; a passing candidate could be a failing one | Rebuild holdout from an unused block; re-run gate | Seed provenance as a required artefact in the gate record / data engineer | Re-audit every cycle; verify manifest covers all workers |
| F-02 | Corrected estimate has a wide interval that does not narrow with more simulation | Variance decomposition (F-04); compute variance share by term | **Confirmed in arithmetic**: paired term is 96.3% of total | Waste: 25,000 extra sim episodes move the half-width by < 0.1 pp | Redirect budget to paired episodes | Gate record must show variance shares / analytics | Verified by recomputation of F-04 |
| F-03 | Paired discrepancy looks small and reassuring | Stratify paired episodes by aisle class; compare class shares against mission shares | **Confirmed in arithmetic**: 120 episodes cover 46% of aisle length, all easy | Discrepancy underestimated; correction too small; false pass | Treat measured delta as a lower bound | Paired sampling frame defined by mission weights, not convenience / test engineering | Coverage audit against mission log each cycle |
| F-04 | Contact gate declared "passed" after a clean trial | Compute zero-event upper bound against limit before trial | **Confirmed in arithmetic**: 4-week bound is 1.24× limit | A null result presented as evidence of safety; the most consequential error in the case | Exposure arithmetic is a mandatory pre-trial artefact | Gate must state required exposure, not required duration / safety engineer | Recompute with actual duty factor; verify homogeneity assumption |
| F-05 | Candidate fails on hard aisles that the policy never saw, with no simulation signal | Range comparison between DR bounds and field survey | **Confirmed in arithmetic**: μ 0.45 vs training min 0.55; overhang 0.12 vs 0.10 | 22% of missions carry 45.8% of mixture variance; unreachable by more simulation | Extend DR; retrain; treat segment as out-of-support | DR ranges derived from field measurement, reviewed at each rebuild / nav lead | Segment estimate still only 24 bench episodes after fix until retrain lands |
| F-06 | Robot stops repeatedly in the aisle for no visible reason | Correlate `veto_reason` with scan sparsity and shrink-wrap presence | **Hypothesis**: monitor's clearance computation degrades on specular dropout, treating unknown space as occupied | Throughput loss; operator distrust; possible monitor bypass pressure | Raise reason codes to operator; log margin | Monitor must distinguish "unknown" from "occupied" and handle each explicitly / embedded | Fault injection with synthetic sparse scans; verify no bypass path |
| F-07 | Intervention rate improves while throughput falls | Jointly track interventions/100 missions and missions/hour | **Hypothesis**: caution rewarded by the metric — the policy learned to wait | Benefit claim is an artefact of the metric; business objective missed | Add missions/hour as co-primary | Never gate on a single benefit metric / director | Requires field data; not testable in simulation |
| F-08 | Gate outcome flips between reviews with no system change | Diff the frozen metric definitions and threshold record | **Hypothesis**: near-miss clearance threshold or denominator changed | Reputational; gate becomes negotiable | Versioned definitions with hash in decision log | Any definition change opens a new decision cycle / safety engineer | Audit trail check at each review |
| F-09 | Field data contradicts simulation in the *opposite* direction on some scenarios | Examine sign of paired difference by family | **Hypothesis**: simulator is pessimistic on narrow gaps because its contact model is too harsh | Correction over-corrects; a good policy is held | Report signed differences by family, not just pooled | Two-sided reporting / analytics | Family counts are small (16–42); sign per family is noisy |
| F-10 | Rollback takes three shifts instead of one | Timed rehearsal on one robot with the previous bundle | **Hypothesis**: bundle includes a map hash the older firmware rejects | Prolonged exposure after a stop decision | Manual procedure with classical planner, tested separately | Rehearse rollback before promotion, not during / fleet engineer | Rehearsal must use the actual previous bundle |
| F-11 | Policy performs well in evaluation but the deployed robot behaves differently | A/B build with telemetry on and off | **Confirmed as unverified**: A-08, telemetry sampling changes timing | Sim-to-deployment gap unmeasured; evaluation describes a system that is not deployed | Sample telemetry at 5 Hz | Measure the deployed configuration, or declare the difference / embedded | Compare cycle-time distributions on and off |
| F-12 | Threshold chosen after seeing results, then defended as preregistered | Compare threshold record timestamp with evaluation run timestamp | **Hypothesis** for other programmes; **not** the case here — threshold was preregistered | Selection on the outcome; gate has no power | Freeze threshold in a dated, immutable record | Preregistration as a gate artefact / safety engineer | Timestamp and hash check |

### 12.1 The failure that did not happen but nearly did

The near-miss here is F-04, and it is worth narrating because it is the failure mode most likely to be repeated by a competent team. The sequence would have been: book the trial; run four weeks; observe zero contacts; observe zero near-misses; report "safety validated, no events in 482 km"; promote. Every step is defensible in isolation and the conclusion is wrong, because 482 km cannot distinguish a rate of 1 per 200,000 m from a rate of 4 per 200,000 m at 95% confidence. The trial would have *felt* like evidence while carrying almost none.

The discriminating check takes two minutes of arithmetic. That is the uncomfortable part: the most expensive error available in this decision is preventable by a calculation that fits on one line, and the organisation's default process did not include it.

### 12.2 Competing causes for "the policy performs worse on the floor"

A real diagnostic question with four plausible causes that produce the same headline:

| Hypothesis | Discriminating check | Expected signal | If confirmed |
|---|---|---|---|
| Simulator overfit (physics/scan fidelity) | Paired study on matched scenarios | Large discordant count, one direction | Fix simulator or widen DR |
| Distribution shift (unseen conditions) | Stratify field results by condition; compare to DR range | Degradation concentrated outside DR support | Widen DR; retrain |
| Deployment mismatch (timing, calibration) | A/B telemetry; compare cycle times; check calibration dates | Degradation appears on first shift, uniform across conditions | Fix the deployment, not the policy |
| Task/metric mismatch | Compare success definitions between sim and field | Field counts episodes as failures that sim counted as successes | Align definitions before any retraining |

The third row is the one teams skip. A policy evaluated at 20 Hz and deployed at 17 Hz effective — because the logger, the monitor, and the drive interface all consume cycles — has a different effective control frequency, and a velocity-command policy is sensitive to that. The check is trivial (compare inter-command intervals) and it distinguishes a scientific problem from an integration problem. Failing to run it leads to retraining a policy that was never the problem.

> **Takeaway:** four of the twelve failure modes are arithmetic-level errors that cost minutes to check and dominate the decision; the expensive ones are not the subtle ones.

## 13. Transferable Lesson

The lesson was drafted before the analysis, in the form "simulation results are not field evidence; use a staged ladder". The analysis narrowed it substantially, because "staged ladder" turned out to be the easy half. The hard half is that each rung has a *cost curve* and the binding rung is determined by which term dominates the variance, not by which rung sounds most rigorous.

| Pattern field | Content |
|---|---|
| Canonical pattern | `transfer validation ladder` — promote only on evidence whose uncertainty is dominated by the rung that can actually be bought |
| General principle | A transfer decision is decided by the *correction term*, and the correction term's precision is set by the paired sample; therefore the sampling frame of the paired study, not the size of the simulation sweep, is the design variable that matters |
| Preconditions | A simulator that can reproduce field episodes under matched initial conditions; the ability to execute those episodes physically; a frozen, versioned metric definition; a decision owner separate from the delivery owner |
| Mechanism | Bias from simulator optimism enters as an additive correction estimated on paired data; its standard error is governed by the number of *discordant* pairs, so paired-sample effort converts directly into decision precision, while simulation effort converts into precision only on the term that is already small |
| Supporting evidence | Variance decomposition: paired term 96.3% of total at 120 pairs; doubling the 2,500-episode holdout moves the half-width by 0.05 pp; 300 further paired episodes move it by 2.6 pp. All arithmetic on ILLUSTRATIVE inputs |
| Avoid when | The simulator is the only access to the population (no physical equivalent exists, e.g. a scenario that cannot be run safely); or the decision is about a condition the simulator was validated to reproduce exactly, where the correction is provably zero |
| Portability test | Ask: "what is the standard error of my sim-to-real correction, and what fraction of my total variance does it represent?" If the answer is unknown, the ladder is a ritual |
| Adaptation | In a perception-only system the paired unit becomes an image pair rather than an episode; in a control system it becomes a trajectory segment. The additive-correction assumption weakens as the gap becomes multiplicative (e.g. errors that scale with speed) |

### 13.1 The part that generalises beyond robotics

Three statements transfer to any evaluation that stands in for a harder-to-obtain ground truth.

**First, an evaluation population is a claim about coverage, and coverage is checkable by set arithmetic.** The seed intersection took minutes and invalidated 13% of the evidence. Every simulation-based, synthetic-data, or backtest-based evaluation has an analogous overlap risk, and in every case the check is cheaper than the analysis it protects.

**Second, when a rare event gates the decision, the gate is an exposure requirement, not a duration.** "Run it for four weeks" is not a validation plan; "obtain 600,000 robot-metres" is. The conversion between them uses one constant (≈3 for a zero-event bound at 95%) and it changes the plan's shape, often to "this instrument cannot answer this question".

**Third, and most portable: if every bias you find flatters the candidate, stop averaging and start bounding.** The correct response to three optimistic defects is not to net them against each other but to ask what the answer looks like when each is repaired, and to report the interval that results rather than the point estimate that survives.

### 13.2 Where following this lesson would be harmful

The ladder is expensive, and applying it to a low-consequence decision wastes the organisation's tolerance for validation. A policy that changes how a robot chooses between two equally safe routes, with no contact exposure and a trivially reversible deployment, does not need a paired study; it needs a canary and a rollback button. The lesson applies where the consequence is irreversible — physical contact, published claims, regulated determinations — and where a false pass cannot be recovered by reverting.

It is also harmful when applied mechanically to a simulator that is *already validated* for the population at hand. If the correction has been measured to be statistically indistinguishable from zero on representative conditions, adding more paired episodes burns budget to refine a number that is already known to be zero. The portability test above distinguishes these cases.

> **Takeaway:** the ladder is not the lesson; the lesson is that the rung carrying the variance must be identified and bought, and that the paired sample's composition is the design variable that decides the answer.

## 14. Connections & Cross-References

### 14a. Related cases, theory and cheat sheets

| Relationship | Target | Mechanism shared or changed | Boundary learned |
|---|---|---|---|
| `uses-method` | Module 40, `40_robot_simulation_synthetic_data_and_sim_to_real/INDEX.md` — sim-to-real validation topic | Staged promotion, domain randomisation, sim-to-real gap | That module's scope is the mechanism; this case adds the decision arithmetic and the exposure sizing |
| `uses-method` | Module 37, `37_sensor_fusion_iot_and_realtime_perception/Cheat_Sheets/02_Sensor_Time_Sync_and_Calibration.md` | Timestamp semantics and staleness bounds underlie the `STALE_PERCEPTION` path | Clock discipline is a gate input, not an implementation detail |
| `uses-method` | Module 39, `39_robot_perception_navigation_manipulation_and_vla/INDEX.md` | Local planning, recovery behaviours, clearance metrics | Recovery behaviour defines the failure signature the case measures |
| `contrasts-with` | `11_data_science_domain_specialisations/13_mobility_travel_and_hospitality/01_mobility_dispatch_eta_and_pricing/Case_Studies/01_ETA_Calibration.md` (authored) | Both correct a fallible model against ground truth; that case has plentiful paired data and corrects a continuous prediction, this one has 120 pairs and corrects a proportion | Continuous calibration tolerates small paired samples; a proportion gate does not, because the discordant count is tiny |
| `contrasts-with` | `02_data_analytics_and_decision_science/07_analytics_metrics_bi_and_product_sense/Case_Studies/01_Marketplace_Conversion_Diagnosis.md` (authored) | Both are gate decisions on a proportion; that case can randomise, this one cannot | Randomisation solves confounding but not exposure; a rare-event gate needs neither |
| `extends` | This part's `Case_Studies/INDEX.md` pattern `transfer validation ladder` | Adds the variance-decomposition rule and the exposure-sizing rule | The INDEX pattern is mechanism; this case supplies the arithmetic |
| `contrasts-with` | `PLAN-P07-13` Fleet rollout readiness review (planned, not authored) | Rollout readiness asks about operational capacity and rollback; this case asks about evidence sufficiency | A fleet can be ready to operate a policy that is not yet validated |
| `planned` | A case on monitor design and fault injection for safety envelopes | The monitor is the subject here only as a boundary | Not authored; do not link as existing |

### 14b. What this case leaves open

| Open question | Why current evidence is insufficient | Data/test that would resolve it | Priority / trigger | Owner role |
|---|---|---|---|---|
| What method can bound a 1-per-200,000-m contact rate without years of exposure? | Field exposure cannot reach the required volume within a trial; bench fidelity for contact is unmeasured | Bench contact-fidelity study against instrumented targets with known impact energy; fault injection on the monitor | High — blocks any promotion | Safety engineer |
| How does the discrepancy behave on the four uncovered aisle classes? | Paired data exist only for central dry aisles | 300 stratified paired episodes weighted by mission share | High — sets the next gate | Test engineering |
| Is the sim-to-real gap additive or multiplicative in speed? | All bench episodes ran at ≤ 0.9 m/s; a multiplicative error would be invisible | Paired episodes at 0.6, 0.9, 1.2 m/s | Medium — affects the correction form | Navigation lead |
| Does the policy's caution reduce interventions without reducing throughput? | No field data; the metric pair is untested jointly | Within-zone alternating schedule with missions/hour as co-primary | Medium — activates only if the gate passes | Director / fleet ops |

### 14c. Interview pack — fourteen questions with checked answers

#### Q1 — Framing: what exactly is being decided?
**Level:** framing | **Skill:** decision contract

**Candidate prompt and given facts:** "A learned navigation policy scores 93.8% in your simulator. The trial slot is booked. What do you recommend, and what is the decision you are actually making?"

**What to clarify or assume:** whether the decision is promotion, and to what exposure; what the safety limits are; what the baseline does; whether the 93.8% is on a clean population.

**Strong answer.** The decision is not "is the policy good" but "does the evidence support putting this artefact on a floor shared with people, for a defined exposure, with a defined undo". Structure it: input (checkpoint plus evidence package), output (promote / hold / reject), owner (someone with authority and not the person whose schedule depends on it), baseline (the classical planner on the same zone and exposure), constraint (safety-case contact limit, unmodified safety chain), and cost asymmetry (a false promotion is irreversible, a false hold is a schedule cost). Then say that 93.8% is a statement about the simulator's population and cannot enter the decision until someone establishes how the simulator's population relates to the field's. **Check:** the answer names an exposure volume, not a duration.

**Acceptable alternatives.** Arguing the decision is "which evidence to buy next" rather than "promote or not" — defensible if the policy clearly fails or clearly passes and the real question is programme allocation.

**Follow-up:** "The safety engineer says the contact limit is 1 per 200,000 m. Does that change your recommendation?" It should: it makes the exposure arithmetic the centre of the decision rather than a footnote.

**Red flags.** Recommending the trial because "we will see how it does"; treating 93.8% as a field estimate; not asking who the decision owner is; treating the trial as free because the slot is booked (sunk-cost reasoning).

**Scoring.** 0 = recommends promotion on the simulation number. 1 = identifies the need for field evidence but no exposure framing. 2 = states the full contract with baseline and constraint. 3 = additionally identifies the asymmetry of errors and states what would falsify the recommendation.

**Answer support:** Sections 1, 3, 3.2.

#### Q2 — Data: the holdout looks clean. How do you check?
**Level:** practitioner | **Skill:** leakage audit

**Candidate prompt and given facts:** "The evaluation used 3,000 episodes from a seed block disjoint from the training block, per the team. Accept the result?"

**What to clarify or assume:** whether "block" was verified by set intersection or asserted; whether the sampler's RNG and the policy's RNG share a base seed; whether normalisation statistics were fit before the split.

**Strong answer.** Do not accept an assertion about disjointness; compute the intersection. The check is set arithmetic on the seed column against the training manifest: `|S_holdout ∩ S_train| / |S_holdout|`. Also check two things the intersection misses: whether the sampler version changed between the training and evaluation runs (a version change can make the same seed produce a different scenario, which is benign, or the same scenario, which is not), and whether any fitted transformation saw the holdout. Then compare success inside and outside the overlap; if they differ materially, the aggregate is a mixture of a memorised and an unseen population.

**Acceptable alternatives.** Arguing that a held-out *time* window is the stronger control for a system that changes over time — true in field data, and worth saying, though it does not apply to a simulator.

**Follow-up:** "The overlap is 13% and those episodes all succeed. What is the corrected number, and does it change your decision?" 0.938 → 0.929 on the disjoint subset; the decision should not rest on 0.9 pp, but the *process* finding matters because it is one of three defects pointing the same way.

**Red flags.** Accepting the assertion; checking only the policy seed and not the scenario seed; concluding "13% is small, ignore it" without computing the effect.

**Scoring.** 0 = accepts. 1 = asks for the intersection. 2 = computes it and interprets the split. 3 = adds the sampler-version and normalisation checks and notes the direction-of-bias pattern.

**Answer support:** Sections 5d, 6.2, A-03.

#### Q3 — Data and method: what is the unit, and what is the label?
**Level:** practitioner | **Skill:** metric contract

**Candidate prompt and given facts:** "Two teams report episode success rates of 0.91 and 0.83 for the same checkpoint. Neither is lying. How?"

**What to clarify or assume:** the two teams may differ in episode termination, goal tolerance, timeout, and whether a takeover counts as failure.

**Strong answer.** Reconstruct the contract for each. Four fields explain almost all such discrepancies: goal tolerance (0.30 m versus 0.50 m), timeout (90 s versus 120 s), treatment of a human takeover (failure versus excluded), and the population mix (field-matched sampler versus default sampler). Then check the denominators: episodes started versus episodes completed, and whether discarded episodes (dropped scans, simulator crashes) were excluded — exclusion of failures inflates success. A concrete reconciliation is to recompute one team's number under the other's contract and see whether the gap closes; if it does not, the populations differ. This matters for a gate because the gate's threshold is anchored to a number measured under *one* contract, and swapping contracts moves the anchor by more than the gate's margin.

**Acceptable alternatives.** Declaring the metric contract as the authority and requiring the other team to conform — legitimate when the contract is preregistered and the other team's is ad hoc.

**Follow-up:** "Team A excluded 40 episodes where the robot was reset by a human. Which number is right?" Neither, until the rule is stated: an operator reset *during* an episode is a failure (the policy did not complete the task); a reset *between* episodes is bookkeeping. The 40 episodes must be classified, not dropped.

**Red flags.** Assuming one team made an error; comparing point estimates without aligning contracts; accepting an exclusion rule that removes failures.

**Scoring.** 0 = "one of them is wrong". 1 = names tolerance or timeout. 2 = reconstructs four fields and checks exclusions. 3 = adds the reconciliation experiment and the threshold-anchoring consequence.

**Answer support:** Sections 5a, 9.1, 9.2.

#### Q4 — Calculation: how much exposure does the contact gate need?
**Level:** practitioner | **Skill:** rare-event sizing

**Candidate prompt and given facts:** "Contact limit is 1 per 200,000 robot-metres. The trial plan is two robots, 6 h/day, 5 d/week, 4 weeks, 0.9 m/s, duty factor 0.62. If we see zero contacts, can we certify?"

**What to clarify or assume:** one-sided 95% confidence; Poisson events; homogeneous operating conditions; whether "certify" means the upper bound falls below the limit.

**Strong answer.** Compute exposure first, in metres. Then apply the zero-event bound `−ln(0.05)/X` with `X` in units of 200,000 m.

```text
metres = 2 × 6 × 5 × 4 × 3600 × 0.9 × 0.62 = 482,112 m
X = 482,112 / 200,000 = 2.4106 units
bound = 2.9957 / 2.4106 = 1.243 × limit
```

So zero contacts bounds the rate at 1.24× the limit — the trial cannot certify even in the best case. Required exposure is `2.9957 × 200,000 = 599,146 m`, which is 4.97 weeks at this duty factor. **Check:** the bound scales as 1/X, so doubling the trial halves it; at 8 weeks X = 4.821 and the bound falls to 0.621, inside the limit. Note also that one observed event at 4 weeks moves to the gamma form and gives an upper bound of roughly 1.39× the limit, which is worse than the zero-event case at 8 weeks: a single event is not a small setback, it is a different regime.

**Acceptable alternatives.** Using a 90% bound (`−ln(0.10) = 2.303`) if the organisation accepts a weaker claim, which needs 460,600 m and just fits the 4-week plan — defensible but must be declared in advance.

**Follow-up:** "The limit is per *policy version*. If we run three candidate versions at 200 km each, can we pool?" No: each version is a different rate, and pooling assumes a common rate that is precisely what is unknown. Pooling would let a bad version hide behind good ones.

**Red flags.** Answering in weeks rather than metres; using 3/X as a "rule" without checking it is the 95% constant; treating zero events as evidence of a low rate without computing the bound; pooling across versions.

**Scoring.** 0 = "four weeks should be enough". 1 = computes exposure. 2 = applies the zero-event bound and finds 1.24×. 3 = inverts to required exposure, notes the single-event case, and refuses version pooling.

**Answer support:** Sections 6.10, F-07.

#### Q5 — Method selection: why not just do a field A/B?
**Level:** practitioner | **Skill:** instrument selection

**Candidate prompt and given facts:** "A colleague says: all this simulation work is indirection; run the two robots with the new policy on one and the old on the other for four weeks and compare. What is wrong with that?"

**What to clarify or assume:** what the A/B is meant to answer — throughput or safety — and what exposure it provides.

**Strong answer.** Nothing is wrong with it as an instrument for the *throughput* question, and it is the right instrument for that question because it controls for season, traffic mix, and picker behaviour, which the historical baseline does not. It is the wrong instrument for the *contact* question: 241,056 m per arm (half of 482,112) gives a zero-event bound of 2.485× the limit per arm, so a clean result on either arm is uninformative. And with two robots, randomisation unit and treatment unit coincide, so any difference is a statement about those two robots, not about the policy — the effective sample size for a robot-level effect is 1 per arm. The correct structure is: A/B for benefit, bench-plus-monitor-plus-exposure-budget for safety, and never let the A/B's clean safety record be read as safety evidence.

**Acceptable alternatives.** Alternating within-robot schedule (each robot alternates weekly) to get within-robot comparison — better for the benefit question, still says nothing about rare events.

**Follow-up:** "What if the A/B shows a 40% intervention reduction with a wide interval?" Then the direction is plausible but the magnitude is not estimated; a within-robot alternating design with many more missions is needed before quoting 40% anywhere.

**Red flags.** Accepting the A/B as safety evidence; not noticing the n = 1 per arm problem at robot level; ignoring that the two arms share a zone and therefore interfere (both robots' behaviour changes traffic).

**Scoring.** 0 = agrees the A/B settles it. 1 = notes the exposure problem only. 2 = separates benefit from safety instruments and computes per-arm exposure. 3 = adds the robot-level sample-size problem and interference.

**Answer support:** Sections 5c, 6.10, 9.2.

#### Q6 — Calculation: where should the next dollar go?
**Level:** advanced | **Skill:** variance allocation

**Candidate prompt and given facts:** "You have $18,000 and a choice: 25,000 more simulation episodes at $0.03 each, or 300 more paired episodes at $58 each. Current: 2,500 sim, 120 paired, SE 0.0295. Which, and by how much does the interval improve?"

**What to clarify or assume:** discordance rate stays near 0.108 per pair; paired episodes can be stratified; simulation episodes come from the same sampler.

**Strong answer.** Decompose first, then buy.

```text
SE_sim = 0.005655  → variance 0.00003198  (3.7% of total)
SE_delta = 0.028977 → variance 0.00083975 (96.3% of total)

25,000 more sim episodes (total 27,500):
  SE_sim = sqrt(0.07993/27500) = 0.001705
  SE_total = sqrt(0.00000291 + 0.00083975) = 0.02903
  half-width 0.05690 vs 0.05787 → improvement 0.10 pp    cost $750

300 more paired (total 420):
  SE_delta = 0.028977 × sqrt(120/420) = 0.015489
  SE_total = sqrt(0.00003198 + 0.00023991) = 0.016489
  half-width 0.03232 vs 0.05787 → improvement 2.56 pp    cost $17,400
```

Per dollar, paired episodes deliver roughly 25 times the interval reduction; per unit of *evidence quality* the comparison is stronger still, because the paired episodes also repair the convenience-sampling defect if stratified. Caveat: the improvement is bought only if the new paired episodes cover the hard aisle classes — 300 more easy-aisle episodes would narrow the interval around a number that remains biased low.

**Acceptable alternatives.** Spending part of it: 150 paired episodes plus a stratify-and-retrain cycle, if the retrain is expected to change the candidate rather than just measure it.

**Follow-up:** "After 420 paired episodes the interval is ±3.2 pp and the lower bound is 0.772 against a 0.780 threshold. What do you do?" Do not relax the threshold; the honest options are (a) retrain with extended DR and re-decide, (b) renegotiate the threshold with the safety engineer on stated grounds, (c) hold. Note that 0.772 is close enough that a *further* bias correction (removing the convenience-sampling defect, which would push the estimate down) makes it worse, not better.

**Red flags.** Choosing simulation because it is cheap per episode; computing the interval but not the variance share; ignoring the composition problem when adding paired episodes.

**Scoring.** 0 = simulation, because cheaper. 1 = paired by intuition. 2 = computes both and quantifies. 3 = adds the composition caveat and the post-improvement decision.

**Answer support:** Sections 6.5, 6.6, F-04.

#### Q7 — Debugging: the robot stops for no visible reason
**Level:** advanced | **Skill:** ambiguous diagnosis

**Candidate prompt and given facts:** "During bench testing, the policy completes 79% of episodes, but the rig's safety monitor vetoes 14% of cycles in short bursts, mostly near shrink-wrapped pallets. The policy's own output looks smooth. Diagnose."

**What to clarify or assume:** which monitor condition fired (clearance floor vs staleness); whether the scan is sparse near those pallets; whether the vetoes cluster at particular poses.

**Strong answer.** Get the veto reason code before theorising — the two conditions have different signatures. If the reason is clearance, the mechanism is most likely the policy's occupancy treatment: a sparse-return scan makes some cells unobserved, and a clearance computation that treats unobserved as occupied will report a small margin where there is free space. The discriminating test is to replay the same poses with synthetic scans at three sparsity levels and check whether the reported minimum clearance tracks sparsity rather than geometry. If the reason is staleness, the cause is transport or compute, and the test is the inter-scan interval distribution, which will show a tail. A third possibility: the policy genuinely commands close passes near those pallets because the simulator's scan model never dropped returns there, so it never learned to widen. That is distinguishable — it should reproduce in simulation only if the scan model is made sparse.

**Acceptable alternatives.** Treating all vetoes as correct and simply measuring the throughput cost — legitimate for an operational short term, wrong as a diagnosis.

**Follow-up:** "Vetoes are 14% of cycles and the episode success rate is unchanged. Does that mean the monitor is harmless?" No — it means the monitor is doing work the policy was supposed to do, and the throughput cost is hidden in slower completions. Check time-to-goal against the unmonitored runs.

**Red flags.** Blaming the policy without reading the reason code; assuming unobserved space is free (also a bug); "add more logging" as the fix rather than a specific replay.

**Scoring.** 0 = guesses at the policy. 1 = asks for the reason code. 2 = proposes the sparsity-versus-geometry replay. 3 = adds the third hypothesis (policy never learned sparse scans) and the time-to-goal check.

**Answer support:** Sections 8b, F-06, 12.2.

#### Q8 — Design: how should the monitor fail?
**Level:** advanced | **Skill:** safety-envelope design

**Candidate prompt and given facts:** "The runtime monitor sits between the policy and the drive. What should happen when the monitor process dies, when its input is stale, and when its clearance estimate is uncertain?"

**What to clarify or assume:** the safety-rated chain (bumper, safety lidar field, e-stop) is independent and remains; the monitor is a performance/competence envelope, not the safety-rated function.

**Strong answer.** Three cases, three behaviours, all fail-closed. Process death: a watchdog holds the last command for at most one or two cycles then commands zero — never "continue with the last valid command" indefinitely, because the last valid command encoded an assumption about the world that is now unverified. Stale input: command zero immediately, and do not resume autonomously when freshness returns, because the gap may have contained motion that the map does not reflect. Uncertain clearance: treat as a veto, and make "unknown" a distinct state from "occupied" so that the reason code is diagnosable. The important architectural point is that the monitor's failure mode must not be a *new* hazard, which is why fail-open is unacceptable even though it preserves throughput.

**Acceptable alternatives.** A degraded mode that caps speed at a low value and requires operator confirmation to continue — defensible if the cap is low enough that the safety-rated field can stop the robot from it, and if it cannot be entered repeatedly without alerting.

**Follow-up:** "The monitor's own p99 is 3 ms but the SoC has a shared memory bus with the logger. What do you do?" Measure the latency *with* the logger active (A-08 in this case was exactly this), and if p99 exceeds the budget, sample telemetry — the monitor's timing must be measured in the deployed configuration, not the nominal one.

**Red flags.** Fail-open; unbounded hold of the last command; treating the monitor as the safety function and thereby removing the requirement for the independent chain; not distinguishing unknown from occupied.

**Scoring.** 0 = fail-open or no watchdog. 1 = fail-closed on input failures only. 2 = all three cases with reasons. 3 = distinguishes unknown/occupied and requires measurement in the deployed configuration.

**Answer support:** Sections 8b, 8c DEC-05, F-06.

#### Q9 — Evaluation: the trial shows zero contacts. What can you claim?
**Level:** practitioner | **Skill:** negative-result interpretation

**Candidate prompt and given facts:** "Four weeks, 482 km, zero contacts, zero near-misses. Marketing wants 'validated as safe'. What do you say?"

**Strong answer.** Two different claims, two different verdicts. For near-misses, the bound is `2.9957/9.642 = 0.311×` the limit, so a claim of "near-miss rate below one-third of the limit at 95% confidence" is supportable **for off-peak, no-picker conditions**. For contacts, the bound is 1.243× the limit, so the correct claim is "the data are consistent with a contact rate anywhere from zero to 1.24 times the permitted value", which is not a safety claim. Additionally: zero observed is not zero rate; the bound is one-sided and conditional on homogeneous exposure; and the trial excluded pickers in weeks 1–2, so it does not speak to the population with the highest consequence. Finally, "validated" is a certification word that this process cannot earn — the safety case is an input, not an output.

**Acceptable alternatives.** Reframing as "no contacts observed in 482 km of off-peak operation, upper bound 1.24× the limit", which is accurate and lets the reader decide.

**Follow-up:** "What if we run 12 weeks and still see zero?" Then X = 7.231 and the bound is 0.414× the limit — supportable, at three times the cost and with the picker exclusion still unresolved.

**Red flags.** Treating zero as zero; quoting "482 km without incident" as risk information; letting marketing choose the denominator; ignoring the homogeneity assumption when the trial was off-peak.

**Scoring.** 0 = "safe". 1 = notes zero is not zero. 2 = computes both bounds and finds the asymmetry. 3 = adds population limits, homogeneity, and the certification boundary.

**Answer support:** Sections 6.10, 9.1, 9.3.

#### Q10 — Changed constraint: the site wants a two-week trial
**Level:** advanced | **Skill:** re-deriving under a new constraint

**Candidate prompt and given facts:** "Site operations can only give you two weeks, not four. They will not extend. What changes?"

**What to clarify or assume:** whether throughput evidence is still wanted; whether pickers can be excluded for two weeks.

**Strong answer.** Recompute exposure: 241,056 m, X = 1.205 for contacts and 4.821 for near-misses. The near-miss gate is still closable (bound 0.621× limit at zero events), so the trial retains *some* value. The contact gate is further out of reach (bound 2.485×). The performance gate is unaffected — it comes from simulation and the bench, not from the trial — so the trial was never the instrument that decided promotion. The decision becomes: run a two-week trial for throughput and near-miss evidence *after* the performance gate passes, or run nothing now. It should not be run now, because a trial whose results cannot change the promotion decision consumes exposure and organisational attention without buying information.

**Acceptable alternatives.** Two weeks with tighter instrumentation and a within-robot alternating schedule, if the purpose is explicitly declared as "throughput estimate, no safety claim".

**Follow-up:** "What if they offer four weeks but only one robot?" Same exposure (241,056 m), so the contact gate is no closer; but the within-robot alternating design becomes available, which is better for the throughput question. Exposure, not robot count or duration, is the quantity.

**Red flags.** Scaling the safety conclusion with duration without recomputing exposure; accepting the trial because it is offered; confusing "more robots" with "more exposure" when hours are fixed.

**Scoring.** 0 = accepts and scales expectations. 1 = recomputes exposure. 2 = separates which gates survive. 3 = reframes the trial's purpose and refuses it before the performance gate.

**Answer support:** Sections 6.10, 10.

#### Q11 — Judgment: would you have written the gate differently?
**Level:** advanced | **Skill:** gate design critique

**Candidate prompt and given facts:** "The gate is 'lower 95% bound ≥ 0.780'. A colleague says gates on confidence bounds are arbitrary because you can always widen an interval until everything fails. Respond."

**What to clarify or assume:** whether the threshold has a decision-theoretic basis or is a round number.

**Strong answer.** The criticism conflates two things. A bound-based gate is *not* arbitrary if the interval is honest: it asks "is the evidence strong enough to distinguish the candidate from the threshold", which is the actual question. A point-estimate gate asks "did the estimate land above the line", which rewards noise — a candidate with 5 episodes can pass by luck. What *is* arbitrary is the threshold's location, and here it is anchored: the incumbent's bench-measured equivalent, minus one point of practical indifference. The real weaknesses are different from the colleague's: the gate ignores the *cost* of holding (it treats the two errors as symmetric), and it is applied to a corrected estimate whose correction is itself uncertain beyond what the interval represents (convenience sampling). A better gate would (a) state a loss ratio between false promote and false hold, and (b) report the composition-adjusted bound alongside the nominal one.

**Acceptable alternatives.** Preferring a two-sided interval and a decision rule on the difference from baseline, which is statistically cleaner but harder to communicate to a non-specialist owner.

**Follow-up:** "If the loss of a false promote is 100× a false hold, where should the threshold go?" Upward, and the right formalisation is a posterior probability that the true rate exceeds the limit rather than a bound-versus-threshold comparison; a bound with a 95% level is a crude stand-in for that.

**Red flags.** Agreeing that bounds are arbitrary; defending the gate without identifying its real weaknesses; proposing to relax the level to make the candidate pass.

**Scoring.** 0 = agrees bounds are arbitrary. 1 = defends bounds. 2 = anchors the threshold and names one weakness. 3 = names both weaknesses and proposes the loss-ratio formalisation.

**Answer support:** Sections 6.9, 6.13, 3.2.

#### Q12 — Calculation: is the benefit worth the programme?
**Level:** practitioner | **Skill:** cost-benefit with units

**Candidate prompt and given facts:** "Baseline 4.3 interventions per 100 missions, candidate claims 2.6. Mission 295 m, intervention 6.5 min, marginal robot-hour worth $40, intervention labour $18. 14 robots, 6 h/day, 300 days/year. Programme costs $46,650. Evaluate."

**What to clarify or assume:** the candidate rate is a training target, not a measurement; duty factor 0.62; speed 0.9 m/s.

**Strong answer.** Build the chain explicitly.

```text
annual distance = 14 × 6 × 3600 × 300 × 0.9 × 0.62 = 50,621,760 m
annual missions = 50,621,760 / 295 = 171,599
interventions saved = 171,599 × (0.043 − 0.026) = 2,917.2
labour saved = 2,917.2 × $18 = $52,509
hours saved = 2,917.2 × 6.5 / 60 = 316.0 h
capacity value at $40/h = $12,641
total annual benefit = $65,150
payback = 46,650 / 65,150 = 0.716 year ≈ 8.6 months
```

Note that the two benefit channels are **not additive by default**: if an intervention consumes 6.5 minutes of a robot's time, the labour saving and the capacity saving are the same 6.5 minutes counted twice. Counting 316 h at $40 *and* the $18 labour cost double-counts unless the $18 is an external cost (a person walking to the robot) rather than the robot's own downtime. The defensible number is one channel: $52,509 or $12,641, not both — so payback is 0.89 to 3.7 years. This is the most common error in benefit cases.

**Acceptable alternatives.** Counting labour only, since the robot's saved time has no value unless the fleet is capacity-constrained — which is a separate question.

**Follow-up:** "The candidate rate is a target, not a result. Does that change the answer?" It changes it from a benefit estimate to a ceiling: the best case is 0.89–3.7 years, and the *expected* case is worse by however much the field rate exceeds 2.6.

**Red flags.** Double-counting the 6.5 minutes; quoting payback without stating which channel; treating the target as measured; forgetting that the programme cost is once and the benefit is annual (no discounting) — acceptable for a short payback but should be stated.

**Scoring.** 0 = no units or no chain. 1 = computes labour saving. 2 = computes both and notices the double count. 3 = presents the range and calls the rate a ceiling.

**Answer support:** Sections 6.12, 9.6.

#### Q13 — Transfer: what changes for a picking policy?
**Level:** advanced | **Skill:** transfer with a changed boundary

**Candidate prompt and given facts:** "Take the same ladder to a learned grasp policy for bin picking. What breaks first?"

**Strong answer.** Two things break, and the second is specific to manipulation. First, the *paired unit* changes from an episode to a grasp attempt, and the discrepancy is no longer binary-outcome-per-scenario but continuous (grasp success probability), so the paired estimator becomes a difference in rates with much higher variance per pair — you need more pairs, not fewer. Second, the correction's additivity assumption gets worse: in navigation, the gap is mostly about geometry and scan fidelity; in grasping, the gap is dominated by *object and contact* properties (mass, friction, compliance, deformation) that the simulator models much more poorly, so the gap is multiplicative with object class and cannot be corrected additively across the object distribution. The consequence is that the stratified approach used here (segment by aisle class) becomes mandatory rather than advisable, and the "out-of-support" segment is the whole long tail of objects — typically the majority of the SKU count by item but a minority by pick volume, which inverts the weighting logic.

**Acceptable alternatives.** Framing the transfer target as "success on the top 200 SKUs by volume" and validating only that, declaring the long tail out of scope — honest, and it makes the ladder tractable.

**Follow-up:** "Where does the exposure argument change?" Physical contact with a person is now mediated by the robot arm's force limits, which are a different safety architecture (power and force limiting rather than protective stopping), so the rare-event gate may be replaced by a certified force limit — which is a *better* position, because certification replaces statistics.

**Red flags.** Assuming the navigation recipe transfers unchanged; treating the object distribution as homogeneous; missing that a certified force limit changes the evidence requirement rather than just the numbers.

**Scoring.** 0 = "same approach, different data". 1 = notes the paired unit changes. 2 = identifies the multiplicative gap across object classes. 3 = adds the volume/weight inversion and the force-limit architecture change.

**Answer support:** Section 13, A-02, A-06.

#### Q14 — Judgment: what single artefact would you demand next time?
**Level:** advanced | **Skill:** process design

**Candidate prompt and given facts:** "You get one process change approved for the next decision cycle. What is it?"

**Strong answer.** A **gate record** — a single versioned artefact produced before evaluation and stored with the decision, containing: the frozen metric definitions with hashes; the threshold and its derivation; the holdout seed provenance and its verification method; the variance decomposition as measured (not asserted); the required exposure per gate; and the named sign-off authorities. Its value is not documentation but *ordering*: it forces the exposure arithmetic and the coverage audit to happen before results are visible, which is the only thing that prevents post-hoc threshold selection and the "run it and see" trial. In this case a gate record would have caught F-01 (no seed audit), F-03 (unstratified paired frame), and F-04 (no exposure sizing) before any of them cost anything.

**Acceptable alternatives.** A paired-sampling-frame requirement (stratified by mission weight) — narrower, but it addresses the largest variance component directly.

**Follow-up:** "What if the organisation will not enforce it?" Then encode the same content in the code that computes the verdict: a script that refuses to emit a verdict when the seed intersection is non-empty or the variance shares are absent. The artefact survives on the critical path, or it does not survive.

**Red flags.** "More documentation"; "a checklist"; anything without a mechanism that makes the check unavoidable; choosing a change that addresses the smallest of the three defects.

**Scoring.** 0 = generic process advice. 1 = names a document without a mechanism. 2 = names the gate record contents. 3 = adds the ordering effect and the enforcement mechanism.

**Answer support:** Sections 5d, 5e, 6.10, 12.

### 14d. Answer support index

| Question | Primary section | Formula or artefact |
|---|---|---|
| Q1 | 1, 3, 3.2 | Decision contract |
| Q2 | 5d, 6.2 | Seed intersection |
| Q4 | 6.10 | F-07 |
| Q5 | 5c, 6.10, 9.2 | F-07, per-arm exposure |
| Q6 | 6.5, 6.6 | F-04 |
| Q7 | 8b, F-06 | Veto reason codes |
| Q8 | 8b, 8c | DEC-05 |
| Q9 | 6.10, 9.1 | F-07 |
| Q10 | 6.10, 10 | Exposure recomputation |
| Q11 | 6.9, 6.13 | Gate form |
| Q12 | 6.12, 9.6 | Cost chain |
| Q13 | 13, A-02, A-06 | Transfer |
| Q14 | 5d, 5e, 12 | Gate record |

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Edge robotics — warehouse mobile navigation, sim-to-real transfer |
| Problem type | Binary promotion decision under a rare-event safety gate (control at point of use, evidence gating at point of decision) |
| Method | Staged transfer validation ladder: clean holdout → paired bench discrepancy → additive correction → traffic-mix weighting → lower confidence bound against a preregistered threshold; separately, exposure sizing for each rate gate |
| Primary result | **Hold.** Corrected field episode success 0.806, 95% CI [0.744, 0.867]; gate requires lower bound ≥ 0.780, shortfall 3.55 pp. Contact gate unresolvable in a 4-week trial (zero-event bound 1.24× limit); near-miss gate resolvable (0.31× limit). ILLUSTRATIVE inputs, DERIVED arithmetic |
| Key lesson | The paired discrepancy, not the simulation sweep, carries 96% of the variance; buy paired episodes on a sampling frame defined by mission weights, and convert every rare-event gate into a required exposure before scheduling anything |
| Tags | `transfer validation ladder`; secondary `scenario coverage`, `controlled recovery` |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | `CASE-P07-012` / 1.0 / 2026-11-09 |
| Evidence level | `illustrative` — all inputs constructed; arithmetic independently checked; no organisation, deployment or incident described |
| Implementation stage | `proposed` — no field trial executed; runtime monitor not built |
| Baseline / primary metric | Classical sampling-based local planner / field episode success (proportion), contact rate (per 200,000 m), intervention rate (per 100 missions) |
| Key limitation | Correction for simulator optimism measured on the easiest 46% of aisle length, so the true discrepancy is larger and the estimate is optimistic; 22% of missions have no simulation counterpart at all |
| Review trigger / owner | Any change to site layout, speed policy, safety-monitor firmware, near-miss definition, simulator version, or the safety case's contact limit / director of warehouse autonomy with safety-engineer veto |
| Gate form | Lower 95% bound on the primary metric ≥ 0.780, preregistered and dated before evaluation |
| Exposure arithmetic | 482,112 m per 4-week trial; zero-event 95% bound = 2.9957/X units; contact gate needs ≈ 599,000 m for zero events |
| The three live defects | 13% holdout seed overlap; convenience paired sampling (46% of aisles); friction 0.45 outside training support [0.55, 0.85] |
| Next action | 300 stratified paired episodes + DR extension to μ ∈ [0.42, 0.88], overhang ≤ 0.14 m, then a fresh decision cycle on a rebuilt holdout |
| Open blocker | No agreed method to certify a 1-per-200,000-m contact rate without multi-year exposure; bench contact fidelity unmeasured |
