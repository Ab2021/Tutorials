# Rollout and Rollback

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 08 of 10
> Estimated study time: 3 hours
> Prerequisites: 02, 04
> Version 2026.3

## 1. Why this topic exists

A rollout is a controlled increase in exposure. A rollback is a controlled decrease. Both are risk
management operations, and both are usually designed after the architecture is fixed, which is why
they often fail when they are needed.

The failure pattern is familiar. A new model version passes offline validation, is deployed to all
traffic in a single switch, and produces an unexpected effect. The team decides to roll back, and
discovers that the previous image requires a database schema that the new version migrated, or that
the model artifact is mounted from a shared volume that was overwritten, or that the feature flag
that would disable the model was never wired to the serving path. The rollback takes hours instead
of minutes, and the exposure continues.

Rollback is not a command. It is a property of the system: the ability to restore the previous
combination of code, model, configuration, schema, and workflow state. That property has to be
designed, tested, and maintained.

### 1.1 The decisions this topic unblocks

| Decision | Who is blocked | Consequence of a weak rollout design |
|---|---|---|
| How much traffic the new version receives first | Release manager, clinical owner | Either a broad exposure or an underpowered test |
| What signal stops the rollout | On-call, clinical safety officer | A stop that comes too late or never |
| Whether to roll back, roll forward, or hold | Incident commander | The wrong action under pressure |
| How long the rollout takes | Product, clinical operations | A slow rollout that never reaches the benefit |
| How a model can be disabled without a deployment | Platform, clinical operations | A kill switch that does not exist |
| What evidence justifies full exposure | Quality, regulatory | An unaudited expansion |

### 1.2 The two errors

A rollout design can fail in two directions.

**Too fast**: all traffic at once. The exposure is large, the evidence is thin, and the rollback is
the only protection. This is common in teams that deploy infrequently and treat each deployment as
an event.

**Too slow**: a canary so small and so short that it cannot detect the effect it is meant to
detect. A 1 percent canary on a service with 1,000 requests per day sees ten requests per day. It
can detect a crash; it cannot detect a 10-point sensitivity loss. The team concludes that the
canary passed, when in fact the canary was incapable of failing.

The design problem is to match the exposure and the duration to the effect size that matters. That
is a statistical question, not a ceremonial one.

### 1.3 Clinical rollout is not only traffic

In a clinical service, exposure is not only the fraction of requests. It is the fraction of
patients, sites, shifts, and clinicians who experience the change. A traffic-based canary can
concentrate on one site or one shift, which makes the canary population unrepresentative.

Clinical rollouts often use a site or a unit as the unit of exposure, not a request. A stepped-wedge
design, in which sites switch in a pre-specified order, is a common approach when a cluster-level
effect matters and individual randomization is impractical.

### 1.4 Rollback has a stack

Rollback must restore every layer that the release changed: traffic routing, the image, the model
artifact, the configuration, the database schema, and the workflow state. A rollback that restores
only the image can leave the system in a combination that never existed and was never tested.

The stack is the reason a rollback drill is necessary. The drill finds the layer that cannot be
restored before an incident does.

### 1.5 The cost of a failed rollout

A rollout that fails and is rolled back in minutes costs little beyond the engineering time. A
rollout that fails and is detected weeks later costs the exposure multiplied by the effect. The
difference is usually detection, not deployment skill.

That is why this topic treats rollout and rollback as one design. The rollout produces the
evidence; the rollback bounds the damage when the evidence is bad. A design that optimizes one
without the other is incomplete.

## 2. Learning outcomes

By the end of this topic you can:

1. **Choose** a rollout pattern from the decision's risk, volume, and clinical structure.
2. **Define** the unit of exposure and the unit of rollback.
3. **Compute** the sample size and duration needed to detect a stated effect.
4. **Specify** guardrail metrics and stop criteria that can fire automatically.
5. **Design** a rollback that restores every layer of the stack.
6. **Run** a rollback drill and measure time to restore.
7. **Explain** why a canary that passes may still be underpowered.

## 3. The core mental model

A rollout is a staircase of exposure with a decision at each step. The decision uses guardrail
metrics, and the fallback is a rollback that restores the whole stack.

```text
exposure
  100% │                                      ┌──── full
       │                                ┌─────┘
   25% │                        ┌───────┘
       │                  ┌─────┘
    5% │            ┌─────┘
       │      ┌─────┘
    1% │  ┌───┘
       └──┴────┴──────┴──────────┴──────────┴────▶ time
          G1   G2     G3         G4         G5
          │    │      │          │          │
          ▼    ▼      ▼          ▼          ▼
       guardrail evaluation at each gate
          │
          ├── pass ──▶ expand
          ├── inconclusive ──▶ hold, extend
          └── breach ──▶ rollback or disable
```

### 3.1 Blast radius and evidence

Every step trades blast radius for evidence. A small step exposes few patients and produces little
evidence. A large step exposes many patients and produces evidence quickly. The curve is not
linear: the evidence accumulates with the number of events, which is proportional to exposure and
duration.

```text
events = exposure_fraction × volume × duration × prevalence
```

For a rare outcome, the duration needed to accumulate enough events can exceed the tolerance for
holding a partial rollout. That is the central tension of clinical canarying: the effect that
matters most is often the one that takes longest to measure.

### 3.2 Two kinds of guardrail

**Fast guardrails** detect service failures: error rate, latency, saturation, crash loops. They
work on seconds to minutes and can stop a rollout automatically.

**Slow guardrails** detect model failures: calibration, sensitivity, subgroup performance, alert
burden. They work on days to weeks and usually require a human decision.

A rollout design needs both. The fast guardrails protect against a broken deployment. The slow
guardrails protect against a broken model. A canary with only fast guardrails will pass a model
that is subtly wrong.

### 3.3 The rollback stack

| Layer | What must be restored | Common failure |
|---|---|---|
| Traffic | Routing rule or service selector | The old version is scaled to zero |
| Image | Previous immutable digest | The digest was deleted by retention |
| Model | Previous artifact digest | The artifact is shared and was overwritten |
| Config | Previous configuration version | ConfigMap edited in place |
| Schema | Backward-compatible database state | Migration is irreversible |
| Workflow | Alert queue, integrations, user settings | The old format is no longer accepted |

A rollback plan that names only the image is incomplete. The drill should test each layer and
measure the time to restore the whole stack.

### 3.4 Deployment and release are different

Deployment is the act of putting an artifact in an environment. Release is the act of exposing it
to users. Feature flags separate the two: an artifact can be deployed with the flag off, and
released later by changing the flag.

The separation is valuable because it makes the release reversible without a redeploy. It is
dangerous when the flag state is not versioned, because the deployed artifact and the live
behaviour can diverge silently.

### 3.5 The rollback budget

A rollback has a time budget, and the budget is set by the clinical tolerance for the degraded
state, not by engineering preference. A service that can tolerate ten minutes of a bad model has a
ten-minute budget. A service that cannot tolerate any bad decisions has a zero budget, which means
the rollout design must prevent exposure rather than reverse it.

The budget drives the pattern. A ten-minute budget requires a traffic switch or a kill switch, not
a redeploy. A one-hour budget allows a rolling update with a tested rollback. A one-day budget
allows a batch rerun.

### 3.6 Contamination

If the unit of exposure is a request, the same patient can be scored by both versions in the same
hospital stay. That is not a problem for a latency comparison and is a serious problem for a
clinical comparison, because the clinician sees inconsistent advice and the outcome is attributed
to both versions.

Sticky assignment by patient or by encounter prevents contamination. When that is not possible, the
two versions must not be compared on clinical outcomes; only service metrics are valid.

The same logic applies to sites. A site-level rollout avoids contamination within a site and
accepts differences between sites. The analysis must model the site effect, not ignore it.


## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Rollout | A controlled increase in exposure to a change | Any deployment | Exposure fraction |
| Rollback | Restoring the previous known-good stack | Restarting the service | Time to restore |
| Roll-forward | Fixing the problem with a new version instead of reverting | A rollback | Version identifier |
| Canary | A limited exposure with pre-specified evaluation | A small deployment | Traffic fraction, duration |
| Blue-green | Two identical environments, one live and one idle | A canary | Switch time |
| Shadow | Scoring without acting | A canary | Predictions per day |
| Feature flag | A runtime switch that controls behaviour | A config file | Flag key |
| Kill switch | A flag that disables a model or feature | A rollback | Time to disable |
| Ramp | The sequence of exposure steps | The final percentage | Step schedule |
| Blast radius | The scope of affected users or patients | The error count | Count or fraction |
| Guardrail metric | A metric that must stay within bounds during rollout | Any metric | Threshold |
| Stop criterion | A rule that halts or reverses the rollout | A dashboard alert | Predicate |
| Unit of exposure | The entity that is assigned to the new version | The request | Patient, site, shift |
| Unit of rollback | The artifact set restored together | The image | Digest set |
| Stepped wedge | A rollout in which clusters switch in a pre-specified order | A staggered launch | Site, period |
| Cluster randomized trial | A trial assigning clusters rather than individuals | Any site rollout | Cluster count |
| Sequential test | A test that evaluates repeatedly with error control | Repeated peeking | Alpha spent |
| Peeking | Looking at results before the planned sample size | Any interim analysis | Count |
| Alpha spending | Allocating Type I error across interim looks | A fixed p-value | Alpha |
| Statistical power | The probability of detecting an effect if it exists | The sample size | Probability |
| Minimum detectable effect | The smallest effect a design can detect with stated power | Any observed effect | Absolute or relative |
| Guardrail breach | A metric outside its bound | Any alert | Predicate |
| Sticky assignment | Keeping a user in the same arm across requests | Random per request | Assignment key |
| Cohort | A group defined by time or characteristics | Any subgroup | Definition |
| Canary analysis | The automated evaluation of a canary | A dashboard | Analysis run |
| Progressive delivery | Rollout with automated analysis and traffic control | Any deployment | Controller |
| Rollback drill | A scheduled test of the rollback | A backup restore | Drill record |

Three distinctions matter most.

**A canary is a test, not a deployment.** Its purpose is to produce evidence. A canary with no
pre-specified effect size and no power calculation is a deployment with extra steps.

**Rollback is not the only safe action.** Disabling a model, degrading to a rule, or holding at the
current exposure can be safer than reverting to a version that shares the defect. The incident
commander chooses among them.

**A feature flag is not a rollback.** A flag that disables a model leaves the service running but
does not restore the previous model. It is a kill switch, which is useful, but it is a different
control from a rollback.

## 5. The mechanics, step by step

The procedure designs the rollout and the rollback together. A rollout plan without a tested
rollback is not a plan; it is a hope.

### Step 1 — Define the release unit and the rollback unit

**Input.** The change description, the artifact manifest, and the deployment architecture.

**Output.** A release unit (the set of artifacts deployed together) and a rollback unit (the set
restored together).

**Rationale.** A release that changes the image but not the model, or the model but not the
configuration, creates a combination that may never have been tested. The release unit should be
the smallest set that can be versioned and restored as a whole.

The rollback unit is usually the same set, but not always. A configuration change may be
rollbackable independently; a schema migration may not be.

Branch point: if the change includes an irreversible migration, the rollback unit cannot include
the schema. The plan must either make the migration backward compatible or accept that rollback is
not possible and design a roll-forward path instead.

### Step 2 — Choose the unit of exposure

**Input.** The clinical structure, the traffic pattern, and the effect being tested.

**Output.** Requests, patients, sites, units, shifts, or clinicians.

**Rationale.** The unit determines what the canary is representative of. A request-level split may
put the same patient in both arms on different requests, which is acceptable for a service metric
and problematic for a clinical outcome. A site-level split avoids contamination but requires more
sites to achieve the same power.

Sticky assignment is necessary for any unit larger than a request. A patient assigned to the
canary must stay in the canary for the duration of the evaluation, or the comparison is diluted.

Branch point: if the unit is a site, the rollout becomes a cluster-level design and the analysis
must account for within-site correlation. That is a statistical design decision, not a routing
configuration.

### Step 3 — Select the rollout pattern

**Input.** The risk, the volume, the clinical structure, and the rollback capability.

**Output.** A pattern with a rationale and rejected alternatives.

**Rationale.** The pattern determines the exposure mechanism and the evidence structure. The table
below has more than two outcomes because the choice depends on the risk and the unit of exposure.

| Situation | Pattern | Why | Main limitation |
|---|---|---|---|
| Low-risk service change, high volume | Canary with traffic split | Fast evidence, small blast radius | Underpowered for rare outcomes |
| High-risk model change, clinical outcome | Shadow, then stepped wedge by site | No action during shadow; cluster-level evidence | Slow; requires many sites |
| Need instant reversibility | Blue-green with a traffic switch | Switch back in seconds | Double infrastructure cost |
| Model must be disableable without a deploy | Feature flag or kill switch | Runtime control | Flag state can drift from the artifact |
| Regulated change with a predetermined change control plan (PCCP) protocol | Pre-specified staged rollout | Matches the authorized protocol | Less flexibility |
| Very low volume | Aggregate across sites or extend duration | Accumulates events | Delays the benefit and the evidence |
| Batch output | Partition or site canary | No traffic to split | Comparability across partitions |

The pattern should be chosen before the release, not during it. A team that improvises a canary
after seeing a bad signal is not running a controlled experiment.

Branch point: if the change is a class 4 change (new feature, population, or intended use), the
rollout is a clinical evaluation, not a deployment. It needs a protocol, a statistical plan, and
ethical review where applicable.

### Step 4 — Define the ramp schedule

**Input.** The pattern, the volume, and the power calculation.

**Output.** A schedule of exposure steps with durations and gates.

**Rationale.** The schedule determines how quickly the change reaches full exposure and how much
evidence accumulates at each step. A ramp that is too fast provides little evidence; a ramp that is
too slow delays the benefit and holds the organization in a partial state.

A workable default for a high-volume service is 1 percent for a day, 5 percent for three days, 25
percent for a week, then 100 percent, with a gate at each step. For a low-volume clinical service,
the steps are sites, not percentages, and the durations are weeks.

Branch point: if the power calculation shows that a step cannot detect the effect, the step is
ceremonial. Either enlarge the step or extend the duration, and state the trade-off.

### Step 5 — Specify fast guardrails with automatic stop

**Input.** The service level objectives (SLOs) and the failure modes.

**Output.** Guardrail metrics, thresholds, and an automatic action.

**Rationale.** Fast guardrails protect against a broken deployment. They must be automatic because
a human may not be watching at the moment of failure. The action should be the least destructive
one that stops the harm: pause the ramp, shift traffic back, or disable the model.

Examples: 5xx rate above 2 percent for five minutes, p99 latency above the SLO for ten minutes,
readiness failures above 20 percent, or a crash loop.

Branch point: if the guardrail is too sensitive, it will stop every rollout; if it is too loose, it
will not stop a real failure. The threshold should be calibrated from the baseline variability, not
from a round number.

### Step 6 — Specify slow guardrails with human gates

**Input.** The model metrics, the label latency, and the clinical effect.

**Output.** Guardrail metrics, evaluation points, and a named decision-maker.

**Rationale.** Slow guardrails detect model failures that service metrics cannot see. They require
a human because the decision involves clinical judgment and because the evidence is uncertain.

Examples: sensitivity below 0.55 on an audit sample, calibration slope below 0.85, alert volume
above 120 per day, subgroup gap above the pre-specified floor, or override rate above 40 percent.

Branch point: if the label latency exceeds the rollout duration, the slow guardrail cannot be
evaluated during the rollout. The plan must say so and must use a shadow or post-rollout
evaluation instead.

### Step 7 — Pre-register the analysis and the power

**Input.** The effect size that matters, the baseline rate, and the planned exposure.

**Output.** A power calculation, a minimum detectable effect, and an analysis plan.

**Rationale.** Pre-registration prevents the team from choosing the effect after seeing the data.
The power calculation prevents the team from running a canary that cannot detect the effect it
cares about.

The minimum detectable effect should be the smallest change that would matter clinically, not the
smallest change the data can detect. A canary powered to detect a 1-point sensitivity change will
be too large and too long for most clinical services.

Branch point: if the required sample exceeds what the rollout can produce, the design must change:
a larger exposure, a longer duration, a different unit, or a different endpoint. Do not proceed
with a design that cannot answer the question.

### Step 8 — Prepare the rollback across the stack

**Input.** The rollback unit and the architecture.

**Output.** A rollback procedure with commands, authority, and a time-to-restore target.

**Rationale.** The rollback must restore every layer. The procedure should be executable by the
on-call engineer without the release author, and it should be tested before it is needed.

The procedure must state what cannot be rolled back: an irreversible migration, a message already
sent, a clinical decision already made. Those require a different mitigation.

Branch point: if a layer cannot be restored, the rollback is partial. The plan must state the
residual state and how it will be reconciled.

### Step 9 — Dry-run in staging

**Input.** The release unit, the ramp schedule, and the rollback procedure.

**Output.** A staging run with the same steps, including a rollback.

**Rationale.** Staging catches the integration and configuration failures that unit tests miss. The
rollback must be exercised in staging, not only the forward path, because the rollback is the part
that is used under pressure.

The staging run should include a simulated guardrail breach to verify that the automatic stop
works.

Branch point: if staging cannot reproduce production scale, the dry run validates the mechanics
but not the statistics. State the limitation.

### Step 10 — Start at the smallest informative exposure

**Input.** The ramp schedule.

**Output.** The first step, with monitoring and a named observer.

**Rationale.** The first step should be small enough to limit the blast radius and large enough to
produce a signal. For a high-volume service, 1 percent may be enough to detect a crash. For a
low-volume service, the first step may be a single site.

The observer should be a person, not a dashboard. Someone should be watching the guardrails during
the first step, especially if the automatic stop has never been exercised in production.

Branch point: if the first step produces no traffic because of a routing error, the canary has not
started. Verify that the canary is receiving requests before interpreting the metrics.

### Step 11 — Evaluate at each gate

**Input.** The guardrail metrics and the pre-registered analysis.

**Output.** A decision at each gate: continue, extend, hold, roll back, or roll forward.

**Rationale.** The gate is a decision, not a review. It should be short, use pre-specified metrics,
and produce a recorded rationale. A gate that becomes a debate about the metric definitions has
failed at the design stage.

For a sequential design, the analysis must account for multiple looks. Repeated peeking at a
fixed p-value inflates the false positive rate. Use an alpha-spending function or a sequential
test designed for the number of gates.

Branch point: if the evidence is inconclusive, the correct action is usually to extend the current
step, not to expand. Expanding on inconclusive evidence is how a small problem becomes a large one.

### Step 12 — Choose the action

**Input.** The gate evaluation and the clinical context.

**Output.** One of five actions, recorded with a rationale.

**Rationale.** The action set is larger than pass or fail. The table below is the core of this
step.

| Gate result | Clinical context | Action | Owner |
|---|---|---|---|
| All guardrails within bounds, effect estimates consistent | No safety concern | Expand to the next step | Release manager |
| Fast guardrail breached | Service failure | Automatic rollback or traffic shift | Automation, confirmed by on-call |
| Slow guardrail breached | Possible model harm | Hold and investigate; roll back if confirmed | Clinical safety officer and data science |
| Inconclusive, wide intervals | No safety concern | Extend the step or increase exposure | Release manager with statistician |
| Guardrail breach caused by an external dependency | Not a model defect | Fix the dependency, hold the rollout | Platform on-call |
| Rollback would restore a shared defect | Both versions affected | Disable the model and revert to the manual process | Incident commander |

The action must be proportionate. Rolling back a service change because of a single slow request is
overreaction; holding a rollout after a confirmed sensitivity drop is underreaction.

Branch point: if the rollback itself is risky, prefer disabling the model or holding at the current
exposure until the rollback is safe.

### Step 13 — Communicate, close, and clean up

**Input.** The final state and the evidence.

**Output.** A release record, a communication, and a flag cleanup.

**Rationale.** The release record documents the exposure, the gates, the decisions, and the final
state. The communication tells users what changed. The flag cleanup removes the temporary switches
that would otherwise become permanent complexity.

A feature flag left in place after the rollout is technical debt with a safety risk: a future
engineer may flip it without understanding its effect.

Branch point: if the rollout is paused indefinitely, the flag and the partial exposure must be
owned and reviewed. A permanent canary is a permanent inconsistency.

### 5.14 Edge cases the procedure must handle

**Schema drift.** The new version expects a field the old version did not. Rollback requires the
old version to tolerate the new field, or the schema change must be reversible.

**Late data.** A canary evaluates on data that arrives late, so its metrics lag. The gate must use
a watermark and compare windows with equal completeness.

**Partial rollout.** The rollout is paused at 25 percent for weeks. The two versions must both be
monitored, and the comparison must account for the different populations.

**Model and feature skew.** The canary uses a different feature path from the control. The
comparison is invalid until the paths are verified to be equivalent.

**Clock skew.** Gate timestamps and metric windows disagree. Use a single time source and record
offsets.

**Quota exhaustion.** A canary doubles the infrastructure footprint in blue-green. Check quotas
before starting.

**Dependency rotation.** A credential expires during the ramp. The rollout pauses; the fix is a
rotation, not a rollback.

**Regional outage.** A region fails during the ramp. Traffic shifts to the remaining region, which
changes the exposure and the population. Pause the ramp and reassess.

### 5.15 Power, worked

A canary is only useful if it can detect the effect that matters. The sample size for a two-
proportion comparison is approximately:

```text
n per arm = (z_alpha/2 + z_beta)^2 × [p1(1 − p1) + p2(1 − p2)] / (p1 − p2)^2
```

For a sensitivity drop from 0.63 to 0.51, with two-sided alpha 0.05 and 80 percent power:

```text
n = (1.96 + 0.84)^2 × [0.63 × 0.37 + 0.51 × 0.49] / 0.12^2
  = 7.84 × [0.2331 + 0.2499] / 0.0144
  = 7.84 × 0.483 / 0.0144
  = 263 events per arm
```

At 3.8 percent prevalence, 263 events require about 6,921 patients per arm. A 5 percent canary on a
service with 1,150 patients per day receives 57.5 patients per day, so it needs 120 days to
accumulate the sample. A 25 percent canary needs 24 days. Neither fits a typical two-week ramp.

The conclusion is not that canaries are useless. It is that a canary cannot detect a 12-point
sensitivity change on a low-volume clinical service in a useful timeframe. The canary should be
used for fast guardrails and gross failures, and the model-performance evaluation should be a
shadow or a longer stepped-wedge design.

### 5.16 The rollback procedure

A rollback procedure should fit on one page and be executable by the on-call engineer.

| Step | Command or action | Verification | Time budget |
|---|---|---|---|
| 1. Declare rollback | Incident commander approval | Decision recorded | 2 minutes |
| 2. Shift traffic | Set the service selector or flag to the previous version | Canary receives zero requests | 2 minutes |
| 3. Restore the artifact | Pin the previous image and model digests | `/readyz` reports the previous version | 5 minutes |
| 4. Restore configuration | Apply the previous config version | Config hash matches | 3 minutes |
| 5. Verify predictions | Replay a golden input set | Outputs match the reference within tolerance | 5 minutes |
| 6. Verify the workflow | Check the alert queue and integration | No format errors, queue drains | 10 minutes |
| 7. Communicate | Send the all-clear or the workaround update | Message logged | 5 minutes |

The time budget is the target, not the estimate. The drill measures the actual time and updates the
budget.

The procedure should state the failure path at each step: what to do if the previous digest cannot
be pulled, if the config version is missing, or if the replay does not match. A rollback procedure
without failure paths is a happy-path script.

### 5.17 Gate pathologies

Three gate designs recur and all three fail.

The first is the **status meeting gate**: the rollout pauses until a weekly meeting reviews it. The
gate becomes a calendar event, and the evidence is stale by the time it is reviewed.

The second is the **author gate**: the person who built the change decides whether it passes. The
decision is not independent, and the incentive is to expand.

The third is the **metric-of-the-week gate**: the criterion changes after each inconclusive result.
The rollout becomes a search for a metric that passes, not a test of a pre-specified hypothesis.

A good gate is short, independent, pre-specified, and recorded. It can be automated for fast
guardrails and human for slow ones, but it cannot be any of the three pathologies above.

## 6. Worked example

A hospital prepares to release version 2 of its deterioration model. The service scores 1,150
admissions per day. The new model has a better AUROC on the temporal holdout and a slightly higher
alert rate at the same threshold. The release plan ramps 5 percent, 25 percent, then 100 percent,
with a gate at each step.

### 6.1 The guardrails

| Guardrail | Type | Threshold | Action |
|---|---|---|---|
| 5xx rate | Fast | Above 2 percent for 5 minutes | Automatic traffic shift back |
| p99 latency | Fast | Above 1,200 ms for 10 minutes | Automatic pause |
| Readiness failures | Fast | Above 20 percent for 5 minutes | Automatic rollback |
| Sensitivity on audit | Slow | Below 0.55 | Hold and investigate |
| Alert volume per 1,000 patients | Slow | Above 100 (baseline 81.7) | Hold and investigate |
| Calibration slope | Slow | Below 0.85 | Hold and investigate |

The fast guardrails are automated. The slow guardrails require the clinical safety officer and the
data science lead to review the evidence at the gate.

### 6.2 The power problem at 5 percent

At 5 percent exposure, the canary receives 57.5 patients per day. Over a seven-day step, that is
402 patients and about 15 events at 3.8 percent prevalence.

```text
patients = 1,150 × 0.05 × 7 = 402.5
events   = 402.5 × 0.038 = 15.3
```

A sensitivity estimate from 15 events has a standard error of about 0.13. A 12-point drop is less
than one standard error. The canary cannot detect the effect it is meant to detect.

The 5 percent step is still useful. It detects crashes, latency failures, and gross output errors.
It is a service canary, not a model canary. The plan should say so, rather than implying that a
passing 5 percent step validates the model.

### 6.3 The power problem at 25 percent

At 25 percent, the canary receives 287.5 patients per day. Over 14 days, that is 4,025 patients
and 153 events.

```text
patients = 1,150 × 0.25 × 14 = 4,025
events   = 4,025 × 0.038 = 153
```

The required sample for a 12-point sensitivity change is 263 events per arm. At 25 percent, the
canary needs:

```text
days = 263 / (287.5 × 0.038) = 263 / 10.9 = 24.1 days
```

The planned 14-day step is underpowered by a factor of about 1.7. To reach the required sample, the
step must be extended to 25 days, or the exposure must be increased to 50 percent, which reaches
263 events in about 12 days.

### 6.4 Fast guardrail exposure

The availability SLO is 99.5 percent over 28 days, which gives a budget of 161 failed requests for
32,200 predictions. During the 5 percent step, the canary receives 402 requests over seven days. If
the error rate is 10 percent, the canary produces 40 failures:

```text
failures = 402 × 0.10 = 40.2
budget_share = 40.2 / 161 = 0.25
```

A 10 percent error rate in the canary consumes a quarter of the monthly error budget in a week. The
automatic rollback should fire well before that, at 2 percent, but the arithmetic shows why the
fast guardrail matters: a small traffic fraction can still consume a large share of the budget if
the error rate is high.

### 6.5 Diagnostic one: the canary passes, the model is different

The 5 percent step passes every fast guardrail. The 25 percent step shows alert volume per 1,000
patients at 96.4, against a baseline of 81.7 and a threshold of 100. The gate is inconclusive but
not breached.

Over 14 days, the canary produces 386 alerts against an expected 328.7:

```text
expected = 287.5 × 14 × (81.7 / 1,000) = 328.7
observed = 386
difference = 57.3
SE = sqrt(328.7 + 386) = 26.7
z = 57.3 / 26.7 = 2.14
```

The increase is statistically significant at the 5 percent level, and it is 18 percent above
baseline. It is below the 25 percent stop threshold, so the automatic rule does not fire. The
clinical reviewer decides to hold at 25 percent and investigate.

The first diagnostic is alert volume by site and shift. The increase is concentrated in two sites
that adopted a new triage protocol in the same week. The second diagnostic is the score
distribution, which is stable after adjusting for those sites. The model is not producing more
alerts per patient at those sites; the patients are different.

The gate decision changes from "expand" to "hold and stratify." A traffic-only canary would have
missed the site-level explanation.

### 6.6 Diagnostic two: the rollback takes 47 minutes

A drill before the release attempts a rollback from version 2 to version 1. The traffic shift takes
two minutes. The image pull takes 14 seconds. The model artifact cannot be found: the registry's
retention policy deleted the version 1 artifact after 90 days, and the model was mounted from a
content-addressed store whose retention was also 90 days.

The team restores the model from a backup, which takes 41 minutes, and verifies predictions. Total
time to restore: 47 minutes against a target of 12.

The first diagnostic is the registry retention policy. The second is the model artifact's storage
location, which was outside the image digest. The fix is to retain every deployed artifact for the
audit period and to include the model in the release unit. The drill found the gap before the
release, which is the purpose of the drill.

### 6.7 Vary one assumption: 50 percent instead of 25

Suppose the second step were 50 percent instead of 25 percent. The canary would receive 575
patients per day, and 263 events would accumulate in:

```text
days = 263 / (575 × 0.038) = 263 / 21.85 = 12.0 days
```

The evidence would arrive twice as fast, and the step could be shorter. The cost is exposure: if
the model is harmful, 575 patients per day experience it instead of 287.5. The expected excess
missed events during a 12-day step with an 8-point sensitivity loss would be:

```text
events = 575 × 12 × 0.038 = 262
excess = 262 × 0.08 = 21
```

At 25 percent over 24 days, the exposure is similar in total events but spread over a longer
period. The trade-off is between faster evidence and concentrated risk. There is no universal
answer; the clinical owner must choose, and the choice should be recorded.

### 6.8 What the example establishes

The 5 percent step could not answer the model question. The 25 percent step could answer it only
with an extension. The fast guardrails protected the service; the slow guardrails required a human
and a statistical design.

The rollback drill found a retention gap that would have turned a 12-minute rollback into a
47-minute one. The drill cost a few hours and prevented a much longer exposure.

A rollout plan that states only the percentages is incomplete. It must state what each step can
detect, how long it must run, and what the rollback restores.

### 6.9 The gate record

Each gate produces a record, and the record is the evidence that the rollout was controlled.

| Gate | Exposure | Evidence | Decision | Decision-maker |
|---|---|---|---|---|
| G1 | 5 percent, 7 days | No fast guardrail breach; 402 requests; sensitivity not estimable | Expand | Release manager |
| G2 | 25 percent, 14 days | Alert volume +18 percent, z = 2.14; site-stratified analysis requested | Hold | Clinical safety officer |
| G2a | 25 percent, +10 days | Increase explained by two sites; score distribution stable | Expand | Clinical safety officer |
| G3 | 100 percent, 14 days | Calibration and sensitivity within bounds on the audit sample | Complete | Clinical safety officer and data science |

The record shows that G2 did not pass on the first look. It was held and explained, not waved
through. That distinction is what separates a controlled rollout from a ceremony.

## 7. Code, query, or calculation

The script computes the sample size for a sensitivity change, the days required at each exposure
level, the z-score for an alert-volume difference, and the error-budget share of a canary failure.
It uses only the standard library.

```python
"""Canary power, exposure, and rollback arithmetic. Standard library only."""

import math


def events_per_arm(p1, p2, z_alpha=1.96, z_beta=0.84):
    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    return numerator / (p1 - p2) ** 2


def patients_for(events, prevalence):
    return events / prevalence


def days_for(patients, volume_per_day, exposure):
    return patients / (volume_per_day * exposure)


def count_z(x1, x2):
    """Poisson-style z for two event counts."""
    return (x1 - x2) / math.sqrt(x1 + x2)


def main() -> None:
    n_events = events_per_arm(0.63, 0.51)
    n_patients = patients_for(n_events, 0.038)
    print(f"events_per_arm        {n_events:.0f}")
    print(f"patients_per_arm      {n_patients:.0f}")
    print(f"days_5pct             {days_for(n_patients, 1150, 0.05):.1f}")
    print(f"days_25pct            {days_for(n_patients, 1150, 0.25):.1f}")
    print(f"days_50pct            {days_for(n_patients, 1150, 0.50):.1f}")
    print(f"days_full             {days_for(n_patients, 1150, 1.00):.1f}")

    expected_canary_alerts = 287.5 * 14 * (81.7 / 1000)
    z = count_z(386, 328.7)
    print(f"expected_alerts       {expected_canary_alerts:.0f}")
    print(f"alert_volume_z        {z:.2f}")

    budget = 161
    canary_requests = 402
    failures = canary_requests * 0.10
    print(f"canary_failures_10pct {failures:.0f}")
    print(f"budget_share          {failures / budget:.2f}")

    assert abs(n_events - 263) < 2
    assert abs(days_for(n_patients, 1150, 0.05) - 120) < 1
    assert abs(days_for(n_patients, 1150, 0.50) - 12) < 1
    assert abs(z - 2.15) < 0.1


if __name__ == "__main__":
    main()
```

Expected output, confirmed on Python 3.11:

```text
events_per_arm        263
patients_per_arm      6920
days_5pct             120.4
days_25pct            24.1
days_50pct            12.0
days_full             6.0
expected_alerts       329
alert_volume_z        2.14
canary_failures_10pct 40
budget_share          0.25
```

The first result is the one that changes rollout design. A 12-point sensitivity change requires 263
events per arm, which is 6,920 patients per arm, which is 120 days at 5 percent exposure. A
two-week 5 percent canary cannot detect it.

The exposure calculation is linear in the exposure fraction. Doubling the exposure halves the days
and doubles the number of patients exposed if the model is harmful. That is the trade-off the
clinical owner is being asked to make.

The `alert_volume_z` of 2.14 is just above the conventional 1.96 threshold. It is evidence, but not
overwhelming. A reviewer should ask for the site-stratified analysis before expanding, which is what
the worked example does.

The `budget_share` shows that a 10 percent error rate in a 5 percent canary consumes a quarter of
the monthly error budget in a week. The automatic stop should fire at 2 percent, but the number is a
reminder that small traffic fractions are not small risk when the error rate is high.

The reference service's `k8s/deployment.yaml` uses a Kubernetes rolling update, which replaces pods
gradually but does not analyze model behaviour. `k8s/hpa.yaml` scales on CPU. Neither provides a
canary analysis or a guardrail. A production rollout would need a progressive delivery controller
and a model-performance gate in addition to the Kubernetes mechanics.

A Kubernetes rolling update with `maxUnavailable: 25%` and `maxSurge: 25%` keeps at least 75
percent of pods available, but it does not compare the new pods with the old ones on any metric.
It is an availability mechanism, not a canary. The analysis has to be added on top.

## 8. Data, standards, and regulatory context

Rollout and rollback are engineering practices shaped by regulatory obligations. The instruments
below determine when a rollout is a clinical evaluation, when a rollback is a field correction, and
what evidence must be retained.

### 8.1 The instruments

| Instrument | Status | What it constrains | What it does not specify |
|---|---|---|---|
| FDA PCCP guidance, 2024, reissued 2025 | Final guidance implementing FD&C 515C | The modifications, protocol, and impact assessment for staged model updates | The canary percentages or duration |
| FDA AI-enabled device software functions guidance, draft January 2025 | Draft | Validation under clinically relevant conditions; postmarket monitoring | Rollout mechanics |
| FDA GMLP principles, 2021 | Non-binding | Testing under clinically relevant conditions, human-AI team performance | Traffic splitting |
| IEC 62304:2006/Amd 1:2015 | Binding when invoked | Change control, configuration management, problem resolution | Deployment strategy |
| ISO 14971:2019 | Binding when invoked | Risk control during and after changes | Stop criteria |
| ISO/IEC 42001:2023 | Voluntary, certifiable | Operational control and continual improvement | Rollout percentages |
| FDA recall authority, 21 CFR Part 806 | Binding | Correction or removal of a device that violates the law or presents a risk | The decision to roll back |
| HIPAA Security Rule | Binding for covered entities | Access control and audit during changes | Rollout design |

### 8.2 When a rollback becomes a field action

A rollback that restores a previous version is usually an internal change. A rollback that removes
or corrects a device already in commercial distribution can be a field correction or removal under
21 CFR Part 806, with reporting obligations. The distinction depends on whether the device is in
commercial distribution and whether the action is taken to address a risk.

For a software-as-a-service model, the boundary is less clear than for a physical device. A model
that is disabled for all customers may be a field action; a model that is rolled back for one
tenant may be an operational change. The determination belongs to regulatory affairs, and it should
be made before the action, not after.

### 8.3 What the PCCP requires of a staged rollout

If the PCCP describes a staged rollout as part of the modification protocol, the stages, the
evaluation, and the acceptance criteria are part of the authorized change. A rollout that deviates
from the protocol is outside the authorization, even if the deviation is more conservative.

The practical consequence is that the rollout plan should be written before the PCCP is submitted,
not after. A team that designs a 5-25-100 ramp and then discovers that the PCCP describes a
site-by-site rollout has a mismatch that requires either a protocol amendment or a different plan.

### 8.4 What the platform standards provide

Kubernetes Deployments support rolling updates with `maxSurge` and `maxUnavailable`, which control
how many pods are created above the desired count and how many may be unavailable during the
update. The defaults are 25 percent for each, and the two cannot both be zero. These semantics
govern pod availability, not model behaviour.

Argo Rollouts and similar controllers add canary steps, analysis runs, pauses, and automatic
aborts. OpenFeature provides a vendor-neutral application programming interface (API) for feature flags with an evaluation context.
These are conventions and tools, not regulatory requirements. They implement controls; they do not
define the criteria.

### 8.5 What is convention

- A 1-5-25-100 ramp. It is a common default, not an evidence-based standard.
- A 24-hour bake time at each step. The duration should follow the effect size and the volume.
- Blue-green for every service. It doubles infrastructure cost and is unnecessary for low-risk
  changes.
- Feature flags for every model. A flag adds a state dimension that must be versioned and cleaned
  up.
- A stepped-wedge design for every site rollout. It is appropriate when cluster-level effects
  matter and individual randomization is impractical.

> **Note:** A canary that passes is evidence about the canary population, not proof of safety for
> the full population. The rollout plan should state what the canary cannot see.

### 8.6 The release record as regulatory evidence

The release record, the gate decisions, and the post-rollout evaluation are evidence that the
change was controlled. In a regulated context, they support the change record and the PCCP impact
assessment. In an unregulated context, they support the internal audit and the next release.

The record should be retained with the quality record, not only in the CI system. A CI log that is
deleted after 90 days cannot support a review that happens after a year.

## 9. Industry practice and case evidence

### 9.1 Canarying at Google

The Google SRE Workbook describes canarying as a partial, time-limited deployment with an
evaluation process integrated into the release. It emphasizes a representative canary population,
metrics that indicate problems, and a gradual multi-stage canary rather than a single before-and-
after comparison.

The documented practice is from high-volume software services. The inference for clinical AI is
that the method transfers, but the population representativeness is harder: a traffic canary can
concentrate on one site, one shift, or one clinician group, and the model effect may differ across
them.

### 9.2 Progressive delivery controllers

Argo Rollouts documents canary and blue-green strategies with analysis runs, pause steps, and
automatic aborts. Flagger provides a similar capability. These tools make the mechanics of a
staged rollout repeatable.

The documented capability is orchestration. The inference is that the tool does not solve the
statistical problem: an analysis run that compares error rates over a ten-minute window cannot
detect a sensitivity change that takes weeks to observe. The tool executes the design; the design
must come from the effect size.

### 9.3 IDx-DR and workflow rollout

IDx-DR was authorized for use in primary care with a defined workflow, including image-quality
checks that reject unusable images. The rollout of such a device depends on training the operators
and integrating the capture and review steps into the clinic workflow.

The documented case is about authorization and workflow, not about a canary. The inference is that
a clinical rollout has a human-adoption dimension that traffic splitting does not address. A model
can be technically live at 100 percent and still not be used.

### 9.4 The Epic Sepsis Model and rollout without local validation

The Epic Sepsis Model was widely deployed and later externally validated with lower performance
than the vendor reported. The deployment appears to have proceeded without a site-level canary or
a local performance gate.

The documented facts are the validation results. The inference is that a rollout without a local
evaluation is a nationwide canary with no control group and no stop criteria. The lesson is not
that canaries are sufficient; it is that the absence of any local evaluation is worse.

### 9.6 Rollout in low-volume settings

Many clinical services are low-volume: a few hundred predictions per day, a handful of sites, a
rare outcome. In these settings, a traffic canary is often neither statistically useful nor
operationally necessary. The rollout is a site-by-site or stepped-wedge evaluation, and the
evidence accumulates over months.

The inference is that the software canary pattern does not transfer directly. The mechanics of
traffic control transfer; the statistics do not. A low-volume service should use shadow and
stepped-wedge designs, and should plan for a longer evaluation than a high-volume service.

### 9.5 What the evidence does not show

There is no public evidence on the optimal canary duration or exposure for clinical models, and no
benchmark that compares rollout strategies on patient outcomes. The software literature provides
mechanics and principles; the clinical literature provides evaluation designs. The combination is
local.

A team should record its rollout decisions, the evidence at each gate, and the outcomes, and should
use that record to calibrate future ramps. The record is more useful than a default percentage.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Underpowered canary | The canary passes in hours | Exposure or duration too small for the effect | A harmful model reaches full traffic | Power calculation before the ramp |
| Unrepresentative canary | Canary is one site or one shift | Traffic routing correlates with population | The effect is missed or misestimated | Stratify assignment and compare case mix |
| Peeking inflates false positives | The gate is checked daily at a fixed p-value | Repeated testing without alpha control | A harmless difference stops the rollout | Sequential design or alpha spending |
| No stop criteria | The gate is a discussion | Criteria not pre-specified | The rollout continues through a breach | Numeric criteria and an owner |
| Over-sensitive auto-stop | Every rollout pauses | Threshold below baseline noise | Rollout fatigue and bypass | Calibrate from baseline variability |
| Missing rollback artifact | The previous digest is not in the registry | Retention policy | Rollback takes hours or fails | Retain deployed artifacts for the audit period |
| Incompatible schema | The migration is irreversible | Rollback restores code but not the database | The old version cannot run | Backward-compatible migrations or roll-forward plan |
| Stale feature flag | The flag state is not versioned | Manual edits after the release | Behaviour diverges from the artifact | Version flags with the release |
| Permanent partial rollout | The ramp is paused for months | No owner for the decision | Two versions in production indefinitely | Time-box the pause and assign an owner |
| Rollback restores a shared defect | Both versions fail the same way | The defect is in a dependency or the data | Rollback does not mitigate | Diagnose before rolling back; disable instead |
| Non-sticky assignment | The same patient switches arms | Random assignment per request | Diluted comparison and inconsistent care | Sticky assignment by patient |
| Unsegmented monitoring | Aggregate metrics look normal | Two versions are pooled | A regression in one version is hidden | Segment every metric by version |
| Roll-forward without validation | The fix is deployed directly | Urgency and confidence | A second failure on top of the first | Treat roll-forward as a new release |
| Missing communication | Users learn from the model's behaviour | No release notice | Confusion and workarounds | Pre-release notice and a change log |

Red flags: a canary plan without a power calculation; a gate with no named decision-maker; a
rollback procedure that has never been run; a feature flag with no expiry; a partial rollout with
no date for the next decision.

A rollout that cannot be stopped is not a rollout; it is a launch.

A canary that passes is evidence, not proof, and the record should say so.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The canary population resembles the full population | The evidence must transfer | Routing by site or shift | Compare case mix and baseline rates |
| The effect size is known before the ramp | Power depends on it | No prior estimate | Use the validation cohort and a sensitivity analysis |
| The guardrail metrics are reliable at canary volume | Detection depends on them | Low volume, noisy metric | Simulate the metric at canary volume |
| The rollback artifact is available | Rollback depends on it | Retention policy | Pull the previous digest monthly |
| The schema is backward compatible | The old version must run | Irreversible migration | Test the old version against the new schema |
| The flag state is versioned | Behaviour must match the artifact | Manual edits | Compare live flag state to the manifest |
| The gate decision is independent | The evidence must be unbiased | The release author decides alone | Separate the approver from the author |
| The rollout can be paused safely | Partial exposure may persist | No owner, no date | Time-box and assign |
| The communication reaches users | Adoption depends on it | No distribution list | Confirm receipt, not just sending |
| The gate can be evaluated at canary volume | Detection depends on exposure | The metric is too rare | Simulate the metric at the planned volume |

## 12. Exercises

### Exercise 12.1 — Choose a pattern

Choose a rollout pattern for each scenario: (a) a logging change on a high-volume API; (b) a new sepsis model with a clinical outcome; (c) a model that must be disableable within one minute; (d) a model update for a device fleet in 12 hospitals.

**Deliverable.** A four-row table with pattern, exposure unit, and rationale.

**Acceptance criteria.** Patterns match risk and structure; the device case uses a fleet or site unit, not a traffic fraction; the kill-switch case names the flag.

**Time.** 25 minutes.

### Exercise 12.2 — Compute the sample size

A sensitivity change from 0.75 to 0.65 must be detected with 80 percent power at a two-sided 5 percent level. Prevalence is 6 percent. Compute events per arm and patients per arm.

**Deliverable.** Two numbers with the formula.

**Acceptance criteria.** Correct formula; correct substitution; units stated.

**Time.** 25 minutes.

### Exercise 12.3 — Design a ramp

Design a four-step ramp for a service with 800 predictions per day and an outcome prevalence of 4 percent. State the exposure, duration, what each step can detect, and the gate decision.

**Deliverable.** A four-row table with exposure, duration, detectable effect, and gate.

**Acceptance criteria.** Each step states its minimum detectable effect; the first step is justified as a service canary; the final step has a post-rollout evaluation.

**Time.** 40 minutes.

### Exercise 12.4 — Write stop criteria

Write three fast guardrails and three slow guardrails for a deterioration model. For each, state the metric, threshold, duration, action, and owner.

**Deliverable.** A six-row table.

**Acceptance criteria.** Fast guardrails are automatable; slow guardrails have a named clinical decision-maker; thresholds are justified against baseline variability.

**Time.** 35 minutes.

### Exercise 12.5 — Design the rollback stack

For a release that changes the image, the model, a threshold, and a database column, write the rollback plan for each layer. Identify which layer cannot be rolled back and what to do instead.

**Deliverable.** A five-row table with layer, action, verification, and time.

**Acceptance criteria.** Every layer covered; the irreversible layer has a roll-forward plan; time to restore is stated.

**Time.** 40 minutes.

### Exercise 12.6 — Critique a canary plan

A plan reads: "Deploy to 1 percent of traffic for one hour. If error rate is normal, deploy to 100 percent." Write a critique and a revised plan.

**Deliverable.** A 350-word critique with a revised plan.

**Acceptance criteria.** Identifies the power problem; distinguishes service from model guardrails; adds a slow evaluation and a rollback test.

**Time.** 30 minutes.

### Exercise 12.7 — Design a rollback drill

Write a quarterly rollback drill for a model service. Include the scenario, the steps, the success criteria, the measurement, and the report.

**Deliverable.** A two-page drill procedure.

**Acceptance criteria.** Restores every layer; measures time to restore; includes a failure injection; produces an action with an owner.

**Time.** 45 minutes.

### Exercise 12.8 — Design a stepped wedge

A model is rolling out to six sites over twelve months. Design the stepped-wedge schedule, the primary endpoint, the analysis approach, and the stopping rules.

**Deliverable.** A schedule table and a one-page analysis plan.

**Acceptance criteria.** Site order is justified; the analysis accounts for time trends and clustering; stopping rules are pre-specified; ethical review is mentioned where applicable.

**Time.** 60 minutes.

The exercises total about five hours. Exercises 12.2, 12.6, and 12.8 are the ones that
distinguish a rollout design from a deployment checklist; if time is short, do those three.

## 13. Self-check answers

### 13.1 Exercise 12.1

(a) Canary with a traffic split. Unit: request. The risk is low and the volume is high, so a small
canary produces evidence quickly.

(b) Shadow, then stepped wedge by site. Unit: site. The outcome is clinical and the model changes
practice, so a traffic canary is both underpowered and ethically awkward.

(c) A feature flag or kill switch, deployed dark and enabled at release. Unit: request or tenant.
The requirement is reversibility in under a minute, which a redeploy cannot guarantee.

(d) A staged fleet rollout with a canary group of devices, then site by site. Unit: device or site.
A traffic split is not available, and the update must be signed and reversible.

### 13.2 Exercise 12.2

```text
n = (1.96 + 0.84)^2 × [0.75 × 0.25 + 0.65 × 0.35] / 0.10^2
  = 7.84 × [0.1875 + 0.2275] / 0.01
  = 7.84 × 0.415 / 0.01
  = 325.4 events per arm
patients = 325.4 / 0.06 = 5,423 per arm
```

The design needs about 325 events and 5,423 patients per arm. At 800 predictions per day, a 25
percent canary receives 200 patients per day, so the canary needs 27 days per arm. A one-week
canary cannot detect this effect.

### 13.3 Exercise 12.3

A strong ramp: 5 percent for seven days (detects crashes and gross output errors), 25 percent for
14 days (detects large alert-volume changes), 50 percent for 14 days (accumulates sensitivity
events), 100 percent with a 30-day post-rollout audit. Each step states what it can and cannot
detect, and the final step includes the outcome evaluation that the earlier steps could not power.

### 13.4 Exercise 12.4

Fast: 5xx rate above 2 percent for five minutes, automatic traffic shift, platform on-call; p99
latency above 1,200 ms for ten minutes, automatic pause, platform on-call; readiness failures above
20 percent for five minutes, automatic rollback, platform on-call.

Slow: sensitivity below 0.55 on the audit sample, hold and investigate, clinical safety officer;
alert volume above 100 per 1,000 patients, hold, clinical safety officer; calibration slope below
0.85, hold, data science lead.

### 13.5 Exercise 12.5

Image: pin the previous digest, verify `/readyz`, 5 minutes. Model: pin the previous artifact
digest, verify hash, 5 minutes. Threshold: restore the previous config version, verify the value,
2 minutes. Database column: if the column is additive and nullable, the old version ignores it and
rollback is safe; if the old version cannot tolerate it, the migration is not reversible and the
plan must be a roll-forward that fixes the new version. The irreversible layer is the schema, and
the alternative is a backward-compatible migration designed before the release.

### 13.6 Exercise 12.6

The plan has three defects. A one-hour, 1 percent canary cannot detect any model effect; it can
detect only a gross service failure. The decision rule uses only the error rate, so a model that is
available but wrong passes. There is no rollback test, so the rollback may fail when needed.

A revised plan: 5 percent for a day with fast guardrails; 25 percent for two weeks with alert
volume and audit-sample sensitivity; 100 percent with a 30-day post-rollout evaluation; a rollback
drill before the first step; a named clinical decision-maker at each gate.

### 13.7 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Power | 25% | No calculation | Sample size stated | Detectable effect tied to clinical importance |
| Guardrails | 25% | Error rate only | Fast and slow | Thresholds calibrated and owned |
| Rollback | 25% | Image only | Full stack | Drilled with measured time |
| Decision | 15% | Pass or fail | Named approver | Pre-registered sequential analysis |
| Record | 10% | Chat notes | Gate table | Auditable release record |

### 13.8 Exercise 12.8

A strong design orders sites by readiness and case mix, uses a primary endpoint measured at the
patient level with site-level clustering, includes calendar time as a fixed effect, and pre-
specifies stopping rules for harm and futility. It also states that sites not yet switched are the
comparison, and that the analysis must account for the fact that the intervention is introduced at
different times.

The record is also the input to the next rollout. A team that has run three canaries knows
which effects its canaries can detect and which require a different design. That knowledge is
worth more than any default ramp published by a vendor.

A grader should weight the power calculation and the rollback stack most heavily. Those are
the two parts that are usually missing from a rollout plan and the two that determine whether the
plan protects patients.

## 14. Interview questions

**1. What is a canary release?**
Strong answers describe partial, time-limited exposure with pre-specified evaluation. Weak answers
describe a small deployment.

**2. Why can a canary pass and the model still be harmful?**
Strong answers explain power, effect size, and the difference between service and model guardrails.
Weak answers say the canary was too short.

**3. How do you choose the canary percentage?**
Strong answers start from the effect size, prevalence, and volume. Weak answers use 1 or 5 percent
because it is common.

**4. What is a rollback stack?**
Strong answers list traffic, image, model, config, schema, and workflow. Weak answers say the
previous image.

**5. What is the difference between a feature flag and a rollback?**
Strong answers say a flag disables behaviour; a rollback restores a previous version. Weak answers
treat them as synonyms.

**6. How do you prevent peeking from inflating false positives?**
Strong answers describe sequential testing or alpha spending. Weak answers say to check only at
the end.

**7. What is a stepped-wedge rollout?**
Strong answers describe clusters switching in a pre-specified order with time as a confounder. Weak
answers describe a staggered launch.

**8. How do you test a rollback?**
Strong answers describe a scheduled drill that restores every layer and measures time. Weak answers
say to run the rollback command in staging.

**9. What stops a rollout automatically?**
Strong answers name fast guardrails and the least destructive action. Weak answers say a person
watches a dashboard.

**10. When is a rollback a regulatory field action?**
Strong answers distinguish internal restoration from correction of a distributed device and defer
to regulatory affairs. Weak answers say any rollback is internal.

## 15. Key terms

- **Alpha spending** — allocating Type I error across interim looks.
- **Blast radius** — the scope of affected users or patients.
- **Blue-green** — two environments with a traffic switch between them.
- **Canary** — a limited exposure with pre-specified evaluation.
- **Canary analysis** — automated evaluation of a canary.
- **Cluster randomized trial** — a trial assigning clusters rather than individuals.
- **Cohort** — a group defined by time or characteristics.
- **Feature flag** — a runtime switch controlling behaviour.
- **Guardrail metric** — a metric that must stay within bounds during rollout.
- **Kill switch** — a flag that disables a model or feature.
- **Minimum detectable effect** — the smallest effect a design can detect with stated power.
- **Peeking** — looking at results before the planned sample size.
- **Progressive delivery** — rollout with automated analysis and traffic control.
- **Ramp** — the sequence of exposure steps.
- **Roll-forward** — fixing a problem with a new version instead of reverting.
- **Rollback** — restoring the previous known-good stack.
- **Rollback drill** — a scheduled test of the rollback.
- **Rollout** — a controlled increase in exposure.
- **Sequential test** — a test that evaluates repeatedly with error control.
- **Shadow** — scoring without acting.
- **Statistical power** — the probability of detecting an effect if it exists.
- **Stepped wedge** — clusters switch in a pre-specified order.
- **Sticky assignment** — keeping a unit in the same arm across requests.
- **Stop criterion** — a rule that halts or reverses the rollout.
- **Unit of exposure** — the entity assigned to the new version.
- **Unit of rollback** — the artifact set restored together.

## 16. Sources

All URLs accessed 26 October 2026.

- Google. *The Site Reliability Workbook*, Chapter 16, Canarying Releases. https://sre.google/workbook/canarying-releases/
- Kubernetes. *Deployments*. https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Kubernetes. *Update a Deployment Without Downtime*. https://kubernetes.io/docs/tasks/run-application/update-deployment-rolling/
- Argo Rollouts. *Canary Deployment Strategy*. https://argo-rollouts.readthedocs.io/en/stable/features/canary/
- Argo Rollouts. *Analysis and Progressive Delivery*. https://argo-rollouts.readthedocs.io/en/latest/features/analysis/
- OpenFeature. *Specification*. https://openfeature.dev/specification/
- OpenFeature. *Evaluation Context*. https://github.com/open-feature/spec/blob/main/specification/sections/03-evaluation-context.md
- US Food and Drug Administration. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, 2024, reissued 2025. https://www.fda.gov/media/166704/download
- US Food and Drug Administration. *Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations*, draft guidance, January 2025. https://www.fda.gov/media/184856/download
- US Food and Drug Administration, Health Canada, and UK MHRA. *Good Machine Learning Practice for Medical Device Development: Guiding Principles*, 2021. https://www.fda.gov/media/153486/download
- IEC 62304:2006/AMD1:2015. *Medical device software — Software life cycle processes*. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. *Medical devices — Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
- ISO/IEC 42001:2023. *Information technology — Artificial intelligence — Management system*. https://www.iso.org/standard/42001
- 21 CFR Part 806, Medical Devices; Reports of Corrections and Removals. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-806
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*. 2021;181(8):1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- US Food and Drug Administration. De Novo classification request DEN180001 (IDx-DR), 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Previous topic: [`07_incident-response.md`](07_incident-response.md)
- Next topic: [`09_model-governance.md`](09_model-governance.md)
- Related topic: [`02_deployment-patterns.md`](02_deployment-patterns.md)
- Related topic: [`04_ci-cd-ct.md`](04_ci-cd-ct.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- Kubernetes deployment: [`../../labs/mlops/k8s/deployment.yaml`](../../labs/mlops/k8s/deployment.yaml)
- Runbook: [`../../labs/mlops/RUNBOOK.md`](../../labs/mlops/RUNBOOK.md)
