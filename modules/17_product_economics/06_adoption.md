# Adoption

> Module 17 — Product, Economics, and Leadership
> Topic 06 of 09
> Estimated study time: 4 hours
> Prerequisites: 01 (Product Discovery), 05 (Procurement)
> Version 2026.3

## 1. Why this topic exists

A product that is bought and not used is worse than a product that is not bought. The buyer has
spent money, the vendor has booked revenue, and the benefit that justified the purchase never
appears. At renewal, the buyer has evidence that the product does not work, and the vendor has no
answer, because the product was never given a chance to work.

Adoption is not a marketing outcome that follows a good product. It is a design property that
determines whether the product's effect is delivered at all. In a model where benefit equals
effect multiplied by use, a product with a large effect and 20 percent use delivers less than a
product with a modest effect and 70 percent use.

### 1.1 The distribution problem

Adoption is not uniform. In most deployments, a minority of users account for a majority of use,
and the minority is identifiable in advance by baseline workload and by attitude. A vendor that
reports an average adoption rate conceals this distribution, and the average is the wrong number
for two reasons: the benefit is concentrated in the high users, and the low users are the ones who
will churn the account at renewal.

The published evidence on team-based documentation makes the point concretely: in a study of
18,265 physicians, high-intensity adopters saved about 54 minutes per week of note time, rising to
72 minutes after a learning curve, and saw 2.5 to 4.5 additional visits per week, while
low-intensity adopters saw no time savings at all (JAMA Internal Medicine, 2024). The average
across all physicians would have described a benefit that most physicians did not receive.

### 1.2 Why pilots overstate adoption

A pilot has selected users, extra support, and a short duration. It measures peak adoption under
favourable conditions, not steady-state adoption under normal ones. The gap between the two is the
single most common reason a successful pilot fails to convert.

Three mechanisms produce the gap. The pilot's users are volunteers, so they are pre-disposed. The
pilot's support is not the support that will exist at scale. And the pilot's duration ends before
the novelty effect wears off, which is typically at three to six months.

### 1.3 What this topic covers

It covers how to define and measure adoption, how to design for it, how to run a rollout that
sustains it, and how to connect it to retention and unit economics. It draws on implementation
science frameworks where they are useful and avoids treating them as compliance checklists.

It does not cover change management as a human resources discipline, clinical training design, or
the psychology of behaviour change in detail.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Define an adoption metric with an explicit denominator and unit | Exercise 1; self-check 1 |
| 2 | Distinguish exposure, trial, use, sustained use, and substitution | Exercise 2; self-check 2 |
| 3 | Instrument a deployment so that adoption is measurable from day one | Exercise 3; self-check 4 |
| 4 | Design for the workflow moment, including interruption cost | Exercise 4; interview question 4 |
| 5 | Build a trust calibration plan covering explanation, uncertainty, and override | Exercise 5; self-check 6 |
| 6 | Run a staged rollout with a continuation gate | Exercise 6; interview question 7 |
| 7 | Measure dose-response and decay, and model their effect on retention | Exercise 7; self-check 8 |
| 8 | Connect adoption to unit economics and set a minimum viable adoption rate | Exercise 8; interview question 9 |

## 3. The core mental model

Adoption is a five-stage progression with a decay term, and the stages are not interchangeable.
A user who is exposed but never tries is not an adopter. A user who tries once and stops is not an
adopter. Only sustained use delivers benefit.

### 3.1 The adoption stack

```text
  EXPOSURE        the user can see the product in their workflow
       |          measured: eligible users with access
       v
  TRIAL           the user completes one real task with it
       |          measured: first-use within 14 days
       v
  USE             the user uses it on a share of eligible events
       |          measured: use rate at eligible encounters
       v
  SUSTAINED USE   the use rate holds after the novelty period
       |          measured: use rate at months 4 to 6
       v
  SUBSTITUTION    the user stops doing the old thing
                  measured: retirement of the workaround
```

Substitution is the stage that matters commercially and the stage least often measured. If the
clinician uses the new tool and still keeps the paper list, the time saving is negative, not
positive. A deployment that does not retire the workaround has added a step.

### 3.2 The decay curve

```text
  use
   |        ****
   |      **    **
   |    **        **
   |  **            ***
   | *                 ****
   |*                       ****
   +--------------------------------> months
     1   3   6   9  12  18  24
        ^           ^
        |           |
     novelty     decay unless
     peak        reinforced
```

The typical shape is a rapid rise to a peak at months two to four, a plateau, and then a decline
that begins between months six and nine. The decline is not caused by the product changing; it is
caused by staff turnover, competing priorities, and the gradual return of old habits under time
pressure.

Reinforcement is the only intervention that reliably changes the slope. Reinforcement means
something recurring: a monthly feedback report to the site lead, a re-training at the point of a
workflow change, a visible metric, or a champion who still uses the product. One-off training does
not reinforce; it delays the decay by a few weeks.

### 3.3 Dose-response

Benefit is a function of use intensity, not of licence count.

```text
benefit = eligible_events x adoption x use_rate x effect_per_event x conversion
```

This is the same chain as in Topic 02, with adoption and use rate as separate terms. Keeping them
separate matters because they have different interventions: adoption is a rollout and training
problem, while use rate is a workflow and trust problem.

A product with 90 percent adoption and 20 percent use rate at eligible events is a product that
everyone has been trained on and no one relies on.

### 3.4 Trust calibration

The goal is calibrated trust: reliance when the product is right, and appropriate scepticism when
it is wrong. Both over-trust and under-trust are failures.

```text
  UNDER-TRUST                     CALIBRATED                    OVER-TRUST
  ignored even when correct       reliance tracks accuracy      followed even when wrong
  cause: poor explanation,        cause: transparent             cause: authoritative
  poor accuracy, loss of          uncertainty, visible           presentation, no
  autonomy, inconsistent          performance, override path     uncertainty shown
  outputs
```

The intervention that most affects trust is not accuracy. It is whether the clinician can see why
the product said what it said, and whether they can disagree without fighting the software. A
conjoint study of 362 clinicians found that aversion was driven mainly by threats to epistemic
authority, particularly inconsistent outputs and an inability to modify the recommendation (EJIS,
2023).

The counter-risk is real. A randomised study of 220 physicians reading chest radiographs with
artificial intelligence (AI) advice found that feature-based local explanations improved accuracy
when the advice was correct, from 85.3 to 92.8 percent, but also increased unverified reliance
regardless of whether the advice was correct (Radiology, 2024). Explanation is not unambiguously
good; it shifts the error from under-reliance to over-reliance.

### 3.5 Redraw test

Draw the five stages, write the metric for each, and mark where your product currently loses
users. If the biggest drop is between exposure and trial, the problem is access or awareness. If
it is between use and sustained use, the problem is workflow or trust.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| Adoption | Sustained use of a product in the workflow | Equated with licence count | Proportion |
| Exposure | Share of eligible users with access | Assumed to equal adoption | Proportion |
| Trial | First real use within a defined window | Counted as adoption | Proportion |
| Use rate | Share of eligible events where the product is used | Confused with adoption | Proportion |
| Sustained use | Use rate that holds after the novelty period | Measured at week two | Proportion |
| Substitution | Retirement of the previous method | Assumed to follow use | Boolean |
| Dose | Intensity of use per user per period | Confused with adoption | Events per user |
| Dose-response | Relationship between use intensity and outcome | Assumed linear | Function |
| Decay | Decline in use over time without reinforcement | Attributed to product quality | Percentage points per month |
| Reinforcement | Recurring intervention that sustains use | Confused with initial training | Programme |
| Champion | Clinician who advocates and models use | Confused with a super-user | Role |
| Super-user | High-frequency user who provides support | Confused with a champion | Role |
| Onboarding | Process of bringing a user to first value | Confused with training | Process |
| Time to first value | Interval from access to first successful use | Not measured | Days |
| Activation | Point at which a user completes the core action | Confused with sign-up | Event |
| Monthly active user (MAU) | User with at least one use in a month | Used as a benefit metric | Count |
| Weekly active user | User with at least one use in a week | Used as a benefit metric | Count |
| Stickiness | Ratio of daily to monthly active users | Confused with retention | Ratio |
| Retention | Share of users still active after a period | Confused with satisfaction | Proportion |
| Churn | Loss of a user or account | Confused with non-renewal only | Proportion per period |
| Renewal | Contract continuation | Confused with retention | Event |
| Net revenue retention | Revenue retained plus expansion | Confused with gross retention | Proportion |
| Override | User rejects a recommendation | Counted as non-adoption | Event |
| Appropriate override | Rejection that is clinically correct | Rarely measured | Proportion |
| Alert fatigue | Sustained decline in appropriate response to alerts | Equated with high override rate | Construct |
| Alert burden | Alerts per user per shift | Confused with fatigue | Count |
| Workflow fit | Degree to which the product matches the real process | Confused with usability | Judgement |
| Interruption cost | Time and attention required at the moment of use | Omitted from value models | Seconds |
| Friction | Any step required beyond the clinical task | Underestimated | Steps |
| Cognitive load | Mental effort required to use the product | Not measured directly | Construct |
| Automation bias | Over-reliance on automated output | Confused with trust | Behaviour |
| Algorithm aversion | Under-reliance on algorithmic output | Confused with appropriate scepticism | Behaviour |
| Explainability | Ability to show the basis of a recommendation | Confused with accuracy | Property |
| Calibrated trust | Reliance proportional to accuracy | Assumed to follow explanation | Construct |
| Champion turnover | Loss of the clinical sponsor | Treated as a personnel issue | Event |
| Pilot effect | Inflated effect observed under pilot conditions | Used as the contract forecast | Bias |

## 5. The mechanics, step by step

The procedure produces an adoption plan with measurable stages, a continuation gate, and a
retention model. It assumes the product has been procured and is entering deployment.

### Step 1. Define the eligible population and the denominator

**Input:** the product's intended use and the buyer's workflow.
**Output:** a written definition of who and what is eligible.
**Rationale:** every adoption metric is a ratio, and the denominator determines the answer. A
vendor that reports adoption against licensed users will show a higher number than one that
reports against eligible clinicians, and a far higher number than one that reports against
eligible encounters.

Three denominators are possible and all three should be reported:

| Denominator | What it measures | Typical value |
|---|---|---|
| Licensed users | Penetration of the contract | 90 to 100 percent |
| Eligible clinicians | Reach into the target population | 40 to 80 percent |
| Eligible encounters | Actual use at the point of decision | 20 to 60 percent |

The third is the only one that predicts benefit. The first is a billing metric. Publishing the
first while implying the third is the most common misrepresentation in adoption reporting.

### Step 2. Choose the adoption unit to match the workflow

**Input:** the workflow analysis from discovery.
**Output:** a unit of use: per encounter, per shift, per patient, per order.
**Rationale:** a unit that does not correspond to a decision produces a metric that does not
correspond to value. A tool used once per shift to review a ward list has a shift-level unit; a
tool used at every order has an order-level unit.

Edge case: a product with multiple use modes needs multiple metrics. An ambient documentation
product used for some note types and not others has a use rate per note type, and averaging
across types conceals which specialties are failing.

### Step 3. Instrument the deployment before launch

**Input:** the metric definitions.
**Output:** an event taxonomy and a dashboard that exists on day one.
**Rationale:** adoption measured retrospectively from interviews is unreliable and measured from
claims is impossible. Instrumentation must be designed before go-live, because adding it later
means the first months are unmeasurable.

The minimum event set:

| Event | Definition | Purpose |
|---|---|---|
| Access granted | User has credentials and entitlement | Exposure |
| First use | User completes the core action once | Trial |
| Use | User completes the core action on an eligible event | Use rate |
| Override | User rejects or dismisses a recommendation | Trust |
| Workaround | User completes the task outside the product | Substitution |
| Abandonment | No use for 30 days after prior use | Decay |
| Support contact | User asks for help | Friction |

Privacy constraint: instrument the workflow, not the person. Aggregate to the site or the role
where individual measurement would be used for performance management, because surveillance
changes behaviour and invalidates the metric. In the United States, workforce monitoring is
subject to state law and to labour agreements; check before deploying user-level dashboards.

### Step 4. Segment users by expected benefit

**Input:** baseline workload data.
**Output:** three or four segments with different onboarding plans.
**Rationale:** the users with the highest baseline burden get the largest benefit and are the
most likely to sustain use. Spreading effort evenly produces a low average and a few successes.

A useful segmentation for a clinical tool:

| Segment | Signal | Expected use | Onboarding |
|---|---|---|---|
| High-volume, high-burden | Top quintile of eligible events | High | Intensive, early |
| High-volume, low-burden | Top quintile, efficient workflow | Moderate | Light |
| Low-volume, high-burden | Bottom quintile, inefficient | Moderate | Targeted |
| Low-volume, low-burden | Bottom quintile, efficient | Low | Self-service only |

The fourth segment should not receive intensive onboarding. The return is low and the cost is the
same as for the first.

### Step 5. Design for the workflow moment

**Input:** the observation data from discovery.
**Output:** a specified insertion point and interaction cost.
**Rationale:** the product must appear at the moment of the decision, with an interaction cost
that is smaller than the value it provides. A tool that requires leaving the workflow to use it
will be used by a minority.

Design rules:

- **In-flow by default.** The product should appear where the decision is made, not in a separate
  application or a portal.
- **Budget the interruption.** A recommendation that takes more than about 10 seconds to read and
  act on will be dismissed under time pressure. Measure the time, do not estimate it.
- **One decision per interaction.** A screen that asks for three judgements gets none of them.
- **Make deferral possible.** A user who cannot defer will dismiss, and dismissal is
  indistinguishable from rejection in the data.
- **Show the basis, not the model.** Clinicians need the evidence for the recommendation, not the
  feature importances.

The interaction cost deserves a budget, in the same way that a latency budget is set for an
interface. The budget is the time the user has available at that moment, which is usually much
smaller than the team assumes.

| Moment | Typical available time | Maximum interaction cost |
|---|---|---:|
| Before a procedure, patient waiting | 60 to 120 seconds | 30 seconds |
| During an order entry | 10 to 20 seconds | 8 seconds |
| During a note | 30 to 60 seconds | 20 seconds |
| Between patients | 2 to 5 minutes | 90 seconds |
| Asynchronous review | Unbounded | Unlimited |

Measure the cost with a stopwatch on five real users, not with an estimate from the team. In the
live usability literature, the expected sequential workflow rarely occurs and users are
interrupted in every session, so an interaction designed for an uninterrupted 90 seconds will be
used at a fraction of the intended rate.

### Step 6. Build the trust calibration plan

**Input:** the accuracy profile and the failure modes.
**Output:** a plan covering explanation, uncertainty, override, and feedback.
**Rationale:** trust is calibrated by experience, and experience is shaped by what the product
shows and what happens when the user disagrees.

The plan has four parts:

1. **Explanation.** Show the two or three inputs that drove the recommendation, in clinical
   language. Do not show a probability without a reference class.
2. **Uncertainty.** Show a range or a confidence band, and say what the product does when it is
   uncertain. A product that always appears confident trains over-trust.
3. **Override path.** Make disagreement one click, record the reason, and never require the user
   to justify disagreement to the software.
4. **Feedback loop.** Tell the user what happened after an override, at least in aggregate. A
   product that never reports back is a product the user cannot learn to trust.

### Step 7. Recruit and equip champions

**Input:** the segment map.
**Output:** two champions per site, with defined time and a defined role.
**Rationale:** champions are associated with adoption in most published studies, but the evidence
base is dominated by observational designs and presence-or-absence measures. A champion without
dedicated time is a name on a slide.

Design rules:

- Recruit at least two per site, from different roles, to survive turnover.
- Give the role a defined time allocation, typically 0.1 to 0.2 full-time equivalent during
  rollout.
- Give the champion a specific job: peer coaching, feedback collection, and escalation, not
  general enthusiasm.
- Choose champions with credibility among peers, not the most senior person available.
- Plan for replacement: name a successor at the start.

A review of 199 articles on champions found only four studies that randomly allocated champion
presence, and described champions as necessary but not sufficient, with multiple coordinated
champions often required (SAGE Open Medicine, 2018). A later mixed-studies review of 23 studies
warned specifically against the solo champion model (BMC Health Services Research, 2024).

### Step 8. Train for the task, not the tool

**Input:** the workflow and the failure modes.
**Output:** a training programme of 20 to 40 minutes, delivered at the point of need.
**Rationale:** clinicians do not need to understand the product; they need to complete a task.
Training that explains the architecture wastes the only attention the deployment will get.

Structure: one real scenario, one core action, one failure mode, one escalation path. Then
practice on the user's own data. Then a follow-up at two weeks, because the first week's memory
decays and the questions appear only after real use.

### Step 9. Run a staged rollout with a feedback loop

**Input:** the site list and the segment map.
**Output:** a wave plan with defined entry and exit criteria per wave.
**Rationale:** a simultaneous rollout across all sites produces a support queue that cannot be
served, and no comparison group. Waves allow the support model to be fixed between waves.

A wave plan:

| Wave | Sites | Purpose | Exit criterion |
|---|---|---|---|
| 1 | 3 to 5 | Prove the workflow and the support model | 50 percent use at eligible events |
| 2 | 8 to 12 | Prove repeatability with less support | 45 percent use, support tickets below a threshold |
| 3 | Remaining | Scale | 40 percent use at 90 days |

The exit criteria are continuation gates. A wave that does not meet the criterion is not a failure
of the product; it is a signal to fix the workflow before adding sites.

### Step 10. Measure dose-response and set the continuation gate

**Input:** the instrumented data.
**Output:** a dose-response curve and a minimum viable adoption rate.
**Rationale:** the continuation decision should be a threshold on use, not a judgement about
enthusiasm.

Derive the threshold from the economics. If the site's gross profit per month is G and the
acquisition or implementation cost is C, the payback period is C/G. If use below a certain rate
predicts churn within the payback period, that rate is the floor.

In the worked example, a site generates 1,968 United States dollars (USD) of gross profit per
month at 2,400 USD per month and an 82 percent margin. Against a 210,000 USD acquisition cost
spread across a 42-site system, four sites cover the cost within 24 months. The continuation gate
is therefore a use rate that keeps at least four sites active, which is a much lower bar than the
marketing target and a much more honest one.

### Step 11. Manage decay and reinforcement

**Input:** the use trend by site.
**Output:** a reinforcement calendar.
**Rationale:** decay is predictable and therefore manageable. The sites that will decay are
identifiable at month four by their use trajectory, and intervention at month four is cheaper
than intervention at month twelve.

Reinforcement mechanisms, in order of cost-effectiveness:

1. A monthly two-line feedback email to the site lead with their use rate and the peer median.
2. A quarterly review with the champion, focused on the sites below the median.
3. Re-training triggered by a workflow change, a new hire, or a drop below threshold.
4. A visible clinical outcome metric connected to use, where one exists.
5. Executive sponsorship renewed at each budget cycle.

The reinforcement calendar should be written as dates, not intentions, and it should be owned by
someone in the buyer's organisation after handover.

| Month | Reinforcement | Owner | Signal it addresses |
|---:|---|---|---|
| 2 | Site-level use report to leads | Vendor success team | Early workflow friction |
| 4 | Champion call; identify sites below median | Vendor and champion | Inflection detection |
| 6 | Peer comparison report to executives | Buyer sponsor | Priority drift |
| 9 | Re-training at sites below threshold | Champion | Skill decay |
| 12 | Gate review against continuation criteria | Buyer owner | Investment decision |
| 18 | New-hire onboarding refresh | Buyer training team | Staff turnover |
| 24 | Renewal review with outcome data | Buyer and vendor | Contract decision |

The calendar is not a communications plan. Each row is an intervention with a measurable target,
and each should be evaluated by whether the use rate at the targeted sites changed relative to
the sites that did not receive it. A reinforcement that cannot show a difference has not been
tested, only performed.

### Step 12. Hand over ownership and remove the training wheels

**Input:** the deployment and the metrics.
**Output:** a named internal owner and a documented operating model.
**Rationale:** a deployment that depends on the vendor's success team decays when that team moves
on. The handover is the point at which adoption becomes durable.

The handover package: the metric definitions, the dashboard, the escalation path, the training
material, the champion list with successors, and the review calendar. The named owner should be
in the buyer's organisation and should have the metric in their objectives. If no one owns it, it
will decay.

### 5.13 Branch points and edge cases

**No reimbursement pathway.** Without a payment route, adoption must be justified by cost
avoidance or quality, and the buyer's operational owner becomes the adoption owner. The metric
should be the operational one the owner is measured on, not a clinical outcome the owner does not
control.

**Single-payer versus multi-payer.** In a single-payer system, adoption is often driven by a
national programme with a mandated workflow, which produces high exposure and variable use. In a
multi-payer system, adoption varies by institution, and the vendor's support model must scale
across heterogeneous sites.

**Capital versus operating budget.** A capital purchase creates a multi-year commitment and a
strong incentive to make the product work, because the asset is on the books. An operating
subscription can be cancelled, so adoption decay converts directly into churn. The same decay
curve has different financial consequences.

**Pilot-to-contract transition.** The pilot's users are volunteers and the pilot's support is
intensive. The contract-year plan should assume a 15 to 25 percentage point reduction in use rate
and budget reinforcement to offset it.

**Procurement security review.** Security constraints can prevent the instrumentation that
adoption measurement requires, for example a prohibition on user-level analytics. Design the
measurement plan around the constraint before signing, not after.

**Champion turnover.** When a champion leaves, use at that site declines within one to two months.
Succession planning and a second champion are the mitigations, and the decline should be expected
in the retention model.

**Alert fatigue and override.** A high override rate is not proof of fatigue. A meta-analysis of
16 studies found an overall override rate of 90 percent, but a review of 22 systematic reviews
found that only one offered an operational definition of alert fatigue, and that the
inappropriate-override rate was reported in only 13.6 percent of reviews. Measure the
appropriateness of overrides, not only their frequency.

**Regulatory constraint on claims.** If the product is a device, the claims made in training and
in the interface are constrained by the cleared intended use. Training material that overstates
the benefit creates a promotional compliance risk.

### 5.14 Decision table for adoption response

| Exposure | Trial | Use rate | Sustained | Diagnosis | Action |
|---|---|---|---|---|---|
| Low | Low | Low | Low | Access or awareness problem | Fix entitlement, communication, and single sign-on |
| High | Low | Low | Low | First-use friction | Reduce time to first value; add in-flow entry |
| High | High | Low | Low | Workflow or trust problem | Redesign insertion point; add explanation and override |
| High | High | Moderate | High | Working product with a supportable minority | Scale support; target the non-users |
| High | High | High | Falling | Decay | Reinforce; check champion status and staff turnover |
| High | High | High | High | Sustained adoption | Protect; monitor for workflow change |
| High | High | High | High but overrides inappropriate | Over-trust | Add uncertainty display and audit |
| Low | Any | Any | Any | Contract without deployment | Escalate; the buyer is not implementing |

The last row is a commercial emergency, not an adoption problem. A contract that has not been
deployed will not renew, and the cause is usually a stalled internal owner rather than the
product.

## 6. Worked example

> **Note:** the health system and the product below are a composite constructed for teaching. All
> financial figures are illustrative 2026 USD. The alert parameters in Section 6.7 are taken from
> a published external validation and are cited as such.

### 6.1 The deployment

A six-hospital system with 42 ambulatory clinics licenses a clinical documentation and order-set
assistant at 2,400 USD per clinic per month. The contract covers 1,848 clinicians. The system
plans to bring three clinics live per month over fourteen months.

The vendor's acquisition cost for the enterprise deal is 210,000 USD, and the gross margin is
82 percent, giving 1,968 USD of gross profit per clinic per month.

The question is not whether the product can be deployed. It is whether use will be high enough,
and last long enough, for the account to be worth what it cost to win.

### 6.2 The adoption trajectory

Use is modelled per site as a ramp to a peak followed by decay:

```text
use(month) = peak x (1 - exp(-month / 2.2)) - decay x max(0, month - decay_start)
```

Two scenarios:

| Parameter | Standard onboarding | Reinforced programme |
|---|---:|---:|
| Peak use | 0.52 | 0.64 |
| Decay starts | month 8 | month 12 |
| Decay rate | 1.2 points per month | 0.4 points per month |
| Champion time | 0.05 FTE | 0.15 FTE |
| Feedback | Quarterly | Monthly |

The resulting trajectories:

| Month | Standard use | Standard renewal probability | Reinforced use | Reinforced renewal probability |
|---:|---:|---:|---:|---:|
| 3 | 0.387 | 0.60 | 0.476 | 0.75 |
| 6 | 0.486 | 0.76 | 0.598 | 0.95 |
| 9 | 0.499 | 0.79 | 0.629 | 0.95 |
| 12 | 0.470 | 0.74 | 0.637 | 0.95 |
| 18 | 0.400 | 0.62 | 0.616 | 0.95 |
| 24 | 0.328 | 0.51 | 0.592 | 0.94 |
| 30 | 0.256 | 0.39 | 0.568 | 0.90 |
| 36 | 0.184 | 0.30 | 0.544 | 0.86 |

Renewal probability is a continuous function of trailing use: 0.30 at 20 percent use, rising
linearly to 0.95 at 60 percent use. That function is an assumption, but it encodes the empirical
pattern that low-use sites churn and high-use sites renew.

### 6.3 Retention and revenue

| Metric | Standard | Reinforced |
|---|---:|---:|
| Active clinics at month 12 | 12.0 | 12.0 |
| Active clinics at month 24 | 10.9 | 13.4 |
| Active clinics at month 36 | 6.0 | 12.6 |
| Monthly recurring revenue at month 36 | 14,342 USD | 30,196 USD |
| Cumulative gross profit, 36 months | 630,774 USD | 779,486 USD |
| Acquisition cost | 210,000 USD | 210,000 USD |
| Payback month | 15 | 15 |
| Cumulative gross profit divided by acquisition cost | 3.00 | 3.71 |

Two results deserve attention.

First, the payback month is identical at 15, because the early months are similar in both
scenarios. Payback is a poor discriminator of adoption quality in the first year. The difference
appears in the terminal revenue: 14,342 USD per month against 30,196 USD per month at month 36, a
factor of 2.1.

Second, the standard scenario ends with 6 of 42 clinics active after 36 months, which is a
retention rate of 14 percent at the site level. That is not a product failure; it is the
compounding of a decay rate that looks small month to month.

### 6.4 Payback arithmetic and the continuation gate

| Calculation | Value |
|---|---:|
| Gross profit per clinic-month | 1,968 USD |
| Clinics needed for 24-month payback | 4.4 |
| Clinics needed for 36-month payback | 3.0 |
| Clinics active at month 36, standard | 6.0 |
| Clinics active at month 36, reinforced | 12.6 |

The continuation gate that follows is deliberately modest: keep at least five clinics above 40
percent use at month 12. Five clinics is enough to cover the acquisition cost within 36 months,
and 40 percent use is the level below which renewal probability falls under 0.65.

That gate is far below the deployment target of 42 clinics, which is the point. A gate set at the
target is not a gate; it is a restatement of the plan.

### 6.5 Two diagnostics that reveal a weak adoption story

**Diagnostic 1: the licence-count test.** The contract covers 1,848 clinicians. At 35 percent
monthly active use, 647 clinicians use the product in a given month; at 48 percent, 887; at 62
percent, 1,146. A vendor reporting "1,848 users" is reporting the contract. A buyer reporting
"1,848 users" in a board paper is reporting a number that no operational metric supports. Ask for
monthly active users by site, and compare with the licensed count. A ratio below 0.4 means the
deployment has not happened, whatever the contract says.

**Diagnostic 2: the decay test.** Compare the use rate at month 3 with the use rate at month 6. In
the standard scenario, use rises from 0.387 to 0.486, which looks healthy. But the model's decay
begins at month 8, and by month 12 the rate has fallen to 0.470 and by month 18 to 0.400. A
six-month review that ends at month 6 will conclude the deployment succeeded and will not fund
reinforcement, which is exactly when reinforcement is cheapest.

The practical version of this test: plot use by site on a monthly basis and look for the
inflection, not the level. A site whose use has stopped rising at month 4 is a site that will
decay.

### 6.6 Varying one assumption

Hold the decay parameters at the standard scenario and vary peak adoption.

| Peak use | Active clinics at month 24 | Cumulative gross profit, 36 months | Multiple of acquisition cost |
|---:|---:|---:|---:|
| 0.34 | 7.4 | 498,107 USD | 2.37 |
| 0.42 | 8.9 | 550,621 USD | 2.62 |
| 0.52 | 10.9 | 630,774 USD | 3.00 |
| 0.60 | 12.4 | 700,706 USD | 3.34 |
| 0.68 | 13.4 | 754,841 USD | 3.59 |
| 0.76 | 13.4 | 773,798 USD | 3.68 |

The relationship is concave. Moving peak use from 0.34 to 0.52 adds 132,667 USD of cumulative
gross profit; moving from 0.52 to 0.68 adds 124,067 USD; moving from 0.68 to 0.76 adds only 18,957
USD. The reason is the renewal function's ceiling: above about 60 percent use, renewal probability
is at its maximum of 0.95, and further adoption gains do not change retention.

That ceiling has a strategic implication. Investment in adoption is worth funding up to the point
where the account reaches the renewal ceiling, and beyond that point the money is better spent on
expansion or on the next account.

Now vary decay instead, holding peak at 0.52.

| Decay (points per month after month 8) | Active clinics at month 24 | Cumulative gross profit, 36 months | Multiple |
|---:|---:|---:|---:|
| 0.000 | 11.8 | 701,578 USD | 3.34 |
| 0.004 | 11.5 | 677,147 USD | 3.22 |
| 0.008 | 11.2 | 653,545 USD | 3.11 |
| 0.012 | 10.9 | 630,774 USD | 3.00 |
| 0.018 | 10.4 | 598,173 USD | 2.85 |
| 0.024 | 9.9 | 577,981 USD | 2.75 |

A decay rate of 2.4 points per month instead of 1.2 costs 52,793 USD of cumulative gross profit,
or 8 percent, over three years. That is the marginal value of halving the decay rate, and it is
the number to put against the cost of a reinforcement that only slows decay. The full reinforced
programme, which also raises peak adoption from 0.52 to 0.64, is worth 148,712 USD over the same
period, the difference between 779,486 and 630,774 USD.

The comparison matters because the two interventions have different costs. A 0.15 full-time-
equivalent champion at a loaded 180,000 USD per year costs 27,000 USD per year, or 81,000 USD over
three years. That is justified by the full programme's 148,712 USD gain, a return of about 1.8 to
1, but not by the decay-only gain of 52,793 USD. A vendor that funds champions solely to slow decay
at this account size will not recover the cost; a vendor that funds them to raise peak adoption as
well will.

### 6.7 The alert case: when adoption is measurably harmful

Adoption of the wrong product at the wrong threshold has a cost, and the clearest public example
is the external validation of a widely implemented proprietary sepsis model.

The validation studied 38,455 hospitalisations in 27,697 patients, with sepsis in 7 percent. At
the threshold used in practice, the model alerted on 18 percent of all hospitalisations and
missed 67 percent of sepsis cases, with a positive predictive value of 12 percent and a number
needed to evaluate of 8 (JAMA Internal Medicine, 2021, doi:10.1001/jamainternmed.2021.2626).

Applying those published proportions to a system of this size:

| Quantity | Value |
|---|---:|
| Sepsis cases per year | 2,692 |
| Alerts per year | 6,922 |
| True positives | 888 |
| False positives | 6,034 |
| Positive predictive value | 12.8 percent |
| Review time at 3.5 minutes per alert | 404 hours per year |
| Review cost at 58 USD per hour | 23,419 USD per year |

The review cost alone is modest. The real cost is the 6,034 false alerts, each of which teaches
the clinician that the alert is usually wrong. That is how an override rate reaches 90 percent,
and it is why override rate alone is a poor measure: the clinicians may be right to override.

The adoption lesson is that a high use rate on a low-precision alert can be worse than low use.
The metric that matters is appropriate action per true positive, not alerts acknowledged.

### 6.8 The recommendation

The analysis supports a reinforced rollout to 12 sites, not 42, with a continuation gate at month
12 and expansion contingent on the gate. The reinforcement budget should be capped at the
retention value it creates, which at this account size means a shared champion across sites rather
than one per site.

It also supports a change to the reporting: the monthly business review should report monthly
active clinics, use rate at eligible encounters, and site-level renewal probability, not licensed
users. The licensed-user number should appear once, in the contract summary, and never in the
value section.

## 7. Code, query, or calculation

The script models site-level adoption, annual renewal, revenue retention, and the alert economics.

```python
"""Adoption trajectory, annual retention, and enterprise unit economics."""

NEW_PER_MONTH = 3
MONTHS_ONBOARD = 14
HORIZON = 36
PRICE = 2_400.0      # USD per clinic per month
GM = 0.82
CAC = 210_000.0      # USD, one enterprise sale covering 42 clinics
SITES = 42


def use_rate(month, peak, decay_start, decay_pp):
    ramp = peak * (1 - 2.718281828 ** (-month / 2.2))
    if month > decay_start:
        ramp -= decay_pp * (month - decay_start)
    return max(ramp, 0.06)


def renewal_probability(u):
    """0.30 at 20% use, rising linearly to 0.95 at 60% use."""
    return 0.30 + 0.65 * min(1.0, max(0.0, (u - 0.20) / 0.40))


def simulate(peak, decay_start, decay_pp):
    cohorts = [{"start": s, "alive": 1.0} for s in range(1, MONTHS_ONBOARD + 1)]
    active, mrr, gp = [], [], []
    for m in range(1, HORIZON + 1):
        n = 0.0
        for c in cohorts:
            if m < c["start"]:
                continue
            age = m - c["start"] + 1
            if age > 1 and (age - 1) % 12 == 0:
                trailing = sum(use_rate(age - k, peak, decay_start, decay_pp)
                               for k in range(3)) / 3
                c["alive"] *= renewal_probability(trailing)
            n += c["alive"]
        active.append(n)
        mrr.append(n * PRICE)
        gp.append(n * PRICE * GM)
    cum = sum(gp)
    payback, running = None, 0.0
    for i, g in enumerate(gp, start=1):
        running += g
        if running >= CAC and payback is None:
            payback = i
    return dict(active=active, mrr=mrr, cum=cum, payback=payback)


base = simulate(0.52, 8, 0.012)
rein = simulate(0.64, 12, 0.004)

print("ADOPTION AND RETENTION, 36 MONTHS")
print("  scenario    active m12  active m24  active m36    MRR m36")
for label, s in (("standard", base), ("reinforced", rein)):
    print(f"  {label:10s} {s['active'][11]:10.1f} {s['active'][23]:11.1f}"
          f" {s['active'][35]:11.1f}  ${s['mrr'][35]:10,.0f}")

print("\nENTERPRISE UNIT ECONOMICS")
print("  scenario    cum GP 36m      CAC  payback  GP/CAC")
for label, s in (("standard", base), ("reinforced", rein)):
    print(f"  {label:10s} ${s['cum']:12,.0f}  ${CAC:9,.0f}"
          f"  {s['payback']:7d}  {s['cum'] / CAC:5.2f}")

print("\nUSE AND RENEWAL, SINGLE SITE")
print("  month  use std  renew std  use rein  renew rein")
for m in (3, 6, 9, 12, 18, 24, 30, 36):
    ub = use_rate(m, 0.52, 8, 0.012)
    ur = use_rate(m, 0.64, 12, 0.004)
    print(f"  {m:5d}  {ub:7.3f}  {renewal_probability(ub):9.2f}"
          f"  {ur:8.3f}  {renewal_probability(ur):10.2f}")

print("\nPAYBACK ARITHMETIC")
per_site = PRICE * GM
print(f"  gross profit per site-month  ${per_site:,.0f}")
print(f"  sites for 24-month payback   {CAC / (per_site * 24):.1f}")
print(f"  sites for 36-month payback   {CAC / (per_site * 36):.1f}")

print("\nSENSITIVITY: peak adoption")
print("  peak  active m24   cum GP 36m   GP/CAC")
for peak in (0.34, 0.42, 0.52, 0.60, 0.68, 0.76):
    s = simulate(peak, 8, 0.012)
    print(f"  {peak:4.2f}  {s['active'][23]:9.1f}   ${s['cum']:11,.0f}   {s['cum'] / CAC:5.2f}")

print("\nSENSITIVITY: decay after month 8")
print("  pp/month  active m24   cum GP 36m   GP/CAC")
for d in (0.0, 0.004, 0.008, 0.012, 0.018, 0.024):
    s = simulate(0.52, 8, d)
    print(f"  {d:9.3f}  {s['active'][23]:10.1f}   ${s['cum']:11,.0f}   {s['cum'] / CAC:5.2f}")

print("\nLICENCE VERSUS ACTIVE USE")
licences = SITES * 44
for share in (0.35, 0.48, 0.62):
    print(f"  licensed {licences:,}; monthly active {licences * share:,.0f}"
          f" ({share:.0%})")

print("\nALERT ECONOMICS (published sepsis model parameters)")
admissions, sepsis_rate, alert_rate, sens = 38_455, 0.07, 0.18, 0.33
cases = admissions * sepsis_rate
alerts = admissions * alert_rate
detected = cases * sens
print(f"  sepsis cases      {cases:,.0f}")
print(f"  alerts            {alerts:,.0f}")
print(f"  true positives    {detected:,.0f}")
print(f"  false positives   {alerts - detected:,.0f}")
print(f"  PPV               {detected / alerts:.1%}")
print(f"  review hours/yr   {alerts * 3.5 / 60:,.0f}")
print(f"  review cost/yr    ${alerts * 3.5 / 60 * 58:,.0f}")
```

Expected output:

```text
ADOPTION AND RETENTION, 36 MONTHS
  scenario    active m12  active m24  active m36    MRR m36
  standard         12.0        10.9         6.0  $    14,342
  reinforced       12.0        13.4        12.6  $    30,196

ENTERPRISE UNIT ECONOMICS
  scenario    cum GP 36m      CAC  payback  GP/CAC
  standard    $     630,774  $  210,000       15   3.00
  reinforced  $     779,486  $  210,000       15   3.71

USE AND RENEWAL, SINGLE SITE
  month  use std  renew std  use rein  renew rein
      3    0.387       0.60     0.476        0.75
      6    0.486       0.76     0.598        0.95
      9    0.499       0.79     0.629        0.95
     12    0.470       0.74     0.637        0.95
     18    0.400       0.62     0.616        0.95
     24    0.328       0.51     0.592        0.94
     30    0.256       0.39     0.568        0.90
     36    0.184       0.30     0.544        0.86

PAYBACK ARITHMETIC
  gross profit per site-month  $1,968
  sites for 24-month payback   4.4
  sites for 36-month payback   3.0

SENSITIVITY: peak adoption
  peak  active m24   cum GP 36m   GP/CAC
  0.34        7.4   $    498,107    2.37
  0.42        8.9   $    550,621    2.62
  0.52       10.9   $    630,774    3.00
  0.60       12.4   $    700,706    3.34
  0.68       13.4   $    754,841    3.59
  0.76       13.4   $    773,798    3.68

SENSITIVITY: decay after month 8
  pp/month  active m24   cum GP 36m   GP/CAC
      0.000        11.8   $    701,578    3.34
      0.004        11.5   $    677,147    3.22
      0.008        11.2   $    653,545    3.11
      0.012        10.9   $    630,774    3.00
      0.018        10.4   $    598,173    2.85
      0.024         9.9   $    577,981    2.75

LICENCE VERSUS ACTIVE USE
  licensed 1,848; monthly active 647 (35%)
  licensed 1,848; monthly active 887 (48%)
  licensed 1,848; monthly active 1,146 (62%)

ALERT ECONOMICS (published sepsis model parameters)
  sepsis cases      2,692
  alerts            6,922
  true positives      888
  false positives   6,034
  PPV               12.8%
  review hours/yr      404
  review cost/yr    $23,419
```

The model has two deliberate simplifications. It treats every site as having the same peak use,
which is false and which understates variance; a real portfolio has a bimodal distribution of
high users and non-users, and the average describes neither. And it models renewal as a function
of trailing use only, ignoring the relationship, the price, and the competitive alternatives,
which in practice matter at renewal.

The first simplification is the more consequential. A portfolio with half the sites at 70 percent
use and half at 20 percent has the same average as a portfolio with all sites at 45 percent, and a
very different retention profile, because the low-use sites churn.

## 8. Data, standards, and regulatory context

### 8.1 Implementation science frameworks

Three frameworks are widely used and each answers a different question.

| Framework | Question | Use in a deployment | Limitation |
|---|---|---|---|
| RE-AIM | Reach, Effectiveness, Adoption, Implementation, Maintenance | Structuring an evaluation across the deployment | Descriptive, not predictive |
| CFIR | Consolidated Framework for Implementation Research | Identifying barriers across context, intervention, and process | Complex; needs tailoring |
| NASSS | Nonadoption, abandonment, scale-up, spread, sustainability | Diagnosing why scale fails | Framework, not a measurement instrument |

RE-AIM's five dimensions map closely to the metrics in this topic: reach is exposure, adoption is
site-level uptake, implementation is fidelity, and maintenance is sustained use. Using the
framework's vocabulary makes the plan legible to implementation scientists and to funders, but it
does not supply thresholds; those come from the economics.

### 8.2 Measurement definitions are not standardised

There is no binding standard for what "adoption" means. A systematic review of 22 systematic
reviews of alert fatigue measurement found that only one offered an operational definition of
alert fatigue, and that the most-used metrics were alert quantity (54.5 percent of reviews),
override rate (45.5 percent), and acceptance rate (45.5 percent), while the inappropriate-override
rate was reported in only 13.6 percent (JAMIA, 2026).

The consequence is that two vendors can both report "high adoption" using incompatible
definitions. A buyer should require the definition to be written into the contract, with the
denominator, the numerator, the measurement window, and the data source.

A defensible contract definition:

> Monthly active use rate means the number of eligible encounters in the month in which the
> product's core action was completed, divided by the number of eligible encounters in the month,
> measured from the vendor's event log and reconciled to the customer's encounter count.

That definition is auditable, and it is short enough to fit in a schedule.

### 8.3 Transparency obligations for decision support

If the product is a decision support intervention embedded in certified health information
technology, the Office of the National Coordinator for Health Information Technology (ONC) HTI-1
final rule, effective January 2024, requires transparency about source attributes and intervention
risk management for predictive decision support interventions. The requirements are about
disclosure, not adoption, but they interact: a product that cannot describe its basis in the
interface is both less adoptable and less compliant.

For devices, the Food and Drug Administration's clinical decision support guidance (final,
September 2022) constrains the claims that can be made in training and in the interface. Adoption
material that promises an outcome outside the cleared intended use is a promotional issue.

### 8.4 Accessibility and usability standards

Section 508 and WCAG 2.2 AA apply to public-sector buyers and are increasingly required
contractually elsewhere. An inaccessible product excludes users with disabilities, which is both
an equity failure and an adoption failure in the segments that need the product most.

Usability is not the same as accessibility. A product can be fully accessible and still unusable
in a four-minute encounter. Measure task completion time under realistic interruption, as the
live usability studies in Section 9 did.

### 8.5 Data protection limits on measurement

Instrumenting adoption means collecting behavioural data about clinicians. In the European Union,
that is personal data under the General Data Protection Regulation, and workforce monitoring may
require consultation with employee representatives. In the United States, state laws and labour
agreements constrain monitoring, and the HIPAA Privacy Rule constrains any use of patient data in
the analytics.

The practical design principle is to measure the workflow, not the person, and to aggregate to the
smallest group that supports the decision. A site-level dashboard is usually sufficient; a
clinician-level leaderboard usually is not, and it changes behaviour in ways that invalidate the
measurement.

## 9. Industry practice and case evidence

### 9.1 Documented: override rates are high and mostly unexamined

A 2024 systematic review and meta-analysis of 16 studies found an overall drug-drug interaction
alert override rate of 90 percent (95 percent confidence interval 85 to 95 percent), with alerts
affecting about 13 percent of orders. High override rates persisted even after interventions that
reduced alert volume (Health Informatics Journal, 2024).

An earlier single-system study found 93 percent of 2,455 alerts overridden and, notably, no
correlation between the number of alerts a physician saw and the override rate, which undercuts
the simple quantity model of alert fatigue (Applied Clinical Informatics, 2014). A study at a
793-bed tertiary centre found 73.3 percent of alerts overridden but only about 60 percent of those
overrides appropriate, meaning roughly 40 percent were inappropriate (2017).

The lesson for adoption measurement: override rate is not a measure of product rejection, and
acceptance rate is not a measure of appropriate use. Both must be paired with an appropriateness
assessment, which almost no deployment conducts.

### 9.2 Documented: a widely adopted model with poor discrimination

Wong and colleagues externally validated the Epic Sepsis Model at Michigan Medicine in 27,697
patients and 38,455 hospitalisations. Discrimination was an area under the receiver operating
characteristic curve of 0.63 (95 percent confidence interval 0.62 to 0.64), against developer
reported values of 0.76 to 0.83. Calibration was poor. At the clinical threshold, the model
alerted on 18 percent of hospitalisations and missed 67 percent of sepsis cases, with a positive
predictive value of 12 percent (JAMA Internal Medicine, 2021;181(8):1065-1070,
doi:10.1001/jamainternmed.2021.2626).

This is the clearest public case of adoption without validity. The product was widely deployed,
which is a commercial success and a clinical problem. It also shows why adoption is not a
sufficient success metric: a product can be used and still be wrong.

### 9.3 Documented: adoption intensity determines benefit

In the team-based documentation study of 18,265 physicians, high-intensity adopters saved about 54
minutes per week of note time (21 percent), rising to 72 minutes per week (28 percent) after a
learning curve of about 20 weeks, and saw 2.5 to 4.5 additional visits per week. Low-intensity
adopters saw no time savings (JAMA Internal Medicine, 2024).

In the ambient documentation evaluation of 45 physicians, the tool was used in 55 percent of
17,428 encounters and produced a median note-time reduction of 0.57 minutes, with the authors
describing the savings as modest and highly heterogeneous (JAMIA, 2024). A separate evaluation at
Intermountain Health with 99 providers found positive engagement trends but no productivity gain,
and a statistically significant worsening of after-hours EHR time (JAMIA, 2024).

Taken together, these results say that the same product can produce a benefit, no benefit, or a
harm depending on who uses it and how. That is the empirical case for adoption as a first-class
design variable.

### 9.4 Documented: trust responds to explanation, with a cost

A randomised study of 220 physicians reading chest radiographs with AI advice found that
feature-based local explanations improved diagnostic accuracy from 85.3 to 92.8 percent when the
AI advice was correct, and produced faster decisions. The same explanations increased unverified
reliance regardless of whether the advice was correct (Radiology, 2024,
doi:10.1148/radiol.233261).

A conjoint study of 362 clinicians found that algorithm aversion was driven mainly by threats to
epistemic authority, particularly inconsistent outputs at the point of care and an inability to
modify the recommendation (European Journal of Information Systems, 2023).

The design implication is that explanation and user control are not optional features. They
determine whether the product is trusted, and they determine whether trust is calibrated.

### 9.5 Documented: champions help, but the evidence is weak

A review of 199 articles on champions found only four studies that randomly allocated champion
presence. Seven studies reported odds ratios, six of them positive, with values from 1.85 to 4.14.
The authors characterised champions as necessary but not sufficient, and noted that multiple
coordinated champions are often required (SAGE Open Medicine, 2018).

A systematic review of 35 quantitative studies found that 71.4 percent measured champion presence
or absence rather than champion activity, and that the association with organisational uptake was
positive in five of seven studies, while evidence at the provider and patient level was mixed or
scarce (Implementation Science Communications, 2022).

The practical conclusion: fund champions, but measure what they do, not whether they exist.

### 9.6 Inference: what this implies for deployment design

The following is inference from the sources above, not a documented finding.

First, a deployment plan should specify the use rate it is trying to achieve and the evidence
that rate is sufficient for the benefit claimed. A plan without a target rate is a plan to
deploy, not a plan to deliver.

Second, the first ninety days determine the next three years. The decay inflection is visible by
month four, and the sites that will churn are identifiable then.

Third, the highest-return adoption investment is usually the removal of a single workflow step,
not more training. Training changes knowledge; friction changes behaviour.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Licence count reported as adoption | Dashboard shows entitled users only | No event instrumentation | Benefit overstated, churn surprise | Report MAU and use rate at eligible events |
| Pilot effect used as forecast | Pilot use rate above 80 percent | Volunteers and extra support | Contract-year shortfall | Discount pilot use by 15 to 25 points |
| No denominator definition | "Adoption is high" with no number | Metric not specified | Unfalsifiable claims | Contract the definition |
| Out-of-workflow design | Users keep a parallel tool | Insertion point wrong | Negative time saving | Redesign the entry point |
| Interruption cost too high | Users dismiss without reading | Interaction exceeds available seconds | Override at scale | Reduce to one decision per interaction |
| No override path | Users work around the product | Autonomy threat | Abandonment | One-click override with reason capture |
| Over-trust | High acceptance, inappropriate overrides rare | Confident interface, no uncertainty | Harm from wrong advice | Show uncertainty, audit accepted cases |
| Solo champion | One name in every plan | Key-person dependency | Collapse on departure | Two champions plus successor |
| Training without reinforcement | Use falls after month 6 | Novelty effect ends | Silent decay | Monthly feedback, quarterly review |
| No continuation gate | Expansion continues regardless | Plan divorced from evidence | Scale a failing deployment | Gate on use rate at month 12 |
| Instrumentation blocked by privacy | No user-level analytics permitted | Security or labour constraint | Cannot measure adoption | Design aggregate metrics pre-contract |
| Wrong product adopted well | High use, poor precision | Adoption without validity | Harm and fatigue | Measure appropriateness, not volume |

Two red flags are diagnostic. The first is an adoption figure reported without a denominator;
that is not a metric. The second is a deployment plan whose success criterion is "go-live", which
measures the vendor's work rather than the buyer's outcome.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | Use rate predicts benefit | The economic model depends on it | Benefit saturates at low use | Measure outcome by use decile |
| 2 | The renewal function is monotonic in use | Retention model needs a form | Relationship or price dominates | Analyse churn by use band |
| 3 | Sites have similar trajectories | Simplifies the portfolio model | Bimodal distribution | Segment the site cohort |
| 4 | Decay is linear after the inflection | Model tractability | Step changes from staff turnover | Plot monthly site-level use |
| 5 | Champion time is the main reinforcement | Budget allocation | Feedback and workflow changes dominate | Run a reinforcement trial |
| 6 | The eligible population is stable | Denominator consistency | Seasonality or case-mix change | Compare denominators month to month |
| 7 | Overrides reflect judgement | Trust measurement | Overrides reflect fatigue or time pressure | Audit override appropriateness |
| 8 | Training transfers to practice | Onboarding design | Knowledge decays in two weeks | Test task completion at day 14 |
| 9 | Event logs are complete | All metrics depend on them | Client-side blocking or offline use | Reconcile with encounter counts |
| 10 | The buyer will act on the metric | Gates require action | No owner for the number | Name the owner in the contract |

The deepest limit is that adoption measurement describes behaviour and not causation. A site with
high use may be a site with an easy workflow, a motivated leader, and a favourable case mix, and
the product may contribute little. Only a comparison, ideally a stepped rollout, distinguishes the
product's contribution from the site's.

## 12. Exercises

### Exercise 1. Adoption metric definition

**Deliverable:** a one-paragraph definition of monthly active use rate for a product you know,
with numerator, denominator, window, and data source.
**Criteria:** the denominator is eligible events, not licensed users; the definition is auditable
from a log; the window is stated. **Time:** 30 minutes.

### Exercise 2. Stage diagnosis

**Deliverable:** for three described deployments, identify the stage at which users are lost and
the corresponding intervention.
**Criteria:** each diagnosis names a stage from Section 3.1 and an intervention from Section 5.14;
no diagnosis is "poor usability" without a specific stage. **Time:** 45 minutes.

### Exercise 3. Instrumentation plan

**Deliverable:** an event taxonomy with at least seven events, their properties, and the dashboard
that would use them.
**Criteria:** includes exposure, trial, use, override, workaround, and abandonment; each event has
a definition and a privacy note. **Time:** 90 minutes.

### Exercise 4. Workflow insertion design

**Deliverable:** a description of where the product appears in a specified workflow, with the
interaction cost in seconds and the failure path.
**Criteria:** the insertion point is in-flow; the interaction cost is estimated and justified; the
failure path is specified. **Time:** 60 minutes.

### Exercise 5. Trust plan

**Deliverable:** a one-page plan covering explanation, uncertainty display, override, and feedback
for a decision support product.
**Criteria:** all four parts present; the uncertainty display is specified, not asserted; the
override path is one click. **Time:** 60 minutes.

### Exercise 6. Rollout waves

**Deliverable:** a three-wave plan with site counts, support model, and exit criteria per wave.
**Criteria:** exit criteria are numeric; the support model changes between waves; the gate owner
is named. **Time:** 60 minutes.

### Exercise 7. Retention model

**Deliverable:** a cohort model with monthly use, annual renewal probability, active sites over 36
months, and cumulative gross profit.
**Criteria:** renewal is a stated function of use; cohorts tracked separately; the payback month
computed. **Time:** 2 hours.

### Exercise 8. Minimum viable adoption

**Deliverable:** for a product with 3,000 USD monthly gross profit per site, a 250,000 USD
acquisition cost, and a 36-month payback target, compute the number of sites required and state
the resulting continuation gate.
**Criteria:** correct arithmetic; the gate expressed as a use rate with a justification; the gate
distinguished from the deployment target. **Time:** 30 minutes.

## 13. Self-check answers

**1. Why must an adoption metric have an explicit denominator?**
Because adoption is a ratio, and the denominator determines the value. Licensed users, eligible
clinicians, and eligible encounters give very different numbers for the same deployment, and only
the third predicts benefit.

**2. What is the difference between use rate and adoption rate?**
Adoption rate is the share of eligible users who use the product at all. Use rate is the share of
eligible events on which an adopter uses it. A clinician can be an adopter and use the product on
20 percent of eligible encounters.

**3. What are the five stages of the adoption stack?**
Exposure, trial, use, sustained use, and substitution. Substitution, the retirement of the old
method, is the stage that determines whether the product created value or added a step.

**4. What is the minimum event set for instrumentation?**
Access granted, first use, use on an eligible event, override, workaround, abandonment, and
support contact. Each maps to a stage or to friction.

**5. Why is a pilot's adoption rate not a forecast?**
Because the pilot has volunteer users, intensive support, and a short duration that ends before
the novelty effect fades. The contract-year rate is typically 15 to 25 percentage points lower.

**6. What are the four parts of a trust calibration plan?**
Explanation of the basis, display of uncertainty, a one-click override path, and a feedback loop
that tells the user what happened.

**7. What is a continuation gate and how is it set?**
A numeric threshold on use rate at a defined date, below which expansion stops and the workflow is
fixed. It is derived from the economics, specifically the use rate below which renewal probability
falls inside the payback period.

**8. Compute the Exercise 8 answer.** Sites required = 250,000 / (3,000 x 36) = 2.3, rounded up to
3 sites. The continuation gate should be the lowest use rate that keeps at least three sites
active beyond the payback period, which requires knowing the renewal function; if renewal falls
below 0.5 at 30 percent use, the gate is 30 percent.

**9. Why is a high override rate not proof of alert fatigue?**
Because overrides can be clinically correct, and because a 2026 review found that only one of 22
systematic reviews defined alert fatigue operationally. The appropriate measure is the
inappropriate-override rate, which is rarely reported.

**10. What does the sepsis model validation imply about adoption as a success metric?**
That a product can be widely adopted and still be wrong. The model alerted on 18 percent of
hospitalisations with a positive predictive value of 12 percent, which means most alerts were
false, and the adoption was a clinical problem rather than a success.

**11. Why does the champion evidence base support funding champions but not counting them?**
Because most studies measure presence or absence rather than activity, and only four randomised
studies exist. The association with uptake is positive but the mechanism is unclear, so the
intervention should be specified and measured.

**12. Rubric for Exercise 6.** Full marks require numeric exit criteria per wave, a support model
that changes between waves, a named gate owner, and a stated action when a wave fails. Half marks
for a wave plan without gates.

## 14. Interview questions

**1. How do you measure adoption?**
Strong: states a numerator and denominator, distinguishes adoption from use rate, and names the
event log. Weak: says "active users" without a denominator.

**2. Why do pilots overstate adoption?**
Strong: volunteer users, extra support, and a duration shorter than the novelty effect. Weak: says
pilots are optimistic.

**3. What would you do if adoption fell after month six?**
Strong: diagnoses decay versus a step change, checks champion status and staff turnover, and
reinforces with a specific mechanism. Weak: says they would retrain.

**4. How does workflow fit affect adoption?**
Strong: discusses insertion point, interruption cost in seconds, and the parallel-workaround test.
Weak: says the product must be easy to use.

**5. How do you build trust in a model's recommendations?**
Strong: explanation, uncertainty, override, feedback, and measurement of appropriate reliance.
Weak: says explainability.

**6. What is the risk of explaining too much?**
Strong: cites the evidence that explanations can increase over-reliance, and proposes
uncertainty display and audit as counterweights. Weak: says more explanation is always better.

**7. How do you decide when to scale a deployment?**
Strong: a numeric gate on use rate at a defined date, derived from the economics. Weak: says when
users are happy.

**8. How does adoption affect the business model?**
Strong: connects use rate to renewal probability, revenue retention, and the multiple of
acquisition cost. Weak: says adoption drives revenue.

**9. What is the minimum viable adoption rate for your product?**
Strong: derives it from gross profit per site, acquisition cost, and a payback target. Weak: gives
a round number without derivation.

**10. How do you handle a champion who leaves?**
Strong: two champions, a named successor, institutional ownership, and a monitoring plan that
expects a decline. Weak: says they would recruit a new champion.

**11. What adoption data would you put in a board paper?**
Strong: monthly active sites, use at eligible events, renewal probability, and the gate status.
Weak: licensed users and satisfaction scores.

**12. Why is a high override rate ambiguous?**
Strong: because overrides can be correct, and the appropriate measure is the inappropriate
override rate, which requires clinical review. Weak: says overrides mean the alerts are bad.

## 15. Key terms

- **Activation** — point at which a user completes the core action.
- **Adoption** — sustained use of a product in the workflow.
- **Alert burden** — alerts per user per shift.
- **Alert fatigue** — sustained decline in appropriate response to alerts.
- **Algorithm aversion** — under-reliance on algorithmic output.
- **Automation bias** — over-reliance on automated output.
- **Calibrated trust** — reliance proportional to accuracy.
- **Champion** — clinician who advocates and models use.
- **Churn** — loss of a user or account.
- **Decay** — decline in use over time without reinforcement.
- **Dose** — intensity of use per user per period.
- **Dose-response** — relationship between use intensity and outcome.
- **Exposure** — share of eligible users with access.
- **Friction** — any step beyond the clinical task.
- **Interruption cost** — time and attention required at the moment of use.
- **Monthly active user** — user with at least one use in a month.
- **Net revenue retention** — revenue retained plus expansion.
- **Onboarding** — process of bringing a user to first value.
- **Override** — user rejects a recommendation.
- **Pilot effect** — inflated effect observed under pilot conditions.
- **Reinforcement** — recurring intervention that sustains use.
- **Renewal** — contract continuation.
- **Retention** — share of users still active after a period.
- **Stickiness** — ratio of daily to monthly active users.
- **Substitution** — retirement of the previous method.
- **Super-user** — high-frequency user who provides support.
- **Sustained use** — use rate that holds after the novelty period.
- **Time to first value** — interval from access to first successful use.
- **Trial** — first real use within a defined window.
- **Use rate** — share of eligible events where the product is used.
- **Workflow fit** — degree to which the product matches the real process.

## 16. Sources

All URLs accessed 26 October 2026.

1. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
   sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*.
   2021;181(8):1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
2. Felisberto M, et al. Override rate of drug-drug interaction alerts in clinical decision support
   systems: a brief systematic review and meta-analysis. *Health Informatics Journal*. 2024.
   https://doi.org/10.1177/14604582241263242
3. Bryant AD, Fletcher GS, Payne TH. Drug interaction alert override rates in the Meaningful Use
   era. *Applied Clinical Informatics*. 2014.
   https://www.thieme-connect.de/products/ejournals/pdf/10.4338/ACI-2013-12-RA-0103.pdf
4. Ray JM, et al. Alert fatigue measurement in clinical decision support: a systematic review of
   systematic reviews. *JAMIA*. 2026.
   https://www.metrohealth.org/globalassets/metrohealth-documents/population-health-research-institute/ray-wilson-et-al-2026-alert-fatigue-systematic-review.pdf
5. Ma S, et al. Ambient artificial intelligence scribes: utilization and impact on documentation
   time. *JAMIA*. 2024. https://doi.org/10.1093/jamia/ocae304
6. Haberle T, et al. Use of an ambient artificial intelligence tool to improve quality of clinical
   documentation. *JAMIA*. 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11452835/
7. Apathy NC, Holmgren AJ, et al. Physician EHR time and visit volume following adoption of
   team-based documentation support. *JAMA Internal Medicine*. 2024.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11348094/
8. Prinster B, et al. Care to explain? AI explanation types differentially impact chest radiograph
   diagnostic performance and physician trust in AI. *Radiology*. 2024.
   https://doi.org/10.1148/radiol.233261
9. Egala B, Liang TP. Algorithm aversion to mobile clinical decision support among clinicians: a
   choice-based conjoint analysis. *European Journal of Information Systems*. 2023.
   https://doi.org/10.1080/0960085x.2023.2251927
10. Miech EJ, et al. Inside help: an integrative review of champions in healthcare-related
    implementation. *SAGE Open Medicine*. 2018.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC5960847/
11. Santos WJ, et al. The effectiveness of champions in implementing innovations in health care: a
    systematic review. *Implementation Science Communications*. 2022.
    https://link.springer.com/article/10.1186/s43058-022-00315-0
12. Pettersen IJ, Eide H, Berg SH. The role of champions in the implementation of technology in
    healthcare services: a systematic mixed studies review. *BMC Health Services Research*. 2024.
    https://doi.org/10.1186/s12913-024-10867-7
13. Greenhalgh T, Wherton J, Papoutsi C, et al. Beyond adoption: the NASSS framework. *Journal of
    Medical Internet Research*. 2017;19(11):e367. https://doi.org/10.2196/jmir.8775
14. Office of the National Coordinator for Health Information Technology. *HTI-1 Final Rule*,
    2024. https://www.healthit.gov/topic/laws-regulation-and-policy/health-data-technology-and-interoperability-certification-program
15. US Food and Drug Administration. *Clinical Decision Support Software: Guidance for Industry
    and Food and Drug Administration Staff*. September 2022.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software

Figures in Section 6 are illustrative. The alert parameters are taken from the cited 2021
validation and applied as published proportions; they are not measurements from the composite
system.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_procurement.md`](05_procurement.md)
- Next topic: [`07_business-models.md`](07_business-models.md)
- Related topic: [`01_product-discovery.md`](01_product-discovery.md)
- Module survey: [`../../course/17_product_economics_leadership.md`](../../course/17_product_economics_leadership.md)
- Labs: [`../../labs/`](../../labs/)
