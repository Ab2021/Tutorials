---
case_id: CASE-01-08-002
case_title: Spotify personalization versus experimentation
case_version: 1.0-draft
created: 2026-11-07
last_reviewed: null
status: production
content_status: draft
case_type: predictive-ml
domain: Music recommendation and experimentation
primary_pattern: evaluation-mismatch
secondary_patterns: [label-delay, cost-latency-tradeoff]
evidence_level: published
source_basis: published and illustrative
as_of_date: 2026-11-07
owner_role: Curriculum maintainer
source_index: INDEX.md
review_trigger: Spotify corrects the source or materially changes stack ownership
---
# Spotify personalization versus experimentation

**Unreviewed draft.** The original case referenced by the assignment was unavailable and has not been inspected; this document does not reconstruct its contents. Published evidence and instructional design are deliberately separated. The numerical inputs, proposed operational controls, and architecture diagrams are illustrative, not Spotify disclosures.

## 1. The One-Line Framing
Given a personalization system and a comparable baseline, separate the low-latency ML system that chooses user experiences from the experimentation system that evaluates whole variants, so product teams can assess user outcomes without conflating selection with causal evidence.
The published decision is stack separation; a specific baseline and effect size are not disclosed.

## 2. Context & Background
Spotify describes audio recommendations, personalized search and a personalized Home screen.
A recommendation system chooses an experience conditional on available user characteristics and context.
An experiment instead assesses outcomes under alternative treatments assigned to eligible units.
A contextual bandit is a personalized policy: its choice depends on features.
A non-contextual experimental bandit changes allocation across arms based on observed average outcomes.
The article places contextual bandits in Spotify's ML stack, not in its experimentation tools.
Spotify states it does not use multi-armed bandits for experimentation.
Spotify describes the ML and experimentation platforms as distinct, tightly integrated systems.
It identifies Confidence as its experimentation platform and describes A/B tests and rollouts of entire personalization systems.
The article reports 58 teams and 520 experiments on mobile Home during the preceding year.
These are activity counts, not a causal lift, traffic count, success rate, or current inventory.
The source also reports 300+ teams and thousands of experiments across the organization.
Neither number reveals how many experiments were simultaneous, independent, successful, or powered.

## 3. The Business Question
**Illustrative stakeholder paraphrase:** How should teams build and assess personalized recommendation systems without making the evaluation platform responsible for real-time inference?
Decision owner: product and ML platform leadership jointly with experimentation and data science owners; roles are illustrative.
Decision: serve competing system variants with ML infrastructure and compare them through experimentation.
Baseline for a particular experiment could be the previous recommendation policy or the best static treatment; the article discusses both conceptually, not one named deployment.
The estimand must describe the eligible population, assigned systems, outcome, and time horizon.
The decision is not whether one individual item or contextual-bandit arm wins in isolation.
A system can improve its arm-selection reward without improving long-term user outcomes.
A meaningful policy comparison must also assess operational and product guardrails.

## 4. Assumptions & Constraints
Published constraints: recommendation inference benefits from fast computation, rich features, and low-latency access to real-time data.
Published concern: long-horizon retention and listening outcomes delay observation, limiting benefits of rapid traffic reallocation.
Published concern: optimizing one bandit metric can obscure competing objectives.
Illustrative assumptions: stable user-level assignment, informed eligibility, complete outcome collection, and sufficiently isolated policies.
These are requirements for the teaching example, not claims about Spotify's assignment mechanism.
Assume an approved way to handle identity changes and consent; neither mechanism is described in the article.
The unit of inference should normally match randomization or account for clustering.
Choose observation windows before outcomes are examined; avoid treating incomplete cohorts as complete.
Separate operational constraints from statistical assumptions: fast serving does not prove causal identification.
No public service-level objective, model hyperparameters, data schema, or experiment randomization algorithm is established by the source.

## 5. Data, Tools & Method
### 5a. Data Specification
The primary evidence is a public engineering article, not event-level Spotify data.
An illustrative experiment requires a pre-assignment eligibility record at user grain.
It requires assignment records keyed by experiment and randomization unit.
It requires policy and model version records for each serving decision.
It requires outcomes keyed to the same eligible unit and observation window.
It requires guardrail observations, missingness flags and versioned definitions.
Use event time for outcomes; separately record ingestion time and permitted late-arrival window.
A recommendation impression is not automatically a randomized unit.
An unexposed assigned user remains in an intention-to-treat analysis when the target is assignment effect.
Actual Spotify schemas, data retention, access permissions, and completeness are undisclosed.

### 5b. Tools & Stack
Published: ML platform for personalization, features, training and serving requirements.
Published: Confidence experimentation platform evaluates systems and integrates with ML and other products.
Published: integrations can pre-create flags and model resources or operate through external interfaces and APIs.
Unspecified: database, stream broker, framework, language, hardware, scheduler, flag algorithm and network topology.
Illustrative: an assignment service, decision log and analysis pipeline are logical roles, not identified Spotify products.

### 5c. Method Summary
Input: eligible units and a predeclared pair of system variants.
Transformation: assign units independently of their prospective outcomes, then serve the assigned policy.
Estimation: compare unit-level outcomes over a fixed window under the declared assignment rule.
Post-processing: inspect confidence intervals, sample-ratio mismatch, guardrails and data completeness.
Decision: ship, iterate, or stop according to an agreed decision rule and observed uncertainty.
Baseline: prior policy or static experience, depending on the business question.
Alternative: one combined platform; source argues it must expand into ML infrastructure or sacrifice contextualization.
Alternative: non-contextual adaptive experiment allocation; Spotify says it does not use it for experimentation.
Alternative: contextual-bandit policy in ML compared with another complete policy in an A/B test.
The article does not provide head-to-head platform cost or causal effectiveness measurements.

### 5d. Data provenance, quality, exclusions and leakage controls
Public source S1 is the only source for Spotify-specific assertions in this case.
No internal event tables, experiment logs, model weights, raw metrics or analysis notebook were available.
The illustrative design excludes users only by pre-assignment eligibility, not based on post-assignment behavior.
Feature snapshots must reflect information available at the time of a decision.
Labels must not leak future behavior into historical training examples.
Reconcile events across devices, duplicates and timezones before final analysis.
Treat deletion and consent requirements as deployment-specific obligations, not established Spotify implementation.
Do not infer missing data are random without investigating by assignment arm and cohort.

### 5e. Reproducibility and method-selection ledger
S1: Yu Zhao and Mårten Schultzberg, [Why We Use Separate Tech Stacks for Personalization and Experimentation](https://engineering.atspotify.com/2026/1/why-we-use-separate-tech-stacks-for-personalization-and-experimentation), Spotify Engineering, January 2026; accessed 2026-11-07.
S1 supports stack separation, contextual-bandit placement, the stated non-use of experimental MABs, the integration examples and the reported activity counts.
S1 does not disclose experimental effect sizes, traffic counts, model architecture, raw data or implementation internals.
The exact calendar interval behind “last year” is not explicitly specified; do not substitute a verified calendar-year dataset.
The article is corporate first-party reporting, not an independently audited impact study.
Recheck S1 if corrected, removed, or superseded; archive source date and access date when reviewing.
For a real run, retain experiment configuration, code revision, data snapshot, policy artifacts and analysis version.
Log assignment algorithm and salt under appropriate access control; a seed alone cannot reproduce changing data or nondeterministic execution.
No Spotify experiment can be independently reproduced from the article alone.

### Claim ledger
| ID | Claim | Evidence and scope | Limitation |
|---|---|---|---|
| CLM-01 | Personalization and experimentation use separate, integrated stacks | Published, S1 | No causal effect of separation is measured |
| CLM-02 | The article places contextual bandits in the ML stack | Published, S1 | No named model architecture is disclosed |
| CLM-03 | The source reports 58 teams and 520 mobile Home experiments | Published, S1 | The counts are activity, not independent successes or impact |
| CLM-04 | The invented retention example has a two-point difference | Illustrative, E1–E2 | Not a Spotify observation |
The original assignment case is unavailable and uninspected; S1 is a separate public source, not a reconstruction of that case.

### Critical assumptions register
| ID | Assumption or constraint | Check | Consequence and fallback | Owner / trigger |
|---|---|---|---|---|
| A-01 | Eligibility precedes policy exposure | Compare event timestamps and rule versions | Post-treatment eligibility invalidates the intended assignment contrast; restart enrollment after correction | Experiment owner / rule change |
| A-02 | Stable assignment across devices | Audit identity transitions and arm switches | Cross-arm contamination changes the delivered contrast; evaluate cluster or account-level design | Identity owner / identity migration |
| A-03 | Outcomes mature uniformly | Compare ingestion delays by arm and client | Differential censoring biases observed rates; defer analysis and investigate missingness | Data owner / late-data alert |
| A-04 | Shared policy learning does not create unmodeled interference | Trace training data by assignment | Spillovers change the estimand; isolate training or randomize at a suitable cluster | ML owner / online-learning change |
| C-01 | Serving must meet a locally specified deadline | Measure end-to-end tail latency at peak load | Slow personalization cannot be evaluated as if every request received it; use a tested fallback | Service owner / traffic change |
| C-02 | Data use must be authorized | Review consent and feature lineage with responsible privacy team | Unauthorized features cannot be used; remove affected data and follow applicable remediation process | Privacy owner / new feature |
All assumptions and controls are proposed for the teaching design; none is a verified Spotify internal control.

## 6. Analysis Walkthrough
The supported architectural finding is organizational separation with integration, not a quantified lift.
Step 1: distinguish the feature that adapts recommendations from the experiment that compares complete features.
Step 2: identify the low-latency feature and inference responsibilities that favor an ML platform.
Step 3: identify the randomization, metric definition and comparative evaluation responsibilities that favor experimentation tooling.
Step 4: define stable handoff between assigned variant and the served model version.
Step 5: protect the analysis from treating policy optimization reward as a randomized system effect.
Step 6: compare primary and guardrail metrics over a mature observation period.
Step 7: interpret published activity counts only as reported experimentation activity.

### Illustrative worked experiment: not Spotify results
Assume 10,000 eligible users per randomized arm, stable allocation, complete 28-day outcomes and no interference.
Suppose 4,000 control users and 4,200 treatment users return at least once in days 22–28.
Define the outcome as the proportion of assigned users returning in that fixed window.
The outcome is a binary user-level measure; numerator is returning assigned users and denominator is all assigned users in each arm.
This arithmetic ignores operational failure, missing data, clustering and multiple testing; those require separate checks.
```text
FORMULA [E1]: p_a = r_a / n_a; delta = p_t - p_c
WHERE: r_a returning users; n_a assigned users; a in {t,c}; p_a probability; delta percentage points as a proportion.
CONSTRAINT: one binary observation per assigned user; same mature window and eligibility in both arms.
NUMERICAL EXAMPLE: p_c=4000/10000=0.40; p_t=4200/10000=0.42; delta=0.02 = 2 percentage points.
INTERPRETATION: the sample difference is 2 percentage points, not proof of a lasting benefit.
```
Independent arithmetic check: 200 extra returns per 10,000 treatment users corresponds to 0.02.
The relative difference is 0.02/0.40 = 5%; it is distinct from a 2 percentage-point absolute difference.
The relative measure is undefined when the control rate is zero.
Under independent Bernoulli observations and large-sample normal approximation, uncertainty is estimated as follows.
```text
FORMULA [E2]: SE = sqrt(p_t*(1-p_t)/n_t + p_c*(1-p_c)/n_c)
WHERE: SE is the standard error of the difference in proportions; all other symbols are defined in E1.
CONSTRAINT: independent users, no interference, sufficient sample size and a fixed-horizon unadjusted comparison.
NUMERICAL EXAMPLE: SE=sqrt(0.42*0.58/10000+0.4*0.6/10000)=0.00695 approximately.
INTERPRETATION: sampling variation of the estimated difference is about 0.695 percentage points.
```
A conventional approximate 95% interval is 0.02 ± 1.96 × 0.00695, or [0.0064, 0.0336].
That is approximately [0.64, 3.36] percentage points under the stated assumptions.
Check: both sample success and failure counts are thousands; the normal approximation is not being used on rare counts.
A confidence interval is not a probability that this fixed parameter lies in the computed interval.
If treatment retention were 4,100 instead, the difference would be 1 percentage point and the approximate interval would include zero.
An overlapping interval or non-significant result does not establish equivalence; a prespecified noninferiority margin is required for that question.
If users share accounts or influence one another, independent-user uncertainty can be too small; cluster appropriately.
If the policy adapts to observations across both groups, shared training may violate the assumed no-interference comparison.

### Illustrative sample-ratio diagnostic
Suppose a nominally equal experiment assigns 10,400 users to control and 9,600 to treatment.
The expected count is 10,000 in each arm under the declared assignment configuration.
Pearson's statistic is (10400−10000)²/10000 + (9600−10000)²/10000 = 32.
For one degree of freedom, that is far beyond the conventional 3.84 threshold for a single 5% test.
This calculation assumes a fixed allocation, independent assignment units, and no legitimate ramp or stratification affecting expected counts.
Repeated automated monitoring needs an appropriate alerting policy rather than interpreting every check at a nominal 5% level.
An imbalance identifies a discrepancy with the expected allocation, not its cause and not necessarily the direction of bias.
Compare pre-assignment eligibility, assignment service output, durable assignment records, and downstream joins in that order.
If the mismatch appears only after the client renders content, investigate exposure or instrumentation rather than the randomizer alone.
Do not repair the discrepancy by dropping users from the larger arm after seeing the outcomes.
An unresolved assignment defect makes the planned causal interpretation unreliable even if the reported lift looks favorable.

### Illustrative delayed-data sensitivity
Suppose control has 4,000 observed returners among 10,000 assigned, but 200 users have incomplete outcome feeds.
Suppose treatment has 4,200 observed returners among 10,000 assigned, with 400 incomplete feeds.
If every incomplete user is actually a non-returner, the observed difference remains 2 percentage points.
If every missing control user returned and no missing treatment user returned, the difference becomes 0 percentage points.
If no missing control user returned and every missing treatment user returned, the difference becomes 6 percentage points.
These extreme bounds assume each missing feed corresponds to one distinct user whose return status is otherwise unknown.
They are not confidence intervals and do not address sampling variation.
The width shows why differential completeness can dominate an apparently precise sampling interval.
Recover events or use a justified missing-data method before making a product decision.

### Illustrative serving workload
Suppose peak arrival is 2,000 requests per second and each instance can sustain 250 requests per second under the actual request mix.
```text
FORMULA [W1]: N = ceil(lambda / (mu * u))
WHERE: N instances; lambda requests/s; mu measured sustainable requests/s/instance; u target utilization fraction.
CONSTRAINT: balanced load, homogeneous instances, u in (0,1], no correlated failure, and mu measured at the latency target.
NUMERICAL EXAMPLE: lambda=2000, mu=250, u=0.8; N=ceil(2000/(250*0.8))=10 instances.
INTERPRETATION: ten instances meet the illustrative utilization target before redundancy and deployment headroom.
```
Check: ten instances × 250 × 0.8 = 2,000 requests/s.
With one instance unavailable, nine × 250 × 0.8 = 1,800 requests/s; provision at least eleven for this particular one-instance failure assumption.
At 3,000 requests/s, the same assumptions require 15 instances before redundancy.
Throughput arithmetic does not prove p99 latency, feature-store capacity, memory adequacy, or failover correctness.
These workload numbers are invented teaching assumptions, not Spotify infrastructure facts.

### Extended decision laboratories
The following independent exercises use constructed inputs and proposed procedures.
They are not records of Spotify experiments, deployments, incidents, or measurements.
Their purpose is to connect the policy-level decision to calculations and operational choices.

#### Laboratory 1: Assignment versus delivery
Assume 1,000 users are assigned to each of two policies.
In control, 900 users receive the assigned policy and 450 of all assigned users return.
In treatment, 700 users receive the assigned policy and 480 of all assigned users return.
The intention-to-treat rates are 450/1,000 = 45% and 480/1,000 = 48%.
The assignment effect estimate is therefore 3 percentage points.
The policy delivery rates are 90% and 70%, a 20-point gap.
Delivery is a post-assignment event and may depend on user or device characteristics.
Dividing 480 by 700 would mix all returners with a delivered-only denominator.
Even if the numerator were restricted to delivered users, the groups would not necessarily be comparable.
A treatment delivery failure is part of the deployed assignment experience under this estimand.
Diagnose whether the gap is caused by capacity, routing, compatibility, or event loss.
A corrected candidate policy might have a different effect, which this assignment contrast does not identify.
Report the delivery breakdown alongside the assignment effect rather than replacing the primary denominator.
If failures violate an agreed reliability guardrail, stop or roll back according to the launch contract.
A later experiment can assess the corrected system without rewriting the historical assignment.

#### Laboratory 2: Event-time maturity
Suppose enrollment occurs over seven consecutive days.
Each user is evaluated for a full 28 days after their own assignment.
Outcome events may arrive up to two days after they occur.
The first assigned user becomes analyzable after 30 days under that lateness rule.
The last user becomes analyzable six days later than the first if enrollment day one is the origin.
A dashboard generated on day 30 therefore cannot compare all assigned users on a complete 28-day outcome.
A control and treatment imbalance in enrollment dates can create apparent differences even with equal true behavior.
The analysis table should contain assignment time, window end, data watermark, and maturity status.
Do not classify an immature user as a non-returner merely because no event has arrived yet.
A fixed calendar window is a different estimand from an individual 28-day window.
Late events beyond the cutoff can be audited in a frozen backfill without silently changing the published metric.
If the ingestion delay changes after launch, estimate its distribution separately by source and arm.
The chosen watermark trades decision speed against the risk of incomplete outcomes.
Near-real-time operational checks may use provisional data, explicitly distinguished from final causal analysis.
If a safety metric demands immediate action, its stop rule does not require waiting for the primary outcome to mature.

#### Laboratory 3: Stratified assignment and aggregation
Suppose 800 established and 200 new users are allocated to each arm.
Among established users, 400 control and 416 treatment users return.
Among new users, 40 control and 44 treatment users return.
The established-user rates are 50% and 52%, an increase of 2 percentage points.
The new-user rates are 20% and 22%, also an increase of 2 percentage points.
Overall rates are 440/1,000 = 44% and 460/1,000 = 46%.
The overall difference is 2 percentage points because the same population weights are used in both arms.
The established cohort contributes 0.8 × 2 = 1.6 percentage points to the weighted difference.
The new cohort contributes 0.2 × 2 = 0.4 percentage points.
Stratum membership must be fixed before treatment if it is used for a causal subgroup contrast.
If allocation probabilities differ by stratum, an unweighted arm average need not represent the target population.
Use prespecified target weights and an estimator consistent with the actual stratified randomization.
A small cohort has less precision even if its point estimate equals the larger cohort's estimate.
An apparent difference between subgroup effects needs an interaction contrast, not one significant and one nonsignificant test.
The source provides no such cohort-level results; this is a numerical teaching exercise.

#### Laboratory 4: Aggregation reversal
Suppose control contains 900 established users and 100 new users.
Their control return counts are 450 and 20, giving rates of 50% and 20%.
Suppose treatment contains 100 established users and 900 new users.
Their treatment return counts are 55 and 225, giving rates of 55% and 25%.
Treatment is better by 5 percentage points within each cohort.
Nevertheless, the unadjusted control rate is 470/1,000 = 47%.
The unadjusted treatment rate is 280/1,000 = 28%.
The aggregate difference is minus 19 percentage points because the cohort mixes differ sharply.
With a common 50/50 target mix, control is 35% and treatment is 40%.
The standardized contrast is therefore plus 5 percentage points for that specified target population.
In a properly randomized experiment, such a large compositional difference calls for an assignment and eligibility investigation.
Standardization cannot repair arbitrary post-treatment subgroup definitions or hidden confounding.
The correct target weights depend on the business population, not on whichever mix produces a preferred result.
Preserve both raw and standardized figures, the weights, and the reason for using them.
Do not claim this constructed reversal occurred in the source's organization.

#### Laboratory 5: Minimum detectable difference
Consider a hypothetical equal-sized two-arm comparison with control rate approximately 40%.
For planning, approximate both arm variances by 0.4 × 0.6 = 0.24.
Use a two-sided 5% significance threshold and 80% power, with normal quantiles approximately 1.96 and 0.84.
For a desired 2-percentage-point difference, the approximation is n per arm = 2 × 0.24 × (1.96 + 0.84)² / 0.02².
The numerator is 0.48 × 7.84 = 3.7632.
Dividing by 0.0004 gives 9,408 users per arm before additional design adjustments.
This is a planning approximation, not a guarantee of power under every true treatment rate.
If the desired detectable difference halves to one point, the squared denominator falls by a factor of four.
The same approximation then gives 37,632 users per arm.
The required sample may grow under clustering, attrition, uneven allocation, or multiplicity adjustment.
If only half the eligible population can be enrolled each week, calendar duration follows from the eligible flow, not total registered accounts.
Mature observation time must be added after the final assignment.
A low-powered study can still show a large observed effect by chance.
A nonsignificant result does not demonstrate that a clinically or commercially relevant effect is absent.
Choose the detectable difference from a decision threshold and resource tradeoff rather than from the observed result.

#### Laboratory 6: Clustered exposure
Suppose the randomization unit is a household rather than an individual listener.
Assume ten listeners per household on average and an intracluster correlation of 0.05 for a simplified planning exercise.
The equal-cluster-size design effect is 1 + (10 − 1) × 0.05 = 1.45.
An individual-level calculation that calls for 9,408 listeners per arm would require approximately 13,642 listeners per arm after multiplying by 1.45.
At ten listeners per household, round upward to at least 1,365 households per arm under this simplified approximation.
The effective sample size of 14,500 listeners would be approximately 14,500/1.45 = 10,000.
This approximation assumes similar cluster sizes and the stated correlation.
Highly unequal household sizes can require a more careful design calculation.
Assignment must remain stable for the entire household if the household is the randomized unit.
Inference should reflect the randomization unit rather than treating every listening event as independently randomized.
Shared devices may still create identity errors even when households are randomized.
If only a few clusters are randomized, ordinary large-sample cluster approximations can be unreliable.
A design-based or other small-cluster method should match the actual allocation procedure.
The household outcome and the individual-weighted outcome answer different business questions.
Neither the cluster size nor the correlation in this exercise is a company measurement.

#### Laboratory 7: Relative versus absolute effects
Assume control retention is 4% and treatment retention is 5% in one population.
The absolute change is 1 percentage point.
The relative change is (5% − 4%)/4% = 25%.
Assume another population moves from 40% to 41%.
Its absolute change is also 1 percentage point, but its relative change is 2.5%.
With 10,000 assigned users per arm, each one-point improvement corresponds to 100 additional observed returners per 10,000.
The same absolute change may have different commercial value across populations if downstream value differs.
A relative metric can look dramatic when the baseline is very small.
A relative metric is undefined at a zero baseline.
The ratio of two noisy estimates also needs suitable uncertainty analysis.
Present the baseline, both arm rates, absolute effect, relative effect, and interval together.
Do not extrapolate a sample effect to all users unless the target population and deployment assumptions support that projection.
A percentage-point effect is not the percentage of users whose individual behavior was caused to change.
Some individuals may be helped while others are harmed even when the average effect is positive.
The examples are arithmetic illustrations rather than observed retention changes.

#### Laboratory 8: Missing-outcome bounds
Assume control has 10,000 users, with 4,000 confirmed returns and 100 unknown outcomes.
Assume treatment has 10,000 users, with 4,100 confirmed returns and 300 unknown outcomes.
A lower bound for control retention is 40%; its upper bound is 41%.
A lower bound for treatment retention is 41%; its upper bound is 44%.
The smallest possible treatment-minus-control difference is 41% − 41% = 0 points.
The largest possible difference is 44% − 40% = 4 points.
The simple observed difference treating unknowns as non-returners is 1 point.
The bounds require that the confirmed returns and unknown outcomes are mutually exclusive categories.
They are worst-case identification bounds, not a 95% confidence interval.
A narrower analysis requires additional defensible assumptions about why data are missing.
Compare missingness by assignment, device version, geography, and data pipeline before choosing such assumptions.
If missingness is caused by treatment-dependent client crashes, assuming missing completely at random is especially weak.
A successful backfill should be reconciled against independent event totals and duplicate rules.
If the possible effect spans the decision threshold, defer the decision or redesign measurement.
Do not interpret a missing return record as verified non-return without a complete source feed.

#### Laboratory 9: Sample-ratio mismatch under a ramp
Imagine the first 4,000 users enter a 50/50 allocation.
That stage expects 2,000 users per arm.
The next 6,000 users enter a 90/10 control-to-treatment allocation.
That stage expects 5,400 control and 600 treatment users.
The correct total expectation is therefore 7,400 control and 2,600 treatment.
Comparing the final counts with a 5,000/5,000 expectation would falsely suggest a severe mismatch.
Expected counts must reflect the allocation configuration in force at each assignment time.
A ramp must also specify whether previously assigned users remain sticky when the allocation changes.
Rehashing all users on a ramp change may switch treatment and alter the policy contrast.
For a randomized ramp, the inference procedure must account for differing inclusion probabilities and calendar time when relevant.
If rollout policy changes in response to early outcomes, the stopping and analysis plan must account for that adaptation.
A versioned configuration log supports reconstruction of the appropriate expected distribution.
Inspect allocation within each ramp stage before using a single aggregate discrepancy statistic.
If stage counts match but one client deviates, focus on client-specific eligibility or identity propagation.
The allocation percentages here are invented and are not a disclosed Spotify configuration.

#### Laboratory 10: Guardrails and utility
Suppose a candidate increases seven-day listening by 4% relative to control.
Suppose the prespecified discovery guardrail falls by 3 percentage points.
The primary metric and guardrail have different units and cannot be subtracted directly.
A launch decision requires a predeclared rule explaining whether the discovery drop is acceptable.
One possible rule is to reject a candidate if the discovery decrease exceeds a specified tolerance.
Another is to defer pending a product review of long-term user value and measurement validity.
A higher listening value can arise from repeatedly presenting already familiar content.
A lower discovery value could instead reflect a broken new-artist event definition.
Investigate the metric pipeline before interpreting the guardrail as a behavioral tradeoff.
Do not redefine the guardrail after observing the treatment result merely to make the launch pass.
Multiple guardrails increase the chance of chance findings if all are interpreted as standalone discoveries.
Distinguish a safety stop rule from a claim of statistically established harm.
Report uncertainty for both the primary metric and the guardrail when sample variation matters.
Consider whether creator exposure concentration moves even if average user discovery is stable.
The values and decision rules are illustrative, not attributed to Spotify.

#### Laboratory 11: Latency budget decomposition
Assume an illustrative request deadline of 150 milliseconds measured at the serving boundary.
Reserve 15 milliseconds for request parsing and eligibility checks.
Reserve 20 milliseconds for experiment assignment lookup.
Reserve 35 milliseconds for feature retrieval and 45 milliseconds for policy inference.
Reserve 15 milliseconds for catalog filtering and response construction.
The named components sum to 130 milliseconds, leaving 20 milliseconds of explicit slack.
This sum is a planning budget, not a mathematical identity for the end-to-end p99.
Percentiles generally do not add to produce the percentile of a sum, especially when components are correlated.
Measure the full request distribution and its dependency spans under realistic load.
A cache hit may have a different latency and freshness profile from a miss.
Timeouts should reserve enough time for the fallback path rather than consuming the entire deadline.
A retry can increase load and tail latency during an already degraded dependency incident.
Track timeout and fallback rates by assigned arm to identify differential delivery.
A low average latency can hide a severe p99 problem for a minority of users.
The numerical deadline and component budgets are invented for this exercise.

#### Laboratory 12: Queueing and headroom
Assume one serving instance sustains 250 requests per second at a measured acceptable workload.
At a target utilization of 80%, the planning capacity per instance is 200 requests per second.
For 2,000 requests per second, the capacity calculation gives 10 instances.
To withstand one instance failure at the same target utilization, use at least 11 instances.
With 11 instances and one failure, 10 remain and their target capacity is exactly 2,000 requests per second.
At 2,200 requests per second, the same one-failure condition requires 12 instances.
This arithmetic assumes homogeneous capacity and traffic distributed evenly across healthy instances.
It does not establish that an autoscaler can add instances before a sudden burst arrives.
It also does not account for shared feature-store or network bottlenecks.
A fallback service can become the limiting dependency when many candidate requests fail together.
Load tests should include the real request mix, cache behavior, feature sizes, and concurrency distribution.
A rollout that doubles model memory can reduce per-node capacity even if per-request compute stays unchanged.
Deployment surge capacity and fault-zone loss may require more than one spare instance.
Cost estimates should price provisioned capacity, not only the number of successfully served requests.
The workload is invented; no Spotify capacity conclusion follows from it.

#### Laboratory 13: Reward delay in an adaptive policy
Suppose a policy chooses between familiar and exploratory recommendations.
An immediate reward counts whether a track is played for a short threshold duration.
A long-term objective measures whether the user returns during a later seven-day window.
The immediate event can arrive within seconds while the return outcome takes weeks to mature.
An adaptive allocator driven only by the immediate reward can favor familiar content before discovery benefits appear.
Updating on incomplete long-term outcomes creates a different bias if newer users have had less opportunity to return.
One option is to freeze the policy during a fixed evaluation window.
Another is to evaluate a policy that learns online, including its learning rule as part of the treatment.
Separate training streams can avoid a treatment arm changing the control policy through shared feedback.
Shared training can instead be part of the intended deployment, but then interference must be explicitly addressed.
A short-term surrogate needs empirical validation against the long-term business objective.
Its correlation with the long-term metric alone does not prove that optimizing it causes long-term improvement.
Investigate policy-induced distribution shifts before carrying a surrogate relationship across variants.
If delayed reward is the main constraint, faster traffic allocation does not automatically mean faster valid decisions.
The source's qualitative concern is published; this particular policy and reward example is illustrative.

#### Laboratory 14: Propensity support in offline evaluation
Suppose a logging policy recommends track A with probability 0.8 and track B with probability 0.2 in a context.
A candidate policy recommends A with probability 0.4 and B with probability 0.6 in the same context.
The importance weight for an observed A action is 0.4/0.8 = 0.5.
The importance weight for an observed B action is 0.6/0.2 = 3.
An inverse-propensity weighted contribution multiplies the observed reward by the corresponding weight.
Suppose one A observation has reward 1 and one B observation has reward 0.
Their contributions are 0.5 and 0, with a simple mean of 0.25 over those two logged observations.
This tiny sample is only an arithmetic demonstration and is not a reliable policy-value estimate.
The logged probabilities must correspond to the actual action-selection process at decision time.
If the logging policy assigns B zero probability, the candidate's B value is not identified by these weights.
Very small propensities produce large weights and potentially severe variance.
Weight clipping may reduce variance but introduces bias and changes the estimator's properties.
A logged slate, position effects, and interactions between items can complicate the action definition.
Offline estimates can screen candidates, but an online randomized system comparison addresses a different source of uncertainty.
No Spotify logging propensities or replay result are asserted here.

#### Laboratory 15: Metric joins and cardinality
Assume a user has one assignment record, three recommendation requests, and four listening events.
A raw join of requests to listening events on user ID alone produces twelve rows.
Counting those rows as listening events would triple the user's true event count.
Counting them as requests would quadruple the true request count.
Aggregate each event family to the appropriate user-window grain before joining for a user-level metric.
Alternatively, use a carefully specified temporal attribution rule when the question requires request-level linkage.
A left join from the assignment table retains assigned users with no recorded outcomes.
A missing outcome row may mean no behavior, a delayed event, or a failed join; distinguish those cases.
Deduplicate events using a documented event identity and tie-breaking rule.
A many-to-many identity mapping can silently expand both counts and weights.
Inspect key uniqueness and row counts at every transformation boundary.
Use the same event-time window and time-zone convention in both arms.
A data-quality check should include synthetic records with known expected counts where permitted.
Recomputing a metric after a join fix requires versioning the analysis artifact and explaining the change.
These data contracts are proposed controls rather than disclosures about Spotify's schema.

#### Laboratory 16: Cost and experiment duration
Assume a candidate needs 12 additional serving instances for 30 days during an experiment.
Assume an illustrative all-in compute price of 0.40 currency units per instance-hour.
The instance-hours are 12 × 30 × 24 = 8,640.
The incremental compute estimate is 8,640 × 0.40 = 3,456 currency units.
Assume an additional 1,000 currency units for data processing and 500 for operational review.
The simple subtotal is 3,456 + 1,000 + 500 = 4,956 currency units.
The currency unit and prices are hypothetical and are not an external price quotation.
This subtotal excludes storage, network egress, training, licensing, failed work, and opportunity cost unless added separately.
If the baseline would have required six of the twelve instances anyway, only the incremental six belong in an incremental comparison.
At the same unit price, six instances for 30 days cost 1,728 currency units.
A longer experiment increases compute expense but may reduce uncertainty about the decision.
An early stop may save infrastructure cost while creating invalid inference if its rule was not planned.
Capacity reserved for failover may be charged even when no failure occurs.
A credible business case should compare expected benefit with the full incremental cost and uncertainty.
No source in this case supplies an actual Spotify infrastructure price or experiment budget.

#### Laboratory 17: Catalog and availability
A music catalog changes as items are added, removed, or become unavailable in a region.
A historical evaluation must use the catalog that was eligible at each historical decision time.
Using today's availability in a past replay can exclude items that were valid then or include items that did not yet exist.
A policy artifact and a catalog snapshot together determine which candidates can actually be served.
Licensing and territorial availability must be enforced by the authoritative current eligibility mechanism.
A feature indicating an artist's future popularity would leak information into historical training examples.
A candidate policy might appear more diverse merely because its eligible catalog is larger.
Compare policies under the same eligibility rules unless catalog access itself is the intended treatment.
If an availability feed fails, the serving contract should specify safe behavior rather than assuming every item remains playable.
Record the catalog version or equivalent audit reference at decision time where feasible.
A replay that lacks historical availability should state that limitation and avoid pretending to reproduce the original choice set.
A recommendation that is ranked highly but cannot be played can reduce trust and inflate impression counts without listening value.
Monitor eligibility filtering and empty-result rates separately from model scoring latency.
Regional differences require enough traffic and a prespecified interpretation before making subgroup claims.
These are general design considerations, not descriptions of Spotify's proprietary rights infrastructure.

#### Laboratory 18: Cold-start comparison
Suppose 2,000 new users are randomized equally to a contextual candidate and a popularity baseline.
Among assigned control users, 300 return; among assigned treatment users, 330 return.
The observed rates are 30% and 33%, yielding a 3-percentage-point difference.
If half the candidate users receive fallback because their features are unavailable, the estimate includes that fallback experience.
Feature availability should be measured before the request or at decision time with clear versioning.
A missing feature can mean a genuinely new user, a delayed pipeline, or a client instrumentation error.
A default vector should not be confused with a real observation that happens to equal the default.
The baseline may be competitive because sparse history limits the candidate's personalization signal.
The candidate may perform better after the user accumulates interactions, making observation window choice important.
A subgroup defined by behavior after assignment can introduce selection bias.
A tenure group defined from pre-assignment account age is more suitable for a prespecified subgroup contrast.
If treatment changes the rate at which users generate features, analyzing only feature-rich users is especially misleading.
Inspect guardrails for users with little history even if the overall experiment is positive.
Avoid claiming the small sample difference is conclusive without a suitable interval and decision rule.
The counts and policies are invented, not Spotify performance results.

#### Laboratory 19: Interference from shared training
Assume a treatment policy recommends more emerging artists than control.
Treatment interactions enter a shared model training dataset used by both policies the next day.
The control model can then learn from treatment-driven interactions.
Control outcomes are no longer independent of the treatment assignments of other users.
A user-level assignment contrast may still describe the mixed deployed system under that allocation.
It may not predict the outcome of moving the entire population to treatment.
One design isolates feedback and model artifacts by arm during the experiment.
This isolation increases infrastructure cost and may create models unlike the eventual shared production model.
Another design randomizes larger clusters and treats within-cluster spillovers as part of the intervention.
Cluster randomization reduces the number of independent units and may increase uncertainty.
A switchback design can compare policies over time but requires attention to carryover and time trends.
Specify whether user recommendations, training data, and model parameters are shared before choosing the design.
The experiment contract should include update cadence, data cutoff, and artifact lineage.
A policy that changes during the test should be described as an algorithm plus learning procedure, not a frozen model.
No shared-training arrangement is asserted for the published source.

#### Laboratory 20: Rollback and causal interpretation
Assume an experiment begins with a candidate artifact and later rolls back due to a latency incident.
The experiment assignment remains stable, but the served treatment changes after the rollback.
An intention-to-treat estimate over the entire period describes assignment to that operational rollout strategy.
It does not isolate the effect of continuously receiving the original candidate.
The incident can affect user behavior through waiting time and missing recommendations as well as model quality.
A time-filtered analysis that removes the outage may condition on an event influenced by treatment.
Before using a restricted period, specify what causal question that period answers and whether selection is defensible.
Preserve deployment timestamps, regional propagation times, and actual artifact identifiers.
A control-plane rollback request is not proof that every serving replica has switched.
Verify data-plane artifact versions and fallback traffic after the change.
If a cached response can outlive the rollback, its time-to-live affects the period of mixed exposure.
Communicate whether the incident requires stopping enrollment, stopping treatment delivery, or both.
After recovery, decide whether a fresh randomized experiment is needed for the revised artifact.
Retain the original result and incident context rather than silently replacing it with a clean-window estimate.
This scenario is hypothetical and does not describe an incident reported in S1.

#### Laboratory 21: Practical significance and decision loss
Assume a launch costs 20,000 currency units per month in incremental infrastructure and support.
Assume each additional retained user produces an estimated 5 currency units per month in contribution.
A break-even calculation requires 20,000/5 = 4,000 additional retained users per month.
For 1,000,000 eligible users, that is 4,000/1,000,000 = 0.4 percentage points.
This simplified threshold assumes the retention gain and contribution value apply to the same population and month.
It also assumes a linear contribution per additional retained user and excludes uncertainty in that value.
An estimated effect of 0.2 points would imply 2,000 extra users and 10,000 currency units before other effects.
An estimated effect of 0.6 points would imply 6,000 extra users and 30,000 currency units before other effects.
A confidence interval spanning both values leaves the decision economically uncertain even when the point estimate is positive.
A user-safety or rights constraint can rule out launch regardless of estimated financial benefit.
The analysis must distinguish incremental retained users from all observed users who return under treatment.
Do not multiply a relative lift by total population without first applying the baseline rate.
If contribution differs by cohort, use a prespecified population-weighted estimate rather than a convenient average.
A finite experiment may not determine long-run contribution because retention and costs can change over time.
The economic inputs are invented; no currency or price source is claimed.

#### Laboratory 22: Inference under repeated looks
Suppose a fixed-horizon test has a nominal two-sided false-positive rate of 5% under its assumptions.
That guarantee applies to the prespecified decision at the planned analysis time.
Looking daily and shipping on the first favorable ordinary p-value changes the decision procedure.
The individual p-value threshold does not by itself control the probability of ever crossing that threshold.
The exact inflation depends on the dependence between looks and the stopping rule.
A valid sequential design can specify boundaries or a compatible always-valid inference procedure in advance.
The analysis must follow the actual randomization and timing assumptions of the chosen method.
Operational monitoring may still look continuously for outages and safety failures.
A safety stop should be recorded separately from a claim that a primary product effect is statistically established.
If unplanned efficacy peeking occurred, disclose it and avoid presenting the final ordinary interval as if the original fixed-horizon plan were followed.
A fresh independent confirmation can provide evidence after an exploratory early signal.
Adjusting the analysis after observing the most favorable look does not recreate a prespecified design automatically.
The cost of a false launch and a missed improvement should guide the monitoring plan.
Repeated monitoring can also affect metric selection if only the strongest outcome is highlighted.
These statements describe general experimental-design choices, not an observed Spotify monitoring procedure.

#### Laboratory 23: Instrumentation reconciliation
Suppose an assignment ledger records 5,000 treatment users for a completed enrollment period.
The serving ledger records at least one request for 4,000 of those users.
The client exposure ledger records at least one visible recommendation for 3,600 users.
The return-event table records 2,000 users with a later return during the defined window.
These counts represent different stages and are not expected to be equal.
The request-to-assignment fraction is 4,000/5,000 = 80%.
The visible-exposure-to-requester fraction is 3,600/4,000 = 90%.
The assigned-user return rate is 2,000/5,000 = 40% if the outcome feed is complete.
The 1,000 assigned users without requests must not automatically be labeled as assignment failures.
They may simply never open the relevant surface during the observation period.
The 400 requesters without visible exposure require investigation of response, rendering, and logging boundaries.
A comparison with control should use the same definitions and maturity cutoff.
If client event loss is differential by treatment, observed exposure differences can be instrumentation artifacts.
If actual rendering fails more in treatment, that operational defect is part of the treatment experience.
Join checks should ensure every stage count uses distinct users rather than raw event rows.
The counts are fabricated to illustrate reconciliation, not a measured production funnel.

## 7. The Finding
Spotify reports that personalization and experimentation serve complementary but different functions: ML builds and serves personalized systems, while Confidence supports evaluation of whole system variants. Contextual bandits belong to the former in Spotify's architecture; non-contextual experimental bandits are not used there according to the article. The scale figures document activity, not causal impact. Appropriate local implementations still need measured latency, trustworthy assignment, mature outcomes and independent review.

## 8. System/Solution Design
The following logical architecture is **illustrative** unless explicitly identified as published.
```text
Eligible user -> assignment [experiment domain] -> variant identifier
Variant identifier + request context -> policy serving [ML domain] -> recommendation
Recommendation + assignment + version -> decision and exposure records
User outcomes -> quality and maturity checks -> experiment analysis -> rollout decision
Historical permitted events -> feature preparation -> model training -> versioned policy artifact
```
The boxes describe logical responsibilities, not Spotify's documented service topology.

### 8a. Offline / Training / Batch Path
An illustrative ingestion job validates event identity, timestamp, permission and schema.
Training datasets are cut at a historical timestamp to avoid future-information leakage.
A candidate policy is trained and tested against a previous policy on time-separated data.
Offline improvements are screening signals, not a causal substitute for online evaluation.
Artifact identifiers bind feature definitions, model weights, policy configuration and evaluation run.
A release gate checks reproducibility, feature availability, operational compatibility and safety.
The deployment controller makes an approved artifact available for an experiment.
Published source does not identify a model registry, training schedule or feature-store implementation.

### 8b. Online / Serving / Real-Time Path
Illustrative successful trace: eligible user U receives stable assignment A in experiment X.
A request carries A to the personalization serving boundary.
The serving component resolves A to a permitted version, obtains authorized context and returns recommendations.
A decision record identifies the experiment, assignment, artifact version and request outcome.
User outcomes later join to the assigned unit after the observation window matures.
Illustrative failed trace: the assigned candidate model is unavailable at serving time.
The service applies a preapproved safe fallback and records both assigned and actually served versions.
The intention-to-treat analysis keeps U in A; a separate diagnostic reports noncompliance.
If assignment is unavailable, a preapproved safe policy serves and the request is marked not reliably assigned.
Do not silently classify fallback requests as if the candidate was delivered.
Timeouts, retry limits and freshness thresholds must be chosen from measured requirements; the source provides no values.
A cold-start request may use a permitted non-personalized baseline; the actual Spotify strategy is not disclosed.

### 8c. Key Design Decisions
| Choice | Alternative | Evidence and tradeoff | Revisit trigger |
|---|---|---|---|
| Separate ML and evaluation responsibilities | Combined platform | S1 supports separation; added integration boundary | Integration cost or requirements change |
| Compare whole policies | Compare selected item arms | S1 motivates policy-level evaluation | Question genuinely concerns a single arm |
| Fixed A/B methodology | Experimental non-contextual MAB | S1 describes multi-objective and delay concerns | Organization's goals and methods differ |
| Rich real-time features | Static context | S1 describes ML requirements; latency and data risk increase | Feature value or freshness changes |
| Multiple guardrails | Single reward only | S1 warns of short-term listening versus discovery tradeoff | Guardrail definitions cease to reflect user value |

### 8d. Architecture artifacts and operational readiness
An experiment contract should specify eligibility, unit, variants, ramp, owner, duration and stop conditions.
A serving contract should specify policy mapping, request fields, deadlines and fallback semantics.
A logging contract should distinguish assignment, decision, exposure and outcome.
A data contract should identify event grain, time basis, version, access and retention.
An incident runbook should identify the rollback owner, triggers, telemetry and communication path.
A release record should identify artifact versions and observed deployment state.
These are proposed controls, not claims about Spotify's internal process.

## 9. Evaluation & Validation
**Executed in source:** Spotify reports using experiments and rollouts for personalization systems; detailed individual experiments and results are not supplied.
**Illustrative proposed protocol:** randomize eligible users to current and candidate policies before treatment.
Specify a primary outcome, denominator, event window, user population and comparator before analysis.
Assess short-term listening, long-term retention and discovery as separately defined measures.
Declare which outcomes drive the decision and which act as guardrails.
Wait for each cohort to complete the intended outcome window and permitted event delay.
Check expected versus actual allocation at assignment, not only among exposed users.
Investigate device, region, client version and cohort imbalances without making unplanned subgroups definitive.
Use predeclared sequential inference if repeated looks affect decisions; an ordinary fixed-horizon interval does not correct optional stopping.
Account for multiplicity if making claims across many metrics or slices.
Assess interference from shared model updates, creator supply, social effects and concurrent experiments.
If shared training creates treatment spillovers, isolate updates or design an experiment whose estimand includes that spillover.
An A/A exercise can expose instrumentation and allocation defects but cannot prove all future experiments unbiased.
No numeric Spotify lift, guardrail threshold, p-value or power calculation is available in S1.

## 10. Action Taken
Published: Spotify says it maintains ML and experimentation stacks, integrates them, and treats personalization systems as variants in A/B tests and rollouts.
Published: Spotify says contextual bandits are in the ML stack and experimental multi-armed bandits are not used.
The article describes API and user-interface integration patterns, but not specific infrastructure implementation details.
The proposed contracts, fallback behavior and evaluation exercise in this case are instructional recommendations only.
No new deployment or operational intervention was performed for this case.

## 11. Outcome & Results
Published organizational outcome: the article describes the ability for recommendation teams to innovate and evaluate in parallel.
Published activity: 58 teams ran 520 mobile-home experiments in the preceding year, per Spotify's authors.
The activity counts do not establish a causal effect, per-team productivity, unique users, launch rate or experiment independence.
The article reports no numerical lift attributable to separating stacks.
Illustrative outcome: the worked example yields a two-percentage-point observed difference under invented inputs.
No real outcome for that worked example exists, and its calculation is not evidence about Spotify.

## 12. Failure Modes & What Went Wrong
No actual incident is documented in S1; the following are illustrative diagnostic scenarios.

### Diagnostic 01: Assignment mismatch — illustrative
- Symptom: unequal arm counts at enrollment.
- Competing explanations: hash bug or eligibility change.
- First discriminating check: compare assignment counts by client and configuration.
- Immediate containment: pause enrollment and preserve assignments.
- Recovery criterion: correct allocation without relabeling historical users.
### Diagnostic 02: Exposure mismatch — illustrative
- Symptom: treatment has fewer impressions.
- Competing explanations: client rendering failure or instrumentation loss.
- First discriminating check: compare server decisions with client impressions by version.
- Immediate containment: restore rendering or logging.
- Recovery criterion: retain assigned users in intention-to-treat.
### Diagnostic 03: Identity churn — illustrative
- Symptom: one person appears in both arms.
- Competing explanations: unstable identifiers or shared devices.
- First discriminating check: trace identity transitions without exposing raw identifiers.
- Immediate containment: freeze an approved identity mapping.
- Recovery criterion: report contamination and cluster when justified.
### Diagnostic 04: Feature staleness — illustrative
- Symptom: relevance falls in recent sessions.
- Competing explanations: delayed ingestion or a weak model.
- First discriminating check: compare feature event times and fallback rates.
- Immediate containment: restore feature freshness or fall back safely.
- Recovery criterion: reassess outcomes after recovery.
### Diagnostic 05: Future leakage — illustrative
- Symptom: offline score jumps unrealistically.
- Competing explanations: future events or accidental target joins.
- First discriminating check: audit event-time cutoffs and join keys.
- Immediate containment: rebuild snapshots at decision time.
- Recovery criterion: rerun offline evaluation before release.
### Diagnostic 06: Training-serving skew — illustrative
- Symptom: offline performance does not transfer.
- Competing explanations: different transforms or missing online values.
- First discriminating check: compare paired feature vectors for sampled requests.
- Immediate containment: unify or version transformations.
- Recovery criterion: monitor parity and online outcomes.
### Diagnostic 07: Model version drift — illustrative
- Symptom: same assignment serves different artifacts.
- Competing explanations: deployment drift or intentional rollout.
- First discriminating check: join decision logs to release manifests.
- Immediate containment: restore intended variant mapping.
- Recovery criterion: analyze period-specific exposure without post-selection.
### Diagnostic 08: Unrecorded fallback — illustrative
- Symptom: candidate appears identical to control.
- Competing explanations: model failure or assignment routing failure.
- First discriminating check: compare assigned, intended and actual policy IDs.
- Immediate containment: restore candidate or stop experiment.
- Recovery criterion: report treatment adherence separately.
### Diagnostic 09: Outcome delay — illustrative
- Symptom: retention seems to collapse.
- Competing explanations: immature cohorts or logging outage.
- First discriminating check: compare cohort age and event ingestion delay.
- Immediate containment: wait for maturity or repair collection.
- Recovery criterion: do not impute improvement from missing users.
### Diagnostic 10: Duplicate outcomes — illustrative
- Symptom: retention rate exceeds feasible range.
- Competing explanations: duplicate events or wrong aggregation grain.
- First discriminating check: count distinct users versus event rows.
- Immediate containment: deduplicate by documented event identity.
- Recovery criterion: recalculate metrics from versioned inputs.
### Diagnostic 11: Metric definition drift — illustrative
- Symptom: dashboard disagrees with analysis.
- Competing explanations: different time zones or windows.
- First discriminating check: compare SQL definitions and population filters.
- Immediate containment: pin one reviewed definition.
- Recovery criterion: backfill only with auditable lineage.
### Diagnostic 12: Selection bias — illustrative
- Symptom: exposed users show a large lift.
- Competing explanations: treatment changes exposure or filtering.
- First discriminating check: compare all-assigned and exposed-only analyses.
- Immediate containment: use assignment-based primary analysis.
- Recovery criterion: treat exposure analysis as descriptive.
### Diagnostic 13: Shared training interference — illustrative
- Symptom: control gradually changes after treatment starts.
- Competing explanations: cross-arm feedback or external seasonality.
- First discriminating check: inspect training inputs and deployment histories.
- Immediate containment: isolate training or redesign estimand.
- Recovery criterion: interpret prior comparison as potentially contaminated.
### Diagnostic 14: Concurrent experiment interaction — illustrative
- Symptom: effect changes when another test starts.
- Competing explanations: interaction or coincident population shift.
- First discriminating check: inspect overlap assignments and prespecified interactions.
- Immediate containment: coordinate allocation or test factorially.
- Recovery criterion: avoid opportunistic subgroup claims.
### Diagnostic 15: Time-of-week imbalance — illustrative
- Symptom: one arm wins only on weekdays.
- Competing explanations: allocation schedule or natural usage seasonality.
- First discriminating check: compare arm counts and outcomes by assignment date.
- Immediate containment: restore concurrent randomized enrollment.
- Recovery criterion: avoid attributing temporal composition to policy.
### Diagnostic 16: Delayed reward bias — illustrative
- Symptom: bandit favors quick clicks.
- Competing explanations: slow reward or genuinely lower long-term value.
- First discriminating check: compare mature cohorts by reward delay.
- Immediate containment: delay updates or redesign reward.
- Recovery criterion: evaluate whole policy on mature outcomes.
### Diagnostic 17: Discovery regression — illustrative
- Symptom: listening rises while novelty falls.
- Competing explanations: familiarity bias or content supply change.
- First discriminating check: inspect novelty and catalog exposure with listening.
- Immediate containment: adjust policy objective or stop rollout.
- Recovery criterion: retain long-term guardrails.
### Diagnostic 18: Latency regression — illustrative
- Symptom: candidate improves reward but misses deadline.
- Competing explanations: larger model or slow feature dependency.
- First discriminating check: trace latency by dependency and version.
- Immediate containment: reduce cost or route to approved fallback.
- Recovery criterion: compare end-to-end tail latency.
### Diagnostic 19: Hot key overload — illustrative
- Symptom: some requests time out.
- Competing explanations: skewed popular entities or uneven partitions.
- First discriminating check: inspect load distribution and per-key latency.
- Immediate containment: rebalance or cache eligible immutable data.
- Recovery criterion: verify correctness under peak load.
### Diagnostic 20: Insufficient capacity — illustrative
- Symptom: errors spike during traffic ramp.
- Competing explanations: underprovisioning or retry amplification.
- First discriminating check: compare arrival, completion and retry rates.
- Immediate containment: reduce ramp and restore service headroom.
- Recovery criterion: load-test representative request mix.
### Diagnostic 21: Unsafe retry — illustrative
- Symptom: duplicate decisions or events appear.
- Competing explanations: unbounded retries or missing idempotency.
- First discriminating check: correlate request identifiers and retry attempts.
- Immediate containment: bound retries and deduplicate appropriately.
- Recovery criterion: monitor duplicate rate and outcome counts.
### Diagnostic 22: Stale assignment — illustrative
- Symptom: users switch variants after configuration update.
- Competing explanations: cache invalidation or identity change.
- First discriminating check: compare assignment timestamps and configuration versions.
- Immediate containment: restore stickiness for active experiment.
- Recovery criterion: retain assignment history for analysis.
### Diagnostic 23: Policy feedback loop — illustrative
- Symptom: narrow content exposure reinforces itself.
- Competing explanations: biased feedback or actual preference concentration.
- First discriminating check: inspect exposure diversity and exploration support.
- Immediate containment: adjust exploration subject to safety constraints.
- Recovery criterion: measure long-term outcomes experimentally.
### Diagnostic 24: Positivity violation — illustrative
- Symptom: offline counterfactual estimate is unstable.
- Competing explanations: some actions never shown or bad propensity logs.
- First discriminating check: inspect propensity distribution by context.
- Immediate containment: collect supported exploration where ethical.
- Recovery criterion: do not extrapolate unsupported actions.
### Diagnostic 25: Privacy violation — illustrative
- Symptom: restricted signal enters a feature.
- Competing explanations: bad permission enforcement or stale consent.
- First discriminating check: audit feature lineage and authorization decisions.
- Immediate containment: block the signal and follow incident procedure.
- Recovery criterion: retrain or delete artifacts when required.
### Diagnostic 26: Late deletion — illustrative
- Symptom: deleted records remain in derived tables.
- Competing explanations: incomplete propagation or delayed batch.
- First discriminating check: trace deletion marker through data lineage.
- Immediate containment: execute approved deletion workflow.
- Recovery criterion: audit downstream artifacts and retention.
### Diagnostic 27: Sample size shortfall — illustrative
- Symptom: interval is too wide for a decision.
- Competing explanations: low enrollment or high variance.
- First discriminating check: compare planned and realized eligible population.
- Immediate containment: extend only under valid preplanned inference.
- Recovery criterion: report uncertainty instead of declaring equivalence.
### Diagnostic 28: Multiple comparisons — illustrative
- Symptom: one of many slices appears significant.
- Competing explanations: chance or genuine heterogeneous effect.
- First discriminating check: review the entire set and selection procedure.
- Immediate containment: use prespecified contrasts and appropriate correction.
- Recovery criterion: replicate discovery in a fresh experiment.
### Diagnostic 29: Guardrail conflict — illustrative
- Symptom: primary improves while discovery worsens.
- Competing explanations: objective tradeoff or measurement defect.
- First discriminating check: validate both metrics and segment outcomes.
- Immediate containment: apply the predeclared tradeoff and stop rule.
- Recovery criterion: escalate unresolved product decisions.
### Diagnostic 30: Rollback inconsistency — illustrative
- Symptom: flag rolls back but old policy persists.
- Competing explanations: cached artifact or control-plane lag.
- First discriminating check: trace configuration propagation and serving versions.
- Immediate containment: invalidate safely and verify the served version.
- Recovery criterion: test rollback before future ramps.
### Diagnostic 31: Logging join failure — illustrative
- Symptom: many assigned users lack outcomes.
- Competing explanations: key mismatch or genuine inactivity.
- First discriminating check: compare join-key completeness and known synthetic events.
- Immediate containment: repair joins without excluding inactive users.
- Recovery criterion: recompute completeness by arm.
### Diagnostic 32: Simpson reversal — illustrative
- Symptom: aggregate lift reverses in every cohort.
- Competing explanations: composition imbalance or weighting error.
- First discriminating check: reconcile common cohort weights and counts.
- Immediate containment: use prespecified standardized estimates.
- Recovery criterion: retain aggregate intention-to-treat as appropriate.
### Diagnostic 33: Incorrect unit — illustrative
- Symptom: sessions treated as independent users.
- Competing explanations: analytics grain mismatch or repeated exposure.
- First discriminating check: count users and sessions per assignment unit.
- Immediate containment: aggregate or use valid clustered inference.
- Recovery criterion: review uncertainty at assignment grain.
### Diagnostic 34: Policy mapping error — illustrative
- Symptom: candidate serves wrong model.
- Competing explanations: configuration typo or stale resolver.
- First discriminating check: compare intended experiment mapping with actual artifact IDs.
- Immediate containment: stop ramp and restore approved mapping.
- Recovery criterion: test mappings before enrollment.
### Diagnostic 35: Catalog rights change — illustrative
- Symptom: recommendation availability drops.
- Competing explanations: licensing constraint or stale availability data.
- First discriminating check: compare serving eligibility with current rights feed.
- Immediate containment: filter unavailable content and use safe fallback.
- Recovery criterion: check availability before exposure.
### Diagnostic 36: Cold-start degradation — illustrative
- Symptom: new users receive poor recommendations.
- Competing explanations: sparse features or missing identity.
- First discriminating check: compare outcomes and fallback rates by tenure.
- Immediate containment: use approved cold-start policy.
- Recovery criterion: measure separately without changing primary population.
### Diagnostic 37: Nonstationary catalog — illustrative
- Symptom: offline replay overstates performance.
- Competing explanations: stale items or changing availability.
- First discriminating check: replay against historical availability snapshots.
- Immediate containment: rebuild evaluation with time-correct catalog.
- Recovery criterion: validate online with appropriate safeguards.
### Diagnostic 38: Operational metric blindness — illustrative
- Symptom: analysis shows uplift during an outage.
- Competing explanations: outcomes missing or selective fallback.
- First discriminating check: align service incidents with assignment and logging.
- Immediate containment: repair telemetry and bound claims.
- Recovery criterion: jointly review service and product metrics.
### Diagnostic 39: Cross-device incompatibility — illustrative
- Symptom: variant looks weak on one client.
- Competing explanations: unsupported feature or rendering bug.
- First discriminating check: compare assignment, rendering and client versions.
- Immediate containment: disable incompatible clients before enrollment.
- Recovery criterion: publish eligibility and compatibility tests.
### Diagnostic 40: Experimental peeking — illustrative
- Symptom: early noisy win triggers rollout.
- Competing explanations: optional stopping or real early effect.
- First discriminating check: inspect decision timestamps and sequential plan.
- Immediate containment: apply the prespecified sequential method or continue.
- Recovery criterion: prespecify monitoring rules.

### Illustrative launch review: eligibility
- Identify the legal entity and product surface under evaluation.
- Identify the population that can receive both variants.
- Specify the unit of randomization before collecting outcomes.
- Specify the timestamp at which eligibility is assessed.
- State whether logged-out sessions are in scope.
- Define how cross-device identities are resolved.
- Identify shared-account limitations and their consequence.
- Verify that assignment happens before treatment-dependent behavior.
- Document geographic and client-version exclusions.
- Establish whether exclusions apply equally to both arms.
- Define how newly eligible users enter during the enrollment period.
- Preserve the denominator even if users never request Home.
- Document how consent changes affect assignment and analysis.
- Capture eligibility-rule version in the experiment record.
- Review the population definition with the responsible product owner.
### Illustrative launch review: assignment
- Identify the assignment mechanism and its owner.
- Specify expected allocation proportions at enrollment.
- Specify how allocation changes are versioned.
- Ensure allocation changes do not silently reassign existing users.
- Define how unassigned requests behave safely.
- Specify whether assignment is durable across app restarts.
- Record the assignment timestamp independently from exposure.
- Define the expected behavior when assignment data are stale.
- Protect assignment keys from unauthorized access.
- Distinguish an experiment flag from the serving policy artifact.
- Verify that both arms can be mapped to available artifacts.
- Reconcile counts from assignment with expected allocation.
- Break down discrepancies by client and enrollment time.
- Investigate mismatches before interpreting outcome comparisons.
- Preserve the original assignment history after a rollback.
### Illustrative launch review: policy serving
- Map each assignment to an approved policy version.
- Ensure model dependencies are available in each deployment region.
- Record which model version actually answered each request.
- Define what constitutes a successful recommendation response.
- Include catalog eligibility and availability checks.
- Define behavior when features are missing.
- Define behavior when features are older than the approved limit.
- Separate a cache hit from a newly computed decision.
- Record fallback decisions without logging unnecessary personal data.
- Make model and feature version identifiers queryable by incident responders.
- State request deadline and dependency budgets from measured requirements.
- Limit retries so a dependency outage does not amplify load.
- Measure tail latency separately from average latency.
- Monitor capacity during both rollout and rollback.
- Confirm fallback remains safe for cold-start users.
### Illustrative launch review: measurement
- Define each outcome's numerator and denominator explicitly.
- Tie outcome identity to the randomization unit.
- Record the observation start and end for each enrolled unit.
- Distinguish event time from ingestion time.
- Define a late-arrival cutoff before final reporting.
- Verify that all compared cohorts have matured.
- Define primary metric direction and practical decision threshold.
- Define discovery and reliability guardrails separately.
- Specify whether a metric is user-level, session-level or request-level.
- Aggregate repeated events consistently with the randomization unit.
- Monitor missing outcome joins by assigned arm.
- Reconcile ingestion counts against independent source totals.
- Version metric definitions rather than silently changing a dashboard.
- Separate provisional monitoring from final decision analysis.
- Record all planned exclusions and deviations.
### Illustrative launch review: causal validity
- Declare the target population and treatment contrast.
- Explain the intended interpretation of assignment effects.
- Check whether serving failures differ across arms.
- Assess whether users can influence each other.
- Inspect whether model training draws feedback from both arms.
- Decide whether shared learning is isolated or part of the target intervention.
- Identify concurrent experiments on the same product surface.
- Specify how overlap is controlled or modeled.
- Consider marketplace and creator-supply feedback where relevant.
- Ensure that aggregate and subgroup estimands are not conflated.
- Choose an uncertainty method consistent with the assignment design.
- Plan multiplicity handling for claims across outcomes or slices.
- Prespecify a sequential method if continuous monitoring affects stopping.
- Assess whether missingness can plausibly differ by treatment.
- Treat unexplained instrumentation differences as a launch blocker.
### Illustrative launch review: recovery and governance
- Name a decision owner for stopping the experiment.
- Name an operational owner for policy rollback.
- Test the rollback path before scaling traffic.
- Distinguish stopping enrollment from changing existing assignments.
- Confirm that the fallback has adequate capacity.
- Record the observed time at which each serving region changes policy.
- Compare actual artifact versions with the intended rollback configuration.
- Preserve incident and deployment records for subsequent interpretation.
- Protect user-level logs with an approved access policy.
- Apply documented retention and deletion rules.
- Escalate unresolved privacy concerns before collecting new features.
- Verify that model artifacts do not contain prohibited data.
- Record deviations from the pre-analysis plan.
- Assign ownership for the final decision and follow-up monitoring.
- Revisit the architecture if feature, latency or experiment requirements change.

### Incident analysis principles
These diagnostics are hypothetical; S1 documents no incident corresponding to them.
A signal at one boundary must be reconciled with upstream and downstream records before a root cause is declared.
An assignment discrepancy, a serving discrepancy, and a missing outcome are different failure classes.
During containment, preserve original assignments and timestamps to support later analysis.
Pause or rollback according to the experiment's predefined user-safety and reliability rules.
A corrected service does not retroactively repair biased or missing historical records.
Document the affected interval and assess whether the intended estimand still matches the operational intervention.
Avoid treating a clean dashboard after recovery as proof that the historical treatment effect is unbiased.

## 13. Transferable Lesson
Keep adaptive decision-making and causal evaluation conceptually separate even if their systems share interfaces.
The source supports this lesson for Spotify's reported infrastructure and objectives, not as a universal theorem.
A small product may choose integrated software while still maintaining clean logical ownership and independent evaluation.
A team with immediate outcomes and one well-defined objective may rationally evaluate an adaptive allocation design.
Such a choice requires its own inference, operational and governance analysis; it does not contradict Spotify's context-specific choice.
The central failure boundary is confusing an optimization signal with evidence that a complete policy benefits users.

## 14. Connections & Cross-References
### 14a. Related cases, theory, and cheat sheets
[Netflix GenRec ranking](01_Netflix_GenRec_Ranking.md) is a related authored case: both distinguish model evaluation from product outcomes, but its publication and comparison are different.
The [product experiment sheet](../../../02_data_science_method_playbooks/07_product_analytics_playbook/Cheat_Sheets/08_AB_Testing_Product_Experiments.md) develops assignment, outcome maturity and uncertainty; its examples are not Spotify results.
[Recommendation collection index](INDEX.md) records other planned cases; planned titles are not evidence.
The [course numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) govern all illustrative arithmetic.
A/B tests identify assignment effects under a valid design; online learning chooses actions during interaction.
Contextual policy evaluation is a separate question from comparing the value of a complete policy.
Offline ranking metrics can screen policies but need not predict long-horizon outcomes.

### 14b. What this case leaves open
| Open question | Evidence gap | Resolving evidence | Owner / trigger |
|---|---|---|---|
| Exact assignment algorithm | S1 does not specify hashing or identity | Approved platform documentation | Experimentation owner / migration |
| Concrete model architectures | S1 discusses model families, not a specific deployed architecture | Model documentation and deployment records | ML owner / model release |
| Experiment effect size | No quantified uplift is disclosed | Reviewed analysis of a named experiment | Data science owner / decision |
| Concurrent experiment controls | S1 states teams experiment in parallel, not isolation mechanics | Allocation and interference design | Platform owner / scale change |
| Guardrail thresholds | No numerical product thresholds supplied | Product-specific pre-analysis plan | Product owner / launch |
| Observability and fallback | No detailed service runbook supplied | Incident and service design records | Reliability owner / deployment |
| Counting interval | Source says last year without precise dates | Author clarification or source update | Curriculum owner / source correction |

### 14c. Intense interview pack — questions with checked answers
#### Question 01: What distinction drives the design?
**Candidate prompt:** What distinction drives the design?
**Answer:** ML selects experiences; experimentation compares assigned whole systems.
**Reasoning:** Prediction optimizes a policy, whereas randomized assignment supports a system-level comparison.
**Follow-up:** Could one physical platform host both?
**Follow-up answer:** Yes, provided logical contracts and independent evaluation remain clear.
**Red flag:** Claiming that ML predictions themselves prove causal lift.
**Scoring:** Strong: identify both boundaries and explain why integration is necessary.
**Answer support:** S1 sections on separation and integration.

#### Question 02: Where does Spotify put contextual bandits?
**Candidate prompt:** Where does Spotify put contextual bandits?
**Answer:** In its ML stack, as personalization systems.
**Reasoning:** The policy conditions decisions on user characteristics and needs model-serving infrastructure.
**Follow-up:** Are experimental traffic-allocation MABs the same thing?
**Follow-up answer:** No: non-contextual MABs adapt aggregate allocation among variants.
**Red flag:** Saying Spotify uses contextual bandits as its experiment allocator.
**Scoring:** Strong: distinguish contextual policy from allocation methodology.
**Answer support:** S1 technical and overlap sections.

#### Question 03: Does Spotify use MABs for experimentation?
**Candidate prompt:** Does Spotify use MABs for experimentation?
**Answer:** The article says it does not use multi-armed bandits for experimentation.
**Reasoning:** The authors cite multi-objective tradeoffs, delayed outcomes and practical usability.
**Follow-up:** Are bandits always wrong elsewhere?
**Follow-up answer:** No; different reward delays and objectives can change the tradeoff.
**Red flag:** Generalizing Spotify’s choice into a universal prohibition.
**Scoring:** Strong: state the qualified published claim and alternatives.
**Answer support:** S1 one-dimensional focus section.

#### Question 04: What do 58 and 520 establish?
**Candidate prompt:** What do 58 and 520 establish?
**Answer:** The authors report 58 teams and 520 mobile Home experiments during the preceding year.
**Reasoning:** These are activity counts with no disclosed success or causal effect.
**Follow-up:** Can one infer average simultaneous experiments?
**Follow-up answer:** No; the source gives neither overlap nor duration.
**Red flag:** Presenting 520 as a percentage improvement.
**Scoring:** Strong: quote population, period and limitation.
**Answer support:** S1 build-tools-that-scale section.

#### Question 05: Why test a contextual bandit with A/B testing?
**Candidate prompt:** Why test a contextual bandit with A/B testing?
**Answer:** Compare the value of the complete adaptive policy to a baseline.
**Reasoning:** The internal reward and arm choices do not establish the policy effect on user outcomes.
**Follow-up:** What if both policies learn online?
**Follow-up answer:** Specify and isolate learning dynamics or define the intended interaction in the estimand.
**Red flag:** Claiming each internal action must be the randomized experiment arm.
**Scoring:** Strong: define policy-level treatment and a relevant comparator.
**Answer support:** S1 contextual-bandit and evaluation sections.

#### Question 06: Compute the illustrative retention difference.
**Candidate prompt:** Compute the illustrative retention difference.
**Answer:** 42% minus 40% is 2 percentage points, or 5% relative to control.
**Reasoning:** The counts are 4200/10000 and 4000/10000; 0.02/0.40 equals 0.05.
**Follow-up:** What if the control rate is zero?
**Follow-up answer:** The relative change is undefined; the absolute difference remains meaningful.
**Red flag:** Confusing a 2 percentage-point increase with 2% relative lift.
**Scoring:** Strong: show denominators and distinguish absolute and relative scales.
**Answer support:** Section 6 E1; all inputs illustrative.

#### Question 07: How uncertain is that difference?
**Candidate prompt:** How uncertain is that difference?
**Answer:** The illustrative standard error is approximately 0.00695 and the approximate 95% interval is 0.64 to 3.36 percentage points.
**Reasoning:** Use the independent two-proportion standard error and 1.96 normal multiplier under the stated assumptions.
**Follow-up:** What changes when users are clustered?
**Follow-up answer:** Use a design and uncertainty estimator appropriate for the randomization and cluster structure.
**Red flag:** Reporting a narrow independent-user interval despite strong clustering.
**Scoring:** Strong: calculate and state approximation assumptions.
**Answer support:** Section 6 E2.

#### Question 08: What is the primary analysis population?
**Candidate prompt:** What is the primary analysis population?
**Answer:** For an assignment effect, all eligible randomized units, regardless of exposure or fallback.
**Reasoning:** Restricting to exposed users conditions on a post-assignment event and can induce bias.
**Follow-up:** Can actual exposure be analyzed?
**Follow-up answer:** Yes, as a clearly labeled diagnostic or with an appropriate causal compliance design.
**Red flag:** Dropping treatment users who failed to receive the candidate.
**Scoring:** Strong: connect denominator to intention-to-treat estimand.
**Answer support:** Sections 5a and 9 illustrative protocol.

#### Question 09: Diagnose fewer treatment impressions.
**Candidate prompt:** Diagnose fewer treatment impressions.
**Answer:** First distinguish a rendering failure from missing exposure instrumentation.
**Reasoning:** Compare assignment counts, server decisions and client impressions by arm and client version.
**Follow-up:** Should exposed-only lift decide launch?
**Follow-up answer:** No; preserve the assigned population and investigate differential noncompliance.
**Red flag:** Assuming a sample-ratio mismatch at assignment from an exposure mismatch.
**Scoring:** Strong: propose discriminating signals and a safe action.
**Answer support:** Diagnostic 02.

#### Question 10: Why can immediate listening be misleading?
**Candidate prompt:** Why can immediate listening be misleading?
**Answer:** A system can maximize familiar listening while harming discovery or longer-term satisfaction.
**Reasoning:** Single-objective rewards need not represent multiple user and product goals.
**Follow-up:** Does a lower novelty metric alone prove harm?
**Follow-up answer:** No; define and validate discovery and evaluate the whole set of outcomes.
**Red flag:** Treating every engagement increase as a sustainable user benefit.
**Scoring:** Strong: articulate a tradeoff and a measurable guardrail.
**Answer support:** S1 multi-objective discussion.

#### Question 11: How do delayed outcomes affect adaptive allocation?
**Candidate prompt:** How do delayed outcomes affect adaptive allocation?
**Answer:** The algorithm may wait weeks for mature cohort outcomes, reducing apparent speed advantages.
**Reasoning:** Updating on immature observations can favor immediate proxies or uneven cohorts.
**Follow-up:** What if an outcome is immediate and reliable?
**Follow-up answer:** Faster adaptive allocation may be feasible but timing and inferential design still matter.
**Red flag:** Assuming every business outcome is available immediately.
**Scoring:** Strong: tie delay to allocation and measurement validity.
**Answer support:** S1 delayed-metric discussion.

#### Question 12: Size the illustrative request-serving pool.
**Candidate prompt:** Size the illustrative request-serving pool.
**Answer:** At 2000 requests/s, 250 per instance and 80% target utilization, ten instances are required before redundancy.
**Reasoning:** The effective target capacity is 200 requests/s per instance; 2000/200 is ten.
**Follow-up:** How many with one instance of headroom?
**Follow-up answer:** Eleven under the stated homogeneous capacity and single-failure assumption.
**Red flag:** Inferring p99 latency or Spotify capacity from this fluid estimate.
**Scoring:** Strong: show capacity units and identify excluded constraints.
**Answer support:** Section 6 W1.

#### Question 13: What if a policy learns from both experimental arms?
**Candidate prompt:** What if a policy learns from both experimental arms?
**Answer:** Shared learning can create interference and change the system-level estimand.
**Reasoning:** Control outcomes may be affected by treatment observations through shared model updates.
**Follow-up:** How could the experiment change?
**Follow-up answer:** Separate training feedback or explicitly design and estimate a policy with shared effects.
**Red flag:** Assuming randomization alone eliminates all cross-arm spillovers.
**Scoring:** Strong: identify mechanism and offer a design-dependent remedy.
**Answer support:** Section 9 interference discussion.

#### Question 14: How would you handle a candidate-serving outage?
**Candidate prompt:** How would you handle a candidate-serving outage?
**Answer:** Use a preapproved safe fallback, record actual and assigned versions, and preserve assignment in the primary analysis.
**Reasoning:** Fallback is noncompliance; silently relabeling units would bias or change the question.
**Follow-up:** When should the experiment stop?
**Follow-up answer:** Apply prespecified operational stop rules and involve the accountable owner.
**Red flag:** Excluding failed candidate requests to inflate observed quality.
**Scoring:** Strong: cover user safety, logging and estimand.
**Answer support:** Section 8b illustrative failed trace.

#### Question 15: What if the only outcome is immediate and there is one objective?
**Candidate prompt:** What if the only outcome is immediate and there is one objective?
**Answer:** An adaptive allocation approach may become more attractive, but it requires a separately justified design.
**Reasoning:** The source gives context-specific reasons, not a theorem forbidding bandits.
**Follow-up:** Would contextual choices still be personalization?
**Follow-up answer:** Yes: conditioning decisions on user features is a personalized policy that needs evaluation.
**Red flag:** Assuming Spotify’s specific organizational choice applies unchanged to every product.
**Scoring:** Strong: change the recommendation while preserving the evaluation boundary.
**Answer support:** S1 bandit limitations and context sections.

#### Question 16: How would you assess a reported policy lift from an observational replay?
**Candidate prompt:** How would you assess a reported policy lift from an observational replay?
**Answer:** Treat it as a model-dependent offline estimate, not a randomized system effect.
**Reasoning:** Historical exposure, propensity support and feedback selection can bias replay.
**Follow-up:** What if some actions have zero historical probability?
**Follow-up answer:** Their counterfactual outcomes are not identified by inverse propensity weighting alone.
**Red flag:** Assigning infinite or invented weights to unsupported actions.
**Scoring:** Strong: discuss support, logging and a credible online evaluation.
**Answer support:** Sections 5 and 9; diagnostic 24.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Domain | Music personalization and experimentation |
| Problem type | Separate policy serving from whole-system evaluation |
| Method | ML policy plus A/B comparison with prior or static policy where appropriate |
| Primary result | Published stack separation and activity counts; effect size not reported |
| Key lesson | A contextual policy needs independent system evaluation; infrastructure requirements differ |
| Tags | evaluation-mismatch; label-delay; cost-latency-tradeoff |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-01-08-002 / 1.0-draft / 2026-11-07 |
| Evidence level | Published first-party description; numerical examples illustrative |
| Implementation stage | Production as described in S1; teaching architecture proposed |
| Baseline / primary metric | Experiment-specific; source does not disclose one |
| Key limitation | No raw data, detailed assignment method, effect size or independent audit |
| Review trigger / owner | Source correction or architecture change / curriculum maintainer |
