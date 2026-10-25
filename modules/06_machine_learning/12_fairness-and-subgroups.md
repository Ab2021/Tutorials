# Fairness and Subgroups

> **Module 06 — Machine Learning Foundations**
> Topic 12 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 02, 07, and 09
> Version 2026.3

## 1. Why this topic exists

A model can perform well on average and badly for the people who need it most. The average AUROC is
a summary over a population, and a population is not homogeneous. When a model is deployed across
groups with different prevalence, different access to care, or different documentation, the same
threshold can produce different sensitivity, different PPV, and different net benefit. A report that
gives only the overall metric has not described the model's behaviour for any particular group.

The evidence that this matters is not hypothetical. Obermeyer et al. (2019) showed that a widely
used population-health algorithm used cost as a proxy for need; at the same risk score, Black
patients were substantially sicker than White patients, and correcting the label would have
increased the proportion of Black patients identified for additional help from 17.7 percent to 46.5
percent. The mechanism was the label, not the algorithm. Vyas et al. (2020) documented race-based
correction factors in clinical algorithms, including the eGFR equation, and argued that race is
often a proxy for racism and social disadvantage rather than a biological variable.

The statistical structure is equally important. When prevalence differs across groups, several
fairness criteria cannot hold simultaneously for an imperfect model. Chouldechova (2017) proved
that predictive parity and equal false positive and false negative rates are incompatible when base
rates differ. Kleinberg, Mullainathan, and Raghavan (2016) proved that calibration within groups,
balance for the positive class, and balance for the negative class can hold together only under
perfect prediction or equal base rates. The impossibility is not a failure of any particular
method; it is a property of the criteria. A team must choose which criterion matters for the
decision and state the trade-off.

Subgroup analysis also has a sample-size problem. A subgroup with 14 events can produce an AUROC
that looks alarming or reassuring and is consistent with almost anything. Testing many subgroups
inflates the chance of a false finding: with 12 independent tests at the 5 percent level, the
probability of at least one false positive is about 46 percent. Brookes et al. (2004) showed that a
trial powered at 80 percent for an overall effect has only about 29 percent power to detect an
interaction of the same magnitude, and that the sample size must be roughly four times larger to
test an interaction. Wallach et al. found that of 117 subgroup claims in trial abstracts, only 46
had a significant interaction test, and none of the five claims that were subsequently tested was
reproduced.

A fairness analysis is also a communication problem, and it is a governance problem. It should be
reviewed by someone who is not the model owner, because the owner has an interest in the result. The
review should check the criterion, the counts, the intervals, and the mechanism, not only the
conclusion.

A fairness analysis is also a communication problem, and it is a governance problem. The
stakeholders include the groups affected, not only the model owners. Their input is needed to choose
the criterion and to judge whether a mitigation is acceptable.

A fairness analysis is also a communication problem, and it is a governance problem. The
stakeholders include the groups affected, not only the model owners. Their input is needed to choose
the criterion and to judge whether a mitigation is acceptable. A technically correct analysis that
no affected group was consulted on may still be illegitimate.

A fairness analysis is also a communication problem, and it is a governance problem. Someone must
own the criterion, the subgroup list, and the monitoring. Without an owner, the analysis becomes a
one-time appendix that no one updates when the population changes.

A fairness analysis is also a communication problem. A report that says "the model is fair" without
a criterion is unfalsifiable. A report that says "PPV differs by 28 points, but the prevalence
differs by 14 points, and equal PPV would require sensitivity 0.74 in one group and 0.23 in the
other" is a claim that a reader can check. The second report is the one that supports a decision.

This topic teaches subgroup analysis as a pre-specified, uncertainty-aware, mechanism-oriented
practice. You will learn to compute subgroup performance with intervals, demonstrate the
impossibility of satisfying all fairness criteria at once, distinguish a real disparity from a
sample-size artefact, and write a subgroup claim that the data can support.

> **Note:** A subgroup claim requires a count, an interval, a pre-specified comparison, and a
> mechanism. Without those four, a difference is a hypothesis, not a finding.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** subgroup prevalence, sensitivity, specificity, PPV, and AUROC with confidence
   intervals and counts.
2. **Explain** the impossibility result among calibration, equalized odds, and predictive parity
   when prevalence differs, and **demonstrate** it with arithmetic.
3. **Choose** a fairness criterion that matches the decision and the harm, and **state** the
   trade-off it implies.
4. **Distinguish** a prevalence-driven PPV difference from a model-driven disparity.
5. **Apply** a pre-specification rule for subgroup claims, including minimum event counts and
   interaction tests.
6. **Detect** sample-size-driven false subgroup findings and **avoid** claiming them.
7. **Examine** mechanisms, including label proxies, feature availability, and access differences.
8. **Design** a monitoring plan for subgroup performance after deployment.

Outcomes 1 and 2 are computational; 3 through 8 are design and critique.

## 3. The core mental model

Fairness is not one property. It is a family of criteria, and they answer different questions.

```text
Criterion                 Question                                      Equal when
---------                 --------                                      ----------
Demographic parity        Are positive predictions equally frequent?    P(flag | A) = P(flag | B)
Equal opportunity         Are true positives equally frequent?          TPR(A) = TPR(B)
Equalized odds            Are TPR and FPR both equal?                   TPR and FPR equal
Predictive parity         Is PPV equal?                                 PPV(A) = PPV(B)
Calibration within group  Does predicted risk match observed risk?      Calibration in each group
Overall accuracy          Is accuracy equal?                            Accuracy(A) = Accuracy(B)
```

The criteria conflict when prevalence differs. The intuition is Bayes' theorem: PPV depends on
prevalence as well as on sensitivity and specificity. If two groups have the same sensitivity and
specificity but different prevalence, their PPVs differ. To equalize PPV, the group with the lower
prevalence must have higher sensitivity (or higher specificity), which means the TPRs differ. Equal
PPV and equal TPR cannot both hold.

```text
PPV = sens * p / (sens * p + (1 - spec) * (1 - p))

If p_A < p_B and sens/spec are equal, PPV_A < PPV_B.
To raise PPV_A to PPV_B, raise sens_A (or spec_A), which breaks equal TPR.
```

The criteria also differ in their data requirements. Equal opportunity requires a reliable outcome
label for each group; predictive parity requires both the label and the flagged set; calibration
requires a sufficient number of events in each group. A group with few events cannot support any of
them with precision. The analysis should state which criteria the data can support.

A useful diagnostic is to compute all five criteria and present them together. The table is the
evidence that the choice is a trade-off, not a technical fix. A reader who sees one criterion in
isolation may believe the model can satisfy all of them.

A useful diagnostic is to compute all five criteria and present them together. If they all agree,
the choice is easy. If they disagree, the table shows the trade-off and the team can choose with the
evidence. The disagreement is not a reason to hide the analysis; it is the reason to do it.

A fairness analysis should report the criterion before the metrics, because the criterion
determines which metrics matter. A report that lists ten metrics without a criterion invites the
reader to choose the one that looks best. The criterion is the argument; the metrics are the
evidence. A report that lists ten metrics without a criterion invites the
reader to choose the one that looks best.

The criteria can be grouped by what they condition on. Demographic parity conditions on the
prediction only. Equal opportunity and equalized odds condition on the outcome. Predictive parity
and calibration condition on both. The grouping explains why the criteria conflict: they condition
on different quantities, and when prevalence differs, the conditionals cannot all be equal.

The criteria also differ in what they require of the model. Demographic parity requires only the
prediction rate; equal opportunity requires the outcome; predictive parity requires both. A team
that wants to avoid using the outcome in the fairness analysis is limited to demographic parity,
which may not reflect need. A team that uses the outcome must have a reliable label, which returns
to Topic 02.

The choice among criteria is a choice about which error matters and who bears it. Equal opportunity
protects the group with the higher prevalence from being under-flagged. Predictive parity protects
patients who are flagged from being disproportionately false positives. Calibration protects the
meaning of the probability for each group. A team cannot have all three; it must decide which harm
the decision can tolerate.

Subgroup analysis is the empirical counterpart. It estimates the criteria within groups and asks
whether the differences are larger than sampling noise. The estimate requires counts and intervals,
because a subgroup with few events has a wide interval and can appear to show a difference that is
not there.

A useful way to remember the analysis is three questions. Who might be affected differently? What
does the data say about that group? What would it take to make the difference smaller? The first
question defines the groups and the criterion. The second estimates the metrics with uncertainty.
The third examines the mechanism and the mitigation. A report that answers only the second has
measured a disparity without explaining it.

A useful sequence is:

1. Describe the subgroups and their prevalence.
2. Estimate performance within each group, with counts and intervals.
3. Compare the estimates using an interaction test or a difference with an interval, not by
   comparing p-values across groups.
4. Examine the mechanism: label, features, access, or model.
5. Decide whether a subgroup-specific threshold or a model change is justified.
6. Pre-specify what would count as a credible claim.

> **Note:** A difference in PPV across groups is not evidence of a biased model until the
> prevalence difference is accounted for. A difference in sensitivity at the same threshold is
> stronger evidence, because sensitivity is conditional on the outcome and does not depend on
> prevalence.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Protected attribute | A characteristic protected by law or policy | Assumed to be the only source of inequity | Category |
| Sensitive attribute | Synonym for protected attribute | Confused with a clinical variable | Category |
| Subgroup | A subset defined by a characteristic | Defined after seeing results | Category |
| Fairness criterion | A formal condition on predictions across groups | Treated as a single property | Criterion |
| Demographic parity | Equal positive prediction rates | Confused with equal need | Proportion |
| Statistical parity | Synonym for demographic parity | Confused with predictive parity | Proportion |
| Equal opportunity | Equal true positive rates | Confused with equal PPV | Proportion |
| Equalized odds | Equal TPR and FPR | Confused with calibration | Proportions |
| Predictive parity | Equal PPV across groups | Confused with calibration | Proportion |
| Calibration within groups | Predicted risk matches observed risk in each group | Confused with overall calibration | Property |
| Overall accuracy parity | Equal accuracy across groups | Used for rare events | Proportion |
| Treatment parity | Equal allocation of the action | Confused with equal benefit | Proportion |
| Base rate | Prevalence in a group | Confused with the flagged rate | Proportion |
| Prevalence difference | Difference in outcome rate across groups | Ignored in PPV comparisons | Proportion |
| Disparity | A difference in performance or outcome | Assumed to be bias | Difference |
| Disparate impact | A neutral rule with unequal effects | Confused with intentional discrimination | Legal concept |
| Proxy variable | A variable correlated with a protected attribute | Treated as neutral | Feature |
| Redlining | Excluding a group through proxies | Confused with explicit exclusion | Practice |
| Label bias | A label that encodes unequal access or treatment | Confused with model bias | Mechanism |
| Measurement bias | A feature measured differently across groups | Confused with true difference | Mechanism |
| Feature availability bias | A feature missing more often in one group | Confused with prevalence | Mechanism |
| Sample-size bias | A subgroup estimate driven by few events | Confused with a real disparity | Mechanism |
| Multiplicity | Many tests increase false positives | Ignored in subgroup mining | Count |
| Interaction test | A test of whether an effect differs across groups | Confused with a within-group test | Test |
| Pre-specification | Defining subgroups and claims before analysis | Confused with post-hoc exploration | Design |
| Post-hoc subgroup | A subgroup defined after seeing results | Presented as confirmatory | Design |
| Intersectionality | Overlapping group identities | Ignored when sample is small | Category |
| Minimum event count | The events needed for a stable estimate | Ignored | Count |
| Confidence interval | Uncertainty in a subgroup estimate | Replaced by a p-value | Interval |
| Forest plot | A plot of subgroup estimates with intervals | Read as significance testing | Plot |
| Calibration by group | Calibration assessed within each group | Confused with overall calibration | Property |
| Threshold by group | A group-specific operating point | Confused with equal treatment | Policy |
| Affirmative action | A policy to increase representation | Confused with fairness criteria | Policy |
| Mitigation | A change to reduce a disparity | Confused with a guarantee | Action |
| Pre-processing mitigation | Changing the data or labels | Confused with post-processing | Method |
| In-processing mitigation | Changing the model or loss | Confused with pre-processing | Method |
| Post-processing mitigation | Changing the threshold or output | Confused with in-processing | Method |
| Fairness-accuracy trade-off | The cost of a fairness constraint | Assumed to be zero | Trade-off |
| Feedback loop | The model changes the data it learns from | Ignored | Mechanism |
| Monitoring | Tracking subgroup performance after deployment | Omitted | Process |
| Health equity | Fair opportunity to achieve health | Confused with equal outcomes | Goal |
| Structural racism | Systemic patterns that produce inequity | Reduced to an individual variable | Structure |
| Race as a social construct | Race is not a biological category | Treated as biology | Concept |
| Ethnicity | A social and cultural category | Confused with race | Category |
| Sex and gender | Distinct categories | Used interchangeably | Category |
| Disability | A functional status | Ignored in subgroup analysis | Category |
| Age | A protected characteristic in some contexts | Treated as a clinical variable | Years |
| Language | A communication and access factor | Ignored | Category |
| Insurance status | An access proxy | Confused with need | Category |
| Area deprivation | A geographic measure of disadvantage | Confused with individual status | Index |
| Group-specific threshold | A different cut-off for each group | Confused with equal treatment | Policy |
| Fairness audit | A structured review of subgroup performance | Confused with a legal review | Process |
| Disparity metric | A quantitative measure of a group difference | Confused with a causal effect | Metric |

Three definitions deserve more than a table row.

**Disparity versus bias.** A disparity is a difference in outcomes or performance across groups. A
bias is a systematic error in the model or the data that contributes to the disparity. A disparity
can exist without a model bias, because the groups differ in prevalence or need. A model bias can
exist without a measured disparity, because the disparity is offset by another factor. The report
should distinguish the two and should not label every difference as bias.

**Equal treatment versus equal outcome.** A single threshold treats everyone the same way at the
point of prediction, but it can produce unequal outcomes because the groups differ. A group-specific
threshold can produce more equal outcomes but treats people differently. The choice is a policy
decision, and it should be made explicitly, with the trade-off stated.

**Fairness versus clinical validity.** A model can be fair by one criterion and clinically invalid
for a group, for example if it is well calibrated but has poor discrimination in that group. The
subgroup analysis should report both the fairness criterion and the clinical performance, because a
model that satisfies a parity condition but cannot discriminate is not useful.

> **Note:** The impossibility result is a constraint on criteria, not an excuse for inaction. A team
> can still choose a criterion, measure it, and mitigate the harm it cares about.

## 5. The mechanics, step by step

Subgroup analysis has fourteen steps. Steps 1 through 4 define the groups and the criterion. Steps 5
through 8 estimate and compare. Steps 9 through 14 examine mechanisms, mitigate, and monitor.

### Step 1. Define the subgroups and the rationale

**Input:** the decision context, the population, and the equity question.

**Output:** a pre-specified list of subgroups, each with a rationale and a minimum size.

**Rationale:** A subgroup is a claim about who might be affected differently. The list should be
driven by the clinical decision and by known inequities, not by the columns available in the
database. A pre-specified list limits the multiplicity problem and prevents the team from reporting
only the groups that look interesting after the fact.

The list should include the groups that the clinical team expects to differ, not only the groups
that are easy to measure. A group that is not in the data cannot be analyzed, but its absence should
be stated as a limitation.

The pre-specified list should be short. A list of 30 subgroups is not a plan; it is a fishing
expedition. Three to six groups, chosen for clinical or policy reasons, is usually enough for a
model report.

**Branch point:** if a subgroup is defined after seeing the results, label it exploratory. An
exploratory subgroup can generate a hypothesis; it cannot support a confirmatory claim.

### Step 2. Choose the fairness criterion

**Input:** the decision, the harm, and the stakeholder input.

**Output:** a primary criterion and one or two secondary criteria, with the reason for each.

**Rationale:** The criteria conflict when prevalence differs, so the team must choose which harm
matters. Equal opportunity protects against under-flagging the group with higher prevalence;
predictive parity protects flagged patients from false positives; calibration protects the meaning
of the probability. The choice should be explicit and should be tied to the decision.

The criterion should be chosen before the subgroup analysis, not after seeing which criterion the
model satisfies. A team that tries all five criteria and reports the one that looks best has
selected on the data. The pre-specified criterion is the one that should be defended.

**Branch point:** if the stakeholders disagree about the criterion, present the trade-off rather
than choosing silently. The disagreement is a legitimate policy question, and the analysis should
inform it rather than settle it.

### Step 3. Describe the groups

**Input:** the cohort and the subgroup identifiers.

**Output:** counts, prevalence, age distribution, comorbidity, and missingness by group.

**Rationale:** The description separates a prevalence difference from a model difference. A group
with a higher prevalence will have a higher PPV at the same sensitivity and specificity, and a group
with more missing data may have a different effective sample. The description is the context for
every subsequent metric.

**Branch point:** if a group has fewer than about 100 events, the subgroup analysis is descriptive
only. State that before computing the metrics.

### Step 4. Pre-specify the claim rule

**Input:** the groups, the criterion, and the sample sizes.

**Output:** a rule stating what would count as a credible subgroup claim: minimum events, a
pre-specified interaction test, an effect-size threshold, and a replication requirement.

**Rationale:** Without a rule, any difference can be presented as a finding. The rule makes the
analysis falsifiable and protects against the multiplicity problem. A common rule is at least 100
events and 100 non-events, a pre-specified interaction test at a corrected alpha, an absolute
difference of at least 0.05 in the metric, and replication in an external cohort.

The rule should be written before the analysis and should apply to every subgroup, including the
ones the team expects to be fine. A rule applied only to the surprising finding is not a rule.

**Branch point:** if no subgroup meets the rule, the correct conclusion is that the data cannot
support a subgroup claim, not that there is no disparity.

### Step 5. Estimate performance within each group

**Input:** the held-out predictions and outcomes.

**Output:** prevalence, sensitivity, specificity, PPV, NPV, AUROC, and calibration slope by group,
with confidence intervals and counts.

**Rationale:** The within-group estimates are the raw evidence. They should be computed on the same
held-out predictions and at the same threshold as the overall analysis, so that the comparison is
consistent. The intervals show whether a difference is larger than sampling noise.

The estimates should be reported on the same threshold as the overall analysis. A subgroup analyzed
at a different threshold is not comparable, and the difference may be the threshold rather than the
group. If a group-specific threshold is part of the mitigation, report the pre-mitigation estimates
at the shared threshold first.

**Branch point:** if a group has no events or no non-events, the corresponding metric is undefined.
A group with 14 events can produce a point estimate that looks alarming; the interval is the evidence,
and it is wide.
Report the count and avoid the metric. An undefined metric is not a zero. A group with 14 events
can produce a point estimate that looks alarming; the interval is the evidence, and it is wide.

### Step 6. Compare with an interaction test, not within-group p-values

**Input:** the subgroup estimates and the model.

**Output:** an interaction test or a difference with an interval, and a statement of the comparison.

**Rationale:** Comparing a significant p-value in one group with a non-significant p-value in
another is not a test of the difference. The correct question is whether the effect differs across
groups, which requires an interaction term or a difference in estimates with an interval. The
distinction is one of the most common errors in subgroup reporting.

The interaction test should be pre-specified with a corrected alpha if multiple subgroups are
tested. A test at the 5 percent level for each of 12 subgroups has a high chance of a false
positive, as the multiplicity arithmetic shows. The correction is part of the pre-specification.

**Branch point:** if the interaction test is underpowered, the absence of a significant interaction
does not prove homogeneity. Report the interval and the power limitation.

### Step 7. Check calibration by group

**Input:** the predicted risks and outcomes by group.

**Output:** calibration slope, intercept, and plot by group, with counts.

**Rationale:** A model can have similar discrimination across groups and different calibration. A
group whose risk is under-predicted may be under-served, even if the ranking is good. Calibration
by group is a fairness criterion in its own right, and it is often the most clinically relevant.

Calibration by group should be reported even when the criterion is equal opportunity, because a
group that is under-predicted may be under-served in a way that the TPR does not capture. The
calibration plot is the evidence that the probability means the same thing in each group.

**Branch point:** if a group is miscalibrated, recalibrate within the group or adjust the threshold,
and report the change in the other criteria.

### Step 8. Quantify the impossibility trade-off

**Input:** the group prevalences and the model's sensitivity and specificity.

**Output:** a table showing the PPV at equal sensitivity, the sensitivity required for equal PPV,
and the maximum attainable PPV for each group.

**Rationale:** The trade-off table makes the impossibility concrete. It shows that equal PPV
requires unequal sensitivity, and that some PPV targets are unattainable in low-prevalence groups.
The table is the evidence that the choice among criteria is a policy decision, not a technical
fix.

The trade-off table should also show the maximum attainable PPV for each group, because a target
above the maximum is not a policy choice but an impossibility. The maximum is `p / (p + (1 - spec) *
(1 - p))` at sensitivity 1.0. A team that sets a target above it will pursue a mitigation that cannot
work.

The trade-off should be presented as a table with the prevalence, the equal-sensitivity PPV, the
sensitivity required for the target PPV, and the maximum PPV. The table is the evidence that the
choice is a policy decision, and it should be in the main text, not an appendix. A reader who sees
only the PPVs will misread the arithmetic.

**Branch point:** if the required sensitivity exceeds 1 for a target PPV, the target is impossible
at that specificity. Report the maximum attainable PPV and the policy implication.

### Step 9. Examine the mechanism

**Input:** the label definition, the features, and the workflow.

**Output:** a mechanism assessment covering label proxy, feature availability, access, and model.

**Rationale:** A disparity can come from the label (cost as a proxy for need), the features (a lab
measured less often in one group), the access (a group treated at a hospital with different
coding), or the model (a functional form that fits one group better). The mechanism determines the
mitigation. A label proxy requires a label change; a feature availability problem requires a data
change; a model problem may require a model change.

A useful mechanism checklist has four items. Is the label a proxy for something else? Is a feature
measured differently across groups? Do the groups access care differently? Does the model fit one
group better? The first two are data problems, the third is a system problem, and the fourth is a
model problem. The mitigation should match the item.

**Branch point:** if the mechanism is label bias, a fairness-aware model cannot fix it. The label
must be changed, as in the Obermeyer case.

### Step 10. Choose a mitigation

**Input:** the mechanism and the criterion.

**Output:** a mitigation plan: pre-processing (label or data), in-processing (constraints or
weights), or post-processing (group-specific thresholds), with the expected effect.

**Rationale:** The mitigation should target the mechanism. A group-specific threshold can equalize
a criterion but does not fix a biased label; a label change can fix the mechanism but may require
re-labelling and revalidation. The plan should state which criteria improve and which worsen.

The mitigation plan should state the expected effect on each criterion, not only the target one. A
group-specific threshold that equalizes PPV may reduce sensitivity in the low-prevalence group; a
label change may improve predictive parity but change the model's overall performance. The plan is a
package, and the package should be evaluated as a whole.

**Branch point:** if the mitigation worsens another criterion materially, the trade-off must be
presented to the decision-maker, not hidden.

### Step 11. Check the mitigation on held-out data

**Input:** the mitigation and an independent sample.

**Output:** the subgroup metrics after mitigation, with intervals.

**Rationale:** A mitigation fitted on the same data used to measure the disparity will appear to
work by construction. The check must be on held-out data, and it must include the criteria that were
not targeted, because the trade-off may have moved the disparity elsewhere.

The held-out check should include the group that was not targeted. A mitigation that improves the
target group and harms another has moved the disparity, not removed it. The report should show both.

**Branch point:** if the mitigation does not improve the target criterion on held-out data, it is
not a mitigation. Report the failure.

### Step 12. Consider intersectional groups

**Input:** the subgroup identifiers and the sample size.

**Output:** an intersectional analysis where the counts permit, and a statement where they do not.

**Rationale:** A disparity can be concentrated at the intersection of two characteristics, for
example older patients in a minority group. A marginal analysis of each characteristic can miss it.
Intersectional groups are usually small, so the analysis is often descriptive.

Intersectional groups should be pre-specified where the clinical question warrants them, not
searched exhaustively. An exhaustive search over many intersections is a multiplicity problem with
small counts, and it will produce spurious findings.

**Branch point:** if an intersectional group has fewer than about 50 events, report the count and do
not make a claim. The absence of a claim is not evidence of absence.

### Step 13. Report the subgroup analysis

**Input:** the full analysis.

**Output:** a forest plot, a table with counts and intervals, a calibration plot by group, and a
written interpretation with the mechanism and the trade-off.

**Rationale:** The report should let a reader see the estimates, the uncertainty, the criterion, and
the mechanism. A table of p-values without counts and intervals is not sufficient. The
interpretation should distinguish disparity from bias and should state which criterion was chosen.

The report should include the unadjusted and adjusted analyses if both were done, and should state
which one is primary. A disparity that disappears after adjustment for prevalence may be arithmetic;
a disparity that persists after adjustment is stronger evidence of a model or access problem.

**Branch point:** if the subgroup claim does not meet the pre-specified rule, report it as
exploratory and say what evidence would make it confirmatory.

### Step 14. Monitor after deployment

**Input:** the deployed model and the subgroup data.

**Output:** a monitoring plan with metrics, cadence, triggers, and an owner.

**Rationale:** Subgroup performance can change when the population, the coding, or the workflow
changes. Monitoring should track the criteria that were chosen, with the same definitions and
counts. A trigger should specify the magnitude of change that warrants action.

The monitoring plan should include the label audit, because a change in coding or access can
change the disparity without changing the model. A subgroup metric that moves should trigger a
review of the label and the workflow before a model update.

**Branch point:** if a subgroup's performance falls below the pre-specified threshold, the model
should be reviewed, not silently retrained. The review should include the mechanism and the label.

### Fairness edge cases

**A subgroup with no events.** Sensitivity and PPV are undefined. Report the count and avoid the
metric.

**A subgroup with 14 events.** The interval is wide enough to include almost any value. Report the
interval and do not claim a disparity.

**A prevalence difference.** PPV differs even when sensitivity and specificity are equal. Report
the prevalence and the PPV at equal sensitivity.

**A proxy for a protected attribute.** The model may use the proxy even if the attribute is
excluded. Check the correlation and the subgroup performance.

**A label that encodes access.** Cost or utilisation labels can produce a disparity that no model
change can fix. Audit the label.

**A missing feature by group.** A lab measured less often in one group creates a feature
availability disparity. Report the missingness by group.

**An intersectional group with few events.** Report descriptively and avoid a claim.

**A post-hoc subgroup.** Label it exploratory and do not present it as confirmatory.

**A mitigation that worsens another criterion.** Report the trade-off; do not hide it.

**A subgroup defined by a clinical variable that is also a proxy.** A comorbidity index can proxy
for access; the disparity may be in the index, not the group. Check the correlation.

**A protected attribute that is not in the data.** The absence of the variable does not mean the
model is unaffected; proxies can carry the association. Audit the proxies.

## 6. Worked example

### The setting

A readmission model is deployed across three groups. Group A has 6,000 patients and 8 percent
prevalence; group B has 3,000 patients and 18 percent prevalence; group C has 1,000 patients and 22
percent prevalence. The overall prevalence is 12 percent. At a threshold of 0.20, the model has
sensitivity 0.65 and specificity 0.85 in every group.

### The within-group metrics

| Group | n | Prevalence | Sensitivity | Specificity | PPV |
|---|---:|---:|---:|---:|---:|
| A | 6,000 | 8% | 0.65 | 0.85 | 0.274 |
| B | 3,000 | 18% | 0.65 | 0.85 | 0.487 |
| C | 1,000 | 22% | 0.65 | 0.85 | 0.550 |

The PPV for A is:

```text
0.65 * 0.08 / (0.65 * 0.08 + 0.15 * 0.92)
= 0.052 / (0.052 + 0.138)
= 0.274
```

For C:

```text
0.65 * 0.22 / (0.65 * 0.22 + 0.15 * 0.78)
= 0.143 / (0.143 + 0.117)
= 0.550
```

The model has the same sensitivity and specificity in every group, and the PPV differs by a factor
of two. The difference is entirely due to prevalence. A report that calls this "model bias" has not
done the arithmetic.

### The impossibility arithmetic

Suppose the team wants predictive parity at PPV 0.30. The sensitivity required in each group, at
specificity 0.85, is:

```text
sens = PPV * (1 - spec) * (1 - p) / (p * (1 - PPV))
```

For A:

```text
0.30 * 0.15 * 0.92 / (0.08 * 0.70)
= 0.0414 / 0.056
= 0.739
```

For B:

```text
0.30 * 0.15 * 0.82 / (0.18 * 0.70)
= 0.0369 / 0.126
= 0.293
```

For C:

```text
0.30 * 0.15 * 0.78 / (0.22 * 0.70)
= 0.0351 / 0.154
= 0.228
```

Equal PPV requires sensitivity 0.74 in A, 0.29 in B, and 0.23 in C. The group with the lowest
prevalence must be flagged much more aggressively to achieve the same precision. That is the
impossibility in one table: equal PPV and equal sensitivity cannot both hold.

If the team instead wants PPV 0.40, the required sensitivity in A is `0.40 * 0.15 * 0.92 / (0.08 *
0.60) = 1.15`, which is impossible. The maximum PPV attainable in A at specificity 0.85 is
`0.08 / (0.08 + 0.138) = 0.367`, achieved only at sensitivity 1.0. Predictive parity at 0.40 cannot
be achieved in A without changing the specificity or the population.

### The criterion choice

The team chooses equal opportunity as the primary criterion, because the programme's harm is
missing a patient who would benefit, and predictive parity as a secondary criterion, because a
flagged patient should not be a false alarm more often in one group. The two conflict, as the
arithmetic shows. The report states the conflict and the reason for the primary choice.

### The sample-size problem

Group C has 1,000 patients and 220 events. Its AUROC is 0.64 with a 95 percent interval of about
0.60 to 0.68. A smaller group D, defined post hoc, has 120 patients and 14 events; its AUROC is 0.45
with an interval of 0.28 to 0.62. The point estimate looks alarming, but the interval includes 0.62,
which is better than the overall model. The difference is not established.

The multiplicity problem is worse. If the team tests 12 subgroups at the 5 percent level, the
probability of at least one false positive is `1 - 0.95^12 = 0.46`. Nearly half of such analyses
will produce a spurious "significant" subgroup. The pre-specified rule is the defence.

### The mechanism assessment

The team examines the label first. Readmission is defined from claims, and claims capture
readmissions to the index hospital and its partners. Patients in group A are more likely to be
readmitted outside the network, so the label under-ascertains their events. That is a label
mechanism, not a model mechanism, and it would bias the model against A. The team also finds that a
key laboratory is measured less often in A, which is a feature-availability mechanism. Both are
reported before any mitigation is proposed.

### Two diagnostics that would reveal a broken fairness analysis

**Diagnostic 1: PPV difference without prevalence.** The PPV difference between A and C is 0.276, but
the prevalence difference is 14 percentage points. A report that presents the PPV difference as
evidence of bias without the prevalence table has not separated the arithmetic from the model. The
diagnostic is a table with prevalence, sensitivity, specificity, and PPV side by side.

**Diagnostic 2: a subgroup claim from 14 events.** The post-hoc group D has an AUROC of 0.45, but
its interval is 0.28 to 0.62 and the group was defined after seeing the results. A report that
highlights D as a failure has over-read the noise. The diagnostic is the count, the interval, and
the pre-specification status.

### The mitigation decision

The team considers three mitigations: a group-specific threshold for A, a label change to capture
out-of-network readmissions, and a feature-availability fix. The threshold is the only one that can
be implemented immediately, but it treats the symptom, not the mechanism. The label change is the
most likely to address the root cause, but it requires data access and revalidation. The
feature-availability fix requires a workflow change at the sites serving A. The report presents all
three, with the criterion each addresses and the evidence each requires.

### Varying one assumption: prevalence in group A falls to 4 percent

At 4 percent prevalence, the PPV at sensitivity 0.65 and specificity 0.85 is:

```text
0.65 * 0.04 / (0.65 * 0.04 + 0.15 * 0.96)
= 0.026 / (0.026 + 0.144)
= 0.153
```

The PPV falls from 0.274 to 0.153. The model has not changed; the base rate has. The disparity
between A and C widens, and the case for a group-specific threshold in A becomes stronger. The
conclusion changes from "the PPV difference is largely prevalence" to "the PPV in A is low enough
that the programme may not be viable without a different threshold or a different intervention".

### The report's bottom line

The report's bottom line is: "The model's sensitivity and specificity are similar across groups, but
PPV is much lower in group A because its prevalence is lower. Equal PPV would require a sensitivity
of 0.74 in A versus 0.23 in C, which is not achievable at the current specificity. The label
under-ascertains readmissions in A, and that is the mechanism to address before a threshold change."
That paragraph separates the arithmetic, the trade-off, and the mechanism.

### Reading the table as a committee

A committee reading the table should ask three questions. Is the sensitivity difference larger than
its interval? In this example, the sensitivity is the same by construction, so the answer is no. Is
the PPV difference explained by prevalence? The table shows that it is. Is the low PPV in A a problem
for the programme? At 0.274, about one in four flagged patients in A has the event, which may be
acceptable for a low-cost intervention; at 0.153 under the 4 percent scenario, it may not be.

The committee should also ask what the model does not show. The table does not show whether the
label captures the same events in each group, whether the features are measured the same way, or
whether the groups have the same access to the intervention. Those questions are answered by the
audits, not by the performance table.

A final question is who bears the errors. In group A, most flagged patients are false positives, so
the burden of the programme falls on people who would not have been readmitted. In group C, the
flagged patients are more likely to be true positives, but the group has a higher prevalence and may
have a greater need. The distribution of burden is a policy question that the table informs but does
not settle.

### What to report

The primary figure is a forest plot of AUROC by subgroup with confidence intervals and counts. The
secondary figure is a calibration plot by group. The table of prevalence, sensitivity, specificity,
PPV, and the equal-PPV sensitivity requirement belongs in the main text, because it is the evidence
for the trade-off. The report should state the chosen criterion, the pre-specified claim rule, the
mechanism assessment, and the limitations of the small groups.

## 7. Code, query, or calculation

The script computes the PPV at equal sensitivity, the maximum attainable PPV, and the sensitivity
required for a target PPV, for three groups with different prevalence. It also computes the
multiplicity probability for a set of subgroup tests.

```python
"""Subgroup PPV arithmetic and the fairness trade-off. Standard library only."""


def ppv(sensitivity, specificity, prevalence):
    return (sensitivity * prevalence
            / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence)))


def max_ppv(specificity, prevalence):
    return ppv(1.0, specificity, prevalence)


def required_sensitivity(target_ppv, specificity, prevalence):
    return (target_ppv * (1 - specificity) * (1 - prevalence)
            / (prevalence * (1 - target_ppv)))


groups = {"A": (0.08, 6000), "B": (0.18, 3000), "C": (0.22, 1000)}
sensitivity, specificity = 0.65, 0.85
target = 0.30

print("group  prevalence  ppv_equal_sens  max_ppv  sens_for_target")
for name, (prevalence, n) in groups.items():
    print(f"{name}      {prevalence:.2f}        {ppv(sensitivity, specificity, prevalence):.3f}     "
          f"{max_ppv(specificity, prevalence):.3f}    "
          f"{required_sensitivity(target, specificity, prevalence):.3f}")

print(f"A at 4% prevalence: {ppv(sensitivity, specificity, 0.04):.3f}")

subgroups = 12
alpha = 0.05
print(f"P(at least one false positive in {subgroups} tests) = "
      f"{1 - (1 - alpha) ** subgroups:.3f}")
```

Expected output:

```text
group  prevalence  ppv_equal_sens  max_ppv  sens_for_target
A      0.08        0.274     0.367    0.739
B      0.18        0.487     0.594    0.293
C      0.22        0.550     0.653    0.228
A at 4% prevalence: 0.153
P(at least one false positive in 12 tests) = 0.460
```

Reading the output:

- At equal sensitivity and specificity, PPV ranges from 0.274 in A to 0.550 in C. The difference is
  prevalence, not model behaviour.
- The maximum PPV in A is 0.367, achieved only at sensitivity 1.0. A target of 0.40 is impossible in
  A without changing the specificity or the population.
- Equal PPV at 0.30 requires sensitivity 0.739 in A, 0.293 in B, and 0.228 in C. The group with the
  lowest prevalence must be flagged far more aggressively, which is the impossibility in practice.
- At 4 percent prevalence, A's PPV falls to 0.153. The same model becomes much less precise.
- Twelve independent tests at the 5 percent level have a 46 percent chance of at least one false
  positive. The pre-specified claim rule is the defence.

The functions are the reusable part. In a real analysis, the inputs come from the subgroup
confusion matrices with intervals, and the trade-off table is reported alongside them.

## 8. Data, standards, and regulatory context

Fairness is addressed by reporting guidelines, regulatory expectations, and civil-rights law. The
table distinguishes binding requirements from conventions.

| Instrument | Status | What it says about fairness | What it does not do |
|---|---|---|---|
| TRIPOD+AI | Reporting convention | Requires fairness considerations and subgroup performance to be reported | Does not define a fairness criterion |
| PROBAST+AI | Appraisal tool | Assesses whether subgroup performance was evaluated | Does not prescribe a method |
| FDA GMLP principles | Guidance, not binding | Calls for representative data and testing across the intended population | Does not define fairness |
| EU AI Act | Binding in the EU, phased | Requires data governance, bias monitoring, and technical documentation for high-risk systems | Does not specify a criterion |
| EU MDR | Binding in the EU | Requires clinical evaluation across the intended population | Does not define fairness |
| NIST AI RMF | Voluntary framework | Calls for bias measurement and management | Not sector-specific |
| US civil-rights law | Binding in the US | Prohibits discrimination in federally assisted programmes | Does not define algorithmic fairness |
| ISO/IEC 42001 | Voluntary standard | AI management systems | Does not specify metrics |

Three points matter for practice.

**TRIPOD+AI expects subgroup reporting.** The checklist includes fairness considerations and
subgroup performance. A report that gives only the overall AUROC has not met the item in substance.
The report should state the subgroups, the counts, the intervals, and the criterion.

**The EU AI Act treats bias monitoring as a lifecycle obligation.** For high-risk systems, the
provider must have data governance and testing that address bias, and must monitor after deployment.
The Act does not prescribe a fairness criterion, so the provider must justify the one it uses.

**Civil-rights law is separate from statistical fairness.** A model that satisfies a statistical
criterion can still be unlawful if it produces disparate impact without a justified necessity. The
legal analysis is outside this module, and the report should not claim legal compliance on the basis
of a fairness metric.

### A fairness reporting checklist

A fairness section that can be audited contains the following.

```text
subgroups and rationale, pre-specified
prevalence, counts, and missingness by group
chosen criterion and the reason
sensitivity, specificity, PPV, and AUROC by group with intervals
calibration by group
equal-sensitivity PPV and required sensitivity for equal PPV
interaction test or difference with interval
mechanism assessment: label, features, access, model
mitigation and its held-out effect
trade-offs across criteria
monitoring plan and triggers
```

The checklist maps onto the TRIPOD+AI fairness items and the PROBAST+AI analysis domain. A table of
subgroup AUROCs without prevalence and counts is not a fairness analysis.

### The fairness section and the model card

The model card's fairness section should state the subgroups, the criterion, the prevalence and
counts, the performance with intervals, the mechanism, the mitigation, and the monitoring. It should
also state what the analysis did not cover, such as groups not in the data or intersections too small
to analyze. A card that reports a single "fairness score" has compressed away the trade-off that the
reader needs.

> **Note:** No regulation defines a single correct fairness criterion. The requirement is that the
> criterion be chosen, justified, measured, and monitored.

## 9. Industry practice and case evidence

**The cost proxy case.** Obermeyer et al. (2019) studied a commercial population-health algorithm
that used health care costs as a proxy for need. At the same predicted risk score, Black patients
had more chronic conditions and worse biomarkers than White patients. The authors estimated that
replacing the cost label with a health-based label would increase the proportion of Black patients
identified for additional care from 17.7 percent to 46.5 percent. The mechanism was the label, not
the model class.

**Race correction in clinical algorithms.** Vyas et al. (2020) documented race-based correction
factors in several clinical algorithms, including the eGFR equation, and argued that race is often a
proxy for racism and social disadvantage rather than a biological variable. The case shows that a
feature can encode a structural inequity even when it is clinically conventional.

**The impossibility results.** Chouldechova (2017) proved that predictive parity and equal error
rates are incompatible when prevalence differs. Kleinberg, Mullainathan, and Raghavan (2016) proved
that calibration within groups, balance for the positive class, and balance for the negative class
can hold together only under perfect prediction or equal base rates. The results are mathematical,
not empirical, and they apply to any imperfect model.

**Subgroup claims in trials.** Wallach et al. found that of 117 subgroup claims in randomized trial
abstracts, only 46 had a significant interaction test, and of the five claims that were subsequently
tested, none was reproduced. Brookes et al. (2004) showed that a trial powered at 80 percent for an
overall effect has only about 29 percent power for an interaction of the same magnitude. The
practical lesson is that most subgroup claims are underpowered and many are false.

**Sepsis model subgroups.** The external validation of the Epic Sepsis Model reported overall
performance but the clinical concern was concentrated in the alert burden and the missed cases. The
case illustrates why subgroup performance and alert burden should be reported together: an overall
AUROC can hide a group whose alerts are mostly false.

**How a team should use the evidence.** A team should start with the label audit, because the
Obermeyer case shows that the label can be the mechanism. It should then describe the groups and
their prevalence, because the arithmetic of PPV depends on prevalence. It should choose a criterion
and state the trade-off, because the impossibility result means no criterion is free. It should
examine the features and access, because a model can encode a structural difference through a
measurable variable. Finally, it should pre-specify the claim rule, because most subgroup findings
do not replicate.

The order matters: a team that starts with a fairness-constrained model and no label audit may
optimise a metric while the mechanism remains. A label change can be more effective than a
threshold change, and it is often cheaper than a new model. A team that starts with the label and
the prevalence will often find that the disparity is arithmetic or documentation, and the model
change is secondary. A team that starts with the label and the prevalence will often find
that the disparity is arithmetic or documentation, and the model change is secondary.

The three audits can be summarised in a table that a clinical committee can read.

| Audit | Question | Evidence | If positive |
|---|---|---|---|
| Label | Is the outcome measured the same way? | Capture, coding, chart review by group | Change the label |
| Feature | Are the inputs measured the same way? | Missingness, units, testing by group | Fix the data or the workflow |
| Access | Are the groups treated in the same system? | Site and clinician distribution | Adjust for site or validate locally |

A positive audit is not proof of bias, but it is a mechanism that a model change cannot fix. The
report should present the audits before the mitigation, because the mitigation depends on which
audit is positive.

**The label audit in practice.** A label audit asks whether the outcome is measured the same way
for every group. For readmission, it compares the proportion of readmissions captured by the
network, the coding completeness, and the out-of-network capture by group. For cost, it compares
spending at the same health status. For diagnosis codes, it compares the sensitivity of the code
against chart review by group. The audit is a data exercise, not a model exercise, and it should
precede any fairness-constrained modelling.

**The feature audit in practice.** A feature audit asks whether the inputs are measured the same
way. It compares missingness, units, and testing rates by group. A laboratory measured less often in
one group will produce a model that is less accurate for that group, even if the model is fair by a
parity criterion. The audit is a missingness table by group, and it is often the most informative
single table in the analysis.

**The access audit in practice.** An access audit asks whether the groups are treated at different
sites, by different clinicians, or under different workflows. A model deployed across sites can
inherit the site differences. The audit compares the site distribution by group and the performance
by site.

**The role of the affected community.** A fairness analysis that does not involve the affected
groups can choose the wrong criterion or the wrong mitigation. Community input is not a statistical
method, but it is part of the evidence that the analysis is legitimate. The report should state who
was consulted and what they said.

**The three questions a committee should ask.** First, what is the criterion, and why was it
chosen? Second, what is the prevalence in each group, and does the PPV difference persist after
accounting for it? Third, what is the mechanism, and does the mitigation address it? A committee
that asks these three questions will not be satisfied by a table of p-values.

**What the evidence does not show.** The empirical literature documents disparities and the
mathematical literature proves the impossibility, but there is little evidence that a particular
fairness-constrained model improves health outcomes. The report should distinguish the evidence for
the disparity from the evidence for the remedy. A mitigation that equalizes a metric may not improve
the lived outcome, and the claim should be limited accordingly.

**Industry practice.** Many model cards now include a subgroup performance table, but few include
the prevalence, the counts, the intervals, and the criterion. A table without those is a gesture,
not an analysis. The practice is improving, and the standard is the pre-specified, interval-based,
mechanism-oriented analysis in this topic.

> **Note:** The cases are documented. The claim that a specific model is unlawful is not made here;
> the legal analysis requires facts and counsel outside this module.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Overall metric only | No subgroup table | Heterogeneity hidden | Harm in a group | Subgroup analysis |
| PPV difference called bias | Prevalence not reported | Base-rate neglect | Wrong remedy | Report prevalence |
| Subgroup claim from 14 events | Wide interval ignored | Small-sample noise | False finding | Minimum event rule |
| Many subgroups tested | No correction | Multiplicity | Spurious finding | Pre-specify and correct |
| Within-group p-values compared | One significant, one not | Incorrect test | False interaction | Interaction test |
| Protected attribute excluded and forgotten | No proxy audit | Proxy carries association | Indirect bias | Proxy analysis |
| Label proxy unexamined | Cost or utilisation label | Label encodes access | Structural bias | Label audit |
| Feature missing by group | Differential missingness | Measurement bias | Unequal performance | Missingness table |
| Calibration not checked by group | Overall slope only | Group miscalibration | Under-service | Group calibration |
| Threshold changed without disclosure | Group-specific cut-off | Policy choice hidden | Perceived unfairness | Disclose and justify |
| Mitigation fitted in-sample | It looks perfect | Overfitting | Fails at deployment | Held-out check |
| Intersectional claim from tiny group | Few events | Noise | False finding | Descriptive only |
| Monitoring absent | No post-launch check | Drift | Silent disparity | Monitoring plan |
| Fairness claim without a criterion | "The model is fair" | Undefined claim | Unfalsifiable | Name the criterion |
| Calibration by group omitted | Overall calibration only | Group miscalibration | Under-service | Group calibration |
| Intersectional claim from a tiny group | Few events | Noise | False finding | Descriptive only |
| Mitigation without held-out check | In-sample improvement | Overfitting | Fails at deployment | Independent check |
| Criterion chosen after results | The reported criterion is the one that looks best | Selection | Overstated fairness | Pre-specify |
| Group definition changed after results | The group boundary moves | Selection | False finding | Pre-specify |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The subgroup identifier is measured correctly | Groups are comparable | Misclassification | Data audit |
| The subgroup is large enough | Estimates are stable | Few events | Count and interval |
| The prevalence is estimated within the group | PPV interpretation | Small group | Report prevalence |
| The criterion matches the decision | The analysis is relevant | Wrong criterion | Stakeholder input |
| The label is comparable across groups | Performance is comparable | Differential ascertainment | Label audit |
| The features are measured comparably | Model inputs are comparable | Differential availability | Missingness table |
| The mitigation targets the mechanism | It can work | Symptom-only fix | Mechanism assessment |
| The held-out sample is independent | The mitigation is validated | In-sample fitting | Split audit |
| The monitoring captures the outcome | Drift is detectable | Delayed outcomes | Proxy and schedule |
| The legal analysis is separate | Statistical fairness is not law | Legal claim | Counsel review |
| The subgroup definition is stable | Groups are comparable over time | Reclassification | Monitor definitions |
| The criterion is acceptable to stakeholders | The analysis is legitimate | Disagreement | Engagement |
| The mitigation is implementable | The plan is real | Workflow constraints | Feasibility study |
| The criterion is reviewed independently | The analysis is legitimate | Owner bias | Independent review |
| The affected groups are consulted | The decision is acceptable | No input | Engagement |

The deepest limit is that fairness is a value choice, not a statistical fact. The data can tell the
team what the trade-offs are; it cannot tell them which trade-off is right. The report should
present the trade-off and the mechanism, and the decision-maker should choose with that evidence.

## 12. Exercises

The exercises move from arithmetic to design. Exercises 1 and 2 can be done with a calculator; 3
through 7 require a model or a paper; 8 produces the fairness section for the module assignment.

**Exercise 1. Subgroup metrics.** For three groups with prevalence 5, 12, and 20 percent and a model
with sensitivity 0.60 and specificity 0.88, compute PPV for each. Deliverable: a table.
Acceptance criteria: values correct; prevalence stated. Time: 30 minutes.

**Exercise 2. Impossibility arithmetic.** For the groups in Exercise 1, compute the sensitivity
required for PPV 0.35 and the maximum attainable PPV. Deliverable: a table and a sentence.
Acceptance criteria: values correct; impossibility identified where it occurs. Time: 45 minutes.

**Exercise 3. Interaction test.** Given subgroup estimates and intervals, state whether an
interaction is established and why a within-group p-value comparison is not enough. Deliverable: a
400-word analysis. Acceptance criteria: interaction logic correct; no claim from overlapping
intervals. Time: 45 minutes.

**Exercise 4. Pre-specification rule.** Write a rule for credible subgroup claims in your project.
Deliverable: a one-page rule. Acceptance criteria: minimum events, test, effect size, and replication
all specified. Time: 45 minutes.

**Exercise 5. Mechanism audit.** For a model you know, assess the label, features, access, and model
as mechanisms of disparity. Deliverable: a table. Acceptance criteria: all four mechanisms
addressed; evidence cited. Time: 60 minutes.

**Exercise 6. Mitigation comparison.** Compare a group-specific threshold, a label change, and a
feature fix for a disparity you have identified. Deliverable: a memo. Acceptance criteria: each
addresses a mechanism; trade-offs stated. Time: 60 minutes.

**Exercise 7. Small-group critique.** Find a paper that claims a subgroup effect and assess its
counts, interval, and pre-specification. Deliverable: a 500-word critique. Acceptance criteria: at
least three specific issues; no claim beyond the data. Time: 60 minutes.

**Exercise 8. Fairness report.** Write the fairness section of a model development report,
including the subgroup list, the prevalence and counts, the criterion, the performance with
intervals, the trade-off table, the mechanism audit, the mitigation, and the monitoring plan.
Deliverable: two pages. Acceptance criteria: all seven elements present; no claim from a group with
fewer than 50 events. Time: 75 minutes.

A learner who completes all eight exercises has the material for the fairness section of the module
assignment. The arithmetic exercises can be checked against Section 13; the design exercises are
assessed with the rubric. Write the fairness section of a model development report.
Deliverable: two pages. Acceptance criteria: criterion, counts, intervals, mechanism, trade-off, and
monitoring all present. Time: 75 minutes.

## 13. Self-check answers

**Answer to Exercise 1.** At 5 percent: `0.60 * 0.05 / (0.60 * 0.05 + 0.12 * 0.95) = 0.030 / 0.144
= 0.208`. At 12 percent: `0.072 / (0.072 + 0.1056) = 0.405`. At 20 percent: `0.120 / (0.120 +
0.096) = 0.556`. The PPV rises with prevalence at constant sensitivity and specificity.

**Answer to Exercise 2.** Required sensitivity for PPV 0.35: at 5 percent, `0.35 * 0.12 * 0.95 /
(0.05 * 0.65) = 0.0399 / 0.0325 = 1.228`, impossible. At 12 percent, `0.35 * 0.12 * 0.88 / (0.12 *
0.65) = 0.03696 / 0.078 = 0.474`. At 20 percent, `0.35 * 0.12 * 0.80 / (0.20 * 0.65) = 0.0336 /
0.130 = 0.258`. The maximum PPV at 5 percent is `0.05 / (0.05 + 0.114) = 0.305`, so 0.35 is
unattainable there at specificity 0.88.

**Answer to Exercise 8 (what a complete fairness section contains).** A complete section includes
the pre-specified groups and rationale, the prevalence and counts, the criterion and its
justification, the performance with intervals, the calibration by group, the trade-off table, the
mechanism audit, the mitigation and its held-out effect, and the monitoring plan. It also states the
limitations, including groups not analyzed and the legal analysis that is outside scope.

**Answer to Exercise 3 (what a strong analysis shows).** A strong analysis reports the subgroup
estimates, the intervals, the difference with its interval, and the interaction test. It states
whether the difference is established, not merely whether one group's estimate is significant. It
also states the power limitation if the interaction test is underpowered.

**Answer to Exercise 1 (interpretation).** The PPV rises from 0.208 to 0.556 as prevalence rises
from 5 to 20 percent, with the same sensitivity and specificity. A team that compares the PPVs
without the prevalence will conclude that the model is much better in the high-prevalence group,
when the model is identical. The comparison is arithmetic, not evidence of differential model
quality.

**Answer to Exercise 2 (interpretation).** The required sensitivity exceeds 1 in the lowest-
prevalence group, which means the target PPV is unattainable at the current specificity. The correct
conclusion is not that the model is unfair but that the target is infeasible; the team must change
the specificity, the threshold, or the population, or accept a lower PPV in that group.

**Answer to Exercise 5 (what a strong mechanism table shows).** A strong table has one row per
mechanism, the evidence for or against it, the group affected, and the mitigation it implies. It
distinguishes a label mechanism from a feature mechanism and from a model mechanism, because the
remedies differ. It does not assert a mechanism without evidence.

**Answer to Exercise 6 (what a strong mitigation memo contains).** A strong memo states the
mechanism, the mitigation that targets it, the criterion it is expected to improve, the expected
effect on the other criteria, the evidence required, and the implementation cost. It distinguishes a
symptom fix (a group-specific threshold) from a mechanism fix (a label change or a feature-
availability change). It recommends a sequence, not a single action.

**Answer to Exercise 4 (what a strong rule contains).** A strong rule specifies the minimum number
of events and non-events per group, the interaction test and its alpha, the minimum absolute
difference in the metric, the requirement for replication in an external cohort, and the status of
exploratory findings. It also states what the team will not claim if the rule is not met. The rule
is written before the analysis and is not changed after seeing the results.

**Answer to Exercise 3.** An interaction is established if the difference between the subgroup
estimates is larger than its interval and the pre-specified interaction test is significant. Two
within-group p-values, one below and one above 0.05, do not establish an interaction, because the
difference between them has its own uncertainty. Overlapping intervals are consistent with a real
difference, and non-overlapping intervals are not required for one.

**Answer to Exercise 7 (what a strong critique contains).** A strong critique reports the subgroup
size and event count, the interval, the pre-specification status, the interaction test, and whether
the claim was replicated. It distinguishes a hypothesis from a finding, and it states what evidence
would make the claim credible. A critique that says "the subgroup is too small" without the counts
is not specific.

**Rubric for Exercises 3, 4, 5, 6, 7, and 8.** A strong submission (1) reports counts and
prevalence, (2) uses intervals and an interaction test, (3) states the criterion and the trade-off,
(4) examines the mechanism, (5) pre-specifies the claim, and (6) monitors after deployment. A
competent submission meets four of the six. A weak submission calls a PPV difference bias without
prevalence or claims a subgroup effect from a handful of events. A strong submission (1) reports counts and
prevalence, (2) uses intervals and an interaction test, (3) states the criterion and the trade-off,
(4) examines the mechanism, (5) pre-specifies the claim, and (6) monitors after deployment. A
competent submission meets four of the six. A weak submission calls a PPV difference bias without
prevalence or claims a subgroup effect from a handful of events.

## 14. Interview questions

**1. Why can PPV differ across groups with the same model?** A strong answer derives the Bayes
relation and shows the prevalence dependence. A weak answer says the model is biased.

**2. What is the fairness impossibility result?** A strong answer states that calibration,
equalized odds, and predictive parity cannot all hold when prevalence differs, and gives the
intuition. A weak answer says fairness is impossible.

**3. Which fairness criterion would you choose?** A strong answer ties the criterion to the harm and
the decision, and states the trade-off. A weak answer says equal accuracy.

**4. How large must a subgroup be?** A strong answer gives a minimum event count and an interval,
and notes that the requirement depends on the precision needed. A weak answer says any subgroup
works.

**5. Why is comparing within-group p-values wrong?** A strong answer explains that the difference
needs its own test, not two separate tests. A weak answer says it is a multiple-comparison issue
only.

**6. What is label bias?** A strong answer gives the cost-proxy example and explains that no model
change fixes it. A weak answer says labels are objective.

**7. How do you distinguish disparity from bias?** A strong answer separates prevalence, label,
feature availability, access, and model form. A weak answer equates the two.

**8. What would you monitor after deployment?** A strong answer monitors subgroup performance,
calibration, and the label, with triggers. A weak answer monitors overall AUROC.

**9. Why is a group-specific threshold controversial?** A strong answer explains that it can
equalize an outcome but treats people differently, and the choice is a policy decision. A weak
answer says it is always fair or always unfair.

**10. How do you audit a label for bias?** A strong answer compares ascertainment, coding, and out-of-network capture by group, and validates against chart review. A weak answer says the label is objective.

**11. What is the difference between disparity and bias?** A strong answer says disparity is a difference and bias is a systematic error contributing to it, and that one can exist without the other. A weak answer equates them.

**12. What would make you reject a subgroup claim?** A strong answer cites few events, a wide
interval, post-hoc definition, and no interaction test. A weak answer accepts any difference.

## 15. Key terms

- **Base rate** — prevalence in a group.
- **Calibration within groups** — predicted risk matches observed risk in each group.
- **Demographic parity** — equal positive prediction rates.
- **Disparate impact** — a neutral rule with unequal effects.
- **Disparity** — a difference in performance or outcome across groups.
- **Equal opportunity** — equal true positive rates.
- **Equalized odds** — equal TPR and FPR.
- **Fairness criterion** — a formal condition on predictions across groups.
- **Feature availability bias** — differential missingness across groups.
- **Feedback loop** — the model changes the data it learns from.
- **Forest plot** — a plot of subgroup estimates with intervals.
- **Health equity** — fair opportunity to achieve health.
- **Interaction test** — a test of whether an effect differs across groups.
- **Intersectionality** — overlapping group identities.
- **Label bias** — a label that encodes unequal access or treatment.
- **Measurement bias** — a feature measured differently across groups.
- **Minimum event count** — events needed for a stable estimate.
- **Mitigation** — a change to reduce a disparity.
- **Multiplicity** — many tests increase false positives.
- **Overall accuracy parity** — equal accuracy across groups.
- **Post-hoc subgroup** — a subgroup defined after seeing results.
- **Predictive parity** — equal PPV across groups.
- **Pre-specification** — defining subgroups and claims before analysis.
- **Prevalence difference** — difference in outcome rate across groups.
- **Protected attribute** — a characteristic protected by law or policy.
- **Proxy variable** — a variable correlated with a protected attribute.
- **Race as a social construct** — race is not a biological category.
- **Redlining** — excluding a group through proxies.
- **Sample-size bias** — a subgroup estimate driven by few events.
- **Statistical parity** — synonym for demographic parity.
- **Structural racism** — systemic patterns that produce inequity.
- **Subgroup** — a subset defined by a characteristic.
- **Threshold by group** — a group-specific operating point.
- **Treatment parity** — equal allocation of the action.

## 16. Sources

All URLs accessed 25 October 2026.

1. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to
   manage the health of populations. *Science* 2019;366(6464):447-453.
   https://doi.org/10.1126/science.aax2342
2. Chouldechova A. Fair prediction with disparate impact: a study of bias in recidivism prediction
   instruments. *Big Data* 2017;5(2):153-163. https://doi.org/10.1089/big.2016.0047
3. Kleinberg J, Mullainathan S, Raghavan M. Inherent trade-offs in the fair determination of risk
   scores. arXiv:1609.05807, 2016. https://arxiv.org/abs/1609.05807
4. Vyas DA, Eisenstein LG, Jones DS. Hidden in plain sight — reconsidering the use of race
   correction in clinical algorithms. *New England Journal of Medicine* 2020;383(9):874-882.
   https://doi.org/10.1056/NEJMms2004740
5. Brookes ST, Whitely E, Egger M, et al. Subgroup analyses in randomized trials: risks of
   subgroup-specific analyses; power and sample size for the interaction test. *Journal of Clinical
   Epidemiology* 2004;57(3):229-236. https://doi.org/10.1016/j.jclinepi.2003.08.009
6. Wang R, Lagakos SW, Ware JH, Hunter DJ, Drazen JM. Statistics in medicine — reporting of
   subgroup analyses in clinical trials. *New England Journal of Medicine* 2007;357(21):2189-2194.
   https://doi.org/10.1056/NEJMsr077003
7. Wallach JD, Sullivan PG, Trepanowski JF, et al. Evaluation of evidence of statistical support
   and corroboration of subgroup claims in randomized clinical trials. *JAMA Internal Medicine*
   2017;177(4):554-560. https://pmc.ncbi.nlm.nih.gov/articles/PMC6657347/
8. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
9. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
   https://doi.org/10.1136/bmj-2024-082505
10. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
    Guiding Principles. 2021.
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
11. European Union. Regulation (EU) 2024/1689 (Artificial Intelligence Act). 2024.
    https://eur-lex.europa.eu/eli/reg/2024/1689/oj
12. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
13. Hardt M, Price E, Srebro N. Equality of opportunity in supervised learning. *Advances in Neural
    Information Processing Systems* 29, 2016. https://arxiv.org/abs/1610.02413
14. Fairlearn developers. Fairlearn documentation. https://fairlearn.org/
15. US Department of Health and Human Services. Section 1557 of the Affordable Care Act.
    https://www.hhs.gov/civil-rights/for-individuals/section-1557/index.html
16. National Institute for Health and Care Excellence. Evidence standards framework for digital
    health technologies. https://www.nice.org.uk/about/what-we-do/our-programmes/evidence-standards-framework-for-digital-health-technologies
17. World Health Organization. Ethics and governance of artificial intelligence for health. 2021.
    https://www.who.int/publications/i/item/9789240029200
18. McCradden MD, Joshi S, Mazwi M, Anderson JA. Ethical limitations of algorithmic fairness
    solutions in health care machine learning. *Lancet Digital Health* 2020;2(5):e221-e223.
    https://doi.org/10.1016/S2589-7500(20)30065-0

The fairness analysis is not a separate appendix; it is part of the evidence that the model is fit
for the population it serves. It is also a continuing obligation, because the population and the
label can change after deployment. A one-time analysis is a snapshot, not a guarantee.

The report should end with the monitoring plan: the metrics, the cadence, the triggers, and the
owner. Without that, the fairness section describes the past, not the deployment. A model that is accurate on average and harmful for a group is not a
successful model.

The fairness analysis is not a separate appendix; it is part of the evidence that the model is fit
for the population it serves.

A fairness report is a record of a decision about values, supported by arithmetic and evidence. It
is not a certification, and it does not replace legal review. The report should be written so that a
reader can see the trade-off, disagree with the choice, and still trust the numbers.

The choice among criteria is not a technical problem with a correct answer. It depends on the
harm the programme is willing to impose and on who bears it. It is a policy decision
that the data can inform. The analysis is successful if it makes the decision explicit, not if it
makes it disappear. It
should be written so that a reader can disagree with the decision while accepting the analysis.

The fairness analysis is the last of the evidence topics; the next topic catalogues the ways the
evidence can be contaminated. A model that is fair in a clean analysis and leaky in the pipeline is
not fair in deployment.

A fairness claim without a criterion is a slogan; with one, it is evidence. The evidence is only
as good as the counts and the intervals behind it.

The analysis is a means to a decision, not a verdict.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`11_uncertainty.md`](11_uncertainty.md)
- Next topic: [`13_leakage.md`](13_leakage.md)
- Labels and outcomes: [`02_labels-and-outcomes.md`](02_labels-and-outcomes.md)
- Metrics: [`07_metrics.md`](07_metrics.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)


