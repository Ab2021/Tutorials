# Fairness

> **Module 16 - Privacy, Security, and Ethics**
> Topic 06 of 10
> Estimated study time: 3 hours
> Prerequisites: 04, 05
> Version 2026.3

## 1. Why this topic exists

A model that is accurate on average can be harmful in a subgroup. A risk score that is well
calibrated on cost can under-refer the patients who cost less to treat. A pulse oximeter that meets
its specification can miss hypoxaemia in patients with darker skin. Fairness is the discipline of
asking who the average hides, measuring the answer, and deciding what to do about it before the
system is deployed rather than after harm has occurred.

The commercial stakes are now explicit. A health system that deploys a biased triage model faces
clinical harm, regulatory exposure, and reputational damage; a vendor that cannot produce subgroup
performance data loses procurement reviews; and a payer that uses a cost proxy for need may be
found to discriminate, even if the model never used race as a feature. Fairness is no longer a
research topic appended to a model card; it is a release criterion.

Three failure patterns recur.

**The single-metric defence.** A team reports demographic parity and stops. The model has equal
selection rates, and the subgroup with higher prevalence has a much lower true positive rate, so
the patients who most need the programme are the ones being missed. Parity was achieved by
equalising the wrong quantity. The reverse failure is equally common: a team equalises true
positive rates and breaks predictive parity, so the same score means different things in different
groups.

**The proxy label.** The model predicts cost, utilisation, or appointment attendance because those
are available, and the team treats the prediction as a measure of need. If access to care differs
by group, the proxy encodes the access difference. The model is accurate on its label and wrong
about the thing that matters. This is the mechanism behind the most cited health-equity case in
the field, and it is a design choice, not a data-quality accident.

**Fairness as a one-time audit.** A fairness report is produced before launch, the model is
deployed, and the population changes. Prevalence shifts, a new site joins, a lab changes its assay,
and the subgroup gaps move. Nobody re-measures because the audit was a gate, not a monitoring
process. Fairness is a property of a deployed system over time, not a certificate.

This topic teaches fairness as a measurement and decision problem. It covers the definitions that
matter in biomedical settings, the arithmetic of subgroup metrics, the impossibility results that
make a single "fair" model impossible, the mitigation options and their costs, and the governance
that keeps the measurement alive. It deliberately does not promise a metric that resolves the
trade-offs; the honest position is that the choice depends on the decision, the harm, and who bears
it.

A fourth pattern is **the metric chosen to pass**. A team tries several thresholds and criteria
until one clears an internal review, then reports that one. The model has not changed; the
reporting has. The record of the other criteria is lost, and the reviewers believe the model is
fair because the number they saw was acceptable. The control is to prespecify the criterion and
report all criteria computed, including the ones that fail.

**The governance question is who decides.** A fairness assessment produces a table of trade-offs;
someone has to choose. If the choice is made by the data scientist, it is made on technical
grounds and is unlikely to reflect the clinical or community view. If it is made by a committee
without the numbers, it is made on intuition. The design that works is a named clinical owner, a
documented harm statement, and a review that includes the affected population.

The fairness literature is large, and most of it is about the choice of criterion. The operational
literature is smaller and more useful: how to audit a label, how to size a subgroup analysis, how to
run a review that ends in a decision. This topic draws on both, but the emphasis is operational,
because most failures in practice are not failures of theory. They are failures to measure the
right group, at the right threshold, with the right label.

> **Note:** Fairness metrics are statistical descriptions of a model in a population. They are not
> legal conclusions, and satisfying one metric does not establish compliance with
> anti-discrimination law.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** demographic parity, equal opportunity, equalized odds, predictive parity, and
   calibration within groups from a confusion matrix.
2. **Explain** why these criteria cannot all hold at once when prevalence differs, using the
   arithmetic of base rates.
3. **Audit** a label or proxy variable for bias and **propose** an alternative that measures need
   rather than access.
4. **Choose** a fairness criterion for a stated decision, with the harm and the stakeholder named.
5. **Measure** subgroup performance with confidence intervals and **identify** when a subgroup is
   too small to support a conclusion.
6. **Select** a mitigation (pre-processing, in-processing, post-processing, or governance) and
   **quantify** its trade-off.
7. **Design** a monitoring plan with thresholds, review cadence, and an escalation path.
8. **Critique** a model card or vendor claim that reports only aggregate performance.

## 3. The core mental model

Bias enters a model at every stage of the pipeline, and each stage has a different remedy. The
mental model is the pipeline with the decision at the end, because fairness is a property of the
decision and its consequences, not of the model alone.

```text
  problem definition
        |  who is the decision for? what outcome matters?
        v
  label / proxy  <-- access, documentation, cost, utilisation
        |
        v
  cohort and features  <-- who is included? who is measured?
        |
        v
  model and threshold
        |
        v
  decision  --->  outcome  --->  benefit or harm by group
        ^                              |
        |                              |
        +--------- monitoring ---------+
```

The stages are not independent. A label that measures cost instead of need makes a well-calibrated
model biased against the group that spends less at equal need. A cohort that excludes uninsured
patients makes the model's subgroup performance unknown for the excluded group. A threshold chosen
for overall precision can concentrate false negatives in the group with lower prevalence.

Three models sit on top of the pipeline.

**The metric is a question, not a verdict.** Each fairness criterion asks a different question.
Demographic parity asks whether the selection rate is equal. Equal opportunity asks whether the
true positive rate is equal. Predictive parity asks whether a positive prediction means the same
thing in each group. Calibration asks whether the predicted probability matches the observed rate.
The criteria conflict because they are answers to different questions, not because one is stricter.

**The impossibility result is arithmetic, not politics.** When prevalence differs between groups
and the model has imperfect accuracy, equal true positive rate, equal false positive rate, and
equal positive predictive value cannot all hold. The proof is a base-rate identity, and it is
reproduced in Section 6. The practical consequence is that a team must choose which error matters
more and say so.

**The decision determines the criterion.** A screening programme that can absorb false positives
should prioritise equal opportunity, because a missed case is the harm. A programme that imposes a
burden on those selected (cost, stigma, invasive follow-up) should attend to predictive parity and
to the false positive rate, because a false positive is the harm. The choice is a policy decision
that belongs to the clinical owner, not to the data scientist alone.

```text
  equal TPR  ---- choose when missing a case is the harm
  equal FPR  ---- choose when false alarms impose burden
  equal PPV  ---- choose when the score must mean the same thing
  calibration -- choose when the probability is used in a rule
  parity      ---- choose when representation is the goal
        \       |       /
         \      |      /
          prevalence differs
                 |
                 v
        cannot satisfy all at once
```

The final model is the harm matrix. For each group and each error type, name the consequence. The
metric choice follows from the column with the larger harm, and the trade-off is documented rather
than hidden.

The harm matrix is also a communication tool. A clinical owner who does not follow the arithmetic
of PPV can follow a table that says "a missed case means a patient does not get the programme; a
false positive means a nurse calls someone who does not need it". The metric choice follows from
that sentence, and the sentence is the part the owner can own.

The harm matrix also has a temporal dimension. A false negative today may become a true positive
tomorrow if the patient returns, while a false positive consumes capacity that could have gone to
another patient. The distribution of those effects across groups is part of the fairness analysis,
and it is rarely measured.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Protected attribute | A characteristic protected by law or policy, such as race, sex, age, disability. | Treated as the only source of bias. | Legal category. |
| Sensitive attribute | A characteristic whose use or disclosure is sensitive, whether or not legally protected. | Confused with protected attribute. | Data category. |
| Demographic parity | Equal selection rates across groups. | Treated as the definition of fairness. | Rate ratio. |
| Equal opportunity | Equal true positive rates across groups. | Confused with equalized odds. | Rate difference. |
| Equalized odds | Equal true positive and false positive rates. | Assumed to imply calibration. | Two rate differences. |
| Predictive parity | Equal positive predictive value across groups. | Assumed to imply equal error rates. | Rate difference. |
| Calibration within groups | Predicted probability matches observed frequency in each group. | Checked only overall. | Calibration curve; ECE. |
| Base rate | Prevalence of the outcome in a group. | Ignored in metric comparisons. | Proportion. |
| Selection rate | Proportion of a group receiving a positive decision. | Confused with prevalence. | Proportion. |
| TPR | True positive rate, sensitivity, recall. | Reported without the threshold. | Proportion. |
| FPR | False positive rate. | Ignored when false positives are harmful. | Proportion. |
| PPV | Positive predictive value, precision. | Compared across groups with different prevalence. | Proportion. |
| Accuracy | Proportion correct. | Used as a fairness metric. | Proportion. |
| F1 | Harmonic mean of precision and recall. | Treated as balanced across harms. | Score. |
| Four-fifths rule | A selection rate ratio below 0.80 is a signal of adverse impact in US employment guidance. | Treated as a legal test for all contexts. | Ratio. |
| Disparate impact | A neutral practice with a disproportionately adverse effect on a protected group. | Confused with disparate treatment. | Legal concept. |
| Disparate treatment | Different treatment based on a protected attribute. | Confused with disparate impact. | Legal concept. |
| Label bias | The target variable measures access or documentation rather than need. | Treated as a data-quality issue. | Mechanism. |
| Measurement bias | The instrument measures differently across groups. | Ignored in model evaluation. | Mechanism. |
| Selection bias | The cohort is not representative of the deployment population. | Assumed away by random splits. | Mechanism. |
| Representation bias | Groups are underrepresented in training data. | Treated as sufficient to explain all gaps. | Mechanism. |
| Historical bias | The data reflect past inequity. | Treated as a reason to give up. | Mechanism. |
| Deployment bias | The model is used in a workflow that differs from the training context. | Discovered after launch. | Mechanism. |
| Pre-processing | Mitigation applied to the data before training. | Assumed to be neutral. | Mitigation. |
| In-processing | Mitigation applied during training, such as a fairness constraint. | Assumed to be free. | Mitigation. |
| Post-processing | Mitigation applied to scores or thresholds after training. | Assumed to preserve calibration. | Mitigation. |
| Equalized odds post-processing | Adjusting group-specific thresholds to equalise TPR and FPR. | Confused with equal opportunity. | Mitigation. |
| Intersectionality | Overlapping group memberships that produce distinct harms. | Ignored by single-attribute analysis. | Analysis unit. |
| Subgroup | A defined population within the evaluation. | Defined after seeing results. | Analysis unit. |
| Confidence interval | A range that expresses sampling uncertainty. | Omitted from subgroup claims. | Interval. |
| Model card | A structured description of a model, its data, and its performance. | Treated as a fairness audit. | Documentation. |
| Datasheet | A structured description of a dataset. | Treated as a substitute for evaluation. | Documentation. |
| ECE | Expected calibration error: weighted average calibration gap. | Reported without binning details. | Score. |
| Threshold | The score cut-off that produces a decision. | Treated as a property of the model. | Decision parameter. |

## 5. The mechanics, step by step

The procedure below produces a fairness assessment for a model that supports a decision. It is
written for a team that has a candidate model and needs to decide whether, and how, to deploy it.
The order matters: the population and the criteria are fixed before the metrics are computed, so
the analysis cannot be steered toward a favourable number.

### Step 1. Define the decision, the beneficiaries, and the harm

**Input:** Intended use, clinical workflow, affected population.
**Output:** A decision statement, a beneficiary list, and a harm statement per error type.
**Rationale:** Fairness is about who receives a benefit and who bears a cost. A model that flags
patients for a care-management programme distributes a benefit (enrolment) and a burden (time,
stigma, follow-up). A false negative denies the benefit; a false positive imposes the burden. The
harm statement should say which is worse, for whom, and why.

The beneficiary list should include people who are not in the training data. A model trained on
insured patients and deployed to an uninsured population has unknown performance for the group that
may need it most.

### Step 2. Fix the evaluation population and subgroups before analysis

**Input:** Deployment population, demographic data, clinical context.
**Output:** A prespecified subgroup list with minimum sample sizes and an analysis plan.
**Rationale:** Choosing subgroups after seeing results is a form of p-hacking. The list should be
justified by the harm, the legal context, and the clinical workflow, and it should include at least
the protected attributes relevant in the jurisdiction, plus the intersectional combinations that
matter (for example, older women, or a racial group within a language group).

The minimum sample size should be set in advance. A subgroup with 30 events cannot support a
calibrated conclusion, and reporting a point estimate without an interval invites over-reading.

Branch point: if a subgroup has too few events, the analysis should report the limitation and use
aggregation or qualitative evidence rather than a precise metric.

### Step 3. Audit the label or proxy

**Input:** Target definition, data dictionary, clinical rationale.
**Output:** A label audit: what the label measures, what it omits, and how it could differ by group.
**Rationale:** The label is the most consequential design choice in a clinical model, and it is
usually inherited rather than chosen. Cost, utilisation, appointment attendance, and diagnosis
codes all measure access and documentation as well as need. If access differs by group, the label
encodes the difference.

A label audit asks three questions. Does the label measure the outcome the decision cares about, or
a proxy for it? Does the measurement process treat groups differently? And would a clinician
recognise the label as a valid measure of need? The answers determine whether the model is
optimising the right thing before any fairness metric is meaningful.

### Step 4. Audit measurement and coverage

**Input:** Data sources, instruments, cohort construction.
**Output:** A measurement audit and a coverage map.
**Rationale:** An instrument can measure differently across groups. Pulse oximetry is the canonical
example: at the same arterial oxygen saturation, readings differ by skin pigmentation, so the
feature and the label are both biased. Creatinine-based kidney function estimates have similar
issues, and so do spirometry reference equations and some imaging protocols.

Coverage is the other half. A cohort drawn from one hospital's emergency department may not
represent the population that a community screening programme will reach. The audit should map who
is in the data, who is missing, and why.

### Step 5. Choose the criteria with the clinical owner

**Input:** Harm statement, decision, stakeholder input.
**Output:** A primary criterion and one or two secondary criteria, with the rationale recorded.
**Rationale:** The choice is a policy decision, and it should be made by the person accountable for
the clinical programme, with the affected community represented. The data scientist can explain the
consequences of each criterion; the clinical owner decides which harm is unacceptable.

| Criterion | Question it answers | Choose when | Fails when |
|---|---|---|---|
| Demographic parity | Are selection rates equal? | Representation or access is the goal | Prevalence differs; equal rates can be inequitable |
| Equal opportunity | Are true positive rates equal? | Missing a case is the main harm | False positives impose a serious burden |
| Equalized odds | Are TPR and FPR both equal? | Both error types matter | Rarely achievable with imperfect models and different base rates |
| Predictive parity | Does a positive score mean the same in each group? | The score is used as a probability or a rule | Prevalence differs; equal PPV implies unequal TPR |
| Calibration within groups | Do predicted probabilities match outcomes? | The probability feeds a decision rule | Calibration can coexist with unequal error rates |
| Accuracy parity | Is overall accuracy equal? | Almost never a sufficient criterion | Hides error-type differences |

The table's last row is a warning. Accuracy parity is the metric most often reported and the least
informative, because a model can have equal accuracy and very different error profiles.

Branch point: if the stakeholders cannot agree on a criterion, the fallback is to report all of
them and make the trade-off explicit in the decision record. An unresolved trade-off is better
than a hidden one.

### Step 6. Compute the metrics with uncertainty

**Input:** Model scores, labels, subgroup definitions.
**Output:** A metrics table with confidence intervals and the threshold stated.
**Rationale:** Every metric depends on the threshold, so the threshold must be reported with the
number. Subgroup estimates carry sampling uncertainty, which is larger for small groups. A
difference that is large in point-estimate terms can be within the interval for a small subgroup,
and a difference that looks small can be consistent across sites.

The output should include the confusion matrix per group, not only the derived rates, because the
matrix lets a reviewer recompute every metric and check the arithmetic.

### Step 7. Diagnose the mechanism

**Input:** Metrics table, label audit, measurement audit, coverage map.
**Output:** A mechanism statement per gap: label, measurement, selection, representation,
historical, or deployment.
**Rationale:** The mechanism determines the remedy. A gap caused by a cost proxy is fixed by
changing the label, not by threshold adjustment. A gap caused by an instrument is fixed by changing
the instrument or the feature, not by reweighting. A gap caused by a cohort that excludes a group
is fixed by data collection. Post-processing a gap whose cause is the label hides the cause and
often worsens the harm.

The mechanism statement should be specific enough to test: "the label is annual spending; the
group with lower access spends less at equal need; therefore the model under-ranks that group at
equal need" is testable by comparing need measures at equal score.

### Step 8. Choose a mitigation and quantify the trade-off

**Input:** Mechanism statement, criteria, operational constraints.
**Output:** A mitigation plan with the expected effect on each metric and on overall utility.
**Rationale:** Mitigations act at different stages and have different costs.

| Mitigation | Stage | What it changes | Cost | Risk |
|---|---|---|---|---|
| Relabel or change the target | Data | What the model predicts | Requires a new outcome measure | Label may be unavailable at scale |
| Reweight or resample | Pre-processing | Group representation in training | Can distort the joint distribution | Over-correction; instability |
| Add fairness constraint | In-processing | The optimisation objective | Reduces overall accuracy | Constraint may be infeasible |
| Group-specific thresholds | Post-processing | The decision rule | Changes PPV and calibration | Legally or ethically contested |
| Calibrate per group | Post-processing | The probability scale | Requires group labels at inference | Group labels may be unavailable |
| Change the workflow | Deployment | Who acts on the score | Process change | Requires clinical engagement |
| Monitor and review | Governance | Nothing directly | Ongoing cost | Detection without remedy |

The trade-off should be quantified in the same units as the decision: additional true positives
found, additional false positives generated, additional staff hours required. A mitigation that
improves a metric without a statement of its cost is not a decision, it is a preference.

### Step 9. Test intersectional and small subgroups

**Input:** Metrics by single attribute, subgroup sizes.
**Output:** An intersectional analysis with explicit power limits.
**Rationale:** Harms often concentrate at intersections. A model can be fair by race and fair by
sex and unfair for Black women, because the intersection has a different prevalence and a
different measurement pathway. The analysis should include the intersections that the community
and the clinical team identify, not every combination.

Small subgroups require restraint. A subgroup with 12 events can produce a dramatic rate
difference that is entirely consistent with chance. The report should state the interval and avoid
ranking subgroups by noisy point estimates.

### Step 10. Monitor, set thresholds, and govern

**Input:** Deployment plan, metrics, owners.
**Output:** A monitoring plan with metrics, cadence, thresholds, and an escalation path.
**Rationale:** Fairness drifts with prevalence, case mix, coding changes, and workflow changes. The
monitoring plan should specify which metrics are tracked, at what frequency, with what threshold
for review, and who is notified. A dashboard without a threshold and an owner is observation, not
governance.

The plan should also specify what happens when a threshold is crossed: a review, a pause, a
recalibration, or a rollback. The decision should be made before the alert fires, because the
pressure after an alert is to explain it away.

### Choosing subgroups without p-hacking

The subgroup list should be written before the metrics are computed and justified by three sources:
the legal context (which attributes are protected), the clinical context (which groups have
different prevalence, access, or measurement), and the community context (which groups have raised
concerns). The list should be short enough to act on. A report with forty subgroups and no
prioritisation is a data dump.

The prespecification matters because every additional subgroup increases the chance of finding a
large difference by noise. With ten independent subgroups and a 5 percent significance level, the
chance of at least one false positive is about 1 - 0.95^10 = 40 percent. A team that scans
subgroups after the fact will always find something, and the finding will not replicate.

### Uncertainty and the minimum detectable difference

Subgroup rates carry sampling error, and the error is largest exactly where the sample is
smallest. For a proportion p with n observations, the standard error is sqrt(p(1-p)/n). Some
worked values from Section 6:

| Estimate | n | Standard error | 95% interval half-width |
|---|---:|---:|---:|
| TPR 0.900 (A) | 200 cases | 0.0212 | 4.2 pp |
| TPR 0.600 (B) | 150 cases | 0.0400 | 7.8 pp |
| TPR 0.500 (small group) | 12 cases | 0.1443 | 28.3 pp |

The difference between A and B is 30 percentage points, with a standard error of
sqrt(0.0212^2 + 0.0400^2) = 0.0453, or about 8.9 percentage points at 95 percent confidence. The
gap is well outside noise. The small-group estimate, by contrast, has an interval of plus or minus
28 points, so a rate of 0.5 in 12 cases is consistent with a true rate anywhere from 0.22 to 0.78.

The minimum detectable difference follows from the same arithmetic. With 50 cases per group, the
standard error of a difference near 0.5 is about 0.10, so a difference below 20 percentage points
cannot be reliably detected. A subgroup analysis with that sample size is descriptive, not
inferential, and the report should say so.

### Fairness in deployment: thresholds and human review

The deployed system is not only the model. It is the model, the threshold, the interface, and the
human who acts on the output. Three deployment effects change the fairness picture:

- **Automation bias.** Clinicians over-rely on a score, especially when it is presented as precise.
  A subgroup with a miscalibrated high band can therefore receive systematically different care
  even when the ranking is fair.
- **Override patterns.** If clinicians override the model more often for one group, the effective
  decision rule differs by group. The override rate is a fairness metric, and it is rarely
  reported.
- **Review capacity.** A threshold that generates more alerts than the team can review produces
  triage by queue order, which can correlate with the group that generates more alerts.

The monitoring plan should include the override rate and the time-to-review by group, not only the
model metrics. A model that is fair on paper and unfair in workflow is unfair.

### Mitigation in practice

Three mitigations are commonly implemented, and each has an operational catch.

**Reweighting** assigns higher sample weights to the underrepresented group or to the outcome
combination that is sparse. It is simple to implement and can change the model's ranking. The catch
is that it distorts the joint distribution: a model trained on reweighted data may be less
accurate for the majority group, and the weights must be recomputed when the population changes.
The method also requires group labels at training time, which may not be available or lawful to
collect.

**Fairness constraints** add a penalty or a constraint to the optimisation, such as a maximum
difference in true positive rates. The catch is that the constraint may be infeasible at the
desired accuracy, and the solver may trade away overall performance in ways that are hard to
explain. The constraint should be reported as a tolerance (for example, a TPR gap no larger than
0.05), not as a guarantee.

**Group-specific thresholds** adjust the decision cut-off per group after training. The catch is
legal and ethical: the approach uses the protected attribute at decision time, which may be
impermissible in some jurisdictions and contested in others, and it requires the attribute to be
available at inference. A variant that adjusts for a proxy is possible but reintroduces the proxy
problem.

The choice among them should follow the mechanism. If the gap comes from the label, none of the
three fixes it; the label must change. If the gap comes from representation, reweighting or more
data may help. If the gap comes from the threshold, post-processing is the direct remedy.

### The fairness review meeting

The review is a decision meeting, not a presentation. It should have four artefacts: the harm
statement, the metrics table with intervals, the mechanism statement, and the mitigation options
with costs. The attendees should include the clinical owner, the data science lead, a privacy or
legal representative, and at least one person representing the affected population.

The meeting should end with a recorded decision: deploy, deploy with conditions, remediate, or do
not deploy. The decision should name the criterion chosen, the trade-off accepted, the monitoring
plan, and the person accountable. A meeting that ends with "looks good" and no record has not made
a decision; it has deferred one to the deployment.

### What a fairness assessment is not

It is not a certification, because the metrics are population-specific and time-specific. It is not
a legal opinion, because disparate impact analysis requires counsel and a factual record. It is not
a substitute for clinical validation, because a fair model can still be inaccurate. And it is not a
one-time deliverable, because the population and the workflow change. The assessment is a
structured argument that the model's benefits and harms have been measured, attributed to a
mechanism, and accepted by someone with authority.

### The decision record

The record should contain the harm statement, the prespecified subgroups and criteria, the metrics
with intervals, the mechanism statement, the mitigation and its cost, the decision, the name of the
acceptor, and the monitoring plan. It is the artefact that a regulator, a procurement reviewer, or a
successor team will read. A record that contains only the final metrics cannot be distinguished
from a record that was written after the fact to justify a decision already made.

### Edge cases

**Small cells.** A subgroup analysis with a handful of events produces unstable rates. Report the
count and the interval, and avoid conclusions that the data cannot support. Suppress cells below a
threshold in any published table, as in Topic 02.

**Quasi-identifiers.** Reporting performance for a small subgroup defined by age, geography, and
race can identify individuals, especially in a rare condition. The fairness report is itself a
disclosure surface, and it needs the same small-cell review as any other release.

**Re-identification through linkage.** A performance table linked to a public dataset can reveal
which patients were in the training set. The report should be aggregate, and the underlying
evaluation data should be access-controlled.

**Free-text PHI.** Notes are both a feature source and a bias source. Language use, documentation
practices, and clinician behaviour differ by group, so a text model can learn proxies for race or
socioeconomic status. The audit should test for proxy performance and consider removing features
that encode identity rather than clinical signal.

**Metadata leakage.** Site identifiers, device models, and timestamps in a fairness report can
reveal which hospital or which population a subgroup came from. Aggregate before publishing.

**Consent withdrawal.** A participant who withdraws may need to be removed from the evaluation,
which can change subgroup sizes and metrics. The monitoring plan should state how withdrawal is
handled and whether the model is retrained.

**Minors and deceased patients.** Age subgroups need special care: a paediatric model evaluated on
adult data is a deployment bias, and a decedent's data may be excluded from follow-up outcomes in
ways that bias the label.

**Cross-border transfer.** Protected attributes and their legal definitions differ by jurisdiction.
A model evaluated on US race categories may not map to EU categories, and a fairness report
transferred across borders may itself be personal data. The analysis should state which definitions
were used and why.

## 6. Worked example

### Setting

A health system wants to deploy a model that flags patients for a diabetes prevention programme.
The programme has capacity for 20 percent of the eligible population. The model is evaluated on
2,000 patients from two groups, A and B, with the same two-year follow-up window. Group A has a
prevalence of 20 percent; group B has a prevalence of 15 percent. The threshold was chosen to fill
the programme at 20 percent overall, which produces a selection rate of 30 percent in A and 15
percent in B.

### The confusion matrices

| Group | TP | FP | TN | FN | N |
|---|---:|---:|---:|---:|---:|
| A | 180 | 120 | 680 | 20 | 1,000 |
| B | 90 | 60 | 790 | 60 | 1,000 |

Read the matrix before the rates. Group A has 200 cases and misses 20 of them. Group B has 150
cases and misses 60. The number of missed cases is three times higher in B even though B has fewer
cases. That single comparison is often more persuasive to a clinical audience than any ratio.

### The metrics

| Metric | Group A | Group B | Gap |
|---|---:|---:|---:|
| Prevalence | 20.0% | 15.0% | 5.0 pp |
| Selection rate | 30.0% | 15.0% | 15.0 pp |
| TPR (sensitivity) | 0.900 | 0.600 | 0.300 |
| FPR | 0.150 | 0.071 | 0.079 |
| PPV (precision) | 0.600 | 0.600 | 0.000 |
| Accuracy | 0.860 | 0.880 | 0.020 |
| F1 | 0.720 | 0.600 | 0.120 |

The arithmetic, in full:

- A TPR = 180 / (180 + 20) = 180 / 200 = 0.900.
- B TPR = 90 / (90 + 60) = 90 / 150 = 0.600.
- A FPR = 120 / (120 + 680) = 120 / 800 = 0.150.
- B FPR = 60 / (60 + 790) = 60 / 850 = 0.0706.
- A PPV = 180 / (180 + 120) = 180 / 300 = 0.600.
- B PPV = 90 / (90 + 60) = 90 / 150 = 0.600.
- A selection = 300 / 1,000 = 0.300.
- B selection = 150 / 1,000 = 0.150.
- Demographic parity ratio = 0.150 / 0.300 = 0.500.
- Equal opportunity gap = 0.900 - 0.600 = 0.300.
- FPR gap = 0.150 - 0.0706 = 0.0794.
- Predictive parity gap = 0.600 - 0.600 = 0.000.

### Diagnostic 1: the parity ratio

The demographic parity ratio is 0.500, below the four-fifths threshold of 0.80 used in US
employment guidance as a signal of adverse impact. The ratio is a screening signal, not a legal
conclusion, and the employment guidance does not govern a clinical programme. It is still useful
because it is easy to explain: group B is selected at half the rate of group A.

The parity ratio alone does not say whether the difference is unjust. If group B had a genuinely
lower prevalence and the model were equally accurate, a lower selection rate could be appropriate.
The next diagnostic answers that question.

### Diagnostic 2: the true positive rate gap

The TPR gap is 0.300, or 30 percentage points. In absolute terms, 60 of 150 cases in B are missed,
against 20 of 200 in A. The missed-case rate is 40 percent in B and 10 percent in A. For a
prevention programme whose purpose is to find people who would benefit, this is the harm that
matters: the group with the higher need per case is the group the model fails to find.

The two diagnostics point to different stories. The parity ratio says "B is selected less often".
The TPR gap says "B's cases are missed more often". A team that reports only the first may respond
by lowering B's threshold to increase selection, which improves parity and may or may not improve
case finding. A team that reports only the second may equalise TPR and create a new burden of false
positives. The metrics are not interchangeable.

### The impossibility in arithmetic

Suppose the team equalises true positive rates by lowering the threshold for group B. To reach TPR
= 0.900 in B, the model must find 135 of 150 cases instead of 90: 45 additional true positives. In
the region where the threshold is lowered, suppose the model finds those 45 cases at the cost of
120 additional false positives, a precision of 45 / 165 = 0.273 in the added region.

The adjusted matrix for B is TP = 135, FP = 180, TN = 670, FN = 15. The metrics become:

- TPR = 135 / 150 = 0.900, equal to A.
- FPR = 180 / 850 = 0.2118, against 0.150 in A.
- PPV = 135 / 315 = 0.4286, against 0.600 in A.
- Selection = 315 / 1,000 = 0.315, against 0.300 in A.
- Parity ratio = 0.300 / 0.315 = 0.952, close to equal.
- Predictive parity gap = 0.600 - 0.4286 = 0.1714.

Equal opportunity now holds. Equalized odds does not, because the false positive rate is higher in
B (0.2118 vs 0.150). Predictive parity does not, because a positive prediction in B is less likely
to be a true case (0.429 vs 0.600). The cost is 120 additional false positives for 45 additional
true positives, or 2.67 false positives per extra case found.

This is the impossibility result in one table. With different base rates and an imperfect model, a
team can equalise TPR, or FPR, or PPV, but not all three. The choice is a statement about which
error the programme can tolerate: 2.67 extra false positives per additional case, or 45 missed
cases.

### Calibration within groups

The model's risk bands show a further difference. Predicted and observed event rates:

| Band | Predicted | A observed (n) | B observed (n) |
|---|---:|---:|---:|
| Low (0-0.2) | 0.10 | 0.05 (600) | 0.05 (700) |
| Medium (0.2-0.5) | 0.35 | 0.30 (250) | 0.28 (200) |
| High (0.5-1.0) | 0.75 | 0.72 (150) | 0.52 (100) |

The expected calibration error (ECE), weighted by band size, is 0.047 for A and 0.072 for B. The
gap is driven by the high-risk band, where B's observed rate is 0.52 against a predicted 0.75. A
clinician using the score as a probability would over-estimate risk for the highest-risk patients
in B, which can change the decision even when the ranking is similar.

Calibration is not a substitute for the error-rate metrics. A model can be calibrated within groups
and still have unequal TPR, because calibration constrains the average prediction, not the
threshold behaviour.

### Vary one assumption: prevalence in B rises to 20 percent

Suppose the programme reaches a subgroup of B with the same 20 percent prevalence as A, while the
model's TPR and FPR stay at 0.600 and 0.0706. The selection rate becomes:

0.20 x 0.600 + 0.80 x 0.0706 = 0.120 + 0.0565 = 0.1765.

The parity ratio becomes 0.1765 / 0.300 = 0.588, up from 0.500 but still below 0.80. The PPV
becomes 0.120 / 0.1765 = 0.680, so predictive parity now fails in the other direction (A 0.600, B
0.680).

The conclusion changes in two ways. First, the parity gap narrows without any change to the model,
which shows that prevalence and access are part of the fairness problem, not only the algorithm.
Second, the PPV gap appears, which shows that equal TPR and FPR do not produce equal PPV when
prevalence differs. A team that had tuned the model to the original population would need to
re-evaluate after a population shift, which is the argument for monitoring rather than a one-time
audit.

### Decision

The assessment recommends three actions, each tied to a mechanism:

1. **Relabel.** Replace the cost-based proxy with a need-based outcome (for example, a composite of
   uncontrolled HbA1c and documented complications) and re-evaluate. This addresses the mechanism
   rather than the symptom.
2. **Set the threshold by harm.** Because the programme's purpose is case finding and it has
   capacity for 20 percent, choose a threshold that equalises TPR subject to a false positive
   budget, and report the 2.67 false positives per extra case to the clinical owner.
3. **Monitor quarterly.** Track TPR, FPR, PPV, and calibration by group at the deployed threshold,
   with a review trigger at a 5 percentage point change.

The decision record states the trade-off explicitly: the programme accepts additional false
positives in exchange for finding 45 more cases per 1,000 patients in group B. That is a clinical
and ethical judgement, not a statistical one, and it is recorded with the name of the person who
made it.

### What the arithmetic does not decide

The metrics describe the model; they do not decide what to do. The 2.67 false positives per extra
case is a quantity, not a verdict. A programme with capacity for 20 percent of the population may
accept it; a programme with a fixed budget per patient may not. The decision belongs to the
clinical owner and the community, and the data scientist's job is to make the exchange rate
visible. A team that presents the trade-off as a technical recommendation has moved a policy
decision into a place where it cannot be contested.

The script reports rates to three decimals and the raw counts are visible in the source, so a
reviewer can recompute every number. That transparency is deliberate: a fairness report that shows
only derived rates cannot be checked, and a number that cannot be checked will not survive a
challenge.

## 7. Code, query, or calculation

The script below computes the metrics, the parity ratio, the calibration error, and the threshold
what-if from Section 6. It uses only the standard library. The confusion matrices are synthetic.

```python
"""Group fairness metrics and a threshold what-if for a synthetic classifier."""

GROUPS = {
    "A": {"tp": 180, "fp": 120, "tn": 680, "fn": 20},
    "B": {"tp": 90, "fp": 60, "tn": 790, "fn": 60},
}

CALIBRATION = {
    "A": [(0.10, 600, 0.05), (0.35, 250, 0.30), (0.75, 150, 0.72)],
    "B": [(0.10, 700, 0.05), (0.35, 200, 0.28), (0.75, 100, 0.52)],
}


def metrics(c):
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    n = tp + fp + tn + fn
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    sel = (tp + fp) / n
    acc = (tp + tn) / n
    f1 = 2 * ppv * tpr / (ppv + tpr) if ppv + tpr else 0.0
    return {"n": n, "tpr": tpr, "fpr": fpr, "ppv": ppv, "selection": sel,
            "accuracy": acc, "f1": f1, "prevalence": (tp + fn) / n}


def ece(rows):
    total = sum(n for _, n, _ in rows)
    return sum(n / total * abs(obs - pred) for pred, n, obs in rows)


if __name__ == "__main__":
    m = {g: metrics(c) for g, c in GROUPS.items()}
    for g, v in m.items():
        print(f"group {g}: n={v['n']}, prevalence={v['prevalence']:.1%}, "
              f"TPR={v['tpr']:.3f}, FPR={v['fpr']:.3f}, PPV={v['ppv']:.3f}, "
              f"selection={v['selection']:.3f}, accuracy={v['accuracy']:.3f}, "
              f"F1={v['f1']:.3f}")

    ratio = min(m["A"]["selection"], m["B"]["selection"]) / max(
        m["A"]["selection"], m["B"]["selection"])
    print(f"demographic parity ratio={ratio:.3f} (four-fifths threshold 0.80)")
    print(f"equal opportunity gap (TPR)={abs(m['A']['tpr'] - m['B']['tpr']):.3f}")
    print(f"FPR gap={abs(m['A']['fpr'] - m['B']['fpr']):.3f}")
    print(f"predictive parity gap (PPV)={abs(m['A']['ppv'] - m['B']['ppv']):.3f}")
    print(f"calibration ECE: A={ece(CALIBRATION['A']):.3f}, "
          f"B={ece(CALIBRATION['B']):.3f}")

    adjusted = {"tp": 135, "fp": 180, "tn": 670, "fn": 15}
    v = metrics(adjusted)
    print(f"adjusted B: TPR={v['tpr']:.3f}, FPR={v['fpr']:.3f}, "
          f"PPV={v['ppv']:.3f}, selection={v['selection']:.3f}")
    print(f"extra TP=45, extra FP=120, FP per extra TP={120 / 45:.2f}")
    print(f"new parity ratio="
          f"{min(m['A']['selection'], v['selection']) / max(m['A']['selection'], v['selection']):.3f}")
    print(f"new PPV gap={abs(m['A']['ppv'] - v['ppv']):.3f}")

    prev, tpr, fpr = 0.20, 0.60, 60 / 850
    sel_b = prev * tpr + (1 - prev) * fpr
    ppv_b = prev * tpr / sel_b
    print(f"B at 20% prevalence: selection={sel_b:.3f}, PPV={ppv_b:.3f}, "
          f"parity ratio={min(0.30, sel_b) / max(0.30, sel_b):.3f}")
```

Expected output, verified on Python 3.11:

```text
group A: n=1000, prevalence=20.0%, TPR=0.900, FPR=0.150, PPV=0.600, selection=0.300, accuracy=0.860, F1=0.720
group B: n=1000, prevalence=15.0%, TPR=0.600, FPR=0.071, PPV=0.600, selection=0.150, accuracy=0.880, F1=0.600
demographic parity ratio=0.500 (four-fifths threshold 0.80)
equal opportunity gap (TPR)=0.300
FPR gap=0.079
predictive parity gap (PPV)=0.000
calibration ECE: A=0.047, B=0.072
adjusted B: TPR=0.900, FPR=0.212, PPV=0.429, selection=0.315
extra TP=45, extra FP=120, FP per extra TP=2.67
new parity ratio=0.952
new PPV gap=0.171
B at 20% prevalence: selection=0.176, PPV=0.680, parity ratio=0.588
```

Four readings of the output. Group B has higher accuracy and lower F1 than group A, which is the
accuracy paradox in one line: the model is more "right" overall in the group where it misses more
cases. The parity ratio of 0.500 and the TPR gap of 0.300 are both true and describe different
harms. The adjusted threshold equalises TPR and breaks predictive parity, with a clear exchange
rate of 2.67 false positives per extra case. The prevalence variation narrows parity and creates a
PPV gap, showing that population change moves the metrics without any model change.

The script is a measurement tool, not a fairness decision. The decision requires the harm
statement and the clinical owner from Steps 1 and 5.


## 8. Data, standards, and regulatory context

Fairness is governed by anti-discrimination law, shaped by AI regulation, and operationalised by
standards that are mostly voluntary. The three layers answer different questions, and a programme
that satisfies one may fail another.

### United States: anti-discrimination law

- Section 1557 of the Affordable Care Act (42 U.S.C. 18116) prohibits discrimination on the basis
  of race, color, national origin, sex, age, or disability in health programmes and activities that
  receive federal financial assistance. The 2024 final rule implementing Section 1557 addresses,
  among other things, patient care decision support tools and the obligation not to discriminate
  through their use. The rule has been subject to litigation, and its status should be checked
  before relying on a specific provision.
- Title VI of the Civil Rights Act of 1964 applies to recipients of federal funds and prohibits
  race, color, and national origin discrimination. It is enforced by federal agencies, including
  HHS.
- The Americans with Disabilities Act (ADA) and Section 504 of the Rehabilitation Act prohibit
  disability discrimination in covered entities.
- The EEOC Uniform Guidelines on Employee Selection Procedures (1978) describe the four-fifths
  rule: a selection rate for a protected group that is less than 80 percent of the rate for the
  highest group is evidence of adverse impact. The rule is employment guidance, not a clinical
  standard, but it is widely used as a screening threshold because it is simple and public.

The legal concepts to keep distinct are disparate treatment (different treatment because of a
protected characteristic) and disparate impact (a neutral practice with a disproportionately
adverse effect). A model that never uses race as a feature can still produce disparate impact. The
legal analysis is fact-specific and requires counsel; a fairness metric is evidence, not a
conclusion.

### European Union

- The GDPR Article 22 restricts solely automated decisions that produce legal or similarly
  significant effects, and provides rights to human intervention, explanation, and contestation.
  Recital 71 addresses profiling and discrimination. Article 9 restricts processing of special
  category data, including racial or ethnic origin and health data, which affects how protected
  attributes can be used in an audit.
- The EU AI Act (Regulation (EU) 2024/1689) classifies systems used in health care as high-risk in
  several contexts and requires a risk management system, data governance, and examination of
  training data for biases (Article 10), as well as accuracy, robustness, and cybersecurity
  (Article 15). The Act applies in stages from 2025; the high-risk obligations apply from 2026 and
  2027 depending on the category. The obligations are about process and evidence, not about a
  specific fairness metric.
- The European Health Data Space Regulation (EU) 2025/327 adds secondary-use governance from 2029,
  including permits and secure processing environments. It does not prescribe fairness metrics, but
  it constrains how evaluation data may be accessed.

### Standards and frameworks

- The NIST AI Risk Management Framework (AI RMF 1.0, January 2023) lists "Fair - with harmful bias
  managed" as one of the trustworthiness characteristics and organizes work into Govern, Map,
  Measure, and Manage. It is voluntary and does not define a metric.
- ISO/IEC TR 24027:2021, *Bias in AI systems and AI aided decision making*, provides a vocabulary
  for bias sources and mitigation. ISO/IEC 42001:2023 specifies an AI management system and is
  certifiable.
- The OECD AI Principles (2019, updated 2024) include human-centred values, fairness, transparency,
  and accountability. They are principles, not requirements.
- The WHO guidance on ethics and governance of AI for health (2021) provides a public-health
  framing, including equity and inclusiveness.

**Fairness and privacy trade off.** A fairness audit needs subgroup labels, and subgroup labels are
sensitive data. Collecting race, ethnicity, language, disability, and gender identity for every
patient increases the privacy surface and may require a legal basis in jurisdictions where those
attributes are special category data. The audit should collect the minimum attributes needed for
the prespecified subgroups, store them separately, restrict access, and report only aggregate
results above the small-cell threshold.

The AI Act's data-governance requirement is worth reading closely, because it asks for examination
of training, validation, and test data for biases that could affect health and safety, and for
measures to detect and mitigate them. It does not name a metric. The evidence that satisfies it is
therefore a process: the label audit, the subgroup evaluation, the mitigation record, and the
monitoring plan. A vendor that claims compliance without those artefacts has not produced the
evidence the regulation asks for.

> **Note:** Regulatory content in this file is educational and is not legal advice. Fairness
> compliance depends on the jurisdiction, the decision, the population, and the evidence, and it
> requires counsel.

The classification to keep straight: anti-discrimination law is binding and fact-specific; the
four-fifths rule is employment guidance used by analogy; a fairness metric is a statistical
description; and the AI RMF is voluntary guidance. A model card that reports demographic parity
does not establish compliance with Section 1557, and a legal review that finds no disparate
treatment does not establish equal opportunity.

## 9. Industry practice and case evidence

### Obermeyer et al., 2019

The most cited health-equity case examined a commercial risk-prediction algorithm used to identify
patients for high-risk care management. The algorithm predicted future health care costs, and the
study found that at a given risk score Black patients were considerably sicker than White patients:
at the 97th percentile, 26.3 percent more chronic illnesses (4.8 versus 3.8 conditions). The
mechanism was the label. Because less was spent on Black patients at equal levels of need, about
USD 1,801 less per year conditional on chronic conditions, the cost model concluded that they were
healthier. The algorithm was well calibrated on cost and biased on health.

The study simulated the effect of removing the disparity: the share of Black patients identified
for additional help at the 97th percentile would rise from 17.7 percent to 46.5 percent. In a label
experiment, the fraction of Black patients in the highest-risk group ranged from 14.1 percent with a
cost-trained model to 26.7 percent with a chronic-conditions-trained model. The manufacturer
replicated the analysis on 3.7 million commercially insured patients and reported that reformulating
the label reduced bias substantially.

The lesson for this topic is that the label is a design decision with distributive consequences.
No amount of post-processing fixes a label that measures access rather than need, and the fairness
metrics computed on the cost label would have looked acceptable because the model was accurate on
its own target.

### Sjoding et al., 2020

The pulse oximetry study compared paired pulse-oximeter readings with arterial blood gas
measurements. Occult hypoxaemia, defined as an arterial oxygen saturation below 88 percent despite
a pulse-oximeter reading of 92 to 96 percent, occurred in 11.7 percent of measurements in Black
patients (88 of 749) and 3.6 percent in White patients (99 of 2,778) in the Michigan cohort. In a
multicenter ICU cohort, the rates were 17.0 percent and 6.2 percent. The authors described the
frequency in Black patients as nearly three times that in White patients. The area under the
receiver operating characteristic curve for detecting hypoxaemia was 0.84 in Black patients and
0.89 in White patients.

The mechanism is measurement bias in the instrument, not in the model that consumes it. A model
trained on pulse-oximeter readings inherits the error, and a clinical workflow that trusts the
reading inherits it too. The remedy is a better instrument or a correction, not a fairness
constraint on the model.

### Representation in imaging and dermatology

Imaging and dermatology datasets are documented to underrepresent darker skin tones, and models
trained on them can perform worse on the underrepresented group. The Fitzpatrick17k dataset work
(Groh et al., 2021) made the imbalance visible and showed performance gaps for dermatology
classifiers. The general lesson is that representation bias is measurable before deployment, and a
subgroup evaluation on a balanced test set is the minimum evidence.

### What mature programmes do

- **Model cards and datasheets.** Structured documentation of intended use, training data,
  evaluation data, subgroup performance, and limitations. A model card is a disclosure, not an
  audit; its value depends on whether the subgroup numbers are computed and reported.
- **Subgroup reporting as a release gate.** A model cannot be released without performance by
  prespecified subgroups, with confidence intervals and the threshold stated.
- **Fairness tooling in the pipeline.** Fairlearn, AI Fairness 360, and similar libraries compute
  the metrics and support mitigation experiments. The tooling reduces arithmetic errors; it does
  not choose the criterion.
- **Post-market monitoring.** Subgroup metrics are tracked after deployment, with review triggers
  and a rollback path. The FDA's device post-market surveillance model is a useful analogue even
  where it does not apply.
- **Community and clinician review.** Affected communities and clinical owners review the harm
  statement and the trade-off. The review is documented, and the decision is attributed.

**Fairness is a procurement requirement.** Health systems increasingly ask vendors for subgroup
performance, the evaluation population, the label definition, and the monitoring plan. A vendor
that cannot supply them is treated as a clinical risk, because the buyer cannot tell whether the
model will work for its population. The artefact that satisfies the request is not a marketing
claim about fairness; it is a table of metrics with sample sizes, thresholds, and intervals.

**What a vendor assessment should include.** The evaluation population, the label definition, the
subgroup metrics with sample sizes and intervals, the deployed threshold, the calibration results,
the mitigation applied and its cost, the monitoring plan, and the update cadence. A vendor that
provides only aggregate AUROC has described discrimination, not fairness, and has not supplied the
information a buyer needs to decide whether the model is appropriate for its population.

The Obermeyer and Sjoding cases are often taught as examples of "bias in AI", but their mechanisms
are different and their remedies do not transfer. The first is a label-choice problem in a
socio-technical system; the second is a measurement problem in a device. A team that responds to
both with a fairness constraint on the model will fix neither. The mechanism diagnosis in Step 7 is
the step that prevents this, and it is the step most often skipped because it requires clinical and
technical knowledge at the same time.

**The report should state what was not measured.** A fairness assessment that lists only the
metrics it computed implies that the others do not matter. The report should name the criteria it
did not evaluate, the subgroups it could not analyse, and the harms it could not quantify. That
honesty is what allows a reviewer to see the boundaries of the conclusion, and it is what
distinguishes an assessment from a marketing document. A report that says "we did not evaluate
equalized odds because the subgroup had 14 events" is more credible than one that omits the
criterion silently.

> **Note:** The case facts above are drawn from the cited publications. Statements about what an
> organisation should have done differently are inference, labelled as such where they appear.

### Where practice is weaker than the documents suggest

Three gaps recur, and they are inference from published evaluations rather than from a single
study. First, subgroup analyses are reported only for groups with enough data, so the smallest
groups, which often have the largest gaps, are omitted. Second, the threshold used in the fairness
report differs from the threshold used in deployment, so the reported metrics do not describe the
deployed system. Third, the label audit is skipped, so the team measures fairness on a proxy that
already encodes inequity. Each gap is checkable: ask for the subgroup sample sizes, the deployed
threshold, and the label definition.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Single-metric defence | Report shows only demographic parity | Other error rates unmeasured | Missed cases concentrate in a group | Report all criteria |
| Proxy label | Target is cost or utilisation | Label measures access, not need | Under-ranking of lower-spend groups | Relabel; validate against need |
| Accuracy as fairness | "Overall accuracy is 88%" | Aggregate hides subgroup error | Unequal harm | Subgroup confusion matrices |
| Threshold mismatch | Report threshold differs from deployment | Metrics describe a different system | Undetected harm in production | Report deployed threshold |
| No confidence intervals | Subgroup rates without n or CI | Small-sample noise read as signal | Wrong remediation | Intervals; minimum n |
| Intersection omitted | Only race and sex reported separately | Harms concentrate at intersections | Group-specific harm missed | Prespecified intersections |
| Calibration ignored | Only ranking metrics reported | Probability scale wrong for a group | Decision rule misapplied | Calibration by group |
| Post-processing only | Threshold adjusted without diagnosis | Symptom treated, cause remains | Gap returns after population shift | Mechanism-based remedy |
| No monitoring | Fairness audit is a launch gate | Population and workflow drift | Gaps emerge silently | Quarterly subgroup metrics |
| Protected attribute unavailable | No data to stratify | Cannot measure | Gaps unknown | Proxies with caution; data collection |
| Legal review mistaken for fairness | "Counsel approved" | Legal test is not a metric | Harm persists | Separate statistical and legal review |
| Small subgroup over-read | A dramatic rate in n=12 | Sampling noise | Misallocated remediation | Intervals; suppress cells |
| Report leaks identities | Subgroup table has small cells | Aggregation is not protection | Privacy harm | Minimum cell size |
| Community not involved | No patient or clinician input | Harm statement is internal | Wrong trade-off chosen | Advisory review |
| Proxy attribute retained | Model uses language or site as a feature | Feature encodes group membership | Proxy discrimination | Feature audit; ablation test |
| Label definition changed silently | New outcome definition in production | Metrics no longer comparable | Undetected drift | Version the label; re-evaluate |
| Evaluation data leakage | Subgroup labels used in training | Inflated performance | False assurance | Separate evaluation set |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The label measures need | All fairness metrics depend on it | Cost or access proxy | Label audit against clinical measures |
| The evaluation population matches deployment | Subgroup rates are population-specific | Case-mix or site change | Compare cohort to deployment data |
| Protected attributes are measured accurately | Stratification depends on them | Self-report missing or misclassified | Missingness analysis; alternative sources |
| The threshold is the deployed threshold | Metrics describe the decision | Report uses a different cut-off | Configuration check |
| Subgroups are large enough | Rates are stable | Small n | Confidence intervals |
| The mechanism diagnosis is correct | Remedy depends on it | Multiple mechanisms | Test the remedy's effect |
| The metric choice reflects the harm | Trade-off is a policy decision | Data scientist chooses alone | Documented owner sign-off |
| Calibration is stable over time | Probability rules depend on it | Prevalence shift | Monitoring |
| Monitoring data are available | Drift detection depends on them | No subgroup labels in production | Data collection plan |
| Legal definitions map to data categories | Compliance analysis depends on them | Jurisdiction or category mismatch | Counsel review |
| The model is the main source of harm | Effort is proportional | Workflow or access is the cause | Process mapping |
| The fairness metric is stable across sites | Monitoring thresholds are portable | Site case mix differs | Site-level evaluation |
| The evaluation data are representative | Subgroup rates generalise | Cohort excludes a group | Coverage map against deployment |

Two limits are structural. First, a fairness metric describes a model in a population at a
threshold; it does not describe the justice of the decision or the distribution of benefit. Second,
the impossibility results mean that no model can satisfy every criterion when base rates differ and
accuracy is imperfect. A programme that promises "a fair model" without naming the criterion and
the trade-off is promising something that cannot be delivered.

## 12. Exercises

The exercises are ordered to build a fairness assessment: compute, explain, audit, choose, trade
off, monitor, critique, and intersect. Exercise 5 is the one that most often changes a team's
mind, because it shows that parity can be achieved by degrading the better-performing group.

Each exercise states a deliverable, acceptance criteria, and a time budget. The calculation
exercises can be done with a calculator; the design exercises need the harm statement from a real
or realistic programme. Exercise 8 requires a synthetic dataset and a script; the point is the
comparison between separate and intersectional results, not the dataset.

A fairness report should be reproducible from the underlying data and code. A table of rates
without the confusion matrices, the threshold, and the subgroup definitions is a claim, not
evidence. The reproducibility requirement is also a protection against the metric-shopping failure:
a reviewer who can recompute the numbers can see the criteria that were not reported. It also makes
the report reusable when the population changes, because the same code can be rerun rather than
rebuilt.

**Exercise 1 - Compute the five metrics (calculation, 40 minutes).**
Given a confusion matrix for two groups, compute demographic parity, TPR, FPR, PPV, and accuracy,
and state which criterion each favours. Deliverable: table with arithmetic. Criteria: rates correct
to three decimals; gaps identified; no metric reported without its threshold.

**Exercise 2 - Explain the impossibility (recall, 25 minutes).**
In your own words, explain why equal TPR, equal FPR, and equal PPV cannot all hold when prevalence
differs. Use the base-rate identity. Deliverable: half a page. Criteria: names the identity; shows
the role of prevalence; does not blame the model.

**Exercise 3 - Audit a label (application, 45 minutes).**
Choose a clinical model you know and audit its label against need, access, and documentation.
Propose an alternative label and describe how it would be collected. Deliverable: one page.
Criteria: distinguishes need from access; alternative is feasible; measurement cost stated.

**Exercise 4 - Choose a criterion (design, 40 minutes).**
For a sepsis alert, a care-management enrolment, and a screening colonoscopy outreach, choose a
primary fairness criterion for each and justify it by the harm. Deliverable: three short memos.
Criteria: harm named; criterion matches the error that matters; trade-off stated.

**Exercise 5 - Threshold trade-off (calculation, 45 minutes).**
Using the Section 6 matrices, compute the effect of raising group A's threshold so that its
selection rate falls to 0.15. State the new TPR, PPV, and parity ratio. Deliverable: calculation.
Criteria: arithmetic correct; trade-off interpreted.

**Exercise 6 - Design monitoring (design, 45 minutes).**
Write a quarterly fairness monitoring plan for a deployed risk model: metrics, subgroups, cadence,
thresholds, owner, and escalation. Deliverable: two pages. Criteria: thresholds are numeric;
escalation has a decision; data availability addressed.

**Exercise 7 - Critique a model card (critique, 40 minutes).**
Find a public model card for a clinical model and assess its fairness reporting. Deliverable:
500-word critique. Criteria: identifies missing subgroups, threshold, intervals, and label
definition; proposes specific additions.

**Exercise 8 - Intersectional analysis (application, 60 minutes).**
Using a synthetic dataset, compute a metric for two attributes separately and for their
intersection, and explain the difference. Deliverable: script and table. Criteria: sample sizes
reported; intervals; interpretation avoids over-reading small cells.

## 13. Self-check answers

**Answer 1.** Using the Section 6 matrices: parity ratio 0.500; TPR 0.900 vs 0.600; FPR 0.150 vs
0.071; PPV 0.600 vs 0.600; accuracy 0.860 vs 0.880. Demographic parity favours a lower selection
gap; equal opportunity favours the TPR comparison; predictive parity is satisfied; accuracy parity
is nearly satisfied and misleading. The exercise is designed to show that the same matrix supports
different conclusions depending on the criterion, and that the criterion must be chosen before the
numbers are seen.

**Answer 2.** For a group with prevalence p, TPR t, and FPR f, PPV = pt / (pt + (1-p)f). If two
groups have the same t and f but different p, their PPVs differ, because the numerator and
denominator scale differently with p. Equal TPR and FPR therefore imply unequal PPV when prevalence
differs, unless the model is perfect. The impossibility is arithmetic, not a failure of modelling
effort, and the only resolutions are to accept unequal PPV, accept unequal error rates, or change
the population or the decision.

**Answer 5.** Raising A's threshold so that its selection rate falls to 0.15 means selecting 150
patients instead of 300. If the removed 150 are the lowest-scored among the selected, the true
positives fall by some amount and the false positives by the rest. The exact result depends on the
score distribution, which the exercise requires you to assume and state. A common assumption is
that the removed group has the same composition as the selected group, which would give TPR 0.45
and PPV 0.60; a more realistic assumption is that the removed records are enriched for false
positives, which raises PPV and lowers TPR. The point is that the answer depends on an assumption
that must be stated, and that parity can be achieved by degrading the better-performing group.

**Answer 6.** The plan should track TPR, FPR, PPV, and calibration by group at the deployed
threshold, quarterly, with confidence intervals. Thresholds should be numeric: for example, a TPR
gap above 0.10 or a parity ratio below 0.80 triggers review. The owner should be the clinical
programme lead, with the data science lead responsible for the analysis. Escalation should include
a defined decision: continue with monitoring, recalibrate, change the threshold, or pause. The plan
should state how subgroup labels are obtained in production and what happens when they are missing.

**Answer 7.** A subgroup with 12 events can produce a rate that looks dramatic and is statistically
unstable. The report should give the count, the rate, and a 95 percent interval; for 6 of 12, the
interval spans roughly 0.21 to 0.79. The analysis should avoid ranking that subgroup against others
by point estimate, should state that the estimate is descriptive, and should propose either
aggregation with a related group or additional data collection. Suppressing the subgroup entirely
is also a choice, but it should be recorded, because omission can hide a harm.

**Answer 8.** A metric computed separately for race and sex can look acceptable while the
intersection is worse, because the intersection has a different prevalence, a different measurement
pathway, or a smaller sample. The analysis should report the intersection with its sample size and
interval, and should avoid ranking intersections by point estimate when the cells are small. The
difference between the separate and intersectional results is the evidence that single-attribute
analysis is insufficient.

**Answer 9.** Monitor subgroup TPR, FPR, PPV, calibration, selection rate, override rate, and
time-to-review at the deployed threshold, quarterly, with confidence intervals. Set numeric review
thresholds, name the clinical owner, and predefine the escalation decision: continue, recalibrate,
change the threshold, or pause. The plan should state how subgroup labels are obtained in
production and how missing labels are handled, because a monitoring plan that depends on data the
workflow does not collect will not run.

**Answer 10.** Ask which criteria were computed, which subgroups and intersections, the sample
sizes and intervals, the deployed threshold, the label definition, the calibration results, the
evaluation population versus the buyer's population, the monitoring plan, and what happens when a
gap appears. A vendor that supplies a single parity number has supplied a marketing artefact, not
an assessment.

**Rubric for Exercise 1.** Full credit requires all five metrics, the threshold, and the gap for
each, with arithmetic correct to three decimals. A table that reports only the rates without the
confusion counts does not pass.

**Rubric for Exercise 2.** Full credit requires the base-rate identity and a demonstration that
equal TPR and FPR produce unequal PPV when prevalence differs. A verbal assertion without the
algebra does not pass.

**Rubric for Exercise 3.** Full credit requires a clear distinction between need, access, and
documentation; a specific alternative label; and a statement of collection cost and feasibility. A
critique that says "the label is biased" without an alternative does not pass.

**Rubric for Exercise 4.** Full credit requires a named harm for each decision, a criterion that
matches the error type, and a stated trade-off. Choosing demographic parity for all three does not
pass, because the harms differ.

**Rubric for Exercise 7.** Full credit requires assessment of subgroup coverage, threshold,
intervals, label definition, and calibration. A critique that only notes the absence of a fairness
section does not pass.

## 14. Interview questions

**1. What is the difference between demographic parity and equal opportunity?**
Strong answer: parity compares selection rates; equal opportunity compares true positive rates; the
first can be satisfied while missing cases in a group, the second targets the missed cases. Weak
answer: "both are fairness metrics".

**2. Why can't a model satisfy every fairness criterion?**
Strong answer: base rates differ; equal TPR and FPR imply unequal PPV when prevalence differs; the
trade-off is arithmetic and requires a policy choice. Weak answer: "because the data are biased".

**3. How do you audit a label?**
Strong answer: ask what the label measures, whether it proxies access or cost, whether measurement
differs by group, and whether a clinician would accept it as need; propose an alternative. Weak
answer: "we check the label distribution".

**4. What is label bias in a clinical model?**
Strong answer: the target measures cost or utilisation rather than need, so groups with less access
are under-ranked at equal need; the Obermeyer cost-proxy case is the canonical example. Weak
answer: "the labels are wrong".

**5. How do you choose a fairness criterion?**
Strong answer: from the harm statement and the decision, with the clinical owner and affected
community; name which error is worse and record the trade-off. Weak answer: "use the one that
passes".

**6. What does calibration within groups tell you?**
Strong answer: whether the predicted probability matches observed frequency in each group; it
constrains the probability scale but not the error rates at a threshold. Weak answer: "the model is
accurate".

**7. How do you handle a subgroup with 12 events?**
Strong answer: report the count and interval, avoid a precise conclusion, aggregate or collect more
data, and do not rank by noisy point estimates. Weak answer: "report the rate".

**8. What is the four-fifths rule and when does it apply?**
Strong answer: an EEOC employment guidance threshold where a selection rate ratio below 0.80
signals adverse impact; it is used by analogy in other contexts and is not a clinical or
constitutional test. Weak answer: "it is the legal definition of fairness".

**9. How do you monitor fairness after deployment?**
Strong answer: subgroup metrics at the deployed threshold, quarterly, with intervals, numeric
thresholds, an owner, and a defined escalation decision. Weak answer: "we review the model
annually".

**10. A vendor says its model is "fairness-tested". What do you ask?**
Strong answer: which criteria, which subgroups, what sample sizes, what threshold, what label, what
calibration, what monitoring, and what happens when a gap appears. Weak answer: asks for the
report.

**11. How do you audit a model for proxy discrimination?**
Strong answer: test whether features predict the protected attribute beyond clinical need, compare
performance with and without the suspect features, and examine whether the label or measurement
encodes access; proxies can survive removal of the protected attribute. Weak answer: "we removed
race from the features".

**12. What is the risk of using group-specific thresholds?**
Strong answer: they use the protected attribute at decision time, which may be legally
impermissible or ethically contested, require the attribute at inference, and can entrench the
categories; they also break predictive parity if not carefully calibrated. Weak answer: "it is the
easiest fix".

## 15. Key terms

- **Accuracy parity** - equal overall accuracy; rarely sufficient.
- **Adverse impact** - a selection rate ratio below the four-fifths threshold.
- **Base rate** - prevalence of the outcome in a group.
- **Calibration within groups** - predicted probability matches observed frequency by group.
- **Confidence interval** - a range expressing sampling uncertainty.
- **Demographic parity** - equal selection rates.
- **Disparate impact** - a neutral practice with disproportionate adverse effect.
- **Disparate treatment** - different treatment based on a protected attribute.
- **ECE** - expected calibration error.
- **Equal opportunity** - equal true positive rates.
- **Equalized odds** - equal TPR and FPR.
- **F1** - harmonic mean of precision and recall.
- **False negative** - a missed case.
- **False positive** - a wrongly flagged case.
- **Four-fifths rule** - selection rate ratio below 0.80 as an adverse-impact signal.
- **Intersectionality** - overlapping group memberships producing distinct harms.
- **Label bias** - the target measures access rather than need.
- **Measurement bias** - the instrument measures differently by group.
- **Model card** - structured model documentation.
- **Post-processing** - mitigation after training.
- **PPV** - positive predictive value.
- **Pre-processing** - mitigation on the data before training.
- **Predictive parity** - equal PPV across groups.
- **Prevalence** - proportion of the population with the outcome.
- **Protected attribute** - a legally protected characteristic.
- **Representation bias** - underrepresentation in training data.
- **Selection rate** - proportion receiving a positive decision.
- **Sensitive attribute** - a characteristic whose use or disclosure is sensitive.
- **Subgroup** - a defined population within the evaluation.
- **Threshold** - the score cut-off producing a decision.
- **TPR** - true positive rate.
- **FPR** - false positive rate.
- **Deployment bias** - the model is used in a workflow different from training.
- **Metric shopping** - selecting the criterion that passes after seeing results.
- **Subgroup power** - the ability of a subgroup sample to support a conclusion.

## 16. Sources

All URLs accessed 26 October 2026.

**Case evidence**
- Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S. *Dissecting racial bias in an algorithm
  used to manage the health of populations*. Science 366(6464):447-453, 2019.
  https://doi.org/10.1126/science.aax2342
- Sjoding, M. W., Dickson, R. P., Iwashyna, T. J., Gay, S. E., Valley, T. S. *Racial Bias in Pulse
  Oximetry Measurement*. New England Journal of Medicine 383:2477-2478, 2020.
  https://doi.org/10.1056/NEJMc2029240
- Groh, M. et al. *Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with
  the Fitzpatrick 17k Dataset*. CVPR Workshops, 2021.
  https://arxiv.org/abs/2104.09957

**Methods and theory**
- Barocas, S., Hardt, M., Narayanan, A. *Fairness and Machine Learning*. https://fairmlbook.org/
- Hardt, M., Price, E., Srebro, N. *Equality of Opportunity in Supervised Learning*. NeurIPS, 2016.
  https://arxiv.org/abs/1610.02413
- Chouldechova, A. *Fair prediction with disparate impact: A study of bias in recidivism prediction
  instruments*. Big Data 5(2), 2017. https://doi.org/10.1089/big.2016.0047
- Kleinberg, J., Mullainathan, S., Raghavan, M. *Inherent Trade-Offs in the Fair Determination of
  Risk Scores*. ITCS, 2017. https://arxiv.org/abs/1609.05807
- Mitchell, M. et al. *Model Cards for Model Reporting*. FAT*, 2019.
  https://doi.org/10.1145/3287560.3287596
- Gebru, T. et al. *Datasheets for Datasets*. Communications of the ACM 64(12), 2021.
  https://doi.org/10.1145/3458723

**Regulation and standards**
- 42 U.S.C. 18116, Section 1557 of the Affordable Care Act.
  https://www.hhs.gov/civil-rights/for-individuals/section-1557/index.html
- Regulation (EU) 2024/1689 (Artificial Intelligence Act).
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Regulation (EU) 2016/679 (GDPR), Article 22.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- NIST, *AI Risk Management Framework (AI RMF 1.0)*, January 2023.
  https://www.nist.gov/itl/ai-risk-management-framework
- ISO/IEC TR 24027:2021, *Bias in AI systems and AI aided decision making*.
- ISO/IEC 42001:2023, *AI management systems*.
- OECD, *AI Principles*, 2019, updated 2024. https://oecd.ai/en/ai-principles
- WHO, *Ethics and governance of artificial intelligence for health*, 2021.
  https://www.who.int/publications/i/item/9789240029200

**Tooling**
- Fairlearn. https://fairlearn.org/
- AI Fairness 360. https://aif360.res.ibm.com/

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_access-control-and-cryptography.md`](05_access-control-and-cryptography.md)
- Next topic: [`07_ethics.md`](07_ethics.md)
- Related topic: [`08_governance.md`](08_governance.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
