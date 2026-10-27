# Statistical Pharmacovigilance and Safety Signal Detection

> Module 13 — Biopharma and Clinical Development
> Topic 12 of 13
> Estimated study time: 5 hours
> Prerequisites: Topics 02, 06, and 07
> Version 2026.3

## 1. Why this topic exists

A Phase III programme enrols a few thousand patients and follows them for months. It can detect an
adverse event that occurs in one patient in a hundred. It cannot detect one that occurs in one
patient in ten thousand, because the expected number of events in the trial is a fraction of a
person. It also cannot detect an interaction with a drug that patients will take after approval, an
event that takes years to develop, or a risk confined to a group the protocol excluded.

The post-marketing system exists to find those events. Its main input is spontaneous reporting: a
clinician, a patient, or a company sends a report to a regulator, and the report enters a database.
The United States Food and Drug Administration (FDA) Adverse Event Reporting System held about 26
million reports by the end of 2025 by one published estimate. The World Health Organization's
VigiBase held more than 40 million reports from over 180 member countries as of February 2025. These
are large databases, and they are not cohorts.

The central difficulty is that the denominator is unknown. A report count cannot be divided by an
exposure count, because the number of people who took the drug is not in the database and the
proportion of events that get reported is unknown and varies by event, drug, country, and year. What
remains is the internal structure of the database: the proportion of reports for a drug that mention
an event, compared with the proportion for other drugs. That comparison is disproportionality, and
every metric in this topic is a variation on it.

### 1.1 The decision this analysis supports

A safety team cannot review every drug-event pair. A large database contains tens of thousands of
drugs and thousands of event terms, and the cross-product runs to millions of pairs. The team has
perhaps fifty review slots a month. The analytical problem is therefore a ranking problem: which
pairs should consume the review capacity, and which can be left alone.

That framing changes what a good metric is. A metric that ranks well but produces uninterpretable
numbers is useful. A metric that produces a precise-looking number and ranks badly is not. The
question is not "what is the relative risk" — the data cannot answer that — but "which pairs are
unusual enough, and plausible enough, to justify a clinician's afternoon".

### 1.2 The failure pattern

1. A new drug launches with 3,400 patients of Phase III exposure.
2. Eighteen months later, three reports of a rare hepatic event appear.
3. The proportional reporting ratio is 18.4, the lower confidence bound is above 1, and the pair is
   flagged as a signal.
4. The team issues a communication and proposes a label restriction.
5. A reviewer asks whether the three reports are duplicates, whether the comparator includes drugs
   for the same indication, and whether reporting was stimulated by a media story.
6. None of these questions has been asked, and the answer to each changes the number.

Steps 5 and 6 are the analytical failure. Disproportionality is a screening method, and screening
output is a hypothesis. Treating it as a finding skips the validation that the rest of the safety
system exists to perform.

### 1.3 Why the arithmetic is not the hard part

The formulas are short. PRR is a ratio of two proportions, ROR is a cross-product ratio, and both
have closed-form intervals. The hard parts are the four choices that precede the arithmetic:

- **What counts as a report?** A report with five drugs and eight events contributes to many pairs.
  Whether it counts once per pair or once per report changes every denominator.
- **What is the comparator?** All other drugs, the same class, or drugs for the same indication. The
  three give different answers, and the choice is an argument about confounding, not a technical
  default.
- **What level of the dictionary?** A preferred term, a higher-level term, or a Standardised MedDRA
  Query. A narrow term may have three cases; the broad term may have three hundred and a different
  story.
- **When is the comparison made?** Once, at a data lock, or repeatedly as reports accumulate. Repeated
  testing without adjustment produces signals eventually in any database.

Each of these is a judgement that must be documented. A metric reported without them is not
reproducible, and a reviewer cannot tell whether the number reflects pharmacology or a choice.

### 1.4 Who is blocked when the statistics are weak

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Safety physician | A defensible review priority | Receives a ranked list without its assumptions |
| Regulator | Capacity to separate signal from noise | Receives thousands of unvalidated pairs |
| Patient | Timely warning, or continued access to a useful drug | Has no view of the evidence |
| Sponsor | A label and a market | Cannot influence the reporting stream |
| Statistician | A method that survives review | Depends on coding and case data owned elsewhere |
| Journal reader | A reproducible finding | Cannot reconstruct the comparator or the data cut |

### 1.5 The industry question

> Given a database of reports with no denominator, which drug-event pairs are worth a clinician's
> attention, and what would have to be true for the apparent association to be an artefact?

The second half of the question is the one that distinguishes a statistical pharmacovigilance
function from a dashboard. A ranking without a falsification plan is a list of hypotheses with no
owner.

## 2. Learning outcomes

By the end of this topic you can:

1. **Construct** a two-by-two table for a drug-event pair and state the unit of analysis.
2. **Compute** the proportional reporting ratio (PRR), the reporting odds ratio (ROR), their
   confidence intervals, and the chi-square statistic, and interpret each as a screening metric.
3. **Explain** why a ratio of proportions is not a relative risk and what the missing denominator
   would have to be.
4. **Apply** empirical Bayes shrinkage to stabilise small counts, and interpret the empirical Bayes
   geometric mean (EBGM) and its lower bound.
5. **Choose** a comparator and a dictionary level, and show how each choice changes the conclusion.
6. **Diagnose** confounding by indication, co-prescription, duplicate reporting, and stimulated
   reporting from the pattern of counts and the case narratives.
7. **Adjust** for multiplicity across thousands of pairs and explain why a fixed alpha is
   inappropriate.
8. **Write** a signal triage note that distinguishes a signal of disproportionate reporting from a
   validated signal and from a confirmed risk.

## 3. The core mental model

### 3.1 The reporting funnel

```text
   Patients exposed                 (unknown, not in the database)
        |
        |  only some events occur
        v
   Events that occur               (unknown)
        |
        |  only some are recognised and reported
        v
   Reports submitted                (observable, biased)
        |
        |  deduplication, validity, coding
        v
   Cases in the database            (observable, cleaner)
        |
        |  pair extraction: drug x event
        v
   Two-by-two tables                (millions of pairs)
        |
        |  disproportionality, shrinkage, multiplicity control
        v
   Ranked pairs                     (thousands)
        |
        |  clinical review of narratives
        v
   Validated signals                (tens)
        |
        |  epidemiological study, mechanism, class comparison
        v
   Confirmed risks                  (a few)
```

Each arrow loses information, and the losses are not random. Serious events are over-reported
relative to mild ones. New drugs are over-reported relative to old ones. Events that received media
attention are over-reported for a period after the attention. The metrics operate on the fourth row
from the bottom and are asked to say something about the second row from the top, which is why they
cannot.

### 3.2 The two-by-two table

```text
                        Target event Y        All other events
   Target drug X              a                      b
   All other drugs            c                      d
```

```text
   PRR = [a / (a + b)] / [c / (c + d)]
   ROR = (a / b) / (c / d) = a d / (b c)
   E   = (a + b)(a + c) / N          (expected count under independence)
```

The table is built from reports, not patients. A report with two suspect drugs and three events
contributes to six pairs. Whether the analysis counts it once per pair or once per report is a
choice that must be stated, because it changes both the numerator and the denominator.

### 3.3 Three families of method

| Family | Examples | What it assumes | What it gives | Where it fails |
|---|---|---|---|---|
| Frequentist ratio | PRR, ROR | Large counts, independence | Interpretable ratio with interval | Unstable at small counts; no borrowing |
| Bayesian shrinkage | MGPS (EBGM), BCPNN (IC) | A prior distribution of true ratios across the database | Stable estimates; small counts pulled toward the prior | Prior is empirical, not clinical; still no denominator |
| Sequence or tree-based | TreeScan, sequential PM | A defined hierarchy or time window | Controls family-wise error across many pairs | Requires a pre-specified tree and windows |

The three are not competitors for a single best answer. They answer slightly different questions. The
frequentist ratio answers "how disproportionate is this pair". The Bayesian method answers "how
disproportionate is this pair, given what we know about the distribution of disproportionality across
all pairs". The tree-based method answers "is there a cluster of related events that is unusual as a
group". A mature function uses more than one and treats disagreement as information.

### 3.4 Why shrinkage works

A ratio of small counts is unstable. If the expected count is 0.05 and one case is observed, the
observed-to-expected ratio is 20, and a frequentist interval around it is so wide as to be useless.
Most such pairs are noise: with tens of thousands of pairs, some will have a ratio of 20 by chance.

Empirical Bayes shrinkage replaces the unstable ratio with a posterior estimate that borrows
strength from the distribution of all ratios in the database. A pair with one observed case and an
expected count of 0.05 is pulled toward the database-wide mean, which is near 1. A pair with 400
observed cases and an expected count of 100 is barely moved, because the data dominate the prior.

```text
   Crude ratio 20.0, one case       ->  EBGM 1.9, EB05 0.6    (not a signal)
   Crude ratio 4.0, 400 cases       ->  EBGM 3.9, EB05 3.5    (a signal)
   Crude ratio 15.8, 42 cases       ->  EBGM 12.6, EB05 9.6  (a signal)
```

The middle row is the one that matters for public health. A ratio of 4 with a large count is a
stronger finding than a ratio of 20 with one case, and only a method that accounts for precision
ranks them correctly.

### 3.5 The multiple-comparison reality

A database with 5,000 drugs and 5,000 preferred terms has up to 25 million pairs. At a nominal alpha
of 0.05, 1.25 million pairs would be flagged by chance alone. Any procedure that tests each pair at
5 percent and reports the rejections is reporting noise.

The responses are familiar from genomics. Control the false discovery rate, pre-specify a smaller
set of pairs, use a hierarchical structure so that related terms are tested together, or use a
Bayesian shrinkage method whose prior implicitly performs the adjustment. Each has costs. A false
discovery rate of 10 percent across 25 million pairs still leaves 2.5 million expected false
positives if everything is null, which is why the output is a ranking rather than a decision.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Pharmacovigilance | The science of detecting, assessing, and preventing adverse effects of medicines | Used as a synonym for case processing | Discipline |
| Spontaneous report | A report submitted voluntarily by a health professional, patient, or company | Treated as an incident case | Report |
| Individual case safety report (ICSR) | A structured report of one or more suspected reactions in one patient | Counted per event rather than per patient | Case |
| Serious adverse event | An event that results in death, is life-threatening, requires hospitalisation, causes disability, or is a congenital anomaly | Confused with severity | Classification |
| Severity | The intensity of an event | Confused with seriousness | Grade |
| Expectedness | Whether an event is listed in the reference safety information | Confused with causality | Classification |
| Causality | The judgement that a drug caused an event | Treated as a statistical output | Judgement |
| Disproportionality | Over-representation of an event among reports for a drug relative to other drugs | Interpreted as relative risk | Ratio |
| Two-by-two table | Counts of reports with and without the drug and the event | Built from patients | Table |
| PRR | Proportional reporting ratio: the ratio of the event proportion among the drug's reports to that among comparator reports | Interpreted as a risk ratio | Dimensionless |
| ROR | Reporting odds ratio: the odds of the event among the drug's reports divided by the odds among comparator reports | Interpreted as an odds ratio of disease | Dimensionless |
| Expected count | The count expected if drug and event were independent, computed from the margins | Confused with the observed count | Reports |
| Chi-square | A test of independence in the two-by-two table | Used as a measure of effect size | Statistic |
| Lower confidence bound | The lower limit of a 95 percent interval for the ratio | Used as a posterior probability | Dimensionless |
| Empirical Bayes | A method that estimates a prior from the data and shrinks each estimate toward it | Confused with a fully Bayesian analysis with clinical priors | Method |
| MGPS | Multi-item gamma Poisson shrinker: an empirical Bayes method using a two-component gamma mixture | Used as a synonym for any shrinkage | Method |
| EBGM | Empirical Bayes geometric mean: the posterior geometric mean of the relative reporting ratio | Interpreted as an observed-to-expected ratio | Dimensionless |
| EB05 | The fifth percentile of the posterior distribution of the relative reporting ratio | Interpreted as a p-value | Dimensionless |
| BCPNN | Bayesian confidence propagation neural network: an empirical Bayes method used by the Uppsala Monitoring Centre | Assumed to be a neural network in the modern sense | Method |
| Information component (IC) | The base-2 logarithm of the observed-to-expected ratio | Interpreted as an information-theoretic quantity | Log2 ratio |
| IC025 | The 2.5th percentile of the posterior distribution of IC | Interpreted as a significance level | Log2 ratio |
| Shrinkage | The pulling of an estimate toward a prior mean | Assumed to remove bias | Effect |
| Comparator | The set of reports against which the drug's reports are compared | Chosen by default | Definition |
| Confounding by indication | The indication for treatment is itself associated with the event | Treated as a reporting artefact | Bias |
| Co-prescription confounding | A drug commonly taken with the target drug causes the event | Missed when the pair is analysed alone | Bias |
| Notoriety bias | Increased reporting after media or regulatory attention | Mistaken for a change in risk | Bias |
| Weber effect | Reporting peaks early after launch and then declines | Mistaken for improving safety | Bias |
| Masking | A high-volume event or drug dilutes the proportions for others | Missed in aggregate analyses | Bias |
| Duplicate | The same case submitted more than once | Inflates counts | Defect |
| MedDRA | The Medical Dictionary for Regulatory Activities, a hierarchical coding dictionary | Treated as a fixed list across versions | Dictionary |
| Preferred term (PT) | The most specific MedDRA term for an event | Analysed without its hierarchy | Code |
| Standardised MedDRA Query (SMQ) | A pre-defined group of terms for a medical concept | Treated as a single PT | Query |
| WHO Drug | The dictionary for medicinal product names | Mapped loosely to local names | Dictionary |
| Signal | Information suggesting a new potentially causal association | Treated as a confirmed risk | Concept |
| Validated signal | A signal supported by case review and documentation | Treated as a confirmed risk | Concept |
| Confirmed risk | An association supported by adequate evidence, including epidemiological study | Used loosely for any signal | Concept |
| False discovery rate | The expected proportion of false positives among reported findings | Confused with the family-wise error rate | Probability |
| Tree-based scan statistic | A method that scans a hierarchy of terms for clusters and adjusts by simulation | Treated as a substitute for clinical review | Method |
| Masking-adjusted analysis | Re-analysis after removing a dominant event or drug | Rarely pre-specified | Analysis |

## 5. The mechanics, step by step

This procedure runs from a database extract to a documented triage decision. It is written for a
safety statistician working with a safety physician, and it assumes access to individual case safety
reports with coded drug and event terms. The order matters: the unit of analysis and the comparator
determine the table, the table determines the metric, and the metric determines what the review can
conclude.

### Step 1. Fix the unit of analysis and the pair definition

**Input:** a database extract of cases, with drug and event terms at the case level.

**Output:** a written definition of what counts as one observation, and a pair table whose margins are
consistent with that definition.

**Rationale:** The unit determines every count. If one report with three suspect drugs and four events
contributes twelve pairs, the margins of the two-by-two table are pair counts, not report counts, and
the expected count is computed from pair margins. If each report contributes once to each drug-event
pair, the table is said to be built at the report level and the margins are report counts. Both are
defensible; mixing them is not.

Decide and document four things: whether all drugs or only suspect drugs count; whether all events or
only serious events count; whether follow-up versions replace or add to the original; and whether
cases from the literature and from partners are included. Each choice changes the denominator.

**Edge case:** a case with the same drug and event reported twice in different terms. After coding,
both may map to the same preferred term, and the case should count once. Deduplication at the coded
level is part of this step, not a later cleaning task.

### Step 2. Select the dictionary level and the event grouping

**Input:** the coded events and the analysis question.

**Output:** an analysis level (preferred term, higher-level term, or query), with the dictionary
version recorded.

**Rationale:** A single preferred term may be too rare to analyse, while a broad group may mix
mechanisms. The choice changes sensitivity and specificity, and it must be made before seeing the
result. Recording the dictionary version matters because MedDRA terms and hierarchies change between
releases, and a result computed on one version is not reproducible on another.

| Level | What it captures | When to use | Main risk |
|---|---|---|---|
| Preferred term | The specific event as reported | Hypothesis-free screening; regulatory submissions | Too sparse for rare events; splitting of related terms |
| Higher-level term | A group of related preferred terms | When a mechanism predicts several related events | Dilution; mixing of distinct mechanisms |
| Standardised MedDRA Query | A pre-defined, validated grouping | Targeted review of a known syndrome | Narrow scope; not hypothesis-free |
| Tree-based hierarchy | All nodes of the dictionary scanned at once | Exploratory surveillance across many events | Requires simulation-based adjustment and a pre-specified tree |

The table has four rows because the choice is a trade-off between sensitivity and interpretability,
not a matter of convention. A team that screens at the preferred-term level and reviews at the
higher-level term is using both, which is usually the right answer.

### Step 3. Build and reconcile the two-by-two table

**Input:** the pair table and the chosen analysis level.

**Output:** a two-by-two table with a, b, c, d, and N, reconciled to the database total.

**Rationale:** Reconciliation catches the errors that invalidate everything downstream. The four
cells must sum to the total number of units in the analysis. A table where a + b is smaller than the
number of reports for the drug indicates that some reports have no event term, which is possible but
must be explained. A table where a + c is larger than the number of reports with the event indicates
double counting.

The reconciliation checks:

1. a + b equals the number of reports containing the drug.
2. a + c equals the number of reports containing the event.
3. a + b + c + d equals N, the total number of reports in the analysis.
4. a is not greater than either margin.
5. The count of distinct case identifiers in the pair equals a after deduplication.

**Edge case:** a report with the drug and the event but where the drug is recorded as concomitant
rather than suspect. Whether it enters a depends on the definition from Step 1. If the analysis
includes concomitant drugs, state that; the resulting metric answers a different question.

### Step 4. Choose the comparator

**Input:** the pair of interest, the clinical context, and the available data.

**Output:** a named comparator with a written justification and, where feasible, a sensitivity
analysis using a second comparator.

**Rationale:** The comparator is the single largest driver of the answer, and it is usually chosen by
default. "All other drugs" answers whether the pair is unusual in the database. "The same class"
answers whether the drug differs from its therapeutic neighbours, which is the question that matters
for a class effect. "Drugs for the same indication" answers whether the event is explained by the
underlying disease.

| Comparator | Question it answers | Strength | Main bias it introduces |
|---|---|---|---|
| All other drugs | Is this pair unusual in the database? | Stable margins; standard for screening | Class and indication effects appear as drug effects |
| Same class | Does this drug differ from its therapeutic neighbours? | Controls for mechanism shared across the class | Small comparator; class members may share the same reporting bias |
| Same indication | Is the event explained by the disease? | Controls confounding by indication at the reporting level | Hard to define; indication coding is incomplete |
| Historical period | Has reporting changed over time? | Detects Weber and notoriety effects | Confounded by co-interventions and by formulary change |

The table has four rows because a serious analysis reports at least two. A pair that is elevated
against all drugs and flat against its class is a class effect, not a drug effect, and the action is
different.

### Step 5. Compute the crude metrics with intervals

**Input:** the reconciled two-by-two table.

**Output:** PRR, ROR, their confidence intervals, the chi-square statistic, and the expected count.

**Rationale:** The crude metrics are the transparent baseline against which any shrinkage method is
judged. If a Bayesian method and the crude ratio disagree materially, the reader should be able to see
why.

```text
   PRR  = [a/(a+b)] / [c/(c+d)]
   ROR  = (a/b) / (c/d)
   SE(ln ROR) = sqrt(1/a + 1/b + 1/c + 1/d)
   95% CI     = exp(ln ROR +/- 1.96 x SE)
   E          = (a+b)(a+c)/N
   chi-square = N (ad - bc)^2 / [(a+b)(c+d)(a+c)(b+d)]
```

Report the interval, not only the point estimate. A ratio of 3.0 with an interval from 0.9 to 10.0
and a ratio of 3.0 with an interval from 2.4 to 3.7 are different findings, and the point estimate
alone conceals the difference.

**Edge case:** the chi-square with Yates' correction is conventional for small tables but conservative.
State which version is used. The interval is more informative than the test statistic in almost every
screening context.

### Step 6. Handle small counts and zero cells

**Input:** a table with a small a, or a zero cell.

**Output:** a defined rule for the estimate and its interval, applied consistently.

**Rationale:** A zero cell makes the ROR undefined or infinite, and a small a makes the interval so
wide that it carries no information. Different conventions give very different numbers, and the
convention must be fixed before the analysis rather than chosen per pair.

| Situation | Common approach | Effect | When it is acceptable |
|---|---|---|---|
| Zero in one cell | Add 0.5 to all cells (Haldane-Anscombe) | Finite estimate; biases toward 1 | Screening only; state the correction |
| a below 3 | Report the count, not the ratio | Avoids unstable ranking | Default for signal detection |
| a between 3 and 10 | Exact or shrinkage method | Wider interval; more honest | Screening with a Bayesian method |
| Very large N, small a | Poisson-based interval for the observed count | Correct coverage | When the expected count is small |
| Structural zero (drug never has the event) | Report as no evidence, not as protective | Avoids a spurious protective signal | Always |

The last row matters: a zero count is absence of evidence, not evidence of absence, and a metric that
ranks it as protective is misleading.

### Step 7. Estimate the expected count and check the independence assumption

**Input:** the table margins.

**Output:** the expected count E, the observed-to-expected ratio, and a statement of whether the
Poisson approximation is adequate.

**Rationale:** The expected count is the number of reports the pair would have if drug and event were
independent in the database. It is not a clinical expectation and it is not adjusted for age, sex, or
indication. The Poisson approximation is reasonable when E is large and the reports are independent,
and both conditions fail in practice: E is often small, and reports are correlated through
duplicate reporting, media events, and country-level reporting practice.

The practical consequence is that the Poisson-based interval understates uncertainty. That is one
reason the Bayesian methods in Step 8 are preferred for ranking: they do not rely on the Poisson
likelihood alone being correct.

### Step 8. Apply empirical Bayes shrinkage

**Input:** the observed and expected counts for all pairs in the analysis.

**Output:** a shrunken estimate for each pair, with a lower bound.

**Rationale:** Shrinkage stabilises small counts by borrowing information across pairs. The
multi-item gamma Poisson shrinker models the observed count as Poisson with mean equal to the
expected count times a relative reporting ratio, and models the ratios across the database as a
mixture of two gamma distributions. The mixture is fitted by maximum likelihood from the data, which
is what makes the method empirical rather than fully Bayesian.

The output is the empirical Bayes geometric mean, which is the posterior geometric mean of the
relative reporting ratio, and its fifth percentile. A common screening rule flags a pair when EB05
exceeds 2, meaning that the posterior probability that the ratio is at least 2 is high. The rule is a
convention, not a regulatory threshold.

The Bayesian confidence propagation neural network produces an analogous quantity on a log base-2
scale, the information component, with a lower bound conventionally required to exceed zero.

**Edge case:** a new drug with few reports. Shrinkage will pull every one of its pairs toward the prior
mean, which is near 1. The drug may therefore produce no signal at all in its first months, not
because it is safe but because there is not enough data. Shrinkage trades early sensitivity for
stability, and the trade must be understood by the reviewers who use the list.

### Step 9. Stratify for age, sex, and calendar year

**Input:** the case-level data with covariates.

**Output:** a Mantel-Haenszel or regression-adjusted estimate, or a statement that stratification was
not possible.

**Rationale:** A drug used mostly in elderly patients will appear associated with events that are
common in the elderly, because the comparator includes younger patients taking other drugs.
Stratification by age and sex removes the part of the association explained by the distribution of
these variables across drugs.

The pooled odds ratio across strata is:

```text
   ROR_MH = sum_k (a_k d_k / n_k) / sum_k (b_k c_k / n_k)
```

where k indexes the strata. The estimate is only as good as the strata: five-year age bands are
usual, and the number of strata should be small enough that each contains reports.

**Edge case:** stratification can create sparse strata and unstable estimates. Report the number of
strata, the number with zero cells, and whether the pooled estimate differs materially from the crude
one. A large change is evidence of confounding; no change is weak evidence against it.

### Step 10. Control multiplicity across pairs

**Input:** the metrics for all pairs, or a pre-specified subset.

**Output:** a ranking with an explicit multiplicity procedure, and a statement of the expected number
of false positives.

**Rationale:** Screening thousands of pairs at a nominal alpha produces false positives by
construction. The choice of procedure determines how many.

| Procedure | What it controls | Cost | Suitable for |
|---|---|---|---|
| Bonferroni | Family-wise error rate | Very low sensitivity | A small pre-specified family |
| Benjamini-Hochberg | False discovery rate | Moderate | A defined set of tested pairs |
| Empirical Bayes threshold | Posterior error given the prior | Depends on the prior fit | Large-scale screening |
| Tree-based scan with simulation | Family-wise error over the tree | Computationally heavy | Exploratory surveillance of grouped events |
| Pre-specified list only | Nothing statistical; controls by design | Misses unanticipated events | Confirmatory review of known risks |

The table has five rows because the choice is a design decision about what the analysis is for. A
confirmatory review of ten known risks and an exploratory scan of a million pairs require different
procedures, and applying the confirmatory procedure to the exploratory setting produces either no
findings or a flood, depending on the threshold.

### Step 11. Review the cases

**Input:** the ranked pairs and the narratives for the top pairs.

**Output:** a case-level assessment covering duplicates, temporality, causality, alternative
explanations, and confounding.

**Rationale:** This is where most screening hits are closed. The statistical output says the pair is
unusual; the narratives say whether the reports describe the same thing, whether the timing is
plausible, whether there is a positive dechallenge, and whether an alternative cause is present.

The review should answer five questions for each pair:

1. **Are the cases duplicates?** Check patient identifiers, dates, reporter, and narrative similarity.
2. **Is the event temporally related?** Time to onset should be consistent with the mechanism.
3. **Is there a positive dechallenge or rechallenge?** Recovery on withdrawal and recurrence on
   re-exposure are strong evidence.
4. **Is there an alternative explanation?** Concomitant drugs, the underlying disease, and procedures.
5. **Is the pattern consistent with confounding by indication or co-prescription?** Compare with
   drugs for the same indication and with drugs commonly co-prescribed.

**Edge case:** the pair is elevated only because of a co-prescribed drug. Removing cases with the
co-prescription and recomputing is a legitimate sensitivity analysis, and it should be reported as
such rather than presented as the primary result.

### Step 12. Decide and document

**Input:** the case review, the comparator analyses, and the epidemiological evidence if available.

**Output:** a classification of the pair as no action, signal, validated signal, or confirmed risk,
with the evidence, the owner, and a review date.

**Rationale:** The classification determines the action, and the four categories are not a
continuum of confidence but different evidentiary states. Collapsing them is the most consequential
error in the topic, because it converts a hypothesis into a communication.

| Classification | Evidence required | Typical action | What it is not |
|---|---|---|---|
| No action | Screening negative, or cases explained | Document and monitor | A statement that the drug is safe |
| Signal | Disproportionate reporting, or cases of concern | Validate: review narratives and comparators | A confirmed risk |
| Validated signal | Case review supports the association; alternatives considered | Assess: epidemiology, mechanism, class | A confirmed risk |
| Confirmed risk | Adequate evidence, often epidemiological, with dose and subgroup information | Label, risk management, communication, further study | A statement of causation in an individual patient |

The table has four rows because the actions differ. A signal justifies a review; a confirmed risk
justifies a label change. A team that treats the first as the fourth will withdraw useful drugs; a
team that treats the fourth as the first will leave patients exposed.

### 5.13 The reporting-bias checklist

Every analysis should carry a short statement of the biases that could produce the observed pattern.
The checklist below is a prompt, not a score.

| Bias | Signature in the data | Check |
|---|---|---|
| Notoriety | Sharp increase in reports after a media or regulatory event | Plot reports by month; check the date of publicity |
| Weber effect | Reporting peaks in the first two years after launch and then declines | Plot by time since launch, not calendar time |
| Masking | A dominant drug or event depresses the ratios for others | Recompute with the dominant pair removed |
| Duplication | Cases cluster by reporter, date, and narrative | Deduplicate and recompute |
| Channeling | The drug is used in sicker patients | Stratify by indication and baseline risk |
| Under-reporting | Very low counts across all events for a drug | Compare reporting intensity with similar products |

### 5.14 What the analysis cannot do

It cannot estimate incidence, because the denominator is unknown. It cannot establish causation,
because the reports are selected. It cannot compare risks between drugs reliably, because reporting
differs by drug age, market share, and prescriber attention. It can rank pairs for review, and that is
the whole of its claim.

## 6. Worked example

The case follows one drug-event pair from a routine monthly screen to a triage decision. The drug is
a kinase inhibitor approved for an inflammatory arthritis; the event is acute hepatic failure. The
numbers are constructed for teaching and are internally consistent, but they are not taken from a
real product.

### 6.1 The pair and the table

The database extract contains 500,000 reports after deduplication and coding to MedDRA version 27.0.
The drug appears in 1,242 reports. The preferred term "hepatic failure" appears in 1,142 reports. The
pair appears in 42.

| | Hepatic failure | All other events | Total |
|---|---:|---:|---:|
| Target drug | 42 | 1,200 | 1,242 |
| All other drugs | 1,100 | 497,658 | 498,758 |
| Total | 1,142 | 498,858 | 500,000 |

The table satisfies the reconciliation checks: the cells sum to 500,000, the row and column margins
match the drug and event counts, and 42 is below both margins.

### 6.2 The crude metrics

```text
PRR = (42 / 1,242) / (1,100 / 498,758)
    = 0.033816 / 0.0022055
    = 15.33

ROR = (42 x 497,658) / (1,200 x 1,100)
    = 20,901,636 / 1,320,000
    = 15.83

E   = (1,242 x 1,142) / 500,000
    = 1,418,364 / 500,000
    = 2.84

Observed / expected = 42 / 2.84 = 14.8
```

The chi-square statistic is 543, which is far beyond any conventional threshold. This is not a
borderline finding; it is a pair that is 15 times more reported than independence would predict.

### 6.3 The interval and what it does and does not say

```text
ln(ROR) = ln(15.83) = 2.762
SE      = sqrt(1/42 + 1/1,200 + 1/1,100 + 1/497,658)
        = sqrt(0.023810 + 0.000833 + 0.000909 + 0.000002)
        = 0.1599
95% CI  = exp(2.762 +/- 1.96 x 0.1599)
        = exp(2.449) to exp(3.075)
        = 11.58 to 21.66
```

The interval excludes 1 by a wide margin, and it would survive almost any multiplicity correction.
Neither fact makes it a risk estimate. The ROR compares the odds of the event among the drug's
reports with the odds among other drugs' reports. It does not compare exposed with unexposed
patients, because the database contains no exposure denominator. A report is not a patient, and a
proportion of reports is not an incidence.

### 6.4 The shrunken estimate

A single gamma prior with shape 0.5 and rate 0.5 is used here for illustration. Production MGPS fits
a two-component gamma mixture across all pairs; the single prior gives the same qualitative behaviour
and is easier to check by hand.

```text
Posterior shape = 0.5 + 42   = 42.5
Posterior rate  = 0.5 + 2.84 = 3.34

EBGM = exp(digamma(42.5) - ln(3.34))
     = exp(3.7377 - 1.2051)
     = exp(2.5326)
     = 12.6

EB05 = fifth percentile of Gamma(42.5, rate 3.34) = 9.7
```

The shrinkage moved the estimate from 15.8 to 12.6, a reduction of about 20 percent. That is what
shrinkage should do with 42 cases: the count is large enough that the data dominate the prior, and
the correction is modest. A pair with one case and an expected count of 0.05 would move from a crude
ratio of 20 to an EBGM near 1.9 with an EB05 near 0.3, which is not a signal.

| Pair | Crude O/E | EBGM | EB05 | Interpretation |
|---|---:|---:|---:|---|
| This pair, 42 cases, E = 2.84 | 14.8 | 12.6 | 9.7 | Signal under any threshold |
| One case, E = 0.05 | 20.0 | 1.9 | 0.3 | Not a signal; prior dominates |
| 400 cases, E = 100 | 4.0 | 4.0 | 3.6 | Signal; data dominate |

The second row is why the method exists. A crude ratio of 20 sounds alarming and is almost certainly
noise. The third row is why the method must be used carefully: a ratio of 4 with 400 cases is a
stronger finding than a ratio of 20 with one case, and only a precision-aware method ranks them the
right way round.

### 6.5 Diagnostic 1: the duplicates

The case review finds that 9 of the 42 reports share a patient identifier, event date, and reporter
with another report in the set. They are follow-up versions submitted by the same company under
different report numbers. After deduplication, a = 33 and b = 1,191.

```text
PRR = (33 / 1,224) / (1,100 / 498,758) = 12.2
ROR = (33 x 497,658) / (1,191 x 1,100) = 12.5
```

The signal survives, but the estimate falls by about 21 percent. The diagnostic matters for a
different reason: it reveals that the intake process is not linking follow-ups, which means every
pair involving this product is inflated, and the inflation is selective because serious events
generate more follow-up correspondence than mild ones.

Two checks would have caught it earlier. A duplicate scan on patient identifier, event date, and
reporter initials takes minutes and should run before every screening cycle. A comparison of the
report count against the case count for each product shows the gap directly.

### 6.6 Diagnostic 2: the signal is confined to a subgroup

The drug is indicated for inflammatory arthritis, and methotrexate is co-prescribed in a large share
of patients. Methotrexate is itself hepatotoxic. Stratifying the drug's reports by methotrexate
co-prescription gives the pattern below.

| Subgroup | Reports with drug | Reports with event | Event proportion | ROR against the same comparator |
|---|---:|---:|---:|---:|
| Methotrexate co-prescribed | 742 | 40 | 5.4 percent | 25.8 |
| No methotrexate | 500 | 2 | 0.4 percent | 1.82 |

The comparator proportion is 0.22 percent in both rows, held constant for comparability. The
contrast is stark: the association is concentrated entirely in the co-prescribed subgroup, and the
estimate in the other subgroup is compatible with no association given its width.

This is confounding by co-prescription, and it changes the interpretation from "the drug causes
hepatic failure" to "the combination may cause hepatic failure, or methotrexate alone does". The
practical implication is a drug-drug interaction question, which is answered by a different study
than a general incidence study. It also means that a label warning about the drug alone would be
misdirected.

The diagnostic is not subtle: it requires asking whether a common co-medication is itself associated
with the event, and then splitting the reports. The reason it is skipped is that the pair-level
analysis never looks at the case level until after the decision has been made.

### 6.7 The varied assumption: changing the comparator

The analysis so far used all other drugs as the comparator. Suppose the team repeats it against the
same therapeutic class, which is known to carry a hepatotoxicity warning.

```text
Target drug:      42 events in 1,242 reports
Same class:      180 events in 8,180 reports

PRR = (42 / 1,242) / (180 / 8,180) = 0.033816 / 0.022005 = 1.54
ROR = (42 x 8,000) / (1,200 x 180)  = 336,000 / 216,000   = 1.56
95% CI for ROR: 1.11 to 2.19
```

The conclusion changes from "15 times disproportionate against the database" to "about 1.5 times
disproportionate against its class, with an interval that excludes 1 narrowly". The first number
describes the drug's place in the database; the second describes its place among its peers. For a
label decision, the second is the relevant comparison, and it supports a class warning rather than a
drug-specific one.

This is the most important arithmetic in the topic. The same 42 cases, the same event, the same
database, and a different comparator produce a factor-of-ten change in the headline number. Neither
analysis is wrong. They answer different questions, and the analyst who reports only one has chosen
the answer without saying so.

### 6.8 What the case shows

Four points carry beyond this example.

First, the crude metric was real and large, and it still did not identify a drug-specific risk. A
large number is not a conclusion.

Second, the diagnostics that mattered were not statistical. Duplicate matching and a subgroup split
required access to the case data and a clinical question, not a better algorithm.

Third, shrinkage changed the estimate by 20 percent and the interpretation by nothing. It is a
stability tool, not a substitute for thinking about confounding.

Fourth, the comparator determined the answer. Any report of a disproportionality metric that does not
name the comparator and show at least one alternative is incomplete by construction.

## 7. Code, query, or calculation

The script reproduces every number in Section 6: the two-by-two metrics, the interval, the expected
count, the empirical Bayes estimates with an exact gamma posterior quantile, the stratified
comparison, and a multiplicity simulation. It uses only the standard library.

Assumptions: reports are independent units after deduplication; the expected count is computed from
the table margins; the prior is a single gamma distribution with shape and rate 0.5, which is a
teaching simplification of the two-component mixture used in production MGPS; and the Poisson
sampler is adequate for small means.

```python
"""Disproportionality metrics, empirical Bayes shrinkage, and a multiplicity check."""
import math
import random


def digamma(x):
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f / 240)))


def _gser(s, z):
    term, total = 1.0 / s, 1.0 / s
    for n in range(1, 300):
        term *= z / (s + n)
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-z + s * math.log(z) - math.lgamma(s))


def _gcf(s, z):
    tiny = 1e-300
    b, c, d = z + 1 - s, 1 / tiny, 1 / (z + 1 - s)
    h = d
    for i in range(1, 300):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1 - math.exp(-z + s * math.log(z) - math.lgamma(s)) * h


def gamma_cdf(x, shape, rate):
    if x <= 0:
        return 0.0
    z = rate * x
    return _gser(shape, z) if z < shape + 1 else _gcf(shape, z)


def gamma_quantile(p, shape, rate):
    lo, hi = 0.0, (shape / rate) * 10 + 10
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gamma_cdf(mid, shape, rate) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def metrics(a, b, c, d):
    n = a + b + c + d
    prr = (a / (a + b)) / (c / (c + d))
    ror = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo, hi = math.exp(math.log(ror) - 1.96 * se), math.exp(math.log(ror) + 1.96 * se)
    expected = (a + b) * (a + c) / n
    chi2 = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    return dict(prr=prr, ror=ror, lo=lo, hi=hi, e=expected, chi2=chi2, n=n)


def shrink(a, expected, alpha=0.5, beta=0.5):
    shape, rate = alpha + a, beta + expected
    ebgm = math.exp(digamma(shape) - math.log(rate))
    return ebgm, gamma_quantile(0.05, shape, rate)


def multiplicity(n_pairs=10000, e=2.84, seed=11):
    """Under the null, how many pairs clear a 95 percent lower bound?"""
    rng = random.Random(seed)
    flags = 0
    for _ in range(n_pairs):
        # Poisson(e) by cumulative inversion
        k, pmf, cdf, u = 0, math.exp(-e), math.exp(-e), rng.random()
        while u > cdf and k < 60:
            k += 1
            pmf *= e / k
            cdf += pmf
        a = k
        if a >= 2 and (a - 1.645 * math.sqrt(a)) / e > 1:
            flags += 1
    return flags, n_pairs


def main():
    a, b, c, d = 42, 1200, 1100, 497658
    m = metrics(a, b, c, d)
    print("TWO-BY-TWO TABLE")
    print(f"  a={a}  b={b}  c={c}  d={d}  N={m['n']}")
    print("\nCRUDE METRICS")
    print(f"  PRR                 {m['prr']:.2f}")
    print(f"  ROR                 {m['ror']:.2f}")
    print(f"  95 percent CI       {m['lo']:.2f} to {m['hi']:.2f}")
    print(f"  expected count      {m['e']:.2f}")
    print(f"  observed/expected   {a / m['e']:.1f}")
    print(f"  chi-square          {m['chi2']:.0f}")
    ebgm, eb05 = shrink(a, m['e'])
    print("\nEMPIRICAL BAYES (single Gamma(0.5, 0.5) prior)")
    print(f"  EBGM                {ebgm:.2f}")
    print(f"  EB05                {eb05:.2f}")
    for label, aa, ee in (("one case, E=0.05", 1, 0.05), ("400 cases, E=100", 400, 100.0)):
        g, l = shrink(aa, ee)
        print(f"  {label:<20} EBGM {g:5.2f}  EB05 {l:5.2f}")
    print("\nSTRATIFIED ANALYSIS (comparator proportion held at 0.22 percent)")
    for label, aa, bb in (("methotrexate co-prescribed", 40, 702), ("no methotrexate", 2, 498)):
        ror = (aa / bb) / 0.0022055
        print(f"  {label:<28} a={aa:3d}  b={bb:3d}  ROR {ror:6.2f}")
    print("\nCOMPARATOR SENSITIVITY")
    print(f"  against all other drugs     ROR {m['ror']:.2f}")
    class_ror = (42 * 8000) / (1200 * 180)
    print(f"  against same therapeutic class  ROR {class_ror:.2f}")
    flags, total = multiplicity()
    print(f"\nMULTIPLICITY UNDER THE NULL ({total} simulated pairs, E={2.84})")
    print(f"  pairs with a lower bound above 1: {flags} ({100 * flags / total:.1f} percent)")


if __name__ == "__main__":
    main()
```

Output on the authoring machine, Python 3.12:

```text
TWO-BY-TWO TABLE
  a=42  b=1200  c=1100  d=497658  N=500000

CRUDE METRICS
  PRR                 15.33
  ROR                 15.83
  95 percent CI       11.58 to 21.66
  expected count      2.84
  observed/expected   14.8
  chi-square          543

EMPIRICAL BAYES (single Gamma(0.5, 0.5) prior)
  EBGM                12.59
  EB05                9.70
  one case, E=0.05     EBGM  1.89  EB05  0.32
  400 cases, E=100     EBGM  3.98  EB05  3.66

STRATIFIED ANALYSIS (comparator proportion held at 0.22 percent)
  methotrexate co-prescribed   a= 40  b=702  ROR  25.84
  no methotrexate              a=  2  b=498  ROR   1.82

COMPARATOR SENSITIVITY
  against all other drugs     ROR 15.83
  against same therapeutic class  ROR 1.56

MULTIPLICITY UNDER THE NULL (10000 simulated pairs, E=2.84)
  pairs with a lower bound above 1: 85 (0.8 percent)
```

### 7.1 What the output shows

The crude and shrunken estimates agree closely because 42 cases is a large count for this database.
The shrinkage matters at the other end of the table: one case with an expected count of 0.05 gives a
crude ratio of 20 and an EBGM of 1.89 with an EB05 of 0.32. Any ranking that puts the second above the
first is ranking noise above signal.

The multiplicity simulation is the most useful number for planning. Under the null, 0.8 percent of
pairs clear a crude lower bound above 1 at this expected count. In a database with 100,000 analysed
pairs, that is about 800 false flags per cycle before any clinical review. A review capacity of fifty
pairs a month cannot absorb them, which is why the threshold, the shrinkage, and the pre-specified
subsets do the work that a p-value cannot.

The comparator rows show the sensitivity directly: 15.83 against all drugs, 1.56 against the class.
Reporting one without the other is a choice disguised as a result.

### 7.2 Limits of this calculation

The script computes what it says and no more. It does not adjust for age, sex, or calendar year; the
stratified rows hold the comparator proportion constant, which is a teaching simplification rather
than a Mantel-Haenszel estimate. It does not implement the two-component mixture that production
MGPS fits, so the EBGM values are illustrative of the shrinkage behaviour rather than equal to a
validated implementation's output. It does not test causality, and no output from it should be
described as a risk.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and describes published frameworks at the access date, 27
> October 2026. It is not legal or regulatory advice.

### 8.1 The databases

| Database | Custodian | Approximate size | Access | Character |
|---|---|---|---|---|
| FAERS | US Food and Drug Administration | About 26 million reports at end-2025 by one published estimate | Public quarterly extracts | Domestic and foreign reports; duplicate reports across quarters |
| EudraVigilance | European Medicines Agency | Not stated here | Restricted; aggregated public access | EU cases plus serious non-EU cases |
| VigiBase | Uppsala Monitoring Centre, for the World Health Organization | More than 40 million reports from over 180 countries, February 2025 | Restricted; national centres | Global, with country-level heterogeneity |
| VAERS | US Centers for Disease Control and Prevention and FDA | Several million reports | Public | Vaccine-specific; stimulated reporting is substantial |

The sizes are not comparable across databases because of deduplication rules, reporting periods, and
what counts as a report. A count quoted without the date, the extract, and the deduplication rule is
not reproducible. FAERS, for example, publishes quarterly extracts in which the same case can appear
in more than one file, so a naive concatenation double counts.

### 8.2 Coding and transmission standards

| Standard | What it governs | Binding or convention | Consequence for analysis |
|---|---|---|---|
| MedDRA | Coding of adverse events, with a five-level hierarchy and standardised queries | Required in most jurisdictions for post-authorisation reporting | The version determines which terms exist; a version change breaks reproducibility |
| WHO Drug Dictionary | Coding of medicinal products | Required in many jurisdictions | Product names map imperfectly to active ingredients |
| ICH E2B(R3) | Electronic transmission of individual case safety reports | Binding where adopted | Determines the fields available for analysis |
| ICH E2D | Post-approval safety data management | Guidance | Defines what must be collected and followed up |
| ICH E2E | Pharmacovigilance planning | Guidance | Frames the risk management plan |
| GVP Module IX and Addendum I | Signal management in the EU, including methodological aspects | Binding in the EU through the GVP structure | Defines the signal management process, not a numeric threshold |
| 21 CFR 314.80 | Post-marketing reporting for US drug products | Binding in the US | Defines reporting timelines and periodic report content |

Two points matter for the statistics.

First, **the dictionary version is part of the result**. MedDRA releases twice a year, and terms are
added, split, and moved. A signal detected in version 26.0 may not reproduce in version 27.0 because a
term was split into two. Any screening report should state the version, and a re-run after an upgrade
is a new analysis, not a refresh.

Second, **the regulatory threshold is not a statistical threshold**. The European Medicines Agency's
GVP Module IX Addendum I describes a signal detection algorithm that combines a disproportionality
statistic, typically the lower bound of a 95 percent interval, with a minimum number of reports. It
does not mandate a single numeric cut-off, and it notes that the choice of threshold drives
performance more than the choice of statistic. National competent authorities and marketing
authorisation holders set their own thresholds in their procedures.

### 8.3 What the regulations do not provide

They do not provide an exposure denominator, so incidence cannot be computed from spontaneous
reports. They do not require a particular disproportionality method, so two sponsors can use
different metrics and both be compliant. They do not define a validated signal in a way that removes
judgement, and they do not accept a disproportionality finding as evidence of causation. The
statistical work sits inside a process that is regulated; the statistics themselves are a
professional judgement that the process must document.

### 8.4 The data-quality realities

Four properties of the data affect every metric and are worth stating explicitly in any analysis
plan.

1. **Reports are not patients.** One patient can generate several reports, and one report can
   describe several events.
2. **The population at risk is unknown.** Market sales give a rough exposure estimate for some
   products, but not the duration of use or the indication mix.
3. **Reporting intensity varies.** By country, by reporter type, by event seriousness, and by time
   since launch.
4. **Coding is imperfect.** Free text becomes a preferred term through a human or automated process,
   and the same narrative can be coded differently by two coders.

A metric computed on these data is a description of the reporting stream. That is a useful thing to
describe, and it is not a description of risk.

## 9. Industry practice and case evidence

### 9.1 Three cases that shaped the field

| Product | Event | What happened | Detection route | Lesson |
|---|---|---|---|---|
| Rofecoxib | Myocardial infarction | Withdrawn September 2004 after the APPROVe trial; a later estimate attributed 88,000 to 140,000 excess serious coronary heart disease cases in the US | Randomised trial, not spontaneous reporting | Spontaneous reporting can miss a common event with a modest relative increase |
| Cerivastatin | Rhabdomyolysis | Withdrawn August 2001 | Case reports and regulatory review | Rare, severe events with a distinctive presentation are the classic spontaneous-reporting signal |
| Rosiglitazone | Cardiovascular risk | Meta-analysis in 2007 raised concern; regulatory restrictions followed in 2010 | Published meta-analysis of trials | Aggregate evidence can emerge outside the sponsor's safety system |

The rofecoxib case is the most instructive and the most often misread. The event was common in the
population, the relative increase was modest, and the reporting system did not produce a clear signal
before the trial did. A method tuned to detect rare events with large ratios will not detect a common
event with a ratio near 1.5, and no amount of statistical refinement changes that. The complement is
that a method tuned to detect common events with small ratios will drown in noise from rare ones.

### 9.2 Documented practice and inference

| Claim | Status | Basis |
|---|---|---|
| Marketing authorisation holders in the EU must operate a signal management process | Documented | GVP Module IX |
| EudraVigilance screening uses a lower-bound threshold with a minimum case count | Documented | GVP Module IX Addendum I describes the algorithm |
| The FDA uses MGPS in its internal screening | Documented | Agency publications and the Empirica Signal documentation |
| The Uppsala Monitoring Centre uses BCPNN | Documented | Published method papers and UMC documentation |
| Most companies screen at the preferred-term level and review at a higher level | Inference | Consistent with published SOP descriptions; not auditable |
| A large share of signals are closed at case review | Inference | Stated in conference material; no public denominator |
| Machine learning rerankers are widely deployed | Not established | Vendor claims; little peer-reviewed evaluation |

The distinction matters because vendor material often presents the inference rows as documented
facts. A claim that "AI reduces false signals by 70 percent" requires a benchmark, a gold standard,
and a reproducible evaluation, and the public record rarely contains all three.

### 9.3 The organisational reality

Signal detection is a process with three queues: the automated screen, the case review, and the
assessment. Most organisations are efficient at the first and bottlenecked at the second, because
case review requires clinicians and the screen produces more candidates than clinicians can read.

The responses are predictable. Raise the threshold until the queue fits, which reduces sensitivity.
Pre-specify a list of events of special interest and review those regardless of the metric, which
restores sensitivity for known concerns at the cost of missing the unanticipated. Add a second,
independent method and review only the pairs that both flag, which reduces false positives but also
reduces sensitivity. None of these is wrong; each is a trade that should be made explicitly and
reviewed periodically.

The failure that recurs is a threshold set by tradition and never revisited. A threshold appropriate
for a database of 100,000 reports in 2005 is not appropriate for a database of 26 million reports in
2026, because the number of pairs and the reporting mix have changed. The threshold is a policy
choice, and it should be reviewed with the same discipline as the metric.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Duplicates counted as cases | Report count exceeds case count for the product | Follow-ups submitted under new numbers | Inflated numerator, selective for serious events | Deduplicate before every cycle; audit the gap |
| Comparator chosen by default | Methods section says "all other drugs" and nothing else | No argument about class or indication | Class effects reported as drug effects | Report at least two comparators |
| Term splitting | The signal disappears when the query is broadened | Related events coded to different preferred terms | Missed syndrome; false reassurance | Screen at the preferred term, review at the higher level |
| Ranking by crude ratio | Top of the list is dominated by one- and two-case pairs | Small counts produce extreme ratios | Review capacity wasted on noise | Shrinkage and a minimum case count |
| Notoriety effect read as risk | Reports spike after a media story or communication | Stimulated reporting | False signal; unnecessary regulatory action | Plot reports by month; compare with the publicity date |
| Weber effect read as safety | Reporting falls after year two | Reporting intensity declines with familiarity | A real risk appears to resolve | Plot by time since launch, not calendar time |
| Masking | A dominant event depresses every other ratio | Denominator inflation by one high-volume pair | True signals suppressed | Recompute with the dominant pair removed |
| Confounding by indication | The event is a known consequence of the disease | The drug is used in the population at risk | Drug blamed for disease effect | Compare with drugs for the same indication |
| Co-prescription confounding | Signal confined to one co-medication subgroup | A interacting or independently toxic concomitant drug | Wrong target for the warning | Stratify by common co-medications |
| Unadjusted multiplicity | Hundreds of pairs meet the threshold each cycle | Testing thousands of pairs at a nominal alpha | Review capacity exhausted | Pre-specify, shrink, or control the false discovery rate |
| Threshold drift | The same threshold is used as the database grows tenfold | Fixed cut-off against a changing pair count | Rising false-positive volume | Review the threshold against the queue capacity |
| Dictionary version change | Results do not reproduce after an upgrade | Terms added, split, or moved | Apparent disappearance or emergence of signals | Record the version; re-run as a new analysis |
| Signal treated as confirmed risk | A communication is drafted at the screening stage | Classification collapse | Withdrawal of a useful drug; litigation | Four-state classification with defined evidence |
| Under-reporting read as safety | Very few reports across all events | Reporting failure, not low risk | False reassurance | Compare reporting intensity with similar products |

The first four rows account for most spurious signals in practice. The middle rows are the ones that
require clinical input, and the last three are governance failures that no statistical method can
repair.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Reports are independent units | The two-by-two table and the Poisson model assume it | Duplicates, clusters from one prescriber, litigation campaigns | Deduplicate; check the distribution of reports per reporter |
| Reporting intensity is comparable across drugs | The comparison of proportions requires it | Drug age, market share, country mix, media attention | Stratify by year since launch and by country |
| The comparator is exchangeable | The ratio is interpretable as a contrast | Class or indication effects | Repeat with class and indication comparators |
| Coding is consistent across drugs and time | Term-level comparisons require it | Dictionary version change, coder variation | Record the version; audit a sample of narratives |
| The event term captures one clinical entity | The ratio describes a coherent event | A term that mixes severities or mechanisms | Review at the higher-level term and the query level |
| Observed counts are Poisson | Intervals and expected counts rely on it | Overdispersion from clustering | Compare observed and expected variance across strata |
| The empirical prior is appropriate for the pair | Shrinkage borrows strength from the database | A pair from a different reporting era or region | Check sensitivity to the prior; compare with the crude ratio |
| A signal is a hypothesis, not a risk | The entire process depends on the distinction | Organisational pressure to act quickly | Review the classification of recent signals |
| Case review is representative of the reports | Conclusions from narratives generalise | Reviewers see a non-random subset | Compare reviewed and unreviewed pairs on key variables |
| The database is stable over the analysis window | Trend comparisons require it | Reporting-system changes, mergers, new mandatory reporting | Plot counts by month; annotate system changes |

The first assumption is the most commonly violated and the easiest to test. The seventh is the most
subtle: shrinkage assumes that the distribution of true ratios is similar across the pairs that share
a prior, and a pair from a newly launched product in a country with different reporting habits may
not belong to that distribution at all.

The overall limit is structural. No method applied to spontaneous reports can produce an incidence, a
relative risk, or a causal conclusion. The methods in this topic rank pairs for review. Everything
after the ranking is clinical and epidemiological work, and the statistics cannot substitute for it.

## 12. Exercises

**Exercise 1 — Build and reconcile a table (application, 20 minutes).**
Deliverable: a two-by-two table for a drug-event pair of your choice, with all four cells, both
margins, N, and the five reconciliation checks.
Acceptance criteria: cells sum to N; margins match the stated drug and event counts; the unit of
analysis is defined in one sentence.
Time: 20 minutes.

**Exercise 2 — Metrics by hand (application, 30 minutes).**
Deliverable: PRR, ROR, the 95 percent interval, the expected count, and the chi-square for a = 15,
b = 985, c = 800, d = 400,000, with the arithmetic shown.
Acceptance criteria: each formula written before substitution; interval computed on the log scale;
interpretation limited to disproportionality.
Time: 30 minutes.

**Exercise 3 — Shrinkage (application, 25 minutes).**
Deliverable: a comparison of the crude and empirical Bayes estimates for a = 1 with E = 0.05 and for
a = 400 with E = 100, with an explanation of the difference.
Acceptance criteria: both pairs computed; the role of the prior stated; a ranking of the two pairs
justified.
Time: 25 minutes.

**Exercise 4 — Comparator sensitivity (critique, 35 minutes).**
Deliverable: the ROR for the worked example against all drugs, the class, and the indication, with a
recommendation for which to report.
Acceptance criteria: all three computed or estimated; the change in conclusion stated; the choice
justified on confounding grounds rather than convenience.
Time: 35 minutes.

**Exercise 5 — Duplicate audit (application, 30 minutes).**
Deliverable: a matching rule for duplicate detection and the effect of removing duplicates on the
worked example's PRR and ROR.
Acceptance criteria: matching variables named; tolerance stated; recomputed metrics shown; the
selectivity of duplication discussed.
Time: 30 minutes.

**Exercise 6 — Confounding by co-prescription (design, 40 minutes).**
Deliverable: a stratified analysis plan for a drug commonly co-prescribed with a hepatotoxic agent.
Acceptance criteria: strata defined; a Mantel-Haenszel or regression estimator named; sparse strata
addressed; a sensitivity analysis without the co-prescribed cases included.
Time: 40 minutes.

**Exercise 7 — Multiplicity policy (design, 35 minutes).**
Deliverable: a one-page screening policy for a database of 20 million reports and a review capacity of
60 pairs per month.
Acceptance criteria: a threshold rule; a shrinkage method; a pre-specified list of events of special
interest; an estimate of the expected number of false flags per cycle.
Time: 35 minutes.

**Exercise 8 — Triage memo (design, 50 minutes).**
Deliverable: a two-page signal triage note for the worked example's pair, classifying it as no action,
signal, validated signal, or confirmed risk.
Acceptance criteria: classification matches the evidence; at least three alternative explanations
addressed; the memo states what would change the conclusion; an owner and review date are named.
Time: 50 minutes.

## 13. Self-check answers

**Exercise 1.** The four cells must sum to N, a + b must equal the number of reports containing the
drug, a + c the number containing the event, and a must not exceed either margin. A table that fails
one of these checks has double counting, missing event terms, or a mixed unit of analysis.

**Exercise 2.** PRR = (15/1,000)/(800/400,800) = 0.0150/0.001996 = 7.52. ROR = (15 x 400,000)/(985 x
800) = 6,000,000/788,000 = 7.61. ln(ROR) = 2.030; SE = sqrt(0.06667 + 0.001015 + 0.00125 + 0.0000025)
= 0.2626; the interval is exp(2.030 +/- 0.515) = 4.55 to 12.74. E = 1,000 x 815/400,800 = 2.03. The
pair is disproportionate; nothing about risk follows.

**Exercise 3.** For a = 1, E = 0.05, the crude ratio is 20.0 and the EBGM is 1.89 with EB05 0.32: the
prior dominates. For a = 400, E = 100, the crude ratio is 4.0 and the EBGM is 3.98 with EB05 3.66: the
data dominate. The second pair is the stronger finding despite the smaller ratio, because 400 cases
is evidence and one case is not.

**Exercise 4.** Against all drugs the ROR is 15.83, against the class 1.56, and against the indication
it lies between depending on the indication's baseline. The conclusion changes from a drug-specific
signal to a class effect. Report the class comparison as primary for a label decision and the
all-drug comparison as context, and state both.

**Exercise 5.** Match on patient identifier, event date, reporter, and narrative similarity, with a
one-day tolerance on dates and a check on event coding. Removing nine duplicates from the worked
example reduces a from 42 to 33 and the PRR from 15.33 to about 12.2. Duplication is selective for
serious events, so the effect is larger for serious pairs than for the database as a whole.

**Exercise 6 rubric.** Full marks require named strata, a pooled estimator, treatment of sparse
strata, and a sensitivity analysis excluding co-prescribed cases. Deduct marks for stratification
that produces cells too small to estimate anything, and for reporting the pooled estimate without the
stratum-specific ones.

**Exercise 7 rubric.** Full marks require a threshold rule tied to the review capacity, a shrinkage
method, a pre-specified event list, and a quantified false-flag expectation. Deduct marks for a policy
that produces more candidates than the queue can absorb, or one that cannot state how many false
flags it expects per cycle.

**Exercise 8 rubric.** Full marks require a classification consistent with the evidence, at least
three alternative explanations, a explicit falsification condition, and an owner with a date. Deduct
marks for calling the pair a confirmed risk on disproportionality alone, or for recommending a label
change without an epidemiological argument.

## 14. Interview questions

**1. What does a PRR of 5 mean?** Strong: reporting of the event is five times more frequent among the
drug's reports than among comparator reports; a screening metric with no exposure denominator, not a
risk ratio. Weak: says the drug increases risk fivefold.

**2. Why can you not compute incidence from FAERS?** Strong: the denominator of exposed patients and
the proportion of events reported are both unknown; reports are not patients; reporting varies by
severity, publicity, and country. Weak: says the data are "dirty" without the denominator argument.

**3. Why does empirical Bayes help?** Strong: it stabilises small counts by borrowing strength across
pairs, so a one-case pair with a ratio of 20 is pulled toward the prior rather than topping the list.
Weak: says it "removes false positives", which overstates what a prior can do.

**4. When would you use the class comparator?** Strong: when the question is whether the drug differs
from its therapeutic neighbours, which is the relevant question for a class effect and for labelling.
Weak: uses it only when the all-drug result looks inconvenient.

**5. A pair is elevated against all drugs and flat against its class. What does that mean?** Strong: a
class effect; the drug shares the association with its peers, so the action is a class-wide one.
Weak: concludes the drug is safe because the class result is null.

**6. How do you handle thousands of comparisons?** Strong: pre-specify the family, use shrinkage or a
false-discovery-rate procedure, estimate the expected false flags, and treat the output as a ranking
for review. Weak: applies Bonferroni to millions of pairs and reports nothing.

**7. What is the difference between a signal and a confirmed risk?** Strong: a signal is information
suggesting an association; a confirmed risk requires case review plus epidemiological or mechanistic
evidence, and only then does a label change follow. Weak: treats them as synonyms.

**8. How would you detect notoriety bias?** Strong: plot reports by month against the date of the
publicity or regulatory communication, and check whether the pair's rise coincides. Weak: asserts
that reporting is biased without a time plot.

**9. What would make you distrust a vendor's signal-detection benchmark?** Strong: no gold standard,
no held-out period, no description of the comparator, and a metric that rewards finding known
associations rather than unknown ones. Weak: distrusts it because it is commercial, without a
technical reason.

## 15. Key terms

- **BCPNN** — Bayesian confidence propagation neural network; the Uppsala Monitoring Centre method.
- **Chi-square** — a test of independence in the two-by-two table, not a measure of effect.
- **Comparator** — the reports against which the target drug's reports are compared.
- **Confounding by indication** — the treated disease, not the drug, explains the event.
- **Disproportionality** — over-representation of an event among a drug's reports.
- **Duplicate** — the same case submitted more than once.
- **EB05** — fifth percentile of the posterior relative reporting ratio.
- **EBGM** — empirical Bayes geometric mean of the relative reporting ratio.
- **Empirical Bayes** — estimation of a prior from the data, with shrinkage toward it.
- **Expected count** — the count expected if drug and event were independent.
- **False discovery rate** — the expected proportion of false positives among reported findings.
- **IC025** — lower bound of the information component, used by the Uppsala Monitoring Centre.
- **Information component** — log base-2 observed-to-expected ratio.
- **Masking** — dilution of other pairs by a dominant drug or event.
- **MedDRA** — the medical dictionary for regulatory activities.
- **MGPS** — multi-item gamma Poisson shrinker.
- **Notoriety bias** — increased reporting after publicity.
- **PRR** — proportional reporting ratio.
- **Preferred term** — the most specific MedDRA term for an event.
- **ROR** — reporting odds ratio.
- **Signal** — information suggesting a new potentially causal association.
- **SMQ** — standardised MedDRA query, a pre-defined group of terms.
- **Shrinkage** — pulling an estimate toward a prior mean.
- **Weber effect** — reporting peaks early after launch and then declines.

## 16. Sources

- Evans SJW, Waller PC, Davis S. Use of proportional reporting ratios for signal generation from
  spontaneous adverse drug reaction reports. *Pharmacoepidemiology and Drug Safety*, 2001;10:483-486.
- DuMouchel W. Bayesian data mining in large frequency tables, with an application to the FDA
  spontaneous reporting system. *The American Statistician*, 1999;53(3):177-190.
- Bate A, Lindquist M, Edwards IR, et al. A Bayesian neural network method for adverse drug reaction
  signal generation. *European Journal of Clinical Pharmacology*, 1998;54(4):315-321.
- Norén GN, Bate A, Orre R, Edwards IR. Extending the methods used to screen the WHO drug safety
  database towards analysis of complex associations and improved accuracy for rare events.
  *Statistics in Medicine*, 2006;25(21):3740-3757.
- Szarfman A, Machado SG, O'Neill RT. Use of screening algorithms and computer systems to signal
  higher-than-expected combinations of drugs and events. *Drug Safety*, 2002;25(6):381-392.
- European Medicines Agency. *Guideline on Good Pharmacovigilance Practices, Module IX: Signal
  Management*, Rev 1, 2017, and *Addendum I: Methodological Aspects of Signal Detection from
  Spontaneous Reports*, 2017.
- US Food and Drug Administration. *Good Pharmacovigilance Practices and Pharmacoepidemiologic
  Assessment*. Guidance for industry, 2005.
- US Code of Federal Regulations, Title 21, Part 314.80: Postmarketing reporting of adverse drug
  experiences.
- International Council for Harmonisation. *E2D: Post-Approval Safety Data Management*. 2003.
- Graham DJ, Campen D, Hui R, et al. Risk of acute myocardial infarction and sudden cardiac death in
  patients treated with cyclo-oxygenase 2 selective and non-selective non-steroidal anti-inflammatory
  drugs. *The Lancet*, 2005;365:475-481.
- Nissen SE, Wolski K. Effect of rosiglitazone on the risk of myocardial infarction and death from
  cardiovascular causes. *New England Journal of Medicine*, 2007;356:2457-2471.
- Kulldorff M, Fang Z, Walsh SJ. A tree-based scan statistic for database disease surveillance.
  *Biometrics*, 2003;59(2):323-331.
- Harpaz R, DuMouchel W, Shah NH, Madigan D, Ryan P, Friedman C. Novel data-mining methodologies for
  adverse drug event discovery and analysis. *Clinical Pharmacology and Therapeutics*,
  2012;91(6):1010-1021.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`11_digital-twins-in-clinical-trials.md`](11_digital-twins-in-clinical-trials.md)
- Next topic: [`13_pass-and-active-safety-surveillance.md`](13_pass-and-active-safety-surveillance.md)
- Related topic: [`06_safety-and-pharmacovigilance.md`](06_safety-and-pharmacovigilance.md)
- Related topic: [`07_rwe-in-pharma.md`](07_rwe-in-pharma.md)
- Survey file: [`../../course/13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
