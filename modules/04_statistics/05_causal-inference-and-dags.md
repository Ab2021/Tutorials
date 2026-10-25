# Causal Inference and DAGs

> Module 04 — Statistics and Causal Inference
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02
> Version 2026.3

---

## 1. Why this topic exists

Every comparative question in biomedical analytics is a causal question in disguise. "Does the program reduce readmissions?" "Does the drug prevent stroke?" "Does the alert shorten time to antibiotics?" The data alone cannot answer them. The data contain associations, and an association becomes a causal claim only when a set of assumptions links the observed comparison to the counterfactual one.

Those assumptions are usually left implicit. An analyst adjusts for every variable in the table, or for the variables that differ at baseline, or for the variables that predict the outcome. Each of these rules can produce a biased estimate, and none of them can be checked against the data. The result is a number that looks like an effect and is a mixture of the effect and the selection process.

Three examples show the cost:

1. **Confounding by indication.** A statin looks harmful in a crude comparison because patients with prior myocardial infarction are both more likely to receive it and more likely to have a stroke. Adjusting for prior infarction reverses the sign. The crude number is not a weak version of the causal effect; it is a different quantity, dominated by who received the drug.
2. **Collider bias.** A program looks effective when the analysis adjusts for "hospitalization during follow-up," a variable caused by both the program and the outcome. Conditioning on the collider creates an association where none exists. The model fits better and the estimate is further from the truth.
3. **Mediator adjustment.** A program works by increasing adherence. Adjusting for adherence removes the mechanism and reports a direct effect near zero. The program may still be effective, but the analysis has answered a different question.

A directed acyclic graph (DAG) is a tool for making these assumptions visible before the analysis. It is a graph with one node per variable and one directed edge per assumed direct causal effect, with no cycles. It encodes what the analyst believes about the data-generating process. From the graph, the backdoor criterion identifies the adjustment sets that would identify the total effect, and it identifies the variables that must not be adjusted for.

The DAG does not validate itself. Two analysts can draw different graphs for the same question, and the data cannot adjudicate between them. That is the point: the graph forces the disagreement into the open, where it can be resolved with subject-matter evidence, a sensitivity analysis, or a different study design.

Regulators have moved in the same direction. The International Council for Harmonisation (ICH) E9(R1) addendum, adopted in 2019, requires the estimand to be defined before analysis, including the population, the treatment contrast, and the handling of intercurrent events. The United States Food and Drug Administration (FDA) real-world evidence guidance, finalized in August 2023, emphasizes pre-specified study design and transparency about assumptions. Neither document mandates a DAG, but both require the causal question and the identification strategy to be stated.

This topic builds the vocabulary and the mechanics: nodes, paths, chains, forks, colliders, d-separation, the backdoor criterion, and the adjustment sets that follow. It also builds the discipline of drawing the graph before touching the data.

> **Note:** A DAG is a model of assumptions, not a discovery tool. Running a structure-learning algorithm on observational data can suggest a graph, but it cannot establish the direction of an edge or the absence of an unmeasured confounder. Use it to generate hypotheses, not to justify an adjustment set.

## 2. Learning outcomes

By the end of this topic you can:

1. **Translate** a comparative question into a DAG with a defined treatment, outcome, and time zero.
2. **Classify** every variable in a proposed analysis as a confounder, mediator, collider, proxy, or irrelevant predictor.
3. **Identify** open backdoor paths and **apply** the backdoor criterion to find a sufficient adjustment set.
4. **Explain** why adjusting for a collider or a mediator biases an effect estimate, with a clinical example.
5. **Distinguish** the total effect from the direct effect and **choose** the estimand that matches the decision.
6. **Recognize** M-bias and selection bias as special cases of collider stratification.
7. **State** the identification assumptions (consistency, positivity, exchangeability, no measurement error) in plain language.
8. **Report** a DAG and its implied adjustment set alongside the estimate, with the assumptions that cannot be tested.

Outcome 3 maps to Exercise 3 and the worked example. Outcome 4 maps to Exercise 4 and self-check item 2. Outcome 7 maps to Exercise 6.

## 3. The core mental model

A DAG has nodes (variables) and directed edges (assumed direct effects). The graph is acyclic: no variable is its own ancestor. Two nodes are connected by a path if you can travel between them along edges, ignoring direction. A path is open or blocked depending on the types of nodes along it.

```text
  CHAIN (mediator)          FORK (confounder)         COLLIDER
  A --> M --> Y             A <-- C --> Y             A --> H <-- Y

  A affects Y through M.    C affects both A and Y.   H is caused by A and Y.
  Adjusting for M blocks    Adjusting for C blocks    Adjusting for H opens
  part of the effect.       the backdoor path.        a spurious path.
```

### 3.1 The three building blocks

| Structure | Pattern | Consequence of adjustment | Clinical example |
|---|---|---|---|
| Chain | A → M → Y | Blocks the indirect effect | Program → adherence → readmission |
| Fork | A ← C → Y | Blocks confounding | Prior MI → statin and → stroke |
| Collider | A → H ← Y | Opens a spurious path | Treatment → hospitalization ← outcome |
| Descendant | A → M → Y, and D is caused by M | Adjusting for D partially blocks the path | A lab value caused by adherence |

The first two are familiar. The third is the one that surprises analysts, because the collider often looks like an important confounder: it is correlated with the outcome, it differs by treatment, and it improves model fit. Those are exactly the properties of a variable that should not be adjusted for.

### 3.2 Paths, backdoors, and the backdoor criterion

A **backdoor path** is a path from treatment to outcome that starts with an arrow into the treatment. It represents a non-causal route by which the treatment and outcome are associated. The direct causal path starts with an arrow out of the treatment.

The **backdoor criterion** for a set Z relative to treatment A and outcome Y:

1. No node in Z is a descendant of A.
2. Z blocks every backdoor path between A and Y.

If Z satisfies the criterion, the effect of A on Y is identified by adjusting for Z, under the assumptions of consistency, positivity, and no measurement error. The adjustment formula is

```text
P(Y | do(A = a)) = sum over z of P(Y | A = a, Z = z) P(Z = z)
```

This is standardization: compute the outcome distribution within each stratum of Z, then average over the distribution of Z in the target population.

### 3.3 d-separation in one paragraph

A path is blocked if it contains a non-collider that is in Z, or a collider that is not in Z and has no descendant in Z. A path is open otherwise. Two sets of variables are d-separated by Z if every path between them is blocked. The backdoor criterion is a special case: Z blocks every backdoor path.

The rule for colliders is stated carefully because it is counterintuitive. A collider is open when you condition on it, and closed when you do not. Conditioning on a descendant of a collider also opens it, which is why adjusting for a variable caused by a collider can be as harmful as adjusting for the collider itself.

### 3.4 Total, direct, and indirect effects

The total effect of A on Y is the effect through all paths, including the path through M. The direct effect is the effect through paths that do not pass through M. The indirect effect is the difference, under assumptions that are stronger than those required for the total effect.

The choice is an estimand choice, not a statistical one:

- A payer deciding whether to cover a program wants the total effect, because the program's effect through adherence is part of what the payer buys.
- A scientist studying the mechanism wants the direct effect, and must justify the additional assumptions.
- A clinician deciding whether to monitor adherence wants to know whether adherence mediates the effect, which is a different question again.

The default for a decision about whether to adopt a strategy is the total effect. Adjusting for a mediator answers the direct-effect question and should be labelled as such.

### 3.5 The four identification assumptions

Every causal estimate from observational data relies on four assumptions, whether or not they are stated:

1. **Consistency:** the outcome observed under the treatment received equals the potential outcome under that treatment, as defined. Multiple versions of the treatment violate consistency.
2. **Positivity:** every covariate pattern in the target population has a positive probability of each treatment. A subgroup that receives only one treatment cannot be compared.
3. **Exchangeability:** the treated and untreated groups would have had the same outcome distribution if they had received the same treatment, conditional on the adjustment set. This is the no-unmeasured-confounder assumption.
4. **No measurement error:** the treatment, outcome, and confounders are measured correctly, or the error is known and corrected.

The DAG addresses exchangeability through the backdoor criterion. It does not address consistency, positivity, or measurement error. Those must be checked separately, and the checks belong in the analysis plan.

### 3.6 What the DAG cannot do

- It cannot tell you whether an edge exists. That is subject-matter knowledge.
- It cannot tell you whether a variable is measured well.
- It cannot handle time-varying treatments without extension (dynamic graphs or g-methods).
- It cannot resolve the direction of an edge between two variables measured at the same time.
- It cannot prove that all confounders are measured. The backdoor criterion is conditional on the graph being correct.

The DAG's value is that it makes the untestable assumptions explicit enough to be criticized.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Directed acyclic graph | A graph with directed edges and no cycles | Treated as a discovered truth | Graph |
| Node | A variable in the graph | Omitted because it is unmeasured | Variable |
| Edge | An assumed direct causal effect | Drawn from correlation | Arrow |
| Path | A sequence of edges between two nodes, ignoring direction | Confused with a direct effect | Sequence |
| Directed path | A path that follows the arrows | Called a "causal path" when it may include a collider | Sequence |
| Backdoor path | A path from treatment to outcome that starts with an arrow into treatment | Missed when it passes through an unmeasured variable | Sequence |
| Chain | A → M → Y | Called confounding | Structure |
| Fork | A ← C → Y | Called a mediator | Structure |
| Collider | A → H ← Y | Adjusted as a confounder | Structure |
| Descendant | A node reachable from another by following arrows | Confused with a child | Graph relation |
| Ancestor | A node that reaches another by following arrows | Confused with a parent | Graph relation |
| d-separation | A graphical criterion for conditional independence | Used as a test of causality | Criterion |
| Backdoor criterion | A sufficient condition for identifying a total effect by adjustment | Applied to a set that includes descendants of treatment | Criterion |
| Adjustment set | A set of variables used to block backdoor paths | Chosen by stepwise regression | Set |
| Minimal sufficient adjustment set | A sufficient set with no redundant variables | Assumed unique | Set |
| Confounder | A common cause of treatment and outcome | Any variable associated with the outcome | Variable |
| Mediator | A variable on the causal path | Adjusted to "be safe" | Variable |
| Collider | A common effect of two variables | Adjusted because it predicts the outcome | Variable |
| Proxy confounder | A measured variable that stands in for an unmeasured confounder | Treated as if it removes all confounding | Variable |
| Unmeasured confounder | A common cause that is not in the data | Assumed absent | Variable |
| M-bias | Bias from conditioning on a collider that is a common effect of two causes of treatment and outcome | Ignored in adjustment decisions | Bias structure |
| Selection bias | Bias from conditioning on a variable affected by treatment or outcome | Called confounding | Bias structure |
| Overadjustment | Adjustment for a mediator or a collider | Defended as "adjusting for everything" | Bias |
| Total effect | The effect through all causal paths | Confused with the direct effect | Effect |
| Direct effect | The effect not through a specified mediator | Reported without the mediation assumptions | Effect |
| Indirect effect | The effect through a mediator | Estimated without the needed assumptions | Effect |
| Consistency | The observed outcome under treatment equals the potential outcome under that treatment definition | Ignored when the treatment has multiple versions | Assumption |
| Positivity | Every covariate pattern has positive probability of each treatment | Checked only by the range of a propensity score | Assumption |
| Exchangeability | The groups would have the same outcome under the same treatment, given the adjustment set | Equated with "no confounding" without stating the set | Assumption |
| Positivity violation | A subgroup with no treated or no control patients | Trimmed away without reporting | Data feature |
| Instrumental variable | A variable that affects treatment but not the outcome except through treatment | Used without defending the exclusion restriction | Variable |
| Front-door criterion | Identification through a mediator when confounding is unmeasured | Applied without checking the mediator is unconfounded | Criterion |
| Time-varying confounding | A confounder whose value changes and is affected by past treatment | Handled by a baseline adjustment | Bias structure |
| Single-world intervention graph | A graph that represents the intervention and the observed data together | Confused with a DAG | Graph |

Two conventions in this topic:

- A DAG is reported with a caption that states the time zero and the direction of every edge. An arrow without a temporal interpretation is a guess.
- An adjustment set is reported with the backdoor paths it blocks, not only with the variable names.

> **Note:** "Adjust for everything you have" is not a causal strategy. It is a bet that no measured variable is a collider or a mediator. That bet is wrong often enough that the DAG should be drawn first.

## 5. The mechanics, step by step

The procedure moves from a decision to a graph, from the graph to an adjustment set, and from the adjustment set to an estimate. Steps 1 to 4 define the question and the variables. Steps 5 to 8 derive the adjustment set. Steps 9 to 13 estimate and test. Step 14 reports.

### Step 1. State the causal question and the estimand

**Input:** The decision, the intervention, the outcome, and the population.

**Output:** A one-sentence causal question plus the five estimand attributes.

**Why it exists:** The graph is built for a specific treatment and outcome. A graph for "statin and stroke" is not the same as a graph for "statin and stroke among patients with prior MI." The estimand fixes the population and the contrast, and the graph must match.

| Attribute | Example |
|---|---|
| Treatment condition | Initiation of a high-intensity statin within 30 days of discharge |
| Population | Adults discharged after myocardial infarction, 2018 to 2024 |
| Endpoint | Ischemic stroke within 3 years |
| Intercurrent-event strategy | Treatment policy for discontinuation; death as a competing event |
| Summary | Risk difference at 3 years, and hazard ratio |

### Step 2. Fix the temporal order and time zero

**Input:** The data dictionary and the clinical timeline.

**Output:** A list of variables ordered by the time they are measured relative to time zero.

**Why it exists:** A DAG is acyclic because causes precede effects. Two variables measured at the same time cannot be oriented from the data, and a variable measured after the outcome cannot be a confounder. The temporal order is the first constraint on the graph.

| Variable | Measured before time zero? | Role candidate |
|---|---|---|
| Age, sex | Yes | Confounder or precision variable |
| Prior MI | Yes | Confounder |
| Baseline LDL | Yes | Confounder or mediator of prior treatment |
| Statin initiation | At time zero | Treatment |
| Adherence | After time zero | Mediator |
| Hospitalization | After time zero | Collider or mediator |
| Stroke | After time zero | Outcome |

**Edge case:** A variable measured after time zero can still be a confounder if it is a proxy for a pre-treatment cause and is not affected by treatment. The graph must represent the proxy relationship explicitly, or the variable should be replaced by its pre-treatment measurement.

### Step 3. List the variables and assign provisional roles

**Input:** The temporal order, clinical knowledge, and the data dictionary.

**Output:** A node list with a provisional role for each variable.

**Why it exists:** The role determines whether the variable enters the adjustment set. Assigning roles before drawing edges prevents the graph from being reverse-engineered to justify a preferred adjustment set.

| Role | Definition | Action |
|---|---|---|
| Confounder | Common cause of treatment and outcome | Adjust, if measured and not a descendant of treatment |
| Mediator | Caused by treatment and causes the outcome | Do not adjust for the total effect |
| Collider | Caused by treatment and by the outcome, or by their causes | Do not adjust |
| Proxy | Correlated with an unmeasured confounder | Adjust with caution; residual confounding remains |
| Precision variable | Predicts the outcome but is unrelated to treatment | Optional; improves precision |
| Instrument | Affects treatment but not the outcome except through treatment | Do not adjust in a regression; use an instrument |
| Irrelevant | Unrelated to treatment and outcome | Exclude |

### Step 4. Draw the DAG

**Input:** The node list and the assumed direct effects.

**Output:** A graph with nodes, directed edges, and explicit unmeasured nodes.

**Why it exists:** The graph is the assumption ledger. It shows which variables are assumed to affect which others, and it makes the unmeasured variables visible rather than silently absent.

Rules for drawing:

- Draw an edge only when there is a plausible direct causal mechanism.
- Include unmeasured confounders as nodes, marked "U."
- Put time zero on the graph and order the nodes left to right.
- Do not draw an edge from a variable to its own descendant.
- Keep the graph as simple as the question allows; every extra edge is an extra assumption.
- Have a clinician or domain expert review the graph before the analysis.

**Edge case:** Two analysts can disagree about an edge. The disagreement is useful. Record both graphs and run the analysis under each; if the adjustment sets differ, the estimate is sensitive to the disputed assumption, and that is a finding.

### Step 5. Enumerate the paths between treatment and outcome

**Input:** The DAG.

**Output:** A list of all paths, each classified as causal, backdoor, or collider-containing.

**Why it exists:** The backdoor criterion is applied path by path. A path that is missed is a confounder that is missed.

| Path type | Pattern | Treatment |
|---|---|---|
| Causal | Starts with an arrow out of treatment | Leave open |
| Backdoor | Starts with an arrow into treatment | Block with adjustment |
| Collider-containing | Contains a collider | Leave closed; do not adjust for the collider |
| Mediated | Passes through a mediator | Leave open for the total effect |

### Step 6. Find a sufficient adjustment set

**Input:** The list of backdoor paths and the candidate variables.

**Output:** One or more sets that block every backdoor path and contain no descendant of treatment.

**Why it exists:** The backdoor criterion is a sufficient condition. Finding a set that satisfies it is the identification step.

Algorithm for a small graph:

1. List every backdoor path.
2. For each path, identify a non-collider on the path that can be adjusted.
3. Collect a set that hits every path.
4. Remove redundant variables to obtain a minimal set.
5. Check that no variable in the set is a descendant of treatment.

**Decision table for a candidate adjustment set:**

| Candidate set | Blocks all backdoor paths? | Contains a descendant of treatment? | Verdict |
|---|---|---|---|
| Empty set | No, if any backdoor path exists | No | Not sufficient |
| Confounders only | Yes | No | Sufficient |
| Confounders plus a mediator | Yes, but blocks part of the effect | Yes | Answers the direct-effect question, not the total effect |
| Confounders plus a collider | No, opens a new path | Usually no | Biased |
| All measured variables | Depends on the graph | Often yes | Not a criterion |

### Step 7. Check for colliders and mediators in the proposed set

**Input:** The candidate set.

**Output:** A revised set with every collider and mediator removed, or an explicit statement that a direct effect is the target.

**Why it exists:** The most common DAG error in applied work is adjusting for a variable that is caused by treatment. The variable is often a clinical event that looks like a confounder, such as hospitalization, discontinuation, or a post-treatment lab value.

| Variable | If caused by treatment | If caused by outcome | Action |
|---|---|---|---|
| Hospitalization | Yes | Yes | Collider; do not adjust |
| Adherence | Yes | No | Mediator; do not adjust for the total effect |
| Discontinuation | Yes | Possibly | Intercurrent event; handle through the estimand |
| Post-treatment biomarker | Yes | No | Mediator or surrogate; do not adjust |
| Baseline severity | No | No | Confounder; adjust |

### Step 8. Handle unmeasured confounding

**Input:** The DAG with unmeasured nodes.

**Output:** A strategy: proxy adjustment, negative control, instrumental variable, or a sensitivity analysis.

**Why it exists:** The backdoor criterion is conditional on the graph. If an unmeasured confounder exists, no measured adjustment set identifies the effect, and the analysis must say so.

| Strategy | What it requires | What it provides | Limitation |
|---|---|---|---|
| Proxy adjustment | A measured variable correlated with the unmeasured confounder | Partial confounding control | Residual confounding remains |
| Negative control outcome | An outcome that should not be affected by treatment | A test for residual bias | Does not correct the estimate |
| Negative control exposure | An exposure that should not affect the outcome | A test for bias | Same |
| Instrumental variable | An instrument satisfying exclusion and monotonicity | Identification under strong assumptions | Local effect only |
| E-value | The observed effect measure | The strength of confounding needed to explain it away | No correction |
| Quantitative bias analysis | Assumed confounder prevalence and associations | A corrected estimate with an interval | Assumption-dependent |

### Step 9. Choose the estimator that matches the estimand

**Input:** The adjustment set, the outcome type, and the target population.

**Output:** A named estimator and its target.

**Why it exists:** The estimator must target the estimand; a regression coefficient and a standardized effect are different quantities.

| Estimator | Target | Requires |
|---|---|---|
| Regression adjustment | Conditional effect | Correct outcome model |
| Standardization (g-computation) | Marginal effect in a target population | Correct outcome model; covariate distribution |
| Inverse probability weighting | Marginal effect | Correct treatment model; positivity |
| Matching | Effect in the matched population | Overlap; a matching specification |
| Doubly robust | Marginal effect | One of the two models correct |
| Instrumental variables | Local average treatment effect | Exclusion restriction, monotonicity |

The estimator does not fix a bad adjustment set. A doubly robust estimator with a collider in the set is biased even if both models are correct, because the collider is in the design.

### Step 10. Check positivity and consistency

**Input:** The data and the treatment definition.

**Output:** A positivity assessment and a consistency statement.

**Why it exists:** The backdoor criterion identifies the effect only if every covariate stratum has both treatment groups and the treatment is defined unambiguously.

**Positivity checks:**

- Cross-tabulate treatment by each categorical confounder; flag empty cells.
- Estimate a propensity score and inspect the overlap; flag values near 0 or 1.
- Report the proportion of patients outside the common support.
- Do not trim silently; report the population after trimming.

**Consistency checks:**

- Is the treatment a single, well-defined intervention, or does it include multiple doses and formulations?
- Does the outcome definition match the one in the estimand?
- Are there versions of the treatment that could produce different outcomes?

A useful consistency test is to ask whether the treatment definition would mean the same thing at two different sites. If one site starts patients on a lower dose because of formulary rules, the pooled estimate is a mixture of two interventions. The fix is to define the treatment narrowly, to stratify by site, or to report the site-specific estimates. Pooling without checking is a consistency violation that no amount of adjustment repairs.

### Step 11. Estimate and check balance

**Input:** The estimator and the data.

**Output:** An estimate, an interval, and a balance table.

**Why it exists:** Balance is a diagnostic for the adjustment, not a proof of identification. A weighted or matched sample that is balanced on measured confounders may still be unbalanced on the unmeasured ones.

- Report standardized mean differences before and after adjustment for every confounder.
- Report the effective sample size after weighting.
- Report the overlap and any trimming.
- Compare the adjusted estimate with the crude estimate and explain the difference through the confounding structure.

### Step 12. Test the graph's implications

**Input:** The DAG and the data.

**Output:** A list of testable conditional independencies and their results.

**Why it exists:** A DAG implies that certain variables are independent conditional on others. If the data contradict those implications, the graph is wrong, and the adjustment set may be wrong with it.

Examples:

- If the graph says C is the only confounder, then A and Y should be independent given C and the other parents. A strong residual association suggests an unmeasured common cause or a missing edge.
- A negative control outcome should show no effect. An effect suggests residual confounding or a design flaw.
- A negative control exposure should show no effect for the same reason.

**Edge case:** A failed implication does not tell you which edge is wrong. It tells you the graph is incomplete. The response is to revise the graph and re-derive the adjustment set, not to add variables until the test passes.

### Step 13. Run a sensitivity analysis for unmeasured confounding

**Input:** The estimate and the graph's unmeasured nodes.

**Output:** An E-value, a bias analysis, or a tipping point.

**Why it exists:** The no-unmeasured-confounder assumption cannot be tested. The sensitivity analysis quantifies how strong an unmeasured confounder would have to be to change the conclusion.

The E-value for a risk ratio is

```text
E = RR + sqrt(RR × (RR - 1))     for RR above 1
E = 1/RR + sqrt((1/RR) × (1/RR - 1))  for RR below 1
```

For an odds ratio with a rare outcome, the risk ratio can be approximated, but the approximation must be stated. The E-value is a summary, not a substitute for a bias analysis with a specific confounder in mind.

The E-value should be reported for the estimate and for the confidence limit closest to the null. The second value is usually smaller and is the one that matters for a decision: it describes how much confounding would be needed to make the result compatible with no effect. Reporting only the point-estimate E-value overstates robustness.

A sensitivity analysis that assumes a specific confounder is more informative than an E-value, because it uses the confounder's actual prevalence and associations. The E-value is the fallback when no specific confounder can be measured.

Neither method corrects the estimate. Both describe how fragile it is.

### Step 14. Report the graph, the set, the estimate, and the assumptions

**Input:** The DAG, the adjustment set, the estimate, and the diagnostics.

**Output:** A results section that a skeptical reviewer can follow.

**Why it exists:** The graph, the set, and the estimate form an argument; omitting one breaks the chain.

The reporting checklist:

1. The causal question and the estimand.
2. The DAG, with time zero and unmeasured nodes marked.
3. The backdoor paths and the adjustment set, with the reason for each excluded variable.
4. The estimator and its target population.
5. The positivity assessment and any trimming.
6. The balance table.
7. The estimate and interval, on the decision scale.
8. The testable implications and their results.
9. The sensitivity analysis and the E-value.
10. The assumptions that cannot be tested.

A minimal sufficient set is preferred to a larger one because every extra variable adds a modelling assumption and a chance of introducing a collider or a mediator. The minimal set is not unique: a graph can have several minimal sufficient sets, and the choice among them may be driven by measurement quality. When two sets are both sufficient, prefer the one with better-measured variables and fewer missing values, and report both estimates if they disagree.

The graph, the set, and the estimate form a chain. A reviewer who disputes the graph can re-derive the set and see whether the estimate changes. That is the form of transparency the DAG is meant to provide.

### Edge cases that change the adjustment set

| Edge case | What happens | Detection | Remedy |
|---|---|---|---|
| M-bias | Adjusting for a collider that is a common effect of two causes of treatment and outcome | The variable is not a confounder but is correlated with both | Do not adjust; draw the graph |
| Selection on a collider | Restricting the cohort to hospitalized patients induces association | The cohort is defined by a post-treatment event | Redesign the cohort; sensitivity analysis |
| Time-varying confounding | A confounder affected by past treatment | Treatment and confounder evolve together | g-methods; marginal structural models |
| Mediation | The adjustment set includes a mediator | The variable is on the causal path | Target the total effect; exclude it |
| Unmeasured confounder | No measured set satisfies the backdoor criterion | The graph has a U node on a backdoor path | Proxy, instrument, or sensitivity analysis |
| Treatment misclassification | The treatment definition has multiple versions | Consistency is violated | Refine the definition; sensitivity analysis |
| Positivity violation | A stratum has no treated or no control patients | Empty cells; propensity scores near 0 or 1 | Restrict the population; report the change |
| Dynamic treatment | The treatment strategy depends on time-varying covariates | The strategy is a rule, not a fixed dose | g-methods; target trial emulation |

### Time-varying confounding, briefly

A confounder that is affected by past treatment cannot be handled by baseline adjustment. In a graph with treatment A0 at time 0, confounder L1 at time 1, and treatment A1 at time 1, adjusting for L1 blocks part of the effect of A0 and opens a path from A1. The g-formula, marginal structural models with inverse probability weighting, and g-estimation are the standard tools. The key point for the DAG is that the graph must include the time ordering, and a single baseline covariate cannot represent a variable that changes.

Target trial emulation (Topic 07) is the design-level response: specify the treatment strategies, align time zero, and use the g-methods to estimate the effect of the strategies rather than the effect of a baseline exposure.

### Branch points at a glance

| If the graph shows | Then the adjustment set is | Because |
|---|---|---|
| A fork C → A, C → Y | {C} or a set that blocks the fork | C is a confounder |
| A chain A → M → Y | No adjustment for the total effect | M is a mediator |
| A collider A → H ← Y | Do not include H | Adjustment opens a path |
| A collider plus a descendant D of H | Do not include D | Descendants of colliders also open the path |
| An unmeasured U on a backdoor path | No measured set is sufficient | Report the limitation; sensitivity analysis |
| A proxy P of U | Include P, with residual confounding acknowledged | Partial control only |
| A variable measured after treatment | Exclude unless it is a pre-treatment proxy | It may be a mediator or collider |
| A selection variable S affected by A or Y | Redesign the cohort or model the selection | Conditioning on S induces bias |
| Time-varying treatment and confounders | g-methods; not baseline regression | The confounder is affected by past treatment |

The table does not replace the graph. It is a checklist for the most common structures, and it is useful when reviewing someone else's adjustment set.

## 6. Worked example

### 6.1 The question and the estimand

A payer is deciding whether to include a high-intensity statin protocol in a post-myocardial-infarction (MI) care bundle. The comparison is initiation within 30 days of discharge versus no initiation. The outcome is ischemic stroke within 3 years. The population is adults discharged alive after an MI.

The estimand:

| Attribute | Specification |
|---|---|
| Treatment condition | High-intensity statin initiated within 30 days of discharge vs no initiation in that window |
| Population | Adults discharged alive after MI, 2018 to 2024 |
| Endpoint | Ischemic stroke within 3 years |
| Intercurrent-event strategy | Treatment policy for discontinuation; death before stroke counts as a competing event |
| Summary | Risk difference at 3 years, standardized to the discharged population; hazard ratio secondary |

### 6.2 The DAG

Nodes: A = statin initiation, Y = stroke, C = prior MI, H = hospitalization during follow-up, U = unmeasured frailty.

Edges:

- C → A: prior MI increases the probability of statin initiation.
- C → Y: prior MI increases stroke risk.
- A → Y: the effect of interest.
- A → H: statin initiation affects hospitalization.
- Y → H: stroke affects hospitalization.

```text
  C (prior MI) -----> A (statin) -----> Y (stroke)
       |                                  ^
       |                                  |
       +----------------------------------+
       |
       +----> H (hospitalization) <-------+
                    ^
                    |
              A ----+  and  Y ----+
```

The paths between A and Y:

| Path | Type | Status without adjustment | Status with C adjusted |
|---|---|---|---|
| A → Y | Causal | Open | Open |
| A ← C → Y | Backdoor | Open | Blocked |
| A → H ← Y | Collider | Blocked | Blocked |

The sufficient adjustment set is {C}. Adjusting for H would open the collider path A → H ← Y and create bias. The empty set leaves the backdoor path open.

### 6.3 The data

| Stratum | Statin: events / n | Risk | No statin: events / n | Risk | Risk difference |
|---|---|---:|---|---:|---:|
| Prior MI | 80 / 400 | 20.00% | 60 / 200 | 30.00% | -10.00 pp |
| No prior MI | 30 / 600 | 5.00% | 150 / 1800 | 8.33% | -3.33 pp |
| Total | 110 / 1000 | 11.00% | 210 / 2000 | 10.50% | +0.50 pp |

The total row is the crude comparison. The stratum-specific rows are the comparisons within levels of prior MI.

### 6.4 The crude analysis

```text
Risk, statin = 110 / 1000 = 0.1100 = 11.00%
Risk, no statin = 210 / 2000 = 0.1050 = 10.50%
Risk difference = 0.1100 - 0.1050 = +0.0050 = +0.50 percentage points
Odds ratio = (110/890) / (210/1790) = 0.123596 / 0.117318 = 1.0535
SE(log OR) = sqrt(1/110 + 1/890 + 1/210 + 1/1790) = 0.12464
95% CI = exp(0.05213 +/- 1.96 × 0.12464) = (0.827, 1.342)
```

The crude analysis suggests a small increase in stroke risk, with an interval that includes no association. An analyst who stopped here would report "no clear evidence of benefit" or "possible harm."

### 6.5 Standardization

Standardize to the distribution of prior MI in the combined population:

```text
P(C = prior MI) = 600 / 3000 = 0.2000
P(C = no prior MI) = 2400 / 3000 = 0.8000
```

Standardized risk under statin:

```text
0.2000 × 0.2000 + 0.8000 × 0.0500
= 0.0400 + 0.0400 = 0.0800 = 8.00%
```

Standardized risk under no statin:

```text
0.2000 × 0.3000 + 0.8000 × 0.083333
= 0.0600 + 0.066667 = 0.126667 = 12.67%
```

Standardized risk difference:

```text
RD = 0.0800 - 0.126667 = -0.046667 = -4.67 percentage points
RR = 0.0800 / 0.126667 = 0.6316
```

Variance of the standardized RD, treating the weights as fixed:

```text
Var = 0.04 × [0.20×0.80/400 + 0.30×0.70/200]
    + 0.64 × [0.05×0.95/600 + 0.08333×0.91667/1800]
    = 0.04 × [0.000400 + 0.001050] + 0.64 × [0.0000792 + 0.0000424]
    = 0.04 × 0.001450 + 0.64 × 0.0001216
    = 0.0000580 + 0.0000778 = 0.0001358
SE = sqrt(0.0001358) = 0.01166
95% CI = -0.046667 +/- 1.96 × 0.01166
       = (-0.06952, -0.02382) = (-6.95, -2.38) percentage points
```

The Mantel-Haenszel odds ratio, as a consistency check, is

```text
MH OR = [80×140/600 + 30×1650/1800] / [320×60/600 + 570×150/1800]
      = [18.667 + 27.500] / [32.000 + 47.500]
      = 46.167 / 79.500 = 0.5807
```

The standardized RD (-4.67 pp), the standardized RR (0.63), and the MH OR (0.58) all point to benefit. The crude analysis pointed to harm. The sign reversal is confounding by indication.

### 6.6 Diagnostic checks

**Check 1. Counts reconcile.** 80 + 30 = 110 statin events; 60 + 150 = 210 no-statin events; 400 + 600 = 1,000 statin patients; 200 + 1,800 = 2,000 no-statin patients; 3,000 total. The crude risk of 11.00% lies between the stratum-specific statin risks of 20.00% and 5.00%, as it must.

**Check 2. Stratum-specific effects agree in direction.** The risk difference is -10.00 pp in the prior-MI stratum and -3.33 pp in the no-prior-MI stratum. Both are negative. The crude estimate is positive because the statin group is enriched with prior-MI patients, who have a much higher stroke risk. This is confounding, not effect modification. If the stratum-specific effects had opposite signs, the interpretation would be effect modification, and a single standardized number would hide it.

**Check 3. Standardization and Mantel-Haenszel agree.** The standardized RD is -4.67 pp and the MH OR is 0.58. Both are below the null. Two methods with different weighting schemes agreeing in direction is a check on the arithmetic and on the homogeneity of the stratum-specific effects.

**Check 4. The crude interval and the adjusted interval tell different stories.** The crude OR interval (0.827, 1.342) includes 1. The adjusted RD interval (-6.95, -2.38) excludes 0. The difference is not a change in precision; it is a change in the quantity being estimated.

**Check 5. Positivity holds.** All four cells have both events and non-events, and both strata have statin and no-statin patients. There is no stratum with a structural zero. If the no-statin group had zero patients in the prior-MI stratum, the standardized risk in that stratum would not be estimable, and the analysis would have to restrict the population or change the estimand.

**Check 6. Weights sum to one.** 0.20 + 0.80 = 1.00. A standardization that does not sum to one indicates a coding error or a missing stratum.

**Check 7. The E-value is moderate.** For RR = 0.6316:

```text
E = 1/0.6316 + sqrt((1/0.6316) × (1/0.6316 - 1))
  = 1.5833 + sqrt(1.5833 × 0.5833)
  = 1.5833 + 0.9606 = 2.5439
```

An unmeasured confounder would need a risk ratio of about 2.54 with both statin initiation and stroke, above and beyond prior MI, to explain away the association. That is not implausible in a frail older population, where frailty affects both prescribing and stroke risk. The result is suggestive, not definitive.

### 6.7 Vary one assumption: prior MI is unmeasured

The entire adjustment rests on measuring prior MI. Suppose the claims data do not contain a reliable prior-MI indicator, so C is unmeasured. The DAG then has a backdoor path A ← U → Y that no measured set can block, where U includes prior MI and frailty.

The only analysis available is the crude one:

```text
RD = +0.50 percentage points
OR = 1.0535 (95% CI 0.827 to 1.342)
```

The conclusion changes from a statistically significant 4.67-percentage-point benefit to a null-to-harmful association. The direction reverses, and the decision changes from "adopt" to "do not adopt without better data."

This is the material variation the exercise asks for. It is not a sensitivity analysis in the technical sense; it is a demonstration that the identification depends on a measured variable. A proper sensitivity analysis would use an E-value or a bias analysis with a plausible unmeasured confounder, which is what Check 7 does.

### 6.8 Vary the target population: the ATT

If the payer is deciding whether to expand the protocol among patients who would actually receive it, the relevant target may be the treated population, not the whole discharged population. Standardize to the prior-MI distribution among treated patients:

```text
P(C = prior MI | A = 1) = 400 / 1000 = 0.4000
P(C = no prior MI | A = 1) = 600 / 1000 = 0.6000

Risk under statin, treated distribution = 0.4000×0.2000 + 0.6000×0.0500 = 0.1100
Risk under no statin, treated distribution = 0.4000×0.3000 + 0.6000×0.083333 = 0.1700
RD = -0.0600 = -6.00 percentage points
```

The ATT is -6.00 pp, 29% larger in magnitude than the ATE of -4.67 pp, because the treated population has a higher prevalence of prior MI, where the absolute benefit is larger. The direction is the same; the magnitude and the budget impact are not.

### 6.9 What to report

Report the standardized risk difference at 3 years as the primary estimate: **-4.67 percentage points (95% CI -6.95 to -2.38)**, with the RR of 0.63 and the MH OR of 0.58 as supporting measures. Report the DAG and the adjustment set {prior MI}. Report the crude estimate and explain the reversal. Report the E-value of 2.54 and the ATT of -6.00 pp as sensitivity analyses.

Report the assumptions explicitly:

- No unmeasured confounding of the statin-stroke association given prior MI. This is the assumption most likely to fail, because frailty and socioeconomic status are not measured.
- Consistency: the treatment is a class of high-intensity statins, and different agents are assumed to have the same effect. This is a strong assumption and should be stated.
- Positivity: all strata have both treatment groups.
- Correct measurement of prior MI and stroke. A validation sample would be required to support this.

The honest conclusion: "Under the assumption that prior MI is the only confounder, initiating a high-intensity statin within 30 days of discharge was associated with a 4.67-percentage-point lower 3-year stroke risk. The crude association was in the opposite direction because prior MI is a strong confounder. An unmeasured confounder with a risk ratio of about 2.5 with both treatment and outcome could explain the result. The estimate is not sufficient on its own for a coverage decision; it is sufficient to justify a prospective evaluation."

That is the estimate to report because it is the one that matches the decision (a population-level coverage question), and because the DAG supports its identification. The crude estimate is reported alongside it, not instead of it.

## 7. Code, query, or calculation

The script has two parts. The first implements the backdoor criterion on a small DAG: it enumerates the paths between treatment and outcome and tests whether a candidate adjustment set blocks each path. The second reproduces the crude, standardized, and Mantel-Haenszel estimates from the worked example, and computes the E-value.

**Assumptions:**

- The DAG is correct as drawn.
- The standardization weights are the population distribution of prior MI.
- The stratum-specific risks are estimated without error beyond binomial sampling.
- The E-value uses the risk ratio scale and assumes no residual confounding within strata.

```python
import math

# --- 1. DAG: backdoor paths and adjustment sets ---
parents = {"A": ["C"], "Y": ["A", "C"], "H": ["A", "Y"], "C": []}

def descendants(node, parents):
    out, stack = set(), [node]
    while stack:
        cur = stack.pop()
        for child, ps in parents.items():
            if cur in ps and child not in out:
                out.add(child)
                stack.append(child)
    return out

def is_collider(prev, node, nxt, parents):
    return prev in parents[node] and nxt in parents[node]

def path_blocked(path, Z, parents):
    Zset = set(Z)
    for i in range(1, len(path) - 1):
        prev, node, nxt = path[i-1], path[i], path[i+1]
        if is_collider(prev, node, nxt, parents):
            if node not in Zset and not (descendants(node, parents) & Zset):
                return True
        else:
            if node in Zset:
                return True
    return False

def all_paths(start, end, parents, maxlen=6):
    out = []
    def dfs(node, path):
        if len(path) > maxlen:
            return
        if node == end:
            out.append(path[:])
            return
        for nxt in parents.get(node, []) + [c for c, ps in parents.items() if node in ps]:
            if nxt not in path:
                dfs(nxt, path + [nxt])
    dfs(start, [start])
    return out

paths = all_paths("A", "Y", parents)
print("paths between A and Y:")
for p in paths:
    print(" ", " - ".join(p))
for Z in ([], ["C"], ["C", "H"], ["H"]):
    status = [(p, path_blocked(p, Z, parents)) for p in paths if len(p) > 2]
    print(f"Z={Z}: " + "; ".join(f"{' - '.join(p)} blocked={b}" for p, b in status))

# --- 2. Confounding reversal in a 2x2x2 table ---
# strata: (events treated, n treated, events control, n control)
strata = {"prior MI": (80, 400, 60, 200), "no prior MI": (30, 600, 150, 1800)}

def risk(e, n):
    return e / n

A = sum(v[0] for v in strata.values()); N1 = sum(v[1] for v in strata.values())
C = sum(v[2] for v in strata.values()); N0 = sum(v[3] for v in strata.values())
crude_rd = risk(A, N1) - risk(C, N0)
crude_or = (A / (N1 - A)) / (C / (N0 - C))
print(f"crude: treated {risk(A,N1):.4f} control {risk(C,N0):.4f} RD={crude_rd:.4f} OR={crude_or:.4f}")

pC = 600 / 3000
std_t = pC * risk(80, 400) + (1 - pC) * risk(30, 600)
std_c = pC * risk(60, 200) + (1 - pC) * risk(150, 1800)
std_rd = std_t - std_c
var = (pC ** 2) * (risk(80,400)*(1-risk(80,400))/400 + risk(60,200)*(1-risk(60,200))/200) + \
      ((1-pC) ** 2) * (risk(30,600)*(1-risk(30,600))/600 + risk(150,1800)*(1-risk(150,1800))/1800)
se = math.sqrt(var)
print(f"standardized: treated {std_t:.4f} control {std_c:.4f} RD={std_rd:.4f} "
      f"95% CI=({std_rd-1.96*se:.4f}, {std_rd+1.96*se:.4f})")
print(f"standardized RR={std_t/std_c:.4f}")

# Mantel-Haenszel odds ratio as a consistency check
num = sum(a * d / (n1 + n0) for a, n1, c, n0 in strata.values() for d in [n0 - c])
den = sum((n1 - a) * c / (n1 + n0) for a, n1, c, n0 in strata.values())
print(f"Mantel-Haenszel OR={num/den:.4f}")

# E-value for the standardized RR (below 1)
rr = std_t / std_c
E = 1 / rr + math.sqrt((1 / rr) * (1 / rr - 1))
print(f"E-value for RR={rr:.4f}: {E:.4f}")
```

**Expected output:**

```text
paths between A and Y:
  A - C - Y
  A - Y
  A - H - Y
Z=[]: A - C - Y blocked=False; A - H - Y blocked=True
Z=['C']: A - C - Y blocked=True; A - H - Y blocked=True
Z=['C', 'H']: A - C - Y blocked=True; A - H - Y blocked=False
Z=['H']: A - C - Y blocked=False; A - H - Y blocked=False
crude: treated 0.1100 control 0.1050 RD=0.0050 OR=1.0535
standardized: treated 0.0800 control 0.1267 RD=-0.0467 95% CI=(-0.0695, -0.0238)
standardized RR=0.6316
Mantel-Haenszel OR=0.5810
E-value for RR=0.6316: 2.5444
```

**What the output demonstrates:**

- With no adjustment, the path A ← C → Y is open, and the collider path A → H ← Y is blocked. The empty set is not sufficient.
- With C adjusted, both paths are blocked. The backdoor criterion is satisfied.
- With C and H adjusted, the backdoor path is blocked but the collider path is opened. This is the formal version of "do not adjust for hospitalization." The estimate would be biased even though the model includes the "right" confounder.
- With only H adjusted, both paths are open. The result is the worst of the four options.
- The crude estimate is +0.50 percentage points, and the standardized estimate is -4.67 percentage points. The sign reversal is reproduced exactly.
- The MH OR of 0.581 agrees with the standardized RD in direction. The two methods weight the strata differently but tell the same story because the stratum-specific effects are homogeneous.
- The E-value of 2.54 is the strength of association an unmeasured confounder would need with both treatment and outcome. It is a sensitivity summary, not a correction.

> **Note:** The DAG path code is a teaching implementation. It handles small graphs with explicit parents and no latent variables beyond those drawn as nodes. Production work should use a validated library such as DAGitty, and the graph should be reviewed by a domain expert before the adjustment set is trusted.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| ICH E9 (1998) | Binding through regional adoption | Pre-specified analysis, analysis populations, subgroup and covariate handling | The use of a DAG |
| ICH E9(R1) (2019) | Binding through regional adoption | The estimand, intercurrent events, sensitivity analyses | The identification strategy |
| FDA RWE guidance (August 2023) | FDA guidance | Reliability and relevance of the data source, pre-specified protocols, transparency | The graphical method |
| Consolidated Standards of Reporting Trials (CONSORT) 2010 | Journal editorial policy | Reporting of the analysis and the adjustment | The DAG |
| Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) and REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) | Journal editorial policy | Reporting of observational design and confounding control | The adjustment set |
| DAGitty and the Textor et al. methods paper | Methodological reference, not binding | Algorithms for adjustment sets | The substantive graph |

### 8.2 What the standards require, and what they leave open

ICH E9(R1) requires the estimand to be defined before analysis. It does not require a causal graph, but a graph is the most direct way to show that the estimand matches the adjustment. A protocol that states "adjusted for age, sex, and comorbidity" without saying why those variables and not others has not documented its identification strategy.

The FDA RWE guidance requires the study design to be pre-specified and the data source to be reliable and relevant. A pre-specified DAG is evidence that the design was not chosen after seeing the associations. A graph drawn after the analysis is a description, not a pre-specification.

STROBE item 16a asks for the methods used to control for confounding. RECORD item 7.1 asks for the code lists for confounders. Neither asks for a graph, but a graph is a compact way to satisfy the spirit of both.

### 8.3 Why a graph is not a regulatory document

A DAG is a model of assumptions. Regulators evaluate the assumptions, not the drawing. A submission that includes a graph but does not discuss the unmeasured confounders, the positivity, and the measurement error has not addressed the review question. The graph is useful because it organizes that discussion, not because it is a required artifact.

### 8.4 Testable implications and negative controls

A DAG implies conditional independencies, and those can be tested. The tests are weak: failure to reject an independence is not proof that the graph is correct, and rejection does not identify which edge is wrong. They are still useful as a check on the most consequential assumptions.

Negative control outcomes and exposures are a related tool. A negative control outcome should not be affected by the treatment; an association suggests residual confounding or a design flaw. The choice of control must be justified by the same kind of causal reasoning as the main analysis, and the control should share the suspected confounders with the outcome.

### 8.5 A DAG reporting template

A graph in a protocol or paper is useful when it is accompanied by six items:

1. The time zero, marked on the graph or in the caption.
2. The treatment, outcome, and every measured variable, with unmeasured variables marked "U."
3. The direction and justification of each edge, in a table if the graph is busy.
4. The backdoor paths and the adjustment set, with the reason for each excluded variable.
5. The assumptions that cannot be tested, stated in plain language.
6. The sensitivity analyses that address them.

A graph without item 3 is a picture. A graph without item 4 is not an identification strategy. A graph without item 5 invites the reader to assume the assumptions are satisfied.

The template is not a regulatory requirement. It is a way to satisfy the requirement that the analysis be transparent about its assumptions, and it makes the disagreement between analysts visible before the estimate is computed.

### 8.6 Three questions a reviewer will ask

1. **Why these variables?** The answer is the backdoor paths, not the univariate associations.
2. **Why not those variables?** The answer is the collider or mediator role of each excluded variable.
3. **What would change the conclusion?** The answer is the E-value, the negative control, or the tipping point.

A protocol that answers all three has an identification strategy. A protocol that answers only the first has a variable list.

The questions are also a useful structure for an internal review. Ask them before the analysis is run, not after the estimate is presented, because the answers are design decisions and the estimate cannot validate them.

### 8.7 What the standards do not cover

- They do not specify how to draw the graph or which variables to include.
- They do not require the graph to be pre-specified, although pre-specification is the point.
- They do not validate the absence of unmeasured confounding.
- They do not resolve disagreements between analysts who draw different graphs.

## 9. Industry practice and case evidence

### 9.1 Documented case: hormone replacement therapy

Observational studies suggested that hormone replacement therapy reduced coronary heart disease in postmenopausal women. The Women's Health Initiative randomized trial, reported by Manson and colleagues in the New England Journal of Medicine in 2003, found an increased risk of coronary heart disease in the estrogen-plus-progestin arm.

The DAG interpretation is that the observational comparison had a backdoor path through socioeconomic status and health behaviors: women who chose therapy were healthier and had better access to care. The adjustment sets available in the observational data did not block that path, because the relevant variables were measured coarsely or not at all. The randomized trial blocked it by design.

**Inference:** the trial does not prove that every observational estimate of HRT was wrong. It shows that a backdoor path through unmeasured health behavior was sufficient to reverse the association, and that the observational adjustment sets did not close it.

### 9.2 Documented case: beta-blockers in heart failure

Early observational studies suggested that beta-blockers might be harmful in heart failure. Randomized trials demonstrated a mortality benefit in heart failure with reduced ejection fraction. The observational signal was confounded by indication: sicker patients were more likely to receive beta-blockers, and sickness also predicted mortality.

A DAG for the observational study has prior severity as a common cause of treatment and outcome. If severity is measured well, adjustment can recover the effect. If it is measured by a proxy such as ejection fraction, residual confounding remains. The trial removes the problem by randomizing within severity strata.

### 9.3 Documented case: sepsis bundles

Observational studies of sepsis bundles are vulnerable to confounding by indication and to temporal trends. A hospital that adopts a bundle may also be improving in other ways at the same time. The DAG includes calendar time as a common cause of bundle adoption and mortality, and a before-after analysis that ignores it has an open backdoor path.

The design response is an interrupted time series with a comparison hospital, or a difference-in-differences design. Both are attempts to block the calendar-time path, not to adjust for it in a regression.

### 9.4 Documented case: algorithmic bias in care management

A widely used algorithm predicted health-care costs as a proxy for need. Because spending was lower for Black patients at the same level of need, the algorithm systematically under-referred Black patients to care management. The label was a proxy for the construct the program intended to measure.

The causal graph has race as a cause of cost through structural inequity and a cause of need through other pathways. The algorithm conditioned on cost, which is a descendant of race and a proxy for need. The result was a biased allocation. The lesson is that the choice of outcome and proxy is a causal decision, and a DAG makes the proxy relationship visible.

### 9.5 Documented practice: target trial emulation

The RCT-DUPLICATE initiative emulated 32 trials using claims data with pre-specified protocols, propensity score matching, and Cox models. The reported correlation of 0.82 between emulated and randomized estimates, rising to 0.93 in the close-emulation subgroup, is evidence that a carefully designed observational analysis can approximate a trial. It is not evidence that an arbitrary DAG and an arbitrary adjustment set will do so.

The design elements that carried the result were the eligibility criteria, the time zero, the treatment strategies, and the outcome definitions, not the choice of regression software.

### 9.6 Documented practice: DAGs in protocol review

Several research groups and regulators use DAGs in protocol review to make confounding control explicit. The practice is not universal, and there is no single template. The common pattern is a graph in the protocol, a list of the adjustment sets it implies, and a statement of the unmeasured variables.

**Inference:** the frequency of DAG use in industry protocols is not documented in a public census. The methodological literature recommends it, and the guidance documents require the assumptions it organizes, but the two are not the same as a mandate.

### 9.7 What the case evidence does and does not establish

| Claim | Status | Basis |
|---|---|---|
| A backdoor path through unmeasured health behavior can reverse an observational association | Documented | The HRT trial and the observational literature |
| Confounding by indication can make an effective treatment look harmful | Documented | Beta-blocker trials versus observational studies |
| A before-after comparison can be confounded by calendar time | Documented | Sepsis bundle implementation studies |
| A proxy label can encode structural inequity | Documented | The care-management algorithm analysis |
| A carefully emulated trial can approximate a randomized result on average | Documented with variation | RCT-DUPLICATE correlation and subgroup results |
| Any DAG plus any adjustment set identifies the effect | Not supported | The graph is an assumption; unmeasured paths and positivity remain |

The first five rows are findings from published studies. The last row is the inference that the evidence does not support, and it is the one most often assumed in practice.

A useful discipline is to write the table before writing the results. If the analysis cannot be placed in one of the documented rows, it is probably a new claim that needs its own evidence, not a variant of an established one.

### 9.8 The cost of skipping the graph

A missing graph rarely produces an obviously wrong number. It produces a number that survives internal review and fails external review, because the adjustment set cannot be defended. The costs are:

- A program evaluation that is rejected by a payer because the confounding control cannot be explained.
- A manuscript returned by a journal because the adjustment set is not justified.
- A replicated analysis that cannot reproduce the published estimate because the original model adjusted for a post-treatment variable.
- A decision that is reversed when a stronger design becomes available.

**Inference:** the frequency of these outcomes is not captured in a single statistic. The existence of STROBE, RECORD, and the FDA RWE guidance is evidence that reviewers encountered them often enough to codify the requirements.

### 9.9 What mature teams do differently

1. They draw the graph before looking at associations.
2. They record the graph in the protocol or analysis plan.
3. They list the backdoor paths and the adjustment set, not only the variables.
4. They mark unmeasured confounders explicitly.
5. They run a negative control and an E-value.
6. They report the crude and adjusted estimates and explain the difference.
7. They refuse to adjust for post-treatment variables without a direct-effect estimand.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Adjusting for a collider | A post-treatment variable improves fit and changes the estimate | Conditioning opens a path | Bias away from the truth | Draw the DAG; exclude descendants of treatment |
| Adjusting for a mediator | The treatment coefficient shrinks when a post-treatment variable is added | The indirect path is blocked | Total effect underestimated | Define the estimand; exclude the mediator |
| "Adjust for everything" | The model has every available variable | Some are colliders or mediators | Unpredictable bias | Use the backdoor criterion |
| Stepwise selection | The adjustment set depends on the sample | Selection on associations | Invalid intervals; collider inclusion | Pre-specify from the graph |
| Missing backdoor path | The graph omits an unmeasured common cause | The adjustment set is insufficient | Residual confounding | Mark U nodes; sensitivity analysis |
| M-bias | Adjusting for a variable that is a common effect of two causes | Conditioning on a collider | Bias in the wrong direction | Leave the collider unadjusted |
| Selection on a collider | The cohort is defined by a post-treatment event | Conditioning on selection | Spurious association | Redesign the cohort |
| Positivity violation | A stratum has no treated or no control patients | The effect is not estimable there | Extrapolation | Restrict the population; report |
| Inconsistent treatment definition | Multiple doses or agents are pooled | Consistency is violated | The estimate refers to a mixture | Define the treatment precisely |
| Proxy treated as the construct | Cost used as a proxy for need | The proxy has its own causes | Biased allocation | Validate the proxy; use the construct |
| Graph drawn after the analysis | The graph matches the variables that gave the desired result | Reverse engineering | The assumptions are not pre-specified | Draw the graph first |
| Unmeasured confounding ignored | No sensitivity analysis | The key assumption is untested | Overconfident conclusion | E-value or bias analysis |
| Time-varying confounding | Baseline adjustment of a variable affected by past treatment | The confounder is a mediator of past treatment | Biased effect | g-methods; target trial emulation |
| Front-door misuse | A mediator is assumed unconfounded without evidence | The front-door criterion fails | Biased estimate | State and defend the assumptions |
| Conditioning on a descendant of a collider | A post-treatment lab or utilization variable is adjusted | Descendants of colliders open paths | Bias | Exclude all descendants of colliders |
| Using the graph to justify a preferred answer | The graph changes when the estimate changes | Reverse engineering | Untrustworthy assumptions | Lock the graph before analysis |
| Ignoring interference | The treatment of one patient affects another's outcome | Vaccination or shared-resource effects | Biased effect | Cluster designs; partial interference models |
| Conditioning on a proxy outcome | Cost is used as a stand-in for need | The proxy has its own causes | Biased allocation | Validate the proxy |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The DAG is correct | The backdoor criterion is conditional on the graph | A missing edge or a wrong direction | Testable implications; domain review |
| No unmeasured confounding | The adjustment set identifies the effect | An unmeasured common cause | E-value; negative control |
| Consistency | The observed treatment corresponds to the defined treatment | Multiple versions of the treatment | Define the treatment narrowly; sensitivity analysis |
| Positivity | Every stratum has both treatment groups | Structural zeros | Cross-tabulate; propensity overlap |
| Correct measurement | The variables represent the constructs | Misclassification of treatment, outcome, or confounder | Validation sample |
| No selection on a collider | The cohort is not defined by a post-treatment event | Enrollment or restriction on a collider | Redesign; sensitivity analysis |
| Correct functional form | Standardization or regression is correctly specified | Nonlinearity or interaction | Splines; alternative specifications |
| No interference | One patient's treatment does not affect another's outcome | Vaccination, herd effects, shared resources | Cluster designs; partial interference models |
| Stable treatment strategy | The treatment does not change over follow-up | Time-varying treatment | g-methods |
| Correct temporal order | Causes precede effects | Simultaneous measurement | Timeline review |
| No interference | One patient's treatment does not affect another's outcome | Herd effects or shared resources | Cluster-randomized design |
| Correct proxy validity | A proxy represents the construct | The proxy has its own causal drivers | Validation study |

**Limits of this topic.** A DAG identifies an effect only under its assumptions; it does not test them. The backdoor criterion is sufficient, not necessary: some effects are identified by the front-door criterion or by instruments when no backdoor set exists, and those methods require stronger assumptions. The worked example is a two-stratum table with no sampling uncertainty in the stratum risks beyond the binomial counts; a real analysis would need a regression or weighting estimator with robust intervals. The graph is a communication and design tool, not a substitute for subject-matter knowledge.

## 12. Exercises

### Exercise 1. Classify the structures (recall, 30 minutes)

For each pattern, name the structure, state whether adjustment is appropriate for a total effect, and give a clinical example:

1. Treatment → adherence → outcome.
2. Insurance status → treatment and → outcome.
3. Treatment → hospitalization ← outcome.
4. Treatment → biomarker → outcome, and the biomarker is measured at 6 weeks.
5. Age → treatment and → outcome, and age also → frailty → outcome.

**Deliverable.** A five-row table.

**Acceptance criteria.**

- The structure names are correct.
- The adjustment decision is justified by the path, not by the variable's association with the outcome.
- The clinical examples are specific.
- The mediator case states which estimand the adjustment would target.

### Exercise 2. Derive an adjustment set (application, 50 minutes)

**DAG.** A is treatment, Y is outcome. C1 and C2 are common causes of A and Y. C2 also causes M, and M causes Y. A causes H, and Y causes H. U is an unmeasured common cause of A and Y.

**Deliverable.** All paths between A and Y, the backdoor paths, a sufficient adjustment set (if one exists), and a statement about U.

**Acceptance criteria.**

- Every path is listed and classified.
- The set blocks all measured backdoor paths.
- M and H are excluded with a reason.
- The answer states that U prevents identification and names a sensitivity analysis.

### Exercise 3. Detect M-bias (critique, 45 minutes)

**Scenario.** An analyst adjusts for "health-system engagement," a variable caused by both socioeconomic status and health literacy. Socioeconomic status affects treatment, and health literacy affects the outcome. The adjusted estimate differs from the crude estimate.

**Deliverable.** A DAG, an explanation of why engagement is a collider, and a recommendation.

**Acceptance criteria.**

- The DAG shows engagement as a common effect.
- The explanation uses the collider rule, not "confounding."
- The recommendation is to exclude engagement from the adjustment set.
- The answer states what the crude and adjusted estimates would look like if the collider were the only issue.

### Exercise 4. Mediator or confounder (application, 40 minutes)

**Scenario.** A program increases follow-up visits, and follow-up visits reduce readmission. The analyst adjusts for follow-up visits and finds no effect of the program.

**Deliverable.** Two estimands (total and direct), the adjustment set for each, and a recommendation for a payer deciding whether to fund the program.

**Acceptance criteria.**

- The total-effect set excludes follow-up visits.
- The direct-effect set includes them, with the mediation assumptions stated.
- The payer recommendation uses the total effect.
- The answer explains why the adjusted result is not evidence that the program fails.

### Exercise 5. Redraw a vendor's causal claim (critique, 60 minutes)

**Claim.** "Our program reduced costs after adjusting for age, sex, comorbidity, and post-enrollment specialist visits."

**Deliverable.** A DAG for the claim, the identification problem, a corrected adjustment set, and three questions for the vendor.

**Acceptance criteria.**

- Specialist visits are identified as a post-treatment variable.
- The DAG shows the possible mediator or collider role.
- The corrected set excludes the post-treatment variable for a total-effect claim.
- The questions ask for the DAG, the timing of measurement, and the estimand.

### Exercise 6. Write the assumptions ledger (design, 45 minutes)

**Scenario.** An observational comparison of two diabetes drugs, with claims data.

**Deliverable.** A one-page ledger with the four identification assumptions, the evidence for each, the test (if any), and the consequence of failure.

**Acceptance criteria.**

- Consistency, positivity, exchangeability, and measurement error are all covered.
- Each assumption has a concrete check or a stated limitation.
- The exchangeability entry names the adjustment set and the unmeasured confounders.
- The consequence of failure is described in decision terms.

### Exercise 7. Audit an "adjusted for everything" analysis (critique, 50 minutes)

**Claim.** "We adjusted for all variables in the dataset, so confounding is controlled."

**Deliverable.** A 500-word critique with a DAG-based counterexample, the conditions under which the claim could be true, and the corrected analysis plan.

**Acceptance criteria.**

- The critique names at least one collider and one mediator as counterexamples.
- It explains why adjustment can increase bias.
- It states the conditions under which "adjust for everything" is defensible (no post-treatment variables, no colliders).
- The corrected plan uses a pre-specified adjustment set.

### Exercise 8. Design a DAG-based analysis plan (design, 120 minutes)

**Scenario.** A health system wants to know whether a remote monitoring program reduces hospitalizations. Data include enrollment, claims, EHR vitals, and a patient survey.

**Deliverable.** A two-page plan with the estimand, the DAG, the adjustment set, the estimator, the positivity check, the testable implications, and the sensitivity analyses.

**Acceptance criteria.**

- The DAG has a time zero and marks unmeasured variables.
- The adjustment set is derived from the backdoor criterion, not from univariate tests.
- The estimator matches the estimand.
- At least two testable implications and two sensitivity analyses are listed.
- The plan states what would change the decision.

## 13. Self-check answers

**Answer 1 (Exercise 1 structures).**

| Pattern | Structure | Adjust for total effect? | Example |
|---|---|---|---|
| Treatment → adherence → outcome | Chain (mediation) | No | Program → medication adherence → readmission |
| Insurance → treatment and outcome | Fork (confounding) | Yes | Coverage → drug choice and → outcome |
| Treatment → hospitalization ← outcome | Collider | No | Treatment and stroke both cause admission |
| Treatment → biomarker → outcome | Chain (mediation) | No for total effect | Drug → LDL → stroke |
| Age → treatment and outcome, age → frailty → outcome | Fork plus chain | Yes for age; no for frailty if it is a descendant of age and a cause of outcome | Age → statin and → stroke |

**Answer 2 (Exercise 2 adjustment set).**

The paths are:

- A → Y (causal).
- A ← C1 → Y (backdoor; block with C1).
- A ← C2 → Y (backdoor; block with C2).
- A ← C2 → M → Y (backdoor through M; block with C2, not M).
- A → H ← Y (collider; do not adjust).
- A ← U → Y (backdoor through U; not blockable).

A sufficient measured set is {C1, C2}. M is a mediator of the C2 path and must not be adjusted for a total effect. H is a collider and must not be adjusted. U prevents identification, so the estimate is biased by an unknown amount; an E-value or a bias analysis is required, and the conclusion must be conditional.

If C2 is measured with error, the set {C1, C2} blocks the path only partially, and residual confounding remains. That is a measurement problem, not a graphical one, and it is why the assumptions ledger includes measurement error alongside exchangeability.

**Answer 3 (Exercise 4 mediator).**

The total effect of the program includes the increase in follow-up visits. The total-effect adjustment set excludes follow-up visits; the direct-effect set includes them and requires the assumption that no unmeasured confounding of the visit-readmission relationship exists. For a payer deciding whether to fund the program, the total effect is the relevant estimand: the program's value includes the visits it causes. The adjusted analysis showing no effect answers "does the program work through channels other than visits?" not "does the program work?"

**Answer 4 (Exercise 6 ledger outline).**

| Assumption | Evidence | Test | Consequence of failure |
|---|---|---|---|
| Consistency | The drug classes are defined by formulary tier and dose | Check the distribution of doses by site | The estimate refers to a mixture of doses |
| Positivity | Both drugs are available at all sites | Cross-tabulate drug by site and comorbidity | Some strata are not comparable |
| Exchangeability | Adjustment for age, sex, comorbidity, prior utilization, and site | E-value; negative control outcome | Residual confounding |
| No measurement error | Claims coding validated in a sample | Chart review of 100 patients | Misclassified exposure or outcome |

**Answer 5 (Exercise 3 M-bias).**

Engagement is caused by socioeconomic status and health literacy. Socioeconomic status affects treatment; health literacy affects the outcome. Conditioning on engagement opens a path between socioeconomic status and health literacy, creating a spurious association between treatment and outcome. The crude estimate may be confounded by socioeconomic status, and the adjusted estimate may be biased by the collider. Neither is automatically correct. The right analysis adjusts for socioeconomic status and health literacy directly, if measured, and leaves engagement out. If those variables are unmeasured, the graph has a U node and the effect is not identified without stronger assumptions.

**Answer 6 (Exercise 7 critique).**

A strong critique:

- "All variables" includes post-treatment variables such as adherence, hospitalization, and discontinuation, which are mediators or colliders.
- Adjusting for a mediator removes part of the effect; adjusting for a collider creates a spurious association.
- The claim would be defensible only if every measured variable were a pre-treatment common cause of treatment and outcome, which is rarely true.
- The corrected plan pre-specifies a DAG, derives a minimal sufficient set, and reports the excluded variables with reasons.
- The plan includes a sensitivity analysis for the variables that could not be measured.

**Answer 7 (Exercise 5 vendor questions).**

1. What is the DAG, and when was it specified?
2. Were specialist visits measured before or after enrollment?
3. What is the estimand, and does it include the effect through specialist visits?

A strong answer notes that specialist visits after enrollment are a post-treatment variable. If the vendor's claim is a total-effect claim, the variable should not be in the adjustment set. If the claim is a direct-effect claim, the mediation assumptions must be stated.

**Answer 8 (Exercise 8 rubric).**

A strong plan:

- Defines the estimand as the effect of the offer of remote monitoring on 12-month hospitalization.
- Draws a DAG with enrollment, age, comorbidity, prior utilization, digital access, and unmeasured health literacy.
- Adjusts for the pre-treatment common causes; excludes post-enrollment utilization.
- Uses IPTW or standardization, with a positivity check by decile of propensity score.
- Lists testable implications: digital access should not predict hospitalization among the unexposed; a negative control outcome should show no effect.
- Runs an E-value and a negative control outcome analysis.
- States the decision threshold and what evidence would change the recommendation.

The exercise is graded on the graph, not on the estimate. A plan with a defensible graph and a weak estimator is easier to repair than a plan with a sophisticated estimator and no identification strategy. The reviewer should be able to redraw the graph from the written description and arrive at the same adjustment set.

A useful self-test before submitting any of these exercises: cover the estimate and ask whether the graph and the estimand alone would justify the design. If the answer depends on the estimate being large or significant, the design was chosen after the fact.

## 14. Interview questions

**1. What is a confounder, and how does a DAG define it?**

A strong answer says a confounder is a common cause of treatment and outcome, and that the DAG defines it by the backdoor path it creates, not by its association with the outcome. A weak answer says a confounder is "a variable that differs between groups."

**2. Why is adjusting for a collider harmful?**

A strong answer explains that conditioning on a common effect opens a path between its causes, creating a spurious association. It gives a clinical example such as hospitalization. A weak answer says it "introduces bias" without the mechanism.

**3. What is the backdoor criterion?**

A strong answer states both conditions: no descendants of treatment, and block every backdoor path. It connects the criterion to standardization. A weak answer says "adjust for confounders."

**4. What is the difference between a mediator and a confounder?**

A strong answer defines the mediator as on the causal path and the confounder as a common cause, and ties the adjustment decision to the estimand. A weak answer says both are "variables to control."

**5. What is M-bias?**

A strong answer describes a collider that is a common effect of two causes of treatment and outcome, and explains that adjusting for it creates bias. A weak answer has not heard of it or calls it confounding.

**6. How do you choose an adjustment set?**

A strong answer uses the DAG, the backdoor criterion, and a preference for pre-treatment variables, and reports the excluded variables. A weak answer uses stepwise regression or "all significant variables."

**7. What are the four identification assumptions?**

A strong answer lists consistency, positivity, exchangeability, and no measurement error, and gives an example of each. A weak answer lists only "no confounding."

**8. What is positivity, and how do you check it?**

A strong answer defines it as a positive probability of each treatment in every covariate stratum and describes cross-tabulation and propensity overlap. A weak answer says "the propensity scores are between 0 and 1."

**9. Why can a randomized trial still be biased?**

A strong answer lists selection, attrition, non-adherence, unblinding, measurement, and analysis, and notes that randomization addresses confounding but not the other threats. A weak answer says randomization eliminates all bias.

**10. What is the front-door criterion?**

A strong answer describes identification through a mediator when confounding is unmeasured, and states the assumptions: no unmeasured confounding of treatment-mediator and mediator-outcome, and no effect of treatment on the outcome except through the mediator. A weak answer says it is "another adjustment method."

**11. What is the E-value?**

A strong answer defines the minimum strength of association an unmeasured confounder would need with both treatment and outcome to explain away the result, and says it is a sensitivity summary. A weak answer calls it a correction.

**12. When would you not use a DAG?**

A strong answer says when the variables and their timing are unknown, when the question is predictive rather than causal, or when the graph cannot be defended and an instrument or trial is needed. A weak answer says "never."

A strong answer also notes that a DAG is still useful for a prediction problem, because it identifies variables that must not be used as predictors if the model will inform a decision. The graph is not only for effect estimation.

## 15. Key terms

- **Adjustment set:** variables used to block backdoor paths.
- **Ancestor:** a node that can reach another by following arrows.
- **Backdoor criterion:** a sufficient condition for identifying a total effect by adjustment.
- **Backdoor path:** a path from treatment to outcome that starts with an arrow into treatment.
- **Chain:** a path A → M → Y.
- **Collider:** a common effect of two variables.
- **Confounder:** a common cause of treatment and outcome.
- **Consistency:** the observed outcome under treatment equals the potential outcome under that treatment definition.
- **d-separation:** a graphical criterion for conditional independence.
- **DAG:** a directed acyclic graph of assumed causal effects.
- **Descendant:** a node reachable from another by following arrows.
- **Direct effect:** the effect not through a specified mediator.
- **Edge:** an assumed direct causal effect.
- **Exchangeability:** the groups would have the same outcome under the same treatment given the adjustment set.
- **Fork:** a path A ← C → Y.
- **Front-door criterion:** identification through a mediator.
- **Indirect effect:** the effect through a mediator.
- **Instrumental variable:** a variable that affects treatment but not the outcome except through treatment.
- **M-bias:** bias from conditioning on a collider that is a common effect of causes of treatment and outcome.
- **Mediator:** a variable on the causal path.
- **Minimal sufficient set:** a sufficient adjustment set with no redundant variables.
- **Node:** a variable in a graph.
- **Overadjustment:** adjustment for a mediator or collider.
- **Path:** a sequence of edges between two nodes.
- **Positivity:** a positive probability of each treatment in every covariate stratum.
- **Proxy confounder:** a measured variable standing in for an unmeasured confounder.
- **Selection bias:** bias from conditioning on a variable affected by treatment or outcome.
- **Single-world intervention graph:** a graph representing an intervention and the observed data.
- **Time-varying confounding:** confounding by a variable affected by past treatment.
- **Total effect:** the effect through all causal paths.
- **Unmeasured confounder:** a common cause not in the data.
- **Collider stratification:** conditioning on a collider or its descendant.
- **Adjustment formula:** the standardization sum over covariate strata.
- **Causal contrast:** the comparison of potential outcomes under two treatment strategies.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. Pearl, J. "Causal diagrams for empirical research." *Biometrika* 1995;82(4):669-710. https://doi.org/10.1093/biomet/82.4.669
2. Greenland, S., Pearl, J., Robins, J. M. "Causal Diagrams for Epidemiologic Research." *Epidemiology* 1999;10(1):37-48. https://doi.org/10.1097/00001648-199901000-00008
3. Hernán, M. A., Robins, J. M. "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available." *American Journal of Epidemiology* 2016;183(8):758-764. https://doi.org/10.1093/aje/kwv254
4. Hernán, M. A., Robins, J. M. *Causal Inference: What If*. Chapman and Hall/CRC, 2020.
5. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
6. United States Food and Drug Administration. *Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products*. August 2023. https://www.fda.gov/media/171667/download
7. Textor, J., van der Zander, B., Gilthorpe, M. S., Liśkiewicz, M., Ellison, G. T. H. "Robust causal inference using directed acyclic graphs: the R package 'dagitty'." *International Journal of Epidemiology* 2016;45(6):1887-1894. https://doi.org/10.1093/ije/dyw341
8. DAGitty. *Drawing and analyzing causal diagrams*. https://dagitty.net/
9. Manson, J. E., Hsia, J., Johnson, K. C., et al. "Estrogen plus Progestin and the Risk of Coronary Heart Disease." *New England Journal of Medicine* 2003;349:523-534. https://doi.org/10.1056/NEJMoa030808
10. MERIT-HF Study Group. "Effect of metoprolol CR/XL in chronic heart failure: Metoprolol CR/XL Randomised Intervention Trial in Congestive Heart Failure (MERIT-HF)." *Lancet* 1999;353(9169):2001-2007. https://doi.org/10.1016/S0140-6736(99)04440-2
11. Seymour, C. W., Liu, V. X., Iwashyna, T. J., et al. "Assessment of Clinical Criteria for Sepsis." *JAMA* 2016;315(8):762-774. https://doi.org/10.1001/jama.2016.0288
12. Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S. "Dissecting racial bias in an algorithm used to manage the health of populations." *Science* 2019;366(6464):447-453. https://doi.org/10.1126/science.aax2342
13. Wang, S. V., Schneeweiss, S., and the RCT-DUPLICATE Initiative. "Emulation of Randomized Clinical Trials With Nonrandomized Database Analyses: Results of 32 Clinical Trials." *JAMA* 2023;329(16):1376-1385. https://doi.org/10.1001/jama.2023.4221
14. Rothman, K. J., Greenland, S., Lash, T. L. *Modern Epidemiology*. 3rd edition. Lippincott Williams and Wilkins, 2008.
15. Imbens, G. W., Rubin, D. B. *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press, 2015.
16. Rosenbaum, P. R., Rubin, D. B. "The central role of the propensity score in observational studies for causal effects." *Biometrika* 1983;70(1):41-55. https://doi.org/10.1093/biomet/70.1.41
17. VanderWeele, T. J., Ding, P. "Sensitivity Analysis in Observational Research: Introducing the E-Value." *Annals of Internal Medicine* 2017;167(4):268-274. https://doi.org/10.7326/M16-2607

> **Note:** Sources 1 and 2 are the foundational methodological references for the backdoor criterion and causal diagrams. Sources 5 and 6 are the regulatory documents that require the estimand and design transparency. Sources 9 to 13 are the case evidence in Section 9; the inferences drawn from them are labeled in the text.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_missing-data.md`](04_missing-data.md) — missing confounders and selection.
- Regression adjustment: [`02_regression-models.md`](02_regression-models.md) — estimating the effect once the set is chosen.
- Propensity scores: [`06_propensity-scores-and-weighting.md`](06_propensity-scores-and-weighting.md) — a design-based estimator for the same estimand.
- Target trial emulation: [`07_target-trial-emulation.md`](07_target-trial-emulation.md) — the design that makes the DAG operational.
- Sensitivity analysis: [`08_sensitivity-analysis.md`](08_sensitivity-analysis.md) — E-values and negative controls.
- Survey file: [`../../04_statistics_causal_inference.md`](../../course/04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — Lab 4.1 builds a DAG and finds adjustment sets.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md) — assumptions ledger.
- Survival analysis: [`03_survival-analysis.md`](03_survival-analysis.md) — time-to-event outcomes and competing risks.
- Subgroups: [`09_subgroups-and-heterogeneity.md`](09_subgroups-and-heterogeneity.md) — effect modification and interaction.
- Reporting: [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) — stating assumptions alongside estimates.
- Estimation: [`01_estimation-and-inference.md`](01_estimation-and-inference.md) — intervals for standardized effects.
- Study design: [`../../05_study_design_evidence.md`](../../course/05_study_design_evidence.md) — design options that block backdoor paths.
- Module 18: [`../../18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md) — evaluation and reporting standards.
