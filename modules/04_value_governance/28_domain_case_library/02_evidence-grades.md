# Evidence Grades A–E: Reading Cases Without Overclaiming

> Module 28 — Domain Case Library and Sector Playbooks  
> Topic 02 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 05 experimentation, Module 07 forecasting, Module 26 evaluation  
> Version 2026.5

## 1. Why this topic exists

A director asks whether to invest in an inventory optimization product.
A presentation reports that a named company cut stock by 20% after installing it.
The slide does not say which stock, which sites, or which months.
It does not explain whether the business also closed a warehouse or reduced its product range.
The director needs a decision, not just a memorable number.

A case library helps practitioners learn from other organizations.
It also collects claims created for different purposes.
A journal article, a securities filing, a customer story, and a news report do not answer the same question.
They differ in scrutiny, incentives, measurement, and access to the underlying data.

The course uses five evidence grades as a source-provenance shorthand.
Grade A is peer-reviewed or audited material.
Grade B is a regulatory filing or official statistic.
Grade C is a company case study or investor claim.
Grade D is vendor marketing.
Grade E is a press summary or anecdote.
The definitions follow the module survey and index; they are not a universal scientific hierarchy.

The business risk is not simply choosing a lower-letter source.
A high-grade source can document an outcome without proving what caused it.
An audited revenue figure can be accurate while a management explanation of its cause is unsupported.
An official statistic may be authoritative for a national total but inappropriate for a local warehouse.
A peer-reviewed benchmark can rank algorithms on a fixed dataset without proving a commercial return.

Conversely, a vendor or employee may supply a useful lead.
A practitioner can use that lead to identify a decision, identify a missing metric, or design a test.
The practitioner must not silently convert the lead into independent proof.

This topic gives you a reproducible way to label the source, inspect the claim, assess causality, compare cases, and decide what further evidence is worth buying.
You will make the distinction visible in a claim register rather than hide it in a footnote.

### The decisions at stake

A procurement team decides whether a supplier's result is a purchasing requirement or merely a question for diligence.
A forecasting team decides whether a benchmark result belongs in a model-selection experiment.
A chief financial officer decides whether an estimated saving belongs in a budget commitment.
A risk committee decides whether evidence is sufficient for automated control of inventory or transportation.
A case author decides whether the case can support a lesson, or only suggest an analogy.

The same source can support one of these decisions but not another.
A company may have installed a system even if the system's effect has not been established.
The fact of installation and the claim of savings therefore need separate rows.

### The cost of a weak inference

Suppose a planner forecasts savings of USD 1 million per year by copying a supplier's percentage improvement.
The percentage was based on a selected six-week pilot, not a full year of comparable operations.
Treating it as a commitment can underfund inventory and overstate the business case.
The avoidable mistake is not reading the case.
The mistake is transferring its conclusion without its population, denominator, comparison, and uncertainty.

> **Note:** An evidence grade describes the source category used in this course.
> It is not a probability that the statement is true, a guarantee of independence, or a substitute for a study-quality assessment.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a source A–E using the course definitions and cite the exact item used.
2. **Separate** a factual observation, an attributed claim, an inference, and a synthetic assumption in a case note.
3. **Trace** a headline to the primary source and identify when several reports share one original claim.
4. **Compute** absolute and relative changes, a difference-in-differences estimate, and a simple net-value scenario without calling them certainty.
5. **Critique** causal claims using a counterfactual, comparison group, selection mechanisms, and alternative explanations.
6. **Compare** cases on decision, population, outcome definition, time, design, and business economics before transferring a result.
7. **Design** an evidence-gathering plan and recommend a decision gate proportionate to the downside.
8. **Document** limitations, conflicts, changes in source version, and an appropriate review owner.

Exercises 1 and 2 assess classification and source tracing.
Exercises 3 and 4 assess calculations and causal critique.
Exercises 5 and 6 assess comparison and decision gates.
Exercises 7 and 8 assess register design and communication.
Self-check answers and rubrics appear in Section 13.

## 3. The core mental model

```text
Decision and proposed action
          |
          v
Specific claim: population, intervention, outcome, period, comparator
          |
          v
Exact source and provenance ------> course source grade A / B / C / D / E
          |                                      |
          v                                      v
Extract observation and method          Document source-specific limits
          |                                      |
          +------------------+-------------------+
                             v
              Assess measurement and causal design
                             |
                             v
             Compare context and local transferability
                             |
                             v
             Decide: use fact / investigate / test / stop
                             |
                             v
                 Record owner, uncertainty, next review
```

Read the diagram from top to bottom.
Start with the decision, because evidence that is adequate to describe a market may be inadequate to change an automated replenishment policy.
Define the exact claim before assigning a grade.
A single publication can contain several claims with different provenance and support.

The source-grade branch records where the statement comes from.
The appraisal branch asks whether the statement is accurate, relevant, measured consistently, and causally defensible.
Neither branch replaces the other.
Then compare the case with the local decision before recommending an action.

### Four statements, four evidential jobs

**Observation:** A report records an inventory value at a particular date.
The report's methods and underlying record determine what the number means.

**Attributed claim:** A supplier says its product caused the reduction.
The attribution tells the reader who is making the claim, not whether the claim is true.

**Inference:** An analyst reasons that a changed product mix might explain part of the reduction.
That is a plausible alternative explanation until independently investigated.

**Synthetic assumption:** An exercise stipulates a stock balance and a cost of capital.
The values are invented for calculation and are not evidence about any real company.

Use all four kinds of statement when useful, but label them visibly.
The label travels with the claim when it is quoted, graphed, or summarized.

### The grade is not the conclusion

The grade is assigned to the source type used for a particular statement.
It is not automatically inherited by every statement on the same page.
A regulatory filing may contain audited financial statements and separate management commentary.
Identify the audited scope; do not call an unaudited operational explanation audited merely because it appears in the filing.
Likewise, a press article quoting a company report is a press source until the analyst actually retrieves and checks the company report.

The course shorthand permits A for peer-reviewed or audited evidence and B for regulatory filings or official statistics.
A document can meet more than one description.
Record the grade based on the specific material relied on and explain the overlap.
For example, an audited financial statement inside a regulatory filing may be recorded A for the audited line item; the filing itself is B as an official filing.
Do not upgrade surrounding claims to A.

### Evidence has several independent dimensions

| Dimension | Question | Example of a negative finding |
|---|---|---|
| Provenance | Who produced and published it? | Marketing copy does not identify its data owner. |
| Scope | What claim does the source actually support? | Audit opinion excludes an operational savings estimate. |
| Measurement | How was the metric constructed? | Inventory percentage lacks the starting balance. |
| Design | What would have happened without intervention? | Before-and-after comparison has no control. |
| Precision | How uncertain is the estimate? | No interval or sample size is reported. |
| Relevance | Does it match the decision? | A daily forecast competition is used to price a network redesign. |
| Independence | Do corroborating reports use separate data? | Five articles repeat one press release. |
| Incentives | Who benefits from a positive result? | The seller selected the customer case. |
| Timeliness | Does the period still represent current conditions? | The case predates a major network change. |
| Consequence | What happens if the inference is wrong? | An untested model controls critical medicines. |

A grade is a label in the provenance dimension.
A decision combines all dimensions; a letter alone does not answer them.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or record |
|---|---|---|---|
| Evidence grade | Course classification of the source used for a particular claim. | Treating the letter as a probability of truth. | A–E plus rationale. |
| Claim | A statement that could be checked or explicitly identified as judgment. | Combining implementation and effect in one sentence. | Claim ID and exact wording. |
| Observation | A recorded measurement under a stated method. | Assuming an observed change is an intervention effect. | Value, unit, period, population. |
| Inference | A conclusion derived from observations and assumptions. | Presenting a plausible explanation as a documented fact. | Reasoning and assumptions. |
| Synthetic case | An invented scenario used for instruction or testing. | Presenting made-up results as real case evidence. | Explicit synthetic label. |
| Primary source | The original source of the specific observation or claim. | Assuming a page is primary because it appears first in search. | Author, document, locator. |
| Secondary source | A source describing another source or event. | Counting a quotation as an independent observation. | Citation and original source. |
| Provenance | Traceable origin and transformation of evidence. | Recording only a homepage URL. | Publisher, date, version, locator. |
| Peer review | Evaluation of a manuscript by expert reviewers before publication. | Treating review as replication or a causal guarantee. | Journal, article, methods. |
| Audit | Independent examination against a defined reporting framework and scope. | Assuming all operational claims in a filing were audited. | Opinion, entity, period, scope. |
| Official statistic | A statistic published by a public statistical authority or official body. | Assuming national and local populations are equivalent. | Agency, series, revision, unit. |
| Regulatory filing | A document filed with a regulator under a reporting process. | Assuming the regulator verified every management assertion. | Filing type, date, section. |
| Company claim | A statement by the organization whose case is described. | Treating self-report as independent measurement. | Named entity, author, period. |
| Vendor marketing | Material promoting a supplier's product or services. | Treating a selected testimonial as representative. | Seller, customer, method. |
| Press summary | Journalistic report or synthesis rather than the original data source. | Using repeated coverage as independent replication. | Article, quoted sources. |
| Anecdote | Individual report without a systematic sampling design. | Generalizing one person's experience to a population. | Speaker, situation, date. |
| Metric | A defined quantity used to assess performance. | Comparing differently defined rates with the same name. | Numerator, denominator, unit. |
| Denominator | The exposure, population, or baseline dividing a rate or percentage. | Reporting a percentage without its base. | Units, orders, dollars, or time. |
| Grain | Level of detail of a data record. | Comparing stock-keeping-unit daily data to site-month totals without aggregation rules. | Entity and time bucket. |
| Baseline | Reference value or period used for comparison. | Calling the prior period a counterfactual. | Population, dates, metric. |
| Comparator | A reference group, period, or policy used for contrast. | Assuming every comparator is exchangeable. | Group definition and dates. |
| Counterfactual | Outcome for the same units under an alternative action. | Claiming it was directly observed for treated units. | Target population, horizon. |
| Confounder | A factor associated with intervention assignment and the outcome that can distort attribution. | Calling any correlated feature a proven confounder. | Candidate factor and pathway. |
| Selection bias | Distortion arising from how units or cases enter the analysis. | Treating published success stories as a random sample. | Selection rule and excluded units. |
| Survivorship bias | Selection that excludes failures or discontinued cases. | Reporting only sites remaining after rollout. | Initial and final cohorts. |
| Difference in differences | Change in a treated group minus change in a comparison group. | Assuming subtraction alone establishes causality. | Outcome units or percentage points. |
| Parallel trends | Assumption that treated and comparison outcomes would have changed similarly absent treatment. | Treating pre-period agreement as proof of the future counterfactual. | Trend and horizon. |
| Spillover | Effect of treatment on the comparison group or other units. | Treating connected facilities as unaffected controls. | Exposure mechanism. |
| Relative change | Difference divided by the baseline. | Confusing a percentage change with percentage points. | Percent of baseline. |
| Percentage-point change | Arithmetic difference between two percentages. | Calling 4% to 3% a 1% relative reduction. | Percentage points. |
| Uncertainty interval | Range compatible with a specified model and procedure. | Calling a point estimate guaranteed savings. | Bounds, confidence level, method. |
| Transportability | Extent to which an effect or finding applies to another setting. | Assuming a result transfers because industries share a name. | Source and target populations. |
| Corroboration | Support from an additional source or observation. | Counting dependent reports as independent confirmation. | Evidence links and dependency. |
| Revision | Change to a previously published data item or document. | Quoting the latest value as if available at the decision date. | Version and publication date. |
| Decision gate | Explicit condition to continue, pause, or stop an action. | Treating an interesting story as authorization to scale. | Owner, threshold, deadline. |
| Forecast | An estimate of a future outcome under stated assumptions. | Calling a scenario a promise. | Value, horizon, uncertainty. |
| Plan | Intended actions and resource allocations. | Calling an unapproved plan a commitment. | Action, owner, dates. |
| Target | Desired performance level. | Treating the target as observed evidence. | Metric and deadline. |
| Commitment | Authorized obligation to deliver a specified result or action. | Committing savings based only on a case analogy. | Approver, scope, conditions. |

## 5. The mechanics, step by step

### Step 1 — Frame the local decision

**Input:** A proposed action, a decision owner, a deadline, and an estimate of downside.

Write the choice as alternatives.
For example, compare continuing the existing reorder policy, running a limited test, and deploying a new policy across all sites.
Identify which outcomes matter: service, working capital, labor, customer harm, and cash cost may pull in different directions.

Specify the decision horizon.
A six-week lead-time experiment and a three-year warehouse investment cannot use the same evidence threshold automatically.
Record the reversibility of the action.
A controlled offline replay may be easy to reverse; a contract or network closure may not be.

**Output:** A decision statement with alternatives, scope, owner, and date.

**Branch:** If the action could affect safety or legal obligations, involve the responsible specialists before deciding on a test.
The course grade does not grant authority to ignore those obligations.

### Step 2 — Split the narrative into atomic claims

**Input:** The exact slide, article, filing, or interview transcript.

Replace a sentence such as “The new system cut stock and improved service across the network” with separate claims.

1. The system was installed.
2. A particular set of sites used it.
3. A defined inventory measure changed over a stated period.
4. A defined service measure changed over the same period.
5. The system caused some portion of those changes.
6. The same effect is expected locally.

These statements require different evidence.
The existence of a deployment is not a measurement of its effect.
A measurement of change is not by itself an estimate of causation.
An estimate in one population is not automatically a prediction for another.

Preserve the source's words as a quotation only where needed.
Then write your own normalized claim with units and period.
If a source says “stock fell by 20%,” record whether it means value, units, days of supply, or another measure.
If the source does not say, mark the field unknown.

**Output:** One claim ID per checkable statement, plus a list of missing qualifiers.

### Step 3 — Find the exact source

**Input:** A link, bibliographic reference, screenshot, or person making the claim.

Open the original item if accessible.
Record the title, author or institution, publisher, publication date, version, URL or stable identifier, and page, table, or section.
Record the date you accessed it if an online page can change.
For a data series, preserve the vintage and the download or query definition.
For an interview, record the speaker's role, date, and permission to use the statement.

Trace quoted material backwards.
A magazine may quote a vendor, whose customer story quotes a company executive, whose statement summarizes an internal dashboard.
Map the chain rather than counting the magazine, vendor, and company as three independent measurements.
If the dashboard cannot be inspected, note that limit.

Check whether the source is the original evidence for this specific statement.
A journal article can be primary for its own experiment and secondary for an example in its introduction.
A filing can be primary for a company's disclosed statement without being primary for an external market figure it quotes.

**Output:** A source record and a dependency graph or source-chain note.

**Branch:** If the original is unavailable, cite the accessible report as secondary and say that the underlying method was not checked.
Do not guess at details from the source's title.

### Step 4 — Assign the course grade

**Input:** The identified source and the portion used for the claim.

| Grade | Course source category | Typical permissible use | Essential caveat |
|---|---|---|---|
| A | Peer-reviewed or audited. | Strong evidence for the particular claim within the study or audit scope. | Peer review or audit is not proof of causal attribution or transfer. |
| B | Regulatory filing or official statistic. | Reliable facts within the source's scope and definitions. | A filing is not a blanket validation of all assertions; statistics may be revised. |
| C | Company case study or investor claim. | Directional company perspective and evidence of what the company reports. | Selection and self-interest can affect the account; causality needs design evidence. |
| D | Vendor marketing. | Hypothesis generation and questions for independent diligence. | A selected promotional result is not independent proof. |
| E | Press summary or anecdote. | A lead to trace or contextual information with attribution. | Original methods, context, and selection may be missing. |

This is the course convention, not a claim that every A item is better designed than every B, C, D, or E item.
An agency dataset can be more appropriate than a journal article for an official population total.
An eyewitness account can be the only direct lead to an operational incident.
The grade tells the next reader what kind of source was used; appraisal explains what it can actually establish.

Where categories overlap, state the relevant scope.
An official filing containing audited statements is not uniformly A or B for every sentence.
A company press release hosted on a regulator's filing system is still a company claim in substance; the filing provides a B record that the statement was filed, not independent verification of the result.
A peer-reviewed paper quoting a vendor's promotional figure does not automatically make that quoted claim peer-reviewed evidence.

**Output:** Grade, source type, exact relevant scope, and brief rationale for each claim.

### Step 5 — Check what was measured

**Input:** The claim, method, and metric definition.

Record numerator and denominator.
For an on-time delivery rate, identify eligible orders, timestamp definitions, treatment of partial shipments, and the promised date version.
For inventory, identify whether the measure is an end-of-period balance, an average, a quantity, or an accounting value.
A single month-end snapshot can move because the counting date moved relative to a replenishment cycle.

Record the population, time bucket, geographic scope, and missingness.
A favorable result for selected facilities may not describe the original rollout cohort.
Check whether business units, stock categories, returns, and third-party inventory are included consistently.
If the measure is reported as currency, record currency and price basis.
If the measure is a rate, record both the rate and the exposure when available.

Inspect whether measurement changed at the same time as the intervention.
A new system can improve data capture and make the measured defect rate rise even if operations do not worsen.
A changed definition of late delivery can make the rate fall without improving physical performance.

**Output:** A metric card: formula, grain, unit, population, dates, exclusions, and comparability risks.

**Branch:** If the denominator or definition is absent, describe the reported number but do not calculate a business benefit from an invented base.

### Step 6 — Separate association from attribution

**Input:** Baseline and follow-up observations plus a description of intervention and comparison.

Ask what would have happened to the same units over the same period without the intervention.
That counterfactual is not observed for treated units.
A credible design approximates it, with assumptions that can be criticized.

Possible designs include randomized assignment, a phased rollout with a defensible comparison, a matched comparison, an interrupted time series, or a carefully specified before-and-after analysis.
Random assignment can reduce systematic allocation differences, but attrition, spillovers, noncompliance, and measurement changes still matter.
Observational comparisons can be informative, but assignment and unmeasured changes demand explicit attention.

Draw a causal story before choosing a calculation.
A promotion may change demand while the model is deployed.
Demand affects inventory and can also influence where management chooses to deploy the model.
Without accounting for this pathway, a pre/post change combines treatment and other changes.

Ask whether the comparison group was selected before outcomes were known.
Ask whether it was exposed indirectly through shared stock, labor, or management.
Ask whether changes in suppliers, policy, price, assortment, or season coincided with the intervention.
An unmeasured simultaneous change is not automatically the explanation, but it is a threat to the claim.

**Output:** An attribution statement that distinguishes observed change, estimated effect, assumptions, and unresolved alternatives.

**Branch:** No defensible counterfactual means describe the change and mark causality unresolved.
Do not replace missing causal evidence with confidence from a persuasive speaker.

### Step 7 — Assess uncertainty and economic significance

**Input:** An estimate, its sample, variability, and economic model.

Record whether the source reports a point estimate, an interval, both, or neither.
Do not invent a confidence interval from a point estimate alone.
A small percentage effect can be economically material at large scale; a large percentage can be immaterial on a small base.
Translate the effect into currency only after checking the applicable base and cost structure.

For inventory, released stock value is generally a one-time working-capital change, not annual profit.
An annual carrying-cost scenario can use average inventory released multiplied by an explicitly assumed annual carrying rate.
Count implementation, subscription, training, and operating costs separately.
State the time horizon and whether benefits are gross, net, recurring, or one-time.

Distinguish statistical uncertainty from scenario uncertainty.
A sensitivity range over assumed carrying rates is not a statistical confidence interval.
A reported confidence interval does not include all uncertainty from implementation, data quality, or transfer.

**Output:** A labeled estimate and an economic sensitivity table, with unmeasured harms and uncertainty listed.

### Step 8 — Test independence and contradictions

**Input:** All references offered for the claim.

Group references by original data source.
A vendor press release, a company blog, and two articles may all use the same customer-provided number.
That is one underlying measurement described four times, not four independent replications.
Different organizations can also depend on the same dataset or analyst.

Look for sources that would challenge the claim, not just support it.
Check whether an earlier release used a different population or date.
Check for retractions, corrections, updated filings, subsequent versions, and disclosed limitations.
Read the methods and limitations, not just the abstract or headline.

When sources disagree, retain the disagreement.
Identify different definitions, dates, units, and populations before concluding one is wrong.
Do not average incompatible numbers.
If the disagreement cannot be resolved, state which figures remain in conflict and what decision is robust to both.

**Output:** An evidence-dependency map and a contradiction log.

### Step 9 — Compare cases on a common basis

**Input:** Two or more case records and the local decision.

Align cases on the question they answer, not the company name.
Use the same columns: decision, intervention, population, measure, period, comparator, source grade, causal design, costs, and limits.
Separate differences in results from differences in measurement.

A forecast-error improvement is not directly comparable with an inventory reduction.
Even two inventory percentages may use different valuation bases or different product sets.
An apparently larger effect in a marketing story is not necessarily stronger evidence than a smaller effect in a controlled evaluation.

Apply the module's five transfer questions:

1. Is the decision the same?
2. Are product economics similar?
3. Is the data environment similar?
4. Is the operating constraint similar?
5. Is the evidence strong enough for the risk?

If three answers are no, treat the source case as an analogy rather than a plan, as the survey specifies.
This rule is a teaching heuristic, not a guarantee that fewer mismatches make a transfer safe.
Even one mismatch can be decisive, for example a safety constraint or an unavailable control variable.
Unknown answers remain unknown; they are not counted as yes.

**Output:** A side-by-side comparison and a list of transfer blockers.

### Step 10 — Choose the evidence action

**Input:** The decision consequence, source grade, appraisal, and transfer assessment.

Possible actions include using the source as a documented fact within its scope, seeking the original data, asking for independent measurement, running an experiment, or declining the proposed action.
A source grade does not prescribe a single action without context.

For a reversible exploratory decision, a C or D source may justify a bounded investigation.
For a large irreversible investment, ask for evidence that addresses the relevant counterfactual, costs, and operating risks.
A benchmark can justify reproducing a method locally, not automatically replacing a production system.

Set a decision gate before collecting outcomes.
State the eligible population, comparison, outcome, guardrails, minimum useful effect, review date, and decision owner.
Specify what happens if evidence is inconclusive.
Separate permission to test from permission to scale.

**Output:** An approved next action and explicit evidence needed for the next gate.

### Step 11 — Write and maintain the claim register

**Input:** Claims, source records, calculations, limitations, and decision gate.

Use one row per claim and a separate source table if several claims cite the same document.
A useful row contains claim ID, exact text, normalized claim, population, period, unit, source locator, source grade, method, evidence dependency, causal status, limitations, reviewer, and next review date.

Record whether a field is unknown, not applicable, or unavailable.
Do not fill an unknown field using a guess that is later mistaken for a source statement.
Preserve historical versions when a source changes.
If a material correction arrives, update both the claim and every dependent decision memo.

Record access limitations.
A private internal audit may be useful locally but cannot be reproduced by a reader without access.
A public source may be accessible but still omit relevant data.

**Output:** A versioned register with accountable ownership and links from decisions back to evidence.

### Step 12 — Challenge the comparison before interpreting it

**Input:** A proposed treatment and comparison, with an assignment timeline.

Start by writing down when the decision to assign a site was made.
Then ask what the decision maker knew at that time.
If sites were selected because their inventory had recently spiked, a subsequent fall may partly reflect regression toward their usual level.
The fall can occur without any effect of the new policy.

This does not mean a high-inventory site should never be selected for a pilot.
It means that the selection rule must be recorded and reflected in the comparison.
Where practical, choose comparable sites before inspecting their later outcomes.
Consider random assignment among eligible sites when operational and ethical constraints permit.

Write down the assignment unit separately from the measurement unit.
A site may be assigned a policy while outcomes are recorded for many stock-keeping units each day.
Thousands of item-day records do not necessarily create thousands of independent site-level assignments.
Shared management and shared demand shocks can make outcomes within a site correlated.

Record deviations from assignment.
A treated site may not use the policy, or a comparison site may adopt a related practice.
An analysis by assigned group and an analysis by actual usage answer different questions.
Actual usage can be associated with managerial skill or local conditions, so it does not automatically isolate the policy effect.

**Output:** Assignment rule, assignment unit, deviations, and the comparison's main threats.

### Step 13 — Check time and denominators explicitly

**Input:** Source dates, event timestamps, population counts, and measurement definitions.

Separate at least four dates:

1. The event or measurement date.
2. The period over which the metric was calculated.
3. The publication date.
4. The date on which your team accessed that publication.

A case published this year can describe results from several years ago.
A revised dataset published today cannot be assumed to have been available to a decision maker at the historical decision date.
When evaluating a past decision, this difference can determine whether the comparison is fair.

For each rate, record the numerator and denominator separately.
If 90 of 100 orders were on time and 900 of 1,000 orders were on time, both rates are 90%.
The equal rates do not imply equal precision or equal operational exposure.
Aggregating the counts gives 990 of 1,100 on-time orders, still 90% in this example.
When rates differ, calculate the pooled rate from summed counts rather than averaging site rates unless equal site weighting is the stated estimand.

If one source reports average daily inventory and another reports a month-end balance, do not join them as equivalent outcomes.
If one source reports units and another reports USD, ask how prices and mix affect their relationship.
If the mismatch cannot be resolved, keep separate rows and describe the difference.

**Output:** A time-and-denominator reconciliation with unresolved fields visible.

### Step 14 — Describe uncertainty without inventing precision

**Input:** The study's design, sample, variability, reported interval, and business assumptions.

If an interval is reported, preserve its level and method.
An interval for the mean change across observations is not automatically an interval for a causal effect.
A range of possible implementation costs is not a statistical confidence interval.
Both can be useful if clearly labeled.

Ask whether the interval addresses the appropriate unit of assignment.
Site-level interventions may require site-level or cluster-aware analysis.
Ask whether the study accounts for repeated observations over time.
The answer depends on the design and cannot be deduced from a headline count.

Separate three kinds of uncertainty in a decision memo:

- **Measurement uncertainty:** Is the inventory or service metric accurate and consistently defined?
- **Effect uncertainty:** How much of the observed change is attributable to the intervention?
- **Decision uncertainty:** What will happen when a different organization implements the policy?

These categories can overlap, but separating them makes the next evidence request clearer.
An audit of inventory data may address measurement without resolving causal attribution.
A well-designed local experiment may improve attribution while leaving long-run transfer uncertain.
A financial sensitivity analysis explores decision assumptions but cannot fix biased measurement.

**Output:** A short uncertainty statement and the specific question each additional analysis would answer.

### Step 15 — Preserve competing explanations

**Input:** The intervention timeline and relevant operational history.

List plausible alternatives before looking for the most favorable narrative.
Potential alternatives include product discontinuation, a demand contraction, a warehouse closure, a changed supplier agreement, or an accounting reclassification.
Their relevance depends on the actual case; do not assert that any occurred without evidence.

For each alternative, identify an observable implication.
If the product range shrank, compare eligible item counts and inventory by item.
If warehouse activity shifted, inspect network-wide stock and transfers rather than a single site.
If valuation changed, compare physical quantities and unit values under a consistent basis.
If a demand decline is suspected, compare volumes and customer mix across groups.

An alternative explanation need not be eliminated with certainty to make a bounded decision.
The analyst should state how much it could matter and whether the proposed action remains acceptable under it.
If the information is unavailable, say so and adjust the gate to the remaining risk.

**Output:** Alternative-explanation table with evidence, unresolved questions, and decision implications.

### Step 16 — Prepare a decision-ready summary

**Input:** The completed claim register, effect appraisal, and transfer assessment.

Use a compact structure that preserves the distinction between evidence and recommendation:

1. **Decision:** What choice must be made, by whom, and by when?
2. **Observed or reported:** What did the named source actually say or measure?
3. **Grade and scope:** Which course category applies to the exact cited material?
4. **Attribution:** What comparison supports the causal statement, if any?
5. **Transfer:** Which local conditions match, differ, or remain unknown?
6. **Economics:** What are the one-time and recurring costs and benefits?
7. **Recommendation:** What bounded action is proposed?
8. **Gate:** What evidence permits scale, pause, or stop?

Do not bury an important limitation in an appendix if it changes the decision.
If the source provides only a vendor claim, say so in the opening summary.
If the result is synthetic, say so beside the number rather than only at the end.
If the comparison is observational, name the main untested alternative.

The summary is complete when a reader can trace each material number to a source or an explicitly labeled assumption.
It is not complete merely because every paragraph has a citation.
The cited source must support the specific statement next to it.

**Output:** A traceable decision memo with an owner and an explicit next review.

## 6. Worked example

### 6.1 Decision and synthetic case

Everything numerical in this worked example is synthetic.
The organizations, measurements, costs, and outcomes are invented to teach the calculations.
They are not evidence about a real company, vendor, or implementation.

A distributor is considering a replenishment policy at 20 warehouses.
Its decision is whether to run a controlled pilot at four warehouses before any network-wide deployment.
A vendor's fictional customer story reports that another distributor reduced average inventory value by 20% after a rollout.
The story has no comparison group or denominator details.
The local team has also prepared a synthetic retrospective comparison of treated and comparison sites.

The team must not combine the fictional vendor story and the fictional site analysis into one unqualified claim.
They have different provenance and answer different questions.
The vendor story is a D-type hypothetical source.
The site table is invented exercise data, not a real A–E publication.
No letter turns the simulated table into actual empirical evidence.

### 6.2 Claims register

| ID | Statement | Source in scenario | Label | What it can establish |
|---|---|---|---|---|
| C1 | A customer used a replenishment system. | Fictional vendor story. | D, hypothetical. | The vendor claims deployment; the original record is not inspected. |
| C2 | Inventory value fell by 20%. | Same fictional story. | D, hypothetical. | The vendor reports a percentage; base and period are missing. |
| C3 | The system caused the full 20% reduction. | Same fictional story. | D, hypothetical. | An attribution claim, not established by the disclosed design. |
| C4 | Local treated-site average stock changed from USD 12.0 million to USD 10.8 million. | Invented local exercise table. | Synthetic. | Arithmetic only; not an observation from real operations. |
| C5 | Comparable sites changed from USD 11.0 million to USD 10.6 million. | Same invented table. | Synthetic. | Arithmetic only; comparability is an assumption. |
| C6 | Local savings will equal the vendor's 20% figure. | Analyst proposal. | Inference, unsupported. | A hypothesis to test, not a sourced fact. |

The register separates what a source reports from what the team concludes.
Even in a real case, an internal table would need a provenance and quality assessment before being used.
The synthetic exercise does not provide a confidence interval, and none can be derived from these four aggregates alone.

### 6.3 Inventory calculation

The invented data represent average inventory value per group over comparable 13-week periods.
Values are in USD millions.
Assume for this exercise that valuation, product scope, and site membership are held fixed.

| Group | Before, USD million | After, USD million | Change, USD million |
|---|---:|---:|---:|
| Treated sites | 12.0 | 10.8 | -1.2 |
| Comparison sites | 11.0 | 10.6 | -0.4 |

Treated change = 10.8 − 12.0 = −1.2 million USD.
Comparison change = 10.6 − 11.0 = −0.4 million USD.
Difference in differences = −1.2 − (−0.4) = −0.8 million USD.

Under the stated comparison assumptions, the illustrative estimate is a reduction of USD 0.8 million in average inventory value relative to the comparison trend.
The negative sign indicates lower inventory; the effect is not automatically a cost saving.
The treated group's raw reduction is 1.2 / 12.0 = 10% of its baseline.
The comparison group's raw reduction is 0.4 / 11.0 ≈ 3.64% of its baseline.
The difference between these relative reductions is approximately 6.36 percentage points.
That is a different estimand from the USD 0.8 million level difference.
Do not substitute one for the other without specifying the target metric.

### 6.4 What the arithmetic does not prove

The comparison assumes that, absent the policy, treated sites would have changed by the same absolute USD amount as the comparison sites.
The equal-trend assumption is on the dollar-level outcome used in the calculation, not automatically on percentage changes.
The table does not test that assumption.

The calculation is biased if treated sites also reduce assortment while comparison sites do not.
A supplier shortage affecting one group differently can distort the estimated effect.
Shared transfers between sites could make the comparison group partly exposed.
A change in inventory valuation or product price can change USD balances without changing physical stock.
A small number of sites can make estimates sensitive to one large facility.

A real analysis would investigate multiple pre-periods, assignment rules, site-level distributions, spillovers, and changes in service.
Pre-period similarity increases plausibility but does not prove future parallel trends.
The average could conceal a stockout increase at a critical site.

### 6.5 Economic scenario

Assume, only for illustration, that USD 0.8 million of lower average inventory value can be sustained for one year without degrading service.
Assume an annual carrying rate of 18% and an incremental annual operating cost of USD 50,000.
Assume the implementation costs USD 110,000 once at the start.
These are scenario inputs, not measured facts or financial advice.

Illustrative annual gross carrying benefit = USD 800,000 × 0.18 = USD 144,000 per year.
Illustrative annual net operating benefit = USD 144,000 − USD 50,000 = USD 94,000 per year.
Illustrative first-year net after implementation = USD 94,000 − USD 110,000 = −USD 16,000.
The USD 800,000 stock-value reduction is a potential one-time release of working capital, not an additional annual profit.
The first-year net calculation excludes financing effects, tax, changes in revenue, service impacts, and any recurring or one-off items not specified.

The calculation is a scenario rather than a forecast because the assumed reduction and carrying rate have not been validated.
It is not a target or a commitment.
A plan would specify the pilot design and allocated resources.
A target would state an approved performance objective.
A commitment would require the appropriate owner to accept a defined obligation.

### 6.6 Sensitivity to carrying rate

Hold the invented USD 800,000 inventory reduction and USD 50,000 annual operating cost constant.
The table excludes the one-time USD 110,000 implementation cost.

| Annual carrying rate | Gross benefit, USD/year | Net operating benefit, USD/year |
|---:|---:|---:|
| 12% | 96,000 | 46,000 |
| 18% | 144,000 | 94,000 |
| 24% | 192,000 | 142,000 |

These are alternative assumptions, not confidence limits.
The analysis is also sensitive to whether the reduction persists and whether service deteriorates.
For example, if only half the assumed inventory reduction persists at 18%, gross benefit would be USD 72,000 per year before operating cost.
That would leave USD 22,000 per year after the assumed USD 50,000 operating cost.

### 6.7 Decision recommendation

The fictional vendor story supports asking for data, not budgeting a 20% benefit.
The synthetic comparison demonstrates a possible method, not a real effect.
The recommended next step in this exercise is a bounded local pilot with prespecified sites, a comparison or randomized rollout where practical, and service guardrails.
The decision owner should review results before scaling.
No production or financial commitment follows from the invented figures.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Save it as `evidence_example.py` and run `python evidence_example.py`.
All inputs are the synthetic values from Section 6.
The program calculates the level difference in differences and a simple economic scenario.
It does not estimate uncertainty, validate comparability, or establish causality.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class InventoryComparison:
    treated_before_usd: float
    treated_after_usd: float
    comparison_before_usd: float
    comparison_after_usd: float

    def treated_change(self) -> float:
        return self.treated_after_usd - self.treated_before_usd

    def comparison_change(self) -> float:
        return self.comparison_after_usd - self.comparison_before_usd

    def difference_in_differences(self) -> float:
        return self.treated_change() - self.comparison_change()


def scenario(
    inventory_reduction_usd: float,
    annual_carrying_rate: float,
    annual_operating_cost_usd: float,
    implementation_cost_usd: float,
) -> tuple[float, float, float]:
    gross = inventory_reduction_usd * annual_carrying_rate
    net_operating = gross - annual_operating_cost_usd
    first_year_net = net_operating - implementation_cost_usd
    return gross, net_operating, first_year_net


def main() -> None:
    comparison = InventoryComparison(
        treated_before_usd=12_000_000,
        treated_after_usd=10_800_000,
        comparison_before_usd=11_000_000,
        comparison_after_usd=10_600_000,
    )
    effect = comparison.difference_in_differences()
    print(f"Treated change: USD {comparison.treated_change():,.0f}")
    print(f"Comparison change: USD {comparison.comparison_change():,.0f}")
    print(f"Difference in differences: USD {effect:,.0f}")

    # A negative effect is a reduction in this particular example.
    reduction = -effect
    gross, net, first_year = scenario(
        inventory_reduction_usd=reduction,
        annual_carrying_rate=0.18,
        annual_operating_cost_usd=50_000,
        implementation_cost_usd=110_000,
    )
    print(f"Annual gross benefit: USD {gross:,.0f}/year")
    print(f"Annual net operating benefit: USD {net:,.0f}/year")
    print(f"First-year net after implementation: USD {first_year:,.0f}")

    print("Sensitivity, excluding implementation cost:")
    for rate in (0.12, 0.18, 0.24):
        gross, net, _ = scenario(reduction, rate, 50_000, 110_000)
        print(f"  {rate:.0%}: gross USD {gross:,.0f}/year; "
              f"net USD {net:,.0f}/year")


if __name__ == "__main__":
    main()
```

Expected output:

```text
Treated change: USD -1,200,000
Comparison change: USD -400,000
Difference in differences: USD -800,000
Annual gross benefit: USD 144,000/year
Annual net operating benefit: USD 94,000/year
First-year net after implementation: USD -16,000
Sensitivity, excluding implementation cost:
  12%: gross USD 96,000/year; net USD 46,000/year
  18%: gross USD 144,000/year; net USD 94,000/year
  24%: gross USD 192,000/year; net USD 142,000/year
```

The code interprets a negative difference as a reduction because the outcome is inventory value.
For an increase, the sign would produce a negative reduction and the economic interpretation would need reconsideration.
Do not reuse the function as a general benefit calculator without defining the sign and cost model for the new metric.
The inputs are aggregates, so this script cannot estimate site-level variability or standard errors.

## 8. Data, standards, and regulatory context

### 8.1 Source classification is not regulation

The A–E scheme is a course convention, not a legal or professional standard.
No regulator has authorized the letter alone as a basis for investment, model approval, or procurement.
Organizations may have stricter internal evidence policies.
Where policies differ, preserve the course label for coursework and document the organization's separate classification.

### 8.2 Audited financial information

An audit opinion concerns a defined entity, financial reporting framework, reporting period, and statements.
Inspect the opinion, notes, and scope before describing a figure as audited.
An audit does not mean an auditor verified every operational metric or causal explanation in a company presentation.
Financial statement amounts can also be revised or restated.
The relevant date and version are part of the evidence record.

A one-time reduction in inventory on a balance sheet is not automatically the annual economic benefit of a model.
Accounting definitions, valuation, scope, and business changes matter.
Ask finance to review any decision-grade valuation and benefits model.

### 8.3 Securities filings

The U.S. Securities and Exchange Commission (SEC) provides access to public-company filings through EDGAR.
A filing is an official record of what was filed, subject to the applicable reporting framework.
Read the filing itself rather than relying on a snippet.
Distinguish audited statements from management discussion, forward-looking statements, exhibits, and non-GAAP measures where relevant.
The fact of filing does not mean the SEC independently established the truth or causal interpretation of every claim.

The governing rules depend on issuer, jurisdiction, date, and filing type.
This lesson does not give securities-law advice.
Consult the regulator's current materials and qualified counsel for an actual obligation.

### 8.4 Official statistics and data vintages

An official statistical agency can be the primary source for a published series.
Read its methodology, population, release calendar, units, seasonal adjustment, and revision policy.
A national series may not represent a particular local facility or customer segment.
A revised historical series can differ from the information available at an earlier decision date.

For a historical evaluation, record the vintage actually available at the time if the purpose is to reconstruct the decision.
For current planning, an updated series may be more appropriate, but label it as revised.
Neither choice is universally correct; the question determines the vintage.

### 8.5 Reporting guidance and causal appraisal

The CONSORT statement provides reporting guidance for randomized trials.
The STROBE statement provides reporting guidance for observational studies.
These checklists help a reader locate design details and omissions.
They do not certify that a study is unbiased or directly applicable to a supply-chain decision.
A study can report clearly and still have an unsuitable design for the claim under review.

The relevant comparison for causal inference is an outcome under an alternative action for the same target units.
That quantity is not directly observed for treated units.
A randomized trial or quasi-experiment can improve the approximation under explicit assumptions.
Peer review does not eliminate the need to inspect those assumptions.

### 8.6 Data protection and access

Operational case records may include employee, customer, supplier, or commercially sensitive data.
A public case study does not grant permission to use underlying individual records.
Apply the organization's access, retention, and privacy controls.
The specific legal requirements depend on jurisdiction and use; obtain appropriate specialist advice.

Minimize data shared for a case exercise.
Use synthetic or appropriately governed data when individual-level records are unnecessary.
Record access restrictions so a reviewer can distinguish inaccessible evidence from absent evidence.

### 8.7 Measurement conventions

Define dates and time zones when comparing order events.
Define the financial currency and conversion date when comparing values across countries.
Record whether a percentage is weighted by units, orders, sites, or revenue.
Record whether a measurement is a point-in-time balance or average over a period.

Standards and common identifiers may improve interoperability, but they do not establish a causal effect.
A consistent item identifier can help join transaction records; it does not make an intervention and comparison exchangeable.
A standardized metric can still be selectively reported.

## 9. Industry practice and case evidence

### 9.1 How to read a public benchmark

The M5 forecasting competition provides a documented forecasting benchmark and associated data and evaluation materials.
The competition website identifies the competition, data, and evaluation context; see the primary source in Section 16.
The competition concerns a defined dataset, forecast horizon, and scoring rules.
Its results are evidence about performance under that benchmark's conditions.

A peer-reviewed article describing competition methods or findings is A under the course convention for the paper itself.
Inspect the particular article and its method before quoting a numerical result.
A competition organizer's published result is not automatically peer-reviewed merely because related papers exist.
The grade and locator should refer to the exact source actually used.

**Inference:** A method that performs well on this benchmark may be a useful candidate for local testing.
That inference does not establish lower local inventory, improved service, or positive return on investment.
Those outcomes depend on decisions, constraints, data, implementation, and costs that a forecast score alone does not establish.

A benchmark ranking is not a forecast of future performance at every retailer.
The benchmark's data-generating period and evaluation rules are part of the claim.
A later operating environment, product portfolio, or forecasting horizon may differ materially.

### 9.2 Official filings versus operating stories

A company filing can provide a dated record of reported financial figures and management statements.
The SEC's EDGAR system allows readers to retrieve the actual filing and inspect the section in which a statement appears.
The source can establish that an issuer made a particular disclosure, subject to the document's scope.
It does not by itself establish that a named analytics system caused a financial change.

Consider a hypothetical press headline asserting that a warehouse algorithm caused an issuer's reported margin to rise.
Trace the headline to the underlying filing, then locate the line item and the management discussion.
If the filing reports margin but does not attribute the change to that algorithm, the headline's causal claim is unsupported by that filing.
The filing can still be useful for the reported margin, with the applicable accounting definitions and scope.

An audited line item and an unaudited operational narrative can coexist in the same document.
Grade the audited statement A where the audit scope covers it; grade the filing record B and a company assertion C as appropriate to the precise material used.
Record the rationale rather than collapsing these into a single document-wide label.

### 9.3 Marketing and independent replication

A vendor case study may describe a real deployment and include a customer quotation.
Its business purpose is often promotional, and the included customers are not necessarily representative.
Read the selection criteria, metric definition, baseline, and omitted sites if available.
Ask who computed the result and whether a customer or independent reviewer can inspect the calculation.

**Inference:** A vendor claim can motivate a candidate pilot and a set of diligence questions.
It cannot by itself establish that every customer will achieve the stated result.
A second article quoting the same vendor story adds distribution, not a second independent estimate.

An independent replication would need an identifiable separate analysis or dataset and a relevant measurement procedure.
Independence is not guaranteed merely because the publisher has a different name.
A commissioned study may still contain useful data, but funding and the sponsor's role should be visible.

### 9.4 Compare evidence by question

| Question | Useful starting source | What still needs checking |
|---|---|---|
| Did a company report a revenue figure? | The actual company filing. | Statement scope, period, accounting definition, revisions. |
| Did a method score well in a named benchmark? | Official results and relevant published methods. | Dataset, metric, competition rules, reproducibility. |
| Did a system cause a particular stock reduction? | An appropriately designed effect evaluation. | Assignment, counterfactual, data quality, spillovers. |
| Could the method work at our facilities? | Local test plus relevant prior studies. | Similarity, implementation, guardrails, economics. |
| What is the national trend? | Relevant official statistical series. | Coverage, adjustment, vintage, local applicability. |
| What should we investigate? | Vendor or press report may be a lead. | Trace to primary data before relying on the claim. |

No row implies that a source is automatically sufficient for an investment decision.
A well-designed local test can be valuable even if an external claim is weak.
Conversely, a high-grade external study may not address a local decision's binding constraint.

### 9.5 Comparing a benchmark with an operating case

A forecast benchmark commonly evaluates a prediction against held-out observations using a specified loss function.
A distribution operation is judged on service, inventory, cost, labor, and other consequences of actions.
The connection from forecast accuracy to those consequences passes through a decision policy.

A model can improve an average error metric while worsening a critical product segment.
A company may use the forecast differently from the competition's assumed application.
The correct interpretation is therefore conditional: benchmark evidence can support model candidacy, while operational benefits require a separate evaluation.

A useful case comparison table should include the unit of analysis and the loss function on one side, and the decision policy and realized outcomes on the other.
Do not average the two cases' headline percentages.
They have different denominators, outcomes, and causal interpretations.

### 9.6 Worked source-chain comparison

Consider the following hypothetical chain, not a real company case.
A software supplier publishes a customer quotation claiming a 15% reduction in inventory.
A newspaper repeats the quotation.
A consulting slide then cites the newspaper.
The three documents have different publishers, but all three numerical claims trace to the same customer statement.

The supplier's article is D under the course convention.
The newspaper summary is E.
The consulting slide must be classified according to its actual source type and contents; the mere presence of a citation does not create independent evidence.
The customer's original statement would be a C company claim if retrieved as such.
If the underlying inventory data are unavailable, the analyst cannot inspect the calculation.

The proper summary is not “three independent sources confirm a 15% reduction.”
It is “a customer claim of 15%, repeated by a supplier and subsequent summaries, has not been independently checked here.”
That sentence preserves both the information and its limitation.

Now suppose a separate study uses data from the same customer and the same original dashboard.
Its design may add analysis, but the underlying measurement is still shared.
Record the shared dependency and inspect the new study's methods before calling it corroboration.
Alternatively, a second customer study may use a different dataset but have the same vendor sponsor.
Different data reduce one dependency; common sponsorship is still relevant to interpretation.

### 9.7 Causal comparison and the role of a negative case

A published library often contains more successful implementations than failures.
Absence of a public failure report is not evidence that failures did not occur.
Ask what cases could have been selected, who selected them, and whether discontinued pilots are visible.
This question applies to peer-reviewed research as well as vendor marketing.

Suppose a synthetic network has ten eligible sites.
Six finish a pilot, while four stop early because of data or process problems.
If the case reports only the six completers, the result describes a selected subset.
The four exits may be directly relevant to the feasibility and cost of a network rollout.

Do not invent outcomes for the four sites.
Report their count, documented reasons for exit, and missing outcome status.
Where the design supports it, assess outcomes for the original assigned groups as well as actual users.
Explain any missing data and how the conclusion changes under defensible alternative assumptions.

A failure case can teach a binding constraint even when its effect size is unknown.
For example, incomplete supplier lead-time histories may make a policy infeasible.
The documented constraint is evidence about that case; whether it also binds locally requires local checking.

### 9.8 A practical case-comparison record

For each candidate case, complete the same fields before comparing outcomes:

| Field | Why it matters | Example entry type |
|---|---|---|
| Decision owner | Affects the authority and incentives behind the action. | Planning director or procurement manager. |
| Action and alternative | Defines what is being compared. | New policy versus existing policy. |
| Eligibility | Shows who could enter the study or pilot. | Sites meeting a prespecified data requirement. |
| Assignment | Reveals selection and the unit of intervention. | Randomized site group or manager selection. |
| Outcome | Prevents unlike metrics being treated as equivalent. | Average USD inventory or on-time order rate. |
| Horizon | Shows whether short-run and long-run results are mixed. | Defined weeks before and after launch. |
| Comparator | Makes the proposed counterfactual visible. | Similar untreated sites over the same period. |
| Source chain | Identifies who produced the underlying information. | Dashboard, analyst, publication, summary. |
| Grade and scope | Records course provenance without overclaiming. | A for a study's own peer-reviewed results. |
| Costs and guardrails | Captures trade-offs omitted by headline improvements. | Operating cost and critical-item service. |
| Transfer constraints | Identifies mismatches with the target organization. | Different lead times or inaccessible data. |
| Unresolved questions | Prevents gaps being mistaken for facts. | Unknown exclusions or missing variance. |

Compare the records row by row.
If case A has a stronger design but a different decision, do not automatically call it the better local answer.
If case B is operationally similar but promotional, use its details to design local diligence, not as proof.
If both differ in outcome definition, explain the mismatch before discussing magnitude.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Grade inflation | A press summary is called A because it cites a paper. | Source chain is skipped. | Readers assume methods were checked. | Retrieve the paper and cite the relevant result. |
| Audit halo | Every slide claim is called audited. | Audit scope is generalized. | Operational attribution is overstated. | Inspect the opinion and label covered figures only. |
| Filing halo | A filed claim is described as regulator verified. | Official receipt is confused with verification. | Unwarranted certainty. | Attribute the statement and identify its section. |
| Vendor selection | Only one successful customer is shown. | Selection on positive outcomes. | Expected benefit is overstated. | Request eligible cohort and unsuccessful cases. |
| Duplicate sourcing | Many articles use identical language. | A single release is repeated. | False impression of corroboration. | Draw the source dependency chain. |
| Missing denominator | A percentage lacks its starting amount. | The base cannot be checked. | Invalid economic calculation. | Obtain the base and definition or mark unknown. |
| Metric drift | Before and after rates use different rules. | Measurement changes with intervention. | Apparent improvement is artificial. | Recompute both periods under a common definition. |
| Wrong comparator | Pilot sites were selected for expected improvement. | Selection creates different trajectories. | Biased effect estimate. | Prespecify assignment and assess comparability. |
| Seasonality | Pilot starts just before a normal seasonal decline. | Time effect is attributed to treatment. | Spurious benefit. | Use relevant control periods or comparison units. |
| Concurrent change | A site also changes its product range. | Multiple interventions coincide. | Attribution is ambiguous. | Record changes and design a suitable comparison. |
| Spillover | Control sites share stock with pilot sites. | Treatment changes comparison outcomes. | Difference is biased or uninterpretable. | Measure exposure or choose independent clusters. |
| Survivorship | Failed pilot sites disappear from the report. | Only continuing sites are counted. | Benefits are exaggerated. | Track the original cohort and reasons for exit. |
| Weak transfer | A large retailer's result is applied to a small wholesaler. | Data, constraints, and economics differ. | Failed implementation. | Apply the five transfer questions and local test. |
| Percent confusion | A rate drops from 4% to 3% and is called a 1% reduction. | Percentage points and relative percent are mixed. | Misstated magnitude. | Report both: 1 percentage point and 25% relative. |
| Stock versus flow | Released capital is counted as annual profit. | One-time balance change is treated as recurring. | Inflated business case. | Separate cash release and annual carrying benefit. |
| Unsupported precision | An estimate is reported to two decimals without sample details. | Formatting is mistaken for accuracy. | False confidence. | Report appropriate precision and uncertainty. |
| Cherry-picked horizon | The best six weeks are selected after review. | Outcome-dependent period selection. | Optimistic estimate. | Prespecify horizon and include all eligible weeks. |
| Outcome omission | Inventory falls but service is not reported. | A competing outcome is excluded. | Local stockouts or customer harm. | Define guardrails before testing. |
| Stale source | A case predates a network redesign. | Context changes over time. | Historical effect does not transfer. | Record dates and refresh material assumptions. |
| Unresolved contradiction | Two sources use different totals without explanation. | Definitions or revisions differ. | Incorrect synthesis. | Compare scope and vintage; preserve unresolved conflict. |
| Benchmark overreach | Forecast rank is presented as profit. | Prediction score is conflated with decision value. | Investment justified by the wrong outcome. | Evaluate the operational policy and costs. |
| Sensitive data leak | A case note contains named employee-level data. | Data minimization is omitted. | Privacy and commercial harm. | Restrict access and use governed aggregates. |

A red flag is a question to investigate, not proof of misconduct.
A source may be useful despite a limitation if the limitation is visible and proportionate to the decision.
An omitted method is nevertheless a real limit: the analyst cannot assume it was sound.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or manage it |
|---|---|---|---|
| Source identity is correct. | The grade and attribution depend on provenance. | A repost omits the original author. | Trace author, publisher, original document, and locator. |
| The cited passage supports the normalized claim. | Summaries often add detail. | Headline is stronger than the source. | Compare wording with the original passage. |
| Units and scope are consistent. | Arithmetic requires comparable measures. | One period includes consignment stock and the other does not. | Recompute both under the same inclusion rules. |
| Outcome measurement is stable. | A change should represent operational change. | A new scanner changes reporting coverage. | Compare definitions and audit a sample of records. |
| Comparison trends would otherwise be parallel. | Difference in differences uses comparison change as counterfactual. | Different demand shocks hit the groups. | Inspect pre-period trends and domain changes; acknowledge residual uncertainty. |
| No material spillover. | The control should represent the untreated condition. | Stock moves from treated to control sites. | Map shared resources and measure exposure. |
| Cohort membership is stable or accounted for. | Changes in composition can mimic effects. | Closed sites vanish from follow-up. | Track all originally eligible sites and report exclusions. |
| Assortment and valuation are comparable. | USD inventory can change without physical improvement. | Prices rise or product mix changes. | Decompose quantity, mix, price, and accounting changes. |
| Local benefits use an applicable base. | A percentage needs a relevant denominator. | External case uses different stock categories. | Map the eligible population and currency basis. |
| Carrying rate is appropriate. | Inventory value does not equal annual cost. | Rate includes items already counted elsewhere. | Finance reviews rate, components, and double counting. |
| Effect persists over the stated horizon. | Annualization assumes continued operation. | Initial gains erode or seasonality reverses. | Measure over a prespecified longer period and scenario-test persistence. |
| Service and safety remain acceptable. | Inventory reductions can shift risk. | Critical stockouts increase. | Prespecify service and safety guardrails. |
| Claimed corroboration is independent. | Repetition of one dataset is not replication. | Multiple sources copy a release. | Map underlying data, analysts, and funding. |
| Evidence remains current. | Context and sources can change. | A report is corrected or operations change. | Version the source and schedule review. |
| Transfer is plausible. | An external result is used for a local decision. | Different decisions, constraints, or data. | Apply the five-question test and pilot locally. |

The worked example does not supply evidence that any of these assumptions hold in a real operation.
It shows what an analyst would need to inspect before interpreting a real result.
A grade does not repair a violated design assumption.

## 12. Exercises

### Exercise 1 — Classify sources

**Time:** 15 minutes.

A. A peer-reviewed paper reports its own randomized study.
B. A national statistical office publishes its own official series.
C. A company posts an investor presentation about its operations.
D. A software supplier publishes a customer success story.
E. A newspaper summarizes the supplier's story without additional data.

**Deliverable:** A five-row table with the course grade, a permissible use, and one limitation per source.

**Acceptance criteria:** Use A through E once in the stated order, distinguish source type from study validity, and do not claim a randomized paper proves transfer.

### Exercise 2 — Trace a source chain

**Time:** 20 minutes.

A trade article quotes a vendor's website.
The website quotes a customer executive.
The executive refers to an internal dashboard that you cannot access.
The article gives the same 15% figure as the website.

**Deliverable:** Draw the chain and write a two-sentence evidence note.

**Acceptance criteria:** Identify the accessible article as E, the vendor item as D, and the underlying dashboard as uninspected; do not count the two publications as independent measurements.
State what access or documentation would improve the assessment.

### Exercise 3 — Compute a before-and-after result

**Time:** 15 minutes.

A synthetic on-time rate rises from 92% to 95% over two comparable 12-week periods.
The denominator is 10,000 eligible orders in each period.

**Deliverable:** Compute the percentage-point change, relative percentage increase, and additional on-time orders, assuming the rates and counts are exact for the exercise.

**Acceptance criteria:** Show the arithmetic, give the correct units, and say why the change is not automatically causal.

### Exercise 4 — Difference in differences

**Time:** 20 minutes.

In a synthetic example, treated sites have 8.0 stockout days per 100 site-days before a policy and 5.5 afterward.
Comparison sites have 7.0 before and 6.0 afterward, on the same scale and over comparable periods.

**Deliverable:** Calculate each change and their difference, state the sign, and list two threats to causal interpretation.

**Acceptance criteria:** Use the outcome unit, explain the parallel-trends assumption, and identify a plausible spillover or concurrent change.

### Exercise 5 — Compare two cases

**Time:** 25 minutes.

Case X is a peer-reviewed forecast benchmark on daily retail demand with a published error metric.
Case Y is a vendor story claiming a reduction in average warehouse stock value but omitting its comparator.
You are deciding whether a local food distributor should replace its replenishment system.

**Deliverable:** A comparison table with decision, outcome, source grade, causal interpretation, and at least three transfer questions.

**Acceptance criteria:** Treat X as A only for its own peer-reviewed content, Y as D, and neither as direct proof of local profit.
Identify the difference between prediction performance and operational value.

### Exercise 6 — Design a decision gate

**Time:** 30 minutes.

The synthetic distributor has eight sites and wants to test a new replenishment policy.
Some sites share inventory transfers.
The firm cannot accept an increase in critical-item stockouts.

**Deliverable:** A one-page pilot design and scale gate.

**Acceptance criteria:** Identify assignment and comparison, prespecify outcome and period, address shared-stock spillovers, define a service guardrail and an inconclusive-result action, and name an accountable owner.
Do not claim that a specified design guarantees an unbiased estimate.

### Exercise 7 — Build a claim register

**Time:** 25 minutes.

A filing contains an audited inventory figure and separate management commentary saying a technology investment contributed to efficiency.
A press article reports that the technology caused the full inventory change.

**Deliverable:** Three claim rows with source, grade, scope, and missing evidence.

**Acceptance criteria:** Separate the audited figure, the management attribution, and the press claim.
Do not grade the causal attribution A merely because it appears beside an audited figure.
Specify where a reviewer should look for the audit scope.

### Exercise 8 — Communicate uncertainty

**Time:** 20 minutes.

An executive asks for a one-sentence commitment to save USD 1 million per year based on the synthetic vendor story in Section 6.

**Deliverable:** A three-sentence response plus one next action.

**Acceptance criteria:** Attribute the vendor claim, separate it from the local synthetic calculation, refuse to present either as an earned saving, and propose a measurable decision gate.
Use appropriate currency and time units.

## 13. Self-check answers

### Answer 1 — Classify sources

| Item | Grade | Permissible use | Limit |
|---|---|---|---|
| A | A | Examine the paper's own study and results. | Peer review and randomization do not ensure local transfer or eliminate attrition bias. |
| B | B | Use the official series within its stated definitions. | Coverage, vintage, and local relevance require checking. |
| C | C | Report what the company says. | Self-report does not independently establish attribution. |
| D | D | Form a diligence question or pilot hypothesis. | Marketing can select favorable cases. |
| E | E | Use as a lead to the original material. | The summary may omit methods and may repeat another source. |

A high-grade source can still be poorly matched to the decision.
A lower-grade source can still be valuable as a lead.

### Answer 2 — Trace a source chain

The chain is internal dashboard → customer executive → vendor website → trade article.
The arrow describes transmission of the number, not independent replication.
The accessible trade article is E; the vendor website is D.
The internal dashboard has not been inspected, so its methods, provenance, and appropriate treatment remain unknown.
Ask for the dashboard definition, source data, period, denominator, and calculation owner.
An independent analysis of those data or a separate evaluation could improve the assessment, subject to its own design review.

### Answer 3 — Compute a before-and-after result

The absolute rate change is 95% − 92% = 3 percentage points.
The relative increase is 3 / 92 × 100 ≈ 3.26% relative to the baseline rate.
At 10,000 orders per period, 95% corresponds to 9,500 on-time orders and 92% to 9,200.
The difference is 300 on-time orders per 10,000 orders in the period.

The increase describes an observed before/after difference in the synthetic exercise.
It does not isolate the effect of any policy because season, customer mix, and other changes could also affect the rate.
The example does not specify uncertainty or a comparison group.

### Answer 4 — Difference in differences

Treated change = 5.5 − 8.0 = −2.5 stockout days per 100 site-days.
Comparison change = 6.0 − 7.0 = −1.0 stockout day per 100 site-days.
Difference in differences = −2.5 − (−1.0) = −1.5 stockout days per 100 site-days.
The negative sign means the treated group improved by 1.5 additional stockout days per 100 site-days relative to the comparison group's change.

The causal reading assumes that, without treatment, treated and comparison outcomes would have changed in parallel on this rate scale.
A concurrent supplier change affecting treated sites could violate this assumption.
Transfers between groups could expose comparison sites to treatment and distort the comparison.
Pre-period examination and operational review help assess plausibility but do not prove the counterfactual.

### Answer 5 — Compare two cases

A strong table says that X assesses a forecasting method against a benchmark metric, not a replenishment-system investment.
X is A for its own peer-reviewed content, subject to the actual paper and its scope.
Y is D for the vendor story and describes a claimed operational change without a disclosed counterfactual.
Neither result proves local profitability.

A good answer asks whether the decision, product economics, data environment, operating constraint, and evidence strength match.
It notes food perishability and service constraints if relevant to the specified local case, without inventing values.
It proposes a local forecast and decision-policy evaluation rather than ranking cases solely by headline percentage.

### Answer 6 — Decision-gate rubric

A complete design defines eligible sites and the assignment mechanism before observing outcomes.
If randomization is feasible, it considers assigning clusters that reflect shared inventory transfers rather than pretending connected sites are independent.
If randomization is not feasible, it explains selection and the limitations of the comparison.

The design defines the stock measure, population, pre-period, test period, and missing-data rules.
It names a critical-item service guardrail and a responsible operational owner.
It specifies the decision rule for success, harm, and inconclusive evidence.
It distinguishes permission to pilot from permission to scale.

A strong answer anticipates changes in suppliers, assortment, and season.
It addresses uncertainty and power or feasibility without asserting a sample-size guarantee from eight sites alone.
The rubric allows multiple defensible designs if assumptions and trade-offs are explicit.

### Answer 7 — Claim-register rubric

The audited inventory figure can be recorded as A for the audited line item if the opinion actually covers it.
The filing is also a B record of what was filed.
The management explanation is a C company attribution unless separately supported by a suitable evaluation.
The press article is an E report of its own causal claim.

The reviewer should inspect the actual audit opinion, financial statements, notes, and the relevant management discussion.
The full causal attribution would require evidence that isolates the technology from other changes.
A company attribution, an audited balance, and a press interpretation are not three independent causal estimates.

### Answer 8 — Communication rubric

A suitable response attributes the percentage to the fictional vendor story and notes that its comparator is not supplied.
It says that the local numbers are synthetic and cannot establish realized savings.
It does not turn the illustrative USD 94,000 per year net operating scenario into a forecast or commitment.
It recommends a prespecified pilot and a scale decision after measurement of inventory, service, and costs.

The decision gate should identify an owner, time horizon, and acceptable guardrails.
If no reliable estimate results, the gate should allow an inconclusive outcome rather than force a positive conclusion.

## 14. Interview questions

| Question | Strong answer contains | Weak answer reveals |
|---|---|---|
| What do A–E mean in this course? | Correct source categories and the distinction between grade and validity. | Treats letters as a universal causal ranking. |
| Is a peer-reviewed result automatically causal? | Design, identification assumptions, measurement, and limitations. | Assumes journal publication proves causality. |
| Is every claim in a filing audited? | The opinion's defined scope and separation of management commentary. | Treats an entire document as equally audited. |
| How would you trace a vendor claim? | Original data, calculation owner, denominator, dates, and source dependencies. | Counts articles repeating the claim as replications. |
| What does a comparison group contribute? | Approximation to the counterfactual under stated assumptions. | Says any untreated group automatically removes bias. |
| What is parallel trends? | Similar untreated changes on the chosen outcome scale and horizon. | Mistakes equal baseline levels for the assumption. |
| How do you handle a benchmark winner? | Reproduce relevant performance and evaluate the downstream local decision. | Converts forecast rank into guaranteed savings. |
| How do you compare two percentage improvements? | Aligns outcome, denominator, population, period, and design. | Picks the larger percentage without context. |
| How do you value lower inventory? | Separates one-time capital release, carrying cost, service, and implementation cost. | Books the entire balance reduction as annual profit. |
| What should happen when sources disagree? | Checks definitions and vintage, preserves unresolved conflict, and considers robust decisions. | Silently averages incompatible figures. |
| When can a D or E source be useful? | Hypothesis generation, leads, or context with explicit attribution and limits. | Either treats it as proof or rejects every useful lead. |
| How do you decide whether a case transfers? | Applies the five questions and considers whether one mismatch is decisive. | Assumes the same sector means the same operating problem. |

An interview answer should name the decision before listing methods.
A strong candidate can explain what evidence would change their mind and what they would do if the result remained inconclusive.

## 15. Key terms

- [Evidence grade](#4-definitions-and-vocabulary) — course source-category label A–E.
- [Claim](#4-definitions-and-vocabulary) — a statement with an identifiable evidential burden.
- [Observation](#4-definitions-and-vocabulary) — a recorded measurement under a method.
- [Inference](#4-definitions-and-vocabulary) — a conclusion derived from evidence and assumptions.
- [Synthetic case](#4-definitions-and-vocabulary) — invented instructional scenario.
- [Primary source](#4-definitions-and-vocabulary) — origin of the specific claim or measurement.
- [Secondary source](#4-definitions-and-vocabulary) — account based on another source.
- [Provenance](#4-definitions-and-vocabulary) — traceable origin and transformations.
- [Peer review](#4-definitions-and-vocabulary) — expert review of a manuscript.
- [Audit](#4-definitions-and-vocabulary) — examination within a defined reporting scope.
- [Official statistic](#4-definitions-and-vocabulary) — a published official statistical series.
- [Regulatory filing](#4-definitions-and-vocabulary) — a document filed with a regulator.
- [Company claim](#4-definitions-and-vocabulary) — the organization's account of its own case.
- [Vendor marketing](#4-definitions-and-vocabulary) — promotional material from a supplier.
- [Anecdote](#4-definitions-and-vocabulary) — individual account without systematic sampling.
- [Denominator](#4-definitions-and-vocabulary) — the reference base of a rate or percentage.
- [Grain](#4-definitions-and-vocabulary) — unit and time level of a record.
- [Baseline](#4-definitions-and-vocabulary) — reference value or period.
- [Comparator](#4-definitions-and-vocabulary) — group or condition used for contrast.
- [Counterfactual](#4-definitions-and-vocabulary) — outcome under the alternative action.
- [Confounder](#4-definitions-and-vocabulary) — factor that can distort causal attribution.
- [Selection bias](#4-definitions-and-vocabulary) — distortion from selection into analysis.
- [Survivorship bias](#4-definitions-and-vocabulary) — distortion from excluding failures or exits.
- [Difference in differences](#4-definitions-and-vocabulary) — treated change minus comparison change.
- [Parallel trends](#4-definitions-and-vocabulary) — assumed untreated trend comparability.
- [Spillover](#4-definitions-and-vocabulary) — treatment exposure outside assigned units.
- [Percentage point](#4-definitions-and-vocabulary) — difference between two percentage rates.
- [Uncertainty interval](#4-definitions-and-vocabulary) — range from a stated inferential method.
- [Transportability](#4-definitions-and-vocabulary) — applicability to a target setting.
- [Corroboration](#4-definitions-and-vocabulary) — additional support whose independence must be checked.
- [Decision gate](#4-definitions-and-vocabulary) — explicit condition to proceed, pause, or stop.
- [Forecast](#4-definitions-and-vocabulary) — estimated future outcome, not a commitment.
- [Plan](#4-definitions-and-vocabulary) — intended actions and resources.
- [Target](#4-definitions-and-vocabulary) — desired performance level.
- [Commitment](#4-definitions-and-vocabulary) — authorized obligation under stated conditions.

## 16. Sources

The sources below support methods and source-checking guidance, not the invented numbers in Sections 6, 7, or 12.
Grades describe the course category for the cited item and do not imply that every statement in it proves causality.

1. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. [M5 forecasting competition](https://mofc.unic.ac.cy/m5-competition/), University of Nicosia competition site. Organizer's primary benchmark materials; inspect each linked item and assign a grade appropriate to that specific publication. Supports the benchmark context, not commercial return.
2. U.S. Securities and Exchange Commission. [Search Filings](https://www.sec.gov/search-filings). Grade B: official access point for filings; the scope and status of claims depend on the retrieved document.
3. U.S. Securities and Exchange Commission. [Beginners' Guide to Financial Statements](https://www.sec.gov/about/reports-publications/beginners-guide-financial-statements). Grade B: official educational explanation of financial statements; not a causal evaluation of operational technology.
4. CONSORT Group. [CONSORT Statement](https://www.consort-statement.org/). Reporting guidance for randomized trials; use to inspect reporting, not to certify a study's causal validity.
5. STROBE Initiative. [STROBE Statement](https://www.strobe-statement.org/). Reporting guidance for observational studies; reporting completeness is not itself proof of unbiased estimates.
6. Hernán, M. A., and Robins, J. M. [*Causal Inference: What If*](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/). Chapman & Hall/CRC, 2020. Open textbook on counterfactual causal inference and study design; it is a methodological source rather than a claim about the synthetic case.
7. Bertrand, M., Duflo, E., and Mullainathan, S. [“How Much Should We Trust Differences-in-Differences Estimates?”](https://doi.org/10.1162/003355304772839588) *Quarterly Journal of Economics*, 2004. Grade A: peer-reviewed methodological discussion of inference issues; it does not validate the worked example's comparison.
8. U.S. Bureau of Labor Statistics. [Consumer Price Index overview](https://www.bls.gov/cpi/overview.htm). Grade B: official example of a statistical series and its scope; not evidence for any local distributor's inventory or prices.

Publication, access, and version dates should be captured in an actual claim register.
Recheck mutable sources before using them for a live decision.

## 17. Links

- [Module 28 index](_index.md)
- [Case anatomy](01_case-anatomy.md)
- [The transfer test](03_transfer-test.md)
- [Case selection bias](04_selection-bias.md)
- [Module 28 survey](../../../course/28_domain_case_library.md)
- [Experimentation, pilots, and evidence survey](../../../course/05_experimentation_pilots_evidence.md)
- [Quality, evaluation, and model risk survey](../../../course/26_quality_evaluation_model_risk.md)
- [Simulation, optimization, and production labs survey](../../../course/30_simulation_optimization_production_labs.md)
