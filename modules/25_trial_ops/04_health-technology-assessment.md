# Health Technology Assessment: Evidence Requirements, Economic Models, and Submission Dossiers

> Module 25 — Trial Operations, HTA, and Global Regulation
> Topic 04 of 07
> Estimated study time: 4 hours
> Prerequisites: topics 01 and 03
> Version 2026.3

## 1. Why this topic exists

A regulator asks whether a product is safe and effective. A health technology assessment (HTA) body
asks whether it is worth paying for compared with what the system already does. The two questions
use the same clinical evidence and reach different conclusions often enough that a product can be
approved and then refused funding in the same market.

The difference is not political. It is analytical. Approval requires a favourable benefit-risk
profile in a studied population. Reimbursement requires a credible estimate of comparative effect
against the right comparator, in the right population, over a time horizon long enough to capture
the costs and outcomes that matter, with uncertainty quantified, and with an affordable budget
impact. Each of those requirements can fail independently.

HTA is therefore the place where data science meets health economics and public policy. The
analysis is a model, the model is an argument, and the argument is judged by a committee that will
ask why a parameter was chosen, why a subgroup was excluded, and what happens if the assumptions are
wrong. A model that cannot answer those questions is not evidence; it is a spreadsheet with a
conclusion.

### 1.1 The pattern that repeats

1. A trial shows a statistically significant improvement in a surrogate endpoint.
2. The sponsor submits a dossier with a cost-effectiveness model built on that endpoint.
3. The HTA body says the surrogate is not validated for this population and asks for the final
   outcome.
4. The sponsor re-runs the model with an indirect comparison against a comparator the trial did not
   include.
5. The estimate becomes uncertain, the confidence interval crosses the threshold, and the committee
   recommends restricted use.
6. The product launches with a narrower label and a price lower than the business case assumed.

The pattern is not a failure of the clinical trial. It is a failure to design the evidence package
for the question the payer asks, which is known before the trial starts.

### 1.2 Who is blocked

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Sponsor | Revenue and launch price | Depends on a committee it does not control |
| Payer | A defensible spending decision | Receives a dossier built for a different question |
| Patient | Access | Depends on a decision made without them |
| Clinician | Ability to prescribe | Bound by the funding decision |
| HTA analyst | Time to assess | Receives late or incomplete evidence |
| Health system | Budget stability | Absorbs the cost of every positive recommendation |

### 1.3 The industry question

> Compared with what, in whom, over what horizon, at what cost per unit of health gained, and can
> the payer afford it?

The question contains five clauses because an incremental cost-effectiveness ratio (ICER) without a
comparator, a population, a horizon, and a budget is not a decision input.

## 2. Learning outcomes

By the end of this topic you can:

1. **Frame** an HTA decision problem as a population, intervention, comparator, outcome, and
   perspective, and explain how each choice changes the result.
2. **Critique** the clinical evidence base for an HTA submission, including indirect comparisons and
   surrogate endpoints.
3. **Choose** a model structure (decision tree, Markov, partitioned survival) for a given disease
   and explain its assumptions.
4. **Compute** discounted costs, quality-adjusted life years (QALYs), an ICER, and a net monetary
   benefit from a set of inputs.
5. **Quantify** uncertainty with deterministic and probabilistic sensitivity analysis, and interpret
   a cost-effectiveness acceptability curve.
6. **Build** a budget-impact model and distinguish affordability from cost-effectiveness.
7. **Map** the evidence requirements of NICE, ICER, IQWiG, HAS, and the EU joint clinical assessment,
   and identify where they diverge.
8. **Write** a dossier outline that answers the assessment questions rather than describing the
   product.

## 3. The core mental model

### 3.1 Two questions, one evidence base

```text
                    Clinical evidence
                           |
          +----------------+----------------+
          v                                 v
  Regulator: safe and effective     HTA body: worth paying for
  - benefit-risk                    - comparative effect
  - quality, consistency            - cost per QALY
  - label                           - budget impact
          |                                 |
          v                                 v
     Marketing authorisation          Funding recommendation
          |                                 |
          +----------------+----------------+
                           v
                    Patient access
```

Approval is necessary but not sufficient. The evidence base can be identical; the question, the
comparator, and the threshold are different.

### 3.2 The HTA triangle

```text
                 Comparative clinical benefit
                        /\
                       /  \
                      /    \
                     /      \
        Cost-effectiveness --- Affordability
              (ICER)            (budget impact)
```

A product must satisfy all three. A strong clinical benefit with a weak economic case fails on cost.
A strong economic case with an unaffordable budget impact fails on affordability. A cheap product
with no demonstrated advantage over the comparator fails on comparative benefit, because the
comparator is often already cheap.

### 3.3 The reference case

Most HTA bodies define a reference case: the analytical conventions they expect. The conventions
usually cover:

| Element | Common reference case | Why it matters |
|---|---|---|
| Perspective | Health system or payer | Determines which costs count |
| Comparator | Standard of care in the jurisdiction | Determines the incremental effect |
| Time horizon | Lifetime or long enough to capture differences | Short horizons favour cheap interventions |
| Discount rate | 3.5% in England; 3% in some other systems | Affects long-term costs and benefits |
| Outcomes | QALYs, sometimes evLYs | Determines what counts as benefit |
| Costs | Public list prices or net prices | Determines the numerator |
| Uncertainty | Probabilistic sensitivity analysis | Determines the confidence in the decision |

A model that departs from the reference case is not wrong, but it must justify the departure and
show the reference-case result alongside.

### 3.4 The ICER plane

```text
 More costly
     ^
     |   Quadrant II          Quadrant I
     |   (dominated:          (trade-off:
     |    reject)              ICER decides)
     |
-----+------------------------------------> More effective
     |
     |   Quadrant IV          Quadrant III
     |   (dominant:           (less effective,
     |    adopt)              cheaper: trade-off)
     |
```

Most new technologies sit in quadrant I: more effective and more costly. The ICER is the slope of
the line from the origin to the point. The decision rule compares that slope with a threshold that
represents the opportunity cost of the next best use of the money.

The threshold is not a natural constant. In England, NICE's cost-effectiveness range changed on
31 March 2026 from £20,000 to £30,000 per QALY to £25,000 to £35,000 per QALY. In the United States,
ICER's health-benefit price benchmark uses $100,000 to $150,000 per QALY or equal value of life year
(evLY), and it is a benchmark for discussion, not a coverage rule. In Germany, IQWiG does not use a
threshold; it assesses added benefit and leaves price to negotiation.

### 3.5 The model is an argument

Every economic model contains a chain of claims:

```text
Trial result -> relative effect -> applied to baseline risk -> events over time
             -> resource use -> costs -> discounted -> compared with comparator
```

Each arrow is an assumption. A well-built model makes each assumption visible and testable. A
model that hides an assumption inside a parameter will be found by the committee, and the finding
will dominate the appraisal.

### 3.6 Uncertainty has three kinds

| Type | Question | Method |
|---|---|---|
| Parameter uncertainty | What if this input is wrong? | One-way and probabilistic sensitivity analysis |
| Structural uncertainty | What if the model structure is wrong? | Scenario analysis with alternative structures |
| Decision uncertainty | What is the probability that the decision is wrong? | Cost-effectiveness acceptability curve, expected value of information |

A dossier that reports only parameter uncertainty has addressed the easiest third of the problem.

### 3.7 Affordability is not cost-effectiveness

The ICER compares a new technology with a comparator per unit of health. The budget impact
multiplies the per-patient net cost by the number of patients treated. A product can be highly
cost-effective and still unaffordable if the eligible population is large or the upfront cost is
high. A product can be affordable and still not cost-effective if the health gain is small.

```text
Budget impact = eligible population x uptake x (net cost new - net cost old)
```

The two analyses answer different questions and are assessed by different parts of a payer's
organisation. A submission needs both.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Health technology assessment (HTA) | Systematic evaluation of the clinical, economic, and social effects of a technology | Used to mean a price decision | Process |
| NICE | National Institute for Health and Care Excellence, England and Wales | Treated as a regulator | HTA body |
| ICER (US) | Institute for Clinical and Economic Review, an independent US assessment body | Confused with the incremental cost-effectiveness ratio | Organisation |
| IQWiG | Institute for Quality and Efficiency in Health Care, Germany | Confused with the G-BA, which decides | Assessment body |
| G-BA | Federal Joint Committee, Germany | Confused with IQWiG | Decision body |
| HAS | Haute Autorité de Santé, France | Treated as the price negotiator | Assessment body |
| Transparency Committee | HAS committee that rates SMR and ASMR | Confused with the pricing committee | Committee |
| SMR | Service médical rendu, the clinical benefit that determines reimbursement | Confused with ASMR | Rating I to V |
| ASMR | Amélioration du service médical rendu, the added benefit | Used as a price | Rating I to V |
| CADTH / Canada's Drug Agency | Canadian HTA body | Name changed; older documents use CADTH | HTA body |
| PBAC | Pharmaceutical Benefits Advisory Committee, Australia | Confused with the pricing authority | Committee |
| EU HTA Regulation | Regulation (EU) 2021/2282 on joint clinical assessment | Assumed to replace national assessments | Regulation |
| Joint clinical assessment (JCA) | EU-level assessment of clinical evidence | Confused with a reimbursement decision | Process |
| PICO | Population, intervention, comparator, outcome | Used loosely without the comparator | Framework |
| Reference case | Analytical conventions required by an HTA body | Treated as optional | Methods |
| Comparator | The technology the new one is compared with | Chosen to flatter the new product | Intervention |
| Perspective | Whose costs and outcomes count | Not stated | Choice |
| Time horizon | Period over which costs and effects are counted | Truncated to the trial duration | Years |
| Discount rate | Rate applied to future costs and benefits | Applied inconsistently | Percent per year |
| QALY | Quality-adjusted life year | Treated as a physical unit | Utility x years |
| evLY | Equal value of life year | Confused with a QALY | Outcome measure |
| Utility | Preference weight for a health state, usually 0 to 1 | Assumed to be objective | Index |
| ICER | Incremental cost per QALY gained | Quoted without its comparator | Currency per QALY |
| Net monetary benefit | Threshold x effect minus cost | Confused with profit | Currency |
| Threshold | Willingness to pay per QALY | Treated as a universal constant | Currency per QALY |
| Budget impact | Net cost to the payer of adopting the technology | Confused with cost-effectiveness | Currency per year |
| Probabilistic sensitivity analysis (PSA) | Uncertainty analysis drawing inputs from distributions | Treated as a scenario analysis | Method |
| Cost-effectiveness acceptability curve (CEAC) | Probability that a technology is cost-effective at each threshold | Read as a confidence interval | Curve |
| Deterministic sensitivity analysis | One-at-a-time or multiway change in inputs | Confused with PSA | Method |
| Expected value of information (EVPI) | Value of collecting more evidence | Confused with the cost of a trial | Currency |
| Managed access | Coverage conditional on further evidence | Confused with a discount | Scheme |
| Coverage with evidence development (CED) | Coverage tied to data collection | Confused with a registry | Policy |
| Severity modifier | Adjustment to the threshold for severe disease | Applied without eligibility rules | Policy |
| Highly specialised technology (HST) | NICE programme for very rare diseases | Confused with a standard appraisal | Programme |
| Efficiency frontier | IQWiG method comparing efficiency across alternatives | Confused with an ICER | Method |

### 4.1 Incremental cost-effectiveness ratio and net monetary benefit

The ICER is the ratio of incremental cost to incremental effect. The net monetary benefit converts
the same information into a single number at a given threshold:

```text
NMB = threshold x incremental effect - incremental cost
```

A positive NMB means the technology is cost-effective at that threshold. The NMB is easier to use in
probabilistic analysis because it avoids dividing by a random quantity that can be near zero. The
ICER remains the communication device that committees and the public understand.

### 4.2 Cost-effectiveness and affordability

Cost-effectiveness compares health gained per unit of money. Affordability compares total spend with
the budget available. A technology can be cost-effective and unaffordable, or affordable and not
cost-effective. The two are reported separately because they are decided by different people with
different constraints.

### 4.3 Regulatory approval and HTA recommendation

A marketing authorisation establishes safety, efficacy, and quality for a defined indication. An HTA
recommendation establishes whether the health system should pay for it, for which patients, and under
what conditions. The second can be narrower than the first, and often is.

## 5. The mechanics, step by step

This procedure builds an HTA submission from a clinical evidence package. It assumes a completed or
ongoing pivotal trial and a target set of HTA bodies. The steps are sequential in logic but iterative
in practice: a finding at step 6 often sends the analyst back to step 3.

### Step 1. Frame the decision problem

**Input:** the target market, the licensed indication, the available comparators, and the HTA body's
scope document.

**Output:** a decision problem statement with population, intervention, comparators, outcomes,
perspective, time horizon, and subgroup definitions.

**Rationale:** The decision problem determines every later choice. A population defined more broadly
than the trial's eligibility criteria requires an assumption about effect modification. A comparator
chosen for convenience rather than current practice produces an ICER that the committee will reject.
The framing should be agreed with the HTA body at a scoping meeting where the body offers one.

**Branch point:** a jurisdiction with a single national payer has one decision problem. A fragmented
system has several, and the same product may need different dossiers for different payers.

### Step 2. Assemble and critique the clinical evidence

**Input:** the trial reports, published literature, and any indirect treatment comparisons.

**Output:** an evidence table with study design, population, comparator, outcomes, follow-up, and risk
of bias; a statement of the direct and indirect evidence; and an assessment of whether the trial's
comparator is relevant in each market.

**Rationale:** Most HTA disputes are about the evidence, not the arithmetic. The common failures are a
trial comparator that is not the local standard of care, a surrogate endpoint without a validated
relationship to the final outcome, a population narrower than the licensed indication, and a
follow-up shorter than the decision horizon.

**Branch point:** when no head-to-head trial exists, an indirect comparison is required. Network
meta-analysis is possible only if the evidence network is connected and the populations are
sufficiently similar. A naive comparison across trials is not an indirect comparison.

### Step 3. Choose the model structure

**Input:** the disease natural history, the outcomes that matter, and the data available.

**Output:** a model specification with states or events, cycle length, time horizon, and the
rationale for the structure.

**Rationale:** The structure encodes clinical assumptions. A Markov model with a fixed cycle length
assumes that transitions happen within cycles and that the cycle is short enough to approximate
continuous time. A partitioned survival model fits survival curves to trial data and assumes that the
relationship between arms persists beyond follow-up. A decision tree is appropriate for a short,
one-off event.

**Branch point:** if the treatment effect is on a time-to-event outcome and extrapolation beyond the
trial is required, the choice of survival model (exponential, Weibull, Gompertz, log-normal, spline)
becomes a structural assumption that can change the ICER more than the treatment effect itself.

### Step 4. Populate baseline risk and natural history

**Input:** epidemiological data, registry data, and the comparator arm of the trial.

**Output:** baseline event rates, mortality, and utility values by health state, with sources.

**Rationale:** Baseline risk drives absolute benefit. A relative risk reduction applied to a low
baseline produces a small absolute gain and a high ICER. A jurisdiction with different baseline
mortality may therefore reach a different conclusion from the same relative effect.

**Branch point:** trial baseline risk may not represent the target population. If the trial enrolled
healthier patients, applying the trial's absolute risk to the population overstates benefit.

### Step 5. Estimate relative effects and extrapolate

**Input:** the trial effect estimates and the model structure.

**Output:** relative effects by outcome, with confidence intervals, and the extrapolation assumptions.

**Rationale:** The relative effect is usually assumed constant over time, which is a strong
assumption. If the treatment delays progression but does not cure, the hazard ratio may change after
the trial ends. The extrapolation should be tested against external data where possible.

### Step 6. Cost the resources

**Input:** resource use from the trial or a Delphi panel, unit costs from national tariffs, and the
perspective.

**Output:** a cost table by health state and year, in local currency and a stated year.

**Rationale:** Costs must be relevant to the perspective. A health-system perspective excludes
productivity losses; a societal perspective includes them. Unit costs must come from the right year
and be inflated or deflated consistently. A model that mixes a 2024 tariff with a 2026 drug price
produces an ICER that cannot be reproduced.

### Step 7. Measure health outcomes

**Input:** utility values, survival, and the quality-of-life instrument used in the trial.

**Output:** QALYs by arm, with the source of utilities and the method for mapping from the trial
instrument.

**Rationale:** Utilities are often the weakest input. Trial-based utilities may be measured with a
generic instrument that is insensitive in the condition, and the mapping algorithm may not have been
validated in that population. The committee will ask which instrument, which tariff, and which
population.

### Step 8. Discount future costs and effects

**Input:** the discount rate required by the HTA body and the time profile of costs and effects.

**Output:** present values for costs and outcomes.

**Rationale:** Discounting reflects that costs and benefits occurring later are worth less today. The
rate is a convention, not a physical constant: England uses 3.5% for the reference case, with 1.5%
permitted in defined circumstances for technologies that restore near-full health for people who
would otherwise die or have a very severely impaired life. Using the wrong rate changes the result
for long-horizon technologies.

### Step 9. Compute the ICER and net monetary benefit

**Input:** discounted costs and effects for each arm.

**Output:** incremental cost, incremental effect, ICER, and NMB at the relevant thresholds.

**Rationale:** The ICER is a ratio, and a ratio is unstable when the denominator is small. The NMB
is more stable and is the appropriate quantity for probabilistic analysis. Both should be reported,
with the ICER accompanied by its comparator and perspective.

### Step 10. Quantify uncertainty

**Input:** the input distributions and the structural alternatives.

**Output:** a tornado diagram for one-way analysis, a scatter plot on the cost-effectiveness plane, a
CEAC, and scenario results for structural alternatives.

**Rationale:** The committee needs to know whether the decision is robust. A deterministic ICER of
$51,000 with a 90% interval from $32,000 to $80,000 supports a different decision from the same point
estimate with an interval from $20,000 to $300,000.

### Step 11. Build the budget-impact model

**Input:** the eligible population, uptake assumptions, net cost per patient, and the time profile.

**Output:** annual net budget impact for the payer, usually over one to five years, with scenarios.

**Rationale:** Budget impact uses the same per-patient net cost as the cost-effectiveness model but
multiplies by the treated population. It is often the number that determines whether a positive
recommendation can be implemented.

**Branch point:** a one-year budget impact with a fixed uptake assumption can be gamed. A three-year
model with a plausible diffusion curve is more defensible and is what most payers expect.

### Step 12. Analyse subgroups and equity

**Input:** the trial subgroups, the HTA body's requirements, and any distributional analysis guidance.

**Output:** subgroup results with a clear statement of which are pre-specified and which are
exploratory, and an equity analysis where required.

**Rationale:** Subgroup analyses multiply the chance of a spurious finding. A pre-specified subgroup
with a plausible biological rationale is evidence; a post hoc subgroup that happens to be
cost-effective is a hypothesis. NICE's methods also allow distributional cost-effectiveness analysis
to examine health inequality effects.

### Step 13. Write the dossier to the body's template

**Input:** the evidence, the model, and the body's submission template.

**Output:** a dossier with the required sections, an executive summary, and a response to the body's
likely questions.

**Rationale:** Each body has a template because each asks a different question in a different order.
NICE expects a reference-case analysis and a structured critique of the evidence. ICER expects a
comparative clinical effectiveness assessment and a health-benefit price benchmark. IQWiG expects an
assessment of added benefit against the appropriate comparator. HAS expects clinical benefit ratings
and, for certain products, an economic evaluation. A single global dossier reused without adaptation
will fail at least one of them.

### Step 14. Respond to the assessment report

**Input:** the preliminary assessment and the committee's questions.

**Output:** a written response with corrections, clarifications, and any reanalysis.

**Rationale:** The assessment report is the last opportunity to correct a factual error or to explain
an assumption. A response that repeats the dossier rather than addressing the specific question
wastes the opportunity.

### Step 15. Plan for the decision and after it

**Input:** the recommendation and any conditions.

**Output:** a plan for managed access, a price agreement, a registry, or a resubmission.

**Rationale:** A negative recommendation is not always final. A product can return with new evidence,
a lower price, or a restricted population. A positive recommendation with managed access creates data
obligations that must be planned before launch.

### 5.15 Decision table: model structure

| Structure | Best for | Key assumption | Failure mode |
|---|---|---|---|
| Decision tree | Short, one-off events; no recurrence | Events occur once within the horizon | Poor fit for chronic disease with repeated transitions |
| Markov cohort | Chronic disease with repeated health states | Transition probabilities constant within a cycle; cycle length approximates time | Cycle length too long; memoryless assumption |
| Partitioned survival | Oncology with time-to-event endpoints | Survival curves represent the counterfactual beyond follow-up | Extrapolation dominates the result |
| Discrete-event simulation | Complex pathways, resource constraints, individual history | Event times and attributes can be modelled | Opaque to reviewers; hard to validate |

### 5.16 Decision table: comparator choice

| Comparator | When it is right | Risk | HTA reaction |
|---|---|---|---|
| Current standard of care | Always, as the reference case | None if correctly identified | Expected |
| The most cost-effective alternative | When the body specifies it | May not reflect practice | Acceptable if justified |
| Placebo or no treatment | Only when no effective treatment exists | Overstates benefit | Rejected if an active comparator exists |
| A different product in the class | When the body asks for a class comparison | Requires indirect evidence | Often requested |

### 5.17 Decision table: presenting uncertainty

| Method | What it answers | What it does not answer | When required |
|---|---|---|---|
| One-way sensitivity | Which inputs matter most | Joint uncertainty | Almost always |
| Probabilistic sensitivity analysis | Probability of cost-effectiveness | Structural error | NICE reference case and most bodies |
| Scenario analysis | Effect of structural choices | Probability | When structure is contested |
| Expected value of information | Whether more research is worth it | Affordability | For research prioritisation |
| Budget-impact scenarios | Affordability under uptake uncertainty | Cost-effectiveness | Payer submissions |

### 5.18 Edge cases

**Payer rejection after a positive HTA recommendation.** An HTA recommendation does not guarantee
local funding. A payer may impose prior authorisation, a narrower population, or a patient access
scheme. The operational response is a reimbursement strategy per payer, not a single national
assumption.

**Price negotiation failure.** When the price demanded by the sponsor exceeds what the payer will
pay, the product may launch with restricted access or not at all. The negotiation is informed by the
ICER: the price at which the ICER equals the threshold is the maximum the payer can justify. A
sponsor that does not know that price is negotiating without a number.

**Cross-border joint assessment.** Under the EU HTA Regulation, a joint clinical assessment is
produced at EU level, but the reimbursement decision remains national. A sponsor must prepare for
both: the JCA dossier and the national economic submission.

**Immature trial data.** A single-arm trial with a surrogate endpoint may support accelerated
approval and still be insufficient for HTA. The gap is usually a comparative effect and a
final-outcome benefit; managed access or a confirmatory trial is the bridge.

**Protocol amendment mid-study.** A change to the primary endpoint or population changes the
evidence available to the HTA body. The economic model must be rebuilt around the amended endpoint,
and the dossier must explain the change rather than presenting the final analysis as if it were
pre-specified.

**Site under-enrolment.** A trial that under-recruits may be underpowered for the subgroup that the
HTA body cares about. The model then relies on indirect evidence, and the uncertainty widens at the
worst moment.

### 5.19 Derive the threshold price

A sponsor often needs to know the price at which the product becomes cost-effective, not only the
ICER at the proposed price. The calculation inverts the ICER:

```text
Price at threshold = (threshold x incremental effect + comparator cost - other new costs)
                     / discounted quantity of the new technology
```

For a product with a three-year treatment duration, the denominator is the discounted sum of the
first three years, about 2.80 at a 3.5% discount rate. A threshold of $100,000 per QALY and an
incremental effect of 1.85 QALYs implies a price of about $77,000 per year in this example, before
subtracting the other costs.

The threshold price is the number that should anchor a price negotiation. It is also the number that
shows how far the proposed price sits from the payer's willingness to pay. A sponsor that enters a
negotiation without it is arguing about margin rather than about value.

### 5.20 Prepare for a joint clinical assessment

Under the EU HTA Regulation, a joint clinical assessment (JCA) covers the clinical evidence for
selected technologies, starting with cancer medicines containing a new active substance and advanced
therapy medicinal products from 12 January 2025, orphan medicines from 13 January 2028, and all
remaining centrally authorised medicines from 13 January 2030.

The JCA is a clinical assessment, not a reimbursement decision. It produces a report on relative
effectiveness and safety that national HTA bodies then use alongside their own economic evaluation.
The dossier must follow the agreed PICO, which is fixed early in the procedure, and the timeline runs
in parallel with the central marketing authorisation.

The practical consequence is that the evidence package must serve two audiences at once: the
European assessors, who need a systematic and transparent clinical comparison, and the national
bodies, who need an economic model with local costs and a local comparator. A JCA report does not
replace the national dossier.

### 5.21 Distinguish HTA evidence from payer evidence

An HTA body and a payer are not the same organisation, even when the payer funds the HTA body. The HTA
body asks whether the technology is cost-effective under a reference case. The payer asks whether it
can be afforded within a budget, for a defined population, with the administrative controls it can
operate.

| Question | HTA body | Payer |
|---|---|---|
| Is the price justified by the health gain? | Yes, through the ICER | Yes, but also through the budget |
| Which patients are eligible? | The licensed indication, sometimes restricted | The prior-authorisation criteria |
| What happens over three years? | Budget-impact analysis | Cash-flow and utilisation management |
| What if uptake is higher than expected? | Scenario analysis | Budget overrun |

A submission that answers only the HTA question can receive a positive recommendation and still fail
to secure access.

### 5.22 What a defensible dossier contains

| Section | Question it answers | Common weakness |
|---|---|---|
| Disease burden | Why does this matter? | Generic epidemiology not linked to the decision |
| Product and comparator | What is being compared with what? | Comparator chosen for convenience |
| Clinical evidence | What is the comparative effect? | Surrogate without validation |
| Economic model | What is the cost per QALY? | Structure not justified |
| Uncertainty | How confident is the result? | Only one-way analysis |
| Budget impact | Can the payer afford it? | Uptake assumption unsupported |
| Subgroups | Who benefits most? | Post hoc analyses presented as confirmatory |
| Equity | Who gains and who loses? | Absent |
| Implementation | How will it be used? | No plan for prior authorisation or registry |

## 6. Worked example

### 6.1 The product and the inputs

A new oral therapy for a chronic condition is compared with standard of care. The trial shows an
improvement in a composite endpoint over a median follow-up of three years. The sponsor extrapolates
to a lifetime horizon and builds a Markov model.

The inputs, in 2026 US dollars, are:

| Input | New therapy | Standard of care |
|---|---:|---:|
| Acquisition cost, years 1 to 3 | $45,000 per year | $0 |
| Monitoring and administration, years 1 to 10 | $2,000 per year | $0 |
| Comparator drug cost, years 1 to 6 | $0 | $8,000 per year |
| Utility while alive | 0.85 | 0.75 |
| Survival assumed in the model | 8 years | 6 years |
| Discount rate | 3.5% | 3.5% |
| Eligible population | 5,000 patients | |
| Uptake, years 1 to 3 | 20%, 30%, 40% | |

The prices are illustrative planning assumptions for the exercise, not a real product's price.

### 6.2 Discount the costs

The present value of a constant annual amount is the amount multiplied by the annuity factor:

```text
Annuity factor, 3 years at 3.5%  = 2.8016
Annuity factor, 6 years at 3.5%  = 5.3286
Annuity factor, 10 years at 3.5% = 8.3166
```

New therapy:

```text
Acquisition = $45,000 x 2.8016 = $126,072
Monitoring  = $2,000 x (8.3166 - 2.8016) = $2,000 x 5.5150 = $11,030
Total       = $137,102
```

Standard of care:

```text
Comparator = $8,000 x 5.3286 = $42,629
```

Incremental cost = $137,102 - $42,629 = $94,473. The code in Section 7 reports $94,475 because it
does not round the annuity factors; the difference is immaterial.

### 6.3 Discount the outcomes

```text
New therapy QALYs = 0.85 x annuity(8 years) = 0.85 x 6.8740 = 5.843
Standard QALYs    = 0.75 x annuity(6 years) = 0.75 x 5.3286 = 3.996
Incremental QALYs = 5.843 - 3.996 = 1.847
```

### 6.4 Compute the ICER and net monetary benefit

```text
ICER = $94,473 / 1.847 = $51,150 per QALY
```

At a threshold of $100,000 per QALY:

```text
NMB = $100,000 x 1.847 - $94,473 = $184,700 - $94,473 = $90,227
```

The positive NMB means the therapy is cost-effective at that threshold. At $50,000 per QALY:

```text
NMB = $50,000 x 1.847 - $94,473 = $92,350 - $94,473 = -$2,123
```

The therapy is marginally not cost-effective at $50,000. The ICER of about $51,000 sits just above
that threshold, which is why the choice of threshold changes the conclusion.

### 6.5 Budget impact

The net cost per treated patient in years 1 to 3 is the new therapy cost minus the comparator cost
that is avoided:

```text
Net per patient per year = $45,000 + $2,000 - $8,000 = $39,000
Year 1: 5,000 x 0.20 x $39,000 = $39,000,000
Year 2: 5,000 x 0.30 x $39,000 = $58,500,000
Year 3: 5,000 x 0.40 x $39,000 = $78,000,000
Three-year total = $175,500,000
```

The ICER is favourable at the US benchmark and the three-year budget impact is $175.5 million. Both
can be true at once. A payer with a fixed budget may still restrict access, because the money has to
come from somewhere.

### 6.6 Diagnostic 1: the comparator drives the ICER

The ICER is a property of the comparison, not of the product. Change the comparator and the same
clinical evidence produces a different number.

| Comparator | Annual cost | Present value | ICER |
|---|---:|---:|---:|
| Standard of care | $8,000 | $42,629 | $51,150 |
| An expensive active comparator | $20,000 | $106,572 | $16,500 |
| Best supportive care | $2,000 | $10,657 | $68,500 |

The range is more than fourfold. A dossier that presents a single ICER without naming the comparator
is not presenting a decision input. The HTA body will use its own comparator, and the sponsor should
model it before submission rather than after.

### 6.7 Diagnostic 2: the extrapolation drives the conclusion

The trial followed patients for a median of three years. The model assumes eight years of survival
for the new therapy. That eight-year figure is an extrapolation from a survival curve, not an
observation.

If the benefit duration is 6.5 years instead of 8, with the same utility:

```text
New QALYs = 0.85 x annuity(6.5 years) = 0.85 x 5.721 = 4.863
Incremental QALYs = 4.863 - 3.996 = 0.867
ICER = $94,473 / 0.867 = $108,976
```

The ICER rises from about $51,000 to about $109,000, crossing the lower end of ICER's benchmark
range. A reduction of 1.5 years in an unobserved survival assumption moves the product from clearly
cost-effective to marginal.

This is why HTA committees ask about extrapolation. The model's conclusion is often more sensitive
to the survival curve than to the treatment effect measured in the trial.

### 6.8 Vary one assumption: a 25% price reduction

Suppose the sponsor reduces the annual price from $45,000 to $33,750, holding everything else
constant.

```text
Acquisition = $33,750 x 2.8016 = $94,554
Total new cost = $94,554 + $11,030 = $105,584
Incremental cost = $105,584 - $42,629 = $62,955
ICER = $62,955 / 1.847 = $34,086 per QALY
```

The ICER falls from about $51,000 to about $34,000. That is below the upper end of NICE's 2026 range
of £25,000 to £35,000 per QALY if one ignores currency and perspective differences, which one should
not do: NICE's threshold is in pounds and reflects NHS opportunity cost, while this model is in US
dollars and a US health-system perspective. The comparison is illustrative of direction, not of a
decision.

The budget impact falls proportionally:

```text
Net per patient = $33,750 + $2,000 - $8,000 = $27,750
Three-year total = 5,000 x (0.20 + 0.30 + 0.40) x $27,750 = $124,875,000
```

A 25% price cut reduces the ICER by about a third and the three-year budget by about 29%. The
conclusion about cost-effectiveness changes; the conclusion about affordability improves but does
not disappear.

### 6.9 What the example teaches

1. An ICER is a ratio with a comparator in the denominator's story. Name the comparator or the number
   is meaningless.
2. Discounting is mechanical but must be done consistently, with the annuity factors shown.
3. The budget impact is a separate number and can contradict the cost-effectiveness conclusion.
4. Extrapolation is often the largest source of uncertainty, and it is a structural assumption rather
   than a parameter.
5. Price and threshold are linked: the same product can be cost-effective at one price and not at
   another, and the threshold price is computable.

## 7. Code, query, or calculation

The model computes discounted costs and QALYs, the ICER, a probabilistic sensitivity analysis, and a
budget impact. It uses only the standard library.

**Assumptions:**

- costs and effects are discounted at 3.5% per year;
- the new therapy costs $45,000 per year for three years plus $2,000 per year for ten years;
- the comparator costs $8,000 per year for six years;
- utilities are 0.85 and 0.75, applied for 8 and 6 years;
- probabilistic analysis draws incremental cost from a normal distribution with standard deviation
  $18,000 and incremental QALYs from a normal distribution with standard deviation 0.35, truncated at
  0.3;
- budget impact uses 5,000 eligible patients and uptake of 20%, 30%, and 40%.

```python
import random
import statistics

R = 0.035

def pv(amount, years):
    return sum(amount / (1 + R) ** t for t in range(1, int(years) + 1))

def pv_q(utility, years):
    whole = int(years)
    frac = years - whole
    total = sum(utility / (1 + R) ** t for t in range(1, whole + 1))
    if frac:
        total += utility * frac / (1 + R) ** (whole + 1)
    return total

def base_case(price=45_000, new_years=8.0, old_cost=8_000, old_years=6.0):
    new_cost = pv(price, 3) + pv(2_000, 10) - pv(2_000, 3)
    old = pv(old_cost, old_years)
    new_q = pv_q(0.85, new_years)
    old_q = pv_q(0.75, old_years)
    return new_cost - old, new_q - old_q

dc, dq = base_case()
print(f"incremental cost USD 2026 {dc:,.0f}")
print(f"incremental QALYs {dq:.3f}")
print(f"ICER USD 2026 per QALY {dc / dq:,.0f}")

dc2, dq2 = base_case(price=45_000 * 0.75)
print(f"ICER at 25% price cut {dc2 / dq2:,.0f}")
dc3, dq3 = base_case(new_years=6.5)
print(f"ICER at 6.5-year benefit {dc3 / dq3:,.0f}")

def psa(n=10_000, seed=2026):
    rng = random.Random(seed)
    ce = {50_000: 0, 100_000: 0, 150_000: 0}
    icers = []
    for _ in range(n):
        c = max(0.0, rng.gauss(dc, 18_000))
        q = max(0.3, rng.gauss(dq, 0.35))
        icers.append(c / q)
        for k in ce:
            if c - k * q < 0:
                ce[k] += 1
    icers.sort()
    return ce, icers

ce, icers = psa()
print(f"PSA P(cost-effective at 50k) {ce[50_000] / 10_000:.3f}")
print(f"PSA P(cost-effective at 100k) {ce[100_000] / 10_000:.3f}")
print(f"PSA P(cost-effective at 150k) {ce[150_000] / 10_000:.3f}")
print(f"PSA median ICER {statistics.median(icers):,.0f}")
print(f"PSA 90% interval {icers[500]:,.0f} to {icers[9_500]:,.0f}")

eligible = 5_000
uptake = (0.20, 0.30, 0.40)
for year, u in enumerate(uptake, start=1):
    print(f"year {year} net budget USD 2026 {eligible * u * 39_000:,.0f}")
print(f"3-year total {sum(eligible * u * 39_000 for u in uptake):,.0f}")
```

**Expected output:**

```text
incremental cost USD 2026 94,475
incremental QALYs 1.846
ICER USD 2026 per QALY 51,166
ICER at 25% price cut 34,096
ICER at 6.5-year benefit 108,980
PSA P(cost-effective at 50k) 0.472
PSA P(cost-effective at 100k) 0.990
PSA P(cost-effective at 150k) 0.999
PSA median ICER 50,989
PSA 90% interval 31,942 to 79,776
year 1 net budget USD 2026 39,000,000
year 2 net budget USD 2026 58,500,000
year 3 net budget USD 2026 78,000,000
3-year total 175,500,000
```

**Interpretation:**

1. The deterministic ICER is $51,166 and the probabilistic median is $50,989. The two agree because
   the input distributions are symmetric and the ratio is not near zero.
2. The probability of being cost-effective is 47.2% at $50,000 and 99.0% at $100,000. A decision
   threshold near the ICER produces a coin-flip probability, which is the correct interpretation of
   a point estimate at the threshold.
3. The 25% price cut moves the ICER to $34,096, a change large enough to cross a threshold in the
   $25,000 to $35,000 range.
4. The 6.5-year benefit moves the ICER to $108,980, which is above ICER's $100,000 lower benchmark.
   The extrapolation, not the treatment effect, decides the case.

**What to vary:** the standard deviation of incremental QALYs. It controls the width of the interval
and therefore the probability near the threshold. A programme that reports only the deterministic
ICER hides the range in which the decision is uncertain.

## 8. Data, standards, and regulatory context

HTA is not regulated in the way that medicines are. There is no international treaty that binds an
HTA body's methods, and the bodies differ in law, structure, and appetite for modelling. What is
binding is the national statute or mandate that creates each body and the procedural rules it
applies.

### 8.1 The bodies and what they decide

| Body | Jurisdiction | What it produces | Binding on whom |
|---|---|---|---|
| NICE | England and Wales | Technology appraisal guidance | NHS bodies are required to fund recommendations within a stated period |
| ICER | United States | Assessment reports and health-benefit price benchmarks | Not binding; informs payers and manufacturers |
| IQWiG | Germany | Assessment of added benefit | Informs the G-BA, which decides |
| G-BA | Germany | Benefit assessment and directives | Binding on statutory insurers |
| HAS Transparency Committee | France | SMR and ASMR ratings | Informs reimbursement and price negotiation |
| Canada's Drug Agency (formerly CADTH) | Canada | Reimbursement recommendations | Informs provincial payers and the pan-Canadian agreement |
| PBAC | Australia | Recommendations on PBS listing | Informs the Minister's decision |
| EU HTA Coordination Group | European Union | Joint clinical assessment reports | Feeds national HTA, but does not decide reimbursement |

The table matters because a sponsor often says "HTA" as if it were one process. It is at least eight
processes with different questions, different evidence standards, and different decision rules.

### 8.2 England: NICE

NICE's methods are set out in the technology appraisal and highly specialised technologies manual
(PMG36). The reference case uses a 3.5% annual discount rate for costs and health effects, a health
and social care perspective, and QALYs as the outcome. A 1.5% rate may be presented as a
non-reference-case analysis for technologies that restore near-full health to people who would
otherwise die or have a very severely impaired life, with benefits sustained over a long period.

The cost-effectiveness range changed on 31 March 2026. The previous range of £20,000 to £30,000 per
QALY was replaced by £25,000 to £35,000 per QALY. Below £25,000, decisions rest primarily on the
cost-effectiveness estimate; between £25,000 and £35,000, factors such as uncertainty, uncaptured
benefits, and health inequalities are weighed; above £35,000, an increasingly stronger case is
needed. Highly specialised technologies retain a reference level of £100,000 per QALY, with QALY
weights up to 3.

NICE also operates a budget-impact test, with a threshold that was updated to £40 million in
June 2025. A technology whose budget impact exceeds the threshold can be subject to a commercial
arrangement even if it is cost-effective.

The change in threshold is a reminder that HTA methods are not fixed. A model built to the 2025
threshold may need to be re-run for a 2026 appraisal, and a dossier should state the version of the
manual it follows.

### 8.3 United States: ICER

ICER is an independent, non-governmental body. It produces assessments of comparative clinical
effectiveness and a health-benefit price benchmark, and its recommendations are not coverage
decisions. The benchmark range is $100,000 to $150,000 per QALY or per equal value of life year
(evLY), and ICER reports threshold analyses at $50,000, $100,000, $150,000, and $200,000.

ICER also operates an annual budget-impact threshold, which it lowered for 2025-2026 to
$821 million from $880 million. Crossing it triggers an access and affordability alert. The
threshold is a signal, not a rule; US payers make their own coverage decisions.

The US system has no single national HTA decision. Commercial insurers, Medicare, and Medicaid each
apply different rules, and Medicare's coverage determinations are separate from ICER's assessments.

### 8.4 Germany: IQWiG and the G-BA

Under the pharmaceutical market reform (AMNOG), a new medicine is assessed for added benefit against
an appropriate comparator therapy. IQWiG prepares the assessment; the G-BA makes the decision. The
extent of added benefit is classified as major, considerable, minor, non-quantifiable, or no added
benefit proven, with a separate category for less benefit. Each conclusion also carries an evidence
level: proof, indication, or hint.

Germany does not use a cost-per-QALY threshold for the early benefit assessment. The assessment is
about added benefit; the price is negotiated separately. IQWiG has used efficiency-frontier methods
in some assessments, but the routine early benefit assessment is not an ICER exercise.

### 8.5 France: HAS

The Transparency Committee rates the clinical benefit (SMR) and the added benefit (ASMR) on scales
from I to V. SMR determines whether the product is reimbursed and at what rate; ASMR informs the
price negotiation with the economic committee. For products with a significant added benefit, HAS
also produces an economic evaluation.

The French process is deliberative and does not publish a single cost-per-QALY threshold. A sponsor
that submits an ICER without addressing the clinical benefit rating has answered a question the
committee did not ask first.

### 8.6 European Union: joint clinical assessment

Regulation (EU) 2021/2282 applies from 12 January 2025 to new cancer medicines containing a new
active substance and to advanced therapy medicinal products. Orphan medicines join from
13 January 2028, and all remaining centrally authorised medicines from 13 January 2030.

A joint clinical assessment covers clinical evidence, not cost-effectiveness. It runs in parallel
with the central marketing authorisation, with the PICO fixed early, a dossier due within a defined
window, and a report finalised around the marketing authorisation date. National HTA bodies remain
responsible for economic evaluation and reimbursement.

### 8.7 What the methods do not constrain

The bodies do not agree on:

- the threshold, or whether to have one;
- the perspective (health system, payer, or societal);
- the preferred model structure;
- the treatment of uncertainty;
- the role of subgroup analysis;
- the acceptability of surrogate endpoints.

A dossier cannot be simultaneously optimal for all of them. It can be transparent about the choices
and provide the reference-case analysis each body asks for.

> **Note:** HTA methods change, sometimes annually. The thresholds and dates above were published at
> the access date and should be re-checked before any submission. This is educational material, not
> legal or reimbursement advice.

## 9. Industry practice and case evidence

### 9.1 What the 2025 ICER analysis found

ICER's 2025 Launch Price and Access Report examined 154 novel drugs approved in the United States
between 2022 and 2024. It reported that median inflation-adjusted list launch prices rose by 24% and
net prices by 51% over the period. Of the 23 drugs ICER reviewed, 16 (70%) had net prices above the
upper end of the health-benefit price benchmark, implying $1.26 billion to $1.49 billion in
first-year excess spending.

These are ICER's estimates under its own methods, not audited prices. They are still the most
detailed public attempt to compare launch prices with a value benchmark, and they illustrate the gap
between what manufacturers charge and what an independent assessor considers justified by health
gain.

### 9.2 Documented divergence between bodies

The same product can receive different ratings in different countries. Published comparisons of
G-BA and HAS decisions have found moderate agreement overall but notable disagreements in both
directions. One often-cited example is vandetanib, rated ASMR IV in France and assessed as having no
added benefit in Germany; rilpivirine showed the reverse pattern.

The divergence is not necessarily an error by either body. The comparators, the accepted evidence,
and the decision rules differ. It does mean that a single global value story is unlikely to be
accepted everywhere without adaptation.

### 9.3 The Alzheimer's case

The first anti-amyloid antibodies for early Alzheimer's disease illustrate the approval-access gap.
Regulators in the United States and Europe approved products on the basis of a statistically
significant reduction in amyloid and a modest clinical effect. In 2024, NICE did not recommend them
for NHS use on cost-effectiveness grounds, and the companies subsequently announced discounted
access arrangements in some settings.

The case is documented in the public decisions, though the interpretation is contested. The
inference for practice is that a regulatory endpoint can be sufficient for approval and insufficient
for funding, and that the gap is closed by price, restricted eligibility, or further evidence, not by
repeating the approval argument.

### 9.4 How companies organise for HTA

Large manufacturers typically maintain a market-access function that sits alongside regulatory
affairs and is involved from phase 2, not after phase 3. Its outputs include:

- an evidence gap analysis against the likely HTA questions;
- a target product profile that includes the payer-relevant claims;
- a health economic model built before the trial database locks;
- a plan for the patient-reported outcome and utility data that the model needs;
- a pricing and access strategy per market.

> **Inference:** the degree to which small and mid-size companies follow this model varies widely.
> The pattern is observable in job structures and public dossiers, but no comprehensive survey is
> publicly available.

### 9.5 What the evidence does not show

There is little public evidence on the return on investment of health economics modelling. The
dossiers are confidential in many jurisdictions, and the counterfactual (what would have happened
with a different model) is unobservable. The field relies on methodological consensus and on the
decisions themselves as evidence that the analysis mattered.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong comparator | Model compares with a product not used locally | Convenience or trial design | ICER rejected as irrelevant | Confirm local standard of care before modelling |
| Surrogate endpoint without validation | Committee asks for final outcomes | Approval endpoint treated as benefit | Negative or restricted recommendation | Collect final outcomes or agree a surrogate relationship |
| Short time horizon | Costs and benefits stop at trial end | Horizon chosen to flatter | Chronic benefit undervalued | Use lifetime horizon with extrapolation |
| Inconsistent discounting | Costs discounted, outcomes not | Mechanical error | ICER biased | State rate and apply to both |
| No probabilistic analysis | Only one-way tornado | Uncertainty treated as sensitivity | Committee cannot judge decision risk | PSA and CEAC |
| Budget impact omitted | Dossier stops at ICER | Affordability treated as someone else's problem | Positive recommendation with restricted access | Three-year budget impact with uptake curve |
| Uptake assumption unsupported | 100% uptake in year 1 | Convenience | Budget overstated or understated | Diffusion curve with evidence |
| Subgroups post hoc | Many subgroup analyses, none pre-specified | Search for a positive result | Spurious claim; committee distrust | Pre-specify and label exploratory |
| Model not reproducible | Reviewer cannot rebuild the base case | Opaque spreadsheet | Rejection or rework | Provide a transparent model and technical report |
| Extrapolation unexamined | Single survival curve, no scenarios | Structural assumption hidden | ICER sensitive to an untested choice | Alternative curves and scenario analysis |
| Price above threshold price | Negotiation without a value-based ceiling | Commercial target detached from ICER | No agreement; restricted access | Compute threshold price before negotiation |
| Evidence from a narrow population | Trial population narrower than the label | Extrapolation to broader group | Committee restricts indication | Model the licensed population explicitly |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The comparator reflects local practice | Defines the incremental effect | Practice varies by region and over time | Clinical expert review and claims data |
| Relative effects are constant over time | Enables extrapolation | Hazard ratios change after treatment stops | Test with landmark analyses |
| Utilities are valid in this population | Converts survival to QALYs | Instrument insensitive or mapping unvalidated | Compare with published values |
| Costs reflect the payer's prices | Defines the numerator | Confidential discounts; list versus net | Use net-price scenarios |
| The model structure captures the disease | Enables a lifetime horizon | Disease has states or interactions not modelled | Compare with alternative structures |
| Uptake follows a plausible curve | Budget impact | Launch dynamics, prior authorisation | Scenario analysis |
| The eligible population is known | Budget impact | Off-label use, diagnostic pathway | Sensitivity on prevalence and diagnosis |
| The threshold is stable during appraisal | Decision rule | Policy change, as in England in 2026 | State the threshold version used |
| The trial population represents the licensed population | Generalisability | Narrow eligibility, geography | Subgroup and external validity review |

### 11.1 Limits of the model

The worked example is a two-state comparison with constant utilities and a fixed survival
assumption. A real model would have more states, time-varying utilities, and a survival curve fitted
to patient-level data. The arithmetic would be the same; the uncertainty would be larger.

The PSA assumes independent normal distributions for incremental cost and effect. In reality the
inputs are correlated, and the correlation can widen or narrow the interval. A production model
samples the underlying inputs, not the summary measures.

The budget impact assumes that every treated patient displaces the comparator cost for the full
year. In practice, treatment duration, persistence, and discontinuation change the net cost, and a
one-year model with a fixed uptake is a planning estimate rather than a forecast.

## 12. Exercises

### Exercise 1. Frame the decision problem (application, 30 minutes)

A new treatment for moderate-to-severe eczema is licensed for adults who have failed topical
therapy. The trial compared it with placebo plus topical therapy. In practice, many patients also
receive an older systemic immunosuppressant.

**Deliverable:** a PICO statement with two candidate comparators, the perspective, the time horizon,
and the outcomes that matter to each HTA body.

**Acceptance criteria:** the comparator choice is justified with practice evidence; the perspective
and horizon are stated; the conflict between the trial comparator and practice is identified.

**Time:** 30 minutes.

### Exercise 2. Critique the evidence base (critique, 45 minutes)

A submission relies on a single randomised trial of 300 patients, with a co-primary endpoint of
itch score and a secondary endpoint of investigator global assessment. Median follow-up is 16 weeks.

**Deliverable:** an evidence critique covering comparator, endpoint validation, follow-up, population,
and the need for indirect comparison.

**Acceptance criteria:** each issue is linked to an HTA question; the critique distinguishes a
limitation from a fatal flaw; a proposed remedy is stated for each.

**Time:** 45 minutes.

### Exercise 3. Choose and justify a model structure (design, 45 minutes)

The disease has three health states (active disease, remission, death), with relapses over a
lifetime. Trial data cover two years.

**Deliverable:** a model specification with structure, cycle length, horizon, and the assumptions
that the structure imposes.

**Acceptance criteria:** the structure matches the disease; the cycle length is justified; the
extrapolation method beyond two years is stated; at least one alternative structure is discussed.

**Time:** 45 minutes.

### Exercise 4. Compute discounted costs and ICER (calculation, 45 minutes)

A new therapy costs $30,000 per year for two years and $1,500 per year for monitoring thereafter. The
comparator costs $6,000 per year for five years. The new therapy provides 4.5 QALYs; the comparator
provides 3.2. Use a 3.5% discount rate.

**Deliverable:** present values of costs and QALYs, incremental cost, incremental QALYs, ICER, and
NMB at $50,000 and $100,000 per QALY.

**Acceptance criteria:** annuity factors shown; discounting applied to both costs and outcomes; NMB
formula stated; the decision at each threshold stated.

**Time:** 45 minutes.

### Exercise 5. Build a budget impact (application, 45 minutes)

An eligible population of 12,000 patients, uptake of 10% in year 1, 25% in year 2, and 35% in year 3.
The net cost per treated patient is $18,000 per year.

**Deliverable:** a three-year budget impact with annual and cumulative totals, and a sensitivity
analysis at half and double the uptake.

**Acceptance criteria:** arithmetic shown; the difference between budget impact and cost-effectiveness
stated; the uncertainty in uptake acknowledged.

**Time:** 45 minutes.

### Exercise 6. Interpret a CEAC (critique, 30 minutes)

A probabilistic analysis reports a 45% probability of cost-effectiveness at $50,000 per QALY, 85% at
$100,000, and 95% at $150,000.

**Deliverable:** a short interpretation for a payer committee, including what the probabilities mean
and what they do not mean.

**Acceptance criteria:** the threshold dependence is explained; the difference between probability and
confidence is stated; no claim that the product is "probably effective".

**Time:** 30 minutes.

### Exercise 7. Write the dossier outline (design, 60 minutes)

**Deliverable:** a dossier outline for NICE and a separate outline for ICER, showing where the
sections differ and why.

**Acceptance criteria:** each outline follows the body's actual questions; the differences in
perspective, threshold, and uncertainty requirements are explicit; no section is copied unchanged
between the two.

**Time:** 60 minutes.

### Exercise 8. Prepare for a price negotiation (design, 45 minutes)

The ICER at the proposed price is $70,000 per QALY. The payer's threshold is $40,000. The
incremental QALY estimate is 1.5 and the discounted treatment quantity is 2.7.

**Deliverable:** the threshold price, the implied discount, and a negotiation brief with two
alternative offers (a price cut and an outcome-based arrangement).

**Acceptance criteria:** the threshold price formula is shown; the discount is computed; the
outcome-based option states the metric, the rebate trigger, and the data source.

**Time:** 45 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

A strong PICO states:

- **Population:** adults with moderate-to-severe eczema who have failed topical therapy.
- **Intervention:** the new treatment.
- **Comparators:** (a) topical therapy alone, as in the trial; (b) an older systemic
  immunosuppressant, as used in practice.
- **Outcomes:** itch, investigator global assessment, quality of life, flares, discontinuation,
  serious infection.
- **Perspective and horizon:** health system, lifetime.

The conflict is that the trial's placebo comparator is not the practice comparator. The submission
must either provide an indirect comparison against the systemic immunosuppressant or persuade the
body that topical therapy alone is the relevant comparator for the licensed population.

### 13.2 Exercise 2

A strong critique notes:

- **Comparator:** placebo plus topical therapy may not reflect practice.
- **Endpoint:** itch score is a patient-relevant symptom, but the 16-week duration cannot establish
  durable benefit; the investigator global assessment is subjective and may be unblinded in practice.
- **Follow-up:** 16 weeks is shorter than a chronic-disease decision horizon; extrapolation is
  required and untested.
- **Population:** trial eligibility may exclude the more severe patients who would receive the
  product in practice.
- **Indirect comparison:** if the practice comparator was not studied, an indirect comparison is
  needed, with its own assumptions.

A weak critique lists limitations without saying which ones change the decision.

### 13.3 Exercise 3

A strong specification uses a Markov model with three states, a one-month cycle, and a lifetime
horizon. The cycle length is short enough to approximate continuous transitions and long enough to
keep the model tractable. Relapse probabilities come from the trial and are assumed constant after
two years, which is a structural assumption that should be tested with an alternative curve.

The specification should also state whether the model is cohort-based or individual-level, and why.
A cohort model is simpler and adequate if the population is homogeneous; an individual-level model
is needed if history matters.

### 13.4 Exercise 4

```text
Annuity 2 years = 1.8997
Annuity 5 years = 4.5151
Annuity 10 years = 8.3166
New cost = 30,000 x 1.8997 + 1,500 x (8.3166 - 1.8997)
         = 56,991 + 1,500 x 6.4169 = 56,991 + 9,625 = 66,616
Comparator cost = 6,000 x 4.5151 = 27,091
Incremental cost = 39,525
Incremental QALYs = 4.5 - 3.2 = 1.3
ICER = 39,525 / 1.3 = 30,404
NMB at 50,000 = 50,000 x 1.3 - 39,525 = 65,000 - 39,525 = 25,475
NMB at 100,000 = 130,000 - 39,525 = 90,475
```

The product is cost-effective at both thresholds, with a larger margin at $100,000.

### 13.5 Exercise 5

```text
Year 1: 12,000 x 0.10 x $18,000 = $21,600,000
Year 2: 12,000 x 0.25 x $18,000 = $54,000,000
Year 3: 12,000 x 0.35 x $18,000 = $75,600,000
Cumulative = $151,200,000
```

At half uptake the total is $75,600,000; at double uptake it is $302,400,000. The cost-effectiveness
result does not change, because the per-patient ratio is unchanged. The affordability conclusion
does.

### 13.6 Exercise 6

The probabilities are the share of simulated simulations in which the technology's net monetary
benefit is positive at each threshold. They are not the probability that the technology works, and
they are not a confidence interval for the ICER. A 45% probability at $50,000 means the decision at
that threshold is close to a coin flip under the model's assumptions and input distributions. A 95%
probability at $150,000 means the decision is robust at that threshold unless the model structure is
wrong.

### 13.7 Exercise 7

A NICE outline follows the reference case: population, intervention, comparator, clinical evidence,
cost-effectiveness, uncertainty, budget impact, equality, and implementation. It expects a
probabilistic analysis and a statement of the reference-case assumptions.

An ICER outline follows the value assessment framework: comparative clinical effectiveness,
long-term cost-effectiveness, a health-benefit price benchmark, and contextual considerations
including affordability and access. It expects a price benchmark rather than a threshold decision.

The difference is not cosmetic. NICE asks whether the NHS should fund at the proposed price; ICER
asks what price would be justified by the evidence.

### 13.8 Exercise 8

```text
Threshold price = (threshold x incremental QALYs + comparator cost - other new costs)
                  / discounted treatment quantity
```

With a threshold of $40,000, incremental QALYs of 1.5, and a discounted quantity of 2.7, the value of
the health gain is $60,000. Dividing by 2.7 gives about $22,222 per year before other costs. If the
proposed price is $45,000, the implied discount is about 51%.

An outcome-based option might tie 20% of the price to a reduction in flare rate at 12 months, with a
rebate if the observed rate exceeds a pre-specified threshold. The metric, the threshold, the data
source, and the reconciliation process must all be specified before launch.

## 14. Interview questions

**1. Why can a product be approved and not reimbursed?**
Strong: approval answers safety and efficacy; reimbursement answers comparative value and
affordability, often against a different comparator and population. Weak: blames politics.

**2. What is an ICER and what does it hide?**
Strong: incremental cost per QALY, hiding the comparator, perspective, horizon, and the uncertainty
around each. Weak: defines the formula only.

**3. How does the choice of comparator change the result?**
Strong: it changes the incremental cost and effect, sometimes by a factor of four or more, and the
body will use its own. Weak: says the comparator should be standard of care without exploring.

**4. What is the difference between cost-effectiveness and budget impact?**
Strong: one is health per unit of money, the other is total spend; a product can pass one and fail
the other. Weak: treats them as the same analysis.

**5. Why is extrapolation a structural assumption?**
Strong: it determines lifetime benefit beyond the trial, and alternative survival models can change
the ICER more than the treatment effect. Weak: calls it a sensitivity analysis.

**6. What does a CEAC show?**
Strong: the probability of cost-effectiveness at each threshold under the model's uncertainty, not
the probability that the drug works. Weak: reads it as a confidence interval.

**7. How would you set a price for a new product?**
Strong: compute the threshold price from the incremental effect and the payer's threshold, then
negotiate around it with evidence on uncertainty and affordability. Weak: prices to the nearest
competitor.

**8. What is the difference between NICE, ICER, and IQWiG?**
Strong: NICE makes binding funding recommendations with a threshold; ICER benchmarks price without
binding power; IQWiG assesses added benefit without a QALY threshold. Weak: describes them all as
"HTA bodies" without the decision rules.

**9. What would you do if the trial comparator is wrong for the market?**
Strong: model the local comparator with an indirect comparison, state the assumption, and consider a
local trial or a managed-access arrangement. Weak: submits the trial result and hopes.

## 15. Key terms

| Term | Meaning | See |
|---|---|---|
| ASMR | French added-benefit rating | 8.5 |
| Budget impact | Net cost to the payer of adoption | 3.7 |
| CEAC | Probability of cost-effectiveness by threshold | 5.10 |
| Comparator | Technology compared against | 5.1 |
| Discount rate | Annual rate applied to future costs and effects | 5.8 |
| evLY | Equal value of life year | 8.3 |
| G-BA | German decision body | 8.4 |
| HAS | French HTA body | 8.5 |
| HBPB | ICER's health-benefit price benchmark | 8.3 |
| HTA | Health technology assessment | 3.1 |
| ICER (measure) | Incremental cost-effectiveness ratio | 3.4 |
| ICER (organisation) | US independent assessment body | 8.3 |
| IQWiG | German assessment institute | 8.4 |
| JCA | EU joint clinical assessment | 8.6 |
| Managed access | Coverage conditional on evidence | 4 |
| NMB | Net monetary benefit | 4.1 |
| NICE | English HTA body | 8.2 |
| PICO | Population, intervention, comparator, outcome | 5.1 |
| PSA | Probabilistic sensitivity analysis | 5.10 |
| QALY | Quality-adjusted life year | 5.7 |
| Reference case | Required analytical conventions | 3.3 |
| SMR | French clinical-benefit rating | 8.5 |
| Threshold | Willingness to pay per QALY | 3.4 |
| Threshold price | Price at which ICER equals the threshold | 5.19 |

## 16. Sources

All URLs accessed 26 October 2026.

1. NICE. *NICE technology appraisal and highly specialised technologies guidance: the manual*
   (PMG36), including the March 2026 threshold update.
   https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/
2. NICE. *Changes to NICE's cost-effectiveness thresholds*.
   https://www.nice.org.uk/what-nice-does/faqs/changes-to-nice-s-cost-effectiveness-thresholds
3. ICER. *Value Assessment Framework*.
   https://icer.org/our-approach/methods-process/value-assessment-framework/
4. ICER. *Reference Case for Economic Evaluations*, v2.2, October 2025.
   https://icer.org/wp-content/uploads/2024/02/ICER_Reference-Case_For-Publication_070126.pdf
5. ICER. *2025 Launch Price and Access Report*.
   https://icer.org/wp-content/uploads/2025/10/ICER_2025_Launch-Price-and-Access-Final-Report_For-Publication.pdf
6. European Commission. *Joint clinical assessments for medicinal products*.
   https://health.ec.europa.eu/health-technology-assessment/implementation-regulation-health-technology-assessment/joint-clinical-assessments-medicinal-products_en
7. European Commission. *Regulation (EU) 2021/2282 on health technology assessment: FAQ*.
   https://health.ec.europa.eu/document/download/b81a42c4-b2ba-47bc-b807-0c667c8de32e_en
8. IQWiG. *Early benefit assessment*.
   https://www.iqwig.de/en/presse/press-releases/press-releases-detailpage_30784.html
9. G-BA. *Categories of additional benefit*.
   https://www.g-ba.de/themen/arzneimittel/arzneimittel-richtlinie-anlagen/nutzenbewertung-35a/zusatznutzen/
10. HAS. *Transparency Committee*.
    https://www.has-sante.fr/jcms/c_412155/en/transparency-committee
11. Canada's Drug Agency. *Reimbursement reviews*.
    https://www.cda-amc.ca/
12. Australian Government Department of Health. *PBAC*.
    https://www.pbs.gov.au/info/industry/listing/elements/pbac-meetings
13. Neumann PJ, et al. *Cost-Effectiveness in Health and Medicine*, 2nd edition. Oxford University
    Press, 2016.
14. Drummond MF, et al. *Methods for the Economic Evaluation of Health Care Programmes*, 4th
    edition. Oxford University Press, 2015.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_clinical-data-management.md`](03_clinical-data-management.md)
- Next topic: [`05_market-access.md`](05_market-access.md)
- Global regulation: [`06_global-regulation.md`](06_global-regulation.md)
- Module survey: [`../../course/25_trial_operations_hta_global_regulatory.md`](../../course/25_trial_operations_hta_global_regulatory.md)
- Labs: [`../../labs/`](../../labs/)
