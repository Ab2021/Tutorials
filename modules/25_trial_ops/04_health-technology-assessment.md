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

### 5.19 What a defensible dossier contains

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
