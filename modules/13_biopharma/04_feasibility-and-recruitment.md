# Feasibility and Recruitment: From Eligible Patients to a Credible Timeline

> Module 13 — Biopharma and Clinical Development
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 08 (EHR and claims); topics 01 and 03
> Version 2026.3

## 1. Why this topic exists

Every clinical trial is a bet that enough eligible patients can be found, consented, and retained
within a window that the programme can afford. The bet is usually made with less evidence than the
scientific hypothesis it is designed to test. A protocol is written with eligibility criteria that
no dataset can reproduce, a sample size is divided by a site count, and the resulting timeline is
presented as a plan. When the timeline slips, the programme loses money, competitive position, and
sometimes the trial itself.

The cost of a wrong feasibility assumption is not linear. A trial that runs six months late pays
for additional site costs, additional central laboratory and management costs, and additional
internal team time, and it delays the readout that the next financing or partnership depends on. If
the delay crosses a patent or exclusivity milestone, the loss can exceed the entire development
budget. Feasibility is therefore a financial model with a clinical input, not an operational
detail.

The data problem is specific. Eligibility criteria are written in clinical language ("adequate
organ function", "no prior therapy in the advanced setting", "measurable disease by RECIST") and
must be translated into computable rules before any dataset can count patients. The translation is
lossy: claims data have diagnosis codes but not laboratory values; EHR data have laboratory values
but not consistently; registries have the disease but not the comorbidity detail. A feasibility
estimate is therefore an estimate with a bias, and the bias has a direction that can often be
determined.

This topic teaches the translation from criteria to queries, the modelling of site activation and
accrual as distributions rather than points, the arithmetic of screen failure and dropout, and the
decision rules for contingency. It follows the biomarker topic, because biomarker-defined
populations are the hardest feasibility problem, and precedes the standards topic, because the
feasibility dataset and the trial dataset must use compatible definitions.

### 1.1 The failure pattern

1. A Phase III protocol requires 800 randomised patients across 60 sites.
2. The feasibility model assumes 60 sites active from month 1 at 0.55 patients per site per month.
3. 800 / (60 × 0.55) = 24.2 months, which matches the agreed 24-month enrolment window.
4. Sites activate over 12 months rather than at month 1, so the true deterministic completion is
   month 30.
5. The model ignores 30% screen failure, so sites must screen 1,143 patients, not 800, and the site
   workload is understated by 43%.
6. The protocol requires 800 evaluable patients, and with 12% dropout the randomised target should
   be 909, not 800.
7. Enrolment completes at month 34. The trial reads out 10 months late, and the partner milestone
   is missed.

Every step after step 3 is arithmetic that could have been done before the protocol was final. The
model was not wrong about the biology; it was wrong about the shape of the accrual curve and the
denominator.

### 1.2 Who is blocked when feasibility is wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Clinical operations | Credibility and budget | Cannot change the protocol's eligibility criteria |
| Programme team | The readout date and the milestone | Depends on sites it does not control |
| Sites | Staff time on screens that fail | Receive criteria they did not write |
| Patients | A place in a trial that closes late | Have no visibility of the plan |
| Biostatistics | Power, if enrolment selects a different population | Receives the enrolled sample, not the intended one |
| Finance | Capital tied up for longer | Sees a single timeline, not a distribution |
| Partner or investor | A milestone payment and a valuation date | Relies on the sponsor's forecast |

### 1.3 The industry question

> How many eligible patients can actually be enrolled, at which sites, by when, and with what
> probability?

The question has four parts because a feasibility estimate without a probability is a wish. "We can
enrol 800 patients in 24 months" is not a forecast; "there is a 20% chance of enrolling 800
randomised patients within 24 months, with a median of 31 months" is.

### 1.4 The three numbers a feasibility review must produce

A review that produces one number will be argued about. A review that produces three can be decided:

1. **The median completion month.** What is most likely to happen.
2. **The probability of meeting the committed date.** What the organisation is actually promising.
3. **The screening burden.** What the sites are actually being asked to do.

The three are related but not interchangeable. A plan can have an acceptable median and an
unacceptable probability, or an acceptable probability produced by asking sites to screen three
patients for every one randomised. The decision depends on which of the three the organisation is
willing to accept.

## 2. Learning outcomes

By the end of this topic you can:

1. **Translate** protocol eligibility criteria into computable rules and state which rules cannot be
   computed from which data source.
2. **Build** a patient funnel from a source population to randomised patients, with screen failure
   and dropout applied at the correct stage.
3. **Model** site activation and per-site accrual as distributions, and produce a completion-time
   distribution rather than a point estimate.
4. **Compute** the screening burden implied by a randomised target and a screen-failure rate.
5. **Compare** contingency options (add sites, extend the window, amend eligibility, add countries)
   on probability of success, cost, and evidence risk.
6. **Diagnose** a feasibility model that hides activation delay inside the accrual rate or omits the
   evaluable-population denominator.
7. **Design** a monitoring and reforecast cadence with pre-specified triggers for action.

## 3. The core mental model

### 3.1 The patient funnel

```text
 Source population (region, diagnosis, line of therapy)
        |  prevalence / incidence, data coverage
        v
 Candidate patients in a database
        |  computable eligibility (diagnosis, age, prior therapy, labs)
        v
 Potentially eligible
        |  site catchment, referral pattern, willingness, competing trials
        v
 Available to a participating site
        |  consent and screening
        v
 Screened
        |  screen failure (eligibility not met, withdrawal, logistics)
        v
 Randomised
        |  discontinuation before the primary endpoint
        v
 Evaluable for the primary analysis
```

Each arrow is a proportion with a source and an uncertainty. The product of the proportions times
the source population is the expected enrolment. A feasibility model that skips an arrow
overestimates the result, and the omitted arrow is usually "available to a participating site" or
"screen failure".

### 3.2 Accrual is a curve, not a rate

```text
 Patients
  800 |                                              ****
      |                                        *****
  600 |                                  *****
      |                            *****
  400 |                      *****
      |                *****
  200 |          *****
      |     ****
    0 +----|----|----|----|----|----|----|----|----|----> months
          3    6    9   12   15   18   21   24   27   30
```

The curve has three phases: activation (few sites, low accrual), ramp (sites maturing, referral
pathways forming), and steady state (all sites active). A model that uses the steady-state rate
from month 1 is wrong by the area between the curves, which in this example is about six months.

### 3.3 Three denominators that are routinely confused

| Denominator | Definition | Used for | Consequence of confusion |
|---|---|---|---|
| Screened | Patients who consent and begin screening | Site workload and screen-failure planning | Understates site burden |
| Randomised | Patients allocated to treatment | Enrolment target and disposition | Overstates evaluable population |
| Evaluable | Patients with the primary endpoint measured | Power calculation | Understates required enrolment |

A protocol that says "enrol 800 patients" without specifying which denominator is ambiguous, and
the ambiguity will be resolved differently by the sponsor, the sites, and the statistician.

### 3.4 Feasibility is a distribution

The completion month is a random variable determined by site activation, per-site accrual,
screen failure, and dropout. Each of those has a distribution. The correct output of a feasibility
model is therefore a distribution of completion dates, summarised as a median, an 80% interval, and
the probability of meeting a specific deadline.

A deterministic model produces a single date that is almost always optimistic, because it uses
central values for inputs that enter multiplicatively. The product of central values is not the
central value of the product when the inputs are uncertain.

### 3.5 Where the leverage is

| Lever | Effect on completion time | Cost | Evidence risk |
|---|---|---|---|
| Add sites | Moderate; adds steady-state capacity but also activation lag | Site payments, management | Lower if new sites are inexperienced |
| Activate faster | Large in short trials; small in long ones | Start-up investment | Low |
| Broaden eligibility | Large; increases screen pass rate | Possible label and safety risk | Higher |
| Add countries | Moderate; adds regulatory lag | Start-up and oversight | Population differences |
| Increase per-site rate | Large but hard to move | Site support, referral pathways | Low |
| Extend the window | Certain but costly | Delay cost | None |

The table is the core of the contingency decision. Adding sites is the most common response and
often the least efficient, because new sites contribute little in the first three months and the
trial's remaining duration may be too short to amortise the activation cost.

### 3.6 The recruitment equation

The model in Section 6 is a simulation, but its logic can be written as one equation:

```text
Randomised by month m = sum over months of [ active sites at that month
                                           x accrual per active site
                                           x site maturity ]
```

and then:

```text
Screens required = randomised target / (1 - screen failure)
Randomised required = evaluable target / (1 - dropout)
Completion month = first m at which cumulative randomised >= randomised required
```

Every feasibility dispute is a dispute about one of the terms in these equations. Naming the term
turns an argument about dates into an argument about evidence, which is a better argument.

### 3.7 The funnel as a product: a check on any site-rate model

A top-down funnel and a bottom-up site model should agree. If they do not, at least one is wrong.

```text
Regional population                              2,400,000
Prevalence of the indication                         0.4%   -> 9,600
Diagnosed                                          35%     -> 3,360
Meet the main eligibility criteria                 60%     -> 2,016
Reach a participating site                         25%     ->   504
Consent                                            70%     ->   353
Pass screening                                     65%     ->   229 randomised
```

With 20 sites over 18 months, 229 patients is 229/(20 × 18) = 0.64 patients per site per month. If
the site model assumes 1.2, the two models disagree by a factor of two and the assumptions behind
the difference must be resolved before either is used.

The funnel is sensitive to the smallest proportion. A change from 25% to 15% in the "reach a
participating site" step reduces the total by 40%, which is larger than most plausible changes in
the clinical eligibility criteria.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Feasibility | Assessment of whether a trial can be conducted as designed in a given setting | Treated as a site count | Assessment |
| Eligibility criteria | Rules defining who may enter the trial | Written without operational definitions | Protocol section |
| Computable phenotype | An algorithm that identifies a clinical condition in a dataset | Assumed to be validated | Algorithm |
| Source population | Patients with the condition in the geography of interest | Confused with the trial-eligible population | Count |
| Screen failure | A screened patient who does not meet eligibility or withdraws before randomisation | Counted as enrolled | Proportion |
| Screen-failure rate | Screen failures divided by screened patients | Applied to randomised patients | Proportion |
| Accrual rate | Patients enrolled per site per month | Quoted without the site maturity assumption | Patients/site/month |
| Activation | The point at which a site can randomise its first patient | Confused with contract signature | Date |
| Activation lag | Time from site selection to activation | Omitted from timelines | Weeks or months |
| Site maturity | The ramp in accrual after activation | Assumed instantaneous | Proportion of steady state |
| Enrolment target | Number of patients to be randomised | Confused with evaluable | Count |
| Evaluable population | Patients with the primary endpoint measured | Assumed equal to randomised | Count |
| Attrition | Loss of patients between randomisation and analysis | Ignored in power calculations | Proportion |
| Retention | Proportion of randomised patients remaining in the study | Confused with compliance | Proportion |
| Competing trial | Another study recruiting the same patients | Ignored in site selection | Count |
| Referral network | The pathway by which patients reach a site | Assumed to exist | Description |
| Catchment area | Geographic area from which a site draws patients | Assumed to match the city | Geography |
| Pre-screening | Identifying potentially eligible patients before consent | Confused with screening | Activity |
| Feeder pattern | The distribution of a site's referrals by source | Unexamined | Description |
| Diversity action plan (DAP) | A plan for enrolment of underrepresented populations, required by statute for certain studies | Treated as optional | Regulatory document |
| Decentralised trial (DCT) | Trial with remote or local elements replacing site visits | Assumed to widen eligibility automatically | Design |
| Site activation curve | Cumulative activated sites over time | Assumed linear | Curve |
| Enrolment curve | Cumulative randomised patients over time | Assumed linear | Curve |
| Confidence interval | Interval estimate for a rate or count | Quoted as a range without a method | Interval |
| Forecast | A distribution over future enrolment | Presented as a single number | Distribution |
| Trigger | A pre-specified deviation from plan that requires action | Set after the problem appears | Threshold |
| Rescue plan | Pre-approved contingency for slow enrolment | Written during the crisis | Plan |
| Screen-to-randomise ratio | Screened patients per randomised patient | Confused with screen-failure rate | Ratio |
| Recruitment | The process of identifying and enrolling patients | Used interchangeably with enrolment | Activity |

### 4.1 The difference between recruitment and enrolment

Recruitment is the activity of finding and approaching patients. Enrolment is the event of
randomisation. A site can recruit actively and enrol nobody, because the patients it finds are
ineligible or decline. Feasibility models that use "recruitment rate" without defining the
denominator mix the two, and the resulting forecast cannot be validated against site data.

## 5. The mechanics, step by step

This procedure produces a feasibility model for a Phase II or III trial. It assumes access to at
least one patient-level dataset (claims, EHR, or registry) and to historical site performance. Where
those are unavailable, the procedure still applies with expert estimates, but the uncertainty
widens and should be reported.

### Step 1. Fix the target and its denominator

**Input:** the protocol's sample size section and the statistical analysis plan.

**Output:** a statement of the randomised target, the evaluable target, the expected dropout, and
the resulting screening target.

**Rationale:** The power calculation is performed on the evaluable population. If 12% of randomised
patients discontinue before the primary endpoint, the randomised target must be 800/0.88 = 909. If
30% of screened patients fail, the screening target is 909/0.70 = 1,299. A plan that quotes 800
screens has understated site workload by 62%.

**Branch point:** if the endpoint is event-driven, the target is an event count, not a patient
count. The patient count follows from the event rate and the follow-up duration, and the feasibility
model must forecast events, not just enrolment.

### Step 2. Translate eligibility into computable rules

**Input:** the inclusion and exclusion criteria.

**Output:** a rule table with the data field, the code set, the time window, and the computability
status (computable, partially computable, not computable).

**Rationale:** A criterion that cannot be computed cannot be counted. "Adequate organ function"
requires laboratory values and thresholds; "no prior advanced therapy" requires a treatment history
with a line definition; "measurable disease" requires imaging assessment data that claims do not
contain. The rule table makes the loss explicit.

**Edge case:** negative criteria that depend on clinical judgement ("life expectancy of at least 12
weeks") cannot be computed from any administrative dataset. They should be treated as a screen-
failure allowance, not a query.

### Step 3. Estimate the source population with a funnel

**Input:** the rule table and the available datasets.

**Output:** counts at each funnel stage, with the data source and the coverage limitation for each.

**Rationale:** The funnel separates the epidemiological question (how many patients have the
condition?) from the data question (how many are visible in this dataset?) from the operational
question (how many can reach a site?). Conflating them produces a single number with no way to
audit its error.

### Step 4. Select sites on evidence, not on relationships

**Input:** candidate site list, historical performance, catchment data, and the competing trial
landscape.

**Output:** a ranked site list with expected accrual per site, an activation estimate, and a
justification.

**Rationale:** Site selection is the highest-leverage feasibility decision. A small number of sites
produce most of the enrolment, and a substantial minority produce none. Historical performance in
the same indication, with the same protocol complexity, is the best available predictor. A site's
reputation or its investigator's publication record is a weaker predictor.

**Branch point — site categories:**

| Category | Evidence | Expected accrual | Risk |
|---|---|---|---|
| Proven performer in the indication | Prior trial data, same population | At or above the median | Complacency, competing studies |
| Experienced site, new indication | Prior trial data, different population | Near the median | Learning curve |
| New site, strong catchment | Epidemiology and referral data | Below the median initially | Activation delay, inexperience |
| Strategic site (access, diversity, region) | Access or population rationale | Uncertain | Underperformance tolerated for non-enrolment reasons |

### Step 5. Model activation explicitly

**Input:** site contracts, regulatory timelines, and historical activation data.

**Output:** an activation month for each site, with a distribution.

**Rationale:** Activation is the largest source of delay in the first six months. Industry
benchmark data commonly cited report average activation times of around seven to eight months from
site identification, with substantial variation between academic centres and community practices.
A model that activates all sites at month 1 has removed the main constraint.

### Step 6. Model per-site accrual as a distribution

**Input:** historical site performance and the eligible population per catchment.

**Output:** a distribution of patients per site per month, with site-level heterogeneity.

**Rationale:** Published Phase III benchmarks put average accrual at below one patient per site per
month, with large variation by therapeutic area and trial size. Using the average as if it applied
to every site ignores the concentration of enrolment in a minority of sites and understates the
risk that a large fraction of sites enrol nothing.

### Step 7. Apply screen failure and dropout at the right stage

**Input:** the rule table, historical screen-failure data, and the protocol's discontinuation rules.

**Output:** a screen-to-randomise ratio and an evaluable fraction.

**Rationale:** Screen failure applies before randomisation and affects site workload. Dropout
applies after randomisation and affects power. Applying both to the same denominator, or applying
either at the wrong stage, produces a target that is wrong in a direction that is hard to detect
later.

### Step 8. Simulate the completion time

**Input:** the activation model, the accrual distribution, and the targets.

**Output:** a distribution of completion months, a median, an 80% interval, and the probability of
meeting the deadline.

**Rationale:** The simulation propagates uncertainty through the multiplicative model. Its main
value is not the median but the tail: a plan with a 31-month median and a 44-month 90th percentile
is a different risk from one with a 31-month median and a 33-month 90th percentile.

### Step 9. Identify the binding constraint

**Input:** the simulation and the sensitivity analysis.

**Output:** the input that explains most of the variance in completion time, and the levers that
move it.

**Rationale:** A feasibility discussion that does not identify the binding constraint produces a
list of actions rather than a decision. If accrual rate explains 80% of the variance, adding sites
addresses the wrong constraint; if activation explains it, start-up investment is the lever.

### Step 10. Price the contingency options

**Input:** the levers and their costs.

**Output:** a comparison of options on probability of meeting the deadline, cost, and evidence risk.

**Rationale:** Each option changes the trial as well as the timeline. Broadening eligibility changes
the population and may weaken the label. Adding countries adds population heterogeneity. Adding
sites adds inexperienced investigators. The decision must price those effects, not only the delay.

### Step 11. Set triggers and a reforecast cadence

**Input:** the plan and the simulation.

**Output:** pre-specified triggers, a monthly reforecast, and an escalation path.

**Rationale:** A feasibility model is a forecast, and forecasts are wrong. The question is whether
the organisation learns early enough to act. A trigger such as "if cumulative enrolment at month 9
is below 60% of plan, activate the rescue option" converts a forecast into a control system.

### Step 12. Document and govern

**Input:** the model, its inputs, and its outputs.

**Output:** a feasibility report with assumptions, sources, uncertainty, and the decision record.

**Rationale:** The report is what allows a later reforecast to be compared with the original. It
also allocates accountability: if the model assumed 0.55 patients per site per month and the sites
deliver 0.35, the report shows whether the assumption was evidence-based or aspirational.

### 5.1 Decision table for a slow-enrolment rescue

| Situation at trigger | Likely cause | Recommended action | Condition |
|---|---|---|---|
| Enrolment below plan, activation on plan | Accrual per site below assumption | Add sites; investigate referral pathways | New sites can activate before the window closes |
| Enrolment below plan, activation behind | Start-up delay | Invest in start-up; add already-active sites from another study | Regulatory timelines allow |
| Screen failure above assumption | Eligibility too narrow or criteria misinterpreted | Clarify criteria; consider a protocol amendment | Amendment does not change the target population |
| Enrolment on plan, evaluable below plan | Dropout above assumption | Retention programme; re-power | Endpoint timing allows |
| Enrolment on plan in aggregate, concentrated in few sites | Site heterogeneity | Reallocate monitoring and support; close non-performers | Contract allows |
| Enrolment above plan | Demand underestimated | Slow accrual to protect data quality | Over-enrolment rule exists |
| No feasible option within the window | Design infeasible | Stop or redesign the trial | Decision at the next governance gate |

### 5.2 Edge cases that break the model

**Rare disease.** The source population may be a few hundred patients globally. Accrual is limited
by diagnosis and travel, not by site rate. Models based on per-site rates are inappropriate;
patient-identification networks and registry-based recruitment are the relevant methods.

**Competing trials.** A site participating in three trials in the same indication does not have
three times the patients. The competing-trial landscape must be subtracted from the eligible pool,
and the subtraction is approximate because trial registries lag.

**Seasonality and referral cycles.** Some indications have referral patterns that vary by season or
by academic calendar. A model fitted on annual data will smooth a pattern that matters for a 24-month
trial.

**Diversity goals.** A diversity action plan may require enrolment goals by race, ethnicity, sex,
and age. These goals change site selection and may extend the timeline. Treating them as a reporting
requirement rather than a design input produces a plan that cannot meet them.

**Decentralised elements.** Remote consent and local laboratory collection can widen the catchment,
but they also change the screen-failure profile and the data quality risk. The feasibility model
should model them as a change in the eligible pool, not as a cost saving.

**Amendments.** A protocol amendment that changes eligibility resets the screen-failure assumption
and often pauses enrolment. The timeline model should include the pause.

### 5.4 From criterion to query: a worked translation

The table below shows how four criteria translate, and where the translation fails.

| Criterion (protocol language) | Computable rule | Data source | Failure mode |
|---|---|---|---|
| Adults with advanced disease | Age ≥ 18 at consent; diagnosis code plus stage code | Claims or EHR | Stage is often missing or coded inconsistently |
| Failure of at least one prior line | ≥ 1 systemic therapy claim after diagnosis and before index date | Claims | Line definitions differ; oral therapies may be missing |
| Adequate renal function | eGFR ≥ 60 within 28 days before index | EHR laboratory | Labs missing for community patients |
| No prior biologic therapy | Absence of biologic drug codes in the prior 12 months | Claims | Cash-pay or out-of-network use invisible |
| Measurable disease by RECIST | Imaging assessment with target lesion measurement | EHR or imaging vendor | Not present in administrative data |
| Life expectancy ≥ 12 weeks | None | None | Clinical judgement; must be a screen-failure allowance |

The last row is the important one. A criterion that cannot be computed is not a reason to omit it
from the model; it is a reason to add an explicit allowance for screen failure caused by clinical
judgement. The allowance should be stated and sourced, not buried in a single 30% figure.

### 5.5 What site-level data actually predict accrual

Site selection models often use variables that are available rather than variables that predict.
The evidence on what predicts accrual is modest, but a consistent pattern appears in operational
data:

| Variable | Predictive value | Why |
|---|---|---|
| Prior accrual in the same indication | High | Captures capability, catchment, and motivation |
| Prior accrual with the same sponsor | Moderate to high | Captures familiarity with systems and expectations |
| Number of active competing trials at the site | Moderate | Direct competition for the same patients |
| Catchment size and referral pattern | Moderate | Defines the supply of patients |
| Investigator publication record | Low | Measures academic output, not recruitment |
| Site type (academic versus community) | Context-dependent | Academic sites have more complex patients; community sites may accrue faster |

A model that ranks sites by publication record will over-select academic centres with slow accrual.
A model that ranks by prior accrual in the indication will select sites that have demonstrated the
behaviour the trial needs.

### 5.6 Activation is a process with owners

Activation is not a single date; it is the completion of a sequence, each step with an owner and a
failure mode.

| Step | Owner | Typical duration | Common delay |
|---|---|---|---|
| Site selection and feasibility questionnaire | Sponsor | 2-4 weeks | Sites return questionnaires late |
| Confidentiality agreement | Legal | 2-6 weeks | Template negotiations |
| Contract and budget | Legal and finance | 4-12 weeks | Budget negotiation, payment terms |
| Regulatory submission and approval | Site and sponsor | 4-16 weeks | Country-specific timelines |
| Ethics approval | Site | 4-12 weeks | Committee scheduling |
| Site initiation visit | Sponsor | 2-4 weeks | Scheduling |
| Pharmacy and laboratory set-up | Site | 2-8 weeks | Equipment, training |
| First patient ready | Site | 1-4 weeks | Referral pathway not established |

The sum is rarely the calendar duration, because steps run in parallel. The critical path is usually
contract and regulatory approval, and the largest variance is in contract negotiation. A model that
uses a single average activation time hides which step is binding and therefore which intervention
would help.

### 5.7 Low-prevalence and rare-disease designs

When the eligible population is small, the per-site accrual model stops being useful. The
constraints change:

- **Diagnosis is the bottleneck.** Patients may be undiagnosed, misdiagnosed, or diagnosed at a
  centre that does not participate. Registries and patient organisations become recruitment
  infrastructure.
- **Travel is the constraint.** A patient may be willing but unable to reach the site monthly. Remote
  or local elements change the eligible pool more than they change the cost.
- **Competing trials exhaust the pool.** In a disease with 2,000 prevalent patients, three trials
  recruiting simultaneously are competing for the same people.
- **Screen failure is high and variable.** Diagnostic confirmation often occurs at screening, so the
  screen-failure rate can exceed 40%.

The appropriate model is a patient-identification funnel with a named source (registry, network,
referral map), not a site-rate model. The output should be a list of identified or identifiable
patients, with consent and eligibility as separate proportions.

### 5.8 Diversity goals as a feasibility input

Where a diversity action plan applies, enrolment goals are disaggregated by race, ethnicity, sex, and
age group. Those goals change the feasibility problem in two ways. First, the site mix must include
sites that serve the populations named in the plan, which may mean fewer experienced sites and more
start-up support. Second, the timeline may lengthen, because the eligible pool for each group is
smaller than the overall pool.

A feasibility model that treats the diversity plan as a reporting obligation will produce a plan that
meets the enrolment target and misses the goals. The goals belong in the site-selection criteria and
in the accrual assumptions, with the same evidence standard as the overall rate.

Where the plan requires a waiver, the waiver rationale should state what makes the study infeasible
for a group, not merely that recruitment would be difficult. The distinction matters because the
first is a design problem that might be fixed, and the second is a choice.


### 5.3 What the feasibility report must state

1. The target, with its denominator (randomised or evaluable).
2. The screening target and the screen-to-randomise ratio.
3. The source population, the dataset, and the coverage limitation.
4. The site list with expected accrual and activation month.
5. The accrual distribution, including site-level heterogeneity.
6. The completion-time distribution, with median, 80% interval, and probability of meeting the
   deadline.
7. The binding constraint and the sensitivity analysis.
8. The contingency options with costs and evidence risks.
9. The triggers and the reforecast cadence.
10. The assumptions that would change the decision.

## 6. Worked example

> **Note:** The trial and site data in this example are fictional and constructed for teaching. The
> benchmark accrual figures cited are published and are referenced in Section 16.

### 6.1 The plan

A sponsor plans a Phase III trial in second-line inflammatory disease:

- Randomised target: 800 patients, 1:1.
- 60 sites, activating at 5 per month from month 1.
- Assumed accrual: 0.55 patients per site per month.
- Screen failure: assumed 0% in the plan.
- Dropout before the primary endpoint: assumed 0% in the plan.
- External deadline: enrolment complete within 24 months.

The plan's arithmetic is 800 / (60 × 0.55) = 24.2 months, which matches the deadline. That match is
the first warning sign: the assumption was almost certainly chosen to produce it.

### 6.2 Step 1: the denominators

```text
Randomised target                         800
Dropout before primary endpoint           12%  -> evaluable = 800 x 0.88 = 704
Required randomised for 800 evaluable     800 / 0.88 = 909
Screen failure                            30%  -> screens = 909 / 0.70 = 1,299
Screen-to-randomise ratio                 1,299 / 909 = 1.43
```

If the programme intends 800 evaluable patients, the site workload is 1,299 screens, not 800. That
is 62% more screening than the plan implies.

### 6.3 Step 8: the deterministic forecast with activation

With activation at 5 sites per month, the active-site count at month m is 5m until month 12, then
60. Cumulative enrolment is 0.55 × 5 × (1 + 2 + ... + 12) = 0.55 × 5 × 78 = 214.5 by month 12.

Remaining to 800: 585.5 patients at 33 per month = 17.7 months. Completion in month 30. The
24-month plan is six months short before any uncertainty is considered.

### 6.4 Step 8b: the simulated forecast

The simulation draws a trial-level accrual rate from a lognormal distribution with median 0.55 and
coefficient of variation 0.35, site-level rates from a gamma distribution with coefficient of
variation 0.60, and monthly counts from a Poisson distribution. Sites activate at 5 per month and
ramp to 60% of steady state in their first month.

| Scenario | Median completion | P(within 24 months) | P(within 30 months) | 90th percentile |
|---|---:|---:|---:|---:|
| 60 sites, 0.55 patients/site/month | 31 months | 0.20 | 0.49 | 44 months |
| 80 sites, 0.55 patients/site/month | 27 months | 0.36 | 0.71 | 37 months |
| 60 sites, 0.84 patients/site/month | 22 months | 0.64 | 0.88 | 31 months |

The base plan has a 20% chance of meeting the deadline. The deterministic forecast of 30 months
understates the median by one month and the 90th percentile by 14 months.

### 6.5 Two diagnostics that reveal a broken forecast

**Diagnostic 1 — the plan equals the deadline.** Compute the implied accrual rate from the deadline
and the site count: 800 / (60 × 24) = 0.56 patients per site per month. If the assumed rate is
within a few percent of the rate implied by the deadline, the assumption was reverse-engineered. The
diagnostic is to compare the assumed rate with the programme's historical rate and with published
benchmarks; a rate chosen to fit the deadline is not evidence.

**Diagnostic 2 — the implied per-site rate rises over time.** Reconstruct the plan's implied rate as
cumulative enrolment divided by cumulative active-site months. In a model with activation, the
implied rate rises through the ramp and then flattens. If the plan shows a flat rate from month 1,
activation has been absorbed into the accrual assumption, and the timeline is too short. In this
example, the plan's flat 0.55 implies 33 patients per month from month 1, which no activation curve
can deliver.

A fourth check is the screen-to-randomise ratio. If the plan does not state it, the site workload is
undefined, and site contracts, budgets, and monitoring visits will be sized on the wrong number.

A third check is the screen-to-randomise ratio. If the plan does not state it, the site workload is
undefined, and site contracts, budgets, and monitoring visits will be sized on the wrong number.

### 6.6 Step 10: the contingency decision

The programme has three options:

- **Option A:** proceed with 60 sites and renegotiate the deadline to 30 months. Cost: six months of
  delay at USD 1.2 million per month, or USD 7.2 million, plus the risk of a later readout.
- **Option B:** add 20 sites at USD 2.4 million, raising the probability of meeting 24 months from
  0.20 to 0.36. Still more likely to miss than to meet.
- **Option C:** reduce the target to 600 randomised patients and re-power on a larger effect. Reduces
  the evidence and may not support the label.

Option B costs less than Option A but does not solve the problem: a 36% chance of meeting the
deadline is not a plan. Option C changes the scientific claim. The rational decision is a
combination: renegotiate to 30 months, add sites only if they can activate by month 6, and keep the
target.

### 6.7 Vary one assumption: the accrual rate

The programme's historical rate of 0.55 is below the published Phase III average of approximately
0.84 patients per site per month. If the trial is more like the benchmark than like the sponsor's
history, the base plan completes at a median of 22 months with a 64% chance of meeting the deadline.

The decision changes from "renegotiate the deadline" to "proceed with the 24-month plan and monitor
monthly". One assumption, supported by a published benchmark but contradicted by the sponsor's own
data, moves the recommendation across the threshold.

The disciplined response is not to pick the benchmark because it is more convenient. It is to ask
why the sponsor's historical rate is lower: different indication, different sites, more complex
protocol, or poor site support. If the reason is specific and fixable, the benchmark may be
achievable. If the reason is structural, the historical rate is the better forecast and the deadline
is not.

### 6.8 What the example establishes

1. The denominator (screened, randomised, evaluable) must be fixed before any forecast.
2. Activation is a first-order determinant of completion time, not a detail.
3. A deterministic forecast hides the tail that determines risk.
4. Adding sites is a weak lever when activation is slow and the window is short.
5. The accrual assumption dominates the decision, and choosing it requires a reason.

### 6.10 Sensitivity of the deterministic forecast to accrual

| Accrual (patients/site/month) | Cumulative at month 12 | Monthly at steady state | Completion month |
|---|---:|---:|---:|
| 0.45 | 175.5 | 27.0 | 36 |
| 0.55 | 214.5 | 33.0 | 30 |
| 0.65 | 253.5 | 39.0 | 26 |
| 0.84 | 327.6 | 50.4 | 22 |

A 0.10 change in accrual moves the deterministic completion by four to six months, and the effect is
larger at lower rates because the fixed activation period is a larger share of a longer trial. This
is why the accrual assumption, not the site count, is the variable to argue about.

### 6.11 What the model does not tell you

The simulation forecasts the number of randomised patients. It does not forecast whether those
patients resemble the intended label population, whether the sites will still be recruiting in month
18, or whether a competitor will open in the same indication and take half the pool. Those risks are
outside the model and belong in the decision record as qualitative threats with named monitoring.

The model also assumes that screen failure and accrual are independent. In practice they are
correlated: sites that enrol few patients often screen more intensively and fail more of them. A
model that treats them as independent will understate the variance of the completion time, which is
why the simulated 90th percentile in this example is a better guide to risk than the median.

Finally, the model does not price the cost of a delayed readout. That cost belongs in the programme
valuation (topic 01), where the launch date moves with the completion date. Feasibility and value are
one calculation, not two.


### 6.9 The monitoring plan that follows

The forecast becomes a control system through four pre-specified triggers:

| Trigger | Threshold | Action | Owner |
|---|---|---|---|
| Cumulative enrolment at month 6 | Below 60% of plan | Review activation and referral pathways | Clinical operations |
| Cumulative enrolment at month 9 | Below 60% of plan | Activate rescue option (add sites or amend) | Programme committee |
| Screen failure at month 6 | Above 40% | Audit eligibility interpretation across sites | Medical monitor |
| Dropout at month 12 | Above 15% | Retention review; re-power assessment | Biostatistics |

The thresholds are set from the simulation, not from intuition: the 60% trigger corresponds to the
lower bound of the 80% interval in the base scenario. A trigger that fires only when the trial is
already lost is not a control; it is a post-mortem.

## 7. Code, query, or calculation

The simulation below reproduces the three scenarios in Section 6. It uses NumPy for the random
draws and is seeded so the output is reproducible.

```python
import numpy as np

RNG = np.random.default_rng(20261025)

def simulate(target=800, n_sites=60, act_per_month=5, median_ppsm=0.55,
             cv_trial=0.35, cv_site=0.60, horizon=60, nsim=20000):
    """Enrolment completion month under trial- and site-level heterogeneity."""
    sigma_trial = np.sqrt(np.log(1 + cv_trial ** 2))
    shape_site = 1 / cv_site ** 2
    months = np.empty(nsim, dtype=int)
    for k in range(nsim):
        ppsm = median_ppsm * np.exp(RNG.normal(0, sigma_trial))
        site_rate = RNG.gamma(shape_site, ppsm / shape_site, size=n_sites)
        enrolled, m = 0.0, 0
        while enrolled < target and m < horizon:
            m += 1
            active = min(n_sites, act_per_month * m)
            new = min(act_per_month, max(0, active - act_per_month * (m - 1)))
            mature = active - new
            lam = site_rate[:mature].sum() + 0.6 * site_rate[mature:mature + new].sum()
            enrolled += RNG.poisson(lam)
        months[k] = m
    return months

for label, kw in [("60 sites, 0.55", dict(n_sites=60, median_ppsm=0.55)),
                  ("80 sites, 0.55", dict(n_sites=80, median_ppsm=0.55)),
                  ("60 sites, 0.84", dict(n_sites=60, median_ppsm=0.84))]:
    m = simulate(**kw)
    print(f"{label}: median {np.median(m):.0f}  P(<=24) {np.mean(m <= 24):.2f}  "
          f"P(<=30) {np.mean(m <= 30):.2f}  P90 {np.percentile(m, 90):.0f}")
```

Expected output (Python 3.12, NumPy 2.4, executed):

```text
60 sites, 0.55: median 31  P(<=24) 0.20  P(<=30) 0.49  P90 44
80 sites, 0.55: median 27  P(<=24) 0.36  P(<=30) 0.71  P90 37
60 sites, 0.84: median 22  P(<=24) 0.64  P(<=30) 0.88  P90 31
```

The deterministic calculation in Section 6.3 can be checked separately:

```python
def deterministic(target=800, n_sites=60, act=5, ppsm=0.55):
    enrolled, m = 0.0, 0
    while enrolled < target:
        m += 1
        enrolled += min(n_sites, act * m) * ppsm
    return m, enrolled

for sites, rate in [(60, 0.55), (80, 0.55), (60, 0.84)]:
    m, e = deterministic(n_sites=sites, ppsm=rate)
    print(sites, rate, (m, round(e, 1)))
```

```text
60 0.55 (30, 808.5)
80 0.55 (26, 814.0)
60 0.84 (22, 831.6)
```

Assumptions: monthly counts are Poisson conditional on the site rate; the trial-level rate is
lognormal; site rates are gamma; new sites run at 60% of steady state in their first month; sites
activate at a constant rate; screen failure and dropout affect the targets, not the accrual
simulation. The last assumption is a simplification: in reality, screening consumes site capacity,
so a high screen-failure rate reduces randomised accrual. A more realistic model would simulate
screens and apply the screen-failure probability to obtain randomised patients.

The model is deliberately simple enough to be audited. A 20,000-run simulation takes a few seconds
and produces a distribution that a committee can interrogate.

## 8. Data, standards, and regulatory context

Feasibility is mostly an internal activity, but three regulatory instruments shape it: the rules on
trial registration and transparency, the rules on the populations that must be enrolled, and the
rules on how remote and local trial elements may be conducted.

| Instrument | What it requires | Bindingness | Effect on feasibility |
|---|---|---|---|
| FDAAA 2007, Section 801; 42 CFR Part 11 | Registration of applicable clinical trials and results reporting | US law | Trial start date and site list become public; competing-trial analysis uses registries |
| EU Clinical Trials Regulation 536/2014 and CTIS | Authorisation and public registration of EU trials | EU law | EU site activation includes a central authorisation step |
| FDORA 2022, FD&C Act 505(z) | Diversity action plans for certain pivotal studies | US statute | Enrolment goals become a design input |
| FDA draft guidance on Diversity Action Plans (June 2024) | Content of a DAP: goals by race, ethnicity, sex, age; rationale; approach | Draft, non-binding; status should be checked | Defines the expected content where a DAP applies |
| ICH E6(R3) (2025) | GCP, including data governance and quality by design | Guideline implemented regionally | Site oversight and data review expectations |
| ICH E8(R1) (2021) | Quality by design, critical-to-quality factors | Guideline | Eligibility criteria should be justified by the question |
| FDA guidance on decentralised trials (2024) | Conduct of trials with remote or local elements | FDA guidance | Can widen catchment; requires data integrity controls |
| Common Rule (45 CFR 46) and institutional review boards | Human subjects protection and consent | US law and institutional policy | Consent logistics and site timelines |
| GDPR | Personal data processing and transfers | EU law | Feasibility datasets must be lawful and minimised |

### 8.1 Requirement versus convention

- **Requirement.** Registering an applicable clinical trial is a legal obligation in the US, and the
  EU requires authorisation through CTIS. A diversity action plan is required by statute for certain
  pivotal studies; the June 2024 guidance describing its content is a draft, so the statutory
  obligation and the guidance are not the same instrument.
- **Regulatory expectation.** Including a feasibility assessment in the trial master file, and
  documenting site selection, is expected by inspectors as part of quality by design under ICH
  E8(R1) and E6(R3), even where no specific document is prescribed.
- **Convention.** The 90-day site activation target published by the US National Cancer Institute,
  the use of a specific accrual benchmark, and the choice of a 60% monitoring trigger are
  conventions. They may be sensible; they are not requirements.

### 8.2 What the registry data can and cannot tell you

ClinicalTrials.gov and the EU CTIS are the best public sources for the competing-trial landscape.
They are incomplete in predictable ways: trials that are not yet registered, trials in countries
with different disclosure rules, and trials registered late. A competing-trial analysis that treats
the registry as complete will overestimate the available patient pool. The analysis should state the
registration lag and the expected number of unregistered trials, or use a range.

Registry data are also the easiest way to validate a feasibility model after the fact. Comparing a
trial's actual enrolment curve with its registered start date and estimated enrolment gives a
benchmark for the next study, albeit a biased one, because trials with poor enrolment are less
likely to report their experience.

### 8.3 Decentralised elements and eligibility

Decentralised trial elements can widen the catchment by removing travel as a barrier. They also
change the screen-failure profile: remote pre-screening may identify more candidates, and local
laboratory collection may introduce variability. The regulatory guidance on decentralised trials
addresses data integrity, oversight, and consent, not the epidemiological question of how many
patients become eligible. That question remains a feasibility model, and the model should treat
remote elements as a change in the eligible pool and in the cost per patient, not as a fixed
efficiency gain.

### 8.4 The competing-trial analysis

A competing-trial analysis answers a narrow question: how many other studies are recruiting the same
patients within the catchment of each proposed site?

| Step | Method | Limitation |
|---|---|---|
| Define the competing set | Same indication, same line, overlapping eligibility | Eligibility is often described loosely in registries |
| Define the geography | Site catchment radius or referral region | Catchment is not published |
| Extract from registries | ClinicalTrials.gov, EU CTIS, national registries | Registration lag and missing trials |
| Weight by phase and size | Larger trials consume more patients | Enrolment estimates are often inaccurate |
| Subtract from the pool | Pool minus competing demand | Double counting across trials |

The output is not a precise number. It is a range that says whether the site is competing with one
other trial or six. A site with six competing studies in the same indication is unlikely to deliver
its historical accrual, regardless of its past performance.

### 8.5 Data protection in feasibility work

Feasibility datasets are built from identifiable or pseudonymised patient data, and the analysis is
usually performed before any consent from those patients. Three controls are standard:

- **Minimisation.** Extract only the fields needed for the eligibility rules, not the full record.
- **Lawful basis.** Under the GDPR, feasibility may rely on legitimate interests or on preparatory
  steps before consent, depending on the jurisdiction and the institution; the basis must be
  documented.
- **Aggregation.** Report counts, not patient lists, unless a patient-identification service with an
  appropriate legal basis is used.

A feasibility team that exports patient-level records to a spreadsheet without these controls has
created a privacy incident that no amount of subsequent analysis justifies.

### 8.6 What an inspector asks about feasibility

Feasibility is not usually inspected as a document. It surfaces through its consequences: a site
that enrolled patients it should not have, a protocol deviation repeated across sites, or a trial
that ran for years beyond its registered completion date. The questions that follow are predictable:

- How was the site selected, and what evidence supported its expected accrual?
- How were eligibility criteria interpreted across sites, and how was consistency checked?
- What monitoring was performed at sites with low accrual or high screen failure?
- What was done when enrolment deviated from plan, and when was it decided?

A programme with a versioned feasibility report, a site-selection rationale, and a monitoring record
can answer these. A programme with a spreadsheet and a memory cannot.

## 9. Industry practice and case evidence

### 9.1 Published benchmarks

The most useful peer-reviewed benchmark for Phase III accrual is a 2024 analysis in *Nature Reviews
Drug Discovery* of 2,140 industry Phase III trials initiated since 2011. It reports an average of
approximately 0.84 patients per site per month, with substantial variation: about 1.05 for trials
under 500 patients, 0.57 for trials of 500-1,500 patients, and 0.28 for trials above 1,500 patients.
Oncology averaged 0.77 and non-oncology 0.85. Trials sponsored by the largest companies averaged
0.62, against 1.0 for other sponsors.

The pattern is that larger trials accrue more slowly per site, because they require more sites and
the marginal site is less productive. A feasibility model that uses a single benchmark for all trial
sizes will overestimate accrual for a large trial.

Industry analyses widely cited in trade press, attributed to the Tufts Center for the Study of Drug
Development, report average site activation of roughly 31 weeks from site identification, about 11%
of activated sites enrolling no patients, and more than half of studies exceeding their planned
enrolment period. These figures are presented as industry analyses rather than peer-reviewed
estimates; they are directionally consistent with the peer-reviewed accrual data.

### 9.2 How sponsors respond

| Response | What it addresses | Evidence of effect | Main risk |
|---|---|---|---|
| Add sites | Steady-state capacity | Increases accrual after activation lag | New sites underperform |
| Site support and referral pathways | Per-site rate | Plausible; limited public evidence | Cost, dependence on site staff |
| Broaden eligibility | Screen pass rate | Direct arithmetic effect | Changes the population and the label |
| Add countries | Patient pool and speed | Variable; regulatory lag | Heterogeneity, ethics |
| Decentralised elements | Travel barrier | Emerging evidence | Data quality, oversight |
| Patient identification services | Pre-screening | Variable; vendor-reported | Privacy, consent, cost |
| Protocol simplification | Site burden and quality | Consistent with quality-by-design principles | Requires discipline in design |

The evidence for most responses is operational rather than experimental. A sponsor that adopts a
referral programme cannot easily separate its effect from seasonality, site maturation, and
concurrent changes. The honest statement is that the direction is plausible and the magnitude is
uncertain.

### 9.3 Case: a trial that missed its enrolment window

Public examples of enrolment failure are common but rarely documented in detail. What is
documentable is the pattern: a trial registered with an estimated primary completion date, a later
amendment extending it, and a final result published years after the original plan. ClinicalTrials.gov
records these changes, and the record is public.

**Documented:** registration dates, estimated and actual completion dates, and protocol amendments.
**Inference:** the reasons for delay, which are usually described in general terms (slow enrolment,
protocol amendment) and rarely quantified. A programme should not assume that a published delay
was caused by the factor named in the amendment; the amendment may be a response to a delay whose
cause is not disclosed.

### 9.4 What the evidence does not support

| Claim | Status |
|---|---|
| Site activation can be reduced to 90 days across all settings | A target, not an observed average; academic centres are slower |
| Adding sites proportionally increases enrolment | False in the short run because of activation lag and site heterogeneity |
| Historical site performance predicts future performance | Supported directionally; the correlation is moderate, not deterministic |
| Decentralised trials enrol faster | Plausible for travel-limited populations; not established as an average effect |

### 9.5 Site networks and patient identification

Three models are used to find patients faster than individual sites can:

- **Site networks.** Groups of practices that share feasibility and recruitment infrastructure. The
  network can pre-screen across members, but the trial is still conducted at accredited sites.
- **Patient identification services.** Vendors query claims or EHR data to find potential
  participants, then route them to sites. The evidence for effect is vendor-reported; independent
  evaluations are scarce.
- **Registry-based recruitment.** Disease registries contact members about trials. Effective in rare
  disease, where the registry may be the only complete list of diagnosed patients.

Each model shifts the bottleneck. Networks shift it to site capacity; identification services shift
it to consent and verification; registries shift it to registry coverage and member responsiveness.
None of them changes the number of eligible patients, which remains the upper bound.

### 9.6 What a credible feasibility package contains

A package that a committee can approve without re-doing the analysis contains: the target and its
denominators; the eligibility rule table with computability; the source population with its
coverage; the site list with evidence; the activation and accrual distributions; the simulation with
its seed and code; the competing-trial range; the diversity assessment where applicable; the
contingency options with costs; and the triggers. Anything less transfers the analysis to the
committee, which is where feasibility arguments go to die.
| Diversity goals necessarily delay enrolment | Depends on site mix and catchment; not a general rule |

### 9.7 How benchmarks are misused

The 0.84 patients per site per month average is the most misused number in feasibility. It is an
average across 2,140 trials of different sizes, phases, and therapeutic areas, and the same analysis
reports 0.28 for trials above 1,500 patients. Using 0.84 for a large Phase III overstates accrual by
a factor of three.

Three rules reduce the misuse:

1. Match the benchmark to the trial size, phase, and therapeutic area before using it.
2. Compare the benchmark with the sponsor's own historical rate, and explain any gap.
3. Report the forecast under both rates rather than choosing the more convenient one.

A benchmark is a reference class, not a target. A programme that plans to beat the benchmark needs a
specific reason, such as a proven site network or a protocol with unusually light burden.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Target chosen to match the deadline | Implied accrual equals the assumed accrual | Reverse-engineered assumption | Plan misses by months | Compare with historical and benchmark rates |
| Activation assumed at month 1 | Flat accrual from the start | No activation model | Timeline too short by the ramp area | Model activation explicitly |
| Screen failure omitted | Screens equal randomised target | Eligibility applied to the wrong denominator | Site workload understated | Apply screen-failure rate with a source |
| Evaluable target ignored | Randomised target equals power target | Dropout excluded | Underpowered analysis | Inflate randomised target |
| Single accrual rate for all sites | No site-level distribution | Heterogeneity ignored | Variance understated | Gamma or empirical site distribution |
| Site selection by reputation | Publication record in the selection model | Wrong predictor | Slow accrual at prestigious sites | Select on prior accrual in the indication |
| Competing trials ignored | No registry analysis | Pool overestimated | Enrolment cannibalised | Competing-trial landscape with lag |
| Diversity goals treated as reporting | Plan written after site selection | Goals not in the design | Goals missed | Goals as site-selection criteria |
| No triggers | First reforecast after the problem | No control system | Late, expensive rescue | Pre-specified thresholds |
| Rescue by adding sites only | Rescue plan is a site list | Wrong lever for the constraint | Cost without time saved | Diagnose the binding constraint |
| Over-enrolment uncontrolled | Enrolment exceeds target before close-out | No stop rule | Analysis population ambiguity | Pre-specified over-enrolment rule |
| Forecast presented as a point | Single date in the governance paper | Uncertainty hidden | False confidence | Distribution and probability of deadline |
| Feasibility data not retained | Model inputs not documented | No audit trail | Cannot learn or defend | Versioned feasibility report |

The red flag that is easiest to detect and most often missed is the coincidence between the assumed
accrual rate and the rate implied by the deadline. If 800 patients, 60 sites, and 24 months imply
0.56 patients per site per month, and the model assumes 0.55, the assumption was not estimated; it
was solved for.

A second red flag is a site list that contains only sites the team has used before. Familiarity is a
real advantage in activation, but a list built entirely on relationships excludes sites with better
catchments and produces a plan that is comfortable rather than feasible.

### 10.2 The five metrics of an enrolment dashboard

A dashboard that reports only cumulative enrolment cannot diagnose a problem. Five metrics, each
with a target from the forecast, are sufficient to distinguish the common causes.

| Metric | Definition | Diagnostic use |
|---|---|---|
| Activated sites | Sites that have randomised at least one patient | Distinguishes activation delay from accrual weakness |
| Enrolment per active site | Cumulative randomised divided by active-site months | Measures the accrual assumption directly |
| Screen failure rate | Screen failures divided by screens | Detects eligibility interpretation problems |
| Screen-to-randomise ratio | Screens divided by randomised | Measures site workload against plan |
| Dropout rate | Discontinuations divided by randomised | Detects a retention problem before it affects power |

The combination is diagnostic. Low activated sites with normal per-site accrual is an activation
problem. Normal activated sites with low per-site accrual is a site-performance or population
problem. High screen failure with normal accrual means sites are working harder for the same
output, which will show up in site satisfaction and cost before it shows up in the timeline.

A dashboard without the per-site rate will attribute every shortfall to site performance, which is
the most expensive diagnosis to act on and often the wrong one.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The source population is measurable | The funnel starts with a count | No dataset covers the geography | Triangulate registry, claims, and literature |
| Eligibility rules are computable | Queries require fields | Clinical judgement criteria | State the non-computable allowance |
| Historical site performance predicts future accrual | Site selection depends on it | New indication, new staff, competing trial | Compare predicted and observed by site |
| Activation follows a constant rate | The model needs an activation curve | Contract and regulatory variability | Use historical activation distribution |
| Accrual is Poisson conditional on the site rate | Simulation requires a likelihood | Clustering, seasonality | Check overdispersion in historical data |
| Screen failure is independent of accrual | Simplifies the model | Slow sites screen more and fail more | Correlate site-level screen failure and accrual |
| Dropout is independent of treatment | Power calculation assumes it | Differential discontinuation | Monitor by arm after unblinding |
| The competing-trial landscape is stable | The pool estimate assumes it | New trials open during recruitment | Re-scan registries quarterly |
| The deadline is fixed | The decision depends on it | Commercial or regulatory change | Confirm with the decision owner |
| Sites will accept the protocol burden | Site contracts assume it | Complex assessments, staffing | Site feasibility questionnaires and pilot |
| The registry landscape is complete enough to estimate competition | Competing-trial analysis | Unregistered and late-registered trials | Range the estimate; re-scan quarterly |
| Screen failure is constant across sites and over time | The model uses one rate | Site learning, seasonal referrals | Monitor by site and month |
| Feasibility assumptions are recorded | Learning requires it | Model inputs not retained | Versioned feasibility report |

The deepest limit is that feasibility forecasts are rarely validated. A programme that records its
predicted and observed accrual by site builds the only evidence that can improve the next forecast.
Without that record, each study starts from the same benchmark and makes the same error.

## 12. Exercises

**Exercise 1 — Funnel arithmetic (application, 25 minutes).**
Deliverable: for a randomised target of 640, 15% dropout, and 35% screen failure, compute the
evaluable population, the required randomised number, and the screening target.
Acceptance criteria: each denominator labelled; arithmetic shown; screen-to-randomise ratio stated.
Time: 25 minutes.

**Exercise 2 — Eligibility translation (application, 40 minutes).**
Deliverable: a rule table for five eligibility criteria, with data field, code set, time window, and
computability status.
Acceptance criteria: at least one criterion marked not computable with an allowance; code systems
named; no criterion left undefined.
Time: 40 minutes.

**Exercise 3 — Activation model (design, 35 minutes).**
Deliverable: a month-by-month activation table for 48 sites activating at 4 per month, with the
cumulative active-site count and cumulative enrolment at 0.6 patients/site/month.
Acceptance criteria: ramp modelled; cumulative enrolment at months 6, 12, and 24 stated; completion
month for a target of 700 stated.
Time: 35 minutes.

**Exercise 4 — Forecast critique (critique, 45 minutes).**
Deliverable: a 450-word critique of a plan that assumes 60 sites active at month 1, 0.55
patients/site/month, no screen failure, and a 24-month deadline.
Acceptance criteria: at least five defects; at least three quantified; a corrected forecast range
proposed.
Time: 45 minutes.

**Exercise 5 — Simulation (application, 60 minutes).**
Deliverable: a simulation of completion time for 50 sites, 0.5 patients/site/month, 25% screen
failure, and 10% dropout; report median, 80% interval, and P(within 30 months).
Acceptance criteria: distributions stated; seed recorded; assumptions listed; code runnable.
Time: 60 minutes.

**Exercise 6 — Rescue decision (design, 50 minutes).**
Deliverable: a decision memo comparing adding 15 sites, extending by 6 months, and broadening
eligibility, on probability of success, cost, and evidence risk.
Acceptance criteria: all three options priced; the binding constraint identified; a recommendation
with conditions.
Time: 50 minutes.

**Exercise 7 — Site selection model (design, 45 minutes).**
Deliverable: a scoring model for 40 candidate sites using prior accrual, competing trials, catchment,
and site type, with weights and a validation plan.
Acceptance criteria: predictors justified; weights stated; a plan to compare predicted with observed
accrual after 6 months.
Time: 45 minutes.

**Exercise 8 — Diversity plan as design input (application, 40 minutes).**
Deliverable: a revised site mix and accrual assumption for a trial with enrolment goals by race,
ethnicity, sex, and age.
Acceptance criteria: goals translated into site criteria; timeline effect estimated; waiver
considered with rationale if applicable.
Time: 40 minutes.

## 13. Self-check answers

**Exercise 1.** Evaluable 640; randomised 640/0.85 = 753; screens 753/0.65 = 1,159. Screen-to-
randomise ratio 1,159/753 = 1.54.

**Exercise 2.** Examples: age ≥ 18 (computable from date of birth; no code set); diagnosis (ICD-10
or SNOMED code, computable but stage often missing); prior therapy (drug codes and dates,
computable with a line definition); renal function (LOINC-coded eGFR, computable only where
laboratory data exist); measurable disease (not computable from claims; requires imaging data or a
screen-failure allowance).

**Exercise 3.** Cumulative active sites: month 6 = 24, month 12 = 48. Cumulative enrolment by month
6 = 0.6 × 4 × (1+2+...+6) = 0.6 × 4 × 21 = 50.4. By month 12 = 0.6 × 4 × 78 = 187.2. Steady state
48 × 0.6 = 28.8 per month. Remaining to 700 = 512.8, or 17.8 months, so completion in month 30.

**Exercise 4.** Defects: no activation ramp; no screen failure; no dropout; a single accrual rate
with no site heterogeneity; no probability statement; no competing-trial analysis; no trigger. The
quantified corrections: completion moves from 24.2 to about 30 months with activation; screens rise
from 800 to about 1,140 with 30% screen failure; randomised target rises from 800 to 909 with 12%
dropout. The corrected forecast should be a distribution with a median near 31 months and a wide
80% interval.

**Exercise 5.** A correct answer reports a median in the range of 28-34 months, an 80% interval that
spans at least 12 months, and P(within 30 months) between 0.4 and 0.6, depending on the variance
assumptions. Full marks require the distributions and the seed to be stated.

**Exercise 6 rubric.** Full marks require each option priced on probability of meeting the deadline,
cost, and evidence risk; the binding constraint named (for example, accrual rate rather than site
count); and a recommendation with conditions and a review date. Deduct marks for recommending the
cheapest option without regard to probability.

**Exercise 7.** A defensible model weights prior accrual in the indication most heavily (for example,
50%), then competing trials (20%), catchment (20%), and site type (10%). The validation plan compares
predicted with observed accrual at 6 months and recalibrates the weights. A model that weights
publication record or investigator seniority highly is not defensible on the available evidence.

**Exercise 8.** Goals should translate into site selection criteria (catchment demographics,
community relationships), an accrual assumption that reflects the smaller per-group pool, and a
timeline that includes the additional start-up. A waiver is appropriate only if the study is
infeasible for a group, not merely slower.

## 14. Interview questions

**1. How would you forecast enrolment for a Phase III trial?**
Strong: fix denominators, translate eligibility, model activation and accrual distributions,
simulate, report a distribution and probability of the deadline, state the binding constraint.
Weak: divide the sample size by sites times a rate.

**2. Why is the deterministic forecast almost always optimistic?**
Strong: inputs are multiplied and uncertain; the product of central values is below the mean of the
product; activation and heterogeneity add a tail. Weak: says "things always take longer".

**3. What is the difference between screen failure and dropout, and where does each apply?**
Strong: screen failure before randomisation affects site workload; dropout after randomisation
affects power. Weak: treats both as one attrition rate.

**4. How would you decide whether to add sites?**
Strong: diagnose the binding constraint, price the activation lag, compare with extending the window
and broadening eligibility, consider evidence risk. Weak: adds sites because enrolment is behind.

**5. What makes a site a good site?**
Strong: prior accrual in the indication, catchment, competing trials, staff stability, and
demonstrated behaviour. Weak: reputation or publication record.

**6. How do you handle a rare disease with 500 eligible patients globally?**
Strong: patient-identification networks, registries, travel and decentralised elements, competing
trials, high screen failure, and a funnel rather than a site-rate model. Weak: adds more sites.

**7. What would you do if screen failure is 50% instead of 30%?**
Strong: audit eligibility interpretation, check whether the criteria are operational, consider an
amendment, recompute the screening target and timeline, and assess the effect on the population.
Weak: increases the number of screens without asking why.

**8. How do diversity goals change a feasibility model?**
Strong: they change site selection and the per-group eligible pool, and may extend the timeline;
they are a design input, not a reporting task. Weak: says they are handled by the site team.

**9. What triggers would you set for a rescue plan?**
Strong: thresholds derived from the forecast distribution, with owners, actions, and review dates.
Weak: "review monthly".

**10. How would you validate a feasibility forecast after the trial?**
Strong: compare predicted and observed accrual by site and month, decompose the error into
activation, accrual, and screen failure, and record it for the next study. Weak: compares total
enrolment only.

## 15. Key terms

- **Accrual rate** — patients enrolled per site per month.
- **Activation** — the point at which a site can randomise.
- **Activation lag** — time from site selection to activation.
- **Catchment area** — the geography from which a site draws patients.
- **Competing trial** — another study recruiting the same patients.
- **Computable phenotype** — an algorithm identifying a condition in a dataset.
- **Decentralised trial (DCT)** — trial with remote or local elements replacing site visits.
- **Diversity action plan (DAP)** — a plan for enrolment of underrepresented populations.
- **Dropout** — loss of randomised patients before the endpoint.
- **Enrolment curve** — cumulative randomised patients over time.
- **Evaluable population** — patients with the primary endpoint measured.
- **Feasibility** — assessment of whether a trial can be conducted as designed.
- **Feeder pattern** — the distribution of a site's referrals by source.
- **Pre-screening** — identifying potentially eligible patients before consent.
- **Randomised target** — number of patients to be allocated.
- **Screen failure** — a screened patient who is not randomised.
- **Screen-to-randomise ratio** — screened patients per randomised patient.
- **Screening target** — patients who must be screened to reach the randomised target.
- **Site maturity** — the ramp in accrual after activation.
- **Source population** — patients with the condition in the geography of interest.
- **Trigger** — a pre-specified deviation that requires action.

## 16. Sources

- Nature Reviews Drug Discovery. Benchmarking recruitment rates for phase III trials. 2024.
  https://doi.org/10.1038/d41573-024-00183-4 (accessed 25 October 2026).
- Tufts Center for the Study of Drug Development. Site activation and enrolment analyses, as
  reported in *Applied Clinical Trials*, "Enrollment Performance: Weighing the Facts".
  https://www.appliedclinicaltrialsonline.com/view/enrollment-performance-weighing-facts
  (accessed 25 October 2026).
- WCG Clinical. *Decoding the Top Site Challenges of 2024*.
  https://www.wcgclinical.com/wp-content/uploads/2024/12/Decoding-the-Top-Site-Challenges-of-2024-Study-Start-Up-Slides.pdf
  (accessed 25 October 2026).
- US Food and Drug Administration. *Diversity Action Plans to Improve Enrollment of Participants
  from Underrepresented Populations in Clinical Studies*. Draft guidance, June 2024.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/diversity-action-plans-improve-enrollment-participants-underrepresented-populations-clinical-studies
  (accessed 25 October 2026).
- US Food and Drug Administration. *Conducting Clinical Trials With Decentralized Elements*.
  Final guidance, 2024.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/conducting-clinical-trials-decentralized-elements
  (accessed 25 October 2026).
- ICH. *E8(R1) General Considerations for Clinical Studies*. 2021.
  https://www.ich.org/page/efficacy-guidelines (accessed 25 October 2026).
- ICH. *E6(R3) Good Clinical Practice*. 2025. https://www.ich.org/page/efficacy-guidelines
  (accessed 25 October 2026).
- ClinicalTrials.gov. API v2 documentation. https://clinicaltrials.gov/data-api/api
  (accessed 25 October 2026).
- European Medicines Agency. *Clinical Trials Information System (CTIS)*.
  https://www.ema.europa.eu/en/human-regulatory-overview/research-development/clinical-trials-human-medicines/clinical-trials-information-system
  (accessed 25 October 2026).
- US Government. *45 CFR 46, Protection of Human Subjects*.
  https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-A/part-46 (accessed 25 October 2026).

> **Note:** The trial, site counts, and forecasts in Section 6 are constructed for teaching. The
> benchmark accrual figure is drawn from the 2024 *Nature Reviews Drug Discovery* analysis and is
> used as a comparator, not as a prediction for this fictional trial.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_biomarker-strategy.md`](03_biomarker-strategy.md)
- Next topic: [`05_cdisc-and-trial-standards.md`](05_cdisc-and-trial-standards.md)
- Related: [`07_rwe-in-pharma.md`](07_rwe-in-pharma.md)
- Related: [`09_decentralized-trials.md`](09_decentralized-trials.md)
- Survey file: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
