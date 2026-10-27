# Global Health and LMIC Data Systems

> Module 02 — The Biomedical Data Landscape
> Topic 11 of 11
> Estimated study time: 4 hours
> Prerequisites: Topics 01 (EHR data), 03 (registries and public health), 08 (standards and terminologies), 10 (data quality dimensions)
> Version 2026.3

## 1. Why this topic exists

Most clinical data science is built on data from high-income hospitals. A 2024 review of 159 artificial intelligence-enabled clinical studies found that 74.0% were conducted in high-income countries, 23.7% in upper-middle-income countries, 1.7% in lower-middle-income countries, and 0.6% in low-income countries. A 2022 review of 7,314 clinical artificial intelligence papers found that the most-used datasets came from the United States (40.8%) and China (13.7%). The evidence base is narrow, and the narrowness is not a moral footnote. It changes what a model learns and where it works.

Health systems in low- and middle-income countries (LMICs) do not run on the same data infrastructure. They run on a small number of open-source platforms, on paper registers that are summarised monthly, on household surveys conducted every few years, and on civil registration systems that record a minority of deaths. The data are produced by operational processes that were designed for reporting, not for analysis. A district health officer, a Ministry of Health monitoring and evaluation unit, a donor, and a research group can each hold a different number for the same indicator, and all four can be defensible.

The industry problem is that decisions are made on those numbers. A national immunisation programme orders vaccines against a coverage estimate. A donor allocates funding against a district ranking. A ministry decides where to place a new health centre. A research consortium trains a model on a survey and deploys it in a clinic. Each decision inherits the error structure of the data that produced it, and the error structure is rarely written down.

Four failures recur.

**The coverage estimate above 100%.** A district reports 12,400 third doses of a three-dose vaccine against a projected cohort of 12,000 children, for a coverage of 103.3%. The number is printed, questioned, and then explained away as "doses from neighbouring districts". It is more likely a denominator that does not match the catchment, a numerator that includes children from outside it, or both. A coverage estimate above 100% is not a data quality nuisance. It is a signal that the ratio is measuring something other than what it claims.

**The research dataset mistaken for a monitoring system.** A team builds a district-level model on a Demographic and Health Survey (DHS), validates it, and proposes it as a routine monitoring tool. The survey is representative at the national and regional level, is fielded every few years, and has a sample sized for national estimates. It cannot monitor a district quarterly, and it cannot detect a change that occurs between survey rounds.

**The small-area estimate that invents precision.** A district with 24 surveyed children has a direct coverage estimate of 75.0% and a 95% confidence interval from 53.1% to 96.9%. A model shrinks that estimate toward the national mean and reports 76.5% with an interval from 66.1% to 86.8%. The second interval is narrower, but it is narrower because of the model, not because of new data. If the model is wrong, the interval is too narrow and the district is ranked on noise.

**The analysis that ignores who owns the data.** A consortium extracts facility-level data, sends it to a server in another country, trains a model, and publishes. The ministry that generated the data cannot reproduce the result, cannot audit it, and in some jurisdictions did not lawfully permit the transfer. The technical result may be correct and the project still fails, because data sovereignty is a legal and political constraint, not an administrative inconvenience.

The scale is not marginal. The World Bank's 2025 income classification places about 7.6% of the world's population in low-income economies, 38% in lower-middle-income, and 34% in upper-middle-income, so roughly four in five people live in low- and middle-income economies. The health data systems that serve them are the subject of this topic.

There is also a definitional trap. "LMIC" is an income classification, not a setting. A lower-middle-income country can contain a tertiary hospital with a modern electronic health record (EHR) and a rural district with paper registers. A study that says "LMIC" without saying which district, which facility level, and which data system has described a country income group, not a data environment.

This topic covers the systems that produce global health data: the routine health information system, the household survey, the facility registry, civil registration and vital statistics, geospatial covariates, and phone-based surveys. It covers the denominator problem, small-area estimation, equity stratification, and data sovereignty. It ends with the distinction that matters most for a data scientist: a research dataset is not an operational system, and the methods that work for one do not transfer to the other.

### 1.1 Who uses the number, and for what

Five users consume the same coverage estimate, and each uses it for a different decision.

A district health officer uses it to decide where to send a supervision team. A regional director uses it to rank districts and allocate a small discretionary budget. A national programme uses it to order vaccines and set a target. A donor uses it to decide whether to renew a grant. A research group uses it as an outcome in an evaluation.

The users differ in the precision they need and in the loss they suffer from error. A supervision visit sent to the wrong facility costs a day. A vaccine order based on an inflated coverage estimate leaves children unvaccinated. A grant renewed on a biased evaluation keeps a failing programme alive.

| User | Decision | Precision needed | Cost of error |
|---|---|---|---|
| District officer | Where to supervise | District | Wasted visit |
| Regional director | Who to prioritise | District, with intervals | Misallocated budget |
| National programme | How much to procure | National | Stockout or waste |
| Donor | Whether to continue | National, trend | Continued ineffective spend |
| Researcher | Effect estimate | Cluster or individual | Wrong conclusion |

A single number cannot serve all five. The national estimate can tolerate a wider interval than the district ranking, and the district ranking needs an interval that the national dashboard rarely reports.

### 1.2 The same indicator, three systems

The four failures above share a structure: a number produced by one system is used as if it came from another.

| System | What it counts | Population it represents | Time structure |
|---|---|---|---|
| Routine health management information system (HMIS) | Service contacts at reporting facilities | People who used those facilities | Monthly, continuous |
| Household survey | Self-reported contacts and status | Sampled households, weighted to the population | Every three to five years |
| Civil registration | Registered vital events | Events that were registered | Continuous, with lag |

A coverage estimate from the routine system describes the people who reached a facility. A coverage estimate from a survey describes the sampled population, including those who did not reach a facility. A death rate from civil registration describes the registered deaths. None of the three is the population truth, and the gaps between them are the subject of the analysis.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** routine facility data, household survey data, civil registration, and geospatial covariates by their production process, their population, and their error structure.
2. **Specify** an indicator with a numerator, a denominator, a period, a geography, and a data source, and identify which of the five is most likely to be wrong.
3. **Compute** reporting completeness, timeliness, and coverage, and explain why a coverage estimate above 100% is a specification error rather than a rounding error.
4. **Reconcile** a facility list against a health management information system hierarchy and show how the reconciliation changes the expected-report denominator.
5. **Estimate** a small-area quantity with a direct estimator and an empirical Bayes estimator, report both intervals, and state what the narrowing depends on.
6. **Stratify** an indicator by an equity dimension and compute absolute and relative gaps with their uncertainty.
7. **Adjudicate** a disagreement between two sources by naming the bias direction of each rather than averaging them.
8. **Assess** a data-sharing arrangement against national data protection law and the principles of data sovereignty.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1. Source taxonomy | Exercise 1 | Items 1, 2 |
| 2. Indicator specification | Exercise 2 | Items 3, 4 |
| 3. Completeness and coverage | Exercise 3 | Items 5, 6 |
| 4. Facility reconciliation | Exercise 4 | Item 7 |
| 5. Small-area estimation | Exercise 5 | Items 8, 9 |
| 6. Equity stratification | Exercise 6 | Item 10 |
| 7. Triangulation | Exercise 7 | Items 11, 12 |
| 8. Data sovereignty | Exercise 8 | Items 13, 14 |

The outcomes are ordered by the sequence of work in a district analysis: understand the sources, specify the indicator, check the reporting, reconcile the list, estimate, stratify, adjudicate, and govern. The middle outcomes are the technical core; the first and last are the ones that determine whether the analysis is used.

The exercises use a single district and a single indicator so that the arithmetic can be checked by hand. The methods transfer to other indicators, but the parameters do not: a coverage indicator, a mortality rate, and a service-readiness score have different denominators and different error structures.

The topic assumes the vocabulary of Topic 03 on registries and public health, Topic 08 on standards and terminologies, and Topic 10 on data quality. A reader who has not covered those topics should read the definitions in Section 4 first, because the rest of the file uses them without re-definition.

## 3. The core mental model

Global health data are produced by four systems that run in parallel and measure different populations. The analyst's job is not to choose the best one. It is to know what each one measures, where it fails, and how to combine them without pretending the combination is a validation.

The routine system is usually an HMIS built on DHIS2 (District Health Information Software 2), the open-source platform described in Section 8.1. The survey system is a household programme such as the Demographic and Health Survey (DHS) or the Multiple Indicator Cluster Survey (MICS). The vital statistics system is civil registration. The geospatial layer is a set of modelled covariates, not a count of people.

```text
   POPULATION            PRODUCTION SYSTEM              PRODUCT

   every facility         routine HMIS                  monthly facility
   contact         --->   register -> tally ->   --->   report -> DHIS2
                          monthly summary               aggregate data
                               |                             |
                               | completeness, timeliness     | coverage,
                               | stockouts, transfers         | equity
                               v                             v
   sampled            ---> household survey   --->   direct estimate
   households              (DHS, MICS)                  with design-based
                               |                        variance
                               | recall, response,
                               | phone ownership
                               v
   every death        ---> civil registration  --->   vital statistics
                          and vital statistics           (cause, age, sex)
                               |
                               | registration coverage,
                               | cause-of-death coding
                               v
   every person       ---> geospatial covariates --->  denominators,
   and place                (population, roads,        catchments,
                            facilities, climate)       accessibility
```

Read the diagram as four parallel production lines, not four stages of one pipeline. They do not share a denominator, a time stamp, or an error model. The only thing they share is the population they are trying to describe.

### 3.1 The denominator is the object

Every coverage indicator is a ratio, and in global health the denominator is usually the weaker half. The numerator comes from a service contact: a child vaccinated, a woman attending antenatal care, a case notified. The denominator comes from a projection: a census that may be a decade old, a catchment population that may not match the facility's actual users, or a survey weight that may not sum to the projected total.

Three consequences follow.

A denominator that is too small produces coverage above 100%. That is not a rare edge case; it is the normal result of a facility that serves people from outside its official catchment, or a census projection that has not been updated for migration.

A denominator that is too large produces coverage that looks inadequate and triggers an unnecessary programme response. A district that is actually at 85% coverage but is measured against a projected population inflated by unrecorded migration will look like it is at 70%, and the response will be to add services where the need is lower than believed.

A denominator that changes between years produces a trend that is an artefact. If the projection is rebased after a census, coverage can jump by ten points without a single additional child being vaccinated. Any trend analysis has to know the denominator vintage.

### 3.2 Missingness is structural

In a clinical trial, missing data are an inconvenience to be modelled. In an operational health system, missing data are a signature of the process. A facility that reports zero malaria cases in a month may have had no cases, or may have run out of rapid diagnostic tests, or may have been closed, or may have failed to submit the report. The zero is the same in the database; the meaning is not.

The practical rule is that a zero is a claim, and every zero should be classified before it is analysed. The classification is usually available from another system: stock records, supervision visits, facility status, or the reports of neighbouring facilities. A dataset without that classification will produce a model that learns the supply chain.

### 3.3 Research dataset versus operational system

The distinction is the most important one in this topic, and it is frequently blurred.

| Property | Research dataset | Operational system |
|---|---|---|
| Purpose | Answer a question | Run a service |
| Population | Defined by the study | Whoever shows up |
| Sampling | Designed, weighted | Census of contacts, not of people |
| Time | Fixed rounds | Continuous, revised |
| Variables | Collected for the question | Collected for the form |
| Missingness | Measured and reported | Absorbed into the workflow |
| Ownership | Investigator or institution | Ministry of Health |
| Lifespan | Until publication | Indefinite |
| Change control | Protocol amendment | Policy and training |
| Failure mode | Bias | Blindness |

A DHS is a research dataset. A DHIS2 instance is an operational system. A model trained on the first and deployed on the second is a transport problem, not a validation problem, and the transport is where most of the error enters.

### 3.4 Three questions every estimate must answer

Before a number is used, it has to answer three questions. Most disagreements between ministries, donors, and researchers are disagreements about one of them, not about the arithmetic.

| Question | Why it changes the number | Typical failure |
|---|---|---|
| Which population? | Facilities, catchments, and administrative areas differ | A facility serves a neighbouring district |
| Which period? | Reporting months, fiscal years, and survey recall differ | A survey recall period overlaps two calendar years |
| Which definition? | Doses, children, contacts, and episodes differ | Doses counted as children vaccinated |

An estimate that answers all three can still be wrong, but it can be checked. An estimate that answers none of them cannot be reconciled with any other source, and the disagreement becomes a political argument rather than an analytical one.

The practical habit is to write the three answers next to every number in a report. If the answer to any of them is unknown, the number is a candidate for the first table in Section 10, not for the executive summary.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| LMIC | Low- and middle-income country, a World Bank income classification | Used as a synonym for "poor" or "rural" | Classification |
| HMIS | Health management information system; the routine facility reporting system | Used for any health data system | System |
| DHIS2 | District Health Information Software 2, an open-source HMIS platform | Used as a synonym for HMIS | Platform |
| Aggregate data | Counts summed over a period and an organisation unit | Confused with patient-level data | Count |
| Tracker data | Individual-level longitudinal records in DHIS2 | Confused with aggregate data | Record |
| Data element | The "what" of a DHIS2 data value | Called a variable | Identifier |
| Organisation unit | The "where" of a DHIS2 data value | Called a facility | Identifier |
| Period | The "when" of a DHIS2 data value | Assumed to be calendar month | Identifier |
| Category option combination | The disaggregation of a data element | Ignored in extracts | Identifier |
| Completeness | Reported facility-months divided by expected facility-months | Confused with coverage | Proportion |
| Timeliness | Reports received by the deadline divided by expected | Confused with completeness | Proportion |
| Coverage | Service contacts divided by the target population | Confused with completeness | Proportion |
| Denominator | The population at risk or eligible | Taken from the wrong year | Count |
| Catchment population | The population a facility actually serves | Assumed to equal the administrative area | Count |
| Master facility list (MFL) | The authoritative registry of health facilities | Assumed to be complete | Registry |
| Civil registration and vital statistics (CRVS) | The system that records births, deaths, and causes | Confused with a survey | System |
| Verbal autopsy | A structured interview to assign cause of death | Treated as a diagnosis | Method |
| DHS | Demographic and Health Survey, a nationally representative household survey | Treated as routine data | Survey |
| MICS | Multiple Indicator Cluster Survey, a household survey run by the United Nations Children's Fund (UNICEF) | Confused with DHS | Survey |
| Cluster | A sampled geographic unit in a survey | Treated as a facility | Sampling unit |
| Design effect | The ratio of the variance under the design to the variance under simple random sampling | Ignored in confidence intervals | Ratio |
| Sampling weight | The inverse probability of selection, adjusted for nonresponse | Ignored in estimates | Weight |
| Small-area estimation | Model-based estimation for a domain with a small sample | Treated as an observation | Method |
| Empirical Bayes | Shrinkage of a local estimate toward a prior mean | Treated as a substitute for data | Method |
| Shrinkage | The weight given to the prior in a model-based estimate | Confused with smoothing for presentation | Proportion |
| Equity stratifier | A dimension such as wealth, sex, residence, or education | Reduced to one dimension | Variable |
| Quintile | A fifth of the population ranked by a socioeconomic index | Treated as an absolute income group | Category |
| Concentration index | A measure of inequality across a ranked distribution | Confused with the Gini coefficient | Index |
| Data sovereignty | The principle that data are governed by the laws and institutions of the country of origin | Treated as an IT policy | Principle |
| Data protection law | National law governing personal data processing | Assumed to be absent in LMICs | Law |
| IHR | International Health Regulations, a binding instrument on disease surveillance | Confused with a recommendation | Treaty |
| PHEIC | Public health emergency of international concern | Confused with a pandemic | Declaration |
| PABS | Pathogen access and benefit-sharing under the World Health Organization (WHO) Pandemic Agreement | Confused with the Nagoya Protocol | Instrument |
| Nagoya Protocol | A protocol on access to genetic resources and benefit-sharing | Applied to pathogens without qualification | Treaty |
| Geospatial covariate | A spatially linked variable used in a model | Treated as a measurement | Variable |
| Gridded population | A population estimate on a regular grid | Treated as a census | Estimate |
| Phone survey | A survey conducted by telephone | Assumed to reach the same population as a household survey | Method |
| Undercoverage | The share of the target population with no chance of selection | Ignored in phone surveys | Proportion |
| Response rate | Completed interviews divided by eligible sampled units | Reported without the eligibility definition | Proportion |
| Administrative coverage | Service contacts divided by a projected target population | Reported as true coverage | Proportion |
| Effective coverage | Contacts that received adequate quality care, divided by need | Replaced by crude coverage | Proportion |
| Target population | The eligible population for an intervention | Taken from the total population | Count |
| Reporting rate | Reports received divided by reports expected | Confused with completeness of the indicator | Proportion |
| Triangulation | Adjudication across sources with different biases | Averaging the sources | Method |
| Geospatial join | Linking a record to a place via coordinates | Treating coordinates as exact | Operation |
| Displaced coordinates | Deliberate offset applied to survey clusters | Reversed or ignored | Metres |
| Data quality assurance | The system of checks and feedback around the data | Reduced to a cleaning script | Process |

Two definitions deserve a note.

**Completeness is not coverage.** Completeness asks whether the reports arrived. Coverage asks whether the services reached the population. A district can have 100% reporting completeness and 60% coverage, or 70% completeness and 95% coverage among the facilities that reported. The two numbers answer different questions, and a dashboard that shows them side by side without labels invites the reader to confuse them.

**Equity is not a subgroup analysis.** A subgroup analysis asks whether the programme worked differently in one group. An equity analysis asks whether the gap between groups is closing, and it requires the groups to be defined on a dimension that policy can act on, with a population weight, and with an uncertainty interval around the gap. A table of coverage by region without weights and intervals is a description, not an equity analysis.

### 4.1 The three proportions, side by side

Completeness, timeliness, and coverage are all proportions, and they are routinely displayed together. Their numerators and denominators are different, and the table makes that explicit.

| Measure | Numerator | Denominator | Question |
|---|---|---|---|
| Completeness | Facility-months reported | Facility-months expected | Did the report arrive |
| Timeliness | Reports by the deadline | Reports expected or received | Did it arrive in time |
| Coverage | Service contacts | Eligible population | Did the service reach people |

A dashboard that shows all three without the definitions invites the reader to compare them. They are not comparable. A district can have 95% completeness, 70% timeliness, and 60% coverage, and the three numbers describe three different parts of the system.

## 5. The mechanics, step by step

The procedure assumes a national or district health information team, access to the routine system, and at least one survey or census. It produces an estimate with a documented denominator, a stated uncertainty, and an owner.

### Step 1. Define the decision, the indicator, and the unit of analysis

**Input:** the policy question, the reporting calendar, the geography, and the available sources.

**Output:** an indicator specification with a numerator, a denominator, a period, a geography, a stratification, and a primary source.

**Rationale:** "coverage" is a ratio, and a ratio without its five elements is not computable. The specification also determines which source is primary: a count can come from the routine system, a proportion may need a survey denominator, and a rate needs a population at risk.

| Indicator type | Numerator | Denominator | Example |
|---|---|---|---|
| Count | Service events | None | Doses administered |
| Coverage | Eligible people reached | Eligible population | Children receiving DTP3 / surviving infants |
| Rate | New events | Person-time at risk | New HIV diagnoses / 100,000 person-years |
| Ratio | One count | Another count | Maternal deaths / 100,000 live births |
| Proportion | Events | All units observed | Caesarean sections / live births |

**Branch point:** if the decision is a national allocation, the indicator needs a national estimate with a confidence interval. If the decision is a district performance review, it needs a district estimate with a denominator the district can verify. The two specifications differ.

**Edge case:** an indicator can be defined in a way that no source can measure. "Effective coverage" requires knowing that a service was delivered with adequate quality, which the routine system does not record. A specification that cannot be operationalised should be reduced to a proxy, and the reduction should be written down.

### Step 2. Fix the denominator before the numerator

**Input:** census projections, catchment maps, facility lists, and survey weights.

**Output:** a documented denominator with a source, a vintage, a method, and a stated range.

**Rationale:** numerator errors are visible because they change the count. Denominator errors are invisible because the denominator is usually a projection that no one re-checks. A denominator that is too small produces coverage above 100%, which is the most reliable signal that the ratio is misspecified.

**Arithmetic:** 12,400 doses against a projected cohort of 12,000 gives 103.3%. If the true catchment is 14,500 children because the facility draws from a neighbouring district, the same numerator gives 85.5%. The numerator did not change. The conclusion did.

**Branch point:** three denominators are available, and they are not interchangeable.

| Denominator | Source | Strength | Weakness |
|---|---|---|---|
| Census projection | National statistics office | Consistent nationally | Stale, ignores migration |
| Catchment estimate | Facility mapping or geospatial model | Reflects actual use | Uncertain, hard to audit |
| Survey-based | Household survey | Measured, with variance | Not available for every district |

**Edge case:** a denominator can be politically contested. A district that wants more resources has an incentive to inflate its population; a ministry that wants to show progress has an incentive to deflate it. The analyst should record the source and the vintage, and should not silently choose the one that produces the preferred coverage.

### Step 3. Map the data production process

**Input:** registers, tally sheets, monthly summary forms, supervision reports, and the DHIS2 metadata.

**Output:** a process map with each handoff, the person responsible, and the failure mode at each step.

**Rationale:** the data are a by-product of the process. A stockout, a transfer, a late report, and a double count each leave a different signature, and the signature is only visible if the process is known.

| Stage | Record | Responsible | Failure mode |
|---|---|---|---|
| Contact | Patient card or register | Clinician | Not recorded, illegible |
| Tally | Tally sheet | Clinic clerk | Double count, missed day |
| Summary | Monthly form | Facility in-charge | Arithmetic error, late |
| Entry | DHIS2 data entry | District officer | Wrong period, wrong org unit |
| Validation | DHIS2 validation rule | District officer | Rule ignored |
| Analysis | National dashboard | Programme officer | No denominator adjustment |

**Edge case:** a facility can report through a parallel system, such as a donor-funded project database, and the two records may disagree. The parallel system is often more complete for the indicator it funds and less complete for everything else. The analyst should know which system is the system of record, and why.

### Step 4. Reconcile the facility list

**Input:** the master facility list, the DHIS2 organisation unit hierarchy, survey cluster coordinates, and supervision records.

**Output:** a reconciled facility list with a unique identifier, a status, coordinates, and opening and closing dates.

**Rationale:** the expected-report denominator is the facility-month, and the facility list is usually wrong. Facilities open, close, change ownership, and change level. A list that is not reconciled produces a completeness figure that measures the list, not the reporting.

**Arithmetic:** 42 facilities over 12 months gives 504 expected reports. If three facilities opened at mid-year and two closed at mid-year, the facility-life expected count is 504 - 18 - 12 = 474. The same 431 received reports give 85.5% completeness against the raw list and 90.9% against the reconciled list.

**Branch point:** five discrepancy types, each with a different treatment.

| Discrepancy | Example | Treatment | Risk if ignored |
|---|---|---|---|
| Facility missing from the list | A new health post | Add with an opening date | Understates the denominator |
| Facility closed | A closed clinic | Censor from the closure date | Understates completeness |
| Facility duplicated | Two identifiers for one site | Merge on coordinates and name | Double-counts reports |
| Level changed | A health centre upgraded to a hospital | Update the level and the expected form | Wrong reporting form |
| Coordinates wrong | A facility mapped to the district capital | Re-geocode from the survey or a visit | Wrong catchment and map |

**Edge case:** the master facility list and the DHIS2 hierarchy can disagree on the district assignment, especially where administrative boundaries have changed. The analysis should use one hierarchy and record the choice, because a facility that moves between districts changes both the numerator and the denominator of every district indicator.

### Step 5. Assess completeness and timeliness

**Input:** the reported facility-months, the expected facility-months, and the reporting deadline.

**Output:** completeness and timeliness overall and by district, facility level, and month.

**Rationale:** completeness and timeliness are data-quality metrics, not coverage metrics. They describe the reporting system, and they are the first thing to check before any indicator is computed. A coverage estimate from a system with 60% completeness is a coverage estimate for the facilities that reported, not for the district.

**Arithmetic:** 431 reports received against 504 expected gives 85.5% completeness. Against the reconciled 474, it gives 90.9%. Of the 431 received, 358 arrived by the deadline, giving 71.0% timeliness against the expected count and 83.1% against the received count. The denominator has to be stated with each figure.

**Branch point:** the WHO Data Quality Review toolkit uses a benchmark of at least 90% reporting completeness for a facility-month dataset. If completeness is below the benchmark, the options are to adjust the estimate, to report it with a completeness caveat, or to restrict the analysis to the facilities that reported consistently.

**Edge case:** completeness can be high and the data still wrong. A facility that submits a report every month with the same numbers has perfect completeness and zero information. The consistency checks in the next step are what catch that.

### Step 6. Check internal consistency

**Input:** the facility-month data.

**Output:** an outlier list, a set of consistency ratios, and a digit-preference check.

**Rationale:** data errors have signatures. A first-dose count below the third-dose count, a delivery count above the antenatal-care count, a month with zero reports followed by a month with double the usual count, and a spike at round numbers are all detectable without an external source.

| Check | Rule | Action |
|---|---|---|
| Dose ordering | DTP1 at least DTP3 | Query the facility |
| Visit ordering | First antenatal care visit (ANC1) at least the fourth (ANC4) | Query the facility |
| Delivery versus ANC | Deliveries not above ANC4 | Query the facility |
| Zero followed by double | Two months sum to about two normal months | Check for a late report |
| Digit preference | Excess of 0 and 5 endings | Check for estimation |
| Repeated values | Same count for three months | Check for a copied form |
| Stockout contradiction | Zero cases with zero tests | Classify as a supply artefact |

**Edge case:** a consistency rule can be wrong in a legitimate case. A facility that receives referrals from another district can have more deliveries than antenatal visits. The query should produce an explanation, not an automatic correction.

### Step 7. Check external consistency

**Input:** the routine data, a household survey, civil registration, and commodity consumption records.

**Output:** a comparison table with ratios, dates, and the direction of the expected bias.

**Rationale:** no source is a gold standard. The routine system tends to over-count contacts and under-count the population; a survey tends to under-report events that are stigmatised and to over-represent households that are reachable. The comparison is informative when the direction of each bias is stated.

**Arithmetic:** administrative coverage of 103.3% against a survey estimate of 73.3% gives a ratio of 1.41. The gap is too large to be sampling error. The candidate explanations are a denominator that is too small, a numerator that counts doses rather than children, and a survey that under-reports recall. Each has a different implication for the programme.

**Branch point:** four disagreement patterns, each pointing to a different cause.

| Pattern | Likely cause | First check |
|---|---|---|
| Administrative much higher than survey | Denominator too small or doses double-counted | Catchment and dose records |
| Administrative much lower than survey | Facilities missing or reports incomplete | Facility list and completeness |
| Both low, CRVS low | System-wide under-provision | Service availability |
| Survey much lower for one antigen only | Recall or supply problem specific to that antigen | Stock records |

**Edge case:** the two sources may refer to different years. A survey fielded in 2024 asks about births in the five years before the survey, while the administrative data are for 2025. A comparison without aligning the periods is not a comparison.

### Step 8. Build the analysis dataset

**Input:** the cleaned sources.

**Output:** a tidy table at the analysis unit, with a documented key and a time alignment.

**Rationale:** joins fail silently. A facility identifier that is a string in one source and an integer in another, a period that is a calendar month in one source and a fiscal quarter in another, and a district boundary that changed between the survey and the routine data will all produce a dataset that looks complete and is wrong.

| Unit | Key | Time | Common join failure |
|---|---|---|---|
| Facility-month | Facility ID, period | Calendar month | Facility ID reused after a merge |
| District-year | District ID, year | Calendar year | Boundary change |
| Child | Cluster, household, line number | Interview date | Duplicate children across rounds |
| Cluster | Survey, cluster ID | Survey round | Coordinates displaced for confidentiality |

**Edge case:** DHS cluster coordinates are displaced, usually by a random offset within a small radius, and some are removed for confidentiality. A spatial join to a facility catchment therefore has an error that is not in the survey documentation. The displacement is deliberate and should not be undone.

### Step 9. Weight and adjust

**Input:** the survey design, the nonresponse pattern, and any benchmark population.

**Output:** weighted estimates with design-based variance, and a statement of the adjustment.

**Rationale:** a survey weight is the inverse of the probability of selection, adjusted for nonresponse and calibrated to a population total. An unweighted estimate describes the sample, not the population. The design effect captures the loss of precision from clustering and weighting, and it is often between 1.5 and 3 for health indicators.

**Arithmetic:** 132 vaccinated children out of 180 gives 73.3%. Under simple random sampling the standard error is the square root of 0.7333 x 0.2667 / 180 = 0.0330, for a 95% interval of 66.9% to 79.8%. With a design effect of 1.6, the standard error is 0.0417, for an interval of 65.2% to 81.5%. The point estimate is unchanged; the uncertainty is 26% wider.

**Branch point:** if the design effect is unknown, the options are to use a plausible range, to estimate it from a prior survey, or to report the simple-random-sampling interval with an explicit warning that it is too narrow.

**Edge case:** a weighted estimate can be unstable when a small number of clusters carry a large weight. The analyst should report the effective sample size and the number of clusters, not only the nominal sample size.

### Step 10. Stratify for equity

**Input:** the disaggregation dimensions, the population weights, and the indicator.

**Output:** estimates by subgroup, absolute and relative gaps, and a summary measure with an interval.

**Rationale:** a national average can improve while the gap between the richest and poorest widens. Equity analysis makes the distribution visible, and it requires the subgroup estimates to carry their own uncertainty.

**Arithmetic:** coverage of 58% in the poorest quintile and 91% in the richest gives an absolute gap of 33 percentage points and a ratio of 1.57. With standard errors of 3.1 and 2.2 points, the 95% interval for the gap is 26 to 40 points. The gap is large and precisely estimated.

**Branch point:** the choice of summary measure depends on the dimension. For an ordered dimension such as wealth, the concentration index and the slope index of inequality use the whole distribution. For a unordered dimension such as sex or region, the difference and the ratio are interpretable.

**Edge case:** the wealth index is a relative measure constructed from household assets. It ranks households within a survey, and it does not compare absolute wealth across countries or years. A quintile gap in one survey is not comparable with a quintile gap in another survey with a different asset list.

### Step 11. Estimate small-area quantities

**Input:** direct estimates with their variances, auxiliary covariates, and a national or regional mean.

**Output:** model-based estimates with intervals, and a statement of the model and its assumptions.

**Rationale:** a survey powered for national estimates produces district estimates with intervals so wide that the ranking is meaningless. A model borrows strength across areas, which reduces the variance and introduces bias if the model is wrong. The output should show both the direct and the model-based estimate, so the reader can see how far the model moved the number.

**Arithmetic:** a district with 132 of 180 children vaccinated has a direct estimate of 73.3% with a variance of 0.00174. Against a national mean of 81.5% and a between-district standard deviation of 6.0 points, the empirical Bayes weight on the direct estimate is 0.00174 / (0.00174 + 0.0036) = 0.33. The model estimate is 0.733 + 0.67 x (0.815 - 0.733) = 0.788, or 78.8%, with an interval of 72.1% to 85.6%.

**Branch point:** four methods, each with a different data requirement.

| Method | Data requirement | Strength | Weakness |
|---|---|---|---|
| Direct | Sufficient sample per area | Design-unbiased | Wide intervals |
| Smoothed direct / Fay-Herriot | Direct estimates plus variance | Transparent, standard | Needs stable variances |
| Cluster-level model | Cluster data and covariates | Uses within-area information | Computationally heavier |
| Geostatistical model | Coordinates and covariates | Maps at fine resolution | Strong spatial assumptions |

**Edge case:** a model can shrink a genuine outlier toward the mean and hide a failing district. The direct estimate should always be reported alongside the model estimate, and a district whose direct and model estimates differ by more than the model's interval deserves an explicit note.

### Step 12. Triangulate and adjudicate

**Input:** the routine, survey, CRVS, and model estimates.

**Output:** a recommended value with a range and a written rationale.

**Rationale:** triangulation is adjudication, not averaging. Averaging two biased estimates does not produce an unbiased one. The analyst has to decide which source is more credible for this indicator, in this district, in this period, and to say why.

**Branch point:** five disagreement patterns, each with a different adjudication.

| Pattern | Most likely explanation | Recommended value |
|---|---|---|
| Routine high, survey low, CRVS low | Denominator too small | Survey estimate, with a denominator note |
| Routine low, survey high, CRVS high | Facility list incomplete | Survey estimate, adjusted for catchment |
| All three low and consistent | Genuine low coverage | Routine estimate, with a service-readiness note |
| CRVS much lower than both | Death registration incomplete | Do not use CRVS as the denominator |
| Model far from both | Model misspecification | Report the direct estimate and the model as a range |

**Edge case:** the most credible source can change over time. A survey is the best source in a survey year and a poor source two years later. The recommended value should be time-stamped, and the next survey should be allowed to revise it.

### Step 13. Quantify uncertainty

**Input:** sampling error, non-sampling error, and model error.

**Output:** an interval or a plausible range, labelled by the type of error it covers.

**Rationale:** a confidence interval that covers only sampling error understates the uncertainty. In global health data, the non-sampling error from the denominator, the completeness, and the reporting process is often larger than the sampling error, and it is not captured by a formula.

| Error type | Source | How to quantify |
|---|---|---|
| Sampling | Survey design | Design-based standard error |
| Coverage | Undercoverage of the sampling frame | Weighting and sensitivity analysis |
| Nonresponse | Differential response | Response propensity model |
| Measurement | Recall, reporting, recording | Validation study or comparison |
| Denominator | Projection and catchment | Range of plausible projections |
| Model | Shrinkage and covariates | Prior sensitivity analysis |

**Edge case:** combining the errors by quadrature assumes independence, which is rarely true. A denominator error and a coverage error can share a cause, such as a facility that serves a larger catchment than recorded. The combined interval should be presented as a range, not as a precise confidence interval.

### Step 14. Document provenance

**Input:** every source, transformation, and assumption.

**Output:** a data dictionary, a version, and a reproducibility record.

**Rationale:** a number without provenance cannot be defended in a ministry meeting or an audit. The record should let another analyst reproduce the estimate from the raw sources, and it should let a reviewer see which choices were made.

**Contents:** source and vintage, facility list version, completeness and timeliness, denominator method, weighting, model specification, and the adjudication decision.

**Branch point:** if the estimate is published, the provenance record should be published with it, at least in summary form. A dashboard without a methods note is a rumour with a chart.

### Step 15. Feed back to the operational system

**Input:** the findings and the data-quality actions.

**Output:** a feedback report to facilities and districts, and a list of corrections.

**Rationale:** the system improves only if the people who produce the data see that it is used. A feedback report that names the facility and shows its own trend is more effective than a national dashboard that the facility never sees.

**Edge case:** feedback can create a gaming incentive. If a facility is ranked on coverage, the facility has an incentive to inflate the numerator or deflate the denominator. The feedback should include data-quality metrics alongside the indicator, so that improvement cannot be achieved by misreporting.

### Step 16. Monitor and revise

**Input:** the time series and the next survey or census.

**Output:** a revision log and a re-estimation schedule.

**Rationale:** estimates are revised when a new survey, census, or facility list arrives. A revision is not a failure; it is the system learning. The revision should be recorded, and the old estimate should be retained with its date so that a trend can distinguish a real change from a rebasing.

**Edge case:** a revision that is not communicated looks like a political intervention. The release note should say what changed, why, and how much.

### Step 17. Pre-specify the choices and review the estimate

**Input:** the indicator specification, the sources, and the candidate methods.

**Output:** a review record that states each choice, the alternative, and the effect on the estimate.

**Rationale:** most of the variation between analysts is not in the arithmetic but in the choices made before it. The denominator, the design effect, the prior, the treatment of zeros, and the adjudication rule can each move the estimate by more than the confidence interval. A pre-specified choice is defensible; a post-hoc choice that produces the preferred number is not.

| Choice | Default | Alternative | Effect on the estimate |
|---|---|---|---|
| Denominator | Census projection | Catchment estimate | Can move coverage by 10 points or more |
| Design effect | Survey estimate | Prior survey or sensitivity range | Changes interval width by 20-40% |
| Zero reports | Excluded | Classified and imputed | Changes numerator and completeness |
| Prior mean | National or regional mean | Neighbouring districts | Moves small-area estimates by several points |
| Adjudication | Survey as reference | Routine with a denominator note | Changes the recommended value |

**Branch point:** if the analysis is for a published national estimate, the choices should be fixed before the data are analysed and recorded in a methods note. If the analysis is exploratory, the choices can be tested, but the testing should be reported.

**Edge case:** a pre-specified choice can be wrong. The record should allow a change, with a reason, and the reason should be visible in the revision log.

### Step 18. Translate the estimate into a decision product

**Input:** the estimate, its interval, its source, and the decision it informs.

**Output:** a one-page brief or dashboard panel with the number, the range, the source, the period, and the action.

**Rationale:** a number without a range and a source is not decision-ready. The decision product is where the analysis either becomes usable or is reduced to a headline. The design of the product determines whether the uncertainty survives contact with the meeting.

| Product | Audience | What must be visible |
|---|---|---|
| District brief | District health team | Estimate, range, source, and the facility-level actions |
| National dashboard | Programme managers | Trend, range, completeness, and the denominator vintage |
| Donor report | Programme officer | Method, comparison with the previous estimate, and limitations |
| Equity profile | Policy team | Subgroup estimates, gaps, and intervals |
| Methods annex | Technical reviewers | Sources, choices, and reproducibility record |

**Edge case:** a map with five colour classes implies precision that a survey-based district estimate does not have. If the estimate's interval spans two classes, the map should show that, for example with hatching or a supplementary table.

### Decision table for the annual estimate review

| Finding | Interpretation | Action |
|---|---|---|
| Estimate stable, interval narrow, sources agree | The measurement is adequate for the decision | Publish and continue |
| Estimate stable, interval wide | The sample or the reporting is too small | Invest in data, or report a range |
| Estimate moved, denominator vintage changed | Rebasing, not a real change | Report the revision and the old series |
| Estimate moved, denominator unchanged | A real change or a reporting change | Check completeness before interpreting |
| Sources disagree by more than 20% | A specification or system problem | Adjudicate and document |
| Completeness below 80% | The routine estimate is not a district estimate | Use the survey or report the caveat |

## 6. Worked example

### 6.1 The case

District A has 42 health facilities and a projected population of 480,000. The national immunisation programme reports coverage for the third dose of diphtheria-tetanus-pertussis vaccine (DTP3) as the number of doses administered divided by the number of surviving infants. The district reports 12,400 DTP3 doses in the calendar year. The projected surviving-infant cohort is 12,000.

A household survey fielded in the same year sampled 180 children aged 12 to 23 months in the district, of whom 132 had received three doses. The survey design effect for this indicator is estimated at 1.6. The national DTP3 coverage from the same survey is 81.5%, with a between-district standard deviation of 6.0 percentage points.

The district health officer has to report a coverage figure to the region, and the region has to decide whether the district is on track for the national target.

### 6.2 The reporting system

The district has 42 facilities on the master facility list. Over the 12 months, 431 monthly reports were received.

| Quantity | Value | Denominator |
|---|---:|---|
| Expected reports, raw list | 504 | 42 facilities x 12 months |
| Facilities opened at mid-year | 3 | 6 months each |
| Facilities closed at mid-year | 2 | 6 months each |
| Expected reports, facility-life | 474 | 504 - 18 - 12 |
| Reports received | 431 | |
| Reports received on time | 358 | |
| Completeness, raw | 85.5% | 431 / 504 |
| Completeness, reconciled | 90.9% | 431 / 474 |
| Timeliness, of expected | 71.0% | 358 / 504 |
| Timeliness, of received | 83.1% | 358 / 431 |

The adjustment removes 18 facility-months for the three facilities that opened at mid-year (3 x 6) and 12 for the two that closed (2 x 6), a total of 30 facility-months from the 504. The numerator is unchanged, because no report was received from a facility that did not exist.

The raw completeness of 85.5% is below the WHO benchmark of 90%. The reconciled completeness of 90.9% is just above it. The difference is entirely the facility list, not the reporting behaviour. A district team that reports 85.5% will be asked to improve reporting; a team that reports 90.9% will be asked to improve timeliness. The same underlying data produce two different management conversations.

### 6.3 Three estimates of the same indicator

| Estimate | Value | 95% interval | What it measures |
|---|---:|---|---|
| Administrative | 103.3% | none | Doses reported / projected cohort |
| Survey, direct | 73.3% | 65.2% to 81.5% | Children with three doses / children surveyed |
| Model-based | 78.8% | 72.1% to 85.6% | Shrunk toward the national mean |

The administrative estimate is above 100%, which is the first diagnostic. The survey estimate is 30 points lower. The model estimate sits between them because it is pulled toward the national mean of 81.5%.

The three numbers are not three measurements of one quantity. The administrative estimate is a ratio of third doses to a projected cohort, and its population is the set of children who received a dose somewhere in the district's reporting system. The survey estimate is a ratio of surveyed children with three doses to surveyed children, and its population is the set of children in sampled households. The model estimate is a weighted average of the survey estimate and a regional mean, and its population is a modelled district.

A report that presents the three as a range has at least described the disagreement. A report that presents one as the truth has chosen a population without saying so.

The 30-point gap also has a practical use. It is the size of the correction that the routine system would need for its numerator or denominator to be consistent with the survey. If the correction is stable across years, it can be applied as a calibration factor. If it moves, the system is changing and the calibration is not safe.

### 6.4 Diagnostic one: a coverage above 100%

A coverage of 103.3% cannot be a true coverage. One of the three quantities is wrong: the numerator, the denominator, or the definition.

The numerator counts doses, not children. A child who receives a dose in two facilities is counted twice. A dose given to a child from a neighbouring district is counted in this district. A dose from a campaign that used a separate tally may be counted in both the campaign and the routine report.

The denominator is a projection. It assumes that the cohort in the district equals 2.5% of the projected population, that migration is zero, and that the census is current. A district with a large refugee settlement, a new industrial employer, or a boundary change will have a cohort that differs from the projection by more than 3%.

The definition is the third possibility. "Doses administered" is not "children vaccinated". A coverage indicator built from doses measures service volume, and a coverage indicator built from children measures population reach. The two are equal only if every child receives exactly three doses in the same district.

The diagnostic is to compare DTP3 with the first dose, DTP1. In a closed cohort, DTP1 must be at least DTP3, because every child who received three doses received a first dose. If the reported DTP3 exceeds DTP1, the two counts are drawing on different populations, or one of them is double-counted. A second diagnostic is the ratio of doses to the projected cohort over successive years: a ratio that rises while the survey estimate is flat is a numerator problem, not a coverage success.

In this district, 12,400 third doses against 12,000 children is arithmetically possible only if almost every child received all three doses and none was counted twice. The survey finds 73.3%, which is about 8,800 children. The gap of about 3,600 children is the size of the double count, the cross-border population, or the denominator error. It should be explained, not rounded.

### 6.5 Diagnostic two: the interval that ignores the design

The survey estimate of 73.3% has two possible intervals. Under simple random sampling, the standard error is the square root of 0.7333 x 0.2667 / 180 = 0.0330, and the 95% interval is 66.9% to 79.8%. With a design effect of 1.6, the standard error is 0.0417, and the interval is 65.2% to 81.5%.

The naive interval is 26% narrower. That matters for a decision. If the national target is 80%, the naive interval includes 80% only at its upper edge and the district looks clearly below target. The design-based interval includes 80% comfortably, and the district looks like it might be on track. The same data support opposite management conclusions depending on a variance assumption that is usually buried in a footnote.

The diagnostic is to check whether the reported standard error is design-based. If the survey documentation reports a design effect, use it. If it does not, estimate it from a prior survey or run the sensitivity in Section 7. A design effect of 1.0 in a clustered household survey is not a finding; it is a missing calculation.

The same issue appears in a different form when a district estimate is computed from a national survey. The survey's design effect is for the national estimate. A district subset has fewer clusters, and the design effect for the subset can be larger, because the between-cluster variance is a bigger share of a smaller sample. Using the national design effect for a district interval is therefore likely to understate the district's uncertainty.

### 6.6 The small-area estimates

The district is one of four in the region. The direct and model-based estimates are shown below. The model uses an empirical Bayes shrinkage toward the regional mean of 81.5%, with a between-district standard deviation of 6.0 percentage points.

| District | n | Direct | Direct 95% CI | Model | Model 95% CI | Weight on direct |
|---|---:|---:|---|---:|---|---:|
| A | 180 | 73.3% | 65.2-81.5 | 78.8% | 72.1-85.6 | 0.33 |
| B | 240 | 87.5% | 82.2-92.8 | 82.5% | 77.7-87.3 | 0.17 |
| C | 150 | 64.0% | 54.3-73.7 | 74.4% | 66.9-81.9 | 0.41 |
| D | 24 | 75.0% | 53.1-96.9 | 76.5% | 66.1-86.8 | 0.78 |

Three things happen in the table, and each is a lesson.

District D has the widest direct interval, from 53.1% to 96.9%, because it has 24 children. The model moves it only 1.5 points, to 76.5%, but narrows the interval to 66.1% to 86.8%. The narrowing is entirely the prior. If the prior is wrong, the interval is too narrow.

District C moves the most, from 64.0% to 74.4%, a shift of 10.4 points. It has a moderate sample and an estimate far below the mean, so the model attributes part of the gap to sampling noise. That may be correct, or it may hide a genuinely failing district. The model cannot tell the difference without covariates that explain why C is low.

The ranking changes. On the direct estimates the order is B (87.5), D (75.0), A (73.3), C (64.0). On the model estimates it is B (82.5), A (78.8), D (76.5), C (74.4). District D falls from second to third, and A rises from third to second, without any new data. A league table built on the model estimates would rank the districts differently from one built on the direct estimates, and neither is wrong.

### 6.7 The equity stratification

The survey also provides coverage by household wealth quintile.

| Quintile | Coverage | 95% CI |
|---|---:|---|
| Q1 poorest | 58% | 52-64 |
| Q2 | 66% | 60-72 |
| Q3 | 74% | 69-79 |
| Q4 | 83% | 78-88 |
| Q5 richest | 91% | 87-95 |

The absolute gap between the richest and poorest quintile is 33 percentage points, with a 95% interval of 26 to 40 points. The ratio is 1.57. The national average of 81.5% is consistent with a distribution in which the poorest fifth is 33 points below the richest fifth.

A second summary measure uses the whole distribution rather than the two ends. Regressing coverage on the midpoint of each quintile's cumulative population share gives a slope of 0.415, which is interpreted as a 41.5 percentage point difference between the bottom and the top of the wealth distribution. The slope uses all five points, so it is less sensitive to the extremes than the difference between Q1 and Q5, and it is the measure that the WHO Health Equity Assessment Toolkit reports as the slope index of inequality.

The equity finding changes the programme question. A district that is at 78.8% overall and 58% in the poorest quintile does not need a general coverage push. It needs a targeted one, and the targeting requires knowing where the poorest quintile lives, which the survey can describe at the regional level and the facility data cannot describe at all. The slope of 41.5 points is the quantity to track over time: a programme that raises every quintile by five points raises the average and leaves the slope unchanged, and a programme that raises the poorest quintile by five points reduces the slope. The average cannot distinguish the two.

### 6.8 The varied assumption: the prior mean

The model estimate for District A depends on the regional mean. Suppose the regional mean is not 81.5% but 70.0%, because the region contains several low-performing districts that were excluded from the earlier calculation.

With a mean of 70.0%, the weight on the direct estimate is unchanged at 0.33, and the model estimate becomes 0.733 + 0.67 x (0.700 - 0.733) = 0.711, or 71.1%. The estimate moves by 7.7 percentage points, and the conclusion changes: District A is now slightly above the regional mean rather than slightly below it.

The prior is not a neutral technical choice. It is a statement about the region, and it moves the district's rank. A small-area estimate should therefore report the prior, test its sensitivity, and show the direct estimate alongside the model estimate.

### 6.9 What the analysis does not show

The analysis does not show that coverage is 78.8%. It shows that three sources produce three different numbers, that the administrative number is not a coverage estimate, and that the survey number has an uncertainty that depends on a design effect and a prior.

It does not show why District C is low. The model shrinks C toward the mean, but the reason for the gap, whether it is distance, stockouts, or demand, is not in the data. That requires a service-readiness assessment or a qualitative study.

It does not show whether the district will reach the target next year. A single cross-section cannot estimate a trend, and a trend built on two survey rounds four years apart cannot detect an annual change.

Those gaps are the agenda for the next data collection, not a reason to withhold the estimate.

The estimate is also not a target. A district that is told to reach 90% coverage will report 90% coverage, because the reporting system is under the district's control and the survey is not. The target should be set on the survey where one exists, and on a data-quality-adjusted routine figure where one does not.

### 6.10 A third diagnostic: the facility that never reports

Aggregate completeness can hide a facility that never reports. If one facility submits zero reports for twelve months, it contributes 12 to the denominator and 0 to the numerator. If the analyst drops it as "non-reporting" and recomputes, completeness rises without a single new report, and the population it serves disappears from the district estimate.

The diagnostic is to compute completeness at the facility level and to look at the distribution, not only the mean. In this district, 36 of 42 facilities reported at least 10 months, four reported between 6 and 9 months, and two reported fewer than 3 months. The two low reporters serve about 9% of the district population. Their absence from the numerator is not random: they are the most remote facilities, and the population they serve is the population least likely to be vaccinated.

A completeness figure that counts a silent facility as a missing report is honest. A completeness figure that drops it is not, because the drop changes the population that the estimate describes.

### 6.11 What the district should actually report

The district has to send one number to the region. It should send a number, a range, and a source, and it should say what the number does not cover.

| Item | Value | Note |
|---|---|---|
| DTP3 coverage, recommended | 73.3% | Survey, direct estimate |
| Range | 65.2% to 81.5% | Design effect 1.6 |
| Alternative estimate | 78.8% | Model, shrunk to regional mean |
| Administrative ratio | 103.3% | Doses over projected cohort, not coverage |
| Completeness | 90.9% | Reconciled facility list |
| Population not covered | 9% | Two low-reporting facilities |

The recommended value is the survey, because it measures children rather than doses and it has a design-based interval. The administrative ratio is reported because the national system will report it anyway, and it needs an explanation rather than a silence. The model estimate is reported as a check, not as the headline.

### 6.12 The varied assumption applied to a second district

The prior also changes District C. With a regional mean of 81.5%, C's model estimate is 74.4%. With a regional mean of 70.0%, the same weight of 0.41 on the direct estimate gives 0.640 + 0.59 x (0.700 - 0.640) = 0.676, or 67.6%.

The gap between A and C therefore changes from 4.4 points (78.8 against 74.4) to 3.5 points (71.1 against 67.6). The ranking is stable, but the size of the gap, which is the quantity a programme uses to decide where to intervene, moves by almost a point on a prior choice.

The ranking is stable in this example, which is not guaranteed. A district with a small sample and an extreme direct estimate can change rank when the prior moves, and the change will look like a data revision even though no data changed.

This is the practical argument for reporting the direct estimates in an appendix. The model estimates are useful for ranking with less noise, and the direct estimates are what the model started from. A reader who sees only the model estimates cannot tell how much of the number is data.

## 7. Code, query, or calculation

The script computes reporting completeness, three coverage estimates, the empirical Bayes small-area estimates, and the equity gap. It uses only the Python standard library, and every input is stated so the arithmetic can be checked by hand.

```python
"""District immunisation data: reporting completeness, coverage, small-area
estimation, and equity. Standard library only. Illustrative values.
"""

import math

# --- 1. Reporting completeness -------------------------------------------
FACILITIES = 42
MONTHS = 12
EXPECTED_RAW = FACILITIES * MONTHS
RECEIVED = 431
ON_TIME = 358
OPENED_MIDYEAR = 3     # six months each in the window
CLOSED_MIDYEAR = 2     # six months each in the window
EXPECTED_ADJ = EXPECTED_RAW - OPENED_MIDYEAR * 6 - CLOSED_MIDYEAR * 6

print(f"expected reports (raw list)      {EXPECTED_RAW}")
print(f"expected reports (facility-life) {EXPECTED_ADJ}")
print(f"received {RECEIVED} -> completeness {RECEIVED / EXPECTED_RAW:.1%} raw, "
      f"{RECEIVED / EXPECTED_ADJ:.1%} adjusted")
print(f"on time {ON_TIME} -> timeliness {ON_TIME / EXPECTED_RAW:.1%} of expected, "
      f"{ON_TIME / RECEIVED:.1%} of received")

# --- 2. Coverage: three estimates of the same quantity --------------------
DOSES = 12400          # DTP3 doses reported by facilities, 12 months
COHORT = 12000         # surviving infants, projected from census
print(f"administrative coverage {DOSES / COHORT:.1%} "
      f"({DOSES} doses / {COHORT} children)")

# Household survey, children aged 12-23 months.
n, x = 180, 132
p = x / n
DEFF = 1.6
se_naive = math.sqrt(p * (1 - p) / n)
se_direct = math.sqrt(p * (1 - p) / n * DEFF)
z = 1.96
print(f"survey coverage {p:.1%}; naive 95% CI "
      f"{p - z * se_naive:.1%}-{p + z * se_naive:.1%}; "
      f"design-effect 95% CI {p - z * se_direct:.1%}-{p + z * se_direct:.1%}")

# --- 3. Empirical Bayes small-area estimates ------------------------------
NATIONAL = 0.815
TAU = 0.060            # between-district SD of true coverage
TAU2 = TAU ** 2
districts = [
    ("A", 132, 180),
    ("B", 210, 240),
    ("C", 96, 150),
    ("D", 18, 24),     # small district, wide direct interval
]
print(f"\n{'dist':>4} {'n':>5} {'raw':>7} {'direct 95% CI':>18} "
      f"{'shrunk':>7} {'shrunk 95% CI':>18} {'weight':>7}")
for name, xx, nn in districts:
    pp = xx / nn
    v = pp * (1 - pp) / nn * DEFF
    w = v / (v + TAU2)
    post = pp + (1 - w) * (NATIONAL - pp)
    pv = v * TAU2 / (v + TAU2)
    print(f"{name:>4} {nn:>5} {pp:>6.1%} "
          f"{pp - z * math.sqrt(v):>8.1%}-{pp + z * math.sqrt(v):<8.1%} "
          f"{post:>7.1%} "
          f"{post - z * math.sqrt(pv):>8.1%}-{post + z * math.sqrt(pv):<8.1%} "
          f"{w:>7.2f}")

# --- 4. Equity stratification -------------------------------------------
quintiles = [("Q1 poorest", 0.58, 0.031), ("Q2", 0.66, 0.030),
             ("Q3", 0.74, 0.028), ("Q4", 0.83, 0.026),
             ("Q5 richest", 0.91, 0.022)]
print("\ncoverage by wealth quintile")
for label, cov, se in quintiles:
    print(f"  {label:<11} {cov:.0%} (95% CI {cov - z * se:.0%}-{cov + z * se:.0%})")
gap = quintiles[-1][1] - quintiles[0][1]
ratio = quintiles[-1][1] / quintiles[0][1]
se_gap = math.sqrt(quintiles[-1][2] ** 2 + quintiles[0][2] ** 2)
print(f"  difference {gap * 100:.0f} pp (95% CI {gap * 100 - z * se_gap * 100:.0f}"
      f"-{gap * 100 + z * se_gap * 100:.0f}); ratio {ratio:.2f}")

# --- 5. What a different design effect does ------------------------------
print("\nsensitivity of the direct interval to the design effect")
for deff in (1.0, 1.6, 2.5, 3.0):
    s = math.sqrt(p * (1 - p) / n * deff)
    print(f"  DEFF {deff:.1f}: half-width {z * s * 100:.1f} pp, "
          f"CI {p - z * s:.1%}-{p + z * s:.1%}")
```

Expected output:

```text
expected reports (raw list)      504
expected reports (facility-life) 474
received 431 -> completeness 85.5% raw, 90.9% adjusted
on time 358 -> timeliness 71.0% of expected, 83.1% of received
administrative coverage 103.3% (12400 doses / 12000 children)
survey coverage 73.3%; naive 95% CI 66.9%-79.8%; design-effect 95% CI 65.2%-81.5%

dist     n     raw      direct 95% CI  shrunk      shrunk 95% CI  weight
   A   180  73.3%    65.2%-81.5%      78.8%    72.1%-85.6%       0.33
   B   240  87.5%    82.2%-92.8%      82.5%    77.7%-87.3%       0.17
   C   150  64.0%    54.3%-73.7%      74.4%    66.9%-81.9%       0.41
   D    24  75.0%    53.1%-96.9%      76.5%    66.1%-86.8%       0.78

coverage by wealth quintile
  Q1 poorest  58% (95% CI 52%-64%)
  Q2          66% (95% CI 60%-72%)
  Q3          74% (95% CI 69%-79%)
  Q4          83% (95% CI 78%-88%)
  Q5 richest  91% (95% CI 87%-95%)
  difference 33 pp (95% CI 26-40); ratio 1.57

sensitivity of the direct interval to the design effect
  DEFF 1.0: half-width 6.5 pp, CI 66.9%-79.8%
  DEFF 1.6: half-width 8.2 pp, CI 65.2%-81.5%
  DEFF 2.5: half-width 10.2 pp, CI 63.1%-83.5%
  DEFF 3.0: half-width 11.2 pp, CI 62.1%-84.5%
```

The output contains four numbers that should be read together.

The completeness pair, 85.5% and 90.9%, is the same numerator against two denominators. The gap is the facility list, and it is larger than the distance to the 90% benchmark. A team that does not reconcile the list will misclassify a district that meets the benchmark as one that does not.

The administrative coverage of 103.3% is not a coverage estimate. It is the ratio of doses to a projected cohort, and the projection is the weak link. The script prints it because it is the number in the national system, not because it is the best estimate.

The direct interval widens from 6.5 to 11.2 percentage points of half-width as the design effect moves from 1.0 to 3.0. The design effect is not a technicality. It determines whether a district is classified as on track.

The shrinkage weight tells you how much of the model estimate is data. District D, with 24 children, has a weight of 0.78 on the prior, so only 22% of its model estimate comes from its own sample. That is a number a district officer should see before being ranked.

### 7.1 Adapting the script to a real dataset

The script uses fixed values so that the arithmetic can be checked. A real analysis replaces each block with a query against the routine system or the survey recode file.

| Block | Real source | What to check first |
|---|---|---|
| Completeness | DHIS2 analytics, expected reports | Facility list version and reporting deadline |
| Administrative coverage | Data element for doses, population projection | Whether the element is doses or children |
| Survey estimate | DHS or MICS recode, children aged 12-23 months | Weight variable and design variables |
| Design effect | Survey documentation or a prior survey | Whether it is for this indicator and domain |
| Small-area inputs | Direct estimates by district, with variance | Whether the district boundaries match |
| Equity | Wealth quintile and weights | Whether the quintile is national or district |

Two adjustments are needed in almost every real case. The first is the survey weight: the recode file has a weight variable, and an unweighted tabulation describes the sample, not the population. The second is the denominator vintage: the population projection has a base year, and a projection based on a census more than ten years old carries an uncertainty that should be reported.

The script's empirical Bayes step assumes a normal prior for the district means. That is a convenience, not a finding. A real analysis should test whether the distribution of district estimates is consistent with the prior, and should report the sensitivity to the between-district standard deviation.

## 8. Data, standards, and regulatory context

### 8.1 The DHIS2 data model

DHIS2 is the platform used by ministries of health in more than 75 low- and middle-income countries at national scale, covering about 3.2 billion people according to the project's 2025 material. It has two data models, and the distinction matters for every analysis built on it.

The aggregate model stores a value against four dimensions: a data element (what), a period (when), an organisation unit (where), and a category option combination (the disaggregation, such as age or sex). An attribute option combination carries additional dimensions, such as the partner reporting the value. The composite key is data element, period, organisation unit, category option combination, and attribute option combination. An extract that omits the category option combination will silently sum across age groups or sexes.

The Tracker model stores individual-level records: a tracked entity instance, an enrollment in a program, events within program stages, and data values within events. It is used for longitudinal case management, such as maternal and child health, HIV care, or disease surveillance. Tracker data can be aggregated into the aggregate model for national reporting, and the aggregation rule is part of the metadata, not of the analysis.

| Feature | Aggregate | Tracker |
|---|---|---|
| Unit | Facility-month | Person or case |
| Key | Data element, period, org unit, category option combo | Tracked entity, enrollment, event |
| Typical use | National reporting | Case management |
| Privacy | Low, if counts are small | High, individual records |
| Analysis risk | Double counting across categories | Linkage error and duplicate records |

### 8.2 Survey standards

The Demographic and Health Surveys program, funded by the United States Agency for International Development and implemented by ICF, publishes model questionnaires, a biomarker manual, and a guide to statistics. The eighth phase (DHS-8) questionnaires cover households, women, men, and biomarkers, with anthropometry and haemoglobin measurement. The survey produces standard recode files at the household, person, and child level, and a geographic file with cluster coordinates.

The Multiple Indicator Cluster Surveys, run by UNICEF, provide a comparable household survey programme with a different questionnaire and a different sampling design. The two are not interchangeable, and a comparison across them has to account for differences in the indicator definitions.

Both surveys are designed for national and, in some cases, regional estimates. A DHS is generally powered to the first administrative level, not the second. A district estimate from a DHS is a small-area estimate, whether or not the analyst calls it one.

### 8.3 Facility registries

The World Health Organization's Master Facility List Resource Package (2019) sets out the minimum content for a facility registry. It distinguishes a signature domain, which is the required minimum, from a service domain, which is optional.

| Domain | Elements |
|---|---|
| Signature | Unique identifier, name, type, ownership, address, administrative areas, coordinates, operational status, record date |
| Service | Services offered, staff by type, beds and cots |

The unique identifier should be random or sequential and should never embed administrative information, because administrative units change. Coordinates should be in decimal degrees in the WGS84 datum. The registry should record the date of the record, because a facility list without a date cannot be reconciled against a reporting period.

### 8.4 Civil registration and vital statistics

The WHO estimates that nearly 40% of the world's deaths are not registered, and that death registration coverage in the African Region is about 10%. Birth registration coverage in the African Region is about 44%. These are not small gaps at the margin. They mean that for a large share of the world's population, the denominator for a mortality rate and the numerator for a cause-of-death analysis do not exist in the civil register.

The UN Principles and Recommendations for a Vital Statistics System set out the standard for registration, and the Sustainable Development Goals include legal identity and birth registration (target 16.9) and the completeness of death registration (indicator 17.19.2). A country's CRVS completeness is therefore both a statistical and a development indicator.

### 8.5 Classifications and verbal autopsy

Cause of death is coded with the International Classification of Diseases (ICD), now in its 11th revision. In settings where most deaths occur outside a health facility and without a medical certificate, the cause is assigned by verbal autopsy: a structured interview with a caregiver, analysed by a probabilistic or machine-learning algorithm.

The WHO published the 2022 Verbal Autopsy Instrument, with questionnaires for neonates, children, and adults. The instrument is designed for use with InterVA-5, InSilicoVA, and the SmartVA/Tariff tools. The algorithms assign a probability distribution over a fixed cause list, and the population-level cause-specific mortality fraction is generally more reliable than the individual-level assignment. A verbal autopsy is not a diagnosis, and a verbal autopsy dataset is not a vital registration system.

### 8.6 Interoperability

OpenMRS is an open-source medical record system built around a concept dictionary, encounters, and observations. It is used in low-resource clinical settings and can exchange data using Fast Healthcare Interoperability Resources (FHIR) through modules. The aggregate and individual systems are connected by exports, not by a shared standard, and the export is where the reconciliation problems in Section 5 arise.

### 8.7 Legal instruments

| Instrument | Adopted | Status | What it constrains |
|---|---|---|---|
| International Health Regulations (IHR 2005), as amended | 2005; amendments 1 June 2024 | Binding on 196 States Parties; amendments in force 19 September 2025 | Surveillance, notification, and response capacity |
| WHO Pandemic Agreement | 20 May 2025 | Not yet in force; requires 60 ratifications | Pathogen access and benefit-sharing, with the annex still under negotiation |
| Nagoya Protocol | 2010; in force 12 October 2014 | Binding on parties | Access to genetic resources and benefit-sharing |
| African Union Malabo Convention | 27 June 2014; in force 8 June 2023 | Binding on parties | Cyber security and personal data protection |
| Kenya Data Protection Act | 2019 | Binding in Kenya | Health data as sensitive personal data |
| India Digital Personal Data Protection Act | 2023 | Binding in India | Processing of digital personal data |
| EU General Data Protection Regulation | 2016/679 | Binding in the EU | Processing of personal data |

The IHR amendments adopted at the World Health Assembly in June 2024 introduced a "pandemic emergency" alert level, national IHR authorities, and a coordinating financial mechanism. They entered into force on 19 September 2025 for most States Parties. The Pandemic Agreement adopted in May 2025 creates a pathogen access and benefit-sharing system in Article 12, with 20% of real-time production of pandemic-related products made available to WHO, at least 10% as donation. The operational annex is still being negotiated, so the instrument's practical effect on data sharing is not yet settled.

The Nagoya Protocol is often cited as governing pathogen data. Its application to pathogens is contested, because the protocol was written for genetic resources and many countries have treated pathogen sharing through separate arrangements, including the Pandemic Influenza Preparedness Framework. The 2025 Pandemic Agreement is an attempt to create a dedicated regime. An analyst who assumes that Nagoya settles the question of pathogen data sharing is ahead of the law.

### 8.8 Binding requirement versus convention

| Requirement | Source | Status | What it constrains |
|---|---|---|---|
| Notification of a PHEIC | IHR 2005 | Binding | Timing and content of notification |
| Health data protection | National law, e.g. Kenya 2019 | Binding | Lawful basis, transfer, retention |
| Birth and death registration | National civil registration law | Binding | Legal record of vital events |
| Facility registry content | WHO MFL Resource Package | Guidance | Minimum data content |
| Survey methodology | DHS-8 and MICS manuals | Methodological standard | Questionnaire and sampling |
| Data quality benchmarks | WHO Data Quality Review | Guidance | Completeness, timeliness, consistency |
| Cause-of-death coding | ICD-11 | Classification standard | Coding and comparability |

The distinction matters for a data-sharing agreement. A ministry's approval is necessary but not always sufficient: the national data protection law may require a separate basis for transfer, and the research ethics approval may impose conditions that the ministry cannot waive.

### 8.9 What the standards do not cover

The standards define the data model, the classification, and the minimum content of a registry. They do not define the analysis.

Nothing in the DHIS2 documentation says which denominator to use for a coverage indicator, how to treat a facility that reports zero for a year, or how to combine a survey and a routine estimate. The WHO Data Quality Review defines completeness and consistency benchmarks; it does not define an adjudication rule. The MFL Resource Package defines the facility record; it does not define the catchment.

Those are analytical choices, and they are where the estimates are made. The standards make the data comparable; the analyst makes them interpretable. A methods note that cites the standards but not the choices has documented the plumbing and not the argument.

## 9. Industry practice and case evidence

### 9.1 What is documented

**DHIS2 is the dominant national platform.** The project reports use by ministries of health in more than 75 LMICs at national scale, covering about 3.2 billion people, with a wider footprint of 130 countries across sectors. It is a digital public good, developed by the HISP Centre at the University of Oslo with a network of regional partners. These figures come from the project's own 2025 material, and they vary by page and by definition of "use".

**Household surveys remain the reference for population-based indicators.** The DHS program has fielded surveys in more than 90 countries since the 1980s, and MICS has done the same under UNICEF. The surveys are the only source that measures the population rather than the service contact, and their sample sizes are designed for national and regional estimates.

**Phone surveys expanded the toolkit, with known bias.** The World Bank's high-frequency phone surveys during the COVID-19 pandemic used re-contact samples from existing household surveys, telecom lists, and random digit dialling. A review of the first wave found coverage rates from 73% in Malawi to 99% in Nigeria and response rates from 60% in Ethiopia to 93% in Uganda. Respondents were disproportionately male, at 62% to 78% across countries. Weighting reduced but did not eliminate the bias.

**Geospatial data fill the denominator gap.** WorldPop and GRID3 produce gridded population estimates at about 100-metre resolution, combining census counts with satellite imagery, building footprints, and covariates. They are used to estimate catchment populations, to map facilities, and to plan campaigns. They are estimates, not censuses, and their uncertainty is not always carried into the coverage calculation.

**Equity analysis has a standard toolkit.** The WHO Health Equity Assessment Toolkit implements disaggregated analysis and a set of summary measures, including difference, ratio, concentration index, and slope index of inequality, with confidence intervals. It is free and can be used offline.

### 9.2 What is inference

The following statements are inferences from the documented material, not documented findings.

Most national dashboards in LMICs report administrative coverage without a denominator adjustment. The incentive structure favours a single number that can be compared across districts, and the denominator adjustment is technical, contested, and slow. This is an inference from the shape of the reporting systems and from the persistence of coverage figures above 100%.

The gap between survey and administrative estimates is often interpreted politically rather than statistically. A ministry may prefer the administrative figure because it shows progress, and a donor may prefer the survey because it is independent. The analyst's contribution is to make the gap explicable, not to choose a side.

Small-area estimates are increasingly used for district planning, but the uncertainty is often dropped in the visualisation. A map with five colour classes implies a precision that the underlying model does not have.

### 9.3 What the evidence base lacks

There is no global gold standard for district-level coverage. The surveys are representative at a coarser level, the routine system is complete only for the facilities that report, and the civil register is incomplete for the events that matter most. The methods in this topic are a set of disciplined approximations, not a solution to the measurement problem.

The evidence on what improves data quality is also thin. Feedback, supervision, and audit are widely used and weakly evaluated. A programme that claims to improve data quality should measure completeness, timeliness, and consistency before and after, and should be honest about the confounders.

### 9.4 How a district actually produces the number

The national dashboard is the end of a chain that runs through a clinic clerk, a facility in-charge, a district officer, and a regional review meeting. Each link has its own constraints, and the constraints explain most of the data quality.

The clinic clerk tallies by hand, often at the end of a shift, from a register that was not designed for tallying. The facility in-charge completes a monthly summary form, checks it against the register if there is time, and submits it through a district office, a mobile app, or a paper form. The district officer enters the data into DHIS2, runs the validation rules, and follows up on the failures. The regional review meeting compares districts, and the comparison creates an incentive to explain a low number rather than to correct it.

| Link | Constraint | Typical effect on the data |
|---|---|---|
| Clinic register | Time, staffing, legibility | Under-recording of contacts |
| Monthly summary | Arithmetic, form design | Transcription errors |
| District entry | Connectivity, workload | Late or missing reports |
| Validation | Rule design, follow-up capacity | Errors carried forward |
| Regional review | Comparison and reputation | Explanation instead of correction |

A data-quality intervention that targets only the district officer misses the clinic and the review meeting, which are where the incentives and the workload actually sit.

### 9.5 What a ministry should ask a research partner

A research partnership can bring methods and capacity. It can also extract data, publish, and leave. The questions below are the minimum for a partnership that leaves the ministry able to reproduce the result.

| Question | Why it matters |
|---|---|
| Who owns the data and the derived dataset | Determines control after the project |
| Where will the data be stored | Determines the lawful basis and the risk |
| Can the ministry reproduce the analysis | Determines whether the result can be defended |
| What is returned to the ministry, and when | Determines whether the partnership builds capacity |
| Who reviews the publication | Determines whether the ministry is surprised |
| What happens at the end of the project | Determines whether the system survives |

A partnership that cannot answer the third question has not built capacity. It has built a dependency.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Coverage above 100% reported as coverage | The number is explained away | Doses counted, not children | Overstates reach | Denominator and definition check |
| Completeness confused with coverage | Both on one dashboard | Different numerators and denominators | Wrong management action | Label and define each |
| Facility list not reconciled | Completeness moves when the list changes | Openings and closures unrecorded | Wrong expected count | Reconcile against the MFL |
| Zero treated as zero cases | Zero reports in a stockout month | Supply constraint | Model learns the supply chain | Classify zeros |
| Parallel reporting systems | Two numbers for one facility-month | Donor and government systems | Double counting | Name the system of record |
| Survey treated as routine | Quarterly district estimates from a DHS | Sample powered for regions | False precision | Small-area methods |
| Design effect ignored | Interval narrower than the survey | Clustering and weighting | Wrong significance | Design-based variance |
| Weights ignored | Sample looks like the population | Differential selection | Biased estimate | Use survey weights |
| Small-area model over-shrinks | A failing district looks average | Prior dominates small samples | Hidden inequality | Report direct and model estimates |
| Prior mean unstated | Ranking changes without new data | Shrinkage depends on the prior | Political dispute | Publish the prior and sensitivity |
| Denominator vintage ignored | Coverage jumps after a census | Rebasing | False trend | Record the vintage |
| Survey and admin years misaligned | Comparison of different periods | Recall and calendar | Spurious gap | Align periods |
| Cluster coordinates treated as exact | Facility catchment joins | Deliberate displacement | Misclassified exposure | Respect the displacement |
| National average used for a district | One number for all | Heterogeneity | Misallocated resources | Stratify and model |
| Data transferred without a legal basis | Server in another country | Sovereignty and law | Project failure, legal risk | Data-sharing agreement first |
| Model deployed across a boundary change | District counts jump | Administrative reorganisation | Broken time series | Freeze geography for analysis |
| Survey recall period ignored | Trend changes at a round number | Five-year recall window | False trend | Align the reference period |
| Catchment treated as the district | Cross-border patients | Facility serves a wider area | Wrong denominator | Catchment mapping |
| Weighted and unweighted tables mixed | Totals do not reconcile | Different populations | Inconsistent estimates | One weight per table |

Five red flags deserve emphasis.

**A coverage figure that is not a ratio of the right two quantities.** The most common version is doses over a projected cohort, reported as if it were children over children. The second most common is service contacts over a total population, reported as if it were the eligible population. Both produce a number that looks like coverage and is not.

**A confidence interval that covers only sampling error.** The interval from a survey with a design effect of 1.6 is 26% wider than the simple-random-sampling interval. If the denominator is also uncertain, the true uncertainty is wider still. An interval that is narrower than the design effect allows is a claim the data do not support.

**A district ranking with no intervals.** A league table of 40 districts, ranked on a survey indicator with 50 children per district, is mostly ranking noise. The top and bottom districts may be distinguishable; the middle is not.

**A data-sharing arrangement that the ministry cannot audit.** If the ministry cannot see the data, cannot reproduce the analysis, and cannot stop the transfer, the project has a legal and ethical problem regardless of its technical quality.

**A map that shows more precision than the estimate has.** A choropleth with five classes, built from district estimates with intervals spanning two classes, is a visual claim that the data cannot support. The map should show the uncertainty, or the text should.

When one of these signs appears, the first check is the denominator, not the model. The table is a triage order.

| Symptom | First check | Likely cause |
|---|---|---|
| Coverage above 100% | Numerator definition and catchment | Doses counted as children |
| Completeness jumps after a list update | Facility-life denominator | Openings and closures |
| Survey and routine diverge by year | Reference periods | Recall versus calendar |
| District ranking changes every round | Sample size and shrinkage | Ranking noise |
| A district has no estimate | Facility list and reporting | Silent facility |
| Trend reverses at a census year | Projection vintage | Rebasing |

The order matters because teams reach for the model first. The model is usually the most expensive thing to change and the least likely to be the cause.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The facility list is complete and current | Expected-report denominator | Openings, closures, and transfers | Reconcile with the MFL and supervision |
| The catchment matches the administrative area | Coverage denominator | Cross-border and cross-district use | Catchment mapping |
| The census projection is current | Population denominator | Migration and boundary change | Compare with a recent census or survey |
| Reports measure service contacts | Coverage numerator | Doses counted rather than children | Dose-per-child ratio |
| Zeros are true zeros | Incidence and coverage | Stockouts and closures | Cross-check with stock records |
| The survey sample is representative | Population estimates | Undercoverage and nonresponse | Compare with the census |
| The design effect is known | Variance | Complex design | Use the survey documentation |
| The wealth index is comparable | Equity analysis | Different asset lists | Use within-survey ranks only |
| The small-area model is correctly specified | District estimates | Prior misspecification | Report direct and model estimates |
| The prior mean is credible | Shrinkage | Regional heterogeneity | Sensitivity analysis |
| The periods are aligned | Any comparison | Recall and calendar differences | Document the reference period |
| Cluster displacement is understood | Spatial joins | Confidentiality offset | Use the documented offset |
| The legal basis for transfer exists | Data sharing | National law and ethics | Legal review before extraction |
| The model transfers across settings | Deployment | Different facility level or protocol | External validation |
| The recall period is consistent | Trend analysis | Different survey rounds | Compare questionnaires |
| The wealth index ranks comparable groups | Equity | Different asset lists | Use within-survey ranks |
| The target population is stable | Coverage trend | Migration and boundary change | Use a range of projections |
| The reporting incentive does not change | Any comparison | A new ranking or target | Compare data-quality metrics |

The limits are as important as the assumptions.

The methods estimate what the systems measured. They cannot recover the people who never contacted a facility, the deaths that were never registered, or the cases that were never tested. In a setting where 40% of deaths are unregistered, a mortality estimate from CRVS describes the registered 60%, and the direction of the bias depends on who is missing.

The equity analysis is limited by the stratifier. A wealth quintile built from household assets does not capture everything that determines access, and a district-level analysis cannot see the inequality within a district. The choice of stratifier is a statement about which inequality matters, and it should be argued rather than defaulted.

The small-area estimates are limited by the model. The interval is only as good as the prior, and the prior is usually chosen for convenience. A model that shrinks a district toward a regional mean will hide a genuinely failing district unless the direct estimate is reported alongside.

### 11.1 What the estimate cannot say

The estimate describes the people who were counted. It does not describe the people who were not.

A coverage survey samples households that exist and are reachable. It under-samples nomadic populations, people in informal settlements, and households that are absent during the survey. Those groups are often the least served, so the survey estimate is more likely to be too high than too low.

A routine system counts contacts at facilities that report. It misses the facilities that do not report and the people who do not attend. The direction of the bias depends on whether non-attendance is concentrated among the poor, which it usually is, and whether the non-reporting facilities are in remote areas, which they usually are.

A civil registration system counts registered events. It misses home deaths, early neonatal deaths, and deaths in populations that do not interact with the registration office. The missing deaths are not a random sample of deaths, and a mortality rate built on the registered deaths can be biased in either direction depending on cause and age.

A small-area model interpolates between the data it has. It cannot create information about a district with no sample, and it cannot detect a change that occurred after the survey.

The honest statement is not "coverage is 73.3%". It is "coverage among surveyed children in District A in the survey year was 73.3%, with a 95% interval of 65.2% to 81.5%, and the estimate is likely to be above the coverage among children in the least accessible households." That sentence is longer, less quotable, and more useful.

## 12. Exercises

### Exercise 1. Source taxonomy (recall, 20 minutes)

**Deliverable:** a table classifying five data sources (DHIS2 aggregate, DHS, CRVS, a facility registry, a phone survey) by population, time structure, primary use, and dominant error.

**Acceptance criteria:** each source has a stated population and a stated error direction; no source is described as a gold standard; the table distinguishes research data from operational data.

**Time:** 20 minutes.

### Exercise 2. Indicator specification (application, 30 minutes)

**Deliverable:** a specification for "effective coverage of hypertension treatment" in a named district, with numerator, denominator, period, geography, stratification, primary source, and the most likely failure point.

**Acceptance criteria:** all six elements are present; the denominator is justified; the failure point is specific to the setting; the specification can be operationalised with the stated sources.

**Time:** 30 minutes.

### Exercise 3. Completeness and coverage (calculation, 35 minutes)

**Deliverable:** for a district with 30 facilities, 12 months, 312 reports received, 270 on time, 2 facilities opened in July, and 1 closed in April, compute raw and reconciled completeness and both timeliness figures. Then compute coverage from 9,800 doses and a projected cohort of 9,000, and state whether the result is a coverage estimate.

**Acceptance criteria:** the denominators are stated for every figure; the facility-life adjustment is shown; the coverage result is interpreted correctly, including the dose-per-child issue.

**Time:** 35 minutes.

### Exercise 4. Facility reconciliation (application, 35 minutes)

**Deliverable:** a reconciliation plan for a facility list of 120 sites that disagrees with the DHIS2 hierarchy on 9 sites, including the matching rule, the treatment of each discrepancy type, and the effect on the expected-report denominator.

**Acceptance criteria:** at least four discrepancy types are handled; the matching rule uses coordinates and names; the effect on completeness is quantified.

**Time:** 35 minutes.

### Exercise 5. Small-area estimation (calculation, 45 minutes)

**Deliverable:** for three districts with direct estimates of 62%, 78%, and 88% and sample sizes of 40, 120, and 260, compute the empirical Bayes estimates against a national mean of 75% and a between-district SD of 5 points. Report the weight on the direct estimate and both intervals.

**Acceptance criteria:** the variance formula is shown; the shrinkage weight is interpreted; the district whose estimate moves most is identified and explained.

**Time:** 45 minutes.

### Exercise 6. Equity analysis (application, 35 minutes)

**Deliverable:** a coverage-by-quintile table with confidence intervals, the absolute gap, the ratio, and a two-sentence interpretation for a programme manager.

**Acceptance criteria:** the intervals are computed; the gap has an interval; the interpretation names a target group and a decision; no causal claim is made.

**Time:** 35 minutes.

### Exercise 7. Triangulation (critique, 40 minutes)

**Deliverable:** a written adjudication of three estimates for the same district: administrative 96%, survey 71% (95% CI 63 to 79), and a model at 80%. State the recommended value, the range, and what would change your mind.

**Acceptance criteria:** each source's bias direction is named; the recommendation is not an average; at least two pieces of evidence would change the conclusion; the uncertainty is stated.

**Time:** 40 minutes.

### Exercise 8. Data sovereignty (design, 40 minutes)

**Deliverable:** a data-sharing agreement outline for a multi-country study that will analyse facility data centrally, covering lawful basis, minimisation, storage location, access control, publication, and the return of results to each ministry.

**Acceptance criteria:** at least two national laws are considered; the ministry's audit rights are explicit; the agreement addresses what happens at the end of the study; the analysis plan is reproducible from the ministry's side.

**Time:** 40 minutes.

### Exercise 9. Source selection (recall, 20 minutes)

**Deliverable:** for a decision on where to place a new immunisation outreach site, name the two sources you would use, the indicator each provides, and the error each carries.

**Acceptance criteria:** the sources are matched to the decision; the errors are named with their direction; no source is described as a gold standard.

**Time:** 20 minutes.

| Exercise | Type | Main skill |
|---|---|---|
| 1 | Recall | Source taxonomy |
| 2 | Application | Indicator specification |
| 3 | Calculation | Completeness and coverage |
| 4 | Application | Facility reconciliation |
| 5 | Calculation | Small-area estimation |
| 6 | Application | Equity |
| 7 | Critique | Triangulation |
| 8 | Design | Data sovereignty |
| 9 | Recall | Source selection |

### Submission standard

An indicator specification must have a numerator, a denominator, a period, a geography, and a source. A completeness figure must state whether the denominator is the raw list or the facility-life list. A coverage figure above 100% must be explained, not rounded. A small-area estimate must be reported with its direct counterpart and its prior. A data-sharing plan must name the law that authorises the transfer.

## 13. Self-check answers

**1. What is the difference between routine data and survey data?** Routine data are a by-product of service delivery, continuous, and complete only for the people who contact the system. Survey data are collected for a question, episodic, and representative of a defined population.

**2. Why is a DHS not a monitoring system?** It is fielded every few years, powered for national and regional estimates, and based on a sample. It cannot detect quarterly change or produce a reliable district estimate on its own.

**3. What are the five elements of an indicator specification?** Numerator, denominator, period, geography, and source, with stratification as a sixth where relevant.

**4. Why does the choice of denominator change the conclusion?** The numerator is usually measured; the denominator is usually projected. A projection that is too small produces coverage above 100%, and one that is too large produces apparent underperformance.

**5. Compute completeness for 431 reports against 504 expected and against 474 expected.** 431 / 504 = 85.5%; 431 / 474 = 90.9%.

**6. Why is coverage of 103.3% a specification error?** A true coverage cannot exceed 100%. The ratio is measuring doses against a projected cohort, not children against children, or the catchment does not match the projection.

**7. How does a facility list affect completeness?** The expected-report denominator is the facility-month. If facilities open, close, or are duplicated, the denominator is wrong, and completeness measures the list rather than the reporting.

**8. What is the design effect, and what does 1.6 mean?** It is the ratio of the variance under the actual design to the variance under simple random sampling. A value of 1.6 means the variance is 60% larger, and the standard error is about 26% larger.

**9. Compute the design-based interval for 132 of 180 with a design effect of 1.6.** p = 0.7333, SE = sqrt(0.7333 x 0.2667 / 180 x 1.6) = 0.0417, interval 65.2% to 81.5%.

**10. Compute the absolute gap and ratio for 58% and 91%.** Gap = 33 percentage points; ratio = 1.57.

**11. Why is triangulation not averaging?** Averaging two biased estimates does not remove either bias. Triangulation requires deciding which source is more credible for this indicator, place, and period, and stating why.

**12. What is the difference between a direct and a model-based small-area estimate?** A direct estimate uses only the area's own sample. A model-based estimate borrows strength from other areas through a prior or covariates, reducing variance and introducing model dependence.

**13. What is data sovereignty?** The principle that data are governed by the laws and institutions of the country where they were generated.

**14. Name two national data protection laws relevant to health data.** Kenya's Data Protection Act 2019 and India's Digital Personal Data Protection Act 2023. The EU General Data Protection Regulation is a third.

**15. Why is a zero not always a zero?** A zero can mean no cases, a stockout, a closure, or a missing report. The meaning has to be classified from another source before the zero is analysed.

**16. Why does the prior matter in a small-area estimate?** The model estimate is a weighted average of the direct estimate and the prior. A different prior mean moves the estimate, and for a small district the prior can carry most of the weight.

**17. What is the difference between a target population and a total population?** The target population is eligible for the intervention, such as surviving infants for DTP3. The total population includes everyone else. Using the total population as the denominator produces a coverage estimate that is too low by the share of the population that is not eligible.

**18. Why is a facility registry a data-quality instrument, not only an administrative list?** The expected-report denominator and the catchment population both depend on it. A registry error becomes a completeness error and a coverage error.

**19. What does a design effect of 1.0 mean?** The design is as efficient as simple random sampling. That is rare in household surveys with clustering and weighting, which is why a design effect of 1.0 in a report is a reason to check the variance calculation.

**20. Why is a verbal autopsy not a diagnosis?** It assigns a probability distribution over causes from a caregiver interview, without a clinical examination or a laboratory result. It is useful at the population level and unreliable for an individual.

**21. What is the difference between a PHEIC and a pandemic emergency under the amended IHR?** A PHEIC is a determination by the Director-General that an event constitutes a public health emergency of international concern. The 2024 amendments add "pandemic emergency" as a higher alert level, defined by wide geographic spread, health-system capacity being exceeded, and substantial social and economic disruption.

**22. Why should a data-sharing agreement be negotiated before the analysis?** Because the lawful basis, the storage location, and the ministry's rights determine what analysis is possible. A transfer that is legally doubtful cannot be fixed by anonymising the data after the fact.

**23. What is the difference between a count and a rate?** A count has no denominator. A rate has a denominator and a time unit. A count of doses and a rate of doses per child per year are different quantities, and a report that calls a count a rate has hidden the denominator.

**24. Why can completeness improve while coverage falls?** Completeness measures reporting. Coverage measures service reach. A district can improve reporting while a stockout reduces the number of children vaccinated.

**Rubric for Exercise 7 (triangulation).** A strong answer names the bias direction of each source, recommends a value with a range, explains why it is not an average, and states what evidence would change the conclusion. A weak answer averages the three numbers or picks the one that matches the target.

**Rubric for Exercise 8 (data sovereignty).** A strong plan names the lawful basis, minimises the data transferred, specifies storage and access, gives the ministry audit and publication rights, and addresses the end of the study. A weak plan says the data will be anonymised and stored securely.

## 14. Interview questions

**1. What is the difference between completeness and coverage?**
A strong answer defines both and gives an example where one is high and the other low. A weak answer treats them as synonyms.

**2. Why can coverage exceed 100%, and what do you do about it?**
A strong answer names the numerator, denominator, and definition causes and proposes a check. A weak answer says it is a data entry error.

**3. How would you estimate district coverage from a national survey?**
A strong answer describes direct estimation, the design effect, and small-area methods, and states the limits. A weak answer computes a district mean from the survey file.

**4. What is a design effect and why does it matter?**
A strong answer gives the variance ratio and the effect on the interval width. A weak answer says it is a survey parameter.

**5. How do you reconcile a facility list?**
A strong answer describes matching on coordinates and names, handling openings and closures, and recomputing the expected-report denominator. A weak answer says to clean the list.

**6. What is the difference between a research dataset and an operational system?**
A strong answer contrasts purpose, population, time, ownership, and change control. A weak answer says one is for research and one is for practice.

**7. When would you not use administrative coverage?**
A strong answer names the denominator and numerator problems and proposes the survey as a cross-check. A weak answer says when the data are bad.

**8. What is small-area estimation, and what is the risk?**
A strong answer describes borrowing strength and the bias-variance trade-off, and names the risk of over-shrinking a genuine outlier. A weak answer calls it a model for small places.

**9. How do you present uncertainty from three different sources?**
A strong answer separates sampling, non-sampling, and model error, and presents a range rather than a single interval. A weak answer adds the standard errors.

**10. What does data sovereignty require in practice?**
A strong answer names the law, the lawful basis, the minimisation, the storage location, and the ministry's audit rights. A weak answer says to keep the data in the country.

**11. Why is a district league table often misleading?**
A strong answer explains sampling error, shrinkage, and the difference between a rank and a difference. A weak answer says the data are poor.

**12. How would you explain a coverage gap to a ministry?**
A strong answer separates the measurement question from the programme question and gives a range with a stated source. A weak answer defends one number.

**13. What would you do if the routine and survey estimates disagreed by 30 points?**
A strong answer names the numerator, denominator, and period differences, proposes the checks in order, and states which source is more credible and why. A weak answer averages them.

**14. How would you design a phone survey to reduce coverage bias?**
A strong answer starts from a sampling frame with known coverage, weights for phone ownership and multiplicity, and reports the undercoverage. A weak answer increases the sample size.

**15. What is the role of a facility registry in a coverage estimate?**
A strong answer links the registry to the expected-report denominator and the catchment population. A weak answer calls it an administrative list.

**16. When would you report a range instead of a confidence interval?**
A strong answer distinguishes sampling error from non-sampling error and uses a range when the latter dominates. A weak answer says when the sample is small.

## 15. Key terms

- **Aggregate data** — counts over a period and an organisation unit.
- **Catchment population** — the population a facility actually serves.
- **Civil registration** — the legal record of births, deaths, and causes.
- **Cluster** — a sampled geographic unit in a survey.
- **Completeness** — reported facility-months divided by expected.
- **Concentration index** — inequality across a ranked distribution.
- **Coverage** — service contacts divided by the eligible population.
- **CRVS** — civil registration and vital statistics.
- **Data element** — the "what" of a DHIS2 value.
- **Data sovereignty** — data governed by the laws of the country of origin.
- **Denominator** — the population at risk or eligible.
- **Design effect** — variance ratio against simple random sampling.
- **DHIS2** — District Health Information Software 2.
- **DHS** — Demographic and Health Survey.
- **Empirical Bayes** — shrinkage toward a prior mean.
- **Equity stratifier** — a dimension such as wealth, sex, or residence.
- **Facility registry** — the authoritative list of health facilities.
- **FHIR** — Fast Healthcare Interoperability Resources.
- **Geospatial covariate** — a spatially linked model input.
- **Gridded population** — a population estimate on a regular grid.
- **HMIS** — health management information system.
- **IHR** — International Health Regulations.
- **LMIC** — low- and middle-income country.
- **Master facility list** — the national facility registry.
- **MICS** — Multiple Indicator Cluster Survey.
- **Nagoya Protocol** — access and benefit-sharing for genetic resources.
- **Organisation unit** — the "where" of a DHIS2 value.
- **PABS** — pathogen access and benefit-sharing.
- **Period** — the "when" of a DHIS2 value.
- **PHEIC** — public health emergency of international concern.
- **Phone survey** — a survey conducted by telephone.
- **Quintile** — a fifth of a ranked population.
- **Response rate** — completed interviews divided by eligible units.
- **Sampling weight** — inverse probability of selection, adjusted.
- **Shrinkage** — the weight given to the prior.
- **Small-area estimation** — model-based estimation for a small domain.
- **Timeliness** — reports received by the deadline divided by expected.
- **Tracker** — individual-level longitudinal data in DHIS2.
- **Undercoverage** — the share of the population with no chance of selection.
- **Verbal autopsy** — a structured interview to assign cause of death.

## 16. Sources

All URLs accessed 27 October 2026.

1. World Health Organization. *Civil registration and vital statistics.* Fact sheet, updated 27 February 2024. https://www.who.int/news-room/fact-sheets/detail/civil-registration-and-vital-statistics
2. World Health Organization. *Data Quality Review: Module 2, Desk Review of Data Quality.* 2017. https://iris.who.int/handle/10665/259225
3. World Health Organization. *Data Quality Assurance: Module 1, Framework and Metrics.* https://cdn.who.int/media/docs/default-source/world-health-data-platform/rhis-modules/dqa-module-1-framework-and-metrics.pdf
4. World Health Organization. *Toolkit for Routine Health Information Systems Data.* https://www.who.int/data/data-collection-tools/health-service-data/toolkit-for-routine-health-information-system-data/modules
5. World Health Organization. *Master Facility List Resource Package.* 2019. https://www.who.int/publications/i/item/9789241516495
6. DHIS2. *About data dimensions.* DHIS2 documentation. https://docs.dhis2.org/en/use/user-guides/dhis-core-version-243/understanding-the-data-model/about-data-dimensions.html
7. DHIS2. *Tracker.* DHIS2 documentation. https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-master/tracker.html
8. DHIS2. *In action.* 2025. https://dhis2.org/in-action/
9. The DHS Program. *Guide to DHS Statistics, DHS-8.* December 2023. https://dhsprogram.com/pubs/pdf/DHSG1/Guide_to_DHS_Statistics_DHS-8.pdf
10. The DHS Program. *DHS-8 Model Biomarker Questionnaire.* February 2023. https://www.dhsprogram.com/pubs/pdf/DHSQ8/DHS8_Biomarker_QRE_EN_03Feb2023_DHSQ8.pdf
11. The DHS Program. *Spatial Modeling for Subnational Administrative Level 2 Small-Area Estimation.* Spatial Analysis Report 21, 2021. https://dhsprogram.com/pubs/pdf/SAR21/SAR21.pdf
12. World Health Organization. *Verbal Autopsy Standards: The 2022 WHO Verbal Autopsy Instrument.* https://www.who.int/standards/classifications/other-classifications/verbal-autopsy-standards-ascertaining-and-attributing-causes-of-death-tool
13. World Health Organization. *Health Equity Assessment Toolkit.* https://www.who.int/data/inequality-monitor/assessment-toolkit
14. World Health Organization. *International Health Regulations (2005), as amended.* https://www.who.int/news-room/questions-and-answers/item/international-health-regulations-amendments
15. World Health Organization. *Amended International Health Regulations enter into force.* 19 September 2025. https://www.who.int/news/item/19-09-2025-amended-international-health-regulations-enter-into-force
16. World Health Organization. *WHO Pandemic Agreement.* https://www.who.int/health-topics/who-pandemic-agreement
17. African Union. *Convention on Cyber Security and Personal Data Protection.* https://au.int/en/treaties/african-union-convention-cyber-security-and-personal-data-protection
18. Government of Kenya. *Data Protection Act, 2019.* https://new.kenyalaw.org/akn/ke/act/2019/24/eng%402022-12-31
19. Office of the Data Protection Commissioner, Kenya. *Guidance Note on the Processing of Health Data.* December 2023. https://www.odpc.go.ke/wp-content/uploads/2024/02/ODPC-Guidance-Note-on-Processing-of-Health-Data.pdf
20. World Bank. *Population by income classification.* World Development Indicators. https://data.worldbank.org/
21. Yang Y, et al. Disparities in clinical studies of AI enabled applications from a global perspective. *npj Digital Medicine.* 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11316833/
22. Celi LA, et al. Sources of bias in artificial intelligence that perpetuate healthcare disparities: a global review. *PLOS Digital Health.* 2022. https://doi.org/10.1371/journal.pdig.0000022
23. Gourlay S, Kilic T, Martuscelli A, Wollburg P, Zezza A. Viewpoint: high-frequency phone surveys on COVID-19: good practices, open questions. *Food Policy.* 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8405596/
24. World Bank. *High-Frequency Mobile Phone Surveys of Households: Guidelines on Sampling Design.* April 2020. https://documents1.worldbank.org/curated/en/742581588695955271/pdf/Guidelines-on-Sampling-Design.pdf
25. GRID3. *High-resolution population estimates.* https://grid3.org/solution/high-resolution-population-estimates
26. WorldPop. *GRID3 2023-25.* https://www.worldpop.org/current-projects/grid3-2023-25/
27. OpenMRS. *OpenMRS Information Model.* https://guide.openmrs.org/getting-started/openmrs-information-model/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Related topic: [`03_registries-and-public-health.md`](03_registries-and-public-health.md)
- Related topic: [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md)
- Module survey file: [`../../course/02_biomedical_data_landscape.md`](../../course/02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Related reading on study design: [`../../course/05_study_design_evidence.md`](../../course/05_study_design_evidence.md)
