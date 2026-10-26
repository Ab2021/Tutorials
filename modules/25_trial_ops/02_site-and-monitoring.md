# Site and Monitoring: Risk-Based Quality Management, Source Data Review, and Query Management

> Module 25 — Trial Operations, HTA, and Global Regulation
> Topic 02 of 07
> Estimated study time: 4 hours
> Prerequisites: topic 01
> Version 2026.3

## 1. Why this topic exists

Monitoring is the part of trial operations that decides whether the data can be trusted. It is also
the largest discretionary cost in a trial after the investigational product itself. A sponsor can
spend three times more on monitoring than another sponsor running the same protocol, and neither
will be able to prove that the extra spend bought better data.

The historical default was to verify every data point against the source at every site. That default
was expensive, it scaled with visit volume rather than risk, and it did not detect the failures that
matter most: a site that fabricates data produces a clean-looking source file, and a site that
misunderstands eligibility produces deviations that are internally consistent. Regulators now expect
a risk-based approach, and the guidance says explicitly that 100% source data verification is not
required.

The result is a discipline that looks like data science but is governed like quality assurance. It
requires a risk assessment, a set of indicators with thresholds, a triage process, documented
actions, and evidence that the actions worked. A dashboard without an action is not monitoring. A
visit without a question is not oversight.

### 1.1 The pattern that repeats

1. A sponsor defines 25 key risk indicators and a dashboard.
2. The dashboard flags 12 sites in month 3.
3. Nobody has defined what a site owner must do when flagged, so the flags are noted and closed.
4. By month 12, the same sites have the same flags, and the operations team has learned to ignore
   the dashboard.
5. A monitoring visit finds a consent documentation problem that the indicators never covered.
6. The programme concludes that risk-based monitoring does not work, when the failure was the
   absence of an action model.

The pattern is common because indicators are easier to build than decisions. The hard part is not
detecting a signal; it is deciding which signals justify which action, and proving afterwards that
the action was taken.

### 1.2 Who is blocked

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Sponsor | Submission credibility | Depends on sites it does not employ |
| Site | Time responding to queries and visits | Receives monitoring intensity it did not choose |
| Data manager | Ability to lock on time | Waits on site responses |
| Statistician | Confidence in the analysis | Inherits data collected by others |
| Regulator | Assurance that the data are reliable | Sees the site only at inspection |
| Patient | Safety oversight | Has no visibility of monitoring |

### 1.3 The industry question

> Where should oversight be concentrated, how much verification is enough, and what evidence shows
> that the monitoring actually changed something?

## 2. Learning outcomes

By the end of this topic you can:

1. **Derive** critical-to-quality factors from a protocol and map them to critical data and processes.
2. **Build** a risk assessment that scores likelihood, impact, and detectability, and use it to
   allocate monitoring effort.
3. **Select** key risk indicators and quality tolerance limits with baselines, thresholds, and
   pre-defined actions.
4. **Design** a monitoring plan that combines central analytics, remote review, and on-site visits.
5. **Compute** monitoring resource requirements and compare a calendar-based plan with a risk-based
   one on cost and coverage.
6. **Model** a query lifecycle and show how the tail, not the median, determines the database lock
   date.
7. **Evaluate** a central statistical monitoring signal and decide whether it warrants a visit.
8. **Document** oversight of a contract research organisation (CRO) so that the sponsor can
   demonstrate control.

## 3. The core mental model

### 3.1 Oversight as a control system

```text
 Protocol
    |
    v
 Critical-to-quality factors
    |
    v
 Risk assessment (likelihood x impact x detectability)
    |
    +---------------------+----------------------+
    v                     v                      v
 Central analytics    Remote review         On-site visit
 (KRIs, statistical   (source data review,  (focused verification,
  monitoring)          query review)         interviews, drug accountability)
    |                     |                      |
    +---------------------+----------------------+
                          v
                    Signal triage
                          |
              +-----------+-----------+
              v                       v
        Action (documented)      No action (documented)
              |
              v
        Effectiveness check ----> update risk assessment
```

The loop is the unit of analysis. A monitoring plan that specifies the left-hand side (what will be
measured) but not the right-hand side (who decides, what they do, how it is verified) is a
measurement plan, not a control system.

### 3.2 Three layers of oversight

| Layer | What it sees | Strength | Blind spot |
|---|---|---|---|
| Central analytics | Patterns across sites and patients | Cheap, continuous, detects outliers | Cannot see source documents or site context |
| Remote review | Documents and data shared electronically | Intermediate cost, focused | Depends on what the site uploads |
| On-site visit | Source, facilities, people | Detects what is not in the data | Expensive, sampled, episodic |

The layers are complements, not substitutes. A site can have perfect data and a broken consent
process; only a document review finds that. A site can have perfect source documents and impossible
data distributions; only central analytics finds that efficiently.

### 3.3 Risk is a product, not a category

A practical scoring model multiplies three judgements:

```text
Risk score = likelihood of failure x impact on participant safety or data reliability
             x (1 / detectability)
```

The third term is the one teams forget. A risk that is easy to detect centrally deserves less
on-site effort than a risk that is invisible until inspection, even if both have the same impact.
Consent documentation is a good example: the impact is high, but the failure is detectable only by
looking at documents, so it attracts document review rather than statistical monitoring.

### 3.4 Critical to quality, then indicators

The chain is:

```text
Critical-to-quality factor -> critical data or process -> indicator -> threshold -> action
```

A critical-to-quality factor is a statement of what must go right for the result to be reliable.
"Participants meet the eligibility criteria" is a factor. The critical data are the eligibility
fields and their source documents. The indicator might be the screen-failure rate by site. The
threshold might be a rate more than two standard deviations from the study mean. The action might be
a targeted review of ten screening logs.

A programme that starts with indicators rather than factors ends up measuring what is easy to
measure. Query counts are easy. Consent quality is not.

### 3.5 The economics of sampling

Monitoring is sampling. The question is not whether to sample but where, and the answer depends on
the yield: how many meaningful findings per monitoring day, and how much harm each undetected
finding would cause.

```text
Value of a visit = (findings expected) x (cost of an undetected finding)
Cost of a visit  = monitor days x day rate + site burden + travel
```

A visit to a low-risk, high-performing site has low expected findings and low harm, so its expected
value is low even though its cost is the same as a visit to a problem site. A monitoring plan that
visits every site on the same schedule is therefore not neutral: it over-invests in good sites and
under-invests in bad ones.

### 3.6 The query is a workflow, not an event

A query passes through states: raised, received by site, answered, reviewed, closed, or re-raised.
Each transition has a delay. The distribution of total cycle time is right-skewed, so the mean is
longer than the median and the maximum is much longer than either.

The operational consequence is that the database lock date is set by the slowest open query, not by
the average. A team that plans from the median will be late by roughly the difference between the
median and the 90th percentile, multiplied by the number of escalation cycles.

### 3.7 What good monitoring looks like

A good monitoring system can answer four questions at any time:

1. Which sites carry the highest risk today, and why?
2. What was done about the last signal, and did it work?
3. Which critical data are least verified, and what is the residual risk?
4. What would an inspector find if they arrived tomorrow?

A system that cannot answer these is a reporting system.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Monitoring | Oversight of trial conduct and data quality | Used to mean site visits only | Activity |
| Risk-based monitoring (RBM) | Monitoring intensity proportional to risk | Used to mean reduced monitoring | Approach |
| Risk-based quality management (RBQM) | Quality system covering design, conduct, and oversight | Used as a synonym for RBM | System |
| Critical to quality (CtQ) | Factor that must be right for participant safety or reliable results | Confused with a key endpoint | Statement |
| Key risk indicator (KRI) | Metric tracked to detect emerging risk | Set without a threshold or action | Metric |
| Quality tolerance limit (QTL) | Pre-defined study-level limit for a critical parameter | Treated as a site-level trigger | Threshold |
| Source data verification (SDV) | Comparison of trial data with source documents | Assumed to detect all errors | Activity |
| Source data review (SDR) | Review of source documents for quality and process | Confused with SDV | Activity |
| Central monitoring | Analysis of aggregated data to detect anomalies | Assumed to replace visits | Activity |
| Statistical monitoring | Formal anomaly detection across sites and variables | Used without a triage process | Method |
| Signal | A detected pattern that may indicate a problem | Treated as a finding | Observation |
| Finding | A confirmed deviation or deficiency | Confused with a signal | Observation |
| Query | A request to resolve a data discrepancy | Treated as a data error | Record |
| Discrepancy | A difference between data fields or between data and source | Assumed to be an error | Record |
| Data clarification form (DCF) | Paper or electronic form used to resolve a query | Used for all query types | Form |
| Edit check | Automated rule that tests data on entry | Assumed to catch all errors | Rule |
| Auto-query | Query generated automatically by an edit check | Confused with a manual query | Record |
| Query ageing | Time a query has been open | Reported without a start event | Days |
| Site risk score | Composite score used to rank sites | Treated as a diagnosis | Score |
| Monitoring visit | On-site or remote review of site conduct | Assumed to require travel | Event |
| Site initiation visit (SIV) | Pre-activation training and readiness review | Treated as a formality | Event |
| Interim monitoring visit (IMV) | Periodic review during conduct | Scheduled by calendar rather than risk | Event |
| Close-out visit (COV) | Final reconciliation and closure | Confused with the last IMV | Event |
| For-cause visit | Visit triggered by a specific concern | Treated as punitive | Event |
| Triggered visit | Visit triggered by a KRI threshold | Confused with for-cause | Event |
| Corrective and preventive action (CAPA) | Action to fix a problem and prevent recurrence | Closed when training is delivered | Process |
| Effectiveness check | Evidence that a CAPA worked | Omitted | Verification |
| Audit | Sponsor's systematic examination of a vendor or site | Confused with inspection | Activity |
| Inspection | Regulator's examination of sponsor or site | Confused with audit | Activity |
| Fabrication | Invention of data that were never collected | Confused with falsification | Finding |
| Falsification | Alteration of true data | Confused with fabrication | Finding |
| ALCOA+ | Data integrity principles | Treated as a software feature | Convention |
| eSource | Original data captured electronically at the site | Assumed to remove the need for SDV | Source type |
| Residual risk | Risk remaining after controls | Assumed to be zero after a visit | Judgement |

### 4.1 Verification, review, and monitoring

Verification compares a recorded value with its source. Review examines the source and the process
for quality: was the value measured correctly, by the right person, at the right time? Monitoring is
the overall oversight activity that decides where verification and review are applied. Using the
three terms interchangeably produces a plan that counts visits without saying what happens during
them.

### 4.2 Signal, finding, and deviation

A signal is a pattern that might indicate a problem. A finding is a confirmed problem. A deviation
is a departure from the protocol or the applicable standard. A signal that is not triaged never
becomes a finding; a finding that is not classified never becomes a deviation; a deviation that is
not analysed never becomes a corrective action.

### 4.3 Key risk indicator and quality tolerance limit

A KRI is a site- or study-level metric used for early detection. A QTL is a pre-defined limit on a
critical study-level parameter, agreed before the trial starts, with a defined action if exceeded
and a reported outcome in the clinical study report. In ICH E6(R2), QTLs were introduced as part of
the sponsor's quality management system, and important deviations from them are reported under
section 9.6 of the ICH E3 clinical study report structure.

## 5. The mechanics, step by step

The procedure below builds a monitoring system for a multicentre trial. It assumes a protocol, an
electronic data capture (EDC) system, and access to site-level data flows. It is written for a
drug trial; device and decentralised trials change the sources, not the logic.

### Step 1. Derive critical-to-quality factors

**Input:** the protocol, the investigational brochure, and the statistical analysis plan.

**Output:** a list of 8 to 15 CtQ factors, each stated as an outcome that must hold, with the data
or process that proves it.

**Rationale:** The CtQ list is the anchor for every later decision. It prevents the monitoring plan
from drifting into what is easy to measure. A typical list covers eligibility, informed consent,
randomisation and allocation, administration of the investigational product, primary endpoint
assessment, safety reporting, and data integrity.

**Branch point:** a trial with a subjective primary endpoint needs a CtQ factor for assessor
training and endpoint adjudication. A trial with a device needs one for device accountability and
malfunction reporting.

### Step 2. Map processes and score risk

**Input:** the CtQ factors and a process map from consent to data entry.

**Output:** a risk register with, for each failure mode, a likelihood score, an impact score, a
detectability score, and a resulting priority.

**Rationale:** The scoring forces an explicit judgement about detectability, which is where
monitoring effort is usually misallocated. A failure that is visible in the data can be handled
centrally; a failure that is invisible until a document review must be handled at the site.

**Branch point:** use a published risk assessment template, such as the TransCelerate Risk
Assessment and Categorization Tool, or an equivalent. The template matters less than the discipline
of recording the rationale for each score.

### Step 3. Identify critical data and their sources

**Input:** the CtQ factors and the protocol's data flow.

**Output:** a data-criticality table naming, for each critical variable, where it is first recorded,
where it is transcribed, and what can go wrong between the two.

**Rationale:** SDV is only meaningful where there is a source to verify against. A variable
transcribed from a laboratory system directly into EDC has no site-level source document, so
verifying it by hand is theatre. A variable entered by a coordinator from a paper worksheet has a
real transcription risk.

### Step 4. Select KRIs and set thresholds

**Input:** the risk register and historical data from comparable trials.

**Output:** 10 to 20 site-level KRIs, each with a definition, a data source, a baseline, a threshold,
and a pre-defined action.

**Rationale:** An indicator without a threshold produces alerts that nobody acts on. An indicator
without an action produces a dashboard that is read and filed. The threshold should distinguish
normal variation from a signal worth investigating, and the action should name a role.

**Branch point:** thresholds can be absolute, relative to the study mean, or trend-based. Absolute
thresholds are simple but ignore site size. Relative thresholds adapt to the study but can flag
everyone when the whole study is drifting.

### Step 5. Define quality tolerance limits

**Input:** the CtQ factors, the protocol's critical parameters, and the study's acceptance criteria.

**Output:** 3 to 5 study-level QTLs with a parameter, a limit, a measurement method, an action, and a
reporting plan.

**Rationale:** QTLs operate at the study level, not the site level. A typical QTL is the proportion
of participants with a major eligibility deviation, or the proportion with a primary endpoint
assessment outside the protocol window. When a QTL is breached, the sponsor must assess the effect
on the analysis and report the breach and the action in the clinical study report.

### Step 6. Design the monitoring plan

**Input:** the risk register, the KRI set, the site list, and the resource budget.

**Output:** a monitoring plan specifying, per site tier, the visit types, their frequency, the SDV
and SDR sampling approach, the central review cadence, and the escalation path.

**Rationale:** The plan is where risk becomes resource allocation. A plan that says "five visits per
site per year" regardless of risk has not used the risk assessment. A plan that says "tier 1 sites
receive a monthly central review and a visit every eight weeks; tier 3 sites receive a quarterly
remote review and one visit per year" is a decision.

**Branch point:** remote visits are appropriate for sites with good electronic source and a stable
team. They are weak where the risk is physical: drug storage, device handling, or a site with new
staff.

### Step 7. Build central analytics

**Input:** EDC extracts, laboratory data, randomisation data, and site metadata.

**Output:** a reproducible site risk score, a set of anomaly flags, and a review queue ordered by
priority.

**Rationale:** Central analytics is cheap per site and continuous in time. Its weakness is
specificity: simple statistical tests flag many sites when none is problematic. Published
simulation work shows that some methods achieve high sensitivity with specificity below 50%, while
others exceed 90% specificity but miss moderate deviations. A programme needs to know which kind of
error it is buying.

### Step 8. Configure the query workflow

**Input:** the data management plan, the edit-check specification, and the site communication
channels.

**Output:** a query lifecycle with defined states, owners, service levels, and escalation rules.

**Rationale:** Most query delay is waiting, not working. A query that sits in a site inbox for two
weeks has the same effect as a query that takes two weeks to answer. The workflow should measure
time in each state, not only total age, or the team will optimise the wrong part.

### Step 9. Run the monthly review

**Input:** the KRI dashboard, the central monitoring output, the query metrics, and the visit
findings.

**Output:** a triage log recording, for each signal, the decision, the rationale, the owner, and the
due date.

**Rationale:** The review meeting is the control point. If it produces a list of flags without
decisions, the system has failed at the only step that matters. The triage log is also the evidence
that oversight happened, which is what an inspector will ask for.

### Step 10. Act, document, and verify

**Input:** the triage log.

**Output:** an action (remote follow-up, targeted visit, retraining, CAPA), a record of completion,
and an effectiveness check.

**Rationale:** An action without an effectiveness check is a hope. The check can be simple: did the
query rate at the site fall, did the deviation rate decline, did the next visit find the same
problem? The answer should feed back into the risk register.

### Step 11. Conduct visits with a focused agenda

**Input:** the site's risk profile, open queries, and prior findings.

**Output:** a visit report with findings classified by severity, actions, and a follow-up date.

**Rationale:** A visit that repeats the same checklist at every site wastes the opportunity that
risk-based monitoring creates. The agenda should be built from the site's specific signals: the ten
consent forms selected because the site's consent query rate is high, the drug accountability
records selected because the temperature log has gaps.

### Step 12. Manage query ageing and escalation

**Input:** the query log with state timestamps.

**Output:** an ageing report by site and by state, and an escalation list.

**Rationale:** Query ageing should be measured from the date the query was raised and reported at
the 50th, 90th, and 95th percentiles. A median-only report hides the tail that controls the lock.
Escalation should be automatic at defined ages, not dependent on a data manager remembering to
raise it.

### Step 13. Oversee vendors and the CRO

**Input:** the CRO's monitoring metrics, the sponsor's oversight plan, and the risk register.

**Output:** a vendor oversight record with periodic review, issue log, and escalation.

**Rationale:** The sponsor remains accountable for monitoring even when the work is delegated. An
oversight plan that reviews only aggregate metrics misses site-level signals. The sponsor should
periodically review a sample of visit reports and query records directly, not only the summary the
vendor prepares.

### Step 14. Close the loop and prepare for inspection

**Input:** the risk register, triage log, visit reports, CAPA records, and query history.

**Output:** an updated risk assessment, a close-out monitoring plan, and an inspection-ready
oversight file.

**Rationale:** The risk assessment is a living document. A trial that ends with the same risk scores
it started with has either been lucky or has not been looking. The oversight file is the story of
what the sponsor knew, when it knew it, and what it did.

### 5.15 Decision table: monitoring intensity by risk tier

| Tier | Criteria | Central review | Remote review | On-site visits | SDV sampling |
|---|---|---|---|---|---|
| Tier 1 (high) | New site, complex endpoint, prior findings, high query rate | Monthly | Monthly | Every 6 to 8 weeks | 100% of critical data for first 10 patients, then 50% |
| Tier 2 (medium) | Experienced site, stable staff, average metrics | Quarterly | Quarterly | Twice yearly | 20 to 30% of critical data |
| Tier 3 (low) | Proven site, simple protocol, low query rate | Six-monthly | As needed | Once yearly or remote | 10% of critical data, plus triggered review |
| For-cause | Specific signal or allegation | Ad hoc | Ad hoc | Within 2 to 4 weeks | Targeted to the signal |

The tiers are not permanent. A site moves between them as its metrics change, and the move should be
recorded with a reason.

### 5.16 Decision table: triage of a central monitoring signal

| Signal pattern | Possible cause | First action | Escalation if unresolved |
|---|---|---|---|
| Single KRI above threshold, one site | Local process issue | Remote review of the underlying records | Targeted visit if the review does not explain it |
| Multiple KRIs above threshold, one site | Site-level quality problem | Targeted visit within four weeks | CAPA and sponsor notification |
| Same KRI above threshold at many sites | Study-level design or training problem | Review the protocol interpretation and training materials | Protocol clarification or amendment |
| Statistical outlier with plausible clinical explanation | Case mix or population difference | Document the explanation | Re-test after the next data cut |
| Statistical outlier with no plausible explanation | Possible fabrication or systematic error | For-cause visit with source review | Audit, re-training, or site closure |
| Signal that disappears after data cleaning | Data processing artefact | Document and close | None |

### 5.17 Decision table: query escalation

| Query age | Status | Action | Owner |
|---|---|---|---|
| 0 to 7 days | Open, site notified | Automatic reminder at day 5 | System |
| 8 to 14 days | Open, no response | Data manager contacts site coordinator | Data manager |
| 15 to 30 days | Open, response incomplete | Clinical research associate (CRA) raises at next contact | CRA |
| 31 to 45 days | Open, affecting critical data | Escalate to principal investigator | CRA and sponsor |
| More than 45 days | Open, unresolved | Sponsor escalation; consider visit | Sponsor clinical lead |
| Any age | Query reopened twice | Root-cause review, not another query | Data manager |

### 5.18 Edge cases

**Site under-enrolment.** A site that has enrolled nobody has no data queries and may look clean on
a data-driven dashboard. Monitoring should include an enrolment indicator, and a low-enrolling site
should be reviewed for whether it can still contribute, not only for whether its data are correct.

**Protocol amendment mid-study.** An amendment changes the reference standard for every prior
finding. The monitoring plan must be re-based: retraining, updated edit checks, a defined date from
which the new version applies, and a review of patients enrolled under the old version.

**Database lock with open queries.** Lock can proceed with queries that affect neither the primary
endpoint nor safety, provided the omission is documented and handled in the analysis. The decision
rule belongs in the lock checklist, not in a meeting.

**Cross-border data transfer.** Remote monitoring requires the monitor to see source documents that
may sit in another jurisdiction. The lawful mechanism, the vendor's access model, and the site's
permission must be settled before the first remote review, not at the first inspection.

**Payer rejection and price negotiation failure.** These sit downstream of this topic, but they
raise the value of inspection-grade data: a health technology assessment submission that relies on
a trial with unresolved data integrity findings is vulnerable to challenge. The monitoring file is
part of the evidence base, not only a compliance artefact.

### 5.19 How a site risk score is built

A site risk score combines several indicators into one number so that a small team can rank 40 or
400 sites. The construction matters more than the number.

A defensible score has four properties:

1. **Components are normalised.** Query counts, deviation counts, and enrolment rates have different
   units. Each is converted to a z-score or a percentile within the study before combination.
2. **Weights reflect impact.** A consent deviation should not carry the same weight as a late visit
   window. The weights come from the risk register, not from convenience.
3. **Direction is explicit.** For some indicators a high value is bad (query rate); for others a low
   value is bad (enrolment). The score must invert the second group.
4. **The components remain visible.** A composite score tells you which sites to look at; it does not
   tell you why. The review screen must show the contributions.

A common error is to score sites on data volume. Large sites generate more queries and more
deviations in absolute terms, so an unnormalised score simply ranks sites by size. Rates per patient
or per visit are usually more informative than counts, except where the absolute number determines
the workload.

A second error is to reweight the score after seeing which sites it flags, so that the ranking
matches the team's prior view. That converts a monitoring tool into a justification tool. Weights
should be fixed before the first review and changed only with a documented reason.

### 5.20 The query state model

A query is not open or closed; it moves through states, and the time in each state has a different
owner and a different fix.

| State | Owner | Typical service level | What delay means |
|---|---|---|---|
| Raised | System or data manager | Immediate | Edit check or review lag |
| Notified | System | 1 day | Notification configuration |
| Acknowledged | Site coordinator | 3 days | Site capacity or inbox management |
| Answered | Site coordinator or investigator | 7 days | Data availability or clinical judgement |
| Reviewed | Data manager | 2 days | Data management capacity |
| Re-raised | Data manager | 2 days | Answer incomplete or implausible |
| Closed | Data manager | 1 day | Confirmation lag |

Measuring only total age hides which state is slow. A site that acknowledges in one day and answers
in twenty has a data availability problem; a site that acknowledges in ten days and answers in one
has an inbox problem. The interventions are different: the first needs a conversation with the
investigator, the second needs a coordinator workflow change.

The state model also makes the reopen rate interpretable. A high reopen rate means answers are
incomplete, which is usually a query-quality problem on the sponsor side, not a site problem.

### 5.21 Building the visit agenda from signals

A monitoring visit has a fixed cost and a variable yield. The agenda determines the yield.

A signal-driven agenda has four parts:

1. **Verification sample.** The specific records selected because an indicator flagged them: the ten
   consent forms from the site with consent queries, the five patients whose visit dates fall outside
   the window, the drug accountability entries with temperature gaps.
2. **Process observation.** A short observation of a live process, such as consent discussion,
   spirometry, or sample handling, chosen because the risk register says the process is critical and
   hard to verify from records.
3. **Interview.** A conversation with the coordinator and investigator about what is not working.
   This is the cheapest source of information and the most frequently omitted.
4. **Reconciliation.** Queries, deviations, and drug accountability reconciled against the site file
   so that the visit leaves a consistent record.

An agenda built only from part 1 is a verification visit. An agenda with all four is an oversight
visit. The difference in cost is small; the difference in findings is not.

### 5.22 Metrics that indicate the system is working

| Metric | Definition | Why it matters |
|---|---|---|
| Signal-to-action rate | Proportion of triaged signals with a documented action | Distinguishes monitoring from reporting |
| Action effectiveness | Proportion of actions that reduced the target metric | Tests whether the action worked |
| Query cycle time, P50 and P90 | Days from raise to close | The tail drives lock |
| Query reopen rate | Queries closed then reopened | Measures answer quality |
| SDV finding rate | Findings per 100 SDV hours | Measures yield, not effort |
| Visit coverage by tier | Visits delivered against plan, by risk tier | Tests whether risk allocation happened |
| Critical data verification | Proportion of critical variables verified | Residual risk indicator |
| Time from signal to visit | Days from flag to on-site review | Responsiveness |

## 6. Worked example

### 6.1 The trial

A Phase III trial enrols 600 patients at 40 sites over 27 months, 15 patients per site. The sponsor
is choosing between a calendar-based monitoring plan with 100% source data verification and a
risk-based plan with tiered coverage. A loaded monitor costs $165,000 per year (2026 US dollars),
works 210 days per year, and performs 1,400 productive hours of review per year. The trial's fixed
operating cost is $600,000 per month, or $20,000 per calendar day on a 30-day month.

### 6.2 Monitoring resource arithmetic

| Component | Calendar-based | Risk-based |
|---|---:|---:|
| Sites by tier | 40 untiered | 8 high, 20 medium, 12 low |
| Visits per site per year | 5 | 6, 3, 1 |
| Total visits per year | 200 | 120 |
| Monitor-days at 3 days per visit | 600 | 360 |
| Visit full-time equivalents (FTEs) | 2.86 | 1.71 |
| Patients with 100% SDV | 600 | 120 |
| Patients with 30% SDV | 0 | 480 |
| SDV hours at 2.5 hours per patient | 1,500 | 660 |
| SDV FTEs at 1,400 hours per year | 1.07 | 0.47 |
| Total FTEs | 3.93 | 2.19 |
| Annual cost | $648,214 | $360,643 |

The risk-based plan saves $287,571 per year in monitoring cost, or about 44%. The saving does not
by itself prove that the plan is better. It proves that the calendar plan was buying verification
where the risk assessment says verification was not needed.

### 6.3 Query lifecycle arithmetic

The trial generates 10 queries per patient, or 6,000 queries over 27 months. That is 222 queries per
month. The observed cycle-time distribution is:

| Share of queries | Days to close | Contribution to mean |
|---:|---:|---:|
| 40% | 7 | 2.80 |
| 25% | 14 | 3.50 |
| 20% | 30 | 6.00 |
| 15% | 60 | 9.00 |
| 100% | | 21.30 |

The mean is 21.3 days, the median is 14 days, and the 90th percentile is 60 days. At any moment,
the number of open queries is approximately the arrival rate multiplied by the mean cycle time:

```text
Open queries = 222 per month x (21.3 / 30.44) months = 155
```

A simulation with random arrival times gives 170 open at last patient last visit (LPLV), because
arrival is not perfectly uniform. The difference is sampling noise, not a modelling error.

### 6.4 The database lock date

The plan assumes that queries close within 21 days of LPLV, so the database lock is scheduled for
LPLV plus 21 days. That assumption uses the mean.

The last query to close is not average. Fifteen per cent of queries take 60 days, and one of them
will have been raised in the final days before LPLV. The last closure therefore lands about 58 days
after LPLV, and the lock moves with it:

```text
Planned lock   = LPLV + 21 days
Actual lock    = LPLV + 58 days
Slip           = 37 days
Cost of slip   = 37 x $20,000 = $740,000
```

The arithmetic is not subtle, but it is routinely missed because the lock plan is built from an
average cycle time. A plan built from the 90th percentile would have scheduled the lock at LPLV plus
60 days and would have been right.

### 6.5 Diagnostic 1: the median hides the tail

The query dashboard reports a median of 14 days, which looks healthy. The same data show that 15% of
queries exceed 45 days and that the 90th percentile is 60 days. The median is the wrong statistic
for a deadline that depends on the last completion.

A second statistic makes the problem visible: the share of queries open for more than 45 days,
tracked weekly. If that share rises, the lock date is at risk even while the median improves. The
two can move in opposite directions when a team closes easy queries quickly and leaves hard ones
open, which is exactly what a productivity target on query count produces.

### 6.6 Diagnostic 2: static tiering allocates visits to the wrong sites

The risk-based plan assigns tiers at study start using site experience and prior performance. At
month 12, the query data show a different picture:

| Observation | Consequence |
|---|---|
| 6 of the 8 sites with the highest query burden are in the medium tier | Each receives 3 visits per year instead of 6 |
| 4 of the 8 designated high-risk sites have query rates below the study median | Each receives 3 visits per year more than its risk justifies |
| The 6 high-burden sites account for 40% of all queries | 2,400 queries sit at sites with half the planned oversight |

The misallocation is 30 visits per year, or 90 monitor-days, about 0.43 FTE. The plan is risk-based
on paper and calendar-based in practice, because the risk assessment was performed once and never
updated.

### 6.7 Vary one assumption: dynamic re-tiering and earlier escalation

Suppose the sponsor re-tiers sites every six months and escalates any query open beyond 30 days.
The six high-burden sites move to the high tier and receive two additional visits per year. Their
query rate falls from about 25 per patient to about 14 as the underlying documentation problems are
corrected.

```text
Queries avoided = 6 sites x 15 patients x 11 queries = 990
New total       = 6,000 - 990 = 5,010
Open at LPLV    = 114 (simulation)
Tail duration   = 45 days, because the 60-day queries are escalated earlier
Lock            = LPLV + 43 days
```

The lock is 15 days earlier than the 58-day base case, saving 15 × $20,000 = $300,000. The
incremental monitoring cost is 12 visits × 3 days = 36 monitor-days, or 0.17 FTE, or $28,286. The
net benefit is about $272,000.

The conclusion reverses if the extra visits do not change site behaviour. If the query rate stays
at 25 per patient, the added visits cost $28,286 and save nothing. The value of monitoring is
therefore not in the visits; it is in whether the visits change what the site does. That is a
testable claim, and it should be tested with a before-and-after metric rather than assumed.

### 6.8 Break-even on an additional visit

An extra visit costs three monitor-days. At $165,000 per year over 210 working days, a monitor-day
costs $786, so the visit costs $2,357 plus travel and site burden, call it $3,500.

The visit pays for itself if it prevents delay. At $20,000 per day of delay, it needs to prevent
0.175 days, or about four hours, to break even. The probability that a well-targeted visit to a
high-risk site prevents at least a day of delay is far above the 1.75% implied by that arithmetic.

This is why the argument for risk-based monitoring should not be framed as "visits are expensive".
Visits are cheap relative to delay. The problem with calendar-based monitoring is not that it visits
too much in total; it is that it visits the wrong places, and the fixed budget crowds out the visits
that would have found something.

### 6.9 Sensitivity: the tail share drives the answer

The base case assumes 15% of queries take 60 days. Suppose the true share is 5%, with the other 10%
moving to 30 days. The 90th percentile falls to 30 days, and the last query closes about 30 days
after LPLV rather than 58.

```text
Lock slip = 30 - 21 = 9 days
Cost of slip = 9 x $20,000 = $180,000
```

The case for the intervention weakens: a $300,000 saving becomes a $180,000 saving, and the
$28,286 cost is a larger fraction of it. If the tail share were 2%, the lock plan of 21 days would be
approximately right and the intervention would not pay.

The sensitivity matters because the tail share is estimable from historical data. A sponsor with
query histories from prior trials can fit the distribution before the trial starts, rather than
discovering it at lock.

### 6.10 What the example teaches

1. Verification cost is computable from visits, days per visit, and hours per patient, and the
   difference between a calendar plan and a risk-based plan is large enough to fund other work.
2. The lock date is a maximum, not a mean. Planning from the mean guarantees lateness.
3. Static risk tiering decays. A risk assessment that is not updated becomes a calendar with extra
   paperwork.
4. The value of an intervention depends on a behavioural assumption that must be measured, not
   asserted.

## 7. Code, query, or calculation

The model below reproduces the worked example. It uses only the standard library.

**Assumptions:**

- 6,000 queries issued uniformly over 27 months in the base case;
- cycle times drawn from 7, 14, 30, and 60 days with probabilities 0.40, 0.25, 0.20, 0.15;
- the intervention case issues 5,010 queries with cycle times 7, 14, 30, and 45 days and
  probabilities 0.45, 0.25, 0.20, 0.10;
- 3 monitor-days per visit, 210 working days per monitor-year;
- 2.5 SDV hours per patient, 1,400 productive hours per monitor-year;
- loaded cost $165,000 per FTE-year in 2026 US dollars.

```python
import random
import statistics

BASE_DAYS = (7, 14, 30, 60)
BASE_PROB = (0.40, 0.25, 0.20, 0.15)
FAST_DAYS = (7, 14, 30, 45)
FAST_PROB = (0.45, 0.25, 0.20, 0.10)

def lifecycle(n_queries, months=27.0, days=BASE_DAYS, prob=BASE_PROB, seed=11):
    rng = random.Random(seed)
    issued, closed = [], []
    for _ in range(n_queries):
        start = rng.uniform(0, months)
        duration = rng.choices(days, prob)[0]
        issued.append(start)
        closed.append(start + duration / 30.44)
    cycles = sorted((c - s) * 30.44 for s, c in zip(issued, closed))
    open_at_lplv = sum(1 for c in closed if c > months)
    tail = (max(closed) - months) * 30.44
    return {
        "mean": statistics.mean(cycles),
        "median": statistics.median(cycles),
        "p90": cycles[int(0.90 * len(cycles))],
        "open": open_at_lplv,
        "tail": tail,
    }

def resourcing(visits, sdv_hours, days_per_year=210,
               hours_per_year=1400, cost_per_fte=165_000):
    fte = visits * 3.0 / days_per_year + sdv_hours / hours_per_year
    return fte, fte * cost_per_fte

base = lifecycle(6000)
fast = lifecycle(5010, days=FAST_DAYS, prob=FAST_PROB)

for name, r in (("base", base), ("re-tiered", fast)):
    print(f"{name}: mean {r['mean']:.1f}  median {r['median']:.1f}  P90 {r['p90']:.1f}  "
          f"open at LPLV {r['open']}  tail {r['tail']:.1f} days")

plans = {
    "calendar, 100% SDV": (200, 600 * 2.5),
    "risk-based, 30% SDV": (120, 0.30 * 480 * 2.5 + 120 * 2.5),
    "re-tiered, 30% SDV": (132, 0.30 * 480 * 2.5 + 120 * 2.5),
}
for name, (visits, hours) in plans.items():
    fte, cost = resourcing(visits, hours)
    print(f"{name}: visits {visits}  FTE {fte:.2f}  cost USD 2026 {cost:,.0f}")

print(f"lock saving from 15 days: {15 * 20_000:,.0f}")
print(f"incremental monitoring:  {(resourcing(132, 660)[1] - resourcing(120, 660)[1]):,.0f}")
```

**Expected output:**

```text
base: mean 21.0  median 14.0  P90 60.0  open at LPLV 170  tail 58.1 days
re-tiered: mean 17.1  median 14.0  P90 45.0  open at LPLV 114  tail 43.1 days
calendar, 100% SDV: visits 200  FTE 3.93  cost USD 2026 648,214
risk-based, 30% SDV: visits 120  FTE 2.19  cost USD 2026 360,643
re-tiered, 30% SDV: visits 132  FTE 2.36  cost USD 2026 388,929
lock saving from 15 days: 300,000
incremental monitoring:  28,286
```

**Caveat on the code:** each query is paired with a single duration draw and a single issue date,
so the tail is a genuine maximum, not an artefact. The model still assumes that arrival and
resolution are independent. In a real trial they are not: sites that generate many queries also
respond slowly, which makes the true tail longer than this model reports.

**What to vary:** the probability of the 60-day category. It has more effect on the lock date than
the mean, because the maximum is drawn from that category. A programme that halves the 60-day share
moves the lock date more than one that reduces every cycle time by 10%.

## 8. Data, standards, and regulatory context

Monitoring is governed by a mixture of binding regulation, international guidance, and industry
convention. The three carry different weight, and a monitoring plan should be able to say which is
which.

### 8.1 Binding requirements

| Instrument | What it requires of monitoring | Consequence |
|---|---|---|
| 21 CFR 312.56 (US) | Sponsor must monitor the investigation and secure compliance | Failure can lead to disqualification of the investigation |
| 21 CFR 312.62 (US) | Investigator must maintain accurate records and retain them | Source documentation must exist and be retrievable |
| 21 CFR 312.68 (US) | Investigator must permit inspection of records | Records must be accessible, not only stored |
| 21 CFR Part 11 (US) | Electronic records and signatures | Audit trails, access control, validation |
| Regulation (EU) 536/2014 | Sponsor oversight and inspection by member states | Inspections can be triggered by the trial or by signal |
| ISO 14155:2020 | Monitoring of device investigations | Device accountability and site training are explicit |
| General Data Protection Regulation | Lawful processing of personal data | Remote access to source across borders needs a mechanism |

The US rules are written in terms of the sponsor's obligation to monitor, not in terms of a
particular method. 21 CFR 312.56 requires the sponsor to review the conduct of the investigation and
to secure compliance; it does not prescribe visit frequency or SDV percentage. The method is left to
the sponsor, which is why guidance rather than regulation carries most of the operational detail.

### 8.2 ICH E6 and the shift to risk

ICH E6(R2), adopted in 2016, added explicit expectations for a quality management system, a
risk-based approach, and pre-defined quality tolerance limits. It also required the sponsor to
report important deviations from those limits in the clinical study report. ICH E6(R3), adopted at
Step 4 on 6 January 2025 and effective in the EU on 23 July 2025 for the Principles and Annex 1,
restates the principles in a more technology-neutral form: oversight should be proportionate to the
risks to participants and to the reliability of results, and data governance should cover the full
life cycle from capture to archive.

E6(R3) does not say how many KRIs to use, how often to visit, or what percentage to verify. Those
choices remain sponsor decisions, and they are the substance of the monitoring plan.

### 8.3 FDA guidance

The FDA's 2013 guidance, *Oversight of Clinical Investigations: A Risk-Based Approach to
Monitoring*, established that sponsors may use a combination of on-site and central monitoring and
that 100% SDV is not required. The 2023 question-and-answer guidance, *A Risk-Based Approach to
Monitoring of Clinical Investigations*, elaborates on planning, monitoring-plan content, and the
communication of results. It applies to drugs, biologics, devices, and combination products.

The guidance is a recommendation, not a regulation. A sponsor that follows it is not guaranteed to
satisfy an inspector, and a sponsor that departs from it is not automatically non-compliant. The
test is whether the oversight was adequate for the risks of the specific trial.

### 8.4 European expectations

The Clinical Trials Regulation requires member states to inspect trials and gives inspectors access
to the trial master file, source data, and facilities. The EMA's GCP inspectors coordinate through
the Good Clinical Practice Inspectors Working Group. Inspections can be routine, triggered by a
signal, or part of a marketing-application assessment.

The practical difference from the US is that European inspections are often conducted at the site
by the national authority, and the sponsor is asked to demonstrate oversight of that site. A sponsor
that delegated everything to a CRO and reviewed only summary metrics may struggle to answer
questions about what it knew and when.

### 8.5 Devices and decentralised elements

For device trials, ISO 14155:2020 adds explicit expectations for monitoring device accountability,
training, and the handling of malfunctions. For trials with decentralised elements, the FDA's 2024
final guidance, *Conducting Clinical Trials With Decentralized Elements*, makes clear that the
regulatory requirements are unchanged: the sponsor must still monitor, and the investigator remains
responsible for trial conduct, including oversight of local health care providers who perform
trial-related activities.

Remote monitoring does not remove the need for a documented oversight model. It changes the medium
in which evidence is reviewed and adds a new dependency: the site's ability to share source
documents electronically, and the legal basis for doing so across borders.

### 8.6 What the standards do not constrain

The standards do not set:

- the number of KRIs or their thresholds;
- the frequency of visits or the proportion of data to verify;
- the acceptable query turnaround time;
- the composition of the site risk score;
- the evidence that a monitoring action worked.

Those are design decisions. A monitoring plan that hides them behind a reference to "industry
practice" is not auditable.

> **Note:** Regulatory statements here reflect published texts at the access date. This is
> educational material, not legal advice.

## 9. Industry practice and case evidence

### 9.1 What the frameworks say

TransCelerate's risk-based monitoring initiative, which ran until January 2021, produced a risk
indicator library of more than 140 indicators built from a blinded 2014 survey, organised into
categories including issue management, essential documents, investigational product, recruitment,
data quality, staffing, and safety. Its guidance recommends that thresholds trigger an action,
preferably a remote follow-up before an on-site visit.

The same body of work recommends 3 to 5 quality tolerance limits per study and reports that
site-level KRI counts typically run from 10 to 20. These are conventions, not requirements. A study
with 40 KRIs is not more compliant than one with 12; it is more likely to produce alerts that
nobody investigates.

### 9.2 What the evidence says about central statistical monitoring

Central statistical monitoring detects atypical sites by testing whether a site's data distribution
is consistent with the rest of the study. The methods range from simple t-tests to multivariate
models and unsupervised anomaly detection.

A 2023 simulation study compared four methods for detecting an atypical site on a continuous
variable. The t-test and a Bayesian finite mixture method had good sensitivity but specificity that
never exceeded about 45%, meaning more than half of flagged sites were false alarms. Linear
mixed-effects and distance-based methods exceeded 90% specificity but had low sensitivity when the
mean deviation was below about 50%. The practical conclusion is that highly specific methods are
suitable for routine early use, but they complement rather than replace on-site review.

A 2019 study using a data inconsistency score across many tests reported specificity above 93% in
simulated contamination scenarios, with the best sensitivity when the score and adjusted p-values
were combined. A 2025 scoping review of 24 papers found the field split between established methods
(descriptive statistics, hypothesis tests, digit-preference checks, Mahalanobis distance,
visualisation) and newer ones (clustering, Bayesian mixtures, mixed models, funnel plots), with
little regulatory guidance on which to use.

The operational reading is that no single method is sufficient, and that a flag is a reason to look,
not a finding.

### 9.3 What the evidence does not show

There is limited public evidence that risk-based monitoring reduces inspection findings or improves
data quality compared with a well-run calendar-based programme. Most published accounts are
descriptions of implementations, not controlled comparisons. The cost difference is easier to
demonstrate than the quality difference, and cost is what most business cases emphasise.

> **Inference:** the claim that risk-based monitoring improves quality is plausible and widely
> believed, but the public evidence base is thinner than the claim. A programme should measure its
> own signal-to-action and effectiveness rates rather than assume the benefit.

### 9.4 Documented failure patterns

Published accounts of data fabrication and systematic error illustrate the limits of both on-site
and central methods. Venet et al. (2012) described abnormalities in multicentre trial data that were
not detected by conventional on-site monitoring but were visible in the distribution of values
across sites. Trotta et al. (2019) showed that unsupervised monitoring could detect a contaminated
site with high specificity.

The inference for practice is that the two layers see different things. A site that fabricates data
can produce source documents that match the fabricated data, so SDV finds nothing. A site that
misunderstands a procedure can produce data that look statistically normal, so central monitoring
finds nothing. Neither layer alone is sufficient.

### 9.5 The vendor oversight problem

Sponsors increasingly delegate monitoring to CROs and then oversee the CRO through monthly metrics.
The metrics are usually aggregate: visits completed, queries closed, sites activated. Aggregates
cannot show whether a visit asked the right questions, and they can improve while site-level quality
declines.

A defensible oversight model reviews a sample of primary records: visit reports, query histories,
and escalation logs. The sample is small but direct. It answers the question the aggregate cannot:
did the monitor do the work, and did the work find anything?

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Dashboard without actions | Signals closed with "reviewed, no action" | No decision model | Monitoring becomes reporting | Define action per threshold |
| Static risk tiering | Tiers unchanged after month 6 | Risk assessment not refreshed | Visits go to the wrong sites | Re-tier every 6 months |
| 100% SDV by default | Monitor days scale with visit volume | Verification treated as quality | Cost without risk targeting | Tiered SDV with a stated rationale |
| Median-only query reporting | P90 and share over 45 days not reported | Tail hidden | Lock date missed | Report P50, P90, P95 and ageing |
| Query target on closure count | Reopen rate rises | Easy queries closed first | Hard problems deferred to lock | Measure reopen rate and age of open critical queries |
| Central flags never validated | Flag rate stable, findings rate zero | Threshold set too low or too high | Alert fatigue or missed signal | Back-test thresholds against known findings |
| Monitoring plan not updated after amendment | Old edit checks still active | Change control gap | Data collected under two standards | Re-baseline plan at each amendment |
| Remote monitoring without source access | Site uploads are incomplete | Legal or technical barrier | Review is superficial | Settle access and transfer before first review |
| CRO oversight by aggregate metrics | Sponsor cannot name a site-level finding | Delegation without control | Inspection finding on oversight | Sample primary records quarterly |
| Visit agenda not site-specific | Same checklist at every site | Monitoring by routine | Low yield per visit | Build agenda from site signals |
| Consent monitored only by query rate | Consent queries rare but findings severe | Query data do not capture process | Serious GCP finding | Add document review for consent |
| Drug accountability verified at close-out only | Temperature excursions discovered late | Deferred verification | Product integrity unknown | Review accountability at each visit |
| Signal triage undocumented | No record of decisions | Poor documentation practice | Cannot demonstrate oversight | Triage log with owner and date |
| Risk score used as a diagnosis | Site ranked high, no review of why | Composite score obscures cause | Wrong action | Require component-level review |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Query arrival is approximately uniform | Converts rate to open count | Batch data entry before visits | Compare monthly arrival counts |
| Cycle time is independent of site performance | Simplifies the model | Slow sites also generate more queries | Correlate site query rate with cycle time |
| 2.5 hours per patient for SDV | Sizes the resource model | Complex endpoints take longer | Time a sample of SDV activities |
| 3 days per monitoring visit | Sizes visit capacity | Travel time and site size vary | Use actual visit logs |
| 1,400 productive hours per FTE-year | Converts hours to headcount | Training, leave, and admin reduce it | Compare with timesheet data |
| Risk tiers predict future query burden | Allocates visits | Tiers are based on history, not current behaviour | Measure predictive accuracy at month 6 |
| A visit changes site behaviour | Justifies the cost | Sites may not act on findings | Before-and-after metric at the site |
| Central methods have stable specificity | Justifies automated flags | Specificity depends on site size and variable | Back-test on historical data |
| Source documents are accessible | Enables verification | eSource access, cross-border limits | Test access at one site before scaling |
| The CRO reports accurately | Enables oversight | Metrics may be curated | Sample primary records |

### 11.1 Limits of the model

The cost model treats monitoring as a linear function of visits and SDV hours. In practice there are
fixed costs (systems, management, travel) that do not vary with visits, so the percentage saving
overstates the absolute saving on a small trial.

The query model treats each query independently. In reality, queries cluster by site and by visit,
and a site that is slow on one query is often slow on all of them. That correlation makes the true
tail longer than the independent model predicts.

The model also says nothing about the quality of the monitoring itself. A visit that spends two days
on the wrong records has the same cost as one that finds the problem.

## 12. Exercises

### Exercise 1. Derive critical-to-quality factors (application, 30 minutes)

A Phase II trial of an inhaled therapy has a primary endpoint of forced expiratory volume at week
12, a secondary endpoint of exacerbation rate, and a safety objective of detecting a rare cardiac
event. Consent is obtained at 30 sites, three of which have no prior trial experience.

**Deliverable:** a list of 8 to 12 CtQ factors, each with the data or process that proves it and the
layer of oversight most likely to detect a failure.

**Acceptance criteria:** each factor is stated as an outcome, not an activity; at least two factors
relate to process rather than data; the oversight layer is named for each.

**Time:** 30 minutes.

### Exercise 2. Score the risk register (application, 45 minutes)

Using the CtQ list from Exercise 1, score five failure modes on likelihood (1 to 5), impact (1 to 5),
and detectability (1 to 5, where 5 is hard to detect). Compute a priority score and rank them.

**Deliverable:** a risk register table with scores, rationale for each score, and a ranked list.

**Acceptance criteria:** rationale is specific enough that a second person could reproduce the
score; detectability is justified separately from impact; the top three risks have a proposed
control.

**Time:** 45 minutes.

### Exercise 3. Design KRIs and QTLs (design, 45 minutes)

For the same trial, define six site-level KRIs and three study-level QTLs. Each KRI needs a
definition, data source, threshold, and action. Each QTL needs a parameter, limit, measurement
method, and reporting plan.

**Deliverable:** two tables, one for KRIs and one for QTLs.

**Acceptance criteria:** no indicator without an action; thresholds distinguish normal variation
from a signal; at least one QTL addresses a process rather than a data value; the QTL reporting plan
names the section of the clinical study report where the outcome will appear.

**Time:** 45 minutes.

### Exercise 4. Compare monitoring resource plans (calculation, 45 minutes)

A trial has 30 sites and 450 patients. Plan A visits every site five times a year and verifies every
patient at 2.5 hours. Plan B tiers sites 6/18/6 with visits of 6, 3, and 1 per year, verifies 100% of
patients at the six high-tier sites and 25% elsewhere. A monitor costs $150,000 per year (2026 US
dollars), works 205 days per year, and has 1,350 productive hours.

**Deliverable:** FTEs and annual cost for each plan, and the percentage saving.

**Acceptance criteria:** visit and SDV components shown separately; the assumption about days per
visit stated; the saving expressed in both dollars and FTE.

**Time:** 45 minutes.

### Exercise 5. Model the lock date (calculation, 60 minutes)

A trial generates 8 queries per patient for 500 patients over 24 months. Cycle times are 10 days for
50% of queries, 20 days for 30%, and 50 days for 20%. The lock plan assumes all queries close within
20 days of LPLV. Fixed cost is $450,000 per month.

**Deliverable:** mean cycle time, open queries at LPLV, the realistic lock date, the slip against
plan, and the cost of the slip.

**Acceptance criteria:** Little's law applied correctly; the maximum, not the mean, used for the
lock; the cost stated in currency and year.

**Time:** 60 minutes.

### Exercise 6. Triage three signals (critique, 45 minutes)

You receive three signals in the same monthly review:

- Site 07 has a screen-failure rate of 55% against a study mean of 24%.
- Sites 03, 11, and 19 all show the same digit preference in systolic blood pressure.
- Site 22 has a query rate three times the study median, with a normal screen-failure rate.

**Deliverable:** a triage note for each signal with a probable cause, an action, an owner, and the
evidence that would close it.

**Acceptance criteria:** no signal is closed without a documented reason; the multi-site pattern is
treated differently from the single-site pattern; at least one action is remote and one is on-site.

**Time:** 45 minutes.

### Exercise 7. Write a CRO oversight plan (design, 60 minutes)

A sponsor delegates monitoring to a CRO. The CRO reports monthly on visits completed, queries
closed, and sites activated.

**Deliverable:** a one-page oversight plan specifying what the sponsor will review beyond the
aggregate metrics, at what frequency, and what would trigger escalation.

**Acceptance criteria:** includes a sample review of primary records; names the roles accountable on
both sides; defines at least three escalation triggers; distinguishes routine oversight from a
for-cause audit.

**Time:** 60 minutes.

### Exercise 8. Critique an inspection-readiness pack (critique, 45 minutes)

A sponsor's inspection pack contains the monitoring plan, the KRI dashboard, and a list of completed
visits. It does not contain the triage log, the CAPA records, or evidence that actions worked.

**Deliverable:** a two-page critique identifying the gaps, why each matters to an inspector, and
what to reconstruct before an inspection.

**Acceptance criteria:** each gap linked to a specific oversight question; reconstruction judged
feasible or not; no claim that missing records can be recreated retrospectively without qualification.

**Time:** 45 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

A strong list includes:

| CtQ factor | Evidence | Layer |
|---|---|---|
| Participants meet eligibility criteria | Eligibility fields and screening log | Central (screen-failure rate) and on-site (source) |
| Consent is informed and documented | Signed form, version, dated discussion | On-site document review |
| Allocation is concealed and correct | Randomisation record, IRT log | Central and on-site |
| Investigational product is stored and dispensed correctly | Temperature log, accountability record | On-site |
| Primary endpoint is measured to protocol | Spirometry records, calibration logs | On-site and central |
| Exacerbations are ascertained consistently | Adjudication records, event definitions | Central |
| Serious adverse events are reported on time | Safety database timestamps | Central |
| Data are attributable and contemporaneous | Audit trail, source documents | Central and on-site |

A weak list restates protocol sections rather than naming what must go right.

### 13.2 Exercise 2

There is no single correct score, but a strong register separates the components and justifies
detectability. For example:

| Failure mode | Likelihood | Impact | Detectability | Score |
|---|---:|---:|---:|---:|
| Eligibility misinterpreted at new sites | 4 | 4 | 3 | 48 |
| Consent form version out of date | 2 | 5 | 4 | 40 |
| Spirometer not calibrated | 3 | 4 | 2 | 24 |
| SAE reported late | 3 | 5 | 2 | 30 |
| Drug stored outside range | 2 | 4 | 3 | 24 |

The ranking puts consent above calibration even though calibration is more likely, because the
impact and the difficulty of detection are higher. A register that ranks by likelihood alone will
under-monitor the low-frequency, high-harm failures.

### 13.3 Exercise 3

A strong KRI table looks like this in structure:

| KRI | Definition | Source | Threshold | Action |
|---|---|---|---|---|
| Screen-failure rate | Screen failures / screened | EDC screening log | Above study mean + 2 SD | Review 10 screening logs |
| Consent query rate | Consent queries / patients | EDC | Any consent query | Document review at next contact |
| Query rate | Queries / patient | EDC | Above 2x median | Remote review of 20 records |
| Query age P90 | 90th percentile days open | EDC | Above 30 days | Escalate to PI |
| Visit window deviation | Visits outside window / visits | EDC | Above 10% | Retraining |
| SAE reporting delay | Median days from event to report | Safety database | Above 3 days | Process review |

A strong QTL table includes, for example, the proportion of participants with a major eligibility
deviation (limit 5%), the proportion with a primary endpoint outside the protocol window (limit
5%), and the proportion of serious adverse events reported within 24 hours (limit 95%). The reporting
plan names the clinical study report section where the QTL outcome and any remedial action will be
described.

### 13.4 Exercise 4

Plan A:

```text
Visits = 30 x 5 = 150
Visit FTE = 150 x 3 / 205 = 2.20
SDV hours = 450 x 2.5 = 1,125
SDV FTE = 1,125 / 1,350 = 0.83
Total FTE = 3.03
Cost = 3.03 x $150,000 = $454,500
```

Plan B:

```text
Visits = 6x6 + 18x3 + 6x1 = 36 + 54 + 6 = 96
Visit FTE = 96 x 3 / 205 = 1.40
SDV hours = 90 x 2.5 + 360 x 0.25 x 2.5 = 225 + 225 = 450
SDV FTE = 450 / 1,350 = 0.33
Total FTE = 1.73
Cost = 1.73 x $150,000 = $259,500
```

The saving is $195,000 per year, or 43%, and 1.30 FTE. The calculation assumes three days per visit
in both plans; if the risk-based plan uses shorter, more focused visits, the saving is larger, and
if the high-tier visits are longer, it is smaller.

### 13.5 Exercise 5

```text
Queries = 500 x 8 = 4,000 over 24 months = 166.7 per month
Mean cycle = 0.50x10 + 0.30x20 + 0.20x50 = 5 + 6 + 10 = 21 days
Open at LPLV = 166.7 x (21 / 30.44) = 115 queries
Last query closes = up to 50 days after LPLV (the 20% tail)
Planned lock = LPLV + 20 days
Realistic lock = LPLV + 50 days
Slip = 30 days
Cost = 30 x ($450,000 / 30) = $450,000
```

The answer depends on the assumption that at least one 50-day query is raised in the final days
before LPLV. With 4,000 queries, the probability of that is high; with 40 queries, it would not be.

### 13.6 Exercise 6

- **Site 07, high screen failure.** Probable cause: criteria interpreted strictly, or a different
  patient mix. Action: remote review of ten screening logs, then a call with the investigator.
  Owner: CRA. Closed when the reason for each failure is recorded and the pattern is explained.
- **Sites 03, 11, 19, digit preference.** Probable cause: a shared measurement device, a shared
  training source, or copying. Action: compare with the study distribution, review calibration
  records, and conduct a for-cause visit if the pattern persists. Owner: central monitoring lead.
  This is a multi-site signal and may indicate a study-level problem, not three site problems.
- **Site 22, high query rate with normal screen failure.** Probable cause: documentation or data
  entry problems, not eligibility. Action: remote review of query types; if concentrated in one
  form, retraining; if diffuse, a targeted visit. Owner: data manager and CRA.

### 13.7 Exercise 7

A strong plan includes:

- a monthly review of the CRO's aggregate metrics;
- a quarterly sample of five visit reports and five query histories, selected by risk rather than
  at random;
- an annual or trigger-based audit of the CRO's own quality system;
- named accountable roles on both sides;
- escalation triggers such as a rise in the query reopen rate, a fall in the SDV finding rate, or a
  site-level signal that the CRO did not act on within its own service level.

A weak plan repeats the CRO's metrics back to the CRO and calls that oversight.

### 13.8 Exercise 8

The pack answers "what did we plan" and "what did we do". It does not answer "what did we learn" or
"what changed". The triage log shows that signals were reviewed and decisions were made. The CAPA
records show that problems were corrected. The effectiveness evidence shows whether the correction
worked.

Reconstructing a triage log after the fact is possible only if the underlying correspondence and
meeting minutes exist. A reconstructed log that is not supported by contemporaneous records is worse
than no log, because it invites a question about why it was created late.

## 14. Interview questions

**1. What makes monitoring risk-based rather than calendar-based?**
Strong: allocation of effort follows a documented risk assessment, and the assessment is updated as
evidence arrives. Weak: says visits are less frequent.

**2. What is the difference between SDV and SDR?**
Strong: SDV checks that recorded values match source; SDR reviews source quality and process, such
as whether the measurement was taken correctly by the right person. Weak: uses the terms
interchangeably.

**3. How would you set a KRI threshold?**
Strong: from historical variation and the cost of a false alert, with a pre-defined action and a
review of the alert rate. Weak: picks a round number such as 20%.

**4. Why is a query median a poor guide to the database lock date?**
Strong: the lock depends on the last query, so the tail and the reopen rate matter more than the
median. Weak: says the median is representative.

**5. What would you do with a site flagged by central statistical monitoring?**
Strong: review the component variables and the clinical context, check data entry and device
issues, then choose remote review or a for-cause visit; document the reasoning. Weak: closes the
site or dismisses the flag without review.

**6. How do you oversee a CRO without duplicating its work?**
Strong: sample primary records and review escalation behaviour, not only aggregate metrics. Weak:
asks for more frequent reports.

**7. What evidence shows that a monitoring action worked?**
Strong: a before-and-after metric on the same site, such as query rate, deviation rate, or finding
rate at the next visit. Weak: says the site was retrained.

**8. When is 100% SDV justified?**
Strong: when the data are critical, the source is paper, the site is new, and the risk of
transcription error is high, usually for a limited period. Weak: says always or never.

**9. What does an inspector look for in a monitoring file?**
Strong: the risk assessment, the plan, the evidence of execution, the actions taken, and the
evidence they worked. Weak: lists documents without linking them to decisions.

## 15. Key terms

| Term | Meaning | See |
|---|---|---|
| ALCOA+ | Data integrity principles | 8.6 |
| Auto-query | Query generated by an edit check | 5.8 |
| CAPA | Corrective and preventive action | 5.10 |
| Central monitoring | Analysis of aggregated data for signals | 3.2 |
| CtQ | Critical-to-quality factor | 3.4 |
| DCF | Data clarification form | 4 |
| Edit check | Automated data validation rule | 5.8 |
| Effectiveness check | Evidence that an action worked | 5.10 |
| For-cause visit | Visit triggered by a specific concern | 5.18 |
| KRI | Key risk indicator | 3.4 |
| Monitoring plan | Document defining oversight by site tier | 5.6 |
| QTL | Quality tolerance limit | 4.3 |
| Query ageing | Time a query has been open | 5.12 |
| Reopen rate | Queries closed then reopened | 5.19 |
| Residual risk | Risk remaining after controls | 11 |
| RBM | Risk-based monitoring | 3.1 |
| RBQM | Risk-based quality management | 3.1 |
| SDV | Source data verification | 4.1 |
| SDR | Source data review | 4.1 |
| Signal | Pattern that may indicate a problem | 4.2 |
| Site risk score | Composite ranking of site risk | 5.7 |
| Statistical monitoring | Formal anomaly detection | 9.2 |
| Tier | Risk band that determines oversight | 5.15 |
| Triage log | Record of signal decisions and actions | 5.9 |

## 16. Sources

All URLs accessed 26 October 2026.

1. ICH. *E6(R3) Good Clinical Practice*, Step 4, 6 January 2025.
   https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
2. US Food and Drug Administration. *Oversight of Clinical Investigations: A Risk-Based Approach to
   Monitoring*, August 2013.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/oversight-clinical-investigations-risk-based-approach-monitoring
3. US Food and Drug Administration. *A Risk-Based Approach to Monitoring of Clinical Investigations:
   Questions and Answers*, April 2023.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/risk-based-approach-monitoring-clinical-investigations-questions-and-answers
4. US Government. *21 CFR 312.56, Review of ongoing investigations*.
   https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-312/subpart-D/section-312.56
5. TransCelerate BioPharma. *Risk Indicators and Thresholds*.
   https://www.transceleratebiopharmainc.com/rbminteractiveguide/best-practices-for-implementation/risk-indicators-and-thresholds/
6. TransCelerate BioPharma. *RBM Initiative Resources*.
   https://www.transceleratebiopharmainc.com/rbminteractiveguide/rbm-toolbox/initiative-resources/
7. Niangoran S, et al. Performance of four centralized statistical monitoring methods for early
   detection of an atypical center in a multicenter study. *Contemporary Clinical Trials
   Communications*. 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10328794/
8. Trotta L, et al. Detection of atypical data in multicenter clinical trials using unsupervised
   statistical monitoring. *Clinical Trials*. 2019.
   https://doi.org/10.1177/1740774519862564
9. Venet D, et al. A statistical approach to central monitoring of data quality in clinical trials.
   *Clinical Trials*. 2012. https://doi.org/10.1177/1740774512447898
10. Fronc M, et al. Central statistical monitoring in clinical trial management: a scoping review.
    *Clinical Trials*. 2025. https://doi.org/10.1177/17407745241304059
11. International Organization for Standardization. *ISO 14155:2020*.
    https://www.iso.org/standard/71690.html
12. European Commission. *Clinical Trials Regulation (EU) No 536/2014*.
    https://health.ec.europa.eu/medicinal-products/clinical-trials/clinical-trials-regulation_en
13. US Food and Drug Administration. *Conducting Clinical Trials With Decentralized Elements*,
    September 2024.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/conducting-clinical-trials-decentralized-elements

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_trial-operations.md`](01_trial-operations.md)
- Next topic: [`03_clinical-data-management.md`](03_clinical-data-management.md)
- HTA: [`04_health-technology-assessment.md`](04_health-technology-assessment.md)
- Module survey: [`../../course/25_trial_operations_hta_global_regulatory.md`](../../course/25_trial_operations_hta_global_regulatory.md)
- Labs: [`../../labs/`](../../labs/)
