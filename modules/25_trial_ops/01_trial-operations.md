# Trial Operations: Start-up, Site Activation, Enrolment, and Close-out

> Module 25 — Trial Operations, HTA, and Global Regulation
> Topic 01 of 07
> Estimated study time: 4 hours
> Prerequisites: none; Module 13 topic 04 is useful background
> Version 2026.3

## 1. Why this topic exists

A protocol is a document. A trial is an operating system: contracts, ethics committees, pharmacy
kits, laboratory ranges, shipping lanes, site coordinators, and a monthly enrolment number that
someone has to explain to a programme board. The distance between the document and the operating
system is where most timeline promises die.

The failure is rarely scientific. A study that cannot activate sites, cannot find patients, or
cannot close its database produces no usable evidence regardless of the quality of its design. The
cost is paid three times: once in direct operating expense for every extra month, once in delayed
revenue or delayed milestone payments, and once in the strategic position lost when a competitor
reaches the market first.

Trial operations is therefore a data problem before it is an administrative one. Every operational
decision consumes a number: an activation date, an accrual rate, a screen-failure proportion, a
query ageing distribution, a cost per patient. Most of those numbers are estimated informally,
recorded in a spreadsheet, and never revisited. The purpose of this topic is to make the numbers
explicit, connect them arithmetically, and show where a plausible-looking plan is arithmetically
impossible.

### 1.1 The pattern that repeats

1. A Phase III protocol targets 600 evaluable patients across 45 sites.
2. The plan divides 600 by 45 sites and by 0.55 patients per site per month, giving 24.2 months.
3. The plan omits 10% dropout, so the randomised target is 667, not 600.
4. The plan omits 28% screen failure, so sites must screen 926 patients, not 667.
5. The plan activates all 45 sites in month 1, although historical activation takes five to eight
   months and sites activate in waves.
6. The true deterministic completion is month 33.3, not month 24.2, and the probability of finishing
   within 30 months is about 2%.
7. The programme learns this at month 9, when cumulative enrolment is 119 against a plan of 205.

None of these steps requires new clinical knowledge. Each is arithmetic that was available before
the protocol was final. The failure is not that the estimate was wrong; it is that the estimate was
never structured in a way that could be checked.

### 1.2 Who is blocked

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Sponsor programme team | Readout date, milestone, budget | Depends on sites and regulators it does not control |
| Sites | Staff time on screening that fails | Receive eligibility criteria they did not write |
| Patients | Access, and continuity when a trial closes early | Have no visibility of the operational plan |
| CRO | Margin and credibility | Works to a scope and timeline it did not set |
| Investors or partners | Valuation date | Rely on the sponsor's forecast |
| Regulators | Predictable inspection workload | See the trial only at submission |

### 1.3 The industry question

> Can this protocol be executed at these sites, in this timeline, at this cost, and with data good
> enough to submit?

The question has four clauses because a trial can fail any one of them independently. A trial that
enrols on time but generates unusable data has failed. A trial with perfect data that finishes two
years late has also failed commercially.

### 1.4 Scope

This topic teaches the operating logic common to sponsor-led trials and the arithmetic that connects
a protocol to a date and a budget. It does not teach clinical judgement, therapeutic area knowledge,
or any sponsor's standard operating procedures (SOPs).

## 2. Learning outcomes

By the end of this topic you can:

1. **Decompose** a protocol into the operational artifacts it implies: visit schedule, eligibility
   rule table, investigational product (IP) supply plan, site training package, and data flow.
2. **Build** an activation and enrolment model from site-level assumptions, and compute the
   completion month with screen failure and dropout applied at the correct stage.
3. **Quantify** the monthly cost of delay from fixed and variable components, and separate direct
   operating cost from deferred revenue and milestone exposure.
4. **Diagnose** a failing enrolment plan from two leading indicators before the enrolment window
   closes.
5. **Choose** among contingency options (add sites, compress activation, broaden eligibility,
   extend the window) using probability of success, cost, and evidence risk.
6. **Assess** the operational impact of a protocol amendment, including the enrolment pause and the
   data reconciliation it forces.
7. **Plan** close-out so that database lock is not delayed by missing essential documents, open
   queries, or unresolved drug accountability.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

### 3.1 The execution chain

```text
Protocol
  |
  v
Operational specification -----> feasibility evidence
  |                                   |
  v                                   v
Site selection ------------------> activation plan
  |                                   |
  |   contract + ethics + regulatory + IP supply + training
  v                                   v
Activated site ------------------> first patient in (FPI)
  |                                   |
  v                                   v
Screening --- screen failure ---> randomisation
  |                                   |
  v                                   v
Treatment and follow-up ---------> last patient last visit (LPLV)
  |                                   |
  v                                   v
Data cleaning and coding --------> database lock (DBL)
  |                                   |
  v                                   v
Analysis -------------------------> clinical study report (CSR) -> submission
```

The chain has two properties that dominate planning. First, it is partly serial: a site cannot
randomise before it is activated, and the database cannot be locked before the last visit. Second,
the serial links have different owners: the sponsor owns activation inputs, the site owns
screening, and the sponsor owns cleaning. A delay at any link moves every later link.

### 3.2 Three clocks

| Clock | Starts | Dominant constraint | Consequence of slipping |
|---|---|---|---|
| Regulatory | Submission of the application | Regulator review and ethics opinion | Site cannot start; enrolment window compresses |
| Enrolment | First site activation | Eligible patients per site per month | Readout date, cost, competitive position |
| Data | First patient in | Query resolution and coding throughput | Database lock, analysis, submission |

The clocks interact. A slow regulatory clock compresses the enrolment window, which forces more
sites or a longer enrolment period, which increases cost and extends the data clock because the last
patient arrives later. Operational plans that manage the clocks separately miss the coupling.

### 3.3 Activation is a distribution, not a date

Activation is the point at which a site can randomise its first patient. It requires, at minimum:

- an executed contract and budget;
- an ethics committee or institutional review board (IRB/IEC) approval;
- regulatory authorisation where the jurisdiction requires a site-level or trial-level approval;
- IP available at the site or via a depot;
- trained staff with system access and a delegation log.

Each input has its own distribution. Published start-up data show a median of 255 days from
protocol delivery to first patient enrolled and 171 days to site activation, with an interquartile
range of 114 to 246 days (Goyal et al., 2021). A plan that writes "all sites active month 1" has
removed the largest source of variance in the first year.

```text
 Cumulative
 activated sites
   45 |                                      ............
      |                             .........
   30 |                    .........
      |            ........
   15 |     .......
      | ....
    0 +----|----|----|----|----|----|----|----|----> months
          1    2    3    4    5    6    7    8    9
```

### 3.4 Accrual is the area under the active-site curve

Monthly enrolment at month *m* equals the number of sites active at *m*, multiplied by the
per-site rate, multiplied by a maturity factor for sites that have just started. Cumulative
enrolment is the sum over months. Two plans with the same final site count but different activation
curves have different cumulative enrolment, and the difference is the area between the curves.

This is why adding sites late in a trial is a weak rescue. A site activated with six months left in
the window contributes at most six months of accrual, of which the first two are ramping. The cost
of activating it is paid in full.

### 3.5 The cost of delay is a rate

Delay cost has four components:

1. **Fixed site and vendor costs.** Site payments often include a monthly component that continues
   while the site is open; internal full-time equivalents (FTEs), CRO fees, and licences accrue too.
3. **Variable costs driven by the enrolment curve.** Laboratory kits, drug supply, and monitoring
   visits follow patients, not calendar time, so a delay with the same patient count does not
   multiply them.
4. **Financial exposure.** Deferred revenue, milestone penalties, and lost exclusivity period.

Only components 1, 2, and 4 scale with delay. A model that multiplies total trial cost by the delay
fraction overstates the direct cost, and a model that counts only site payments understates it.

### 3.6 The control loop

```text
 Plan (curve + triggers)
    |
    v
 Monthly actuals ----> variance decomposition
    |                        |
    |                        +--> activation variance
    |                        +--> accrual-per-site variance
    |                        +--> screen-failure variance
    v
 Trigger met? -- no --> continue
    |
   yes
    v
 Pre-approved action: add sites | compress activation | amend eligibility | extend window
    |
    v
 Reforecast and record the decision
```

A plan without triggers is a forecast. A plan with triggers is a control system. The difference is
whether the organisation acts at month 9 or at month 30.

### 3.7 What the model cannot see

The arithmetic model treats sites as interchangeable capacity. They are not. Sites differ in
referral networks, competing trials, staff turnover, and enthusiasm. A model that uses a single
per-site rate will produce a plausible median and an implausible tail, because the tail is driven by
the sites that enrol nobody. Section 6 shows how to diagnose this with a concentration check.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Sponsor | Organisation responsible for the trial | Used to mean the funder only | Role |
| Contract research organisation (CRO) | Vendor that performs trial functions under contract | Assumed to absorb the sponsor's accountability | Role |
| Principal investigator (PI) | Site-level lead responsible for conduct | Confused with the sponsor's medical lead | Role |
| Site coordinator | Staff member who runs visits and data entry | Assumed to be interchangeable with a nurse | Role |
| Site activation | Point at which a site may randomise its first patient | Confused with contract signature | Date |
| Site initiation visit (SIV) | Training and readiness visit before activation | Treated as a formality | Event |
| First patient in (FPI) | Date the first patient is randomised or dosed at a site | Used as the trial-level start date | Date |
| Last patient in (LPI) | Date the last patient is randomised | Confused with last patient last visit | Date |
| Last patient last visit (LPLV) | Date the last patient completes the last protocol visit | Confused with database lock | Date |
| Database lock (DBL) | Declared point after which the database cannot be changed | Declared with open queries | Milestone |
| Accrual | Rate at which patients are enrolled | Used interchangeably with cumulative enrolment | Patients per site per month |
| Enrolment | The event of randomisation or allocation | Used to mean screening | Count |
| Screening | Assessment of eligibility after consent | Confused with pre-screening | Activity |
| Screen failure | A consented patient who does not meet eligibility or withdraws before randomisation | Counted as enrolled | Count or proportion |
| Screen-to-randomise ratio | Screened patients per randomised patient | Confused with the screen-failure rate | Ratio |
| Dropout | Loss of a randomised patient before the primary endpoint | Ignored in sample-size planning | Proportion |
| Retention | Proportion of randomised patients still participating | Confused with adherence | Proportion |
| Protocol deviation | Departure from the protocol not approved in advance | Confused with a violation | Event |
| Protocol waiver | Prospective sponsor approval to deviate | Treated as retrospective | Document |
| Eligibility criteria | Rules defining who may enter | Written without operational definitions | Protocol section |
| Investigational product (IP) | The drug, biologic, or device under study | Used to include comparator by default | Material |
| Comparator | The intervention against which the IP is compared | Assumed to be standard of care | Intervention |
| Randomisation | Allocation of participants to arms | Confused with enrolment | Procedure |
| Stratification | Allocation balanced on pre-specified factors | Omitted from operational planning | Design factor |
| Electronic data capture (EDC) | System for entering trial data | Treated as the whole data system | System |
| Clinical trial management system (CTMS) | System for site, milestone, and enrolment tracking | Confused with EDC | System |
| Electronic patient-reported outcome (ePRO) | Patient-entered outcome data | Assumed equivalent to paper | System |
| Electronic consent (eConsent) | Digitally captured informed consent | Assumed to remove the consent conversation | System |
| Decentralised trial (DCT) | Trial with remote or local elements replacing site visits | Assumed to reduce cost automatically | Design |
| Essential documents | Records proving the trial was conducted to standard | Confused with the full trial file | Document set |
| Trial master file (TMF) | Sponsor's and investigator's record of the trial | Assumed to be complete at lock | Document collection |
| Investigational product accountability | Records of receipt, dispensing, and return | Treated as a pharmacy task only | Log |
| Site close-out visit (COV) | Final monitoring visit that reconciles records and IP | Confused with the last monitoring visit | Event |
| Clinical study report (CSR) | Integrated report of the trial | Confused with the publication | Document |
| Fair market value (FMV) | Defensible price for site services | Set by precedent without analysis | Currency per unit |

### 4.1 Enrolment, accrual, and recruitment

Recruitment is the activity of finding patients. Enrolment is the event of randomisation. Accrual
is the rate. A site can recruit actively and enrol nobody because the patients it finds are
ineligible. A plan that says "recruitment rate" without naming the denominator cannot be compared
with site data, because sites report screened, randomised, and treated counts separately.

### 4.2 Activation, initiation, and first patient in

Activation is a permission. Initiation is a visit. First patient in is an event. The three can be
weeks apart. Contracts often key payment to activation, while the sponsor's timeline keys to FPI,
so the gap between them is a hidden schedule risk. In the published start-up data, a further 66 days
elapsed between activation and first enrolment.

### 4.3 Deviation, violation, and waiver

A deviation is any departure from the protocol, whether or not it affects safety or data. A
violation is a serious deviation that undermines participant safety or data reliability. A waiver is
prospective permission from the sponsor to depart from a protocol requirement. Using "violation" for
every deviation inflates the serious category; using "deviation" for a waiver hides a governance
failure.

### 4.4 Lock, unblinding, and analysis

Database lock is a declaration that the data are complete and correct enough for analysis. It is not
the same as unblinding, which is the release of treatment codes, and it is not the same as the
analysis-ready dataset, which is produced after lock by transformation into analysis datasets.
Confusing the three produces timelines that are short by weeks.

## 5. The mechanics, step by step

This procedure takes a protocol from an approved document to an archived trial. It assumes a
sponsor-led interventional study with at least one external site. The procedure is written for a
drug trial; the device-specific differences are noted where they change the step. Each step states
its input, its output, and the reason it exists, because a step whose output nobody uses is a step
that will be skipped under pressure.

### Step 1. Convert the protocol into an operational specification

**Input:** the final protocol, the investigator's brochure (IB) or device instructions for use, and
the statistical analysis plan (SAP) outline.

**Output:** an operational specification containing:

- a visit schedule with permitted windows and the assessments performed at each visit;
- an eligibility rule table that names the data field, the threshold, the time window, and whether
  the criterion is objective or judgement-based;
- an IP handling table covering storage, dispensing, accountability, and destruction;
- a sample flow diagram from collection to analysis, including centrifugation, freezing, and
  shipping conditions;
- a list of source documents expected at each visit;
- the endpoint definitions in the form the sites will have to collect them.

**Rationale:** Every ambiguity in the protocol is resolved at the site, and different sites resolve
it differently. A visit window of "week 12 plus or minus two weeks" becomes three different
scheduling conventions and a data point that cannot be compared. The operational specification is
the artefact that fixes the interpretation before the first patient arrives, and it is the reference
for training, monitoring, and edit checks.

**Branch point:** in a device trial, the equivalent artefact must also specify device accountability,
sterilisation or reprocessing, and the handling of device malfunctions. In an adaptive or
platform trial, it must specify what changes when an adaptation is triggered.

### Step 2. Build the feasibility evidence base

**Input:** the eligibility rule table, available patient-level data (claims, electronic health
records, registry), historical site performance, and the competing-trial landscape.

**Output:** a patient funnel from source population to randomised patients, a site-level expected
accrual distribution, and a written statement of which assumptions are evidence-based and which are
judgement.

**Rationale:** Feasibility fails when three different questions are collapsed into one number: how
many patients have the disease, how many are visible in the data, and how many can reach a
participating site. Separating them makes the error auditable. A funnel that produces 229 eligible
patients from a 2.4 million catchment can be checked. A statement that "the sites can enrol 600
patients" cannot.

**Branch point:** for a common disease with many sites, a per-site rate model is appropriate. For a
rare disease, the constraint is diagnosis and travel, and the relevant methods are patient
identification networks, registry recruitment, and referring-physician outreach. A per-site rate
model applied to a rare disease will produce a confident and meaningless forecast.

### Step 3. Design the site network

**Input:** the feasibility output, the country strategy, and the evidence requirements of the target
regulators and health technology assessment (HTA) bodies.

**Output:** a ranked site list with a primary and backup tier, an activation wave plan, a country
mix, and a documented rationale for each strategic site.

**Rationale:** Enrolment is concentrated. A minority of sites typically produces the majority of
patients, and a substantial minority produces none. The site network should be designed around the
sites likely to be in the productive minority, with enough backup capacity to absorb the ones that
underperform. Country mix is not only an enrolment decision: it determines which regulators see the
data, which populations are represented, and which data-transfer regimes apply.

**Branch point:** a single-country trial simplifies regulatory and data-transfer work but concentrates
commercial and enrolment risk. A multi-country trial diversifies enrolment but adds ethics
submissions, import licences, language validation, and country-specific standard-of-care differences
that can change the comparator.

### Step 4. Price and contract the sites

**Input:** the operational specification, a fair-market-value analysis, the sponsor's contract
template, and the site's institutional requirements.

**Output:** a site budget with per-patient, per-visit, pass-through, start-up, and overhead components;
a payment schedule tied to milestones; and an executed contract.

**Rationale:** Contract execution is one of the two largest components of activation time. Published
data put median contract execution at 143 days, close to the 132-day median for regulatory approval,
and the two run in parallel rather than in sequence. A budget that is below the site's cost of doing
the work does not save money; it produces renegotiation, delayed activation, or silent cost shifting
into per-patient rates.

**Branch point:** sites covered by a master agreement or a central contracting vehicle activate
faster. Public hospitals and academic institutions often require indemnity, publication, and
intellectual-property terms that the sponsor's template does not contain, and those terms are
negotiated by the institution's office, not the investigator.

**Edge case:** currency and tax treatment change the effective budget. A site budget denominated in
a currency that depreciates against the sponsor's reporting currency loses value during a long
start-up, and the site may reopen the budget. Anti-kickback rules in the United States and equivalent
rules elsewhere constrain what can be paid for, and to whom.

### Step 5. Prepare regulatory and ethics submissions

**Input:** the protocol, the IB, the investigational medicinal product dossier (IMPD) or device
technical file, the operational specification, and the site list.

**Output:** a submission plan showing, for each country, the regulator, the ethics body, the
application format, the expected review clock, and the dependencies between them.

**Rationale:** Regulatory authorisation and ethics approval are separate decisions made by separate
bodies, and neither guarantees the other. In the United States, an investigational new drug (IND)
application is reviewed by the Food and Drug Administration (FDA) while site-level review is
performed by an IRB. In the European Union, a trial application is submitted through the Clinical
Trials Information System (CTIS) and assessed by the member states concerned, with ethics assessment
organised nationally. A plan that sequences these as "regulatory, then ethics, then contract" adds
months that could have been parallel.

**Branch point:** central versus local IRB review changes activation time materially. In the
cardiovascular start-up cohort cited above, sites using a central IRB activated in a median of 199
days, compared with 287 days for local IRB review.

### Step 6. Secure IP supply and the import pathway

**Input:** the dosing plan, the enrolment forecast, shelf-life and stability data, packaging and
labelling requirements, and the comparator sourcing strategy.

**Output:** a supply plan with a first-shipment date, depot and site inventory targets, resupply
triggers, and an interactive response technology (IRT) specification for allocation and dispensing.

**Rationale:** An activated site without IP cannot enrol. Supply failures are usually caused by
forecasting the shipment against the plan rather than against the upper bound of the enrolment
distribution, by labelling that does not meet the destination country's requirements, or by a
customs classification that was never tested. Comparator supply adds a second problem: the foreign
product may be labelled differently, may require re-labelling, and may be subject to a separate
import authorisation.

**Edge case:** controlled substances require additional national licences, and those licences are
often held by the site rather than the sponsor. A trial that assumes the sponsor's import licence
covers a controlled comparator will discover the gap at the first shipment.

### Step 7. Activate and train the site

**Input:** the executed contract, approvals, IP availability, the training package, and system
access.

**Output:** a signed activation checklist, a delegation log, training records, and a site file that
can survive an inspection.

**Rationale:** Activation is a verified state, not a date in a spreadsheet. The checklist is what
converts the inputs into an assertion that the site can conduct the protocol. Missing training
records and incomplete delegation logs are among the most common inspection findings, and they are
cheap to prevent at activation and expensive to reconstruct at close-out.

**Branch point:** a remote or hybrid site initiation visit is acceptable in many settings and can
shorten activation, but it shifts risk into the first monitoring visit. The decision should depend on
site experience and protocol complexity, not on travel budget alone.

### Step 8. Drive first patient in and the ramp

**Input:** the activated site, the pre-screening list, the referral pathway map, and a site-level
enrolment target.

**Output:** a first-patient-in date, and 30-, 60-, and 90-day accrual actuals.

**Rationale:** The gap between activation and first enrolment is a distinct operational problem. In
the published cohort the median was 66 days. Sites that activate and then wait for patients have
usually not built a referral pathway: they have a PI who is willing but no systematic way to find
eligible patients. Pre-screening before activation, and a named referral route from primary care or
community practice, are the interventions that shorten this gap.

### Step 9. Forecast and control enrolment

**Input:** monthly site-level actuals for activation, screening, randomisation, and screen failure.

**Output:** a reforecast distribution, a variance decomposition, and a recommendation against
pre-agreed triggers.

**Rationale:** A single cumulative number cannot tell the programme whether the problem is
activation, per-site accrual, or eligibility. Decomposing the variance into those three components
determines which lever will work. If activation explains most of the gap, adding sites does not help
until activation is fixed. If per-site accrual explains it, adding sites may help, but only if the
new sites can activate inside the remaining window.

**Branch point:** the trigger threshold should be set before enrolment starts. A common design is a
review at month 9 against 80% of the planned cumulative curve, with a second review at month 15.
The threshold is a governance decision, not a statistical one, and it should be documented with the
action it authorises.

### Step 10. Manage protocol amendments

**Input:** the proposed change, the reason, and the current enrolment and data status.

**Output:** an impact assessment covering regulatory submissions, ethics submissions, re-consent,
retraining, database changes, statistical consequences, budget, and timeline; an amendment package;
and a communication plan.

**Rationale:** An amendment is not a document change. It is a change to a running system. Sites must
be trained, patients may need to be re-consented, the EDC must be updated, and the analysis plan may
need to change. A substantial amendment in the European Union requires assessment before
implementation, which can pause enrolment in the affected countries. An impact assessment that
counts only the writing time understates the disruption by an order of magnitude.

**Branch point:** the treatment of patients already enrolled depends on whether the change affects
their safety, their consent, or only logistics. A change to a visit window can be implemented
prospectively. A change to an eligibility criterion creates a mixed population that must be handled
in the analysis, not only in the operations.

### Step 11. Manage deviations and corrective actions

**Input:** monitoring findings, site reports, and central statistical signals.

**Output:** a deviation log with root cause, a corrective and preventive action (CAPA) plan, and
evidence of effectiveness.

**Rationale:** The purpose of the deviation log is to detect patterns, not to record incidents. Ten
isolated deviations at ten sites mean something different from ten at one site. A CAPA that is
closed when the training is delivered, without a follow-up measure, is a documentation exercise. The
measure of success is a reduction in the recurrence rate, not the existence of a form.

### Step 12. Plan close-out from last patient in

**Input:** the LPI date, the site-level query backlog, the IP accountability status, and the
essential-document completeness report.

**Output:** a close-out schedule that sequences sites by risk and volume, a query rundown target, and
a resource plan for the close-out period.

**Rationale:** Close-out is a queue with a fixed service rate. Each site needs a final monitoring
visit, IP reconciliation, essential-document review, and payment reconciliation. If 45 sites close
at four per week, the queue takes 11 weeks before the last close-out visit, and the database cannot
lock until the sites that hold the remaining queries are resolved. Planning the queue from LPI,
rather than from the date someone notices it is urgent, is the difference between a lock on the
planned date and a lock three months late.

### Step 13. Execute close-out

**Input:** the close-out schedule and the site files.

**Output:** a close-out visit report per site, reconciled IP, a complete essential-document set, and
a site closure letter.

**Rationale:** Close-out is where missing documentation becomes visible. A delegation log that was
never signed, a training record that was never filed, or a drug accountability discrepancy that was
never resolved cannot be fixed after the site staff have moved to another study. The visit should
reconcile the story the site file tells against the story the data tell.

### Step 14. Lock, report, register, and archive

**Input:** the cleaned database, the SAP, the TMF, and the registry record.

**Output:** a database lock certificate, analysis datasets, a CSR, a registry results submission,
and an archive that meets the applicable retention period.

**Rationale:** Lock is a governance event with a defined evidence base. It should be declared only
when pre-specified lock criteria are met: no open queries affecting primary or key secondary
endpoints, medical coding complete, external data reconciled, and the blind review completed where
applicable. The downstream obligations have their own clocks: in the United States, results for an
applicable clinical trial are due no later than one year after the primary completion date under
42 CFR 11.44, and in the European Union the CTR requires a summary of results within a defined
period after the end of the trial.

### 5.15 Decision table: activation strategy

| Strategy | When it fits | Effect on first patient in | Main risk |
|---|---|---|---|
| Activate all sites in one wave | Small network, mature sites, master agreements in place | Earliest if contracts and approvals are ready | Capital and training spent on sites that may not enrol |
| Staggered waves by region | Multi-country trial with different review clocks | First wave starts early; later waves follow approvals | Later waves may arrive after the enrolment window has closed |
| Sequential by country, starting with the fastest regulator | Competitive indication, need for early data | Early signal, later global capacity | Population differences across waves; regulator questions on generalisability |
| Activate a small high-performing core, then expand | Uncertain accrual, adaptive design | Fast start with low wasted spend | Core sites may be saturated; expansion takes months |

The choice is not about which strategy is best in general. It is about whether the programme needs
the earliest possible first patient, the earliest possible last patient, or the lowest cost per
randomised patient. Those are different objectives and they produce different activation plans.

### 5.16 Decision table: response to slow enrolment

| Diagnosis at trigger | Evidence | First action | If the action fails |
|---|---|---|---|
| Activation behind plan | Fewer sites active than planned; accrual per active site on plan | Fix the binding activation input: contract, ethics, or IP | Replan the window; adding sites will not help |
| Accrual per active site below plan | Active sites on plan; screens low | Referral pathway work; site-level coaching; pre-screening support | Replace or supplement underperforming sites |
| Screens normal, randomisations low | High screen-failure rate | Review eligibility interpretation; consider a substantial amendment | Amend criteria and accept a mixed population |
| Screens low, eligible pool adequate | Few referrals, competing trials | Add sites in different catchments; patient identification support | Extend the window |
| Enrolment on plan but concentrated | Two or three sites produce most patients | Protect those sites; reduce monitoring burden; secure supply | Diversify before a site closes |
| Enrolment ahead of plan | Cumulative above the upper curve | Slow accrual to protect data quality and supply | Do not over-enrol beyond the statistical plan without review |

### 5.17 Decision table: timing of a protocol amendment

| Timing | Regulatory and ethics burden | Effect on enrolled patients | Recommended handling |
|---|---|---|---|
| Before first patient in | Resubmission of the package; review clock restarts in some jurisdictions | None | Implement as the initial protocol; no amendment category |
| During enrolment, change affects safety or eligibility | Substantial amendment; assessment before implementation in the EU | Re-consent may be required; mixed population | Pause affected sites; pre-specify the analysis handling |
| During enrolment, logistics only | Notification or non-substantial amendment depending on jurisdiction | None | Implement prospectively with retraining |
| After last patient in, before lock | Variation to the analysis rather than the conduct | None, but data already collected may not support the change | Treat as an SAP amendment, not a protocol amendment |

### 5.18 Edge cases

**Site under-enrolment.** The site is active, has patients, and enrols at a third of the planned
rate. The first question is whether the site's eligible population was overestimated or its effort
is diverted. The second is whether the protocol's eligibility criteria are being interpreted more
strictly than intended. Adding a new site to compensate takes five to eight months and rarely helps
inside a 12-month window.

**Protocol amendment mid-study.** The operational cost is in retraining, re-consent, system changes,
and the enrolment pause. The evidence cost is a mixed population. Both must be estimated before the
amendment is approved, not after.

**Database lock with open queries.** Lock can proceed with open queries if they affect neither the
primary endpoint nor safety, and if the omission is documented and handled in the analysis. It
cannot proceed if the queries affect the endpoint, the randomisation, or the analysis population.
The decision rule should be written into the lock checklist rather than negotiated at the meeting.

**Cross-border data transfer.** Patient data moving from the EU to a sponsor outside the EU requires
a lawful transfer mechanism, and data moving to or from China is subject to a separate security
assessment regime. Operational plans that assume the EDC vendor's hosting region solves the problem
discover it at the first data transfer, when the sites are already enrolling. The transfer
mechanism, the de-identification approach, and the destination of the analysis datasets should be
settled at site selection.

**Payer rejection downstream.** A payer rejection does not stop a trial, but it changes what the
trial must produce. If reimbursement depends on a head-to-head comparison, a quality-of-life
endpoint, or a subgroup analysis that the protocol does not include, the operational plan must
accommodate an additional study or an amendment. The HTA topics in this module explain how that
requirement arises.

**Price negotiation failure.** When a price agreement fails after launch, the sponsor may run a
named-patient or early-access programme while negotiating. Those programmes create data and safety
reporting obligations that sit outside the trial, and they consume the same medical affairs and
pharmacovigilance capacity that the trial needs for close-out.

### 5.19 What good looks like

| Metric | Definition | Typical use |
|---|---|---|
| Activation cycle time | Days from site selection to activation | Start-up performance |
| Activation-to-FPI interval | Days from activation to first randomised patient | Site readiness |
| Accrual per active site per month | Randomisations divided by active site-months | Enrolment forecast input |
| Screen-to-randomise ratio | Screened divided by randomised | Eligibility feasibility |
| Enrolment against plan | Cumulative actual divided by cumulative plan | Trigger monitoring |
| Query ageing | Median and 90th percentile days open | Data cleaning capacity |
| Close-out completion | Sites closed divided by sites planned | Close-out queue management |
| Lock delay | Days from LPLV to DBL against plan | Data operations performance |

The metrics are only useful if they are defined the same way every month. A query-ageing metric that
restarts when a query is reissued will always look good.

## 6. Worked example

### 6.1 The trial

A Phase III trial in a common oncology indication. The protocol states a primary analysis population
of 600 evaluable patients, randomised 1:1, with a primary endpoint measured at 12 months. The
sponsor plans 45 sites across three countries. The programme has committed to a database lock that
supports a partner milestone payment of $15,000,000 (2026 US dollars) by month 33 from first site
activation.

The planning assumptions supplied by the operations team are:

| Assumption | Value | Source |
|---|---|---|
| Evaluable target | 600 patients | Protocol sample-size section |
| Dropout before the primary endpoint | 10% | Prior trial in the indication |
| Screen-failure rate | 28% | Site questionnaire and prior trial |
| Sites | 45 | Site selection plan |
| Activation | Three waves of 15 at months 1, 4, and 7 | Start-up plan |
| Steady-state accrual | 0.50 patients per site per month | Historical site performance |
| Site maturity | 50% of steady state in month 1, 80% in month 2, 100% from month 3 | Assumption |
| Fixed operating cost | $600,000 per month | Finance estimate |

### 6.2 Step 1: fix the denominators

The sample size is stated on the evaluable population. The randomised target must therefore be:

```text
Randomised target = 600 / (1 - 0.10) = 666.7 -> 667 patients
Screened target   = 667 / (1 - 0.28) = 926.4 -> 927 patients
```

The screen-to-randomise ratio is 927/667 = 1.39. Each randomised patient costs the site 1.39
screens. A plan that budgets site work from 600 patients understates screening workload by 35%.

### 6.3 Step 2: build the monthly curve

Each wave contributes 15 sites. A site contributes 0.50 × 0.5 = 0.25 patients in its first active
month, 0.50 × 0.8 = 0.40 in its second, and 0.50 from its third month onward.

| Month | Sites active | Monthly randomisations | Cumulative | Plan cumulative | Actual as % of plan |
|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 3.75 | 3.75 | 12.38 | 30% |
| 2 | 15 | 6.00 | 9.75 | 32.18 | 30% |
| 3 | 15 | 7.50 | 17.25 | 56.93 | 30% |
| 4 | 30 | 11.25 | 28.50 | 81.68 | 35% |
| 5 | 30 | 13.50 | 42.00 | 106.43 | 39% |
| 6 | 30 | 15.00 | 57.00 | 131.18 | 43% |
| 7 | 45 | 18.75 | 75.75 | 155.93 | 49% |
| 8 | 45 | 21.00 | 96.75 | 180.68 | 54% |
| 9 | 45 | 22.50 | 119.25 | 205.43 | 58% |
| 10 | 45 | 22.50 | 141.75 | 230.18 | 62% |
| 11 | 45 | 22.50 | 164.25 | 254.93 | 64% |
| 12 | 45 | 22.50 | 186.75 | 279.68 | 67% |

The plan cumulative column is the curve the programme actually committed to: 45 sites active from
month 1 at 0.55 patients per site per month. It reaches 205.43 by month 9.

From month 9 the actual curve adds 22.50 per month. To reach 667:

```text
Remaining after month 9 = 667 - 119.25 = 547.75
Months at steady state  = 547.75 / 22.50 = 24.34
Completion month        = 9 + 24.34 = 33.34
```

The deterministic completion is month 33.3, nine months later than the 24.2-month figure obtained by
dividing 600 by 45 × 0.55. The difference is entirely the two denominators and the activation curve.

### 6.4 Step 3: price the delay

The fixed operating cost is $600,000 per month (2026 US dollars), composed of site payments
($103,500), internal team ($252,000), CRO fixed fees ($180,000), and other fixed cost ($64,500).
Variable costs driven by patient volume are unchanged because the patient count is unchanged.

```text
Direct delay cost = (33.33 - 27.64) x $600,000 = 5.69 x $600,000 = $3,416,000
```

The comparison is against a correctly specified plan that activates all sites in month 1 at 0.55
patients per site per month, which completes in month 27.64. Against the original 24.2-month
promise, the gap is 9.1 months and $5,460,000.

The milestone is deferred, not cancelled. At an 8% annual cost of capital, the carry on
$15,000,000 for the additional 5.69 months is:

```text
Carry = $15,000,000 x 0.08 x 5.69 / 12 = $569,000
```

Total quantified exposure is approximately $3,985,000, before any allowance for lost competitive
position.

### 6.5 Diagnostic 1: the denominator does not reconcile

Take the plan's own site workload. If the plan assumes 600 patients and 45 sites, it implies each
site screens 13.3 patients over the enrolment period. The protocol requires each site to screen
927/45 = 20.6 patients. The gap of 7.3 screens per site is the hidden workload that appears as
"the site is slow" at month 6.

This diagnostic is available before the first patient. It requires only the protocol, the dropout
assumption, the screen-failure assumption, and the site count. A plan that fails it cannot be fixed
by better site management, because the sites were never resourced for the work.

### 6.6 Diagnostic 2: the curve shape was never achievable

At month 9 the trial has enrolled 119.25 patients against a plan of 205.43, or 58% of plan. The naive
reading is that the trial is 42% behind and needs a rescue. The correct reading is that the plan
assumed all 45 sites were active in month 1.

Refitting the curve with the actual activation dates and the actual per-site rate reproduces the
actual cumulative almost exactly (119.25 predicted versus 119.25 observed). The trial is running
exactly as the activation plan implies. The plan, not the execution, was wrong.

That distinction changes the action. A rescue aimed at site productivity would spend money on
coaching sites that are already performing. A rescue aimed at activation would compress the
remaining waves, which is the lever that moves the curve.

### 6.7 Decomposing the variance

| Scenario | Activation | Rate | Completion month | Difference from plan |
|---|---|---|---:|---:|
| Plan as promised | All 45 at month 1 | 0.55 | 27.64 | 0.00 |
| Activation only | Three waves | 0.55 | 30.64 | +3.00 |
| Activation and rate | Three waves | 0.50 | 33.33 | +5.69 |

The decomposition is sequential, so the order matters. Reversing the order attributes 2.70 months to
the rate and 2.99 months to activation. What is stable across both orderings is that activation
accounts for roughly half the gap. The lesson is not the exact split but the fact that two distinct
causes exist and require two distinct responses.

### 6.8 Vary one assumption: compress activation

Suppose the sponsor invests in start-up capacity, moving the three waves to months 1, 2, and 3 while
leaving the per-site rate at 0.50. The curve becomes:

```text
Month 1: 3.75   cumulative  3.75
Month 2: 9.75   cumulative 13.50
Month 3: 17.25  cumulative 30.75
Month 4: 21.00  cumulative 51.75
Month 5: 22.50  cumulative 74.25
Remaining: 667 - 74.25 = 592.75
Months at steady state: 592.75 / 22.50 = 26.34
Completion: 5 + 26.34 = 31.33
```

Completion moves from 33.33 to 31.33, a saving of 2.00 months and $1,200,000 in fixed operating
cost. The milestone carry falls by $200,000. If the start-up investment costs less than $1,400,000
and does not delay any other trial in the portfolio, it is worth doing on financial grounds alone.

The conclusion changes again if the compressed activation brings first patient in earlier and the
trial is competing for the same patients. In a competitive indication, two months of earlier
exposure can matter more than the direct saving, but that value is not in this model and should be
argued separately rather than folded into the cost case.

### 6.9 What the example teaches

Three things transfer to any trial:

1. The denominator is the most common error and the cheapest to check.
2. A cumulative shortfall at month 9 is not evidence of poor site performance until the plan's
   activation assumptions have been refitted.
3. The cost of delay is computable from the fixed-cost rate, and it is large enough to justify
   spending on activation before the trial starts.

The example also shows the limit of deterministic arithmetic. It produces one date. The decision
needs a distribution, which is what Section 7 builds.

## 7. Code, query, or calculation

The model below reproduces the worked example and adds a Monte Carlo layer. It uses only the Python
standard library, so it runs in any environment with Python 3.9 or later.

**Assumptions encoded:**

- randomisation target = 600 / (1 - 0.10);
- screening target = randomised / (1 - 0.28);
- activation months are known for the deterministic case;
- per-site rate is 0.50 patients per site per month in the base case;
- site maturity is 50%, 80%, then 100% of steady state;
- Monte Carlo activation jitter is normal with standard deviation 1.2 months, truncated at month 1;
- Monte Carlo site rates are gamma distributed with shape 4 and the requested mean, giving a
  right-skewed distribution in which some sites enrol far more than the mean and many enrol less;
- fixed cost is $600,000 per month in 2026 US dollars.

```python
import random
import statistics

RAMP = (0.50, 0.80, 1.00)

def completion_month(waves, rate, target, horizon=72):
    """Deterministic month of reaching target. waves = [(start_month, n_sites)]."""
    sites = []
    for start, n in waves:
        sites.extend([start] * n)
    cumulative = 0.0
    for month in range(1, horizon + 1):
        increment = 0.0
        for activation in sites:
            if month >= activation:
                k = month - activation
                increment += rate * (RAMP[k] if k < len(RAMP) else 1.0)
        previous = cumulative
        cumulative += increment
        if cumulative >= target:
            return month - 1 + (target - previous) / increment
    return float("inf")

def simulate(waves, rate_mean, target, n_iter=5000, seed=20261026):
    """Monte Carlo: activation jitter (sd 1.2 months) and gamma site rates."""
    rng = random.Random(seed)
    shape = 4.0
    results = []
    for _ in range(n_iter):
        sites = []
        for start, n in waves:
            for _ in range(n):
                sites.append((max(1.0, rng.gauss(start, 1.2)),
                              rng.gammavariate(shape, rate_mean / shape)))
        cumulative = 0.0
        done = None
        for month in range(1, 73):
            increment = 0.0
            for activation, rate in sites:
                if month >= activation:
                    k = int(month - activation)
                    increment += rate * (RAMP[k] if k < len(RAMP) else 1.0)
            previous = cumulative
            cumulative += increment
            if cumulative >= target:
                done = month - 1 + (target - previous) / increment
                break
        results.append(done if done is not None else 99.0)
    results.sort()
    q = statistics.quantiles(results, n=10)
    return {
        "median": statistics.median(results),
        "p10": q[0],
        "p90": q[8],
        "p_le_30": sum(1 for x in results if x <= 30) / n_iter,
        "p_le_36": sum(1 for x in results if x <= 36) / n_iter,
    }

TARGET = 600 / 0.90          # 667 randomised
SCREEN = TARGET / 0.72       # 927 screened

PLAN = [(1, 45)]
WAVES = [(1, 15), (4, 15), (7, 15)]
FAST = [(1, 15), (2, 15), (3, 15)]

print(f"randomised target {TARGET:.1f}  screened {SCREEN:.1f}")
print(f"plan (all month 1, 0.55):      {completion_month(PLAN, 0.55, TARGET):.2f}")
print(f"waves, 0.55:                   {completion_month(WAVES, 0.55, TARGET):.2f}")
print(f"waves, 0.50:                   {completion_month(WAVES, 0.50, TARGET):.2f}")
print(f"compressed waves, 0.50:        {completion_month(FAST, 0.50, TARGET):.2f}")

base = simulate(WAVES, 0.50, TARGET)
print("Monte Carlo, waves: median {median:.2f}  P10 {p10:.2f}  P90 {p90:.2f} "
      "P(<=30) {p_le_30:.3f}  P(<=36) {p_le_36:.3f}".format(**base))
fast = simulate(FAST, 0.50, TARGET)
print("Monte Carlo, fast:  median {median:.2f}  P10 {p10:.2f}  P90 {p90:.2f} "
      "P(<=30) {p_le_30:.3f}  P(<=36) {p_le_36:.3f}".format(**fast))

base_case = completion_month(WAVES, 0.50, TARGET)
plan_case = completion_month(PLAN, 0.55, TARGET)
fixed = 600_000
print(f"direct delay cost USD 2026: {(base_case - plan_case) * fixed:,.0f}")
print(f"milestone carry at 8%:      {15_000_000 * 0.08 * (base_case - plan_case) / 12:,.0f}")
```

**Expected output:**

```text
randomised target 666.7  screened 925.9
plan (all month 1, 0.55):      27.64
waves, 0.55:                   30.64
waves, 0.50:                   33.33
compressed waves, 0.50:        31.33
Monte Carlo, waves: median 33.96  P10 31.28  P90 36.98 P(<=30) 0.028  P(<=36) 0.812
Monte Carlo, fast:  median 31.97  P10 29.30  P90 34.99 P(<=30) 0.178  P(<=36) 0.951
direct delay cost USD 2026: 3,416,162
milestone carry at 8%:      569,360
```

**What the output says:**

1. The deterministic model reproduces the worked example to within a hundredth of a month. The
   small difference from the hand calculation comes from interpolating within the month in which
   the target is crossed.
2. The Monte Carlo median, 33.96, is close to the deterministic 33.33, but the 90th percentile is
   36.98. The tail is 3.6 months longer than the median, which is the range the programme should
   plan cash for.
3. The probability of completing within 30 months under the current plan is 2.8%. The probability
   within 36 months is 81.2%. A commitment to 30 months would be a commitment the plan cannot
   support.
4. Compressing activation raises the probability of completing within 30 months from 2.8% to 17.8%
   and the median falls by two months. It does not make 30 months likely, which is the honest
   conclusion.

**What to change first:** the seed, to check percentile stability; the rate distribution shape, to
test whether heterogeneity drives the tail; and the activation standard deviation, which has little
effect here compared with site-to-site rate variation. The model omits competing trials,
seasonality, country differences, and the correlation between late activation and slow accrual.
That last omission matters and biases the tail optimistically.

## 8. Data, standards, and regulatory context

Trial operations sits under a small number of binding instruments and a larger number of
conventions. The distinction matters: a binding requirement can be inspected and sanctioned, a
convention can be negotiated, and a sponsor's SOP is neither.

### 8.1 Binding requirements

| Instrument | What it binds | Operational consequence |
|---|---|---|
| ICH E6(R3) Good Clinical Practice | Trial conduct, sponsor and investigator responsibilities, data governance | Quality-by-design, risk-proportionate monitoring, traceable records |
| ICH E8(R1) General Considerations for Clinical Studies | Study design and quality | Critical-to-quality factors must be identified before the protocol is final |
| ICH E9(R1) Addendum on Estimands | Definition of the treatment effect | Intercurrent events determine what data must be collected and how |
| 21 CFR Part 312 (US) | Investigational new drug applications | IND, safety reporting, record retention |
| 21 CFR Part 812 (US) | Investigational device exemptions | Device accountability, investigator agreements |
| 42 CFR Part 11 (US) | ClinicalTrials.gov registration and results | Registration within 21 days of first enrolment; results within 1 year of primary completion |
| Regulation (EU) 536/2014 (Clinical Trials Regulation) | Trials conducted in the EU | Single application through CTIS; national assessment; transparency obligations |
| ISO 14155:2020 | Clinical investigation of medical devices | Device-specific documentation, risk management, monitoring |
| 21 CFR Part 11 (US) | Electronic records and signatures | Audit trails, access control, validation of EDC and eConsent |

ICH E6(R3) was adopted at Step 4 on 6 January 2025. The European Medicines Agency (EMA) published
the Step 5 version on 27 January 2025, and the Principles and Annex 1 took effect in the EU on
23 July 2025. Annex 2, which covers non-traditional trial designs including decentralised and
pragmatic elements, has a later EU effective date of 15 January 2027; a consolidated Step 5 version
was published on 15 July 2026. A sponsor operating across jurisdictions should expect the same
text to have different effective dates and should not assume that an inspection in one region
applies the version in force in another.

E6(R3) is principles-based. It does not prescribe a monitoring percentage, a query timeline, or a
specific technology. It requires that the sponsor identify what matters for participant safety and
reliable results, and that oversight be proportionate to risk. That is a requirement about
reasoning, not about a checklist, and it is the reason risk-based monitoring appears in Section 5
as a control system rather than a sampling rate.

### 8.2 Registration and results

In the United States, 42 CFR Part 11 implements FDAAA section 801. Registration is required no later
than 21 calendar days after the first human participant is enrolled, and results are due no later
than one year after the primary completion date. A certification can delay results submission in
defined circumstances, with a cap of two years after the certification. Non-compliance is a
prohibited act, with civil monetary penalties and consequences for federal funding.

In the European Union, the Clinical Trials Regulation requires a summary of results and a plain
language summary within a defined period after the end of the trial, published through CTIS. The
transition period from the Clinical Trials Directive ended on 30 January 2025, with more than 5,000
trials transitioned; trials still running after that date that were not transitioned require a new
application. The practical consequence for operations is that the trial's public record is part of
the close-out deliverable, not an afterthought for the communications team.

### 8.3 Records and retention

US regulation requires records to be retained for two years after a marketing application is
approved or two years after the investigation is discontinued (21 CFR 312.62(c)). Other
jurisdictions set different periods, and the applicable period is the longest one that applies to
the site or the sponsor. The operational implication is that close-out does not end the record
obligation: the trial master file must remain retrievable for the retention period, and sites must
be told in writing what they are holding and for how long.

### 8.4 Electronic records and data integrity

Where electronic systems are used to create, modify, maintain, or transmit records, 21 CFR Part 11
requires validated systems, secure access, audit trails, and controls over electronic signatures.
The EU has an equivalent expectation through E6(R3) and the CTR's data integrity requirements. In
practice the audit trail is the artefact that inspections test: it must show who changed what, when,
and why, and it must not be possible for a user to disable it.

Data integrity expectations are often stated as ALCOA+ principles: attributable, legible,
contemporaneous, original, accurate, complete, consistent, enduring, and available. ALCOA+ is a
convention, but it is the vocabulary inspectors use.

### 8.5 Cross-border data

Moving patient data across borders invokes separate legal regimes from the trial regulation: the
General Data Protection Regulation (GDPR) in the EU, the Personal Information Protection Law (PIPL)
in China, and sector-specific rules elsewhere. China additionally regulates the export of human
genetic resources, which affects sample shipment as well as data. These are binding requirements
with their own approval processes, and they can delay a trial independently of the regulator that
approves the protocol.

### 8.6 What the standards do not constrain

ICH E6(R3) does not set an enrolment target, a site count, a monitoring frequency, a query
turnaround time, or a budget. Those are sponsor decisions. The standards also do not determine
whether the evidence will satisfy a health technology assessment body, which applies a different
question and a different evidentiary standard. A trial can be fully GCP-compliant and still fail to
support reimbursement.

> **Note:** This section describes requirements as published at the access date. It is educational
> material, not legal advice. A sponsor must confirm the current text and the applicable national
> implementation before relying on any statement here.

## 9. Industry practice and case evidence

### 9.1 What the start-up data show

The most detailed public benchmark of North American site start-up comes from a study of nine
cardiovascular trials and approximately 2,200 sites (Goyal et al., 2021). The medians were:

| Interval | Median | Interquartile range |
|---|---:|---:|
| Protocol delivery to first patient enrolled | 255 days | 177 to 350 days |
| Site selection to activation | 171 days | 114 to 246 days |
| Regulatory approval | 132 days | Not reported in the summary |
| Contract execution | 143 days | Not reported in the summary |
| Activation to first enrolment | 66 days | Not reported in the summary |

Two findings from the same study are operationally important. Sites using a central IRB activated in
199 days versus 287 days for local IRB review, a difference of nearly three months. The top decile
of sites completed start-up in 106 days, which shows that the median is not a physical limit.

A survey of more than 400 sponsors, biotechs, and CROs found the full site identification-to-
activation process averaging five to six months, with CROs completing site activities six to eleven
weeks faster than sponsors (Lamberti et al., 2018). A 2025 analysis of 308 site activations in the
National Center for Advancing Translational Sciences Trial Innovation Network reported a median of
133 days with a full accelerated start-up programme, 191 days without site navigators, and 277 days
without a lean workflow (Hillery et al., 2025). The same paper notes a National Cancer Institute
target of 90-day activation.

These are observational benchmarks from particular settings, not guarantees. They are useful because
they establish the order of magnitude and the spread. A plan that assumes 30-day activation is not
ambitious; it is inconsistent with the published record unless the sites are already under master
agreements and the product is already available.

### 9.2 What the evidence implies for planning

Three implications are well supported: model activation as a distribution; run contract and
regulatory work in parallel; and pre-screen before activation, because the activation-to-FPI
interval is about two months. The claim that central review is faster is moderate and
setting-dependent. The claim that CROs activate faster than sponsors is low-confidence survey data.

### 9.3 The RECOVERY counter-example

The RECOVERY trial of COVID-19 treatments in the United Kingdom is a documented case of operational
speed. It used a simplified protocol, a small data set collected at each visit, existing National
Health Service infrastructure, and a randomisation system that sites could use without new
technology. It enrolled more than 11,000 participants across 176 hospitals within roughly three
months of opening.

The inference often drawn is that the trial was fast because it was simple. That is partly right and
partly misleading: it also benefited from a public health emergency, a large eligible population,
and a single-payer system with existing research networks. A commercial trial cannot copy the
context, but it can copy the design choices: the smallest data set that answers the question, and
randomisation that does not require a new site system.

> **Inference, not documented fact:** the claim that simplification caused most of the speed
> advantage is an inference. The published record supports the timeline and the design; it does not
> isolate the causal contribution of each design choice.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Plan uses steady-state accrual from month 1 | Month-3 actual is 30% of plan | Activation curve ignored | Completion forecast short by months | Model activation explicitly |
| Denominators confused | Site workload per patient does not reconcile | Screens, randomised, evaluable mixed | Sites under-resourced; screen failure surprises | Fix all three denominators before budgeting |
| Site selected on relationship | Site has no prior trial in the indication | No catchment or capability evidence | Zero enrolment at a costly site | Require historical accrual evidence |
| Contract terms reopened late | Legal review starts after site selection | Indemnity, publication, IP disputes | Activation delayed by months | Use master agreements; start legal early |
| IP not available at activation | First shipment date after planned activation | Supply forecast against the plan, not the upper bound | Site idle; patients lost | Forecast supply at P90 enrolment |
| Ethics and regulatory sequenced | Submission dates are not parallel | Misread dependencies | Start-up lengthened | Build a dependency map per country |
| Screen failure above assumption | Screens high, randomisations low | Criteria interpreted strictly or population different | Enrolment shortfall; possible amendment | Review eligibility interpretation; monitor ratio |
| Enrolment concentrated in few sites | Top three sites above 50% of total | Referral networks differ | Site closure becomes a crisis | Protect key sites; diversify |
| Amendment without impact assessment | Enrolment pauses with no resumption date | Regulatory and training dependencies missed | Months lost; mixed population | Require an impact assessment before approval |
| Close-out starting at LPLV | Query backlog rising through the last quarter | Close-out treated as an afterthought | Lock delayed | Plan the close-out queue from LPI |
| Essential documents incomplete | Delegation logs unsigned at monitoring | Documentation deferred to close-out | Inspection finding; lock delay | Verify at each monitoring visit |
| Cross-border transfer unplanned | Vendor cannot confirm hosting region | Legal mechanism not in place | Data cannot leave the site | Settle transfer at site selection |
| Registry results late | No owner for the submission | Close-out ends at CSR | Non-compliance; public record incomplete | Assign registry owner at study start |
| Over-enrolment beyond plan | Cumulative above the upper curve | No enrolment cap in the protocol | Statistical and supply consequences | Pre-specify an enrolment cap and stopping rule |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The per-site accrual rate is representative of the selected sites | Converts site count into monthly enrolment | Sites differ in catchment and motivation | Compare with prior trials at the same sites |
| Screen-failure rate is constant over time | Converts randomised to screened target | Criteria reinterpreted after training; population shifts | Monitor monthly screen-to-randomise ratio |
| Dropout is independent of treatment arm | Converts evaluable to randomised target | Differential toxicity or efficacy | Review blinded discontinuation rates |
| Activation months are known | Builds the monthly curve | Contract, ethics, or supply delays | Track activation against plan weekly |
| Fixed cost is constant per month | Prices delay | Site payments step down after enrolment; team size changes | Use a cost model with phases |
| No competing trial affects the same patients | Keeps the eligible pool stable | A competitor opens in the same catchment | Monitor registry and site intelligence |
| The protocol will not change materially | Keeps the curve and budget valid | Amendment mid-study | Pre-specify amendment scenarios |
| Currency and inflation are stable | Keeps the budget comparable | Exchange-rate movement; local inflation | State currency and year; index if multi-year |
| Regulatory clocks are as published | Sets the start date | Review clock restarts; questions extend the timetable | Build a per-country dependency plan |
| Sites report screening accurately | Makes the funnel measurable | Screening logs incomplete | Audit screening logs at monitoring |

### 11.1 Limits of the model

The model in Section 7 is a planning tool, not a prediction of any specific trial. It omits the
correlation between activation delay and accrual performance, which biases the tail optimistically.
It treats sites as independent, which ignores that sites in the same region compete for the same
patients. It assumes the protocol is fixed, which is false for any trial that amends.

The cost model is a fixed monthly rate. Real trial budgets have step changes: site payments fall
after the enrolment period, monitoring frequency changes, and the internal team is reassigned. A
delay that crosses one of those step changes costs less than the linear model implies.

Finally, the model says nothing about data quality. A trial that enrols on time but generates
unusable data has not succeeded. The next two topics deal with that part of the system.

## 12. Exercises

### Exercise 1. Reconcile the denominators (application, 30 minutes)

A protocol specifies 480 evaluable patients, 8% dropout, and a 30% screen-failure rate. The
operations plan states "480 patients across 32 sites, 15 per site over 24 months".

**Deliverable:** a one-page reconciliation showing the randomised target, the screened target, the
screen-to-randomise ratio, and the screens per site. State whether the plan's per-site workload is
adequate.
**Acceptance criteria:** all three denominators named; arithmetic shown to one decimal place; a
clear statement of which number the site budget should be built on.
**Time:** 30 minutes.

### Exercise 2. Build the activation and enrolment curve (application, 45 minutes)

Use 32 sites in three waves of 12, 10, and 10 at months 1, 5, and 9. Steady-state accrual is 0.45
patients per site per month, with the same 50/80/100% maturity ramp. The randomised target is the
one you computed in Exercise 1.

**Deliverable:** a month-by-month table for months 1 to 36 with active sites, monthly
randomisations, and cumulative randomisations, plus the completion month.
**Acceptance criteria:** the table is internally consistent (each cumulative equals the previous
plus the increment); the completion month is derived from the table, not asserted; the calculation
is reproducible from the table alone.
**Time:** 45 minutes.

### Exercise 3. Price the delay (calculation, 45 minutes)

Using the completion month from Exercise 2 and a correctly specified plan that activates all 32
sites in month 1 at 0.45 patients per site per month, compute the delay in months. Fixed operating
cost is $410,000 per month (2026 US dollars). A milestone of $8,000,000 is deferred, and the cost of
capital is 7%.

**Deliverable:** direct delay cost, milestone carry, and total exposure, each with its formula.
**Acceptance criteria:** currency and year stated; formula shown; the difference between direct cost
and financing cost explained in one sentence.
**Time:** 45 minutes.

### Exercise 4. Diagnose a failing plan (critique, 60 minutes)

A trial reports month-9 cumulative enrolment at 54% of plan. The operations lead proposes adding 20
sites. You have the activation log, which shows 12 of 40 planned sites were activated more than 60
days late, and the screen-failure rate, which is 34% against a planned 25%.

**Deliverable:** a two-page diagnosis that separates activation variance, accrual variance, and
screen-failure variance, and recommends one action with a justification.
**Acceptance criteria:** each variance is quantified or explicitly marked as not quantifiable from
the available data; the recommendation names the binding constraint; the memo states what evidence
would change the conclusion.
**Time:** 60 minutes.

### Exercise 5. Design a contingency decision (design, 60 minutes)

You are the programme lead for a trial that is 12 months into a 30-month enrolment window and is 25%
behind. The options are: add 15 sites, extend the window by 6 months, broaden one eligibility
criterion, or add a country.

**Deliverable:** a decision memo with a comparison table scored on probability of meeting the
readout date, direct cost, evidence risk, and reversibility, ending in a recommendation.
**Acceptance criteria:** at least three options compared; each score has a stated basis; the evidence
risk of broadening eligibility is described in clinical terms, not only financial ones; the
recommendation states the trigger that would cause it to be revisited.
**Time:** 60 minutes.

### Exercise 6. Assess an amendment (design, 75 minutes)

Mid-enrolment, the sponsor proposes to change the primary endpoint from progression-free survival
assessed by the investigator to blinded independent central review, and to add a stratification
factor.

**Deliverable:** an impact assessment covering regulatory, ethics, re-consent, EDC and edit checks,
statistical analysis, budget, and timeline, with a recommendation on timing.
**Acceptance criteria:** each of the seven areas addressed; the effect on already-enrolled patients
stated; the analysis consequence (mixed assessment methods) described; a realistic pause estimate
given with its basis.
**Time:** 75 minutes.

### Exercise 7. Plan the close-out queue (application, 60 minutes)

A trial has 45 sites and reaches LPLV on 30 June. Each close-out visit takes one monitor two days,
and the team has four monitors available at 50% time. Query resolution at sites averages three weeks
after the close-out visit. Database lock requires all sites closed and all queries resolved.

**Deliverable:** a week-by-week close-out schedule and the earliest credible lock date, with
assumptions stated.
**Acceptance criteria:** capacity calculation shown; queue modelled rather than averaged; the lock
date accounts for query resolution after the last visit; risks to the date identified.
**Time:** 60 minutes.

### Exercise 8. Map the downstream obligations (critique, 45 minutes)

A trial completes its primary endpoint in March 2027. The sponsor plans to submit a marketing
application in the United States and the European Union in 2028.

**Deliverable:** a one-page obligation map covering registration updates, results submission in both
regions, record retention, and the owner of each obligation.
**Acceptance criteria:** each obligation has a trigger date, a legal basis, and an owner; the
difference between the US one-year rule and the EU summary requirement is stated; no obligation is
attributed to a regulator that does not hold it.
**Time:** 45 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

```text
Randomised target = 480 / (1 - 0.08) = 521.7 -> 522
Screened target   = 522 / (1 - 0.30) = 745.7 -> 746
Screen-to-randomise ratio = 746 / 522 = 1.43
Screens per site = 746 / 32 = 23.3
```

The plan's 15 patients per site is a randomised figure. The site is actually being asked to screen
23.3 patients to deliver 16.3 randomised patients per site (522/32). The plan understates screening
workload by 43%, and the site budget should be built on 23.3 screens per site.

### 13.2 Exercise 2

With 12 sites at month 1, 10 at month 5, and 10 at month 9, at 0.45 per site per month:

```text
Month 1: 12 x 0.45 x 0.5 = 2.70    cumulative  2.70
Month 2: 12 x 0.45 x 0.8 = 4.32    cumulative  7.02
Month 3: 12 x 0.45       = 5.40    cumulative 12.42
Month 4: 5.40                      cumulative 17.82
Month 5: 5.40 + 10x0.45x0.5=2.25   cumulative 25.47
Month 6: 5.40 + 3.60      = 9.00   cumulative 34.47
Month 7: 5.40 + 4.50      = 9.90   cumulative 44.37
Month 8: 9.90                      cumulative 54.27
Month 9: 9.90 + 2.25      = 12.15  cumulative 66.42
Month 10: 9.90 + 3.60     = 13.50  cumulative 79.92
Month 11: 9.90 + 4.50     = 14.40  cumulative 94.32
Month 12: 14.40                    cumulative 108.72
```

From month 12 the curve adds 14.40 per month. To reach 522:

```text
Remaining = 522 - 108.72 = 413.28
Months = 413.28 / 14.40 = 28.70
Completion = 12 + 28.70 = 40.70
```

The trial completes in month 40.7, not month 24. The plan is 16.7 months short, which is why the
denominator check in Exercise 1 matters: the plan was built on 480 patients rather than 522
randomised.

### 13.3 Exercise 3

All 32 sites active in month 1 at 0.45: month 1 contributes 7.20, month 2 contributes 11.52, and
from month 3 the rate is 14.40 per month.

```text
Cumulative at month 3 = 7.20 + 11.52 + 14.40 = 33.12
Remaining = 522 - 33.12 = 488.88
Months = 488.88 / 14.40 = 33.95
Plan completion = 3 + 33.95 = 36.95
Delay = 40.70 - 36.95 = 3.75 months
Direct cost = 3.75 x $410,000 = $1,537,500
Milestone carry = $8,000,000 x 0.07 x 3.75 / 12 = $175,000
Total exposure = $1,712,500
```

The direct cost is the operating expense of running the trial for 3.75 more months. The milestone
carry is a financing cost: the same cash arrives later, so its present value falls. They are
different in kind and should be reported separately.

### 13.4 Exercise 4

A strong answer separates three quantities:

- **Screen-failure variance.** With 34% actual against 25% planned, the screening target rises by
  (1-0.25)/(1-0.34) = 1.136, or 13.6%. If screens are on plan, randomisations are 11.9% below plan.
- **Activation variance.** Twelve sites activated more than 60 days late. With a 0.45 per site per
  month rate, 12 sites x 2 months x 0.45 = 10.8 lost randomisations, or about 2.4% of a 450-patient
  target.
- **Accrual variance.** Whatever remains after the first two are subtracted.

If the total gap is 46%, and screen failure explains about 12 points and activation about 2 points,
the residual is roughly 32 points, which points to per-site accrual. Adding 20 sites addresses the
smallest component and will not close the gap inside the window. The binding constraint is site
productivity, and the first action should be a site-level review of screening and referral.

### 13.5 Exercise 5

There is no single correct answer. A strong memo:

- states the readout date and the cost of missing it;
- notes that adding sites has a five-to-eight-month activation lag and helps only if more than 12
  months remain;
- prices the window extension as certain but expensive;
- treats broadening eligibility as the only option that changes the population and the label, and
  therefore carries an evidence risk that outlives the trial;
- recommends the reversible option and names the trigger for revisiting it.

### 13.6 Exercise 6

A strong assessment notes:

- **Regulatory.** A change to the primary endpoint and a new stratification factor is a substantial
  amendment in the EU and a protocol amendment in the US; both require review before implementation
  in the affected jurisdictions.
- **Ethics and consent.** Participants must be informed of the change if it affects their
  participation or the use of their data; re-consent may be required.
- **EDC and edit checks.** A new assessment source (central review) adds an external data import,
  reconciliation rules, and a new query pathway.
- **Statistical.** Patients already assessed by the investigator have no central review. The
  analysis must either treat the assessment method as a stratification variable or restrict the
  primary analysis to centrally reviewed patients, which changes the sample size.
- **Budget and timeline.** The pause, retraining, and system changes typically cost weeks to months.

### 13.7 Exercise 7

```text
Monitor capacity = 4 monitors x 0.5 FTE x 5 days = 10 monitor-days per week
Visits per week = 10 / 2 = 5 sites per week
Queue = 45 / 5 = 9 weeks
Last close-out visit = week 9 after 30 June
Query resolution = 3 weeks after the last visit
Earliest lock = 30 June + 9 weeks + 3 weeks = 30 June + 12 weeks = 22 September
```

The credible lock date is late September, assuming no site reschedules and no query reopens. The
last site determines the date, and the last site is usually the one with the largest query backlog.

### 13.8 Exercise 8

A strong map includes:

| Obligation | Trigger | Basis | Owner |
|---|---|---|---|
| Registry registration maintained | First enrolment | 42 CFR Part 11 | Regulatory operations |
| US results submission | 1 year after primary completion (March 2028) | 42 CFR 11.44 | Regulatory operations |
| EU summary of results | Within the CTR period after trial end | Regulation (EU) 536/2014 | Regulatory operations |
| CSR | Before submission | ICH E3 | Medical writing |
| Record retention | 2 years after US approval or discontinuation | 21 CFR 312.62(c) | TMF owner |
| Site records | Per applicable national period | National law | Site |

A weak map assigns the EU summary to the FDA or assumes that a 2028 submission resets the results
clock; it does not.

## 14. Interview questions

**1. How do you tell whether an enrolment plan is achievable?**
A strong answer reconstructs the denominators, checks whether the plan has an activation curve, and
compares the implied per-site rate with historical performance at the same sites. A weak answer
quotes the plan's own number back.

**2. What is the difference between activation and first patient in?**
Strong: activation is a verified permission to enrol; first patient in is an event that follows by
weeks, and the gap is a distinct operational problem. Weak: treats them as the same date.

**4. How would you price a six-month delay?**
Strong: separates fixed operating cost per month, variable cost that does not scale with delay, and
financing or milestone effects, and states the currency and year. Weak: multiplies total trial cost
by the delay fraction.

**5. What happens operationally when a protocol is amended mid-study?**
Strong: describes the impact assessment, the regulatory category, re-consent, retraining, EDC
changes, the enrollment pause, and the mixed-population consequence. Weak: describes it as a
document update.

**6. How do you decide whether a site should be closed for underperformance?**
Strong: compares the site's expected remaining contribution with the cost and risk of keeping it
open, considers the patients already enrolled, and checks contractual and ethics obligations.
Weak: closes on a threshold without considering enrolled patients.

**8. Which obligations survive trial close-out?**
Strong: registry results, CSR, record retention, site notification, and any post-approval
commitments, each with its own clock. Weak: says the CSR is the end.

**9. How would you plan a trial in a country with a long regulatory clock?**
Strong: starts the submission early, runs ethics and contracting in parallel, and plans a later
activation wave rather than assuming a global month-1 start. Weak: adds a buffer to the whole
timeline without changing the sequence.

**10. When would you recommend stopping a trial for operational reasons?**
Strong: when the probability of completing within the window is low under every available lever, or
when the cost of the remaining options exceeds the value of the readout. Weak: when enrolment is
behind plan, without quantifying whether any lever can recover it.

## 15. Key terms

| Term | Meaning | See |
|---|---|---|
| Accrual | Rate of randomisation per site per month | 3.4 |
| Activation | Verified state in which a site may enrol | 3.3 |
| CAPA | Corrective and preventive action | 5.11 |
| Close-out visit | Final monitoring visit reconciling records and IP | 5.13 |
| CSR | Clinical study report | 5.14 |
| CTIS | Clinical Trials Information System for the EU | 8.2 |
| DBL | Database lock | 5.14 |
| DCT | Decentralised trial | 8.1 |
| Deviation | Departure from the protocol | 4.3 |
| Dropout | Loss before the primary endpoint | 6.2 |
| EDC | Electronic data capture | 4 |
| Eligibility rule table | Operational translation of eligibility criteria | 5.1 |
| Essential documents | Records proving GCP conduct | 5.13 |
| FPI | First patient in | 3.2 |
| IP | Investigational product | 5.6 |
| LPI | Last patient in | 3.2 |
| LPLV | Last patient last visit | 3.2 |
| Operational specification | The site-facing interpretation of the protocol | 5.1 |
| Protocol amendment | Change to an approved protocol | 5.10 |
| RBQM | Risk-based quality management | 5.9 |
| Screen failure | Consented patient not randomised | 6.2 |
| Screen-to-randomise ratio | Screens per randomised patient | 6.2 |
| TMF | Trial master file | 5.13 |
| Trigger | Pre-agreed threshold that authorises an action | 5.9 |

## 16. Sources

All URLs accessed 26 October 2026.

1. ICH. *E6(R3) Good Clinical Practice, Step 4 Final Guideline*, 6 January 2025.
   https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
2. European Medicines Agency. *ICH E6 Good Clinical Practice*. 
   https://www.ema.europa.eu/en/scientific-guidelines/ich-e6-good-clinical-practice
3. Accelerating Clinical Trials in the EU. *Good clinical practice modernisation*.
   https://accelerating-clinical-trials.europa.eu/our-work/good-clinical-practice-modernisation_en
4. ICH. *Efficacy guidelines* (E8(R1), E9(R1), E3).
   https://www.ich.org/page/efficacy-guidelines
5. US Government. *42 CFR Part 11, Clinical Trials Registration and Results Information Submission*.
   https://www.ecfr.gov/current/title-42/chapter-I/subchapter-A/part-11
6. US Government. *21 CFR 312.62, Investigator recordkeeping and record retention*.
   https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-312/subpart-D/section-312.62
7. US Food and Drug Administration. *Oversight of Clinical Investigations: A Risk-Based Approach to
   Monitoring* (2013) and *A Risk-Based Approach to Monitoring: Questions and Answers* (2023).
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/oversight-clinical-investigations-risk-based-approach-monitoring
8. US Food and Drug Administration. *Conducting Clinical Trials With Decentralized Elements*, final
   guidance, September 2024.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/conducting-clinical-trials-decentralized-elements
9. European Medicines Agency. *Clinical Trials Regulation becomes fully applicable*, 31 January
    2025.
    https://www.ema.europa.eu/en/news/clinical-trials-regulation-becomes-fully-applicable
10. European Commission. *Clinical Trials Regulation (EU) No 536/2014*.
    https://health.ec.europa.eu/medicinal-products/clinical-trials/clinical-trials-regulation_en
11. Goyal N, et al. Assessment of North American Clinical Research Site Performance During the
    Start-up of Large Cardiovascular Clinical Trials. *JAMA Network Open*. 2021;4(7):e2117963.
    https://doi.org/10.1001/jamanetworkopen.2021.17963
12. Lamberti MJ, et al. Assessing Study Start-up Practices, Performance, and Perceptions Among
    Sponsors and Contract Research Organizations. *Therapeutic Innovation and Regulatory Science*.
    2018. https://pubmed.ncbi.nlm.nih.gov/29714558/
13. Hillery C, et al. Accelerating start-up cycles in investigator-initiated multicenter clinical
    trials. *Journal of Clinical and Translational Science*. 2025.
    https://www.cambridge.org/core/journals/journal-of-clinical-and-translational-science/article/accelerating-startup-cycles-in-investigatorinitiated-multicenter-clinical-trials/63330032F606C0527E8FEC91DDC7AF63
14. International Organization for Standardization. *ISO 14155:2020, Clinical investigation of
    medical devices*. https://www.iso.org/standard/71690.html
15. ClinicalTrials.gov. https://clinicaltrials.gov/
16. European Union. *General Data Protection Regulation (EU) 2016/679*.
    https://eur-lex.europa.eu/eli/reg/2016/679/oj
17. RECOVERY Collaborative Group. Dexamethasone in Hospitalized Patients with Covid-19.
    *New England Journal of Medicine*. 2021;384:693-704. https://doi.org/10.1056/NEJMoa2021436

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_site-and-monitoring.md`](02_site-and-monitoring.md)
- Data management: [`03_clinical-data-management.md`](03_clinical-data-management.md)
- Global regulation: [`06_global-regulation.md`](06_global-regulation.md)
- Module survey: [`../../course/25_trial_operations_hta_global_regulatory.md`](../../course/25_trial_operations_hta_global_regulatory.md)
- Labs: [`../../labs/`](../../labs/)
- Related module: [`../13_biopharma/04_feasibility-and-recruitment.md`](../13_biopharma/04_feasibility-and-recruitment.md)
