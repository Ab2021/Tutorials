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

### 1.4 What this topic is not

This topic does not teach clinical judgement, therapeutic area knowledge, or the content of a
specific sponsor's standard operating procedures (SOPs). It teaches the operating logic that is
common across them, and the arithmetic that connects a protocol to a date and a budget. Regulatory
content is educational and is not legal advice.

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

1. **Fixed site costs.** Site payments often include a monthly or per-visit component that continues
   while the site is open.
2. **Fixed sponsor and vendor costs.** Internal full-time equivalents (FTEs), CRO management fees,
   and technology licences accrue monthly.
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
Direct delay cost = (33.34 - 27.65) x $600,000 = 5.69 x $600,000 = $3,414,000
```

The comparison is against a correctly specified plan that activates all sites in month 1 at 0.55
patients per site per month, which completes in month 27.65. Against the original 24.2-month
promise, the gap is 9.1 months and $5,460,000.

The milestone is deferred, not cancelled. At an 8% annual cost of capital, the carry on
$15,000,000 for the additional 5.69 months is:

```text
Carry = $15,000,000 x 0.08 x 5.69 / 12 = $569,000
```

Total quantified exposure is approximately $3,983,000, before any allowance for lost competitive
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
| Plan as promised | All 45 at month 1 | 0.55 | 27.65 | 0.00 |
| Activation only | Three waves | 0.55 | 30.65 | +3.00 |
| Activation and rate | Three waves | 0.50 | 33.34 | +5.69 |

The decomposition is sequential, so the order matters. Reversing the order attributes 2.35 months to
the rate and 3.34 months to activation. What is stable across both orderings is that activation
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
Completion: 5 + 26.34 = 31.34
```

Completion moves from 33.34 to 31.34, a saving of 2.00 months and $1,200,000 in fixed operating
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
