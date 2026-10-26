# Global Regulation: EMA, MHRA, PMDA, NMPA, Health Canada, and TGA

> Module 25 — Trial Operations, HTA, and Global Regulation
> Topic 06 of 07
> Estimated study time: 4 hours
> Prerequisites: topics 01 and 03
> Version 2026.3

## 1. Why this topic exists

A marketing application is not a global document. It is a core dossier plus a set of regional
decisions about what evidence to accept, what to test locally, what to inspect, and how long to
take. The core is increasingly harmonised through the International Council for Harmonisation
(ICH) Common Technical Document (CTD); the regional decisions are not.

The commercial consequence is sequencing. The first approval determines when revenue starts. The
last major approval determines when the global launch is complete. A programme that assumes all
markets can be filed on the same day, and approved on the same timetable, will misplan both the
budget and the launch.

The regulatory differences are also a source of risk that data scientists underestimate. A local
clinical data requirement can add years and change the population studied. A reliance pathway can
shorten a review but only if a reference approval exists first. A quality or manufacturing issue can
stop an approval in one country while others proceed. These are not legal technicalities; they are
schedule and evidence decisions.

### 1.1 The pattern that repeats

1. A sponsor prepares one global dossier and plans to file in seven markets in the same quarter.
2. The European submission requires a good manufacturing practice inspection of a new facility, which
   adds four months.
3. Japan requires local clinical data because the dose-response relationship differs, adding a
   bridging study.
4. China requires a local trial for the indication, adding two years.
5. The UK could have used the International Recognition Procedure after the EU approval, but the
   sponsor filed nationally first and waited longer.
6. The global launch completes four years after the first approval, not twelve months.

Each step is a jurisdictional requirement or a sequencing choice. The sponsor did not lack a dossier;
it lacked a map of what each regulator would accept and when.

### 1.2 Who is blocked

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Sponsor | Revenue timing and capital efficiency | Depends on seven agencies |
| Local affiliate | Launch readiness | Depends on a global dossier |
| Patients | Access in their country | Depend on a national approval |
| Regulator | Review predictability | Receives dossiers that ignore local requirements |
| Payer | Budget planning | Cannot plan without an approval date |

### 1.3 The industry question

> Which markets can be filed from the same core dossier, which need local evidence, in what order,
> and what is the critical path to global access?

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a harmonised ICH dossier from the regional modules and local requirements that sit
   around it.
2. **Compare** the approval pathways and statutory review clocks of the EMA, MHRA, PMDA, NMPA, Health
   Canada, TGA, and FDA.
3. **Decide** whether a market needs local clinical data, a bridging study, or can rely on foreign
   data.
4. **Choose** between a national review and a reliance or recognition pathway, and explain when
   reliance is slower.
5. **Build** a global submission sequence with dependencies and identify the critical path.
6. **Quantify** the cost of a delayed market entry and the value of an earlier approval.
7. **Anticipate** post-approval obligations, including variations, inspections, and pharmacovigilance.
8. **Assess** whether a regulatory strategy is consistent with the evidence the trial will produce.

## 3. The core mental model

### 3.1 One core, many peripheries

```text
                    ICH Common Technical Document
                 +--------------------------------+
                 | Quality (Module 3)             |
                 | Nonclinical (Module 4)         |
                 | Clinical (Module 5)            |
                 +--------------------------------+
                                |
        +-----------------------+-----------------------+
        v                       v                       v
  Regional Module 1       Local data              Local requirements
  (forms, labels)         (bridging, trial)       (GMP, inspection, PV)
        |                       |                       |
        v                       v                       v
   EMA / MHRA /           PMDA / NMPA /            All agencies
   Health Canada /        some others
   TGA / FDA
```

The core is shared. The periphery is not. A global strategy is a plan for which parts of the
periphery can be satisfied from the core and which cannot.

### 3.2 Four axes of difference

| Axis | Question | Example |
|---|---|---|
| Evidence | Will the agency accept foreign clinical data? | Japan may require bridging; China may require a local trial |
| Quality | Will the agency inspect the manufacturing site? | EU and US inspections are routine for new facilities |
| Procedure | Is there a centralised, national, or reliance route? | EMA centralised; UK IRP; Canada national |
| Clock | Is the review clock in days, working days, or months, and does it stop? | TGA working days; Health Canada calendar days with clock stops |

A comparison that uses only the clock is misleading. A 200-day clock with a two-year local trial
requirement is slower than a 300-day clock with no local requirement.

### 3.3 Reliance is a dependency, not a shortcut

Reliance means one regulator uses another's assessment. Recognition means it accepts the decision.
Both require a reference approval, which means the relying market cannot finish before the reference
market. A reliance pathway can therefore be slower than a national review if the reference approval
is late.

```text
Reference approval date + reliance review time vs national review time
```

If the reference approval takes 10 months and reliance takes 4, the relying market approves at month
14. A national review started at month 0 with a 12-month clock approves at month 12. Reliance is
slower in this case, even though it requires less work.

### 3.4 The approval sequence

```text
 Dossier ready
    |
    +--> US (10 months) -----> UK IRP (60-110 days)
    |
    +--> EU (210 days + clock stop) -----> UK IRP
    |
    +--> Japan (12 months, or bridging + 12 months)
    |
    +--> Canada (300 days) -----> ...
    |
    +--> Australia (255 working days)
    |
    +--> China (200 days, or local trial + 200 days)
```

The sequence is a network, not a list. The critical path is the longest dependency chain, and it is
often a local trial requirement rather than a review clock.

### 3.5 Statutory clock and real time

Every published review time is a target or a statutory clock. Real time includes validation,
clock stops for questions, inspections, and administrative approval. The gap between the clock and
the approval is usually months, and it is larger in agencies with limited capacity or with a
requirement for local inspection.

A planning model should use the clock as a lower bound and add a country-specific allowance for
stops, derived from the sponsor's own history or published performance data.

### 3.6 Labelling is negotiated, not copied

The label approved in one market is not the label in another. Agencies may require different
indications, different warnings, different dosing, or different comparative claims. A global launch
plan that assumes one label will need a local labelling workstream and a medical review of every
claim.

### 3.7 Post-approval is part of the strategy

Approval creates obligations: variations, periodic safety reports, risk management plans,
inspections, and in some markets local manufacturing or local clinical data commitments. A strategy
that stops at approval under-resources the first three years of the product's life.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| National competent authority (NCA) | National regulator | Confused with the EMA | Organisation |
| EMA | European Medicines Agency | Assumed to approve products | Agency |
| CHMP | Committee for Medicinal Products for Human Use | Confused with the EMA | Committee |
| Centralised procedure | Single EU application through the EMA | Confused with mutual recognition | Procedure |
| Mutual recognition procedure (MRP) | Recognition of a national assessment by other member states | Confused with decentralised | Procedure |
| Decentralised procedure (DCP) | Simultaneous application in several member states | Confused with MRP | Procedure |
| MHRA | UK Medicines and Healthcare products Regulatory Agency | Assumed to be bound by EMA decisions | Agency |
| International Recognition Procedure (IRP) | UK route relying on a reference regulator | Confused with the old EC decision reliance | Procedure |
| PMDA | Japan Pharmaceuticals and Medical Devices Agency | Confused with the MHLW, which grants approval | Agency |
| MHLW | Japan Ministry of Health, Labour and Welfare | Confused with PMDA | Ministry |
| NMPA | China National Medical Products Administration | Confused with the former CFDA | Agency |
| Health Canada | Canadian regulator | Confused with CADTH, the HTA body | Agency |
| TGA | Australia Therapeutic Goods Administration | Confused with the PBAC, the HTA body | Agency |
| CTD | Common Technical Document | Assumed to be a single global application | Format |
| eCTD | Electronic CTD | Assumed to be identical across agencies | Format |
| Module 1 | Regional administrative information | Assumed to be harmonised | CTD module |
| Module 3 | Quality | Assumed to be identical | CTD module |
| Reliance | Using another regulator's assessment | Confused with recognition | Regulatory route |
| Recognition | Accepting another regulator's decision | Confused with reliance | Regulatory route |
| Abridged application | Application relying on existing data | Confused with a generic | Procedure |
| Bridging study | Small study to support extrapolation between regions | Confused with a full local trial | Study |
| Local clinical data | Data generated in the applying country | Assumed to be always required | Requirement |
| GMP | Good manufacturing practice | Confused with GCP | Standard |
| MAH | Marketing authorisation holder | Confused with the manufacturer | Role |
| Orphan designation | Status for rare-disease products | Confused with approval | Designation |
| PRIME | EMA priority medicines scheme | Confused with conditional approval | Scheme |
| Sakigake | Japan priority review scheme | Confused with PRIME | Scheme |
| Priority review | Shorter review clock | Confused with accelerated approval | Procedure |
| Conditional approval | Approval with obligations | Confused with full approval | Status |
| Variation | Change to an approved application | Confused with a new application | Procedure |
| PSUR | Periodic safety update report | Confused with a signal report | Report |
| RMP | Risk management plan | Confused with a pharmacovigilance plan | Document |
| UDI | Unique device identification | Applied to drugs incorrectly | Identifier |

### 4.1 Approval, authorisation, and registration

The three words describe the same event in different legal systems. The EU grants a marketing
authorisation, the US approves an application, China registers a product, and Japan grants approval
after a PMDA review and an MHLW decision. The differences matter when writing a global plan, because
the legal applicant and the holder of the authorisation differ by jurisdiction.

### 4.2 Reliance, recognition, and equivalence

Reliance means the relying authority uses the reference authority's assessment as an input but makes
its own decision. Recognition means it accepts the decision. Equivalence is a technical judgement
about whether two products or two sets of data are comparable. The three are often used loosely, and
the looseness produces plans that assume a reliance route guarantees a fast approval.

### 4.3 Clock stop

A clock stop is a period during which the review clock is paused, usually while the applicant
responds to questions. It is not a delay in the agency's work; it is time transferred to the
applicant. A plan that uses the statutory clock without adding the expected clock stop understates
the approval date by months.

## 5. The mechanics, step by step

This procedure builds a global regulatory strategy for a new medicine. It assumes a completed
pivotal programme and a target set of markets. Device-specific pathways differ in classification and
conformity assessment, but the sequencing logic is similar.

### Step 1. Define the global target product profile

**Input:** the clinical data, the competitive landscape, and the commercial priorities.

**Output:** a target product profile with the indication, population, dosing, comparative claims, and
the evidence supporting each.

**Rationale:** The profile determines which agencies can approve which claim. A claim that is
acceptable in one market may require a different trial in another. Defining the profile before the
strategy prevents a dossier from being built around a claim that only one agency will accept.

### Step 2. Classify the product in each jurisdiction

**Input:** the product type, mechanism, and intended use.

**Output:** a classification table with the pathway, the data requirements, and the expected review
route for each market.

**Rationale:** Classification determines the pathway. A new chemical entity, a biologic, a biosimilar,
a vaccine, and a device each take a different route. A product that is a medicine in one market may
be regulated as a device or a combination product in another.

### Step 3. Map the evidence requirements

**Input:** the target profile and the classification.

**Output:** an evidence matrix showing, for each market, whether the core clinical package is
sufficient, whether a bridging study is needed, and whether local data are required.

**Rationale:** The matrix is where most schedule risk sits. Japan may accept foreign data with a
bridging study under ICH E5; China may require a local trial for certain products; the EU may require
a specific comparator or endpoint. The requirement is not uniform, and it is not always published in
a form that is easy to find.

### Step 4. Choose the lead market and the sequence

**Input:** the evidence matrix, the commercial priorities, and the reliance relationships.

**Output:** a recommended sequence with the lead market, the reference approvals, and the dependent
markets.

**Rationale:** The lead market should be the one whose approval unlocks the most reliance pathways
and whose review is achievable with the available data. The US and the EU are common leads because
their approvals are accepted as references by several reliance routes. A lead market chosen for
commercial reasons alone may delay every dependent market.

**Branch point:** a product with a large US commercial opportunity and a smaller EU opportunity may
still choose the EU first if the EU approval unlocks more reliance pathways or if the US review is
likely to be delayed by a manufacturing issue.

### Step 5. Decide on local data

**Input:** the evidence matrix and the agency's guidance.

**Output:** a decision for each market: no local data, a bridging study, or a full local trial, with
the rationale and the timeline.

**Rationale:** A local trial is the longest single item in most global plans. It should be justified
by a documented requirement or a genuine uncertainty about extrapolation, not by habit. A bridging
study is cheaper and faster but needs to be designed to answer the agency's question, which usually
concerns dose, exposure, or ethnic sensitivity.

### Step 6. Plan quality and manufacturing

**Input:** the manufacturing sites, the process, and the stability data.

**Output:** a GMP inspection plan and a quality module that meets each agency's expectations.

**Rationale:** A new manufacturing site triggers inspections in the EU and the US, and the inspection
can be on the critical path. A site that is not ready when the dossier is filed will delay the
approval regardless of the clinical data.

### Step 7. Prepare the core dossier

**Input:** the clinical study reports, the quality data, and the nonclinical data.

**Output:** a CTD with Modules 2 to 5, written to the common technical document structure.

**Rationale:** The CTD is the harmonised part, and it is where a well-run programme saves time. A
module that is written once and reused is cheaper than seven separate documents. The regional
Module 1 is where the differences live.

### Step 8. Build the regional modules

**Input:** the core dossier and each agency's format requirements.

**Output:** Module 1 for each market, with the application form, the proposed label, the risk
management plan, and the local administrative documents.

**Rationale:** Module 1 is small in volume and large in delay. A missing local form, an untranslated
label, or a pharmacovigilance system that does not meet the local requirement can stop validation.

### Step 9. Engage the agencies early

**Input:** the draft programme and the questions that the data cannot answer.

**Output:** scientific advice, pre-submission meetings, or consultation minutes that shape the
programme.

**Rationale:** Advice is cheap relative to a repeated trial. The EMA, MHRA, PMDA, Health Canada, and
TGA all offer some form of scientific advice or pre-submission meeting. The advice is not binding,
but a programme that ignores it carries avoidable risk.

### Step 10. Submit and manage the review

**Input:** the validated application.

**Output:** a list of questions, responses, and clock-stop dates.

**Rationale:** The review is a project with a critical path. The applicant's response time is often
the largest controllable component. A team that treats a question as a request for information
rather than a deadline will lose weeks.

### Step 11. Negotiate the label

**Input:** the agency's proposed label and the clinical data.

**Output:** an agreed label with the indication, warnings, and claims.

**Rationale:** The label determines the marketable claim and the payer evidence. A label narrower
than expected changes the commercial case, and a label broader than the data support creates
post-approval risk.

### Step 12. Plan post-approval obligations

**Input:** the approval conditions and the local requirements.

**Output:** a commitments register with owners and dates.

**Rationale:** Conditional approvals, risk management plans, and post-authorisation studies are
binding commitments. A company that treats them as administrative follow-up will miss them.

### Step 13. Use reliance deliberately

**Input:** the reference approvals and the reliance routes.

**Output:** a reliance plan with the reference date, the route, and the expected review time.

**Rationale:** Reliance can reduce the work and the review time, but only if the reference approval
arrives early enough. The plan should compare the reliance date with the national route date and
choose the earlier one.

### Step 14. Maintain the approval

**Input:** the variations, the safety data, and the inspections.

**Output:** an updated dossier and a compliance record.

**Rationale:** An approval is a living authorisation. Changes to manufacturing, indication, or
label require variations, and the variation timeline differs by market. A global change control
process is the only way to keep seven approvals consistent.

### Step 15. Govern the global strategy

**Input:** the sequence, the commitments, and the performance data.

**Output:** a governance forum with decision rights, a risk register, and a quarterly review.

**Rationale:** Global regulatory strategy fails when each affiliate optimises its own market. A
governance forum that sees the whole sequence can make trade-offs, such as delaying one market to
protect the reference approval.

### 5.15 Decision table: market entry sequence

| Sequence | When it fits | Advantage | Risk |
|---|---|---|---|
| US first | Largest market; FDA approval unlocks reliance | Early revenue; reference for IRP | Manufacturing or clinical delay affects all |
| EU first | EU approval unlocks UK, Switzerland, and others | Reference for European reliance | Centralised procedure clock stops |
| Japan first | Large market with local data already generated | Early Asian access | Bridging study may delay global filing |
| Parallel filing | Dossier mature; markets independent | Earliest global access | Highest cost and coordination burden |

### 5.16 Decision table: reliance route

| Route | What it accepts | Typical time | When to use |
|---|---|---|---|
| Full national review | Full dossier | 210 days or more | No reference approval, or local requirements |
| Abridged review | Foreign approval plus local assessment | 150 to 210 days | Reference approval available; local data adequate |
| Recognition | Reference decision accepted | 60 days (MHRA Route A) | Same manufacturing, recent reference approval |
| Reliance with assessment | Reference assessment used, local decision | 110 days (MHRA Route B) | Reference older, or Route B triggers apply |

### 5.17 Decision table: local clinical data

| Situation | Decision | Evidence |
|---|---|---|
| Foreign data, similar epidemiology and standard of care | No local trial; justify extrapolation | ICH E5, published PK/PD |
| Different dose-response or ethnic sensitivity | Bridging study | PK and dose-ranging data |
| Agency requires local trial for the indication | Full local trial | Published requirement or advice |
| Conditional approval pathway available | Local confirmatory study post-approval | Agency agreement |

### 5.18 Edge cases

**Trial under-enrolment in a local study.** A local trial that under-recruits delays the market
approval and may not satisfy the agency's question. The decision to run it should include a
feasibility assessment, not only a regulatory requirement.

**Protocol amendment during a global programme.** A change to the core protocol must be reflected in
every submission and may require variations in markets already approved. The cost is multiplied by
the number of markets.

**Database lock delay.** A late lock delays every submission that depends on the same data. The
global sequence should be built from the lock date, not from the approval target.

**Cross-border data transfer.** Local clinical data may need to be transferred to a global analysis
centre, which invokes data protection and, in China, human genetic resource rules. The transfer
mechanism is part of the regulatory plan.

**Payer rejection and price negotiation failure.** Approval does not guarantee reimbursement. A market
can be approved, launched, and inaccessible, and the global revenue model must reflect the access
topic, not only the approval date.

### 5.19 Plan scientific advice

**Input:** the draft programme, the questions the data cannot answer, and each agency's advice
process.

**Output:** a schedule of advice meetings, the questions submitted, and the minutes that record the
agency's position.

**Rationale:** Scientific advice is the cheapest way to discover that a programme is unacceptable. The
EMA, MHRA, PMDA, Health Canada, and TGA all offer advice or pre-submission meetings, and the advice
usually addresses the adequacy of the clinical package, the proposed indication, and the design of
confirmatory studies.

Advice is not binding. An agency can change its position, and a different committee can reach a
different view. It is still the best available evidence of what the agency will accept, and a
programme that ignores it is choosing to learn the answer during review.

### 5.20 Plan the GMP inspection

**Input:** the manufacturing sites, the process, the validation status, and the inspection history.

**Output:** an inspection readiness plan with site owners, a mock inspection, and a remediation
timeline.

**Rationale:** A new facility or a major process change triggers an inspection in the EU and often in
the US. The inspection is on the critical path because the approval cannot issue until it is closed,
and a finding can require a re-inspection months later.

The plan should identify which sites are inspection-critical, which have recent inspections, and
which are in jurisdictions with mutual recognition arrangements. A site with a recent successful
inspection by a recognised authority is a lower risk than a new site with no history.

### 5.21 Manage variations and lifecycle

**Input:** the approved applications, the planned changes, and the safety data.

**Output:** a global change-control register with the variation category, the filing date, and the
approval dependency for each market.

**Rationale:** A change to the manufacturing process or the label must be filed in every market, and
the categories differ. A variation that is a notification in one country may be a prior-approval
application in another. A global change that is not coordinated produces seven different versions of
the product and a labelling inconsistency that is difficult to unwind.

### 5.22 Choose the submission format and validate it

**Input:** each agency's eCTD version, validation criteria, and file structure requirements.

**Output:** a format matrix and a validation test report for each market.

**Rationale:** A technically invalid submission is not reviewed. The eCTD structure, the file naming,
the study tagging, and the use of validated PDF are all specified, and the specifications differ by
agency and version. A submission that passes one agency's validator can fail another's.

**Branch point:** eCTD v4.0 is being introduced by several agencies on different timetables. A
programme filing in multiple markets may need to maintain both formats for a period, which is a
resource and quality-control cost rather than a scientific one.

### 5.23 Run a submission readiness review

**Input:** each market's module status, the validation criteria, and the inspection status.

**Output:** a readiness decision per market: file, file with conditions, or delay.

**Rationale:** Filing a dossier that is not ready consumes the agency's validation capacity and
restarts the clock. A readiness review is cheaper than a rejection.

| Check | Ready | Not ready |
|---|---|---|
| Module 1 forms | Complete and signed | Missing local forms or translations |
| Module 2 summaries | Written and quality-controlled | Draft or inconsistent with Module 5 |
| Module 3 quality | Complete; sites inspection-ready | Open stability or validation issues |
| Module 4 nonclinical | Complete | Outstanding study reports |
| Module 5 clinical | Complete; datasets locked | Interim data or open queries on the endpoint |
| eCTD validation | Passed the agency's validator | Errors in structure or file naming |
| GMP | Sites inspected or scheduled | Inspection not requested |

A market that fails more than one check should be delayed. A market that fails only a local
administrative item may file with a commitment, if the agency allows it.

### 5.24 Global regulatory metrics

| Metric | Definition | Use |
|---|---|---|
| Submission readiness | Proportion of modules complete at the planned date | Predicts validation risk |
| Validation time | Days from submission to validation | Agency-specific |
| Clock-stop days | Days the review clock is paused | Applicant performance |
| Approval versus plan | Months early or late | Strategy accuracy |
| Number of markets on the lead dossier | Markets filed from the core | Efficiency |
| Local data burden | Studies required by market | Cost and timeline |
| Post-approval commitments | Open obligations | Compliance risk |

## 6. Worked example

### 6.1 The programme

A new chemical entity for a chronic disease is ready for filing in month 0. The sponsor plans seven
markets: the United States, the European Union, Japan, Canada, Australia, the United Kingdom, and
China. The core dossier is complete and the same clinical package is proposed everywhere, with local
modules to follow.

The planning assumptions, in months from dossier readiness, are:

| Market | Review route | Clock | Planning months | Basis |
|---|---|---|---:|---|
| United States | NDA standard | 10 months | 10 | FDA standard review |
| European Union | Centralised | 210 active days | 10 | 7 months active plus a 3-month clock stop |
| Japan | PMDA standard | 12 months | 12 | No local data |
| Canada | NDS standard | 300 days | 10 | Calendar days, no major stop |
| Australia | TGA standard | 255 working days | 12 | Working days converted to months |
| United Kingdom | National | 210 days | 7 | National route from month 0 |
| United Kingdom | IRP Route B after US | 110 days | 4 after US | Reference approval required |
| China | Local trial plus review | 200 days after trial | 31 | 24-month local trial plus 7-month review |

The times are planning estimates, not commitments. They use the published clocks and add an
allowance for validation and questions.

### 6.2 The sequence

```text
Month 0   dossier ready
Month 7   UK national approval
Month 10  US, EU, Canada approval
Month 12  Japan, Australia approval
Month 14  UK via IRP after US approval
Month 31  China approval after local trial
```

The first approval is month 7. The last major market is month 31. The global launch is complete 31
months after the dossier was ready, not 10.

### 6.3 Critical path

The longest chain is the China local trial:

```text
Local trial 24 months + review 7 months = 31 months
```

The next longest is Japan and Australia at 12 months. The difference is 19 months, which is the
length of the tail that the local trial adds to the global programme.

A plan that reports the average review time across markets, or the time to the first approval, hides
this tail entirely.

### 6.4 Cost of the local data

The illustrative regulatory cost by market is:

| Market | Cost (2026 US dollars) | Share |
|---|---:|---:|
| United States | 3,000,000 | 5% |
| European Union | 4,000,000 | 7% |
| Japan | 3,000,000 | 5% |
| Canada | 2,000,000 | 3% |
| Australia | 2,000,000 | 3% |
| United Kingdom | 1,500,000 | 2% |
| China (including local trial) | 45,000,000 | 74% |
| Total | 60,500,000 | 100% |

The China local trial is three quarters of the regulatory cost and the entire critical path. A
budget review that looks only at the total sees a large number; a schedule review sees where the
time goes.

### 6.5 Diagnostic 1: the local trial is the schedule

If the China local trial were removed, the global completion would be month 12, set by Japan and
Australia. The local trial therefore accounts for 19 of the 31 months, or 61% of the global timeline.

The diagnostic is available before the trial starts: compare the local data requirement with the
review clocks. A requirement for a 24-month local trial dominates every review clock in the
portfolio, and no amount of dossier quality changes it.

### 6.6 Diagnostic 2: reliance can be slower than a national route

The UK can file nationally at month 0 and be approved at month 7, or wait for the US approval at
month 10 and use the IRP Route B, which takes about 4 months and finishes at month 14.

The reliance route requires less work and produces a later approval. If the objective is UK access,
the national route is seven months faster. If the objective is to avoid a duplicate assessment, the
reliance route is more efficient. The two objectives conflict, and the plan must say which one it is
optimising.

### 6.7 Vary one assumption: China accepts foreign data with a bridging study

Suppose the Chinese regulator accepts a six-month bridging study instead of a 24-month local trial,
with the same seven-month review. The China timeline becomes 13 months.

```text
China = 6 + 7 = 13 months
Global completion = max(13, 12, 10, 10, 12, 14) = 14 months
Months saved = 31 - 14 = 17
Value at $8,000,000 contribution per month = 17 x $8,000,000 = $136,000,000
```

The global completion falls from 31 to 14 months. The saving is 17 months, not 18, because the UK
IRP now becomes the critical path at month 14.

### 6.8 The second-order constraint

Once China is accelerated, the binding constraint moves. The UK IRP at month 14 is later than China
at 13 and Japan and Australia at 12. If the UK also files nationally at month 7, the global
completion becomes 13 months and the saving is 18 months, or $144,000,000.

The sequence of constraints is the lesson. Fixing the largest constraint reveals the next one, and a
programme that stops after the first fix leaves a month on the table. The analysis should be repeated
after every change, not run once.

### 6.9 The cost of a delayed first approval

The first approval is the start of revenue. A six-month delay in the US approval, with a contribution
margin of $8,000,000 per month, costs $48,000,000 in deferred contribution, before any allowance for
lost market share.

The delay can come from a manufacturing inspection, a clinical question, or a validation failure.
The financial exposure is the same regardless of cause, which is why the regulatory timeline should
be modelled with the same rigour as the enrolment timeline.

### 6.10 Sensitivity: a longer EU clock stop

The EU planning assumption is a three-month clock stop. If the stop is six months, the EU approval
moves from month 10 to month 13.

```text
EU approval = 7 months active + 6 months stop = 13 months
Global completion = max(31, 13, 12, 12, 10) = 31 months
```

The global completion does not change in the base case because China is still the constraint. It
does change in the accelerated China scenario: global completion becomes 14 months, set by the UK
IRP, and the EU at 13 is close behind. A delay that is invisible in the base case becomes material
once the largest constraint is removed.

This is the general point about critical paths: the importance of a task depends on the other tasks,
so sensitivity analysis must be run on the scenario, not only on the base case.

### 6.11 Statutory time and real time

Every number in the table is a planning estimate built from a statutory clock. The statutory clock
is a lower bound. Real time adds validation, agency questions, inspections, and administrative
approval.

A programme that plans from the statutory clock and does not add an allowance will be late in every
market. A programme that adds a uniform allowance will be wrong in a different way, because the
allowance differs by agency and by dossier quality.

The practical approach is to use the sponsor's own history for each agency, and to use the statutory
clock only when no history exists. The difference between the two is the learning that the
organisation has already paid for.

### 6.12 What the example teaches

1. The global timeline is the longest dependency chain, not the sum of review clocks and not the
   first approval.
2. Local data requirements dominate review clocks when they exist.
3. Reliance is a dependency. It can be slower than a national route if the reference approval is
   late.
4. Accelerating one market changes the critical path, so the analysis must be iterative.
5. The cost of a local trial is large, but the schedule effect is larger.

## 7. Code, query, or calculation

The model computes approval months from a dependency graph, identifies the critical path, and tests
two scenarios. It uses only the standard library.

**Assumptions:**

- all tasks start at month 0 unless they depend on another task;
- durations are planning estimates in months, including expected clock stops;
- the UK IRP depends on the US approval;
- the China local trial is 31 months in the base case and 13 months with a bridging study;
- contribution margin is $8,000,000 per month of earlier access in 2026 US dollars.

```python
TASKS = {
    "US NDA":            {"duration": 10, "depends": []},
    "EU centralised":    {"duration": 10, "depends": []},
    "Japan":             {"duration": 12, "depends": []},
    "Canada":            {"duration": 10, "depends": []},
    "Australia":         {"duration": 12, "depends": []},
    "UK national":       {"duration": 7,  "depends": []},
    "UK IRP (after US)": {"duration": 4,  "depends": ["US NDA"]},
    "China local trial": {"duration": 31, "depends": []},
}

def schedule(tasks):
    finish = {}
    previous = {}
    for name in tasks:
        start = max((finish[d] for d in tasks[name]["depends"]), default=0)
        finish[name] = start + tasks[name]["duration"]
        previous[name] = max(tasks[name]["depends"], key=lambda d: finish[d], default=None)
    return finish, previous

def critical_path(tasks, finish, previous):
    last = max(finish, key=finish.get)
    path = [last]
    while previous[path[-1]]:
        path.append(previous[path[-1]])
    return list(reversed(path)), last

finish, previous = schedule(TASKS)
for name in sorted(finish, key=finish.get):
    print(f"{name:20s} approval month {finish[name]:>3}")
path, last = critical_path(TASKS, finish, previous)
print(f"global completion month {finish[last]}")
print("critical path: " + " -> ".join(path))

scenario = {k: dict(v) for k, v in TASKS.items()}
scenario["China local trial"] = {"duration": 13, "depends": []}
finish2, previous2 = schedule(scenario)
path2, last2 = critical_path(scenario, finish2, previous2)
print(f"scenario completion month {finish2[last2]}")
print(f"months saved {finish[last] - finish2[last2]}")
print(f"value of earlier access USD 2026 {(finish[last] - finish2[last2]) * 8_000_000:,.0f}")

scenario2 = {k: dict(v) for k, v in scenario.items() if k != "UK IRP (after US)"}
finish3, previous3 = schedule(scenario2)
path3, last3 = critical_path(scenario2, finish3, previous3)
print(f"second scenario completion month {finish3[last3]}")
print(f"second scenario months saved {finish[last] - finish3[last3]}")
```

**Expected output:**

```text
UK national          approval month   7
US NDA               approval month  10
EU centralised       approval month  10
Canada               approval month  10
Japan                approval month  12
Australia            approval month  12
UK IRP (after US)    approval month  14
China local trial    approval month  31
global completion month 31
critical path: China local trial
scenario completion month 14
months saved 17
value of earlier access USD 2026 136,000,000
second scenario completion month 13
second scenario months saved 18
```

**Interpretation:**

1. The base case is 31 months, set by a single task. The other six markets are irrelevant to the
   global completion date.
2. Accelerating China to 13 months saves 17 months, not 18, because the UK IRP at month 14 becomes
   the constraint.
3. Removing the UK IRP dependency, which means filing nationally earlier, saves a further month.
4. The model is deliberately simple: it has no resource constraints and no shared submissions. A real
   programme would add validation time, agency question cycles, and the possibility that a delay in
   one market consumes the same regulatory team as another.

**What to vary:** the duration of the China local trial. It is the only input that changes the base
case by more than a few months, and it is the input most often estimated by the commercial team
rather than by the regulatory team.

## 8. Data, standards, and regulatory context

Global regulation is a patchwork of national law, regional procedure, and international guidance.
The International Council for Harmonisation provides the common technical document and a set of
technical guidelines, but it does not create a global approval.

### 8.1 The harmonised layer

| Instrument | What it harmonises | Status |
|---|---|---|
| ICH M4 | Common Technical Document structure | Guideline |
| ICH M1 | Regional administrative information (not harmonised) | Regional |
| ICH E5 | Ethnic factors in the acceptability of foreign data | Guideline |
| ICH E6(R3) | Good clinical practice | Guideline, EU effective 23 July 2025 |
| ICH Q1 | Stability testing | Guideline |
| ICH Q8 to Q12 | Pharmaceutical development and quality systems | Guidelines |
| eCTD | Electronic submission format | Specification |

The CTD is a format, not a decision. Two agencies can receive the same CTD and reach different
conclusions because the legal standard, the comparator, and the population differ.

### 8.2 United States

The FDA reviews investigational new drug applications under 21 CFR Part 312 and marketing
applications under 21 CFR Part 314. Standard review is ten months and priority review is six months
from filing, under the Prescription Drug User Fee Act (PDUFA) commitments. The clock stops when the
FDA asks questions.

The FDA does not rely on foreign approvals for its own decision. It may accept foreign clinical data
under 21 CFR 312.120 if the studies were conducted according to GCP and the data are applicable to
the US population and medical practice, but the application is still reviewed independently.

### 8.3 European Union

The centralised procedure is mandatory for certain products and optional for others. It runs through
the EMA and results in a single marketing authorisation valid in all member states. The review is
210 active days, with a clock stop for questions; accelerated assessment is 150 days. PRIME provides
early dialogue and accelerated assessment for products addressing unmet need.

Conditional marketing authorisation is available for products with a positive benefit-risk balance
where comprehensive data will be provided later. It is not a lower standard; it is a different
evidence timeline.

### 8.4 United Kingdom

Following Brexit, the MHRA is a separate regulator. The International Recognition Procedure, launched
on 1 January 2024, offers two routes. Route A is a 60-day timetable for products approved by a
reference regulator within the previous two years, with the same manufacturing process, and without
Route B triggers. Route B is a 110-day timetable for older approvals or where triggers apply, with
one clock stop at day 70.

The reference regulators include the FDA, EMA and EU member states, PMDA, Health Canada, TGA,
Swissmedic, and the Singapore Health Sciences Authority. The national route remains at 210 days. The
guidance was updated on 9 October 2026, and the Route B reference window was reduced from ten to five
years for applications from 13 July 2026.

### 8.5 Japan

The PMDA reviews and the MHLW approves. The standard review target for a new drug is about twelve
months from receipt to approval, with priority reviews commonly cited at six to nine months. The
Sakigake designation provides priority review and early dialogue for innovative products.

Japan may require local clinical data. ICH E5 provides a framework for assessing ethnic factors and
for deciding whether a bridging study is needed. The decision depends on the product class, the
dose-response relationship, and the similarity of the standard of care.

### 8.6 China

The NMPA regulates drug registration under the Provisions for Drug Registration. The statutory
review timeline is 200 days for a marketing authorisation, 130 days for priority review, and 70 days
for certain rare-disease drugs already marketed overseas. Clinical trial applications are reviewed
within 60 days, and the administrative approval decision follows within 20 days.

China may require local clinical data, and the export of human genetic resources is separately
regulated. A global programme that plans to use Chinese sites must address both the trial approval
and the sample and data transfer rules.

### 8.7 Canada and Australia

Health Canada's review targets are 300 days for a standard new drug submission, 180 days for priority
review, and 200 days for a notice of compliance with conditions. The clock excludes sponsor response
time. Clinical trial applications are reviewed within 30 days.

The TGA's standard prescription medicine registration is 255 working days, with priority and
provisional pathways shorter. Australia operates a comparable overseas regulator reliance pathway,
which can reduce the assessment for products approved by a recognised authority.

### 8.8 Reliance and recognition

Reliance and recognition are defined differently across jurisdictions. In general, reliance means
the relying authority uses the reference authority's assessment as an input but makes its own
decision; recognition means it accepts the decision. The MHRA's IRP is a recognition-style route with
conditions; the Canadian and Australian reliance pathways are closer to abridged review.

### 8.9 What the regulatory standards do not constrain

Regulatory approval does not determine:

- the price;
- whether a payer will cover the product;
- the health technology assessment outcome;
- the commercial launch sequence, except through the approval dependency.

Those are separate decisions covered in topics 04, 05, and 07.

> **Note:** Timelines and procedures in this section are published planning figures at the access
> date. They are not commitments and they change. This is educational material, not legal advice.

## 9. Industry practice and case evidence

### 9.1 The IRP as a natural experiment

The MHRA's IRP, launched on 1 January 2024, was intended to speed UK access by recognising approvals
from trusted regulators. Route A offers a 60-day timetable and Route B 110 days, compared with a
210-day national route.

The design reveals the trade-off. A relying agency can only recognise an approval that exists, so the
UK timetable starts after the reference approval. For a product whose reference approval is later
than the UK national route would have been, reliance is slower. The published guidance does not
resolve this for the sponsor; it is a sequencing decision.

> **Inference:** the IRP is likely to be fastest for products whose reference approval is already
> available at the time of the UK filing, such as a product already approved in the EU and filed in
> the UK afterwards. It is less useful for a simultaneous global filing.

### 9.2 China's statutory timelines and local data

The 2022 Provisions for Drug Registration set statutory timelines that are shorter than the
historical averages: 200 days standard, 130 days priority, and 70 days for certain rare-disease
products already marketed overseas. The statutory clock excludes the time for a local trial and for
applicant responses.

A product that requires a local trial therefore faces a timeline that is dominated by the trial, not
by the review. The published 200-day figure can be accurate and irrelevant at the same time.

### 9.3 Japan's review time

The PMDA's 12-month target for standard new drug review is among the shorter major-market clocks, but
it assumes a complete dossier and no major questions. Priority and Sakigake reviews can be shorter,
and local data requirements can be longer than the review itself.

### 9.4 Divergence after Brexit

The UK's departure from the EU created a separate approval requirement for the UK market. A product
approved through the EU centralised procedure is not automatically approved in Great Britain. The
IRP is the mechanism designed to reduce the duplication, but it does not eliminate the separate
application, the separate label, and the separate post-approval obligations.

### 9.5 What the evidence does not show

Public comparisons of approval times usually compare statutory clocks or median review times, not
total time to access. A country with a fast clock and a long local data requirement can be slower
than a country with a slow clock and no local requirement. A sponsor should build its own estimate
from the requirements that apply to its product.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| One global filing date | All markets planned for the same quarter | Local requirements ignored | Staggered approvals and missed launch | Build a dependency map |
| Local trial not costed | China timeline equals the review clock | Trial requirement omitted | Two-year delay | Cost and schedule the trial |
| Reliance assumed faster | UK IRP planned before the reference approval | Dependency ignored | Later access than national route | Compare both routes |
| GMP inspection not planned | Site not inspected before filing | Quality module incomplete | Approval delayed | Inspection readiness plan |
| Label assumed identical | One global label | National negotiation | Commercial gap | Local labelling workstream |
| Clock stop not budgeted | Approval at the statutory day | Questions take months | Late approval | Add a country-specific stop allowance |
| Submission format wrong | eCTD version mismatch | Technical rejection | Review not started | Check the agency's format |
| Post-approval commitments unowned | No commitments register | Approval conditions forgotten | Compliance finding | Register with owners |
| Local data transfer blocked | Samples cannot leave the country | Human genetic resource rules | Trial delay | Plan approvals early |
| Agency advice ignored | Programme contradicts advice | Review questions | Delay or rejection | Document advice and response |
| Local trial not feasible | No sites or patients in the country | Feasibility not assessed | Local data requirement unmet | Assess feasibility before committing |
| Filing capacity exceeded | Same team assigned to seven dossiers | Parallel filings scheduled | Queueing delays every market | Stagger filings by capacity |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Review clocks are achievable | Builds the timeline | Agency capacity, questions | Compare with published performance |
| Clock stops are known | Converts active days to months | Complexity of the dossier | Use historical stop data |
| Local data requirement is known | Sizes the critical path | Agency discretion | Scientific advice |
| Reference approval date is known | Plans reliance | Delay in the reference market | Scenario analysis |
| Manufacturing site is inspection-ready | Avoids approval delay | Facility or process issue | Mock inspection |
| Label can be agreed | Enables launch | National requirements | Early label discussions |
| No major safety issue emerges | Avoids variation or withdrawal | New safety signal | Pharmacovigilance monitoring |
| Agency guidance is stable | Plan remains valid | Guidance change | Monitor updates |
| Resources are available for parallel filings | Avoids queueing | Team capacity | Resource plan |

### 11.1 Limits of the model

The timeline model assumes deterministic durations and no resource constraints. In reality, the same
regulatory team prepares several submissions, and a delay in one consumes capacity needed by
another. The model therefore understates the effect of a delay in a lead market.

The model also treats reliance as a simple dependency. In practice, a reliance application can be
rejected for procedural reasons, or the reference approval itself can be conditional, which changes
the reliance route. Those branches are not represented.

## 12. Exercises

### Exercise 1. Map three pathways (application, 45 minutes)

A novel monoclonal antibody for a rare disease is ready for filing in the US, the EU, and Japan.

**Deliverable:** a pathway table showing the procedure, the review clock, the local data requirement,
and the expected approval month for each market.

**Acceptance criteria:** each row names the legal basis; clock units are stated; the local data
assumption is justified; the approval months are derived, not asserted.

**Time:** 45 minutes.

### Exercise 2. Compare clock and real time (calculation, 30 minutes)

A dossier is filed in Canada with a 300-day review target and in Australia with a 255-working-day
target. Assume a five-day working week, ten public holidays per year, and a two-month clock stop in
Canada.

**Deliverable:** the calendar-time estimate for each market, and a statement of which is faster.

**Acceptance criteria:** working days converted correctly; the clock stop added; the difference
between statutory and real time explained.

**Time:** 30 minutes.

### Exercise 3. Decide on local data (design, 45 minutes)

A product has global phase 3 data, but the Japanese dose-response may differ because the body weight
distribution is different.

**Deliverable:** a recommendation on whether to run a bridging study, a full local trial, or neither,
with the evidence and the timeline.

**Acceptance criteria:** the ICH E5 framework is applied; the recommendation states the study design
and duration; the cost and the delay are quantified.

**Time:** 45 minutes.

### Exercise 4. Build a sequence (application, 60 minutes)

Using the durations in Section 6, add a scenario in which the EU is the lead market and the UK files
via the IRP after the EU approval.

**Deliverable:** a revised approval table, the new critical path, and the change in global completion.

**Acceptance criteria:** dependencies are explicit; the critical path is traced; the change is
compared with the US-led sequence.

**Time:** 60 minutes.

### Exercise 5. Compare reliance with a national route (calculation, 30 minutes)

A reference approval is expected at month 9. The reliance route takes 4 months. The national route
takes 7 months from month 0.

**Deliverable:** the approval month under each route and the difference.

**Acceptance criteria:** arithmetic shown; the conclusion states which route is faster and why; the
non-time factors (cost, work, label) are noted.

**Time:** 30 minutes.

### Exercise 6. Critique a global filing plan (critique, 45 minutes)

A plan states: "File in seven markets in Q2, using the same dossier, with approval expected within 12
months everywhere."

**Deliverable:** a critique identifying at least five incorrect assumptions and the consequence of
each.

**Acceptance criteria:** each critique is specific to a market; the consequence is a delay, a cost, or
an approval risk; a correction is proposed.

**Time:** 45 minutes.

### Exercise 7. Plan post-approval commitments (design, 45 minutes)

A product is approved with a conditional marketing authorisation in the EU and a post-marketing
requirement in the US.

**Deliverable:** a commitments register with the obligation, the market, the due date, the owner, and
the evidence required.

**Acceptance criteria:** each commitment has a date and an owner; the difference between a condition
and a standard commitment is stated; the resourcing implication is identified.

**Time:** 45 minutes.

### Exercise 8. Write the strategy memo (design, 60 minutes)

**Deliverable:** a two-page memo recommending a lead market and a sequence, with the evidence, the
critical path, and the risks.

**Acceptance criteria:** the recommendation is justified by the reliance relationships and the
evidence; the critical path is identified; at least two risks have mitigations; the memo states what
would change the recommendation.

**Time:** 60 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

A strong table for the US, EU, and Japan:

| Market | Procedure | Clock | Local data | Approval month |
|---|---|---|---|---|
| US | NDA standard | 10 months | None | 10 |
| EU | Centralised | 210 active days | None | 10 with a 3-month stop |
| Japan | PMDA standard | 12 months | Bridging study likely | 18 to 24 |

The Japanese row is the one that changes the plan. A rare disease with a small global trial may not
support a bridging study, and the agency may accept foreign data with a justification.

### 13.2 Exercise 2

```text
Canada: 300 calendar days + 60-day clock stop = 360 days = about 11.8 months
Australia: 255 working days / 5 = 51 weeks = 357 calendar days
Add 10 public holidays as working days lost: 255 + 10 = 265 working days
265 / 5 = 53 weeks = 371 calendar days = about 12.2 months
```

Canada is marginally faster on these assumptions. The difference is small, and the real difference
is likely to come from the clock stop and the agency's questions, not from the statutory clock.

### 13.3 Exercise 3

A strong recommendation applies ICH E5:

- the product is ethnically sensitive if its clearance or dose-response differs by body weight or
  genotype;
- a bridging study should be a PK or PK/PD study in Japanese subjects, typically 20 to 40 subjects,
  designed to show similar exposure;
- if the exposure is similar, the foreign efficacy data can be extrapolated;
- if not, a dose-adjustment study or a local efficacy trial is needed.

The bridging study adds about 6 to 12 months; a full local trial adds 24 months or more. The
recommendation should quantify both and state the assumption about ethnic sensitivity.

### 13.4 Exercise 4

With the EU as lead, the EU approval is at month 10, the UK IRP is at month 14, and the US is at month
10. China remains at month 31. The global completion is unchanged at 31 because China is still the
constraint.

The critical path is unchanged. Changing the lead market moves the first approval and the reliance
dates but not the longest chain. A sequence optimisation that ignores the local trial is optimising
the wrong part of the network.

### 13.5 Exercise 5

```text
National route: month 0 + 7 = month 7
Reliance route: month 9 + 4 = month 13
Difference: 6 months slower
```

The reliance route is faster only if the reference approval arrives before month 3. The rule is:

```text
Reliance is faster if reference approval month + reliance duration < national duration
```

Here, 9 + 4 = 13 > 7, so the national route wins on time.

### 13.6 Exercise 6

A strong critique identifies:

- local data requirements in Japan and China, which invalidate the 12-month assumption;
- the UK's separate post-Brexit approval and the IRP dependency;
- GMP inspection timelines for new facilities;
- clock stops, which extend the statutory review;
- the difference between filing and validation;
- the resource constraint of filing in seven markets at once.

Each critique should state the consequence: a two-year delay in China, a four-month delay for an
inspection, or a validation failure that resets the clock.

### 13.7 Exercise 7

A strong register includes:

| Obligation | Market | Due | Owner | Evidence |
|---|---|---|---|---|
| Confirmatory trial | EU | 24 months after approval | Medical affairs | Protocol and recruitment |
| Post-marketing requirement | US | Per FDA letter | Regulatory affairs | Study report |
| Periodic safety update | Both | Annual or per cycle | Pharmacovigilance | PSUR |
| Risk management plan | EU | With variation | Regulatory affairs | Updated RMP |
| Label update | Both | With variation | Regulatory affairs | Approved label |

A conditional approval is a legal obligation, not an option. A missed commitment can lead to
suspension.

### 13.8 Exercise 8

A strong memo states the lead market, the sequence, the critical path, and the risks. It should say
why the lead market was chosen, for example because its approval unlocks the most reliance pathways
and its review is achievable with the current data. It should identify the local trial as the
critical path if one is required. It should state what would change the recommendation, such as a
manufacturing delay or a new local data requirement.

## 14. Interview questions

**1. What is the difference between the CTD and a marketing application?**
Strong: the CTD is a format; the application is a legal submission with regional modules and a
decision. Weak: treats them as the same.

**2. Why is the first approval not the global launch?**
Strong: because later markets have their own review clocks, local data requirements, and reliance
dependencies. Weak: assumes approvals are simultaneous.

**3. When is reliance slower than a national route?**
Strong: when the reference approval arrives late relative to the national clock. Weak: assumes
reliance is always faster.

**4. What does ICH E5 govern?**
Strong: the acceptability of foreign clinical data and the assessment of ethnic factors, including
bridging studies. Weak: confuses it with GCP.

**5. Why might China add years to a programme?**
Strong: local trial requirements and human genetic resource rules, which sit outside the review
clock. Weak: cites the 200-day review and stops there.

**6. What is the UK IRP?**
Strong: a recognition route with 60-day and 110-day timetables, dependent on a reference regulator's
approval. Weak: describes it as a national review.

**7. How do you plan a GMP inspection?**
Strong: identify inspection-critical sites, assess history, run a mock inspection, and put the
inspection on the critical path. Weak: treats it as the quality team's problem.

**8. What post-approval obligations should be planned at filing?**
Strong: commitments, variations, PSURs, RMPs, and inspections, each with an owner and a date. Weak:
says "post-marketing studies" without specifics.

**9. How would you choose a lead market?**
Strong: by the reliance pathways it unlocks, the evidence fit, and the review risk. Weak: by market
size alone.

## 15. Key terms

| Term | Meaning | See |
|---|---|---|
| Bridging study | Study supporting extrapolation between regions | 5.17 |
| CHMP | EMA scientific committee | 4 |
| CTD | Common Technical Document | 3.1 |
| eCTD | Electronic CTD | 5.22 |
| EMA | European Medicines Agency | 8.3 |
| GMP | Good manufacturing practice | 5.20 |
| Health Canada | Canadian regulator | 8.7 |
| IRP | UK International Recognition Procedure | 8.4 |
| Local clinical data | Data generated in the applying country | 5.17 |
| MAH | Marketing authorisation holder | 4 |
| MHRA | UK regulator | 8.4 |
| MHLW | Japanese ministry that grants approval | 8.5 |
| NCA | National competent authority | 4 |
| NMPA | Chinese regulator | 8.6 |
| PMDA | Japanese review agency | 8.5 |
| PRIME | EMA priority scheme | 8.3 |
| Recognition | Accepting another regulator's decision | 4.2 |
| Reliance | Using another regulator's assessment | 4.2 |
| Sakigake | Japanese priority scheme | 8.5 |
| TGA | Australian regulator | 8.7 |
| Variation | Change to an approved application | 5.21 |

## 16. Sources

All URLs accessed 26 October 2026.

1. ICH. *M4: Common Technical Document*.
   https://www.ich.org/page/ctd
2. ICH. *E5: Ethnic Factors in the Acceptability of Foreign Clinical Data*.
   https://www.ich.org/page/efficacy-guidelines
3. ICH. *E6(R3) Good Clinical Practice*, 6 January 2025.
   https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
4. European Medicines Agency. *Central authorisation of medicines*.
   https://www.ema.europa.eu/en/about-us/what-we-do/authorisation-medicines
5. European Medicines Agency. *PRIME: priority medicines*.
   https://www.ema.europa.eu/en/human-regulatory-overview/PRIME-priority-medicines
6. Medicines and Healthcare products Regulatory Agency. *International Recognition Procedure*,
   updated 9 October 2026.
   https://www.gov.uk/government/publications/international-recognition-procedure/international-recognition-procedure
7. Medicines and Healthcare products Regulatory Agency. *MHRA's new International Recognition
   Procedure goes live from 1 January 2024*.
   https://www.gov.uk/government/news/mhras-new-international-recognition-procedure-irp-goes-live-from-1-january-2024
8. Pharmaceuticals and Medical Devices Agency. *Review timelines and consultations*.
   https://www.pmda.go.jp/english/
9. National Medical Products Administration. *Provisions for Drug Registration*, 2022.
   https://english.nmpa.gov.cn/2022-06/30/c_785628_8.htm
10. Health Canada. *Drug submission review timelines*.
    https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/applications-submissions.html
11. Therapeutic Goods Administration. *Prescription medicines registration process*.
    https://www.tga.gov.au/how-we-regulate/ingredients-and-products/prescription-medicines
12. US Government. *21 CFR Part 314*.
    https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-314
13. European Commission. *Regulation (EC) No 726/2004*.
    https://eur-lex.europa.eu/eli/reg/2004/726/oj

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_market-access.md`](05_market-access.md)
- Next topic: [`07_pricing-and-contracting.md`](07_pricing-and-contracting.md)
- HTA: [`04_health-technology-assessment.md`](04_health-technology-assessment.md)
- Module survey: [`../../course/25_trial_operations_hta_global_regulatory.md`](../../course/25_trial_operations_hta_global_regulatory.md)
- Labs: [`../../labs/`](../../labs/)
