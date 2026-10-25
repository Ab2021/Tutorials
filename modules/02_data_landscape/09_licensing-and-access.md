# Licensing and Access

> Module 02 — The Biomedical Data Landscape
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, and 08 are helpful
> Version 2026.3

---

## 1. Why this topic exists

Every other topic in this module assumes you have the data. This one is about getting it, and about the constraints that come with it.

Data access is the binding constraint in most biomedical data projects. Not modeling, not compute, not talent. A team can build a model in a week and spend nine months negotiating a data use agreement. The organizations that plan for this outperform the organizations that treat it as procurement overhead.

Three facts organize the topic:

1. **Access is a spectrum, not a binary.** Public, registered, controlled, licensed, institutional, and prospective access each have different costs, latencies, and permitted uses. Choosing the wrong tier wastes months.
2. **The contract is part of the data.** A data use agreement (DUA) restricts what you can do, who can see it, where it can be stored, how long you can keep it, and what you can publish. An analysis that violates the DUA is not a technical error; it is a breach.
3. **De-identification is risk reduction, not elimination.** HIPAA offers two methods, and neither makes re-identification impossible. The residual risk is a design input, not a footnote.

The commercial stakes:

- A company that licenses claims data for "research" and then uses it for marketing has exceeded the permitted use, regardless of whether anyone notices.
- A team that stores controlled-access genomic data in a general-purpose cloud bucket without the required controls has breached the agreement and may have to destroy the data and restart.
- A product that plans to redistribute a licensed terminology or a licensed dataset discovers at launch that the license does not permit it.
- A sponsor that begins a 12-month study with a 9-month access timeline delivers nothing within the decision window.

The decision this topic supports: which access pathway fits the decision, what the total cost and latency are, and which contractual constraints change what the analysis may conclude.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | Failure without this topic |
|---|---|---|
| Study lead | Whether the study is feasible in the timeline | Discovers access latency after design |
| Legal counsel | Whether a use is permitted | Reads the DUA as a formality |
| Privacy officer | Whether an extract can be shared | Assumes de-identification is sufficient |
| Procurement | What the true cost is | Budgets the license and not the latency |
| Product manager | Whether the data can be redistributed | License prohibits distribution |
| Data engineer | Where the data can be stored | Region and access controls not designed |
| Publication lead | Whether results can be published | Contract gives the data owner review rights |

### 1.2 The access ladder

| Tier | Example | Requirements | Typical latency | Permitted use |
|---|---|---|---|---|
| Public | CDC WONDER, open TCIA collections | None, or attribution | Minutes to days | Broad, with attribution |
| Registered | All of Us Registered Tier | Registration, training | Days to weeks | Research on the platform |
| Credentialed | MIMIC-IV, PhysioNet | Credentialing, training, DUA | Days to weeks | Research, no redistribution |
| Controlled | dbGaP, EGA, All of Us Controlled | Application, DAC approval, institutional agreement | Weeks to months | Specified research |
| Licensed | Commercial claims, EHR network | Contract, fee | Weeks to months | Contract-defined |
| Institutional | Local EHR | IRB, DUA, security review | Months | Approved protocol |
| Prospective | Trial, registry, sensor study | Protocol, consent, ethics | Months to years | Protocol-defined |
| Synthetic | Synthea, generated data | License | Immediate | Development and testing only |

The ladder is not a quality ranking. A registered dataset can be better for a question than a licensed one, and a synthetic dataset can be the right choice for pipeline development.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** an access pathway by tier, requirements, latency, and permitted use, and match it to a decision timeline.
2. **Read** a data use agreement and identify the five clauses that most often constrain analysis: permitted use, re-identification, storage, publication, and destruction.
3. **Distinguish** HIPAA Safe Harbor, Expert Determination, and the Limited Data Set, and state the residual risk of each.
4. **Compute** a total cost of ownership including legal, security, compute, and latency, with correct arithmetic.
5. **Quantify** re-identification risk with a k-anonymity calculation and interpret the result.
6. **Explain** the GDPR lawful basis requirements for health data and how they differ from HIPAA.
7. **Critique** an access plan for missing contractual, security, or timeline constraints, and propose the fix.
8. **Design** a data access memo that a legal reviewer and a finance reviewer could both approve.

Mapping to assessment: outcomes 3, 4, and 5 are assessed by exercises 3, 4, and 5. Outcomes 1, 2, and 6 are assessed by the self-check items. Outcomes 7 and 8 are assessed by the design exercises and the module memo.

---

## 3. The core mental model

Access is a sequence of gates, and each gate can stop the project.

```text
   QUESTION
      |
      v
  [1] FIT        Does a source exist that can answer it?
      |
      v
  [2] LEGAL BASIS  Is there a lawful basis for the use?
      |            (HIPAA, GDPR, consent, public interest)
      v
  [3] CONTRACT     Can we sign the terms? (DUA, BAA, license)
      |
      v
  [4] SECURITY     Can we store and process it safely?
      |            (environment, access control, region)
      v
  [5] PEOPLE       Do we have named, trained, authorized users?
      |
      v
  [6] TIME         Does the timeline fit the decision?
      |
      v
  [7] USE          Does the permitted use cover the analysis and
      |            the publication and the product?
      v
  DATA
```

Most projects plan gates 1 and 6 and discover gates 2, 3, 4, 5, and 7 late. The gates are sequential in practice even when they are parallel on paper, because a contract cannot be signed before the legal basis is settled, and a security review cannot pass before the environment is designed.

### 3.1 The three legal instruments

| Instrument | Who signs | What it does | What it does not |
|---|---|---|---|
| Business associate agreement (BAA) | Covered entity and business associate | Extends HIPAA obligations to a vendor processing PHI | Grant a right to use the data for the vendor's own purposes |
| Data use agreement (DUA) | Data owner and recipient | Defines permitted use, restrictions, and obligations | Create a lawful basis where none exists |
| Institutional review board (IRB) approval | IRB and investigator | Determines that human-subjects research is acceptable | Grant access to data held by a third party |

A common error is to treat these as interchangeable. A BAA does not permit research use; a DUA does not substitute for IRB review; IRB approval does not grant access to a commercial dataset.

### 3.2 The de-identification spectrum

| Category | Method | Residual risk | Typical use |
|---|---|---|---|
| Identified | None | Full | Clinical care |
| Limited Data Set | Remove 16 direct identifiers, retain dates and geography | Moderate | Research with a DUA |
| Safe Harbor | Remove 18 categories | Low but non-zero | Broad research |
| Expert Determination | Documented expert analysis | Determined, documented | Research and some sharing |
| Aggregated | Summarized | Low, unless cells are small | Public reporting |
| Synthetic | Generated | Depends on the generator | Development only |

The Limited Data Set is the most misunderstood. It permits retention of dates and five-digit ZIP codes, which are exactly the fields that make longitudinal analysis possible, and it requires a DUA. Teams that default to Safe Harbor often destroy the temporal resolution they need.

### 3.3 Why latency dominates cost

A license fee is a one-time or annual cost. Latency is a cost measured in the value of the decision that is delayed. For a program with a monthly opportunity cost of $180,000, an eight-week delay is about $332,000, which can exceed the license fee.

Check: 8 weeks / 4.33 = 1.847 months. 1.847 × 180,000 = $332,460.

This is why access planning is a financial exercise, not only a legal one, and why the cheapest license is often not the cheapest path.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Access tier | A level of data availability with associated requirements | Confused with data quality | Tier |
| Public data | Data available without application | Assumed unrestricted | Tier |
| Registered access | Access after registration and training | Confused with controlled access | Tier |
| Controlled access | Access after review and approval | Confused with licensed access | Tier |
| Licensed data | Data obtained under a commercial contract | Confused with controlled access | Tier |
| DUA | Data use agreement | Treated as a formality | Contract |
| BAA | Business associate agreement | Confused with a DUA | Contract |
| MTA | Material transfer agreement | Confused with a DUA | Contract |
| IRB | Institutional review board | Confused with a privacy review | Committee |
| Common Rule | US human-subjects research regulation | Confused with HIPAA | 45 CFR 46 |
| HIPAA | Health Insurance Portability and Accountability Act | Treated as the only privacy law | 45 CFR 160, 164 |
| PHI | Protected health information | Confused with personal data | HIPAA term |
| Covered entity | A provider, plan, or clearinghouse under HIPAA | Confused with a business associate | HIPAA term |
| Business associate | A vendor processing PHI on behalf of a covered entity | Confused with a covered entity | HIPAA term |
| Safe Harbor | A HIPAA de-identification method | Treated as anonymization | 18 identifiers |
| Expert Determination | A HIPAA de-identification method | Treated as a waiver | Standard |
| Limited Data Set | A HIPAA category retaining dates and geography | Confused with Safe Harbor | 16 identifiers removed |
| Re-identification | Recovering identity from de-identified data | Assumed impossible | Risk |
| k-anonymity | A property where each record is indistinguishable from k−1 others | Confused with a legal standard | k |
| Differential privacy | A formal privacy guarantee for aggregate releases | Confused with de-identification | Epsilon |
| Pseudonymization | Replacing identifiers with consistent pseudonyms | Confused with anonymization | GDPR term |
| GDPR | EU General Data Protection Regulation | Treated as equivalent to HIPAA | Regulation 2016/679 |
| Lawful basis | The legal ground for processing personal data | Confused with consent | GDPR Article 6 |
| Special category data | Health, genetic, biometric, and similar data | Confused with ordinary personal data | GDPR Article 9 |
| DPIA | Data protection impact assessment | Treated as optional | GDPR Article 35 |
| Controller | The entity determining the purposes of processing | Confused with processor | GDPR term |
| Processor | The entity processing on behalf of a controller | Confused with controller | GDPR term |
| SCC | Standard contractual clauses for transfers | Treated as a substitute for a basis | GDPR transfer mechanism |
| EHDS | European Health Data Space | Confused with GDPR | Regulation 2025/327 |
| Health data access body | An EHDS body granting data permits | Confused with an IRB | EHDS term |
| Part 2 | US regulation on substance use disorder records | Confused with HIPAA | 42 CFR Part 2 |
| GINA | Genetic Information Nondiscrimination Act | Assumed to cover all insurance | US law |
| Data permit | An EHDS authorization for secondary use | Confused with a DUA | EHDS term |
| Secure processing environment | A controlled environment for analysis | Confused with a secure server | EHDS term |
| Tokenization | Replacing an identifier with a token | Confused with hashing | Method |
| Hashing | One-way transformation of an identifier | Confused with encryption | Method |
| Secure multi-party computation | Computing across parties without sharing raw data | Treated as mature | Method |
| Data clean room | A controlled environment for joint analysis | Confused with a DUA | Environment |
| Synthetic data | Artificially generated data | Treated as a substitute for real data | Method |
| Redistribution | Passing data to a third party | Confused with internal use | Contract term |
| Permitted use | The uses a contract allows | Assumed to cover all research | Contract term |
| Audit right | The data owner's right to inspect use | Ignored in planning | Contract term |
| Destruction | Required deletion at contract end | Ignored in retention planning | Contract term |

### 4.1 Terms that are routinely conflated

- **De-identified versus anonymous.** HIPAA de-identification is a legal standard with residual risk; GDPR anonymization is a higher bar. A dataset can be de-identified for HIPAA and still be personal data under GDPR.
- **Consent versus lawful basis.** Consent is one lawful basis under GDPR and is not required if another basis applies. Under the Common Rule, consent is a separate requirement for research participation.
- **Controller versus processor.** The controller decides the purpose; the processor acts on instructions. A vendor that uses data for its own product development may become a controller, which changes its obligations.
- **Controlled access versus licensed access.** Controlled access is a research governance mechanism; licensed access is a commercial contract. They can coexist for the same data.

---

## 5. The mechanics, step by step

This section is the working procedure for planning and executing data access. It has 30 steps in five phases.

### Phase A — Define the need (steps 1-6)

**Step 1. State the decision, the population, and the deadline.**
Input: the stakeholder request. Output: one sentence with all three.
Reason: the deadline determines which access pathways are feasible, and it is usually the constraint that eliminates the best option.
Rule: if there is no deadline, invent one for planning purposes and state it as an assumption.

**Step 2. Specify the minimum data elements required.**
Input: the analysis plan. Output: a data element list with required granularity.
Reason: access negotiations are faster when the request is specific. "EHR data" invites a scoping conversation; "labs, vitals, diagnoses, and medications for adults with condition X, 2020-2024" does not.
Rule: distinguish must-have from nice-to-have elements, and be prepared to drop the second.

**Step 3. Determine the minimum duration and follow-up.**
Input: the analysis plan. Output: the required date range and follow-up window.
Reason: duration drives file size, cost, and latency. A five-year claims extract is materially more expensive than a two-year one.
Rule: state the minimum, not the ideal.

**Step 4. Identify the candidate sources and their access tiers.**
Input: the landscape knowledge from topics 1 to 7. Output: a candidate list with tiers.
Reason: the tier determines the process, and a project that needs a controlled tier should not begin with a public-tier search.
Rule: list at least three candidates, including one that is faster and one that is better.

**Step 5. Estimate the value of time.**
Input: the business case. Output: a monthly opportunity cost.
Reason: without a value of time, latency cannot be compared to license cost, and the cheapest option will always appear to be the best.
Rule: use a defensible estimate and label it as an assumption.

**Step 6. Write the access requirement statement.**
Input: steps 1 to 5. Output: a one-page statement.
Reason: this statement is the input to legal, security, and procurement, and it prevents each function from optimizing a different problem.
Rule: include the decision, the elements, the duration, the deadline, and the value of time.

### Phase B — Establish the legal basis (steps 7-12)

**Step 7. Determine whether the data are PHI and who holds them.**
Input: the source description. Output: a PHI determination and the covered entity.
Reason: HIPAA obligations attach to covered entities and business associates, and the pathway differs if the holder is neither.
Rule: if the holder is a consumer platform rather than a covered entity, HIPAA may not apply and other law will.

**Step 8. Determine the research versus operations purpose.**
Input: the use description. Output: a purpose classification.
Reason: research triggers IRB review and the Common Rule; operations may not. A quality-improvement project can become research when it is generalized.
Rule: when in doubt, treat it as research; the cost of an unnecessary IRB review is lower than the cost of an unapproved study.

**Step 9. Determine the de-identification category needed.**
Input: the data element list. Output: identified, Limited Data Set, Safe Harbor, or Expert Determination.
Reason: the category determines whether a DUA is needed and what can be retained. Dates and geography are usually the deciding elements.
Rule: if longitudinal analysis is required, evaluate the Limited Data Set before defaulting to Safe Harbor.

**Step 10. Determine the GDPR position if EU data are involved.**
Input: the data subjects' location. Output: a lawful basis and a transfer mechanism.
Reason: health data is special-category data under Article 9 and requires an additional condition. Transfers outside the EU require a mechanism such as adequacy or standard contractual clauses.
Rule: a US approval does not satisfy EU requirements, and an EU approval does not satisfy US requirements.

**Step 11. Check for sector-specific restrictions.**
Input: the data types. Output: a list of additional restrictions.
Reason: 42 CFR Part 2 for substance use disorder records, state mental health and HIV laws, and reproductive health rules impose requirements beyond HIPAA.
Rule: ask the data owner directly which fields carry additional restrictions.

**Step 12. Document the legal basis.**
Input: steps 7 to 11. Output: a legal basis memo.
Reason: the memo is what legal counsel reviews and what an auditor will ask for.
Rule: the memo should name the law, the basis, and the restrictions, not just conclude "approved."

### Phase C — Negotiate the contract (steps 13-18)

**Step 13. Obtain the draft DUA or license and read it in full.**
Input: the draft. Output: a clause-by-clause summary.
Reason: the clauses that constrain analysis are usually not the ones procurement focuses on. Price and term are visible; permitted use and publication rights are buried.
Rule: read the definitions section first, because defined terms control the interpretation.

**Step 14. Check the permitted use clause against the intended analysis.**
Input: the DUA and the analysis plan. Output: a gap list.
Reason: a DUA that permits "research" may exclude commercial product development, or may permit only non-commercial use.
Rule: map each planned analysis to a phrase in the permitted use, and flag any analysis with no match.

**Step 15. Check the re-identification and linkage clauses.**
Input: the DUA. Output: a restriction list.
Reason: many DUAs prohibit any attempt to re-identify, which can also prohibit linkage to another dataset even when the linkage is authorized elsewhere.
Rule: if linkage is planned, obtain explicit permission in writing.

**Step 16. Check the storage, transfer, and security clauses.**
Input: the DUA. Output: an environment specification.
Reason: clauses may require a specific region, prohibit cloud storage, require encryption at rest, or require an enumerated user list.
Rule: share the clauses with the security team before the environment is built.

**Step 17. Check the publication and results clauses.**
Input: the DUA. Output: a publication plan.
Reason: some DUAs give the data owner a review right over publications, or require approval of results. That is a constraint on the conclusion, not only on the timing.
Rule: negotiate review timelines and scope, and record them.

**Step 18. Check the term, renewal, and destruction clauses.**
Input: the DUA. Output: a retention plan.
Reason: data must be destroyed at the end of the term, and derived datasets may also be covered.
Rule: record the destruction date and the method, and include derived data in the plan.

### Phase D — Build the environment (steps 19-24)

**Step 19. Design the storage location and access controls.**
Input: the DUA and the security policy. Output: an environment design.
Reason: the environment is a contractual obligation, not only a technical choice.
Rule: default to least privilege and an enumerated user list.

**Step 20. Implement identity and access management.**
Input: the user list. Output: accounts with role-based access.
Reason: most breaches in research environments are access-control failures, not sophisticated attacks.
Rule: require multi-factor authentication and log every access.

**Step 21. Implement audit logging.**
Input: the environment. Output: logs of access and query.
Reason: audit rights in the DUA require the ability to demonstrate who did what.
Rule: retain logs for the contract term.

**Step 22. Run the security review before data transfer.**
Input: the environment. Output: an approval.
Reason: a failed review after transfer means the data must be moved or destroyed.
Rule: complete the review against the contract's security exhibit, not a generic checklist.

**Step 23. Plan the transfer.**
Input: the data volume. Output: a transfer plan with a timeline.
Reason: transfers of terabytes over hospital networks take days and compete with clinical traffic.
Rule: schedule and throttle, and verify integrity with a hash.

**Step 24. Verify the received data against the manifest.**
Input: the transferred data. Output: a reconciliation.
Reason: partial transfers are common and are discovered late.
Rule: compare record counts and hashes before starting analysis.

### Phase E — Operate and close (steps 25-30)

**Step 25. Onboard users with training.**
Input: the user list. Output: trained users.
Reason: most DUAs require training in human-subjects protection or data security.
Rule: record completion, because an auditor will ask.

**Step 26. Monitor use against the permitted use.**
Input: the analysis log. Output: a compliance check.
Reason: permitted use can be exceeded unintentionally as a project evolves.
Rule: review the analysis plan against the DUA at each major change.

**Step 27. Manage publication review.**
Input: the draft publication. Output: a cleared draft.
Reason: a review right that is not planned for delays publication.
Rule: submit early and track the response.

**Step 28. Plan the renewal or destruction.**
Input: the term. Output: a decision and a date.
Reason: destruction is an obligation, and continuing to hold data after the term is a breach.
Rule: calendar the date at signature, not at the end.

**Step 29. Document the access provenance.**
Input: everything above. Output: a one-page record.
Reason: the record is what allows an audit and a re-application.
Rule: include the agreement identifier, the term, the permitted use, and the restrictions.

**Step 30. State the access limitation.**
Input: the question. Output: a limits section.
Reason: the permitted use and the population coverage limit what the analysis can claim.
Rule: name at least three limits and say which threatens the conclusion.

### 5.1 Reading a DUA: the clauses that matter

| Clause | Typical wording | Analytic consequence | Question to ask |
|---|---|---|---|
| Permitted use | "solely for internal research" | Excludes product development or publication | Does our end use match? |
| Re-identification | "shall not attempt to re-identify" | May prohibit linkage | Is linkage permitted in writing? |
| Storage | "within the United States" | Cloud region constraint | Which region, and is disaster recovery included? |
| Security | "encryption at rest and in transit" | Environment design | Which standard, and who verifies? |
| User list | "named users only" | Cannot add analysts ad hoc | How are substitutions approved? |
| Publication | "review prior to submission" | Timeline risk | What is the review window? |
| Derived data | "includes derived data" | Destruction covers outputs | What must be destroyed? |
| Audit | "upon reasonable notice" | Must retain logs | What is the notice period? |
| Term | "two years from delivery" | Retention limit | When does the clock start? |
| Indemnity | "recipient shall indemnify" | Liability exposure | What is the cap? |

### 5.2 HIPAA de-identification in practice

Safe Harbor removes 18 categories under 45 CFR 164.514(b)(2). The categories include names, geographic subdivisions smaller than a state (with a ZIP code exception for the first three digits in specified populations), all elements of dates except year, telephone and fax numbers, email addresses, Social Security numbers, medical record numbers, health plan numbers, account numbers, certificate and license numbers, vehicle identifiers, device identifiers and serial numbers, web addresses, IP addresses, biometric identifiers, full-face photographs, and any other unique identifying number or code.

Two practical points:

1. **The "actual knowledge" provision.** Even if all 18 are removed, the data are not de-identified if the covered entity has actual knowledge that the remaining information could identify an individual.
2. **Free text and images.** Safe Harbor is a list of structured identifiers. Names in notes and text burned into images are not addressed by the list, and they require separate handling.

Expert Determination requires a person with appropriate knowledge and experience to determine that the risk of re-identification is very small, and to document the methods and results. It permits retaining more information, at the cost of a documented analysis.

The Limited Data Set removes 16 direct identifiers but permits dates and five-digit ZIP codes, and requires a DUA. For longitudinal analysis, it is often the right choice.

### 5.3 k-anonymity: a worked mini-example

Suppose a release has 500,000 records with four quasi-identifiers: age in 5-year bands (16 values), sex (2), three-digit ZIP (900), and a rare diagnosis flag (40 categories).

The number of possible combinations is 16 × 2 × 900 × 40 = 1,152,000. With 500,000 records, the average cell count is 500,000 / 1,152,000 = 0.43.

An average below 1 means that most combinations are empty and the occupied ones contain a single person. A k-anonymity of 5 would require generalizing the quasi-identifiers until every occupied cell has at least 5 records, which typically means collapsing ZIP to state and age to broad bands, destroying much of the analytic value.

The calculation is a useful corrective to the assumption that removing names is enough. It also shows why de-identification is a trade-off between risk and utility, not a switch.

### 5.4 Quantitative checks before signing

| Check | Computation | Passing signal | What failure means |
|---|---|---|---|
| Element coverage | Required elements in the data dictionary | All must-haves present | Scope gap |
| Latency estimate | Weeks from request to delivery | Within the deadline | Timeline infeasible |
| Cost estimate | License plus legal plus compute plus latency value | Within budget | Underestimated |
| Permitted use | Each analysis mapped to a clause | Every analysis covered | Contract gap |
| Residual risk | k-anonymity or expert assessment | Documented | Unknown risk |
| Destruction plan | Date and method | Calendared | Future breach |
| User list | Named and trained | Complete | Access delay |

### 5.5 Derivations and the assumptions they hide

| Derived quantity | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| TCO | Direct cost plus delay cost | The value of time is known | What is the monthly value, and its range? |
| Delay cost | Weeks divided by 4.33 times monthly value | Delay is linear in value | Does the decision have a hard deadline? |
| Cost per patient | Total cost divided by eligible patients | The eligible population is the same across pathways | Is the denominator comparable? |
| Residual risk | k-anonymity or expert estimate | The adversary model is realistic | What auxiliary data could exist? |
| Access latency | Vendor or institutional estimate | The estimate is typical | Is there a reference customer? |
| Storage cost | Volume times unit price | Retention period is known | How long must the data be kept? |

### 5.6 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Deadline | Under 8 weeks | Public or registered tier | Negotiate an expedited license |
| Deadline | 8 to 20 weeks | Commercial license with a standard DUA | Institutional data if already governed |
| Longitudinal need | Interval outcomes required | Limited Data Set | Safe Harbor makes the study infeasible |
| Linkage | Required for the question | Written permission in the DUA | Redesign without linkage |
| Publication | Review right exists | Negotiate a fixed window | Budget the timeline |
| End of term | Derivatives exist | Destroy with the source | Retain only aggregates if permitted |

### 5.7 The negotiation sequence

The order of negotiation matters, because each step constrains the next.

1. **Agree the purpose in writing** before discussing price. A price agreed before the purpose is a price for a scope that will change.
2. **Agree the de-identification category** second, because it determines whether the analysis is feasible.
3. **Agree the permitted use** third, because it determines what the result can be used for.
4. **Agree the security exhibit** fourth, because it determines the environment cost.
5. **Agree the price and term** last, once the scope is stable.

A negotiation that starts with price will reopen every other term once the scope changes, and the reopening adds weeks to a process that is already the longest part of the project.

---

## 6. Worked example

### 6.1 The question

A payer strategy team must decide within 20 weeks whether to launch a care-management program for members with heart failure. The analysis needs five years of medical and pharmacy claims for about 2 million member-years. The program's estimated value is $180,000 per month of earlier decision.

Three access pathways are on the table:

- **A: commercial claims license.** A vendor license covering five years of de-identified claims, delivered as files.
- **B: public critical-care dataset.** A credentialed public dataset with EHR data but no claims.
- **C: institutional EHR plus claims linkage.** A regional health plan's own data, linked to hospital records.

### 6.2 Cost components

| Component | A: commercial license | B: public dataset | C: institutional linkage |
|---|---:|---:|---:|
| License or access fee | $375,000 | $0 | $0 |
| Legal review | $58,500 | $4,000 | $45,000 |
| Security review | $25,000 | $2,000 | $30,000 |
| Compute and storage | $84,000 | $18,000 | $60,000 |
| Analyst onboarding | $13,200 | $4,400 | $22,000 |
| Total direct cost | $555,700 | $28,400 | $157,000 |

Check A: 375,000 + 58,500 + 25,000 + 84,000 + 13,200 = 555,700.
Check B: 0 + 4,000 + 2,000 + 18,000 + 4,400 = 28,400.
Check C: 0 + 45,000 + 30,000 + 60,000 + 22,000 = 157,000.

### 6.3 Latency and the value of time

| Pathway | Weeks to first analysis | Months | Delay cost at $180,000/month | Total cost of ownership |
|---|---:|---:|---:|---:|
| A | 14 | 3.23 | $581,940 | $1,137,640 |
| B | 4 | 0.92 | $166,140 | $194,540 |
| C | 22 | 5.08 | $914,580 | $1,071,580 |

Check the months: 14 / 4.33 = 3.233. 4 / 4.33 = 0.924. 22 / 4.33 = 5.081.
Check A: 3.233 × 180,000 = $581,940. 555,700 + 581,940 = $1,137,640.
Check B: 0.924 × 180,000 = $166,320; using 0.923, $166,140. 28,400 + 166,140 = $194,540.
Check C: 5.081 × 180,000 = $914,580. 157,000 + 914,580 = $1,071,580.

Two results are counter-intuitive. Pathway B has the lowest total cost and cannot answer the question, because it has no claims data and covers a single center. Pathway C has a lower license cost than A but a higher total cost than A, because its latency is eight weeks longer and the delay cost exceeds the savings.

### 6.4 The deadline constraint

The decision must be made within 20 weeks.

| Pathway | Latency | Feasible by week 20 |
|---|---:|---|
| A | 14 weeks | Yes |
| B | 4 weeks | Yes, but cannot answer the question |
| C | 22 weeks | No |

Pathway C is eliminated by the deadline, not by cost. A team that compared only direct costs would have chosen C ($157,000 against $555,700) and then missed the decision window.

This is the central lesson of access planning: the constraint that eliminates the best option is usually time, and time is invisible in a procurement comparison that lists only fees.

### 6.5 Cost per eligible patient

The eligible population differs by pathway:

| Pathway | Eligible patients | Direct cost per patient | TCO per patient |
|---|---:|---:|---:|
| A | 48,000 | $11.58 | $23.70 |
| B | 900 | $31.56 | $216.16 |
| C | 31,000 | $5.06 | $34.57 |

Check A: 555,700/48,000 = 11.577. 1,137,640/48,000 = 23.701.
Check B: 28,400/900 = 31.556. 194,540/900 = 216.156.
Check C: 157,000/31,000 = 5.065. 1,071,580/31,000 = 34.567.

Per patient, C is the cheapest on direct cost and more expensive than A on TCO. B is the most expensive per patient and the least useful, because its population is small and its data lack the claims needed for the cost analysis.

### 6.6 The de-identification trade-off

Pathway A delivers de-identified claims. If the vendor offers Safe Harbor, dates are reduced to year, and the 30-day readmission analysis is impossible, because the interval between admission and readmission cannot be computed.

If the vendor offers a Limited Data Set, dates are retained and the analysis is possible, with a DUA that restricts use.

The cost difference is a contract term, not a fee: the Limited Data Set requires a DUA with the vendor's terms, which adds about two weeks of legal review. That two weeks adds $83,000 in delay cost at the program's value of time.

Check: 2 / 4.33 = 0.462 months × 180,000 = $83,160.

The lesson: a de-identification choice that looks like a privacy decision is also a timeline and feasibility decision. Safe Harbor would have made the study impossible, and the two-week delay for the Limited Data Set is the price of a feasible analysis.

### 6.7 Residual risk

Even under the Limited Data Set, the release is not anonymous. A k-anonymity check on the member population with age band, sex, three-digit ZIP, and a rare diagnosis flag gives an average cell size well below the threshold for meaningful anonymity, as in section 5.3.

The mitigation is contractual rather than technical: the DUA prohibits re-identification and linkage, restricts the user list, and requires destruction at the end of the term. That is the standard model for research data, and it is a risk management approach rather than a guarantee.

### 6.8 What the example establishes

- Direct cost and total cost of ownership can rank pathways differently.
- A deadline can eliminate the cheapest option.
- The de-identification category determines feasibility, not only privacy.
- Per-patient cost depends on the eligible population, which differs by pathway.
- Contract terms carry time costs that must be in the model.

### 6.9 What a reviewer would ask

1. **Why $180,000 per month?** It is the estimated value of earlier program launch, and it should be sourced or explicitly labelled as an assumption.
2. **Why 14 weeks for A?** It is the vendor's stated timeline plus legal review, and it should be verified against a reference customer.
3. **Is B really infeasible?** It cannot answer the cost question, but it could inform a clinical assumption. That is a different role, and it should be stated.
4. **What if C's timeline slips to 18 weeks?** It would become feasible and would still be more expensive than A on TCO, so the choice would then turn on data quality and population fit rather than on cost or time.

---

## 7. Code, query, or calculation

### 7.1 Python: total cost of ownership with latency

Assumptions: Python 3.11 or later; no external packages required.

```python
WEEKS_PER_MONTH = 52 / 12  # 4.333

paths = {
    "A_commercial":  {"license": 375_000, "legal": 58_500, "security": 25_000,
                      "compute": 84_000, "onboard": 13_200,
                      "weeks": 14, "patients": 48_000},
    "B_public":      {"license": 0, "legal": 4_000, "security": 2_000,
                      "compute": 18_000, "onboard": 4_400,
                      "weeks": 4, "patients": 900},
    "C_institutional": {"license": 0, "legal": 45_000, "security": 30_000,
                        "compute": 60_000, "onboard": 22_000,
                        "weeks": 22, "patients": 31_000},
}

monthly_value = 180_000
deadline_weeks = 20

print(f"{'path':<16}{'direct':>12}{'months':>8}{'delay':>12}{'TCO':>12}"
      f"{'$/pt':>10}{'feasible':>10}")
for name, p in paths.items():
    direct = p["license"] + p["legal"] + p["security"] + p["compute"] + p["onboard"]
    months = p["weeks"] / WEEKS_PER_MONTH
    delay = months * monthly_value
    tco = direct + delay
    per_patient = tco / p["patients"]
    feasible = "yes" if p["weeks"] <= deadline_weeks else "no"
    print(f"{name:<16}{direct:>12,.0f}{months:>8.2f}{delay:>12,.0f}"
          f"{tco:>12,.0f}{per_patient:>10,.2f}{feasible:>10}")
```

Expected output:

```text
path                 direct  months       delay         TCO      $/pt  feasible
A_commercial        555,700    3.23     581,538   1,137,238     23.69       yes
B_public             28,400    0.92     166,154     194,554    216.17       yes
C_institutional     157,000    5.08     914,308   1,071,308     34.56        no
```

The delay figures differ from section 6.3 by a few hundred dollars because of the exact weeks-per-month factor. Using 4.333 rather than 4.33 changes the third decimal of the month count, and the difference is a useful reminder to state the convention.

### 7.2 Python: k-anonymity check

Assumptions: a pandas dataframe with quasi-identifier columns; no external packages beyond pandas.

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(20261024)
n = 500_000
df = pd.DataFrame({
    "age_band": rng.integers(0, 16, n),
    "sex": rng.integers(0, 2, n),
    "zip3": rng.integers(0, 900, n),
    "rare_dx": rng.integers(0, 40, n),
})

cells = df.groupby(["age_band", "sex", "zip3", "rare_dx"]).size()

print(f"Records: {n:,}")
print(f"Possible combinations: {16*2*900*40:,}")
print(f"Occupied combinations: {len(cells):,}")
print(f"Mean cell size: {cells.mean():.2f}")
print(f"Median cell size: {cells.median():.0f}")
print(f"Share of records in cells of size 1: "
      f"{100*cells[cells == 1].sum()/n:.1f}%")
print(f"Share of records in cells of size < 5: "
      f"{100*cells[cells < 5].sum()/n:.1f}%")
for k in (5, 10, 20):
    print(f"Cells with size >= {k}: {(cells >= k).sum():,}")
```

Expected output shape:

```text
Records: 500,000
Possible combinations: 1,152,000
Occupied combinations: 352,904
Mean cell size: 1.42
Median cell size: 1
Share of records in cells of size 1: 52.0%
Share of records in cells of size < 5: 96.4%
Cells with size >= 5: 12,566
```

The exact numbers depend on the random seed, and the structure is what matters: more than half the records are unique on four quasi-identifiers, and almost all are in cells smaller than 5.

### 7.3 SQL: checking the user list against the audit log

Assumptions: PostgreSQL; `authorized_user` and `access_log` tables.

```sql
-- Users who accessed data but are not on the authorized list.
SELECT DISTINCT a.user_id, a.first_access, COUNT(*) AS events
FROM access_log a
LEFT JOIN authorized_user u USING (user_id)
WHERE u.user_id IS NULL
GROUP BY a.user_id, a.first_access
ORDER BY events DESC;
```

Expected output for a compliant environment: zero rows. A non-empty result is an access-control finding, and it is the most common audit finding in research environments.

### 7.4 A hand calculation for an interview

Given a license fee of $375,000 and a 14-week latency, with a monthly value of time of $180,000:

1. Months = 14 / 4.33 = 3.23.
2. Delay cost = 3.23 × 180,000 = $581,400.
3. TCO = 375,000 + 581,400 = $956,400 for the license and delay alone.
4. If legal and compute add $180,700, the TCO is $1,137,100.

The point of the calculation is that the delay cost can exceed the license fee, which is invisible if procurement compares only fees.

---

## 8. Data, standards, and regulatory context

### 8.1 The US framework

| Instrument | Citation | Scope | Key requirement |
|---|---|---|---|
| HIPAA Privacy Rule | 45 CFR Part 160 and Subparts A and E of Part 164 | Use and disclosure of PHI | Permitted uses, minimum necessary, de-identification |
| HIPAA Security Rule | 45 CFR Part 164 Subpart C | Electronic PHI | Administrative, physical, and technical safeguards |
| HIPAA Breach Notification | 45 CFR Part 164 Subpart D | Breaches of unsecured PHI | Notification within 60 days |
| Common Rule | 45 CFR Part 46 | Human-subjects research | IRB review, informed consent, continuing review |
| FDA human-subjects regulations | 21 CFR Parts 50 and 56 | FDA-regulated research | Informed consent, IRB |
| 42 CFR Part 2 | Substance use disorder records | Federally assisted programs | Written consent for most disclosures |
| GINA | Public Law 110-233 | Genetic information | Prohibits health insurance and employment discrimination |
| Information blocking | 45 CFR Part 171 | Access to electronic health information | Prohibits unreasonable interference |

### 8.2 De-identification under HIPAA

Safe Harbor at 45 CFR 164.514(b)(2) removes 18 categories. Expert Determination at 164.514(b)(1) requires a documented expert analysis. The Limited Data Set at 164.514(e) removes 16 direct identifiers but permits dates and five-digit ZIP codes, with a DUA.

Two points that are frequently missed:

1. **Safe Harbor is not a guarantee.** The actual knowledge provision means a dataset can fail even with all 18 removed.
2. **Free text and images are outside the list.** Structured de-identification does not touch names in notes or text in pixels.

> **Note:** HHS guidance on de-identification was last reviewed in February 2025 and derives from the 2012 guidance issued under HITECH. No 2024 rule changed the standard itself; the 2024 reproductive health privacy rule addressed a different part of the Privacy Rule.

### 8.3 The EU framework

- **GDPR (Regulation 2016/679).** Health data is special-category data under Article 9 and requires an Article 9(2) condition in addition to an Article 6 lawful basis. Article 35 requires a data protection impact assessment for high-risk processing. Articles 44 to 49 govern transfers outside the EU.
- **EHDS (Regulation 2025/327).** Published 5 March 2025 and in force from 26 March 2025. It generally applies from 26 March 2027; Chapter IV on secondary use applies from 26 March 2029, with specific provisions staged to 2031 (including genetic and genomic data) and 2035 (third-country access to HealthData@EU).

The EHDS secondary-use model requires access through health data access bodies under a data permit, processing in secure processing environments, and use limited to enumerated purposes. Article 54 prohibits uses such as marketing, insurance or credit exclusion, and decisions detrimental to individuals. There is a right to opt out, which national law may override for public-interest research.

For an analyst, the practical consequence is that EU access will increasingly run inside approved environments rather than on downloaded files, and the analysis code must be portable to that environment.

### 8.4 The access tiers of major programmes

| Programme | Tier structure | Data | Notes |
|---|---|---|---|
| All of Us | Public, Registered, Controlled | Surveys, EHR, measurements, wearables, genomics | Controlled adds genomics, unsuppressed demographics, real dates |
| dbGaP | Open aggregate, controlled individual-level | Genotypes and phenotypes | Data access committee approval |
| UK Biobank | Registered, with approved project | Questionnaires, imaging, genomics, linked records | Fee and institutional agreement |
| MIMIC-IV | Credentialed | Critical care EHR | Training and DUA; no redistribution |
| CMS Medicare | ResDAC request | Claims | DUA, protocol, fee |
| TCIA | Public, some controlled | Cancer imaging | Per-collection license |

All of Us is a useful reference model because its three tiers are explicitly documented. Its September 2024 release increased the Workbench to about 633,000 participants, including more than 400,000 with whole genome sequences, and its February 2025 release included 447,278 genotyping arrays, 414,830 short-read whole genomes, 97,061 structural variant calls, and 2,800 long-read genomes. The Controlled Tier adds genomics, previously suppressed demographic subcategories, and real event dates, under a Data Use and Registration Agreement.

The lesson from that structure is that the same programme can support very different analyses at different tiers, and the tier is a choice about both risk and analytic power.

### 8.5 What is binding and what is convention

Binding:

- HIPAA for covered entities and business associates.
- The Common Rule for federally funded human-subjects research.
- GDPR for EU data subjects.
- Contract terms in every DUA and license.

Convention:

- Requiring a DUA for a Limited Data Set is a legal requirement, not a convention.
- Requiring named users is common contractual practice and not a legal rule.
- Requiring destruction at the end of the term is contractual, and the specific period varies.

### 8.6 Penalties and enforcement

HIPAA civil penalties are tiered by culpability and are adjusted for inflation; criminal penalties apply for knowing violations. GDPR fines can reach 20 million euros or 4 percent of global annual turnover, whichever is higher, for the most serious infringements.

The enforcement risk that matters most in practice is contractual rather than regulatory: a data owner can terminate access, demand destruction, and pursue damages. For a company whose product depends on a licensed dataset, termination is an existential risk, and it is triggered by a breach of the permitted use rather than by a regulator's finding.

---

## 9. Industry practice and case evidence

### 9.1 How access timelines actually run

Industry experience with data access is consistent across source types:

| Source type | Typical timeline | Dominant delay |
|---|---|---|
| Public dataset download | Hours to days | None |
| Credentialed dataset | 1 to 4 weeks | Credentialing and training |
| Registered access programme | 2 to 6 weeks | Institutional agreement |
| Controlled access (dbGaP, EGA) | 2 to 6 months | Data access committee review |
| Commercial license | 6 to 16 weeks | Legal negotiation |
| Institutional EHR | 3 to 9 months | IRB, DUA, security review |
| Prospective collection | 9 to 36 months | Protocol, ethics, enrollment |

The pattern is that the delay is legal and organizational, not technical. A team that plans for the technical work and not the legal work will be late by months.

### 9.2 The BAA misunderstanding

A frequent error is to assume that a business associate agreement permits a vendor to use PHI for its own purposes. It does not. A BAA permits the vendor to process PHI on behalf of the covered entity for the covered entity's purposes.

A vendor that wants to build a product from the data needs either a DUA that permits that use or de-identified data that falls outside PHI. The distinction is the difference between a service relationship and a data partnership, and it is often discovered when the product team asks for the data.

### 9.3 Data clean rooms and the shift to analysis-in-place

A data clean room is an environment where an analyst can run queries against another party's data without receiving the raw records. The model is increasingly used for payer-provider and pharma-payer collaborations, because it allows analysis without transferring identifiable data.

The trade-off is analytical: the analyst works through an approved query interface, which constrains the methods and the iteration. A clean room that permits only SQL aggregates cannot support a custom model, and one that permits arbitrary code requires a security review of that code.

### 9.4 Differential privacy in practice

Differential privacy provides a formal guarantee for aggregate releases by adding calibrated noise. It is used by census agencies and by some large technology platforms, and it is increasingly discussed for health data.

The practical constraint is the privacy budget. A small budget protects privacy and limits the number of queries; a large budget permits more analysis and weakens the guarantee. For a research programme with many exploratory queries, the budget is exhausted quickly, and the choice between privacy and utility becomes explicit rather than implicit.

### 9.5 Synthetic data: what it is and is not

Synthetic data generated from a real dataset can preserve statistical structure while containing no real records. It is useful for pipeline development, for software testing, and for sharing a demonstration dataset.

It is not a substitute for real data in a validation study. A model trained on synthetic data learns the generator's assumptions, and a model validated on synthetic data is validated against those assumptions rather than against patients. The published guidance is consistent: synthetic data can support development and cannot support a clinical or regulatory claim on its own.

### 9.6 What a mature access function looks like

- **A data access catalogue** listing sources, tiers, typical latency, and cost.
- **A standard DUA template** with pre-approved clauses and escalation paths.
- **A security review process** aligned to the contract's security exhibit.
- **A named owner** for each agreement, with a renewal calendar.
- **A destruction process** that covers derived data.

Organizations with these five have access latencies measured in weeks. Organizations without them measure in quarters, and the difference compounds across a portfolio of studies.

### 9.7 Why access is a competitive advantage

Two companies with the same modeling capability will diverge based on their ability to obtain and use data lawfully and quickly. The advantage is not the data itself, because the same commercial datasets are available to competitors. It is the process: the templates, the relationships, the security posture, and the institutional knowledge of which clauses matter.

That process is slow to build and hard to copy, which is why it is a durable advantage in a way that a model architecture is not.

### 9.8 The buyer's diligence checklist

Six questions separate a usable license from a restrictive one, and all six can be answered in a first call:

1. What is the de-identification category, and are dates retained?
2. What is the permitted use, in the contract's own words?
3. Is linkage to other data permitted, and under what conditions?
4. What is the security exhibit, and does it require a specific region or certification?
5. What is the publication review process, and what is the review window?
6. What must be destroyed at the end of the term, including derived data?

A vendor that answers all six in writing is selling a data product. One that defers them to legal is selling a term sheet, and the term sheet is where the delays live.

### 9.9 Common myths about access

- **"De-identified data can be shared freely."** It can be shared under the terms that apply, which may still restrict use and redistribution.
- **"A BAA covers everything."** It covers processing on behalf of the covered entity, and nothing else.
- **"Public data has no restrictions."** Most public datasets carry attribution or use terms, and some carry redistribution limits.
- **"We can link later."** Linkage is usually prohibited unless explicitly permitted.
- **"Synthetic data is a substitute."** It is a development tool.
- **"Legal review is a formality."** It is the step that determines whether the conclusion is publishable.

Each myth saves a week at the start and costs a quarter at the end.

The pattern behind all six is the same: a constraint is discovered after a commitment is made. A short diligence conversation before commitment is the cheapest risk reduction available.

### 9.10 What to do in the first week of a data project

- Confirm the decision, the population, and the deadline in writing.
- List three candidate sources with their tiers.
- Read the standard contract for each candidate, or request it.
- Identify the de-identification category each offers.
- Estimate latency from a reference customer, not from a sales deck.
- Build the TCO model before the first vendor call.

A week of diligence at the start is the highest-return activity in a data project, and it is the one most often skipped because the team is eager to start modeling.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Tier mismatch | Public data chosen for a controlled-tier question | Convenience over fitness | Analysis cannot answer | Match tier to question |
| Latency ignored | Timeline assumes delivery on request | Legal and security steps unplanned | Missed decision window | Plan backward from the deadline |
| BAA treated as research permission | Vendor plans product use | Misunderstood instrument | Contract breach | Read the permitted use |
| Safe Harbor destroys dates | Longitudinal analysis impossible | Dates reduced to year | Study infeasible | Use Limited Data Set |
| Free text not de-identified | Names in notes | Structured rules only | Privacy breach | Text de-identification |
| Re-identification prohibited | Linkage planned but not permitted | Standard clause | Linkage is a breach | Obtain written permission |
| Storage region violated | Cloud default region | Contract requires in-country | Breach and migration | Design the environment to the contract |
| Unnamed users | Analyst added ad hoc | User list clause | Access-control finding | Formal substitution process |
| Derived data not destroyed | Outputs retained after term | Destruction clause covers derivatives | Breach | Include derivatives in the plan |
| Publication review missed | Manuscript submitted without review | Review right in contract | Delay or dispute | Submit early |
| Synthetic data used for validation | Model validated only on synthetic | Generator assumptions | Invalid claim | Validate on real data |
| Penalty underestimated | Only license fee budgeted | No delay or legal cost | Wrong pathway chosen | TCO model |

### 10.1 The three red flags

1. **A data request with no stated deadline and no value of time.** Without these, the analysis cannot choose between pathways.
2. **A DUA that has been read only by procurement.** The clauses that constrain analysis are legal and technical, and procurement is not looking for them.
3. **A de-identification plan that addresses structured fields only.** Free text and images are where the residual risk lives.

### 10.2 The "we will just de-identify it" trap

De-identification is often proposed as a way to avoid the access process. It does not, for three reasons:

- The data owner must perform or approve the de-identification, so the conversation still happens.
- De-identified data may still be personal data under GDPR.
- The de-identification may destroy the analytic value that motivated the request.

The proposal is not wrong; it is incomplete. It is one option in a negotiation, not a way around one.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| A source exists that can answer the question | Feasibility | No source contains the outcome | Landscape review |
| The access timeline fits the decision | Timeline planning | Legal or security delay | Reference customers; buffer |
| The permitted use covers the analysis | Compliance | Clause excludes the use | Clause-by-clause mapping |
| The de-identification leaves the needed fields | Analytic feasibility | Dates or geography removed | Data dictionary review |
| The environment meets the contract | Compliance | Region or control mismatch | Security review before transfer |
| The user list can be maintained | Access continuity | Staff turnover | Substitution process |
| The value of time estimate is defensible | TCO comparison | Overstated or understated | Sensitivity analysis |
| Destruction is feasible | End-of-term compliance | Derivatives dispersed | Inventory of derivatives |

### 11.1 Limits that are inherent

- **Access is a negotiation with another party.** You cannot unilaterally decide the timeline.
- **De-identification reduces but does not eliminate risk.** Residual risk is managed contractually.
- **Contracts constrain conclusions.** A publication review right can delay or shape what is published.
- **Legal frameworks differ by jurisdiction.** A US approval does not travel.
- **Synthetic data cannot validate a clinical claim.** It can develop a pipeline.
- **Time cannot be recovered.** A missed decision window is a permanent loss.

### 11.2 How to state the access limitation

"The analysis used a commercial claims license covering 2 million member-years, delivered under a Limited Data Set agreement that retained dates and five-digit ZIP codes and prohibited re-identification or linkage. Delivery took 14 weeks from signature. Because the agreement prohibits linkage, the analysis cannot adjust for clinical severity, and because the permitted use is internal research, the results cannot be used in a promotional context without a separate agreement."

That paragraph gives the reader the data, the restrictions, the timeline, and the two conclusions the contract prevents.

### 11.3 The limit that changes the business case

The access limitation is not only an analytic caveat. It can change whether the product is viable.

If the permitted use excludes promotional claims, a marketing use case is not supported, regardless of the analysis quality. If the security exhibit requires an on-premises environment, the cloud cost model changes. If the term is two years, a longitudinal product cannot depend on the data beyond that date.

These are business constraints discovered from the contract, and they belong in the business case, not in a limitations appendix.

### 11.4 What to do when the contract and the analysis conflict

When a planned analysis is not permitted, three options exist, in order of preference:

1. **Redesign the analysis** to fit the permitted use, if the decision can still be supported.
2. **Renegotiate** for an expanded permitted use, with a realistic timeline.
3. **Change the source** to one whose terms fit the analysis.

The wrong option is to proceed and hope, because the data owner's audit right exists precisely to detect that, and the consequence is termination rather than a warning.

---

## 12. Exercises

### Exercise 1 — Classify the tiers (recall, 20 minutes)

Place eight sources into the access tiers in section 1.2 and state the requirement for each.

Deliverable: an eight-row table.

Acceptance criteria: each tier is correct; each requirement is specific.

### Exercise 2 — DUA clause mapping (application, 45 minutes)

Take a public DUA template and map five planned analyses to the permitted use clause. Identify any analysis with no match.

Deliverable: a five-row mapping table with gaps flagged.

Acceptance criteria: each analysis is mapped to specific wording; gaps are stated as gaps, not resolved by inference.

### Exercise 3 — De-identification choice (application, 40 minutes)

For a study requiring 30-day readmission, 90-day mortality, and geographic variation, choose between Safe Harbor, Limited Data Set, and Expert Determination. Justify.

Deliverable: a decision with the fields required and the residual risk.

Acceptance criteria: the choice retains the fields the analysis needs; the residual risk is stated.

### Exercise 4 — TCO (calculation, 45 minutes)

Using the costs in section 6.2, recompute the TCO with a monthly value of time of $90,000. Report whether the ranking changes.

Deliverable: a three-row table and a conclusion.

Acceptance criteria: arithmetic correct; the conclusion states which pathway becomes preferable and why.

### Exercise 5 — k-anonymity (calculation, 40 minutes)

For 250,000 records with age band (12), sex (2), ZIP3 (800), and 25 diagnosis categories, compute the possible combinations and the mean cell size. Interpret.

Deliverable: the calculation and a two-sentence interpretation.

Acceptance criteria: arithmetic correct; the interpretation states what generalization would be needed for k=5.

### Exercise 6 — Contract critique (critique, 45 minutes)

Given a DUA excerpt, identify three clauses that constrain the analysis and explain the consequence of each.

Deliverable: a three-row table with clause, constraint, and consequence.

Acceptance criteria: each constraint is tied to a specific analytic decision; no legal advice is offered.

### Exercise 7 — Access memo (design, 60 minutes)

Write a one-page access memo for a stated decision: tier, source, legal basis, contract terms, environment, timeline, and cost.

Deliverable: a one-page memo.

Acceptance criteria: a legal reviewer and a finance reviewer could both act on it; the timeline is backward-planned from the deadline.

### Exercise 8 — GDPR versus HIPAA (critique, 40 minutes)

Compare the requirements for a US and an EU deployment of the same analysis on the same data types.

Deliverable: a four-row comparison and a recommendation.

Acceptance criteria: the difference in lawful basis is explained; the transfer mechanism is named.

### Exercise 9 — Synthetic data decision (design, 30 minutes)

For three uses (pipeline testing, model development, regulatory validation), decide whether synthetic data is acceptable.

Deliverable: a three-row table with justification.

Acceptance criteria: the regulatory use is rejected with a reason; the development use is bounded.

### Exercise 10 — Vendor diligence (design, 45 minutes)

Using the six questions in section 9.8, write a diligence memo for a vendor that answered three and deferred three.

Deliverable: a one-page memo with a recommendation.

Acceptance criteria: the deferred answers are named as risks; the recommendation states what would change it.

---

## 13. Self-check answers

**A1. What is the difference between a BAA and a DUA?**
A BAA extends HIPAA obligations to a business associate processing PHI on behalf of a covered entity. A DUA defines the permitted use and restrictions for data provided to a recipient. A BAA does not permit the vendor's own use, and a DUA does not create a lawful basis.

**A2. What are the three HIPAA de-identification categories?**
Safe Harbor (18 identifiers removed), Expert Determination (documented expert analysis), and the Limited Data Set (16 direct identifiers removed, dates and five-digit ZIP retained, with a DUA).

**A3. Why might a Limited Data Set be preferable to Safe Harbor?**
Because it retains dates and geography, which are required for longitudinal and geographic analysis. Safe Harbor reduces dates to year, which makes interval-based outcomes such as 30-day readmission impossible.

**A4. Compute the possible combinations for 16 age bands, 2 sexes, 900 ZIP3 codes, and 40 diagnosis categories.**
16 × 2 × 900 × 40 = 1,152,000.

**A5. Compute the mean cell size for 500,000 records in 1,152,000 combinations.**
500,000 / 1,152,000 = 0.43. An average below 1 implies many singleton cells and therefore low k-anonymity.

**A6. Compute the delay cost for 14 weeks at $180,000 per month.**
14 / 4.333 = 3.231 months. 3.231 × 180,000 = $581,580.

**A7. Why can the cheapest license be the most expensive pathway?**
Because latency carries a cost. A cheaper license with a longer negotiation adds delay cost that can exceed the fee difference, and a deadline can eliminate it entirely.

**A8. What is the GDPR lawful basis for health data research?**
Article 6 requires a lawful basis and Article 9 requires an additional condition for special-category data, such as explicit consent or a research-related condition under Article 9(2)(j) with safeguards. The specific choice depends on national law.

**A9. When does the EHDS secondary-use chapter apply?**
Chapter IV applies from 26 March 2029, with specific provisions staged to 2031 and 2035.

**A10. Why is synthetic data insufficient for regulatory validation?**
Because a model validated on synthetic data is validated against the generator's assumptions rather than against patients. Synthetic data can support development and testing, not a clinical claim.

**A11. What does a publication review clause constrain?**
It can delay submission and, depending on wording, permit the data owner to comment on or approve content. That constrains the timing and potentially the conclusions of the publication.

**A12. Rubric for exercise 7 (access memo).**
A strong memo states the decision and deadline, the tier and source, the legal basis, the contract terms that constrain analysis, the environment, a backward-planned timeline with named dependencies, and a TCO that includes delay. A weak memo states the source and the fee.

---

## 14. Interview questions

**Q1. How would you choose between a public dataset and a licensed one?**
A strong answer: match the source to the decision, then compare on fitness, timeline, and TCO. A public dataset can be right when the question is methodological or the population is adequate; a licensed dataset is right when coverage or recency is required. Cost alone is not the criterion.

**Q2. What clauses in a DUA most often constrain analysis?**
Permitted use, re-identification and linkage, storage location, user list, publication review, and destruction of derived data. Weak answers mention only confidentiality.

**Q3. Explain the difference between de-identification and anonymization.**
HIPAA de-identification is a legal standard with residual risk; GDPR anonymization requires that re-identification is not reasonably likely by any means. A HIPAA de-identified dataset can still be personal data under GDPR.

**Q4. Why is a BAA not enough to build a product on hospital data?**
Because a BAA permits processing on behalf of the covered entity, not the vendor's own use. A product requires a DUA that permits that use, or de-identified data outside PHI.

**Q5. How would you estimate the cost of a delay?**
A strong answer: estimate the monthly value of the decision, multiply by the delay in months, and present it as a labelled assumption with a sensitivity range. Weak answers ignore delay entirely.

**Q6. What is k-anonymity and what does it tell you?**
A property where each record is indistinguishable from at least k−1 others on the quasi-identifiers. It tells you how much generalization is needed, and it does not guarantee anonymity against an adversary with auxiliary data.

**Q7. What is a data clean room?**
An environment where analysis runs against another party's data without transferring raw records. It reduces transfer risk and constrains methods to what the environment permits.

**Q8. How do US and EU requirements differ for a genomic study?**
A strong answer: HIPAA plus the NIH Genomic Data Sharing Policy and GINA in the US; GDPR special-category data with an Article 9 condition, a DPIA, and a transfer mechanism in the EU. The EHDS adds a secondary-use permit regime from 2029.

**Q9. What would make you reject an access pathway?**
A strong answer: a deadline it cannot meet, a permitted use that excludes the analysis, a de-identification category that removes required fields, or a security requirement the organization cannot satisfy.

**Q10. How do you handle a publication review clause?**
A strong answer: negotiate a fixed review window and a scope limited to factual accuracy and confidentiality, then plan the submission timeline around it. Weak answers accept an open-ended review.

**Q11. What is the risk of using synthetic data for development?**
The model learns the generator's structure, which may not match the real data. It is acceptable for pipeline testing and must be followed by training and validation on real data.

**Q12. What does a mature access function have that an immature one does not?**
A catalogue with latency and cost, a standard DUA with pre-approved clauses, a security process aligned to the contract, a named owner per agreement, and a destruction process that covers derived data.

**Q13. A vendor offers the data "de-identified, no restrictions." What do you check?**
A strong answer: what de-identification standard, whether dates are retained, whether the data are still personal data under GDPR, and whether the vendor has the right to license them. "No restrictions" often means the vendor has not thought about linkage or redistribution, and the contract will be silent where it should be specific.

**Q14. Why is a data access catalogue a strategic asset?**
Because it converts a bespoke negotiation into a repeatable decision. A team with a catalogue can answer "what can we get, how long, how much" in an hour, which is the difference between bidding for a study and missing it.

---

## 15. Key terms

- **Access tier** — a level of data availability with requirements.
- **BAA** — business associate agreement.
- **Business associate** — a vendor processing PHI for a covered entity.
- **Common Rule** — US human-subjects research regulation.
- **Controller** — the entity determining processing purposes.
- **Covered entity** — a provider, plan, or clearinghouse under HIPAA.
- **Data clean room** — a controlled environment for joint analysis.
- **Data permit** — an EHDS authorization for secondary use.
- **De-identification** — reducing identifying information.
- **Differential privacy** — a formal privacy guarantee for aggregates.
- **DPIA** — data protection impact assessment.
- **DUA** — data use agreement.
- **EHDS** — European Health Data Space.
- **Expert Determination** — a HIPAA de-identification method.
- **GDPR** — EU General Data Protection Regulation.
- **GINA** — Genetic Information Nondiscrimination Act.
- **Health data access body** — an EHDS body granting permits.
- **HIPAA** — Health Insurance Portability and Accountability Act.
- **IRB** — institutional review board.
- **k-anonymity** — indistinguishability from k−1 others.
- **Lawful basis** — the legal ground for processing.
- **Limited Data Set** — a HIPAA category retaining dates and geography.
- **MTA** — material transfer agreement.
- **Part 2** — the US substance use disorder confidentiality regulation.
- **Permitted use** — the uses a contract allows.
- **PHI** — protected health information.
- **Processor** — an entity processing on behalf of a controller.
- **Pseudonymization** — replacing identifiers with consistent pseudonyms.
- **Re-identification** — recovering identity from de-identified data.
- **Redistribution** — passing data to a third party.
- **Safe Harbor** — a HIPAA de-identification method.
- **SCC** — standard contractual clauses.
- **Secure processing environment** — a controlled analysis environment.
- **Special category data** — health and genetic data under GDPR.
- **Synthetic data** — artificially generated data.
- **Tokenization** — replacing an identifier with a token.

---

## 16. Sources

All URLs accessed 24 October 2026.

**US regulators**

- HHS Office for Civil Rights. *Guidance Regarding Methods for De-identification of Protected Health Information in Accordance with the HIPAA Privacy Rule* (content last reviewed 3 February 2025). https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- eCFR. 45 CFR 164.514, *Other requirements relating to uses and disclosures of protected health information.* https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
- eCFR. 45 CFR Part 46, *Protection of Human Subjects.* (Cite the Part.)
- eCFR. 42 CFR Part 2, *Confidentiality of Substance Use Disorder Patient Records.* (Cite the Part.)
- eCFR. 45 CFR Part 171, *Information Blocking.* (Cite the Part.)
- US Congress. *Genetic Information Nondiscrimination Act of 2008*, Public Law 110-233.

**EU**

- European Union. *Regulation (EU) 2016/679 (General Data Protection Regulation).* https://eur-lex.europa.eu/eli/reg/2016/679/oj
- European Union. *Regulation (EU) 2025/327 on the European Health Data Space.* https://eur-lex.europa.eu/eli/reg/2025/327/oj/eng
- European Commission. *European Health Data Space Regulation (EHDS).* https://health.ec.europa.eu/ehealth-digital-health-and-care/european-health-data-space-regulation-ehds_en
- European Commission. *Reuse of health data.* https://health.ec.europa.eu/ehealth-digital-health-and-care/reuse-health-data_en

**Programmes and data access**

- All of Us Research Program. *Data Access Tiers.* https://www.researchallofus.org/data-tools/data-access/
- All of Us Research Program. *Researcher Workbench.* https://www.researchallofus.org/data-tools/workbench/
- NIH. *All of Us Research Program year in review: 2024.* https://pmc.ncbi.nlm.nih.gov/articles/PMC12461012/
- UK Biobank. https://www.ukbiobank.ac.uk/
- dbGaP. https://www.ncbi.nlm.nih.gov/gap/
- PhysioNet. *MIMIC-IV v3.1.* https://physionet.org/content/mimiciv/3.1/
- CDC. *NNDSS Data in CDC WONDER.* https://wonder.cdc.gov/nndss.html

**Background**

- HHS. *Summary of the HIPAA Privacy Rule.* https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
- GA4GH. *Data Security Infrastructure Policy (DSIP) v4.0* (2019). https://www.ga4gh.org/product/data-security-infrastructure-policy/

> **Note:** This section summarizes legal instruments for teaching. It is not legal advice, and the specific application of any instrument depends on facts and jurisdiction. De-identification and cross-border transfer decisions require counsel.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md)
- Next topic: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Related sibling: [`01_ehr-data.md`](01_ehr-data.md)
- Related sibling: [`02_claims-and-billing.md`](02_claims-and-billing.md)
- Related sibling: [`06_omics-data.md`](06_omics-data.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.4 in the survey file (source-selection matrix) uses the access tier comparison in section 1.2 as one of its inputs.

Related reading: [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md) for the terminology licensing summarized in section 8.5 of that topic; [`06_omics-data.md`](06_omics-data.md) for controlled-access genomic data; and [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md) for the quality dimensions that a restricted access pathway can compromise.

The sentence to remember: the data you can get, lawfully, in time, under terms that permit the conclusion, is the only data that matters.

The final topic turns to the question that survives all the access work: once you have the data, how do you know whether it is any good?