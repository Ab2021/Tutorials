# Privacy Principles

> **Module 16 - Privacy, Security, and Ethics**
> Topic 01 of 10
> Estimated study time: 3 hours
> Prerequisites: Modules 02, 03
> Version 2026.3

## 1. Why this topic exists

Every biomedical data system makes the same four decisions, whether or not anyone records them:
what data is collected, why it may be used, who may see it, and when it must be destroyed. When
those decisions are made implicitly, they are made by whoever wrote the query, configured the
bucket, or copied the extract to a laptop. Privacy principles exist to force the four decisions
into the open, attach them to an owner, and leave evidence that they were followed.

The operational cost of skipping this step is not abstract. A hospital analytics team that
receives a request for "all diabetic patients, plus notes, for a vendor pilot" has no way to
answer it without a purpose statement, a field-level justification, and a legal basis. In practice
the team does one of three things: sends everything because filtering takes time, refuses
everything because review takes time, or sends something and cannot explain afterwards why it was
allowed. Each outcome is expensive. The first creates a breach surface. The second blocks a
legitimate programme. The third fails an audit even when the data flow was defensible.

Three failure patterns recur.

**Purpose creep.** A dataset collected for quality improvement is reused for commercial product
development, then for marketing, then for a foundation-model training corpus. Each step is small
and each has a plausible internal justification. The cumulative drift is the problem: the original
notice, consent, and data use agreement described none of the later purposes. Regulators treat
this as a change of purpose requiring a new basis, not an implementation detail. The engineering
signal is that nobody can name the purpose of a table without reading its name.

**Minimisation by default is nobody's default.** Systems are built to be useful later, so they
retain every column and every raw payload. A message queue keeps full HL7 (Health Level Seven)
messages because parsing is hard. A data lake keeps identified source tables because joins are
easier. A notebook caches a dataframe with names and medical record numbers (MRNs) because
re-running the query is slow. None of these decisions is malicious; all of them widen the set of
records that a single compromised credential exposes. Minimum necessary is a design constraint on
the pipeline, not a promise made in a policy document.

**Consent treated as a signature rather than a scope.** A consent form says "research" and the
organisation reads that as permission for anything a researcher later wants to do, including
commercial partnerships, model training, and data sale. In the United States, HIPAA (Health
Insurance Portability and Accountability Act) authorisation is purpose-specific and revocable. In
the European Union, consent under the General Data Protection Regulation (GDPR) must be freely
given, specific, informed, and unambiguous, and must be as easy to withdraw as to give. A consent
record that stores only a boolean has already lost the information needed to honour it.

These patterns matter commercially as well as legally. Procurement questionnaires from health
systems, payers, and pharmaceutical sponsors now ask for a data map, a lawful-basis statement, a
retention schedule, and evidence of access review. A vendor that cannot produce them loses the
deal before any model is evaluated. Privacy principles are the vocabulary in which those answers
are written.

This topic builds the operating model that the rest of the module assumes: a purpose envelope
around every dataset, a field-level justification for every column that leaves its source system,
a legal basis recorded per purpose and per population, and an evidence trail that survives staff
turnover. De-identification, threat modelling, access control, and breach response all consume the
artefacts produced here. Without them, those later topics have nothing to protect and no scale
against which to judge whether protection is adequate.

> **Note:** Privacy principles are not a compliance checklist. A programme can satisfy every
> disclosure requirement and still over-collect, over-retain, and over-share. The test is whether
> each control can be tied to a specific harm it reduces.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a legal requirement, a security best practice, and a statistical guarantee, and
   classify each control in a privacy programme into one of those categories.
2. **Write** a purpose statement for a dataset that names the decision it supports, the population,
   the minimum fields, the retention period, and the permitted recipients.
3. **Apply** the HIPAA minimum necessary standard to a request, including the six categories of
   exception, and document the determination.
4. **Select** a legal basis under GDPR for a described processing activity, separating Article 6
   bases from Article 9 conditions for health data.
5. **Compare** consent, authorisation, waiver of authorisation, and broad consent, and choose the
   correct route for a given research or product scenario.
6. **Quantify** over-collection and residual exposure for a data extract using record-field counts
   and a risk-weighted exposure score.
7. **Handle** the edge cases that break naive policies: minors, deceased patients, withdrawal,
   free-text identifiers, metadata, small cells, and cross-border transfer.
8. **Critique** a data use agreement or consent form against the purpose envelope it is supposed
   to enforce.

## 3. The core mental model

Privacy principles are usually taught as a list. A list is hard to operate. The model that makes
them operable is a nested envelope: an outer legal layer that says whether processing is allowed
at all, a middle purpose layer that says what the data may be used for, and an inner data layer
that says which fields, for how long, and to whom.

```text
+---------------------------------------------------------------+
|  LEGAL LAYER                                                  |
|  Is there a lawful basis?  Consent | contract | legal duty |  |
|  public interest | vital interests | legitimate interests     |
|  Health data adds a second condition (GDPR Art. 9).           |
+---------------------------------------------------------------+
        |  authorises a stated purpose, not data in general
        v
+---------------------------------------------------------------+
|  PURPOSE LAYER                                                |
|  Purpose statement: decision, population, horizon, recipients |
|  Purpose limitation: new purpose -> new assessment, new basis |
|  Consent/authorisation scope is stored, not inferred          |
+---------------------------------------------------------------+
        |  defines the minimum data needed for that purpose
        v
+---------------------------------------------------------------+
|  DATA LAYER                                                   |
|  Field list with justification | granularity | retention     |
|  Access roles | recipients | transfer mechanism | deletion    |
+---------------------------------------------------------------+
        |
        v
+---------------------------------------------------------------+
|  EVIDENCE LAYER                                               |
|  Data map | DPIA | DUA | access logs | retention job output   |
|  Review dates | incident record | withdrawal propagation     |
+---------------------------------------------------------------+
```

Read the diagram from the outside in. The legal layer answers "may we". The purpose layer answers
"for what". The data layer answers "how much, how long, to whom". The evidence layer answers "how
do you know". A control that cannot be placed in one of these layers is usually decorative.

The causal flow matters. Purpose determines the field list, not the other way round. A team that
starts from an existing table and asks "what may we do with this" has inverted the model and will
always over-retain, because every column has some conceivable use. A team that starts from a
decision and asks "what is the least data that supports it" produces a smaller, cheaper, and more
defensible system. The second team also builds better models, because fields that cannot be
justified are usually proxies for something the model should not learn.

Three properties of the model deserve emphasis.

**Principles are constraints on design, not on intentions.** "We only use the data for good
purposes" is not a control. "The extract contains nine fields, each mapped to a justification in
the data dictionary, and the pipeline drops three fields that the purpose does not require" is a
control. The difference is testability.

**Purpose limitation is temporal.** A purpose statement has a review date. Populations change,
programmes end, vendors are replaced, and legislation shifts. An extract that was minimal in 2024
may be excessive in 2026 because the outreach programme ended and the table stayed. Retention is
part of purpose, not an afterthought.

**Consent is one basis among several, and often the weakest.** Consent is appropriate when the
individual genuinely has a choice and the processing is not necessary for care. It is a poor fit
for treatment, for statutory reporting, and for many secondary uses where refusal would be
impractical or coercive. Choosing consent by reflex produces two harms: it burdens patients with
decisions they cannot meaningfully make, and it creates a consent record that cannot support the
uses the organisation actually performs. The correct question is not "do we have consent" but
"which basis fits this processing, and what does it require us to do".

A useful discipline is to write the harm statement first. For a given dataset, name the person or
group who could be harmed if the data were misused, what the harm is, and which principle reduces
it. If no harm can be named, the control is probably unnecessary. If a harm can be named but no
control addresses it, the programme has a gap. This is the same structure used in the threat
model in Topic 04 and the impact assessment in Topic 08, which is why the vocabulary is introduced
here.

## 4. Definitions and vocabulary

The table below defines every term used later in this file. The "common misuse" column is the part
that matters in review: most privacy defects are vocabulary defects, where two teams use the same
word for different obligations.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Personal data | Information relating to an identified or identifiable natural person, including identifiers held indirectly (GDPR Art. 4(1)). | Treated as a synonym for "name and address". | Legal category; no unit. |
| Identifiability | The degree to which a person can be singled out, linked, or inferred from data, in the hands of a stated actor. | Treated as a property of the data alone, ignoring who holds auxiliary information. | Assessed per adversary model. |
| PHI | Protected health information: individually identifiable health information held or transmitted by a covered entity or business associate (45 CFR 160.103). | Applied to de-identified data, which is outside PHI by definition. | US legal category. |
| IIHI | Individually identifiable health information: health information created or received by a covered entity that relates to past, present, or future physical or mental health, care, or payment, and identifies the individual. | Confused with the broader "health data" that includes consumer app data. | US legal category. |
| Covered entity | A health plan, health care clearinghouse, or health care provider that transmits health information electronically in connection with a transaction covered by HIPAA. | Assumed to cover any company handling health data. | US legal category. |
| Business associate | A person or entity that performs functions on behalf of, or provides services to, a covered entity involving PHI. | Treated as equivalent to a covered entity; obligations differ. | US legal category. |
| Minimum necessary | The requirement to limit uses, disclosures, and requests for PHI to the least needed for the purpose (45 CFR 164.502(b), 164.514(d)). | Applied to treatment disclosures, which are exempt. | Legal standard; documented determination. |
| Purpose limitation | Data collected for specified, explicit, legitimate purposes may not be further processed in an incompatible manner (GDPR Art. 5(1)(b)). | Read as "the purpose must be written down once". | Legal principle; compatibility assessment. |
| Purpose statement | An operational artefact naming the decision, population, fields, recipients, retention, and review date for a dataset. | Replaced by a project title. | Internal control; versioned document. |
| Legal basis | The Article 6 condition that makes processing lawful; health data additionally requires an Article 9(2) condition. | Conflated with consent alone. | GDPR legal category. |
| Special category data | Personal data revealing racial or ethnic origin, political opinions, religious beliefs, trade union membership, genetic data, biometric data for identification, health data, sex life or sexual orientation (GDPR Art. 9(1)). | Applied to all data about a patient, including billing. | GDPR legal category. |
| Consent (GDPR) | A freely given, specific, informed, and unambiguous indication of wishes by statement or clear affirmative action (Art. 4(11), Art. 7). | Collected as a pre-ticked box or bundled with terms. | Legal basis; must be demonstrable and withdrawable. |
| Authorisation (HIPAA) | A written, purpose-specific, revocable permission for a use or disclosure not otherwise permitted (45 CFR 164.508). | Treated as equivalent to GDPR consent. | US legal instrument. |
| Waiver of authorisation | An Institutional Review Board (IRB) or Privacy Board approval allowing research use without authorisation when four criteria are met (45 CFR 164.512(i)(2)(ii)). | Assumed to permit any secondary use. | US legal instrument. |
| Broad consent | Consent to an unspecified range of future research, permitted under the revised Common Rule with governance (45 CFR 46.116(d)). | Treated as unlimited. | US research regulation. |
| Secondary use | Use of data for a purpose other than the one for which it was collected. | Assumed to be prohibited; it is conditional. | Use category. |
| Data minimisation | Data adequate, relevant, and limited to what is necessary for the purpose (GDPR Art. 5(1)(c)). | Interpreted as "collect less than last time". | Legal principle; field-level test. |
| Storage limitation | Data kept no longer than necessary for the purpose (GDPR Art. 5(1)(e)). | Replaced by "keep everything, delete on request". | Legal principle; retention schedule. |
| Accountability | The controller must be able to demonstrate compliance (GDPR Art. 5(2)). | Read as "we have a policy". | Legal principle; evidence. |
| DPIA | Data protection impact assessment: a structured assessment required for high-risk processing (GDPR Art. 35). | Done after deployment to satisfy procurement. | Process artefact. |
| TPO | Treatment, payment, and health care operations: the three HIPAA purposes for which PHI may be used without authorisation (45 CFR 164.506). | Treated as covering research and marketing. | US use category. |
| Personal representative | A person authorised under law to act for an individual in making health care decisions, including privacy decisions (45 CFR 164.502(g)). | Assumed to be the next of kin in all cases. | Legal role. |
| Decedent | A deceased individual; HIPAA protection generally ends 50 years after death (45 CFR 164.502(f)). | Treated as never expiring. | US temporal rule. |
| Incidental use or disclosure | A secondary use or disclosure that cannot reasonably be prevented as part of an otherwise permitted use (45 CFR 164.502(a)(1)(iii)). | Used to excuse careless workflow design. | US legal category. |
| Data use agreement (DUA) | A contract restricting how a recipient may use and re-disclose a limited data set or other shared data (45 CFR 164.514(e)). | Treated as a substitute for a legal basis. | Contract artefact. |
| Limited data set | PHI with 16 direct identifiers removed but dates and geography retained (45 CFR 164.514(e)(2)). | Confused with de-identified data. | US data category. |
| Designated record set | Records used to make decisions about individuals, subject to access and amendment rights (45 CFR 164.501). | Assumed to include every system copy. | US legal category. |
| Accounting of disclosures | A record of certain PHI disclosures that must be provided to individuals on request (45 CFR 164.528). | Assumed to cover all internal uses. | US legal obligation. |
| Information blocking | A practice that interferes with access, exchange, or use of electronic health information, subject to exceptions (45 CFR Part 171). | Used as a reason to share more than necessary. | US regulatory category. |
| Data protection by design | Technical and organisational measures built into processing from the outset (GDPR Art. 25). | Claimed on the basis of a privacy policy. | Legal obligation. |
| Cross-border transfer | Movement of personal data to a country or international organisation, subject to Chapter V conditions. | Treated as a network configuration detail. | Legal event. |

## 5. The mechanics, step by step

The procedure below converts a request into a governed data flow. It is written for a data
steward or analytics lead who has to answer a request in days, not a legal team that has weeks.
Each step states its input, output, and the reason it exists. Steps 4 and 5 are the ones most
often skipped, and they are the ones that decide whether the rest of the work is defensible.

### Step 1. Establish the decision the data supports

**Input:** A request, usually phrased as a dataset ("all type 2 diabetes patients with HbA1c above
9 percent in the last year") rather than a decision.
**Output:** One sentence naming the decision, the actor, and the action: "The diabetes outreach
team will call patients with HbA1c above 9 percent and no endocrinology visit in 12 months to
offer an appointment."
**Rationale:** Purpose is defined by the decision, not by the table. A request without a decision
cannot be minimised, because every field might be relevant to some unspecified use. The decision
statement also exposes requests that have no action attached, which are usually requests for
data hoarding.

Branch point: if the requester cannot name an action and an actor, stop. Either the request is
exploratory, in which case it should run in a controlled environment on de-identified data, or it
is a request for a copy, which requires a stronger justification.

### Step 2. Draft the purpose statement

**Input:** The decision statement.
**Output:** A six-part purpose statement: decision, population, fields, recipients, retention
period, review date.
**Rationale:** The statement is the contract against which every later question is answered. Six
parts are the minimum because each maps to a different control. Population maps to eligibility and
minimum necessary. Fields map to the extract. Recipients map to the DUA and access control.
Retention maps to deletion. Review date maps to accountability.

A usable example:

```text
Purpose: identify patients for a diabetes outreach call
Population: adults 18+ with type 2 diabetes, active in the last 24 months, HbA1c >= 9.0%
Fields: patient ID, preferred name, phone, preferred language, HbA1c value and date,
        last endocrinology visit date, primary care site
Recipients: hospital outreach team (internal), no external recipients
Retention: 90 days after the outreach cycle closes, then delete the extract
Review: 2026-04-30 or when the programme changes
```

Branch point: if the request includes a recipient outside the organisation, the statement must
name the recipient entity and the transfer mechanism. "Partner" is not a recipient.

### Step 3. Map the processing and data flows

**Input:** Purpose statement and system knowledge.
**Output:** A data-flow list: source system, transformation, storage location, access path,
recipient, retention, deletion mechanism.
**Rationale:** Minimum necessary is applied to flows, not to spreadsheets. The same field may be
necessary in the source system and unnecessary in the extract. Mapping exposes copies that nobody
owns: the shared drive folder, the notebook cache, the vendor's staging bucket, the backup that
outlives the retention period.

The map should record, for each flow, whether the data is identifiable, limited, or de-identified.
That label determines which controls apply at the next step. A flow that changes label mid-path
(for example, identified source to de-identified extract) is a control point and should have an
owner.

Branch point: if a flow crosses an organisational boundary, mark it as a transfer and apply Step
7. If a flow crosses a national border, also apply the cross-border edge case below.

### Step 4. Select the legal basis

**Input:** Purpose statement, jurisdiction, entity role, population.
**Output:** A recorded basis with its conditions and the evidence that satisfies them.
**Rationale:** In the EU, processing needs an Article 6 basis and, for health data, an Article
9(2) condition. In the US, the question is whether the entity is a covered entity or business
associate, and whether the use falls within treatment, payment, operations, a permission, or
requires authorisation. These are different structures and cannot be mapped one-to-one. A team
that writes "consent" as the basis for a hospital's treatment analytics has usually chosen the
wrong instrument, because treatment does not require consent under HIPAA and consent is not the
natural basis under GDPR either.

The decision table below is the working tool. It is deliberately simplified; the notes column
names the condition that most often decides the case.

| Scenario | HIPAA route (US) | GDPR Article 6 basis | GDPR Article 9 condition | Consent instrument needed | Practical note |
|---|---|---|---|---|---|
| Treatment of the individual | Permitted, TPO (164.506) | Art. 6(1)(c) or (f), often (c) with Member State health law | Art. 9(2)(h) | No | Minimum necessary does not apply to treatment disclosures. |
| Payment and health care operations | Permitted, TPO | Art. 6(1)(c) or (f) | Art. 9(2)(h) | No | Operations include quality assessment and population health, not marketing. |
| Research with IRB waiver | Waiver under 164.512(i) | Art. 6(1)(e) or (f) with Art. 89(1) safeguards | Art. 9(2)(j) | No, but waiver criteria must be met | Waiver requires minimal risk, impracticability, and a plan to protect identifiers. |
| Research with authorisation | Authorisation under 164.508 | Art. 6(1)(a) | Art. 9(2)(a) explicit consent | Yes | Authorisation must describe purpose, expiration, and right to revoke. |
| Public health reporting | Permitted, 164.512(b) | Art. 6(1)(c) | Art. 9(2)(i) | No | Report to the public health authority, not to a vendor acting for one. |
| Marketing to patients | Authorisation required for paid marketing | Art. 6(1)(a) | Art. 9(2)(a) if health data used | Yes | HIPAA marketing rules are narrower than "we thought it was outreach". |
| Commercial product development on identifiable data | Authorisation or limited data set plus DUA; often neither fits | Art. 6(1)(a) or (f) with safeguards | Art. 9(2)(a) or (j) | Usually yes | "We are improving health" is not a basis for a vendor's product. |
| Serious threat to health or safety | Permitted, 164.512(j) | Art. 6(1)(d) vital interests | Art. 9(2)(c) | No | Disclosure must be to someone able to prevent the threat. |
| Deceased patient | Protection ends 50 years after death | Member State law may protect; GDPR recital 27 excludes deceased | Not applicable after exclusion | No after the period | Pre-death authorisation conditions may still bind the estate. |
| Minor patient | Personal representative under 164.502(g) | Member State rules on age of consent; Art. 8 for information society services | Art. 9(2)(a) or (h) | Representative, unless minor can consent | Adolescent confidentiality rules vary by state and by service. |

Two rules for reading the table. First, a row is not a legal opinion; it is a prompt for the
jurisdiction-specific analysis. Second, the US and EU columns are not interchangeable. A
multi-national programme needs both analyses, and the stricter constraint governs the design.

### Step 5. Test purpose compatibility for secondary use

**Input:** Original purpose statement, proposed new purpose.
**Output:** A compatibility determination: compatible, compatible with conditions, or requires a
new basis.
**Rationale:** Secondary use is where most privacy failures occur, because the new purpose is
usually reasonable in isolation. GDPR Article 6(4) sets out factors: the link between purposes,
the context of collection, the nature of the data, the consequences for individuals, and the
existence of safeguards. HIPAA asks a different question: is the new use within TPO, covered by a
permission, or does it require authorisation or a waiver?

A practical compatibility test uses five questions:

1. Would the individual reasonably expect this use when the data was collected?
2. Does the new use increase the number of recipients or the sensitivity of the inference?
3. Does it involve commercial exploitation, publication, or model training that produces a
   reusable artefact?
4. Does it change the population, for example by adding patients from another site?
5. Can the new purpose be met with a de-identified or in-place alternative?

A "no" to question 1, or a "yes" to questions 2 through 4 without a corresponding safeguard, means
the use needs a new basis. Model training deserves special attention: a trained model can retain
information about its training records, so "we only use the data to train, then delete it" does
not end the privacy question. That is the subject of Topic 03.

Branch point: if the new purpose is research, the waiver route may apply. If it is product
development by a commercial recipient, the waiver route usually does not, because the recipient is
not acting as a covered entity for the research.

### Step 6. Apply minimum necessary at field level

**Input:** Purpose statement, proposed field list, data dictionary.
**Output:** A field-level justification table and a reduced extract specification.
**Rationale:** Minimum necessary is the only principle that changes the shape of the data rather
than the process around it. Field-level review catches the three common excesses: identifiers that
are convenient but unnecessary (full date of birth when age band suffices), free-text fields that
carry unpredictable content, and derived fields that encode sensitive inferences (a frailty score
can reveal disability status).

For each field, record: name, why the purpose requires it, whether a coarser version would
suffice, and whether it is a direct identifier, quasi-identifier, sensitive attribute, or free
text. A field with no justification is dropped. A field with a justification but no granularity
decision is a finding, because the default granularity is always the finest available.

HIPAA's minimum necessary standard has six exceptions where it does not apply: disclosures to a
provider for treatment, disclosures to the individual, uses or disclosures with authorisation,
uses required for compliance with the Administrative Simplification Rules, disclosures to the
Department of Health and Human Services for enforcement, and uses or disclosures required by
other law. These exceptions are frequently misread as "treatment is exempt from privacy". They
exempt only the minimum necessary determination, not the rest of the rule.

Branch point: if the purpose requires the entire record, the determination must say so explicitly
and justify it. "The whole record" is a decision, not a default.

### Step 7. Choose the sharing vehicle

**Input:** Field list, recipient, purpose, legal basis.
**Output:** One of: in-place access, de-identified extract, limited data set plus DUA, identifiable
extract under authorisation or contract.
**Rationale:** The vehicle determines the residual risk and the contractual controls. In-place
access is usually the best option when the recipient is internal or a trusted collaborator,
because the data never leaves the governed environment and access can be revoked. De-identified
extracts are appropriate when the recipient needs only aggregate or record-level research data and
the re-identification risk has been assessed. Limited data sets retain dates and geography and
therefore require a DUA and a recipient who is not trying to re-identify.

The hierarchy is not a ladder of increasing legal difficulty; it is a set of trade-offs. A
de-identified extract that destroys the very variable the analysis needs is not a better control,
it is a failed project. A limited data set with a weak DUA can be worse than an in-place analysis
with strong access control, because the data has left the perimeter.

### Step 8. Set retention and deletion

**Input:** Purpose statement, legal retention obligations, system capabilities.
**Output:** A retention period per data category and a deletion mechanism with evidence.
**Rationale:** Storage limitation is the principle most often satisfied on paper and violated in
practice, because deletion is operationally hard. Backups, snapshots, search indexes, caches, and
derived tables all hold copies. A retention policy that names only the primary table is
incomplete.

Record the legal minimum, the programme need, and the chosen period. Where a legal minimum
exceeds the programme need, the legal minimum governs, but access should be restricted after the
programme ends. Deletion evidence can be a job log, a row count before and after, or a certificate
from a vendor. The evidence matters because "we deleted it" is otherwise unverifiable.

Branch point: if the data is under a litigation hold or a regulatory retention requirement,
deletion is suspended and the hold must be recorded. A hold that is not recorded becomes an
excuse to keep everything.

### Step 9. Define access roles and recipients

**Input:** Purpose statement, organisational roles, minimum data set.
**Output:** A role-to-field matrix, with approval owner and review cadence.
**Rationale:** Access control is covered in Topic 05, but the privacy decision is made here: which
role needs which fields for which purpose. Role-based access is the default; attribute-based
access is used where role alone is too coarse, for example when a researcher needs access only to
records from their own site. The matrix should be small enough to review. A matrix with 60 roles
is a sign that roles were created by copying job titles rather than by analysing need.

Branch point: if a recipient needs access to free-text notes, treat that as a separate
justification with a separate approval. Notes are the highest-risk field class in most clinical
systems because they contain identifiers, family information, and narrative that no schema
constrains.

### Step 10. Record consent or authorisation scope and withdrawal propagation

**Input:** Consent or authorisation instrument, participant register, downstream dataset inventory.
**Output:** A scope record that can answer "is this use permitted for this person" and a
withdrawal process that reaches derived data.
**Rationale:** Consent is only operational if the scope is machine-readable. A boolean is not
enough. The record needs the purposes consented to, the date, the version of the form, the
populations or data types covered, and the withdrawal date. Withdrawal must propagate to extracts,
derived variables, and models trained on the data where feasible. Complete unlearning is often
impossible; the honest position is to document what withdrawal does and does not reach, and to
stop future use.

Branch point: if withdrawal cannot be propagated to a trained model, the consent instrument should
say so before collection. Discovering this after a withdrawal request is a governance failure, not
a technical one.

### Step 11. Assemble the evidence and set the review cadence

**Input:** Outputs of Steps 1 to 10.
**Output:** A privacy pack: purpose statement, data-flow map, basis determination, field
justification, DUA or authorisation, retention schedule, access matrix, review date.
**Rationale:** Accountability is demonstrated, not asserted. The pack is also the artefact that
makes the next request faster, because most requests are variants of an existing flow. A pack that
is versioned and stored with the dataset prevents the common failure where the person who made
the decision leaves and the next team reconstructs it from memory.

Review triggers: a new recipient, a new purpose, a new population, a new jurisdiction, a material
change in volume, a breach, or the passage of the review date. The review date should be no longer
than 12 months for identifiable data and no longer than the programme cycle for extracts.

### Edge cases

**Small cells.** Aggregate outputs with counts of one to five can identify individuals, especially
in rare diagnoses or small geographies. The control is suppression or perturbation, not a promise.
HIPAA Safe Harbor does not itself require small-cell suppression, but expert determination and
GDPR anonymisation assessments do, because a cell of one in a small area is identifiable. The
practical rule is to define a minimum cell size before analysis and to apply secondary suppression
so that totals cannot be reconstructed by subtraction.

**Quasi-identifiers.** Age, sex, ZIP code, dates, and race are not direct identifiers, but
combinations can be unique. The classical result is that a large fraction of a population can be
unique on a small set of such fields. The privacy decision is whether the combination is needed at
full granularity; if it is, the extract moves from de-identified to identifiable and needs the
corresponding controls.

**Re-identification through linkage.** De-identified data can be linked to public datasets (voter
rolls, census, social media, obituary notices) to recover identity. The relevant question is not
whether the data contains names but what auxiliary data a motivated recipient could obtain. This
is why de-identification is an assessment, not a checklist, and why the recipient's environment
matters.

**Free-text PHI.** Clinical notes, referral letters, and radiology reports contain names, phone
numbers, addresses, family details, and rare events. A structured extract can be clean while the
notes field is not. The controls are de-identification of text, restriction of the field, or
in-place analysis. Pattern matching catches some identifiers and misses others; it is a screening
tool, not a guarantee.

**Metadata leakage.** File names, DICOM headers, document properties, email headers, access logs,
and dataset schema can reveal identity or sensitive context even when the content is clean. A
folder named with a patient's name, a DICOM study description containing a clinic name, or a
spreadsheet with the author's username in the properties are all disclosures.

**Consent withdrawal.** Withdrawal stops future processing but does not automatically erase lawful
past processing, and it may not reach backups or models. The policy must state the effect, the
timeline, and the exceptions. The technical process must find the person across datasets, which
requires a linkage key that is itself protected.

**Minors and deceased patients.** Minors are usually represented by a parent or guardian, but
adolescent confidentiality rules differ by jurisdiction and by service, and a parent's access may
be limited for sensitive services. Deceased patients lose HIPAA protection 50 years after death,
but state law, family interests, and the sensitivity of genetic information may impose other
constraints. Both cases break the assumption that the patient is the decision-maker.

**Cross-border transfer.** Moving data to another country is a separate legal event under GDPR
Chapter V, requiring an adequacy decision, standard contractual clauses, or another transfer
tool, plus a transfer impact assessment. In the US, state laws and contract terms may add
conditions. The technical control (encryption, pseudonymisation) supports the legal analysis but
does not replace it. A cloud region choice is a transfer decision, not a procurement detail.

**Business associate chains.** A vendor that subcontracts processing creates another link in the
chain. Each subcontractor that handles PHI needs a business associate agreement, and each
transfer to a subprocessor needs to be authorised. The privacy pack should name the chain, not
only the prime vendor.

### Choosing the sharing vehicle: a second decision table

Step 7 has more than two outcomes, and the choice is frequently made by habit ("we always send a
CSV"). The table below makes the trade-off explicit. Read it after the field list is fixed, never
before, because the vehicle cannot rescue a field list that was never minimised.

| Vehicle | Residual risk | Contract required | Works when | Fails when |
|---|---|---|---|---|
| In-place access in the controller's environment | Lowest: data does not move, access revocable | Access agreement, user terms | Recipient needs record-level data and can work inside the perimeter | Recipient needs to combine with its own data at scale, or latency blocks the analysis |
| Aggregate output only | Low if cell sizes are controlled | Output review procedure | The decision needs rates, counts, or model performance, not records | The analysis needs patient-level trajectories or linkage |
| De-identified record-level extract | Moderate: depends on quasi-identifiers and recipient auxiliary data | DUA, re-identification prohibition | Research or benchmarking where dates and geography are not needed | Rare populations, free text, longitudinal linkage, or a motivated recipient |
| Limited data set plus DUA | Moderate to high: dates and geography retained | DUA under 45 CFR 164.514(e) | US research where temporal precision is essential | Recipient is commercial and the purpose is product development |
| Identifiable extract under authorisation or contract | Highest: full PHI leaves the perimeter | Authorisation, BAA, security schedule | Treatment, coordination of care, or a purpose that genuinely needs identifiers | Any purpose that could be met another way; any recipient without audited controls |

The last row is legitimate but should be rare. A programme in which most flows sit in the last row
has not designed a privacy programme; it has designed a contract library.

### Recording the determination

Each step produces a line in a decision record: what was decided, by whom, on what date, against
which version of the purpose statement. The record can be a table in the privacy pack rather than
a separate document, but it must be specific enough that a reviewer can reconstruct the reasoning
without interviewing the author. "Approved by privacy office" is not a determination. "Approved:
nine fields, no free text, 90-day retention, internal recipients only; rationale: outreach requires
contact details and eligibility, not clinical history" is.

## 6. Worked example

### Setting

Northside Health Partners, a three-hospital US health system, runs a quality-improvement (QI)
programme for adults with type 2 diabetes. A vendor proposes a pilot: it will build a
"social-risk-adjusted adherence model" using Northside data, then license the model back to the
health system and to other systems. The analytics team receives a request for an extract covering
all adults with type 2 diabetes active in the last 24 months. The request lists 34 fields,
including name, medical record number, date of birth, full dates of admission and discharge,
five-digit ZIP code, race, and three free-text note types.

The team applies the procedure above. The decision statement is "the outreach team will call
patients with HbA1c at or above 9.0 percent and no endocrinology visit in 12 months". The purpose
statement names internal recipients, a 90-day retention period, and a review date. The vendor's
proposed purpose - commercial model development - is a second purpose, not the stated one, and it
fails the compatibility test at question 1: no patient would reasonably expect their notes to be
used to build a product licensed to other health systems.

### Step 1: quantify over-collection

The eligible cohort is 41,280 adults. Notes are present for 38,400 of them. Record-field counts
and a weighted exposure score are computed with the ledger in Section 7.

| Field class | Count in request | Values per record | Record-field values | Weight | Weighted units |
|---|---:|---:|---:|---:|---:|
| Direct identifier | 4 | 1 | 165,120 | 5 | 825,600 |
| Quasi-identifier | 7 | 1 | 288,960 | 3 | 866,880 |
| Free text | 3 | 1 (38,400 populated) | 115,200 | 8 | 921,600 |
| Clinical | 20 | 1 | 825,600 | 1 | 825,600 |
| **Total requested** | **34** | | **1,394,880** | | **3,439,680** |

The justified field list contains nine fields: phone, age band, site, HbA1c value, HbA1c date,
last endocrinology visit, active flag, preferred language, and outreach priority. Its totals are:

| Field class | Count | Record-field values | Weight | Weighted units |
|---|---:|---:|---:|---:|
| Direct identifier | 1 | 41,280 | 5 | 206,400 |
| Quasi-identifier | 2 | 82,560 | 3 | 247,680 |
| Free text | 0 | 0 | 8 | 0 |
| Clinical | 6 | 247,680 | 1 | 247,680 |
| **Total justified** | **9** | **371,520** | | **701,760** |

The arithmetic:

- Record-field values removed: 1,394,880 - 371,520 = 1,023,360, which is 73.4 percent of the
  request.
- Weighted exposure removed: 3,439,680 - 701,760 = 2,737,920, which is 79.6 percent.
- Free text contributes 8.3 percent of the record-field values (115,200 / 1,394,880) but 26.8
  percent of the weighted exposure (921,600 / 3,439,680). A field-class weighting is what makes
  that visible; a raw column count would rank free text equal to a lab value.

The weights are a judgement, not a measurement. Their purpose is to make the ranking explicit and
reviewable, not to produce a precise risk number. Section 11 states the limits of the score.

### Diagnostic 1: uniqueness on quasi-identifiers

A sample of 2,000 synthetic patients drawn from the same age, sex, and ZIP distribution is used to
measure how often a combination of date of birth, sex, and five-digit ZIP is unique. The sample
produces 110 singleton equivalence classes out of 2,000 records, a singleton rate of 5.5 percent.

Applied to the cohort: 41,280 x 0.055 = 2,270.4, so approximately 2,270 patients sit in a class of
one on those three fields alone. For those patients, a recipient that can obtain a public voter
file or a commercial people-file can attach a name without any direct identifier in the extract.
This is the classic quasi-identifier result, and it is why removing name and MRN does not make the
extract anonymous.

The diagnostic is a floor, not a ceiling. Adding admission date, discharge date, race, and site
would shrink classes further. Suppressing every singleton class would remove at least 2,270
records, and in practice more, because secondary suppression is needed to stop totals being
reconstructed by subtraction.

### Diagnostic 2: free-text identifiers and detector recall

If notes are included, the detector in Section 7 is run on a sample of 600 notes. Forty-one notes
contain at least one match for an MRN pattern, a US phone number, or an email address: 41 / 600 =
6.83 percent. Extrapolated to the 38,400 patients with notes, that is 38,400 x 0.0683 = 2,624
notes with a detectable direct identifier.

Detection is not complete. A manual review of 60 notes that the detector did not flag finds 9 that
contain a person's name in narrative text. Estimated recall is therefore 41 / (41 + 9) = 0.82.
Correcting the extrapolation: 2,624 / 0.82 = 3,200 notes. The correction matters because it changes
the estimate by 22 percent and because it shows the direction of the error: automated screening
under-counts.

The conclusion from the two diagnostics is the same: the extract as requested is not de-identified
in any operational sense. It contains direct identifiers by design, 2,270 records that are unique
on quasi-identifiers, and up to 3,200 notes with identifiers in free text.

### The purpose test

The consent and notice register records the purposes each patient was told about. In the sample of
eight records used in Section 7, 5 of 8 include quality improvement (62.5 percent), 2 of 8 include
research (25.0 percent), and 0 of 8 include commercial model development (0.0 percent). The
vendor's purpose is outside every recorded scope. The correct finding is not "consent is missing
for some patients"; it is "the proposed purpose has no basis for any patient in the register".

### Decision

The privacy pack records four decisions:

1. The vendor purpose is incompatible with the QI purpose and requires a new basis. The pilot is
   not approved as described.
2. No record-level extract leaves the perimeter. The vendor receives aggregate performance
   reports and works inside a controlled analysis environment if record-level work is needed.
3. Free-text fields are excluded. If narrative is essential, it is de-identified and reviewed
   before use.
4. If a future extract is approved, the field list is the nine justified fields, retention is 90
   days, and the quasi-identifier combination is generalised to age band, three-digit ZIP, and
   year of visit.

### Vary one assumption

The programme narrows the cohort to patients with an endocrinology visit in the last 24 months, a
population of 12,940. Nothing else changes: same fields, same weights, same purpose.

- Weighted exposure: 12,940 x (4 direct x 5 + 7 quasi x 3 + 20 clinical x 1) = 12,940 x 61 =
  789,340, plus free text 3 x 12,040 x 8 = 288,960, for a total of 1,078,300. That is 68.7 percent
  below the original 3,439,680.
- Justified exposure: 12,940 x 17 = 219,980, a 79.6 percent reduction within the smaller cohort.
- Singleton rate: in the narrower, more homogeneous cohort the measured rate rises to 12.1
  percent. Singleton records: 12,940 x 0.121 = 1,565.7, so about 1,566.
- Free-text identifiers: 12,040 x 0.0683 = 822.3, corrected for 0.82 recall gives 1,003.

The absolute number of at-risk records falls, but the proportion rises from 5.5 percent to 12.1
percent, and a record-level extract of the smaller cohort is therefore less defensible per record,
not more. The conclusion changes from "a de-identified extract with suppression might be
acceptable" to "no record-level extract; aggregate or in-place only". This is the counter-intuitive
result that makes cohort size a privacy variable: minimisation reduces total exposure while
increasing concentration.

### Sensitivity of the conclusion

If the free-text weight is reduced from 8 to 4, the requested weighted exposure falls to
2,978,880 - computed as 3,439,680 - 460,800 - and the removal percentage rises to 76.4 percent. The decision does not change, because it is driven by the direct identifiers and the
purpose test, not by the weighting. That is the test of a robust conclusion: it survives a
reasonable change in the scoring assumption. A conclusion that flips when a weight changes was a
conclusion about the weights, not about the data.

### Breach impact of the requested extract

The extract as requested would contain name, MRN, date of birth, phone, and 38,400 notes for
41,280 people. If the recipient environment were compromised, the notification population would
be the full cohort, because direct identifiers are present for every record: 41,280 individuals.

Three impact components can be counted before any dollar assumption:

- Directly identified records: 41,280.
- Records still uniquely identifiable after removal of the four direct identifiers: 2,270 (5.5
  percent), from the quasi-identifier diagnostic.
- Notes containing at least one detectable or manually found identifier: 3,200.

A planning cost can be attached to notification using an explicit input assumption. At an assumed
USD 12 per notified individual for mail, call-centre handling, and identity-monitoring offer, the
notification cost alone would be 41,280 x 12 = USD 495,360. That figure is an assumption for
planning, not a benchmark from any breach study; the point of the arithmetic is that the
notification population is set by the field list, and the field list was 34 columns when 9 were
needed. Reducing to the justified list would remove direct identifiers entirely and shrink the
notification population to whatever the residual re-identification assessment supports.

Retention compounds the impact. A 90-day retention period bounds the window in which the extract
exists; a two-year period multiplies the exposure window by roughly eight. Neither number changes
the probability of a break-in, but both change the expected number of records exposed, which is
the quantity that drives notification and remediation work.

### What the diagnostics do not show

Neither diagnostic models linkage with the recipient's own data. A vendor that has worked with
other health systems may hold overlapping patients and can match on date of birth, sex, and ZIP
without any public dataset. Neither diagnostic inspects metadata: the extract file name, the
spreadsheet author field, the DICOM headers if imaging is added later, and the access log that
records who downloaded what. Neither diagnostic covers inference: a record without a name can
still reveal a diagnosis that the patient has not disclosed. These gaps are the reason the
decision in this case does not rely on the diagnostics alone; it relies on the purpose test and on
keeping the data inside the perimeter.

## 7. Code, query, or calculation

The script below implements the audit used in Section 6. It has three parts: a field ledger with
class weights, a uniqueness check on a synthetic population, and a note scanner with a consent
scope check. It uses only the Python standard library and runs in under a second.

Assumptions stated in the code: the cohort size and note count come from the case; the uniqueness
sample is generated with a fixed seed and is not real patient data; the note scanner is a
screening aid with known false negatives; the consent register is a synthetic sample of eight
records.

```python
"""Purpose-envelope audit for a synthetic diabetes outreach extract."""
import random
import re
from dataclasses import dataclass

WEIGHTS = {"direct": 5, "quasi": 3, "sensitive": 4, "free_text": 8, "clinical": 1}


@dataclass(frozen=True)
class Field:
    name: str
    cls: str
    justified: bool
    populated: int


def build_ledger(n: int = 41_280, notes: int = 38_400) -> list[Field]:
    fields = [
        Field("patient_name", "direct", False, n),
        Field("mrn", "direct", False, n),
        Field("date_of_birth", "direct", False, n),
        Field("phone", "direct", True, n),
        Field("zip5", "quasi", False, n),
        Field("sex", "quasi", False, n),
        Field("race", "quasi", False, n),
        Field("admit_date", "quasi", False, n),
        Field("discharge_date", "quasi", False, n),
        Field("note_primary_care", "free_text", False, notes),
        Field("note_endocrinology", "free_text", False, notes),
        Field("note_social_work", "free_text", False, notes),
        Field("age_band", "quasi", True, n),
        Field("site", "quasi", True, n),
        Field("hba1c_value", "clinical", True, n),
        Field("hba1c_date", "clinical", True, n),
        Field("last_endocrinology_visit", "clinical", True, n),
        Field("active_flag", "clinical", True, n),
        Field("preferred_language", "clinical", True, n),
        Field("outreach_priority", "clinical", True, n),
    ]
    for i in range(14):
        fields.append(Field(f"clinical_{i:02d}", "clinical", False, n))
    return fields


def totals(fields):
    record_fields = sum(f.populated for f in fields)
    weighted = sum(f.populated * WEIGHTS[f.cls] for f in fields)
    kept_rf = sum(f.populated for f in fields if f.justified)
    kept_w = sum(f.populated * WEIGHTS[f.cls] for f in fields if f.justified)
    return record_fields, weighted, kept_rf, kept_w


def synthetic_people(n=2_000):
    rng = random.Random(11)
    dates = [f"19{40 + i // 12}-{1 + i % 12:02d}-15" for i in range(28)]
    zips = [f"606{10 + i:02d}" for i in range(12)]
    return [(rng.choice(dates), rng.choice("MF"), rng.choice(zips)) for _ in range(n)]


def uniqueness(records):
    counts = {}
    for key in records:
        counts[key] = counts.get(key, 0) + 1
    singleton = sum(1 for v in counts.values() if v == 1)
    return len(records), singleton, singleton / len(records)


ID_PATTERNS = {
    "mrn": re.compile(r"\bMRN[:\s]*\d{6,8}\b", re.IGNORECASE),
    "phone": re.compile(r"\b\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
}


def scan_notes(notes):
    hits = {k: 0 for k in ID_PATTERNS}
    flagged = 0
    for note in notes:
        matched = any(p.search(note) for p in ID_PATTERNS.values())
        for name, pattern in ID_PATTERNS.items():
            hits[name] += int(bool(pattern.search(note)))
        flagged += int(matched)
    return {"notes": len(notes), "flagged": flagged, **hits}


def consent_coverage(scopes, purpose):
    permitted = sum(1 for s in scopes.values() if purpose in s)
    return permitted, len(scopes)


if __name__ == "__main__":
    ledger = build_ledger()
    rf, weighted, kept_rf, kept_w = totals(ledger)
    print(f"requested fields: {len(ledger)}, justified: {sum(f.justified for f in ledger)}")
    print(f"record-field values: {rf:,} -> {kept_rf:,} ({(1 - kept_rf / rf) * 100:.1f}% removed)")
    print(f"weighted exposure: {weighted:,} -> {kept_w:,} ({(1 - kept_w / weighted) * 100:.1f}% removed)")

    n, singleton, rate = uniqueness(synthetic_people())
    print(f"sample records: {n}, singleton classes: {singleton}, singleton rate: {rate:.1%}")

    notes = [
        "Patient MRN: 4481203 called about insulin refill.",
        "Daughter (312) 555-0148 requested a copy of the plan.",
        "Discussed diet. No identifiers in this note.",
        "Please email jane.doe@example.org with the appointment time.",
        "Left message at 773.555.0199.",
        "Transport arranged via clinic van.",
    ]
    print(f"note scan: {scan_notes(notes)}")

    scopes = {
        "P001": {"treatment", "quality_improvement"},
        "P002": {"treatment"},
        "P003": {"treatment", "quality_improvement", "research"},
        "P004": {"treatment", "quality_improvement"},
        "P005": {"treatment"},
        "P006": {"treatment", "research"},
        "P007": {"treatment", "quality_improvement"},
        "P008": {"treatment", "quality_improvement"},
    }
    for purpose in ("quality_improvement", "research", "commercial_model_development"):
        ok, total = consent_coverage(scopes, purpose)
        print(f"scope coverage for {purpose}: {ok}/{total} ({ok / total:.1%})")
```

Expected output, verified on Python 3.11:

```text
requested fields: 34, justified: 9
record-field values: 1,394,880 -> 371,520 (73.4% removed)
weighted exposure: 3,439,680 -> 701,760 (79.6% removed)
sample records: 2000, singleton classes: 110, singleton rate: 5.5%
note scan: {'notes': 6, 'flagged': 4, 'mrn': 1, 'phone': 2, 'email': 1}
scope coverage for quality_improvement: 5/8 (62.5%)
scope coverage for research: 2/8 (25.0%)
scope coverage for commercial_model_development: 0/8 (0.0%)
```

Three things to notice in the output. The justified list removes 73.4 percent of record-field
values but 79.6 percent of weighted exposure, because the fields removed are the higher-weight
classes. The note scan flags four of six synthetic notes; the two it misses are the deliberately
benign one and a note that contains a name but no pattern, which is exactly the false-negative
mode that the manual review in Section 6 quantifies. The consent check returns 0.0 percent for the
vendor purpose while returning 62.5 percent for quality improvement, which is the difference
between "some patients consented" and "the purpose is covered".

To adapt the script, replace `build_ledger` with the actual data dictionary and `scopes` with the
real consent register. The ledger should be reviewed by the data owner, not by the engineer alone,
because the `justified` flag is a policy decision recorded in code.


## 8. Data, standards, and regulatory context

Privacy principles are expressed in law as broad obligations and in standards as management
systems. The difference matters: a law tells you what must be true, a standard tells you what
artefacts usually make it demonstrably true, and neither tells you what will work in your
organisation.

### United States: HIPAA Privacy Rule

The HIPAA Privacy Rule, at 45 CFR Part 160 and Subparts A and E of Part 164, is the binding
requirement for covered entities and business associates. The provisions most relevant to this
topic are:

| Provision | What it requires | What it does not do |
|---|---|---|
| 45 CFR 164.502(b) and 164.514(d) | Limit uses, disclosures, and requests to the minimum necessary, with six categories of exception | Require case-by-case review for routine disclosures, or apply to treatment disclosures |
| 45 CFR 164.506 | Permit use and disclosure for treatment, payment, and health care operations | Cover research, marketing, or sale of PHI |
| 45 CFR 164.508 | Require a written, specific, revocable authorisation for other uses | Prescribe a single national form |
| 45 CFR 164.512(i) | Permit research with IRB or Privacy Board waiver of authorisation when four criteria are met | Permit a waiver for commercial product development by a non-covered entity |
| 45 CFR 164.514(e) | Permit a limited data set with a data use agreement | Make a limited data set de-identified |
| 45 CFR 164.502(f) | End protection 50 years after death | Override state law or family interests |
| 45 CFR 164.502(g) | Define personal representatives for minors and incapacitated adults | Resolve adolescent confidentiality, which is largely state law |
| 45 CFR 164.528 | Require an accounting of certain disclosures | Cover all internal uses |

The Privacy Rule is a floor. State laws such as the California Confidentiality of Medical
Information Act and Washington's My Health My Data Act add requirements, and where state law is
more protective it generally governs. The HIPAA preemption analysis is fact-specific and is a
legal question, not an engineering one.

### European Union: GDPR and the EHDS

The GDPR (Regulation (EU) 2016/679) states seven principles in Article 5: lawfulness, fairness and
transparency; purpose limitation; data minimisation; accuracy; storage limitation; integrity and
confidentiality; and accountability. Health data is a special category under Article 9(1), so
processing needs both an Article 6 basis and an Article 9(2) condition. Article 25 requires data
protection by design and by default, Article 35 requires a data protection impact assessment for
high-risk processing, and Chapter V governs transfers outside the European Economic Area.

The European Data Protection Board (EDPB) adopted Guidelines 01/2025 on pseudonymisation in
January 2025. The guidance confirms that pseudonymised data remains personal data where
attribution is reasonably likely, including when the additional information is held by a different
party, and introduces the concept of a "pseudonymisation domain" - the set of actors from whom
attribution must be precluded. That framing is directly useful for vendor arrangements, because it
forces the design to name who could re-link the data, not merely whether a key exists.

The European Health Data Space (EHDS) Regulation (EU) 2025/327 was adopted on 11 February 2025,
published on 5 March 2025, and entered into force on 26 March 2025. Its secondary-use chapter
applies from 26 March 2029 for most data categories. It creates health data access bodies that
issue data permits, requires access through secure processing environments, prohibits
re-identification, and allows opt-out. For a team building a secondary-use pipeline in the EU,
the EHDS changes the access route: the question becomes whether a permit is required and which
secure environment is used, not only whether a controller has a lawful basis.

### Research regulation and sector-specific rules

- The revised Common Rule (45 CFR 46) governs federally funded human-subjects research in the US
  and permits broad consent under 46.116(d) with additional governance. The Food and Drug
  Administration (FDA) has separate informed-consent regulations at 21 CFR Part 50 for clinical
  investigations.
- 42 CFR Part 2 protects substance use disorder records and is stricter than HIPAA in several
  respects. The 2024 final rule aligned many provisions with HIPAA and set a compliance date of
  16 February 2026, but the single consent and redisclosure rules still constrain sharing.
- The FTC Health Breach Notification Rule (16 CFR Part 318) covers vendors of personal health
  records and related service providers that are not covered by HIPAA. The amended rule took
  effect on 29 July 2024 and reaches consumer health apps.
- The 21st Century Cures Act information-blocking rule (45 CFR Part 171) requires sharing of
  electronic health information with patients and other requesters, subject to exceptions. It
  pushes in the opposite direction from minimum necessary, and the two must be reconciled in
  policy: patient access is not a disclosure requiring minimum necessary analysis.

### Standards and frameworks

The NIST Privacy Framework (version 1.0, January 2020) is voluntary and organizes privacy work
into five functions: Identify-P, Govern-P, Control-P, Communicate-P, and Protect-P. Its value is
that it links privacy outcomes to cybersecurity controls and gives an auditor a structure to test.
ISO/IEC 27701:2019 extends ISO/IEC 27001 to privacy information management and is certifiable,
which makes it useful in procurement. Neither standard creates legal obligations; they are
management-system conventions that help demonstrate accountability.

> **Note:** Regulatory content in this file is educational and is not legal advice. Statutes,
> guidance, and enforcement priorities change, and the analysis of a specific data flow requires
> counsel qualified in the relevant jurisdiction.

The distinction to keep straight is between a legal requirement (binding, enforceable, and
jurisdiction-specific), a security best practice (reduces risk but is not mandated), and a
statistical guarantee (a mathematical property such as epsilon-differential privacy, which is
independent of law and can be defeated by poor implementation). Minimum necessary is a legal
requirement. Encrypting an extract at rest is a best practice that supports several legal
requirements. A k-anonymity claim is a statistical property of a released table, not a legal safe
harbour.

## 9. Industry practice and case evidence

### What mature organisations do

Three practices separate organisations that operationalise privacy principles from those that
merely publish a notice.

**Tiered access instead of binary sharing.** Research networks such as PhysioNet distinguish open
access, credentialed access with a data use agreement, and controlled access for the most
sensitive datasets. The UK's secure data environments and the EHDS secure processing environments
follow the same logic: move the analysis to the data, issue project-specific permissions, and
review outputs before release. This is a design pattern, not a legal requirement, but it is the
one that most often survives audit.

**A data access committee with published criteria.** The committee records the purpose, the fields
approved, the retention period, and the conditions. Publishing criteria reduces the variation
between reviewers and makes refusals defensible. The practical test of such a committee is whether
its minutes contain field-level decisions; if every approval says "approved as requested", the
committee is a rubber stamp.

**Purpose-bound consent registers.** Organisations that rely on consent store the scope, version,
date, and withdrawal status, and can answer "may this person's data be used for purpose X" without
reading a PDF. The register is the difference between consent as an event and consent as an
operational constraint.

### Documented cases

**Anthem, 2018.** The US Department of Health and Human Services (HHS) Office for Civil Rights
(OCR) announced a USD 16 million settlement with Anthem on 15 October 2018, covering a breach
affecting 78.8 million individuals. OCR cited failure to conduct an enterprise-wide risk analysis,
insufficient review of information system activity, failure to respond to a known incident,
insufficient technical access controls, and impermissible disclosure of electronic protected
health information. The settlement included a two-year corrective action plan.

The privacy lesson is often missed because the case is filed under security. The impermissible
disclosure count exists because the data existed in a warehouse with broad access and no
field-level minimisation. A smaller, purpose-bound dataset would not have prevented the phishing
email, but it would have reduced what the attackers could take.

**Royal Free London and DeepMind, 2017.** The UK Information Commissioner's Office (ICO) concluded
that the Royal Free NHS Foundation Trust, in its arrangement with DeepMind for the Streams app,
processed approximately 1.6 million partial patient records without adequately informing patients
and without a valid condition for processing. The ICO also questioned whether the volume of data
was proportionate to the stated clinical safety purpose. The Trust signed an undertaking rather
than receiving a fine, and agreed to complete a privacy impact assessment and a third-party audit.

This is the clearest public example of purpose limitation failing at the design stage. The
clinical goal was legitimate; the data flow was not justified field by field, and patients were
not told. The corrective action was procedural - a PIA and an audit - which is what regulators
typically require when the harm is loss of control rather than a demonstrated misuse.

**NHS care.data, 2013-2016.** England's programme to extract general-practice data into a central
repository was paused in 2014 after criticism of its communication and opt-out arrangements, and
was closed in 2016; the data collected were subsequently destroyed. The documented failure was
one of transparency and public legitimacy, not of a specific technical control. For this topic,
the lesson is that a lawful basis and a security architecture are insufficient if the population
does not understand or accept the purpose.

**23andMe, 2023.** The company confirmed that a credential-stuffing attack gave attackers access to
data belonging to approximately 6.9 million users, including profile and health-related
information for a subset. The mechanism was reused passwords and missing protections on a feature
that allowed profile sharing. The case is a reminder that the privacy boundary includes account
and feature design: a sharing feature that is convenient for users is also an attack surface for
every linked profile.

> **Note:** The case descriptions above report published findings and settlements. Statements
> about what an organisation "should have done" are inference from the findings, not findings
> themselves.

### Where practice is weaker than the documents suggest

Three gaps recur in assessments, and they are inference from repeated audit findings rather than
from a single study. First, retention schedules exist but deletion is not executed, so the
schedule describes intent. Second, business associate agreements name the prime vendor but not the
subcontractors, so the processor chain is invisible. Third, consent withdrawal is honoured in the
primary system but not in extracts or models, so the scope record is accurate and the data are
not. Each gap is detectable with a simple test: ask for the deletion job log, the subprocessor
list, and the last withdrawal reconciliation report. If any of the three cannot be produced within
a day, the control is not operational.


## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Purpose creep by phase | Phase 2 inherits Phase 1 data with a new vendor and no new review | Each phase is approved as an extension of the last | Original notice and consent do not cover the use; regulatory finding | Re-run the compatibility test at every phase; version the purpose statement |
| Consent stored as a boolean | The consent table has one column and no form version | Scope and date are lost at ingestion | Organisation cannot honour restrictions or prove consent | Store purposes, version, date, withdrawal status; make scope queryable |
| Over-collection by default | Extract has more columns than the analysis uses; nobody can name the owner of each | Retention is cheaper than justification | Larger breach surface and notification population | Field-level ledger; default deny for new columns |
| Minimum necessary over-applied | Treatment teams cannot see records they need; workarounds appear | Rule applied outside its six exceptions | Care delays and shadow copies that are less controlled | Train on the exceptions; separate treatment access from analytics review |
| De-identification treated as anonymisation | "It is de-identified, so GDPR does not apply" | Legal and statistical concepts conflated | Unlawful processing; re-identification incident | Record the standard used and the residual risk; legal review for anonymity claims |
| Free text shipped unexamined | Notes included "because they might help" | Narrative contains identifiers and rare events | Direct identifiers in a nominally structured extract | Exclude by default; screen and review before any release |
| Retention without deletion | Schedule exists; no job log; backups older than the period | Deletion is cross-system and unowned | Data retained beyond purpose; discoverable in litigation | Automated deletion with evidence; include backups, caches, indexes |
| Invisible subcontractor chain | BAA names the vendor but not its cloud or annotation vendor | Subprocessing added after contract signature | Unauthorised disclosure; transfer without basis | Subprocessor register; notification and approval clause |
| Withdrawal not propagated | Withdrawal honoured in the EHR but not in the research extract | No person-level reconciliation | Continued use after withdrawal; loss of trust | Linkage key and reconciliation job; document what cannot be undone |
| Small-cell disclosure | Report shows counts of 1 or 2 by rare diagnosis and area | Aggregation assumed to be safe | Re-identification from a published table | Minimum cell size; secondary suppression; output review |
| Metadata leakage | File names, DICOM headers, document author fields not inspected | Metadata is outside the data table | Disclosure without any content field | Metadata stripping in the export pipeline; test the artefact, not the schema |
| Cross-border transfer by configuration | New cloud region chosen for latency; no transfer review | Infrastructure decision made outside privacy process | Unlawful transfer; contract breach | Transfer inventory; region allowlist; Chapter V analysis |
| Minor or representative access error | Parent portal shows adolescent-sensitive encounters | Age rules not encoded in access control | Privacy harm to a minor; state-law violation | Age-aware access rules; legal review by service line |
| Deceased records released without review | "HIPAA does not apply after 50 years" used as blanket rule | Other laws and family interests ignored | Harm to relatives; state-law breach | Check state law and genetic sensitivity; apply review regardless |

The red flags share a pattern: each is visible before any incident, in an artefact that somebody
already has. The extract column list, the consent table schema, the retention job log, the
subprocessor register, and the cloud region configuration are all routine outputs. A privacy
review that does not read them is reviewing intentions.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The purpose statement remains valid for the programme | All later controls derive from it | Programme scope expands, vendor changes, population changes | Review at each phase gate and at the review date |
| The requester can name the decision and the actor | Purpose cannot be minimised without it | Exploratory requests and data hoarding | Ask for the action and the person; refuse copies without one |
| Field-level justification is reviewed by the data owner | Engineers cannot decide policy | Ledger approved only by the requester | Check sign-off in the privacy pack |
| Sensitivity weights are ordinal judgements | They rank field classes for review | Weights treated as measured risk | Sensitivity analysis: change a weight and see if the decision holds |
| Sample singleton rate generalises to the cohort | Measuring the full cohort may be infeasible | Sample drawn from a different geography or age mix | Stratify the sample; compare with a full-cohort check where possible |
| Note-detector recall estimate generalises | Manual review is sampled | Note style differs by clinic or specialty | Re-estimate recall on a second sample; report the confidence interval |
| Consent register is complete and current | Scope queries depend on it | Historical consents not migrated; paper forms | Reconcile a random sample against source documents |
| Recipients comply with the DUA | Contract is the control after transfer | Recipient re-shares or links data | Audit clause; technical restrictions; periodic attestation |
| Retention jobs cover all copies | Storage limitation depends on deletion | Backups, caches, indexes omitted | Test deletion end to end and count residual rows |
| Entity classification is correct | Determines which legal regime applies | A consumer app or research spin-out is misclassified | Legal review of the entity and the data flow, not the product name |
| Jurisdiction does not change during the programme | Basis and transfer analysis are jurisdiction-specific | New sites, new users, new cloud region | Maintain a jurisdiction register; review on change |

Two limits deserve emphasis. First, nothing in this topic measures harm to individuals; it
measures exposure and compliance. A small, fully compliant dataset can still support a harmful
inference. Second, the procedure is designed for a single organisation with a functioning privacy
function. In a startup without counsel, the same steps are necessary but the legal-basis step must
be escalated earlier, because the cost of getting it wrong is disproportionate.

## 12. Exercises

**Exercise 1 - Classify the control (recall, 30 minutes).**
Take ten controls from a privacy programme you know: encryption at rest, minimum necessary,
purpose limitation, access review, k-anonymity with k=5, DUA, retention deletion, MFA, DPIA,
pseudonymisation. Classify each as a legal requirement, a security best practice, or a statistical
guarantee, and name the jurisdiction or standard that makes it so. Deliverable: a three-column
table with one sentence per row. Criteria: every row has a category and a source; at least two
rows are correctly identified as having mixed status.

**Exercise 2 - Write a purpose statement (application, 45 minutes).**
A payer asks for 18 months of claims data for "analytics". Write a purpose statement with the six
parts from Section 5, inventing a plausible decision and actor. Deliverable: one page. Criteria:
decision names an action and an actor; population is bounded; field list is justified; recipient
and retention are explicit; review date is present.

**Exercise 3 - Minimum necessary review (application, 60 minutes).**
Using the 34-field request in Section 6, remove fields until only justified ones remain. For each
removed field, write one sentence on what analysis capability is lost. Deliverable: field table
with keep/drop and consequence. Criteria: no direct identifier is kept without a stated need;
free text is either excluded or given a separate justification; the lost-capability column is
specific.

**Exercise 4 - Two-jurisdiction mapping (application, 45 minutes).**
For six scenarios (treatment, research with waiver, marketing, public health reporting,
commercial model development, deceased patient), fill the US and EU columns of the Section 5 table
without looking at it, then compare. Deliverable: completed table plus a list of three places where
the two regimes diverge. Criteria: Article 6 and Article 9 are treated separately; HIPAA route is
named by permission or authorisation, not by "HIPAA".

**Exercise 5 - Modify the audit (application, 60 minutes).**
Run the Section 7 script. Add a field class "genetic" with weight 6, add two synthetic genetic
fields, and recompute. Deliverable: modified script plus output. Criteria: script runs, totals are
internally consistent, and the write-up explains whether the decision in Section 6 changes.

**Exercise 6 - Critique a consent form (critique, 45 minutes).**
Find a public consent or authorisation form from a health system or research study. Assess it
against specificity, revocability, comprehension, and scope. Deliverable: a 400-word critique with
three concrete defects and rewrites. Criteria: defects are tied to a legal or operational
requirement; rewrites are usable text, not advice.

**Exercise 7 - Design withdrawal propagation (design, 90 minutes).**
Design the process that runs when a participant withdraws from a biobank with three downstream
datasets and one trained model. Specify the linkage key, the reconciliation steps, what is deleted,
what is retained under which exception, and what the participant is told. Deliverable: process
diagram plus a two-page procedure. Criteria: covers extracts, backups, and the model; states what
cannot be undone; names an owner and a timeline.

**Exercise 8 - Notification population under two retention policies (calculation, 45 minutes).**
An extract of 41,280 records with direct identifiers exists for 90 days under Policy A and two
years under Policy B. Assume a 0.8 percent annual probability of a reportable compromise of the
environment, independent of retention. Compute the expected number of records exposed per year
under each policy over a three-year programme, and state the assumption that makes the comparison
valid. Deliverable: calculation with units. Criteria: correct compounding; the independence
assumption is stated; the answer distinguishes probability from expected count.

## 13. Self-check answers

**Answer 1.** Encryption at rest is a security best practice that supports the HIPAA Security Rule
and GDPR Article 32; it is not, by itself, a legal requirement in every context. Minimum necessary
is a legal requirement under 45 CFR 164.502(b). Purpose limitation is a legal requirement under
GDPR Article 5(1)(b) and a principle under HIPAA's treatment of uses and disclosures. Access review
is a best practice and an implied control under the Security Rule. k-anonymity with k=5 is a
statistical guarantee about a released table, not a legal standard; it does not satisfy GDPR
anonymity by itself. A DUA is a contract control, required for a HIPAA limited data set. Retention
deletion is a legal requirement under GDPR Article 5(1)(e) and a HIPAA documentation-retention
control. MFA is a best practice, increasingly expected by insurers and procurement. A DPIA is a
legal requirement for high-risk processing under GDPR Article 35. Pseudonymisation is a security
measure and, under GDPR, a risk-reduction technique that does not remove data from scope.

**Answer 3.** The kept fields should be phone, age band, site, HbA1c value, HbA1c date, last
endocrinology visit, active flag, preferred language, and outreach priority. Removing name, MRN,
and date of birth loses the ability to confirm identity at the point of contact; the mitigation is
that the outreach team calls the number on file and confirms two data points, or that the extract
is joined back to the EHR inside the perimeter at call time. Removing ZIP, sex, race, admission
and discharge dates loses geographic equity analysis and utilisation adjustment; if those are
required, the request is a different purpose and needs its own review. Removing notes loses
context for social barriers; the mitigation is a structured social-risk field, which is a data
collection change, not an extraction change. The exercise is designed to show that "drop it" is
not free and that the lost capability should be named before the decision, not after.

**Answer 5.** The modified script should show total weighted exposure rising because genetic
fields carry weight 6 and are populated for the full cohort. If the two genetic fields are not
justified, the removal percentage rises; if they are justified, the kept exposure rises and the
case for sharing weakens. In either case the Section 6 decision does not change, because it turns
on the purpose test and on direct identifiers, not on the weight of an additional field. A
conclusion that changes when a field is added would indicate that the decision was driven by the
score rather than by the purpose.

**Answer 8.** Policy A: the extract exists for 90/365 = 0.2466 years per cycle. With an annual
compromise probability of 0.008, the per-cycle probability is approximately 0.008 x 0.2466 =
0.00197, so the expected records exposed per cycle is 41,280 x 0.00197 = 81.4. Over three cycles,
about 244 records. Policy B: two years per cycle gives a per-cycle probability of 0.008 x 2 =
0.016, expected records 660.5 per cycle, and about 1,981 over three cycles. The comparison assumes
independence between years and that the compromise probability is proportional to exposure time;
both are simplifications. The point is that retention changes expected harm even when the annual
probability is unchanged.

**Rubric for Exercise 2.** Full credit requires all six parts, a bounded population, a field list
that a reviewer could minimise, and a review date. A purpose statement that names "analytics" or
"research" without a decision loses half the marks, because it cannot support the later steps.

**Rubric for Exercise 4.** Full credit requires the US column to name a HIPAA route (TPO,
permission, authorisation, waiver) and the EU columns to separate Article 6 from Article 9. A row
that says "consent" for treatment in either jurisdiction is incorrect.

**Rubric for Exercise 6.** Full credit requires three defects that map to specificity,
revocability, or scope, plus rewrites that a patient could understand. Generic criticism of "legal
language" without a specific clause does not earn credit.

**Rubric for Exercise 7.** Full credit requires a linkage key design, a reconciliation procedure,
explicit treatment of backups and the trained model, a statement of what cannot be undone, and an
owner with a timeline. A design that promises complete erasure from a trained model without
qualification is marked down for overclaiming.

## 14. Interview questions

**1. What is the difference between a legal requirement, a security best practice, and a
statistical guarantee?**
Strong answer: gives an example of each (minimum necessary, encryption at rest, epsilon-differential
privacy) and explains that they fail differently - legal requirements are enforced, best practices
reduce risk, statistical guarantees are properties of a mechanism that can be invalidated by
implementation. Weak answer: treats compliance as risk management and lists frameworks.

**2. When does minimum necessary not apply?**
Strong answer: names the six HIPAA exceptions, especially treatment, the individual, and
authorisation, and notes that the exception is to the minimum necessary determination, not to the
rest of the Privacy Rule. Weak answer: says "in emergencies" without the categories.

**3. Why is consent often the wrong basis for hospital analytics?**
Strong answer: explains that treatment and operations have other bases, that consent must be
freely given and withdrawable, and that a coerced consent record is worse than no consent because
it misrepresents the relationship. Weak answer: says consent is always required.

**4. How would you decide whether a new use is compatible with the original purpose?**
Strong answer: uses the Article 6(4) factors and a small set of questions about expectations,
recipients, commercialisation, population, and alternatives. Weak answer: asks whether the new use
is "also for health".

**5. What does de-identification remove, and what does it leave?**
Strong answer: removes direct identifiers under a specified standard, leaves quasi-identifiers,
free text, metadata, and linkage risk; distinguishes HIPAA Safe Harbor from expert determination
and from GDPR anonymity. Weak answer: says de-identification makes data anonymous.

**6. How do you handle a withdrawal request that reaches a trained model?**
Strong answer: stops future use, removes the record from datasets where feasible, documents what
the model retains, and explains the limit to the participant; ideally the consent instrument
disclosed this before collection. Weak answer: promises deletion from the model.

**7. What is the privacy risk of free-text notes, and how do you control it?**
Strong answer: unpredictable identifiers, family information, rare events, detector false
negatives; controls are exclusion, de-identification with review, or in-place analysis. Weak
answer: "we run a regex".

**8. How does retention affect breach impact?**
Strong answer: expected exposed records scale with exposure time; retention also affects
discoverability and regulatory exposure; deletion must cover backups and caches. Weak answer:
retention is a storage cost issue.

**9. A vendor says it is HIPAA compliant. What do you ask next?**
Strong answer: asks whether it is a covered entity or business associate, for its risk analysis,
for the subprocessor list, for the BAA, for breach notification terms, and for evidence of access
review. Weak answer: accepts the claim.

**10. What would make you stop a data-sharing proposal?**
Strong answer: no lawful basis for the stated purpose, no way to minimise, a recipient who cannot
be contracted or audited, or a harm that cannot be mitigated. Weak answer: only a legal prohibition
would stop it.

## 15. Key terms

- **Accountability** - the obligation to demonstrate compliance, not merely assert it.
- **Authorisation** - a specific, revocable HIPAA permission for a use or disclosure (45 CFR 164.508).
- **Broad consent** - consent to a range of future research under governance (45 CFR 46.116(d)).
- **Business associate** - an entity handling PHI on behalf of a covered entity.
- **Consent** - a freely given, specific, informed, unambiguous indication of wishes under GDPR.
- **Covered entity** - a health plan, clearinghouse, or provider subject to HIPAA.
- **Cross-border transfer** - movement of personal data subject to GDPR Chapter V.
- **Data minimisation** - collecting and using only what the purpose requires.
- **Data use agreement** - a contract restricting use and redisclosure of shared data.
- **De-identification** - removing or transforming identifiers under a specified standard.
- **DPIA** - data protection impact assessment for high-risk processing.
- **EHDS** - European Health Data Space Regulation (EU) 2025/327.
- **HIPAA** - Health Insurance Portability and Accountability Act, and its implementing rules.
- **IIHI** - individually identifiable health information.
- **Incidental disclosure** - a secondary disclosure that cannot reasonably be prevented.
- **Information blocking** - interference with access or exchange of electronic health information.
- **Legal basis** - the Article 6 condition that makes processing lawful.
- **Limited data set** - PHI with 16 direct identifiers removed but dates and geography retained.
- **Minimum necessary** - the requirement to limit PHI to the least needed for the purpose.
- **Personal data** - information relating to an identified or identifiable person.
- **Personal representative** - a person authorised to act for an individual on privacy matters.
- **PHI** - protected health information under HIPAA.
- **Pseudonymisation** - replacing identifiers while retaining a linkage mechanism.
- **Purpose limitation** - restriction of processing to specified, compatible purposes.
- **Purpose statement** - the operational artefact that fixes decision, population, fields, recipients, retention, and review.
- **Secondary use** - use of data for a purpose other than the one it was collected for.
- **Special category data** - GDPR Article 9 data, including health and genetic data.
- **Storage limitation** - retention only as long as necessary.
- **TPO** - treatment, payment, and health care operations.
- **Waiver of authorisation** - IRB or Privacy Board approval for research without authorisation.

## 16. Sources

All URLs accessed 26 October 2026.

**US regulation and guidance**
- HHS Office for Civil Rights, *Minimum Necessary Requirement*.
  https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/minimum-necessary-requirement/index.html
- HHS Office for Civil Rights, *Summary of the HIPAA Privacy Rule*.
  https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html
- HHS Office for Civil Rights, *Guidance on De-identification of Protected Health Information*.
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- Electronic Code of Federal Regulations, 45 CFR Part 164.
  https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164
- HHS Office for Civil Rights, *Anthem Resolution Agreement and press release*, 15 October 2018.
  https://www.hhs.gov/hipaa/for-professionals/compliance-enforcement/agreements/anthem/index.html
- HHS Office for Civil Rights, *Breach Notification Rule*.
  https://www.hhs.gov/hipaa/for-professionals/breach-notification/index.html
- Federal Trade Commission, *Health Privacy*.
  https://www.ftc.gov/business-guidance/privacy-security/health-privacy

**EU regulation and guidance**
- Regulation (EU) 2016/679 (General Data Protection Regulation).
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Regulation (EU) 2025/327 (European Health Data Space).
  https://eur-lex.europa.eu/eli/reg/2025/327/oj
- European Data Protection Board, *Guidelines 01/2025 on Pseudonymisation*, adopted January 2025.
  https://www.edpb.europa.eu/public-consultations/guidelines-012025-on-pseudonymisation_en
- European Data Protection Board, *Guidelines 01/2025 on Pseudonymisation* (PDF).
  https://www.edpb.europa.eu/system/files/2025-01/edpb_guidelines_202501_pseudonymisation_en.pdf

**Frameworks and standards**
- NIST, *Privacy Framework, Version 1.0*, January 2020. https://www.nist.gov/privacy-framework
- NIST, *SP 800-66 Rev. 2, Implementing the HIPAA Security Rule*. https://csrc.nist.gov/pubs/sp/800/66/r2/final
- ISO/IEC 27701:2019, *Security techniques - Extension to ISO/IEC 27001 and ISO/IEC 27002 for privacy information management*.

**Case evidence**
- UK Information Commissioner's Office, *Google DeepMind and the Royal Free London NHS Foundation
  Trust*, 2017. https://ico.org.uk/for-the-public/ico-40/google-deepmind-and-class-action-lawsuit/
- Royal Free London NHS Foundation Trust, *Undertaking to the Information Commissioner*, 2017.
  https://www.bailii.org/uk/cases/UKICO/2017/2017-66.pdf
- NHS England, *care.data programme closure*, 2016. Cited as a programme announcement; no stable
  public URL is maintained.

**Operational practice**
- PhysioNet, credentialed and controlled access model. https://physionet.org/
- NIH, *Genomic Data Sharing Policy*. https://sharing.nih.gov/

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_deidentification-and-anonymization.md`](02_deidentification-and-anonymization.md)
- Related topic: [`04_threat-modeling.md`](04_threat-modeling.md)
- Related topic: [`10_regulatory-map.md`](10_regulatory-map.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 16.2, de-identification audit, in the survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
