# De-identification and Anonymization

> **Module 16 - Privacy, Security, and Ethics**
> Topic 02 of 10
> Estimated study time: 3 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

De-identification is the control that most biomedical data programmes rely on and the one most
often misdescribed. A dataset is labelled "de-identified" in a data catalogue, the label travels
through three systems and two contracts, and by the time an analyst receives it the label has
become a claim that no one can defend: that the data cannot be linked to a person. Sometimes that
claim is true. Often it is not, and the gap between the label and the reality is where
re-identification incidents, regulatory findings, and lost public trust originate.

The commercial stakes are concrete. A hospital that wants to share data with a research partner, a
device maker that wants to publish a clinical dataset, a payer that wants to release claims for
benchmarking, and a startup that wants to train a model on clinical text all face the same
question: how much must be removed, generalized, or perturbed before the release is acceptable,
and how do we show our reasoning? Answering "we removed names and MRNs" is not an answer. The
recipient can join the data to a voter file, the dates can be unique, the free text can contain a
daughter's phone number, and the DICOM header can carry the treating physician's name.

Three failure patterns recur.

**Checklist de-identification.** A team applies the HIPAA Safe Harbor list to a structured table,
removes the 18 categories, and stops. The table still contains admission and discharge dates
converted to day offsets, five-digit ZIP codes, and an age field with values up to 97. Each is a
quasi-identifier; together they can be unique. The checklist was followed and the release is still
identifiable.

**Free text treated as a column.** The structured extract is clean; the notes field is copied
unchanged because "the vendor needs context". Notes contain names, family relationships, phone
numbers, addresses, and rare events. A regex sweep catches some patterns and misses names in
narrative. The release is identifiable through a field that nobody classified as an identifier.

**Anonymity asserted instead of assessed.** A privacy notice says the data are anonymous because a
key was deleted. Under GDPR, anonymity is a property of the data in the hands of a particular
actor, considering all means reasonably likely to be used. If the recipient holds the key, or can
obtain the auxiliary data, the data are pseudonymised at best. The legal consequence is not
cosmetic: anonymous data fall outside the GDPR, pseudonymised data do not.

This topic teaches de-identification as an assessment with a documented method, not a label. It
covers the two HIPAA routes, the GDPR standard, the statistical measures that make residual risk
visible, and the operational controls that keep a release defensible after it leaves the building.
The distinction between de-identification and anonymisation is the spine of the topic: the first
reduces risk under a defined standard, the second claims that re-identification is not reasonably
possible, and only the second removes data from the scope of data protection law.

A fourth pattern is worth naming because it is the inverse of the others: **over-suppression that
destroys the project**. A team, aware that de-identification is hard, removes every field that
could conceivably identify anyone and produces a dataset that cannot answer the research question.
The release is safe and useless. The cost is not only the wasted effort; it is that the next
request is treated as impossible before it is analysed, and the organisation loses the ability to
do legitimate secondary research. The remedy is the same as for under-protection: state the
purpose, measure the utility cost of each transformation, and choose a release mode that preserves
what the purpose requires.

> **Note:** A de-identified dataset is not a safe dataset. It is a dataset whose residual risk has
> been assessed against a stated adversary and is judged acceptable for a stated release.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** de-identification, pseudonymisation, anonymisation, and synthetic data, and
   state which legal regime each falls under.
2. **Apply** the HIPAA Safe Harbor method, including the ZIP code, age, and date rules, and
   **identify** when the method cannot be used.
3. **Design** an expert determination that states the adversary, the auxiliary data, the method,
   the measured risk, and the residual controls.
4. **Compute** k-anonymity, l-diversity, entropy, and prosecutor and marketer risk for a small
   release, and **explain** what each measure does and does not capture.
5. **Choose** a transformation - suppression, generalization, perturbation, masking, tokenization -
   for a given field and justify the utility cost.
6. **Audit** a nominally de-identified release for free-text identifiers, metadata, and
   quasi-identifier combinations.
7. **Document** a de-identification determination that a regulator, an IRB, or a procurement
   reviewer can follow.
8. **Critique** the claim "this dataset is anonymous" and replace it with a defensible statement.

## 3. The core mental model

De-identification is a comparison between two populations: the people in the dataset and the
people an adversary can find in the outside world. The dataset is identifiable to the extent that
a record can be matched to a unique person, or a person's sensitive attribute can be inferred,
using the data plus whatever the adversary already knows.

```text
                    AUXILIARY DATA
        voter file | census | claims | social media
        |  device logs | prior releases | the recipient
                          |
                          v
   +----------------------+----------------------+
   |                 RELEASED DATA               |
   |  direct identifiers  -> removed?            |
   |  quasi-identifiers  -> how unique?          |
   |  sensitive values   -> how homogeneous?     |
   |  free text          -> what is inside?      |
   |  metadata           -> what travels with it?|
   +----------------------+----------------------+
                          |
                          v
              MATCH  ->  IDENTITY
              INFER  ->  ATTRIBUTE
              DIFFER ->  PRESENCE IN DATA

   Control levers: suppress | generalize | perturb | mask | tokenize | restrict
```

The left side of the diagram is the part teams forget. De-identification is not a property of the
released file alone; it is a property of the file relative to an adversary's knowledge. The same
file can be adequately protected against a journalist with a name and a ZIP code and inadequately
protected against a data broker with a commercial people-file. This is why the HIPAA expert
determination asks about "the anticipated recipient" and "reasonably available information", and
why the GDPR standard refers to means reasonably likely to be used.

A second model is the hierarchy of identifiability:

```text
identified          name, MRN, full date of birth, address
   |
pseudonymised       keyed code; linkage possible with the key or additional data
   |
de-identified       direct identifiers removed or transformed under a standard
   |                (HIPAA Safe Harbor, expert determination); residual risk remains
   |
anonymous           re-identification not reasonably possible for any actor,
   |                considering all means reasonably likely to be used (GDPR)
   |
aggregate           counts and statistics; small cells still disclose
```

The arrows are not automatic. Moving from identified to pseudonymised is an engineering step.
Moving from pseudonymised to de-identified is a standard-based assessment. Moving from
de-identified to anonymous is a legal and statistical conclusion that requires evidence about the
recipient and the environment. Most organisations stop at de-identified and describe it as
anonymous, which is the central error this topic exists to correct.

Three properties follow from the model.

**Risk is relative to a release, not to a dataset.** The same table released publicly, shared under
a data use agreement, or analysed in place has three different risk profiles. The determination
must name the release mode.

**Utility and privacy trade along a curve, not a single point.** Generalizing age to decades
reduces uniqueness and destroys the ability to study age-specific effects. Suppressing rare
diagnoses protects a few patients and biases every rate computed from the remainder. The choice is
a decision about which analyses matter, made with the analyst, not a technical default.

**Residual risk is managed, not eliminated.** Even a well-protected release can be re-identified
by a determined actor with unusual auxiliary data. The controls that remain are contractual (a
DUA with a re-identification prohibition), technical (secure environment, output review), and
legal (a documented determination and a response plan). A de-identification programme that does
not name its residual controls is incomplete.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Direct identifier | A value that identifies a person on its own, such as name, MRN, email, or phone. | Treated as the only kind of identifier. | Field class. |
| Quasi-identifier | A value that is not unique alone but can be combined with others to single out a record, such as age, sex, ZIP, dates. | Called "non-identifying" because it is not a name. | Field class; measured by k. |
| Sensitive attribute | A value whose disclosure is harmful, such as diagnosis, HIV status, or genetic variant. | Confused with quasi-identifier. | Field class. |
| De-identification | Removal or transformation of identifiers under a specified standard, with residual risk remaining. | Treated as anonymisation. | HIPAA standard; legal category. |
| Anonymisation | The result of processing such that re-identification is not reasonably possible for any actor. | Asserted because a key was deleted. | GDPR Recital 26; legal conclusion. |
| Pseudonymisation | Replacing identifiers with a code while retaining a linkage mechanism separately. | Treated as anonymous. | GDPR Art. 4(5); still personal data. |
| Safe Harbor | The HIPAA method that removes the 18 identifier categories and requires no actual knowledge of re-identifiability. | Treated as sufficient for GDPR. | 45 CFR 164.514(b)(2). |
| Expert determination | The HIPAA method in which a qualified expert determines that re-identification risk is very small and documents the method. | Treated as a rubber stamp. | 45 CFR 164.514(b)(1). |
| Actual knowledge | Knowledge that the remaining information could identify an individual, which defeats Safe Harbor. | Ignored in practice. | HIPAA standard. |
| Limited data set | PHI with 16 direct identifiers removed but dates and geography retained. | Called de-identified. | 45 CFR 164.514(e). |
| k-anonymity | Every combination of quasi-identifiers appears for at least k records. | Treated as sufficient protection. | Integer k; per release. |
| l-diversity | Each equivalence class contains at least l distinct sensitive values. | Assumed to follow from k. | Integer l; per class. |
| t-closeness | The distribution of a sensitive attribute in each class is within t of the overall distribution. | Rarely implemented. | Distance t; per class. |
| Equivalence class | A set of records sharing the same quasi-identifier values. | Called a "group" without definition. | Count of records. |
| Prosecutor risk | Probability that a target known to be in the data is correctly matched, often 1/k. | Reported as the only risk. | Probability per record. |
| Journalist risk | Probability that any record can be re-identified given the release, often the proportion in small classes. | Ignored. | Probability or proportion. |
| Marketer risk | Expected proportion of the release that can be correctly re-identified, often the number of classes divided by records. | Ignored. | Proportion. |
| Entropy l-diversity | Shannon entropy of the sensitive attribute within a class, in bits. | Reported without stating the base. | Bits. |
| Generalization | Replacing a value with a broader category, such as age 34 to 30-39. | Confused with suppression. | Transformation. |
| Suppression | Removing a value or a record entirely. | Used without counting the utility cost. | Transformation. |
| Perturbation | Adding noise or swapping values to break exact matching. | Treated as anonymisation. | Transformation. |
| Tokenization | Replacing an identifier with a surrogate, usually via a keyed function. | Assumed irreversible. | Transformation; HMAC common. |
| HMAC | Hash-based message authentication code: a keyed hash used to produce deterministic tokens. | Plain hash called secure. | Cryptographic primitive. |
| Salt | A random value added before hashing to prevent precomputed attacks. | Omitted, allowing dictionary attacks. | Cryptographic parameter. |
| Adversary model | A description of who might attack, with what data and resources. | Left implicit. | Assessment input. |
| Auxiliary data | External information an adversary can use to link or infer. | Assumed unavailable. | Assessment input. |
| Re-identification | Recovery of identity or sensitive attribute from released data. | Used only for deliberate attacks. | Event. |
| Singling out | Isolating one individual in a dataset. | Confused with naming them. | Privacy harm category. |
| Linkability | Learning that two records, possibly in different datasets, refer to the same person. | Ignored in release review. | Privacy harm category. |
| Inference | Learning a sensitive attribute without identifying the person. | Treated as acceptable. | Privacy harm category. |
| Homogeneity attack | Re-identification of a sensitive value because all records in a class share it. | Missed by k-anonymity checks. | Attack. |
| Background knowledge attack | Using outside knowledge to narrow a class to one person. | Ignored in k-only assessments. | Attack. |
| Anonymisation domain | The set of actors from whom attribution must be precluded, per EDPB Guidelines 01/2025. | Not defined, so "anonymous" is undefined. | Assessment scope. |
| Data use agreement | Contract restricting use and redisclosure of shared data. | Treated as a technical control. | Contract. |

## 5. The mechanics, step by step

The procedure below produces a de-identification determination: a documented decision that a
specific dataset, released to a specific recipient through a specific channel, carries residual
re-identification risk that is acceptable for the stated purpose. It is deliberately longer than
"remove the 18 fields", because the 18-field removal is only one route through it.

### Step 1. Define the release purpose, recipient, and adversary

**Input:** A request to share data, the purpose statement from Topic 01, and the recipient profile.
**Output:** A release statement and an adversary model with three named adversaries.
**Rationale:** De-identification is relative to an actor. The release statement fixes the mode
(public download, restricted transfer, in-place analysis), the recipient type (academic, commercial,
internal), and the purpose. The adversary model names who might attempt re-identification and what
they hold: a curious journalist with public records, a commercial data broker with a people-file, a
motivated insider with access to the source system, and an academic with a linkage dataset.

A practical model uses three adversaries:

| Adversary | Auxiliary data | Goal | Typical control |
|---|---|---|---|
| Curious analyst | Public voter file, census, search engines | Find one person | Quasi-identifier generalisation |
| Commercial broker | Purchased people-file, marketing data | Build a linked profile | Contract, access restriction, perturbation |
| Motivated insider | Source system access, key material | Confirm a suspicion about a known person | Key separation, audit, access control |
| Academic linker | Another research dataset with overlapping subjects | Demonstrate linkage | DUA, secure environment, output review |

If the release is public, assume the first two adversaries. If the release is to a commercial
partner, add the broker. If the recipient operates the source system, add the insider.

Branch point: if the purpose requires record-level data for an unlimited period to an unrestricted
recipient, stop. That is a publication decision, and it needs the strongest method, not the
weakest.

### Step 2. Inventory direct identifiers

**Input:** Data dictionary, sample records, file metadata.
**Output:** A direct-identifier inventory mapped to the HIPAA categories and to the GDPR concept
of identifiers.
**Rationale:** Direct identifiers are the easy part, and they are still routinely missed because
they appear in places the data dictionary does not cover: file names, column comments, free-text
fields, DICOM headers, PDF attachments, and spreadsheet properties. The inventory should be built
from a sample of actual records, not only from the schema.

The HIPAA Safe Harbor list has 18 categories. The categories that cause the most implementation
errors are:

1. Names.
2. Geographic subdivisions smaller than a state, including street address, city, county, precinct,
   and ZIP code. The first three digits of a ZIP code may be retained only if the area contains
   more than 20,000 people and the three digits are not in the restricted list; otherwise the
   value becomes 000.
3. All elements of dates (except year) related to birth, admission, discharge, death, and ages over
   89. Ages over 89 may be aggregated into a single category of 90 or older.
4. Telephone numbers.
5. Fax numbers.
6. Email addresses.
7. Social Security numbers.
8. Medical record numbers.
9. Health plan beneficiary numbers.
10. Account numbers.
11. Certificate and license numbers.
12. Vehicle identifiers and serial numbers, including license plates.
13. Device identifiers and serial numbers.
14. Web uniform resource locators (URLs).
15. Internet Protocol (IP) addresses.
16. Biometric identifiers, including fingerprints and voice prints.
17. Full-face photographs and comparable images.
18. Any other unique identifying number, characteristic, or code, except a re-identification code
    that meets the conditions in 45 CFR 164.514(c).

The re-identification code exception is narrow: the code must not be derived from or related to
the individual's information, must not be translatable to identify the individual, must not be
used for any other purpose, and the mechanism must not be disclosed. A hash of the MRN without a
secret key does not meet the "not derived from" condition in practice, because the input space is
small and the hash can be recomputed.

Branch point: if any of the 18 categories cannot be removed without destroying the analysis (for
example, full dates are essential for a survival analysis), Safe Harbor is unavailable and the
release moves to expert determination or to a limited data set with a data use agreement.

### Step 3. Inventory quasi-identifiers

**Input:** Remaining fields after direct-identifier removal.
**Output:** A quasi-identifier list with granularity and a uniqueness estimate.
**Rationale:** Quasi-identifiers are where de-identification actually succeeds or fails. The
classic demonstration is that date of birth, sex, and five-digit ZIP code can uniquely identify a
large share of a population; the HHS guidance notes that year of birth, sex, and three-digit ZIP
is unique for roughly 0.04 percent of US residents, while full date of birth, sex, and five-digit
ZIP is unique for more than 50 percent. The difference between those two statements is the entire
job.

For each candidate field, record the granularity available, the granularity needed, and the
estimated number of distinct values in the release. Fields with high cardinality relative to the
release size are the dangerous ones: dates, ZIP codes, ages in single years, rare diagnoses, device
identifiers, and facility codes. A release of 500 records with 480 distinct admission dates has an
obvious problem regardless of what else is removed.

Branch point: if the release is small (fewer than a few thousand records), any high-cardinality
field is effectively a direct identifier. Small releases need suppression or aggregation, not
clever generalization.

### Step 4. Choose the standard and the route

**Input:** Release statement, identifier inventories, jurisdiction, contractual context.
**Output:** One of six routes, with the rationale recorded.
**Rationale:** The route determines what evidence is required. Safe Harbor requires a documented
removal of the 18 categories and no actual knowledge. Expert determination requires a qualified
expert, a stated method, measured risk, and documentation. GDPR anonymisation requires an
assessment of means reasonably likely to be used, including singling out, linkability, and
inference. Pseudonymisation requires key separation and a legal analysis that keeps the data in
scope. A limited data set requires a DUA and retains dates and geography. No release is a valid
route when the purpose cannot be met safely.

| Route | Legal basis | What must be removed | Evidence required | Fails when |
|---|---|---|---|---|
| Safe Harbor | HIPAA de-identification | The 18 categories, with ZIP and age rules | Checklist, field mapping, actual-knowledge review | Dates or geography are essential; free text is included |
| Expert determination | HIPAA de-identification | Determined by the expert | Method, adversary model, measured risk, documentation | No qualified expert; method not reproducible |
| Limited data set + DUA | HIPAA permission | 16 direct identifiers | DUA, recipient identification | Recipient is commercial and purpose is product development |
| Pseudonymisation | GDPR (still personal data) | Direct identifiers replaced, key separated | Key management, attribution analysis, DPIA | Key is held by the recipient |
| Anonymisation | GDPR (outside scope) | Whatever is needed to make re-identification not reasonably possible | Anonymisation assessment, motivated-intruder test, residual risk | Any actor can link with reasonably available data |
| No release | Not applicable | Not applicable | Decision record | Purpose can be met another way |

The table is not a ladder. A limited data set is not "less anonymous" than Safe Harbor in every
respect; it is a different legal instrument with different obligations. The right question is which
instrument matches the purpose and the jurisdiction.

### Step 5. Transform the fields

**Input:** Field list, chosen route, utility requirements.
**Output:** A transformation specification per field.
**Rationale:** Four transformation families cover most releases. Suppression removes a field or a
record. Generalization replaces a value with a broader category. Perturbation adds noise, swaps
values, or microaggregates. Masking and tokenization replace identifiers with surrogates.

| Transformation | Example | Utility effect | Privacy effect | Watch for |
|---|---|---|---|---|
| Suppression (field) | Drop free-text notes | Loses narrative context | Removes an uncontrolled identifier channel | Analysis may become impossible |
| Suppression (record) | Remove 3 records in a rare class | Biases rates; loses rare cases | Removes singleton classes | Totals no longer reconcile |
| Generalization | Age 34 to 30-39; ZIP 02134 to 021 | Preserves coarse trends | Reduces uniqueness | Threshold choices leak information |
| Date shifting | Shift all dates by a per-patient offset | Preserves intervals | Breaks absolute-date linkage | Offsets must be consistent per patient |
| Perturbation | Add noise to lab values | Preserves distributions approximately | Breaks exact matching | Noise can be reversed with repeated releases |
| Tokenization | Replace MRN with HMAC token | Preserves linkage within the release | Removes the direct identifier | Key management and dictionary attacks |
| Aggregation | Publish counts by region | Preserves population statistics | Removes record level | Small cells still disclose |

Date shifting deserves a note. Shifting each patient's dates by a random offset preserves the
intervals that clinical analyses need while breaking the absolute dates that link to public
records. It fails if the offset is constant across releases, if the shift is applied only to some
tables, or if the shift range is small enough to be guessed. The specification must state the
offset distribution, whether it is per patient, and whether it is reused across releases.

Tokenization deserves a second note. A plain hash of an MRN is not a privacy control, because the
space of MRNs is small and an attacker can hash every candidate. A keyed hash (HMAC) with a secret
key held outside the release is better, but it is still pseudonymisation: anyone with the key, or
with the ability to query the tokenization service, can re-link. Deterministic tokens also enable
linkage across datasets, which is useful for research and dangerous for privacy.

Branch point: if the analysis needs exact values for a field that is also a quasi-identifier, the
release probably needs a secure environment rather than a transformed extract.

### Step 6. Measure disclosure risk

**Input:** Transformed dataset, adversary model.
**Output:** k, l, entropy, and the three risk measures, with the class table.
**Rationale:** Measurement makes the residual risk visible and gives the reviewer something to
challenge. k-anonymity answers "how many records share this quasi-identifier combination". It does
not answer "how varied are the sensitive values", which is what l-diversity and entropy add. None
of them answer "what can an adversary infer from background knowledge", which is why t-closeness
and differential privacy exist.

The measures are defined as follows, for a release of N records partitioned into classes C:

- k = the minimum class size. A record in a class of size k has prosecutor risk at most 1/k if the
  target is known to be in the data.
- l = the minimum number of distinct sensitive values in any class. l = 1 means at least one class
  reveals the sensitive value with certainty.
- Entropy of a class = negative sum over values of p log2 p. Zero entropy means no uncertainty.
- Prosecutor risk = 1/k for the smallest class, or per class if reported per record.
- Journalist risk = the proportion of records in classes below a threshold, or the probability that
  a random record can be singled out.
- Marketer risk = the expected proportion of the release that can be correctly matched, which for
  a release with no other information is the number of classes divided by N.

The measures conflict. Increasing k by generalizing reduces precision and can merge classes with
different sensitive distributions, lowering l. Suppressing records to raise l reduces N and can
lower k in the remaining classes. There is no single number that resolves the trade-off; the
determination states which measure the purpose requires and why.

Branch point: if the release is longitudinal, compute the measures per time slice and across the
linked record. A dataset that is k=5 at each visit can be k=1 when visits are linked.

### Step 7. Inspect free text

**Input:** Any free-text field in the release, or a decision to exclude it.
**Output:** Either exclusion, or a de-identified text with a measured residual identifier rate.
**Rationale:** Free text is the highest-risk field class because it has no schema. Names,
addresses, phone numbers, family relationships, employers, and rare events appear in narrative.
Pattern-based detection catches phone numbers, emails, MRNs, and some dates; it misses names,
nicknames, and indirect identifiers. A measured residual rate requires manual review of a sample
and an estimate of recall, as in Topic 01.

The decision hierarchy is: exclude the field; replace it with a structured variable; de-identify
and review; or analyse in place. Sending raw notes to an external recipient is rarely justified.

Branch point: if the recipient insists on raw notes, the release is identifiable for practical
purposes and needs the identifiable-data controls, not a de-identification label.

### Step 8. Inspect metadata and residual channels

**Input:** The actual release artefact, not the schema.
**Output:** A metadata inventory and a stripping procedure.
**Rationale:** Metadata travels with files: document author and last-modified-by, file names,
folder paths, PDF properties, DICOM headers, spreadsheet hidden sheets, comments, and embedded
objects. A CSV can carry a header comment; a Parquet file can carry a schema with field
descriptions; a zip can carry file timestamps. Each is a channel.

The test is to open the artefact as the recipient would and inspect everything that is not a data
cell. If the release is generated by a pipeline, the pipeline should strip metadata by construction,
and a test should assert that the output contains no author, no original path, and no source
identifier in the file name.

### Step 9. Attempt linkage in a safe environment

**Input:** Transformed dataset, a plausible auxiliary dataset, a controlled environment.
**Output:** A linkage experiment result: match rate, unique match rate, and the fields that drive
matching.
**Rationale:** Measurement is a model; linkage is an observation. Where a plausible auxiliary
dataset exists (public voter file, census aggregates, another research release), a linkage
experiment in a secure environment tests whether the transformation actually breaks matching.
The experiment must itself be governed: synthetic or minimised data, no re-identification of real
people, and no retention of matched records.

Branch point: if linkage produces a unique match for a nontrivial share of records, the release
fails regardless of the k value, because the adversary model was wrong.

### Step 10. Document the determination and the residual controls

**Input:** Outputs of Steps 1 to 9.
**Output:** A determination document with method, measurements, residual risk, controls, and
approval.
**Rationale:** The document is the artefact that survives staff turnover and supports the release
in a regulatory review. It must state the standard used, the fields removed or transformed, the
adversary model, the measurements, the known limitations, the residual controls, the approval, and
the review date. An expert determination must also identify the expert and the qualifications that
make the determination credible.

Residual controls are part of the determination, not an afterthought: a DUA with a
re-identification prohibition, access restrictions, output review, a prohibition on linking, and a
breach notification clause. The determination should state which controls are technical and which
are contractual, because a reviewer will ask what happens if the contract is breached.

### Step 11. Re-assess on change

**Input:** Release inventory, change log.
**Output:** A re-assessment trigger and a new determination where needed.
**Rationale:** De-identification degrades over time. New public datasets appear, the release is
combined with another release, the recipient changes, the population changes, and the same data
released twice allows differencing. A determination has a date and an owner, and it is re-run when
any of those change.

### Edge cases

**Small cells.** A table of counts with a cell of one or two discloses. Suppression of the small
cell alone is often insufficient, because the total allows the value to be recovered by
subtraction; secondary suppression is needed. The rule should be set before the analysis, not after
the table is seen.

**Quasi-identifiers.** The combination matters more than any single field. Age, sex, ZIP, dates,
race, facility, and rare diagnosis interact. A release that is safe on each field alone can be
unique on the combination. Measure the combination, not the columns.

**Re-identification through linkage.** Linkage is the most common real attack: the dataset is
joined to public records or to another release. The defence is to assume linkage and measure it,
not to assume the recipient will not try.

**Free-text PHI.** Names in narrative, family members' details, and rare events are the failure
mode that regex cannot close. Exclude, replace, or review manually.

**Metadata leakage.** The file, not the table, is the release. Strip properties, rename files,
remove paths, and test the artefact.

**Consent withdrawal and de-identified data.** Withdrawal obligations may still apply to
de-identified data where the data can be linked back, and the original consent may promise
deletion. The determination should state whether a withdrawal can be honoured after release; if it
cannot, the consent instrument should have said so.

**Minors and deceased patients.** De-identification does not resolve the special protections
around minors or the interests of relatives in genetic data. A de-identified paediatric release
can still harm a child if the family is identifiable, and a de-identified genomic release can
reveal information about parents and siblings.

**Cross-border transfer.** De-identification may support a transfer analysis but does not
automatically make a transfer lawful, and the recipient country's re-identification capabilities
are part of the assessment. An anonymous dataset under GDPR falls outside Chapter V; a
pseudonymised one does not.

**Repeated releases.** Two releases of overlapping data allow differencing: a record present in
one and absent in the other, or a value that changes between them, can be isolated. The
determination should treat the programme of releases, not a single file, as the unit of analysis.

**Genetic data.** Genomic data are special because a released variant file can reveal information
about relatives who never consented, and because a person's genome is a stable identifier across
datasets. Removing names and dates does not make a variant file anonymous: a modest number of
rare variants can single out an individual, and a genealogy database can link a distant cousin.
The controls are aggregation (allele frequencies rather than genotypes), controlled access with a
data use agreement, and explicit consent that covers relatives where the jurisdiction requires it.
A de-identification determination for genomic data that treats it like a clinical table is
incomplete.

**Actual knowledge.** Safe Harbor is defeated if the covered entity knows the remaining data could
identify an individual. A release that is technically compliant but includes a record the team
knows is the only patient with a condition in a small town is not Safe Harbor.

## 6. Worked example

### Setting

A specialty clinic wants to release a record-level extract to a university group studying
treatment adherence in serious mental illness and diabetes. The extract covers 16 patients treated
in 2025. The fields are patient ID (internal), age in years, sex, five-digit ZIP code, admission
date, and primary diagnosis coded as T2D (type 2 diabetes), HIV, or DEP (depression). The clinic
proposes to remove the patient ID and generalise age to ten-year bands, ZIP to three digits, and
admission date to year. The question is whether the resulting release is adequately protected.

### Step 1: the release and the adversary

The release is record-level, to a university group, under a data use agreement that prohibits
re-identification and linking to other datasets. The adversaries are a curious analyst with public
records and an academic linker with another research dataset. The purpose requires diagnosis and
coarse demographics, but not exact dates.

### Step 2: direct identifiers

The patient ID is removed. No name, MRN, phone, email, or device identifier is present. At the
level of direct identifiers, the release is clean. This is the point at which a checklist-based
review would stop, and it is the point at which the interesting analysis begins.

### Step 3: the generalised table and the class structure

The 16 records generalise into four equivalence classes:

| Class | Age band | Sex | ZIP3 | Year | n | Diagnoses | Distinct | Entropy (bits) |
|---|---|---|---:|---:|---:|---|---:|---:|
| A | 30-39 | F | 021 | 2025 | 5 | T2D x3, HIV x1, DEP x1 | 3 | 1.371 |
| B | 40-49 | M | 021 | 2025 | 4 | HIV x2, T2D x1, DEP x1 | 3 | 1.500 |
| C | 50-59 | F | 021 | 2025 | 4 | T2D x4 | 1 | 0.000 |
| D | 60-69 | M | 036 | 2025 | 3 | DEP x3 | 1 | 0.000 |

### Step 4: the measurements

- k = minimum class size = 3 (class D).
- l = minimum distinct diagnoses per class = 1 (classes C and D).
- Records in classes with l = 1: 4 + 3 = 7, which is 7 / 16 = 43.75 percent of the release.
- Prosecutor risk for a target known to be in class D: 1 / 3 = 0.333.
- Prosecutor risk for a target in class C: 1 / 4 = 0.250.
- Marketer risk: four classes / 16 records = 0.250. The interpretation is that an adversary who
  can place a record in its class can match 25 percent of the release on average.
- Entropy: classes A and B carry 1.371 and 1.500 bits of uncertainty about diagnosis; classes C
  and D carry zero.

The k value of 3 passes a common internal threshold of k >= 3. The l value of 1 fails any
l-diversity threshold above 1. This is the first diagnostic that the protection is inadequate:
a rule based on k alone would approve the release, and every patient in classes C and D would have
their diagnosis disclosed with certainty. For class C, the disclosure is "this patient has type 2
diabetes", which may be less sensitive than HIV, but it is still a health disclosure. For class D,
the disclosure is depression in a small ZIP area.

### Diagnostic 1: homogeneity in two classes

The homogeneity attack does not need names. An adversary who knows that a 62-year-old man from ZIP
036 was treated at the clinic in 2025 looks up class D and learns that he has depression, because
all three records in the class do. The probability of correct inference is 1.0, not 1/k. The
k-anonymity measure never sees this, because it counts records, not values.

The clinic's own purpose makes this worse: the study is about serious mental illness, so the
homogeneous classes are precisely the cases the researchers want. A release that protects them by
suppression removes the study's subject matter.

### Diagnostic 2: a ZIP code below the Safe Harbor threshold

ZIP3 036 is one of the 17 three-digit prefixes that the HHS guidance requires to be recoded to 000,
because the area contains 20,000 or fewer people. The clinic's local planning data put the
population at 12,400. The three patients in class D therefore sit in a geographic area small enough
that the ZIP itself is a strong identifier, independent of the age and sex combination. This is a
second, independent failure: even if l-diversity were repaired, the geography would still violate
the Safe Harbor rule.

The general point is that the two diagnostics catch different things. The l-diversity check catches
attribute disclosure within a class. The ZIP check catches a violation of the standard's
geographic rule. A release can fail either one while passing the other.

### The suppression option and its cost

Suppressing the two homogeneous classes produces a release of 9 records with two classes:

| Class | Age band | Sex | ZIP3 | n | Distinct | Entropy |
|---|---|---|---:|---:|---:|---:|
| A | 30-39 | F | 021 | 5 | 3 | 1.371 |
| B | 40-49 | M | 021 | 4 | 3 | 1.500 |

The measurements improve: k = 4, l = 3, prosecutor risk 1/4 = 0.250, marketer risk 2/9 = 0.222,
zero records in low-l classes. The cost is 7 / 16 = 43.75 percent of the records, and the records
removed are all of the depression-only cases and all of the small-area geography. The surviving
dataset cannot answer the study's question about depression, because the depression cases are
gone, and it cannot support any conclusion about the 60-69 age group. A 43.75 percent suppression
rate is not a technical detail; it is a different study.

This is the utility wall that most de-identification projects hit. The transformations that protect
the rare, sensitive records are the ones that remove them from the analysis. At that point the
honest options are: change the release mode to in-place analysis under a DUA, collect more records
so that the rare classes are no longer rare, or accept a weaker protection with stronger
contractual and technical controls. The dishonest option is to publish the k=3 table and call it
de-identified.

### Vary one assumption: the study needs seasonality

The researchers ask for admission month, because they hypothesise that adherence varies by season.
The clinic adds the month to the generalisation. The class structure becomes:

- Class A splits into 2, 2, and 1 records.
- Class B splits into 2, 1, and 1 records.
- Class C splits into 1, 2, and 1 records.
- Class D splits into 1, 1, and 1 records.

The measurements change to:

- 12 classes, k = 1, l = 1.
- Singleton records: 8 of 16 = 50 percent.
- Prosecutor risk for a singleton: 1/1 = 1.0.
- Marketer risk: 12/16 = 0.75.
- Records in classes with fewer than three distinct diagnoses: 16/16 = 100 percent.

The conclusion changes completely. With month included, half the records are unique on the
quasi-identifiers, and an adversary with a candidate list can confirm presence and diagnosis for
those records with certainty. The release is not de-identified in any useful sense. The earlier
conclusion ("acceptable if the homogeneous classes are suppressed") was conditional on the
generalisation; changing one field invalidates it.

The arithmetic also shows why "add a field and re-check" must be a rule. Each additional
quasi-identifier multiplies the cell space. Sixteen records spread over four classes can absorb a
new field badly; the same sixteen records spread over twelve classes cannot.

### The corrected design

The clinic has three viable designs, none of which is a record-level release of the original table:

1. **Aggregate release.** Publish counts by age band, sex, and diagnosis, with cells below 5
   suppressed. The seasonality hypothesis cannot be tested, but the descriptive question can.
2. **In-place analysis.** The researchers analyse the identified data inside the clinic's
   environment under a DUA, with output review. Seasonality is available, and no record-level data
   leave.
3. **Synthetic release.** Generate a synthetic dataset that preserves the joint distribution, test
   it for disclosure, and release it with the caveat that it is synthetic. This is Topic 03
   territory, and it has its own failure modes.

The determination records design 2 as the approved route, with design 1 as a fallback if the
researchers cannot work in place. The rationale is explicit: the purpose requires month-level
timing and rare diagnoses, the record-level release cannot provide both at acceptable risk, and
the in-place environment provides both without a release.

### What the example demonstrates

The arithmetic is small enough to check by hand and the pattern generalises. k-anonymity alone
approved a table that disclosed two diagnoses with certainty. l-diversity alone would have missed
the ZIP rule. Adding one field destroyed the protection. Suppression repaired the measures and
destroyed the study. The correct answer was not a better transformation of the same release; it
was a different release mode.

### Feasibility: why no transformation rescues 16 records

Before choosing another transformation, it is worth checking whether any transformation could
satisfy the target. Suppose the clinic adopts k >= 5 and l >= 3. A class with l >= 3 must contain
at least three distinct diagnoses, which means at least one HIV record, because only T2D, HIV, and
DEP appear. The release contains 3 HIV records, so at most three classes can reach l = 3. Each
such class must contain at least 5 records to reach k = 5, so the three classes need at least 15
records. The release has 16, which leaves one spare record. In principle, three classes of 5, 5,
and 6 could satisfy both targets.

The quasi-identifier structure prevents it. The available fields after generalisation are age band,
sex, and ZIP3. There are four age bands, two sexes, and two ZIP3 values, but the populated
combinations are only four: (30-39, F, 021), (40-49, M, 021), (50-59, F, 021), and (60-69, M, 036).
The three HIV records are all in 021, in classes A and B. To create three classes that each contain
an HIV record, the clinic would have to split class A or add a field. Splitting class A requires a
field that distinguishes its members, and every available field at a finer granularity (month,
exact age, five-digit ZIP) destroys k. Adding a field and keeping k are therefore in direct
conflict at this sample size.

The arithmetic conclusion is that the release cannot satisfy k >= 5 and l >= 3 for any
transformation of the existing fields. The options are to collect more records so that the HIV and
depression classes are larger, to change the release mode, or to accept a lower target with
stronger residual controls. This is a feasibility check, and it should run before a team spends
weeks tuning generalization hierarchies. A target that the data cannot meet is not a target; it is
a reason to change the design.

## 7. Code, query, or calculation

The script below reproduces the measurements in Section 6 and adds the two checks that matter:
the effect of adding a quasi-identifier, and the Safe Harbor ZIP3 rule. It uses only the standard
library. All 16 records are synthetic.

```python
"""k-anonymity, l-diversity and Safe Harbor ZIP checks on 16 synthetic records."""
from collections import Counter, defaultdict
from math import log2

RESTRICTED_ZIP3 = {
    "036", "059", "063", "102", "203", "556", "692", "790", "821", "823",
    "830", "831", "878", "879", "884", "890", "893",
}

RECORDS = [
    {"id": "P01", "age": 31, "sex": "F", "zip5": "02134", "admit": "2025-01-14", "dx": "T2D"},
    {"id": "P02", "age": 34, "sex": "F", "zip5": "02139", "admit": "2025-01-28", "dx": "T2D"},
    {"id": "P03", "age": 36, "sex": "F", "zip5": "02143", "admit": "2025-02-03", "dx": "T2D"},
    {"id": "P04", "age": 38, "sex": "F", "zip5": "02134", "admit": "2025-02-19", "dx": "HIV"},
    {"id": "P05", "age": 39, "sex": "F", "zip5": "02139", "admit": "2025-03-05", "dx": "DEP"},
    {"id": "P06", "age": 41, "sex": "M", "zip5": "02143", "admit": "2025-01-09", "dx": "HIV"},
    {"id": "P07", "age": 44, "sex": "M", "zip5": "02134", "admit": "2025-01-22", "dx": "HIV"},
    {"id": "P08", "age": 47, "sex": "M", "zip5": "02139", "admit": "2025-02-11", "dx": "T2D"},
    {"id": "P09", "age": 49, "sex": "M", "zip5": "02143", "admit": "2025-03-18", "dx": "DEP"},
    {"id": "P10", "age": 51, "sex": "F", "zip5": "02134", "admit": "2025-01-30", "dx": "T2D"},
    {"id": "P11", "age": 54, "sex": "F", "zip5": "02139", "admit": "2025-02-14", "dx": "T2D"},
    {"id": "P12", "age": 57, "sex": "F", "zip5": "02143", "admit": "2025-02-27", "dx": "T2D"},
    {"id": "P13", "age": 59, "sex": "F", "zip5": "02134", "admit": "2025-03-21", "dx": "T2D"},
    {"id": "P14", "age": 61, "sex": "M", "zip5": "03601", "admit": "2025-01-17", "dx": "DEP"},
    {"id": "P15", "age": 64, "sex": "M", "zip5": "03603", "admit": "2025-02-24", "dx": "DEP"},
    {"id": "P16", "age": 68, "sex": "M", "zip5": "03608", "admit": "2025-03-12", "dx": "DEP"},
]


def generalise(records, by_month=False):
    out = []
    for r in records:
        g = {
            "age_band": (r["age"] // 10) * 10,
            "sex": r["sex"],
            "zip3": r["zip5"][:3],
            "period": r["admit"][:7] if by_month else r["admit"][:4],
            "dx": r["dx"],
        }
        out.append((g, r["id"]))
    return out


def classes(generalised):
    groups = defaultdict(list)
    for g, pid in generalised:
        groups[(g["age_band"], g["sex"], g["zip3"], g["period"])].append(g["dx"])
    return groups


def entropy(values):
    n = len(values)
    return max(0.0, -sum((c / n) * log2(c / n) for c in Counter(values).values()))


def report(label, groups):
    sizes = [len(v) for v in groups.values()]
    k_min = min(sizes)
    l_min = min(len(set(v)) for v in groups.values())
    n = sum(sizes)
    low_l = sum(len(v) for v in groups.values() if len(set(v)) < 3)
    singleton = sum(1 for v in groups.values() if len(v) == 1)
    print(f"{label}: records={n}, classes={len(groups)}, k_min={k_min}, "
          f"l_min={l_min}, singleton_records={singleton}")
    print(f"  records in classes with fewer than 3 distinct diagnoses: "
          f"{low_l}/{n} ({low_l / n:.1%})")
    print(f"  prosecutor risk (smallest class): 1/{k_min} = {1 / k_min:.3f}")
    print(f"  marketer risk: {len(groups)}/{n} = {len(groups) / n:.3f}")
    for key, values in sorted(groups.items()):
        print(f"  class {key}: n={len(values)}, distinct={len(set(values))}, "
              f"entropy={entropy(values):.3f} bits, dx={sorted(set(values))}")


if __name__ == "__main__":
    base = generalise(RECORDS)
    report("generalised to age decade, sex, ZIP3, year", classes(base))

    by_month = classes(generalise(RECORDS, by_month=True))
    report("add admission month", by_month)

    # Suppress the two homogeneous classes (age bands 50 and 60).
    kept = [r for r in RECORDS if (r["age"] // 10) * 10 not in (50, 60)]
    report("suppress homogeneous classes", classes(generalise(kept)))
    print(f"suppression cost: {1 - len(kept) / len(RECORDS):.1%} of records")

    flagged = [r["id"] for r in RECORDS if r["zip5"][:3] in RESTRICTED_ZIP3]
    print(f"Safe Harbor ZIP3 flag (restricted list): {flagged} "
          f"({len(flagged)}/{len(RECORDS)} records)")
```

Expected output, verified on Python 3.11:

```text
generalised to age decade, sex, ZIP3, year: records=16, classes=4, k_min=3, l_min=1, singleton_records=0
  records in classes with fewer than 3 distinct diagnoses: 7/16 (43.8%)
  prosecutor risk (smallest class): 1/3 = 0.333
  marketer risk: 4/16 = 0.250
  class (30, 'F', '021', '2025'): n=5, distinct=3, entropy=1.371 bits, dx=['DEP', 'HIV', 'T2D']
  class (40, 'M', '021', '2025'): n=4, distinct=3, entropy=1.500 bits, dx=['DEP', 'HIV', 'T2D']
  class (50, 'F', '021', '2025'): n=4, distinct=1, entropy=0.000 bits, dx=['T2D']
  class (60, 'M', '036', '2025'): n=3, distinct=1, entropy=0.000 bits, dx=['DEP']
add admission month: records=16, classes=12, k_min=1, l_min=1, singleton_records=8
  records in classes with fewer than 3 distinct diagnoses: 16/16 (100.0%)
  prosecutor risk (smallest class): 1/1 = 1.000
  marketer risk: 12/16 = 0.750
  class (30, 'F', '021', '2025-01'): n=2, distinct=1, entropy=0.000 bits, dx=['T2D']
  class (30, 'F', '021', '2025-02'): n=2, distinct=2, entropy=1.000 bits, dx=['HIV', 'T2D']
  class (30, 'F', '021', '2025-03'): n=1, distinct=1, entropy=0.000 bits, dx=['DEP']
  class (40, 'M', '021', '2025-01'): n=2, distinct=1, entropy=0.000 bits, dx=['HIV']
  class (40, 'M', '021', '2025-02'): n=1, distinct=1, entropy=0.000 bits, dx=['T2D']
  class (40, 'M', '021', '2025-03'): n=1, distinct=1, entropy=0.000 bits, dx=['DEP']
  class (50, 'F', '021', '2025-01'): n=1, distinct=1, entropy=0.000 bits, dx=['T2D']
  class (50, 'F', '021', '2025-02'): n=2, distinct=1, entropy=0.000 bits, dx=['T2D']
  class (50, 'F', '021', '2025-03'): n=1, distinct=1, entropy=0.000 bits, dx=['T2D']
  class (60, 'M', '036', '2025-01'): n=1, distinct=1, entropy=0.000 bits, dx=['DEP']
  class (60, 'M', '036', '2025-02'): n=1, distinct=1, entropy=0.000 bits, dx=['DEP']
  class (60, 'M', '036', '2025-03'): n=1, distinct=1, entropy=0.000 bits, dx=['DEP']
suppress homogeneous classes: records=9, classes=2, k_min=4, l_min=3, singleton_records=0
  records in classes with fewer than 3 distinct diagnoses: 0/9 (0.0%)
  prosecutor risk (smallest class): 1/4 = 0.250
  marketer risk: 2/9 = 0.222
  class (30, 'F', '021', '2025'): n=5, distinct=3, entropy=1.371 bits, dx=['DEP', 'HIV', 'T2D']
  class (40, 'M', '021', '2025'): n=4, distinct=3, entropy=1.500 bits, dx=['DEP', 'HIV', 'T2D']
suppression cost: 43.8% of records
Safe Harbor ZIP3 flag (restricted list): ['P14', 'P15', 'P16'] (3/16 records)
```

Three readings of the output. The first block passes k >= 3 and fails l >= 2, which is why both
measures are reported. The second block shows k collapsing to 1 when one field is added, with 8
singletons and marketer risk rising from 0.250 to 0.750. The third block shows the cost of
repairing l-diversity: k and l both improve, and 43.8 percent of the records disappear.

The script is deliberately a checker, not an anonymiser. Production tools such as ARX, sdcMicro, or
the Amnesia library implement generalization hierarchies and optimal suppression, which are search
problems over transformation lattices. The logic to review is the same: which fields, which
granularity, which measure, and what happens when the adversary model changes.


## 8. Data, standards, and regulatory context

De-identification sits at the intersection of a legal safe harbour, a statistical method, and a
contract. The three are often conflated in procurement documents, which is why a release can be
"HIPAA compliant" and still fail a GDPR anonymity test.

### HIPAA: the de-identification standard

45 CFR 164.514(a) defines de-identified health information as information that does not identify an
individual and with respect to which there is no reasonable basis to believe it can be used to
identify an individual. The regulation provides two methods.

**Safe Harbor (164.514(b)(2)).** Remove the 18 categories of identifiers listed in Section 5, and
ensure the covered entity has no actual knowledge that the remaining information could identify an
individual. The rules that cause the most errors are the geographic rule (three-digit ZIP codes may
be retained only for areas with more than 20,000 people and only if the prefix is not on the
restricted list of 17), the date rule (all date elements except year related to birth, admission,
discharge, death, and ages over 89), and the catch-all for any other unique identifying number,
characteristic, or code.

**Expert determination (164.514(b)(1)).** A person with appropriate knowledge and experience in
statistical and scientific principles determines that the risk is very small that the information
could be used, alone or in combination with other reasonably available information, by an
anticipated recipient to identify an individual. The expert must document the methods and results
of the analysis. The regulation does not set a numeric threshold for "very small", and HHS guidance
states that no universal threshold exists. The determination is therefore an argument, not a
lookup.

**Re-identification code (164.514(c)).** A covered entity may assign a code or other means of
record identification to allow re-identification, provided the code is not derived from or related
to the individual's information, is not translatable to identify the individual, is not used for
any other purpose, and the mechanism is not disclosed. A hash of an MRN with no secret key is
derived from the identifier and is vulnerable to a dictionary attack over the small space of MRNs,
so it fails the spirit of the provision even where the letter is arguable.

**Limited data set (164.514(e)).** A limited data set removes 16 direct identifiers but retains
dates and geographic information smaller than a state (except street address). It is PHI, it
requires a data use agreement, and it is not de-identified. The distinction matters because many
teams describe a limited data set as de-identified and then skip the DUA.

### GDPR: anonymisation and pseudonymisation

Recital 26 of the GDPR states that the principles of data protection do not apply to anonymous
information, defined as information that does not relate to an identified or identifiable natural
person, or to personal data rendered anonymous in such a manner that the data subject is not or no
longer identifiable. The recital instructs that account be taken of all the means reasonably likely
to be used, such as singling out, either by the controller or by another person, to identify the
natural person directly or indirectly.

Article 4(5) defines pseudonymisation as processing personal data in such a manner that the data
can no longer be attributed to a specific data subject without the use of additional information,
provided that the additional information is kept separately and is subject to technical and
organisational measures. Pseudonymised data remain personal data.

The EDPB Guidelines 01/2025 on pseudonymisation, adopted in January 2025, make two points that
bear directly on de-identification projects. First, pseudonymised data can remain personal data
even when the additional information is held by a different party, if the means to combine them
are reasonably likely to be used. Second, the assessment needs a defined "pseudonymisation domain"
- the actors from whom attribution must be precluded. That concept is the operational answer to
"anonymous to whom", and it is the question a release determination must answer explicitly.

The GDPR does not provide a safe harbour equivalent to HIPAA Safe Harbor. There is no list of 18
fields whose removal establishes anonymity. The assessment is contextual: the data, the
recipient, the auxiliary data, the state of technology, and the cost of identification all matter.
A release that would be anonymous in one recipient's hands may be pseudonymised in another's.

### Standards and technical guidance

- ISO/IEC 20889:2018, *Privacy enhancing data de-identification terminology and classification of
  techniques*, provides a common vocabulary for suppression, generalization, perturbation, and
  synthetic data. It is a terminology standard, not a compliance route.
- NIST Special Publication 800-188, *De-Identifying Government Datasets* (2016), describes the
  process of de-identification, the role of the expert, and the need to document assumptions. It is
  written for government datasets but the process transfers.
- The UK Information Commissioner's Office publishes anonymisation and pseudonymisation guidance
  organized around the motivated intruder test: would a reasonably competent person, with access to
  public resources, be able to identify the individual? The test is useful because it names a
  realistic adversary rather than an abstract one.
- The NHS England anonymisation standard ISB1523 defines a set of anonymisation techniques for
  health data and is used in UK commissioning datasets. It is a national convention, not EU law.

> **Note:** Regulatory content in this file is educational and is not legal advice. The
> determination of anonymity for a specific release requires counsel and, for HIPAA expert
> determination, a qualified expert.

The classification to keep straight is: Safe Harbor is a legal safe harbour with defined field
removals; expert determination is a legal route whose substance is a statistical argument; GDPR
anonymisation is a legal conclusion with no fixed method; k-anonymity and l-diversity are
statistical properties of a table that support an argument but do not establish a legal result; and
a DUA is a contractual control that manages residual risk after the release. A determination that
cites k-anonymity as if it were a legal safe harbour has confused the categories.

## 9. Industry practice and case evidence

### Documented re-identification results

**Sweeney, 1997.** Latanya Sweeney showed that a de-identified health dataset could be linked to a
public voter roll using five-digit ZIP code, sex, and date of birth, and identified the health
records of a state governor. The often-cited 87 percent figure refers to the share of the US
population estimated to be uniquely identified by those three fields in 1990 census data. The
lesson is not the exact percentage; it is that the linkage used data the recipient could buy or
download.

**AOL search logs, 2006.** AOL released about 20 million search queries from roughly 650,000 users
with user identifiers replaced by numbers. Journalists identified a named user from the content of
the queries within days. The release had removed the direct identifier and retained a
pseudonymous key, which is exactly the pseudonymisation-versus-anonymisation distinction. The
content of the queries, not the identifier, did the disclosing.

**Netflix Prize, 2006-2009.** Netflix released anonymised movie ratings for a recommendation
contest. Narayanan and Shmatikov (2008) showed that the data could be de-anonymised by linking to
public Internet Movie Database ratings, and that the linkage could be strengthened with a small
amount of outside knowledge. The dataset had no names, addresses, or dates; the quasi-identifiers
were rating patterns.

**Rocher, Hendrickx, and de Montjoye, 2019.** Using a generative model, the authors estimated that
99.98 percent of Americans would be correctly re-identified in any dataset using 15 demographic
attributes. The result is important because it addresses incomplete, sampled releases: sampling
does not provide plausible deniability. The paper's own framing distinguishes uniqueness in the
population from correctness of a match, and both are large.

**HHS guidance, current.** The OCR de-identification guidance illustrates the granularity effect:
year of birth, sex, and three-digit ZIP code are unique for roughly 0.04 percent of US residents,
while full date of birth, sex, and five-digit ZIP code are unique for more than 50 percent. The
comparison is the most useful single fact in de-identification training, because it shows that the
same three concepts at different granularities produce a three-orders-of-magnitude difference in
uniqueness.

### What organisations actually do

**Safe Harbor dominates US practice.** The NCVHS 2017 letter to the HHS Secretary observed that
Safe Harbor is the most-used HIPAA de-identification method, while expert determination is less
used and lacks consensus on expert qualifications and on what "very small" means. That pattern is
consistent with the incentives: Safe Harbor is auditable as a checklist, expert determination
requires a judgement that is harder to defend internally.

**Risk-based releases are common in research networks.** Registries and research consortia often
use a combination of generalization, suppression, and a DUA rather than claiming anonymity.
PhysioNet's tiered access model - open, credentialed, controlled - is an example of managing
residual risk through access rather than through transformation alone. This is inference from
published access policies, not a statement about any particular dataset's risk.

**Commercial de-identification is often weakest at the edges.** Vendors that sell "de-identified"
clinical datasets frequently remove structured identifiers and retain dates, free text, or
facility identifiers, because those fields carry analytic value. The buyer's due diligence should
ask for the determination, not the marketing label.

**Aggregation is not a safe default.** Published tables with small cells are a recurring source of
disclosure. The standard controls are minimum cell sizes and secondary suppression, and they are
often applied after the table has been designed, which is why totals can be reconstructed by
subtraction.

### The gap between the standard and the release

Two patterns in practice deserve separate mention because they are structural rather than
individual errors. The first is that de-identification is usually performed once, at the point of
release, by the data provider, while the risk is realized later, in the recipient's environment,
after additional data becomes public. A determination that is correct at release can be wrong two
years later, and there is no mechanism in most contracts for re-assessment. The second is that
the recipient's own data are rarely part of the assessment. A university that holds a cohort with
overlapping patients, or a vendor that has worked with neighbouring health systems, can link
without any public dataset. The determination must therefore ask what the recipient already knows,
not only what the public can obtain.

Both patterns point to the same control: treat the release as a relationship rather than a
transaction. The relationship includes a named recipient, a purpose, a DUA with audit and
re-assessment clauses, an output-review process, and a termination date. A release with no
relationship is a publication, and publication should be assessed as such.

> **Note:** The case summaries above report published results. Statements about what an
> organisation should have done differently are inference, labelled as such where they appear.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Checklist de-identification | The review consists of a field-removal list and nothing else | Quasi-identifiers, free text, and metadata are outside the checklist | Re-identification; regulatory finding | Measure k, l, and linkage; inspect the artefact |
| Free text left in | Notes included "for context" with no review | Narrative contains names and rare events | Direct identifiers in a "clean" release | Exclude, replace, or de-identify with measured recall |
| Dates retained at full precision | Admission and discharge dates present | Dates are high-cardinality quasi-identifiers | Linkage to public records and other releases | Year, interval, or per-patient date shift |
| ZIP code retained below threshold | Five-digit ZIP present; ZIP3 with small population | Geographic area smaller than 20,000 | Safe Harbor violation; singling out | ZIP3 rule; recode to 000; aggregate |
| Ages over 89 in single years | A 97-year-old in the table | Rare age is identifying | Disclosure of a small group | Aggregate to 90 or older |
| k-anonymity only | Report cites k=5 and nothing else | Homogeneity and background knowledge ignored | Attribute disclosure with certainty | Add l-diversity, entropy, t-closeness or DP |
| l-diversity only | Report cites l=3 and nothing else | Geographic and linkage risk ignored | Re-identification through auxiliary data | Combine measures; linkage test |
| Suppression without utility accounting | 40 percent of records removed and no analysis plan | Privacy target met by deleting the study | Biased results; wasted release | State utility cost; choose another release mode |
| Metadata not stripped | File properties show author and original path | Metadata travels with the artefact | Disclosure without a data field | Strip and test the artefact |
| Re-identification code misused | Hash of MRN stored with the release | Small input space; dictionary attack | Reversible pseudonymisation | HMAC with separated key; no key with recipient |
| Repeated releases | Two overlapping extracts published | Differencing isolates records | Disclosure by comparison | Treat the release programme as the unit |
| Actual knowledge ignored | Team knows a record is unique in a small town | Safe Harbor defeated | Non-compliance | Document and escalate known cases |
| Anonymity claimed for pseudonymised data | Key held by the recipient | Legal category error | GDPR applies; unlawful processing | Define the anonymisation domain; legal review |
| Cross-border assumption | "It is de-identified, so transfer is fine" | Transfer law applies to personal data | Unlawful transfer | Transfer analysis; recipient-country risk |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The adversary model covers the plausible attackers | Risk is relative to an actor | A new recipient or a new public dataset | Re-run on recipient or data change |
| The auxiliary data assumed are available | Linkage drives re-identification | Data brokers hold more than assumed | Linkage experiment; broker data inventory |
| The release is the unit, not the dataset | Risk is release-specific | Multiple releases of overlapping data | Inventory releases; differencing test |
| Quasi-identifier list is complete | Measures depend on it | A field added after assessment | Schema diff at release time |
| Generalization is applied consistently | Inconsistent levels leak | One table at age decade, another at year | Test joins across tables |
| Free-text recall estimate is valid | Residual identifier rate depends on it | Different note style or specialty | Second manual sample |
| k and l thresholds match the purpose | Measures are not self-interpreting | Threshold chosen by habit | Document rationale with the purpose |
| Suppression cost is acceptable | Utility is part of the decision | Rare classes are the study subject | Analysis plan review before release |
| The DUA is enforceable | Contract is a residual control | Recipient in a jurisdiction with weak enforcement | Legal review; technical restrictions |
| The expert is qualified | Expert determination depends on it | Expert lacks relevant experience | Credentials and method review |
| The legal classification is correct | Route depends on it | Entity is a consumer app, not a covered entity | Legal review of entity and flow |

Two limits are structural. First, k-anonymity and l-diversity protect against a specified class of
attacks; they do not bound what an adversary can infer from a model trained on the data, which is
why Topic 03 covers membership inference and differential privacy. Second, a determination is a
snapshot. The same release becomes less protected as public data accumulates and as linkage tools
improve, so the review date is part of the control, not a bureaucratic add-on.

## 12. Exercises

**Exercise 1 - Classify the four terms (recall, 20 minutes).**
Define de-identification, pseudonymisation, anonymisation, and synthetic data in one sentence each,
and name the legal regime each falls under. Deliverable: four sentences plus a table. Criteria:
correctly states that pseudonymised data remain personal data; does not treat de-identification as
a legal conclusion of anonymity.

**Exercise 2 - Apply Safe Harbor (application, 45 minutes).**
Given a 25-field extract (invent the fields, including dates, ZIP5, age 94, notes, and a device
serial number), apply Safe Harbor. Deliverable: field-by-field disposition table. Criteria:
correct ZIP3 rule, correct age-over-89 rule, correct date rule, free text addressed explicitly.

**Exercise 3 - Compute k and l (calculation, 40 minutes).**
Using the 16-record table in Section 6, add one record to class D with a T2D diagnosis. Recompute
k, l, entropy, and the three risk measures by hand. Deliverable: calculations with units. Criteria:
class sizes correct; l for class D becomes 2; entropy computed to three decimals.

**Exercise 4 - Design an expert determination (design, 90 minutes).**
Design an expert determination for a release of 5,000 records with dates retained. Specify the
adversary, auxiliary data, method, measured risk, residual controls, and documentation. Deliverable:
a four-page determination outline. Criteria: adversary is specific; method is reproducible;
residual controls distinguish technical from contractual; limitations stated.

**Exercise 5 - Audit a release (critique, 60 minutes).**
Take any public "de-identified" dataset description or data dictionary and identify three residual
risks. Deliverable: a 500-word audit. Criteria: risks are specific to the dataset, not generic;
each has a proposed control; at least one involves metadata or free text.

**Exercise 6 - Free-text screening (application, 60 minutes).**
Run a regex screen over a synthetic note sample (or the six notes in Topic 01) and estimate recall
against manual review. Deliverable: script, sample review sheet, recall estimate. Criteria:
false negatives counted; estimate labelled as sample-based.

**Exercise 7 - Date-shifting specification (design, 45 minutes).**
Write the specification for per-patient date shifting in a longitudinal release. State the offset
distribution, consistency rules, handling of cross-table joins, and the failure modes. Deliverable:
one-page spec. Criteria: offsets are per patient and consistent; reuse across releases addressed;
intervals preserved.

**Exercise 8 - Choose the release mode (design, 60 minutes).**
A study needs diagnosis, month of admission, and age in years for 800 patients. Recommend a release
mode and justify it against the alternatives. Deliverable: decision memo. Criteria: compares at
least three modes; states residual risk; names the control that makes the recommendation
defensible.

## 13. Self-check answers

**Answer 1.** De-identification is the removal or transformation of identifiers under a specified
standard, with residual risk remaining; under HIPAA it is a legal category with two methods.
Pseudonymisation replaces identifiers with a keyed code and remains personal data under GDPR
Article 4(5). Anonymisation is the conclusion that re-identification is not reasonably possible for
any actor considering means reasonably likely to be used, which removes the data from GDPR scope.
Synthetic data are generated records that may preserve statistical properties; they are not
automatically anonymous, because a generative model can leak training records.

**Answer 2.** Safe Harbor requires removal of the 18 categories, with three rules that are easy to
get wrong. First, a five-digit ZIP code is removed; the first three digits may be retained only if
the area has more than 20,000 people and the prefix is not on the restricted list, otherwise the
value becomes 000. Second, all date elements except year are removed for birth, admission,
discharge, and death, and ages over 89 are aggregated into 90 or older. Third, the catch-all
category covers any unique number, characteristic, or code, which is why a device serial number, a
barcode, or a non-secret hash of an MRN must be removed. The method also fails if the entity has
actual knowledge that the remaining data could identify someone.

**Answer 3.** With one T2D record added to class D, class D has n=4, diagnoses DEP x3 and T2D x1,
so l=2 for that class and l_min across the table becomes 1 because class C still has T2D x4.
Class D entropy: p=0.75 and 0.25, so H = -(0.75 log2 0.75 + 0.25 log2 0.25) = 0.75 x 0.415 +
0.25 x 2 = 0.311 + 0.5 = 0.811 bits. k remains 3 (class D now 4, class C 4, class B 4, class A 5,
so k=4 actually; recompute: all classes are at least 4, so k=4). Prosecutor risk 1/4 = 0.25.
Marketer risk 4/17 = 0.235. The exercise shows that adding one record can change k and the risk
measures without changing the underlying homogeneity problem in class C.

**Answer 5.** A good audit names specific fields and artefacts. Examples of acceptable findings: a
"de-identified" dataset that retains admission and discharge dates at day precision, allowing
linkage to a claims dataset with the same dates; free-text discharge instructions containing family
contact details; a Parquet schema with a column description that names the source clinic; a README
that states the dataset covers "all patients at a single Vermont clinic in 2024", which is a
small-population disclosure even without a ZIP code. Generic findings such as "re-identification is
possible" without a mechanism do not earn credit.

**Answer 7.** The specification should state: offsets drawn from a defined distribution (for example
uniform over plus or minus 180 days), applied per patient, consistent across all tables and all
releases for that patient, with the offset stored separately and never released. Intervals within a
patient are preserved; intervals across patients are not comparable. Cross-table joins must use the
patient token, not the shifted date. Failure modes: a constant offset across patients, a small
offset range that can be guessed, re-use of the same offset in a second release with different data,
and shifts that move events across a care boundary in a way that changes the analysis.

**Rubric for Exercise 2.** Full credit requires every one of the 18 categories to be addressed,
the ZIP3 restricted list applied, ages over 89 aggregated, all date elements except year removed,
and free text either removed or explicitly assessed. A table that removes "obvious" identifiers and
leaves dates loses half the marks.

**Rubric for Exercise 4.** Full credit requires a named expert role, a stated adversary, a
reproducible method, measured risk with a threshold and its justification, residual controls, and a
limitations section. A determination that asserts "very small risk" without a measurement does not
pass.

**Rubric for Exercise 6.** Full credit requires a confusion matrix over the sample, an explicit
false-negative count, a recall estimate with its sample size, and a statement that the estimate
does not generalise without a second sample.

**Rubric for Exercise 8.** Full credit requires comparison of at least three release modes,
including in-place analysis and aggregation, a statement of residual risk for the recommended mode,
and a named control. A recommendation of "release de-identified record-level data" without
addressing month-level dates does not pass.

## 14. Interview questions

**1. What is the difference between de-identification and anonymisation?**
Strong answer: de-identification is a standard-based reduction with residual risk; anonymisation is
a conclusion that re-identification is not reasonably possible for any actor, which removes data
from GDPR scope; HIPAA has no anonymisation safe harbour equivalent. Weak answer: uses the terms
interchangeably.

**2. Walk me through Safe Harbor.**
Strong answer: the 18 categories, the ZIP3 rule with the 20,000 threshold and restricted list, the
date rule and age-89 aggregation, the actual-knowledge clause, and the re-identification code
conditions. Weak answer: "remove 18 identifiers".

**3. When can you not use Safe Harbor?**
Strong answer: when dates, geography, or free text are essential, when a code is derived from PHI,
or when the entity has actual knowledge of identifiability. Weak answer: "when the data is small".

**4. What does expert determination require that Safe Harbor does not?**
Strong answer: a qualified expert, a stated adversary and auxiliary data, a reproducible method, a
documented risk measurement, and a justification that the risk is very small; there is no numeric
threshold. Weak answer: "a statistician signs off".

**5. Why is k-anonymity insufficient?**
Strong answer: homogeneity, background knowledge, and linkability; a class of k records can all
share the sensitive value, and k says nothing about auxiliary data. Weak answer: "k should be
higher".

**6. What is l-diversity and what does it miss?**
Strong answer: at least l distinct sensitive values per class; it misses skewness (a class can be
l-diverse but concentrated), similarity attacks, and linkability. Weak answer: "it is k-anonymity
plus diversity".

**7. How do you handle free text in a release?**
Strong answer: exclude by default, replace with structured variables, or de-identify with measured
recall and manual review; regex alone is insufficient. Weak answer: "we run a scrubber".

**8. What is the anonymisation domain?**
Strong answer: the set of actors from whom attribution must be precluded, per EDPB Guidelines
01/2025; it forces the assessment to name the recipient and the auxiliary data. Weak answer: "the
dataset".

**9. How do you test whether a release is re-identifiable?**
Strong answer: measure k, l, and linkage against plausible auxiliary data in a controlled
environment, and inspect the artefact for metadata and free text. Weak answer: "we check for
names".

**10. What would you do if the utility requirement conflicts with the privacy requirement?**
Strong answer: change the release mode (in-place, aggregate, synthetic), collect more data, or
decline; do not weaken the standard and keep the label. Weak answer: "add noise until it passes".

## 15. Key terms

- **Actual knowledge** - knowledge that remaining data could identify a person, defeating Safe Harbor.
- **Anonymisation** - conclusion that re-identification is not reasonably possible for any actor.
- **Anonymisation domain** - the actors from whom attribution must be precluded.
- **Auxiliary data** - external information used to link or infer.
- **Background knowledge attack** - using outside knowledge to narrow a class to one person.
- **Date shifting** - per-patient offset that preserves intervals but breaks absolute dates.
- **De-identification** - removal or transformation of identifiers under a standard.
- **Direct identifier** - a value that identifies a person on its own.
- **Entropy l-diversity** - Shannon entropy of the sensitive attribute within a class.
- **Equivalence class** - records sharing the same quasi-identifier values.
- **Expert determination** - HIPAA route requiring a qualified expert and documented risk analysis.
- **Generalization** - replacing a value with a broader category.
- **HMAC** - keyed hash used for deterministic tokens.
- **Homogeneity attack** - disclosure because all records in a class share a value.
- **Journalist risk** - probability that any record can be re-identified.
- **k-anonymity** - every quasi-identifier combination appears at least k times.
- **l-diversity** - at least l distinct sensitive values per class.
- **Limited data set** - PHI with 16 identifiers removed, dates and geography retained.
- **Linkability** - learning that two records refer to the same person.
- **Marketer risk** - expected proportion of a release that can be matched.
- **Metadata leakage** - disclosure through file properties, headers, or names.
- **Perturbation** - adding noise or swapping values to break exact matching.
- **Prosecutor risk** - probability that a known target in the data is correctly matched.
- **Pseudonymisation** - replacing identifiers while retaining a linkage mechanism.
- **Quasi-identifier** - a value that can single out a record in combination.
- **Re-identification** - recovery of identity or a sensitive attribute.
- **Safe Harbor** - HIPAA method removing the 18 identifier categories.
- **Sensitive attribute** - a value whose disclosure is harmful.
- **Singling out** - isolating one individual in a dataset.
- **Suppression** - removing a field or record.
- **t-closeness** - class distribution within distance t of the overall distribution.
- **Tokenization** - replacing an identifier with a surrogate.

## 16. Sources

All URLs accessed 26 October 2026.

**US regulation and guidance**
- 45 CFR 164.514, *Other requirements relating to uses and disclosures of protected health
  information*. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
- HHS Office for Civil Rights, *Guidance Regarding Methods for De-identification of Protected
  Health Information*. https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- HHS Office for Civil Rights, *Minimum Necessary Requirement*.
  https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/minimum-necessary-requirement/index.html
- NCVHS, *Recommendations on De-identification of Protected Health Information under HIPAA*, 2017.
  https://ncvhs.hhs.gov/wp-content/uploads/2018/03/2017-Ltr-Privacy-DeIdentification-Feb-23-Final-w-sig.pdf
- NIST, *SP 800-188, De-Identifying Government Datasets*, 2016.
  https://csrc.nist.gov/pubs/sp/800/188/final

**EU regulation and guidance**
- Regulation (EU) 2016/679, Recital 26 and Article 4(5).
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- European Data Protection Board, *Guidelines 01/2025 on Pseudonymisation*.
  https://www.edpb.europa.eu/public-consultations/guidelines-012025-on-pseudonymisation_en
- European Data Protection Board, *Guidelines 01/2025 on Pseudonymisation* (PDF).
  https://www.edpb.europa.eu/system/files/2025-01/edpb_guidelines_202501_pseudonymisation_en.pdf

**Standards**
- ISO/IEC 20889:2018, *Privacy enhancing data de-identification terminology and classification of
  techniques*.
- ISO/IEC 27701:2019, *Privacy information management*.
- NHS England, *Anonymisation Standard for Publishing Health and Social Care Data*, ISB1523.

**Peer-reviewed evidence**
- Sweeney, L. *k-anonymity: a model for protecting privacy*. International Journal of Uncertainty,
  Fuzziness and Knowledge-Based Systems, 2002. https://doi.org/10.1142/S0218488502001648
- Machanavajjhala, A., Kifer, D., Gehrke, J., Venkitasubramaniam, M. *l-diversity: Privacy beyond
  k-anonymity*. ACM Transactions on Knowledge Discovery from Data, 2007.
  https://doi.org/10.1145/1217299.1217302
- Li, N., Li, T., Venkatasubramanian, S. *t-Closeness: Privacy beyond k-anonymity and l-diversity*.
  IEEE ICDE, 2007. https://doi.org/10.1109/ICDE.2007.367856
- Narayanan, A., Shmatikov, V. *Robust de-anonymization of large sparse datasets*. IEEE Symposium
  on Security and Privacy, 2008. https://doi.org/10.1109/SP.2008.33
- Rocher, L., Hendrickx, J. M., de Montjoye, Y.-A. *Estimating the success of re-identifications in
  incomplete datasets using generative models*. Nature Communications 10:3069, 2019.
  https://doi.org/10.1038/s41467-019-10933-3
- Sweeney, L. *Weaving technology and policy together to maintain confidentiality*. Journal of Law,
  Medicine and Ethics, 1997. Cited for the governor re-identification case.

**Practical guidance**
- UK Information Commissioner's Office, *Anonymisation and pseudonymisation guidance*.
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/anonymisation/
- PhysioNet, tiered access to datasets. https://physionet.org/

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_privacy-principles.md`](01_privacy-principles.md)
- Next topic: [`03_privacy-enhancing-technologies.md`](03_privacy-enhancing-technologies.md)
- Related topic: [`09_breach-and-incident.md`](09_breach-and-incident.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
