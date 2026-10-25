# De-identification Engineering

> Module 03 — Data Engineering and Interoperability
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: 06, 07
> Version 2026.3

## 1. Why this topic exists

De-identification is the point where data engineering meets law. The engineering question is "which
transformations preserve the analytic signal?" The legal question is "does the result meet the standard
for de-identification?" A pipeline can satisfy one and fail the other.

The failures are common and expensive.

- A team removes names and medical record numbers, then releases a table with age in years, five-digit
  ZIP code, and full dates of service. The combination identifies individuals in small areas, and the
  release is a disclosure even though no name appears.
- A date-shift function applies a random offset per row. The intervals between a patient's diagnosis and
  treatment change, and a survival analysis produces a different hazard ratio than the source data.
- A free-text de-identifier replaces names with `[NAME]` but leaves dates, phone numbers, and a facility
  name. The note is still identifiable, and the replacement gave a false sense of completeness.
- A crosswalk from pseudonym to medical record number is stored in the same bucket as the de-identified
  release. Anyone with read access can re-identify every patient.
- A team invokes expert determination without hiring an expert, without documenting a method, and without
  a risk assessment. The phrase appears in a data use agreement and nothing supports it.
- A "de-identified" imaging archive contains burned-in text. The header is clean and the pixels are not.
- A researcher joins a de-identified extract to a public voter file using age, sex, and ZIP. The join
  re-identifies a subset, and the result is published before anyone checks.

The HIPAA Privacy Rule permits two paths: safe harbor, which requires removing 18 categories of
identifiers and having no actual knowledge that the remaining data could identify someone, and expert
determination, which requires a qualified expert to determine that the risk is very small and to document
the methods and results. Both are legal standards with engineering consequences. Neither is satisfied by
a regex.

This topic covers the engineering of de-identification: identifier inventory, transformation operators,
date handling, free text, small-cell risk, k-anonymity, residual risk, crosswalk protection, and
validation. It assumes the reader has read the DICOM topic, because imaging de-identification has its own
failure modes.

> **Note:** This topic is educational and is not legal advice. De-identification determinations require
> privacy counsel and, for expert determination, a qualified expert. The engineering practices here
> support a determination; they do not make one.

## 2. Learning outcomes

By the end of this topic you can:

1. **Inventory** direct and quasi-identifiers in a dataset against the 18 safe-harbor categories and
   classify each as direct, quasi, or sensitive. Assessed in Exercise 1.
2. **Choose** between safe harbor and expert determination for a stated release and justify the choice.
   Assessed in Exercise 2 and self-check item 2.
3. **Apply** transformation operators (remove, tokenize, generalize, shift, suppress) and state the
   utility cost of each. Assessed in Exercise 3.
4. **Implement** per-patient date shifting that preserves intervals and detect boundary inference risk.
   Assessed in Exercise 4.
5. **Compute** k-anonymity for a quasi-identifier set and the effect of generalization and suppression.
   Assessed in Exercise 5.
6. **Evaluate** a free-text de-identifier with precision and recall by identifier type, and state the
   residual risk. Assessed in Exercise 6.
7. **Protect** a crosswalk and design a re-identification test. Assessed in Exercise 7.
8. **Document** a de-identification report with method, validation, residual risk, and acceptance.
   Assessed in Exercise 8.

## 3. The core mental model

De-identification is a transformation of a dataset plus an assessment of the result in a context.

```text
+---------------------+     +----------------------+     +----------------------+
| INPUT               |     | TRANSFORMATION       |     | OUTPUT               |
| direct identifiers  | --> | remove               | --> | de-identified dataset|
| quasi-identifiers   |     | tokenize             |     | + residual risk      |
| sensitive values    |     | generalize           |     | + documentation      |
| free text, pixels   |     | shift                |     |                      |
+---------------------+     | suppress             |     +----------+-----------+
                            +----------------------+                |
                                                                     v
                                                        +----------------------+
                                                        | CONTEXT              |
                                                        | recipient, linkage   |
                                                        | data, environment    |
                                                        +----------------------+
```

The output is not "anonymous" in the abstract. It is de-identified relative to a stated threat model:
who receives it, what auxiliary data they hold, and what linkage attacks are feasible. A release that is
safe for a trusted research environment may be unsafe for a public download.

### 3.1 Three classes of identifier

**Direct identifiers** uniquely identify a person on their own: name, medical record number, Social
Security number, email address, full-face photograph. Removing them is necessary and usually
straightforward.

**Quasi-identifiers** identify in combination: age, sex, ZIP code, dates of service, race, occupation. A
table with these fields can be linked to an external dataset to re-identify a subset. The classic result
is that age, sex, and ZIP can uniquely identify a large share of a population, though the exact share
depends on the population and the granularity.

**Sensitive attributes** are the values the dataset exists to study: diagnosis, lab result, genotype,
medication. They are not identifiers, but their combination with quasi-identifiers can disclose a
condition. A record that is not re-identified to a name can still reveal that a person in a small group
has a stigmatized diagnosis.

The engineering task is to reduce identifiability while preserving the joint distribution of the
sensitive attributes and the analytic variables.

### 3.2 The transformation operators

| Operator | Example | Privacy effect | Utility cost |
|---|---|---|---|
| Remove | Drop name, phone, email | Eliminates direct identifier | Loses contact or linkage |
| Tokenize | Replace MRN with HMAC token | Preserves within-dataset linkage | Crosswalk becomes a target |
| Generalize | Age 34 → 30-34; date → month | Reduces uniqueness | Loses precision |
| Shift | Add a per-patient offset to dates | Preserves intervals | Breaks absolute-date analyses |
| Suppress | Remove cells with count < 11 | Prevents small-cell disclosure | Loses rare cases |
| Perturb | Add noise to a value | Reduces exact matching | Changes results |
| Truncate | ZIP 02139 → 021 | Reduces geographic precision | Loses local analysis |
| Replace | Name → `[NAME]` in text | Removes identifier | Can miss variants |

No operator is free. A release that removes every quasi-identifier is private and useless; a release that
keeps all of them is useful and identifiable. The design is the choice of which utility to sacrifice for
which threat.

### 3.3 Risk is a function of context

Residual risk depends on:

- The quasi-identifier set and its granularity.
- The size of the population and the number of releases.
- The auxiliary data available to the recipient.
- Whether the same patients appear in multiple releases.
- The sensitivity of the attributes.
- The legal and contractual controls on the recipient.

Two datasets with identical columns can have different risk because one is released publicly and the
other is available only inside a trusted research environment with audit and no export.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| PHI | Protected health information under HIPAA | Confused with any health data | 45 CFR 160.103 |
| De-identification | Removing or transforming identifiers so the standard is met | Confused with pseudonymization | 45 CFR 164.514 |
| Safe harbor | Removing 18 identifier categories with no actual knowledge | Assumed to be the only method | 45 CFR 164.514(b)(2) |
| Expert determination | Expert determines risk is very small and documents it | Treated as a formality | 45 CFR 164.514(b)(1) |
| Direct identifier | Identifies a person on its own | Confused with quasi-identifier | — |
| Quasi-identifier | Identifies in combination | Ignored in releases | — |
| Sensitive attribute | Value whose disclosure matters | Confused with identifier | — |
| Pseudonymization | Replacing identifiers with pseudonyms, reversible with a key | Confused with anonymization | GDPR Art. 4(5) |
| Anonymization | Irreversible removal of identifiability | Assumed absolute | GDPR Recital 26 |
| Tokenization | Replacing a value with a keyed token | Confused with hashing without a key | HMAC |
| HMAC | Keyed hash for deterministic tokens | Plain hash treated as secure | RFC 2104 |
| Salt | Random value added before hashing | Confused with a secret key | — |
| Crosswalk | Mapping from token to original identifier | Stored with the release | — |
| Generalization | Reducing precision | Confused with suppression | — |
| Suppression | Removing values or cells | Confused with generalization | — |
| Date shifting | Adding a per-patient offset | Applied per row | Days |
| Date truncation | Reducing to month or year | Confused with shifting | — |
| k-anonymity | Each record is indistinguishable from at least k-1 others on the quasi-identifiers | Treated as sufficient | k |
| l-diversity | Each equivalence class has at least l distinct sensitive values | Confused with k-anonymity | l |
| t-closeness | Distribution of sensitive values is close to the overall distribution | Rarely implemented | t |
| Differential privacy | Formal guarantee on the effect of one record | Claimed without a budget | Epsilon |
| Equivalence class | Records with the same quasi-identifier values | Confused with a group | — |
| Small cell | A count below a disclosure threshold | Assumed safe if not a name | Count |
| Complementary suppression | Suppressing additional cells to prevent back-calculation | Forgotten | — |
| Residual risk | Risk remaining after transformation | Assumed zero | — |
| Re-identification | Linking a record to a person | Confused with pseudonymization | — |
| Linkage attack | Joining with auxiliary data | Ignored | — |
| Auxiliary data | External data used to re-identify | Assumed unavailable | — |
| DUA | Data use agreement | Treated as a technical control | Contract |
| Expert | Person with statistical or scientific expertise | Any staff member | 45 CFR 164.514(b)(1) |
| Actual knowledge | Clear and direct knowledge of identifiability | Confused with suspicion | 45 CFR 164.514(b)(2)(ii) |
| Limited data set | A HIPAA category with dates and geography but no direct identifiers | Confused with de-identified | 45 CFR 164.514(e) |
| Minimum necessary | Limit use to what is needed | Ignored | 45 CFR 164.502(b) |

Three terms need expansion.

**Limited data set.** HIPAA permits a limited data set that may include dates and city, state, and ZIP,
but not direct identifiers, for research, public health, or health care operations, with a data use
agreement. It is not de-identified. A team that conflates the two may release data that the law still
treats as PHI.

**k-anonymity.** A dataset satisfies k-anonymity if every combination of quasi-identifier values appears
at least k times. It protects against unique identification but not against attribute disclosure: if all
k records in a class have the same diagnosis, learning the class reveals the diagnosis. That is why
l-diversity and t-closeness exist.

**Differential privacy.** A formal guarantee that the output distribution changes only slightly when one
record is added or removed, parameterized by epsilon. It is not achieved by adding arbitrary noise or by
removing names. Claiming differential privacy without a stated mechanism and budget is incorrect.

## 5. The mechanics, step by step

The procedure takes a dataset from identifiable to a documented release. It covers both legal paths and
the engineering work that supports them.

### Step 1 — Classify the data and the legal regime

Input: the dataset and its source.
Output: a classification: PHI, limited data set, de-identified, or not health data, with the jurisdiction.

A dataset can be PHI in the United States and personal data under GDPR in the EU. The obligations differ:
HIPAA has a de-identification standard, GDPR has anonymization and pseudonymization concepts and does not
treat pseudonymized data as anonymous.

Reason: the legal regime determines the standard and the documentation. Applying the wrong standard
produces a release that is defensible in one jurisdiction and not another.

### Step 2 — Map the data flow and the recipients

Input: the release plan.
Output: a flow diagram from source to recipient, with each copy and each transformation.

Include the crosswalk, the intermediate files, the backups, the analytics workspace, and the recipient's
environment. A de-identified release with an identifiable intermediate file on a shared drive is not a
de-identified release.

Reason: de-identification is a property of the whole path, not of the final file. Most leaks occur in the
intermediates.

### Step 3 — Inventory identifiers

Input: the schema, the free-text fields, and the images.
Output: an identifier inventory with column, category, example, and direct/quasi classification.

Use the 18 safe-harbor categories as a checklist, then add domain-specific quasi-identifiers: admission
date, discharge date, age over 89, ZIP, facility, provider, rare diagnosis, device identifier, and free
text.

Reason: an inventory that covers only the obvious columns misses dates, ZIP codes, and free text, which
are the categories most often left in releases.

### Step 4 — Choose the de-identification path

Input: the inventory and the intended use.
Output: safe harbor, expert determination, or a limited data set with a DUA.

Safe harbor is simpler to document but removes dates and geography, which may destroy the analysis.
Expert determination preserves utility but requires a qualified expert, a method, and documentation. A
limited data set preserves dates and geography but is not de-identified and requires a DUA.

Reason: the choice determines which transformations are mandatory and which analyses remain possible.
Making it after the analysis is designed usually forces a redo.

### Step 5 — Remove direct identifiers

Input: the columns classified as direct.
Output: a dataset with those columns removed or replaced.

Remove names, MRNs, SSNs, phone and fax numbers, emails, account numbers, certificate and license numbers,
vehicle identifiers, device identifiers, URLs, IP addresses, biometric identifiers, and full-face photos.
Check free-text fields and file paths for the same values.

Reason: direct identifiers are the easiest to remove and the most damaging to leave. A single MRN column
defeats every other transformation.

### Step 6 — Tokenize where linkage is required

Input: identifiers needed for within-dataset linkage.
Output: deterministic tokens and a protected crosswalk.

Use HMAC-SHA-256 with a secret key, not a plain hash. A plain hash of an MRN is reversible by enumerating
the MRN space. Use a different key per release if cross-release linkage is not intended.

Reason: deterministic tokens preserve joins; a keyed function prevents offline enumeration. The key is
the security boundary, and it must be stored separately from the data.

### Step 7 — Generalize quasi-identifiers

Input: age, ZIP, dates, and other quasi-identifiers.
Output: generalized values with a documented rule.

Age to 5- or 10-year bands, with ages over 89 aggregated to 90+. ZIP to 3 digits, and the 17 restricted
3-digit prefixes set to 000. Dates to month or year if absolute dates are not needed.

Reason: generalization reduces uniqueness while preserving the distribution at the band level. The band
width is a utility decision: 5-year bands preserve more signal than 10-year bands and carry more risk.

### Step 8 — Shift dates when intervals matter

Input: all date and time attributes.
Output: shifted dates with one offset per patient.

Derive the offset from a keyed hash of the patient identifier, in the range 1 to 365 days. Apply it to
every date for that patient, including birth date, and preserve time-of-day where needed. Verify that
intervals are unchanged.

Reason: per-row offsets destroy longitudinal analysis; per-patient offsets preserve it. Birth date must
be shifted with the same offset, or age changes.

### Step 9 — Handle free text

Input: clinical notes, reports, and comments.
Output: de-identified text with a measured residual.

Use a hybrid of rules and named-entity recognition, followed by human review of a sample. Evaluate
precision and recall by identifier type, not as a single number. Replace identifiers with consistent
placeholders so that coreference is preserved.

Reason: free text is where names, dates, and locations hide. A rule-based scrubber catches phone numbers
and dates but misses names; a model catches names but may miss formats it has not seen.

### Step 10 — Handle images and pixels

Input: DICOM or other images.
Output: images with headers and pixels cleaned, plus a pixel audit.

Apply the DICOM confidentiality profile for headers and a pixel-cleaning method for burned-in text. Run
OCR and visual review. Do not assume that a clean header implies a clean image.

Reason: burned-in annotations are a known re-identification vector. Header-only cleaning leaves them
intact.

### Step 11 — Assess small cells and rare values

Input: the candidate release tables.
Output: a small-cell report with counts below the threshold.

Apply a minimum cell size, commonly 11, to counts and cross-tabulations. Apply complementary suppression
so that a suppressed cell cannot be recovered by subtracting from a total. Check rare diagnoses,
rare devices, and unique combinations.

Reason: a table with a cell of 1 discloses that one person has the attribute, even without a name. Small
cells are the most common disclosure in aggregate releases.

### Step 12 — Compute k-anonymity and related measures

Input: the quasi-identifier set.
Output: the distribution of equivalence-class sizes and the number of records below k.

Group by the quasi-identifiers, count, and report the minimum, median, and the number of classes with
fewer than k records. If k is not met, generalize further or suppress the small classes, then recompute.

Reason: k-anonymity gives a concrete, checkable property. It is not sufficient, but it is a useful floor
and a way to compare generalization choices.

### Step 13 — Measure attribute disclosure

Input: the equivalence classes and the sensitive attribute.
Output: l-diversity and, where feasible, t-closeness.

For each class, count distinct sensitive values and the share of the dominant value. A class where all
records share a diagnosis discloses the diagnosis even if k is large.

Reason: k-anonymity alone can leave attribute disclosure. A release that meets k but has homogeneous
classes is not safe for sensitive attributes.

### Step 14 — Model residual risk

Input: the transformed dataset, the recipient, and the auxiliary data.
Output: a risk assessment with scenarios, likelihood, impact, and mitigation.

Scenarios: linkage to a public voter file, linkage to a previous release, recognition by a clinician,
and inference from a rare diagnosis. For each, state the assumed auxiliary data and the estimated
re-identification rate.

Reason: risk is contextual. A numeric k-anonymity value is not a risk assessment; it is one input to
one.

### Step 15 — Document an expert determination

Input: the method and results.
Output: a determination memo.

The memo states the dataset, the quasi-identifiers, the transformation, the risk method, the results, the
assumptions about recipients and auxiliary data, the expert's qualifications, and the conclusion that the
risk is very small. HIPAA does not prescribe a numeric threshold or a credential; the documentation must
support the conclusion.

Reason: an undocumented determination is indistinguishable from an assertion. The memo is the evidence
if the release is challenged.

### Step 16 — Protect the crosswalk

Input: the token-to-identifier mapping.
Output: a separately governed, access-controlled, audited store.

Store the crosswalk in a different system from the release, with its own access policy, encryption keys,
and audit log. Do not include it in backups that are broadly accessible. Define who can use it, for what
purpose, and for how long.

Reason: the crosswalk re-identifies every record. It is the highest-value target in the pipeline and
must be governed as such.

### Step 17 — Validate with a re-identification test

Input: the release and a test auxiliary dataset.
Output: an attempted linkage and a measured match rate.

Attempt linkage on the quasi-identifiers using a public or synthetic auxiliary dataset. Measure how many
records match uniquely. If the match rate is higher than the risk assessment assumed, revise the
transformation.

Reason: a validation that only checks for names does not test the actual threat. The re-identification
test is the empirical check on the risk model.

### Step 18 — Review by privacy, legal, and the IRB

Input: the release package.
Output: approvals with conditions.

The review covers the legal path, the documentation, the residual risk, the DUA, and the recipient's
controls. The engineering team supplies the evidence; it does not make the determination.

Reason: de-identification is a legal and ethical decision. An engineering sign-off without privacy review
leaves the organization exposed.

### Step 19 — Release with contractual controls

Input: the approved package.
Output: a release with a DUA and a data dictionary.

The DUA states the permitted use, the prohibition on re-identification, the security requirements, the
breach notification obligation, and the destruction requirement. The data dictionary states the
transformations and the limitations.

Reason: technical controls reduce risk; contractual controls allocate it. Both are part of the release.

### Step 20 — Monitor and re-assess

Input: the released dataset and its recipients.
Output: a periodic reassessment and an incident process.

Re-assess when the data are refreshed, when a new public dataset enables linkage, when a recipient
requests a change, or when a re-identification technique is published. Record the reassessment date and
the conclusion.

Reason: risk changes without the data changing. A release that was safe in 2020 may be linkable in 2026
because auxiliary data became public.

### Step 21 — Delete and retain per policy

Input: the retention schedule and the DUA.
Output: a documented deletion with evidence.

Delete the identifiable intermediates when they are no longer needed. Keep the de-identified release for
the agreed period. Record what was deleted, when, and by whom.

Reason: retaining identifiable data beyond the purpose increases breach exposure and may violate the DUA
or the minimum-necessary principle.

### Step 22 — Handle identifier collisions and crosswalk linkage

Input: identifiers from multiple sources.
Output: one token per person, with collisions detected before tokenization.

A token is only as stable as the identifier it hashes. If two facilities issue the same MRN, the same
token maps two people; if one person has two MRNs, two tokens map one person. Resolve identity before
tokenization, and record the resolution method in the crosswalk.

| Case | Detection | Rule |
|---|---|---|
| Same MRN, different people | Demographics disagree | Do not merge; assign a source-qualified key |
| Two MRNs, same person | EMPI match | One token; crosswalk stores both sources |
| Missing identifier | Null or placeholder | Quarantine; do not tokenize a placeholder |
| Identifier reused after death | Date overlap | Flag; require manual review |
| Cross-site duplicate | Same demographics and dates | Probabilistic match with review |

Reason: a collision is invisible in the de-identified data and catastrophic in the release. It merges two
patients' diagnoses into one record.

### Step 23 — Handle nulls, sentinels, and date boundaries

Input: the release fields.
Output: a value rule for each edge case.

| Edge case | Value | Rule |
|---|---|---|
| Null age | Missing | Exclude from age bands; report the count |
| Age 90 or older | 92 | Bucket to `90+` |
| Birth date year only | `1950` | Store precision; use 1 July for age |
| Sentinel date | `1900-01-01` | Map to null before shifting |
| Sentinel numeric | `-1`, `9999` | Map to null before aggregation |
| ZIP with leading zero | `02139` | Store as text; never as integer |
| Restricted ZIP3 | `036` | Set to `000` |
| Date at midnight | `2024-01-01T00:00` | Shift the date, preserve time if present |
| Leap day | `2024-02-29` | Date arithmetic; no day-of-year conversion |

Reason: sentinels become real values after aggregation, and leading-zero loss turns a valid ZIP into a
different one. Both are silent.

### Step 24 — Decide the disclosure response for small counts

Input: an aggregate table with small cells.
Output: a suppression rule and a complementary suppression.

| Cell count | Rule | Action |
|---|---|---|
| 0 | Publish | No individual represented |
| 1-10 | Suppress | Primary suppression |
| 11+ | Publish | Above threshold |
| One suppressed per row | Complementary suppression needed | Suppress the next smallest |
| All cells suppressed | Publish the margin only | Redesign the table |

Reason: suppression without complementary suppression can be reversed by subtracting published totals.
The rule must cover the whole table, not one cell.

### Step 25 — Record the acceptance decision

Input: the residual-risk assessment and the release package.
Output: a signed acceptance with conditions and an expiry.

The acceptance names the recipient, the permitted use, the residual risk, the conditions (no
re-identification, no re-sharing, secure storage), and the date on which the assessment must be repeated.
An acceptance without an expiry becomes permanent by default.

Reason: risk changes when auxiliary data change. An acceptance that is never revisited is a claim that
the world has not changed.

### Branch point A — The analysis needs exact dates

Use expert determination or a limited data set rather than safe harbor. Safe harbor removes date elements
other than year, which makes exact-date analyses impossible. Document the additional controls that
compensate.

### Branch point B — The release is public

Public releases need a stronger standard than a controlled research environment: no crosswalk in the same
organization, a re-identification test with public auxiliary data, and a conservative small-cell
threshold. A public release cannot rely on a DUA.

### Branch point C — The data are genomic

Genomic data are inherently identifying and cannot be made anonymous by removing names. Use a controlled
access model, a data use agreement, and expert determination. Aggregate summary statistics may still
permit inference in rare variants.

## 6. Worked example

A research team prepares a release of 25,000 patients with heart failure for an external collaborator. The
release includes demographics, encounters, laboratory values, and 3,412 clinical notes. The team chooses
expert determination because the analysis needs date intervals.

### 6.1 Identifier inventory

| Category | Column or field | Records affected | Classification | Transformation |
|---|---|---:|---|---|
| Name | `patient_name` | 25,000 | Direct | Remove |
| MRN | `mrn` | 25,000 | Direct | Tokenize |
| Phone | `phone` | 18,412 | Direct | Remove |
| Email | `email` | 4,207 | Direct | Remove |
| Full date | `admit_date`, `discharge_date` | 1,200,000 encounters | Quasi | Shift per patient |
| ZIP | `zip5` | 24,988 | Quasi | Truncate to 3, restricted to 000 |
| Age | `age` | 25,000 | Quasi | 5-year bands, 90+ |
| Account number | `account_no` | 22,340 | Direct | Remove |
| Free text | `note_text` | 3,412 notes | Direct and quasi | Hybrid de-identification |
| Facility | `facility_name` | 1,200,000 | Quasi | Generalize to type |

Seven of the 18 safe-harbor categories are present: names, geographic subdivisions, dates, phone numbers,
email addresses, medical record numbers, and account numbers. That count is a coverage check, not a
compliance conclusion; the other 11 categories must still be checked.

### 6.2 Generalization and suppression

ZIP handling:

- 24,988 patients have a five-digit ZIP.
- 17 three-digit prefixes are restricted by HHS: 036, 059, 063, 102, 203, 556, 692, 790, 821, 823, 830,
  831, 878, 879, 884, 890, 893.
- 1,842 patients have a ZIP in a restricted prefix. Their ZIP3 is set to `000`.
- The remaining 23,146 patients keep a three-digit ZIP. A three-digit ZIP is permitted only if the
  geographic unit contains more than 20,000 people; the team verifies this against Census data for each
  prefix.

Small-cell suppression:

A table of 30-day readmission by age band and sex has 12 cells. Two cells have counts below 11: one with
7 and one with 9. Both are suppressed. Because the row and column totals are published, a third cell must
be suppressed to prevent back-calculation. The complementary suppression is chosen as the smallest
remaining cell, which is 14. The released table has 9 of 12 cells visible.

### 6.3 k-anonymity

Quasi-identifiers: 5-year age band, sex, 3-digit ZIP.

| Measure | 5-year bands | 10-year bands |
|---|---:|---:|
| Records | 25,000 | 25,000 |
| Equivalence classes | 1,240 | 612 |
| Classes with k < 5 | 186 | 63 |
| Records in classes with k < 5 | 412 | 141 |
| Minimum class size | 1 | 1 |
| Median class size | 14 | 31 |

At 5-year bands, 412 records (1.65%) are in classes smaller than 5. The team has two options: suppress
those 412 records, or generalize to 10-year bands and suppress 141. Suppressing 412 records removes 1.65%
of the cohort; generalizing loses age precision for every record.

The team chooses 10-year bands and suppression, accepting the loss of age detail for a smaller
suppression. After suppression, the release has 24,859 records, and every remaining class has k ≥ 5.

The arithmetic: 25,000 − 141 = 24,859. The suppressed share is 141 / 25,000 = 0.564%.

### 6.4 Date shifting

Each patient receives an offset in [1, 365] derived from HMAC-SHA-256 of the patient key.

- Patient A: diagnosis 2024-01-10, treatment 2024-05-22. Interval = 133 days.
- Offset for Patient A: 97 days.
- Shifted diagnosis: 2024-04-16. Shifted treatment: 2024-08-27. Interval = 133 days.
- Birth date 1950-03-14 becomes 1950-06-19. Age at diagnosis: 73 years 302 days before and after.

The interval is preserved because the same offset applies to both dates. If the birth date were shifted by
a different offset, the age would change by the difference.

Boundary inference: the dataset covers 2019-01-01 to 2024-12-31. A patient with an event on 2019-01-01
must have an offset near 0; a patient with an event on 2024-12-31 must have an offset near 365. The
estimated offsets narrow the true dates. The team truncates the first and last 30 days of the observation
window, removing 1.8% of events at each edge, to reduce the inference.

### 6.5 Residual risk

| Scenario | Auxiliary data | Estimated match rate | Mitigation |
|---|---|---:|---|
| Voter file linkage | Age, sex, ZIP, name | 0.4% of records | 10-year bands; ZIP3; suppression |
| Previous release linkage | Same cohort, earlier dates | 12% overlap detectable | Consistent tokens across releases if linked; different key if not |
| Clinician recognition | Local knowledge of rare cases | Not quantifiable | Rare-diagnosis review; case suppression |
| Free-text residual | Names left in notes | 1.1 per 100 notes | Hybrid NER; human review of 5% sample |

The expert concludes that the risk is very small for the stated recipient and use, with the conditions
that the DUA prohibits re-identification, the release is not public, and the recipient does not receive
the crosswalk. The determination is documented and reviewed by privacy counsel.

### 6.6 Free-text evaluation

The team evaluates the de-identifier on a sample of 300 notes with manually annotated identifiers.

| Identifier type | True positives | False positives | False negatives | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Names | 412 | 18 | 31 | 95.8% | 93.0% |
| Dates | 1,204 | 42 | 58 | 96.6% | 95.4% |
| Phone numbers | 96 | 3 | 6 | 97.0% | 94.1% |
| Locations | 188 | 27 | 41 | 87.4% | 82.1% |
| MRNs | 74 | 1 | 2 | 98.7% | 97.4% |

Precision for names: 412 / (412 + 18) = 412 / 430 = 95.81%. Recall: 412 / (412 + 31) = 412 / 443 =
93.00%. Locations are the weakest category: 188 / (188 + 27) = 87.44% precision and 188 / (188 + 41) =
82.10% recall.

The residual risk is concentrated in locations and in name variants. The team adds a location gazetteer
and a review of notes containing rare diagnoses, and re-measures.

### 6.7 Transformation ledger

| Field | Before | Operator | After | Records affected |
|---|---|---|---|---|
| `patient_name` | `DOE^JOHN` | Remove | — | 25,000 |
| `mrn` | `MRN12345` | HMAC token | `a3f1...` | 25,000 |
| `phone` | `617-555-0101` | Remove | — | 18,412 |
| `email` | `j@example.org` | Remove | — | 4,207 |
| `admit_date` | `2024-03-14` | Shift +56 | `2024-05-09` | 1,200,000 |
| `birth_date` | `1950-03-14` | Shift +56 | `1950-05-09` | 25,000 |
| `zip5` | `03615` | Truncate, restrict | `000` | 1,842 |
| `zip5` | `02139` | Truncate | `021` | 23,146 |
| `age` | `92` | Generalize | `90+` | 41 |
| `note_text` | free text | Hybrid de-id | placeholders | 3,412 |

The ledger is the operational record. It states what happened to every field, how many records were
affected, and what the output looks like. A report that says only "de-identified" omits all three.

### 6.8 Reconciliation of records through the transformations

| Stage | Records | Calculation |
|---|---:|---|
| Source patients | 25,000 | — |
| After removing direct identifiers | 25,000 | No rows dropped |
| After 10-year generalization | 25,000 | No rows dropped |
| After small-class suppression | 24,859 | −141 |
| Free-text notes processed | 3,412 | Note-level |
| Tokens issued | 25,000 | One per patient |
| Restricted ZIPs set to 000 | 1,842 | Subset of 24,988 |

The suppression removes 141 records, 0.564% of the cohort. The remaining 24,859 all have an equivalence
class of at least 5. The arithmetic reconciles: 25,000 − 141 = 24,859.

### 6.9 Re-identification test

The team links the release to a public voter file on 10-year age band, sex, and 3-digit ZIP.

| Metric | Value | Calculation |
|---|---:|---|
| Release records | 24,859 | — |
| Records with a unique voter-file match | 99 | — |
| Match rate | 0.398% | 99 / 24,859 |
| Records in classes of size 1 | 0 | k ≥ 5 enforced |
| Records with a suppressed class | 141 | 0.564% |

The observed match rate of 0.398% is below the risk assessment's assumed 0.4%, so the model is
conservative. If the rate had been 4%, the release would be revised before distribution.

### 6.10 Diagnostic checks

**Check A: minimum class size.** Expected minimum k = 5. Observed 5. Pass.

**Check B: restricted ZIP handling.** Expected every record with a restricted prefix to have ZIP3 `000`.
Observed 1,842 of 1,842. Pass.

**Check C: free-text location recall.** Expected at least 90%. Observed 82.1% (188 / 229). Fail. The
weakness is concentrated in facility names and street addresses, which the NER model treats as
organizations rather than locations. The remediation is a gazetteer plus a review of notes containing
rare diagnoses.

**Check D: date-shift interval.** Expected the diagnosis-to-treatment interval to remain 133 days.
Observed 133. Pass.

### 6.11 Decision table for a failed de-identification check

| Symptom | First check | Likely cause | Action |
|---|---|---|---|
| k below 5 | Class-size distribution | Insufficient generalization | Generalize or suppress |
| Restricted ZIP present | ZIP3 rule | Rule not applied | Set to 000; re-release |
| Free-text recall low | Confusion matrix by type | Model misses a category | Add rules or gazetteer |
| Interval changed | Date-shift function | Per-row offset | Recompute per patient |
| Crosswalk in release | Object inventory | Packaging error | Remove; re-release |
| Re-id rate above model | Linkage test | Auxiliary data changed | Re-assess; strengthen |

### 6.12 Second run and stability

The transformation is run twice on the same input with the same secret.

| Metric | Run 1 | Run 2 | Expected |
|---|---:|---:|---|
| Records released | 24,859 | 24,859 | Equal |
| Tokens issued | 25,000 | 25,000 | Equal |
| Minimum k | 5 | 5 | Equal |
| Release digest | `c19e...7b02` | `c19e...7b02` | Equal |

A different digest means the token or the date shift is not deterministic. The usual causes are a random
salt generated per run, a shift derived from the current date, or a set iteration order in the suppression
step. All three make the release unreproducible and the re-identification test incomparable across runs.
A stable digest is therefore a privacy control as well as a reproducibility control.

The digest covers the released fields, not the crosswalk. Two releases with the same digest have the same
de-identified content; they may still differ in the crosswalk, which is why the crosswalk has its own
version and access log.

## 7. Code, query, or calculation

### 7.1 Tokenization, date shifting, and k-anonymity (standard library, executable)

The script demonstrates a keyed token, per-patient date shifting with interval verification, and
k-anonymity before and after generalization. It was executed on CPython 3.11.15.

```python
"""De-identification primitives. Standard library only."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
from collections import Counter

SECRET = b"deid-secret-2026"
RESTRICTED_ZIP3 = {
    "036", "059", "063", "102", "203", "556", "692", "790",
    "821", "823", "830", "831", "878", "879", "884", "890", "893",
}


def token(value: str, key: bytes = SECRET) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def shift_days(patient_key: str, max_days: int = 365) -> int:
    digest = hmac.new(SECRET, patient_key.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:4], "big") % max_days + 1


def shift_date(date_text: str, patient_key: str) -> str:
    d = dt.date.fromisoformat(date_text)
    return (d + dt.timedelta(days=shift_days(patient_key))).isoformat()


def zip3(zip5: str) -> str:
    prefix = zip5[:3]
    return "000" if prefix in RESTRICTED_ZIP3 else prefix


def age_band(age: int, width: int) -> str:
    if age >= 90:
        return "90+"
    lower = (age // width) * width
    return f"{lower}-{lower + width - 1}"


# --- tokenization and date shift ---
raw_mrn = "MRN12345"
tok1 = token(raw_mrn)
tok2 = token(raw_mrn)
print("token_length", len(tok1))
print("token_deterministic", tok1 == tok2)
print("token_differs_from_raw", tok1 != raw_mrn)
print("restricted_zip", zip3("03615"))
print("allowed_zip", zip3("02139"))

patient = "MRN12345"
before_a, before_b = "2024-01-10", "2024-05-22"
after_a, after_b = shift_date(before_a, patient), shift_date(before_b, patient)
interval_before = (dt.date.fromisoformat(before_b) - dt.date.fromisoformat(before_a)).days
interval_after = (dt.date.fromisoformat(after_b) - dt.date.fromisoformat(after_a)).days
print("interval_before", interval_before)
print("interval_after", interval_after)
print("interval_preserved", interval_before == interval_after)
print("offset_in_range", 1 <= shift_days(patient) <= 365)

# --- k-anonymity ---
# 20 synthetic patients: (id, age, sex, zip5)
RECORDS = [
    (1, 31, "F", "02139"), (2, 32, "F", "02139"), (3, 33, "F", "02139"),
    (4, 34, "F", "02139"), (5, 30, "F", "02139"), (6, 30, "F", "02139"),
    (7, 36, "F", "02139"), (8, 37, "F", "02139"), (9, 38, "F", "02139"),
    (10, 39, "F", "02139"),
    (11, 41, "M", "02139"), (12, 42, "M", "02139"), (13, 43, "M", "02139"),
    (14, 44, "M", "02139"), (15, 40, "M", "02139"),
    (16, 45, "M", "10001"), (17, 46, "M", "10001"), (18, 47, "M", "10001"),
    (19, 51, "F", "10001"), (20, 52, "F", "10001"),
]


def kanon(records, width: int, k: int = 5) -> dict:
    keys = [
        (age_band(age, width), sex, zip3(zip5))
        for _pid, age, sex, zip5 in records
    ]
    counts = Counter(keys)
    small = {key: n for key, n in counts.items() if n < k}
    return {
        "classes": len(counts),
        "small_classes": len(small),
        "records_in_small": sum(small.values()),
        "min_class": min(counts.values()),
        "median_class": sorted(counts.values())[len(counts) // 2],
    }


five = kanon(RECORDS, 5)
ten = kanon(RECORDS, 10)
print("kanon_5y", five)
print("kanon_10y", ten)
print("suppressed_share_10y", round(ten["records_in_small"] / len(RECORDS), 3))
```

Expected output:

```text
token_length 16
token_deterministic True
token_differs_from_raw True
restricted_zip 000
allowed_zip 021
interval_before 133
interval_after 133
interval_preserved True
offset_in_range True
kanon_5y {'classes': 5, 'small_classes': 3, 'records_in_small': 9, 'min_class': 2, 'median_class': 4}
kanon_10y {'classes': 4, 'small_classes': 2, 'records_in_small': 5, 'min_class': 2, 'median_class': 5}
suppressed_share_10y 0.25
```

The 5-year bands produce five classes, three of which have fewer than five records, covering nine
patients. The 10-year bands merge the two smallest age groups and reduce the small-class burden to five
patients, or 25% of the sample. The median class size is five in both cases.

### 7.2 PostgreSQL 18 k-anonymity query

The following targets PostgreSQL 18 and computes equivalence classes directly in SQL.

```sql
-- PostgreSQL 18
WITH quasi AS (
    SELECT
        person_id,
        CASE WHEN age >= 90 THEN '90+'
             ELSE ((age / 10) * 10)::text || '-' ||
                  (((age / 10) * 10) + 9)::text
        END AS age_band,
        sex,
        CASE WHEN left(zip5, 3) IN (
                 '036','059','063','102','203','556','692','790',
                 '821','823','830','831','878','879','884','890','893')
             THEN '000' ELSE left(zip5, 3)
        END AS zip3
    FROM release.patient
),
classes AS (
    SELECT age_band, sex, zip3, COUNT(*) AS k
    FROM quasi
    GROUP BY age_band, sex, zip3
)
SELECT
    COUNT(*)                                   AS classes,
    MIN(k)                                     AS min_k,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY k) AS median_k,
    COUNT(*) FILTER (WHERE k < 5)              AS classes_below_5,
    SUM(k) FILTER (WHERE k < 5)                AS records_below_5
FROM classes;
```

PostgreSQL 18's `percentile_cont` is an ordered-set aggregate; the `WITHIN GROUP` clause is required. The
age expression uses integer division, which truncates toward zero for positive ages.

## 8. Data, standards, and regulatory context

### 8.1 HIPAA de-identification

The standard is at 45 CFR 164.514(a)-(c). Two methods satisfy it:

**Safe harbor, §164.514(b)(2).** Remove the 18 categories of identifiers and have no actual knowledge that
the remaining information could identify the individual. The 18 categories:

1. Names.
2. Geographic subdivisions smaller than a state, including street address, city, county, precinct, ZIP,
   and equivalent geocodes, except the initial three digits of a ZIP if the unit has more than 20,000
   people.
3. All elements of dates (except year) related to an individual, including birth date, admission,
   discharge, and death, and all ages over 89 (which must be aggregated to 90+).
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
14. Web URLs.
15. IP addresses.
16. Biometric identifiers, including fingerprints and voiceprints.
17. Full-face photographs and comparable images.
18. Any other unique identifying number, characteristic, or code.

HHS guidance (2012, page last reviewed February 2025) adds that free-text fields must be scrubbed,
partial identifiers such as initials or the last four digits of an SSN are not permitted, and the 17
restricted three-digit ZIP prefixes must be set to 000.

**Expert determination, §164.514(b)(1).** A person with appropriate statistical or scientific knowledge
and experience determines that the risk is very small that the information could be used to identify an
individual, and documents the methods and results. HHS does not prescribe a credential, a numeric risk
threshold, or a specific technique; k-anonymity is one model among several. The determination may be
time-limited.

### 8.2 What the standard does not require

- A specific software tool.
- A specific k.
- A specific date-shift range.
- A specific suppression threshold.
- A DUA for de-identified data (though one is permitted).
- The same method for every release.

Those are choices that must be documented and justified.

### 8.3 GDPR and other regimes

GDPR distinguishes pseudonymization (Art. 4(5)) from anonymization. Pseudonymized data remain personal
data and are subject to the regulation; anonymized data are outside its scope, but the standard for
anonymization is high and Recital 26 requires that re-identification be impossible by reasonable means,
considering all relevant factors. Removing names is not anonymization.

The EU Network and Information Security Directive, national health data laws, and institutional policies
may add requirements. A release that crosses jurisdictions needs a review of each.

### 8.4 Related standards

- **NIST SP 800-188,** "De-Identifying Government Datasets," provides a process and a vocabulary for
  de-identification, including the distinction between de-identification and anonymization.
- **ISO/IEC 20889** provides privacy-enhancing data de-identification terminology and techniques.
- **HIPAA Privacy Rule, §164.514(e),** defines the limited data set, which is not de-identified.
- **21 CFR Part 11** applies when de-identified data support a regulated submission; the audit trail
  requirement is about the records, not the de-identification method.

## 9. Industry practice and case evidence

### 9.1 Date shifting in practice

A 2023 audit of 71 U.S. health systems (Evans, London, and Palchuk, Applied Clinical Informatics) reported
that 55% did not shift dates, and that among those that did, about 25% used offsets of ±7 days and about
20% used ±365 days. The study also described detection methods: weekday-only patterns of routine visits
expose shifts of seven days or more, and seasonal diagnoses estimate the magnitude. The finding is
evidence that date-shift practice is heterogeneous and that small shifts are detectable.

Hripcsak et al. (JAMIA, 2016) documented the "shift and truncate" approach and its limits: boundary
inference from fixed dataset start and end dates, inference across successive releases, and loss of
seasonality when dates are truncated. Their recommended mitigation is truncation of the dataset edges,
at the cost of losing events.

### 9.2 Free-text failures

Alexander and Beatty (JAMIA, 2022) described a case in which nonspecific shifting of date-like text in
de-identified notes allowed recovery of dates, including birthdates, for some patients. The lesson is that
over-redaction is not automatically safe and that date-like text requires a method that preserves
consistency without preserving the original value.

### 9.3 Trusted research environments

OpenSAFELY and the All of Us Research Program use controlled environments and tiered access rather than
public release. The control model shifts risk from the dataset to the environment: the data may retain
more detail because the recipient cannot export records. This is a legitimate alternative to aggressive
de-identification, and it depends on audit, code review, and export controls.

### 9.4 What is inference

- That a k of 5 is sufficient. It is a common floor, not a legal standard; sufficiency depends on the
  attributes and the auxiliary data.
- That a 365-day shift prevents inference. Boundary and update inference can narrow the offset; the shift
  reduces but does not eliminate the risk.
- That an expert determination is permanent. HHS permits time-limited determinations, and new auxiliary
  data can change the risk.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Names only | Release has dates and ZIP | Direct identifiers removed, quasi retained | Linkage re-identification | Quasi-identifier assessment |
| Per-row date shift | Intervals change | Offset recomputed per row | Invalid longitudinal analysis | One offset per patient |
| Birth date unshifted | Age differs after shift | Date tag missed | Age disclosure or error | Shift all dates |
| Free-text residue | Names in notes | Rules miss variants | Identifier leak | Hybrid NER; review |
| Crosswalk co-located | Token map in the release bucket | Convenience | Full re-identification | Separate governed store |
| No small-cell rule | Cell of 1 in a table | Aggregate treated as safe | Attribute disclosure | Minimum cell size; complementary suppression |
| k-anonymity only | Homogeneous classes | Attribute disclosure ignored | Diagnosis inference | l-diversity; review |
| Undocumented expert determination | Memo absent | Formality | Unsupported claim | Document method and results |
| Restricted ZIP retained | ZIP3 in the 17 prefixes | Rule missed | Geographic disclosure | Set to 000 |
| Age over 89 in years | 92 in the release | Aggregation rule missed | Age disclosure | 90+ bucket |
| Plain hash token | MRN space enumerable | No secret key | Reversal | HMAC with a key |
| No re-identification test | Risk asserted | No empirical check | Unknown residual risk | Linkage test |

Two red flags deserve emphasis.

**A release with no identifier inventory.** If the team cannot list every field and its classification,
the transformations were chosen by intuition. The inventory is the evidence that the 18 categories were
checked.

**A risk assessment with one number.** "k = 5, therefore safe" is not an assessment. The assessment names
the auxiliary data, the linkage scenario, the estimated match rate, and the residual risk.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The identifier inventory is complete | Coverage of the standard | A forgotten free-text field | Schema and text review |
| The secret key is protected | Token security | Key stored with data | Key management audit |
| The population is large enough for k | Generalization works | Small cohort or rare disease | Class-size distribution |
| Auxiliary data are bounded | Risk model | A new public dataset | Re-assessment trigger |
| Date shift preserves intervals | Longitudinal utility | Per-row offset | Interval test |
| OCR detects most burned-in text | Pixel safety | Stylized text | Visual review |
| The recipient honors the DUA | Contractual control | Re-sharing | Audit and contract |
| The expert is qualified | Legal validity | Self-designation | Credentials and review |
| Suppression does not bias the analysis | Utility | Non-random suppression | Compare suppressed and retained |

The largest limit is that de-identification is not a property of a file. It is a property of a release,
a recipient, and a use. The same dataset can be de-identified for one recipient and not for another. The
documentation must state the context, or the claim is meaningless.

A second limit is that utility and privacy trade off in ways that are not linear. Generalizing age from
5-year to 10-year bands may reduce re-identification risk slightly and destroy an age-specific analysis.
The choice should be made with the analyst, not only by the privacy team.

## 12. Exercises

### Exercise 1 — Identifier inventory (application, 40 minutes)

Given a schema with 40 columns, deliverable: an inventory classifying each as direct, quasi, sensitive, or
non-identifying, mapped to the 18 categories where applicable.

Acceptance criteria: all 18 categories addressed; free-text and date fields included; at least three
quasi-identifiers identified that are not in the 18 categories.

### Exercise 2 — Choose the path (design, 35 minutes)

Given two releases (a public dataset for benchmarking and a controlled collaborator analysis with
survival endpoints), deliverable: a path choice, a justification, and the utility consequences.

Acceptance criteria: public release uses a stronger standard; survival analysis addresses date handling;
the documentation requirements are stated.

### Exercise 3 — Transformation plan (application, 45 minutes)

Deliverable: a table of fields with operator, parameter, utility cost, and privacy effect.

Acceptance criteria: every direct identifier has an operator; every quasi-identifier has a
generalization or retention rationale; the crosswalk is addressed.

### Exercise 4 — Date shift (application, 50 minutes)

Deliverable: a per-patient shift function and a test that verifies three intervals and one age are
preserved across a leap year.

Acceptance criteria: one offset per patient; birth date shifted; boundary inference discussed; the test
fails if the offset is per-row.

### Exercise 5 — k-anonymity (calculation, 45 minutes)

Given 20 records, deliverable: the equivalence classes at 5-year and 10-year bands, the small classes, and
the suppression share.

Acceptance criteria: counts reconcile; the effect of generalization is quantified; the trade-off is
stated.

### Exercise 6 — Free-text evaluation (application, 60 minutes)

Deliverable: a confusion matrix by identifier type for a sample of 50 notes, with precision, recall, and
a residual-risk statement.

Acceptance criteria: at least four identifier types; precision and recall computed correctly; the weakest
category is identified with a remediation.

### Exercise 7 — Crosswalk protection (design, 40 minutes)

Deliverable: an architecture for the crosswalk with access control, encryption, audit, and retention.

Acceptance criteria: separate system; separate keys; named roles; audit retention; deletion procedure.

### Exercise 8 — De-identification report (design, 60 minutes)

Deliverable: a report for a release, including method, transformations, validation, residual risk, and
acceptance.

Acceptance criteria: states the legal path; documents the parameters; includes a re-identification test;
has a reviewer sign-off line.

## 13. Self-check answers

**1. What are the two HIPAA de-identification methods?**

Safe harbor (remove 18 categories and have no actual knowledge) and expert determination (a qualified
expert determines the risk is very small and documents the methods and results). Either satisfies the
standard; neither is a specific software configuration.

**2. When would you choose expert determination?**

When the analysis needs dates, geography, or linkage that safe harbor would remove, and the organization
can support a qualified expert and documentation. Safe harbor is simpler but less flexible.

**3. Why is a plain hash of an MRN not safe?**

Because the MRN space is small and enumerable. An attacker can hash every possible MRN and match the
tokens. HMAC with a secret key, stored separately, prevents offline enumeration.

**4. Compute the interval preservation.**

Diagnosis 2024-01-10, treatment 2024-05-22: the interval is 133 days (January 21 remaining days + February
29 in a leap year + March 31 + April 30 + May 22 = 133). With a 97-day offset, the dates become 2024-04-16
and 2024-08-27, and the interval is still 133 days.

**5. What does k-anonymity not protect against?**

Attribute disclosure. If every record in an equivalence class has the same sensitive value, learning the
class reveals the value. l-diversity and t-closeness address this.

**6. Why is a limited data set not de-identified?**

Because it may include dates and city, state, and ZIP, which are quasi-identifiers, and it remains PHI
subject to the Privacy Rule. It is permitted for certain purposes with a DUA, but it is a separate
category.

**7. What is the small-cell rule for?**

To prevent disclosure of a small number of individuals in an aggregate table. A cell of 1 reveals that one
person has the attribute. Complementary suppression prevents recovery by subtraction.

**8. Compute the suppression share.**

141 records in small classes of 25,000 = 0.00564, or 0.564%. After suppression, 24,859 records remain.

**9. Why shift the birth date with the same offset?**

Because age is derived from birth date and event date. If only the event date moves, age changes by the
offset, which can move a patient across an eligibility threshold or disclose the offset.

**10. What belongs in a de-identification report?**

The legal path, the identifier inventory, the transformations and parameters, the validation results, the
re-identification test, the residual-risk assessment, the recipient and use, the retention and deletion
terms, and the reviewer approvals.

## 14. Interview questions

**1. "How do you de-identify a clinical dataset?"**

Strong: inventory identifiers, choose safe harbor or expert determination, apply remove/tokenize/
generalize/shift/suppress, validate with a linkage test, document residual risk, and obtain review. Weak:
"remove names and MRNs."

**2. "What is the difference between safe harbor and expert determination?"**

Strong: names the 18 categories and the no-actual-knowledge condition for safe harbor; describes the
expert, the very-small-risk determination, and documentation for expert determination. Weak: "safe harbor
is the checklist."

**3. "How do you handle dates?"**

Strong: one offset per patient, all date attributes including birth date, interval verification, boundary
inference, and a documented option. Weak: "randomize each date."

**4. "What is k-anonymity, and what does it miss?"**

Strong: equivalence classes of size at least k; misses attribute disclosure, which l-diversity addresses.
Weak: "it means k people look the same."

**5. "How do you de-identify free text?"**

Strong: hybrid rules and NER, evaluation by identifier type with precision and recall, human review of a
sample, residual-risk statement. Weak: "regex for names."

**6. "Where do you store the crosswalk?"**

Strong: a separate governed system with separate keys, access control, audit, and retention. Weak: "in
the same database with a restricted view."

**7. "How do you test whether a release is re-identifiable?"**

Strong: attempt linkage with public or synthetic auxiliary data, measure unique matches, compare to the
risk model. Weak: "search for names."

**8. "What is the risk of a public release versus a controlled environment?"**

Strong: public release removes the contractual and environmental controls, so the technical standard must
be stronger; a TRE can retain more detail because export is controlled. Weak: "they are the same if the
data are de-identified."

**9. "Can genomic data be de-identified?"**

Strong: not by removing names; use controlled access, expert determination, and aggregate release with
inference risk assessment. Weak: "yes, remove the identifiers."

**10. "How do you know the expert determination is valid?"**

Strong: qualified expert, documented method and results, assumptions about auxiliary data, time limit,
and review. Weak: "a consultant signed it."

## 15. Key terms

- **Actual knowledge** — clear and direct knowledge of identifiability.
- **Anonymization** — irreversible removal of identifiability.
- **Attribute disclosure** — revealing a sensitive value without naming a person.
- **Auxiliary data** — external data used for linkage.
- **Complementary suppression** — suppressing extra cells to prevent back-calculation.
- **Crosswalk** — token-to-identifier mapping.
- **Date shifting** — per-patient date offset.
- **De-identification** — meeting the legal standard for removing identifiability.
- **Differential privacy** — formal guarantee parameterized by epsilon.
- **Direct identifier** — identifies on its own.
- **DUA** — data use agreement.
- **Equivalence class** — records sharing quasi-identifier values.
- **Expert determination** — qualified expert's very-small-risk determination.
- **Generalization** — reducing precision.
- **HMAC** — keyed hash.
- **k-anonymity** — at least k indistinguishable records per class.
- **l-diversity** — at least l distinct sensitive values per class.
- **Limited data set** — HIPAA category with dates and geography but no direct identifiers.
- **Linkage attack** — joining with auxiliary data.
- **Minimum necessary** — limit use to what is needed.
- **PHI** — protected health information.
- **Pseudonymization** — reversible replacement of identifiers.
- **Quasi-identifier** — identifies in combination.
- **Re-identification** — linking a record to a person.
- **Residual risk** — risk remaining after transformation.
- **Safe harbor** — removal of 18 identifier categories.
- **Sensitive attribute** — value whose disclosure matters.
- **Small cell** — count below the disclosure threshold.
- **Suppression** — removing values or cells.
- **t-closeness** — distribution closeness to the overall data.
- **Tokenization** — replacing a value with a keyed token.

## 16. Sources

All URLs accessed 25 October 2026.

1. eCFR. "45 CFR 164.514." https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
2. HHS. "Guidance Regarding Methods for De-identification of Protected Health Information."
   https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
3. HHS. "Workshop on the HIPAA Privacy Rule's De-Identification Standard."
   https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/2010-de-identification-workshop/index.html
4. NCVHS. "Recommendations on De-identification of Protected Health Information under HIPAA," 2017.
   https://ncvhs.hhs.gov/wp-content/uploads/2018/03/2017-Ltr-Privacy-DeIdentification-Feb-23-Final-w-sig.pdf
5. Hripcsak, G. et al. "Preserving temporal relations in clinical data while maintaining privacy."
   JAMIA, 2016. https://doi.org/10.1093/jamia/ocw001
6. Alexander, J. and Beatty, J. "Nonspecific deidentification of date-like text in deidentified clinical
   notes enables reidentification of dates." JAMIA, 2022. https://doi.org/10.1093/jamia/ocac147
7. Evans, R., London, J., and Palchuk, M. "The Detection of Date Shifting in Real-World Data."
   Applied Clinical Informatics, 2023. https://doi.org/10.1055/a-2130-2197
8. NIST. "SP 800-188, De-Identifying Government Datasets." https://csrc.nist.gov/pubs/sp/800/188/final
9. ISO/IEC. "ISO/IEC 20889:2018 Privacy enhancing data de-identification terminology and classification of
   techniques." https://www.iso.org/standard/69373.html
10. GDPR. Regulation (EU) 2016/679. https://eur-lex.europa.eu/eli/reg/2016/679/oj
11. OpenSAFELY. https://www.opensafely.org/
12. All of Us Research Program. https://allofus.nih.gov/
13. eCFR. "45 CFR 164.502." https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.502
14. eCFR. "21 CFR Part 11." https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_data-versioning-and-lineage.md`](07_data-versioning-and-lineage.md)
- Next topic: [`09_security-engineering.md`](09_security-engineering.md)
- Related sibling: [`06_dicom-pipelines.md`](06_dicom-pipelines.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../course/03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — Lab 3.5 audits a synthetic dataset for identifiers.
