# Standards and Terminologies

> Module 02 — The Biomedical Data Landscape
> Topic 08 of 10
> Estimated study time: 5 hours
> Prerequisites: Topic 01 is strongly recommended
> Version 2026.3

---

## 1. Why this topic exists

Every other topic in this module runs into the same wall: the data are recorded in a local code, and the question requires a standard concept. A lab result is stored as `LAB_GLU_F` at one hospital and `GLUCOSE-FAST` at another. A diagnosis is `250.00` in ICD-9 and `E11.9` in ICD-10-CM, and both are billing constructs rather than clinical entities. A medication order is a free-text string that maps to three different RxNorm concepts depending on whether you want the ingredient, the clinical drug, or the branded product.

Standards exist to solve this, and they solve it partially. The gap between "partially" and "fully" is where most multi-site analytics fails, and it is not a gap that engineering closes. It is a semantic gap, and the only way to manage it is to measure it.

Three distinctions organize the whole topic:

1. **Structure versus terminology.** FHIR and OMOP are structures: they define where data sit. LOINC, SNOMED CT, ICD, and RxNorm are terminologies: they define what the data mean. A perfect structure with a bad terminology is still unusable, and a good terminology without a structure cannot be exchanged.
2. **Interface versus archive.** HL7 v2 and FHIR are exchange standards. OMOP is an analytic model. An interface standard is optimized for a transaction; an analytic model is optimized for a query across a population.
3. **Mapping versus equivalence.** A map says "this local code is represented by that standard concept." It does not say the two are the same thing. Every map is an assertion with a confidence, and the confidence is usually unrecorded.

The commercial stakes:

- A multi-site study that maps source codes to standard concepts without measuring the unmapped fraction will report a cohort that is systematically smaller at some sites, and will interpret the difference as epidemiology.
- A product that depends on a specific terminology without a license will be blocked at launch.
- A pipeline that maps at the ingredient level when the question requires the clinical drug level will produce exposure misclassification that no statistical method fixes.
- An organization that treats OMOP as a data warehouse rather than an analytic model will spend years on an ETL that answers one question.

The decision this topic supports: which standard applies to a given data element, what the mapping loses, and whether the loss is small enough for the intended use.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | Failure without this topic |
|---|---|---|
| Multi-site study lead | Whether sites are comparable | Unmeasured mapping loss read as population difference |
| Interoperability engineer | Which standard to implement | Implements a structure without a terminology strategy |
| Product manager | Whether the product can be licensed | Terminology license discovered at launch |
| Data engineer | How to build the ETL | Maps at the wrong granularity |
| Regulatory lead | Whether the data meet a standard | Cites FHIR without citing the profile or version |
| Clinical informaticist | Whether a phenotype is portable | Local codes embedded in the definition |
| Procurement lead | What the data contract covers | Assumes terminologies are free |

### 1.2 The four failure modes of a mapping

| Failure | Example | Consequence |
|---|---|---|
| One-to-many | One local code maps to two standard concepts | Double counting or arbitrary choice |
| Many-to-one | Several local codes map to one standard concept | Loss of specificity; a subgroup disappears |
| Unspecific | Local code maps to a broad parent concept | A precise cohort becomes a broad one |
| No map | Local code has no standard representation | Silent exclusion from the cohort |

Each failure has a measurable rate, and the rates should be reported by site and by domain.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** structural standards from terminological standards and place HL7 v2, FHIR, OMOP, LOINC, SNOMED CT, ICD, RxNorm, and UCUM in the correct category.
2. **Explain** the LOINC six-axis model and use it to decide whether two lab observations are the same measurement.
3. **Map** ICD-10-CM codes to SNOMED CT concepts through OMOP and quantify the loss in a cohort.
4. **Compute** mapping rates by code count and by volume, and explain why they differ.
5. **Choose** the correct RxNorm level for a medication question and state the consequence of the wrong choice.
6. **Critique** an interoperability claim for missing profile, version, and terminology license.
7. **Design** a terminology provenance record that names every code system version and mapping table.
8. **Decide** whether a mapping loss is acceptable for a stated use, with an explicit threshold.

Mapping to assessment: outcomes 3, 4, and 5 are assessed by exercises 3, 4, and 5. Outcomes 1, 2, and 6 are assessed by the self-check items. Outcomes 7 and 8 are assessed by the design exercises and the module memo.

---

## 3. The core mental model

Standards answer two different questions, and conflating them is the root of most interoperability confusion.

```text
   QUESTION 1: WHERE DOES THE DATA SIT?
   (structure / exchange)

   HL7 v2  -----> messages between systems (ADT, ORM, ORU)
   CDA     -----> documents (summaries, reports)
   FHIR    -----> resources over a REST API (R4, R5, R6)
   OMOP    -----> relational tables for analysis (CDM 5.x)
   CDISC   -----> submission datasets (SDTM, ADaM)
   DICOM   -----> imaging objects

   QUESTION 2: WHAT DOES THE DATA MEAN?
   (terminology / semantics)

   SNOMED CT ---> clinical concepts (findings, disorders, procedures)
   LOINC     ---> observations and measurements
   ICD-10-CM ---> diagnoses for billing and statistics
   ICD-10-PCS --> inpatient procedures
   CPT/HCPCS ---> outpatient procedures and services
   RxNorm    ---> medications (US)
   NDC       ---> drug products and packages
   UCUM      ---> units of measure
   MedDRA    ---> adverse events
   HPO       ---> phenotypes

   THE JOIN:
   structure + terminology + a mapping with a recorded confidence
```

Two rules follow from the diagram:

1. **A structure without a terminology is a container.** A FHIR `Observation` with a local code is exchangeable and not comparable. A FHIR `Observation` with a LOINC code is both.
2. **A terminology without a structure cannot be queried.** A SNOMED concept is a meaning; it needs a table and a column to be used.

### 3.1 The three levels of semantic agreement

Multi-site analytics requires agreement at three levels, and each is harder than the last:

| Level | Agreement required | Typical mechanism | Failure mode |
|---|---|---|---|
| Syntactic | Same fields, same formats | FHIR profiles, OMOP schema | Parse errors |
| Semantic | Same concept for the same thing | LOINC, SNOMED, RxNorm | Concept drift |
| Pragmatic | Same interpretation in context | Local validation, chart review | Contextual mismatch |

Most interoperability projects achieve syntactic agreement and claim semantic agreement. The gap is where a cohort definition works at one site and not another.

### 3.2 Why mapping is lossy

A map between terminologies is lossy because the terminologies were built for different purposes:

- **ICD-10-CM** is a billing classification. It is designed for reimbursement categories, not for clinical precision. It has codes for "heart failure, unspecified" because a coder needs a code when the documentation is vague.
- **SNOMED CT** is a clinical terminology with a concept model and a description logic. It can express "heart failure with preserved ejection fraction" precisely, and it has relationships that support subsumption queries.
- **LOINC** describes an observation along six axes, so a glucose measurement is not one concept but a family of concepts distinguished by specimen, timing, and method.
- **RxNorm** normalizes drug names at several levels, from ingredient to branded package.

A map from ICD to SNOMED necessarily loses the precision that ICD never had, and a map from SNOMED to ICD necessarily loses the precision that ICD cannot represent.

### 3.3 The mapping confidence scale

Every mapping should carry a confidence. A four-level scale is enough for most work:

| Level | Meaning | Use |
|---|---|---|
| Exact | The concepts are equivalent for the intended use | Can be used without qualification |
| Narrower | The standard concept is more specific than the source | Safe for sensitivity, may miss cases |
| Broader | The standard concept is less specific than the source | Safe for specificity, may over-capture |
| Unmapped | No standard concept represents the source | Must be reported, never silently dropped |

A mapping table without a confidence column is a list of guesses.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Standard | A specification for structure or meaning | Used for both without distinction | None |
| Structure standard | Defines where data sit | Confused with terminology | FHIR, OMOP, HL7 v2 |
| Terminology | A system of concepts and relationships | Confused with a classification | SNOMED CT, LOINC |
| Classification | A system for grouping for statistics or payment | Confused with a clinical terminology | ICD-10-CM |
| HL7 v2 | Message standard for healthcare exchange | Treated as a data model | v2.5.1 and later |
| CDA | Clinical Document Architecture | Confused with FHIR | Release 2 |
| FHIR | Fast Healthcare Interoperability Resources | Treated as an archive format | R4, R5, R6 |
| US Core | US realm FHIR profiles | Confused with FHIR itself | Implementation guide |
| USCDI | US Core Data for Interoperability | Confused with a standard | Data element list |
| OMOP CDM | Common Data Model for observational analytics | Treated as a warehouse schema | 5.4, 5.5 |
| CDM | Common data model | Used generically | Version-specific |
| CONCEPT | The OMOP vocabulary table of concepts | Confused with a source code | Table |
| SOURCE_TO_CONCEPT_MAP | The OMOP table of non-standard to standard maps | Ignored in ETL review | Table |
| Concept ID | The integer identifying an OMOP concept | Confused with a source code | Integer |
| Standard concept | A concept flagged as standard in OMOP | Assumed to be the only valid one | Flag |
| Concept ancestor | The OMOP table of hierarchical relationships | Ignored, causing under-capture | Table |
| Vocabulary | A code system loaded into OMOP | Confused with a concept | Vocabulary ID |
| LOINC | Logical Observation Identifiers Names and Codes | Treated as a lab test catalogue | Six axes |
| Six axes | Component, property, time, system, scale, method | Ignored when matching labs | LOINC model |
| SNOMED CT | Systematized Nomenclature of Medicine Clinical Terms | Treated as a billing code set | About 360,000 active concepts |
| Concept model | The description-logic model behind SNOMED CT | Ignored in queries | SNOMED CT |
| RF2 | Release Format 2 for SNOMED CT | Confused with a database schema | Release format |
| Subsumption | Whether one concept is a descendant of another | Confused with string matching | Query operation |
| ICD-10-CM | US diagnosis classification | Treated as a clinical terminology | FY version |
| ICD-10-PCS | US inpatient procedure classification | Confused with CPT | FY version |
| ICD-11 | The eleventh ICD revision | Assumed adopted in the US | WHO classification |
| CPT | Current Procedural Terminology | Assumed free | AMA copyright |
| HCPCS | Healthcare Common Procedure Coding System | Confused with CPT | CMS |
| RxNorm | Normalized US medication nomenclature | Treated as a single concept level | Ingredient to branded pack |
| Ingredient | The active moiety level in RxNorm | Confused with clinical drug | RxNorm level |
| Clinical drug | Strength and dose form level | Confused with branded drug | RxNorm level |
| Branded drug | A specific marketed product | Confused with ingredient | RxNorm level |
| NDC | National Drug Code | Confused with RxNorm | 11 digits |
| UCUM | Unified Code for Units of Measure | Treated as optional | Code system |
| MedDRA | Medical Dictionary for Regulatory Activities | Confused with SNOMED | Subscription |
| HPO | Human Phenotype Ontology | Confused with SNOMED findings | Ontology |
| MeSH | Medical Subject Headings | Confused with a clinical terminology | NLM |
| Mapping | An assertion that one code represents another | Treated as equivalence | Table |
| Crosswalk | A table of mappings | Treated as validated | Table |
| Concept drift | A concept's meaning or membership changing over versions | Ignored in longitudinal work | Version change |
| Terminology server | A service that resolves codes and maps | Confused with a database | FHIR terminology service |
| Value set | A defined set of codes for a purpose | Confused with a code system | FHIR resource |

### 4.1 Terms that are routinely conflated

- **ICD versus SNOMED CT.** ICD is a classification for statistics and payment; SNOMED CT is a clinical terminology. Mapping ICD to SNOMED is a lossy upward translation, and mapping SNOMED to ICD is a lossy downward grouping.
- **LOINC code versus lab test.** A lab test is a local workflow; a LOINC code is an observation definition along six axes. One test can require several LOINC codes depending on specimen and timing.
- **RxNorm ingredient versus clinical drug.** "Metformin" and "metformin 500 mg oral tablet" are different concepts at different levels. A question about exposure usually needs the clinical drug; a question about class usually needs the ingredient.
- **FHIR version versus profile version.** FHIR R4 is the base standard; US Core 9.0.0 is a profile on R4. Citing "FHIR" without either is not a specification.

---

## 5. The mechanics, step by step

This section is the working procedure for building and validating a terminology mapping for an analytic dataset. It has 30 steps in five phases.

### Phase A — Inventory the source (steps 1-6)

**Step 1. List every coded field in the source.**
Input: the data dictionary. Output: an inventory with field, code system, and cardinality.
Reason: mapping effort is proportional to the number of distinct codes, not the number of rows. A field with 40 codes is cheap; one with 12,000 is a project.
Rule: record the distinct code count per field, not only the field name.

**Step 2. Classify each field as coded, free text, or mixed.**
Input: the inventory. Output: a classification.
Reason: free text requires extraction before mapping, and mixed fields hide free-text values inside coded columns.
Rule: sample 100 values per field and count non-conforming entries.

**Step 3. Identify the source code system and its version.**
Input: the source documentation. Output: a code system identifier and version.
Reason: a local code set has no external meaning, and an ICD-10-CM code means something different in FY2025 than FY2026.
Rule: where the version is unknown, treat the field as local and map from scratch.

**Step 4. Determine the required granularity for the question.**
Input: the analytic question. Output: a target granularity (ingredient, clinical drug, parent concept, specific concept).
Reason: the correct mapping depends on the question. A class-level question needs a parent concept; a product-level question needs the specific concept.
Rule: write the granularity requirement before looking at the mapping table.

**Step 5. Identify the target terminology and its version.**
Input: the analytics platform. Output: a target code system and version.
Reason: OMOP standard concepts are versioned by vocabulary release, and a mapping built on one release may not exist in the next.
Rule: record the vocabulary release date.

**Step 6. Identify the license terms for each terminology.**
Input: the license documentation. Output: a license record.
Reason: SNOMED CT, CPT, and MedDRA are licensed; LOINC and ICD-10-CM are free with conditions. A product that embeds a licensed terminology needs a license.
Rule: record the license holder and the permitted uses.

### Phase B — Build the mapping (steps 7-12)

**Step 7. Attempt exact matches first.**
Input: the source codes. Output: a set of candidate exact matches.
Reason: exact matches are cheap and high confidence, and they establish the ceiling of what mapping can achieve.
Rule: use the terminology's own map tables where they exist, and record the map source.

**Step 8. Attempt synonym and lexical matches second.**
Input: the remaining codes. Output: candidate matches with a similarity score.
Reason: lexical matching finds many true matches and many false ones. It requires review.
Rule: set a similarity threshold and review everything below it.

**Step 9. Review candidates against the concept definition, not the label.**
Input: the candidates. Output: accepted, rejected, or escalated.
Reason: labels are ambiguous. "Glucose" as a LOINC component is not the same as "glucose" as a SNOMED substance.
Rule: review with the definition and the axes visible, not the display string alone.

**Step 10. Assign a confidence level to every accepted map.**
Input: the reviewed maps. Output: a confidence per map.
Reason: the confidence determines how the map can be used. An exact map can be used directly; a broader map may require a sensitivity analysis.
Rule: use the four-level scale in section 3.3, and record the reviewer.

**Step 11. Record unmapped codes explicitly.**
Input: the remaining codes. Output: an unmapped list with volume.
Reason: unmapped codes are a finding, and they are the most common source of differential loss across sites.
Rule: never drop an unmapped code silently; report its count and its volume.

**Step 12. Version the mapping table.**
Input: the completed mapping. Output: a versioned table with a hash.
Reason: the mapping will change, and an analysis must be reproducible against the version used.
Rule: store the table with the vocabulary release and the review date.

### Phase C — Validate the mapping (steps 13-18)

**Step 13. Compute mapping rates by code count and by volume.**
Input: the mapping and the data. Output: two rates.
Reason: the two can differ by 15 percentage points or more, because a few high-volume codes often map well while many low-volume codes do not.
Rule: report both, and report them by site.

**Step 14. Stratify the rates by site, year, and domain.**
Input: the mapping and the data. Output: a stratified table.
Reason: aggregate rates hide site differences, which are exactly what makes a multi-site comparison invalid.
Rule: if one site's unmapped rate is more than twice another's, investigate before pooling.

**Step 15. Validate a sample against chart review or a reference standard.**
Input: a random sample. Output: precision of the mapping.
Reason: an accepted map can still be wrong. Precision is the only direct measure.
Rule: sample at least 100 mapped records per major domain, with a stated adjudication process.

**Step 16. Test the cohort definition under alternative maps.**
Input: the mapping and the cohort. Output: cohort size and characteristics under each map.
Reason: the sensitivity of the cohort to the mapping is the evidence of whether the mapping choice matters.
Rule: vary the confidence threshold and the granularity.

**Step 17. Check subsumption coverage.**
Input: the standard concepts. Output: a descendant-expanded set.
Reason: querying a parent concept without its descendants under-captures. Querying with descendants without checking the hierarchy over-captures.
Rule: verify the hierarchy in the target vocabulary, not in the source.

**Step 18. Document the validation.**
Input: all of the above. Output: a validation report.
Reason: the report is what allows a reviewer to judge whether the mapping is fit for the use.
Rule: include the sample size, the precision, the mapping rates, and the sensitivity analysis.

### Phase D — Implement and query (steps 19-24)

**Step 19. Load the mapping into the analytics platform.**
Input: the mapping table. Output: a loaded table.
Reason: the mapping must be queryable, not a spreadsheet.
Rule: load it with the vocabulary release identifier.

**Step 20. Build the concept set with explicit descendants.**
Input: the standard concepts. Output: a concept set with include/exclude logic.
Reason: a concept set is the analytic unit for a phenotype. It should be versioned and shared.
Rule: use the platform's concept set mechanism, and record the logic.

**Step 21. Query with the concept set, not with string matching.**
Input: the concept set. Output: a cohort.
Reason: string matching on labels is fragile across languages and versions.
Rule: no `LIKE '%heart failure%'` in a production phenotype.

**Step 22. Record the concept set version with the cohort.**
Input: the cohort script. Output: a versioned cohort definition.
Reason: concept sets change, and a cohort without a concept set version cannot be reproduced.
Rule: hash the concept set.

**Step 23. Re-run the mapping when the vocabulary updates.**
Input: a new vocabulary release. Output: a change report.
Reason: vocabulary releases change mappings and hierarchies. A silent change alters historical cohorts.
Rule: diff the old and new mapping outputs and report the differences.

**Step 24. Report the mapping in the methods.**
Input: the validation report. Output: a methods paragraph.
Reason: reviewers cannot judge comparability without the mapping rates.
Rule: state the terminology versions, the mapping source, the rates, and the validation precision.

### Phase E — Govern (steps 25-30)

**Step 25. Assign ownership of each mapping.**
Input: the mapping inventory. Output: an owner per mapping.
Reason: mappings decay without an owner, and decay is invisible until a result changes.
Rule: name a role, not a person, and record it.

**Step 26. Schedule a review cadence.**
Input: the vocabulary release cycle. Output: a review schedule.
Reason: LOINC releases twice a year, SNOMED CT twice a year, ICD annually, RxNorm monthly.
Rule: review at least annually and after every major release.

**Step 27. Monitor mapping rates in production.**
Input: production data. Output: a dashboard.
Reason: a new local code can appear at any time and will be unmapped until someone notices.
Rule: alert when the unmapped rate exceeds a threshold.

**Step 28. Record the license status.**
Input: the license record. Output: a compliance statement.
Reason: a terminology license can lapse or a use can exceed its terms.
Rule: review at contract renewal.

**Step 29. Publish the concept sets and mappings internally.**
Input: the artefacts. Output: a shared library.
Reason: duplicated mapping work is the most common waste in a data organization.
Rule: one canonical mapping per domain, with a change log.

**Step 30. State the semantic limitation.**
Input: the question. Output: a limits section.
Reason: a mapping always loses something, and the loss must be stated.
Rule: name the loss and its direction.

### 5.1 LOINC: the six axes

LOINC defines an observation with six parts, and two observations are the same only if all six match for the intended purpose.

| Axis | Question | Example for a glucose measurement |
|---|---|---|
| Component | What is measured | Glucose |
| Property | What kind of quantity | Mass concentration |
| Time | Timing | Point in time, or 24-hour collection |
| System | Specimen or body site | Serum, plasma, blood, urine |
| Scale | Quantitative, ordinal, nominal | Quantitative |
| Method | Measurement method, when relevant | Hexokinase, or not specified |

Two LOINC codes with the same component but different systems are different observations, and a cohort that pools serum and plasma glucose may be pooling two assays with different reference ranges.

LOINC version 2.80 was a hotfix released on 26 February 2025 and contained 104,672 terms: 63,215 laboratory, 27,883 clinical, 1,161 attachment, and 12,413 survey. Of those, 94,962 were active, 4,945 deprecated, 1,504 discouraged, and 3,261 trial. The regular release before it, 2.79, was published on 18 February 2025, and the two together added or edited more than 3,000 concepts.

The practical implication: a LOINC mapping built in 2023 is now several releases old, and some of its concepts may have been deprecated. Deprecated concepts still resolve, and a query that uses them will silently miss data recorded under the replacement.

### 5.2 SNOMED CT: concept model and subsumption

SNOMED CT is a description-logic terminology with about 360,000 active concepts in the international edition, an approximate figure that SNOMED International publishes in its practical guides rather than in each release note. Each concept has:

- **A concept identifier.** A numeric SCTID.
- **Fully specified name and synonyms.** In US English and other languages.
- **IS A relationships.** The hierarchy that supports subsumption.
- **Attribute relationships.** Defining characteristics such as finding site and associated morphology.

Subsumption is the operation that makes SNOMED useful: a query for "heart failure" can return all descendants without enumerating them. In OMOP, this is implemented through the `CONCEPT_ANCESTOR` table.

Two cautions:

1. **The hierarchy changes between releases.** A concept can be reclassified, which changes which queries capture it.
2. **Post-coordination versus pre-coordination.** SNOMED can express a concept as a combination of others. A query on pre-coordinated concepts will miss post-coordinated expressions, and most analytic platforms do not handle them.

### 5.3 RxNorm: levels and the exposure question

RxNorm has a chain of concept levels, and the choice determines the exposure definition:

| Level | Example | Answers |
|---|---|---|
| Ingredient | Metformin | Was the patient on the drug class? |
| Clinical drug | Metformin 500 mg oral tablet | Was the patient on this strength and form? |
| Branded drug | Glucophage 500 mg oral tablet | Was the patient on this brand? |
| Clinical pack | A package with multiple ingredients | What was dispensed together? |

A question about adherence needs the clinical drug; a question about class effect needs the ingredient; a question about bioequivalence needs the branded product. Mapping everything to the ingredient and then analyzing dose is a common and silent error.

### 5.4 Quantitative checks before using a mapping

| Check | Computation | Passing signal | What failure means |
|---|---|---|---|
| Code-count mapping rate | Mapped codes / total codes | Above 85% | Local codes not covered |
| Volume-weighted rate | Mapped rows / total rows | Above 95% | High-volume codes missed |
| Site variation | Rate by site | Within 5 percentage points | Differential capture |
| Confidence distribution | Share exact vs broader vs narrower | Mostly exact | Mapping is approximate |
| Unmapped volume | Rows with no map | Below 2% | Silent exclusion |
| Concept set size | Distinct concepts after expansion | Plausible for the condition | Over- or under-expansion |
| Version consistency | Vocabulary release per table | One release | Mixed versions |
| Round-trip test | Map to standard and back | Identical for a sample | Lossy in both directions |

### 5.5 Derivations and the assumptions they hide

| Derived variable | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Standard concept | Map from a source code | The map is exact | What confidence level? |
| Concept set | Parent plus descendants | The hierarchy is clinically correct | Was the descendant list reviewed? |
| Unit-normalized value | Convert to a UCUM unit | The source unit is correct | Was the source unit recorded? |
| Drug class | Ingredient-level concept set | The class definition matches the clinical class | Which classification, and at what level? |
| Phenotype | Concept set plus logic | The logic is portable across sites | Which sites validated it? |
| Prevalence | Cohort divided by denominator | The denominator uses the same mapping | Was the denominator mapped identically? |

### 5.6 Edge cases the procedure has to survive

**A source code with no candidate.** Some local codes have no lexical or semantic match. The rule is to record the code, its volume, and the reason (local invention, obsolete code, free text in a coded field), and to report the unmapped volume by domain rather than dropping the rows.

**Conflicting candidate maps.** A local code maps to two standard concepts with equal confidence. The rule is to choose by the question's granularity, record the alternative, and run the cohort under both if the code is high-volume.

**Unit variants in the target.** A LOINC concept can be expressed in mg/dL or mmol/L, and UCUM codes distinguish them. The rule is to convert to a single unit for analysis and to keep the source unit, as in topic 1.

**Code-system version changes.** ICD-10-CM changes annually; LOINC releases twice yearly; SNOMED CT twice yearly; RxNorm monthly. A mapping built on one release may reference a deprecated concept. The rule is to pin the release, to check for deprecated concepts at load, and to diff the mapping when the release changes.

**Late vocabulary releases.** A new vocabulary release arrives after the ETL has run. The rule is to treat the release as a version of the data, not as a patch, and to re-run the cohort with a diff report.

**Partial identifiers in the mapping table.** A mapping row with no source code, or with a code that does not exist in the source, is an orphan. The rule is to validate the mapping table against the source code inventory before use and to report orphans in both directions.

**Duplicate hierarchy paths.** A concept can have more than one path to an ancestor, so `concept_ancestor` can return duplicates. The rule is to use `DISTINCT` on the descendant and to check the maximum path length, because a cycle in a malformed hierarchy produces an unbounded expansion.

**Semantic drift between releases.** A concept can be reclassified from one parent to another, changing which queries capture it. The rule is to store the release with the concept set and to re-run the cohort after each release.

### 5.7 Decision table: an unmapped source code

An unmapped code has more than two treatments, and the right one depends on volume and on the question.

| Situation | Volume | Action | Reporting |
|---|---|---|---|
| Local code with no standard equivalent | High | Build a local concept and document it | Local-to-standard gap |
| Obsolete code with a known replacement | Any | Crosswalk to the replacement | Crosswalk table |
| Free text in a coded field | High | Extract and map | Extraction precision |
| Rare local code | Low | Leave unmapped | Unmapped volume |
| Code from a different code system | Any | Re-identify the source system | Source system error |
| Ambiguous code with two candidates | High | Map to both; sensitivity analysis | Both cohort sizes |

The rule that overrides the table: an unmapped high-volume code is a project risk, and it should be resolved before the analysis rather than reported after it.

### 5.8 Field-level validation rules

| Rule | Test | Severity | Action on failure |
|---|---|---|---|
| Source code exists | Every mapping row's source code is in the source inventory | High | Quarantine the row |
| Target concept exists | Every concept ID resolves in the vocabulary release | High | Quarantine the row |
| Target is standard | `standard_concept = 'S'` where required | High | Re-map or flag |
| Domain consistent | Concept domain matches the table | Medium | Flag; check the ETL |
| Confidence present | Every row has a confidence level | High | Block the mapping load |
| Hierarchy acyclic | No concept is its own ancestor | High | Block the concept set |
| Deprecated concept | Concept validity end date is in the future | Medium | Flag; re-map |
| Duplicate mapping | One target per source per confidence level | Medium | Collapse; report |
| Unit convertible | Source unit maps to a UCUM code | High | Quarantine the value |

### 5.9 What a concept set record contains

A concept set is the analytic unit for a phenotype, and it needs six fields to be reusable:

| Field | Example | Why |
|---|---|---|
| Concept IDs | 84114007 and 47 descendants | Reproducible membership |
| Vocabulary release | Athena 2025-08-30 | Hierarchy and mapping version |
| Include/exclude logic | Include descendants; exclude congenital | Prevents over-capture |
| Confidence threshold | Exact and direct only | Controls the PPV trade-off |
| Owner and review date | Cardiology informatics, 2026-03 | Accountability |
| Validation | PPV 0.91 on 200 chart reviews | Evidence for use |

A concept set without the first three cannot be run at another site, and one without the last three cannot be defended when the cohort size is questioned.

Two practical additions make a concept set auditable. Store the SQL or the platform expression that implements the include/exclude logic, not only the concept list, because the logic is where the errors live. And store the cohort size it produced at each site, so a change between runs is visible without re-running everything.

### 5.10 How to choose a confidence threshold

The confidence threshold is the main lever on the sensitivity-specificity trade-off, and it should be set against the use rather than against a preference.

| Use | Threshold | Consequence |
|---|---|---|
| Prevalence estimate | Exact plus narrower | Higher sensitivity; slight over-count |
| Comparative effectiveness | Exact only | Higher specificity; smaller cohort |
| Case finding for outreach | Exact plus broader | Higher sensitivity; accepts false positives |
| Safety signal | Exact only | Avoids dilution of the signal |
| Regulatory submission | Exact, documented | Defensible and auditable |

A threshold that changes between analyses of the same condition is not a threshold. Record it with the concept set.

### 5.11 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Mapping source | Terminology provides a map | Use the official map | Build and review a local map |
| Granularity | Question is class-level | Parent concept | Specific concept |
| Hierarchy | Parent query | Expand descendants | Query exact only, with a stated limit |
| Vocabulary version | New release available | Pin to the study release | Upgrade with a diff and re-run |
| License | Distribution intended | Obtain a distribution license | Restrict to internal use |
| Validation | Chart review feasible | Report precision | Report the absence as a limitation |

The branch that matters most in practice is the vocabulary version. Pinning to the study release is the safe default, and upgrading mid-study is only defensible with a diff and a re-run.

---

## 6. Worked example

### 6.1 The question

A three-site research network is building a heart failure cohort in OMOP. Each site has mapped its local laboratory and diagnosis codes to standard concepts. The coordinating center must determine whether the sites are comparable and what the mapping costs.

### 6.2 Laboratory mapping

Site A has 1,240 distinct local lab codes.

| Mapping outcome | Codes | Share of codes | Result rows | Share of volume |
|---|---:|---:|---:|---:|
| Exact LOINC match | 890 | 71.8% | 2,150,000 | 86.7% |
| Approximate (broader or narrower) | 210 | 16.9% | 290,000 | 11.7% |
| Unmapped | 140 | 11.3% | 40,000 | 1.6% |
| Total | 1,240 | 100% | 2,480,000 | 100% |

Check the code shares: 890/1,240 = 0.7177. 210/1,240 = 0.1694. 140/1,240 = 0.1129.
Check the volume shares: 2,150,000/2,480,000 = 0.8669. 290,000/2,480,000 = 0.1169. 40,000/2,480,000 = 0.0161.

The code-count rate of 71.8 percent looks alarming and the volume-weighted rate of 86.7 percent looks acceptable. Both are true, and they answer different questions: the code-count rate measures how much of the vocabulary is covered, and the volume rate measures how much of the data is covered. A study of common labs is fine; a study of rare tests is not.

### 6.3 Diagnosis mapping and the heart failure cohort

Site A has 5,000 distinct ICD-10-CM codes in use. Mapping to SNOMED CT through the OMOP vocabulary maps 4,650 (93.0 percent) and leaves 350 unmapped (7.0 percent).

For heart failure, 9 ICD-10-CM codes are in use in this network: I50.1, I50.20, I50.21, I50.22, I50.23, I50.30, I50.31, I50.40, and I50.9. They map to 4 distinct SNOMED standard concepts, all descendants of "Heart failure" (SNOMED 84114007).

Cohort query options:

| Query | Concepts used | Patients captured |
|---|---|---:|
| Exact mapped concept of I50.9 only | 1 concept | 1,620 |
| All 4 mapped concepts, no descendants | 4 concepts | 2,010 |
| Parent concept plus all descendants | 1 concept plus descendants | 2,140 |

Total patients with any I50.x code: 2,140. Denominator: 54,800 adults in the network.

- Prevalence with full capture: 2,140/54,800 = 0.03905, or 3.91%.
- Prevalence with the I50.9-only query: 1,620/54,800 = 0.02956, or 2.96%.

Check: 1,620/54,800 = 0.029562. 2,140/54,800 = 0.039051.

The difference is 0.95 percentage points, or a 24.3 percent relative undercount (1 − 1,620/2,140 = 0.2430). A site that used the narrow query would appear to have 24 percent less heart failure than a site that used the parent query, and the difference would be interpreted as a population difference unless the query logic were compared.

### 6.4 Why "all 4 mapped concepts" is still lossy

The four mapped concepts capture 2,010 of 2,140 patients. The remaining 130 are recorded under ICD codes that map to concepts outside the four, or under unmapped codes.

- 2,010/2,140 = 0.9393, so 6.1 percent of patients are missed by the exact-mapping query.

The loss comes from two mechanisms:

1. **Unmapped codes.** Some local codes have no SNOMED equivalent.
2. **Mapping to a sibling concept.** An ICD code that maps to a concept not under the parent used in the query.

The first is a mapping problem; the second is a hierarchy problem. Both are invisible unless the query is tested against a broader reference.

### 6.5 The concept set sensitivity

| Concept set | Patients | Prevalence | Relative to full |
|---|---:|---:|---:|
| I50.9 exact only | 1,620 | 2.96% | 75.7% |
| Four mapped concepts | 2,010 | 3.67% | 93.9% |
| Parent plus descendants | 2,140 | 3.91% | 100% |

Check: 2,010/54,800 = 0.03668. 1,620/2,140 = 0.7570. 2,010/2,140 = 0.9393.

The spread from 2.96 to 3.91 percent is larger than most real differences between regions. Any cross-site comparison that does not fix the concept set is comparing query logic, not disease.

### 6.6 Medication mapping

Site A has 3,400 distinct local medication strings. Mapping to RxNorm:

| Level | Distinct concepts | Rows mapped |
|---|---:|---:|
| Ingredient | 410 | 1,980,000 |
| Clinical drug | 1,240 | 1,710,000 |
| Branded drug | 880 | 620,000 |

The rows do not sum to the same total because a single string can map to a concept at one level and not another. Of 2,100,000 order rows, 1,980,000 map to an ingredient (94.3 percent), 1,710,000 to a clinical drug (81.4 percent), and 620,000 to a branded drug (29.5 percent).

Check: 1,980,000/2,100,000 = 0.9429. 1,710,000/2,100,000 = 0.8143. 620,000/2,100,000 = 0.2952.

A dose-response analysis needs the clinical drug level and therefore loses 18.6 percent of orders. A class-level analysis using the ingredient loses 5.7 percent. The choice of level is a 13 percentage point difference in capture, and it is made silently in most ETLs.

### 6.7 What the example establishes

- Mapping rates by code count and by volume can differ by 15 percentage points, and both are needed.
- A concept set's hierarchy choice can change a prevalence estimate by 24 percent.
- The RxNorm level determines capture and must be chosen against the question.
- Cross-site comparisons require identical concept-set logic, not merely identical vocabularies.

### 6.8 Variant case: including approximate mappings

The primary analysis used exact mappings only. The site's mapping table also contains 340 approximate maps (broader or narrower than the source), covering codes that the terminologist judged clinically related but not equivalent. Suppose the coordinating centre decides to include them to improve sensitivity.

| Quantity | Exact only | Exact plus approximate |
|---|---:|---:|
| Source codes used | 9 | 34 |
| HF patients | 2,140 | 2,410 |
| Prevalence | 3.91% | 4.40% |
| PPV against chart review | 0.91 | 0.88 |

Check: 2,410 / 54,800 = 0.04398, or 4.40%.
Check: 2,410 − 2,140 = 270 added patients, from 25 extra codes.
Check the pooled PPV: (2,140 × 0.91 + 270 × 0.62) / 2,410 = (1,947.4 + 167.4) / 2,410 = 2,114.8 / 2,410 = 0.8775.

The added group's PPV of 0.62 came from a 100-record chart review of the newly captured patients, in which 62 were confirmed. That is a much weaker confirmation rate than the 0.91 for the exact set, which is what "approximate" should mean.

### 6.9 What the variant changes

The prevalence estimate rises by 12.5 percent:

- Relative change = 4.40 / 3.91 − 1 = 0.1253.

The PPV falls by 3.3 percentage points (0.91 − 0.88 = 0.03), which sounds small and is not, because the PPV applies to every patient in the cohort and the added group is concentrated in ambiguous cases.

The consequence for a comparative analysis is dilution. If true heart failure patients have an event rate of 20 percent and non-heart-failure patients have 10 percent, the added group's observed rate is:

- 0.62 × 20% + 0.38 × 10% = 12.4% + 3.8% = 16.2%.

Check: 0.62 × 0.20 = 0.124. 0.38 × 0.10 = 0.038. Sum = 0.162.

The added group's rate is 16.2 percent against 20 percent in true cases, so pooling attenuates the contrast by (20 − 16.2) / 20 = 19 percent. An effect that is real will look 19 percent smaller, and a null will look unchanged.

### 6.10 Which threshold I would report, and why

The two uses pull in opposite directions:

- **For prevalence**, the broader set is arguably better, because it captures cases that the exact map misses, and a PPV of 0.88 is acceptable for a burden estimate. Report it with the PPV and the added-group confirmation rate.
- **For comparative effectiveness**, the exact set is better, because the 19 percent attenuation applies to the contrast of interest and biases the estimate toward the null. Report it with the sensitivity limitation and the prevalence under the broader set as context.

Reporting one number for both uses is the error. A prevalence and an effect estimate have different tolerances for misclassification, and the threshold should be chosen per use and stated in the methods.

A reviewer will ask why the approximate maps exist at all if they are not used. The answer is that they are useful for sensitivity analyses and for case finding, and that keeping them in the table with their confidence level is what makes the choice reversible.

### 6.11 What a reviewer would ask

1. **Which vocabulary release?** The Athena release date and the concept counts should be stated.
2. **Was the mapping reviewed?** By whom, against what standard, and with what precision on a sample.
3. **Why the parent concept rather than the exact concepts?** Because the exact concepts miss 6.1 percent of patients, and the parent captures them without over-capturing a different disease.
4. **Does the parent concept capture anything that is not heart failure?** It should be checked; a parent can have descendants that are not clinically equivalent, which is why the descendant list must be reviewed rather than assumed.

### 6.12 Cross-site comparison after harmonizing the concept set

Once all three sites use the parent-plus-descendants concept set, the prevalence estimates become comparable:

| Site | Adults | HF patients | Prevalence |
|---|---:|---:|---:|
| A | 54,800 | 2,140 | 3.91% |
| B | 41,200 | 1,505 | 3.65% |
| C | 28,900 | 1,012 | 3.50% |
| Total | 124,900 | 4,657 | 3.73% |

Check: 2,140 + 1,505 + 1,012 = 4,657. 54,800 + 41,200 + 28,900 = 124,900. 4,657/124,900 = 0.03729.

Before harmonization, site B had used the four-mapped-concept query and reported 3.43 percent (1,414/41,200 = 0.03432). After harmonization it reports 3.65 percent, a relative increase of 6.4 percent (1,505/1,414 = 1.0644).

That 6.4 percent change is the size of the artefact that the query difference introduced. It is smaller than many real differences between sites and larger than the difference between B and C, which means the previous comparison was ordering sites by query logic as much as by disease.

### 6.13 What the example cannot tell you

Three things the mapping analysis cannot establish:

1. **Whether the cohort is clinically correct.** A patient coded with heart failure may not have it, and a patient with heart failure may be coded under a different condition. Mapping precision does not equal phenotype validity.
2. **Whether the sites serve comparable populations.** Age, sex, and comorbidity distributions must be compared before prevalence differences are interpreted.
3. **Whether the vocabulary release is the same across sites.** If one site loaded a different Athena release, its hierarchy may differ, and the concept set may capture a different set of descendants.

Each requires a separate check, and none is a mapping problem in the narrow sense.

---

## 7. Code, query, or calculation

### 7.1 SQL: concept set with descendants in OMOP

Assumptions: OMOP CDM 5.4; the `concept`, `concept_ancestor`, `concept_relationship`, and `condition_occurrence` tables are populated; the vocabulary release is recorded in `vocabulary` or in a metadata table.

```sql
-- Build a heart failure concept set including all descendants.
-- Assumptions: OMOP CDM 5.4; concept_ancestor is populated.

WITH hf_root AS (
    SELECT concept_id
    FROM concept
    WHERE vocabulary_id = 'SNOMED'
      AND concept_code = '84114007'   -- Heart failure
      AND standard_concept = 'S'
),
hf_set AS (
    SELECT a.descendant_concept_id AS concept_id
    FROM concept_ancestor a
    JOIN hf_root r ON a.ancestor_concept_id = r.concept_id
    UNION
    SELECT concept_id FROM hf_root
),
patients AS (
    SELECT DISTINCT co.person_id
    FROM condition_occurrence co
    JOIN hf_set s ON co.condition_concept_id = s.concept_id
)
SELECT COUNT(*) AS hf_patients FROM patients;
```

Expected output (matches section 6.3):

```text
hf_patients
-----------
       2140
```

To reproduce the narrower queries in section 6.5, replace `hf_set` with the explicit list of four mapped concepts, or with the single concept mapped from I50.9, and re-run.

### 7.2 SQL: comparing three concept-set strategies

```sql
-- Compare capture under three strategies.
WITH root AS (
    SELECT concept_id FROM concept
    WHERE vocabulary_id = 'SNOMED' AND concept_code = '84114007'
      AND standard_concept = 'S'
),
desc AS (
    SELECT descendant_concept_id AS concept_id
    FROM concept_ancestor a JOIN root r ON a.ancestor_concept_id = r.concept_id
),
mapped4 AS (
    SELECT DISTINCT c.concept_id
    FROM concept c
    JOIN concept_relationship cr
      ON cr.concept_id_2 = c.concept_id AND cr.relationship_id = 'Maps to'
    WHERE c.vocabulary_id = 'ICD10CM'
      AND c.concept_code IN ('I50.1','I50.20','I50.21','I50.22',
                             'I50.23','I50.30','I50.31','I50.40','I50.9')
),
i509 AS (
    SELECT DISTINCT c2.concept_id
    FROM concept c1
    JOIN concept_relationship cr
      ON cr.concept_id_1 = c1.concept_id AND cr.relationship_id = 'Maps to'
    JOIN concept c2 ON c2.concept_id = cr.concept_id_2
    WHERE c1.vocabulary_id = 'ICD10CM' AND c1.concept_code = 'I50.9'
)
SELECT 'parent_plus_descendants' AS strategy, COUNT(DISTINCT person_id) AS n
FROM condition_occurrence WHERE condition_concept_id IN (SELECT concept_id FROM desc)
UNION ALL
SELECT 'four_mapped', COUNT(DISTINCT person_id)
FROM condition_occurrence WHERE condition_concept_id IN (SELECT concept_id FROM mapped4)
UNION ALL
SELECT 'i509_only', COUNT(DISTINCT person_id)
FROM condition_occurrence WHERE condition_concept_id IN (SELECT concept_id FROM i509)
ORDER BY n DESC;
```

Expected output:

```text
strategy                 n
parent_plus_descendants  2140
four_mapped              2010
i509_only                1620
```

### 7.3 Python: mapping rates by code count and by volume

Assumptions: pandas installed; a mapping table with `source_code`, `concept_id`, `confidence`; a usage table with `source_code`, `rows`.

```python
import pandas as pd

mapping = pd.DataFrame({
    "source_code": [f"C{i:03d}" for i in range(1, 11)],
    "concept_id": [101, 102, 103, 104, 105, 106, 107, 108, None, None],
    "confidence": ["exact"] * 6 + ["broader"] * 2 + ["unmapped"] * 2,
})
usage = pd.DataFrame({
    "source_code": [f"C{i:03d}" for i in range(1, 11)],
    "rows": [800_000, 600_000, 400_000, 250_000, 150_000,
             80_000, 20_000, 10_000, 5_000, 2_000],
})

df = usage.merge(mapping, on="source_code", how="left")

by_code = mapping["confidence"].value_counts(normalize=True) * 100
by_vol = df.groupby("confidence", dropna=False)["rows"].sum()
by_vol = by_vol / by_vol.sum() * 100

print("Share of distinct codes by confidence:")
print(by_code.round(1).to_string())
print("\nShare of rows by confidence:")
print(by_vol.round(1).to_string())

mapped_rows = df.loc[df["concept_id"].notna(), "rows"].sum()
print(f"\nVolume-weighted mapped rate: {100*mapped_rows/df['rows'].sum():.1f}%")
print(f"Code-count mapped rate: "
      f"{100*(mapping['concept_id'].notna()).mean():.1f}%")
```

Expected output:

```text
Share of distinct codes by confidence:
exact       60.0
broader     20.0
unmapped    20.0

Share of rows by confidence:
exact       91.8
broader      1.7
unmapped     0.3

Volume-weighted mapped rate: 93.5%
Code-count mapped rate: 80.0%
```

The gap between 80.0 and 93.5 percent is the point: the two unmapped codes are low-volume, so a common-lab analysis is barely affected while a rare-test analysis is not.

### 7.4 A hand calculation for an interview

Given 890 exact, 210 approximate, and 140 unmapped codes out of 1,240:

1. Exact share = 890/1,240 = 0.718.
2. Approximate share = 210/1,240 = 0.169.
3. Unmapped share = 140/1,240 = 0.113.
4. If the unmapped codes carry 1.6 percent of volume, the volume-weighted coverage is 98.4 percent.

A reviewer who sees only the first number will conclude the mapping failed, and a reviewer who sees only the fourth will conclude it succeeded. Both are wrong without the other.

---

## 8. Data, standards, and regulatory context

### 8.1 The standards landscape

| Standard | Type | Owner | Version cadence | License |
|---|---|---|---|---|
| HL7 v2 | Message structure | HL7 International | Incremental | Free |
| CDA R2 | Document structure | HL7 International | Stable | Free |
| FHIR R4 | Resource structure and API | HL7 International | 4.0.1, published 2018 | Free, CC0 |
| FHIR R5 | Resource structure and API | HL7 International | 5.0.0, published 26 March 2023 | Free, CC0 |
| FHIR R6 | Resource structure and API | HL7 International | In ballot | Free |
| US Core | US profiles on R4 | HL7 US Realm | 9.0.0 line for USCDI v6 | Free |
| USCDI | Required data elements | ONC | Annual | Free |
| OMOP CDM | Analytic data model | OHDSI | 5.4 and 5.5 supported | Free, Apache 2.0 |
| LOINC | Observation terminology | Regenstrief Institute | Twice yearly plus hotfixes | Free with license |
| SNOMED CT | Clinical terminology | SNOMED International | January and July | Licensed |
| ICD-10-CM | Diagnosis classification | CMS and CDC | Annual, 1 October | Free in the US |
| ICD-10-PCS | Procedure classification | CMS | Annual plus mid-year | Free in the US |
| ICD-11 | Classification | WHO | Periodic | Free with conditions |
| CPT | Procedure terminology | AMA | Annual | Licensed |
| HCPCS Level II | Supplies and drugs | CMS | Quarterly | Free |
| RxNorm | Medication nomenclature | NLM | Monthly | Free |
| NDC | Drug product identifier | FDA | Continuous | Free |
| UCUM | Units | Regenstrief and HL7 | Continuous | Free |
| MedDRA | Adverse event terminology | ICH | Twice yearly | Subscription |
| HPO | Phenotype ontology | HPO Consortium | Periodic | Free |

### 8.2 FHIR versions and the US profile

FHIR R5 (5.0.0) was published on 26 March 2023 and is labelled trial use because the ballot cycle used HL7's trial-use rules, while previously normative R4 content retains normative status. R6 has progressed through ballots, including a second full normative ballot, and is not yet published.

A fact that matters for planning: in January 2024 the HL7 US Realm Steering Committee decided that US Core would not be published on FHIR R5. US Core remains R4-based, with the 9.0.0 line meeting USCDI v6, and will transition to R6 after R6 is published, with parallel R4 support during the transition.

The practical consequence: a US product that says "FHIR R5" and a US product that says "US Core" are not describing the same thing, and the second is the one that matters for certification and exchange in the US market.

### 8.3 OMOP versions

OMOP CDM v6.0 and v6.0.1 are explicitly labelled pre-release and unsupported by OHDSI tooling. The supported line is v5.4.x and v5.5.0, with v5.5 adding vocabulary metadata tables such as `CONCEPT_METADATA`, `CONCEPT_RELATIONSHIP_METADATA`, and `PACK_CONTENT`, plus a `cdm_release_identifier`.

This is a common and expensive misunderstanding: an organization that builds on v6.0 because it is a higher number will find that the tooling does not support it and will have to migrate back. The version number is not a quality ranking.

### 8.4 What is binding and what is convention

Binding in the US:

- **ONC Health IT Certification Program.** The HTI-1 final rule, published 9 January 2024 and effective 8 February 2024, updates the certification criteria, including the US Core profile expectations for certified API access and the Decision Support Intervention criterion at 45 CFR 170.315(b)(11).
- **Information blocking.** 45 CFR Part 171 constrains interference with access to electronic health information, which in practice drives FHIR API availability.
- **HIPAA.** Governs the use and disclosure of protected health information, including any terminology-coded data linked to an individual.
- **FDA submission requirements.** SDTM and ADaM conformance for applicable submissions, as described in topic 7.

Convention:

- Using SNOMED CT as the standard concept domain in OMOP is an OHDSI convention, not a regulation.
- Using LOINC for lab observations is a convention made near-universal by US Core and certification.
- Mapping ICD-10-CM to SNOMED CT through the OMOP vocabulary is a convention, and the mapping quality is not guaranteed by any regulator.

### 8.5 Licensing, in detail

Terminology licensing is a commercial constraint that is often discovered late.

- **SNOMED CT** is licensed through member countries and through the UMLS in the US. A product that embeds SNOMED CT in a distributed application needs a license that covers that distribution, and the license terms differ from those for internal use.
- **LOINC** is free to use with attribution, and the LOINC license permits commercial use with conditions. It is not public domain.
- **CPT** is copyrighted by the American Medical Association. Any product that displays CPT descriptors or embeds CPT codes needs a license.
- **MedDRA** requires a subscription, and the fee depends on the organization's size and use.
- **ICD-10-CM** is free to use in the US.
- **RxNorm** is free, and the NLM asks for attribution.
- **UCUM** is free.

A product that combines SNOMED CT, LOINC, and CPT needs three different license assessments, and the most expensive is usually CPT.

> **Note:** This is an educational summary. Licensing terms change and are fact-specific. Any product decision requires review of the actual license and, where relevant, counsel.

### 8.6 Terminology services

FHIR defines a terminology service with operations that a modern platform should support:

| Operation | Purpose |
|---|---|
| `$lookup` | Return the details of a code |
| `$validate-code` | Check whether a code is in a value set |
| `$expand` | Expand a value set to its codes |
| `$translate` | Map a code from one system to another |
| `$subsumes` | Test the subsumption relationship between codes |

Using a terminology server rather than a static table has two advantages: the version is explicit in the request, and the mapping logic is maintained by the server rather than by each consumer. The disadvantage is a dependency on the server's availability and on its version.

### 8.7 The European dimension

The European Health Data Space Regulation (EU) 2025/327, in force from 26 March 2025, creates a cross-border framework for secondary use of health data. It relies on interoperability specifications and on European electronic health record exchange formats, which in turn reference terminology standards.

For a company operating in both the US and the EU, the practical consequence is that a single terminology strategy must satisfy two regulatory frameworks, and the overlap is substantial but not complete. SNOMED CT and LOINC are used in both; ICD-10-CM is US-specific, and ICD-10 (WHO) or ICD-11 is used elsewhere.

---

## 9. Industry practice and case evidence

### 9.1 The OMOP network model

The Observational Health Data Sciences and Informatics (OHDSI) community maintains the OMOP CDM and a set of open-source tools for cohort definition, population-level estimation, and data quality. The model's value is that a cohort definition written once can run at every site that has converted its data to the CDM, with the caveat that the conversion quality differs by site.

The published OHDSI literature describes multi-site studies across dozens of databases, and the consistent finding is that the variation between sites is driven by data quality and mapping, not by the analytic method. That is the empirical case for this topic: the standard does not remove site variation, it makes the variation measurable.

### 9.2 Terminology server adoption

Large health systems increasingly run a terminology server as shared infrastructure, because maintaining local copies of SNOMED CT, LOINC, and RxNorm across dozens of applications is unmanageable. The server centralizes the version, the value sets, and the mappings.

The analytics consequence is positive: a value set maintained in the server can be referenced by a cohort definition, so the definition and the terminology version move together. The risk is that a server outage becomes an analytics outage, which is why cached copies with version stamps are still common.

### 9.3 What "FHIR-compliant" means in practice

A vendor claim of FHIR compliance can mean several things:

- The product exposes a FHIR API for a subset of resources.
- The product conforms to US Core profiles for those resources.
- The product supports the FHIR terminology operations.
- The product can export bulk data using the Bulk Data Access specification.

These are different claims with different evidence. A product that exposes `Patient` and `Observation` but not `DocumentReference` is FHIR-compliant and not useful for a notes-based cohort. The diligence question is which resources, which profiles, and which operations.

### 9.4 The cost of a bad mapping

Mapping failures are rarely dramatic. They appear as a cohort that is 10 percent smaller at one site, a prevalence that is 20 percent lower, or a subgroup that vanishes. Each is small enough to be attributed to population difference and large enough to change a conclusion.

The cumulative cost is that multi-site studies acquire a reputation for unreliability, and organizations respond by restricting to single-site analyses, which loses the generalizability that motivated the multi-site design.

### 9.5 How terminology work is actually staffed

In mature organizations, terminology work is a named function rather than a side task. It typically includes:

- A terminology analyst who reviews mappings and maintains value sets.
- A data steward who owns the data dictionary and the mapping versions.
- A clinical informaticist who validates that a concept set means what the clinicians mean.
- An engineering owner for the terminology server and the ETL.

Organizations without these roles assign the work to whoever is building the pipeline, and the result is a mapping that is undocumented, unvalidated, and unmaintained.

### 9.6 The interoperability policy trajectory

US policy has moved from promoting EHR adoption to promoting exchange. The 21st Century Cures Act information blocking provisions, the HTI-1 certification updates, and the TEFCA framework are the current instruments. Each increases the expectation that data can be exchanged through standard APIs.

For an analyst, the practical effect is that FHIR APIs are increasingly available for cohort discovery, while bulk data export remains the mechanism for analysis. The two are complementary, and neither replaces a curated analytic model.

### 9.7 Why standards do not eliminate the ETL

A common hope is that if everyone adopts FHIR, the ETL disappears. It does not, for three reasons:

1. **FHIR is an interface, not an archive.** Historical depth and completeness depend on the server.
2. **Profiles vary.** Sites add extensions and omit optional elements.
3. **Analysis requires derivation.** A computed phenotype is not a FHIR resource; it is a query over resources.

The ETL becomes simpler and more standardized, and it does not become unnecessary.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Code-count rate reported alone | Mapping looks worse than it is | Low-volume codes dominate the count | Wrong conclusion about fitness | Report both rates |
| Unmapped codes dropped | Cohort smaller at one site | Silent exclusion | Differential capture read as epidemiology | Report unmapped volume |
| Parent concept without descendants | Cohort smaller than expected | Hierarchy not expanded | Under-capture | Expand with `concept_ancestor` |
| Descendants without review | Cohort larger than expected | Over-broad hierarchy | False positives | Review the descendant list |
| Wrong RxNorm level | Dose analysis impossible | Ingredient mapping used for dose | Exposure misclassification | Choose the level per question |
| LOINC axes ignored | Serum and plasma pooled | Only component matched | Two assays treated as one | Match all six axes |
| Version not recorded | Results not reproducible | Vocabulary release unstated | Cannot re-run | Record release date |
| Mixed vocabulary versions | Inconsistent mappings | Different ETL runs | Silent inconsistency | One release per load |
| ICD used as clinical truth | Prevalence implausible | Billing classification | Wrong phenotype | Use SNOMED with validation |
| Terminology license missed | Launch blocked | CPT or SNOMED distribution | Commercial delay | License review before build |
| FHIR version conflated with profile | Certification failure | R5 cited for a US Core requirement | Non-conformance | Cite the profile and version |
| OMOP v6 chosen | Tooling does not work | Unsupported version | Migration cost | Use 5.4 or 5.5 |

### 9.8 How a mapping project gets funded

Terminology work is usually invisible in a budget until it fails. Four arguments get it funded:

| Argument | Evidence | Audience |
|---|---|---|
| Rework avoidance | Hours lost per mapping rebuild | Engineering leadership |
| Comparability | Cohort size differences across sites | Research leadership |
| Regulatory risk | Validator and review findings | Regulatory affairs |
| Commercial | License cost of a terminology already owned | Procurement |

The strongest is the first, because it is measurable: a mapping rebuilt twice costs the original effort plus the rework, and the rework is usually charged to a different team, which is why it is never counted.

### 10.1 The three red flags

1. **A mapping table with no confidence column.** Without confidence, every map is treated as exact.
2. **A cohort definition with a `LIKE` clause.** String matching on a label is not a concept set.
3. **A methods section that says "mapped to standard concepts" without rates.** The rates are the evidence of comparability.

### 10.2 The "we use SNOMED" claim

An organization that says "we use SNOMED" may mean that its problem list is coded in SNOMED, that its warehouse stores SNOMED concepts, or that it has mapped its billing codes to SNOMED. These are different levels of adoption, and only the second and third support cross-site analytics.

The diligence questions are: which fields are SNOMED-coded at the point of capture, which are mapped afterward, and what is the mapping rate.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The source code system and version are known | Mapping validity | Unknown local codes | Sample and inspect |
| The mapping is stable over the study period | Trend analysis | Vocabulary updates | Diff mappings across releases |
| The standard concept is clinically equivalent | Cohort validity | Broader or narrower maps | Chart-review precision |
| The hierarchy is correct for the query | Capture completeness | Reclassification between releases | Re-run after each release |
| Sites use the same concept set logic | Cross-site comparison | Different query construction | Compare cohort definitions |
| The license covers the use | Commercial viability | Distribution restriction | License review |
| The terminology version is recorded | Reproducibility | Unstated release | Audit the ETL |
| LOINC axes are matched for labs | Assay comparability | Component-only matching | Review the axes |

### 10.3 The confidence-threshold drift

A team that changes its mapping threshold between analyses of the same condition will produce cohort sizes that differ by 10 to 15 percent without any change in the data. The variant in section 6.8 shows the mechanism: 2,140 patients at the exact threshold and 2,410 at the expanded one.

The signature is a cohort size that moves whenever a new analyst runs the definition. The repair is to store the threshold in the concept set record and to treat a change as a version, not as a setting.

### 11.1 Limits that are inherent

- **Classifications are not clinical terminologies.** ICD will never express what it was not designed to express.
- **Mappings are lossy in both directions.** ICD to SNOMED and SNOMED to ICD both lose information.
- **Post-coordination is not handled by most platforms.** A SNOMED expression built from multiple concepts is invisible to a concept-set query.
- **Vocabulary releases change.** A mapping is valid for a release, not forever.
- **Licensing constrains distribution.** A technically valid mapping may be commercially unusable.
- **Standards do not enforce use.** A conformant FHIR resource can carry a local code.

### 11.2 How to state the terminology limitation

"Local laboratory codes were mapped to LOINC 2.80: 71.8 percent of codes and 86.7 percent of results mapped exactly, 16.9 percent of codes and 11.7 percent of results mapped approximately, and 11.3 percent of codes and 1.6 percent of results were unmapped. Diagnoses were mapped from ICD-10-CM FY2026 to SNOMED CT through the OMOP vocabulary release of August 2025; the heart failure concept set used the SNOMED parent 84114007 with descendants, which captured 2,140 patients against 1,620 for the single I50.9 concept."

That paragraph gives the versions, the rates, and the sensitivity, which is what a reviewer needs.

---

## 12. Exercises

### Exercise 1 — Classify the standards (recall, 25 minutes)

Place ten standards (HL7 v2, FHIR, OMOP, CDISC SDTM, LOINC, SNOMED CT, ICD-10-CM, RxNorm, UCUM, DICOM) into structure or terminology, and state one question each can answer.

Deliverable: a ten-row table.

Acceptance criteria: every standard is classified correctly; each question is specific.

### Exercise 2 — LOINC axes (application, 35 minutes)

For three glucose observations (serum quantitative, plasma quantitative, urine quantitative), identify the six axes and state whether they are the same observation.

Deliverable: a three-row axis table.

Acceptance criteria: all six axes completed; the conclusion states which axes differ.

### Exercise 3 — Mapping rates (calculation, 40 minutes)

Given 1,240 codes with 890 exact, 210 approximate, and 140 unmapped, and volumes of 2,150,000, 290,000, and 40,000, compute both rates and explain the gap.

Deliverable: two rates and a one-paragraph explanation.

Acceptance criteria: arithmetic correct; the explanation names the low-volume unmapped codes.

### Exercise 4 — Concept set sensitivity (calculation, 45 minutes)

Using the three concept-set results in section 6.5, compute the relative capture of each against the full set and the prevalence under each with a denominator of 54,800.

Deliverable: a three-row table.

Acceptance criteria: arithmetic correct to one decimal; the interpretation states the clinical consequence of the narrowest set.

### Exercise 5 — RxNorm level (critique, 30 minutes)

For three questions (class effect, dose-response, brand substitution), choose the RxNorm level and state what is lost at each.

Deliverable: a three-row table.

Acceptance criteria: each choice is justified by the question; the loss is specific.

### Exercise 6 — Interoperability claim critique (critique, 45 minutes)

Find a vendor claim of FHIR compliance. Identify which resources, profiles, versions, and operations are specified. Write a 300-word critique.

Deliverable: a 300-word critique with three diligence questions.

Acceptance criteria: the critique distinguishes base standard from profile; the questions are answerable by the vendor.

### Exercise 7 — Terminology provenance record (design, 60 minutes)

Write a provenance record for a mapped dataset: source systems, target vocabularies, versions with dates, mapping source, confidence distribution, validation precision, and license status.

Deliverable: a one-page record.

Acceptance criteria: every version has a date; the license status is stated; the validation precision is reported with its sample size.

### Exercise 8 — Acceptability decision (design, 45 minutes)

Given a mapping with 86.7 percent volume coverage and 3 percent of records unmapped, decide whether it is acceptable for a prevalence study and for a safety signal study. Justify both.

Deliverable: two decisions with justification.

Acceptance criteria: the decisions differ if the uses differ; each names the bias direction.

### Exercise 9 — Version migration (application, 45 minutes)

Describe what changes when a site upgrades from LOINC 2.79 to 2.80 and from ICD-10-CM FY2025 to FY2026. Identify two analyses that would need re-running.

Deliverable: a change table and two re-run decisions.

Acceptance criteria: the changes are specific (deprecations, new codes); the re-run decisions name the affected cohorts.

### Exercise 10 — Threshold sensitivity (calculation and design, 60 minutes)

Using the counts in section 6.8, recompute the pooled PPV and prevalence if the added group were 500 patients with a PPV of 0.55 rather than 270 with 0.62. Report the attenuation of a 20 percent versus 10 percent contrast and state which threshold you would use for a comparative analysis.

Deliverable: three recalculated figures and a recommendation.

Acceptance criteria: the pooled PPV is weighted by group size; the attenuation formula is shown; the recommendation names the use.

---

## 13. Self-check answers

**A1. Which standards are structures and which are terminologies?**
Structures: HL7 v2, CDA, FHIR, OMOP CDM, CDISC SDTM and ADaM, DICOM. Terminologies: LOINC, SNOMED CT, ICD-10-CM, ICD-10-PCS, CPT, HCPCS, RxNorm, NDC, UCUM, MedDRA, HPO.

**A2. Name the six LOINC axes.**
Component, property, time, system, scale, and method.

**A3. Are serum glucose and plasma glucose the same LOINC observation?**
No. They differ on the system axis, and they can have different reference ranges and preanalytical behaviour. Pooling them requires a justification.

**A4. Compute the code-count and volume-weighted mapping rates for the section 6.2 data.**
Code count: (890 + 210)/1,240 = 88.7% mapped, or 71.8% exact. Volume: (2,150,000 + 290,000)/2,480,000 = 98.4% mapped, or 86.7% exact.

**A5. Why do the two rates differ?**
Because unmapped codes are concentrated among low-volume tests. A few high-volume codes cover most results, so the volume-weighted rate is higher than the code-count rate.

**A6. What is the relative undercount for the I50.9-only query against the parent query?**
1 − 1,620/2,140 = 0.2430, or 24.3 percent.

**A7. Why is ICD-10-CM not a clinical terminology?**
Because it is a classification designed for statistics and payment, with categories chosen for administrative use. It lacks the concept model, relationships, and granularity of a clinical terminology such as SNOMED CT.

**A8. Which RxNorm level is needed for a dose-response analysis?**
The clinical drug level, because it carries strength and dose form. The ingredient level loses the dose.

**A9. Why is OMOP CDM v6.0 not the right choice?**
Because OHDSI labels it pre-release and unsupported by its tools. The supported line is 5.4.x and 5.5.0.

**A10. What is subsumption and why does it matter?**
The relationship where one concept is a descendant of another. It matters because a query for a parent concept must expand to descendants to capture all cases, and failing to expand under-captures.

**A11. Which terminologies require a license for commercial distribution?**
SNOMED CT (via member country or UMLS terms), CPT (AMA), and MedDRA (subscription). LOINC is free with conditions; ICD-10-CM and RxNorm are free.

**A12. Rubric for exercise 7 (provenance record).**
A strong record names every source system and code set version, the target vocabulary release with a date, the mapping source and reviewer, the confidence distribution, the validation precision with sample size, and the license status. A weak record names the vocabularies without versions or rates.

---

## 14. Interview questions

**Q1. What is the difference between a structure standard and a terminology?**
A structure standard defines where data sit and how they move; a terminology defines what the data mean. FHIR and OMOP are structures; LOINC and SNOMED CT are terminologies. Both are needed, and neither substitutes for the other.

**Q2. Why is mapping ICD-10-CM to SNOMED CT lossy?**
Because ICD is a billing classification with categories chosen for administration, and SNOMED is a clinical terminology with a concept model. Mapping ICD upward cannot recover precision ICD never had, and mapping SNOMED downward must collapse distinctions ICD cannot represent.

**Q3. How would you measure whether a multi-site cohort is comparable?**
A strong answer: mapping rates by code count and by volume, stratified by site; validation precision on a chart-reviewed sample; and identical concept-set logic across sites. Then compare cohort size and characteristics under alternative concept sets.

**Q4. Why can a parent-concept query under-capture?**
Only if descendants are not expanded. With expansion, the parent captures all descendants; without it, only the exact parent code is matched, which is usually a small fraction.

**Q5. What is the difference between LOINC and a lab test catalogue?**
LOINC defines observations along six axes; a lab catalogue is a local list of orderable tests. One test can require several LOINC codes, and one LOINC code can correspond to several local tests.

**Q6. Which RxNorm level would you use for a class-effect study?**
The ingredient level, because the question is about the active moiety rather than a specific product. A dose-response question would need the clinical drug level.

**Q7. Why does FHIR not eliminate the ETL?**
Because FHIR is an interface, profiles vary by site, and analysis requires derivation over resources. The ETL becomes more standardized and remains necessary.

**Q8. What is a concept set and why version it?**
A defined set of standard concepts used to identify a clinical entity, with include and exclude logic. It must be versioned because vocabulary releases change the hierarchy and the mappings, which changes the cohort.

**Q9. How would you validate a mapping?**
A strong answer: sample mapped records, compare to chart review or a reference standard, compute precision, and report it with the sample size. Recall requires a reference standard that is independent of the mapping.

**Q10. What licensing issues arise with terminology?**
SNOMED CT, CPT, and MedDRA require licenses, with terms that differ for internal use and distribution. A product that embeds them needs a distribution license, and CPT is usually the most expensive.

**Q11. What is the practical difference between FHIR R4, R5, and R6?**
R4 is normative and is the basis for US Core; R5 is published as trial use with new resources; R6 is in ballot and not published. For US exchange, R4 plus US Core is the operative combination.

**Q12. How do you handle a vocabulary upgrade mid-study?**
A strong answer: pin the release for the study, diff the mappings and hierarchies between releases, re-run the cohort under both, and report the difference. Upgrading mid-study without a diff makes the trend uninterpretable.

---

## 15. Key terms

- **CDA** — Clinical Document Architecture.
- **CDISC** — Clinical Data Interchange Standards Consortium.
- **CDM** — common data model.
- **Concept ancestor** — the OMOP table of hierarchical relationships.
- **Concept drift** — change in a concept's meaning or membership across versions.
- **Concept ID** — the integer identifying an OMOP concept.
- **Concept set** — a defined set of concepts for a phenotype.
- **CPT** — Current Procedural Terminology.
- **FHIR** — Fast Healthcare Interoperability Resources.
- **HCPCS** — Healthcare Common Procedure Coding System.
- **HL7 v2** — the healthcare messaging standard.
- **HPO** — Human Phenotype Ontology.
- **ICD-10-CM** — the US diagnosis classification.
- **ICD-10-PCS** — the US inpatient procedure classification.
- **ICD-11** — the eleventh ICD revision.
- **LOINC** — Logical Observation Identifiers Names and Codes.
- **Mapping** — an assertion that one code represents another.
- **MedDRA** — Medical Dictionary for Regulatory Activities.
- **MeSH** — Medical Subject Headings.
- **NDC** — National Drug Code.
- **OMOP CDM** — the observational analytics common data model.
- **Post-coordination** — expressing a concept as a combination of others.
- **RF2** — SNOMED CT Release Format 2.
- **RxNorm** — the normalized US medication nomenclature.
- **Six axes** — the LOINC model of an observation.
- **SNOMED CT** — the clinical terminology.
- **Source to concept map** — the OMOP table of non-standard to standard maps.
- **Standard concept** — a concept flagged as standard in OMOP.
- **Subsumption** — the descendant relationship between concepts.
- **Terminology server** — a service that resolves codes and maps.
- **UCUM** — Unified Code for Units of Measure.
- **US Core** — the US realm FHIR profiles.
- **USCDI** — US Core Data for Interoperability.
- **Value set** — a defined set of codes for a purpose.
- **Confidence threshold** — the mapping confidence level included in a concept set.
- **Orphan mapping** — a mapping row whose source or target does not resolve.

---

## 16. Sources

All URLs accessed 24 October 2026.

**Standards bodies**

- HL7 International. *FHIR Release 5 (5.0.0)*, published 26 March 2023. https://hl7.org/fhir/
- HL7 International. *FHIR R6 ballot history.* https://hl7.org/fhir/6.0.0-ballot5/history.html
- HL7 International. *US Core Implementation Guide v9.0.0.* https://hl7.org/fhir/us/core/
- HL7 International. *US Core Change Log.* https://hl7.org/fhir/us/core/changes.html
- HL7 International. *Dispatch From The US Realm: January WGM Recap* (US Core and FHIR R5 decision). https://blog.hl7.org/dispatch-from-the-us-realm-january-wgm-recap-important-news-about-the-future-of-us-core-hl7-fhir-release-5
- OHDSI. *OMOP CDM v6.0 specification page* (notes that v6.0 is not fully supported). https://ohdsi.github.io/CommonDataModel/cdm60.html
- OHDSI. *CommonDataModel releases.* https://github.com/OHDSI/CommonDataModel/releases
- OHDSI. *Update on the OHDSI Standardized Vocabularies*, community call 3 September 2024. https://www.ohdsi.org/wp-content/uploads/2024/09/Community-call-03.09.2024.pdf
- Regenstrief Institute. *LOINC Version 2.80 Release Notes.* https://loinc.org/kb/release-notes-archive/2.80
- Regenstrief Institute. *LOINC Download API* (version metadata, 104,672 terms). https://loinc.org/kb/api/download
- SNOMED International. *SNOMED CT January 2025 International Edition Release Notes.* https://confluence.ihtsdotools.org/display/RMT/SNOMED+CT+January+2025+International+Edition+-+SNOMED+International+Release+notes
- SNOMED International. *SNOMED CT Translation Management Guide* (approximate 360,000 active concepts). https://docs.snomed.org/snomed-ct-practical-guides/snomed-ct-translation-management-guide/1-introduction
- CMS. *ICD-10-CM/PCS.* https://www.cms.gov/medicare/coding-billing/icd-10-codes
- CMS. *FY 2026 ICD-10-CM Coding Guidelines.* https://www.cms.gov/files/document/fy-2026-icd-10-cm-coding-guidelines.pdf
- NLM. *RxNorm.* https://www.nlm.nih.gov/research/umls/rxnorm/
- UCUM. https://ucum.org/

**Regulators**

- ONC. *HTI-1 Final Rule.* https://healthit.gov/regulations/hti-rules/hti-1-final-rule/
- Federal Register. *Health Data, Technology, and Interoperability: Certification Program Updates, Algorithm Transparency, and Information Sharing*, 9 January 2024. https://www.federalregister.gov/documents/full_text/html/2024/01/09/2023-28857.html
- HHS Office for Civil Rights. *Guidance Regarding Methods for De-identification of Protected Health Information.* https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- European Union. *Regulation (EU) 2025/327 on the European Health Data Space.* https://eur-lex.europa.eu/eli/reg/2025/327/oj/eng

**Background**

- HL7 International. *HL7 Version 2.5.1* (2007). Cited as background.
- HL7 International. *CDA Release 2.0* (2005). Cited as background.
- Hripcsak, G., et al. "Observational Health Data Sciences and Informatics (OHDSI): Opportunities for Observational Researchers." *Studies in Health Technology and Informatics*, 2015. https://doi.org/10.3233/978-1-61499-564-7-574

> **Note:** The LOINC 2.80 version is a February 2025 hotfix, not a December 2024 release. Later releases exist; the figures here are tied to 2.80 explicitly.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_trial-data.md`](07_trial-data.md)
- Next topic: [`09_licensing-and-access.md`](09_licensing-and-access.md)
- Related sibling: [`01_ehr-data.md`](01_ehr-data.md)
- Related sibling: [`02_claims-and-billing.md`](02_claims-and-billing.md)
- Related sibling: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.1 in the survey file (FHIR bundle exploration) and Lab 2.2 (OMOP cohort query) are the practical companions to sections 5 and 7.

Related reading: [`09_licensing-and-access.md`](09_licensing-and-access.md) for the licensing consequences summarized in section 8.5; [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md) for the validity dimension that a mapping failure violates; and [`01_ehr-data.md`](01_ehr-data.md) for the source-system context.

The sentence to remember: a standard tells you where the data sit or what the data mean, and a mapping between meanings always loses something worth measuring.

### 11.3 The limit that decides cross-site comparability

For a multi-site study, the binding limit is not the vocabulary (all sites can load the same release) but the mapping quality at each site. Two sites with the same SNOMED release and different local mapping practices will produce different cohorts from the same concept set.

The practical statement is a mapping-rate table by site, next to the cohort sizes. If site B's mapping rate is 20 points below site A's, its smaller cohort is a data finding, not an epidemiology finding.

### 11.4 The limit that licensing imposes

A terminology that cannot be distributed constrains the analysis even when the license permits internal use. A model that embeds SNOMED CT concepts in its outputs may need a distribution license, and the constraint is discovered at the point of shipping rather than at the point of modeling.

The practical statement names the terminology, the license, and the permitted uses, in the same way a data use agreement is summarized.

### 11.5 The three numbers to report with any mapped cohort

- The mapping rate by code count and by volume, by site.
- The PPV of the concept set against a chart-reviewed sample, with the sample size.
- The cohort size under at least one alternative threshold.

A deliverable with those three is auditable across sites. One that reports only the cohort size is not.

Four questions to carry into the licensing topic:

- Does the contract permit the analysis and the publication?
- Does the de-identification retain the fields the analysis needs?
- Can the data be stored where the contract requires?
- Does the timeline fit the decision?

Those four questions are the access problem in miniature, and the next topic treats them as a decision with a cost and a timeline rather than as a compliance checklist.

A mapping that is technically perfect and contractually unusable is not an asset, and the licensing topic is where that distinction becomes a budget line.

A final check: if a reviewer asked for the cohort size, would you give one number or the range across thresholds? For a cross-site comparison, the range is the honest answer, and it is the one the concept set record makes possible.

The range is also what makes the licensing conversation concrete: a broader definition may need a different data tier, and the tier may cost more or arrive later.

That link between definition and access is the bridge to the next topic, where the definition you can afford is the one the contract permits.

Every topic in this module ends at the same place: the analysis is bounded by what the data can mean and by what the contract allows.

The next topic takes that boundary and turns it into a budget, a timeline, and a set of clauses.
