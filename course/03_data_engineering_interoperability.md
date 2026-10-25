# Module 03 — Data Engineering and Interoperability

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 28 hours (10 lecture, 4 case clinic, 14 lab)
> **Track fit:** Common core
> **Prerequisites:** Module 02; Python; basic SQL
> **Tools:** DuckDB or PostgreSQL, dbt, Python, Great Expectations or dbt tests, Git, Docker
> **Industry question:** "Can you produce the same cohort, with the same definitions, on a new data refresh — and prove it?"

---

## 1. Why industry cares

A model is a small part of a data product. The expensive, fragile, and politically visible part is the pipeline that turns raw clinical records into a trustworthy analytic dataset.

Industry failures are usually pipeline failures:

- A code set changes and the cohort silently doubles.
- A unit conversion is applied twice.
- A lab feed starts sending nulls and the model treats them as normal.
- A site changes its EHR build and a field disappears.
- A patient moves between systems and is counted twice.
- A date shift breaks the temporal ordering.
- A model is retrained on data that includes the future.

This module teaches the engineering discipline needed to make biomedical analytics **reproducible, testable, observable, and auditable**.

## 2. Learning outcomes

By the end of this module, you can:

- Design a layered data architecture for clinical and biomedical data.
- Write reproducible SQL for cohort construction, deduplication, and temporal joins.
- Build a Python or dbt pipeline with explicit contracts, tests, and lineage.
- Ingest and validate FHIR resources and map them to an OMOP-style model.
- Handle DICOM metadata and imaging file organization without leaking identifiers.
- Apply de-identification and date-shifting while preserving analytic utility.
- Version data, code, and environments together.
- Implement data-quality monitors and alerting.
- Explain the trade-offs among warehouse, lake, lakehouse, and federated architectures.
- Produce a pipeline that another engineer can run from a clean checkout.

## 3. Mental model: the medallion, clinically adapted

```
Sources                 Raw / bronze          Standardized / silver       Curated / gold             Consumption
EHR extract      →      immutable landing      terminology-mapped          cohort + features     →    models
Claims           →      files, hashes          patient/encounter spine     quality flags         →    dashboards
FHIR API         →      schemas, metadata      deduplicated events         labels                →    regulatory
DICOM            →      provenance             unit-normalized            feature store         →    research
Wearables        →      access controls        de-identified              documentation         →    products
Omics            →                             provenance
```

**Principles:**

1. **Land raw data immutably.** Never transform in place. Keep the original extract and its hash.
2. **Separate standardization from feature logic.** Terminology mapping is reusable; a sepsis feature is not.
3. **Make time explicit.** Every event needs an effective time, a recorded time, and a source.
4. **Test at every boundary.** Schema, ranges, referential integrity, temporal ordering, and clinical plausibility.
5. **Version everything.** Code, config, data, model, and ontology versions.
6. **Minimize data movement.** Bring compute to governed data when possible.

## 4. Architecture patterns

| Pattern | Description | Strengths | Weaknesses | Best fit |
|---|---|---|---|---|
| Data warehouse | Structured, schema-on-write | Governance, performance, SQL | Less flexible for images/omics | Claims, curated EHR marts |
| Data lake | Raw files, schema-on-read | Cheap, flexible, all modalities | Governance and quality risk | Imaging, omics, raw waveforms |
| Lakehouse | Open table formats over object storage | ACID, versioning, mixed data | Operational complexity | Modern multimodal platform |
| Federated | Analysis travels to sites | Privacy, data residency | Heterogeneity, coordination | Multi-hospital consortia |
| Event streaming | Continuous ingestion | Timeliness | Complexity, ordering, replay | ICU monitors, wearables |
| Trusted research environment | Controlled workspace | Strong governance | Access friction | Sensitive national data |

### 4.1 Reference architecture

```
[Identity & access] ── [Audit & lineage] ── [Secrets & keys]
          │
[Ingestion] → [Raw zone] → [Standardization] → [Conformed zone] → [Feature/label zone]
          │         │              │                    │
     [FHIR API] [Checksums]   [Terminology]        [Cohort builder]
     [SFTP]     [PII scan]     [OMOP mapping]       [Quality flags]
     [Streaming][Retention]    [Unit conversion]    [Model training]
```

### 4.2 Batch vs streaming

| Characteristic | Batch | Streaming |
|---|---|---|
| Latency | Minutes–days | Milliseconds–seconds |
| Complexity | Lower | Higher |
| Reprocessing | Easy | Requires replay/retention |
| Best for | Claims, nightly EHR extracts, omics | ICU monitoring, wearable alerts, device telemetry |
| Failure mode | Stale data | Out-of-order, duplicates, partial events |

**Clinical rule:** the pipeline's latency must match the decision's latency. A sepsis alert that refreshes every 12 hours is a different product from one that refreshes every 15 minutes.

## 5. SQL for clinical data

### 5.1 Cohort construction

A cohort definition should be a versioned, testable artifact.

```sql
-- Adults with a first heart-failure encounter in 2024
WITH hf AS (
  SELECT
    p.person_id,
    vo.visit_occurrence_id,
    vo.visit_start_date,
    ROW_NUMBER() OVER (
      PARTITION BY p.person_id
      ORDER BY vo.visit_start_date, vo.visit_occurrence_id
    ) AS rn
  FROM person p
  JOIN visit_occurrence vo ON vo.person_id = p.person_id
  JOIN condition_occurrence co ON co.person_id = p.person_id
  WHERE co.condition_concept_id IN (SELECT concept_id FROM hf_concepts)
    AND vo.visit_start_date >= DATE '2024-01-01'
    AND vo.visit_start_date < DATE '2025-01-01'
    AND p.year_of_birth BETWEEN 18 AND 120
)
SELECT * FROM hf WHERE rn = 1;
```

**Questions to ask:**

- Is the cohort defined by first-ever, first-in-period, or any encounter?
- Are prevalent cases excluded by a lookback window?
- Are transfers merged?
- Are observation periods required?
- Are rule-out diagnoses excluded?
- Is the index date the admission, diagnosis, or treatment date?

### 5.2 Temporal joins

Clinical events are intervals, not points. Use half-open intervals `[start, end)` to avoid double counting.

```sql
-- Labs during an encounter, allowing 6-hour pre-admission window
SELECT e.encounter_id, l.lab_value, l.effective_time
FROM encounters e
JOIN labs l
  ON l.person_id = e.person_id
 AND l.effective_time >= e.admit_time - INTERVAL '6 hours'
 AND l.effective_time <  e.discharge_time;
```

### 5.3 Deduplication

Duplicates arise from interfaces, merges, and repeat transmissions.

```sql
SELECT *
FROM (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY person_id, code, effective_time
           ORDER BY recorded_time DESC, source_priority
         ) AS rn
  FROM observations
) t
WHERE rn = 1;
```

**Do not deduplicate blindly.** Two measurements at the same timestamp may be legitimate (e.g., bilateral blood pressures). Understand the source.

### 5.4 Gaps and islands

For continuous exposures (e.g., vasopressors, ventilation), collapse adjacent intervals.

```sql
WITH ordered AS (
  SELECT *, start_time - LAG(end_time) OVER (PARTITION BY person_id ORDER BY start_time) AS gap
  FROM infusions
),
grouped AS (
  SELECT *, SUM(CASE WHEN gap IS NULL OR gap > INTERVAL '1 hour' THEN 1 ELSE 0 END)
              OVER (PARTITION BY person_id ORDER BY start_time) AS grp
  FROM ordered
)
SELECT person_id, MIN(start_time) AS start_time, MAX(end_time) AS end_time
FROM grouped
GROUP BY person_id, grp;
```

### 5.5 Slowly changing dimensions

Patients change address, insurance, weight, and code status. Use type-2 dimensions with `valid_from` and `valid_to` when history matters.

## 6. Python engineering

### 6.1 Project layout

```
project/
├── README.md
├── pyproject.toml
├── config/
│   ├── base.yaml
│   └── sites/
├── src/
│   ├── ingest/
│   ├── standardize/
│   ├── cohorts/
│   ├── features/
│   └── quality/
├── tests/
│   ├── unit/
│   └── integration/
├── dags/
├── notebooks/        # exploration only; not production
└── data/
    ├── raw/          # immutable, access-controlled
    ├── interim/
    └── processed/
```

### 6.2 Configuration over code

```python
from pydantic import BaseModel, Field
from datetime import date

class CohortConfig(BaseModel):
    name: str
    index_start: date
    index_end: date
    min_age: int = Field(ge=0, le=120)
    lookback_days: int = Field(ge=0, default=365)
    exclude_prevalent: bool = True
```

**Why:** site-specific parameters should not be hard-coded, and invalid configurations should fail before a job runs.

### 6.3 Typed data and validation

Use pandas, polars, or PyArrow with explicit schemas. Validate dtypes, ranges, and categorical values at ingestion.

```python
import pandera as pa

schema = pa.DataFrameSchema({
    "person_id": pa.Column(int, nullable=False),
    "heart_rate": pa.Column(float, pa.Check.in_range(20, 300), nullable=True),
    "effective_time": pa.Column(pa.DateTime, nullable=False),
})
```

### 6.4 Testing

| Test type | What it catches | Example |
|---|---|---|
| Unit | Logic errors | Age calculation at boundary |
| Schema | Feed changes | Missing column |
| Referential | Orphan records | Lab without encounter |
| Temporal | Impossible sequences | Discharge before admission |
| Distribution | Drift | Lab values shift by 10× |
| Clinical plausibility | Data errors | Systolic BP < diastolic |
| Regression | Unintended change | Cohort count changes > 5% |

### 6.5 Logging and observability

Log:

- Run ID, code version, config hash.
- Row counts at each stage.
- Rejected rows and reasons.
- Runtime and resource use.
- Data-quality metric results.
- Source extract timestamp.

Never log PHI.

## 7. Orchestration and transformation

| Tool | Role | Strength | Caution |
|---|---|---|---|
| Apache Airflow | Workflow orchestration | Mature, broad ecosystem | Operational overhead |
| Dagster | Data orchestration | Asset-centric, lineage | Newer ecosystem |
| Prefect | Workflow orchestration | Pythonic, dynamic | Less clinical tooling |
| dbt | SQL transformation | Testing, docs, lineage | SQL-centric |
| Apache Spark | Large-scale processing | Scalable | Complexity |
| DuckDB | Local/embedded analytics | Fast, simple | Not a multi-user warehouse |
| Polars | DataFrame processing | Fast, lazy | Smaller ecosystem |

### 7.1 dbt example

```sql
-- models/marts/hf_cohort.sql
{{ config(materialized='table') }}

with hf as (
    select
        person_id,
        visit_occurrence_id,
        visit_start_date,
        row_number() over (
            partition by person_id
            order by visit_start_date, visit_occurrence_id
        ) as rn
    from {{ ref('stg_condition') }}
    where condition_concept_id in (select concept_id from {{ ref('hf_concepts') }})
      and visit_start_date >= '2024-01-01'
      and visit_start_date < '2025-01-01'
)

select * from hf where rn = 1
```

Add tests:

```yaml
models:
  - name: hf_cohort
    columns:
      - name: person_id
        tests: [not_null, unique]
      - name: visit_start_date
        tests: [not_null]
```

## 8. FHIR ingestion

### 8.1 Patterns

| Pattern | Description | Use |
|---|---|---|
| Bulk export | `$export` operation | Population-level extraction |
| REST search | Query resources | Targeted pulls |
| Subscription | Event notifications | Near-real-time updates |
| Batch bundle | Bundle of resources | Migrations, vendor feeds |

### 8.2 Ingestion checklist

- Authenticate with OAuth 2.0 and SMART on FHIR where applicable.
- Paginate and handle rate limits.
- Store raw resources with server URL, retrieval time, and version.
- Validate against profiles.
- Resolve references and detect dangling links.
- Normalize timestamps and units.
- Map codes to standard terminologies.
- Handle deletions and updates.
- Audit access.

### 8.3 FHIR-to-OMOP mapping

| FHIR resource | OMOP table | Mapping challenge |
|---|---|---|
| `Patient` | `PERSON` | Race/ethnicity concepts; birth date precision |
| `Encounter` | `VISIT_OCCURRENCE` | Visit type mapping; transfers |
| `Condition` | `CONDITION_OCCURRENCE` | SNOMED to standard concept |
| `Observation` | `MEASUREMENT` / `OBSERVATION` | Lab vs survey vs vital |
| `MedicationRequest` | `DRUG_EXPOSURE` | Order vs administration; dose normalization |
| `Procedure` | `PROCEDURE_OCCURRENCE` | Coding granularity |
| `DiagnosticReport` | `NOTE` / `MEASUREMENT` | Report vs result |
| `ImagingStudy` | Imaging extension | DICOM linkage |

**Rule:** validate mapping with a sample of source records reviewed by a clinician or informaticist.

## 9. DICOM and imaging pipelines

### 9.1 File organization

A typical hierarchy:

```
patient/study/series/instance.dcm
```

Use `StudyInstanceUID`, `SeriesInstanceUID`, and `SOPInstanceUID` for linkage, not file names.

### 9.2 Metadata extraction

Extract:

- Modality, body part, study date (shifted).
- Manufacturer, model, software version.
- Acquisition parameters: kVp, mA, slice thickness, kernel, field strength.
- Contrast agent and phase.
- Series description and protocol name.
- Pixel spacing and orientation.

### 9.3 De-identification

- Remove or replace identifying tags.
- Remove private tags.
- Handle burned-in annotations with OCR or manual review.
- Date-shift consistently per patient.
- Remove DICOM viewers and structured reports that may contain identifiers.
- Validate with a DICOM de-identification tool and manual audit.

### 9.4 Storage and access

- Object storage with lifecycle policies.
- Metadata database for query.
- Pixel data in a PACS or research image archive.
- Access controls and audit logs.
- Thumbnail/preview generation without exposing PHI.

## 10. Data versioning and lineage

| Layer | Tool examples | What to version |
|---|---|---|
| Code | Git | Commit SHA, branch, tag |
| Environment | Docker, conda, uv | Image digest, lockfile |
| Data | DVC, lakeFS, Delta Lake, Iceberg | Snapshot ID, hash |
| Schema | Avro, Protobuf, dbt docs | Schema version |
| Ontology | Terminology release | SNOMED/LOINC release |
| Model | MLflow, model registry | Model version, training data hash |
| Configuration | Git + secrets manager | Config hash |

**Lineage question:** given a prediction, can you trace back to the exact source records, transformations, and model version? If not, you cannot investigate an incident or support a regulatory audit.

## 11. De-identification and privacy engineering

### 11.1 Text de-identification

- Rule-based: regex for phone numbers, MRNs, dates.
- Statistical: named-entity recognition for names, locations, organizations.
- Hybrid: rules plus models plus human review.
- Evaluation: precision, recall, and residual risk by identifier type.

### 11.2 Date handling

Options:

- Remove dates.
- Shift dates by a random offset per patient.
- Generalize to month/year.
- Preserve intervals within a patient.

**Trade-off:** date shifting preserves intervals but can break linkage with external data and create impossible ages. Document the method.

### 11.3 Synthetic data

Synthetic data can support development, demos, and software testing. It should not be used to claim clinical validity. Validate that synthetic data preserves the relationships needed for the task and does not leak real records.

## 12. Security engineering for data pipelines

| Control | Purpose |
|---|---|
| Least privilege | Limit access to necessary data |
| Encryption at rest and in transit | Protect confidentiality |
| Key management | Separate keys from data |
| Network segmentation | Limit lateral movement |
| Audit logging | Detect and investigate access |
| Secrets management | Avoid credentials in code |
| Vulnerability scanning | Find known issues |
| Dependency pinning | Reproducible, auditable builds |
| Data loss prevention | Detect exfiltration |
| Backup and recovery | Resilience |
| Incident response | Contain and learn |

**Threat model early.** Ask: what are we protecting, from whom, and what happens if it leaks?

## 13. Industry case studies

### 13.1 OHDSI and multi-site OMOP

The Observational Health Data Sciences and Informatics (OHDSI) community maintains the OMOP CDM and open-source tools for multi-site observational research. Sites transform local data to a common model and run standardized analyses.

**What transferred:** common data model, open methods, distributed analysis.
**What did not:** mapping loss and site heterogeneity remain; results require local interpretation.

Source: OHDSI. https://ohdsi.org/

### 13.2 OpenSAFELY and trusted research environments

OpenSAFELY is a UK platform that allows approved researchers to run analyses against primary-care data inside a secure environment, with code review and audit, rather than extracting records.

**Industry lesson:** federated/trusted environments can enable research while reducing data movement, but they require strong governance and reproducible code.

Source: OpenSAFELY. https://www.opensafely.org/

### 13.3 All of Us Researcher Workbench

The All of Us Research Program provides a cloud-based researcher workbench with tiered access to participant data.

**Industry lesson:** access tiers, participant consent, and cloud controls are product features, not afterthoughts.

Source: All of Us Research Program. https://allofus.nih.gov/

### 13.4 MIMIC-IV reproducibility

MIMIC-IV provides documentation, schemas, and code examples, enabling reproducible cohort definitions.

**Industry lesson:** documentation and versioning are part of the dataset. Without them, "open data" is not reproducible.

Source: Johnson, A. E. W. et al., *Scientific Data*, 2023. https://doi.org/10.1038/s41597-022-01899-x

### 13.5 Imaging de-identification failures

Research has shown that burned-in text and metadata can re-identify imaging patients. De-identification must cover pixel data and headers.

**Industry lesson:** a DICOM header cleaner is not enough; validate the pixels too.

Source: National Electrical Manufacturers Association, DICOM de-identification guidance; peer-reviewed imaging privacy studies.

## 14. Labs

### Lab 3.1 — Cohort pipeline in DuckDB

**Objective:** Build a reproducible cohort from synthetic EHR tables.

**Steps:**

1. Create raw, standardized, and mart schemas.
2. Load Synthea CSV or OMOP demo data.
3. Build patient and encounter spines.
4. Define a cohort with inclusion/exclusion criteria.
5. Add tests for uniqueness, referential integrity, and temporal order.
6. Parameterize the date range.

**Deliverable:** SQL scripts and a cohort count report.

**Acceptance criteria:** rerun produces identical counts; tests pass; assumptions documented.

### Lab 3.2 — dbt project with tests and docs

**Objective:** Turn the cohort into a dbt project.

**Steps:**

1. Create staging and mart models.
2. Add schema tests and a custom clinical plausibility test.
3. Generate dbt docs and inspect lineage.
4. Add a source freshness check.
5. Run `dbt build` from a clean environment.

**Deliverable:** dbt project and lineage screenshot.

**Acceptance criteria:** all tests pass; lineage is understandable; no hard-coded site values.

### Lab 3.3 — FHIR ingestion and mapping

**Objective:** Ingest a FHIR bundle and map key resources to a flat analytic table.

**Steps:**

1. Parse and validate resources.
2. Resolve patient and encounter references.
3. Map observations to LOINC and units to UCUM.
4. Detect dangling references and duplicates.
5. Produce a timeline and a mapping-loss report.

**Deliverable:** ingestion script and mapping report.

**Acceptance criteria:** no silent drops; unmapped codes are listed; timestamps are timezone-aware.

### Lab 3.4 — Data-quality monitor

**Objective:** Implement six quality checks and an alert.

**Steps:**

1. Row count vs previous run.
2. Null rate for critical fields.
3. Range check for vitals.
4. Referential integrity.
5. Temporal ordering.
6. Distribution shift for a lab.
7. Write an alert rule and a runbook.

**Deliverable:** quality module and runbook.

**Acceptance criteria:** each check has a threshold, owner, and action; false-positive plan exists.

### Lab 3.5 — De-identification audit

**Objective:** Audit a synthetic dataset for identifiers.

**Steps:**

1. Scan structured fields for direct identifiers.
2. Run a text de-identification tool on notes.
3. Inspect DICOM headers if available.
4. Attempt a re-identification risk assessment.
5. Propose remediation and residual-risk acceptance.

**Deliverable:** audit report.

**Acceptance criteria:** no real PHI used; risks categorized; remediation is specific.

### Lab 3.6 — Pipeline incident simulation

**Objective:** Diagnose a broken pipeline.

**Data:** Instructor injects a schema change, unit change, and duplicate feed.

**Steps:**

1. Detect the failure.
2. Trace lineage to the source.
3. Quantify the impact on cohort and model.
4. Roll back or patch.
5. Write a post-incident review.

**Deliverable:** incident report.

**Acceptance criteria:** root cause identified; blast radius quantified; preventive action defined.

## 15. Assignment: reproducible cohort pipeline

Build a pipeline that:

1. Ingests a public or synthetic dataset.
2. Lands raw data immutably.
3. Standardizes codes and units.
4. Builds a cohort with a written definition.
5. Produces a feature table with a data dictionary.
6. Runs at least eight automated tests.
7. Generates a quality report.
8. Can be rerun from a clean checkout with one command.
9. Includes a lineage diagram and runbook.
10. Documents limitations and known failure modes.

## 16. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Reproducibility | 25% | Runs only on author's machine | Clean checkout works | Deterministic, versioned, tested |
| Data model | 20% | Ad hoc tables | Layered, documented | Extensible, standards-aligned |
| Testing | 20% | No tests | Core tests | Clinical plausibility and regression tests |
| Observability | 15% | Print statements | Logs and metrics | Alerting, runbook, lineage |
| Security and privacy | 10% | Ignores risks | Basic controls | Threat model and residual-risk plan |
| Documentation | 10% | Sparse | Clear README | Onboarding-ready |

## 17. Regulatory, ethical, and commercial lens

- **Audit trail:** regulated products need traceability from source to output. Pipeline logs are evidence.
- **Change control:** a schema or code change can alter clinical output. Treat it as a controlled change.
- **Data minimization:** do not copy more data than needed.
- **Retention:** delete data when the DUA or policy requires it.
- **Vendor management:** cloud and analytics vendors must meet security and privacy obligations.
- **Cost:** storage and compute are real. Optimize for total cost, not just speed.

## 18. Failure modes and red flags

- Transforming raw data in place.
- Hard-coded file paths and site names.
- No schema validation.
- Silent coercion of units.
- Cohort logic embedded in a notebook.
- No test for temporal ordering.
- Logging PHI.
- Using production data in development without controls.
- No rollback plan.
- Treating a pipeline success as a data-quality success.

## 19. Interview and discussion questions

1. **Walk me through a clinical data pipeline you would build.**
   Hint: ingestion, raw, standardization, cohort, features, tests, lineage, monitoring.
2. **How do you make a cohort reproducible?**
   Hint: versioned code/config/data, explicit definitions, tests, deterministic outputs.
3. **What is the difference between a data lake and a warehouse?**
   Hint: schema-on-read vs schema-on-write; governance and modality trade-offs.
4. **How would you detect a broken data feed?**
   Hint: freshness, row counts, schema, distributions, referential integrity, clinical plausibility.
5. **Why is `ROW_NUMBER()` both useful and dangerous?**
   Hint: dedup depends on correct partition and ordering; it can hide duplicates.
6. **How do you handle unit changes?**
   Hint: store original unit, convert to canonical unit, validate range, version conversion.
7. **What is a data contract?**
   Hint: agreed schema, semantics, quality, and ownership between producer and consumer.
8. **How do you de-identify free text?**
   Hint: hybrid NER/rules, human review, evaluation by identifier type, residual risk.
9. **What would you log in a clinical pipeline?**
   Hint: run metadata, counts, rejects, quality metrics; never PHI.
10. **How do you test a pipeline that depends on external data?**
    Hint: fixtures, contract tests, canary data, schema versions, integration environment.
11. **When would you use a federated architecture?**
    Hint: data residency, privacy, multi-site governance, heterogeneity costs.
12. **How do you handle late-arriving data?**
    Hint: event time vs processing time, watermarks, reprocessing, versioned snapshots.

## 20. Key terms

| Term | Definition |
|---|---|
| ETL / ELT | Extract-transform-load vs extract-load-transform |
| Data lake | Raw, schema-on-read storage |
| Data warehouse | Structured, schema-on-write analytics store |
| Lakehouse | Lake plus table format with ACID and versioning |
| Medallion architecture | Bronze/silver/gold layered data design |
| Data contract | Agreed interface and quality guarantees |
| Lineage | Traceability of data through transformations |
| Idempotency | Re-running a job produces the same result |
| Determinism | Same inputs produce same outputs |
| Slowly changing dimension | History-preserving dimension table |
| Watermark | Threshold for late-arriving events |
| Data freshness | How current the data are |
| Schema drift | Unplanned change in data structure |
| De-identification | Removing or transforming identifiers |
| Date shifting | Consistent offset applied to dates |
| Trusted research environment | Controlled environment for sensitive analysis |
| Federated analytics | Analysis without centralizing raw data |
| Feature store | Centralized, versioned feature definitions |
| Runbook | Operational procedure for incidents |

## 21. Sources and further reading

### Standards and tools
- HL7 FHIR: https://hl7.org/fhir/
- OMOP CDM: https://ohdsi.github.io/CommonDataModel/
- dbt: https://docs.getdbt.com/
- Apache Airflow: https://airflow.apache.org/
- Dagster: https://dagster.io/
- Great Expectations: https://greatexpectations.io/
- Delta Lake: https://delta.io/
- Apache Iceberg: https://iceberg.apache.org/
- DVC: https://dvc.org/
- lakeFS: https://lakefs.io/

### Governance and security
- HHS HIPAA Security Rule: https://www.hhs.gov/hipaa/for-professionals/security/index.html
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework
- NIST Privacy Framework: https://www.nist.gov/privacy-framework
- ONC HTI-1: https://www.healthit.gov/
- OpenSAFELY: https://www.opensafely.org/
- All of Us Research Program: https://allofus.nih.gov/

### Methods and industry
- OHDSI: https://ohdsi.org/
- Johnson, A. E. W. et al., "MIMIC-IV, a freely accessible electronic health record dataset," *Scientific Data*, 2023. https://doi.org/10.1038/s41597-022-01899-x
- Hripcsak, G. et al., "Observational Health Data Sciences and Informatics (OHDSI)," *Studies in Health Technology and Informatics*, 2015. https://doi.org/10.3233/978-1-61499-564-7-574
- DICOM Standard, de-identification: https://www.dicomstandard.org/

> **Next:** [Module 04](04_statistics_causal_inference.md) provides the statistical and causal foundation for turning data into defensible evidence.
