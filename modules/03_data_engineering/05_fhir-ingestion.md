# FHIR Ingestion

> Module 03 — Data Engineering and Interoperability
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: 02, 03
> Version 2026.3

## 1. Why this topic exists

Fast Healthcare Interoperability Resources (FHIR) is the dominant standard for exchanging clinical data
through modern application programming interfaces (APIs). It defines resources, references, search
parameters, and operations. It does not define an analytics schema. The gap between a FHIR server and a
cohort table is where most interoperability projects fail.

The failures are specific.

- A team extracts every `Observation` for a cohort and loads them into one table. Laboratory results,
  vital signs, social-history answers, and survey scores are mixed together. The mean "value" of the
  column is meaningless, and a query for creatinine returns values in mg/dL, µmol/L, and percent.
- A `Patient` resource has a birth date with year precision: `"birthDate": "1974"`. The loader casts it to
  a date of 1 January 1974. Every age calculation is wrong by up to 364 days, and the cohort's age
  threshold silently includes and excludes the wrong patients.
- An `Observation` references an `Encounter` that was not included in the export. The loader drops the
  row, and the missing 4% of results is never reported. The cohort's baseline lab rate is understated.
- A search returns the same resource twice because the server paginated while a new version was written.
  The loader inserts both, and the duplicate inflates the count of patients with a positive result.
- A `MedicationRequest` is mapped to a drug exposure table, but the resource represents an order, not an
  administration. The pipeline records that the patient received a drug they never took.
- A `Bundle` contains 500 resources and one `OperationOutcome` with a warning. The loader ignores the
  outcome, processes 499 resources, and reports success.

FHIR is precise about structure and permissive about content. A resource can be valid and still carry a
local code, a missing unit, a partial date, or a reference to a resource the receiver does not have. An
ingestion pipeline must make those gaps visible rather than converting them into plausible rows.

This topic covers the whole path: choosing an access pattern, landing raw resources, validating profiles,
resolving the reference graph, normalizing codes and units, flattening to analytics tables, and
quantifying what was lost.

> **Note:** Examples target FHIR R4 (4.0.1) because most production APIs and the US Core implementation
> guide are R4-based, while noting R5 (5.0.0) differences. A pipeline should record which release it
> consumes; resource shapes and search behavior differ between them.

## 2. Learning outcomes

By the end of this topic you can:

1. **Choose** between bulk export, REST search, subscription, and batch bundle ingestion for a stated
   workload, with the trade-offs named. Assessed in Exercise 1.
2. **Land** raw FHIR resources with server URL, retrieval time, version, and content hash, and explain why
   the raw form must be preserved. Assessed in Exercise 2.
3. **Resolve** references across a bundle and quantify dangling references by type. Assessed in Exercise 3
   and self-check item 4.
4. **Normalize** timestamps, codes, and units, including partial dates and missing units. Assessed in
   Exercise 4.
5. **Flatten** `Patient`, `Encounter`, `Observation`, `Condition`, and `MedicationRequest` into an
   analytics model with a mapping-loss report. Assessed in Exercise 5.
6. **Detect** duplicates using resource identity, version, and clinical key, and distinguish a retransmission
   from a legitimate repeat. Assessed in Exercise 6.
7. **Write** an ingestion reconciliation that accounts for every resource received. Assessed in Exercise 7.
8. **Critique** a FHIR-to-OMOP mapping for order-versus-administration and code-granularity errors.
   Assessed in Exercise 8.

## 3. The core mental model

FHIR ingestion is a funnel with an explicit loss account at every stage.

```text
SOURCE                    LANDED                     LINKED                  ANALYTIC
FHIR server      ->       raw resources      ->      resolved graph   ->     rows
$export / search          NDJSON + manifest          references             OMOP-shaped
                                                      validated              tested

  1,000 patients            18,400 resources           18,339 linked          12,115 rows
                            + 61 dangling              (99.7%)                (65.8% of resources)
                            + 264 duplicates
```

The funnel is not a failure. Most FHIR resources are not rows in an analytics table: a `Patient` becomes
one person row, an `Encounter` one visit, but a `Bundle` wrapper becomes nothing, and a `DiagnosticReport`
may become a note or a parent of observations. The requirement is that every loss is counted and
classified.

### 3.1 The resource graph

FHIR resources are nodes; `Reference` elements are edges. A `Patient` has encounters; an `Encounter` has
observations; an `Observation` may have a specimen; a `MedicationRequest` references a medication and a
patient.

```text
Patient 1 ---< Encounter 1 ---< Observation (lab)
   |               |                  |
   |               +---< Condition     +--- Specimen
   |               |
   +---< MedicationRequest --- Medication
   |
   +---< Observation (vital sign, no encounter)
```

Two properties matter for ingestion.

First, references are relative or absolute. `"Patient/123"` is relative to the server base;
`"https://emr.example.org/fhir/Patient/123"` is absolute. A bundle may mix both, and a loader that
compares strings will fail to match them.

Second, a reference may be unresolved. The target resource may not be in the export, may have been
deleted, or may be on another server. FHIR does not require a bundle to be closed under references. The
loader must decide: drop the row, keep it with a null foreign key, or quarantine it. Each choice changes
the analytics population.

### 3.2 Four access patterns

| Pattern | Mechanism | Strength | Weakness | Use |
|---|---|---|---|---|
| Bulk export | `$export` asynchronous operation | Efficient population extraction; NDJSON | Server support varies; async lifecycle | Cohort-wide or full-server extraction |
| REST search | `GET /Observation?patient=...` | Targeted, widely supported | Pagination overhead; rate limits | Small cohorts, incremental pulls |
| Subscription | Server pushes notifications | Near-real-time | Delivery guarantees vary; topic support | Updates for an active cohort |
| Batch bundle | `POST` a `Bundle` of resources | Atomic transaction; migrations | Size limits; not for bulk reads | Vendor feeds, migrations |

The choice is driven by volume and freshness, not preference. A 1,000-patient cohort with 18,400
resources is a bulk export problem. A single patient's latest lab is a search problem. A daily update for
an active registry is a subscription or a search on `_lastUpdated`.

### 3.3 Bulk export lifecycle

The `$export` operation follows the FHIR asynchronous request pattern:

```text
POST [base]/$export?_type=Patient,Observation&_since=...
  -> 202 Accepted
     Content-Location: https://server/fhir/$export-status/abc123

GET https://server/fhir/$export-status/abc123
  -> 202 In Progress
  -> 200 OK
     { "transactionTime": "...",
       "output": [ {"type":"Patient","url":".../Patient.ndjson"},
                   {"type":"Observation","url":".../Observation.ndjson"} ],
       "error": [ ... ] }
```

The `_since` parameter filters on `Resource.meta.lastUpdated`, not on clinical effective time. A resource
created today about an event in 2019 will appear in a `_since` export for today. That is correct for
change capture and wrong for a clinical time-window extract. The pipeline must distinguish the two.

### 3.4 Validation is layered

Validation happens at three levels:

1. **JSON syntax**: the payload parses.
2. **FHIR structure**: required elements exist, cardinalities hold, terminology bindings are respected.
3. **Profile conformance**: the resource satisfies a profile such as US Core, including must-support
   elements and extensions.

A resource can pass level 1 and fail level 3. Most production servers return structurally valid resources
that do not conform to the profile the receiver expects. The receiver must validate against the profile
it actually needs, not only against the base specification.

### 3.5 Mapping is a lossy translation, not a copy

FHIR is designed for exchange; OMOP is designed for analysis. The translation loses information in at
least four places:

- **Granularity**: FHIR `Observation` covers labs, vitals, surveys, and social history; OMOP separates
  `MEASUREMENT`, `OBSERVATION`, and survey tables.
- **Terminology**: FHIR uses LOINC, SNOMED CT, RxNorm, and local codes; OMOP requires a standard concept
  and retains the source value.
- **Time**: FHIR has `effectiveDateTime`, `issued`, and `meta.lastUpdated`; OMOP has start and end dates
  with different semantics per table.
- **Negation and absence**: FHIR can represent "not done" or "unknown" with data-absent-reason; OMOP has
  no universal representation, and a null is not the same as a negative finding.

A mapping document that lists resource-to-table pairs without addressing these four is incomplete.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| FHIR | Fast Healthcare Interoperability Resources | Treated as an analytics model | HL7 standard |
| Resource | A typed FHIR object such as `Patient` | Called a record | — |
| Reference | A link from one resource to another | Assumed always resolvable | Relative or absolute URL |
| Bundle | A container of resources | Assumed to be a complete graph | `transaction`, `batch`, `collection` |
| Profile | A constrained FHIR structure definition | Confused with the base resource | StructureDefinition |
| Implementation guide | A set of profiles and conformance rules | Confused with the specification | IG, e.g. US Core |
| CapabilityStatement | A server's declared capabilities | Assumed to be accurate | — |
| Search parameter | A queryable field | Assumed indexed | `_lastUpdated`, `patient`, `code` |
| `_include` | Include referenced resources in a search result | Confused with a join | — |
| `_revinclude` | Include resources that reference the matches | — | — |
| `_lastUpdated` | Filter on resource metadata update time | Confused with clinical time | Instant |
| Bulk export | `$export` asynchronous extraction | Confused with a search | NDJSON |
| NDJSON | Newline-delimited JSON | Confused with a JSON array | One resource per line |
| SMART on FHIR | OAuth 2.0 profile for clinical apps | Confused with FHIR itself | — |
| Scopes | Permissions granted to a client | Assumed to filter data | `patient/Observation.read` |
| Version | `meta.versionId` of a resource | Confused with the FHIR release | String |
| `meta.lastUpdated` | When the resource was last changed on the server | Confused with `issued` | Instant |
| Conditional create | Create only if a search finds no match | Assumed atomic | — |
| OperationOutcome | A resource describing an error or warning | Ignored by loaders | — |
| Contained resource | A resource embedded in another | Confused with a reference | `#id` |
| Dangling reference | A reference whose target is absent | Silently dropped | — |
| LOINC | Logical Observation Identifiers Names and Codes | Assumed to cover all local tests | Code system |
| UCUM | Unified Code for Units of Measure | Confused with display units | Code system |
| SNOMED CT | Clinical terminology | Confused with ICD | Code system |
| RxNorm | Medication terminology | Confused with NDC | Code system |
| CodeableConcept | A code plus system, version, display, and text | Read as a single string | — |
| Quantity | A value with a unit and comparator | Unit ignored | UCUM |
| Partial date | A date with reduced precision, e.g. `1974` | Cast to 1 January | FHIR date |
| Data absent reason | An extension explaining a missing value | Treated as null | Extension |
| Mapping loss | Resources or codes not represented in the target | Unreported | Count and percent |

Two terms need expansion.

**CodeableConcept.** A `CodeableConcept` can contain multiple codings, a text representation, and no code
at all. A loader that reads `code.coding[0].code` assumes an ordering that FHIR does not guarantee. Read
all codings, prefer the system you can map, and retain the text as a fallback.

**Partial date.** FHIR allows `YYYY`, `YYYY-MM`, and `YYYY-MM-DD`. Casting a year to a full date invents
precision. The correct approach is to store the precision alongside the value and to compute age with an
explicit rule, such as using 1 July for a year-only birth date and reporting the approximation.

## 5. The mechanics, step by step

The procedure ingests a FHIR cohort into an analytics model. It assumes R4 (4.0.1) with US Core profiles
and notes R5 differences where they change behavior.

### Step 1 — Define the cohort and the resource types

Input: the clinical question.
Output: a patient list or cohort query, and the resource types required.

Example: adults with a heart-failure diagnosis in 2024 require `Patient`, `Encounter`, `Condition`,
`Observation`, `MedicationRequest`, and `DiagnosticReport`. `Practitioner` and `Organization` are needed
only if the analysis uses provider or site attributes.

Reason: exporting every resource type for every patient is expensive and increases the PHI footprint.
The resource list is a data-minimization decision.

### Step 2 — Verify server capabilities

Input: the server's `CapabilityStatement`.
Output: a compatibility note stating which resources, search parameters, and operations are supported.

Check `$export` support, the maximum page size, supported search parameters, and the FHIR version. A
server may claim R4 and implement a subset; the capability statement is the evidence.

Reason: pipelines built against an assumed capability fail at the first request. The check takes minutes
and prevents a week of rework.

### Step 3 — Authenticate with the right grant

Input: the server's authorization model.
Output: a client registration, scopes, and a token acquisition procedure.

Use OAuth 2.0 with SMART on FHIR scopes for patient-context access, or a backend service grant for
system-level export. Request the minimum scopes: `patient/Observation.read` rather than
`patient/*.read`. Record the token issuer and expiry.

Reason: over-broad scopes widen the breach radius and often fail an access review. Under-scoped requests
fail late, after the pipeline is built.

### Step 4 — Choose the extraction pattern

Input: volume, freshness, and server support.
Output: bulk export, search, subscription, or bundle, with a pagination strategy.

For population extraction, use `$export` with `_type` and, if supported, `_typeFilter`. For incremental
updates, use `_lastUpdated` with a lookback overlap. For a small cohort, use search with `_include` to
pull referenced resources in the same request.

Reason: the pattern determines the pagination, rate-limit, and idempotency design. Mixing patterns in one
pipeline creates inconsistent version semantics.

### Step 5 — Set the extraction window and overlap

Input: the clinical time window and the source's update behavior.
Output: a `_since` value with an overlap, such as 48 hours before the last successful run.

`_since` filters on `meta.lastUpdated`. An overlap captures resources updated during the previous run.
Without it, a resource updated at the boundary can be missed. The overlap must be idempotent downstream.

Reason: change capture is not a watertight stream. The overlap is the practical substitute for a
watermark that the server does not provide.

### Step 6 — Paginate and respect rate limits

Input: the server's page size and rate policy.
Output: a pagination loop with backoff and a maximum page count.

Follow `Bundle.link[rel=next]` rather than constructing page URLs. Apply exponential backoff on HTTP 429
and 503. Record the number of pages and the elapsed time.

Reason: servers change page URLs and enforce rate limits. A hard-coded page parameter breaks; a
retry-free loop triggers throttling and partial extraction.

### Step 7 — Land raw resources immutably

Input: each page or NDJSON file.
Output: an object per extraction under a path with source, resource type, extraction time, and hash.

Store the original JSON, the server base URL, the request URL, the response status, the retrieval time,
and the ETag or version. Do not parse and re-serialize before landing; re-serialization can lose ordering,
extensions, or precision.

Reason: the raw resource is the evidence of what the server returned. When a mapping is challenged, the
raw form is the only way to re-derive the answer.

### Step 8 — Validate structure and profile

Input: the landed resources.
Output: a validation report with counts by error type.

Validate JSON, required elements, cardinality, and profile conformance. Separate fatal errors (cannot
parse, missing resource type) from warnings (missing must-support element, unknown code system). Do not
let one invalid resource fail an entire batch; quarantine it with the error.

Reason: a single malformed resource should not block 18,000 valid ones, but it must not disappear
silently either.

### Step 9 — Build the resource index

Input: the validated resources.
Output: a map from `(resourceType, id)` to the resource and its version.

The index is the basis for reference resolution. Include the full URL form and the relative form so that
both reference styles resolve. Record the version so that a later duplicate can be resolved by version.

Reason: without an index, reference resolution becomes a per-row lookup and the dangling count cannot be
computed.

### Step 10 — Resolve references

Input: the index and each resource's references.
Output: resolved links with a status of resolved, dangling, or external.

Normalize relative and absolute references to a canonical key. Treat a reference to another server as
external, not dangling. Record the reference path, such as `Observation.subject`, so the loss report can
say which relationship failed.

Reason: a dangling `subject` and a dangling `encounter` have different analytic consequences. The former
loses the patient link; the latter loses the visit attribution.

### Step 11 — Decide the policy for unresolved references

Input: the dangling references.
Output: a documented policy: drop, null, or quarantine.

For a missing `subject`, quarantine the resource; a clinical observation without a patient cannot be
used. For a missing `encounter`, keep the observation with a null encounter and flag it, because the
value is still clinically meaningful. For a missing `performer`, keep the row with a null performer.

Reason: the policy determines the denominator. Dropping all rows with any dangling reference removes
valid clinical data; keeping all rows hides broken linkages.

### Step 12 — Normalize identifiers

Input: patient and encounter identifiers.
Output: a stable enterprise identifier and a source identifier.

FHIR `Identifier` has a system and a value. Use the system to distinguish an MRN from a national ID.
Never join on the value alone, because two systems can issue the same string. Keep the source identifier
for audit and the enterprise identifier for linkage.

Reason: identifier collisions across facilities are common and produce merged patients that are hard to
detect after the fact.

### Step 13 — Normalize timestamps

Input: `effectiveDateTime`, `effectivePeriod`, `issued`, and `meta.lastUpdated`.
Output: UTC timestamps with an explicit precision and a flag for missing offsets.

FHIR `dateTime` may include an offset. If it does not, the time zone is unknown. Do not assume local time
silently; either apply a documented site default or store the value as unknown-offset and flag it.

Reason: a timestamp without an offset can shift a clinical event by hours, changing which encounter it
belongs to and whether it falls inside a window.

### Step 14 — Map codes to standard terminologies

Input: `CodeableConcept` values.
Output: a standard code, a source code, a mapping status, and a vocabulary version.

Read every coding, not only the first. Prefer the coding system the target model uses. When no coding
maps, retain the source text and code, and mark the row as unmapped. Record the vocabulary release used.

Reason: a silent drop of unmapped codes biases the cohort toward the conditions and tests that happen to
map well, which are often the ones the mapping team prioritized.

### Step 15 — Normalize units to UCUM

Input: `Quantity` values and units.
Output: a numeric value in a canonical unit, the original unit, and a conversion status.

Map display units such as `mg/dL` to UCUM codes such as `mg/dL` (the UCUM code is case-sensitive and
uses `mg/dL`). Handle `{score}` for unitless survey values. Reject or quarantine values with no unit when
the concept requires one. Never assume the unit from the code alone; the same LOINC code can be reported
in different units.

Reason: unit errors are among the most consequential data defects. A creatinine reported in µmol/L but
loaded as mg/dL is off by a factor of 88.4.

### Step 16 — Classify observations by category

Input: `Observation.category` and the code.
Output: a target table: measurement, observation, survey, or vital sign.

Use the category coding (`laboratory`, `vital-signs`, `survey`, `social-history`) and the code system.
Do not infer category from the value type. A numeric survey score is not a laboratory measurement.

Reason: mixing categories in one table destroys the meaning of aggregates and makes unit normalization
impossible.

### Step 17 — Deduplicate by resource identity and clinical key

Input: the classified observations.
Output: one row per clinical fact, with duplicates flagged.

First deduplicate on `(resourceType, id, version)`: the same version delivered twice is a retransmission.
Then examine `(patient, code, effectiveDateTime, value)` for clinical duplicates. Two identical results
at the same time may be a retransmission or a legitimate repeat; the source's order identifier or
specimen can disambiguate.

Reason: deduplication by clinical key alone can remove legitimate repeats, such as bilateral blood
pressures. Deduplication by resource ID alone keeps duplicates created by re-encoding.

### Step 18 — Map to the analytics model

Input: the normalized resources.
Output: rows in `person`, `visit_occurrence`, `condition_occurrence`, `measurement`, `drug_exposure`.

Map `Patient` to person, `Encounter` to visit, `Condition` to condition, `Observation` to measurement or
observation by category, and `MedicationRequest` to drug exposure with an explicit order-versus-
administration flag.

Reason: the target model's grain differs from FHIR's. A `MedicationRequest` is an order; a
`MedicationAdministration` is an administration. Mapping an order to an exposure without a flag asserts
that the patient received the drug.

### Step 19 — Handle updates and deletions

Input: resources with versions and `meta.lastUpdated`.
Output: an upsert policy and a deletion policy.

FHIR servers rarely send deletes in a bulk export. A resource removed from the source may remain in the
warehouse indefinitely. Use `meta.lastUpdated` and periodic reconciliation to detect stale rows, and
document whether the warehouse represents "current state" or "as observed."

Reason: a cohort built on stale rows can include patients whose diagnosis was entered in error and later
removed. The warehouse's temporal semantics must be explicit.

### Step 20 — Produce the mapping-loss report

Input: counts at every stage.
Output: a report with total resources, resolved, dangling, unmapped, duplicate, quarantined, and loaded,
by resource type.

The report is the deliverable that makes the pipeline auditable. A pipeline that reports "18,400
processed" without the breakdown cannot answer whether the cohort is complete.

Reason: mapping loss is not random. It concentrates in local codes, specialized assays, and
under-resourced sites. Without the report, the bias is invisible.

### Step 21 — Reconcile against the source

Input: the source's counts and the warehouse's counts.
Output: a reconciliation table with differences and explanations.

Ask the server for a count of the resource type for the cohort, if the API supports it, and compare.
Where counts differ, identify whether the difference is due to the extraction window, the cohort filter,
quarantine, or deduplication.

Reason: reconciliation catches extraction bugs that the pipeline's own counts cannot see, such as a
pagination loop that stops early.

### Step 22 — Record lineage and audit

Input: the run metadata.
Output: lineage events linking source endpoint, raw objects, and target tables, plus an access log.

Record the server URL, the query parameters, the token issuer (not the token), the resource counts, the
code and vocabulary versions, and the target snapshot. Emit OpenLineage or equivalent events.

Reason: an audit asks which source records produced a number. Without lineage, the answer requires
re-running the extraction against a source that may have changed.

### Step 23 — Resolve identifier collisions and superseded versions

Input: resources with identifiers and versions from multiple servers.
Output: one current resource per `(server, resourceType, id)`, with a version history.

Two servers can issue the same `Patient/123`; the pair is only unique with the server base. A resource can
be superseded by a higher `meta.versionId`; the lower version must be retained for audit but not used for
analysis. A resource can also be deleted; a bulk export may omit it, so absence is not proof of deletion.

| Case | Key | Rule |
|---|---|---|
| Same ID, different servers | `(server, type, id)` | Keep both; link by identifier, not ID |
| Same ID, higher version | `(server, type, id, version)` | Keep highest version for analysis |
| Same ID, no version | `(server, type, id)` | Keep latest `meta.lastUpdated`, flag ambiguity |
| Deleted resource | `(server, type, id)` | Tombstone; do not infer deletion from absence |
| Contained resource | `#id` within parent | Resolve within the parent only |

Reason: identifier collisions merge patients; version handling either drops corrections or duplicates
facts. Both are invisible in a resource count.

### Step 24 — Decide behavior for partial pages and partial bundles

Input: a paginated search or a bundle with an `OperationOutcome`.
Output: a commit rule per page and a failure rule per outcome.

Commit each page idempotently on its resource keys, record the page URL and count, and fail the run if a
page is missing. Parse `OperationOutcome` at the bundle level: an error fails the batch; a warning is
recorded and the affected resources are flagged.

| Event | Behavior | Evidence |
|---|---|---|
| Page 4 of 10 fails | Retry; if exhausted, fail the run | Page manifest gap |
| Bundle has an error outcome | Reject the bundle | Outcome code and diagnostics |
| Bundle has a warning outcome | Load with flag | Outcome in the quarantine table |
| Resource fails profile | Quarantine, continue | Validation message |
| Duplicate resource ID | Keep highest version | Version audit |

Reason: partial extraction is the most common cause of an understated cohort, and it produces no error
unless the page manifest is checked.

### Branch point A — The server does not support `$export`

Fall back to search with `_include` and careful pagination, or request a bulk extract from the vendor.
Do not page through millions of resources with a naive loop; the rate limits and the runtime make it
impractical. Document the limitation and the smaller cohort scope it implies.

### Branch point B — The resources are R5

Check the differences that affect ingestion: R5 reworked subscriptions, added resource types, and changed
some element cardinalities. The `Observation` and `Patient` shapes used here are largely stable, but
`MedicationRequest` and subscription behavior are not. Record the release in the raw manifest.

### Branch point C — Multi-site FHIR

Each site may use a different FHIR release, profile, and code system. Normalize at the site before
pooling, and run the mapping-loss report per site. A pooled count without site-level loss is not
comparable.

## 6. Worked example

A research team extracts a cohort of 1,000 adult patients with heart failure from a hospital FHIR R4
server using `$export`. The export contains 18,400 resources.

### 6.1 Resource counts

| Resource | Count | Notes |
|---|---:|---|
| Patient | 1,000 | Cohort list |
| Encounter | 3,200 | Inpatient and outpatient |
| Observation | 12,400 | Labs, vitals, surveys, social history |
| Condition | 900 | Includes heart failure and comorbidities |
| MedicationRequest | 700 | Orders, not administrations |
| DiagnosticReport | 200 | Reports that may duplicate observations |
| **Total** | **18,400** | |

Observation breakdown by category:

| Category | Count | Share |
|---|---:|---:|
| Laboratory | 8,900 | 71.8% |
| Vital signs | 2,600 | 21.0% |
| Survey | 600 | 4.8% |
| Social history | 300 | 2.4% |
| **Total** | **12,400** | **100%** |

The largest category is laboratory, which is also the category with the most unit and code variation.
That is not a coincidence; it is where the mapping effort should concentrate.

### 6.2 Code mapping

Of the 8,900 laboratory observations:

- 8,365 have a LOINC code that maps to a standard concept. Mapping rate: 8,365 / 8,900 = 93.99%.
- 535 have only a local code. They are retained with the local code and marked unmapped.
- Of the 8,365 mapped, 312 have no unit. They are loaded with a null unit and flagged, not imputed.
- Of the 8,053 with a unit, 47 use a non-canonical display string such as `mg/dl`; these are normalized to
  the UCUM code `mg/dL`.
- 23 use a unit that is not valid UCUM for the concept; these are quarantined for review.

The unmapped share, 535 / 8,900 = 6.01%, is not evenly distributed. A review of the 535 shows that 410
come from two senders that report esoteric assays under local codes. The cohort's creatinine analysis is
unaffected; its NT-proBNP analysis loses 18% of values. That is the kind of finding a mapping-loss report
exists to surface.

### 6.3 Reference resolution

| Reference path | Total references | Resolved | Dangling | External |
|---|---:|---:|---:|---:|
| `Observation.subject` | 12,400 | 12,389 | 11 | 0 |
| `Observation.encounter` | 9,800 | 9,762 | 38 | 0 |
| `Condition.subject` | 900 | 900 | 0 | 0 |
| `Condition.encounter` | 860 | 858 | 2 | 0 |
| `MedicationRequest.subject` | 700 | 700 | 0 | 0 |
| `MedicationRequest.medicationReference` | 640 | 630 | 10 | 0 |
| `Encounter.subject` | 3,200 | 3,200 | 0 | 0 |
| **Total** | **28,500** | **28,439** | **61** | **0** |

The dangling rate is 61 / 28,500 = 0.21%. The consequence depends on the path: a dangling `subject`
removes a row from the analytic population, while a dangling `encounter` removes attribution only.

Policy: an observation with a dangling `subject` is quarantined (11 rows). An observation with a dangling
`encounter` is kept with a null encounter and a flag (38 rows). A medication request with a dangling
medication reference is quarantined (10 rows) because the drug identity is unknown.

### 6.4 Duplicates

Of the 12,400 observations, 264 are duplicates by `(patient, code, effectiveDateTime, value)`:

- 201 are exact retransmissions with the same resource ID and version.
- 63 have different resource IDs but identical clinical content, likely re-encoded by an interface.

Deduplication removes 264 rows, leaving 12,136. The exact retransmissions are removed by resource
identity; the re-encoded duplicates are removed by clinical key with a documented rule. The 63 carry a
flag because they may be legitimate repeats; a manual review of 20 finds that all are duplicates from a
single sender.

### 6.5 Final counts

| Stage | Count | Calculation |
|---|---:|---|
| Resources received | 18,400 | — |
| Duplicates removed | 264 | 201 + 63 |
| Quarantined (dangling subject or medication) | 21 | 11 + 10 |
| Unmapped but retained | 535 | local codes |
| Final observation rows | 12,115 | 12,400 − 264 − 21 |
| Final patient rows | 1,000 | — |
| Patients with at least one observation | 993 | 7 have none |

The final observation count is 12,115, which is 65.84% of all resources received (12,115 / 18,400). That
percentage is not a quality metric; it is an accounting identity that depends on the resource mix.

### 6.6 The medication error

The 700 `MedicationRequest` resources are orders. Of them, 540 have a matching
`MedicationAdministration` in the source, but the export did not request that resource type. The pipeline
initially maps all 700 to `drug_exposure`, which asserts that 160 patients received a drug that was only
ordered.

The correction is to add `MedicationAdministration` to the export, map only administrations to exposures,
and retain orders in a separate `medication_order` table. The corrected count is 540 exposures and 700
orders, with 160 orders that were never administered. That difference, 160 / 700 = 22.9%, is large enough
to change a medication-adherence analysis.

### 6.7 Resolved staging schema

| Column | Type | Source element | Rule |
|---|---|---|---|
| `observation_id` | text | `Observation.id` | Required |
| `person_id` | text | resolved `subject.reference` | Quarantine if unresolved |
| `encounter_id` | text | resolved `encounter.reference` | Null if unresolved, flagged |
| `category` | text | `category.coding.code` | Controlled vocabulary |
| `loinc_code` | text | `code.coding[system=loinc].code` | Null if unmapped |
| `source_code` | text | first non-LOINC coding | Always retained |
| `value_num` | numeric | `valueQuantity.value` | Null if absent |
| `unit_ucum` | text | `valueQuantity.code` | Normalized; null if absent |
| `effective_at` | timestamptz | `effectiveDateTime` | UTC; flag unknown offset |
| `mapped` | boolean | derived | `loinc_code IS NOT NULL` |

The schema is deliberately narrow. It retains the source code alongside the standard code, keeps the
original unit until normalization, and separates "absent" from "unmapped" with two fields.

### 6.8 Reconciliation by resource type

| Resource | Received | Duplicates | Quarantined | Loaded to |
|---|---:|---:|---:|---:|
| Patient | 1,000 | 0 | 0 | 1,000 |
| Encounter | 3,200 | 0 | 0 | 3,200 |
| Observation | 12,400 | 264 | 21 | 12,115 |
| Condition | 900 | 0 | 0 | 900 |
| MedicationRequest | 700 | 0 | 10 | 690 |
| DiagnosticReport | 200 | 0 | 0 | 200 |

The Observation row reconciles as 12,400 − 264 − 21 = 12,115. The MedicationRequest row loses 10 because
the medication reference is unresolved and the drug identity is unknown. The DiagnosticReport is loaded
as a document index, not as measurements, because its observations are already in the Observation table.

### 6.9 Second run and idempotency

| Metric | Run 1 | Run 2 | Expected |
|---|---:|---:|---|
| Resources received | 18,400 | 18,400 | Equal |
| Observation rows | 12,115 | 12,115 | Equal |
| Quarantined | 21 | 21 | Equal |
| Staging digest | `8d41...2f0a` | `8d41...2f0a` | Equal |

The second run reads the same raw NDJSON objects and produces the same digest. If the second run produces
12,379 rows, the duplicate removal is not running; if it produces 12,094, the quarantine rule changed.

### 6.10 Diagnostic checks

**Check A: dangling reference rate.** Observed 61 / 28,500 = 0.214%. Threshold 0.5%. Pass. A rate above
0.5% usually means the export scope excluded referenced resources, not that the server is broken.

**Check B: unmapped code rate.** Observed 535 / 8,900 = 6.01%. Threshold 10%. Pass. The rate is within
tolerance, but the report breaks it down by sender because the concentration matters more than the total.

**Check C: missing unit rate.** Observed 312 / 8,365 = 3.73%. Threshold 2%. Fail. The units are missing
for one sender's creatinine assays. The rows are loaded with a null unit and flagged, and the sender is
asked to correct the interface. Without this check, the creatinine analysis would silently drop 3.73% of
values.

**Check D: timestamp offset coverage.** Observed 18 resources with no offset out of 12,400 = 0.145%.
Threshold 0.1%. Fail. The 18 are loaded with an unknown-offset flag and excluded from time-window
features until the source is corrected.

### 6.11 Decision table for a resource-count mismatch

| Symptom | First check | Likely cause | Action |
|---|---|---|---|
| Loaded < received | Quarantine report | Validation failures | Review and correct source |
| Loaded > received | Duplicate handling | Re-encoding or append | Deduplicate by identity |
| Observation count low | Page manifest | Pagination stopped early | Re-run from the missing page |
| Condition count low | Code filter | Local codes excluded | Retain unmapped codes |
| Patient count high | Identifier join | Same patient, two IDs | Review the crosswalk |
| Same count, different values | Version audit | Superseded resource dropped | Keep highest version |

## 7. Code, query, or calculation

### 7.1 Bundle ingestion and reconciliation (standard library, executable)

The script parses a synthetic R4 bundle, resolves references, maps LOINC codes, normalizes units,
deduplicates, and prints a reconciliation. It was executed on CPython 3.11.15.

```python
"""Minimal FHIR R4 bundle ingestion with reconciliation. Standard library only."""
from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP

BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [
        {"resource": {"resourceType": "Patient", "id": "p1",
                      "birthDate": "1950-03-14"}},
        {"resource": {"resourceType": "Patient", "id": "p2",
                      "birthDate": "1960"}},
        {"resource": {"resourceType": "Patient", "id": "p3",
                      "birthDate": "2010-07-01"}},
        {"resource": {"resourceType": "Encounter", "id": "e1",
                      "subject": {"reference": "Patient/p1"},
                      "period": {"start": "2024-01-05T08:00:00Z"}}},
        {"resource": {"resourceType": "Encounter", "id": "e2",
                      "subject": {"reference": "Patient/p2"},
                      "period": {"start": "2024-02-10T08:00:00+00:00"}}},
        {"resource": {"resourceType": "Encounter", "id": "e3",
                      "subject": {"reference": "Patient/p3"},
                      "period": {"start": "2024-04-01T08:00:00Z"}}},
        {"resource": {"resourceType": "Observation", "id": "o1",
                      "status": "final",
                      "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                                "code": "laboratory"}]}],
                      "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0",
                                           "display": "Creatinine"}]},
                      "subject": {"reference": "Patient/p1"},
                      "encounter": {"reference": "Encounter/e1"},
                      "effectiveDateTime": "2024-01-06T10:00:00Z",
                      "valueQuantity": {"value": 1.2, "unit": "mg/dL",
                                        "code": "mg/dL"}}},
        {"resource": {"resourceType": "Observation", "id": "o1",
                      "status": "final",
                      "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                                "code": "laboratory"}]}],
                      "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0"}]},
                      "subject": {"reference": "Patient/p1"},
                      "encounter": {"reference": "Encounter/e1"},
                      "effectiveDateTime": "2024-01-06T10:00:00Z",
                      "valueQuantity": {"value": 1.2, "unit": "mg/dL", "code": "mg/dL"}}},
        {"resource": {"resourceType": "Observation", "id": "o3",
                      "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                                "code": "vital-signs"}]}],
                      "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
                      "subject": {"reference": "Patient/p1"},
                      "encounter": {"reference": "Encounter/e1"},
                      "effectiveDateTime": "2024-01-06T10:05:00Z",
                      "valueQuantity": {"value": 78, "unit": "/min", "code": "/min"}}},
        {"resource": {"resourceType": "Observation", "id": "o4",
                      "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                                "code": "laboratory"}]}],
                      "code": {"coding": [{"system": "http://hospital.example/lab", "code": "LAB123"}]},
                      "subject": {"reference": "Patient/p2"},
                      "encounter": {"reference": "Encounter/e2"},
                      "effectiveDateTime": "2024-02-11T09:00:00Z",
                      "valueQuantity": {"value": 5.4, "unit": "mg/dL", "code": "mg/dL"}}},
        {"resource": {"resourceType": "Observation", "id": "o5",
                      "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                                "code": "laboratory"}]}],
                      "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0"}]},
                      "subject": {"reference": "Patient/p3"},
                      "encounter": {"reference": "Encounter/e999"},
                      "effectiveDateTime": "2024-04-02T10:00:00Z",
                      "valueQuantity": {"value": 0.9, "unit": "mg/dL", "code": "mg/dL"}}},
        {"resource": {"resourceType": "Observation", "id": "o6",
                      "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                                "code": "laboratory"}]}],
                      "code": {"coding": [{"system": "http://loinc.org", "code": "2160-0"}]},
                      "subject": {"reference": "Patient/p2"},
                      "encounter": {"reference": "Encounter/e2"},
                      "effectiveDateTime": "2024-02-11T10:00:00Z",
                      "valueQuantity": {"value": 1.1}}},
        {"resource": {"resourceType": "Observation", "id": "o7",
                      "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                                "code": "laboratory"}]}],
                      "code": {"coding": [{"system": "http://loinc.org", "code": "718-7"}]},
                      "subject": {"reference": "Patient/p1"},
                      "encounter": {"reference": "Encounter/e1"},
                      "effectiveDateTime": "2024-01-06T10:10:00Z",
                      "valueQuantity": {"value": 13.2, "unit": "g/dL", "code": "g/dL"}}},
        {"resource": {"resourceType": "Condition", "id": "c1",
                      "subject": {"reference": "Patient/p1"},
                      "encounter": {"reference": "Encounter/e1"},
                      "code": {"coding": [{"system": "http://snomed.info/sct",
                                           "code": "84114007"}]}}},
        {"resource": {"resourceType": "MedicationRequest", "id": "m1",
                      "subject": {"reference": "Patient/p1"},
                      "medicationReference": {"reference": "Medication/med999"},
                      "authoredOn": "2024-01-06"}},
    ],
}

LOINC = {"2160-0": "Creatinine", "8867-4": "Heart rate", "718-7": "Hemoglobin"}
LAB_CATEGORY = "laboratory"


def ref_key(reference: str) -> str:
    """Normalize a relative or absolute reference to Type/id."""
    if not reference:
        return ""
    if reference.startswith("#"):
        return reference
    parts = reference.rstrip("/").split("/")
    return f"{parts[-2]}/{parts[-1]}"


def ingest(bundle: dict) -> dict:
    entries = bundle.get("entry", [])
    index: dict[str, dict] = {}
    resources = []
    for entry in entries:
        res = entry.get("resource")
        if not res:
            continue
        resources.append(res)
        key = f"{res['resourceType']}/{res['id']}"
        index[key] = res

    counts = Counter(r["resourceType"] for r in resources)
    dangling = Counter()
    quarantined = []
    analytic = []
    seen_version = set()

    for res in resources:
        if res["resourceType"] != "Observation":
            continue
        subject = res.get("subject", {}).get("reference")
        subject_key = ref_key(subject) if subject else ""
        if not subject_key or subject_key not in index:
            dangling["Observation.subject"] += 1
            quarantined.append((res["id"], "dangling subject"))
            continue
        enc = res.get("encounter", {}).get("reference")
        enc_key = ref_key(enc) if enc else ""
        if enc and enc_key not in index:
            dangling["Observation.encounter"] += 1

        # duplicate by resource identity (same id delivered twice)
        ident = (res["resourceType"], res["id"])
        if ident in seen_version:
            continue
        seen_version.add(ident)

        coding = res.get("code", {}).get("coding", [])
        loinc_code = next((c["code"] for c in coding if c.get("system") == "http://loinc.org"), None)
        category = None
        for cat in res.get("category", []):
            for c in cat.get("coding", []):
                category = c.get("code")
                break
        qty = res.get("valueQuantity", {})
        value = qty.get("value")
        unit = qty.get("unit")
        if loinc_code and unit == "mg/dL" and loinc_code == "2160-0":
            value = float((Decimal(str(value)) * Decimal("88.4")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP))
            unit = "umol/L"
        analytic.append({
            "observation_id": res["id"],
            "person_id": subject_key.split("/")[1],
            "encounter_id": enc_key.split("/")[1] if enc and enc_key in index else None,
            "category": category,
            "loinc": loinc_code,
            "value": value,
            "unit": unit,
            "effective": res.get("effectiveDateTime"),
            "mapped": loinc_code is not None,
        })

    return {
        "counts": dict(counts),
        "total_resources": len(resources),
        "dangling": dict(dangling),
        "quarantined": quarantined,
        "analytic": analytic,
    }


result = ingest(BUNDLE)
print("counts", result["counts"])
print("total_resources", result["total_resources"])
print("dangling", result["dangling"])
print("quarantined", result["quarantined"])
print("analytic_rows", len(result["analytic"]))
for row in result["analytic"]:
    print(row)
```

Expected output:

```text
counts {'Patient': 3, 'Encounter': 3, 'Observation': 7, 'Condition': 1, 'MedicationRequest': 1}
total_resources 15
dangling {'Observation.encounter': 1}
quarantined []
analytic_rows 6
{'observation_id': 'o1', 'person_id': 'p1', 'encounter_id': 'e1', 'category': 'laboratory', 'loinc': '2160-0', 'value': 106.08, 'unit': 'umol/L', 'effective': '2024-01-06T10:00:00Z', 'mapped': True}
{'observation_id': 'o3', 'person_id': 'p1', 'encounter_id': 'e1', 'category': 'vital-signs', 'loinc': '8867-4', 'value': 78, 'unit': '/min', 'effective': '2024-01-06T10:05:00Z', 'mapped': True}
{'observation_id': 'o4', 'person_id': 'p2', 'encounter_id': 'e2', 'category': 'laboratory', 'loinc': None, 'value': 5.4, 'unit': 'mg/dL', 'effective': '2024-02-11T09:00:00Z', 'mapped': False}
{'observation_id': 'o5', 'person_id': 'p3', 'encounter_id': None, 'category': 'laboratory', 'loinc': '2160-0', 'value': 79.56, 'unit': 'umol/L', 'effective': '2024-04-02T10:00:00Z', 'mapped': True}
{'observation_id': 'o6', 'person_id': 'p2', 'encounter_id': 'e2', 'category': 'laboratory', 'loinc': '2160-0', 'value': 1.1, 'unit': None, 'effective': '2024-02-11T10:00:00Z', 'mapped': True}
{'observation_id': 'o7', 'person_id': 'p1', 'encounter_id': 'e1', 'category': 'laboratory', 'loinc': '718-7', 'value': 13.2, 'unit': 'g/dL', 'effective': '2024-01-06T10:10:00Z', 'mapped': True}
```

The duplicate `o1` is removed by resource identity: the same `(Observation, o1)` appears twice. The
observation `o5` keeps its row with a null encounter because the patient link resolves; only its encounter
attribution is lost. The local code `LAB123` is retained with `mapped=False`, not dropped. The creatinine
values are converted to µmol/L, including 0.9 mg/dL → 79.56 µmol/L.

### 7.2 SQLite staging query (executable)

Once the rows are in a staging table, the reconciliation can be expressed in SQL. SQLite 3.50.4 is the
engine.

```python
import sqlite3

con = sqlite3.connect(":memory:")
con.execute("""
CREATE TABLE observation_stage (
    observation_id TEXT, person_id TEXT, encounter_id TEXT,
    category TEXT, loinc TEXT, value REAL, unit TEXT, mapped INTEGER
)""")
rows = [
    ("o1", "p1", "e1", "laboratory", "2160-0", 106.08, "umol/L", 1),
    ("o3", "p1", "e1", "vital-signs", "8867-4", 78.0, "/min", 1),
    ("o4", "p2", "e2", "laboratory", None, 5.4, "mg/dL", 0),
    ("o5", "p3", None, "laboratory", "2160-0", 79.56, "umol/L", 1),
    ("o6", "p2", "e2", "laboratory", "2160-0", 1.1, None, 1),
    ("o7", "p1", "e1", "laboratory", "718-7", 13.2, "g/dL", 1),
]
con.executemany("INSERT INTO observation_stage VALUES (?,?,?,?,?,?,?,?)", rows)

report = con.execute("""
SELECT category,
       COUNT(*) AS rows,
       SUM(CASE WHEN mapped = 0 THEN 1 ELSE 0 END) AS unmapped,
       SUM(CASE WHEN unit IS NULL THEN 1 ELSE 0 END) AS missing_unit,
       COUNT(DISTINCT person_id) AS patients
FROM observation_stage
GROUP BY category
ORDER BY category
""").fetchall()
for row in report:
    print(row)
```

Expected output:

```text
('laboratory', 5, 1, 1, 3)
('vital-signs', 1, 0, 0, 1)
```

The laboratory group has five rows across three patients, one unmapped code, and one missing unit. The
report is the minimum evidence that the load is understood.

## 8. Data, standards, and regulatory context

### 8.1 FHIR releases and profiles

| Specification | Version | Status | Relevance |
|---|---|---|---|
| FHIR R4 | 4.0.1 | Widely deployed; normative content | Most production APIs and US Core |
| FHIR R4B | 4.3.0 | Intermediate release | Some vendors; limited adoption |
| FHIR R5 | 5.0.0, published 26 March 2023 | Current published release; not fully normative | New resources, reworked subscriptions |
| FHIR R6 | In ballot | Not published | Future |
| US Core | 7.0.0 (STU 7), 8 May 2024; later versions exist | US realm profiles on R4 | Constrains what US servers expose |
| Bulk Data Access IG | 2.0.0 (STU 2), 26 November 2021; 3.0.0 supersedes | `$export` | Population extraction |
| SMART App Launch | 2.x | OAuth 2.0 profile | App authorization |
| LOINC | Current release, updated twice yearly | Terminology | Observation codes |
| UCUM | Current specification | Units | Quantity units |
| SNOMED CT | National release | Terminology | Conditions and procedures |

FHIR R5 is the current published release but is not normative as a whole. Much of its content was balloted
as STU, while content that was normative in R4 remains labeled normative. A vendor claim of "R5
compliant" therefore needs a capability statement and a profile list, not a version string.

US Core 7.0.0 is based on R4 and updates to USCDI v4, including must-support `meta.lastUpdated` on several
profiles. Later US Core versions exist; a pipeline should record the version it validates against, because
must-support requirements change between releases.

### 8.2 What FHIR requires and what it does not

FHIR requires:

- Resources conform to their structure definitions.
- References use the `Reference` datatype with a `reference` string.
- Servers declare capabilities in a `CapabilityStatement`.
- Bulk export follows the asynchronous request pattern.

FHIR does not require:

- A bundle to be closed under references.
- A server to support every search parameter.
- A code to be present in a `CodeableConcept`.
- A unit to be present in a `Quantity`.
- A receiver to preserve resources it does not understand.
- A particular analytics schema.

The last point is the one teams forget. FHIR is an exchange standard; the analytics model is a separate
design decision.

### 8.3 Regulatory context

- **ONC certification.** The ONC Health IT Certification Program includes FHIR-based API criteria, and the
  Interoperability Standards Platform lists approved standards and versions. Certification constrains
  products that seek it; it does not require a hospital's internal pipeline to use a specific FHIR
  release.
- **Information blocking.** The 21st Century Cures Act information-blocking rules affect access to
  electronic health information. A pipeline that cannot extract data because a vendor restricts an API may
  raise an information-blocking question; that is a legal determination, not an engineering one.
- **HIPAA.** FHIR resources can contain PHI. The raw zone, the access controls, the audit logs, and the
  de-identification boundary are all in scope. A FHIR bundle stored in an ungoverned bucket is a
  disclosure risk regardless of its format.
- **21 CFR Part 11.** If FHIR-derived data support a regulated submission, the ingestion and
  transformation must be attributable and auditable. Raw resource versions and run metadata are part of
  the evidence.

### 8.4 What the standards leave open

- Which resources constitute a complete patient record.
- How to treat a missing `encounter` reference.
- Whether a `MedicationRequest` is an exposure.
- How to handle partial dates.
- How to resolve duplicate resources across servers.
- How to map local codes.

These are the design decisions in Section 5. A team that can quote the FHIR specification but cannot state
its dangling-reference policy has not designed an ingestion.

## 9. Industry practice and case evidence

### 9.1 Bulk export as the population path

The FHIR Bulk Data Access implementation guide defines `$export` at system, group, and patient levels,
with NDJSON output and an asynchronous status workflow. It is the standard mechanism for population
extraction and is referenced in ONC certification criteria. The practical caveat is support: not every
server implements `$export`, and those that do may limit `_type` or `_since`. A pipeline should probe
capabilities before committing to the pattern.

### 9.2 US Core and vendor APIs

US Core profiles define the minimum data elements that US EHRs expose through certified APIs. They include
`Patient`, `Encounter`, `Condition`, `Observation`, `MedicationRequest`, and others, with must-support
elements and terminology bindings. The profiles reduce variation but do not eliminate it: a must-support
element can be present with a null value and a data-absent-reason extension, and local codes can appear
alongside standard codes. Validation against US Core is necessary and not sufficient for analytics.

### 9.3 FHIR-to-OMOP mapping

The OHDSI community maintains mapping guidance and tools for FHIR-to-OMOP conversion. The documented
challenges match Section 3.5: `Observation` covers multiple OMOP domains, terminology mapping is lossy,
and time semantics differ. A mapping should be validated on a sample reviewed by a clinician or
informaticist, and the mapping-loss report should be part of the deliverable.

### 9.4 What is inference

- That `$export` is faster than search for a 1,000-patient cohort. It depends on the server's
  implementation and the resource count; measure both.
- That US Core conformance guarantees analytics usability. It guarantees a minimum element set, not
  correct values, complete units, or resolvable references.
- That a 94% LOINC mapping rate is good. The acceptable rate depends on which codes are missing and
  whether they are central to the analysis.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Category mixing | One table has labs and surveys | `Observation` loaded as one entity | Meaningless aggregates | Classify by category |
| Partial date cast | Ages cluster at 1 January | `"1974"` cast to a full date | Age misclassification | Store precision; document rule |
| Dangling references dropped | Row counts fall without explanation | Missing target resource | Understated cohort | Quarantine and report |
| Duplicate resources | Counts exceed source | Retransmission or re-encoding | Inflated rates | Dedup by identity and clinical key |
| Unit assumed from code | Values off by a constant factor | Missing or ignored `Quantity.unit` | Dangerous misinterpretation | UCUM normalization; quarantine |
| Order mapped as administration | Exposure count too high | `MedicationRequest` used as `DrugExposure` | False adherence | Use `MedicationAdministration`; flag orders |
| `_since` used as clinical time | Old events appear in new extracts | Filter on `meta.lastUpdated` | Wrong study window | Separate change time from event time |
| Pagination stops early | Counts lower than source | Ignored `next` link or rate limit | Incomplete cohort | Reconciliation; page count check |
| OperationOutcome ignored | Partial success reported as success | Warnings not parsed | Silent data loss | Parse outcomes; fail on errors |
| Absolute/relative refs mismatch | Dangling count inflated | String comparison | Valid links lost | Normalize references |
| Token logged | Credential in logs | Debug logging | Security incident | Redaction; secret handling |
| Profile validation skipped | Resources accepted that lack must-support | No validation layer | Downstream nulls | Validate against the required profile |

Two red flags deserve emphasis.

**A pipeline that reports only a total.** "18,400 resources processed" says nothing about how many were
quarantined, deduplicated, or unmapped. The reconciliation table is the minimum evidence of correctness.

**A mapping table with no unmapped column.** If every source code maps, either the mapping is
extraordinarily complete or unmapped rows are being dropped before the report.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The server returns all resources for the cohort | Cohort completeness | `$export` scoped by group or filter | Reconcile counts with source |
| References are resolvable within the export | Graph linkage | Resources outside the cohort | Dangling-reference report |
| Codes are present and standard | Terminology mapping | Local-only codes | Mapping-loss report |
| Units are present and UCUM | Safe aggregation | Missing or non-UCUM units | Unit validation rate |
| `meta.lastUpdated` reflects changes | Incremental extraction | Server does not update it reliably | Compare successive exports |
| Patient identity is stable | Linkage | Identifier reuse or merge | Duplicate-patient audit |
| The FHIR release is consistent | Parser correctness | Mixed R4 and R5 sources | Version manifest per source |
| Profile validation reflects analytics needs | Data quality | Must-support present but null | Null-rate report by element |
| The target model preserves clinical meaning | Downstream analysis | Mapping collapses granularity | Clinician review of a sample |

The largest limit is that FHIR completeness is a property of the server, not of the standard. A server can
be conformant and still omit data that exists in the EHR. The pipeline can measure what it received; it
cannot prove that the source sent everything.

A second limit is that mapping loss is biased. The codes that fail to map are not a random sample; they
are often the specialized assays and local practices that matter for a particular study. A 94% mapping
rate can be excellent for one analysis and disqualifying for another.

## 12. Exercises

### Exercise 1 — Choose the access pattern (recall, 20 minutes)

Deliverable: a table matching four workloads (population cohort, single-patient chart, daily registry
update, vendor migration) to an access pattern, with one strength and one weakness each.

Acceptance criteria: the choice cites volume and freshness; no pattern is used for all four; the
weaknesses are specific to the workload.

### Exercise 2 — Raw landing layout (application, 30 minutes)

Deliverable: a path convention and a manifest schema for raw FHIR extracts, including server URL,
retrieval time, request, status, and hash.

Acceptance criteria: the path is partitionable by source and time; the manifest supports replay; no
overwrite of an existing extract is possible.

### Exercise 3 — Dangling references (application, 45 minutes)

Given the reference table in Section 6.3, deliverable: a policy per reference path and the resulting row
counts.

Acceptance criteria: each policy is justified by analytic consequence; the counts reconcile; quarantined
rows are stored, not discarded.

### Exercise 4 — Partial dates and units (application, 40 minutes)

Deliverable: a function that computes age from a FHIR `birthDate` with `YYYY`, `YYYY-MM`, and
`YYYY-MM-DD` precision, and a unit normalizer for mg/dL, µmol/L, and g/dL.

Acceptance criteria: precision is stored; the age rule is documented; unit conversion uses exact decimal
arithmetic; unknown units are quarantined.

### Exercise 5 — Flatten to analytics (design, 90 minutes)

Deliverable: a mapping document from the six resource types in Section 6.1 to an OMOP-shaped model, with
grain, keys, and loss notes.

Acceptance criteria: every resource has a target or an explicit "not loaded" rationale; the observation
category split is addressed; order versus administration is flagged.

### Exercise 6 — Duplicate policy (critique, 40 minutes)

Deliverable: a 300-word policy distinguishing retransmission, re-encoding, and legitimate repeat, with a
test for each.

Acceptance criteria: resource identity and clinical key are both used; the policy states what is lost;
a legitimate repeat example is included.

### Exercise 7 — Reconciliation (application, 60 minutes)

Deliverable: a reconciliation table that accounts for every resource from receipt to load, with counts and
percentages.

Acceptance criteria: the totals reconcile exactly; each loss category has a reason; the report is
machine-readable.

### Exercise 8 — Medication mapping (critique, 30 minutes)

Deliverable: a corrected mapping for `MedicationRequest` and `MedicationAdministration`, with the effect
on an adherence analysis.

Acceptance criteria: distinguishes order from administration; quantifies the difference; states the
export change required.

## 13. Self-check answers

**1. Why is FHIR not an analytics model?**

Because it is designed for exchange: resources are independently versioned, references may be unresolved,
and one resource type covers multiple clinical domains. An analytics model needs stable grain, resolved
keys, and domain-specific tables. The mapping between them is a design decision.

**2. What does `_since` filter on?**

`Resource.meta.lastUpdated`, the server's update time. It is not the clinical effective time. A resource
created today about a 2019 event appears in a `_since` export for today.

**3. Why preserve the raw resource?**

Because the mapping is lossy and may be wrong. The raw resource is the only way to re-derive the analytics
row after a mapping change, and it is the evidence of what the server returned during an audit.

**4. Compute the dangling rate.**

61 dangling of 28,500 references = 61 / 28,500 = 0.002140, or 0.214%. The rate is small, but the
consequences differ by path: a dangling `subject` removes a row; a dangling `encounter` removes
attribution only.

**5. Why is `code.coding[0]` unsafe?**

FHIR does not guarantee coding order or that the first coding is the most specific. A resource may list a
local code first and a LOINC code second, or list only text. Reading all codings and selecting by system
is the correct approach.

**6. How do you handle a missing unit?**

Do not impute it from the code. Load the value with a null unit and a flag, quarantine it if the analysis
requires a unit, and report the missing-unit rate. If the source consistently omits a unit for a specific
code, document that as a source convention and apply it explicitly.

**7. What is the difference between a retransmission and a legitimate repeat?**

A retransmission has the same resource identity and version, or identical clinical content from the same
sender at the same time. A legitimate repeat is a separate clinical event, such as a second blood pressure
or a repeat lab, and usually has a different order or specimen identifier.

**8. Compute the medication administration rate.**

540 administrations of 700 orders = 540 / 700 = 0.7714, or 77.1%. The remaining 160 orders (22.9%) were
not administered. Mapping orders to exposures would overstate exposure by 29.6% relative to the
administration count (700 / 540 = 1.296).

**9. What should the reconciliation report contain?**

Total resources received, by type; resolved and dangling references, by path; duplicates removed, by
method; unmapped codes, by system; quarantined rows, by reason; loaded rows, by target table; and the
percentage at each stage. It should reconcile exactly to the total.

**10. Why is a 94% mapping rate not automatically good?**

Because the 6% may include the codes the study depends on. Mapping loss is biased toward specialized and
local codes. The rate must be evaluated by domain and by the analysis's critical concepts.

## 14. Interview questions

**1. "How would you ingest a FHIR cohort?"**

Strong: defines the cohort and resource list, checks capabilities, chooses `$export` or search, lands raw
with metadata, validates profiles, resolves references, maps codes and units, reconciles, and reports
loss. Weak: "call the API and flatten the JSON."

**2. "What is the difference between R4 and R5?"**

Strong: names the release, notes that R5 is the current published release but not fully normative, and
identifies the changes that affect ingestion, such as subscriptions and new resources. Weak: "R5 is
newer."

**3. "How do you handle a reference to a resource you did not export?"**

Strong: classifies it as dangling or external, applies a path-specific policy, quarantines or nulls, and
reports the count. Weak: "we drop it."

**4. "Why not load all Observations into one table?"**

Strong: category mixing destroys aggregate meaning and unit normalization; labs, vitals, surveys, and
social history have different grains and semantics. Weak: "it is easier."

**5. "How do you deduplicate FHIR resources?"**

Strong: identity and version first, then clinical key, with a documented rule for legitimate repeats.
Weak: `DISTINCT` on the JSON.

**6. "What does a `MedicationRequest` represent?"**

Strong: an order, not an administration; mapping it to exposure asserts something unproven. Weak: "a
prescription, so the patient took it."

**7. "How do you handle partial dates?"**

Strong: store the precision, apply a documented rule for age, and report the approximation. Weak: cast to
1 January.

**8. "What is in your raw zone?"**

Strong: original JSON, server URL, request, status, retrieval time, version, hash, and manifest. Weak:
"the parsed resources."

**9. "How do you know the extract is complete?"**

Strong: reconcile counts with the source, check pagination, compare successive exports, and report
dangling and quarantine counts. Weak: "the API returned 200."

**10. "When would you not use FHIR?"**

Strong: when the source only offers a flat file or a proprietary interface, or when the analytics need
data the FHIR profile does not expose; then use the available interface and document the gap. Weak:
"FHIR is always best."

## 15. Key terms

- **Bundle** — container of FHIR resources.
- **Bulk export** — `$export` asynchronous extraction.
- **CapabilityStatement** — server capability declaration.
- **CodeableConcept** — code plus system, version, display, text.
- **Conditional create** — create only if no match exists.
- **Contained resource** — resource embedded in another.
- **Dangling reference** — reference whose target is absent.
- **FHIR** — Fast Healthcare Interoperability Resources.
- **Implementation guide** — profiles and conformance rules.
- **LOINC** — observation code system.
- **Mapping loss** — resources or codes not represented downstream.
- **NDJSON** — newline-delimited JSON.
- **OperationOutcome** — error or warning resource.
- **Partial date** — reduced-precision date.
- **Profile** — constrained structure definition.
- **Quantity** — value with unit and comparator.
- **Reference** — link between resources.
- **Resource** — typed FHIR object.
- **RxNorm** — medication terminology.
- **Search parameter** — queryable field.
- **SMART on FHIR** — OAuth 2.0 profile for clinical apps.
- **SNOMED CT** — clinical terminology.
- **Subscription** — server push of changes.
- **UCUM** — unit code system.
- **US Core** — US realm FHIR profiles.
- **Version** — `meta.versionId`.
- **`_include`** — include referenced resources.
- **`_lastUpdated`** — filter on update time.
- **`_revinclude`** — include referencing resources.
- **`_since`** — bulk export change filter.

## 16. Sources

All URLs accessed 25 October 2026.

1. HL7. "FHIR R5 (v5.0.0)." https://hl7.org/fhir/R5/
2. HL7. "FHIR Versions." https://hl7.org/fhir/versions.html
3. HL7. "FHIR R5 History." https://hl7.org/fhir/R5/history.html
4. HL7. "US Core Implementation Guide v7.0.0." https://hl7.org/fhir/us/core/STU7/
5. HL7. "US Core Change Log." https://www.hl7.org/fhir/us/core/STU7/changes.html
6. HL7. "FHIR Bulk Data Access IG 2.0.0." https://hl7.org/fhir/uv/bulkdata/STU2/
7. HL7. "Bulk Data `$export` OperationDefinition." http://hl7.org/fhir/uv/bulkdata/STU2/OperationDefinition-export.html
8. HL7. "SMART App Launch." https://hl7.org/fhir/smart-app-launch/
9. LOINC. https://loinc.org/
10. UCUM. https://ucum.org/
11. SNOMED International. https://www.snomed.org/
12. OHDSI. "OMOP CDM v5.4." https://ohdsi.github.io/CommonDataModel/cdm54.html
13. ONC. "Interoperability Standards Platform." https://isp.healthit.gov/
14. ONC. "HTI-1 Final Rule." https://healthit.gov/regulations/hti-rules/hti-1-final-rule/
15. HHS. "HIPAA Security Rule." https://www.hhs.gov/hipaa/for-professionals/security/index.html

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_orchestration.md`](04_orchestration.md)
- Next topic: [`06_dicom-pipelines.md`](06_dicom-pipelines.md)
- Related sibling: [`02_sql-for-clinical-data.md`](02_sql-for-clinical-data.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../course/03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — Lab 3.3 ingests a FHIR bundle and produces a mapping report.
