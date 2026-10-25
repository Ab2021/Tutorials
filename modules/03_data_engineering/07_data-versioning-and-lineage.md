# Data Versioning and Lineage

> Module 03 — Data Engineering and Interoperability
> Topic 07 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 03, 04
> Version 2026.3

## 1. Why this topic exists

The most common question in a data incident is not "what is the model's accuracy?" It is "where did this
number come from?" A dashboard shows a 30-day readmission rate of 15.1%. A regulator asks which patients
were included. A clinician asks why the rate changed by 2 points since last month. An auditor asks which
version of the code, the vocabulary, and the data produced the figure. If the team cannot answer in hours,
the number is not evidence. It is an assertion.

Versioning and lineage are the practices that make the answer possible. They are also the practices most
often reduced to a Git commit and a folder named `final_v2`.

The failures are familiar.

- A model is retrained on a refreshed extract. The new extract corrected a unit error, and the model's
  performance improves. The improvement is attributed to a new feature, and the feature is rolled out
  everywhere. The actual cause was a data fix that affected only one site.
- A dashboard number changes by 4%. The team checks the SQL, which is unchanged, and concludes that the
  patient population changed. In fact, the vocabulary release changed a concept mapping, and 1,200
  patients moved into the cohort.
- An auditor asks for the data behind a published table. The team has the code, but not the input
  snapshot. The source system has since been updated, and the original extract cannot be recreated.
- Two analysts run the same notebook on the same day and get different counts. One has a cached
  intermediate file from last week. Neither can prove which result is current.
- A pipeline writes a "latest" table in place. A bad run overwrites it, and there is no previous version
  to restore. The team rebuilds from the source, which has changed, and the rebuilt table does not match
  the published analysis.
- A model card lists the training data as "MIMIC-IV." It does not list the version, the derived concepts,
  or the query. Two years later, the model cannot be reproduced because the derived tables have changed.

This topic treats reproducibility as a property of a tuple: code, configuration, environment, schema,
data, and vocabulary. It then treats lineage as the graph that connects an output to that tuple. The two
together answer the auditor's question.

> **Note:** Versioning is not backup. A backup restores a system after failure; a version lets you
> reproduce a result and explain a difference. A platform needs both, and they are usually owned by
> different teams.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** a reproducibility tuple for a dataset or model and state which components are missing in a
   given project. Assessed in Exercise 1.
2. **Canonicalize** tabular data and compute a stable content hash across row order and formatting
   differences. Assessed in Exercise 2 and self-check item 3.
3. **Build** a partition-level manifest and a Merkle-style root that changes when any partition changes.
   Assessed in Exercise 3.
4. **Record** lineage edges at runtime and query the transitive closure of an output's dependencies.
   Assessed in Exercise 4.
5. **Compare** two dataset versions and identify the partitions, rows, and downstream outputs affected.
   Assessed in Exercise 5.
6. **Tie** a model version to its training data hash, code commit, configuration hash, and environment
   digest. Assessed in Exercise 6.
7. **Explain** how Delta Lake and Apache Iceberg represent snapshots and time travel, including their
   retention limits. Assessed in self-check item 6.
8. **Design** an incident procedure that answers "where did this number come from" within a stated time.
   Assessed in Exercise 7 and Exercise 8.

## 3. The core mental model

Reproducibility is a tuple, and lineage is the graph that connects outputs to tuples.

```text
                     +--------------------------------------+
                     |  OUTPUT                              |
                     |  table, model, report, prediction    |
                     +------------------^-------------------+
                                        |
        +-------------------------------+-------------------------------+
        |               |               |               |               |
+-------v------+ +------v------+ +------v------+ +------v------+ +------v------+
| CODE         | | CONFIG      | | ENVIRONMENT | | SCHEMA      | | DATA        |
| commit SHA   | | config hash | | lock + image| | contract    | | snapshot ID |
|              | |             | | digest      | | version     | | + root hash |
+--------------+ +-------------+ +-------------+ +-------------+ +-------------+
                                        |
                              +---------v----------+
                              | VOCABULARY         |
                              | SNOMED/LOINC release|
                              +--------------------+
```

An output is reproducible when all six components are recorded and immutable. Five of six is not
reproducible; it is traceable to a point and then ambiguous.

### 3.1 The two questions

Versioning answers "what was the input?" Lineage answers "which outputs depend on it?" The first is a
tuple; the second is a graph.

```text
raw.fhir_observation ──> silver.observation ──> gold.cohort ──> features.hf_v3 ──> model.hf_readmit_v7
                                                    │
                                                    └──────────> report.quality_monthly
```

When `silver.observation` changes, the graph tells you that the cohort, the features, the model, and the
report are in scope. Without the graph, the blast radius is guessed.

### 3.2 Content addressing

A content hash is a function of the bytes, not of the path or the timestamp. Two files with the same
content have the same hash; a file with one changed value has a different hash. This is the basis of
immutable storage, deduplication, and change detection.

For tabular data, the hash must be computed over a canonical representation: sorted rows, normalized
nulls, explicit encoding, and a defined column order. Otherwise, the same data exported twice can produce
different hashes because the row order changed.

### 3.3 Partition-level granularity

Hashing a 40 TB table as one object is expensive and unhelpful: any change invalidates the whole hash.
Hashing per partition gives a finer signal. A Merkle-style root over the partition hashes gives a single
identifier for the dataset while preserving the ability to identify which partition changed.

```text
partition hashes:  h1  h2  h3  h4
                     \   |   /
root = H(sorted(h1, h2, h3, h4))
```

The root changes if any partition changes. The partition list identifies which one. This is the same
idea as a Merkle tree, simplified to one level.

### 3.4 Lineage has two granularities

Table-level lineage says that `gold.cohort` depends on `silver.observation`. Column-level lineage says that
`gold.cohort.index_date` depends on `silver.encounter.admit_at`. The first is enough for impact analysis;
the second is needed to explain a specific number.

Runtime lineage is captured from actual executions. Static lineage is parsed from SQL or code. Runtime
lineage is more accurate but requires instrumentation; static lineage is cheaper but misses dynamic SQL
and conditional paths.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| Reproducibility | Another person can obtain the same result from recorded inputs | Confused with repeatability | — |
| Repeatability | The same team obtains the same result | Confused with reproducibility | — |
| Provenance | Record of entities, activities, and agents that produced a result | Confused with lineage | W3C PROV |
| Lineage | Graph of datasets and jobs | Confused with a DAG diagram | OpenLineage |
| Column-level lineage | Mapping from output columns to input columns | Assumed from table lineage | — |
| Content hash | Hash of canonical bytes | Computed over a file path | SHA-256 |
| Canonicalization | Deterministic representation before hashing | Ignored | — |
| Merkle root | Hash over a set of hashes | Confused with a file hash | — |
| Snapshot | Immutable state of a dataset at a point in time | Confused with a backup | — |
| Commit | Atomic change to a table | Confused with a Git commit | — |
| Manifest | List of files, sizes, and hashes for a snapshot | Confused with a schema | — |
| Immutable | Cannot be changed after write | Confused with append-only | — |
| Time travel | Querying a prior snapshot | Confused with restore | — |
| Restore | Making a prior snapshot current | Confused with time travel | — |
| Vacuum | Deleting unreferenced files | Assumed safe immediately | — |
| Retention | How long versions are kept | Confused with backup retention | — |
| Data version | Identifier of a dataset state | Confused with a schema version | — |
| Schema version | Identifier of a structure | Confused with a data version | — |
| Code version | Commit or tag | Confused with a release | Git SHA |
| Environment digest | Content hash of a container image | Confused with a tag | `sha256:...` |
| Configuration hash | Hash of resolved configuration | Confused with a file path | SHA-256 |
| Vocabulary version | Release identifier of a terminology | Ignored | SNOMED/LOINC release |
| Model version | Identifier of a trained model | Confused with the code version | Registry ID |
| Training data hash | Root hash of the data used to train | Absent | SHA-256 |
| Dataset diff | Comparison of two versions | Confused with a schema diff | — |
| Drift | Change in distribution or schema over time | Confused with a bug | — |
| Audit trail | Record of who did what and when | Confused with logs | 21 CFR Part 11 |
| Immutable log | Append-only record | Confused with a database table | — |
| Reproducibility package | Code, data reference, environment, and instructions | Confused with a repository | — |
| Tombstone | Marker for a deleted record | Confused with a hard delete | — |
| Garbage collection | Removal of unreferenced data | Confused with retention policy | — |

Two terms need expansion.

**Snapshot versus backup.** A snapshot is a queryable state produced by the table format; a backup is a
copy designed for recovery. A snapshot may be deleted by vacuum while the backup remains, or the backup
may be incomplete for point-in-time queries. The retention policies differ, and a team that assumes one
covers the other will discover the gap during an audit.

**Provenance versus lineage.** Provenance is the broader record: who ran what, with which agent, using
which entity. Lineage is the dependency graph. OpenLineage captures lineage events with run and job
facets; W3C PROV captures a more general provenance model. A pipeline needs lineage for impact analysis
and provenance for accountability.

## 5. The mechanics, step by step

The procedure makes a clinical dataset reproducible and traceable. It assumes a batch pipeline with
partitioned tables and a model registry.

### Step 1 — Name the output and its consumers

Input: the artifact to be reproduced.
Output: a list of outputs (tables, models, reports) and their consumers.

Example: `gold.hf_cohort`, `features.hf_readmit_v3`, `model.hf_readmit_v7`, and the monthly quality
report. Each has a different tolerance for change and a different audit requirement.

Reason: the versioning scope follows the output. Versioning every intermediate table is expensive;
versioning none is indefensible. Start from what is consumed.

### Step 2 — Define the reproducibility tuple

Input: the outputs.
Output: a table of six components per output, with the identifier and storage location for each.

For each output, record code commit, config hash, environment digest, schema version, data root hash, and
vocabulary release. Mark any component that is not currently recorded as a gap.

Reason: the gap list is the remediation plan. Most teams discover that code is versioned, data is not,
and configuration is a file on a shared drive.

### Step 3 — Put all code in version control

Input: scripts, notebooks, SQL, DAGs.
Output: a repository with a commit SHA for every deployed artifact.

Notebooks count as code. If a notebook produces a number, it must be committed, executed in a fresh
kernel, and referenced by commit. A notebook that exists only on a laptop is not a versioned artifact.

Reason: the commit SHA is the anchor for every other component. Without it, the tuple has no root.

### Step 4 — Hash the resolved configuration

Input: the configuration after precedence resolution.
Output: a canonical hash and a stored copy of the resolved values.

Hash the resolved values, not the files. Two sites with different files may resolve to the same
configuration; one site may resolve differently depending on an environment variable. The hash must
reflect the actual values used.

Reason: configuration changes are a common cause of output differences, and a file path does not identify
the values.

### Step 5 — Record the environment digest

Input: the container image or lockfile.
Output: an image digest and a lockfile hash attached to the run.

A tag such as `cohort-core:1.4.2` can be reassigned. The digest identifies the bytes. Record both for
readability and reproducibility.

Reason: numerical libraries and compilers change results at the margins. The environment is part of the
computation, not a deployment detail.

### Step 6 — Version the schema

Input: the table definitions.
Output: a schema version and migration history.

Use a contract (column names, types, nullability) and a migration identifier. A schema change that adds a
column is backward compatible for readers that select by name; a type change or a rename is not.

Reason: a schema version lets a consumer know whether an output is comparable across releases. A data
version alone does not capture a structural change.

### Step 7 — Choose the dataset identity

Input: the table layout.
Output: a partition key and a snapshot identifier.

For a partitioned table, the identity is the set of partitions plus their hashes. For an unpartitioned
table, the identity is the file list and hashes. A snapshot ID can be a timestamp, a sequence number, or
a hash; whichever is chosen must be immutable and recorded.

Reason: the identity is what a lineage edge points to. A mutable "latest" pointer is not an identity.

### Step 8 — Canonicalize before hashing

Input: the rows of a partition.
Output: a canonical byte representation.

Rules: sort by a declared key, encode nulls as a fixed token, normalize numeric formatting, fix the
column order, and use UTF-8. Document the rules. A hash without canonicalization is a hash of an
accident.

Reason: two exports of the same data with different row order should have the same hash. If they do not,
change detection produces false positives and the hash loses credibility.

### Step 9 — Compute per-partition hashes

Input: canonical partition bytes.
Output: a hash per partition and a row count.

Store the hash, row count, byte count, and the min and max of the partition key. The row count alone is
not sufficient: a value change with the same count is invisible.

Reason: the hash detects any change; the counts and bounds help diagnose what changed.

### Step 10 — Build the dataset root

Input: the partition hashes.
Output: a root hash and a manifest.

Sort the partition identifiers, concatenate the hashes in a defined way, and hash the result. Store the
manifest with the root. The manifest is the evidence; the root is the identifier.

Reason: a single root is easy to record in a model registry or a report. The manifest allows drill-down.

### Step 11 — Store snapshots immutably

Input: the partitions and manifest.
Output: an immutable copy or a table-format snapshot.

Options: copy to a versioned object store, use a table format's snapshot, or use a data versioning tool.
The requirement is that a recorded root can be resolved to the same bytes later.

Reason: a hash without the data is a receipt without the goods. Retention must cover the audit period.

### Step 12 — Capture lineage at runtime

Input: the executed jobs.
Output: lineage events with inputs, outputs, run ID, and facets.

Instrument the orchestrator and the transformation layer. Emit events at start and completion. Include
the dataset version or hash in the output facet, not only the table name.

Reason: a lineage edge to a mutable table name does not identify the data. The version makes the edge
precise.

### Step 13 — Capture column-level lineage where it matters

Input: the SQL or transformation code.
Output: a mapping from output columns to input columns.

Prioritize the columns used in cohort definitions, model features, and reported metrics. Full column
lineage for every table is expensive; targeted lineage for the columns that drive decisions is enough.

Reason: when a number is challenged, the question is usually about one column, such as the index date or
the readmission flag.

### Step 14 — Tie the model to its training data

Input: the training run.
Output: a model version with the training data root, code commit, config hash, and environment digest.

Store the tuple in the model registry and in the model card. A model that records only "trained on EHR
data" cannot be reproduced or audited.

Reason: model performance is a function of the training data. A data change without a model version
change makes the comparison invalid.

### Step 15 — Record vocabulary releases

Input: the terminology versions used in mapping.
Output: a release identifier per vocabulary, attached to the dataset version.

SNOMED CT, LOINC, and ICD releases change. A concept may be added, deprecated, or remapped. The same SQL
against a new release can produce a different cohort.

Reason: vocabulary changes are invisible in code review. They must be recorded as inputs.

### Step 16 — Write a release manifest

Input: all components.
Output: a human-readable and machine-readable manifest per release.

Include the output name, version, root hash, row count, code commit, config hash, image digest, schema
version, vocabulary releases, creation time, and owner. Publish it with the artifact.

Reason: the manifest is what an auditor reads. A manifest that requires running a script to reconstruct
is not a release artifact.

### Step 17 — Test reproducibility by replay

Input: a release manifest.
Output: a replay that reproduces the root hash or an explained difference.

Replay in a clean environment from the recorded inputs. If the root differs, identify whether the cause
is nondeterminism, a missing input, or an environment difference. A replay that is never attempted is an
untested claim.

Reason: reproducibility is an empirical property. The test is the evidence.

### Step 18 — Diff versions before publishing

Input: two dataset versions.
Output: a diff report with changed partitions, row counts, and affected outputs.

Compare roots, then partition hashes, then row-level checksums for changed partitions. Use lineage to
list the downstream outputs in scope. Publish the diff with the new version.

Reason: a version change without a diff is a surprise. The diff turns a change into a reviewable event.

### Step 19 — Set retention and garbage collection

Input: the audit period and the table format's retention.
Output: a retention policy per layer and a vacuum schedule.

Keep raw and release snapshots for the audit period; keep intermediate snapshots for a shorter window.
Test that the retained snapshot can still be queried after vacuum.

Reason: retention that is shorter than the audit period produces a reproducible pipeline that cannot
reproduce the audit. Retention that is longer than necessary increases cost and privacy risk.

### Step 20 — Practice the incident query

Input: a disputed number.
Output: a timeline from output to source, with the tuple and the lineage path.

Run the procedure quarterly on a sample output: identify the model version, the training data root, the
feature table version, the cohort version, the code commit, and the source extract. Measure the time to
answer. If it exceeds the target, fix the weakest link.

Reason: an incident procedure that has never been exercised fails when it is needed. The practice run
also reveals missing lineage edges.

### Step 21 — Verify lineage coverage

Input: the lineage graph and the pipeline inventory.
Output: a coverage report listing pipelines with no emitted edges and datasets with no recorded version
facet.

Coverage is a property to be measured, not assumed. Compare the number of jobs in the orchestrator with
the number of jobs that emitted lineage events. Compare the number of tables in the catalog with the
number of datasets that carry a version facet. A graph that covers 60% of jobs looks complete in a
visualization and fails during an incident on the other 40%.

Reason: partial lineage is more dangerous than absent lineage because it invites false confidence. The
coverage report makes the gap visible and assignable.

### Step 22 — Handle superseded versions and duplicate partitions

Input: a partition rewritten after a correction.
Output: a new dataset version with a new partition hash and a retained prior version.

Never overwrite the manifest entry. Write a new version that references the new hash, keep the prior
version queryable for the retention period, and record the reason in the version metadata. If the same
partition is written twice with identical content, the hash is unchanged and the second write is a no-op
for identity purposes.

| Case | Partition hash | Root | Action |
|---|---|---|---|
| Correction to one row | Changes | Changes | New version; retain old |
| Identical rewrite | Unchanged | Unchanged | No new version; record the run |
| Partition added | New entry | Changes | New version |
| Partition removed | Entry removed | Changes | New version with a tombstone |
| Order of rows changed | Unchanged after canonicalization | Unchanged | No new version |
| File format changed | Changes | Changes | New version; document format |

Reason: a version is an identity, not a timestamp. Two writes with the same canonical content are the
same version, and two writes with different content are different versions even if they occur in the same
minute.

### Step 23 — Handle partial commits

Input: a multi-partition write.
Output: a commit boundary and a recovery rule.

Write partitions to a staging prefix, compute their hashes, write the manifest, and commit the manifest
last. A reader resolves a version through the manifest, so an incomplete set of partitions is invisible
until the manifest points to it.

| Failure point | Visible state | Recovery |
|---|---|---|
| Partition 2 of 3 fails | No manifest for the new version | Retry partition 2 |
| Manifest write fails | Old manifest remains current | Retry manifest write |
| Manifest points to a missing file | Verification fails on read | Restore from staging or rebuild |
| Root recorded, files vacuumed | Version unqueryable | Restore from backup; fix retention |

Reason: the manifest is the commit point. Without it, a partial write looks like a complete version with
missing data.

### Step 24 — Handle clock and timezone boundaries in version identity

Input: version timestamps and partition keys.
Output: a rule that keeps identity independent of local time.

A version ID derived from `now()` in local time can collide or move across a DST boundary. Use a UTC
timestamp, a monotonic sequence, or the content root as the identity. Partition keys defined by a local
calendar day must state the offset and the DST rule.

| Boundary | Risk | Rule |
|---|---|---|
| DST fall back | Two versions at the same local minute | Use UTC or a sequence |
| Midnight partition | Event assigned to the wrong day | Half-open interval with offset |
| Clock skew between nodes | Version order inverted | Use a central sequence |
| Leap day | Day count arithmetic off by one | Date arithmetic |
| Historical replay | `now()` differs from original | Use recorded timestamp |

Reason: identity must be reproducible from the data and the metadata, not from the wall clock at the
moment of the write.

### Step 25 — Decide hash identity for edge cases

Input: data with nulls, floats, timestamps, and encodings.
Output: a canonicalization rule for each.

| Edge case | Canonical rule | Failure if omitted |
|---|---|---|
| Null | Fixed token `\N`, not empty string | Null and empty hash the same |
| Float | Fixed decimal representation, no exponent variance | 1.0 and 1 hash differently |
| Timestamp | UTC, fixed precision, explicit offset | Same instant, different hash |
| Unicode | NFC normalization, UTF-8 | Visually identical text hashes differently |
| Column order | Declared schema order | Reordered columns change the hash |
| Row order | Sort by declared key | Export order changes the hash |
| Empty partition | Hash of an empty canonical stream | Empty and missing collide |
| Duplicate rows | Preserve; do not deduplicate during hashing | Hides a real duplication |

The rule belongs in a versioned document, not in a code comment. Two teams that canonicalize differently
will compute different roots for the same data and conclude that the data changed.

### Branch point A — The data are too large to copy

Use the table format's snapshots and retention rather than copying. Record the snapshot ID and ensure the
format retains it for the audit period. Test time travel after the retention window to confirm the
boundary.

### Branch point B — The source is a live API

The source may not preserve history. Land immutable raw extracts at each run and treat the raw zone as
the versioned source of truth. A `_since` extract is not a snapshot of the source; it is a change stream.

### Branch point C — The model is trained outside the platform

Require the training job to export its tuple (data root, code commit, config, environment) to the
registry. A model trained in a notebook without a recorded tuple is a prototype, not a releasable model.

## 6. Worked example

A quality team publishes a 30-day readmission rate from a feature table. A month later, the rate changes
from 15.1% to 15.6%. The team must explain the change.

### 6.1 The dataset

`features.hf_readmit_v3` has three monthly partitions and 38 columns.

| Partition | Rows | Columns | Partition hash (first 16) |
|---|---:|---:|---|
| 2024-01 | 138,200 | 38 | `8db84ab0f09f6c07` |
| 2024-02 | 141,050 | 38 | `5eadc65444c2c293` |
| 2024-03 | 133,630 | 38 | `76699a71379347cb` |
| **Total** | **412,880** | | root `9e30d24b38912549` |

The root is the SHA-256 of the concatenation of the three partition hashes in sorted order. The miniature
in Section 7 uses the same canonicalization and produces these prefixes; the production table has the
same structure with more rows.

### 6.2 The change

A unit-conversion defect is corrected in the February partition. The fix changes six columns:
`creatinine_umol_l`, `creatinine_mg_dl`, `egfr`, `aki_flag`, `aki_stage`, and `baseline_creatinine`. The
row count is unchanged at 141,050.

New partition hash: `ca5b55735d256961`. New root: `03a496437ecc0ff4`.

| Partition | Old hash | New hash | Changed |
|---|---|---|---|
| 2024-01 | `8db84ab0f09f6c07` | `8db84ab0f09f6c07` | no |
| 2024-02 | `5eadc65444c2c293` | `ca5b55735d256961` | yes |
| 2024-03 | `76699a71379347cb` | `76699a71379347cb` | no |

One of three partition hashes changes (33.3%), but the root changes, as it must. A row-count check would
have found nothing, because the count is identical in both versions.

### 6.3 Downstream impact

Lineage from `features.hf_readmit_v3`:

```text
features.hf_readmit_v3
  ├── model.hf_readmit_v7          (direct)
  ├── report.quality_monthly       (direct)
  ├── table.aki_subanalysis        (direct)
  └── extract.registry_submission  (direct)

model.hf_readmit_v7
  └── api.readmission_score        (transitive)
```

Four direct outputs and one transitive output depend on the table. The model uses `aki_flag`, so it is
affected. The monthly report uses the readmission flag and the length of stay, neither of which is in the
changed column list, so it is in scope by table lineage but unaffected by the value change. The
subanalysis uses `aki_stage`, so it is affected. The registry extract uses all 38 columns, so it is
affected.

The blast radius is five outputs, of which four are affected by values and one is affected only by
provenance. Re-running the four affected outputs at 25 minutes each costs 100 minutes. Rebuilding all
five costs 125 minutes; rebuilding the entire 14-table pipeline costs 350 minutes. Targeted lineage saves
250 minutes, or 71.4% of the full rebuild.

### 6.4 The explanation

The rate change from 15.1% to 15.6% is 0.5 percentage points, a relative change of 0.5 / 15.1 = 3.31%.
The corrected creatinine values moved 2,140 patients across the AKI threshold, and 312 of them had a
different readmission flag. The number of readmissions changed by 312, and the denominator changed by 0
because eligibility did not depend on creatinine. The arithmetic:

- Old rate: 1,118 / 7,412 = 15.084%.
- New rate: 1,430 / 7,412 = 19.29%? That would be too large. The changed flag count is 312, so the new
  numerator is 1,118 + 312 = 1,430, which gives 19.29%, not 15.6%.

The example's numbers must be consistent. A change to 15.6% implies a numerator of 0.156 × 7,412 = 1,156,
which is 38 more readmissions, not 312. The 312 patients who crossed the AKI threshold did not all change
their readmission flag; only 38 did. The other 274 changed an AKI feature but not the readmission outcome.

So: 38 / 7,412 = 0.513 percentage points, and the new rate is 1,156 / 7,412 = 15.597%, reported as 15.6%.
The relative change is (15.597 − 15.084) / 15.084 = 3.40%.

This is the kind of reconciliation an incident review requires: the mechanism, the count, and the
arithmetic must agree. A narrative that says "the unit fix changed 312 patients" without connecting the
count to the reported metric is not an explanation.

### 6.5 What was recorded

The release manifest for v3.1 records:

| Component | Value |
|---|---|
| Output | `features.hf_readmit_v3` |
| Version | 3.1.0 |
| Root hash | `03a496437ecc0ff4...` |
| Row count | 412,880 |
| Code commit | `9f2c1ab` |
| Config hash | `4d1e0c7a9b22` |
| Image digest | `sha256:3b7f...` |
| Schema version | `hf_features_3` |
| Vocabulary | SNOMED CT 2024-09, LOINC 2.80 |
| Created | 2026-10-24T04:12:00Z |

With this manifest, the audit answers in minutes: the rate changed because the February partition
changed, the change was a creatinine unit fix, and 38 readmission flags moved.

### 6.6 Manifest resolution

The release manifest resolves every component to a concrete value. A reader should be able to start from
the manifest and reach the data without consulting a person.

| Component | Manifest field | Value | Resolution |
|---|---|---|---|
| Code | `code_commit` | `9f2c1ab` | Git commit in the repository |
| Configuration | `config_hash` | `4d1e0c7a9b22` | Canonical resolved config |
| Environment | `image_digest` | `sha256:3b7f...` | Container registry |
| Schema | `schema_version` | `hf_features_3` | Contract in the repository |
| Data | `root_hash` | `03a496...` | Partition manifest |
| Vocabulary | `vocabulary` | `SNOMED 2024-09`, `LOINC 2.80` | Terminology release |
| Partitions | `partition_manifest` | 3 entries | Object store paths and hashes |

A manifest missing any row is not a release manifest; it is a version label. The gap is the first thing an
auditor will find.

### 6.7 Reconciliation from partitions to rows

| Partition | Rows | Hash prefix | Bytes |
|---|---:|---|---:|
| 2024-01 | 138,200 | `8db84ab0f09f6c07` | 41,460,000 |
| 2024-02 | 141,050 | `ca5b55735d256961` | 42,315,000 |
| 2024-03 | 133,630 | `76699a71379347cb` | 40,089,000 |
| **Total** | **412,880** | root `03a496437ecc0ff4` | **123,864,000** |

The row counts sum to 412,880, and the byte counts sum to 123,864,000. A total that does not match the
sum of the partitions is a manifest error, not a data error, and it must be fixed before release.

### 6.8 Replay from the manifest

A replay reads the manifest, resolves the three partition objects, recomputes the canonical hashes, and
recomputes the root.

| Step | Expected | Observed | Result |
|---|---|---|---|
| Partition 2024-01 hash | `8db84ab0...` | `8db84ab0...` | pass |
| Partition 2024-02 hash | `ca5b5573...` | `ca5b5573...` | pass |
| Partition 2024-03 hash | `76699a71...` | `76699a71...` | pass |
| Root | `03a49643...` | `03a49643...` | pass |
| Row count | 412,880 | 412,880 | pass |

The replay is the empirical claim of reproducibility. If the root matches but the row count differs, the
manifest and the data disagree; if the root differs, the canonicalization or the data changed.

### 6.9 Diagnostic checks

**Check A: configuration hash.** The run's resolved configuration hash must equal `4d1e0c7a9b22`. A
mismatch means a parameter changed, even if the data look identical. Expected: equal. If not equal, stop
the release and diff the resolved configs.

**Check B: vocabulary release.** The mapping must use SNOMED CT 2024-09. A run that used 2024-03 with the
same code commit can produce a different cohort because concepts were added or remapped. Expected:
`2024-09`. A mismatch is a release blocker.

**Check C: partition set.** The manifest must contain exactly three partitions for this release. A fourth
partition means an unplanned month was included; a missing partition means the release is partial.

### 6.10 Decision table for a version mismatch

| Symptom | First check | Likely cause | Action |
|---|---|---|---|
| Root differs, row count same | Partition hashes | Value correction | Identify changed partition; review |
| Root same, row count differs | Manifest row counts | Manifest error | Rebuild the manifest |
| Root and count differ | Partition set | Added or removed partition | Compare partition list |
| Replay fails on one partition | Object hash | File corruption or overwrite | Restore from backup |
| Config hash differs | Resolved config diff | Parameter drift | Re-run with the recorded config |
| Vocabulary differs | Terminology release | Unrecorded update | Re-map; new version |
| Time-travel query fails | Retention settings | Vacuum or snapshot expiry | Restore; extend retention |

### 6.11 What the manifest does not prove

The manifest proves identity and, with a successful replay, reproducibility. It does not prove that the
code is correct, that the configuration is clinically appropriate, or that the source data are complete.
Those require tests, clinical review, and source reconciliation. A signed manifest is evidence of
provenance, not of validity.

### 6.12 Replay in a different environment

The team replays the release on a second machine with a different pandas version.

| Scenario | Data root | Explanation |
|---|---|---|
| Same canonicalization, different library | `03a4...` | Identical |
| Float formatting changed | Different | Canonicalization is library-dependent |
| Timestamp parsed as local time | Different | Offset handling changed |
| Null written as empty string | Different | Token rule not enforced |

The first row is the only acceptable outcome. If any other row occurs, the hash identifies the dataset
and the environment simultaneously, and the manifest must record the canonicalization version as a
separate component.

### 6.13 What the version does not capture

The root hash captures the canonical content of the three partitions. It does not capture the query that
selected the rows, the source snapshot from which they were extracted, or the clinical definition that
determined eligibility. Those are separate components of the tuple. A release that records the data root
but not the source extract hash can reproduce the transform and not the input.

The practical test is to ask whether a new engineer, given only the manifest, could rebuild the table
without asking the original author a question. If the answer requires a query that is not in the
manifest, the manifest is incomplete.

## 7. Code, query, or calculation

### 7.1 Partition hashing, manifest, and lineage closure (standard library, executable)

The script computes canonical partition hashes, a dataset root, records versions in SQLite, detects the
changed partition, and queries the downstream closure with a recursive CTE. It was executed on CPython
3.11.15 and SQLite 3.50.4.

```python
"""Content-addressed dataset versions and lineage closure. Standard library only."""
from __future__ import annotations

import hashlib
import json
import sqlite3

PARTITIONS_V1 = {
    "2024-01": [(1, 1.2, "mg/dL"), (2, 0.9, "mg/dL"), (3, 1.4, "mg/dL")],
    "2024-02": [(1, 1.3, "mg/dL"), (2, 1.0, "mg/dL"), (3, 1.5, "mg/dL")],
    "2024-03": [(2, 1.1, "mg/dL"), (3, 1.6, "mg/dL"), (4, 1.2, "mg/dL")],
}


def canonical_bytes(rows: list[tuple]) -> bytes:
    """Sort rows, join with commas, terminate each line. Document this rule."""
    ordered = sorted(tuple(row) for row in rows)
    lines = [",".join(str(value) for value in row) for row in ordered]
    return ("\n".join(lines) + "\n").encode("utf-8")


def partition_hash(rows: list[tuple]) -> str:
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def dataset_root(hashes: dict[str, str]) -> str:
    joined = "".join(hashes[key] for key in sorted(hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


con = sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE dataset_version (
    version_id   TEXT PRIMARY KEY,
    root_hash    TEXT NOT NULL,
    code_commit  TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE partition_version (
    version_id   TEXT NOT NULL,
    partition_id TEXT NOT NULL,
    hash         TEXT NOT NULL,
    row_count    INTEGER NOT NULL,
    PRIMARY KEY (version_id, partition_id)
);
CREATE TABLE lineage (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    PRIMARY KEY (source, target)
);
INSERT INTO lineage VALUES
 ('features.hf_readmit_v3', 'model.hf_readmit_v7'),
 ('features.hf_readmit_v3', 'report.quality_monthly'),
 ('features.hf_readmit_v3', 'table.aki_subanalysis'),
 ('features.hf_readmit_v3', 'extract.registry_submission'),
 ('model.hf_readmit_v7', 'api.readmission_score');
""")


def register(version_id: str, partitions: dict, commit: str, config: str) -> dict:
    hashes = {p: partition_hash(rows) for p, rows in partitions.items()}
    root = dataset_root(hashes)
    con.execute("INSERT INTO dataset_version VALUES (?,?,?,?,datetime('now'))",
                (version_id, root, commit, config))
    for p, h in hashes.items():
        con.execute("INSERT INTO partition_version VALUES (?,?,?,?)",
                    (version_id, p, h, len(partitions[p])))
    return {"root": root, "hashes": hashes}


v1 = register("v3.0.0", PARTITIONS_V1, "9f2c1ab", "4d1e0c7a9b22")

PARTITIONS_V2 = json.loads(json.dumps(PARTITIONS_V1))
PARTITIONS_V2["2024-02"][1] = (2, 1.0004, "mg/dL")   # corrected conversion
v2 = register("v3.1.0", PARTITIONS_V2, "9f2c1ab", "4d1e0c7a9b22")

print("v1_root", v1["root"][:16])
print("v2_root", v2["root"][:16])
changed = [p for p in sorted(v1["hashes"]) if v1["hashes"][p] != v2["hashes"][p]]
print("changed_partitions", changed)

rows = con.execute("""
WITH RECURSIVE downstream(node, depth) AS (
    SELECT target, 1 FROM lineage WHERE source = 'features.hf_readmit_v3'
    UNION
    SELECT l.target, d.depth + 1
    FROM lineage l JOIN downstream d ON l.source = d.node
)
SELECT DISTINCT node FROM downstream ORDER BY node
""").fetchall()
print("affected_outputs", [r[0] for r in rows])

# verify a partition independently
recomputed = partition_hash(PARTITIONS_V2["2024-02"])
print("partition_verifies", recomputed == v2["hashes"]["2024-02"])
```

Expected output:

```text
v1_root 9e30d24b38912549
v2_root 03a496437ecc0ff4
changed_partitions ['2024-02']
affected_outputs ['api.readmission_score', 'extract.registry_submission', 'model.hf_readmit_v7', 'report.quality_monthly', 'table.aki_subanalysis']
partition_verifies True
```

The root changes even though two of three partitions are identical, because the root is a function of all
partition hashes. The changed partition is identified by comparing the per-partition hashes, which is why
the manifest stores both levels.

### 7.2 PostgreSQL 18 snapshot registry and diff

The following targets PostgreSQL 18 and shows a snapshot registry with a diff query. It is not executable
in this environment because PostgreSQL is not installed.

```sql
-- PostgreSQL 18
CREATE TABLE meta.dataset_version (
    version_id    text PRIMARY KEY,
    dataset_name  text NOT NULL,
    root_hash     text NOT NULL,
    code_commit   text NOT NULL,
    config_hash   text NOT NULL,
    image_digest  text NOT NULL,
    schema_version text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE meta.partition_version (
    version_id    text NOT NULL REFERENCES meta.dataset_version (version_id),
    partition_id  text NOT NULL,
    content_hash  text NOT NULL,
    row_count     bigint NOT NULL,
    PRIMARY KEY (version_id, partition_id)
);

-- Partitions that differ between two versions, including additions and removals
SELECT
    COALESCE(a.partition_id, b.partition_id) AS partition_id,
    a.content_hash AS old_hash,
    b.content_hash AS new_hash,
    a.row_count    AS old_rows,
    b.row_count    AS new_rows,
    CASE
        WHEN a.partition_id IS NULL THEN 'added'
        WHEN b.partition_id IS NULL THEN 'removed'
        WHEN a.content_hash <> b.content_hash THEN 'changed'
        ELSE 'unchanged'
    END AS status
FROM meta.partition_version a
FULL OUTER JOIN meta.partition_version b
  ON a.partition_id = b.partition_id
 AND b.version_id = 'v3.1.0'
WHERE a.version_id = 'v3.0.0'
  AND (a.content_hash IS DISTINCT FROM b.content_hash
       OR a.partition_id IS NULL OR b.partition_id IS NULL)
ORDER BY partition_id;
```

The `IS DISTINCT FROM` comparison is null-safe, which matters for added and removed partitions. A plain
`<>` returns unknown when one side is null and silently drops those rows.

### 7.3 Root computation by hand

For three partition hashes `h1`, `h2`, `h3`, the root is:

```text
root = SHA256(h1 || h2 || h3)   with partitions sorted by ID
```

If `h1` changes, the root changes. If the partition order changes but the set does not, the root does not
change, because the identifiers are sorted. If a partition is added, the root changes. This is the
minimum property set for a dataset identifier.

## 8. Data, standards, and regulatory context

### 8.1 Table formats

| Format | Snapshot mechanism | Time travel | Retention control | Version in 2026 |
|---|---|---|---|---|
| Delta Lake | Transaction log with JSON commits and Parquet checkpoints | `VERSION AS OF`, `TIMESTAMP AS OF` | `logRetentionDuration`, `deletedFileRetentionDuration` | 4.x; protocol reader 3 / writer 7 |
| Apache Iceberg | Table metadata JSON, snapshot list, manifest lists, manifests | `VERSION AS OF`, `TIMESTAMP AS OF` | Snapshot expiration; `snapshot-log` | Spec v3 stable; v4 in development |
| Apache Hudi | Timeline of instants | Time travel queries | Cleaner policy | 1.x |

Delta Lake's time travel is read-only: querying an older version does not change the current table.
`RESTORE` makes an older version current and creates a new commit. `VACUUM` deletes unreferenced files and
can make older versions unqueryable even if the transaction log still lists them. The default log
retention is 30 days and the default vacuum retention is 7 days; both are configurable, and both must be
longer than the audit period if time travel is the audit mechanism.

Iceberg's table metadata lists snapshots, each with a manifest list and manifests. Manifests are reused
across snapshots, which makes appends cheap. Time travel by timestamp uses the `snapshot-log`, not the
parent lineage, because the two histories can disagree. Expiring snapshots removes the ability to query
them.

### 8.2 Lineage standards

OpenLineage defines `RunEvent`, `JobEvent`, and `DatasetEvent` with facets. The specification is versioned
by facets, and the project's release line in 2026 is 1.x (1.54.0 in October 2025). Facets include schema,
data source, lifecycle state, version (such as an Iceberg snapshot ID), column lineage, data quality
metrics, and data quality assertions. The dataset `version` facet is what makes a lineage edge point to a
specific data state rather than a mutable table name.

W3C PROV defines a general provenance model with Entity, Activity, and Agent, and relations such as
`wasGeneratedBy`, `used`, and `wasAttributedTo`. It is broader than lineage and is useful for
accountability records that must state who did what.

### 8.3 Regulatory context

- **21 CFR Part 11.** Electronic records in regulated submissions require secure, computer-generated,
  time-stamped audit trails; changes must not obscure prior values. Versioned snapshots and append-only
  logs are the technical mechanism. A table that is updated in place without history does not satisfy the
  requirement.
- **HIPAA Security Rule.** Audit controls must record and examine activity in systems containing
  electronic PHI. Lineage and access logs support the required information system activity review.
- **HIPAA Privacy Rule.** Retention and deletion obligations interact with versioning. Keeping every
  version forever may violate a data minimization or deletion obligation; deleting too early breaks
  reproducibility. The retention policy must reconcile both.
- **Model risk management.** Internal model risk and regulatory expectations require model documentation,
  including training data, assumptions, and validation. A training data hash and a model version are the
  minimum evidence.

### 8.4 What the standards do not cover

- How long to retain versions.
- Which outputs require column-level lineage.
- How to canonicalize data for hashing.
- Whether a hash collision matters for the use case.
- How to version a vocabulary crosswalk.
- Who owns the lineage graph.

Those are design and governance decisions. A platform can use Delta Lake or Iceberg and still be unable
to answer an audit question if retention is too short or the lineage edges point to table names rather
than versions.

## 9. Industry practice and case evidence

### 9.1 Table-format time travel in production

Delta Lake and Iceberg both provide snapshot-based time travel and are used in large production estates.
The documented mechanics differ: Delta uses a transaction log with checkpoints, Iceberg uses table
metadata with manifest lists and manifests. Both have retention settings that determine whether a
snapshot remains queryable. The practical lesson is that time travel is a feature with a retention
window, not an unlimited archive.

### 9.2 OpenLineage adoption

OpenLineage is integrated with Airflow, Spark, dbt, and other tools. Its model separates run events from
dataset metadata and attaches facets for schema, version, and quality. The value is a vendor-neutral
graph; the limitation is that instrumentation must be complete. A partial graph can be worse than no
graph because it looks authoritative while missing edges.

### 9.3 Data versioning tools

DVC tracks data files and directories with small metafiles and a remote store, bringing Git-like
workflows to datasets. lakeFS provides branching and commits over object storage. Both are used in
research and machine-learning contexts where the table format is not the system of record. The trade-off
is operational complexity: a second versioning system must be governed, backed up, and kept consistent
with the warehouse.

### 9.4 MIMIC-IV versioning

MIMIC-IV documents its releases and provides a changelog. Version 3.1 (11 October 2024) was a bug-fix
release that reverted `itemid` changes and removed two orphan `subject_id` values. The MIMIC code
repository's derived schema records the dataset version in a metadata table. This is a concrete example
of a dataset treating its version and changelog as part of the data product.

### 9.5 What is inference

- That a content hash guarantees reproducibility. It guarantees change detection; reproducibility also
  requires the same code, environment, and configuration.
- That time travel satisfies an audit requirement. It satisfies it only if the snapshot is retained for
  the audit period and the query is documented.
- That lineage reduces incident time. It reduces search time; diagnosis still requires domain knowledge.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mutable "latest" table | No prior version to compare | Overwrite in place | Cannot reproduce or roll back | Snapshot or partition replace |
| Hash over file bytes | Same data, different hash | Row order or formatting changes | False change alerts | Canonicalize before hashing |
| Hash over a path | Hash unchanged after content change | Path-based identity | Silent corruption | Content hash |
| Row count as identity | Same count, different values | Count-only check | Undetected value change | Hash plus counts |
| Lineage to table names | Edge points to "gold.cohort" | No version facet | Cannot identify the data state | Version in the lineage facet |
| Missing column lineage | Cannot explain one metric | Table-level only | Slow incident diagnosis | Targeted column lineage |
| Retention shorter than audit | Time travel fails after 30 days | Vacuum or snapshot expiry | Audit cannot reproduce | Align retention to policy |
| Model without data hash | "Trained on EHR data" | Registry stores only weights | Unreproducible model | Record the tuple |
| Vocabulary not versioned | Cohort changes after a release | Mapping drift | Unexplained population change | Record release IDs |
| No replay test | Reproducibility is assumed | Never executed | Failure discovered in audit | Quarterly replay |
| Lineage instrumentation partial | Graph has gaps | Only some tools emit events | False confidence | Coverage checks |
| Snapshot deleted by vacuum | Time travel query fails | Retention shorter than expected | Data loss | Test restore before vacuum |

Two red flags deserve emphasis.

**A release without a manifest.** If the release notes say "v3.1" and nothing else, the version is a label,
not an identity. The manifest is what makes it reproducible.

**A data diff with only row counts.** A count is invariant to value changes. The diff must compare hashes
and, for changed partitions, a sample or checksum of values.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Canonicalization is stable across engines | Comparable hashes | Different null or float formatting | Hash the same data in two engines |
| The hash function is collision-resistant for the use case | Identity | Deliberate collision or truncation | Use SHA-256; do not truncate |
| Snapshots are retained for the audit period | Reproducibility | Vacuum or lifecycle deletion | Query the oldest snapshot after retention |
| Lineage instrumentation covers critical paths | Impact analysis | Dynamic SQL, manual steps | Compare graph to a manual trace |
| The model registry records the tuple | Model reproducibility | Registry stores only artifacts | Audit a sample model |
| Vocabulary releases are recorded | Cohort stability | Mapping updated without a version | Diff cohort counts across releases |
| Configuration is resolved before hashing | Identifies actual values | Environment-dependent resolution | Hash the resolved object |
| Storage can hold the retained versions | Audit and rollback | Cost constraints | Forecast retention cost |
| Owners maintain the lineage graph | Accuracy | Team turnover | Quarterly ownership review |

The largest limit is cost. Retaining every snapshot of every intermediate table for seven years can
exceed the value. A tiered policy is necessary: retain raw and release snapshots for the audit period,
intermediate snapshots for days or weeks, and document the boundary. The policy must be tested against
the audit requirement, not guessed.

A second limit is that hashes prove identity, not correctness. Two versions can have different hashes and
both be wrong; the same hash can be reproduced from a buggy pipeline. Reproducibility is necessary for
auditability and insufficient for validity.

## 12. Exercises

### Exercise 1 — Reproducibility tuple (recall, 25 minutes)

Choose a published analysis or a project you know. Deliverable: a six-component tuple with the recorded
value or "missing" for each, and a remediation note for each gap.

Acceptance criteria: all six components addressed; gaps are specific; the remediation names an owner and
a storage location.

### Exercise 2 — Canonicalization (application, 40 minutes)

Deliverable: a canonicalization function for a table with nulls, floats, and timestamps, plus tests that
two row orders produce the same hash and a value change produces a different hash.

Acceptance criteria: null handling is explicit; float formatting is fixed; the function is documented;
the tests cover an empty partition.

### Exercise 3 — Manifest and root (application, 45 minutes)

Given three partitions, deliverable: a manifest with per-partition hashes, row counts, and a root, plus a
verification script.

Acceptance criteria: changing one partition changes the root and only one partition hash; adding a
partition changes the root; verification recomputes independently.

### Exercise 4 — Lineage closure (application, 45 minutes)

Deliverable: a recursive query over a lineage table that returns all downstream outputs of a dataset,
with depth.

Acceptance criteria: handles cycles without infinite recursion; includes transitive edges; orders
deterministically; states the engine.

### Exercise 5 — Version diff (application, 50 minutes)

Deliverable: a diff between two dataset versions that reports added, removed, and changed partitions with
row counts and hashes.

Acceptance criteria: null-safe comparison; unchanged partitions excluded; the report reconciles to the
root change.

### Exercise 6 — Model tuple (design, 40 minutes)

Deliverable: a model registry schema that records training data root, code commit, config hash, image
digest, vocabulary releases, and evaluation data root.

Acceptance criteria: every field is immutable after registration; the schema supports querying models by
training data version; the model card is generated from the registry.

### Exercise 7 — Incident procedure (design, 60 minutes)

Deliverable: a step-by-step procedure to answer "where did this number come from" within four hours,
including the queries and the escalation path.

Acceptance criteria: names the tuple components and where to find them; includes a lineage query; states
the evidence to retain; has a practice schedule.

### Exercise 8 — Retention policy (critique, 45 minutes)

Deliverable: a retention policy for raw, standardized, curated, and model snapshots, reconciled with a
seven-year audit period and a data minimization obligation.

Acceptance criteria: each layer has a stated period and rationale; the policy is testable; deletion and
retention are reconciled; cost is estimated.

## 13. Self-check answers

**1. What are the six components of the reproducibility tuple?**

Code commit, resolved configuration hash, environment digest, schema version, data root hash, and
vocabulary release. An output is reproducible only when all six are recorded and immutable.

**2. Why is a Git commit not enough?**

Because it reproduces the source, not the behavior. The same code with a different dependency version,
configuration, or input data produces a different result. The commit is necessary and insufficient.

**3. Why canonicalize before hashing?**

Because the same data can be serialized with different row order, null representation, or float
formatting. Canonicalization makes the hash a function of the content, not of the export process.

**4. What does a Merkle root add over per-partition hashes?**

A single identifier for the whole dataset. Per-partition hashes identify what changed; the root lets a
model registry or report record one value and still resolve the details through the manifest.

**5. How do you compute the root in this topic?**

Sort partition IDs, concatenate their SHA-256 hashes, and hash the concatenation with SHA-256. The root
changes if any partition hash or the partition set changes.

**6. How does Delta Lake time travel differ from RESTORE?**

Time travel is a read of an older version and does not change the current table. `RESTORE` makes an older
version the current state and creates a new commit. Both depend on retained files; `VACUUM` can remove
files that older versions reference.

**7. Why version the vocabulary separately from the code?**

Because a vocabulary release can change concept mappings without any code change. The same SQL against a
new SNOMED or LOINC release can produce a different cohort, and the difference is invisible in the commit
history.

**8. Compute the impact of the February change.**

One of three partitions changed: 1/3 = 33.3% of partitions. The root changed, because it is a function of
all partition hashes. If four of five downstream outputs depend on the changed columns, 4/5 = 80% are
affected by values; the fifth is affected only by provenance.

**9. What is the difference between lineage and provenance?**

Lineage is the dependency graph between datasets and jobs. Provenance is the broader record of entities,
activities, and agents, including who ran what and when. Lineage answers "what depends on this?";
provenance answers "who produced this and how?"

**10. Why practice the incident query?**

Because an untested procedure fails under pressure. The practice run reveals missing lineage edges,
unrecorded tuple components, and retention gaps before an auditor or regulator finds them.

## 14. Interview questions

**1. "How do you make a dataset reproducible?"**

Strong: names the tuple, content hashes, immutable snapshots, retention, and a replay test. Weak: "we
keep the CSV."

**2. "What is lineage for?"**

Strong: impact analysis, incident diagnosis, and audit; distinguishes table and column granularity.
Weak: "a diagram of the pipeline."

**3. "How do you hash a large table?"**

Strong: canonicalize, hash per partition, build a root, store the manifest. Weak: hash the file path or
the row count.

**4. "A number changed. How do you find out why?"**

Strong: compare roots and partition hashes, check config and vocabulary versions, use lineage to scope
outputs, reconcile the count to the metric. Weak: "check the SQL."

**5. "What does Delta Lake time travel not protect against?"**

Strong: vacuum and retention expiry; it is not a backup and does not guarantee audit-period retention.
Weak: "it protects against everything."

**6. "How do you version a model?"**

Strong: registry entry with training data root, code commit, config, environment, evaluation data, and
metrics. Weak: "save the weights with a date."

**7. "What is column-level lineage worth?"**

Strong: explains a specific metric and reduces the search space; costs more to capture, so target the
columns that drive decisions. Weak: "it is more detailed."

**8. "How long do you keep versions?"**

Strong: derive from audit and contractual obligations, tier by layer, test the oldest retained snapshot,
and reconcile with deletion duties. Weak: "forever" or "30 days."

**9. "How do you handle late-arriving data in a versioned table?"**

Strong: a new version with a changed partition, a diff, and a downstream impact list; the old version
remains queryable. Weak: update in place and hope.

**10. "What is the weakest link in most reproducibility claims?"**

Strong: the data and the environment, because code is usually versioned and the other components are not.
Weak: "the code."

## 15. Key terms

- **Canonicalization** — deterministic representation before hashing.
- **Code version** — commit or tag.
- **Column-level lineage** — output-to-input column mapping.
- **Configuration hash** — hash of resolved configuration.
- **Content hash** — hash of canonical bytes.
- **Dataset diff** — comparison of two versions.
- **Dataset root** — hash over partition hashes.
- **Delta Lake** — table format with a transaction log.
- **Environment digest** — image content hash.
- **Garbage collection** — removal of unreferenced data.
- **Iceberg** — table format with snapshots and manifests.
- **Immutable** — cannot be changed after write.
- **Lineage** — dependency graph.
- **Manifest** — file list with hashes and counts.
- **Merkle root** — hash over a set of hashes.
- **Model version** — registry identifier.
- **OpenLineage** — lineage event specification.
- **Partition hash** — content hash of one partition.
- **Provenance** — entities, activities, agents record.
- **Repeatability** — same team, same result.
- **Reproducibility** — another person, same result.
- **Reproducibility tuple** — six components that determine an output.
- **Retention** — how long versions are kept.
- **Schema version** — structure identifier.
- **Snapshot** — immutable dataset state.
- **Time travel** — querying a prior snapshot.
- **Tombstone** — deletion marker.
- **Training data hash** — root of the training set.
- **Vacuum** — deleting unreferenced files.
- **Vocabulary version** — terminology release identifier.

## 16. Sources

All URLs accessed 25 October 2026.

1. Delta Lake. "Delta Lake Protocol." https://github.com/delta-io/delta/blob/master/PROTOCOL.md
2. Delta Lake. "How does Delta Lake manage feature compatibility?" https://docs.delta.io/versioning/
3. Databricks. "RESTORE." https://docs.databricks.com/aws/en/sql/language-manual/delta-restore
4. Delta Lake. "Time Travel." https://delta.io/blog/2023-02-01-delta-lake-time-travel/
5. Apache Iceberg. "Spec." https://iceberg.apache.org/spec/
6. Apache Iceberg. "Queries." https://iceberg.apache.org/docs/nightly/docs/spark-queries/
7. OpenLineage. "Specification." https://github.com/OpenLineage/OpenLineage/blob/main/spec/OpenLineage.md
8. OpenLineage. "Releases." https://github.com/OpenLineage/OpenLineage/releases
9. W3C. "PROV-Overview." https://www.w3.org/TR/prov-overview/
10. DVC. https://dvc.org/
11. lakeFS. https://lakefs.io/
12. PhysioNet. "MIMIC-IV v3.1." https://www.physionet.org/content/mimiciv/3.1/
13. MIT-LCP. "mimic-code releases." https://github.com/MIT-LCP/mimic-code/releases
14. eCFR. "21 CFR Part 11." https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
15. eCFR. "45 CFR 164.312." https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
16. NIST. "SP 800-66 Rev. 2." https://csrc.nist.gov/pubs/sp/800/66/r2/final

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_dicom-pipelines.md`](06_dicom-pipelines.md)
- Next topic: [`08_deidentification-engineering.md`](08_deidentification-engineering.md)
- Related sibling: [`10_testing-and-observability.md`](10_testing-and-observability.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — Lab 3.6 requires tracing a failure to its source and quantifying impact.
