# Security Engineering

> Module 03 — Data Engineering and Interoperability
> Topic 09 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 07, 08
> Version 2026.3

## 1. Why this topic exists

A clinical data platform is a high-value target with a large attack surface. It holds identifiers, diagnoses,
genotypes, and imaging; it connects to electronic health record (EHR) systems, cloud storage, and vendor
application programming interfaces (APIs); and it is operated by a small team that is measured on delivery,
not on security. The result is predictable: credentials in repositories, service accounts with
administrator rights, buckets with public read, and audit logs that nobody reads.

The failures are ordinary.

- A pipeline configuration contains a database password. The repository is private, so the team treats the
  password as safe. A contractor clones the repository, and the password is in the clone, the CI logs, and
  a support bundle.
- A service account used by the nightly job has read access to every dataset in the account because it was
  easier than maintaining a list. When the account is compromised, the blast radius is the entire
  warehouse.
- A research bucket is set to public for a collaboration. The collaboration ends; the setting remains. The
  bucket is indexed by a search engine, and the exposure is discovered by a journalist.
- Audit logs record that a query ran, but not which user ran it, because all analysts share one database
  account. An investigation cannot attribute the access.
- A dependency is typosquatted, and the install script exfiltrates environment variables. The build had no
  lockfile, so the malicious version was resolved on a developer machine.
- A laptop with a local copy of a de-identified dataset is stolen. The dataset was de-identified, but the
  laptop also had the crosswalk.
- A notebook exports a table to a personal cloud drive for a quick analysis. The export bypasses every
  control the platform spent a year implementing.

Security engineering is the discipline that makes these failures less likely and their consequences
smaller. It is not a product or a compliance checkbox. It is a set of design decisions: who can access
what, under which conditions, with what evidence, and for how long.

This topic covers threat modeling, identity, least privilege, encryption, secrets, audit logging,
monitoring, incident response, and the regulatory context. It follows de-identification because the two
share a threat model: the data are sensitive, the recipients are varied, and the controls must be
proportionate.

> **Note:** This topic is educational and is not legal advice or a certification. A security program
> requires organizational commitment, qualified staff, and independent assessment. The practices here are
> a technical foundation, not a guarantee of compliance.

## 2. Learning outcomes

By the end of this topic you can:

1. **Build** a threat model for a data pipeline using assets, actors, flows, trust boundaries, and STRIDE.
   Assessed in Exercise 1.
2. **Score** risks by likelihood and impact, and show the effect of controls on the residual score.
   Assessed in Exercise 2 and self-check item 4.
3. **Design** an identity model that separates human and workload identities and enforces least privilege.
   Assessed in Exercise 3.
4. **Write** an attribute-based access policy and test it against allowed and denied requests. Assessed in
   Exercise 4.
5. **Explain** envelope encryption and key rotation, including why rotation does not require re-encrypting
   all data. Assessed in self-check item 5.
6. **Implement** a tamper-evident audit log and verify it after a modification. Assessed in Exercise 5.
7. **Design** monitoring and egress controls that detect exfiltration without blocking legitimate work.
   Assessed in Exercise 6.
8. **Write** an incident response runbook with containment, evidence preservation, and notification
   triggers. Assessed in Exercise 7 and Exercise 8.

## 3. The core mental model

Security is a control loop around a data flow, not a wall around a database.

```text
+-------------+      +------------------+      +------------------+      +-------------+
| ASSET       |      | TRUST BOUNDARY   |      | CONTROL          |      | EVIDENCE    |
| PHI, keys,  | ---> | identity change  | ---> | prevent, detect, | ---> | audit log,  |
| models, code|      | network, tenant  |      | respond, recover |      | metrics     |
+-------------+      +------------------+      +------------------+      +-------------+
       ^                                                                        |
       |                                                                        v
       +---------------------------- feedback / review -------------------------+
```

Every control should be tied to an asset, a threat, and a piece of evidence. A control with no evidence
cannot be audited; an asset with no control is a finding waiting to happen.

### 3.1 Assets, actors, and flows

**Assets**: PHI, de-identified datasets, the crosswalk, encryption keys, credentials, models, source code,
and the availability of the pipeline itself.

**Actors**: clinicians, analysts, data scientists, engineers, administrators, vendors, attackers, and
automated workloads. Each has a different trust level and a different legitimate need.

**Flows**: ingestion from EHR, transformation in the warehouse, export to a research environment, model
serving, backup, and vendor access. Each flow crosses at least one trust boundary.

A threat model lists these three and then asks, for each flow, what could go wrong.

### 3.2 STRIDE

STRIDE is a mnemonic for threat categories:

| Category | Threat | Example in a clinical pipeline |
|---|---|---|
| Spoofing | Impersonating an identity | Stolen service account token |
| Tampering | Modifying data or code | Editing a partition or an audit log |
| Repudiation | Denying an action | Shared account with no attribution |
| Information disclosure | Exposing data | Public bucket, over-broad read role |
| Denial of service | Disrupting availability | Saturated warehouse or deleted dataset |
| Elevation of privilege | Gaining unauthorized rights | Analyst exploits a definer function |

STRIDE is useful because it is exhaustive enough to structure a conversation and simple enough to
complete in a workshop. It is not a risk score; each threat still needs a likelihood and an impact.

### 3.3 Least privilege and blast radius

Least privilege means every identity has only the permissions required for its task, for the duration of
the task. Blast radius is the set of assets an identity can reach if compromised. The two are related but
not identical: a role can be least-privilege for its task and still have a large blast radius if the task
itself touches many datasets.

```text
Broad role:        service-ingest --> 14 datasets (read), 3 buckets (write)
Task-scoped role:  service-ingest --> 1 dataset (write), 1 bucket (write)
Blast radius reduction: 11 of 14 datasets, 78.6%
```

The design question is not only "does this role have more than it needs?" but "if this identity is
compromised at 3 a.m., what can the attacker reach before anyone notices?"

### 3.4 Defense in depth

No single control is sufficient. Identity, network, encryption, monitoring, and response are layered so
that a failure in one does not become a breach.

```text
Layer 1: identity and authentication
Layer 2: authorization and least privilege
Layer 3: network segmentation and private endpoints
Layer 4: encryption at rest and in transit
Layer 5: audit logging and anomaly detection
Layer 6: data loss prevention and egress control
Layer 7: backup, recovery, and incident response
```

A public bucket is a failure of layer 2 and layer 6; encryption at rest does not save it, because the
attacker reads through the authorized path. Defense in depth is not redundancy for its own sake; it is the
recognition that controls fail independently.

### 3.5 The audit log is a security control

An audit log that records who accessed what, when, and from where is the evidence for every other
control. Without it, an incident cannot be scoped, a breach cannot be attributed, and a regulator cannot
be answered. The log itself must be protected: if an attacker can edit it, the evidence is worthless.

A hash chain is a simple tamper-evidence mechanism: each record includes the hash of the previous record.
Changing an earlier record breaks every subsequent hash, and verification detects the break. It does not
prevent deletion of the entire log; that requires append-only storage or an external anchor.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| PHI | Protected health information | Confused with any health data | 45 CFR 160.103 |
| ePHI | Electronic PHI | — | 45 CFR 160.103 |
| Threat model | Structured analysis of assets, actors, flows, and threats | Confused with a vulnerability scan | — |
| STRIDE | Threat categories: spoofing, tampering, repudiation, disclosure, denial, elevation | Used as a risk score | — |
| Trust boundary | Where the trust level changes | Confused with a network firewall | — |
| Least privilege | Minimum permissions for the task | Confused with a read-only role | — |
| Blast radius | Assets reachable if an identity is compromised | Confused with severity | — |
| RBAC | Role-based access control | Confused with ABAC | — |
| ABAC | Attribute-based access control | Confused with a role list | — |
| Identity | A human or workload principal | Confused with an account name | — |
| Workload identity | Identity assigned to a service | Confused with a shared service account | SPIFFE, cloud IAM |
| MFA | Multi-factor authentication | Assumed phishing-resistant | — |
| Service account | Non-human identity for a workload | Shared across pipelines | — |
| Secret | Credential or key | Stored in configuration | — |
| Secret manager | Service for storing and rotating secrets | Confused with a config file | — |
| Encryption at rest | Encryption of stored data | Assumed to prevent insider access | AES-256 |
| Encryption in transit | Encryption of network traffic | Confused with at-rest encryption | TLS 1.2+ |
| Envelope encryption | Data key encrypts data; key-encryption key encrypts the data key | Confused with double encryption | — |
| KMS | Key management service | Confused with a secret store | — |
| Key rotation | Replacing a key periodically | Assumed to require data re-encryption | — |
| mTLS | Mutual TLS authentication | Confused with TLS | — |
| Zero trust | No implicit trust based on network location | Confused with a product | — |
| Segmentation | Dividing a network into zones | Confused with a firewall rule | — |
| Audit log | Record of access and actions | Confused with application logs | — |
| Tamper-evident | Modifications are detectable | Confused with tamper-proof | Hash chain |
| SIEM | Security information and event management | Confused with a log store | — |
| DLP | Data loss prevention | Confused with encryption | — |
| Egress | Outbound data transfer | Ignored until a bill or a breach | GB |
| Vulnerability | A weakness that can be exploited | Confused with a risk | CVE |
| CVSS | Severity score for a vulnerability | Confused with organizational risk | 0-10 |
| SBOM | Software bill of materials | Confused with a security guarantee | SPDX, CycloneDX |
| Incident | A confirmed or suspected security event | Confused with an outage | — |
| Runbook | Procedure for an operational event | Confused with documentation | — |
| BAA | Business associate agreement | Confused with a DUA | HIPAA |
| DUA | Data use agreement | Confused with a BAA | Contract |
| Residual risk | Risk after controls | Assumed zero | — |
| Tabletop | Discussion-based incident exercise | Confused with a live test | — |

Three terms need expansion.

**Workload identity.** A pipeline should authenticate as itself, not as a human and not with a static
password. Cloud providers offer workload identity federation, and SPIFFE/SPIRE provides a platform-neutral
identity for services. The benefit is short-lived credentials and attributable actions.

**Envelope encryption.** A data key encrypts the data; a key-encryption key in a KMS encrypts the data
key. Rotating the key-encryption key requires rewrapping the data keys, not re-encrypting the data. That
is why rotation is operationally feasible at scale, and why losing the key-encryption key is catastrophic.

**Tamper-evident versus tamper-proof.** A hash chain makes modification detectable; it does not prevent it.
Tamper-proof requires append-only storage, a write-once medium, or an external anchor. Most organizations
implement tamper-evident logs and call them tamper-proof, which overstates the guarantee.

## 5. The mechanics, step by step

The procedure builds a threat model and a control set for a clinical data platform. It is written to be
run as a workshop with engineering, security, and privacy participants.

### Step 1 — Inventory assets and classify data

Input: the platform inventory.
Output: an asset register with owner, location, classification, and retention.

Assets include datasets, the crosswalk, keys, credentials, models, source code, images, backups, and the
pipeline's availability. Classify by sensitivity: PHI, de-identified, internal, public.

Reason: controls are proportionate to classification. A public reference dataset does not need the same
access review as the crosswalk, and treating them the same wastes effort and obscures the real risk.

### Step 2 — Map data flows and trust boundaries

Input: the architecture diagram.
Output: a flow diagram with each boundary marked.

Draw the flow from source to consumer, including backups, exports, vendor access, and developer
workstations. Mark where identity changes, where the network changes, and where the data classification
changes.

Reason: most breaches cross a boundary that was not drawn. A vendor's support tunnel and a scientist's
laptop are part of the flow.

### Step 3 — Enumerate actors and their legitimate needs

Input: the flow diagram.
Output: an actor list with role, purpose, data scope, and duration.

Include humans, workloads, vendors, and administrators. For each, state the minimum data and the minimum
duration required. A data scientist who needs one cohort for one month does not need standing access to
the warehouse.

Reason: access that is not tied to a purpose becomes permanent. The actor list is the basis for the
authorization model.

### Step 4 — Build the threat model with STRIDE

Input: assets, flows, actors.
Output: a threat table with category, scenario, affected asset, and existing controls.

For each flow, ask the six STRIDE questions. Example: for the export flow, information disclosure
(export to a personal drive), tampering (modified file in transit), repudiation (no record of who
exported).

Reason: a structured question list produces coverage that brainstorming misses. The table is also the
agenda for the risk scoring.

### Step 5 — Score inherent risk

Input: the threat table.
Output: likelihood and impact scores, and an inherent risk score.

Use a 1-5 scale for each and multiply. Define the scales before scoring: likelihood from "rare" to
"almost certain," impact from "minor" to "severe" with clinical, financial, legal, and reputational
dimensions.

Reason: scoring forces a conversation about impact that a list of threats does not. The scales must be
defined, or the scores are opinions.

### Step 6 — Select controls and score residual risk

Input: the scored threats.
Output: a control per threat and a residual score.

Prefer preventive controls for high-impact threats, detective controls where prevention is impractical,
and compensating controls where a requirement cannot be met. Record the residual score and the owner.

Reason: a control that does not change the score is not a control. The residual score is the basis for
acceptance.

### Step 7 — Design the identity model

Input: the actor list.
Output: a human identity provider, a workload identity mechanism, and a break-glass procedure.

Humans authenticate through the organization's identity provider with MFA. Workloads use short-lived
credentials from a workload identity federation or a managed identity. Administrators use a separate,
audited role with just-in-time elevation.

Reason: shared accounts destroy attribution and make least privilege impossible. Break-glass access must
exist and must be monitored, or administrators will create shadow access.

### Step 8 — Implement least privilege

Input: the identity model and the data scope.
Output: roles or policies scoped to the task.

Start from deny by default, grant the minimum, and review. For a pipeline, scope to the specific dataset
and operation: write to one partition, read from one source. For an analyst, scope to a curated view, not
the raw zone.

Reason: broad roles are convenient and permanent. The blast radius of a compromised pipeline is the set
of datasets its role can reach.

### Step 9 — Add attribute-based conditions where roles are insufficient

Input: the access requirements that depend on context.
Output: policies that consider purpose, department, time, location, and training status.

Example: a researcher may read de-identified data if their training is current, their purpose is
approved, and the request comes from a managed device. A role alone cannot express those conditions.

Reason: clinical access often depends on context, not only on job title. ABAC expresses the context and
produces an auditable decision.

### Step 10 — Centralize secrets

Input: all credentials, tokens, and keys.
Output: a secret manager with rotation, access policy, and audit.

Remove secrets from code, configuration files, notebooks, and CI variables. Inject them at runtime.
Rotate on a schedule and on suspicion. Never log the value.

Reason: secrets in repositories persist in history and clones. A secret manager gives one place to
rotate and one place to audit.

### Step 11 — Encrypt at rest and in transit

Input: storage and network paths.
Output: encryption configuration with key ownership.

Use the cloud provider's encryption with customer-managed keys where the classification requires it.
Enforce TLS 1.2 or later for all connections, and mTLS for service-to-service where supported. Disable
legacy protocols.

Reason: encryption at rest protects against media theft and some insider paths; encryption in transit
protects against interception. Neither protects against an authorized application that reads data and
sends it elsewhere.

### Step 12 — Manage keys separately from data

Input: the encryption design.
Output: a KMS configuration, key hierarchy, and rotation schedule.

Use envelope encryption. Store key-encryption keys in a KMS with access policies separate from the data
store. Rotate annually or on a shorter schedule for high-sensitivity data. Test that a revoked key
actually blocks access.

Reason: a key stored with the data is not a control. Separation is what makes encryption meaningful
against an attacker who obtains the storage credentials.

### Step 13 — Segment the network

Input: the flow diagram.
Output: private endpoints, subnets, and firewall rules.

Put storage and databases on private endpoints. Allow egress only to approved destinations. Separate
development, staging, and production accounts. Do not give a notebook direct internet access by default.

Reason: segmentation limits lateral movement and makes exfiltration harder. It also makes accidental
exposure less likely.

### Step 14 — Log access and actions

Input: the identity and data model.
Output: an audit log with actor, action, resource, timestamp, source, and outcome.

Log authentication, authorization decisions, data reads and writes, exports, configuration changes, and
key use. Do not log the data values or the secrets. Use a correlation ID to link a request across
services.

Reason: the audit log is the evidence for every investigation. Without the actor and the resource, it is
a volume metric, not an audit trail.

### Step 15 — Make the log tamper-evident

Input: the audit log design.
Output: a hash chain or append-only store, with verification.

Chain each record to the previous one, store the chain in append-only storage, and verify periodically.
Export a digest to a separate account or a write-once store. Alert on verification failure.

Reason: an attacker with write access to the log can erase evidence. Tamper-evidence raises the cost and
provides a detection signal.

### Step 16 — Monitor and alert

Input: the audit log and infrastructure metrics.
Output: detection rules with thresholds, owners, and runbooks.

Detect: access from unusual locations, bulk reads, privilege changes, failed authentication spikes, key
disablement, public bucket changes, and exports above a threshold. Tune thresholds to the environment and
measure false positives.

Reason: an unread log is storage cost. Detection converts the log into a control, and tuning keeps the
on-call team from ignoring it.

### Step 17 — Control egress and exports

Input: the export paths.
Output: an approved export process with DLP and approval.

Require exports to go through a governed channel with a purpose, an approval, and a record. Apply DLP
rules for identifiers and data volumes. Block personal cloud storage and removable media on managed
devices.

Reason: exfiltration often uses an authorized path. The control is not to block all exports but to make
them visible and approved.

### Step 18 — Manage vulnerabilities and dependencies

Input: the software inventory.
Output: a patch process, a dependency policy, and an SBOM.

Scan container images and dependencies. Pin versions with a lockfile. Track CVEs and have a timeline by
severity. Generate an SBOM per release.

Reason: most pipelines depend on hundreds of packages. A vulnerability in a transitive dependency is
invisible without an inventory.

### Step 19 — Back up and test recovery

Input: the data and configuration inventory.
Output: a backup schedule, a restore test, and a recovery time objective.

Back up data, code, configuration, and keys. Encrypt backups and store them separately. Test a restore
quarterly and record the time.

Reason: ransomware and accidental deletion are security incidents. A backup that has never been restored
is a hope, not a control.

### Step 20 — Write the incident response plan

Input: the threat model and the controls.
Output: a plan with roles, severity levels, containment steps, evidence preservation, and notification
triggers.

Define who declares an incident, who leads, who communicates, and who decides on notification. Include
the legal and privacy contacts. Define the evidence to preserve: logs, snapshots, access records.

Reason: incident response is a team sport under time pressure. The plan removes decisions that should not
be made at 3 a.m.

### Step 21 — Exercise the plan

Input: the incident plan.
Output: a tabletop exercise with findings and actions.

Run a scenario: a service account token is found in a public repository. Walk through detection,
containment, scoping, notification, and recovery. Record the gaps and assign owners.

Reason: plans fail on contact lists, tooling access, and unclear authority. The exercise finds those
failures before a real incident.

### Step 22 — Review access and controls periodically

Input: the access list and the control set.
Output: a quarterly review with removals and exceptions.

Review human access, service accounts, roles, and exceptions. Remove what is no longer needed. Record the
review and the approver. Re-score the risk register annually or when the architecture changes.

Reason: access accumulates. A quarterly review is the mechanism that keeps least privilege from decaying
into least effort.

### Step 23 — Handle credential compromise and rotation

Input: a suspected or confirmed credential exposure.
Output: a containment sequence and a rotation record.

Revoke the credential first, then rotate, then scope the access from the audit log, then notify. Do not
wait to determine the blast radius before revoking; the audit is preserved independently of the
credential. Record the time of revocation, the time of rotation, and the events attributed to the
credential.

| Case | First action | Evidence to preserve |
|---|---|---|
| Token in a repository | Revoke and rotate | Commit history, clone list, audit events |
| Key in a support bundle | Disable the key; rewrap data keys | Bundle contents, key-use log |
| Shared password | Disable the account; issue individual identities | Authentication log |
| Vendor credential | Suspend the vendor account | Session recordings, access log |
| Leaked backup | Re-encrypt; rotate the backup key | Backup inventory, restore log |

Reason: containment precedes investigation. An investigation that runs while the credential is active
gives the attacker time to erase evidence.

### Step 24 — Handle partial failure and audit gaps

Input: a service outage or a pipeline failure.
Output: a rule for what is logged and what is replayed.

If the audit pipeline is down, the application must fail closed for high-sensitivity operations or buffer
events locally with integrity protection. If a batch job fails after a partial write, the audit must
record the attempt and the rollback, not only the successful commit.

| Failure | Audit behavior | Recovery |
|---|---|---|
| Audit service unavailable | Buffer or fail closed | Replay buffer; verify order |
| Partial data write | Log attempt and rollback | Retry with idempotency key |
| Log shipper gap | Detect by sequence number | Backfill from source |
| Clock skew | Record both node and source time | Normalize to UTC |
| Key service outage | Fail closed for decryption | Restore key service; no bypass |

Reason: a gap in the audit log is indistinguishable from an absence of activity. The sequence number and
the buffer make the gap visible.

### Step 25 — Handle clock and timezone boundaries

Input: log timestamps from multiple services.
Output: a normalized time model.

Record event time in UTC with the source offset, and record ingestion time separately. A log that stores
local time without an offset cannot be ordered across regions, and a log that stores only ingestion time
cannot reconstruct when an action occurred.

| Boundary | Risk | Rule |
|---|---|---|
| DST fall back | Duplicate local timestamps | Store UTC plus offset |
| Multi-region | Events out of order | Normalize to UTC |
| Clock skew | False ordering | NTP; record both times |
| Leap second | Rare parser failure | Use a monotonic sequence for order |
| Retention boundary | Log deleted before audit | Retain per policy; test |

Reason: audit evidence is temporal. An event that cannot be placed in time cannot be attributed to a
session or a change window.

### Branch point A — The platform is a trusted research environment

The controls shift toward export prevention and session monitoring: no bulk download, no clipboard, no
unapproved external network. The identity model is stronger and the audit is more detailed because the
environment is the boundary.

### Branch point B — The data are de-identified

De-identified data still require access control, audit, and encryption, but the impact of disclosure is
lower. Do not remove controls; adjust the risk score and the review frequency.

### Branch point C — A vendor needs access

Use a vendor account with scoped permissions, time-bound access, and session recording. Require a BAA
where PHI is involved. Do not share a service account or a permanent VPN.

## 6. Worked example

A research platform holds 3.2 million patient records, 180 human users, and 42 service accounts. The team
runs a threat modeling workshop and produces a risk register.

### 6.1 Risk register

Likelihood (L) and impact (I) are scored 1-5; risk is L × I.

| # | Threat | L | I | Inherent | Controls | Residual L | Residual I | Residual |
|---|---|---:|---:|---:|---|---:|---:|---:|
| 1 | Stolen service account credential | 4 | 5 | 20 | Workload identity, short-lived tokens, secret manager | 2 | 5 | 10 |
| 2 | Public storage bucket | 4 | 5 | 20 | Policy guardrail, auto-remediation, weekly scan | 1 | 5 | 5 |
| 3 | Over-broad analyst access | 4 | 4 | 16 | ABAC, quarterly review, curated views | 2 | 4 | 8 |
| 4 | Phishing of an administrator | 4 | 4 | 16 | MFA, conditional access, training | 2 | 4 | 8 |
| 5 | Dependency compromise | 3 | 5 | 15 | Lockfile, SBOM, private index, review | 2 | 3 | 6 |
| 6 | Export to personal storage | 4 | 3 | 12 | DLP, egress control, managed devices | 2 | 3 | 6 |
| 7 | Audit log tampering | 3 | 4 | 12 | Hash chain, append-only store, external digest | 1 | 3 | 3 |
| 8 | SQL injection in an API | 3 | 4 | 12 | Parameterized queries, WAF, review | 1 | 4 | 4 |
| 9 | Unencrypted backup | 2 | 5 | 10 | Encryption, separate key, restore test | 1 | 3 | 3 |
| 10 | Lost laptop with a local copy | 2 | 5 | 10 | Full-disk encryption, no local PHI, MDM | 1 | 2 | 2 |
| **Total** | | | | **143** | | | | **55** |

The aggregate residual is 55 / 143 = 38.46% of the inherent score, a reduction of 61.54%. The reduction
is not evenly distributed: the public bucket and the audit log improve the most, while the over-broad
access and phishing risks remain at 8 because the human and organizational components are harder to
control.

### 6.2 Blast radius

The ingestion service account originally had read access to 14 datasets and write access to 3 buckets.

After scoping:

- Read: 3 datasets (source staging, terminology, and the target cohort's reference table).
- Write: 1 bucket (the landing zone prefix for its own source).

The read blast radius falls from 14 to 3 datasets, a reduction of 11 / 14 = 78.6%. The write blast radius
falls from 3 to 1, a reduction of 66.7%. The account can no longer read the crosswalk, the genomic
archive, or the imaging metadata.

### 6.3 Audit volume and retention

180 users generate an average of 240 audit events per user per day.

- Events per day: 180 × 240 = 43,200.
- Events per year: 43,200 × 365 = 15,768,000.
- At 1 KB per event: 15,768,000 KB = 15.77 GB per year (decimal).
- Over a six-year retention: 94.6 GB.

The volume is small relative to the data platform, which is why there is no excuse for dropping audit
events for cost. The 1 KB estimate includes the actor, action, resource, timestamp, source IP, and
outcome; it does not include query text, which can be larger and should be sampled or truncated.

### 6.4 Key rotation

The platform uses envelope encryption: 42 TB of data is encrypted with data keys, and the data keys are
encrypted with a key-encryption key in a KMS.

- Rotating the key-encryption key requires rewrapping the data keys, not re-encrypting 42 TB.
- If there are 1,240 data keys (one per partition), rotation rewraps 1,240 small values.
- Re-encrypting 42 TB at 200 MB/s would take 42,000,000 MB / 200 MB/s = 210,000 seconds = 58.3 hours,
  plus compute cost. Rewrapping takes seconds.

The arithmetic explains why envelope encryption is the default. It also explains the risk: if the
key-encryption key is lost, every data key is unrecoverable, and the data are unreadable even though the
ciphertext is intact.

### 6.5 The incident

A service account token is found in a public repository. The token belongs to a reporting job with read
access to the curated zone.

The timeline:

- 09:12: a scanner detects the token pattern and opens an alert.
- 09:14: the on-call engineer confirms the token is active.
- 09:18: the token is revoked and the job is rotated to workload identity.
- 09:25: audit logs are queried for the token's identity over the past 30 days.
- 09:40: the query shows 14 access events from an unrecognized IP on the previous day.
- 10:05: the accessed datasets are identified: the curated cohort and the quality report.
- 10:30: the privacy officer is engaged; the data are de-identified, and the preliminary assessment is
  that notification is not required.
- 11:00: the repository history is purged and the scanner rule is tuned.

The token had read access to 14 datasets but the audit shows only 2 were accessed. Blast radius is a
worst-case measure; the audit determines the actual scope. Without the audit, the organization would have
to assume all 14.

### 6.6 Control evidence ledger

Each control in the risk register has an evidence source and an expected value.

| Control | Evidence | Query or test | Expected |
|---|---|---|---|
| Human MFA | Identity provider configuration | Count active users without MFA | 0 |
| Workload identity | Cloud IAM inventory | Count service accounts with static keys | 0 |
| Least privilege | Role-dataset mapping | Entitlements per role | ≤ task scope |
| Encryption at rest | Bucket and database configuration | Count unencrypted resources | 0 |
| Key separation | KMS key policy | Data-store identities with key admin | 0 |
| Audit completeness | Event sequence | Missing sequence numbers per day | 0 |
| Log integrity | Hash chain | Verification failures | 0 |
| Egress control | Network policy | Allowed destinations outside the list | 0 |
| Backup | Restore test | Restore time and checksum | ≤ RTO, match |

The ledger is the audit pack. A control without evidence is a claim, and a claim without a query is not
testable.

### 6.7 Access reconciliation

| Metric | Value | Calculation |
|---|---:|---|
| Human users | 180 | Active accounts |
| Datasets in scope | 14 | Curated and raw |
| Potential entitlements | 2,520 | 180 × 14 |
| Actual entitlements | 1,842 | Role assignments |
| Entitlements unused for 90 days | 312 | Audit query |
| Unused share | 16.9% | 312 / 1,842 |
| Excessive entitlements removed | 208 | Review decision |

The unused share is 16.9%, not the 12.4% that a simple 312 / 2,520 would suggest, because not every user
has every dataset. The correct denominator is the actual entitlements, not the potential ones. This is the
kind of arithmetic error that makes an access review look better than it is.

### 6.8 Audit chain verification

The platform records 43,200 events per day. A 30-day sample contains 1,296,000 events.

| Check | Expected | Observed | Result |
|---|---:|---:|---|
| Events in sample | 1,296,000 | 1,296,000 | pass |
| Sequence gaps | 0 | 0 | pass |
| Hash-chain failures | 0 | 0 | pass |
| Events without an actor | 0 | 0 | pass |
| Events without a resource | 0 | 0 | pass |

The verification is a batch job that reads the chain and recomputes each hash. A single tampered event
breaks the chain at that index and every later event, which is why the verifier reports the first bad
index rather than a count.

### 6.9 Diagnostic checks

**Check A: MFA coverage.** Expected 180 of 180 human users. Observed 180. Pass. A single exception is a
finding, not a rounding error.

**Check B: workload identity coverage.** Expected 42 of 42 service accounts with short-lived credentials.
Observed 30. Fail: 12 accounts still use static keys, 12 / 42 = 28.6%. These are the highest-risk
identities because their credentials do not expire.

**Check C: unused entitlements.** Expected ≤10% of actual entitlements unused for 90 days. Observed 16.9%.
Fail. The review removes 208 entitlements, reducing the unused share to 104 / 1,634 = 6.4%.

### 6.10 Decision table for a failed control

| Symptom | First check | Likely cause | Action |
|---|---|---|---|
| MFA exception | IdP user status | Break-glass or legacy account | Enforce; document exception |
| Static service key | IAM inventory | Legacy pipeline | Migrate to workload identity |
| Unused entitlement | Audit query | Role drift | Remove; re-review |
| Chain verification failure | First bad index | Tampering or corruption | Preserve evidence; investigate |
| Public bucket | Storage policy | Misconfiguration | Block public access; audit reads |
| Egress to unknown host | Network flow logs | Shadow export | Block; investigate |
| Missing audit events | Sequence numbers | Shipper gap | Backfill; test alert |
| Backup not restorable | Restore test | Untested backup | Fix; retest quarterly |
| Key access too broad | KMS policy | Admin role includes decrypt | Separate roles |

### 6.11 What the residual score does not prove

The residual score of 55 is a relative measure on a 1-5 scale. It does not prove that the platform is
secure; it proves that the team has a documented, comparable view of ten threats. The score excludes
unknown threats, supply-chain attacks that have not been modeled, and insider misuse that the audit log
cannot distinguish from legitimate work. Those gaps belong in the risk acceptance, not in the arithmetic.

### 6.12 Second access review

The review is repeated after the removals.

| Metric | Review 1 | Review 2 | Expected |
|---|---:|---:|---|
| Actual entitlements | 1,842 | 1,634 | −208 |
| Unused for 90 days | 312 | 104 | −208 |
| Unused share | 16.9% | 6.4% | ≤10% |
| Users without MFA | 0 | 0 | 0 |
| Static service keys | 12 | 5 | 0 |

The second review removes 208 entitlements and finds 104 still unused, which is 6.4% of 1,634. The static
keys fall from 12 to 5 because only seven pipelines could be migrated in the quarter; the remaining five
have documented exceptions with an expiry date. A review that removes entitlements but leaves the
exception list open has not finished the work.

### 6.13 What would change the residual score

The residual score is a snapshot. It changes when:

1. A new site joins with different identity infrastructure, raising the likelihood of access-control
   drift.
2. Genomic data are added, raising the impact of disclosure.
3. A new cloud region is used, changing the residency and key-management assumptions.
4. A vendor receives standing access, raising the third-party risk.
5. An incident occurs, which resets the evidence for the affected controls and requires a re-score.

The register is reviewed annually and on each of these triggers. A score that is updated only annually is
a historical document, not a risk management tool.

The residual score is also not additive across business units. A score of 55 for the research platform and
55 for the clinical warehouse does not produce a score of 110 for the organization; the threats, controls,
and risk appetite differ. The comparison is valid only within the same register and the same scales.

The register's most useful property is that it names an owner for every row. A score without an owner is
an observation; a score with an owner and a review date is a commitment.

The review records dissent as well as agreement. A control that the security team believes is inadequate
but the business accepts is a documented exception, not a silent risk.

## 7. Code, query, or calculation

### 7.1 Tamper-evident audit log and ABAC policy (standard library, executable)

The script implements a hash-chained audit log, verifies it, tampers with one record, and verifies again.
It then evaluates attribute-based access requests. It was executed on CPython 3.11.15.

```python
"""Hash-chained audit log and attribute-based access control. Standard library only."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def append_event(chain: list[dict], event: dict) -> dict:
    previous = chain[-1]["hash"] if chain else "0" * 64
    body = dict(event)
    body["prev_hash"] = previous
    body["hash"] = hashlib.sha256(previous.encode() + canonical(event)).hexdigest()
    chain.append(body)
    return body


def verify(chain: list[dict]) -> tuple[bool, int | None]:
    previous = "0" * 64
    for index, entry in enumerate(chain):
        body = {k: v for k, v in entry.items() if k != "hash"}
        expected = hashlib.sha256(previous.encode() + canonical(
            {k: v for k, v in body.items() if k != "prev_hash"}
        )).hexdigest()
        if entry["prev_hash"] != previous or entry["hash"] != expected:
            return False, index
        previous = entry["hash"]
    return True, None


chain: list[dict] = []
append_event(chain, {"actor": "alice", "action": "read", "resource": "gold.cohort"})
append_event(chain, {"actor": "bob", "action": "export", "resource": "gold.cohort"})
append_event(chain, {"actor": "carol", "action": "read", "resource": "gold.quality"})

ok, bad = verify(chain)
print("verified", ok, bad)
print("chain_length", len(chain))

chain[1]["actor"] = "mallory"
ok2, bad2 = verify(chain)
print("verified_after_tamper", ok2, "first_bad_index", bad2)


@dataclass(frozen=True)
class Request:
    user: str
    role: str
    purpose: str
    department: str
    training_current: bool
    managed_device: bool
    resource_class: str


def decide(request: Request) -> tuple[bool, str]:
    if not request.training_current:
        return False, "training_expired"
    if not request.managed_device:
        return False, "unmanaged_device"
    if request.resource_class == "phi":
        if request.role not in {"clinician", "privacy_officer"}:
            return False, "role_not_permitted_for_phi"
        if request.purpose not in {"treatment", "audit"}:
            return False, "purpose_not_permitted_for_phi"
        return True, "allowed_phi"
    if request.resource_class == "deidentified":
        if request.role in {"researcher", "analyst", "clinician"}:
            if request.purpose in {"research", "quality", "treatment"}:
                return True, "allowed_deidentified"
        return False, "purpose_or_role_not_permitted"
    return False, "unknown_resource_class"


REQUESTS = [
    Request("alice", "researcher", "research", "cardiology", True, True, "deidentified"),
    Request("bob", "researcher", "research", "cardiology", True, True, "phi"),
    Request("carol", "clinician", "treatment", "cardiology", True, True, "phi"),
    Request("dave", "analyst", "quality", "finance", False, True, "deidentified"),
    Request("erin", "analyst", "quality", "finance", True, False, "deidentified"),
]

for request in REQUESTS:
    allowed, reason = decide(request)
    print(request.user, allowed, reason)
```

Expected output:

```text
verified True None
chain_length 3
verified_after_tamper False first_bad_index 1
alice True allowed_deidentified
bob False role_not_permitted_for_phi
carol True allowed_phi
dave False training_expired
erin False unmanaged_device
```

Tampering with record 1 breaks the chain at index 1, and every later record is also invalid. The ABAC
policy denies a researcher access to PHI even with a valid purpose, denies an analyst with expired
training, and denies an unmanaged device. Each denial has a reason code that can be logged and reviewed.

### 7.2 PostgreSQL 18 access review query

The following targets PostgreSQL 18 and finds entitlements that have not been used in 90 days.

```sql
-- PostgreSQL 18
SELECT
    u.user_name,
    g.role_name,
    d.dataset_name,
    MAX(a.event_time) FILTER (WHERE a.action = 'read') AS last_read,
    COUNT(*) FILTER (WHERE a.action = 'read'
                     AND a.event_time >= now() - INTERVAL '90 days') AS reads_90d
FROM meta.app_user u
JOIN meta.user_role ur      ON ur.user_id = u.user_id
JOIN meta.role g            ON g.role_id = ur.role_id
JOIN meta.role_dataset rd   ON rd.role_id = g.role_id
JOIN meta.dataset d         ON d.dataset_id = rd.dataset_id
LEFT JOIN audit.access_event a
       ON a.user_id = u.user_id
      AND a.dataset_id = d.dataset_id
WHERE u.active
GROUP BY u.user_name, g.role_name, d.dataset_name
HAVING COUNT(*) FILTER (WHERE a.action = 'read'
                        AND a.event_time >= now() - INTERVAL '90 days') = 0
ORDER BY u.user_name, d.dataset_name;
```

The query returns standing entitlements with no recent use. It is a review input, not an automatic
removal: a role may be legitimately unused for a quarter and still required for an on-call rotation. The
reviewer decides.

## 8. Data, standards, and regulatory context

### 8.1 HIPAA Security Rule

The Security Rule is at 45 CFR 164.308 (administrative), 164.310 (physical), 164.312 (technical), 164.314
(organizational), and 164.316 (policies and documentation). The technical safeguards are the most
relevant to a pipeline:

| Standard | Requirement | Implementation specifications |
|---|---|---|
| Access control, §164.312(a) | Limit access to authorized persons and software | Unique user identification (required), emergency access (required), automatic logoff (addressable), encryption (addressable) |
| Audit controls, §164.312(b) | Record and examine activity in systems with ePHI | No implementation specifications |
| Integrity, §164.312(c) | Protect against improper alteration or destruction | Mechanism to authenticate ePHI (addressable) |
| Person or entity authentication, §164.312(d) | Verify identity | None specified |
| Transmission security, §164.312(e) | Guard against unauthorized access in transit | Integrity controls (addressable), encryption (addressable) |

"Addressable" does not mean optional. The entity must assess whether the specification is reasonable and
appropriate, implement it if so, or document why not and implement an equivalent alternative.

The audit control standard is deliberately open: it requires mechanisms that record and examine activity,
without prescribing the mechanism. A hash-chained log with review is one implementation; a cloud audit
service with alerts is another.

### 8.2 NIST guidance

NIST SP 800-66 Rev. 2, "Implementing the HIPAA Security Rule: A Cybersecurity Resource Guide," was
published in February 2024 and supersedes Rev. 1 from 2008. It provides a risk assessment process aligned
with SP 800-30, risk management guidance, and per-standard key activities, descriptions, and sample
questions. It is non-prescriptive and is not legal advice.

The NIST Cybersecurity Framework 2.0 was released on 26 February 2024 as CSWP 29. It organizes outcomes
into six functions: Govern, Identify, Protect, Detect, Respond, and Recover. The addition of Govern
reflects the expectation that cybersecurity is enterprise risk, not only an IT function. It is a framework
for organizing a program, not a certification.

### 8.3 Other regimes and standards

- **21 CFR Part 11.** Electronic records in regulated submissions require secure, computer-generated,
  time-stamped audit trails, and changes must not obscure prior values. The audit trail components include
  date and time, user identity, and old and new values.
- **GDPR Article 32.** Requires appropriate technical and organizational measures, including
  pseudonymization and encryption, and a process for testing and evaluating effectiveness.
- **ISO/IEC 27001:2022.** An information security management system standard with certifiable controls.
- **HITRUST CSF.** A certifiable framework that maps HIPAA and other requirements.
- **HHS NPRM, 90 FR 898 (6 January 2025).** A proposed rule to strengthen the Security Rule, including
  redesignating audit controls and adding implementation specifications such as real-time monitoring and
  retention. As of October 2026 it remains a proposal; the binding text is the eCFR version. Do not cite
  the NPRM as current law.

### 8.4 What the standards do not specify

- Which cloud services to use.
- How to implement least privilege.
- The retention period for audit logs.
- The detection thresholds.
- The incident severity levels.
- Whether to use ABAC or RBAC.

Those are engineering and organizational decisions. A control set can be compliant on paper and ineffective
in practice if the logs are not reviewed, the alerts are ignored, or the access review is a rubber stamp.

## 9. Industry practice and case evidence

### 9.1 Audit controls in practice

HHS OCR's audit protocol asks whether the entity determines what activities to track, selects auditing
tools, develops an audit policy, and implements a review process. The emphasis on process is deliberate:
the regulation requires examination of activity, not only recording. A log that is written and never read
does not satisfy the intent.

### 9.2 Framework consolidation

NIST CSF 2.0 added the Govern function and expanded scope to all organizations, not only critical
infrastructure. NIST SP 800-66 Rev. 2 aligns HIPAA implementation with SP 800-30 risk assessment and
SP 800-53 controls, and moves the crosswalk to an online tool so it can be updated. The practical effect
is that a small clinical data team can map its controls to a recognized framework without inventing a
taxonomy.

### 9.3 Cloud shared responsibility

Cloud providers document a shared responsibility model: the provider secures the infrastructure, and the
customer secures identity, configuration, data, and access. Most cloud data breaches attributed to
providers are actually customer misconfigurations, such as public buckets or over-broad roles. The
engineering lesson is that migrating to a cloud does not transfer the controls that matter most.

### 9.4 What is inference

- That MFA prevents credential theft. It prevents many password-based attacks; it does not prevent
  session token theft or phishing-resistant bypass.
- That encryption at rest protects against insider access. It protects the storage medium; an authorized
  application can still read the plaintext.
- That a SIEM improves detection. It improves visibility; detection depends on rules, tuning, and
  response capacity.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Secret in repository | Credential-shaped string in a commit | Convenience | Credential exposure | Secret manager; history scan |
| Shared account | Audit cannot attribute actions | One login for a team | Repudiation; no least privilege | Unique identities |
| Over-broad role | One role for all pipelines | Ease of administration | Large blast radius | Task-scoped roles |
| Public bucket | Storage policy allows `*` principal | Misconfiguration | Data exposure | Guardrails; auto-remediation |
| Logs never reviewed | No alert owner | Volume without examination | Undetected incident | Detection rules; review |
| Logs editable | No hash chain or append-only store | Same account as data | Evidence destruction | Tamper-evident storage |
| Key stored with data | KMS not used | Simplicity | Encryption defeated | Key separation |
| No egress control | Personal cloud uploads allowed | Open network | Exfiltration | DLP; approved export path |
| Unpinned dependencies | Build resolves new versions | No lockfile | Supply-chain compromise | Lockfile; SBOM |
| No restore test | Backups exist, never restored | Untested assumption | Extended outage | Quarterly restore |
| Incident plan untested | Contact list stale | No exercise | Slow response | Tabletop |
| Access review rubber-stamped | 100% approved each quarter | No criteria | Entitlement creep | Usage data; reviewer accountability |

Two red flags deserve emphasis.

**Audit logs without an actor.** If every event records a shared service account, the log cannot answer
"who did this?" The identity model, not the logging library, is the problem.

**A risk register with no residual scores.** Inherent scores describe the world before controls; residual
scores describe the world after. A register that stops at inherent risk cannot support an acceptance
decision.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The identity provider is trustworthy | Authentication and authorization | Compromised IdP or misconfigured federation | Audit IdP configuration; test conditional access |
| Least privilege is enforceable | Blast radius reduction | Legacy roles with broad access | Role usage analysis |
| Keys are recoverable | Availability after disaster | KMS key deletion without backup | Key recovery drill |
| The audit log is complete | Investigation and compliance | Agents miss a data path | Coverage check against the asset register |
| Detection thresholds are tuned | Useful alerts | High false-positive rate | Measure precision and response time |
| The incident plan is current | Response | Staff turnover | Quarterly tabletop |
| Vendors meet obligations | Third-party risk | No BAA or audit right | Contract review; assessment |
| Backups are restorable | Recovery | Untested restore | Quarterly test |
| The threat model is current | Risk coverage | Architecture change | Annual review and change triggers |

The largest limit is that security is a property of the organization, not only of the code. A technically
sound control set fails if the on-call engineer cannot reach the key administrator, if the privacy officer
is not in the incident channel, or if the executive sponsor overrides a control for a deadline. The
organizational design is part of the engineering.

A second limit is that risk scoring is subjective. Two teams can score the same threat differently. The
value is in the structured conversation and the documented rationale, not in the precision of the number.

## 12. Exercises

### Exercise 1 — Threat model (application, 60 minutes)

Deliverable: a STRIDE table for a pipeline with at least 12 threats, each with asset, flow, and existing
control.

Acceptance criteria: all six categories present; at least three threats cross a trust boundary; each
threat names a specific asset.

### Exercise 2 — Risk scoring (calculation, 40 minutes)

Deliverable: a register of 10 threats with likelihood, impact, inherent, controls, and residual scores,
plus the aggregate reduction.

Acceptance criteria: scales are defined; arithmetic is correct; the aggregate is computed; at least two
risks remain above the acceptance threshold with a stated plan.

### Exercise 3 — Identity model (design, 45 minutes)

Deliverable: a table of human roles, workload identities, and break-glass access, with authentication
method, duration, and scope.

Acceptance criteria: no shared accounts; workload credentials are short-lived; break-glass is audited;
administrators use just-in-time elevation.

### Exercise 4 — ABAC policy (application, 50 minutes)

Deliverable: a policy with at least four attributes and six test requests, including two denials.

Acceptance criteria: deny by default; reason codes; tests cover training, device, purpose, and role; the
policy is deterministic.

### Exercise 5 — Audit log (application, 45 minutes)

Deliverable: a hash-chained log with verification and a tamper test, plus a retention estimate for a
stated event volume.

Acceptance criteria: tampering is detected at the first modified record; the chain covers all records;
the retention arithmetic is shown.

### Exercise 6 — Detection and egress (design, 45 minutes)

Deliverable: five detection rules with thresholds, owners, and runbooks, and an approved export process.

Acceptance criteria: each rule names the signal and the false-positive plan; exports require purpose and
approval; personal storage is blocked.

### Exercise 7 — Incident runbook (design, 60 minutes)

Deliverable: a runbook for a leaked service account token, with detection, containment, scoping,
notification, and recovery.

Acceptance criteria: roles are named; evidence preservation is explicit; notification triggers are stated;
the timeline is realistic.

### Exercise 8 — Control critique (critique, 40 minutes)

Given a control set that lists encryption, MFA, and a firewall, deliverable: a 400-word critique of what
is missing for a clinical data platform.

Acceptance criteria: covers identity, least privilege, audit, detection, egress, and response; no claim
that the existing controls are useless.

## 13. Self-check answers

**1. What is a trust boundary?**

A point where the trust level changes: between networks, tenants, identities, or data classifications. It
is where controls are placed and where threats are most likely to cross.

**2. What does STRIDE stand for?**

Spoofing, Tampering, Repudiation, Information disclosure, Denial of service, Elevation of privilege. It is
a threat enumeration framework, not a risk score.

**3. Why is least privilege not the same as a read-only role?**

A read-only role can still read every dataset, which is a large blast radius. Least privilege scopes both
the operation and the data to the task.

**4. Compute the residual reduction.**

Inherent total 143, residual total 55. Reduction = (143 − 55) / 143 = 88 / 143 = 0.6154, or 61.54%. The
residual is 38.46% of the inherent score.

**5. Why does key rotation not require re-encrypting all data?**

With envelope encryption, the data are encrypted with data keys, and the data keys are encrypted with a
key-encryption key. Rotating the key-encryption key rewraps the data keys; the data ciphertext is
unchanged.

**6. What does a hash chain prove?**

That a record has not been modified without breaking the chain, assuming the verifier has an authentic
copy of the latest hash. It does not prevent deletion of the whole log or modification by someone who can
recompute the entire chain.

**7. Compute the audit volume.**

180 users × 240 events/day = 43,200 events/day. × 365 = 15,768,000 events/year. At 1 KB each = 15.77 GB/
year. At six years = 94.6 GB.

**8. What is the difference between a BAA and a DUA?**

A BAA is required under HIPAA when a business associate handles PHI on behalf of a covered entity. A DUA
is a contract governing data use, often for research or a limited data set. They address different
obligations and can coexist.

**9. Why is encryption at rest insufficient?**

Because an authorized application or user can read the plaintext. Encryption at rest protects the storage
medium, not the access path. Access control, audit, and egress controls address the path.

**10. What makes an incident plan usable?**

Named roles, clear severity levels, tested contact paths, preserved evidence, and a practiced tabletop.
An untested plan is a document, not a capability.

## 14. Interview questions

**1. "How do you secure a clinical data pipeline?"**

Strong: threat model, identity, least privilege, encryption, secrets, audit, detection, egress, response.
Weak: "we use encryption and a firewall."

**2. "How do you scope a service account?"**

Strong: task-scoped data and operations, short-lived credentials, workload identity, and a measured blast
radius. Weak: "read-only."

**3. "What would you log?"**

Strong: actor, action, resource, timestamp, source, outcome, and correlation ID; never data values or
secrets. Weak: "everything."

**4. "How do you detect exfiltration?"**

Strong: egress controls, DLP, volume anomalies, approved export paths, and audit review. Weak: "block all
downloads."

**5. "How do you manage keys?"**

Strong: KMS, envelope encryption, separation from data, rotation, recovery, and access policy. Weak: "we
encrypt with AES-256."

**6. "What is the biggest risk in a research platform?"**

Strong: over-broad access and unmonitored exports, with a specific scenario and blast radius. Weak:
"hackers."

**7. "How do you handle a leaked credential?"**

Strong: revoke, rotate, scope via audit, contain, notify per policy, and prevent recurrence. Weak: "change
the password."

**8. "How do you review access?"**

Strong: usage data, role scope, reviewer accountability, and removal of unused entitlements; not a 100%
approval. Weak: "annually, managers approve."

**9. "What does HIPAA require for audit?"**

Strong: record and examine activity in systems with ePHI; no prescribed mechanism; the review process
matters. Weak: "keep logs for six years."

**10. "How do you balance security and research productivity?"**

Strong: curated self-service within guardrails, approved export paths, and measured friction; security
that blocks all work creates shadow IT. Weak: "security comes first," without a design.

## 15. Key terms

- **ABAC** — attribute-based access control.
- **Audit log** — record of access and actions.
- **BAA** — business associate agreement.
- **Blast radius** — reachable assets if compromised.
- **CVSS** — vulnerability severity score.
- **DLP** — data loss prevention.
- **DUA** — data use agreement.
- **Egress** — outbound data transfer.
- **ePHI** — electronic protected health information.
- **Envelope encryption** — data keys wrapped by a key-encryption key.
- **Incident** — confirmed or suspected security event.
- **KMS** — key management service.
- **Least privilege** — minimum permissions for the task.
- **MFA** — multi-factor authentication.
- **mTLS** — mutual TLS.
- **PHI** — protected health information.
- **RBAC** — role-based access control.
- **Residual risk** — risk after controls.
- **Runbook** — incident procedure.
- **SBOM** — software bill of materials.
- **Secret** — credential or key.
- **Segmentation** — network zoning.
- **Service account** — non-human identity.
- **SIEM** — security information and event management.
- **STRIDE** — threat categories.
- **Tabletop** — discussion-based exercise.
- **Tamper-evident** — modification is detectable.
- **Threat model** — structured threat analysis.
- **Trust boundary** — where trust level changes.
- **Workload identity** — identity assigned to a service.
- **Zero trust** — no implicit trust from location.

## 16. Sources

All URLs accessed 25 October 2026.

1. eCFR. "45 CFR 164.312, Technical safeguards."
   https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
2. HHS. "Summary of the HIPAA Security Rule."
   https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html
3. HHS. "Security Rule Guidance Material."
   https://www.hhs.gov/hipaa/for-professionals/security/guidance/index.html
4. NIST. "SP 800-66 Rev. 2, Implementing the HIPAA Security Rule."
   https://csrc.nist.gov/pubs/sp/800/66/r2/final
5. NIST. "CSWP 29, The NIST Cybersecurity Framework (CSF) 2.0."
   https://csrc.nist.gov/pubs/cswp/29/the-nist-cybersecurity-framework-csf-20/final
6. NIST. "NIST Releases Version 2.0 of Landmark Cybersecurity Framework."
   https://www.nist.gov/news-events/news/2024/02/nist-releases-version-20-landmark-cybersecurity-framework
7. Federal Register. "HIPAA Security Rule to Strengthen the Cybersecurity of Electronic Protected Health
   Information," 90 FR 898, 6 January 2025.
   https://www.federalregister.gov/documents/2025/01/06/2024-30983/hipaa-security-rule-to-strengthen-the-cybersecurity-of-electronic-protected-health-information
8. eCFR. "21 CFR Part 11." https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
9. FDA. "Part 11, Electronic Records; Electronic Signatures: Scope and Application."
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application
10. FDA. "Electronic Systems, Electronic Records, and Electronic Signatures in Clinical Investigations:
    Questions and Answers." https://www.fda.gov/media/166215/download
11. ISO/IEC. "ISO/IEC 27001:2022." https://www.iso.org/standard/27001
12. HITRUST. https://www.hitrustalliance.net/
13. GDPR. Regulation (EU) 2016/679. https://eur-lex.europa.eu/eli/reg/2016/679/oj
14. HHS OCR. "Audit Protocol." https://www.hhs.gov/hipaa/for-professionals/compliance-enforcement/audit/protocol-edited/index.html
15. CISA. "Zero Trust Maturity Model." https://www.cisa.gov/zero-trust-maturity-model

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_deidentification-engineering.md`](08_deidentification-engineering.md)
- Next topic: [`10_testing-and-observability.md`](10_testing-and-observability.md)
- Related sibling: [`07_data-versioning-and-lineage.md`](07_data-versioning-and-lineage.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — Lab 3.6 includes a post-incident review with root cause and prevention.
