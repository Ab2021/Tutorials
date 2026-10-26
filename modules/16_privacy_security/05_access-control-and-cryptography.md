# Access Control and Cryptography

> **Module 16 - Privacy, Security, and Ethics**
> Topic 05 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 04
> Version 2026.3

## 1. Why this topic exists

Access control and cryptography are the two controls that decide whether a privacy policy is real.
A policy says who may see what; access control enforces it at the moment of request. A policy says
data must be protected; cryptography protects it when the perimeter fails. Both are routinely
deployed in forms that look correct and fail in operation: roles that grant more than the job
requires, service accounts shared by a team, encryption keys stored beside the data, and "addressable"
safeguards that were never implemented because nobody documented why they were not reasonable.

The commercial consequence is direct. A health system that cannot show least privilege cannot pass
a security review; a vendor that cannot describe its key management cannot pass a procurement
questionnaire; an analytics team that grants database-wide read access to avoid ticket queues
creates the exact condition that turns a stolen credential into a breach. The Change Healthcare
incident is the clearest recent example: the entry point was a remote-access portal without
multi-factor authentication, and the attacker moved laterally through an environment where access
was broader than the task required.

Three failure patterns recur.

**Role explosion.** Every exception becomes a new role. Within two years there are 180 roles, most
of them used by one person, and nobody can explain what a role grants without reading its
permission list. The organisation has replaced a policy with an archaeology project. The symptom is
a role name that contains a person's name or a project code.

**Encryption without key management.** Data at rest is encrypted with a key that was created at
deployment, shared by three services, stored in the same cloud account as the ciphertext, and never
rotated. The encryption is real and the protection is nominal: anyone who compromises the account
gets both the data and the key. Encryption shifts the problem to key management; it does not remove
it.

**Addressable read as optional.** The HIPAA Security Rule marks some implementation specifications
as "addressable", which means the entity must implement them or document why an alternative is
reasonable and equivalent. Teams read "addressable" as "optional", skip encryption at rest, and
document nothing. The same misreading appears with automatic logoff and emergency access.

This topic teaches access control and cryptography as one design problem. The access model decides
who can reach a resource; the cryptographic design decides what happens when the access model is
bypassed. The two share a dependency: encryption keys are resources that need access control, and
access decisions are only as trustworthy as the identity that underpins them.

The distinction that runs through the topic is between authentication (who you are), authorisation
(what you may do), and accountability (what you did). Cryptography supports all three: signatures
and certificates support authentication, encrypted tokens support authorisation, and signed logs
support accountability. A deployment that implements one and not the others is incomplete.

A fourth pattern is **the encryption pilot that never became the default**. One sensitive dataset
is encrypted with a customer-managed key and a rotated secret, and the rest of the estate continues
with storage-level encryption and a shared key. The pilot proves the design works and does not
change the risk for the data that matters most. The control is a data-class-to-encryption matrix
with an owner per row, not a successful pilot.

The access review also changes what "compliance" means. A covered entity can satisfy 164.312(a)(1)
with a policy that describes access control and a system that enforces something narrower, and an
auditor may accept the policy. The operational test is a negative test: ask the system to deny an
access that the policy forbids, and see what happens. That test is the difference between a
documented control and a working one.

> **Note:** Access control is a policy problem expressed in code. The technical mechanism is
> secondary to the question of whether the policy is correct, reviewed, and testable.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** authentication, authorisation, and accountability, and map each to a control.
2. **Design** a role model from job functions, with object-level permissions and separation of
   duties, and **detect** role explosion.
3. **Choose** between role-based, attribute-based, and relationship-based access control for a
   stated decision, and justify the choice.
4. **Compute** least-privilege metrics: effective versus used permissions, orphaned accounts, and
   cross-boundary access.
5. **Design** an encryption scheme for data at rest, in transit, and in use, including envelope
   encryption and key hierarchy.
6. **Apply** key management practices: cryptoperiods, rotation, separation of duties, and
   revocation.
7. **Specify** logging and access review that produce evidence, not just events.
8. **Critique** a vendor's access-control or encryption claim and identify the missing operational
   control.

## 3. The core mental model

Access control is a decision pipeline. A subject requests a resource; a policy decision point
evaluates the request against attributes and policy; a policy enforcement point allows or denies;
and an audit trail records the outcome. Cryptography sits underneath, protecting the resource when
the pipeline is bypassed and protecting the credentials the pipeline depends on.

```text
  subject --- request ---> PEP --- decision ---> PDP
     |                      |                     |
     | attributes           | enforce             | policy + attributes
     v                      v                     v
  identity provider     resource             PAP (policy admin)
     |                      |
     |                      +--> audit log <---+
     |
  MFA / FIDO2 / certificate

  Under the pipeline:
  data at rest  -> envelope encryption -> KMS/HSM -> key policy
  data in transit -> TLS 1.3 -> certificate validation
  data in use   -> TEE / secure computation (Topic 03)
```

The pipeline model comes from NIST SP 800-162 and is useful because it separates the decision from
the enforcement. A common failure is a policy that exists in a wiki and an enforcement point that
implements something else. The model forces the question: where is the decision made, and where is
it enforced?

Three access models sit on top of the pipeline.

| Model | Decision input | Strength | Weakness | Use when |
|---|---|---|---|---|
| RBAC | Role assignments | Simple to audit; stable | Coarse; role explosion; no context | Job functions are stable and resources are few |
| ABAC | Subject, resource, action, environment attributes | Fine-grained; context-aware | Policy complexity; attribute quality | Access depends on relationship, time, location, or sensitivity |
| ReBAC | Relationships in a graph | Natural for care teams and hierarchies | Graph maintenance; query cost | Access follows the care relationship or organisational graph |

Most clinical systems need a hybrid: RBAC for the base grant, ABAC for the patient-relationship and
break-glass conditions, and ReBAC where the care team is the unit. The mixture is not a failure of
design; it is the normal shape of a real policy.

Cryptography has its own model: a key hierarchy. A root key protects key-encryption keys, which
protect data-encryption keys, which protect data. The hierarchy lets you rotate a data key without
re-encrypting everything, and it lets you revoke access by revoking a key-encryption key. The
hierarchy is only as strong as the root, which is why roots live in hardware security modules
(HSMs) or managed key services with strict policy.

```text
root key (HSM, offline, ceremony)
   |
   +-- key-encryption key (per tenant / per data class)
          |
          +-- data-encryption key (per object or per volume)
                 |
                 +-- ciphertext
```

The last model is defence in depth. Access control, encryption, logging, and network segmentation
each fail in different ways. A stolen credential defeats access control but not encryption if the
key is separate. A compromised key defeats encryption but not access logging if the access path is
instrumented. The controls are not redundant; they cover different failures.

The policy model also has a failure mode that is not technical: a policy that nobody can explain.
If a clinician asks "why can I see this patient" and the answer requires a database query, the
policy is not operable. The access model should produce a human-readable explanation for every
decision, stored with the audit record. That explanation is what a privacy officer uses in an
investigation and what a user sees in an access request.

A fourth property is that access control and encryption fail differently under the same attack. A
stolen session token defeats both the login and the role check, but it does not defeat a data key
held in a separate trust domain. A compromised key defeats encryption, but the audit record still
shows which identity used it. Designing for both means asking, for each threat path, which control
remains after the first one fails. A path with only one control is a single point of failure,
however strong that control appears.

The same logic applies to logging: a log that records the decision but not the reason cannot answer
why an access was allowed, and a log that records the content of the record becomes a second copy
of the data. The design target is the minimum information that supports an investigation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Authentication | Verifying that a subject is who it claims to be. | Confused with authorisation. | Identity protocol. |
| Authorisation | Deciding whether a subject may perform an action on a resource. | Treated as a role check only. | Policy decision. |
| Accountability | Being able to attribute actions to a subject. | Reduced to having logs. | Audit property. |
| Subject | The entity requesting access: user, service, device. | Assumed to be a person. | Identity. |
| Resource | The object being accessed: record, field, endpoint, key. | Assumed to be a table. | Object identifier. |
| Permission | An allowed action on a resource. | Treated as a role. | Action tuple. |
| Role | A named collection of permissions assigned to subjects. | Created per person. | RBAC construct. |
| RBAC | Role-based access control. | Assumed to be least privilege. | Access model. |
| ABAC | Attribute-based access control. | Assumed to require a policy engine. | Access model. |
| ReBAC | Relationship-based access control. | Confused with ABAC. | Access model. |
| PDP | Policy decision point: evaluates policy. | Co-located with the application without documentation. | Component. |
| PEP | Policy enforcement point: enforces the decision. | Missing or bypassable. | Component. |
| PIP | Policy information point: supplies attributes. | Stale attribute source. | Component. |
| PAP | Policy administration point: manages policy. | No change control. | Component. |
| Least privilege | Granting the minimum access needed for the task. | Interpreted as "fewer roles". | Principle. |
| Separation of duties | No single subject can complete a sensitive transaction alone. | Applied only to finance. | Control. |
| Break-glass | Emergency access that bypasses normal policy with logging and review. | Permanent admin account. | Procedure. |
| Orphaned account | An account with no active owner or no recent login. | Counted only after an audit. | Account state. |
| Role explosion | Excessive role proliferation, often one role per exception. | Treated as granularity. | Anti-pattern. |
| Service account | A non-human identity used by a system. | Shared by humans. | Identity type. |
| MFA | Multi-factor authentication. | SMS treated as phishing-resistant. | Control. |
| FIDO2 | Phishing-resistant authentication standard using public-key credentials. | Assumed to be universal. | Standard. |
| Session fixation | An attack that reuses a session identifier. | Ignored in web apps. | Attack. |
| Token scope | The permissions carried by an access token. | Over-broad scopes. | OAuth construct. |
| Encryption at rest | Protecting stored data with encryption. | Assumed to protect against a compromised account. | Control. |
| Encryption in transit | Protecting data on the network with TLS. | TLS version not checked. | Control. |
| Encryption in use | Protecting data during computation (TEE, MPC, HE). | Assumed to be mature for all workloads. | Control. |
| Envelope encryption | Encrypting data with a data key, and the data key with a key-encryption key. | Keys stored with ciphertext. | Scheme. |
| KMS | Key management service. | Assumed to be an HSM. | Service. |
| HSM | Hardware security module. | Used without ceremony or backup. | Hardware. |
| Cryptoperiod | The time a key is authorised for use. | Set to "never". | Time. |
| Key rotation | Replacing a key and re-encrypting or re-wrapping. | Rotating without re-wrapping. | Procedure. |
| Key ceremony | A documented, witnessed process for generating root keys. | Done ad hoc. | Procedure. |
| AES-GCM | Authenticated encryption with associated data. | Used with a reused nonce. | Algorithm. |
| TLS 1.3 | Transport Layer Security version 1.3. | Assumed on all internal links. | Protocol. |
| Revocation | Invalidating a credential or key before expiry. | Missing from the runbook. | Procedure. |

## 5. The mechanics, step by step

The procedure below designs access control and encryption for a clinical data platform. It assumes
that the threat model from Topic 04 exists, because the access design should address the paths that
model identified. Each step names its input, output, and rationale.

### Step 1. Inventory resources and classify data

**Input:** Data catalogue, system inventory, threat model.
**Output:** A resource inventory with a data class per resource: public, internal, confidential,
restricted (PHI, genetic, mental health, substance use).
**Rationale:** Access policy is only as good as the resource model. A table-level inventory is too
coarse for clinical data, because a patient record contains fields with different sensitivity and
different legal regimes. The inventory should reach field level for restricted data: notes,
genetic results, HIV status, substance use records, and reproductive health data.

The classification drives both the access rule and the encryption rule. A field that is restricted
needs object-level authorisation, field-level encryption or tokenization, and audit on read. A
field that is internal may need only role-level access.

### Step 2. Derive roles from job functions

**Input:** Job descriptions, observed access patterns, workflow interviews.
**Output:** A role catalogue with a permission set per role, and a mapping from jobs to roles.
**Rationale:** Roles should describe what a person does, not who they are or which project they
work on. Role mining starts from observed activity: which resources does a clinical pharmacist
actually open in a week? The answer defines the permission set. Starting from the org chart
produces roles that mirror hierarchy and grant too much.

A useful test is whether a new hire can be assigned a role on day one based on their job title. If
the answer requires a conversation about exceptions, the role model is incomplete, and the
exception should become an attribute rather than a new role.

Branch point: if a job function genuinely needs a unique permission combination, prefer an
attribute or a small additional role over a new role per person. Role explosion is the default
failure of RBAC.

### Step 3. Define permissions at the right granularity

**Input:** Resource inventory, role catalogue.
**Output:** A permission matrix: role x resource x action.
**Rationale:** A permission should be the smallest action that the policy needs to distinguish.
"Read patient" is too coarse if the policy distinguishes reading demographics from reading notes.
"Read note" is too coarse if the policy distinguishes the author's own notes from others'. The
right granularity is the level at which the policy actually differs; finer granularity adds
administration cost without changing a decision.

The matrix should be small enough to review. If it has thousands of cells, group by resource class
and document the exceptions separately.

### Step 4. Assign subjects and handle exceptions with attributes

**Input:** Identity inventory, role catalogue, HR data.
**Output:** Role assignments with an expiry and a justification, plus attribute rules.
**Rationale:** Every assignment should have a reason and a review date. Standing access is the
norm in most organisations and is the main source of over-privilege. Time-bound assignment,
project-based assignment, and just-in-time elevation reduce the standing surface without slowing
work if the elevation path is fast.

Attributes handle the cases that roles cannot: the patient relationship ("this clinician is on the
care team"), the site ("this analyst may query site B"), the on-call status, and the time window.
Attribute sources must be authoritative and current, because a stale attribute is an access bug.

### Step 5. Write the policy and place enforcement

**Input:** Role and attribute model, resource inventory.
**Output:** A policy specification and an enforcement architecture with a PDP and a PEP.
**Rationale:** The policy should be written in a form that can be tested: subject, resource,
action, condition, effect. Deny by default. The enforcement point must be on the path of every
request; an enforcement point that can be bypassed by calling the database directly is not an
enforcement point.

| Access model | Decision inputs | Best for | Cost | Failure mode |
|---|---|---|---|---|
| RBAC | Role assignments | Stable job functions, coarse resources | Low | Role explosion; over-broad roles |
| ABAC | Subject, resource, action, environment | Patient relationship, site, time, sensitivity | Medium to high | Stale attributes; policy complexity |
| ReBAC | Relationship graph | Care teams, organisational hierarchies | Medium | Graph staleness; query cost |
| Hybrid RBAC + ABAC | Role base plus relationship and context | Most clinical systems | Medium | Two sources of truth that diverge |

The hybrid is usually the right answer. RBAC answers "may this person ever do this"; ABAC answers
"may this person do it for this patient, now". Keeping the two layers separate makes each testable.

Branch point: if the application cannot enforce object-level authorisation, add an API gateway or
a data-access layer that can. Retrofitting enforcement into a UI is not a control.

### Step 6. Implement separation of duties and break-glass

**Input:** Sensitive transactions, role model.
**Output:** A separation-of-duties matrix and a break-glass procedure.
**Rationale:** Separation of duties prevents one person from completing a sensitive transaction
alone: creating a vendor and approving it, prescribing and overriding a dispense, administering
users and editing the audit log. The matrix names the pairs that must not coexist in one identity.
The control is enforced at assignment time, not only at runtime.

Break-glass is the exception path for emergencies. It should be a distinct role with a short
session, a reason code, an automatic alert, and a retrospective review. The failure mode is a
permanent admin account labelled "emergency"; the success mode is a process that a clinician can
use in seconds and that the privacy office reviews the next day.

### Step 7. Design authentication and sessions

**Input:** User population, risk profile, clinical workflow.
**Output:** An authentication policy per population: factor type, session length, re-authentication
rules, device posture.
**Rationale:** Access control is only as strong as the identity behind it. Phishing-resistant MFA
(FIDO2 security keys or platform authenticators) is materially stronger than SMS or push approval
against real attacks. Shared workstations need fast lock and re-authentication without breaking
workflow; a control that adds thirty seconds to every patient lookup will be circumvented.

Session management matters as much as the login. Short-lived tokens, refresh-token rotation, and
binding tokens to the client reduce the value of a stolen token. A token with a broad scope and a
long lifetime is a credential, whether or not it is called one.

### Step 8. Select encryption for data at rest and in transit

**Input:** Data classes, architecture, performance requirements.
**Output:** An encryption specification per data class and per channel.
**Rationale:** Encryption at rest protects against media theft and some account compromises; it
does not protect against an attacker who can use the application's own credentials. Encryption in
transit protects against network interception; it does not protect against a compromised endpoint.
The specification should say which threat each control addresses.

| Approach | Protects against | Key location | Cost | Use when |
|---|---|---|---|---|
| Storage-level encryption (SSE-KMS) | Media theft, snapshot exposure | Cloud KMS | Low | Default for all stores |
| Client-side envelope encryption | Cloud account compromise | Separate KMS or HSM | Medium | Restricted data, multi-tenant |
| Field-level encryption | Application compromise, broad queries | Application key service | High | Identifiers, genetic, mental health |
| Tokenization | Application compromise, linkage control | Token vault | Medium | Research datasets, joins |
| Format-preserving encryption | Legacy schemas needing fixed width | KMS | Medium | Constrained integrations |
| No encryption | Nothing | Not applicable | None | Never for restricted data |

Two rules apply to all rows. First, authenticated encryption (AES-GCM or ChaCha20-Poly1305) is the
default; unauthenticated modes allow tampering. Second, nonce reuse in AES-GCM is catastrophic,
so the implementation must guarantee uniqueness per key.

### Step 9. Design the key hierarchy and lifecycle

**Input:** Encryption specification, organisational roles.
**Output:** A key hierarchy with owners, cryptoperiods, rotation, backup, and revocation procedures.
**Rationale:** Keys are the real asset. A key hierarchy separates the root (in an HSM, under
ceremony) from key-encryption keys (per tenant or data class) and data-encryption keys (per object
or volume). Rotation replaces a data key and re-wraps it with the current key-encryption key;
re-encryption is only needed when the data key itself is compromised.

NIST SP 800-57 Part 1 Rev. 5 provides cryptoperiod guidance and usage limits, including a limit of
2^32 invocations for AES-GCM with a given key. The operational rules are: no key without an owner,
no key without a rotation schedule, no key stored with the ciphertext, and no root key without a
tested recovery procedure.

Branch point: if the organisation cannot operate an HSM, use a managed KMS with a documented trust
assessment. "We use a KMS" is not sufficient if the KMS policy allows the same administrators to
read the data and the keys.

### Step 10. Instrument audit and monitoring

**Input:** Policy, enforcement points, data classes.
**Output:** An audit specification: what is logged, where, for how long, who can read it, and what
triggers an alert.
**Rationale:** Accountability requires that actions be attributable and that the record be
tamper-evident. Log the subject, the resource, the action, the decision, the reason code, the time,
and the source. Do not log the content of the record. Protect the log with append-only storage and
separate access, because an attacker who can edit the log can hide the access.

Alerts should cover the behaviours that matter: bulk export, access outside the care relationship,
break-glass use, admin role changes, and access to VIP records. A log with no alerting is a
forensic tool, not a detective control.

### Step 11. Review, recertify, and test

**Input:** Access inventory, audit data, change log.
**Output:** A quarterly recertification, a negative-test suite, and an annual key-compromise drill.
**Rationale:** Access drifts. People change roles, projects end, contractors leave, and attributes
go stale. Recertification asks managers to confirm each assignment, with the last-used date
visible. Negative tests ask the system to deny: a clinician requesting another clinician's patient,
an analyst requesting notes, a token with the wrong scope. The key-compromise drill tests whether
the organisation can revoke a key and re-encrypt within the stated recovery time.

### Authentication in the clinical workflow

Authentication policy is where security meets usability, and usability wins unless the design is
deliberate. A shared workstation in an emergency department cannot require a full login for every
patient lookup; a clinician who is locked out during a resuscitation will find a workaround, and
the workaround will be worse than the control. The practical design uses:

- A fast unlock (badge tap or platform biometric) with a short idle timeout, rather than a full
  credential entry on every task.
- Re-authentication for sensitive actions (prescribing controlled substances, exporting, changing
  access), not for every read.
- Phishing-resistant MFA for remote access and for administrators, where the session is long-lived
  and the risk is highest.
- Session binding to the device and to the network context, so a stolen token cannot be replayed
  from another location.
- A visible session indicator, so a clinician knows which identity is active on a shared device.

The common failure is "MFA everywhere" applied uniformly, which produces push fatigue and shared
badges. The better pattern is risk-based step-up: low-risk reads use the fast path, high-risk
actions require a stronger factor. The policy should be written per action, not per user.

### Encryption in transit and in use

Encryption in transit is usually described as "TLS", which is not a specification. The minimum
specification names the version (TLS 1.3), the cipher suites, certificate validation, and the
internal segments that are covered. Internal traffic is often unencrypted on the assumption that
the network is trusted; the assumption fails in flat networks and after lateral movement.

| Channel | Minimum specification | Common gap |
|---|---|---|
| Client to API | TLS 1.3, valid certificate, HSTS | TLS 1.0 still enabled on a legacy endpoint |
| Service to service | mTLS or TLS with service identity | Plaintext inside the cluster |
| Service to database | TLS with certificate validation | `sslmode=disable` in a connection string |
| Service to object store | TLS plus signed requests | Long-lived access keys |
| Vendor callback | TLS plus request signing | Static API key in a configuration file |
| Backup and replication | TLS plus encryption at rest | Unencrypted snapshot export |

Encryption in use is the hardest layer. A trusted execution environment protects data during
computation, but it depends on attestation, on the absence of side channels, and on the host not
being able to observe memory. Secure multi-party computation and homomorphic encryption provide
stronger guarantees for specific computations at a significant performance cost. For most clinical
platforms, encryption in use is a targeted control for the most sensitive computations (Topic 03),
not a default for every query.

### The audit record

An audit record should answer who did what to which resource, when, from where, and why. A schema
that supports those questions has seven fields and no content:

| Field | Example | Why it matters |
|---|---|---|
| Subject | `user:U1234` or `service:etl-01` | Attribution |
| Resource | `patient:88421` or `note:55120` | Scope of access |
| Action | `read`, `export`, `update`, `delete` | Nature of access |
| Decision | `allow`, `deny` | Policy outcome |
| Reason | `treatment`, `break_glass`, `research` | Purpose |
| Time | ISO 8601 with time zone | Sequence and correlation |
| Source | IP, device, session | Context and anomaly detection |

The record must not contain the content of the note or the values of the fields read, because the
audit log would become a second copy of the data with weaker protection. It should be append-only,
stored separately from the application, and readable only by the audit function. Retention should
match the legal requirement and the investigation window, and the log should be tested by asking an
investigator to reconstruct a specific access event.

### Access review procedure

A review that changes behaviour has five properties. It is scoped to a population (all users of a
platform, all holders of a privileged permission), not to a whole organisation. It presents, per
assignment, the permission, the last-used date, the grant date, and the consequence of removal. It
has a deadline and an escalation path for non-response, with removal as the default. It records the
decision and the reviewer. And it measures the number of permissions removed, not the completion
rate.

The cadence should follow risk: privileged and cross-boundary access quarterly, standard clinical
access annually, and service accounts quarterly with a named owner. A review that runs annually for
admin access leaves eleven months in which an orphaned admin account can be used.

### Key compromise response

A key compromise runbook has six steps: detect and confirm; contain by disabling the key or the
identity that can use it; assess what was encrypted or accessed with the key; re-key and re-encrypt
or re-wrap; notify according to the legal and contractual requirements; and review how the key was
exposed. The step that is usually missing is the third: without an inventory that maps keys to
objects and objects to data classes, the organisation cannot say what was exposed and therefore
cannot scope the notification.

The runbook should be tested annually with a tabletop exercise and, where feasible, a technical
drill in a non-production environment. The recovery objective (for example, 24 hours) should be
stated and measured. An untested runbook is a document, not a control.

### Edge cases

**Small cells.** Access control does not protect an aggregate that reveals a small cell. A user
with legitimate access to a dashboard can infer a diagnosis from a count of one. The control is
output review and minimum cell size, applied at the query layer, not only at the row level.

**Quasi-identifiers.** A user who can query demographics and geography can re-identify a
de-identified table. The access policy should treat the combination as restricted even when each
field is classified as internal, and the query layer should enforce a minimum group size.

**Re-identification through linkage.** A user with access to two datasets can join them. The
control is to restrict joins across data classes, to require approval for linkage, and to log the
query. A database permission that allows both reads is a linkage permission whether or not it is
called one.

**Free-text PHI.** Notes need separate permission from structured data. A role that can read the
patient record should not automatically read notes; the note permission should be granted to the
care team and audited on access.

**Metadata leakage.** Logs, error messages, file names, and export headers can reveal identity.
The audit specification should include metadata flows, and the export path should strip properties.

**Consent withdrawal and access revocation.** When a patient withdraws or a user leaves, access
must be revoked across every system, including caches, tokens, and downstream copies. The
revocation procedure should be tested, not assumed. A token that remains valid after termination is
an access control failure even if the account is disabled.

**Minors and deceased patients.** Proxy access for parents and representatives needs age-aware
rules, and access to deceased records needs a purpose check. A parent portal that shows
adolescent-sensitive encounters is an authorisation bug, not a UI issue.

**Cross-border access and key residency.** If the key is in one jurisdiction and the data in
another, the access path crosses a border. Key residency, support access from another country, and
subprocessor access are part of the access design, not only the contract.

**Break-glass abuse.** A break-glass account used daily is not an emergency control. Monitor the
rate, require a reason, and review every use. The target is a small number of genuine emergencies,
not zero, because a control that blocks emergency care will be bypassed.

## 6. Worked example

### Setting

Meridian Health operates an analytics platform across three hospitals with 540,000 patients. The
platform has 2,400 users, 38 roles, and 412 permissions. A quarterly access review is due, and the
security team has been asked whether the platform meets least privilege. The review uses a 10
percent sample of users (240 accounts) plus a full inventory of admin accounts.

### Step 1: least-privilege measurement

The 240-account sample yields an average of 84 effective permissions per user, after resolving
role unions. The 90-day audit log shows an average of 22 permissions actually used. The unused
share is therefore:

(84 - 22) / 84 = 62 / 84 = 73.8 percent.

Extrapolated to the platform, effective permission assignments are 2,400 x 84 = 201,600, of which
2,400 x 22 = 52,800 are used, leaving 148,800 unused assignments. The extrapolation assumes the
sample is representative; the review should stratify by role family before relying on it for a
budget decision.

The arithmetic matters because it converts a principle into a work item. If the platform removes
half of the unused assignments, it removes 74,400 permission grants, and every one of them is a
path that a compromised credential would otherwise have. The number is also a warning: a 73.8
percent unused rate means the role model is not describing the work.

### Step 2: orphaned and terminated accounts

The full inventory finds 144 accounts (6.0 percent of 2,400) with no login in 90 days or a
termination flag. Of those, 14 hold the `admin_users` permission. One has a last login 412 days
ago. At an assumed annual compromise probability of 0.02 per admin account, the expected number of
compromised admin accounts per year is 14 x 0.02 = 0.28. That number looks small until it is
multiplied by the estate: a single admin compromise reaches all 540,000 patient records.

The diagnostic is not the count of orphaned accounts; it is the count of orphaned accounts with
privilege. An organisation can tolerate dormant read-only accounts longer than dormant admin
accounts, and the review should rank by privilege, not by age.

### Step 3: cross-site access

The `read_all_sites` permission is held by 912 users (38.0 percent). The access policy says that
only 240 users (10.0 percent) should be able to query all three sites. The excess is 672 users, or
2.8 times the intended population.

Each of the 912 users can see three sites instead of one, so the reachable patient population per
credential is 540,000 instead of 180,000, a 3x multiplier. At an assumed annual credential-phishing
success of 0.03 per user, the expected number of compromised accounts per year is 912 x 0.03 =
27.4, each with the 3x reach. With the intended 240 users, the expected number is 7.2. The
difference is 20.2 compromised accounts per year with three times the reach, which is the
quantified cost of a role that was convenient.

### Diagnostic 1: orphaned admin accounts

Fourteen admin accounts with no active owner is the first diagnostic. The mechanism is simple:
deprovisioning removes the HR record but not the role assignment, because the role was granted by
a project team that no longer exists. The consequence is an unattributable admin path that no
current manager owns. The control is an automated joiner-mover-leaver process that revokes roles on
the HR event, plus a privileged access review every quarter.

The arithmetic to report is not "14 accounts" but "14 admin accounts, 0.28 expected compromises
per year, each reaching 540,000 records". That framing gets the remediation funded.

### Diagnostic 2: cross-site over-access

The second diagnostic is the 672-user excess on `read_all_sites`. The mechanism is a role that
grants a site-wide permission to solve a workflow problem for a small group (on-call radiology),
and then gets assigned to everyone in the job family. The consequence is a 3x reach multiplier for
38 percent of users.

The two diagnostics are independent. Fixing orphaned accounts does nothing for cross-site reach,
and fixing the role does nothing for dormant admins. A remediation plan that addresses only one
will leave the other path open.

### Vary one assumption: remove the broad role outright

The simplest remediation is to remove `read_all_sites` from the clinician and analyst role families
and grant it only to the it_admin and radiologist roles. The arithmetic after removal:

- Cross-site readers fall to 216 users (9.0 percent), close to the 240 target.
- The excess falls from 672 to 0 against the target, and the reach multiplier drops for 696 users.
- Expected compromised accounts with 3x reach fall from 27.4 to 6.5 per year.

The conclusion changes from "over-access requiring an ABAC project" to "acceptable after a role
change". But the change breaks a workflow: 36 on-call radiology reads per day cross sites, and the
on-call radiologists who cover another site lose access. The service desk starts granting the role
back as an exception, and within a quarter the role is broad again.

This is the point of the exercise. The role-only fix is arithmetically attractive and
operationally unstable. The stable fix is an attribute: on-call status plus the site being covered,
evaluated at request time, with an expiry at the end of the shift. The ABAC rule keeps cross-site
access at 2 percent of users while allowing the 36 daily reads, at a cost of about 40 milliseconds
per authorisation check.

### The encryption side of the same platform

The platform stores 4 TB across 12,000 objects, encrypted with AES-256-GCM under a single
data-encryption key created at launch four years ago. The key is stored in the same cloud account
as the objects, and the same three service identities can read both.

Three arithmetic checks:

- Usage limit: at 50,000 encryptions per day, the NIST SP 800-57 guidance limit of 2^32
  invocations for a GCM key would be reached after 4,294,967,296 / 50,000 = 85,899 days, about 235
  years. The usage limit is not the binding constraint; the four-year age and the shared access
  are.
- Rotation cost: re-encrypting 4 TB at 200 MB/s takes 4,000,000 MB / 200 MB/s = 20,000 seconds,
  or 5.6 hours, plus verification. That is feasible in a maintenance window, which removes the
  usual excuse for not rotating.
- Blast radius: because the key is in the same account and readable by the same identities as the
  data, a single compromised service identity yields both. The envelope separation would require an
  attacker to compromise two identities in two trust domains.

The recommendation is a key hierarchy: a root key in an HSM under ceremony, a key-encryption key
per tenant, and data-encryption keys rotated annually with re-wrapping. The recovery objective is
24 hours for a suspected key compromise, which the 5.6-hour re-encryption supports.

### Summary

| Finding | Measure | Consequence | Control |
|---|---|---|---|
| Unused permissions | 73.8% of effective grants | 148,800 excess grants | Recertification; role redesign |
| Orphaned admin accounts | 14 accounts | 0.28 expected admin compromises/year | Joiner-mover-leaver automation |
| Cross-site access | 912 users vs 240 target | 3x reach for 38% of users | ABAC on-call attribute |
| Shared data key | 4 years, same account | Single compromise yields data and key | Key hierarchy; annual rotation |

The table is the review deliverable. Each row has a measure, a consequence, and a control, and each
control has an owner in the remediation plan.

### Sampling and extrapolation

The 73.8 percent unused-permission figure comes from 240 accounts. A 95 percent confidence
interval for a proportion with n = 240 and p = 0.738 is:

standard error = sqrt(p(1 - p) / n) = sqrt(0.738 x 0.262 / 240) = sqrt(0.000806) = 0.0284.
95 percent interval = 0.738 +/- 1.96 x 0.0284 = 0.738 +/- 0.056, or 68.2 to 79.4 percent.

The interval is wide enough that the remediation budget should use the lower bound (68.2 percent)
rather than the point estimate. The extrapolation also assumes the sample is representative; a
sample stratified by role family would reduce the variance and catch a role family where the unused
share is much higher or lower. The review should report the interval, not only the headline.

The same logic applies to the cross-site count. The 912 users are a full inventory, not a sample,
so there is no sampling error; the uncertainty is in the policy target of 240, which is a
judgement about how many users genuinely need cross-site access.

### The insider path

The access review also has to consider the user who is authorised and misuses access. In the
platform, 40 analysts can query the lake, and three can export in bulk. The expected loss from this
path is not modelled by the phishing probability; it is modelled by the probability that one of the
three exports data for an improper purpose. At an assumed 2 percent annual probability, the
expected number of improper exports is 0.02 x 3 = 0.06 per year. Each export can carry 12,000
records, so the expected records exposed through this path is 0.06 x 12,000 = 720 per year.

The control set is different from the external path: least privilege (fewer bulk exporters),
per-record audit (so an export is attributable), behavioural monitoring (volume and timing), and a
policy that requires approval for bulk export. None of these is encryption. This is the point of
separating the paths: the controls that stop an external attacker do not stop an authorised user,
and a programme that measures only external risk will under-invest in the insider controls.

## 7. Code, query, or calculation

The script below implements the least-privilege analysis on a 12-user synthetic population: role
union, unused permissions, orphaned accounts, cross-site readers, separation-of-duties conflicts,
and an ABAC site check. It uses only the standard library.

```python
"""RBAC/ABAC least-privilege analysis on a synthetic user population."""
from collections import Counter

PERMISSIONS = {
    "read_patient", "write_patient", "read_notes", "export_bulk",
    "admin_users", "view_finance", "prescribe", "dispense_override",
    "read_all_sites", "approve_vendor", "create_vendor", "audit_log",
}

ROLES = {
    "clinician": {"read_patient", "write_patient", "read_notes", "prescribe",
                  "export_bulk", "view_finance"},
    "nurse": {"read_patient", "write_patient", "read_notes", "view_finance"},
    "site_admin": {"read_patient", "admin_users", "audit_log", "export_bulk"},
    "analyst": {"read_patient", "export_bulk", "read_notes",
                "audit_log", "view_finance"},
    "researcher": {"read_patient", "export_bulk", "read_notes", "audit_log"},
    "billing": {"view_finance", "read_patient", "export_bulk"},
    "it_admin": {"admin_users", "audit_log", "read_all_sites", "read_patient",
                 "export_bulk"},
    "radiologist": {"read_patient", "read_notes", "read_all_sites", "export_bulk"},
    "on_call": {"read_patient", "read_all_sites", "read_notes"},
    "vendor_manager": {"create_vendor", "approve_vendor", "view_finance",
                       "read_patient"},
}

USERS = [
    {"id": "U001", "roles": {"clinician"}, "site": "A", "last_login": 2,
     "terminated": False, "used": {"read_patient", "write_patient", "read_notes", "prescribe"}},
    {"id": "U002", "roles": {"clinician", "researcher"}, "site": "A", "last_login": 9,
     "terminated": False, "used": {"read_patient", "read_notes", "prescribe", "export_bulk"}},
    {"id": "U003", "roles": {"nurse"}, "site": "B", "last_login": 1,
     "terminated": False, "used": {"read_patient", "write_patient"}},
    {"id": "U004", "roles": {"site_admin"}, "site": "B", "last_login": 4,
     "terminated": False, "used": {"read_patient", "admin_users", "audit_log"}},
    {"id": "U005", "roles": {"analyst"}, "site": "C", "last_login": 15,
     "terminated": False, "used": {"read_patient", "export_bulk"}},
    {"id": "U006", "roles": {"it_admin"}, "site": "A", "last_login": 412,
     "terminated": True, "used": set()},
    {"id": "U007", "roles": {"it_admin"}, "site": "B", "last_login": 380,
     "terminated": True, "used": set()},
    {"id": "U008", "roles": {"radiologist", "on_call"}, "site": "C", "last_login": 0,
     "terminated": False, "used": {"read_patient", "read_notes", "read_all_sites"}},
    {"id": "U009", "roles": {"billing"}, "site": "A", "last_login": 30,
     "terminated": False, "used": {"view_finance", "read_patient"}},
    {"id": "U010", "roles": {"vendor_manager"}, "site": "B", "last_login": 22,
     "terminated": False, "used": {"create_vendor", "approve_vendor"}},
    {"id": "U011", "roles": {"clinician"}, "site": "C", "last_login": 120,
     "terminated": False, "used": {"read_patient", "prescribe"}},
    {"id": "U012", "roles": {"researcher", "analyst"}, "site": "A", "last_login": 7,
     "terminated": False, "used": {"read_patient", "export_bulk", "read_notes"}},
]

SOD_CONFLICTS = [
    ({"prescribe"}, {"dispense_override"}),
    ({"create_vendor"}, {"approve_vendor"}),
    ({"admin_users"}, {"audit_log"}),
]


def effective(user):
    perms = set()
    for role in user["roles"]:
        perms |= ROLES.get(role, set())
    return perms


def abac_allows(user, patient_site, on_call=False):
    if "read_all_sites" in effective(user):
        return True
    return user["site"] == patient_site or on_call


if __name__ == "__main__":
    n = len(USERS)
    eff = {u["id"]: effective(u) for u in USERS}
    avg_eff = sum(len(v) for v in eff.values()) / n
    avg_used = sum(len(u["used"]) for u in USERS) / n
    print(f"users={n}, roles={len(ROLES)}, permissions={len(PERMISSIONS)}")
    print(f"average effective permissions={avg_eff:.1f}, average used={avg_used:.1f}, "
          f"unused share={(avg_eff - avg_used) / avg_eff:.1%}")

    orphans = [u for u in USERS if u["last_login"] > 90 or u["terminated"]]
    admin_orphans = [u for u in orphans if "admin_users" in eff[u["id"]]]
    print(f"orphaned or terminated accounts={len(orphans)}/{n} "
          f"({len(orphans) / n:.1%}), with admin={len(admin_orphans)}")

    cross = [u for u in USERS if "read_all_sites" in eff[u["id"]]]
    print(f"cross-site readers={len(cross)}/{n} ({len(cross) / n:.1%})")

    conflicts = []
    for u in USERS:
        for a, b in SOD_CONFLICTS:
            if a <= eff[u["id"]] and b <= eff[u["id"]]:
                conflicts.append((u["id"], tuple(sorted(a)), tuple(sorted(b))))
    print(f"separation-of-duties conflicts={len(conflicts)} {conflicts}")

    allowed = sum(1 for u in USERS if abac_allows(u, "B"))
    print(f"ABAC site check: {allowed}/{n} users may read site B ({allowed / n:.1%})")

    roles_per_user = Counter(len(u["roles"]) for u in USERS)
    print(f"roles per user distribution={dict(sorted(roles_per_user.items()))}")
```

Expected output, verified on Python 3.11:

```text
users=12, roles=10, permissions=12
average effective permissions=4.8, average used=2.2, unused share=53.4%
orphaned or terminated accounts=3/12 (25.0%), with admin=2
cross-site readers=3/12 (25.0%)
separation-of-duties conflicts=4 [('U004', ('admin_users',), ('audit_log',)), ('U006', ('admin_users',), ('audit_log',)), ('U007', ('admin_users',), ('audit_log',)), ('U010', ('create_vendor',), ('approve_vendor',))]
ABAC site check: 6/12 users may read site B (50.0%)
roles per user distribution={1: 9, 2: 3}
```

The sample output shows the same patterns as the 2,400-user review at smaller scale: a high unused
share, orphaned admin accounts, separation-of-duties conflicts created by combining roles, and a
cross-site permission that reaches a quarter of users. The numbers differ from Section 6 because
the sample is 12 users and the population is different; the logic is the same.

Two adaptations are needed for production. First, `used` permissions should come from the audit
log, not from a hand-built set. Second, the ABAC function should evaluate a real attribute store
with an expiry, not a static site field, or the attribute becomes as stale as the role it replaces.


## 8. Data, standards, and regulatory context

Access control and cryptography appear in law as required safeguards and in standards as
engineering practice. The legal text says what must be achieved; the standards say how it is
usually demonstrated.

### HIPAA Security Rule

The HIPAA Security Rule (45 CFR Part 164, Subpart C) is the binding requirement for covered
entities and business associates. The provisions most relevant here are:

| Provision | Status | Requirement |
|---|---|---|
| 164.312(a)(1) | Required | Access control: technical policies and procedures that allow access only to authorised persons or software |
| 164.312(a)(2)(i) | Required | Unique user identification: assign a unique name or number for identifying and tracking user identity |
| 164.312(a)(2)(ii) | Required | Emergency access procedure: obtain necessary ePHI during an emergency |
| 164.312(a)(2)(iii) | Addressable | Automatic logoff |
| 164.312(a)(2)(iv) | Addressable | Encryption and decryption of ePHI |
| 164.312(b) | Required | Audit controls: record and examine activity in systems that contain ePHI |
| 164.312(d) | Required | Person or entity authentication |
| 164.312(e)(1) | Required | Transmission security: guard against unauthorised access to ePHI in transit |
| 164.312(e)(2)(ii) | Addressable | Encryption of ePHI in transit |

"Addressable" does not mean optional. Under 45 CFR 164.306(d), a covered entity must implement the
specification if reasonable and appropriate, or document why it is not, and implement an equivalent
measure if reasonable and appropriate. The documentation is the control. A missing encryption
implementation with no documented rationale is a finding.

NIST SP 800-66 Rev. 2 maps each Security Rule provision to implementation guidance and is the
reference an OCR reviewer is likely to use. It is guidance, not regulation.

### NIST access control and key management guidance

- NIST SP 800-162, *Guide to Attribute Based Access Control (ABAC) Definition and Considerations*
  (January 2014, updated August 2019), defines the PDP/PEP/PIP/PAP architecture used in Section 3
  and the attributes that drive ABAC decisions.
- NIST SP 800-207, *Zero Trust Architecture* (2020), argues that access decisions should be made
  per request based on identity, device, and context rather than on network location. It is
  guidance; "zero trust" is a design philosophy, not a product.
- NIST SP 800-57 Part 1 Rev. 5, *Recommendation for Key Management: Part 1 - General* (May 2020),
  provides cryptoperiod guidance, key hierarchy models, and usage limits. It is the reference for
  the rotation and lifecycle rules in Section 5.
- FIPS 140-3, *Security Requirements for Cryptographic Modules*, is the standard for validated
  cryptographic modules. A FIPS 140-3 validation is evidence about a module, not about the system
  that uses it.
- NIST SP 800-63B, *Digital Identity Guidelines: Authentication and Lifecycle Management*, provides
  authenticator assurance levels and is the basis for most enterprise MFA policy.

### GDPR

Article 32(1)(a) names encryption and pseudonymisation as examples of appropriate technical
measures, and Article 32(4) requires that persons acting under the controller's authority process
data only on instruction. Article 25 requires data protection by design and by default, which in
practice means least privilege and minimal access. Article 33 requires notification of a personal
data breach, and the encryption state of the data is relevant to whether the breach poses a risk.

Encryption is not a safe harbour. If the key is compromised with the data, the breach is still a
breach; the assessment of risk considers the encryption and the key separation together.

### Sector-specific rules

- The European Health Data Space Regulation (EU) 2025/327 requires access to secondary-use data
  through secure processing environments from 2029, which makes access control and output control
  part of the regulated environment rather than an internal policy.
- The EU AI Act (Regulation (EU) 2024/1689) requires high-risk AI systems to be resilient and
  secure against exploitation of vulnerabilities (Article 15). Access control and cryptography are
  part of the evidence.
- FDA premarket cybersecurity guidance for medical devices expects authentication, access control,
  and cryptography to be described in the premarket submission, with a threat model and a software
  bill of materials.

### Standards

ISO/IEC 27001:2022 Annex A includes access control (5.15), identity management (5.16-5.18),
cryptography (8.24), and logging (8.15-8.16). ISO/IEC 27002:2022 provides implementation guidance
for each. These are certifiable management-system standards, not law.

**Cryptography is a dependency, not a feature.** An organisation that uses a library with a broken
mode, a nonce reuse bug, or a key stored in an environment variable has the appearance of
encryption and none of the protection. The review question is not "is it encrypted" but "which
library, which mode, where is the key, who can use it, and when was it last rotated". The answers
are checkable, and they are the ones a serious assessor asks.

### What the law does not specify

The Security Rule does not name an access-control model, an encryption algorithm, a key length, or
a rotation period. It requires reasonable and appropriate safeguards and documentation of
decisions. That flexibility is why two organisations with very different designs can both be
compliant, and why a claim of compliance is not evidence of effectiveness. The assessment question
is whether the design addresses the risk, not whether it names the regulation's examples.

> **Note:** Regulatory content in this file is educational and is not legal advice. Whether a
> specific access-control or encryption design satisfies a regulator depends on the system, the
> jurisdiction, and the evidence produced.

The classification to keep straight: the HIPAA Security Rule is binding law; NIST SP 800-66 and
SP 800-57 are guidance; FIPS 140-3 is a validation standard for modules; and "zero trust" is a
design philosophy. A vendor that says "we are FIPS compliant" has made a claim about a module, not
about its access control.

## 9. Industry practice and case evidence

### Change Healthcare, 2024

The HHS OCR incident FAQ and UnitedHealth Group statements describe initial access through a Citrix
remote-access portal that lacked multi-factor authentication, using compromised credentials. The
intrusion was detected on 21 February 2024; the attacker moved laterally, exfiltrated about six
terabytes, and deployed ransomware. The breach count reached 192.7 million individuals in the
filing updated in July 2025.

For this topic, the case is an access-control failure with three layers. The first is
authentication: a remote-access portal without MFA is a single-factor door. The second is
authorisation: lateral movement implies that the compromised account could reach systems beyond its
task. The third is monitoring: six terabytes leaving the environment implies that egress volume was
not alerted on. Each layer maps to a control in Section 5, and none of them is exotic.

### Anthem, 2015

OCR's findings in the Anthem settlement included insufficient technical access controls and
failure to conduct an enterprise-wide risk analysis. The breach began with spear-phishing and
resulted in the theft of data for 78.8 million people. The access-control lesson is that a large
warehouse with broad read access converts one credential into a very large disclosure. The
minimisation lesson from Topic 01 applies as well: the data that were reachable were broader than
any single task required.

### 23andMe, 2023

The company confirmed that a credential-stuffing attack gave attackers access to data belonging to
approximately 6.9 million users, including health-related information for a subset. The mechanism
was reused passwords and a sharing feature that allowed access to linked profiles. The case is about
authentication and feature design: MFA would have blocked credential stuffing, and a sharing model
that exposes linked profiles turns one compromised account into many.

### What mature programmes do

- **Phishing-resistant MFA.** FIDO2 security keys or platform authenticators for privileged and
  clinical users, with SMS and push approval treated as weaker factors. The change is driven by
  attacks that defeat one-time codes, not by a compliance deadline.
- **Just-in-time privilege.** Standing admin access is replaced by time-bound elevation with a
  reason and an approval, which reduces the population of accounts that can be used in an attack.
- **Privileged access management.** Admin sessions are brokered, recorded, and reviewed; service
  accounts are vaulted and rotated. The practice is common in finance and spreading in health.
- **Envelope encryption with separated keys.** Data keys are wrapped by a KMS or HSM key in a
  different trust domain, so a compromised storage account does not yield the key.
- **Break-glass with review.** Emergency access is a distinct role with a reason code, an alert,
  and a next-business-day review. The measure of success is a small number of genuine uses, not
  zero.
- **Access recertification with last-used data.** Managers see the last-used date next to each
  permission, which turns the review from a rubber stamp into a decision.

**Access reviews are often theatre.** A recertification that presents a role name and an approve
button produces a high approval rate and no reduction in access. The practice that works presents
the permission, the last-used date, and the consequence of removal, and asks the manager to justify
each exception. It is slower per record and far more effective. The measure of a review is not the
completion rate; it is the number of permissions removed.

> **Note:** The case facts above are drawn from the cited public reports. Statements about which
> control would have prevented an incident are inference, labelled as such in the text.

### Where practice is weaker than the documents suggest

Three gaps recur, and they are inference from published incident reviews rather than from a single
study. First, service accounts are exempt from MFA and from rotation, so they become the durable
credential an attacker prefers. Second, encryption keys are managed by the same team that manages
the data, which collapses the separation the encryption was supposed to provide. Third, access
reviews certify roles rather than permissions and last-used dates, so a manager approves a role
without seeing that it grants 84 permissions of which 22 are used. Each gap is detectable: ask for
the service-account inventory, the key custodian list, and a sample recertification record.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Role explosion | Roles named after people or projects | Every exception becomes a role | No one can explain effective access | Role mining; attributes for exceptions |
| Orphaned admin accounts | Admin accounts with no owner | Deprovisioning misses role assignments | Unattributable privileged path | Automated joiner-mover-leaver; quarterly review |
| Shared service account | One credential used by a team | No per-user attribution | Repudiation; blast radius | Per-service identities; vaulting |
| Broad read permission | Analysts can query all sites | Convenience grant | 3x reach per credential | ABAC site attribute |
| Missing MFA on remote access | VPN or portal with password only | Single-factor authentication | Credential stuffing and phishing | Phishing-resistant MFA |
| Token with broad scope | Long-lived token with admin scope | Scope creep | Stolen token is an admin credential | Short-lived, scoped tokens |
| Encryption key stored with data | Key in the same account or repository | Envelope separation absent | Compromise yields data and key | KMS/HSM in a separate trust domain |
| Key never rotated | Key age exceeds cryptoperiod | Rotation not scheduled | Long exposure window | Rotation with re-wrap; tested procedure |
| Nonce reuse in AES-GCM | Random nonce generated poorly | Catastrophic integrity failure | Plaintext recovery | Library with nonce management; tests |
| Unauthenticated encryption mode | CBC without MAC | Tampering undetected | Data manipulation | AES-GCM or ChaCha20-Poly1305 |
| Addressable treated as optional | No documented rationale | Misreading 164.306(d) | Compliance finding | Document decision and alternative |
| No audit of reads | Logs record logins only | Access not attributable | Investigation fails | Log read events with subject and resource |
| Break-glass used daily | High emergency-access rate | Permanent admin labelled emergency | Control defeated | Reason codes; review; rate alert |
| Cross-border key access | Support staff abroad can use KMS | Key residency ignored | Unlawful access or transfer | Key residency; access restrictions |
| Shared badge on a workstation | Staff use a colleague's session | Fast unlock traded for accountability | Unattributable access | Session binding; audit; workflow fix |
| Key inventory missing | No list of keys, owners, and data classes | Compromise cannot be scoped | Notification delayed and over-broad | Key inventory with owner and data map |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The role model reflects the work | Least privilege depends on it | Roles copied from the org chart | Compare used permissions to grants |
| Attribute sources are current | ABAC decisions depend on them | HR or scheduling data stale | Attribute freshness check |
| The PEP is on every path | Policy is enforced only where it runs | Direct database access bypasses it | Attempt a bypass in a test |
| Identity is strongly authenticated | All authorisation depends on identity | Phishing or credential stuffing | MFA coverage; phishing test |
| Keys are separated from data | Encryption protects against account compromise | Key in the same trust domain | Key policy and custodian review |
| Cryptoperiods are observed | Limits exposure and usage | Rotation skipped | Key age inventory |
| Audit logs are complete and protected | Accountability depends on them | Logs editable or incomplete | Log integrity test; sampling |
| Recertification is meaningful | Access drifts over time | Managers approve without data | Sample records; last-used visibility |
| Break-glass is exceptional | Emergency path is a control, not a bypass | Daily use | Rate monitoring |
| The data classification is correct | Access and encryption rules depend on it | Misclassified field | Field-level review with clinicians |
| Legal entity and jurisdiction are stable | Key residency and transfer rules depend on them | New country or cloud region | Jurisdiction register |
| The audit log is tamper-evident | Accountability depends on its integrity | Logs editable by admins | Integrity test; append-only review |
| Users do not share credentials | Attribution depends on unique identity | Workflow pressure | Session anomaly detection |

Two limits are structural. First, access control cannot protect against a user who is authorised to
see the data and misuses that access; the controls for that are audit, behavioural monitoring, and
governance, not authorisation. Second, encryption protects confidentiality, not availability or
integrity of the business process; ransomware encrypts the ciphertext as easily as the plaintext.
The controls are complementary, and a programme that relies on one is incomplete.

## 12. Exercises

The exercises are ordered from vocabulary to design. Exercises 1 to 4 build the access model and
its measurements; Exercises 5 to 7 build the cryptographic and audit controls; Exercise 8 tests
whether you can evaluate a vendor claim. Each exercise produces an artefact that could go into an
access review pack, so the set can be run as a small project rather than as separate drills.

**Exercise 1 - Distinguish the three terms (recall, 20 minutes).**
Define authentication, authorisation, and accountability, and give one control and one failure for
each. Deliverable: three-row table. Criteria: each control maps to the term; the failure is specific
(not "a breach").

**Exercise 2 - Build a role model (application, 45 minutes).**
For a 20-person clinical analytics team, define five roles and their permissions, then assign two
users with one attribute exception each. Deliverable: role-permission matrix and assignment table.
Criteria: permissions at object level; exceptions use attributes, not new roles; each assignment
has a review date.

**Exercise 3 - Compute least privilege (calculation, 40 minutes).**
A sample of 100 users has an average of 60 effective permissions and 14 used. Compute the unused
share, the excess grants for a population of 1,500, and the reduction if half the unused grants are
removed. Deliverable: calculations. Criteria: percentages and counts correct; assumption stated.

**Exercise 4 - Design an ABAC rule (design, 45 minutes).**
Write the policy for on-call radiologists who need cross-site access during a shift. Specify
attributes, the decision, the expiry, and the audit. Deliverable: policy statement. Criteria: the
rule is testable; expiry is explicit; the audit records the reason.

**Exercise 5 - Compare encryption approaches (critique, 40 minutes).**
For a 4 TB object store with restricted data, compare SSE-KMS, client-side envelope encryption, and
field-level encryption on threat coverage, cost, and operational burden. Deliverable: comparison
table with a recommendation. Criteria: names the threat each addresses; addresses key separation;
states the residual risk.

**Exercise 6 - Key rotation plan (design, 45 minutes).**
Write the rotation plan for a platform with 12,000 objects and a 24-hour recovery objective.
Include the schedule, the re-wrap versus re-encrypt decision, and the rollback. Deliverable:
two-page plan. Criteria: arithmetic for re-encryption time; rollback tested; owner named.

**Exercise 7 - Audit specification (application, 40 minutes).**
Specify the audit events for a patient-record API: fields, retention, access control, and three
alerts. Deliverable: table. Criteria: no record content in logs; alerts cover bulk export,
out-of-relationship access, and break-glass.

**Exercise 8 - Review a vendor claim (critique, 40 minutes).**
A vendor says it is "HIPAA compliant, AES-256 encrypted, and role-based". List eight questions you
would ask. Deliverable: questions with rationale. Criteria: covers key management, MFA, audit,
least privilege, addressable safeguards, and incident history.

## 13. Self-check answers

**Answer 1.** Authentication verifies identity (control: FIDO2 or MFA; failure: credential
stuffing). Authorisation decides what an identity may do (control: object-level policy at a PEP;
failure: a clinician can read another clinician's patient). Accountability attributes actions to an
identity (control: tamper-evident audit with subject, resource, and decision; failure: shared
service account with no per-user attribution). The three are dependent: authorisation is only as
good as authentication, and accountability is only as good as the identity in the log.

**Answer 2.** A role model for a 20-person analytics team typically has roles such as analyst,
senior analyst, data engineer, research lead, and platform administrator. Analysts get read access
to de-identified tables and no export; senior analysts get export with approval; data engineers get
pipeline permissions but not patient records; the research lead gets notes access with a purpose
attribute; the platform administrator gets infrastructure permissions but no data read. The two
exceptions should be attributes (a project scope and an expiry date), not new roles, because a role
per exception is how role explosion starts.

**Answer 3.** Unused share: (60 - 14) / 60 = 46 / 60 = 76.7 percent. Excess grants for 1,500
users: 1,500 x 46 = 69,000 unused grants. Removing half: 34,500 grants removed, leaving 34,500
unused. The calculation assumes the sample is representative; a stratified sample by role family
would be better before a budget decision.

**Answer 4.** The rule should grant `read_patient` and `read_notes` when the subject has the
attribute `on_call = true`, the resource site is in the subject's `covering_sites` list, and the
current time is within the shift window; otherwise deny. The grant should expire at the end of the
shift, and the audit record should include the reason `on_call_coverage`. The rule is testable with
three cases: on-call during the shift for a covered site (allow), on-call outside the shift (deny),
and not on-call for the same site (deny). The attribute source is the scheduling system, which must
be current; a stale schedule is the main failure mode.

**Answer 5.** SSE-KMS protects against media theft and snapshot exposure but not against an
attacker who can use the application's credentials or the KMS policy; it is cheap and should be the
default. Client-side envelope encryption protects against storage-account compromise because the
data key is wrapped by a key in a separate trust domain; it costs a key service and a re-wrap
process. Field-level encryption protects specific fields even when the application is compromised,
but it breaks queries, indexing, and analytics on those fields and costs the most operationally.
The recommendation for restricted data is envelope encryption with field-level encryption for
identifiers that are not needed for joins.

**Answer 7.** Log the subject identifier, the resource identifier, the action, the decision, the
reason code, the timestamp, and the source address; do not log the record content. Retain according
to the legal and operational requirement, protect with append-only storage, and restrict read
access to the audit team. Alerts: bulk export above a threshold; access outside the care
relationship; break-glass use; admin role change; access to VIP records. The test of the
specification is whether an investigator can reconstruct who saw what, when, and why.

**Answer 8.** Ask: which HIPAA provisions are addressable and how were they implemented or
documented; where are the keys stored and who can use them; is MFA enforced for all users and
especially admins; are service accounts individually identified and rotated; what is logged and
who can read the logs; how is least privilege enforced at the object level; how often is access
recertified; and what is the incident history and breach-notification process. A vendor that
answers only "AES-256" has described an algorithm, not a control.

**Rubric for Exercise 2.** Full credit requires five roles, object-level permissions, attributes
for both exceptions, and a review date. A role named after a person or project is marked down.

**Rubric for Exercise 4.** Full credit requires attributes (on-call status, covering site, shift
window), a decision rule, an expiry, and an audit entry with reason. A rule that grants permanent
cross-site access to all radiologists does not pass.

**Rubric for Exercise 6.** Full credit requires the re-encryption arithmetic, the re-wrap versus
re-encrypt decision, a rollback, an owner, and a test. A plan that says "rotate annually" without
the operational steps does not pass.

## 14. Interview questions

**1. What is the difference between RBAC and ABAC?**
Strong answer: RBAC grants by role and is simple but coarse; ABAC evaluates subject, resource,
action, and environment attributes and is fine-grained but depends on attribute quality; most
clinical systems use both. Weak answer: "ABAC is more modern".

**2. How do you detect role explosion?**
Strong answer: roles per user, roles with one user, roles named after people or projects, and a
rising ratio of roles to job families; the fix is attributes and role mining. Weak answer: "count
the roles".

**3. What does least privilege mean in practice?**
Strong answer: the minimum permission for the task, time-bound where possible, measured by
effective versus used permissions and recertified with last-used data. Weak answer: "give people
only what they need".

**4. Why is encryption at rest not enough?**
Strong answer: it protects media and some account compromises, not an attacker who can use the
application or the key; key separation and access control are the real controls. Weak answer:
"encryption solves it".

**5. What is envelope encryption?**
Strong answer: data keys encrypt data, a key-encryption key wraps the data keys, and a root key in
an HSM protects the hierarchy; rotation re-wraps without re-encrypting. Weak answer: "encrypting
the key".

**6. How long should a key live?**
Strong answer: the cryptoperiod depends on the key type, usage volume, and data sensitivity; NIST
SP 800-57 provides guidance and usage limits such as 2^32 GCM invocations; the policy must be
tested. Weak answer: "rotate annually".

**7. What is break-glass and how do you control it?**
Strong answer: emergency access with a reason code, short session, alert, and retrospective review;
measured by rate, not by existence. Weak answer: "an admin account for emergencies".

**8. How do you audit access without logging PHI?**
Strong answer: log subject, resource identifier, action, decision, reason, time, and source; never
log content; protect the log with append-only storage and separate access. Weak answer: "log
everything".

**9. A clinician shares a password. What control failed?**
Strong answer: authentication and accountability, plus workflow design; unique user identification
is required, and the sharing usually signals a usability problem. Weak answer: "policy violation".

**10. How do you manage keys across cloud and on-premises?**
Strong answer: a documented hierarchy, a KMS or HSM per trust domain, key residency, escrow and
recovery, and a tested compromise procedure. Weak answer: "use the cloud KMS".

**11. How do you prevent a service account from becoming the weakest credential?**
Strong answer: give every service its own identity, vault and rotate credentials, restrict scope to
the specific resource, log its actions, and include it in the access review with a named owner.
Weak answer: "we use an API key".

**12. What would you do if a key was exposed in a public repository?**
Strong answer: revoke and rotate immediately, assess what the key protected and who could have used
it, re-encrypt if the key was a data key, check logs for use, notify per the legal and contractual
requirements, and review how the secret reached the repository. Weak answer: "rotate the key".

**13. What is the difference between encryption and tokenization?**
Strong answer: encryption is reversible with a key and preserves the ability to recover the value;
tokenization replaces the value with a surrogate whose mapping lives in a vault, and the token may
have no mathematical relationship to the value. Tokenization is useful for joins and format
constraints; encryption is useful for storage and transport. Both are pseudonymisation techniques
when the mapping exists, not anonymisation.

## 15. Key terms

- **ABAC** - attribute-based access control.
- **Accountability** - attributing actions to a subject.
- **AES-GCM** - authenticated encryption with associated data.
- **Authentication** - verifying identity.
- **Authorisation** - deciding what an identity may do.
- **Break-glass** - emergency access with logging and review.
- **Cryptoperiod** - the time a key is authorised for use.
- **Data-encryption key** - the key that encrypts data.
- **Envelope encryption** - data keys wrapped by a key-encryption key.
- **FIDO2** - phishing-resistant authentication standard.
- **FIPS 140-3** - validation standard for cryptographic modules.
- **HSM** - hardware security module.
- **Key ceremony** - documented generation of root keys.
- **Key-encryption key** - the key that wraps data keys.
- **Key rotation** - replacing a key and re-wrapping or re-encrypting.
- **KMS** - key management service.
- **Least privilege** - minimum access for the task.
- **MFA** - multi-factor authentication.
- **Nonce** - a value used once in encryption.
- **Orphaned account** - an account with no active owner.
- **PAP** - policy administration point.
- **PDP** - policy decision point.
- **PEP** - policy enforcement point.
- **Permission** - an allowed action on a resource.
- **PIP** - policy information point.
- **RBAC** - role-based access control.
- **ReBAC** - relationship-based access control.
- **Revocation** - invalidating a credential before expiry.
- **Role explosion** - excessive role proliferation.
- **Separation of duties** - no single subject completes a sensitive transaction.
- **Service account** - a non-human identity.
- **Session fixation** - an attack reusing a session identifier.
- **TLS 1.3** - transport security protocol version 1.3.
- **Token scope** - permissions carried by a token.
- **Zero trust** - per-request access decisions independent of network location.
- **Data class** - the sensitivity category that drives access and encryption rules.
- **Object-level authorisation** - checking access to a specific record.
- **Privileged access management** - brokering, recording, and reviewing admin sessions.
- **Just-in-time access** - time-bound elevation instead of standing privilege.
- **Tokenization** - replacing a value with a surrogate held in a vault.
- **Envelope re-wrap** - encrypting a data key under a new key-encryption key.
- **Attribute store** - the authoritative source of ABAC attributes.
- **Negative test** - a test that asks the system to deny an action.
- **HSTS** - HTTP Strict Transport Security, which forces TLS.
- **mTLS** - mutual TLS, where both client and server present certificates.
- **Session binding** - tying a session to a device or context.

## 16. Sources

All URLs accessed 26 October 2026.

**US regulation and guidance**
- 45 CFR 164.312, *Technical safeguards*.
  https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
- 45 CFR 164.306, *Security standards: general rules*.
  https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.306
- NIST, *SP 800-66 Rev. 2, Implementing the HIPAA Security Rule*, 2024.
  https://csrc.nist.gov/pubs/sp/800/66/r2/final
- HHS Office for Civil Rights, *Change Healthcare Cybersecurity Incident Frequently Asked
  Questions*. https://www.hhs.gov/hipaa/for-professionals/special-topics/change-healthcare-cybersecurity-incident-frequently-asked-questions/index.html
- HHS OCR, *Anthem Resolution Agreement*, 2018.
  https://www.hhs.gov/hipaa/for-professionals/compliance-enforcement/agreements/anthem/index.html

**NIST guidance**
- NIST, *SP 800-162, Guide to Attribute Based Access Control*, 2014, updated 2019.
  https://csrc.nist.gov/pubs/sp/800/162/upd2/final
- NIST, *SP 800-207, Zero Trust Architecture*, 2020.
  https://csrc.nist.gov/pubs/sp/800/207/final
- NIST, *SP 800-57 Part 1 Rev. 5, Recommendation for Key Management*, 2020.
  https://csrc.nist.gov/pubs/sp/800/57/pt1/r5/final
- NIST, *SP 800-63B, Digital Identity Guidelines: Authentication and Lifecycle Management*, 2017,
  updated.
  https://pages.nist.gov/800-63-3/sp800-63b.html
- NIST, *FIPS 140-3, Security Requirements for Cryptographic Modules*, 2019.
  https://csrc.nist.gov/pubs/fips/140-3/final

**EU regulation**
- Regulation (EU) 2016/679 (GDPR), Article 32.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Regulation (EU) 2025/327 (European Health Data Space).
  https://eur-lex.europa.eu/eli/reg/2025/327/oj
- Regulation (EU) 2024/1689 (Artificial Intelligence Act).
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj

**Standards**
- ISO/IEC 27001:2022, *Information security, cybersecurity and privacy protection - Information
  security management systems - Requirements*.
- ISO/IEC 27002:2022, *Information security controls*.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_threat-modeling.md`](04_threat-modeling.md)
- Next topic: [`06_fairness.md`](06_fairness.md)
- Related topic: [`03_privacy-enhancing-technologies.md`](03_privacy-enhancing-technologies.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
