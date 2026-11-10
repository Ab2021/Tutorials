---
case_id: "CASE-19-01"
case_title: "Choosing a code-assistance model with authorised repository context"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Developer tooling"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["authorization-scope", "cost-latency-tradeoff"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Developer productivity owner"
source_index: "INDEX.md"
review_trigger: "Model, repository permission, benchmark, toolchain, or cost contract change"
---

## 1. The One-Line Framing

**Illustrative design exercise; no repository, model run, deployment, or measured developer outcome is asserted.**
Given an authorised repository task and a pinned revision, select a code-assistance candidate or abstain for a developer, subject to access, test, latency, cost, and human-approval constraints, compared with manual implementation and search.
The system may draft a patch or explanation; it may not merge, disclose unauthorised code, or claim correctness without the declared test and review evidence.

## 2. Context & Background

Developer tasks require reading a specific revision, locating relevant code, proposing a bounded change, executing allowed checks in a sandbox, and obtaining human review.
Manual development is the baseline because it establishes existing review, access, and accountability boundaries.
A model benchmark score is not the workflow outcome: a plausible patch can target the wrong revision, omit a dependency, leak a secret, or pass an insufficient test.

## 3. The Business Question

**Illustrative stakeholder question:** “Which authorised model-and-context configuration should be evaluated for drafting small repository changes without weakening code review or repository isolation?”
Decision owner: developer-productivity owner; repository owners control access and maintainers approve changes.
Actions: manual baseline; assistant without retrieval; assistant with authorised pinned retrieval; defer/abstain; reject access.
Hard constraints: least-privilege retrieval, pinned source revision, sandbox-only tools, explicit test contract, human merge approval, and a bounded cost/latency budget.

## 4. Assumptions & Constraints

| ID | Condition | Check | Impact | Fallback |
|---|---|---|---|---|
| A-01 | Task and repository revision are pinned. | commit hash. | Wrong-code evaluation. | Abstain. |
| A-02 | Retrieved paths are authorised for requester/purpose. | access decision. | Disclosure. | Deny. |
| A-03 | Sandbox test oracle covers intended behavior. | test manifest. | False success. | Manual review. |
| C-01 | Human maintainer approves merge. | review event. | Unauthorised change. | Block merge. |
| C-02 | Cost/latency are measured per accepted request. | telemetry plan. | Budget breach. | throttle. |

## 5. Data, Tools & Method

### 5a. Data Specification

- Unit: one task at one repository revision with authorised file set, expected behavior, and test manifest.
- Inputs: task text, revision hash, permission decision, retrieved path IDs/content hashes, tool outputs, and deadline.
- Label: independent maintainer assessment of functional correctness, scope, reviewability, and policy compliance.
- Split by repository and time; never allow the same change family or future commit in both train and evaluation.

### 5b. Tools & Stack

- Proposed permission service filters repository paths before retrieval.
- Proposed retrieval service binds chunks to commit hash, path, and access decision.
- Proposed model service records model, tokenizer, prompt, decoding, and tool versions.
- Proposed sandbox runs an allowlisted build/test command with bounded resources.
- Proposed reviewer UI displays patch, evidence paths, tool logs, and abstention reasons.

### 5c. Method Summary

- 1. Validate requester, task scope, repository, branch, and revision.
- 2. Retrieve only authorised, revision-pinned evidence; record each path hash.
- 3. Ask a model for a bounded draft with cited files and stated assumptions.
- 4. Run allowed static/build/test checks in an isolated sandbox.
- 5. Route to maintainer review; approval is a human action, not model confidence.
- 6. Evaluate against manual, no-retrieval, and authorised-retrieval baselines.

### 5d. Data provenance, quality, exclusions and leakage controls

- Freeze commit hash and retrieval snapshot before generation.
- Exclude secrets, unauthorised paths, future commits, external network results, and post-review corrections from generation inputs.
- Deduplicate patches and split by repository/time; audit toolchain and dependency versions.
- Report missing tests, ambiguous tasks, unsupported languages, and permission denials as explicit exclusions.

### 5e. Reproducibility and method-selection ledger

| ID | Artifact | Purpose | Limitation |
|---|---|---|---|
| SRC-01 | illustrative revision manifest | identity/access | no real repository accessed |
| CFG-01 | model/retrieval/sandbox bundle | repeatable proposal | proposed only |
| RUN-01 | capacity arithmetic | decision support | not a service run |
| GOLD-01 | future maintainer review set | correctness oracle | unavailable |

## 6. Analysis Walkthrough

**ILLUSTRATIVE workload:** 480 eligible tasks/day. A proposed route accepts 30% for maintainer review after sandbox checks; each reviewed patch takes 12 staffed minutes; two maintainers provide 8 staffed hours/day.
Reviewed tasks = `480 × 0.30 = 144`; review work = `144 × 12 = 1,728 minutes = 28.8 staffed hours/day`.
Two maintainers provide 16 staffed hours/day; the modelled shortfall is 12.8 hours/day, so this route policy is infeasible without narrower scope, lower volume, more capacity, or manual prioritisation.
At 15% routing, work is `480 × .15 × 12 = 864 minutes = 14.4 hours`, leaving 1.6 nominal hours.
```text
FORMULA [CODE-01]: W = N * r * m
WHERE: W review minutes/day; N eligible tasks/day; r review fraction; m staffed minutes/task.
CONSTRAINT: constant means; excludes rework, queue variation, CI contention, and escalation.
NUMERICAL EXAMPLE: 480*.30*12=1728 minutes=28.8 hours.
INTERPRETATION: throughput must be admitted by reviewer capacity, not model generation rate.
```

## 7. The Finding

The illustrative proposal should not be piloted at a 30% review route with two maintainers because its calculated review workload exceeds capacity by 12.8 hours/day.
A narrower 15% route appears capacity-feasible under the simplified assumptions but still requires independent task-correctness, security, and regression evaluation.
No model quality, cost, latency, repository safety, or developer outcome is measured.

## 8. System/Solution Design

`Task/access gate -> pinned revision -> authorised retrieval -> model draft -> sandbox checks -> maintainer review -> approved merge path`.
All retrieval is permission-filtered before ranking; a relevant file is still unavailable if unauthorised.
State includes task ID, revision, retrieved hashes, configuration bundle, tool logs, patch hash, review disposition, and export/merge decision.
Failure recovery invalidates a draft when revision, permission, model, tokenizer, prompt, or toolchain changes.

## 9. Evaluation & Validation

Measure task completion under an independent oracle, patch scope correctness, test outcome, review minutes, accepted/rejected/abstained populations, retrieval authorization violations, cost, and latency.
Define numerator, denominator, repository/time split, language slice, task class, and comparator for every metric.
A passing test is a signal, not proof of security, requirement completeness, or future regression safety.
Run negative permission tests, future-commit contamination checks, and target-equivalence checks for tool use.

## 10. Action Taken

No production action was taken.
Proposed action: collect an authorised task set, lock revisions and independent review rubric, measure the manual baseline, and test a bounded no-merge sandbox workflow.
Stop on access violation, source/version mismatch, missing oracle, sandbox boundary failure, or review queue breach.

## 11. Outcome & Results

**ILLUSTRATIVE / arithmetic verified / operational outcome not measured.**
At 30% routing, 144 reviews require 28.8 staffed hours/day, 12.8 hours above two-maintainer capacity.
At 15% routing, 72 reviews require 14.4 hours/day, leaving 1.6 nominal hours.
No performance result or claim of code correctness is asserted.

## 12. Failure Modes & What Went Wrong

### FM-01 — repository leakage
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of repository leakage.
- Mechanism: repository leakage changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for repository leakage.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-02 — future-code contamination
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of future-code contamination.
- Mechanism: future-code contamination changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for future-code contamination.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-03 — license mismatch
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of license mismatch.
- Mechanism: license mismatch changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for license mismatch.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-04 — secret exposure
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of secret exposure.
- Mechanism: secret exposure changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for secret exposure.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-05 — unauthorized retrieval
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of unauthorized retrieval.
- Mechanism: unauthorized retrieval changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for unauthorized retrieval.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-06 — stale branch context
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of stale branch context.
- Mechanism: stale branch context changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for stale branch context.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-07 — wrong language route
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of wrong language route.
- Mechanism: wrong language route changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for wrong language route.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-08 — test-harness mismatch
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of test-harness mismatch.
- Mechanism: test-harness mismatch changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for test-harness mismatch.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-09 — dependency hallucination
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of dependency hallucination.
- Mechanism: dependency hallucination changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for dependency hallucination.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-10 — API-version drift
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of API-version drift.
- Mechanism: API-version drift changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for API-version drift.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-11 — unsafe patch scope
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of unsafe patch scope.
- Mechanism: unsafe patch scope changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for unsafe patch scope.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-12 — latency overload
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of latency overload.
- Mechanism: latency overload changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for latency overload.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-13 — token-budget truncation
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of token-budget truncation.
- Mechanism: token-budget truncation changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for token-budget truncation.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-14 — retrieval miss
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of retrieval miss.
- Mechanism: retrieval miss changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for retrieval miss.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-15 — retrieval poison
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of retrieval poison.
- Mechanism: retrieval poison changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for retrieval poison.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-16 — duplicate files
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of duplicate files.
- Mechanism: duplicate files changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for duplicate files.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-17 — generated-test overfit
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of generated-test overfit.
- Mechanism: generated-test overfit changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for generated-test overfit.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-18 — benchmark contamination
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of benchmark contamination.
- Mechanism: benchmark contamination changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for benchmark contamination.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-19 — merge conflict
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of merge conflict.
- Mechanism: merge conflict changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for merge conflict.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-20 — formatter-only success
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of formatter-only success.
- Mechanism: formatter-only success changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for formatter-only success.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-21 — build-cache masking
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of build-cache masking.
- Mechanism: build-cache masking changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for build-cache masking.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-22 — non-deterministic test
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of non-deterministic test.
- Mechanism: non-deterministic test changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for non-deterministic test.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-23 — sandbox escape
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of sandbox escape.
- Mechanism: sandbox escape changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for sandbox escape.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-24 — tool timeout
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of tool timeout.
- Mechanism: tool timeout changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for tool timeout.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-25 — rate-limit retry
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of rate-limit retry.
- Mechanism: rate-limit retry changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for rate-limit retry.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-26 — context-window overflow
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of context-window overflow.
- Mechanism: context-window overflow changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for context-window overflow.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-27 — incomplete diff
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of incomplete diff.
- Mechanism: incomplete diff changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for incomplete diff.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-28 — wrong repository
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of wrong repository.
- Mechanism: wrong repository changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for wrong repository.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-29 — access revocation
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of access revocation.
- Mechanism: access revocation changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for access revocation.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-30 — model-version drift
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of model-version drift.
- Mechanism: model-version drift changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for model-version drift.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-31 — prompt-version drift
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of prompt-version drift.
- Mechanism: prompt-version drift changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for prompt-version drift.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-32 — tokenizer mismatch
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of tokenizer mismatch.
- Mechanism: tokenizer mismatch changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for tokenizer mismatch.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-33 — temperature variance
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of temperature variance.
- Mechanism: temperature variance changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for temperature variance.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-34 — fallback model change
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of fallback model change.
- Mechanism: fallback model change changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for fallback model change.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-35 — cost-budget breach
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of cost-budget breach.
- Mechanism: cost-budget breach changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for cost-budget breach.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-36 — queue starvation
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of queue starvation.
- Mechanism: queue starvation changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for queue starvation.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-37 — long-file bias
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of long-file bias.
- Mechanism: long-file bias changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for long-file bias.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-38 — rare-language regression
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of rare-language regression.
- Mechanism: rare-language regression changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for rare-language regression.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-39 — reviewer fatigue
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of reviewer fatigue.
- Mechanism: reviewer fatigue changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for reviewer fatigue.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### FM-40 — acceptance bias
- Symptom: a proposed patch or report is unsafe, incomplete, delayed, or unsupported because of acceptance bias.
- Mechanism: acceptance bias changes the validity of source, evaluation, authority, capacity, or release evidence.
- Diagnostic: retain task/revision/access/configuration identifiers and test the discriminating invariant for acceptance bias.
- Recovery: abstain or invalidate the draft, route to the accountable maintainer, and preserve audit evidence.
- Verification: repeat the authorised check on the pinned revision and confirm no forbidden path, tool, or state transition occurred.

### Failure recovery ledger

#### FM-01 state trace — repository leakage
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-02 state trace — future-code contamination
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-03 state trace — license mismatch
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-04 state trace — secret exposure
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-05 state trace — unauthorized retrieval
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-06 state trace — stale branch context
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-07 state trace — wrong language route
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-08 state trace — test-harness mismatch
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-09 state trace — dependency hallucination
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-10 state trace — API-version drift
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-11 state trace — unsafe patch scope
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-12 state trace — latency overload
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-13 state trace — token-budget truncation
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-14 state trace — retrieval miss
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-15 state trace — retrieval poison
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-16 state trace — duplicate files
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-17 state trace — generated-test overfit
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-18 state trace — benchmark contamination
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-19 state trace — merge conflict
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-20 state trace — formatter-only success
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-21 state trace — build-cache masking
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-22 state trace — non-deterministic test
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-23 state trace — sandbox escape
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-24 state trace — tool timeout
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-25 state trace — rate-limit retry
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-26 state trace — context-window overflow
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-27 state trace — incomplete diff
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-28 state trace — wrong repository
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-29 state trace — access revocation
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-30 state trace — model-version drift
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-31 state trace — prompt-version drift
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-32 state trace — tokenizer mismatch
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-33 state trace — temperature variance
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-34 state trace — fallback model change
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-35 state trace — cost-budget breach
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-36 state trace — queue starvation
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-37 state trace — long-file bias
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-38 state trace — rare-language regression
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-39 state trace — reviewer fatigue
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

#### FM-40 state trace — acceptance bias
1. Preconditions: requester, repository, commit, purpose, and tool policy are all bound to the task.
2. Trigger: the diagnostic detects a source, authority, evaluation, capacity, or release inconsistency.
3. Competing explanation: an upstream identity/configuration mismatch can resemble the same symptom.
4. Discriminator: inspect the immutable revision, retrieval-path log, tool manifest, and reviewer record.
5. State transition: `DRAFT` becomes `ABSTAINED` or `REVIEW_REQUIRED`, never `MERGE_APPROVED`.
6. Immediate mitigation: revoke inaccessible context and stop further tool calls for that task.
7. Correction: create a new versioned task only after the owner resolves the invariant.
8. Prevention: monitor rate by repository, language, task type, model bundle, and permission policy.
9. Verification: a maintainer can reconstruct the decision from retained evidence.
10. Escalation: security, repository, or tool-boundary faults go to their accountable owner.

## 13. Transferable Lesson

Code assistance is a constrained evidence-and-review workflow, not a text-completion contest.
Repository/time splits, permission filtering before retrieval, pinned source versions, independent oracles, and human approval are all necessary to make a model comparison meaningful.
Reviewer capacity is a safety and quality constraint: an unstaffable route policy cannot be made valid by lowering a confidence threshold.

## 14. Connections & Cross-References
Related Module 19 material includes token budgets, attention/context limits, inference admission, and scaling; those concepts do not validate this proposed workflow.

### Interview pack

#### Q1. How is repository leakage prevented?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q2. Why is a benchmark score insufficient?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q3. What is the correct split?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q4. When should the system abstain?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q5. How do you size reviewer capacity?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q6. Why pin a commit?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q7. What is the baseline?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q8. How do tests fail as an oracle?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q9. What does retrieval authorization mean?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q10. How do you roll back?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q11. What changes with a long context?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q12. How is cost measured?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q13. Why separate model and editorial state?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q14. What validates a language route?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

#### Q15. What makes a patch auditable?
**Answer:** Bind each task to authorised revision-pinned evidence, use an independent task oracle and human approval, and treat any missing authority or evidence as abstention rather than a model-confidence decision.
**Reasoning:** Correctness, access, source time, review capacity, and release control are separate contracts.
**Follow-up:** Name the denominator and boundary that would invalidate the proposed comparison.
**Acceptable alternatives:** A different model or workflow is acceptable only when it preserves these contracts.
**Red flags:** Calling a passing test, fluent patch, or aggregate benchmark a merge decision.
**Scoring:** 3 for contracts and boundary; 2 for mechanism; 1 for generic model claim.
**Answer support:** Sections 4–12.

## 15. Quick Reference Card
| Field | Rule | Boundary |
|---|---|---|
| Input | authorised task + pinned revision | no implicit repository context |
| Retrieve | filter permission before ranking | relevance is not authority |
| Generate | draft only | no direct merge |
| Validate | independent oracle + sandbox | passing test is not full proof |
| Capacity | W=N*r*m | means are not queue guarantees |
| State | DRAFT -> REVIEW_REQUIRED -> APPROVED | no direct approval |
| Evidence | illustrative/not measured | no deployment claim |
