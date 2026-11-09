---
collection: "program-orientation-ai-landscape"
file_id: "03"
topic: "technology due diligence questions"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Vendor, model, data-processing, contract, or workload change"
notation_style: "plain-text"
prerequisites: ["01_AI_Use_Case_Charter.md", "02_AI_Market_Map.md"]
pattern_tags: ["evidence", "acceptance-test", "vendor-review"]
source_status: "OPEN — inventory names only; primary evidence must be obtained per candidate"
owner_role: "technical due-diligence lead"
source_index: "INDEX.md"
---

# Technology Due Diligence: Claims, Tests, and Operating Boundaries

> **Purpose:** Turn a technology claim into a bounded acceptance test and a decision record rather than a feature checklist.
> **Prerequisites:** [AI Use-Case Charter](01_AI_Use_Case_Charter.md), [AI Market Map](02_AI_Market_Map.md), basic arithmetic with fractions and currency.
> **Key outcome:** Given a candidate system claim, write the evidence request, test population, pass/fail threshold, and decision consequence.
> **This file does NOT cover:** legal advice, certification, or a conclusion about a named vendor.
> **Level:** 2 Core Methods
> **Version / reviewed:** 1.0 / not reviewed
> **Evidence:** DERIVED test logic; ILLUSTRATIVE test data; SOURCED method references; OPEN candidate-specific evidence.
> **Retrieval cue:** If a claim says “secure,” “accurate,” “enterprise-ready,” or “cheaper,” begin with Section 3.

## 2. Concept Map / Mental Model

```text
[Candidate claim]
       |
       v
[Operational interpretation: task, population, window, metric]
       |
       +--> missing definition --> [OPEN evidence request]
       v
[Acceptance protocol: baseline + candidate + held-out cases]
       |
       +--> constraint violation --> [reject / narrow / escalate]
       v
[Evidence ledger: artifact, version, owner, expiry]
       |
       v
[Decision: adopt, pilot, retain baseline, or collect evidence]
```

Due diligence asks “what would make this safe enough and useful enough for this decision?” It does not ask for an unlimited proof that a technology is good. A feature exists only when its version, configuration, user boundary, failure behavior, and evidence are known.

**Takeaway:** Translate an adjective into an observable claim before comparing tools.

## 3. Foundational Definitions

| Term | Operational meaning | Evidence needed | Common confusion |
|---|---|---|---|
| Claim | falsifiable statement with scope | exact wording, version, owner | marketing category |
| Acceptance criterion | predeclared pass/fail or escalation condition | metric, population, window, comparator | desired outcome after seeing data |
| Baseline | current credible process on same units | configuration and measured outputs | weakest imaginable option |
| Evidence artifact | inspectable report, log, contract, test output, or source | identifier/date/access limitation | a verbal assurance |
| Reproducibility | another reviewer can repeat the material reasoning | inputs, config, version, method | one screenshot |
| Data boundary | permitted collection, transfer, retention, and access path | field map and actual configuration | country label alone |
| Availability | defined proportion of eligible requests successfully served | numerator, denominator, window | uptime without user impact |
| False acceptance | accepting a candidate that fails the true requirement | test design/risk | any failed test |

A useful evidence ledger states what the artifact supports and what it does not. For example, a penetration-test summary may support a bounded assessment of a named configuration; it does not establish that every integration or future version is secure.

**Takeaway:** Evidence has a scope and expiration, not a halo.

## 4. Core Formulas with Full Derivation

### 4.1 Base: measured pass rate

Validity before use: test cases have a frozen eligibility rule and an auditable expected outcome. This is a test result, not necessarily field performance.

```text
FORMULA [DD-01]: p = n_pass / n_test
WHERE: p pass rate [fraction]; n_pass [test cases]; n_test [test cases].
CONSTRAINT: each case is scored once with a declared rubric; no post-hoc removal.
NUMERICAL EXAMPLE: ILLUSTRATIVE n_pass=91, n_test=100; p=0.91=91%.
INTERPRETATION: the evidence supports 91 passes in this test set, not “91% reliable everywhere.”
```

Derivation: the empirical proportion is successful units divided by evaluated units. Its usefulness depends on case construction, test independence, and scoring reliability.

### 4.2 General: total cost of ownership estimate

Validity before use: all quantities share a window and currency; excluded costs are named. The estimate is not a price quote.

```text
FORMULA [DD-02]: TCO = F + N*k + H*w + R
WHERE: F fixed subscription/integration cost [currency/window]; N [requests/window];
       k variable cost [currency/request]; H review/support hours [hours/window];
       w loaded labor rate [currency/hour]; R risk/reserve estimate [currency/window].
CONSTRAINT: avoid double-counting vendor support in F and H; price retries and failures if expected.
NUMERICAL EXAMPLE: F=$2,000/month, N=20,000/month, k=$0.08/request,
                   H=30 hours/month, w=$60/hour, R=$500/month.
                   TCO=2,000+20,000*0.08+30*60+500=$5,900/month.
INTERPRETATION: the candidate needs at least $5,900/month of comparable benefit to break even
                under these illustrative inputs.
```

Arithmetic check: variable cost `$1,600`; labor `$1,800`; the components sum to `$5,900`. Sensitivity: doubling retries effectively doubles `k` to `$0.16`, increasing TCO by `$1,600/month`; retry behavior is therefore a first-class test item.

### 4.3 Variant: error budget for an acceptance gate

Let `e_max` be allowed errors among `n_test` cases under a stated rubric. Then `n_fail <= e_max`. For a 95% pass threshold on 100 cases, at most five failures are allowed; rounding the threshold before the count can be wrong on small samples.

```text
FORMULA [DD-03]: n_fail <= floor(n_test*(1 - p_min))
WHERE: p_min minimum pass fraction; n_fail [cases].
CONSTRAINT: this is an administrative gate, not a confidence interval.
NUMERICAL EXAMPLE: n_test=47, p_min=0.90; floor(47*0.10)=4 failures allowed.
                   43 passes gives 43/47=91.49%; 42 passes gives 89.36%.
INTERPRETATION: the discrete test gate must be stated before execution.
```

### 4.4 Limit / counterexample

A candidate can pass a common-case test but be unsuitable if it fails a hard constraint: inaccessible audit trail, prohibited data route, or nonreversible action. Averaging a hard failure into a high pass rate is invalid. Use a separate zero-tolerance gate.

**Takeaway:** A score summarizes a test; it cannot waive a disqualifying constraint.

## 5. Worked Numerical Example

**Illustrative vendor-claim protocol.** Claim: “The assistant resolves incoming knowledge requests accurately and cheaply.” Interpret “accurately” as a reviewer rubric: a response must cite an authorized source, contain no unsupported material claim, and make a correct routing decision. Interpret “cheaply” as total cost per eligible request relative to baseline.

| Protocol field | Declared value |
|---|---|
| Eligible unit | one English-language internal knowledge request with an authorized source |
| Window | fixed 4-week historical sample, held out from prompt tuning |
| Candidate | version/configuration recorded before test |
| Baseline | authorized search plus human response |
| Quality gate | at least 90% of 100 cases pass every rubric component |
| Safety gate | zero prohibited-data disclosures in adversarial and ordinary cases |
| Cost gate | estimated TCO no more than $6,000/month at 20,000 requests |
| Decision | pilot only if all gates pass; otherwise retain baseline or narrow scope |

Suppose 92 of 100 cases pass the full rubric, but two outputs disclose a prohibited field in a red-team test. `DD-01` gives 92%, so quality clears the 90% administrative threshold. The zero-tolerance safety gate fails. The decision is **not pilot** until the data path/configuration is remediated and retested; an average cannot compensate for the violation.

For cost, Section 4 estimates `$5,900/month`, leaving `$100/month` margin. A 10% variable-cost increase produces `20,000*(0.088-0.08)=$160/month`, exceeding the gate even without retries. The cost conclusion is fragile and should trigger an updated quote and observed token/request distribution.

**Takeaway:** Use multiple gates when an unacceptable failure cannot be traded against convenience.

## 6. Visual / Geometric Intuition

```text
                   hard constraint fails
                         |
                         v
         [reject/narrow] +--------------------+
                         |  quality / cost    |
                         |  points here only  |
                         +--------------------+
                              pass region
```

The acceptance region is not always a diagonal “higher score is better” line. Some dimensions—authorization, action reversibility, or data route—are walls. Quality and cost choose among candidates inside the permitted region; they do not tunnel through a wall.

**Takeaway:** Model hard constraints as gates, not weighted preferences.

## 7. Comparison / Selection Table

| Question | Strong evidence | Weak substitute | Decision consequence |
|---|---|---|---|
| What task is solved? | input/output contract with examples and exclusions | capability label | cannot compare results |
| Does quality hold? | frozen representative and adversarial test with rubric | public benchmark | require acceptance test |
| Is data handling permitted? | field/flow/retention map plus actual configuration and contract | “enterprise security” | block or security review |
| Can output cause a side effect? | approval/idempotency/audit/rollback trace | demo click-through | restrict to propose-only |
| What does it cost? | dated quote and volume model including retries/review | unit price alone | recalculate TCO |
| Can it be operated? | named owner, SLO/alerts, incident and exit plan | roadmap assurance | no production promotion |
| Can it be exited? | export/deletion/versioning terms tested or documented | promise of portability | record lock-in risk |

**Takeaway:** Request the artifact that directly tests the decision, not the most polished collateral.

## 8. Failure Modes & Edge Cases

| ID | Symptom | Competing causes | Discriminating check | Correction | Verification |
|---|---|---|---|---|---|
| DD-F01 | Benchmark looks strong but pilot disappoints | task mismatch; leakage; different abstention policy | replay same candidate on frozen local units | replace benchmark claim with acceptance result | independent scorer can reproduce rubric result |
| DD-F02 | “Compliant” claim has no usable scope | certification applies to another product; configuration differs | match certificate/system/version/control to actual boundary | obtain scoped evidence or remove claim | owner signs artifact-to-deployment mapping |
| DD-F03 | Cost forecast is too low | output length, retries, review, egress omitted | reconcile invoice/usage fields to model | include all material terms and sensitivity | revised model matches measured usage window |
| DD-F04 | Safety test passes but incident occurs | adversarial coverage too narrow; integration changes inputs | test integration path and abuse cases | add regressions and approval boundary | rerun includes prior failing input |
| DD-F05 | Same prompt returns inconsistent action | nondeterminism; hidden version change; state contamination | pin version/configuration and run repeated trials | make action deterministic or require review | change log and repeatability record exist |
| DD-F06 | Exit plan is unusable | proprietary state format; missing logs; contractual delay | perform limited export/restore exercise | negotiate/export early, keep canonical records | restore reproduces sampled records within stated tolerance |

**Takeaway:** A failed due-diligence check is information; hiding it creates a production surprise.

## 9. Complexity & Cost Analysis

Testing has a cost that scales with the number of cases and judges. If `n` cases take `m` reviewer-minutes each and two independent reviewers are needed, labor hours are `2*n*m/60`. For 100 cases at 6 minutes, that is `20 reviewer-hours`; adjudication is additional. Sampling fewer cases reduces cost but weakens coverage of rare, consequential failures.

A claim ledger is usually O(number of claims) to maintain, but its value is avoiding repeated investigation after a version or contract change. Store only necessary sensitive evidence; evidence retention itself can create a data boundary.

**Takeaway:** Budget review time and evidence handling as part of the evaluation, not as invisible overhead.

## 10. Key Theorems / Results Reference

| Result / reference | What it supports | Boundary |
|---|---|---|
| [NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf), 2023 | governed risk-management vocabulary | not a candidate certification |
| [NIST SP 800-161r1](https://csrc.nist.gov/pubs/sp/800/161/r1/final), 2022 | supply-chain risk considerations | not evidence that a vendor is secure |
| Binomial count arithmetic | translating a pass-rate gate into allowed failures | not a confidence or generalization guarantee |
| Expected-cost decomposition | comparably accounting fixed/variable/review cost | input estimates can be wrong |

Accessed 2026-11-09. Recheck these sources and any candidate-specific claims at decision time.

**Takeaway:** Framework references guide questions; only candidate-specific artifacts answer them.

## 11. Interview / Exam Questions

1. **What makes a vendor claim testable?** A task, population, window, metric/rubric, comparator, configuration, and pass/fail consequence.
2. **Compute allowed failures for 47 tests and a 90% gate.** `floor(47*0.10)=4`; five failures means 42/47 = 89.36%, below gate.
3. **Why separate safety and quality gates?** A serious prohibited action/data event may be unacceptable regardless of average quality.
4. **What is wrong with a public benchmark as procurement evidence?** Dataset, task, prompt, version, cost, refusal behavior, and operating constraints may not match the local decision.
5. **How do you test a “secure” claim?** Define the applicable threat/control boundary, then request or execute scoped evidence for the actual configuration; do not treat a label as proof.
6. **What is a legitimate reason to retain the baseline?** Candidate evidence is insufficient, a hard constraint fails, or expected benefit is not robust after total cost and harm.
7. **How does retry behavior change a cost model?** It raises effective variable calls/tokens and can amplify latency or duplicate side effects; price and test it.
8. **What evidence proves portability?** A tested or contractually specific export of relevant state/logs and a restore/use demonstration, not a roadmap statement.
9. **A system scores 92% but leaks data twice. What decision follows?** Fail the safety gate, remediate, and retest; do not average the leak into quality.
10. **What should expire in a claim ledger?** Model/version/configuration, contract, data flow, workload, price, control evidence, and ownership assumptions.
11. **Name a plausible wrong acceptance protocol.** Tune prompts on the same set used for final scoring; it contaminates the estimate.
12. **What changes for nonreversible automated action?** Add strict authorization, durable audit, idempotency, approval or segregation, failure recovery, and a narrow rollout.

**Takeaway:** Good due diligence makes rejection, adoption, and “collect more evidence” equally defensible outcomes.

## 12. Quick Reference Card

| Step | Ask / calculate | Required artifact | Stop condition | Next action |
|---|---|---|---|---|
| Interpret | What exact task and population? | claim contract | adjective has no operational meaning | request definition |
| Compare | What baseline on same units? | frozen protocol | incomparable populations | redesign test |
| Gate | Which constraints are non-negotiable? | security/action/data checklist | any hard gate fails | reject or narrow |
| Score | `p=n_pass/n_test` | rubric and scored cases | post-hoc exclusions | rerun frozen test |
| Price | `TCO=F+N*k+H*w+R` | dated cost assumptions | retry/review omitted | revise model |
| Record | What supports and limits claim? | versioned evidence ledger | evidence has no scope/date | mark OPEN |
| Decide | adopt/pilot/baseline/evidence | owner decision record | no named owner or rollback | retain baseline |
