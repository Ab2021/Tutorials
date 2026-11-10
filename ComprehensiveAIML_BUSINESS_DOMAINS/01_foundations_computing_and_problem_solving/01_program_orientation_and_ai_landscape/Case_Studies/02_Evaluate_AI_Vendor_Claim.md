---
case_id: "CASE-01-002"
case_title: "Evaluate an AI Vendor Claim"
case_version: "3.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "AI procurement and evidence evaluation"
primary_pattern: "evidence-to-operation"
secondary_patterns: ["claim-ledger", "capacity-gated pilot", "total-cost comparison"]
evidence_level: "illustrative"
source_basis: "constructed claim packet; no vendor document, price list, or evaluation run exists"
as_of_date: "2026-11-10"
owner_role: "procurement lead"
source_index: "01_program_orientation_and_ai_landscape/Case_Studies/INDEX.md"
review_trigger: "Model version, price schedule, retention terms, evaluation protocol, or intake mix changes"
---

# Evaluate an AI Vendor Claim

**Evidence record.** Every number, document, price, and protocol in this case is ILLUSTRATIVE and constructed for teaching. No vendor was contacted, no price list was read, no evaluation run was executed, and no procurement decision was signed. Where a figure is printed, the arithmetic that produces it is shown and was recomputed by hand; the inputs themselves remain assumptions. Claims about what a vendor "would" do are hypotheses, not observations. Contract, retention, residency, and security questions are explicitly left OPEN because answering them requires an authorized reviewer and current primary documents.

**Implementation stage:** proposed pilot at gate 1 of 4. **Editorial status:** draft. **Decision recorded in this manuscript:** defer for evidence, retain manual review.

## 1. The One-Line Framing

A vendor offers a hosted triage model that it says classifies support documents at 94% accuracy, cuts handling time by 60%, and costs $0.04 per document; the procurement lead has to decide whether that package justifies a reversible pilot, and the honest answer computed below is *not yet*, because the claim's denominator, workload, version, reviewer load, and total cost are all unspecified or unfavourable.

The input is one claim packet: the sentence above, a model identifier, a price schedule, and a supported-task description. The output is a signed decision record with four possible states — reject, defer for evidence, accept a bounded pilot, retain manual review — plus the evidence that would move the state. The beneficiary is the accountable business sponsor; the decision owner is the procurement lead, who may not delegate the recommendation to a vendor-supplied score.

**The framing rule this case teaches:** a capability claim is a *promise about a population*. Until the population, the label, the denominator, and the version are pinned, the percentage in the claim has no operational meaning, and a pilot authorized on it is an unpriced bet on the buyer's own workload happening to resemble the vendor's.

**Decision rule preview.** Accept a pilot only when six predicates hold simultaneously: the claim is scoped to a task taxonomy the buyer actually runs; the acceptance sample is frozen and stratified before outputs exist; every outcome family — match, abstention, error, malformed — is retained in the denominator; the reviewer load implied by abstention and disagreement fits *staffed* hours after a productivity factor; total cost at pilot volume is below the manual baseline it replaces; and a version or price change provably returns the record to review. Five of the six predicates fail or are unverifiable in the packet as received.

## 2. Context & Background

### 2.1 The operating environment

The buying organization runs a document-intake desk. Incoming support artefacts — customer emails, scanned forms, portal submissions, and forwarded attachments — are read by trained reviewers who assign one of three routing labels (billing, technical, account) and record a reason code.

The desk is capacity-bound rather than demand-bound: the arrival rate is roughly stable, the review is manual end to end, and the only lever the operations manager has today is overtime. That is exactly the profile in which a vendor claim about "60% faster handling" sounds decisive and is in fact under-specified, because the desk's real constraint is not keystrokes per document but the number of documents that need a *human* decision after the model has had its say.

Four roles matter, and they hold different powers. The procurement lead owns the recommendation and signs the ledger. The domain review lead owns label quality and can refuse to certify a rubric. The finance partner owns the cost model and refuses unpriced terms. The business sponsor owns the budget and can accept risk, but cannot convert a missing security review into an approval. A fifth actor, the vendor, supplies the packet and — importantly — supplies it on its own terms.

### 2.2 The claim as received, verbatim in structure

The packet contains five statements and omits nine facts that a decision needs. This asymmetry is the case.

| Claim element | What the packet says | What the decision needs | Present? |
|---|---|---|---|
| Capability | "94% accuracy on document triage" | metric definition, label definition, denominator, sampling frame | no |
| Workload | "trained on support documents" | task taxonomy, language, format mix, length distribution, hard-case share | no |
| Baseline | "60% reduction in handling time" | baseline time study, per-document distribution, reviewer seniority mix | no |
| Price | "$0.04 per document" | fixed platform fee, retry billing, egress, support tier, minimum commit, period | partial |
| Version | "our latest model" | immutable version identifier, release date, change notification terms | no |
| Governance | "enterprise-ready" | retention period, subprocessor list, residency, deletion proof, audit rights | no |

The nine omitted facts are not pedantry. Each one is a lever that can move the decision: the workload mix decides whether the sample is representative, the baseline decides whether there is any saving to capture, retry billing decides whether $0.04 is the price, and the version decides whether any measured result survives next month.

### 2.3 Why the desk cannot simply test it

A pilot is not free even when the vendor waives fees. The desk must freeze a sample, blind reviewers, adjudicate disagreements, staff an exception queue, run a finance worksheet, and hold a decision record for audit. In this illustration that overhead is modelled as $1,200 of one-period fixed effort (500 integration, 300 monitoring, 400 contingency). A buyer who runs three unfrozen vendor comparisons in a quarter has spent more reviewer hours on evaluation than the pilot would have saved.

So the correct posture is not "test everything" but "test the claim whose answer changes the decision." A claim about API latency, for instance, is nearly irrelevant here — Section 6.9 computes why.

### 2.4 The proxy problem

The natural proxy is accuracy: one number, easy to compare, easy to put in a slide. The objective is different — it is *decision quality per reviewed document at acceptable cost and risk*. A model with 94% accuracy on easy documents and 71% on the policy-sensitive slice can be worse than a 76% manual baseline that is uniformly mediocre, because the desk's cost is concentrated in the hard slice.

This gap between proxy and objective is the reason the case spends its analysis budget on denominators, strata, paired differences, and reviewer load rather than on the headline number.

### 2.5 Scale and timing

The illustrative pilot is capped at 1,000 documents per period, against an intake of about 2,900 per period, so the pilot is a 34% sample of live traffic and a fully reversible one. Review staffing is three reviewers at 10 available hours each. Event time (when a document arrives) and processing time (when the model answers, when the reviewer rules) are recorded separately throughout, because the queue arithmetic in Section 6.8 depends on that distinction and the retention arithmetic in Section 5d depends on the arrival timestamp alone.

**Takeaway:** the context is not "should we buy AI" but "which unspecified number in a sales sentence would have to be true for a 34% reversible slice of a capacity-bound desk to be worth its own evaluation cost."

## 3. The Business Question

### 3.1 The decision contract

| Field | Contract |
|---|---|
| Question | Under the buyer's own intake mix, does vendor version V produce a decision-quality gain over manual review that survives denominator, stratum, capacity, and total-cost scrutiny? |
| Decision owner | Procurement lead, signing a ledger; review lead certifies labels; finance certifies cost; sponsor owns budget risk |
| Trigger | A purchase-gate date exists, but no date pressure permits skipping a hard gate |
| Options | Reject; defer for evidence; accept a bounded pilot (volume and time capped); retain manual review unchanged |
| Value at stake | Reviewer hours released, routing delay reduced, error cost transferred — all hypotheses until measured |
| Downside | Wrong routing on costly labels, unpriced retries, unstaffed exception queue, lock-in after version drift, terms breach |
| Hard constraints | Human decision authority; reversibility; frozen protocol; named version; no unsupported security or legal assertion |
| Population | Illustrative English-language support documents in three routing labels, formats and lengths as stratified in 5a |
| Out of scope | Production rollout, legal sufficiency, and any statement that one vendor is generally better than another |

### 3.2 Why the objective and the proxy diverge

Three mechanisms separate them, and each has a numerical signature the case will compute.

First, **denominator effects**: dropping abstentions and malformed responses raises reported agreement from 80.0% to 90.8% with no change in behaviour (Section 6.2). Second, **stratum effects**: an aggregate 80.0% can coexist with a policy-sensitive slice far below the baseline, and that slice carries the highest error cost (Section 6.10). Third, **capacity effects**: a model that abstains more can raise reviewer load even as it improves accuracy, so the "automation" claim and the staffing claim point in opposite directions (Section 6.5).

A single scalar cannot carry those three mechanisms. That is why the decision contract lists four options rather than a score threshold.

### 3.3 What a wrong decision costs

Accepting a pilot that should have been deferred costs the evaluation overhead plus the risk of an operational habit forming around an unvalidated path — reviewers start trusting suggestions they cannot audit. Rejecting a vendor that would have worked costs the forgone gain, which in this illustration is bounded by the manual baseline's reviewer cost of $6,000 per period at 1,000 documents, so the upside of a correct accept is at most $6,000 per period minus pilot cost.

The asymmetry matters: a false accept can cost more than the whole plausible gain, because a single systematic routing error at scale is a customer-visible event, while a false reject costs only the measured 3.3 percentage-point difference the pilot would have detected.

### 3.4 The question a sponsor will actually ask

The sponsor asks: "what would you need to see to say yes?" The answer must be a predicate list with numbers, not adjectives. The predicate list used here appears in Section 9.1 and is repeated as a card in Section 15; the interview pack tests whether a candidate can reconstruct it under changed constraints rather than recite it.

## 4. Assumptions & Constraints

### 4.1 Assumption register

| ID | Assumption | Type | Verification method | Status | If wrong | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | Claim names an immutable model version | evidence | capture version string, release date, change-notice clause | unverified | results cannot be attributed or repeated | defer | procurement |
| A-02 | Acceptance sample mirrors intake mix | sampling | compare stratum shares to 30-day intake census | unverified | score is not decision-relevant | resample | evaluation lead |
| A-03 | Labels are adjudicated blind to system identity | measurement | blinded console audit log | proposed | comparison bias inflates the vendor | retain manual baseline | review lead |
| A-04 | Every field used was available before the decision instant | time | per-field first-availability timestamp | proposed | leakage | drop field, rerun | data owner |
| A-05 | Three reviewers, 10 available hours each, 85% productive | operations | staffing plan and time study | unverified | exception queue grows without bound | cap pilot volume | operations |
| A-06 | Price covers retries, egress, support, and minimum commit | commercial | dated quote with billing rules | unverified | total cost understated | defer commercial gate | finance |
| A-07 | Retention and residency terms are acceptable | governance | current primary terms plus authorized review | OPEN | pilot not authorized at all | do not transmit data | security/legal |
| A-08 | Version change is detectable before a request is served | configuration | preflight assertion on every call | proposed | silent upgrade invalidates the run | block run | engineering |
| A-09 | Disagreement and abstention triggers overlap only 20% | operations | logged per-document trigger audit | unverified | reviewer load overstated by 20 items/1,000 | measure the overlap | operations |
| A-10 | Severe routing errors occur at 0.3% and cost $450 each | cost of error | incident ledger from manual process | unverified | net value sign may flip | sensitivity band | finance |
| A-11 | Document arrivals are roughly Poisson within a shift | queue model | inter-arrival histogram | unverified | waiting-time estimate unfounded | use empirical queue | operations |
| A-12 | Manual baseline is 76.7% agreement on the same frozen set | measurement | paired baseline run on identical records | unverified | no measured comparator at all | no comparison, only absolute gate | review lead |
| C-01 | No automated business action is permitted | authority | policy statement | required | unsafe automation | human final action | sponsor |
| C-02 | Pilot volume capped at 1,000 documents per period | scope | ledger counter with hard stop | proposed | unbounded exposure | stop intake | procurement |
| C-03 | Any material version, price, or term change reopens the decision | change control | snapshot hash comparison | required | stale approval persists | re-evaluate | procurement |
| C-04 | Evaluation data may only be used under A-07 clearance | governance | authorization record | required | unauthorized processing | no run | security |

### 4.2 The three assumptions that would reverse the decision

**A-12, the baseline.** If manual review were actually at 61% agreement on the frozen set instead of 76.7%, the vendor's 80.0% would be a genuinely large gain and the review-capacity argument would become the only binding constraint. The decision would move from *defer* to *accept a smaller pilot*, because the evidence question would have been answered by the comparison itself.

**A-05, effective reviewer capacity.** If the desk could staff six reviewers rather than three, the volume cap computed in Section 6.5 would double to roughly 1,416 documents and the pilot could cover real traffic rather than a slice. Capacity is the assumption with the largest single effect on whether the pilot is operationally meaningful.

**A-10, the cost of a severe routing error.** At $450 per severe error the illustrative net value is negative by $210 per period (Section 6.7). At $150 per error the same arithmetic yields +$1,020, and at $900 it yields −$1,560. The sign of the recommendation is therefore carried by an assumption the vendor packet never mentions, which is precisely why it is registered rather than buried.

### 4.3 Non-negotiable constraints and their reasons

Human final authority (C-01) exists because a routing error is a customer-facing event with no automatic remedy, and because the organization cannot yet demonstrate calibration on the costly label. Reversibility exists because the pilot must be able to fail cheaply. A frozen protocol exists because a protocol amended after seeing failures cannot distinguish a capability limit from a measurement choice.

C-04 exists because transmitting documents to a third party before clearance is not a decision the procurement lead is authorized to make, and no amount of favourable accuracy evidence substitutes for that authorization.

### 4.4 Constraints the team controls

The team controls sample size (270 documents were affordable; 1,646 would be needed for the claim's implied effect, Section 6.4), the review rubric wording, the acceptance thresholds, the audit rate, and the pilot's duration. Those are policy choices and are recorded as such in Section 10.2 rather than presented as facts about the vendor.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Specification |
|---|---|
| Unit of analysis | One document-decision pair, keyed by `document_id` plus `evaluation_run_id`; a retried request keeps the same key and a new attempt number |
| Grain caution | The pair is not the customer and not the conversation; a customer with four documents contributes four pairs and must not be counted as one unit |
| Population | Illustrative English-language support documents across three routing labels; declared strata in 5a table below |
| Decision-time inputs | Document text, intake metadata, rubric version, prompt/configuration version, asserted model version, arrival timestamp |
| Reference label | Two blinded reviewers assign a label plus confidence; a third adjudicates only declared disagreements |
| Vendor output | Label, abstention flag, error code, response timestamp, request id, asserted version, reported token counts |
| Join semantics | One run joins one frozen input snapshot to one configuration snapshot; no cross-run joins are permitted in the acceptance calculation |
| Window | Single four-week pilot period; event time and processing time retained separately for every record |
| Not available at decision time | Any label produced after the cutoff, any metadata field completed by later human resolution, and any vendor statement issued after the claim date |

| Stratum | Intake share | Acceptance documents | Error-cost driver | Inclusion rule |
|---|---:|---:|---|---|
| routine routing | 44% | 120 | low per-item cost, high volume | uniform sample of ordinary intake |
| ambiguous requests | 22% | 60 | reviewer time dominates | oversampled 1.5× then reweighted in reporting |
| policy-sensitive | 15% | 40 | high false-negative cost | all documents available in window |
| long documents | 11% | 30 | latency and token cost | length bucket above p90 |
| degraded format | 8% | 20 | input robustness | failures retained, never cleaned |

Stratum shares sum to 100% and sample counts sum to 270. The sample is deliberately *not* proportional: policy-sensitive and degraded-format documents are over-represented because they carry the error cost the decision cares about, and reporting therefore separates per-stratum rates from the weighted aggregate.

### 5b. Tools & Stack

- A frozen-manifest store that refuses to amend a sample after the first output is retrieved; amendments create a new run id.
- A content-hash index that flags near-duplicate documents across strata without removing them, so a duplicate can be reported rather than silently deduplicated.
- A request broker that enforces the volume cap, records the asserted version on every call, and distinguishes a retry of an existing request from a new request.
- A blinded review console that hides system identity and randomizes presentation order to both reviewers.
- An independent metric script that recomputes every published rate from the exported run file; the script is an evidence artifact, not a production service.
- A cost worksheet with one row per cost input, each carrying currency, period, and source date.
- A queue worksheet that converts arrival and service rates into backlog and waiting time, used only for planning.
- A kill switch that halts submissions and preserves the ledger; it is exercised before the first real request, not after the first failure.

### 5c. Method Summary

1. Convert the sales sentence into a claim record with task, slice, metric, comparator, threshold, version, and validity window.
2. Freeze a stratified acceptance sample drawn from the buyer's own intake census, before any vendor output exists.
3. Run the manual baseline and the vendor recommendation against identical frozen records, preserving abstentions, errors, and malformed responses as outcome families.
4. Compute agreement on the full evaluated denominator, then recompute on the response-only denominator to expose the denominator effect.
5. Compare vendor and baseline as *paired* outcomes on the same documents rather than as two independent proportions.
6. Convert model behaviour into reviewer demand, then into staffed hours, then into a volume cap that the pilot must respect.
7. Build total cost from fixed, variable, retry, reviewer, integration, monitoring, and contingency terms on one currency and period.
8. Compute net value under an explicit cost-of-error assumption, and find the error rate at which the sign flips.
9. Apply hard gates in order. A failed gate decides the case regardless of the aggregate metric; no weighting scheme substitutes for it.

| Option | Evidence status | Strength | Limitation | State | Revisit when |
|---|---|---|---|---|---|
| Manual review as-is | baseline protocol defined, not executed | transparent, reversible, no third-party data movement | capacity-bound, reviewer variance | retained comparator | staffing or mix changes |
| Vendor recommendation plus human decision | illustrative protocol | may remove routine effort | needs frozen representative evidence | candidate | gates pass |
| Vendor auto-action | not tested | apparent latency gain | violates C-01 | rejected | only if authority policy changes |
| Rules-only triage | not tested | fully auditable | coverage unknown on degraded format | considered | rules coverage measured |
| Defer all | available fallback | no unsafe action | delay persists | current state | evidence arrives |

### 5d. Data provenance, quality, exclusions and leakage controls

Provenance rules are the part of an evaluation that vendors cannot supply for the buyer, so they are specified here in full.

- Freeze the manifest before requesting outputs; record who selected each stratum, on what census, and on what date.
- Hash every document's normalised text; report duplicates inside and across strata rather than deleting them, because deletion changes the denominator.
- Timestamp the first availability of every metadata field; exclude any field that only exists because a human later resolved the document.
- Keep vendor support staff away from label adjudication and away from selecting which documents enter the acceptance set.
- Preserve timeouts, refusals, malformed responses, and unsupported formats as first-class outcome families with their own counts.
- Separate the prompt/configuration revision from the label rubric revision; a change to either creates a new run id.
- Record every redaction and OCR transformation, since both alter task semantics and comparability.
- Never let a threshold be chosen on the acceptance set; thresholds come from the design set or from policy.
- Retain raw vendor responses only where retention terms permit; otherwise retain derived fields and the response hash.
- Log access to the acceptance set itself, because the vendor seeing items destroys the sample in a way no metric reveals.

### 5e. Reproducibility and method-selection ledger

| ID | Artifact | Version control | Purpose | Access | Reproduction limit |
|---|---|---|---|---|---|
| SRC-01 | Claim packet | date-stamped copy required | scope assertion | not supplied | no primary document exists in this case |
| CFG-01 | Prompt and request configuration | content hash per run | repeat the call | proposed | endpoint behaviour may drift |
| DATA-01 | Frozen manifest | immutable, run-scoped | workload fairness | restricted | not supplied; counts illustrative |
| LAB-01 | Blinded rubric | semantic version | reference label | proposed | reviewer calibration unmeasured |
| RUN-01 | Vendor output export | run id and export hash | metric computation | proposed | no run was executed |
| RUN-02 | Baseline export | run id and export hash | comparator | proposed | no baseline run was executed |
| FIN-01 | Cost worksheet | dated quote plus assumption list | total cost | proposed | prices and usage unverified |
| ORC-01 | Independent metric script | revision | arithmetic check | proposed | verifies declared formula only |
| ORC-02 | Queue worksheet | revision | capacity planning | proposed | fluid model, not a simulation |

**Takeaway:** the method is chosen so that a sceptical reviewer can reproduce the denominator from artifacts, not so that the vendor's best number can be reproduced.

## 6. Analysis Walkthrough

Every figure below is constructed. The arithmetic is shown so it can be recomputed; the inputs are not evidence of anything about any vendor. Two conventions are used throughout: percentages printed to one decimal place, and money in USD at a single illustrative pilot period, never annualised.

### 6.1 What gets counted

The evaluation unit is the document-decision pair. For one illustrative run of 270 acceptance documents, the outcome families are:

| Outcome family | Count | Counts in denominator? | Counts in reviewer load? | Counts in vendor cost? |
|---|---:|---|---|---|
| recommendation matches adjudicated label | 216 | yes | no (unless audited) | yes |
| recommendation differs from label | 27 | yes | yes (disagreement review) | yes |
| model abstains, reviewer decides | 14 | yes | yes | yes |
| response malformed, reviewer decides | 11 | yes | yes | yes |
| request errored then succeeded on retry | 7 retried, 2 failed permanently | yes | yes (both families) | yes, retry billed |
| document excluded before run | reported separately | no | no | no |

The row structure itself resolves the most common procurement dispute. Agreement of 216/270 is 0.8000, or 80.0%. If a vendor reports 216/238 — dividing only by documents that produced a usable response — it reports 0.9076, or 90.8%. Both arithmetic statements are true; they answer different questions. The first answers "how often does the system decide correctly per document submitted," which is what the desk experiences. The second answers "how often is the system right when it answers," which is a conditional quantity that silently deletes the 32 documents the desk still has to handle by hand.

### 6.2 Derivation D1 — the denominator effect

```text
FORMULA [D1]: agreement_full = matches / (matches + mismatches + abstentions + malformed + hard_failures)
              agreement_resp = matches / (matches + mismatches)
              inflation     = agreement_resp / agreement_full - 1
WHERE: all terms are counts of document-decision pairs [count] in one run;
       matches=216, mismatches=27, abstentions=14, malformed=11, hard_failures=2.
CONSTRAINT: abstentions, malformed responses and permanent failures must be classified as outcomes,
            not as missing data; a run where they are "removed for data quality" cannot be compared
            to a baseline that handles those same documents.
NUMERICAL EXAMPLE:
    denominator_full = 216+27+14+11+2 = 270
    agreement_full   = 216/270 = 0.80000 -> 80.0%
    denominator_resp = 216+27 = 243  (used here to isolate the pure response-conditional form)
    agreement_resp   = 216/243 = 0.88889 -> 88.9% on 243 responses
    vendor-style 238-response denominator: 216/238 = 0.90756 -> 90.8%
    inflation vs 80.0%: 0.90756/0.80000 - 1 = 0.13445 -> +13.4% relative
INTERPRETATION: the same behaviour supports an 80.0% or a 90.8% headline depending only on
                which documents the reporter agreed to count; the choice of denominator moves the
                published number by 13.4% relative without a single change in model behaviour.
```

The boundary case is instructive. If abstentions were zero and no response ever failed, the two denominators coincide at 243/243 and the inflation term is exactly zero — denominator discipline buys nothing when behaviour is complete. The formula's value is proportional to the abstention-and-failure rate, so a vendor claiming 99% response completeness makes the distinction harmless; a vendor claiming 94% accuracy while abstaining on 20% of documents makes it decisive.

### 6.3 Derivation D2 — uncertainty and the sample the claim would need

The claim of "94%" invites a comparison that ignores sampling error. Using a normal approximation for a proportion as a *planning* device only:

```text
FORMULA [D2]: se = sqrt(p(1-p)/n) ; half_width = z * se ; interval = p +/- half_width
WHERE: p = observed agreement [fraction], n = evaluated pairs [count], z = 1.96 for a two-sided
       95% display; se in the same units as p (a fraction, dimensionless).
CONSTRAINT: valid for independent, identically distributed Bernoulli outcomes with np>=10 and
            n(1-p)>=10; stratified, clustered, or paired data require a different variance.
NUMERICAL EXAMPLE (p=0.800, n=270):
    p(1-p)    = 0.8*0.2 = 0.16
    se        = sqrt(0.16/270) = sqrt(0.000592593) = 0.024343
    halfwidth = 1.96*0.024343 = 0.047713
    interval  = [0.7523, 0.8477] -> [75.2%, 84.8%]
SAMPLE SIZE REQUIRED TO RESOLVE THE CLAIM's 94% against 80%:
    difference d = 0.94-0.80 = 0.14 ; pooled pbar = 0.87 ; pbar(1-pbar) = 0.1131
    n_per_arm ~ 2*(1.96+0.8416)^2*0.1131/0.14^2
              = 2*7.849*0.1131/0.0196 = 1.7755/0.0196 = 90.6 -> 91 documents per arm
INTERPRETATION: 270 documents is ample to resolve a 14-point gap, but the observed gap is
                3.3 points, not 14; the claim's own number is 13 points away from the measurement
                and the measurement cannot confirm or refute a gap of that size from any single
                stratified slice -- which is why the per-stratum analysis in 6.10 matters more
                than the aggregate interval.
```

Two limits. First, the interval is a display device: it assumes documents are independent draws, but 270 documents from a desk arrive in clusters (one customer, four documents), so the effective sample size is smaller and the interval understates uncertainty. Second, the sample-size line computes the size needed to *detect* the claimed gap, not the size needed to justify a pilot; a pilot whose purpose is to produce a decision can be much smaller, provided its thresholds are set in advance.

### 6.4 Derivation D3 — paired comparison, because both systems saw the same documents

The vendor and the baseline are not two independent samples; they scored the same 270 documents. Treating them as independent wastes the pairing and overstates uncertainty. With 2×2 paired counts:

|  | Vendor right | Vendor wrong |
|---|---:|---:|
| Baseline right | 198 | 9 |
| Baseline wrong | 18 | 45 |

Marginal agreements are 216/270 = 80.0% for the vendor and 207/270 = 76.7% for the baseline, but the *discordant* pairs are only 27 documents (9 + 18), and they split 18 to 9.

```text
FORMULA [D3]: d  = (b - c)/n            (difference in paired proportions)
              se = sqrt(pb + pc - d^2)/sqrt(n)
              z  = d/se ; p_upper = 2*(1 - Phi(z)) for a two-sided display
WHERE: pb = b/n = 18/270, pc = c/n = 9/270, b = baseline-wrong/vendor-right, c = baseline-right/vendor-wrong,
       n = 270 pairs [count]; d, pb, pc are dimensionless fractions.
CONSTRAINT: requires the identical document set under both systems and one adjudicated label per document;
            McNemar's exact form is preferable when b+c < 25.
NUMERICAL EXAMPLE:
    d        = (18-9)/270 = 9/270 = 0.033333
    pb+pc    = (18+9)/270 = 27/270 = 0.100000
    d^2      = 0.001111
    variance = (0.100000 - 0.001111)/270 = 0.098889/270 = 0.000366256
    se       = sqrt(0.000366256) = 0.019138
    z        = 0.033333/0.019138 = 1.742
INTERPRETATION: the paired difference is +3.3 points with a standard error of 1.9 points, so the
                observed gain is under two standard errors from zero; the pilot as sized cannot
                separate "modest real gain" from "no gain", and it certainly cannot separate the
                vendor's claimed 94% from the 80.0% measured.
```

The boundary where the formula misleads: if the two systems are compared by McNemar on the *same* documents but the adjudicated label was itself produced by reviewers who saw the vendor output for those documents, the pairing is contaminated in a direction the variance term cannot detect. Pairing corrects for document difficulty, not for label bias; that is a separate control (Section 5d), and it is the reason blinding is a hard gate rather than a nicety.

### 6.5 Derivation D4 — reviewer demand is the binding constraint, not accuracy

A model that abstains or disagrees hands work back to a human. The planning rule is an additive sum, deliberately conservative:

```text
FORMULA [D4a]: R = N * (a + d + q)
WHERE: R = reviewer items [items/period]; N = submitted documents [documents/period];
       a = abstention rate, d = disagreement rate, q = audit rate [fractions, 0..1].
       Overlap-adjusted form: R_adj = N * (a + d + q - o), o = fraction triggering more than one review.
CONSTRAINT: requires a, d, q measured on the same workload; additive form overstates load whenever
            triggers overlap (an abstention that is also audited is still one review).
NUMERICAL EXAMPLE (N=1,000; a=0.05; d=0.12; q=0.10):
    a+d+q = 0.27 ; R = 1000*0.27 = 270 review items
    at 8 minutes per item: 270*8 = 2,160 reviewer-minutes
    2,160/60 = 36.0 staffed hours
    overlap version (o=0.02): R_adj = 1000*0.25 = 250 items = 33.3 hours, 25 minutes saved
INTERPRETATION: the vendor's accuracy number never enters this formula. Reviewer load is driven by
                the rate at which the system declines or is distrusted, and a model can improve
                accuracy while increasing abstention and therefore increasing load.
```

Effective capacity is lower than paid capacity, and this is where most pilot plans fail:

```text
FORMULA [D4b]: H_eff = reviewers * paid_hours * productive_factor
               N_cap  = (H_eff*60) / ((a+d+q)*m)      [documents per period]
WHERE: H_eff [staffed hours/period], productive_factor [fraction], m [minutes per review item].
CONSTRAINT: N_cap assumes every review item costs m and the queue is staffed across the whole
            period; a desk staffed in one 4-hour block cannot use the whole H_eff.
NUMERICAL EXAMPLE:
    H_eff = 3*10*0.85 = 25.5 staffed hours = 1,530 minutes
    N_cap = 1530/(0.27*8) = 1530/2.16 = 708.3 -> 708 documents per period (round down)
    pilot request of 1,000 documents needs 1,000*0.27*8 = 2,160 minutes = 36.0 h > 25.5 h
    shortfall = 36.0 - 25.5 = 10.5 hours per period
    volume that fits 30 paid hours with no productivity loss: 1800/2.16 = 833.3 -> 833 documents
INTERPRETATION: the pilot cannot be run at the requested 1,000-document volume. The correct response
                is to cap volume at 708 documents, not to assume reviewers will find the time; the
                naive 833 figure is the optimistic variant that ignores the 15% productivity loss.
```

The counterexample where the formula breaks: if the desk's review is done in a single daily block, capacity is not a daily total but a per-block limit. At 3 reviewers working one 4-hour block at 0.85 productivity, block capacity is 3×4×60×0.85 = 612 minutes, which fits only 612/2.16 = 283 documents per block even though the daily total of 1,530 minutes would nominally fit 708. Averaging over the day overstates what can be processed, and the queue arithmetic in 6.8 shows the backlog that results.

### 6.6 Derivation D5 — total cost on one currency and one period

```text
FORMULA [D5]: C_total = F + N*c_var + R*m*c_reviewer + I + M + K
WHERE: F fixed platform fee [currency/period]; c_var variable price per document [currency/document];
       R review items [items]; m minutes per item [minutes/item]; c_reviewer [currency/minute];
       I integration, M monitoring, K contingency [currency/period].
CONSTRAINT: one currency, one period, one tax treatment; retry billed separately must enter as
            N*(1+r)*c_var where r = retry rate, not as a footnote; a minimum commit is a fixed
            cost even at zero volume.
NUMERICAL EXAMPLE (F=2000; N=1000; c_var=0.04; R=270; m=8; c_reviewer=0.75; I=500; M=300; K=400):
    fixed+variable   = 2000 + 1000*0.04 = 2000 + 40 = 2,040
    reviewer         = 270*8*0.75 = 2160*0.75 = 1,620
    overhead         = 500+300+400 = 1,200
    C_total          = 2040 + 1620 + 1200 = 4,860 USD per illustrative pilot period
    unit cost        = 4860/1000 = 4.86 USD per document
    reviewer share   = 1620/4860 = 33.3% ; API share = 40/4860 = 0.82%
INTERPRETATION: at this workload the API price is 0.82% of total cost while reviewer time is 33.3%
                and the fixed plus overhead block is 66.7%; a negotiation that halves the token price
                changes total cost by 20 dollars, while a disagreement rate that doubles changes it
                by more than 600 dollars.
```

Retry billing is the term most often lost, and it compounds with the denominator question:

```text
FORMULA [D5b]: C_retry = N*r*c_var ; C_total_retry = C_total + C_retry
WHERE: r = retry fraction [attempts per document beyond the first, 0..1].
NUMERICAL EXAMPLE: r=0.24 -> 1000*0.24 = 240 extra billed attempts
                   240*0.04 = 9.60 USD per period (1,240 billed attempts vs 1,000 documents)
                   at r=0.24 the variable line becomes 1240*0.04 = 49.60, not 40.00
INTERPRETATION: retries are small in absolute money here but they matter twice: they change the
                price per document actually paid (0.0496), and they change the denominator question
                because a retried document that finally succeeds is one document, not two decisions.
```

The minimum-commit boundary is where this formula stops being proportional. If the vendor's schedule carried a $2,000 monthly minimum, then any volume under 50,000 documents at $0.04 costs the same $2,000; the per-document price becomes a fiction and only the marginal reviewer and overhead terms vary. The pilot does not approach that boundary, which is why volume-based negotiation is nearly pointless at this scale and term-based negotiation is where the money is.

### 6.7 Derivation D6 — net value and the error-rate breakpoint

```text
FORMULA [D6a]: Net = reviewer_time_saved - pilot_cost - severe_errors*error_cost
               reviewer_time_saved = N*m*c_reviewer          (if the pilot replaces manual routing)
               severe_errors       = N*e                    (e = severe-error rate)
WHERE: Net [currency/period]; N [documents]; m [minutes]; c_reviewer [currency/minute];
       e = severe routing error rate [fraction]; error_cost [currency/error].
CONSTRAINT: signed comparison assumes the vendor path fully replaces the manual decision for the
            documents in scope; where reviewers still decide every document (the safe design here),
            reviewer cost does not disappear and the saving collapses to zero minus integration cost.
NUMERICAL EXAMPLE:
    manual baseline reviewer cost = 1000*8*0.75 = 6,000 USD per period
    pilot total cost (D5)         = 4,860 USD per period
    gross difference              = 6,000 - 4,860 = 1,140 USD
    severe errors at e=0.003      = 1000*0.003 = 3 errors
    error cost                    = 3*450 = 1,350 USD
    Net                           = 1,140 - 1,350 = -210 USD per period
INTERPRETATION: the illustrative pilot is net negative by 210 USD per period at a 0.3% severe-error
                rate; the recommendation cannot rest on cost saving alone, and must rest on the
                decision-quality question, which is itself unresolved (D3).
```

The breakpoint is the number worth carrying into the interview:

```text
FORMULA [D6b]: e* = gross_difference / (N*error_cost)
NUMERICAL EXAMPLE: e* = 1140/(1000*450) = 0.002533 -> 0.253% severe-error rate
    sensitivity: at 0.15% -> 1.5 errors -> 675 USD -> Net = +465
                 at 0.253% -> 2.53 errors -> 1,140 USD -> Net = 0
                 at 0.60% -> 6 errors -> 2,700 USD -> Net = -1,560
INTERPRETATION: the sign of the business case flips inside a band of severe-error rates that the
                manual process has never measured. The decision therefore needs an error-rate
                estimate before it needs a better accuracy estimate.
```

The volume breakpoint answers the sponsor's "does this get better at scale" question:

```text
FORMULA [D6c]: Net(N) = N*(m*c_reviewer - c_unit_variable) - F - R(N)*m*c_reviewer_share - overhead
    simplify with per-document variable content of 4.34 USD and fixed 3,200 USD:
    Net(N) = N*4.34 - 3,200
NUMERICAL EXAMPLE: N = 3200/4.34 = 737.33 -> 737 documents gives 737*4.34 = 3,198.58 (Net = -1.42)
                  -> 738 documents gives 738*4.34 = 3,202.92 (Net = +2.92)
                  total cost at 738 = 3200 + 738*1.66 = 3200 + 1,225.08 = 4,425.08
                  cost per document at 738 = 4,425.08/738 = 6.00 USD
INTERPRETATION: against a 6.00 USD manual unit cost, the pilot only beats baseline above roughly
                737 documents per period; the capped 708-document volume from D4b therefore cannot
                pay for itself at all, which is consistent with the negative Net in D6a.
```

Counterexample where D6c's linearity breaks: at 500 documents the same cost structure yields 500*0.04 = 20 variable, 500*0.27 = 135 review items, 135*8*0.75 = 810 reviewer cost, so C_total = 2000+20+810+500+300+400 = 4,030 and cost per document is 8.06, 39.7% above the 1,000-document unit cost. Unit economics improve with volume only while the fixed block is spread; below roughly 250 documents the fixed fee alone exceeds the entire manual reviewer cost for those documents, and the pilot becomes a fixed-cost purchase rather than a per-document one.

### 6.8 Derivation D7 — queue behaviour, which decides latency and backlog

Accuracy and cost are stock measures; the desk experiences flow. With Poisson arrivals and exponential service, the M/M/c waiting-time form gives a planning estimate that the fluid model cannot:

```text
FORMULA [D7a]: rho = lambda/(c*mu) ; Lq = rho^2/(1-rho) for a single server (M/M/1)
               Wq = Lq/lambda
WHERE: lambda = arrival rate [documents/hour]; mu = service rate per reviewer [documents/hour];
       c = number of servers [count]; rho = utilisation [dimensionless, must be <1];
       Lq = expected queue length [documents]; Wq = expected wait [hours].
CONSTRAINT: requires memoryless arrivals and service; at rho >= 1 the queue has no stationary
            distribution and the formula does not apply at all (boundary, not slow growth).
NUMERICAL EXAMPLE (single review stream, lambda=13/h, mu=16/h):
    rho   = 13/16 = 0.8125
    Lq    = 0.8125^2/(1-0.8125) = 0.660156/0.1875 = 3.5208 documents
    Wq    = 3.5208/13 = 0.27083 hour = 16.25 minutes average wait
    utilisation headroom = 1-0.8125 = 18.75%
    at mu=14/h: rho=0.92857 ; Lq = 0.862245/0.071429 = 12.0714 ; Wq = 55.71 minutes
    wait inflation = 55.71/16.25 = 3.43x for a 12.5% service-rate loss
INTERPRETATION: the queue is hyper-sensitive near saturation: losing 2 documents/hour of reviewer
                throughput (a 12.5% capacity loss, e.g. one reviewer on leave) multiplies expected
                wait by 3.4 while utilisation moves from 81% to 93%. This is why the abstention rate
                in D4a is a latency parameter and not just a staffing parameter.
```

The fluid model is the right tool for a surge, and it disagrees with the stationary model by design:

```text
FORMULA [D7b]: B = (lambda - mu_served)*T_surge ; T_drain = B/(mu_drain - lambda_drain)
WHERE: B = backlog accumulated [documents]; T_surge [hours]; T_drain [hours].
NUMERICAL EXAMPLE: lambda=22/h, service=16/h for T=5 h
    B = (22-16)*5 = 30 documents
    after the surge, lambda=10/h and service=16/h: T_drain = 30/(16-10) = 5.0 hours
    if post-surge arrivals were also 16/h: T_drain = 30/0 = undefined -> the queue never drains
INTERPRETATION: a five-hour spike at 6 documents/hour above service capacity leaves 30 documents
                behind and takes another full shift to clear; if the recovery-period arrival rate
                equals service capacity, the backlog is permanent and the desk has silently
                converted a spike into a new baseline.
```

The boundary where both formulas fail is the one this pilot design actually sits on: reviewers do not work as a queue in steady state, they work in one block, so the relevant check is block capacity (283 documents per block from D4b's counterexample) against block arrivals of 22×4 = 88 documents. That comparison shows the block is not overloaded on average, yet the 30-document spike in D7b still appears because arrivals are not uniform inside the block. Neither the stationary nor the fluid model captures intra-block burstiness; only a timestamped arrival histogram does, which is why 5a requires event-time recording.

### 6.9 Latency arithmetic, and why the vendor's speed claim is not decisive

The claim implies reviews will complete in a fraction of the current time. Check the magnitude:

```text
FORMULA [D9]: capacity_minutes_per_period = reviewers*hours*60 ; load_share = latency_minutes/capacity_minutes
WHERE: latency_minutes = total model waiting time in the period [minutes].
NUMERICAL EXAMPLE: 1,000 documents * 1.2 s = 1,200 s = 20.0 minutes of aggregate model wait
    20.0/(2160+20) = 20.0/2180 = 0.917% of the reviewer-and-model time budget
INTERPRETATION: even if model latency doubled to 2.4 s per document, the aggregate wait becomes
                40 minutes, 1.8% of the budget; latency is not where the desk's time goes. The
                decision-relevant speed question is reviewer throughput, not API response time.
```

This is a case where a true claim is irrelevant, which is worth stating plainly because procurement teams routinely spend weeks on it. The vendor's latency number may be accurate and still not move any gate. The interview pack uses this as a discrimination test: a candidate who requests latency evidence but not a paired baseline has misunderstood which uncertainty is decision-relevant.

### 6.10 Stratum reporting and the weighted aggregate

Aggregates can hide the slice that carries the cost. Using illustrative per-stratum vendor and baseline rates with the sample counts from 5a:

| Stratum | n | Vendor agreement | Baseline agreement | Difference | False-negative cost weight |
|---|---:|---:|---:|---:|---:|
| routine routing | 120 | 90.0% | 84.0% | +6.0 pts | 1 |
| ambiguous requests | 60 | 78.0% | 72.0% | +6.0 pts | 4 |
| policy-sensitive | 40 | 71.0% | 79.0% | -8.0 pts | 12 |
| long documents | 30 | 83.0% | 75.0% | +8.0 pts | 2 |
| degraded format | 20 | 65.0% | 70.0% | -5.0 pts | 6 |

Unweighted mean of vendor rates: (90+78+71+83+65)/5 = 387/5 = 77.4%. Unweighted mean of baseline rates: (84+72+79+75+70)/5 = 380/5 = 76.0%. Both are close to the paired totals of 80.0% and 76.7% and both conceal the sign change on the policy-sensitive slice.

Sample-weighted by the acceptance counts: vendor = (0.90*120 + 0.78*60 + 0.71*40 + 0.83*30 + 0.65*20)/270 = (108+46.8+28.4+24.9+13)/270 = 221.1/270 = 81.9%; baseline = (0.84*120+0.72*60+0.79*40+0.75*30+0.70*20)/270 = (100.8+43.2+31.6+22.5+14)/270 = 212.1/270 = 78.6%. Note these are stratum-rate reconstructions and need not equal the paired counts in 6.4 exactly; the discrepancy between 81.9% and the 80.0% paired figure is a rounding-and-weighting artifact that a real report would have to reconcile or explain.

Cost-weighted scoring is the trap. Weighting each stratum rate by its error-cost weight gives vendor (90*1 + 78*4 + 71*12 + 83*2 + 65*6)/(1+4+12+2+6) = (90+312+852+166+390)/25 = 1,810/25 = 72.4%; baseline (84*1 + 72*4 + 79*12 + 75*2 + 70*6)/25 = (84+288+948+150+420)/25 = 1,890/25 = 75.6%. On the measure the desk actually cares about, the vendor is 3.2 points *worse*. This is the single most important table in the case: the decision reverses between the volume-weighted and the cost-weighted view.

### 6.11 The first approach, and why it fails

The tempting approach: compare the vendor's 94% to the desk's reported 76.7%, observe a 17.3-point gap, and authorize a pilot. It fails on four independent grounds, each sufficient:

1. **Different population.** The vendor's number is over a population never described; the desk's is over its own strata, two of which the vendor does not do well on.
2. **Different metric.** Agreement with an adjudicated label is not the vendor's likely metric (accuracy, F1, or a curated "resolution rate").
3. **Different denominator.** No abstention, error, or malformed-response policy is stated, and D1 shows that choice alone moves the headline 13.4% relative.
4. **No capacity or cost term.** The 17.3-point gap says nothing about whether 270 items per 1,000 documents can be reviewed in 25.5 staffed hours; D4b says they cannot.

The counterexample that makes the point sharpest: suppose the vendor genuinely delivers 94% agreement on the desk's own frozen set. At that accuracy the reviewer load term *falls* because disagreement drops, but abstention may not fall — and if abstention rises from 5% to 20%, R becomes 1,000×(0.20+0.06+0.10) = 360 items = 3,240 minutes = 54 hours, which is more than twice the available 25.5. A substantially more accurate system can therefore be operationally infeasible. This is not a paradox; it is the difference between a quality metric and a workload metric.

### 6.12 Stress table: which changed constraint reverses the decision

| Change | From | To | Recomputed consequence | Does the decision move? |
|---|---|---|---|---|
| Severe-error cost | 450 | 150 | Net = 1,140 - 150 = +990 | yes — accept a capped pilot |
| Severe-error cost | 450 | 900 | Net = 1,140 - 2,700 = -1,560 | no — reinforces defer |
| Abandoned trial volume | 1,000 | 500 | cost per document 8.06 vs 4.86; Net negative at any plausible e | yes — pilot is pointless at that size |
| Reviewer headcount | 3 | 6 | N_cap = 708 → 1,416 documents | yes — pilot can cover live traffic |
| Abstention rate | 5% | 20% | R = 360 items = 54 hours vs 25.5 available | no — becomes infeasible even if accuracy improves |
| Vendor version | V1 pinned | V2 shipped silently | no comparable evidence at all | yes — REVIEW_REQUIRED, evidence void |
| Audit rate | 10% | 30% | R = 1,000×0.47 = 470 items = 62.7 hours | no — infeasible, cap volume to 406 |
| Retained baseline accuracy | 76.7% | 61.0% | paired gap becomes ~19 points | yes — accept, baseline is the weak path |

Reading the table is the transferable skill. Four of the eight changes are about the buyer's own operating parameters, not about the vendor at all — headcount, audit policy, error cost, and baseline quality. A procurement process that only interrogates the vendor will miss most of the levers that determine the outcome.

**Takeaway:** the walkthrough converts one sales sentence into seven checked derivations, and the honest conclusion is that the decision is dominated by denominator discipline, reviewer capacity, and an unmeasured error cost — none of which the vendor's claim addresses.

## 7. The Finding

The finding is a decomposition, not a verdict, and it has six parts that a sponsor can check independently.

**First, the claim as written is not evaluable.** It lacks a task taxonomy, a label definition, a denominator policy, an immutable version, a baseline definition, and any statement about abstention or retries. Four of the six predicates in Section 1 fail on completeness alone, before any arithmetic is done.

**Second, when the missing pieces are supplied with illustrative values, the aggregate gain is small and statistically unstable.** 216/270 = 80.0% against 207/270 = 76.7% is a paired difference of +3.3 percentage points with a standard error of 1.9 points, so the observed gain is 1.74 standard errors from zero. A pilot of this size cannot distinguish a modest genuine gain from no gain.

**Third, the aggregate conceals a sign reversal on the slice that matters.** On the policy-sensitive stratum the vendor is 8 points *worse* than the manual baseline, and weighting strata by error cost puts the vendor at 72.4% against the baseline's 75.6%. Volume-weighted and cost-weighted views point in opposite directions, and the desk's cost lives in the cost-weighted view.

**Fourth, the operation cannot absorb the volume at which the pilot would be meaningful.** Reviewer demand is R = 1,000 × 0.27 = 270 items = 36.0 staffed hours against 25.5 effectively available hours, a 10.5-hour shortfall. The volume that fits is 708 documents per period; at that volume the cost structure yields a unit cost above the manual baseline, so the pilot cannot pay for itself.

**Fifth, the economics are carried by an unmeasured parameter.** Gross difference is $1,140 per period and the severe-error breakpoint is 0.253%. Below 1.5 severe errors per 1,000 documents the pilot is net positive; at the illustrative 0.3% it is net negative by $210. Nobody has measured the manual process's severe-error rate, so the sign of the business case is currently unknown.

**Sixth, no hard constraint is satisfied by evidence.** Security and retention terms are OPEN (A-07), the version is not pinnable (A-01), and the label blinding is proposed rather than executed (A-03). These are disqualifiers for *authorization*, independent of the metrics.

The decision that follows is **defer for evidence, retain manual review**, with a specific evidence request. Deferral here is not indecision: it is a decision whose deliverable is a scoped, costed evidence request that names the seven artifacts in Section 5e and the four measurements that would move the state. Rejection would also be defensible — the packet is poor enough that a case can be made for closing it — but rejection discards the possibility that a redesigned pilot answers the question, and the cost of one more evidence cycle is bounded by the $1,200 fixed evaluation block.

What the finding explicitly does **not** say: that the vendor is dishonest, that the model is bad, that 94% is false, or that a pilot would fail. None of those is supported. The finding is that the available evidence cannot support an authorization, and that the specific things missing are known and requestable.

## 8. System/Solution Design

### 8.1 Architecture implied by the decision

The design is a governed evaluation instrument, not a production integration. Each component exists to make one gate auditable.

```text
[claim packet]-->(1) scope & version validator-->(2) frozen manifest-->(3) request broker--+
                                                                                          |
[manual baseline]-->(4) blinded review + adjudication-->(5) metric oracle-->(6) cost model-+
                                                                                          |
                                                                     (7) gate evaluator    |
                                                                                          v
                                                                    (8) decision ledger --+--> accept / defer / reject
                                                                                          |
                                                                    (9) kill switch ------+--> [manual review queue]
```

Nine components, each with a single owner and a refusal behaviour:

**(1) Scope and version validator.** Rejects any claim packet lacking task mapping, metric definition, denominator policy, and an immutable version identifier. Its refusal is a *defer*, never a warning. It writes the packet hash into the ledger so a later reading can prove the claim did not change under it.

**(2) Frozen manifest.** Immutable after the first output is retrieved. Holds the stratified document ids, the census date they were drawn from, the inclusion rule per stratum, and the exclusion reason codes. Amendment creates a new run id rather than editing the manifest, which is the mechanism that makes "the vendor saw our sample and we resampled" impossible to do silently.

**(3) Request broker.** Enforces the volume cap (C-02), asserts the version on every call (A-08), and separates `request_id` from `attempt_number` so retries cannot inflate the denominator (D1) or the cost line (D5b). It fails closed: an unassertable version blocks the call.

**(4) Blinded review and adjudication.** Hides system identity from the reviewer display, randomizes order, records per-reviewer elapsed time (the input to D4's minutes-per-item), and routes only declared disagreements to a third adjudicator.

**(5) Metric oracle.** Deterministic recomputation from the exported run file: per-stratum rates, the full and response-conditional denominators, the paired 2×2 table, and the paired standard error. It refuses to produce a headline number without the outcome-family counts beside it.

**(6) Cost model.** One row per cost input with currency, period, and source date. A missing row is a hard error in the model, not a zero.

**(7) Gate evaluator.** Applies the six predicates in fixed order and short-circuits on a hard failure. Order matters: cost and capacity gates are cheap to evaluate and can disqualify early, whereas the paired comparison is expensive.

**(8) Decision ledger.** Records state, actor, timestamp, rationale, gate results, and the version/price/term snapshot hashes. It is append-only; a correction is a new entry pointing at the superseded one.

**(9) Kill switch.** Halts new submissions, preserves the ledger, and returns the queue to manual review. It is exercised *before* the first production request, as a rehearsal, because a kill switch first tested during an incident is a hypothesis.

### 8.2 The four stores

| Store | Content | Write authority | Retention | Failure consequence |
|---|---|---|---|---|
| Claim store | packet text, version string, price schedule, terms | procurement only | decision + 3 periods | unevaluable claim, no attribution |
| Sample store | manifest, census snapshot, exclusion codes | evaluation lead, append-only | run-scoped, immutable | denominator drift |
| Run store | requests, attempts, responses, timings, exports | broker only | per retention terms | no reproduction, no audit |
| Decision store | gate results, state transitions, signatures | procurement lead only | audit policy | undefendable decision |

Separating the stores is what makes the "who changed what" question answerable. A single database with one write role cannot distinguish a vendor-driven change from an internal one.

### 8.3 Request lifecycle including the failure path

```text
RECEIVED -> VALIDATED(scope, version) -> QUEUED -> SENT -> {MATCH | MISMATCH | ABSTAIN | MALFORMED | ERROR}
                                                            |
                                             ERROR -> RETRY(<=2) -> {RESPONSE | HARD_FAIL}
                                                            |
                            all outcomes -> REVIEW_ITEM(if abstain/mismatch/audit) -> ADJUDICATED
                                                                                        |
                                                              -> LEDGER_ENTRY -> GATE_EVALUATION
```

Invariants worth stating because they are the ones broken in practice:

- A document may have many attempts but exactly one terminal outcome family per run.
- An abstention is a terminal outcome, not a pending state; it must not be re-requested until it becomes a match.
- A hard failure after the retry budget is a denominator member, not an exclusion.
- No ledger entry exists without an adjudicated label for the document it references.
- The gate evaluator never reads the run store directly; it reads the oracle's output, so arithmetic cannot be recomputed differently per reader.

### 8.4 Offline and online paths

The pilot has two paths with different guarantees. The **offline** path (evaluation runs) may use a frozen export, batched requests, and delayed adjudication; its output is evidence. The **online** path (if a pilot proceeds) must assert version per request, respect the volume cap in real time, and fall back to manual review on any broker refusal. Confusing the two is a common failure: numbers from the offline path get quoted as if they described online behaviour, and the offline path had 30 days of adjudication the online path will never have.

The online path additionally needs a queue-depth guard: if review backlog exceeds a threshold (say 60 documents), the broker stops submitting new documents rather than growing the backlog (D7b shows the non-drain condition).

### 8.5 Design choices and rejected alternatives

| Decision | Chosen | Rejected alternative | Why rejected |
|---|---|---|---|
| Sample selection | stratified, frozen, buyer-drawn | vendor-supplied evaluation set | vendor controls difficulty and can seed test items |
| Denominator | all submitted documents | successful responses only | D1: inflates headline by 13.4% relative |
| Comparison | paired 2×2 on identical documents | two independent proportions | wastes pairing, overstates uncertainty |
| Reviewer display | blinded to system identity | show source to "add context" | label contamination cannot be corrected later |
| Threshold setting | design set or policy, before the run | tuned on acceptance set | turns measurement into fitting |
| Cost basis | fixed + variable + retry + reviewer + overhead | per-token price only | API is 0.82% of total cost here |
| Volume | capped at staffed capacity | start at 1,000 and hire if needed | hiring is not reversible within the pilot |
| Version | pinned, asserted per call | trust product name | silent upgrade voids comparability |

Two of those choices cost real money and are worth defending explicitly. Freezing the sample means the pilot cannot start until the census is drawn, which may delay it two weeks. Blinding means reviewers cannot see vendor explanations, which they will argue makes them slower — and that argument may be correct, which is why elapsed time is recorded so the claim can be tested rather than asserted.

### 8.6 Operational readiness

Before the first real request, six things must be demonstrated, not planned: the kill switch halts submissions within one request; the fallback queue accepts a synthetic document and routes it to a named reviewer; the version assertion fires on an injected mismatch; the volume counter stops at the cap; the oracle reproduces a hand-computed rate to four decimal places; and the ledger rejects an unsigned transition. Each is a five-minute test with a binary result, and each has failed in real deployments at least once.

## 9. Evaluation & Validation

### 9.1 The six acceptance predicates

| # | Predicate | Numerator | Denominator | Population | Window | Comparator | Guardrail | Uncertainty | Status |
|---|---|---|---|---|---|---|---|---|---|
| P1 | Claim completeness | mandatory fields present | mandatory fields required (7) | claim packet | claim date | schema | any missing field = defer | none needed | not executed |
| P2 | Workload fairness | stratum counts in manifest | stratum counts in census | acceptance set | census window | declared intake mix | share deviation ≤5 pts per stratum | binomial on shares | not executed |
| P3 | Decision relevance | paired discordant gain | evaluated pairs | full acceptance set | one run | manual baseline, same documents | cost-weighted rate must not be lower than baseline | paired SE (D3) | not executed |
| P4 | Capacity fit | required review minutes | effective staffed minutes | pilot volume | one period | cap in C-02 | shortfall ≤0 | queue model (D7) | not executed |
| P5 | Cost fit | total cost (D5) | manual baseline reviewer cost | 1,000 documents | one period | baseline | unit cost must be under baseline | sensitivity on c_var, r | not executed |
| P6 | Reversibility | demonstrated rollback steps | required rollback steps (6) | pilot workflow | rehearsal | readiness checklist | every step binary-pass | none | not executed |

None of these is executed. The words *not executed* are the honest status; the design is complete and untested, which is a legitimate state for a proposed pilot, and printing "passed" here would be the exact defect this manuscript exists to avoid.

### 9.2 Guardrails and stop conditions

| Guardrail | Threshold | Measured how | Action on breach | Reset condition |
|---|---|---|---|---|
| Volume cap | 1,000 documents/period | broker counter | stop submissions | new period or new approval |
| Review backlog | 60 documents | queue depth | pause submissions | backlog < 20 |
| Severe routing errors | 3 in 1,000 | incident ledger | pause, escalate to review lead | root cause recorded |
| Abstention rate | 10% | run store | pause; capacity recalculated | abstention < 7% for one period |
| Malformed response rate | 3% | oracle | pause; version and format check | two consecutive clean periods |
| Version mismatch | any | broker assertion | block run | pinned version restored |
| Reviewer elapsed time | >12 min/item | review console | recalibrate rubric or cost model | time study agrees within 20% |
| Decision age | 90 days | ledger timestamp | mandatory re-review | new evidence recorded |

The stop conditions are deliberately set *below* the levels at which the illustrative arithmetic turns negative, so the pilot stops before it can accumulate an unaffordable backlog. Setting a guardrail at the exact break-even point means every breach is also a financial loss.

### 9.3 How many observations are needed

Four different questions require four different sample sizes, and conflating them is a standard error.

```text
FORMULA [N1]: n = z^2*p*(1-p)/E^2        (precision of a single proportion)
WHERE: E = desired half-width [fraction]; z = 1.96 for 95% display.
NUMERICAL EXAMPLE (p=0.80, p(1-p)=0.16):
    E=0.05 -> n = 3.8416*0.16/0.0025 = 245.86 -> 246 documents
    E=0.04 -> n = 3.8416*0.16/0.0016 = 384.16 -> 385 documents
    E=0.03 -> n = 3.8416*0.16/0.0009 = 682.95 -> 683 documents
INTERPRETATION: the 270-document acceptance set gives roughly a +/-5 point half-width on the
                aggregate, which is why D2's interval was about 9.5 points wide.
```

```text
FORMULA [N2]: n_stratum = z^2*p(1-p)/E^2  with p=0.85, E=0.05
NUMERICAL EXAMPLE: p(1-p)=0.1275 ; n = 3.8416*0.1275/0.0025 = 195.92 -> 196 per stratum
    available for policy-sensitive: 40 -> half-width actually achieved = 1.96*sqrt(0.1275/40)
                                    = 1.96*0.056458 = 0.11066 -> +/-11.1 points
INTERPRETATION: the policy-sensitive slice cannot support a +/-5 point claim at 40 documents; any
                stratum statement must carry its own interval, and +/-11 points overlaps zero on
                the observed -8 point difference.
```

```text
FORMULA [N3]: n_pairs = (z_alpha*sqrt(pb+pc) + z_beta*sqrt(pb+pc-d^2))^2 / d^2   (paired design)
NUMERICAL EXAMPLE (pb=0.0667, pc=0.0333, d=0.0333, alpha=0.05, power=0.80):
    sqrt(0.1000)=0.31623 ; sqrt(0.09889)=0.31447
    numerator = (1.96*0.31623 + 0.8416*0.31447)^2 = (0.61981+0.26464)^2 = 0.88445^2 = 0.78225
    n = 0.78225/0.001111 = 704 pairs
INTERPRETATION: detecting a 3.3-point paired gain at 80% power needs about 704 documents, not 270;
                the pilot as sized is underpowered by a factor of 2.6 for the effect it observed.
```

The fourth question is the zero-event one: how many clean audits before a defect class can be declared rare? For a rule-of-three style bound with 95% confidence and zero observed events, n ≈ 3/rate; at a target rate of 0.3% that is 1,000 documents, which happens to equal the pilot cap. That coincidence is why the volume cap and the error-rate target should be chosen together rather than independently.

### 9.4 The evaluation protocol as a sequence

1. Packet intake and completeness check (P1). Refuse to proceed on a missing field.
2. Census extraction and manifest freeze (P2 inputs). Record the census date.
3. Baseline run on the frozen set, blinded. Produce RUN-02.
4. Vendor run under the broker, capped, version-asserted. Produce RUN-01.
5. Disagreement adjudication; produce final labels and per-reviewer timings.
6. Oracle recomputation: outcome families, full and conditional denominators, paired table, per-stratum rates, cost-weighted rates.
7. Cost model completion with dated inputs; sensitivity on retries and error rate.
8. Capacity and queue check against staffed hours.
9. Gate evaluation in fixed order; record each result with its artifact hash.
10. Ledger signature and state transition; schedule expiry.

The protocol has a property worth naming: every step produces an artifact that a later auditor can hash. A protocol whose intermediate results exist only in a slide is not reproducible, and the common failure of re-litigating an old decision is caused by exactly this absence.

### 9.5 What would falsify the recommendation

| Falsifier | Observation that would overturn "defer" | Where it would appear |
|---|---|---|
| Baseline is weaker than assumed | paired baseline agreement near 61% on the same frozen set | RUN-02 export |
| Error cost is lower than assumed | severe-error ledger shows cost below $150 and rate below 0.15% | incident ledger |
| Capacity is larger | staffing plan authorizes six reviewers at 0.85 productivity | operations plan |
| Abstention is much lower | broker-logged abstention below 2% at the claimed accuracy | run store |
| Cost is much lower | dated quote with no fixed fee and retry-inclusive pricing | FIN-01 |
| Policy-sensitive slice is fine | stratum rate within 2 points of baseline on 196+ documents | oracle stratum report |

Any one of the first four would move the state to *accept a bounded pilot*, subject to the governance gates. The fifth and sixth would strengthen the case but not decide it, because capacity and authority still bind.

### 9.6 What the numbers do not mean

- 80.0% is not the vendor's accuracy; it is agreement with one adjudicated rubric on one constructed sample.
- 3.3 percentage points is not a causal effect of the vendor; the baseline and vendor were measured, not randomized.
- $4,860 is not a quote, a market price, or a saving; it is one arithmetic result from constructed inputs.
- 708 documents is not a recommendation to buy 708 documents; it is the volume at which the illustrative assumptions exactly saturate staffed capacity.
- 0.253% is not a measured error rate; it is the rate at which this cost structure breaks even.

**Takeaway:** a validation plan is complete when it names, for each predicate, what artifact would prove it, and honest when it states that none has been produced.

## 10. Action Taken

### 10.1 What the procurement lead does

The lead records a **defer** and issues a four-part evidence request, each part with an owner and a due date expressed in working days from the request:

| Request | Content | Owner | Due | Blocking? |
|---|---|---|---|---|
| R1 | Immutable version identifier, release date, change-notification clause | vendor via procurement | 5 days | yes |
| R2 | Dated price schedule covering retries, minimum commit, egress, support | vendor via finance | 10 days | yes |
| R3 | Written metric definition, label definition, denominator policy, exclusions | vendor via evaluation lead | 10 days | yes |
| R4 | Retention, residency, subprocessor, deletion-proof terms for authorized review | vendor via security | 15 days | yes |

Four blocking requests, not fourteen. The discipline is that only facts which can change the gate result are requested; a request for "more benchmarks" would produce more unevaluable numbers.

### 10.2 The decision record as an auditor would want it

| Field | Value |
|---|---|
| Record id | `DEC-CASE-01-002-0001` |
| State | `EVIDENCE_REQUESTED` |
| Decision | defer; retain manual review; pilot not authorized |
| Actor and role | procurement lead (signature required) |
| Timestamp | authoring date of this manuscript; no live event exists |
| Rationale | four of six predicates fail on completeness; arithmetic on illustrative inputs shows a negative net at the illustrative error cost and a capacity shortfall |
| Gate results | P1 fail (7 fields missing), P2 not executed, P3 not executed, P4 fail on illustrative demand, P5 fail on illustrative cost, P6 not executed |
| Snapshot hashes | claim packet hash, price schedule hash, terms hash — all required before the state can move |
| Expiry | 90 days from signature, or immediately on any version, price, or term change |
| Supersedes | nothing; this is the first record |

The record contains no percentage, and that is deliberate. The state transition is justified by predicate failures and by a cost-capacity mismatch, both of which are structural; the 80.0% figure would invite an argument about sampling that does not change the state.

### 10.3 What was explicitly not done

No vendor was contacted. No data was transmitted. No request was made to any endpoint. No price was negotiated. No threshold was agreed with the vendor. No contract language was reviewed. No production system was touched. No reviewer time was committed. These omissions are the content of the record: a decision not to act is only auditable if it names what was avoided.

### 10.4 The alternative action seriously considered

**Reject outright.** The packet is poor enough that rejecting saves the $1,200 evaluation block and closes the file. It was rejected as an action for three reasons: the evidence request is cheap relative to one pilot period; a rejection without a stated resubmission condition invites the same packet to return through another channel; and the vendor may hold evidence (an immutable version string, a retry-inclusive price) that costs nothing to disclose.

**Accept a small pilot anyway.** Considered and rejected: at 200 documents the fixed cost dominates (unit cost above $8), the strata are too small to report separately, and the pilot would produce an anecdote rather than evidence. A pilot smaller than its own analysis floor is a demonstration, not a test.

## 11. Outcome & Results

**Outcome status: not measured.** No pilot ran, so no saving, latency, accuracy, security posture, or adoption claim exists. The only verified statements are arithmetic derived from declared illustrative inputs, and they are enumerated below so a reader can check them without re-reading the walkthrough.

| ID | Derived quantity | Inputs | Arithmetic | Value | Validity condition |
|---|---|---|---|---|---|
| R-01 | Full-denominator agreement | 216 matches of 270 pairs | 216/270 | 80.00% | all outcome families counted |
| R-02 | Response-conditional agreement | 216 of 238 responses | 216/238 | 90.76% | abstentions and failures dropped |
| R-03 | Denominator inflation | R-01, R-02 | 90.76/80.00 − 1 | +13.4% relative | same behaviour, different counting |
| R-04 | Baseline agreement | 207 of 270 | 207/270 | 76.67% | identical frozen set, blinded |
| R-05 | Aggregate planning interval | p=0.80, n=270 | 0.80 ± 1.96·0.024343 | [75.2%, 84.8%] | independent Bernoulli outcomes |
| R-06 | Paired difference | b=18, c=9, n=270 | 9/270 | +3.33 pts | one label per document, same set |
| R-07 | Paired standard error | pb, pc, d | sqrt((0.1−0.001111)/270) | 1.914 pts | discordant count above 25 |
| R-08 | Paired z | R-06, R-07 | 3.333/1.914 | 1.74 | two-sided display only |
| R-09 | Review items per 1,000 | a=0.05, d=0.12, q=0.10 | 1000×0.27 | 270 items | triggers treated as additive |
| R-10 | Reviewer minutes | R-09, m=8 | 270×8 | 2,160 min = 36.0 h | 8 min/item measured, not assumed |
| R-11 | Effective staffed hours | 3 reviewers, 10 h, 0.85 | 3×10×0.85 | 25.5 h = 1,530 min | productive factor applies uniformly |
| R-12 | Capacity-limited volume | R-10, R-11 | 1530/2.16 | 708 documents | whole-period staffing |
| R-13 | Total illustrative cost | D5 inputs | 2,040+1,620+1,200 | $4,860 | one currency, one period |
| R-14 | Unit cost at 1,000 | R-13 | 4860/1000 | $4.86 | fixed block fully absorbed |
| R-15 | Unit cost at 500 | recomputed | 4030/500 | $8.06 | fixed block half-absorbed |
| R-16 | Reviewer share of cost | R-13 | 1620/4860 | 33.3% | 8 min/item holds |
| R-17 | API share of cost | R-13 | 40/4860 | 0.82% | retries excluded from this line |
| R-18 | Gross difference vs baseline | 6000−4860 | — | $1,140/period | pilot replaces manual routing |
| R-19 | Net at 0.3% severe errors | 1140−1350 | — | −$210/period | $450 per severe error |
| R-20 | Severe-error breakpoint | 1140/(1000×450) | — | 0.253% | linear error cost |
| R-21 | Volume break-even | 3200/4.34 | — | 737 documents | constant variable content |
| R-22 | Queue wait, λ=13, μ=16 | ρ=0.8125 | 0.66016/0.1875/13 | 16.25 min | M/M/1, memoryless |
| R-23 | Queue wait, μ=14 | ρ=0.92857 | 0.86224/0.07143/13 | 55.71 min | same model, 12.5% slower |
| R-24 | Surge backlog | (22−16)×5 | — | 30 documents | constant rates in surge |
| R-25 | Drain time | 30/(16−10) | — | 5.0 hours | recovery rate exceeds arrivals |
| R-26 | Model wait share of budget | 20/(2180) | — | 0.92% | 1.2 s per document |
| R-27 | Retry-billed attempts | 1000×1.24 | — | 1,240 attempts | retries billed separately |
| R-28 | Cost-weighted vendor rate | 1810/25 | — | 72.4% | five strata, illustrative weights |
| R-29 | Cost-weighted baseline rate | 1890/25 | — | 75.6% | same weights |
| R-30 | Pairs needed for 80% power | paired design | 0.78225/0.001111 | 704 pairs | effect as observed |

Three observations about the table itself. First, every row states a validity condition, because a number without one is a rumour. Second, the rows that carry the decision (R-12, R-19, R-20, R-28) are the ones whose inputs are least certain — capacity, error cost, error weight — which is the normal situation in procurement and the reason sensitivity analysis is not optional. Third, some rows will look precise to four significant figures; that precision is arithmetic, not measurement, and the honest report quantifies the arithmetic while labelling the inputs.

| Claim ID | Exact scoped claim | Supported by | Does not establish | Recheck trigger |
|---|---|---|---|---|
| CLM-01 | A frozen stratified manifest makes the denominator reproducible | design | representativeness of any real intake | census or manifest change |
| CLM-02 | Agreement on the illustrative acceptance set is 80.0% on 270 pairs | R-01 | vendor accuracy in production | any input change |
| CLM-03 | The paired gain is 3.33 points ± 1.91 points | R-06, R-07 | causality or superiority at 95% | larger paired run |
| CLM-04 | Reviewer demand is 36.0 staffed hours per 1,000 documents | R-10 | actual staffing need | time study |
| CLM-05 | Illustrative total cost is $4,860 with API at 0.82% | R-13, R-17 | market price or saving | dated quote |
| CLM-06 | The cost-weighted rate reverses the aggregate ordering | R-28, R-29 | that weights are correct | owner-approved weights |
| CLM-07 | A version change voids prior evidence | design invariant | quality decline | version policy change |

**Takeaway:** the deliverable of a deferred procurement decision is a table of derivations with validity conditions and a signed record naming what was avoided — not a percentage.

## 12. Failure Modes & What Went Wrong

This section records the failure mechanisms that actually corrupt vendor evaluations. Each entry states the mechanism, the competing cause it is confused with, the check that discriminates between them, and the correction. None of these is a measured event in this case; they are the failure catalogue that the design in Section 8 exists to prevent.

### FM-01 — Denominator laundering through "data quality"

**Mechanism.** Abstentions, timeouts, and malformed responses are declared data-quality problems and removed before scoring. The reported rate rises with no behavioural change: 216/238 = 90.8% instead of 216/270 = 80.0%, a 13.4% relative inflation (D1).

**Competing cause.** Genuine data corruption, where a document truly cannot be scored by either system and removing it is correct. The distinction is whether the same removal is applied to the baseline: if the manual baseline still has to handle those 32 documents, they are workload, not noise.

**Discriminating check.** Recompute both rates from the same run file and compare outcome-family counts. If the removed documents exist in the baseline denominator but not the vendor's, the asymmetry is the proof.

**Correction.** Publish the full-denominator rate as the headline and the conditional rate as a labelled secondary figure. Freeze the exclusion list before the run so it cannot be authored after seeing results.

**Verification.** Oracle recomputation on the exported file must reproduce both rates exactly; a run whose family counts do not sum to the submitted total is rejected outright.

### FM-02 — Convenience sampling dressed as representativeness

**Mechanism.** The acceptance set is drawn from documents that are easy to route — short, clean, one-label — because those are easy to label and quick to run. The resulting sample is not the intake, and the difference is systematic rather than random: degraded-format documents are 8% of intake and roughly 0% of the sample.

**Competing cause.** Legitimate stratification that deliberately over-weights hard cases, which produces a *non-proportional* sample on purpose and reweights at report time. That is defensible; silent convenience sampling is not.

**Discriminating check.** Compare sample shares per stratum against a dated intake census. Over-weighting hard cases shows hard strata above intake share *and* a documented reweighting rule; convenience sampling shows hard strata absent.

**Correction.** Draw from the census with an explicit inclusion rule per stratum, record census date, and state the reweighting formula. Never let the vendor select the set.

**Verification.** Manifest-to-census share table with per-stratum deviation; a deviation above 5 points without a reweighting rule blocks the run.

### FM-03 — Test-set exposure to the vendor

**Mechanism.** Documents from the acceptance set, or their near-duplicates, reach the vendor before the formal run — through a support conversation, a "trial", or a shared folder. The measured rate then reflects memorisation or prompt-specific tuning rather than capability on unseen input.

**Competing cause.** Legitimate prompt engineering on a *separate* design set, which is normal practice and improves the result honestly. The failure is only exposure of the acceptance set.

**Discriminating check.** Access logs on the acceptance store plus a content-hash comparison between any document the vendor has seen and the acceptance manifest; also compare agreement on seen versus unseen subsets.

**Correction.** Invalidate the exposed set entirely — not partially — and draw a new one from the same census with a new run id.

**Verification.** Post-run hash audit showing zero intersection between vendor-accessed documents and the acceptance manifest.

### FM-04 — Label contamination by the system under test

**Mechanism.** Reviewers see the vendor's suggested label or explanation while adjudicating the reference label. Anchoring shifts the reference toward the vendor, so the paired comparison measures agreement between two copies of the same suggestion.

**Competing cause.** Reviewer fatigue or rubric ambiguity, which also degrades labels, but degrades them randomly and both arms equally.

**Discriminating check.** Compare the vendor arm's label distribution with the baseline arm's on the same documents. Contamination compresses the difference; a random-error story widens intervals without shifting the mean. Second check: disagreement rate between the two primary reviewers should be similar across arms.

**Correction.** Blind the console, randomize order, and re-adjudicate any document adjudicated in a non-blinded session.

**Verification.** Console audit log proving identity was hidden for every adjudicated document; any session without the flag is re-run.

### FM-05 — Accuracy reported, workload ignored

**Mechanism.** The system abstains at a high rate or requires confirmation on most documents, so the "automation" claim coexists with reviewer load at or above the manual baseline. The arithmetic: N=1,000 with a=0.20, d=0.06, q=0.10 gives R=360 items = 54 hours against 25.5 available.

**Competing cause.** Legitimate safety abstention on a genuinely ambiguous slice, which is a *feature* but still consumes the same reviewer minutes. The failure is not abstaining; it is quoting the accuracy gain while omitting the derived load.

**Discriminating check.** Compute R from logged behaviour rather than from the vendor's claims, then compare to staffed minutes. If R×m exceeds effective minutes, the configuration is infeasible regardless of accuracy.

**Correction.** Cap volume at capacity (708 documents here), or negotiate a lower abstention target, or staff more reviewers — and state which.

**Verification.** Broker-logged abstention and disagreement rates over one period, reconciled to the review console's item count.

### FM-06 — Version drift between claim and run

**Mechanism.** The vendor's "latest model" changes between the claim date and the run, or during the run. Results are then attributed to a product name rather than a version, and the approval survives into an environment it never measured.

**Competing cause.** Planned version upgrade with notification and a re-run, which is normal release management. The failure is silent drift with no re-evaluation.

**Discriminating check.** Per-request asserted version compared to the pinned value; a single mismatch voids the run. Log the vendor's release notes date against the run date.

**Correction.** Pin the version contractually, assert per request, and route any change to `REVIEW_REQUIRED` with a frozen re-run.

**Verification.** Broker log showing 100% version assertion coverage and zero mismatches across the accepted run window.

### FM-07 — Price that excludes retries, egress, and the minimum commit

**Mechanism.** The quoted per-document price is applied to the document count, omitting billed retries, egress, support tier, and a minimum monthly fee. Here retries at 24% move 1,000 documents to 1,240 billed attempts, so the variable line is $49.60 rather than $40.00.

**Competing cause.** Genuine price simplicity, where the vendor really does bill only successful first attempts. That is testable and occasionally true.

**Discriminating check.** Request a sample invoice or the billing rules section of the price schedule; reconcile attempt count from the broker log against billed units for one period.

**Correction.** Model retries as `N*(1+r)*c_var`, put the minimum commit in the fixed term, and sensitivity-test r at 0, 0.1, and 0.3.

**Verification.** Invoice reconciliation for one period must match the cost model within 2%; larger gaps block the commercial gate.

### FM-08 — Unstaffed exception queue

**Mechanism.** The pilot routes exceptions to a queue that exists on the diagram but has no allocated reviewer hours. Backlog grows, latency rises, and the pilot's measured "handling time" improves precisely because exceptions sit unprocessed.

**Competing cause.** Underestimated review time per item, which produces the same symptom through a different route. The distinction is whether the queue has allocated hours at all.

**Discriminating check.** Queue depth time series against a staffed-hours calendar. A queue that grows during staffed hours and never drains overnight indicates missing allocation, not slow reviewers.

**Correction.** Allocate named reviewer hours to the exception queue before the first submission, and set a backlog guardrail (60 documents) that pauses submissions.

**Verification.** One week of queue depth whose maximum stays under the guardrail during a period with no unplanned absence.

### FM-09 — High-cost label failure hidden by aggregate success

**Mechanism.** The system is better than baseline on routine and ambiguous documents but worse on policy-sensitive ones (−8 points here), and the aggregate is dominated by volume. Cost-weighted, the ordering reverses: vendor 72.4% against baseline 75.6%.

**Competing cause.** Small-sample noise in the policy stratum (40 documents, ±11 points), which could also produce a −8 point reading. The distinction is whether the direction is stable across adjacent strata and repeats in a larger sample.

**Discriminating check.** Report per stratum with intervals; check the sign of the difference on the two next-most-expensive strata. Consistent negative sign on costly strata is a mechanism, not noise.

**Correction.** Set a separate threshold on the costly slice (for example ≥ baseline on 196+ documents) and make it a hard gate rather than a weighted input.

**Verification.** Stratum report with intervals, produced by the oracle, containing at least the minimum sample per costly stratum.

### FM-10 — Interval overinterpretation

**Mechanism.** A planning interval of [75.2%, 84.8%] is quoted as proof of a 3.3-point gain because the point estimate is favourable. The interval overlaps the baseline's plausible range, and the paired z is 1.74 — under two standard errors.

**Competing cause.** A genuinely large effect measured with a small sample, where the point difference is real and the interval is wide but entirely above the comparator. That case shows a lower bound above the comparator; this case does not.

**Discriminating check.** Read the lower bound against the comparator, not the point estimate. Also check whether the interval used the paired variance; using independent-proportion variance overstates width and invites the same misreading in the opposite direction.

**Correction.** Use the paired form (D3), publish the interval, and state the decision rule: the gate passes only if the lower bound exceeds the comparator.

**Verification.** Oracle emits the paired interval and the lower-bound comparison; no headline without both.

### FM-11 — Manual baseline measured under protest

**Mechanism.** The baseline arm is run with a weakened protocol — fewer reviewers, no rubric rehearsal, no adjudication — so the vendor looks better than it is. Alternatively the baseline is not measured at all and an internal recollection ("we're about 70%") becomes the comparator.

**Competing cause.** A genuinely weak baseline process, which would produce a large honest gain. The distinction is whether the baseline ran on the identical frozen set with the same rubric and blinding.

**Discriminating check.** Confirm identical document ids across arms, identical rubric version, and comparable reviewer seniority; check that the baseline exports carry their own run id and hash.

**Correction.** Run both arms under one rubric, one blinding regime, and one adjudication step; publish both exports.

**Verification.** Set equality of document ids between RUN-01 and RUN-02, checked mechanically by the oracle.

### FM-12 — Prompt and configuration drift inside a run

**Mechanism.** The request template, temperature, system prompt, or retrieval corpus changes mid-run, so the run averages two configurations. The aggregate is then attributable to neither.

**Competing cause.** Legitimate iteration between *runs*, which is expected; the failure is iteration *within* one run id.

**Discriminating check.** Configuration hash recorded per request; a single hash per run id is required. Any second hash splits or voids the run.

**Correction.** Freeze configuration with the manifest, require a new run id for any change, and assert the hash alongside the version.

**Verification.** Broker log showing one distinct configuration hash per accepted run id.

### FM-13 — Human approval reduced to a rubber stamp

**Mechanism.** The workflow nominally requires a human decision, but the reviewer's screen shows a pre-selected label, the default action is accept, and elapsed time is measured in seconds. Authority exists on paper and not in practice.

**Competing cause.** A genuinely high-confidence system that deserves fast confirmation. The distinction is whether the reviewer ever *reverses* the suggestion and whether dissents are recorded with reasons.

**Discriminating check.** Measure the override rate and read a sample of override rationales. A zero or near-zero override rate on a system known to err is evidence of automation bias, not of accuracy.

**Correction.** Record override reasons as a required field, audit a sample monthly, and rotate reviewers so no single person becomes the sole approver.

**Verification.** Override rate and rationale completeness reported per period; an empty rationale cohort fails the authority gate.

### FM-14 — Decision that never expires

**Mechanism.** A pilot approved on one version, price, and workload remains in force after all three change, because nothing in the process observes them. The approval's evidentiary basis is gone while the approval itself persists.

**Competing cause.** Legitimate continuity after a change that was shown to be immaterial by a documented re-test. The distinction is documentation of the re-test.

**Discriminating check.** Compare the snapshot hashes in the ledger against current version, price, and census; any difference without a re-test record is drift.

**Correction.** Attach a 90-day expiry and a change-triggered transition to `REVIEW_REQUIRED`; require the snapshot hashes to match before a state can be extended.

**Verification.** Ledger replay showing that a simulated version change produces the transition without human intervention.

### FM-15 — Cost of error assumed away

**Mechanism.** The business case compares vendor cost to reviewer cost and treats an incorrect routing decision as free. At $450 per severe error and 0.3% incidence, three errors cost $1,350 and flip a $1,140 gross gain into a $210 loss.

**Competing cause.** Genuinely low error cost, where a misroute is corrected by the customer in minutes. That is measurable from the incident ledger and occasionally true.

**Discriminating check.** Compute the break-even rate `e* = gross_gain/(N*error_cost)` and ask whether anyone has measured the manual process's actual rate. If the answer is no, the sign of the business case is unknown, not positive.

**Correction.** Estimate severe-error rate and cost from the existing manual operation before the pilot, and publish the break-even rate beside the net.

**Verification.** Incident ledger extract with per-error cost, reconciled to at least one quarter of manual operation.

**Takeaway:** each mechanism has a check that distinguishes it from its look-alike; "monitor carefully" is not a check, and a failure catalogue without discriminating tests is a worry list.

## 13. Transferable Lesson

### 13.1 The transferable rule

A capability claim is a promise about a population, and it becomes evidence only when population, label, denominator, and version are pinned. Everything else in this case follows from that rule.

The operational form: **before comparing any two systems, write down the denominator, the outcome-family list, and the version string.** If any of the three cannot be written, the comparison is not yet possible and the correct output is an evidence request, not a recommendation. This is cheap — an hour of work — and it disqualifies most vendor comparisons before anyone spends reviewer time on them.

### 13.2 Six transfers that hold beyond procurement

**To internal model comparisons.** An internal team evaluating two of its own models faces the same denominator problem, and internal teams are *more* prone to denominator laundering because there is no external party to ask awkward questions. The frozen-manifest practice transfers unchanged.

**To A/B tests in product.** The paired structure of D3 is exactly the structure of a within-subject experiment. The 2×2 discordant table, not the two marginal rates, is the analysis; product teams routinely compare marginals and conclude "lift" from a difference that is not significant.

**To capacity planning.** D4 and D7 are generic: any system that escalates work to humans converts a quality metric into a workload metric, and any workload near saturation converts a small capacity loss into a large delay. The 3.43× wait inflation for a 12.5% service loss is a property of the queue, not of this vendor.

**To cost models generally.** The 0.82% API share is the general lesson about fixed-versus-variable cost structure: optimising the smallest line item is a category error, and negotiation effort should follow cost mass.

**To governance.** A decision without expiry is a decision that outlives its evidence. The snapshot-hash mechanism is portable to any approval whose validity depends on facts that can change.

**To error economics.** Wherever a benefit is offset by a rare costly failure, the break-even rate is a more useful number than the expected net, because it converts an argument about point estimates into a question about a measurable rate.

### 13.3 What does not transfer

The specific values do not transfer: $0.04 per document, $450 per severe error, 8 minutes per review, 76.7% baseline agreement, and 25.5 staffed hours are constructed. A reader who carries the numbers instead of the method has learned the wrong thing, and the case's own sensitivity table (6.12) shows how far the conclusion moves when two of them change.

The six-predicate structure does not transfer unchanged either. A batch pipeline has no queue and a document-intelligence pilot has no latency tail; the predicate *set* must be re-derived per context, using the same principle (each predicate names an artifact that could prove it).

### 13.4 A compact derivation of the whole decision logic

```text
Given: a claim C = (task, metric, threshold, version) and a buyer workload W.
Step 1  Scopability:  C is evaluable iff task maps to W's labels and version is immutable.
                      Missing -> defer. (P1)
Step 2  Denominator:  rate = matches/(matches+mismatches+abstain+malformed+hard_fail).
                      Response-conditional rate is a secondary, labelled figure. (D1)
Step 3  Comparison:   pair on identical documents; d=(b-c)/n; se=sqrt(pb+pc-d^2)/sqrt(n).
                      Gate passes iff d - 1.96*se > 0. (D3)
Step 4  Capacity:     R = N(a+d+q); N_cap = (H_eff*60)/((a+d+q)m).
                      Infeasible iff N_requested > N_cap. (D4)
Step 5  Cost:         C_total = F + N(1+r)c_var + R*m*c_rev + I + M + K. (D5)
Step 6  Error economics: e* = (baseline_review_cost - C_total)/(N*error_cost).
                      Accept cost case iff measured e < e*. (D6)
Step 7  Authority and reversibility: human final action; demonstrated rollback; terms cleared. (C-01..C-04)
Decision = ACCEPT_PILOT iff all seven hold; DEFER iff any input to 1-7 is unmeasured;
           REJECT iff a hard constraint is provably unsatisfiable.
```

Applied to this case: step 1 fails (four of six packet fields missing), step 4 fails on illustrative inputs (1,000 requested against 708 capacity), step 6 fails at the illustrative error rate (−$210), and step 7 is unverified. Two failures and two unknowns give `DEFER`, and the evidence request in 10.1 is the list of unknowns.

**Takeaway:** the decision logic is seven steps with one inequality each; a procurement memo that cannot be reduced to those inequalities is a narrative, not an analysis.

## 14. Connections & Cross-References

### Upstream and adjacent material

| Connection | Why it matters here | Boundary |
|---|---|---|
| [AI Use-Case Charter](../Cheat_Sheets/01_AI_Use_Case_Charter.md) | supplies the owner, decision, baseline, and authority framing that the claim packet lacks | method reference only; it contains no quantity used in this case |
| [AI Market Map](../Cheat_Sheets/02_AI_Market_Map.md) | positions deployment modes and control boundaries that determine the terms gate | does not supply version or price facts |
| [Technology Due Diligence](../Cheat_Sheets/03_Technology_Due_Diligence.md) | motivates the claim ledger, the expiry rule, and the reversal-on-change test | does not validate any number here |
| [Select an AI Use-Case Portfolio](01_Select_AI_Use_Case_Portfolio.md) | sibling case: chooses among internal opportunities under a capacity gate | opposite direction of evidence flow — internal selection versus external claim |
| Section 6.2 | denominator discipline | arithmetic only; no external validity |
| Section 6.4 | paired comparison | requires one adjudicated label per document |
| Section 6.5 | capacity gate | fluid and stationary models are planning devices |
| Section 9.3 | sample-size arithmetic | assumes independent outcomes |
| Section 12 | failure catalogue | constructed mechanisms, not observed incidents |

The comparison with the sibling case is worth stating precisely, because the two cases are often confused. The portfolio case answers "which of our own ideas should we fund given a fixed reviewer minute budget" — the evidence question is estimation from the buyer's own data. This case answers "is an external party's claim about our workload credible" — the evidence question is *admissibility*: whether a claim can be tested at all. The capacity gate appears in both, but in the portfolio case it disciplines ranking, whereas here it disqualifies a vendor whose accuracy may be genuine.

### Where this case stops

Contract terms, data residency, subprocessor risk, security posture, and legal sufficiency are outside this case and are marked OPEN rather than answered. Benchmark methodology beyond the definitions used here — task construction, annotator agreement statistics, contamination audits at the vendor's end — is also out of scope; the case assumes the buyer must build its own evidence rather than audit the vendor's.

### Interview pack

Sixteen questions, ordered by increasing judgement demand. Questions 1–5 test recall of the contract; 6–13 require computation or diagnosis; 14–16 test whether the candidate can act under ambiguity. Each answer is specific to its question.

#### Q1. What is being accepted when a procurement lead signs a pilot approval?

**Answer.** A bounded, reversible test with a named owner, a named version, a fixed volume cap, and an expiry — not the vendor's general capability claim. The signature authorizes the production of evidence, and it is void the moment version, price, terms, or workload change.

**Reasoning.** The approval's content is the predicate list it satisfies. Without a volume cap and an expiry the signature would amount to an open-ended authorization to operate, which no procurement lead is empowered to grant.

**Follow-up.** If all six predicates pass but the terms review is OPEN, can the pilot start? No — terms clearance is a prerequisite for transmitting documents, not a parallel workstream.

**Red flag.** "It's a pilot, so it's low risk by definition." Reversibility is a property that must be demonstrated, not assumed from the word.

**Scoring.** 0 = describes a purchase; 1 = mentions testing; 2 = names scope and version; 3 = names scope, version, cap, expiry, and the voiding conditions.

#### Q2. Recompute agreement for the illustrative run and explain why two different percentages are defensible.

**Answer.** Full denominator: 216 matches over 216+27+14+11+2 = 270 pairs, so 80.0%. Response-conditional: 216 over 216+27 = 243, or 216/238 = 90.8% if only the 238 usable responses are counted. Both are arithmetically correct and answer different questions: per document submitted, and per document the system chose to answer.

**Reasoning.** The denominator is a policy choice about which outcome families count. Because abstentions and malformed responses are documents the desk must still handle, they belong in the operational figure.

**Follow-up.** Which one should appear in the approval record? The full-denominator figure, with the conditional one as a labelled secondary number, because the desk experiences 270 documents.

**Red flag.** Quoting 90.8% as "the accuracy" without stating the denominator.

**Scoring.** 0 = one number, no derivation; 1 = correct arithmetic on one denominator; 2 = both figures; 3 = both plus the policy argument for which is primary.

#### Q3. Compute the reviewer hours for 1,000 documents and say whether the pilot fits.

**Answer.** With a=0.05, d=0.12, q=0.10, R = 1,000×0.27 = 270 items; at 8 minutes each that is 2,160 minutes = 36.0 hours. Available capacity is 3×10×0.85 = 25.5 hours = 1,530 minutes. The pilot does not fit; the shortfall is 10.5 hours, and the volume that fits is 1,530/2.16 = 708 documents.

**Reasoning.** Reviewer load is driven by abstention, disagreement, and audit rates, not by accuracy. Capacity must be computed after a productivity factor, because paid hours are not review hours.

**Follow-up.** What if triggers overlap by 20% of the abstention count? R becomes 250 items, 33.3 hours — still over capacity, so the conclusion is unchanged.

**Red flag.** Concluding "36 hours across three people is 12 hours each, fine" — that ignores the 25.5-hour ceiling.

**Scoring.** 0 = no units; 1 = 36 hours but no comparison; 2 = identifies the shortfall; 3 = also computes 708 and notes the overlap sensitivity.

#### Q4. Why is the paired analysis different from comparing 80.0% with 76.7%?

**Answer.** The two systems scored the same 270 documents, so the informative quantity is the 27 discordant pairs: 18 documents the vendor got right and the baseline did not, 9 the reverse. That gives d = 9/270 = 3.33 points with se = sqrt((0.0667+0.0333−0.001111)/270) = 0.0191, so z = 1.74. The marginal comparison ignores the pairing and both overstates and misstates the uncertainty.

**Reasoning.** Paired designs remove document-difficulty variance, which is the dominant source of noise when the same items are scored twice. Comparing marginals throws that information away.

**Follow-up.** What would make the point estimate significant at 95%? Roughly 704 pairs by the paired power formula, or a larger discordant imbalance; at 270 pairs the effect would need to be about 4.9 points.

**Red flag.** Treating the 3.33-point difference as a 3.33% improvement in performance.

**Scoring.** 0 = compares marginals; 1 = identifies discordance; 2 = correct d; 3 = correct d, se, z, and the interpretation that the pilot is underpowered.

#### Q5. How is total cost built, and which line dominates?

**Answer.** C_total = fixed 2,000 + variable 1,000×0.04 = 40 + reviewer 270×8×0.75 = 1,620 + integration 500 + monitoring 300 + contingency 400 = 4,860. Reviewer time is 33.3% and the fixed-plus-overhead block is 66.7%; the API line is 0.82%.

**Reasoning.** Cost terms have different sensitivities. Halving the token price changes the total by $20; doubling the disagreement rate changes reviewer cost by more than $600.

**Follow-up.** Where do retries enter? As N×(1+r)×c_var: at r=0.24 the variable line is 1,240×0.04 = $49.60 and the effective per-document price is $0.0496, not $0.04.

**Red flag.** "The vendor is cheaper because $0.04 is less than our $6.00 per document" — that compares an API price with a fully loaded human cost.

**Scoring.** 0 = one line item; 1 = names several; 2 = correct total; 3 = correct total plus share analysis and the retry correction.

#### Q6. At what severe-error rate does the business case break even?

**Answer.** Gross gain is 6,000 − 4,860 = $1,140 per period. Each severe error costs $450, so e* = 1,140/(1,000×450) = 0.002533, or 0.253%. At the illustrative 0.3% the loss is 3×450 = $1,350 against $1,140, netting −$210.

**Reasoning.** The net is linear in the error rate over the plausible range, so the break-even rate is a more informative statistic than any single point estimate — it converts an argument into a measurement target.

**Follow-up.** What if the manual process has never measured its error rate? Then the sign of the business case is unknown, which is itself a reason to defer rather than to accept.

**Red flag.** Excluding error cost because "errors are rare".

**Scoring.** 0 = ignores error cost; 1 = computes net; 2 = computes break-even; 3 = both plus the argument that the unmeasured rate blocks a positive recommendation.

#### Q7. The vendor offers 94% accuracy. What is wrong with that number as evidence?

**Answer.** It is unaccompanied by population, metric, denominator, and version. On this desk the same behaviour supports 80.0% or 90.8% depending on counting; the policy-sensitive slice can be 8 points below baseline while the aggregate looks favourable; and the sample size needed to detect a 14-point gap (about 91 per arm) is larger than any single stratum here.

**Reasoning.** A rate is a triple (numerator, denominator, population). A percentage without the triple is a number about an unnamed population, which cannot bear on a decision about a named one.

**Follow-up.** What single question most cheaply exposes the gap? "What is the denominator, and are abstentions in it?"

**Red flag.** "94% is much better than our 76.7%, so it's worth a pilot."

**Scoring.** 0 = accepts the number; 1 = notes it is vendor-supplied; 2 = names two missing elements; 3 = names population, metric, denominator, version, and the stratum problem.

#### Q8. Diagnose this: reported accuracy rose after a configuration change, but reviewer load also rose.

**Answer.** The likely mechanism is a shift toward abstention. A model tuned for precision can decline ambiguous documents more often; accuracy conditional on answering rises while the number of documents returned to humans rises with it. Compute R before and after: with a rising from 0.05 to 0.20, R goes from 270 to 360 items — 36.0 to 54.0 hours — against 25.5 available.

**Reasoning.** Accuracy and workload are separate metrics with no monotone relationship. Precision-oriented abstention policies move them in opposite directions.

**Follow-up.** What check discriminates this from a reviewer-slowdown story? Read the logged abstention rate and the per-item elapsed time separately; abstention shows in the broker log, slowdown in the review console.

**Red flag.** "Accuracy improved, so the configuration is better."

**Scoring.** 0 = no diagnosis; 1 = one cause; 2 = names abstention as the mechanism; 3 = computes both loads and names the discriminating check.

#### Q9. How would you compute the smallest pilot that can still change the decision?

**Answer.** Size it to the narrowest gate that is currently undecided. If the question is whether the paired gain exceeds zero, use the paired formula: with pb=0.0667, pc=0.0333, d=0.0333, α=0.05, power 0.8, n = (1.96·0.3162 + 0.8416·0.3145)²/0.001111 = 0.7822/0.001111 ≈ 704 pairs. If the question is only whether the policy-sensitive slice is within 2 points of baseline, size that stratum instead: about 196 documents for a ±5-point interval at p=0.85.

**Reasoning.** Pilot size is a function of the decision, not of vendor ambition. Sizing to the wrong gate produces either waste or an underpowered inconclusive run.

**Follow-up.** Why not size to the aggregate 94% claim? Because the claim is not the decision; the decision is whether the buyer's workload improves, and a claim about an unnamed population does not fix an effect size to detect.

**Red flag.** "Run 1,000 because it's a round number."

**Scoring.** 0 = no method; 1 = sizes to precision only; 2 = right formula, rough number; 3 = distinguishes which gate is being sized and gives both numbers.

#### Q10. What breaks if the acceptance manifest is amended after the first outputs arrive?

**Answer.** Attribution and comparability break simultaneously. The denominator changes, so rates before and after amendment are not comparable; the selection can be influenced by observed failures, which converts the evaluation into a fit; and the comparison with the baseline, which used the original set, becomes a comparison across different populations.

**Reasoning.** The manifest is the operational definition of the population. Amending it after seeing outputs is the mechanical equivalent of moving a threshold after seeing results.

**Follow-up.** Is there a legitimate way to add documents? Yes — a new run id with its own manifest, reported separately and never pooled into the original figures.

**Red flag.** "We just removed three outliers."

**Scoring.** 0 = no consequence; 1 = mentions bias; 2 = names denominator and comparability; 3 = also names the baseline-population mismatch and the new-run-id remedy.

#### Q11. Distinguish a model that is genuinely better from one that merely abstains on hard cases.

**Answer.** Compare per-stratum rates and the abstention composition. A genuinely better model improves accuracy *within* each stratum at stable abstention; an abstention-driven improvement shows unchanged or worse per-stratum accuracy among answered documents and a rising abstention rate concentrated in the hard strata. The discriminating quantity is the stratum-level rate among answered documents, not the aggregate.

**Reasoning.** Abstention changes the population that the conditional rate describes. Any aggregate that mixes abstention with correctness is sensitive to a behaviour change that has nothing to do with capability.

**Follow-up.** What if both happen — better within-stratum accuracy and higher abstention in the hard slice? Then the decision depends on capacity rather than quality: compute R against staffed hours before deciding.

**Red flag.** Comparing only aggregates between two systems with different abstention policies.

**Scoring.** 0 = no distinction; 1 = mentions abstention; 2 = names per-stratum conditional rates; 3 = adds the capacity consequence when both effects appear.

#### Q12. A vendor offers a 60% handling-time reduction. How do you test that claim specifically?

**Answer.** Specify the unit of time (per document, wall clock, or reviewer minutes), the phase included (reading, deciding, recording), and the workload and version. Then measure paired elapsed time on identical documents with blinded reviewers, and report a percentile distribution rather than a mean. In this case the model's own latency contributes 20.0 minutes per 1,000 documents — 0.92% of the combined reviewer-and-model budget — so the claim can only be about reviewer minutes, not about API latency.

**Reasoning.** "Handling time" conflates at least three different clocks. The claim is only testable once one clock and one population are fixed, and the magnitude check shows which clock could conceivably move.

**Follow-up.** What if reviewers work faster because they trust the suggestion, but make more errors? Then time improved and quality fell; both metrics need paired measurement, and the authority gate must record overrides.

**Red flag.** Accepting a percentage without a clock definition.

**Scoring.** 0 = accepts the claim; 1 = asks for a definition; 2 = specifies clock and population; 3 = adds paired measurement, percentiles, and the magnitude check.

#### Q13. What is the failure mode when the exception queue is unstaffed, and how would you detect it early?

**Answer.** Exceptions accumulate, measured handling time falls because exceptions are never processed, and the pilot appears to succeed while the backlog grows. Detection is a queue-depth time series against the staffed-hours calendar: growth during staffed hours with no drain afterwards is the signature. The guardrail is a 60-document backlog that pauses submissions.

**Reasoning.** Any metric computed only over completed items is a survival-biased metric. The unprocessed items are exactly the ones the system found hardest, so the bias is systematic, not random.

**Follow-up.** How does this differ from genuinely slow review? Slow review drains overnight; an unstaffed queue does not. Compare depth at the start and end of each staffed block.

**Red flag.** Reporting handling time without reporting queue depth alongside it.

**Scoring.** 0 = "we would monitor"; 1 = names backlog; 2 = names survival bias; 3 = adds the discriminating check and a numeric guardrail.

#### Q14. You have 40 policy-sensitive documents and an observed −8 point difference. What can you responsibly say?

**Answer.** Very little. At n=40 and p≈0.85 the half-width is 1.96·sqrt(0.1275/40) = 11.1 points, so an interval around −8 points spans zero and also admits −19. The responsible statements are that the direction is unfavourable, that the slice is too small to support a threshold decision, and that at least 196 documents per stratum are needed for a ±5-point statement.

**Reasoning.** A stratum estimate inherits the sample size of the stratum, not of the study. Reporting a stratum difference without its interval is the most common way an aggregate-looking number becomes a false certainty.

**Follow-up.** Would you combine strata to gain precision? Only with an explicit cost weighting and a statement that the weights are policy choices; pooling unequal strata without weights changes the question being asked.

**Red flag.** "The difference is 8 points, so it fails the threshold."

**Scoring.** 0 = accepts the point estimate; 1 = notes small sample; 2 = computes the interval; 3 = computes the interval, states what cannot be said, and names the required size.

#### Q15. The sponsor says "just run it for 30 days and we'll see." What is the best response?

**Answer.** Thirty days is a duration, not a design. Without a frozen manifest, a denominator policy, a paired baseline, and pre-set thresholds, thirty days produces an anecdote that cannot be compared to anything — and if the pilot is capped at 708 documents for capacity reasons, the run will produce roughly 191 review items per week, far too few for the policy-sensitive stratum. Offer instead a 30-day run with a frozen 196-document hard-case stratum and a pre-registered decision rule.

**Reasoning.** Time is not sample size. The binding question is how many observations of the *decision-relevant* slice the design yields, and that is a per-stratum count.

**Follow-up.** What if volume cannot be raised? Then the honest expectation is a directional read on the routine strata and an inconclusive result on the costly slice — state that before running, not after.

**Red flag.** Agreeing to the duration without fixing the design.

**Scoring.** 0 = agrees; 1 = adds metrics; 2 = names frozen manifest and baseline; 3 = converts duration into per-stratum counts and pre-registers the decision rule.

#### Q16. When is rejecting the vendor the correct action rather than deferring?

**Answer.** When a hard constraint is provably unsatisfiable rather than merely unverified: no human-authority policy can be met, terms are unacceptable after authorized review, the version cannot be pinned at all, or there is no reversible fallback. Unknown evidence is a defer; a known disqualifier is a reject.

**Reasoning.** The distinction is evidential, not attitudinal. Deferring a proven disqualifier wastes a cycle; rejecting an unverified claim discards a possibly viable option.

**Follow-up.** Can a rejection be revisited? Yes, with a stated resubmission condition — for example "reconsider if an immutable version identifier and retry-inclusive pricing are provided" — which also prevents the same packet re-entering through another channel.

**Red flag.** Rejecting because the packet is annoying, or deferring because rejection feels aggressive.

**Scoring.** 0 = no criterion; 1 = mentions one hard gate; 2 = distinguishes unverified from unsatisfiable; 3 = adds the resubmission condition and the anti-recycling rationale.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Decision | defer for evidence; retain manual review; pilot not authorized |
| Unit | one document-decision pair in one run, not one customer or conversation |
| Headline metric | matches / all submitted documents (80.0% illustratively), abstentions and failures included |
| Never report | a rate without its denominator, outcome-family counts, and version string |
| Comparison | paired 2×2 on identical documents; d=(b−c)/n with se from discordant counts |
| Capacity | R = N(a+d+q); N_cap = (H_eff·60)/((a+d+q)·m) = 708 documents at 25.5 staffed hours |
| Cost | F + N(1+r)c_var + R·m·c_rev + I + M + K = $4,860 per 1,000 documents |
| Error economics | e* = gross gain/(N·error cost) = 0.253% severe-error break-even |
| Reversal trigger | version, price, term, or workload change → REVIEW_REQUIRED, evidence void |
| Evidence limit | every number is constructed arithmetic; no vendor was evaluated and no pilot ran |

