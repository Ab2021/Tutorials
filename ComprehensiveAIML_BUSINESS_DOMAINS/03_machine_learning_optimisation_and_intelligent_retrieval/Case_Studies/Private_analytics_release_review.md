---
case_id: "CASE-03-16-002"
source_plan_id: "PLAN-M16-02"
case_title: "Private analytics release review: whether to publish a workforce equity and attrition benchmark"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "draft"
content_status: "draft"
source_status: "OPEN"
case_type: "analytical"
domain: "analytics — people analytics, statistical disclosure control, external reporting"
primary_pattern: "privacy threat model"
secondary_patterns: ["differential privacy", "release governance"]
evidence_level: "illustrative"
source_basis: "composite"
as_of_date: "2026-11-10"
owner_role: "Chief Privacy Officer (decision); Head of People Analytics (delivery)"
source_index: "Train/ComprehensiveAIML_BUSINESS_DOMAINS/03_machine_learning_optimisation_and_intelligent_retrieval/Case_Studies/INDEX.md"
review_trigger: "Change in the declared privacy budget, cell granularity, release cadence, jurisdiction, or a works-council ruling on the benchmark publication"
---

# Private analytics release review: whether to publish a workforce equity and attrition benchmark

> **Evidence declaration.** This is a composite teaching case. The employer ("Vantia Group"), its 48,000 employees, its regional splits, its attrition, its pay figures, its release cadence, its reviewers, and its committee are **constructed illustrative values**. The arithmetic in Sections 6, 8 and 9 is executed step by step and independently re-checked, but nothing here is a measurement of any named organisation, and no organisation is reported to have made this decision. Mechanisms attributed to published work — the Laplace and Gaussian mechanisms, sequential and parallel composition, advanced composition, amplification by subsampling, group privacy, zCDP conversion, k-anonymity and its composition failures — are **sourced** to the references named in Section 5e and Section 14a. Where the case asserts a numeric consequence, that consequence is **derived** from the cited mechanism under stated assumptions; where it asserts a utility judgement, the judgement is labelled **illustrative**. Section 12 marks every failure row with its own label.

> **What a reader should be able to do afterwards.** Given a proposed release table, its grain, its mechanism, its budget and its cadence, reconstruct the disclosure risk, compute the accuracy the release can actually deliver, decide whether the release answers the question it was commissioned to answer, and defend or refuse the release under a changed budget or a changed adversary.

## 1. The One-Line Framing

> Given **an internal HR record for each of 48,000 employees, aggregated into a proposed published cross-tabulation of headcount, attrition and base pay**, decide **whether to release the benchmark externally and to all 2,300 managers, and at what grain and under what privacy mechanism**, for the **Chief Privacy Officer acting on a works-council recommendation**, to improve **the organisation's ability to detect and correct pay disparities and to give external bodies a defensible benchmark**, subject to **a single annual disclosure budget of ε ≤ 1.0 at δ = 10⁻⁶ per released table family and a prohibition on any publication that lets a named individual's attrition status be inferred with material confidence**, compared with **the current practice of an Excel extract that suppresses cells below 10 employees and rounds to the nearest five**.

The task category is **release engineering under a disclosure constraint**: the artefact is a published table plus a written privacy argument, not a model and not a prediction. The unit of analysis is the **employee-cell**: an intersection of region, job family, tenure band and recorded sex, evaluated at a fixed quarterly snapshot date. The decision recurs **once per quarter** at the operations layer and **once per year** at the governance layer, when the annual budget is re-authorised.

One primary success condition, in plain language: *the published release must let a competent reader answer the question it was commissioned to answer — is there a pay gap by sex within a job family and tenure band, and is attrition materially different across regions — while bounding how much any single release can change an adversary's belief about one named employee.* The operational definition of each half of that sentence lives in Section 9; the plain-language version is deliberately vague so that it cannot be mistaken for a benchmark claim.

The reusable pattern is **privacy threat model**: before choosing a mechanism, write down who the adversary is, what auxiliary data they hold, which inference is harmful, and what the decision will do if the bound cannot be met. The secondary patterns are **differential privacy**, which supplies the bound, and **release governance**, which supplies the durable state — a budget ledger, an approval record, and a refusal path — without which the bound is decorative.

The decisive tradeoff, and the reason the obvious framing fails: the release is asked to answer a question about a **small difference** (a pay gap of a few percent) inside **small groups** (a few hundred people per sex within a job family and tenure band), and the noise that a ε ≈ 1 budget requires is an order of magnitude larger than the difference. A framing that asks only "is this table compliant?" produces a yes, a published table, and a reader who concludes from it that no gap exists — because the noisy estimate sits within a few thousand currency units of zero either way. The framing that survives asks a narrower, answerable question and prices the answer before publishing it.

The second, less obvious finding: the binding constraint is not the strength of the mechanism but the **uniformity of its noise across cells**. In a published table of per-cell means, an ε-DP release calibrated on the table's worst cell imposes that worst cell's noise on every cell. One 120-person cell in a 96-cell table therefore degrades the accuracy of the 4,100-person cells by roughly a factor of six relative to a targeted query at the same budget. The organisation's instinct — "publish the table, the mechanism will protect it" — destroys the utility of the very cells that were worth publishing.

The third: an organisation can satisfy the letter of a differential-privacy budget and still disclose, because the mechanism protects the *release* and not the *workflow*. In the simulated audit of Section 12, both successful re-identifications came from an exact internal dashboard outside the release boundary, not from the noisy table.

## 2. Context & Background

### 2.1 Domain, product and the artefact under review

The organisation is a food manufacturing and distribution group with operations in three regions and four job families, and it is the kind of employer that a works council, a sector association and a regulator all ask for workforce statistics at the same time and in slightly different shapes. The artefact under review is a quarterly **Workforce Equity and Attrition Report**: a set of tables describing how many people are employed in each cell, how many left in the trailing twelve months, and what base pay looks like in each cell. It is not a dashboard, not an API, and not a model output. It is a document that leaves the building and cannot be recalled.

That permanence is the whole difficulty. A model score can be recomputed tomorrow; a PDF that a journalist downloaded last quarter, combined with this quarter's PDF, is an arithmetic input the publisher no longer controls. Every design choice in this case is downstream of the fact that releases **compose**: two individually defensible publications can be jointly indefensible when one can be subtracted from the other.

Three consumers, three different questions, one artefact:

| Consumer | Question actually asked | Grain demanded | Consequence of a wrong answer |
|---|---|---|---|
| Works council | Is pay equal within comparable roles, and is attrition concentrated in particular groups? | region × job family × tenure × sex | A negotiated settlement built on a statistical artefact |
| Sector association benchmark pool | How does our attrition compare with peers? | region × job family | Misallocation of retention budget across regions |
| Regulator / public reporting | Are the published figures consistent with the filings? | region totals, plus a supporting note | A correction notice and loss of credibility |

The three demands are not compatible at one grain. The works council needs the finest cut; the benchmark pool needs a comparable cut across firms; the regulator needs a *consistent* cut whose totals reconcile. A release designed to satisfy all three at once typically ends up satisfying none, because the reconciliation requirement forces the publisher to make the small cells recoverable by subtraction.

### 2.2 Workflow before and after the decision

The workflow that produces the report runs on a quarterly clock. Two weeks before the publication date, an analyst extracts a snapshot of the HR system as at the quarter-end date, applies the cell definitions, computes counts and means, applies the current suppression rule, and sends the result to a single HR business partner for review. That review is a visual check for anything that "looks identifying". It is not a disclosure calculation, and the reviewer has no way to know what an external reader could reconstruct by combining the table with the previous quarter's table or with a professional-network profile.

After publication, the report is posted to the intranet for managers and sent to the two external bodies. Nothing observes what happens next: there is no log of who downloaded it, no register of which combinations have been published, and no process that would notice if a later table, combined with an earlier one, narrowed a cell to a single person. The release is treated as a document, not as a sequence of queries against a finite resource.

The intervention in this case changes three things: the mechanism (suppression to a calibrated mechanism with a documented bound), the state (an append-only budget ledger that survives staff turnover), and the gate (a refusal path that a reviewer can invoke without needing the publisher's agreement). The workflow after the intervention still has one extract, one computation and one review; what changes is that the review has a number to check and the publication writes a record that constrains the next one.

### 2.3 Stakeholders and their actual authority

| Role | Authority | What they can and cannot do |
|---|---|---|
| Chief Privacy Officer | Approves, refuses or conditions any release | Can stop publication; cannot invent budget that the board has not authorised |
| Head of People Analytics | Owns the extract and the computation | Can change grain and mechanism; cannot waive the budget |
| Works council representative | Consults; can escalate to the board | Can delay or force rework; cannot approve publication alone |
| Legal counsel | Interprets applicable obligations | Determines scope; does not select mechanisms |
| Data engineering lead | Owns the pipeline and the ledger service | Can refuse to run an unbudgeted query |
| Communications lead | Publishes and corrects | Owns the correction notice if a release is withdrawn |

The separation matters. In the baseline process the analyst who computed the table also decided that it was safe, which is a conflict of interest in the specific sense that the person with the strongest incentive to publish (their deliverable is due) is the person certifying the safety property.

### 2.4 Scale, timing and the shape of the population

The snapshot used throughout this case is dated 2026-09-30 and contains **48,000 employees**. The regional split is uneven, which is the first thing that breaks naive cell-size rules: a rule calibrated on the group total is wrong for every region.

| Region | Employees | Share | 32-cell average | Trailing-12-month leavers | Attrition rate |
|---|---:|---:|---:|---:|---:|
| R1 Europe | 26,400 | 55.0% | 825 | 2,136 | 8.09% |
| R2 Americas | 12,960 | 27.0% | 405 | 1,216 | 9.38% |
| R3 Asia-Pacific | 8,640 | 18.0% | 270 | 968 | 11.20% |
| **Group** | **48,000** | **100%** | **500** | **4,320** | **9.00%** |

The arithmetic behind the last row is worth stating because it is the first place a reader can catch an inconsistency: `2,136 + 1,216 + 968 = 4,320`, and `4,320 / 48,000 = 0.09`. The regional rates are `2,136 / 26,400 = 0.080909`, `1,216 / 12,960 = 0.093827`, and `968 / 8,640 = 0.112037`. A weighted check of the group rate gives `0.55 × 8.0909 + 0.27 × 9.3827 + 0.18 × 11.2037 = 4.450 + 2.533 + 2.017 = 9.000` percentage points, which reconstructs the group figure to three decimals. That reconciliation is not decoration: it is the identity an external reader will use to test whether the published regional rates are self-consistent, and therefore the identity an attacker will use to recover anything the publisher tried to suppress.

Pay is recorded as annual base salary in a single currency unit (CU) with no bonus, no overtime and no benefits, and the case says so explicitly because a released "pay" figure that silently includes or excludes a component is a comparability defect, not a privacy one. The group mean is **60,870 CU**, derived as `(66,000 × 26,400 + 59,000 × 12,960 + 48,000 × 8,640) / 48,000 = 2,921,760,000 / 48,000`. The median is 54,000 CU. The mean exceeds the median by 12.7%, which is the signature of a right tail — and that tail is what determines the clipping bound in Section 6.5.

Tenure splits identically across regions at 15 / 30 / 25 / 30 per cent, giving the four bands used later: 0–12 months (**7,200** employees), 12–36 months (14,400), 36–60 months (12,000), and 60 months or more (14,400). The split is identical by construction in this teaching snapshot, which keeps the cell arithmetic legible; in a real snapshot it would not be, and Section 9's sensitivity analysis treats the real case where it is not. The smallest region-by-tenure cell is R3 with 0–12 months at **1,296** employees (`8,640 × 0.15 = 1,296`), which is the number that decides whether the pay table is publishable at all.

### 2.5 Baseline: what "we already suppress small cells" actually does

The current rule is: compute the cell, suppress any cell with fewer than 10 employees, round every remaining count to the nearest five, and publish row and column totals computed from the *unrounded* data. It has three properties that the organisation believes are protective and that are not.

First, the totals are computed before suppression, so the published table does not sum to the published total. The gap between the two is not noise; it is exactly the aggregate of the suppressed cells. Second, suppression is applied per cell without complementary suppression, so a suppressed cell in a row whose other cells and row total are published is recoverable by one subtraction. Third, rounding to five is applied to cell counts but not to totals, which creates a second inconsistency an adversary can exploit in the opposite direction. Section 6.2 works all three numerically.

The organisation has run this rule for eleven quarters. No incident has been recorded. That absence is not evidence of safety; it is evidence that nobody has looked, which the case treats as a finding rather than a comfort.

### 2.6 Applicable requirements and what this case does not determine

Legal review in this exercise identified that the release falls within the scope of applicable data-protection law in the jurisdictions of operation, and that the works council has a consultation right over workforce reporting. The case does **not** determine which specific obligations apply, does not interpret them, and does not treat any jurisdiction's rules as interchangeable. That determination is recorded as **OPEN** and is owned by legal counsel, with the trigger being any change in jurisdiction or in the release's recipient list.

What the case does treat as settled, because the organisation settled it before the analysis began: an individual's attrition status is a sensitive attribute for the purposes of this release; the employer's legitimate interest in benchmarking does not extend to identifying which named employee left; and a release that permits that identification is not made acceptable by the fact that the information is "already known to managers".

That last clause is load-bearing and it is where most real releases fail. "Managers already know who left" is true and irrelevant: the managers know about their own teams, and the release tells them about everybody's.

### 2.7 Why the obvious framing misses the decision

The obvious framing is a yes/no compliance question: "does the table satisfy the suppression rule and the legal review?" Under that framing the answer is yes, the report ships, and the organisation has discharged its obligation. The framing fails because compliance with a rule that does not bound an adversary's inference is not the property the organisation needs, and because the question that the report was commissioned to answer — whether a pay gap exists within comparable groups — is a question about a quantity smaller than the noise the safe mechanism requires. The decision is therefore not "is it compliant" but "which of four incompatible releases do we ship, and what do we tell the reader we cannot answer".

**Takeaway:** the release is a sequence of queries against a finite, non-renewable resource, published to an adversary who will combine it with everything else; treat the document as one query in that sequence, not as a finished artefact.

## 3. The Business Question

### 3.1 The question in the decision owner's words

The following is an **illustrative reconstruction**, written for this teaching case. It is not a quotation from a real person and no named organisation is described.

> "Every quarter I am asked to sign a report that goes out to two thousand managers and two external bodies. I am told it is safe because small cells are suppressed. I cannot verify that claim, I have no record of what we published last year, and I have no way to say no that does not look like obstruction. So tell me: what exactly may we publish, under what guarantee, and what do we tell the people whose question the answer does not settle?"

That question is useful because it contains three separable demands that most privacy reviews collapse into one. It asks for a **permission** (which tables may ship), a **guarantee** (what property makes them safe), and a **communication decision** (what to say when the published data cannot answer the question asked). A review that answers only the first has not answered the decision owner.

### 3.2 The decision contract

| Decision contract field | Content |
|---|---|
| **Question** | May the quarterly Workforce Equity and Attrition Report be released externally and to all managers, and if so at which grain, under which mechanism and privacy parameters, with which tables withheld? |
| **Decision owner** | Chief Privacy Officer, acting with the Head of People Analytics as the accountable delivery owner and the works council as consulting party |
| **Trigger and deadline** | The Q3 2026 report is due for release in fourteen days; the works council has tabled a motion that the current suppression rule is "not a privacy control"; a sector association has asked for a comparable table at the same time |
| **Available actions** | (a) ship the current suppression-based report unchanged; (b) ship the report at region grain only; (c) ship counts at fine grain and pay means at region × job-family grain under a declared DP budget; (d) withdraw the report and answer the works council's questions internally; (e) ship a redacted report with a written statement of what it cannot establish |
| **Value and downside** | Value: a defensible, repeatable benchmark that lets the council test pay equity without bespoke extracts, and that removes an ad hoc quarterly negotiation. Downside of a wrong release: an irreversible disclosure about identifiable employees, a correction notice, a negotiated settlement built on a statistical artefact, and the loss of the release channel for future quarters |
| **Objective** | Maximise the number of pre-declared questions the release can answer at a stated confidence, subject to a disclosure bound; equivalently, choose the grain that maximises statistical power per unit of privacy budget spent |
| **Hard constraints** | Annual disclosure budget ε ≤ 1.0 at δ = 10⁻⁶ per table family; no published table may permit a named individual's attrition status to be inferred with material confidence; totals must reconcile with the published cells; the release must be reproducible from a versioned snapshot |
| **Scope** | 48,000 employees at the 2026-09-30 snapshot; three regions; four job families; four tenure bands; two recorded sex categories; base salary only; trailing-twelve-month voluntary and involuntary leavers combined |
| **Out of scope** | Individual-level disclosures to managers; bonus, equity and benefits comparability; any determination of unlawful discrimination; promotion and performance outcomes (deferred to a separate review because the outcome is manager-assigned and not comparable across regions) |

### 3.3 The proxy-versus-objective conflict

The objective as stated — "let the council test pay equity" — is a proxy. The underlying business objective is that the organisation should be able to *identify and correct* an unjustified pay gap. Those are different, and the difference determines the release design.

If the objective is to *test*, the release needs enough statistical resolution to detect a gap of a few percentage points inside comparable groups, which is a statement about power. If the objective is to *correct*, the release is almost irrelevant: corrections happen through the compensation review process, on internal individual data, and the published table is a governance artefact that documents that the process ran. A release optimised for "testing" at the grain the council asked for consumes the entire budget and still fails to detect anything; a release optimised for "documenting that the process ran" is cheap but cannot support an external claim.

The case resolves this by making the conflict explicit in the decision: the release is designed to support **a pre-declared set of testable claims**, and the claims that cannot be supported at the accepted budget are moved to the internal, access-controlled analysis with a written explanation. Neither the council nor the regulator is told that the release "shows pay equity"; they are told which specific comparisons it supports and which it does not.

There is also a guardrail that prevents the proxy from being gamed. If the release is scored on "number of cells published", the cheapest way to win is to publish many noisy cells and let the reader average them, which manufactures false precision. The guardrail is a **pre-declared analysis plan**: the comparisons are fixed before the budget is spent, the multiplicity correction is fixed with them, and adding a comparison after seeing the results is a budget violation, not a refinement.

### 3.4 What "no intervention" looks like

The no-change option is not neutral. Under the current rule the organisation publishes an internally inconsistent table (Section 6.2 shows the suppressed cells are recoverable by subtraction), accumulates an unrecorded series across eleven quarters, and holds no record that would let it answer "what have we already revealed?" if challenged. Choosing "no change" therefore means choosing to keep publishing a release whose safety property the organisation cannot state, and to keep the decision inside a single reviewer's judgement.

The honest comparison is not "current report versus new report". It is "current report plus its unquantified risk versus a report whose risk is stated, whose budget is finite, and which will run out of budget and have to stop". The second is not obviously better on every dimension — it is better on verifiability and worse on flexibility, because a budget cannot be argued upward in a crisis.

**Takeaway:** the decision is not whether to publish; it is how to divide one finite budget across a fixed set of questions, and what to say about the questions that lose.
