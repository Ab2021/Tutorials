# How to Read This List

> Module 22 — Sources and Further Reading
> Topic 01 of 09
> Estimated study time: 3 hours
> Prerequisites: none
> Version 2026.3

## 1. Why this topic exists

A reading list looks like an artefact of teaching. In practice it is a decision instrument. Every week, someone in a biomedical data role has to decide what to believe under time pressure: whether a model claim is supportable, whether a guidance document is current, whether a textbook statement still holds, whether a market number can go into a board deck. The list is where those decisions start, and the quality of the decision is bounded by the quality of the source behind it.

The failures are ordinary and expensive:

- A regulatory analyst cites a law firm summary that describes a draft guidance as if it were final. The validation plan is built on a recommendation the agency has not adopted.
- A data scientist cites a preprint reporting a strong result. The peer-reviewed version reverses the direction of the effect six months later, after the slide deck has circulated.
- A clinical lead cites a textbook statement on sepsis definitions that predates the definition the hospital now uses, so the cohort and the label disagree.
- A product manager cites a market size from a vendor white paper, where the number comes from a survey the vendor commissioned about its own category.
- A diligence team cites a peer-reviewed paper without noticing that the journal is on a publisher-integrity watchlist and the paper has no independent replication.
- A team cites a dataset paper for a version of the data that was superseded, so the reported counts do not match the file they downloaded.

None of these is a research failure. Each is a source-judgement failure, and each is cheap to prevent with a procedure that takes minutes.

Three structural changes make the judgement harder than it was a decade ago. First, the volume of unreviewed material has grown: medRxiv and bioRxiv host hundreds of thousands of preprints, and search engines rank them beside peer-reviewed articles. Second, regulatory and standards material churns. The United States Food and Drug Administration (FDA) issues or updates dozens of device guidance documents each year, the European Union has layered the Medical Device Regulation, the In Vitro Diagnostic Regulation, and the Artificial Intelligence Act, and standards bodies issue amendments that change the binding text without changing the title. Third, synthesis is now cheap. A fluent paragraph can be produced from a mixture of a press release, a preprint, and a blog, with no signal to the reader about which sentence came from where.

The people blocked by weak source judgement are specific. The regulatory lead cannot defend a predicate choice. The statistician cannot write a methods section that survives review. The clinician cannot reconcile a model's features with current practice. The investor cannot separate a category from a company. The analyst cannot answer "how do you know?" in a design review. A shared procedure gives all of them the same vocabulary and the same record.

This topic is not epistemology and not library science. It is a working method with five tests, a claim router, a scoring rubric, a worked case, and a maintenance routine. The rest of the module applies the method to categories: textbooks, regulation, methods, clinical background, datasets, industry evidence, communities, and the ongoing maintenance of a source base. Read this topic first, because every later topic assumes its vocabulary.

> **Note:** A bibliography is a claim about relevance, not a guarantee of quality. Inclusion in this list means a source is worth inspecting for a stated purpose. It does not mean the source is current, unbiased, or sufficient for your claim. The evaluation is yours, and it has to be redone when the claim changes.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a claim by type (regulatory, efficacy or safety, methodological, clinical background, market, or operational) and state the minimum source tier that can support it. Assessed by Exercise 1 and self-check item 1.
2. **Apply** the five tests — authority, currency, applicability, accessibility, and provenance — to a candidate source and record a pass, conditional, or fail result for each with one line of evidence. Assessed by Exercise 2 and the worked example.
3. **Distinguish** primary, secondary, and tertiary material, and identify the point at which a secondary source stops being adequate support. Assessed by Exercise 3 and self-check item 3.
4. **Detect** the common currency failures: draft guidance read as final, a superseded edition, a preprint cited after a contradicting publication, and a dataset version mismatch. Assessed by Exercise 4.
5. **Score** a candidate set with a reproducible rubric, including veto rules and interest caps, and explain why a weighted mean alone produces the wrong verdict. Assessed by Exercise 5 and section 7.
6. **Write** a source ledger entry that a colleague could audit a year later: claim, candidate, test results, verdict, access date, and residual uncertainty. Assessed by Exercise 6.
7. **Critique** a citation chain in a document you did not write, and name the specific link where the claim outruns the evidence. Assessed by Exercise 7.
8. **Design** a re-verification schedule for a source set, with cadence chosen from the volatility of each source class rather than from habit. Assessed by Exercise 8.

Each outcome is observable in a document. "Understands source quality" is not an outcome. "Produces a five-test record with a dated verdict and a named failing test" is.

## 3. The core mental model

The model is a router followed by a funnel. The router decides which kind of evidence could settle the claim. The funnel removes candidates that cannot carry it. What survives is a small set of citable sources, often one primary source plus one independent confirmation.

```text
                    CLAIM (one sentence)
        population | intervention/exposure | outcome | time | setting
                              |
                              v
                     CLAIM TYPE ROUTER
      regulatory | efficacy/safety | method | clinical background | market
                              |
                              v
                 CANDIDATE SET (>= 3, >= 1 primary)
        best available | independent confirmation | context/definition
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
   AUTHORITY             CURRENCY             APPLICABILITY
   who says it,          is it the current     does the population,
   in what role,         version, and does     setting, task, and
   under what            the claim depend on   time match the claim
   independence          a moving target
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                      ACCESSIBILITY
              can a reviewer obtain and check it
                              |
                              v
                       PROVENANCE
        methods, data, conflicts, and chain of custody
                              |
                              v
                  VERDICT + SOURCE LEDGER
        primary support | with limitation | background | reject
```

Read the diagram left to right and top to bottom. The claim comes first because every test is relative to it. The same paper can be primary support for a narrow performance claim and background only for a general efficacy claim. There is no such thing as a good source in the abstract.

**The claim type sets the authority ceiling.** A regulatory obligation can only be supported by the regulation, the statute, or an official agency statement; a peer-reviewed article about the regulation is commentary. An efficacy claim needs a comparative study in the relevant population, not a mechanism paper. A definitional claim can be settled by a consensus statement or a standards document. A market-size claim needs a disclosed method and a denominator, which most vendor material does not provide. Getting the router wrong is the most common way a well-read team produces an unsupportable sentence.

**Currency is relative to the half-life of the claim.** Anatomy changes slowly; sepsis definitions changed in 2016; device guidance changes yearly; a market estimate is stale within two quarters. Currency is not "published recently." A 2019 paper can be entirely current for a physiological mechanism and entirely stale for a regulatory pathway.

**Applicability is the test most often skipped.** A trial in 28 tertiary academic centres does not establish performance in a 90-bed community hospital. A model validated on adults does not transfer to children. A dataset from 2008 to 2012 does not represent care after a definition change. Applicability is not a single yes or no; it decomposes into population, setting, task, and era, and each can fail separately.

**Accessibility is part of evidence quality.** A claim that a reviewer cannot check is a claim that cannot be audited. Paywalled is not disqualifying, because the abstract, the supplementary material, and the DOI record are often enough to check the specific sentence. Unobtainable is different: a source with no locator, no version, and no archive cannot be verified at all.

**Provenance is independent of prestige.** A paper in a high-impact journal can have undisclosed industry authorship, a selectively reported endpoint, or a data availability statement that resolves to nothing. Provenance asks who produced the evidence, with what method, under whose funding, and whether the chain from raw data to stated conclusion is visible.

A useful two-axis view separates evidential strength from fit:

| | Low applicability | High applicability |
|---|---|---|
| **High authority** | Strong source, wrong question. Cite as context, not support. | Primary support. Still check currency and provenance. |
| **Low authority** | Do not cite. It adds noise and invites challenge. | Usable as a pointer to a better source, never as support. |

Most citation errors sit in the top-left cell: a respectable source that answers a different question. A randomised trial of a sepsis alert in one health system is high authority and, for a claim about national mortality benefit, low applicability. The citation looks strong and is not.

The ledger is what makes the model durable. For each claim, record the sentence being supported, the candidate set, the five test results, the verdict, the access date, and what would change the verdict. A year later, the ledger tells a successor whether the source was wrong or the claim drifted.

The five tests are not equally cheap to run, and the order reflects that. Authority and currency are desk checks that take minutes. Applicability requires reading the methods and knowing the deployment context. Accessibility requires obtaining the document. Provenance requires reading the methods, the data statement, and sometimes the supplementary material. Running the cheap tests first is a filtering strategy: most bad candidates fail before the expensive reading.

Two failure patterns belong to the model itself, not to any source. The first is **false confidence from a complete-looking record**: a candidate that passes all five tests on paper can still be the wrong source if the search that produced it was narrow. The second is **false rigor from the rubric**: a precise score can create an impression of measurement that the underlying judgements do not support. Both are corrected by the same habit, which is to state the uncertainty in words alongside the score.

| Test | Typical time | Who can run it | Artefact produced |
|---|---|---|---|
| Authority | 5 minutes | Any analyst | Issuer and interest note |
| Currency | 5 to 15 minutes | Any analyst | Version and status line |
| Applicability | 20 to 40 minutes | Analyst with domain context | Four-dimension comparison |
| Accessibility | 10 minutes | Any analyst | Locator and archive reference |
| Provenance | 30 to 60 minutes | Analyst or methodologist | Methods and conflict assessment |

## 4. Definitions and vocabulary

The vocabulary is split into five families because the words are used loosely and the looseness causes most disagreements. Each entry gives the definition used in this module, the common misuse, and the unit or coding where one applies.

### 4.1 Document and publication types

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Primary research | A study that generates new observations: trial, cohort, case-control, cross-sectional, simulation, bench experiment. | Used for any paper with data, including a review that reanalyses published estimates. | Study design label |
| Secondary research | Work that synthesises primary studies: systematic review, meta-analysis, scoping review, modelling study built on published parameters. | Treated as independent confirmation of the primary studies it summarises. | Review type |
| Tertiary material | Summaries that point to evidence: textbooks, point-of-care topics, guidelines, encyclopaedias. | Cited as if it were primary evidence. | Reference type |
| Preprint | A manuscript posted before peer review, with a DOI and a version history. | Cited as peer-reviewed, or dismissed entirely. | Version number and server |
| Conference abstract | A short report of work, often without full methods or peer review of the analysis. | Treated as equivalent to a full paper. | Abstract number |
| Conference paper | A full paper in proceedings, reviewed to a variable standard by conference. | Assumed to be journal-equivalent. | Proceedings identifier |
| Editorial | An opinion piece by an editor or invited author, usually tied to an article in the same issue. | Cited as evidence for the underlying claim. | Article type |
| Commentary | An external expert response, often with a different interpretation. | Confused with a correction or a consensus. | Article type |
| Practice guideline | Recommendations for care, graded by a body using a stated method. | Read as a regulation or as settled evidence. | Recommendation strength |
| Consensus statement | Agreement among experts, not necessarily a systematic appraisal of evidence. | Treated as a guideline with the same evidentiary basis. | Consensus method |
| Regulation | A legally binding instrument adopted by a legislature or agency under delegated authority. | Confused with guidance issued under it. | Legal citation |
| Guidance | An agency's interpretation or recommendation, usually non-binding unless stated. | Cited as law. | Document status |
| Standard | A technical specification adopted by a standards body, sometimes incorporated by reference into law. | Assumed to be freely available or legally binding in all jurisdictions. | Standard number and edition |
| White paper | A position document, often commercial, that may contain original analysis or may not. | Treated as peer-reviewed evidence. | Publisher and funding |
| Press release | An institutional announcement, usually without methods. | Used as the citation for a number that appears nowhere else. | Issuing body and date |
| Blog post | Informal commentary, variable in both expertise and accountability. | Cited when no better source was found. | Author and date |

The distinction that matters most in practice is between a document's genre and its evidential role. A guideline is tertiary material when it summarises evidence and quasi-regulatory when a payer or accreditor enforces it. A regulation is binding; the guidance that explains it is not, even when the guidance uses mandatory-sounding language. A standard becomes binding only when a contract or a regulation incorporates it by reference.

### 4.2 Appraisal terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Internal validity | Whether the comparison within the study supports a causal or predictive claim. | Asserted from journal reputation. | Risk-of-bias judgement |
| External validity | Whether the result holds in the population and setting of interest. | Assumed from a large sample. | Applicability judgement |
| Risk of bias | The likelihood that a design feature distorts the estimated effect. | Used as a synonym for low quality. | Low, some concerns, high |
| Confounding | A third variable associated with exposure and outcome that distorts the comparison. | Addressed by adjusting for everything measured. | Directed acyclic graph |
| Effect size | The magnitude of the difference or association, with a unit. | Reported as a p-value. | Risk difference, odds ratio, hazard ratio |
| Confidence interval | A range of values compatible with the data at a stated confidence level. | Read as a probability that the true value lies in the range. | Interval with confidence level |
| Statistical significance | A decision rule about compatibility with a null hypothesis, not a measure of importance. | Treated as proof of an effect. | p-value threshold |
| Power | The probability of detecting an effect of a given size if it exists. | Computed after the study to explain a null result. | Proportion |
| Absolute risk | The difference in event rates between groups, in percentage points. | Replaced by relative risk in communication. | Percentage points |
| Number needed to treat | The number of patients who must be treated for one additional outcome. | Quoted without the time horizon. | Count with time window |
| Surrogate endpoint | A marker substituted for the outcome of interest. | Assumed to predict the clinical outcome. | Endpoint definition |
| Composite endpoint | A combined outcome whose components may differ in importance. | Reported without component breakdown. | Component list |
| Heterogeneity | Variation in effect across studies beyond chance. | Ignored when a pooled estimate is convenient. | I-squared, tau-squared |
| Publication bias | Selective publication of results by direction or significance. | Assumed absent because a funnel plot looks symmetric. | Funnel plot, trim-and-fill |
| Spin | Reporting that makes a null or harmful result appear favourable. | Treated as a writing style issue. | Narrative judgement |

A p-value is not an effect size and a confidence interval is not a probability about the parameter under a default Bayesian reading. These distinctions matter for source judgement because a claim copied from an abstract often preserves the p-value and drops the interval, the unit, and the population.

### 4.3 Identifiers and versions

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Digital Object Identifier (DOI) | A persistent identifier for a digital object, resolvable through a registration agency. | Assumed to prove the content is correct or the version final. | Prefix and suffix |
| PubMed Identifier (PMID) | An identifier for a record in PubMed. | Confused with a DOI or with the publisher's version. | Integer |
| PubMed Central Identifier (PMCID) | An identifier for a full-text record in PubMed Central. | Assumed to mean the article is open access. | PMC prefix and number |
| International Standard Book Number (ISBN) | An identifier for a specific book edition and format. | Used without the edition, so the citation is ambiguous. | 13-digit code |
| International Standard Serial Number (ISSN) | An identifier for a serial publication. | Assumed unique per journal title change. | 8-digit code |
| Open Researcher and Contributor Identifier (ORCID) | A persistent identifier for a person. | Treated as verification of identity. | 16-digit identifier |
| Version of record | The publisher's final, citable version after peer review and copy-editing. | Assumed identical to the accepted manuscript. | Version label |
| Ahead of print | A version published online before assignment to an issue. | Cited without the eventual volume and pages. | Publication stage |
| Corrigendum | A publisher correction of an error introduced in production. | Confused with a retraction. | Notice type |
| Erratum | A correction, often of author or publisher error. | Ignored when the correction changes a number. | Notice type |
| Expression of concern | A notice that raises a question without retracting the paper. | Treated as a retraction or ignored entirely. | Notice type |
| Retraction | A withdrawal of the paper's validity by the publisher or authors. | Cited from the original PDF without checking status. | Notice and date |

A DOI resolves to a landing page, not to a truth claim. Checking a DOI confirms that the identifier exists and points to the work you think you are citing. It does not confirm that the work is retracted, that the version is current, or that the abstract matches the conclusion.

### 4.4 Status and currency terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Draft guidance | A proposed agency position open for comment. | Cited as the agency's position. | Document status |
| Final guidance | An agency position issued after comment; usually non-binding. | Read as a regulation. | Document status |
| Effective date | The date on which a legal obligation begins to apply. | Confused with the publication date. | Date |
| Transition period | A window during which an older requirement may still be met. | Ignored, producing a false claim of non-compliance. | Date range |
| Superseded | Replaced by a newer version or instrument. | Cited because it is the version in a familiar PDF. | Status flag |
| Withdrawn | Removed by the issuer before replacement. | Assumed to still reflect policy. | Status flag |
| Living guideline | A guideline updated continuously or at short intervals. | Cited without the update date. | Update cadence |
| Legacy document | Material kept for history or for devices already on the market. | Applied to new submissions. | Status flag |
| Binding requirement | An obligation enforceable by law or contract. | Applied to voluntary guidance. | Legal status |
| Non-binding recommendation | Persuasive but not enforceable on its own. | Quoted with "must" as if mandatory. | Legal status |
| Access date | The date on which a web source was read and verified. | Omitted, making the citation unverifiable later. | ISO date |
| Source half-life | The period over which a source type remains adequate for a claim. | Assumed uniform across types. | Years |

### 4.5 The five tests

| Test | Question | Evidence required to pass | Typical failure |
|---|---|---|---|
| Authority | Who is accountable for this statement, and in what capacity? | Named issuing body, author role, and independence from the claim's beneficiary | A vendor or advocacy group presented as neutral |
| Currency | Is this the version in force, and does the claim depend on a moving target? | Version or status check against the issuer's current list | Draft read as final; superseded edition |
| Applicability | Do population, setting, task, and era match the claim? | Explicit comparison of the source's scope to the claim's scope | A single-site study used for a national claim |
| Accessibility | Can a reviewer obtain the source and the specific passage? | Locator, stable link or archive, and page or section | A number that exists only behind a login |
| Provenance | Is the chain from data to conclusion visible and independent? | Methods, data availability, funding, and conflict disclosure | Undisclosed industry authorship |

The tests are conjunctive for primary support. A source that fails authority or provenance cannot be rescued by strong applicability; a source that fails applicability cannot be rescued by prestige. The scoring rubric in section 7 makes this explicit by applying vetoes before weights.

### 4.6 Phrases that signal a source problem

Certain phrases recur in documents where a claim has outrun its evidence. They are not proof of error, but they are reliable prompts to look.

- **"Studies show"** — which studies, in whom, and do they agree.
- **"It is well established"** — established by whom, and since when.
- **"FDA approved"** — approval, clearance, and authorisation are different pathways with different evidence standards.
- **"Clinically proven"** — usually a marketing phrase with no defined endpoint.
- **"Up to"** — a maximum from a favourable subgroup, presented as typical.
- **"Validated"** — validated on what data, by whom, and against what reference standard.
- **"Real-world evidence"** — a data source description, not a study design.
- **"Peer-reviewed"** — a process claim, not a quality claim.
- **"Published research"** — includes predatory venues and unreviewed preprints.
- **"According to industry reports"** — which report, commissioned by whom, with what sample.

The phrases are useful because they are cheap to detect. A document that contains several of them in a single paragraph is a document whose sources should be read before its numbers are reused.

## 5. The mechanics, step by step

The procedure has 18 steps in four phases. Each step states its input, its output, the reason it exists, and the branch or edge case that changes the answer. Steps 1 to 4 frame the claim. Steps 5 to 7 assemble candidates. Steps 8 to 14 apply the five tests and two cross-checks. Steps 15 to 18 reach a verdict, record it, and schedule re-verification.

Do not compress the framing. A team that starts at step 8 with a vague claim will score different sources against different sentences and argue past each other. The framing steps take about fifteen minutes and prevent most of the rework.

### Phase A — Frame the claim and route it

**Step 1. Write the claim as one sentence with five slots.**
Input: the sentence as it appears in the draft, deck, or protocol.
Output: a rewritten sentence naming the population, the intervention or exposure, the outcome, the time horizon, and the setting, with a verb that states the strength of the assertion.
Rationale: every test is relative to these words. "AI improves sepsis outcomes" cannot be scored, because it does not say in whom, over what period, against what comparator, or with what outcome definition. "In adults admitted to a tertiary academic hospital with suspected infection, a proprietary sepsis alert reduces 30-day all-cause mortality compared with usual care" can be scored.
Branch: if the sentence contains a comparison, add the comparator as a sixth slot. If it contains a causal verb, record whether the design can support causation.
Edge case: the claim is about a class of products rather than one product. Then the unit of evidence is a systematic review, and a single-product trial can only support the single-product sub-claim.

The verb is the cheapest signal of what design is required, and it is the part of a sentence that editors and reviewers change last:

| Verb in the claim | Design required | What is insufficient |
|---|---|---|
| reduces, causes, prevents | Randomised trial or strong quasi-experimental design with a counterfactual | Pre-post comparison, correlation, mechanism |
| is associated with | Observational study with the relevant adjustment set stated | Unadjusted group comparison |
| predicts, detects, classifies | External validation in the target population and setting | Internal cross-validation, training-set performance |
| is safe | Controlled study or surveillance with a denominator and a ascertainment method | Case reports, spontaneous reports without denominators |
| is required, is permitted | The legal instrument, with article and effective date | Guidance, FAQ, law firm summary |
| costs, saves | Costing study with perspective, year, and currency | List price, vendor claim, charge data used as a proxy for cost |
| is used by, is adopted by | Workflow data or a survey with a sampling frame | Anecdote, conference poll |

**Step 2. Classify the claim type and set the authority floor.**
Input: the framed sentence.
Output: one of six labels — regulatory, efficacy or safety, methodological, definitional or clinical background, market or financial, operational — plus the minimum source that can support it.
Rationale: the claim type determines what evidence could possibly settle it. A regulatory obligation cannot be settled by a paper about the regulation. A definitional claim cannot be settled by a trial.
Branch: if the sentence mixes types, split it (step 3).
Edge case: a claim about what clinicians do in practice is an operational claim, not a clinical one. It needs workflow data or a survey with a disclosed sampling frame, not a guideline.

A useful diagnostic is to ask what document would have to exist for the sentence to be false. If the answer is "a different regulation," the claim is regulatory. If it is "a trial with a different result," the claim is empirical. If it is "nothing, because the sentence is a definition," the claim is definitional and needs a standards source rather than evidence.

| Claim type | Minimum acceptable source | Cannot support it | Typical error |
|---|---|---|---|
| Regulatory obligation | The legal instrument, or the agency's own page stating the requirement and its effective date | Law firm summary, vendor compliance page, news article | Citing guidance as law |
| Efficacy or safety | Comparative study in the target population, or a systematic review of such studies | Mechanism paper, bench study, case series, vendor benchmark | Citing a preprint as settled |
| Methodological | Peer-reviewed methodological paper or an authoritative text, plus the software documentation for the implementation | A blog tutorial, a Stack Overflow answer | Citing a library as if it were a method |
| Definitional or clinical background | Consensus statement, guideline, standards body, or a current textbook | A single review article, a slide deck | Using an outdated definition |
| Market or financial | Audited filing, government statistic, or a report with a disclosed method and sample | Vendor white paper, press release, undated web page | Treating a vendor estimate as a market measurement |
| Operational | Workflow study, audit, or survey with a sampling frame | Expert opinion alone | Generalising from one site |

**Step 3. Split the sentence into independently checkable sub-claims.**
Input: the framed sentence and the surrounding paragraph.
Output: a numbered list of sub-claims, each with its own type and its own source requirement.
Rationale: compound sentences hide unsupported parts. "The FDA has authorized more than 1,600 AI devices, and most are radiology tools with limited external validation" contains a count, a classification, and a quality judgement, each from a different source.
Branch: if a sub-claim is interpretive rather than factual, label it as interpretation and require an argument rather than a citation.
Edge case: a causal chain ("the alert shortens length of stay, which reduces cost") is two claims; the second needs a costing source even when the first is well supported.

A worked split makes the value visible. The sentence "AI triage reduced radiologist workload by 44% and maintained diagnostic accuracy" becomes four sub-claims: (a) a specific AI triage system was deployed in a named setting; (b) the measured workload metric fell by 44% relative to a stated baseline; (c) diagnostic accuracy was non-inferior on a stated reference standard; (d) the setting is comparable to the reader's. Sub-claim (b) needs the trial's primary result, (c) needs the accuracy analysis, (a) needs the product and version, and (d) is an inference the reader must make. A single citation can support (a) to (c) but not (d), and a sentence that implies (d) is making a claim the paper did not test.

**Step 4. Fix the review date and the decision deadline.**
Input: the project calendar.
Output: an ISO review date and a statement of how long the source must remain valid.
Rationale: currency is measured against the review date, and the required validity period depends on the decision. A board deck read once needs less durability than a regulatory submission that will be reviewed over eighteen months.
Branch: if the decision deadline is after the source's expected update, plan to re-verify before submission.
Edge case: a source can expire between drafting and publication. Record the date on which the claim was last checked, not the date the document was written.

The four framing steps produce a one-page brief that the rest of the procedure consumes. If the brief cannot be written, the claim is not ready for sourcing, and the correct next step is to return to the document and rewrite the sentence.

| Branch at framing | Resolution |
|---|---|
| The sentence has no population | Name the population or narrow the claim to a mechanism |
| The sentence has no comparator | Either add one or change the verb to "is associated with" |
| The sentence mixes regulatory and clinical claims | Split and source separately |
| The claim is about a class of products | Use a systematic review; do not generalise from one product |
| The claim is a prediction about the future | Treat as a forecast, with a scenario and a stated base rate |
| The claim is normative ("should") | Separate the empirical premise from the value judgement |
| The team cannot agree on the outcome definition | Source the definition first; it is a separate claim |
| The decision deadline is shorter than the verification | State the residual uncertainty explicitly in the document |

### Phase B — Assemble the candidate set

**Step 5. Build a candidate set of at least three sources with different roles.**
Input: the sub-claims and a search across the relevant indexes.
Output: a table with one best-available source, one independent confirmation, and one definitional or contextual source.
Rationale: a single source produces a justification, not a judgement. Three roles force the reviewer to ask whether the claim has independent support and whether the terms are being used consistently.
Branch: if only one source exists, record "single source" and cap the verdict at "cite with limitation" regardless of the score.
Edge case: three papers from the same group, using the same cohort, are one source for independence purposes. Count cohorts, not PDFs.

Where to look depends on the claim type, and searching the wrong index produces confident nonsense:

| Claim type | First index | Second index | Avoid as first stop |
|---|---|---|---|
| Regulatory | Agency guidance database, Federal Register, EUR-Lex, official journal | Law firm tracker for orientation, then verify | News articles, vendor compliance pages |
| Efficacy or safety | PubMed or Embase, then the reference lists of a systematic review | Trial registries for unpublished and terminated studies | Google Scholar ranking alone |
| Methodological | PubMed, journal of record, software documentation | Statistical society guidance | Tutorials and course notes |
| Definitional | Standards body, consensus statement, current guideline | Society glossary | Wikipedia, even when it is accurate |
| Market | Company filings, government statistics, payer data | Analyst reports with disclosed method | Vendor white papers, press releases |
| Operational | Workflow studies, audit reports, survey with sampling frame | Society workforce reports | Anecdote from one site |

**Step 6. Trace the citation chain backwards to the primary source.**
Input: a review, guideline, or news article that states the claim.
Output: the primary study or dataset the statement ultimately rests on, or a finding that no primary source exists.
Rationale: secondary sources compress and sometimes invert. A guideline sentence may cite a review that cites a study that measured a surrogate endpoint, while the guideline sentence names a clinical outcome.
Branch: if the chain ends in a press release or an unpublished analysis, the claim is unsupported at the level asserted.
Edge case: circular citation. Two reviews can cite each other, and a guideline can cite a review that cites the guideline. Stop when a citation loop appears and record it.

**Step 7. Search the negative space around each candidate.**
Input: the candidate's DOI, title, and authors.
Output: a status check covering retraction and correction notices, later contradictory studies, and any regulatory or institutional action.
Rationale: a paper's own PDF never reports that it was retracted. The status has to be checked in a separate index.
Branch: if a retraction or expression of concern exists, the source is excluded unless the claim is specifically about the retracted work.
Edge case: a retracted paper can remain in a citation network for years, and some reference managers keep the original metadata. Check the DOI landing page rather than the saved PDF.

The status check has three levels, and skipping the third is common. The first is the publisher's landing page, which carries the retraction or correction notice. The second is the bibliographic index, where the record may be tagged. The third is the integrity database, which can show an expression of concern that the publisher has not linked to the article. When the three disagree, the most recent notice from the publisher governs, and the discrepancy itself is worth recording.

### Phase C — Apply the tests

**Step 8. Test authority.**
Input: the candidate's issuing body, author affiliations, and funding.
Output: a pass, conditional, or fail, with one line naming who is accountable and what interest they have in the claim.
Rationale: authority is about accountability and independence, not prestige. A regulator is authoritative for its own requirements. A professional society is authoritative for its own guideline. A vendor is authoritative for what its product does, not for whether it works.
Branch: if the issuer benefits commercially from the claim, the source can describe the product but cannot be primary support for its efficacy.
Edge case: an author with a relevant patent or equity stake is a disclosed conflict, not automatically a disqualification. Undisclosed is the failure.

Independence is checked, not assumed. Read the funding statement, the conflict-of-interest section, and the author list against the entity that benefits from the claim. A study funded by a manufacturer can still be well conducted; it cannot be the only support for a claim about that manufacturer's product. The same logic applies to a professional society that owns a registry, a consultancy that sells benchmarking data, and a foundation that advocates for a therapy area.

| Issuer | Can support | Cannot support | Caveat |
|---|---|---|---|
| Regulator | Its own legal requirements, effective dates, and decisions | Clinical efficacy beyond its published review | Guidance is usually non-binding |
| Standards body | The content of its standard | Legal obligation unless incorporated by reference | Editions and amendments matter |
| Peer-reviewed journal | Empirical claims within the study's scope | Claims outside the studied population or time horizon | Check retraction and correction status |
| Professional society | Guidelines and position statements | Independent empirical confirmation | Guideline panels can have conflicts |
| Vendor | Product specifications and intended use | Efficacy, comparative performance, market size | Promotional genre, commercial interest |
| Consultancy | Market structure and reported transactions | Clinical evidence | Method and client base often undisclosed |
| News outlet | That an event was reported | The underlying scientific or legal fact | Secondary by construction |

**Step 9. Test currency.**
Input: the document's version, status, and effective date, checked against the issuer's current list.
Output: one of five states: current, current with a transition period, superseded but applicable to legacy products, superseded, or withdrawn.
Rationale: currency is a property of the document's legal or scientific status, not of the web page's last-modified date.
Branch: a superseded document can still be correct for a device already on the market under a transition provision. Record the transition window.
Edge case: guidance is often revised without a new title. The revision date and the document number are the reliable discriminators.

Status is checked at the issuer, not at the copy. For FDA material, the guidance database and the Federal Register notice give the issue date and the docket number. For EU material, EUR-Lex gives the consolidated version and the date of application, while the Medical Device Coordination Group (MDCG) guidance list gives the endorsement date. For standards, the national body's catalogue gives the edition, the amendment, and whether the edition has been withdrawn. A PDF downloaded two years ago carries no status information at all.

| Currency state | Test result | Action | Example pattern |
|---|---|---|---|
| Current | Pass | Cite with version and date | Final guidance listed on the agency's current page |
| Current with transition | Conditional | Cite both the old and new rule, with dates | MDR transition provisions for legacy devices |
| Superseded, legacy applicable | Conditional | Cite only for the legacy scope, state the boundary | A standard edition retained for certified products |
| Superseded | Fail for new work | Replace with the current version | A 2019 textbook edition replaced in 2023 |
| Withdrawn | Fail | Do not cite as policy; cite only as history | A guidance withdrawn after legal challenge |

**Step 10. Test applicability on four dimensions.**
Input: the source's stated population, setting, task, and dates.
Output: four separate judgements, each pass, conditional, or fail, plus the direction of the likely bias.
Rationale: applicability is not one question. A study can match on population and fail on era, or match on task and fail on setting.
Branch: if the claim is about a subgroup and the source reports only an overall effect, the source is conditional at best. Subgroup effects in a trial that was not powered for them are hypothesis-generating.
Edge case: a model validated on data from one country's coding system may fail applicability in another country even when the clinical population is similar.

A compact way to record applicability is a four-cell row, because a single "applicable" judgement hides which dimension failed:

| Dimension | Source says | Claim says | Result |
|---|---|---|---|
| Population | Adults 18 to 89, suspected infection | Adults in medical and surgical wards | Conditional: excludes obstetric and paediatric patients |
| Setting | 27 hospitals in one integrated system | 90-bed community hospital | Fail for the national claim; conditional for the system |
| Task | Flag within 6 hours of arrival | Flag at triage | Fail: the timing does not match the workflow |
| Era | Data 2018 to 2019 | Deployment in 2027 | Conditional: coding and treatment have changed |

The direction of the likely bias belongs in the same row. If the source population is healthier than the deployment population, the reported performance is probably optimistic. If the source era predates a definition change, the label may mean something different.

Era deserves its own paragraph because it fails silently. A cohort that spans a definition change, a coding-system change, or a treatment-guideline change is not one population. Sepsis definitions changed in 2016; chronic obstructive pulmonary disease grouping changed in 2023 and again in 2026; coding moved from ICD-9 to ICD-10 in the United States in 2015. A claim that ignores the boundary is comparing measurements made with different instruments.

**Step 11. Test accessibility and archive the passage.**
Input: the locator, the access route, and the specific page, section, or table that supports the claim.
Output: a stable link or archive reference, the access date, and the quoted passage.
Rationale: a claim that cannot be checked is not evidence. Accessibility is also a durability question, because web pages move and documents are withdrawn.
Branch: if the source is paywalled, record the abstract, the DOI, and the specific claim that can be checked from the abstract; do not cite a number that appears only in the full text you have not seen.
Edge case: standards are often paywalled and cannot be quoted at length. Cite the clause number and the edition, and state that the clause was read under licence.

Link rot is a real source-quality problem, not a formatting annoyance. Government pages are reorganised, society pages are rebuilt, and publisher pages change their article identifiers. The defence is twofold: record the DOI or the official document number, which survives reorganisation, and archive the exact page you read in a web archive with a dated snapshot. A citation that depends on a URL alone is one site migration away from unverifiable.

**Step 12. Test provenance.**
Input: the methods section, data availability statement, funding, conflicts, and any post-publication correspondence.
Output: a pass, conditional, or fail, with the specific gap named.
Rationale: provenance is the chain from observation to conclusion. A paper can be peer-reviewed and still have an invisible chain: no data, no code, an undisclosed sponsor, or an analysis that changed after review.
Branch: if the data are unavailable and the claim depends on a derived number, mark the claim as unverifiable rather than false.
Edge case: paper-mill indicators include tortured phrases, citation padding, implausible author productivity, and peer review handled by a guest editor with no relevant expertise. These are signals, not proof; escalate to the journal and the integrity database.

Four provenance checks cover most cases. First, does the data availability statement resolve to data a reviewer could actually obtain, or to a request form that is never answered. Second, is there code, and does it reproduce the headline number. Third, does the funding statement name an entity with an interest in the result. Fourth, does the analysis in the methods match the analysis reported in the results, including the pre-specified primary outcome. A paper can pass peer review and fail all four.

Derived numbers need a chain of custody. If a paper reports a cost saving, the chain runs from the source of each unit cost, through the utilisation assumption, to the arithmetic in the appendix. If any link is missing, the number is an estimate with an unstated method. This matters most for figures that travel into slides, where the appendix does not follow.

**Step 13. Check peer-review status and compare versions.**
Input: the preprint, the accepted manuscript, and the version of record.
Output: a statement of which version supports the claim and whether the claim changed between versions.
Rationale: peer review is a filter, not a guarantee, but the version difference is informative. Endpoints, sample sizes, and conclusions can change.
Branch: if the claim appears only in the preprint and was removed in the published version, the claim is unsupported.
Edge case: a preprint that was never published is not automatically wrong; it is unreviewed, and the verdict should say so.

Version comparison is mechanical. Put the preprint, the accepted manuscript, and the version of record side by side and compare five fields: sample size, primary outcome, effect estimate and interval, subgroup analyses, and the sentence in the abstract that states the conclusion. Any change in one of those fields is material. Changes to wording elsewhere are usually not, unless the wording changes the clinical action implied.

**Step 14. Count independent evidence, not papers.**
Input: the candidate set and the underlying cohorts, datasets, or trials.
Output: the number of independent data sources and a statement of overlap.
Rationale: ten papers from one registry are one source for independence. Reviews that pool the same trials do not multiply the evidence.
Branch: if all candidates descend from one study, the claim has single-study support regardless of how many citations are attached.
Edge case: a meta-analysis that includes the same trial twice, or includes a preprint and its published version as separate studies, overstates the evidence base.

A concrete count: if a claim rests on three papers, one reporting a trial, one reporting a secondary analysis of the same trial, and one meta-analysis that includes that trial, the number of independent data sources is one. The correct verdict for a claim about the intervention is single-trial support with whatever limitations that trial has. The citation count is three and the evidence count is one, and reviewers who confuse the two produce the appearance of replication without its substance.

### Phase D — Verdict, record, and monitor

**Step 15. Apply vetoes and caps before any weighted score.**
Input: the five test results and the interest assessment.
Output: a list of excluded candidates and a list of candidates with a capped verdict.
Rationale: a weighted mean lets a strong score on one criterion compensate for a disqualifying zero. An inaccessible vendor claim with perfect applicability is still not evidence.
Branch: if every candidate is vetoed, the claim is unsupported and the deliverable changes from a citation to a data-collection or wording change.
Edge case: a source can be excluded as support and retained as a pointer to the primary source it summarises.

The distinction between a veto and a cap is practical. A veto says the source cannot be used at all, as with an anonymous claim or a paper with no disclosed method. A cap says the source can be used for a narrower role, as with a vendor study that can describe the product but not establish its efficacy. Applying a cap rather than a veto keeps useful material in the record while preventing it from carrying weight it cannot bear.

**Step 16. Assign one of four verdicts.**
Input: the surviving candidates.
Output: primary support, cite with limitation, background only, or reject, attached to each sub-claim.
Rationale: a verdict forces a decision and makes disagreement visible. "It is a useful paper" is not a verdict.
Branch: if the best candidate is conditional, write the limitation into the sentence rather than into a footnote nobody reads.
Edge case: different sub-claims can receive different verdicts from the same source. Record them separately.

The verdict should appear in the sentence, not only in the reference list. Four phrasings, in descending order of strength:

| Verdict | Sentence pattern | When to use |
|---|---|---|
| Primary support | "X reduced Y in a randomised trial of Z patients." | Source directly tested the claim in the relevant population |
| Cite with limitation | "In one academic health system, X was associated with Y; the study did not measure Z." | Direction matches but scope, design, or outcome differs |
| Background only | "X has been proposed as a way to Y." | Source motivates the question without testing it |
| Reject | No citation; rewrite or collect evidence | Source fails a test or cannot be obtained |

The phrase "studies show" is a signal that no single source was checked. It hides whether the studies agree, what they measured, and whether they are independent. Replace it with a named result and a named population.

**Step 17. Write the ledger entry.**
Input: the claim, candidates, test results, verdict, and access date.
Output: a row per sub-claim with the fields: claim, source, version, tests, verdict, access date, reviewer, and what would change the verdict.
Rationale: the ledger is the artefact that survives staff turnover. It also converts a judgement into something a second reviewer can audit.
Branch: if the verdict depends on an unstated assumption, put the assumption in the row rather than in the prose.
Edge case: a source used in several documents should have one ledger row, referenced by each document, so that a retraction propagates to every use.

A ledger row is short enough to write in a spreadsheet and specific enough to audit:

| Field | Example |
|---|---|
| Sub-claim | A proprietary sepsis alert reduces 30-day mortality in admitted adults |
| Source and version | Wong et al., JAMA Internal Medicine 2021;181(8):1065-1070, version of record |
| Authority | Peer-reviewed external validation; authors independent of the vendor |
| Currency | Current for the model version tested; vendor has released later versions |
| Applicability | One academic health system; discrimination only, no mortality outcome |
| Accessibility | DOI and PubMed Central full text; table 2 read |
| Provenance | Methods and cohort described; vendor scores available under agreement |
| Verdict | Cite with limitation: performance in one system, not mortality benefit |
| Access date | 2026-10-26 |
| Would change if | An external cluster trial shows a mortality difference, or the model version changes |

**Step 18. Set a re-verification cadence and explicit triggers.**
Input: the source class, the decision horizon, and the issuer's update pattern.
Output: a review interval and a list of triggers that force an earlier check.
Rationale: currency decays at different rates. A regulation changes on a legislative calendar; a market estimate changes quarterly; a textbook changes on an edition cycle.
Branch: if the source is a living guideline or a continuously updated database, check for an update before every external use.
Edge case: a trigger can be an event rather than a date: a new draft guidance, a retraction, a label change, or a safety communication.

Cadence follows volatility, not habit:

| Source class | Default interval | Trigger for earlier review |
|---|---|---|
| Regulation or statute | 12 months, or on legislative calendar | Consolidated version changes, transition deadline |
| Agency guidance | 6 months | New draft, revision, withdrawal, Federal Register notice |
| Standard | 12 months, or on amendment | New edition, amendment, harmonised-standard listing |
| Peer-reviewed study | No scheduled review; status check at each reuse | Retraction, correction, contradicting trial |
| Living guideline | Before each external use | Any update notice |
| Dataset | On each download | New version, licence change, deprecation |
| Market report | 3 months | New quarter, revised methodology, client-only update |
| Textbook | On edition cycle | New edition, errata, guideline change in the subject |

The procedure is deliberately front-loaded. Steps 1 to 4 cost about fifteen minutes; steps 8 to 14 cost an hour for a serious claim; steps 15 to 18 cost twenty minutes and save the most time later, because they are what makes the next review cheap.

### A branch worked in miniature

The following branch is the one reviewers most often get wrong, because both paths feel like success.

A team writes: "Our model detects sepsis 12 hours earlier than the existing alert." The claim type is efficacy or safety, and the verb is comparative. Step 8 finds a peer-reviewed external validation, which passes authority. Step 9 finds the paper is current. Step 10 compares the source's task to the claim: the paper reports discrimination (an area under the receiver operating characteristic curve) at several lead times, not a head-to-head comparison with another alert. The task dimension fails. Step 15 therefore caps the verdict at "background only" for the comparative sentence, even though the paper is excellent.

The branch has two legitimate resolutions. Either the claim is rewritten to match what the paper measured ("the model's discrimination at 12 hours was X in one health system"), or the team finds a comparative study. Rewriting the claim is usually correct and always cheaper. The failure mode is keeping the sentence and citing the paper anyway, because the paper is strong and the sentence sounds like what the paper found.

### What the procedure does not do

It does not tell you whether a well-conducted study is correct. It tells you whether the study can carry the sentence you want to write. Those are different questions, and conflating them is how teams end up defending a citation rather than a claim.

It does not replace domain expertise. A clinician is needed to judge whether an outcome definition is meaningful, and a lawyer is needed to judge whether a requirement applies. The procedure organises their input; it does not substitute for it.

It does not eliminate judgement. Two reviewers can score the same source differently and both be right, because they are supporting different sentences. The ledger makes the disagreement visible, which is the point.

The most common misapplication is to run the procedure once and treat the result as permanent. Sources change status, claims drift as documents are edited, and a verdict that was correct at drafting can be wrong at submission. The cadence in step 18 exists because source evaluation is a maintenance task, not a one-off gate.

A second misapplication is to apply the full procedure to every sentence. It is designed for claims that carry weight: a regulatory statement, a clinical recommendation, a performance number, a market figure. Background sentences and definitions of common terms do not need a scored ledger.

> **Note:** The 18 steps assume the output will be read by someone who may challenge it. For an internal note, steps 1 to 4 can be compressed. Do not compress them for anything that leaves the team, because the compressed version is where the claim and the source drift apart.

## 6. Worked example

The claim comes from a hospital strategy document: "Our sepsis alert is validated at an area under the receiver operating characteristic curve (AUC) of 0.76 to 0.83 and has reduced sepsis mortality since deployment." The sentence carries three sub-claims: a performance figure, a source for that figure, and a mortality effect. The team wants to reuse the sentence in a payer presentation.

Review date is 26 October 2026. The decision deadline is four weeks away. The claim must remain valid for the duration of a contract cycle, so about twelve months.

### Step 1 to 4: framing

The framed sub-claims are:

1. **Performance.** In adults hospitalised with suspected infection, the deployed alert discriminates sepsis at AUC 0.76 to 0.83.
2. **Provenance of the figure.** That figure comes from a validation the team can inspect.
3. **Effect.** Deployment of the alert reduced sepsis mortality relative to usual care.

Sub-claim 1 is a prediction-performance claim. Sub-claim 2 is a provenance claim about sub-claim 1. Sub-claim 3 is an efficacy claim and needs a comparative design with a mortality outcome. The verb "has reduced" requires a counterfactual. The setting is not stated, which is itself a defect.

| Sub-claim | Type | Design that could support it | Design that cannot |
|---|---|---|---|
| 1. AUC 0.76 to 0.83 | Prediction performance | External validation in the target population | Internal cross-validation, vendor benchmark |
| 2. Figure is inspectable | Provenance | Published methods, cohort, and analysis | Marketing page, conference abstract |
| 3. Mortality reduced | Efficacy | Controlled comparison with a counterfactual | Before-and-after series, press release, AUC |

### Step 5 to 7: candidates

Five candidates were assembled. No candidate was found that directly tested mortality in a controlled design.

| # | Candidate | Type | Role in the set |
|---|---|---|---|
| C1 | Vendor product page describing the model and its validation | Vendor material | The origin of the 0.76 to 0.83 figure |
| C2 | Wong et al., *JAMA Internal Medicine*, 2021 | Peer-reviewed external validation | Independent test of discrimination |
| C3 | Accompanying editorial in the same issue | Editorial | Independent interpretation by methods experts |
| C4 | Hospital press release announcing deployment | Press release | The source of the "reduced mortality" wording |
| C5 | Vendor-coauthored conference proceeding reporting AUC 0.73 | Conference paper | The vendor's own earlier validation |

The citation chain is short and revealing. The strategy document's figure traces to C1. The mortality sentence traces to C4, which cites no study. C2 and C3 were not in the original reference list at all.

### Step 8 to 14: scoring each candidate

**C1, vendor product page.**
- Authority: commercial. The issuer benefits from the claim, so the page can describe the product and cannot establish its performance.
- Currency: passes for the product version described, but the page does not state which model version the figure refers to.
- Applicability: unstated. No cohort, setting, or outcome definition is given.
- Accessibility: passes, because the page is public.
- Provenance: fails, because no method or cohort is disclosed.
- Verdict before scoring: conflicted, background only.

**C2, external validation.**
- Authority: passes. Peer-reviewed, and the authors are independent of the vendor.
- Currency: adequate for the model version tested. Later versions exist and would need separate validation.
- Applicability: strong on population, moderate on setting. It is one academic health system, which limits a national claim but not a claim about that system.
- Accessibility: passes. The article is in PubMed Central and the tables are readable.
- Provenance: passes. Cohort, period, outcome definition, and analysis are described, and the vendor scores were available under an agreement.
- Verdict: primary support for the performance sub-claim, with the setting limitation stated.

**C3, editorial.**
- Authority: passes as independent expert interpretation.
- Currency: current for the study it discusses.
- Applicability: same population as C2, so it inherits the same setting limit.
- Accessibility: passes.
- Provenance: passes, but it is not independent evidence, because it analyses the same study.
- Verdict: background and interpretation, not support.

**C4, hospital press release.**
- Authority: fails for an efficacy claim. The issuing hospital has an institutional interest and the release reports no method.
- Currency: passes trivially, in that it is recent.
- Applicability: cannot be assessed, because no population or outcome is defined.
- Accessibility: passes.
- Provenance: fails. No numbers, no cohort, no comparator.
- Verdict: reject as support. It is evidence only that the hospital made the announcement.

**C5, vendor-coauthored conference proceeding.**
- Authority: capped by the conference genre and by the commercial interest.
- Currency: passes for the period it covers.
- Applicability: partial, because it reports internal or semi-internal validation.
- Accessibility: partial, because only the abstract is available.
- Provenance: partial. Methods are summarised and data are not available.
- Verdict: background only, and only for the vendor's own claim about its model.

The five records take about twenty minutes to assemble and produce a comparison the original document never had. Two of the five candidates cannot be used at all, and the strongest candidate was absent from the original reference list.

### The numbers, checked

The external validation reported an AUC of 0.63 (95% confidence interval 0.62 to 0.64) at the hospitalisation level, in 27,697 patients and 38,455 hospitalisations, with a sepsis incidence of 7%. The vendor's stated range was 0.76 to 0.83. The gap is 0.13 to 0.20 AUC points, which is large for a deployed model and is not a sampling artefact: the confidence interval excludes the vendor's lower bound by a wide margin.

At the deployed threshold of 6 or more, the study reported sensitivity 33%, specificity 83%, positive predictive value 12%, and negative predictive value 95%. Alerts fired in 6,971 of 38,455 hospitalisations. Two arithmetic checks are worth doing by hand:

- Alert rate: 6,971 / 38,455 = 0.1813, or 18.1%, which matches the reported 18%.
- Number needed to evaluate: 1 / 0.12 = 8.3, which matches the reported 8.

The clinical reading of those numbers is the part a table cannot supply. Two thirds of sepsis cases were not flagged, and eight alerts had to be reviewed for each true case. A model can have a respectable AUC and still produce an alert burden that wards will learn to ignore.

The full test record for C2, which is the source the verdict rests on:

| Test | Result | Evidence |
|---|---|---|
| Authority | Pass | Peer-reviewed, independent authors, no vendor funding disclosed |
| Currency | Conditional | Model version tested; later versions need separate validation |
| Applicability | Conditional | Adult inpatients in one academic system; no mortality outcome |
| Accessibility | Pass | PubMed Central full text, DOI, tables readable |
| Provenance | Pass | Cohort, period, outcome definition, and analysis reported |
| Independence | Pass | Not funded by the vendor; scores obtained under agreement |

A single conditional test is enough to attach a limitation to the sentence. It is not enough to discard the source, and the distinction matters: discarding good sources produces a different kind of error, in which a real finding is treated as unusable because it is imperfect.

### The verdict

| Sub-claim | Best source | Verdict | Sentence that is supportable |
|---|---|---|---|
| Performance at AUC 0.76 to 0.83 | C2 contradicts it | Reject as stated | "External validation in one academic health system reported an AUC of 0.63, below the vendor's stated range of 0.76 to 0.83." |
| The figure has an inspectable source | C1 only | Reject | No supportable sentence; the figure is a vendor claim without a disclosed cohort |
| Deployment reduced mortality | C4 only | Reject | "The hospital announced the deployment; no controlled evaluation of mortality was identified." |

The overall verdict is that the original sentence cannot be used in the payer presentation. The performance claim is not merely unsupported; it is contradicted by the best available independent evidence. The mortality claim has no comparative source at all.

This is the outcome the procedure is designed to produce: not "find a better citation for the same sentence" but "the sentence is not supportable and must change." The team has three honest options. It can present the external validation result with its setting limitation. It can present the vendor figure as a vendor claim, clearly labelled. Or it can commission or find a prospective evaluation, which is a different project with a different timeline.

### Where the original sentence went wrong

The sentence did not fail because the team was careless. It failed at three specific points, and each is a recognisable pattern:

- **The figure was inherited without a source.** The 0.76 to 0.83 range had been in the organisation's materials since before anyone currently on the team joined. Nobody owned it, so nobody checked it.
- **The genre was upgraded.** A vendor page became "validated" in the retelling, and "validated" became "reduced mortality" in the strategy document. Each step was small; the cumulative distance is large.
- **The comparator disappeared.** The verb "has reduced" survived from a draft in which a before-and-after internal review had been cited. That review had no control group and was removed for other reasons, but its verb stayed.

The pattern generalises. Most unsupportable claims in industry documents are not invented; they are inherited, upgraded, and detached from their comparator.

### What the team could write instead

Three replacement sentences, each matched to the evidence:

1. "In an external validation at one academic health system, the alert's AUC was 0.63, below the vendor's stated 0.76 to 0.83."
2. "The vendor states an AUC of 0.76 to 0.83; no independent validation of the currently deployed version was identified."
3. "The alert is deployed at our hospital; we have not identified a controlled evaluation of its effect on mortality."

Each sentence is weaker than the original and each is true. The third is the one most likely to be useful in a payer conversation, because it invites a pilot rather than making a claim that a reviewer can dismantle.

The replacement also changes the burden of proof. The original sentence asked the reader to accept a benefit. The replacements ask the reader to accept a measurement, which is a smaller request and a more defensible one.

### Two diagnostics that would change the verdict

**Diagnostic 1: a prospective controlled evaluation in comparable hospitals.** A cluster-randomised or stepped-wedge trial showing a mortality difference, with the alert as the intervention and usual care as the comparator, would support sub-claim 3. It would not repair sub-claim 1, because discrimination and effectiveness are different properties. A trial that shows benefit despite modest discrimination is entirely possible, and it would change the recommendation while leaving the AUC statement false.

**Diagnostic 2: evidence that the validated model is not the deployed model.** The external validation tested a specific model version. If the vendor can show that the deployed version is a later model with an independent external validation reporting AUC at or above 0.75 in a comparable setting, sub-claim 1 becomes supportable for that version, with the setting limitation stated. The diagnostic is version-specific, not vendor-specific: a new validation of a different product does not transfer.

A third diagnostic, weaker but common, is a reanalysis of the same health system's data after a workflow change. It would raise a hypothesis, not settle the claim, because it lacks a counterfactual.

### What would not change the verdict

- A second vendor document repeating the same 0.76 to 0.83 range. Repetition is not independent evidence.
- A conference abstract reporting better performance in a different product. The claim is about this product.
- A clinician's favourable impression of the alert. Experience is a reason to study the alert, not a result.
- A reduction in alert-to-antibiotic time at one site, without a control group. It is a process measure with no counterfactual.
- A statistical correction to the published validation that changes the AUC by 0.01. The gap to the vendor range is 0.13 or more.
- A testimonial from a clinician at a site that uses the product. It is a reason to run a pilot, not evidence that the pilot succeeded.
- A claim that the model has been "cleared" by a regulator. Clearance establishes that a submission met a standard; it does not establish the performance claim in the sentence.
- A larger internal dataset from the same health system. It would strengthen a local claim and would not address the setting limitation for a national one.
- A revision of the vendor's marketing page that removes the 0.76 to 0.83 range. Removing a claim does not make the opposite claim true; it leaves the performance question open.
- A meta-analysis that pools this model with unrelated sepsis alerts. Pooling different interventions cannot validate this one.

### What the team should record

The ledger entry has one row per sub-claim, the access date of 26 October 2026, and a trigger to re-check if the vendor publishes a validation of a later model version or if a trial is reported. The original sentence is retained in the document's history with a note explaining why it was removed, because the next author will otherwise reintroduce it from an older draft.

The episode also produces a reusable finding: the vendor's performance range had been circulating in the organisation for two years without anyone checking whether an independent validation existed. One existed, was open access, and had been published in 2021. The cost of the check was about forty minutes.

A final observation about the case. The hospital's use of the alert may still be justified on other grounds: earlier antibiotics, better triage, or a workflow benefit that the validation did not measure. The verdict concerns the sentence, not the product decision. This distinction is important in practice, because teams defend a product decision by defending a sentence, and the two are not the same.

The ledger row in this case took about ten minutes to write and replaced an argument that had recurred in three meetings. That ratio, ten minutes against three meetings, is the practical case for the procedure.

## 7. Code, query, or calculation

The script below implements the rubric used in this topic. It is standard library only (Python 3.9 or later), takes about a second to run, and prints a ranked table plus the component scores that produced it. The logic is deliberately explicit: ceilings, half-lives, weights, vetoes, and interest caps are separate policy choices that a reviewer can inspect and change.

```python
from dataclasses import dataclass
from datetime import date

REVIEW = date(2026, 10, 26)
CEILING = {"systematic_review": 4, "rct": 4, "peer_reviewed_study": 3,
           "guideline": 3, "conference_paper": 2, "preprint": 2,
           "press_release": 1, "blog": 1}
HALF_LIFE = {"systematic_review": 3.0, "rct": 10.0, "peer_reviewed_study": 8.0,
             "guideline": 5.0, "conference_paper": 5.0, "preprint": 2.0,
             "press_release": 0.5, "blog": 1.0}
W = {"authority": .30, "currency": .15, "applicability": .25,
     "accessibility": .10, "provenance": .20}
PROMOTIONAL = {"press_release", "blog"}


@dataclass
class Source:
    name: str
    kind: str
    year: int
    authority: int
    applicability: int
    accessibility: int
    provenance: int
    interest: str = "none"


def currency(year, kind):
    age, half = max(REVIEW.year - year, 0), HALF_LIFE[kind]
    return 3 if age <= half / 2 else 2 if age <= half else 1 if age <= half * 2 else 0


def score(s, claim="efficacy"):
    a = min(s.authority, CEILING[s.kind])
    c = currency(s.year, s.kind)
    p = min(s.applicability, 1) if s.accessibility == 0 else s.applicability
    raw = dict(authority=a, currency=c, applicability=p,
               accessibility=s.accessibility, provenance=s.provenance)
    pct = round(100 * sum(W[k] * raw[k] / cap for k, cap in
                          [("authority", 4), ("currency", 3), ("applicability", 4),
                           ("accessibility", 3), ("provenance", 4)]), 1)
    if s.authority == 0:
        return pct, raw, "no accountable author", "excluded"
    if s.provenance == 0:
        return pct, raw, "no methods disclosed", "excluded"
    if s.interest == "commercial" and claim in ("efficacy", "safety"):
        return pct, raw, None, "conflicted: background only"
    if s.kind in PROMOTIONAL and claim in ("efficacy", "safety"):
        return pct, raw, None, "promotional: background only"
    verdict = ("primary support" if pct >= 75 else "cite with limitation"
               if pct >= 55 else "background only" if pct >= 35 else "do not cite")
    return pct, raw, None, verdict


SOURCES = [
    Source("Vendor product page (sepsis alert)", "press_release", 2026, 1, 2, 3, 1, "commercial"),
    Source("Wong 2021, JAMA Intern Med", "peer_reviewed_study", 2021, 4, 4, 3, 4),
    Source("Vendor-sponsored conference abstract", "conference_paper", 2025, 1, 3, 2, 2, "commercial"),
    Source("Hospital press release on deployment", "press_release", 2026, 1, 2, 2, 1),
    Source("Living systematic review of sepsis alerts", "systematic_review", 2026, 4, 3, 3, 4),
    Source("Anonymous blog comparison", "blog", 2026, 0, 1, 2, 1),
]

rows = sorted(((s.name, s.kind) + score(s) for s in SOURCES), key=lambda r: r[2], reverse=True)
print(f"claim: efficacy   review: {REVIEW}   n={len(rows)}")
print("-" * 88)
for name, kind, pct, raw, veto, verdict in rows:
    print(f"{name[:44]:44s} {kind:17s} {pct:5.1f}  {verdict}")
print("-" * 88)
for name, kind, pct, raw, veto, verdict in rows:
    extra = f"  VETO: {veto}" if veto else ""
    print(f"{name[:36]:36s} A{raw['authority']} C{raw['currency']} "
          f"P{raw['applicability']} X{raw['accessibility']} Pr{raw['provenance']}{extra}")
usable = [r for r in rows if r[5].startswith(("primary", "cite"))]
print(f"\nbest citable: {usable[0][0]} ({usable[0][2]}/100)")
```

Expected output on the stated review date:

```text
claim: efficacy   review: 2026-10-26   n=6
----------------------------------------------------------------------------------------
Living systematic review of sepsis alerts    systematic_review  93.8  primary support
Wong 2021, JAMA Intern Med                   peer_reviewed_study  87.5  primary support
Vendor-sponsored conference abstract         conference_paper   57.9  conflicted: background only
Vendor product page (sepsis alert)           press_release      50.0  conflicted: background only
Hospital press release on deployment         press_release      46.7  promotional: background only
Anonymous blog comparison                    blog               32.9  excluded
----------------------------------------------------------------------------------------
Living systematic review of sepsis a A4 C3 P3 X3 Pr4
Wong 2021, JAMA Intern Med           A3 C2 P4 X3 Pr4
Vendor-sponsored conference abstract A1 C3 P3 X2 Pr2
Vendor product page (sepsis alert)   A1 C3 P2 X3 Pr1
Hospital press release on deployment A1 C3 P2 X2 Pr1
Anonymous blog comparison            A0 C3 P1 X2 Pr1  VETO: no accountable author

best citable: Living systematic review of sepsis alerts (93.8/100)
```

Four things about the output are worth reading carefully. First, the two sources that pass are not the two most recent; currency is only 15 percent of the weight. Second, the vendor conference abstract scores 57.9 and is still capped, because the interest rule runs after the arithmetic. Third, the anonymous blog scores 32.9 and is excluded by veto, not by score. Fourth, the systematic review outranks the validation study on the composite, which is the correct ranking for a claim about the class of alerts and the wrong ranking for a claim about this product. The rubric orders sources; the claim decides relevance.

To adapt the script, change four places: the `CEILING` table for genre, the `HALF_LIFE` table for volatility, `W` for the weights, and the `interest` field for conflicts. Keep the veto block above the arithmetic. A weighted mean without a veto is the single most common mistake in published source-scoring tools, because it converts a disqualifying fact into a small deduction.

Three extensions are worth building once the basic script works:

- Add a `locator` field and a `verified_on` date, then flag any row whose verification is older than its class interval.
- Add a `supersedes` field so that replacing a source propagates to every claim that referenced the old row.
- Emit the table as comma-separated values so that the ledger can live in a spreadsheet without a second data entry step.

The script is a decision aid, not an oracle. It cannot see that a paper's cohort overlaps another paper's, that a guidance was withdrawn, or that a vendor changed its model version. Those require the checks in section 5.

## 8. Data, standards, and regulatory context

Source evaluation is governed by a mixture of technical standards, editorial conventions, and, in regulated settings, legal requirements. The distinction matters because a convention can be departed from with a reason, while a binding requirement cannot.

### 8.1 The editorial and metadata layer

| Instrument | Issuer | Status | What it constrains |
|---|---|---|---|
| Recommendations for the Conduct, Reporting, Editing, and Publication of Scholarly Work in Medical Journals | International Committee of Medical Journal Editors (ICMJE) | Voluntary, adopted by member journals | Authorship, conflicts of interest, trial registration, data sharing statements |
| Retraction Guidelines, version 3 | Committee on Publication Ethics (COPE) | Voluntary, adopted by member journals | When and how a retraction, expression of concern, or correction is issued |
| Reporting guideline library | EQUATOR Network | Voluntary | Which checklist applies to which study design |
| DOI Handbook | International DOI Foundation | Technical standard | Syntax, resolution, and persistence of DOIs |
| ISO 3297 | International Organization for Standardization (ISO) | Technical standard | ISSN structure and assignment |
| ISO 2108 | ISO | Technical standard | ISBN structure and assignment |
| ORCID | ORCID Inc. | Community infrastructure | Persistent researcher identifiers |
| FAIR Guiding Principles | Community, published in *Scientific Data* | Voluntary | Findability, accessibility, interoperability, reusability of data |

None of these is law in the ordinary sense. A journal that ignores COPE guidance does not commit an offence; it damages its standing and, eventually, its indexation. The practical consequence is that compliance with the editorial layer is a signal, not a guarantee. A paper can follow every ICMJE recommendation and still be wrong, and a journal can be indexed in PubMed and still publish material that is later retracted.

The metadata layer is more mechanical and more reliable. A DOI that resolves through a registration agency identifies an object and its landing page. Crossref and DataCite are the two registration agencies most relevant here: Crossref for publications, DataCite for datasets and other research objects. A DOI is not a quality mark, but a broken DOI is a reliable sign that the citation was constructed rather than copied.

### 8.2 What is binding in a regulated context

In a regulatory submission, citation practice becomes part of the record. Two requirements recur across jurisdictions:

- **Traceability.** Every factual assertion in a submission must be traceable to a document that a reviewer can obtain. A citation to a personal communication or an internal report that is not in the submission is a gap, not a source.
- **Version specificity.** A reference to a standard must name the edition and any amendments, because the applicable edition is usually fixed at the time of submission or at a transition deadline. "ISO 14971" is not a citation; "ISO 14971:2019" is.

The European Union's Medical Device Regulation and In Vitro Diagnostic Regulation require manufacturers to maintain technical documentation that demonstrates conformity, and the documentation must reference the standards applied. The United States FDA accepts consensus standards through its standards-recognition programme, and a declaration of conformity must cite the recognised standard and its edition. In both systems, the citation is auditable and a wrong edition is a finding.

Electronic records add a further layer. Where records are kept electronically and used to demonstrate compliance, the record system itself is subject to requirements on audit trails, access control, and retention. The citation in the document is only as durable as the system that holds the source.

> **Note:** The regulatory statements in this section are educational. They describe the shape of the requirement, not the requirement for a specific product or jurisdiction. Confirm the applicable text with the regulator or qualified counsel before relying on it.

### 8.3 Citation styles and why they matter less than the record

Vancouver, American Medical Association (AMA), and American Psychological Association (APA) styles differ in punctuation and ordering. They do not differ in what makes a citation checkable: authors, title, venue, year, volume, pages, and a persistent identifier. A document that follows a style guide perfectly and omits the version of a standard is worse than a document that is inconsistently punctuated and names the edition.

The practical rule is to store structured metadata rather than formatted strings. A reference manager record with DOI, PMID, version, and access date can be rendered into any style. A hand-typed citation cannot be re-rendered, audited, or batch-checked when a retraction notice appears.

### 8.4 Checking the container, not only the article

A citation has two objects: the article and the journal or server that published it. Both need a check, and the second is often skipped.

| Check | Where | What a pass means | What it does not mean |
|---|---|---|---|
| Journal is indexed in MEDLINE | National Library of Medicine (NLM) Catalog | The journal met NLM's selection and technical criteria at the time of indexing | The specific article is correct |
| Journal is in DOAJ | DOAJ journal record | The journal meets DOAJ's open-access and integrity criteria | The article was competently peer reviewed |
| Article has a DOI that resolves | DOI resolver, Crossref or DataCite metadata | The object exists and the metadata match | The content is valid or the version is current |
| Article is in PubMed Central | PMC record | A full text is archived and retrievable | The article is open licence for reuse |
| Preprint server is moderated | Server's moderation policy | A basic screening occurred before posting | Peer review occurred |

The container check is not about prestige. A journal can be indexed, legitimate, and publishing work that is later retracted. The check is about whether the container has a process at all, because a container with no process produces citations that cannot be verified by any later reader.

### 8.5 Data citation

Datasets are cited less consistently than articles, and the inconsistency is costly because a dataset citation has to identify a version. The elements are: creator, title, repository, publication year, version, and persistent identifier. DataCite registers DOIs for datasets, and the FAIR principles provide the vocabulary for whether the dataset can be found and reused. A citation that names a dataset without a version cannot be reproduced, because the underlying data may have been replaced.

### 8.6 The hierarchy of documents in a regulated market

A single requirement can appear in four documents with four different statuses. Confusing them is the most common regulatory citation error.

| Layer | United States example | European Union example | Status |
|---|---|---|---|
| Statute | Federal Food, Drug, and Cosmetic Act | Medical Device Regulation 2017/745 | Binding law |
| Implementing regulation | 21 CFR Part 820 | Commission implementing acts | Binding law |
| Agency guidance | FDA guidance documents | MDCG guidance | Generally non-binding interpretation |
| Consensus standard | Recognised standard under the FDA standards programme | Harmonised standard cited in the Official Journal | Binding only through the legal instrument that cites it |

The practical consequence is that a sentence citing a guidance cannot establish an obligation, and a sentence citing a standard cannot establish that the standard applies unless the legal instrument incorporates it. Both errors are common in vendor compliance material, where the layers are compressed into a single "required by FDA" or "required by EU MDR" claim.

### 8.7 What the standards do not cover

- They do not guarantee that a specific article was reviewed by a competent statistician.
- They do not require that a preprint be labelled as unreviewed at the point of citation.
- They do not prevent an author from omitting a conflict that falls outside the disclosure form.
- They do not make a paywalled standard quotable; copyright still applies.
- They do not keep a citation valid after a retraction; that is the citing author's job.
- They do not resolve disagreements between a guideline and the primary evidence it cites.

The gap between the standards and the practice is where source evaluation lives. The standards define the record; the evaluator decides whether the record supports the sentence.

## 9. Industry practice and case evidence

Organisations differ less in the sources they use than in the records they keep. The documented cases below show what happens when the record is missing or the genre is upgraded.

**The Wakefield case (documented).** A 1998 paper in *The Lancet* reported an association between measles, mumps, and rubella vaccination and autism. The paper was partially retracted in 2004 and fully retracted in 2010, after the UK General Medical Council found serious professional misconduct, and *The Lancet* stated that several elements of the paper were incorrect. The episode is the standard example of a source that passed peer review, was cited for a decade, and was ultimately withdrawn. The lesson for source evaluation is not that peer review failed in general; it is that a single small case series cannot carry a population-level causal claim, and that the retraction status has to be checked at the point of reuse.

**The Surgisphere case (documented).** In 2020, two high-profile papers, one in *The Lancet* and one in the *New England Journal of Medicine*, were retracted within days of each other after questions about the underlying database and its provenance could not be answered. The retractions followed open letters from clinicians and statisticians who noticed implausible numbers. The lesson is that provenance is a separate test from authority: the venue was excellent and the data chain was not.

**The Epic Sepsis Model (documented).** An external validation published in *JAMA Internal Medicine* in 2021 reported an AUC of 0.63 for a widely deployed proprietary sepsis model, against a vendor-stated range of 0.76 to 0.83, and reported that two thirds of sepsis cases were not flagged at the deployed threshold. The case is documented in the paper and the accompanying editorial. The inference, which is ours and not the authors', is that procurement documents in many health systems had inherited the vendor figure without an independent check.

**Paper mills and indexed journals (documented pattern, variable specifics).** Investigations by publishers and by the Committee on Publication Ethics have documented industrial-scale fabrication and manipulated peer review, concentrated in some subject areas and some publishers. The Directory of Open Access Journals (DOAJ) tightened its criteria in response, including requirements on peer review and on editorial board composition. The inference for a reader is that indexation is a filter with a non-zero error rate, and that a journal's presence in a database is not evidence that a specific article was reviewed competently.

**Regulatory churn (documented).** The FDA's Quality Management System Regulation became effective on 2 February 2026 and incorporates ISO 13485:2016 by reference, replacing the earlier Quality System Regulation. Any document written before that date that describes the US quality requirement in the old terms is now wrong, regardless of how well it was sourced when written. The case is a reminder that currency failures are often systemic rather than individual.

**The AI Act timeline (documented, and still moving).** The European Union's Artificial Intelligence Act entered into force on 1 August 2024. Secondary summaries written in 2025 commonly state that high-risk obligations for regulated products, including medical devices, apply from August 2027. The European Commission's AI Act Service Desk, which reflects the Digital Omnibus amendments, now lists 2 August 2028 for high-risk AI embedded in products covered by Annex I. A document that quotes the earlier date is not lying; it is quoting a superseded timeline, which is exactly the failure mode this topic describes. The correct practice is to cite the Commission timeline with an access date and to state that the date has been amended.

**What practitioners actually do (inference, from observed practice).** Three patterns are common in organisations that get this right. They keep a source register with owners and review dates. They separate "sourced" from "asserted" in review, so a sentence without a citation is visible rather than invisible. And they re-verify regulatory claims on a fixed calendar rather than on an as-needed basis. The patterns are inferred from practice, not measured in a controlled study.

**What practitioners often do instead (inference).** They treat a bibliography as a formatting task completed at the end of drafting. That sequencing guarantees that the sources are selected to justify sentences already written, which is the reverse of the procedure in section 5.

### Separating what is documented from what is inferred

The distinction matters because a case can be cited to support a lesson it does not establish. The table keeps the two apart.

| Case | Documented | Inference used in this topic | What would overturn the inference |
|---|---|---|---|
| Wakefield | Retraction in 2010 after a misconduct finding; *The Lancet* stated elements were incorrect | A small case series cannot carry a population-level causal claim | Evidence that the original case series was larger and systematically ascertained |
| Surgisphere | Two papers retracted in 2020 after data provenance could not be verified | Provenance is a separate test from venue prestige | Documentation showing the database and analysis were as described |
| Epic Sepsis Model | External AUC 0.63 against a vendor range of 0.76 to 0.83 in one system | Inherited vendor figures are rarely checked | A multi-site validation showing the vendor range |
| Paper mills | Documented fabrication and manipulated review in some publishers | Indexation is a filter with a non-zero error rate | Evidence that indexation predicts article-level integrity |
| QMSR | Effective 2 February 2026; incorporates ISO 13485:2016 | Documents written before the change are now outdated | A transition provision extending the old requirement |
| AI Act timeline | Commission lists 2 August 2028 for Annex I high-risk products | Secondary summaries can lag amended timelines | A further amendment restoring the earlier date |

### Practice patterns that reduce source risk

- A source register with an owner, a version, an access date, and a review trigger for every citation that carries weight.
- A drafting rule that no number enters a document without a source and a date in the same sentence or footnote.
- A review step that reads the cited passage, not the citation, for the ten most consequential claims.
- A regulatory calendar that is maintained centrally rather than by each project.
- A habit of citing the primary study and the guideline separately, rather than the guideline alone.
- A rule that vendor material is labelled as vendor material in the document, not only in the ledger.
- A periodic sweep for retractions across the active bibliography.

None of these requires software. The register can be a spreadsheet, and the sweep can be a quarterly task. What they require is that someone owns the bibliography as an artefact rather than as a formatting output.

### The limit of case evidence

Case evidence shows that a failure mode is possible and what it costs. It does not show how often the failure occurs, and the cases above are selected because they were visible. The ordinary failures, a superseded edition in a slide deck or a denominator omitted from a market chart, are invisible and probably more consequential in aggregate. The procedure in section 5 is designed for the ordinary case, not only for the spectacular one.

Procurement documents deserve a specific note, because they are where vendor claims become institutional facts. Four questions reduce most of the risk:

- Which model version does the performance figure refer to, and is that the version being purchased.
- Was the validation independent of the vendor, and in what setting.
- What is the performance at the threshold the hospital will deploy, not at the best threshold in the paper.
- What is the alert volume at that threshold, and who reviews the alerts.

The fourth question is the one most often omitted, and it is the one that determines whether the alert is used or ignored.

## 10. Failure modes and red flags

The failures below are ordered roughly by frequency in industry documents. Each has an early warning that appears before the document is challenged, which is the point of listing them: the warning is cheap and the consequence is not.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Genre upgrade | "Validated" appears without a study name | A summary is cited as if it were the primary source | Claim collapses under review | Trace to the primary source and cite it |
| Draft read as final | Citation omits a date and a status | Draft guidance circulates as a PDF after comment closes | Compliance plan built on a proposal | Check the agency's current guidance list |
| Superseded edition | Book cited without an edition number | A familiar edition is quoted after revision | Clinical or technical statement is outdated | Add edition and check the publisher's current list |
| Preprint cited as settled | Citation has an arXiv or medRxiv identifier | Speed outruns peer review | Direction of effect may reverse | Label as preprint; check for a published version |
| Retracted source in use | Reference list predates the retraction | Reference manager keeps the original record | Document cites withdrawn work | Status check at each reuse |
| Single-study support | Several citations, all from one cohort | Secondary analyses multiply the PDF count | Apparent replication that is not replication | Count cohorts, not papers |
| Circular citation | Two reviews cite each other | Reviews treat each other as evidence | Claim has no primary source | Trace to the base of the chain |
| Undisclosed interest | Funding statement is vague or absent | Commercial sponsor not named | Apparent independence is false | Check ICMJE forms and filings |
| Paywalled standard quoted | Clause quoted without edition or licence | A secondary summary is used instead of the standard | Quote may be wrong or outdated | Read the clause under licence; cite clause and edition |
| Dead link cited | URL returns an error or a redirect to a homepage | Site restructured after publication | Reader cannot verify | Use DOI or document number; archive a snapshot |
| Predatory venue | Fast review, broad scope, unclear fees | Author pays, review is minimal | Paper is not a reliable source | Check DOAJ, NLM catalog, and integrity lists |
| Denominator missing | A percentage with no base | Numerator reported, base omitted | Market or operational claim is uninterpretable | Require the denominator and the year |
| Version mismatch | Dataset counts do not match the file | Citation is to a paper about an earlier release | Numbers cannot be reproduced | Cite the dataset version and DOI |
| Definition drift | Outcome name is stable, definition changed | Guideline or coding changed | Cohort mixes two measurements | Check definition dates against cohort dates |
| Overgeneralised setting | "Hospitals" without a setting description | Single-site study summarised broadly | Performance claim does not transfer | State site, era, and population in the sentence |
| Clearance read as efficacy | "FDA cleared" used to imply clinical benefit | Regulatory status conflated with comparative evidence | Product claim exceeds the decision basis | Separate the regulatory claim from the performance claim |
| Unit mismatch | A rate per 1,000 compared with a percentage | Units dropped between source and summary | Order-of-magnitude error | Carry units into every table and sentence |
| FAQ cited as requirement | Citation is to an agency FAQ | Informal interpretation treated as binding text | Compliance scope misstated | Cite the instrument; use the FAQ only as orientation |
| Guideline cited after revision | Recommendation quoted without a version | Guideline updated, sentence unchanged | Practice statement contradicts current guidance | Record guideline version and update date |

### Using the table in review

- Read the table against the document's ten most consequential sentences, not against every sentence.
- Ask which row each sentence would fail if a reviewer were hostile.
- Fix the sentence, not only the citation; a better source for an overstated sentence rarely exists.
- Record the failure in the ledger so the same source is not reused the same way.
- Check whether the failure is individual or systemic; a repeated row indicates a process problem.
- Re-run the check after any revision that changes a number or a comparator.

The table is also a teaching device. A new analyst who can name the row for a failure can usually propose the fix, because the mechanism column points at the cause rather than at the symptom.

Three red flags are worth naming separately because they appear in the citation list rather than in the prose.

- A reference list in which every citation is to a review. Reviews are efficient, but a document built only from reviews has no primary evidence base.
- A citation that points to a URL with a query string and no DOI, especially for a journal article. It usually means the citation was copied from a search result.
- A reference to a "personal communication" carrying a number that appears nowhere else. If the number matters, it needs a document.
- A table whose source note says "internal analysis" with no author, date, or method.
- A figure reused across documents with a different value each time, which usually means nobody owns it.

In a review meeting, the questions that surface these failures fastest are: "What is the primary source for that number?", "What version of the document is that?", and "What would have to be true for this sentence to be false?" The third question is the one that separates a sourced claim from an asserted one.

## 11. Assumptions and limits

The procedure rests on assumptions that are usually true and occasionally not. Each is stated with the test that would reveal a violation.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The claim can be stated in one sentence | The five tests need a fixed object | The document asserts a vague aspiration | Try to write the sentence; if you cannot, the claim is not ready |
| A primary source exists | Tracing backwards assumes a base | The claim originates in marketing or folklore | Follow the chain to a study, dataset, or legal text |
| The issuer's current list is authoritative | Currency checks depend on it | The issuer publishes inconsistently or in several places | Cross-check two official locations |
| Peer review filters some error | It justifies a tier, not a guarantee | The venue is predatory or the review was compromised | Check indexation and integrity signals |
| Abstracts reflect the article | Many checks rely on abstracts | Abstract overstates or omits the limitation | Read the results table for the specific number |
| Publication status is stable | A citation is assumed to remain valid | Retraction, correction, or withdrawal | Status check at reuse |
| The reader can obtain the source | Verifiability depends on access | Paywall, login, or dead link | Provide DOI and archive; state the access route |
| Version numbers are meaningful | Dataset and model claims depend on them | Vendor reuses a version label after a change | Record the release date and hash where possible |
| The evaluation is independent of the claim's beneficiary | Authority and provenance tests assume it | Funding, authorship, or institutional interest | Read the funding and conflict statements |
| Reviewers share the claim definition | Scoring assumes a common object | Different teams score different sentences | Put the framed sentence at the top of the ledger |
| The source's population is representative of the claim's | Applicability scoring assumes a comparison is possible | The source is a convenience sample | Compare the sampling frame to the target population |
| Usual care is a stable comparator | Efficacy claims are relative to current practice | Practice changed after the study | Check the study dates against current guidelines |
| The number is stable across versions | A citation is assumed to remain accurate | Reanalysis or correction changes the estimate | Check the version of record and any correction |
| The reviewer can read the full text | Provenance scoring assumes access | Abstract only, paywall, or restricted report | State what was read and what was not |
| The source's date is the date of the evidence | Currency checks assume alignment | Publication lag of one to three years | Check the data collection period, not the publication year |
| The claim's setting resembles the source's | Applicability assumes a comparison | Different country, payer, or care pathway | Compare the pathway, not only the disease |
| Conflicts are disclosed completely | Provenance assumes the disclosure is honest | Undisclosed equity or patents | Check filings and patent records where material |

Two limits are structural rather than conditional. First, the procedure cannot establish that a well-sourced claim is true; it can only establish that the source is adequate for the claim as stated. Second, the procedure is only as good as the search that produced the candidate set. A missing primary source cannot be evaluated, and the most common failure in practice is not a bad source but an absent one.

A third limit is social. The procedure assumes that the reviewer is willing to conclude that a favoured source is inadequate. In organisations where a citation has become part of an approved narrative, that conclusion carries a cost, and the ledger is only useful if the cost can be paid. Naming the test result rather than the person helps: "provenance fails because the method is not disclosed" is easier to accept than "this source is unreliable."

The scoring weights in section 7 are a policy choice, not a measurement. They encode a view that authority and applicability matter more than accessibility. A different organisation with different risks can choose different weights, provided it keeps the vetoes and writes down the choice. The value of the rubric is consistency across reviewers, not precision in the decimal.

## 12. Exercises

The exercises move from recall to design. Each states a deliverable, acceptance criteria, and a time estimate for a competent learner working alone. Exercises 5 and 8 produce artefacts that can be reused in a real project.

**Exercise 1 — Frame and classify (recall and application, 25 minutes).**
Take five sentences from a document you have recently written or reviewed. For each, rewrite it with the five slots (population, intervention or exposure, outcome, time, setting), label the claim type, and name the minimum source that could support it.
Deliverable: a five-row table with the original sentence, the framed sentence, the claim type, and the minimum source.
Acceptance criteria: every framed sentence has all five slots or an explicit "not stated"; every claim type is one of the six in section 5; at least one original sentence is shown to be unscoreable as written.
Time: 25 minutes.

**Exercise 2 — Five-test record (application, 40 minutes).**
Choose one factual claim from your current work and apply the five tests to its current source. Record pass, conditional, or fail for each test with one line of evidence.
Deliverable: a one-page record with the claim, the source, five test results, and a verdict from the four in section 5.
Acceptance criteria: each test result cites a specific passage, date, or metadata field; the verdict is one of the four allowed values; a limitation is written into the sentence rather than a footnote.
Time: 40 minutes.

**Exercise 3 — Primary versus secondary (application, 30 minutes).**
Take a guideline recommendation or a review sentence and trace it backwards to the primary study. Record each link in the chain and note whether the claim changed in transit.
Deliverable: a citation chain of three to six links with a note at each step describing what was added, dropped, or strengthened.
Acceptance criteria: the chain ends at a primary study or at a documented dead end; at least one change between links is described; a circular citation, if present, is identified.
Time: 30 minutes.

**Exercise 4 — Currency audit (application, 35 minutes).**
Select four sources of different types (a regulation or guidance, a textbook, a peer-reviewed paper, and a web page) and determine the current status of each using the issuer's own listing.
Deliverable: a four-row status table with version, status, effective date, and the URL of the authoritative listing used.
Acceptance criteria: each status is supported by an issuer page, not a third-party summary; at least one source is found to be superseded, withdrawn, or ambiguous; the table records the access date.
Time: 35 minutes.

**Exercise 5 — Build and run a scoring rubric (design and calculation, 60 minutes).**
Adapt the script in section 7 to your own domain by changing the ceiling table, the half-life table, and the weights. Score six candidate sources for one claim and produce the ranked output.
Deliverable: the modified script, its output, and a one-paragraph justification of the weights and vetoes.
Acceptance criteria: the script runs with the standard library only; the output is reproducible; the justification explains at least one weight and one veto; a candidate that scores well but is capped is identified.
Time: 60 minutes.

**Exercise 6 — Write a ledger entry (application, 25 minutes).**
Write a ledger row for the claim you used in Exercise 2, using the fields in section 5 step 17, including what would change the verdict.
Deliverable: one completed ledger row.
Acceptance criteria: all ten fields are present; the "would change if" field names a specific study, version, or event rather than "new evidence"; a second reviewer could reproduce the verdict from the row alone.
Time: 25 minutes.

**Exercise 7 — Critique a citation chain (critique, 45 minutes).**
Take a document you did not write, preferably a vendor white paper or a strategy deck, and identify the three weakest citation links. For each, name the failing test and the sentence that is not supportable.
Deliverable: a marked-up excerpt with three findings, each naming the test, the evidence, and a replacement sentence.
Acceptance criteria: findings are specific to a sentence and a source; at least one finding concerns provenance rather than authority; the replacement sentence is weaker than the original and true.
Time: 45 minutes.

**Exercise 8 — Design a maintenance schedule (design, 40 minutes).**
Build a re-verification schedule for a source set of at least fifteen sources drawn from a real project. Assign each source a class, an interval, and a trigger.
Deliverable: a spreadsheet or table with source, class, owner, interval, trigger, and last-verified date.
Acceptance criteria: intervals are justified by volatility rather than by convenience; every regulatory and dataset source has an event trigger; the schedule fits within the team's actual capacity, measured in hours per month.
Time: 40 minutes.

## 13. Self-check answers

**Self-check 1. What is the minimum source for a regulatory obligation, and why is a law firm summary insufficient?**
The minimum source is the legal instrument itself, or the agency's own statement of the requirement with its effective date. A law firm summary is a secondary interpretation: it can be accurate, but it can also describe a draft, omit a transition provision, or generalise from one jurisdiction. It is useful for orientation and citation chaining, not for the obligation.

**Self-check 2. A paper reports a hazard ratio of 1.4 with a 95 percent confidence interval of 0.9 to 2.1. Is the claim "the intervention increases risk" supported?**
No, not at the stated strength. The point estimate is above 1, but the interval includes 1, so the data are compatible with no effect and with a substantial increase. The supportable sentence is "the study estimated a hazard ratio of 1.4, with an interval that includes no effect." Whether that is clinically important depends on the baseline risk and the harm, not on the point estimate alone.

**Self-check 3. Why does a systematic review not count as independent confirmation of the trials it includes?**
Because it is a function of those trials. If the trials share a bias, the review inherits it, and pooling increases precision without increasing independence. A review can confirm that the literature is consistent; it cannot supply a second data source.

**Self-check 4. Name the four currency states in section 5 and give an example of each.**
Current: a final guidance listed on the agency's current page. Current with a transition period: a regulation whose older requirements still apply to devices placed on the market before a date. Superseded but applicable to legacy products: an older standard edition retained for products already certified. Superseded or withdrawn: a guidance replaced by a new version, or a standard edition withdrawn from the catalogue.

**Self-check 5. A dataset paper reports 364,627 patients and the downloaded file has 364,625. What is the likely explanation and what should the citation say?**
The likely explanation is a version difference or a minor data correction between releases. The citation should name the dataset, the version, and the DOI, and the analysis should report the count in the file actually used. The discrepancy is small but it is a signal that the version was not checked, and the same check would catch larger differences.

**Self-check 6. Why is a weighted score without a veto unsafe?**
Because weights allow compensation. A source with no disclosed method can score well on applicability and accessibility and still be unusable as evidence. The veto converts a disqualifying fact into an exclusion rather than a deduction, which is the correct arithmetic for a conjunctive decision.

**Self-check 7. What is the difference between a preprint and a conference abstract as evidence?**
A preprint is a full manuscript with a version history, usually complete enough to appraise, but unreviewed. A conference abstract is a short summary, often without methods or full results, and its review standard varies. Neither is equivalent to a peer-reviewed full paper, and the abstract is generally weaker because there is less to appraise.

**Self-check 8. A vendor says its model has "FDA clearance." What does that establish?**
It establishes that a regulatory submission for a specified intended use met the standard for the clearance pathway at the time of the decision. It does not establish comparative performance, clinical benefit, performance in a different population, or performance of a later model version. The clearance should be checked in the agency's database, including the product code, the intended use, and the date.

**Self-check 9. Why is a DOI check insufficient on its own?**
A DOI check confirms that the identifier resolves and that the metadata match the citation. It does not confirm the publication status, the version, the retraction record, or whether the article supports the sentence. All four require separate checks.

**Self-check 10. A claim rests on a trial, a secondary analysis of the same trial, and a meta-analysis including the trial. How many independent sources support the claim?**
One. The secondary analysis and the meta-analysis are functions of the same data. The evidence is single-trial support, with whatever limitations that trial has, regardless of the citation count.

**Self-check 11. What are the three levels of a retraction status check?**
The publisher's landing page, which carries the notice; the bibliographic index, which may tag the record; and an integrity database, which may show a notice the publisher has not linked. The publisher's most recent notice governs, and any disagreement among the three is recorded.

**Self-check 12. Why is a guidance document's use of "must" not proof of a legal obligation?**
Agency guidance often uses mandatory language to describe how the agency will interpret or apply a statutory requirement. The obligation comes from the statute or regulation, not from the guidance's grammar. The instrument, not the modal verb, determines the status.

**Self-check 13. What is wrong with citing the publication year as the date of the evidence?**
Publication lags data collection, sometimes by years, and a 2026 paper may report 2018 data. The relevant date for currency and era effects is the data collection period, which is in the methods. A citation that gives only the publication year can make an old dataset look current.

**Self-check 14. When is it legitimate to cite a press release?**
When the claim is about the announcement itself, such as what an organisation said and when. It is not legitimate support for a clinical, performance, or market claim, because it does not disclose a method. If the release contains a number that appears nowhere else, treat the number as unsupported.

**Rubric for Exercises 5 and 8.** A strong submission for Exercise 5 has a working script, a reproducible output, a justification that ties weights to decision risk, and an explicit example of a capped source. A weak submission changes the weights without justification, removes the veto, or reports a score without the component detail. A strong submission for Exercise 8 has intervals that differ by source class, event triggers for regulatory and dataset sources, and a capacity estimate that the team could actually meet. A weak submission assigns every source a twelve-month interval and no triggers.

### Marking guide

| Exercise | Evidence of a pass | Common failure |
|---|---|---|
| 1 Frame and classify | All five slots present or explicitly absent; type from the fixed list | Rewriting the sentence without adding population or time |
| 2 Five-test record | Each result cites a passage, date, or metadata field | Scores with no evidence line |
| 3 Primary versus secondary | Chain reaches a primary study or a documented dead end | Stopping at the review and calling it primary |
| 4 Currency audit | Status confirmed at the issuer | Confirming at a mirror or a search snippet |
| 5 Scoring rubric | Runs, reproduces, and explains a cap | Removing the veto to make the numbers work |
| 6 Ledger entry | Reproducible by a second reviewer | "Would change if: new evidence" |
| 7 Citation critique | Names the failing test and a replacement sentence | General criticism with no sentence-level finding |
| 8 Maintenance schedule | Intervals differ by class; triggers present | Uniform annual review |

## 14. Interview questions

The questions below are the ones that separate a candidate who has read about evidence appraisal from one who has had to defend a citation in a review. Each entry states what a strong answer contains and what a weak answer reveals.

**1. Tell me about a time you found that a source you were relying on was wrong or out of date.**
Strong answer: names the document, the specific failure (draft versus final, superseded edition, retracted paper, version mismatch), how it was caught, what changed in the deliverable, and what process change followed. Weak answer: describes a general belief that sources should be checked, with no specific episode or consequence.

**2. How do you decide whether a paper is good enough to cite?**
Strong answer: frames the claim first, then applies authority, currency, applicability, accessibility, and provenance, and says that a strong paper can be inadequate for a particular sentence. Weak answer: lists journal impact factor, peer review, or sample size as a single criterion.

**3. What is the difference between a regulation and a guidance document?**
Strong answer: a regulation is legally binding; guidance describes the agency's interpretation and is generally non-binding, though it can be persuasive and can be referenced in enforcement. Names an example and the effective date. Weak answer: treats them as synonyms or says guidance is "basically law."

**4. How would you check whether a clinical claim is still current?**
Strong answer: identifies the claim type, finds the governing guideline or definition, checks the version and date, and compares the cohort or deployment period to the definition change. Weak answer: searches for the most recent paper and assumes recency equals currency.

**5. A vendor gives you a validation AUC. What do you ask for?**
Strong answer: the cohort, the setting, the outcome definition, the model version, whether the validation was independent, the confidence interval, and performance at the deployed threshold. Weak answer: asks only for the sample size or accepts the number because it is in a paper.

**6. How do you handle a preprint that supports your argument?**
Strong answer: labels it as a preprint, checks for a published version, compares the two, and does not let it carry a claim that needs peer-reviewed support. Weak answer: cites it as evidence without qualification, or refuses to read it at all.

**7. What do you do when two credible sources disagree?**
Strong answer: compares populations, designs, outcome definitions, and versions; looks for a third source that explains the difference; and reports the disagreement rather than picking the convenient result. Weak answer: chooses the newer one, or the one from the more prestigious venue, without examining why they differ.

**8. How do you know a citation will still work in two years?**
Strong answer: uses a DOI or official document number, archives a dated snapshot, records the version, and sets a review trigger. Weak answer: says the URL is stable because it is a government site.

**9. What is your process for keeping a bibliography current?**
Strong answer: describes a register with owners and cadences, alerts for key sources, status checks at reuse, and a regulatory calendar. Weak answer: says they re-run the search when a document is due.

**10. How would you evaluate a market-size claim in a vendor white paper?**
Strong answer: asks for the definition of the market, the base year, the denominator, the method, the sample, and the sponsor's interest; prefers filings or government statistics; and reports the number as an estimate with a range. Weak answer: quotes the compound annual growth rate because it sounds precise.

**11. When is it acceptable to cite a secondary source?**
Strong answer: when the secondary source is the object of the claim (for example, a guideline recommendation as evidence of what the guideline says), or as a pointer when the primary is inaccessible, with the limitation stated. Weak answer: whenever the primary is inconvenient.

**12. What would make you remove a citation from a finished document?**
Strong answer: a retraction, a correction that changes the number, a superseded version, a discovered conflict that undermines independence, or a finding that the source does not support the sentence. Weak answer: only a broken link.

**13. How do you cite a standard you cannot afford to buy?**
Strong answer: cite the standard number, edition, and clause; state that the clause was read through an institutional licence or a colleague's copy; and avoid quoting text that copyright does not permit. If the clause could not be read, say so rather than relying on a secondary summary. Weak answer: quotes a blog summary as if it were the standard.

**14. What is the difference between a source being wrong and a source being inadequate?**
Strong answer: wrong means the source's own claim is false or retracted; inadequate means the source is sound but cannot carry the sentence being written, because of population, setting, design, or version. Inadequate is more common and is fixed by changing the sentence. Weak answer: treats every inadequate source as wrong and discards it, losing useful context.

**15. How would you audit a bibliography you inherited?**
Strong answer: samples the most consequential claims, checks status and version at the issuer, traces the top ten citations to primary sources, and looks for single-study dependence and circular chains. Records findings in a ledger and fixes the sentences, not only the references. Weak answer: runs a link checker and stops there.

**16. What is the first question you ask about a dataset citation?**
Strong answer: which version, with which DOI, and which release date, because the counts and variables change between versions. Weak answer: asks only whether the dataset is public.

## 15. Key terms

- **Authority** — accountability and independence of the issuing body for the specific claim.
- **Applicability** — match between the source's population, setting, task, and era and the claim's.
- **Accessibility** — whether a reviewer can obtain the source and the passage relied on.
- **Provenance** — visibility and independence of the chain from data to conclusion.
- **Currency** — whether the source is the version in force for the claim's purpose.
- **Primary source** — a study, dataset, or legal instrument that generates the evidence.
- **Secondary source** — a synthesis or interpretation of primary sources.
- **Tertiary source** — a summary that points to evidence, such as a textbook or point-of-care topic.
- **Preprint** — a manuscript posted before peer review, with a version history.
- **Version of record** — the publisher's final citable version.
- **Retraction** — withdrawal of a paper's validity by the publisher or authors.
- **Expression of concern** — a notice raising a question without retracting.
- **Corrigendum** — a correction of a production error.
- **DOI** — persistent identifier resolvable through a registration agency.
- **PMID** — identifier for a PubMed record.
- **ORCID** — persistent identifier for a researcher.
- **Draft guidance** — a proposed agency position open for comment.
- **Final guidance** — an agency position issued after comment, usually non-binding.
- **Effective date** — the date a legal obligation begins to apply.
- **Transition period** — a window in which an older requirement may still be met.
- **Superseded** — replaced by a newer version.
- **Veto** — a test result that excludes a source regardless of other scores.
- **Interest cap** — a rule limiting the role of a source with a commercial or institutional interest.
- **Source ledger** — the record of claim, source, tests, verdict, and review date.
- **Half-life** — the period over which a source type remains adequate.
- **Claim router** — the step that assigns a claim type and an authority floor.
- **Predatory journal** — a venue that charges fees without genuine editorial or peer-review process.
- **Paper mill** — an operation that produces fabricated or manipulated manuscripts for authorship credit.
- **Tortured phrase** — a rewritten sentence that evades plagiarism detection, often a paper-mill signal.
- **Snowballing** — finding sources through the reference lists and citations of a known source.
- **Citation chain** — the sequence from a summary back to the primary evidence.
- **Circular citation** — a loop in which sources cite each other without an independent base.
- **Predatory conference** — an event that charges registration fees without genuine peer review.
- **Indexation** — inclusion of a journal in a bibliographic database such as MEDLINE.
- **Metadata** — structured descriptive fields for a source, such as title, venue, and year.
- **Persistent identifier** — an identifier designed to remain stable as locations change.
- **Landing page** — the publisher page to which a DOI resolves.
- **Evidence gap** — a claim for which no adequate source exists at the required tier.
- **Single-source claim** — a claim supported by one independent dataset or study.
- **Register** — the maintained list of sources with owners, versions, and review dates.
- **Snapshot** — a dated archived copy of a web page used as evidence of what it said.

## 16. Sources

Each entry states what the source is good for and what it is not. All URLs were checked on 26 October 2026.

**International Committee of Medical Journal Editors (ICMJE).** *Recommendations for the Conduct, Reporting, Editing, and Publication of Scholarly Work in Medical Journals.* https://www.icmje.org/recommendations/
Good for: authorship criteria, conflict-of-interest disclosure, trial registration, data-sharing statements. Not for: study design or statistical appraisal. The recommendations are adopted by member journals and are not law.

**Committee on Publication Ethics (COPE).** *Retraction Guidelines*, version 3, September 2025. https://publicationethics.org/guidance/guideline/retraction-guidelines
Good for: what a retraction notice must contain, when an expression of concern is appropriate, and how corrections differ from retractions. Not for: judging whether a specific finding is true. Version 3 supersedes the 2019 version 2.

**EQUATOR Network.** Reporting guideline library. https://www.equator-network.org/
Good for: identifying the correct checklist for a study design (TRIPOD+AI, CONSORT, STARD, PRISMA). Not for: appraising the quality of a completed study; a checklist is a reporting aid, not a risk-of-bias tool.

**Oxford Centre for Evidence-Based Medicine (CEBM).** *The Oxford 2011 Levels of Evidence* and background document. https://www.cebm.ox.ac.uk/resources/levels-of-evidence/ocebm-levels-of-evidence
Good for: a question-type-specific hierarchy and a fast search heuristic. Not for: grading recommendations; the authors state it is not a recommendation system and requires judgement.

**Crossref.** DOI registration and metadata API, including Retraction Watch retraction data. https://www.crossref.org/
Good for: verifying that a DOI exists, retrieving title and venue metadata, and checking retraction links via the `update-to` field. Not for: content-level verification.

**Retraction Watch.** Retraction Watch Database. https://retractionwatch.com/
Good for: locating retractions and expressions of concern, including notices not linked by the publisher. Not for: a complete census; coverage depends on what has been reported.

**Directory of Open Access Journals (DOAJ).** Journal index and criteria. https://doaj.org/
Good for: checking whether an open-access journal meets basic editorial and licensing criteria. Not for: article-level quality; indexation is a floor, not a certification.

**National Library of Medicine.** NLM Catalog. https://www.ncbi.nlm.nih.gov/nlmcatalog/
Good for: checking journal indexation, title changes, and ISSNs. Not for: evaluating a specific article.

**Wilkinson, M. D. et al.** "The FAIR Guiding Principles for scientific data management and stewardship." *Scientific Data* 3:160018, 2016. https://doi.org/10.1038/sdata.2016.18
Good for: the vocabulary of findability, accessibility, interoperability, and reuse, and for data-citation practice. Not for: a compliance checklist; the principles are aspirational and require local interpretation.

**International Organization for Standardization (ISO).** ISO 3297 (ISSN) and ISO 2108 (ISBN). https://www.iso.org/
Good for: identifier structure and assignment rules. Not for: free access; the standards are paywalled, so cite the edition and clause rather than quoting at length.

**Food and Drug Administration (FDA).** *Search for Guidance Documents.* https://www.fda.gov/regulatory-information/search-fda-guidance-documents
Good for: confirming whether a guidance is draft or final and its issue date. Not for: legal advice; guidance is generally non-binding.

**European Commission.** *AI Act Service Desk: timeline for the implementation of the EU AI Act.* https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
Good for: the current application dates, including the amended date of 2 August 2028 for high-risk AI embedded in Annex I products. Not for: sector-specific device obligations, which are covered by MDR and IVDR guidance.

**Think. Check. Submit.** Journal quality checklist. https://thinkchecksubmit.org/
Good for: a short, practical set of questions about a journal's identity, editorial board, and peer-review claims. Not for: a formal assessment; it is a screening aid for authors, and the same questions apply when reading.

**ORCID.** Researcher identifier registry. https://orcid.org/
Good for: disambiguating authors with common names and checking publication lists. Not for: verifying that a claimed affiliation or credential is current.

**DataCite.** DOI registration for research data. https://datacite.org/
Good for: resolving dataset DOIs and retrieving version metadata. Not for: assessing dataset quality or licence terms, which live with the repository.

**National Institutes of Health (NIH), U.S. National Library of Medicine.** PubMed and PubMed Central. https://pubmed.ncbi.nlm.nih.gov/
Good for: locating indexed literature and free full texts. Not for: a complete search; coverage varies by field and by journal, and preprints are indexed separately.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_supplied-clinical-library.md`](02_supplied-clinical-library.md)
- Regulation and policy: [`03_regulation-and-policy.md`](03_regulation-and-policy.md)
- Methods: [`04_methods.md`](04_methods.md)
- Keeping current: [`09_keeping-current.md`](09_keeping-current.md)
- Survey file: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md)
- Labs: [`../../labs/`](../../labs/)
- Module 21, dataset selection: [`../21_datasets/01_five-gate-decision.md`](../21_datasets/01_five-gate-decision.md)
- Module 18, quality and appraisal: [`../18_quality/`](../18_quality/)
- Module 14, device regulation: [`../14_devices_regulation/`](../14_devices_regulation/)

The next topic applies this procedure to the books supplied with the course, where edition and printing dates are the main source of error.
