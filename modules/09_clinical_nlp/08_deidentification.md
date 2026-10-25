# De-identification

> **Module 09 — Clinical NLP**
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 06, and 07
> Version 2026.3

## 1. Why this topic exists

Clinical text is the highest-risk data in a health system. It contains names, dates, addresses,
identifiers, family details, and narratives that can re-identify a person even when the obvious
fields are removed. A de-identification pipeline is the control that allows text to be used for
research, secondary analysis, or model development without disclosing who the patient is.

The control fails in predictable ways. A regex misses a name written without a title. A dictionary
misses a local facility. A date is removed but the note says "admitted on the day of the storm,"
which identifies the admission in a small town. A provider name is replaced, but the note says "seen
by the transplant surgeon who joined last spring," which identifies the surgeon and, through the
schedule, the patient. De-identification is not a find-and-replace problem; it is a risk-reduction
problem with a stated standard and a measured residual.

The regulatory standard is specific. The HIPAA Privacy Rule at 45 CFR 164.514(b) defines two paths
to de-identification. The Safe Harbor path requires removal of 18 categories of identifiers and
requires that the covered entity have no actual knowledge that the remaining information could
identify an individual. The expert determination path requires a person with appropriate knowledge
and experience to apply generally accepted statistical and scientific principles, determine that
the risk of identification is very small, and document the methods and results. The two paths are
not interchangeable, and a pipeline that satisfies neither is not de-identified under HIPAA.

The measurement problem is asymmetric. A false negative, an identifier left in the text, is a
privacy breach. A false positive, a clinical term removed as an identifier, is a utility loss. The
first is a legal and ethical failure; the second is a data quality problem. A pipeline tuned for
balanced F1 will under-weight the false negatives, because identifiers are a small fraction of
tokens. The evaluation must prioritize recall by identifier type and must report the residual
risk in units a privacy officer can use: identifiers per note, share of notes with at least one
residual identifier, and the type distribution of the residuals.

The utility problem is also real. Over-redaction destroys the clinical signal that makes the text
useful. Replacing every number with a placeholder removes lab values as well as dates. Replacing
every capitalized phrase removes medication names as well as people. A de-identification pipeline
that achieves perfect recall by removing everything is not a solution; it is a different way of
failing.

The operational problem is scale and change. A pipeline validated on one year of notes may fail on
the next year's template. A new service line introduces new identifier forms. A new EHR vendor
changes the header format. The pipeline needs monitoring, and the monitoring needs a gold sample
that is re-annotated periodically.

There is also a threat-model problem. De-identification is usually discussed as if the only
adversary is a curious recipient with no other data. In practice, the relevant adversary may have
the patient's date of birth from a registry, the admission date from a claims file, or the
provider's schedule from a public directory. The combination of a retained quasi-identifier with
auxiliary data is often enough to identify. A pipeline that does not state its threat model cannot
claim that its residual risk is small; it can only claim that its recall on a category list is high.

This topic covers the regulatory standard, the identifier taxonomy, the detection methods, the
transformation choices, the evaluation design, and the residual-risk report. It treats
de-identification as a safety-critical process with a measured failure rate, not as a preprocessing
step.

> **Note:** "De-identified" is a legal conclusion about a dataset under a specific standard. A tool
> that removes names does not produce a de-identified dataset unless the standard's conditions are
> met and documented. The distinction matters for data use agreements, IRB determinations, and
> publication.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** the Safe Harbor and expert determination paths and **state** what each requires.
2. **Map** the 18 Safe Harbor identifier categories to detection methods and local forms. The test
   is whether each category has at least one detector and one local variant.
3. **Build** a detection pipeline with regex, dictionary, and context rules, and **evaluate** it per
   identifier type. The test is a per-category table, not an aggregate.
4. **Choose** a transformation: redaction, surrogate replacement, date shifting, or generalization,
   and **explain** the utility and risk consequences. The test is whether the choice is consistent
   with the regulatory path.
5. **Compute** per-type recall, precision, and residual identifier rates with confidence intervals.
   The test is whether the intervals support the release threshold.
6. **Measure** utility loss in a downstream task and **report** the trade-off. The test is whether
   the task still meets its own acceptance criterion on transformed text.
7. **Write** a residual-risk assessment suitable for a privacy officer or an expert determination.
   The test is whether the threat model and the evidence are stated, not only the recall.
8. **Design** a monitoring plan for template and vocabulary drift. The test is whether a trigger and
   an owner are named.

Outcomes 3 through 6 are computational; 1, 2, 7, and 8 are regulatory and design.

## 3. The core mental model

De-identification is a transformation from an identifiable text to a text that meets a standard,
with a measured amount of residual risk.

```text
  identifiable note
        |
        v
  identifier detection  ----> candidate spans + types
        |
        v
  context resolution   ----> patient vs provider vs family; real vs placeholder
        |
        v
  transformation       ----> redact | surrogate | shift | generalize
        |
        v
  transformed note     ----> utility for the downstream task
        |
        v
  residual risk        ----> missed identifiers, inference risk, linkage risk
```

The diagram separates detection from transformation. A team can have good detection and a bad
transformation (for example, replacing a name with a unique code that is linkable across notes), or
bad detection and a transformation that happens to remove the risk (for example, deleting the whole
header). The evaluation must cover both.

A second model is the **identifier taxonomy**, which is broader than the 18 categories suggest.

```text
  direct identifiers        names, MRN, SSN, phone, email, address
  quasi-identifiers         dates, age, ZIP, sex, race, occupation
  contextual identifiers    rare disease, small town, event, family structure
  inferred identifiers      combination of quasi-identifiers
```

Safe Harbor addresses the first two categories and part of the third. It does not address inferred
identifiers directly; the "actual knowledge" clause and the expert determination path are the
mechanisms for those. A dataset can pass every Safe Harbor removal and still be re-identifiable by
combination, which is why the regulation includes the actual-knowledge condition and why expert
determination exists.

A third model is the **risk equation** for a released dataset.

```text
  risk = P(release contains an identifier)
       x P(recipient can link it to a person)
       x P(the link is attempted)
```

A pipeline controls the first factor. The second depends on the recipient's auxiliary data. The
third depends on the threat model. Safe Harbor is a conservative rule that tries to drive the first
factor to a small value by removing categories. Expert determination estimates the product for an
anticipated recipient. Neither eliminates risk; both define an acceptable level.

A fourth model is the **utility-risk frontier**.

```text
  utility
    |  * no transformation
    |     * redact direct identifiers only
    |          * redact + shift dates
    |               * redact + generalize all quasi-identifiers
    +--------------------------------------------> privacy protection
```

Moving right increases protection and decreases utility. The choice depends on the downstream task.
A temporal phenotyping task needs dates; a topic model may not. A medication extraction task needs
drug names; a demographic analysis may not. The frontier is not a single optimum; it is a set of
defensible operating points, each with a stated use.

A fifth model is the **residual-risk report**, which is the artifact a privacy officer reviews.

```text
  corpus: 5,000 notes, 3,200 patients
  gold: 9,835 identifiers, 7 categories
  detected: 9,310 (recall 0.947)
  residual: 525 identifiers in 410 notes (8.2% of notes)
  by type: LOCATION 89, NAME 198, DATE 98, ...
  utility: medication recall 0.850 -> 0.798
  standard: Safe Harbor, with actual-knowledge review
```

The report is not a certification. It is a measurement that supports a determination, and it should
be dated, versioned, and reproducible.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| De-identification | Removal or transformation of identifiers under a standard | Called anonymization | Dataset |
| Anonymization | Irreversible removal of identifiability | Used as a synonym for de-identification | Dataset |
| Pseudonymization | Replacing identifiers with consistent tokens | Confused with anonymization | Dataset |
| Safe Harbor | The HIPAA rule-based de-identification path | Called the only HIPAA path | Regulatory method |
| Expert determination | The HIPAA risk-based path | Treated as a checklist | Regulatory method |
| Actual knowledge | The Safe Harbor condition that remaining data cannot identify | Ignored | Regulatory condition |
| PHI | Protected health information | Used for de-identified data | Data category |
| Direct identifier | A value that identifies a person on its own | Confused with quasi-identifier | Category |
| Quasi-identifier | A value that identifies in combination | Treated as harmless | Category |
| Residual identifier | An identifier left after transformation | Called a false negative | Span |
| Redaction | Removing or masking a span | Called de-identification | Transformation |
| Surrogate | A realistic replacement value | Called fake data | Transformation |
| Date shifting | Moving dates by a patient-specific offset | Assumed to satisfy Safe Harbor | Transformation |
| Generalization | Replacing a value with a broader category | Confused with redaction | Transformation |
| Placeholder | A fixed token such as `___` | Assumed to be irreversible | Token |
| Over-redaction | Removing non-identifier text | Counted as a false positive | Span |
| Utility loss | Reduction in downstream task performance | Ignored in de-id reports | Metric delta |
| Linkage risk | Risk from combining datasets | Confused with direct identification | Risk |
| Inference risk | Risk from contextual clues | Ignored by regex pipelines | Risk |
| Gold annotation | Human-labeled identifier spans | Assumed complete | Span set |
| Inter-annotator agreement | Consistency of identifier annotation | Reported as recall | Statistic |
| Token-level recall | Share of identifier tokens detected | Confused with span recall | Token |
| Span-level recall | Share of identifier spans detected | Confused with token recall | Span |
| Category recall | Recall within an identifier type | Reported as aggregate | Proportion |
| Residual rate | Missed identifiers per note | Confused with error rate | Count |
| De-identification certificate | Documentation of method and result | Treated as legal advice | Document |
| Data use agreement | Contract governing access | Confused with de-identification | Legal instrument |
| Expert | A person with statistical and scientific knowledge | Assumed to be a clinician | Role |
| Re-identification | Linking de-identified data to a person | Confused with detection | Event |
| Threat model | The recipient and auxiliary data assumed in a risk analysis | Omitted | Assumption |
| Release decision | The documented outcome of the review | Treated as the engineer's call | Decision |
| Identifier inventory | Local forms of each identifier category | Confused with the category list | Inventory |
| Rolling residual sample | Periodic notes re-audited for residual identifiers | Confused with the gold sample | Sample |

Three definitions need emphasis.

**De-identification versus anonymization.** De-identification under HIPAA produces information that
is not individually identifiable, but the regulation allows re-identification codes under 45 CFR
164.514(c) if the code is not derived from the individual's information and the mechanism is
secured. Anonymization, in the general privacy literature, usually implies irreversibility. The two
are not the same, and a contract that says "anonymized" may impose a stricter standard than HIPAA.

**Safe Harbor versus expert determination.** Safe Harbor is a removal checklist. Expert
determination is a risk assessment. A dataset that removes the 18 categories but has an obvious
contextual identifier, such as a note about the only patient in a town with a rare disease, may fail
the actual-knowledge condition. A dataset that keeps dates but demonstrates very small risk under a
documented model may pass expert determination. The paths are different methods, not different
levels of effort.

**Detection recall versus residual risk.** Recall measures the share of gold identifiers detected.
Residual risk is the probability that a released dataset identifies someone. The two are related and
not identical. A missed identifier in a note with no other identifying context may pose less risk
than a correctly detected identifier that is replaced with a linkable surrogate. The report should
present both.

## 5. The mechanics, step by step

This section is the procedure for building and evaluating a de-identification pipeline under a
stated regulatory path.

### Step 1. Choose the regulatory path and state it

**Input:** The intended use, the recipient, and the legal context.

**Output:** A written statement: Safe Harbor, expert determination, or both (Safe Harbor plus
additional safeguards).

**Rationale:** The path determines the transformations, the documentation, and the evaluation. A
team that chooses Safe Harbor must remove all 18 categories and address actual knowledge. A team
that chooses expert determination must define the threat model and the risk threshold.

**Branch point:** The paths have different requirements.

| Path | Requirement | Documentation | Flexibility |
|---|---|---|---|
| Safe Harbor | Remove 18 categories; no actual knowledge | Removal method and verification | Low; rule-based |
| Expert determination | Very small risk under accepted methods | Methods, results, expert qualifications | Higher; risk-based |
| Both | Safe Harbor plus additional safeguards | Both | Highest assurance |

### Step 2. Inventory identifier forms in the corpus

**Input:** A stratified sample of notes.

**Output:** A table of identifier categories, local forms, and frequencies.

**Rationale:** The 18 categories are abstract. The corpus contains local forms: a hospital's MRN
format, a clinic's phone pattern, a provider's name convention, a template's placeholder. The
inventory is the basis for detection rules and for the gold annotation scheme.

**Branch point:** If a category has no observed instances in the sample, it still requires a
detector if the category is in scope. Absence in a sample is not absence in the corpus.

### Step 3. Build the gold annotation scheme

**Input:** The inventory and the regulatory categories.

**Output:** A category scheme with definitions and examples.

**Rationale:** The evaluation is only as good as the gold standard. The scheme should be granular
enough to diagnose failures and coarse enough to annotate reliably. The 2014 i2b2/UTHealth corpus
used seven main categories and 25 subcategories, which is a useful reference point.

**Branch point:** If the scheme has too many subcategories, agreement falls and the evaluation
becomes noisy. If it has too few, a failure in one form is hidden inside a broad category.

### Step 4. Annotate a stratified gold sample

**Input:** The scheme and a sample stratified by note type, site, and year.

**Output:** Gold spans with category labels and adjudication.

**Rationale:** The gold sample estimates recall and precision. It must include the note types and
eras where the pipeline will run, because identifier forms differ.

**Branch point:** If the sample is small, the per-category intervals will be wide. A category with
10 gold instances cannot support a recall claim of 0.95 with any precision.

### Step 5. Normalize text and preserve offsets

**Input:** Raw text.

**Output:** Normalized text plus an offset map.

**Rationale:** Unicode normalization, whitespace collapse, and de-hyphenation change character
positions. Detection on normalized text must map back to the raw text for redaction and evaluation.

**Branch point:** If the pipeline redacts raw text using normalized offsets without a map, it will
redact the wrong characters. The error is silent and can leave part of an identifier in place.

### Step 6. Detect structured identifiers

**Input:** Normalized text.

**Output:** Candidate spans for dates, phone numbers, MRNs, SSNs, emails, URLs, IP addresses, and
account numbers.

**Rationale:** Structured identifiers have patterns and are the easiest category to detect with high
recall. They are also the categories where a single miss is most likely to be a direct identifier.

**Branch point:** Patterns need local variants. A phone regex tuned to one country misses another.
An MRN pattern tuned to one length misses another. The inventory from Step 2 supplies the variants.

### Step 7. Detect names, locations, and organizations

**Input:** Normalized text plus a local dictionary.

**Output:** Candidate spans for names, locations, and organizations.

**Rationale:** Names and locations are open-class and cannot be fully enumerated. A combination of
dictionary matching for known providers and facilities, plus a statistical model for unknown names,
is the practical approach.

**Branch point:** Detection methods have different profiles.

| Method | Recall | Precision | Strength | Weakness |
|---|---|---|---|---|
| Local dictionary | High for known | High | Providers, facilities | Misses new names |
| Title and role patterns | Medium | High | "Dr.", "RN" | Misses untitled names |
| Statistical NER | High | Medium | Unknown names | Domain shift |
| Gazetteer of places | Medium | High | Cities, states | Misses local names |
| Context rules | Medium | High | Patient vs provider | Requires parsing |

### Step 8. Resolve context and attribution

**Input:** Candidate spans plus sentence and section.

**Output:** Spans labeled by role: patient, provider, family, other.

**Rationale:** The transformation depends on the role. A provider name may be replaced with a role
label rather than removed, preserving clinical meaning. A family member's name is PHI and must be
removed. A patient name in a header may be removed entirely.

**Branch point:** Context resolution has three common outcomes.

| Context | Example | Transformation |
|---|---|---|
| Patient | "Ms. Doe was admitted" | Remove or surrogate |
| Provider | "Dr. Smith reviewed" | Replace with role or surrogate |
| Family | "her daughter, Anne" | Remove |
| Placeholder | "John Doe" in a template | Detect and exclude from counts |

### Step 9. Resolve overlaps and choose span boundaries

**Input:** Candidate spans.

**Output:** Non-overlapping spans with the longest sensible boundary.

**Rationale:** "Dr. A. Smith" and "Smith" overlap. "123 Main Street, Brookline" contains a street
and a city. The span boundary determines what is removed and what is counted.

**Branch point:** If the boundary is too short, part of the identifier remains. If it is too long,
clinical text is removed. The gold standard should define the boundary convention.

### Step 10. Choose the transformation

**Input:** Spans, roles, and the regulatory path.

**Output:** Transformed text.

**Rationale:** The transformation determines both residual risk and utility. Redaction is safest and
least useful. Surrogates preserve realism and can introduce linkage if reused. Date shifting
preserves intervals and is not a Safe Harbor removal. Generalization preserves category and loses
specificity.

**Branch point:** The choice has different consequences.

| Transformation | Privacy | Utility | Safe Harbor | Use |
|---|---|---|---|---|
| Redaction to placeholder | High | Low for dates and numbers | Yes, if all categories removed | Release datasets |
| Surrogate replacement | Medium | High | Ambiguous; depends on category | Model development |
| Date shifting | Medium | High for intervals | Not a removal | Expert determination |
| Generalization | Medium | Medium | Yes for some categories | Aggregated analysis |
| Role replacement | Medium | High for providers | Depends on category | Notes for research |

### Step 11. Apply consistently across a patient

**Input:** Transformed notes.

**Output:** A consistent transformation within each patient.

**Rationale:** A surrogate that differs across notes breaks the patient's timeline; a surrogate
reused across patients creates false links. Date shifts must be patient-specific and consistent.
Name surrogates should map one-to-one within a patient and not collide across patients.

**Branch point:** If the transformation is inconsistent, the dataset may be less useful and no
safer. Consistency is a design requirement, not a polish step.

### Step 12. Evaluate detection per category

**Input:** Gold spans and predicted spans.

**Output:** Recall, precision, and F1 per category, with confidence intervals.

**Rationale:** Aggregate recall hides the category that fails. The per-category table is the
primary evaluation artifact.

**Branch point:** Report both span-level and token-level metrics. Span recall is the privacy metric;
token recall explains partial redactions.

### Step 13. Measure residual risk

**Input:** Missed identifiers and their context.

**Output:** Residual identifiers per note, share of notes with at least one, and a contextual risk
review.

**Rationale:** The residual is what matters for privacy. A count of missed spans is not a risk
statement until it is translated into notes and contexts.

**Branch point:** If the residual is concentrated in a note type or section, the fix is targeted.
If it is spread evenly, the detection method is the problem.

### Step 14. Measure utility loss

**Input:** A downstream task run on original and transformed text.

**Output:** Performance delta for the task.

**Rationale:** Utility loss is the cost of protection. A de-identification pipeline that destroys
the task's signal is not usable, regardless of its recall.

**Branch point:** If utility loss is large, consider a less aggressive transformation for the
categories the task needs, or a task-specific release with additional governance.

### Step 15. Document and monitor

**Input:** All measurements.

**Output:** A de-identification report with method, gold sample, per-category metrics, residual
risk, utility, and monitoring plan.

**Rationale:** The report supports the determination and the data use agreement. Monitoring detects
drift after template or vocabulary changes.

**Branch point:** If the report is for expert determination, it must include the threat model, the
risk threshold, and the expert's qualifications. A Safe Harbor report needs the removal method and
the actual-knowledge review.

### Step 16. Make the release decision explicit

**Input:** Per-category metrics, residual rate, utility delta, and the regulatory path.

**Output:** A release decision with conditions and a named approver.

**Rationale:** The pipeline produces measurements; a person or committee produces the decision. The
decision should state which thresholds were met, which were not, and what compensating controls
apply. Without that statement, the measurements are advisory and the release is undocumented.

**Branch point:** The decision has four common outcomes.

| Outcome | Condition | Compensating control |
|---|---|---|
| Release as de-identified | All thresholds met; actual knowledge reviewed | None |
| Release with restrictions | Utility or precision below threshold | Limited recipient, DUA, no redistribution |
| Release after remediation | A category below recall threshold | Fix, re-annotate, re-evaluate |
| Do not release | Residual risk too high | Redesign the data request |

The approver should not be the engineer who built the pipeline. Separation of duties is a control,
not a formality: the builder is invested in the result, and the reviewer's job is to ask whether the
gold sample supports the claim.

### Step 17. Monitor after release

**Input:** The release and its known identifier forms.

**Output:** A monitoring plan with triggers and a schedule.

**Rationale:** Templates change, services merge, and new identifier forms appear. A pipeline
validated on one era can fail on the next without any code change.

**Branch point:** Monitoring has three levels, and each has a different trigger.

| Level | What is monitored | Trigger | Response |
|---|---|---|---|
| Input | Note templates, header formats, new services | New header or field | Inventory and rule update |
| Output | Residual rate on a rolling sample | Rate above baseline | Re-audit; pause release |
| Downstream | Utility on the task | Delta beyond threshold | Reconsider transformation |

A rolling sample of 50 notes per quarter is enough to detect a large change in the residual rate.
Detecting a small change requires a larger sample, and the report should state the detectable
difference rather than implying that the sample proves stability.

### Edge cases that decide de-identification quality

**Negation.** "Patient denies any phone number" still contains a phone number if one is written.
Negation does not make an identifier non-identifying. A pipeline that applies a clinical negation
filter before de-identification can skip a real identifier.

**Hypotheticals.** "If the patient moves to 12 Oak Street" contains an address in a hypothetical.
It is still an identifier if it is the patient's actual address; if it is an example, it may be a
placeholder. Context is required.

**Family history.** Family members' names, ages, and locations are PHI under Safe Harbor because the
rule covers identifiers of relatives, employers, and household members. A pipeline that only targets
the patient misses them.

**Copy-forward duplicates.** An identifier copied into ten notes is ten spans and one person. The
residual-risk report should count notes, not only spans, because a missed identifier in a copied
block propagates.

**Template boilerplate.** Templates may contain example names, default phone numbers, or placeholder
MRNs. These are not real PHI, but they look like it. A pipeline that counts them inflates its
detection numbers; one that removes them may remove a real value in a filled slot.

**Abbreviation ambiguity.** "MS" can be a person's initials or a diagnosis. "PT" can be a patient or
a procedure. Context rules reduce errors; a global dictionary increases them.

**Section boundaries.** Headers and footers contain names, MRNs, and dates. A pipeline that strips
headers before detection loses the identifiers; one that ignores them misses them. The section
policy must be explicit.

**Unicode and OCR noise.** Curly quotes, non-breaking spaces, and OCR substitutions break patterns.
A scanned fax header may contain a phone number with an "O" for a zero. Normalization and a
character-error-aware pattern are required.

## 6. Worked example

A research team is releasing 5,000 discharge summaries for a multi-site study. The regulatory path
is Safe Harbor, with an actual-knowledge review. The team builds a pipeline with regex for structured
identifiers, a local dictionary for providers and facilities, and a statistical model for names.

### Gold annotation

Two annotators label a stratified sample of 800 notes: 400 from the current year and 400 from three
years earlier, across three note templates. Adjudication resolves disagreements. The gold sample
contains 9,835 identifiers across seven categories.

| Category | Gold identifiers | Share |
|---|---:|---:|
| DATE | 3,498 | 35.6% |
| NAME | 2,248 | 22.9% |
| ID (MRN, SSN, account) | 1,455 | 14.8% |
| CONTACT (phone, email, fax, URL) | 1,021 | 10.4% |
| LOCATION | 849 | 8.6% |
| AGE | 462 | 4.7% |
| PROFESSION | 302 | 3.1% |
| Total | 9,835 | 100% |

Token-level inter-annotator F1 on identifier spans is 0.938. The team uses span-level agreement for
the evaluation because the privacy unit is the identifier, not the token.

### Detection results

| Category | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| DATE | 3,400 | 120 | 98 | 0.966 | 0.972 | 0.969 |
| NAME | 2,050 | 310 | 198 | 0.869 | 0.912 | 0.890 |
| ID | 1,430 | 45 | 25 | 0.969 | 0.983 | 0.976 |
| CONTACT | 980 | 60 | 41 | 0.942 | 0.960 | 0.951 |
| LOCATION | 760 | 410 | 89 | 0.650 | 0.895 | 0.753 |
| AGE | 430 | 25 | 32 | 0.945 | 0.931 | 0.938 |
| PROFESSION | 260 | 270 | 42 | 0.491 | 0.861 | 0.625 |
| Total | 9,310 | 1,240 | 525 | 0.882 | 0.947 | 0.913 |

Aggregate recall is 0.947 and aggregate precision is 0.882. The Wilson 95 percent interval for the
aggregate recall, with 9,310 of 9,835, is 0.942 to 0.951. The interval is narrow because the sample
is large. For a rarer category the interval is wider: NAME recall of 2,050 of 2,248 has a Wilson
interval of 0.900 to 0.923, and a category with 100 gold instances would have an interval roughly
four times as wide.

The aggregate hides two problems: LOCATION precision is 0.650, and PROFESSION precision is 0.491.
Both are precision failures, and both affect utility more than privacy. A report that ranked
categories by F1 would put PROFESSION last and might conclude it is the most urgent; a report that
ranked by consequence would put the missed NAME and ID identifiers first, because they are direct
identifiers.

The LOCATION false positives come from a gazetteer that matches common words in clinical text, such
as "Mobile" (a city) and "Reading" (a city). The PROFESSION false positives come from a provider-role
dictionary that matches "attending" and "resident" in clinical phrases, not only in name contexts.
Both are precision failures, not recall failures.

### Residual risk

The 525 missed identifiers are distributed across 410 notes. The residual rate is 525 / 5,000 =
0.105 identifiers per note, and the share of notes with at least one residual identifier is 410 /
5,000 = 0.082, or 8.2 percent.

A per-type residual table is more useful than the aggregate:

| Category | Missed | Residual per 1,000 notes | Share of residuals |
|---|---:|---:|---:|
| NAME | 198 | 39.6 | 37.7% |
| DATE | 98 | 19.6 | 18.7% |
| LOCATION | 89 | 17.8 | 17.0% |
| PROFESSION | 42 | 8.4 | 8.0% |
| CONTACT | 41 | 8.2 | 7.8% |
| AGE | 32 | 6.4 | 6.1% |
| ID | 25 | 5.0 | 4.8% |
| Total | 525 | 105.0 | 100% |

The ID category has the lowest residual count and the highest consequence per miss. A single missed
MRN is a direct identifier. The report should state both the count and the consequence, not rank
categories by count alone.

The share of notes with a residual is 8.2 percent, with a Wilson 95 percent interval of 7.5 to 9.0
percent. The interval is relevant to the release decision: if the policy requires fewer than 5
percent of notes with a residual, the upper bound of 9.0 percent is far above it, and the pipeline
does not meet the policy regardless of the point estimate.

### Where the residuals are

The team reviews the 410 notes with residuals and records the section where each missed identifier
appears.

| Section | Missed identifiers | Share |
|---|---:|---:|
| Header and footer | 320 | 61.0% |
| Free-text body | 150 | 28.6% |
| Signature block | 40 | 7.6% |
| Template field | 15 | 2.9% |
| Total | 525 | 100% |

Sixty-one percent of the residuals are in headers and footers, which the pipeline's section stripper
removed before detection. The stripper was intended to remove boilerplate; it also removed the
identifiers. The fix is to detect on the full text and strip only after detection, or to run a
header-specific detector. This is the first diagnostic: **residuals by section**. A concentration in
one structural region points to a pipeline ordering bug, not a model weakness.

### Utility loss

The downstream task is medication extraction. On the original text, the extractor achieves recall
0.850 (5,100 of 6,000 gold medication mentions). On the de-identified text, 310 medication mentions
are removed because the NAME detector classified drug names as people, and the extractor's recall
falls to 4,790 / 6,000 = 0.798.

| Metric | Original | De-identified | Delta |
|---|---:|---:|---:|
| Medication recall | 0.850 | 0.798 | −0.052 |
| Medication precision | 0.880 | 0.861 | −0.019 |
| Note-level medication F1 | 0.865 | 0.828 | −0.037 |

A 5.2-point recall loss is a real cost. The team reduces it by adding a drug-name exclusion list to
the NAME detector and re-running. The second run removes 120 medication mentions instead of 310, and
recall rises to 4,880 / 6,000 = 0.813. The residual risk is unchanged, because the exclusion affects
false positives, not true identifiers.

### The date transformation decision

Safe Harbor requires removing all date elements except the year, and ages over 89 must be grouped.
The team compares two transformations.

| Transformation | Temporal relation F1 | Safe Harbor | Note |
|---|---:|---|---|
| Remove all dates | 0.520 | Yes | Temporal order lost |
| Keep year only | 0.630 | Yes | Month and day lost |
| Shift dates per patient | 0.910 | No, as a removal method | Preserves intervals |

The shift preserves intervals and raises temporal relation F1 by 28 points over year-only. It is not
a Safe Harbor removal, because the dates are still present, albeit shifted. The team chooses
year-only for the Safe Harbor release and documents the utility loss. A separate expert
determination track is proposed for a shifted-date release with additional governance.

Ages over 89 require separate handling. The gold sample contains 14 notes with an age of 90 or
older. Under Safe Harbor, the age must be grouped as "90 or older," and any date element that
reveals the age must be removed. The pipeline's age detector catches numeric ages but misses ages
written as "nonagenarian" and ages implied by a birth year combined with the note year. The team
adds both patterns and re-annotates the 14 notes. The residual AGE count falls from 32 to 19, and
the residual rate falls from 0.105 to 0.102 per note.

This is the second diagnostic: **utility by transformation**. A pipeline that reports only privacy
metrics will choose the most aggressive transformation and destroy the task. A pipeline that reports
both will make the trade-off explicit.

### Vary one assumption: the gold sample is from one year

Suppose the team had annotated only current-year notes. The three-year-old notes use a different
header format with a fax number and a provider name in the footer. The current-year gold sample
contains no fax numbers, so the pipeline has no fax detector. On the full corpus, the fax residuals
add 140 missed identifiers, raising the residual rate from 105 to 133 per 1,000 notes and the share
of notes with a residual from 8.2 to 9.9 percent.

The conclusion changes: the pipeline is not safe for the historical notes, and the release must
either restrict to the current template era or add a fax detector and re-annotate a historical
sample. A gold sample drawn from one era would have hidden the gap.

## 7. Code, query, or calculation

The following module implements pattern-based detection for structured identifiers, a local name
dictionary, overlap resolution, redaction, and date shifting. It uses only the standard library and
synthetic text.

```python
import re
from datetime import date, timedelta

NOTE = ("Patient: Jane Q. Doe  MRN: 00451239  DOB: 03/14/1958  "
        "Seen 12/02/2025 by Dr. A. Smith. Call (617) 555-0142 or jane.doe@example.org. "
        "SSN 123-45-6789. Visit https://example.org/portal/abc. "
        "Lives in Brookline. Started metformin 500 mg on 11/20/2025.")

NAMES = ["Jane Q. Doe", "A. Smith"]
PATTERNS = {
    "DATE": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    "PHONE": r"\(?\d{3}\)?[-. ]\d{3}[-.]\d{4}",
    "MRN": r"\bMRN[:\s]*\d{6,10}\b",
    "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "URL": r"https?://[^\s]+[^\s.,;)]",
}


def detect(text):
    out = []
    for label, pat in PATTERNS.items():
        for m in re.finditer(pat, text):
            out.append((m.start(), m.end(), label))
    for name in NAMES:
        for m in re.finditer(re.escape(name), text):
            out.append((m.start(), m.end(), "NAME"))
    return sorted(out)


def resolve(spans):
    """Drop spans fully contained in a longer span."""
    kept = []
    for s in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        if any(s[0] >= k[0] and s[1] <= k[1] for k in kept):
            continue
        kept.append(s)
    return kept


spans = resolve(detect(NOTE))
for s in spans:
    print(s, repr(NOTE[s[0]:s[1]]))

redacted = NOTE
for s, e, label in sorted(spans, key=lambda x: -x[0]):
    redacted = redacted[:s] + f"[{label}]" + redacted[e:]
print(redacted)


def shift_dates(text, days):
    def repl(m):
        d = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        return (d + timedelta(days=days)).strftime("%m/%d/%Y")
    return re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", repl, text)


shifted = shift_dates(NOTE, -37)
print([m.group(0) for m in re.finditer(r"\b\d{1,2}/\d{1,2}/\d{4}\b", shifted)])
print("interval days preserved:", (date(2025, 12, 2) - date(2025, 11, 20)).days)
```

Expected output:

```text
(9, 20, 'NAME') 'Jane Q. Doe'
(22, 35, 'MRN') 'MRN: 00451239'
(42, 52, 'DATE') '03/14/1958'
(59, 69, 'DATE') '12/02/2025'
(77, 85, 'NAME') 'A. Smith'
(92, 106, 'PHONE') '(617) 555-0142'
(110, 130, 'EMAIL') 'jane.doe@example.org'
(136, 147, 'SSN') '123-45-6789'
(155, 185, 'URL') 'https://example.org/portal/abc'
(235, 245, 'DATE') '11/20/2025'
Patient: [NAME]  [MRN]  DOB: [DATE]  Seen [DATE] by Dr. [NAME]. Call [PHONE] or [EMAIL]. SSN [SSN]. Visit [URL]. Lives in Brookline. Started metformin 500 mg on [DATE].
['02/05/1958', '10/26/2025', '10/14/2025']
interval days preserved: 12
```

The output shows the intended behavior and two limitations. "Brookline" is not detected because it
is a location without a gazetteer entry. "metformin" is not detected because the name dictionary is
limited to known names. Both are reasons a production pipeline needs a location dictionary and a
statistical name model, and both would appear as false negatives in a per-category evaluation.

A second block computes per-category metrics and the residual rate:

```python
rows = [
    ("DATE", 3400, 120, 98),
    ("NAME", 2050, 310, 198),
    ("ID", 1430, 45, 25),
    ("CONTACT", 980, 60, 41),
    ("LOCATION", 760, 410, 89),
    ("AGE", 430, 25, 32),
    ("PROFESSION", 260, 270, 42),
]
for name, tp, fp, fn in rows:
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    print(f"{name}: p={p:.3f} r={r:.3f} f1={2*p*r/(p+r):.3f}")

tp = sum(r[1] for r in rows)
fp = sum(r[2] for r in rows)
fn = sum(r[3] for r in rows)
p, r = tp / (tp + fp), tp / (tp + fn)
print(f"total: p={p:.3f} r={r:.3f} f1={2*p*r/(p+r):.3f}")
print("gold:", tp + fn, "residual per note:", round(fn / 5000, 3))
print("notes with residual:", round(410 / 5000, 3))
```

Expected output:

```text
DATE: p=0.966 r=0.972 f1=0.969
NAME: p=0.869 r=0.912 f1=0.890
ID: p=0.969 r=0.983 f1=0.976
CONTACT: p=0.942 r=0.960 f1=0.951
LOCATION: p=0.650 r=0.895 f1=0.753
AGE: p=0.945 r=0.931 f1=0.938
PROFESSION: p=0.491 r=0.861 f1=0.625
total: p=0.882 r=0.947 f1=0.913
gold: 9835 residual per note: 0.105
notes with residual: 0.082
```

The per-category lines reproduce Section 6. The total recall of 0.947 is above the 0.90 threshold
the team set for release, but the PROFESSION precision of 0.491 and the LOCATION precision of 0.650
are below the 0.75 threshold. The release decision depends on which metric the policy prioritizes:
recall for privacy, precision for utility. The report should state both and the threshold for each.

The code deliberately prints the per-category table before the total. A reader who sees only the
total would not know that LOCATION precision is below any reasonable utility threshold. The
ordering is a small design choice with a large effect on how the result is read.

## 8. Data, standards, and regulatory context

**HIPAA Safe Harbor.** 45 CFR 164.514(b)(2) permits a covered entity or business associate to
treat information as de-identified if 18 categories of identifiers are removed. The categories are:
names; geographic subdivisions smaller than a state, with a ZIP-code exception for the first three
digits where the Census unit has more than 20,000 people; all date elements other than year for
dates related to the individual, including birth, admission, discharge, and death, and all ages over
89; telephone numbers; fax numbers; email addresses; social security numbers; medical record
numbers; health plan beneficiary numbers; account numbers; certificate and license numbers; vehicle
identifiers and license plates; device identifiers and serial numbers; web URLs; IP addresses;
biometric identifiers including fingerprints and voice prints; full-face photographs and comparable
images; and any other unique identifying number, characteristic, or code, except as permitted by
paragraph (c).

The rule also requires that the covered entity have no actual knowledge that the remaining
information could be used, alone or in combination with other reasonably available information, to
identify the individual. The actual-knowledge condition is the part most often omitted from
technical pipelines, and it is the part that addresses contextual identifiers.

**Re-identification codes.** 45 CFR 164.514(c) permits a code or other record identifier to be
attached to de-identified information if the code is not derived from or related to information
about the individual, cannot be translated to identify the individual, is not used or disclosed for
any other purpose, and the mechanism for re-identification is not disclosed. A surrogate that is
derived from the original name, such as a hash of the name, may fail the first condition. A
sequential study ID assigned independently satisfies it.

**Expert determination.** 45 CFR 164.514(b)(1) requires a person with appropriate knowledge of and
experience with generally accepted statistical and scientific principles and methods to determine
that the risk is very small that the information could be used, alone or in combination with
reasonably available information, by an anticipated recipient to identify an individual. The expert
must document the methods and results. The regulation does not specify a risk threshold, a model, or
a validation method; those are the expert's responsibility and the subject of professional debate.

**What the standard does not cover.** Safe Harbor does not require removal of clinical information,
free-text narrative, or rare diagnoses. It does not address inference from combinations of retained
quasi-identifiers beyond the actual-knowledge condition. It does not certify a pipeline. A dataset
that removes all 18 categories can still be re-identifiable in a small population, which is why
expert determination and data use agreements remain relevant.

**GDPR and pseudonymization.** The EU General Data Protection Regulation treats pseudonymized data
as personal data and does not equate pseudonymization with anonymization. A dataset that is
de-identified under HIPAA may still be personal data under GDPR. A multi-jurisdiction project needs
a legal analysis per jurisdiction, not a single pipeline claim.

**PhysioNet and MIMIC.** MIMIC-IV-Note v2.2 is distributed under the PhysioNet Credentialed Health
Data License 1.5.0, with access requiring CITI training and a data use agreement. The dataset
documentation reports that PHI was replaced with three underscores, that de-identification used a
rule-based system plus a neural network, that sensitivity for radiology reports was 99.9 percent,
and that manual review of discharge summaries found no PHI. Those are the dataset owners' reported
results on their corpus; they are not a guarantee for a different corpus or a different pipeline.

**Common Rule and IRB.** The HIPAA de-identification standard and the Common Rule's human-subjects
regulations are separate. An IRB may require protections beyond de-identification, and a
determination that research is not human-subjects research may depend on the data being
de-identified. The two analyses should be documented separately.

**FDA and device data.** A de-identified dataset used to develop a device is still subject to the
device's validation and cybersecurity expectations. De-identification reduces privacy risk; it does
not reduce the need for data provenance, reference standards, or subgroup performance reporting.

**State law.** Some U.S. states impose stricter de-identification or breach-notification rules than
HIPAA. A pipeline designed for HIPAA may not satisfy a state standard. The legal review should
identify the applicable regimes before the pipeline is built.

**Institutional policy.** Many health systems add internal requirements beyond HIPAA: a minimum
recall for direct identifiers, manual review of a sample, a prohibition on releasing free-text
notes without a second review, or a data classification that restricts where the data can be
stored. These policies are binding on the project even when they are not law. A de-identification
report written for an external audience may not satisfy an internal reviewer who is applying a
stricter standard.

**Research versus operations.** A quality-improvement project that reviews notes internally has a
different risk profile from a research release that sends notes to an external collaborator. The
former may be covered by treatment, payment, and operations provisions; the latter usually requires
a data use agreement or an IRB determination. The de-identification method may be the same, and the
governance is not. The report should state which use it supports.

**Clinical trial and registry contexts.** A trial sponsor may require de-identified text for
central review, and the protocol may specify the identifiers to remove. The sponsor's specification
is a contractual requirement that can be stricter than HIPAA. A pipeline that meets Safe Harbor and
not the sponsor's specification has not met the protocol.

**Secondary use of de-identified data.** De-identified data can generally be used for research
without authorization, subject to IRB and contractual requirements. That permission is the reason
de-identification matters commercially: it enables secondary use at scale. It also creates an
incentive to overstate de-identification quality, which is why the measurement and the documentation
matter. This is an inference from the structure of the rules, not a claim about any specific
organization.

> **Note:** Date shifting preserves intervals and is useful for research, but it is not a Safe
> Harbor removal. A release that shifts dates instead of removing them needs an expert determination
> or another legal basis, and the documentation should say so.

## 9. Industry practice and case evidence

**MIMIC-IV-Note.** The dataset documentation for v2.2 reports 331,794 discharge summaries and
2,321,355 radiology reports, with PHI replaced by three underscores. De-identification combined a
rule-based approach with a neural network, the annotations were unioned, and the reported sensitivity
for radiology reports was 99.9 percent. Manual review of discharge summaries found no PHI. The
documentation also states that notes are generally limited to within one year of an encounter and
that the release is HIPAA Safe Harbor compliant. These are the dataset owners' claims about their
corpus.

**n2c2 2014 de-identification challenge.** The 2014 i2b2/UTHealth corpus contained 1,304 longitudinal
records for 296 patients, split into 790 training and 514 test records. Annotation used seven main
PHI categories and 25 subcategories, with double annotation and arbitration; the training set
contained 17,389 mentions and the test set 11,462. The annotation paper reported token-level
inter-annotator F1 of 0.927. The challenge's best system reached 0.964 entity-based micro F1, and the
mean across systems was 0.872. These are the challenge's reported figures on its corpus and
evaluation script.

**The category-level lesson.** The n2c2 2014 results were reported per category, not only in
aggregate, because the categories have different base rates and different consequences. A system
with high aggregate recall can have low recall on a rare category such as device identifiers, and
that category may be the one that matters for a specific release. The practice is now standard in
de-identification evaluations and should be required in procurement.

**Residual-risk practice.** Organizations that release de-identified text commonly perform a
residual audit on a sample after automated de-identification, and some perform a manual review of
every note for a limited release. The MIMIC documentation's manual review of discharge summaries is
an example of the latter. The practice is expensive and is justified when the release is broad or
the corpus is small enough that a single missed identifier is high-risk. Inference: the choice
between automated-only and automated-plus-review is a risk decision, not a technical one.

**Date shifting in practice.** Many research data warehouses shift dates by a patient-specific
offset to preserve intervals while reducing re-identification risk. The method is widely used and is
generally treated as a risk-reduction technique under expert determination, not as a Safe Harbor
removal. The documentation should state the offset distribution and whether the shift is consistent
across datasets. Inference: the practice is more common than the published literature suggests,
because the implementations are internal.

**Vendor claims.** De-identification vendors often report recall on a public benchmark and describe
their product as "HIPAA compliant." Compliance is a property of a use, not a product. A buyer should
ask for the standard path, the per-category recall on the buyer's note types, the residual rate, and
the utility impact. A vendor that cannot provide them has not evaluated the deployment. This is an
inference from procurement practice.

**What the evidence does not establish.** No published pipeline has been shown to achieve zero
residual identifiers on an open clinical corpus. The reported figures are estimates on samples, with
confidence intervals that are often wide for rare categories. A claim of "99.9 percent" should be
read with its denominator and its category definition.

**Human review as a control.** Some releases use automated de-identification followed by manual
review of every note. The MIMIC documentation's manual review of discharge summaries is an example
of the pattern. The control is expensive and it changes the error profile: human reviewers catch
contextual identifiers that rules miss, and they introduce fatigue and inconsistency. A hybrid
pipeline should evaluate the combined recall, not the automated component's recall, and should
report the reviewer's throughput and agreement.

**Benchmark-to-deployment gap.** Public de-identification benchmarks use a fixed category scheme
and a fixed corpus. A deployment has local identifier forms, a different note mix, and a different
threat model. A system that scores 0.96 on a benchmark can have a materially lower recall on a
local corpus with a different header format. The gap is not a model failure; it is a distribution
change, and the only defence is local evaluation. This is an inference from the structure of the
benchmarks, not a measured effect size.

**Procurement questions.** A buyer of a de-identification service should ask for the standard and
path, the per-category recall on a sample of the buyer's notes, the residual rate, the transformation
method, and the utility impact. A vendor that reports only a benchmark score has not addressed the
deployment. The questions are the same ones an expert determination would ask, which makes them a
useful screening tool even when the buyer is not the one making the legal determination.

**The economics of recall.** Increasing recall on names and locations has a cost: more false
positives, more manual review, and more utility loss. The marginal cost rises as recall approaches
1, because the remaining misses are the hardest cases: untitled names, local place names, and
contextual references. A program that targets 0.99 recall should expect a disproportionate increase
in review cost. The target should follow from the threat model, not from a general preference for
higher numbers.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Header stripped before detection | Residuals concentrated in headers | Section filter removes identifiers | Leak | Detect on full text |
| Name detector misses untitled names | Recall gap in free text | Dictionary only | Leak | Statistical NER |
| Location gazetteer over-matches | Low location precision | Common words are city names | Utility loss | Context rules |
| Date removal breaks temporal task | Downstream F1 falls | All dates removed | Utility loss | Year-only or expert determination |
| Surrogate derived from identifier | Hash or initials | Linkable | Re-identification | Independent study IDs |
| Surrogate reuse across patients | Same fake name appears | Pool too small | False links | One-to-one mapping |
| Family names missed | Relatives not in dictionary | Patient-only focus | Leak | Family context rules |
| Template placeholders counted as PHI | High detection in templates | Example names | Inflated metrics | Placeholder detection |
| Fax or scanner identifiers missed | Historical notes fail | No pattern for fax | Leak | Era-specific inventory |
| Unicode normalization breaks offsets | Redaction shifts | Offset map absent | Partial leak | Map to raw text |
| Token-level recall reported | Span recall lower | Partial redaction | Overstated privacy | Report both |
| Aggregate recall only | Rare category fails | Averaging | Leak in specific type | Per-category table |
| No utility measurement | Task performance unknown | Privacy-only evaluation | Unusable release | Downstream delta |
| Actual knowledge not addressed | Contextual identifiers remain | Checklist-only | Legal failure | Context review |
| Pipeline not monitored | Template change unnoticed | No re-audit | Drift | Periodic gold sample |
| Age over 89 not grouped | Numeric age present | Safe Harbor requires grouping | Legal failure | Age rule and review |
| ZIP code retained fully | Full ZIP in address | Three-digit rule ignored | Legal failure | ZIP-specific rule |
| Derived surrogate | Hash or initials | Linkable to identifier | Re-identification | Independent IDs |
| Threat model unstated | "Low risk" claim without recipient | No auxiliary-data analysis | Unsupported determination | Document threat model |

The red flags cluster in two patterns. The first is a pipeline that reports one aggregate recall
without per-category results, residual rate, or utility. The second is a transformation that is
described as Safe Harbor but uses a method Safe Harbor does not permit, such as date shifting or a
derived surrogate. Both are detectable by asking for the method and the category table.

A third pattern is the missing threat model. A report that says "risk is low" without naming the
anticipated recipient and the auxiliary data available to that recipient has not assessed risk; it
has asserted it. The reviewer's question is not "how accurate is the model" but "who could identify
whom, using what other data, and how would they do it." A pipeline can have excellent recall and
still fail that question if the retained quasi-identifiers are sufficient for the recipient's
auxiliary data.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The gold sample represents the corpus | Supports recall estimate | Era or note-type skew | Stratify; re-annotate |
| Identifier forms are stable | Detection rules depend on them | New template or vendor | Monitor residuals |
| The category scheme covers the risk | Evaluation scope | Contextual identifiers | Expert review |
| Detection recall is the main risk | Prioritizes metric | Linkable surrogates | Transformation audit |
| Transformations are irreversible | Supports release | Derived surrogates | Code review |
| Utility loss is acceptable | Justifies transformation | Task-critical terms removed | Downstream evaluation |
| The regulatory path is correctly chosen | Legal basis | Shifting dates under Safe Harbor | Legal review |
| Annotators agree on boundaries | Gold quality | Ambiguous spans | Agreement study |
| OCR noise is bounded | Pattern performance | Scanned documents | Character error rate |
| The recipient's auxiliary data is unknown | Threat model | Public registry linkage | Expert determination |
| Monitoring will catch drift | Sustained safety | No re-audit | Schedule and owner |

The strongest assumption is that the gold sample represents the corpus. In the worked example, a
current-year-only sample missed the fax identifiers in historical notes, and the residual rate rose
from 8.2 to 9.9 percent when the full corpus was considered. That is a 21 percent relative increase
in the share of notes with a residual identifier, caused entirely by sampling.

The second strongest is that detection recall is the main risk. A pipeline with high recall and a
linkable surrogate can still re-identify. The transformation must be audited as carefully as the
detection.

A limit of the standard itself is that Safe Harbor is a rule, not a risk model. It can be satisfied
while leaving contextual identifiers, and it can be failed while a dataset is effectively
non-identifiable. The actual-knowledge condition and expert determination are the mechanisms for
the gap, and they require judgment that a pipeline cannot supply.

A practical limit is that the gold annotation is itself a disclosure risk. The gold spans quote
identifiers, and the annotation file may be stored outside the de-identified environment. The
annotation workflow should treat the gold file as PHI, restrict access, and avoid copying
identifiers into shared spreadsheets or chat tools. This is an operational failure that is easy to
overlook because the annotation is "just labels."

A final limit is that de-identification does not address the risk of membership inference or
memorization in a model trained on the text. A model can encode information about a patient even
when the training text is de-identified, and a generative model can reproduce rare strings. The
evaluation of a model's privacy properties is a separate exercise from the de-identification of its
training data, and it should be reported separately.

## 12. Exercises

**Exercise 8.1 — Safe Harbor mapping (application, 40 minutes).**
Deliverable: a table mapping the 18 Safe Harbor categories to detection methods and local forms for
a corpus you know or can describe.
Acceptance criteria: 18 rows; a method per category; at least three local forms.

**Exercise 8.2 — Per-category metrics (calculation, 40 minutes).**
Deliverable: using the Section 6 table, compute aggregate precision and recall, and identify the two
categories with the largest utility problem.
Acceptance criteria: 0.882 and 0.947; LOCATION and PROFESSION identified with reasons.

**Exercise 8.3 — Residual rate (calculation, 30 minutes).**
Deliverable: compute residual identifiers per note and the share of notes with a residual, then
recompute if the gold sample misses a historical category that adds 140 misses.
Acceptance criteria: 0.105 and 0.082; 0.133 and 0.099 after the addition.

**Exercise 8.4 — Transformation choice (design, 45 minutes).**
Deliverable: for a temporal phenotyping task, compare redaction, year-only, and date shifting on
privacy, utility, and regulatory path.
Acceptance criteria: three options; a table; a recommendation with a legal caveat.

**Exercise 8.5 — Utility evaluation (design, 40 minutes).**
Deliverable: a plan to measure downstream utility loss for a medication extraction task, including
the metric, sample, and acceptance threshold.
Acceptance criteria: metric and unit; sample size; threshold; a decision rule.

**Exercise 8.6 — Expert determination outline (design, 50 minutes).**
Deliverable: an outline of an expert determination for a dataset that keeps shifted dates, including
threat model, risk threshold, and evidence.
Acceptance criteria: threat model; threshold; evidence; expert qualifications.

**Exercise 8.7 — Residual audit (application, 45 minutes).**
Deliverable: a residual audit procedure for 200 released notes, with a sampling rule, a review
form, and an escalation rule for a found identifier.
Acceptance criteria: sampling rule; form fields; escalation owner.

**Exercise 8.8 — Critique a vendor claim (critique, 30 minutes).**
Deliverable: critique "99.9 percent accurate, HIPAA compliant" with five questions.
Acceptance criteria: five questions; each tied to a standard or metric.

**Exercise 8.9 — Release decision (application, 40 minutes).**
Deliverable: using the Section 6 metrics and a policy of at least 0.95 recall for NAME and ID and
at most 0.05 notes with a residual, state the release decision, the conditions, and the remediation
steps.
Acceptance criteria: decision; each threshold assessed; a remediation plan with an owner.

## 13. Self-check answers

**8.2 (per-category).** Sums: TP 9,310, FP 1,240, FN 525. Aggregate precision = 9,310 / 10,550 =
0.882; recall = 9,310 / 9,835 = 0.947. The largest utility problems are LOCATION precision 0.650
(410 false positives, mostly common words in the gazetteer) and PROFESSION precision 0.491 (270
false positives, role words matched outside name contexts). Both are precision problems, so the fix
is context rules, not more detection.

**8.3 (residual).** Base: 525 / 5,000 = 0.105 per note; 410 / 5,000 = 0.082. Adding 140 misses:
665 / 5,000 = 0.133 per note; if the 140 are in 85 additional notes, 495 / 5,000 = 0.099. The
relative increase in notes with a residual is 21 percent.

**8.1 (Safe Harbor) rubric.** Full credit: 18 categories, a method per category, three local forms,
and a note on actual knowledge. Half credit: 15 categories without local forms.

**8.4 (transformation) rubric.** Full credit: privacy, utility, and regulatory path for each of the
three options; recommends year-only for Safe Harbor and shifting for expert determination; states
that shifting is not a Safe Harbor removal. Half credit: comparison without the regulatory column.

**8.5 (utility) rubric.** Full credit: medication recall at the mention and note level, a stratified
sample, an acceptance threshold, and a decision rule if the threshold is missed. Half credit: metric
without threshold.

**8.6 (expert determination) rubric.** Full credit: threat model naming anticipated recipients and
auxiliary data; a stated risk threshold; evidence from per-category recall and linkage analysis;
expert qualifications. Half credit: outline without a threat model.

**8.7 (residual audit) rubric.** Full credit: sampling rule, review form with category and context,
escalation owner and timeline, and a remediation step for the release. Half credit: procedure
without escalation.

**8.8 (vendor critique).** Questions: which de-identification standard and path; per-category recall
on our note types; residual rate per note and share of notes; utility impact on our downstream task;
how surrogates are generated and whether they are derived from identifiers. "HIPAA compliant" is a
property of a use, not a product, and "99.9 percent accurate" needs a denominator and a category.

**8.9 (release decision) rubric.** Full credit: notes that ID recall 0.983 meets the threshold but
NAME recall 0.912 does not; notes that 8.2 percent of notes with a residual exceeds the 5 percent
policy; recommends remediation (name model, header detection) and a restricted release in the
interim. Half credit: decision without threshold-by-threshold assessment.

## 14. Interview questions

**1. "What are the two HIPAA de-identification paths?"**
Strong answer: Safe Harbor removal of 18 categories plus actual knowledge, and expert determination
of very small risk with documented methods. Weak answer: "remove names and dates."

**2. "Why is de-identification recall more important than precision?"**
Strong answer: a false negative is a privacy breach; a false positive is utility loss; the
consequences are asymmetric. Weak answer: "both matter equally."

**3. "What is a quasi-identifier?"**
Strong answer: a value that identifies in combination, such as date, ZIP, age, or rare diagnosis;
Safe Harbor addresses some but not inference. Weak answer: "a fake identifier."

**4. "How would you evaluate a de-identification pipeline?"**
Strong answer: per-category recall and precision, residual rate, share of notes with residuals,
utility delta, and confidence intervals. Weak answer: "accuracy on a test set."

**5. "What is date shifting and does it satisfy Safe Harbor?"**
Strong answer: patient-specific offset preserving intervals; not a Safe Harbor removal because dates
remain; expert determination or another basis is needed. Weak answer: "yes, it anonymizes dates."

**6. "How do you handle names of family members?"**
Strong answer: they are PHI under Safe Harbor; detect with context; remove; test in the gold sample.
Weak answer: "only patient names matter."

**7. "What is residual risk?"**
Strong answer: the probability that released data identifies someone after transformation; a
function of missed identifiers, linkage, and context. Weak answer: "the error rate."

**8. "Why can aggregate recall mislead?"**
Strong answer: rare categories can fail while common categories carry the average; gives a device
identifier example. Weak answer: "averages are fine."

**9. "How would you test utility loss?"**
Strong answer: run the downstream task on original and transformed text, report the delta at the
task unit, and set an acceptance threshold. Weak answer: "check that the text is readable."

**10. "What is actual knowledge?"**
Strong answer: the Safe Harbor condition that remaining information cannot identify; requires a
context review beyond the checklist. Weak answer: "knowing the patient."

**11. "How do you keep surrogates from being linkable?"**
Strong answer: independent study IDs, one-to-one mapping, no derivation from identifiers, and
secured re-identification. Weak answer: "use fake names."

**12. "What would you monitor after release?"**
Strong answer: residual audits on new eras, template changes, new identifier forms, and downstream
utility. Weak answer: "nothing, it is de-identified."

## 15. Key terms

- **Actual knowledge** — the Safe Harbor condition on remaining identifiability.
- **Anonymization** — irreversible removal of identifiability.
- **Category recall** — recall within an identifier type.
- **Data use agreement** — a contract governing access.
- **Date shifting** — moving dates by a patient-specific offset.
- **De-identification** — removal or transformation of identifiers under a standard.
- **Direct identifier** — a value that identifies a person on its own.
- **Expert determination** — the HIPAA risk-based path.
- **Generalization** — replacing a value with a broader category.
- **Gold annotation** — human-labeled identifier spans.
- **Identifier taxonomy** — categories of direct, quasi-, and contextual identifiers.
- **Inference risk** — risk from contextual clues.
- **Linkage risk** — risk from combining datasets.
- **Over-redaction** — removing non-identifier text.
- **PHI** — protected health information.
- **Placeholder** — a fixed token replacing an identifier.
- **Pseudonymization** — replacing identifiers with consistent tokens.
- **Quasi-identifier** — a value that identifies in combination.
- **Redaction** — removing or masking a span.
- **Re-identification** — linking de-identified data to a person.
- **Residual identifier** — an identifier left after transformation.
- **Residual rate** — missed identifiers per note.
- **Safe Harbor** — the HIPAA rule-based de-identification path.
- **Span-level recall** — share of identifier spans detected.
- **Surrogate** — a realistic replacement value.
- **Token-level recall** — share of identifier tokens detected.
- **Transformation** — redaction, surrogate, shift, or generalization.
- **Utility loss** — reduction in downstream task performance.
- **Threat model** — the recipient and auxiliary data assumed in a risk analysis.
- **Release decision** — the documented outcome of the de-identification review.
- **Identifier inventory** — the local forms of each identifier category.
- **Rolling residual sample** — periodic notes re-audited for residual identifiers.

## 16. Sources

All URLs accessed 25 October 2026.

1. 45 CFR 164.514, de-identification standard. Cornell Legal Information Institute.
   https://www.law.cornell.edu/cfr/text/45/164.514
2. U.S. Department of Health and Human Services. "Summary of the HIPAA Privacy Rule:
   De-identification."
   https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
3. PhysioNet. MIMIC-IV-Note: Deidentified free-text clinical notes, v2.2 (6 January 2023).
   https://physionet.org/content/mimic-iv-note/2.2/
4. Stubbs A, Uzuner O. "Annotating longitudinal clinical narratives for de-identification: The 2014
   i2b2/UTHealth corpus." J Biomed Inform. 2015;58 Suppl:S20-S29.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC4978170/
5. Uzuner O, Luo Y, Szolovits P. "Evaluating the state-of-the-art in automatic de-identification."
   J Am Med Inform Assoc. 2007;14(5):550-563. https://doi.org/10.1197/jamia.M2444
6. Yang H, Garibaldi JM. "Automatic detection of protected health information from clinic
   narratives." J Biomed Inform. 2015;58 Suppl:S30-S38.
   https://doi.org/10.1016/j.jbi.2015.06.015
7. n2c2. "2014 De-identification and Heart Disease Risk Factors Challenge."
   https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/
8. PhysioNet. "Credentialed Health Data License 1.5.0."
   https://physionet.org/content/mimic-iv-note/2.2/license/
9. U.S. Department of Health and Human Services. "Business Associates."
   https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/business-associates/index.html
10. European Parliament and Council. Regulation (EU) 2016/679 (General Data Protection Regulation),
    Article 4(5) on pseudonymisation.
    https://eur-lex.europa.eu/eli/reg/2016/679/oj
11. U.S. Food and Drug Administration. "Artificial Intelligence-Enabled Device Software Functions:
    Lifecycle Management and Marketing Submission Recommendations." Draft guidance, January 2025.
    https://www.fda.gov/media/184856/download

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_evaluation.md`](07_evaluation.md)
- Next topic: [`09_summarization-and-grounding.md`](09_summarization-and-grounding.md)
- Annotation of identifier spans: [`06_annotation.md`](06_annotation.md)
- Evaluation metrics: [`07_evaluation.md`](07_evaluation.md)
- Source provenance and data use: [`01_clinical-text-sources.md`](01_clinical-text-sources.md)
- Module survey: [`../../09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
