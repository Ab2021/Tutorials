# Classical Methods

> **Module 09 — Clinical NLP**
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Topics 01, 02, and 03
> Version 2026.3

## 1. Why this topic exists

The clinical NLP literature has a long record of rule-based and dictionary-based systems that
remain in production years after transformer models became available. This is not nostalgia. Rules
are auditable, deterministic, cheap to run, and easy to constrain. A rule that says "a medication
mention preceded by 'no' within the same clause is negated" can be read by a clinician, tested on a
sample, and changed in an afternoon. A transformer's behavior cannot be read, and changing it
requires retraining and revalidation.

The evidence supports the continued use of classical methods in specific conditions. The 2018 n2c2
cohort-selection challenge reported a best micro F1 of 0.91 from a rule-based system, with four
rule-based and six hybrid systems in the top ten and no purely machine-learning system in that
group. The 2010 i2b2/VA assertion task, which supplied gold concepts to competitors, was won by
systems that combined support vector machines with context features and negation dictionaries, with
the best F1 at 0.936. These results do not show that rules beat learning in general. They show that
when the task is well specified, the evidence is explicit, and the label set is small, rules and
dictionaries are competitive.

The practical reason classical methods matter is different: they establish the baseline that any
learned model must beat. A transformer evaluated without a dictionary or rule baseline may look
strong while a twenty-line regex achieves the same recall. The comparison is not academic. It
determines whether the project needs a model at all, and it determines what the model's marginal
value is.

Classical methods also carry most of the module's spine explicitly. NegEx, published by Chapman and
colleagues in 2001, is a regular-expression algorithm for identifying negated findings in discharge
summaries. It uses negation phrases, pseudo-negation filters such as "gram-negative" and "not ruled
out," and a scope limit of up to five intervening tokens. ConText, published by Harkema and
colleagues in 2009, extends the approach to negation, temporality, and experiencer, and replaces the
token window with scope that runs to a termination term or the end of the sentence. These are
readable specifications of the mention-to-assertion transition, and reading them teaches the
problem better than most model papers.

The limitations are equally well documented. NegEx achieved a sensitivity of 77.8 percent and a
specificity of 94.5 percent on 1,000 sentences from intensive care discharge summaries, against a
simpler baseline with higher sensitivity and much lower specificity. It missed long comma-separated
lists that exceeded the five-token window, passive constructions such as "nephrotic syndrome was
ruled out," and the distinction between past history and current visit. ConText improved recall but
reported a historical-status F-score of 0.76 against 0.93 for negation, and its authors concluded
that historical status needs lexical disambiguation, section context, discourse, and medical
knowledge beyond surface triggers.

Those error analyses are the reason this topic is not a defense of rules. They are a catalog of what
surface patterns cannot do. A team that adopts a rule-based system without reading the failure modes
will reproduce them. A team that reads them can decide which failures matter for its decision and
which can be tolerated.

Classical methods also provide the interpretable layer in hybrid systems. A common production
pattern is a learned extractor followed by a rule-based assertion filter and a dictionary-based
normalizer. The rules handle the cases where the model is least reliable and most opaque: negation
scope, family history, section-specific conventions, and local abbreviations. The hybrid is not a
compromise; it is a division of labor that matches each method's strengths.

There is a governance argument as well. In a regulated environment, a reviewer may ask why a
specific patient was flagged. A rule system can answer with the exact pattern, lexicon entry, and
scope decision. A model can answer with an attribution score that is not a reason. The difference
matters for appeals, audits, and clinician trust. It does not make rules more accurate; it makes
their errors legible, which is often the more valuable property in a clinical workflow.

> **Note:** A rule-based system is software, and software has versions. A regex change that alters
> negation scope can change a cohort size. The change-control requirements in regulated settings
> apply to rules as much as to models.

## 2. Learning outcomes

By the end of this topic you can:

1. **Implement** a dictionary-based mention extractor with longest-match overlap resolution and
   character offsets.
2. **Implement** a NegEx-style negation detector with pseudo-trigger filtering, termination terms,
   and sentence scope.
3. **Compare** token-window scope and termination-based scope on a sample and **quantify** the
   difference in negation recall and precision.
4. **Build** a rule-based pipeline with section filtering, abbreviation handling, and assertion
   assignment, and **evaluate** it at span and patient level.
5. **Explain** the published performance and failure modes of NegEx and ConText.
6. **Choose** between a rule-based, dictionary-based, classical statistical, and transformer
   approach for a stated task, with reasons.
7. **Design** a lexicon maintenance process with ownership, versioning, and regression tests.
8. **Critique** a rule-based system description that omits scope rules, pseudo-triggers, or
   evaluation at the decision unit.

Outcomes 1 through 4 are implementation; 5 through 8 are analysis and design.

## 3. The core mental model

A classical clinical NLP pipeline is a sequence of filters, each of which removes or labels
candidates. Recall is lost at every filter, and precision is gained.

```text
raw text
   |
   v
normalize  ------------------------>  case, Unicode, whitespace
   |
   v
segment    ------------------------>  sections, sentences
   |
   v
candidate generation  ------------->  dictionary / regex  (high recall, low precision)
   |
   v
overlap resolution  --------------->  longest match, coordination rules
   |
   v
section filter  ------------------->  keep assessment, drop indication
   |
   v
negation and context  ------------->  cue, pseudo-trigger, scope, termination
   |
   v
normalization  -------------------->  abbreviation, synonym, code
   |
   v
aggregation  ---------------------->  document, encounter, patient
```

The funnel is the right metaphor for recall. Each stage narrows the candidate set, and a candidate
lost early cannot be recovered later. A dictionary that lacks "HFpEF" loses those mentions before
the assertion stage ever runs. This is why candidate generation is measured by recall against a
gold span set, not by precision.

A second model is the **cue-scope-termination** triad for assertion.

```text
   ... [cue] .... [pseudo-trigger] .... [mention] .... [termination] ....
        |                                |                |
        +--- starts scope                 +--- target      +--- ends scope
```

Negation is not a property of a word. It is a property of a span within a scope. The cue opens the
scope, the termination closes it, and the pseudo-trigger cancels a cue that looks like negation but
is not. Every rule-based assertion system implements some version of this triad, and every failure
can be traced to one of the three: a missing cue, a missed pseudo-trigger, or a wrong boundary.

A third model is the **precision-recall exchange rate** at each filter. Adding a section filter
raises precision and lowers recall. Adding a negation filter raises precision on the "active"
label and can lower recall if it over-fires. Adding a synonym raises recall and can lower precision
if the synonym is ambiguous. The exchange rate is measurable on a sample, and it is the basis for
choosing a configuration.

| Filter | Effect on recall | Effect on precision | Typical cost |
|---|---|---|---|
| Add synonym | Up | Down if ambiguous | Lexicon curation |
| Longest match | Down slightly | Up | Coordination errors |
| Section restriction | Down | Up | Misses narrative evidence |
| Negation filter | Down on positive label | Up | Over-negation |
| Abbreviation expansion | Up | Down if ambiguous | Context rules |
| Template removal | Down | Up | Removes filled slots |

A fourth model is the **lexicon as a versioned artifact**. The dictionary is not a static resource.
Local abbreviations change, new drugs appear, and templates introduce new phrases. A lexicon
without an owner and a change log becomes an unmeasured source of drift. The model below treats the
lexicon as code.

```text
terminology release --> local additions --> reviewed --> versioned --> regression test
        |                    |                  |            |              |
   UMLS/RxNorm         site abbreviations   clinical lead   tag + date   fixed sample
```

The regression test is the part teams skip. A fixed sample of documents with adjudicated labels,
run on every lexicon or rule change, converts a silent behavior change into a visible metric. It is
the rule-based equivalent of a model evaluation set.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Rule-based system | Logic expressed as patterns and conditions | Called "deterministic" when it uses dictionaries that change | Rule set |
| Dictionary-based | Matching against a curated term list | Called NER | Term list |
| Regex | A regular expression pattern | Used for tasks that need syntax | Pattern |
| Lexicon | A term list with metadata | Confused with a terminology | Entry |
| Gazeetter | A list of terms used for matching | Misspelled as gazetteer in code | Entry |
| Longest match | Prefer the longest overlapping term | Assumed always correct | Span |
| Coordination | Conjunction of terms sharing a modifier | Ignored in matching | Span |
| Negation cue | A phrase that opens a negation scope | Treated as a boolean keyword | Phrase |
| Pseudo-trigger | A phrase containing a cue but not negating | Omitted from simple systems | Phrase |
| Termination term | A phrase that closes a scope | Confused with sentence boundary | Phrase |
| Scope | The text region a cue applies to | Assumed to be the sentence | Character range |
| Window | A fixed token distance for scope | Confused with scope | Tokens |
| Assertion | Status assigned to a mention | Called negation | Label |
| Experiencer | Who the assertion applies to | Ignored in many systems | Label |
| Temporality | When the event occurred | Reduced to past or present | Label |
| Abbreviation expansion | Mapping a short form to a long form | Applied globally | Entry |
| Ambiguous abbreviation | A short form with multiple expansions | Expanded by first match | Entry |
| Section filter | Restricting extraction to sections | Applied without validation | Rule |
| Candidate generation | Producing possible mentions | Evaluated by precision | Span set |
| Baseline | The simplest system that could work | Skipped | System |
| Hybrid system | Rules plus a learned component | Called rule-based | System |
| CRF | Conditional random field, a sequence labeler | Called a classifier | Model |
| SVM | Support vector machine | Called a neural model | Model |
| Feature | An input to a classical model | Confused with a token | Feature |
| Regularization | Penalty on model complexity | Ignored in small-data settings | Hyperparameter |
| Rule regression test | Fixed labeled sample run on every change | Omitted | Test set |
| Over-negation | Negating a mention that is affirmed | Counted as recall gain | Error |
| Under-negation | Missing a negation | Counted as precision loss | Error |
| Section header lexicon | Terms used to detect section boundaries | Treated as complete | Entry |
| Label precedence | Order in which assertion labels override | Left implicit | Rule |
| Candidate recall | Share of gold mentions produced by generation | Confused with final recall | Ratio |
| Rule interaction | Combined effect of rules on the same span | Untested | Test case |

Two definitions need emphasis.

**Scope versus window.** A window is a fixed number of tokens. A scope is a linguistic region
defined by cues and boundaries. NegEx used a five-token window, which is a window approximating a
scope. ConText used termination terms and sentence end, which is a scope approximation. The
distinction matters because a window's errors are distance-dependent and a scope's errors are
boundary-dependent. A long sentence with a distant cue will be missed by a window and caught by a
scope; a sentence with an internal "but" will be over-scoped by a naive sentence rule and correctly
terminated by a termination-term rule.

**Pseudo-trigger versus termination.** A pseudo-trigger cancels a cue ("no increase" is not a
negation of the thing that follows). A termination ends a scope ("but" ends the negated region). Both
prevent false negation, but they act at different points: the pseudo-trigger acts on the cue, the
termination acts on the scope. Conflating them produces rules that either over-negate or
under-negate depending on word order.

## 5. The mechanics, step by step

This section is the construction procedure for a rule-based extraction and assertion pipeline. It
uses medication extraction as the running example, but the steps generalize to problems,
procedures, and findings.

### Step 1. Define the target and the assertion schema

**Input:** The decision and the clinical concept of interest.

**Output:** A target definition with inclusion and exclusion criteria, and an assertion schema with
named labels.

**Rationale:** The target definition determines the lexicon. "Medication" can mean active
medications, historical medications, allergies, or any drug string. Each requires a different
section policy and a different assertion schema.

**Branch point:** The assertion schema has three common forms.

| Schema | Labels | Use | Cost |
|---|---|---|---|
| Binary | Affirmed, negated | Simple extraction | Misses uncertainty and history |
| NegEx-style | Affirmed, negated, possible | General clinical text | Requires uncertainty cues |
| ConText-style | Affirmed, negated, possible, historical, hypothetical, family | Phenotyping and cohort work | Requires scope and section rules |

### Step 2. Build the terminology and the local lexicon

**Input:** A standard terminology and a sample of local documents.

**Output:** A term list with synonyms, abbreviations, and metadata.

**Rationale:** Standard terminologies cover the general vocabulary but not local shorthand. A sample
of documents supplies the shorthand. Both are needed, and the local additions must be reviewed.

**Branch point:** Lexicon sources have different licenses and coverage.

| Source | Coverage | License consideration |
|---|---|---|
| RxNorm | Medication names and ingredients | Free with UMLS license terms |
| SNOMED CT | Clinical findings and procedures | Affiliate license required |
| UMLS | Metathesaurus across vocabularies | License required; restrictions on redistribution |
| Local formulary | Site-specific products and names | Institutional |
| Manual curation | Abbreviations and shorthand | Institutional; needs review |

### Step 3. Normalize the text

**Input:** Raw text.

**Output:** Normalized text plus a mapping to raw offsets.

**Rationale:** Case, Unicode, and whitespace variation break matching. Normalization must be
reversible for span reporting.

**Branch point:** Aggressive normalization destroys distinctions. Lowercasing loses gene and drug
case conventions; stripping punctuation merges "b.i.d." into "bid" but also merges list items.
Choose normalization per target and record it.

### Step 4. Segment sections and sentences

**Input:** Normalized text.

**Output:** Section spans and sentence spans.

**Rationale:** Section context drives assertion and filtering. Sentence boundaries define the
default scope. Both are preprocessing tasks with direct effects on the rule layer.

**Branch point:** If sentence splitting is unreliable, use clause boundaries or fixed windows and
document the choice. A wrong sentence boundary is worse than a coarse unit for negation.

### Step 5. Generate candidates

**Input:** Sections and sentences.

**Output:** Candidate spans with a provisional type.

**Rationale:** Candidate generation should maximize recall. Precision is the job of later filters.
A candidate generator that is too strict loses mentions that no later stage can recover.

**Branch point:** Generation methods have different recall profiles.

| Method | Recall | Precision | Best for |
|---|---|---|---|
| Exact dictionary | Moderate | High | Known vocabulary |
| Case-insensitive dictionary | Higher | Moderate | Note text |
| Stemmed dictionary | Higher | Lower | Morphological variation |
| Regex pattern | Targeted | Variable | Doses, dates, IDs |
| Fuzzy match | High | Low | OCR and misspellings |

### Step 6. Resolve overlaps

**Input:** Candidate spans.

**Output:** One span per mention region.

**Rationale:** "Insulin glargine" and "insulin" overlap. "Congestive heart failure" and "heart
failure" overlap. Reporting both double-counts and confuses the assertion stage.

**Branch point:** Overlap resolution has four common rules.

| Rule | Behaviour | Failure |
|---|---|---|
| Longest match | Keep the longest span | Can cross coordination boundaries |
| Leftmost-longest | Prefer leftmost, then longest | Misses nested terms |
| Priority by type | Keep the type with higher priority | Type errors propagate |
| Keep all, dedupe later | Retain overlaps for downstream | Double counting |

### Step 7. Apply section filters

**Input:** Spans with section labels.

**Output:** Spans retained or dropped by section policy.

**Rationale:** Medication mentions in an "Indication" field or a "Home medications" list have
different status from mentions in the plan. Section policy is a cheap precision gain.

**Branch point:** Section filtering is a recall-precision trade. A strict policy that keeps only the
plan will miss medications documented in the hospital course. A permissive policy keeps indication
text. The choice should be validated on a sample, not assumed.

### Step 8. Detect negation cues

**Input:** Span plus sentence.

**Output:** Candidate cues and their positions.

**Rationale:** Cues are the entry point to assertion. Missing a cue is an under-negation error, and
spurious cues cause over-negation.

**Branch point:** Cue matching should be token-aware. Matching "no" as a substring finds it inside
"not" and "north." Matching with word boundaries avoids that but still finds "no" in "no change,"
which is a pseudo-trigger.

### Step 9. Filter pseudo-triggers

**Input:** Candidate cues.

**Output:** Cues that remain after pseudo-trigger removal.

**Rationale:** "No increase in metformin dose" is not a negation of metformin. "Gram-negative" is
not a negation of a finding. Without pseudo-trigger filtering, the system over-negates and loses
true positives.

**Branch point:** Pseudo-triggers can be handled before scope detection or after. Handling them
first is simpler and matches ConText's design; handling them after allows the scope to be computed
once. The order matters when a pseudo-trigger appears after the cue and before the mention.

### Step 10. Determine scope and termination

**Input:** Cues and sentence text.

**Output:** A scope interval per cue.

**Rationale:** Scope determines which mentions are affected. Window-based scope is simple but
distance-dependent. Termination-based scope handles long sentences but needs a termination lexicon.

**Branch point:** The choice has a measurable effect.

| Scope method | Strength | Failure | Typical fix |
|---|---|---|---|
| Five-token window | Simple, fast | Misses distant mentions | Increase window; accept more over-negation |
| Sentence | Simple, language-independent | Over-negates after "but" | Add termination terms |
| Termination-based | Handles coordination and contrast | Needs a curated list | Review term list on errors |
| Section-level | Handles templated headings | Cross-section leakage | Combine with sentence scope |

### Step 11. Assign assertion and experiencer

**Input:** Span plus scope plus context.

**Output:** Assertion label and experiencer label.

**Rationale:** A span inside a negation scope is negated. A span inside a family-history scope is
attributed to a family member. The two are different labels and must not be collapsed.

**Branch point:** If a span is inside multiple scopes, precedence rules are needed. A common order
is: family or other experiencer overrides negation; hypothetical overrides historical; negation
overrides affirmed. The order should be written down and tested.

### Step 12. Resolve abbreviations

**Input:** Spans and abbreviations.

**Output:** Expanded forms or a decision to leave ambiguous.

**Rationale:** Abbreviations are the largest source of concept errors in many corpora. "MS" has at
least four common expansions. A global expansion rule guarantees errors.

**Branch point:** Abbreviation handling has four strategies.

| Strategy | Rule | Precision | Recall |
|---|---|---|---|
| Global expansion | Always map to the most common sense | Low | High |
| Section-specific | Map per section | Medium | Medium |
| Local definition detection | Use an in-document definition | High | Low |
| Context classifier | Predict the sense from context | High | Medium |

### Step 13. Aggregate to the decision unit

**Input:** Asserted spans.

**Output:** Document, encounter, or patient labels.

**Rationale:** Aggregation converts extraction into a decision. The rule must state whether it
counts mentions, documents, or distinct evidence, and whether negated mentions are excluded.

**Branch point:** The aggregation rule should be chosen before evaluation, not after seeing which
rule gives the best number.

### Step 14. Evaluate and error-analyze

**Input:** System output and a reference standard.

**Output:** Span and assertion metrics, plus an error taxonomy with counts.

**Rationale:** A single F1 does not guide improvement. An error taxonomy by mechanism does.

**Branch point:** If span recall is low, the problem is candidate generation or lexicon. If span
recall is high and assertion accuracy is low, the problem is scope or cues. If both are high and
patient-level PPV is low, the problem is aggregation or prevalence.

### Step 15. Version, test, and monitor

**Input:** Rules, lexicon, and a fixed evaluation sample.

**Output:** A versioned artifact, a regression test, and a monitoring plan.

**Rationale:** Rules change. A lexicon addition can alter negation behavior if it changes
tokenization or overlap resolution. The regression test makes the change visible.

**Branch point:** If the regression sample is small, report confidence intervals. A five-document
sample cannot detect a meaningful change.

### Step 16. Attribute errors to a mechanism

**Input:** System output, reference standard, and the intermediate stage outputs.

**Output:** An error table with counts by mechanism and an owner for each.

**Rationale:** "F1 is 0.87" does not tell a team what to fix. The stage at which an error enters
determines whether the fix is a lexicon entry, a scope rule, a section policy, or a model. Without
stage-level outputs, every error looks like a model error.

**Branch point:** The attribution table below maps symptoms to causes. It is the most useful
artifact for maintaining a rule system.

| Symptom | Most likely stage | Diagnostic | Fix |
|---|---|---|---|
| Mention missed entirely | Candidate generation | Check if the string is in the lexicon | Add synonym or abbreviation |
| Mention found but wrong type | Lexicon metadata | Inspect the entry | Correct the type |
| Two overlapping mentions counted | Overlap resolution | Count spans per region | Longest match; dedupe |
| Mention found in wrong section | Segmentation | Check header detection | Expand header lexicon |
| Affirmed mention marked negated | Scope or pseudo-trigger | Per-cue false-negation table | Add pseudo-trigger or termination |
| Negated mention marked affirmed | Cue list or scope | Per-construction error review | Add cue; extend scope |
| Family mention counted as patient | Experiencer | Check experiencer cues | Add attribution rule |
| Abbreviation mapped wrongly | Normalization | Per-abbreviation accuracy | Context or section rule |
| Counts differ from chart review | Aggregation | Compare rules on same sample | Change aggregation policy |
| Performance fell after update | Any stage | Diff outputs on regression set | Roll back; version pin |

### Rule ordering and precedence

Rules interact, and the order of application is part of the specification. A typical order is:
normalize, segment, generate candidates, resolve overlaps, apply section policy, apply negation,
apply experiencer, normalize concepts, aggregate. Changing the order changes the output.

Consider a family-history mention inside a negated clause: "Mother does not have diabetes."
If negation runs before experiencer, the mention is negated and then attributed to the mother,
producing a negated family-history fact. If experiencer runs first, the mention is attributed to the
mother and then negated within that attribution, producing the same fact. The result is the same in
this case, but not in "Mother has diabetes, patient does not." Here the second clause's negation
must not reach the first clause's mention. Scope boundaries, not rule order, determine the result.

A more consequential ordering issue is section policy versus negation. If section filtering removes
the family-history section before negation runs, the family mention disappears and cannot be
misattributed. If negation runs first, it may mark the family mention as affirmed, and the section
filter then removes it. The output is the same, but the intermediate audit trail differs. For
debugging, run negation before section filtering and log both, so that a change in section policy
does not silently change assertion statistics.

Precedence among labels should be written explicitly. A common order is: other experiencer, then
hypothetical, then historical, then negated, then affirmed. The order reflects clinical priority:
a family-history mention is not a patient fact even if affirmed, and a hypothetical mention is not
a historical fact even if the tense is past. The order must be tested on cases where labels
conflict, and the test cases should be kept in the regression sample.

### Edge cases that decide rule quality

**Negation scope across clauses.** "No fever, but continued tachycardia" contains a negation that
should not extend past "but." A sentence-scope rule negates tachycardia incorrectly. A termination
list containing "but" fixes it. The reverse error occurs when "but" is part of a phrase that does
not contrast, such as "all but one."

**Hypotheticals.** "If she develops AF, start anticoagulation" contains a disease and a treatment
inside a conditional. A negation-only system treats both as affirmed. A ConText-style system marks
them hypothetical. The distinction matters for any alert or cohort that counts active problems.

**Family history.** "Mother had breast cancer" contains a mention that is affirmed and attributed to
another person. A system that assigns only negation will mark it affirmed, which is correct for the
mention and wrong for the patient fact. Experiencer is a separate label.

**Copy-forward duplicates.** A copied sentence produces repeated mentions. Rule-based systems count
them each time. The aggregation rule must deduplicate or the counts become note-formatting
artifacts.

**Template boilerplate.** A templated "denies chest pain" in every review of systems is a negation
that carries little information. It is not wrong to extract it, but it is wrong to treat its
presence as evidence about the patient. Template detection belongs upstream.

**Abbreviation ambiguity.** "PT" can be prothrombin time, physical therapy, or patient. "MS" can be
mitral stenosis, multiple sclerosis, morphine sulfate, or mental status. A local lexicon with
section and context rules reduces errors; a global dictionary increases them.

**Section boundaries.** A section header misread as body text can place a medication list inside an
assessment, changing the assertion. A header read correctly but with the wrong label can apply the
wrong section policy.

**Unicode and OCR noise.** Curly apostrophes, non-breaking spaces, and OCR substitutions break
exact regex matching. Normalize Unicode and measure character error rate for scanned content. A
regex tuned to "patient's" will miss "patient?s" from OCR.

## 6. Worked example

A pharmacy analytics team is building a medication list from 250 discharge summaries. The decision
is whether a patient is on a specific chronic medication at discharge, for a medication
reconciliation dashboard. The unit is the patient, but the intermediate evaluation is at the mention
and document levels.

### Reference standard

Two pharmacists annotate every medication mention in the 250 summaries and assign an assertion
status. Adjudication resolves disagreements.

| Reference statistic | Count |
|---|---:|
| Documents | 250 |
| Gold medication mentions | 1,200 |
| Gold mentions negated | 380 |
| Gold mentions affirmed | 820 |
| Patients | 250 |

The 380 negated mentions are mostly discontinuations ("metformin held"), allergies ("no
metformin"), and historical references ("was on metformin"). The distinction between held and
allergic matters clinically but not for the dashboard's binary active-medication flag, so both are
treated as not active.

### Baseline: exact dictionary, document order

The first system matches the formulary dictionary case-insensitively, keeps all overlapping
matches, and ignores sections and assertion.

| Span result | Count |
|---|---:|
| True positive | 840 |
| False positive | 260 |
| False negative | 360 |

Precision = 840 / 1,100 = 0.764.
Recall = 840 / 1,200 = 0.700.
F1 = 2 × 0.764 × 0.700 / 1.464 = 1.070 / 1.464 = 0.731.

The baseline is weak. The 360 false negatives are mostly abbreviations and brand names absent from
the formulary list. The 260 false positives are mostly overlapping matches counted twice and
medication names inside indication text.

### Rule-based system: longest match, section filter, abbreviation table

The second system adds longest-match resolution, a section filter that keeps the hospital course,
discharge medications, and plan, and a local abbreviation table derived from 30 sample documents.

| Span result | Count |
|---|---:|
| True positive | 1,020 |
| False positive | 120 |
| False negative | 180 |

Precision = 1,020 / 1,140 = 0.895.
Recall = 1,020 / 1,200 = 0.850.
F1 = 2 × 0.895 × 0.850 / 1.745 = 1.522 / 1.745 = 0.872.

The improvement is 14.1 F1 points, from 0.731 to 0.872. The gain comes from three sources: 108
additional true positives from abbreviations and brand names, 84 fewer false positives from overlap
resolution, and 68 fewer false positives from the section filter. The decomposition is approximate
because the sources interact; the team reports it as an attribution, not an experiment.

### Assertion layer

Among the 1,020 true positive spans, 310 are negated in the reference standard and 710 are affirmed.
The negation module uses 143 negation cues, 17 pseudo-triggers, and 12 termination classes,
following the ConText design. On the 1,020 spans:

| | Gold negated | Gold affirmed |
|---|---:|---:|
| System negated | 250 | 30 |
| System affirmed | 60 | 680 |

Assertion accuracy = (250 + 680) / 1,020 = 930 / 1,020 = 0.912.
Negation precision = 250 / 280 = 0.893.
Negation recall = 250 / 310 = 0.806.

The 60 missed negations are split into three causes:

| Cause | Count | Example pattern |
|---|---:|---|
| Cue beyond the clause boundary | 26 | Negation in a prior sentence with no explicit cue repeat |
| Passive construction | 18 | "metformin was discontinued" |
| Local phrasing absent from the cue list | 16 | "off metformin", "metformin not restarted" |

The 30 false negations are split into two causes:

| Cause | Count | Example pattern |
|---|---:|---|
| Pseudo-trigger missing | 19 | "no change in metformin dose" |
| Family-history cue misapplied | 11 | "mother not on metformin" attributed to patient |

Both error sets are small enough to fix with lexicon changes, and both are the kind of error that a
transformer might handle without explicit rules. The team's decision to keep rules is based on the
audit trail, not on accuracy alone.

### Patient-level effect

The dashboard flag is "patient is on medication X at discharge." A patient-level positive requires
at least one affirmed mention in a discharge medication section.

| | Reference positive | Reference negative |
|---|---:|---:|
| System positive before assertion | 410 | 90 |
| System negative | 70 | 180 |

Sensitivity = 410 / 480 = 0.854.
Specificity = 180 / 270 = 0.667.
PPV = 410 / 500 = 0.820.
Accuracy = 590 / 750 = 0.787.

Wait: the table sums to 750 patients, but the corpus has 250. The team evaluated at the
medication-patient pair level, because each patient can be on several medications. There are 750
patient-medication pairs, 480 positive in the reference standard. The unit is the pair, and the
report must say so. This is a deliberate illustration of the module's rule that every metric carries
its unit; "patient-level" here means a patient-medication pair, not a patient.

After removing mentions the assertion layer labels negated, assuming the 30 false negations and the
60 missed negations distribute as observed:

| | Reference positive | Reference negative |
|---|---:|---:|
| System positive after assertion | 410 | 56 |
| System negative | 70 | 214 |

Sensitivity = 410 / 480 = 0.854 (unchanged, because the removed mentions were all system-negated
and none were the sole positive evidence for a pair).
Specificity = 214 / 270 = 0.793.
PPV = 410 / 466 = 0.880.
Accuracy = 624 / 750 = 0.832.

The assertion layer improves PPV by 6.0 points and specificity by 12.6 points, at no recall cost in
this sample. The result depends on the assumption that no pair's only evidence was a system-negated
mention. If that assumption fails, recall falls. The team checks it by auditing the 34 removed
pairs and finds 0 with no other evidence, which supports the assumption for this corpus.

### Two diagnostics that reveal a broken pipeline

**Diagnostic 1: over-negation concentrated in a few phrases.** Compute the false-negation rate by
cue. In this system:

| Cue or pattern | Spans affected | False negations | Rate |
|---|---:|---:|---:|
| "no change" | 41 | 12 | 0.293 |
| "no increase" | 28 | 7 | 0.250 |
| "not cause" | 9 | 0 | 0.000 |
| Other pseudo-triggers | 22 | 0 | 0.000 |
| All cues | 310 | 30 | 0.097 |

Two phrases account for 19 of 30 false negations. A pipeline with a missing pseudo-trigger shows a
spike in a small number of phrases, not a uniform error rate. The diagnostic is the per-cue table.
If the table does not exist, the team cannot see the spike.

**Diagnostic 2: section distribution of false positives.** Of the 120 span false positives, 74 come
from one section, the "Indication" field of medication orders embedded in the note. That section
contains drug names because it names the reason for a drug, not because the patient takes it.
A pipeline that reads the whole document shows a section-concentrated false-positive rate. The
diagnostic is the false-positive rate by section; a rate above 50 percent in a section that should
not contain the target is a section-boundary failure, not a lexicon failure.

### Vary one assumption: replace sentence scope with a five-token window

Suppose the team replaces termination-based scope with NegEx's five-token window. Among the 250
correctly negated spans, 46 have the cue more than five tokens from the mention. Those become
missed negations.

| Metric | Termination scope | Five-token window |
|---|---:|---:|
| Correctly negated | 250 | 204 |
| Missed negations | 60 | 106 |
| False negations | 30 | 30 |
| Assertion accuracy | 0.912 | 0.867 |
| Negation precision | 0.893 | 0.872 |
| Negation recall | 0.806 | 0.658 |

Assertion accuracy with the window = (204 + 680) / 1,020 = 884 / 1,020 = 0.867.
Negation precision = 204 / 234 = 0.872. Recall = 204 / 310 = 0.658.

At the patient-medication level, suppose 14 of the 34 pairs whose false positive depended on a
negation involved a long-range cue. With the window, those 14 are not removed, so the false
positives fall from 34 to 20 rather than to 34.

| | Reference positive | Reference negative |
|---|---:|---:|
| System positive with window | 410 | 70 |
| System negative | 70 | 200 |

PPV = 410 / 480 = 0.854, against 0.880 with termination scope. The conclusion changes by 2.6 PPV
points. The direction is unchanged, but the magnitude matters for a dashboard where a 2.6-point
difference changes the review workload by about 14 patients per 250. The team keeps termination
scope and documents the window comparison as a sensitivity analysis.

## 7. Code, query, or calculation

The following module implements the core of a NegEx-style system: longest-match dictionary lookup,
termination-based left scope, pseudo-trigger filtering, and assertion metrics. It uses only the
standard library and synthetic sentences.

```python
import re

SENTENCES = [
    "She denies chest pain but takes metformin daily.",
    "No metformin given today.",
    "Continue insulin glargine at bedtime.",
    "There is no increase in lisinopril dose.",
    "Negative for hypoglycemia on this regimen.",
]

TERMS = {"metformin": "DRUG", "insulin glargine": "DRUG", "lisinopril": "DRUG"}
NEGATION = ["no", "denies", "without", "negative for", "not"]
PSEUDO = ["no increase", "no change", "not cause", "gram negative"]
TERMINATION = ["but", "however", "though", "except", "although"]


def find_terms(sentence):
    """Longest-match dictionary lookup with character offsets."""
    hits = []
    occupied = []
    for term in sorted(TERMS, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", sentence, re.I):
            if any(m.start() < e and s < m.end() for s, e in occupied):
                continue
            hits.append((m.start(), m.end(), term, TERMS[term]))
            occupied.append((m.start(), m.end()))
    return sorted(hits)


def left_context(sentence, start):
    """Text from the last termination term to the match."""
    window = sentence[:start].lower()
    cut = 0
    for t in TERMINATION:
        pos = window.rfind(t)
        if pos > cut:
            cut = pos + len(t)
    return window[cut:]


def is_negated(sentence, start):
    ctx = left_context(sentence, start)
    for p in PSEUDO:
        if p in ctx:
            ctx = ctx.replace(p, " ")
    for n in NEGATION:
        if re.search(r"\b" + re.escape(n) + r"\b", ctx):
            return True, n
    return False, None


for s in SENTENCES:
    for start, end, term, typ in find_terms(s):
        neg, cue = is_negated(s, start)
        print(f"{term:16s} {typ:5s} negated={str(neg):5s} cue={cue}")

gold_neg, gold_aff = 310, 710
pred_neg_correct, pred_neg_wrong = 250, 30
missed_neg, correct_aff = 60, 680
acc = (pred_neg_correct + correct_aff) / (gold_neg + gold_aff)
neg_p = pred_neg_correct / (pred_neg_correct + pred_neg_wrong)
neg_r = pred_neg_correct / gold_neg
print("assertion accuracy:", round(acc, 3))
print("negation precision:", round(neg_p, 3), "recall:", round(neg_r, 3))
```

Expected output:

```text
metformin        DRUG  negated=False cue=None
metformin        DRUG  negated=True  cue=no
insulin glargine DRUG  negated=False cue=None
lisinopril       DRUG  negated=False cue=None
assertion accuracy: 0.912
negation precision: 0.893 recall: 0.806
```

The first line is the termination test: "denies" appears before "but," and the mention appears after
it, so the negation scope does not reach the medication. The second line is a direct negation. The
fourth line is the pseudo-trigger test: "no increase" neutralizes the cue, and lisinopril is
correctly left affirmed. The fifth sentence contains no dictionary term, so it produces no output;
that is a recall failure of the lexicon, not of the assertion module, and the distinction matters
for error attribution.

A second block computes the precision-recall exchange for the two scope methods:

```python
def prf(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)


configs = {
    "termination scope": (250, 30, 60),
    "five-token window": (204, 30, 106),
}
for name, (tp, fp, fn) in configs.items():
    p, r, f1 = prf(tp, fp, fn)
    acc = (tp + 680) / 1020
    print(f"{name}: neg_p={p:.3f} neg_r={r:.3f} acc={acc:.3f}")

# Patient-medication pairs
for name, fp_pairs in {"termination scope": 56, "five-token window": 70}.items():
    ppv = 410 / (410 + fp_pairs)
    print(f"{name}: pair PPV={ppv:.3f}")
```

Expected output:

```text
termination scope: neg_p=0.893 neg_r=0.806 acc=0.912
five-token window: neg_p=0.872 neg_r=0.658 acc=0.867
termination scope: pair PPV=0.880
five-token window: pair PPV=0.854
```

The comparison isolates one design choice. Termination scope costs a slightly more complex cue list
and buys 14.8 points of negation recall, 4.5 points of assertion accuracy, and 2.6 points of
patient-level PPV.

## 8. Data, standards, and regulatory context

Classical methods depend on resources whose licenses and versions are part of the method. A rule
system that cites a terminology without a version is not reproducible.

**UMLS.** The Unified Medical Language System Metathesaurus is distributed by the U.S. National
Library of Medicine under a license agreement. It requires an account, restricts redistribution,
and requires attribution. It includes SNOMED CT, RxNorm, LOINC, and many other vocabularies, each
with its own terms. Using UMLS to build a lexicon commits the project to those terms, including
restrictions on sharing the lexicon.

**SNOMED CT.** SNOMED CT is licensed through the UMLS Metathesaurus in the United States and
through National Release Centers elsewhere. An affiliate license is required for some uses. The
terminology has frequent releases, and concept identifiers can be inactivated or replaced. A
concept mapper that pins no version will produce different results after an update. The
specification should name the release, for example the January 2026 international edition.

**RxNorm.** RxNorm provides normalized names for clinical drugs and is available through UMLS. It
has monthly releases and a prescribable-content subset. Medication extraction that maps to RxNorm
should state whether it maps to the ingredient, the clinical drug, or the branded product, because
the three are different concepts with different granularity.

**LOINC.** LOINC identifies laboratory observations and document types. It is free to use under
the LOINC license, which requires attribution and prohibits using the codes as a basis for a
competing terminology. A section or document classifier that maps note types to LOINC should state
the version.

**cTAKES.** The Mayo clinical Text Analysis and Knowledge Extraction System is an open-source
Apache-licensed pipeline that includes sentence splitting, tokenization, section identification,
dictionary lookup, negation, and concept mapping. It is a reference implementation of many of the
steps in this topic. Its existence is useful for two reasons: it provides a baseline, and it shows
that the classical pipeline is a designed system with components, not a single regex.

**Rule systems as software.** If a rule-based system informs diagnosis or treatment for a specific
patient, it may be a medical device regardless of whether it uses machine learning. The FDA's
January 2025 draft guidance on AI-enabled device software functions applies to devices containing
AI-enabled functions; a purely deterministic rule system may fall outside that guidance's scope but
still be a device under general software provisions. The regulatory question is about intended use
and risk, not about the technique. Change control, versioning, and validation apply to rule changes
in the same way they apply to model updates.

**Data use agreements and lexicon sharing.** A local lexicon built from a site's documents may
contain PHI or site-identifying information. Sharing it outside the institution can breach the data
use agreement even if the terms are generic drug names. The safe practice is to share the rule
logic and a synthetic test set, not the curated lexicon.

**Reproducibility.** A rule-based system is reproducible only if the text normalization, the
lexicon version, the rule version, and the evaluation sample are fixed. The system's determinism is
not enough; the inputs change.

**FHIR output and assertion semantics.** A rule system that writes its output to FHIR must choose
a resource. Writing a medication mention to `MedicationStatement` asserts a medication the patient
is taking; writing it to `MedicationRequest` asserts an order. A negated mention written to either
is a semantic error. The rule layer must therefore expose assertion status at the output boundary,
not only inside the pipeline. This is the same issue discussed in Topic 03, and it is more common in
rule systems because the rules often write directly to a table.

**OMOP `NOTE_NLP`.** The OMOP `NOTE_NLP` table has fields for the extracted term, the source span,
the concept, the temporal relationship, and the assertion. A rule pipeline can populate it
directly, which makes the output comparable across sites at the schema level. The values in the
assertion field are still local unless a controlled vocabulary is used. Sites that share a schema
but not an assertion vocabulary produce data that looks comparable and is not.

**Clinical decision support certification.** If the rule system is embedded in certified health IT
and presents a recommendation to a clinician, the ONC HTI-1 decision support intervention criterion
may apply, including source attribute transparency for predictive DSIs. A deterministic rule that
is not predictive may fall outside the predictive DSI definition, but the general DSI criterion
still governs intervention risk management. The determination depends on function, not on whether
the logic is a rule or a model.

**Audit logs.** Regulated environments often require an audit log of who changed a rule, when, and
why. A git history is necessary and not sufficient: the log should connect a rule version to the
validation evidence that supported it. Without that link, a reviewer cannot tell whether a rule
change was validated or merely deployed.

> **Note:** "Deterministic" does not mean "stable." A rule pipeline run on next month's notes will
> produce different output because the notes changed, the lexicon changed, or the terminology
> release changed. Version all four.

## 9. Industry practice and case evidence

**NegEx.** Chapman et al. (2001) evaluated NegEx on 1,000 sentences from de-identified intensive
care discharge summaries containing 1,235 annotated findings and diseases. NegEx used 35 negation
phrases, pseudo-negation filters, and a five-token scope. It achieved sensitivity 77.8 percent,
specificity 94.5 percent, PPV 84.5 percent, and NPV 91.7 percent. A simpler baseline had higher
sensitivity (88.3 percent) and lower specificity (85.3 percent). The authors reported that "no,"
"without," and "no evidence of" accounted for 82 percent of correctly identified pertinent
negatives, and that weak phrases such as "versus" had poor PPV. These are the paper's reported
figures on its corpus.

**ConText.** Harkema et al. (2009) extended the approach to negation, temporality, and experiencer,
using 143 negation triggers, 10 historical triggers, 11 hypothetical triggers, and 26
other-experiencer triggers, plus 12 termination-term classes. On 240 de-identified reports across
six report types, test-set F-scores were 0.93 for negation, 0.76 for historical, 0.86 for
hypothetical, and 1.0 for experiencer on only five non-default cases. The historical result is the
important one: the trigger-based method transferred less well, and the authors attributed the gap to
lexical ambiguity of "history," section dependence, and the need for medical knowledge.

**cTAKES.** Savova et al. (2010) described cTAKES as a modular pipeline for clinical text, with
components for sentence detection, tokenization, section identification, dictionary lookup,
negation, and concept mapping. The architecture is evidence that classical methods were engineered
as systems with interfaces, not as ad hoc scripts. Inference: the component boundaries in cTAKES
anticipate the task taxonomy in Topic 03.

**n2c2 2018.** The cohort-selection track reported a best micro F1 of 0.91 from a rule-based system,
with four rule-based and six hybrid systems in the top ten and no purely machine-learning system in
that group. The mean was 0.799 and the median 0.8227. The result is consistent with the task's
structure: 13 heterogeneous criteria, explicit evidence, and a small training set.

**i2b2 2010 assertion.** The assertion task supplied gold concepts and asked systems to classify
them. The best system reached F1 0.936, and the top systems combined support vector machines with
contextual features and negation dictionaries, often with rule-based scope detection. The task's
design, giving concepts as input, isolated assertion from extraction and made the result
interpretable.

**MedCAT.** MedCAT performs dictionary-based concept extraction and linking with a learned
disambiguation layer. Its design is a hybrid: the dictionary provides candidate concepts and the
model selects among them. The pattern is common in production because it preserves the auditability
of the dictionary while handling ambiguity with a model. Inference: the boundary between classical
and learned methods is often inside a component, not between systems.

**What the evidence does not show.** None of these results establishes that rules outperform
transformers on a new task. They establish that rules are competitive on tasks with explicit
evidence and small label sets, and that they provide a baseline that must be reported. A team that
skips the baseline cannot tell whether its model earned its complexity.

**Toolkits and the rule-plus-model default.** Open-source clinical NLP toolkits commonly ship both
a rule layer and a statistical layer. scispaCy provides rule-based and dictionary-based components
alongside biomedical language models, and its entity linking uses a candidate generator plus a
ranker. MedCAT uses a dictionary for candidate concepts and a learned disambiguator. cTAKES uses
rule-based components throughout. The prevalence of the hybrid pattern is evidence that production
teams rarely choose one method exclusively. Inference: the interesting design question is not
"rules or models" but "which component should be which."

**Operational monitoring of rule performance.** A rule system can be monitored without labels by
tracking the distribution of matched terms, the rate of negation, the share of documents with no
match, and the frequency of unmatched candidate strings. A sudden change in any of these signals a
lexicon gap or a template change. The cheapest monitor is the unmatched-term list: strings that
look like the target but are not in the lexicon are a direct queue for curation. This is an
operational practice inferred from tooling design, not a published evaluation.

**Rule maintenance as a cost center.** Organizations that run rule systems at scale report that
maintenance, not initial development, dominates total cost. The drivers are terminology updates,
new templates, new services, and local shorthand. A learned model shifts some maintenance to
retraining, which has its own cost and its own failure modes. The trade-off is real, and it should
be estimated before choosing an approach rather than discovered after deployment. This is an
inference from practice; published cost studies are scarce.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Lexicon misses local shorthand | Recall plateaus at 0.70 | Dictionary built from standard terms only | Missed mentions | Mine sample for abbreviations |
| Over-negation from pseudo-triggers | False negations cluster in a few phrases | Cue matched without pseudo-trigger filter | Lost true positives | Pseudo-trigger list; per-cue audit |
| Under-negation from window scope | Long sentences lose negation | Cue beyond token window | False positives in cohort | Termination-based scope |
| Section filter too strict | Recall drops after filter | Evidence in excluded section | Missed patients | Validate filter on sample |
| Section filter too loose | FPs concentrated in indication fields | Whole-document extraction | Wrong active medications | Section-aware rules |
| Overlap double counting | Mention counts exceed gold | All overlapping matches kept | Inflated counts | Longest match; dedupe |
| Abbreviation global expansion | "MS" always multiple sclerosis | No context disambiguation | Systematic concept errors | Context rules; local lexicon |
| Regex matches substrings | "no" matches "north" | Missing word boundaries | Spurious negation | Token-aware matching |
| Unicode mismatch | Rules fail on curly apostrophes | Raw text not normalized | Silent recall loss | Normalize; keep offset map |
| Lexicon drift | Performance falls after formulary update | No version pin | Irreproducible results | Version and regression test |
| Rule change unvalidated | Cohort size changes after a "small" fix | No regression sample | Undetected behavior change | Fixed test set; CI |
| Template text treated as evidence | "Denies chest pain" in 100% of notes | Boilerplate not detected | False negatives | Upstream template handling |
| Evaluation at wrong unit | Span F1 high, cohort PPV low | Metric substitution | Overstated capability | Evaluate at decision unit |
| No baseline | Model compared only to itself | Baseline skipped | Unknown marginal value | Report rule baseline |
| Rule precedence undocumented | Same input gives different labels after refactor | Order of negation and experiencer rules implicit | Irreproducible assertions | Write precedence order; test conflicts |
| Lexicon shared across sites | Local terms appear in another site's output | Data use agreement breach | Legal and privacy exposure | Share logic, not curated terms |
| Assertion status lost at output | FHIR resource has no status field | Rule writes only the concept | Negated mention becomes diagnosis | Carry status through to output |
| Regression sample too small | No change detected after a major edit | Underpowered test | Undetected behavior change | Size sample for smallest detectable difference |

The red flags for a rule-based system are specific. A rule description without a scope definition
is incomplete. A negation result without a pseudo-trigger list is suspect. A lexicon without a
version is not reproducible. An evaluation without an error taxonomy cannot guide maintenance.
Any two of these together mean the system is a demo, not a production component.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The lexicon covers the target vocabulary | Candidate generation depends on it | Local shorthand or new drugs | Recall audit on random sample |
| Negation cues are finite and listable | Rule approach depends on it | Novel phrasing or languages | Per-cue error analysis |
| Scope can be approximated by termination terms | Simpler than parsing | Complex coordination | Compare with manual scope annotation |
| Sections are detectable and stable | Section filter depends on it | Template changes | Header detection rate |
| Abbreviations can be resolved by context | Avoids global errors | Highly ambiguous short forms | Per-abbreviation accuracy |
| The aggregation rule matches the decision | Converts spans to labels | Rule counts duplicates | Compare alternative rules |
| The regression sample represents deployment | Detects drift | Sample from a different era | Stratify by note type and time |
| Terminology versions are pinned | Reproducibility | Silent release update | Check release in output metadata |
| Rules are deterministic given inputs | Auditability | Dictionary ordering effects | Run twice; diff |
| The baseline is strong enough to be informative | Model comparison | Weak baseline inflates gain | Tune baseline fairly |

The most important limit is linguistic. Rules encode patterns the authors anticipated. Clinical
language is productive: clinicians invent shorthand, omit subjects, and write in fragments. A rule
system's recall is bounded by the imagination of its authors and the coverage of its lexicon. That
bound is not a flaw to be eliminated; it is a property to be measured and stated.

The second limit is maintenance. Every new abbreviation, drug, or template requires a change. The
cost is not the change itself but the validation around it. A team that cannot afford a regression
test cannot safely maintain a rule system at scale, and should consider a learned component for the
parts of the vocabulary that change fastest.

The third limit is that rules do not generalize across languages or institutions without
translation and curation. The ConText historical result is a warning: even within one language and
one health system, a trigger list transferred unevenly across report types. Cross-site deployment
requires local validation, not just local vocabulary.

A fourth limit is that rule performance is reported on the corpus that motivated the rules. A
negation cue list built by reading 100 documents will perform well on those documents and less well
on a random sample. The evaluation should use a held-out sample, stratified by note type and time,
and the sample should be annotated without access to the rule list. The bias from rule-informed
annotation is subtle and common.

A fifth limit is combinatorial. Each rule interacts with every other rule that can apply to the
same span. A system with 40 rules has hundreds of possible interactions, and only the ones the
authors thought of are tested. This is the strongest argument for keeping rule sets small and for
using a learned component where interactions multiply. It is also the reason a regression sample
should include adversarial cases, not only typical ones.

## 12. Exercises

**Exercise 4.1 — Longest match (implementation, 40 minutes).**
Deliverable: extend the Section 7 code with a term list that includes overlapping terms ("insulin,"
"insulin glargine") and a coordination case ("metformin and lisinopril"). Show the matches before
and after longest-match resolution.
Acceptance criteria: code runs; offsets correct; one coordination case discussed.

**Exercise 4.2 — Pseudo-trigger audit (analysis, 45 minutes).**
Deliverable: using the per-cue table in Section 6, compute the overall false-negation rate with and
without the two problematic phrases.
Acceptance criteria: both rates computed; the effect on negation recall stated.

**Exercise 4.3 — Scope comparison (calculation, 40 minutes).**
Deliverable: reproduce the termination-scope and five-token-window metrics in Section 6, including
assertion accuracy and patient-pair PPV.
Acceptance criteria: all values correct; a one-paragraph recommendation.

**Exercise 4.4 — Section policy (design, 45 minutes).**
Deliverable: write a section policy for medication extraction with three inclusion rules and two
exclusion rules. Predict the effect on recall and precision.
Acceptance criteria: five rules; predictions with mechanisms; a validation plan.

**Exercise 4.5 — Abbreviation strategy (design, 40 minutes).**
Deliverable: for five ambiguous abbreviations in your domain, choose a resolution strategy from
Section 5 Step 12 and justify it.
Acceptance criteria: five abbreviations; strategy per abbreviation; expected precision and recall
effect.

**Exercise 4.6 — Baseline fairness (critique, 30 minutes).**
Deliverable: critique a study that reports a transformer at F1 0.91 against a regex baseline at
0.62, with no tuning of the regex. Name three ways the baseline could be strengthened.
Acceptance criteria: three concrete improvements; a statement of how each would change the
comparison.

**Exercise 4.7 — Rule regression test (design, 50 minutes).**
Deliverable: design a regression test for a negation rule change, including sample size, strata, and
the metric that triggers a rollback.
Acceptance criteria: sample size justified; at least three strata; rollback threshold stated.

**Exercise 4.8 — Hybrid design (design, 60 minutes).**
Deliverable: design a hybrid pipeline for a task of your choice, naming which components are rules,
which are learned, and why.
Acceptance criteria: component table; rationale per component; interface definitions.

## 13. Self-check answers

**4.2 (pseudo-trigger audit).** All cues: 30 false negations among 310 gold-negated spans, rate
0.097. Removing "no change" (41 spans, 12 errors) and "no increase" (28 spans, 7 errors) leaves 19
errors among 241 spans, rate 0.079. The absolute reduction is 11 errors; the rate falls by 1.8
points. Negation recall would rise from 250/310 = 0.806 to 261/310 = 0.842 if the 11 spans were
correctly handled, assuming no new errors.

**4.3 (scope comparison).** Termination scope: correctly negated 250, false negations 30, missed
60. Precision 250/280 = 0.893; recall 250/310 = 0.806; accuracy (250+680)/1,020 = 0.912.
Five-token window: correctly negated 204, false negations 30, missed 106. Precision 204/234 =
0.872; recall 204/310 = 0.658; accuracy (204+680)/1,020 = 0.867. Pair PPV: 410/(410+56) = 0.880
versus 410/(410+70) = 0.854. Recommend termination scope; the additional cue list is small and the
recall gain is large.

**4.1 (longest match) rubric.** Full credit: code returns "insulin glargine" as one span and does
not also return "insulin"; coordination case shows both drugs matched separately; offsets verified.
Half credit: code runs but double-counts the nested term.

**4.4 (section policy) rubric.** Full credit: three inclusion rules (discharge medications, hospital
course, plan) and two exclusions (indication, allergy), with predicted recall loss from excluding
allergy and predicted precision gain from excluding indication. Half credit: rules without
predictions.

**4.5 (abbreviation strategy) rubric.** Full credit: five abbreviations with a strategy and an
expected effect; at least one assigned to context classification because global expansion would be
wrong. Half credit: strategies without effects.

**4.6 (baseline fairness).** Improvements: add a local abbreviation table mined from the corpus; add
section filtering; tune the regex on a development split rather than using an untuned pattern. Each
would raise baseline recall and narrow the reported gap. A study that does not do this reports the
model's advantage over an straw man, not over a competent classical system.

**4.7 (regression test) rubric.** Full credit: sample size justified by the smallest detectable
difference, at least three strata (note type, negation-heavy section, abbreviation-heavy service),
and a rollback threshold on negation recall and patient-level PPV. Half credit: sample without
justification.

**4.8 (hybrid design) rubric.** Full credit: component table with rules for section filtering,
negation scope, and lexicon matching; learned components for abbreviation disambiguation and
concept ranking; interfaces defined; rationale tied to auditability and error profile. Half credit:
components named without interfaces.

## 14. Interview questions

**1. "Why would you use a rule-based system in 2026?"**
Strong answer: auditability, determinism, small labeled data, explicit evidence, and a baseline
requirement; names a task where rules are competitive. Weak answer: "because it is simpler," with
no task analysis.

**2. "What is a pseudo-trigger?"**
Strong answer: a phrase containing a negation cue that does not negate, such as "no increase" or
"gram-negative"; explains the over-negation it prevents. Weak answer: "a fake negation," with no
example.

**3. "How does NegEx scope work, and what are its limits?"**
Strong answer: cue plus five-token window, pseudo-negation filters; limits are long lists, passive
constructions, and history versus current visit. Weak answer: "it looks for 'no' before the word."

**4. "How would you improve negation recall?"**
Strong answer: termination-based scope, expanded cue list from error analysis, section context,
and a learned classifier for ambiguous cases. Weak answer: "increase the window."

**5. "What is the risk of a global abbreviation dictionary?"**
Strong answer: ambiguous short forms produce systematic errors; gives examples such as MS and PT;
recommends context or section rules. Weak answer: "abbreviations are ambiguous," with no mechanism.

**6. "How do you version a rule-based system?"**
Strong answer: version the code, lexicon, terminology release, and normalization; keep a fixed
regression sample; report change impact. Weak answer: "use git."

**7. "When does a rule system become a medical device?"**
Strong answer: when its intended use informs diagnosis or treatment for a specific patient,
regardless of technique; cites intended use and risk. Weak answer: "rules are not AI, so no."

**8. "How would you compare a rule system and a transformer fairly?"**
Strong answer: same corpus, same reference standard, same unit, tuned baseline, confidence
intervals, per-stratum results. Weak answer: compares headline F1 only.

**9. "What is the first thing you check when a rule system's cohort doubles after a lexicon
update?"**
Strong answer: overlap resolution and negation behavior; run the regression sample; diff the
outputs. Weak answer: assumes the new terms are correct.

**10. "Why is span F1 not enough for a medication dashboard?"**
Strong answer: the dashboard needs active medications per patient; assertion and aggregation change
the number; evaluate at the pair or patient unit. Weak answer: "F1 is standard."

**11. "How would you maintain a lexicon at scale?"**
Strong answer: ownership, source tracking, review, versioning, regression tests, and monitoring of
unmatched terms. Weak answer: "add terms as needed."

**12. "What does ConText's historical F-score tell you?"**
Strong answer: 0.76 versus 0.93 for negation; surface triggers do not capture historical status
reliably; needs section, discourse, and knowledge. Weak answer: "it works well."

## 15. Key terms

- **Abbreviation expansion** — mapping a short form to a long form.
- **Assertion** — the status assigned to a mention.
- **Baseline** — the simplest system that could work.
- **Candidate generation** — producing possible mentions.
- **Coordination** — conjunction of terms sharing a modifier.
- **CRF** — conditional random field, a sequence labeler.
- **cTAKES** — an open-source clinical NLP pipeline.
- **Cue** — a phrase that opens an assertion scope.
- **Dictionary-based** — matching against a curated term list.
- **Experiencer** — who an assertion applies to.
- **Fuzzy match** — approximate string matching.
- **Gazetteer** — a term list used for matching.
- **Hybrid system** — rules plus a learned component.
- **Lexicon** — a term list with metadata.
- **Longest match** — preferring the longest overlapping term.
- **NegEx** — a rule-based negation algorithm for clinical text.
- **Over-negation** — negating an affirmed mention.
- **Pseudo-trigger** — a phrase containing a cue that does not negate.
- **Regex** — a regular expression pattern.
- **Rule regression test** — a fixed labeled sample run on every change.
- **Scope** — the text region a cue applies to.
- **Section filter** — restricting extraction by section.
- **SVM** — support vector machine.
- **Termination term** — a phrase that closes a scope.
- **Temporality** — when an event occurred.
- **Under-negation** — missing a negation.
- **UMLS** — a metathesaurus of biomedical vocabularies.
- **Window** — a fixed token distance for scope.
- **Rule precedence** — the order in which assertion labels override one another.
- **Unmatched-term queue** — candidate strings absent from the lexicon, used for curation.
- **Stage-level output** — intermediate results that allow errors to be attributed to a component.

## 16. Sources

All URLs accessed 25 October 2026.

1. Chapman WW, Bridewell W, Hanbury P, Cooper GF, Buchanan BG. "A simple algorithm for identifying
   negated findings and diseases in discharge summaries." J Biomed Inform. 2001;34(5):301-310.
   https://doi.org/10.1006/jbin.2001.1029
2. Harkema H, Dowling JN, Thornblade T, Chapman WW. "ConText: An algorithm for determining
   negation, experiencer, and temporal status from clinical reports." J Biomed Inform.
   2009;42(5):839-851. https://doi.org/10.1016/j.jbi.2009.05.002
3. Savova GK, Masanz JJ, Ogren PV, et al. "Mayo clinical Text Analysis and Knowledge Extraction
   System (cTAKES): system description and performance." J Am Med Inform Assoc. 2010;17(5):507-513.
   https://doi.org/10.1136/jamia.2009.001560
4. Uzuner O, South BR, Shen S, DuVall SL. "2010 i2b2/VA challenge on concepts, assertions, and
   relations in clinical text." J Am Med Inform Assoc. 2011;18(5):552-556.
   https://doi.org/10.1136/amiajnl-2011-000203
5. Stubbs A, Filannino M, Soysal E, Henry S, Uzuner O. "Cohort selection for clinical trials:
   n2c2 2018 shared task track 1." J Am Med Inform Assoc. 2019;26(11):1163-1171.
   https://doi.org/10.1093/jamia/ocz163
6. Kraljevic Z, Searle T, Shek A, et al. "Multi-domain clinical natural language processing with
   MedCAT." Artif Intell Med. 2021;117:102083.
   https://doi.org/10.1016/j.artmed.2021.102083
7. U.S. National Library of Medicine. "UMLS License Agreement."
   https://www.nlm.nih.gov/databases/umls.html
8. SNOMED International. "Licensing and National Release Centers." https://www.snomed.org/
9. U.S. National Library of Medicine. "RxNorm." https://www.nlm.nih.gov/research/umls/rxnorm/
10. Regenstrief Institute. "LOINC License." https://loinc.org/license/
11. Apache Software Foundation. "Apache cTAKES." https://ctakes.apache.org/
12. U.S. Food and Drug Administration. "Artificial Intelligence-Enabled Device Software Functions:
    Lifecycle Management and Marketing Submission Recommendations." Draft guidance, January 2025.
    https://www.fda.gov/media/184856/download

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_task-taxonomy.md`](03_task-taxonomy.md)
- Next topic: [`05_transformers-for-clinical-text.md`](05_transformers-for-clinical-text.md)
- Annotation of cues and scope: [`06_annotation.md`](06_annotation.md)
- Evaluation at span and patient level: [`07_evaluation.md`](07_evaluation.md)
- Module survey: [`../../09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
