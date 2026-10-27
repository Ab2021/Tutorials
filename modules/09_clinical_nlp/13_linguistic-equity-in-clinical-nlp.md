# Linguistic Equity and Multilingual Clinical NLP

> **Module 09 — Clinical Natural Language Processing**
> Topic 13 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, 06, and 07
> Version 2026.3

## 1. Why this topic exists

A symptom-extraction model reports recall 0.89 on the validation set. The number is true and
useless. For patients whose preferred language is English, recall is 0.91. For patients whose
preferred language is Vietnamese, it is 0.71. The aggregate is a weighted average of strata that
the model serves unequally, and the weights are the population, not the risk.

Linguistic equity is the practice of measuring that difference, explaining it, and deciding what to
do about it. It is a measurement discipline before it is an engineering one. A team that cannot
state its recall by language stratum cannot know whether it has a problem, and a team that measures
only the aggregate will usually conclude that it does not.

The population is not marginal. In the United States, 71.1 million people, or 22.5 percent of the
population aged five and over, speak a language other than English at home, and about 25.8 million
people speak English less than "very well" (American Community Survey, 2023 one-year estimates).
Those 25.8 million are the population with limited English proficiency (LEP), and they are the
population for whom an English-only clinical NLP pipeline is a filter on care rather than a tool for
it.

The evidence that language barriers harm patients is old and consistent. Divi et al. (International
Journal for Quality in Health Care, 2007) found that LEP patients in US hospitals were more likely
to experience an adverse event and more likely to experience harm than English-speaking patients.
Flores et al. (Annals of Emergency Medicine, 2012) audited interpreted emergency department
encounters and found that 18 percent of interpreter errors had potential clinical consequences, with
a higher rate for ad hoc interpreters than for professionally trained ones. The case of Willie
Ramirez, a young man left quadriplegic in 1980 after the Spanish word "intoxicado" was rendered as
"intoxicated", settled for a reported 71 million US dollars and remains the standard teaching
example for why a single word matters.

The language technology literature has a matching gap. Névéol et al. (Journal of Biomedical
Semantics, 2018) found that roughly 10 percent of clinical NLP publications addressed languages other
than English, and that terminology coverage outside English was a fraction of English coverage: in
the 2016 UMLS release, English had about 9.1 million terms, Spanish about 1.3 million, and most other
languages less than 5 percent of the English total. The models that dominate clinical NLP,
including ClinicalBERT, GatorTron, and the medical language models derived from them, were
pretrained predominantly on English text from a small number of US academic medical centres.

Three mechanisms produce the performance gap, and conflating them is the most common analytical
error in this area.

**The language of the text.** A tokenizer trained on English fragments Spanish, Vietnamese, or
Chinese differently, and a model trained on English has weaker representations for the resulting
subwords. Petrov et al. (NeurIPS, 2023) measured a tokenization premium of up to about 15 times for
some languages relative to English on parallel text, with Italian at about 1.6 times. More tokens mean higher cost, shorter effective context, and more opportunities for the model
to lose a negation.

**The language of the encounter.** In most US health systems the note is written in English by an
English-speaking clinician, even when the patient speaks another language and an interpreter is
present. The note is then shorter, more template-driven, and filtered through a third party. The
model reads English text and still performs worse, because the text is a poorer record of the
encounter. This mechanism is invisible to anyone who stratifies only by note language.

**The patient population.** Documentation practices, access, and trust differ by population. A
patient who has seen the same clinician for ten years has a richer record than a patient who arrived
last month. A population that moves frequently has fragmented records. These differences are not
linguistic, but they appear in a language-stratified audit and are easily misread as model bias.

The legal context has tightened. Section 1557 of the Affordable Care Act, as revised by the 2024
final rule, prohibits discrimination on the basis of national origin in health programmes and, at 45
CFR 92.210, requires covered entities to identify and mitigate discrimination risk from patient care
decision support tools. Title VI of the Civil Rights Act of 1964 and Executive Order 13166 require
meaningful access for LEP individuals. The Office of the National Coordinator for Health Information
Technology (ONC) HTI-1 rule requires developers of predictive decision support interventions to
disclose fairness performance in test, external, and local data. None of these instruments defines a
numeric threshold, and all of them assume that the covered entity has measured something.

This topic builds the measurement. It covers the strata, the reference standard, the disparity
metrics, the decomposition of a gap into its layers, the specific mechanics of tokenization and
translation, and the interventions that follow from each mechanism. The output is an audit that a
quality committee can read and an engineering team can act on.

> **Note:** Language categories are not a proxy for race, ethnicity, or citizenship, and a
> language-stratified audit is not a complete fairness audit. It is one axis among several, and it
> should be reported alongside the others rather than instead of them.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** the language strata for an audit, distinguishing patient preferred language, note
   language, and encounter language, and **explain** why the choice changes the conclusion.
2. **Compute** recall, precision, and disparity gaps by stratum with Wilson confidence intervals,
   and **state** the unit and denominator of each.
3. **Decompose** an aggregate disparity into a text-language component and a documentation
   component using a stratified cross-tabulation.
4. **Measure** tokenization premium and its cost, context, and latency consequences for a given
   corpus and tokenizer.
5. **Evaluate** translation quality for clinical text using back-translation and error
   classification, and **identify** false cognates and hedging loss.
6. **Design** a reference standard that is equivalent across languages, including bilingual
   annotation and adjudication.
7. **Choose** an intervention (model, translation, workflow, or monitoring) that matches the
   mechanism behind the gap.
8. **Report** a subgroup audit with denominators, intervals, and an explicit statement of what the
   design cannot establish.

## 3. The core mental model

The mental model has two parts: a chain of translations, and a two-axis map of what is being
measured.

### The chain of translations

A clinical NLP system that serves a multilingual population is not one model reading one text. It is
a chain in which every link converts meaning from one representation to another.

```text
  patient's speech (language A)
        |
        |  1. interpreter, ad hoc interpreter, or machine translation
        v
  clinician's understanding (language B, usually English)
        |
        |  2. clinician's documentation, in the note language (usually English)
        v
  note text (language B)
        |
        |  3. tokenizer, vocabulary, model
        v
  model output (structured concepts or generated text)
        |
        |  4. clinician reads and acts
        v
  decision
```

Each arrow is lossy, and the losses are not independent. A hedging phrase lost at link 1 becomes a
definite statement at link 2, which the model at link 3 represents with high confidence. An
interpreter who summarises instead of interpreting produces a shorter note, which reduces the
evidence available to the model and to the next clinician.

Equity means that the final decision is as good for a patient entering the chain in language A as
for one entering in language B. It does not mean that each link is identical. A system can treat
every link differently and still produce equal outcomes, and it can treat every link identically and
produce unequal ones, because the inputs differ.

The practical consequence is that a disparity measured at the end of the chain cannot be attributed
to a link without decomposition. That is the purpose of the next part.

### The two-axis map

Most audits stratify by one variable and attribute the result to the model. The map below separates
two axes that are usually confounded.

|  | Note written in English | Note written in another language |
|---|---|---|
| **Patient prefers English** | Baseline. The population the models were built for. | Rare; occurs with bilingual clinicians or translated templates. |
| **Patient prefers another language** | The common case in US health systems. The gap here is documentation and interpretation, not tokenization. | The case the multilingual literature studies. The gap here combines text processing and documentation. |

The cells have different mechanisms and therefore different interventions.

- In the top-left cell, the intervention is better data and better evaluation.
- In the bottom-left cell, the intervention is interpreter workflow, documentation support, and
  clinician training, not a multilingual model.
- In the bottom-right cell, the intervention includes the multilingual model, the tokenizer, and the
  translation path.
- The top-right cell is small and usually excluded from audits, which is itself a finding about
  coverage.

An audit that reports "patients with preferred language Spanish have lower recall" without saying
which cell dominates has reported a correlation between language and outcome and called it a model
property.

### Why the aggregate hides the gap

Recall over the whole population is a weighted average:

```text
 R_total = sum over strata s of  w_s * R_s
```

where w_s is the share of positive cases in stratum s. If a stratum is 10 percent of positives and
its recall is 20 points lower, the aggregate falls by 2 points. That is within the noise of most
validation studies, and it is easy to report as "no meaningful difference".

The same arithmetic applies to a 2 percent stratum with a 40-point gap: the aggregate moves by 0.8
points. The patients in that stratum experience double the error rate, and the dashboard shows
nothing.

This is the statistical form of the problem. The equity question is not "is the average good" but
"is the worst-served group good enough", and the average is structurally incapable of answering it.

### Tokenization arithmetic

Two quantities are used throughout this topic and they are not interchangeable.

```text
 fertility(language) = tokens produced by the tokenizer / words in the text

 premium(A, B)       = tokens for content in language A
                       --------------------------------
                       tokens for the same content in language B
```

Fertility is a within-language ratio and depends on how the language marks word boundaries. A text
of 100 English words that produces 130 tokens has fertility 1.30. The same content in Spanish might
produce 190 tokens from 100 words, fertility 1.90, which yields a premium of 190 / 130 = 1.46.

The premium is the quantity that matters for cost, latency, and effective context length, because
it compares the same meaning across languages. Fertility is the quantity that matters for a single
language's context budget. A report that compares fertility across a language with spaces and a
language without them has compared two different units and called the result a disparity.

### What equity is not

Three common misunderstandings are worth naming.

- **Equity is not equal accuracy on a translated benchmark.** A model evaluated on machine-translated
  English test sets measures the translation, not the population. The translation introduces its own
  errors and removes the code-switching, dialect, and register that real text contains.
- **Equity is not a single number.** A disparity ratio of 0.78 is a description at one operating
  point. Change the threshold and it changes. Report the curve or report the operating point with
  its threshold.
- **Equity is not achieved by removing the language field.** A model that never sees language cannot
  discriminate by language directly, but it can discriminate through every correlate of language:
  note length, vocabulary, section structure, provider. Blinding is not a fix; measurement is.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Limited English proficiency (LEP) | Speaking English less than "very well" | Treated as a binary; proficiency is a spectrum and is self-reported | Census category |
| Preferred language | The language a patient prefers for care communication | Confused with the language of the note | ISO 639 code |
| Language concordance | Clinician and patient share a language | Assumed to mean the clinician is fluent; concordance is not proficiency | Boolean |
| Interpreter-mediated encounter | Encounter with a third-party interpreter | Treated as a two-party encounter with extra noise | Encounter type |
| Professional interpreter | Interpreter with formal training and assessment | Assumed to be any bilingual staff member | Training hours |
| Ad hoc interpreter | Family member, friend, or untrained staff member interpreting | Treated as equivalent to a professional | Role code |
| Sight translation | Oral translation of a written document | Confused with interpreting a conversation | Task type |
| Back-translation | Translating a translated text back to the source language to check meaning | Treated as a validation of clinical adequacy | Procedure |
| Code-switching | Alternating between languages within a conversation or text | Treated as poor language rather than a normal bilingual practice | Count of switches |
| Dialect | A variety of a language associated with a region or social group | Treated as an error to be normalised | Variety label |
| African American Vernacular English (AAVE) | A systematic variety of English with its own grammar and lexicon | Treated as informal or incorrect English | Variety label |
| False cognate | A word that resembles a word in another language but differs in meaning | Confused with a spelling error | Lexical class |
| Tokenizer | Component that splits text into subword units | Treated as language-neutral | Vocabulary size |
| Subword token | A unit produced by the tokenizer, between a character and a word | Counted as a word in cost calculations | Count |
| Fertility | Tokens per whitespace-delimited word | Compared across scripts where word boundaries differ | Ratio, tokens per word |
| Tokenization premium | Ratio of tokenized length in one language to another on parallel text | Confused with fertility; premium is a cross-language comparison | Ratio |
| Out-of-vocabulary (OOV) rate | Share of tokens not in the model vocabulary | Reported without the tokenizer version | Percent |
| Vocabulary coverage | Share of gold entities whose surface form or synonym is in the vocabulary or terminology | Assumed to equal model accuracy | Percent |
| Stratum | A subgroup defined by a variable of interest | Defined post hoc to produce a result | Label |
| Disparity gap | Difference in a metric between a reference and a comparison stratum | Reported without a confidence interval | Percentage points |
| Disparity ratio | Ratio of a metric in a comparison stratum to the reference stratum | Reported without the base rate | Unitless |
| Equalized odds | Equality of true and false positive rates across groups | Confused with demographic parity | Criterion |
| Calibration by group | Agreement between predicted risk and observed risk within each group | Assumed to follow from overall calibration | Calibration slope |
| Wilson interval | A confidence interval for a proportion with good small-sample behaviour | Replaced by the normal approximation at small n | Interval |
| Annotation equivalence | The same construct is labelled the same way across languages | Assumed because the guideline was translated | Agreement statistic |
| USCDI | United States Core Data for Interoperability, the ONC data element set | Assumed to include interpreter use; it does not | Data element set |
| CLAS standards | National Culturally and Linguistically Appropriate Services standards | Treated as regulation; they are guidance | Standard |
| Title VI | The Civil Rights Act provision prohibiting national origin discrimination in federally assisted programmes | Applied to all entities regardless of federal funding | Statute |
| Section 1557 | ACA provision prohibiting discrimination in health programmes, with a 2024 rule on decision support tools | Treated as covering only insurance | 45 CFR Part 92 |
| HTI-1 | ONC rule requiring transparency for decision support interventions in certified health IT | Assumed to apply to every clinical model | Certification criterion |
| ISO 17100 | Standard for translation services | Applied to interpreting | Standard |
| ISO 13611 | Standard for community interpreting | Applied to written translation | Standard |

Two definitions need expansion because they carry most of the analytical weight.

**Fertility versus premium.** Fertility is tokens per word within one language. It is useful for
estimating context consumption, and it is not comparable across scripts, because a language written
without spaces does not have the same "word" unit. Tokenization premium is the ratio of tokenized
length between two languages on the same content, and it is the right measure for cross-language
comparison. A report that compares fertility across Chinese and English is comparing a tokens-per-
word ratio in one script to a tokens-per-character ratio in another, and the comparison is
meaningless.

**Gap versus ratio.** A gap of 20 percentage points between 0.91 and 0.71 and a ratio of 0.78 describe
the same difference. The gap is easier to interpret when the base rate is high; the ratio is easier
to interpret when the base rate is low. A metric at 0.10 falling to 0.05 is a 5-point gap and a 0.50
ratio, and the two framings support different arguments. Report both, with the base rate.

## 5. The mechanics, step by step

This section is the procedure for a subgroup performance audit. It is written for a team that has a
deployed model, a language field in the EHR, and a quality committee that has asked whether the
model works equally well for everyone.

The procedure produces a report, not a model. The report is the deliverable, and it should be
readable by a clinician who does not know what a tokenizer is.

### Step 1. State the decision and the unit

**Input:** The model, its intended use, and the decision it supports.

**Output:** A decision statement, a unit of analysis, and a harm statement.

**Rationale:** The metric depends on the decision. A triage model that decides who is called back
needs high recall, because a missed patient is a missed follow-up. A coding model that assigns a
billable code needs high precision, because a wrong code is a claim error. A summarisation model
needs claim-level faithfulness, not recall.

The unit matters as much. A note-level metric treats each document as one observation; a
patient-level metric treats each patient as one. If a patient has ten notes and the model fails on
all ten, the note-level recall counts ten failures and the patient-level recall counts one. The
equity question is usually about patients, so the patient-level metric is usually the right one for
the headline.

**Branch point:** If the model supports more than one decision, the audit needs more than one
metric. Do not average a screening metric and a billing metric into a single quality score.

### Step 2. Define the strata

**Input:** The EHR language fields, the note metadata, and the encounter record.

**Output:** A stratification plan with a primary axis and at least one secondary axis.

**Rationale:** Three axes are available and they measure different things.

- **Patient preferred language** is the equity axis. It is the population attribute that the
  regulation and the quality committee care about.
- **Note language** is the processing axis. It determines what the tokenizer and model actually
  read.
- **Encounter interpreter mode** is the documentation axis. It determines how much of the encounter
  reached the note.

An audit that uses only the first axis will attribute documentation effects to the model. An audit
that uses only the second will miss the population entirely. The primary axis should be patient
preferred language, because that is the accountability axis, and the secondary axes explain the
mechanism.

**Branch point:** Stratification choices and their consequences.

| Choice | What it measures | What it hides | Use when |
|---|---|---|---|
| Patient preferred language only | Population equity | Mechanism | Reporting to a quality committee |
| Note language only | Processing performance | Population equity | Debugging the model |
| Both, cross-tabulated | Population and mechanism | Nothing, but cells get small | Investigating a gap |
| Interpreter mode only | Documentation quality | Model performance | Auditing the encounter workflow |
| Race or ethnicity only | A different axis | Language entirely | A separate audit; not a substitute |
| No stratification | The average | Everything | Never for an equity claim |

### Step 3. Build a bilingual reference standard

**Input:** A sample of encounters, the annotation guideline, and bilingual annotators.

**Output:** Gold labels in each language, with annotator identity, language, and adjudication
records.

**Rationale:** The reference standard is the ceiling on every conclusion. If the gold labels are
produced by translating the English guideline and asking English-speaking annotators to apply it to
translated text, the audit measures the translation and the annotators, not the model.

Three practices improve equivalence. First, translate the guideline, not only the examples, and have
a bilingual clinician review the translation for construct drift: a concept such as "shortness of
breath" may have a natural expression in one language that does not map to the English phrase.
Second, annotate in the source language, not in translation, so that the annotator sees the same
text the model would see in that language. Third, double-annotate a sample in each language and
compute agreement within and across languages.

**Branch point:** If the cross-language agreement is materially lower than the within-language
agreement, the construct is not equivalent, and no model comparison across languages is
interpretable until the guideline is fixed.

### Step 4. Test annotation equivalence

**Input:** Double-annotated samples from each language stratum.

**Output:** Agreement statistics within and across languages, and a list of systematically
disagreed concepts.

**Rationale:** Agreement within a language measures the clarity of the guideline. Agreement across
languages, measured on parallel or comparable content, measures whether the same construct is being
labelled. The two are different, and only the second supports a cross-language claim.

A useful diagnostic is the concept-level disagreement table: for each entity type, the share of
cross-language disagreements that are boundary differences, type differences, or genuine
disagreements about whether the concept is present. A high share of genuine disagreements is a
construct problem. A high share of boundary differences is a guideline problem that can be fixed
with examples.

**Branch point:** If the reference standard cannot be made equivalent, restrict the audit to
within-language comparisons and say so. A within-language disparity (for example, by interpreter
mode) is still actionable.

### Step 5. Describe the strata before comparing them

**Input:** The stratified sample and the note metadata.

**Output:** A table of base rates, note lengths, section completeness, interpreter use, and
document type by stratum.

**Rationale:** Many apparent model disparities are descriptions of the data. If notes for one
stratum are half as long, the model has less evidence and will recall less. If one stratum's notes
are more often templated, the entity density differs. If one stratum has more telephone encounters,
the text is different in kind.

The description table is also where implausible strata become visible. A stratum with 40 notes
cannot support a claim about a 5-point difference, and a stratum with 12 positives cannot support
any claim at all. Seeing the counts before the metrics prevents the most common overreach in
subgroup reporting.

**Branch point:** If two strata differ by more than about 30 percent in mean note length, plan a
length-matched comparison before interpreting the gap.

### Step 6. Choose the metric and the disparity measure

**Input:** The decision from Step 1 and the data description from Step 5.

**Output:** A primary metric, a disparity measure, and a pre-specified comparison.

**Rationale:** The metric follows from the decision, and the disparity measure follows from the
base rate. Recall is the right primary metric for a screening or triage decision. Precision or PPV
is right for a decision that triggers an action with a cost. Calibration is right when the output is
a probability that a clinician uses to counsel.

For the disparity, report the absolute gap in percentage points and the ratio. The gap is
interpretable when base rates are similar; the ratio is interpretable when they are not. Do not
report a p-value without an interval, and do not compare many strata pairwise without a
pre-specified primary comparison.

**Branch point:** Metric selection.

| Decision type | Primary metric | Disparity measure | Why |
|---|---|---|---|
| Screening, recall-critical | Recall at the operating threshold | Gap in percentage points | A missed case is the harm |
| Alerting, action-costly | PPV at the operating threshold | Ratio, with base rate | False alerts consume staff time |
| Risk score for counselling | Calibration slope and intercept by group | Calibration gap | A miscalibrated score misleads |
| Extraction for analytics | F1 with exact-match rule | Gap and ratio | Both errors matter |
| Generative summary | Claim-level faithfulness by clinician review | Rate of unsupported claims | Fluency is not faithfulness |
| Any of the above | Add coverage or abstention rate | Gap | A model that abstains unequally is inequitable |

### Step 7. Plan the precision of the comparison

**Input:** The expected stratum sizes and base rates.

**Output:** The expected width of each interval and a minimum detectable difference.

**Rationale:** A subgroup of 300 notes with 48 positives yields a Wilson interval roughly 25
percentage points wide. That width is larger than most differences the audit is trying to detect.
Reporting a point estimate without the interval invites a conclusion the data cannot support, in
either direction.

The planning calculation is simple. For a proportion near 0.8, the standard error is about
sqrt(0.8 × 0.2 / n), and the 95 percent interval is about plus or minus 1.96 standard errors. At
n = 48, the half-width is about 11 percentage points. At n = 500, it is about 3.5 points.

**Branch point:** If the minimum detectable difference is larger than the difference the committee
cares about, either collect more data, pool strata with a stated rationale, or report the audit as
descriptive and say that it cannot establish a difference.

### Step 8. Compute performance by stratum with intervals

**Input:** Gold labels and predictions, joined at the unit of analysis.

**Output:** A table of metric, numerator, denominator, point estimate, and interval per stratum.

**Rationale:** The table is the core artefact. Every cell needs a numerator and a denominator,
because a rate without them cannot be checked. Every estimate needs an interval, because the eye
ranks point estimates and the intervals often overlap.

The table should also include the aggregate, so that the reader can see how much the aggregate hides.
A useful addition is the "worst-stratum" metric: the recall of the lowest-performing stratum, which
is what an equity-focused reader actually wants to know.

**Branch point:** If the interval for the worst stratum overlaps the reference stratum, the correct
statement is that the audit did not detect a difference, not that the model is equitable. Absence of
evidence at small n is not evidence of absence.

### Step 9. Decompose the gap by layer

**Input:** The stratified results and the intermediate pipeline artefacts: token counts, vocabulary
coverage, translations, and note characteristics.

**Output:** An attribution of the gap to lexical, tokenization, model, translation, or documentation
components, with the evidence for each.

**Rationale:** The intervention depends on the mechanism. A gap caused by vocabulary coverage is
fixed with terminology and tokenizer work. A gap caused by documentation is not fixed by any model
change.

Four comparisons do most of the work.

- **Length-matched comparison.** Restrict the reference stratum to notes of similar length and
  recompute. If the gap shrinks substantially, documentation volume explains much of it.
- **Note-language split.** Within a patient-language stratum, compare notes written in the patient's
  language with notes written in English. If the gap is concentrated in the non-English notes, the
  mechanism is text processing; if it persists in English notes, the mechanism is the encounter.
- **Vocabulary coverage.** Compute the share of gold entity mentions whose surface form or a listed
  synonym is in the model vocabulary. Low coverage points to a lexical problem.
- **Ablation.** Replace the model's tokenizer with one trained on the target language, or fine-tune
  on in-language data, and re-measure. A large improvement confirms the model layer.

**Branch point:** Decomposition outcomes.

| Pattern | Most likely mechanism | Next test | Typical intervention |
|---|---|---|---|
| Gap disappears after length matching | Documentation volume | Compare section completeness | Documentation support, interpreter workflow |
| Gap concentrated in non-English notes | Text processing | Vocabulary coverage, tokenizer ablation | Tokenizer, in-language training data |
| Gap persists in English notes about LEP patients | Encounter and documentation | Interpreter mode analysis | Interpreter access, note templates |
| Gap correlates with vocabulary coverage | Lexical | OOV rate by stratum | Terminology expansion, transliteration handling |
| Gap appears at one site only | Local workflow | Site-level process review | Local training, local validation |
| Gap appears in calibration but not discrimination | Population base rates | Recalibration by group | Group-specific recalibration |

A fifth comparison isolates the translation layer when the pipeline machine-translates before
processing. Take a sample of source-language texts, translate them with the production engine, and
have a bilingual clinician mark each translated sentence as adequate, meaning-preserving but
awkward, meaning-changing, or clinically dangerous. Then run the model on both the original and the
translation and compare. If the model performs well on the original and poorly on the translation,
the translation is the mechanism; if it performs poorly on both, the model is the mechanism. This
test is the only way to separate the two, and it is rarely run because it requires bilingual
reviewers and access to the source text.

The decomposition should be reported as an ordering of mechanisms with evidence, not as a single
attributed percentage. The evidence is rarely strong enough to support "60 percent documentation,
40 percent model", and a report that states that precision invites false confidence. A report that
states "the gap persists after length matching and is concentrated in non-English notes" is
defensible and actionable.

### Step 10. Test the confounders

**Input:** Case-mix variables and the stratified results.

**Output:** An adjusted comparison and a statement of what remains unexplained.

**Rationale:** Language is correlated with age, insurance, visit type, provider, and documentation
era. A crude comparison attributes all of the difference to language. A stratified or matched
comparison removes the part explained by the confounders and leaves a smaller, more credible
estimate.

The confounders worth measuring are note length, section completeness, document type (telephone,
portal, progress note, discharge summary), provider, visit year, and patient panel duration. The
last one is often the strongest: a patient with a long relationship has a richer record, and a
population with high turnover has thinner records.

**Branch point:** If adjustment removes the gap, report that the crude gap was explained by case
mix. That is a finding, not a failure. If adjustment does not remove it, report the adjusted gap and
the variables included.

### Step 11. Choose the intervention

**Input:** The decomposition and the adjusted gap.

**Output:** An intervention plan with a target, an owner, and a re-measurement date.

**Rationale:** The four mechanisms have four different fixes, and applying the wrong one wastes a
year.

**Branch point:** Intervention selection.

| Mechanism | Intervention | Does not fix it |
|---|---|---|
| Tokenization and vocabulary | Domain tokenizer, in-language pretraining, terminology expansion | More English training data |
| Translation quality | Clinician-reviewed translation, entity preservation, glossary constraints | A larger general model |
| Interpreter workflow | Professional interpreter access, training, encounter documentation standards | Any model change |
| Documentation volume | Templates, scribes, structured capture for LEP encounters | Any model change |
| Calibration | Group-specific recalibration, threshold adjustment | Retraining on balanced data alone |
| Measurement gap | Mandatory language fields, interpreter-use fields, note-language metadata | A fairness dashboard without fields |

The last row is the most common. Many organisations cannot compute the audit because the language
field is missing, free-text, or inconsistently populated. The first intervention is then a data
capture intervention, and it precedes every model intervention.

### Step 12. Monitor and set re-audit triggers

**Input:** The audit results, the intervention plan, and the monitoring infrastructure.

**Output:** A dashboard specification, a re-audit schedule, and explicit triggers.

**Rationale:** Language performance drifts when the population changes, when the model is updated,
when the note template changes, and when the interpreter service changes. An audit performed once is
a snapshot of a system that has already moved.

The triggers should be specific. A re-audit is required when the share of a stratum changes by more
than 5 percentage points, when the model or tokenizer version changes, when a new site or language
is added, and when a safety event involves a patient in a monitored stratum.

The dashboard should show the worst-stratum metric, not only the aggregate, and it should show the
denominator, because a stratum that shrinks to 30 notes produces a volatile line that will be
misread as a trend.

The procedure ends with a report that a committee can act on: a decision, a stratum definition, a
reference standard with its limitations, a table with intervals, a decomposition with evidence, an
intervention with an owner, and a monitoring plan. That report is the artefact. The model is not
changed by the audit; the organisation is.

## 6. Worked example

A health system has deployed a symptom-extraction model that reads encounter notes and flags
patients for a nurse triage queue. The model extracts a fixed symptom set, including chest pain and
shortness of breath, and the queue is worked in order of the model's score. The quality committee
asks whether the model works equally well for patients with limited English proficiency.

All numbers are synthetic. The structure of the calculation is the point.

### The sample

The audit draws 4,800 notes from 3,200 patients over 18 months, stratified by the patient's
preferred language as recorded in the EHR.

| Stratum | Notes | Patients | Positives | True positives | Recall | 95% CI | Gap vs English | Ratio |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| English | 3,000 | 1,950 | 300 | 273 | 0.910 | 0.872-0.937 | 0.000 | 1.000 |
| Spanish | 1,100 | 760 | 132 | 110 | 0.833 | 0.761-0.887 | −0.077 | 0.916 |
| Chinese | 400 | 290 | 56 | 42 | 0.750 | 0.623-0.845 | −0.160 | 0.824 |
| Vietnamese | 300 | 200 | 48 | 34 | 0.708 | 0.568-0.818 | −0.202 | 0.778 |
| Total | 4,800 | 3,200 | 536 | 459 | 0.856 | | | |

Aggregate recall = 459 / 536 = 0.856.

The aggregate is 0.856 and the worst stratum is 0.708. The difference is 14.8 points, and the
aggregate is much closer to the English value because English patients contribute 300 of the 536
positives, or 56 percent.

The averaging effect is easy to quantify. If the Vietnamese stratum performed at the English rate,
it would yield 48 × 0.910 = 43.7 true positives instead of 34, an increase of 9.7. Aggregate recall
would rise from 459/536 = 0.856 to 468.7/536 = 0.874. Closing the entire gap for the worst-served
stratum moves the headline by 1.8 percentage points, which is within the run-to-run variation of
most model updates. A dashboard showing only the aggregate would never surface the problem.

### Describing the strata before comparing them

| Stratum | Base rate | Mean note length (words) | Interpreter documented | Mean sections complete (of 8) |
|---|---:|---:|---:|---:|
| English | 0.100 | 420 | 0.00 | 7.1 |
| Spanish | 0.120 | 330 | 0.62 | 6.4 |
| Chinese | 0.140 | 250 | 0.71 | 5.9 |
| Vietnamese | 0.160 | 230 | 0.68 | 5.7 |

The strata differ in three ways that could each explain a performance gap: the base rate is higher
in the smaller strata, the notes are shorter, and the notes are less complete. None of these
differences is about the model.

Two of them push in opposite directions. A higher base rate usually makes recall easier, because
there are more positive examples and the model's prior is better matched. Shorter and less complete
notes make recall harder, because there is less evidence. The observed pattern, with recall falling
as notes get shorter, suggests the documentation effect dominates.

### Length-matched comparison

The English stratum is restricted to notes between 200 and 320 words, matching the Vietnamese
distribution. The matched subset has 800 notes, 80 positives, and 68 true positives.

Recall = 68 / 80 = 0.850, Wilson 95 percent interval 0.756 to 0.912.

The gap against the Vietnamese stratum falls from 20.2 to 14.2 percentage points. Length explains
about six points of the twenty, or roughly 30 percent of the crude gap. The remaining 14.2 points
are not explained by volume alone.

This is the first diagnostic in the audit. A gap that disappears after length matching is a
documentation finding; a gap that persists is a model or content finding.

### Splitting the Spanish stratum by note language

Ninety-six percent of the notes in the audit are written in English, including almost all of the
notes for Chinese-speaking and Vietnamese-speaking patients. The Spanish stratum is the only one
with a substantial number of notes in the patient's language: 180 of 1,100.

| Sub-stratum | Notes | Positives | True positives | Recall | 95% CI |
|---|---:|---:|---:|---:|---|
| Spanish preferred, Spanish note | 180 | 26 | 18 | 0.692 | 0.500-0.835 |
| Spanish preferred, English note | 920 | 106 | 92 | 0.868 | 0.790-0.920 |

The two sub-strata differ by 17.6 percentage points. The confidence interval for the Spanish-note
subset is wide because only 26 positives are in it, and the intervals overlap at the edges, so the
difference is suggestive rather than established. The point estimate is consistent with the
literature on in-language processing, and it is large enough to justify a targeted test.

The English-note sub-stratum at 0.868 is 4.2 points below the English-patient stratum at 0.910. The
intervals overlap, so this difference is not established at these sample sizes. It is the
documentation and encounter effect, and the length-matched comparison suggests it is real.

The decomposition is therefore:

| Component | Estimate | Evidence |
|---|---:|---|
| Crude gap, English to Vietnamese | 20.2 points | Table above |
| Explained by note length | about 6 points | Length-matched recall 0.850 |
| Associated with in-language text | about 17.6 points within Spanish | Spanish-note versus English-note split |
| Residual, English text about LEP patients | about 4.2 points | Spanish-preferring patients with English notes |

The components do not add to the crude gap, because they are measured on different strata and with
different precision. That is normal. The report states the ordering, not a decomposition identity.

### What parity would cost

The Vietnamese threshold is currently set at the same value as the English threshold. To raise
Vietnamese recall to the English level, the operating point would move.

| Operating point | True positives | False positives | Precision | Reviews per true positive |
|---|---:|---:|---:|---:|
| Current | 34 | 18 | 0.654 | 1.53 |
| Parity recall | 44 | 41 | 0.518 | 1.93 |

Parity costs 10 additional true positives and 23 additional false positives on 300 notes. The nurse
queue grows by 33 items, or 11 percent, and the review burden per true case rises from 1.53 to
1.93.

That is the trade-off in its simplest form. A single global threshold cannot equalise recall across
strata with different score distributions; equalising recall necessarily lowers precision in the
stratum with the weaker distribution. Whether that is acceptable is a clinical and operational
decision, and it should be made explicitly rather than by leaving the threshold at its default.

### Tokenization and cost

The pipeline sends note text to a hosted model priced per token. The mean token counts per note and
the resulting cost are:

| Stratum | Tokens per note | Cost per 1,000 notes (US dollars) | Relative to English |
|---|---:|---:|---:|
| English | 546 | 2.73 | 1.00 |
| Spanish | 780 | 3.90 | 1.43 |
| Chinese | 410 | 2.05 | 0.75 |
| Vietnamese | 690 | 3.45 | 1.26 |

The Spanish premium is 1.43 times per note. The Chinese figure is lower than English, not because
Chinese is cheaper per unit of meaning, but because the Chinese notes in this sample are much
shorter. A per-note cost comparison therefore understates the per-word premium, and a per-word
comparison overstates the operational cost. Report both, and state the unit.

The context consequence is separate from the cost. At a fixed context window, a note that consumes
43 percent more tokens leaves 43 percent less room for the rest of the prompt, which for a
long-document pipeline can mean truncating the oldest part of the note. Truncation is not random: it
removes the beginning, which is where the history of present illness usually lives.

### Diagnostic A: length-matched comparison

The comparison in this example is the diagnostic. Compute recall for the reference stratum on a
subset matched to the comparison stratum's note-length distribution.

| Result | Interpretation |
|---|---|
| Gap falls by more than half | Documentation volume is the main mechanism |
| Gap falls by 20 to 50 percent | Mixed mechanism; continue decomposition |
| Gap unchanged | The model or the content is the mechanism |
| Gap grows after matching | Length was masking a larger model gap |

The diagnostic is cheap, it requires no new data collection, and it prevents the most expensive
error in this area: buying a multilingual model to fix a documentation problem.

### Diagnostic B: vocabulary coverage

For each stratum, compute the share of gold entity mentions whose surface form, or a listed synonym
in the terminology, appears in the model's vocabulary.

| Coverage | Recall | Interpretation |
|---|---:|---|
| Above 0.95 | 0.71 | The words are known; the model is not using them |
| 0.85 to 0.95 | 0.71 | Marginal lexical loss; check tokenization of the missing terms |
| Below 0.85 | 0.71 | Lexical failure; expand terminology or change the tokenizer |

In this audit, coverage for the Vietnamese stratum is 0.88, driven by diacritics and by compound
terms that the tokenizer splits. That is consistent with a partial lexical contribution, and it
does not explain the full gap, because the English-note sub-stratum for Spanish-preferring patients
also underperforms with near-complete coverage.

### The varied assumption

Suppose the audit had stratified by **note language** instead of patient preferred language, which
is what a team without a reliable language field would do.

Under that stratification, 4,620 notes are English and 180 are Spanish. English-note recall is
441/510 = 0.865 and Spanish-note recall is 18/26 = 0.692. The gap is 17.3 points, the Vietnamese
stratum has disappeared into the English group, and the recommendation is to improve Spanish text
processing.

The population conclusion is different. Under patient-language stratification, the worst-served
group is Vietnamese-speaking patients, whose notes are in English and who would never appear in a
note-language audit. The mechanism is the encounter, not the tokenizer, and the intervention is
interpreter access and documentation support rather than a Spanish model.

Both audits are correct about what they measure. Only one of them answers the committee's question.
That is why Step 2 of the procedure insists on the patient-language axis as primary and on the
note-language axis as explanatory.

### Summary

| Finding | Value | Unit |
|---|---:|---|
| Aggregate recall | 0.856 | proportion of positives |
| Worst-stratum recall | 0.708 | proportion of positives, Vietnamese |
| Crude gap | 20.2 | percentage points |
| Gap after length matching | 14.2 | percentage points |
| Gap within Spanish stratum, by note language | 17.6 | percentage points |
| Additional true positives for parity | 10 | cases per 300 notes |
| Additional false positives for parity | 23 | cases per 300 notes |
| Spanish token premium per note | 1.43 | ratio to English |

## 7. Code, query, or calculation

The script below reproduces every number in Section 6: stratified recall with Wilson intervals, a
bootstrap interval for the disparity gap, the length-matched comparison, the note-language split,
the token cost table, and the threshold-parity arithmetic. It uses only the standard library and a
fixed random seed.

Assumptions: the counts are supplied rather than computed from predictions; the bootstrap resamples
within each stratum independently; the token counts are inputs from a production tokenizer rather
than computed here, because a real tokenizer is not a standard-library component.

```python
import math
import random

def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half

strata = {
    "English":    (273, 300, 420),
    "Spanish":    (110, 132, 330),
    "Chinese":    (42,  56,  250),
    "Vietnamese": (34,  48,  230),
}
print("stratum       tp   n   recall   95% CI          gap vs EN  ratio")
base = None
for name, (k, n, length) in strata.items():
    r = k / n
    lo, hi = wilson(k, n)
    if base is None:
        base = r
    print(f"{name:12s} {k:4d} {n:4d}  {r:.3f}  {lo:.3f}-{hi:.3f}  {r - base:+.3f}     {r / base:.3f}")

# Bootstrap the English-minus-Vietnamese gap.
rng = random.Random(20261027)
def draw(k, n):
    p = k / n
    return sum(rng.random() < p for _ in range(n)) / n

gaps = []
for _ in range(5000):
    gaps.append(draw(273, 300) - draw(34, 48))
gaps.sort()
print("gap EN-VI bootstrap 95% CI:", round(gaps[125], 3), "to", round(gaps[4874], 3))

# Length-matched English subset: 800 notes, 80 positives, 68 found, mean 250 words.
lo, hi = wilson(68, 80)
print(f"length-matched EN: 68/80 = {68/80:.3f} CI {lo:.3f}-{hi:.3f}, "
      f"gap vs VI {(68/80)-(34/48):+.3f}")

# Spanish stratum split by note language.
for label, k, n in (("ES notes, Spanish text", 18, 26),
                    ("ES notes, English text", 92, 106)):
    lo, hi = wilson(k, n)
    print(f"{label}: {k}/{n} = {k/n:.3f} CI {lo:.3f}-{hi:.3f}")

# Token cost per note at 5 US dollars per million tokens.
tokens = {"English": 546, "Spanish": 780, "Chinese": 410, "Vietnamese": 690}
price = 5.0 / 1_000_000
for name, t in tokens.items():
    print(f"{name:12s} {t:4d} tokens/note  ${t*price*1000:6.2f} per 1,000 notes  "
          f"x{t/546:.2f} vs English")

# Threshold shift for Vietnamese.
print("VI at current threshold: tp=34 fp=18 ppv=%.3f review/TP=%.2f" % (34/52, 52/34))
print("VI at parity threshold:  tp=44 fp=41 ppv=%.3f review/TP=%.2f" % (44/85, 85/44))
print("extra TP", 44-34, "extra FP", 41-18)
```

Output:

```text
stratum       tp   n   recall   95% CI          gap vs EN  ratio
English       273  300  0.910  0.872-0.937  +0.000     1.000
Spanish       110  132  0.833  0.761-0.887  -0.077     0.916
Chinese        42   56  0.750  0.623-0.845  -0.160     0.824
Vietnamese     34   48  0.708  0.568-0.818  -0.202     0.778
gap EN-VI bootstrap 95% CI: 0.073 to 0.337
length-matched EN: 68/80 = 0.850 CI 0.756-0.912, gap vs VI +0.142
ES notes, Spanish text: 18/26 = 0.692 CI 0.500-0.835
ES notes, English text: 92/106 = 0.868 CI 0.790-0.920
English       546 tokens/note  $  2.73 per 1,000 notes  x1.00 vs English
Spanish       780 tokens/note  $  3.90 per 1,000 notes  x1.43 vs English
Chinese       410 tokens/note  $  2.05 per 1,000 notes  x0.75 vs English
Vietnamese    690 tokens/note  $  3.45 per 1,000 notes  x1.26 vs English
VI at current threshold: tp=34 fp=18 ppv=0.654 review/TP=1.53
VI at parity threshold:  tp=44 fp=41 ppv=0.518 review/TP=1.93
extra TP 10 extra FP 23
```

Four readings matter.

The confidence intervals in the first table show the difference between a finding and a direction.
The English and Vietnamese intervals do not overlap, and the bootstrap interval for the gap runs
from 7.3 to 33.7 percentage points. The gap is established; its size is not. A report that says "the
gap is 20 points" is overstating the precision, while a report that says "there is a gap, probably
between 7 and 34 points" is accurate and still motivating.

The Spanish split shows the small-sample problem directly. The Spanish-note estimate of 0.692 has an
interval from 0.500 to 0.835, which overlaps the English-note interval. The point estimate suggests a
17.6-point difference; the interval says the data are consistent with anything from a small
difference to a large one. The correct action is to collect more Spanish notes, not to conclude that
the difference is absent.

The token table shows why the unit matters. Per note, Spanish costs 1.43 times English and Chinese
costs 0.75 times. Per word, Spanish and Vietnamese would both show a premium because their notes are
shorter. Neither number is wrong; they answer different questions, and a cost model should use the
per-note figure while a fairness argument should use the per-unit-of-content figure.

The threshold arithmetic is the decision aid. Equalising recall adds 10 true positives and 23 false
positives per 300 notes. Whether that trade is acceptable depends on the cost of a missed case
relative to a nurse review, which is a local judgement. What is not acceptable is to present the
parity option as free.

The script also demonstrates a limitation of audit code in general: it takes the counts as inputs.
The hard part is producing the counts correctly, which requires the reference standard, the joins,
and the language fields described in Section 5. The arithmetic is the easy layer.

## 8. Data, standards, and regulatory context

| Instrument | Issuer | What it constrains | Status |
|---|---|---|---|
| Title VI, Civil Rights Act of 1964 | US Congress | National origin discrimination in federally assisted programmes | Binding |
| Executive Order 13166 | US President, 2000 | Meaningful access for LEP individuals | Binding on federal agencies; applies to recipients through Title VI |
| Section 1557, Affordable Care Act; 45 CFR Part 92 | US Department of Health and Human Services (HHS) | Discrimination in health programmes, including through decision support tools | Binding |
| 45 CFR 92.210 | HHS Office for Civil Rights | Identification and mitigation of discrimination risk from patient care decision support tools | Binding; compliance within 300 days of the July 2024 effective date |
| HTI-1, 45 CFR 170.315(b)(11) | Office of the National Coordinator for Health IT (ONC) | Transparency for predictive decision support interventions in certified health IT | Binding on certified developers |
| USCDI | ONC | Core data elements for interoperability, including patient demographic elements | Required for certification |
| CLAS standards | HHS Office of Minority Health | Culturally and linguistically appropriate services | Guidance |
| ISO 639 | International Organization for Standardization | Language codes | Convention |
| ISO 17100 | ISO | Requirements for translation services | Voluntary standard |
| ISO 13611 | ISO | Requirements for community interpreting | Voluntary standard |
| Regulation (EU) 2024/1689 (AI Act) | European Union | Risk management and data governance for high-risk AI systems | Binding, phased |
| General Data Protection Regulation, Article 9 | European Union | Special categories of personal data, including racial or ethnic origin | Binding |

### The American legal frame

Title VI prohibits discrimination on the basis of national origin in programmes that receive
federal financial assistance. Because most hospitals and health systems receive Medicare and
Medicaid payments, Title VI reaches most of the industry. Executive Order 13166, issued in 2000,
directs federal agencies to ensure that their recipients provide meaningful access to LEP
individuals, and the Department of Justice guidance that followed describes what "meaningful"
means in practice: a mix of language assistance services proportionate to the size of the LEP
population, the frequency of contact, and the resources of the entity.

Section 1557 codified and extended these obligations for health programmes. The 2024 final rule
added a provision that is directly relevant to this topic. At 45 CFR 92.210, a covered entity must
not discriminate through the use of a patient care decision support tool, must make reasonable
efforts to identify tools that use input variables measuring race, colour, national origin, sex,
age, or disability, and must make reasonable efforts to mitigate the risk of discrimination. The
rule does not set a numeric threshold, does not require access to a vendor's training data, and does
not prescribe a mitigation method.

The practical consequence for a data scientist is that the obligation is procedural: identify,
assess, mitigate, document. A health system that has never stratified its clinical NLP performance
by language cannot show that it made reasonable efforts to identify the risk. The audit is the
evidence.

### The transparency frame

HTI-1 is narrower and more specific. It applies to certified health IT developers and to predictive
decision support interventions supplied as part of a certified module. For those interventions, the
developer must disclose a set of source attributes, 31 in total across nine categories, including the
demographic characteristics of the training and test data and the fairness performance in test,
external, and local data. The rule requires disclosure, not a threshold. A model that performs
poorly for a subgroup can be certified as long as the poor performance is disclosed.

That is a deliberate design choice, and it puts the interpretive burden on the buyer. A procurement
team that does not ask for the stratified performance table will not be shown it, because the
regulation requires it to exist rather than to be good.

### The data elements

Stratification requires fields, and the fields are weaker than the obligation.

- **Preferred language** is captured in most EHRs, but it is often a free-text field with
  inconsistent values ("Spanish", "spanish", "Español", "Spanish - Mexico"), and it is often
  defaulted to English at registration.
- **Interpreter use** is recorded in some systems as an encounter procedure and in others not at
  all. Where it is recorded, the mode (in person, video, telephone, ad hoc) is frequently missing.
- **Note language** is almost never a structured field. It can be inferred, badly, from the text.
- **Clinician language proficiency** is rarely recorded in a usable form.

A team that wants to run the audit in Section 6 will spend most of its time on these four fields,
not on the model. That is the normal experience, and it is why the first recommendation of many
audits is a data capture recommendation.

### The international frame

The EU AI Act classifies most clinical AI as high risk when it is a medical device, and imposes data
governance duties that include examination of possible biases affecting health and safety. Language
is not a protected characteristic in the Act's list, but national and ethnic origin are, and language
is a close proxy. The GDPR treats racial and ethnic origin as a special category, which constrains
how language and nationality data can be processed and retained.

The practical difference from the US frame is that the EU requires a risk management system rather
than a disclosure. A conformity assessment asks for the analysis, the test results, and the
mitigation, not only for the table.

### What the standards do not cover

No standard defines a minimum acceptable performance for a language subgroup. No standard specifies
the granularity of stratification: is Spanish one stratum, or is Mexican Spanish different from
Caribbean Spanish, and does a health system with 40 patients of one variety need to report it? No
standard defines how to evaluate a tokenizer for fairness, although the research literature has
proposals.

The gap is filled by professional judgement and by the expectations of the quality committee, the
accreditor, and the plaintiff's expert. The audit in this topic is designed to be defensible under
all three, which means it reports denominators, intervals, and limitations rather than a single
composite fairness score.

## 9. Industry practice and case evidence

### What is documented

The evidence base on language and clinical NLP has three parts, and they are of different quality.

**The health services evidence** is the strongest. Divi et al. (2007) analysed reported adverse
events in US hospitals and found that LEP patients were more likely to experience an adverse event
and more likely to experience harm. Flores et al. (2012) audited 57 interpreted paediatric emergency
department encounters and found 1,884 interpreter errors, of which 18 percent had potential clinical
consequences; the share of errors with potential clinical consequences was 12 percent for
professional interpreters, 22 percent for ad hoc interpreters, and 20 percent with no interpreter. Among professional interpreters, those with at
least 100 hours of training had a 2 percent consequential-error rate against 12 percent for those
with fewer. The Willie Ramirez case (1980) remains the canonical illustration of a false cognate
changing a diagnosis.

**The translation quality evidence** is smaller but direct. Khoong et al. (JAMA Internal Medicine,
2019) translated 100 emergency department discharge instruction sets into Spanish and Chinese with a
consumer translation engine and back-translated them for clinician review. Ninety-two percent of
Spanish sentences and 81 percent of Chinese sentences were accurate. Clinically significant or
life-threatening potential harm appeared in 2 percent of Spanish sentences and 8 percent of Chinese
sentences, with examples including "hold the kidney medicine" rendered as "keep taking" it.

**The NLP evidence** is the weakest for this specific question. Névéol et al. (2018) documented the
publication and resource gap, and Petrov et al. (2023) quantified tokenizer unfairness on parallel
text. There is much less work that measures the performance of a deployed clinical NLP system by
patient language in a real health system, and almost none that reports the decomposition in
Section 6. Most of what is claimed in vendor material is inference from benchmark results.

### What the industry does

The vendor landscape divides into three groups.

- **EHR platform vendors** provide a preferred-language field, an interpreter documentation module
  in some configurations, and machine translation of patient-facing text in some portals. The
  coverage of these features varies by country and by contract.
- **Interpreter service vendors** provide video and audio interpreting, usually with a documented
  quality program and interpreter qualification. Their performance is measured by connection time
  and customer satisfaction more often than by interpretation accuracy.
- **Language technology vendors** provide translation APIs and multilingual language models. Their
  benchmarks are usually general-domain, and their clinical evaluations are usually small.

The deployment pattern that appears in public accounts is that human interpreting remains the
standard for clinical conversations, machine translation is used for written patient-facing material
and for clinician comprehension of foreign-language text, and multilingual NLP is used for
back-office analytics more often than for direct clinical decisions.

### The evidence gap in procurement

A procurement team evaluating a clinical NLP product will typically receive a validation report with
an overall sensitivity and specificity, a description of the development data, and a statement that
the model was validated on a diverse population. The report rarely contains a table by language,
because the vendor's test set rarely has enough non-English cases to populate one.

The inference, clearly labelled as inference, is that most deployed clinical NLP systems have never
been evaluated for linguistic equity in the population they serve, and that the absence of a
language-stratified table in a validation report is the normal case rather than a red flag about a
particular vendor. The buyer's leverage is the contract: require the stratified table as a deliverable
and as a condition of renewal.

### The counter-example

The counter-example is worth naming because it shows what is possible. A health system that operates
a large interpreter service can link interpreter encounters to notes and to outcomes, and can
therefore stratify by interpreter mode as well as by language. That linkage supports the analysis in
Section 6 and turns an abstract fairness question into an operational one: are professional
interpreter encounters documented differently from ad hoc encounters, and does the difference reach
the model?

The organisations that can answer that question are the ones that treated language as a structured
clinical variable rather than as a registration detail.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Aggregate-only reporting | One recall number in the validation report | Averages hide small strata | Inequity invisible | Report worst-stratum metric |
| Wrong stratification axis | Audit groups by note language only | Patient population disappears into the majority | Wrong intervention | Patient language primary; note language explanatory |
| Translated test set as multilingual validation | "Evaluated in 12 languages" | Translation benchmark measures the translator | Overstated coverage | In-language evaluation with bilingual annotators |
| Language field defaulted to English | 95 percent English in a diverse panel | Registration workflow defaults | Strata misclassified | Audit field completeness; offer language at check-in |
| Interpreter use not recorded | Cannot compute interpreter-mode strata | No structured field | Documentation effect unmeasurable | Add encounter procedure code |
| Tokenizer fragmentation | High tokens per word for one language | Vocabulary trained on English | Cost, truncation, lost negation | Domain tokenizer; measure premium |
| Diacritic stripping | "año" normalised to "ano" | Aggressive text normalisation | Meaning changed | Preserve diacritics; normalise after tokenisation |
| Code-switching split into monolingual halves | Entities lost at language boundaries | Language identification applied per sentence | Missed concepts | Language-agnostic model; span-level handling |
| False cognate | "Intoxicado" rendered as "intoxicated" | General translation engine | Wrong diagnosis | Glossary constraints; bilingual review |
| Hedging loss | "Could be" rendered as "is" | Translation normalises modality | Certainty overstated | Preserve modality; flag uncertain statements |
| Back-translation accepted as validation | "Back-translated and reviewed" with no error count | Fluency is not adequacy | Silent meaning errors | Error taxonomy with counts and severity |
| Annotation guideline drift | Cross-language kappa much lower than within-language | Construct translated, not adapted | Incomparable labels | Bilingual guideline review; adjudication |
| One global threshold | Same cut-off for every stratum | Score distributions differ | Unequal recall | Stratified operating points or recalibration |
| Calibration shift | Predicted risk overestimates in one stratum | Base rates differ | Wrong resource allocation | Group-specific recalibration |
| Subgroup too small | 12 positives in a reported stratum | No power calculation | False confidence or false reassurance | Report intervals; pool with rationale |
| Language as race proxy | Language stratum used in place of an ethnicity audit | Correlation treated as identity | Misattributed findings | Report both axes separately |

The red flags cluster around two mistakes. The first is measuring the wrong axis, which produces a
technically correct audit that answers a different question from the one the committee asked. The
second is measuring with too little data and reporting the point estimate without the interval,
which produces a conclusion that the sample cannot support.

A third cluster is specific to text processing: normalisation steps that were designed for English
and that destroy information in other languages. Diacritic stripping, lowercasing, aggressive
punctuation removal, and English-specific abbreviation expansion all fall into this category. Each
is invisible in an English-only evaluation and each can change a clinical meaning in another
language.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The language field is accurate | Strata depend on it | Defaults, free text, changes over time | Compare with interpreter records and self-report |
| A language stratum is internally homogeneous | Sample sizes require pooling | Regional varieties, dialects, literacy | Report variety where recorded; sensitivity analysis |
| Note language equals the processing language | Tokenization analysis depends on it | English notes about non-English patients | Language identification on a sample |
| Gold labels are equivalent across languages | Cross-language comparison depends on it | Guideline translation drift | Double annotation; cross-language agreement |
| Interpreter use is recorded | Documentation effect depends on it | No structured field | Chart audit on a sample |
| Base rates are stable over the audit window | Pooling across time | Seasonality, policy change, pandemic | Stratify by period |
| The tokenizer version is fixed | Token counts depend on it | Silent vendor updates | Record version per request |
| The translation engine is fixed | Translation quality depends on it | Model updates, language additions | Record engine and version |
| The operating threshold is comparable across strata | Gap depends on it | Different score distributions | Report the curve, not only the point |
| The audit sample represents deployment | Generalisation depends on it | Convenience sampling, one site | Compare sample and population characteristics |

The largest limit is the reference standard. Every number in the audit is measured against labels,
and the labels are produced by annotators who themselves work in a language and a clinical culture.
A bilingual annotator is not a neutral instrument, and cross-language agreement is a measure of how
well two clinical cultures share a construct, not only of how well two annotators agree.

The second limit is that the audit measures the model on the records that exist. Patients who never
reach the health system, or who leave without a note, are absent from every stratum. A model can
look equitable on the documented population while access itself is unequal, and the audit cannot
see that.

## 12. Exercises

### Exercise 1. Compute a stratified audit

Given a table of notes, positives, and true positives for four language strata, compute recall, the
Wilson 95 percent interval, the gap against the reference stratum, and the ratio. Then compute the
aggregate and the change in the aggregate if the worst stratum reached the reference rate.

**Deliverable:** A results table and a one-paragraph interpretation.

**Acceptance criteria:** Every rate has a numerator and a denominator; intervals are reported; the
aggregate is computed from totals rather than by averaging stratum rates; the interpretation
distinguishes an established difference from a directional one.

**Time:** 45 minutes.

### Exercise 2. Decompose a gap

You are given a patient-language audit showing a 20-point gap, a note-language split for one
stratum, and mean note lengths. Decompose the gap into a documentation component and a text-language
component.

**Deliverable:** A short analysis with the two components, their evidence, and a statement of what
cannot be separated with the available data.

**Acceptance criteria:** The analysis uses a length-matched comparison; it does not claim an exact
percentage attribution; it states which stratum the residual comes from.

**Time:** 60 minutes.

### Exercise 3. Design a bilingual annotation study

Design the annotation of 600 notes in English and Spanish for a symptom-extraction task.

**Deliverable:** A protocol with annotator qualifications, guideline translation and adaptation,
double-annotation plan, adjudication, and agreement analysis.

**Acceptance criteria:** The guideline is adapted, not only translated; cross-language agreement is
measured and distinguished from within-language agreement; the protocol states how disagreements
about the construct are resolved.

**Time:** 90 minutes.

### Exercise 4. Audit a tokenizer

Using a parallel corpus of the same clinical content in English and one other language, compute the
tokenization premium for a tokenizer of your choice.

**Deliverable:** A script or spreadsheet, the premium, and a paragraph on the cost and context
consequences.

**Acceptance criteria:** The comparison uses parallel content, not separate corpora; the unit is
stated as a ratio of token counts; the paragraph distinguishes per-document from per-word effects.

**Time:** 60 minutes.

### Exercise 5. Evaluate a translation

Take ten clinical sentences, translate them with a general engine, back-translate them, and classify
the errors.

**Deliverable:** An error table with the source, translation, back-translation, error type, and
severity.

**Acceptance criteria:** Errors are classified as meaning-preserving, meaning-changing, or
clinically dangerous; at least one false cognate or hedging loss is identified if present; the
limitations of back-translation are stated.

**Time:** 60 minutes.

### Exercise 6. Set a threshold policy

Given recall and precision at several operating points for two strata, recommend a threshold policy.

**Deliverable:** A one-page decision memo with the options, the trade-off, and a recommendation.

**Acceptance criteria:** The memo quantifies the true and false positives added by parity; it names
the decision-maker; it states how the policy will be monitored.

**Time:** 45 minutes.

### Exercise 7. Critique a validation report

Find or construct a validation report for a clinical NLP product that lacks a language-stratified
table. Write the questions a procurement team should ask.

**Deliverable:** A list of ten questions with the reason for each and the acceptable answer.

**Acceptance criteria:** Questions cover the test set composition, the stratification, the reference
standard, the operating point, the translation path, and the monitoring plan; each question has a
criterion for a satisfactory answer.

**Time:** 45 minutes.

### Exercise 8. Specify the dashboard

Design a monitoring dashboard for linguistic equity in a deployed clinical NLP system.

**Deliverable:** A specification with metrics, strata, refresh cadence, thresholds, and alert routing.

**Acceptance criteria:** The dashboard shows the worst-stratum metric and the denominator; it
includes data-completeness metrics for the language field; it defines a re-audit trigger; it does not
reduce equity to a single composite score.

**Time:** 60 minutes.

## 13. Self-check answers

**Exercise 1.** Recall is true positives divided by positives, per stratum. The Wilson interval for
34/48 is 0.568 to 0.818, and for 273/300 it is 0.872 to 0.937. The gap is 0.910 − 0.708 = 0.202, or
20.2 percentage points, and the ratio is 0.778. The aggregate must be computed from totals: if the
four strata have 459 true positives out of 536 positives, the aggregate is 0.856. Averaging the four
stratum rates unweighted would give 0.800, which is wrong because it gives the 300-note stratum the
same weight as the 3,000-note stratum. If the worst stratum reached 0.910, it would contribute 43.7
rather than 34 true positives, and the aggregate would rise to 468.7/536 = 0.874, a change of 1.8
points.

**Exercise 2.** A strong answer reports the crude gap, the length-matched gap, and the within-stratum
split by note language. In the worked example, length matching reduces the gap from 20.2 to 14.2
points, and the Spanish-note versus English-note split shows a 17.6-point difference within the
Spanish stratum. The answer should say that the components are measured on different subsets with
different precision, that they do not sum to the crude gap, and that the residual 4.2 points in
English notes about LEP patients is the part most likely to reflect the encounter rather than the
text.

**Exercise 3.** A strong protocol has bilingual annotators who annotate in the source language,
a guideline adapted by a bilingual clinician rather than translated word for word, at least 20
double-annotated notes per language, adjudication by a third annotator, and agreement reported
within and across languages. It states that low cross-language agreement invalidates the
cross-language comparison until the guideline is fixed.

**Exercise 4.** A strong audit uses the same content in both languages, reports the premium as a
ratio of token counts, and separates the per-document effect from the per-word effect. It notes that
Chinese and other scripts without whitespace delimiters cannot be compared on a tokens-per-word
basis, and that the premium translates into cost, latency, and effective context length.

**Exercise 5.** A strong table classifies each error and assigns severity. Meaning-preserving errors
include word order and register. Meaning-changing errors include a lost negation or a changed
subject. Clinically dangerous errors include a false cognate, a dose change, or a lost instruction
to stop a medicine. The answer must state that back-translation can hide errors that survive the
round trip and that bilingual clinical review is the stronger check.

**Exercise 6.** A strong memo quantifies the trade-off, for example 10 additional true positives and
23 additional false positives per 300 notes, and expresses the review burden per true case. It names
the clinical owner of the decision, states that the policy will be reviewed after a defined period,
and specifies the monitoring metric. It does not recommend a single global threshold without
acknowledging the precision cost.

**Exercise 7.** Strong questions ask for the language composition of the test set, the stratified
performance table, the reference standard and its language, the operating threshold and how it was
chosen, the translation path and its validation, the handling of code-switching, and the monitoring
plan. A satisfactory answer includes denominators and intervals, not only point estimates.

**Exercise 8.** A strong dashboard shows the worst-stratum recall and its denominator, the gap and
ratio against the reference stratum, data completeness for the language and interpreter fields, and
the volume of each stratum. It refreshes monthly, alerts when a stratum falls below a pre-specified
floor for two consecutive periods, and triggers a re-audit on model version change, population shift,
or a safety event in a monitored stratum. It does not collapse the metrics into a single score,
because a single score hides which stratum is failing.

## 14. Interview questions

**1. Why is an aggregate metric insufficient for an equity claim?**

A strong answer explains that the aggregate is a weighted average dominated by the largest stratum,
gives the arithmetic (a 20-point gap in a 9 percent stratum moves the total by under 2 points), and
says that the equity question is about the worst-served group. A weak answer says aggregate metrics
are less accurate.

**2. What are the three language axes, and why does the choice matter?**

A strong answer distinguishes patient preferred language, note language, and encounter interpreter
mode, and gives an example where a note-language audit hides a population that is served in English
but underserved. A weak answer treats language as a single variable.

**3. How would you decompose a performance gap?**

A strong answer describes length matching, the note-language split, vocabulary coverage, and
ablation, and says that the decomposition orders mechanisms rather than assigning exact
percentages. A weak answer proposes retraining on more data.

**4. What is tokenization premium, and why is fertility not comparable across scripts?**

A strong answer defines premium as a ratio of tokenized lengths on parallel content and explains
that fertility depends on whitespace-delimited words, which do not exist in the same form in
Chinese. A weak answer uses the two terms interchangeably.

**5. How would you validate a translation used in a clinical pipeline?**

A strong answer describes bilingual clinical review with an error taxonomy and severity, notes the
limits of back-translation, and proposes testing the model on the original and the translation
separately. A weak answer says to back-translate and check that it matches.

**6. What does Section 1557 require of a health system using a clinical NLP tool?**

A strong answer describes the prohibition, the duty to identify tools that use protected
characteristics, the duty to mitigate, and the absence of a numeric threshold. A weak answer says it
requires fairness, without the procedural content.

**7. How would you set a threshold across strata?**

A strong answer describes the recall-precision trade-off, quantifies the additional reviews, and
says that the decision belongs to a named clinical owner with monitoring. A weak answer says to use
the same threshold for everyone because it is fair.

**8. What would make you distrust a vendor's multilingual validation claim?**

A strong answer asks whether the test set was translated, whether the annotators were bilingual,
whether the table is stratified, and whether the operating point is the same. A weak answer says it
would want more languages.

**9. What data would you ask for before running this audit?**

A strong answer names preferred language, interpreter use and mode, note language or a proxy, note
length and section completeness, and the model's predictions at the deployed threshold. A weak
answer asks only for the model's overall performance.

## 15. Key terms

- **Ad hoc interpreter** — untrained family member or staff member interpreting.
- **Annotation equivalence** — the same construct labelled consistently across languages.
- **Back-translation** — translating a translation back to check meaning.
- **Calibration by group** — agreement of predicted and observed risk within a stratum.
- **CLAS standards** — US guidance for culturally and linguistically appropriate services.
- **Code-switching** — alternating languages within a conversation or text.
- **Disparity gap** — difference in a metric between strata, in percentage points.
- **Disparity ratio** — ratio of a metric between strata.
- **Equalized odds** — equality of true and false positive rates across groups.
- **False cognate** — a word that resembles one in another language but differs in meaning.
- **Fertility** — tokens per whitespace-delimited word.
- **HTI-1** — ONC rule on transparency for decision support interventions.
- **Interpreter-mediated encounter** — encounter with a third-party interpreter.
- **Language concordance** — clinician and patient share a language.
- **Limited English proficiency (LEP)** — speaking English less than "very well".
- **Note language** — the language in which the note is written.
- **Out-of-vocabulary rate** — share of tokens absent from the model vocabulary.
- **Patient preferred language** — the language the patient prefers for care.
- **Preferred language field** — the EHR field that records it.
- **Professional interpreter** — trained and assessed interpreter.
- **Section 1557** — ACA antidiscrimination provision, including 45 CFR 92.210.
- **Sight translation** — oral translation of a written document.
- **Stratum** — a subgroup defined by a variable of interest.
- **Subword token** — a tokenizer unit between a character and a word.
- **Title VI** — Civil Rights Act provision on national origin discrimination.
- **Tokenization premium** — ratio of tokenized lengths between languages on parallel text.
- **Tokenizer** — component that splits text into subword units.
- **USCDI** — ONC core data element set.
- **Vocabulary coverage** — share of gold mentions present in the vocabulary or terminology.
- **Wilson interval** — confidence interval for a proportion with good small-sample behaviour.

## 16. Sources

All URLs accessed 27 October 2026.

1. US Census Bureau. "American Community Survey 2023, 1-Year Estimates, Table S1601: Language
   Spoken at Home." https://data.census.gov/table/ACSST1Y2023.S1601
2. Névéol A, Dalianis H, Velupillai S, Savova G, Zweigenbaum P. "Clinical natural language
   processing in languages other than English: opportunities and challenges." Journal of
   Biomedical Semantics. 2018;9:12. https://doi.org/10.1186/s13326-018-0179-8
3. Petrov A, La Malfa E, Torr PHS, Bibi A. "Language model tokenizers introduce unfairness between
   languages." Advances in Neural Information Processing Systems 36 (NeurIPS). 2023.
   https://doi.org/10.48550/arXiv.2305.15425
4. Khoong EC, Steinbrook E, Brown C, Fernandez A. "Assessing the use of Google Translate for Spanish
   and Chinese translations of emergency department discharge instructions." JAMA Internal
   Medicine. 2019;179(4):580-582. https://doi.org/10.1001/jamainternmed.2018.7653
5. Flores G, Abreu M, Barone CP, Bachur R, Lin H. "Errors of medical interpretation and their
   potential clinical consequences: a comparison of professional versus ad hoc versus no
   interpreters." Annals of Emergency Medicine. 2012;60(5):545-553.
   https://doi.org/10.1016/j.annemergmed.2012.01.025
6. Divi C, Koss RG, Schmaltz SP, Loeb JM. "Language proficiency and adverse events in US hospitals:
   a pilot study." International Journal for Quality in Health Care. 2007;19(2):60-67.
   https://doi.org/10.1093/intqhc/mzl069
7. Price-Wise G. "Language, culture, and medical tragedy: the case of Willie Ramirez." Health
   Affairs Forefront. 19 November 2008.
   https://www.healthaffairs.org/content/forefront/language-culture-and-medical-tragedy-case-willie-ramirez
8. US Department of Health and Human Services. "Nondiscrimination in Health Programs and
   Activities," 89 FR 37522, 6 May 2024; 45 CFR Part 92, including 92.210.
   https://www.federalregister.gov/documents/2024/05/06/2024-08711/nondiscrimination-in-health-programs-and-activities
9. Office of the National Coordinator for Health IT. "HTI-1 Final Rule," 89 FR 1192, 9 January 2024.
   https://www.federalregister.gov/documents/2024/01/09/2023-28857
10. Office of the National Coordinator for Health IT. "HTI-1 Decision Support Interventions (DSI)
    Fact Sheet." December 2023.
    https://healthit.gov/wp-content/uploads/2023/12/HTI-1_DSI-fact-sheet_508.pdf
11. US Department of Justice. "Executive Order 13166: Improving Access to Services for Persons with
    Limited English Proficiency." 2000.
12. US Department of Health and Human Services, Office of Minority Health. "National Standards for
    Culturally and Linguistically Appropriate Services in Health and Health Care." 2013.
13. International Organization for Standardization. "ISO 17100:2015 Translation services:
    Requirements for translation services." 2015.
14. International Organization for Standardization. "ISO 13611:2014 Interpreting: Guidelines for
    community interpreting." 2014.
15. European Parliament and Council. "Regulation (EU) 2024/1689 (Artificial Intelligence Act)."
    Official Journal of the European Union, 12 July 2024.
16. NLLB Team. "No Language Left Behind: Scaling human-centered machine translation." 2022.
    https://doi.org/10.48550/arXiv.2207.04672

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`12_icsr-and-pharmacovigilance-nlp.md`](12_icsr-and-pharmacovigilance-nlp.md)
- Topic 01, text sources and provenance: [`01_clinical-text-sources.md`](01_clinical-text-sources.md)
- Topic 06, annotation and agreement: [`06_annotation.md`](06_annotation.md)
- Topic 07, evaluation and subgroup analysis: [`07_evaluation.md`](07_evaluation.md)
- Topic 11, ambient documentation and consent: [`11_ambient-ai-and-documentation.md`](11_ambient-ai-and-documentation.md)
- Module survey: [`../../course/09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
