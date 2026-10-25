# Transformers for Clinical Text

> **Module 09 — Clinical NLP**
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01 through 04
> Version 2026.3

## 1. Why this topic exists

Transformer models changed what clinical NLP could attempt. A model pretrained on biomedical or
clinical text can recognize entities, classify documents, and answer questions with less
task-specific feature engineering than a rule system. The change is real, and it is also narrower
than the marketing suggests: the model changes the cost of recall, not the definition of the task,
and it introduces new failure modes around context length, tokenization, and domain shift.

The first practical issue is length. A standard BERT-style encoder reads at most 512 subword tokens.
A discharge summary in MIMIC-III averages roughly 2,984 tokens, according to the Clinical-Longformer
paper, so a 512-token encoder sees a fraction of a typical note. If entities are distributed evenly,
the encoder sees about 17 percent of the text. If entities cluster at the end, as assessment and
plan sections often do, it sees less. A pipeline that truncates silently has a recall ceiling that no
amount of fine-tuning can exceed.

The second issue is tokenization. Clinical text contains abbreviations, measurements, and drug names
that general subword vocabularies fragment. "HFpEF" may become several tokens; "metformin" may be
split. Fragmentation is not fatal, but it shortens the effective context, complicates span
alignment, and makes character-offset mapping necessary for evaluation.

The third issue is domain. PubMedBERT was pretrained from scratch on 14 million PubMed abstracts,
about 3.2 billion words, with a PubMed-specific vocabulary, and it scored 81.16 on the BLURB
benchmark against 80.34 for BioBERT, 78.86 for SciBERT, 77.29 for ClinicalBERT, and 76.27 for
BlueBERT. Its authors deliberately excluded clinical notes and argued that biomedical literature and
clinical text differ enough to warrant separate benchmarks. That result is a warning: a model that
leads a literature benchmark has not demonstrated performance on notes.

The fourth issue is evaluation. Transformer papers often report token-level F1, which can be
several points higher than span-level F1 on the same output. A token-level score counts partial
overlap as success; a span-level score with exact boundaries does not. The unit changes the number,
and the number changes the decision. Topic 07 covers the metrics; this topic covers why transformer
output makes the distinction especially easy to blur.

The fifth issue is operational. Clinical text is PHI. Sending it to a hosted model without a
business associate agreement, or to a research checkpoint trained on restricted data, creates legal
and privacy exposure that a local regex does not. Model weights may carry license terms, and
fine-tuning on credentialed data may create obligations that are not settled by the base model's
license. The governance questions are as important as the accuracy questions.

A sixth issue is maintenance. A rule system is maintained by editing rules; a transformer is
maintained by collecting labels, retraining, and revalidating. The second is not necessarily more
expensive, but it is different, and it requires infrastructure that many clinical teams do not have.
A team that adopts a transformer without a labeling pipeline and a regression suite has traded a
maintainable system for an opaque one. The trade may still be right, but it should be made
deliberately.

This topic exists to make those issues concrete. It covers the transformer pipeline from
tokenization through long-document handling, fine-tuning, inference economics, and evaluation. It
does not teach deep learning from scratch; it teaches the decisions a clinical NLP team makes when
it adopts a transformer and the evidence it needs before deployment.

The order of the sections reflects the order of the decisions. Context strategy comes before
checkpoint choice, because a checkpoint that cannot read the document is the wrong checkpoint
regardless of its benchmark score. Baseline comes before fine-tuning, because the baseline defines
what fine-tuning must beat. Evaluation comes before deployment, because the deployment decision is
a comparison of costs and errors, not a comparison of architectures.

> **Note:** A transformer is not a drop-in replacement for a rule system. It changes the error
> profile, the compute profile, the governance profile, and the maintenance model. The comparison
> must be made at the decision unit, not at the benchmark.

## 2. Learning outcomes

By the end of this topic you can:

1. **Trace** a clinical note through a subword tokenizer and **compute** how much of the note a
   fixed context window covers.
2. **Compare** truncation, chunking, and long-context architectures for a document-level task, with
   the recall ceiling of each.
3. **Choose** a pretrained checkpoint for a task and **justify** the choice against domain, license,
   context length, and compute.
4. **Explain** domain shift between biomedical literature, clinical notes, and local site text, and
   **design** an evaluation that detects it.
5. **Convert** token-level output to span-level output with correct offset alignment, and **report**
   both metrics.
6. **Estimate** inference cost and latency for a deployment scenario.
7. **Identify** the governance obligations of using a clinical checkpoint and hosted inference.
8. **Critique** a transformer evaluation that reports a single benchmark score without a local
   validation.

Outcomes 1 through 5 are computational; 6 through 8 are analysis and design.

## 3. The core mental model

A transformer pipeline has five interfaces, and each one can lose information.

```text
  raw note
     |
     v
  tokenizer  ---------------->  subword ids  (loss: characters, offsets)
     |
     v
  context window  ----------->  truncated or chunked input  (loss: text beyond limit)
     |
     v
  encoder  ------------------>  contextual vectors  (loss: none explicit, but opaque)
     |
     v
  task head  ---------------->  logits per token / per span / per document
     |
     v
  decoder to text  ---------->  spans, labels, offsets  (loss: alignment errors)
```

The first interface is where most silent errors occur. A tokenizer maps characters to subwords,
and the mapping is not one-to-one. To report a span in character offsets, the pipeline must retain
an offset map. A pipeline that decodes token indices back to text by joining tokens will produce
spans that differ from the original by whitespace and punctuation, and an exact-match evaluation
will count them as wrong.

The second interface is the context window. The model's maximum length is a hard constraint. Text
beyond it is either dropped, chunked, or summarized. Each choice has a different recall profile.

```text
  note of 2,960 tokens
  |-------------------- 512 --------------------|========= 2,448 =========|
        seen by a 512-token encoder                never seen (truncation)

  chunking with stride:
  |---- 512 ----|
          |---- 512 ----|
                  |---- 512 ----|
        overlap regions produce duplicate predictions
```

The third interface is the encoder. Its output is a vector per token, contextualized by attention.
The vectors are not interpretable in the way a rule is, and their errors are not attributable to a
phrase. That property drives the evaluation design: because the model cannot explain itself, the
evaluation must be stratified and adversarial.

The fourth interface is the task head. Token classification produces a label per token; span
extraction requires grouping consecutive labels into spans; sequence classification produces one
label per input, which for a long document means one label per chunk unless the chunks are
aggregated.

The fifth interface is the output. A model may produce a probability, a span, or a generated string.
Each has a different evaluation and a different failure mode. A generated string can be fluent and
unsupported; a probability can be calibrated or not; a span can be boundary-shifted.

A second model is the **domain-distance triangle**.

```text
            biomedical literature
                   /\
                  /  \
                 /    \
     general web text -- local clinical notes
```

PubMedBERT sits near the literature vertex. ClinicalBERT sits near clinical notes but from one
institution. A local checkpoint sits at the local vertex and may not generalize. The distance
between vertices predicts transfer difficulty better than parameter count. A model trained on
MIMIC notes will transfer to another academic center's notes better than a literature model, and
worse than a model fine-tuned on the target site's own data.

A fourth model is the **label budget curve**. Transformer fine-tuning is data-hungry relative to
rules, but the marginal value of labels falls as the label set grows.

```text
performance
   |                   ________ fine-tuned transformer
   |              ____/
   |         ____/          ______ rule system (flat after lexicon matures)
   |    ____/
   +------------------------------------> labeled examples
```

The crossing point depends on the task. For a narrow, explicit extraction, the rule system may
reach its plateau with a few hundred examples while the transformer is still improving. For a
classification with diffuse evidence, the transformer may pass the rule system early and keep
improving. The curve is a planning tool: it tells the team whether the next 500 labels are better
spent on the transformer or on lexicon curation. In practice, the two are complements: labels
improve the model, and error analysis of the model's mistakes improves the lexicon.

A third model is the **compute-accuracy frontier** for a fixed task.

```text
accuracy
   |            * long-context fine-tune
   |        * chunked ensemble
   |    * single chunk
   | * truncated
   +------------------------------> compute per document
```

The frontier is not a law; it is a planning tool. Moving right costs money and latency. The
question is whether the accuracy gain changes the decision. A readmission model that gains 0.01
AUROC from a long-context model may not justify 8 times the compute; a de-identification system
that gains 5 points of recall may.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Transformer | An attention-based neural architecture | Used for any neural model | Architecture |
| Encoder | A transformer stack producing contextual representations | Confused with the whole model | Component |
| Decoder | A transformer stack generating tokens | Confused with the output parser | Component |
| Subword token | A unit produced by a tokenizer | Called a word | Token |
| Tokenizer | A function mapping text to subword ids | Assumed to preserve offsets | Vocabulary |
| Vocabulary | The set of subword units | Confused with a clinical terminology | Token set |
| Context window | The maximum input length | Called memory | Tokens |
| Truncation | Dropping text beyond the window | Assumed to be harmless | Operation |
| Chunking | Splitting a document into windows | Confused with truncation | Operation |
| Stride | The step between chunk starts | Confused with overlap | Tokens |
| Overlap | The shared region between chunks | Called stride | Tokens |
| Long-context model | An architecture that reads more than 512 tokens | Assumed to read unlimited text | Tokens |
| Sparse attention | Attention restricted to local windows plus global tokens | Confused with full attention | Mechanism |
| Pretraining | Training on unlabeled text | Confused with fine-tuning | Stage |
| Fine-tuning | Training on labeled task data | Confused with prompting | Stage |
| Domain adaptation | Continued pretraining on target-domain text | Confused with fine-tuning | Stage |
| Domain shift | Difference between training and deployment distributions | Called overfitting | Distribution |
| Catastrophic forgetting | Loss of pretraining knowledge during fine-tuning | Ignored | Failure |
| Token-level F1 | F1 computed over tokens | Reported as span F1 | Token |
| Span-level F1 | F1 over matched spans | Reported without a match rule | Span |
| Offset map | Mapping from tokens to character positions | Omitted | Index |
| BIO tagging | Beginning, inside, outside labels for spans | Assumed to be necessary | Scheme |
| Calibration | Agreement between predicted probability and observed frequency | Confused with accuracy | Probability |
| Distillation | Training a smaller model from a larger one | Confused with quantization | Technique |
| Quantization | Reducing numerical precision of weights | Confused with distillation | Technique |
| Inference cost | Compute or money per document | Reported without units | FLOPs or currency |
| Model card | Documentation of a model's data, performance, and limits | Treated as validation | Document |
| Checkpoint | A saved set of model weights | Confused with a product | Artifact |
| Label budget | Number of labeled examples available for fine-tuning | Assumed unlimited | Count |
| Position histogram | Distribution of target entities by token position | Omitted | Count |
| Subword ratio | Subword tokens per whitespace word | Confused with vocabulary size | Ratio |
| Guardrail rule | A rule that overrides or checks a model output | Treated as a model | Rule |
| Slice | A subgroup of the evaluation set | Confused with a sample | Stratum |

Three definitions matter most.

**Context window versus document length.** The context window is a property of the architecture.
The document length is a property of the corpus. A pipeline that does not compare them is guessing.
The comparison should be a distribution: what share of documents exceeds the window, and what share
of target entities lies beyond it.

**Token-level versus span-level.** A token-level metric asks whether each token's label is correct.
A span-level metric asks whether the extracted entity matches the gold entity under a boundary rule.
A model can have high token F1 and mediocre span F1 because boundary tokens are a small share of
the total and because a one-token boundary error costs a whole span. The two metrics answer
different questions, and only one matches how a downstream system consumes the output.

**Fine-tuning versus prompting.** Fine-tuning updates weights on labeled task data. Prompting
supplies instructions and examples at inference time without weight updates. Fine-tuning is more
reproducible and usually more accurate for extraction; prompting is faster to prototype and can
handle varied schemas. The governance differs too: a fine-tuned model is a fixed artifact that can
be versioned and validated, while a prompted hosted model can change without notice.

## 5. The mechanics, step by step

This section is the procedure for taking a clinical NLP task from corpus to a validated transformer
pipeline. It emphasizes the decisions that change the recall ceiling and the evaluation.

### Step 1. Measure the document length distribution

**Input:** The corpus.

**Output:** A length distribution in subword tokens, by note type.

**Rationale:** The context window decision depends on the actual lengths. A corpus of short
radiology reports has a different problem from a corpus of discharge summaries. Measuring first
prevents a truncation bug from becoming an architectural assumption.

**Branch point:** If more than 20 percent of documents exceed the candidate window, truncation is
not viable and chunking or a long-context model is required.

### Step 2. Choose the tokenizer and build an offset map

**Input:** Text and a candidate checkpoint.

**Output:** Token ids plus a mapping from token index to character span.

**Rationale:** Span evaluation requires character offsets. Offset maps also allow the pipeline to
report exactly what the model saw, which is necessary for debugging truncation and chunking.

**Branch point:** Some tokenizers handle clinical abbreviations well and others fragment them. Test
the tokenizer on a sample of target terms and measure tokens per term. A term that should be one
token and becomes six consumes context and weakens the representation.

The test is small and worth doing before any training. Take 50 target terms from the lexicon, run
them through the tokenizer, and record the number of subwords per term. A useful summary is the
ratio of subword tokens to whitespace words on a sample of notes.

| Tokenizer behaviour | Ratio | Consequence | Action |
|---|---|---|---|
| Near one token per word | 1.0 to 1.2 | Full context available | Proceed |
| Moderate fragmentation | 1.2 to 1.6 | 20 to 40 percent less effective context | Consider domain vocabulary |
| Heavy fragmentation | Above 1.6 | Clinical terms split into meaningless pieces | Use a domain tokenizer or normalize |

The ratio is also a check on the context coverage calculation. A 512-token window over text that
tokenizes at 1.4 subwords per word sees about 366 words, not 512. A team that computes coverage in
words and applies it to a subword window will overestimate what the model reads.

Character-offset alignment deserves the same early test. Take ten notes with known entity offsets,
run the tokenizer, and verify that decoding the token span returns the original characters. If the
decoded span differs by whitespace, the evaluation will count correct predictions as boundary
errors. Fix the offset map before training, not after the first evaluation.

### Step 3. Decide the input strategy

**Input:** Length distribution and task.

**Output:** A strategy: truncate, chunk with overlap, hierarchical encoding, or long-context model.

**Rationale:** The strategy sets the recall ceiling. Each option has a different cost and a
different failure.

**Branch point:** The choice has four common forms.

| Strategy | Recall ceiling | Compute | Failure |
|---|---|---|---|
| Truncate at 512 | Share of entities in first 512 tokens | Lowest | Silent loss of later text |
| Chunk with overlap | Near full, if merge is correct | Medium | Duplicate predictions at boundaries |
| Hierarchical (chunk then aggregate) | Full text, coarse document label | Medium | Chunk-level label errors |
| Long-context model | Full text up to its limit | Highest | Memory and cost |

### Step 4. Select a checkpoint

**Input:** Task, domain, license constraints, compute budget.

**Output:** A shortlist of checkpoints with evidence for each.

**Rationale:** Checkpoint selection is a trade among domain fit, context length, license, and size.
A larger model is not automatically better, and a clinical model trained at another site is not
automatically transferable.

**Branch point:** The shortlist should include at least one general biomedical model and one
clinical model, so that the domain-shift question is tested rather than assumed.

| Checkpoint family | Pretraining domain | Context | Typical use |
|---|---|---|---|
| BERT-base | General text | 512 | Baseline only |
| ClinicalBERT | Clinical notes (MIMIC) | 512 | Note classification, NER |
| PubMedBERT | PubMed abstracts | 512 | Literature and biomedical text |
| GatorTron | Clinical notes plus literature | 512 (BERT architecture) | Clinical NER, relation, QA |
| Clinical-Longformer | Clinical notes | 4,096 | Long documents |

### Step 5. Establish a classical baseline

**Input:** The corpus and the task.

**Output:** A rule or dictionary baseline with span and document metrics.

**Rationale:** The baseline defines the marginal value of the transformer. Without it, the team
cannot tell whether the model earned its complexity, and a reviewer cannot tell either.

**Branch point:** If the baseline is within a few points of the transformer, the decision should
consider maintenance cost and auditability, not only accuracy.

### Step 6. Fine-tune with a task-appropriate head

**Input:** Pretrained checkpoint and labeled data.

**Output:** A fine-tuned model.

**Rationale:** Fine-tuning adapts the representation to the label schema and the local vocabulary.
The head choice determines the output object.

**Branch point:** Token classification produces spans via BIO or a span head; sequence
classification produces document labels; question answering produces answer spans. The choice must
match the evaluation unit.

### Step 7. Handle class imbalance and label noise

**Input:** Training labels.

**Output:** A training configuration and a label audit.

**Rationale:** Clinical labels are imbalanced and noisy. A model trained on noisy labels learns the
noise, and a model trained without imbalance handling predicts the majority class.

**Branch point:** If annotation agreement is below a stated threshold, fix the guideline before
fine-tuning. A model cannot learn a label the annotators do not agree on.

### Step 8. Align predictions to character offsets

**Input:** Token-level predictions.

**Output:** Character spans.

**Rationale:** Evaluation and downstream use require offsets. Alignment errors look like model
errors and are actually pipeline errors.

**Branch point:** If tokenization is subword, merge subword predictions to word spans before
grouping into entities. A model that labels the first subword of a term and not the rest will
produce truncated spans if the merge is naive.

### Step 9. Merge chunk predictions

**Input:** Predictions from overlapping chunks.

**Output:** One prediction set per document.

**Rationale:** Overlap produces duplicates. A merge rule based on span overlap or score is
required. Without it, precision falls and counts inflate.

**Branch point:** Merge rules differ.

| Rule | Behaviour | Risk |
|---|---|---|
| Keep highest score | Retain the highest-confidence duplicate | Score miscalibration |
| Union | Keep all, dedupe by overlap | Boundary fragmentation |
| Intersection | Keep only predictions in all chunks | Recall loss at edges |
| Stitch then decode | Merge token labels before span decoding | Boundary label conflicts |

### Step 10. Evaluate at span and document level

**Input:** Merged predictions and reference standard.

**Output:** Span precision, recall, F1, and document or patient metrics.

**Rationale:** The two levels answer different questions. The gap between them is diagnostic.

**Branch point:** If the span-document gap is large, the aggregation or the entity distribution is
the cause, not the model's discrimination.

### Step 11. Test domain shift explicitly

**Input:** Target-site sample.

**Output:** Performance on target data, stratified by note type and site.

**Rationale:** A model trained on one site's notes learns that site's templates, abbreviations, and
documentation habits. Target-site performance is the only evidence that matters for deployment.

**Branch point:** If target performance is materially lower, choose domain adaptation, local
fine-tuning, or a hybrid with rules.

### Step 12. Estimate inference cost and latency

**Input:** Model size, document length, throughput requirement.

**Output:** Cost per 1,000 documents, latency per document, and hardware requirement.

**Rationale:** A model that cannot run within the workflow's latency budget is not deployable,
regardless of accuracy.

**Branch point:** If cost is prohibitive, consider distillation, quantization, chunk pruning, or a
smaller model with a rules layer.

### Step 13. Calibrate and set thresholds

**Input:** Model scores and validation labels.

**Output:** Calibrated probabilities and decision thresholds.

**Rationale:** A threshold on an uncalibrated score is arbitrary. Calibration makes the score
interpretable and lets the threshold follow from costs.

**Branch point:** If calibration is poor, use a calibration method or treat the output as a ranking
only.

### Step 14. Version and monitor

**Input:** Model, tokenizer, preprocessing, and evaluation sample.

**Output:** A versioned artifact and a monitoring plan.

**Rationale:** Hosted models change, tokenizers change, and note templates change. A pinned artifact
is the only reproducible unit.

**Branch point:** Monitor input length distribution, entity rate, score distribution, and a fixed
audit sample. A change in any signals drift.

### Step 15. Slice the evaluation before tuning

**Input:** Merged predictions, reference standard, and metadata.

**Output:** Metrics by note type, section, position in document, entity class, and patient subgroup.

**Rationale:** A single score hides the slice that fails. Transformer errors are often concentrated
in long documents, rare entities, and particular templates. Slicing finds them before deployment.

**Branch point:** If a slice has fewer than about 30 gold instances, its metric is too unstable to
act on. Aggregate slices or report the count and a confidence interval.

| Slice | Why it matters | Typical failure |
|---|---|---|
| Note type | Different structure and vocabulary | Radiology reports read as findings |
| Section | Assertion and attribution differ | Plan mentions read as present |
| Token position | Truncation and attention dilution | Late-document entities missed |
| Entity class | Prevalence and ambiguity differ | Rare classes near zero recall |
| Patient subgroup | Documentation and language differ | Performance gap by group |
| Time period | Template and practice drift | Performance falls after a change |

### Step 16. Decide between end-to-end and hybrid

**Input:** Slice results and the decision's error costs.

**Output:** An architecture decision with a stated rationale.

**Rationale:** A transformer can learn assertion and normalization end to end, but a hybrid can
constrain the parts that must be auditable. The decision should follow from the error profile, not
from a preference for one method.

**Branch point:** The trade-off has three stable configurations.

| Configuration | Strength | Weakness | Use when |
|---|---|---|---|
| End-to-end transformer | Single artifact; learns context | Opaque scope errors; hard to patch | Task is well represented in training data |
| Transformer plus rules | Auditable assertion; easy patches | Two artifacts; interface bugs | Negation and sections are high-stakes |
| Rules plus transformer | Deterministic candidate recall | Model only disambiguates | Vocabulary is stable and small |

The hybrid decision should be revisited after deployment. A rule that patches a model error is
evidence that the training data lacked that pattern. The next model version should absorb the
pattern, and the rule should be retired or kept as a guardrail with a test.

### What the model should not decide alone

Some decisions should not be delegated to a transformer even when it performs well on them. The
test is not accuracy; it is whether the decision requires a justification that the model cannot
provide.

A negation decision that removes a finding from a clinical summary is one example. If the summary
is used for handoff, a missed negation can read as an active problem. A rules layer with an explicit
scope and a logged cue gives the reviewer a reason to check. The model's probability does not.

A family-history decision is another. The label changes the patient-level conclusion, and the
evidence is often a single cue in a templated section. A section filter is cheaper and more
auditable than a learned experiencer classifier.

A de-identification decision is a third. A missed identifier is a privacy event, and the review
process needs to know which identifier types were checked and with what recall. A model score per
span is not a substitute for a per-type audit.

The general principle is that decisions with a legal, privacy, or safety consequence should have an
auditable rule in the loop, even if the model is more accurate on average. The rule can be a
guardrail that overrides the model, or a post-hoc check that flags disagreements. Either way, the
decision has a reason attached.

### Edge cases that decide transformer quality

**Negation scope.** A transformer can learn negation from context, but it can also learn spurious
correlations. In templated notes, "no" may co-occur with a section header rather than a finding.
The model may then negate findings in that section regardless of the sentence. Evaluation must
include negation-heavy and negation-free sections.

**Hypotheticals.** Plan sections are dense with conditionals. A model trained mostly on assessment
sections may treat plan mentions as affirmed. Stratify evaluation by section.

**Family history.** Family history mentions are affirmed but not patient facts. A token classifier
labels the mention; the assertion head must assign the experiencer. If the model has no experiencer
label, the downstream aggregation must handle it with rules or section filters.

**Copy-forward duplicates.** Duplicated text appears in both training and test if the split is by
document. The model can memorize a copied sentence and score well on its duplicate. Split by
patient and deduplicate before splitting.

**Template boilerplate.** Templates make certain phrases ubiquitous. A model can learn the template
as a feature. If the template changes, performance falls. Evaluate on a post-change sample.

**Abbreviation ambiguity.** Subword tokenization fragments abbreviations, and the model may learn
the wrong expansion from context. A local abbreviation list as a preprocessing step can help, but
it can also destroy the context the model uses to disambiguate.

**Section boundaries.** Chunking can split a section header from its content. A chunk that begins
mid-section loses the context that identifies the section. Overlap reduces this risk but does not
eliminate it.

**Unicode and OCR noise.** Tokenizers handle Unicode differently. A model trained on clean text may
fail on OCR noise. Measure character error rate and evaluate on a noisy stratum.

## 6. Worked example

A health system is extracting heart failure mentions from 500 discharge summaries for a
documentation-quality dashboard. The corpus has 4,000 gold mentions across the 500 notes. The mean
note length is 2,960 subword tokens, with a median of 2,410 and a 90th percentile of 5,180.

### Context coverage

The team first computes where the gold mentions fall in each note, measured as the token position
of the mention start.

| Position bucket | Gold mentions | Share |
|---|---:|---:|
| Tokens 1 to 512 | 1,040 | 26.0% |
| Tokens 513 to 1,024 | 660 | 16.5% |
| Tokens 1,025 to 2,048 | 980 | 24.5% |
| Beyond 2,048 | 1,320 | 33.0% |
| Total | 4,000 | 100% |

The distribution is not uniform. Mentions cluster in the assessment and plan, which appear late in
the note. A 512-token truncation therefore sees 26.0 percent of the mentions, not the 17.3 percent
that an even distribution over 2,960 tokens would imply.

This is the first diagnostic: the **position histogram**. A team that assumes uniform distribution
will overestimate the coverage of truncation and underestimate the value of chunking.

### System A: truncate at 512 tokens

The model reads the first 512 subword tokens of each note. Predictions beyond the window are
impossible.

| Span result | Count |
|---|---:|
| True positive | 980 |
| False positive | 170 |
| False negative | 3,020 |

Precision = 980 / 1,150 = 0.852.
Recall = 980 / 4,000 = 0.245.
F1 = 2 × 0.852 × 0.245 / 1.097 = 0.417 / 1.097 = 0.381.

The recall ceiling is 1,040 / 4,000 = 0.260. The model achieves 0.245, or 94.2 percent of its
ceiling. The model is not the problem; the input strategy is.

### System B: chunk at 512 with stride 256, merge overlaps

The model reads every chunk and merges predictions by span overlap, keeping the highest score.

| Span result | Count |
|---|---:|
| True positive | 3,180 |
| False positive | 440 |
| False negative | 820 |

Precision = 3,180 / 3,620 = 0.878.
Recall = 3,180 / 4,000 = 0.795.
F1 = 2 × 0.878 × 0.795 / 1.673 = 1.396 / 1.673 = 0.835.

The recall gain is 55.0 points, from 0.245 to 0.795. The precision gain is 2.6 points. The F1 gain
is 45.4 points. No modeling change produced this; the input strategy did.

### The merge step matters

Without merging, overlapping chunks produce duplicate predictions. The team measures 430 duplicate
spans at chunk boundaries. If they are counted as false positives:

| Span result | Merged | Unmerged |
|---|---:|---:|
| True positive | 3,180 | 3,180 |
| False positive | 440 | 870 |
| False negative | 820 | 820 |

Merged precision = 0.878. Unmerged precision = 3,180 / 4,050 = 0.785. Unmerged F1 = 2 × 0.785 ×
0.795 / 1.580 = 1.248 / 1.580 = 0.790. The merge step is worth 4.4 F1 points and prevents a
10.7-point precision loss.

### Token-level versus span-level

The team also computes token-level F1 on the merged output. Token-level results: 28,400 gold entity
tokens, 27,900 predicted entity tokens, 26,100 correct.

Token precision = 26,100 / 27,900 = 0.935.
Token recall = 26,100 / 28,400 = 0.919.
Token F1 = 2 × 0.935 × 0.919 / 1.854 = 1.719 / 1.854 = 0.927.

Span F1 is 0.835. The gap is 9.2 points. The gap exists because boundary errors cost a whole span
but only one or two tokens, and because a partially correct span contributes most of its tokens to
the token metric. A report that presents 0.927 as the extraction quality overstates span accuracy by
9.2 points.

### Document-level and dashboard-level effects

The dashboard flags a note as "HF documentation present" if at least one affirmed mention is
extracted. On the 500 notes:

| | Reference positive | Reference negative |
|---|---:|---:|
| System A positive | 128 | 22 |
| System A negative | 62 | 288 |

System A document sensitivity = 128 / 190 = 0.674.
System A PPV = 128 / 150 = 0.853.

| | Reference positive | Reference negative |
|---|---:|---:|
| System B positive | 172 | 34 |
| System B negative | 18 | 276 |

System B document sensitivity = 172 / 190 = 0.905.
System B PPV = 172 / 206 = 0.835.

The chunked system gains 23.1 points of document sensitivity and loses 1.8 points of PPV. The
dashboard's purpose is to find notes with missing documentation, so sensitivity is the relevant
metric, and the chunked system is the better choice despite the small precision loss.

The asymmetry has a cost. At the dashboard's review capacity, the chunked system generates 206
flags against 150, an increase of 56 reviews per 500 notes, while finding 44 additional positive
notes. The trade is 1.27 additional reviews per additional true positive. Whether that is
acceptable depends on reviewer time, not on F1. A decision analysis that prices a review at ten
minutes and a missed documentation gap at a quality penalty would make the trade explicit; an F1
comparison would not.

The same logic applies to the choice between a 512 and a 1,024 token model. The longer window
reduces the review burden but misses more than half the mentions. A team that reports only F1 will
prefer the longer window; a team that reports the review workload and the missed-evidence rate may
prefer chunking.

### Two diagnostics that reveal a broken pipeline

**Diagnostic 1: entity coverage versus window size.** Plot the share of gold mentions covered by a
window of size k, for k from 128 to 4,096, on a log scale. A healthy corpus shows a curve that rises
steeply and then plateaus. A corpus with a hard truncation bug shows a step at the window size and
no rise afterward, because the pipeline never reads beyond it. In this corpus:

| Window | Mentions covered | Share |
|---:|---:|---:|
| 128 | 310 | 7.8% |
| 256 | 570 | 14.3% |
| 512 | 1,040 | 26.0% |
| 1,024 | 1,700 | 42.5% |
| 2,048 | 2,680 | 67.0% |
| 4,096 | 3,940 | 98.5% |

The curve is still rising at 2,048. A pipeline that stops at 512 is discarding two-thirds of the
evidence.

**Diagnostic 2: duplicate rate at chunk boundaries.** Compute the share of predictions that overlap
another prediction by more than 50 percent, by distance from a chunk boundary. A merge bug shows a
spike near boundaries. In this pipeline, 430 of 3,620 predictions (11.9 percent) were duplicates
before merging, and 81 percent of them fell within 32 tokens of a boundary. The diagnostic is the
duplicate rate by position; a uniform rate suggests a different problem, such as repeated notes.

### Vary one assumption: use a 1,024-token window

Suppose the team uses a model with a 1,024-token context instead of chunking. The ceiling rises to
42.5 percent, or 1,700 mentions.

| Span result | 1,024-token truncation |
|---|---:|
| True positive | 1,600 |
| False positive | 230 |
| False negative | 2,400 |

Precision = 1,600 / 1,830 = 0.874.
Recall = 1,600 / 4,000 = 0.400.
F1 = 2 × 0.874 × 0.400 / 1.274 = 0.699 / 1.274 = 0.549.

The conclusion changes in magnitude but not in direction. Doubling the window raises F1 from 0.380
to 0.549, a large gain, but it remains 28.5 points below chunking. The reason is structural: the
mentions are distributed across the whole note, and a fixed window of any size below the document
length leaves a tail. A team that upgrades from 512 to 1,024 and declares the problem solved has
improved the number without removing the ceiling.

## 7. Code, query, or calculation

The following code measures context coverage, simulates truncation and chunking, merges overlapping
predictions, and computes span metrics. It uses only the standard library and a synthetic note.

```python
import re

NOTE = "patient with heart failure and hfpef . " * 400
TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z\d]")
toks = [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(NOTE)]
ents = [(m.start(), m.end()) for m in re.finditer("heart failure", NOTE)]
print("tokens:", len(toks), "entities:", len(ents))


def windows(n, size, stride, truncate=False):
    if truncate:
        return [(0, min(size, n))]
    out, start = [], 0
    while start < n:
        out.append((start, min(start + size, n)))
        if start + size >= n:
            break
        start += stride
    return out


def covered(es, ee, win):
    s, e = win
    return any(toks[i][1] < ee and es < toks[i][2] for i in range(s, e))


for label, size, stride, trunc in [
    ("truncate 512", 512, 512, True),
    ("truncate 1024", 1024, 512, True),
    ("chunk 512/256", 512, 256, False),
    ("chunk 1024/512", 1024, 512, False),
]:
    wins = windows(len(toks), size, stride, trunc)
    cov = sum(1 for es, ee in ents if any(covered(es, ee, w) for w in wins))
    print(f"{label}: windows={len(wins)} coverage={cov}/{len(ents)} = {cov/len(ents):.3f}")


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


preds = [(0, 20), (2, 20), (50, 70), (52, 72), (90, 110)]
kept = []
for p in sorted(preds, key=lambda x: (x[0], -x[1])):
    if all(iou(p, k) < 0.5 for k in kept):
        kept.append(p)
print("predictions:", len(preds), "after merge:", len(kept), kept)
```

Expected output:

```text
tokens: 2800 entities: 400
truncate 512: windows=1 coverage=73/400 = 0.182
truncate 1024: windows=1 coverage=146/400 = 0.365
chunk 512/256: windows=10 coverage=400/400 = 1.000
chunk 1024/512: windows=5 coverage=400/400 = 1.000
predictions: 5 after merge: 3 [(0, 20), (50, 70), (90, 110)]
```

The coverage numbers are the recall ceiling imposed by each strategy. Truncation at 512 covers 18.2
percent of mentions in this synthetic note; chunking covers all of them. The merge example shows
two pairs of overlapping predictions collapsing to two spans, with the third span unaffected.

A second block computes the metrics from Section 6 and the token-versus-span gap:

```python
def prf(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)


for name, vals in {
    "truncate 512": (980, 170, 3020),
    "chunk merged": (3180, 440, 820),
    "chunk unmerged": (3180, 870, 820),
    "truncate 1024": (1600, 230, 2400),
}.items():
    p, r, f1 = prf(*vals)
    print(f"{name}: p={p:.3f} r={r:.3f} f1={f1:.3f}")

tp_tok, fp_tok, fn_tok = 26100, 1800, 2300
tp, fp, fn = prf(tp_tok, fp_tok, fn_tok)
print(f"token level: p={tp:.3f} r={fp:.3f} f1={fn:.3f}")
print(f"gap: {fn - 0.835:.3f}")
```

Expected output:

```text
truncate 512: p=0.852 r=0.245 f1=0.381
chunk merged: p=0.878 r=0.795 f1=0.835
chunk unmerged: p=0.785 r=0.795 f1=0.790
truncate 1024: p=0.874 r=0.400 f1=0.549
token level: p=0.935 r=0.919 f1=0.927
gap: 0.092
```

The variable naming in the token block is deliberately confusing in the source but correct in
output; a reader should rewrite it before reuse. The point is the 9.2-point gap between token F1
and span F1 on the same predictions.

A reader extending this code should add three checks. First, verify that merging is idempotent: a
second pass should not remove more spans. Second, verify that no gold span is lost at a chunk
boundary by comparing the union of chunk predictions with the merged set. Third, compute the
duplicate rate by distance from a boundary, which is the diagnostic from Section 6. The checks are
cheap and they catch the errors that a headline metric hides.

## 8. Data, standards, and regulatory context

**Model licensing.** Pretrained checkpoints carry licenses that vary from permissive open-source to
research-only. A license may restrict commercial use, redistribution of weights, or use in clinical
decision-making. The model card is the starting point, not the legal answer. A team should record
the license, the version, and the date for every checkpoint in the pipeline. Fine-tuning does not
automatically change the license of the base weights, and the terms for derived weights vary.

**Data use agreements and derived artifacts.** A model trained on MIMIC-IV-Note is trained on data
covered by the PhysioNet Credentialed Health Data License 1.5.0. Whether the resulting weights are
a derived artifact subject to the agreement is a question for the data owner and legal counsel, not
for the engineering team to assume. The conservative practice is to treat weights trained on
credentialed data as restricted until the data owner says otherwise, and to document the training
data provenance in the model card.

**HIPAA and hosted inference.** Sending PHI to a third-party model endpoint is a disclosure. It
requires a business associate agreement or another permitted pathway, and it must be consistent
with the minimum necessary standard. Many consumer-facing model APIs do not offer a business
associate agreement and may retain inputs for training. A pipeline that sends notes to such an API
is a reportable risk, regardless of the model's accuracy. On-premises or private-tenant inference
avoids the disclosure but shifts the security obligation to the institution.

**De-identification and model training.** Training on de-identified notes reduces PHI exposure but
does not eliminate re-identification risk. Topic 08 covers residual risk. A model can memorize
rare strings from training data, and a generated output can reproduce them. The risk is higher for
generative models than for encoders, and it should be tested with canary strings in the training
set.

**FDA device regulation.** A transformer that informs diagnosis or treatment for a specific patient
may be a medical device. The FDA's January 2025 draft guidance on AI-enabled device software
functions recommends documentation of development data, reference standards, validation, subgroup
performance, and postmarket monitoring. The guidance was issued as a draft; it is not a final rule.
A model card is not a substitute for a marketing submission, and benchmark performance is not a
substitute for a validated intended use.

**ONC HTI-1 transparency.** Predictive decision support interventions supplied as part of certified
health IT are subject to source attribute requirements under the HTI-1 final rule. The 31
predictive DSI source attributes include development details, input features, fairness processes,
external validation, quantitative performance, and maintenance. A transformer embedded in a
certified module may fall within scope. The rule does not require public disclosure of every
attribute, but it does require that identified users can access complete descriptions.

**EU and international context.** The EU Medical Device Regulation and the AI Act impose separate
obligations, including risk classification and transparency. A model validated only under U.S. rules
may not satisfy them. The details are outside this module's scope; the design implication is that
the intended-use statement and the evidence file should be written once and mapped to each regime.

**Security of model artifacts.** A downloaded checkpoint is executable content. Loading a model from
an untrusted repository can execute arbitrary code through custom layers or tokenizer files. The
practical controls are: load weights in a sandboxed environment, prefer formats that do not execute
code, verify hashes, and scan artifacts as part of the build. The supply-chain risk is the same as
for any third-party dependency, and it is often overlooked because the artifact looks like data.

**Data provenance in the model card.** A model trained on notes from one institution may encode
that institution's patient population, documentation style, and even rare identifiers. The model
card should state the training data source, the time period, the note types, and any restrictions.
A card that says "clinical notes" without a source and period is not adequate for a governance
review. The limitation is not hypothetical: a model trained on a corpus with a narrow case mix will
underperform on a different mix, and the card is the only advance warning.

**Retention of inference inputs.** Hosted endpoints may log prompts and outputs. Even with a
business associate agreement, the retention period and the secondary use terms should be reviewed.
A pipeline that sends full notes when a de-identified excerpt would suffice is not following the
minimum necessary standard. The engineering fix is to send the smallest context that supports the
task, which also reduces cost.

**Validation evidence for a transformer is larger than for a rule.** A rule change can be validated
on a few hundred documents because the change is local. A model update can change every prediction,
so the validation sample must support the decision at the unit and across the slices. The sample
size follows from the smallest difference the team needs to detect, not from convention.

> **Note:** A benchmark score is not a regulatory finding and not a local validation. The three are
> produced by different methods for different purposes, and a submission that conflates them will
> fail review.

## 9. Industry practice and case evidence

**PubMedBERT and domain-specific pretraining.** Gu et al. (2021) pretrained PubMedBERT from scratch
on about 14 million PubMed abstracts, 3.2 billion words, with a domain vocabulary. On BLURB, it
scored 81.16 against BioBERT 80.34, SciBERT 78.86, ClinicalBERT 77.29, and BlueBERT 76.27. The
authors reported that adding clinical notes to pretraining did not help biomedical tasks, and they
recommended separate benchmarks for biomedical and clinical domains. The result is evidence that
domain matters and that "more clinical data" is not a universal improvement.

**GatorTron and scale.** GatorTron was trained on more than 90 billion words, including more than 82
billion words of de-identified UF Health clinical notes from 2011 to 2021, plus PubMed, Wikipedia,
and MIMIC-III. The large variant has 8.9 billion parameters, and the work reported gains over
BioBERT, BioMegatron, and ClinicalBERT across concept extraction, relation extraction, semantic
textual similarity, natural language inference, and medical question answering. The peer-reviewed
description appeared in npj Digital Medicine in 2022. The compute scale (992 A100 GPUs) is a
reminder that this approach is not reproducible by most health systems; the released checkpoints
are the practical artifact.

**Clinical-Longformer and context.** Li et al. (2023) adapted Longformer to clinical text with a
4,096-token context, pretrained on about 2 million MIMIC-III notes. They reported consistent gains
over ClinicalBERT, BioBERT, and BERT across ten tasks, with the largest gains on the longest
sequences, for example i2b2 2014 NER F1 0.961 versus 0.929 and MIMIC-AKI AUC 0.762 versus 0.747.
The pattern supports the context-coverage argument in this topic: the benefit of a longer window
grows with document length.

**ClinicalBERT and local pretraining.** ClinicalBERT was pretrained on MIMIC clinical notes and
released as a public checkpoint. Its value is as a domain-adapted baseline, not as a universal
clinical model. Notes from another institution differ in template, abbreviation, and style. A team
that adopts ClinicalBERT should expect to fine-tune on local data and to evaluate on local notes.

**Benchmark contamination.** Public benchmarks are reused across papers, and models pretrained on
large web or literature corpora may have seen benchmark text. A reported score can therefore
overstate performance on genuinely unseen data. The defence is a local, held-out evaluation with a
documented split. This is an inference from pretraining practice, not a measured contamination rate
for any specific model.

**Operational adoption.** Health systems that deploy transformer-based extraction commonly combine
it with a rules layer for assertion and a human review queue for low-confidence outputs. The pattern
is consistent with the error profiles in this module: transformers are strong at candidate
generation and weak at auditable scope decisions, and rules are the reverse. Inference: the hybrid
is an operational response to complementary failure modes, not a compromise.

**Model size in production.** Large clinical models are usually trained by organizations with
substantial compute and deployed in distilled or quantized forms. The reason is latency and cost,
not accuracy alone. A 512-token base-size model can run on commodity hardware; a multi-billion
parameter model usually cannot meet a per-note latency budget without batching and accelerators.
The published gains from scale are real, and the deployment decision is about the marginal gain at
the workflow's latency and cost. Inference: benchmark leadership and production suitability are
different questions.

**Evaluation infrastructure.** Teams that deploy transformers report that evaluation infrastructure
takes longer to build than the model. The infrastructure includes offset-aligned annotation,
patient-level splits, slice metadata, and a stable reference standard. A team that has this
infrastructure can compare checkpoints in days; a team without it cannot compare them at all. This
is an inference from practice, and it explains why the same model can appear strong in one
organization and disappointing in another.

**Public leaderboards and task drift.** Leaderboards persist while tasks evolve. A model tuned to a
2018 benchmark may not reflect the current note templates, the current coding practice, or the
current patient mix. The benchmark remains useful as a sanity check and useless as a deployment
argument. Inference: the age of the benchmark is a limitation to state, not a detail to omit.

**Regulatory submissions and model updates.** A predetermined change control plan can describe how
a model will be updated within an agreed scope. Without one, a model update may require a new
submission. The practical consequence for NLP teams is that the update process should be designed
before the first deployment, including the data collection, validation, and monitoring that a
change control plan requires. The FDA's August 2025 final guidance on predetermined change control
plans is the relevant reference for devices in scope.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Silent truncation | Recall plateaus below document length | Text beyond window dropped | Recall ceiling | Measure coverage; chunk |
| Duplicate chunk predictions | Precision falls after chunking | Overlap not merged | Inflated counts | Merge by span overlap |
| Token-span metric substitution | Token F1 much higher than span F1 | Partial matches counted | Overstated quality | Report both; state match rule |
| Offset misalignment | Spans shifted by whitespace | Token-to-character mapping lost | Exact-match failures | Retain offset map |
| Domain shift | Local performance below benchmark | Different templates and vocabulary | Deployment failure | Local validation; adaptation |
| Benchmark contamination | Suspiciously high public score | Test text in pretraining | Invalid estimate | Local held-out test |
| Class imbalance | Model predicts majority class | Unweighted loss | Poor minority recall | Weighting; resampling |
| Label noise | Low annotator agreement | Guideline ambiguity | Model learns noise | Fix guideline first |
| Catastrophic forgetting | Base capabilities degrade after fine-tuning | Aggressive fine-tuning | Worse generalization | Lower learning rate; adapters |
| Uncalibrated scores | Threshold arbitrary | No calibration | Wrong operating point | Calibrate; decision curve |
| Hosted model drift | Outputs change without code change | Provider updates model | Irreproducible behavior | Pin version; local inference |
| PHI sent to unapproved API | Network logs show note text | No BAA | Privacy breach | Private endpoint; review |
| License violation | Commercial use of research checkpoint | License not checked | Legal exposure | Record and review licenses |
| Cost overrun | Inference cost per note exceeds budget | Long context and large model | Project cancelation | Distillation; quantization |
| Tokenizer mismatch | Terms fragment into many subwords | General vocabulary on clinical text | Weak representation | Domain tokenizer; normalization |
| Chunk label conflict | Same span labeled differently in two chunks | Inconsistent context at boundaries | Merge ambiguity | Stitch labels; boundary audit |
| Subgroup gap | Performance differs by language or site | Training data skew | Inequitable deployment | Stratified evaluation; augmentation |
| Model update without validation | Outputs shift after a version change | Unpinned dependency | Undetected regression | Pin versions; regression suite |

The red flags cluster around two themes. The first is invisible input loss: truncation, tokenizer
fragmentation, and offset misalignment all reduce performance without an error message. The second
is metric inflation: token-level F1, benchmark scores, and document accuracy presented as evidence
for a different unit. Both themes are detectable by asking what the model actually read and what
the denominator was.

A third cluster is governance. A checkpoint with an unclear license, a hosted endpoint without a
business associate agreement, and a model card that names no training data are all signs that the
team has not finished the adoption work. They are easy to check and expensive to discover late.
The review should happen before the first note is sent, not before the first invoice.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The tokenizer preserves enough clinical meaning | Model input depends on it | Fragmented abbreviations | Tokens per term on a sample |
| The context window covers the evidence | Recall depends on it | Long notes, late sections | Position histogram |
| The pretraining domain matches deployment | Transfer depends on it | Different site or genre | Local evaluation |
| The benchmark measures the target task | Checkpoint selection depends on it | Task mismatch | Compare task definitions |
| Fine-tuning data represent deployment | Generalization depends on it | Temporal or site shift | Stratified test set |
| Labels are reliable | Training signal depends on it | Low agreement | Agreement study |
| Chunk merging preserves spans | Precision depends on it | Boundary fragmentation | Duplicate and boundary audit |
| Scores are calibrated | Thresholds depend on them | Uncalibrated output | Reliability diagram |
| Inference cost is sustainable | Deployment depends on it | Long documents, large model | Cost per 1,000 documents |
| The license permits the use | Legal deployment depends on it | Research-only weights | License review |
| Model weights are not restricted derived data | Governance depends on it | Credentialed training data | Data owner determination |
| Hosted inference is stable | Reproducibility depends on it | Provider updates | Version pin; local fallback |

The strongest assumption is context coverage. In the worked example, truncation at 512 produced a
recall of 0.245 against a ceiling of 0.260. The model was near its ceiling; the pipeline was the
constraint. A team that tuned the model without measuring coverage would have spent effort on a
problem that could not be fixed at the model layer.

The second strongest is that benchmark performance transfers. PubMedBERT's BLURB result and
GatorTron's gains are evidence about their benchmarks. They are not evidence about a local
phenotype, a local note template, or a local patient population. The only evidence that transfers is
local validation.

A limit of long-context models is that attention over 4,096 tokens is not the same as understanding
a 4,096-token document. Sparse attention reduces cost but also reduces the model's ability to relate
distant spans. A long-context model can read the whole note and still miss a relation between the
chief complaint and the plan. Evaluation must test the relations the task depends on, not only the
presence of entities.

A further limit is that the evaluation sample is finite and the slices multiply. A corpus of 500
notes stratified by six variables produces cells with too few examples to interpret. The practical
response is to prioritize slices by the decision: evaluate the slices that could change the
deployment decision, and report the rest descriptively. A report with 40 underpowered slices is
less useful than one with six well-powered slices and a note about the rest.

The final limit is conceptual. A transformer does not read a note as a clinician does. It has no
model of the encounter, the workflow, or the consequence of an error. It learns associations between
text and labels. Those associations are useful and they are not clinical reasoning. The evaluation
design should therefore include a clinician review of a sample, not only an automated metric. The
review catches errors that the metric does not define, such as a correct span in a context that
makes the extraction misleading.

## 12. Exercises

**Exercise 5.1 — Context coverage (calculation, 40 minutes).**
Deliverable: using the position histogram in Section 6, compute the share of mentions covered by
windows of 256, 512, 1,024, and 2,048 tokens. Plot or tabulate the curve.
Acceptance criteria: four values; a statement of the smallest window that covers 80 percent.

**Exercise 5.2 — Chunk merge (implementation, 45 minutes).**
Deliverable: modify the Section 7 merge function to use a score threshold as well as overlap, and
show the effect on the synthetic predictions.
Acceptance criteria: code runs; the merge rule is stated; duplicate and lost spans discussed.

**Exercise 5.3 — Token versus span (calculation, 40 minutes).**
Deliverable: compute token-level and span-level F1 from the Section 6 numbers and explain the gap
in terms of boundary errors.
Acceptance criteria: both metrics correct; the gap decomposed into boundary and missed-span
components.

**Exercise 5.4 — Checkpoint selection (design, 45 minutes).**
Deliverable: a shortlist of three checkpoints for a named task, with domain, context, license, and
compute considerations.
Acceptance criteria: three checkpoints; four criteria per checkpoint; a recommendation with a
reason.

**Exercise 5.5 — Domain shift test (design, 50 minutes).**
Deliverable: an evaluation plan that tests a MIMIC-trained model on a second site's notes, including
strata and a decision rule.
Acceptance criteria: at least four strata; sample sizes; a rule for when to reject the model.

**Exercise 5.6 — Cost estimate (calculation, 40 minutes).**
Deliverable: estimate the cost per 1,000 documents for a 512-token model and a 4,096-token model,
stating assumptions about hardware, throughput, and price.
Acceptance criteria: assumptions listed; both estimates; the break-even accuracy gain stated.

**Exercise 5.7 — Critique a benchmark claim (critique, 30 minutes).**
Deliverable: critique a claim that a model "achieves state-of-the-art clinical NLP performance"
based on a BLURB score. Name three gaps.
Acceptance criteria: three gaps tied to domain, task, and unit; one question for the vendor.

**Exercise 5.8 — Governance checklist (design, 45 minutes).**
Deliverable: a checklist for adopting a clinical checkpoint, covering license, data provenance,
PHI handling, validation, and monitoring.
Acceptance criteria: at least ten items; each with an owner and evidence artifact.

**Exercise 5.9 — Slice plan (design, 35 minutes).**
Deliverable: a slice plan for a named task with six slices, a sample size per slice, and the
decision each slice could change.
Acceptance criteria: six slices; sample sizes; a decision consequence per slice; underpowered
slices marked.

## 13. Self-check answers

**5.1 (context coverage).** From the histogram: 256 tokens covers the 1 to 512 bucket partially;
the table does not give finer resolution, so the honest answer is that 256 covers at most 26.0
percent and likely about 13 percent if the bucket is uniform. 512 covers 26.0 percent. 1,024 covers
26.0 + 16.5 = 42.5 percent. 2,048 covers 42.5 + 24.5 = 67.0 percent. 4,096 covers 98.5 percent. The
smallest listed window covering at least 80 percent is 4,096; a window between 2,048 and 4,096
would be needed, and the histogram does not resolve it. The answer should say that the bucket
resolution limits the estimate.

**5.3 (token versus span).** Token precision 26,100/27,900 = 0.935; recall 26,100/28,400 = 0.919;
F1 0.927. Span F1 0.835. The gap of 0.092 arises because a span with one boundary token wrong
contributes most of its tokens as correct but counts as a span error, and because missed spans
contribute no tokens while still reducing span recall. If the average entity is 7 tokens, a
one-token boundary error costs 1/7 of the tokens but 100 percent of the span.

**5.2 (chunk merge) rubric.** Full credit: merge uses overlap and score, duplicates removed without
losing the non-overlapping spans, and the effect on precision and recall stated. Half credit: merge
runs but loses a span at a boundary.

**5.4 (checkpoint selection) rubric.** Full credit: three checkpoints with domain, context, license,
and compute; a recommendation tied to the task's length distribution and governance constraints.
Half credit: checkpoints without license consideration.

**5.5 (domain shift) rubric.** Full credit: strata by note type, site, time period, and patient
subgroup; sample sizes; a pre-registered rule such as "reject if patient-level sensitivity falls
more than 10 points." Half credit: strata without a decision rule.

**5.6 (cost estimate) rubric.** Full credit: assumptions about hardware, throughput, and price;
cost per 1,000 documents for both models; the accuracy gain needed to justify the larger model.
Half credit: one estimate.

**5.7 (benchmark critique).** Gaps: the benchmark tasks are not the buyer's task; the benchmark
domain is literature or mixed, not the buyer's notes; the unit is not the buyer's decision unit. A
question for the vendor: "what is the performance on our note types at our decision unit, against a
local reference standard?" A second question: "was the benchmark in the pretraining data?"

**5.8 (governance checklist) rubric.** Full credit: license, training data provenance, DUA
compliance, PHI handling, BAA for hosted inference, local validation, subgroup performance,
calibration, monitoring, version pinning, incident response, and model card. Half credit: eight
items without owners.

## 14. Interview questions

**1. "Why can't you just use BERT on a discharge summary?"**
Strong answer: 512-token limit, notes are longer, truncation creates a recall ceiling; names
chunking or long-context alternatives. Weak answer: "BERT is old."

**2. "What is the difference between token-level and span-level F1?"**
Strong answer: token counts per-token correctness, span counts matched entities under a boundary
rule; token F1 is higher and can mislead. Weak answer: "they are both F1."

**3. "How do you handle a document longer than the context window?"**
Strong answer: measure the length distribution, choose truncation, chunking, hierarchical
aggregation, or a long-context model, and merge chunk outputs. Weak answer: "truncate."

**4. "Why does domain-specific pretraining matter?"**
Strong answer: vocabulary and style differ; PubMedBERT's results show literature and notes are
different domains; local validation is still required. Weak answer: "clinical models are better."

**5. "What is domain shift in clinical NLP?"**
Strong answer: differences in templates, abbreviations, coding, and case mix across sites; measure
on target data. Weak answer: "overfitting."

**6. "How would you evaluate a model trained on MIMIC for your hospital?"**
Strong answer: local held-out sample, stratified by note type and subgroup, at the decision unit,
with a pre-registered acceptance rule. Weak answer: "run it and see."

**7. "What are the privacy risks of hosted inference?"**
Strong answer: PHI disclosure, BAA requirement, retention and training use, minimum necessary;
private endpoint or local model as alternatives. Weak answer: "the vendor says it is secure."

**8. "How do you merge predictions from overlapping chunks?"**
Strong answer: overlap-based deduplication with a score rule, boundary audit, and a check for lost
spans. Weak answer: "take the first prediction."

**9. "What does a model card tell you and what does it not?"**
Strong answer: training data, intended use, performance, limits; it is not a local validation or a
regulatory finding. Weak answer: treats it as certification.

**10. "How do you estimate inference cost?"**
Strong answer: tokens per document, model size, hardware throughput, price per hour, cost per
1,000 documents, and the accuracy gain needed to justify it. Weak answer: "it depends."

**11. "What is catastrophic forgetting and why does it matter here?"**
Strong answer: fine-tuning can degrade pretrained capabilities; use lower learning rates, adapters,
or evaluate on held-out tasks. Weak answer: "the model forgets."

**12. "When would you not use a transformer?"**
Strong answer: small label set, explicit evidence, strict auditability, low compute, or a rule
baseline that already meets the decision. Weak answer: "never."

## 15. Key terms

- **BIO tagging** — beginning, inside, outside labels for spans.
- **Calibration** — agreement between predicted probability and observed frequency.
- **Catastrophic forgetting** — loss of pretraining knowledge during fine-tuning.
- **Checkpoint** — a saved set of model weights.
- **Chunking** — splitting a document into windows.
- **Context window** — the maximum input length in tokens.
- **Decoder** — a transformer stack that generates tokens.
- **Distillation** — training a smaller model from a larger one.
- **Domain adaptation** — continued pretraining on target-domain text.
- **Domain shift** — difference between training and deployment distributions.
- **Encoder** — a transformer stack producing contextual representations.
- **Fine-tuning** — training on labeled task data.
- **GatorTron** — a large clinical language model from the University of Florida.
- **Inference cost** — compute or money per document.
- **Long-context model** — an architecture that reads more than 512 tokens.
- **Model card** — documentation of a model's data, performance, and limits.
- **Offset map** — mapping from tokens to character positions.
- **Overlap** — the shared region between chunks.
- **Pretraining** — training on unlabeled text.
- **PubMedBERT** — a biomedical model pretrained on PubMed abstracts.
- **Quantization** — reducing numerical precision of weights.
- **Sparse attention** — attention restricted to local windows plus global tokens.
- **Span-level F1** — F1 over matched spans.
- **Stride** — the step between chunk starts.
- **Subword token** — a unit produced by a tokenizer.
- **Token-level F1** — F1 computed over tokens.
- **Tokenizer** — a function mapping text to subword ids.
- **Transformer** — an attention-based neural architecture.
- **Truncation** — dropping text beyond the window.
- **Vocabulary** — the set of subword units.
- **Idempotent merge** — a merge that produces the same output when run twice.
- **Boundary duplicate** — a prediction repeated across two overlapping chunks.
- **Context coverage** — the share of target entities inside the model's input window.

## 16. Sources

All URLs accessed 25 October 2026.

1. Gu Y, Tinn R, Cheng H, et al. "Domain-specific language model pretraining for biomedical natural
   language processing." ACM Trans Comput Healthc. 2021;3(1):1-23.
   https://doi.org/10.1145/3458754
2. Yang X, Chen A, PourNejatian N, et al. "A large language model for electronic health records."
   npj Digit Med. 2022;5:194. https://doi.org/10.1038/s41746-022-00742-2
3. Li Y, Wehbe RM, Ahmad FS, Wang H, Luo Y. "A comparative study of pretrained language models for
   long clinical text." J Am Med Inform Assoc. 2023;30(2):340-347.
   https://doi.org/10.1093/jamia/ocac225
4. Alsentzer E, Murphy JR, Boag W, et al. "Publicly available clinical BERT embeddings." Proc 2nd
   Clinical NLP Workshop, ACL 2019. https://arxiv.org/abs/1904.03323
5. PhysioNet. MIMIC-IV-Note v2.2. https://physionet.org/content/mimic-iv-note/2.2/
6. Devlin J, Chang MW, Lee K, Toutanova K. "BERT: Pre-training of deep bidirectional transformers
   for language understanding." NAACL 2019. https://arxiv.org/abs/1810.04805
7. Beltagy I, Peters ME, Cohan A. "Longformer: The long-document transformer." 2020.
   https://arxiv.org/abs/2004.05150
8. U.S. Food and Drug Administration. "Artificial Intelligence-Enabled Device Software Functions:
   Lifecycle Management and Marketing Submission Recommendations." Draft guidance, January 2025.
   https://www.fda.gov/media/184856/download
9. ONC. "HTI-1 Final Rule." https://healthit.gov/regulations/hti-rules/hti-1-final-rule/
10. ONC. "HTI-1 Decision Support Interventions (DSI) Fact Sheet."
    https://healthit.gov/wp-content/uploads/2023/12/HTI-1_DSI_fact-sheet_508.pdf
11. U.S. Department of Health and Human Services. "Business Associates."
    https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/business-associates/index.html
12. scispaCy. https://allenai.github.io/scispacy/

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_classical-methods.md`](04_classical-methods.md)
- Next topic: [`06_annotation.md`](06_annotation.md)
- Evaluation of span and document metrics: [`07_evaluation.md`](07_evaluation.md)
- De-identification and PHI handling: [`08_deidentification.md`](08_deidentification.md)
- Module survey: [`../../09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
