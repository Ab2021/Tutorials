---
lesson_id: "FT-M00-01"
lesson_title: "What Is a Large Language Model?"
version: "1.0-draft"
created: "2026-11-07"
reviewed: null
status: "draft"
evidence_basis: "primary papers and constructed examples"
owner_role: "curriculum maintainer"
review_trigger: "New evidence changes the stated model objective, evaluation boundary, or deployment distinction"
---

# 0.1 — What Is a Large Language Model?

> **Course location:** Module 19, Transformers, LLMs and Foundation-Model Internals. This lesson maps to `finetuning_course/Module_00_Foundations/0.1_What_is_a_Large_Language_Model.md` in [`Train/Templates/directory.txt`](../../../Templates/directory.txt). The visible source title is a scope cue, not evidence that its original lesson was inspected.
>
> **Status:** Draft; independently reviewed examples and runtime behavior remain outstanding.

## Learning outcomes

After this lesson, learners can:

1. Define a language model as a probability distribution over token sequences under a particular tokenizer and conditioning context.
2. Derive an autoregressive sequence probability from conditional next-token probabilities.
3. Compute a sequence log-likelihood and mean negative log-likelihood with explicit token boundaries and units.
4. Distinguish the model's training objective from task competence, truth, calibration, and safe action.
5. Explain how tokenization, context, decoding policy, and stopping rules change the observable output.
6. Choose a test that could falsify a claim about a model instead of relying on fluency or a single demonstration.

## Prerequisites and scope

Assume familiarity with multiplication, logarithms, conditional probability, vectors, and the idea that a neural network maps inputs to outputs. No prior knowledge of transformer internals is required here. Attention, positional representations, parameter counting, and training systems are separate lessons.

The term **large language model (LLM)** is used descriptively, not as a standardized size threshold. A model does not become reliable merely because it has many parameters or was trained on a large corpus. The central object in this lesson is the conditional distribution over tokens; the system that serves it may add prompts, retrieval, tools, safety filters, caching, and product logic.

All numeric examples below are **ILLUSTRATIVE** teaching values. They are not measurements of a named model, training run, benchmark, or deployment.

## 1. The basic object: a distribution over token sequences

A tokenizer maps text to a sequence of discrete token identifiers. Tokens may correspond to words, word fragments, punctuation, whitespace, control markers, or other vocabulary items. Token boundaries are defined by a particular tokenizer; they are not guaranteed to match words or characters.

Let a vocabulary contain `V` token IDs. A language model assigns a conditional probability to a possible next token given a preceding context. In an autoregressive model, the factorization is:

```text
P(x_1, x_2, ..., x_T) = product_{t=1..T} P(x_t | x_1, ..., x_{t-1})
```

Here `x_t` is the token at position `t`; `T` is the token count; each conditional probability is in `[0,1]`; and the product is the probability assigned to the whole token sequence under the model and tokenization convention. For the first position, the context may be an explicit beginning-of-sequence marker or an empty prefix according to the implementation.

The chain rule of probability justifies this factorization. It is not an independence assumption: later token probabilities are conditioned on earlier tokens. A change in an earlier token can change every later conditional distribution.

The factorization is useful because a long sequence probability can be evaluated as a succession of next-token predictions. It does **not** imply that the model stores a symbolic grammar or explicitly reasons over a complete future sequence. The neural network produces a finite score vector at each prediction step, and a decoding procedure chooses or samples from that distribution.

### What “language model” does and does not mean

- **Does mean:** a model defines scores or probabilities for token sequences or next-token continuations under its learned parameters and inference conventions.
- **Does not mean:** every generated sentence is grammatical, factual, useful, harmless, or calibrated.
- **Does not mean:** training on text gives authorized access to every fact represented in that text.
- **Does not mean:** one model checkpoint alone is the complete assistant or application. A deployed product may include an orchestration layer and non-model state.
- **Does not mean:** a next-token objective directly optimizes a business outcome such as correct claim adjudication or customer satisfaction.

The model learns statistical regularities from its training process. Whether a particular regularity corresponds to a causal relationship, verified fact, current state, or permitted action is a separate question.

## 2. From scores to probabilities

A neural model typically emits one real-valued logit per vocabulary entry. A logit is an unnormalized score; it is not itself a probability. Given logits `z_i`, the softmax map produces probabilities:

```text
p_i = exp(z_i) / sum_{j=1..V} exp(z_j)
```

For finite logits, each probability is nonnegative and the sum is one (up to floating-point error). Adding the same constant to every logit leaves the distribution unchanged because the factor `exp(c)` cancels between numerator and denominator. Multiplying logits by an inverse temperature changes concentration while preserving their rank when temperature is positive.

For numerical stability, implementations usually subtract the maximum logit before exponentiating. This does not change the mathematical distribution, but avoids overflow from unnecessarily large exponentials. Probabilities stored in finite precision can underflow; implementations often compute log-softmax directly for loss calculations.

A top-ranked next token is an argmax decision. Sampling is a different decision rule. Temperature, top-k, nucleus sampling, repetition penalties, constrained decoding, and stopping criteria alter the distribution or its use. They are properties of the generation procedure, not automatically properties learned by the checkpoint.

### A two-token calculation

Suppose a model assigns next-token probabilities `0.75` to token A and `0.25` to token B, with no other vocabulary entries in this simplified example. These sum to one. Greedy decoding chooses A. Sampling chooses A with probability 0.75 and B with probability 0.25. Neither rule turns the probabilities into a guarantee about semantic correctness.

If a temperature greater than one softens the logits, lower-ranked alternatives receive relatively more probability. A temperature below one sharpens the distribution. The exact new values depend on the original logits and the temperature; rescaling probabilities directly is not equivalent to rescaling logits then applying softmax.

## 3. Sequence probability, log-likelihood, and loss

Products of many probabilities become extremely small. Logarithms turn products into sums:

```text
log P(x_1:T) = sum_{t=1..T} log P(x_t | x_<t)
```

Natural logarithms produce values in **nats**; base-2 logarithms produce **bits**. Since probabilities are at most one, log probabilities are zero or negative. The negative log-likelihood (NLL) is nonnegative. Mean NLL divides by the selected number of target tokens and is reported in nats/token or bits/token; the token-count convention must be stated.

Validity: this calculation evaluates a specified token sequence under specified model parameters, tokenization, context, and special-token policy. It is not a probability that the model is correct in a real-world sense.

Derivation: take the natural logarithm of the chain-rule product; use `log(ab)=log(a)+log(b)`; negate to obtain a nonnegative loss; divide by the number of scored targets when a per-token average is required.

```text
FORMULA [LM-01]: L = -(1/T) * sum_{t=1..T} ln p_t
WHERE: L = mean negative log-likelihood [nats/token]; T = scored target-token count [tokens];
       p_t = model probability of the observed target token given its prefix [dimensionless].
CONSTRAINT: identify whether beginning/end markers, padding, prompt tokens, and masked labels are included.
NUMERICAL EXAMPLE: p=(0.5, 0.25, 0.1); product=0.0125;
                   ln(product)=ln(0.0125)=-4.3820266 nats;
                   L=4.3820266/3=1.4606755 nats/token.
INTERPRETATION: the model assigned the exact three-token sequence probability 0.0125; its mean observed-token surprise is 1.4607 nats/token.
```

Independent check: `0.5 * 0.25 * 0.1 = 0.0125`; `ln(0.5)+ln(0.25)+ln(0.1) = -4.3820266` within rounding. Mean NLL is not a semantic quality score by itself. It can improve while a downstream task, safety measure, or subgroup outcome worsens.

A token that has probability one contributes zero NLL. A token assigned probability near zero contributes a large positive NLL. If the model assigns exactly zero probability, NLL is infinite; practical softmax models with finite logits generally assign positive probabilities mathematically, though numeric underflow and masking can create implementation-level infinities.

## 4. Cross-entropy as a training objective

For a one-hot target distribution, cross-entropy reduces to the negative log probability assigned to the observed target token. With a dataset of sequences, training commonly minimizes an average token loss over selected target positions. The objective encourages higher probability for observed training tokens under their contexts.

This description omits many implementation choices: data mixture, packing, loss masks, sequence truncation, optimizer, schedule, distributed reduction, precision, regularization, and checkpoint selection. A reported “training loss” is interpretable only when these choices and the denominator are understood.

### Teacher forcing and next-token targets

In standard next-token training, the input is a prefix and the target is the following token. During training, the prefix commonly contains the observed earlier tokens, even when the model's own sampled continuation would differ. This is often called teacher forcing. The loss evaluates the next-token prediction against the supplied target sequence.

A basic alignment example:

| Input positions | Target positions scored |
|---|---|
| `A B C` | `B C D` |

A causal attention mask prevents a position from using future target tokens when the architecture is trained autoregressively. Padding, prompt, or instruction positions may be masked from loss depending on the objective. Incorrectly shifting targets or masking the wrong positions changes the task being optimized.

Teacher-forced evaluation and free generation answer different questions. Teacher forcing asks how well the model scores reference tokens under reference prefixes. Generation asks what outputs result when model-selected prefixes feed subsequent steps. Errors during generation can alter later contexts; per-token loss on reference contexts does not directly capture that trajectory effect.

### Conditional versus joint normalization

A conditional distribution over the next token sums to one over the vocabulary for each fixed context. A sequence probability is the product of those conditionals. These are different normalizations: one is local to a prefix, the other is a probability of a complete sequence (with a defined endpoint convention).

Comparing sequence likelihoods across tokenizers requires care. Different tokenizers create different token sequences and denominators. Mean NLL per token is not automatically comparable across tokenizers; byte-normalized or otherwise harmonized measures may be needed, and even those answer a specific comparison question rather than universal quality.

## 5. Context, prompt, and output boundaries

At inference, the model conditions on a context assembled by the application. This can include system instructions, user messages, retrieved documents, tool results, previous assistant responses, and special control tokens. A model sees token IDs and position structure, not the application's abstract labels unless those labels are encoded in the input.

A context window is an implementation/model constraint on processed tokens, not a guarantee that every included token receives equal attention or is used correctly. Context may be truncated, summarized, reordered, or partially cached by the serving system. Always distinguish the logical conversation from the concrete model input.

The output also has boundaries. Generation may stop at an end-of-sequence token, a designated stop sequence, a maximum token count, a tool-call marker, or a server-side cancellation. The rendered text may omit control tokens, and the application may parse structured output after decoding. These details affect reproducibility and evaluation.

### Prompting is not weight training

Prompting changes the conditioning input; it does not, by itself, update model parameters. Demonstrations in a prompt can steer outputs for the current request, sometimes called in-context learning, but the name does not imply gradient-based learning or persistent model change.

Retrieval adds selected external content to the input. It may improve access to current or domain-specific evidence when retrieval and permission handling are sound. It does not prove that the model follows the source faithfully, cites it correctly, or generalizes beyond the retrieved material.

Fine-tuning updates some or all model parameters using a training procedure. It can change behavior but introduces data, evaluation, governance, regression, and deployment costs. Training is not automatically the right response to a prompt, retrieval, or product-logic problem.

## 6. Training stages and artifact contracts

The words “pretraining,” “continued pretraining,” “supervised fine-tuning,” “preference optimization,” and “serving” identify different processes. Real systems may combine or reorder steps, but an author or operator should state the actual sequence and artifact handoff rather than infer it from a model label.

| Stage | Typical input signal | Intended artifact or change | What it does not establish |
|---|---|---|---|
| Pretraining | Large token corpus and a self-supervised objective such as next-token prediction | Base checkpoint and tokenizer/config lineage | Task accuracy, factuality, permission, or safe tool behavior |
| Continued pretraining | Additional domain or updated text under a language-model objective | Adapted checkpoint | That data was licensed, representative, or free of forgetting |
| Supervised fine-tuning (SFT) | Input/target examples for desired responses or formats | Checkpoint or adapter optimized on demonstrations | That demonstrations are correct, representative, or preferred by users |
| Preference optimization | Comparisons, rankings, or related preference signals | A policy/checkpoint changed to better match an encoded preference objective | That preference labels represent truth, safety, or every population |
| Distillation | Teacher-generated or teacher-scored targets used to train a student | Smaller or task-specific student | Equal behavior across all inputs or reliable transfer beyond measured domains |
| Serving | Versioned checkpoint plus tokenizer, runtime, decoding and application contracts | Responses or structured outputs to requests | Training quality beyond observed evaluations or correct end-to-end product behavior |

These are common abstractions, not an exhaustive or universal pipeline. Some systems use retrieval or tools instead of changing weights; some use adapters; some models are trained from scratch; some serving pipelines perform routing or postprocessing. Stage labels should be connected to explicit artifacts and tests.

A useful lineage record includes: base checkpoint identifier and cryptographic digest; tokenizer and configuration; dataset manifests and permissions; preprocessing code version; training objective and masking policy; optimizer/schedule/seed; adapter targets; evaluation dataset versions; serving runtime and decoding configuration; and release decision. A name alone is not a reproducibility record.

## 7. What scale does and does not tell you

“Large” is an underspecified adjective. Possible scale dimensions include parameter count, training tokens, compute, context length, vocabulary size, active parameters per token, data diversity, and serving footprint. Sparse architectures complicate the comparison because total parameters and active parameters can differ. Parameter count alone does not determine latency, memory, quality, or cost.

A larger model can have greater representational capacity, but realized behavior depends on data, objective, optimization, architecture, inference, and evaluation. An impressive result on one benchmark does not prove robust ability on a distinct task or distribution. A small task-specific model may be preferable when latency, privacy, cost, or deterministic behavior dominates and a validated model satisfies the requirement.

The term “foundation model” usually describes a model intended or demonstrated to support adaptation across multiple downstream tasks. It is not a formal reliability grade. Reuse can reduce the need for training from scratch, but licenses, access controls, model provenance, and task-specific evaluation still matter.

## 8. Evaluation: choose evidence for the claim

A claim must be matched to an evaluation. “The model predicts next tokens well” calls for held-out likelihood under a defined dataset and scoring contract. “The assistant answers domain questions correctly” calls for task-specific examples, a defensible reference standard, uncertainty and error analysis. “The deployed system is safe” requires threat and policy evaluations against the actual system boundary, not just a model benchmark.

At minimum, document:

- **Population:** which prompts, users, documents, languages, or environments the sample represents.
- **Unit:** token, sequence, conversation, user, tool call, or task episode.
- **Target:** exact reference or success criterion, with annotator procedure if applicable.
- **Split:** how training, tuning, and held-out evaluation are separated, including time and entity leakage controls.
- **Metric:** numerator, denominator, units, aggregation, missing/abstention convention, and direction of improvement.
- **Comparator:** unchanged base model, prompt-only baseline, retrieval baseline, prior system, or task-specific alternative.
- **Uncertainty:** sample size, repeated-seed variation or confidence interval as appropriate.
- **Guardrails:** quality slices, privacy/security, latency, cost, refusal behavior, and tool-action correctness.
- **Execution status:** planned, simulated, offline-tested, shadowed, canaried, or deployed; do not conflate them.

A benchmark score is not an intrinsic property of a checkpoint independent of prompt format, scoring code, sample selection, and decoding settings. Contamination can inflate scores if evaluation content or close variants appear in training or tuning data. State what was checked and what remains unknown.

## 9. Worked decision: prompt, retrieve, or train?

A support team has 800 de-identified question/answer pairs and a policy handbook that changes monthly. The requirement is to answer policy questions with citations and escalate uncertain cases. The following comparison is a decision exercise, not a measured result.

1. **Prompt-only baseline:** test whether a fixed instruction and output schema meet the task criteria without adding private examples to training. This is the cheapest first control but cannot supply handbook passages that are absent from the context.
2. **Retrieval baseline:** index authorized handbook versions, retrieve evidence at request time, enforce document permissions before generation, require citations to retrieved spans, and test stale/out-of-scope questions. This addresses changing facts more directly than weight updates, but retrieval and citation errors need independent measurement.
3. **SFT candidate:** train on a consented, versioned subset if response structure or workflow behavior remains deficient after prompt/retrieval. It may improve format consistency; it can also learn stale policy text and does not make citations grounded by itself.
4. **Preference candidate:** collect reviewed comparisons only if the unresolved decision is a preference tradeoff (for example, concise versus explanatory) and the preference protocol is valid. Preference tuning does not repair a missing handbook or an incorrect policy source.

The conservative decision is to establish prompt-only and permission-aware retrieval baselines first, with held-out policy versions and an escalation criterion. Proceed to weight updates only if a measured failure is plausibly addressed by training and the data owner approves the training use. The 800 examples alone do not justify any guaranteed quality claim.

### Decision matrix

| Observed problem | First intervention to test | Why | Disqualifier or guardrail |
|---|---|---|---|
| Current policy fact missing from model context | Retrieval from approved current source | Evidence freshness is a data-access problem | Permission filtering must precede exposure to model/context |
| Output schema is inconsistent | Prompt/schema enforcement; then SFT only if needed | Constrain or teach format before broad training | Validate parser behavior, retries, and malformed outputs |
| User asks for unsupported fact | Abstain/escalate policy and evidence sufficiency check | More fluent generation is not evidence | Measure false answer rate and escalation burden |
| Model lacks task-specific terminology | Evaluate retrieval and domain SFT options | Context may suffice; training may alter internal behavior | Check rights, held-out entities, and general regression |
| A business workflow needs an external action | Explicit tool contract and authorization | A text response is not an executed transaction | Authorization and confirmation belong outside model claims |

## 10. Common conceptual traps

1. **“The model knows the answer.”** A high-probability continuation is not proof of a verified internal fact. Ask what source, date, and evaluation support the answer.
2. **“The model understands exactly like a person.”** This is a broad philosophical claim, not established by next-token loss. Operationally specify observable competencies and failure boundaries.
3. **“More parameters means more accurate.”** Parameter count is one scale dimension. Compare on a matched task, compute budget, latency and quality slices.
4. **“Training loss is accuracy.”** Training loss measures a probabilistic objective on a specific target distribution. It does not equal task accuracy or user benefit.
5. **“A prompt teaches the model permanently.”** Prompting changes request context; it does not update the checkpoint unless a separate learning process occurs.
6. **“Few-shot means gradient updates.”** In-context demonstrations are input tokens; ordinary inference does not change stored weights.
7. **“A model always returns a probability.”** APIs may return text only; sequence likelihoods may not be exposed, and sampling scores can differ from calibrated probabilities.
8. **“Low perplexity proves factuality.”** Likelihood on text does not identify truth; plausible misinformation can be assigned high likelihood.
9. **“If the output is fluent, it is reliable.”** Fluency is a surface property. Verify claims and tool effects independently.
10. **“Open weights means open training data.”** Weights access, source code, data, recipe, and license are distinct disclosures.
11. **“A model update is a drop-in replacement.”** Tokenizer, chat template, special tokens, scores, safety behavior and runtime can change; use regression gates.
12. **“Retrieval makes hallucination impossible.”** Retrieval supplies context; the system can still retrieve poorly, misread evidence, miscite, or ignore it.

## 11. Interview and self-check questions

### Q1. What is a language model mathematically?
**Answer:** A distribution over token sequences, or a family of conditional next-token distributions that can be multiplied via the chain rule.
**Reasoning:** State the tokenization and conditioning context; do not define it as a database or an answer generator alone.
**Follow-up:** Does this definition guarantee truth?
**Answer:** No. Probability under learned parameters is not a truth predicate.

### Q2. Given conditional probabilities 0.5, 0.25, and 0.1, what is the sequence probability?
**Answer:** `0.5 × 0.25 × 0.1 = 0.0125` under the specified factorization.
**Reasoning:** Each term conditions on the prior prefix, so the chain rule applies without independence.
**Follow-up:** What assumption would be wrong to claim?
**Answer:** That the tokens are independent.

### Q3. Compute mean NLL in nats/token for those three values.
**Answer:** `-(ln 0.5 + ln 0.25 + ln 0.1)/3 ≈ 1.4607` nats/token.
**Reasoning:** The sequence log probability is approximately `-4.3820` nats; negate and divide by three scored tokens.
**Follow-up:** What if the target includes an end marker?
**Answer:** State whether it is scored and update both the sum and denominator consistently.

### Q4. Why use logits and softmax?
**Answer:** The network can produce unnormalized real-valued scores; softmax maps them to a normalized categorical distribution.
**Reasoning:** The normalizer is the sum of exponentiated logits. Numerically stable implementations subtract a shared maximum first.
**Follow-up:** Is a logit of 2 a 2% probability?
**Answer:** No; its probability depends on all vocabulary logits and any temperature or transformations.

### Q5. Does greedy decoding choose the most likely full sequence?
**Answer:** Not necessarily. It chooses the locally highest-probability token at each step; locally greedy choices need not maximize the joint probability over all sequences under constrained search or later context effects.
**Reasoning:** The locally best token changes the next context and subsequent conditionals.
**Follow-up:** What procedure can seek a higher-probability sequence?
**Answer:** Search methods such as beam search can explore alternatives but remain approximate and may optimize a likelihood objective misaligned with user utility.

### Q6. Why can validation loss improve while task success declines?
**Answer:** The distributions and objectives differ; the model may fit token statistics while worsening relevant task behavior, safety slices, or generation trajectories.
**Reasoning:** Check data mixture, task metric, output protocol, and subgroup regressions.
**Follow-up:** What should be held constant in a model comparison?
**Answer:** Evaluation population, prompt, decoding, scoring code, and any systems components not intended to change.

### Q7. Contrast prompting with fine-tuning.
**Answer:** Prompting changes the conditioning input; fine-tuning changes parameters through an optimization procedure.
**Reasoning:** Prompt effects are generally request-scoped unless stored externally; fine-tuning creates a new checkpoint or adapter artifact.
**Follow-up:** Which is more suitable for rapidly changing factual policy?
**Answer:** Usually test authorized retrieval and versioned source controls first; do not presume retraining is fresher.

### Q8. What does a tokenizer change affect?
**Answer:** Token IDs, sequence lengths, embedding/output dimensions, special-token handling and loss denominators can change; compatibility with a checkpoint is required.
**Reasoning:** The model's learned parameters are defined over a vocabulary and token convention.
**Follow-up:** Can mean NLL/token be compared directly across tokenizers?
**Answer:** Not generally; token counts and units differ.

### Q9. What is teacher forcing, and what does it miss?
**Answer:** The model is conditioned on the reference prefix while predicting reference next tokens. It measures conditional fit on those prefixes, not the complete distribution of self-generated trajectories.
**Reasoning:** Errors in generated prefixes can compound or change later behavior.
**Follow-up:** What evaluation complements it?
**Answer:** Task-level free-generation tests with a fixed decoding and scoring contract.

### Q10. Does a large parameter count establish a foundation model's value?
**Answer:** No. It is a resource and capacity descriptor, not an outcome measure.
**Reasoning:** Compare task performance, robustness, operational footprint, licensing and constraints directly.
**Follow-up:** What extra count matters for a sparse model?
**Answer:** Distinguish total from activated parameters per token, while also measuring routing and serving costs.

### Q11. A model answers correctly on five examples. What can you conclude?
**Answer:** Only that it produced correct answers on those five examples under the tested conditions.
**Reasoning:** Sample size, selection, reference quality, leakage, and distribution coverage limit inference.
**Follow-up:** What do you request next?
**Answer:** A prespecified held-out evaluation with relevant slices, uncertainty, comparator and failure review.

### Q12. Which artifact should be versioned to reproduce a response?
**Answer:** At minimum checkpoint, tokenizer, prompt/template, retrieval corpus snapshot, tool configuration, runtime, decoding parameters and application code version.
**Reasoning:** Any component can change token input, evidence, sampling, stopping, or postprocessing.
**Follow-up:** Why include corpus snapshot?
**Answer:** The same query against a changed index may receive different evidence.

### Q13. A domain task fails because the required source document is absent from context. Should you fine-tune?
**Answer:** First test authorized retrieval or explicit data access; the failure is evidence availability, not necessarily a learned-format deficit.
**Reasoning:** Training can encode stale facts and still fail to provide traceable evidence.
**Follow-up:** When might fine-tuning become relevant?
**Answer:** If a measured residual failure concerns behavior that training examples can validly teach and the data use is authorized.

### Q14. What claim does a benchmark support?
**Answer:** A scoped claim about the tested system, dataset, prompts, metrics, and protocol, subject to leakage and sampling limitations.
**Reasoning:** A score is not context-free and cannot support every downstream claim.
**Follow-up:** How can contamination be investigated?
**Answer:** Audit corpus overlap where feasible, use fresh/private items, and report residual uncertainty.

### Q15. Why is serving not simply “run the model”?
**Answer:** The deployed behavior includes input assembly, authorization, retrieval/tool routing, decoding, resource limits, stopping, parsing, retries, monitoring and fallback.
**Reasoning:** The model output alone may not equal the user-visible or action-causing outcome.
**Follow-up:** What is the boundary for a consequential tool call?
**Answer:** Authenticate and authorize outside the model, validate arguments, require policy-compliant approval when needed, and record the result.

## 12. Quick reference

| Item | Rule |
|---|---|
| Model object | Conditional probabilities over token IDs under a defined context and tokenizer. |
| Autoregressive factorization | Joint sequence probability is the product of next-token conditionals; no independence assumption. |
| Log-likelihood | Sum of token log probabilities; natural logs are nats. |
| Mean NLL | Negated log-likelihood divided by a documented scored-token count. |
| Training objective | Optimizes a specified loss on specified examples and masks; not a universal reliability measure. |
| Inference output | Depends on prompt, model, decoding, stopping, runtime, and application postprocessing. |
| Prompt vs fine-tuning | Prompt changes input; fine-tuning updates parameters. |
| Retrieval | Adds selected external evidence; it does not guarantee faithful use or correct citation. |
| Evaluation | Match the claim to population, unit, target, split, metric, comparator, uncertainty and guardrails. |
| Training decision | Establish prompt-only and retrieval baselines; update weights only for an evidenced, trainable gap. |
| Evidence status | All calculations are illustrative; no model benchmark, deployment, or runtime has been tested here. |

## Sources and boundaries

1. Vaswani et al., [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762), 2017. Primary transformer paper; consulted for architectural history, not a modern LLM reliability claim.
2. Brown et al., [“Language Models are Few-Shot Learners”](https://arxiv.org/abs/2005.14165), 2020. Primary GPT-3 paper; its abstract distinguishes few-shot prompting without gradient updates from fine-tuning and reports task-specific results that should not be generalized beyond their evaluation.
3. [Course authoring and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md), including the five-part equation block, evidence labels, and publication requirements.
4. [Fine-tuning source coverage plan](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-foundations), which maps the visible Module 00 subtopics to focused sheets and states the illustrative sequence exercise.
5. [`Train/Templates/directory.txt`](../../../Templates/directory.txt), visible lines 857–859: source inventory lists Module 00 and the two subtopic filenames. The original lesson bodies are not available in this course workspace.

This lesson introduces the probability object and its boundaries. It does not teach transformer attention, corpus curation, optimization, alignment algorithms, licensing advice, or deployment security in depth; those require separate lessons and evidence.
