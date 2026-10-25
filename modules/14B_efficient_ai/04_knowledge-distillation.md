# Knowledge Distillation

> Module 14B — Efficient AI and On-Device Inference
> Topic 04 of 11
> Estimated study time: 3 hours
> Prerequisites: 03
> Version 2026.3

## 1. Why this topic exists

A large model often learns more than the labels it was trained on. It learns which classes are confusable, which examples are ambiguous, and how much evidence is needed before it commits. That information is present in the full distribution over classes, not in the one-hot label. Knowledge distillation transfers it to a smaller model by training the student on the teacher's softened outputs rather than only on the labels.

The technique is attractive because it appears to be free. The teacher already exists, the data already exists, and the student is smaller by construction. In practice, distillation is one of the least predictable methods in the compression family, because its benefit depends on a relationship between two models that is difficult to measure in advance.

Three failure patterns dominate.

**The capacity gap.** A student that is too small cannot represent what the teacher knows, no matter how good the soft targets are. A student that is too large gains little over training on labels and costs more than necessary. The right size is neither the smallest that fits nor the largest that fits the budget; it is the size at which the student's own capacity, not the teacher's knowledge, becomes the limit. Finding it requires a sweep, and the sweep is the part teams skip.

**The biased teacher.** A student trained on a teacher's outputs inherits the teacher's errors, including its subgroup errors. If the teacher under-detects a condition in one population, the student will learn the same shortcut, and the soft targets give the error a smoother, more confident appearance. The teacher must be audited before it is used as a source of truth, because after distillation the student's errors are partly the teacher's errors wearing a smaller architecture.

**The calibration surprise.** Distillation often improves accuracy while leaving the student's probabilities poorly calibrated, or improves calibration while leaving accuracy unchanged. Temperature scaling during training and temperature scaling after training are different operations, and a student that matches the teacher's logits at temperature T does not necessarily produce calibrated probabilities at temperature 1.

There is also a regulatory dimension. A distilled model is a new architecture. In a regulated product it is a larger change than a quantization step: the model structure, the training procedure, and often the intended operating point all change. Whether that change is covered by a predetermined change control plan (PCCP) is a question that must be answered before the student is trained, not after it is validated.

This topic teaches the mechanics, the choice of signal, the tuning of temperature and weight, the audit of the teacher, and the evaluation that makes a distilled model defensible.

### 1.1 Four questions before any distillation run

A distillation project should be able to answer four questions before training starts. A "no" or "unknown" to any of them is a reason to stop.

1. **Is the teacher actually better than the alternative?** If a student of the target size trained on labels matches the teacher on the clinical metric, the teacher adds cost without adding signal.
   - Evidence: a label-trained baseline at the same size.
   - Failure if skipped: a quarter spent to reproduce the baseline.

2. **Does the teacher carry information the labels do not?** Soft targets help when classes are confusable or labels are noisy. They help little when the task is easy and the labels are clean.
   - Evidence: the teacher's class-confusion structure and the label noise rate.
   - Failure if skipped: no gain, and a team that concludes distillation does not work.

3. **Is the teacher safe to imitate?** A biased or miscalibrated teacher transfers its problems.
   - Evidence: a subgroup and calibration audit of the teacher.
   - Failure if skipped: a student that inherits a gap the team did not know existed.

4. **Is the change covered?** A distilled model is a new architecture.
   - Evidence: a change classification against the approved plan.
   - Failure if skipped: a regulatory question at launch.

The questions are cheap. The failures they prevent are not.

### 1.2 What distillation is not

- It is not model compression in the storage sense. The student is a new model, not a smaller encoding of the teacher.
- It is not a way to avoid data collection. The student learns from the transfer set, and a transfer set that misses a subgroup produces a student that misses it.
- It is not a guarantee of calibration. The student inherits the teacher's confidence, which may be wrong.
- It is not automatically within a regulatory envelope. Architecture changes are usually larger changes than precision changes.

The rest of the topic assumes these boundaries and works inside them.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** a temperature-scaled softmax and the KL divergence between teacher and student distributions by hand.
2. **Explain** what the temperature and the loss weight do to the gradient and to the information transferred.
3. **Choose** between response-based, feature-based, and relation-based distillation for a given architecture and task.
4. **Design** a student capacity sweep and interpret its result.
5. **Audit** a teacher for bias, calibration, and shortcut learning before distillation.
6. **Evaluate** a distilled model on accuracy, calibration, subgroups, latency, and energy, with confidence intervals.
7. **Combine** distillation with quantization and explain the order.
8. **Document** a distillation run so that the student can be reproduced and the change classified.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 and Section 7 | Correct probabilities and KL |
| 2 | Exercise 2 | A gradient-scale argument |
| 3 | Exercise 3 | A signal choice with a rationale |
| 4 | Exercise 5 | A sweep table and a chosen size |
| 5 | Exercise 6 | A teacher audit with subgroup results |
| 6 | Exercise 7 | An evaluation table with intervals |
| 7 | Exercise 8 | An ordered pipeline with a rationale |
| 8 | Exercise 8 | A reproducible record and classification |

## 3. The core mental model

Distillation is a transfer of a distribution, not of a weight matrix. The teacher converts an input into a distribution over classes; the student is trained to match that distribution while also learning from the label.

```text
                    ┌──────────────────────┐
   input x ────────►│  teacher T (frozen)  │────► logits z_t ──► softmax(z_t / T) ──┐
        │           └──────────────────────┘                                       │
        │                                                                          ▼
        │           ┌──────────────────────┐                              ┌──────────────────┐
        └──────────►│  student S (trained) │────► logits z_s ──► softmax(z_s / T) │  KL divergence  │
                    └──────────────────────┘                              └──────────────────┘
                              │                                                     │
                              └────────────► softmax(z_s) ──► cross-entropy ◄──────┘
                                                              with label y
```

The student receives two gradients: one from the hard label, which says which class is correct, and one from the teacher's distribution, which says how the teacher would distribute probability among the alternatives. The second is the "dark knowledge": the relative probabilities of incorrect classes encode similarity, ambiguity, and the teacher's confidence.

### 3.1 Temperature

The softmax with temperature is:

```text
p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)
```

At T = 1 it is the ordinary softmax. As T increases, the distribution flattens and the small differences between the non-target logits become visible in the probabilities. As T approaches 0, the distribution approaches a one-hot vector and the dark knowledge disappears.

An example with teacher logits [2.0, 0.5, -1.0]:

| Temperature | p(class 1) | p(class 2) | p(class 3) | Entropy (nats) |
|---:|---:|---:|---:|---:|
| 1 | 0.7856 | 0.1753 | 0.0391 | 0.6216 |
| 2 | 0.5898 | 0.2786 | 0.1316 | 0.9343 |
| 4 | 0.4630 | 0.3182 | 0.2187 | 1.0533 |

At T = 1 the third class has 3.9% probability; at T = 4 it has 21.9%. If the third class is a clinically confusable condition, that difference is the information the student needs, and T = 1 hides it.

The cost of a high temperature is that the distribution approaches uniform and stops discriminating. The useful range is usually between 2 and 10 for classification, and it must be tuned on validation data, not assumed.

### 3.2 The loss

The standard objective combines a cross-entropy term on the hard label and a Kullback-Leibler (KL) divergence term on the softened distributions:

```text
L = α × CE(y, softmax(z_s)) + (1 - α) × T² × KL(softmax(z_t / T) || softmax(z_s / T))
```

The T² factor compensates for the gradient scaling introduced by the temperature. Without it, changing T also changes the effective learning rate of the distillation term, and comparisons across temperatures are confounded.

The weight α controls the balance. At α = 1 the student learns only from labels and the teacher is irrelevant. At α = 0 the student learns only from the teacher and may drift away from the label, particularly if the teacher is wrong. Typical values are 0.1 to 0.5, with the distillation term dominant.

### 3.3 Three signals

Response-based distillation matches the output distribution. It is the simplest and the most general, and it requires only that both models produce logits over the same classes.

Feature-based distillation matches intermediate representations. It requires a mapping between teacher and student layers, and it works best when the architectures are similar. It can transfer spatial or temporal structure that the output distribution does not expose.

Relation-based distillation matches relationships between examples or between layers, such as pairwise similarities in a batch. It does not require the teacher and student to share a representation space, which makes it attractive for cross-architecture distillation, but it is more sensitive to batch composition.

| Signal | What is matched | Requires | Best for | Main risk |
|---|---|---|---|---|
| Response | Output distribution | Same label space | Classification, detection | Teacher miscalibration |
| Feature | Intermediate activations | Layer mapping, similar width | Vision, speech | Dimension mismatch |
| Relation | Similarities between examples | Batches with structure | Cross-architecture | Batch-dependent noise |
| Attention | Attention maps | Transformer-like teacher | Transformers | Architecture specificity |
| Rationale | Explanations or chain-of-thought | Generated rationales | Language models | Propagated reasoning errors |

### 3.4 What the student can and cannot learn

The student learns the teacher's function on the data distribution it sees. It does not learn:

- Behavior on inputs the teacher never saw.
- The teacher's internal computation, only its outputs (for response-based distillation).
- A better decision boundary than the teacher's, except where the label term corrects it.
- Calibration automatically; a student can match the teacher's probabilities and still be miscalibrated if the teacher was.

The last point is the most consequential. Distillation transfers the teacher's confidence as well as its knowledge. If the teacher is overconfident on out-of-distribution inputs, the student will be too, and the student's smaller capacity may make the overconfidence worse because it has less room to represent ambiguity.

### 3.5 The capacity curve

The relationship between student size and distilled accuracy is not monotonic in usefulness. It usually looks like this:

```text
accuracy
   ▲
   │            ●────────●────────●   teacher-level plateau
   │        ●
   │     ●
   │   ●
   │  ●
   └──┴────┴────┴────┴────┴────┴────► student size
       too small   useful range   wasteful
```

Below a threshold, the student cannot represent the function and distillation gives little over label training. Above a threshold, additional capacity buys little accuracy but costs latency and memory. The useful range is between the two, and it is model- and task-specific.

A capacity sweep is the only reliable way to find it. The sweep should hold the training data, the temperature, and the loss weight fixed, and vary only the student width or depth. The result is a table of size against accuracy, calibration, and latency, which is the input to the student choice.

### 3.6 What the temperature does to the gradient

The temperature is not only a view into the teacher's distribution; it also changes the gradient the student receives.

For a soft-target cross-entropy, the gradient with respect to a student logit `z_s,i` is:

```text
∂L/∂z_s,i = (1/T) × (p_s,i - p_t,i)
```

The difference between the distributions is divided by T. When the student is far from the teacher, the gradient shrinks as T grows; when the student is close, the difference itself shrinks. The T² factor in the loss restores the overall scale, so the effective learning rate of the soft term stays comparable across temperatures.

This has two practical consequences.

- A temperature sweep without the T² factor is also a learning-rate sweep. The apparent optimum may reflect the step size rather than the information.
- A student trained at high temperature may need a lower learning rate for the soft term even with T², because the gradient direction, not only its magnitude, changes when the distribution flattens.

The gradient also explains why distillation is most useful early in training. When the student is far from the teacher, the soft gradient provides a direction that the hard label does not: it says which wrong classes are less wrong. Near convergence, the soft and hard gradients often agree, and the label term dominates.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Teacher | The model providing soft targets | Assumed to be correct | Model |
| Student | The model being trained | Assumed to be uniformly smaller | Model |
| Logit | Pre-softmax output | Called a probability | Real |
| Soft target | Temperature-scaled teacher distribution | Confused with label smoothing | Probability vector |
| Hard target | One-hot or annotated label | Assumed to be noise-free | Label |
| Temperature (T) | Softmax scaling parameter | Confused with training temperature | Unitless |
| Alpha (α) | Weight on the hard-label loss | Confused with learning rate | 0–1 |
| KL divergence | Asymmetric measure between distributions | Treated as a distance metric | Nats |
| Cross-entropy | Negative log-likelihood of the label under the prediction | Confused with KL | Nats |
| Dark knowledge | Information in non-target class probabilities | Treated as a formal quantity | Informal |
| Response-based | Distillation from output distributions | Called "logit distillation" loosely | Method |
| Feature-based | Distillation from intermediate representations | Assumed to need equal widths | Method |
| Relation-based | Distillation from similarities between examples | Confused with metric learning | Method |
| Attention transfer | Distillation from attention maps | Assumed to work across architectures | Method |
| Self-distillation | Teacher and student share an architecture or model family | Confused with fine-tuning | Method |
| Online distillation | Teacher and student trained together | Confused with co-training | Method |
| Born-again | A student becomes the next teacher | Confused with iterative pruning | Method |
| Ensemble distillation | Multiple teachers | Assumed to require synchronized training | Method |
| Capacity gap | Difference in expressive power | Ignored when choosing student size | Model property |
| Label smoothing | Softening labels with a uniform distribution | Confused with soft targets | Method |
| Calibration | Agreement between probability and frequency | Confused with quantization calibration | Metric |
| Temperature scaling | Post-hoc calibration with a single temperature | Confused with distillation temperature | Method |
| Reliability diagram | Plot of predicted probability against observed frequency | Omitted after distillation | Plot |
| Expected calibration error (ECE) | Bin-weighted calibration gap | Compared across bin counts | Proportion |
| Brier score | Mean squared error of probabilities | Reported without base rate | 0–1 |
| Calibration slope | Slope of outcome on predicted log-odds | Ignored after distillation | Unitless |
| Subgroup | A population defined by a clinically relevant attribute | Treated as optional | Group |
| Shortcut learning | Reliance on spurious correlates | Assumed absent in teachers | Failure |
| Bias transfer | Inheritance of teacher bias by the student | Assumed away by more data | Failure |
| Rationale distillation | Training on teacher explanations | Assumed to improve reasoning | Method |
| Sequence-level distillation | Training on teacher-generated sequences | Confused with logit distillation | Method |
| Logit distillation | Training on teacher logits | Requires a shared vocabulary | Method |
| Teacher audit | Evaluation of the teacher before use | Skipped under schedule pressure | Activity |
| Recovery | Accuracy regained relative to label training | Assumed to be complete | Proportion |
| Student sweep | Systematic variation of student size | Replaced by a single guess | Activity |
| Distillation loss | The combined objective | Called "the loss" without qualification | Nats |
| Gradient scale | Magnitude of gradients from a loss term | Ignored when T changes | Real |
| Soft-label entropy | Entropy of the teacher distribution | Assumed constant across inputs | Nats |
| Agreement | Fraction of inputs where teacher and student predict the same class | Confused with accuracy | Proportion |
| Transfer set | Data used to generate soft targets | Assumed to be labeled | Dataset |
| Unlabeled transfer | Using unlabeled data for distillation | Assumed to be free of bias | Method |

Two definitions need emphasis because they are frequently confused.

**Temperature scaling for distillation** is a training parameter that softens the teacher's distribution. **Temperature scaling for calibration** is a post-hoc fit of a single scalar to a trained model's logits, used to improve probability calibration. They share a symbol and nothing else. A student distilled at T = 4 is not calibrated by that choice; it may still need a post-hoc temperature fit, and that fit is a separate change to the model.

**KL divergence and cross-entropy** differ by the entropy of the teacher distribution. Since the teacher is fixed, the difference is constant with respect to the student's parameters, so optimizing KL and optimizing cross-entropy against soft targets produce the same gradients. The distinction matters for reporting, not for training, and a paper that reports one while claiming the other is not necessarily wrong but is imprecise.

### 4.1 Notation

| Symbol | Meaning | Range |
|---|---|---|
| z_t | Teacher logits | Real vector |
| z_s | Student logits | Real vector |
| T | Temperature | Positive real |
| p_t | softmax(z_t / T) | Simplex |
| p_s | softmax(z_s / T) | Simplex |
| y | Hard label | One-hot or probabilistic |
| α | Weight on the hard term | 0–1 |
| L | Combined loss | Nats |
| H(p) | Entropy of p | ≥ 0 nats |
| D_KL(p || q) | KL divergence | ≥ 0 nats |
| N | Number of classes | Integer |
| P_s | Student parameters | Count |
| M_s | Student MACs | Count |

The combined objective in notation:

```text
L = α · CE(y, softmax(z_s)) + (1 - α) · T² · D_KL(p_t || p_s)
```

When the hard label is also softened by label smoothing, the label term becomes a cross-entropy against a smoothed distribution rather than a one-hot vector. The two softenings are different: label smoothing is uniform and data-independent, while the teacher's distribution is input-dependent. Using both is possible, but the effective alpha must be interpreted against the smoothed label, not the original one.

## 5. The mechanics, step by step

The procedure produces a distilled student with evidence. It assumes a trained teacher and a defined deployment budget.

### Step 1. Define the student budget before choosing the architecture

**Input:** The constraint ledger and the profile of the teacher or baseline.

**Output:** A target parameter range, MAC range, latency ceiling, and memory ceiling for the student.

**Rationale:** The student architecture should be chosen to fit the budget, not chosen for accuracy and then compressed. A student designed to the budget avoids a second compression stage and keeps the change surface smaller.

The budget is a range, not a point, because the capacity sweep will explore it:

```text
parameters:  20,000 to 35,000
MACs:        2.0 M to 3.2 M
latency:     ≤ 40 ms at p99 on the target
memory:      ≤ 64 KiB of flash, weights plus metadata
```

**Branch point:** If the budget cannot be met by any plausible student, distillation is the wrong method. Return to Topic 03 and consider structured pruning of the teacher or a different architecture.

**Edge case:** A student that meets the parameter budget may still miss the latency budget if it uses operators the target does not accelerate. Check operator support before the sweep, not after.

### Step 2. Train or obtain the teacher and audit it

**Input:** A trained teacher, a held-out set, and a subgroup table.

**Output:** A teacher audit with accuracy, calibration, subgroup performance, and a list of known failure modes.

**Rationale:** The student inherits what the teacher knows, including its errors. An unaudited teacher is an unexamined source of truth, and the student's failures will be attributed to distillation rather than to the teacher.

The audit includes:

- Accuracy and calibration on a held-out set that was not used to train the teacher.
- Sensitivity and specificity at the operating threshold.
- Subgroup performance with confidence intervals.
- Performance on edge cases: rare classes, ambiguous inputs, out-of-distribution inputs.
- A check for shortcut learning: does performance collapse when a spurious correlate is removed or randomized?
- The teacher's calibration curve, because the student will learn its confidence.

**Branch point:** If the teacher fails a subgroup floor, do not distill from it without mitigation. Mitigations include reweighting the transfer set, correcting the teacher's outputs, or using a different teacher. Distilling first and auditing later transfers the problem to a model that is harder to inspect.

**Edge case:** A teacher can be accurate overall and badly calibrated. In that case, temperature scaling the teacher before generating soft targets can improve what the student learns, but it changes the target distribution and must be recorded.

### Step 3. Choose the distillation signal

**Input:** The teacher and student architectures and the task.

**Output:** A signal type and, if feature-based, a layer mapping.

**Rationale:** The signal determines what can be transferred and what infrastructure is required. Response-based distillation needs only logits; feature-based needs matched intermediate tensors; relation-based needs batches with meaningful structure.

| Situation | Recommended signal | Reason |
|---|---|---|
| Same task, same label space, different architectures | Response | Only the output space must match |
| Similar architectures, vision or speech | Feature plus response | Intermediate structure is transferable |
| Cross-architecture, no shared representation | Relation | Does not require a common feature space |
| Transformer teacher and student | Attention plus response | Attention maps encode token relations |
| Language model with rationales | Sequence-level plus response | Rationales carry reasoning, with risk |
| Very small student | Response only | Feature matching may exceed capacity |

**Branch point:** If a feature-based mapping requires a learned projection between teacher and student widths, add that projection to the training graph but not to the deployed model. The projection is a training artifact.

**Edge case:** Feature-based distillation can transfer noise if the teacher's intermediate layers are not meaningful for the student's architecture. Validate the signal by comparing feature-distilled and response-distilled students on held-out data.

### Step 4. Choose the temperature and the loss weight

**Input:** A validation set and a grid of candidate temperatures and weights.

**Output:** A selected (T, α) pair with evidence.

**Rationale:** Temperature controls how much of the non-target distribution is visible; alpha controls how much the label constrains the student. Both are hyperparameters that interact with student capacity.

A practical grid:

```text
T ∈ {1, 2, 4, 8}
α ∈ {0.1, 0.3, 0.5, 0.7}
```

Evaluate on validation data with the final metric, not on training loss. A configuration that minimizes training loss can produce a student that is overconfident or that follows the teacher's errors too closely.

**Branch point:** If the best temperature is at the edge of the grid, extend the grid. If the best alpha is 1.0, the teacher is adding nothing and the problem is the signal or the teacher, not the weight.

**Edge case:** Temperature interacts with the number of classes. With many classes, a high temperature can flatten the distribution so much that the KL term becomes dominated by noise in the tail. Evaluate the effective number of classes in the softened distribution, not only the temperature value.

### Step 5. Prepare the transfer set

**Input:** Labeled data, optionally unlabeled data, and the teacher.

**Output:** A transfer set with soft targets and a documented provenance.

**Rationale:** The transfer set determines what the student learns. It can be the training set, a subset, or unlabeled data; each choice has different bias and cost implications.

Options:

- **Labeled training set:** simplest, but the student sees the same examples as the teacher and may overfit the same way.
- **Unlabeled data:** can broaden coverage, but the teacher's errors on those inputs become training targets.
- **Augmented data:** improves robustness but can produce soft targets that are inconsistent with the label.
- **Site-stratified data:** improves subgroup coverage but requires that the teacher performs well on every site.

**Branch point:** If unlabeled data is used, evaluate the teacher's accuracy on a labeled subset of it. A teacher that is unreliable on the transfer set produces unreliable targets.

**Edge case:** A transfer set that over-represents one site teaches the student that site's shortcuts. Stratify the set and report its composition.

### Step 6. Train the student with the combined loss

**Input:** The student architecture, the transfer set, the teacher, T, and α.

**Output:** A trained student checkpoint and a training log.

**Rationale:** The training procedure determines whether the student converges to a good solution. The log is the evidence that the run is reproducible.

The procedure:

1. Initialize the student randomly, or from a subset of teacher layers if the architectures allow.
2. Freeze the teacher and generate soft targets, either on the fly or cached.
3. Compute the hard-label cross-entropy and the soft-target KL term.
4. Weight them by α and (1 - α) × T².
5. Train with the same optimizer and schedule family used for the baseline, adjusted for the smaller model.
6. Monitor both loss terms separately, not only the sum.

**Branch point:** If the KL term dominates and the student's hard-label accuracy stalls, alpha is too low or the teacher is too far from the student's capacity. Increase alpha or reduce student size.

**Edge case:** Caching soft targets saves compute but fixes the teacher distribution; on-the-fly targets allow augmentation but cost a teacher forward pass per batch. Record which was used.

### Step 7. Calibrate the student after training

**Input:** The trained student and a calibration set disjoint from the held-out set.

**Output:** A calibrated student and a calibration report.

**Rationale:** Distillation does not guarantee calibration. A student can match the teacher's ranking while producing probabilities that are too extreme or too flat.

Methods:

- **Temperature scaling:** fit a single temperature on the calibration set. Simple and usually effective for neural networks.
- **Platt scaling:** fit a logistic function, useful for small calibration sets.
- **Isotonic regression:** flexible but prone to overfitting with small samples.
- **Histogram binning:** simple, but produces a step function.

**Branch point:** If the calibration fit changes the operating threshold, the threshold must be re-selected and the change documented. A calibration correction is part of the model, not a reporting adjustment.

**Edge case:** A calibration set drawn from a different distribution than the deployment population can make calibration worse. Stratify and audit.

### Step 8. Evaluate against the teacher, the baseline, and the floor

**Input:** The student, the teacher, the label-trained baseline, and the held-out set.

**Output:** A comparison table with confidence intervals.

**Rationale:** The relevant question is not whether the student is close to the teacher, but whether it meets the clinical floor and how it compares with the alternative of training the same architecture on labels alone. A student that matches the teacher but costs more than the baseline is not a good trade.

The table includes:

- Accuracy, sensitivity, specificity at the operating threshold.
- Calibration: ECE, slope, Brier score.
- Subgroup performance with confidence intervals.
- Latency and energy on the target.
- Model size.
- Agreement with the teacher, as a diagnostic rather than a goal.

**Branch point:** If the student matches the teacher but the baseline is better, distillation is not adding value. If the student beats the baseline but not the teacher, the question is whether the gap is clinically acceptable.

**Edge case:** A student can have higher overall accuracy than the teacher and worse subgroup performance, because the capacity reduction removes a rare-class feature. Always inspect the subgroup table.

### Step 9. Combine with quantization, in the right order

**Input:** The distilled student and the quantization plan.

**Output:** A quantized student with accuracy and calibration results.

**Rationale:** Distillation and quantization interact. Distilling a student that will be quantized should use quantization-aware training (QAT) or at least evaluate the quantized model, because the student's capacity margin may be consumed by quantization.

Order:

1. Distill the student in floating point.
2. Evaluate.
3. Quantize with post-training quantization (PTQ); if accuracy fails, use QAT starting from the distilled weights.
4. Re-evaluate accuracy, calibration, and subgroups on the quantized model.
5. Measure latency and energy on the target.

**Branch point:** If the distilled student has little accuracy margin, quantizing it will likely fail. Consider a larger student or a weaker quantization scheme before spending a QAT run.

**Edge case:** QAT from a distilled checkpoint can converge faster than QAT from scratch, but it can also inherit the teacher's biases more strongly. Audit the quantized model, not only the float student.

### Step 10. Record the run and classify the change

**Input:** All artifacts and results.

**Output:** A distillation record and a change classification.

**Rationale:** A distilled model is a new architecture. The record must allow reproduction and the classification must determine the regulatory path.

The record includes:

- Teacher identity, version, training data, and audit results.
- Student architecture, parameter count, and initialization.
- Transfer set composition, size, and provenance.
- Temperature, alpha, optimizer, schedule, and seed.
- Soft-target generation method (cached or on-the-fly).
- Calibration method and data.
- Accuracy, calibration, subgroup, latency, and energy results.
- Quantization steps applied after distillation.

**Branch point:** If the change is not covered by the PCCP, the options are a new submission, a redesign within the approved envelope, or a decision not to distill. The last is a legitimate outcome and should be recorded as such.

**Edge case:** A distillation run that cannot be reproduced because the teacher checkpoint or the transfer set version is missing is not evidence, even if the metrics are good.

### 5.1 Decision table for distillation outcomes

| Observation | Likely cause | Next action |
|---|---|---|
| Student far below label-trained baseline | Capacity too small or alpha too low | Increase student size; raise alpha |
| Student matches baseline, not teacher | Teacher signal not transferable at this capacity | Feature signal; larger student |
| Student matches teacher, calibration poor | Teacher miscalibrated or temperature too high | Post-hoc calibration; audit teacher |
| Student better overall, worse subgroup | Rare-class feature lost in compression | Reweight transfer set; larger student |
| Distillation gives no gain over labels | Teacher and label agree; task is easy | Skip distillation; use labels |
| Student unstable across seeds | Training too sensitive to initialization | More seeds; learning-rate warmup |
| Quantized student fails after distillation | Accuracy margin consumed | Larger student; QAT; mixed precision |
| Student latency exceeds budget | Architecture choice, not distillation | Revisit Step 1 |

### 5.2 Edge cases in distillation

**Outlier channels.** A teacher's activation or weight outlier can dominate the soft targets if the outlier occurs in a logit. The student learns an extreme probability that is not supported by the data. Diagnose by examining the teacher's logit distribution and clipping extreme logits before the softmax.

**Per-tensor versus per-channel scales.** When the student is quantized, per-tensor scales may collapse small logit differences that distillation worked to preserve. Per-channel quantization of the final layer is often necessary to keep the transferred ranking.

**Activation clipping.** Clipping the student's activations can remove the small differences that encode dark knowledge. Evaluate calibration after clipping, not only accuracy.

**Operator fallback.** A student architecture chosen for distillation may use operators the target does not accelerate. The latency benefit of the smaller model can be lost to fallback.

**Unsupported operators.** Feature-based distillation often requires intermediate outputs that the deployment runtime does not expose. The training graph and the deployment graph diverge, and the deployed model may not match the distilled one.

**Zero-point drift.** A student quantized after distillation has zero points computed from its own activation distribution, which differs from the teacher's. Recalibrate after distillation; do not reuse the teacher's parameters.

**Calibration-set bias.** A transfer set from one site teaches the student that site's distribution. The student's calibration will be wrong elsewhere, and the error will be invisible in the overall metric.

**Thermal throttling.** A student that is only marginally faster than the teacher may not change the thermal profile enough to avoid throttling. Measure sustained behavior, not only burst latency.

### 5.3 A tuning grid and how to read it

The grid below is a template. The values are placeholders; the structure is the point.

| T | α | Student sensitivity | ECE | Worst subgroup | Verdict |
|---:|---:|---:|---:|---:|---|
| 1 | 0.3 | 0.930 | 0.026 | 0.913 | Subgroup below floor |
| 2 | 0.3 | 0.934 | 0.027 | 0.919 | Best overall |
| 4 | 0.3 | 0.933 | 0.029 | 0.918 | Acceptable |
| 8 | 0.3 | 0.931 | 0.031 | 0.916 | Calibration degrading |
| 2 | 0.1 | 0.935 | 0.030 | 0.917 | Teacher-dominated |
| 2 | 0.5 | 0.932 | 0.028 | 0.918 | Balanced |
| 2 | 0.7 | 0.930 | 0.029 | 0.916 | Label-dominated |
| 2 | 1.0 | 0.928 | 0.031 | 0.914 | No distillation |

Read the table in three passes.

- **Rows that fail a constraint are removed first.** T=1 fails the subgroup floor; α=1.0 fails it too. No amount of averaging rescues them.
- **Among the survivors, choose by the metric that is binding.** If the overall floor is tight, T=2, α=0.1 has the highest overall sensitivity but a thinner subgroup margin. If the subgroup floor is tight, T=2, α=0.3 is safer.
- **Do not choose the maximum of a single column.** The maximum overall often comes at the cost of the minimum subgroup, which is the pattern the table exists to expose.

The grid also shows why reporting a single distilled result is uninformative. The same architecture spans 0.928 to 0.935 in overall sensitivity and 0.913 to 0.919 in the worst subgroup, depending on two hyperparameters. A claim without them is not reproducible.

### 5.4 Training instabilities and their signatures

Distillation runs fail in a small number of recognizable ways.

- **The KL term collapses to zero immediately.** The student has memorized the training set through the hard term, or the teacher's distribution is nearly one-hot. Check the teacher's entropy; if it is near zero, raise T before training.
- **The hard term plateaus while the KL keeps falling.** The student is matching the teacher's distribution but not the labels. Alpha is too low, or the teacher is wrong on a subset. Inspect examples where the teacher and label disagree.
- **Both terms oscillate.** The learning rate is too high for the combined loss, or the temperature is changing during training. Fix T and reduce the soft-term learning rate.
- **The student's training loss is low and validation loss is high.** The student has overfit the transfer set, which is common with a small set and a large alpha. Augment or reduce capacity.
- **The student's confidence exceeds the teacher's.** The capacity reduction has removed the teacher's uncertainty representation. Apply post-hoc calibration and check the reliability diagram.
- **Results vary widely across seeds.** The student is near a capacity threshold. Either increase capacity or accept a distribution of outcomes and report it.

Each signature points to a specific intervention. A team that responds to all of them with "train longer" will spend the schedule without fixing the cause.

### 5.5 Unlabeled transfer and its hidden dependency

Unlabeled data is attractive because it is abundant. In a clinical setting it is also sensitive, and its use for distillation has consequences that are easy to overlook.

- **The teacher's errors become labels.** On an unlabeled input, the only supervision is the teacher's output. If the teacher is wrong, the student is trained to be wrong with high confidence.
- **The distribution of unlabeled data is not the distribution of labeled data.** A pool of unlabeled recordings may over-represent easy cases, because difficult cases are more likely to be referred and labeled.
- **The legal basis for use must cover training.** A dataset collected for care may not be consented for model development. This is a governance question, not an engineering one.
- **The transfer set becomes part of the device's design history.** Its composition, source, and version must be recorded, because the student's behavior depends on it.

A practical check is to label a random sample of the unlabeled pool and measure the teacher's accuracy on it. If the teacher is materially worse there than on the labeled set, the unlabeled pool is out of distribution for the teacher, and the student will learn the teacher's behavior in a region where it is unreliable.

The alternative is to use the labeled training set and accept the reduced coverage. That choice is often correct for a regulated product, because it keeps the evidence chain short and the data provenance clear.

### 5.6 Ensembles, self-distillation, and born-again training

Three variants appear often enough to warrant a short note.

**Ensemble distillation.** Several teachers produce an averaged or geometric-mean distribution, and the student learns from the ensemble. The method can reduce the variance of the teacher's errors and improve the student's robustness. The cost is that the evidence chain now includes every teacher, and a subgroup failure could come from any of them.

**Self-distillation.** A model is trained on its own softened outputs, sometimes with a deeper or wider version as the teacher. The method can improve calibration and generalization without an external teacher. It is not compression: the student is usually the same size, and the benefit is accuracy rather than size.

**Born-again training.** A student becomes the teacher for the next student, iteratively. The method can produce a sequence of models of the same size with improved accuracy, or a final smaller model trained through the chain. The evidence chain grows with each generation, and the reproducibility requirement grows with it.

For a regulated product, the simpler the teacher structure, the easier the audit. An ensemble of five teachers may improve the student, but it also means five models to version, audit, and explain. The improvement must justify the evidence burden.

## 6. Worked example

This is a constructed case. The models, measurements, and accuracy values are illustrative and chosen so that the arithmetic closes.

### 6.1 The case

A three-class ECG classifier distinguishes normal rhythm, atrial fibrillation, and other arrhythmia. The teacher is a 420,000-parameter transformer that runs on an edge server. The deployment target is the Cortex-M4F patch from earlier topics, with a 40 ms inference budget and a 64 KiB flash budget.

The student is a 26,000-parameter convolutional network that performs 2,600,000 MACs per inference.

The teacher achieves sensitivity 0.951 for the AF class and ECE 0.018 on the held-out set. The label-trained student of the same architecture achieves 0.928 and ECE 0.031. The distilled student achieves 0.934 and ECE 0.027 before quantization.

### 6.2 Softmax and KL by hand

Teacher logits for one ambiguous window are [2.0, 0.5, -1.0]; student logits before training are [1.6, 0.7, -0.8].

At T = 2:

```text
Teacher:
  z/T        = [1.00, 0.25, -0.50]
  exp        = [2.7183, 1.2840, 0.6065]
  sum        = 4.6088
  p_teacher  = [0.5898, 0.2786, 0.1316]

Student:
  z/T        = [0.80, 0.35, -0.40]
  exp        = [2.2255, 1.4191, 0.6703]
  sum        = 4.3149
  p_student  = [0.5158, 0.3289, 0.1553]
```

The KL divergence:

```text
KL = Σ p_t log(p_t / p_s)
   = 0.5898 × log(0.5898 / 0.5158)
   + 0.2786 × log(0.2786 / 0.3289)
   + 0.1316 × log(0.1316 / 0.1553)
   = 0.5898 × 0.1341
   + 0.2786 × (-0.1659)
   + 0.1316 × (-0.1656)
   = 0.0791 - 0.0462 - 0.0218
   = 0.01104 nats
```

The hard-label cross-entropy, with the true class 1 and the student at T = 1:

```text
p_student(T=1) = [0.6679, 0.2715, 0.0606]
CE = -log(0.6679) = 0.4037 nats
```

The combined loss with α = 0.3 and T = 2:

```text
L = 0.3 × 0.4037 + 0.7 × 4 × 0.01104
  = 0.1211 + 0.0309
  = 0.1520 nats
```

The distillation term is small relative to the hard term at this initialization, which is typical early in training. As the student improves, the KL term shrinks and the hard term dominates near convergence, unless alpha is lowered.

### 6.3 What the temperature does to the information

| Temperature | p(teacher, class 3) | Entropy (nats) | KL to student |
|---:|---:|---:|---:|
| 1 | 0.0391 | 0.6216 | 0.03370 |
| 2 | 0.1316 | 0.9343 | 0.01104 |
| 4 | 0.2187 | 1.0533 | 0.00278 |

At T = 1 the third class carries 3.9% probability and the KL is 0.0337. At T = 4 it carries 21.9% and the KL is 0.0028, because both distributions have flattened toward uniform. The information about the third class has increased in absolute probability but decreased in relative discrimination between teacher and student.

This is the central tuning trade-off: a temperature high enough to expose the dark knowledge, but not so high that the distributions become indistinguishable. The validation sweep in Step 4 is how the trade-off is resolved for a specific model.

### 6.4 Memory before and after

The teacher in FP32:

```text
420,000 × 4 = 1,680,000 B = 1,640.6 KiB
```

The student in INT8 with per-channel metadata for 120 channels:

```text
26,000 + 120 × 5 = 26,600 B = 26.0 KiB
ratio = 1,680,000 / 26,600 = 63.2×
```

The comparison is not like-for-like: the teacher is FP32 and the student is INT8. A fairer comparison is the student in FP32, 104,000 B (101.6 KiB), which is still a 16.2× reduction. The 63.2× figure is the deployment-relevant one if the teacher is never deployed, and the 16.2× figure is the architecture-only reduction.

### 6.5 Latency budget and energy

The student performs 2,600,000 MACs. At 89.6 MMAC/s effective on the Cortex-M4F:

```text
latency = 2,600,000 / 89,600,000 = 0.0290 s = 29.0 ms
```

Against the 40 ms model budget, the margin is 11.0 ms. Against the 250 ms pathway budget, the model uses 11.6% of the total.

Energy at 3.0 V and 8.2 mA:

```text
E = 3.0 × 0.0082 × 0.0290 = 0.713 mJ
```

For comparison, the label-trained student of the same size has the same latency and energy; distillation does not change the architecture, only the weights. The benefit of distillation is accuracy at the same cost, not lower cost at the same accuracy.

### 6.6 Accuracy and calibration

| Model | Sensitivity | Specificity | ECE | Slope | Worst subgroup |
|---|---:|---:|---:|---:|---:|
| Teacher (FP32) | 0.951 | 0.966 | 0.018 | 0.98 | 0.938 |
| Student, labels only | 0.928 | 0.951 | 0.031 | 0.93 | 0.914 |
| Student, distilled (T=2, α=0.3) | 0.934 | 0.955 | 0.027 | 0.95 | 0.919 |
| Student, distilled + INT8 PTQ | 0.931 | 0.953 | 0.033 | 0.92 | 0.916 |
| Student, distilled + INT8 QAT | 0.933 | 0.954 | 0.029 | 0.94 | 0.920 |

Distillation recovers 0.6 percentage points over label training, at the same latency and memory. Quantization then costs 0.3 percentage points, so the final model is 0.3 percentage points above label training and 2.0 below the teacher. The floor is 0.925; the distilled quantized model passes overall with 0.6 points of margin, and its worst subgroup is 0.916, which passes a 0.915 subgroup floor by 0.1 point. That margin is too thin to survive a confidence interval, which is the key risk in the case.

### 6.7 Diagnostic one: outlier-driven scale collapse

The student's first convolution has 40 weights with a minimum of -0.061, a maximum of 0.412, and a median absolute value of 0.0182. The symmetric scale is:

```text
S = 0.412 / 127 = 0.00324409
step / median = 0.003244 / 0.0182 = 0.178
```

Twenty-four of the 255 levels are used, and one weight rounds to zero. The aggregate SQNR is 37.54 dB, which looks acceptable, but the non-outlier weights are represented by roughly four effective bits. The diagnostic is the pair (0.178, 24 levels), not the aggregate.

Affine quantization reduces the step to 0.001855 and raises SQNR to 42.04 dB. It does not remove the outlier; it stops wasting the negative half of the range.

### 6.8 Diagnostic two: per-channel skew

The eight output channels have ranges from 0.0407 to 0.4703, a ratio of 11.6. Under a per-tensor scale, the smallest channel gets about 12 levels and the outlier channel gets 145. Per-channel scales raise SQNR from 37.54 to 51.37 dB, a gain of 13.83 dB.

The metadata cost is 40 bytes for the layer and 600 bytes for the model. On a 26,600-byte model, that is 2.3%, which is a material fraction at this size. The metadata overhead grows as the model shrinks, which is why small models need explicit byte accounting rather than a nominal 4× ratio.

### 6.9 Varying one assumption

**Temperature: T = 2 to T = 1.** At T = 1 the dark knowledge is compressed into small probabilities. The student's sensitivity falls from 0.934 to 0.930, and the worst subgroup falls from 0.919 to 0.913, below the 0.915 floor. The loss is 0.4 percentage points overall and 0.6 in the subgroup, which is enough to change the deployment decision.

**Temperature: T = 2 to T = 8.** The distribution flattens further and the KL term becomes small. Sensitivity is 0.933, essentially unchanged from T = 2, but calibration worsens: ECE rises from 0.027 to 0.031 because the student learns a flatter distribution than the data supports. The useful range is bracketed.

**Alpha: 0.3 to 0.7.** The label term dominates and the student converges toward the label-trained result: sensitivity 0.931, ECE 0.030, worst subgroup 0.916. The teacher's contribution is reduced but not eliminated. At α = 1.0 the student is label-trained and the distillation is a no-op.

**Calibration set: 512 to 64 windows for the post-hoc fit.** The fitted temperature is noisier and the ECE on the held-out set rises from 0.033 to 0.038 for the quantized student. The accuracy is unchanged. The calibration fit is a small-data problem, and its sample size must be justified.

**Bit width: 8 to 4 for the student's weights.** The model's weight bytes fall to 13,000 + 600 = 13,600 B (13.3 KiB). The first-layer SQNR falls to 12.65 dB and sensitivity falls by about 2.4 percentage points to 0.907, well below the floor. On this small model, INT4 is not viable.

**Layer excluded: the first convolution kept in FP32.** Adding 120 bytes (0.45% of the model) recovers about 0.2 percentage points of sensitivity, because the first layer's error propagates through the distilled student. The trade is favourable, and it illustrates a general rule: keep the input layer at higher precision when the model is small and the layer is cheap.

### 6.10 The decision

The distilled, quantized student meets the latency budget at 29.0 ms with 11.0 ms of margin and the flash budget at 26.6 KiB with 37.4 KiB of margin. Its overall sensitivity of 0.931 clears the 0.925 floor by 0.6 points. Its worst subgroup of 0.916 clears a 0.915 floor by 0.1 point.

That last margin is the decision. A 0.1-point margin is smaller than the width of a 95% confidence interval at any plausible sample size, so the subgroup result is consistent with a true value below the floor. The options are:

1. Enlarge the evaluation sample until the interval is narrow enough to decide.
2. Increase the student size, spending latency margin to buy subgroup margin.
3. Improve the transfer set's coverage of the subgroup.
4. Accept the model with a monitoring requirement and a fallback.

Option 1 is the cheapest and should come first, because the others are expensive and may be unnecessary. Option 2 is the fallback if the subgroup result is real. Option 3 addresses the cause. Option 4 is a clinical and regulatory decision, not an engineering one.

The decision framework, not the specific choice, is what this case teaches: a margin that is smaller than its uncertainty is not a margin.

## 7. Code, query, or calculation

The script implements temperature-scaled softmax, KL divergence, the combined distillation loss, quantization, memory accounting, and error metrics. It reproduces Section 6.

```python
import math


def softmax(logits, temperature=1.0):
    scaled = [z / temperature for z in logits]
    peak = max(scaled)
    exps = [math.exp(z - peak) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy(label, probs):
    return -sum(y * math.log(p) for y, p in zip(label, probs) if y > 0)


def kl_divergence(p, q):
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0)


def distillation_loss(teacher_logits, student_logits, label,
                      temperature=2.0, alpha=0.3):
    p_t = softmax(teacher_logits, temperature)
    p_s_t = softmax(student_logits, temperature)
    p_s_1 = softmax(student_logits, 1.0)
    hard = cross_entropy(label, p_s_1)
    soft = kl_divergence(p_t, p_s_t)
    total = alpha * hard + (1 - alpha) * temperature ** 2 * soft
    return {"hard": hard, "kl": soft, "total": total,
            "p_teacher": p_t, "p_student": p_s_t}
```

Compute the example:

```python
teacher = [2.0, 0.5, -1.0]
student = [1.6, 0.7, -0.8]
label = [1.0, 0.0, 0.0]

for t in (1.0, 2.0, 4.0):
    result = distillation_loss(teacher, student, label, temperature=t, alpha=0.3)
    print("T=%.0f teacher=%s" % (t, [round(p, 4) for p in result["p_teacher"]]))
    print("     student=%s kl=%.5f total=%.5f"
          % ([round(p, 4) for p in result["p_student"]], result["kl"], result["total"]))

print("entropy T=1 %.4f" % -sum(p * math.log(p) for p in softmax(teacher, 1.0)))
print("entropy T=2 %.4f" % -sum(p * math.log(p) for p in softmax(teacher, 2.0)))
print("entropy T=4 %.4f" % -sum(p * math.log(p) for p in softmax(teacher, 4.0)))
```

Expected output:

```text
T=1 teacher=[0.7856, 0.1753, 0.0391]
     student=[0.6679, 0.2715, 0.0606] kl=0.03370 total=0.14469
T=2 teacher=[0.5898, 0.2786, 0.1316]
     student=[0.5158, 0.3289, 0.1553] kl=0.01104 total=0.15202
T=4 teacher=[0.463, 0.3182, 0.2187]
     student=[0.426, 0.3402, 0.2338] kl=0.00278 total=0.15228
entropy T=1 0.6216
entropy T=2 0.9343
entropy T=4 1.0533
```

Memory accounting:

```python
def model_bytes(params, channels, bits=8):
    return params * bits / 8 + channels * 5


teacher_bytes = 420_000 * 4
student_fp32 = 26_000 * 4
student_int8 = model_bytes(26_000, 120, 8)
print("teacher fp32 =", teacher_bytes, "B =", round(teacher_bytes / 1024, 1), "KiB")
print("student fp32 =", student_fp32, "B =", round(student_fp32 / 1024, 1), "KiB")
print("student int8 =", student_int8, "B =", round(student_int8 / 1024, 1), "KiB")
print("ratio =", round(teacher_bytes / student_int8, 2))
print("latency ms =", round(2_600_000 / 89_600_000 * 1000, 2))
print("energy mJ =", round(3.0 * 0.0082 * 0.0290, 3))
```

Expected output:

```text
teacher fp32 = 1680000 B = 1640.6 KiB
student fp32 = 104000 B = 101.6 KiB
student int8 = 26600.0 B = 26.0 KiB
ratio = 63.16
latency ms = 29.02
energy mJ = 0.713
```

Quantization and error metrics:

```python
def round_half_away(value):
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def symmetric_scale(values, bits=8):
    return max(abs(v) for v in values) / (2 ** (bits - 1) - 1)


def affine_params(values, bits=8):
    lo, hi = min(values), max(values)
    scale = (hi - lo) / (2 ** bits - 1)
    zero_point = min(2 ** bits - 1, max(0, round_half_away(-lo / scale)))
    return scale, zero_point


def quantize(values, scale, zero_point=0, bits=8, symmetric=False):
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax - 1
    else:
        qmin, qmax = 0, 2 ** bits - 1
    return [min(qmax, max(qmin, round_half_away(v / scale) + zero_point))
            for v in values]


def dequantize(codes, scale, zero_point=0):
    return [(q - zero_point) * scale for q in codes]


def metrics(original, reconstructed):
    n = len(original)
    mse = sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / n
    signal = sum(a * a for a in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, reconstructed))
    return {"mae": sum(abs(a - b) for a, b in zip(original, reconstructed)) / n,
            "rmse": math.sqrt(mse),
            "sqnr_db": 10 * math.log10(signal / noise) if noise else float("inf")}


import random
random.seed(7)
weights = [round(random.gauss(0, 0.035), 4) for _ in range(39)]
weights.insert(11, 0.412)

s = symmetric_scale(weights, 8)
qs = quantize(weights, s, bits=8, symmetric=True)
sa, zp = affine_params(weights, 8)
qa = quantize(weights, sa, zp, 8)
print("symmetric", round(s, 8), {k: round(v, 4) for k, v in metrics(weights, dequantize(qs, s)).items()})
print("affine", round(sa, 8), "zp", zp, {k: round(v, 4) for k, v in metrics(weights, dequantize(qa, sa, zp)).items()})
print("int4 model bytes =", model_bytes(26_000, 120, 4))
```

Expected output:

```text
symmetric 0.00324409 {'mae': 0.0008, 'rmse': 0.0009, 'sqnr_db': 37.5354}
affine 0.0018549 zp 33 {'mae': 0.0005, 'rmse': 0.0006, 'sqnr_db': 42.0374}
int4 model bytes = 13600.0
```

Assumptions: the teacher's logits and student's logits are the values shown; the loss uses natural logarithms; temperature is applied to both models; the hard term uses T = 1; metadata is 5 bytes per channel; latency uses the INT8 effective throughput; and energy uses board current at 3.0 V.

## 8. Data, standards, and regulatory context

Distillation is not standardized. No ISO or IEC document defines a temperature, an alpha, or a soft-target format. What standards constrain is the evidence that the distilled model is safe and effective for its intended use, and the change control that governs its substitution for the teacher or the baseline.

### 8.1 IEC 62304

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), is recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79. It requires requirements, architecture, detailed design, verification, integration testing, and configuration management.

For distillation:

- The student architecture is a design item with a rationale for its size and signal.
- The teacher is a development tool and, if it is used to generate targets, a configuration item.
- The transfer set is a configuration item with provenance.
- The training procedure, including temperature, alpha, and seed, is part of the design record.
- The distilled model is verified against the same requirements as the baseline.

A distillation run recorded only as a training script with no teacher version or transfer set version does not meet the configuration-management expectation.

### 8.2 ISO 14971

ISO 14971, "Medical devices — Application of risk management to medical devices", requires hazards to be identified and controlled. Distillation can introduce hazards:

- The student inherits a teacher's subgroup error, which may be larger after capacity reduction.
- The student's calibration changes, altering the false-alarm rate at the operating threshold.
- The student learns a shortcut that the teacher used, which may not generalize to a new site.
- The student's smaller capacity removes a rare-class feature.

Each is a foreseeable hazard requiring a control and a residual-risk judgement. The control may be a subgroup monitoring requirement, a fallback model, or a restriction on the intended use.

### 8.3 FDA Good Machine Learning Practice

The FDA, Health Canada, and the United Kingdom Medicines and Healthcare products Regulatory Agency published ten guiding principles for Good Machine Learning Practice (GMLP) in October 2021. Four are directly relevant:

- Principle 3: clinical study participants and data sets are representative of the intended patient population.
- Principle 4: training data sets are independent of test sets.
- Principle 6: model design is tailored to the available data and reflects the intended use.
- Principle 10: deployed models are monitored for performance and re-training risks are managed.

A transfer set that is not representative violates principle 3. A distillation run that tunes temperature on the test set violates principle 4. A student chosen only for size without regard to the intended use violates principle 6. A distilled model deployed without monitoring violates principle 10.

### 8.4 FDA PCCP

The FDA's final guidance on Predetermined Change Control Plans for AI-enabled device software functions was issued on 3 December 2024, with a Federal Register notice on 4 December 2024, under section 515C of the Food, Drug, and Cosmetic Act as added by the Food and Drug Omnibus Reform Act of 2022. A PCCP has three components: a description of modifications, a modification protocol, and an impact assessment.

Distillation is a larger change than quantization because it changes the architecture. A PCCP that anticipated "periodic re-training of the approved architecture" may not cover a switch to a distilled student. A PCCP that anticipated "architecture changes within a specified parameter range, validated by a specified protocol" might. The distinction is factual and must be checked against the plan, not assumed.

### 8.5 EU framework

In the European Union, Regulation (EU) 2017/745 governs the device and Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations for high-risk AI systems. A medical device that is a safety component, or is itself a device requiring third-party conformity assessment, is high-risk under Article 6(1).

The applicability date for Annex I products was originally 2 August 2027; the Commission's AI Omnibus, adopted on 19 November 2025 with political agreement reported on 7 May 2026, moves it to 2 August 2028. Verify the current date before relying on it. MDCG 2025-6 (June 2025) remains the reference for the interaction between the two regulations.

For distillation, the relevant obligations are transparency about the model's development, human oversight, and post-market monitoring. A student that differs from the teacher in subgroup performance is a transparency issue as well as an engineering one.

### 8.6 What is binding and what is convention

| Item | Status | Consequence for distillation |
|---|---|---|
| IEC 62304 lifecycle | Binding where required | Teacher, transfer set, and procedure are records |
| ISO 14971 risk management | Binding where required | Bias transfer and calibration shift enter the risk file |
| GMLP principles | Non-binding guidance | Representativeness and independence are expected |
| FDA PCCP guidance | Non-binding guidance; statutory pathway binding | Architecture change may require a new submission |
| EU AI Act | Binding regulation, dates moving | Transparency and monitoring obligations |
| Temperature and alpha values | Convention | No standard; document what was used |
| "Distillation" as a method name | Convention | Describe the signal, loss, and data |

### 8.7 What the standards do not say

- No standard defines a distillation method or a soft-target format.
- No standard specifies a temperature, an alpha, or a student size.
- No standard defines "teacher equivalence", so the acceptance margin is a clinical judgement.
- No standard requires subgroup reporting, although GMLP's representativeness principle implies it.
- No standard says whether a distilled model is a new device; that depends on the change and the approved plan.

The absence is not a gap to be filled by a vendor's interpretation. It means the team must write down its own acceptance criteria, justify them clinically, and apply them consistently. A criterion invented after the results is not a criterion.

## 9. Industry practice and case evidence

### 9.1 The foundational result

Hinton, Vinyals, and Dean introduced soft-target distillation in 2015 (arXiv:1503.02531). The paper's contribution was not a new architecture but a training objective: train the student on the teacher's temperature-softened distribution, and the student learns the relative similarities among classes that a one-hot label does not express. The paper reported gains on MNIST and a commercial acoustic model, and introduced specialist models for fine-grained classes.

The result has been replicated widely. The important qualification is that the benefit depends on the teacher being better than the student and on the soft targets carrying information the labels do not. When the teacher and the labels agree on almost every example, distillation has little to transfer.

### 9.2 Transformer distillation

DistilBERT (Sanh et al., 2019) is the most cited transformer distillation result. The student has 6 layers instead of 12 and 66 million parameters instead of 110 million, a 40% reduction. The paper reports 60% faster inference on CPU (410 seconds versus 668 seconds for a full STS-B pass at batch 1 on an Intel Xeon E5-2690 v3), 97% of BERT-base's GLUE performance, and 71% faster on an iPhone 7 Plus for a question-answering application excluding tokenization.

The method combines a distillation loss on the output distribution, a masked-language-modeling loss, and a cosine loss on hidden states. The student is initialized from every other teacher layer, which the paper's ablations show matters: random initialization costs about 3.7 GLUE points in one reported ablation.

TinyBERT and MobileBERT extended the approach with layer-wise attention transfer and bottleneck architectures. Their reported gains are task- and configuration-specific, and the numbers should be read as evidence that the method can work, not as predictions for a new model.

### 9.3 Distillation for language models

For large language models, distillation takes several forms:

- **Logit distillation:** the student matches the teacher's next-token distribution. It requires a shared tokenizer and vocabulary, and it is expensive because the teacher's logits are large.
- **Sequence-level distillation:** the student is trained on teacher-generated sequences. It is simpler and works across tokenizers, but it learns only the sampled behavior, not the full distribution.
- **Rationale distillation:** the student is trained on the teacher's explanations or chain-of-thought. It can transfer reasoning strategies, and it can also transfer reasoning errors.
- **Preference distillation:** the student is aligned to preferences, usually with a reward model or human comparisons. It requires a careful evaluation of what the preference encodes.

A distilled language model is a new clinical model. Factuality, grounding, omission, and safety must be evaluated on the student, not inferred from the teacher. The smaller model may be more prone to hallucination in narrow domains because it has less capacity to represent uncertainty.

### 9.4 What the case evidence establishes

| Claim | Status |
|---|---|
| Soft targets transfer class similarity information | Established by Hinton et al. and replications |
| DistilBERT is 40% smaller and 60% faster on CPU | Authors' measurement, specific hardware and task |
| DistilBERT retains 97% of GLUE | Authors' measurement, specific benchmark |
| Teacher initialization matters | Authors' ablation, specific architecture |
| Distillation always improves a smaller model | Not supported; depends on teacher, data, and capacity |
| Distillation preserves subgroup performance | Not established; subgroup results are rarely reported |
| A distilled medical model is safe | Not established by any paper; requires device-specific evidence |

The last three rows are the ones a deployment team must not skip. The literature establishes the mechanism; it does not establish the safety of a particular student for a particular population.

### 9.6 A medical distillation checklist

The following checklist is derived from the evidence above and from the regulatory expectations in Section 8. It is not a standard; it is the minimum a reviewer is likely to ask for.

- The teacher has a documented version, training data description, and audit.
- The student's size was chosen by a sweep, not by a guess.
- The transfer set's composition is described and stratified.
- Temperature and alpha were selected on validation data, not on the test set.
- The student is compared with a label-trained model of the same architecture.
- Calibration is reported before and after any post-hoc fit.
- Subgroup results include confidence intervals.
- The quantized student is evaluated separately from the float student.
- Latency and energy are measured on the target, not estimated.
- The change is classified against the approved plan.
- A fallback model or rule exists if the student fails in service.
- The monitoring plan covers subgroup drift, not only overall accuracy.

A team that can produce all twelve items has a distillation package. A team that can produce only the accuracy number has a result, not a device.

### 9.5 Why the medical case differs

Three differences make medical distillation harder than benchmark distillation.

**The label is expensive and the prevalence is low.** A student trained on soft targets from a teacher that saw the same limited positive examples inherits the teacher's uncertainty about the positive class. In a low-prevalence setting, the negative class dominates the loss, and the student can appear accurate while missing the cases that matter.

**The subgroup structure is clinically meaningful.** A benchmark reports an average; a device must report performance for the populations it will serve. A student that matches the teacher on average but loses a rare-class feature in a small subgroup is not equivalent, even if the average is unchanged.

**The change is regulated.** A benchmark model can be replaced by a distilled version in a week. A medical device must establish that the change is within its approved envelope, which requires evidence that the benchmark does not produce.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Student too small | Distilled accuracy below label baseline | Capacity gap | Wasted run, missed floor | Capacity sweep |
| Student too large | No gain over labels, budget exceeded | Excess capacity, teacher adds nothing | Cost without benefit | Stop at the plateau |
| Biased teacher | Subgroup gaps match teacher | Bias transfer | Equity harm | Teacher audit; reweighting |
| Miscalibrated teacher | Student ECE worse than baseline | Confidence transfer | Wrong threshold, alarm rate | Calibrate teacher or student |
| Temperature too low | Student ignores non-target classes | Dark knowledge hidden | No gain over labels | Sweep T |
| Temperature too high | Probabilities near uniform | Loss of discrimination | Calibration loss | Sweep T; entropy check |
| Alpha too high | Student equals label-trained baseline | Teacher signal suppressed | No transfer | Lower alpha |
| Alpha too low | Student drifts from labels | Teacher errors dominate | Accuracy loss | Raise alpha |
| Transfer set biased | Subgroup results worse than teacher | Distribution mismatch | Site-specific failure | Stratify; audit composition |
| Feature mapping wrong | Feature distillation worse than response | Incompatible representations | No transfer | Validate signal |
| Quantization after distillation | Accuracy falls below floor | Margin consumed | Deployment failure | QAT; larger student |
| Calibration fit on test data | ECE improves unexpectedly | Leakage | Invalid evidence | Disjoint calibration set |
| Unreproducible run | Metrics differ between runs | Missing seed or checkpoint | Evidence invalid | Record everything |
| Unclassified change | Regulatory question at launch | Architecture change not covered | Delay | Classify at design time |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The teacher is better than the student on the task | Transfer must have a source | Teacher overfit or biased | Teacher audit |
| The teacher's soft targets carry information beyond labels | Distillation must add signal | Teacher and labels agree | Compare distilled vs label-trained |
| The transfer set represents deployment | Student generalizes | Site or demographic shift | Stratified evaluation |
| The student can represent the teacher's function | Capacity must suffice | Capacity gap | Size sweep |
| Temperature and alpha transfer across tasks | Hyperparameters are reusable | Class count, imbalance | Re-tune per task |
| Calibration is stable after distillation | Threshold remains valid | Distribution shift | Calibration audit |
| Feature mappings are meaningful | Feature distillation works | Architecture mismatch | Compare signals |
| Quantization margin survives | Deployment accuracy holds | Small student, tight floor | Quantize and re-evaluate |
| Subgroup performance is preserved | Equity requirement | Rare-class feature loss | Subgroup table |
| The change is within the approved envelope | Regulatory path | Architecture change | PCCP review |
| The run is reproducible | Evidence is valid | Missing seed or data version | Re-run from record |
| Teacher latency is irrelevant | Teacher is not deployed | Teacher on the critical path | Check deployment architecture |

Two limits deserve emphasis.

The first is that distillation cannot create information. If the teacher never saw a subgroup, the student cannot learn it from the teacher, and the soft targets will encode the teacher's ignorance as confidence. More data, not more distillation, is the remedy.

The second is that the student's errors are not a scaled copy of the teacher's. Capacity reduction changes which features survive, and the change can be favorable or unfavorable for a given subgroup. The only way to know is to measure the student, not to reason from the teacher's results.

## 12. Exercises

**Exercise 1 — Softmax and KL (calculation, 30 minutes).**
Teacher logits [3.0, 1.0, 0.0], student logits [2.4, 1.2, -0.2]. Compute the temperature-2 softmax for both, the KL divergence, and the entropy of the teacher distribution at T = 1, 2, and 4.
*Deliverable:* hand calculations with four decimal places.
*Acceptance criteria:* probabilities sum to 1; KL non-negative; entropy increases with temperature.
*Time:* 30 minutes.

**Exercise 2 — Temperature and gradients (analysis, 35 minutes).**
Explain why the T² factor appears in the distillation loss and what happens to the effective learning rate of the soft term if it is omitted.
*Deliverable:* a 250-word explanation.
*Acceptance criteria:* refers to the gradient of the softmax; states that the soft term's gradient scales as 1/T²; explains the compensation.
*Time:* 35 minutes.

**Exercise 3 — Signal choice (application, 30 minutes).**
A convolutional teacher and a recurrent student must be distilled for a signal-classification task. Choose a signal, justify it, and state the infrastructure required.
*Deliverable:* a one-page recommendation.
*Acceptance criteria:* the choice follows from the architecture mismatch; the infrastructure is named; an alternative is considered.
*Time:* 30 minutes.

**Exercise 4 — Teacher audit (critique, 40 minutes).**
Given a teacher with 0.95 overall sensitivity and 0.88 in a subgroup, decide whether to distill. State the conditions under which you would proceed and the mitigations you would require.
*Deliverable:* a 300-word decision note.
*Acceptance criteria:* names bias transfer; proposes at least two mitigations; states a stopping condition.
*Time:* 40 minutes.

**Exercise 5 — Capacity sweep (design, 45 minutes).**
Design a sweep over student widths of 16, 24, 32, and 48 channels, with fixed data, temperature, and alpha. State the metrics, the stopping rule, and how you would choose.
*Deliverable:* a sweep plan and a results table template.
*Acceptance criteria:* only one variable changes; metrics include calibration and latency; the choice rule is stated before the results.
*Time:* 45 minutes.

**Exercise 6 — Subgroup evaluation (analysis, 45 minutes).**
Using the table in Section 6.6, identify the model with the best overall sensitivity and the model with the best worst-subgroup sensitivity. Explain why they differ and which you would choose.
*Deliverable:* a short analysis with the table.
*Acceptance criteria:* identifies the models correctly; explains the aggregate-versus-subgroup trade-off; states a clinical rationale.
*Time:* 45 minutes.

**Exercise 7 — Quantization after distillation (calculation, 40 minutes).**
A distilled student has 0.3 percentage points of margin above the floor. Estimate whether INT8 PTQ is likely to fit, using the quantization loss observed in Section 6.6, and state what you would do if it does not.
*Deliverable:* a calculation and a decision.
*Acceptance criteria:* uses the observed 0.3-point loss; concludes correctly; proposes QAT or a larger student.
*Time:* 40 minutes.

**Exercise 8 — Distillation record (design, 60 minutes).**
Write the record for the Section 6 run, including teacher version, transfer set, temperature, alpha, seed, calibration method, and results. Classify the change.
*Deliverable:* a two-page record.
*Acceptance criteria:* reproducible from the record; calibration data described; change classification justified; subgroup results included.
*Time:* 60 minutes.

## 13. Self-check answers

**A1. Softmax and KL for Exercise 1.**
At T = 2, teacher scaled logits [1.5, 0.5, 0.0], exponentials [4.4817, 1.6487, 1.0], sum 7.1304, probabilities [0.6285, 0.2312, 0.1402]. Student scaled [1.2, 0.6, -0.1], exponentials [3.3201, 1.8221, 0.9048], sum 6.0470, probabilities [0.5491, 0.3013, 0.1496]. KL = 0.6285 × log(1.1446) + 0.2312 × log(0.7673) + 0.1402 × log(0.9372) = 0.0849 - 0.0618 - 0.0091 = 0.0147 nats. Teacher entropy: T=1 gives 0.5243, T=2 gives 0.9060, T=4 gives 1.0481 nats.

**A2. Why the T² factor.**
The gradient of the softmax with respect to the logits scales as 1/T. The distillation loss therefore produces gradients that are smaller by a factor of about 1/T² when the logits are near zero-mean. Multiplying by T² restores the gradient magnitude, so changing the temperature does not implicitly change the learning rate of the soft term. Without it, a temperature sweep also sweeps the effective step size, and the comparison is confounded.

**A3. Why response-based distillation is the default for cross-architecture pairs.**
Because it requires only a shared label space, not a shared internal representation. Feature-based methods need a layer mapping and compatible widths, and relation-based methods need structured batches. Response-based is the lowest-infrastructure option and the most portable, at the cost of transferring less intermediate structure.

**A4. When to distill from a teacher with a subgroup gap.**
Proceed only if the gap is understood and mitigated: reweight or oversample the subgroup in the transfer set, correct the teacher's outputs on that subgroup, or use a different teacher. The student will inherit the gap otherwise. If the gap cannot be mitigated, do not distill, because the student's subgroup performance is the deployment risk.

**A5. What makes a capacity sweep valid?**
Only the student size varies. Data, temperature, alpha, optimizer, schedule, and seed policy are fixed. The metrics include accuracy, calibration, subgroup performance, and latency. The choice rule is stated before the results, so the sweep is not used to rationalize a preferred size.

**A6. Best overall versus best worst-subgroup.**
In Section 6.6, the distilled float student has the best overall sensitivity among students (0.934), while the label-trained student has the worst (0.928). The distilled INT8 QAT model has 0.933 overall and 0.920 worst subgroup; the float distilled model has 0.934 overall and 0.919. The QAT model is slightly worse overall but better in the worst subgroup, so the choice depends on whether the claim is aggregate or subgroup-specific. If the claim applies to every subgroup, the QAT model is preferable despite the lower average.

**A7. Will INT8 fit after distillation?**
The distilled float student is 0.934 against a 0.925 floor, a margin of 0.9 points. The observed INT8 loss is about 0.3 points, which leaves 0.6 points of margin. That is likely to fit, but the subgroup margin is only 0.4 points (0.919 against 0.915), which is thin. Proceed with PTQ, then QAT if the subgroup falls below the floor.

**A8. Rubric for Exercise 8 (record).**
A strong record includes teacher identity and version, transfer set composition, temperature, alpha, seed, soft-target method, calibration method and data, accuracy and subgroup results, latency and energy, and a change classification with a justification. A weak record lists the architecture and the final accuracy without the information needed to reproduce the run.

**A9. Rubric for Exercise 5 (capacity sweep).**
A strong plan fixes data, temperature, alpha, optimizer, and seed policy; varies only width; states the metrics and the stopping rule in advance; and includes a label-trained baseline at each size so the distillation gain is visible. A weak plan varies several factors at once, reports only accuracy, or chooses the size after seeing the results without a rule.

**A10. Rubric for Exercise 7 (quantization decision).**
A strong answer uses the observed quantization loss rather than a generic assumption, compares the remaining margin with the subgroup margin, and states a conditional plan: PTQ first, QAT if the subgroup fails, a larger student if both fail. A weak answer assumes quantization is lossless or rejects it without arithmetic.

## 14. Interview questions

**Q1. What is knowledge distillation?**
A strong answer describes training a smaller student on a teacher's softened output distribution plus the hard label, and explains that the soft targets carry class-similarity information. A weak answer calls it "training a small model on a big model's labels".

**Q2. What does the temperature do?**
A strong answer explains that it flattens the softmax, exposing non-target class probabilities, and that too high a temperature destroys discrimination. A weak answer says it "makes the probabilities softer" without the trade-off.

**Q3. Why is there a T² factor in the loss?**
A strong answer connects it to the 1/T scaling of the softmax gradient and explains that T² keeps the soft term's effective learning rate stable across temperatures. A weak answer says it is "for numerical stability".

**Q4. When would you use feature-based rather than response-based distillation?**
A strong answer chooses feature-based when the architectures are similar and intermediate structure is transferable, and notes the layer-mapping cost. A weak answer says feature-based is "more powerful" without conditions.

**Q5. What is the capacity gap?**
A strong answer defines it as the difference in expressive power between teacher and student, explains that too small a student cannot represent the teacher's function, and proposes a size sweep. A weak answer says smaller is always better.

**Q6. How do you audit a teacher before distillation?**
A strong answer checks accuracy, calibration, subgroup performance, edge cases, and shortcut learning on held-out data. A weak answer checks only overall accuracy.

**Q7. Why can distillation worsen subgroup performance?**
A strong answer explains bias transfer and capacity loss: the student inherits the teacher's errors and may lose the rare-class feature that the teacher represented. A weak answer assumes the student is a scaled copy.

**Q8. Does distillation improve calibration?**
A strong answer says it can improve or worsen it, that it depends on the teacher's calibration and the temperature, and that post-hoc calibration is still required. A weak answer assumes soft targets calibrate the student.

**Q9. How do distillation and quantization interact?**
A strong answer says distillation changes the architecture and quantization reduces precision, that the student should be quantized and re-evaluated, and that a small accuracy margin may be consumed. A weak answer treats them as independent.

**Q10. What is the regulatory significance of distillation?**
A strong answer says it is an architecture change, larger than quantization, and may fall outside a PCCP that anticipated only re-training. A weak answer treats it as a training detail.

**Q11. When would you decide not to distill?**
A strong answer: when the teacher is not better than the label-trained student, when the teacher is biased, when the student cannot meet the floor, or when the change path does not cover it. A weak answer always distills because it is available.

**Q12. How would you know the student learned something the labels do not contain?**
A strong answer compares distilled and label-trained students of the same architecture on held-out data, and checks whether the gain persists across seeds and subgroups. A weak answer points to training loss.

## 15. Key terms

- **Alpha** — weight on the hard-label loss in the combined objective.
- **Attention transfer** — distillation from attention maps.
- **Bias transfer** — inheritance of teacher bias by the student.
- **Born-again** — using a student as the next teacher in an iterative process.
- **Calibration** — agreement between predicted probability and observed frequency.
- **Capacity gap** — difference in expressive power between teacher and student.
- **Cross-entropy** — negative log-likelihood of the label under the prediction.
- **Dark knowledge** — information in non-target class probabilities.
- **Distillation loss** — the combined hard and soft objective.
- **Ensemble distillation** — distillation from multiple teachers.
- **Entropy** — uncertainty of a distribution, in nats or bits.
- **Feature-based distillation** — matching intermediate representations.
- **Hard target** — a one-hot or annotated label.
- **KL divergence** — asymmetric divergence between two distributions.
- **Label smoothing** — softening labels with a uniform distribution.
- **Logit** — pre-softmax output.
- **Logit distillation** — distillation from teacher logits, usually with a shared vocabulary.
- **Online distillation** — teacher and student trained together.
- **Rationale distillation** — training on teacher explanations.
- **Relation-based distillation** — matching similarities between examples.
- **Response-based distillation** — matching output distributions.
- **Self-distillation** — distillation within the same model family.
- **Sequence-level distillation** — training on teacher-generated sequences.
- **Shortcut learning** — reliance on spurious correlates.
- **Soft target** — temperature-scaled teacher distribution.
- **Student** — the model being trained.
- **Student sweep** — systematic variation of student size.
- **Subgroup** — population defined by a clinically relevant attribute.
- **Teacher** — the model providing soft targets.
- **Teacher audit** — evaluation of the teacher before distillation.
- **Temperature (T)** — softmax scaling parameter.
- **Temperature scaling** — post-hoc calibration with one scalar.
- **Transfer set** — data used to generate soft targets.
- **Unlabeled transfer** — distillation using unlabeled data.

## 16. Sources

All URLs accessed 25 October 2026.

**Foundational**
- Hinton, G., Vinyals, O., Dean, J., "Distilling the Knowledge in a Neural Network", 2015. https://arxiv.org/abs/1503.02531

**Transformer distillation**
- Sanh, V., et al., "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter", 2019. https://arxiv.org/abs/1910.01108
- Jiao, X., et al., "TinyBERT: Distilling BERT for Natural Language Understanding", 2019. https://arxiv.org/abs/1909.10351
- Sun, Z., et al., "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices", 2020. https://arxiv.org/abs/2004.02984

**Quantization after distillation**
- Jacob, B., et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2017/2018. https://arxiv.org/abs/1712.05877
- Nagel, M., et al., "A White Paper on Neural Network Quantization", 2021. https://arxiv.org/abs/2106.08295

**Regulatory**
- FDA, Health Canada, MHRA, "Good Machine Learning Practice for Medical Device Development: Guiding Principles", October 2021. https://www.fda.gov/media/153486/download
- FDA, PCCP final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. https://www.iso.org/standard/72704.html
- Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6. https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en

**Runtimes**
- PyTorch, "ExecuTorch quantization overview". https://docs.pytorch.org/executorch/stable/quantization-overview.html
- ONNX Runtime, "Quantize ONNX models". https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

> **Note:** DistilBERT figures are the authors' measurements on their hardware. The Section 6 case is constructed for teaching and is not a measured product.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`03_the-compression-family.md`](03_the-compression-family.md)
- Next topic: [`05_quantization.md`](05_quantization.md)
- Related topic: [`06_pruning-and-sparsity.md`](06_pruning-and-sparsity.md)
- Related topic: [`09_medical-and-regulatory-constraints.md`](09_medical-and-regulatory-constraints.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
