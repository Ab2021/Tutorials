# Deep Learning for Signals: Representations, Architectures, Training, and Calibration

> Module 11 — Physiological Signals and Wearables
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (signal fundamentals), Topic 06 (feature engineering)
> Version 2026.3

## 1. Why this topic exists

Deep learning changed what is possible with physiological signals and changed the failure modes. A one-dimensional convolutional network (1D CNN) can learn morphology from a raw electrocardiogram (ECG) without handcrafted features. A recurrent network can model rhythm over time. A transformer can attend across a 30 s epoch. The same flexibility makes it easy to build a model that learns the device, the site, the patient, or the label leakage rather than the physiology.

Four failures recur.

**Leakage through the split.** A model is trained on 30 s segments, and the train and test sets are split by segment. Segments from the same patient appear in both sets. The model memorizes patient-specific morphology and reports an area under the receiver operating characteristic curve (AUROC) of 0.95. A patient-level split gives 0.78. The difference is not a modelling improvement; it is leakage.

**Leakage through preprocessing.** A feature or a normalization is computed on the full dataset before the split. The mean and standard deviation of the test set influence the training data. The model's performance is optimistic by an amount that is hard to quantify but easy to avoid.

**Calibration ignored.** A model with a good AUROC can be badly calibrated. If it predicts 0.9 for a group in which 60% are positive, the threshold-based decision is wrong even though the ranking is good. Calibration is a separate property and a separate metric.

**Aggregation unstated.** A patient has 20 segments, 3 of which are abnormal. The patient-level prediction depends on whether the segments are averaged, max-pooled, or voted. The three rules give different answers, and the choice is often not reported.

This topic covers the deep-learning chain for physiological signals: representations, architectures, receptive fields, training, augmentation, splitting, calibration, aggregation, and monitoring. It builds on Topic 06 for features and sets up Topic 09 for validation and Topic 10 for deployment. The emphasis is on the decisions that determine whether a reported performance is real.

### 1.1 Three composite failures and their costs

**The 0.95 that was 0.78.** A team trains a CNN on 30 s ECG segments from 1,000 patients and splits the segments randomly. The AUROC is 0.95. A reviewer asks for a patient-level split, and the AUROC falls to 0.78. The model has learned patient-specific morphology, not the disease. The cost is a product that appears to work until it meets a new patient, and a regulatory submission that must be redone.

**The threshold that was tuned on the test set.** A team reports an F1 score of 0.82 and a threshold of 0.37. The threshold was chosen by maximizing F1 on the test set. A second team applies the model to a new dataset with the same threshold and gets an F1 of 0.61. The difference is partly population and partly the threshold, which was fitted to the test noise. The cost is a deployment that underperforms the published number.

**The model that learned the hospital.** A multi-site dataset has positive cases concentrated at one site. The model learns a site-specific noise pattern and uses it as a shortcut. The internal test is good because the site is represented in both train and test. The external test at a new site fails. The cost is a model that does not transport, discovered after a expensive prospective study.

The three failures share a structure: the evaluation design, not the architecture, determines whether the number is real. A deep model is powerful enough to exploit any leak, and the leaks are in the split, the preprocessing, and the threshold.

A fourth failure is a model that is never deployed because its compute budget was ignored. A 100 MB ensemble that wins a benchmark cannot run on a watch. The architecture and the deployment target must be designed together, which is the subject of Topic 10.

### 1.2 Why deep learning is not a default

A deep model is the right choice when the data are large, the signal is high-dimensional, and the morphology is hard to specify. It is the wrong choice when the data are small, the label is noisy, or the decision needs an interpretable measurement.

A practical decision rule:

- If the dataset has fewer than about 1,000 patients, start with features and a regularized model.
- If the signal has a well-defined morphology and a small number of classes, a 1D CNN is often enough.
- If the decision needs long-range context, use a sequence model or attention.
- If the decision needs an explanation for a regulator or a clinician, keep an interpretable feature layer alongside the model.

The rule is a starting point, not a law. The important part is that the choice is justified by the data and the decision, not by the availability of a pretrained model.

A second decision is whether to use a pretrained model. A pretrained model can reduce the labelled data requirement, but it adds a dependency on the pretraining corpus and a risk that the corpus's devices and populations are baked into the model. The dependency must be documented, and the fine-tuned model must be validated on the target data.

## 2. Learning outcomes

By the end of this topic you can:

1. **Choose** an input representation (raw waveform, spectrogram, or feature vector) for a stated task, and justify the choice.
2. **Compute** the receptive field, parameter count, and multiply-accumulate count of a 1D CNN.
3. **Design** a patient-level, device-held-out, or site-held-out split, and explain what each protects against.
4. **Apply** augmentation that is physiologically plausible and label-preserving.
5. **Calibrate** a model and report expected calibration error alongside discrimination.
6. **Aggregate** segment-level predictions to a patient-level decision with a stated rule.
7. **Evaluate** a model for subgroup performance, external validity, and failure modes.
8. **Document** a model card with the data, the split, the metrics, and the limits.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

The outcomes are ordered from representation and architecture to evaluation and documentation. The last three are the ones that determine whether a reported number is a claim. A learner who can build a model but cannot design a split or read a calibration plot has the technical skill and not the evaluation skill.

## 3. The core mental model

A deep model is a composition of learned transformations. The architecture determines what the model can see; the data determine what it can learn; the split determines what the reported performance means.

```text
 raw signal (N samples, C channels)
        |
   normalization (train statistics only)
        |
   representation (raw, spectrogram, features)
        |
   encoder (CNN, RNN, transformer)
        |
   embedding
        |
   head (classification, regression)
        |
   probability
        |
   calibration
        |
   aggregation (segment -> patient)
        |
   decision (threshold, alert)
```

Redraw it with the train/test boundary on the left. Everything to the left of the boundary must be fit on training data only; everything to the right is evaluated on held-out data.

The diagram is also a checklist for leakage. Normalization, representation parameters, augmentation statistics, calibration, and the decision threshold are all fitted values. If any of them is fitted on the full data, the boundary is broken and the evaluation is optimistic.

### 3.1 What the architecture can see

The receptive field is the number of input samples that can influence one output. A 1D CNN with kernel 5, stride 2, and three layers has a receptive field of 29 samples, which is 116 ms at 250 samples/s. That is enough to see a QRS complex and not enough to see a whole 30 s rhythm. A model that needs rhythm context needs a larger receptive field, a recurrent layer, or attention.

The receptive field arithmetic is:

```text
 RF = 1 + sum over layers of (kernel - 1) * product of previous strides
```

For kernel 5, stride 2, three layers: 1 + 4 + 8 + 16 = 29 samples. For kernel 7, stride 1, five layers: 1 + 6×5 = 31 samples. A stride of 1 preserves resolution but costs compute; a stride of 2 halves the length at each layer.

### 3.2 What the data can teach

A model learns the distribution of the training data, including its shortcuts. If the positive class comes from one hospital and the negative class from another, the model can learn the hospital. If the positive class is recorded on one device, the model can learn the device. The shortcut is not a bug in the model; it is a property of the data.

The defences are design-level: balance the classes across sites and devices, hold out a site or device, and test whether the model's performance drops when the shortcut is removed. A model that relies on a shortcut will fail the held-out test.

### 3.3 What the split means

The split defines the claim. A random segment split estimates performance on new segments from the same patients. A patient-level split estimates performance on new patients from the same population. A site-held-out split estimates performance on a new site. A prospective split estimates performance in the intended setting.

Most published numbers are segment-level or patient-level. Very few are site-held-out or prospective. The claim should name the split.

### 3.4 The three metrics that matter

- **Discrimination**: AUROC, AUPRC, sensitivity at a fixed specificity. It answers "can the model rank?"
- **Calibration**: expected calibration error, calibration slope, Brier score. It answers "does the probability mean what it says?"
- **Utility**: net benefit, decision curve, or the clinical outcome at the chosen threshold. It answers "does the decision improve?"

A model can be good at one and bad at another. A report that gives only AUROC is incomplete.

The three metrics also have different data requirements. Discrimination can be estimated with a few hundred patients if the effect is large. Calibration needs enough patients in each probability bin, which can be thousands. Utility needs an outcome and a decision rule, which may require a prospective study. The evaluation should be sized to the metric, not the other way around.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| 1D CNN | Convolutional network over time | Assuming it sees long context | architecture |
| Receptive field | Input span influencing one output | Confusing with kernel size | samples or ms |
| Stride | Step between convolution windows | Confusing with pooling | samples |
| Pooling | Downsampling operation | Assuming it preserves timing | method |
| Residual connection | Skip connection around a block | Assuming it solves all training issues | architecture |
| RNN | Recurrent neural network | Assuming it handles long sequences | architecture |
| LSTM | Long short-term memory | Treating as a default | architecture |
| GRU | Gated recurrent unit | Treating as equivalent to LSTM | architecture |
| Transformer | Attention-based architecture | Assuming it needs no positional encoding | architecture |
| Attention | Weighted aggregation over positions | Confusing with interpretability | weights |
| Embedding | Learned representation | Treating as a feature with a unit | vector |
| Segment | Fixed-length input window | Confusing with patient | s |
| Patient-level split | Split by patient | Confusing with record-level | method |
| Site-held-out | Split by collection site | Confusing with patient split | method |
| Data leakage | Test information in training | Ignoring preprocessing leakage | error |
| Augmentation | Label-preserving transformation | Changing the label | method |
| Class imbalance | Unequal class frequencies | Ignoring in accuracy | ratio |
| Class weight | Loss weight per class | Confusing with sampling | dimensionless |
| Focal loss | Loss that down-weights easy examples | Using without tuning | method |
| AUROC | Area under ROC curve | Using with extreme imbalance | dimensionless |
| AUPRC | Area under precision-recall curve | Confusing with AUROC | dimensionless |
| Sensitivity | True positive rate | Confusing with PPV | % |
| Specificity | True negative rate | Confusing with NPV | % |
| PPV | Positive predictive value | Confusing with sensitivity | % |
| Calibration | Agreement of predicted and observed risk | Confusing with discrimination | method |
| ECE | Expected calibration error | Using with small bins | dimensionless |
| Brier score | Mean squared probability error | Confusing with log loss | dimensionless |
| Temperature scaling | Calibration by scaling logits | Assuming it fixes discrimination | method |
| Threshold | Decision cut-point | Choosing on test data | probability |
| Aggregation | Combining segment predictions | Leaving rule unstated | method |
| Majority vote | Segment vote for patient | Ignoring confidence | method |
| Max pooling | Maximum segment probability | Sensitive to one false positive | method |
| Mean pooling | Average segment probability | Dilutes focal events | method |
| Model card | Documentation of a model | Treating as marketing | document |
| Drift | Change in input distribution | Confusing with concept drift | monitoring |
| Concept drift | Change in input-outcome relationship | Confusing with data drift | monitoring |
| Dilated convolution | Convolution with gaps | Confusing with stride | architecture |
| Global pooling | Pooling over the whole length | Assuming it preserves timing | method |
| Softmax | Normalized exponential output | Treating as calibrated | method |
| Logit | Pre-softmax score | Confusing with probability | real number |
| Temperature | Scaling parameter for calibration | Confusing with training temperature | dimensionless |
| Isotonic regression | Monotone calibration map | Assuming it extrapolates | method |
| Reliability plot | Confidence versus accuracy | Confusing with ROC | plot |
| Brier decomposition | Reliability, resolution, uncertainty | Reporting only the total | method |
| Top-k mean | Mean of the k highest segments | Choosing k after seeing test | method |
| Segment-level metric | Metric over segments | Confusing with patient-level | metric |
| Patient-level metric | Metric over patients | Confusing with segment-level | metric |
| Multi-label | More than one label per record | Treating as multiclass | coding |
| Macro average | Average over classes | Ignoring class size | metric |
| Micro average | Average over samples | Ignoring class balance | metric |
| Weighted metric | Metric with class weights | Hiding poor minority performance | metric |
| Confusion matrix | Table of predictions | Reporting only accuracy | table |
| Net benefit | Clinical utility measure | Confusing with accuracy | dimensionless |
| Decision curve | Net benefit across thresholds | Treating as a diagnostic | plot |
| Calibration slope | Slope of observed on predicted | Confusing with intercept | dimensionless |
| Prior shift | Change in prevalence | Confusing with calibration | method |
| Covariate shift | Change in input distribution | Confusing with concept drift | method |

## 5. The mechanics, step by step

The procedure is written for a supervised deep-learning model on a physiological signal with patient identifiers. It assumes the signal has already passed the acquisition and quality steps of Topics 01 to 05.

### Step 1. Define the task, the label, and the unit

**Input:** the clinical or product decision, the available labels, and the patient identifiers.

**Output:** a task specification with the input, the label definition, the unit of prediction, and the evaluation metric.

**Rationale:** the label definition determines what the model learns. "AF present" can mean present on the recording, present in the patient's history, or present during the 30 s window. Three different labels produce three different models.

**Branch point:** if the label is patient-level, the model must aggregate segments. If the label is segment-level, the patient-level decision must be derived. The two require different evaluation.

**Edge case:** a label derived from a billing code is not the same as a label derived from adjudication. The code has a different sensitivity and a different time window.

**The label hierarchy.** A label can be defined at several levels, and the model's task changes with the level.

| Level | Example | Prediction unit | Aggregation |
|---|---|---|---|
| Sample | Abnormal waveform | Beat or window | None |
| Segment | AF in 30 s | Segment | Mean or max |
| Record | AF in 24 h | Record | Max |
| Patient | Ever had AF | Patient | Any record |
| Outcome | Stroke within 1 year | Patient | Survival model |

A model trained on segments and evaluated on patients is not the same model. The label level and the prediction unit must match, or the aggregation must be explicit.

**Label noise.** Clinical labels are noisy. A billing code can be wrong; an over-read can disagree with another over-read. A model trained on noisy labels learns the noise. The noise can be mitigated with adjudication, multiple readers, or a robust loss, but it cannot be eliminated. The label noise sets a ceiling on the achievable performance.

### Step 2. Choose the representation

**Input:** the signal, the sampling rate, and the compute budget.

**Output:** raw waveform, spectrogram, or feature vector, with a justification.

**Rationale:** the representation determines what the first layer sees. Raw waveforms preserve morphology and timing; spectrograms expose frequency content; features are compact and interpretable.

**Trade-offs:**

| Representation | Strength | Weakness | Typical use |
|---|---|---|---|
| Raw 1D | Morphology, timing | Needs more data | ECG, EEG |
| Spectrogram | Frequency content | Loses fine timing | EEG, audio-like |
| Wavelet | Multi-resolution | Redundant | EEG, ECG |
| Feature vector | Compact, interpretable | Limited by features | Small data |
| Multi-modal | Combines context | Complex | Wearables |

**Branch point:** if the dataset is small, a feature vector plus a simple model is often better than a raw-signal deep model. If the dataset is large, raw signal can learn features that the handcrafted set misses.

**Edge case:** a spectrogram computed with a window that is shorter than the event smears the event across time and frequency. The window must match the phenomenon.

**Representation and sampling.** A raw-waveform model inherits the sampling rate of the data. A model trained at 500 Hz and applied at 250 Hz sees a different input length and a different high-frequency content. A spectrogram model inherits the window and the hop, which are additional parameters that must match.

A practical rule is to resample all data to a common rate before training and to record the rate in the model card. The resampling must be an anti-aliased decimation, not a sample drop. If the model needs the high-frequency content, the common rate must be high enough to preserve it.

**Multi-channel input.** A 12-lead ECG is 12 channels; a single-lead wearable is one. A model trained on 12 leads and applied to one lead needs a channel dropout augmentation or a separate model. The channel count is part of the input specification.

### Step 3. Split by patient, device, and site

**Input:** the patient identifiers, the device identifiers, and the site identifiers.

**Output:** train, validation, and test sets with no overlap on the split unit.

**Rationale:** the split determines the claim. A segment split leaks patient identity; a patient split does not. A patient split still leaks device and site, which a device- or site-held-out split removes.

**Method:** group the data by patient, assign whole patients to folds, and stratify the folds by label and site. If the dataset has multiple devices, hold out one device. If it has multiple sites, hold out one site.

**Arithmetic:** with 1,000 patients and 20 segments each, a segment split has 20,000 segments and near-certain patient overlap. A patient split has 800 patients for training and 200 for test, with no overlap. The effective sample size for the patient-level claim is 200, not 20,000.

**Branch point:** if the claim is "works on a new patient," a patient split is sufficient. If the claim is "works on a new device," a device-held-out split is required.

**Edge case:** a patient with recordings on two devices can leak the patient across a device split. The split must be by patient within the device, or the patient must be assigned to one side.

### Step 4. Normalize and augment on training data only

**Input:** the training set and the augmentation policy.

**Output:** normalization statistics and augmented batches.

**Rationale:** normalization statistics computed on the full dataset leak test information. Augmentation that changes the label teaches the wrong thing.

**Augmentations that are usually label-preserving:** time shift within a tolerance, amplitude scaling within a range, additive noise at a stated SNR, small time warping, and channel dropout. **Augmentations that can change the label:** time reversal for ECG, large time warping for rhythm, and frequency shifting for a signal whose diagnosis depends on frequency.

**Arithmetic:** a time shift of ±200 ms is label-preserving for a rhythm classification and label-changing for a beat-alignment task. The tolerance must be defined by the task, not by the augmentation library.

**Branch point:** if the model is evaluated on a device with a different noise level, noise augmentation can improve robustness. If the evaluation is on the same device, the augmentation may only slow training.

**Edge case:** augmentation of the minority class can balance the classes, but it can also create unrealistic examples. A synthetic ECG with an impossible morphology teaches the model an impossible pattern.

### Step 5. Choose the architecture and compute the budget

**Input:** the representation, the receptive field requirement, and the compute budget.

**Output:** an architecture, a parameter count, and a multiply-accumulate count.

**Rationale:** the architecture must see the phenomenon. A 116 ms receptive field cannot classify a 10 s rhythm. A 30 s receptive field with a 1 s kernel and stride 1 is expensive.

**Arithmetic for a 1D CNN:**

- Three conv layers, channels 1→8→16→32, kernel 5, stride 2, padding 2.
- Output lengths: 3,750, 1,875, 938 for a 7,500-sample input.
- Parameters: 48 + 656 + 2,592 + 66 = 3,362.
- Receptive field: 29 samples, 116 ms at 250 samples/s.
- MACs: 3,751,280 per inference.

The model is small enough for a microcontroller and too small to see a 30 s rhythm. The architecture and the task are mismatched if the task needs rhythm context.

**Branch point:** if the task needs long context, use a larger receptive field, a recurrent layer, or attention. If the task needs fine morphology, keep the stride low and the kernel small.

**Edge case:** a model with a large receptive field can learn the recording rather than the event. The receptive field is a capability, not a guarantee.

**Architecture comparison.**

| Architecture | Receptive field | Strength | Weakness |
|---|---|---|---|
| 1D CNN | Local | Morphology, efficient | Limited context |
| Dilated CNN | Large, sparse | Long context, efficient | Grid artifacts |
| LSTM | Sequential | Variable context | Slow, hard to parallelize |
| Transformer | Global | Long-range attention | Data-hungry |
| Hybrid CNN + attention | Local + global | Both | More parameters |

The table is a guide, not a ranking. A small CNN with a 116 ms receptive field is the right choice for a beat-level task and the wrong choice for a rhythm task. A transformer with global attention is the right choice for a long record and the wrong choice for 500 patients.

### Step 6. Train with an appropriate loss and class handling

**Input:** the training set, the loss, and the class weights.

**Output:** a trained model and a training curve.

**Rationale:** class imbalance is common in clinical data. A model trained with unweighted cross-entropy on a 1:100 imbalance learns to predict the majority class and achieves 99% accuracy with zero sensitivity.

**Arithmetic:** with 1% prevalence, a model that predicts negative for every case has 99% accuracy, 0% sensitivity, and an AUROC of 0.5. Accuracy is not a useful metric at this prevalence.

**Branch point:** if the decision is screening, optimize sensitivity at an acceptable false-positive rate. If the decision is confirmation, optimize PPV.

**Edge case:** class weights change the calibration. A model trained with weighted loss outputs probabilities that are not the population probabilities. Recalibrate after training.

**Training diagnostics.** Four curves should be watched.

- **Training and validation loss**: a gap indicates overfitting; both high indicates underfitting.
- **Learning rate**: a loss that diverges or plateaus suggests a rate that is too high or too low.
- **Class-specific sensitivity**: an aggregate loss can hide a class that is never predicted.
- **Gradient norm**: exploding or vanishing gradients suggest an architecture or normalization problem.

The curves are not a substitute for a held-out evaluation, but they catch training failures early. A model that has not learned the minority class will not be fixed by calibration.

**Early stopping.** Early stopping uses the validation set to choose the epoch. The test set must not be used. The stopping criterion should be the metric that matters, not the training loss.

### Step 7. Calibrate the model

**Input:** the trained model and a validation set.

**Output:** a calibration map and a calibrated probability.

**Rationale:** a deep model's softmax output is not a calibrated probability. It is often overconfident, especially with weight decay and small data. Calibration makes the probability usable for a threshold decision.

**Methods:** temperature scaling fits a single parameter on the validation set; isotonic regression fits a monotone map; histogram binning fits a step function. Temperature scaling preserves the ranking and changes only the confidence.

**Arithmetic:** expected calibration error with 10 bins is the weighted average of |accuracy - confidence| across bins. A model with ECE 0.15 is, on average, 15 percentage points away from its stated confidence. That is large for a clinical decision.

**Branch point:** if the model is used only for ranking, calibration is less important. If it is used for a threshold or a probability, calibration is essential.

**Edge case:** calibrating on the test set is leakage. Calibrate on a held-out validation set, and evaluate on a separate test set.

### Step 8. Evaluate discrimination, calibration, and subgroups

**Input:** the test set and the calibrated predictions.

**Output:** AUROC, AUPRC, sensitivity, specificity, PPV, ECE, Brier score, and subgroup metrics.

**Rationale:** each metric answers a different question. A model can have a good AUROC and a bad ECE, or a good overall performance and a bad subgroup performance.

**Arithmetic:** AUROC is the probability that a randomly chosen positive is ranked above a randomly chosen negative. With 30 positives and 170 negatives, the denominator is 30 × 170 = 5,100 pairs. A single misranking changes the AUROC by 1/5,100 = 0.0002.

**Branch point:** if the prevalence is low, AUPRC is more informative than AUROC. If the decision has a fixed threshold, sensitivity and specificity at that threshold are more informative than either.

**Edge case:** a subgroup with 20 patients has a wide confidence interval. Report the interval, not only the point estimate.

**Metric definitions and when to use them.**

| Metric | Definition | Use when | Avoid when |
|---|---|---|---|
| AUROC | Ranking across all thresholds | Balanced classes | Extreme imbalance |
| AUPRC | Precision-recall area | Low prevalence | Balanced classes |
| Sensitivity | TP/(TP+FN) | Screening | Confirmation |
| Specificity | TN/(TN+FP) | Confirmation | Screening |
| PPV | TP/(TP+FP) | Decision to treat | Prevalence varies |
| NPV | TN/(TN+FN) | Rule-out | Prevalence varies |
| F1 | Harmonic mean of precision and recall | Single summary | Cost asymmetry |
| ECE | Calibration error | Probability decisions | Ranking only |
| Brier | Squared probability error | Overall probability quality | Ranking only |

The table shows why a single metric is not enough. A screening decision needs sensitivity at an acceptable false-positive rate; a confirmation decision needs PPV; a probability decision needs calibration. The metric should match the decision.

### Step 9. Aggregate segments to the patient

**Input:** segment-level probabilities and patient identifiers.

**Output:** a patient-level probability and a decision.

**Rationale:** the aggregation rule determines the patient-level performance. Mean pooling dilutes a focal event; max pooling is sensitive to a single false positive; a vote ignores confidence.

**Arithmetic:** with segment probabilities 0.9, 0.8, 0.7, 0.2, 0.1, the mean is 0.54, the max is 0.90, and the vote is 0.60. A threshold of 0.5 gives positive for all three rules but different confidence. A threshold of 0.7 gives positive for max only.

**Branch point:** if the event is focal, use a top-k mean or a max. If the event is diffuse, use a mean. The rule must be pre-specified and validated.

**Edge case:** a patient with many segments has a different aggregation variance from a patient with few. The rule should account for the number of segments or the model should be trained on the same distribution.

**Aggregation rules compared.**

| Rule | Formula | Focal event | Diffuse event | Outlier sensitivity |
|---|---|---|---|---|
| Mean | mean(p) | Dilutes | Good | Low |
| Max | max(p) | Good | Overreacts | High |
| Top-k mean | mean of top k | Good | Good | Medium |
| Vote | share above threshold | Depends on threshold | Good | Medium |
| Noisy-or | 1 - prod(1-p) | Good | Overreacts | High |

The choice should be pre-specified and validated at the patient level. A model that is trained on segments but evaluated on patients with a max rule has an operating point that depends on the number of segments, not only on the model.

**The number of segments as a confounder.** A patient with more segments has a higher chance of a high-scoring segment. A max rule therefore favours patients with more data. The rule should be adjusted for the number of segments, or the model should be evaluated with a fixed number of segments per patient.

### Step 10. Analyse errors and subgroups

**Input:** the test predictions and the metadata.

**Output:** an error taxonomy, a subgroup table, and a list of failure modes.

**Rationale:** the aggregate metrics hide the errors that matter. A model can be accurate overall and wrong on the patients with the highest risk.

**Method:** stratify by age, sex, device, site, signal quality, and comorbidity. Inspect the false positives and false negatives for a common cause. Check whether the errors are concentrated in a subgroup or in a quality band.

**Branch point:** if a subgroup has a large error, the model may need a different threshold, a different representation, or more data. The choice depends on the cause.

**Edge case:** a subgroup with no errors may be too small to detect one. Report the confidence interval.

### Step 11. Validate externally

**Input:** the model and an external dataset.

**Output:** an external performance estimate.

**Rationale:** internal validation estimates performance on the same population. External validation estimates transportability. The gap between the two is the generalization gap.

**Method:** apply the model without retraining to the external dataset, with the same preprocessing, and report the metrics. If the external dataset uses a different device or sampling rate, harmonize or report the mismatch.

**Branch point:** if the external performance is much lower, the cause may be a shortcut, a distribution shift, or a label definition difference. Diagnose before retraining.

**Edge case:** an external dataset with a different prevalence changes PPV even when sensitivity and specificity are unchanged. Report prevalence.

### Step 12. Document and monitor

**Input:** the model, the data, the metrics, and the limits.

**Output:** a model card and a monitoring plan.

**Rationale:** a model is a product component, not a one-time analysis. The model card records what it was trained on, how it was evaluated, and what it does not do. The monitoring plan records what will be watched after deployment.

**Contents:** intended use, training data, split, architecture, metrics with confidence intervals, subgroup performance, calibration, limits, and monitoring signals.

**Branch point:** if the model is regulated, the model card is part of the submission, and a change to the model is a change to the device. The FDA predetermined change control plan is the mechanism for planned updates.

**Edge case:** a model that is retrained on new data is a new model, even if the architecture is unchanged. Version it and re-evaluate.

**A monitoring minimum.** At minimum, monitor the input distribution, the score distribution, the positive rate, and the calibration on a labelled sample. A shift in any of the four is a signal to investigate, not a proof of failure.

The monitoring should be pre-specified in the model card, with thresholds and an owner. A monitor without an owner is a dashboard, not a control.

**A model card template.** The card should answer the questions a reviewer will ask.

| Section | Content |
|---|---|
| Intended use | Population, setting, decision |
| Input | Signal, sampling rate, leads, window |
| Output | Class, probability, aggregation |
| Training data | Sources, size, prevalence, dates |
| Split | Patient, site, device, temporal |
| Architecture | Layers, parameters, receptive field |
| Metrics | AUROC, AUPRC, sensitivity, specificity, ECE |
| Subgroups | Age, sex, device, site, quality |
| Calibration | Method, validation set, ECE |
| Limits | What the model does not do |
| Monitoring | Drift, calibration, alerts |

A model card without the split and the limits is a marketing document. A model card with them is an engineering document.

### Decision table for model development

| Condition | Choice | Rationale |
|---|---|---|
| Small dataset (<1,000 patients) | Features + simple model | Lower variance |
| Large dataset (>10,000) | Raw signal + CNN | Learns morphology |
| Long context needed | RNN/transformer or dilated CNN | Receptive field |
| Low prevalence | AUPRC, sensitivity at fixed FPR | AUROC misleading |
| Probability decision | Calibration required | Threshold validity |
| Focal event | Max or top-k aggregation | Mean dilutes |
| Diffuse event | Mean aggregation | Max overreacts |
| New device claim | Device-held-out test | Transportability |
| Regulated update | PCCP | Pre-authorized change |

## 6. Worked example

### 6.1 The case

A 1D CNN is designed to classify 30 s single-lead ECG segments as normal or abnormal. The input is 7,500 samples at 250 samples/s. The architecture has three convolutional blocks with channels 1→8→16→32, kernel 5, stride 2, and padding 2, followed by global average pooling and a two-class linear head. The training set has 800 patients and the test set has 200 patients, split by patient. The model is evaluated with AUROC, ECE, and patient-level aggregation.

### 6.2 Shape and parameter arithmetic

For each layer, the output length is (n + 2p - k)/s + 1.

| Layer | Input length | Channels | Kernel | Stride | Padding | Output length |
|---|---:|---|---:|---:|---:|---:|
| Conv 1 | 7,500 | 1→8 | 5 | 2 | 2 | 3,750 |
| Conv 2 | 3,750 | 8→16 | 5 | 2 | 2 | 1,875 |
| Conv 3 | 1,875 | 16→32 | 5 | 2 | 2 | 938 |

Parameters:

- Conv 1: 1 × 8 × 5 + 8 = 48.
- Conv 2: 8 × 16 × 5 + 16 = 656.
- Conv 3: 16 × 32 × 5 + 32 = 2,592.
- Head: 32 × 2 + 2 = 66.
- Total: 3,362.

At int8 precision, the weights occupy 3,362 bytes, or 3.28 kB, before overhead. The model is small enough for a microcontroller and large enough to learn local morphology.

The parameter count scales with the channel widths and the kernel, not with the input length. A wider model with channels 1→32→64→128 has 192 + 10,304 + 41,088 + 258 = 51,842 parameters, which is still small. The input length affects the compute, not the parameter count.

| Configuration | Parameters | int8 size | MACs |
|---|---:|---:|---:|
| 8-16-32, k=5 | 3,362 | 3.28 kB | 3.75 M |
| 32-64-128, k=5 | 51,842 | 50.6 kB | 58.2 M |
| 16-32-64, k=7 | 18,274 | 17.8 kB | 20.6 M |

The table shows the trade-off: a wider model is still small in absolute terms, but the compute rises by an order of magnitude. The right size depends on the task and the deployment target.

### 6.3 Receptive field arithmetic

The receptive field is 1 + (5-1) + (5-1)×2 + (5-1)×4 = 1 + 4 + 8 + 16 = 29 samples.

At 250 samples/s, 29 samples is 116 ms. The model can see a QRS complex (80-100 ms) and the surrounding ST segment, but it cannot see a 30 s rhythm. A task that depends on rhythm, such as AF detection from RR irregularity, needs a larger receptive field or a sequence model.

### 6.4 Compute arithmetic

The multiply-accumulate count per inference is the sum over layers of output length × input channels × output channels × kernel.

- Conv 1: 3,750 × 1 × 8 × 5 = 150,000.
- Conv 2: 1,875 × 8 × 16 × 5 = 1,200,000.
- Conv 3: 938 × 16 × 32 × 5 = 2,401,280.
- Total: 3,751,280 MACs.

At 1 GOPS effective throughput, the inference takes about 3.75 ms of compute. At 100 MOPS, it takes 37.5 ms. The latency budget determines whether the model can run on the device or must run in the cloud.

The per-sample compute is a useful sanity check. A 7,500-sample input with 3.75 M MACs is about 500 MACs per input sample. For comparison, a 12-lead ECG model with 12 input channels and the same architecture would have about 12 times the first-layer compute, or roughly 4.5 M MACs. The channel count multiplies the first layer, not the later layers, because the later layers see the fused channels.

### 6.5 Discrimination and calibration arithmetic

On the test set of 200 patients with 30 positives and 170 negatives, the model's predictions give an AUROC of 0.893 and an ECE of 0.152.

The AUROC of 0.893 means that a randomly chosen positive has a higher score than a randomly chosen negative in 89.3% of pairs. With 5,100 pairs, the standard error of the AUROC is roughly 0.02 to 0.03, depending on the correlation between pairs. The confidence interval is therefore about 0.84 to 0.94.

The ECE of 0.152 is large. On average, the predicted probability is 15 percentage points away from the observed frequency in its bin. A model that says 0.8 for a bin in which 65% are positive is overconfident by 15 points.

The reliability table makes the pattern visible.

| Bin | n | Mean predicted | Observed positive | Gap |
|---|---:|---:|---:|---:|
| 0.0-0.1 | 14 | 0.052 | 0.000 | 0.052 |
| 0.1-0.2 | 21 | 0.149 | 0.000 | 0.149 |
| 0.2-0.3 | 26 | 0.242 | 0.038 | 0.204 |
| 0.3-0.4 | 30 | 0.350 | 0.133 | 0.217 |
| 0.4-0.5 | 29 | 0.455 | 0.172 | 0.283 |
| 0.5-0.6 | 24 | 0.546 | 0.417 | 0.129 |
| 0.6-0.7 | 31 | 0.641 | 0.613 | 0.028 |
| 0.7-0.8 | 12 | 0.754 | 0.917 | 0.163 |
| 0.8-0.9 | 9 | 0.842 | 0.889 | 0.047 |
| 0.9-1.0 | 4 | 0.962 | 1.000 | 0.038 |

The gaps are not monotone, and the high-probability bins are small. The model is badly calibrated in the middle bins (0.3-0.5), where most predictions lie, and reasonably calibrated at the extremes. A calibration map fitted on a validation set would reduce the middle-bin gaps.

| Calibration variant | ECE |
|---|---:|
| Original | 0.152 |
| Overconfident (probabilities × 1.15) | 0.193 |
| Underconfident (probabilities shrunk toward 0.5) | 0.115 |

The table shows that the ECE moves in the expected direction when the probabilities are distorted. The underconfident variant has a lower ECE in this example, which is a reminder that ECE is not a monotone measure of calibration quality; a model can be underconfident and have a lower ECE than a better-calibrated model.

### 6.6 Aggregation arithmetic

A patient has five segments with probabilities 0.9, 0.8, 0.7, 0.2, and 0.1.

| Rule | Value | Decision at 0.5 | Decision at 0.7 |
|---|---:|---|---|
| Mean | 0.54 | positive | negative |
| Max | 0.90 | positive | positive |
| Vote (>0.5) | 0.60 | positive | negative |

The three rules agree at a 0.5 threshold and disagree at 0.7. The max rule is sensitive to one high segment; the mean rule dilutes a focal event; the vote loses confidence. The rule must be chosen for the task and reported with the result.

### 6.7 Diagnostic 1: segment split versus patient split

Suppose the same model is trained and tested with a segment split. The training set has 16,000 segments from 800 patients, and the test set has 4,000 segments from the same 800 patients. The AUROC rises from 0.893 to about 0.96, because the model has seen each patient's morphology in training.

The diagnostic is to compare the patient-level and segment-level splits. If the segment split is much higher, the difference is leakage. A second diagnostic is to check whether the model's errors are concentrated on patients whose segments appear in both sets.

The consequence of missing the diagnostic is a claimed performance that does not generalize. A model with 0.96 on a segment split and 0.89 on a patient split is not a 0.96 model.

The split comparison is summarised below.

| Split | Effective N | AUROC | What it estimates |
|---|---:|---:|---|
| Segment random | 20,000 | 0.96 | New segments from seen patients |
| Patient | 200 | 0.89 | New patients from the same population |
| Site-held-out | 60 | 0.81 | New site |
| Prospective | 200 | 0.78 | Intended setting |

The table is illustrative of the pattern, not of a specific study. The general pattern is that performance falls as the split becomes more distant from the training distribution. The claim should name the split, and the product should be validated at the level of the claim.

### 6.8 Diagnostic 2: calibration drift

Suppose the model is deployed on a new device with a different front-end filter. The AUROC remains 0.88, but the ECE rises to 0.25 because the score distribution shifts. The ranking is preserved; the probabilities are not.

The diagnostic is to monitor the score distribution and the calibration on a labelled sample after deployment. A shift in the mean score without a shift in prevalence is a calibration problem; a shift in both is a prevalence or case-mix problem.

### 6.9 Varying one assumption

Suppose the receptive field is increased to 121 samples (484 ms) by adding two more convolutional layers. The model can see more context, and the rhythm classification improves, but the parameter count and the compute rise. The receptive field is a design choice with a cost.

Suppose the threshold is chosen on the test set to maximize the F1 score. The reported F1 is optimistic, because the threshold was tuned on the evaluation data. The threshold must be chosen on the validation set and applied to the test set.

Suppose the aggregation is changed from mean to max. The patient-level sensitivity rises and the specificity falls, because a single high-scoring segment is enough. The choice changes the operating point without changing the model.

### 6.10 Conclusion of the case

The model has 3,362 parameters, a 116 ms receptive field, and 3.75 million MACs per inference. It achieves an AUROC of 0.893 and an ECE of 0.152 on a patient-level test set. The AUROC is moderate and the calibration is poor; the model is not ready for a probability-based decision without recalibration.

The two diagnostics (segment versus patient split, calibration drift) catch the two errors that would otherwise inflate the reported performance. The assumption that matters most is the split, because it defines the claim.

### 6.10 What the case does not show

The case is a small synthetic model on a single task. It does not show:

- **Multi-lead input.** A 12-lead model has 12 input channels and a larger first layer.
- **Long context.** The 116 ms receptive field cannot see rhythm; the case does not test a rhythm task.
- **Training dynamics.** The weights are hand-set; the case does not train or tune.
- **Class imbalance.** The synthetic labels are 30% positive, not 1%.
- **External data.** The test set is from the same distribution as the training set.
- **Subgroup performance.** There is no age, sex, device, or site metadata.

Each omission is a reason to validate on real data with the intended task. The arithmetic transfers; the performance does not.

The most important omission is the last one. A model that is accurate on average and inaccurate for a subgroup is not a safe model, and the case has no subgroup metadata to test. A subgroup analysis requires the metadata to be collected with the signal, not reconstructed afterward from the device or the site.

A second omission is the threshold. The case reports AUROC and ECE but does not choose a threshold, because the threshold depends on the decision and the cost of errors. A model card should report the operating point that the product will use, not only the ranking.

A third omission is the confidence interval. The AUROC of 0.893 has a standard error of about 0.02, so the plausible range is 0.85 to 0.93. The interval is wide because the test set has only 200 patients, and a product decision should account for it.

A fourth omission is the operating threshold. The case stops at the probability and does not convert it to a decision, which is the next topic's subject. A product decision on a point estimate without the interval is overconfident. A model card should report the operating point that the product will use, not only the ranking. A real validation must include the metadata before the model is trained, because a subgroup analysis cannot be added after the fact if the data do not carry the group labels.

## 7. Code, query, or calculation

The script computes the shape, parameter count, receptive field, and MACs of the 1D CNN, runs a forward pass with hand-set filters, and computes AUROC and ECE for synthetic predictions. It uses only the standard library.

```python
import math, random

def conv_out(n,k,s,p): return (n + 2*p - k)//s + 1

n=7500
layers=[(1,8,5,2,2),(8,16,5,2,2),(16,32,5,2,2)]
shapes=[]; cur=n; params=0
for cin,cout,k,s,p in layers:
    cur=conv_out(cur,k,s,p)
    shapes.append((cin,cout,k,s,p,cur))
    params += cin*cout*k + cout
print("shapes=%s" % [x[-1] for x in shapes])
print("conv_params=%d fc_params=%d total=%d" % (params, 32*2+2, params+32*2+2))

rf=1; stride=1
for cin,cout,k,s,p in layers:
    rf += (k-1)*stride; stride *= s
print("receptive_field_samples=%d ms=%.1f" % (rf, rf/250*1000))

macs=0; inp=n
for cin,cout,k,s,p in layers:
    out=conv_out(inp,k,s,p)
    macs += out*cin*cout*k
    inp=out
print("macs=%.0f" % macs)
print("int8_kb=%.2f" % ((params+32*2+2)/1024))

def conv1d(x, w, b, stride=1, pad=0):
    xp=[0.0]*pad + list(x) + [0.0]*pad
    cout=len(w); k=len(w[0][0])
    outlen=(len(xp)-k)//stride+1
    out=[[0.0]*outlen for _ in range(cout)]
    for o in range(cout):
        for i in range(outlen):
            s=b[o]; base=i*stride
            for ci in range(len(w[o])):
                for kk in range(k):
                    s += w[o][ci][kk]*xp[base+kk]
            out[o][i]=max(0.0,s)
    return out

random.seed(1)
x=[math.sin(2*math.pi*1.2*i/250)+0.2*random.gauss(0,1) for i in range(200)]
w1=[[[0.1,0.3,0.4,0.2,0.0]] for _ in range(8)]
b1=[0.0]*8
y=conv1d(x,w1,b1,stride=2,pad=2)
print("conv1_out=%d channels=%d first=%.3f" % (len(y[0]), len(y), y[0][0]))

def auroc(y,s):
    pairs=sorted(zip(s,y))
    n1=sum(y); n0=len(y)-n1
    rank_sum=0.0; i=0
    while i < len(pairs):
        j=i
        while j+1 < len(pairs) and pairs[j+1][0]==pairs[i][0]: j+=1
        avg=(i+1+j+1)/2
        for k in range(i,j+1):
            if pairs[k][1]==1: rank_sum += avg
        i=j+1
    return (rank_sum - n1*(n1+1)/2)/(n1*n0)

def ece(y,p,bins=10):
    tot=len(y); e=0.0
    for b in range(bins):
        lo=b/bins; hi=(b+1)/bins
        idx=[i for i in range(tot) if (p[i]>lo or (b==0 and p[i]>=lo)) and p[i]<=hi]
        if not idx: continue
        conf=sum(p[i] for i in idx)/len(idx)
        acc=sum(y[i] for i in idx)/len(idx)
        e += len(idx)/tot*abs(acc-conf)
    return e

random.seed(2)
y=[1 if random.random()<0.3 else 0 for _ in range(200)]
p=[min(0.99,max(0.01, (0.65 if yy else 0.35)+random.gauss(0,0.18))) for yy in y]
print("auroc=%.3f ece=%.3f" % (auroc(y,p), ece(y,p)))

p_ov=[min(0.99, v*1.15) for v in p]
p_un=[0.5+(v-0.5)*1.4 for v in p]
print("overconf_ece=%.3f underconf_ece=%.3f" % (ece(y,p_ov), ece(y,p_un)))

segs=[0.9,0.8,0.7,0.2,0.1]
print("mean_prob=%.2f max_prob=%.2f vote=%.2f"
      % (sum(segs)/5, max(segs), sum(1 for v in segs if v>0.5)/5))
```

Expected output:

```text
shapes=[3750, 1875, 938]
conv_params=3296 fc_params=66 total=3362
receptive_field_samples=29 ms=116.0
macs=3751280
int8_kb=3.28
conv1_out=100 channels=8 first=0.167
auroc=0.893 ece=0.152
overconf_ece=0.193 underconf_ece=0.115
mean_prob=0.54 max_prob=0.90 vote=0.60
```

Three implementation notes.

The forward pass uses hand-set filters, not learned weights. It demonstrates the shape and the ReLU behaviour, not the model's accuracy. A real forward pass would load trained weights and run all three layers.

The AUROC is computed with the rank-sum formula, which is equivalent to the Mann-Whitney U statistic. Ties are handled by averaging ranks. The implementation is O(N log N) and is adequate for the test set size.

The ECE uses 10 equal-width bins from 0 to 1. Bins with no samples are skipped. The binning choice changes the ECE; a report should state the number of bins.

## 8. Data, standards, and regulatory context

Deep-learning models for physiological signals are regulated as device software functions when they make a medical claim. The standards and guidance cover the software process, the data, and the change management.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| FDA device software guidance (2023) | Premarket software documentation | Binding through review | Description, hazard analysis, V&V |
| FDA AI-enabled device software guidance (2025) | Lifecycle management | Binding through review | PCCP and monitoring |
| FDA PCCP guidance (final Dec 2024, updated Aug 2025) | Planned model changes | Binding through review | Pre-authorized updates |
| IEC 62304 | Medical device software lifecycle | Consensus; often required | Development and maintenance process |
| ISO 14971 | Risk management | Consensus; often required | Hazard analysis |
| ISO/IEC 42001 | AI management systems | Voluntary | Governance |
| EU AI Act (2024) | AI systems, including medical | Binding in the EU | Risk classification and transparency |
| EU MDR | Clinical evaluation | Binding in the EU | Clinical evidence |
| HIPAA | Data protection | Binding in the US | De-identification |

### 8.1 The PCCP in practice

The FDA's predetermined change control plan guidance, finalized in December 2024 and updated in August 2025, allows a manufacturer to pre-authorize specific, planned changes to an AI-enabled device without a new marketing submission. The plan has three components:

1. **Description of modifications**: what will change, whether the change is automatic or manual, and how often.
2. **Modification protocol**: data management, retraining practices, performance evaluation with pre-defined acceptance criteria, and update procedures.
3. **Impact assessment**: benefits and risks of the planned changes, including bias, and how the protocol mitigates them.

The PCCP is relevant to this topic because a deep model is expected to change. A model that is retrained on new data, or that adds a new input source, is a candidate for a PCCP. A change outside the plan requires a new submission.

The practical consequence is that the model development process must be designed for change from the start: versioned data, versioned code, reproducible training, and a monitoring plan.

### 8.1.1 The evidence package for a deep-learning device

A submission for a deep-learning device typically contains the following evidence, whether or not a specific guidance names each item.

| Evidence | Question | Artefact |
|---|---|---|
| Intended use | Who, what, where | Indications for use |
| Data provenance | Where did the data come from | Data card |
| Split | What does the performance estimate | Model card |
| Reference standard | What is the truth | Adjudication protocol |
| Discrimination | Can it rank | AUROC/AUPRC with CIs |
| Calibration | Do probabilities mean anything | Reliability plot, ECE |
| Subgroups | Does it work for everyone | Stratified table |
| External test | Does it transport | External validation |
| Human factors | Can users use it | Formative and summative |
| Cybersecurity | Is it protected | SBOM, threat model |
| Change control | How will it evolve | PCCP |
| Monitoring | How will drift be found | Post-market plan |

The package is larger than the model. A team that starts with the architecture and works backward will underestimate the evidence; a team that starts with the evidence and works backward will design the model and the study together.

### 8.1.2 What a reviewer will ask first

A reviewer typically asks four questions before the architecture.

1. What is the intended use, and does the data match it?
2. What is the reference standard, and how was it adjudicated?
3. What is the split, and what does the performance estimate?
4. What happens to the patients the model gets wrong?

The fourth question is the one that teams prepare least. A confusion matrix is not an error analysis; the error analysis names the patients, the conditions, and the consequences. A model that is wrong on 5% of patients is acceptable if the 5% are low-risk and unacceptable if they are the high-risk group the product exists to find.

### 8.2 What the standards do not cover

Three gaps.

First, the standards do not specify the split. A submission can report a patient-level split or a segment-level split, and the reviewer must infer the claim. The split should be stated explicitly in the model card and the submission.

Second, the standards do not specify the calibration requirement. A model can be cleared on discrimination alone, even if the probabilities are not calibrated. If the product uses a threshold, calibration should be part of the evidence.

Third, the standards do not specify the aggregation rule for patient-level decisions. A model evaluated on segments and deployed on patients can have a different operating point. The aggregation should be pre-specified and validated.

### 8.2.1 Three regulatory scenarios

The evidence package depends on the claim.

| Scenario | Claim | Evidence emphasis |
|---|---|---|
| Wellness | General wellness trend | Bench, human factors, no diagnostic claim |
| Screening | Flag users for confirmatory testing | Sensitivity, PPV, inconclusive rate, care pathway |
| Diagnosis | Diagnose a condition | Clinical validation, external test, subgroup, calibration |

A screening claim can be cleared with a narrower package than a diagnosis claim, but the package still needs a reference standard and a confirmation pathway. A wellness claim avoids the diagnostic evidence but limits the language and the population.

The scenario should be chosen before the model is designed, because it determines the data collection, the reference standard, and the metrics.

### 8.3 The EU AI Act

The EU AI Act, adopted in 2024, classifies AI systems by risk. Medical devices are generally high-risk, and the Act imposes obligations for risk management, data governance, transparency, and human oversight. The Act interacts with the MDR rather than replacing it; a medical AI system must satisfy both.

The practical consequence is that documentation for a European launch must cover both the device regulation and the AI Act's transparency and data governance requirements. The model card and the data card are useful artefacts for both.

A second consequence is that human oversight must be designed, not assumed. The Act expects a human to be able to interpret the output and intervene. A product that presents a probability without the evidence or the limits makes oversight difficult, even if a human is nominally in the loop.

A third consequence is that the data governance requirements extend to the training data. The provenance, the consent basis, and the representativeness of the data are part of the compliance story, not only the model metrics.

## 9. Industry practice and case evidence

### 9.1 The PhysioNet/Computing in Cardiology Challenge 2021

The 2021 challenge is the largest public ECG classification benchmark. It distributed 88,253 training ECGs from eight datasets and evaluated on hidden validation and test sets, including 12-, 6-, 4-, 3-, and 2-lead versions. The splits were patient-level: recordings from the same individual were assigned to only one of training, validation, or test.

The results paper reports 131,149 recordings from 10 sources, 68 teams, and 1,056 algorithms. About 90% of the algorithms used deep learning. The key finding is that performance differences across lead combinations were small (a mean change of about 2.6%), while differences across test databases were large. On a completely hidden "Undisclosed" dataset, the median relative change was about -39%.

The lesson is that generalization across databases, not lead count, was the main challenge. A model that performs well on the public training distribution can lose a large fraction of its performance on a new source. The inference is that external validation is not optional for a claim that crosses populations.

### 9.1.1 The hidden-dataset result

The 2021 challenge's most cited result is the gap between the public leaderboard and the hidden "Undisclosed" dataset. The median relative change was about -39% on the completely hidden source. The result is a direct measurement of the generalization gap in a controlled benchmark.

The gap has several causes:

- **Population difference.** The hidden source has a different patient mix.
- **Device difference.** The recording equipment and the sampling chain differ.
- **Label difference.** The annotation conventions differ.
- **Prevalence difference.** The class balance differs.

The challenge did not isolate the causes, and a product team cannot either without an external dataset. The inference is that a single-source benchmark, however large, is not an external validation.

A second result is that the lead-count effect was small (about 2.6% mean change) compared with the source effect. A model that works on 12 leads usually works on 2 leads with a modest loss, but a model that works on one source may not work on another. The source, not the lead count, is the dominant domain variable.

### 9.2 The winning approaches

The top teams used ensembles of residual CNNs with attention. The specific architectures matter less than the pattern: a residual 1D CNN, a multi-lead input, a weighted loss for the challenge metric, and an ensemble. The pattern is reproducible; the exact hyperparameters are less important.

The challenge also showed that the scoring metric shapes the solution. The challenge used a weighted confusion-matrix metric that penalized certain misclassifications more than others. Teams optimized the metric, and the ranking reflected the metric. A product team should choose the metric that reflects the clinical cost, not the metric that is convenient.

### 9.2.1 What the benchmark does not measure

The challenge metric is a weighted confusion matrix over diagnostic statements. It does not measure calibration, and it does not measure the clinical utility of a threshold. A model that ranks well but is overconfident can win the challenge and fail a product review.

The challenge also does not measure the cost of the compute. A large ensemble can win the leaderboard and be impossible to deploy on a device. The inference is that a benchmark is a research instrument, not a product specification.

A product team should use the benchmark for architecture ideas and for sanity checks, not as a substitute for a prospective validation in the intended setting.

### 9.2.2 The lesson for a product team

A product team reading the challenge results should take three things.

- **Benchmark performance is an upper bound.** The hidden-dataset gap is the best public estimate of what a new source can do to a model.
- **Patient-level splitting is the minimum.** The challenge organizers enforced it; a product team should enforce it without being told.
- **The metric is a choice.** The challenge metric weights errors by clinical importance; a product metric should do the same, and the weights should be justified.

The inference is that a benchmark is a starting point for a validation, not a substitute. A model that wins a benchmark has demonstrated that it can learn the benchmark; it has not demonstrated that it can work in the intended setting.

### 9.2.3 A benchmark-to-product gap table

| Benchmark property | Product requirement | Gap |
|---|---|---|
| Public training data | Intended-use data | Population and device |
| Hidden test set | Prospective validation | Setting and workflow |
| Challenge metric | Clinical utility | Cost and threshold |
| Segment or record score | Patient decision | Aggregation |
| No calibration | Probability decision | Calibration |
| No deployment cost | Latency and power | Engineering |

The table is a reminder that a benchmark optimizes for a different objective than a product. A team that treats the leaderboard as a product roadmap will build a model that wins the benchmark and misses the deployment.

### 9.3 Wearable and consumer models

Consumer devices use deep models for AF detection, sleep staging, and activity recognition. The models are often small enough for the device and trained on large proprietary datasets. The published evidence is thinner than the deployment scale.

The documented pattern is that the validation is often a controlled study with a reference device, and the free-living performance is reported separately or not at all. The inference is that a consumer claim should be read with the study conditions in mind.

### 9.4 Foundation models for signals

Signal foundation models pretrain on large unlabeled corpora and fine-tune on a task. The promise is transfer across tasks and devices; the evidence is still emerging. The documented risk is that the pretraining corpus encodes the devices and populations in the corpus, and the fine-tuning inherits them.

The inference for product teams is that a foundation model is not a substitute for a validation on the target device and population. It may reduce the data needed for fine-tuning, but it does not remove the need for an external test.

### 9.4.1 Foundation models and the validation question

A foundation model changes the data requirement, not the validation requirement. The model may need fewer labelled examples for fine-tuning, but the fine-tuned model still needs a patient-level and external evaluation on the target device.

The risk is that the pretraining corpus becomes the new confounder. If the corpus over-represents one device or one population, the fine-tuned model inherits the imbalance, and the imbalance is harder to see because the pretraining data are often not public.

The inference is that a foundation model should be evaluated with the same split discipline as a trained-from-scratch model, and the model card should describe the pretraining data at the level of detail that is available.

### 9.5 Regulatory examples

The FDA's PCCP webpage lists authorized devices with predetermined change control plans, including consumer ECG and hearing-aid features. The list is a useful reference for what a planned change looks like in practice.

The documented lesson is that a PCCP is specific. It names the modification, the protocol, and the acceptance criteria. A vague plan ("the model may be updated") is not a PCCP. The specificity is what allows the manufacturer to update without a new submission.

A second lesson is that the PCCP is a commitment. The manufacturer must follow the protocol; a deviation can make the device adulterated or misbranded. The plan is not a permission slip; it is a validated procedure.

A third lesson is that the PCCP must cover the monitoring. The plan should state how the post-market performance will be measured and what would trigger a corrective action. A model that is updated without monitoring is a model that cannot detect its own drift.

### 9.6 A composite deployment case

A company deploys a deep model for AF detection from a wrist PPG. The model is a 1D CNN trained on 2 million segments from 50,000 users of one device. The internal validation uses a patient-level split and reports an AUROC of 0.91 and an ECE of 0.18. The product launches with a threshold chosen to give 95% specificity.

Six months later, the company releases a new device with a different optical front end and a 25 Hz sampling rate instead of 50 Hz. The model is unchanged. The AF notification rate falls by 30%, and the confirmed AF rate among notified users rises.

The explanation is a distribution shift: the new device's PPG has less high-frequency content, and the model's first-layer filters are tuned to the old content. The model is not wrong; it is seeing a different signal. The fix is to resample and retrain with device augmentation, or to train a device-specific model.

The case is a composite, not a named product. The pattern is documented in cross-device studies: a model trained on one device does not automatically transfer to another, and the transfer failure is often in the preprocessing rather than the classifier.

The lesson is that the device is part of the model's input specification. A model card that says "PPG" without the sampling rate and the optical geometry is under-specified.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Segment leakage | AUROC much higher than patient-level | Same patient in train and test | Inflated performance | Patient-level split |
| Preprocessing leakage | Normalization uses test data | Statistics computed globally | Optimistic metrics | Fit on train |
| Site shortcut | Site predicts label | Confounded collection | Poor external validity | Site-held-out test |
| Device shortcut | Device predicts label | Different sensors | Deployment failure | Device-held-out test |
| Label leakage | Too-good performance | Label-derived feature | Invalid model | Feature audit |
| Class imbalance ignored | High accuracy, low sensitivity | Majority class | Useless screening | AUPRC; class weights |
| Uncalibrated probabilities | ECE high | Softmax overconfidence | Wrong threshold decisions | Temperature scaling |
| Threshold on test | Optimistic F1 | Test-set tuning | Invalid claim | Validation threshold |
| Aggregation unstated | Patient score varies | Mean versus max | Non-reproducible | Pre-specify |
| Receptive field too small | Rhythm errors | Model cannot see context | Poor performance | Larger RF |
| Receptive field too large | Overfitting to record | Model sees irrelevant context | Poor generalization | Regularization |
| Augmentation changes label | Confused model | Time reversal, warping | Wrong learning | Label-preserving only |
| Overfitting | Train-test gap | Small data, large model | Poor generalization | Regularization; simpler model |
| Distribution shift | Performance drop | New device or population | Deployment failure | Monitoring; revalidation |
| Silent update | Behaviour changes | Retraining or firmware | Unverified performance | Versioning; PCCP |
| Multi-lead mismatch | Input shape error | 12 leads to 1 | Deployment failure | Channel dropout |
| Sampling mismatch | Resampling artifact | 500 to 250 Hz | Feature shift | Harmonize |
| Label noise ceiling | Plateau below expected | Noisy labels | Underperformance | Adjudication |
| Ensemble too large | Latency budget exceeded | Many models | Cannot deploy | Distill; prune |
| Overconfident OOD | High score on artifact | No OOD detection | False confidence | OOD monitor |
| Temporal leakage | Future data in training | Random time split | Inflated performance | Temporal split |

Three red flags deserve emphasis.

**A performance number without a split description.** The number is not a claim until the split is named. A reader cannot tell whether 0.95 means new segments or new patients.

**A probability without a reliability plot.** A softmax output is not a probability until it has been calibrated on held-out data. The plot is the evidence.

**A model card without a limits section.** Every model has a population, a device, and a condition it was not validated on. The limits are the most useful part of the card for a deployment team.

A useful audit question is: what would make this model unsafe? The answer is usually a population, a device, or a workflow, not an architecture. The answer should be in the card and in the monitoring plan.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The split matches the claim | Valid performance | Segment split for patient claim | Compare splits |
| The label is consistent | Learning the right target | Different annotation rules | Audit labels |
| The preprocessing is train-only | No leakage | Global normalization | Refit on train |
| The augmentation preserves the label | Correct supervision | Label-changing transforms | Manual review |
| The receptive field sees the phenomenon | Model capacity | Too small or too large | Ablation |
| The class balance is handled | Sensitivity | Imbalance | AUPRC; weights |
| The probabilities are calibrated | Threshold decisions | Overconfidence | ECE; reliability plot |
| The aggregation matches deployment | Patient-level performance | Different rule | Pre-specify; validate |
| The test population matches deployment | Transportability | Different site or device | External test |
| The model is versioned | Reproducibility | Silent update | Model card |
| The monitoring captures drift | Early warning | No post-market data | Monitoring plan |
| The claim matches the evidence | Regulatory | Overclaim | Claim review |
| The input shape is stable | Deployment | Multi-lead model on one lead | Input test |
| The label noise is bounded | Performance ceiling | Noisy labels | Adjudication sample |
| The model is not overconfident OOD | Safety | Artifact scored high | OOD test |
| The temporal split is respected | Valid performance | Random time split | Temporal test |
| The input length is fixed | Deployment | Variable window | Shape test |
| The label definition is stable | Reproducibility | Annotation change | Label audit |
| The pretraining corpus is representative | Transfer | Device or population skew | Corpus audit |
| The model is small enough for the device | Deployment | Latency or memory | Benchmark |
| The monitoring can detect drift | Safety | No post-market data | Monitoring plan |
| The model is interpretable enough for oversight | Governance | Opaque output | Explanation layer |
| The training data consent is documented | Compliance | Unclear provenance | Data audit |
| The evaluation is sized for the metric | Statistical validity | Too few events | Power calculation |

## 12. Exercises

### Exercise 1. Receptive field (application, 20 min)

**Deliverable:** compute the receptive field for a CNN with kernel 7, stride 2, four layers, and state the time span at 250 samples/s.

**Acceptance criteria:** formula shown; samples and ms; the task suitability stated.

**Time:** 20 minutes.

### Exercise 2. Parameter count (application, 25 min)

**Deliverable:** compute the parameters for channels 1→16→32→64, kernel 5, stride 2, plus a 64→2 head.

**Acceptance criteria:** per-layer counts; total; int8 size.

**Time:** 25 minutes.

### Exercise 3. Split design (design, 35 min)

**Deliverable:** design a patient-level and a site-held-out split for a three-site dataset, and state what each estimates.

**Acceptance criteria:** no patient overlap; site held out; the claim for each stated.

**Time:** 35 minutes.

### Exercise 4. Calibration (application, 30 min)

**Deliverable:** compute ECE for 10 predictions and labels, and describe how temperature scaling would change it.

**Acceptance criteria:** binning stated; ECE computed; the ranking-preservation property explained.

**Time:** 30 minutes.

### Exercise 5. Aggregation (application, 25 min)

**Deliverable:** for five segment probabilities, compute mean, max, and vote, and state which rule fits a focal event.

**Acceptance criteria:** three values; threshold decisions; rationale.

**Time:** 25 minutes.

### Exercise 6. Leakage audit (critique, 40 min)

**Deliverable:** list five ways a physiological deep-learning pipeline can leak, and the test for each.

**Acceptance criteria:** each leak has a mechanism and a test; at least one preprocessing leak.

**Time:** 40 minutes.

### Exercise 7. Model card (design, 45 min)

**Deliverable:** a one-page model card with intended use, data, split, metrics, calibration, subgroups, and limits.

**Acceptance criteria:** all fields; confidence intervals; explicit limits.

**Time:** 45 minutes.

### Exercise 8. External validation plan (design, 50 min)

**Deliverable:** a plan for testing a model on a new device, including harmonization, metrics, and a decision rule.

**Acceptance criteria:** device differences named; preprocessing matched; prevalence reported.

**Time:** 50 minutes.

### Exercise 9. Benchmark audit (critique, 30 min)

**Deliverable:** compare a benchmark result with a product claim and list the evidence that is missing for the product.

**Acceptance criteria:** at least five gaps; each gap mapped to an evidence item; no claim that the benchmark is invalid.

**Time:** 30 minutes.

### Submission standard

Model development exercises must report the split, the metrics, and the calibration. A number without a split is not a performance claim. A probability without a calibration check is not a probability. A patient-level decision without an aggregation rule is not reproducible.

The model card exercise is assessed against the template in Section 5, Step 12. A card that omits the limits or the subgroup performance is incomplete, even if the architecture is described well.

## 13. Self-check answers

**1. Compute the receptive field for kernel 7, stride 2, four layers.** 1 + 6 + 12 + 24 + 48 = 91 samples.

**2. What is the time span at 250 samples/s?** 91/250 = 0.364 s = 364 ms.

**3. Compute the parameters for 1→16, k=5.** 1×16×5 + 16 = 96.

**4. Compute the parameters for 16→32, k=5.** 16×32×5 + 32 = 2,592.

**5. Compute the parameters for 32→64, k=5.** 32×64×5 + 64 = 10,304.

**6. Compute the head 64→2.** 64×2 + 2 = 130.

**7. What is the total?** 96 + 2,592 + 10,304 + 130 = 13,122.

**8. What is the int8 size?** 13,122 bytes = 12.8 kB.

**9. What is AUROC?** The probability that a randomly chosen positive is ranked above a randomly chosen negative.

**10. What is ECE?** The weighted average absolute difference between predicted confidence and observed accuracy across bins.

**11. Why is accuracy misleading at 1% prevalence?** A model that predicts negative always has 99% accuracy and zero sensitivity.

**12. Why is a patient-level split necessary?** Segments from the same patient share morphology and identity, so a segment split leaks patient information.

**13. What does a site-held-out split estimate?** Performance on a new collection site, which is a stronger transportability claim than a patient split.

**14. What is the difference between data drift and concept drift?** Data drift is a change in the input distribution; concept drift is a change in the input-outcome relationship.

**15. Why does temperature scaling preserve AUROC?** It is a monotone transformation of the logits, so the ranking is unchanged.

**16. Why is max aggregation sensitive to false positives?** A single high-scoring segment makes the patient positive, regardless of the other segments.

**17. Compute the AUROC denominator for 40 positives and 160 negatives.** 40 × 160 = 6,400 pairs.

**18. What is the ECE if a model predicts 0.8 for 100 cases and 70 are positive?** The gap is |0.70 - 0.80| = 0.10 for that bin; the contribution depends on the bin's share of the data.

**19. Why is a temporal split different from a patient split?** A temporal split holds out future records, which tests transport across time; a patient split tests transport across people. A model can pass one and fail the other.

**20. What is the first diagnostic for a model that performs well internally and poorly externally?** Compare the input distributions, the preprocessing, and the label definition before retraining. A distribution or definition difference is more likely than a model failure.

**21. Why is a model with 3,362 parameters still useful?** The parameter count is not the capability; the receptive field and the task matter. A small model can learn local morphology and deploy on a device.

**22. What is the difference between an ensemble and a distillation?** An ensemble averages several models at inference; a distillation trains a small model to mimic a large one. The first improves accuracy at a cost, the second reduces the cost at a small accuracy loss.

**23. Why is the number of bins in the ECE a choice?** The ECE depends on the bin width; too few bins hide miscalibration, too many bins make each bin noisy. Ten is common, but it must be reported.

**Rubric for Exercise 3 (split design).** A strong design assigns whole patients to folds, holds out a site, and states the claim for each. A weak design splits by segment or ignores site.

**Rubric for Exercise 7 (model card).** A strong card includes intended use, data, split, metrics with intervals, calibration, subgroup performance, and limits. A weak card lists only the architecture and accuracy.

## 14. Interview questions

**1. How do you split physiological data for a deep model?**
A strong answer splits by patient and, for a transportability claim, by site or device, and explains the claim each split supports. A weak answer uses a random split.

**2. Why is AUROC not enough?**
A strong answer adds calibration, AUPRC at low prevalence, and the operating point. A weak answer says AUROC is standard.

**3. What is the receptive field and why does it matter?**
A strong answer computes it and relates it to the phenomenon's duration. A weak answer says it is the kernel size.

**4. How do you handle class imbalance?**
A strong answer uses AUPRC, class weights or sampling, and a threshold chosen for the decision. A weak answer oversamples the minority class without evaluation.

**5. What is calibration and how do you measure it?**
A strong answer defines ECE, reliability plots, and Brier score, and describes temperature scaling. A weak answer says "the probabilities are accurate."

**6. How do you aggregate segment predictions to a patient?**
A strong answer states the rule, the rationale, and the validation. A weak answer averages without justification.

**7. What would you do if external performance dropped by 39%?**
A strong answer diagnoses the source: device, population, label definition, or preprocessing, before retraining. A weak answer retrains on the new data.

**8. What is a PCCP?**
A strong answer describes the three components and the pre-authorized change concept. A weak answer says it is a regulatory document.

**9. How do you detect leakage?**
A strong answer compares splits, audits preprocessing, and checks for label-derived features. A weak answer checks only the split.

**10. Why is a small CNN sometimes better than a large one?**
A strong answer cites sample size, overfitting, and deployment constraints. A weak answer says "it is faster."

**11. How do you monitor a deployed model?**
A strong answer monitors input drift, score distribution, calibration on labelled samples, and subgroup performance. A weak answer monitors accuracy only.

**12. What is the difference between a feature and a learned representation?**
A strong answer explains interpretability, specification, and validation differences. A weak answer says one is handcrafted and one is learned.

**13. What would you report alongside an AUROC of 0.89?**
A strong answer reports the confidence interval, the prevalence, the operating point, calibration, and subgroup performance. A weak answer reports the AUROC alone.

**14. Why is a temporal split important for a wearable model?**
A strong answer explains that devices, firmware, and populations change over time, and a random split can leak future information into training. A weak answer says it is a valid alternative to a patient split.

**15. How do you decide between a CNN and a transformer?**
A strong answer compares the receptive field requirement, the data size, and the compute budget. A weak answer chooses the newer architecture.

## 15. Key terms

- **1D CNN** — convolutional network over time.
- **Receptive field** — input span influencing one output.
- **Stride** — step between convolution windows.
- **Pooling** — downsampling operation.
- **Residual connection** — skip connection.
- **RNN** — recurrent neural network.
- **LSTM** — long short-term memory.
- **GRU** — gated recurrent unit.
- **Transformer** — attention-based architecture.
- **Attention** — weighted aggregation.
- **Embedding** — learned representation.
- **Segment** — fixed-length input window.
- **Patient-level split** — split by patient.
- **Site-held-out** — split by site.
- **Data leakage** — test information in training.
- **Augmentation** — label-preserving transformation.
- **Class imbalance** — unequal class frequencies.
- **AUROC** — area under ROC curve.
- **AUPRC** — area under precision-recall curve.
- **Calibration** — agreement of predicted and observed risk.
- **ECE** — expected calibration error.
- **Brier score** — mean squared probability error.
- **Temperature scaling** — logit scaling calibration.
- **Aggregation** — combining segment predictions.
- **Model card** — model documentation.
- **Drift** — input distribution change.
- **Concept drift** — input-outcome change.
- **PCCP** — predetermined change control plan.
- **Dilated convolution** — convolution with gaps.
- **Global pooling** — pooling over the whole length.
- **Logit** — pre-softmax score.
- **Reliability plot** — confidence versus accuracy.
- **Calibration slope** — observed on predicted slope.
- **Prior shift** — change in prevalence.
- **Covariate shift** — change in input distribution.
- **Top-k mean** — mean of the highest k segments.
- **Macro average** — average over classes.
- **Micro average** — average over samples.
- **Net benefit** — clinical utility measure.
- **Decision curve** — net benefit across thresholds.
- **Channel dropout** — augmentation that removes a lead.
- **Multi-label** — more than one label per record.
- **Weighted loss** — loss with class weights.
- **Focal loss** — loss that down-weights easy examples.
- **Global average pooling** — mean over the length.
- **Dilated convolution** — convolution with gaps.
- **Temporal split** — split by time.
- **Prospective validation** — validation in the intended setting.
- **Data card** — documentation of a dataset.
- **Human factors** — usability evidence.
- **Cybersecurity** — protection of software and data.
- **SBOM** — software bill of materials.
- **Predetermined change** — pre-authorized model update.
- **Post-market surveillance** — monitoring after launch.
- **Out-of-distribution** — input unlike the training data.
- **OOD detection** — flagging unusual inputs.
- **Prior correction** — adjusting probabilities to a new prevalence.
- **Net benefit** — utility at a threshold.
- **Decision threshold** — cut-point for action.
- **Ensemble** — combination of models.
- **Distillation** — training a small model from a large one.
- **Benchmark** — fixed dataset and metric for comparison.
- **Leaderboard** — ranked benchmark results.

## 16. Sources

All URLs accessed 25 October 2026.

1. Reyna MA, Sadr N, Perez Alday EA, et al. Will two do? Varying dimensions in electrocardiography: the PhysioNet/Computing in Cardiology Challenge 2021. *Physiological Measurement*. 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9469795/
2. PhysioNet/CinC Challenge 2021. https://physionet.org/content/challenge-2021/1.0.3/
3. PhysioNet/CinC Challenge 2021 results. https://moody-challenge.physionet.org/2021/results/
4. US Food and Drug Administration. Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions. Final December 2024, updated August 2025. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
5. US Food and Drug Administration. Predetermined Change Control Plans. https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-and-preparing-submission/predetermined-change-control-plans
6. US Food and Drug Administration. Content of Premarket Submissions for Device Software Functions. 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions
7. IEC 62304, Medical device software — software life cycle processes. https://webstore.iec.ch/
8. ISO 14971, Application of risk management to medical devices. https://www.iso.org/
9. European Union. Artificial Intelligence Act, Regulation (EU) 2024/1689. https://eur-lex.europa.eu/
10. Perez MV, Mahaffey KW, Hedlin H, et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. *N Engl J Med*. 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
11. NeuroKit2 documentation. https://neuropsychology.github.io/NeuroKit/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_feature-engineering.md`](06_feature-engineering.md)
- Next topic: [`08_digital-biomarkers.md`](08_digital-biomarkers.md)
- Related topic: [`10_deployment-on-device.md`](10_deployment-on-device.md)
- Module survey file: [`../../11_physiological_signals_wearables.md`](../../11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/)
