# Module 07 — Deep Learning, Multimodal Models, LLMs, and Agents

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 32 hours (12 lecture, 4 case clinic, 16 lab)
> **Track fit:** Common core; advanced for all tracks
> **Prerequisites:** Module 06; basic linear algebra and calculus
> **Tools:** PyTorch, MONAI, Hugging Face Transformers, PEFT, LangChain or LlamaIndex, vector database
> **Industry question:** "When does a deep or foundation model earn its cost, risk, and complexity over a simpler baseline?"

---

## 1. Why industry cares

Deep learning changed what is technically possible in imaging, waveforms, text, and multimodal data. Foundation models changed the economics of building on top of them. LLMs changed the interface between data and people.

They also changed the failure modes:

- Models that memorize rather than generalize.
- Benchmarks contaminated by training data.
- Confident hallucinations.
- Silent degradation after a model update.
- Enormous compute and energy costs.
- Regulatory uncertainty around adaptive systems.
- Security risks from prompt injection and data exfiltration.

This module teaches enough architecture and training to use these tools competently, and enough evaluation and governance to use them safely.

## 2. Learning outcomes

By the end of this module, you can:

- Explain the core mechanics of CNNs, recurrent networks, transformers, and multimodal models.
- Choose an architecture appropriate to the data modality and sample size.
- Train, fine-tune, and evaluate a deep model with leakage-free splits.
- Apply transfer learning and parameter-efficient fine-tuning.
- Design a retrieval-augmented generation (RAG) system for clinical knowledge.
- Evaluate LLM outputs for factuality, grounding, safety, bias, and robustness.
- Recognize when an agentic workflow is appropriate and when it is dangerous.
- Estimate compute, latency, and cost for a deployment.
- Write a model card and evaluation report for a foundation-model-based product.
- Identify regulatory and security issues specific to generative AI.

## 3. Mental model: representation, adaptation, evaluation

```
Raw data → Representation → Task head → Prediction → Decision
             ↑
        Pretraining (self-supervised, supervised, multimodal)
             ↑
        Adaptation (linear probe, fine-tune, LoRA, prompt, RAG)
             ↑
        Evaluation (external, subgroup, safety, robustness, cost)
```

The central question is not "Which model is largest?" but **"What representation is appropriate, how was it adapted, and what evidence supports its use in this population?"**

## 4. Neural network fundamentals

### 4.1 Building blocks

- **Linear layer:** \(y = Wx + b\).
- **Activation:** ReLU, GELU, sigmoid, softmax.
- **Loss:** cross-entropy, binary cross-entropy, MSE, contrastive, Dice.
- **Optimizer:** SGD, Adam, AdamW.
- **Regularization:** weight decay, dropout, early stopping, data augmentation.
- **Normalization:** batch norm, layer norm, group norm.

### 4.2 Training loop

```python
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        validate(model, val_loader)
```

**Checklist:**

- Reproducible seeds.
- Deterministic data order where possible.
- Learning-rate schedule.
- Gradient clipping for unstable training.
- Checkpointing.
- Early stopping on validation, not test.
- Logging of metrics and learning curves.

### 4.3 Overfitting and underfitting

| Symptom | Diagnosis | Remedy |
|---|---|---|
| Train loss low, val loss high | Overfitting | More data, augmentation, regularization, simpler model |
| Both losses high | Underfitting | Larger model, better features, longer training |
| Val loss erratic | Learning rate too high | Lower LR, schedule, gradient clipping |
| Train/val gap grows late | Memorization | Early stopping, dropout, weight decay |

## 5. Architecture families

### 5.1 Convolutional neural networks

CNNs exploit local spatial structure with convolution, pooling, and weight sharing. They remain strong for images and signals, especially with limited data.

Key concepts:

- Kernel, stride, padding.
- Receptive field.
- Pooling.
- Batch normalization.
- Residual connections.

**Medical imaging uses:** classification, detection, segmentation, registration, reconstruction.

### 5.2 Recurrent networks

RNNs process sequences with a hidden state. LSTMs and GRUs mitigate vanishing gradients. Transformers have largely replaced them for long sequences, but RNNs remain useful for streaming signals and low-latency applications.

### 5.3 Transformers

Self-attention computes relationships between all positions:

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Key concepts:

- Multi-head attention.
- Positional encoding.
- Layer normalization.
- Residual connections.
- Encoder, decoder, encoder-decoder.
- Context window.

**Why transformers matter:** they scale, handle long-range dependencies, and support multimodal tokenization.

### 5.4 Vision transformers

Images are split into patches, embedded as tokens, and processed by a transformer. They require substantial data or pretraining but can capture global context.

### 5.5 Self-supervised learning

Learn representations without labels:

- Masked language modeling.
- Masked image modeling.
- Contrastive learning (SimCLR, MoCo, CLIP).
- Next-token prediction.
- Denoising autoencoders.

**Why it matters:** biomedical labels are scarce and expensive; pretraining can exploit unlabeled data.

### 5.6 Multimodal models

| Fusion strategy | Description | Strength | Weakness |
|---|---|---|---|
| Early fusion | Concatenate raw inputs | Simple | Different modalities misaligned |
| Intermediate fusion | Combine learned representations | Flexible | Complex training |
| Late fusion | Combine predictions | Robust to missing modalities | Ignores interactions |
| Cross-attention | Attend across modalities | Rich interactions | Data hungry |
| Contrastive | Align paired modalities | Strong retrieval | Needs paired data |

**Clinical example:** combine chest X-ray, clinical notes, labs, and vitals for diagnosis or prognosis. Missing modalities are common; design for robustness.

## 6. Medical foundation models

### 6.1 Imaging

Examples:

- **CheXpert / CheXnet:** chest X-ray classification at scale.
- **BiomedCLIP:** multimodal biomedical vision-language pretraining.
- **MedSAM:** promptable segmentation for medical images.
- **MONAI:** open framework for medical imaging AI.
- **RadImageNet:** pretrained imaging representations.

**Caution:** foundation models can encode site-specific shortcuts, scanner artifacts, and demographic biases. Evaluate on external data and with subgroup analysis.

### 6.2 Text

Examples:

- **ClinicalBERT / PubMedBERT:** domain-specific language models.
- **Med-PaLM / Med-PaLM 2:** medical question-answering research.
- **GPT-4 and later general models:** strong general capabilities, variable medical accuracy.
- **Open-weight models:** Llama, Mistral, Qwen, and domain variants.

**Caution:** benchmark performance does not equal clinical safety. Evaluation must reflect the task, population, and workflow.

### 6.3 Omics

Examples:

- **Geneformer:** transformer pretrained on single-cell transcriptomes.
- **scGPT:** foundation model for single-cell biology.
- **ESM:** protein language models.
- **AlphaFold:** protein structure prediction.

**Caution:** biological validation is essential; benchmark improvements may not translate to discovery.

### 6.4 Signals

Examples:

- Pretrained ECG models (e.g., transformer-based).
- EEG foundation models.
- Wearable PPG models.

**Caution:** device, sampling rate, and preprocessing differences can dominate performance.

## 7. LLMs

### 7.1 What an LLM does

An LLM predicts the next token given previous tokens. It learns statistical patterns, not a verified knowledge base. It can produce fluent, confident, and wrong output.

### 7.2 Adaptation methods

| Method | Description | Cost | Best for |
|---|---|---|---|
| Prompting | Instructions and examples in context | Low | Rapid prototyping |
| RAG | Retrieve documents, then generate | Medium | Knowledge-grounded answers |
| Fine-tuning | Update model weights | High | Task/style adaptation |
| PEFT / LoRA | Train small adapter weights | Medium | Efficient adaptation |
| Distillation | Train smaller model from larger | Medium | Cost/latency reduction |
| Tool use | Call calculators, databases, APIs | Medium | Precise operations |

### 7.3 RAG architecture

```
Query → Retriever → Relevant chunks → Prompt with context → LLM → Answer + citations
          ↑
   Vector index / keyword index / hybrid
```

**Requirements:**

- Document ingestion and chunking.
- Metadata and access control.
- Retrieval evaluation (recall@k, MRR).
- Grounded generation with citations.
- Hallucination detection.
- Prompt injection defenses.
- Audit logging.

### 7.4 Evaluation

| Dimension | What to measure |
|---|---|
| Factuality | Correctness against trusted sources |
| Grounding | Citations support claims |
| Completeness | Covers required content |
| Safety | Harmful advice, omissions, unsafe escalation |
| Bias | Differential quality across groups |
| Robustness | Paraphrase, typos, adversarial prompts |
| Calibration | Confidence matches correctness |
| Cost | Tokens, latency, compute |
| Privacy | No leakage of PHI or secrets |

**Evaluation methods:**

- Expert review with rubrics.
- Reference-based metrics (with caution).
- LLM-as-judge (with bias checks).
- Human-in-the-loop.
- Red teaming.
- Prospective silent evaluation.

### 7.5 Hallucination

Types:

- Fabricated fact.
- Misattributed citation.
- Overgeneralization.
- Outdated recommendation.
- Numeric error.
- Omission of safety-critical information.

Mitigations:

- Retrieval with authoritative sources.
- Constrain output format.
- Require citations.
- Abstain when evidence is insufficient.
- Human review for high-stakes decisions.
- Test with adversarial and edge cases.

## 8. Agents

An agent uses an LLM to plan, call tools, observe results, and iterate.

### 8.1 When agents help

- Multi-step retrieval and synthesis.
- Structured data querying.
- Workflow automation with clear checkpoints.
- Drafting and review tasks.

### 8.2 When agents are dangerous

- Irreversible clinical actions.
- Open-ended access to PHI.
- Unbounded tool permissions.
- High-stakes decisions without human review.
- Adversarial environments.

### 8.3 Agent safety architecture

```
User → Policy / guardrails → Planner → Tool calls → Validation → Human approval → Action
                                  ↑
                            Audit log + sandbox
```

**Rules:**

- Least privilege for tools.
- Explicit allowlists.
- Sandboxing.
- Human approval for irreversible actions.
- Rate limits and cost caps.
- Full audit trail.
- Prompt injection defenses.
- Fail closed.

## 9. Compute, latency, and cost

For model compression and on-device deployment, see Module 14B.

| Factor | Impact |
|---|---|
| Model size | Memory, latency, cost |
| Context length | Quadratic attention cost |
| Batch size | Throughput vs latency |
| Quantization | Lower memory, possible accuracy loss |
| Distillation | Smaller, faster, some quality loss |
| Caching | Lower cost for repeated queries |
| Retrieval | Additional latency and infrastructure |
| Human review | Dominant cost in high-stakes settings |

**Industry rule:** the total cost of ownership includes inference, monitoring, human review, integration, and regulatory maintenance.

## 10. Industry case studies

### 10.1 CheXpert and chest X-ray classification

CheXpert introduced a large chest radiograph dataset with uncertain labels and demonstrated deep-learning performance comparable to radiologists on some tasks.

**What transferred:** large-scale pretraining and label uncertainty handling.
**What did not:** performance on selected tasks did not imply comprehensive diagnostic ability.

Source: Irvin, J. et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison," *AAAI*, 2019. https://doi.org/10.1609/aaai.v33i01.3301590

### 10.2 Med-PaLM 2

Med-PaLM 2 was a research model evaluated on medical question answering and long-form responses, with expert evaluation of factuality and safety.

**What transferred:** domain adaptation and expert evaluation.
**What did not:** benchmark scores did not establish clinical deployment readiness.

Source: Singhal, K. et al., "Large language models encode clinical knowledge," *Nature*, 2023. https://doi.org/10.1038/s41586-023-06291-2

### 10.3 BiomedCLIP

BiomedCLIP is a multimodal foundation model pretrained on biomedical image-text pairs, enabling zero-shot and few-shot tasks.

**Industry lesson:** multimodal pretraining can improve data efficiency, but domain shift and label semantics remain challenges.

Source: Zhang, S. et al., "BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs," 2023. https://arxiv.org/abs/2303.00915

### 10.4 MedSAM

MedSAM is a promptable segmentation foundation model adapted to medical images.

**Industry lesson:** foundation models can reduce annotation burden, but require validation across modalities and anatomies.

Source: Ma, J. et al., "Segment anything in medical images," *Nature Communications*, 2024. https://doi.org/10.1038/s41467-024-44824-z

### 10.5 AlphaFold

AlphaFold predicted protein structures with high accuracy, transforming structural biology.

**Industry lesson:** a breakthrough in one task does not automatically solve drug discovery; validation and integration with wet lab work remain essential.

Source: Jumper, J. et al., "Highly accurate protein structure prediction with AlphaFold," *Nature*, 2021. https://doi.org/10.1038/s41586-021-03819-2

## 11. Labs

### Lab 7.1 — Train a CNN on medical images

**Objective:** Build a leakage-free chest X-ray classifier.

**Steps:**

1. Split by patient, not image.
2. Normalize and augment training data only.
3. Train a small CNN and a pretrained backbone.
4. Evaluate AUROC, AUPRC, calibration, and subgroup performance.
5. Inspect errors and saliency maps.

**Deliverable:** model and evaluation report.

**Acceptance criteria:** patient-level split; external or temporal validation if possible; no test-set tuning.

### Lab 7.2 — Fine-tune a clinical language model

**Objective:** Adapt a pretrained model to a clinical text classification task.

**Steps:**

1. De-identify text.
2. Split by patient.
3. Fine-tune with LoRA or a small classifier head.
4. Compare with a keyword baseline.
5. Evaluate subgroup performance and calibration.
6. Document limitations.

**Deliverable:** fine-tuning notebook and model card.

**Acceptance criteria:** no PHI leakage; baseline comparison; error analysis.

### Lab 7.3 — Build a RAG system for guidelines

**Objective:** Answer clinical guideline questions with citations.

**Steps:**

1. Ingest a small corpus of public guidelines.
2. Chunk with metadata.
3. Build a hybrid retriever.
4. Generate answers with mandatory citations.
5. Evaluate retrieval recall and answer grounding.
6. Test adversarial prompts and out-of-scope questions.

**Deliverable:** RAG prototype and evaluation.

**Acceptance criteria:** citations support claims; abstains when evidence is missing; prompt injection tested.

### Lab 7.4 — Multimodal fusion

**Objective:** Combine imaging and tabular data for a prediction task.

**Steps:**

1. Build unimodal baselines.
2. Implement early and late fusion.
3. Handle missing modalities.
4. Compare performance and robustness.
5. Analyze whether fusion helps equally across subgroups.

**Deliverable:** comparison report.

**Acceptance criteria:** missing-modality analysis; no leakage; calibration reported.

### Lab 7.5 — Red-team an LLM

**Objective:** Identify failure modes in a clinical assistant.

**Steps:**

1. Test factuality, omission, unsafe advice, bias, and prompt injection.
2. Create a taxonomy of failures.
3. Rate severity.
4. Propose guardrails.
5. Re-test after mitigation.

**Deliverable:** red-team report.

**Acceptance criteria:** specific reproducible prompts; severity rubric; mitigations tested.

### Lab 7.6 — Cost and latency budget

**Objective:** Estimate deployment cost for three model options.

**Steps:**

1. Measure tokens per request and requests per day.
2. Estimate inference cost.
3. Add retrieval and human review.
4. Compare cloud API vs self-hosted.
5. Recommend an option with a break-even analysis.

**Deliverable:** cost model.

**Acceptance criteria:** assumptions explicit; sensitivity analysis; total cost included.

## 12. Assignment: foundation-model evaluation dossier

Choose a clinical task and evaluate a foundation model or multimodal system. Write a 2,000-word dossier:

1. Intended use and decision.
2. Model and adaptation method.
3. Data and evaluation population.
4. External validation.
5. Factuality, grounding, and safety.
6. Subgroup performance.
7. Robustness and adversarial testing.
8. Cost and latency.
9. Regulatory and security issues.
10. Recommendation and monitoring plan.

## 13. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Architecture choice | 15% | Follows hype | Matches modality and data | Justifies against simpler baselines |
| Training | 15% | Runs notebook | Reproducible pipeline | Robust to shift and imbalance |
| Evaluation | 25% | Benchmark only | External and subgroup | Safety, grounding, calibration, cost |
| LLM safety | 20% | No red teaming | Basic failure taxonomy | Systematic adversarial evaluation |
| Governance | 15% | Ignores regulation | Identifies issues | Designs controls and monitoring |
| Communication | 10% | Technical | Clear | Executive-ready |

## 14. Regulatory, ethical, and commercial lens

- **FDA:** generative AI in a medical device raises questions about intended use, validation, change control, and output variability.
- **EU AI Act:** high-risk AI systems face requirements for risk management, data governance, transparency, human oversight, and post-market monitoring.
- **HIPAA/GDPR:** sending PHI to external LLM APIs requires a BAA or equivalent and a lawful basis.
- **Security:** prompt injection, data exfiltration, model inversion, and supply-chain risks.
- **Intellectual property:** training data and generated content may have license restrictions.
- **Commercial:** foundation models can reduce development time but increase dependency, inference cost, and vendor risk.

## 15. Failure modes and red flags

- Using a benchmark that overlaps with pretraining data.
- Splitting images rather than patients.
- Fine-tuning on the test set.
- Reporting benchmark accuracy as clinical validity.
- No hallucination evaluation.
- No citation verification.
- Sending PHI to an unapproved API.
- Giving an agent unrestricted tool access.
- Ignoring latency and cost.
- No plan for model updates.
- Assuming a larger model is always better.

## 16. Interview and discussion questions

1. **When would you use a CNN instead of a transformer?**
   Hint: data size, spatial locality, compute, latency.
2. **What is self-supervised learning, and why is it useful in medicine?**
   Hint: unlabeled data, pretraining, label scarcity.
3. **How do you evaluate an LLM for clinical use?**
   Hint: expert review, grounding, safety, bias, robustness, prospective silent mode.
4. **What is RAG, and what does it not solve?**
   Hint: retrieval quality, conflicting sources, generation errors.
5. **How do you prevent prompt injection?**
   Hint: least privilege, input sanitization, tool allowlists, sandboxing, human approval.
6. **What is the difference between fine-tuning and RAG?**
   Hint: weights vs external knowledge; cost, freshness, attribution.
7. **How do you detect benchmark contamination?**
   Hint: canary strings, temporal splits, private test sets, memorization probes.
8. **What are the risks of an agent in a clinical workflow?**
   Hint: irreversible actions, PHI exposure, compounding errors, accountability.
9. **How would you estimate inference cost?**
   Hint: tokens, requests, latency, hardware, human review.
10. **What is a multimodal fusion strategy?**
    Hint: early, intermediate, late, cross-attention.
11. **Why can a foundation model fail on a new hospital?**
    Hint: scanner, population, coding, workflow, preprocessing differences.
12. **What does "human in the loop" actually require?**
    Hint: time, authority, information, accountability, override tracking.

## 17. Key terms

| Term | Definition |
|---|---|
| CNN | Convolutional neural network |
| Transformer | Attention-based sequence model |
| Self-attention | Relating all positions in a sequence |
| Foundation model | Large pretrained model adaptable to many tasks |
| Transfer learning | Reusing pretrained representations |
| LoRA | Low-rank adaptation for efficient fine-tuning |
| RAG | Retrieval-augmented generation |
| Hallucination | Fluent but unsupported or false output |
| Grounding | Tying output to retrieved evidence |
| Agent | LLM that plans and calls tools |
| Prompt injection | Malicious input that hijacks model behavior |
| Multimodal | Combining multiple data types |
| Contrastive learning | Learning by pulling similar pairs together |
| Distillation | Training a smaller model from a larger one |
| Benchmark contamination | Test data present in pretraining |

## 18. Sources and further reading

### Architecture and foundation models
- LeCun, Y., Bengio, Y., Hinton, G., "Deep learning," *Nature*, 2015. https://doi.org/10.1038/nature14539
- Vaswani, A. et al., "Attention Is All You Need," *NeurIPS*, 2017. https://arxiv.org/abs/1706.03762
- Dosovitskiy, A. et al., "An Image is Worth 16x16 Words," *ICLR*, 2021. https://arxiv.org/abs/2010.11929
- Radford, A. et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021. https://arxiv.org/abs/2103.00020

### Medical AI
- Irvin, J. et al., "CheXpert," *AAAI*, 2019. https://doi.org/10.1609/aaai.v33i01.3301590
- Singhal, K. et al., "Large language models encode clinical knowledge," *Nature*, 2023. https://doi.org/10.1038/s41586-023-06291-2
- Zhang, S. et al., "BiomedCLIP," 2023. https://arxiv.org/abs/2303.00915
- Ma, J. et al., "Segment anything in medical images," *Nature Communications*, 2024. https://doi.org/10.1038/s41467-024-44824-z
- Jumper, J. et al., "Highly accurate protein structure prediction with AlphaFold," *Nature*, 2021. https://doi.org/10.1038/s41586-021-03819-2
- Thirunavukarasu, A. J. et al., "Large language models in medicine," *Nature Medicine*, 2023. https://doi.org/10.1038/s41591-023-02448-8

### Guidance and tools
- WHO, *Ethics and governance of artificial intelligence for health*, 2021. https://www.who.int/publications/i/item/9789240029200
- FDA, *AI/ML SaMD Action Plan*. https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
- NIST AI Risk Management Framework. https://www.nist.gov/itl/ai-risk-management-framework
- PyTorch: https://pytorch.org/
- MONAI: https://monai.io/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- PEFT: https://huggingface.co/docs/peft/

> **Next:** [Module 08](08_ehr_claims_operations.md) applies these methods to EHR, claims, and operations — the highest-volume data in health systems.
