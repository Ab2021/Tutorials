# Module 14B — Efficient AI: Distillation, Quantization, and On-Device Inference

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 28 hours (10 lecture, 2 case clinic, 16 lab)
> **Track fit:** MedTech, Sensors & Imaging advanced elective; useful for Health Systems edge deployment
> **Prerequisites:** Modules 06, 07, 11, 14A
> **Tools:** PyTorch, ONNX, ONNX Runtime, TensorRT, TensorFlow Lite Micro, CMSIS-NN, STM32Cube.AI, Edge Impulse, ExecuTorch, llama.cpp or MLC-LLM, a target board or simulator
> **Industry question:** "Can this model meet its clinical accuracy, latency, energy, memory, and safety requirements on the device that will actually run it?"

---

## 1. Why this module exists

A model that runs in a cloud notebook has no hardware constraints. A model inside a wearable, monitor, infusion pump, surgical robot, handheld ultrasound, or implant does.

On-device inference is not a deployment detail added at the end. It changes:

- Which architecture is feasible.
- How much battery the device needs.
- Whether the latency supports the clinical decision.
- How much memory and silicon the product costs.
- Whether the model can be updated safely.
- Whether subgroup performance survives compression.
- Whether the regulator sees a new model or a pre-authorized change.

The common failure pattern:

1. Train a large model on a workstation.
2. Quantize it to INT8.
3. Measure accuracy on the original test set.
4. Deploy.
5. Discover latency is 4× the budget, memory overflows, the NPU does not support a key operator, calibration has shifted, and a subgroup has degraded.
6. Retrain, re-quantize, revalidate, and delay the product.

This module teaches the full efficiency stack: **distillation, quantization, pruning, runtime optimization, target-hardware measurement, clinical validation, and regulatory change control.**

## 2. Learning outcomes

By the end of this module, you can:

- Specify accuracy, calibration, latency, energy, memory, and cost budgets for a device.
- Profile a model's parameters, MACs, memory, latency, and energy on a target.
- Choose among distillation, quantization, pruning, low-rank approximation, and architecture search.
- Distill a teacher model into a student without hiding subgroup or calibration regressions.
- Apply post-training quantization and quantization-aware training correctly.
- Evaluate quantization error, operator fallback, and hardware utilization.
- Optimize inference graphs for CPU, GPU, NPU, DSP, FPGA, and MCU targets.
- Design an on-device runtime with buffering, watchdogs, fallback, and secure updates.
- Validate a compressed model on target hardware, including latency tails and thermal throttling.
- Map model changes to PCCP, IEC 62304, ISO 14971, and post-market monitoring.
- Write an efficient-model dossier that a hardware, clinical, and regulatory reviewer can accept.

## 3. Mental model: the deployment Pareto frontier

```
Clinical utility
      ▲
      │        ● large cloud model
      │      ●
      │    ●   ← feasible frontier
      │  ●
      └──────────────────────────► latency / energy / memory / cost
```

A model is deployable only if it sits inside **all** constraints simultaneously:

\[
\text{deployable} = \text{accuracy} \ge a_{\min} \land \text{calibration} \le c_{\max} \land \text{latency} \le l_{\max} \land \text{energy} \le e_{\max} \land \text{memory} \le m_{\max}
\]

There is no single "best" compressed model. There is a Pareto frontier of models that trade accuracy for resources.

### 3.1 End-to-end latency budget

\[
L_{\text{total}} = L_{\text{sensor}} + L_{\text{preprocess}} + L_{\text{inference}} + L_{\text{postprocess}} + L_{\text{network}} + L_{\text{human}}
\]

A 10 ms model is useless if sensor buffering adds 200 ms or the clinician reviews the alert 30 minutes later. Budget the whole pathway.

### 3.2 The efficiency questions

1. What is the clinical decision, and how fast must it happen?
2. What is the target device, runtime, and operator support?
3. What are the accuracy and calibration floors?
4. What are the latency, energy, memory, and cost ceilings?
5. Which compression methods preserve the clinical signal?
6. What evidence proves it on the target hardware?
7. What change-control pathway covers the compressed model?

## 4. Target hardware and performance

### 4.1 Hardware classes

| Target | Strengths | Constraints | Typical use |
|---|---|---|---|
| MCU | Ultra-low power, deterministic, cheap | KB–MB memory, no OS or RTOS, limited math | Wearables, sensors, implants, pumps |
| DSP | Efficient fixed-point signal processing | Specialized toolchain, limited model support | ECG/PPG/audio, filtering |
| FPGA | Low latency, deterministic, reconfigurable | Engineering effort, power, cost | Imaging front ends, real-time control |
| SoC + NPU | High throughput/watt for neural nets | Operator support, memory bandwidth, thermal | Phones, wearables, handheld ultrasound |
| GPU | Flexible, mature tooling | Power, thermal, cost | Workstations, edge servers, robots |
| ASIC | Best efficiency at volume | Huge NRE, long lifecycle, inflexible | High-volume implants, sensors |
| Cloud/edge server | Elastic, easy updates | Network latency, privacy, connectivity | Cohort analytics, non-real-time |

**Rule:** the target is a system, not a chip. Memory bandwidth, thermal design, battery, sensor interface, and runtime support often dominate the model architecture.

### 4.2 Memory hierarchy

| Level | Typical size | Latency | Implication |
|---|---:|---|---|
| Registers | Bytes–KB | 1 cycle | Keep hot loops small |
| SRAM/cache | KB–MB | Few cycles | Weights and activations must fit |
| DRAM | GB | Hundreds of cycles | Bandwidth-bound inference |
| Flash | MB–GB | Slow reads | Model storage; execution from flash is slower |
| External storage | GB+ | Very slow | Logs, images, updates |

**Peak memory** is not model file size. It includes weights, activations, workspace, input/output buffers, runtime, and fragmentation.

### 4.3 Roofline intuition

Arithmetic intensity:

\[
I = \frac{\text{operations}}{\text{bytes moved}}
\]

- Low intensity: memory-bound; faster compute does not help.
- High intensity: compute-bound; NPU/GPU throughput matters.
- Quantization reduces bytes moved, which can help memory-bound models more than FLOPs reduction suggests.

### 4.4 Metrics that matter

| Metric | Definition | Why it matters |
|---|---|---|
| Parameters | Learned weights | Storage and memory |
| MACs/FLOPs | Arithmetic operations | Compute estimate, not latency |
| Model size | Serialized bytes | Flash/storage |
| Peak memory | Maximum resident bytes | Feasibility |
| Latency p50/p95/p99 | Time per inference | Clinical deadline and tail risk |
| Throughput | Inferences/second | Batch or multi-patient use |
| Energy/inference | Joules per inference | Battery and thermal |
| Power | Watts | Thermal and battery |
| TOPS/W | Effective compute per watt | Chip comparison, marketing caveat |
| Utilization | Active fraction of compute/memory | Headroom and inefficiency |
| Cost | BOM + cloud + energy + maintenance | Total product cost |

**Rule:** FLOPs are a proxy. Measure latency and energy on the target.

## 5. The compression family

| Method | What it reduces | Accuracy risk | Hardware benefit | When to use |
|---|---|---|---|---|
| Distillation | Model size/compute | Teacher bias, capacity gap | Smaller student | Teacher available; latency matters |
| Quantization | Weight/activation precision | Calibration, outliers, operator issues | Memory, bandwidth, compute | Most edge deployments |
| Pruning | Parameters/connections | Over-pruning, bias | Only with hardware sparsity | Sparse-aware target |
| Structured pruning | Channels/blocks | Capacity loss | Real latency reduction | Dense hardware |
| Low-rank | Matrix rank | Representation loss | Compute/memory | Large linear layers |
| Weight sharing | Unique values | Codebook error | Storage | Tiny models |
| NAS | Architecture | Search cost, overfitting | Task-specific efficiency | Large engineering budget |
| Operator fusion | Memory traffic | Numerical differences | Latency | Runtime/graph optimization |

### 5.1 Order of operations

A practical sequence:

1. Establish an uncompressed baseline on target hardware.
2. Profile bottlenecks: compute, memory, I/O, preprocessing.
3. Try runtime/graph optimization first.
4. Quantize to INT8 with PTQ.
5. If accuracy/calibration fails, try QAT or mixed precision.
6. Distill if the architecture itself is too large.
7. Prune only if the target supports sparsity.
8. Revalidate accuracy, calibration, subgroup, latency, energy, and safety.
9. Document the final model and change path.

## 6. Knowledge distillation

### 6.1 Core idea

A teacher model \(T\) transfers knowledge to a student \(S\). The student is trained on a combination of hard labels and soft teacher outputs.

\[
\mathcal{L} = \alpha \mathcal{L}_{\text{CE}}(y, S(x)) + (1-\alpha) T^2 \mathrm{KL}\left(\mathrm{softmax}\left(\frac{T(x)}{T}\right) \,\middle\|\, \mathrm{softmax}\left(\frac{S(x)}{T}\right)\right)
\]

- \(T\): temperature; higher softens probabilities.
- \(\alpha\): balance between hard and soft targets.
- \(T^2\): keeps gradient scale stable when temperature changes.

### 6.2 Variants

| Variant | Signal | Use |
|---|---|---|
| Response-based | Teacher logits/probabilities | Classification, detection |
| Feature-based | Intermediate representations | Vision, speech |
| Relation-based | Similarities between examples/layers | Representation learning |
| Self-distillation | Same architecture, deeper/ensemble teacher | No external teacher |
| Online distillation | Teacher and student trained together | Limited compute |
| Ensemble distillation | Multiple teachers | Robustness |
| Born-again | Student becomes next teacher | Iterative compression |
| LLM distillation | Logits, rationales, chain-of-thought | Task-specific small models |

### 6.3 Medical pitfalls

- **Teacher bias:** the student inherits the teacher's subgroup errors. Audit both.
- **Teacher overconfidence:** soft labels can be miscalibrated; temperature and calibration matter.
- **Capacity gap:** a very small student cannot match a large teacher; choose the student size empirically.
- **Label noise:** distillation can amplify noisy labels; clean or reweight.
- **Distribution shift:** student may learn teacher shortcuts; test external sites.
- **Calibration:** re-check reliability and decision curves after distillation.
- **Regulatory:** a distilled model is a new model, not a formatting change.

### 6.4 Distillation for LLMs

Options:

- **Logit distillation:** needs teacher vocabulary and logits; expensive.
- **Sequence-level distillation:** train on teacher-generated outputs.
- **Rationale distillation:** train on explanations/chain-of-thought; can propagate reasoning errors.
- **Task-specific distillation:** small model for one workflow, not general chat.
- **Preference distillation:** align to clinician preferences; needs careful evaluation.

**Rule:** a distilled LLM is a new clinical model. Evaluate factuality, grounding, omission, bias, and safety, not only benchmark accuracy.

## 7. Quantization

### 7.1 Number formats

| Format | Bits | Range/use | Notes |
|---|---:|---|---|
| FP32 | 32 | Training/reference | Baseline |
| FP16/BF16 | 16 | Inference/training | BF16 has wider range, less precision |
| FP8 (E4M3/E5M2) | 8 | Accelerator inference | Needs hardware support |
| INT8 | 8 | Common edge inference | Mature tooling |
| INT4 | 4 | Weight-only LLM/edge | Accuracy and packing complexity |
| Binary/ternary | 1–2 | Research/ultra-low power | Severe accuracy risk |
| MX/block formats | 4–8 | Microscaling | Shared exponent per block |

### 7.2 Affine quantization

For a real value \(x\), scale \(S\), zero point \(Z\), and bit width \(b\):

\[
q = \mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{S}\right) + Z,\ 0,\ 2^b-1\right)
\]

\[
\hat{x} = S(q-Z)
\]

Asymmetric quantization uses a zero point; symmetric quantization sets \(Z=0\).

**Symmetric scale:**

\[
S = \frac{\max |x|}{2^{b-1}-1}
\]

**Asymmetric scale:**

\[
S = \frac{x_{\max}-x_{\min}}{2^b-1}, \quad Z = \mathrm{round}\left(-\frac{x_{\min}}{S}\right)
\]

### 7.3 Granularity

| Granularity | Description | Trade-off |
|---|---|---|
| Per-tensor | One scale for whole tensor | Simple, sensitive to outliers |
| Per-channel | One scale per output channel | Better accuracy, more metadata |
| Per-group/block | Scale per group of weights | Common for LLMs |
| Per-token/activation | Dynamic scale per input | Better for transformers |

### 7.4 PTQ vs QAT

| Method | How | Data | Accuracy | Effort |
|---|---|---|---|---|
| Post-training quantization (PTQ) | Calibrate ranges on sample data | Hundreds–thousands of samples | Good for CNNs; can fail on transformers/outliers | Low |
| Quantization-aware training (QAT) | Simulate quantization during training | Full training set | Better, especially low bit | High |
| Dynamic quantization | Quantize activations at runtime | Calibration | Good for NLP/linear layers | Medium |
| Weight-only quantization | Quantize weights, keep activations FP16 | Calibration | Good for LLM memory | Medium |

### 7.5 Calibration

- Use representative data, not the test set.
- Cover subgroups, sites, devices, and edge cases.
- Use min-max, percentile, MSE, or KL divergence.
- Check activation outliers; transformers often need percentile or clipping.
- Keep a held-out calibration audit set.
- Document calibration data version and provenance.

### 7.6 QAT and the straight-through estimator

Quantization rounds; rounding has zero gradient almost everywhere. QAT uses a straight-through estimator (STE):

\[
\frac{\partial \hat{x}}{\partial x} \approx 1
\]

This passes gradients through the quantizer as if it were identity. Batch normalization folding, bias correction, and careful warm-up improve stability.

### 7.7 Operator support and fallback

An NPU may support convolution, depthwise convolution, and ReLU, but not:

- Certain activations (GELU, SiLU).
- LayerNorm in some layouts.
- Dynamic shapes.
- Custom attention.
- Non-max suppression.
- Some resize/interpolation modes.

Unsupported operators fall back to CPU, causing latency spikes. **Check the operator list before choosing the architecture.**

### 7.8 Quantization and calibration/fairness

Quantization can:

- Shift predicted probabilities.
- Increase false negatives in a subgroup.
- Change the optimal threshold.
- Interact with class imbalance.
- Degrade rare classes more than common ones.

**Required evaluation:** accuracy, calibration, sensitivity/specificity at the operating threshold, subgroup performance, and decision-curve net benefit, before and after quantization.

## 8. Pruning, sparsity, and low-rank methods

### 8.1 Pruning types

| Type | What is removed | Hardware speedup |
|---|---|---|
| Unstructured | Individual weights | Only with sparse hardware |
| Structured | Channels, heads, blocks | Dense hardware |
| N:M sparsity | N nonzeros per M block | Supported on some GPUs/NPUs |
| Movement pruning | Weights moving toward zero | Similar caveats |
| Lottery ticket | Sparse trainable subnetwork | Research; not automatic deployment |

### 8.2 Low-rank and weight sharing

- SVD/low-rank factorization of large matrices.
- Shared codebooks/weight clustering.
- Tensor decomposition.
- Useful for LLM and large linear layers; may not map efficiently to all hardware.

### 8.3 The latency trap

A model with 90% sparse weights can be slower than a dense model if the hardware lacks sparse kernels. **Pruning reduces parameters, not necessarily latency.** Measure on target.

## 9. Inference optimization

### 9.1 Graph and runtime optimization

| Optimization | What it does | Risk |
|---|---|---|
| Constant folding | Precompute constant subgraphs | None if correct |
| Operator fusion | Combine conv+BN+ReLU, etc. | Numerical differences |
| Layout optimization | NHWC/NCHW, tiling | Wrong layout slows or fails |
| Kernel autotuning | Select fastest kernel | Target-specific |
| Memory planning | Reuse buffers | Peak memory bugs |
| Quantization | Lower precision | Accuracy/calibration |
| Graph partitioning | Split CPU/NPU/GPU | Fallback latency |
| Caching | Reuse computations | Staleness, privacy |

### 9.2 Runtime/framework choices

| Runtime | Targets | Strengths | Cautions |
|---|---|---|---|
| ONNX Runtime | CPU/GPU/NPU | Broad model support | Operator coverage varies |
| TensorRT | NVIDIA GPU | Strong NVIDIA optimization | NVIDIA-specific |
| TensorFlow Lite / Lite Micro | Mobile/MCU | Mature TinyML path | Operator and memory limits |
| CMSIS-NN | Arm Cortex-M | Hand-optimized kernels | Model/toolchain constraints |
| STM32Cube.AI | STM32 | MCU deployment | Memory and operator limits |
| Edge Impulse | MCU/edge | End-to-end workflow | Vendor workflow |
| ExecuTorch | Mobile/edge | PyTorch-native | Maturing ecosystem |
| OpenVINO | Intel CPU/GPU/NPU | Intel optimization | Intel-specific |
| TVM/XLA | Multiple | Compiler optimization | Engineering effort |
| llama.cpp / MLC-LLM | Local LLM | Quantized LLM inference | Medical validation still required |

### 9.3 LLM-specific inference

- **KV cache:** memory grows with context; quantize or page it.
- **Paged attention:** reduces fragmentation.
- **Speculative decoding:** small draft model + large verifier; latency gain depends on acceptance.
- **Weight-only quantization:** W4A16/W8A16 reduce memory; activations stay higher precision.
- **Continuous batching:** throughput for servers, not single-patient latency.
- **Prompt caching:** reuse system/context; privacy and staleness.
- **Distillation:** task-specific small model.

**Rule:** an LLM that fits in memory may still be too slow or too hot for a handheld device.

## 10. On-device architecture

```
Sensor → driver → DMA buffer → preprocess → model → postprocess → output
                                      │
                              runtime / NPU / CPU
                                      │
                         watchdog, fallback, logging, OTA
```

### 10.1 Runtime design

- Static memory allocation where possible.
- Double buffering for sensor input.
- DMA to reduce CPU copy.
- Real-time scheduling and priority inversion avoidance.
- Watchdog and timeout.
- Fallback to a simple rule or cloud path.
- Deterministic behavior under power loss.
- Model and firmware version reporting.

### 10.2 Model updates

- Signed firmware/model bundles.
- A/B partitions with rollback.
- Anti-rollback protection.
- Compatibility checks.
- Post-update self-test.
- Audit log.
- PCCP or new submission if the change is not pre-authorized.

### 10.3 Data and privacy

- Infer on device when raw data should not leave.
- Send only necessary features or events.
- Encrypt storage and transport.
- Log without PHI.
- Consider membership inference and model extraction.

## 11. Medical and regulatory constraints

| Requirement | Implication for efficiency work |
|---|---|
| IEC 62304 | Model, preprocessing, runtime, and toolchain are software items; version and verify them |
| IEC 60601-1 | Electrical safety, essential performance, alarms |
| ISO 14971 | Compression-induced failure modes belong in the risk file |
| IEC 62366-1 | Latency and uncertainty must be usable |
| FDA PCCP | Planned model changes need a modification protocol and impact assessment |
| FDA GMLP | Representative data, independent test sets, clinically relevant conditions |
| Cybersecurity | Secure boot, signed updates, SBOM, vulnerability monitoring |
| EU MDR / AI Act | Clinical evaluation, transparency, human oversight, post-market monitoring |

**Rule:** quantization, distillation, pruning, and runtime changes can alter clinical output. Treat them as controlled changes, not build optimizations.

## 12. Validation on target hardware

### 12.1 Accuracy and calibration

- Compare compressed vs uncompressed on the same held-out set.
- Report confidence intervals and paired tests.
- Check calibration slope/intercept and Brier score.
- Check sensitivity/specificity at the clinical operating threshold.
- Check subgroup performance and intersectional groups.
- Check rare classes and edge cases.

### 12.2 Performance

| Metric | How to measure |
|---|---|
| Latency | p50/p95/p99 over many runs; cold and warm start |
| Throughput | Inferences/second at expected batch size |
| Energy | Joules/inference with a power monitor |
| Peak memory | High-water mark, not model file size |
| Thermal | Sustained load, throttling, skin temperature |
| Utilization | CPU/GPU/NPU/memory utilization |
| Stability | 24–72 hour soak test |
| Power loss | Recovery and safe state |
| Update | OTA success, rollback, version report |

### 12.3 Non-inferiority

Pre-specify a margin. For example:

\[
\Delta \text{sensitivity} > -2\% \quad \text{and} \quad \Delta \text{calibration slope} > -0.1
\]

The margin must be clinically justified, not chosen after seeing results.

### 12.4 Hardware-in-the-loop

Test with real sensor input, real power, real thermal conditions, and real runtime. A desktop emulator is not target validation.

## 13. Economics

| Cost | Driver |
|---|---|
| BOM | Silicon, memory, battery, cooling |
| NRE | Chip/board/toolchain engineering |
| Inference energy | Battery life and electricity |
| Cloud offload | Connectivity, latency, privacy |
| Validation | Target-hardware testing and regulatory evidence |
| Maintenance | Model updates, fleet support, recalls |
| Opportunity cost | Delayed launch, larger device |

**Rule:** the cheapest model is not the one with the fewest parameters; it is the one that meets the clinical requirement at the lowest total lifecycle cost.

## 14. Case studies

### 14.1 ECG arrhythmia on a wearable MCU

A compact 1D CNN is distilled from a larger transformer, quantized to INT8, and deployed with CMSIS-NN or TFLite Micro.

**Bottlenecks:** SRAM for activations, flash for weights, ADC/DMA buffering, battery, motion artifact.

**Validation:** sensitivity at fixed false alarms per hour, calibration, subgroup performance, latency, energy per inference, and performance after firmware update.

### 14.2 PPG heart rate and AF screening

A PPG model is compressed for a wrist device.

**Bottlenecks:** optical front end, motion rejection, skin-tone bias, battery, memory.

**Validation:** stratified error by skin tone and perfusion, false-alert rate, latency, and behavior during motion.

### 14.3 Handheld or robotic ultrasound

An imaging model runs on a handheld probe or edge box.

**Bottlenecks:** frame rate, memory bandwidth, thermal, image preprocessing, operator dependence.

**Validation:** per-frame latency, frame drop rate, image quality, detection sensitivity, and clinical workflow.

### 14.4 Surgical robot perception

A vision model runs on a robot's edge compute for tool tracking or phase recognition.

**Bottlenecks:** deterministic latency, thermal, safety monitor, camera synchronization.

**Validation:** p99 latency, dropped frames, tool-tracking error, and safe fallback.

### 14.5 CGM and insulin delivery

A glucose forecasting or anomaly model runs on a pump/phone.

**Bottlenecks:** battery, memory, sensor lag, safety, cybersecurity.

**Validation:** hypoglycemia sensitivity, false suspension rate, latency, and behavior during sensor loss.

### 14.6 On-device LLM triage

A small quantized LLM drafts a triage note or patient response on a phone.

**Bottlenecks:** memory, latency, thermal, hallucination, privacy.

**Validation:** factuality, omission, safety escalation, bias, latency, and human review. Do not deploy without a clinical safety net.

## 15. Labs

### Lab 14B.1 — Profile a model

**Objective:** Measure parameters, MACs, model size, peak memory, and latency.

**Steps:**

1. Choose a CNN and a transformer.
2. Count parameters and MACs.
3. Export to ONNX.
4. Measure latency on CPU and target.
5. Measure peak memory.
6. Compare FLOPs to latency.

**Deliverable:** profile report.

**Acceptance criteria:** cold/warm latency; memory measured; FLOPs not treated as latency.

### Lab 14B.2 — Post-training quantization

**Objective:** Quantize a model to INT8 and evaluate.

**Steps:**

1. Calibrate on representative data.
2. Quantize weights and activations.
3. Compare accuracy, calibration, and subgroup performance.
4. Inspect per-layer error.
5. Test operator fallback.
6. Measure latency and memory.

**Deliverable:** PTQ report.

**Acceptance criteria:** calibration data documented; subgroup analysis; target latency.

### Lab 14B.3 — Quantization-aware training

**Objective:** Recover accuracy with QAT.

**Steps:**

1. Add fake quantization.
2. Fine-tune with STE.
3. Fold batch normalization.
4. Compare PTQ vs QAT.
5. Evaluate calibration and subgroups.
6. Export and benchmark.

**Deliverable:** QAT model and comparison.

**Acceptance criteria:** accuracy/calibration recovery; no test-set tuning.

### Lab 14B.4 — Distillation

**Objective:** Distill a teacher into a smaller student.

**Steps:**

1. Train or obtain a teacher.
2. Choose student capacity.
3. Tune temperature and alpha.
4. Compare hard-label training vs distillation.
5. Evaluate calibration, subgroups, and robustness.
6. Measure latency and energy.

**Deliverable:** distilled model and report.

**Acceptance criteria:** teacher/student comparison; bias audit; target benchmark.

### Lab 14B.5 — Deploy to MCU or edge

**Objective:** Run a model on a constrained target.

**Steps:**

1. Convert to TFLite Micro, ONNX Runtime, or target toolchain.
2. Measure flash, SRAM, latency, and energy.
3. Test sensor input and preprocessing.
4. Test watchdog and fallback.
5. Run a soak test.
6. Record firmware/model versions.

**Deliverable:** deployment report.

**Acceptance criteria:** target measurements; no desktop-only claims.

### Lab 14B.6 — LLM quantization

**Objective:** Quantize a small LLM and evaluate clinical safety.

**Steps:**

1. Choose a small open model.
2. Apply weight-only INT4/INT8 quantization.
3. Measure memory, latency, and throughput.
4. Evaluate factuality, grounding, omission, and safety.
5. Compare with the unquantized model.
6. Document failure modes.

**Deliverable:** LLM efficiency and safety report.

**Acceptance criteria:** clinical rubric; no benchmark-only claim.

### Lab 14B.7 — Hardware-in-the-loop latency and energy

**Objective:** Measure p99 latency and energy on target.

**Steps:**

1. Build a HIL fixture.
2. Inject real sensor data.
3. Measure cold/warm p50/p95/p99.
4. Measure joules/inference.
5. Run sustained load and thermal test.
6. Test power loss and recovery.

**Deliverable:** HIL report.

**Acceptance criteria:** tail latency; energy; thermal; recovery.

### Lab 14B.8 — Subgroup robustness after compression

**Objective:** Test whether compression widens disparities.

**Steps:**

1. Define subgroups.
2. Compare uncompressed, PTQ, QAT, and distilled models.
3. Measure sensitivity, calibration, and PPV by group.
4. Identify the worst regression.
5. Decide whether to accept, recalibrate, or reject.

**Deliverable:** fairness report.

**Acceptance criteria:** confidence intervals; clinical threshold; decision.

### Lab 14B.9 — Pareto frontier and deployment decision

**Objective:** Choose a model for a device.

**Steps:**

1. Plot accuracy vs latency vs energy for candidates.
2. Add clinical and regulatory constraints.
3. Identify dominated models.
4. Choose a candidate.
5. Write the deployment decision memo.

**Deliverable:** Pareto plot and memo.

**Acceptance criteria:** constraints explicit; decision defensible.

## 16. Assignment: efficient-model dossier

Choose a device and model. Write a 2,500-word dossier:

1. Clinical decision and latency budget.
2. Target hardware and runtime.
3. Baseline model and profile.
4. Compression strategy.
5. Distillation/quantization details.
6. Calibration data and ranges.
7. Accuracy, calibration, and subgroup results.
8. Latency, energy, memory, and thermal results.
9. Failure modes and fallback.
10. Regulatory change path and monitoring.

## 17. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem and budget | 15% | Model-first | Clear latency/energy budget | End-to-end clinical budget |
| Profiling | 15% | Params only | Latency and memory | Roofline and target utilization |
| Compression | 20% | One method | PTQ + comparison | Distillation/quantization/pruning trade-off |
| Target validation | 25% | Desktop only | Target latency/accuracy | p99, energy, thermal, HIL |
| Fairness and safety | 15% | Overall accuracy | Subgroup analysis | Mitigation and monitoring |
| Regulatory | 10% | Ignores | Change path identified | PCCP-aligned evidence |

## 18. Regulatory, ethical, and commercial lens

- **Model change:** compression can change outputs; assess whether it is a PCCP-covered change.
- **Traceability:** record model, calibration data, toolchain, runtime, and target hardware versions.
- **Cybersecurity:** quantized models and runtimes are part of the attack surface.
- **Equity:** compression can widen subgroup gaps; test before and after.
- **Battery and access:** efficiency determines whether a device works in low-resource settings.
- **Commercial:** BOM, battery, and thermal constraints can decide product viability.

## 19. Failure modes and red flags

- Measuring accuracy on the original test set only.
- Using the test set for calibration.
- Assuming INT8 is always faster.
- Ignoring operator fallback.
- Reporting model file size as peak memory.
- Ignoring p99 latency.
- Ignoring thermal throttling.
- Distilling from a biased teacher.
- Pruning without hardware sparsity.
- Quantizing LLM KV cache without evaluation.
- No model/firmware version traceability.
- Treating compression as a non-regulated change.

## 20. Interview and discussion questions

1. **Why is FLOPs not latency?**
   Hint: memory bandwidth, operator support, utilization, overhead.
2. **What is the difference between PTQ and QAT?**
   Hint: calibration vs training with simulated quantization.
3. **How do you choose a calibration set?**
   Hint: representative, subgroup-aware, not test.
4. **What is per-channel quantization?**
   Hint: separate scale per output channel.
5. **Why can pruning fail to speed up inference?**
   Hint: hardware lacks sparse kernels.
6. **What is knowledge distillation?**
   Hint: teacher soft targets to student.
7. **How do you evaluate a quantized medical model?**
   Hint: accuracy, calibration, subgroup, target latency, safety.
8. **What is operator fallback?**
   Hint: unsupported op runs on CPU, causing latency spikes.
9. **How do you update a model on a device?**
   Hint: signed bundle, A/B, rollback, PCCP.
10. **What is the difference between weight-only and weight+activation quantization?**
    Hint: memory vs compute; accuracy trade-off.
11. **Why measure p99 latency?**
    Hint: tail latency affects clinical deadlines.
12. **When would you reject a compressed model?**
    Hint: unacceptable subgroup harm, calibration shift, latency, regulatory path.

## 21. Key terms

| Term | Definition |
|---|---|
| Distillation | Training a smaller student from a teacher |
| Teacher/student | Large source model / compact target model |
| Temperature | Softening parameter for distillation |
| PTQ | Post-training quantization |
| QAT | Quantization-aware training |
| Calibration | Choosing quantization ranges from data |
| Per-tensor | One scale for a tensor |
| Per-channel | One scale per channel |
| Zero point | Offset for asymmetric quantization |
| STE | Straight-through estimator |
| Operator fallback | Unsupported op runs on another processor |
| Pruning | Removing weights/channels |
| Structured pruning | Removing whole channels/blocks |
| N:M sparsity | N nonzeros per M block |
| KV cache | Stored attention keys/values for LLM decoding |
| Speculative decoding | Draft model proposes, large model verifies |
| Roofline | Compute vs memory-bandwidth limit |
| p99 latency | 99th percentile latency |
| HIL | Hardware-in-the-loop |
| PCCP | Predetermined change control plan |

## 22. Sources and further reading

### Distillation
- Hinton, G., Vinyals, O., Dean, J., "Distilling the Knowledge in a Neural Network," 2015. https://arxiv.org/abs/1503.02531
- Sanh, V. et al., "DistilBERT," 2019. https://arxiv.org/abs/1910.01108
- Jiao, X. et al., "TinyBERT," 2019. https://arxiv.org/abs/1909.10351
- Sun, Z. et al., "MobileBERT," 2020. https://arxiv.org/abs/2004.02984

### Quantization
- Jacob, B. et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," 2017. https://arxiv.org/abs/1712.05877
- Krishnamoorthi, R., "Quantizing deep convolutional networks for efficient inference," 2018. https://arxiv.org/abs/1806.08342
- Dettmers, T. et al., "LLM.int8()," 2022. https://arxiv.org/abs/2208.07339
- Frantar, E. et al., "GPTQ," 2022. https://arxiv.org/abs/2210.17323
- Lin, J. et al., "AWQ," 2023. https://arxiv.org/abs/2306.00978
- Xiao, G. et al., "SmoothQuant," 2022. https://arxiv.org/abs/2211.10438
- Dettmers, T. et al., "QLoRA," 2023. https://arxiv.org/abs/2305.14314
- Ma, S. et al., "The Era of 1-bit LLMs: BitNet b1.58," 2024. https://arxiv.org/abs/2402.17764

### Pruning and efficiency
- Han, S. et al., "Deep Compression," 2015. https://arxiv.org/abs/1510.00149
- Frankle, J., Carbin, M., "The Lottery Ticket Hypothesis," 2018. https://arxiv.org/abs/1803.03635
- Elsken, T. et al., "Neural Architecture Search: A Survey," 2018. https://arxiv.org/abs/1808.05377

### Runtimes and hardware
- TensorFlow Lite Micro. https://developers.google.com/edge/litert/microcontrollers/overview
- CMSIS-NN. https://github.com/ARM-software/CMSIS-NN
- ONNX Runtime. https://onnxruntime.ai/
- TensorRT. https://developer.nvidia.com/tensorrt
- OpenVINO. https://docs.openvino.ai/
- ExecuTorch. https://docs.pytorch.org/executorch/
- Apache TVM. https://tvm.apache.org/
- llama.cpp. https://github.com/ggml-org/llama.cpp
- MLC-LLM. https://llm.mlc.ai/
- NVIDIA Jetson Thor. https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/
- Arm Ethos-U. https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85
- Hailo. https://hailo.ai/
- Google Coral. https://coral.ai/

### Standards and regulation
- IEC 62304. https://www.iec.ch/
- IEC 60601-1. https://www.iec.ch/
- ISO 14971. https://www.iso.org/standard/72704.html
- IEC 62366-1. https://www.iec.ch/
- FDA, *PCCP guidance*. https://www.fda.gov/media/166704/download
- FDA, *GMLP*. https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
- FDA, *Cybersecurity*. https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity
- EU AI Act. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- EU MDR. https://eur-lex.europa.eu/eli/reg/2017/745/oj

> **Next:** [Module 15](15_deployment_mlops.md) moves from efficient models to production deployment and monitoring.
