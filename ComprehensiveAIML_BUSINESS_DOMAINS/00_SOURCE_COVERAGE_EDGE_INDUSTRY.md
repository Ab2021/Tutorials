# Biomedical edge, streaming, insurance and industry — source coverage

**State:** planned topic architecture and future-content briefs. No full lesson, implemented system, measured outcome, completed issue entry or reviewed interview answer is claimed. Source inventory reviewed for planning only.

## Scope and authoring authority

This plan assigns **245 exact visible source entries** from the first fenced tree in the [source audit](../Templates/archive/TRAIN_DIRECTORY_STRUCTURE_AUDIT_56_IMAGES.md), matching the supplied tree. It extends the local course indexes rather than replacing their existing topics. Source contents were not supplied; names establish syllabus scope, not technical correctness or source access.

Apply the [course authoring and numerical standards](00_AUTHORING_AND_NUMERICAL_STANDARDS.md), [case master](../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md), and [cheat-sheet master](../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). Future cases require 15 sections and a seven-section collection INDEX; future sheets require 12 sections and an 11-section collection INDEX. Each substantive equation needs validity, derivation and FORMULA → WHERE → CONSTRAINT → NUMERICAL EXAMPLE → INTERPRETATION, with units and a counterexample. Use a precise invariant/trace where a closed-form equation is not meaningful.

Each visible technical title in the registers below is a **separate focused-topic minimum**. Split independent methods into focused sheets; reuse a genuine equivalent sheet rather than duplicating it. A group-level exercise does not replace the topic-specific derivations, numerical examples, four diagnostic failures, checked interview answers, source ledger and final card required when each sheet is authored. Each distinct source case remains a separately framed case, not an oversized anthology.

All exercise inputs are **ILLUSTRATIVE** and all validation protocols are **proposed, not executed**. No named company, harness, framework, device, regulator or frontier model behavior is certified by a filename. Pin primary sources, versions, geography and dates before attributing architecture, rules or results. Existing 50/100 Common Issues titles remain targets; none are counted as completed.

Hidden folders and clipped filenames remain `unresolved-source`; visible prefixes and child titles may support limited topic planning but their full path/content is not invented. Ancillary READMEs, source PDFs, collators, archives and generic exercises are tracked without counting them as authored lessons. The permanent [source crosswalk](00_SOURCE_TOPIC_CROSSWALK.md) records every node and disposition.

## Focused teaching groups

- [Sampling, ECG anatomy, filtering, QRS/HRV, PPG/SpO2, FFT/wavelets, IMU and biomedical datasets](#edge-signals)
- [1D CNN, causal dilation, depthwise networks, BiLSTM attention, PatchTST, TimesNet/TSMixer, augmentation, focal loss and TinyECGNet](#edge-biosignal-learning)
- [INT8/INT4, PTQ/QAT, per-channel scales, AWQ/SmoothQuant, structured/unstructured pruning, distillation, mixed precision and NAS](#edge-compression)
- [TFLite Micro/CMSIS-NN, ONNX Runtime, TVM/AutoTVM/Ansor, llama.cpp/GGUF, Edge Impulse, model conversion and runtime gaps](#edge-runtimes)
- [Cortex-M4F device constraints, STM32F401RE, ADC/DMA, FreeRTOS, CMSIS-DSP/NN, UART, power, Cortex-A76/NEON and debugging](#edge-arm-firmware)
- [MLIR, TVM Relay/TIR, graph/loop optimization, microTVM/AOT, TOSA/IREE, RISC-V/RVV, quantization compilation and pass debugging](#edge-compilers)
- [On-device SLM/LLM, GGUF, llama.cpp, Gemma3, KV cache, FAISS retrieval, medical RAG, QLoRA, ONNX embeddings and sovereign deployment](#edge-edge-llm)
- [STM32–Pi5 UART, preprocessing/inference, serial service, RAG, llama-cpp Python, Streamlit, HIL tests, profiling, portfolio and OSS roadmap](#edge-capstone)
- [HighEnd source outlines, study/career schedules, clipped overview files and duplicate compression directory](#edge-source)
- [Flink CEP/broadcast fraud, IoT session windows and financial reconciliation exactly-once](#flink)
- [Multi-tenant platform, Debezium CDC and tiered/infinite-retention design](#kafka)
- [Event-sourced CQRS orders, ecommerce clickstream analytics and SIEM/log aggregation](#stream-design)
- [Hot-partition consumer lag and Flink checkpoint failure/OOM incidents](#stream-incidents)
- [Streaming framework, HLD/LLD and fixes/tradeoffs source documents](#stream-framework)
- [Healthcare payer research: landscape tensions, UHG/Optum, CVS, FWA, pharmacy/member experience, government, SDOH and value-based care](#deep-payer)
- [Chubb P&C, other major insurers, climate risk, document intelligence and insurance AI research](#deep-insurance)
- [Agentic/autonomous frontier, federated learning, climate, genomics and governance](#deep-frontier)
- [Speech-to-text/TTS in insurance and healthcare](#deep-speech)
- [P&C computer vision, medical imaging, clinical vision and multimodal documents](#deep-vision)
- [Pharma manufacturing, quality and supply-chain operations](#pharma-operations)
- [Clinical trials, R&D operations, discovery, RWE, biostatistics and pharmacovigilance](#pharma-clinical)
- [Field force, medical affairs, market access and patient services](#pharma-commercial)
- [Pharma landscape, cross-domain architecture, issues and implemented RAG/agent case research](#pharma-patterns)
- [Insurance actuarial probability, survival, pricing/GLM, reserving, reinsurance, EVT and Bayesian credibility](#insurance-actuarial)
- [Insurance course outlines, 180-day curriculum, Werner–Modlin source, governance, customer analytics and MLOps](#insurance-course)
- [Collapsed GLCC, learn, raw and speech source folders](#unknown-sources)

<a id="edge-signals"></a>

## Sampling, ECG anatomy, filtering, QRS/HRV, PPG/SpO2, FFT/wavelets, IMU and biomedical datasets

**Teaching home:** [07_edge_ai_robotics_and_cyber_physical_systems/37_sensor_fusion_iot_and_realtime_perception/INDEX.md](07_edge_ai_robotics_and_cyber_physical_systems/37_sensor_fusion_iot_and_realtime_perception/INDEX.md). **Status:** planned.

**Testable outcome.** Design a timestamped biomedical preprocessing and feature pipeline without claiming clinical validity.

**Focused-sheet derivations and mechanisms.** Derive Nyquist/aliasing assumptions, IIR/FIR/SOS filtering, Pan–Tompkins detection, HRV time/frequency/nonlinear measures, spectral estimates and ratio-of-ratios calibration; distinguish LF/HF measures from unsupported autonomic conclusions.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 360 Hz ECG sampled for 10 s gives 3600 points; compute Nyquist and 10 s DFT bin spacing; RR intervals 800, 810, 790 ms support a hand RMSSD calculation. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose filter and signal-quality gates before classifying ECG or computing HRV on an STM32.

**Proposed validation.** Compare waveform/peak oracles and patient-separated MIT-BIH/PTB-XL/CPSC/PhysioNet evaluations; test aliasing, motion, mains, baseline drift and timestamp gaps.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Aliasing; wrong filter units/state; motion-corrupted peaks; short/invalid HRV segment. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets` | Source scope/container; not a completed lesson | `source-container` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/01_Sampling_Theory_Nyquist_Shannon_Aliasing_AntiAliasing_ECG360Hz.md` | Sampling Theory Nyquist Shannon Aliasing AntiAliasing ECG360Hz | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/02_ECG_Anatomy_PWave_QRS_TWave_LeadSystems_Clinical_AFib_Detection.md` | ECG Anatomy PWave QRS TWave LeadSystems Clinical AFib Detection | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/03_Butterworth_Bandpass_Filter_Design_0p5to40Hz_ECG_Signal_Processing.md` | Butterworth Bandpass Filter Design 0p5to40Hz ECG Signal Processing | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/04_IIR_vs_FIR_Filters_SOS_Notation_50Hz_Notch_India_Powerline_Interference.md` | IIR vs FIR Filters SOS Notation 50Hz Notch India Powerline Interference | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/05_Pan_Tompkins_Algorithm_RPeak_Detection_Implementation_MIT_BIH.md` | Pan Tompkins Algorithm RPeak Detection Implementation MIT BIH | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/06_HRV_Time_Domain_SDNN_RMSSD_pNN50_Clinical_ANS_STM32.md` | HRV Time Domain SDNN RMSSD pNN50 Clinical ANS STM32 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/07_HRV_Frequency_Domain_LF_HF_Welch_PSD_ANS_Respiratory_Coupling.md` | HRV Frequency Domain LF HF Welch PSD ANS Respiratory Coupling | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/08_HRV_Nonlinear_Analysis_Poincare_SampleEntropy_DFA_Complexity.md` | HRV Nonlinear Analysis Poincare SampleEntropy DFA Complexity | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/09_PPG_Signal_SpO2_RatioOfRatios_MAX30102_Waveform_Analysis.md` | PPG Signal SpO2 RatioOfRatios MAX30102 Waveform Analysis | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/10_FFT_DFT_Windowing_SpectralLeakage_ECG_Frequency_Analysis.md` | FFT DFT Windowing SpectralLeakage ECG Frequency Analysis | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/11_STFT_Wavelet_Transform_TimeFrequency_ECG_AFib_Analysis.md` | STFT Wavelet Transform TimeFrequency ECG AFib Analysis | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/12_IMU_Signal_Processing_Accelerometer_Gyroscope_ComplementaryFilter_Fall.md` | IMU Signal Processing Accelerometer Gyroscope ComplementaryFilter Fall | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/13_Noise_Sources_BaselineWander_MotionArtifact_EMG_Removal_STM32.md` | Noise Sources BaselineWander MotionArtifact EMG Removal STM32 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/14_Datasets_MIT_BIH_PTBXL_CPSC2018_PhysioNet_Download_Guide.md` | Datasets MIT BIH PTBXL CPSC2018 PhysioNet Download Guide | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/15_Signal_To_ML_Feature_Engineering_Edge_Preprocessing_Pipeline.md` | Signal To ML Feature Engineering Edge Preprocessing Pipeline | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_HRV_FFT_Wavelets_FiltersDatasets/README.md` | README | `supporting-artifact` |

<a id="edge-biosignal-learning"></a>

## 1D CNN, causal dilation, depthwise networks, BiLSTM attention, PatchTST, TimesNet/TSMixer, augmentation, focal loss and TinyECGNet

**Teaching home:** [04_deep_learning_generative_ai_and_multimodal_systems/17_deep_learning_computer_vision_and_representation_learning/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/17_deep_learning_computer_vision_and_representation_learning/INDEX.md). **Status:** planned.

**Testable outcome.** Select and evaluate a device-feasible time-series classifier with leakage-safe patient splits.

**Focused-sheet derivations and mechanisms.** Derive receptive field, parameter/activation counts, attention/state costs, focal loss and sensitivity/specificity; distinguish future-aware BiLSTM from causal streaming.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Three kernel-3 stride-1 dilation-1/2/4 convolutions: compute receptive field; compare 90/100 sensitivity with 180/200 specificity and an explicit memory budget. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose TinyECGNet-style model versus larger sequence model for an ECG-to-STM32 prototype.

**Proposed validation.** Patient/time splits, augmentation only on training data, quantized accuracy, inference-cycle budgets and external dataset shift; no fictional benchmarks.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Patient leakage; future context in online path; augmentation corruption; model exceeds MCU memory. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…` | OPEN — Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/  [TRUNCATED/UNVERIFIED] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/01_From_NLP_To_Biosignals_TransferLearning_Analogies_TemporalPatterns.md` | From NLP To Biosignals TransferLearning Analogies TemporalPatterns | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/02_1DCNN_ECG_Kernel_Design_ReceptiveField_LocalMotif_Detection.md` | DCNN ECG Kernel Design ReceptiveField LocalMotif Detection | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/03_DilatedCausalConvolutions_WaveNet_Temporal_ECG_LongRange_Dependencies.md` | DilatedCausalConvolutions WaveNet Temporal ECG LongRange Dependencies | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/04_DepthwiseSeparable_MobileNet_Style_Efficient_ECG_Classification.md` | DepthwiseSeparable MobileNet Style Efficient ECG Classification | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/05_BiLSTM_Attention_Gates_ECG_Sequence_Modeling_HiddenState.md` | BiLSTM Attention Gates ECG Sequence Modeling HiddenState | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/06_PatchTST_Patch_Tokenization_TimeSeries_Transformers_ECG.md` | PatchTST Patch Tokenization TimeSeries Transformers ECG | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/07_TimesNet_TSMixer_MLPMixer_Modern_TimeSeries_Architectures.md` | TimesNet TSMixer MLPMixer Modern TimeSeries Architectures | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/08_HardwareFirst_MCU_Parameter_Budget_INT8_Friendly_Architecture_Design.md` | HardwareFirst MCU Parameter Budget INT8 Friendly Architecture Design | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/09_Signal_Augmentation_TimeWarping_MagnitudeScaling_ECG_Training.md` | Signal Augmentation TimeWarping MagnitudeScaling ECG Training | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/10_Focal_Loss_Class_Imbalance_Medical_ECG_Arrhythmia_Detection.md` | Focal Loss Class Imbalance Medical ECG Arrhythmia Detection | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/11_Training_Strategy_LR_Warmup_Cosine_Annealing_EarlyStopping_ECG.md` | Training Strategy LR Warmup Cosine Annealing EarlyStopping ECG | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/12_Medical_Evaluation_Metrics_AUC_Sensitivity_Specificity_ECG_Arrhythmia.md` | Medical Evaluation Metrics AUC Sensitivity Specificity ECG Arrhythmia | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/13_TinyECGNet_Architecture_Specification_Benchmarks_STM32_Deployment.md` | TinyECGNet Architecture Specification Benchmarks STM32 Deployment | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/14_EndToEnd_ECG_Training_Pipeline_MIT_BIH_CPSC2018_to_STM32.md` | EndToEnd ECG Training Pipeline MIT BIH CPSC2018 to STM32 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module02_DeepLearningTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_FocalL…/README.md` | README | `supporting-artifact` |

<a id="edge-compression"></a>

## INT8/INT4, PTQ/QAT, per-channel scales, AWQ/SmoothQuant, structured/unstructured pruning, distillation, mixed precision and NAS

**Teaching home:** [07_edge_ai_robotics_and_cyber_physical_systems/36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md](07_edge_ai_robotics_and_cyber_physical_systems/36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md). **Status:** planned.

**Testable outcome.** Construct a supported accuracy-size-latency-energy compression comparison across MCU and larger-model contexts.

**Focused-sheet derivations and mechanisms.** Derive affine quantization/dequantization and error, calibration assumptions, straight-through gradients, distillation loss and sparsity accounting; AWQ/SmoothQuant are not automatically MCU CNN solutions.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Weights -1,0,1 use symmetric signed scale 1/127: compute integer/reconstructed values and rounding error; compare 100000 parameters in FP32 versus INT8 excluding overhead. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose PTQ, QAT or a smaller architecture when a biomedical model violates flash/SRAM constraints.

**Proposed validation.** Hold calibration apart from test, measure per-slice quality, operator support, actual bytes, latency and energy; test recurrent state precision and kernel-effective sparsity.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unrepresentative calibration; unsupported INT4 kernel; unstructured sparsity without acceleration; recurrent-state error growth. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS` | Source scope/container; not a completed lesson | `source-container` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/01_Quantization_Fundamentals_INT8_INT4_Scale_ZeroPoint_Symmetric_Asymmetric.md` | Quantization Fundamentals INT8 INT4 Scale ZeroPoint Symmetric Asymmetric | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/02_PostTraining_Quantization_PTQ_Calibration_MinMax_Histogram_Entropy.md` | PostTraining Quantization PTQ Calibration MinMax Histogram Entropy | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/03_Quantization_Aware_Training_QAT_FakeQuantization_StraightThrough_Estimator.md` | Quantization Aware Training QAT FakeQuantization StraightThrough Estimator | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/04_PerChannel_vs_PerTensor_Quantization_Accuracy_CNN_Implications.md` | PerChannel vs PerTensor Quantization Accuracy CNN Implications | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/05_AWQ_Activation_Aware_Weight_Quantization_LLM_Theory.md` | AWQ Activation Aware Weight Quantization LLM Theory | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/06_SmoothQuant_Quantization_Difficulty_Migration_LLM_Activation_Outliers.md` | SmoothQuant Quantization Difficulty Migration LLM Activation Outliers | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/07_Unstructured_Pruning_Weight_Magnitude_GMP_Lottery_Ticket.md` | Unstructured Pruning Weight Magnitude GMP Lottery Ticket | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/08_Structured_Pruning_Channel_Filter_Hardware_Friendly_Sparsity.md` | Structured Pruning Channel Filter Hardware Friendly Sparsity | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/09_Knowledge_Distillation_Teacher_Student_SoftTargets_FeatureMimicking.md` | Knowledge Distillation Teacher Student SoftTargets FeatureMimicking | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/10_Mixed_Precision_Quantization_INT8_INT4_FP16_Sensitivity_Analysis.md` | Mixed Precision Quantization INT8 INT4 FP16 Sensitivity Analysis | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/11_Quantization_Recurrent_Networks_LSTM_INT8_State_Quantization_Challenges.md` | Quantization Recurrent Networks LSTM INT8 State Quantization Challenges | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/12_Neural_Architecture_Search_NAS_Hardware_Aware_OnceForAll_ProxylessNAS.md` | Neural Architecture Search NAS Hardware Aware OnceForAll ProxylessNAS | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/13_Compression_Pipeline_PTQ_Pruning_Distillation_Combined_Strategy.md` | Compression Pipeline PTQ Pruning Distillation Combined Strategy | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/14_Benchmarking_Compressed_Models_Accuracy_Size_Latency_Energy.md` | Benchmarking Compressed Models Accuracy Size Latency Energy | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/15_Hero_Project_Compression_Pipeline_TinyECGNet_Float32_to_STM32_INT8.md` | Hero Project Compression Pipeline TinyECGNet Float32 to STM32 INT8 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Pruning_Distillation_NAS/README.md` | README | `supporting-artifact` |

<a id="edge-runtimes"></a>

## TFLite Micro/CMSIS-NN, ONNX Runtime, TVM/AutoTVM/Ansor, llama.cpp/GGUF, Edge Impulse, model conversion and runtime gaps

**Teaching home:** [07_edge_ai_robotics_and_cyber_physical_systems/36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md](07_edge_ai_robotics_and_cyber_physical_systems/36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md). **Status:** planned.

**Testable outcome.** Select a supported runtime and conversion path for STM32 versus Pi5 without hiding unsupported operators behind a fallback.

**Focused-sheet derivations and mechanisms.** Derive tensor arena peak liveness, operator coverage, layout/dtype conversion and cross-runtime equivalence; distinguish legacy/current compiler APIs with pinned versions.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Assume 96 KiB usable SRAM, 20 KiB stack, 16 KiB buffers and 50 KiB arena: compute headroom and reject 64 KiB replacement arena. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose TFLite Micro versus generated AOT C on MCU and ONNX/llama.cpp on Pi5 for supported models.

**Proposed validation.** Convert a fixed model, compare numerical tolerances, inspect operator resolution and measure peak memory/cycles/energy on actual target when available.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unsupported op; conversion/layout drift; arena overflow; hidden slow fallback. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse` | Source scope/container; not a completed lesson | `source-container` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/01_TFLite_Micro_Runtime_Architecture_Memory_Arena_Interpreter_Lifecycle.md` | TFLite Micro Runtime Architecture Memory Arena Interpreter Lifecycle | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/02_TFLite_Micro_Operator_Kernels_CMSIS_NN_Delegate_Quantized_Inference_STM32.md` | TFLite Micro Operator Kernels CMSIS NN Delegate Quantized Inference STM32 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/03_ONNX_Runtime_Raspberry_Pi5_Execution_Providers_XNNPACK_MLAS_Inference.md` | ONNX Runtime Raspberry Pi5 Execution Providers XNNPACK MLAS Inference | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/04_ONNX_Runtime_Graph_Optimization_Operator_Fusion_Quantization_Aware_Inference.md` | ONNX Runtime Graph Optimization Operator Fusion Quantization Aware Inference | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/05_Apache_TVM_Compilation_Stack_Relay_IR_Graph_Lowering_Cortex_M_Target.md` | Apache TVM Compilation Stack Relay IR Graph Lowering Cortex M Target | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/06_TVM_AutoTVM_Ansor_Autotuning_ARM_Cortex_Search_Space_Hardware_Aware_Optimization.md` | TVM AutoTVM Ansor Autotuning ARM Cortex Search Space Hardware Aware Optimization | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/07_llamacpp_Architecture_GGUF_Format_Quantized_LLM_Inference_Raspberry_Pi5.md` | llamacpp Architecture GGUF Format Quantized LLM Inference Raspberry Pi5 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/08_llamacpp_KV_Cache_Context_Management_Sampling_Medical_RAG_Prompting_Pi5.md` | llamacpp KV Cache Context Management Sampling Medical RAG Prompting Pi5 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/09_Edge_Impulse_SDK_Data_Collection_DSP_Pipeline_Neural_Network_Deployment.md` | Edge Impulse SDK Data Collection DSP Pipeline Neural Network Deployment | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/10_Edge_Impulse_EON_Compiler_STM32_Deployment_UART_Data_Collection_OSS_Contribution.md` | Edge Impulse EON Compiler STM32 Deployment UART Data Collection OSS Contribution | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/11_Model_Format_Conversion_PyTorch_ONNX_TFLite_CoreML_GGUF_Interoperability.md` | Model Format Conversion PyTorch ONNX TFLite CoreML GGUF Interoperability | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/12_Runtime_Operator_Coverage_Custom_Ops_Fallback_Strategies_Deployment_Gaps.md` | Runtime Operator Coverage Custom Ops Fallback Strategies Deployment Gaps | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/13_Cross_Runtime_Benchmarking_Latency_Memory_Power_Profiling_STM32_Pi5.md` | Cross Runtime Benchmarking Latency Memory Power Profiling STM32 Pi5 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/14_Hero_Project_Runtime_Integration_TFLite_Micro_STM32_ONNX_llamacpp_Pi5_UART_Pipeline.md` | Hero Project Runtime Integration TFLite Micro STM32 ONNX llamacpp Pi5 UART Pipeline | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_TVM_llamacpp_EdgeImpulse/README.md` | README | `supporting-artifact` |

<a id="edge-arm-firmware"></a>

## Cortex-M4F device constraints, STM32F401RE, ADC/DMA, FreeRTOS, CMSIS-DSP/NN, UART, power, Cortex-A76/NEON and debugging

**Teaching home:** [07_edge_ai_robotics_and_cyber_physical_systems/35_edge_ai_embedded_systems_and_tinyml/INDEX.md](07_edge_ai_robotics_and_cyber_physical_systems/35_edge_ai_embedded_systems_and_tinyml/INDEX.md). **Status:** planned.

**Testable outcome.** Specify interrupt/task/memory ownership and a firmware schedule that respects measured hardware budgets.

**Focused-sheet derivations and mechanisms.** Derive sample-buffer ownership, UART framing/serialization time, priority inversion constraints, linker layout and power-state energy; use actual device documentation rather than mixing MCU variants.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 360 samples/s at 2 bytes/sample, 20-byte header each 100 samples: calculate payload/framing bytes/s; 115200-baud 8N1 gives 11520 bytes/s upper serial payload rate before software overhead. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose DMA/buffer/task priorities for continuous ECG inference and UART transfer to Pi5.

**Proposed validation.** Inspect exact MCU pin/memory/ADC capabilities; stress DMA wrap, queue pressure, task priority, low-power wake and stack/arena collision using SWD/JTAG/ITM where supported.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: DMA race; priority inversion; UART frame loss; linker/stack collision. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32` | Source scope/container; not a completed lesson | `source-container` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/01_ARM_CortexM4F_Pipeline_ISA_Thumb2_DSP_FPU_Architecture_Deep_Dive.md` | ARM CortexM4F Pipeline ISA Thumb2 DSP FPU Architecture Deep Dive | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/02_STM32F401RE_Memory_Map_Registers_HAL_Abstraction_Peripheral_Access.md` | STM32F401RE Memory Map Registers HAL Abstraction Peripheral Access | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/03_DMA_Controller_ADC_ECG_Sampling_Circular_Buffer_Interrupt_STM32.md` | DMA Controller ADC ECG Sampling Circular Buffer Interrupt STM32 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/04_FreeRTOS_Task_Scheduling_Priority_Inversion_Mutexes_Semaphores_STM32.md` | FreeRTOS Task Scheduling Priority Inversion Mutexes Semaphores STM32 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/05_CMSIS_DSP_Library_FIR_IIR_FFT_ARM_Math_ECG_Signal_Processing_STM32.md` | CMSIS DSP Library FIR IIR FFT ARM Math ECG Signal Processing STM32 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/06_STM32_UART_DMA_Communication_Protocol_Pi5_UART_Bridge_Frame_Design.md` | STM32 UART DMA Communication Protocol Pi5 UART Bridge Frame Design | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/07_STM32_Power_Modes_Sleep_Stop_Standby_Wake_On_Event_Battery_Optimization.md` | STM32 Power Modes Sleep Stop Standby Wake On Event Battery Optimization | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/08_ARM_CortexA76_Architecture_OutOfOrder_NEON_SIMD_Cache_Hierarchy_Pi5.md` | ARM CortexA76 Architecture OutOfOrder NEON SIMD Cache Hierarchy Pi5 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/09_MCU_vs_MPU_Programming_Models_Bare_Metal_RTOS_Linux_Memory_Protection.md` | MCU vs MPU Programming Models Bare Metal RTOS Linux Memory Protection | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/10_STM32CubeIDE_Development_Workflow_Code_Generation_Build_System_Debug.md` | STM32CubeIDE Development Workflow Code Generation Build System Debug | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/11_SWD_JTAG_Debugging_Segger_JLink_ITM_Trace_Semihosting_STM32.md` | SWD JTAG Debugging Segger JLink ITM Trace Semihosting STM32 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/12_Linker_Scripts_Memory_Layout_Flash_SRAM_Sections_STM32_TFLite_Arena.md` | Linker Scripts Memory Layout Flash SRAM Sections STM32 TFLite Arena | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/13_STM32_HAL_Driver_Architecture_GPIO_ADC_UART_Timer_Abstraction_Layers.md` | STM32 HAL Driver Architecture GPIO ADC UART Timer Abstraction Layers | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/14_CMSIS_NN_INT8_Inference_Optimized_Kernels_SMLAD_Quantized_NeuralNetwork_STM32.md` | CMSIS NN INT8 Inference Optimized Kernels SMLAD Quantized NeuralNetwork STM32 | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/15_Hero_Project_Embedded_Firmware_Architecture_FreeRTOS_TFLite_UART_Complete_Integration.md` | Hero Project Embedded Firmware Architecture FreeRTOS TFLite UART Complete Integration | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_FreeRTOS_STM32/README.md` | README | `supporting-artifact` |

<a id="edge-compilers"></a>

## MLIR, TVM Relay/TIR, graph/loop optimization, microTVM/AOT, TOSA/IREE, RISC-V/RVV, quantization compilation and pass debugging

**Teaching home:** [07_edge_ai_robotics_and_cyber_physical_systems/36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md](07_edge_ai_robotics_and_cyber_physical_systems/36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md). **Status:** planned.

**Testable outcome.** Trace a model through version-pinned IR passes and verify correctness before claiming optimization.

**Focused-sheet derivations and mechanisms.** Derive dependence-preserving fusion/tiling, dataflow, liveness/bufferization and quantization layout invariants; distinguish analysis from transformation and ISA from accelerator capability.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Two operators exchange 4096 FP32 values: compute intermediate write-plus-read traffic; compare removed traffic only if fusion preserves values and valid lifetime. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose TVM, MLIR/IREE, XLA or existing runtime based on supported target, operators and measurable bottleneck.

**Proposed validation.** Compare pass-by-pass outputs against reference on edge shapes/dtypes, inspect generated code and profile target execution; benchmark tuning cost separately.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Illegal fusion; wrong buffer lifetime; quantization lowering drift; source/runtime version mismatch. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS` | Source scope/container; not a completed lesson | `source-container` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/01_AI_Compiler_Stack_Overview_Frontend_Midend_Backend_IR_Levels_Why_Compilers_Matter.md` | AI Compiler Stack Overview Frontend Midend Backend IR Levels Why Compilers Matter | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/02_MLIR_Architecture_Multi_Level_IR_Dialects_Passes_Transformations_Design_Philosophy.md` | MLIR Architecture Multi Level IR Dialects Passes Transformations Design Philosophy | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/03_TVM_Architecture_Relay_IR_TIR_Lowering_Pipeline_Schedule_Primitives_Hardware_Targets.md` | TVM Architecture Relay IR TIR Lowering Pipeline Schedule Primitives Hardware Targets | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/04_Graph_Level_Optimizations_Operator_Fusion_Constant_Folding_Dead_Code_Layout_Transform.md` | Graph Level Optimizations Operator Fusion Constant Folding Dead Code Layout Transform | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/05_Loop_Level_Optimizations_Tiling_Vectorization_Loop_Unrolling_Polyhedral_Model_Cache.md` | Loop Level Optimizations Tiling Vectorization Loop Unrolling Polyhedral Model Cache | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/06_AutoTVM_Ansor_Hardware_Aware_Tuning_Search_Space_ML_Cost_Model_Benchmark.md` | AutoTVM Ansor Hardware Aware Tuning Search Space ML Cost Model Benchmark | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/07_microTVM_AOT_Executor_STM32_Deployment_No_Heap_Static_Memory_Generated_C.md` | microTVM AOT Executor STM32 Deployment No Heap Static Memory Generated C | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/08_MLIR_TOSA_Dialect_Edge_ML_TFLite_Lowering_IREE_Compiler_Edge_Deployment.md` | MLIR TOSA Dialect Edge ML TFLite Lowering IREE Compiler Edge Deployment | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/09_RISC_V_ISA_Architecture_Vector_Extensions_RVV_Edge_ML_Accelerator_Future.md` | RISC V ISA Architecture Vector Extensions RVV Edge ML Accelerator Future | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/10_Compiler_Passes_Analysis_vs_Transformation_Dataflow_Analysis_IR_Correctness.md` | Compiler Passes Analysis vs Transformation Dataflow Analysis IR Correctness | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/11_Quantization_Aware_Compilation_INT8_Flow_TVM_Relay_Quantize_CMSIS_NN_Backend.md` | Quantization Aware Compilation INT8 Flow TVM Relay Quantize CMSIS NN Backend | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/12_MLIR_Linalg_Dialect_Structured_Ops_Fusion_Tiling_Bufferization_Lowering.md` | MLIR Linalg Dialect Structured Ops Fusion Tiling Bufferization Lowering | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/13_TVM_vs_MLIR_vs_XLA_vs_IREE_Compiler_Comparison_Tradeoffs_When_To_Use_Each.md` | TVM vs MLIR vs XLA vs IREE Compiler Comparison Tradeoffs When To Use Each | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/14_OSS_Contribution_Strategy_TVM_MLIR_IREE_microTVM_PR_Guide_Issue_Triage_Community.md` | OSS Contribution Strategy TVM MLIR IREE microTVM PR Guide Issue Triage Community | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/15_Compiler_IR_Debugging_Tools_mlir_opt_tvmc_Visualization_Pass_Tracing_Correctness.md` | Compiler IR Debugging Tools mlir opt tvmc Visualization Pass Tracing Correctness | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/16_Hero_Project_Compiler_Pipeline_TinyECGNet_TFLite_to_C_AOT_STM32_Pi5_Full_Stack.md` | Hero Project Compiler Pipeline TinyECGNet TFLite to C AOT STM32 Pi5 Full Stack | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module06_AICompilerFundamentals_MLIR_TVM_RISCV_GraphOpt_OperatorFusion_OSS/README.md` | README | `supporting-artifact` |

<a id="edge-edge-llm"></a>

## On-device SLM/LLM, GGUF, llama.cpp, Gemma3, KV cache, FAISS retrieval, medical RAG, QLoRA, ONNX embeddings and sovereign deployment

**Teaching home:** [05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md). **Status:** planned.

**Testable outcome.** Size and evaluate an offline Pi5 assistant with explicit data boundaries, evidence and abstention.

**Focused-sheet derivations and mechanisms.** Derive model/adapter/KV/index memory, FAISS IVF/PQ/HNSW tradeoffs, RoPE/GQA/RMSNorm where appropriate, retrieval eligibility and cost per useful local answer.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Assume 3 GiB model, 1 GiB KV, 0.5 GiB index and 1 GiB OS/other on 8 GiB device: compute illustrative headroom, then vary KV to 4 GiB. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose small local model plus authorized RAG versus remote service under privacy, quality and latency constraints.

**Proposed validation.** Measure actual tokens/s, first-token delay, RSS, power and thermal behavior; audit retrieval/medical evidence, voice/UART integration and cloud-free operation; no clinical efficacy inferred.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: RAM exhaustion; stale medical corpus; unintended egress; thermal slowdown. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment` | Source scope/container; not a completed lesson | `source-container` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/01_OnDevice_LLM_Landscape_Why_Edge_LLM_Matters_SLM_vs_LLM_Sovereign_AI_Architecture.md` | OnDevice LLM Landscape Why Edge LLM Matters SLM vs LLM Sovereign AI Architecture | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/02_GGUF_Format_Deep_Dive_Quantization_Types_Metadata_Tensors_KV_Cache_Memory_Layout.md` | GGUF Format Deep Dive Quantization Types Metadata Tensors KV Cache Memory Layout | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/03_llamacpp_Architecture_Threading_Metal_BLAS_Context_Params_Sampling_Inference_Loop.md` | llamacpp Architecture Threading Metal BLAS Context Params Sampling Inference Loop | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/04_Gemma3_Architecture_RoPE_GQA_RMSNorm_Attention_Layers_Embedding_Vocab_Edge_Inference.md` | Gemma3 Architecture RoPE GQA RMSNorm Attention Layers Embedding Vocab Edge Inference | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/05_KV_Cache_Memory_Arena_Attention_Computation_Pi5_NEON_Optimization_Throughput_Analysis.md` | KV Cache Memory Arena Attention Computation Pi5 NEON Optimization Throughput Analysis | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/06_FAISS_Vector_Database_IVF_PQ_HNSW_Index_Construction_Medical_Knowledge_Embeddings.md` | FAISS Vector Database IVF PQ HNSW Index Construction Medical Knowledge Embeddings | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/07_Medical_RAG_Pipeline_Design_Query_Embedding_Retrieval_Context_Injection_Prompt_Engineering.md` | Medical RAG Pipeline Design Query Embedding Retrieval Context Injection Prompt Engineering | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/08_QLoRA_Theory_NF4_Quantization_LoRA_Adapters_Rank_Alpha_Gradient_Checkpointing_Memory.md` | QLoRA Theory NF4 Quantization LoRA Adapters Rank Alpha Gradient Checkpointing Memory | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/09_Fine_Tuning_Medical_LLM_Dataset_Preparation_Instruction_Tuning_RLHF_Alignment_Eval.md` | Fine Tuning Medical LLM Dataset Preparation Instruction Tuning RLHF Alignment Eval | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/10_Streamlit_UI_Architecture_UART_Integration_Real_Time_ECG_Display_LLM_Response_Streaming.md` | Streamlit UI Architecture UART Integration Real Time ECG Display LLM Response Streaming | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/11_Sovereign_AI_Architecture_Privacy_By_Design_No_Cloud_Dependency_Data_Governance_Edge.md` | Sovereign AI Architecture Privacy By Design No Cloud Dependency Data Governance Edge | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/12_Edge_Inference_Optimization_Speculative_Decoding_Context_Caching_Batching_Pi5_Limits.md` | Edge Inference Optimization Speculative Decoding Context Caching Batching Pi5 Limits | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/13_ONNX_Runtime_Pi5_Deployment_BERT_Embeddings_Sentence_Transformers_Quantized_Inference.md` | ONNX Runtime Pi5 Deployment BERT Embeddings Sentence Transformers Quantized Inference | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/14_Vector_Database_Alternatives_ChromaDB_Weaviate_Qdrant_vs_FAISS_Edge_Tradeoffs_Selection.md` | Vector Database Alternatives ChromaDB Weaviate Qdrant vs FAISS Edge Tradeoffs Selection | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/15_Benchmarking_LLM_Inference_Pi5_Tokens_Per_Second_Memory_Power_Latency_Profiling_Tools.md` | Benchmarking LLM Inference Pi5 Tokens Per Second Memory Power Latency Profiling Tools | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/16_Hero_Project_Complete_Pi5_Stack_UART_STM32_FAISS_Gemma3_Streamlit_End_To_End_Integration.md` | Hero Project Complete Pi5 Stack UART STM32 FAISS Gemma3 Streamlit End To End Integration | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module07_OnDeviceLLM_EdgeRAG_SLM_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deployment/README.md` | README | `supporting-artifact` |

<a id="edge-capstone"></a>

## STM32–Pi5 UART, preprocessing/inference, serial service, RAG, llama-cpp Python, Streamlit, HIL tests, profiling, portfolio and OSS roadmap

**Teaching home:** [10_architecture_practice_and_portfolio_capstones/56_end_to_end_capstones_and_portfolio/INDEX.md](10_architecture_practice_and_portfolio_capstones/56_end_to_end_capstones_and_portfolio/INDEX.md). **Status:** planned.

**Testable outcome.** Integrate an observable end-to-end biomedical demonstration with bounded failures and truthful evidence.

**Focused-sheet derivations and mechanisms.** Derive message version/CRC/sequence/timestamp invariants, queue budgets, cycle-time accounting and owned recovery; distinguish portfolio and career plans from technical proof.

**Numerical/trace exercise to author.** ILLUSTRATIVE: At 360 samples/s, 100-sample frames span 277.78 ms; add illustrative 20 ms inference and 30 ms transport/processing to compute one serial-path budget, not a p99 guarantee. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose architecture and release gates for a local ECG display/assistive explanation demonstration, not clinical diagnosis.

**Proposed validation.** Run recorded serial traces and dropped/corrupt frames, verify signal/model outputs, retrieval citations and UI state; use real HIL for hardware claims; document missing hardware evidence.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Malformed/duplicate serial frame; stale UI state; unreliable clinical interpretation; component timings mistaken for end-to-end tail. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone` | Source scope/container; not a completed lesson | `source-container` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/01_System_Architecture_Overview_STM32_Pi5_UART_Data_Pipeline_Component_Diagram.md` | System Architecture Overview STM32 Pi5 UART Data Pipeline Component Diagram | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/02_STM32_Hardware_Setup_ECG_Circuit_ADS1115_ADC_GPIO_UART_Power_Supply_Schematic.md` | STM32 Hardware Setup ECG Circuit ADS1115 ADC GPIO UART Power Supply Schematic | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/03_STM32_Firmware_Architecture_FreeRTOS_Tasks_ADC_DMA_ECG_Inference_UART_Output.md` | STM32 Firmware Architecture FreeRTOS Tasks ADC DMA ECG Inference UART Output | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/04_TFLite_Micro_Integration_Model_Loading_Arena_Allocation_Operator_Resolver_Invoke.md` | TFLite Micro Integration Model Loading Arena Allocation Operator Resolver Invoke | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/05_Signal_Processing_Pipeline_STM32_Notch_Bandpass_Pan_Tompkins_QRS_Feature_Extraction.md` | Signal Processing Pipeline STM32 Notch Bandpass Pan Tompkins QRS Feature Extraction | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/06_Raspberry_Pi5_OS_Setup_UART_Configuration_Python_Environment_Service_Architecture.md` | Raspberry Pi5 OS Setup UART Configuration Python Environment Service Architecture | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/07_Pi5_Serial_Reader_Service_JSON_Parsing_Validation_Queue_Architecture_Error_Recovery.md` | Pi5 Serial Reader Service JSON Parsing Validation Queue Architecture Error Recovery | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/08_RAG_Pipeline_Implementation_Embedding_FAISS_Retrieval_Context_Prompt_Assembly.md` | RAG Pipeline Implementation Embedding FAISS Retrieval Context Prompt Assembly | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/09_llama_cpp_Python_Integration_Model_Loading_Inference_Streaming_Context_Management.md` | llama cpp Python Integration Model Loading Inference Streaming Context Management | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/10_Streamlit_Application_Complete_Implementation_Layout_State_Charts_Streaming_Deploy.md` | Streamlit Application Complete Implementation Layout State Charts Streaming Deploy | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/11_System_Testing_Integration_Testing_Unit_Testing_Hardware_In_Loop_Validation.md` | System Testing Integration Testing Unit Testing Hardware In Loop Validation | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/12_Performance_Profiling_Bottleneck_Analysis_Memory_CPU_Power_Optimization_Tradeoffs.md` | Performance Profiling Bottleneck Analysis Memory CPU Power Optimization Tradeoffs | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/13_OSS_Contribution_Roadmap_TVM_MLIR_TFLite_llama_cpp_FAISS_Issues_PRs_Community.md` | OSS Contribution Roadmap TVM MLIR TFLite llama cpp FAISS Issues PRs Community | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/14_Portfolio_Project_Documentation_GitHub_README_Architecture_Diagrams_Benchmarks_Demo.md` | Portfolio Project Documentation GitHub README Architecture Diagrams Benchmarks Demo | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/15_Career_Roadmap_OSS_Strategy_Research_Papers_Conference_Talks_Next_18_Months.md` | Career Roadmap OSS Strategy Research Papers Conference Talks Next 18 Months | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/16_Capstone_Summary_Module_Synthesis_Complete_Course_Review_Future_Directions_Sovereign_Edge_AI.md` | Capstone Summary Module Synthesis Complete Course Review Future Directions Sovereign Edge AI | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module08_SystemIntegration_SovereignHealthAI_STM32_Pi5_Streamlit_Capstone/README.md` | README | `supporting-artifact` |

<a id="edge-source"></a>

## HighEnd source outlines, study/career schedules, clipped overview files and duplicate compression directory

**Teaching home:** [10_architecture_practice_and_portfolio_capstones/56_end_to_end_capstones_and_portfolio/INDEX.md](10_architecture_practice_and_portfolio_capstones/56_end_to_end_capstones_and_portfolio/INDEX.md). **Status:** planned.

**Testable outcome.** Track the eight visible module prefixes and plans without reconstructing missing filenames or duplicate directory contents.

**Focused-sheet derivations and mechanisms.** Define source versus course module identities, plan dependencies and milestone acceptance; chronology is not evidence of technical completion.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A study plan has 8 modules and 12 available weeks: compare equal allocation with prerequisite-heavy allocation; show assumptions rather than promising mastery. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose a feasible portfolio scope from available hardware and measurable milestones.

**Proposed validation.** Resolve clipped paths from a filesystem export; inspect the duplicate-looking compression module before merging; verify every milestone through real outputs.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Assumed duplicate equivalence; invented path; schedule mistaken for competence; missing hardware prerequisite. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `HighEnd` | Source scope/container; not a completed lesson | `source-container` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse` | Source scope/container; not a completed lesson | `source-container` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Distillation_NAS` | OPEN — Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_Distillation_NAS/  [contents not visible] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/[VISIBLE PREFIX ONLY] Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_Filtering_HRV_FrequencyDomain_FundamentalsFor…` | OPEN — [VISIBLE PREFIX ONLY] Module01_BiomedicalSignalProcessing_ECG_PPG_IMU_Filtering_HRV_FrequencyDomain_FundamentalsFor…  [extension shown: .md; exact filename incomplete] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/[VISIBLE PREFIX ONLY] Module02_DeepLearningForTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_ClassImbala…` | OPEN — [VISIBLE PREFIX ONLY] Module02_DeepLearningForTimeSeries_1DCNN_BiLSTM_Attention_Transformers_Augmentation_ClassImbala…  [extension shown: .md; exact filename incomplete] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/[VISIBLE PREFIX ONLY] Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_StructuredPruning_KnowledgeDistillation_NAS…` | OPEN — [VISIBLE PREFIX ONLY] Module03_ModelCompression_PTQ_QAT_AWQ_SmoothQuant_StructuredPruning_KnowledgeDistillation_NAS…  [extension shown: .md; exact filename incomplete] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/[VISIBLE PREFIX ONLY] Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_ApacheTVM_AutoTuning_llamacpp_Edge…` | OPEN — [VISIBLE PREFIX ONLY] Module04_EdgeDeploymentRuntimes_TFLiteMicro_ONNXRuntime_ApacheTVM_AutoTuning_llamacpp_Edge…  [extension shown: .md; exact filename incomplete] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/[VISIBLE PREFIX ONLY] Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_NN_MemoryHierarchy_PowerAnal…` | OPEN — [VISIBLE PREFIX ONLY] Module05_ARMArchitecture_EmbeddedSystems_CortexM_CortexA_CMSIS_NN_MemoryHierarchy_PowerAnal…  [extension shown: .md; exact filename incomplete] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/[VISIBLE PREFIX ONLY] Module06_AICompilerFundamentals_MLIR_Dialects_TVMPasses_RISCV_OperatorFusion_GraphOptimization_…` | OPEN — [VISIBLE PREFIX ONLY] Module06_AICompilerFundamentals_MLIR_Dialects_TVMPasses_RISCV_OperatorFusion_GraphOptimization_…  [extension shown: .md; exact filename incomplete] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/[VISIBLE PREFIX ONLY] Module07_OnDeviceLLM_EdgeRAG_SLMFineTuning_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deplo…` | OPEN — [VISIBLE PREFIX ONLY] Module07_OnDeviceLLM_EdgeRAG_SLMFineTuning_QLoRA_GGUF_FAISS_MedicalRAG_SovereignAI_Pi5Deplo…  [extension shown: .md; exact filename incomplete] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/[VISIBLE PREFIX ONLY] Module08_SystemIntegration_SovereignHealthAI_EndToEndArchitecture_STM32_Pi5_Streamlit_Benchmarking…` | OPEN — [VISIBLE PREFIX ONLY] Module08_SystemIntegration_SovereignHealthAI_EndToEndArchitecture_STM32_Pi5_Streamlit_Benchmarking…  [extension shown: .md; exact filename incomplete] | `unresolved-source` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/CourseIndex_Overview_Prerequisites_LearningPath_ModuleSummaries_ResourcesAndTimeline.md` | CourseIndex Overview Prerequisites LearningPath ModuleSummaries ResourcesAndTimeline | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/day_by_day_18mo_plan.md` | day by day 18mo plan | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/master_oss_plan.md` | master oss plan | `planned-topic` |
| `HighEnd/AdvancedCourse_HardwareAwareML_EdgeAI_7Domain_ComprehensiveMasteryCourse/STUDY_GUIDE_DataScientist_3Month_EdgeAI_Roadmap.md` | STUDY GUIDE DataScientist 3Month EdgeAI Roadmap | `planned-topic` |

<a id="flink"></a>

## Flink CEP/broadcast fraud, IoT session windows and financial reconciliation exactly-once

**Teaching home:** [streaming systems module](02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/INDEX.md). **Status:** partial, unreviewed drafts include an [event-time session-window sheet](02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/Cheat_Sheets/03_Flink_Event_Time_Session_Windows_and_Stateful_Processing.md) and a [hypothetical CEP/broadcast fraud case](02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/Case_Studies/02_Flink_CEP_Streaming_Fraud_Detection.md). Neither reconstructs unavailable source content; the IoT and reconciliation cases remain planned.

**Testable outcome.** Choose event-time, state and sink semantics independently for each source case.

**Focused-sheet derivations and mechanisms.** Derive watermark/lateness, keyed/broadcast state, session merging, checkpoints and transaction/idempotency boundaries.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Events at 0,4,12 s with 5 s gap: trace session windows, then add late event at 8 s and reason about merging; 1000 events/s times 60 s recovery implies 60000 events to replay. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** For separate cases choose fraud rule update/CEP behavior, IoT session aggregation, and reconciliation sink correctness under restart.

**Proposed validation.** Test out-of-order/late/duplicate records, broadcast rule versions, checkpoint restore and end-to-end sink reconciliation; exactly-once requires a stated boundary.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Late-event miscount; state explosion; rule-version race; duplicate sink effects. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Flink_Stateful_Processing` | Source scope/container; not a completed lesson | `source-container` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Flink_Stateful_Processing/00_FIXES_AND_TRADEOFFS.md` | FIXES AND TRADEOFFS | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Flink_Stateful_Processing/00_HLD_LLD_SYSTEM_DESIGN.md` | HLD LLD SYSTEM DESIGN | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Flink_Stateful_Processing/01_Real_Time_Fraud_Detection_CEP_Broadcast.md` | Real Time Fraud Detection CEP Broadcast | `partial-draft` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Flink_Stateful_Processing/02_IoT_Telemetry_Analytics_Session_Windows.md` | IoT Telemetry Analytics Session Windows | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Flink_Stateful_Processing/03_Financial_Reconciliation_Exactly_Once.md` | Financial Reconciliation Exactly Once | `planned-topic` |

<a id="kafka"></a>

## Multi-tenant platform, Debezium CDC and tiered/infinite-retention design

**Teaching home:** [02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/INDEX.md](02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/INDEX.md). **Status:** mostly planned; unreviewed partial drafts cover [Kafka consumer lag](02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/Cheat_Sheets/01_Kafka_Consumer_Partitions_and_Lag.md) and [CDC snapshot handoff](02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/Cheat_Sheets/02_CDC_Ordering_Snapshot_and_Materialization.md). They do not verify inaccessible source cases.

**Testable outcome.** Choose partitioning/isolation, snapshot-to-log correctness and bounded retention/economics.

**Focused-sheet derivations and mechanisms.** Derive partition throughput/skew, replication/storage traffic, CDC snapshot/order/schema contracts and tiered retention recovery.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 20000 events/s at 500 bytes for one day: compute uncompressed payload and 3-replica local footprint before compression/index/remote-storage effects. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Design separate multi-tenant admission, CDC transition and tiered-storage cases; compare Kafka/Redpanda through verified versioned semantics.

**Proposed validation.** Replay hot tenants, schema evolution, connector restart/snapshot overlap, broker loss, remote-tier unavailability and consumer rebuild.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Hot partition; CDC duplicate/gap; schema drift; unlimited-retention cost/recovery assumption. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Kafka_Architecture` | Source scope/container; not a completed lesson | `source-container` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Kafka_Architecture/00_FIXES_AND_TRADEOFFS.md` | FIXES AND TRADEOFFS | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Kafka_Architecture/00_HLD_LLD_SYSTEM_DESIGN.md` | HLD LLD SYSTEM DESIGN | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Kafka_Architecture/01_Multi_Tenant_Event_Streaming_Platform.md` | Multi Tenant Event Streaming Platform | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Kafka_Architecture/02_Real_Time_CDC_Data_Replication_Debezium.md` | Real Time CDC Data Replication Debezium | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Kafka_Architecture/03_Tiered_Storage_Infinite_Retention_Platform.md` | Tiered Storage Infinite Retention Platform | `planned-topic` |

<a id="stream-design"></a>

## Event-sourced CQRS orders, ecommerce clickstream analytics and SIEM/log aggregation

**Teaching home:** [02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/INDEX.md](02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/INDEX.md). **Status:** planned.

**Testable outcome.** Select read-model, aggregation and alerting correctness boundaries for separate workflows.

**Focused-sheet derivations and mechanisms.** Derive event identity/version, materialized-view consistency, session attribution, security-event correlation and replay invariants.

**Numerical/trace exercise to author.** ILLUSTRATIVE: An order stream has versions 1,3,2,3: trace gap detection, reorder and dedup; compute accepted unique versions rather than raw event count. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose eventual-consistency and recovery rules separately for order read models, clickstream reports and SIEM alerts.

**Proposed validation.** Compare replay to an authoritative ledger, inject out-of-order/schema changes, validate query staleness and alert precision/latency.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Lost order update; attribution bias; log-source skew; alert duplication. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Streaming_Architecture_Design` | Source scope/container; not a completed lesson | `source-container` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Streaming_Architecture_Design/00_FIXES_AND_TRADEOFFS.md` | FIXES AND TRADEOFFS | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Streaming_Architecture_Design/00_HLD_LLD_SYSTEM_DESIGN.md` | HLD LLD SYSTEM DESIGN | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Streaming_Architecture_Design/01_Event_Sourced_CQRS_Order_Management.md` | Event Sourced CQRS Order Management | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Streaming_Architecture_Design/02_Clickstream_Analytics_Ecommerce.md` | Clickstream Analytics Ecommerce | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Streaming_Architecture_Design/03_Log_Aggregation_SIEM_Threat_Detection.md` | Log Aggregation SIEM Threat Detection | `planned-topic` |

<a id="stream-incidents"></a>

## Hot-partition consumer lag and Flink checkpoint failure/OOM incidents

**Teaching home:** [10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md). **Status:** planned.

**Testable outcome.** Distinguish skew/slow downstream from checkpoint-state and heap-pressure failures.

**Focused-sheet derivations and mechanisms.** Derive backlog growth/drain, per-partition metrics, checkpoint duration/alignment/state size and memory accounting; incidents remain hypothetical unless evidenced.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Input 1200 events/s, service 1000 for 600 s: calculate 120000 backlog; recovery at input 800/service 1000 drains in 600 s under constant-rate assumptions. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose safe containment and data repair for separate hot-partition and checkpoint/OOM exercises.

**Proposed validation.** Use timestamped onset/detection/restore/repair timeline, competing-cause tests, replay correctness and verified recurrence-prevention criteria.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Scale-out without key repartition; checkpoint timeout hides memory cause; replay duplicates; false recovery from average lag. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Production_Incidents` | Source scope/container; not a completed lesson | `source-container` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Production_Incidents/01_Hot_Partition_Consumer_Lag_Incident.md` | Hot Partition Consumer Lag Incident | `planned-topic` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/Production_Incidents/02_Flink_Checkpoint_Failure_OOM_Incident.md` | Flink Checkpoint Failure OOM Incident | `planned-topic` |

<a id="stream-framework"></a>

## Streaming framework, HLD/LLD and fixes/tradeoffs source documents

**Teaching home:** [02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/INDEX.md](02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/INDEX.md). **Status:** planned.

**Testable outcome.** Use explicit delivery/state/failure and capacity contracts to compare Kafka, Flink and Redpanda designs.

**Focused-sheet derivations and mechanisms.** Derive a workload/state/correctness matrix, not a product-feature checklist; HLD and LLD explain one consistent design.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A 5-minute buffer at 1000 events/s and 1 KiB/event requires 300000 KiB payload before replication and metadata: compute MiB and finite-buffer boundary. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose broker/processor/sink boundaries given event ordering, replay and operational ownership.

**Proposed validation.** Verify product versions and each case-specific HLD/LLD against successful, failed and restored event traces.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Broker guarantee treated as application guarantee; missing sink contract; guessed product equivalence; unpriced retention. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Case_Studies_Streaming_Kafka_Flink_Redpanda` | Source scope/container; not a completed lesson | `source-container` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/00_INDEX.md` | INDEX | `supporting-artifact` |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda/00_STREAMING_SYSTEMS_FRAMEWORK.md` | STREAMING SYSTEMS FRAMEWORK | `planned-topic` |

<a id="deep-payer"></a>

## Healthcare payer research: landscape tensions, UHG/Optum, CVS, FWA, pharmacy/member experience, government, SDOH and value-based care

**Teaching home:** [08_business_domains_and_decision_intelligence/42_healthcare_life_sciences_and_biomedical_ai/INDEX.md](08_business_domains_and_decision_intelligence/42_healthcare_life_sciences_and_biomedical_ai/INDEX.md). **Status:** planned.

**Testable outcome.** Define distinct clinical, administrative, pharmacy, member and government workflows rather than a single company AI stack.

**Focused-sheet derivations and mechanisms.** Derive workflow utility, cost-sensitive error and selection/causal boundaries; separate social determinants from clinical prediction and value-based contracts from proxy accuracy.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 1000 eligible members, 200 contacted, 60 enrolled: compute contact/enrollment denominators; explain why uptake is not an intervention effect. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose a bounded pharmacy/member or care-management workflow and compare rules/manual review with proposed assistance.

**Proposed validation.** Collect primary company/unit/date/region evidence before attributing a system; use temporal/provider splits, expert review and intervention protocols for outcome claims.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Company name mistaken for architecture; selected uptake; unsupported clinical effect; subgroup access disparity. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `deep_dive` | Source scope/container; not a completed lesson | `source-container` |
| `deep_dive/Deep_Dive_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `deep_dive/Deep_Dive_Cheat_Sheets/00_INDEX.md` | INDEX | `supporting-artifact` |
| `deep_dive/Deep_Dive_Cheat_Sheets/01_Landscape_Overview_Three_Tensions.md` | Landscape Overview Three Tensions | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/02_UHG_Optum_Core_AI_Stack.md` | UHG Optum Core AI Stack | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/03_UHG_Optum_Pharmacy_Member_Experience_FWA.md` | UHG Optum Pharmacy Member Experience FWA | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/04_UHG_Optum_Extended_Domains.md` | UHG Optum Extended Domains | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/05_UHG_Optum_Government_SDOH_VBC.md` | UHG Optum Government SDOH VBC | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/06_CVS_Health_Core_AI_Stack.md` | CVS Health Core AI Stack | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/07_CVS_Health_Pharmacy_Equity_Extended.md` | CVS Health Pharmacy Equity Extended | `planned-topic` |
| `deep_dive/part1` | OPEN — part1/  [contents not visible] | `unresolved-source` |
| `deep_dive/11_UHG_Optum_New_Domains_Deep_Dive.md` | UHG Optum New Domains Deep Dive | `planned-topic` |
| `deep_dive/12_CVS_Health_New_Domains_Deep_Dive.md` | CVS Health New Domains Deep Dive | `planned-topic` |

<a id="deep-insurance"></a>

## Chubb P&C, other major insurers, climate risk, document intelligence and insurance AI research

**Teaching home:** [08_business_domains_and_decision_intelligence/43_financial_services_fraud_risk_and_fintech/INDEX.md](08_business_domains_and_decision_intelligence/43_financial_services_fraud_risk_and_fintech/INDEX.md). **Status:** planned.

**Testable outcome.** Separate underwriting, claims, fraud and customer-service decisions and their delayed feedback.

**Focused-sheet derivations and mechanisms.** Derive exposure-adjusted rates, expected claim cost and triage utility under stated development/censoring assumptions.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 1000 policy-years with 80 claims averaging 2500 currency units: compute frequency, severity and pure premium before expenses/development. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose one claims or underwriting intervention supported by public evidence or label it illustrative.

**Proposed validation.** Verify named-company evidence; validate exposure/time/catastrophe slices, adjuster review, model risk and ongoing cost.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unmatured claims; missing exposure denominator; catastrophe shift; hypothetical gain attributed to insurer. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `deep_dive/Deep_Dive_Cheat_Sheets/00_HLD_LLD_SYSTEM_DESIGN_Part1_UHG_CVS_Chubb_Insurers.md` | HLD LLD SYSTEM DESIGN Part1 UHG CVS Chubb Insurers | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/09_Other_Major_Insurers_AI.md` | Other Major Insurers AI | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/16_50_Common_Issues_Insurance_Healthcare_AI.md` | Common Issues Insurance Healthcare AI | `planned-topic` |
| `deep_dive/13_Other_Major_Insurers_AI_Deep_Dive.md` | Other Major Insurers AI Deep Dive | `planned-topic` |

<a id="deep-frontier"></a>

## Agentic/autonomous frontier, federated learning, climate, genomics and governance

**Teaching home:** [08_business_domains_and_decision_intelligence/48_energy_climate_public_sector_and_scientific_ai/INDEX.md](08_business_domains_and_decision_intelligence/48_energy_climate_public_sector_and_scientific_ai/INDEX.md). **Status:** planned.

**Testable outcome.** Choose a bounded frontier method only when it fits data access, intervention and physical/biological constraints.

**Focused-sheet derivations and mechanisms.** Derive federated aggregation/privacy accounting where relevant, agent termination bounds, and genomic split/relatedness controls; do not conflate privacy and decentralization.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Three sites with 100,200,700 samples report scalar updates 1,2,3: compute sample-weighted aggregate and compare unweighted mean; this establishes no privacy. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose centralized, federated or rule-based analysis for a multi-site research question.

**Proposed validation.** Use site/subject/time external validity, leakage audits, privacy/adversarial tests and evidence-led comparison.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Site confounding; genomic relatedness leakage; decentralization mistaken for privacy; autonomy without authority. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `deep_dive/Deep_Dive_Cheat_Sheets/00_HLD_LLD_SYSTEM_DESIGN_Part2_Frontiers_Modalities_Governance.md` | HLD LLD SYSTEM DESIGN Part2 Frontiers Modalities Governance | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/10_New_Frontiers_Agentic_Autonomous_AI.md` | New Frontiers Agentic Autonomous AI | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/11_New_Frontiers_Federated_Learning_Climate_Genomics.md` | New Frontiers Federated Learning Climate Genomics | `planned-topic` |

<a id="deep-speech"></a>

## Speech-to-text/TTS in insurance and healthcare

**Teaching home:** [04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md). **Status:** planned.

**Testable outcome.** Select speech quality metrics and expert review under language, speaker, privacy and timing constraints.

**Focused-sheet derivations and mechanisms.** Derive word error rate, speaker attribution and downstream extraction utility; TTS clarity is not ASR accuracy.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Reference has 100 words, 5 substitutions, 3 deletions and 2 insertions: compute WER; compare downstream critical-term errors separately. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose transcription/voice generation and review policy for a domain call workflow.

**Proposed validation.** Stratify accents, languages, noise and device/site; audit speaker alignment, critical terms, rights and data access.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Low average WER hides critical error; speaker swap; noisy domain shift; unauthorized voice use. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `deep_dive/Deep_Dive_Cheat_Sheets/12_Speech_AI_Voice_Insurance_Healthcare.md` | Speech AI Voice Insurance Healthcare | `planned-topic` |
| `deep_dive/14_Speech_To_Text_TTS_Insurance_Healthcare_Deep_Dive.md` | Speech To Text TTS Insurance Healthcare Deep Dive | `planned-topic` |

<a id="deep-vision"></a>

## P&C computer vision, medical imaging, clinical vision and multimodal documents

**Teaching home:** [04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md). **Status:** planned.

**Testable outcome.** Separate damage estimation, clinical image triage and document extraction into independently evaluated decisions.

**Focused-sheet derivations and mechanisms.** Derive IoU/Dice, calibration and field-level correctness with coordinate/timestamp/modality alignment; no image metric establishes safe clinical use.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Two masks have 80-pixel intersection and 120-pixel union: compute IoU; masks of 100 pixels each give Dice using the same 80-pixel intersection. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose CV/VLM versus manual/OCR baseline separately for claims images, medical scans and multimodal records.

**Proposed validation.** Use patient/property/site-separated data, slice errors, source evidence, human review and missing-modality tests.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Image/record leakage; incorrect coordinate alignment; scanner shift; valid JSON with wrong field values. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `deep_dive/Deep_Dive_Cheat_Sheets/08_Chubb_PnC_Insurance_AI.md` | Chubb PnC Insurance AI | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/13_Computer_Vision_PnC_Insurance.md` | Computer Vision PnC Insurance | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/14_Medical_Imaging_Clinical_Vision_AI.md` | Medical Imaging Clinical Vision AI | `planned-topic` |
| `deep_dive/Deep_Dive_Cheat_Sheets/15_Multimodal_AI_Document_Intelligence.md` | Multimodal AI Document Intelligence | `planned-topic` |
| `deep_dive/15_PnC_Computer_Vision_Image_AI_Deep_Dive.md` | PnC Computer Vision Image AI Deep Dive | `planned-topic` |
| `deep_dive/16_Medical_Imaging_Clinical_Vision_AI_Deep_Dive.md` | Medical Imaging Clinical Vision AI Deep Dive | `planned-topic` |
| `deep_dive/17_Multimodal_AI_Document_Intelligence_Deep_Dive.md` | Multimodal AI Document Intelligence Deep Dive | `planned-topic` |

<a id="pharma-operations"></a>

## Pharma manufacturing, quality and supply-chain operations

**Teaching home:** [11_data_science_domain_specialisations/09_pharma_healthcare_and_life_sciences/03_manufacturing_quality_and_supply_chain/INDEX.md](11_data_science_domain_specialisations/09_pharma_healthcare_and_life_sciences/03_manufacturing_quality_and_supply_chain/INDEX.md). **Status:** planned.

**Testable outcome.** Frame independent pharmaceutical workflows with explicit data grain, timing, expert authority and evidence status.

**Focused-sheet derivations and mechanisms.** Derive domain decision objective, metric denominator and state/approval invariants; split manufacturing, clinical, research and commercial regimes.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A process produces 1000 units with 30 rejects: compute yield and compare a hypothetical reduction to 20 rejects as arithmetic, not observed improvement. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which production/quality or supply exception should an agent investigate, recommend or execute within validated authority?

**Proposed validation.** Trace batch/lot, deviation, audit and authorized action; compare rules/manual triage with bounded agent support. Include proposed failure injections and baseline comparison; no validation result is claimed.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Selected or delayed labels; unsourced company claim; unsafe action scope; opaque evidence chain. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `deep_dive/Pharma_Agentic_Case_Studies/02_Operations_Agentic_AI_Manufacturing_Quality_SupplyChain.md` | Operations Agentic AI Manufacturing Quality SupplyChain | `planned-topic` |

<a id="pharma-clinical"></a>

## Clinical trials, R&D operations, discovery, RWE, biostatistics and pharmacovigilance

**Teaching home:** [11_data_science_domain_specialisations/09_pharma_healthcare_and_life_sciences/02_clinical_rwe_safety_and_document_intelligence/INDEX.md](11_data_science_domain_specialisations/09_pharma_healthcare_and_life_sciences/02_clinical_rwe_safety_and_document_intelligence/INDEX.md). **Status:** planned.

**Testable outcome.** Frame independent pharmaceutical workflows with explicit data grain, timing, expert authority and evidence status.

**Focused-sheet derivations and mechanisms.** Derive domain decision objective, metric denominator and state/approval invariants; split manufacturing, clinical, research and commercial regimes.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Of 100 candidate safety records, 20 are confirmed signals and the system retrieves 15 among 30 alerts: calculate signal recall and alert precision. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which study, safety or research evidence is sufficient for an expert-reviewed recommendation?

**Proposed validation.** Separate observational estimands, trial protocol, safety reporting and discovery metrics; require label maturity and expert adjudication. Include proposed failure injections and baseline comparison; no validation result is claimed.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Selected or delayed labels; unsourced company claim; unsafe action scope; opaque evidence chain. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `deep_dive/Pharma_Agentic_Case_Studies/03_Operations_Agentic_AI_ClinicalTrials_RnD_Ops.md` | Operations Agentic AI ClinicalTrials RnD Ops | `planned-topic` |
| `deep_dive/Pharma_Agentic_Case_Studies/06_DataScience_Agentic_AI_DrugDiscovery_RnD.md` | DataScience Agentic AI DrugDiscovery RnD | `planned-topic` |
| `deep_dive/Pharma_Agentic_Case_Studies/07_DataScience_Agentic_AI_RWE_Biostatistics_Pharmacovigilance.md` | DataScience Agentic AI RWE Biostatistics Pharmacovigilance | `planned-topic` |

<a id="pharma-commercial"></a>

## Field force, medical affairs, market access and patient services

**Teaching home:** [11_data_science_domain_specialisations/09_pharma_healthcare_and_life_sciences/01_commercial_kda_and_omnichannel/INDEX.md](11_data_science_domain_specialisations/09_pharma_healthcare_and_life_sciences/01_commercial_kda_and_omnichannel/INDEX.md). **Status:** planned.

**Testable outcome.** Frame independent pharmaceutical workflows with explicit data grain, timing, expert authority and evidence status.

**Focused-sheet derivations and mechanisms.** Derive domain decision objective, metric denominator and state/approval invariants; split manufacturing, clinical, research and commercial regimes.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 100 eligible HCPs in a hypothetical randomized comparison have 20/50 and 15/50 responses: compute absolute rate difference, not significance without design. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which HCP/patient-service next action is eligible and supported, rather than merely predicted to correlate with response?

**Proposed validation.** Validate eligibility, content approval, access/adherence outcomes, confounding, review capacity and action logs. Include proposed failure injections and baseline comparison; no validation result is claimed.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Selected or delayed labels; unsourced company claim; unsafe action scope; opaque evidence chain. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `deep_dive/Pharma_Agentic_Case_Studies/04_Commercialization_Agentic_AI_FieldForce_MedicalAffairs.md` | Commercialization Agentic AI FieldForce MedicalAffairs | `planned-topic` |
| `deep_dive/Pharma_Agentic_Case_Studies/05_Commercialization_Agentic_AI_MarketAccess_PatientServices.md` | Commercialization Agentic AI MarketAccess PatientServices | `planned-topic` |

<a id="pharma-patterns"></a>

## Pharma landscape, cross-domain architecture, issues and implemented RAG/agent case research

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md). **Status:** planned.

**Testable outcome.** Frame independent pharmaceutical workflows with explicit data grain, timing, expert authority and evidence status.

**Focused-sheet derivations and mechanisms.** Derive domain decision objective, metric denominator and state/approval invariants; split manufacturing, clinical, research and commercial regimes.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A 6-step plan permits 2 attempts/step: bound attempts, then trace one external action committed before timeout. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which one consequential workflow decision has sufficient attributable evidence to become a case?

**Proposed validation.** Require primary evidence for any real-world implemented label; distinguish retrieved content from permission to act and claimed results from reproduced evidence. Include proposed failure injections and baseline comparison; no validation result is claimed.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Selected or delayed labels; unsourced company claim; unsafe action scope; opaque evidence chain. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `deep_dive/Pharma_Agentic_Case_Studies` | Source scope/container; not a completed lesson | `source-container` |
| `deep_dive/Pharma_Agentic_Case_Studies/00_FIXES_AND_TRADEOFFS.md` | FIXES AND TRADEOFFS | `planned-topic` |
| `deep_dive/Pharma_Agentic_Case_Studies/00_HLD_LLD_SYSTEM_DESIGN.md` | HLD LLD SYSTEM DESIGN | `planned-topic` |
| `deep_dive/Pharma_Agentic_Case_Studies/00_INDEX.md` | INDEX | `supporting-artifact` |
| `deep_dive/Pharma_Agentic_Case_Studies/01_Landscape_Overview_Pharma_Agentic_AI.md` | Landscape Overview Pharma Agentic AI | `planned-topic` |
| `deep_dive/Pharma_Agentic_Case_Studies/08_CrossDomain_Agentic_Architecture_Patterns_Pharma.md` | CrossDomain Agentic Architecture Patterns Pharma | `planned-topic` |
| `deep_dive/Pharma_Agentic_Case_Studies/09_Common_Issues_Pharma_Agentic_AI.md` | Common Issues Pharma Agentic AI | `planned-topic` |
| `deep_dive/Pharma_Agentic_Case_Studies/10_RealWorld_Implemented_RAG_Agent_AgenticAI_Case_Studies.md` | RealWorld Implemented RAG Agent AgenticAI Case Studies | `planned-topic` |

<a id="insurance-actuarial"></a>

## Insurance actuarial probability, survival, pricing/GLM, reserving, reinsurance, EVT and Bayesian credibility

**Teaching home:** [11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/01_pricing_underwriting_and_reserving/INDEX.md](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/01_pricing_underwriting_and_reserving/INDEX.md). **Status:** mostly planned; [non-life pricing](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/01_pricing_underwriting_and_reserving/Cheat_Sheets/03_NonLife_Pricing_Ratemaking.md) and [claims reserving](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/01_pricing_underwriting_and_reserving/Cheat_Sheets/05_Claims_Reserving_Methods.md) are unreviewed partial drafts, not reproductions of the unavailable source lessons.

**Testable outcome.** Compute a bounded actuarial estimate with exposure, censoring/development, model assumptions and uncertainty.

**Focused-sheet derivations and mechanisms.** Derive distribution moments, survival/hazard and life-contingency valuation, frequency-severity premium/GLM links and offsets, chain-ladder/development assumptions, reinsurance layers/risk measures, extreme-value tails and hierarchical credibility. Split these into focused sheets.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Assume 1000 policy-years, 100 claims and mean severity 2000 currency units: derive pure premium 200; compute an excess layer payment for loss 150000 with attachment 100000 and limit 30000. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose actuarial baseline versus ML for a clearly bounded pricing or reserving decision.

**Proposed validation.** Time/exposure/claim-development backtests; check GLM link/offset, tail sensitivity, censored observations, reserve uncertainty and payout accounting.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Exposure omission; immature development; extrapolated tail; layer attachment/limit error. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/00_INDEX.md` | INDEX | `supporting-artifact` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/01_Probability_Insurance_Distributions.md` | Probability Insurance Distributions | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/02_Life_Contingencies_Survival_Models.md` | Life Contingencies Survival Models | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/03_NonLife_Pricing_Ratemaking.md` | NonLife Pricing Ratemaking | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/04_GLM_Theory_Insurance.md` | GLM Theory Insurance | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/05_Claims_Reserving_Methods.md` | Claims Reserving Methods | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/06_Reinsurance_Capital_Risk_Measures.md` | Reinsurance Capital Risk Measures | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/09_Extreme_Value_Theory_Catastrophe.md` | Extreme Value Theory Catastrophe | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/10_Bayesian_Credibility_Hierarchical.md` | Bayesian Credibility Hierarchical | `planned-topic` |

<a id="insurance-course"></a>

## Insurance course outlines, 180-day curriculum, Werner–Modlin source, governance, customer analytics and MLOps

**Teaching home:** [11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/03_insurance_mlops_model_risk_and_climate/INDEX.md](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/03_insurance_mlops_model_risk_and_climate/INDEX.md). **Status:** planned.

**Testable outcome.** Link full actuarial curriculum planning to customer, governance and deployed model ownership without inferring hidden course text.

**Focused-sheet derivations and mechanisms.** Derive CLV under explicit discount/retention assumptions, model-risk monitoring and compatible deployment/recovery; retain 50 distinct issues as a source title coverage target.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Annual margin 100 currency units, retention 0.8 and discount 0.1: compute a clearly defined one-period expected discounted margin; do not silently use an infinite-horizon CLV formula. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose review and release controls for an insurance price/claim model with delayed outcomes.

**Proposed validation.** Obtain hidden outlines/course content before claiming exact lessons; audit customer/claims temporal joins, explainability, drift, model-risk evidence and release ownership.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unverified regulatory applicability; CLV conditioning error; delayed-feedback drift; workspace file treated as curriculum. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Insurance_Analytical_Modelling` | Source scope/container; not a completed lesson | `source-container` |
| `Insurance_Analytical_Modelling/course outline` | OPEN — course outline/  [contents not visible] | `unresolved-source` |
| `Insurance_Analytical_Modelling/Course_Content` | OPEN — Course_Content/  [contents not visible] | `unresolved-source` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/07_ML_Ensembles_Insurance_Applications.md` | ML Ensembles Insurance Applications | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/08_Customer_Analytics_CLV_Acquisition.md` | Customer Analytics CLV Acquisition | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/11_Model_Risk_Governance_Explainability.md` | Model Risk Governance Explainability | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/12_MLOps_Insurance_Production.md` | MLOps Insurance Production | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/13_50_Common_Insurance_Analytics_Issues.md` | Common Insurance Analytics Issues | `planned-topic` |
| `Insurance_Analytical_Modelling/comprehensive_insurance_analytics_actuarial_ml_180_day_course.md` | comprehensive insurance analytics actuarial ml 180 day course | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance Analytics Actuarial Course Structure.md` | Insurance Analytics Actuarial Course Structure | `planned-topic` |
| `Insurance_Analytical_Modelling/Insurance_Analytical_Modelling.code-workspace` | Insurance Analytical Modelling | `supporting-artifact` |
| `Insurance_Analytical_Modelling/studynotes_werner_modlin_ratemaking.pdf` | studynotes werner modlin ratemaking | `supporting-artifact` |

<a id="unknown-sources"></a>

## Collapsed GLCC, learn, raw and speech source folders

**Teaching home:** [01_foundations_computing_and_problem_solving/01_program_orientation_and_ai_landscape/INDEX.md](01_foundations_computing_and_problem_solving/01_program_orientation_and_ai_landscape/INDEX.md). **Status:** planned.

**Testable outcome.** Distinguish named source containers from knowable topic coverage.

**Focused-sheet derivations and mechanisms.** No discipline can be inferred from an unexplained folder acronym or hidden contents; preserve inventory identity and access status.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Count unresolved source nodes separately from planned topics; a coverage ratio for visible entries cannot prove hidden-topic completeness. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose to request a filesystem/content export only when exact hidden coverage is needed; finish all visible-topic planning now.

**Proposed validation.** Resolve source access, then extend the exact-node inventory and syllabus without replacing uncertainty with invented lessons.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Acronym guessed as syllabus; hidden source counted complete; invented file; source metadata mistaken for teaching content. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `GLCC` | OPEN — GLCC/  [contents not visible] | `unresolved-source` |
| `learn` | OPEN — learn/  [contents not visible] | `unresolved-source` |
| `raw` | OPEN — raw/  [contents not visible] | `unresolved-source` |
| `speechtotxt` | OPEN — speechtotxt/  [contents not visible] | `unresolved-source` |

## Publication boundary

This is complete **visible-source topic planning**, not completed teaching content or hidden-source coverage. Preserve all existing course topics. Before publishing individual artifacts, obtain sources, author the required structures and topic-specific mathematics, execute numerical/trace checks and relevant validation, answer interview questions, finish actual issue counts, verify links/prerequisites, and update local catalog status and review records.
