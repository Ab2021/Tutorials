# 📅 ABHISHEK'S 18-MONTH DAILY EXECUTION PROTOCOL
## Maximum Open Source & Deep Learning Integration
*Derived from plan_agy.md, plan_cc.md, rough_agy.md, and rough_cc.md*

---

> **THE WEEKLY CADENCE (10-11 Hours/Week)**
> *   **Monday-Thursday (1.5 hrs/day):** Core Implementation & Coding
> *   **Friday (1.0 hr):** OSS Issue Triaging & Community
> *   **Saturday (3.0 hrs):** Deep Study (Courses) & Complex Debugging
> *   **Sunday (1.5 hrs):** Writing (Documentation, Blogs, Social Media) & Planning

**OPEN SOURCE STRATEGY:** Every project must live in public. Every bug is an issue to file. Every missing feature is a PR to make.

---

## 🚀 PHASE 1: FOUNDATIONS & OPEN SOURCE INITIATION (MONTHS 1-3)
*Goal: Signal Processing + PyTorch/TFLite Base + First OSS PRs*

### MONTH 1: The Biosensing Edge Pipeline
**Focus:** Domain 1 (Signals), NeuroKit2, Arduino Deployment

*   **Week 1: Setup & Signal Processing (Domain 1.1 - 1.2)**
    *   **Mon (D1):** Set up `edge-ml-pivot` repo. Install Conda env (PyTorch, NeuroKit2, edgeimpulse).
    *   **Tue (D2):** Download PhysioNet MIT-BIH dataset. Read MIT OCW 6.003 (Signals/Systems) Lecture 1.
    *   **Wed (D3):** Write `ecg_filter_bank()` in Python (Butterworth bandpass, notch filter). Commit to GitHub.
    *   **Thu (D4):** Implement R-peak detection using NeuroKit2.
    *   **Fri (D5):** **OSS Target:** Fork `neuropsychology/NeuroKit`. Clone locally. Run their test suite.
    *   **Sat (D6):** Connect MPU-6050 to Arduino. Read MPU data via Serial. Study I2C basics.
    *   **Sun (D7):** Post LinkedIn Update #1 (Signal filtering viz). Plan Week 2.

*   **Week 2: Deep ML for Signals (Domain 2.1 - 2.2)**
    *   **Mon (D8):** Design `TinyECGNet` (1D-CNN) in PyTorch.
    *   **Tue (D9):** Write Kaggle training script. Train baseline model on MIT-BIH data.
    *   **Wed (D10):** Implement Focal Loss to handle AFib class imbalance. Retrain.
    *   **Thu (D11):** Implement `DilatedECGNet` and compare vs baseline. Commit results.
    *   **Fri (D12):** **OSS Target:** Review open issues on `tensorflow/tflite-micro` labeled "good first issue".
    *   **Sat (D13):** Deep Study: Fast.ai Practical Deep Learning (Part 1). Build gesture dataset via Arduino.
    *   **Sun (D14):** Post Twitter Thread #1: "Training 1D-CNNs for Edge devices."

*   **Week 3: PTQ Quantization & Initial Deployment (Domain 3.1 & 4.1)**
    *   **Mon (D15):** Apply PyTorch Post-Training Quantization (PTQ) to ECG model.
    *   **Tue (D16):** Export INT8 model to ONNX, then TFLite. Measure size reduction.
    *   **Wed (D17):** Convert `.tflite` to C array (`xxd`). Flash to Arduino.
    *   **Thu (D18):** Run live gesture inference on Arduino. Log latency.
    *   **Fri (D19):** **OSS Target:** Draft a documentation fix or example script PR for `NeuroKit2` (e.g., batch processing ECG).
    *   **Sat (D20):** Deep Study: Read "Deep Compression" paper (Song Han). Debug TFLite Micro arena sizing.
    *   **Sun (D21):** Post LinkedIn Update #2: "Benchmarking FP32 vs INT8 on Arduino."

*   **Week 4: Month 1 Demo & OSS Push**
    *   **Mon (D22):** Build unified Python demo (simulating ECG + reading live gestures).
    *   **Tue (D23):** Write comprehensive README.md with Mermaid architecture diagrams.
    *   **Wed (D24):** Clean up repository code, add docstrings, push v1.0.
    *   **Thu (D25):** Create short demo GIF (screen recording).
    *   **Fri (D26):** **OSS Target:** Submit PR to `NeuroKit2`.
    *   **Sat (D27):** Deep Study: Start Harvard CS249r / TinyMLedu course material.
    *   **Sun (D28):** Publish Reddit post to r/embedded: Month 1 project architecture.

### MONTH 2: STM32 CubeAI & HRV Depth
**Focus:** Domain 5 (ARM Arch), STM32 Nucleo, PTB-XL Dataset

*   **Week 5: STM32 & Cortex-M Architecture (Domain 5.1 - 5.3)**
    *   **Mon (D29):** Unbox STM32 Nucleo. Install STM32CubeIDE and X-CUBE-AI.
    *   **Tue (D30):** Flash basic UART blink script to STM32. Read ARM Cortex-M4 spec.
    *   **Wed (D31):** Download PTB-XL dataset. Build data loader in PyTorch.
    *   **Thu (D32):** Train lightweight PPG Anomaly Net (<30K params).
    *   **Fri (D33):** **OSS Target:** Read `ARM-software/CMSIS-NN` README and directory structure.
    *   **Sat (D34):** Deep Study: MIT 6.S965 EfficientML Lecture 1 (Intro) & 2 (Pruning).
    *   **Sun (D35):** LinkedIn Post #3: "Cortex-M4 memory hierarchy for ML Engineers."

*   **Week 6: Deployment & Power (Domain 5.4)**
    *   **Mon (D36):** Convert PPG model to C via X-CUBE-AI.
    *   **Tue (D37):** Flash model to STM32. Run inference.
    *   **Wed (D38):** Profile STM32 latency (Target <50ms @ 84MHz).
    *   **Thu (D39):** Implement FreeRTOS task to run inference periodically.
    *   **Fri (D40):** **OSS Target:** Find a bug/typo in STM32 documentation or TFLite Micro examples; file issue.
    *   **Sat (D41):** Deep Study: MIT 6.S965 Lecture 3 (Quantization 1). Build HRV Streamlit dashboard.
    *   **Sun (D42):** Twitter Thread: "Deploying ML to STM32: TFLite Micro vs X-CUBE-AI."

*   **Week 7: Advanced Signal Processing (Domain 1.3 - 1.5)**
    *   **Mon (D43):** Add AC/DC separation for PPG signal in Python.
    *   **Tue (D44):** Implement SpO2 calculation (Ratio-of-ratios).
    *   **Wed (D45):** Add Welch's method for LF/HF ratio extraction to Dashboard.
    *   **Thu (D46):** Optimize dashboard to read live from STM32 UART.
    *   **Fri (D47):** **OSS Target:** Engage in `tensorflow/tflite-micro` GitHub discussions.
    *   **Sat (D48):** Deep Study: MIT 6.S965 Lecture 4 (Quantization 2). Read QAT PyTorch docs.
    *   **Sun (D49):** Draft Article 1: "From Data Scientist to Edge AI."

*   **Week 8: QAT Implementation (Domain 3.2)**
    *   **Mon (D50):** Refactor PyTorch training code to use Fake Quantization (QAT).
    *   **Tue (D51):** Train AFib detector v1 with QAT.
    *   **Wed (D52):** Handle BatchNorm folding issues during QAT.
    *   **Thu (D53):** Export QAT model and deploy to STM32.
    *   **Fri (D54):** **OSS Target:** Submit minor PR to `tflite-micro` (e.g., adding an assertion/comment to an example).
    *   **Sat (D55):** Deep Study: MIT 6.S965 Lecture 5 (NAS). Compare QAT vs PTQ results.
    *   **Sun (D56):** Publish Article 1 to Medium/Substack.

### MONTH 3: Knowledge Distillation & First Major OSS Merge
**Focus:** Domain 3 (Distillation), ONNX Runtime on Pi 5

*   **Week 9: Knowledge Distillation (Domain 3.5)**
    *   **Mon (D57):** Train a large Teacher model (DilatedECGNet, ~64K params) to 90% accuracy.
    *   **Tue (D58):** Build `IntermediateDistillationLoss` class.
    *   **Wed (D59):** Train Student model (TinyECGNet, ~8K params) via Distillation.
    *   **Thu (D60):** Compare Student accuracy vs standalone training.
    *   **Fri (D61):** **OSS Target:** Track status of previous PRs. Review `huggingface/optimum` repo structure.
    *   **Sat (D62):** Deep Study: MIT 6.S965 Lecture 6 (Knowledge Distillation).
    *   **Sun (D63):** LinkedIn Post: "Knowledge Distillation for Edge devices."

*   **Week 10: ONNX Runtime Basics (Domain 4.2)**
    *   **Mon (D64):** Boot Raspberry Pi 5. Install `onnxruntime`.
    *   **Tue (D65):** Export Student model to ONNX. Run inference on Pi 5.
    *   **Wed (D66):** Enable ONNX Runtime profiling. Identify bottlenecks.
    *   **Thu (D67):** Test multi-threading on Pi 5's Cortex-A76 cores.
    *   **Fri (D68):** **OSS Target:** Find an unhandled edge case in a NeuroKit2 function. Create fix branch.
    *   **Sat (D69):** Deep Study: Read ONNX Runtime documentation for execution providers.
    *   **Sun (D70):** Reddit Post: "Pi 5 ONNX Runtime Benchmarks for 1D-CNNs."

*   **Week 11: Multi-Vital Monitor Integration**
    *   **Mon (D71):** Wire AD8232 (ECG) to STM32. Read analog values.
    *   **Tue (D72):** Send ECG values via UART from STM32 to Pi 5.
    *   **Wed (D73):** Run Pi 5 ONNX Runtime inference on live incoming UART data.
    *   **Thu (D74):** Update Streamlit dashboard on Pi 5 to display live predictions.
    *   **Fri (D75):** **OSS Target:** Submit PR to `NeuroKit2` for new biosignal algorithm or fix.
    *   **Sat (D76):** Deep Study: Finish Harvard CS249r material.
    *   **Sun (D77):** Twitter Thread: "Building a Multi-Vital Monitor using Pi 5 and STM32."

*   **Week 12: Month 3 Review & Portfolio Polish**
    *   **Mon (D78):** Finalize code for Multi-Vital repo.
    *   **Tue (D79):** Create architecture diagrams for the dual-chip system.
    *   **Wed (D80):** Publish benchmark table (STM32 vs Pi 5 latency).
    *   **Thu (D81):** Refactor code for PEP8 compliance.
    *   **Fri (D82):** **OSS Target:** Engage with maintainers on PR reviews.
    *   **Sat (D83):** Deep Study: Consolidation week. Review signal processing concepts.
    *   **Sun (D84):** LinkedIn Post: "Month 3 Update: Running dual-inference pipelines."

---

## 🛠️ PHASE 2: HARDWARE DEPTH & TVM (MONTHS 4-6)
*Goal: Apache TVM Mastery, Compression Benchmarks, CMSIS-NN*

### MONTH 4: TVM Deep Dive & MLIR Intro
**Focus:** Domain 4.3 (TVM), Domain 6.2 (MLIR)

*   **Week 13: Apache TVM Basics (Domain 4.3)**
    *   **Mon (D85):** Install Apache TVM on laptop and Pi 5 (RPC server).
    *   **Tue (D86):** Import ONNX ECG model into TVM Relay IR.
    *   **Wed (D87):** Compile model for `llvm -mcpu=cortex-a76`. Run on Pi.
    *   **Thu (D88):** Measure latency vs ONNX Runtime baseline.
    *   **Fri (D89):** **OSS Target:** Fork `apache/tvm`. Build from source. Read contribution guidelines.
    *   **Sat (D90):** Deep Study: TVM Auto-scheduling documentation.
    *   **Sun (D91):** LinkedIn Post: "First look at Apache TVM on Raspberry Pi 5."

*   **Week 14: TVM AutoScheduler**
    *   **Mon (D92):** Setup TVM TaskScheduler for Pi 5 target.
    *   **Tue (D93):** Run 500 tuning trials (AutoScheduler).
    *   **Wed (D94):** Apply best schedule (`ApplyHistoryBest`). Re-measure latency.
    *   **Thu (D95):** Analyze generated C/assembly code for fused operators.
    *   **Fri (D96):** **OSS Target:** Draft a Pi 5 specific tutorial for TVM documentation.
    *   **Sat (D97):** Deep Study: "MCUNet" Paper (NAS + TinyML).
    *   **Sun (D98):** Draft Article 2: "Apache TVM vs ONNX Runtime on Pi 5."

*   **Week 15: CMSIS-NN Depth (Domain 5.2)**
    *   **Mon (D99):** Write naive C convolution. Benchmark on STM32.
    *   **Tue (D100):** Link `arm_convolve_s8` from CMSIS-NN.
    *   **Wed (D101):** Benchmark CMSIS-NN vs naive C (Observe 4x+ speedup).
    *   **Thu (D102):** Integrate custom CMSIS-NN op into TFLite Micro via Custom Op.
    *   **Fri (D103):** **OSS Target:** Submit Pi 5 tutorial PR to `apache/tvm`.
    *   **Sat (D104):** Deep Study: Read CMSIS-NN source code for `arm_fully_connected_s8`.
    *   **Sun (D105):** Twitter Thread: "Why CMSIS-NN is magic for Cortex-M."

*   **Week 16: MLIR Literacy (Domain 6.2)**
    *   **Mon (D106):** Read MLIR "Toy Language" tutorial chapters 1-2.
    *   **Tue (D107):** Compile LLVM/MLIR project locally.
    *   **Wed (D108):** Complete MLIR Toy chapters 3-4 (Dialects).
    *   **Thu (D109):** Dump MLIR IR from a basic PyTorch export using `torch-mlir`.
    *   **Fri (D110):** **OSS Target:** File documentation issue or minor fix in `llvm/llvm-project` (MLIR docs).
    *   **Sat (D111):** Deep Study: "MLIR: A Compiler Infrastructure" Paper.
    *   **Sun (D112):** Publish Article 2 on Medium/Substack.

### MONTH 5: The Grand Compression Benchmark
**Focus:** Domain 3 (Pruning, NAS, AWQ)

*   **Week 17: Structured Pruning (Domain 3.4)**
    *   **Mon (D113):** Implement L1-norm filter pruning on ECG model.
    *   **Tue (D114):** Prune 30% of channels. Re-train to recover accuracy.
    *   **Wed (D115):** Prune 50%. Observe accuracy drop.
    *   **Thu (D116):** Export pruned models and measure real-world inference speedup.
    *   **Fri (D117):** **OSS Target:** Explore `neuralmagic/sparseml` repo. Run their examples.
    *   **Sat (D118):** Deep Study: MIT 6.S965 Lecture 7 & 8 (On-Device Training).
    *   **Sun (D119):** LinkedIn Post: "Why Unstructured Pruning is a trap for Edge ML."

*   **Week 18: Neural Architecture Search (Domain 3.6)**
    *   **Mon (D120):** Setup `MCUNet/tinynas` toolkit.
    *   **Tue (D121):** Define search space for STM32F411 (Target 50ms, 128KB flash).
    *   **Wed (D122):** Run NAS search loop (Evolutionary algorithm/RL).
    *   **Thu (D123):** Train the best found architecture.
    *   **Fri (D124):** **OSS Target:** Open issue/PR in `MCUNet` adding ECG dataset support.
    *   **Sat (D125):** Deep Study: "Once-for-All" Paper (Han Lab).
    *   **Sun (D126):** Reddit Post to r/MachineLearning: "Using NAS for MCU constraints."

*   **Week 19: AWQ and Edge Impulse (Domain 3.3 & 4.5)**
    *   **Mon (D127):** Create Edge Impulse account. Upload gesture data.
    *   **Tue (D128):** Use EON Tuner to automatically find optimal model.
    *   **Wed (D129):** Deploy Edge Impulse firmware to Arduino.
    *   **Thu (D130):** Test AutoAWQ library on a small language model locally.
    *   **Fri (D131):** **OSS Target:** Write a tutorial for `huggingface/optimum` exporting biosignals.
    *   **Sat (D132):** Deep Study: Edge Impulse DSP block implementation details.
    *   **Sun (D133):** LinkedIn Post: "EON Tuner vs Manual NAS."

*   **Week 20: Publishing the Benchmark Repository**
    *   **Mon (D134):** Compile data: FP32 vs PTQ vs QAT vs Pruned vs NAS.
    *   **Tue (D135):** Format beautiful Markdown tables and graphs.
    *   **Wed (D136):** Write the "Compression Benchmark Study" README.
    *   **Thu (D137):** Polish repo: `edge-compression-benchmarks`. Make public.
    *   **Fri (D138):** **OSS Target:** Submit PR to `huggingface/optimum` (example script).
    *   **Sat (D139):** Deep Study: Consolidation week. Review compiler basics.
    *   **Sun (D140):** Draft Article 3: "INT8 vs QAT vs AWQ vs Pruning."

### MONTH 6: Production AFib Detector & RISC-V
**Focus:** Domain 6.4 (RISC-V), CPSC 2018 Dataset

*   **Week 21: Production AFib Dataset (Domain 2.5)**
    *   **Mon (D141):** Download CPSC 2018 Challenge dataset (much larger than MIT-BIH).
    *   **Tue (D142):** Write advanced data loaders with Time Masking augmentation.
    *   **Wed (D143):** Train combined 1D-CNN + Attention gate model (FP32).
    *   **Thu (D144):** Implement k-fold cross-validation.
    *   **Fri (D145):** **OSS Target:** Review active PRs in `tensorflow/tflite-micro`.
    *   **Sat (D146):** Deep Study: RISC-V ISA spec overview (Vector extensions).
    *   **Sun (D147):** Publish Article 3 to Medium/Substack.

*   **Week 22: QAT & Porting for Production**
    *   **Mon (D148):** Apply QAT to the new production model.
    *   **Tue (D149):** Optimize for STM32 (Target: AUC > 0.92, <64KB).
    *   **Wed (D150):** Test against raw noisy sensor data (Domain 2.4).
    *   **Thu (D151):** Finalize deployment C code.
    *   **Fri (D152):** **OSS Target:** Fix a test case in `tflite-micro` and submit PR.
    *   **Sat (D153):** Deep Study: Learn QEMU emulation basics.
    *   **Sun (D154):** LinkedIn Post: "Handling real-world noise in ECG deployment."

*   **Week 23: RISC-V Emulation (Domain 6.4)**
    *   **Mon (D155):** Install `qemu-riscv64` and cross-compilation toolchain.
    *   **Tue (D156):** Cross-compile basic C inference code to `rv64gcv`.
    *   **Wed (D157):** Run inference in QEMU. Print cycle counts.
    *   **Thu (D158):** Compare RISC-V cycle counts with ARM Cortex-M4 estimates.
    *   **Fri (D159):** **OSS Target:** Explore `shaktiproject/SHAKTI-SoC` repository.
    *   **Sat (D160):** Deep Study: Mindgrove SOC architecture (Indian Semiconductor).
    *   **Sun (D161):** Twitter Thread: "Cross-compiling ML for RISC-V."

*   **Week 24: Phase 2 Review & Industry Targeting**
    *   **Mon (D162):** Document RISC-V experiment in a new repo.
    *   **Tue (D163):** Update Resume with Phase 1 & 2 projects (Metrics focused).
    *   **Wed (D164):** Analyze target companies: Sophrosyne, Mindgrove.
    *   **Thu (D165):** Send 5 personalized LinkedIn connection requests to engineers there.
    *   **Fri (D166):** **OSS Target:** PR review follow-ups.
    *   **Sat (D167):** Deep Study: Refresh LLM architecture knowledge (prep for Phase 3).
    *   **Sun (D168):** Draft Article 4: "India's Biosensing Semiconductor Boom."

---

## 🧠 PHASE 3: EDGE RAG & LLM INTEGRATION (MONTHS 7-9)
*Goal: Sovereign Health System, llama.cpp, Medical RAG*

### MONTH 7: SLM & llama.cpp Basics
**Focus:** Domain 7.1 & 4.4

*   **Week 25: llama.cpp Setup (Domain 4.4)**
    *   **Mon (D169):** Clone `llama.cpp`. Compile on Pi 5.
    *   **Tue (D170):** Download Gemma 3 2B (Q4_K_M). Run basic CLI inference.
    *   **Wed (D171):** Profile tokens/sec on Pi 5. Adjust thread count.
    *   **Thu (D172):** Test Phi-3-mini and Qwen2.5-1.5B. Benchmark all three.
    *   **Fri (D173):** **OSS Target:** Add ARM benchmark results to `llama.cpp` wiki/issues.
    *   **Sat (D174):** Deep Study: GGUF format specification.
    *   **Sun (D175):** Publish Article 4 to Medium/Substack.

*   **Week 26: Python Integration**
    *   **Mon (D176):** Install `llama-cpp-python`.
    *   **Tue (D177):** Write wrapper class `LlamaCppInference`.
    *   **Wed (D178):** Prompt engineering for medical symptom extraction.
    *   **Thu (D179):** Build API endpoint using FastAPI on the Pi 5.
    *   **Fri (D180):** **OSS Target:** Search for open issues in `llama-cpp-python`.
    *   **Sat (D181):** Deep Study: MIT 6.S965 Lecture on Efficient LLMs.
    *   **Sun (D182):** LinkedIn Post: "Benchmarking 2B parameter LLMs on a Raspberry Pi 5."

*   **Week 27: Building the Vector Index (Domain 7.2)**
    *   **Mon (D183):** Install `faiss-cpu` and `sentence-transformers`.
    *   **Tue (D184):** Download medical abstracts/guidelines text corpus.
    *   **Wed (D185):** Chunk texts and generate embeddings (`all-MiniLM-L6-v2`).
    *   **Thu (D186):** Create FAISS `IndexFlatIP` index. Save to disk.
    *   **Fri (D187):** **OSS Target:** Submit minor fix/doc update to `llama-cpp-python`.
    *   **Sat (D188):** Deep Study: FAISS Index structures (IVF vs Flat).
    *   **Sun (D189):** Twitter Thread: "Edge-native RAG: Fitting vector DBs in 4GB RAM."

*   **Week 28: Edge RAG Pipeline Completion**
    *   **Mon (D190):** Combine FAISS retriever with `LlamaCppInference`.
    *   **Tue (D191):** Implement `generate_with_context()` function.
    *   **Wed (D192):** Test QA pipeline with medical queries. Tune temperature.
    *   **Thu (D193):** Optimize prompt template for Gemma 3.
    *   **Fri (D194):** **OSS Target:** Start working on `MicroTVM` ECG integration research.
    *   **Sat (D195):** Deep Study: Consolidation. Review all edge RAG code.
    *   **Sun (D196):** Reddit Post to r/LocalLLaMA: "Fully offline Medical RAG on Pi 5."

### MONTH 8: The Sovereign Health Hero Project
**Focus:** Integrating MCU (STM32) + MPU (Pi 5 RAG)

*   **Week 29: System Communication Integration**
    *   **Mon (D197):** Set up UART parsing on Pi 5 to read STM32 AFib results.
    *   **Tue (D198):** Create `sensor_context` dict to hold live vitals.
    *   **Wed (D199):** Feed `sensor_context` into RAG prompt dynamically.
    *   **Thu (D200):** Test end-to-end: STM32 detects anomaly -> Pi 5 LLM generates clinical advice based on RAG.
    *   **Fri (D201):** **OSS Target:** Open issue in `tvm` outlining MicroTVM ECG tutorial idea.
    *   **Sat (D202):** Deep Study: Designing real-time ML systems (System Architecture).
    *   **Sun (D203):** LinkedIn Post: "Connecting Cortex-M inference to Cortex-A LLMs."

*   **Week 30: Dashboard & UI**
    *   **Mon (D204):** Build Streamlit dashboard UI (Sensor charts + Chat interface).
    *   **Tue (D205):** Integrate Plotly for live ECG/PPG waveform plotting.
    *   **Wed (D206):** Connect Chat interface to local RAG backend.
    *   **Thu (D207):** Optimize Streamlit to run smoothly alongside `llama.cpp`.
    *   **Fri (D208):** **OSS Target:** Draft code for `MicroTVM` ECG integration.
    *   **Sat (D209):** Deep Study: Dashboard optimization techniques.
    *   **Sun (D210):** Draft Article 5: "Running Gemma 3 on 7,500 INR Hardware."

*   **Week 31: Hero Project Polish & Launch**
    *   **Mon (D211):** Name project `sovereign-health-edge-ai`.
    *   **Tue (D212):** Write extensive 5-page README.md.
    *   **Wed (D213):** Record Demo GIF (screen recording of Streamlit, no face).
    *   **Thu (D214):** Push repo to GitHub. Tag appropriately.
    *   **Fri (D215):** **OSS Target:** Refine `MicroTVM` PR based on maintainer feedback.
    *   **Sat (D216):** Deep Study: Write technical arXiv-style report.
    *   **Sun (D217):** Publish Article 5 to Medium/Substack. Massive LinkedIn launch post.

*   **Week 32: Community Blitz**
    *   **Mon (D218):** Post Hero project to r/MachineLearning.
    *   **Tue (D219):** Post Hero project to HackerNews (Show HN).
    *   **Wed (D220):** Send repo to LinkedIn connections at Target Companies.
    *   **Thu (D221):** Add project explicitly to Resume.
    *   **Fri (D222):** **OSS Target:** Submit the `MicroTVM` ECG integration PR to `apache/tvm` (High Impact).
    *   **Sat (D223):** Deep Study: Rest.
    *   **Sun (D224):** Twitter Thread: Breakdown of the Hero Project Architecture.

### MONTH 9: On-Device Fine-Tuning & Advanced Optimization
**Focus:** Domain 7.3, AWQ

*   **Week 33: Edge LoRA (Domain 7.3)**
    *   **Mon (D225):** Setup HuggingFace `peft` and `trl` on cloud GPU (Kaggle/Colab).
    *   **Tue (D226):** Format a small medical QA dataset for fine-tuning.
    *   **Wed (D227):** Fine-tune Qwen2.5-1.5B using QLoRA (Rank 8).
    *   **Thu (D228):** Export LoRA adapters and merge.
    *   **Fri (D229):** **OSS Target:** Submit issue to `peft` regarding edge SLM configurations.
    *   **Sat (D230):** Deep Study: Read LoRA and QLoRA papers.
    *   **Sun (D231):** LinkedIn Post: "Why Rank 8 LoRA is enough for SLMs."

*   **Week 34: GGUF Conversion & AWQ (Domain 3.3)**
    *   **Mon (D232):** Convert fine-tuned model to GGUF format.
    *   **Tue (D233):** Deploy custom fine-tuned GGUF to Pi 5.
    *   **Wed (D234):** Implement AutoAWQ on a baseline model.
    *   **Thu (D235):** Compare GGUF (llama.cpp) vs AWQ (vLLM/transformers) speeds.
    *   **Fri (D236):** **OSS Target:** Add biosignal LoRA example to `peft` or `optimum` docs.
    *   **Sat (D237):** Deep Study: AWQ paper (Activation-aware Weight Quantization).
    *   **Sun (D238):** Draft Article 6: "On-Device Medical RAG: Architecture."

*   **Week 35: System Benchmarking & Profiling**
    *   **Mon (D239):** Conduct full system power analysis (USB ammeter).
    *   **Tue (D240):** Profile RAM usage across the pipeline.
    *   **Wed (D241):** Optimize memory leaks in Python/Streamlit integration.
    *   **Thu (D242):** Finalize all benchmark metrics for portfolio.
    *   **Fri (D243):** **OSS Target:** Submit PR for the biosignal LoRA example.
    *   **Sat (D244):** Deep Study: Linux `perf` tool for edge profiling.
    *   **Sun (D245):** Publish Article 6 to Medium/Substack.

*   **Week 36: Phase 3 Review & Portfolio Lock**
    *   **Mon (D246):** Ensure all 3 major repos are fully documented.
    *   **Tue (D247):** Update LinkedIn Featured section.
    *   **Wed (D248):** Review OSS PR tracker (Goal: 6-8 merged PRs by now).
    *   **Thu (D249):** Final Resume Polish (Ready for Tier 1 applications).
    *   **Fri (D250):** **OSS Target:** Triage MLIR codebase for a "good first issue".
    *   **Sat (D251):** Deep Study: Consolidation.
    *   **Sun (D252):** LinkedIn Post: "9 Months In: From Cloud to Silicon."

---

## 🎯 PHASE 4, 5, 6: ADVANCED COMPILERS & REMOTE JOBS (MONTHS 10-18)
*(The exact day-by-day repeats the execution rigor of M1-M9, shifting heavily toward MLIR, Interviews, and Job Applications.)*

### Months 10-12 (Weeks 37-52): MLIR & Tier 1 Applications
*   **Implementation:** Complete MLIR toy tutorial. Write custom passes.
*   **OSS Target:** MLIR test cases (`llvm-project`), ONNX Runtime embedded optimization.
*   **Job Strategy:** Apply aggressively to Sophrosyne, Mindgrove, Netrasemi. Leverage GitHub portfolio. Follow up with engineers.

### Months 13-15 (Weeks 53-65): Remote Platform Positioning
*   **Implementation:** MicroPython on RISC-V, FPGA basics (optional exploration).
*   **Job Strategy:** Build profiles on Turing.com, Toptal, Crossover. Apply to US/EU remote-first edge AI startups.
*   **Writing:** Publish articles targeting global technical recruiters.

### Months 16-18 (Weeks 66-78): Interview Mastery
*   **Implementation:** Practice whiteboarding the 10 core interview questions (CMSIS-NN, Operator Fusion, QAT math).
*   **Job Strategy:** Negotiate offers. Leverage competing offers from Tier 1 India vs Remote Tier 3.

---

## 📊 THE OSS & CONTENT TRACKER CHECKLIST
Print this and put it on your wall.

**Open Source PR Targets (12 Total):**
*   [ ] PR 1: NeuroKit2 (Documentation/Utility) - Month 1
*   [ ] PR 2: TFLite Micro (Example Fix) - Month 2
*   [ ] PR 3: NeuroKit2 (Bug Fix) - Month 3
*   [ ] PR 4: TFLite Micro (Issue Fix) - Month 4
*   [ ] PR 5: Apache TVM (Pi 5 Tutorial) - Month 4/5
*   [ ] PR 6: MCUNet (Dataset Addition) - Month 5
*   [ ] PR 7: HuggingFace Optimum (Biosignal Export) - Month 6
*   [ ] PR 8: Apache TVM (MicroTVM ECG Integration) - Month 8
*   [ ] PR 9: PEFT/Optimum (Edge LoRA Example) - Month 9
*   [ ] PR 10: ONNX Runtime (Mobile/Embedded Test) - Month 11
*   [ ] PR 11: LLVM/MLIR (Tutorial/Test case) - Month 13
*   [ ] PR 12: ONNX Runtime (Embedded Optimization) - Month 15

**Article Targets (1 per month):**
*   [ ] M2: "From Data Scientist to Edge AI"
*   [ ] M4: "Apache TVM vs ONNX Runtime on Pi 5"
*   [ ] M6: "INT8 vs QAT vs AWQ vs Pruning"
*   [ ] M8: "Running Gemma 3 on 7,500 INR Hardware"
*   [ ] M9: "On-Device Medical RAG Architecture"
*   [ ] (Continue monthly...)
