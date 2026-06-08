# 🎯 THE MASTER OSS PLAN: 78-Week Day-by-Day Career Pivot
## Senior Data Scientist → Hardware-Aware ML Engineer (All Open Source, All Implementation)
### Abhishek Bhardwaj | June 2026 – December 2027
### Philosophy: "Every week produces a commit — to GitHub or to an OSS repo."

---

## 📋 Plan Architecture

```
PHASE 1 (Weeks 1-12):  FOUNDATIONS + OSS ENTRY
PHASE 2 (Weeks 13-24): HARDWARE DEPTH + OSS GROWTH
PHASE 3 (Weeks 25-36): DOMAIN MASTERY + SUBSTANTIVE OSS
PHASE 4 (Weeks 37-48): HERO PROJECT + SENIOR OSS
PHASE 5 (Weeks 49-60): ADVANCED COMPILER TOUCH + REMOTE PREP
PHASE 6 (Weeks 61-78): INTERVIEW PREP + TRANSITION EXECUTION
```

**Weekly time budget:** 10-12 hours (working professional)
**Allocation:**
- 30% Study (courses, papers, docs) — ~3.5 hrs
- 30% Implementation (code, hardware) — ~3.5 hrs
- 20% OSS (research, PRs, issues) — ~2.5 hrs
- 15% Content (writing, posts) — ~1.5 hrs
- 5% Community (LinkedIn, Reddit, replies) — ~0.5 hrs

**Golden Rule:** No week passes without at least one commit to a public repository (your own or OSS).

---

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: FOUNDATIONS + OSS ENTRY (WEEKS 1-12)
# Months 1-3 | Goal: First OSS PR merged, hardware running, signal processing mastered
# ═══════════════════════════════════════════════════════════════════

> **Phase 1 Philosophy:** You are a beginner in hardware and OSS. Treat every interaction as learning. File issues before PRs. Read code before writing it.

---

## WEEK 1: Environment + First Contact with OSS
### Theme: "Set up your lab. Download data. Make your first mark on the internet."

#### Learning Objectives
- [ ] MIT OCW 6.003: Signals & Systems — Lecture 1-2 (sampling, aliasing)
- [ ] Read: `neuropsychology/NeuroKit` README + CONTRIBUTING.md fully
- [ ] Understand: Why ECG is sampled at 360Hz (Nyquist theorem)

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `neuropsychology/NeuroKit` | Star, fork, read full codebase structure | Entry | Know where ECG modules live |
| `tensorflow/tflite-micro` | Star, read CONTRIBUTING.md, join mailing list | Entry | Understand contribution process |

#### Day-by-Day Schedule

**Monday — Study: Sampling Theory**
- **Time:** 1.5 hrs
- **Activity:** MIT OCW 6.003 Lecture 1 (YouTube) + notes
- **Deliverable:** `learning_log/week01_sampling.md` with your own summary
- **Resource:** https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/
- **Key concept:** Nyquist rate = 2 × f_max. ECG max frequency ~180Hz → 360Hz minimum.

**Tuesday — Implement: Environment Setup**
- **Time:** 1.5 hrs
- **Activity:** Create conda env, install all packages, create GitHub repo `edge-ml-pivot`
- **Deliverable:** Working Python env + public GitHub repo with README skeleton
- **Commands:**
```bash
conda create -n edgeai python=3.11 -y
conda activate edgeai
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow==2.15.0 onnx onnxruntime onnx-simplifier
pip install neurokit2 scipy matplotlib pandas numpy jupyter wfdb heartpy biosppy
pip install torchao optimum neural-compressor
pip install edgeimpulse pyserial streamlit plotly kaleido
pip install torchinfo fvcore thop
```

**Wednesday — OSS: NeuroKit2 Deep Read**
- **Time:** 1.5 hrs
- **Activity:** Read `neurokit2/ecg/` directory source code. Find the R-peak detection module.
- **Deliverable:** `oss_notes/neurokit2_ecg_structure.md` — document the module layout
- **Files to read:** `ecg_clean.py`, `ecg_peaks.py`, `ecg_findpeaks.py`
- **Why:** Understanding codebase structure before contributing is mandatory.

**Thursday — Integrate: Download Datasets**
- **Time:** 1.5 hrs
- **Activity:** Download MIT-BIH from PhysioNet, download UCI HAR dataset
- **Deliverable:** Data in `data/ecg/` and `data/gesture/` + download script committed
- **MIT-BIH:** https://physionet.org/content/mitdb/1.0.0/ (create free account)
- **UCI HAR:** https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using-smartphones

**Friday — Content: LinkedIn Setup**
- **Time:** 1.0 hr
- **Activity:** Update LinkedIn headline, add skills (Edge AI, TinyML, Signal Processing)
- **Deliverable:** Live optimized profile + draft first post for Week 2
- **Headline template:** `Senior Data Scientist | Building TinyML biosensing systems | Edge AI + Open Source`

**Saturday — Deep Work: Hardware Procurement**
- **Time:** 3.0 hrs
- **Activity:** Order all Phase 1 hardware from Amazon India / Robu.in
- **Deliverable:** Orders placed, tracking numbers saved, delivery calendar set
- **Order list:** Arduino Nano 33 BLE Sense (₹3,200), MPU-6050 (₹220), MAX30102 (₹280), breadboard+jumpers (₹400), AD8232 (₹450), STM32 Nucleo-F401RE (₹2,200)

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Write `week01_retrospective.md`. Plan Week 2. Rest.
- **Deliverable:** Retro doc + Week 2 schedule in calendar

#### Weekly Checklist
- [ ] MIT OCW 6.003 Lectures 1-2 watched and noted
- [ ] Conda environment working
- [ ] GitHub repo `edge-ml-pivot` public with README
- [ ] NeuroKit2 forked, ECG module structure documented
- [ ] TFLite Micro CONTRIBUTING.md read
- [ ] Datasets downloaded
- [ ] Hardware ordered
- [ ] LinkedIn optimized

---

## WEEK 2: Signal Processing + First ECG Pipeline
### Theme: "I can read a heartbeat. I can filter noise. I'm learning in public."

#### Learning Objectives
- [ ] MIT OCW 6.003: Lectures 3-4 (convolution, Fourier transform intuition)
- [ ] NeuroKit2 ECG tutorial: https://neuropsychology.github.io/NeuroKit/examples/ecg/
- [ ] Understand: Butterworth filter, 50Hz notch filter (India powerline)

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `neuropsychology/NeuroKit` | File first issue: "Documentation request: batch ECG processing example" | Entry | Engagement with maintainers |

#### Day-by-Day Schedule

**Monday — Study: Digital Filters**
- **Time:** 1.5 hrs
- **Activity:** Read scipy.signal docs on `butter`, `sosfilt`, `iirnotch`. Read NeuroKit2 ECG processing docs.
- **Deliverable:** `learning_log/week02_filters.md` with filter design notes
- **Key concept:** Second-order sections (SOS) are numerically stable for high-order filters.

**Tuesday — Implement: ECG Filter Bank**
- **Time:** 1.5 hrs
- **Activity:** Write `week2/ecg_filter_bank.py` — bandpass 0.5-40Hz + 50Hz notch
- **Deliverable:** Working filter script + visualization of raw vs filtered ECG
- **Code:** See plan_cc.md Week 1 Day 2 for full template

**Wednesday — OSS: Issue Filed + Mailing List**
- **Time:** 1.5 hrs
- **Activity:** File issue on NeuroKit2 requesting batch processing example. Join TFLite Micro mailing list.
- **Deliverable:** Issue link saved + confirmation email from mailing list
- **Issue template:** Be specific. "I processed 50 MIT-BIH records and want to export HRV features to CSV. An example would help."

**Thursday — Integrate: R-Peak Detection**
- **Time:** 1.5 hrs
- **Activity:** Implement Pan-Tompkins-inspired R-peak detection using scipy + NeuroKit2
- **Deliverable:** `week2/ecg_peaks.py` with detected peaks + RR interval calculation
- **Metrics:** SDNN, RMSSD, pNN50 computed for first 5 MIT-BIH records

**Friday — Content: First LinkedIn Post**
- **Time:** 1.0 hr
- **Activity:** Publish Week 1-2 learnings. Attach ECG visualization.
- **Deliverable:** Live LinkedIn post + screenshot saved
- **Post template:** See plan_cc.md Section 11

**Saturday — Deep Work: HRV Feature Extraction**
- **Time:** 3.0 hrs
- **Activity:** Batch process 20 MIT-BIH records, extract all HRV features, export to CSV
- **Deliverable:** `week2/hrv_batch_features.csv` + analysis script committed
- **Libraries:** neurokit2 `hrv_time`, `hrv_frequency`, `hrv_nonlinear`

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro + plan Week 3. Read ahead: PyTorch quantization tutorial.
- **Deliverable:** `week02_retrospective.md` + calendar blocks for Week 3

#### Weekly Checklist
- [ ] MIT OCW 6.003 Lectures 3-4 completed
- [ ] ECG filter bank working
- [ ] NeuroKit2 issue filed
- [ ] TFLite Micro mailing list joined
- [ ] HRV features extracted from 20 records
- [ ] First LinkedIn post published
- [ ] Code committed to GitHub

---

## WEEK 3: 1D-CNN Architecture + Deep Learning for Signals
### Theme: "My NLP intuition transfers. A sequence is a sequence."

#### Learning Objectives
- [ ] Fast.ai Part 1, Lesson 1: Practical Deep Learning (free at fast.ai)
- [ ] Understand 1D convolution: kernel slides over time axis, detects local patterns
- [ ] Understand why dilated convolutions increase receptive field

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `neuropsychology/NeuroKit` | Comment on 2 existing issues with helpful technical input | Entry | Build karma |
| `tensorflow/tflite-micro` | Read 5 closed PRs to understand review style | Entry | Learn OSS norms |

#### Day-by-Day Schedule

**Monday — Study: 1D Convolutions**
- **Time:** 1.5 hrs
- **Activity:** Fast.ai Lesson 1 + PyTorch 1D Conv tutorial
- **Deliverable:** `learning_log/week03_1dcnn.md` with your own diagram of kernel sliding over ECG
- **Key insight:** A 1D conv kernel of size 25 on 360Hz ECG covers ~70ms — enough to capture QRS morphology.

**Tuesday — Implement: TinyECGNet v0.1**
- **Time:** 1.5 hrs
- **Activity:** Build first 1D-CNN for ECG binary classification (Normal vs Arrhythmia)
- **Deliverable:** `week3/tiny_ecg_net.py` — model trains to >75% accuracy on MIT-BIH
- **Constraint:** Keep parameters <50K for edge deployment later
- **Architecture:** Conv1d(1→16→32→64) + GlobalAvgPool + Linear(64→2)

**Wednesday — OSS: NeuroKit2 Community Engagement**
- **Time:** 1.5 hrs
- **Activity:** Comment helpfully on 2 open issues. Read 3 merged PRs to learn patterns.
- **Deliverable:** 2 comment links saved + notes on PR quality
- **Rule:** Never comment "+1". Always add technical value or a reproducible suggestion.

**Thursday — Integrate: Kaggle Training Pipeline**
- **Time:** 1.5 hrs
- **Activity:** Port model to Kaggle notebook, train on GPU, log metrics
- **Deliverable:** Kaggle notebook saved + training curves downloaded
- **Notebook name:** `edge-ml-week3-ecg-cnn`
- **Log:** Accuracy, loss, parameter count, model size

**Friday — Content: Twitter/X Launch**
- **Time:** 1.0 hr
- **Activity:** Create/pivot Twitter to tech focus. Follow key accounts. First tweet.
- **Deliverable:** Live Twitter with 10+ relevant follows + first tweet
- **Accounts to follow:** @tinyMLsummit, @EdgeImpulse, @hanlab_mit, @ApacheTVM, @ggerganov

**Saturday — Deep Work: Model Analysis**
- **Time:** 3.0 hrs
- **Activity:** Analyze model: receptive field, parameter count per layer, FLOPs
- **Deliverable:** `week3/model_analysis.md` — layer-by-layer breakdown
- **Tools:** `torchinfo`, manual calculation of receptive field

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro + plan. Read ahead: quantization basics.
- **Deliverable:** `week03_retrospective.md`

#### Weekly Checklist
- [ ] Fast.ai Lesson 1 completed
- [ ] TinyECGNet trains on MIT-BIH (>75% acc)
- [ ] 2 NeuroKit2 issues commented on
- [ ] 3 TFLite Micro PRs studied
- [ ] Kaggle notebook created
- [ ] Twitter launched
- [ ] Model analysis document written

---

## WEEK 4: Quantization Basics + OSS First PR Prep
### Theme: "Make it smaller. Make it faster. Document everything."

#### Learning Objectives
- [ ] PyTorch Quantization Tutorial: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
- [ ] Understand: INT8 math, scale, zero-point, symmetric vs asymmetric
- [ ] Understand: per-tensor vs per-channel quantization

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `neuropsychology/NeuroKit` | **FIRST PR:** Add batch ECG processing example notebook | Easy-Medium | PR submitted |

#### Day-by-Day Schedule

**Monday — Study: Quantization Theory**
- **Time:** 1.5 hrs
- **Activity:** PyTorch quantization tutorial + "Deep Compression" paper (Han et al., 2016)
- **Deliverable:** `learning_log/week04_quantization.md` with math notes
- **Key formula:** `x_quant = round(x / scale) + zero_point`

**Tuesday — Implement: PTQ on TinyECGNet**
- **Time:** 1.5 hrs
- **Activity:** Apply post-training INT8 quantization to Week 3 model
- **Deliverable:** `week4/ptq_baseline.py` + benchmark (FP32 vs INT8 size, accuracy drop)
- **Measure:** Model size in KB, accuracy before/after, inference latency

**Wednesday — OSS: First PR Draft**
- **Time:** 1.5 hrs
- **Activity:** Create Jupyter notebook for NeuroKit2 batch ECG processing. Follow repo style.
- **Deliverable:** Notebook in your fork, ready for PR submission
- **Content:** Load 10 MIT-BIH records → clean → detect peaks → compute HRV → export CSV

**Thursday — Integrate: QAT Introduction**
- **Time:** 1.5 hrs
- **Activity:** Implement Quantization-Aware Training (QAT) on TinyECGNet
- **Deliverable:** `week4/qat_first.py` — 5 epochs of QAT, compare to PTQ
- **Observation:** QAT should recover 1-3% accuracy vs PTQ

**Friday — Content: LinkedIn Post #2**
- **Time:** 1.0 hr
- **Activity:** Post quantization benchmark results (text + table image)
- **Deliverable:** Live post with engagement
- **Hook:** "I shrank my neural network 10x. Here's what broke and what didn't."

**Saturday — Deep Work: NeuroKit2 PR Submission**
- **Time:** 3.0 hrs
- **Activity:** Finalize notebook, add README instructions, submit PR
- **Deliverable:** PR link saved + tweeted about it
- **PR checklist:** Clear title, description with motivation, example output, tested locally

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Respond to any PR feedback. Plan Week 5.
- **Deliverable:** `week04_retrospective.md`

#### Weekly Checklist
- [ ] PyTorch quantization tutorial completed
- [ ] PTQ benchmark: FP32 vs INT8
- [ ] QAT implemented and compared
- [ ] NeuroKit2 **PR SUBMITTED**
- [ ] LinkedIn post #2 published
- [ ] All code committed

---

## WEEK 5: Arduino Arrival + TFLite Micro First Contact
### Theme: "My model meets silicon. The real learning begins."

#### Learning Objectives
- [ ] TFLite Micro docs: https://www.tensorflow.org/lite/microcontrollers
- [ ] Understand: static tensor arena, no malloc, no OS
- [ ] Read: `tensorflow/tflite-micro` Arduino examples

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `tensorflow/tflite-micro` | File issue: "Documentation gap: Arduino Nano 33 BLE Sense IMU integration example" | Easy | Real usage signal |
| `neuropsychology/NeuroKit` | Respond to feedback on Week 4 PR, update if needed | Easy | PR merged (hopefully) |

#### Day-by-Day Schedule

**Monday — Study: TFLite Micro Architecture**
- **Time:** 1.5 hrs
- **Activity:** Read TFLite Micro "Overview" + "Get started with microcontrollers" docs
- **Deliverable:** `learning_log/week05_tflite_micro.md` — notes on arena, interpreter, operator support
- **Key concept:** No dynamic memory. All tensors pre-allocated in a static arena.

**Tuesday — Implement: ONNX → TFLite Conversion**
- **Time:** 1.5 hrs
- **Activity:** Convert quantized TinyECGNet to TFLite format
- **Deliverable:** `week5/ecg_model.tflite` file + conversion script
- **Pipeline:** PyTorch → ONNX → TensorFlow → TFLite INT8

**Wednesday — OSS: TFLite Micro Issue + PR Follow-up**
- **Time:** 1.5 hrs
- **Activity:** File TFLite Micro issue about IMU example gap. Check NeuroKit2 PR status.
- **Deliverable:** Issue link + PR status updated
- **If PR needs changes:** Make them immediately.

**Thursday — Integrate: Arduino Setup**
- **Time:** 1.5 hrs
- **Activity:** Install Arduino IDE 2.x, connect Nano 33 BLE Sense, flash Blink
- **Deliverable:** Working Arduino with blinking LED
- **Troubleshoot:** If board not recognized, check driver, try different USB cable

**Friday — Content: LinkedIn Post #3 (Hardware Arrival)**
- **Time:** 1.0 hr
- **Activity:** Post photo of hardware setup (no face). Share excitement + plan.
- **Deliverable:** Live post
- **Hook:** "₹4,000 of hardware just arrived. Time to make it intelligent."

**Saturday — Deep Work: TFLite C Array + First Deployment**
- **Time:** 3.0 hrs
- **Activity:** Convert TFLite model to C array (`xxd -i`), write minimal Arduino inference sketch
- **Deliverable:** Arduino sketch compiles and loads (even if inference is dummy data)
- **Script:** `xxd -i model.tflite > model_data.h`

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Hardware inventory check. Plan Week 6.
- **Deliverable:** `week05_retrospective.md`

#### Weekly Checklist
- [ ] TFLite Micro architecture understood
- [ ] Model converted to .tflite <50KB
- [ ] TFLite Micro issue filed
- [ ] NeuroKit2 PR responded to
- [ ] Arduino Nano connected and blinking
- [ ] First TFLite Micro sketch compiles

---

## WEEK 6: IMU Gesture + Edge Impulse Introduction
### Theme: "Motion is data. Gestures are classification problems."

#### Learning Objectives
- [ ] Edge Impulse Academy: "Continuous motion recognition" tutorial (free)
- [ ] Understand I2C protocol: how sensors talk to microcontrollers
- [ ] Read MPU-6050 datasheet (first 10 pages — registers, sensitivity)

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `edgeimpulse/example-standalone-inferencing` | Star, fork, read 3 examples | Entry | Understand Edge Impulse C++ library structure |
| `neuropsychology/NeuroKit` | If PR merged: celebrate + tweet. If not: ping maintainers politely. | Entry | PR closure |

#### Day-by-Day Schedule

**Monday — Study: Edge Impulse + IMU**
- **Time:** 1.5 hrs
- **Activity:** Edge Impulse Academy tutorial + MPU-6050 datasheet skim
- **Deliverable:** `learning_log/week06_imu.md` with I2C basics and register map notes
- **Key concept:** SDA (data) + SCL (clock) = I2C. Pull-up resistors required.

**Tuesday — Implement: IMU Data Collection**
- **Time:** 1.5 hrs
- **Activity:** Wire MPU-6050 to Arduino. Flash data collection sketch. Capture first readings.
- **Deliverable:** Serial output showing accel + gyro values
- **Wiring:** VCC→3.3V, GND→GND, SDA→A4, SCL→A5

**Wednesday — OSS: Edge Impulse SDK Read**
- **Time:** 1.5 hrs
- **Activity:** Read `example-standalone-inferencing` source. Find where model is loaded and inference runs.
- **Deliverable:** `oss_notes/edge_impulse_structure.md` — map of key files
- **Files:** `main.cpp`, `model-parameters/`, `tflite-model/`

**Thursday — Integrate: Gesture Dataset Creation**
- **Time:** 1.5 hrs
- **Activity:** Collect 5 gestures × 20 samples each. Label and save to CSV.
- **Deliverable:** `week6/gesture_data.csv` + collection protocol documented
- **Gestures:** flick_up, flick_down, shake, circle, stationary

**Friday — Content: Reddit Lurking + First Comment**
- **Time:** 1.0 hr
- **Activity:** Join r/embedded, r/MachineLearning. Make 1 helpful technical comment.
- **Deliverable:** Comment link saved
- **Rule:** No self-promotion. Add value.

**Saturday — Deep Work: Edge Impulse Project Creation**
- **Time:** 3.0 hrs
- **Activity:** Create Edge Impulse account, start "gesture-tinyml" project, upload CSV data
- **Deliverable:** Edge Impulse project with raw data, first impulse designed
- **Process:** Upload CSV → create impulse (spectral features → NN classifier) → train

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 7.
- **Deliverable:** `week06_retrospective.md`

#### Weekly Checklist
- [ ] Edge Impulse tutorial completed
- [ ] MPU-6050 wired and reading data
- [ ] Edge Impulse SDK structure documented
- [ ] 100 gesture samples collected
- [ ] First Reddit comment made
- [ ] Edge Impulse project created

---

## WEEK 7: Gesture Classifier + ONNX Runtime Introduction
### Theme: "Same model, different runtimes. The runtime is the interface to silicon."

#### Learning Objectives
- [ ] ONNX Runtime docs: https://onnxruntime.ai/docs/
- [ ] Understand execution providers: CPU (ARM NEON), CUDA, etc.
- [ ] Understand graph optimization: constant folding, fusion

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `microsoft/onnxruntime` | Star, read CONTRIBUTING.md + 3 closed PRs in `mobile/` label | Entry | Understand mobile/embedded contribution path |
| `neuropsychology/NeuroKit` | If PR merged, write a Twitter thread about the experience | Entry | Content from OSS |

#### Day-by-Day Schedule

**Monday — Study: ONNX Runtime Architecture**
- **Time:** 1.5 hrs
- **Activity:** Read ONNX Runtime "Overview" + "Performance tuning" docs
- **Deliverable:** `learning_log/week07_onnx.md` with notes on execution providers
- **Key concept:** `CPUExecutionProvider` uses ARM NEON on Pi 5 for 4x vectorized ops.

**Tuesday — Implement: Train Gesture Classifier**
- **Time:** 1.5 hrs
- **Activity:** Train 1D-CNN on gesture data (Kaggle or local). Target >85% accuracy.
- **Deliverable:** `week7/gesture_model.pt` + training script
- **Architecture:** Conv1d(6→16→32) + GAP → similar to ECG model but input channels=6

**Wednesday — OSS: ONNX Runtime Issue Research**
- **Time:** 1.5 hrs
- **Activity:** Browse ONNX Runtime issues tagged `mobile` and `good first issue`. Find 2 that interest you.
- **Deliverable:** `oss_notes/onnxruntime_issue_candidates.md` with issue numbers and notes
- **Goal:** Understand what kinds of problems the project needs help with.

**Thursday — Integrate: ONNX Export + Runtime Inference**
- **Time:** 1.5 hrs
- **Activity:** Export gesture model to ONNX. Run inference with ONNX Runtime Python.
- **Deliverable:** `week7/gesture_model.onnx` + `onnx_infer.py` script
- **Measure:** Latency per inference (will be fast on CPU for this tiny model)

**Friday — Content: LinkedIn Post #4**
- **Time:** 1.0 hr
- **Activity:** Post about gesture classifier + ONNX Runtime comparison
- **Deliverable:** Live post with model architecture diagram

**Saturday — Deep Work: TFLite Micro Gesture Deployment**
- **Time:** 3.0 hrs
- **Activity:** Convert gesture model to TFLite INT8, deploy to Arduino, test live
- **Deliverable:** Arduino running live gesture classification (move device → see label)
- **Celebrate:** This is your first end-to-end deployed ML model on hardware!

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Video/screen record the gesture demo for later content. Plan Week 8.
- **Deliverable:** `week07_retrospective.md` + demo recording saved

#### Weekly Checklist
- [ ] ONNX Runtime architecture understood
- [ ] Gesture classifier trained (>85%)
- [ ] 2 ONNX Runtime mobile issues identified
- [ ] ONNX model exported and running
- [ ] LinkedIn post published
- [ ] **Gesture model deployed live on Arduino**

---

## WEEK 8: Benchmarking + First OSS PR (Any Repo)
### Theme: "Numbers are credibility. Benchmarks are your currency."

#### Learning Objectives
- [ ] Read: "How to benchmark deep learning models" (any good blog post)
- [ ] Understand: warm-up, median vs mean, p99 latency, throughput
- [ ] Harvard CS249r: Start Unit 1 (free audit on edX)

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `edgeimpulse/example-standalone-inferencing` | **PR:** Add gesture classifier example with MPU-6050 wiring diagram | Easy | PR submitted |

#### Day-by-Day Schedule

**Monday — Study: Benchmarking Methodology**
- **Time:** 1.5 hrs
- **Activity:** Read benchmarking best practices + start Harvard CS249r Unit 1
- **Deliverable:** `learning_log/week08_benchmarking.md` + CS249r notes
- **Key rule:** Always warm up 10 iterations before measuring. Report median + p99, not just mean.

**Tuesday — Implement: Arduino Benchmark Sketch**
- **Time:** 1.5 hrs
- **Activity:** Write sketch that runs 100 inferences, reports min/median/max latency
- **Deliverable:** `week8/benchmark.ino` + results table
- **Metrics:** Inference latency in microseconds, model size, arena size

**Wednesday — OSS: Edge Impulse PR Draft**
- **Time:** 1.5 hrs
- **Activity:** Create minimal example: MPU-6050 → Edge Impulse library → gesture output
- **Deliverable:** Example code in your fork, ready for PR
- **Include:** README with wiring diagram, expected output, tested hardware list

**Thursday — Integrate: Pi 5 Benchmark (ONNX Runtime)**
- **Time:** 1.5 hrs
- **Activity:** Run gesture model on Pi 5 (or laptop for now if Pi not set up) with ONNX Runtime
- **Deliverable:** `week8/pi5_benchmark.py` + comparison table (Arduino vs Pi 5)
- **Script:** Warm-up + 100 inferences + stats

**Friday — Content: LinkedIn Post #5 (Benchmarks)**
- **Time:** 1.0 hr
- **Activity:** Post benchmark table: Arduino vs Pi 5, TFLite vs ONNX Runtime
- **Deliverable:** Live post with markdown table image
- **Hook:** "38ms on a ₹3,200 chip. 2ms on a ₹7,500 board. The same model."

**Saturday — Deep Work: Edge Impulse PR Submission**
- **Time:** 3.0 hrs
- **Activity:** Polish example, write comprehensive README, submit PR
- **Deliverable:** PR link + tweet
- **Quality bar:** Would a beginner be able to follow your README and get it working?

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 9.
- **Deliverable:** `week08_retrospective.md`

#### Weekly Checklist
- [ ] Benchmarking methodology learned
- [ ] CS249r Unit 1 started
- [ ] Arduino benchmark sketch running
- [ ] Pi 5 benchmark script ready
- [ ] **Edge Impulse PR SUBMITTED**
- [ ] LinkedIn benchmark post published

---

## WEEK 9: Knowledge Distillation Theory + Hardware-Aware Design
### Theme: "A teacher teaches a student. Both are neural networks."

#### Learning Objectives
- [ ] MIT 6.S965 EfficientML.ai: Lecture on Knowledge Distillation (free on YouTube)
- [ ] Understand: soft labels, temperature scaling, intermediate distillation
- [ ] Read: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `tensorflow/tflite-micro` | Comment on 1 issue with technical analysis | Entry | Build credibility |
| `neuropsychology/NeuroKit` | If PR still open, address feedback. If merged, start second PR idea. | Entry | Maintain momentum |

#### Day-by-Day Schedule

**Monday — Study: Knowledge Distillation**
- **Time:** 1.5 hrs
- **Activity:** Hinton paper + MIT 6.S965 lecture
- **Deliverable:** `learning_log/week09_kd.md` with T=4, alpha=0.7 explained in your own words
- **Key formula:** `L_KD = alpha * T² * KL(softmax(teacher/T), softmax(student/T)) + (1-alpha) * CE`

**Tuesday — Implement: Teacher Model (Large ECG Net)**
- **Time:** 1.5 hrs
- **Activity:** Build larger ECG model (~100K params) as teacher. Train to >88% accuracy.
- **Deliverable:** `week9/teacher_model.pt` + training log
- **Architecture:** Dilated convolutions + attention gate

**Wednesday — OSS: TFLite Micro Community**
- **Time:** 1.5 hrs
- **Activity:** Read 5 recent issues. Comment on 1 with reproduction help or technical insight.
- **Deliverable:** Comment link + notes on community norms
- **Focus:** Issues tagged `arduino` or `help wanted`

**Thursday — Integrate: Student Model Design**
- **Time:** 1.5 hrs
- **Activity:** Design student model (<15K params). Must fit in <30KB INT8.
- **Deliverable:** `week9/student_model.py` — architecture diagram + param count
- **Constraint:** 1/7th the size of teacher, target accuracy within 3%

**Friday — Content: Twitter Thread on KD**
- **Time:** 1.0 hr
- **Activity:** Thread: "How I taught a 15K-parameter model to match a 100K-parameter teacher"
- **Deliverable:** 5-tweet thread + GitHub link

**Saturday — Deep Work: Distillation Training**
- **Time:** 3.0 hrs
- **Activity:** Implement KD loss. Train student with teacher's soft labels.
- **Deliverable:** `week9/kd_training.py` + accuracy comparison table
- **Compare:** Teacher acc, Student from scratch, Student with KD

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 10.
- **Deliverable:** `week09_retrospective.md`

#### Weekly Checklist
- [ ] KD theory learned (Hinton paper + MIT lecture)
- [ ] Teacher model trained (>88%)
- [ ] Student model designed (<15K params)
- [ ] 1 TFLite Micro issue commented
- [ ] KD training implemented
- [ ] Twitter thread published

---

## WEEK 10: Harvard CS249r + QAT Mastery
### Theme: "Train with quantization in mind. Fake quantization is real learning."

#### Learning Objectives
- [ ] Harvard CS249r: Unit 2 — Quantization (free audit)
- [ ] Understand: fake quantization nodes, STE, BatchNorm folding
- [ ] PyTorch FX Graph Mode quantization

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `huggingface/optimum` | Read CONTRIBUTING.md + `examples/` directory | Entry | Understand HuggingFace OSS patterns |
| `neuropsychology/NeuroKit` | **SECOND PR:** Add HRV frequency-domain visualization example | Easy | PR submitted |

#### Day-by-Day Schedule

**Monday — Study: QAT Deep Dive**
- **Time:** 1.5 hrs
- **Activity:** CS249r Unit 2 + PyTorch FX QAT tutorial
- **Deliverable:** `learning_log/week10_qat.md` with BN folding notes
- **Key sequence:** Train → prepare_qat → train with fake quant → convert → INT8

**Tuesday — Implement: QAT on Student Model**
- **Time:** 1.5 hrs
- **Activity:** Apply QAT to KD student model. 20 epochs of QAT training.
- **Deliverable:** `week10/qat_student.py` + INT8 model
- **Compare:** FP32 student, PTQ student, QAT student (size same, accuracy different)

**Wednesday — OSS: Optimum + NeuroKit2 PR**
- **Time:** 1.5 hrs
- **Activity:** Study Optimum examples. Draft NeuroKit2 HRV viz PR.
- **Deliverable:** `oss_notes/optimum_structure.md` + draft notebook in NeuroKit fork
- **Optimum focus:** How they export to ONNX and quantize — learn the patterns

**Thursday — Integrate: Full Benchmark Table**
- **Time:** 1.5 hrs
- **Activity:** Compile master benchmark: FP32 teacher, FP32 student, PTQ, QAT, KD+PTQ, KD+QAT
- **Deliverable:** `week10/master_benchmark.md` — 6 rows, 5 metrics each
- **Metrics:** Size (KB), Accuracy (%), Latency (ms if on Pi), Method

**Friday — Content: LinkedIn Post #6**
- **Time:** 1.0 hr
- **Activity:** Post master benchmark table. "Every quantization method I tried on one model."
- **Deliverable:** Live post with table image

**Saturday — Deep Work: NeuroKit2 PR + Optimum Study**
- **Time:** 3.0 hrs
- **Activity:** Finalize HRV viz notebook for NeuroKit2. Read Optimum quantization source.
- **Deliverable:** NeuroKit2 PR submitted + `optimum_quantization_notes.md`

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 11.
- **Deliverable:** `week10_retrospective.md`

#### Weekly Checklist
- [ ] CS249r Unit 2 completed
- [ ] QAT implemented on student model
- [ ] 6-method benchmark table complete
- [ ] Optimum structure documented
- [ ] **NeuroKit2 SECOND PR SUBMITTED**
- [ ] LinkedIn post published

---

## WEEK 11: STM32 Introduction + CMSIS-NN Awareness
### Theme: "ARM Cortex-M4. Bare metal. No OS. This is real embedded."

#### Learning Objectives
- [ ] Read: ARM Cortex-M4 Technical Reference Manual (first 50 pages)
- [ ] Understand: Flash vs SRAM, bus matrix, clock domains
- [ ] CMSIS-NN: Read README at `ARM-software/CMSIS-NN`

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `ARM-software/CMSIS-NN` | Star, read `Source/ConvolutionFunctions/` code | Entry | Understand optimized kernel structure |
| `tensorflow/tflite-micro` | File issue: "Request: STM32 Nucleo-F401RE bare-metal example with UART output" | Easy | Real hardware request |

#### Day-by-Day Schedule

**Monday — Study: ARM Architecture**
- **Time:** 1.5 hrs
- **Activity:** Cortex-M4 TRM + CMSIS-NN README
- **Deliverable:** `learning_log/week11_arm.md` with Flash/SRAM/diagram notes
- **Key numbers:** STM32F401RE: 512KB Flash, 96KB SRAM, 84MHz max

**Tuesday — Implement: STM32CubeIDE Setup**
- **Time:** 1.5 hrs
- **Activity:** Download STM32CubeIDE (free). Create first project for Nucleo-F401RE. Blink LED.
- **Deliverable:** Working STM32 project with blinking LED
- **Download:** https://www.st.com/en/development-tools/stm32cubeide.html

**Wednesday — OSS: CMSIS-NN Code Read**
- **Time:** 1.5 hrs
- **Activity:** Read `arm_convolve_s8.c` line by line. Document what you understand.
- **Deliverable:** `oss_notes/cmsis_nn_convolve_notes.md` — function walkthrough
- **Key insight:** SIMD DSP instructions process 4 INT8 MACs per cycle.

**Thursday — Integrate: STM32 + UART**
- **Time:** 1.5 hrs
- **Activity:** Implement UART printf on STM32. Connect USB-TTL, see output on PC.
- **Deliverable:** STM32 sending "Hello STM32" to serial terminal
- **Baud rate:** 9600. Use PuTTY (Windows) or screen (Linux/Mac).

**Friday — Content: Reddit Post Draft**
- **Time:** 1.0 hr
- **Activity:** Draft first Reddit effortpost: "[P] My first month of TinyML — what I built and what broke"
- **Deliverable:** Draft in `content/week11_reddit_draft.md`

**Saturday — Deep Work: STM32CubeAI Exploration**
- **Time:** 3.0 hrs
- **Activity:** Install STM32CubeAI extension in CubeIDE. Convert a simple model.
- **Deliverable:** `week11/stm32cubeai_first_try.md` — what worked, what didn't
- **Note:** STM32CubeAI generates C code from Keras/TFLite models.

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 12.
- **Deliverable:** `week11_retrospective.md`

#### Weekly Checklist
- [ ] ARM Cortex-M4 architecture understood
- [ ] STM32CubeIDE installed, LED blinking
- [ ] CMSIS-NN `arm_convolve_s8.c` read and noted
- [ ] TFLite Micro STM32 issue filed
- [ ] UART working on STM32
- [ ] STM32CubeAI explored
- [ ] Reddit post drafted

---

## WEEK 12: Phase 1 Consolidation + Content Blitz
### Theme: "Three months of work. One public story."

#### Learning Objectives
- [ ] Review all 11 weeks. Fill gaps in learning log.
- [ ] Complete Harvard CS249r Unit 3 (if not done)
- [ ] Read: "MCUNet: Tiny Deep Learning on IoT Devices" (NeurIPS 2020)

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `neuropsychology/NeuroKit` | Ensure both PRs are merged or in final review | Entry | Phase 1 OSS goal: 2 PRs submitted |
| `edgeimpulse/example-standalone-inferencing` | Ensure Week 8 PR is merged or addressed | Entry | 3rd PR submitted |

#### Day-by-Day Schedule

**Monday — Study: MCUNet Paper**
- **Time:** 1.5 hrs
- **Activity:** Read MCUNet paper. Take notes on two-stage NAS.
- **Deliverable:** `learning_log/week12_mcunet.md` with NAS intuition
- **Key idea:** Search architecture space first, then find optimal for given memory/latency.

**Tuesday — Implement: Gap Fill**
- **Time:** 1.5 hrs
- **Activity:** Complete any unfinished Week 1-11 implementation tasks
- **Deliverable:** All repos clean, all code running, READMEs updated

**Wednesday — OSS: PR Follow-up Blitz**
- **Time:** 1.5 hrs
- **Activity:** Check ALL open PRs. Respond to feedback. Fix issues. Push updates.
- **Deliverable:** All PRs updated. Zero stale contributions.

**Thursday — Integrate: Phase 1 Portfolio Polish**
- **Time:** 1.5 hrs
- **Activity:** Update main `edge-ml-pivot` README with all Phase 1 results
- **Deliverable:** Polished README with architecture diagram, benchmark table, hardware photos
- **Sections:** About, Architecture, Results, Hardware, Repos, Articles

**Friday — Content: Reddit Post Published**
- **Time:** 1.0 hr
- **Activity:** Publish Week 11 draft to r/MachineLearning or r/embedded
- **Deliverable:** Live Reddit post + monitor comments for 2 hours
- **Format:** [P] tag, technical substance, GitHub link, honest failures

**Saturday — Deep Work: Phase 1 Retrospective Article**
- **Time:** 3.0 hrs
- **Activity:** Write long-form article: "From Data Scientist to Edge AI: 3 Months of Open Source"
- **Deliverable:** Article draft for Substack/Medium
- **Length:** 1500-2000 words. Include benchmark tables, hardware photos, OSS PR links.

**Sunday — Review & Plan Phase 2**
- **Time:** 1.5 hrs
- **Activity:** Full Phase 1 retro. Set Phase 2 goals. Rest.
- **Deliverable:** `phase1_retrospective.md` + `phase2_goals.md`

#### Phase 1 Final Checklist
- [ ] 2+ OSS PRs submitted (NeuroKit2 ×2, Edge Impulse ×1)
- [ ] 1+ OSS PR merged (minimum viable)
- [ ] ECG pipeline: filter → peaks → HRV → features
- [ ] Gesture classifier: trained → quantized → deployed on Arduino
- [ ] Benchmark table: FP32 vs PTQ vs QAT
- [ ] KD experiment: teacher vs student
- [ ] STM32: LED blink + UART working
- [ ] Harvard CS249r: Units 1-3
- [ ] 6+ LinkedIn posts published
- [ ] 3+ Twitter threads published
- [ ] 1 Reddit effortpost published
- [ ] 1 long-form article drafted
- [ ] GitHub contribution graph: green every week

---

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: HARDWARE DEPTH + OSS GROWTH (WEEKS 13-24)
# Months 4-6 | Goal: STM32 deployment, CMSIS-NN, 5 OSS PRs total
# ═══════════════════════════════════════════════════════════════════

> **Phase 2 Philosophy:** Hardware is no longer intimidating. OSS is no longer foreign. Now we go deeper on both.

---

## WEEK 13: STM32 Model Deployment + CMSIS-NN Integration
### Theme: "The same INT8 model. Different silicon. Different speed."

#### Learning Objectives
- [ ] STM32CubeAI: Full documentation read
- [ ] Understand: how CMSIS-NN accelerates TFLite Micro (backend integration)
- [ ] Domain 5 Topic 5.2: CMSIS-NN kernel deep dive

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `ARM-software/CMSIS-NN` | File issue: "Documentation request: INT8 convolution example with full pipeline" | Easy | Community need identified |
| `tensorflow/tflite-micro` | Comment on issue with STM32-specific insight from your experience | Entry | Technical credibility |

#### Day-by-Day Schedule

**Monday — Study: STM32CubeAI Pipeline**
- **Time:** 1.5 hrs
- **Activity:** Read STM32CubeAI user manual. Understand model import → C code generation.
- **Deliverable:** `learning_log/week13_stm32cubeai.md` with pipeline diagram

**Tuesday — Implement: ECG Model on STM32**
- **Time:** 1.5 hrs
- **Activity:** Import KD+INT8 ECG model into STM32CubeAI. Generate C code.
- **Deliverable:** Generated C code + compilation attempt
- **Target:** Model <64KB, RAM <96KB

**Wednesday — OSS: CMSIS-NN Issue + TFLite Micro Comment**
- **Time:** 1.5 hrs
- **Activity:** File CMSIS-NN issue. Comment on TFLite Micro STM32 discussion.
- **Deliverable:** 2 community touchpoints

**Thursday — Integrate: Compile and Flash**
- **Time:** 1.5 hrs
- **Activity:** Compile generated code in STM32CubeIDE. Flash to Nucleo. Measure inference.
- **Deliverable:** Working STM32 inference + latency number
- **Metric:** How many milliseconds at 84MHz?

**Friday — Content: LinkedIn Post #7**
- **Time:** 1.0 hr
- **Activity:** Post STM32 benchmark vs Arduino vs Pi 5 comparison
- **Deliverable:** Live post with 3-platform table

**Saturday — Deep Work: CMSIS-NN vs Naive C Benchmark**
- **Time:** 3.0 hrs
- **Activity:** If possible, compile same model with/without CMSIS-NN. Measure difference.
- **Deliverable:** `week13/cmsis_vs_naive.md` — speedup factor
- **Expected:** 3-5× faster with CMSIS-NN

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 14.
- **Deliverable:** `week13_retrospective.md`

---

## WEEK 14: Pi 5 Setup + ONNX Runtime ARM Optimization
### Theme: "Linux on ARM. This is where edge inference gets real."

#### Learning Objectives
- [ ] Domain 4 Topic 4.2: ONNX Runtime depth
- [ ] Understand ARM NEON: SIMD instructions for 4x throughput
- [ ] Raspberry Pi 5 architecture: Cortex-A76, 4 cores, 4GB RAM

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `microsoft/onnxruntime` | File issue: "Documentation request: Pi 5 (Cortex-A76) optimization guide" | Easy | Real need for Pi 5 users |
| `neuropsychology/NeuroKit` | If new PRs merged, tweet. Start third PR idea. | Easy | Maintain streak |

#### Day-by-Day Schedule

**Monday — Study: Pi 5 + ONNX Runtime**
- **Time:** 1.5 hrs
- **Activity:** Read Pi 5 specs + ONNX Runtime ARM optimization docs
- **Deliverable:** `learning_log/week14_pi5_ort.md` with NEON notes

**Tuesday — Implement: Pi 5 OS Setup**
- **Time:** 1.5 hrs
- **Activity:** Flash Raspberry Pi OS to microSD. Boot Pi 5. SSH in. Install Python env.
- **Deliverable:** Working Pi 5 accessible via SSH

**Wednesday — OSS: ONNX Runtime Issue**
- **Time:** 1.5 hrs
- **Activity:** Research Pi 5 ONNX Runtime setup. File documentation issue with your findings.
- **Deliverable:** Issue with reproduction steps

**Thursday — Integrate: ONNX Runtime on Pi 5**
- **Time:** 1.5 hrs
- **Activity:** Install ONNX Runtime on Pi 5. Run ECG model. Benchmark.
- **Deliverable:** `week14/pi5_ort_benchmark.py` + latency numbers
- **Compare:** Pi 5 vs laptop CPU vs STM32

**Friday — Content: Twitter Thread on Pi 5 ML**
- **Time:** 1.0 hr
- **Activity:** Thread: "What I learned running neural networks on a ₹7,500 computer"
- **Deliverable:** 5-tweet thread

**Saturday — Deep Work: Multi-Vital Sign Monitor (Design)**
- **Time:** 3.0 hrs
- **Activity:** Design system architecture: ECG + PPG → Pi 5 → dashboard
- **Deliverable:** Mermaid diagram in `week14/system_arch.md`

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 15.
- **Deliverable:** `week14_retrospective.md`

---

## WEEK 15: Multi-Vital Pipeline + Streamlit Dashboard
### Theme: "Real-time biosensing. Not in the cloud. On my desk."

#### Learning Objectives
- [ ] Streamlit advanced features: caching, session state, real-time updates
- [ ] Understand: I2C multiple device addressing (ECG + PPG on same bus)
- [ ] Sensor fusion: combining ECG + PPG for SpO2 and HR

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `neuropsychology/NeuroKit` | **THIRD PR:** Add PPG quality assessment utility | Medium | PR submitted |

#### Day-by-Day Schedule

**Monday — Study: Sensor Fusion + I2C**
- **Time:** 1.5 hrs
- **Activity:** Read MAX30102 datasheet + I2C addressing. Understand sensor fusion basics.
- **Deliverable:** `learning_log/week15_sensor_fusion.md`

**Tuesday — Implement: PPG Data Capture on Pi 5**
- **Time:** 1.5 hrs
- **Activity:** Wire MAX30102 to Pi 5 GPIO. Read SpO2 and heart rate.
- **Deliverable:** Python script reading PPG data

**Wednesday — OSS: NeuroKit2 PPG PR Draft**
- **Time:** 1.5 hrs
- **Activity:** Implement PPG quality index in NeuroKit2 style. Write tests.
- **Deliverable:** Code in fork + test file

**Thursday — Integrate: Combined ECG + PPG Dashboard**
- **Time:** 1.5 hrs
- **Activity:** Streamlit app showing both waveforms + computed HR/SpO2/HRV
- **Deliverable:** `week15/dashboard.py` — working local dashboard

**Friday — Content: LinkedIn Post #8**
- **Time:** 1.0 hr
- **Activity:** Post dashboard screenshot + architecture diagram
- **Deliverable:** Live post

**Saturday — Deep Work: NeuroKit2 PR + Dashboard Polish**
- **Time:** 3.0 hrs
- **Activity:** Submit PPG PR. Polish dashboard with historical trends.
- **Deliverable:** PR link + improved dashboard

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 16.
- **Deliverable:** `week15_retrospective.md`

---

## WEEK 16: Edge Impulse EON Tuner + Fall Detection
### Theme: "AutoML for microcontrollers. The machine finds the model for the machine."

#### Learning Objectives
- [ ] Edge Impulse EON Tuner documentation
- [ ] Understand: Neural Architecture Search (NAS) at microcontroller scale
- [ ] MIT 6.S965: NAS lecture (if available)

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `edgeimpulse/example-standalone-inferencing` | **SECOND PR:** Add fall detection example with IMU | Easy | PR submitted |
| `tensorflow/tflite-micro` | File issue with reproduction of any bug you encounter | Easy | Real usage signal |

#### Day-by-Day Schedule

**Monday — Study: NAS + EON Tuner**
- **Time:** 1.5 hrs
- **Activity:** Read EON Tuner docs + MCUNet paper sections on NAS
- **Deliverable:** `learning_log/week16_nas.md`

**Tuesday — Implement: Fall Detection Dataset**
- **Time:** 1.5 hrs
- **Activity:** Collect IMU data for fall vs normal activity. Upload to Edge Impulse.
- **Deliverable:** Edge Impulse project with labeled data

**Wednesday — OSS: Edge Impulse PR Draft**
- **Time:** 1.5 hrs
- **Activity:** Build standalone C++ fall detection example. Write README.
- **Deliverable:** Example in fork

**Thursday — Integrate: EON Tuner Optimization**
- **Time:** 1.5 hrs
- **Activity:** Run EON Tuner on fall detection. Compare default vs optimized.
- **Deliverable:** Benchmark: default model vs EON-tuned model

**Friday — Content: LinkedIn Post #9**
- **Time:** 1.0 hr
- **Activity:** Post about EON Tuner results. "The machine designed a model for my ₹220 sensor."
- **Deliverable:** Live post

**Saturday — Deep Work: PR Submission + Multi-Platform Deploy**
- **Time:** 3.0 hrs
- **Activity:** Submit Edge Impulse PR. Deploy fall model to both Arduino and Pi 5.
- **Deliverable:** PR link + dual-platform demo

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 17.
- **Deliverable:** `week16_retrospective.md`

---

## WEEK 17: Compression Benchmark Study (The Big One)
### Theme: "5 techniques. One model. Real numbers."

#### Learning Objectives
- [ ] Domain 3: Review all compression topics
- [ ] Understand trade-offs deeply enough to explain in an interview
- [ ] AWQ paper (if time permits)

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `huggingface/optimum` | Read quantization source code deeply. File doc improvement issue. | Easy | Engagement with HF ecosystem |
| `mit-han-lab/mcunet` | Star, read, file issue asking for ECG dataset example | Easy | Connect with MIT Han Lab |

#### Day-by-Day Schedule

**Monday — Study: Compression Review**
- **Time:** 1.5 hrs
- **Activity:** Review all compression methods. Prepare benchmark design.
- **Deliverable:** `learning_log/week17_compression_design.md` — experiment protocol

**Tuesday — Implement: Baseline + PTQ**
- **Time:** 1.5 hrs
- **Activity:** Run baseline FP32 and PTQ INT8 on ECG model
- **Deliverable:** Rows 1-2 of benchmark table

**Wednesday — OSS: Optimum + MCUNet**
- **Time:** 1.5 hrs
- **Activity:** Read Optimum quantization internals. File MCUNet issue.
- **Deliverable:** `oss_notes/optimum_quantization_deep.md` + issue link

**Thursday — Integrate: QAT + Pruning**
- **Time:** 1.5 hrs
- **Activity:** Run QAT and structured pruning. Add to benchmark.
- **Deliverable:** Rows 3-4 complete

**Friday — Content: Twitter Thread on Compression**
- **Time:** 1.0 hr
- **Activity:** Thread: "I applied 5 compression techniques to the same ECG model"
- **Deliverable:** 6-tweet thread

**Saturday — Deep Work: KD + Full Table + Article Draft**
- **Time:** 3.0 hrs
- **Activity:** Run KD experiment. Compile final table. Start article.
- **Deliverable:** 5-row benchmark table + article outline

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Week 18.
- **Deliverable:** `week17_retrospective.md`

---

## WEEK 18: Article Publication + Phase 2 Midpoint
### Theme: "Publish or perish. In open source, publish AND perish less."

#### Learning Objectives
- [ ] Review all 17 weeks. Consolidate learning log.
- [ ] Read: "Once-for-All" paper (ICLR 2020)

#### OSS Contribution Target
| Repo | Action | Difficulty | Expected Outcome |
|------|--------|------------|------------------|
| `neuropsychology/NeuroKit` | Ensure 3rd PR merged or close to merge | Entry | 3 PRs in flight |
| `huggingface/optimum` | If doc issue accepted, submit PR | Easy | 4th PR |

#### Day-by-Day Schedule

**Monday — Study: Once-for-All**
- **Time:** 1.5 hrs
- **Activity:** Read OFA paper. Understand supernet + specialization.
- **Deliverable:** `learning_log/week18_ofa.md`

**Tuesday — Implement: Article Finalization**
- **Time:** 1.5 hrs
- **Activity:** Finalize compression benchmark article. Add all tables, diagrams.
- **Deliverable:** Article ready for publish

**Wednesday — OSS: PR Blitz**
- **Time:** 1.5 hrs
- **Activity:** Update all open PRs. Respond to feedback.
- **Deliverable:** Zero stale PRs

**Thursday — Integrate: Publish Article**
- **Time:** 1.5 hrs
- **Activity:** Publish to Medium/Substack. Cross-post to LinkedIn as PDF carousel.
- **Deliverable:** Live article + LinkedIn carousel post

**Friday — Content: LinkedIn Carousel**
- **Time:** 1.0 hr
- **Activity:** Create 8-slide PDF: "5 compression techniques, one model, real numbers"
- **Deliverable:** Carousel posted

**Saturday — Deep Work: GitHub Repo Polish**
- **Time:** 3.0 hrs
- **Activity:** Create dedicated repo: `compression-benchmark`. Beautiful README.
- **Deliverable:** `github.com/YOUR_USERNAME/compression-benchmark` live

**Sunday — Review & Plan**
- **Time:** 1.5 hrs
- **Activity:** Retro. Plan Weeks 19-24.
- **Deliverable:** `week18_retrospective.md`

---

## WEEKS 19-24: Phase 2 Second Half — Pi 5 Mastery + llama.cpp + OSS Growth

> These 6 weeks follow the same daily structure. Each week has a specific focus.

### WEEK 19: llama.cpp Introduction (Edge LLM)
- **Monday:** Study: llama.cpp GGUF format, quantization levels (Q4_K_M, Q5_K_M)
- **Tuesday:** Implement: Build llama.cpp on Pi 5. Download Gemma 3 2B Q4_K_M.
- **Wednesday:** OSS: Read `ggerganov/llama.cpp` source. File documentation issue.
- **Thursday:** Integrate: Run first inference on Pi 5. Measure tokens/sec.
- **Friday:** Content: LinkedIn post about edge LLM on Pi 5
- **Saturday:** Deep Work: Benchmark 3 models (Gemma 2B, Qwen 1.5B, Phi-3 3.8B) on Pi 5
- **Sunday:** Review & Plan

### WEEK 20: Edge RAG System Design
- **Monday:** Study: FAISS for small-scale retrieval. Embedding models for edge.
- **Tuesday:** Implement: Build minimal RAG on Pi 5 (embedding + FAISS + llama.cpp)
- **Wednesday:** OSS: Study `huggingface/peft` LoRA examples. File doc issue.
- **Thursday:** Integrate: End-to-end RAG pipeline working
- **Friday:** Content: Twitter thread: "RAG without cloud — my Pi 5 setup"
- **Saturday:** Deep Work: Medical literature index (PubMed abstracts) for RAG
- **Sunday:** Review & Plan

### WEEK 21: AWQ + INT4 Quantization
- **Monday:** Study: AWQ paper + `casymcc/awq` repo documentation
- **Tuesday:** Implement: AWQ on Qwen 2.5-1.5B. Compare to INT8.
- **Wednesday:** OSS: File issue on AWQ repo: "Edge deployment example request"
- **Thursday:** Integrate: Convert AWQ model to GGUF for llama.cpp
- **Friday:** Content: LinkedIn post on AWQ vs INT8 vs Q4_K_M benchmarks
- **Saturday:** Deep Work: Full edge LLM benchmark: 4 models × 3 quantization levels
- **Sunday:** Review & Plan

### WEEK 22: ONNX Runtime Mobile Deep Dive
- **Monday:** Study: ORT mobile docs. Execution providers for ARM.
- **Tuesday:** Implement: ORT mobile on Android emulator (or documented for future)
- **Wednesday:** OSS: Comment on ORT mobile issue with your ARM benchmark data
- **Thursday:** Integrate: Compare ORT vs TFLite vs llama.cpp for same model
- **Friday:** Content: Article draft: "ONNX Runtime on ARM: A Practitioner's Guide"
- **Saturday:** Deep Work: ORT optimization flags benchmark (all 64 combinations)
- **Sunday:** Review & Plan

### WEEK 23: Apache TVM Introduction
- **Monday:** Study: TVM tutorial "Getting Started with TVM"
- **Tuesday:** Implement: Compile simple model with TVM for CPU target
- **Wednesday:** OSS: Read `apache/tvm` issues. Find `good first issue` or doc gap.
- **Thursday:** Integrate: TVM vs ONNX Runtime benchmark on laptop CPU
- **Friday:** Content: LinkedIn post: "First steps with Apache TVM"
- **Saturday:** Deep Work: TVM Relay IR exploration. Read model import code.
- **Sunday:** Review & Plan

### WEEK 24: Phase 2 Consolidation
- **Monday:** Study: Review all Phase 2 learning logs
- **Tuesday:** Implement: Complete any unfinished hardware/software tasks
- **Wednesday:** OSS: Ensure all Phase 2 PRs merged or updated (target: 5 total PRs)
- **Thursday:** Integrate: Full system demo: sensors → Pi 5 → dashboard → LLM Q&A
- **Friday:** Content: Phase 2 retrospective article
- **Saturday:** Deep Work: GitHub repo `edge-ml-pi5` with full system
- **Sunday:** Review & Plan Phase 3

#### Phase 2 Final Checklist
- [ ] 5+ OSS PRs submitted total (across all repos)
- [ ] 3+ OSS PRs merged
- [ ] STM32 deployment: ECG model running on Nucleo-F401RE
- [ ] Pi 5: ONNX Runtime + llama.cpp + RAG working
- [ ] Edge Impulse: Fall detection on EON Tuner
- [ ] Compression benchmark: 5 techniques, published article
- [ ] TVM: First compilation successful
- [ ] Harvard CS249r: Units 1-6 complete
- [ ] MIT 6.S965: Lectures 1-8
- [ ] 12+ LinkedIn posts
- [ ] 8+ Twitter threads
- [ ] 3+ Reddit posts
- [ ] 2+ long-form articles published

---

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: DOMAIN MASTERY + SUBSTANTIVE OSS (WEEKS 25-36)
# Months 7-9 | Goal: 8 PRs merged, TVM tutorial, RISC-V touch, hero project v1
# ═══════════════════════════════════════════════════════════════════

> **Phase 3 Philosophy:** You're no longer a beginner. PRs should be substantive. Code quality should be production-grade.

---

## WEEKS 25-27: TVM Deep Dive + Pi 5 Auto-Tuning

### WEEK 25: TVM Relay + AutoTVM
- **Monday:** Study: TVM Relay IR documentation. Understand graph representation.
- **Tuesday:** Implement: Import ECG ONNX model into TVM Relay. Compile for `llvm` target.
- **Wednesday:** OSS: File issue on TVM: "Tutorial request: ECG model on Raspberry Pi 5"
- **Thursday:** Integrate: AutoTVM tuning for Pi 5 (Cortex-A76). 500 trials.
- **Friday:** Content: LinkedIn post on TVM compilation pipeline
- **Saturday:** Deep Work: Full TVM benchmark vs ONNX Runtime. Document speedup.
- **Sunday:** Review & Plan

### WEEK 26: TVM Tutorial Creation (OSS Target)
- **Monday:** Study: TVM tutorial format. Read 3 existing tutorials.
- **Tuesday:** Implement: Write complete TVM tutorial for ECG on Pi 5.
- **Wednesday:** OSS: **SUBMIT TVM PR** — Add Pi 5 biosignal tutorial. This is a MAJOR PR.
- **Thursday:** Integrate: Test tutorial on fresh Pi 5 install. Verify every step.
- **Friday:** Content: Twitter thread: "How I got my first PR to Apache TVM"
- **Saturday:** Deep Work: Address PR feedback immediately if any.
- **Sunday:** Review & Plan

### WEEK 27: MetaScheduler + MicroTVM Awareness
- **Monday:** Study: MetaScheduler vs AutoTVM differences
- **Tuesday:** Implement: Run MetaScheduler on same model. Compare results.
- **Wednesday:** OSS: Read MicroTVM docs. File issue: "Documentation gap for STM32F4"
- **Thursday:** Integrate: Benchmark table: untuned vs AutoTVM vs MetaScheduler
- **Friday:** Content: LinkedIn post: "TVM tuning comparison: 3 schedulers, one model"
- **Saturday:** Deep Work: Article: "Apache TVM for Edge AI: A Practical Introduction"
- **Sunday:** Review & Plan

## WEEKS 28-30: RISC-V + Indian Semiconductor Ecosystem

### WEEK 28: RISC-V Fundamentals
- **Monday:** Study: RISC-V ISA spec (free). Understand RV64GC, vector extension.
- **Tuesday:** Implement: Install RISC-V toolchain. Cross-compile "hello world".
- **Wednesday:** OSS: Star `shaktiproject/SHAKTI-SoC`. Read first 20 issues.
- **Thursday:** Integrate: QEMU RISC-V: run compiled binary. Measure cycles.
- **Friday:** Content: LinkedIn post: "Why RISC-V matters for Indian semiconductors"
- **Saturday:** Deep Work: Article draft: "RISC-V for ML Engineers"
- **Sunday:** Review & Plan

### WEEK 29: RISC-V + ML Inference
- **Monday:** Study: RISC-V vector extension (RVV) for ML
- **Tuesday:** Implement: Cross-compile simple matrix multiply to RISC-V. Run in QEMU.
- **Wednesday:** OSS: File issue on `riscv/riscv-gnu-toolchain` with your ML cross-compile notes
- **Thursday:** Integrate: Compare ARM NEON vs RISC-V RVV for same operation
- **Friday:** Content: Twitter thread on RISC-V vs ARM for ML
- **Saturday:** Deep Work: GitHub repo `riscv-ml-benchmark` with QEMU experiments
- **Sunday:** Review & Plan

### WEEK 30: India Semiconductor Ecosystem Content
- **Monday:** Study: ISM 2.0, DLI scheme, IndiaAI Compute Portal
- **Tuesday:** Implement: Apply for IndiaAI Compute (free A100 access)
- **Wednesday:** OSS: File issue or contribute to Indian OSS project (Mindgrove, SHAKTI)
- **Thursday:** Integrate: Research 5 Indian chip startups. Document their tech stacks.
- **Friday:** Content: LinkedIn article: "India's Semiconductor Boom: What ML Engineers Need to Know"
- **Saturday:** Deep Work: Full ecosystem map document
- **Sunday:** Review & Plan

## WEEKS 31-33: Edge Medical RAG (Unique Project)

### WEEK 31: SLM Fine-Tuning for Medical Domain
- **Monday:** Study: QLoRA for small models (rank 4-8). Domain adaptation techniques.
- **Tuesday:** Implement: Fine-tune Qwen 2.5-1.5B on medical Q&A dataset (MedQA subset)
- **Wednesday:** OSS: Read `huggingface/trl` SFT examples. File doc improvement issue.
- **Thursday:** Integrate: Convert fine-tuned model to GGUF for llama.cpp
- **Friday:** Content: LinkedIn post: "Fine-tuning a 1.5B model on a free GPU"
- **Saturday:** Deep Work: Full fine-tuning pipeline documented
- **Sunday:** Review & Plan

### WEEK 32: FAISS + Medical Knowledge Base
- **Monday:** Study: FAISS index types. IVF vs Flat vs HNSW for small scale.
- **Tuesday:** Implement: Build FAISS index of 1000 PubMed abstracts. Embed with MiniLM.
- **Wednesday:** OSS: File issue on `facebookresearch/faiss`: "Small-scale edge deployment tips"
- **Thursday:** Integrate: Retrieval + SLM generation pipeline on Pi 5
- **Friday:** Content: Twitter thread: "Medical RAG on ₹7,500 hardware"
- **Saturday:** Deep Work: End-to-end demo: query → retrieve → generate with citations
- **Sunday:** Review & Plan

### WEEK 33: Streamlit Dashboard + Real-Time Integration
- **Monday:** Study: Streamlit session state, real-time updates, caching
- **Tuesday:** Implement: Dashboard showing live vitals + medical Q&A chatbot
- **Wednesday:** OSS: File issue on `streamlit/streamlit`: "Feature request: real-time sensor widget"
- **Thursday:** Integrate: Full system: sensors → Pi 5 → FAISS → SLM → dashboard
- **Friday:** Content: LinkedIn post with demo GIF (screen recording, no face)
- **Saturday:** Deep Work: Record full demo. Write comprehensive README.
- **Sunday:** Review & Plan

## WEEKS 34-36: Phase 3 Consolidation + MLIR Touch

### WEEK 34: MLIR Introduction
- **Monday:** Study: MLIR Toy Tutorial (official). Understand dialects, lowering, passes.
- **Tuesday:** Implement: Complete Toy tutorial. Build and run.
- **Wednesday:** OSS: File issue on `llvm/llvm-project`: "Tutorial clarification request"
- **Thursday:** Integrate: Read MLIR IR for a simple PyTorch model (via torch-mlir if possible)
- **Friday:** Content: LinkedIn post: "My first steps with MLIR"
- **Saturday:** Deep Work: Article: "MLIR for ML Engineers: What You Actually Need to Know"
- **Sunday:** Review & Plan

### WEEK 35: Substantive OSS PR Push
- **Monday:** Study: Review all open PRs. Identify blockers.
- **Tuesday-Tuesday:** Implement: Address feedback on ALL open PRs.
- **Wednesday:** OSS: **Submit 2nd TVM PR** (MicroTVM doc or small feature)
- **Thursday:** Integrate: Verify all PRs are green (CI passing)
- **Friday:** Content: LinkedIn post celebrating merged PRs
- **Saturday:** Deep Work: GitHub profile README update with all contributions
- **Sunday:** Review & Plan

### WEEK 36: Phase 3 Consolidation
- **Monday:** Study: Review all 12 weeks of learning
- **Tuesday:** Implement: Complete any unfinished system components
- **Wednesday:** OSS: Ensure 8+ PRs submitted, 5+ merged
- **Thursday:** Integrate: Full sovereign health system v1 demo
- **Friday:** Content: Phase 3 retrospective article
- **Saturday:** Deep Work: Hero project repo `sovereign-health-edge-ai` created
- **Sunday:** Review & Plan Phase 4

#### Phase 3 Final Checklist
- [ ] 8+ OSS PRs submitted
- [ ] 5+ OSS PRs merged
- [ ] TVM: Pi 5 tutorial PR submitted (major)
- [ ] RISC-V: QEMU experiments documented
- [ ] Edge Medical RAG: working on Pi 5
- [ ] MLIR: Toy tutorial completed
- [ ] SLM fine-tuned for medical domain
- [ ] 18+ LinkedIn posts total
- [ ] 12+ Twitter threads total
- [ ] 5+ Reddit posts total
- [ ] 4+ long-form articles published
- [ ] GitHub: 4-5 polished public repos

---

# ═══════════════════════════════════════════════════════════════════
# PHASE 4: HERO PROJECT + SENIOR OSS (WEEKS 37-48)
# Months 10-12 | Goal: Career-defining project, 12 PRs, interview-ready portfolio
# ═══════════════════════════════════════════════════════════════════

> **Phase 4 Philosophy:** This is where you ship the project that gets you hired. Everything before was preparation.

---

## WEEKS 37-40: Hero Project Build (Sovereign Health System)

### WEEK 37: System Architecture + STM32 Integration
- **Monday:** Study: System design patterns for embedded ML
- **Tuesday:** Implement: STM32 code for multi-sensor reading (ECG + PPG + IMU)
- **Wednesday:** OSS: File issue on `STMicroelectronics/X-CUBE-AI` with integration feedback
- **Thursday:** Integrate: UART protocol between STM32 and Pi 5 defined
- **Friday:** Content: LinkedIn post: "Designing a sovereign health system — architecture thread"
- **Saturday:** Deep Work: Mermaid architecture diagram + full system spec
- **Sunday:** Review & Plan

### WEEK 38: Pi 5 Compute Layer
- **Monday:** Study: ONNX Runtime + llama.cpp integration patterns
- **Tuesday:** Implement: Multi-model inference scheduler on Pi 5
- **Wednesday:** OSS: File issue on `ggerganov/llama.cpp`: "Feature request: multi-model context switching"
- **Thursday:** Integrate: Sensor data → AFib detector → anomaly alert → LLM explanation
- **Friday:** Content: Twitter thread on system integration challenges
- **Saturday:** Deep Work: Full inference pipeline benchmark
- **Sunday:** Review & Plan

### WEEK 39: Dashboard + User Interface
- **Monday:** Study: Streamlit advanced + Plotly real-time charts
- **Tuesday:** Implement: Real-time waveform display + health score + chat interface
- **Wednesday:** OSS: File issue on `streamlit/streamlit` with real-time rendering feedback
- **Thursday:** Integrate: End-to-end system test. 24-hour continuous run.
- **Friday:** Content: LinkedIn post with dashboard screenshot
- **Saturday:** Deep Work: Stress test: memory leaks, performance degradation, edge cases
- **Sunday:** Review & Plan

### WEEK 40: Documentation + Polish
- **Monday:** Study: Technical writing best practices for GitHub READMEs
- **Tuesday:** Implement: Hero project README: architecture, benchmarks, setup, troubleshooting
- **Wednesday:** OSS: Ensure all hero project dependencies have issues/PRs filed for any gaps
- **Thursday:** Integrate: Demo GIF creation (screen recording, NO FACE)
- **Friday:** Content: LinkedIn post: "The project that took 10 months and ₹17,000"
- **Saturday:** Deep Work: Technical report PDF (arXiv-style, 5-10 pages)
- **Sunday:** Review & Plan

## WEEKS 41-44: OSS Senior Contributions

### WEEK 41: TVM Substantive Contribution
- **Monday:** Study: TVM Relay pass infrastructure
- **Tuesday-Tuesday:** Implement: Write a simple graph optimization pass OR fix a bug
- **Wednesday:** OSS: **SUBMIT TVM CODE PR** (not just docs)
- **Thursday:** Integrate: Test PR on multiple targets
- **Friday:** Content: LinkedIn post: "My first code PR to Apache TVM"
- **Saturday:** Deep Work: Address review feedback
- **Sunday:** Review & Plan

### WEEK 42: ONNX Runtime Mobile Contribution
- **Monday:** Study: ORT mobile execution provider code
- **Tuesday-Tuesday:** Implement: Small optimization or documentation improvement
- **Wednesday:** OSS: **SUBMIT ORT PR**
- **Thursday:** Integrate: Test on actual mobile/ARM hardware
- **Friday:** Content: Twitter thread on ORT contribution
- **Saturday:** Deep Work: Review iteration
- **Sunday:** Review & Plan

### WEEK 43: TFLite Micro Contribution
- **Monday:** Study: TFLite Micro operator implementation
- **Tuesday-Tuesday:** Implement: Fix or small feature
- **Wednesday:** OSS: **SUBMIT TFLITE MICRO PR**
- **Thursday:** Integrate: Test on Arduino and STM32
- **Friday:** Content: LinkedIn post on TFLite Micro contribution
- **Saturday:** Deep Work: Review iteration
- **Sunday:** Review & Plan

### WEEK 44: MLIR Contribution Attempt
- **Monday:** Study: MLIR dialect definition
- **Tuesday-Tuesday:** Implement: Tutorial contribution or test case
- **Wednesday:** OSS: **SUBMIT LLVM/MLIR PR**
- **Thursday:** Integrate: Build and test in LLVM ecosystem
- **Friday:** Content: Article: "How I Got My First PR to LLVM"
- **Saturday:** Deep Work: Review iteration
- **Sunday:** Review & Plan

## WEEKS 45-48: Phase 4 Consolidation + Job Prep Start

### WEEK 45: Resume + Portfolio Optimization
- **Monday:** Study: Resume formats for ML engineers
- **Tuesday:** Implement: Rewrite resume with edge AI focus + real numbers
- **Wednesday:** OSS: GitHub profile README update with contribution stats
- **Thursday:** Integrate: Pin best repos. Create `ABOUT_ME.md` in profile repo.
- **Friday:** Content: LinkedIn post: "10 months of open source — what I built"
- **Saturday:** Deep Work: Full portfolio website (GitHub Pages, free)
- **Sunday:** Review & Plan

### WEEK 46: Interview Question Preparation
- **Monday:** Study: 10 key technical questions (see plan_agy.md Phase 6)
- **Tuesday:** Implement: Write model answers with YOUR project numbers
- **Wednesday:** OSS: Contribute to `DataTalksClub/mlops-zoomcamp` if relevant
- **Thursday:** Integrate: Mock interview practice (record yourself, no face needed)
- **Friday:** Content: LinkedIn post on a specific technical insight
- **Saturday:** Deep Work: Full Q&A document
- **Sunday:** Review & Plan

### WEEK 47: Networking Blitz
- **Monday:** Study: Company research — Sophrosyne, Mindgrove, Netrasemi, etc.
- **Tuesday:** Implement: Send 5 LinkedIn connection requests with personalized notes
- **Wednesday:** OSS: Engage with maintainers of target repos on professional topics
- **Thursday:** Integrate: Attend 1 virtual meetup or watch 1 conference recording
- **Friday:** Content: LinkedIn post tagging company/technology
- **Saturday:** Deep Work: Cold email drafts to 3 startup founders
- **Sunday:** Review & Plan

### WEEK 48: Phase 4 Consolidation
- **Monday:** Study: Review all 48 weeks
- **Tuesday:** Implement: Complete any unfinished hero project items
- **Wednesday:** OSS: Ensure 12+ PRs submitted, 8+ merged
- **Thursday:** Integrate: Full system demo recording
- **Friday:** Content: Phase 4 retrospective article
- **Saturday:** Deep Work: Applications to Tier 1 companies (Sophrosyne, Mindgrove)
- **Sunday:** Review & Plan Phase 5

#### Phase 4 Final Checklist
- [ ] 12+ OSS PRs submitted
- [ ] 8+ OSS PRs merged
- [ ] Hero project: `sovereign-health-edge-ai` complete with README, demo, paper
- [ ] TVM code PR submitted (not just docs)
- [ ] Resume rewritten with edge AI focus
- [ ] GitHub profile optimized
- [ ] Portfolio website live
- [ ] 24+ LinkedIn posts total
- [ ] 16+ Twitter threads total
- [ ] 8+ Reddit posts total
- [ ] 6+ long-form articles published
- [ ] Applications to Tier 1 companies sent

---

# ═══════════════════════════════════════════════════════════════════
# PHASE 5: ADVANCED COMPILER TOUCH + REMOTE PREP (WEEKS 49-60)
# Months 13-15 | Goal: MLIR literacy, AWQ mastery, remote positioning
# ═══════════════════════════════════════════════════════════════════

> **Phase 5 Philosophy:** You're now a senior contributor. Focus on high-leverage activities: compiler literacy, advanced quantization, and positioning for remote roles.

---

## WEEKS 49-51: MLIR + TVM Passes

### WEEK 49: MLIR Dialect Deep Dive
- **Monday:** Study: Write a custom MLIR dialect (follow tutorial)
- **Tuesday:** Implement: Build dialect, test lowering
- **Wednesday:** OSS: Submit MLIR tutorial improvement PR
- **Thursday:** Integrate: Read PyTorch → Torch-MLIR → LLVM pipeline
- **Friday:** Content: LinkedIn post on MLIR dialects
- **Saturday:** Deep Work: Article: "MLIR Dialects Explained for ML Engineers"
- **Sunday:** Review & Plan

### WEEK 50: TVM Pass Infrastructure
- **Monday:** Study: TVM Relay pass manager
- **Tuesday:** Implement: Write a custom optimization pass
- **Wednesday:** OSS: Submit TVM pass or test PR
- **Thursday:** Integrate: Benchmark impact of custom pass
- **Friday:** Content: Twitter thread on TVM passes
- **Saturday:** Deep Work: Deep technical article
- **Sunday:** Review & Plan

### WEEK 51: AutoTuning + Hardware Targeting
- **Monday:** Study: TVM AutoTVM cost model (XGBoost)
- **Tuesday:** Implement: Tune for new hardware target (simulate)
- **Wednesday:** OSS: File issue or PR on tuning documentation
- **Thursday:** Integrate: Full auto-tuning benchmark
- **Friday:** Content: LinkedIn post
- **Saturday:** Deep Work: Open-source tuning results
- **Sunday:** Review & Plan

## WEEKS 52-54: Advanced Quantization + On-Device Training

### WEEK 52: AWQ Production Implementation
- **Monday:** Study: AWQ implementation details (protecting salient channels)
- **Tuesday:** Implement: Full AWQ pipeline on your edge SLM
- **Wednesday:** OSS: Submit PR to `casymcc/awq` with edge benchmark
- **Thursday:** Integrate: AWQ vs GPTQ vs QAT comparison
- **Friday:** Content: Article: "AWQ on Edge: A Complete Walkthrough"
- **Saturday:** Deep Work: Benchmark table + analysis
- **Sunday:** Review & Plan

### WEEK 53: TinyTL (On-Device Transfer Learning)
- **Monday:** Study: On-device fine-tuning papers (TinyTL, On-Device Learning)
- **Tuesday:** Implement: Fine-tune classification head on Pi 5 itself
- **Wednesday:** OSS: File issue on `huggingface/peft` for edge adaptation docs
- **Thursday:** Integrate: Personalize model to user's biosignal patterns
- **Friday:** Content: LinkedIn post
- **Saturday:** Deep Work: Full TinyTL system
- **Sunday:** Review & Plan

### WEEK 54: ORT Mobile + Android Expansion
- **Monday:** Study: ONNX Runtime Mobile for Android
- **Tuesday:** Implement: Convert model to ORT mobile format. Test on emulator.
- **Wednesday:** OSS: Submit ORT mobile example PR
- **Thursday:** Integrate: Cross-platform benchmark (Arduino vs STM32 vs Pi vs Android)
- **Friday:** Content: LinkedIn post: "One model, four platforms"
- **Saturday:** Deep Work: Article on cross-platform deployment
- **Sunday:** Review & Plan

## WEEKS 55-57: RISC-V Production + India Ecosystem

### WEEK 55: Mindgrove/SHAKTI Targeting
- **Monday:** Study: Mindgrove MC-01 specs. Read their SDK if available.
- **Tuesday:** Implement: Target your model for RISC-V (QEMU or actual hardware)
- **Wednesday:** OSS: Contribute to Indian RISC-V project (doc or code)
- **Thursday:** Integrate: RISC-V benchmark vs ARM
- **Friday:** Content: LinkedIn post on Indian RISC-V chips
- **Saturday:** Deep Work: Article: "India's RISC-V Revolution for Edge AI"
- **Sunday:** Review & Plan

### WEEK 56: IndiaAI Compute + Government Resources
- **Monday:** Study: IndiaAI Compute Portal, AIKosh, DLI scheme
- **Tuesday:** Implement: Apply for and use IndiaAI Compute (free A100)
- **Wednesday:** OSS: File issue or contribute to IndiaAI-related open source
- **Thursday:** Integrate: Benchmark on A100 vs your local/free setups
- **Friday:** Content: LinkedIn post: "Free A100s for Indian developers"
- **Saturday:** Deep Work: Guide document for other Indian developers
- **Sunday:** Review & Plan

### WEEK 57: Ecosystem Networking
- **Monday:** Study: 10 more Indian semiconductor startups
- **Tuesday:** Implement: Send 10 LinkedIn connections to engineers at target companies
- **Wednesday:** OSS: Engage with Indian maintainers (Mindgrove, SHAKTI, etc.)
- **Thursday:** Integrate: Attend 1 India tech meetup (virtual or physical)
- **Friday:** Content: LinkedIn article: "The Indian ML Engineer's Guide to Semiconductors"
- **Saturday:** Deep Work: Full ecosystem database
- **Sunday:** Review & Plan

## WEEKS 58-60: Phase 5 Consolidation + Remote Prep

### WEEK 58: Resume Polish + Turing/Toptal Profiles
- **Monday:** Study: Turing.com requirements. Read success stories.
- **Tuesday:** Implement: Create Turing profile with all project links
- **Wednesday:** OSS: Ensure all repos have clear READMEs for recruiters
- **Thursday:** Integrate: Create Toptal + Crossover profiles
- **Friday:** Content: LinkedIn post: "Building a borderless resume with open source"
- **Saturday:** Deep Work: Full profile optimization across all platforms
- **Sunday:** Review & Plan

### WEEK 59: Mock Interviews + Technical Deep Dives
- **Monday:** Study: System design for edge ML (interview format)
- **Tuesday:** Implement: Practice 5 technical questions out loud
- **Wednesday:** OSS: Review complex PRs to understand senior-level code review
- **Thursday:** Integrate: Write 5 "perfect" answers with your project data
- **Friday:** Content: LinkedIn post on interview preparation
- **Saturday:** Deep Work: Full interview prep document
- **Sunday:** Review & Plan

### WEEK 60: Phase 5 Consolidation
- **Monday:** Study: Review all 60 weeks
- **Tuesday:** Implement: Complete any unfinished items
- **Wednesday:** OSS: Ensure 15+ PRs submitted, 10+ merged
- **Thursday:** Integrate: Full portfolio review
- **Friday:** Content: Phase 5 retrospective article
- **Saturday:** Deep Work: GitHub contribution graph celebration + next phase plan
- **Sunday:** Review & Plan Phase 6

#### Phase 5 Final Checklist
- [ ] 15+ OSS PRs submitted
- [ ] 10+ OSS PRs merged
- [ ] MLIR: custom dialect or pass attempted
- [ ] TVM: code PR submitted
- [ ] AWQ: production implementation benchmarked
- [ ] RISC-V: model targeted for Indian chips
- [ ] Turing/Toptal profiles created
- [ ] 30+ LinkedIn posts total
- [ ] 20+ Twitter threads total
- [ ] 12+ Reddit posts total
- [ ] 9+ long-form articles published

---

# ═══════════════════════════════════════════════════════════════════
# PHASE 6: INTERVIEW PREP + TRANSITION EXECUTION (WEEKS 61-78)
# Months 16-18 | Goal: Hired. Remote or hybrid. The OSS work pays off.
# ═══════════════════════════════════════════════════════════════════

> **Phase 6 Philosophy:** The learning never stops, but the goal shifts from "building skills" to "getting paid for them." Every activity should support interviews or applications.

---

## WEEKS 61-66: Interview Intensification

### WEEK 61: System Design for Edge ML
- **Monday:** Study: "Design a wearable AFib detector" — full system design
- **Tuesday:** Implement: Draw architecture. Write 500-word explanation.
- **Wednesday:** OSS: Contribute to `crespum/edge-ai` curated list
- **Thursday:** Integrate: Practice explaining your hero project in 5 minutes
- **Friday:** Content: LinkedIn post: "How I'd design a medical wearable from scratch"
- **Saturday:** Deep Work: Full system design document
- **Sunday:** Review & Plan

### WEEK 62: Coding Interview Prep (C++ for ML)
- **Monday:** Study: LeetCode easy/medium in C++ (arrays, strings, trees)
- **Tuesday:** Implement: 3 LeetCode problems
- **Wednesday:** OSS: Read complex C++ in TVM/ORT. Learn patterns.
- **Thursday:** Integrate: Practice explaining your C++ contributions
- **Friday:** Content: LinkedIn post
- **Saturday:** Deep Work: 10 LeetCode problems
- **Sunday:** Review & Plan

### WEEK 63: Behavioral Interview Prep
- **Monday:** Study: STAR method. Prepare 10 stories from your OSS journey.
- **Tuesday:** Implement: Write out answers to "Tell me about yourself" + "Why this pivot?"
- **Wednesday:** OSS: Engage with community to build more relationships
- **Thursday:** Integrate: Practice 5 behavioral questions out loud
- **Friday:** Content: LinkedIn post on career pivot learnings
- **Saturday:** Deep Work: Full behavioral Q&A document
- **Sunday:** Review & Plan

### WEEK 64: Company-Specific Preparation (Tier 1)
- **Monday:** Study: Sophrosyne tech stack, recent hires, products
- **Tuesday:** Implement: Customize resume + cover letter for Sophrosyne
- **Wednesday:** OSS: Find if Sophrosyne has open source. Contribute or engage.
- **Thursday:** Integrate: Apply to Sophrosyne with personalized message
- **Friday:** Content: LinkedIn post tagging relevant technology
- **Saturday:** Deep Work: Same for Mindgrove
- **Sunday:** Review & Plan

### WEEK 65: Company-Specific Preparation (Tier 2)
- **Monday:** Study: Qualcomm India, ARM India, STMicroelectronics India
- **Tuesday-Tuesday:** Implement: Customize applications for each
- **Wednesday:** OSS: Find and engage with their open source
- **Thursday:** Integrate: Submit applications
- **Friday:** Content: LinkedIn post
- **Saturday:** Deep Work: Application tracking spreadsheet
- **Sunday:** Review & Plan

### WEEK 66: Company-Specific Preparation (Tier 3 Remote)
- **Monday:** Study: Edge Impulse, Turing, Toptal, Crossover
- **Tuesday-Tuesday:** Implement: Apply to 3 remote-friendly companies
- **Wednesday:** OSS: Ensure all profiles have OSS contribution highlights
- **Thursday:** Integrate: Follow up on all pending applications
- **Friday:** Content: LinkedIn post
- **Saturday:** Deep Work: Full application pipeline
- **Sunday:** Review & Plan

## WEEKS 67-72: Active Interviewing + Continued OSS

### WEEK 67: Interview 1 (Expected)
- **Monday:** Study: Company-specific deep dive
- **Tuesday:** Implement: System design practice
- **Wednesday:** OSS: Light contribution (issue comment or doc fix)
- **Thursday:** INTEGRATE: **PHONE SCREEN / INTERVIEW**
- **Friday:** Content: LinkedIn post (general, not about specific interview)
- **Saturday:** Deep Work: Interview debrief + improvement notes
- **Sunday:** Review & Plan

### WEEK 68: Interview 2 (Expected)
- Same structure as Week 67
- Focus: Technical deep dive (quantization, deployment, benchmarking)

### WEEK 69: Interview 3 (Expected)
- Same structure
- Focus: Behavioral + culture fit

### WEEK 70: OSS Maintenance + Learning
- **Monday:** Study: Any gaps identified in interviews
- **Tuesday-Tuesday:** Implement: Fill gaps immediately
- **Wednesday:** OSS: Submit PR addressing interview feedback
- **Thursday:** Integrate: Practice weak areas
- **Friday:** Content: LinkedIn post on learning from interviews
- **Saturday:** Deep Work: Deep study of weak area
- **Sunday:** Review & Plan

### WEEK 71: Offer Negotiation Prep
- **Monday:** Study: Salary negotiation for Indian tech roles
- **Tuesday:** Implement: Research salary bands (₹20-50 LPA for this level)
- **Wednesday:** OSS: Continue light contributions
- **Thursday:** Integrate: Prepare negotiation script
- **Friday:** Content: LinkedIn post (career general)
- **Saturday:** Deep Work: Full negotiation prep
- **Sunday:** Review & Plan

### WEEK 72: Offer Evaluation
- **Monday:** Study: Evaluate offer vs your goals (remote? salary? growth?)
- **Tuesday-Tuesday:** Implement: Decision matrix
- **Wednesday:** OSS: Continue contributions
- **Thursday:** Integrate: Make decision or negotiate
- **Friday:** Content: Celebrate (careful, professional)
- **Saturday:** Deep Work: Acceptance or counter-offer
- **Sunday:** Review & Plan

## WEEKS 73-78: Transition + Continuous OSS

### WEEK 73: Handover at Current Job
- **Monday:** Study: Transition best practices
- **Tuesday-Tuesday:** Implement: Document current work for handover
- **Wednesday:** OSS: Continue 1 contribution/week minimum
- **Thursday:** Integrate: Plan transition timeline
- **Friday:** Content: LinkedIn post on gratitude (current role)
- **Saturday:** Deep Work: Clean professional transition
- **Sunday:** Review & Plan

### WEEK 74: New Role Onboarding
- **Monday:** Study: New company tech stack
- **Tuesday-Tuesday:** Implement: Setup dev environment
- **Wednesday:** OSS: Introduce yourself to new company's OSS if any
- **Thursday:** INTEGRATE: Start new role
- **Friday:** Content: LinkedIn post (new beginning, humble)
- **Saturday:** Deep Work: First week impact
- **Sunday:** Review & Plan

### WEEKS 75-78: Continuous Learning + OSS
- Continue 1 OSS contribution per week
- Continue 2 LinkedIn posts per month
- Continue 1 technical article per month
- Keep `learning_log/` alive — never stop documenting
- **Goal:** By Month 24, have 20+ merged PRs, 15+ articles, recognized community member

#### Phase 6 Final Checklist
- [ ] 18+ OSS PRs submitted
- [ ] 12+ OSS PRs merged
- [ ] 3+ interviews completed
- [ ] 1+ job offer received
- [ ] 36+ LinkedIn posts total
- [ ] 24+ Twitter threads total
- [ ] 15+ Reddit posts total
- [ ] 12+ long-form articles published
- [ ] New role accepted
- [ ] Transition completed professionally
- [ ] OSS commitment: 1 PR/week continuing

---

# ═══════════════════════════════════════════════════════════════════
# APPENDIX A: MASTER OSS REPOSITORY REFERENCE
# All repos mapped by phase and contribution target
# ═══════════════════════════════════════════════════════════════════

## Tier 1: Primary Targets (Code Contributions)

| Repo | Phase | Target PRs | Difficulty | Key Files to Study |
|------|-------|------------|------------|-------------------|
| `neuropsychology/NeuroKit` | 1-2 | 3-4 | Easy | `ecg/`, `ppg/`, `hrv/` |
| `tensorflow/tflite-micro` | 1-4 | 2-3 | Medium | `tensorflow/lite/micro/`, `arduino/` |
| `edgeimpulse/example-standalone-inferencing` | 2 | 2 | Easy | `main.cpp`, `tflite-model/` |
| `apache/tvm` | 3-4 | 2-3 | Hard | `python/tvm/relay/`, `tutorial/` |
| `microsoft/onnxruntime` | 2-4 | 1-2 | Medium | `onnxruntime/core/providers/cpu/` |
| `ggerganov/llama.cpp` | 2-3 | 1 | Medium | `ggml.c`, `examples/` |
| `huggingface/optimum` | 2-3 | 1-2 | Medium | `onnxruntime/quantization/` |
| `llvm/llvm-project` (MLIR) | 4-5 | 1 | Hard | `mlir/examples/toy/` |
| `mit-han-lab/mcunet` | 3 | 1 | Medium | `tinynas/` |

## Tier 2: Secondary Targets (Issues, Docs, Community)

| Repo | Phase | Action | Why |
|------|-------|--------|-----|
| `ARM-software/CMSIS-NN` | 2 | File issues, read code | Understand optimized kernels |
| `shaktiproject/SHAKTI-SoC` | 3 | File issues, engage | Indian RISC-V ecosystem |
| `facebookresearch/faiss` | 3 | File issues | Edge deployment tips |
| `streamlit/streamlit` | 2-3 | File feature requests | Dashboard tool you use |
| `huggingface/peft` | 3 | File doc issues | LoRA for edge fine-tuning |
| `huggingface/trl` | 3 | File doc issues | SLM training |
| `riscv/riscv-gnu-toolchain` | 3 | File issues | Cross-compilation |

## Tier 3: Reference Only (Study, Star, Fork)

| Repo | Why |
|------|-----|
| `mlc-ai/mlc-llm` | Multi-backend edge LLM |
| `ollama/ollama` | Local LLM management |
| `vllm-project/vllm` | High-throughput inference |
| `tinygrad/tinygrad` | Minimalist DL framework |
| `openvinotoolkit/openvino` | Intel edge inference |
| `crespum/edge-ai` | Curated resources |
| `mit-han-lab/efficientml.ai` | MIT course materials |
| `DataTalksClub/mlops-zoomcamp` | MLOps production |
| `GokuMohandas/Made-With-ML` | Production ML systems |

---

# APPENDIX B: DAILY STRUCTURE TEMPLATE (Every Week)

```
MONDAY    → Study (1.5 hrs): Read/watch course material. Take notes.
TUESDAY   → Implement (1.5 hrs): Write code. Flash hardware. Train model.
WEDNESDAY → OSS (1.5 hrs): Read source. Draft PR. File issue. Comment.
THURSDAY  → Integrate (1.5 hrs): Combine study + implementation. Benchmark.
FRIDAY    → Content (1.0 hr): Write post. Draft article. Create diagram.
SATURDAY  → Deep Work (3.0 hrs): Major weekly milestone. Long session.
SUNDAY    → Review (1.5 hrs): Retro. Plan. Rest. Non-negotiable.
```

**If you miss a day:** Make it up on Saturday. Never skip Wednesday (OSS day).
**If work is brutal:** Minimum viable week = Monday study + Wednesday OSS + Saturday deep work.
**If burned out:** Take Sunday + Monday off. Resume Tuesday. One missed week is fine. Two is dangerous.

---

# APPENDIX C: LEARNING LOG TEMPLATE

Create this file every Monday: `learning_log/weekXX.md`

```markdown
# Week XX Learning Log

## What I Studied
- [ ] Resource: ___
- [ ] Key insight: ___
- [ ] Question I still have: ___

## What I Built
- [ ] Code: ___
- [ ] Hardware: ___
- [ ] Benchmark result: ___

## OSS Activity
- [ ] Repo: ___
- [ ] Action: ___
- [ ] Link: ___

## Content Created
- [ ] Platform: ___
- [ ] Topic: ___
- [ ] Engagement: ___

## Blockers
- [ ] ___

## Next Week Priority
- [ ] ___
```

---

# APPENDIX D: SUCCESS METRICS BY PHASE

| Phase | PRs Submitted | PRs Merged | LinkedIn Posts | Articles | Key Milestone |
|-------|--------------|------------|----------------|----------|---------------|
| 1 (W1-12) | 3 | 1 | 6 | 1 | Arduino gesture demo working |
| 2 (W13-24) | 5 | 3 | 12 | 2 | STM32 + Pi 5 both running models |
| 3 (W25-36) | 8 | 5 | 18 | 4 | TVM tutorial PR + Edge RAG working |
| 4 (W37-48) | 12 | 8 | 24 | 6 | Hero project complete, 8 PRs merged |
| 5 (W49-60) | 15 | 10 | 30 | 9 | MLIR PR + remote profiles live |
| 6 (W61-78) | 18 | 12 | 36 | 12 | **Job offer accepted** |

---

*Master OSS Plan | June 2026 | 78 weeks | All Open Source | All Implementation*
*Synthesized from: plan_agy.md, plan_cc.md, rough_agy.md, rough_cc.md*
*Philosophy: Every week produces a commit — to GitHub or to an OSS repo.*

> **START TODAY:**
> 1. Fork `neuropsychology/NeuroKit`
> 2. Download MIT-BIH ECG data
> 3. File your first issue
> 4. Write `learning_log/week01.md`
> The rest is just showing up for 78 weeks.
