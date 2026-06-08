# 🔬 ROUGH ANALYSIS: Pre-Plan Deep Thinking
## Abhishek's Career Pivot — Internal Reasoning Document
*Created: June 2026 | Before final plan_agy.md | Raw synthesis*

---

## SECTION 1: WHO IS ABHISHEK? — PROFILE SYNTHESIS

### From Resume + All Research Docs

**Current Role:** Senior Data Scientist (likely at an Indian enterprise/insurance-scale firm — references to Chubb in docs)
**Core Stack (confirmed across all files):**
- PyTorch, transformers, NLP pipelines
- MLflow, Airflow (MLOps)
- PySpark, Big Data processing
- Production RAG systems (FAISS, LangChain)
- BERT-based NLP, survival analysis, time series
- Streamlit dashboards
- CI/CD with GitHub Actions

**What He Lacks (confirmed gaps):**
- C/C++ depth for embedded/kernel work
- Hardware-aware ML knowledge (quantization at production depth)
- Deployment to edge hardware (TinyML, TFLite, ONNX Runtime)
- Signal processing (ECG, PPG, biosensing domain)
- Compiler knowledge (MLIR, TVM — these are advanced)
- GPU cluster distributed training at scale (DeepSpeed, FSDP)
- Open source contributions (currently ZERO publicly visible)
- Public presence (no GitHub stars, no LinkedIn posts, no blogs)

**Personal Constraints:**
- Working professional — very limited time (~10-13 hours/week realistic)
- Camera shy → NO video content, NO face-forward presentations
- Male
- India-based → INR constraint on spending
- Physical learning planned → can buy edge hardware
- GPU access → online clusters (affordable, not owning GPU)

---

## SECTION 2: THE FUNDAMENTAL CONFUSION — WHICH PATH?

Abhishek is confused because his research has surfaced 3 overlapping but distinct career paths:

### PATH A: Deep ML / LLM Engineering Track
- Fine-tuning, RLHF, DPO, distributed training
- OSS: HuggingFace TRL, PEFT, vLLM, Axolotl
- Jobs: AI-first startups, GCCs (NVIDIA, Google)
- Remote potential: HIGH (this is software-layer work)
- Salary range: ₹25–65 LPA senior level
- Time to pivot: 9-12 months of focused work

### PATH B: Physical AI / Edge AI / TinyML Track
- Model compression, quantization, edge deployment
- Hardware: RPi, Arduino, STM32, sensors
- OSS: TFLite-Micro, ONNX Runtime, Apache TVM, EdgeImpulse SDK
- Jobs: Indian semiconductor startups (Sophrosyne, Mindgrove, Netrasemi)
- Remote potential: MEDIUM (hardware teams prefer in-person, but ML layer can be remote)
- Salary range: ₹20–50 LPA depending on company
- Time to pivot: 12-18 months (hardware learning takes physical time)

### PATH C: AI Compiler / Semiconductor Infrastructure Track
- MLIR, TVM backends, kernel optimization
- Very low-level: assembly, SIMD, GPU kernels (OpenCL/CUDA)
- OSS: LLVM/MLIR, Apache TVM, ONNXRuntime
- Jobs: Qualcomm, ARM, NVIDIA (India hubs)
- Remote potential: LOW (these are highly specialized, often on-site)
- Salary range: ₹40–80+ LPA for senior specialists
- Time to pivot: 24-36 months minimum (very steep learning curve)
- Abhishek is NOT starting from zero for this but it requires C++ mastery first

---

## SECTION 3: THE REASONING — WHY PATHS ARE LINKED

**Critical Insight:** These paths are NOT mutually exclusive — they form a pyramid.

```
         ╔══════════════════════════╗
         ║  PATH C: COMPILER/HW     ║  ← Top: Hardest, highest paid
         ╚══════════════════════════╝
                      ↑ builds on
         ╔══════════════════════════╗
         ║  PATH B: EDGE AI/TINY ML ║  ← Middle: Hardware meets ML
         ╚══════════════════════════╝
                      ↑ builds on
         ╔══════════════════════════╗
         ║  PATH A: DEEP ML/LLM     ║  ← Foundation: Model mastery
         ╚══════════════════════════╝
```

**REASONING:**
1. Path A (Deep ML) is already 40-50% done for Abhishek (RAG, LLM pipelines, PyTorch)
2. Path B (Physical AI) requires Path A depth + hardware knowledge
3. Path C (Compilers) requires Path B fluency + deep C++/LLVM knowledge

**For Abhishek, the OPTIMAL strategy is:**
- **Build on Path A strengths first** (months 1-4)
- **Transition to Path B hardware** (months 5-12) — this is the DIFFERENTIATOR
- **Touch Path C through OSS** (months 13-18) — compiler contributions are the golden ticket

This is NOT three separate paths. It's ONE unified journey from top to bottom of the AI stack.

---

## SECTION 4: THE CONFUSION RESOLUTION

### Why Physical AI is the RIGHT BET for Abhishek:

1. **India Semiconductor Mission tailwind** — ₹76,000 crore government program creating demand
2. **Indian startup ecosystem** — Sophrosyne, Mindgrove, Netrasemi, Saankhya are all hiring
3. **His RAG/NLP skills transfer** — sequence modeling (ECG/PPG) = temporal NLP
4. **His MLOps skills transfer** — edge model CI/CD is the same pipeline
5. **Physical AI is LESS competitive** than pure LLM engineering at the application layer
6. **He already has the ML depth** that 90% of embedded engineers lack
7. **Hardware is tactile** — physical projects create VISIBLE portfolio proof
8. **Biosensing + Edge AI = remote-friendly** for the ML layer (signal processing + model training)

### Why NOT pure LLM/Distributed path:
- Extremely competitive (thousands of Indian engineers chasing same roles)
- Requires expensive GPU clusters for truly impressive projects
- Application-layer LLM work is being commoditized rapidly (GPT API wrappers)
- His biggest differentiation would be LOST — hardware empathy is rare among LLM engineers

### The HYBRID ANSWER:
**Primary direction:** Physical AI / Edge AI with focus on TinyML and hardware-aware ML
**Secondary thread:** LLM fine-tuning / efficient inference (SLMs for edge) — this connects both worlds
**Long-term horizon:** AI Compiler contributions (TVM, MLIR) — the premium tier

---

## SECTION 5: THE CONNECTIVITY MAP

```
Abhishek's Current Stack
         │
         ▼
[Month 1-3] Foundation Layer
 PyTorch → Quantization (PTQ/QAT) → Model compression pipeline
 Signal Processing basics (scipy, neurokit2)
 TFLite/ONNX conversion mastery
         │
         ▼
[Month 4-6] Hardware Hands-On
 RPi 5 → Edge inference (ONNX Runtime on ARM)
 STM32 → TFLite Micro → C arrays
 ECG/PPG sensors → biosignal pipeline
 Edge Impulse → EON Tuner (AutoML for TinyML)
         │
         ▼
[Month 7-10] Domain Depth + OSS
 Apache TVM → custom hardware targeting
 RISC-V basics (Indian chip startup signal)
 Edge LLM / SLM on Pi 5 (llama.cpp, GGUF)
 OSS contributions (3-5 merged PRs by month 10)
         │
         ▼
[Month 11-14] Advanced + AI Compiler Touch
 MLIR basics (not production depth — exploration)
 Knowledge distillation + NAS fundamentals
 Hero project: integrated biosensing system
 OSS: TVM/tflite-micro substantive PRs
         │
         ▼
[Month 15-18] Job Targeting + Presence
 Resume optimized for physical AI roles
 LinkedIn authority established (50+ technical posts)
 Active interviews: Tier 1 India companies
 Remote roles via Turing/Crossover targeted
```

---

## SECTION 6: ONLINE PRESENCE STRATEGY (Camera Shy Edition)

### What WORKS without video:

**LinkedIn (PRIMARY platform):**
- PDF carousels (Canva-designed, no face needed) — Algorithm LOVES these
- Text-first posts: "Learn in Public" format
- Weekly posts: project updates, compression benchmark results
- "Document" feature → PDF uploads of technical writeups
- 20 mins/day minimum for 18 months
- Target: 500+ connections in edge AI niche, 5-10 posts/month

**Twitter/X (TECHNICAL NETWORKING):**
- Threads documenting projects: "Building ECG detector on ₹450 hardware — Thread 🧵"
- Reply to: @tinyMLsummit, @EdgeImpulse, @hanlab_mit, @ApacheTVM, @mlc_llm
- Code snippets, benchmark tables, GitHub links
- 10-15 mins/day engagement with key accounts
- Target: Follow 200 relevant accounts, be recognized by 3-5 community leaders

**Reddit (CREDIBILITY BUILDER):**
- r/MachineLearning — share project write-ups
- r/LocalLLaMA — edge LLM experiments
- r/embedded — signal processing projects
- r/RISCV — RISC-V tinyml work
- r/India_semiconductor (if exists) or r/ECE
- Deep, helpful comments — NOT self-promotion
- One substantial post per month minimum

**GitHub (THE TRUTH MACHINE):**
- Public repos with EXCELLENT READMEs (benchmarks, diagrams, results)
- HuggingFace Hub: quantized model uploads with model cards
- Consistent contribution graph — daily commits, even small ones
- Target: 6-8 polished repos by month 18

**Technical Writing (Substack or Medium):**
- One in-depth article per month (long-form, 1500+ words)
- Cross-post to LinkedIn as PDF carousel
- Topics: project walkthroughs, compression benchmarks, hardware comparisons
- No face — just code, data, diagrams

---

## SECTION 7: HARDWARE PROCUREMENT PLAN (INR Constrained)

### Phase 1 Hardware (Month 1, ~₹17,000-19,000 total):
```
RPi 5 (4GB): ₹7,500-8,500    → Primary edge inference platform
Arduino Nano 33 BLE Sense: ₹3,200  → TinyML MCU
MAX30102 (PPG): ₹280          → Biosensing
AD8232 ECG: ₹450              → ECG capture
MPU-6050 IMU: ₹220            → Motion/gesture
STM32 Nucleo-F401RE: ₹2,200  → ARM Cortex-M4 target
USB-TTL: ₹150                 → Debug
Breadboard + wires: ₹400      → Prototyping
5V/3A power: ₹600             → Pi power
16GB microSD: ₹350            → Pi OS
Total: ~₹15,350-17,000
```

### Phase 2 Optional (Month 4-6, ~₹4,000-8,000):
```
Google Coral USB Edge TPU: ₹4,000-6,000 (Amazon India)
OR
Raspberry Pi AI Kit (Hailo-8L NPU): ~₹8,000
```

### Phase 3 Optional (Month 10+, ~₹8,000):
```
Budget FPGA kit (if Path C): Lattice iCEstick ₹3,500-4,000
OR just continue with RPi+STM32 stack
```

**SKIP LIST (too expensive or overkill):**
- NVIDIA Jetson Orin Nano Super: ₹22,000+ → SKIP for now
- NVIDIA Jetson AGX: ₹80,000+ → HARD NO

---

## SECTION 8: GPU CLUSTER STRATEGY (India-Optimized)

### Free Tier (Use first 6 months heavily):
- **Kaggle Notebooks**: 30 GPU hours/week, NVIDIA T4 (16GB) — FREE
  - Best for: QLoRA fine-tuning 7B models, TFLite conversion, ONNX experiments
- **Google Colab Free**: T4 GPU, limited but useful
- **Lightning.ai**: Free tier with limited GPU time

### Paid Tier (Month 7-18, INR-friendly):
- **JarvisLabs (jarvislabs.ai)**: India-based, INR billing, per-minute
  - A100 (40GB): ~₹70-80/hr | RTX 3090: ~₹35-45/hr
  - Best for: Multi-GPU experiments, 13B model fine-tuning
- **E2E Networks**: India-based, from ₹49/hr entry-level GPU
- **Vast.ai**: USD pricing but cheapest — spot A100 at $0.50-1.50/hr (~₹42-126/hr)
  - Good for: Overnight training runs with checkpointing
- **RunPod Community**: $0.30-0.60/hr (~₹25-50/hr) for RTX class GPUs

### Budget Allocation (18 months, approximate):
- Months 1-6: ~FREE (Kaggle + Colab) 
- Months 7-12: ~₹2,000-3,000/month on JarvisLabs/E2E (small experiments)
- Months 13-18: ~₹3,000-5,000/month (larger models, benchmarking)
- Total GPU spend: ~₹30,000-50,000 over 18 months

---

## SECTION 9: KEY INSIGHT — THE DIFFERENTIATOR

**Most Data Scientists → LLM apps (highly competitive)**
**Most Embedded Engineers → no ML depth (limited)**

**Abhishek's MOAT:** Senior ML depth + Hardware empathy = VERY RARE

The reference profile Dhamodharan (Qualcomm) is instructive:
- He had EE background + SW training → took 5 years to reach Qualcomm Senior
- Abhishek already has ML depth → can compress this to 18 months with hardware learning

**The portfolio that gets hired:**
1. ECG/PPG biosignal pipeline → STM32 inference (proves hardware empathy)
2. Quantization benchmark study (proves compression mastery)
3. On-device LLM with RAG on Pi 5 (bridges both worlds — UNIQUE)
4. Merged PRs to TFLite Micro / Apache TVM (proves OSS credibility)
5. Technical writing corpus (10+ articles by month 18)

---

## SECTION 10: REALISTIC TIME BUDGET ANALYSIS

Abhishek is a working professional. Brutal honesty:

```
Weekday evenings (Mon-Fri): 1.5 hrs/day = 7.5 hrs
Weekend (Sat-Sun): 4 hrs + 3 hrs = 7 hrs
────────────────────────────────
Weekly total: ~14.5 hours

Conservative (accounting for fatigue, life): ~11-12 hrs/week
Over 18 months (78 weeks): ~858-936 hours total
```

**How to allocate 12 hours/week:**
- 40% → Hands-on projects (hardware + code): ~5 hrs
- 30% → Theory, courses, papers: ~3.5 hrs
- 20% → Writing, documentation, READMEs: ~2.5 hrs
- 10% → Community (OSS, LinkedIn, Reddit): ~1 hr

This is ACHIEVABLE without burnout if treated as a sustainable sprint, not a marathon of pain.

---

## SECTION 11: OSS CONTRIBUTION STRATEGY — TIERED APPROACH

### Entry Phase (Months 1-3): No code changes yet
- Star and fork repos: tflite-micro, onnxruntime, neurokit2, tvm
- Read CONTRIBUTING.md docs
- File bug reports from actual usage (real signal = instant credibility)
- Help answer questions in GitHub Issues and Discussions

### First PR Phase (Months 4-6): Low-barrier contributions
- **NeuroKit2**: Add a new biosignal processing algorithm or improve ECG documentation
- **TFLite Micro**: Fix documentation inconsistency you discovered during deployment
- **Edge Impulse SDK**: Add sensor driver example
- **PhysioNet Python toolkit**: New ECG utility function

### Substantive PRs (Months 7-12): 
- **Apache TVM**: Add tutorial for deploying biosignal model on Pi 5
- **ONNX Runtime**: Mobile/embedded optimization documentation
- **llama.cpp**: ARM NEON benchmark addition or example
- **tinygrad**: Small backend improvement (ambitious but impressive)

### Senior Contributions (Months 13-18):
- **Apache TVM**: Bug fix or small feature in MicroTVM
- **MLIR**: Contribution to a tutorial or test case
- **ONNXRuntime**: Execution provider improvement
- Target: 8-12 merged PRs total across the 18 months

---

## SECTION 12: REMOTE JOB STRATEGY — REALISTIC ASSESSMENT

**Truth about remote work in Indian semiconductor:**
- Hardware/Physical design roles: 95% in-office (Bengaluru, Hyderabad)
- ML optimization roles: 60% hybrid possible
- Signal processing / model training roles: 70-80% remote possible

**Realistic remote opportunities:**
1. **Model optimization engineer** at Indian startups (Sophrosyne can be hybrid)
2. **Edge AI ML engineer** at AI-first startups (more flexible)
3. **Turing / Crossover** — connects to US companies, 100% remote, USD pay
4. **Upwork / Toptal** — freelance edge AI projects
5. **Global OSS fellowship** — MLH Fellowship (stipend, remote, 12 weeks)

**Strategy:** Build skills and portfolio locally (physical hardware), then negotiate remote or hybrid. The OSS contributions make remote credibility possible because employers can SEE your work quality online.

---

## SECTION 13: WHAT MAKES THIS PLAN DIFFERENT FROM EXISTING DOCS

The existing files (Physical_AI roadmap, Strategic Transition Roadmap) are GOOD but:
1. They are 12-month plans — Abhishek asked for 18 months (more depth needed)
2. They don't address the PATH CONFUSION clearly
3. They don't integrate the Deep ML track with Physical AI
4. They don't have a detailed ONLINE PRESENCE strategy
5. They treat OSS as a side activity, not a primary career tool
6. They don't address the camera-shy constraint
7. They don't explicitly connect the LLM/fine-tuning skills to hardware
8. They underestimate how much Path A (Deep ML) work is needed first

**The new 18-month plan must:**
- Resolve the path confusion with explicit reasoning
- Be a UNIFIED path: Deep ML → Edge AI → Compiler touches
- Have implementation-level specificity (exact code, exact tools)
- Integrate OSS as a primary career lever (not an afterthought)
- Have a month-by-month presence calendar
- Account for INR constraints throughout
- Address the camera-shy constraint creatively
- Show how ALL current skills transfer

---

## SECTION 14: RISK ANALYSIS

### Risk 1: Time management failure
**Mitigation:** 12-hour week cap, scheduled slots, no single-day 8-hour sprints

### Risk 2: Hardware compatibility issues
**Mitigation:** Stick to well-documented, beginner-friendly hardware. RPi 5 has huge community support.

### Risk 3: OSS contributions not getting merged
**Mitigation:** Start with documentation PRs (almost always merged). Build reputation before code PRs.

### Risk 4: Job market shift (AI landscape changes fast)
**Mitigation:** The physical/edge AI path is LESS affected by model-layer shifts. Hardware constraints don't disappear.

### Risk 5: INR deprecation or platform pricing changes
**Mitigation:** Keep GPU spend optional. Hardware kit is one-time purchase. Free tiers (Kaggle) always exist.

### Risk 6: Burnout from dual track (job + learning)
**Mitigation:** Plan has natural "consolidation weeks" every 2 months. Presence/writing tasks are lower cognitive load.

---

## SECTION 15: FINAL SYNTHESIS BEFORE PLAN

**The Answer to Abhishek's Core Question:**

> "I am confused on which path to choose, as most of these are linked together."

**VERDICT:** They ARE linked. Choose Physical AI as the PRIMARY specialization because:
1. Your ML depth is already there — you just need hardware empathy
2. India's semiconductor boom creates growing demand you can position for NOW
3. Physical AI roles are less competitive than application-layer LLM roles
4. The hardware work creates undeniable, tangible portfolio evidence
5. OSS contributions to TinyML projects compound over 18 months

**But make it HYBRID:**
- Keep the LLM/fine-tuning thread alive (SLMs for edge = the future)
- Touch compiler knowledge through TVM (not MLIR production depth)
- Build presence as "The engineer who bridges AI models and silicon"

**The one-sentence positioning:**
> *"Senior ML Engineer who deploys transformer-scale intelligence onto microcontroller-class silicon"*

This is a rare position that commands premium because it requires BOTH deep ML knowledge AND hardware empathy.

---

*End of rough analysis. The final plan_agy.md synthesizes all of the above into an executable 18-month roadmap.*
