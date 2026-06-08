# 🎯 The Integrated Edge-Deep ML Pivot Plan — Chapter-Level Implementation Guide
## Abhishek Bhardwaj | Month 1 Sprint: From Cloud Data Scientist → Hardware-Aware ML Practitioner
### June–July 2026 | Working Professional Budget: ~10-13 hrs/week | Hardware Budget: Under ₹5,000
### Version 2.0 | Expanded with 2026-2027 Industry Data, Exact Repos, Topic-by-Topic Courses

---

## 📋 Table of Contents

1. [Plan Overview and Core Philosophy](#1-plan-overview-and-core-philosophy)
2. [The 4-Week Arc at a Glance](#2-the-4-week-arc-at-a-glance)
3. [Hardware Procurement — Exact Vendors and Alternatives](#3-hardware-procurement--exact-vendors-and-alternatives)
4. [Free GPU Compute — Exact Login Links and Weekly Limits](#4-free-gpu-compute--exact-login-links-and-weekly-limits)
5. [Week 1: Foundation — Day-by-Day Implementation](#5-week-1-foundation--day-by-day-implementation)
6. [Week 2: Deep ML + Quantization — Day-by-Day Implementation](#6-week-2-deep-ml--quantization--day-by-day-implementation)
7. [Week 3: Physical AI Deployment — Day-by-Day Implementation](#7-week-3-physical-ai-deployment--day-by-day-implementation)
8. [Week 4: Integration + Public Launch — Day-by-Day Implementation](#8-week-4-integration--public-launch--day-by-day-implementation)
9. [Topic-by-Topic Course Curriculum (2026-2027 Industry-Relevant)](#9-topic-by-topic-course-curriculum-2026-2027-industry-relevant)
10. [GitHub Repository Strategy — Exact Repos to Star, Fork, and Contribute To](#10-github-repository-strategy--exact-repos-to-star-fork-and-contribute-to)
11. [Online Presence Content Calendar — Exact Templates](#11-online-presence-content-calendar--exact-templates)
12. [LinkedIn Algorithm-Optimized Posting Strategy (2026 Data)](#12-linkedin-algorithm-optimized-posting-strategy-2026-data)
13. [Reddit Strategy — Exact Subreddits and Post Templates](#13-reddit-strategy--exact-subreddits-and-post-templates)
14. [Twitter/X Strategy — Exact Thread Templates](#14-twitterx-strategy--exact-thread-templates)
15. [OSS Contribution Roadmap — Week-by-Week Targets](#15-oss-contribution-roadmap--week-by-week-targets)
16. [Exact Code Templates — Copy-Paste Ready](#16-exact-code-templates--copy-paste-ready)
17. [Troubleshooting Guide — Common Failures and Fixes](#17-troubleshooting-guide--common-failures-and-fixes)
18. [Budget Tracker — INR Spend Week by Week](#18-budget-tracker--inr-spend-week-by-week)
19. [Post-Month 1 Expansion Path — Months 2-6 Overview](#19-post-month-1-expansion-path--months-2-6-overview)
20. [Industry Certifications That Matter in 2026](#20-industry-certifications-that-matter-in-2026)
21. [Remote Job Application Templates](#21-remote-job-application-templates)
22. [Daily Micro-Habit System](#22-daily-micro-habit-system)
23. [Success Criteria and Checklist](#23-success-criteria-and-checklist)

---

## 1. Plan Overview and Core Philosophy

**The unifying mission:** Build and deploy a **TinyML Biosensing + Gesture System** in 4 weeks, while simultaneously launching your public technical presence and making your first open-source contributions.

This isn't three separate tracks (Deep ML + Physical AI + OSS). It is one track:
> *"I used my PyTorch skills to train a 1D-CNN on ECG data, quantized it to INT8, deployed it on an Arduino Nano 33 BLE Sense, measured inference latency, and documented the entire pipeline in public."*

That sentence alone checks every career box you need.

**Why this specific project?**
1. **Biosensing** uses your existing time-series/signal processing intuition (survival analysis, temporal modeling)
2. It directly maps to **Sophrosyne Technologies** (India), **Ultrahuman** (India), and global wearable companies
3. Sensors (MAX30102, AD8232, MPU-6050) are dirt-cheap in India (~₹300-500)
4. The data (**PhysioNet MIT-BIH**) is free and gold-standard
5. It lets you use your NLP/transformer knowledge (1D temporal transformers for ECG sequences)
6. The hardware constraint (256KB RAM) forces you to learn quantization, pruning, and knowledge distillation

**Core philosophy: Implementation over theory.**
Every week must produce a runnable artifact. No week ends with "I read a chapter." Every week ends with "I committed code to GitHub."

---

## 2. The 4-Week Arc at a Glance

```
Week 1: FOUNDATION — Setup, Signal Processing, Online Presence Launch
Week 2: DEEP ML — Train 1D-CNN/BiLSTM on ECG, Quantization Theory + Practice
Week 3: PHYSICAL AI — Deploy to Arduino, Benchmark, Build the Edge Pipeline
Week 4: INTEGRATION — End-to-end Demo, OSS Contributions, Content Blast
```

**Weekly time budget:**
- Weekdays (Mon–Thu): 1.5 hrs/day × 4 = 6 hrs
- Saturday: 3 hrs
- Sunday: 2 hrs (light: writing/planning)
- **Total: ~11 hrs/week** | **Minimum viable: 6 hrs/week** (do only starred ⭐ tasks)

**How to split time each week:**
| Activity | Hours/Week | % |
|----------|-----------|---|
| Hands-on projects (hardware + code) | 5 hrs | 45% |
| Theory, courses, papers | 2.5 hrs | 23% |
| Technical writing (READMEs, posts) | 2 hrs | 18% |
| Community (OSS, LinkedIn, Reddit) | 1.5 hrs | 14% |

---

## 3. Hardware Procurement — Exact Vendors and Alternatives

### Essential Order List (Order on Day 1, Monday Week 1)

| # | Component | Purpose | Approx Cost (INR) | Primary Source | Backup Source |
|---|-----------|---------|-------------------|----------------|---------------|
| 1 | **Arduino Nano 33 BLE Sense** | TinyML MCU with onboard IMU, mic, temp, humidity, light, BLE | ₹3,200 | [Amazon India](https://amazon.in) (search "Arduino Nano 33 BLE Sense") | [Robu.in](https://robu.in), [Evelta.com](https://evelta.com) |
| 2 | **MPU-6050 IMU Module** | 3-axis accelerometer + gyroscope for gesture recognition | ₹220 | [Amazon India](https://amazon.in) (search "MPU-6050 GY-521") | Local electronics shop (SP Road Bengaluru, Lamington Road Mumbai) |
| 3 | **MAX30102 Pulse Oximeter** | PPG / SpO2 sensor (targets Sophrosyne domain) | ₹280 | [Amazon India](https://amazon.in) (search "MAX30102 heart rate sensor") | [Robu.in](https://robu.in) |
| 4 | **Breadboard (830 tie-points)** | Prototyping circuits | ₹120 | Any electronics shop | Amazon kit |
| 5 | **Jumper wires (M-M, M-F, F-F, 40pcs each)** | Connecting sensors to Arduino | ₹150 | Any electronics shop | Amazon "breadboard jumper wires" |
| 6 | **USB-micro cable** | Programming Arduino (if you don't have one) | ₹100 | Any mobile accessory shop | Included with some Arduino kits |
| 7 | **Arduino Nano 33 IoT** (fallback) | If BLE Sense out of stock — loses onboard sensors | ₹2,400 | Amazon India | Robu.in |
| 8 | **ESP32-S3 DevKit** (ultra-budget fallback) | If both above too expensive — no Edge Impulse support | ₹650 | Amazon India | Robu.in |
| **Total (essential)** | | **~₹4,070** | | |

### Optional Add-ons (Week 2+ if budget allows)

| Component | Purpose | Cost | When to Buy |
|-----------|---------|------|-------------|
| AD8232 ECG Module + electrodes | ECG signal capture for biosensing | ₹450 | Week 2 if MAX30102 arrives and works |
| 16GB microSD Class 10 | Storage for future Pi 5 | ₹350 | Week 4 if continuing to Month 2 |
| Raspberry Pi 5 (4GB) | Primary edge inference board | ₹7,500 | Month 2 if validated interest |

### Vendor Strategy for India

**Amazon India:** Fastest delivery (1-2 days Prime), easy returns. Slightly higher prices.
**Robu.in:** Lower prices, specializes in robotics/electronics. 3-5 day delivery. Good for bulk orders.
**Evelta.com:** Professional electronics distributor. Reliable but slower.
**Local electronics markets:** SP Road (Bengaluru), Lamington Road (Mumbai), Chandni Chowk (Delhi). Cheapest prices but requires in-person visit.

**Shipping hedge strategy:** Order from Amazon India AND Robu.in simultaneously. Cancel the slower one when the first arrives.

---

## 4. Free GPU Compute — Exact Login Links and Weekly Limits

### Tier 1: Completely Free (Zero INR)

| Platform | GPU | Weekly Limit | Login URL | Best For | Notes |
|----------|-----|--------------|-----------|----------|-------|
| **Kaggle Notebooks** | Dual T4 (30GB combined VRAM) | 30 hrs GPU + 20 hrs TPU v3-8 | [kaggle.com/code](https://www.kaggle.com/code) | Training 1D-CNN, QLoRA 7B models | 12hr max session. Save to `/kaggle/working/` |
| **Google Colab** | T4 (15GB VRAM) | ~4-6 hrs/day | [colab.research.google.com](https://colab.research.google.com) | Quick experiments, hyperparameter sweeps | Disconnects after ~90min idle. Mount Drive for persistence. |
| **Lightning AI Studios** | T4 / L4 / A10G | 22 GPU-hrs/month | [lightning.ai](https://lightning.ai) | Full VS Code environment, dev work | Best IDE experience of free tiers |
| **Paperspace Gradient** | M4000 (8GB) | 6 hrs/session | [gradient.paperspace.com](https://gradient.paperspace.com) | Persistent notebooks | Requeue after session ends |
| **Saturn Cloud** | T4 | 30 hrs/month | [saturncloud.io](https://saturncloud.io) | Dask integration | Good for distributed data processing |

### Tier 2: India-Specific Government Resources (Free/Subsidized)

| Platform | GPU | Cost | Login URL | Eligibility | Notes |
|----------|-----|------|-----------|-------------|-------|
| **AIKosh** | NVIDIA A100 (5GB & 20GB) | **FREE** | [aikosh.indiaai.gov.in](https://aikosh.indiaai.gov.in) | Indian citizens | 4 hrs/day fixed slots. Files deleted after session. |
| **IndiaAI Compute Portal** | A100, H100 | **Subsidized** | [compute.indiaai.gov.in](https://compute.indiaai.gov.in) | DigiLocker/e-Pramaan login | <5,000 GPU hours auto-approved. INR billing. |
| **AMD Developer Cloud** | AMD Instinct MI300X | **100K free hours** | [amd.com/en/blogs/2025/100k-hours-free-developer-cloud-access.html](https://www.amd.com/en/blogs/2025/100k-hours-free-developer-cloud-access.html) | Researchers, startups | Requires application. Training in ROCm. |

### Tier 3: Cheapest Paid (Use from Month 7+)

| Provider | GPU | Approx INR/hr | URL | Best For |
|----------|-----|---------------|-----|----------|
| **JarvisLabs** | RTX 3090 | ~₹40-50/hr | [jarvislabs.ai](https://jarvislabs.ai) | Small experiments, per-minute billing |
| **E2E Networks** | A100 | ~₹120-150/hr | [e2enetworks.com](https://e2enetworks.com) | India DC, INR billing, reliable |
| **Vast.ai** | A100 (spot) | ~₹40-100/hr | [vast.ai](https://vast.ai) | Overnight training, cheapest globally |
| **RunPod Community** | RTX 4090 | ~₹25-35/hr | [runpod.io](https://runpod.io) | Community cloud, risk of preemption |

**Recommended workflow for this plan:**
1. **Primary:** Kaggle Notebooks for all model training (free, reproducible, versioned)
2. **Secondary:** Google Colab for quick tests and longer training jobs
3. **Tertiary:** Lightning AI for development environment (VS Code in browser)
4. **Never needed in Month 1:** Paid GPU (models are tiny, train in minutes on T4)

---

## 5. Week 1: Foundation — Day-by-Day Implementation

### Theme: "I can read a biosignal and I'm now visible on the internet."

---

### ⭐ Day 1 (Monday) — Procurement + Environment Setup [1.5 hrs]

**Hardware actions:**
- [ ] Order Arduino Nano 33 BLE Sense from Amazon India (Prime delivery if available)
- [ ] Order MPU-6050 + breadboard + jumper wires from Amazon India
- [ ] If same-day delivery unavailable, also order from Robu.in as backup
- [ ] Set calendar reminder for delivery date (typically 2-3 days)

**Software actions:**

Step 1: Create project directory
```bash
mkdir -p ~/edge-ml-pivot
cd ~/edge-ml-pivot
git init
git remote add origin https://github.com/YOUR_USERNAME/edge-ml-pivot.git
```

Step 2: Create conda environment
```bash
conda create -n edgeml python=3.11 -y
conda activate edgeml
```

Step 3: Install core packages
```bash
# PyTorch (CPU version is fine for Month 1 — training on Kaggle GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# TensorFlow (for TFLite conversion)
pip install tensorflow==2.15.0

# ONNX ecosystem
pip install onnx onnxruntime onnx-simplifier

# Signal processing / biosensing
pip install neurokit2 scipy matplotlib pandas numpy jupyter wfdb heartpy biosppy

# Quantization / compression
pip install torchao optimum neural-compressor

# Edge AI tools
pip install edgeimpulse

# Hardware communication (for later weeks)
pip install pyserial

# Visualization (your existing skill)
pip install streamlit plotly kaleido

# Dev tools
pip install torchinfo fvcore thop
```

Step 4: Create GitHub repo structure
```bash
mkdir -p week1 week2 week3 week4 data models benchmarks docs
 touch README.md .gitignore requirements.txt
```

Step 5: Download datasets
```bash
# Create data directory
mkdir -p data/ecg data/gesture

# PhysioNet MIT-BIH Arrhythmia Database
# Go to: https://physionet.org/content/mitdb/1.0.0/
# Create free account, request access (usually instant)
# Download all .dat and .hea files (~11MB total)
# Place in data/ecg/

# UCI HAR Dataset (Human Activity Recognition)
# Go to: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using-smartphones
# Direct download, no auth needed
# Place in data/gesture/
```

Step 6: Create `.gitignore`
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.env
.venv
env/
venv/

# Data (too large for git)
data/
*.dat
*.hea
*.csv

# Models (use Git LFS or releases)
models/*.pt
models/*.onnx
models/*.tflite

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

Step 7: Initial commit
```bash
git add .
git commit -m "Initial project structure for edge-ml-pivot"
git push -u origin main
```

**Deliverable:** Working Python env, empty GitHub repo with structure, datasets downloaded.

---

### Day 2 (Tuesday) — Signal Processing Basics [1.5 hrs]

**Learning objectives:**
- Understand ECG signal structure: P-wave, QRS complex, T-wave
- Learn why filtering matters: baseline wander, powerline interference (50Hz in India)
- Practice with scipy.signal and wfdb

**Reading (30 min):**
- PhysioNet tutorial: "Introduction to ECG Signal Processing" (free, online)
- MIT OpenCourseWare 6.003 Signals & Systems — Lecture 1 notes (free PDF)

**Implementation (1 hr):**

Create `week1/ecg_explore.py`:
```python
"""
Week 1, Day 2: ECG Signal Exploration
Author: Abhishek Bhardwaj
Dataset: PhysioNet MIT-BIH Arrhythmia Database
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import wfdb
import os

# Ensure output directory exists
os.makedirs('week1/outputs', exist_ok=True)

# Load MIT-BIH record 100 (first 10 seconds @ 360Hz)
# Download from https://physionet.org/content/mitdb/1.0.0/
record = wfdb.rdrecord('data/ecg/100', sampto=3000)
ecg = record.p_signal[:, 0]  # First channel (MLII lead)
fs = record.fs  # Sampling frequency: 360 Hz

print(f"Loaded {len(ecg)} samples at {fs} Hz")
print(f"Duration: {len(ecg)/fs:.1f} seconds")

# Design bandpass filter: 0.5-40 Hz
# Removes baseline wander (low freq) and high-frequency noise
def bandpass_filter(signal, fs, low=0.5, high=40, order=4):
    """
    Butterworth bandpass filter for ECG signals.
    
    Args:
        signal: Raw ECG signal (1D numpy array)
        fs: Sampling frequency in Hz
        low: Low cutoff frequency (Hz)
        high: High cutoff frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered signal
    """
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, signal)

filtered = bandpass_filter(ecg, fs)

# R-peak detection using scipy.find_peaks on squared derivative
# This is a simple approach; neurokit2 has more robust methods
derivative = np.diff(filtered)
squared = derivative ** 2
peaks, properties = find_peaks(
    squared, 
    distance=int(fs * 0.6),  # Minimum 0.6s between peaks (max ~100 BPM)
    prominence=np.std(squared) * 0.5
)

# Map peaks back to original signal indices
r_peaks = peaks + 1  # +1 because np.diff reduces length by 1

# Calculate heart rate
rr_intervals = np.diff(r_peaks) / fs  # in seconds
heart_rate = 60 / np.mean(rr_intervals)  # BPM
print(f"Estimated heart rate: {heart_rate:.1f} BPM")

# Plot raw vs filtered with R-peaks
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Raw signal
time = np.arange(len(ecg)) / fs
axes[0].plot(time, ecg, color='gray', alpha=0.7)
axes[0].set_title('Raw ECG (Record 100, First 10s)', fontsize=12)
axes[0].set_ylabel('Amplitude (mV)')
axes[0].grid(True, alpha=0.3)

# Filtered signal with R-peaks
axes[1].plot(time, filtered, color='blue', linewidth=1.2)
axes[1].scatter(r_peaks / fs, filtered[r_peaks], color='red', s=50, zorder=5, label='R-peaks')
axes[1].set_title(f'Filtered ECG (0.5-40 Hz) + R-peaks | HR: {heart_rate:.1f} BPM', fontsize=12)
axes[1].set_ylabel('Amplitude (mV)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Zoomed QRS complex (first detected peak)
if len(r_peaks) > 0:
    peak_idx = r_peaks[0]
    window = 100  # samples around peak
    start = max(0, peak_idx - window)
    end = min(len(filtered), peak_idx + window)
    zoom_time = np.arange(start, end) / fs
    axes[2].plot(zoom_time, filtered[start:end], color='blue', linewidth=1.5)
    axes[2].scatter(peak_idx / fs, filtered[peak_idx], color='red', s=100, zorder=5)
    axes[2].set_title('Zoomed QRS Complex (First R-peak)', fontsize=12)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude (mV)')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('week1/outputs/ecg_baseline.png', dpi=150, bbox_inches='tight')
print("Saved: week1/outputs/ecg_baseline.png")

# Save summary
with open('week1/outputs/day2_summary.txt', 'w') as f:
    f.write(f"ECG Exploration Summary\n")
    f.write(f"{'='*40}\n")
    f.write(f"Record: MIT-BIH 100\n")
    f.write(f"Samples: {len(ecg)}\n")
    f.write(f"Duration: {len(ecg)/fs:.1f}s\n")
    f.write(f"Sampling rate: {fs} Hz\n")
    f.write(f"R-peaks detected: {len(r_peaks)}\n")
    f.write(f"Estimated HR: {heart_rate:.1f} BPM\n")
    f.write(f"Filter: Butterworth bandpass 0.5-40 Hz, order 4\n")
```

Run it:
```bash
cd ~/edge-ml-pivot
python week1/ecg_explore.py
```

**Deliverable:** `week1/ecg_explore.py` + `week1/outputs/ecg_baseline.png` committed to repo.

---

### Day 3 (Wednesday) — Feature Extraction + Heart Rate Variability [1.5 hrs]

**Learning objectives:**
- Understand HRV (Heart Rate Variability) as a biomarker
- Learn time-domain features: SDNN, RMSSD, pNN50
- Learn frequency-domain features: LF, HF, LF/HF ratio
- Use NeuroKit2 for professional-grade ECG processing

**Implementation:**

Create `week1/ecg_hrv.py`:
```python
"""
Week 1, Day 3: HRV Feature Extraction from ECG
Uses NeuroKit2 for robust peak detection and HRV analysis
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('week1/outputs', exist_ok=True)

# Load MIT-BIH record 100 (full record for better HRV stats)
record = wfdb.rdrecord('data/ecg/100')
ecg = record.p_signal[:, 0]
fs = record.fs

# NeuroKit2 pipeline: clean → detect peaks → compute HRV
print("Running NeuroKit2 ECG pipeline...")

# Clean ECG signal
ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=fs, method='neurokit')

# Detect R-peaks
peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method='neurokit')

# Calculate HRV metrics
hrv_time = nk.hrv_time(peaks, sampling_rate=fs, show=False)
hrv_freq = nk.hrv_frequency(peaks, sampling_rate=fs, show=False)
hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=fs, show=False)

# Combine all HRV features
hrv_features = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)

print("\nHRV Features (Record 100):")
print("-" * 40)
print(f"SDNN: {hrv_features['HRV_SDNN'].values[0]:.2f} ms")
print(f"RMSSD: {hrv_features['HRV_RMSSD'].values[0]:.2f} ms")
print(f"pNN50: {hrv_features['HRV_pNN50'].values[0]:.2f}%")
print(f"LF Power: {hrv_features['HRV_LF'].values[0]:.2f} ms²")
print(f"HF Power: {hrv_features['HRV_HF'].values[0]:.2f} ms²")
print(f"LF/HF Ratio: {hrv_features['HRV_LFHF'].values[0]:.2f}")

# SDNN interpretation:
# <50 ms: High stress / poor recovery
# 50-100 ms: Normal
# >100 ms: Good recovery / high vagal tone
sdnn = hrv_features['HRV_SDNN'].values[0]
if sdnn < 50:
    interpretation = "Low HRV: Indicates stress, poor recovery, or potential cardiac risk"
elif sdnn < 100:
    interpretation = "Normal HRV: Healthy autonomic balance"
else:
    interpretation = "High HRV: Good recovery, high vagal tone"

print(f"\nInterpretation: {interpretation}")

# Save features to CSV
hrv_features.to_csv('week1/outputs/hrv_features_100.csv', index=False)
print("\nSaved: week1/outputs/hrv_features_100.csv")

# Plot ECG with detected peaks
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

time = np.arange(len(ecg_cleaned)) / fs
axes[0].plot(time, ecg_cleaned, color='blue', linewidth=0.8)
peak_times = info['ECG_R_Peaks'] / fs
axes[0].scatter(peak_times, ecg_cleaned[info['ECG_R_Peaks']], 
                color='red', s=30, zorder=5, label='R-peaks')
axes[0].set_title('ECG with NeuroKit2 R-peak Detection (Record 100)', fontsize=12)
axes[0].set_ylabel('Amplitude (mV)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# HRV tachogram (RR intervals over time)
rr_intervals = np.diff(info['ECG_R_Peaks']) / fs * 1000  # ms
rr_time = peak_times[1:]  # Time points for RR intervals
axes[1].plot(rr_time, rr_intervals, color='green', marker='o', markersize=3, linewidth=1)
axes[1].axhline(y=np.mean(rr_intervals), color='red', linestyle='--', label=f'Mean: {np.mean(rr_intervals):.1f} ms')
axes[1].set_title('RR Interval Tachogram', fontsize=12)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('RR Interval (ms)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('week1/outputs/ecg_hrv_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: week1/outputs/ecg_hrv_analysis.png")

# Batch process multiple records for downstream ML
records_to_process = ['100', '101', '102', '103', '104']
all_features = []

print("\nBatch processing multiple records...")
for rec_id in records_to_process:
    try:
        rec = wfdb.rdrecord(f'data/ecg/{rec_id}')
        sig = rec.p_signal[:, 0]
        cleaned = nk.ecg_clean(sig, sampling_rate=rec.fs, method='neurokit')
        pks, inf = nk.ecg_peaks(cleaned, sampling_rate=rec.fs)
        hrv = nk.hrv_time(pks, sampling_rate=rec.fs, show=False)
        hrv['record_id'] = rec_id
        all_features.append(hrv)
        print(f"  Processed record {rec_id}")
    except Exception as e:
        print(f"  Skipped record {rec_id}: {e}")

if all_features:
    batch_df = pd.concat(all_features, ignore_index=True)
    batch_df.to_csv('week1/outputs/hrv_batch_features.csv', index=False)
    print("\nSaved: week1/outputs/hrv_batch_features.csv")
    print(f"Total records processed: {len(all_features)}")
```

Run it:
```bash
python week1/ecg_hrv.py
```

**Deliverable:** `week1/ecg_hrv.py` + `week1/outputs/hrv_batch_features.csv` committed to repo.

---

### Day 4 (Thursday) — Online Presence Setup [1.5 hrs]

**LinkedIn optimization:**

Step 1: Update headline
```
Senior Data Scientist | Building at the intersection of Deep ML, Edge AI & Biosignals | Documenting the journey in public
```

Step 2: Add skills
- Edge AI & Embedded ML
- TinyML
- PyTorch
- Signal Processing
- Quantization
- Open Source

Step 3: Write first post (save as draft, publish Friday evening 7:30 AM IST for US visibility)

```
Week 1 of my pivot from cloud-scale LLMs to edge AI.

What I'm doing:
→ Training a 1D-CNN to detect cardiac arrhythmia from ECG signals
→ Deploying it on an Arduino Nano 33 BLE Sense (< 256KB RAM)
→ Documenting quantization trade-offs in public

Why? Because Indian semiconductor startups like Sophrosyne need ML engineers who understand both the model AND the silicon it runs on.

Today's win: I extracted SDNN and RMSSD features from the MIT-BIH database using NeuroKit2. Turns out signal processing is just feature engineering with physics.

If you're curious about edge AI, follow along. I'll share benchmarks, failures, and the occasional "why did this catch fire?" moment.

#TinyML #EdgeAI #IndiaSemiconductor #MachineLearning #SignalProcessing
```

**Twitter/X setup:**
- Create account or pivot existing to tech focus
- Follow these exact accounts: @tinyMLsummit, @EdgeImpulse, @hanlab_mit, @karpathy, @AndrewYNg, @ApacheTVM, @mlc_llm, @ggerganov
- First tweet (save as draft):
```
I'm spending June learning how to squeeze neural networks into devices smaller than my thumb.

Week 1: ECG signal processing + HRV feature extraction from MIT-BIH database.

Next: Quantization-aware training → INT8 → Arduino deployment.

Follow along if you like watching models shrink.
```

**Reddit setup:**
- Join: r/MachineLearning, r/embedded, r/LocalLLaMA, r/IndianStartups
- Lurk and comment on 2 posts this week (no self-promotion yet, build karma)
- Read subreddit rules carefully — r/MachineLearning requires [P], [R], [D] flairs

**GitHub profile optimization:**
- Pin this repo once it goes public (Week 4)
- Add bio: "Senior Data Scientist → Edge AI. Building TinyML biosensing systems. PyTorch | Signal Processing | Quantization"
- Add location: India
- Add README to your profile repo (github.com/YOUR_USERNAME/YOUR_USERNAME)

**Deliverable:** Live social presence. First post drafted. 8+ relevant accounts followed.

---

### Day 5 (Friday) — Arduino Arrival Prep + Edge Impulse Account [1.5 hrs]

**Software:**
- [ ] Download Arduino IDE 2.x from [arduino.cc/en/software](https://www.arduino.cc/en/software)
- [ ] Install Edge Impulse CLI:
```bash
npm install -g edge-impulse-cli
```
- [ ] Create Edge Impulse account (free): [studio.edgeimpulse.com](https://studio.edgeimpulse.com)
- [ ] Create new project: "gesture-tinyml-month1"
- [ ] Read: Edge Impulse "Continuous motion recognition" tutorial (30 min): [docs.edgeimpulse.com](https://docs.edgeimpulse.com)

**Hardware prep (even if not arrived):**
- [ ] Watch: "Arduino Nano 33 BLE Sense getting started" on Edge Impulse YouTube channel
- [ ] Print pinout diagram for Nano 33 BLE Sense:
```
Arduino Nano 33 BLE Sense Pinout (Key Pins)
============================================
D13 (LED)        - Built-in LED
A4 (SDA)         - I2C Data (connect to MPU-6050 SDA)
A5 (SCL)         - I2C Clock (connect to MPU-6050 SCL)
3.3V             - Power out (connect to MPU-6050 VCC)
GND              - Ground (connect to MPU-6050 GND)
VIN              - External power input
USB              - Programming / Serial
Reset            - Reset button
```

**Deliverable:** Edge Impulse project created, Arduino IDE installed, pinout understood.

---

### Weekend (Sat + Sun) — Buffer + Planning [3 hrs Sat, 1 hr Sun]

**Saturday:**
- [ ] If Arduino arrived: Connect via USB, flash built-in LED blink example
```cpp
// Arduino IDE → File → Examples → 01.Basics → Blink
// Select board: "Arduino Nano 33 BLE"
// Select port: COMx (Windows) or /dev/ttyACM0 (Linux)
// Upload → verify LED blinks
```
- [ ] If Arduino not arrived: Continue signal processing — try PPG simulation using neurokit2
```python
import neurokit2 as nk
ppg = nk.ppg_simulate(duration=10, sampling_rate=100)
nk.ppg_plot(ppg, sampling_rate=100)
```
- [ ] Write `week1/summary.md` documenting what you learned

**Sunday:**
- [ ] Plan Week 2 tasks
- [ ] Read ahead: PyTorch Quantization tutorial [pytorch.org/tutorials/advanced/static_quantization_tutorial.html](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [ ] Draft LinkedIn post for Week 2 (saves time mid-week)

**Week 1 OSS Target:**
- [ ] ⭐ File 1 GitHub issue on `neurokit2/neurokit2`: "Documentation request: Add example for batch processing multiple ECG records and exporting HRV features to CSV"
- Keep it polite, specific, and reference your use case

---

## 6. Week 2: Deep ML + Quantization — Day-by-Day Implementation

### Theme: "I can train a deep model and make it 10x smaller without breaking it."

---

### Day 6 (Monday) — 1D-CNN for ECG Classification [1.5 hrs]

**Learning objectives:**
- Understand why 1D-CNNs work for time-series (local pattern detection across time)
- Design a model small enough for edge deployment (< 50KB after INT8)
- Train on Kaggle free GPU

**Implementation:**

Create a Kaggle Notebook: `edge-ml-week2-ecg-cnn`

```python
# %% [markdown]
# ## Week 2, Day 6: Train 1D-CNN for ECG Arrhythmia Detection
# **Author:** Abhishek Bhardwaj
# **Hardware target:** Arduino Nano 33 BLE Sense (256KB RAM, 1MB Flash)
# **Goal:** Binary classification: Normal vs Arrhythmia
# **Model constraint:** < 50KB after INT8 quantization

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import wfdb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# %% [markdown]
# ### Data Preparation
# We'll use MIT-BIH records, labeling based on annotation symbols.
# Normal rhythm: N, L, R, e, j
# Arrhythmia: A, a, J, S, V, E, F, /, f, Q, !

# %%
class ECGDataset(Dataset):
    """
    MIT-BIH ECG Dataset for binary classification.
    
    Each sample is a 5-second window (1800 samples @ 360Hz).
    Label: 0 = Normal, 1 = Arrhythmia
    """
    def __init__(self, record_ids, window_size=1800, target_sr=360):
        self.samples = []
        self.labels = []
        
        for rec_id in record_ids:
            try:
                # Read signal and annotations
                record = wfdb.rdrecord(f'/kaggle/input/mitbih-database/mitbih_database/{rec_id}')
                annotation = wfdb.rdann(f'/kaggle/input/mitbih-database/mitbih_database/{rec_id}', 'atr')
                
                signal = record.p_signal[:, 0]  # MLII lead
                fs = record.fs
                
                # Resample if needed
                if fs != target_sr:
                    from scipy.signal import resample
                    num_samples = int(len(signal) * target_sr / fs)
                    signal = resample(signal, num_samples)
                
                # Extract windows with labels
                for i in range(0, len(signal) - window_size, window_size // 2):  # 50% overlap
                    window = signal[i:i + window_size]
                    
                    # Determine label from annotations in this window
                    window_start = i
                    window_end = i + window_size
                    
                    # Find annotations in window
                    ann_in_window = [
                        sym for idx, sym in zip(annotation.sample, annotation.symbol)
                        if window_start <= idx < window_end
                    ]
                    
                    # Label: 1 if any arrhythmia symbol present
                    arrhythmia_symbols = {'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q', '!}
                    has_arrhythmia = any(sym in arrhythmia_symbols for sym in ann_in_window)
                    
                    # Z-score normalize
                    window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                    
                    self.samples.append(window)
                    self.labels.append(1 if has_arrhythmia else 0)
                    
            except Exception as e:
                print(f"Skipping record {rec_id}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)  # (1, 1800)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Load dataset
record_ids = [str(i) for i in range(100, 125)]  # Records 100-124
dataset = ECGDataset(record_ids)

# Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# %% [markdown]
# ### Model: Tiny 1D-CNN for Edge Deployment
# Design constraints:
# - < 30K parameters (will be ~15KB in INT8)
# - No BatchNorm (problematic for INT8 TFLite)
# - Use GroupNorm or LayerNorm instead

# %%
class TinyECGNet(nn.Module):
    """
    Tiny 1D-CNN for ECG arrhythmia detection.
    
    Target: < 30K parameters, < 50KB INT8 model
    """
    def __init__(self, num_classes=2):
        super(TinyECGNet, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 1800 -> 900
            nn.Conv1d(1, 16, kernel_size=25, stride=2, padding=12),
            nn.GroupNorm(4, 16),  # GroupNorm instead of BatchNorm for INT8 compatibility
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Block 2: 900 -> 450
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Block 3: 450 -> 225
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Initialize model
model = TinyECGNet().to(DEVICE)
print(f"Model parameters: {model.count_parameters():,}")
print(f"Estimated FP32 size: {model.count_parameters() * 4 / 1024:.1f} KB")

# %% [markdown]
# ### Training Loop

# %%
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_targets

# Training
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\nStarting training...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, preds, targets = validate(model, val_loader, criterion)
    scheduler.step()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_ecg_model.pt')
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

print(f"\nBest validation accuracy: {best_val_acc:.4f}")

# %% [markdown]
# ### Evaluation

# %%
# Load best model
model.load_state_dict(torch.load('best_ecg_model.pt'))
_, _, preds, targets = validate(model, val_loader, criterion)

print("\nClassification Report:")
print(classification_report(targets, preds, target_names=['Normal', 'Arrhythmia']))

# Confusion matrix
cm = confusion_matrix(targets, preds)
print(f"\nConfusion Matrix:\n{cm}")

# Save model for quantization
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/ecg_model.pt')
print("\nSaved: models/ecg_model.pt")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses, label='Val')
axes[0].set_title('Loss Curves')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Train')
axes[1].plot(val_accs, label='Val')
axes[1].set_title('Accuracy Curves')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("Saved: training_curves.png")
```

**Deliverable:** Trained `ecg_model.pt`, accuracy logged, training curves saved.

---

### Day 7 (Tuesday) — Quantization-Aware Training (QAT) in PyTorch [1.5 hrs]

**Learning objectives:**
- Understand difference between PTQ (Post-Training Quantization) and QAT
- Learn why QAT preserves accuracy better for tiny models
- Implement QAT with PyTorch FX Graph Mode

**Reading (30 min):**
- PyTorch QAT tutorial: [pytorch.org/tutorials/advanced/static_quantization_tutorial.html](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- Read: "Why QAT outperforms PTQ on small architectures" (understand the intuition)

**Implementation:**

Create `week2/quantize_ecg.py` (run on Kaggle or locally):
```python
"""
Week 2, Day 7: Quantization-Aware Training (QAT) for ECG Model
Compares: FP32 baseline, PTQ, QAT
"""

import torch
import torch.quantization
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
import copy
import time
import os

os.makedirs('week2/outputs', exist_ok=True)

# Load trained model
model_fp32 = TinyECGNet()
model_fp32.load_state_dict(torch.load('models/ecg_model.pt'))
model_fp32.eval()

# ========== BASELINE: FP32 ==========
print("=" * 50)
print("BASELINE: FP32 Model")
print("=" * 50)

# Measure size
import pickle
fp32_size = len(pickle.dumps(model_fp32.state_dict()))
print(f"Model size: {fp32_size / 1024:.1f} KB")

# Benchmark latency
def benchmark_latency(model, input_shape=(1, 1, 1800), n=100):
    dummy = torch.randn(*input_shape)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    # Benchmark
    start = time.time()
    for _ in range(n):
        with torch.no_grad():
            model(dummy)
    elapsed = time.time() - start
    return (elapsed / n) * 1000  # ms

fp32_latency = benchmark_latency(model_fp32)
print(f"Inference latency: {fp32_latency:.2f} ms")

# Validate FP32
_, fp32_acc, _, _ = validate(model_fp32, val_loader, criterion)
print(f"Validation accuracy: {fp32_acc:.4f}")

# ========== METHOD 1: Post-Training Quantization (PTQ) ==========
print("\n" + "=" * 50)
print("METHOD 1: Post-Training Quantization (PTQ)")
print("=" * 50)

model_ptq = copy.deepcopy(model_fp32)
model_ptq.eval()

# Fuse Conv+ReLU where possible (improves quantization)
# Note: Our model uses GroupNorm, which cannot be fused. 
# For production, consider replacing GroupNorm with BatchNorm + fusion.

model_ptq.qconfig = get_default_qconfig('x86')
model_ptq_prepared = torch.quantization.prepare(model_ptq)

# Calibrate with representative data
print("Calibrating with validation data...")
with torch.no_grad():
    for i, (data, _) in enumerate(val_loader):
        if i >= 10:  # Use first 10 batches for calibration
            break
        model_ptq_prepared(data)

model_ptq_int8 = torch.quantization.convert(model_ptq_prepared)

# Benchmark
ptq_size = len(pickle.dumps(model_ptq_int8.state_dict()))
ptq_latency = benchmark_latency(model_ptq_int8)
_, ptq_acc, _, _ = validate(model_ptq_int8, val_loader, criterion)

print(f"Model size: {ptq_size / 1024:.1f} KB ({fp32_size/ptq_size:.1f}x smaller)")
print(f"Inference latency: {ptq_latency:.2f} ms ({fp32_latency/ptq_latency:.1f}x faster)")
print(f"Validation accuracy: {ptq_acc:.4f} (drop: {fp32_acc - ptq_acc:.4f})")

# ========== METHOD 2: Quantization-Aware Training (QAT) ==========
print("\n" + "=" * 50)
print("METHOD 2: Quantization-Aware Training (QAT)")
print("=" * 50)

model_qat = copy.deepcopy(model_fp32)
model_qat.train()  # Must be in train mode for QAT

# QAT config
model_qat.qconfig = get_default_qat_qconfig('x86')
model_qat_prepared = prepare_qat(model_qat)

# Fine-tune with simulated quantization
optimizer_qat = torch.optim.AdamW(model_qat_prepared.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

print("Fine-tuning with QAT (5 epochs)...")
for epoch in range(5):
    model_qat_prepared.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target  # Keep on CPU for QAT
        optimizer_qat.zero_grad()
        output = model_qat_prepared(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_qat.step()
        total_loss += loss.item()
    
    # Evaluate
    model_qat_prepared.eval()
    _, qat_acc, _, _ = validate(model_qat_prepared, val_loader, criterion)
    print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {qat_acc:.4f}")

# Convert to quantized model
model_qat_int8 = convert(model_qat_prepared)

# Benchmark
qat_size = len(pickle.dumps(model_qat_int8.state_dict()))
qat_latency = benchmark_latency(model_qat_int8)
_, qat_acc_final, _, _ = validate(model_qat_int8, val_loader, criterion)

print(f"\nModel size: {qat_size / 1024:.1f} KB ({fp32_size/qat_size:.1f}x smaller)")
print(f"Inference latency: {qat_latency:.2f} ms ({fp32_latency/qat_latency:.1f}x faster)")
print(f"Validation accuracy: {qat_acc_final:.4f} (drop: {fp32_acc - qat_acc_final:.4f})")

# ========== COMPARISON TABLE ==========
print("\n" + "=" * 50)
print("QUANTIZATION BENCHMARK RESULTS")
print("=" * 50)

results = {
    'Method': ['FP32', 'PTQ', 'QAT'],
    'Size (KB)': [f"{fp32_size/1024:.1f}", f"{ptq_size/1024:.1f}", f"{qat_size/1024:.1f}"],
    'Size Reduction': ['1.0x', f"{fp32_size/ptq_size:.1f}x", f"{fp32_size/qat_size:.1f}x"],
    'Latency (ms)': [f"{fp32_latency:.2f}", f"{ptq_latency:.2f}", f"{qat_latency:.2f}"],
    'Speedup': ['1.0x', f"{fp32_latency/ptq_latency:.1f}x", f"{fp32_latency/qat_latency:.1f}x"],
    'Accuracy': [f"{fp32_acc:.4f}", f"{ptq_acc:.4f}", f"{qat_acc_final:.4f}"],
    'Accuracy Drop': ['0.0000', f"{fp32_acc - ptq_acc:.4f}", f"{fp32_acc - qat_acc_final:.4f}"]
}

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save
results_df.to_csv('week2/outputs/quantization_benchmark.csv', index=False)
print("\nSaved: week2/outputs/quantization_benchmark.csv")

# Save best model (QAT if better, else FP32)
if qat_acc_final >= ptq_acc:
    torch.save(model_qat_int8.state_dict(), 'models/ecg_model_int8.pt')
    print("Saved best quantized model: models/ecg_model_int8.pt (QAT)")
else:
    torch.save(model_ptq_int8.state_dict(), 'models/ecg_model_int8.pt')
    print("Saved best quantized model: models/ecg_model_int8.pt (PTQ)")
```

**Deliverable:** `week2/outputs/quantization_benchmark.csv` with size, latency, and accuracy comparison.

---

### Day 8 (Wednesday) — ONNX Export + TFLite Micro Conversion [1.5 hrs]

**Learning objectives:**
- Understand ONNX as the interoperability bridge between PyTorch and edge runtimes
- Convert quantized model to TFLite format for Arduino deployment
- Measure final model size — target < 50KB

**Implementation:**

Create `week2/convert_to_tflite.py`:
```python
"""
Week 2, Day 8: Convert PyTorch → ONNX → TensorFlow → TFLite
Target: < 50KB INT8 model for Arduino Nano 33 BLE Sense
"""

import torch
import onnx
import tensorflow as tf
import numpy as np
import os

os.makedirs('week2/outputs', exist_ok=True)

# Load quantized model
model = TinyECGNet()
model.load_state_dict(torch.load('models/ecg_model_int8.pt'))
model.eval()

# Step 1: Export to ONNX
dummy_input = torch.randn(1, 1, 1800)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    'week2/outputs/ecg_model.onnx',
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Verify ONNX model
onnx_model = onnx.load('week2/outputs/ecg_model.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX model verified successfully")

# Print model info
print(f"ONNX model size: {os.path.getsize('week2/outputs/ecg_model.onnx') / 1024:.1f} KB")

# Step 2: Convert ONNX → TensorFlow (requires onnx-tf)
# pip install onnx-tf
print("\nConverting ONNX to TensorFlow...")
from onnx_tf.backend import prepare
tf_rep = prepare(onnx_model)
tf_rep.export_graph('week2/outputs/ecg_model_tf')
print("TensorFlow SavedModel exported")

# Step 3: Convert TensorFlow → TFLite with INT8 quantization
print("\nConverting to TFLite with INT8 quantization...")

converter = tf.lite.TFLiteConverter.from_saved_model('week2/outputs/ecg_model_tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

# Representative dataset for calibration
def representative_dataset():
    for i in range(100):
        # Generate representative ECG-like data
        data = np.random.randn(1, 1, 1800).astype(np.float32) * 0.5
        yield [data]

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

# Save TFLite model
with open('week2/outputs/ecg_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model size: {os.path.getsize('week2/outputs/ecg_model.tflite') / 1024:.1f} KB")

# Verify with TFLite interpreter
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\nTFLite model details:")
print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Input dtype: {input_details[0]['dtype']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

# Test inference
import time
test_input = np.random.randn(1, 1, 1800).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)

start = time.time()
interpreter.invoke()
latency = (time.time() - start) * 1000

output = interpreter.get_tensor(output_details[0]['index'])
print(f"\nTest inference: {latency:.2f} ms")
print(f"Output: {output}")

# Summary
print("\n" + "=" * 50)
print("CONVERSION SUMMARY")
print("=" * 50)
print(f"PyTorch FP32:     {os.path.getsize('models/ecg_model.pt') / 1024:.1f} KB")
print(f"ONNX:             {os.path.getsize('week2/outputs/ecg_model.onnx') / 1024:.1f} KB")
print(f"TFLite INT8:      {os.path.getsize('week2/outputs/ecg_model.tflite') / 1024:.1f} KB")
print(f"\nTarget: < 50KB for Arduino Nano 33 BLE Sense")
if os.path.getsize('week2/outputs/ecg_model.tflite') < 50 * 1024:
    print("✅ Target achieved!")
else:
    print("❌ Target NOT achieved. Reduce model size further.")
```

**If model is too big:** Reduce Conv1d filters (16→8, 32→16, 64→32) and retrain.

**Deliverable:** `week2/outputs/ecg_model.tflite` under 50KB.

---

### Day 9 (Thursday) — MPU-6050 Gesture Data Collection [1.5 hrs]

**Hardware (Arduino should have arrived):**

**Wiring diagram:**
```
MPU-6050          Arduino Nano 33 BLE Sense
--------          ------------------------
VCC    ---------> 3.3V
GND    ---------> GND
SDA    ---------> A4 (SDA)
SCL    ---------> A5 (SCL)
```

**Implementation:**

Flash `week2/collect_imu.ino`:
```cpp
/*
  Week 2, Day 9: IMU Data Collection for Gesture Recognition
  Target: Arduino Nano 33 BLE Sense (onboard IMU)
  Author: Abhishek Bhardwaj
  
  Collects accelerometer + gyroscope data at 20Hz
  Outputs CSV format over Serial for Python ingestion
*/

#include <Arduino_LSM9DS1.h>  // Onboard IMU library

// Sampling configuration
const float SAMPLE_RATE_HZ = 20.0;
const unsigned long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;

void setup() {
  Serial.begin(9600);
  while (!Serial);  // Wait for Serial connection
  
  if (!IMU.begin()) {
    Serial.println("ERROR: IMU initialization failed!");
    while (1);  // Halt
  }
  
  Serial.println("timestamp_ms,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z");
}

void loop() {
  float ax, ay, az;   // Accelerometer (g)
  float gx, gy, gz;   // Gyroscope (degrees/sec)
  
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    
    Serial.print(millis());
    Serial.print(",");
    Serial.print(ax, 4);
    Serial.print(",");
    Serial.print(ay, 4);
    Serial.print(",");
    Serial.print(az, 4);
    Serial.print(",");
    Serial.print(gx, 4);
    Serial.print(",");
    Serial.print(gy, 4);
    Serial.print(",");
    Serial.println(gz, 4);
    
    delay(SAMPLE_INTERVAL_MS);
  }
}
```

**Data collection protocol:**
1. Open Arduino IDE Serial Monitor ( baud rate)
2. Copy header line, then perform each gesture for 10 seconds
3. Save output to CSV files:
   - `gesture_flick_up.csv`
   - `gesture_flick_down.csv`
   - `gesture_shake.csv`
   - `gesture_circle.csv`
   - `gesture_stationary.csv`

**Target:** 100 samples per class (5 seconds each = 20 samples per gesture session, 5 sessions)

**Deliverable:** `gesture_data.csv` with 5 classes, 100 samples each.

---

### Day 10 (Friday) — Train Gesture Classifier for Edge [1.5 hrs]

**Implementation:**

Create `week2/train_gesture.py`:
```python
"""
Week 2, Day 10: Train Tiny Gesture Classifier for Arduino
Target: < 20KB INT8 model, >90% accuracy
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

os.makedirs('week2/outputs', exist_ok=True)

# Load gesture data
gesture_files = {
    'flick_up': 'data/gesture_flick_up.csv',
    'flick_down': 'data/gesture_flick_down.csv',
    'shake': 'data/gesture_shake.csv',
    'circle': 'data/gesture_circle.csv',
    'stationary': 'data/gesture_stationary.csv'
}

# Window parameters
WINDOW_SIZE = 40  # 2 seconds @ 20Hz
STRIDE = 20       # 50% overlap

samples = []
labels = []

for label_idx, (gesture, filepath) in enumerate(gesture_files.items()):
    df = pd.read_csv(filepath, skiprows=1)  # Skip header
    data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    
    # Create windows
    for i in range(0, len(data) - WINDOW_SIZE, STRIDE):
        window = data[i:i + WINDOW_SIZE]  # (40, 6)
        # Transpose to (6, 40) for Conv1d
        window = window.T
        samples.append(window)
        labels.append(label_idx)

X = np.array(samples)  # (N, 6, 40)
y = np.array(labels)

print(f"Total samples: {len(X)}")
print(f"Class distribution: {np.bincount(y)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-1])
scaler.fit(X_train_flat)

X_train = scaler.transform(X_train_flat).reshape(X_train.shape)
X_test = scaler.transform(X_test_flat).reshape(X_test.shape)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Model
class GestureNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 40 -> 20
            
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))

model = GestureNet()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(X_test).argmax(dim=1)
            acc = (pred == y_test).float().mean().item()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

# Save
model.eval()
torch.save(model.state_dict(), 'models/gesture_model.pt')
print("Saved: models/gesture_model.pt")

# Convert to TFLite (same pipeline as ECG model)
# ... (reuse conversion script from Day 8)
```

**Deliverable:** `gesture_model.tflite` (< 20KB) + training script.

---

### Weekend (Sat + Sun) — Content Creation + Buffer [3 hrs + 2 hrs]

**Saturday:**
- [ ] ⭐ **LinkedIn Post #2 (Week 2 Build Log):**

```
Week 2: I trained a neural network, then I broke it — on purpose.

What I did:
→ Built a 1D-CNN for ECG arrhythmia detection on MIT-BIH (85% accuracy)
→ Applied Quantization-Aware Training → reduced model from 180KB to 18KB (10x smaller)
→ INT8 accuracy drop: only 2.1%. QAT is magic.

What surprised me:
Post-training quantization (PTQ) killed 8% accuracy. QAT recovered almost all of it.
Moral: if you're deploying to edge, train with quantization in mind from epoch 1.

Benchmark table 👇
[Attach: your benchmark table image]

Next week: Flashing this to an Arduino. Yes, the 18KB model. On a 256KB RAM chip.

#TinyML #EdgeAI #Quantization #PyTorch #MachineLearning
```

- [ ] Create benchmark visualization (matplotlib bar chart: FP32 vs INT8 size + accuracy)

**Sunday:**
- [ ] Draft Reddit post: "[P] From PyTorch to Arduino: How I compressed a 1D-CNN 10x for edge deployment"
- [ ] Plan Week 3 hardware wiring
- [ ] Read: Edge Impulse "Running your model on Arduino" deployment guide

**Week 2 OSS Target:**
- [ ] ⭐ Submit PR to `neurokit2/neurokit2` documentation: Add a Jupyter notebook example showing batch ECG processing with HRV export. Or, if that feels too big, add a small doc fix to `edgeimpulse/example-standalone-inferencing` README.

---

## 7. Week 3: Physical AI Deployment — Day-by-Day Implementation

### Theme: "My model now lives on a chip. I can measure its heartbeat."

---

### Day 11 (Monday) — TFLite Micro + Arduino Deployment [1.5 hrs]

**Learning:**
- TFLite Micro vs TFLite: Micro has no OS, no malloc, runs on bare metal
- Arduino_TensorFlowLite library

**Implementation:**

In Arduino IDE: Install `Arduino_TensorFlowLite` library (version 2.4.0+)

Convert model to C array:
```bash
# On your laptop (needs xxd tool)
xxd -i gesture_model.tflite > gesture_model.h
```

Write `week3/gesture_infer.ino`:
```cpp
/*
  Week 3, Day 11: TFLite Micro Gesture Inference
  Target: Arduino Nano 33 BLE Sense
  Model: GestureNet INT8, < 20KB
  Author: Abhishek Bhardwaj
*/

#include <TensorFlowLite.h>
#include "gesture_model.h"  // xxd-generated header

// TFLite Micro arena (adjust if inference fails)
constexpr int kTensorArenaSize = 8 * 1024;  // 8KB
alignas(16) byte tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(9600);
  while (!Serial);
  
  Serial.println("Initializing TFLite Micro...");
  
  const tflite::Model* model = tflite::GetModel(gesture_model_tflite);
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema mismatch!");
    return;
  }
  
  tflite::AllOpsResolver resolver;
  interpreter = new tflite::MicroInterpreter(
    model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter
  );
  
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: Tensor allocation failed!");
    return;
  }
  
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.print("Model loaded. Input dims: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
  Serial.println("Ready for inference!");
}

void loop() {
  // Dummy inference to measure latency
  // In real use: fill input->data.f with IMU window
  for (int i = 0; i < input->bytes / sizeof(float); i++) {
    input->data.f[i] = 0.0f;  // Replace with actual sensor data
  }
  
  unsigned long start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long end = micros();
  
  if (invoke_status == kTfLiteOk) {
    Serial.print("Inference: ");
    Serial.print(end - start);
    Serial.print(" us | Output: ");
    for (int i = 0; i < output->dims->data[1]; i++) {
      Serial.print(output->data.f[i], 4);
      Serial.print(" ");
    }
    Serial.println();
  } else {
    Serial.println("Invoke failed!");
  }
  
  delay(1000);
}
```

**Deliverable:** Working Arduino sketch that loads TFLite model and reports inference latency.

---

### Day 12 (Tuesday) — Live Gesture Inference on Arduino [1.5 hrs]

**Implementation:**

Integrate IMU reading + TFLite inference:
```cpp
// Add to setup():
if (!IMU.begin()) {
  Serial.println("IMU init failed!");
  while (1);
}

// In loop(), replace dummy data with real IMU:
float ax, ay, az, gx, gy, gz;
if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);
  
  // Fill input tensor (normalized)
  input->data.f[0] = ax;
  input->data.f[1] = ay;
  input->data.f[2] = az;
  input->data.f[3] = gx;
  input->data.f[4] = gy;
  input->data.f[5] = gz;
  
  // ... invoke ...
  
  // Map argmax to gesture label
  int predicted = 0;
  float max_val = output->data.f[0];
  for (int i = 1; i < 5; i++) {
    if (output->data.f[i] > max_val) {
      max_val = output->data.f[i];
      predicted = i;
    }
  }
  
  const char* labels[] = {"FLICK_UP", "FLICK_DOWN", "SHAKE", "CIRCLE", "STATIONARY"};
  Serial.print("Gesture: ");
  Serial.println(labels[predicted]);
}
```

**Test with physical gestures.** Document accuracy.

**Deliverable:** End-to-end working demo: move Arduino → see gesture label printed.

---

### Day 13 (Wednesday) — Benchmarking: Latency, Power, Memory [1.5 hrs]

**Implementation:**

Write `week3/benchmark.ino`:
```cpp
// Run 100 inferences, collect stats
unsigned long times[100];
for (int i = 0; i < 100; i++) {
  unsigned long t0 = micros();
  interpreter->Invoke();
  times[i] = micros() - t0;
}

// Sort and compute statistics
// (implement bubble sort or use Arduino sort library)
// Print: min, median, mean, max, p99
```

Create `week3/benchmark_results.md`:
```markdown
| Metric | Value |
|--------|-------|
| Model size (flash) | ?? KB |
| Arena size (RAM) | 8 KB |
| Inference latency (median) | ?? μs |
| Inference latency (p99) | ?? μs |
| Accuracy (5-class gesture) | ?? % |
| Power draw (approx) | ?? mA |
```

**Deliverable:** Benchmark table complete, committed to repo.

---

### Day 14 (Thursday) — ECG Pipeline on Laptop (Simulated Edge) [1.5 hrs]

Create `week3/ecg_edge_pipeline.py`:
```python
"""
Simulate edge deployment pipeline on laptop
1. Read MIT-BIH ECG segment
2. Apply bandpass filter
3. Run INT8 quantized model
4. Output: Normal vs Arrhythmia + confidence
5. Measure latency
"""
# ... (implementation similar to earlier but with latency benchmarking)
```

Build Streamlit dashboard:
```bash
pip install streamlit
streamlit run week3/ecg_dashboard.py
```

**Deliverable:** Working Streamlit dashboard + latency comparison.

---

### Day 15 (Friday) — Architecture Diagram + System Thinking [1.5 hrs]

Create `week3/system_architecture.md`:
```markdown
```mermaid
graph LR
    A[MIT-BIH ECG Signal] -->|Bandpass Filter| B[Clean Signal]
    B -->|5s Window| C[1D-CNN INT8]
    C -->|Arduino| D[Inference < 50ms]
    D -->|Serial| E[Gestures/Biosignals]
    E -->|MQTT/Bluetooth| F[Dashboard]
```
```

Write `week3/LESSONS_LEARNED.md`:
- What broke during quantization
- Why TFLite Micro arena sizing is critical
- How INT8 changes understanding of model precision

**Deliverable:** System architecture diagram + lessons learned doc.

---

### Weekend — Content Blitz [3 hrs + 2 hrs]

**LinkedIn Post #3:**
```
Week 3: I held a neural network in my hand. Literally.

Deployed a 18KB gesture classifier to an Arduino Nano 33 BLE Sense.

Results:
→ Inference latency: 8.3ms median (120 inferences/second)
→ RAM footprint: 8KB arena + model
→ 5-class gesture accuracy: 91% (tested live)

What broke:
- TFLite Micro arena too small → hard fault. Doubled it → worked.
- BatchNorm incompatible with INT8 TFLite. Replaced with GroupNorm.
- IMU clipped on fast flicks. Changed range to ±8g.

These are the exact problems edge AI engineers solve daily.

[Attach: photo of Arduino + serial monitor]
```

**Twitter thread** (5-7 tweets):
```
1/ I spent the week deploying a neural network to a $3 chip. Here's what nobody told me about TinyML.

2/ Problem 1: Your beautiful PyTorch model means nothing. TFLite Micro doesn't understand BatchNorm. Rewrite. Retrain. Re-quantize.

3/ Problem 2: Memory isn't managed. You allocate a 8KB "tensor arena" and pray. Too small = crash. Too big = wasted RAM.

4/ Problem 3: Sensors lie. The IMU clips, drifts, and picks up table vibrations. Signal preprocessing isn't optional.

5/ The reward: 8ms inference on a chip consuming 10mA. That's the magic of edge AI.

6/ Code + benchmarks: github.com/YOUR_USERNAME/edge-ml-pivot

7/ Next week: ECG biosensing pipeline + my first OSS contribution. 🧵
```

---

## 8. Week 4: Integration + Public Launch — Day-by-Day Implementation

### Theme: "I shipped something. In public."

---

### Day 16 (Monday) — ECG + Gesture Integration Demo [1.5 hrs]

Write `week4/unified_demo.py`:
```python
"""
Unified Biosensing + Gesture Edge Demo
Simulates a wearable device monitoring ECG + detecting gestures
"""
# ... (integration script)
```

**Deliverable:** Unified demo script working end-to-end.

---

### Day 17 (Tuesday) — Repository Polish + README [1.5 hrs]

Make repo **public**.

Write `README.md`:
```markdown
# Edge ML Pivot: TinyML Biosensing + Gesture System

> Month 1 of my journey from cloud-scale LLMs to edge AI.

## What This Is
A 1-month implementation sprint training a 1D-CNN for ECG arrhythmia detection,
quantizing it to INT8, and deploying gesture recognition on an Arduino Nano 33 BLE Sense.

## Architecture
[Mermaid diagram]

## Hardware
- Arduino Nano 33 BLE Sense (~₹3,200)
- MPU-6050 IMU (~₹220)
- Breadboard + jumpers (~₹270)
- Total: Under ₹4,000

## Results
| Model | Task | Size | Accuracy | Latency |
|-------|------|------|----------|---------|
| TinyECGNet | Arrhythmia | 18KB (INT8) | 83% | N/A (laptop) |
| GestureNet | 5 gestures | 12KB (INT8) | 91% | 8.3ms (Arduino) |

## Repo Structure
- `week1/` — Signal processing, ECG exploration
- `week2/` — Model training, quantization, gesture data
- `week3/` — Arduino deployment, benchmarking
- `week4/` — Unified demo, dashboard

## Running It
...setup instructions...

## What's Next
- Month 2: STM32 Nucleo deployment, CMSIS-NN, Apache TVM
- Month 3: On-device LLM with llama.cpp on Raspberry Pi 5

## License
MIT
```

**Deliverable:** Polished public GitHub repo with >100 commits.

---

### Day 18 (Wednesday) — OSS Contribution Day [1.5 hrs]

**Option A (Documentation/Example — Easiest):**
- Fork `edgeimpulse/example-standalone-inferencing`
- Add your gesture model as an example
- Submit PR

**Option B (Bug Fix — Medium):**
- Find "good first issue" in ONNX Runtime (mobile/embedded tags)
- Reproduce, fix, test, submit PR

**Option C (Your Own Library — Fallback):**
- Publish `tinyecg-utils` to PyPI

**Deliverable:** Submitted PR or published package.

---

### Day 19 (Thursday) — Online Presence Blitz [1.5 hrs]

**LinkedIn Post #4 (Project Launch):**
```
I just open-sourced my Month 1 edge AI project.

What I built:
→ A 1D-CNN that detects cardiac arrhythmia from ECG signals
→ Quantized it 10x smaller using PyTorch QAT (18KB)
→ Deployed a gesture classifier to Arduino Nano 33 BLE Sense (8ms inference)
→ Total hardware cost: Under ₹4,000

Why this matters:
Indian semiconductor startups are building AI-enabled biosensing chips.
They need ML engineers who understand both the model AND the 256KB constraint.

I'm documenting every month of this transition in public.

Repo link in comments 👇

#TinyML #EdgeAI #IndiaSemiconductor #MachineLearning #OpenSource
```

**Reddit:**
- Post to r/MachineLearning: "[P] TinyML Biosensing: ECG arrhythmia detection + gesture recognition on Arduino"

**Twitter:**
- Thread announcing repo launch (3-4 tweets)
- Pin repo link tweet

**Deliverable:** Content live on 3 platforms. Repo has visibility.

---

### Day 20 (Friday) — Networking + Outreach [1.5 hrs]

**LinkedIn:**
- Send 5 connection requests to:
  - Engineers at Sophrosyne Technologies
  - Edge AI engineers at Qualcomm India
  - Founders of Netrasemi, Mindgrove
- Personalized note: "Hi [Name], I'm documenting my transition from cloud ML to edge AI via a public build log. I'd love to follow your work at [Company]."

**Reddit:**
- Comment thoughtfully on 3 posts

**Twitter:**
- Reply meaningfully to 2 tweets from @tinyMLsummit or @EdgeImpulse

**Deliverable:** 5+ new connections, 3+ Reddit comments, 2+ Twitter engagements.

---

### Weekend — Reflection + Month 2 Planning [3 hrs + 2 hrs]

**Saturday:**
- Write `MONTH1_RETROSPECTIVE.md`
- Update resume with new "Edge AI & Embedded ML Projects" section

**Sunday:**
- Read ahead: 12-month Physical_AI roadmap
- Plan Month 2 hardware if continuing: STM32 Nucleo-F401RE (~₹2,200)

---

## 9. Topic-by-Topic Course Curriculum (2026-2027 Industry-Relevant)

### Tier 1: Free Courses (Start Here)

| # | Course | Provider | URL | Duration | Why Take It |
|---|--------|----------|-----|----------|-------------|
| 1 | **Harvard CS249r: Tiny Machine Learning** | edX (audit free) | [harvardonline.harvard.edu/course/professional-certificate-tiny-machine-learning-tinyml](https://www.harvardonline.harvard.edu/course/professional-certificate-tiny-machine-learning-tinyml) | 4 months | THE reference course. Vijay Janapa Reddi, Pete Warden, Laurence Moroney. Arduino-based. |
| 2 | **MIT 6.S965: TinyML and Efficient Deep Learning** | YouTube (free) | Search "MIT 6.S965 TinyML" | Self-paced | Song Han's group. Cutting-edge research-level material. |
| 3 | **Edge Impulse Academy** | Edge Impulse (free) | [edgeimpulse.com/academy](https://edgeimpulse.com/academy) | Self-paced | Hands-on TinyML with real deployments. Now owned by Qualcomm (2025). |
| 4 | **FastAI Practical Deep Learning** | fast.ai (free) | [course.fast.ai](https://course.fast.ai) | 7 weeks | Code-first approach. Excellent for practical PyTorch skills. |
| 5 | **MIT OpenCourseWare: 6.003 Signals & Systems** | MIT (free) | [ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/) | Self-paced | Foundation for biosignal processing. |
| 6 | **Embedded Systems Shape the World** | UTAustinX, edX (free) | [edx.org/course/embedded-systems-shape-the-world](https://www.edx.org/course/embedded-systems-shape-the-world) | 8 weeks | Microcontroller basics, GPIO, interrupts. |
| 7 | **PyTorch Quantization Tutorial** | PyTorch (free) | [pytorch.org/tutorials/advanced/static_quantization_tutorial.html](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html) | 2 hours | Essential for your QAT work. |
| 8 | **TensorFlow Lite Micro Documentation** | Google (free) | [tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers) | Self-paced | Deployment to microcontrollers. |
| 9 | **ONNX Runtime Documentation** | Microsoft (free) | [onnxruntime.ai/docs](https://onnxruntime.ai/docs/) | Self-paced | Cross-platform inference optimization. |
| 10 | **Apache TVM Documentation** | Apache (free) | [tvm.apache.org/docs](https://tvm.apache.org/docs/) | Self-paced | ML compiler stack. Start with tutorials. |

### Tier 2: Low-Cost Certifications (Months 4-12)

| # | Certification | Provider | Cost | Value |
|---|-------------|----------|------|-------|
| 1 | **Professional Certificate in TinyML** | Harvard/edX | $897 (audit free) | Highest employer recognition. Academic prestige + Google TF team instructors. |
| 2 | **Applied TinyML for Scale** | Harvard/edX | $807 | Adds MLOps for scaling TinyML. Includes federated learning, NAS. |
| 3 | **Edge AI for Microcontrollers Specialization** | Edge Impulse/Coursera | Coursera subscription | Updated March 2026. Hands-on motion, keyword spotting, object detection. |
| 4 | **AI Upskilling Certificate** | Qualcomm Academy | **FREE** | 3-4 hours. Official Qualcomm badge. Uses Edge Impulse + Arduino. |
| 5 | **RISC-V Foundational** | RISC-V International | **FREE** | Essential for Indian RISC-V startups (Mindgrove, InCore). |
| 6 | **AWS Machine Learning Specialty** | AWS | ~₹15,000 | Signals cloud ML competency. Good for hybrid roles. |

### Tier 3: Advanced (Months 13-18, Optional)

| # | Course/Resource | Provider | Cost | When |
|---|-----------------|----------|------|------|
| 1 | **MLIR for Beginners** | LLVM Project (free) | [mlir.llvm.org/docs/Tutorials](https://mlir.llvm.org/docs/Tutorials/) | Month 15+ |
| 2 | **CUDA Programming** | NVIDIA (free) | [developer.nvidia.com/cuda-training](https://developer.nvidia.com/cuda-training) | Month 16+ |
| 3 | **Computer Architecture** | MIT OpenCourseWare (free) | [ocw.mit.edu/courses/6-004-computation-structures](https://ocw.mit.edu/courses/6-004-computation-structures/) | Month 14+ |
| 4 | **Compilers: Principles, Techniques, and Tools** | Book (Dragon Book) | ~₹2,500 | Month 16+ |

---

## 10. GitHub Repository Strategy — Exact Repos to Star, Fork, and Contribute To

### Phase 1: Star and Study (Month 1)

| Repository | Stars | Language | Why Star | What to Learn |
|------------|-------|----------|----------|---------------|
| `tensorflow/tflite-micro` | 1.5K+ | C++ | Google's official TinyML runtime | How TFLite runs on bare metal |
| `microsoft/onnxruntime` | 15K+ | C++ | Cross-platform inference engine | ARM NEON optimizations, execution providers |
| `apache/tvm` | 12K+ | Python/C++ | ML compiler for heterogeneous hardware | Relay IR, AutoTVM, MicroTVM |
| `ARM-software/CMSIS-NN` | 1K+ | C | ARM's optimized NN kernels for Cortex-M | How matrix multiply is optimized for ARM |
| `ggerganov/llama.cpp` | 70K+ | C++ | LLM inference on CPU/ARM via GGUF | Quantization formats, ARM NEON kernels |
| `mlc-ai/mlc-llm` | 5K+ | Python/C++ | TVM-based LLM inference | How to compile LLMs for edge targets |
| `neurokit2/neurokit2` | 2K+ | Python | Biosignal processing | ECG/PPG algorithms you can contribute to |
| `mit-han-lab/ncnn` | 20K+ | C++ | Mobile inference framework | Tencent's optimized ARM inference |
| `tinygrad/tinygrad` | 25K+ | Python | Minimalist deep learning framework | Excellent for understanding DL from scratch |
| `openvinotoolkit/openvino` | 8K+ | C++ | Intel's inference engine, ARM-optimized | Production-grade edge deployment |
| `ollama/ollama` | 169K+ | Go | Local LLM runtime | Dominant local LLM tool; edge/consumer focus |
| `edgeimpulse/example-standalone-inferencing` | 200+ | C++ | Edge Impulse C++ examples | How to run Edge Impulse models standalone |
| `open-edge-platform/anomalib` | 4K+ | Python | Anomaly detection + edge inference | Niche, high-value industrial AI skill |

### Phase 2: Fork and Experiment (Months 2-3)

| Repository | Action | Difficulty |
|------------|--------|------------|
| `neurokit2/neurokit2` | Fork, add ECG batch processing example | Easy |
| `edgeimpulse/example-standalone-inferencing` | Fork, add your gesture model as example | Easy |
| `tensorflow/tflite-micro` | Fork, read Arduino examples | Medium |
| `ggerganov/llama.cpp` | Fork, build for ARM, benchmark | Medium |

### Phase 3: Contribute (Months 4-18)

| Target Repo | Contribution Type | When | Difficulty |
|-------------|-------------------|------|------------|
| `neurokit2/neurokit2` | Documentation/tutorial PR | Month 2 | Easy |
| `edgeimpulse/example-standalone-inferencing` | Add sensor driver example | Month 3 | Easy |
| `apache/tvm` | Fix documentation, add tutorial | Month 6 | Medium |
| `microsoft/onnxruntime` | Mobile/embedded optimization docs | Month 7 | Medium |
| `ggerganov/llama.cpp` | ARM NEON benchmark or example | Month 8 | Hard |
| `tensorflow/tflite-micro` | Add new op or improve docs | Month 9 | Medium |
| `tinygrad/tinygrad` | Small backend improvement | Month 12 | Hard |
| `openvinotoolkit/openvino` | Notebook example for edge deployment | Month 10 | Medium |

**Minimum viable OSS targets by month:**
- Month 3: 1 documentation PR merged
- Month 6: 2 PRs merged (1 doc, 1 small code)
- Month 9: 4 PRs merged
- Month 12: 6 PRs merged
- Month 18: 8-12 PRs merged

---

## 11. Online Presence Content Calendar — Exact Templates

### Month 1 Content Calendar

| Week | Platform | Content | Format | Time to Create |
|------|----------|---------|--------|----------------|
| 1 | LinkedIn | Build log: ECG signal processing | Text + image | 20 min |
| 1 | Twitter | "Starting edge AI journey" thread | 2-3 tweets | 15 min |
| 2 | LinkedIn | Quantization benchmark results | Text + table image | 25 min |
| 2 | Twitter | QAT vs PTQ learnings | 3-4 tweets | 20 min |
| 3 | LinkedIn | Arduino deployment results | Text + hardware photo | 25 min |
| 3 | Twitter | TinyML failure modes thread | 5-7 tweets | 30 min |
| 4 | LinkedIn | Project launch announcement | Text + repo link in comment | 30 min |
| 4 | Reddit | [P] project write-up | 800-1200 words | 45 min |
| 4 | Twitter | Repo launch thread | 3-4 tweets | 20 min |

### Content Templates (Copy-Paste Ready)

**LinkedIn Build Log Template:**
```
Week [X] of my pivot from cloud-scale LLMs to edge AI.

What I did:
→ [Specific technical achievement]
→ [Specific technical achievement]
→ [Specific technical achievement]

What surprised me:
[Unexpected finding or failure mode]

Next week: [Preview of next task]

[Attach: relevant image/chart]

#[Hashtag1] #[Hashtag2] #[Hashtag3]
```

**Twitter Thread Template:**
```
1/ I spent the week [doing X]. Here's what I learned about [topic].

2/ [Problem 1]: [Description]. [Solution or lesson].

3/ [Problem 2]: [Description]. [Solution or lesson].

4/ [Surprising insight or result].

5/ The reward: [Positive outcome].

6/ Code + benchmarks: [GitHub link]

7/ Next week: [Preview]. Follow for more [topic]. 🧵
```

**Reddit Post Template:**
```
[P] [Project Name]: [One-line description]

**What this is:**
[2-3 sentence overview]

**Key results:**
| Metric | Value |
|--------|-------|
| [Metric 1] | [Value] |
| [Metric 2] | [Value] |

**Tech stack:**
- [Tool 1]
- [Tool 2]
- [Tool 3]

**What I learned:**
[3-5 bullet points of insights]

**GitHub:** [link]

**Questions welcome!**
```

---

## 12. LinkedIn Algorithm-Optimized Posting Strategy (2026 Data)

### What Works in 2026 (360Brew Algorithm)

**Format priority:**
1. Document Carousels (PDF) — HIGHEST reach, 2.3-5× median impressions
2. Long-form Text (1500+ chars) — +49% engagement
3. Short Vertical Video (screen recordings) — +69% performance
4. Newsletters/Articles — +48% reach growth

**Engagement hierarchy (algorithmic weight):**
1. Saves/Bookmarks — 5× more powerful than likes
2. Meaningful Comments (3+ sentences) — 2-2.5× more weight
3. DM Shares — high trust signal
4. Dwell Time (31-60 seconds) — optimal
5. Likes — lowest weight

**Critical rules:**
- First 60-90 minutes determines ~70% of total reach
- External links in post body = ~40% less reach → put link in FIRST COMMENT
- >5 hashtags = 68% reach reduction → use 0-3 hashtags
- Posting >2× per day = reach penalty → max 2/day, ideally 2-5/week
- Topic consistency for 90+ days → pick 2-3 pillars

**Best posting times for India targeting US recruiters:**
- Tue-Thu, 7:30-9:00 AM IST (overlaps with US evening)
- Secondary: 12-2 PM IST

**Your 2-3 content pillars:**
1. Edge AI deployment benchmarks (quantization, latency, power)
2. Indian semiconductor ecosystem commentary
3. Open-source contribution logs

---

## 13. Reddit Strategy — Exact Subreddits and Post Templates

### Target Subreddits

| Subreddit | Subscribers | Best Content | Post Frequency |
|-----------|-------------|--------------|----------------|
| r/MachineLearning | 3.04M | [P] project write-ups, [R] reproductions | 1/month |
| r/LocalLLaMA | 695K | Edge LLM experiments, quantization | 1/month |
| r/embedded | Active | STM32/TinyML hardware projects | 1/month |
| r/IndianStartups | Active | India semiconductor ecosystem | 1/month |
| r/cscareerquestions | Active | Career transition advice | Comment only |

### What Wins on Reddit (2026 Data)

| Format | Length | Performance |
|--------|--------|-------------|
| Use-case writeups | 600-1200 words | Highest median upvotes (~1,180) |
| Hot-take threads | 200-500 words | Higher comments-per-upvote (~0.55) |
| Failure post-mortems | Varies | Highest comments-per-upvote (~0.78) |

**Rule:** Operational > Novelty. Posts about inference efficiency, quantization quality, and latency budgets outperform "I trained a new model" posts.

---

## 14. Twitter/X Strategy — Exact Thread Templates

### Thread Type 1: Project Update
```
1/ Week [X] of building [project].

2/ Started with [task]. Expected [outcome]. Got [surprising result].

3/ The problem: [technical challenge].

4/ The solution: [approach]. Took [time/iterations].

5/ Result: [metric]. [Comparison to baseline].

6/ What I learned: [insight].

7/ Code: [GitHub link]. Questions welcome.
```

### Thread Type 2: Failure Analysis
```
1/ I tried [approach]. It failed spectacularly. Here's why.

2/ [What I expected]

3/ [What actually happened — error message, unexpected behavior]

4/ [Root cause analysis]

5/ [The fix]

6/ [Lesson learned that applies beyond this project]

7/ Have you faced this? How did you solve it?
```

### Thread Type 3: Industry Commentary
```
1/ [Industry trend or news]. Here's what it means for [role].

2/ [Context for non-experts]

3/ [Technical implication 1]

4/ [Technical implication 2]

5/ [What engineers should do about it]

6/ [Personal opinion / prediction]
```

---

## 15. OSS Contribution Roadmap — Week-by-Week Targets

| Month | Week | Target Repo | Action | Expected Outcome |
|-------|------|-------------|--------|------------------|
| 1 | 4 | `neurokit2/neurokit2` | File documentation issue | Engagement with maintainers |
| 2 | 8 | `neurokit2/neurokit2` | Submit doc/tutorial PR | 1st PR merged |
| 3 | 12 | `edgeimpulse/example-standalone-inferencing` | Add example | 2nd PR merged |
| 4 | 16 | `tensorflow/tflite-micro` | Fix doc inconsistency | 3rd PR merged |
| 5 | 20 | `microsoft/onnxruntime` | File issue with reproduction | Community recognition |
| 6 | 24 | `apache/tvm` | Add tutorial PR | 4th PR merged |
| 7 | 28 | `ggerganov/llama.cpp` | Add ARM benchmark | 5th PR merged |
| 8 | 32 | `openvinotoolkit/openvino` | Notebook example | 6th PR merged |
| 9 | 36 | `tensorflow/tflite-micro` | Code contribution | 7th PR merged |
| 10 | 40 | `apache/tvm` | Substantive PR | 8th PR merged |
| 11 | 44 | `tinygrad/tinygrad` | Backend fix | 9th PR merged |
| 12 | 48 | Any target repo | Major contribution | 10th+ PR merged |

---

## 16. Exact Code Templates — Copy-Paste Ready

### Template 1: Kaggle Notebook Header
```python
# %% [markdown]
# ## Edge ML Month 1: [Week X, Task Y]
# **Author:** Abhishek Bhardwaj
# **Hardware target:** Arduino Nano 33 BLE Sense (256KB RAM, 1MB Flash)
# **Goal:** [Specific goal]

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
```

### Template 2: Arduino Sketch Header
```cpp
/*
  Edge ML Month 1 — Week X
  Target: Arduino Nano 33 BLE Sense
  Task: [Description]
  Author: Abhishek Bhardwaj
  Model size: X KB | Arena: Y KB
*/

#include <TensorFlowLite.h>
// ... includes ...

#define DEBUG_PRINT(x) Serial.println(x)
```

### Template 3: Benchmark Logging (Python)
```python
import json
from datetime import datetime

benchmark = {
    "date": datetime.now().isoformat(),
    "model": "GestureNet",
    "target": "Arduino Nano 33 BLE Sense",
    "quantization": "INT8",
    "flash_kb": 12,
    "ram_kb": 8,
    "latency_ms": {"median": 8.3, "p99": 12.1},
    "accuracy": 0.91,
    "notes": "IMU set to ±8g range to prevent clipping"
}

with open("benchmarks.json", "a") as f:
    f.write(json.dumps(benchmark) + "\n")
```

### Template 4: LinkedIn Post (Text-Only)
```
Week [X] of my pivot from cloud-scale LLMs to edge AI.

What I did:
→ [Achievement 1]
→ [Achievement 2]
→ [Achievement 3]

What surprised me:
[Unexpected finding]

Next week: [Preview]

#[Hashtag1] #[Hashtag2]
```

---

## 17. Troubleshooting Guide — Common Failures and Fixes

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Arduino won't flash | Wrong board selected | Select "Arduino Nano 33 BLE" (not "Arduino Nano") |
| TFLite model too big (>50KB) | Too many filters/parameters | Reduce Conv1d channels: 16→8, 32→16, 64→32 |
| QAT accuracy drops >5% | Insufficient QAT epochs | Train 10-20 epochs with QAT, not 5 |
| Kaggle GPU disconnects | 12-hour limit reached | Save checkpoints every epoch to `/kaggle/working/` |
| IMU data is noisy | Clipping or insufficient null data | Increase window size, add low-pass filter, collect more stationary data |
| `interpreter->Invoke()` crashes | Arena too small | Increase `kTensorArenaSize` (try 16KB) |
| BatchNorm incompatible with INT8 TFLite | Known TFLite Micro limitation | Replace BatchNorm with GroupNorm or LayerNorm |
| Can't think of LinkedIn content | Overthinking | Post your failures. "Tried X, got Y error, learned Z." |
| No OSS maintainer responds | PR too large or unclear | Start with doc fixes. Wait 7 days, bump politely. |
| MIT-BIH download fails | Need PhysioNet account | Create free account at physionet.org, request access |
| Arduino IDE won't install library | Network or version issue | Use Library Manager, search exact name, install specific version |

---

## 18. Budget Tracker — INR Spend Week by Week

| Week | Item | Cost | Cumulative | Notes |
|------|------|------|-----------|-------|
| 1 | Arduino Nano 33 BLE Sense | ₹3,200 | ₹3,200 | Essential |
| 1 | MPU-6050 IMU | ₹220 | ₹3,420 | Essential |
| 1 | Breadboard + jumpers | ₹400 | ₹3,820 | Essential |
| 1 | USB-micro cable | ₹100 | ₹3,920 | If needed |
| 2 | MAX30102 (optional) | ₹280 | ₹4,200 | Only if budget allows |
| 3-4 | No spend | ₹0 | ₹4,200 | Software only |

**Total Month 1: ~₹4,200**

**Month 2 onwards (if continuing):**
| Month | Item | Cost | Cumulative |
|-------|------|------|-----------|
| 2 | STM32 Nucleo-F401RE | ₹2,200 | ₹6,400 |
| 2 | AD8232 ECG Module | ₹450 | ₹6,850 |
| 3 | Raspberry Pi 5 (4GB) | ₹7,500 | ₹14,350 |
| 4 | Google Coral USB TPU | ₹5,000 | ₹19,350 |
| 5-6 | Sensors, misc | ₹1,000 | ₹20,350 |

**Total 6-month hardware: ~₹20,350**

---

## 19. Post-Month 1 Expansion Path — Months 2-6 Overview

### Month 2: STM32 + CMSIS-NN
- Deploy PPG anomaly detector to STM32 Nucleo-F401RE
- Learn CMSIS-NN optimized kernels
- Measure speedup vs naive C implementation
- Target: inference < 50ms at 84MHz

### Month 3: Raspberry Pi 5 + ONNX Runtime
- Set up Pi 5 with Raspberry Pi OS
- Deploy multi-modal vital signs monitor (ECG + PPG + temp)
- ONNX Runtime INT8 with ARM NEON acceleration
- Streamlit dashboard for real-time monitoring

### Month 4: Edge Impulse Mastery
- Fall detection wearable on Edge Impulse
- EON Tuner for Pareto-optimal models
- Deploy to both Arduino and Pi from same pipeline
- Write blog post: "Building a Fall Detection Wearable with TinyML"

### Month 5: Model Compression Deep Dive
- Compression benchmark study: PTQ vs QAT vs Pruning vs Distillation
- Mixed-precision quantization (different layers, different bit-widths)
- Publish results as technical article

### Month 6: Apache TVM Introduction
- Compile PPG model with TVM for Pi 5 (ARM Cortex-A76)
- Compare latency: TFLite vs ONNX Runtime vs TVM
- Document speedup and discuss why TVM wins (or doesn't)
- GitHub: `biosignal-tvm-benchmark`

---

## 20. Industry Certifications That Matter in 2026

| Certification | Provider | Cost | Duration | Employer Recognition | Best For |
|--------------|----------|------|----------|---------------------|----------|
| **Professional Certificate in TinyML** | Harvard/edX | $897 | 4 months | **Highest** | Academic prestige + Google TF team. 3-course series. |
| **Applied TinyML for Scale** | Harvard/edX | $807 | 5 months | **High** | Adds MLOps, federated learning, NAS. Starts May 2026. |
| **Edge AI for Microcontrollers** | Edge Impulse/Coursera | Subscription | 8 weeks | **Medium-High** | Hands-on motion, keyword spotting, object detection. |
| **AI Upskilling Certificate** | Qualcomm Academy | **FREE** | 3-4 hours | **Medium-High** | Official Qualcomm badge. Uses Edge Impulse + Arduino. |
| **RISC-V Foundational** | RISC-V International | **FREE** | Self-paced | **Medium** | Indian RISC-V startups (Mindgrove, InCore). |
| **TensorFlow Lite for ML on Edge** | DeepLearning.AI | ~₹2,000/mo | 1 month | **Medium** | TFLite-specific, good for beginners. |
| **Embedded Systems** | UColorado/Coursera | ~₹2,000/mo | 2 months | **Low-Medium** | Microcontroller fundamentals. |

**Recommended sequence:**
1. Month 1-3: Harvard CS249r (audit free)
2. Month 4: Qualcomm AI Upskilling (free, quick win)
3. Month 6-9: Harvard Professional Certificate (paid, if budget allows)
4. Month 10: Edge Impulse Coursera Specialization
5. Month 12: RISC-V Foundational

---

## 21. Remote Job Application Templates

### Template 1: LinkedIn Connection Request
```
Hi [Name],

I'm a Senior Data Scientist documenting my transition to edge AI through a public build log. I've been following [Company]'s work on [specific product/technology] and would love to stay connected.

Currently building: TinyML biosensing systems with PyTorch + Arduino.

Best,
Abhishek
```

### Template 2: Cold Email to Startup Founder
```
Subject: ML Engineer interested in [Company]'s [product] — portfolio attached

Hi [Name],

I came across [Company] through [source] and was impressed by [specific detail]. 

I'm a Senior Data Scientist with 5+ years in PyTorch, transformers, and production MLOps. Over the past [X] months, I've been building at the intersection of ML and edge hardware:

- [Project 1]: [One-line description + result]
- [Project 2]: [One-line description + result]
- [OSS contribution]: [Repo + PR link]

I'm particularly interested in [Company]'s approach to [specific technical challenge]. Would you be open to a brief conversation about how my background in [skill] could contribute?

Portfolio: [GitHub link]
LinkedIn: [LinkedIn link]

Best regards,
Abhishek Bhardwaj
```

### Template 3: Application via Turing/Toptal
```
[Standard profile]

Highlight:
- Senior Data Scientist → Edge AI pivot
- Public portfolio: [GitHub link]
- OSS contributions: [Repo links]
- Technical writing: [Blog link]
- Specific skills: PyTorch, TFLite Micro, ONNX Runtime, quantization, signal processing

Rate expectation: [Your rate]
Availability: [Hours/week]
```

---

## 22. Daily Micro-Habit System

**5 minutes. Every morning. No exceptions.**

1. Open your project repo
2. Read one GitHub issue from a target OSS repo (TFLite Micro, Edge Impulse, ONNX Runtime)
3. Jot one insight in `daily_log.md` (even if it's "I don't understand this yet")

**Weekly micro-habits:**
- Monday: Review last week's progress, plan this week
- Wednesday: Engage on LinkedIn (comment on 2 posts)
- Friday: Commit code (even if small)
- Sunday: Write one paragraph of technical reflection

---

## 23. Success Criteria and Checklist

### Month 1 Checklist

| # | Task | Status | Week |
|---|------|--------|------|
| 1 | Hardware ordered and received | ☐ | 1 |
| 2 | Development environment set up | ☐ | 1 |
| 3 | First ECG signal processed with filters | ☐ | 1 |
| 4 | HRV features extracted (SDNN, RMSSD) | ☐ | 1 |
| 5 | LinkedIn presence launched | ☐ | 1 |
| 6 | Twitter account active | ☐ | 1 |
| 7 | Reddit accounts joined | ☐ | 1 |
| 8 | 1D-CNN trained on MIT-BIH (>80% accuracy) | ☐ | 2 |
| 9 | Quantization benchmark (FP32 vs PTQ vs QAT) | ☐ | 2 |
| 10 | Model converted to TFLite (<50KB) | ☐ | 2 |
| 11 | IMU gesture data collected (5 classes) | ☐ | 2 |
| 12 | Gesture classifier trained (>90% accuracy) | ☐ | 2 |
| 13 | TFLite model deployed to Arduino | ☐ | 3 |
| 14 | Inference latency measured | ☐ | 3 |
| 15 | Benchmark table published | ☐ | 3 |
| 16 | GitHub repo made public | ☐ | 4 |
| 17 | README with architecture diagram | ☐ | 4 |
| 18 | LinkedIn project launch post | ☐ | 4 |
| 19 | Reddit effortpost published | ☐ | 4 |
| 20 | Twitter launch thread | ☐ | 4 |
| 21 | 1 OSS PR submitted | ☐ | 4 |
| 22 | 5 LinkedIn connections sent | ☐ | 4 |
| 23 | Resume updated with edge AI section | ☐ | 4 |

### Post-Month 1 Decision Matrix

| If you enjoyed... | Then pursue... | Next Steps |
|--------------------|---------------|------------|
| Training models, quantization, compression | **Deep ML / ML Efficiency Engineer** | LoRA fine-tuning, LLM quantization, Kaggle competitions |
| Arduino deployment, sensors, soldering | **Edge AI / Embedded ML Engineer** | STM32, CMSIS-NN, Raspberry Pi, Hailo NPU |
| Both equally | **Hardware-Aware ML Engineer** | Both tracks in parallel — the unicorn path |
| Neither — found it tedious | **Stay in Cloud ML** | You validated cheaply. No sunk cost. |

---

*Plan Version 2.0 | June 2026 | Expanded with 2026-2027 industry data, exact repositories, topic-by-topic courses, and copy-paste templates.*
*Designed for: Working Indian professional, INR-constrained, camera-shy, targeting remote roles via OSS + reach.*
