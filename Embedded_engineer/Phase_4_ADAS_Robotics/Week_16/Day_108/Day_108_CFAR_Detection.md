# Day 108: CFAR (Constant False Alarm Rate)
## Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing

---

> **📝 Day 108 Focus:**
> In a Range-Doppler map, "Noise" is not flat. It's high near the car (engine vibration) and low far away. A fixed threshold (e.g., `> 50dB`) will either miss distant cars or detect 1000 ghost cars nearby. **CFAR** adapts the threshold locally to maintain a constant probability of false alarm.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** why fixed thresholding fails in Radar.
2.  **Implement** CA-CFAR (Cell Averaging).
3.  **Implement** OS-CFAR (Ordered Statistic) for multi-target situations.
4.  **Apply** CFAR to a 1D Range Profile and 2D Range-Doppler Map.
5.  **Tune** Guard Cells and Training Cells.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 107:** Range-Doppler Map.
-   **Statistics:** Mean, Median.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The CFAR Concept

Instead of a global threshold, we calculate a **Local Threshold** for each cell (CUT - Cell Under Test).
-   **Training Cells:** Neighbors used to estimate the noise floor.
-   **Guard Cells:** Neighbors immediately next to CUT (ignored to avoid including the target itself in the noise estimate).
-   **Offset:** A safety margin added to the noise estimate.

### 🔹 Part 2: CA-CFAR (Cell Averaging)

The simplest method.
1.  Take average of Training Cells (Left + Right).
2.  $Threshold = \alpha \times Mean(Training)$.
3.  If $CUT > Threshold$, Detect!
-   **Pros:** Optimal for uniform noise (Gaussian).
-   **Cons:** **Masking Effect**. If two targets are close, one enters the training cells of the other, raising the threshold and hiding the second target.

### 🔹 Part 3: OS-CFAR (Ordered Statistic)

Robust against multiple targets.
1.  Sort the Training Cells.
2.  Pick the $k$-th value (e.g., the Median or 75th percentile).
3.  $Threshold = \alpha \times Value_k$.
-   **Pros:** Ignores outliers (other targets) in the training window.
-   **Cons:** Computationally expensive (Sorting).

---

## 💻 Implementation: 1D CA-CFAR

**Scenario:**
-   Signal with variable noise floor.
-   Two targets.
-   Compare Fixed Threshold vs CA-CFAR.

### 🛠️ Setup
Create `week16_day108` and `cfar_demo.py`.

```bash
mkdir -p ~/ros2_ws/src/week16_day108
cd ~/ros2_ws/src/week16_day108
touch cfar_demo.py
```

### 👨‍💻 Code: CA-CFAR Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_signal():
    # 100 bins
    x = np.linspace(0, 100, 200)
    
    # Variable Noise Floor (Ramping up)
    noise = np.random.exponential(1.0, 200) + (x / 20.0)
    
    # Signal
    sig = noise.copy()
    
    # Target 1 at bin 50 (Strong)
    sig[50] += 15.0
    
    # Target 2 at bin 150 (Weak, but in high noise area)
    sig[150] += 15.0 
    
    return sig

def ca_cfar(signal, num_train, num_guard, offset):
    thresholds = np.zeros_like(signal)
    detections = np.zeros_like(signal)
    
    N = len(signal)
    
    # Sliding Window
    for i in range(num_train + num_guard, N - (num_train + num_guard)):
        # Extract Training Cells
        # Left Side
        left = signal[i - num_train - num_guard : i - num_guard]
        # Right Side
        right = signal[i + num_guard + 1 : i + num_guard + num_train + 1]
        
        # Estimate Noise (Mean)
        noise_level = np.mean(np.concatenate((left, right)))
        
        # Calculate Threshold
        # Offset is usually in dB, so we multiply if linear, add if log
        # Here signal is linear amplitude
        thresh = noise_level * offset
        thresholds[i] = thresh
        
        # Detection
        if signal[i] > thresh:
            detections[i] = signal[i]
            
    return thresholds, detections

def main():
    sig = generate_signal()
    
    # Fixed Threshold
    fixed_thresh = 10.0
    fixed_dets = sig > fixed_thresh
    
    # CA-CFAR
    # Train: 10 cells each side
    # Guard: 2 cells each side
    # Offset: Factor of 3 (Signal must be 3x noise)
    cfar_thresh, cfar_dets = ca_cfar(sig, 10, 2, 3.0)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(sig, label='Signal (Noisy)', color='gray', alpha=0.5)
    plt.plot(cfar_thresh, label='CFAR Threshold', color='blue', linewidth=2)
    plt.axhline(fixed_thresh, label='Fixed Threshold', color='red', linestyle='--')
    
    # Mark Detections
    det_indices = np.where(cfar_dets > 0)[0]
    plt.scatter(det_indices, sig[det_indices], color='green', marker='x', s=100, label='CFAR Detections')
    
    plt.title("CA-CFAR vs Fixed Threshold")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Masking Problem

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Fixed Threshold (Red Line): Misses Target 2 (because noise is high there) OR detects noise at the end.
    -   CFAR Threshold (Blue Line): Follows the noise ramp. Detects both targets.
3.  **Experiment:**
    -   Place two targets very close: `sig[50] += 15`, `sig[55] += 15`.
    -   Run CA-CFAR.
    -   **Result:** The threshold *rises* between them (because Target 1 is in Target 2's training window). Both might be missed.
    -   **Solution:** Switch to **OS-CFAR**. (Homework: Implement `np.median` instead of `np.mean`).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Edge Effects
**Symptom:** No detections at the start/end of the array.
**Cause:** The sliding window cannot fit.
**Solution:** Pad the signal with zeros or mirror it. Or just accept that min range = window size.

#### 2. False Alarms in Clutter
**Symptom:** CFAR detects "Ground" as a target.
**Cause:** Ground reflection is not "Noise" (random), it's "Clutter" (structured). CFAR assumes noise is random.
**Solution:** Static Clutter Removal (Day 107) *before* CFAR.

---

## ⚡ Optimization & Best Practices

### 1. 2D CFAR
For Range-Doppler Maps, we use a 2D window.
-   Training Cells: A rectangular ring around the CUT.
-   Guard Cells: An inner rectangular ring.
-   Logic is the same: Average the ring -> Threshold.

### 2. Log-Domain CFAR
Radar data is usually in dB ($20 \log_{10}(x)$).
-   In Linear domain: $Threshold = Noise \times Offset$.
-   In Log domain: $Threshold_{dB} = Noise_{dB} + Offset_{dB}$.
-   Addition is faster than multiplication on DSPs.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we need Guard Cells?
    *   **A:** If the target spills over into the Training Cells (due to leakage), it raises the noise estimate, potentially hiding itself (Self-Masking). Guard cells ensure we only measure pure noise.
2.  **Q:** What is the "False Alarm Rate"?
    *   **A:** The probability that a noise spike crosses the threshold. $P_{fa} = 10^{-6}$ means 1 false alarm per million samples.
3.  **Q:** Which is slower: CA-CFAR or OS-CFAR?
    *   **A:** OS-CFAR, because it requires sorting the training cells.

### Challenge Task
**Task:** 2D CFAR.
1.  Create a 2D matrix with noise and 2 targets.
2.  Implement `cfar_2d(matrix, train_r, train_d, guard_r, guard_d)`.
3.  Iterate `row` and `col`.
4.  Extract the window. Mask out the guard zone. Mean the rest.

---

## 📚 Further Reading & References
-   [CFAR Algorithms Review](https://www.mathworks.com/help/phased/ug/constant-false-alarm-rate-cfar-detection.html)
-   [Radar Basics - CFAR](https://www.radartutorial.eu/01.basics/False%20Alarm%20Rate.en.html)

---

**Day 108 Complete** | Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing
