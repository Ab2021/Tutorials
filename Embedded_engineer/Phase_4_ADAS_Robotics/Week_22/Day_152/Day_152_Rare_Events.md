# Day 152: Rare Events (Emergency Vehicles)
## Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases

---

> **📝 Day 152 Focus:**
> You hear a siren. You pull over. An autonomous car must do the same. **Emergency Vehicle Detection** is a critical safety requirement. It involves **Audio Processing** (Sirens) and **Visual Detection** (Flashing Lights).

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Analyze** Audio Signals using FFT (Fast Fourier Transform).
2.  **Detect** Siren frequencies (Yelp, Wail, Hi-Lo).
3.  **Identify** Flashing Lights using temporal visual analysis.
4.  **Implement** a "Yield" state in the Behavior Planner.
5.  **Discuss** the legal requirements for AVs and emergency vehicles.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Signal Processing:** Frequency, Spectrograms.
-   **Day 138:** Behavior Planning (FSM).

### Hardware Requirements
-   **Microphone:** For real testing (Optional).

### Software Stack
-   **Python:** `scipy.signal`, `numpy`, `librosa` (Audio analysis).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Audio Detection (The Siren)

-   **Siren Types:**
    -   **Wail:** Slow sweep (500Hz - 1500Hz).
    -   **Yelp:** Fast sweep.
    -   **Hi-Lo:** Two distinct tones (European).
-   **Technique:** Short-Time Fourier Transform (STFT). Look for high energy in specific frequency bands with periodic modulation.
-   **Doppler Effect:** Frequency shift indicates if the ambulance is approaching or receding.

### 🔹 Part 2: Visual Detection (The Lights)

-   **Feature:** High intensity, specific colors (Red/Blue), periodic flashing (1-4 Hz).
-   **Challenge:** Distinguishing from traffic lights, billboards, or turn signals.
-   **Solution:** Temporal analysis. Track a bright spot over time. If it oscillates Red/Blue at 2Hz, it's a police car.

### 🔹 Part 3: The Behavior (Yield)

-   **Rule:** "Pull over to the right and stop."
-   **Logic:**
    1.  Detect EV (Emergency Vehicle).
    2.  Check Right Lane availability.
    3.  Change Lane Right.
    4.  Stop.
    5.  Wait until EV passes (Audio intensity drops).

---

## 💻 Implementation: Siren Detector

**Scenario:**
-   Input: Audio file (WAV).
-   Task: Detect if a siren is present.

### 🛠️ Setup
Create `week22_day152` and `siren_detect.py`.

```bash
mkdir -p ~/ros2_ws/src/week22_day152
cd ~/ros2_ws/src/week22_day152
pip install scipy librosa matplotlib
touch siren_detect.py
```

### 👨‍💻 Code: FFT Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def generate_synthetic_siren(duration=5.0, fs=44100):
    t = np.linspace(0, duration, int(fs*duration))
    # Wail: 500-1500Hz sweep, 4 second period
    freq = 1000 + 500 * np.sin(2 * np.pi * 0.25 * t)
    signal = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / fs)
    
    # Add Noise (Traffic)
    noise = np.random.normal(0, 0.1, len(t))
    return t, signal + noise, fs

def detect_siren(signal, fs):
    # Compute Spectrogram
    f, t, Sxx = spectrogram(signal, fs, nperseg=1024)
    
    # Siren Band: 500Hz - 2000Hz
    idx_min = np.searchsorted(f, 500)
    idx_max = np.searchsorted(f, 2000)
    
    # Extract energy in band
    band_energy = np.mean(Sxx[idx_min:idx_max, :], axis=0)
    
    # Threshold
    threshold = np.mean(Sxx) * 5.0
    detections = band_energy > threshold
    
    return t, band_energy, detections

def main():
    # 1. Generate Audio
    t, signal, fs = generate_synthetic_siren()
    
    # 2. Detect
    time_bins, energy, detections = detect_siren(signal, fs)
    
    # 3. Visualize
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, signal)
    plt.title("Audio Signal (Synthetic Siren + Noise)")
    
    plt.subplot(3, 1, 2)
    plt.specgram(signal, Fs=fs, NFFT=1024, noverlap=512)
    plt.title("Spectrogram")
    plt.ylabel("Frequency (Hz)")
    
    plt.subplot(3, 1, 3)
    plt.plot(time_bins, energy, label='Band Energy')
    plt.plot(time_bins, detections * np.max(energy), 'r', alpha=0.3, label='Detection')
    plt.title("Detection Logic")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Yield State Machine

### Lab Objectives
1.  **Modify Day 138 FSM:**
    -   Add state `YIELD_TO_EV`.
    -   Add input `ev_detected` (Boolean).
2.  **Logic:**
    ```python
    if ev_detected:
        if lane != RIGHT_MOST:
            state = CHANGE_LANE_RIGHT
        else:
            state = STOP
    ```
3.  **Simulate:**
    -   Trigger `ev_detected = True`.
    -   Observe the car pulling over.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. False Positives (Music)
**Symptom:** Car yields when loud music plays.
**Cause:** Music also has energy in 500-2000Hz.
**Solution:** Look for the *sweep* pattern (Frequency changing over time). Music beats are rhythmic; sirens are continuous sweeps.

#### 2. Localization of Sound
**Symptom:** Car yields to an ambulance 3 blocks away.
**Cause:** Sirens are loud. Hard to tell distance/direction with one mic.
**Solution:** **Microphone Array**. Use Time Difference of Arrival (TDOA) between 4 mics to estimate direction (DOA).

---

## ⚡ Optimization & Best Practices

### 1. V2X (Vehicle-to-Everything)
The future solution.
-   Ambulance broadcasts "I am here" via DSRC/C-V2X.
-   AV receives message and yields before hearing the siren.
-   **Latency:** < 100ms.
-   **Range:** 300m+.

### 2. Flashing Light Detection
-   Use **Event Cameras** (DVS).
-   They only record changes. Flashing lights trigger massive event streams at the flash frequency.
-   Extremely fast response time (< 1ms).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is audio detection necessary if we have cameras?
    *   **A:** Line of Sight. You can hear an ambulance around a corner before you can see it.
2.  **Q:** What is the frequency range of a typical siren?
    *   **A:** 500 Hz to 1500 Hz (approx).
3.  **Q:** How does a Microphone Array help?
    *   **A:** It allows "Sound Source Localization" to determine if the siren is behind you (yield) or on a cross street (ignore).

### Challenge Task
**Task:** Visual Flasher.
1.  Create a video with a flashing red circle (2Hz).
2.  Write a script to extract the average intensity of the red channel in a ROI.
3.  Compute FFT of the intensity signal.
4.  Look for a peak at 2Hz.

---

## 📚 Further Reading & References
-   [Acoustic Detection of Sirens](https://ieeexplore.ieee.org/document/8500547)
-   [Waymo Emergency Vehicle Testing](https://waymo.com/safety/)

---

**Day 152 Complete** | Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases
