# Day 111: Micro-Doppler Signatures
## Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing

---

> **📝 Day 111 Focus:**
> A car is a rigid body. A pedestrian is a bag of swinging limbs. These internal motions cause **Micro-Doppler** shifts around the main Doppler frequency. By analyzing these signatures, Radar can distinguish a human from a mailbox or a car.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the Micro-Doppler effect (modulation of Doppler shift).
2.  **Generate** a Spectrogram using STFT (Short-Time Fourier Transform).
3.  **Differentiate** signatures: Pedestrian (swinging arms), Cyclist (pedaling), Car (rigid).
4.  **Extract** features (Bandwidth, Periodicity) for classification.
5.  **Train** a simple classifier (SVM/CNN) on Radar Spectrograms.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 107:** Doppler FFT.
-   **Signal Processing:** Spectrograms.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `scipy.signal`, `sklearn`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Rigid vs Non-Rigid Motion

-   **Rigid Body (Car):** Every point moves at velocity $v$. Doppler spectrum is a sharp peak at $f_D = 2v/\lambda$.
-   **Non-Rigid Body (Human):**
    -   Torso moves at $v$.
    -   Arms/Legs swing forward ($v + v_{swing}$) and backward ($v - v_{swing}$).
    -   **Result:** The Doppler peak spreads out and oscillates periodically.

### 🔹 Part 2: The Spectrogram (Time-Frequency)

To see these oscillations, we can't just take one FFT. We need to see how the frequency changes over time.
-   **STFT:** Break the signal into short windows (e.g., 50ms) and FFT each window.
-   **Plot:** X-axis = Time, Y-axis = Doppler Frequency, Color = Intensity.

### 🔹 Part 3: Signatures

-   **Pedestrian:** Torso line + sinusoidal envelopes (limbs).
-   **Cyclist:** Torso line + high frequency spikes (wheels/pedals).
-   **Drone:** High frequency lines (propellers).

---

## 💻 Implementation: Micro-Doppler Simulator

**Scenario:**
-   Simulate a Pedestrian walking.
-   Torso: Constant velocity.
-   Arms: Sinusoidal velocity modulation.
-   Generate Spectrogram.

### 🛠️ Setup
Create `week16_day111` and `micro_doppler.py`.

```bash
mkdir -p ~/ros2_ws/src/week16_day111
cd ~/ros2_ws/src/week16_day111
touch micro_doppler.py
```

### 👨‍💻 Code: Generating Signatures

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# --- Config ---
fc = 77e9
c = 3e8
lambda_ = c / fc
fs = 1000.0 # Sampling Rate (Hz) - Slow time sampling
duration = 2.0 # Seconds

def simulate_pedestrian():
    t = np.linspace(0, duration, int(fs * duration))
    
    # 1. Torso Motion (Constant 1.5 m/s)
    v_torso = 1.5
    phase_torso = 2 * np.pi * (2 * v_torso / lambda_) * t
    sig_torso = np.exp(1j * phase_torso)
    
    # 2. Arm Motion (Swing relative to torso)
    # Swing freq ~ 1 Hz, Max swing speed ~ 1.0 m/s
    v_swing_max = 1.0
    f_swing = 1.0
    v_arm = v_torso + v_swing_max * np.cos(2 * np.pi * f_swing * t)
    
    # Integrate velocity to get phase
    # phi = integral(2 * v / lambda) dt
    phase_arm = (4 * np.pi / lambda_) * (v_torso * t + (v_swing_max / (2*np.pi*f_swing)) * np.sin(2*np.pi*f_swing*t))
    sig_arm = 0.5 * np.exp(1j * phase_arm) # Lower RCS than torso
    
    # 3. Leg Motion (Similar to arm but anti-phase and stronger)
    v_leg = v_torso + v_swing_max * np.cos(2 * np.pi * f_swing * t + np.pi)
    phase_leg = (4 * np.pi / lambda_) * (v_torso * t + (v_swing_max / (2*np.pi*f_swing)) * np.sin(2*np.pi*f_swing*t + np.pi))
    sig_leg = 0.8 * np.exp(1j * phase_leg)
    
    # Combine
    signal = sig_torso + sig_arm + sig_leg
    
    # Add Noise
    noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.1
    return t, signal + noise

def plot_spectrogram(t, signal):
    f, t_spec, Zxx = stft(signal, fs, nperseg=128, noverlap=120)
    
    # Convert Freq to Velocity
    # f_D = 2v/lambda -> v = f_D * lambda / 2
    v = f * lambda_ / 2
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t_spec, v, np.abs(Zxx), shading='gouraud', cmap='jet')
    plt.title("Micro-Doppler Spectrogram (Pedestrian)")
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Time (s)")
    plt.ylim(-1, 4) # Focus on forward motion
    plt.colorbar(label="Magnitude")
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    t, sig = simulate_pedestrian()
    plot_spectrogram(t, sig)
```

---

## 🔬 Lab Exercise: The Classification

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   A central line at $1.5$ m/s (Torso).
    -   Wavy lines going up to $2.5$ m/s and down to $0.5$ m/s (Limbs).
    -   This "envelope" is characteristic of a human.
3.  **Experiment:**
    -   Change `v_swing_max` to 0.0 (Rigid Body).
    -   **Result:** A straight line. Looks like a slow car or a robot.
    -   **Application:** If Radar sees a straight line at 1.5 m/s, it might be a shopping cart. If it sees waves, it's a child.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Low Resolution
**Symptom:** Spectrogram looks blocky.
**Cause:** `nperseg` (Window size) is too small.
**Solution:** Increase `nperseg` for better frequency resolution, but this reduces time resolution (Heisenberg Uncertainty). Find a balance.

#### 2. Aliasing
**Symptom:** Limbs wrap around to negative velocity.
**Cause:** Sampling rate ($f_s$) is too low relative to Doppler shift.
**Solution:** Increase PRF (Pulse Repetition Frequency).

---

## ⚡ Optimization & Best Practices

### 1. Feature Extraction
Don't feed the whole image to a CNN (too slow). Extract:
-   **Torso Velocity:** Mean of the envelope.
-   **Bandwidth:** Max - Min velocity.
-   **Period:** Time between peaks.
-   Feed these 3 numbers into a Decision Tree.

### 2. 4D Radar
Modern Imaging Radars have high resolution.
-   They can separate the Arm from the Torso in *Range* and *Angle* too.
-   This makes classification trivial (You literally see a stick figure).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why does a wheel have a Micro-Doppler signature?
    *   **A:** The top of the wheel moves at $2v$, the bottom at $0$, and the hub at $v$. This creates a spread of velocities.
2.  **Q:** Can Micro-Doppler detect a heartbeat?
    *   **A:** Yes! Vital Signs Monitoring radars (60 GHz) can detect chest displacement of 0.5mm.
3.  **Q:** What is STFT?
    *   **A:** Short-Time Fourier Transform. It gives frequency content over time.

### Challenge Task
**Task:** Drone Signature.
1.  Simulate a drone.
2.  Body $V=0$.
3.  Propellers: High frequency modulation (e.g., 100 Hz).
4.  Result: A line at 0 with sidebands at $\pm 100$ Hz.

---

## 📚 Further Reading & References
-   [Micro-Doppler Effect in Radar (Chen)](https://us.artechhouse.com/The-Micro-Doppler-Effect-in-Radar-Second-Edition-P1963.aspx)
-   [Google Soli (Gesture Radar)](https://atap.google.com/soli/)

---

**Day 111 Complete** | Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing
