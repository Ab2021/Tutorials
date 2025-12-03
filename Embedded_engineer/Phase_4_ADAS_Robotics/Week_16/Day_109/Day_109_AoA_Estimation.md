# Day 109: Angle of Arrival (AoA)
## Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing

---

> **📝 Day 109 Focus:**
> We know Range (Time Delay). We know Velocity (Doppler Phase). But where is the car? Left or Right? **Angle of Arrival (AoA)** uses multiple antennas to measure the phase difference of the incoming wave, allowing us to "see" in 3D.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the geometry of a Uniform Linear Array (ULA).
2.  **Derive** the relationship between Phase Difference ($\Delta \phi$) and Angle ($\theta$).
3.  **Implement** Angle FFT (Spatial FFT).
4.  **Visualize** the Radar Point Cloud (Range, Velocity, Angle).
5.  **Analyze** the Field of View (FOV) vs. Antenna Spacing.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 107:** Phase Shift.
-   **Geometry:** $\sin(\theta)$.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Phase Difference

Imagine a wavefront hitting two antennas separated by distance $d$.
-   The wave hits Antenna 1 first, then travels an extra distance $\Delta d = d \sin(\theta)$ to hit Antenna 2.
-   This extra distance causes a phase shift:
    $$ \Delta \phi = \frac{2\pi \cdot d \sin(\theta)}{\lambda} $$
-   If we measure $\Delta \phi$, we can solve for $\theta$:
    $$ \theta = \arcsin \left( \frac{\lambda \Delta \phi}{2\pi d} \right) $$

### 🔹 Part 2: Angle FFT

If we have $N_{rx}$ antennas (e.g., 4 or 8), we get a sequence of phase shifts.
-   Just like Doppler FFT measures phase change over *time*, **Angle FFT** measures phase change over *space*.
-   Performing an FFT across the antenna dimension gives us peaks corresponding to the angles of targets.

### 🔹 Part 3: Field of View (FOV)

-   To avoid aliasing (Grating Lobes), we must keep $|\Delta \phi| < \pi$.
-   This implies $d \sin(\theta) < \lambda / 2$.
-   Max FOV occurs when $d = \lambda / 2$. Then FOV = $\pm 90^\circ$.
-   If $d > \lambda / 2$, the FOV shrinks, and ghost targets appear.

---

## 💻 Implementation: Angle Estimation

**Scenario:**
-   **Target:** At $30^\circ$ azimuth.
-   **Antennas:** 8 elements, spaced $\lambda/2$.
-   **Task:** Simulate received signals and perform Angle FFT.

### 🛠️ Setup
Create `week16_day109` and `aoa_sim.py`.

```bash
mkdir -p ~/ros2_ws/src/week16_day109
cd ~/ros2_ws/src/week16_day109
touch aoa_sim.py
```

### 👨‍💻 Code: Spatial FFT

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
fc = 77e9
c = 3e8
lambda_ = c / fc
d = lambda_ / 2     # Antenna Spacing
N_rx = 8            # Number of Antennas

def simulate_array_response(theta_deg):
    theta_rad = np.radians(theta_deg)
    
    # Phase shift per antenna
    # phi = 2*pi * d * sin(theta) / lambda
    # Since d = lambda/2, phi = pi * sin(theta)
    phase_step = np.pi * np.sin(theta_rad)
    
    # Signal vector [1, exp(j*phi), exp(j*2phi), ...]
    response = np.exp(1j * phase_step * np.arange(N_rx))
    
    # Add Noise
    noise = (np.random.randn(N_rx) + 1j * np.random.randn(N_rx)) * 0.1
    return response + noise

def angle_fft(signal):
    # Zero Padding for smoother plot
    N_fft = 64
    fft_out = np.fft.fft(signal, n=N_fft)
    fft_out = np.fft.fftshift(fft_out) # Center 0 degrees
    
    mag = np.abs(fft_out)
    mag_db = 20 * np.log10(mag + 1e-9)
    
    # Map FFT bins to Angles
    # w = 2*pi*k/N
    # w = pi * sin(theta)
    # sin(theta) = 2*k/N
    
    k = np.arange(-N_fft/2, N_fft/2)
    sin_theta = 2 * k / N_fft
    
    # Filter out |sin(theta)| > 1 (Invisible region)
    valid_idx = np.abs(sin_theta) <= 1
    sin_theta = sin_theta[valid_idx]
    mag_db = mag_db[valid_idx]
    
    angles = np.degrees(np.arcsin(sin_theta))
    
    return angles, mag_db

def main():
    # Target at 30 degrees
    true_angle = 30.0
    print(f"Simulating Target at {true_angle} degrees...")
    
    rx_signal = simulate_array_response(true_angle)
    
    # Add a second target at -15 degrees
    rx_signal += simulate_array_response(-15.0)
    
    angles, spectrum = angle_fft(rx_signal)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(angles, spectrum, 'b-o')
    plt.axvline(true_angle, color='g', linestyle='--', label='Ground Truth 1')
    plt.axvline(-15.0, color='r', linestyle='--', label='Ground Truth 2')
    
    plt.title(f"Angle FFT (N_rx={N_rx})")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Power (dB)")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Resolution Limit

### Lab Objectives
1.  Run the script.
2.  **Observation:** Peaks at $30^\circ$ and $-15^\circ$.
3.  **Experiment:**
    -   Set targets at $10^\circ$ and $15^\circ$.
    -   **Result:** They merge into one blob.
    -   **Theory:** Angle Resolution $\theta_{res} = \frac{2}{N_{rx}}$. (In radians).
    -   For $N_{rx}=8$, $\theta_{res} \approx 14^\circ$.
    -   To separate $10^\circ$ and $15^\circ$, you need more antennas (or Super-resolution algorithms like MUSIC).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Grating Lobes
**Symptom:** Target at $30^\circ$ appears at $30^\circ$ AND $-60^\circ$.
**Cause:** Antenna spacing $d > \lambda/2$.
**Solution:** Reduce spacing or use a non-uniform array.

#### 2. Calibration
**Symptom:** Peak is offset by $5^\circ$.
**Cause:** Cable lengths or PCB trace lengths differ between antennas, adding extra phase.
**Solution:** **Phase Calibration**. Measure a known target at $0^\circ$ and store the phase offsets. Subtract them during processing.

---

## ⚡ Optimization & Best Practices

### 1. Virtual Antennas (MIMO)
Physical antennas are expensive.
-   Use 3 Tx and 4 Rx.
-   Fire Tx1, measure 4 Rx.
-   Fire Tx2, measure 4 Rx.
-   Fire Tx3, measure 4 Rx.
-   Result: $3 \times 4 = 12$ **Virtual Antennas**.
-   Resolution improves as if you had 12 physical Rx.

### 2. MUSIC / ESPRIT
FFT resolution is limited by $N_{rx}$.
-   **Subspace Methods (MUSIC):** Can resolve angles much closer than the FFT limit, but require high SNR and heavy computation (Eigenvalue Decomposition).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** How many antennas do I need to distinguish Left from Right?
    *   **A:** At least 2. With 1 antenna, you only get Range.
2.  **Q:** Why is $d = \lambda/2$ the magic number?
    *   **A:** It corresponds to the Nyquist sampling theorem in space. Sampling closer is waste; sampling further causes aliasing.
3.  **Q:** Does Angle FFT work for Elevation (Up/Down)?
    *   **A:** Only if you have antennas stacked vertically. Most automotive radars are 2D (Azimuth only), but 4D Imaging Radars have vertical antennas too.

### Challenge Task
**Task:** FOV Calculation.
1.  If $d = 0.7 \lambda$.
2.  Aliasing starts when $\sin(\theta) = \lambda / (2d) = 1 / 1.4 = 0.71$.
3.  $\theta = \arcsin(0.71) \approx 45^\circ$.
4.  FOV is limited to $\pm 45^\circ$. Beyond that, ghosts appear.

---

## 📚 Further Reading & References
-   [Phased Array Antennas](https://www.microwaves101.com/encyclopedias/phased-array-antennas)
-   [MIMO Radar Explained](https://www.ti.com/lit/wp/spry328/spry328.pdf)

---

**Day 109 Complete** | Phase 4: ADAS & Robotics Systems | Week 16: Radar Signal Processing
