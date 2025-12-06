# Day 190: Neuromorphic Computing for Robotics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 28: Future Technologies

---

> **📝 Content Creator Instructions:**
> Computing like a brain, not a calculator.
> - **Focus:** Event-based Vision (DVS), Spiking Neural Networks (SNN), and Energy Efficiency.
> - **Code:** `event_cam_sim.py`. Simulate event data generation (pixel intensity change) and a basic SNN (Leaky Integrate-and-Fire) to process it.
> - **Concept:** Asynchronous processing vs Frame-based processing.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** Frame-based cameras (FPS) with Event-based cameras (TPS).
2.  **Simulate** an Event Stream from a standard video.
3.  **Implement** a Leaky Integrate-and-Fire (LIF) neuron model.
4.  **Explain** the potential for microsecond-latency control.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulated). Real Event Cameras (Prophesee/Inivation) are expensive ($3k+).

### Software Environment
```bash
pip install numpy opencv-python matplotlib
```

### Prior Knowledge
- Neural Networks (ReLU).
- Basic Electronics (Capacitors/Integrators).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Frame Bottleneck

Standard cameras take a snapshot 30 times a second.
*   **Redundancy:** If nothing moves, you still process 1080p data.
*   **Latency:** You must wait 33ms for the next frame.
*   **Dynamic Range:** Bright sun + Dark Tunnel = Blindness.

### 🔹 Part 2: Silicon Retinas (Event Cameras)

Pixels work independently.
*   **Logic:** If `log(Intensity)` changes by `threshold`, emit event `(x, y, t, p)`.
*   **Output:** Asynchronous stream of spikes. 
*   **Benefits:** >10,000 Hz equivalent, HDR (140dB).

### 🔹 Part 3: Spiking Neural Networks (SNN)

ANNs calculate $y = \sigma(Wx)$. SNNs calculate membrane potential $V(t)$.
*   **LIF Model:** $\tau \frac{dV}{dt} = -(V - V_{rest}) + I(t)$.
*   When $V > V_{thresh}$, Fire Spike, Reset $V$.
*   Energy efficient (only compute on spike).

---

## 💻 Implementation: Event Simulator & SNN

We "fake" an event camera by diffing video frames. Then we feed events to an SNN.

### 🛠️ Project Structure
```text
day190_neuromorphic/
├── src/
│   ├── sim_events.py
│   └── snn_basic.py
└── output/
    └── spike_plot.png
```

### 👨‍💻 Event Simulator (`src/sim_events.py`)

```python
import cv2
import numpy as np

def video_to_events(frame_prev, frame_curr, threshold=10):
    """
    Generate events where pixel difference > threshold.
    """
    diff = frame_curr.astype(np.float32) - frame_prev.astype(np.float32)
    
    # Positive Events (ON)
    on_events = np.where(diff > threshold)
    # Negative Events (OFF)
    off_events = np.where(diff < -threshold)
    
    # Return list of (x, y, p)
    events = []
    
    # Vectorized extraction is faster, but loop for clarity
    # Zip ON events
    for y, x in zip(on_events[0], on_events[1]):
        events.append((x, y, 1))
        
    for y, x in zip(off_events[0], off_events[1]):
        events.append((x, y, -1))
        
    return events

# Test loop not shown, see main.
```

### 👨‍💻 SNN Implementation (`src/snn_basic.py`)

A simple LIF neuron that detects "Optical Flow" (conceptually).

```python
import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, tau=20.0, threshold=1.0):
        self.tau = tau # Time constant (ms)
        self.v_thresh = threshold
        self.v = 0.0 # Membrane potential
        self.v_rest = 0.0
        self.dt = 1.0 # ms
        
        self.spike_history = []
        self.v_history = []
        
    def step(self, input_current):
        # Differential Equation: dV/dt = -(V - V_rest)/tau + I
        dv = (-(self.v - self.v_rest) + input_current) / self.tau
        
        self.v += dv * self.dt
        
        spike = 0
        if self.v >= self.v_thresh:
            spike = 1
            self.v = self.v_rest # Reset
            
        self.v_history.append(self.v)
        self.spike_history.append(spike)
        return spike

def main():
    # Simulation: 100ms
    neuron = LIFNeuron()
    
    # Input: Random spikes coming in (Poisson process)
    # 0 = No input, 10 = Strong input
    inputs = np.zeros(100)
    inputs[20:40] = 5.0 # Stimulus A
    inputs[60:80] = 5.0 # Stimulus B
    
    for t in range(100):
        neuron.step(inputs[t])
        
    # Plot
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(inputs)
    plt.ylabel("Input I(t)")
    
    plt.subplot(2,1,2)
    plt.plot(neuron.v_history, label='Membrane V')
    plt.axhline(neuron.v_thresh, color='r', linestyle='--', label='Thresh')
    
    # Plot spikes
    spikes_t = [i for i, x in enumerate(neuron.spike_history) if x==1]
    plt.plot(spikes_t, [1.1]*len(spikes_t), 'kx', markersize=10, label='Spike')
    
    plt.legend()
    plt.savefig('output/snn_response.png')
    print("SNN Sim Complete.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Dodgeball"

### 1. Lab Objectives
- **Scenario:** A ball is thrown at the robot at 20 m/s.
- **Compare:**
    *   **Frame Camera (30fps):** See ball at $t=0$, next at $t=33ms$ (Ball moved 0.66m).
    *   **Event Camera:** See ball edge at microsecond resolution.
- **Simulate:** Calculate trajectory update rate. Event cam allows updating motor command at 1000Hz (1ms).
- **Result:** Frame camera realizes too late. Event camera dodges.

---

## 🚀 Project: "SNN Lane Follower"

**Goal:** Drive using spikes.
1.  **Input:** Event stream of lane markers.
2.  **Network:** 2 Output Neurons (Left, Right).
3.  **Weights:** Hebbian Learning. If Left pixels spike $\to$ Fire Left Neuron.
4.  **Control:** Steer = (SpikeRate_Right - SpikeRate_Left).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Too Much Noise"
*   **Cause:** Threshold too low. Thermal noise triggers pixel events.
*   **Fix:** Increase Log-Intensity threshold. Use "Hot Pixel" filter (Remove events without neighbors).

#### 2. "Data Explosion"
*   **Cause:** Moving the camera in a textured scene. Every edge fires.
*   **Fix:** Event cameras generate data proportional to scene complexity * motion. Stop moving if you don't need data.

---

## ⚡ Optimization: SpiNNaker / Loihi

Running SNNs on CPU is inefficient.
*   **Neuromorphic Hardware (Intel Loihi):** Hardware neurons. Extremely low power (milliwatts).
*   **Application:** Drone surveillance (Sleeping until movement is seen).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What information is LOST in an event camera?
    *   **A:** Static Intensity (Absolute brightness). You only know change. If you stare at a static wall, you see nothing (Gray).
2.  **Q:** Why are SNNs harder to train?
    *   **A:** The "Spike" function is non-differentiable (Step function). Backprop doesn't work directly. (Need "Surrogate Gradient" methods).

### Challenge Task
> **Task:** "Reconstruct the Image".
> 1. Integrate the events over time to guess the absolute brightness.
> 2. $I(t) = I(0) + \sum Events$.
> 3. Observe "Drift" and "Ghosting".

---

## 📚 Further Reading
- **Scaramuzza (UZH):** Event-based Vision Survey.
- **Prophesee:** Metavision SDK.

---

**Day 190 Complete**
