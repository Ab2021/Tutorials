# Day 131: Neuromorphic Control (Spiking Neural Networks)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 19: Soft Robotics & Bio-Inspired Control

---

> **📝 Content Creator Instructions:**
> Digital is 1s and 0s. Biology is Spikes.
> - **Focus:** Leaky Integrate-and-Fire (LIF) Neurons, Event-Driven processing, Spike timing, and Energy efficiency compared to Deep Learning (ANNs).
> - **Code:** A Python script `snn_obstacle_avoidance.py` implementing a small SNN (Sensors $\to$ Spikes $\to$ Motor Neurons) to steer a simulated robot.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** ANN (Rate Coding, Continuous Relu) vs SNN (Temporal Coding, Binary Spikes).
2.  **Simulate** a LIF Neuron: $V(t+1) = V(t) - Leak + Input$. If $V > V_{thresh}$ then Spike & Reset.
3.  **Implement** Braitenberg Vehicle logic using Spikes (Direct sensor-motor connection).
4.  **Explain** why SNNs are crucial for Event Cameras (Day 108).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulation). Hardware (Intel Loihi / SpiNNaker) is expensive/rare.

### Software Environment
```bash
pip install numpy matplotlib
# Optional: bindsnet or snntorch
```

### Prior Knowledge
- Neural Networks (Weights).
- Event Cameras.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Biological Neuron

Real neurons don't output "0.75". They stay silent, then **BANG** (Action Potential), then silence.
*   **Information:** Encoded in the *timing* and *frequency* of spikes.
*   **Energy:** No spikes = Zero energy consumption. (Unlike updates in CNNs which burn power constantly).

### 🔹 Part 2: Leaky Integrate-and-Fire (LIF)

Mathematical Model:
$$ \tau \frac{dV}{dt} = -(V - V_{rest}) + R I(t) $$
*   **Integrate:** Input current charges the membrane voltage $V$.
*   **Leak:** Voltage decays over time (Forget).
*   **Fire:** If $V > V_{thresh}$, emit Spike ($S=1$), Reset $V \to V_{reset}$.

### 🔹 Part 3: Encoding & Decoding

*   **Rate Coding:** Stronger signal $\to$ More spikes per second.
*   **Temporal Coding:** Stronger signal $\to$ Spike happens *sooner*.
*   **Motor Control:** Muscles integrate spikes. More spikes = More contraction force.

---

## 💻 Implementation: Spiking Braitenberg

We actuate a robot using simulated SNN.
*   **Left Sensor:** Distance to obstacle.
*   **Right Sensor:** Distance to obstacle.
*   **Logic:** Cross-wiring (Excitatory). Left Sensor stimulates Right Wheel.
*   **Result:** Simple Avoidance (Fear).

### 🛠️ Project Structure
```text
day131_snn/
├── src/
│   ├── snn_sim.py
└── output/
    ├── spike_train.png
```

### 👨‍💻 SNN Simulation (`src/snn_sim.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, id_name):
        self.id = id_name
        self.v = 0.0
        self.v_thresh = 1.0
        self.leak = 0.1 # Decay factor
        self.history = []
        self.spikes = []

    def step(self, input_current, time_step):
        # 1. Integration
        self.v += input_current
        
        # 2. Leak
        self.v -= self.leak * self.v
        
        # 3. Fire
        spike = 0
        if self.v >= self.v_thresh:
            spike = 1
            self.v = 0.0 # Reset (or V_rest)
            
        self.history.append(self.v)
        if spike: self.spikes.append(time_step)
        
        return spike

class SpikingRobot:
    def __init__(self):
        # 2 Input Neurons (Sensors)
        self.n_sensor_L = LIFNeuron("SensL")
        self.n_sensor_R = LIFNeuron("SensR")
        
        # 2 Motor Neurons (Wheels)
        self.n_motor_L = LIFNeuron("MotL")
        self.n_motor_R = LIFNeuron("MotR")
        
        # Synaptic Weights
        # Braitenberg "Fear": Left Sensor excites Right Motor
        self.w_LR = 2.0 # Left -> Right
        self.w_RL = 2.0 # Right -> Left
        self.w_Self = 0.1 # Self excitation to keep moving?
        
        # Robot State
        self.pos = np.array([0.0, 0.0])
        self.angle = np.pi/2 # Facing Up
        self.dt = 0.1

    def step(self, t, obstacles):
        # 1. Sense (Rate Coding)
        # Distance to obstacles
        # Simple Raycast for Left and Right sensors
        # Angles: +30, -30 deg
        pL = self.raycast(self.angle + 0.5, obstacles)
        pR = self.raycast(self.angle - 0.5, obstacles)
        
        # Input Current = 1 / Distance (Closer = Higher Current)
        i_L = 1.0 / (pL + 0.1)
        i_R = 1.0 / (pR + 0.1)
        
        # 2. Update Sensor Neurons
        s_L = self.n_sensor_L.step(i_L, t)
        s_R = self.n_sensor_R.step(i_R, t)
        
        # 3. Synaptic Transmission
        # Current to motors depends on SPIKES from sensors
        # If Sensor Spikes, Inject charge into Motor
        # Add base 0.5 to keep moving forward in open space
        i_mot_L = 0.5 + (s_R * self.w_RL) 
        i_mot_R = 0.5 + (s_L * self.w_LR)
        
        # 4. Update Motor Neurons
        spike_mL = self.n_motor_L.step(i_mot_L, t)
        spike_mR = self.n_motor_R.step(i_mot_R, t)
        
        # 5. Actuate (Spike Integration)
        # Motor Speed is filtered spike train (Low Pass)
        # Simplified: Speed = 1.0 if spike, decays otherwise?
        # Let's say Speed is proportional to recent spike rate.
        # Simple Step: If spike, move.
        
        vL = 1.0 if spike_mL else 0.0
        vR = 1.0 if spike_mR else 0.0
        
        # Unicycle Kinematics
        V = (vL + vR) / 2.0
        W = (vR - vL) / 1.0 # Width 1.0
        
        self.pos[0] += V * np.cos(self.angle) * self.dt
        self.pos[1] += V * np.sin(self.angle) * self.dt
        self.angle += W * self.dt
        
        return self.pos

    def raycast(self, angle, obstacles):
        # Find closest obstacle in direction
        min_d = 10.0
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        for obs in obstacles:
            # Sphere intersection logic (simplified point check)
            # Find t where pos + t*dir is close to obs
            # Just distance check for simplicity:
            # Project vector to obstacle onto direction vector
            vec_to = obs - self.pos
            proj = np.dot(vec_to, np.array([dx, dy]))
            if proj > 0:
                perp_dist = np.linalg.norm(vec_to - proj*np.array([dx, dy]))
                if perp_dist < 0.5: # Radius
                    d = proj
                    if d < min_d: min_d = d
        return min_d

def main():
    bot = SpikingRobot()
    obstacles = [np.array([0.0, 5.0])] # Obstacle straight ahead
    
    path = []
    
    for t in range(200):
        pos = bot.step(t, obstacles)
        path.append(pos.copy())
        
    path = np.array(path)
    
    # Plot Path
    plt.figure()
    plt.plot(path[:,0], path[:,1], 'b-', label='Robot Path')
    plt.plot(obstacles[0][0], obstacles[0][1], 'ro', markersize=20, label='Obstacle')
    plt.legend()
    plt.axis('equal')
    plt.title("SNN Braitenberg Vehicle")
    plt.savefig("output/snn_path.png")
    
    # Plot Spikes
    plt.figure()
    plt.eventplot(bot.n_sensor_L.spikes, lineoffsets=1, linelengths=0.5, label='SensL')
    plt.eventplot(bot.n_motor_R.spikes, lineoffsets=2, linelengths=0.5, label='MotR')
    plt.yticks([1, 2], ['SensL', 'MotR'])
    plt.title("Spike Raster Plot")
    plt.savefig("output/spike_train.png")
    print("SNN Sim Complete.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Synaptic Plasticity"

### 1. Lab Objectives
- **Run:** Sim. Robot avoids obstacle.
- **Analyze:** Raster plot. When Robot approaches obstacle (Distance $\downarrow$), Sensor L fires rapidly. This triggers Motor R to fire rapidly. Robot turns Right.
- **Modify:** Add STDP (Spike-Time Dependent Plasticity).
    *   If Motor fires *after* Sensor (Causal), increase weight.
    *   If Motor fires *before* Sensor (Acausal), decrease weight.
- **Result:** The robot "Learns" the reflex. Weights start small, but after hitting a few walls, the connections strengthen. Associative Learning (Pavlovian).

---

## 🚀 Project: "Event-Based Flow"

**Goal:** Process Event Camera data with SNN.
1.  **Input:** Event stream (from Day 108).
2.  **Network:** 10x10 Input Layer (Retina).
3.  **Task:** Detect "Right Motion".
4.  **Wiring:**
    *   Pixel $(x, t)$ connects to Neuron Output with delay $\delta$.
    *   Pixel $(x+1, t+\delta)$ connects to Neuron Output.
    *   If Object moves Right, both spikes arrive at Neuron Output at the *same time* (coincidence detection).
    *   Neuron fires.
5.  **Result:** Direction Selective Ganglion Cells (DSGC).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Sim is Slow"
*   **Cause:** Iterating every neuron in Python.
*   **Fix:** Matrix multiplication. `V += I` where V is a vector of 1000 neurons. Use PyTorch or NumPy.

#### 2. "No Spikes"
*   **Cause:** Input current ($I$) < Leak * Threshold. The voltage never reaches threshold.
*   **Fix:** Increase Gain (Weights) or decrease Threshold.

---

## ⚡ Optimization: Hardware SNN

Intel Loihi / SpiNNaker.
*   Mapping this Python code to Loihi saves 1000x energy.
*   Used in drones for "Always On" surveillance.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why are SNNs harder to train than CNNs?
    *   **A:** The "Fire" step (Threshold) is non-differentiable (Step function). Backprop doesn't work directly. Surrogate Gradients are used.
2.  **Q:** What is "Leak"?
    *   **A:** The memory limit. If inputs are too far apart in time, the neuron forgets the first one before the second arrives.
3.  **Q:** Braitenberg Vehicle "Love" vs "Fear"?
    *   **A:** "Fear" = Crossed (L->R). Turn away from source. "Love" = Uncrossed (L->L). Turn towards source.

### Challenge Task
> **Task:** Logic Gates.
> 1. Build an AND gate using 2 Input Neurons + 1 Output Neuron.
> 2. Weights must be such that 1 spike is not enough, but 2 coincident spikes trigger Output.
> 3. Tune $V_{thresh}$.

---

## 📚 Further Reading
- **SpiNNaker:** Massive parallel neuromorphic supercomputer.
- **Nengo:** Python framework for building large brain models.

---

**Day 131 Complete**
