# Day 60: Domain Randomization (DR)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 9: Simulation & Sim-to-Real

---

> **📝 Content Creator Instructions:**
> Reality is varied. Simulation is uniform.
> - **Focus:** Visual Randomization (Texture/Light), Dynamics Randomization (Mass/Friction), and the concept of "Sim-to-Real Gap".
> - **Code:** Scripting scene variations in Gazebo to create a diverse dataset.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Domain Randomization and its role in bridging the Reality Gap.
2.  **Script** randomized World generation (Lighting, Object position, Textures).
3.  **Randomize** physics parameters (Friction coefficients, Mass) for robustness.
4.  **Train** a simple object detector on synthetic DR data and test on real webcam images.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
# Python scripts for modifying SDF/World files
pip install xml.etree.ElementTree
```

### Prior Knowledge
- Neural Network Overfitting.
- Gazebo Interaction.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Sim-to-Real Gap

If you support vector machines trained on blue cubes in Sim, they fail on blue cubes in Real Life. Why?
1.  **Visual Gap:** Real light has shadows, caustics, noise, ISO grain. Sim is perfect Phong shading.
2.  **Dynamics Gap:** Real friction is non-uniform. Motors have backlash. Sim is rigid body math.

### 🔹 Part 2: Domain Randomization (DR)

Instead of trying to make Sim *perfect* (Photorealism is hard), make Sim *chaotic*.
*   **Hypothesis:** If the model learns to detect a cube despite purple lights, polka-dot floors, and flying geometric distractors...
*   Then "Reality" is just another random variation of the domain.
*   **Result:** A Robust Policy.

### 🔹 Part 3: Parameters to Randomize

1.  **Visual:** Camera Pos, Light Pos/Color, Textures, Backgrounds.
2.  **Physics:** Link Mass ($\pm 10\%$), Joint Damping, Friction ($\mu \in [0.5, 1.0]$).
3.  **Sensor:** Noise levels, Calibration offsets.

---

## 💻 Implementation: Gazebo Randomizer

We will write a python script that generates random world files and launches them.

### 🛠️ Project Structure
```text
day60_dr/
├── templates/
│   └── world_template.sdf
├── textures/
│   ├── pattern1.png ...
└── scripts/
    └── generate_worlds.py
```

### 👨‍💻 World Generator (`scripts/generate_worlds.py`)

```python
import random
import xml.etree.ElementTree as ET

def randomize_lighting():
    # Random Diffuse Color
    r = random.random()
    g = random.random()
    b = random.random()
    return f"{r} {g} {b} 1"

def generate_world(template_path, output_path):
    tree = ET.parse(template_path)
    root = tree.getroot()
    
    world = root.find('world')
    
    # 1. Randomize Sun
    light = world.find('light')
    light.find('diffuse').text = randomize_lighting()
    
    # 2. Randomize Object Position
    # Assume we identified the object model link
    for model in world.findall('model'):
        if model.get('name') == 'target_cube':
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            model.find('pose').text = f"{x} {y} 0.5 0 0 0"
            
    # 3. Randomize Physics (Friction)
    physics = world.find('physics')
    # Modification of ODE params...
    
    tree.write(output_path)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    for i in range(5):
        generate_world('templates/world_template.sdf', f'generated_world_{i}.sdf')
```

### 👨‍💻 The Loop (Data Collection)

1.  Launch `generated_world_0.sdf`.
2.  Capture Camera Image $\to$ Save to `dataset/images/0.jpg`.
3.  Get Ground Truth Object Pose $\to$ Bounding Box $\to$ Save to `dataset/labels/0.txt`.
4.  Kill Gazebo. Repeat.

---

## 🔬 Lab Exercise: The "Psychedelic" Robot

### 1. Lab Objectives
- Train a CNN to classify "Cube" vs "Sphere".
- **Dataset A:** 100 images, Perfect Lighting, Grey Floor.
- **Dataset B (DR):** 100 images, Random Colors, Random Texture Floor, Random Lights.
- **Test:** Real Webcam feed (messy room).
- **Result:** Model A fails (detects floor as object). Model B works (ignores floor).

---

## 🚀 Project: "Physics Robustness"

**Goal:** Train an RL agent (PPO) to balance a pole (CartPole) in Sim.
1.  **Training:**
    *   Episode 1: Pole Mass = 1.0kg.
    *   Episode 2: Pole Mass = 1.5kg.
    *   Episode 3: Gravity = 9.0 m/s2.
2.  **Evaluator:** A "Real" CartPole.
3.  **Observation:** If you train only on Mass=1.0kg, the policy overfits. It fails if Real Mass is 1.01kg. DR forces the policy to be conservative and robust.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Sim is too weird"
*   **Symptom:** Training fails to converge.
*   **Cause:** Randomization range is too wide. (e.g., Gravity from -10 to +10). The task becomes impossible.
*   **Fix:** **Curriculum Learning**. Start with narrow randomization ($\pm 1\%$). Slowly expand to $\pm 20\%$.

#### 2. "Texture Mapping Fail"
*   **Symptom:** Objects appear all white.
*   **Cause:** Gazebo path issues for material scripts.
*   **Fix:** Ensure Ogre material scripts and texture PNGs are in `GAZEBO_RESOURCE_PATH`.

---

## ⚡ Optimization: Automatic DR (ADR)

OpenAI's approach for the Rubik's Cube Hand.
*   Don't set ranges manually.
*   Let the AI output the *difficulty* (randomization range).
*   If Policy succeeds, widen the range. If fails, narrow it.
*   Keeps the agent in the "Goldilocks Zone" of learning.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Catastrophic Forgetting" in DR?
    *   **A:** Not strictly DR, but if we randomize too much, the model might forget the "core" visual features.
2.  **Q:** Photorealism vs DR?
    *   **A:** Photorealism tries to match $P_{sim}(x) \approx P_{real}(x)$. DR tries to make $P_{real}(x) \subset Support(P_{sim}(x))$.
3.  **Q:** Texture Randomization?
    *   **A:** Applying random noise textures (Coco dataset images) to walls/floors prevents the Neural Net from overfitting to specific edge patterns.

### Challenge Task
> **Task:** Dynamic Lighting.
> 1. Write a plugin that moves the Sun light source in a circle *during* the simulation.
> 2. Watch the shadows move.
> 3. Does the Vision algorithm track the object or the shadow?

---

## 📚 Further Reading
- **Domain Randomization for Sim-to-Real Transfer:** Tobin et al. (OpenAI).
- **NVIDIA Isaac Gym:** Parallel GPU physics for massive DR.

---

**Day 60 Complete**
