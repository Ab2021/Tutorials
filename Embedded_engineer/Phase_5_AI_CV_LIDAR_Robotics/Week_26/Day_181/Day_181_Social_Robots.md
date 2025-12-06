# Day 181: Social Robots
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 26: Human-Robot Collaboration

---

> **📝 Content Creator Instructions:**
> Manners matter, even for machines.
> - **Focus:** Social Navigation, Proxemics, Emotion Recognition, and "Polite" Interaction.
> - **Code:** `social_costmap_layer.cpp` (Mocked in Python for concept). A custom ROS 2 Costmap Layer that inflates cost around humans based on their orientation (Gaussian Filter).
> - **Concept:** "Passing on the Left", "Face-to-Face" interaction.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Hall's Proxemic Zones (Intimate, Personal, Social, Public).
2.  **Implement** a Social Costmap Layer that respects Personal Space ($d > 0.45m$).
3.  **Detect** Basic Emotions from facial expressions to modulate robot behavior.
4.  **Execute** a "Polite Passing" maneuver.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Camera or Lidar (to detect Humans).
- Mobile Robot (TurtleBot).

### Software Environment
```bash
pip install fer # Facial Expression Recognition
```

### Prior Knowledge
- Nav2 Costmaps (Week 11).
- Gaussian Distributions.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Proxemics (Edward Hall)

The distance we keep from others defines our relationship.
1.  **Intimate (< 0.45m):** Embracing, whispering. Robot entry = "Threat".
2.  **Personal (0.45 - 1.2m):** Friends. "Collaborative Mode".
3.  **Social (1.2 - 3.6m):** Strangers. "Navigation Mode".
4.  **Public (> 3.6m):** Public speaking.

### 🔹 Part 2: Social Navigation

It's not just obstacle avoidance.
*   **Asymmetric Cost:** Humans walk forward. They are more uncomfortable if you cross *in front* of them than *behind* them.
*   **The Social Force:** Treat humans as repulsors, but shape the field elliptically.

---

## 💻 Implementation: The Social Costmap

We implement a simplified "Social Layer" calculation.
Usually done in C++ (ROS 2 Nav2 plugin), we prototype the math in Python.

### 🛠️ Project Structure
```text
day181_social/
├── src/
│   ├── social_layer_proto.py
│   └── emotion_detector.py
└── config/
    └── social_nav_params.yaml
```

### 👨‍💻 Emotion Detector (`src/emotion_detector.py`)

```python
from fer import FER
import cv2

class EmpathyModule:
    def __init__(self):
        self.detector = FER(mtcnn=True) 
        
    def analyze(self, frame):
        # Detect emotions: {angry, disgust, fear, happy, sad, surprise, neutral}
        analysis = self.detector.detect_emotions(frame)
        
        if not analysis:
            return "neutral", 0.0
            
        top_emotion = max(analysis[0]['emotions'], key=analysis[0]['emotions'].get)
        score = analysis[0]['emotions'][top_emotion]
        
        # Behavioral Logic
        if top_emotion == "angry":
            print("[ROBOT] Human is Angry -> Increasing Safety Margins.")
        elif top_emotion == "happy":
            print("[ROBOT] Human is Happy -> Enabling Fast Mode.")
            
        return top_emotion, score

# (Usage involves passing CV2 frames)
```

### 👨‍💻 Social Layer Prototype (`src/social_layer_proto.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class SocialCostmap:
    def __init__(self, size=100, res=0.1):
        self.size = size # pixels
        self.res = res   # m/pixel
        self.grid = np.zeros((size, size))
        
    def add_human(self, h_x, h_y, h_theta):
        """
        Add a human constraint.
        Asymmetric Gaussian: Longer in front, shorter in back.
        """
        sigma_x = 0.6 / self.res # Standard Person Radius + Buffer
        sigma_y = 0.4 / self.res 
        
        # Front bias
        # If x_rel > 0 (front), sigma = large. If x_rel < 0 (back), sigma = small.
        
        for i in range(self.size):
            for j in range(self.size):
                # Pixel to World
                wx = i * self.res
                wy = j * self.res
                
                # Transform to Human Frame
                dx = wx - h_x
                dy = wy - h_y
                
                # Rotate
                local_x = dx * np.cos(h_theta) + dy * np.sin(h_theta)
                local_y = -dx * np.sin(h_theta) + dy * np.cos(h_theta)
                
                # Asymmetric Sigma
                sig_x_eff = sigma_x * 2.0 if local_x > 0 else sigma_x * 0.5
                
                # Gaussian
                val = np.exp(-0.5 * ( (local_x**2)/(sig_x_eff**2) + (local_y**2)/(sigma_y**2) ))
                
                # Add to grid cost (Scale 0-254)
                cost = val * 254.0
                self.grid[j, i] = max(self.grid[j, i], cost)

    def visualize(self):
        plt.imshow(self.grid, cmap='jet', origin='lower')
        plt.colorbar(label='Cost')
        plt.title("Social Costmap (Asymmetric)")
        plt.xlabel("X (cells)")
        plt.ylabel("Y (cells)")
        
        # Arrow for human
        # (Assuming center visual)
        # plt.arrow(...)
        
        plt.savefig('social_costmap.png')
        print("Saved visualization.")

def main():
    # 5m x 5m room
    sc = SocialCostmap(size=50, res=0.1)
    
    # Human at (2.5, 2.5) facing Right (0 rad)
    # Expect "Balloon" shape extending to Right
    sc.add_human(2.5, 2.5, 0.0)
    
    # Human at (1.0, 1.0) facing Up (PI/2)
    sc.add_human(1.0, 1.0, np.pi/2)
    
    sc.visualize()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Face the Music"

### 1. Lab Objectives
- **Run:** `social_layer_proto.py`.
- **Visualize:** The asymmetric Gaussian. Confirm the "Front" zone is larger (higher cost) than the "Back" zone.
- **Tweak:** Change `sig_x_eff` multipliers.
- **Sim:** Send a robot to pass the human. With Social Layer, it should loop *wide* around the front or duck *tight* behind the back. Standard Navigation treats human as a cylinder and grazes them equally.

---

## 🚀 Project: "The Queue"

**Goal:** Respect Queuing theory.
1.  **Scenario:** 3 Humans standing in a line ($d < 1.0m$).
2.  **Detection:** Recognize the "Line" structure.
3.  **Constraint:** Do *not* cut through the line. Treat the gaps between humans as "High Cost" connections.
4.  **Behavior:** Go around the end of the line.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot Freezes"
*   **Cause:** Social Layers inflate obstacles too much. In narrow corridors, adding buffers makes the space "Lethal".
*   **Fix:** Soft Costs vs Lethal Costs. Allow the robot to enter Personal Space *if* velocity is extremely low < 0.1 m/s (Apologetic behavior).

#### 2. "Oscillating Cost"
*   **Cause:** Human detection flickers or orientation jumps $\pi$ (180 flip).
*   **Fix:** Filter human pose. Assume velocity direction = orientation.

---

## ⚡ Optimization: Group Navigation

If 2 humans are talking (Face-to-Face):
*   **Standard Local Planner:** Sees 2 obstacles, might try to go *between* them.
*   **Social Planner:** Sees an "O-Space" (Conversation circle). Adds cost *between* them. Robot goes around the pair.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is passing behind preferred?
    *   **A:** It minimizes disturbance. The human doesn't see you, so they don't react/flinch. Crossing in front forces them to stop or slow down.
2.  **Q:** What is the "Uncanny Valley"?
    *   **A:** The dip in emotional response where a robot looks *almost* human but imperfectly so, causing revulsion. (Avoid hyper-realistic faces; rely on stylized cues).

### Challenge Task
> **Task:** "Review your Path".
> 1. Robot generates a Global Path.
> 2. Check: Does path intersect any "Intimate Zones"?
> 3. If Yes: Re-plan with higher cost function.
> 4. If No feasible path: Stop and "Ask for permission" (Beep/Voice).

---

## 📚 Further Reading
- **Socially Aware Navigation:** Survey papers.
- **SEAN 2.0:** Social Environment for Autonomous Navigation simulator.

---

**Day 181 Complete**
