# Day 82: Diffusion Policies (State of the Art)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 12: Robot Learning

---

> **📝 Content Creator Instructions:**
> "Go Left" (50%) + "Go Right" (50%) = "Go Straight into the pole" (Average).
> - **Focus:** Why Multimodality matters. Denoising Diffusion Probabilistic Models (DDPM) for Control.
> - **Code:** A PyTorch Diffusion Policy that learns to generate multimodal trajectories (e.g., avoiding an obstacle by going left OR right).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the Limitation of Explicit Policies (Gaussian/MSE) in multimodal tasks.
2.  **Implement** the Forward Process (Adding Noise) and Reverse Process (Denoising Network).
3.  **Train** a Diffusion Policy to predict Action Sequences from Observations.
4.  **Visualize** the iterative denoising steps ($K=100 \to K=0$).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU (Essential for Diffusion).

### Software Environment
```bash
pip install diffusers torch matplotlib
```

### Prior Knowledge
- Generative Models (VAEs, GANs).
- Day 78 (Behavior Cloning).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Multimodality Criterion

Imagine a robot facing a pole.
*   Expert A goes Left.
*   Expert B goes Right.
*   **MSE Loss (BC):** Minimizes distance to both. Result: Mean (Straight). CRASH.
*   **GMM (Gaussian Mixture Model):** Can model modes, but hard to train (collapse).
*   **Diffusion:** Naturally handles multimodality by learning the *gradient of the data distribution*.

### 🔹 Part 2: Diffusion for Control

1.  **Input:** Observation $O_t$.
2.  **Output:** Sequence of Actions $A_{t:t+H}$.
3.  **Training:**
    *   Take ground truth action sequence $A_{gt}$.
    *   Add Gaussian Noise $\epsilon \sim \mathcal{N}(0, I)$ (Forward Process).
    *   Train a Noise Prediction Network $\epsilon_\theta(A_{noisy}, t, O_t)$ to predict the noise.
4.  **Inference:**
    *   Start with pure noise $A_K \sim \mathcal{N}(0, I)$.
    *   Iteratively denoise: $A_{k-1} = \text{Denoise}(A_k, \epsilon_\theta(A_k))$.
    *   Result: A valid action sequence.

### 🔹 Part 3: Architecture

*   **Visual Encoder:** ResNet/ViT to encode $O_t$.
*   **Noise Predictor:** U-Net (1D) or Transformer.
*   **Conditioning:** Feature FiLM or Cross-Attention injects observation info into the Denoising process.

---

## 💻 Implementation: 1D Trajectory Diffusion

We will learn to navigate a "Y" junction.

### 🛠️ Project Structure
```text
day82_diffusion/
├── src/
│   ├── diffusion_policy.py
│   └── train_y_junction.py
└── plots/
    └── generation.gif
```

### 👨‍💻 Diffusion Network (`src/diffusion_policy.py`)

Simplified Denoising Network (MLP-based for simple 2D actions).

```python
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class NoisePredNet(nn.Module):
    def __init__(self, action_dim, obs_dim):
        super().__init__()
        # Input: Noisy Action + Observation + Time
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(32),
            nn.Linear(32, 64),
            nn.Mish(),
            nn.Linear(64, 64),
        )
        self.mid_mlp = nn.Sequential(
            nn.Linear(action_dim + obs_dim + 64, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, action_dim) # Predicts epsilon noise
        )

    def forward(self, noisy_action, time, observation):
        t_emb = self.time_mlp(time)
        x = torch.cat([noisy_action, observation, t_emb], dim=-1)
        return self.mid_mlp(x)

class DiffusionModel:
    def __init__(self, net, beta_min=1e-4, beta_max=0.02, T=100):
        self.net = net
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def loss(self, action, observation):
        batch_size = action.shape[0]
        # 1. Sample Time t
        t = torch.randint(0, self.T, (batch_size,)).to(action.device)
        
        # 2. Add Noise
        noise = torch.randn_like(action)
        alpha_bar = self.alphas_cumprod.to(action.device)[t][:, None]
        noisy_action = torch.sqrt(alpha_bar) * action + torch.sqrt(1 - alpha_bar) * noise
        
        # 3. Predict Noise
        pred_noise = self.net(noisy_action, t, observation)
        
        return nn.MSELoss()(pred_noise, noise)
        
    def sample(self, observation):
        # Reverse Process (DDPM)
        batch_size = observation.shape[0]
        action = torch.randn((batch_size, 2)).to(observation.device) # Action Dim 2
        
        for t in reversed(range(self.T)):
            time_tensor = torch.full((batch_size,), t, device=observation.device, dtype=torch.long)
            
            # Predict Noise
            with torch.no_grad():
                pred_noise = self.net(action, time_tensor, observation)
            
            # Remove Noise step
            alpha = self.alphas.to(observation.device)[t]
            alpha_bar = self.alphas_cumprod.to(observation.device)[t]
            beta = self.betas.to(observation.device)[t]
            
            if t > 0:
                noise = torch.randn_like(action)
            else:
                noise = 0
                
            action = (1 / torch.sqrt(alpha)) * (action - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * pred_noise) + torch.sqrt(beta) * noise
            
        return action
```

---

## 🔬 Lab Exercise: "The Y-Intersection"

### 1. Lab Objectives
- **Data Generation:** Create a synthetic dataset of 2D points.
    *   Start at (0,0).
    *   Obstacle at (0, 0.5).
    *   Experts go to (-1, 1) [Left] or (1, 1) [Right].
- **Train:** Train Diffusion Policy.
- **Inference:** Generate 100 samples from (0,0).
- **Result:**
    *   Scatter plot shows points clustering in Left and Right branches.
    *   **Crucial:** No points in the middle (Collision).
    *   **Comparison:** Train an MLP (MSE Loss) on the same data. It will output (0, 1) - Straight into the obstacle.

---

## 🚀 Project: "Consistency Distillation"

**Goal:** Speed up inference.
1.  **Problem:** Diffusion takes 100 steps ($T=100$). Too slow for 50Hz control.
2.  **DDIM:** Denoising Diffusion Implicit Models (Faster sampling, skip steps).
3.  **Distillation:** Train a student network to predict the result of 10 diffusion steps in 1 step.
4.  **Result:** 100Hz Inference capability.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Mode Averaging"
*   **Symptom:** Policy still averages modes.
*   **Cause:** Not enough Noise steps ($T$). Or Network too small.
*   **Fix:** Increase $T$ to 1000. Increase Model Capacity.

#### 2. "OOD Observations"
*   **Symptom:** Weird actions when camera view changes.
*   **Cause:** Visual Encoder backbone is brittle.
*   **Fix:** Use Data Augmentation (Crop/Color Jitter) on the visual observations during training.

---

## ⚡ Optimization: Receding Horizon Control

Action Sequence $A_{t:t+H}$.
*   At time $t$, we generate 16 steps.
*   We execute step 0 $A_t$.
*   At time $t+1$, we generate new 16 steps.
*   **Smoothing:** We can use the previous prediction as a "Warm Start" or Inpainting constraint for the new generation.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not use GANs?
    *   **A:** GANs suffer from Mode Collapse (They might only learn the Left Turn and ignore the Right Turn). Diffusion covers the distribution better.
2.  **Q:** What is the "Forward Process"?
    *   **A:** Gradually destroying data by adding Gaussian noise until it is pure noise.
3.  **Q:** Is Diffusion slow?
    *   **A:** Yes. Iterative generation is the bottleneck. But for complex tasks, the performance gain is worth it.

### Challenge Task
> **Task:** Conditional Generation.
> 1. Add a "Goal" input (Left or Right).
> 2. Train Conditional Diffusion $P(A|O, Goal)$.
> 3. Now you can *for* the mode you want.

---

## 📚 Further Reading
- **Diffusion Policy Paper (Chi et al., 2023):** "Visuomotor Policy Learning via Action Diffusion".
- **HuggingFace Diffusers:** Library for implementation.

---

**Day 82 Complete**
