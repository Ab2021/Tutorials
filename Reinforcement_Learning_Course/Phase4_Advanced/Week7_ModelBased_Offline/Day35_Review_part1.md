# Day 35 Deep Dive: Deployment Case Studies

## 1. Google Data Center Cooling
**Problem:** Optimize cooling systems to reduce energy consumption.
**Solution:** Model-Based RL with safety constraints.
**Result:** 40% reduction in cooling energy, 15% reduction in overall PUE.
**Key Techniques:** Offline pretraining, conservative updates, human override.

## 2. Waymo Autonomous Driving
**Approach:**
*   Train in high-fidelity simulation (billions of miles).
*   Use offline RL from human driving data.
*   Validate in controlled real-world testing.
*   Gradual deployment with safety drivers.

## 3. OpenAI's Rubik's Cube Solving Robot Hand
**Challenge:** Dexterous manipulation in the real world.
**Approach:**
*   Train in simulation with domain randomization.
*   Use Asymmetric Actor-Critic (privileged information in sim).
*   Transfer to real robot without fine-tuning.

## 4. AlphaStar (StarCraft II)
**Techniques:**
*   Multi-agent self-play.
*   League training (population of diverse agents).
*   Supervised learning from human replays + RL.
*   Successfully deployed against top human pros.
