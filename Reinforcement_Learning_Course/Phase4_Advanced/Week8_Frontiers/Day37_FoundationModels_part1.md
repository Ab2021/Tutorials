# Day 37 Deep Dive: Embodied AI and Open-Source Models

## 1. Open-Source Foundation Models for Robotics
*   **OpenVLA:** Open-source Vision-Language-Action model.
*   **Octo:** General-purpose robot policy (trained on 800k trajectories).
*   **RT-1 & RT-2 Open Datasets:** Google released robot manipulation data.

## 2. Sim-to-Real with Foundation Models
*   **Pretrain on Internet Data:** Vision-language understanding from web.
*   **Fine-Tune in Simulation:** Domain adaptation to robotics.
*   **Transfer to Real Robot:** Zero-shot or few-shot transfer.

## 3. Multimodal Perception
Foundation models excel at understanding rich sensory input:
*   **Vision:** CLIP embeddings for object recognition.
*   **Language:** Task descriptions, user instructions.
*   **Proprioception:** Robot joint angles, forces.
*   **Audio:** Environmental sounds, speech commands.

## 4. Open Problems
*   **Data Efficiency:** Still needs large robot datasets.
*   **Generalization:** Brittle to out-of-distribution scenarios.
*   **Real-Time Control:** Foundation models are slow for low-level control.
*   **Safety:** LLM hallucinations can lead to unsafe actions.
