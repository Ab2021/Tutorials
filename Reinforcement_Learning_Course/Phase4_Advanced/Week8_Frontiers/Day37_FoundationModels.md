# Day 37: Foundation Models for Decision Making

## 1. What are Foundation Models?
**Foundation Models** are large-scale models pretrained on massive datasets:
*   **Language:** GPT-4, Claude, LLaMA.
*   **Vision:** CLIP, DALL-E.
*   **Multi-Modal:** Flamingo, PaLM-E.

## 2. Foundation Models + RL
### Vision-Language-Action (VLA) Models
Combine vision, language understanding, and action prediction:
*   **RT-2 (Robotic Transformer 2):** Google's robot control model.
    *   Pretrained on web data (vision-language).
    *   Fine-tuned on robot demonstrations.
    *   Can follow natural language instructions.

### PaLM-E: Embodied Multimodal Language Model
*   588B parameters trained on vision, language, and robotics data.
*   Can plan, reason, and control robots using natural language.

## 3. LLMs as Planners
Use LLMs to generate high-level plans:
*   **SayCan (Google):** LLM generates task plans, RL policy executes low-level actions.
*   **Code as Policies:** LLM writes Python code to control robots.
*   **Inner Monologue:** LLM reasons about failures and replans.

## 4. Reward Design with LLMs
*   **Eureka (NVIDIA):** LLM generates reward functions for RL tasks.
    *   Achieves better performance than hand-designed rewards.
    *   Enables teaching complex skills (pen spinning) to robot hands.

## 5. Hierarchical Control
*   **High Level:** LLM/Foundation Model decides "what" to do.
*   **Low Level:** RL policy decides "how" to do it.
*   **Advantages:** Leverages language understanding and RL's fine motor control.

## 6. Example: RT-2 Architecture
```
Natural Language Instruction: "Pick up the blue ball"
                    ↓
         [Vision Encoder (CLIP)]
                    ↓
         [Language Encoder (T5)]
                    ↓
         [Transformer Backbone]
                    ↓
         [Action Decoder]
                    ↓
    Robot Actions: (x, y, z, gripper, rotation)
```

### Key Takeaways
*   Foundation Models bring world knowledge and language understanding to RL.
*   VLA models enable robots to follow natural language instructions.
*   LLMs can generate plans, rewards, and code for RL tasks.
*   Hierarchical approaches combine strengths of both paradigms.
