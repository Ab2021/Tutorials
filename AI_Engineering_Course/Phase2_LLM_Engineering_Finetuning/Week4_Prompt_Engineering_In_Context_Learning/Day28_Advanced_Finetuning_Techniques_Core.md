# Day 28: Advanced Fine-tuning Techniques
## Core Concepts & Theory

### Beyond Single-Task SFT

Standard SFT trains a model on a single dataset (e.g., Alpaca) to follow instructions.
However, real-world models need to handle multiple domains (Coding, Math, Chat) simultaneously without forgetting.

### 1. Multi-Task Learning (MTL)

**Concept:** Train the model on a mixture of datasets simultaneously.
**Mixture:**
- 50% General Chat (ShareGPT)
- 30% Code (StackOverflow/GitHub)
- 10% Math (GSM8K)
- 10% Logic (FLAN)
**Benefit:** The model learns shared representations. Learning code helps with logic; learning math helps with reasoning.
**Challenge:** **Task Interference**. If gradients from Task A conflict with Task B, the model learns neither well.

### 2. Domain Adaptation (Continued Pre-training)

**Scenario:** You want a Legal LLM.
**Method:**
1.  **Pre-training:** LLaMA-2 (General).
2.  **Domain Adaptation:** Continue pre-training (CLM) on 100GB of Legal Text (Case Law, Contracts).
3.  **SFT:** Fine-tune on Legal Q&A.
**Why not just SFT?** SFT is for *style*. Domain Adaptation is for *knowledge*. You can't teach a model Law just by showing it Q&A pairs; it needs to read the books first.

### 3. Task Arithmetic (Model Merging)

**Concept:** We can manipulate model weights like vectors.
**Formula:** $\theta_{new} = \theta_{base} + \lambda (\theta_{finetuned} - \theta_{base})$.
**Task Vector:** $\tau = \theta_{ft} - \theta_{base}$.
**Merging:**
- Train Model A on Code. $\tau_A$.
- Train Model B on Math. $\tau_B$.
- Combined Model: $\theta_{combined} = \theta_{base} + \tau_A + \tau_B$.
- **Result:** A model that is good at both Code and Math, without multi-task training!

### 4. NEFTune (Noisy Embedding Fine-Tuning)

**Concept:** Add random noise to the embedding vectors during fine-tuning.
**Benefit:** Prevents overfitting to the specific phrasing of the instruction dataset.
**Result:** Improves performance on AlpacaEval by ~10% (Jain et al., 2023).

### Summary of Techniques

| Technique | Goal | Method |
| :--- | :--- | :--- |
| **MTL** | Generalist Model | Mix datasets with sampling weights |
| **Domain Adapt** | New Knowledge | CLM on raw domain text |
| **Task Arithmetic** | Combine Skills | Add weight differences |
| **NEFTune** | Regularization | Add noise to embeddings |

### Next Steps
In the Deep Dive, we will implement Task Arithmetic to merge two LoRA adapters into a single model.
