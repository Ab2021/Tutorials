# Day 15: Scaling Laws & Model Size
## Core Concepts & Theory

### The Era of Large Language Models

Before 2020, AI progress was driven by architectural innovations (LSTM -> Transformer -> BERT).
After 2020, progress has been largely driven by **Scale**.
- **Compute:** More FLOPs used for training.
- **Data:** More tokens in the training set.
- **Parameters:** Larger models.

### 1. The Power Law of Scaling

Kaplan et al. (OpenAI, 2020) discovered that model performance (test loss) scales as a power law with respect to compute, dataset size, and parameters.

**The Formula:**
$$ L(C) \propto C^{-\alpha} $$
Where $L$ is the loss and $C$ is the compute budget.
This implies that **diminishing returns are predictable**. To reduce loss linearly, you need to increase compute exponentially.

### 2. The Chinchilla Scaling Laws (Hoffmann et al., 2022)

Kaplan's paper suggested that model size matters most.
DeepMind's Chinchilla paper corrected this: **Data matters just as much as size.**

**Key Finding:**
For a compute-optimal model, the number of parameters ($N$) and the number of training tokens ($D$) should scale equally.
$$ N_{opt} \propto C^{0.5} $$
$$ D_{opt} \propto C^{0.5} $$

**The Golden Ratio:**
For every parameter, you should train on roughly **20 tokens**.
- **Kaplan (Old):** Train huge models on small data (e.g., GPT-3: 175B params on 300B tokens). Ratio $\approx$ 1.7.
- **Chinchilla (New):** Train smaller models on more data (e.g., LLaMA: 65B params on 1.4T tokens). Ratio $\approx$ 21.

**Implication:**
Most older models (GPT-3, Megatron-Turing) were **severely undertrained**. Modern models (LLaMA, Mistral) are "compute-optimal" or even "over-trained" (inference-optimal).

### 3. Emergent Abilities

As models scale, they don't just get "better" at the same things; they suddenly acquire **new capabilities** that were near-zero in smaller models.
- **Few-Shot Arithmetic:** Emerges around 10B-100B params.
- **Code Generation:** Emerges with scale + code data.
- **Reasoning (Chain of Thought):** Emerges around 60B+ params.

**Why?**
The "Phase Transition" hypothesis: The model learns simple heuristics at small scales, but learns complex algorithms (circuits) once it has enough capacity and data.

### 4. Compute-Optimal vs. Inference-Optimal

**Compute-Optimal (Chinchilla):** Best model you can train for a given training budget ($C$).
- Result: Large model, medium data.
- Best for: Research, one-off training.

**Inference-Optimal (LLaMA):** Best model for a given inference latency/cost constraint.
- Result: Smaller model, massive data (train way beyond Chinchilla point).
- **Why?** Inference cost depends only on model size ($N$). If you train a smaller model longer, you pay more upfront (training) but save forever on inference.
- LLaMA-3 (8B) trained on 15T tokens (Ratio: 1875 tokens/param!) is extremely inference-optimal.

### Summary of Scaling

| Law | Focus | Recommendation | Example |
| :--- | :--- | :--- | :--- |
| **Kaplan** | Training Efficiency | Scale Params > Data | GPT-3 |
| **Chinchilla** | Compute Optimality | Scale Params $\approx$ Data (20:1) | Gopher |
| **LLaMA** | Inference Efficiency | Scale Data >>> Params | LLaMA-3 |

### Next Steps
In the Deep Dive, we will derive the Chinchilla optimal point and analyze the economics of training large models.
