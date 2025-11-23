# Day 104: AGI, Superalignment & Ethics
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Weak-to-Strong Generalization (OpenAI)

Experiment:
1.  Train a **Weak Model** (GPT-2).
2.  Use Weak Model to generate labels.
3.  Train a **Strong Model** (GPT-4) on those weak labels.
4.  **Result:** The Strong Model often outperforms the Weak Supervisor. It "generalizes" beyond the noisy supervision.
*   **Implication:** We might be able to align ASI using human feedback, even if we don't understand what the ASI is doing.

### Mechanistic Interpretability

Opening the black box.
*   **Sparse Autoencoders:** Finding "features" in the neuron activations (e.g., a "Golden Gate Bridge" neuron).
*   **Linear Probes:** Checking if the model "knows" it is lying.
*   **Goal:** Detecting deception at the neuron level.

### Constitutional AI (Anthropic)

1.  **Constitution:** A set of principles (UN Declaration of Human Rights).
2.  **SFT:** Train model to critique itself based on principles.
3.  **RLAIF:** Reinforcement Learning from AI Feedback. The AI generates preferences based on the Constitution.

### Summary

*   **Interpretability** is our best hope for verification.
*   **RLAIF** scales alignment beyond human labeling speed.
