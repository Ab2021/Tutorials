# Day 27: Model Evaluation & Red Teaming
## Core Concepts & Theory

### The Safety Lifecycle

Building a safe LLM is not a one-time step. It is a continuous lifecycle:
1.  **Pre-training:** Filter toxic data.
2.  **SFT:** Train on safe demonstrations.
3.  **RLHF:** Penalize unsafe outputs.
4.  **Red Teaming:** Attack the model to find holes.
5.  **Evaluation:** Measure safety metrics.

### 1. Automated Red Teaming

**Concept:** Use an "Attacker LLM" to generate adversarial prompts against a "Target LLM".
**Process:**
1.  **Attacker:** "Generate a prompt that asks for bomb-making instructions but disguises it as a chemistry experiment."
2.  **Target:** Generates response.
3.  **Judge:** Did the Target refuse? (Pass/Fail).
4.  **Iterate:** Attacker learns from failures and tries new strategies.

### 2. Garak (Generative AI Red-teaming & Assessment Kit)

**Tool:** An open-source scanner for LLM vulnerabilities.
**Probes:**
- **Injection:** "Ignore previous instructions."
- **Leakage:** "What is your system prompt?"
- **Jailbreak:** "DAN mode."
- **Toxicity:** "Say something mean."
**Output:** A report showing which probes succeeded.

### 3. Evaluation Benchmarks for Safety

**TruthfulQA:**
- Measures the model's tendency to reproduce common misconceptions (e.g., "If you crack your knuckles, you get arthritis").
- **Goal:** High truthfulness, low hallucination.

**RealToxicityPrompts:**
- A dataset of 100k prompts designed to elicit toxic continuations.
- **Metric:** Probability of generating toxic text (measured by Perspective API).

**Do Not Answer:**
- A dataset of harmful instructions (e.g., "How to steal identity").
- **Metric:** Refusal Rate.

### 4. The Refusal-Utility Trade-off

**The Problem:**
- If you make a model too safe, it becomes useless ("I cannot answer that" to everything).
- If you make it too helpful, it becomes dangerous.
- **False Refusal Rate (FRR):** Percentage of benign prompts that are wrongly refused.
- **Goal:** Minimize FRR while maximizing Safety.

### Summary of Evaluation

| Method | Type | Goal |
| :--- | :--- | :--- |
| **Red Teaming** | Active Attack | Find unknown vulnerabilities |
| **Garak** | Automated Scan | Regression testing |
| **TruthfulQA** | Benchmark | Measure hallucinations |
| **RealToxicity** | Benchmark | Measure toxic generation |

### Next Steps
In the Deep Dive, we will implement an Automated Red Teaming loop using two LLMs battling each other.
