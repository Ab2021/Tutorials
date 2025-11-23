# Day 29: Instruction Tuning Datasets
## Core Concepts & Theory

### The Data is the Model

In SFT, the dataset quality matters more than the model size.
**LIMA (Less Is More for Alignment):** 1,000 carefully curated examples outperform 100,000 noisy ones.

### 1. Classic Instruction Datasets

**A. Alpaca (Stanford):**
- **Size:** 52k instruction-response pairs.
- **Method:** Self-Instruct. Use GPT-3.5 to generate synthetic instructions from seed tasks.
- **Format:** `{"instruction": "...", "input": "...", "output": "..."}`
- **Limitation:** Synthetic data has "GPT-isms" (overly polite, verbose).

**B. ShareGPT:**
- **Size:** ~90k real conversations scraped from users sharing ChatGPT logs.
- **Quality:** High (real user queries), but contains PII and toxic content.
- **Format:** Multi-turn conversations.

**C. Dolly (Databricks):**
- **Size:** 15k human-written instruction-response pairs.
- **Method:** Crowdsourcing (employees).
- **License:** Open (CC-BY-SA).

### 2. Evol-Instruct (WizardLM)

**Concept:** Iteratively make instructions more complex using an LLM.
**Algorithm:**
1. Start with simple instruction: "List fruits."
2. **Evolve:** "List 10 exotic fruits and explain their health benefits."
3. **Evolve:** "Compare 10 exotic fruits, rank by antioxidant content, cite studies."
**Result:** Creates challenging, diverse instructions that push the model's reasoning.

### 3. Data Quality Metrics

**A. Instruction Complexity:**
- Measured by: Verb diversity, sentence length, dependency depth.
- Simple: "What is X?"
- Complex: "Compare X and Y, then propose Z."

**B. Response Quality:**
- Factual Accuracy (NLI check against Wikipedia).
- Coherence (Perplexity).
- Safety (Toxicity score).

### 4. The Contamination Problem

**Issue:** If your instruction dataset contains test set questions from MMLU or HumanEval, your evaluation is invalid.
**Detection:** N-gram overlap, embedding similarity.
**Solution:** Deduplication against all known benchmarks before training.

### Summary of Datasets

| Dataset | Size | Source | Quality |
| :--- | :--- | :--- | :--- |
| **Alpaca** | 52k | Synthetic (GPT-3.5) | Medium |
| **ShareGPT** | 90k | Real Users | High (needs cleaning) |
| **Dolly** | 15k | Human Crowdsourced | High |
| **Evol-Instruct** | 250k | Evolved (GPT-4) | Very High |
| **LIMA** | 1k | Curated | Extremely High |

### Next Steps
In the Deep Dive, we will implement the Evol-Instruct algorithm to generate complex instructions from simple seeds.
