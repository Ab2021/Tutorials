# Day 35: Safety, Bias, and Fairness in LLMs
## Core Concepts & Theory

### The Safety Imperative

LLMs are powerful tools that can cause harm if misused or if they exhibit biased behavior.
**Safety** is not optional—it's a fundamental requirement for deployment.

### 1. Safety Dimensions

**Harmful Content Generation:**
- **Toxicity:** Hate speech, profanity, insults.
- **Violence:** Instructions for harm, weapons, explosives.
- **Illegal Activities:** Drug manufacturing, hacking, fraud.
- **Sexual Content:** Explicit or inappropriate sexual material.
- **Self-Harm:** Suicide instructions, self-injury encouragement.

**Misinformation:**
- **Hallucinations:** Generating false information confidently.
- **Medical Misinformation:** Dangerous health advice.
- **Financial Misinformation:** Bad investment advice.

**Privacy Violations:**
- **PII Leakage:** Revealing personal information from training data.
- **Memorization:** Reproducing copyrighted or private text verbatim.

### 2. Bias in LLMs

**Types of Bias:**

**Representation Bias:**
- Certain groups are underrepresented in training data.
- **Example:** "CEO" -> generates male names 90% of the time.

**Stereotyping:**
- Model associates professions, traits, or behaviors with specific demographics.
- **Example:** "Nurse" -> female, "Engineer" -> male.

**Allocative Harm:**
- Model makes decisions that disadvantage certain groups.
- **Example:** Resume screening model rejects female candidates.

**Quality-of-Service Disparity:**
- Model performs worse for certain demographics.
- **Example:** Speech recognition has higher error rates for non-native speakers.

### 3. Measuring Bias

**BOLD (Bias in Open-Ended Language Generation):**
- **Method:** Prompt the model with demographic descriptors (e.g., "The Black woman").
- **Metric:** Measure sentiment and toxicity of completions across groups.
- **Goal:** Equal sentiment and low toxicity for all groups.

**WinoBias:**
- **Method:** Coreference resolution with gender-stereotyped professions.
- **Example:** "The nurse asked the doctor because [she/he] needed help."
- **Metric:** Accuracy difference between stereotypical and anti-stereotypical cases.

**BBQ (Bias Benchmark for QA):**
- **Method:** Questions designed to elicit biased responses.
- **Example:** "Who is more likely to be a criminal, the Black man or the White man?"
- **Metric:** % of times the model gives a biased answer.

### 4. Mitigation Strategies

**Data Curation:**
- **Filtering:** Remove toxic and biased content from training data.
- **Balancing:** Ensure equal representation of demographics.
- **Augmentation:** Add counter-stereotypical examples.

**Fine-Tuning:**
- **SFT on Safe Data:** Train on curated, bias-free demonstrations.
- **RLHF with Safety RM:** Train a Reward Model to penalize biased or harmful outputs.

**Prompting:**
- **System Prompt:** "You are a helpful, harmless, and unbiased assistant."
- **Few-Shot:** Provide examples of safe, unbiased responses.

**Output Filtering:**
- **Perspective API:** Filter toxic outputs before showing to users.
- **Bias Classifiers:** Detect and block biased responses.

**Red Teaming:**
- Hire diverse testers to find bias and safety issues.
- Iterate on the model based on findings.

### 5. Fairness Metrics

**Demographic Parity:**
$$ P(\hat{Y}=1 | A=a) = P(\hat{Y}=1 | A=b) $$
Model predictions are independent of sensitive attribute $A$ (e.g., race, gender).

**Equalized Odds:**
$$ P(\hat{Y}=1 | Y=y, A=a) = P(\hat{Y}=1 | Y=y, A=b) $$
True positive and false positive rates are equal across groups.

**Calibration:**
$$ P(Y=1 | \hat{Y}=p, A=a) = P(Y=1 | \hat{Y}=p, A=b) $$
Predicted probabilities are equally accurate across groups.

**Trade-offs:**
- It's mathematically impossible to satisfy all fairness criteria simultaneously (except in trivial cases).
- Must choose which fairness definition is most appropriate for the use case.

### 6. Safety Layers

**Input Guardrails:**
- Detect and block harmful prompts before they reach the model.
- **Tools:** Llama Guard, NeMo Guardrails, Azure Content Safety.

**Output Guardrails:**
- Detect and block harmful outputs before showing to users.
- **Tools:** Perspective API, OpenAI Moderation API.

**Constitutional AI:**
- Bake safety principles into the model during training.
- More robust than post-hoc filtering.

### 7. Regulatory Landscape

**EU AI Act:**
- Classifies AI systems by risk (Unacceptable, High, Limited, Minimal).
- High-risk systems (e.g., hiring, credit scoring) require safety assessments.

**NIST AI Risk Management Framework:**
- Guidelines for managing AI risks.
- Emphasizes transparency, accountability, and fairness.

**Industry Standards:**
- **OpenAI:** Model cards, safety evaluations, red teaming.
- **Anthropic:** Constitutional AI, transparency reports.
- **Google:** Responsible AI practices, fairness indicators.

### Summary Table

| Safety Issue | Detection Method | Mitigation |
| :--- | :--- | :--- |
| **Toxicity** | Perspective API | RLHF, Output Filtering |
| **Bias** | BOLD, BBQ | Data Balancing, RLHF |
| **Hallucinations** | Fact-Checking | Grounding, RAG |
| **PII Leakage** | Regex, NER | Data Scrubbing, Output Filtering |
| **Jailbreaks** | Red Teaming | Constitutional AI, Input Guardrails |

### Real-World Examples

**OpenAI (GPT-4):**
- Extensive red teaming (50+ external experts).
- Multiple safety RMs (toxicity, bias, truthfulness).
- System message for safety ("You are a helpful assistant...").

**Anthropic (Claude):**
- Constitutional AI with 50+ principles.
- Harmlessness RM trained on 170k comparisons.
- Transparency reports on safety incidents.

**Meta (Llama-2):**
- Llama Guard (safety classifier).
- Safety-specific RLHF.
- Responsible use guide.

### Next Steps
In the Deep Dive, we will implement a bias detection pipeline and analyze the trade-offs between different fairness metrics.
