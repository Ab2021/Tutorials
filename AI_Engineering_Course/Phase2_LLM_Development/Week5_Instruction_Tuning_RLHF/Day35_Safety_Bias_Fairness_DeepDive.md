# Day 35: Safety, Bias, and Fairness in LLMs
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. The Impossibility Theorem of Fairness

**Theorem (Chouldechova, 2017):**
For any classifier with imperfect accuracy, it is impossible to simultaneously satisfy:
1. **Demographic Parity:** $P(\hat{Y}=1|A=0) = P(\hat{Y}=1|A=1)$
2. **Equalized Odds:** $P(\hat{Y}=1|Y=y,A=0) = P(\hat{Y}=1|Y=y,A=1)$ for all $y$
3. **Calibration:** $P(Y=1|\hat{Y}=p,A=0) = P(Y=1|\hat{Y}=p,A=1)$ for all $p$

**Implication:**
You must choose which fairness criterion to prioritize based on the application.

**Example:**
- **Hiring:** Equalized Odds (equal TPR/FPR across demographics).
- **Lending:** Calibration (predicted probability matches true default rate).
- **Content Moderation:** Demographic Parity (equal moderation rates).

### 2. Bias Amplification

**Phenomenon:**
Even if training data has mild bias, the model can amplify it.

**Mechanism:**
- Training data: 60% of CEOs are male.
- Model learns: "CEO" strongly correlates with "male".
- Generation: 90% of generated CEO descriptions use male pronouns.

**Measurement:**
$$ \text{Amplification Factor} = \frac{P_{model}(\text{male}|\text{CEO})}{P_{data}(\text{male}|\text{CEO})} $$
If > 1, the model amplifies bias.

### 3. Counterfactual Data Augmentation

**Concept:**
Create synthetic training examples by swapping demographic attributes.

**Example:**
- Original: "The nurse prepared the medication. She was very careful."
- Counterfactual: "The nurse prepared the medication. He was very careful."

**Process:**
1. Identify demographic mentions (gender, race, age).
2. Generate counterfactual by swapping attributes.
3. Add both original and counterfactual to training data.

**Result:**
Model learns that profession is independent of demographics.

### 4. Adversarial Debiasing

**Concept:**
Train the model to be unable to predict sensitive attributes from its representations.

**Architecture:**
- **Main Model:** Predicts the task (e.g., sentiment).
- **Adversary:** Tries to predict sensitive attribute (e.g., gender) from hidden states.
- **Objective:** Maximize task performance while minimizing adversary's accuracy.

**Loss:**
$$ L = L_{task} - \lambda L_{adversary} $$

**Result:**
Representations are "blind" to sensitive attributes.

### 5. Toxicity Mitigation: DAPT (Detoxifying with Adversarial Prompts)

**Problem:**
Models can generate toxic text when prompted adversarially.

**DAPT Algorithm:**
1. **Generate Adversarial Prompts:** Use a red team model to generate prompts that elicit toxic responses.
2. **Generate Responses:** Sample responses from the target model.
3. **Filter:** Keep only non-toxic responses (Perspective API < 0.5).
4. **Fine-Tune:** Train the model on (adversarial prompt, non-toxic response) pairs.

**Result:**
Model becomes more robust to adversarial prompts.

### 6. Memorization and Privacy

**Canary Extraction Attack:**
- Insert a unique "canary" string into training data (e.g., "My SSN is 123-45-6789").
- Prompt the model to complete: "My SSN is..."
- If it outputs the canary, the model memorized it.

**Mitigation:**
- **Differential Privacy:** Add noise during training to prevent memorization.
- **Data Deduplication:** Remove repeated sequences from training data.
- **Output Filtering:** Detect and redact PII in outputs.

### Code: Bias Detection Pipeline

```python
from transformers import pipeline
import numpy as np

def measure_gender_bias(model, tokenizer, professions):
    """
    Measure gender bias in profession associations.
    """
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    results = {}
    
    for profession in professions:
        # Generate completions
        prompt = f"The {profession}"
        outputs = generator(prompt, max_length=20, num_return_sequences=100, do_sample=True)
        
        # Count gender pronouns
        male_count = 0
        female_count = 0
        
        for output in outputs:
            text = output['generated_text'].lower()
            if ' he ' in text or ' his ' in text or ' him ' in text:
                male_count += 1
            if ' she ' in text or ' her ' in text or ' hers ' in text:
                female_count += 1
        
        # Calculate bias
        total = male_count + female_count
        if total > 0:
            male_ratio = male_count / total
            results[profession] = {
                'male_ratio': male_ratio,
                'female_ratio': 1 - male_ratio,
                'bias_score': abs(male_ratio - 0.5)  # 0 = no bias, 0.5 = maximum bias
            }
    
    return results

# Example Usage
professions = ['nurse', 'engineer', 'teacher', 'CEO', 'secretary', 'pilot']
bias_results = measure_gender_bias(model, tokenizer, professions)

for prof, scores in bias_results.items():
    print(f"{prof}: Male={scores['male_ratio']:.2%}, Bias Score={scores['bias_score']:.2f}")
```

### 7. Red Teaming for Safety

**Systematic Red Teaming Protocol:**

**Phase 1: Threat Modeling**
- Identify potential harms (toxicity, bias, misinformation, jailbreaks).
- Prioritize by severity and likelihood.

**Phase 2: Attack Generation**
- Manual: Hire diverse red teamers.
- Automated: Use adversarial models to generate attacks.

**Phase 3: Evaluation**
- Measure Attack Success Rate (ASR).
- Categorize successful attacks by type.

**Phase 4: Mitigation**
- Add successful attacks to training data (with safe responses).
- Update safety filters.
- Retrain with RLHF.

**Phase 5: Iteration**
- Retest. ASR should decrease.
- Repeat until ASR < 5%.

### 8. Fairness-Accuracy Trade-off

**Observation:**
Enforcing strict fairness constraints often reduces overall accuracy.

**Example:**
- Unconstrained model: 90% accuracy, but biased.
- Fair model (Demographic Parity): 85% accuracy, unbiased.

**Pareto Frontier:**
Plot accuracy vs. fairness. Find the optimal trade-off point.

**Decision:**
Depends on the application. In high-stakes domains (hiring, lending), fairness may be prioritized over accuracy.
