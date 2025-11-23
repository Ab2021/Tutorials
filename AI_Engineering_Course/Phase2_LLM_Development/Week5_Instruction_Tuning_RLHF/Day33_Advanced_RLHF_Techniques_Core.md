# Day 33: Advanced RLHF Techniques
## Core Concepts & Theory

### Beyond Basic RLHF

Standard RLHF (single RM, single PPO run) is just the beginning.
Production systems use advanced techniques for better performance, safety, and efficiency.

### 1. Iterative RLHF

**Concept:** Continuously improve the model through multiple RLHF cycles.

**Process:**
1. **Cycle 1:** Train RM on initial preferences. Run PPO. Get Model v1.
2. **Collect New Data:** Deploy Model v1. Collect new user interactions and preferences.
3. **Cycle 2:** Train new RM on combined data (old + new). Run PPO. Get Model v2.
4. **Repeat:** Continue for 3-5 cycles.

**Benefits:**
- **Continuous Improvement:** Each cycle addresses weaknesses discovered in the previous one.
- **Distribution Shift:** The RM is trained on data from the current policy, reducing overfitting.
- **Emergent Capabilities:** Models discover new strategies through exploration.

**Example:**
- **GPT-4:** Went through multiple RLHF cycles over 6+ months.
- **Claude-3:** Uses continuous RLHF with weekly updates.

### 2. Multi-Objective RLHF

**Problem:** A single reward signal cannot capture all desired behaviors.
- **Helpfulness** vs **Conciseness** (trade-off).
- **Creativity** vs **Factual Accuracy** (trade-off).

**Solution:** Train multiple Reward Models, one for each objective.

**Objectives:**
- $R_1$: Helpfulness
- $R_2$: Harmlessness
- $R_3$: Honesty
- $R_4$: Conciseness

**Combined Reward:**
$$ R_{total} = w_1 R_1 + w_2 R_2 + w_3 R_3 + w_4 R_4 $$

**Pareto Optimization:**
Instead of a weighted sum, find the Pareto frontier (solutions where improving one objective hurts another).

### 3. Process Supervision (PRM)

**Outcome Supervision (Standard RLHF):**
- Reward is based on the final answer.
- **Problem:** Doesn't reward correct reasoning if the final answer is wrong (or vice versa).

**Process Supervision:**
- Reward each step of the reasoning process.
- **Example (Math):**
  - Step 1: "Let x = 5" ✓ (Correct)
  - Step 2: "x + 3 = 9" ✗ (Incorrect, should be 8)
  - Step 3: "Therefore, x = 6" ✗ (Incorrect)
- **Reward:** +1 for Step 1, -1 for Steps 2 and 3.

**Benefits:**
- **Better Generalization:** Model learns correct reasoning, not just pattern matching.
- **Interpretability:** Can inspect which steps are rewarded.

**Challenges:**
- **Labeling Cost:** Requires humans to annotate every step (expensive).
- **Granularity:** How to define a "step"?

### 4. Best-of-N Sampling

**Concept:** Generate $N$ responses, score them with the RM, return the best one.

**Algorithm:**
1. Sample $N$ responses: $y_1, \dots, y_N \sim \pi(y|x)$.
2. Score each: $R(x, y_i)$ using the Reward Model.
3. Return: $y^* = \arg\max_i R(x, y_i)$.

**Benefits:**
- **No Retraining:** Uses the existing policy and RM.
- **Immediate Improvement:** Can boost performance by 10-20% (N=16).

**Costs:**
- **Latency:** Generating 16 responses is 16x slower.
- **Compute:** 16x more expensive.

**Practical Use:**
- **Critical Queries:** Use Best-of-16 for important user queries.
- **Standard Queries:** Use Best-of-1 (greedy) for speed.

### 5. Reward Model Ensembles

**Problem:** A single RM can be biased or overfit.

**Solution:** Train multiple RMs (3-5) with different:
- Random seeds.
- Architectures (GPT-2, LLaMA, Mistral).
- Training data subsets.

**Aggregation:**
- **Mean:** $R_{ensemble} = \frac{1}{K} \sum_{k=1}^K R_k(x, y)$.
- **Weighted:** $R_{ensemble} = \sum_{k=1}^K w_k R_k(x, y)$ where $w_k$ is based on validation accuracy.
- **Uncertainty:** Use variance across RMs as a measure of uncertainty. High variance = uncertain prediction.

### 6. Reward Shaping

**Problem:** Sparse rewards (only at the end) make RL slow.

**Solution:** Add intermediate rewards.

**Example (Coding):**
- **Sparse:** +1 if code passes all tests, 0 otherwise.
- **Shaped:**
  - +0.1 for correct syntax.
  - +0.2 for passing 50% of tests.
  - +0.3 for passing 80% of tests.
  - +0.4 for passing 100% of tests.

**Caution:** Poorly designed shaped rewards can lead to reward hacking.

### 7. Offline RL (Conservative Q-Learning)

**Problem:** PPO requires online interaction (generating new responses during training).

**Offline RL:** Train only on a fixed dataset of (prompt, response, reward) tuples.

**Benefits:**
- **Safety:** No risk of the model generating harmful content during training.
- **Efficiency:** Can reuse data from previous runs.

**Algorithms:**
- **CQL (Conservative Q-Learning):** Penalizes out-of-distribution actions.
- **IQL (Implicit Q-Learning):** Avoids explicit policy optimization.

### Summary Table

| Technique | Goal | Complexity | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Iterative RLHF** | Continuous Improvement | High | +20-30% |
| **Multi-Objective** | Balance Trade-offs | Medium | +10-15% |
| **Process Supervision** | Better Reasoning | Very High | +15-25% (Math/Code) |
| **Best-of-N** | Inference-Time Boost | Low | +10-20% |
| **RM Ensemble** | Robustness | Low | +5-10% |
| **Offline RL** | Safety | Medium | Comparable to PPO |

### Real-World Examples

**OpenAI (GPT-4):**
- Uses Iterative RLHF (multiple cycles).
- Uses Process Supervision for math and coding.
- Uses Multi-Objective RM (Helpfulness, Harmlessness, Honesty).

**Anthropic (Claude-3):**
- Uses Constitutional AI (a form of Multi-Objective).
- Uses Iterative RLHF with weekly updates.

**Google (Gemini):**
- Uses Best-of-N sampling for critical queries.
- Uses RM Ensembles for robustness.

### Next Steps
In the Deep Dive, we will implement Process Supervision for a math problem solver and analyze the Pareto frontier for Multi-Objective RLHF.
