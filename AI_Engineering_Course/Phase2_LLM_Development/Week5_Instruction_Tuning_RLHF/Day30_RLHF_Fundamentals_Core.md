# Day 30: RLHF Fundamentals
## Core Concepts & Theory

### The Alignment Problem

**The Gap Between Language Modeling and Helpfulness:**
Pre-trained LLMs are trained to predict the next token. This makes them excellent at completing text, but not necessarily at being helpful, harmless, and honest.
- **Example:** Prompt: "How to make a bomb?"
- **Base Model:** Continues with instructions (it's just predicting likely text from the internet).
- **Aligned Model:** "I cannot help with that."

**SFT Limitations:**
Supervised Fine-Tuning teaches the model *what* to say (format, style) by showing examples.
But it doesn't teach the model *how good* the answer is or how to choose between multiple valid responses.
- **Problem:** Given "Explain quantum physics", there are infinite valid responses. Which one is best?
- **Solution:** RLHF (Reinforcement Learning from Human Feedback).

### 1. The RLHF Pipeline (InstructGPT/ChatGPT)

**Historical Context:**
- **2020:** GPT-3 (Base Model) - Powerful but unaligned.
- **2022:** InstructGPT - First major RLHF application (OpenAI).
- **2022:** ChatGPT - RLHF on steroids.
- **2023:** GPT-4 - Extensive RLHF + Constitutional AI.

**Three Stages:**

**Stage 1: Supervised Fine-Tuning (SFT)**
- **Data:** 10k-100k high-quality (Prompt, Response) pairs written by human labelers.
- **Process:** Standard next-token prediction (CLM) with masking on prompts.
- **Output:** Creates a baseline policy $\pi_{SFT}$ that can follow instructions.
- **Quality:** Better than base model, but still makes mistakes, can be verbose, or miss nuances.

**Stage 2: Reward Model (RM) Training**
- **Data Collection:** 
  - Sample prompts from a distribution (user queries, synthetic).
  - Generate 4-9 responses using $\pi_{SFT}$ with different temperatures.
  - Human labelers rank the responses: Best to Worst.
- **Model:** Take the SFT model, remove the language modeling head, add a scalar output head.
- **Training:** Predict pairwise preferences: $P(A > B | Prompt, A, B)$.
- **Loss:** Ranking loss (Bradley-Terry model).
$$ L = -\log \sigma(R(x, y_w) - R(x, y_l)) $$
where $y_w$ is the preferred (winner) response, $y_l$ is the rejected (loser) response.
- **Dataset Size:** InstructGPT used ~33k comparisons. ChatGPT likely used 100k+.

**Stage 3: RL Fine-tuning (PPO)**
- **Environment:** Sample a prompt $x$ from the dataset.
- **Action:** Generate a response $y$ using the current policy $\pi$.
- **Reward:** $R(x, y)$ from the Reward Model.
- **Penalty:** KL divergence from the reference policy (frozen SFT model).
$$ R_{total}(x, y) = R_{RM}(x, y) - \beta \cdot KL(\pi(y|x) || \pi_{ref}(y|x)) $$
- **Optimization:** Use PPO to update $\pi$ to maximize $\mathbb{E}[R_{total}]$.
- **Iterations:** Train for 10k-100k steps.

### 2. Preference Data Collection

**Why Preferences Instead of Ratings?**
- **Easier for Humans:** Comparing two answers is easier than assigning an absolute score.
- **More Reliable:** Reduces labeler bias and calibration issues.
- **Richer Signal:** A ranking of 4 responses gives $\binom{4}{2} = 6$ pairwise comparisons.

**Pairwise Comparison:**
- **Prompt:** "Write a poem about cats."
- **Response A:** "Cats are cute and fluffy. They like to play."
- **Response B:** "Whiskers dance in moonlit grace, / Silent paws in shadowed space. / Eyes that gleam like emerald fire, / Feline hearts that never tire."
- **Human:** B > A (more creative, poetic, engaging).

**Ranking (K-wise):**
- Show 4 responses. Human ranks them: B > D > A > C.
- Convert to pairwise: (B>D), (B>A), (B>C), (D>A), (D>C), (A>C).

**Labeling Guidelines:**
OpenAI's guidelines for InstructGPT labelers:
1. **Helpfulness:** Does it follow the instruction? Is it useful?
2. **Truthfulness:** Is it factually accurate?
3. **Harmlessness:** Does it avoid toxic, biased, or dangerous content?

**Inter-Annotator Agreement:**
- Measured by Cohen's Kappa or Krippendorff's Alpha.
- InstructGPT reported ~73% agreement (good but not perfect).
- Disagreements are valuable: they reveal ambiguous cases.

### 3. The Reward Model

**Architecture:**
- **Base:** Same as the policy model (e.g., GPT-3 6.7B).
- **Modification:** Replace the LM head (vocab_size outputs) with a scalar head (1 output).
- **Pooling:** Use the last token's hidden state (or mean pooling) as the representation.

**Training Details:**
- **Input:** Concatenate Prompt + Response. Tokenize.
- **Forward:** Get scalar reward $R(x, y)$.
- **Loss:** For each comparison $(x, y_w, y_l)$:
$$ L = -\log \sigma(R(x, y_w) - R(x, y_l)) $$
- **Batch:** Process multiple comparisons per batch.
- **Optimizer:** AdamW, LR ~1e-5.
- **Epochs:** 1-3 (to avoid overfitting).

**Evaluation:**
- **Accuracy:** On held-out comparisons. InstructGPT RM: ~72% accuracy.
- **Calibration:** Does $P(A > B)$ match the true probability?

**Challenges:**
- **Overfitting:** RM can memorize the training distribution.
- **Distribution Shift:** During PPO, the policy generates new responses that the RM has never seen.
- **Reward Hacking:** The policy exploits RM weaknesses.

### 4. PPO (Proximal Policy Optimization)

**Why PPO?**
- **Stability:** Prevents large policy updates that could break the model.
- **Sample Efficiency:** Reuses data (unlike vanilla policy gradient).
- **Industry Standard:** Used by OpenAI, Anthropic, DeepMind.

**The Objective:**
$$ \max_\pi \mathbb{E}_{x \sim D, y \sim \pi} [R_{RM}(x, y) - \beta \cdot KL(\pi(y|x) || \pi_{ref}(y|x))] $$

**Components:**
1. **Reward:** $R_{RM}(x, y)$ from the Reward Model.
2. **KL Penalty:** $-\beta \cdot KL(\pi || \pi_{ref})$ keeps the model close to the SFT baseline.
   - **Why?** Without this, the model might drift into a degenerate mode (e.g., always outputting "I don't know" if that scores high).
3. **Reference Policy:** $\pi_{ref}$ is the frozen SFT model.

**PPO Algorithm (Simplified):**
1. **Sample:** Generate $N$ responses $y_1, \dots, y_N$ using the current policy $\pi$ for prompts $x_1, \dots, x_N$.
2. **Score:** Compute rewards $R(x_i, y_i)$ using the RM.
3. **Advantage:** Compute advantage $A_i = R_i - V(x_i)$ where $V$ is a value function (baseline).
4. **Update:** Use PPO clipped objective to update $\pi$.
5. **Repeat:** For 10k-100k iterations.

**Hyperparameters:**
- **KL Coefficient ($\beta$):** 0.01-0.1. Higher = more conservative.
- **Learning Rate:** 1e-6 to 1e-5.
- **Batch Size:** 64-512 prompts.
- **Epochs per Batch:** 4 (reuse data).

### 5. The Role of the Value Function

**Problem:** High variance in policy gradient.
**Solution:** Subtract a baseline (Value Function $V(x)$) to reduce variance.
$$ A(x, y) = R(x, y) - V(x) $$
**Training:** The Value Function is trained to predict the expected reward for a given prompt.
$$ L_V = \mathbb{E}[(V(x) - R(x, y))^2] $$

### 6. RLHF Variants

**RLAIF (RL from AI Feedback):**
- Use a strong LLM (GPT-4) as the judge instead of humans.
- **Benefit:** Cheaper, faster, scalable.
- **Drawback:** Inherits biases of the judge model.

**Constitutional AI (Anthropic):**
- Define a "Constitution" (set of principles).
- Use the model itself to critique and revise its outputs.
- Train on the revised outputs.

**Iterative RLHF:**
- After PPO, collect new preference data on the improved model.
- Train a new RM.
- Run PPO again.
- **Result:** Continuous improvement (used in GPT-4).

### 7. Production Considerations

**Cost:**
- **Labeling:** $1-5 per comparison. 100k comparisons = $100k-500k.
- **Compute:** PPO training on 175B model = $50k-200k (GPU hours).

**Timeline:**
- **SFT:** 1-3 days.
- **RM Training:** 1 day.
- **PPO:** 3-7 days.
- **Total:** 1-2 weeks for a full RLHF run.

**Failure Modes:**
- **Reward Hacking:** Model exploits RM bugs.
- **Mode Collapse:** Model always outputs safe, generic responses.
- **Sycophancy:** Model agrees with the user even when wrong.

### Summary Table

| Component | Input | Output | Training Data | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **SFT** | Prompt | Response | 10k-100k (Prompt, Response) | Baseline Instruction Following |
| **Reward Model** | (Prompt, Response) | Scalar Score | 30k-100k Comparisons | Preference Predictor |
| **PPO** | Prompt | Response | Prompts (no labels) | Optimize for Human Preference |

### Key Insights

1. **RLHF is Expensive:** Requires massive human labeling effort.
2. **RLHF is Fragile:** PPO can be unstable; hyperparameters matter.
3. **RLHF is Powerful:** Transforms a base model into a helpful assistant.
4. **RLHF is Iterative:** ChatGPT/GPT-4 went through multiple RLHF cycles.

### Real-World Example: InstructGPT

**Base Model:** GPT-3 (175B parameters).
**SFT Data:** 13k demonstrations written by 40 labelers.
**RM Data:** 33k comparisons.
**Result:** InstructGPT (1.3B) was preferred over GPT-3 (175B) by human evaluators.
**Key Takeaway:** Alignment matters more than scale.

### Next Steps
In the Deep Dive, we will derive the PPO clipped objective, analyze the Bradley-Terry model, and implement a complete Reward Model training loop in PyTorch.
