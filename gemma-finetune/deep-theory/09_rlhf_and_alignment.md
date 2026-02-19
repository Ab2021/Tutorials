# 9. RLHF and Alignment — Making Models Helpful, Harmless, and Honest

## Table of Contents
- [What Is Alignment?](#what-is-alignment)
- [The Three Stages of LLM Training](#the-three-stages-of-llm-training)
- [Stage 1: Supervised Fine-Tuning (SFT)](#stage-1-supervised-fine-tuning-sft)
- [Stage 2: Reward Modeling](#stage-2-reward-modeling)
- [Stage 3: RLHF with PPO](#stage-3-rlhf-with-ppo)
- [DPO: Direct Preference Optimization](#dpo-direct-preference-optimization)
- [ORPO: Odds Ratio Preference Optimization](#orpo-odds-ratio-preference-optimization)
- [Constitutional AI (CAI)](#constitutional-ai-cai)
- [Comparison of Alignment Methods](#comparison-of-alignment-methods)
- [When to Use Each Method](#when-to-use-each-method)

---

## What Is Alignment?

### The Problem

```
Raw pre-trained model (before alignment):
  User: "How do I cook pasta?"
  Model: "I cook pasta the traditional way my grandmother
          taught me in Naples in 1983 when the sun was
          setting over the Mediterranean and the smell of
          basil filled the..."   ← rambling, not helpful

  User: "Write Python code for sorting"
  Model: "sorting in Python is a common task that many
          developers encounter daily in their professional
          life..."   ← doesn't write code, just talks about it

After alignment:
  User: "How do I cook pasta?"
  Model: "Here's a simple pasta recipe:
          1. Boil water with salt
          2. Add pasta, cook 8-10 minutes
          3. Drain and serve with sauce"   ← helpful and concise!
```

### The Alignment Triad

```
         HELPFUL
        ╱       ╲
       ╱   AI    ╲
      ╱  should   ╲
     ╱   be all    ╲
    ╱    three!     ╲
   ╱                 ╲
HARMLESS ─────────── HONEST

Helpful:   Follows instructions, provides useful information
Harmless:  Refuses harmful requests, doesn't produce dangerous content
Honest:    Doesn't hallucinate, admits uncertainty
```

---

## The Three Stages of LLM Training

```
Stage 1: PRE-TRAINING
  Data: Raw internet text (trillions of tokens)
  Goal: Learn language, facts, reasoning
  Method: Next-token prediction
  Result: Completion model (can generate text, but not aligned)

Stage 2: SUPERVISED FINE-TUNING (SFT) ← We stop here
  Data: (instruction, response) pairs (thousands)
  Goal: Learn to follow instructions
  Method: Standard fine-tuning (our project!)
  Result: Instruction-following model

Stage 3: ALIGNMENT (RLHF/DPO)
  Data: Human preference comparisons (which response is better?)
  Goal: Generate responses humans prefer
  Method: Reinforcement learning or preference optimization
  Result: Aligned model (helpful, harmless, honest)
```

---

## Stage 1: Supervised Fine-Tuning (SFT)

### What Our Project Does

```
Training data format:
  Input:  "Review: 'Great phone, amazing camera!' Rating: 5/5"
  Output: "Recommendation: BUY. Strengths: Camera quality..."

The model learns:
  1. To follow the instruction format
  2. To generate structured recommendations
  3. To base recommendations on review content

This is SFT — the same method ChatGPT, Claude, and Gemini use
as their foundation before RLHF.
```

### SFT Quality Depends On Data Quality

```
High-quality SFT data:
  ✅ Diverse instructions (many different review types)
  ✅ Consistent format (always the same structure)
  ✅ Correct labels (BUY for genuinely good products)
  ✅ Detailed responses (explain reasoning)

Low-quality SFT data:
  ❌ Repetitive (same phrasing over and over)
  ❌ Inconsistent format (sometimes structured, sometimes not)
  ❌ Wrong labels (BUY for terrible products)
  ❌ Shallow responses ("It's good. Buy it.")
```

---

## Stage 2: Reward Modeling

### The Problem with SFT

```
SFT teaches the model to MIMIC the training data.
But some responses are BETTER than others, and SFT can't capture this:

Response A: "This phone is good. Consider buying."
Response B: "Based on the 4.5 star rating and specific praise for
            battery life and camera quality, I recommend BUYING this
            phone. The only concern mentioned is price..."

Both are "correct" responses. SFT treats them equally.
But Response B is clearly BETTER.

We need a way to tell the model: "B is better than A"
```

### Training a Reward Model

```
Step 1: Generate multiple responses for the same prompt
  Prompt: "Review: 'Amazing battery life!' Rating: 5"
  
  Response A: "Buy it."                              (low quality)
  Response B: "Recommendation: BUY. Great battery."   (medium quality)
  Response C: "Recommendation: BUY. Key strength:     (high quality)
              battery life. Strategy: promote this
              feature in marketing."

Step 2: Human annotators RANK the responses
  C > B > A    (C is best, A is worst)

Step 3: Train a reward model R(prompt, response) → score
  R(prompt, A) = 0.2
  R(prompt, B) = 0.6
  R(prompt, C) = 0.9

The reward model learns what humans PREFER.
```

### Reward Model Architecture

```
Take any LLM (e.g., Gemma-2B) and replace the output head:
  Original: token → probability distribution (vocab_size outputs)
  Reward model: sequence → single scalar score

Loss function (Bradley-Terry model):
  L = -log(σ(R(prompt, y_w) - R(prompt, y_l)))
  
  Where y_w = preferred response, y_l = dispreferred response
  σ = sigmoid function
  
  "The reward for the preferred response should be HIGHER
   than for the dispreferred response."
```

---

## Stage 3: RLHF with PPO

### The Full Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  SFT Model   │    │ Reward Model │    │ Reference    │
│   (policy)   │    │   (judge)    │    │   Model      │
│              │    │              │    │ (unchanged   │
│  generates   │    │  scores      │    │  SFT model)  │
│  responses   │    │  responses   │    │              │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       ▼                   ▼                   ▼
   ┌───────────────────────────────────────────────┐
   │                PPO ALGORITHM                    │
   │                                               │
   │  For each prompt:                             │
   │  1. Generate response using policy model      │
   │  2. Score with reward model: r = R(prompt, y) │
   │  3. Compute KL penalty: D_KL(policy || ref)   │
   │  4. Total reward: r - β × D_KL               │
   │  5. Update policy to maximize total reward    │
   └───────────────────────────────────────────────┘
```

### Why KL Penalty?

```
Without KL penalty:
  The model learns to "hack" the reward model.
  It finds responses that score HIGH on the reward model
  but are actually NONSENSICAL to humans.
  
  Example: "AMAZING GREAT BEST PERFECT BUY BUY BUY!!!"
  This might score highly on reward model (positive words)
  but is useless to a human.

With KL penalty:
  KL divergence measures how much the policy DIVERGES from the reference.
  Large divergence → large penalty.
  
  The model must score highly on reward
  WHILE STAYING CLOSE to the original SFT model.
  This prevents reward hacking.
```

### PPO Algorithm (Simplified)

```python
# Pseudocode for PPO in RLHF
for batch in prompts:
    # 1. Generate responses
    responses = policy.generate(batch)
    
    # 2. Get reward scores
    rewards = reward_model(batch, responses)
    
    # 3. Get log probabilities from both models
    log_probs = policy.log_prob(responses)
    ref_log_probs = ref_model.log_prob(responses)
    
    # 4. KL penalty
    kl = log_probs - ref_log_probs
    total_reward = rewards - beta * kl
    
    # 5. PPO objective
    advantages = compute_advantages(total_reward)
    ratio = exp(log_probs_new - log_probs_old)
    clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
    loss = -min(ratio * advantages, clipped_ratio * advantages)
    
    # 6. Update policy
    loss.backward()
    optimizer.step()
```

### PPO Challenges

```
❌ Complex: Three models in memory (policy, reward, reference)
❌ Expensive: Generate responses at each step → slow
❌ Unstable: RL training can diverge easily
❌ Hyperparameter sensitive: β, ε, learning rate all critical
❌ Reward hacking: Model can still find exploits
```

---

## DPO: Direct Preference Optimization

### The Key Insight

```
RLHF with PPO requires:
  1. Train a reward model
  2. Use RL (PPO) to optimize against it
  
DPO eliminates BOTH steps:
  Directly optimize the policy using preference data!
  No reward model needed. No RL needed.
```

### DPO Loss Function

```
L_DPO = -log σ(β × (log π(y_w|x)/π_ref(y_w|x) 
                    - log π(y_l|x)/π_ref(y_l|x)))

Where:
  y_w = preferred response
  y_l = dispreferred response
  π = current policy model
  π_ref = reference model (frozen SFT model)
  β = temperature (controls how much policy can deviate from reference)

In English:
  "The log probability ratio (policy vs reference) should be
   HIGHER for the preferred response than the dispreferred response."
```

### Why DPO Works

```
Mathematical proof (Rafailov et al., 2023):
  DPO optimizes the SAME objective as RLHF!
  
  The optimal policy under the RLHF objective can be expressed as:
    π*(y|x) = (1/Z) × π_ref(y|x) × exp(R(y|x) / β)
  
  Where R is the IMPLICIT reward model.
  
  DPO bypasses the reward model entirely by reparameterizing:
    R(y|x) = β × log(π*(y|x)/π_ref(y|x)) + C
  
  No need to train R explicitly — it's captured in the policy!
```

### DPO vs RLHF

```
                    │ RLHF (PPO)     │ DPO
────────────────────┼────────────────┼──────────────
Models needed       │ 3 (policy,     │ 2 (policy,
                    │ reward, ref)   │ reference)
Training complexity │ Very high      │ Low (like SFT)
Memory              │ Very high      │ Moderate
Stability           │ Low (RL)       │ High
Quality             │ State-of-art   │ Comparable
Implementation      │ Complex        │ Simple
Hyperparameters     │ Many           │ Few (mainly β)
```

### DPO with LoRA

```python
from trl import DPOTrainer

# DPO training data format:
# Each example needs: prompt, chosen response, rejected response
dataset = [
    {
        "prompt": "Review: 'Great battery life!' Rating: 5",
        "chosen": "Recommendation: BUY. Strong battery praised...",
        "rejected": "Buy it. Good phone."
    },
    ...
]

# DPO trainer (very similar to SFT)
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=dataset,
    beta=0.1,  # KL penalty strength
)
trainer.train()
```

---

## ORPO: Odds Ratio Preference Optimization

### Simplifying Even Further

```
DPO still needs a reference model (frozen copy of SFT model).
ORPO eliminates even that!

ORPO combines SFT and preference optimization into ONE loss:
  L = L_SFT + λ × L_preference

Where L_preference uses odds ratios instead of log probability ratios.
No reference model needed at all!
```

### ORPO Loss

```
L_ORPO = L_NLL(y_w) + λ × log(1 + (odds(y_w)/odds(y_l)))

Where:
  L_NLL = standard cross-entropy loss on preferred response
  odds(y) = P(y|x) / (1 - P(y|x))
  
This simultaneously:
  1. Teaches the model to generate the preferred response (SFT part)
  2. Makes the model prefer chosen over rejected (preference part)
```

### ORPO vs DPO vs RLHF

```
                    │ RLHF   │ DPO      │ ORPO
────────────────────┼────────┼──────────┼──────────
Models in memory    │ 3      │ 2        │ 1 (!!)
Training stages     │ 3      │ 2        │ 1
Complexity          │ Very   │ Moderate │ Simple
Quality             │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐    │ ⭐⭐⭐⭐
Memory              │ 3×     │ 2×       │ 1×
```

---

## Constitutional AI (CAI)

### AI Teaching Itself

```
Problem: Collecting human preference data is expensive and slow.

CAI idea: Use AI to generate preference rankings!

Algorithm:
  1. Define a "constitution" (set of principles):
     - Be helpful and informative
     - Don't assist with harmful activities
     - Be honest about uncertainty
     - Respect privacy
     
  2. Generate multiple responses to a prompt
  
  3. Ask the AI to RANK responses based on the constitution:
     "Which response better follows the principle 'Be helpful'?"
     
  4. Use these AI-generated rankings for DPO training

Advantages:
  ✅ Scalable (no human annotators needed)
  ✅ Consistent (AI applies principles uniformly)
  ✅ Controllable (change principles, change behavior)
  
Disadvantages:
  ❌ AI preferences may not match human preferences
  ❌ Can amplify AI biases
  ❌ Quality limited by the ranking AI
```

---

## Comparison of Alignment Methods

| Method | Stages | Models | Memory | Quality | Complexity |
|--------|--------|--------|--------|---------|------------|
| **SFT only** (ours) | 1 | 1 | Low | Good | Simple |
| RLHF (PPO) | 3 | 3 | Very high | Excellent | Very high |
| **DPO** | 2 | 2 | Moderate | Very good | Moderate |
| **ORPO** | 1 | 1 | Low | Good-Very good | Simple |
| CAI + DPO | 2 | 2 | Moderate | Very good | Moderate |

---

## When to Use Each Method

```
JUST STARTING OUT (our project):
  → SFT only
  Get the basics working first. Most fine-tuning projects
  don't even need alignment — SFT is sufficient.

WANT BETTER OUTPUT QUALITY:
  → SFT + DPO
  If SFT outputs are okay but not great,
  and you can get preference data.

PRODUCTION CHATBOT:
  → SFT + DPO or ORPO
  Users expect "helpful" responses, not just correct ones.

SAFETY-CRITICAL APPLICATION:
  → SFT + RLHF (full pipeline)
  Maximum control over model behavior.

LIMITED RESOURCES:
  → SFT + ORPO (single model, simple)
  Best quality per compute dollar.
```
