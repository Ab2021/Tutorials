# Day 47: Reflection & Self-Correction (Reflexion)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between Reflexion and Chain of Thought?

**Answer:**
*   **CoT:** Reasoning *before* the answer. (Feedforward).
*   **Reflexion:** Reasoning *after* the answer (and failure). (Feedback).
*   They are complementary. You use CoT to generate the attempt, and Reflexion to improve it.

#### Q2: Does Reflexion require a Ground Truth?

**Answer:**
Yes, usually.
*   **Hard Evaluation:** Unit tests, Math answers. (Binary Success/Fail).
*   **Soft Evaluation:** "Is this poem good?" Requires a strong Critic (Reward Model or GPT-4). If the Critic is weak, Reflexion fails (Garbage In, Garbage Out).

#### Q3: What is "Mode Collapse" in Self-Correction?

**Answer:**
The model keeps making the same mistake despite reflecting.
*   *Reflection:* "I need to fix the index error."
*   *Next Attempt:* Makes the same index error.
*   *Cause:* The model's weights strongly prefer the wrong pattern. The prompt context isn't strong enough to override it.

#### Q4: How does Reflexion relate to "Verbal RL"?

**Answer:**
It is a form of RL where the "Policy" is the LLM, the "State" is the Context, and the "Update" is appending text (Reflection) instead of gradient descent. It allows "learning" without training.

### Production Challenges

#### Challenge 1: Infinite Loops

**Scenario:** Agent writes code -> Fails -> Reflects -> Writes same code -> Fails.
**Root Cause:** Deterministic sampling.
**Solution:**
*   **Temperature:** Increase temp on retries.
*   **History Check:** If the new code is identical to previous code, force a drastic change or stop.

#### Challenge 2: The "Sycophantic" Critic

**Scenario:** The Critic says "Good job!" even when the output is bad.
**Root Cause:** LLMs are trained to be helpful/agreeable.
**Solution:**
*   **Role Prompting:** "You are a harsh code reviewer. Find bugs."
*   **Separate Model:** Use a different, stronger model for the Critic.

#### Challenge 3: Context Window Explosion

**Scenario:** After 10 retries, the context is full of failed code and reflections.
**Root Cause:** Accumulation.
**Solution:**
*   **Summarization:** Summarize the last 5 failures into "Key Lessons".
*   **FIFO:** Keep only the last 2 retries.

### System Design Scenario: Auto-GPT for Bug Fixing

**Requirement:** Fix a GitHub issue automatically.
**Design:**
1.  **Reproduce:** Write a test case that fails (The Evaluator).
2.  **Loop:**
    *   Read code.
    *   Attempt fix.
    *   Run test.
    *   If fail: Read stack trace -> Reflect ("I modified the wrong variable") -> Retry.
3.  **Success:** If test passes, submit PR.

### Summary Checklist for Production
*   [ ] **Evaluator:** You need a reliable signal for success/failure.
*   [ ] **Max Retries:** Stop after N attempts to save money.
*   [ ] **Diversity:** Force the agent to try a *different* approach, not just tweak the same one.
