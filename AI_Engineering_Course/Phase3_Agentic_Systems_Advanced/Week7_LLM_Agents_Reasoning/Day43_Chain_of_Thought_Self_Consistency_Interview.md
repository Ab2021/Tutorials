# Day 43: Chain of Thought (CoT) & Self-Consistency
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why does "Let's think step by step" work?

**Answer:**
LLMs are autoregressive. They predict the next token based on previous tokens.
*   Without CoT: `P(Answer | Question)`
*   With CoT: `P(Answer | Question, Reasoning)`
The reasoning tokens provide **computation time** and **intermediate state**. They ground the final answer in a logical derivation rather than a direct jump. It effectively increases the depth of the computation graph.

#### Q2: When should you *not* use CoT?

**Answer:**
*   **Simple Tasks:** "What is the capital of France?" CoT adds latency and cost for no gain.
*   **Retrieval Tasks:** "Who is the CEO of Apple?" CoT might cause hallucination ("Let's think... Steve Jobs founded it... so he must be CEO").
*   **Latency-Sensitive:** Real-time chat.

#### Q3: How do you evaluate Reasoning capabilities?

**Answer:**
*   **GSM8K:** Grade School Math benchmark.
*   **MATH:** Harder math problems.
*   **Big-Bench Hard (BBH):** Tasks that require multi-step reasoning.
*   **Metric:** Exact Match (EM) of the final answer.

#### Q4: What is "Faithfulness" in CoT?

**Answer:**
Does the model actually follow its own reasoning?
*   *Unfaithful:* Reasoning says "X is true, Y is false, so Z is false." Final Answer: "Z is true."
*   *Problem:* The model might ignore the reasoning and just guess the answer based on priors.

### Production Challenges

#### Challenge 1: The "Context Window" Tax

**Scenario:** You use 5-shot CoT. Each example is 500 tokens. Your prompt is now 2500 tokens.
**Root Cause:** Verbose reasoning.
**Solution:**
*   **Zero-Shot CoT:** Use "Let's think step by step" (0 tokens).
*   **Fine-Tuning:** Fine-tune the model to output reasoning without needing examples in the prompt.

#### Challenge 2: Parsing Failures

**Scenario:** You ask for JSON. The model writes a long CoT essay and *then* the JSON. The JSON parser breaks.
**Root Cause:** Mixed content.
**Solution:**
*   **Chain of Density:** Ask for reasoning in a specific XML tag `<thinking>...</thinking>` and the answer in `<answer>...</answer>`. Parse with Regex.

#### Challenge 3: Hallucinated Logic

**Scenario:** The model makes a logical error in Step 3. Step 4 and 5 are correct based on Step 3, but the final answer is wrong.
**Root Cause:** Error propagation.
**Solution:**
*   **Verifier:** Train a separate "Reward Model" or "Verifier" to check each step of the reasoning (Process Supervision).

### System Design Scenario: Math Tutor Bot

**Requirement:** A bot that helps kids with homework but doesn't just give the answer.
**Design:**
1.  **Prompt:** "Solve this problem step by step. Do NOT output the final answer yet. Ask the student to verify Step 1."
2.  **State:** Store the full CoT in the backend.
3.  **Interaction:** Reveal one step at a time.
4.  **Correction:** If the student is stuck, generate a "Hint" based on the next step of the CoT.

### Summary Checklist for Production
*   [ ] **Temperature:** Use `temp=0` for greedy CoT, `temp=0.7` for Self-Consistency.
*   [ ] **Stop Sequences:** Ensure the model stops after the answer.
*   [ ] **Parsing:** Robust regex to extract the final number/class.
