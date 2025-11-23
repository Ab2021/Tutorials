# Day 54: Evaluation of Tool-Using Agents
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between "Process Reward" and "Outcome Reward" in agent evaluation?

**Answer:**
*   **Outcome Reward:** Did the agent get the right answer? (Binary: Pass/Fail).
    *   *Pros:* Easy to measure.
    *   *Cons:* Sparse signal. Doesn't tell you *why* it failed.
*   **Process Reward:** Did the agent take the correct steps? (Dense signal).
    *   *Pros:* Helps identify if the retrieval failed, or the tool call failed, or the reasoning failed.
    *   *Cons:* Harder to define. Requires a "Judge" to evaluate the logic.

#### Q2: How do you evaluate an agent that has "Side Effects" (e.g., sending emails)?

**Answer:**
You cannot run these in a live environment.
**Strategy:**
1.  **Sandboxing:** Run the agent in a Docker container where network calls are intercepted.
2.  **Mocking:** Replace the `send_email` tool with a `mock_send_email` that just logs the call arguments.
3.  **Evaluation:** Check the logs. Did the agent call `mock_send_email` with the correct recipient and body?

#### Q3: Why is "Pass@K" a better metric than "Accuracy" for coding agents?

**Answer:**
Coding agents are highly non-deterministic. They might fail 50% of the time due to syntax errors.
**Pass@K:** If we give the agent K attempts (e.g., 10), what is the probability that *at least one* is correct?
This reflects real-world usage: A developer generates 10 snippets and picks the one that works.

#### Q4: What is "Data Contamination" in benchmarks?

**Answer:**
When the questions in your test set (e.g., MMLU) were present in the training data of the LLM. The model isn't reasoning; it's remembering.
**Detection:** Modify the numbers or names in the question. If the model performance drops drastically, it was memorizing.

### Production Challenges

#### Challenge 1: The "Flaky" Eval

**Scenario:** You run the eval suite. It passes. You run it again. It fails.
**Root Cause:** LLM Temperature > 0.
**Solution:**
*   **Greedy Decoding:** Set `temperature=0` for evaluation runs to ensure reproducibility.
*   **Averaging:** If you need creativity (temp > 0), run the eval 5 times and average the score.

#### Challenge 2: Judge Bias

**Scenario:** Your GPT-4 Judge consistently rates long, verbose answers higher than short, correct answers.
**Root Cause:** LLM bias (Length Bias).
**Solution:**
*   **Reference-Free Eval:** Don't just ask "Is this good?". Ask "Does this match the reference answer?".
*   **Pairwise Comparison:** Show the judge two answers (A and B) and ask "Which is better?". This is more reliable than absolute scoring.

#### Challenge 3: Cost of Eval

**Scenario:** Running 1000 test cases with GPT-4 as a judge costs $50 per run. You can't run this on every commit.
**Root Cause:** Expensive models.
**Solution:**
*   **Tiered Eval:**
    *   **Unit Tests (Free):** Run on every commit.
    *   **Small Model Judge (Cheap):** Run Llama-3-70B judge on nightly builds.
    *   **GPT-4 Judge (Expensive):** Run only before release.

#### Challenge 4: Metric Gaming

**Scenario:** You optimize for "Fewest Steps". The agent starts guessing answers instead of using tools to verify them.
**Root Cause:** Goodhart's Law.
**Solution:**
*   **Counter-Metrics:** Track "Hallucination Rate" alongside "Efficiency".
*   **Holistic Score:** A weighted average of Accuracy, Efficiency, and Safety.

### System Design Scenario: Agent CI/CD Pipeline

**Requirement:** Automatically test a Customer Support Agent before deployment.
**Design:**
1.  **Commit:** Dev pushes code.
2.  **Build:** Docker image built.
3.  **Unit Test:** Mocked tool tests run (Python `pytest`).
4.  **Integration Test:** Agent runs against a **Simulated Database** (Dockerized Postgres).
5.  **Trajectory Eval:** Run 50 "Golden Questions".
    *   Capture traces.
    *   Send to LangSmith.
    *   GPT-4 grades them.
6.  **Gate:** If Score > 90%, deploy to Staging.

### Summary Checklist for Production
*   [ ] **Golden Dataset:** Maintain a set of 50+ high-quality Q&A pairs.
*   [ ] **Mocking:** Mock all external APIs in the test suite.
*   [ ] **Judge:** Use LLM-as-a-Judge for reasoning tasks.
*   [ ] **CI/CD:** Integrate eval into the build pipeline.
*   [ ] **Cost:** Monitor the cost of the evaluation itself.
