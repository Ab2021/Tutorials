# Day 46: Planning Algorithms (RAP, LLM+P)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why use PDDL instead of just asking the LLM for a plan?

**Answer:**
*   **Correctness:** LLMs hallucinate plans (e.g., "Teleport to B"). PDDL solvers guarantee the plan is physically possible based on the defined rules.
*   **Optimality:** Solvers find the *shortest* plan. LLMs just find *a* plan.
*   **Complexity:** For a logistics problem with 50 trucks, an LLM will fail. A solver will succeed.

#### Q2: What is "Monte Carlo Tree Search" (MCTS) in the context of LLMs?

**Answer:**
It's a search strategy.
1.  **Selection:** Pick a promising node.
2.  **Expansion:** Generate possible next thoughts/actions.
3.  **Simulation:** Ask the LLM "If I do this, will I succeed?" (Rollout).
4.  **Backpropagation:** Update the score of the parent node.
It allows the agent to explore the "future" before committing to an action.

#### Q3: Difference between "Planning" and "Reasoning"?

**Answer:**
*   **Reasoning:** Deriving new information from existing information (Logic).
*   **Planning:** Deriving a sequence of actions to change the state of the world (Action).
*   Planning *requires* Reasoning.

#### Q4: What is "Skill Library" in planning?

**Answer:**
A collection of reusable plans or functions.
*   Instead of planning "How to make coffee" from scratch every time, the agent saves the plan as a "Skill".
*   Next time, the planner just invokes `MakeCoffee()`.

### Production Challenges

#### Challenge 1: The "Frozen" Planner

**Scenario:** The agent generates a 20-step plan. Step 1 fails. The agent keeps trying to execute Step 2.
**Root Cause:** Open-loop execution.
**Solution:**
*   **Feedback Loop:** Check state after *every* step.
*   **Dynamic Replanning:** If state != expected, pause and replan.

#### Challenge 2: Translation Errors (LLM+P)

**Scenario:** The LLM generates invalid PDDL syntax. The solver crashes.
**Root Cause:** LLMs are not perfect compilers.
**Solution:**
*   **Syntax Correction:** Use a linter or a second LLM pass to fix PDDL syntax errors.
*   **Few-Shot:** Provide many examples of valid PDDL in the prompt.

#### Challenge 3: Latency

**Scenario:** MCTS takes 5 minutes to generate a response. User leaves.
**Root Cause:** Too many LLM calls (Simulations).
**Solution:**
*   **Small Model:** Use a small, fast model (Llama-3-8B) for the simulations/rollouts, and a large model (GPT-4) for the final decision.

### System Design Scenario: Autonomous Coding Agent

**Requirement:** "Refactor this repo to use AsyncIO."
**Design:**
1.  **Plan:**
    *   Scan files.
    *   Identify blocking calls.
    *   Create dependency graph.
    *   Refactor bottom-up (utils first).
2.  **Execute:**
    *   Agent picks file A.
    *   Refactors.
    *   Runs tests.
3.  **Replan:**
    *   Tests fail.
    *   Agent realizes `utils.py` broke `main.py`.
    *   Adds "Fix main.py" to the plan.

### Summary Checklist for Production
*   [ ] **State Tracking:** You need a reliable way to know the current state of the world.
*   [ ] **Timeout:** Don't plan forever.
*   [ ] **Human Interrupt:** Allow the user to cancel/modify the plan mid-execution.
