# Day 24: Advanced Prompt Engineering Techniques
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Tree of Thoughts (ToT) Implementation

**The Problem:** "Game of 24". Given 4 numbers (e.g., 4, 9, 10, 13), use basic arithmetic (+, -, *, /) to reach 24.
**Why CoT fails:** It commits to a path early. If it says "4 + 9 = 13", it might be a dead end. It cannot backtrack.

**ToT Algorithm:**
1.  **Propose:** Given current state (numbers left), propose 3 possible next steps.
2.  **Evaluate:** Rate each proposal (Sure/Maybe/Impossible) using the LLM.
3.  **Select:** Keep top $k$ states.
4.  **Repeat:** Until 24 is reached or steps exhausted.

**Code Structure:**
```python
def solve_tot(numbers):
    current_states = [numbers] # Initial state
    for step in range(3): # Max 3 operations for 4 numbers
        next_states = []
        for state in current_states:
            # 1. Propose
            proposals = generate_proposals(state) 
            # 2. Evaluate
            scores = evaluate_proposals(proposals)
            # 3. Filter
            good_states = [p for p, s in zip(proposals, scores) if s > 0.5]
            next_states.extend(good_states)
        current_states = next_states
    return current_states
```

### 2. Self-Consistency: The Math

Let $P(a|q)$ be the probability of answer $a$ given question $q$.
In greedy decoding, we pick $\text{argmax}_t P(t_i | t_{<i})$. This is a local optimum at each token, not necessarily the global optimum for the whole sequence.
By sampling $N$ times with temperature $T > 0$, we approximate the posterior distribution $P(a|q)$.
Taking the majority vote is equivalent to marginalizing over the reasoning paths $r$:
$$ P(a|q) = \sum_r P(a|r, q) P(r|q) $$
This is robust because there are many reasoning paths that lead to the *correct* answer, but usually only one path that leads to any specific *incorrect* answer.

### 3. Prompt Optimization (DSPy)

**Concept:** Prompting is "manual gradient descent". We tweak words to improve metrics.
**DSPy (Declarative Self-improving Language Programs):**
- Treat prompts as **parameters**.
- Define a "Signature" (Input -> Output).
- Define a "Teleprompter" (Optimizer).
- The optimizer iterates on the prompt (few-shot examples, instructions) to maximize a metric (e.g., accuracy) on a validation set.
- **Result:** Automated prompt engineering.

### 4. System Prompts vs. User Prompts

**System Prompt:** Sets the behavior/persona. "You are a helpful assistant."
**User Prompt:** The specific task.
**Attention Mechanism:**
In most models, the System Prompt is at the very beginning.
Due to the "Lost in the Middle" phenomenon (and attention decay), extremely long system prompts can be ignored.
**Tip:** Repeat critical instructions at the *end* of the User Prompt ("Recency Bias").

### Code: Self-Consistency Wrapper

```python
from collections import Counter

def self_consistency_solve(question, n=5):
    answers = []
    for _ in range(n):
        # Use high temp for diversity
        response = generate(question, temperature=0.7)
        ans = extract_answer(response)
        answers.append(ans)
    
    # Majority Vote
    counts = Counter(answers)
    most_common, count = counts.most_common(1)[0]
    return most_common
```
