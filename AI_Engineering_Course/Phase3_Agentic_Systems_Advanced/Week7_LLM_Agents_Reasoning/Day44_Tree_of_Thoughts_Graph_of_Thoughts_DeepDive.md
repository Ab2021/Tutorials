# Day 44: Tree of Thoughts (ToT) & Graph of Thoughts (GoT)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing Tree of Thoughts (BFS)

We will solve the "Game of 24" (Use 4 numbers and basic math to get 24).
*   Input: `4 9 10 13`
*   Goal: `(13 - 9) * (10 - 4) = 24`

```python
import itertools

class TreeOfThoughts:
    def __init__(self, client, model="gpt-4"):
        self.client = client
        self.model = model

    def generate_thoughts(self, current_state, k=3):
        """Propose k next steps."""
        prompt = f"""
        Current numbers: {current_state}
        Goal: Reach 24.
        Propose {k} possible next operations (e.g., '4 + 9 = 13').
        Only one operation per line.
        """
        # ... call LLM ...
        return ["4 + 9 = 13", "13 - 10 = 3", ...]

    def evaluate_states(self, states):
        """Score states: 1.0 (Sure win) to 0.0 (Impossible)."""
        prompt = f"""
        Evaluate if these states can reach 24:
        {states}
        Output a score 0-1 for each.
        """
        # ... call LLM ...
        return [0.8, 0.1, ...]

    def solve(self, initial_numbers, breadth=3, depth=3):
        current_level = [initial_numbers]
        
        for d in range(depth):
            print(f"--- Depth {d} ---")
            next_level = []
            
            for state in current_level:
                # 1. Generate
                thoughts = self.generate_thoughts(state, k=breadth)
                
                # 2. Apply (Simulate the math)
                new_states = [self.apply_op(state, t) for t in thoughts]
                
                # 3. Evaluate
                scores = self.evaluate_states(new_states)
                
                # 4. Prune
                for s, score in zip(new_states, scores):
                    if score > 0.5:
                        next_level.append(s)
                        if self.check_24(s):
                            return s
            
            # Keep top B states
            current_level = next_level[:breadth]
            
        return "Failed"

    def apply_op(self, state, op):
        # Helper to parse "4 + 9 = 13" and update the list of numbers
        # [4, 9, 10, 13] -> [13, 10, 13]
        pass
```

### Graph of Thoughts (GoT) Pattern

GoT is powerful for **Summarization**.
1.  **Generate:** Split document into 3 chunks. Summarize each. (Nodes A, B, C).
2.  **Aggregate:** Combine A+B -> D. Combine B+C -> E.
3.  **Refine:** Improve D -> D'. Improve E -> E'.
4.  **Final:** Combine D' + E' -> Final Summary.

This topology allows the model to find connections between distant parts of the text that a linear summary might miss.

### The "Controller"

In ToT/GoT, the Python script acts as the **Controller**.
*   It manages the memory (Tree/Graph structure).
*   It calls the LLM (Worker) for specific tasks (Generate, Evaluate).
*   It decides the control flow (Pruning, Backtracking).

### Summary

ToT is **System 2 Thinking** (Slow, Deliberate) implemented in software. It allows an LLM to solve problems that are strictly harder than what it can solve in a single pass.
