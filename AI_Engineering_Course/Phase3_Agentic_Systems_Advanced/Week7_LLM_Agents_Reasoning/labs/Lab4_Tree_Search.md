# Lab 4: Tree of Thoughts (ToT)

## Objective
Explore multiple reasoning paths.
Implement **BFS** (Breadth-First Search) for thought generation.

## 1. The Search (`tot.py`)

```python
# Problem: Write a poem with 4 lines.
initial_state = ""

def generate_thoughts(state):
    # Returns 3 possible next lines
    return [state + f"Line {len(state)//10 + 1} A\n", 
            state + f"Line {len(state)//10 + 1} B\n",
            state + f"Line {len(state)//10 + 1} C\n"]

def evaluate_states(states):
    # Score each state (Mock)
    scores = []
    for s in states:
        scores.append(1.0 if "A" in s else 0.5)
    return scores

# BFS
beam_width = 2
current_states = [initial_state]

for step in range(4): # 4 lines
    candidates = []
    for s in current_states:
        next_thoughts = generate_thoughts(s)
        candidates.extend(next_thoughts)
        
    # Prune
    scores = evaluate_states(candidates)
    # Sort by score and keep top k
    sorted_candidates = [x for _, x in sorted(zip(scores, candidates), reverse=True)]
    current_states = sorted_candidates[:beam_width]
    
    print(f"Step {step}: Best state so far:\n{current_states[0]}")
```

## 2. Analysis
We keep the best `k` partial solutions at each step.
This avoids getting stuck in a local optimum (bad early choice).

## 3. Submission
Submit the final generated "poem".
