# Lab 3: Self-Correction (Reflexion)

## Objective
Agents make mistakes. **Reflexion** allows them to fix them.
Loop: Act -> Evaluate -> Reflect -> Retry.

## 1. The Loop (`reflexion.py`)

```python
def solve_math(problem):
    # Mock Model that fails first
    return "The answer is 5." # Wrong for 2+2

def evaluate(problem, solution):
    if problem == "2+2" and "4" not in solution:
        return False, "You said 5, but 2+2 is 4."
    return True, "Correct."

def reflect(history):
    # Mock Reflection
    return "I made a calculation error. I should verify my math."

# Run
problem = "2+2"
history = ""

for i in range(3):
    print(f"--- Attempt {i+1} ---")
    solution = solve_math(problem)
    
    # Mock improvement on 2nd try
    if i > 0: solution = "The answer is 4."
    
    print(f"Solution: {solution}")
    
    success, feedback = evaluate(problem, solution)
    if success:
        print("Success!")
        break
        
    reflection = reflect(history)
    print(f"Feedback: {feedback}")
    print(f"Reflection: {reflection}")
    history += f"Attempt {i}: {solution}\nFeedback: {feedback}\nReflection: {reflection}\n"
```

## 2. Analysis
The `history` context allows the model to "learn" from its mistake within the episode.

## 3. Submission
Submit the trace of a 3-step reflection loop.
