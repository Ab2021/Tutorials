# Day 47: Reflection & Self-Correction (Reflexion)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing Reflexion (Python)

We will solve a coding problem. The agent tries to write code that passes tests.

```python
class ReflexionAgent:
    def __init__(self, client):
        self.client = client
        self.memory = [] # List of reflections

    def solve(self, problem, tests, max_trials=3):
        for trial in range(max_trials):
            # 1. Generate Code
            code = self.generate_code(problem)
            
            # 2. Evaluate (Run Tests)
            success, error_log = self.run_tests(code, tests)
            
            if success:
                return code
            
            # 3. Reflect
            reflection = self.reflect(code, error_log)
            self.memory.append(reflection)
            print(f"Trial {trial} Failed. Reflection: {reflection}")
            
        return "Failed"

    def generate_code(self, problem):
        # Inject past reflections into context
        reflections_str = "\n".join(self.memory)
        prompt = f"""
        Problem: {problem}
        Previous Mistakes & Lessons:
        {reflections_str}
        
        Write Python code to solve this.
        """
        return self.client.chat.completions.create(...).content

    def reflect(self, code, error_log):
        prompt = f"""
        I wrote this code:
        {code}
        
        It failed with this error:
        {error_log}
        
        Analyze WHY it failed. Be specific. Suggest a fix.
        """
        return self.client.chat.completions.create(...).content

    def run_tests(self, code, tests):
        # ... exec() logic ...
        try:
            exec(code)
            # ... assert tests ...
            return True, ""
        except Exception as e:
            return False, str(e)
```

### Chain of Hindsight (CoH)

Similar to Reflexion, but used during **Fine-Tuning**.
*   Data: `(Prompt, Bad Output, Critique, Good Output)`.
*   Train the model to predict the Good Output given the Bad Output + Critique.
*   This teaches the model to "listen" to feedback.

### Self-Refine Prompting

For creative writing:
```
Prompt: Write an email to the CEO.
Draft: Hey boss...
Critique: Too informal.
Refined: Dear Mr. CEO...
```
This can be automated in a single loop.

### Summary

*   **Short-Term Memory:** Reflections help the *current* task (Retry).
*   **Long-Term Memory:** If you save reflections to a Vector DB, the agent learns across *different* tasks ("I remember I'm bad at recursion, so I should be careful").
