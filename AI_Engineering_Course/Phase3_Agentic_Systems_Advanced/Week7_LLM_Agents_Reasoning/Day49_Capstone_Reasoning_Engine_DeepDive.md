# Day 49: Capstone: Building a Reasoning Engine
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing the "Thinker-Doer-Critic" Loop

We will build a system to answer: "Who is the current Prime Minister of the country with the highest GDP per capita?"

```python
class ReasoningEngine:
    def __init__(self, client, tools):
        self.client = client
        self.tools = tools
        self.memory = []

    def run(self, query):
        print(f"--- Goal: {query} ---")
        
        # 1. Think (Plan)
        plan = self.think(query)
        print(f"Plan: {plan}")
        
        for step in plan:
            # 2. Do (Act)
            result = self.do(step)
            print(f"Result: {result}")
            
            # 3. Critic (Reflect)
            status, feedback = self.critic(step, result, query)
            if status == "FAIL":
                print(f"Step Failed: {feedback}. Replanning...")
                return self.run(f"{query} (Previous attempt failed: {feedback})")
                
        return "Task Complete"

    def think(self, query):
        prompt = f"Create a concise list of steps to solve: {query}"
        return self.client.chat.completions.create(..., messages=[{"role": "user", "content": prompt}]).content.split("\n")

    def do(self, step):
        # Simple ReAct-style execution
        prompt = f"Execute this step using tools: {step}"
        # ... tool calling logic ...
        return "Executed"

    def critic(self, step, result, query):
        prompt = f"""
        Step: {step}
        Result: {result}
        Goal: {query}
        
        Did this step succeed? Output PASS or FAIL.
        """
        response = self.client.chat.completions.create(..., messages=[{"role": "user", "content": prompt}]).content
        if "FAIL" in response:
            return "FAIL", response
        return "PASS", ""

# Usage
# engine = ReasoningEngine(client, tools)
# engine.run("Who is the PM of the richest country?")
```

### Advanced: Monte Carlo Planning

Instead of a single linear plan, the **Thinker** generates 3 possible plans.
The **Critic** evaluates which one is most likely to succeed *before* execution.
This is **Inference-Time Compute** applied to planning.

### Handling Ambiguity

If the query is "What is the best phone?", the engine should:
1.  **Think:** "Best is subjective."
2.  **Do:** `AskUser("What is your budget and preference?")`
3.  **Wait:** Stop execution until user responds.

### Summary

*   **Modularity:** Separating Think/Do/Critic allows you to use different models (e.g., o1-preview for Thinking, GPT-4o for Doing, GPT-4o-mini for Critic).
*   **Recursion:** The engine can call itself recursively for sub-tasks.
