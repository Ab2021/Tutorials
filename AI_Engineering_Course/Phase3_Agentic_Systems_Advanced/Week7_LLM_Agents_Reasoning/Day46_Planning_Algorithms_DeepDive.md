# Day 46: Planning Algorithms (RAP, LLM+P)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Simple Planner (LLM-Based)

We will implement a "Travel Planner" that generates a plan and then executes it.

```python
class PlanningAgent:
    def __init__(self, client):
        self.client = client

    def make_plan(self, goal):
        prompt = f"""
        Goal: {goal}
        Create a step-by-step plan to achieve this goal.
        Format:
        1. [ToolName: Args] - Description
        2. ...
        """
        plan_text = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
        
        return self.parse_plan(plan_text)

    def execute_plan(self, plan):
        results = []
        for step in plan:
            print(f"Executing: {step}")
            # ... execute tool ...
            # ... check if successful ...
            # If fail, trigger Replanning
            if not success:
                print("Step failed. Replanning...")
                return self.replan(remaining_goal)
        return "Success"

    def parse_plan(self, text):
        # ... parsing logic ...
        return ["BookFlight", "BookHotel"]

# Usage
agent = PlanningAgent(client)
plan = agent.make_plan("Go to Tokyo for 5 days")
agent.execute_plan(plan)
```

### LLM+P (PDDL Example)

**PDDL Domain (defined once):**
```lisp
(define (domain logistics)
  (:action move-truck
     :parameters (?t - truck ?from ?to - location)
     :precondition (at ?t ?from)
     :effect (and (not (at ?t ?from)) (at ?t ?to)))
  ...
)
```

**LLM Prompt (Translation):**
"User: Move the package from A to B."
"Output PDDL Problem:"
```lisp
(define (problem p1)
  (:init (at package1 A) (at truck1 A))
  (:goal (at package1 B))
)
```

**Solver:**
Running a solver on this PDDL guarantees a correct plan: `(load package1 truck1 A), (move-truck truck1 A B), (unload package1 truck1 B)`.

### Replanning

No plan survives contact with reality.
**Replanning** is crucial.
*   **Precondition Checking:** Before executing Step 3, check if Step 2 actually worked.
*   **Refinement:** If Step 2 failed, ask the LLM: "The plan failed at Step 2 because X. Generate a new plan from the current state."

### Summary

*   **LLM as Translator:** Use the LLM to interface with formal logic (PDDL/Code).
*   **LLM as Simulator:** Use the LLM to evaluate plans.
*   **Closed Loop:** Planning is not a one-off event; it's a loop of `Plan -> Execute -> Monitor -> Replan`.
