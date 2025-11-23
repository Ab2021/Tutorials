# Lab 2: Plan-and-Solve Agent

## Objective
Build an agent that creates a **Plan** first, then executes it.
This is better for long-horizon tasks than ReAct (which thinks one step at a time).

## 1. The Planner (`planner.py`)

```python
PLANNER_PROMPT = """
You are a Planner.
Given a complex task, break it down into a numbered list of sub-tasks.
Do not solve the task. Just plan.
"""

def create_plan(task):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": task}
        ]
    ).choices[0].message.content
```

## 2. The Executor (`executor.py`)

```python
EXECUTOR_PROMPT = """
You are a Worker.
You are given a sub-task and the context from previous tasks.
Execute the sub-task.
"""

def execute_plan(plan, original_task):
    steps = plan.split("\n")
    context = ""
    
    for step in steps:
        print(f"Executing: {step}")
        result = client.chat.completions.create(
            model="gpt-3.5-turbo", # Workers can be cheaper models
            messages=[
                {"role": "system", "content": EXECUTOR_PROMPT},
                {"role": "user", "content": f"Task: {original_task}\nContext: {context}\nCurrent Step: {step}"}
            ]
        ).choices[0].message.content
        
        context += f"\nStep: {step}\nResult: {result}\n"
        
    return context

# Run
task = "Write a blog post about AI, translate it to Spanish, and save it to a file."
plan = create_plan(task)
print(f"Plan:\n{plan}")
final_output = execute_plan(plan, task)
print(final_output)
```

## 3. Challenge
*   **Replanning:** If a step fails (e.g., "Translation API Error"), the Planner should be called again to update the remaining steps.

## 4. Submission
Submit the output log showing the Plan and the Execution of each step.
