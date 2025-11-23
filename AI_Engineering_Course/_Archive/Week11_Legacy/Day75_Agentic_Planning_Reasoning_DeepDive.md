# Day 75: Agentic Planning & Reasoning
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Tree of Thoughts (ToT) Implementation

Implementing a BFS search for a creative writing task.

```python
import heapq

class Node:
    def __init__(self, state, score, parent=None):
        self.state = state
        self.score = score
        self.parent = parent

def generate_thoughts(llm, state, k=3):
    # Generate k possible next steps
    prompt = f"Current state: {state}. Generate {k} possible next sentences."
    # response = llm.generate(prompt)
    return ["Option A", "Option B", "Option C"]

def evaluate_state(llm, state):
    # Score the state (0.0 to 1.0)
    prompt = f"Rate the quality of this story segment: {state}. Return float."
    # score = float(llm.generate(prompt))
    return 0.8

def tree_of_thoughts_bfs(llm, initial_state, depth=3, beam_width=2):
    queue = [Node(initial_state, 1.0)]
    
    for d in range(depth):
        next_queue = []
        for node in queue:
            # 1. Generate
            thoughts = generate_thoughts(llm, node.state)
            
            # 2. Evaluate
            for thought in thoughts:
                new_state = node.state + " " + thought
                score = evaluate_state(llm, new_state)
                next_queue.append(Node(new_state, score, node))
                
        # 3. Prune (Beam Search)
        next_queue.sort(key=lambda x: x.score, reverse=True)
        queue = next_queue[:beam_width]
        
    return queue[0].state # Best state

# Usage
# best_story = tree_of_thoughts_bfs(llm, "Once upon a time")
```

### 2. Reflexion Loop (Self-Correction)

Agent that improves code through error feedback.

```python
def run_code(code):
    # Mock execution
    if "error" in code:
        return False, "Syntax Error"
    return True, "Success"

def reflexion_agent(llm, task, max_retries=3):
    code = llm.generate(f"Write code for: {task}")
    memory = []
    
    for i in range(max_retries):
        success, output = run_code(code)
        
        if success:
            return code
            
        # Reflect
        reflection = llm.generate(f"""
        Task: {task}
        Code: {code}
        Error: {output}
        Previous Reflections: {memory}
        
        Analyze why it failed and propose a fix.
        """)
        
        memory.append(reflection)
        
        # Retry
        code = llm.generate(f"""
        Task: {task}
        Previous Code: {code}
        Error: {output}
        Reflection: {reflection}
        
        Write fixed code.
        """)
        
    return "Failed"
```

### 3. Plan-and-Solve Agent

Generating a DAG of tasks.

```python
def plan_and_solve(llm, goal):
    # 1. Plan
    plan = llm.generate(f"Goal: {goal}. Create a numbered list of steps.")
    steps = plan.split("\n")
    print(f"Plan: {steps}")
    
    context = {}
    
    # 2. Execute
    for step in steps:
        # Check dependencies? (Simplified: Sequential)
        result = llm.generate(f"Execute step: {step}. Context: {context}")
        context[step] = result
        print(f"Finished: {step}")
        
    # 3. Final Answer
    final = llm.generate(f"Goal: {goal}. Context: {context}. Summarize answer.")
    return final
```

### 4. LLM Compiler (Parallelism Concept)

Parsing a plan into parallelizable tasks.

```python
# Goal: "Get weather in NY and London"
# Plan:
# 1. get_weather("NY")
# 2. get_weather("London")
# 3. summarize(1, 2)

# Dependency Graph:
# 1 -> 3
# 2 -> 3

# Execution:
# Run 1 and 2 in parallel (AsyncIO).
# Wait for both.
# Run 3.
```
