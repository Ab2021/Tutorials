# Day 61: Consensus Protocols (Voting, Debate)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Debate System

We will build a **Debate Loop** where two agents argue about a topic until a Judge decides.

### 1. The Debate Loop

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

pro_agent = "You are arguing IN FAVOR of the topic. Be concise."
con_agent = "You are arguing AGAINST the topic. Be concise."
judge_agent = "You are a Judge. Decide who won. Output 'WINNER: PRO' or 'WINNER: CON' or 'CONTINUE'."

topic = "Remote work is better than office work."
history = [f"Topic: {topic}"]

def run_debate_round(history):
    # Pro speaks
    pro_msg = llm.invoke([{"role": "system", "content": pro_agent}] + 
                         [{"role": "user", "content": "\n".join(history)}]).content
    history.append(f"Pro: {pro_msg}")
    
    # Con speaks
    con_msg = llm.invoke([{"role": "system", "content": con_agent}] + 
                         [{"role": "user", "content": "\n".join(history)}]).content
    history.append(f"Con: {con_msg}")
    
    # Judge decides
    verdict = llm.invoke([{"role": "system", "content": judge_agent}] + 
                         [{"role": "user", "content": "\n".join(history)}]).content
    return verdict, history

# Loop
for i in range(3):
    verdict, history = run_debate_round(history)
    print(f"Round {i}: {verdict}")
    if "WINNER" in verdict:
        break
```

### 2. Self-Consistency (Sampling)

Implementing "Majority Vote" on a math problem.

```python
from collections import Counter

def solve_math(problem, n=5):
    answers = []
    for _ in range(n):
        # High temp for diversity
        resp = llm.invoke(problem, temperature=0.7).content
        # Extract final number (simplified)
        ans = extract_number(resp) 
        answers.append(ans)
    
    # Vote
    counts = Counter(answers)
    winner, count = counts.most_common(1)[0]
    return winner, count/n # Confidence

# Usage
# ans, conf = solve_math("What is 123 * 456?")
# If 4/5 runs say "56088", confidence is 0.8.
```

### 3. Reflection (Reflexion)

An agent critiquing itself.

```python
def reflexion_loop(task):
    draft = llm.invoke(f"Solve: {task}").content
    
    for _ in range(3):
        critique = llm.invoke(f"Review this solution for errors: {draft}").content
        if "No errors" in critique:
            break
            
        draft = llm.invoke(f"Fix the solution based on critique: {critique}\nOriginal: {draft}").content
        
    return draft
```

### 4. Graph-Based Consensus (LangGraph)

Using a graph to enforce a "Review" step.

```python
# Nodes: Generator, Reviewer
# Edge: Reviewer -> Generator (if rejected)
# Edge: Reviewer -> End (if accepted)

def reviewer_node(state):
    score = grade(state['draft'])
    if score > 8:
        return "accept"
    return "reject"

workflow.add_conditional_edges(
    "reviewer",
    reviewer_node,
    {"accept": END, "reject": "generator"}
)
```

### Summary

*   **Voting:** Fast, parallelizable, good for factual QA.
*   **Debate:** Slow, sequential, good for nuanced/subjective topics.
*   **Reflection:** Single-agent loop, good for code generation.
*   **Key:** You need a strong "Judge" or "Verifier". If the Judge is weak, the consensus is meaningless.
