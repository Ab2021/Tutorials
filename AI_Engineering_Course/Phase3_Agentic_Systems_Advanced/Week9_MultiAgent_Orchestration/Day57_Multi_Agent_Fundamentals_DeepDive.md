# Day 57: Multi-Agent Fundamentals
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Basic Multi-Agent Loop

We will build a simple **Two-Agent System** (Writer + Critic) from scratch, without using a framework like AutoGen yet. This helps understand the underlying message passing.

### 1. Defining the Agents

```python
from openai import OpenAI

client = OpenAI()

class Agent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": system_prompt}]
        
    def receive_message(self, sender, message):
        # Add message to history
        self.history.append({"role": "user", "content": f"{sender}: {message}"})
        
    def generate_reply(self):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.history
        )
        content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": content})
        return content
```

### 2. The Orchestration Loop (Round Robin)

We manually pass messages between them.

```python
# Setup
writer = Agent("Writer", "You are a creative writer. Write short stories.")
critic = Agent("Critic", "You are a harsh critic. Critique the story and suggest improvements.")

topic = "A robot learning to love."
print(f"Topic: {topic}\n")

# Step 1: Writer writes
writer.receive_message("User", f"Write a story about: {topic}")
story = writer.generate_reply()
print(f"--- Writer ---\n{story}\n")

# Step 2: Critic critiques
critic.receive_message("Writer", story)
feedback = critic.generate_reply()
print(f"--- Critic ---\n{feedback}\n")

# Step 3: Writer improves
writer.receive_message("Critic", feedback)
final_story = writer.generate_reply()
print(f"--- Writer (Final) ---\n{final_story}\n")
```

### 3. The "Handoff" Pattern (Router)

Implementing a Triage agent that routes to sub-agents.

```python
triage_prompt = """
You are a Triage Agent. 
If the user asks about Tech, output 'TRANSFER: TECH'.
If the user asks about Sales, output 'TRANSFER: SALES'.
Otherwise, answer yourself.
"""

tech_agent = Agent("Tech", "You are Tech Support.")
sales_agent = Agent("Sales", "You are Sales Support.")

def run_router(user_input):
    triage = Agent("Triage", triage_prompt)
    triage.receive_message("User", user_input)
    response = triage.generate_reply()
    
    if "TRANSFER: TECH" in response:
        print(">> Transferring to Tech Agent...")
        tech_agent.receive_message("User", user_input) # Pass context
        return tech_agent.generate_reply()
        
    elif "TRANSFER: SALES" in response:
        print(">> Transferring to Sales Agent...")
        sales_agent.receive_message("User", user_input)
        return sales_agent.generate_reply()
        
    else:
        return response

# Usage
# print(run_router("My laptop is broken"))
```

### 4. Shared State (The Blackboard)

Instead of passing huge strings, agents read/write to a shared dict.

```python
blackboard = {
    "code": None,
    "tests": None,
    "review": None
}

def coder_step():
    blackboard["code"] = "print('Hello')"

def tester_step():
    code = blackboard["code"]
    if code:
        blackboard["tests"] = "assert code runs"

# This decouples the agents. The Tester doesn't need to talk to the Coder,
# it just needs to read the Blackboard.
```

### Summary

At its core, Multi-Agent Orchestration is just **Message Passing** and **State Management**. Frameworks like AutoGen and LangGraph simply provide abstractions (Classes, Graphs, State Machines) to manage this complexity so you don't have to write `receive_message` calls manually.
