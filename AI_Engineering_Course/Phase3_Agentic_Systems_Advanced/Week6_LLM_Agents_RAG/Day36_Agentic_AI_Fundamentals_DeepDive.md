# Day 36: Agentic AI Fundamentals
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. The Agent Loop: Formal Definition

**Markov Decision Process (MDP) Formulation:**
An agent operates in an environment defined by:
- **States** $S$: All possible environment configurations.
- **Actions** $A$: All possible agent actions.
- **Transition** $T(s'|s,a)$: Probability of reaching state $s'$ from state $s$ after action $a$.
- **Reward** $R(s,a)$: Immediate reward for taking action $a$ in state $s$.
- **Policy** $\pi(a|s)$: Agent's strategy (probability of action $a$ given state $s$).

**Objective:**
$$ \max_\pi \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)\right] $$
where $\gamma \in [0,1]$ is the discount factor.

### 2. ReAct: Reasoning and Acting

**Prompt Template:**
```
You run in a loop of Thought, Action, Observation.
At the end of the loop, you output an Answer.

Use Thought to describe your reasoning about the question.
Use Action to run one of the actions available to you.
Observation will be the result of running those actions.

Available actions:
- search[query]: Search the web
- calculate[expression]: Evaluate a math expression
- finish[answer]: Return the final answer

Question: {question}
Thought:
```

**Example Execution:**
```
Question: What is the population of the capital of France?

Thought 1: I need to find the capital of France first.
Action 1: search["capital of France"]
Observation 1: The capital of France is Paris.

Thought 2: Now I need to find the population of Paris.
Action 2: search["population of Paris"]
Observation 2: The population of Paris is approximately 2.2 million.

Thought 3: I have the answer.
Action 3: finish["2.2 million"]
```

### 3. Tool Use: Function Calling Implementation

**OpenAI Function Calling:**
```python
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["C", "F"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Extract function call
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# Execute function
result = get_weather(**arguments)

# Send result back to model
messages.append(response.choices[0].message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result)
})

final_response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
)
```

### 4. Planning Algorithms

**Hierarchical Task Network (HTN):**
- Decompose high-level tasks into subtasks recursively.
- **Example:** "Book a trip" -> ["Book flight", "Book hotel", "Rent car"]

**Monte Carlo Tree Search (MCTS):**
- Explore multiple action sequences.
- Simulate outcomes and select the best path.
- **Used in:** AlphaGo, game-playing agents.

**A* Search:**
- Find the optimal path from start to goal.
- **Heuristic:** Estimate cost-to-goal.
- **Used in:** Navigation, puzzle solving.

### 5. Memory Systems

**Short-Term Memory (STM):**
- Current conversation context (last N turns).
- **Implementation:** Sliding window over messages.

**Long-Term Memory (LTM):**
- Persistent storage of past experiences.
- **Implementation:** Vector database (Pinecone, Weaviate).

**Working Memory:**
- Intermediate reasoning steps.
- **Implementation:** Scratchpad in the prompt.

**Memory Retrieval:**
$$ \text{Relevant Memories} = \text{TopK}(\text{Similarity}(\text{Query}, \text{Memory})) $$

### 6. Reflection and Self-Critique

**Reflexion Algorithm:**
1. **Act:** Execute a plan.
2. **Evaluate:** Check if the goal was achieved.
3. **Reflect:** If failed, analyze what went wrong.
4. **Revise:** Update the plan based on reflection.
5. **Retry:** Execute the revised plan.

**Example:**
```
Attempt 1: search["Paris population"] -> No results
Reflection: The query was too vague.
Revised Plan: search["population of Paris France 2024"]
Attempt 2: Success!
```

### Code: Complete ReAct Agent

```python
import openai
import re

class ReActAgent:
    def __init__(self, tools):
        self.tools = tools
        self.max_iterations = 10
    
    def run(self, question):
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        messages.append({"role": "user", "content": f"Question: {question}"})
        
        for i in range(self.max_iterations):
            # Get model response
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )
            
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            
            # Check if finished
            if "finish[" in content:
                answer = re.search(r'finish\[(.*?)\]', content).group(1)
                return answer
            
            # Extract action
            action_match = re.search(r'Action \d+: (\w+)\[(.*?)\]', content)
            if not action_match:
                break
            
            action_name = action_match.group(1)
            action_input = action_match.group(2)
            
            # Execute action
            if action_name in self.tools:
                observation = self.tools[action_name](action_input)
            else:
                observation = f"Error: Unknown action {action_name}"
            
            # Add observation
            messages.append({"role": "user", "content": f"Observation {i+1}: {observation}"})
        
        return "Failed to find answer"
    
    def get_system_prompt(self):
        return """You run in a loop of Thought, Action, Observation.
At the end, output Answer using finish[answer].

Available actions:
- search[query]: Search the web
- calculate[expression]: Evaluate math
- finish[answer]: Return final answer

Format:
Thought 1: <reasoning>
Action 1: <action>[<input>]
(wait for Observation)
Thought 2: <reasoning>
..."""

# Define tools
def search(query):
    # Mock implementation
    return f"Search results for: {query}"

def calculate(expr):
    try:
        return str(eval(expr))
    except:
        return "Error in calculation"

# Run agent
agent = ReActAgent({"search": search, "calculate": calculate})
answer = agent.run("What is 25 * 37?")
print(answer)
```

### 7. Multi-Agent Collaboration

**Debate Pattern:**
- Two agents argue different sides.
- A judge agent selects the better answer.

**Consensus Pattern:**
- Multiple agents propose solutions.
- Aggregate via voting or averaging.

**Specialization Pattern:**
- Each agent has a specific role (researcher, coder, writer).
- Coordinator agent orchestrates.
