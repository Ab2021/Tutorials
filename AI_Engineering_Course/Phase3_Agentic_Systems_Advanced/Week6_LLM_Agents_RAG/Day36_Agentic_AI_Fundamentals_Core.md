# Day 36: Agentic AI Fundamentals
## Core Concepts & Theory

### From Chatbots to Agents

**Traditional LLMs (Chatbots):**
- Respond to user queries.
- Single-turn or multi-turn conversations.
- **Limitation:** Cannot take actions in the world.

**Agentic AI:**
- Autonomous systems that can:
  - **Perceive:** Observe the environment.
  - **Reason:** Plan and make decisions.
  - **Act:** Execute actions using tools.
  - **Learn:** Improve from feedback.

### 1. The Agent Loop

**Basic Agent Architecture:**
```
1. Observe: Get current state (user query, environment)
2. Think: Reason about what to do (planning)
3. Act: Execute an action (tool call, response)
4. Repeat: Continue until task is complete
```

**ReAct Pattern (Reasoning + Acting):**
- **Thought:** "I need to find the current weather in Paris."
- **Action:** `search("Paris weather")`
- **Observation:** "Temperature: 18°C, Sunny"
- **Thought:** "Now I can answer the user."
- **Answer:** "The weather in Paris is 18°C and sunny."

### 2. Agent Components

**A. Perception:**
- **Input:** User query, environment state, previous actions.
- **Processing:** Parse and understand the current situation.

**B. Memory:**
- **Short-term:** Current conversation context.
- **Long-term:** Past experiences, learned knowledge.
- **Working Memory:** Intermediate reasoning steps.

**C. Planning:**
- **Task Decomposition:** Break complex tasks into subtasks.
- **Strategy Selection:** Choose the best approach.
- **Execution Order:** Determine sequence of actions.

**D. Action:**
- **Tool Use:** Call external APIs, databases, calculators.
- **Response Generation:** Formulate natural language responses.

**E. Reflection:**
- **Self-Critique:** Evaluate own performance.
- **Error Correction:** Retry or adjust strategy if failed.

### 3. Tool Use (Function Calling)

**Function Schema:**
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  }
}
```

**Agent Workflow:**
1. **User:** "What's the weather in Tokyo?"
2. **Agent Thinks:** "I need to call get_weather function."
3. **Agent Generates:** `get_weather(location="Tokyo", unit="celsius")`
4. **System Executes:** Returns `{"temp": 22, "condition": "Cloudy"}`
5. **Agent Responds:** "The weather in Tokyo is 22°C and cloudy."

### 4. Planning Strategies

**Zero-Shot Planning:**
- Agent plans on-the-fly without examples.
- **Prompt:** "To solve this, I will..."

**Few-Shot Planning:**
- Provide examples of task decomposition.
- **Example:** "Task: Book a flight. Steps: 1. Search flights, 2. Compare prices, 3. Book ticket."

**Chain-of-Thought Planning:**
- Explicit step-by-step reasoning.
- **Prompt:** "Let's break this down step by step..."

**Tree-of-Thoughts Planning:**
- Explore multiple possible plans.
- Evaluate and select the best one.

### 5. Agent Frameworks

**LangChain:**
- Python framework for building LLM applications.
- **Components:** Chains, Agents, Tools, Memory.
- **Use Case:** RAG, chatbots, agents.

**AutoGPT:**
- Autonomous agent that sets its own goals.
- **Loop:** Goal -> Plan -> Execute -> Reflect -> Repeat.
- **Limitation:** Can get stuck in loops, expensive.

**BabyAGI:**
- Task-driven autonomous agent.
- **Components:** Task creation, prioritization, execution.

**LlamaIndex:**
- Framework for data-augmented LLM applications.
- **Focus:** Indexing, retrieval, query engines.

### 6. Agent Types

**Reactive Agents:**
- Simple stimulus-response.
- No planning or memory.
- **Example:** Rule-based chatbot.

**Deliberative Agents:**
- Plan before acting.
- Maintain internal state.
- **Example:** ReAct agent.

**Hybrid Agents:**
- Combine reactive and deliberative.
- Fast reactions + strategic planning.
- **Example:** Game-playing AI.

**Multi-Agent Systems:**
- Multiple agents collaborate.
- **Example:** Debate, consensus, specialization.

### 7. Challenges in Agentic AI

**Reliability:**
- Agents can make mistakes (wrong tool, bad reasoning).
- **Mitigation:** Validation, human-in-the-loop.

**Cost:**
- Multiple LLM calls per task.
- **Mitigation:** Caching, smaller models for simple tasks.

**Latency:**
- Sequential tool calls add delay.
- **Mitigation:** Parallel execution, streaming.

**Safety:**
- Agents can take harmful actions.
- **Mitigation:** Sandboxing, approval gates.

**Evaluation:**
- Hard to measure agent performance.
- **Metrics:** Task success rate, efficiency, cost.

### Real-World Examples

**ChatGPT Plugins (2023):**
- Agents that can browse web, run code, access databases.
- **Tools:** Web search, Python interpreter, Wolfram Alpha.

**Copilot (GitHub):**
- Code generation agent.
- **Tools:** Code search, documentation, unit tests.

**Devin (Cognition AI):**
- Autonomous software engineer.
- **Tools:** Terminal, browser, code editor.

**GPT-4 with Advanced Data Analysis:**
- Data science agent.
- **Tools:** Python, pandas, matplotlib.

### Summary Table

| Component | Purpose | Example |
| :--- | :--- | :--- |
| **Perception** | Understand state | Parse user query |
| **Memory** | Store context | Conversation history |
| **Planning** | Decide actions | Task decomposition |
| **Action** | Execute | Call API, generate text |
| **Reflection** | Self-improve | Critique and retry |

### Next Steps
In the Deep Dive, we will implement a complete ReAct agent from scratch and analyze different planning strategies.
