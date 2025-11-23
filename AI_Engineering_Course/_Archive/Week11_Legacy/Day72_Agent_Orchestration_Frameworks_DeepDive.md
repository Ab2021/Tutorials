# Day 72: Agent Orchestration Frameworks
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. LangGraph Cyclic Agent Implementation

Building a "Reason-Act" loop with state persistence.

```python
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    iterations: int

# 2. Define Nodes
def agent_node(state):
    print("---Agent Thinking---")
    messages = state['messages']
    # Mock LLM call
    # response = llm.invoke(messages)
    response = AIMessage(content="I need to use a tool.") 
    return {"messages": [response], "iterations": state['iterations'] + 1}

def tool_node(state):
    print("---Executing Tool---")
    # Mock Tool Execution
    tool_result = HumanMessage(content="Tool Result: 42", name="calculator")
    return {"messages": [tool_result]}

# 3. Define Conditional Logic
def should_continue(state):
    last_message = state['messages'][-1]
    if "tool" in last_message.content:
        return "tools"
    if state['iterations'] > 3:
        return END
    return END

# 4. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent") # Loop back

# 5. Compile
app = workflow.compile()

# 6. Run
inputs = {"messages": [HumanMessage(content="Calculate 2+2")], "iterations": 0}
for output in app.stream(inputs):
    pass
```

### 2. AutoGen Multi-Agent Chat

Setting up a User Proxy and an Assistant.

```python
from autogen import AssistantAgent, UserProxyAgent

# 1. Define Agents
assistant = AssistantAgent(
    name="coder",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="TERMINATE"
)

# 2. Start Chat
# The user proxy initiates the chat, asking the assistant to solve a task.
# The assistant writes code.
# The user proxy executes the code and returns the output.
# The loop continues until the task is done.

# user_proxy.initiate_chat(
#     assistant,
#     message="Plot a chart of NVDA stock price YTD."
# )
```

### 3. CrewAI Task Delegation

Defining Roles and Tasks.

```python
from crewai import Agent, Task, Crew

# 1. Agents
researcher = Agent(
    role='Researcher',
    goal='Uncover new AI trends',
    backstory='You are a senior analyst.',
    tools=[]
)

writer = Agent(
    role='Writer',
    goal='Write a blog post',
    backstory='You are a tech blogger.',
    verbose=True
)

# 2. Tasks
task1 = Task(
    description='Research the latest AI agents.',
    agent=researcher
)

task2 = Task(
    description='Write a blog post based on research.',
    agent=writer
)

# 3. Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2
)

# result = crew.kickoff()
```

### 4. Semantic Kernel Planner (Stepwise)

Using a planner to auto-chain functions.

```python
import semantic_kernel as sk
from semantic_kernel.planning import SequentialPlanner

kernel = sk.Kernel()

# Import Skills (Plugins)
# kernel.import_skill(MathSkill(), "math")
# kernel.import_skill(TextSkill(), "text")

# Create Planner
planner = SequentialPlanner(kernel)

# Ask
goal = "Calculate 5+5 and write a poem about the result."

# Plan
# plan = await planner.create_plan_async(goal)

# Execute
# result = await plan.invoke_async()
```

### 5. LangGraph Checkpointing (Time Travel)

Persisting state to allow resumption.

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("checkpoints.db")
memory = SqliteSaver(conn)

# Compile with checkpointer
app = workflow.compile(checkpointer=memory)

# Run with thread_id
config = {"configurable": {"thread_id": "1"}}
# app.invoke(inputs, config=config)

# Resume / Inspect
# state = app.get_state(config)
# print(state.values)
```
