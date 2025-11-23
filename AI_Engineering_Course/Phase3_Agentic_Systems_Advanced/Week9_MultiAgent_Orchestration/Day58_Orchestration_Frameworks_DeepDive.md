# Day 58: Orchestration Frameworks (AutoGen & CrewAI)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing AutoGen and CrewAI

We will build a **Research Team** using both frameworks to see the difference.

### 1. AutoGen Implementation

AutoGen uses a "Group Chat" model.

```python
import autogen

config_list = [{"model": "gpt-4", "api_key": "..."}]

# 1. Define Agents
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

researcher = autogen.AssistantAgent(
    name="Researcher",
    llm_config={"config_list": config_list},
    system_message="Research the topic on the web."
)

writer = autogen.AssistantAgent(
    name="Writer",
    llm_config={"config_list": config_list},
    system_message="Write a blog post based on the research."
)

# 2. Define Group Chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, writer], 
    messages=[], 
    max_round=10
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# 3. Start Chat
user_proxy.initiate_chat(
    manager,
    message="Research the latest AI trends and write a blog post."
)
```

**What happens internally?**
1.  User sends message to Manager.
2.  Manager (LLM) looks at the message and the agent list.
3.  Manager selects `Researcher` to speak.
4.  Researcher generates output.
5.  Manager selects `Writer` to speak.
6.  Writer generates output.
7.  Manager selects `User_Proxy`.
8.  User_Proxy (if configured) executes code or asks for human input.

### 2. CrewAI Implementation

CrewAI uses a "Task-Based" model.

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# 1. Define Agents
researcher = Agent(
    role='Researcher',
    goal='Uncover latest AI trends',
    backstory='You are an expert analyst.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

writer = Agent(
    role='Tech Writer',
    goal='Write compelling content',
    backstory='You are a tech journalist.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 2. Define Tasks
task1 = Task(
    description='Research AI trends in 2024.',
    agent=researcher,
    expected_output='A bulleted list of trends.'
)

task2 = Task(
    description='Write a blog post using the research.',
    agent=writer,
    expected_output='A 500-word blog post.'
)

# 3. Define Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential # Explicit order
)

# 4. Kickoff
result = crew.kickoff()
print(result)
```

**Difference:**
*   **AutoGen:** Dynamic. The Manager *decides* who speaks. The Researcher might speak 3 times in a row if needed.
*   **CrewAI:** Sequential. Task 1 *must* finish before Task 2 starts. The output of Task 1 is automatically passed as input to Task 2.

### 3. LangGraph Implementation (State Machine)

LangGraph requires defining the graph explicitly.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# Nodes
def researcher_node(state):
    return {"messages": ["Researching..."]}

def writer_node(state):
    return {"messages": ["Writing..."]}

# Graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()
# app.invoke({"messages": ["Start"]})
```

### Summary

*   **AutoGen:** Best for "Swarm" intelligence where the path is unknown.
*   **CrewAI:** Best for "Assembly Line" work where the process is fixed.
*   **LangGraph:** Best for "Engineering" where you need total control over state and transitions.
