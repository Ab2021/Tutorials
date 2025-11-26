# Lab: Day 45 - Build an Agent

## Goal
Create an Agent that can use tools.

## Prerequisites
- `pip install langchain langchain-openai`

## Step 1: The Code (`agent.py`)

```python
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Define Tools
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

tools = [get_word_length, add_numbers]

# 2. Model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You can calculate word lengths and add numbers."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Agent
agent = create_openai_tools_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Run
print("--- Agent Run ---")
query = "What is the length of 'Antigravity' plus the length of 'Backend'?"
response = agent_executor.invoke({"input": query})
print(f"Final Answer: {response['output']}")
```

## Step 2: Run It
`python agent.py`

*   **Observe the Output (Verbose)**:
    1.  Thought: I need to find the length of 'Antigravity'.
    2.  Action: `get_word_length('Antigravity')` -> 11.
    3.  Thought: I need to find the length of 'Backend'.
    4.  Action: `get_word_length('Backend')` -> 7.
    5.  Thought: I need to add 11 and 7.
    6.  Action: `add_numbers(11, 7)` -> 18.
    7.  Final Answer: 18.

## Challenge
Add a **Search Tool**.
Use `DuckDuckGoSearchRun` (from `langchain_community.tools`).
Ask the agent: "Who won the Super Bowl in 2024?"
It should search the web and answer.
