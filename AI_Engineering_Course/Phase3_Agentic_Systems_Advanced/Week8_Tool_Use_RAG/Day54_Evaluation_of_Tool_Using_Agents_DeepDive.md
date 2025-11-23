# Day 54: Evaluation of Tool-Using Agents
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Building an Agent Evaluation Harness

We will build a system to evaluate our "Math Agent" using **Trajectory Scoring**.

### 1. The Agent to Evaluate

A simple ReAct agent that has a calculator tool.

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True
)

# The `return_intermediate_steps=True` is crucial. 
# It gives us the Trajectory.
```

### 2. The Evaluator (LLM-as-a-Judge)

We write a prompt that asks GPT-4 to grade the agent's logic.

```python
EVAL_PROMPT = """
You are an expert agent evaluator.
Your job is to grade the performance of an AI agent based on its trajectory.

Goal: {input}
Trajectory:
{trajectory}

Criteria:
1. Tool Selection: Did the agent pick the right tool?
2. Arguments: Were the inputs to the tool correct?
3. Efficiency: Did the agent solve it in the minimum steps?

Output a score from 1-5 and a brief reasoning.
Format:
Score: [1-5]
Reasoning: ...
"""

def evaluate_trajectory(input_text, steps, final_answer):
    # Format trajectory string
    traj_str = ""
    for action, observation in steps:
        traj_str += f"Action: {action.tool}({action.tool_input})\n"
        traj_str += f"Observation: {observation}\n"
    traj_str += f"Final Answer: {final_answer}\n"
    
    # Call Judge
    response = llm.invoke(EVAL_PROMPT.format(input=input_text, trajectory=traj_str))
    return response.content

# Usage
# result = agent.invoke({"input": "What is 25 * 4?"})
# score = evaluate_trajectory("What is 25 * 4?", result["intermediate_steps"], result["output"])
# print(score)
```

### 3. Creating a Synthetic Test Set

Using GPT-4 to generate test cases.

```python
GENERATOR_PROMPT = """
Generate 5 test cases for a Math Agent.
The cases should involve arithmetic, percentages, and word problems.
Format: JSON list of {"question": "...", "expected_answer": "..."}
"""

def generate_test_set():
    response = llm.invoke(GENERATOR_PROMPT)
    # Parse JSON...
    return test_cases
```

### 4. Deterministic Tool Mocking

To test the agent's logic without calling real APIs (which might be slow or change), we mock the tools.

```python
from unittest.mock import MagicMock

# Mock the Calculator
mock_calc = MagicMock()
mock_calc.run.return_value = "100"

# Inject into Agent
tools[0].func = mock_calc.run

# Run Agent
agent.invoke("What is 10 * 10?")

# Assert
mock_calc.run.assert_called_with("10 * 10")
```

### 5. LangSmith Evaluation (The Pro Way)

LangSmith automates this.

```python
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

client = Client()

# 1. Create Dataset in LangSmith
dataset_name = "Math Agent Test"
# client.create_dataset(dataset_name) ...

# 2. Define Eval Config
eval_config = RunEvalConfig(
    evaluators=[
        "qa", # Correctness
        "context_qa", # Hallucination check
        "cot_qa" # Chain of Thought grading
    ]
)

# 3. Run
# run_on_dataset(
#     client=client,
#     dataset_name=dataset_name,
#     llm_or_chain_factory=agent,
#     evaluation=eval_config,
# )
```

### Summary

Building an eval harness involves:
1.  **Instrumentation:** Capturing the trajectory (`intermediate_steps`).
2.  **Grading:** Using a stronger model to score the trajectory.
3.  **Mocking:** Isolating the agent logic from external API flakiness.
4.  **Automation:** Running this on every commit (CI/CD).
