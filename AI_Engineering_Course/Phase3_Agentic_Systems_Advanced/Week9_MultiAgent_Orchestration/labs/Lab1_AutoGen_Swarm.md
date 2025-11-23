# Lab 1: AutoGen Swarm

## Objective
Create a multi-agent chat using Microsoft's **AutoGen**.
We will have a **UserProxy** (Admin), a **Coder**, and a **Product Manager**.

## 1. Setup

```bash
poetry add pyautogen
```

## 2. The Swarm (`swarm.py`)

```python
import autogen

config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# 1. Define Agents
llm_config = {"config_list": config_list, "seed": 42}

user_proxy = autogen.UserProxyAgent(
   name="User_Proxy",
   system_message="A human admin.",
   code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
   human_input_mode="TERMINATE"
)

coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="You are a Python Coder. Write code to solve the task."
)

pm = autogen.AssistantAgent(
    name="Product_Manager",
    llm_config=llm_config,
    system_message="You are a PM. Review the code and suggest improvements."
)

# 2. Define Group Chat
groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# 3. Start Chat
user_proxy.initiate_chat(
    manager,
    message="Write a Python script to plot a sine wave and save it to sine.png."
)
```

## 3. Running the Lab

1.  Create `OAI_CONFIG_LIST` with your API key.
2.  Run `python swarm.py`.
3.  Watch the agents talk.
    *   Coder writes code.
    *   UserProxy executes it (if Docker is configured) or you execute it manually.
    *   PM critiques it.

## 4. Submission
Submit the chat log.
