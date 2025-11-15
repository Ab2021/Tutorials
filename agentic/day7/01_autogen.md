# Day 7: Diving into Agentic Frameworks: AutoGen

Today, we'll explore **AutoGen**, a powerful open-source framework from Microsoft for building and managing multi-agent AI applications. AutoGen simplifies the orchestration, automation, and optimization of complex LLM workflows, making it easier to build sophisticated applications where multiple agents collaborate to solve problems.

## Morning Session (9:00 AM - 12:00 PM): Introduction to AutoGen

### Topic 1: What is AutoGen and Why Use It?

AutoGen is a framework that enables you to create and manage a team of AI agents that can communicate and cooperate to perform tasks. It provides a high-level abstraction for building multi-agent systems, allowing you to focus on the logic of your agents and the flow of the conversation, rather than the low-level details of orchestration.

**Why AutoGen?**

*   **Simplified Multi-Agent Workflows:** AutoGen makes it easy to define and manage conversations between multiple agents, each with its own role and capabilities.
*   **Enhanced LLM Performance:** By combining the strengths of multiple specialized agents, you can often achieve better performance than you would with a single, monolithic agent.
*   **Flexible and Extensible:** AutoGen is highly customizable. You can create your own specialized agents, integrate with external tools, and define custom conversation patterns.
*   **Human-in-the-Loop Integration:** AutoGen provides built-in support for integrating human feedback into the conversation, allowing you to create systems that are both autonomous and controllable.

### Topic 2: The Core Concepts of AutoGen

AutoGen is built around a few core concepts:

1.  **Agents:** An agent is an entity that can send and receive messages. Agents can be powered by LLMs, by code, or by human input. AutoGen provides a few built-in agent types, but you can also create your own custom agents.
2.  **Conversations:** A conversation is a sequence of messages exchanged between agents. AutoGen manages the conversation history and ensures that each agent has the context it needs to participate in the conversation.
3.  **Multi-Agent Collaboration:** AutoGen allows you to define different patterns of collaboration between agents. For example, you can have a simple two-agent conversation, or you can have a more complex "group chat" where multiple agents can contribute to the conversation.

### Exercise

1.  **Design a simple multi-agent system on paper.**
    *   Imagine you want to create a system that can write a blog post about a given topic.
    *   Define the roles of at least two agents (e.g., a "researcher" agent that finds information and a "writer" agent that composes the blog post).
    *   What messages would these agents exchange to collaborate on the task?

## Afternoon Session (1:00 PM - 4:00 PM): Building a Simple AutoGen Application

### Topic 3: Setting up Your AutoGen Environment

1.  **Installation:** You can install AutoGen with pip:
    ```bash
    pip install pyautogen
    ```
2.  **LLM Configuration:** You'll need to configure AutoGen to use an LLM. This typically involves setting an environment variable with your API key and creating a configuration file that specifies which model to use.

### Topic 4: Building a Basic Multi-Agent System with AutoGen

Let's build a simple two-agent system to solve a coding problem. This will give you a feel for the basic workflow of an AutoGen application.

Here's a high-level overview:

1.  **Define the Agents:** We'll create two agents:
    *   A **`UserProxyAgent`**: This agent will act as a proxy for the human user. It will get the problem description from the user and then pass it to the worker agent.
    *   An **`AssistantAgent`**: This agent will be powered by an LLM and will be responsible for solving the coding problem.
2.  **Initiate the Conversation:** We'll initiate a conversation between the two agents, giving the `AssistantAgent` the problem to solve.
3.  **Observe the Collaboration:** We'll then observe as the two agents collaborate to solve the problem. The `AssistantAgent` will write the code, and the `UserProxyAgent` will execute the code and report the results back to the `AssistantAgent`. This process will continue until the problem is solved.

### Exercise

1.  **Follow a tutorial to build and run a simple two-agent system using AutoGen.**
    *   The official AutoGen documentation has excellent tutorials that walk you through the process of building a simple coding assistant.
2.  **Experiment with the system.**
    *   Give it a few different coding problems to solve.
    *   Observe the conversation between the agents. How do they collaborate? What happens when the code has a bug?

This exercise will give you a solid understanding of the core concepts of AutoGen and how to use it to build your own multi-agent applications.

## Advanced AutoGen Concepts

As you get more comfortable with AutoGen, you can start to explore some of its more advanced features:

*   **The AutoGen Ecosystem:** AutoGen is more than just a library; it's a complete ecosystem for building and managing agentic applications. The ecosystem includes:
    *   **AutoGen Core:** The low-level framework for creating and managing multi-agent systems.
    *   **AutoGen AgentChat:** A high-level framework that simplifies the process of building multi-agent conversational workflows.
    *   **AutoGen Studio:** A low-code, user-friendly interface for rapidly prototyping, configuring, and testing AI agents with a drag-and-drop interface.
*   **Advanced Conversation Patterns:** AutoGen supports a variety of sophisticated conversation patterns beyond a simple two-agent chat. These include:
    *   **Sequential Chats:** You can create a sequence of chats, where the output of one chat becomes the input for the next. This is useful for building complex, multi-step workflows.
    *   **Nested Chats:** You can nest chats within each other to create hierarchical conversations. This is useful for breaking down a complex problem into a set of smaller, more manageable sub-problems.

By leveraging these advanced features, you can build highly sophisticated and capable multi-agent systems with AutoGen.

## A Simple Code Example

Here is a small code snippet to illustrate the core concepts of AutoGen:

```python
import autogen

# In a real application, you would configure your LLM here
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model": ["gpt-4"]},
)

# 1. Define the agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "coding"},
    llm_config={"config_list": config_list},
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

# 2. Initiate the conversation
user_proxy.initiate_chat(
    assistant,
    message="""Write a Python function to print the first 10 Fibonacci numbers."""
)
```
This example shows a simple two-agent system. The `UserProxyAgent` gets the task from the user and then passes it to the `AssistantAgent`. The `AssistantAgent` writes the code, and the `UserProxyAgent` executes it. This back-and-forth conversation continues until the task is completed.


