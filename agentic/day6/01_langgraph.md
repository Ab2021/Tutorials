# Day 6: Diving into Agentic Frameworks: LangGraph

Welcome to the first of our deep dives into specific agentic frameworks. Today, we'll be exploring **LangGraph**, a powerful library from the creators of LangChain for building complex, stateful, and multi-agent applications.

## Morning Session (9:00 AM - 12:00 PM): Introduction to LangGraph

### Topic 1: What is LangGraph and Why Use It?

LangGraph is a library that allows you to build agentic and multi-agent applications as **graphs**. It extends the declarative, composable nature of LangChain with the ability to create cyclical and stateful workflows, which are essential for building sophisticated agents.

**Why LangGraph?**

While you can build simple agents with LangChain alone, LangGraph excels when you need to:

*   **Create cyclical workflows:** In many agentic systems, you need the ability to loop, reflect, and retry. For example, an agent might need to try a tool, realize it failed, and then loop back to the reasoning step to try a different approach. LangGraph is designed to handle these kinds of cyclical graphs.
*   **Manage complex state:** LangGraph provides a robust way to manage the state of your application as it flows through the graph. This is crucial for building agents that can remember past actions and adapt their behavior accordingly.
*   **Build multi-agent systems:** LangGraph is an excellent choice for orchestrating collaboration between multiple specialized agents.
*   **Add human-in-the-loop checkpoints:** LangGraph makes it easy to pause the execution of a graph and wait for human input, which is essential for many real-world applications.

### Topic 2: The Core Concepts of LangGraph

At its core, a LangGraph application is a **state machine**. The application's state is passed between nodes in a graph, and each node can modify the state. The three core concepts are:

1.  **State:** The state is a central, shared data structure that is passed between all the nodes in the graph. It's typically a Python dictionary or a Pydantic class. It holds all the information that the agents need to do their work, such as the conversation history, the current plan, and the results of any tool calls.
2.  **Nodes:** A node is a function or a callable class that represents a unit of work in the graph. Each node receives the current state as input and can return a modified state. A node could be a function that calls an LLM, a function that executes a tool, or any other piece of Python code.
3.  **Edges:** An edge is a connection between two nodes that defines the flow of control. There are two main types of edges:
    *   **Standard Edges:** These are simple, direct connections. After a node is executed, the state is passed to the next node in the graph.
    *   **Conditional Edges:** These allow you to add branching logic to your graph. A conditional edge takes a function that inspects the current state and decides which node to go to next. This is how you can implement loops and other complex control flows.

### Exercise

1.  **Conceptualize a simple multi-step process (e.g., making a cup of tea) as a LangGraph graph.**
    *   What would the **state** look like? (e.g., `{'water_boiled': False, 'tea_steeped': False, 'milk_added': False}`)
    *   What would the **nodes** be? (e.g., `boil_water`, `steep_tea`, `add_milk`)
    *   What would the **edges** be? (e.g., after `boil_water`, go to `steep_tea`).

## Afternoon Session (1:00 PM - 4:00 PM): Building a Simple LangGraph Application

### Topic 3: Setting up Your LangGraph Environment

Before we can start building, we need to set up our environment.

1.  **Installation:** You'll need to install the `langchain` and `langgraph` libraries. You can do this with pip:
    ```bash
    pip install langchain langgraph
    ```
2.  **LLM API Key:** You'll also need an API key for an LLM provider, such as OpenAI. You'll need to set this as an environment variable.

### Topic 4: Building a Basic ReAct Agent with LangGraph

Let's build a simple ReAct-style agent that can use a search tool to answer questions. This will give you a hands-on feel for how LangGraph works.

Here's a high-level overview of the steps:

1.  **Define the State:** We'll define a state object that includes the user's query, the agent's thoughts, and the results of any tool calls.
2.  **Define the Nodes:** We'll create two main nodes:
    *   `reason`: This node will call the LLM to get the agent's next thought and action.
    *   `act`: This node will execute the action (e.g., call the search tool).
3.  **Define the Edges:** We'll use a conditional edge to decide what to do after the `reason` node. If the agent decides to use a tool, we'll go to the `act` node. If the agent is ready to give a final answer, we'll end the graph.
4.  **Compile and Run the Graph:** We'll use LangGraph's `StateGraph` class to compile our graph and then run it with a user's query.

### Exercise

1.  **Follow a tutorial to build and run a simple ReAct agent using LangGraph.**
    *   There are many excellent tutorials available on the LangChain and LangGraph documentation websites.
    *   Start with a simple example that uses a search tool.
2.  **Experiment with the agent.**
    *   Ask it a few different questions and observe its reasoning process.
    *   Try to identify the limitations of this simple agent. What happens if the search tool fails? What if the agent gets stuck in a loop?

This exercise will give you a solid foundation for building more complex and robust agents with LangGraph.

## Advanced LangGraph Concepts

Beyond the basics of states, nodes, and edges, LangGraph has a few more advanced concepts that are important to understand as you build more complex applications:

*   **Durable Execution:** LangGraph supports **durable execution**, which means that an agent can persist through failures and run for extended periods. You can save the state of a graph at any point and then resume it later. This is crucial for building robust applications that can handle errors and interruptions.
*   **State Management:** LangGraph provides a few different ways to manage the state of your application. You can use a simple dictionary, or you can use a more structured approach with **state schemas** (defined using Pydantic or TypedDict) and **reducers**. A reducer is a function that defines how the output of a node should be applied to the current state.
*   **Message Passing:** LangGraph's underlying algorithm is inspired by **Pregel**, a system for large-scale graph processing. It uses a message-passing model where nodes send messages to each other along the edges of the graph. This allows for a high degree of parallelism and scalability.
*   **LangGraph Studio:** For those who prefer a more visual approach to development, LangChain offers **LangGraph Studio**, a graphical user interface for designing and building LangGraph workflows. This can be a great way to get started with LangGraph and to visualize the structure of your agentic systems.

## A Simple Code Example

Here is a small code snippet to illustrate the core concepts of LangGraph:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. Define the state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 2. Define the nodes
def call_model(state):
    messages = state['messages']
    # In a real application, you would call an LLM here
    response = "Hello! This is a response from the model."
    return {"messages": [response]}

# 3. Define the graph
workflow = StateGraph(AgentState)

# 4. Add the nodes
workflow.add_node("model", call_model)

# 5. Add the edges
workflow.set_entry_point("model")
workflow.add_edge("model", END)

# 6. Compile the graph
app = workflow.compile()

# 7. Run the graph
inputs = {"messages": ["Hello!"]}
for output in app.stream(inputs):
    print(output)
```
This example shows a very simple graph with a single node that calls a "model" (in this case, just a function that returns a hardcoded string). The state is a simple dictionary with a list of messages. The graph has a single edge from the "model" node to the end. This is the basic building block of any LangGraph application.


