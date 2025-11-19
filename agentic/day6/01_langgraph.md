# Day 6: An Expert's Guide to LangGraph

## 1. The Philosophy of LangGraph: Graphs as a Model for Computation

LangGraph is built on a simple but powerful idea: that any complex computation can be represented as a **graph**. In this model, the nodes of the graph represent units of computation, and the edges represent the flow of data between them.

This is a departure from the linear, "chain" model of computation that is used in LangChain. While chains are good for simple, sequential workflows, they are not well-suited for more complex applications that require cycles, branching, and state management.

LangGraph's graph-based model provides a much more flexible and powerful way to build agentic systems. It allows you to create complex, cyclical workflows where agents can reason, act, and reflect in a continuous loop.

## 2. A Deep Dive into LangGraph's Core Components

*   **StateGraph:** The `StateGraph` class is the main entry point for building a LangGraph application. It is a state machine where the state is passed between the nodes of the graph.
*   **Nodes and Edges:**
    *   **Nodes:** A node is a function or a callable class that performs a unit of work.
    *   **Edges:** An edge is a connection between two nodes. LangGraph supports three types of edges:
        *   **Standard Edges:** The output of one node is passed to the next.
        *   **Conditional Edges:** The next node is chosen based on a condition.
        *   **End Edges:** The graph finishes execution.
*   **State Management:** The state of a LangGraph application is typically a `TypedDict`. You can use `operator.add` to specify that the output of a node should be added to the state, rather than replacing it. This is useful for managing lists of messages.

## 3. Building a Multi-Agent System with LangGraph

LangGraph is an excellent choice for building multi-agent systems. A common pattern is to have a "supervisor" agent that orchestrates a team of "worker" agents.

The supervisor receives a high-level goal, breaks it down into smaller tasks, and then delegates those tasks to the appropriate worker agents. The workers then execute their tasks and report back to the supervisor, who synthesizes the results and decides on the next course of action.

## 4. Advanced LangGraph Techniques

*   **Streaming and Async Operations:** LangGraph supports streaming and asynchronous operations, which can be useful for building responsive, real-time applications.
*   **Persistence and Checkpoints:** You can save the state of a graph at any point and then resume it later. This is crucial for building robust applications that can handle errors and interruptions.
*   **Human-in-the-Loop:** LangGraph makes it easy to add human-in-the-loop checkpoints to your graph. This allows you to pause the execution of the graph and wait for human input before proceeding.

## 5. Real-World Case Studies

*   **Klarna:** The fintech company Klarna uses LangGraph to power its AI Assistant, which handles customer support for millions of users.
*   **Uber:** Uber's Developer Platform team uses LangGraph to automate unit test generation and fix code.
*   **LinkedIn:** LinkedIn uses a hierarchical agent system powered by LangGraph for its AI recruiter.

## 6. A Complete, Runnable Code Example

This example demonstrates a simple multi-agent system with a supervisor and two workers.

```python
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# --- Environment Setup ---
# Make sure to set your OpenAI API key in your environment variables
# export OPENAI_API_KEY="..."

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# --- Agent Definitions ---
class MyAgent:
    def __init__(self, model, system_message):
        self.model = model.bind(system_message=system_message)

    def __call__(self, state):
        messages = state['messages']
        response = self.model.invoke(messages)
        return {"messages": [response]}

# --- Graph Definition ---
class MyGraph:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.supervisor = MyAgent(self.llm, "You are a supervisor. You manage a team of two workers: a writer and a critic.")
        self.writer = MyAgent(self.llm, "You are a writer. You write short stories.")
        self.critic = MyAgent(self.llm, "You are a critic. You critique short stories.")
        self.workflow = StateGraph(AgentState)

        self.workflow.add_node("supervisor", self.supervisor)
        self.workflow.add_node("writer", self.writer)
        self.workflow.add_node("critic", self.critic)

        self.workflow.add_conditional_edges(
            "supervisor",
            self.supervisor_router,
            {"writer": "writer", "critic": "critic", "end": END}
        )
        self.workflow.add_edge("writer", "supervisor")
        self.workflow.add_edge("critic", "supervisor")
        self.workflow.set_entry_point("supervisor")

        self.app = self.workflow.compile()

    def supervisor_router(self, state):
        # A simple router that alternates between the writer and the critic
        if len(state['messages']) % 2 == 1:
            return "writer"
        else:
            return "critic"

if __name__ == "__main__":
    graph = MyGraph()
    inputs = {"messages": ["Write a short story about a robot who learns to love."]}
    for output in graph.app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")
```

### Code Walkthrough

1.  **Environment Setup:** We import the necessary libraries and assume that the OpenAI API key is set as an environment variable.
2.  **State Definition:** We define the state of our graph as a `TypedDict` with a single key, `messages`. The `Annotated` type hint with `operator.add` tells LangGraph to append new messages to the list, rather than replacing it.
3.  **Agent Definitions:** We create a simple `MyAgent` class that takes a model and a system message. The `__call__` method is what will be executed when the agent's node is called.
4.  **Graph Definition:**
    *   We create a `MyGraph` class to encapsulate our graph.
    *   We instantiate our three agents: a supervisor, a writer, and a critic.
    *   We create a `StateGraph` with our `AgentState`.
    *   We add our three agents as nodes to the graph.
    *   We use `add_conditional_edges` to create a simple routing logic. The `supervisor_router` function is called after the supervisor node is executed, and it decides which node to go to next.
    *   We use `add_edge` to create the edges from the writer and the critic back to the supervisor.
    *   We use `set_entry_point` to specify that the supervisor is the first node to be called.
    *   We compile the graph to create a runnable application.
5.  **Running the Graph:**
    *   In the `if __name__ == "__main__":` block, we create an instance of our graph.
    *   We define the initial input to the graph, which is a list with a single message.
    *   We use `app.stream(inputs)` to run the graph and stream the output.

## 7. Exercises

1.  Modify the `supervisor_router` function to have a more intelligent routing logic. For example, you could have the supervisor call an LLM to decide which worker to call next.
2.  Add a human-in-the-loop checkpoint to the graph that allows a human to review the story before it is passed to the critic.

## 8. Further Reading and References

*   The official LangGraph documentation.
*   "Building Multi-Agent AI Systems with LangGraph" (YouTube tutorial).
*   The LangChain blog for case studies and announcements.
