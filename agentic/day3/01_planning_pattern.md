# Day 3, Topic 1: The Planning Pattern

The ReAct and Tool Use patterns are excellent for tasks that can be solved in a few steps. However, for more complex, multi-step tasks, it's often beneficial for the agent to create a plan upfront. This is where the **Planning Pattern** comes in.

The Planning Pattern involves having the agent break down a high-level goal into a sequence of smaller, executable sub-tasks. This plan then serves as a roadmap for the agent's subsequent actions.

## From Simple Tasks to Complex Plans

Consider the difference between these two tasks:

1.  "What is the capital of France?"
2.  "Plan a 3-day trip to Paris for a family of four on a budget of $2000."

The first task can be answered in a single step with a simple search. The second task, however, is much more complex. It requires:

*   Finding flights.
*   Finding accommodation.
*   Planning an itinerary of activities.
*   Estimating costs for food and transportation.
*   All while staying within the specified budget.

Attempting to solve this kind of complex task with a purely reactive approach (like ReAct) can be inefficient and unreliable. The agent might get lost in the details, lose track of the overall goal, or fail to consider all the constraints.

The Planning Pattern addresses this by introducing an explicit planning step before the agent starts to execute.

## The Role of the LLM as a "Planner"

In the Planning Pattern, the agent's Large Language Model (LLM) takes on the role of a "planner." The process typically looks like this:

1.  **Goal Decomposition:** The agent is given a high-level goal (e.g., "plan a trip to Paris").
2.  **Plan Generation:** The LLM is prompted to generate a plan to achieve this goal. The plan is typically a sequence of steps, each of which is a smaller, more manageable sub-task.
3.  **Plan Representation:** The generated plan is stored in a structured format, such as a list of strings, a JSON object, or even a directed acyclic graph (DAG) for more complex plans with dependencies.
4.  **Plan Execution:** The agent then executes the plan, one step at a time. For each step, the agent might use the ReAct pattern or call a specific tool.
5.  **Plan Adaptation (Optional):** In more advanced implementations, the agent might be able to adapt its plan based on the results of previous steps. For example, if it discovers that a particular museum is closed, it might update the plan to visit a different one.

## Techniques for Prompting an LLM to Generate a Plan

The key to the Planning Pattern is to effectively prompt the LLM to generate a good plan. Here are some techniques:

*   **Be Specific:** Clearly state the high-level goal and any constraints.
*   **Provide Examples:** In few-shot prompting, you can provide the LLM with examples of good plans for similar tasks.
*   **Specify the Output Format:** Instruct the LLM to generate the plan in a specific format (e.g., a numbered list, a JSON array) to make it easier to parse and execute.

## Comparing Planning Techniques

There are several ways to represent a plan, each with its own trade-offs:

### 1. Simple Sequential Plans

This is the simplest form of a plan, where the sub-tasks are represented as a linear sequence of steps.

*   **Representation:** A simple list or array of strings.
*   **Pros:** Easy to generate, parse, and execute.
*   **Cons:** Cannot represent dependencies between tasks or allow for parallel execution.
*   **Best for:** Tasks that are inherently sequential and do not have complex dependencies.

### 2. Directed Acyclic Graphs (DAGs)

For more complex tasks, a plan can be represented as a **Directed Acyclic Graph (DAG)**. In a DAG, each node represents a sub-task, and the edges represent the dependencies between them.

*   **Representation:** A graph data structure, where each node has a list of its dependencies.
*   **Pros:** Can represent complex dependencies between tasks and allows for parallel execution of non-dependent tasks.
*   **Cons:** More complex to generate, parse, and execute.
*   **Best for:** Complex tasks with multiple, non-sequential dependencies. For example, in a travel planning task, you might be able to search for flights and hotels in parallel, but you must book the flights before you book the airport transfer.

The choice of plan representation will depend on the complexity of the task and the desired trade-off between simplicity and expressiveness. For many tasks, a simple sequential plan is sufficient. But for more complex, real-world problems, a DAG representation can be much more powerful.

