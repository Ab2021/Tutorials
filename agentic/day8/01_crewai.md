# Day 8: Diving into Agentic Frameworks: CrewAI

Our final framework deep dive is into **CrewAI**, a cutting-edge open-source framework for orchestrating role-playing, autonomous AI agents. CrewAI is designed to enable agents to collaborate seamlessly to solve complex tasks.

## Morning Session (9:00 AM - 12:00 PM): Introduction to CrewAI

### Topic 1: What is CrewAI and Why Use It?

CrewAI is a framework that helps you build and manage a "crew" of AI agents that work together to achieve a common goal. It is built on the idea that by giving agents specific roles, goals, and backstories, you can create more effective and specialized multi-agent systems.

**Why CrewAI?**

*   **Sophisticated Multi-Agent Collaboration:** CrewAI provides a robust framework for managing collaboration between multiple agents, with support for both sequential and hierarchical workflows.
*   **Role-Playing Agents:** The emphasis on giving agents roles, goals, and backstories helps to create more specialized and effective agents.
*   **Flexible Task Management:** CrewAI's task management system allows you to define complex, multi-step tasks and assign them to the appropriate agents.
*   **Seamless Tool Integration:** It's easy to give your agents access to external tools, allowing them to interact with the world and access a wider range of information.

### Topic 2: The Core Concepts of CrewAI

CrewAI is built around a few key concepts:

1.  **Agents:** An agent is an autonomous unit that is assigned a specific role, goal, and backstory. This context helps the agent to understand its purpose and to perform its tasks more effectively.
2.  **Tasks:** A task is a specific unit of work that is assigned to an agent. Each task has a description and an expected output.
3.  **Tools:** A tool is a function or an API that an agent can use to perform its tasks. CrewAI provides a simple way to integrate with a wide range of tools.
4.  **Crews:** A crew is a group of agents that work together to complete a set of tasks.
5.  **Processes:** A process defines the workflow for a crew. CrewAI supports two main types of processes:
    *   **Sequential:** Tasks are executed in a predefined order.
    *   **Hierarchical:** A manager agent delegates tasks to other agents.

### Exercise

1.  **Flesh out the multi-agent system you designed for the AutoGen exercise.**
    *   For each agent, define a `role`, a `goal`, and a `backstory`.
    *   For each task, define a `description` and an `expected_output`.
    *   This will help you to think in terms of the CrewAI concepts.

## Afternoon Session (1:00 PM - 4:00 PM): Building a Simple CrewAI Application

### Topic 3: Setting up Your CrewAI Environment

1.  **Installation:** You can install CrewAI with pip:
    ```bash
    pip install crewai
    ```
2.  **LLM API Key:** You'll need to set up an API key for your chosen LLM provider (e.g., OpenAI, Google AI Platform).

### Topic 4: Building a Basic Crew with CrewAI

Let's build a simple two-agent crew that can research a topic and write a short summary.

Here's a high-level overview:

1.  **Define the Agents:** We'll create two agents:
    *   A **"Researcher" Agent:** This agent's role is to find and gather information about a given topic.
    *   A **"Writer" Agent:** This agent's role is to take the information from the researcher and write a short, engaging summary.
2.  **Define the Tasks:** We'll create two tasks:
    *   A **"Research" Task:** This task will be assigned to the researcher agent and will instruct it to find information about the topic.
    *   A **"Writing" Task:** This task will be assigned to the writer agent and will instruct it to write a summary based on the researcher's findings.
3.  **Assemble the Crew:** We'll create a crew with our two agents and two tasks, and we'll specify a sequential process.
4.  **Kickoff the Crew:** We'll then "kick off" the crew to start the process. The researcher will perform its task, and then the writer will perform its task, using the output from the researcher.

### Exercise

1.  **Follow a tutorial to build and run a simple two-agent crew using CrewAI.**
    *   The official CrewAI documentation and various online tutorials provide excellent step-by-step guides.
2.  **Experiment with your crew.**
    *   Try giving them a few different topics to research and write about.
    *   Observe how the agents collaborate and how the output of the first agent is used by the second agent.

This exercise will give you a solid understanding of how to use CrewAI to build your own collaborative, role-playing AI agents.

## Advanced CrewAI Concepts

Once you have mastered the basics of CrewAI, you can start to explore some of its more advanced features:

*   **Memory and Caching:** CrewAI has built-in support for both short-term and long-term memory.
    *   **Short-term Memory:** The conversation history between agents is automatically managed and passed between them, so they have the context they need to collaborate effectively.
    *   **Long-term Memory:** You can configure your crew to use a long-term memory system (such as a vector database) to remember information across multiple sessions.
    *   **Caching:** CrewAI also supports caching of tool executions. This means that if an agent calls the same tool with the same inputs multiple times, the result will be returned from the cache instead of being re-computed, which can save time and money.
*   **The Hierarchical Process:** In addition to the simple sequential process, CrewAI supports a **hierarchical process**. In this model, you have a "manager" agent that is responsible for delegating tasks to a set of "worker" agents. The manager can then review the work of the worker agents and decide on the next course of action. This is a powerful pattern for solving complex problems that require a high degree of coordination.

By leveraging these advanced features, you can build highly sophisticated and capable multi-agent systems with CrewAI.

## A Simple Code Example

Here is a small code snippet to illustrate the core concepts of CrewAI:

```python
from crewai import Agent, Task, Crew, Process

# In a real application, you would set up your LLM API key here
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 1. Define the agents
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting
  actionable insights."""
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for
  your insightful and engaging articles.
  You transform complex concepts into compelling narratives."""
)

# 2. Define the tasks
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="A full analysis report",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, write a compelling blog post
  that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
  expected_output="A 4-paragraph blog post",
  agent=writer
)

# 3. Assemble the crew
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  process=Process.sequential
)

# 4. Kick off the crew
result = crew.kickoff()

print(result)
```
This example shows a simple two-agent crew. The `researcher` agent is responsible for gathering information, and the `writer` agent is responsible for writing a blog post based on that information. The tasks are executed sequentially, and the output of the first task is automatically passed as context to the second task.


