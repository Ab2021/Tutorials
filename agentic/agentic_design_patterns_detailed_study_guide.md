# Agentic Design Patterns: A Detailed 5-Day Study Guide

This guide provides a structured, day-by-day plan for learning agentic design patterns. Each day focuses on a specific set of topics, building from foundational concepts to more advanced patterns, and includes practical exercises to reinforce learning.

## Prerequisites

This guide is intended for individuals with a basic understanding of programming (preferably Python) and a conceptual understanding of machine learning and APIs. Familiarity with the following concepts will be beneficial but is not strictly required:

*   **Python Programming:** Basic syntax, data structures (lists, dictionaries), functions, and classes.
*   **APIs:** What an API is and how to make requests to a REST API.
*   **Large Language Models (LLMs):** A general awareness of what LLMs are and their capabilities (e.g., GPT-3, PaLM).

## Tools and Technologies

The exercises in this guide can be completed using the following tools and technologies:

*   **Python 3.x:** The primary programming language for the exercises.
*   **Jupyter Notebook or any Python IDE:** For writing and executing code.
*   **Access to an LLM API:** You will need an API key for an LLM provider like OpenAI, Google AI Platform, or an open-source model provider.
*   **(Optional) LangChain or LlamaIndex:** While not required, these frameworks can be helpful for building more complex agents and orchestrating workflows. You can explore them after you have a solid understanding of the fundamental patterns.


## Day 1: Foundations of Agentic AI

*   **Morning Session (9:00 AM - 12:00 PM): Introduction to AI Agents**
    *   **Topic 1: What is an AI Agent?**
        *   Detailed definition: An autonomous entity that perceives its environment, makes decisions, and takes actions to achieve specific goals.
        *   Key characteristics: Autonomy, Reactivity, Pro-activeness, Social Ability.
        *   Examples: Chatbots, virtual assistants, game characters, autonomous vehicles.
    *   **Topic 2: A Brief History of Agentic Systems**
        *   Symbolic AI and early agent architectures (e.g., BDI - Belief-Desire-Intention).
        *   The impact of machine learning and deep learning.
        *   The rise of Large Language Models (LLMs) as the core of modern agents.
    *   **Exercise:**
        *   Identify and list 5 examples of AI agents you interact with in your daily life.
        *   For each example, describe its goals, environment, and actions.

*   **Afternoon Session (1:00 PM - 4:00 PM): Core Components and Design Patterns**
    *   **Topic 3: The Anatomy of an AI Agent**
        *   **Perception:** Sensors, data input (text, image, audio), APIs.
        *   **Reasoning:** The "brain" - decision-making, planning, and knowledge representation. The role of LLMs in reasoning.
        *   **Action:** Actuators, API calls, generating text/code, interacting with UIs.
    *   **Topic 4: Introduction to Agentic Design Patterns**
        *   Analogy to software design patterns.
        *   Why patterns are crucial for building robust, scalable, and maintainable agents.
        *   A high-level overview of the patterns to be covered (ReAct, Tool Use, Planning, etc.).
    *   **Exercise:**
        *   Sketch a diagram of an AI agent for a specific task (e.g., a customer service chatbot).
        *   Label the perception, reasoning, and action components.

## Day 2: The ReAct Pattern and Tool Use

*   **Morning Session (9:00 AM - 12:00 PM): The ReAct (Reason and Act) Pattern**
    *   **Topic 1: Deep Dive into ReAct**
        *   The core loop: **Reason** (formulate a thought), **Act** (take an action), **Observe** (analyze the result).
        *   The importance of "chain of thought" reasoning.
        *   How ReAct makes agent behavior more interpretable and debuggable.
    *   **Topic 2: ReAct in Practice**
        *   Use cases: Question answering, fact-checking, simple web navigation.
        *   Walkthrough of a ReAct agent's thought process for a given query.
    *   **Exercise:**
        *   Given a sample query, manually write out the Reason, Act, and Observe steps for a ReAct agent.
        *   (Optional) Write a simple Python script to simulate a ReAct loop for a basic task.

*   **Afternoon Session (1:00 PM - 4:00 PM): The Tool Use Pattern**
    *   **Topic 3: Extending Agent Capabilities with Tools**
        *   Why LLMs alone are not enough (the "knowledge cutoff" problem).
        *   Types of tools: Search engines, calculators, code interpreters, custom APIs.
        *   How agents decide which tool to use and with what inputs.
    *   **Topic 4: Implementing Tool Use**
        *   Function calling APIs (e.g., OpenAI's function calling).
        *   Parsing tool requests from the LLM's output.
        *   Handling tool execution and returning the results to the agent.
    *   **Exercise:**
        *   Design a tool for a specific purpose (e.g., a tool to get the current stock price for a company).
        *   Define the tool's input and output schema.
        *   (Optional) Write a Python function that implements this tool.

## Day 3: Planning and Self-Correction

*   **Morning Session (9:00 AM - 12:00 PM): The Planning Pattern**
    *   **Topic 1: From Simple Tasks to Complex Plans**
        *   The need for planning when a task requires multiple steps.
        *   Decomposition: Breaking down a high-level goal into a sequence of smaller, executable sub-tasks.
        *   The role of the LLM as a "planner".
    *   **Topic 2: Plan Generation and Execution**
        *   Techniques for prompting an LLM to generate a plan.
        *   Representing the plan (e.g., as a list of steps, a DAG).
        *   Executing the plan and handling errors or unexpected outcomes.
    *   **Exercise:**
        *   Choose a complex task (e.g., "plan a weekend trip to a nearby city").
        *   Write a prompt to an LLM to generate a detailed plan for this task.
        *   Review and refine the generated plan.

*   **Afternoon Session (1:00 PM - 4:00 PM): The Reflection/Critique Pattern**
    *   **Topic 3: Enabling Agents to Learn from Experience**
        *   The concept of self-correction and iterative improvement.
        *   How agents can evaluate their own outputs, decisions, and actions.
        *   The role of feedback, both from humans and from the environment.
    *   **Topic 4: Implementing Reflection**
        *   Prompting techniques for self-critique (e.g., "review your previous response and identify any errors or omissions").
        *   Using a separate "critic" agent to evaluate the "worker" agent's output.
        *   Storing and using feedback to improve future performance.
    *   **Exercise:**
        *   Take a previously generated output from an agent (e.g., the travel plan from the morning session).
        *   Write a prompt to an LLM to act as a "critic" and provide feedback on the plan.
        *   Use the feedback to improve the original plan.

## Day 4: Multi-Agent Systems

*   **Morning Session (9:00 AM - 12:00 PM): Multi-Agent Collaboration**
    *   **Topic 1: The Power of Many - Introduction to Multi-Agent Systems**
        *   When and why to use multiple agents instead of a single agent.
        *   Specialization and division of labor.
        *   Improved performance, robustness, and scalability.
    *   **Topic 2: Patterns of Collaboration**
        *   **Coordinator Pattern:** A "manager" agent that assigns tasks to "worker" agents.
        *   **Parallel Pattern (Concurrent):** Multiple agents working on different parts of a problem simultaneously.
        *   **Network/Swarm Intelligence:** Decentralized collaboration and emergent behavior.
    *   **Exercise:**
        *   Design a multi-agent system for a specific problem (e.g., writing a research report).
        *   Define the roles and responsibilities of each agent in the system.
        *   Sketch a diagram showing the communication flow between the agents.

*   **Afternoon Session (1:00 PM - 4:00 PM): Orchestration and Sequential Workflows**
    *   **Topic 3: Building Agentic Pipelines**
        *   The concept of a sequential workflow where agents operate in a predefined order.
        *   The output of one agent becomes the input for the next.
        *   Use cases: Document processing (summarize -> translate -> format), code generation (plan -> write -> test).
    *   **Topic 4: Implementing Orchestration**
        *   Using a simple loop or a more sophisticated orchestration framework (e.g., LangChain, LlamaIndex).
        *   Passing context and state between agents in the pipeline.
        *   Handling errors and retries within the workflow.
    *   **Exercise:**
        *   Design a 3-step agentic pipeline for a task of your choice.
        *   For each step, define the agent's role and the expected input and output.

## Day 5: Human-in-the-Loop and Advanced Concepts

*   **Morning Session (9:00 AM - 12:00 PM): Human-in-the-Loop and Advanced Routing**
    *   **Topic 1: The Human-in-the-Loop Pattern**
        *   The importance of human oversight for safety, ethics, and quality.
        *   Use cases: High-stakes decisions (medical, financial), creative tasks, personalization.
        *   Implementing "human-in-the-loop" checkpoints in an agentic workflow.
    *   **Topic 2: The LLM as a Router**
        *   Using an LLM to classify user queries and route them to the appropriate agent, tool, or workflow.
        *   A powerful pattern for building complex, multi-faceted AI systems.
        *   Examples: A customer support bot that can handle sales, technical, and billing questions.
    *   **Exercise:**
        *   Design a system that uses an LLM as a router to handle different types of user requests.
        *   Define the different "routes" and the agents or tools associated with each route.

## Day 6: Diving into Agentic Frameworks: LangGraph

*   **Morning Session (9:00 AM - 12:00 PM): Introduction to LangGraph**
    *   **Topic 1: What is LangGraph and Why Use It?**
        *   Core concepts: building stateful, multi-actor applications.
        *   Advantages: handling cycles, robustness, human-in-the-loop.
    *   **Topic 2: Core Concepts of LangGraph**
        *   State: the central, shared data structure.
        *   Nodes: the functions that perform the work.
        *   Edges: the connections that define the workflow.
    *   **Exercise:**
        *   Conceptualize a simple multi-step process (e.g., making a cup of tea) as a graph with states, nodes, and edges.

*   **Afternoon Session (1:00 PM - 4:00 PM): Building a Simple LangGraph Application**
    *   **Topic 3: Setting up Your LangGraph Environment**
        *   Installation of `langchain` and `langgraph`.
        *   Getting API keys for an LLM.
    *   **Topic 4: Building a Basic ReAct Agent with LangGraph**
        *   Defining the state, nodes (reason, act), and conditional edges.
        *   Compiling and running the graph.
    *   **Exercise:**
        *   Build and run a simple ReAct agent using LangGraph that can answer a question using a search tool.

## Day 7: Diving into Agentic Frameworks: AutoGen

*   **Morning Session (9:00 AM - 12:00 PM): Introduction to AutoGen**
    *   **Topic 1: What is AutoGen and Why Use It?**
        *   Core concepts: a framework for simplifying the orchestration, automation, and optimization of complex LLM workflows.
        *   Advantages: multi-agent collaboration, tool integration, human-in-the-loop capabilities.
    *   **Topic 2: Core Concepts of AutoGen**
        *   Agents: the fundamental building blocks of AutoGen applications.
        *   Conversations: how agents interact and collaborate.
        *   Multi-Agent Collaboration: patterns for structuring conversations between agents.
    *   **Exercise:**
        *   Design a simple multi-agent system on paper (e.g., a "writer" agent and a "critic" agent that collaborate to write a story). Define the roles of each agent and the messages they would exchange.

*   **Afternoon Session (1:00 PM - 4:00 PM): Building a Simple AutoGen Application**
    *   **Topic 3: Setting up Your AutoGen Environment**
        *   Installation of `pyautogen`.
        *   Configuring LLM providers.
    *   **Topic 4: Building a Basic Multi-Agent System with AutoGen**
        *   Defining a "user proxy" agent and a "worker" agent.
        *   Initiating a conversation between the agents to solve a simple task.
    *   **Exercise:**
        *   Build and run a simple two-agent system using AutoGen that can solve a coding problem (e.g., "write a Python function to calculate the factorial of a number").

## Day 8: Diving into Agentic Frameworks: CrewAI

*   **Morning Session (9:00 AM - 12:00 PM): Introduction to CrewAI**
    *   **Topic 1: What is CrewAI and Why Use It?**
        *   Core concepts: a framework for orchestrating role-playing, autonomous AI agents.
        *   Advantages: sophisticated multi-agent collaboration, flexible task management, seamless tool integration.
    *   **Topic 2: Core Concepts of CrewAI**
        *   Agents: specialized agents with roles, goals, and backstories.
        *   Tasks: the specific assignments for each agent.
        *   Tools: enabling agents to interact with the world.
        *   Crews: how agents collaborate to execute tasks.
        *   Processes: defining the workflow (e.g., sequential, hierarchical).
    *   **Exercise:**
        *   Flesh out the multi-agent system you designed for the AutoGen exercise. For each agent, define a role, a goal, and a backstory. For each task, define a description and an expected output.

*   **Afternoon Session (1:00 PM - 4:00 PM): Building a Simple CrewAI Application**
    *   **Topic 3: Setting up Your CrewAI Environment**
        *   Installation of `crewai`.
        *   Setting up LLM API keys.
    *   **Topic 4: Building a Basic Crew with CrewAI**
        *   Defining agents with roles, goals, and backstories.
        *   Defining tasks with descriptions and expected outputs.
        *   Assembling the agents and tasks into a crew.
        *   Kicking off the crew to run the process.
    *   **Exercise:**
        *   Build and run a simple two-agent crew using CrewAI that can research a topic and write a short summary. For example, a "researcher" agent that finds information about a topic and a "writer" agent that writes a summary of the information.

## Day 9: Review, Comparison, and Future Trends

*   **Morning Session (9:00 AM - 12:00 PM): Grand Review and Framework Comparison**
    *   **Topic 1: Grand Review**
        *   A comprehensive recap of all the design patterns and frameworks covered.
        *   Q&A session to clarify any remaining doubts.
    *   **Topic 2: A Detailed Comparison of Agentic Frameworks**
        *   A deep dive into the similarities and differences between LangGraph, AutoGen, and CrewAI.
        *   Guidance on how to choose the right framework for your project.

*   **Afternoon Session (1:00 PM - 4:00 PM): The Future of Agentic AI**
    *   **Topic 3: The Future of Agentic AI**
        *   Emerging trends: Autonomous agents, agent swarms, embodied agents (robotics).
        *   The ethical considerations of building and deploying autonomous agents.
        *   Resources for continued learning (blogs, papers, open-source projects).

## Day 10: Evaluating Agentic Systems and Final Project

*   **Morning Session (9:00 AM - 12:00 PM): Evaluating Agentic Systems**
    *   **Topic 1: The Philosophy of Evaluation**
        *   The challenges of evaluating non-deterministic and open-ended systems.
        *   The need for a multi-faceted approach to evaluation.
    *   **Topic 2: A Taxonomy of Evaluation Methodologies**
        *   Human Evaluation, Simulation-based Evaluation, LLM-as-a-Judge, and Benchmarks.
    *   **Topic 3: Key Evaluation Metrics**
        *   Task Success Rate, Cost and Latency, Robustness and Reliability, Safety and Alignment.
    *   **Topic 4: A Survey of Evaluation Benchmarks**
        *   AgentBench, ToolBench, SWE-bench, and others.

*   **Afternoon Session (1:00 PM - 4:00 PM): Final Project**
    *   **Topic 5: Final Project**
        *   Choose a complex, open-ended task.
        *   Design and (optionally) implement an AI agent or multi-agent system to solve the task, using a combination of the design patterns and frameworks learned throughout the course.
        *   Write a short report describing your design choices, the patterns and frameworks you used, and a plan for how you would evaluate your system.

## Glossary

A comprehensive glossary of all the key terms and concepts used in this study guide can be found in the `glossary.md` file.



