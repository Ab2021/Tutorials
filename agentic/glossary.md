# Glossary of Key Terms

This glossary provides definitions for the key terms and concepts used in this study guide.

---

### A

*   **Action:** The output of an agent's reasoning process. An action is how an agent affects its environment.
*   **Actuator:** The component of an agent that executes an action.
*   **AI Agent:** An autonomous entity that perceives its environment, makes decisions, and takes actions to achieve specific goals.
*   **Agentic Design Patterns:** Reusable solutions to commonly occurring problems in the design of AI agents.
*   **Agent (in AutoGen):** An entity that can send and receive messages, and can be powered by LLMs, code, or human input.
*   **Agent (in CrewAI):** An autonomous unit with a specific role, goal, and backstory.
*   **AssistantAgent (in AutoGen):** An agent that is powered by an LLM and is responsible for solving a given task.
*   **Autonomy:** The ability of an agent to operate without direct human intervention.
*   **AutoGen:** An open-source framework from Microsoft for building and managing multi-agent AI applications.

### B

*   **BDI (Belief-Desire-Intention) Model:** An early agent architecture where an agent's behavior is driven by its beliefs about the world, its desires (goals), and its intentions (commitments to plans).

### C

*   **Chain of Thought (CoT) Reasoning:** A prompting technique for getting a Large Language Model (LLM) to explain its reasoning process step-by-step.
*   **Conversation (in AutoGen):** A sequence of messages exchanged between agents.
*   **Coordinator Pattern:** A multi-agent collaboration pattern where a "manager" agent assigns tasks to "worker" agents.
*   **Crew (in CrewAI):** A group of agents that work together to complete a set of tasks.
*   **CrewAI:** An open-source framework for orchestrating role-playing, autonomous AI agents.

### D

*   **DAG (Directed Acyclic Graph):** A graph where the nodes represent tasks and the edges represent dependencies between them. Used for representing complex plans.
*   **Deep Learning:** A subfield of machine learning that uses neural networks with many layers to learn from data.

### E

*   **Edge (in LangGraph):** A connection between two nodes in a LangGraph graph that defines the flow of control.

### F

*   **Function Calling:** A feature of some LLMs that allows them to request that a specific function be called with specific arguments.

### G

*   **Goal Decomposition:** The process of breaking down a high-level goal into a sequence of smaller, executable sub-tasks.

### H

*   **Human-in-the-Loop (HITL) Pattern:** A design pattern for integrating human feedback and oversight into an agentic system.

### L

*   **LangGraph:** An open-source library from the creators of LangChain for building complex, stateful, and multi-agent applications as graphs.
*   **Large Language Model (LLM):** A large, general-purpose language model that can be adapted to a wide range of tasks. The "brain" of most modern AI agents.
*   **LLM as a Router Pattern:** A design pattern for using an LLM to classify incoming queries and route them to the most appropriate agent, tool, or workflow.

### M

*   **Machine Learning:** A field of AI that focuses on creating systems that can learn from data.
*   **Multi-Agent System:** A system composed of multiple interacting agents.

### N

*   **Network/Swarm Intelligence:** A decentralized multi-agent collaboration pattern where the overall behavior of the system emerges from the local interactions of the individual agents.
*   **Node (in LangGraph):** A function or a callable class that represents a unit of work in a LangGraph graph.

### P

*   **Parallel Pattern:** A multi-agent collaboration pattern where multiple agents work on different parts of a problem simultaneously.
*   **Percept:** A single piece of information that an agent's sensors have detected from the environment.
*   **Perception:** The process by which an agent senses its environment.
*   **Plan Representation:** The format in which a plan is stored (e.g., a list of steps, a DAG).
*   **Planning Pattern:** A design pattern for having an agent break down a high-level goal into a sequence of smaller, executable sub-tasks.
*   **Pro-activeness:** The ability of an agent to take the initiative to achieve its goals.
*   **Process (in CrewAI):** The workflow for a crew (e.g., sequential, hierarchical).

### R

*   **ReAct (Reason and Act) Pattern:** A design pattern that structures an agent's workflow into an iterative loop of reasoning, acting, and observing.
*   **Reactivity:** The ability of an agent to perceive its environment and respond in a timely fashion to changes that occur in it.
*   **Reasoning:** The process by which an agent processes information and makes decisions.
*   **Reflection/Critique Pattern:** A design pattern for enabling agents to evaluate their own performance and learn from their mistakes.
*   **Reinforcement Learning:** A type of machine learning where an agent learns to make a sequence of decisions to maximize a cumulative reward.

### S

*   **Sequential Workflows/Orchestration Pattern:** A design pattern for building a "pipeline" of agents, where the output of one agent becomes the input for the next.
*   **Social Ability:** The ability of an agent to interact with other agents.
*   **State (in LangGraph):** A central, shared data structure that is passed between all the nodes in a LangGraph graph.
*   **Symbolic AI:** A branch of AI research that focuses on creating systems that can reason about the world using explicit symbols and rules.

### T

*   **Task (in CrewAI):** A specific unit of work that is assigned to an agent.
*   **Tool (in CrewAI):** A function or an API that an agent can use to perform its tasks.
*   **Tool Use Pattern:** A design pattern for giving an agent access to a set of tools (functions or APIs) that it can call to perform specific tasks.

### U

*   **UserProxyAgent (in AutoGen):** An agent that acts as a proxy for a human user.
