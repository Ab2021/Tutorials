# Day 1, Topic 1: An Expert's Introduction to AI Agents

## 1. Introduction to the Theory of Agency in AI

### 1.1. What is an AI Agent?

In the field of Artificial Intelligence, an **agent** is formally defined as any entity that can perceive its environment through **sensors** and act upon that environment through **actuators**. This definition, while simple, provides a powerful abstraction for thinking about and building intelligent systems.

The core idea that distinguishes an AI agent from a simple program is the concept of **agency**. Agency implies a capacity for autonomous, goal-directed action. While a simple script might automate a task, an agent is expected to exhibit a degree of self-direction and decision-making in its pursuit of a goal.

### 1.2. The Agent-Environment Interface

The interaction between an agent and its environment is a continuous loop:

1.  The agent receives a **percept** from the environment. A percept is a piece of information about the state of the environment at a particular instant.
2.  The agent's internal reasoning process, its **agent function**, processes the sequence of percepts it has received and decides on an action.
3.  The agent executes the action, which in turn affects the state of the environment.

The "intelligence" of the agent is determined by the quality of its agent function. A rational agent is one that, for any given sequence of percepts, chooses an action that is expected to maximize its performance measure, given the evidence provided by the percept sequence and whatever built-in knowledge the agent has.

**Types of Environments:**

The nature of the environment has a significant impact on the design of an agent. Environments can be classified along several dimensions:

*   **Fully Observable vs. Partially Observable:** Can the agent's sensors detect all aspects of the environment that are relevant to the choice of action?
*   **Deterministic vs. Stochastic:** Is the next state of the environment completely determined by the current state and the agent's action?
*   **Episodic vs. Sequential:** Is the agent's experience divided into atomic episodes, or is the choice of action in one episode dependent on the actions taken in previous episodes?
*   **Static vs. Dynamic:** Can the environment change while the agent is deliberating?
*   **Discrete vs. Continuous:** Is the state of the environment and the set of actions discrete or continuous?
*   **Single-agent vs. Multi-agent:** Is the agent operating by itself in the environment, or are there other agents?

## 2. Core Characteristics of AI Agents (Expanded)

*   **Autonomy:** An agent is autonomous to the extent that its behavior is determined by its own experience, rather than by the built-in knowledge of its designer. A truly autonomous agent should be able to learn and adapt to new situations.
*   **Reactivity:** The ability to respond in a timely fashion to changes in the environment.
    *   **Simple Reflex Agents:** These agents react directly to percepts, without considering the history of percepts.
    *   **Model-based Reflex Agents:** These agents maintain an internal model of the world and use it to inform their decisions.
*   **Pro-activeness:** The ability to take the initiative to achieve goals.
    *   **Goal-based Agents:** These agents have explicit goals and choose actions that will lead them closer to achieving those goals.
*   **Social Ability:** The ability to interact with other agents. This can range from simple communication to complex negotiation and collaboration.

## 3. A Taxonomy of Agent Architectures

*   **Reactive Architectures:** These are the simplest architectures. They are based on a direct mapping from percepts to actions. They are fast and efficient, but they are also limited in their ability to handle complex situations.
*   **Deliberative Architectures:** These architectures are based on the idea of explicit reasoning and planning. They maintain an internal model of the world and use it to deliberate about the best course of action. The **BDI (Belief-Desire-Intention)** model is a classic example of a deliberative architecture.
*   **Hybrid Architectures:** These architectures combine reactive and deliberative components. They typically have a reactive layer for handling immediate threats and opportunities, and a deliberative layer for long-term planning.

## 4. Applications of AI Agents

*   **E-commerce:** Personalized recommendation agents, automated bidding agents in online auctions.
*   **Healthcare:** Diagnostic agents that assist doctors in interpreting medical images, patient monitoring agents.
*   **Finance:** Algorithmic trading agents, fraud detection agents.
*   **Entertainment:** Non-player characters (NPCs) in video games, personalized content recommendation agents.

## 5. Intricacies and Advanced Concepts

*   **The "Sense of Agency" (SoA):** In cognitive science, the sense of agency is the subjective experience of controlling one's own actions. A key research question in AI is whether and how we can build agents that have a similar sense of agency.
*   **Agency and Intelligence:** While agency and intelligence are closely related, they are not the same thing. An agent can be autonomous without being particularly intelligent. A key challenge is to build agents that are both autonomous and intelligent.

## 6. Exercises

1.  Consider a vacuum-cleaner agent. Describe its environment, percepts, actions, and a suitable performance measure. Classify its environment according to the properties listed in section 1.2.
2.  Compare and contrast the BDI model with the ReAct pattern. What are the similarities and differences in their approach to agentic reasoning?

## 7. Further Reading and References

*   Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*. Pearson. (This is the classic textbook on AI and has excellent chapters on agent theory.)
*   Wooldridge, M. (2009). *An Introduction to MultiAgent Systems*. John Wiley & Sons.