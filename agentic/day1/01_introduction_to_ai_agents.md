# Day 1, Topic 1: Introduction to AI Agents

## What is an AI Agent?

At its core, an **AI agent** is an autonomous entity that exists in an environment, perceives that environment through sensors, and acts upon that environment through actuators to achieve specific goals. Think of it as a self-contained program or system that can make its own decisions and take actions to get things done.

The term "agent" comes from the Latin word *agere*, which means "to do". This emphasizes the active, goal-oriented nature of these systems. They are not just passive programs that execute a fixed set of instructions; they are dynamic, responsive, and capable of adapting their behavior to changing circumstances.

## Key Characteristics of AI Agents

To be considered an agent, a system should exhibit several key characteristics:

*   **Autonomy:** An agent can operate without direct human intervention. It has control over its own actions and internal state. This is the most crucial characteristic.
*   **Reactivity:** An agent can perceive its environment and respond in a timely fashion to changes that occur in it. For example, a self-driving car's agent must react immediately to a pedestrian stepping onto the road.
*   **Pro-activeness:** An agent does not simply act in response to its environment; it is capable of taking the initiative to achieve its goals. It exhibits goal-directed behavior. For example, a cleaning robot's agent will actively seek out and clean dirt, rather than waiting to be told where to clean.
*   **Social Ability:** An agent can interact with other agents (and possibly humans) via some kind of agent-communication language. This is essential for multi-agent systems where collaboration is required to solve complex problems.

## The Agent-Environment Interface

To be more formal, we can describe the interaction between an agent and its environment as a continuous loop:

1.  The agent receives a **percept** from the environment. A percept is a single piece of information that the agent's sensors have detected.
2.  The agent's internal **reasoning process** (its "brain") processes the percept and decides on an **action** to take.
3.  The agent executes the action, which in turn affects the state of the environment.
4.  The agent receives a new percept from the now-changed environment, and the loop continues.

This loop is the fundamental basis of all agentic behavior. The "intelligence" of the agent lies in its ability to choose actions that will lead it closer to its goals, based on the sequence of percepts it has received.

## Agent vs. Model: A Key Distinction

It's important to understand the difference between an **AI agent** and an **AI model**.

*   An **AI model** (like a Large Language Model or LLM) is the "brain" of the agent. It is the component that does the reasoning and decision-making.
*   An **AI agent** is the complete entity, which includes the model, as well as the perception and action components. The agent is the system that perceives the world, thinks about what to do (using the model), and then takes action.

In short, the model is a part of the agent, but the agent is more than just the model.



## Examples of AI Agents in the Real World

AI agents are all around us, in various forms:

*   **Virtual Personal Assistants:** Siri, Alexa, and Google Assistant are all examples of AI agents. They perceive your voice commands (environment), reason about your intent, and act by playing music, setting reminders, or answering questions (actuators).
*   **Chatbots and Customer Service Bots:** These agents interact with customers on websites and messaging platforms. They perceive customer queries, reason about the problem, and act by providing information, resolving issues, or escalating to a human agent.
*   **Game Characters (NPCs):** Non-player characters in video games are agents that perceive the game world and the player's actions, and act according to their programmed behaviors and goals (e.g., to help the player, to fight them, or to simply populate the world).
*   **Autonomous Vehicles:** The control system of a self-driving car is a highly sophisticated agent. It perceives the world through cameras, LiDAR, and other sensors, reasons about the best course of action (e.g., to accelerate, brake, or turn), and controls the car's physical components (actuators).
*   **Robotic Process Automation (RPA):** In a business context, RPA bots are agents that can automate repetitive digital tasks, such as data entry, by interacting with software applications and websites.

## Exercise

1.  **Identify and list 5 examples of AI agents you interact with in your daily life.**
    *   *Example 1: A spam filter in your email client.*
    *   *Example 2: A recommendation engine on a streaming service.*
    *   *Example 3: ...*
    *   *Example 4: ...*
    *   *Example 5: ...*
2.  **For each example, describe its goals, environment, and actions.**
    *   **Spam Filter:**
        *   **Goal:** To keep your inbox free of unsolicited and malicious emails.
        *   **Environment:** The stream of incoming emails.
        *   **Actions:** To classify an email as spam or not spam, and to move spam emails to a separate folder.
    *   **Recommendation Engine:**
        *   **Goal:** ...
        *   **Environment:** ...
        *   **Actions:** ...
