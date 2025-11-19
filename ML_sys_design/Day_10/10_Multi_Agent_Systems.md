# Day 10: Many Minds, One Goal - An Introduction to Multi-Agent Systems

We have focused so far on designing a single, autonomous agent. But what happens when we put multiple agents together? This is the domain of **Multi-Agent Systems (MAS)**, a field that studies how independent agents can interact—by competing or collaborating—to solve problems.

In the context of LLM-powered agents, multi-agent design patterns have emerged as a powerful way to improve reliability, handle complexity, and achieve more sophisticated outcomes.

---

## Part 1: Why Use Multiple Agents?

Using a single, monolithic agent to solve a very complex problem can be difficult. The prompt can become incredibly long and confusing, and the agent can lose focus. By breaking a problem down and assigning different roles to different specialized agents, we can often achieve better results.

**Key Advantages:**
1.  **Modularity and Simplicity:** Each agent can be given a much simpler and more focused prompt. It's easier to build and debug an "analyst" agent and a "writer" agent separately than to build one giant "analyst-writer" agent.
2.  **Improved Performance:** By assigning specific roles, you can prime each agent to be an expert in its narrow task, leading to higher-quality work.
3.  **Resilience and Error Checking:** You can create agent teams where one agent's job is to critique and double-check the work of another, catching errors and improving the final output. This formalizes the "self-correction" loop we learned about on Day 8.
4.  **Parallelization:** Some problems can be broken down into sub-tasks that can be worked on in parallel by multiple agents, speeding up the overall process.

---

## Part 2: Competition vs. Collaboration

Interactions in multi-agent systems generally fall into two categories:

### **Competition**
*   **Concept:** Agents have conflicting goals. Each agent tries to maximize its own utility, potentially at the expense of others.
*   **Analogy:** A game of chess or an auction.
*   **LLM Use Case:** Simulating negotiations or running "red teaming" exercises where one agent tries to find security flaws in another agent's system. For example, one agent acts as a chatbot while a "malicious" agent tries to prompt-inject it.

### **Collaboration**
*   **Concept:** Agents share a common goal and work together to achieve it. This is the most common pattern in modern agentic engineering.
*   **Analogy:** A software development team, where a product manager, a developer, and a QA tester all work together to ship a feature.
*   **LLM Use Case:** A research task where one agent finds sources, a second agent synthesizes them, and a third agent writes the final report.

---

## Part 3: Architectures for Multi-Agent Collaboration

Several common patterns have emerged for structuring collaborative multi-agent systems.

### **1. The Hierarchical (Manager-Worker) Architecture**
*   **Structure:** A "manager" or "orchestrator" agent is responsible for breaking down a complex task into smaller sub-tasks. It then dispatches these sub-tasks to one or more "worker" agents. The manager collects the results from the workers and synthesizes them into a final answer.
*   **Example: The Research Team**
    1.  **User:** "Write a report on the impact of LoRA on LLM fine-tuning."
    2.  **Orchestrator Agent:** Receives the request. Its plan is:
        *   "Sub-task 1: Find 3-5 seminal papers or articles about LoRA. Assign to **Researcher Agent**."
        *   "Sub-task 2: Summarize the key findings from the provided sources. Assign to **Summarizer Agent**."
        *   "Sub-task 3: Write a final, well-structured report based on the summaries. Assign to **Writer Agent**."
    3.  The **Orchestrator** executes this plan, passing the output of one worker as the input to the next, and finally presents the output from the **Writer Agent** to the user.

### **2. The "Debate" or "Expert Panel" Architecture**
*   **Structure:** Multiple agents are given the same prompt or problem. They each generate their own answer or perspective independently. Then, a final "synthesizer" or "judge" agent reads all the individual responses and either chooses the best one or combines them into a more comprehensive final answer.
*   **This is a great pattern for reducing bias and improving the robustness of answers.** If three different agents with slightly different prompts all arrive at the same conclusion, you can be much more confident that it's correct.
*   **Example: The Medical Diagnosis Assistant**
    1.  **User:** Provides a list of patient symptoms.
    2.  **Agent 1 (Cardiologist Persona):** Analyzes symptoms from a cardiovascular perspective.
    3.  **Agent 2 (Pulmonologist Persona):** Analyzes symptoms from a respiratory perspective.
    4.  **Agent 3 (Infectious Disease Persona):** Analyzes symptoms from an infectious disease perspective.
    5.  **Synthesizer Agent:** Reads the three differential diagnoses and creates a final report for the human doctor, listing the most likely conditions and noting where the expert agents agreed or disagreed.

---

## Part 4: Communication and State Management

In a multi-agent system, you need to manage the flow of information.
*   **Communication Protocol:** How do agents pass messages to each other? This can be as simple as passing the text output of one agent as the input to the next.
*   **Shared State:** Do the agents have a shared "scratchpad" or memory where they can all read and write information? This could be a simple text file, a dictionary in your code, or a more complex database.
*   **Orchestration Logic:** The code you write that controls the multi-agent workflow is called the orchestrator. Frameworks like **LangChain** or **AutoGen** provide tools specifically for building and managing these complex orchestrations.

---

## Activity: Design a Multi-Agent Team for Your Project

Let's rethink your course project through a multi-agent lens. Even if a single agent *could* do the job, how could a team of specialized agents do it *better*?

For the agent you chose (**Code Documenter**, **ELI5 Researcher**, or **Personal Chef**):

1.  **Propose a Team:** Propose a team of 2-3 specialized agents that could work together to solve the problem. Give each agent a specific role and name (e.g., "Code Analyst," "Docstring Writer").
2.  **Choose an Architecture:** Would you use a Hierarchical (Manager-Worker) model or a Debate/Panel model? Why?
3.  **Map the Workflow:** Describe the step-by-step workflow.
    *   What is the user's initial input?
    *   What does the first agent do?
    *   How is the output of the first agent passed to the second agent?
    *   What does the second agent do?
    *   What is the final output delivered to the user?

**Example for the "Personal Chef" project:**
*   **Team:**
    *   `Dietary_Analyst_Agent`: Analyzes the ingredients and identifies potential recipe categories (e.g., "low-carb," "vegetarian," "quick-and-easy").
    *   `Recipe_Generator_Agent`: Takes a category and the ingredients and generates a specific recipe.
*   **Architecture:** Hierarchical. A main orchestrator would first call the `Dietary_Analyst` to get ideas, then pass the best idea to the `Recipe_Generator`.
*   **Workflow:**
    1.  User provides: "I have chicken, broccoli, and lemon."
    2.  Orchestrator sends to `Dietary_Analyst`: "Given these ingredients, what kind of healthy meal could be made?"
    3.  `Dietary_Analyst` replies: "A good option is a simple, low-carb roasted chicken and broccoli dish."
    4.  Orchestrator sends to `Recipe_Generator`: "Generate a recipe for a low-carb roasted chicken and broccoli dish using lemon."
    5.  `Recipe_Generator` replies with the final, step-by-step recipe, which is shown to the user.
