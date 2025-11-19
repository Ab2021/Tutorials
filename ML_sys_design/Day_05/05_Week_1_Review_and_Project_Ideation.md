# Day 5: Week 1 Review and Project Ideation

Congratulations on completing the first week! We have moved at a rapid pace, establishing the foundational concepts that underpin all of agentic AI. Today is about consolidation, clarification, and creation. We will review the key topics from this week, host an open Q&A, and most importantly, kick off the course-long project where you will build your very own AI agent.

---

## Part 1: Week 1 Conceptual Review

Let's briefly summarize the critical concepts we've covered.

### **Day 1: The Agentic Shift**
*   We moved from **predictive AI** (classifying, translating) to **performative AI** (acting to achieve a goal).
*   An **Agent** perceives, reasons, and acts autonomously.

### **Day 2: The PEAS Framework**
*   A formal method to describe any agent's task. It is your primary tool for design and analysis.
*   **P**erformance Measure: What does success look like? Must be objective.
*   **E**nvironment: The world the agent lives in (Observable, Deterministic, Episodic, Static, Discrete?).
*   **A**ctuators: The agent's "muscles" (e.g., API calls).
*   **S**ensors: The agent's "senses" (e.g., reading API responses).

### **Day 3: Agent Architectures**
*   The internal blueprint of an agent.
*   **Reflex Agents (Simple, Model-Based):** Fast, but limited. They react to percepts.
*   **Deliberative Agents (Goal-Based, Utility-Based):** "Smarter" agents that can plan ahead to achieve goals or maximize utility. They are more flexible but computationally more expensive.
*   **Learning Agents:** Agents that can improve their performance over time.

### **Day 4: The LLM Revolution**
*   **Large Language Models (LLMs)** provide a general-purpose, pre-trained **reasoning engine**.
*   This allows us to rapidly build the "brain" for deliberative agents without starting from scratch.
*   **Prompt Engineering** is the primary way we instruct and control the LLM's behavior.

**The Grand Synthesis:** The core idea of modern agent engineering is to use an LLM as the reasoning component inside one of the classical agent architectures, using the PEAS framework to guide our design.

---

## Part 2: Open Q&A

This is your opportunity to ask any clarifying questions about the topics covered this week. No question is too basic.
*   Are you unclear on the difference between a goal-based and a utility-based agent?
*   Do you want to discuss the environment properties of a specific example?
*   Are you curious about the limits of prompt engineering?

(In a real course, this would be a live, interactive session).

---

## Part 3: The Course Project - Introduction

The single best way to learn agentic engineering is to build an agent. Over the remainder of this course, you will design, build, and refine your own AI agent. This project will be your opportunity to apply the concepts from each lesson, from PEAS and architectures to reasoning and tool use.

### The Goal
To create a functioning prototype of an AI agent that can reliably perform a useful, multi-step task.

### Project Choices
To provide structure, you will choose **one** of the following three agent concepts to build. All three have been chosen because they require the full range of skills you will learn in this course.

#### **Option 1: The Code Documentation Agent**
*   **Goal:** To automatically write documentation for a given Python function.
*   **Core Task:** The agent will be given a Python file. It must read the file, identify a function that lacks a docstring, understand what the function does, and then write a high-quality docstring explaining the function's purpose, arguments, and return value.
*   **Why it's a good project:** It requires file I/O (actuators/sensors), code understanding (reasoning), and structured text generation.

#### **Option 2: The "ELI5" (Explain Like I'm 5) Research Agent**
*   **Goal:** To research a complex topic and explain it in simple terms.
*   **Core Task:** The user provides a topic (e.g., "Quantum Computing" or "Black Holes"). The agent must use a web search tool to find 1-2 good introductory articles, read them, and then generate a simple, easy-to-understand explanation of the topic, as if explaining it to a child.
*   **Why it's a good project:** It requires tool use (web search), information synthesis (reasoning), and adapting its communication style (prompt engineering).

#### **Option 3: The Automated Personal Chef Agent**
*   **Goal:** To generate a recipe based on the ingredients a user has in their fridge.
*   **Core Task:** The user provides a list of ingredients (e.g., "chicken breast, rice, tomatoes, and spinach"). The agent must reason about what kind of meal can be created, and then generate a step-by-step recipe for that meal.
*   **Why it's a good project:** It requires creative problem-solving (reasoning), handling unstructured input, and generating structured output (a recipe).

---

## Your First Assignment: The Project Proposal

By Day 7, you will submit a formal project proposal for the agent you have chosen. This proposal must be a markdown file and must contain the following sections:

1.  **Project Title:** e.g., "Proposal for the ELI5 Research Agent"
2.  **Chosen Agent:** State which of the three options you have chosen.
3.  **High-Level Goal:** In one sentence, what is the primary goal of your agent?
4.  **Detailed PEAS Framework:**
    *   **Performance Measure:** How will you judge if your agent is successful? Be specific.
    *   **Environment:** Describe the agent's environment. Is it software-based? What are its properties (static/dynamic, etc.)?
    *   **Actuators:** What are the specific actions your agent will take? (e.g., `write_file()`, `run_shell_command('curl...')`, etc.).
    *   **Sensors:** How will your agent perceive its environment? (e.g., reading a file, processing the output of a shell command).
5.  **Initial Architectural Choice:** Based on what you learned on Day 3, which architecture (e.g., model-based, goal-based) do you think is the most appropriate starting point for your agent, and why?

This proposal will be the foundational document for your project. Take your time and think through each section carefully. It will serve as your blueprint for the weeks to come.
