# Day 9: Agents with Tools - From Thinker to Doer

So far, our agent can reason and access a knowledge base (RAG), but it's still fundamentally a "thinker." It processes information and generates text. To become a true "doer," an agent needs tools.

A **tool** is any resource outside the LLM that the agent can use to acquire information or take action. Giving an agent tools is what allows it to break out of the "text-in, text-out" box and interact with the world.

---

## Part 1: Why Do Agents Need Tools?

LLMs, for all their power, have fundamental limitations that tools can solve:

1.  **The Knowledge Cutoff:** An LLM's knowledge is frozen at the time of its training. It doesn't know about today's news, stock prices, or weather.
    *   **Tool Solution:** A **Web Search Tool** gives the agent access to real-time information.

2.  **Lack of Precision:** LLMs are not good at precise mathematical calculations. They "guess" at arithmetic rather than calculating it.
    *   **Tool Solution:** A **Calculator Tool** allows the agent to offload precise calculations to a system that can do them perfectly.

3.  **Inability to Act:** An LLM cannot, by itself, send an email, save a file, or book a flight. Its only ability is to generate text.
    *   **Tool Solution:** An **Email API Tool**, a **File System Tool**, or a **Flight Booking API Tool** give the agent the actuators it needs to perform real-world actions.

4.  **The Grounding Problem:** An LLM's understanding of your specific business is limited. It doesn't know your customers, your inventory, or your internal processes.
    *   **Tool Solution:** A **Database Query Tool** or an **Internal API Tool** connects the agent to your proprietary data, grounding it in the reality of your business.

---

## Part 2: The ReAct Framework - Reason and Act

How does an agent decide when to use a tool? This is where the **ReAct** framework comes in. It's a simple but powerful prompting strategy that combines **Reasoning** (using Chain of Thought) with **Action** (using tools).

The ReAct loop encourages the LLM to generate a specific sequence of text:
1.  **Thought:** The agent thinks about the problem and what it needs to do next. It forms a plan.
2.  **Action:** The agent decides to use one of its available tools and specifies the input to that tool.
3.  **Observation:** The system executes the tool with the specified input and pastes the tool's output back into the context.
4.  **Repeat:** The agent then sees the `Observation` and loops back to step 1 (`Thought`), using the new information to decide on its next action, until it has enough information to give a final answer.

### **Example: A Simple ReAct Interaction**

Let's say the agent has access to two tools: `Search[query]` and `Calculator[expression]`.

**User Query:** "Who was the US president when the first iPhone was released, and what is their current age?"

---
**Thought 1:** I need to find out when the first iPhone was released. Then I need to find who was president at that time. Finally, I need to calculate their current age. I'll start by searching for the iPhone's release date.
**Action 1:** `Search[first iPhone release date]`

**(System executes the search tool and gets the following result)**
---
**Observation 1:** The first iPhone was released on June 29, 2007.

---
**Thought 2:** OK, the iPhone was released in 2007. Now I need to find out who was the US president in 2007.
**Action 2:** `Search[US president in 2007]`

**(System executes the search tool)**
---
**Observation 2:** George W. Bush was the US president for all of 2007. His birth date is July 6, 1946.

---
**Thought 3:** George W. Bush was president. He was born on July 6, 1946. The current year is 2025. I need to calculate his age. I will use the calculator tool. 2025 - 1946.
**Action 3:** `Calculator[2025 - 1946]`

**(System executes the calculator tool)**
---
**Observation 3:** 79

---
**Thought 4:** I have all the information. The president was George W. Bush, and his current age is 79. I can now give the final answer.
**Final Answer:** George W. Bush was the US president when the first iPhone was released in 2007. His current age is 79.

This loop of `Thought -> Action -> Observation` is the fundamental pattern for building agents that can use tools to solve problems.

---

## Part 3: Modern Function Calling

The ReAct framework is a prompting strategy. In the past, developers had to write code to parse the `Action:` line from the LLM's text output to actually execute the tool. This was brittle and error-prone.

Modern LLMs (like Gemini and recent GPT models) now have **native function calling** capabilities built in.

**How it works:**
1.  **Provide Tool Specifications:** When you call the LLM, you also provide a list of the tools (functions) the agent is allowed to use, along with a schema describing what each function does, what its parameters are, and what it returns. This is typically provided in a JSON format.
2.  **LLM Decides:** The LLM processes the user's prompt. If it decides that it needs to use one of the tools, it doesn't just generate text saying so. It outputs a special, structured object that says, "I want to call the `Search` function with the `query` parameter set to 'first iPhone release date'."
3.  **You Execute:** Your code receives this structured object. You then call your actual `Search` function with the provided arguments.
4.  **You Respond:** You take the return value from your function (the search results) and send it back to the LLM in a new API call, letting it know what the observation was.
5.  **LLM Continues:** The LLM receives the observation and continues its reasoning process, deciding whether to call another function or generate a final answer.

This native support makes building tool-using agents much more robust and reliable.

---

## Part 4: Designing Good Tools

The quality of your agent is often determined by the quality of its tools.

**Principles of Good Tool Design:**
*   **Do One Thing Well:** Each tool should have a clear, specific purpose. Avoid creating a single, massive tool that does many different things.
*   **Good Descriptions are Crucial:** The LLM decides which tool to use based *only* on the description you provide in the schema. Be clear, concise, and descriptive.
    *   *Bad Description:* "Search tool"
    *   *Good Description:* "A tool for searching the public web for real-time information about news, events, and people. Use this for any questions about current events."
*   **Use Simple Arguments:** Tools should take simple arguments like strings and numbers. Avoid complex, nested objects as inputs if possible.
*   **Return Informative Outputs:** The tool's output (the `Observation`) should be clear and easy for the LLM to understand. If a tool fails, it should return a clear error message (e.g., "API request failed: 404 Not Found") so the agent can reason about the failure.

---

## Activity: Design a Tool for Your Agent

For the agent you chose for your course project (**Code Documenter**, **ELI5 Researcher**, or **Personal Chef**), design one key tool it would need.

1.  **Agent:** Which agent are you designing for?
2.  **Tool Name:** Give your tool a clear, programmatic name (e.g., `get_file_contents`, `search_web`, `find_recipes_by_ingredient`).
3.  **Tool Description:** Write a high-quality description for your tool. This is what the LLM will see, so make it good!
4.  **Input Parameters:** What arguments does your tool take? For each argument, specify its name, its type (e.g., string, integer), and a description of what it is.
5.  **Output (Return Value):** What does the tool return on success? What does it return on failure?
