# Day 45: Building AI Agents

## 1. From Chatbot to Agent

*   **Chatbot**: Passive. Answers questions.
*   **Agent**: Active. Takes actions. "Book me a flight."

---

## 2. The ReAct Pattern (Reason + Act)

How does an LLM solve complex tasks?
1.  **Thought**: "I need to find the weather in London."
2.  **Action**: `get_weather("London")`
3.  **Observation**: "It is raining."
4.  **Thought**: "I should tell the user to bring an umbrella."
5.  **Final Answer**: "Bring an umbrella."

---

## 3. Tools (Function Calling)

We give the LLM a list of functions it can call.
*   `search_google(query)`
*   `calculator(expression)`
*   `sql_query(query)`

The LLM outputs JSON: `{"tool": "calculator", "args": "5 * 5"}`.
We execute it and feed the result back.

---

## 4. LangGraph (Stateful Agents)

LangChain is a DAG (Directed Acyclic Graph). Agents are loops.
*   **LangGraph**: A library to build cyclic graphs.
*   **State**: The agent maintains state (messages, variables) as it loops through steps.
*   **Human-in-the-loop**: Pause execution, ask human for approval, resume.

---

## 5. Summary

Today we gave the AI hands.
*   **ReAct**: The brain loop.
*   **Tools**: The hands.
*   **LangGraph**: The nervous system.

**Week 9 Wrap-Up**:
We have covered:
1.  LLM Foundations (Prompting).
2.  LangChain (Orchestration).
3.  RAG (Knowledge).
4.  Vector DBs (Memory).
5.  Agents (Action).

**Next Week (Week 10)**: We enter the world of **Event-Driven Architecture**. Kafka, RabbitMQ, and Real-Time Systems.
