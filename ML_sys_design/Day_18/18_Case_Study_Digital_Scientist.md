# Day 18: Case Study - The Digital Scientist

Today we explore a frontier application of agentic AI: its use as a tool for **scientific discovery and research**. The sheer volume of scientific knowledge is growing exponentially, making it impossible for a human researcher to keep up. A single researcher might read a few hundred papers a year, while millions are published.

Agents that can read, understand, and synthesize this vast corpus of information have the potential to accelerate discovery, uncover hidden connections, and act as a powerful assistant for scientists, analysts, and researchers in any field.

---

## Part 1: The Problem - Information Overload

*   **The Data Deluge:** Every day, thousands of new scientific papers, clinical trial results, patents, and financial reports are published.
*   **Siloed Knowledge:** Discoveries in one field (e.g., materials science) could be revolutionary for another (e.g., battery technology), but researchers may not be aware of them because they are published in different journals and use different terminology.
*   **The "Undiscovered Public Knowledge" Problem:** The answer to a critical question may already exist in published literature, but it's hidden, spread across multiple papers, and not explicitly stated in any single one.

An AI research assistant's primary goal is to solve this information overload problem.

---

## Part 2: Case Study - Perplexity AI & Consensus

Two prominent examples of this type of agent are **Perplexity AI** and **Consensus**. While they have different user interfaces, their core agentic architecture is similar.

*   **Goal:** Answer a user's question by synthesizing information from multiple high-quality sources (academic papers, reputable web sources) and providing citations.
*   **PEAS Framework:**
    *   **Performance Measure:** Accuracy of the answer, relevance of the sources, clarity of the synthesis, and trustworthiness (correctly linking claims to sources).
    *   **Environment:** The entire public internet, with a strong focus on academic search engines like Semantic Scholar, PubMed, ArXiv, and publisher websites. This is a vast, noisy, and partially observable environment.
    *   **Sensors:** A sophisticated web browsing/scraping tool capable of reading HTML, parsing PDFs (often the format of scientific papers), and extracting text.
    *   **Actuators:** Displaying the synthesized text and a list of source links to the user.

### **Architectural Deep Dive - The "Research, Synthesize, Cite" Loop**

These agents follow a multi-step reasoning process that is a perfect example of an advanced RAG and ReAct loop.

1.  **Query Understanding and Decomposition:** The agent first analyzes the user's query. A complex query like "What is the impact of metformin on cellular senescence in mice?" might be broken down into sub-questions: "Define metformin," "Define cellular senescence," "Find studies linking metformin and senescence," "Filter studies for mice models."

2.  **Strategic Retrieval (The "R" in RAG):** This is not a simple web search. The agent uses multiple tools strategically.
    *   **Tool 1: `academic_search_engine`:** It might first query a specialized academic search engine to find high-authority papers.
    *   **Tool 2: `web_browser`:** It might then use a general web browser to find explanatory articles or news reports that provide context.
    *   **Re-ranking and Filtering:** The agent gets back a list of many potential sources. It then uses the LLM to perform a re-ranking step, evaluating the titles and abstracts to predict which sources are most likely to contain the answer. It prioritizes these for full text reading.

3.  **Information Extraction and Synthesis (The "A" and "G" in RAG):**
    *   The agent "reads" the full text of the top-ranked documents.
    *   For each document, it extracts the key claims, findings, and supporting data relevant to the user's original query.
    *   This is the critical step: The agent then **synthesizes** the information from *multiple* sources. It looks for consensus (where multiple papers agree), notes contradictions (where papers disagree), and weaves this information into a coherent, easy-to-read summary.

4.  **Generation with Citation:** As the agent writes its final summary, it meticulously tracks which sentence came from which source document. It then inserts citations directly into the text and provides a formatted bibliography. This is crucial for building user trust and allowing for verification.

This entire process is a complex chain of `Thought -> Action (Search) -> Observation (Results) -> Thought (Re-rank) -> Action (Read) -> Observation (Content) -> Thought (Synthesize) -> Final Answer`.

---

## Part 3: The Design Challenge

Let's design an agent for a slightly different research domain: a **Financial Analyst Assistant**.

**Your Task:** Design an agent that can answer the question: **"Should I invest in Company X?"** This is an open-ended question that a good analyst would answer by looking at multiple types of data. Your agent should not give financial advice, but should *gather and summarize the key information* a human would need to make a decision.

### **Step 1: Decomposing the Problem**
What are the key pieces of information a financial analyst would look for? Your agent needs a plan to find them. Brainstorm a list of sub-questions the agent needs to answer. For example:
*   What is the company's recent stock performance?
*   What is the latest news and sentiment surrounding the company?
*   What are the company's key financial metrics (P/E ratio, revenue growth)?
*   Who are its main competitors?

### **Step 2: Tool Design**
To answer the sub-questions from Step 1, your agent needs specialized tools. Design **three** distinct tools. For each tool, provide:
1.  **Tool Name:** Be specific (e.g., `get_stock_price_history`, `search_financial_news`, `get_company_financials`).
2.  **Description:** A clear description for the LLM. For `search_financial_news`, you might add, "Use this to find recent news articles and analyze their sentiment (positive, neutral, negative) for a given company."
3.  **Input Parameters:** e.g., `company_ticker` (string).
4.  **Output:** What data structure does the tool return? (e.g., a JSON object with a list of articles and their sentiment scores).

### **Step 3: The Multi-Agent Workflow**
A single agent trying to do all of this might get confused. This is a perfect use case for a **hierarchical multi-agent system**. Sketch out a multi-agent workflow.

*   **Orchestrator Agent:** What is its role? (e.g., to take the user's query, call the specialized agents, and synthesize their findings).
*   **Specialist Agents:** Define at least two specialist agents that the orchestrator would call.
    *   **Example Specialist 1: `Quantitative_Analyst_Agent`:** Its job is to use the financial data tools (`get_stock_price_history`, `get_company_financials`) to report on the hard numbers.
    *   **Example Specialist 2: `Qualitative_Analyst_Agent`:** Its job is to use the news search tool (`search_financial_news`) to report on the sentiment, news, and competitive landscape.
*   **Final Output:** The orchestrator agent would then take the reports from both specialist agents and synthesize them into a final "Investment Briefing" for the user, summarizing the quantitative and qualitative findings without giving a final "buy" or "sell" recommendation.

This design challenge mirrors how real-world research agents work: they decompose complex questions, use specialized tools to gather diverse types of information, and synthesize the results into a single, coherent report.
