# Day 7: Memory and Learning in Agents

A key limitation of LLMs is their finite context window. An LLM can't remember the start of a very long conversation, nor does it have access to your private documents or information that appeared on the web after its training date. For an agent to be truly useful, it needs a **memory**.

Today, we explore how to give our agents both short-term and long-term memory, with a deep dive into the most important architectural pattern in modern agentic AI: **Retrieval Augmented Generation (RAG)**.

---

## Part 1: Types of Agent Memory

When we talk about an agent's memory, we can categorize it into three types:

### 1. **Working Memory (The Context Window)**
*   **What it is:** This is the text that is currently inside the LLM's context window. It's the prompt, the instructions, the few-shot examples, and the immediate back-and-forth of the conversation.
*   **Analogy:** Human short-term memory. It's what you are actively thinking about right now.
*   **Limitation:** It's fast and effective but very small. Once information scrolls out of the context window, it's gone forever.

### 2. **Episodic Memory (Conversation History)**
*   **What it is:** A summary of past conversations or interactions. As a conversation gets too long for the context window, we can use an LLM to summarize the earlier parts and feed that summary into the prompt.
*   **Analogy:** Remembering the gist of a conversation you had yesterday.
*   **Limitation:** Summarization is lossy. Nuances and details can be lost over long interactions.

### 3. **Long-Term Memory (External Knowledge)**
*   **What it is:** A vast, external source of information that the agent can access on demand. This is where we store knowledge that is too large to fit in the context window, such as textbooks, product manuals, company wikis, or a database of past user interactions.
*   **Analogy:** A library or a personal journal. You can't hold it all in your head, but you know how to look things up when you need them.
*   **Implementation:** The primary technique for implementing long-term memory today is **RAG**.

---

## Part 2: Retrieval Augmented Generation (RAG) - The Core Idea

RAG is a simple but incredibly powerful idea that combines the strengths of information retrieval (like a search engine) with the reasoning power of an LLM.

> **The RAG Process:** Before the LLM answers a question, the system first **retrieves** relevant information from an external knowledge source. Then, it **augments** the user's original prompt with this retrieved information and sends the combined text to the LLM.

**Analogy: An Open-Book Exam**
Instead of trying to force the LLM to memorize a book (fine-tuning) or giving it a tiny "cheat sheet" (in-context learning), RAG allows the LLM to have the entire book open in front of it and find the exact page with the relevant information before answering the question.

**Why is RAG so powerful?**
1.  **Reduces Hallucinations:** The LLM is "grounded" by the provided text. It is instructed to answer based on the retrieved context, making it much less likely to invent facts.
2.  **Access to Current Information:** You can constantly update your knowledge source without ever retraining the model, allowing your agent to have up-to-the-minute information.
3.  **Access to Private Data:** You can point the RAG system at your company's private documents, giving the agent expertise on your internal processes without that data ever being sent for training.
4.  **Verifiability:** Since you know what information was retrieved, you can often cite the sources for the agent's answer, increasing trust and allowing for verification.

---

## Part 3: Building a Simple RAG Pipeline

A RAG pipeline has two main stages: the **Indexing Stage** (done once, offline) and the **Retrieval-Generation Stage** (done in real-time for every query).

### **The Indexing Stage: Creating the Library**

1.  **Load Data:** Ingest your documents from various sources (e.g., text files, PDFs, websites).
2.  **Chunk Data:** Break the large documents down into smaller, manageable chunks (e.g., paragraphs or sentences). This is critical because we want to retrieve only the most relevant snippets, not whole documents.
3.  **Embed Chunks:** Use an **embedding model** (a special type of neural network) to convert each chunk of text into a numerical vector. This vector captures the semantic "meaning" of the text. Chunks with similar meanings will have similar vectors.
4.  **Store in Vector Database:** Store these vectors (along with the original text chunks) in a **Vector Database**. A vector database is a specialized database that is highly optimized for finding the most similar vectors to a given query vector. Examples include Chroma, Pinecone, and FAISS.

### **The Retrieval-Generation Stage: Answering the Question**

1.  **Embed Query:** When a user asks a question, take their query text and convert it into a vector using the *same* embedding model.
2.  **Search Vector Database:** Use this query vector to search the vector database. The database will return the `k` most similar text chunks (e.g., the top 3 chunks whose vectors are closest to the query vector). This is the **retrieval** step.
3.  **Augment Prompt:** Create a new prompt by combining the original user query with the retrieved text chunks. A common template looks like this:
    ```
    You are a helpful assistant. Answer the user's question based only on the context provided below.

    CONTEXT:
    [...retrieved chunk 1...]
    [...retrieved chunk 2...]
    [...retrieved chunk 3...]

    USER QUESTION:
    [...original user query...]

    ANSWER:
    ```
4.  **Generate Answer:** Send this augmented prompt to the LLM to get your final answer. This is the **generation** step.

---

## Activity: Design a RAG System

Your project for the week was to write a proposal for your course project agent. Now, let's think about how RAG could apply to it.

For the agent you chose (**Code Documenter**, **ELI5 Researcher**, or **Personal Chef**), answer the following questions:

1.  **Is RAG applicable?** Is there a long-term memory component that would make your agent better? What kind of knowledge would it store?
2.  **The Knowledge Source:** What specific documents or data would you put into your vector database?
    *   *For the Code Documenter:* Would you index existing codebases or documentation about Python libraries?
    *   *For the ELI5 Researcher:* (This is a trick question! The project description involves *live* web search, which is a form of retrieval, but not a classic RAG pipeline with a pre-indexed database. Think about the difference.)
    *   *For the Personal Chef:* Would you index a database of existing recipes? Cooking techniques? Flavor pairings?
3.  **A Sample Interaction:** Write out a hypothetical user query and then show what the "augmented prompt" sent to the LLM might look like after the retrieval step.

This exercise will help you move from the theoretical understanding of RAG to a practical application for your own agent design.
