# Day 42: LangChain & Orchestration

## 1. Why LangChain?

Raw OpenAI API calls are messy.
*   You have to manage history manually.
*   You have to parse strings into JSON manually.
*   You have to glue multiple calls together manually.
*   **LangChain** is the framework to solve this.

---

## 2. Core Components

### 2.1 Models (LLMs & ChatModels)
Wrappers around APIs (OpenAI, Anthropic, HuggingFace).
*   `chat = ChatOpenAI(temperature=0)`

### 2.2 Prompts (PromptTemplates)
Dynamic templates.
*   `template = "Translate {text} to {language}."`
*   `prompt = PromptTemplate.from_template(template)`

### 2.3 Output Parsers
Convert string output to Python objects.
*   `PydanticOutputParser`: Guarantees the LLM returns data matching a Pydantic class.

---

## 3. Chains

Linking steps together.
*   **LLMChain**: Prompt + LLM.
*   **SequentialChain**: Output of Chain A -> Input of Chain B.
    *   Step 1: "Summarize this email."
    *   Step 2: "Draft a reply to the summary."

---

## 4. Memory

LLMs are stateless. Memory stores the conversation history.
*   **ConversationBufferMemory**: Stores everything (expensive).
*   **ConversationSummaryMemory**: Asks the LLM to summarize old messages to save tokens.

---

## 5. Summary

Today we built a pipeline.
*   **LangChain**: The glue for AI apps.
*   **Templates**: Reusable prompts.
*   **Chains**: Complex workflows.

**Tomorrow (Day 43)**: We will give the LLM access to our own data using **RAG (Retrieval Augmented Generation)**.
