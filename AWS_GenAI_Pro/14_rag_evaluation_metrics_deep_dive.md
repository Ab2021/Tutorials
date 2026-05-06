# Deep Dive: RAG Evaluation Metrics Across Problem Statements

Evaluating Retrieval-Augmented Generation (RAG) systems requires assessing both the **Retrieval Component** (how well it finds the right information) and the **Generation Component** (how well it uses that information to generate an answer). 

This guide breaks down the core metrics, their underlying concepts, when to use them across different problem statements, and the Python packages required for implementation.

---

## 1. Core Concepts of RAG Evaluation

RAG evaluations are typically structured around the **"RAG Triad"** which evaluates the relationships between three key elements:
1. **User Query (Prompt)**
2. **Retrieved Context**
3. **Generated Answer**

The triad assesses:
- **Context Relevance:** Is the retrieved context relevant to the user query?
- **Groundedness / Faithfulness:** Is the generated answer supported by the retrieved context? (No hallucinations)
- **Answer Relevance:** Does the generated answer actually address the user's query?

To properly evaluate these, you also often need **Ground Truth** (the ideal reference answer or reference context).

---

## 2. Detailed RAG Metrics & Their Concepts

### A. Retrieval Metrics (Evaluating the Retriever)

#### 1. Context Precision
*   **Concept:** Measures whether all the ground-truth relevant items present in the `contexts` are ranked higher or not. It penalizes retrieving relevant documents at lower ranks.
*   **When to use:** When you are retrieving multiple chunks and the order matters (e.g., passing top-K to an LLM with limited context window, prioritizing the highest quality context first).

#### 2. Context Recall
*   **Concept:** Measures the extent to which the retrieved context aligns with the annotated ground truth answer. It answers: "Did we retrieve everything we needed to answer the question?"
*   **When to use:** When complete information is critical (e.g., Legal discovery, Medical diagnosis).

#### 3. Context Relevance (or Context Entity Recall)
*   **Concept:** Measures how much of the retrieved context is actually relevant to the query. Low context relevance means your retriever is pulling in "noise," which costs extra tokens and can confuse the LLM.
*   **When to use:** Cost-optimization, prompt pollution prevention, and strictly focused Q&A tasks.

### B. Generation Metrics (Evaluating the Generator/LLM)

#### 4. Faithfulness (Groundedness)
*   **Concept:** Measures the factual consistency of the generated answer against the retrieved context. If the answer contains claims that cannot be inferred from the context, the faithfulness score is lowered (indicating hallucination).
*   **When to use:** Crucial for **Financial**, **Healthcare**, and **Enterprise** applications where hallucinating facts is unacceptable.

#### 5. Answer Relevance
*   **Concept:** Evaluates how relevant the generated answer is to the prompt. It does not consider factuality, only whether the LLM directly answered the user's question without going on tangents.
*   **When to use:** Conversational AI, Customer Support Chatbots, to ensure the user doesn't get frustrated by evasive or overly verbose answers.

#### 6. Answer Correctness (vs. Ground Truth)
*   **Concept:** Compares the generated answer to a human-annotated ground truth answer. Often calculated using semantic similarity and factual overlap.
*   **When to use:** When you have a golden dataset of Q&A pairs for automated regression testing of your RAG pipeline.

### C. Traditional NLP / Extractive Metrics

#### 7. ROUGE & BLEU
*   **Concept:** Measures pure string/n-gram overlap between the generated text and a reference text. 
*   **When to use:** Only useful for highly extractive problem statements (like exact quote extraction) or summarization. Not recommended for semantic Q&A as LLMs phrase things differently.

#### 8. Semantic Similarity (e.g., BERTScore)
*   **Concept:** Computes the cosine similarity of embeddings between the generated answer and the ground truth.
*   **When to use:** Better than ROUGE/BLEU for checking if the "meaning" of the generated text matches the ground truth.

---

## 3. Metrics Relevance Across Problem Statements

| Problem Statement | Primary Focus | Critical Metrics | Why? |
| :--- | :--- | :--- | :--- |
| **Customer Support Q&A Chatbots** | High accuracy, direct answers, low latency. | Answer Relevance, Faithfulness | Users want quick, accurate answers without hallucinations. Context recall is less critical if a single doc answers it. |
| **Legal/Contract Analysis** | Zero hallucination, exhaustive search. | Context Recall, Faithfulness | Missing a clause (low recall) or hallucinating a legal obligation (low faithfulness) carries massive liability. |
| **Financial Report Summarization** | Accurate extraction, no fabricated numbers. | Faithfulness, Context Precision | The LLM must not invent financial metrics. The best context must be in the top-K to ensure accurate math/aggregation. |
| **Internal Enterprise Search (Wiki)** | Finding the needle in the haystack. | Context Precision, Context Relevance | Users search for very specific policies. The retriever must rank the exact wiki page at position 1. |
| **Code Generation with RAG** | Executable, syntactically correct code. | Answer Correctness, Execution Success Rate | Semantic similarity is less useful here; the code either compiles/runs or it doesn't. |

---

## 4. Packages to Import & How to Use Them

The Python ecosystem has two dominant, purpose-built frameworks for RAG evaluation using "LLMs-as-a-Judge": **Ragas** and **TruLens**.

### A. Ragas (RAG Assessment)
Ragas is excellent for evaluating pipelines when you have an evaluation dataset.

**Installation:** `pip install ragas langchain openai`

**Imports and Usage Example:**
```python
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_recall,
    context_precision,
)

# Ragas expects a HuggingFace Dataset format with: 
# question, answer, contexts, ground_truth
data = {
    "question": ["What is Amazon Bedrock?"],
    "answer": ["Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models..."],
    "contexts": [["Amazon Bedrock is a fully managed service from AWS..."]],
    "ground_truth": ["Bedrock is an AWS service for foundation models."]
}
dataset = Dataset.from_dict(data)

# Run evaluation
result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevance,
    ],
)
print(result)
```

### B. TruLens (by TruEra)
TruLens is great for evaluating RAG applications built with LangChain or LlamaIndex, allowing you to track the RAG triad dynamically.

**Installation:** `pip install trulens_eval`

**Imports and Usage Example:**
```python
from trulens_eval import Tru, Feedback, TruChain
from trulens_eval.feedback.provider import OpenAI
import numpy as np

tru = Tru()
provider = OpenAI()

# Define the RAG Triad Feedbacks
# 1. Answer Relevance
f_qa_relevance = Feedback(provider.relevance_with_cot_reasons).on_input_output()

# 2. Context Relevance
f_qs_relevance = Feedback(provider.qs_relevance_with_cot_reasons).on_input().on(
    TruChain.select.context
).aggregate(np.mean)

# 3. Groundedness (Faithfulness)
f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons).on(
    TruChain.select.context
).on_output()

# Wrap your existing LangChain (or LlamaIndex) app
# assuming `qa_chain` is your LangChain ConversationalRetrievalChain
tru_recorder = TruChain(
    qa_chain,
    app_id='RAG_App_v1',
    feedbacks=[f_qa_relevance, f_qs_relevance, f_groundedness]
)

# Run with context manager to evaluate on the fly
with tru_recorder as recording:
    qa_chain("What is Amazon Bedrock?")

# View results in a Streamlit dashboard
tru.run_dashboard()
```

### C. Traditional Metrics (HuggingFace Evaluate)
For extractive tasks or traditional NLP metrics like BLEU or semantic similarity.

**Installation:** `pip install evaluate rouge_score bert_score`

**Imports and Usage:**
```python
import evaluate

# Load ROUGE
rouge = evaluate.load('rouge')
results = rouge.compute(
    predictions=["The quick brown fox jumps over the lazy dog"],
    references=["A fast brown fox jumps over the lazy dog"]
)
print(results)

# Load BERTScore for Semantic Similarity
bertscore = evaluate.load("bertscore")
results = bertscore.compute(
    predictions=["AWS Bedrock is a managed service."], 
    references=["Amazon Bedrock provides managed foundational models."], 
    lang="en"
)
print(results['f1'])
```

### Summary of Best Practices
1. **Do not rely on ROUGE/BLEU for generative tasks.** They penalize paraphrasing.
2. **Use LLMs-as-a-Judge (Ragas/TruLens) for semantic evaluation.** It correlates much higher with human judgment.
3. **Always evaluate the Retriever independently of the Generator.** If the generator gets the wrong answer, you need to know if it's because the context was missing (Retriever failure) or because the LLM hallucinated (Generator failure).
4. **Tailor the threshold to the problem statement.** A healthcare bot might require a `faithfulness` score of 0.99, while an internal creative writing bot might tolerate 0.80.
