# Lab 4: Dynamic Few-Shot Selector

## Objective
Few-shot performance depends on the *quality* of examples.
We will select examples semantically related to the query.

## 1. The Selector (`selector.py`)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

examples = [
    "Input: 2+2, Output: 4",
    "Input: Capital of France, Output: Paris",
    "Input: 5*5, Output: 25",
    "Input: Capital of Spain, Output: Madrid"
]

# 1. Index
db = Chroma.from_texts(examples, OpenAIEmbeddings())

# 2. Query
query = "Capital of Germany"
docs = db.similarity_search(query, k=2)

# 3. Construct Prompt
selected_examples = "\n".join([d.page_content for d in docs])
prompt = f"""
Examples:
{selected_examples}

Input: {query}
Output:
"""

print(prompt)
```

## 2. Analysis
For a math query, it should select math examples.
For a geography query, it should select geography examples.

## 3. Submission
Submit the generated prompt for a Math query.
