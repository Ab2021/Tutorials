# Lab: Day 42 - LangChain Basics

## Goal
Build your first AI Chain using LCEL.

## Prerequisites
- `pip install langchain langchain-openai`
- OpenAI API Key.

## Step 1: The Code (`chain.py`)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# 1. Model
model = ChatOpenAI(model="gpt-3.5-turbo")

# 2. Prompt Template
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}.")

# 3. Output Parser (Converts ChatMessage to string)
parser = StrOutputParser()

# 4. Chain (LCEL)
chain = prompt | model | parser

# 5. Invoke
print("--- Joke Bot ---")
topic = input("Enter a topic: ")
response = chain.invoke({"topic": topic})
print(response)
```

## Step 2: Run It
`python chain.py`

## Challenge: Sequential Chain
Build a "Translator & Summarizer" chain.
1.  Chain 1: Translate English text to French.
2.  Chain 2: Summarize the French text.
3.  Combine them: `chain = translate_chain | summarize_chain`.
