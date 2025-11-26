# Lab: Day 41 - Hello LLM

## Goal
Make your first LLM API call.

## Prerequisites
- `pip install openai`
- An OpenAI API Key (or use Ollama for local).

## Step 1: The Code (`chat.py`)

```python
from openai import OpenAI
import os

# Set env var OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat(prompt, system_role="You are a helpful assistant."):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # or gpt-4
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# 1. Basic Chat
print("--- Basic ---")
print(chat("Explain Quantum Computing in one sentence."))

# 2. Persona
print("\n--- Pirate Persona ---")
print(chat("Hello there!", system_role="You are a pirate."))

# 3. Structured Output
print("\n--- JSON Extraction ---")
prompt = """
Extract the name and age from this text:
'My name is John and I am 30 years old.'
Return JSON.
"""
print(chat(prompt, system_role="You are a data extraction bot. Output only JSON."))
```

## Step 2: Run It
`python chat.py`

## Challenge
Build a **CLI Chatbot**.
1.  Use a `while True` loop to accept user input.
2.  Maintain a `history` list of messages `[{"role": "user", ...}, {"role": "assistant", ...}]`.
3.  Pass the full history to the API on every turn so the model "remembers" the conversation.
