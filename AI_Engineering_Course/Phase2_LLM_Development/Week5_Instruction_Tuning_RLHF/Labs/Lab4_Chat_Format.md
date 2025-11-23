# Lab 4: Chat Format Converter

## Objective
LLMs expect specific prompt formats (ChatML, Alpaca).
We will write a converter.

## 1. The Converter (`convert.py`)

```python
raw_data = [
    {"user": "Hi", "bot": "Hello!"},
    {"user": "Bye", "bot": "See ya."}
]

def to_chatml(data):
    output = []
    for turn in data:
        text = f"<|im_start|>user\n{turn['user']}<|im_end|>\n<|im_start|>assistant\n{turn['bot']}<|im_end|>"
        output.append(text)
    return output

def to_alpaca(data):
    output = []
    for turn in data:
        item = {
            "instruction": turn['user'],
            "input": "",
            "output": turn['bot']
        }
        output.append(item)
    return output

print("ChatML:", to_chatml(raw_data)[0])
print("Alpaca:", to_alpaca(raw_data)[0])
```

## 2. Challenge
Implement **Llama-2 Chat** format: `[INST] <<SYS>>...<</SYS>> ... [/INST]`.

## 3. Submission
Submit the Llama-2 format converter code.
