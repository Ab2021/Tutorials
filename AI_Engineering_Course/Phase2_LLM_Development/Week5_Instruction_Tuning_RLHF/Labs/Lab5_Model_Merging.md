# Lab 5: Model Merging

## Objective
Merge a LoRA adapter back into the base model for deployment.
This eliminates inference latency overhead.

## 1. The Script (`merge.py`)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "gpt2"
adapter_path = "./my_adapter" # Assume this exists from Lab 1

# 1. Load Base
base = AutoModelForCausalLM.from_pretrained(base_model_name)

# 2. Load Adapter
model = PeftModel.from_pretrained(base, adapter_path)

# 3. Merge
merged_model = model.merge_and_unload()

# 4. Save
merged_model.save_pretrained("./merged_model")
```

## 2. Analysis
The `merged_model` is now a standard `AutoModelForCausalLM`.
It requires no `peft` library to run.

## 3. Submission
Submit the file size of the merged model folder.
