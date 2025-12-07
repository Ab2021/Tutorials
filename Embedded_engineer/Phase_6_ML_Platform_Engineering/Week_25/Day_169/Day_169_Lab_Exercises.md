# Days 169-175: Week 25 - LLM Infrastructure Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 169: vLLM Serving

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=2)
params = SamplingParams(temperature=0.8, max_tokens=256)

outputs = llm.generate(["Hello, my name is"], params)
print(outputs[0].outputs[0].text)
```

---

## Day 170: RAG Pipeline

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

---

## Day 171: LoRA Fine-tuning

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(base_model, config)
```

---

## Day 172: Quantization

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
)
```

---

## Day 173: Prompt Engineering

```python
# System prompt template
SYSTEM_PROMPT = """You are a helpful AI assistant.
Always respond in JSON format.
Be concise and accurate."""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_query},
]
```

---

## Day 174-175: Week 25 Project

```yaml
# LLM serving stack
# 1. vLLM for inference
# 2. FAISS for vector search
# 3. Redis for caching
# 4. Prometheus for metrics
```

---

## 📝 Week 25 Summary
| Day | Topic |
|-----|-------|
| 169 | vLLM |
| 170 | RAG |
| 171 | LoRA |
| 172 | Quantization |
| 173 | Prompts |
| 174-175 | Project |
