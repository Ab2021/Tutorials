# Day 63: Edge Deployment & Mobile LLMs
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Converting Model to GGUF (Llama.cpp)

GGUF is the standard format for CPU/Edge inference. It supports k-quantization (mixing precisions).

```bash
# 1. Install llama.cpp dependencies
pip install gguf protobuf sentencepiece

# 2. Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 3. Convert HF model to GGUF (FP16)
python convert.py --outfile models/phi-3-mini-fp16.gguf \
    --outtype f16 \
    path/to/Phi-3-mini-4k-instruct

# 4. Quantize to INT4 (Q4_K_M)
# Q4_K_M: Medium 4-bit quantization (balanced)
./quantize models/phi-3-mini-fp16.gguf \
           models/phi-3-mini-q4_k_m.gguf \
           Q4_K_M
```

### 2. Running Inference with Llama-cpp-python

Python bindings for Llama.cpp.

```python
from llama_cpp import Llama

# Load model (offload to GPU if available)
llm = Llama(
    model_path="./models/phi-3-mini-q4_k_m.gguf",
    n_gpu_layers=-1, # -1 = all layers to GPU (Metal/CUDA)
    n_ctx=2048,      # Context window
    verbose=False
)

# Inference
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum physics in 1 sentence."}
    ],
    temperature=0.7,
    max_tokens=100
)

print(output['choices'][0]['message']['content'])
```

### 3. WebLLM (WebGPU Inference)

Running LLM in the browser with JavaScript.

```javascript
import * as webllm from "@mlc-ai/web-llm";

async function main() {
    // 1. Initialize Engine
    const engine = new webllm.MLCEngine();
    engine.setInitProgressCallback((report) => {
        console.log("Loading:", report.text);
    });

    // 2. Load Model (downloads from HuggingFace)
    const selectedModel = "Llama-3-8B-Instruct-q4f32_1-MLC";
    await engine.reload(selectedModel);

    // 3. Chat
    const messages = [
        { role: "system", content: "You are a browser assistant." },
        { role: "user", content: "What is 2+2?" }
    ];

    const reply = await engine.chat.completions.create({
        messages,
        temperature: 0.5,
        max_tokens: 50,
    });

    console.log(reply.choices[0].message.content);
}

main();
```

### 4. On-Device RAG with LanceDB (Embedded)

Implementing a local vector search.

```python
import lancedb
from sentence_transformers import SentenceTransformer
import pandas as pd

# 1. Setup DB
db = lancedb.connect("./data/my_lancedb")
model = SentenceTransformer('all-MiniLM-L6-v2') # Quantize this for mobile!

# 2. Create Data
data = [
    {"text": "Meeting with Alice at 2pm", "id": 1},
    {"text": "Buy milk and eggs", "id": 2},
    {"text": "Project deadline is Friday", "id": 3},
]
df = pd.DataFrame(data)
df['vector'] = model.encode(df['text'].tolist()).tolist()

# 3. Create Table
tbl = db.create_table("notes", data=df, mode="overwrite")

# 4. Search
query = "What do I need to buy?"
query_vec = model.encode(query).tolist()

results = tbl.search(query_vec).limit(1).to_pandas()
print("Context:", results['text'].values[0])

# 5. Generation (Hybrid)
# Pass this context to the local LLM (Phi-3)
```

### 5. Memory Estimation for Edge

How to calculate if a model fits on a phone.

```python
def estimate_memory(params_billion, quant_bits, context_len, hidden_dim):
    # 1. Weights Memory
    weights_gb = (params_billion * 1e9 * quant_bits / 8) / 1e9
    
    # 2. KV Cache Memory (FP16)
    # 2 * layers * hidden * seq * batch * 2 bytes
    # Simplified approximation:
    kv_cache_gb = (2 * 32 * hidden_dim * context_len * 2) / 1e9 
    # (Assuming 32 layers)
    
    # 3. Activation Overhead (Buffer)
    overhead_gb = 0.5 # Rough estimate
    
    total_gb = weights_gb + kv_cache_gb + overhead_gb
    return total_gb

# Example: Llama-3-8B (8B params), 4-bit, 4k context
# Weights: 8 * 0.5 = 4GB
# KV Cache: ~0.5GB
# Total: ~4.5GB -> Fits on 8GB iPhone
print(f"Estimated: {estimate_memory(8, 4, 4096, 4096):.2f} GB")
```

### 6. Battery Optimization Strategy

Logic for "Race to Sleep".

```python
def process_queue_efficiently(queue):
    """
    Instead of processing 1 token every second (keeps CPU awake),
    Wait for queue to fill or timeout, then process batch.
    """
    BATCH_SIZE = 4
    TIMEOUT = 0.5 # seconds
    
    buffer = []
    last_time = time.time()
    
    while True:
        if not queue.empty():
            buffer.append(queue.get())
            
        # Condition to wake up NPU
        if len(buffer) >= BATCH_SIZE or (time.time() - last_time > TIMEOUT and buffer):
            run_inference_batch(buffer) # High power burst
            buffer = []
            last_time = time.time()
            # NPU goes to sleep immediately after
        else:
            time.sleep(0.1) # Low power wait
```

### 7. CoreML Conversion (Apple)

```python
import coremltools as ct
import torch

# Trace PyTorch model
traced_model = torch.jit.trace(model, dummy_input)

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=dummy_input.shape)],
    compute_units=ct.ComputeUnit.ALL # Use ANE + GPU + CPU
)

mlmodel.save("model.mlpackage")
```
