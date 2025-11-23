# Lab 3: KV Cache Visualization

## Objective
Understand **KV Caching**.
It's the key to fast autoregressive generation.

## 1. The Calculator (`kv_calc.py`)

```python
# Llama-2-7B Params
d_model = 4096
n_layers = 32
n_heads = 32
head_dim = d_model // n_heads # 128
dtype_size = 2 # float16 = 2 bytes

def calc_kv_size(seq_len, batch_size):
    # 2 for K and V
    # Size = 2 * n_layers * n_heads * head_dim * seq_len * batch_size * dtype_size
    total_bytes = 2 * n_layers * n_heads * head_dim * seq_len * batch_size * dtype_size
    return total_bytes / (1024**3) # GB

batch = 1
seq = 1000
print(f"KV Cache for Batch={batch}, Seq={seq}: {calc_kv_size(seq, batch):.2f} GB")

batch = 32
print(f"KV Cache for Batch={batch}, Seq={seq}: {calc_kv_size(seq, batch):.2f} GB")
```

## 2. Analysis
KV Cache grows linearly with Sequence Length and Batch Size.
It can easily exceed GPU memory (OOM).

## 3. Submission
Submit the KV Cache size for Batch=64, Seq=4096.
