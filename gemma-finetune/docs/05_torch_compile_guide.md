# 5. torch.compile — Complete Guide

## Table of Contents
- [What Is torch.compile?](#what-is-torchcompile)
- [Eager vs Graph Mode](#eager-vs-graph-mode)
- [How torch.compile Works Internally](#how-torchcompile-works-internally)
- [Backends Deep Dive](#backends-deep-dive)
- [Compilation Modes](#compilation-modes)
- [Graph Breaks](#graph-breaks)
- [Dynamic vs Static Shapes](#dynamic-vs-static-shapes)
- [torch.compile + LoRA](#torchcompile--lora)
- [Triton: The Secret Engine](#triton-the-secret-engine)
- [Performance Benchmarks](#performance-benchmarks)
- [Debugging torch.compile](#debugging-torchcompile)
- [When to Use and When to Skip](#when-to-use-and-when-to-skip)

---

## What Is torch.compile?

`torch.compile()` is PyTorch's **model compiler**, introduced in PyTorch 2.0 (2023). It transforms your PyTorch model from interpreted Python code into optimized GPU kernels.

### Before torch.compile (PyTorch 1.x)
```python
# Python executes each line one at a time
y = x @ W          # Launch CUDA kernel #1: matrix multiply
y = y + bias       # Launch CUDA kernel #2: addition
y = torch.relu(y)  # Launch CUDA kernel #3: ReLU

# 3 separate kernel launches = 3 round-trips Python ↔ GPU
# Python overhead between each operation
```

### After torch.compile (PyTorch 2.x)
```python
@torch.compile
def forward(x, W, bias):
    y = x @ W
    y = y + bias
    y = torch.relu(y)
    return y

# torch.compile traces the graph, then generates:
# 1 fused CUDA kernel that does matmul + bias + relu in one shot
# No Python overhead between operations
```

### The Speedup

```
                    Without compile      With compile
Python overhead:    ████████████████     ██
Kernel launches:    ████████████████     ████
Memory transfers:   ████████████         ████
GPU compute:        ████████████████     ████████████████

Total time:         ████████████████████ ████████████████
                    ~1.0x                ~1.5-2.0x faster
```

---

## Eager vs Graph Mode

### Eager Mode (Default PyTorch)

```python
# How Python normally executes PyTorch code:

x = torch.randn(32, 2048)    # Creates tensor, Python knows about it
y = model.layer1(x)            # Python calls layer1, waits for result
z = model.layer2(y)            # Python calls layer2, waits for result

# At each step, Python:
# 1. Interprets the Python bytecode
# 2. Dispatches to the C++ ATen library
# 3. Launches a CUDA kernel
# 4. Returns control to Python
# 5. Repeat

# Advantages:
# ✅ Easy to debug (add print statements anywhere)
# ✅ Dynamic (can change behavior at each step)
# ✅ Supports Python control flow (if/else, loops)
# ✅ No compilation wait time

# Disadvantages:
# ❌ Python interpreter overhead between operations
# ❌ Can't fuse operations across Python boundaries
# ❌ Each op launches a separate GPU kernel
```

### Graph Mode (torch.compile)

```python
# How torch.compile executes:

@torch.compile
def model_forward(x):
    y = model.layer1(x)
    z = model.layer2(y)
    return z

# First call (compilation):
# 1. TorchDynamo traces the Python code
# 2. Builds a computation graph (FX graph)
# 3. Optimizes the graph (fuses ops, eliminates redundancy)
# 4. Backend compiles to optimized GPU code
# 5. Caches the compiled code
# 6. Executes the compiled code
# Takes: 30 seconds - 5 minutes

# Subsequent calls:
# 1. Looks up cached compiled code
# 2. Executes directly on GPU
# Takes: Faster than eager!
```

---

## How torch.compile Works Internally

### The Compilation Pipeline

```
┌────────────────┐
│   Your Model   │  Python code with PyTorch ops
└───────┬────────┘
        ▼
┌────────────────┐
│  TorchDynamo   │  Traces Python bytecode, captures ops
│  (Frontend)    │  Detects graph breaks (unsupported ops)
└───────┬────────┘
        ▼
┌────────────────┐
│   FX Graph     │  Intermediate representation (IR)
│                │  Pure computation graph, no Python
└───────┬────────┘
        ▼
┌────────────────┐
│  AOTAutograd   │  Traces forward AND backward pass
│                │  Creates gradient computation graph
└───────┬────────┘
        ▼
┌────────────────┐
│   Backend      │  Converts IR to optimized GPU code
│  (Inductor/    │  Fuses operations, optimizes memory
│   CUDAGraphs)  │
└───────┬────────┘
        ▼
┌────────────────┐
│ Optimized Code │  Ready-to-run GPU kernels
│                │  Cached for reuse
└────────────────┘
```

### Step 1: TorchDynamo (The Frontend)

TorchDynamo is a Python bytecode analyzer that captures PyTorch operations:

```python
def my_function(x):
    y = x * 2        # TorchDynamo captures: mul(x, 2)
    if y.sum() > 0:   # GRAPH BREAK! Python control flow
        y = y + 1     # TorchDynamo captures: add(y, 1)
    return y

# TorchDynamo splits this into TWO subgraphs at the "if" statement:
# Graph 1: mul(x, 2)
# Python: if statement (runs in eager mode)
# Graph 2: add(y, 1)
```

### Step 2: FX Graph (The IR)

The captured operations become an FX Graph:

```python
# Example FX graph for a simple transformer layer:
graph():
    %x : [#users=1] = placeholder[target=x]
    %linear1 : [#users=1] = call_function[target=torch.nn.functional.linear]
    %relu : [#users=1] = call_function[target=torch.relu](%linear1)
    %linear2 : [#users=1] = call_function[target=torch.nn.functional.linear](%relu)
    return %linear2
```

### Step 3: AOTAutograd

Traces both forward AND backward passes ahead of time:

```
Forward:  x → linear → relu → linear → output
Backward: grad_output → linear_bwd → relu_bwd → linear_bwd → grad_input

Both are compiled together for maximum optimization
```

### Step 4: Backend (Inductor)

Generates optimized GPU code. Example fusion:

```python
# Before optimization (3 kernels):
y = x @ W           # Kernel 1: matmul
y = y + bias         # Kernel 2: add
y = torch.relu(y)    # Kernel 3: relu

# After Inductor optimization (1 kernel):
# Generated Triton kernel:
@triton.jit
def fused_matmul_bias_relu(x, W, bias, output):
    # Does matmul + bias + relu in one GPU kernel
    # Reads memory once, writes once
    # No intermediate tensors needed
```

---

## Backends Deep Dive

### Inductor (Default, Recommended)

```
torch.compile(model, backend="inductor")
```

**How it works:**
1. Takes the FX graph
2. Converts operations to Triton IR
3. Generates Triton GPU kernels
4. Triton compiles to CUDA PTX code
5. PTX runs on the GPU

**Optimizations it performs:**
- Operation fusion (matmul + bias + activation)
- Memory planning (minimize temporary allocations)
- Tiling (process data in cache-friendly chunks)
- Auto-tuning (try different tile sizes, pick fastest)

**Requirements:** Triton package (Linux only)

### CUDA Graphs

```
torch.compile(model, backend="cudagraphs")
```

**How it works:**
1. Records the entire sequence of CUDA operations
2. Replays the recorded sequence (bypasses Python entirely)

**Concept:**
```
Normal execution:
  Python → CUDA kernel → Python → CUDA kernel → Python → CUDA kernel
           ^^^^^           ^^^^^           ^^^^^
           GPU work        overhead        GPU work

CUDA Graphs:
  Python → [CUDA kernel → CUDA kernel → CUDA kernel]
                    Replayed as one unit
                    No Python overhead!
```

**Limitations:**
- Input shapes must be FIXED (same batch size, same seq length)
- Cannot handle dynamic control flow
- Uses more memory (needs to pre-allocate all tensors)

### Eager (Debugging)

```
torch.compile(model, backend="eager")
```

Does nothing — runs in normal eager mode. Useful to check if torch.compile is causing an issue.

### AOT Eager (Debugging)

```
torch.compile(model, backend="aot_eager")
```

Traces the graph (catches graph break errors) but executes in eager mode. Good for debugging without the full compilation overhead.

---

## Compilation Modes

### "default" (Our Choice)

```python
torch.compile(model, mode="default")
```

- **Compilation time:** 1-3 minutes
- **Speedup:** 1.3-1.7x
- **Memory:** Same as eager
- **Best for:** Most use cases, including ours

### "reduce-overhead"

```python
torch.compile(model, mode="reduce-overhead")
```

- **Compilation time:** 2-5 minutes
- **Speedup:** 1.5-2.0x (eliminates Python overhead)
- **Memory:** Uses MORE memory (CUDA graphs pre-allocate)
- **Best for:** Small models with fast iterations
- **Caveat:** Fixed shapes only

### "max-autotune"

```python
torch.compile(model, mode="max-autotune")
```

- **Compilation time:** 10-30 minutes (tries many kernel variants)
- **Speedup:** 1.7-2.5x (picks optimal kernels)
- **Memory:** Same as default
- **Best for:** Final production training where compilation time doesn't matter

### Comparison

```
                    Compile Time    Runtime Speed    Memory Usage
default             ████            ████████████     ████████
reduce-overhead     ████████        ██████████████   ██████████████
max-autotune        ██████████████  ████████████████ ████████
```

---

## Graph Breaks

### What Are Graph Breaks?

A **graph break** occurs when TorchDynamo encounters an operation it can't trace through. It splits the computation into multiple subgraphs.

```python
@torch.compile
def forward(x):
    y = x @ W              # ← Compiled subgraph 1
    print(y.shape)          # ← GRAPH BREAK (Python print)
    z = y + bias            # ← Compiled subgraph 2
    return z

# Instead of one optimized graph, we get two smaller graphs
# with eager-mode Python code between them
# Still faster than full eager, but not optimal
```

### Common Causes of Graph Breaks

| Cause | Example | Solution |
|-------|---------|----------|
| Python print | `print(x.shape)` | Remove or use `torch._logging` |
| Data-dependent control flow | `if x.sum() > 0:` | Restructure code |
| Python built-ins on tensors | `len(x)`, `list(x)` | Use `x.shape[0]` |
| NumPy operations | `np.array(x)` | Use PyTorch ops |
| Custom autograd functions | `ctx.save_for_backward()` | May need rewriting |
| some PEFT operations | LoRA adapter switching | Use `fullgraph=False` |

### fullgraph=True vs False

```python
# fullgraph=True: STRICT mode
torch.compile(model, fullgraph=True)
# If ANY graph break occurs → ERROR
# Use this to ensure maximum optimization
# ❌ Often fails with LoRA/PEFT

# fullgraph=False: LENIENT mode (our default)
torch.compile(model, fullgraph=False)
# Graph breaks are OK → falls back to eager for those parts
# Still optimizes everything it can
# ✅ Works with LoRA/PEFT
```

---

## Dynamic vs Static Shapes

### The Problem

```python
# During training, sequence lengths may vary per batch:
batch_1 = tokenizer("Short text", padding=True)       # shape: [1, 64]
batch_2 = tokenizer("A longer review text", padding=True)  # shape: [1, 128]

# torch.compile ASSUMES shapes are fixed!
# If shape changes → recompilation (slow)
```

### Solutions

**1. `dynamic=None` (auto-detect, our default)**
```python
torch.compile(model, dynamic=None)
# PyTorch uses heuristics to decide
# Usually works well for most cases
```

**2. `dynamic=True` (explicit dynamic shapes)**
```python
torch.compile(model, dynamic=True)
# Generates code that handles ANY shape
# Slightly less optimized but no recompilation
```

**3. `dynamic=False` (static shapes)**
```python
torch.compile(model, dynamic=False)
# Assumes all shapes are fixed
# Most optimized code
# Recompiles if shapes change (slow!)
```

**4. Pad to fixed lengths (best approach for training)**
```python
# In our data_loader.py, we pad all sequences to max_seq_length
# This means all training batches have the same shape
# → No recompilation needed!
tokenizer(text, max_length=512, padding="max_length", truncation=True)
```

---

## torch.compile + LoRA

### The Challenge

LoRA introduces dynamic adapter switching that can cause graph breaks:

```python
# Inside PEFT's LoRA forward:
def forward(self, x):
    base_result = self.base_layer(x)  # Compilable ✅
    if self.disable_adapters:          # GRAPH BREAK! ❌ (data-dependent flow)
        return base_result
    lora_result = self.lora_A(x)      # Compilable ✅
    lora_result = self.lora_B(lora_result)  # Compilable ✅
    return base_result + lora_result * self.scaling  # Compilable ✅
```

### Our Solution

```python
# In config.py:
torch_compile_fullgraph = False  # Allow graph breaks
torch_compile_backend = "inductor"  # Still get kernel fusion
torch_compile_mode = "default"  # Balanced optimization
```

**Result:** The compiled subgraphs (matrix multiplications, attention) are still optimized, even though there are graph breaks at LoRA's adapter logic. You get ~1.3x speedup instead of ~1.5x, but it works reliably.

### Alternative: Apply torch.compile to specific layers

```python
# Instead of compiling the entire model:
model = torch.compile(model)  # May have graph breaks

# You can compile individual layers:
for layer in model.model.layers:
    layer.self_attn = torch.compile(layer.self_attn)  # Compile attention only
    layer.mlp = torch.compile(layer.mlp)  # Compile MLP only

# This avoids LoRA-related graph breaks entirely
```

---

## Triton: The Secret Engine

### What Is Triton?

[Triton](https://github.com/openai/triton) is a programming language and compiler for writing GPU kernels. It's created by OpenAI and is what powers `torch.compile`'s `inductor` backend.

### Why Triton Instead of CUDA C++?

```
Writing a GPU kernel:

CUDA C++ (traditional):
  - 100+ lines of code for a matmul kernel
  - Manual memory management
  - Manual thread block sizing
  - Manual shared memory tiling
  - Easy to get wrong

Triton (modern):
  - 20 lines for the same kernel
  - Automatic memory optimization
  - Automatic parallelization
  - Automatic tiling
  - Much simpler
```

### How torch.compile Uses Triton

```
Your PyTorch code
    ↓
TorchDynamo (trace)
    ↓
FX Graph (IR)
    ↓
Inductor (optimization)
    ↓
Triton kernel code (auto-generated)  ← THIS IS WHERE TRITON COMES IN
    ↓
Triton compiler → CUDA PTX
    ↓
GPU execution
```

### Installing Triton

```bash
pip install triton  # Linux only, requires CUDA

# Verify:
python -c "import triton; print(triton.__version__)"
```

**Platform support:**
- ✅ Linux with NVIDIA GPU
- ❌ Windows (not available)
- ❌ macOS (not available)
- ❌ AMD GPUs (experimental)

---

## Performance Benchmarks

### Typical Speedups for LLM Fine-Tuning

```
Model        | Eager | torch.compile | Speedup
─────────────┼───────┼───────────────┼────────
Gemma-2B     | 1.0x  | 1.3-1.5x      | 30-50%
Llama-2-7B   | 1.0x  | 1.4-1.7x      | 40-70%
Mistral-7B   | 1.0x  | 1.3-1.6x      | 30-60%
GPT-2 (1.5B) | 1.0x  | 1.5-2.0x      | 50-100%

Note: With LoRA/PEFT, speedups are typically at the lower end
due to graph breaks. Still worth it for training runs > 30 min.
```

### When torch.compile Pays Off

```
Training time (eager): 60 minutes
Compilation overhead:  5 minutes
Training time (compiled): 40 minutes + 5 min compile = 45 minutes

Net savings: 15 minutes (25% faster overall)

Rule of thumb:
  If training > 20 minutes → torch.compile is worth it
  If training < 10 minutes → skip it (compilation overhead dominates)
```

---

## Debugging torch.compile

### Useful Environment Variables

```bash
# See what torch.compile is doing:
TORCH_LOGS="dynamo" python train.py          # TorchDynamo logs
TORCH_LOGS="inductor" python train.py        # Inductor optimization logs
TORCH_LOGS="graph_breaks" python train.py    # Log all graph breaks

# See generated Triton code:
TORCH_COMPILE_DEBUG=1 python train.py         # Saves to torch_compile_debug/

# Disable torch.compile entirely (for comparison):
TORCH_COMPILE_DISABLE=1 python train.py
```

### Useful Python Debugging

```python
import torch._dynamo

# Reset compilation cache (fix stale compilation issues):
torch._dynamo.reset()

# Get graph break explanation:
torch._dynamo.explain(model, sample_input)

# Count graph breaks:
explanation = torch._dynamo.explain(model, sample_input)
print(f"Graph breaks: {explanation.graph_break_count}")
print(f"Compiled regions: {explanation.graph_count}")
```

---

## When to Use and When to Skip

### Use torch.compile When

| Situation | Reason |
|-----------|--------|
| Training > 20 minutes | Compilation overhead amortized |
| On Linux with NVIDIA GPU | Full Triton support |
| Using PyTorch 2.2+ | Mature, stable |
| Fixed input shapes | Maximum optimization |
| **Our project** | **Training ~30-60 mins → worth it** |

### Skip torch.compile When

| Situation | Reason |
|-----------|--------|
| Quick experiments (< 10 min) | Compilation overhead dominates |
| On Windows or macOS | Triton not available |
| Frequent shape changes | Constant recompilation |
| Debugging model issues | Eager mode easier to debug |
| Using very new/custom ops | May not be supported |
| Google Colab (sometimes) | Can be unstable |

### Decision Flowchart

```
        Will training take > 20 minutes?
                    │
            ┌───────┴───────┐
            │               │
           YES             NO
            │               │
            ▼               ▼
    Are you on Linux     Skip compile
    with NVIDIA GPU?     Set: use_torch_compile = False
            │
      ┌─────┴─────┐
      │           │
     YES         NO
      │           │
      ▼           ▼
  Use compile!   Skip compile
  Backend:       
  "inductor"     
```
