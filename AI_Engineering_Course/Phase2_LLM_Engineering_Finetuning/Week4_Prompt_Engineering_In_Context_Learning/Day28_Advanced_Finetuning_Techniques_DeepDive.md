# Day 28: Advanced Fine-tuning Techniques
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Task Arithmetic & Model Merging

**The Geometry of Loss Landscapes:**
Fine-tuning a pre-trained model moves the weights in a specific direction.
If we fine-tune on Task A, we move to $\theta_A$.
If we fine-tune on Task B, we move to $\theta_B$.
Surprisingly, the vector $\tau_A = \theta_A - \theta_{pre}$ represents the "skill" of Task A.
Adding this vector to another model often transfers the skill.

**TIES-Merging (Trim, Elect Sign, Merge):**
Simple addition $\theta_{pre} + \tau_A + \tau_B$ often causes interference (noise).
**Algorithm:**
1.  **Trim:** Set small values in $\tau$ to zero (top-k pruning). Only keep the most important weight changes.
2.  **Elect Sign:** If $\tau_A$ wants to increase a weight and $\tau_B$ wants to decrease it, there is a conflict. We resolve this by majority vote or summing.
3.  **Merge:** Add the resolved vector to the base model.

### 2. Sample Packing in Multi-Task Learning

**Problem:**
Dataset A has short samples (100 tokens). Dataset B has long samples (2000 tokens).
If we batch naively, we have massive padding for Dataset A.
**Solution:**
Concatenate samples from different tasks into a single sequence of length 4096.
`[TaskA_Sample1] <EOS> [TaskB_Sample1] <EOS> [TaskA_Sample2] ...`
**Attention Masking:**
We must ensure that Task B tokens do not attend to Task A tokens (Block Diagonal Masking).
If we don't mask, we get "Cross-Task Contamination", which can be good (transfer) or bad (confusion).

### 3. NEFTune Math

**Algorithm:**
During the forward pass of the embedding layer:
$$ E' = E + \epsilon $$
where $\epsilon \sim \text{Uniform}(- \frac{\alpha}{\sqrt{Ld}}, \frac{\alpha}{\sqrt{Ld}} )$.
- $L$: Sequence length.
- $d$: Embedding dimension.
- $\alpha$: Scaling factor (e.g., 5).
**Why it works:**
It acts as a regularizer, similar to Adversarial Training. It forces the model to be robust to small perturbations in the input representation, preventing it from memorizing the exact token sequence of the instruction.

### Code: Simple Linear Model Merging

```python
import torch
import copy

def merge_models(base_model, model_a, model_b, alpha=0.5):
    """
    Simple linear merge: (1-alpha)*ModelA + alpha*ModelB
    Or Task Arithmetic: Base + (A-Base) + (B-Base)
    """
    merged_model = copy.deepcopy(base_model)
    
    state_dict_base = base_model.state_dict()
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    
    for key in state_dict_base:
        # Calculate Task Vectors
        tau_a = state_dict_a[key] - state_dict_base[key]
        tau_b = state_dict_b[key] - state_dict_base[key]
        
        # Merge
        merged_model.state_dict()[key] = state_dict_base[key] + tau_a + tau_b
        
    return merged_model

# Note: Real merging libraries (mergekit) handle memory efficiently
# by loading tensors layer-by-layer.
```
