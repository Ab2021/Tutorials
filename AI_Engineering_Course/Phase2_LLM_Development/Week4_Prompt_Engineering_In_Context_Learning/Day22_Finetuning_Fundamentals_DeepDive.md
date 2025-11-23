# Day 22: Fine-tuning Fundamentals (SFT)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. The Mathematics of SFT Loss

The objective function is identical to Pre-training (CLM), but the **mask** is different.
$$ L(\theta) = - \sum_{t=1}^T m_t \log P(x_t | x_{<t}; \theta) $$
- $x$: The concatenated sequence `[Prompt, Response]`.
- $m_t$: The loss mask at step $t$.
    - If $x_t$ belongs to Prompt: $m_t = 0$.
    - If $x_t$ belongs to Response: $m_t = 1$.

**Why Masking Matters:**
If we don't mask the prompt, the model learns to predict the prompt tokens.
Since prompts are often repetitive (e.g., "You are a helpful assistant..."), the model minimizes loss by memorizing these prefixes.
This dilutes the gradient signal for the actual task (generating the response).

### 2. Data Collator Implementation

The Data Collator is responsible for:
1.  Padding the batch to the longest sequence.
2.  Creating the `labels` tensor.
3.  Setting `labels` to `-100` (PyTorch ignore index) for prompt tokens and padding tokens.

**Algorithm:**
1.  Tokenize `Prompt` -> `[p1, p2, p3]`.
2.  Tokenize `Response` -> `[r1, r2, r3, EOS]`.
3.  Concatenate: `Input = [p1, p2, p3, r1, r2, r3, EOS]`.
4.  Create Labels: `[-100, -100, -100, r1, r2, r3, EOS]`.
5.  Pad both `Input` and `Labels` to max length.

### 3. Packing in SFT

Just like pre-training, we can pack multiple (Prompt, Response) pairs into one sequence to save compute.
**Challenge:** Masking becomes complex.
`[P1, R1, EOS, P2, R2, EOS]`
Labels: `[-100, R1, EOS, -100, R2, EOS]`
We must ensure P2 doesn't attend to R1 (Block Diagonal Masking), OR we accept some cross-contamination (standard practice in many open-source trainers like Axolotl).

### 4. The "Alignment Tax"

Fine-tuning often reduces the model's diversity and creativity.
- **Base Model:** High entropy, diverse outputs.
- **SFT Model:** Low entropy, converges to the "average" style of the annotators.
- **Mode Collapse:** The model starts giving the same safe answer to everything.

### Code: SFT Data Collator

```python
import torch
from transformers import DataCollatorForLanguageModeling

class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        # Pad inputs
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        # Pad labels (with -100)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def preprocess(example, tokenizer):
    # Example: {'prompt': 'Hi', 'response': 'Hello'}
    prompt_ids = tokenizer.encode(example['prompt'], add_special_tokens=False)
    response_ids = tokenizer.encode(example['response'], add_special_tokens=False) + [tokenizer.eos_token_id]
    
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    
    return dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(labels))
```
