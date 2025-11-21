# Day 13: Transfer Learning - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: Catastrophic Forgetting, L2-SP, and PEFT

## 1. Catastrophic Forgetting

When fine-tuning on Task B, the model forgets Task A.
Why? Weights move to a new optimum far from the old one.
**Solution**:
*   **L2-SP (Starting Point)**: Regularize weights to stay close to pre-trained weights.
    $$ \Omega(w) = \frac{\alpha}{2} ||w - w_{pretrained}||^2 + \frac{\beta}{2} ||w||^2 $$
*   **Elastic Weight Consolidation (EWC)**: Penalize changing weights that were important for Task A (using Fisher Information Matrix).

## 2. Parameter Efficient Fine-Tuning (PEFT)

Fine-tuning a 1B parameter model is expensive (storage & memory).
PEFT methods update only a tiny subset of parameters.

### Adapters
Insert small MLP layers *between* frozen pre-trained layers.
Only train the adapters.

### LoRA (Low-Rank Adaptation)
Inject trainable rank decomposition matrices into dense layers.
$$ W_{new} = W_{frozen} + B A $$
*   $W \in \mathbb{R}^{d \times d}$
*   $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$ (where $r \ll d$, e.g., $r=8$).
*   Trainable params reduced by 10,000x.

## 3. Batch Normalization in Transfer Learning

**Trap**: When fine-tuning, should you freeze BatchNorm stats (`running_mean`, `running_var`)?
*   **Yes (Usually)**: If target batch size is small (2-4), updating stats will destroy the model. Set `model.eval()` or freeze BN.
*   **No**: If target domain is very different (Medical), you *must* update stats to match new distribution.

## 4. Linear Probing vs Fine-Tuning

Paper: "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution".
*   **Linear Probing** (Frozen backbone): Preserves robust features. Better OOD generalization.
*   **Fine-Tuning**: Higher accuracy on ID (In-Distribution) data, but risks overfitting.
*   **Best Practice**: LP first, then FT (LP-FT).
