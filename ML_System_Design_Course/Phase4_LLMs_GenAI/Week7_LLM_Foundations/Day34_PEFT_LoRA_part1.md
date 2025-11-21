# Day 34 (Part 1): Advanced PEFT

> **Phase**: 6 - Deep Dive
> **Topic**: Efficient Tuning
> **Focus**: DoRA, Adapter Fusion, and BitFit
> **Reading Time**: 60 mins

---

## 1. DoRA (Weight-Decomposed LoRA)

### 1.1 The Concept
*   Decompose weight into Magnitude ($m$) and Direction ($V$).
*   $W = m \frac{V}{||V||}$.
*   Apply LoRA to Direction $V$. Train Magnitude $m$ fully.
*   **Result**: Closer to Full Fine-Tuning performance than LoRA.

---

## 2. Other PEFT Methods

### 2.1 IA3
*   Multiply activations by a learned vector (rescaling).
*   Cheaper than LoRA.

### 2.2 BitFit
*   Train *only* the Bias terms.
*   Surprisingly effective for some tasks.

---

## 3. Tricky Interview Questions

### Q1: Can LoRA increase inference latency?
> **Answer**:
> *   **No**: If merged ($W + AB$).
> *   **Yes**: If served dynamically (Load $A, B$ from RAM for each request). Extra matrix mults.

### Q2: How to fine-tune for multiple tasks?
> **Answer**:
> *   **AdapterFusion**: Train adapter for Task A, Task B.
> *   Learn a "Router" (Attention) to mix adapters for a new input.

### Q3: QLoRA Double Quantization?
> **Answer**:
> *   Quantize the *quantization constants*.
> *   Saves extra 0.5 bits per parameter.

---

## 4. Practical Edge Case: LoRA Dropout
*   **Tip**: Use higher dropout (0.1) on LoRA layers because they overfit easily (few params).

