# Day 28: Advanced Fine-tuning Techniques
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between Multi-Task Learning and Sequential Fine-tuning?

**Answer:**
- **Multi-Task:** Train on Task A and Task B simultaneously (mixed batches). The model finds a solution that satisfies both.
- **Sequential:** Train on Task A, then train on Task B.
- **Problem:** Sequential training often leads to **Catastrophic Forgetting** of Task A. Multi-task avoids this but requires balancing the dataset mixing ratios carefully.

#### Q2: Why does "Task Arithmetic" work?

**Answer:**
- It relies on the "Linear Mode Connectivity" hypothesis.
- Fine-tuning models from the same pre-trained initialization keeps them in the same "basin" of the loss landscape.
- In this basin, the weights are linearly connected. Adding the weight difference vector ($\Delta W$) moves the model in the direction of the new skill without leaving the basin of general capability.

#### Q3: What is "Task Interference" in Multi-Task Learning?

**Answer:**
- When the gradients for Task A and Task B point in opposite directions.
- Updating the weights to improve Task A hurts Task B, and vice versa.
- **Solution:** Gradient Projection (project Task A's gradient to be orthogonal to Task B's) or simply increasing the model size (MoE) to give each task its own capacity.

#### Q4: How does Domain Adaptation differ from RAG?

**Answer:**
- **Domain Adaptation (Fine-tuning):** Internalizes the knowledge into the weights. Good for "vocabulary", "style", and "deep understanding" of a domain (e.g., speaking legalese).
- **RAG:** Provides external knowledge at inference time. Good for "facts", "statistics", and "up-to-date info".
- **Best Practice:** Use Domain Adaptation for the language/reasoning style, and RAG for the specific facts.

#### Q5: What is NEFTune and why should I use it?

**Answer:**
- NEFTune adds noise to embeddings during fine-tuning.
- It prevents the model from over-fitting to the exact syntax of the instruction dataset.
- It forces the model to rely on the semantic meaning of the instruction, leading to more robust and conversational responses.

---

### Production Challenges

#### Challenge 1: Balancing the Data Mix

**Scenario:** You mix 50% Chat and 50% Code. The model becomes bad at both.
**Root Cause:** The Code dataset is much harder/larger, or the Chat dataset is too noisy.
**Solution:**
- **Dynamic Sampling:** Adjust the mixing weights during training based on the loss. If Code loss is high, sample more Code.
- **Annealing:** Start with General data, slowly transition to Specialized data.

#### Challenge 2: Merging Models with Different Tokenizers

**Scenario:** You want to merge LLaMA-2 (Base) with a Mistral-based adapter.
**Result:** Impossible.
**Constraint:** Model merging (Task Arithmetic) only works if the models share the **exact same architecture and initialization**. You cannot merge LLaMA and Mistral.

#### Challenge 3: OOM during Multi-Task Training

**Scenario:** You are training on 10 datasets. The data loader is consuming 100GB RAM.
**Solution:**
- **Streaming:** Use `IterableDataset` to stream data from disk instead of loading it all into RAM.
- **Interleaved Dataset:** Pre-process the datasets into a single interleaved file on disk (Arrow/Parquet) before training.

#### Challenge 4: "Frankenstein" Model after Merging

**Scenario:** You merged a Math Adapter and a Chat Adapter. The model speaks gibberish.
**Root Cause:** The adapters pushed the weights too far in different directions, leaving the "linear connectivity basin".
**Solution:**
- **TIES-Merging:** Prune the conflicting updates.
- **Weight:** Reduce the merging weight ($\lambda$). Instead of $1.0 * \tau$, try $0.5 * \tau$.

### Summary Checklist for Production
- [ ] **Mix:** Use **50% General / 50% Domain** data.
- [ ] **Merge:** Use **MergeKit** to combine adapters.
- [ ] **Noise:** Enable **NEFTune** (noise_alpha=5).
- [ ] **Eval:** Test on **both** tasks after merging.
