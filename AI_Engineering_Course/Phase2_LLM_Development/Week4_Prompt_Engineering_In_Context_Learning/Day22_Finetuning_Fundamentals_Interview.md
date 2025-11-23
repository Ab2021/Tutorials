# Day 22: Fine-tuning Fundamentals (SFT)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why do we mask the user prompt in the loss function during SFT?

**Answer:**
- **Goal:** We want the model to learn to *answer* questions, not to *ask* them.
- **Mechanism:** If we calculate loss on the prompt, the model minimizes loss by memorizing the prompt distribution (e.g., "You are a helpful...").
- **Consequence:** This dilutes the gradient signal. The model might become better at generating prompts but not better at following instructions. Masking ensures 100% of the gradient signal comes from the desired behavior (the response).

#### Q2: What is the difference between Pre-training and Fine-tuning?

**Answer:**
- **Pre-training:**
    - **Data:** Massive, noisy, unlabeled text (Web).
    - **Goal:** Learn world knowledge, grammar, reasoning.
    - **Result:** Base Model (Next token predictor).
- **Fine-tuning (SFT):**
    - **Data:** Small, high-quality, labeled (Prompt-Response).
    - **Goal:** Learn to follow instructions and adopt a specific format/style.
    - **Result:** Chat/Instruct Model.

#### Q3: Explain "Catastrophic Forgetting" in the context of SFT.

**Answer:**
- **Phenomenon:** When fine-tuning on a specific domain (e.g., Medical), the model's performance on general tasks (e.g., Coding, Math) degrades significantly.
- **Cause:** The optimization trajectory moves the weights into a region that is optimal for Medical data but suboptimal for General data.
- **Solution:** Use a **Replay Buffer**. Mix 10-20% of the original pre-training data (or a high-quality subset like Wikipedia) into the fine-tuning dataset to anchor the model's general knowledge.

#### Q4: What is a Chat Template and why is it necessary?

**Answer:**
- **Problem:** Base models only see a stream of tokens. They don't inherently understand "User" vs "Assistant" turns.
- **Solution:** We wrap the conversation in a structured format using special tokens (e.g., `<|im_start|>user ... <|im_end|>`).
- **Necessity:** This allows the model to distinguish between the instructions it must follow and the text it must generate. It also prevents "Prompt Injection" to some degree by delineating system instructions.

#### Q5: How much data do you need for SFT?

**Answer:**
- **LIMA Paper:** "Less Is More for Alignment". showed that 1,000 high-quality examples can be enough to turn a base model into a good assistant.
- **Quality > Quantity:** 1,000 clean, diverse examples are better than 100,000 noisy ones.
- **Domain Adaptation:** Might need more (10k-100k) if teaching new knowledge (e.g., a new programming language), but SFT is generally not for teaching *new* facts, but for *eliciting* existing ones.

---

### Production Challenges

#### Challenge 1: The "Repetition Loop"

**Scenario:** After fine-tuning, your model answers: "The capital of France is Paris. The capital of France is Paris. The capital..."
**Root Cause:**
- **EOS Token:** The training data might be missing the `<EOS>` token at the end of the response. The model never learned to stop.
- **Padding:** The model attended to padding tokens during training (bad masking).
**Solution:**
- Ensure every training example ends with `tokenizer.eos_token_id`.
- Verify the Data Collator correctly masks padding tokens with `-100`.

#### Challenge 2: Overfitting to the Prompt Format

**Scenario:** You trained on "Question: {Q}\nAnswer: {A}". In production, users type "Hey, {Q}". The model fails.
**Root Cause:** The model overfitted to the specific template "Question: ...".
**Solution:**
- **Data Augmentation:** Randomly vary the prompt template during training.
- **Chat Format:** Use a standard, robust chat format (ChatML) instead of ad-hoc strings.

#### Challenge 3: Training is too slow / OOM

**Scenario:** You want to fine-tune LLaMA-70B. You have 4x A100s.
**Analysis:**
- 70B params = 140GB (FP16).
- Optimizer (Adam) = 560GB.
- Total > 700GB. 4x A100 (320GB) is not enough for Full Fine-Tuning.
**Solution:**
- **LoRA (Low-Rank Adaptation):** Freeze the model. Train only adapters (1% params). Reduces VRAM by 3-4x.
- **QLoRA:** Quantize base model to 4-bit. Reduces VRAM further.
- **FSDP/ZeRO-3:** Shard the model across GPUs (if doing FFT).

#### Challenge 4: Evaluation is hard

**Scenario:** Loss is decreasing, but is the model actually better?
**Solution:**
- **Validation Set:** Keep a held-out set of prompts.
- **LLM-as-a-Judge:** Periodically (every epoch) generate responses for the validation set and have GPT-4 score them against the previous epoch's responses.
- **WandB:** Log these generations to Weights & Biases to visually inspect progress.

### Summary Checklist for Production
- [ ] **Data:** Clean, high-quality, diverse.
- [ ] **Format:** Use ChatML or standard template.
- [ ] **Masking:** Ensure only response is trained.
- [ ] **EOS:** Ensure every sample ends with EOS.
- [ ] **Eval:** Set up an automated evaluation pipeline (LLM-as-Judge).
