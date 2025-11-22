# Day 21: Evaluation Metrics for Language Models
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why is Perplexity (PPL) not a good metric for chatbot quality?

**Answer:**
- **PPL measures prediction confidence**, not generation quality.
- A model that repeats "I don't know" to every question might have very low perplexity (high confidence), but it is useless as a chatbot.
- PPL also cannot capture semantic correctness, factual accuracy, or tone.

#### Q2: Explain the difference between BLEU and ROUGE.

**Answer:**
- **BLEU (Precision):** Measures how many n-grams in the *generated* text appear in the *reference* text. Penalizes generating extra garbage. Used for Translation.
- **ROUGE (Recall):** Measures how many n-grams in the *reference* text appear in the *generated* text. Penalizes missing information. Used for Summarization.

#### Q3: What is "Position Bias" in LLM-as-a-Judge?

**Answer:**
- When asking an LLM (like GPT-4) to compare two answers (A and B), it has a statistical tendency to prefer the first answer presented (Answer A).
- **Mitigation:** Run the evaluation twice: once with (A, B) and once with (B, A). If the judge prefers A both times (or B both times), it's a valid win. If it flips, it's a tie.

#### Q4: How does "Pass@k" differ from standard accuracy?

**Answer:**
- Standard accuracy checks if the *single* output is correct.
- Pass@k checks if *at least one* of the top $k$ generated outputs is correct.
- It is crucial for code generation because users often generate multiple suggestions and pick the working one. It reflects the "retry" behavior of coding assistants.

#### Q5: Can you compare the Perplexity of LLaMA-2 and GPT-2 directly?

**Answer:**
- **No.** Unless they use the *exact same tokenizer*.
- PPL is calculated per token. If Tokenizer A splits "Hello" into 1 token and Tokenizer B splits it into 3 tokens, the probability mass is distributed differently.
- You must normalize by word count or byte count to make a fair comparison, but even then, it's tricky.

---

### Production Challenges

#### Challenge 1: Evaluating a RAG System

**Scenario:** You built a RAG bot for internal docs. How do you know if it's good?
**Solution:**
- **Retrieval Metrics:** Hit Rate, MRR (Mean Reciprocal Rank). (Did we find the right doc?)
- **Generation Metrics:** Faithfulness (Did the answer come from the doc?), Relevance (Did it answer the user?).
- **Tool:** RAGAS (Retrieval Augmented Generation Assessment) framework uses GPT-4 to compute these scores.

#### Challenge 2: Detecting Hallucinations

**Scenario:** Your bot makes up facts.
**Solution:**
- **SelfCheckGPT:** Sample multiple responses from the model. If they are all consistent, it's likely fact. If they contradict each other, it's likely hallucination.
- **NLI (Natural Language Inference):** Use a small BERT model trained on NLI to check if the generated sentence entails the source document.

#### Challenge 3: The "Length Bias" in RLHF

**Scenario:** You are training a Reward Model for RLHF.
**Issue:** The Reward Model learns that "Longer = Better" because human labelers often prefer longer, more detailed answers.
**Result:** The model starts rambling.
**Solution:**
- **Length Penalty:** Normalize the reward score by the length of the response.
- **Instruction:** Explicitly instruct labelers to penalize verbosity.

#### Challenge 4: Monitoring in Production (Drift)

**Scenario:** Your model was great last month, but users are complaining now.
**Root Cause:** Data Drift (User queries changed) or Concept Drift.
**Solution:**
- **Cluster Queries:** Use embeddings to cluster user queries. Detect new clusters (topics) appearing.
- **Feedback Loop:** Track "Thumbs Up/Down" rate. If it drops, trigger an alert.

### Summary Checklist for Production
- [ ] **Pre-training:** Track **Perplexity** and **Gradient Norm**.
- [ ] **Chatbot:** Use **LLM-as-a-Judge** (MT-Bench / AlpacaEval).
- [ ] **Code:** Use **Pass@k** (HumanEval).
- [ ] **RAG:** Use **RAGAS** (Faithfulness/Relevance).
- [ ] **Human:** Always have a "Golden Set" of manual evaluations.
