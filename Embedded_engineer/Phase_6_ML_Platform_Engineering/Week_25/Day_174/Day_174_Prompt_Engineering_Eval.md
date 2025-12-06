# Day 174: Beyond Strings: Programming Prompts with DSPy
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 25: Large Language Model Infrastructure

---

> **🎯 Focus Area:** "You are a helpful assistant..." is fragile. **DSPy (Declarative Self-improving Python)** treats prompts as *Optimization Problems*. Instead of manually tuning strings, you define inputs/outputs and let the compiler optimize the prompt.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Refactor** brittle f-string prompts into DSPy Signatures and Modules.
2.  **Compile** a DSPy program to automatically find the best Few-Shot examples.
3.  **Evaluate** RAG pipelines using **Ragas** metrics (Faithfulness, Context Recall).
4.  **Implement** a "Chain of Thought" module without writing a single line of prompt text.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install dspy-ai ragas datasets`.

---

## 📖 Theoretical Foundation

### 1. The Death of Prompt Engineering
*   **Old Way:** Tweak "Please think step by step" vs "Take a deep breath". (Voodoo).
*   **DSPy Way:** Define `Question -> Answer`. Define Metric `correctness(prediction, gold)`. Run `BootstrapFewShot`. The system tests 50 variations and selects the best prompts + examples.

### 2. Ragas Metrics
*   **Faithfulness:** Is the answer derived *only* from the retrieved context? (Hallucination check).
*   **Answer Relevancy:** Does the answer actually address the user's question?
*   **Context Precision:** Did the retriever rank the relevant chunk at the top?

---

## 💻 Implementation

### 👨‍💻 Core Implementation: DSPy Module

Replacing f-strings with Classes.

#### 📁 `src/dspy_intro.py`
```python
import dspy

# 1. Configure Language Model
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo)

# 2. Define Signature (The Interface)
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# 3. Define Module (The Logic)
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought augments BasicQA with "reasoning" field automatically
        self.prog = dspy.ChainOfThought(BasicQA)
        
    def forward(self, question):
        return self.prog(question=question)

# 4. Run (Zero Shot)
module = CoT()
response = module("What is the capital of France?")
print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")
```

### 👨‍💻 Core Implementation: The Optimizer (Compiler)

Automatically finding good few-shot examples.

#### 📁 `src/dspy_optimize.py`
```python
from dspy.teleprompt import BootstrapFewShot

# 1. Dataset (Training Data)
train_examples = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs('question'),
    dspy.Example(question="Capital of Germany?", answer="Berlin").with_inputs('question'),
    # ... add 20 more ...
]

# 2. Metric (Validation Logic)
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# 3. Compile
# This creates a NEW prompt that includes the "best" examples from train_examples
# that help the model solve harder problems.
teleprompter = BootstrapFewShot(metric=validate_answer)
compiled_module = teleprompter.compile(CoT(), trainset=train_examples)

# 4. Inspect Optimized Prompt
compiled_module.save("optimized_cot.json")
# print(turbo.inspect_history(n=1))
```

### 👨‍💻 Infrastructure: Ragas Evaluation

Auditing the RAG pipeline.

#### 📁 `src/evaluate_rag.py`
```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# 1. Prepare Data (From your RAG Logs)
data = {
    'question': ['Who is the president of US?', 'What is Python?'],
    'answer': ['Joe Biden', 'A snake'],
    'contexts': [['Joe Biden is the 46th president...'], ['Python is a high-level language...']],
    'ground_truth': ['Joe Biden', 'A programming language']
}
dataset = Dataset.from_dict(data)

# 2. Run Eval
results = evaluate(
    dataset,
    metrics=[
        faithfulness,      # Checks if 'answer' supported by 'contexts'
        answer_relevancy,  # Checks if 'answer' fits 'question'
    ]
)

# 3. Report
print(results)
# {'faithfulness': 1.0, 'answer_relevancy': 0.5} 
# Note: "A snake" is not faithful to context if context says "Language", 
# but if context was empty, it might be hallucination.
```

---

## 🔬 Lab Exercise: "The Hallucination Trap"

### Task
Fix a bad RAG system.
1.  **Scenario:** Document says "Apples are Blue." (Fake fact).
2.  **LLM:** "Apples are usually Red, but the document says Blue."
3.  **Ragas Faithfulness:** High (Failed to detect conflict properly).
4.  **Action:** Tune the Prompt via DSPy.
    *   Signature: `Context, Question -> Answer`.
    *   Instruction: "Answer ONLY based on Context. Ignore prior knowledge."
    *   Compile with `BootstrapFewShot`.
5.  **Result:** Optimized prompt strictly adheres to context. LLM says "Apples are Blue." (Correct behavior for RAG, even if factually wrong in real world).

---

## 📖 Advanced Theory: APE (Automatic Prompt Engineering)
DSPy is an implementation of APE.
Instead of gradients updating Weights (Backprop), we have text feedback updating Prompts (Prompt Optimization).
*   **Opro (Optimization by PROmpting):** Ask an LLM to "Propose a better prompt that solves these failed examples".
*   **Result:** The LLM optimizes itself.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Modular AI:** Stop treating prompts as string constants. Treat them as Modules (`dspy.Predict`, `dspy.ChainOfThought`).
2.  **Evaluation First:** You cannot optimize what you do not measure. Implement Ragas metrics *before* you start tweaking prompts.
3.  **Portability:** A DSPy program compiled for GPT-4 usually works reasonably well on Llama-2-70B, or can be re-compiled (re-optimized) for the new model in minutes. Fixed strings break immediately.

### API Summary
```python
dspy.Signature("q -> a")
dspy.ChainOfThought(Signature)
teleprompter.compile(module, trainset)
```

---

**Day 174 Complete** ✅

*Next: Day 175 - Week 25 Review & Project - The Enterprise LLM Platform.*
