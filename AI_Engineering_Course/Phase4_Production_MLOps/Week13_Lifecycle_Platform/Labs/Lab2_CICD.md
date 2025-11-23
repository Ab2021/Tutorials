# Lab 2: CI/CD for LLMs (GitHub Actions)

## Objective
Prevent regressions.
If you change the prompt, does the accuracy drop?
We will write a GitHub Action that runs `eval.py` on every push.

## 1. The Eval Script (`eval.py`)

(Reuse the script from Week 4 Lab 2)
Ensure it returns `exit(1)` if score < threshold.

```python
import sys
score = 0.9 # Mock
if score < 0.8:
    sys.exit(1)
```

## 2. The Workflow (`.github/workflows/eval.yml`)

```yaml
name: LLM Evaluation

on: [push]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
          
      - name: Install Dependencies
        run: pip install openai
        
      - name: Run Eval
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python eval.py
```

## 3. Running the Lab
1.  Create a new GitHub repo.
2.  Push this file.
3.  Add `OPENAI_API_KEY` to Repo Secrets.
4.  Check the "Actions" tab.

## 4. Submission
Submit a link to the GitHub Action run.
