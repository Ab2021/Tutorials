# Day 98: Finance & Trading Agents
## Core Concepts & Theory

### The High-Speed Domain

Finance is about **Information Asymmetry**. Whoever processes information faster and better wins.
*   **Tasks:** Sentiment Analysis, Earnings Call Summarization, Fraud Detection, Algo Trading.

### 1. Financial Sentiment Analysis (FinBERT)

"Apple crushed earnings" -> Positive.
"Apple crushed by lawsuit" -> Negative.
Standard BERT fails here. **FinBERT** (trained on financial news) understands the nuance.
*   **Aspect-Based Sentiment:** "Revenue up, but Guidance down." (Positive for Revenue, Negative for Guidance).

### 2. Earnings Call Analysis

*   **Input:** 1-hour audio recording of CEO/CFO.
*   **Process:** ASR (Whisper) -> Diarization -> Summarization.
*   **Insight:** "Did the CEO sound confident?" (Audio analysis) + "What did they imply about Q4?" (Text analysis).

### 3. Quantitative Trading (Alpha Generation)

LLMs can generate **Alpha** (Trading Signals).
*   **News Trading:** Read headlines -> Predict stock move.
*   **Code Generation:** "Write a Python script to backtest a Moving Average Crossover strategy."
*   **Data Cleaning:** Cleaning messy Excel sheets.

### 4. Fraud Detection

*   **Pattern Matching:** "This transaction looks weird."
*   **Graph Analysis:** "These 5 accounts are linked to the same IP."
*   **LLM Role:** Explaining the fraud score to the analyst. "Flagged because location mismatch + high value."

### Summary

Finance Agents are **Analysts**. They read the 100-page 10-K report so you don't have to. In trading, they are **Signal Generators**.
