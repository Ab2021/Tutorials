# Day 98: Finance & Trading Agents
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Financial News Sentiment Agent

We will build an agent that reads headlines and scores them.

```python
class SentimentAgent:
    def __init__(self, client):
        self.client = client

    def analyze(self, headline, company):
        prompt = f"""
        Analyze the sentiment of this headline for the company: {company}.
        Headline: "{headline}"
        
        Output JSON:
        {{
            "sentiment": "Positive" | "Negative" | "Neutral",
            "score": 0.0 to 1.0,
            "reasoning": "..."
        }}
        """
        return self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        ).choices[0].message.content

# Usage
headline = "Tesla recalls 2M cars, but stock hits all-time high."
agent = SentimentAgent(client)
print(agent.analyze(headline, "Tesla"))
# Output: {"sentiment": "Mixed", "reasoning": "Recall is bad, but stock price action suggests market doesn't care."}
```

### RAG for 10-K Reports

10-K reports are 100+ pages.
*   **Chunking:** Chunk by "Item" (Item 1A: Risk Factors, Item 7: MD&A).
*   **Query:** "What are the primary risks related to AI?"
*   **Retrieval:** Fetch chunks from "Risk Factors".
*   **Comparison:** "Compare the Risk Factors of 2023 vs 2024." (Requires retrieving both and diffing).

### Table Extraction

Financial data is in Tables.
*   **Problem:** LLMs are bad at reading ASCII tables.
*   **Solution:** Convert PDF Table -> Markdown Table or CSV -> LLM.
*   **Tools:** `unstructured`, `tabula-py`, or GPT-4V (Vision).

### Summary

*   **Latency:** News trading requires ms latency. LLMs are too slow (seconds). LLMs are used for *Research*, not *High-Frequency Trading (HFT)* execution.
*   **Hallucination:** Making up a number in a balance sheet is fatal. Always link to the source page.
