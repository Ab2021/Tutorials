# Day 98: Finance & Trading Agents
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is "Look-Ahead Bias" in backtesting?

**Answer:**
Using data from the future to predict the past.
*   **Example:** Training on earnings reports using the "Release Date" timestamp, but in reality, the report was released *after market close*.
*   **Fix:** Strict Point-in-Time architecture.

#### Q2: How do you handle "Non-Stationarity" in financial data?

**Answer:**
Market regimes change. A strategy that worked in 2020 (Bull Market) fails in 2022 (Bear Market).
*   **Solution:** Rolling Window training. Retrain the model every week on the last 6 months of data.

#### Q3: Explain "Sentiment Decay".

**Answer:**
News is only relevant for a short time.
*   **Decay Function:** $Impact_t = Impact_0 \cdot e^{-\lambda t}$.
*   A headline from 3 days ago has near-zero impact on intraday trading.

#### Q4: Why not use GPT-4 for High Frequency Trading (HFT)?

**Answer:**
*   **Latency:** GPT-4 takes ~500ms. HFT operates in microseconds (µs).
*   **Cost:** Too expensive per token.
*   **Use Case:** GPT-4 is for *Low Frequency* (Daily/Weekly) rebalancing or *Research*.

### Production Challenges

#### Challenge 1: The "Fake News" Flash Crash

**Scenario:** A fake tweet says "Explosion at Pentagon". Market drops 1%.
**Root Cause:** Agent blindly trusts the source.
**Solution:**
*   **Source Verification:** Only trade on Tier-1 sources (Bloomberg, Reuters).
*   **Cross-Validation:** Wait for 2 independent sources before executing a trade.

#### Challenge 2: API Rate Limits during Volatility

**Scenario:** Fed announces rates. 10,000 requests hit your agent. OpenAI rate limits you.
**Root Cause:** Spike in volume.
**Solution:**
*   **Provisioned Throughput:** Buy dedicated capacity.
*   **Fallback:** Switch to a local Llama-3 model during spikes.

#### Challenge 3: Explainability to Regulators

**Scenario:** SEC asks "Why did you short Apple?"
**Root Cause:** Black box model.
**Solution:**
*   **Chain of Thought:** Log the reasoning trace ("Shorting because CFO resigned").
*   **Audit Logs:** Immutable logs of every decision.

### System Design Scenario: Earnings Call Summarizer

**Requirement:** Summarize 500 calls/day during earnings season.
**Design:**
1.  **Ingest:** Audio stream from vendor.
2.  **Transcribe:** Whisper-v3 (Batch mode).
3.  **Diarize:** Identify CEO vs Analyst.
4.  **Analyze:** Extract Guidance (Revenue +5%).
5.  **Alert:** If Guidance > Consensus -> Slack Alert.

### Summary Checklist for Production
*   [ ] **Paper Trading:** Run the agent with fake money for 3 months.
*   [ ] **Kill Switch:** A big red button to stop all trading if the agent goes crazy.
*   [ ] **Data License:** Ensure you have the right to use Bloomberg data for ML.
