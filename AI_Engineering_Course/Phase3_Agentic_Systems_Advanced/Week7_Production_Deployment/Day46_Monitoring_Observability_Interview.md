# Day 46: Monitoring & Observability
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What are the key metrics to monitor for LLM services?

**Answer:**
**Latency:**
- TTFT (Time to First Token): <500ms target.
- TPOT (Time Per Output Token): <50ms target.
- Total latency: p95 <2s for interactive.

**Throughput:**
- Requests/second, tokens/second.

**Resources:**
- GPU utilization (>80% target), GPU memory.

**Cost:**
- $ per request, $ per 1K tokens, monthly spend.

**Quality:**
- User satisfaction (thumbs up/down rate).
- Hallucination rate, refusal rate.

#### Q2: Explain the three pillars of observability.

**Answer:**
- **Metrics:** Numerical measurements over time (latency, throughput, GPU util). Aggregated data.
- **Logs:** Event records with context (request started, error occurred). Detailed, searchable.
- **Traces:** Request flow through system (tokenize → model → detokenize). Shows bottlenecks.

**Together:** Metrics show WHAT is wrong, logs show WHY, traces show WHERE.

#### Q3: How do you detect hallucinations in production?

**Answer:**
**Automated Detection:**
- **Fact-checking:** Compare claims against knowledge base.
- **Consistency:** Check if response contradicts itself.
- **Source attribution:** Verify citations are real.

**User Feedback:**
- Thumbs down, explicit hallucination reports.

**Sampling:**
- Manually review random sample (1-5%) of responses.

**LLM-as-Judge:**
- Use another LLM to evaluate for hallucinations.

**Metrics:**
- Track hallucination rate over time. Alert if >5%.

#### Q4: What is the difference between Prometheus and ELK stack?

**Answer:**
- **Prometheus:** Time-series metrics database. Stores numerical data (latency, throughput). Good for dashboards, alerts.
- **ELK (Elasticsearch, Logstash, Kibana):** Log aggregation and search. Stores text logs. Good for debugging, searching events.

**Use Together:** Prometheus for metrics, ELK for logs.

#### Q5: How do you set up cost alerts?

**Answer:**
**Track Cost:**
- Calculate cost per request (input_tokens × input_price + output_tokens × output_price).
- Aggregate by user, model, time period.

**Alert Rules:**
- Daily spend > $1000.
- Single request > $1.
- Monthly spend > budget.
- Cost per request increasing (anomaly detection).

**Implementation:** Prometheus counter for cost, alert rules in Alertmanager.

---

### Production Challenges

#### Challenge 1: High Cardinality Metrics

**Scenario:** You add user_id label to all metrics. Prometheus crashes (too many time series).
**Root Cause:** High cardinality (millions of users = millions of time series).
**Solution:**
- **Aggregate:** Track cost by user in separate system (database), not Prometheus.
- **Sample:** Only track top 100 users in Prometheus.
- **Use Logs:** Store per-user data in logs, not metrics.
- **Rule:** Keep metric labels low cardinality (<1000 unique values).

#### Challenge 2: Alert Fatigue

**Scenario:** You get 100 alerts per day. Team ignores them.
**Root Cause:** Too many alerts, many false positives.
**Solution:**
- **Increase Thresholds:** Only alert if p95 latency >2s (not >1s).
- **Increase Duration:** Alert only if condition persists for 10 minutes (not 1 minute).
- **Reduce Alerts:** Combine related alerts (high latency + high error rate → single "service degraded" alert).
- **Severity Levels:** Critical (page on-call), Warning (email), Info (log only).

#### Challenge 3: Missing Context in Logs

**Scenario:** Error log says "Request failed" but no context (which request? what input?).
**Root Cause:** Unstructured logging, missing request ID.
**Solution:**
- **Structured Logging:** Use JSON format with all context.
- **Request ID:** Add unique request_id to all logs for that request.
- **Correlation:** Use request_id to trace all logs for a request.
- **Context Variables:** Use context variables (Python contextvars) to automatically add request_id.

#### Challenge 4: Slow Queries in Grafana

**Scenario:** Grafana dashboards take 30 seconds to load.
**Root Cause:** Querying too much data (1 year of metrics at 1-second resolution).
**Solution:**
- **Reduce Time Range:** Default to last 24 hours, not 1 year.
- **Increase Interval:** Query at 1-minute intervals, not 1-second.
- **Downsample:** Use recording rules to pre-aggregate data.
- **Limit Series:** Reduce number of time series in query.

#### Challenge 5: Cost Tracking Inaccuracy

**Scenario:** Tracked cost is $5000/month but actual bill is $7000/month.
**Root Cause:** Not tracking all costs (only API calls, not storage, network).
**Solution:**
- **Track All Costs:** API calls, vector DB storage, GPU hours, network egress.
- **Reconcile:** Compare tracked cost with actual bill monthly.
- **Add Buffer:** Assume 10-20% overhead for untracked costs.

### Summary Checklist for Production
- [ ] **Metrics:** Track **TTFT, TPOT, throughput, GPU util, cost**.
- [ ] **Logs:** Use **structured logging** with **request_id**.
- [ ] **Traces:** Implement **distributed tracing** with OpenTelemetry.
- [ ] **Alerts:** Set alerts for **p95 latency >2s**, **error rate >1%**, **cost >budget**.
- [ ] **Dashboards:** Create Grafana dashboards for **latency, throughput, cost, quality**.
- [ ] **Quality:** Track **user satisfaction**, **hallucination rate**.
- [ ] **Low Cardinality:** Keep metric labels <1000 unique values.
