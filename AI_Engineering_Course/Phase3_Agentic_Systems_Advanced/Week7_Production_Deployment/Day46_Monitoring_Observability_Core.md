# Day 46: Monitoring & Observability
## Core Concepts & Theory

### Observability Fundamentals

**Three Pillars:**
- **Metrics:** Numerical measurements over time.
- **Logs:** Event records with context.
- **Traces:** Request flow through system.

**LLM-Specific Observability:**
- Traditional metrics (latency, throughput).
- LLM-specific metrics (token usage, quality).
- Cost tracking (per-token pricing).

### 1. Key Metrics

**Latency Metrics:**
- **TTFT (Time to First Token):** Time until first token generated.
  - **Target:** <500ms for interactive.
- **TPOT (Time Per Output Token):** Average time per token.
  - **Target:** <50ms per token.
- **Total Latency:** TTFT + (num_tokens × TPOT).
  - **Target:** p95 <2s for interactive.

**Throughput Metrics:**
- **Requests/Second:** Total requests processed.
- **Tokens/Second:** Total tokens generated.
- **Batch Size:** Average batch size.

**Resource Metrics:**
- **GPU Utilization:** % of time GPU is busy.
  - **Target:** >80%.
- **GPU Memory:** % of GPU memory used.
- **CPU Utilization:** % of CPU used.
- **Network Bandwidth:** Data transfer rate.

**Cost Metrics:**
- **Cost per Request:** $ per request.
- **Cost per 1K Tokens:** $ per 1000 tokens.
- **Monthly Spend:** Total monthly cost.

**Quality Metrics:**
- **User Satisfaction:** Thumbs up/down rate.
- **Hallucination Rate:** % of responses with hallucinations.
- **Refusal Rate:** % of requests refused.
- **Error Rate:** % of failed requests.

### 2. Prometheus Setup

**Metrics Collection:**
```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Request metrics
request_count = Counter(
    'llm_requests_total',
    'Total number of requests',
    ['model', 'status']
)

request_latency = Histogram(
    'llm_request_latency_seconds',
    'Request latency',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Token metrics
tokens_generated = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

# Resource metrics
gpu_utilization = Gauge(
    'llm_gpu_utilization_percent',
    'GPU utilization',
    ['gpu_id']
)

gpu_memory = Gauge(
    'llm_gpu_memory_used_bytes',
    'GPU memory used',
    ['gpu_id']
)

# Cost metrics
cost_total = Counter(
    'llm_cost_dollars_total',
    'Total cost in dollars',
    ['model']
)
```

### 3. Logging Best Practices

**Structured Logging:**
```python
import logging
import json

logger = logging.getLogger(__name__)

def log_request(request_id, prompt, response, latency, tokens):
    logger.info(json.dumps({
        "event": "llm_request",
        "request_id": request_id,
        "prompt_length": len(prompt),
        "response_length": len(response),
        "latency_ms": latency * 1000,
        "tokens_generated": tokens,
        "timestamp": time.time()
    }))
```

**Log Levels:**
- **DEBUG:** Detailed debugging information.
- **INFO:** General informational messages.
- **WARNING:** Warning messages (e.g., high latency).
- **ERROR:** Error messages (e.g., request failed).
- **CRITICAL:** Critical issues (e.g., service down).

### 4. Distributed Tracing

**OpenTelemetry:**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Trace request
@tracer.start_as_current_span("llm_generate")
def generate(prompt):
    with tracer.start_as_current_span("tokenize"):
        tokens = tokenize(prompt)
    
    with tracer.start_as_current_span("model_forward"):
        output = model.generate(tokens)
    
    with tracer.start_as_current_span("detokenize"):
        response = detokenize(output)
    
    return response
```

### 5. Alerting

**Alert Rules:**
```yaml
# Prometheus alert rules
groups:
  - name: llm_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, llm_request_latency_seconds) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High p95 latency"
      
      - alert: HighErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 1%"
      
      - alert: LowGPUUtilization
        expr: llm_gpu_utilization_percent < 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU utilization < 50%"
```

### 6. Dashboards

**Grafana Dashboard Panels:**
- **Latency:** p50, p95, p99 over time.
- **Throughput:** Requests/sec, tokens/sec.
- **GPU Metrics:** Utilization, memory usage.
- **Error Rate:** % of failed requests.
- **Cost:** Cumulative cost over time.
- **Quality:** User satisfaction, hallucination rate.

### 7. LLM-Specific Observability Tools

**LangSmith:**
- Trace LLM chains and agents.
- Debug prompts and outputs.
- Monitor cost and latency.

**Helicone:**
- LLM observability platform.
- Track requests, costs, latency.
- Cache management.

**Weights & Biases:**
- Experiment tracking.
- Model performance monitoring.
- Cost analysis.

**Arize AI:**
- ML observability platform.
- Drift detection.
- Performance monitoring.

### 8. Cost Tracking

**Token-Based Costing:**
```python
PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
}

def calculate_cost(model, input_tokens, output_tokens):
    input_cost = (input_tokens / 1000) * PRICING[model]["input"]
    output_cost = (output_tokens / 1000) * PRICING[model]["output"]
    return input_cost + output_cost
```

**Cost Alerts:**
- Daily spend > $1000.
- Single request > $1.
- Monthly spend > budget.

### 9. Quality Monitoring

**Automated Quality Checks:**
- **Length Check:** Response too short/long.
- **Toxicity Check:** Detect toxic content.
- **PII Detection:** Detect personal information.
- **Hallucination Detection:** Check for unsupported claims.

**User Feedback:**
- Thumbs up/down.
- Explicit ratings (1-5 stars).
- Free-form feedback.

### 10. Real-World Monitoring Stack

**Example Stack:**
- **Metrics:** Prometheus + Grafana.
- **Logs:** ELK Stack (Elasticsearch, Logstash, Kibana).
- **Traces:** Jaeger or Zipkin.
- **Alerts:** Prometheus Alertmanager + PagerDuty.
- **LLM-Specific:** LangSmith or Helicone.

### Summary

**Monitoring Checklist:**
- [ ] **Latency:** Track TTFT, TPOT, p95 latency.
- [ ] **Throughput:** Track requests/sec, tokens/sec.
- [ ] **Resources:** Track GPU utilization, memory.
- [ ] **Cost:** Track $ per request, monthly spend.
- [ ] **Quality:** Track user satisfaction, hallucination rate.
- [ ] **Alerts:** Set alerts for high latency, errors, cost.
- [ ] **Dashboards:** Create Grafana dashboards.

### Next Steps
In the Deep Dive, we will implement complete monitoring with Prometheus, Grafana, and LLM-specific observability tools.
