# Day 46: Monitoring & Observability
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Complete Prometheus + Grafana Setup

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from fastapi import FastAPI
import time
import torch

app = FastAPI()

# Define metrics
request_count = Counter('llm_requests_total', 'Total requests', ['model', 'status'])
request_latency = Histogram('llm_request_latency_seconds', 'Request latency', ['model'])
ttft = Histogram('llm_ttft_seconds', 'Time to first token', ['model'])
tpot = Histogram('llm_tpot_seconds', 'Time per output token', ['model'])
tokens_generated = Counter('llm_tokens_generated_total', 'Total tokens', ['model'])
active_requests = Gauge('llm_active_requests', 'Active requests')
gpu_utilization = Gauge('llm_gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
gpu_memory_used = Gauge('llm_gpu_memory_used_bytes', 'GPU memory used', ['gpu_id'])
cost_total = Counter('llm_cost_dollars_total', 'Total cost', ['model'])

# Background task to update GPU metrics
import threading

def update_gpu_metrics():
    while True:
        for i in range(torch.cuda.device_count()):
            util = torch.cuda.utilization(i)
            memory = torch.cuda.memory_allocated(i)
            
            gpu_utilization.labels(gpu_id=str(i)).set(util)
            gpu_memory_used.labels(gpu_id=str(i)).set(memory)
        
        time.sleep(5)

threading.Thread(target=update_gpu_metrics, daemon=True).start()

# Start Prometheus metrics server
start_http_server(9090)

@app.post("/generate")
async def generate(request: GenerateRequest):
    model_name = "llama-2-7b"
    active_requests.inc()
    
    start_time = time.time()
    first_token_time = None
    tokens = 0
    
    try:
        # Generate with streaming
        for token in model.generate_stream(request.prompt):
            if first_token_time is None:
                first_token_time = time.time() - start_time
                ttft.labels(model=model_name).observe(first_token_time)
            
            tokens += 1
            yield token
        
        # Record metrics
        total_time = time.time() - start_time
        request_latency.labels(model=model_name).observe(total_time)
        tpot.labels(model=model_name).observe((total_time - first_token_time) / tokens)
        tokens_generated.labels(model=model_name).inc(tokens)
        
        # Calculate cost
        cost = calculate_cost(model_name, len(request.prompt.split()), tokens)
        cost_total.labels(model=model_name).inc(cost)
        
        request_count.labels(model=model_name, status="success").inc()
    
    except Exception as e:
        request_count.labels(model=model_name, status="error").inc()
        raise
    
    finally:
        active_requests.dec()
```

### 2. Structured Logging with Context

```python
import logging
import json
from contextvars import ContextVar

# Context variable for request ID
request_id_var = ContextVar('request_id', default=None)

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log(self, level, event, **kwargs):
        """Log structured event."""
        log_data = {
            "timestamp": time.time(),
            "level": level,
            "event": event,
            "request_id": request_id_var.get(),
            **kwargs
        }
        
        self.logger.log(
            getattr(logging, level.upper()),
            json.dumps(log_data)
        )
    
    def info(self, event, **kwargs):
        self.log("info", event, **kwargs)
    
    def error(self, event, **kwargs):
        self.log("error", event, **kwargs)
    
    def warning(self, event, **kwargs):
        self.log("warning", event, **kwargs)

logger = StructuredLogger(__name__)

@app.post("/generate")
async def generate(request: GenerateRequest):
    # Set request ID in context
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    
    logger.info("request_started",
        prompt_length=len(request.prompt),
        max_tokens=request.max_tokens
    )
    
    try:
        output = model.generate(request.prompt)
        
        logger.info("request_completed",
            tokens_generated=len(output.split()),
            latency_ms=(time.time() - start_time) * 1000
        )
        
        return output
    
    except Exception as e:
        logger.error("request_failed",
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

### 3. Distributed Tracing with OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to Jaeger/Zipkin
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# Auto-instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

@app.post("/generate")
async def generate(request: GenerateRequest):
    with tracer.start_as_current_span("llm_generate") as span:
        # Add attributes
        span.set_attribute("model", "llama-2-7b")
        span.set_attribute("prompt_length", len(request.prompt))
        
        # Tokenization span
        with tracer.start_as_current_span("tokenize"):
            tokens = tokenizer.encode(request.prompt)
            span.set_attribute("num_tokens", len(tokens))
        
        # Model forward span
        with tracer.start_as_current_span("model_forward"):
            output_tokens = model.generate(tokens)
        
        # Detokenization span
        with tracer.start_as_current_span("detokenize"):
            output = tokenizer.decode(output_tokens)
            span.set_attribute("output_length", len(output))
        
        return output
```

### 4. Custom Quality Metrics

```python
class QualityMonitor:
    def __init__(self):
        self.hallucination_rate = Gauge('llm_hallucination_rate', 'Hallucination rate')
        self.user_satisfaction = Gauge('llm_user_satisfaction', 'User satisfaction')
        self.refusal_rate = Gauge('llm_refusal_rate', 'Refusal rate')
        
        # Sliding window for rates
        self.window_size = 1000
        self.recent_requests = []
    
    def record_request(self, has_hallucination, user_feedback, was_refused):
        """Record quality metrics for a request."""
        self.recent_requests.append({
            "hallucination": has_hallucination,
            "feedback": user_feedback,  # 1 (thumbs up) or -1 (thumbs down)
            "refused": was_refused
        })
        
        # Keep only recent requests
        if len(self.recent_requests) > self.window_size:
            self.recent_requests.pop(0)
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Calculate and update quality metrics."""
        if not self.recent_requests:
            return
        
        # Hallucination rate
        hallucinations = sum(1 for r in self.recent_requests if r["hallucination"])
        self.hallucination_rate.set(hallucinations / len(self.recent_requests))
        
        # User satisfaction
        feedbacks = [r["feedback"] for r in self.recent_requests if r["feedback"] is not None]
        if feedbacks:
            satisfaction = sum(1 for f in feedbacks if f > 0) / len(feedbacks)
            self.user_satisfaction.set(satisfaction)
        
        # Refusal rate
        refusals = sum(1 for r in self.recent_requests if r["refused"])
        self.refusal_rate.set(refusals / len(self.recent_requests))

quality_monitor = QualityMonitor()

@app.post("/feedback")
async def feedback(request_id: str, thumbs_up: bool):
    """Record user feedback."""
    quality_monitor.record_request(
        has_hallucination=False,  # Would be detected separately
        user_feedback=1 if thumbs_up else -1,
        was_refused=False
    )
```

### 5. Cost Tracking Dashboard

```python
class CostTracker:
    def __init__(self):
        self.cost_by_user = Counter('llm_cost_by_user_dollars', 'Cost by user', ['user_id'])
        self.cost_by_model = Counter('llm_cost_by_model_dollars', 'Cost by model', ['model'])
        self.daily_cost = Gauge('llm_daily_cost_dollars', 'Daily cost')
        
        self.daily_total = 0
        self.last_reset = time.time()
    
    def record_cost(self, user_id, model, cost):
        """Record cost for a request."""
        self.cost_by_user.labels(user_id=user_id).inc(cost)
        self.cost_by_model.labels(model=model).inc(cost)
        
        self.daily_total += cost
        self.daily_cost.set(self.daily_total)
        
        # Reset daily total at midnight
        if time.time() - self.last_reset > 86400:
            self.daily_total = 0
            self.last_reset = time.time()
```

### 6. Alerting Rules (Prometheus)

```yaml
# prometheus_alerts.yml
groups:
  - name: llm_alerts
    rules:
      # Latency alerts
      - alert: HighP95Latency
        expr: histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency > 2s"
          description: "P95 latency is {{ $value }}s"
      
      # Error rate alerts
      - alert: HighErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 1%"
      
      # GPU alerts
      - alert: LowGPUUtilization
        expr: llm_gpu_utilization_percent < 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU utilization < 50%"
      
      # Cost alerts
      - alert: HighDailyCost
        expr: llm_daily_cost_dollars > 1000
        labels:
          severity: warning
        annotations:
          summary: "Daily cost > $1000"
      
      # Quality alerts
      - alert: HighHallucinationRate
        expr: llm_hallucination_rate > 0.05
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Hallucination rate > 5%"
```
