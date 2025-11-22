# Lab 10: Distributed Tracing with OpenTelemetry

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
- Implement distributed tracing
- Propagate trace context through Kafka
- Visualize traces in Jaeger

## Problem Statement
Instrument a Kafka producer and Flink consumer with OpenTelemetry. Trace a message from production through Kafka to Flink processing, visualizing the end-to-end latency in Jaeger.

## Starter Code
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# TODO: Configure tracer and inject context into Kafka headers
```

## Hints
<details>
<summary>Hint 1</summary>
Use Kafka message headers to propagate `traceparent`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**Producer with Tracing:**
```python
from confluent_kafka import Producer
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.propagate import inject

# Setup tracer
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name='localhost',
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

def produce_with_trace():
    producer = Producer({'bootstrap.servers': 'localhost:9092'})
    
    with tracer.start_as_current_span(\"kafka-produce\") as span:
        headers = {}
        inject(headers)  # Inject trace context
        
        producer.produce(
            'traced-topic',
            value='Hello',
            headers=list(headers.items())
        )
        producer.flush()
```

**Consumer with Tracing:**
```python
from confluent_kafka import Consumer
from opentelemetry.propagate import extract

def consume_with_trace():
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'trace-group'
    })
    consumer.subscribe(['traced-topic'])
    
    while True:
        msg = consumer.poll(1.0)
        if msg:
            # Extract trace context
            headers = dict(msg.headers() or [])
            ctx = extract(headers)
            
            with tracer.start_as_current_span(\"kafka-consume\", context=ctx):
                print(f\"Processed: {msg.value()}\")
```

**Jaeger Setup:**
```bash
docker run -d -p 6831:6831/udp -p 16686:16686 jaegertracing/all-in-one:latest
```

**Verify:**
Open `http://localhost:16686` and search for traces. You should see:
- Producer span
- Kafka span (implicit)
- Consumer span
- End-to-end latency
</details>
