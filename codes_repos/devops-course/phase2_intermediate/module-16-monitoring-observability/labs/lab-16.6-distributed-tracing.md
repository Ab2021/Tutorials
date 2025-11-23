# Lab 16.6: Distributed Tracing

## Objective
Implement distributed tracing with Jaeger.

## Learning Objectives
- Set up Jaeger
- Instrument applications
- Analyze traces
- Optimize performance

---

## Jaeger Setup

```yaml
# docker-compose.yml
version: '3'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"
      - "14268:14268"
```

## Instrument Python App

```python
from jaeger_client import Config
from opentracing.ext import tags
from opentracing.propagation import Format

def init_tracer(service_name):
    config = Config(
        config={
            'sampler': {'type': 'const', 'param': 1},
            'logging': True,
        },
        service_name=service_name,
    )
    return config.initialize_tracer()

tracer = init_tracer('my-service')

@app.route('/api/users')
def get_users():
    with tracer.start_active_span('get-users') as scope:
        scope.span.set_tag(tags.HTTP_METHOD, 'GET')
        scope.span.set_tag(tags.HTTP_URL, '/api/users')
        
        users = fetch_users()
        
        scope.span.set_tag(tags.HTTP_STATUS_CODE, 200)
        return jsonify(users)
```

## Success Criteria
✅ Jaeger running  
✅ Traces collected  
✅ Performance analyzed  
✅ Bottlenecks identified  

**Time:** 45 min
