# Lab 24.5: Application Performance Monitoring

## Objective
Implement comprehensive APM with distributed tracing and metrics.

## Learning Objectives
- Set up APM tools
- Instrument applications
- Analyze performance bottlenecks
- Optimize application performance

---

## APM with New Relic

```python
import newrelic.agent
newrelic.agent.initialize('newrelic.ini')

@newrelic.agent.background_task()
def process_order(order_id):
    with newrelic.agent.FunctionTrace('fetch_order'):
        order = db.get_order(order_id)
    
    with newrelic.agent.FunctionTrace('validate_order'):
        validate(order)
    
    with newrelic.agent.FunctionTrace('process_payment'):
        payment = process_payment(order)
    
    return payment
```

## Custom Metrics

```python
# Record custom metric
newrelic.agent.record_custom_metric('Custom/OrderValue', order.total)

# Record custom event
newrelic.agent.record_custom_event('OrderProcessed', {
    'orderId': order.id,
    'amount': order.total,
    'status': 'completed'
})
```

## Datadog APM

```python
from ddtrace import tracer

@tracer.wrap(service='order-service', resource='process_order')
def process_order(order_id):
    span = tracer.current_span()
    span.set_tag('order.id', order_id)
    span.set_metric('order.amount', order.total)
    
    # Process order
    result = do_processing()
    
    return result
```

## Success Criteria
✅ APM tool configured  
✅ Application instrumented  
✅ Traces collected  
✅ Performance optimized  

**Time:** 45 min
