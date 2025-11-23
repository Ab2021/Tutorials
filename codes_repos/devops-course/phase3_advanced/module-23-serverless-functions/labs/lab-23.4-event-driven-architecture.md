# Lab 23.4: Event-Driven Architecture

## Objective
Build event-driven serverless applications.

## Learning Objectives
- Use EventBridge
- Implement event patterns
- Create event-driven workflows
- Monitor events

---

## EventBridge Rule

```bash
# Create event bus
aws events create-event-bus --name custom-bus

# Create rule
aws events put-rule \
  --name order-created \
  --event-bus-name custom-bus \
  --event-pattern '{
    "source": ["myapp.orders"],
    "detail-type": ["Order Created"]
  }'

# Add Lambda target
aws events put-targets \
  --rule order-created \
  --event-bus-name custom-bus \
  --targets "Id"="1","Arn"="arn:aws:lambda:...:function:process-order"
```

## Publish Events

```python
import boto3
import json

events = boto3.client('events')

def publish_order_created(order_id):
    response = events.put_events(
        Entries=[
            {
                'Source': 'myapp.orders',
                'DetailType': 'Order Created',
                'Detail': json.dumps({
                    'orderId': order_id,
                    'timestamp': '2024-01-01T12:00:00Z'
                }),
                'EventBusName': 'custom-bus'
            }
        ]
    )
    return response
```

## Event Pattern Matching

```json
{
  "source": ["myapp.orders"],
  "detail-type": ["Order Created", "Order Updated"],
  "detail": {
    "status": ["pending", "processing"],
    "amount": [{"numeric": [">", 100]}]
  }
}
```

## Success Criteria
✅ EventBridge configured  
✅ Events published  
✅ Patterns matching  
✅ Workflows triggered  

**Time:** 40 min
