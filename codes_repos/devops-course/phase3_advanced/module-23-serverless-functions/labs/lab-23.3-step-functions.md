# Lab 23.3: Step Functions

## Objective
Orchestrate serverless workflows with AWS Step Functions.

## Learning Objectives
- Create state machines
- Coordinate Lambda functions
- Handle errors and retries
- Monitor executions

---

## State Machine

```json
{
  "Comment": "Order processing workflow",
  "StartAt": "ValidateOrder",
  "States": {
    "ValidateOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123:function:validate-order",
      "Next": "ProcessPayment",
      "Catch": [{
        "ErrorEquals": ["ValidationError"],
        "Next": "OrderFailed"
      }]
    },
    "ProcessPayment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123:function:process-payment",
      "Retry": [{
        "ErrorEquals": ["PaymentError"],
        "IntervalSeconds": 2,
        "MaxAttempts": 3,
        "BackoffRate": 2.0
      }],
      "Next": "FulfillOrder"
    },
    "FulfillOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123:function:fulfill-order",
      "End": true
    },
    "OrderFailed": {
      "Type": "Fail",
      "Error": "OrderProcessingFailed"
    }
  }
}
```

## Parallel Execution

```json
{
  "Type": "Parallel",
  "Branches": [
    {
      "StartAt": "SendEmail",
      "States": {
        "SendEmail": {
          "Type": "Task",
          "Resource": "arn:aws:lambda:...:function:send-email",
          "End": true
        }
      }
    },
    {
      "StartAt": "SendSMS",
      "States": {
        "SendSMS": {
          "Type": "Task",
          "Resource": "arn:aws:lambda:...:function:send-sms",
          "End": true
        }
      }
    }
  ],
  "Next": "Complete"
}
```

## Success Criteria
✅ State machine created  
✅ Workflow executing  
✅ Error handling working  
✅ Executions monitored  

**Time:** 45 min
