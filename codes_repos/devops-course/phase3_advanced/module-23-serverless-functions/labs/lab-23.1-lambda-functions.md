# Lab 23.1: Lambda Functions

## Objective
Create and deploy AWS Lambda functions.

## Learning Objectives
- Create Lambda functions
- Configure triggers
- Use environment variables
- Monitor with CloudWatch

---

## Create Function

```python
# lambda_function.py
import json

def lambda_handler(event, context):
    name = event.get('name', 'World')
    return {
        'statusCode': 200,
        'body': json.dumps(f'Hello, {name}!')
    }
```

## Deploy

```bash
# Create deployment package
zip function.zip lambda_function.py

# Create function
aws lambda create-function \
  --function-name my-function \
  --runtime python3.11 \
  --role arn:aws:iam::123456789012:role/lambda-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip
```

## Invoke

```bash
# Synchronous
aws lambda invoke \
  --function-name my-function \
  --payload '{"name":"DevOps"}' \
  response.json

cat response.json
```

## API Gateway Trigger

```bash
# Create API
aws apigatewayv2 create-api \
  --name my-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:us-east-1:123:function:my-function
```

## Success Criteria
✅ Lambda function created  
✅ Function invoked successfully  
✅ API Gateway integrated  
✅ Logs in CloudWatch  

**Time:** 40 min
