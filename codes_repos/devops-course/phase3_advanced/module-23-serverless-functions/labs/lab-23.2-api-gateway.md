# Lab 23.2: API Gateway

## Objective
Create REST APIs with API Gateway and Lambda.

## Learning Objectives
- Create REST API
- Configure routes and methods
- Implement authentication
- Enable CORS

---

## Create API

```bash
# Create REST API
aws apigateway create-rest-api \
  --name my-api \
  --description "My REST API"

# Get root resource
API_ID=<api-id>
ROOT_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[0].id' --output text)

# Create resource
aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ROOT_ID \
  --path-part users

# Create method
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id <resource-id> \
  --http-method GET \
  --authorization-type NONE
```

## Lambda Integration

```bash
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id <resource-id> \
  --http-method GET \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123:function:my-function/invocations
```

## Deploy API

```bash
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name prod
```

## Success Criteria
✅ REST API created  
✅ Lambda integrated  
✅ API deployed  
✅ Endpoints accessible  

**Time:** 40 min
