# Lab 23.5: Serverless Framework

## Objective
Use Serverless Framework for deployment automation.

## Learning Objectives
- Install Serverless Framework
- Define serverless.yml
- Deploy functions
- Manage stages

---

## Install

```bash
npm install -g serverless
serverless --version
```

## Create Service

```bash
serverless create --template aws-python3 --path my-service
cd my-service
```

## serverless.yml

```yaml
service: my-service

provider:
  name: aws
  runtime: python3.11
  region: us-east-1

functions:
  hello:
    handler: handler.hello
    events:
      - http:
          path: hello
          method: get
  
  users:
    handler: handler.users
    events:
      - http:
          path: users
          method: post
```

## Deploy

```bash
# Deploy to dev
serverless deploy --stage dev

# Deploy to prod
serverless deploy --stage prod

# Invoke function
serverless invoke -f hello --stage dev

# View logs
serverless logs -f hello --stage dev
```

## Success Criteria
✅ Serverless Framework installed  
✅ Service deployed  
✅ Functions working  
✅ Multiple stages managed  

**Time:** 35 min
