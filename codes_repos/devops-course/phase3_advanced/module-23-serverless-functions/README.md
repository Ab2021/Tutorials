# Serverless & Functions as a Service

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Serverless computing, including:
- **Concepts**: Understanding FaaS (Function as a Service) and when to use it.
- **AWS Lambda**: Writing, deploying, and triggering functions.
- **Integration**: Connecting Lambda to API Gateway, S3, DynamoDB, and SQS.
- **Optimization**: Reducing cold starts and managing costs.
- **Tooling**: Using the Serverless Framework and AWS SAM for IaC.

---

## 📖 Theoretical Concepts

### 1. What is Serverless?

"Serverless" doesn't mean no servers. It means you don't manage them.
- **No Infrastructure**: AWS provisions, patches, and scales the servers.
- **Pay-per-Execution**: Billed by the millisecond. If your function doesn't run, you pay $0.
- **Auto-Scaling**: From 0 to 10,000 concurrent executions automatically.

### 2. AWS Lambda

- **Handler**: The entry point function (e.g., `lambda_handler(event, context)`).
- **Event**: The input data (JSON). Could be an HTTP request, S3 upload, or SQS message.
- **Context**: Metadata (request ID, remaining time).
- **Execution Environment**: A micro-VM that runs your code. Reused for subsequent invocations (warm start).

### 3. Triggers & Integrations

- **API Gateway**: HTTP endpoint -> Lambda.
- **S3**: File upload -> Lambda (e.g., Resize image).
- **DynamoDB Streams**: DB change -> Lambda (e.g., Send email when user signs up).
- **EventBridge**: CRON schedule -> Lambda.

### 4. Cold Starts

The first invocation (or after idle period) is slow because AWS must:
1.  Download your code.
2.  Start the execution environment.
3.  Initialize your runtime (import libraries).

**Mitigation**:
- **Provisioned Concurrency**: Keep N instances warm (costs more).
- **Smaller Packages**: Fewer dependencies = faster init.
- **Compiled Languages**: Go/Rust start faster than Python.

---

## 🔧 Practical Examples

### Basic Lambda (Python)

```python
import json

def lambda_handler(event, context):
    name = event.get('name', 'World')
    return {
        'statusCode': 200,
        'body': json.dumps(f'Hello, {name}!')
    }
```

### Serverless Framework (`serverless.yml`)

```yaml
service: my-service

provider:
  name: aws
  runtime: python3.9
  region: us-east-1

functions:
  hello:
    handler: handler.lambda_handler
    events:
      - http:
          path: hello
          method: get
```

### Deploy

```bash
serverless deploy
```

### Invoke

```bash
curl https://abc123.execute-api.us-east-1.amazonaws.com/dev/hello
```

---

## 🎯 Hands-on Labs

- [Lab 23.1: Serverless with AWS Lambda & Terraform](./labs/lab-23.1-aws-lambda.md)
- [Lab 23.2: Knative Serving (Kubernetes Serverless)](./labs/lab-23.2-knative.md)
- [Lab 23.3: Event Driven Architecture](./labs/lab-23.3-event-driven-architecture.md)
- [Lab 23.4: Step Functions](./labs/lab-23.4-step-functions.md)
- [Lab 23.5: Serverless Framework](./labs/lab-23.5-serverless-framework.md)
- [Lab 23.6: Sam Templates](./labs/lab-23.6-sam-templates.md)
- [Lab 23.7: Lambda Layers](./labs/lab-23.7-lambda-layers.md)
- [Lab 23.8: Cold Start Optimization](./labs/lab-23.8-cold-start-optimization.md)
- [Lab 23.9: Serverless Monitoring](./labs/lab-23.9-serverless-monitoring.md)
- [Lab 23.10: Cost Optimization](./labs/lab-23.10-cost-optimization.md)

---

## 📚 Additional Resources

### Official Documentation
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Serverless Framework](https://www.serverless.com/framework/docs)

### Tools
- [AWS SAM](https://aws.amazon.com/serverless/sam/)
- [LocalStack](https://localstack.cloud/) - Run AWS services locally.

---

## 🔑 Key Takeaways

1.  **Stateless**: Lambda functions should not store state on disk. Use DynamoDB/S3.
2.  **Timeouts**: Max execution time is 15 minutes. For longer jobs, use ECS/Batch.
3.  **Vendor Lock-in**: Lambda is AWS-specific. For portability, consider Knative on K8s.
4.  **Cost**: Serverless is cheap at low scale, but can be expensive at high scale (millions of requests/day).

---

## ⏭️ Next Steps

1.  Complete the labs to build event-driven applications.
2.  Proceed to **[Module 24: Advanced Monitoring](../module-24-advanced-monitoring/README.md)** to observe your serverless functions.
