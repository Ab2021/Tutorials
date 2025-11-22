# Lab 23.1: Serverless with AWS Lambda & Terraform

## 🎯 Objective

No Servers. Just Code. You will deploy a Python function to **AWS Lambda** and expose it via **API Gateway**, entirely managed by Terraform.

## 📋 Prerequisites

-   AWS Account.
-   Terraform installed.
-   Python installed.

## 📚 Background

### Concepts
-   **FaaS (Function as a Service)**: You upload code, cloud provider runs it.
-   **Cold Start**: The delay when the function runs for the first time (container spin-up).
-   **Event-Driven**: Functions are triggered by events (HTTP Request, S3 Upload, DynamoDB Change).

---

## 🔨 Hands-On Implementation

### Part 1: The Code 🐍

1.  **Create `lambda_function.py`:**
    ```python
    import json

    def lambda_handler(event, context):
        print("Received event:", json.dumps(event))
        return {
            'statusCode': 200,
            'body': json.dumps('Hello from Terraform Serverless!')
        }
    ```

2.  **Zip it:**
    On Windows (Powershell): `Compress-Archive lambda_function.py lambda.zip`.
    On Linux/Mac: `zip lambda.zip lambda_function.py`.

### Part 2: The Infrastructure (Terraform) 🏗️

1.  **Create `main.tf`:**
    ```hcl
    provider "aws" { region = "us-east-1" }

    # IAM Role
    resource "aws_iam_role" "lambda_exec" {
      name = "serverless_lambda"
      assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [{
          Action = "sts:AssumeRole"
          Effect = "Allow"
          Principal = { Service = "lambda.amazonaws.com" }
        }]
      })
    }

    resource "aws_iam_role_policy_attachment" "lambda_logs" {
      role       = aws_iam_role.lambda_exec.name
      policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    }

    # Lambda Function
    resource "aws_lambda_function" "hello" {
      function_name = "hello_terraform"
      role          = aws_iam_role.lambda_exec.arn
      handler       = "lambda_function.lambda_handler"
      runtime       = "python3.9"
      filename      = "lambda.zip"
      source_code_hash = filebase64sha256("lambda.zip")
    }

    # API Gateway (HTTP API - Cheaper/Simpler)
    resource "aws_apigatewayv2_api" "lambda" {
      name          = "serverless_lambda_gw"
      protocol_type = "HTTP"
    }

    resource "aws_apigatewayv2_stage" "lambda" {
      api_id = aws_apigatewayv2_api.lambda.id
      name   = "$default"
      auto_deploy = true
    }

    resource "aws_apigatewayv2_integration" "hello" {
      api_id           = aws_apigatewayv2_api.lambda.id
      integration_type = "AWS_PROXY"
      integration_uri  = aws_lambda_function.hello.invoke_arn
    }

    resource "aws_apigatewayv2_route" "hello" {
      api_id    = aws_apigatewayv2_api.lambda.id
      route_key = "GET /hello"
      target    = "integrations/${aws_apigatewayv2_integration.hello.id}"
    }

    # Permission for Gateway to invoke Lambda
    resource "aws_lambda_permission" "api_gw" {
      statement_id  = "AllowExecutionFromAPIGateway"
      action        = "lambda:InvokeFunction"
      function_name = aws_lambda_function.hello.function_name
      principal     = "apigateway.amazonaws.com"
      source_arn    = "${aws_apigatewayv2_api.lambda.execution_arn}/*/*"
    }

    output "base_url" {
      value = aws_apigatewayv2_stage.lambda.invoke_url
    }
    ```

### Part 3: Deploy & Test 🚀

1.  **Apply:**
    `terraform init && terraform apply`.

2.  **Invoke:**
    Copy the `base_url` output.
    `curl <base_url>/hello`
    *Result:* "Hello from Terraform Serverless!"

---

## 🎯 Challenges

### Challenge 1: Environment Variables (Difficulty: ⭐⭐)

**Task:**
Add an environment variable `GREETING` to the Lambda resource.
Update Python code to use `os.environ['GREETING']`.
Redeploy.

### Challenge 2: S3 Trigger (Difficulty: ⭐⭐⭐)

**Task:**
Create an S3 bucket.
Configure Lambda to trigger whenever a file is uploaded (`s3:ObjectCreated:*`).
Print the filename in the logs.
*Hint:* `aws_s3_bucket_notification`.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```hcl
environment {
  variables = {
    GREETING = "Hola"
  }
}
```
</details>

---

## 🔑 Key Takeaways

1.  **Cost**: You pay per millisecond. If no one calls it, you pay $0.
2.  **Limits**: 15 minutes max execution time. 10GB max RAM. Not for long-running jobs.
3.  **IaC**: Managing Lambdas manually is a nightmare. Always use Terraform, SAM, or Serverless Framework.

---

## ⏭️ Next Steps

We did Serverless in AWS. Can we do it in Kubernetes?

Proceed to **Lab 23.2: Knative**.
