# Lab 14.2: Remote State & Locking

## 🎯 Objective

Stop storing `terraform.tfstate` on your laptop. If your laptop breaks, the infrastructure is orphaned. If two people run `apply` at the same time, the state gets corrupted. You will move the state to **S3** and enable locking with **DynamoDB**.

## 📋 Prerequisites

-   AWS Account.
-   Terraform installed.

## 📚 Background

### The Backend
Terraform supports "Backends". The default is `local`. The standard for AWS is `s3`.

### Locking
Prevents race conditions.
-   **Alice** runs `terraform apply`. Terraform writes a Lock ID to DynamoDB.
-   **Bob** runs `terraform apply`. Terraform sees the Lock ID and waits/fails.
-   **Alice** finishes. Terraform removes the Lock ID.

---

## 🔨 Hands-On Implementation

### Part 1: Create Backend Infrastructure (Bootstrap) 🥾

We need an S3 bucket and DynamoDB table *before* we can use them.

1.  **Create `bootstrap/main.tf`:**
    ```hcl
    provider "aws" { region = "us-east-1" }

    resource "aws_s3_bucket" "tf_state" {
      bucket = "devops-course-state-<yourname>"
      # Enable Versioning (Crucial for rollback)
      versioning { enabled = true }
    }

    resource "aws_dynamodb_table" "tf_lock" {
      name         = "terraform-locks"
      billing_mode = "PAY_PER_REQUEST"
      hash_key     = "LockID"
      attribute {
        name = "LockID"
        type = "S"
      }
    }
    ```

2.  **Apply:**
    `terraform apply`.

### Part 2: Configure Backend ⚙️

1.  **Go to your project folder (e.g., Lab 14.1 folder).**
    
2.  **Add `backend.tf`:**
    ```hcl
    terraform {
      backend "s3" {
        bucket         = "devops-course-state-<yourname>"
        key            = "prod/terraform.tfstate"
        region         = "us-east-1"
        dynamodb_table = "terraform-locks"
        encrypt        = true
      }
    }
    ```

3.  **Migrate State:**
    ```bash
    terraform init
    ```
    *Prompt:* "Do you want to copy existing state to the new backend?"
    *Answer:* `yes`.

4.  **Verify:**
    Check S3 Console. You should see the `.tfstate` file.
    Check local folder. `terraform.tfstate` is now empty/proxy.

### Part 3: Test Locking 🔒

1.  **Open Terminal A:**
    Run a long operation (or simulate one).
    Since we don't have a long op, we can cheat.
    Run `terraform apply`. **Do not type yes**. Just leave it at the prompt.

2.  **Open Terminal B:**
    Run `terraform plan`.

3.  **Result:**
    Terminal B says: `Error: Error acquiring the state lock`.
    It shows who holds the lock (Terminal A).

4.  **Release:**
    Cancel Terminal A (Ctrl+C).
    Run Terminal B again. It works.

---

## 🎯 Challenges

### Challenge 1: State Isolation (Difficulty: ⭐⭐)

**Task:**
Create a `dev` and `prod` environment.
Change the `key` in `backend.tf` to `dev/terraform.tfstate`.
Init.
Then change to `prod/terraform.tfstate`.
Init.
*Observation:* You now have two separate states in the same bucket.

### Challenge 2: Force Unlock (Difficulty: ⭐⭐⭐)

**Task:**
If Terraform crashes, the lock might stay in DynamoDB.
Learn how to force unlock.
`terraform force-unlock <LOCK_ID>`
*Warning:* Only do this if you are sure no one is running apply.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
The Lock ID is a UUID shown in the error message.
</details>

---

## 🔑 Key Takeaways

1.  **Single Source of Truth**: S3 becomes the truth.
2.  **Versioning**: If you corrupt the state, S3 Versioning allows you to download the previous version.
3.  **Teamwork**: This is mandatory for any team > 1 person.

---

## ⏭️ Next Steps

We have mastered Infrastructure. Let's master Configuration.

Proceed to **Module 15: Advanced Configuration Management**.
