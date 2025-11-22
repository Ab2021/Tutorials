# Lab 10.2: IAM (Identity and Access Management)

## 🎯 Objective

Secure your cloud. You will create an IAM User, assign permissions using Policies, and understand the Principle of Least Privilege.

## 📋 Prerequisites

-   AWS Root Account (Lab 10.1).

## 📚 Background

### IAM Concepts
1.  **User**: A person (e.g., `alice`).
2.  **Group**: A collection of users (e.g., `Developers`).
3.  **Role**: A hat that a service (EC2) or user can put on to get permissions temporarily.
4.  **Policy**: A JSON document defining permissions (`Allow` `s3:ListBucket`).

---

## 🔨 Hands-On Implementation

### Part 1: Create an Admin User 👤

**Stop using Root!**

1.  **Go to IAM Console.**
2.  **Users** -> **Create user**.
3.  Name: `admin-user`.
4.  **Permissions**:
    -   Select "Attach policies directly".
    -   Search `AdministratorAccess`. (This is god-mode, be careful).
    -   Select it.
5.  **Create User.**
6.  **Create Access Keys:**
    -   Click User -> **Security credentials**.
    -   **Create access key** -> Select "CLI".
    -   **Save the CSV**. You will need this for Terraform/AWS CLI.

### Part 2: Create a Limited User (Least Privilege) 🛡️

1.  **Create user:** `dev-user`.
2.  **Permissions**:
    -   Do NOT attach `AdministratorAccess`.
    -   Search `AmazonEC2ReadOnlyAccess`.
    -   Select it.
3.  **Login as dev-user:**
    -   Open Incognito Window.
    -   Login URL: `https://<account-id>.signin.aws.amazon.com/console`.
4.  **Test:**
    -   Go to EC2. You can see instances.
    -   Try to **Launch Instance**.
    -   *Result:* **Access Denied**. (Success! The policy worked).

### Part 3: IAM Roles (For Services) 🎭

**Scenario:** An EC2 instance needs to upload files to S3.
**Bad Way:** Save Access Keys on the EC2 instance.
**Good Way:** Attach an IAM Role to the EC2 instance.

1.  **Roles** -> **Create role**.
2.  **Trusted entity type**: AWS Service -> EC2.
3.  **Permissions**: `AmazonS3FullAccess`.
4.  **Name**: `EC2-S3-Admin-Role`.
5.  *We will use this in the next lab.*

---

## 🎯 Challenges

### Challenge 1: Custom Policy (Difficulty: ⭐⭐⭐)

**Task:**
Create a custom JSON policy that allows a user to **Start** and **Stop** EC2 instances, but **NOT Terminate** them.
*Hint:*
```json
{
    "Effect": "Allow",
    "Action": ["ec2:StartInstances", "ec2:StopInstances"],
    "Resource": "*"
}
```

### Challenge 2: MFA Enforcement (Difficulty: ⭐⭐⭐⭐)

**Task:**
Create a policy that denies all actions unless the user has signed in with MFA.
Attach it to a test group.
*This is a best practice for Admins.*

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
JSON Policy:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:StartInstances",
                "ec2:StopInstances",
                "ec2:DescribeInstances"
            ],
            "Resource": "*"
        }
    ]
}
```

**Challenge 2:**
Condition: `"Bool": { "aws:MultiFactorAuthPresent": "true" }`.
</details>

---

## 🔑 Key Takeaways

1.  **Never Share Keys**: Each developer gets their own IAM User.
2.  **Roles for Machines**: Never put long-term credentials (access keys) on an EC2 instance. Use Roles.
3.  **Least Privilege**: Give users only what they need. Start with 0 permissions and add as needed.

---

## ⏭️ Next Steps

We are secure. Let's launch compute.

Proceed to **Lab 10.3: EC2 Fundamentals**.
