# Lab 10.1: Introduction to Cloud Computing & AWS Free Tier

## 🎯 Objective

Understand the Cloud. You will create an AWS Account (if you haven't), set up Billing Alerts to prevent surprise costs, and explore the Global Infrastructure.

## 📋 Prerequisites

-   Credit Card (Required by AWS for identity verification, even for Free Tier).

## 📚 Background

### What is Cloud?
"Someone else's computer."
-   **IaaS (Infrastructure as a Service)**: EC2 (VMs). You manage OS.
-   **PaaS (Platform as a Service)**: RDS (Database). AWS manages OS.
-   **SaaS (Software as a Service)**: Gmail. You manage nothing.

### The Free Tier
AWS offers 12 months free for new accounts:
-   **EC2**: 750 hours/month of `t2.micro` or `t3.micro`.
-   **S3**: 5GB storage.
-   **RDS**: 750 hours/month.

---

## 🔨 Hands-On Implementation

### Part 1: Account Setup ☁️

1.  **Sign Up:**
    Go to `aws.amazon.com`. Create a Root Account.
    *Note:* The Root Account is powerful. **Secure it immediately.**

2.  **MFA (Multi-Factor Authentication):**
    -   Go to **IAM** (Identity and Access Management).
    -   Click **Add MFA** for Root User.
    -   Use Google Authenticator or Authy app.
    -   **CRITICAL STEP**: Do not skip this.

### Part 2: Billing Alarms (Save Money!) 💰

1.  **Go to Billing Dashboard:**
    Search "Billing".

2.  **Enable Billing Alerts:**
    -   Go to **Billing Preferences**.
    -   Check "Receive Billing Alerts".

3.  **Create Alarm (CloudWatch):**
    -   Search "CloudWatch".
    -   **Alarms** -> **Create Alarm**.
    -   Metric: **Billing** -> **Total Estimated Charge**.
    -   Condition: Greater than **$1.00**.
    -   Notification: Send email to `you@example.com`.
    -   *Result:* If you accidentally leave a huge server running, you get an email when cost hits $1.

### Part 3: Global Infrastructure 🌍

1.  **Regions:**
    Click the dropdown in top right (e.g., `N. Virginia`).
    These are physical locations. `us-east-1` is the biggest.

2.  **Availability Zones (AZs):**
    Inside a Region (e.g., `us-east-1`), there are AZs (`us-east-1a`, `us-east-1b`).
    These are separate data centers.
    *Rule:* Always deploy to at least 2 AZs for High Availability.

---

## 🎯 Challenges

### Challenge 1: Cost Explorer (Difficulty: ⭐)

**Task:**
Find the **Cost Explorer** service.
Enable it (takes 24h to show data).
This is where you see exactly what is costing money.

### Challenge 2: Support Plans (Difficulty: ⭐)

**Task:**
Go to **Support Center**.
Verify you are on the **Basic Plan** (Free).
Look at the differences between Developer and Business plans.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Billing Dashboard -> Cost Explorer -> Launch.

**Challenge 2:**
Support -> Support Center.
</details>

---

## 🔑 Key Takeaways

1.  **Root User**: Only use it to create the first IAM Admin user. Then lock it away.
2.  **Region Matters**: Latency, Cost, and Services vary by region.
3.  **Free Tier Limits**: 750 hours = 1 instance running 24/7. If you run 2 instances, you use 1500 hours (750 free + 750 paid).

---

## ⏭️ Next Steps

We have an account. Now let's create a user.

Proceed to **Lab 10.2: IAM (Identity and Access Management)**.
