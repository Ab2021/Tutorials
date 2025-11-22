# Lab 10.4: S3 Storage (Simple Storage Service)

## 🎯 Objective

Store data in the cloud. You will create a Bucket, upload objects, enable versioning, and host a static website.

## 📋 Prerequisites

-   AWS Account.

## 📚 Background

### S3 Concepts
-   **Bucket**: A container for objects. Must have a **Globally Unique Name** (e.g., `my-bucket-123` might be taken).
-   **Object**: A file (image, video, text).
-   **Storage Classes**:
    -   **Standard**: Frequent access.
    -   **Glacier**: Archival (cheap, slow retrieval).

---

## 🔨 Hands-On Implementation

### Part 1: Create Bucket 🪣

1.  **Go to S3 Console** -> **Create bucket**.
2.  **Name**: `devops-lab-bucket-<yourname>-<randomnumber>`.
3.  **Region**: Same as your EC2 (e.g., `us-east-1`).
4.  **Block Public Access**: Keep checked (Secure by default).
5.  **Create**.

### Part 2: Upload & Versioning 📤

1.  **Enable Versioning**:
    -   Click Bucket -> **Properties**.
    -   Bucket Versioning -> **Enable**.

2.  **Upload File**:
    -   Create `hello.txt` ("Version 1"). Upload it.

3.  **Update File**:
    -   Edit `hello.txt` ("Version 2"). Upload it again.

4.  **Verify**:
    -   Click `hello.txt`.
    -   Click **Versions** tab.
    -   You see both versions. You can restore the old one.

### Part 3: Static Website Hosting 🌐

1.  **Upload HTML**:
    -   Upload `index.html` ("Hello S3").

2.  **Disable Block Public Access**:
    -   **Permissions** tab -> **Block public access** -> **Edit** -> Uncheck all -> Save -> Confirm "confirm".

3.  **Bucket Policy (Allow Public Read)**:
    -   **Permissions** tab -> **Bucket policy** -> **Edit**.
    -   Paste:
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
            }
        ]
    }
    ```
    *(Replace YOUR-BUCKET-NAME)*.

4.  **Enable Hosting**:
    -   **Properties** tab -> Scroll down to **Static website hosting**.
    -   **Enable**.
    -   Index document: `index.html`.
    -   Save.

5.  **Visit URL**:
    -   Scroll down to find "Bucket website endpoint".
    -   Click it. You have a website!

---

## 🎯 Challenges

### Challenge 1: Lifecycle Policy (Difficulty: ⭐⭐⭐)

**Task:**
Create a Lifecycle Rule that automatically moves objects to **Glacier** after 30 days to save money.
*Hint: Management tab.*

### Challenge 2: Presigned URL (Difficulty: ⭐⭐)

**Task:**
Upload a private file (remove public policy).
Generate a "Presigned URL" using AWS CLI that allows access for only 5 minutes.
`aws s3 presign s3://mybucket/myfile.txt --expires-in 300`

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Management -> Lifecycle rules -> Create rule -> Move to Glacier Deep Archive after 30 days.

**Challenge 2:**
(Requires AWS CLI installed).
Output is a long URL. Paste in browser. It works. Wait 5 mins. It fails.
</details>

---

## 🔑 Key Takeaways

1.  **Global Namespace**: Bucket names are shared across all AWS users.
2.  **Security**: S3 Leaks are a common security failure. Always keep "Block Public Access" on unless you intentionally want a public website.
3.  **Durability**: S3 provides "11 9s" of durability (99.999999999%). You won't lose data.

---

## ⏭️ Next Steps

We have Compute and Storage. Let's connect them with Databases.

Proceed to **Lab 10.5: RDS & Databases**.
