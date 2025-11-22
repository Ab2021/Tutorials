# Lab 10.5: RDS (Relational Database Service)

## 🎯 Objective

Launch a managed database. Instead of installing MySQL on EC2 (and managing backups/patches yourself), you will use RDS to get a production-ready database in minutes.

## 📋 Prerequisites

-   AWS Account.
-   EC2 Instance (from Lab 10.3) to connect from.

## 📚 Background

### Managed vs Unmanaged
-   **Unmanaged (EC2)**: You install MySQL. You patch OS. You setup backups. You fix replication.
-   **Managed (RDS)**: AWS does all that. You just get an endpoint.

### Engines
MySQL, PostgreSQL, MariaDB, Oracle, SQL Server, Aurora (AWS Native).

---

## 🔨 Hands-On Implementation

### Part 1: Create Database 🗄️

1.  **Go to RDS Console** -> **Create database**.
2.  **Method**: Standard create.
3.  **Engine**: MySQL.
4.  **Template**: **Free Tier**. (Crucial!).
5.  **Settings**:
    -   Identifier: `devops-db`.
    -   Master username: `admin`.
    -   Master password: `password123` (Make it complex).
6.  **Instance Config**: `db.t3.micro`.
7.  **Connectivity**:
    -   **Don't connect to an EC2 compute resource** (We will do it manually).
    -   **Public access**: **No** (Security Best Practice).
    -   **VPC Security Group**: Create new `rds-sg`.
8.  **Create database**. (Takes 5-10 mins).

### Part 2: Security Group Peering 🛡️

Your EC2 instance needs to talk to RDS.

1.  **Go to EC2 Console** -> **Security Groups**.
2.  Find `rds-sg` (created by RDS).
3.  **Edit Inbound rules**.
4.  **Add Rule**:
    -   Type: MySQL/Aurora (3306).
    -   Source: **Custom** -> Select the **Security Group ID** of your EC2 instance (e.g., `sg-12345...`).
    -   *Why?* This allows traffic ONLY from your Web Server, not the internet.

### Part 3: Connect 🔌

1.  **SSH into EC2**:
    ```bash
    ssh ubuntu@<EC2-IP>
    ```

2.  **Install Client**:
    ```bash
    sudo apt update
    sudo apt install -y mysql-client
    ```

3.  **Connect**:
    Get Endpoint from RDS Console (e.g., `devops-db.abc.us-east-1.rds.amazonaws.com`).
    ```bash
    mysql -h <RDS-ENDPOINT> -u admin -p
    ```
    Enter password.
    *Result:* `mysql>` prompt.

---

## 🎯 Challenges

### Challenge 1: Multi-AZ (Difficulty: ⭐⭐)

**Task:**
Read about Multi-AZ deployment.
*Concept:* AWS runs a standby DB in a different Availability Zone. If the primary fails, it auto-switches.
*Note:* Free Tier does NOT support Multi-AZ. Do not enable it unless you want to pay.

### Challenge 2: Snapshots (Difficulty: ⭐)

**Task:**
Take a manual snapshot of your DB.
Delete the DB.
Restore from Snapshot.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
RDS Console -> Databases -> Actions -> Take snapshot.
Wait for completion.
Actions -> Delete.
Snapshots -> Actions -> Restore snapshot.
</details>

---

## 🔑 Key Takeaways

1.  **Never Public**: Databases should almost never have a Public IP. Use a Bastion Host or VPN to access them for admin tasks.
2.  **Security Groups**: The pattern "Allow Port 3306 from Web-SG" is the standard way to secure 3-tier apps.
3.  **Maintenance Window**: RDS will auto-patch your DB during a specific window.

---

## ⏭️ Next Steps

We have all the components. Let's build the final project.

Proceed to **Lab 10.6: Cloud Capstone Project**.
