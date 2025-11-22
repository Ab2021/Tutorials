# Lab 10.6: Cloud Capstone Project

## 🎯 Objective

Deploy a **Highly Available Web Application** on AWS. You will combine EC2, S3, and RDS to build a robust architecture manually (ClickOps), setting the stage for automating it with Terraform in Phase 2.

## 📋 Prerequisites

-   Completed Module 10.
-   AWS Account.

## 📚 Background

### The Architecture
1.  **Web Server (EC2)**: Runs Apache/PHP.
2.  **Database (RDS)**: Stores user data.
3.  **Media Storage (S3)**: Stores images.
4.  **Security**: Strict Security Groups.

---

## 🔨 Hands-On Implementation

### Step 1: The Database (RDS) 🗄️

1.  Reuse the RDS instance from Lab 10.5 (`devops-db`).
2.  Ensure the Security Group allows traffic from your EC2 instance.

### Step 2: The Storage (S3) 🪣

1.  Reuse the S3 bucket from Lab 10.4.
2.  Upload an image `logo.png`.
3.  Make it public (or use a Presigned URL).

### Step 3: The Web Server (EC2) 🖥️

1.  **Launch Instance**: Ubuntu 22.04.
2.  **User Data**:
    ```bash
    #!/bin/bash
    apt update
    apt install -y apache2 php libapache2-mod-php php-mysql
    systemctl start apache2
    systemctl enable apache2
    
    # Create Config
    cat <<EOF > /var/www/html/config.php
    <?php
    \$servername = "YOUR-RDS-ENDPOINT";
    \$username = "admin";
    \$password = "password123";
    \$dbname = "devops";
    ?>
    EOF
    
    # Create App
    cat <<EOF > /var/www/html/index.php
    <?php
    include 'config.php';
    \$conn = new mysqli(\$servername, \$username, \$password);
    if (\$conn->connect_error) {
        die("Connection failed: " . \$conn->connect_error);
    }
    echo "<h1>Connected to MySQL successfully!</h1>";
    echo "<img src='https://YOUR-BUCKET.s3.amazonaws.com/logo.png'>";
    ?>
    EOF
    
    rm /var/www/html/index.html
    ```
    *Replace YOUR-RDS-ENDPOINT and YOUR-BUCKET.*

### Step 4: Verification ✅

1.  **Visit EC2 Public IP**.
2.  You should see:
    -   "Connected to MySQL successfully!" (Proves EC2 -> RDS).
    -   The Logo image (Proves S3 access).

### Step 5: Cleanup (CRITICAL) 🧹

**Don't get a bill!**
1.  **Terminate EC2**.
2.  **Delete RDS** (Skip final snapshot).
3.  **Empty and Delete S3 Bucket**.
4.  **Release Elastic IPs** (if any).

---

## 🎯 Challenges

### Challenge 1: Domain Name (Difficulty: ⭐⭐⭐)

**Task:**
Use **Route 53** to buy a cheap domain ($3) or use a free one.
Point the domain to your EC2 IP using an **A Record**.

### Challenge 2: Load Balancer (Difficulty: ⭐⭐⭐⭐)

**Task:**
Launch a second EC2 instance.
Create an **Application Load Balancer (ALB)**.
Put both instances behind the ALB.
Access the site via the ALB DNS Name.
*Benefit:* If one server dies, the site stays up.

---

## 🔑 Key Takeaways

1.  **Integration**: The cloud is about connecting services (Compute + Storage + Database).
2.  **State**: The Web Server is "Stateless" (code only). The Database and S3 hold the "State" (data). This allows the Web Server to scale easily.
3.  **Manual is Hard**: Doing this by hand took 30 minutes. With Terraform (Phase 2), it takes 2 minutes.

---

## ⏭️ Next Steps

**Congratulations!** You have completed Phase 1 of the DevOps Course.
You have built a strong foundation.

**Phase 2 (Intermediate)** will take you from "Junior" to "Mid-Level":
-   **Advanced Docker**: Multi-stage, Distroless.
-   **Kubernetes**: Orchestration at scale.
-   **Advanced CI/CD**: GitOps, ArgoCD.
-   **Terraform**: Modules, State Management.

Proceed to **Phase 2: Module 11 - Advanced Docker**.
