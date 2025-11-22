# Lab 4.10: Networking Capstone Project

## 🎯 Objective

Design and implement a **Secure 3-Tier Network Architecture**. You will produce a network diagram and a Terraform configuration that implements a production-grade network with Public, App, and Data layers.

## 📋 Prerequisites

-   Completed Module 4.
-   Terraform installed.
-   Diagramming tool (Draw.io or Mermaid.js).

## 📚 Background

### The 3-Tier Architecture
The gold standard for web apps.
1.  **Presentation Tier (Public)**: Load Balancers, Bastion Hosts. (Direct Internet Access).
2.  **Application Tier (Private)**: Web Servers, API Nodes. (Access via LB only).
3.  **Data Tier (Private)**: Databases. (Access via App Tier only).

---

## 🔨 Hands-On Implementation

### Part 1: The Diagram 🎨

Create a file `architecture.mermaid`.

```mermaid
graph TD
    User((User)) --> IGW[Internet Gateway]
    IGW --> LB[Load Balancer (Public Subnet)]
    
    subgraph VPC
        subgraph Public_Subnet
            LB
            Bastion[Bastion Host]
        end
        
        subgraph Private_App_Subnet
            App1[Web Server 1]
            App2[Web Server 2]
        end
        
        subgraph Private_Data_Subnet
            DB[Database]
        end
    end
    
    LB --> App1
    LB --> App2
    App1 --> DB
    App2 --> DB
    Bastion -.-> App1
    Bastion -.-> DB
```

### Part 2: The Implementation (Terraform) 🏗️

Create `capstone.tf`. We will define the Security Groups (Firewalls) which are the most critical part.

1.  **Public SG (Load Balancer):**
    -   Allow HTTP (80) from `0.0.0.0/0` (Anywhere).
    -   Allow HTTPS (443) from `0.0.0.0/0`.

2.  **App SG (Web Servers):**
    -   Allow HTTP (80) **ONLY** from `Public SG`.
    -   Allow SSH (22) **ONLY** from `Bastion SG`.

3.  **Data SG (Database):**
    -   Allow SQL (5432) **ONLY** from `App SG`.
    -   Allow SSH (22) **ONLY** from `Bastion SG`.

### Part 3: The Code 💻

```hcl
# 1. Public SG
resource "aws_security_group" "public_sg" {
  name = "public_sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 2. App SG
resource "aws_security_group" "app_sg" {
  name = "app_sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.public_sg.id] # Chaining
  }
}

# 3. Data SG
resource "aws_security_group" "data_sg" {
  name = "data_sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app_sg.id] # Chaining
  }
}
```

### Part 4: Validation ✅

**Scenario Check:**
1.  **Hacker** tries to connect to Database IP on port 5432.
    -   **Result**: Blocked. (Not in App SG).
2.  **Hacker** tries to connect to App Server IP on port 80 directly.
    -   **Result**: Blocked. (Not in Public SG).
3.  **Valid User** connects to Load Balancer.
    -   **Result**: Allowed -> Forwarded to App -> Query to DB.

---

## 🎯 Challenges

### Challenge 1: Network ACLs (Difficulty: ⭐⭐⭐⭐)

**Background:** Security Groups act at the *Instance* level. Network ACLs act at the *Subnet* level.
**Task:**
Design a NACL for the Data Subnet that explicitly DENIES all traffic from the Public Subnet (except established connections).
*Why?* Defense in Depth. Even if SG fails, NACL blocks it.

### Challenge 2: The "Outbound" Problem (Difficulty: ⭐⭐⭐)

**Scenario:** Your App Servers need to download `yum update`.
**Task:**
Ensure your App SG allows outbound traffic to `0.0.0.0/0` (for the NAT Gateway).
*Note:* Terraform SGs usually deny outbound by default or allow all. Be explicit.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1 (NACL):**
```hcl
resource "aws_network_acl" "data_nacl" {
  vpc_id = aws_vpc.main.id
  subnet_ids = [aws_subnet.data.id]

  # Allow Inbound from App Subnet
  ingress {
    protocol   = "tcp"
    rule_no    = 100
    action     = "allow"
    cidr_block = "10.0.2.0/24" # App Subnet
    from_port  = 5432
    to_port    = 5432
  }
  
  # Deny everything else (Implicit)
}
```

**Challenge 2 (Outbound):**
```hcl
resource "aws_security_group" "app_sg" {
  # ... ingress rules ...

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # All protocols
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```
</details>

---

## 🔑 Key Takeaways

1.  **Least Privilege**: Only open ports to specific sources (Security Group Chaining).
2.  **3-Tier is Standard**: Memorize this pattern. It's in every interview.
3.  **Visuals Matter**: Always draw the diagram before writing the code.

---

## ⏭️ Next Steps

**Congratulations!** You have completed Module 4: Networking Fundamentals.
You understand the pipes. Now let's fill them with containers.

Proceed to **Module 5: Docker Fundamentals**.
