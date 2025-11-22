# Lab 4.9: Cloud Networking Concepts (VPC)

## 🎯 Objective

Understand Virtual Private Clouds (VPC). Since we might not have an AWS account yet, we will simulate the architecture using Terraform code to understand how the pieces fit together.

## 📋 Prerequisites

-   Terraform installed (Lab 1.4).
-   VS Code.

## 📚 Background

### The Cloud Network Stack
1.  **VPC**: Your private slice of the cloud network (e.g., `10.0.0.0/16`).
2.  **Subnet**: A partition of the VPC (e.g., `10.0.1.0/24`).
    -   **Public**: Has a route to the Internet Gateway.
    -   **Private**: No direct route to Internet.
3.  **Internet Gateway (IGW)**: The door to the internet.
4.  **NAT Gateway**: Allows private instances to talk out (e.g., for updates) but prevents internet from talking in.
5.  **Route Table**: The GPS. Tells traffic where to go.

---

## 🔨 Hands-On Implementation

### Part 1: Defining the VPC (Terraform) 🏗️

We will write the Infrastructure as Code to visualize the setup.

1.  **Create `main.tf`:**
    ```hcl
    provider "aws" {
      region = "us-east-1"
    }

    # 1. The VPC
    resource "aws_vpc" "main" {
      cidr_block = "10.0.0.0/16"
      tags = { Name = "DevOps-Lab-VPC" }
    }

    # 2. The Internet Gateway
    resource "aws_internet_gateway" "gw" {
      vpc_id = aws_vpc.main.id
    }
    ```

### Part 2: Public vs Private Subnets 🏘️

1.  **Add Subnets to `main.tf`:**
    ```hcl
    # Public Subnet (Web Server)
    resource "aws_subnet" "public" {
      vpc_id     = aws_vpc.main.id
      cidr_block = "10.0.1.0/24"
      map_public_ip_on_launch = true
    }

    # Private Subnet (Database)
    resource "aws_subnet" "private" {
      vpc_id     = aws_vpc.main.id
      cidr_block = "10.0.2.0/24"
    }
    ```

### Part 3: Routing 🗺️

1.  **Add Route Tables:**
    ```hcl
    # Public Route Table (Traffic -> IGW)
    resource "aws_route_table" "public_rt" {
      vpc_id = aws_vpc.main.id

      route {
        cidr_block = "0.0.0.0/0" # The Internet
        gateway_id = aws_internet_gateway.gw.id
      }
    }

    # Associate Public Subnet with Public RT
    resource "aws_route_table_association" "a" {
      subnet_id      = aws_subnet.public.id
      route_table_id = aws_route_table.public_rt.id
    }
    ```

### Part 4: Analysis 🧠

**Scenario:**
-   **Instance A** is in `public` subnet.
-   **Instance B** is in `private` subnet.

**Questions:**
1.  Can Instance A reach Google.com? **Yes** (via IGW).
2.  Can Instance B reach Google.com? **No** (No route to IGW).
3.  Can I SSH into Instance A? **Yes** (It has a Public IP + IGW).
4.  Can I SSH into Instance B? **No** (Private IP only).

**Solution for B:** Use a **Bastion Host** (Jump Box) in Subnet A, or a **NAT Gateway**.

---

## 🎯 Challenges

### Challenge 1: The NAT Gateway (Difficulty: ⭐⭐⭐)

**Task:**
Add Terraform code for a NAT Gateway.
1.  Create an Elastic IP (`aws_eip`).
2.  Create `aws_nat_gateway` in the **Public** subnet.
3.  Create a Private Route Table that routes `0.0.0.0/0` to the NAT Gateway.

### Challenge 2: Security Groups (Difficulty: ⭐⭐)

**Task:**
Define a Security Group for the Database (Private Subnet).
Rule: Allow Port 5432 (Postgres) ONLY from the CIDR of the Public Subnet (`10.0.1.0/24`).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```hcl
resource "aws_eip" "nat" {}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public.id
}

resource "aws_route_table" "private_rt" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }
}
```

**Challenge 2:**
```hcl
resource "aws_security_group" "db_sg" {
  vpc_id = aws_vpc.main.id
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.1.0/24"]
  }
}
```
</details>

---

## 🔑 Key Takeaways

1.  **Isolation**: Databases should always be in Private Subnets.
2.  **Cost**: NAT Gateways cost money (hourly). IGWs are free.
3.  **IaC**: Defining networks in code (Terraform) is safer than clicking in the console.

---

## ⏭️ Next Steps

We have the design. Let's build the final project.

Proceed to **Lab 4.10: Networking Capstone Project**.
