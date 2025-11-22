# Lab 20.1: High Availability (Auto Scaling)

## 🎯 Objective

Survive the traffic spike. You will build a **Highly Available** architecture using Terraform. It includes an Application Load Balancer (ALB) and an Auto Scaling Group (ASG) that automatically adds servers when CPU usage is high.

## 📋 Prerequisites

-   Terraform installed.
-   AWS Account.

## 📚 Background

### Concepts
-   **High Availability (HA)**: System remains operational even if components fail. (Deploy to 2 AZs).
-   **Scalability**: System handles increased load by adding resources.
-   **ASG (Auto Scaling Group)**: Manages the fleet of EC2 instances.
-   **ALB (Application Load Balancer)**: Distributes traffic to the ASG.

---

## 🔨 Hands-On Implementation

### Part 1: The Launch Template 📄

This defines *what* to launch (Ubuntu, t2.micro, Apache).

1.  **Create `main.tf`:**
    ```hcl
    provider "aws" { region = "us-east-1" }

    resource "aws_launch_template" "web" {
      name_prefix   = "web-"
      image_id      = "ami-0aa2b7722dc1b5612" # Ubuntu 20.04
      instance_type = "t2.micro"

      user_data = base64encode(<<-EOF
                  #!/bin/bash
                  apt update
                  apt install -y apache2
                  echo "Hello from $(hostname)" > /var/www/html/index.html
                  systemctl start apache2
                  EOF
      )
    }
    ```

### Part 2: The Auto Scaling Group 📈

This defines *how many* to launch and *where*.

1.  **Add to `main.tf`:**
    ```hcl
    resource "aws_autoscaling_group" "web" {
      desired_capacity    = 2
      max_size            = 5
      min_size            = 1
      vpc_zone_identifier = ["subnet-xyz", "subnet-abc"] # Replace with your Default VPC Subnets

      launch_template {
        id      = aws_launch_template.web.id
        version = "$Latest"
      }
    }
    ```
    *Note:* You need to find your Default VPC Subnet IDs in the AWS Console.

### Part 3: The Load Balancer ⚖️

1.  **Add to `main.tf`:**
    ```hcl
    resource "aws_lb" "web" {
      name               = "web-alb"
      internal           = false
      load_balancer_type = "application"
      security_groups    = ["sg-xyz"] # Replace with Default VPC Security Group
      subnets            = ["subnet-xyz", "subnet-abc"]
    }

    resource "aws_lb_target_group" "web" {
      name     = "web-tg"
      port     = 80
      protocol = "HTTP"
      vpc_id   = "vpc-xyz" # Replace with Default VPC ID
    }

    resource "aws_lb_listener" "front_end" {
      load_balancer_arn = aws_lb.web.arn
      port              = "80"
      protocol          = "HTTP"

      default_action {
        type             = "forward"
        target_group_arn = aws_lb_target_group.web.arn
      }
    }

    # Attach ASG to ALB
    resource "aws_autoscaling_attachment" "asg_attachment" {
      autoscaling_group_name = aws_autoscaling_group.web.id
      lb_target_group_arn    = aws_lb_target_group.web.arn
    }
    ```

### Part 4: Deploy & Test 🚀

1.  **Apply:**
    `terraform apply`.

2.  **Verify:**
    -   Go to EC2 Console. You see 2 instances.
    -   Go to Load Balancers. Copy DNS Name.
    -   Refresh browser. You should see "Hello from ip-10-..." changing (Load Balancing).

3.  **Simulate Failure:**
    -   Terminate one instance manually.
    -   Wait 1 minute.
    -   ASG detects it and launches a new one. **Self-Healing!**

---

## 🎯 Challenges

### Challenge 1: Dynamic Scaling Policy (Difficulty: ⭐⭐⭐)

**Task:**
Add a scaling policy.
"If Average CPU > 50%, add 1 instance."
*Hint:* `aws_autoscaling_policy` with `TargetTrackingScaling`.

### Challenge 2: Stress Test (Difficulty: ⭐⭐)

**Task:**
SSH into an instance. Run `stress --cpu 2`.
Watch CloudWatch.
Watch the ASG launch new instances automatically.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```hcl
resource "aws_autoscaling_policy" "cpu" {
  name                   = "cpu-scaling"
  autoscaling_group_name = aws_autoscaling_group.web.name
  policy_type            = "TargetTrackingScaling"
  target_tracking_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ASGAverageCPUUtilization"
    }
    target_value = 50.0
  }
}
```
</details>

---

## 🔑 Key Takeaways

1.  **Statelessness**: For ASG to work, instances must be identical. Don't store user sessions on the disk (use Redis/DB).
2.  **Cost**: You pay for the ALB and the running instances.
3.  **Golden AMI**: Instead of running `apt install` on every boot (slow), build an AMI with Packer that has everything pre-installed (fast).

---

## ⏭️ Next Steps

We can survive a server failure. Can we survive a Region failure?

Proceed to **Lab 20.2: Disaster Recovery**.
