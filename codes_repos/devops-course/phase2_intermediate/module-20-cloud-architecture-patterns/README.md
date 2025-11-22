# Cloud Architecture Patterns

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Cloud Architecture, including:
- **Scalability**: Scaling Up (Vertical) vs Scaling Out (Horizontal).
- **Reliability**: Designing systems that survive failure (Multi-AZ, Multi-Region).
- **Performance**: Using Caching (CDN/Redis) and Event-Driven patterns.
- **Cost**: Optimizing spend with Spot Instances and Savings Plans.
- **Framework**: Applying the **AWS Well-Architected Framework**.

---

## 📖 Theoretical Concepts

### 1. Scaling Strategies

- **Vertical Scaling**: "Make the server bigger" (t2.micro -> t2.large). Easy, but has a limit. Requires downtime.
- **Horizontal Scaling**: "Add more servers". Infinite scale. No downtime. Requires a Load Balancer.

### 2. Reliability Patterns

- **Redundancy**: N+1 rule. If you need 1 server, run 2.
- **Circuit Breaker**: If Service A calls Service B and B fails, stop calling B immediately to prevent cascading failure.
- **Bulkhead**: Isolate components so failure in one doesn't crash the whole ship.

### 3. Caching & Performance

- **CDN (Content Delivery Network)**: Cache static assets (images, CSS) at the edge (CloudFront).
- **In-Memory Cache**: Cache database queries in RAM (Redis/Memcached).
- **Event-Driven**: Decouple services using Queues (SQS) and Topics (SNS). "Fire and Forget".

### 4. The Well-Architected Framework

1.  **Operational Excellence**: Automate changes, respond to events.
2.  **Security**: Protect data and systems.
3.  **Reliability**: Recover from failure.
4.  **Performance Efficiency**: Use computing resources efficiently.
5.  **Cost Optimization**: Avoid unnecessary costs.
6.  **Sustainability**: Minimize environmental impact.

---

## 🔧 Practical Examples

### Auto Scaling Group (Terraform)

```hcl
resource "aws_autoscaling_group" "web" {
  desired_capacity   = 2
  max_size           = 5
  min_size           = 1
  vpc_zone_identifier = ["subnet-123", "subnet-456"]

  launch_template {
    id      = aws_launch_template.web.id
    version = "$Latest"
  }
}
```

### Circuit Breaker (Python Pseudocode)

```python
def call_service_b():
    if failures > threshold:
        return "Service B is temporarily unavailable"
    
    try:
        return http.get("http://service-b")
    except Timeout:
        failures += 1
        return "Timeout"
```

### Event-Driven (SQS)

```python
import boto3

sqs = boto3.client('sqs')

# Send message
sqs.send_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/123/my-queue',
    MessageBody='Process Order #101'
)
```

---

## 🎯 Hands-on Labs

- [Lab 20.1: High Availability (Auto Scaling)](./labs/lab-20.1-high-availability.md)
- [Lab 20.2: Disaster Recovery (Cross-Region Replication)](./labs/lab-20.2-disaster-recovery.md)
- [Lab 20.3: Disaster Recovery](./labs/lab-20.3-disaster-recovery.md)
- [Lab 20.4: Multi Region](./labs/lab-20.4-multi-region.md)
- [Lab 20.5: Caching Strategies](./labs/lab-20.5-caching-strategies.md)
- [Lab 20.6: Cdn Implementation](./labs/lab-20.6-cdn-implementation.md)
- [Lab 20.7: Microservices Architecture](./labs/lab-20.7-microservices-architecture.md)
- [Lab 20.8: Event Driven](./labs/lab-20.8-event-driven.md)
- [Lab 20.9: Serverless Intro](./labs/lab-20.9-serverless-intro.md)
- [Lab 20.10: Architecture Best Practices](./labs/lab-20.10-architecture-best-practices.md)

---

## 📚 Additional Resources

### Official Documentation
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/)

### Books
- "Designing Data-Intensive Applications" by Martin Kleppmann.

---

## 🔑 Key Takeaways

1.  **Design for Failure**: Assume everything will fail.
2.  **Loose Coupling**: Components should not know the internal details of other components.
3.  **Statelessness**: Servers should be cattle, not pets. Store state in DB/Redis, not on the server disk.
4.  **Right Sizing**: Don't use a sledgehammer to crack a nut. Use the smallest instance type that works.

---

## ⏭️ Next Steps

1.  Complete the labs to build a resilient, auto-scaling architecture.
2.  **Congratulations!** You have completed Phase 2 (Intermediate). Proceed to **Phase 3 (Advanced)** to master Kubernetes, GitOps, and Chaos Engineering.
