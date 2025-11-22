# Capacity Planning - Deep Dive

## Advanced Sizing

### Network Bandwidth
- Ingress: Producer traffic
- Egress: Consumer + Replication traffic
- Rule: Egress = Ingress  (Consumers + Replication - 1)

### CPU Sizing
- Compression: CPU-intensive
- Encryption: CPU-intensive
- Rule: Monitor CPU, scale when > 70%
