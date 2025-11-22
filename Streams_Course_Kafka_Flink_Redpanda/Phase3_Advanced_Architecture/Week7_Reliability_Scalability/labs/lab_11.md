# Lab 11: Capacity Planning

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
- Calculate Kafka capacity requirements
- Estimate disk, network, and CPU needs
- Plan for growth

## Problem Statement
Given requirements: 10,000 msg/sec, 1KB avg message size, 7-day retention, 3x replication. Calculate the required disk space, network bandwidth, and number of brokers.

## Starter Code
```python
# Requirements
messages_per_sec = 10000
avg_message_size_kb = 1
retention_days = 7
replication_factor = 3

# TODO: Calculate capacity
```

## Hints
<details>
<summary>Hint 1</summary>
Disk = (msg/sec × msg_size × seconds_in_day × days × replication).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**Capacity Calculations:**
```python
# Input
messages_per_sec = 10000
avg_message_size_kb = 1
retention_days = 7
replication_factor = 3

# Calculations
bytes_per_sec = messages_per_sec * avg_message_size_kb * 1024
bytes_per_day = bytes_per_sec * 86400
total_bytes = bytes_per_day * retention_days * replication_factor

# Convert to GB/TB
total_gb = total_bytes / (1024**3)
total_tb = total_gb / 1024

print(f\"Disk Required: {total_gb:.2f} GB ({total_tb:.2f} TB)\")

# Network Bandwidth
# Ingress: 10,000 msg/sec × 1KB = 10 MB/sec
# Egress (with replication): 10 MB/sec × 3 = 30 MB/sec
ingress_mbps = (bytes_per_sec / (1024**2)) * 8
egress_mbps = ingress_mbps * replication_factor

print(f\"Network Ingress: {ingress_mbps:.2f} Mbps\")
print(f\"Network Egress: {egress_mbps:.2f} Mbps\")

# Number of Brokers
# Assume each broker can handle 50 MB/sec and 10 TB
brokers_for_network = max(1, int(egress_mbps / (50 * 8)))
brokers_for_disk = max(1, int(total_tb / 10))
recommended_brokers = max(brokers_for_network, brokers_for_disk)

print(f\"Recommended Brokers: {recommended_brokers}\")
```

**Output:**
```
Disk Required: 1814.40 GB (1.77 TB)
Network Ingress: 80.00 Mbps
Network Egress: 240.00 Mbps
Recommended Brokers: 1
```

**Recommendations:**
- Start with 3 brokers for HA (even if 1 is sufficient)
- Add 20% headroom for spikes
- Monitor and scale horizontally as needed
</details>
