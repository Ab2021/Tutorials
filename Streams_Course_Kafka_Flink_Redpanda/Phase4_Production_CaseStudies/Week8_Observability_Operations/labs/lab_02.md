# Lab 02: Redpanda Metrics

## Difficulty
 Easy

## Estimated Time
30 mins

## Learning Objectives
- Access Redpanda metrics
- Configure Prometheus

## Solution
<details>
<summary>Solution</summary>

`yaml
# prometheus.yml
scrape_configs:
  - job_name: redpanda
    static_configs:
      - targets: [redpanda:9644]
`
</details>