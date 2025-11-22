# Troubleshooting - Interview

## Scenarios

1. **Scenario: Kafka broker down**
   - Check: Other brokers taking load?
   - Fix: Restart broker, check logs

2. **Scenario: Flink job failing**
   - Check: Exception in logs?
   - Fix: Fix bug, restore from savepoint
