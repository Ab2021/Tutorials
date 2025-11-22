# Flink Monitoring - Interview

## Questions

1. **Q: How to detect Flink backpressure?**
   - A: Check Web UI Backpressure tab or monitor outPoolUsage metric

2. **Q: What causes high checkpoint duration?**
   - A: Large state, slow storage, or network issues

### Troubleshooting
- High numRestarts: Check logs for exceptions
- Increasing lag: Scale up parallelism
