# Runbook — Risk Service

## Signals

- `HighErrorRate`: 5xx > 2% for 10 minutes.
- `HighP99Latency`: p99 > 2 seconds for 10 minutes.
- `NoPredictions`: zero predictions for 30 minutes.
- Data-quality alert: missing/out-of-range feature rate > 5%.

## Triage

1. Check `/healthz` and `/readyz`.
2. Check deployment rollout and pod restarts.
3. Check model version and config map.
4. Check upstream data freshness and schema.
5. Check latency by endpoint and instance.
6. Check error logs without PHI.

## Rollback

```bash
kubectl rollout undo deployment/risk-service -n biomed
kubectl rollout status deployment/risk-service -n biomed
```

If the previous model artifact is incompatible, restore the prior image tag and model config together.

## Communication

- Notify clinical/operational owner.
- State whether predictions are unavailable, delayed, or unreliable.
- Do not silently fall back to a different model.
- Record the incident and the decision in the risk log.

## Post-incident

- Timeline, root cause, blast radius, detection gap, corrective action, owner, due date.
- Add a regression test or monitor.
- Update the model card and PCCP impact assessment if the model changed.
