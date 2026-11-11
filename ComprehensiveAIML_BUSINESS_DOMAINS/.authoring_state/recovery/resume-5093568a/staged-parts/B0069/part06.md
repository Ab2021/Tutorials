## 9. Complexity & Cost Analysis

Costs are computed for a declared reference workload so the arithmetic can be checked: `N = 50,000,000` item vectors, dimension `d = 128`, float32 storage, `Q = 1,000,000` requests per day, candidate budget `K = 100`, and a compute price of `$0.04` per core-hour. All prices and rates are ILLUSTRATIVE assumptions; the structure of the calculation is the deliverable, not the dollar figures.

### 9.1 Storage

```text
FORMULA: vector_bytes = N * d * bytes_per_element
WHERE:   N = item count, d = dimension, one element per float32
CONSTRAINT: no replicas, no index overhead, no id map
NUMERICAL EXAMPLE: 50,000,000 * 128 * 4 = 25,600,000,000 bytes
                   25,600,000,000 / 2^30 = 23.84 GiB raw vectors
                   int8 quantization: 50,000,000 * 128 = 5.96 GiB
                   product quantization at 96 bytes/vector: 4.47 GiB
INTERPRETATION: quantization buys a 4x to 5x memory reduction, which is the only
                way this catalog fits on one node
VALIDITY: dimension and item count fixed; per-replica, add index graph links
```

Index overhead is not small. A graph index stores neighbour lists in addition to vectors; at 64 links per node and 4 bytes per link the adjacency costs `50,000,000 * 64 * 4 = 12.8 GB`, comparable to the quantized vector data itself. An IVF index instead stores centroids (`4096 * 128 * 4 = 2.10 MB`) plus an inverted list per centroid, plus an id map at 8 bytes per item (`50,000,000 * 8 / 2^20 = 381.5 MiB`). The id map is easy to forget and cannot be avoided if the index stores internal offsets.

### 9.2 Query cost: full scan versus partitioned scan

A full scan touches every vector: `N * d = 6.4e9` multiply-accumulate operations per query, or 6.4 GFLOP. At an assumed 2 GFLOP/s effective per core, that is 3.2 core-seconds per query.

```text
FORMULA: core_seconds_total = Q * N * d / throughput
WHERE:   Q = 1e6 queries/day, throughput = 2e9 flops/s/core
CONSTRAINT: single-threaded per query; ignore memory bandwidth limits
NUMERICAL EXAMPLE: 1e6 * 6.4e9 / 2e9 = 3.2e6 core-seconds = 888.9 core-hours
                   at $0.04/core-hour: $35.56 per day
INTERPRETATION: mathematically affordable, operationally useless at 120 ms of
                budget per request, because each query needs 3.2 core-seconds
VALIDITY: real kernels achieve less than peak; bandwidth-bound scans are worse
```

The latency failure, not the bill, is what rules out a full scan. At a 120 ms budget per request, 3.2 core-seconds requires 27 cores per in-flight request; with 64 concurrent requests that is 1,707 cores, and the queueing behaviour at that utilisation makes tail latency unmanageable.

An IVF index with 4,096 centroids probing 32 lists scans `50,000,000 / 4096 * 32 = 390,625` vectors, or `390,625 * 128 = 50,000,000` operations — a 128-fold reduction.

```text
FORMULA: scanned = (N / nlist) * nprobe; flops = scanned * d
WHERE:   nlist = 4096, nprobe = 32
CONSTRAINT: nprobe fixed; centroids fit in memory; lists balanced
NUMERICAL EXAMPLE: 390,625 * 128 = 5.0e7 flops = 0.025 core-seconds per query
                   1e6 queries -> 6.94 core-hours -> $0.278 per day
INTERPRETATION: the scan cost falls by a factor of 128, but fidelity must be
                re-measured because nprobe trades recall for that saving
VALIDITY: a skewed list distribution scans more than nprobe * N/nlist
```

### 9.3 Build cost

Training 4,096 centroids by k-means for 25 iterations over 50 million vectors costs `4096 * 25 * 50,000,000 * 128 = 6.55e14` operations, or about 91 core-hours at the same throughput, roughly \$4 per full rebuild. That figure is misleading in practice because the dominant build cost is I/O and the assignment pass over the full corpus, not the centroid arithmetic; a rebuild that must re-embed a changed catalog is bounded by the embedding throughput instead. At an assumed 20,000 vectors per second, a full 50 million vector index build takes `50,000,000 / 20,000 / 3600 = 0.69` hours, and that number, not the k-means cost, sets the rebuild cadence.

### 9.4 The cost of over-retrieval

Over-retrieval is cheaper than it looks at the index and more expensive than it looks at the ranker, because the ranker's per-candidate cost is a forward pass while the index's is a distance computation.

| Quantity | Value | Derivation |
|---|---:|---|
| Required candidates k | 100 | product requirement |
| Filter pass rate f | 0.13 | measured |
| n at α = 0.05 | 1,593 | Section 4.3.3 |
| n at naive 1/f | 769 | expected-value rule |
| Index cost ratio | 2.07x | 1593 / 769 |
| Ranker forward passes per request | 100 | unchanged; filtering precedes ranking |
| Slate | 12 | product requirement |

The last two rows carry the point: over-retrieval inflates the index scan by 2.07x and leaves the ranker bill unchanged, because the filter sits between them. A pipeline that instead over-retrieves into the ranker and filters after ranking pays 15.93x on the ranker as well, which is the most common way this cost is mis-modelled.

### 9.5 Latency budget

| Stage | Budget (ms) | Failure mode when exceeded | Mitigation |
|---|---:|---|---|
| Auth and context fetch | 8 | Blocks all retrieval | Cache entitlements with policy-keyed entries |
| Query encoding | 5 | Serialises the request | Batch or precompute for known users |
| ANN search (n = 1,593, nprobe 32) | 25 | Tail latency blows out under load | Cap nprobe under load; degrade n before quality |
| Eligibility filter and dedupe | 3 | CPU-bound on large lists | Hash membership with preallocated sets |
| Ranker over 100 candidates | 30 | Linear in candidate count | Batch inference; cap candidates, not features |
| Constraints and final check | 7 | Adds a serial hop | Parallelise the check with slate assembly |
| Serialisation and render | 15 | Outside the serving tier | Excluded from the retrieval SLO |
| **Total inside the tier** | **78** | Against a 120 ms budget | 42 ms headroom for queueing |

Little's law converts the budget into a concurrency requirement. At 1,200 requests per second and 12 ms mean service time inside the retrieval tier, mean in-flight requests are `1,200 * 0.012 = 14.4`. A 64-worker pool at the same service time supports `64 / 0.012 = 5,333` requests per second in the mean, so the binding constraint is the tail rather than the mean: a 1 percent tail at 90 ms consumes `1,200 * 0.090 = 108` of those 64 workers and the pool saturates. This is why the latency table above must be read at a percentile, not a mean.

### 9.6 Cost per point of quality

A useful decision ratio is marginal cost per marginal recall point. Increasing K from 100 to 200 costs an extra 100 ranker forward passes per request (from 30 ms to roughly 60 ms of ranker time at the stated rates) and raises candidate recall by whatever the measured curve says — on the ILLUSTRATIVE curve in Section 5.6, from 0.7143 to 0.8571, or 14.28 percentage points. The ratio is 30 ms per 14.28 points, about 2.1 ms per point. Compare that with the alternative of relaxing the slate cap, which costs nothing in latency and recovers the same 14.28 points on this request. The ratio ranks the options; the product constraint decides between them.

### 9.7 What this section cannot tell you

Three costs are excluded because they are dominated by factors this card does not model: GPU versus CPU inference economics, cross-region replication and egress, and the engineering cost of maintaining an eligibility pipeline that must be audited. The last is usually the largest and is never in the capacity plan. A team that costs only the retrieval tier will consistently under-invest in the eligibility tier, which is the component whose failure is not a quality regression but a compliance incident.

**Takeaway:** the index is cheap per query and expensive to rebuild; the ranker is cheap per query and linear in candidates; the eligibility path is cheap per query and expensive to get wrong.
