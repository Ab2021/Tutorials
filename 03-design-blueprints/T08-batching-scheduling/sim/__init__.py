"""T08 -- batching and scheduling: admission is the optimisation.

Continuous batching is the single largest free win in the serving stack, and it is not a kernel or
a cache trick -- it is a decision about WHEN a slot is refilled. This package models the layer that
makes that decision:

  scheduler.py   the iteration-level admission loop; static vs continuous utilisation
  fairness.py    FCFS, priority bands, least-attained service, turn priority
  policies.py    chunked prefill, the saturation gate, goodput
  experiments.py seven scheduling decisions an operator actually has to make

Two facts drive everything here, and they are easy to confuse because they are opposites:

  * least-attained service addresses COMPUTE saturation  (dispatch cycles are the scarce resource)
  * turn priority        addresses KV CACHE saturation  (memory occupancy is the scarce resource)

They are not two settings in one slot -- they are responses to two different bottlenecks, which is
why the corpus describes integrating both rather than choosing [T].

Provenance is inline in each module: [T] corpus, [R] supporting repo, [D] derived.
"""
from .scheduler import (Seq, SchedulerState, static_utilisation, continuous_utilisation,
                        run_continuous, static_schedule, throughput, completion_stats)
from .fairness import (AgentProgram, Request, TPS_PER_CYCLE, POLICIES, fcfs_order,
                       priority_band_order, least_attained_order, turn_priority_order,
                       dispatch, starve_report)
from .policies import (prefill_chunk_size, chunk_plan, chunking_effect, saturation_gate,
                       goodput, batch_sweep)
from . import experiments

__all__ = [
    "Seq", "SchedulerState", "static_utilisation", "continuous_utilisation",
    "run_continuous", "static_schedule", "throughput", "completion_stats",
    "AgentProgram", "Request", "TPS_PER_CYCLE", "POLICIES", "fcfs_order", "priority_band_order",
    "least_attained_order", "turn_priority_order", "dispatch", "starve_report",
    "prefill_chunk_size", "chunk_plan", "chunking_effect", "saturation_gate",
    "goodput", "batch_sweep", "experiments",
]
