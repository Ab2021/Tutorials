#!/usr/bin/env python3
"""T15 — Autoscaling and SLOs: runnable core.

Proves the design's central mechanism: an autoscaler for LLM serving is a CONTROL LOOP
WITH A DEAD TIME, and the dead time -- the cold start -- dominates every other decision.
The signal you scale on matters. The mechanism you scale with matters more.

Four findings the model is built to make visible:

  1. A compressed CPU signal fires late; a queue/KV signal fires on time.
  2. Even a PERFECT instantaneous signal loses to a predictive one, because a replica
     that starts booting now serves nothing for the whole boot window.
  3. Scale-down must be slower than the cold start, or the controller cancels replicas
     it has already paid to boot.
  4. Past a long enough cold start, AUTOSCALING LOSES TO OVERPROVISIONING OUTRIGHT.

WHAT THIS IS:  a model of a control loop.
WHAT THIS IS NOT:  a benchmark. No cluster scaled, no replica booted, no SLO was
                   measured. Every number is a simulation output from sim/autoscaler.py.

Run:  python run.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import autoscaler as a  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    # --------------------------------------------------------------- the contradiction
    rule("1. The corpus contradicts itself, and that is the finding")
    print("  Two guide files in the corpus give incompatible autoscaling advice, and this")
    print("  blueprint records the conflict rather than silently picking a winner.")
    print()
    print("  File A -- 11-infrastructure-and-mlops/01-llm-infrastructure.md -- ships a")
    print("  Kubernetes HorizontalPodAutoscaler for an LLM service whose metrics are [R]:")
    print()
    print("      - type: Resource")
    print("        resource: {name: cpu, target: {type: Utilization, averageUtilization: 70}}")
    print("      - type: Pods")
    print("        pods: {metric: {name: requests_per_second}, target: {averageValue: 100}}")
    print()
    print("  File B -- 04-inference-optimization/06-serving-infrastructure.md -- says the")
    print("  opposite in one line [R]:")
    print()
    print("      Autoscaling: Scaling based on KV Cache utilization rather than CPU or")
    print("      standard memory usage.")
    print()
    print("  File B is right and File A is a landmine, and the model below quantifies how")
    print("  big a landmine. But the honest framing is not 'A is wrong'. A's HPA is")
    print("  syntactically valid, applies cleanly, and will run in production for months")
    print("  without ever erroring. It scales on a signal that is only loosely coupled to")
    print("  whether requests are meeting their latency budget.")
    print()
    print("  This is a first-class edge case, not a documentation bug: an operator")
    print("  following the more detailed, more concrete guide gets the worse outcome.")
    print("  Resolving it silently would hide the only interesting thing about it.")

    # ------------------------------------------------------------------ the control loop
    rule("2. The control loop: signal choice at the corpus's own cold start")
    print("  A bursty workload against a fleet that takes time to grow. The burst is a")
    print("  PLATEAU, not a spike -- 240 steps of 44 req/step against a base of 8 -- so a")
    print("  control loop that reacts in time can actually track it. A spike would only")
    print("  measure the delay, not the policy.")
    print()
    sc = a.Scenario()
    print(f"  base {sc.base_rate:.0f} req/step, burst {sc.peak_rate:.0f} req/step for"
          f" {sc.burst_len} steps, then back down")
    print(f"  for {sc.settle_steps} steps -- which is where the scale-DOWN behaviour gets"
          " tested.")
    print(f"  One replica serves {sc.rep_capacity:.0f} req/step. SLO budget: a request"
          f" must wait no more than")
    print(f"  {sc.slo_wait_steps} steps.")
    print(f"  Cold start {sc.boot_steps} steps -- the corpus's best case, 15-20 seconds")
    print("  from an un-quantized base image with weights on a fast mount [R].")
    print()
    rows = a.run_all(sc)
    base = [r for r in rows if r.policy == "static_peak"][0]
    print(a.render(rows, baseline=base))
    print()
    print("  'replica-steps' is cost, and it counts BOOTING replicas too. They cost money")
    print("  and serve nothing. 'boot waste' is the share of the bill spent that way, and")
    print("  it is the number a ready-replica count hides.")
    print()
    print("  Read the two ends first:")
    print("    static_min -- 2 replicas, the cheapest possible fleet, 11.9% SLO. This is")
    print("      the 'we have an autoscaler, why is it slow' configuration.")
    print("    static_peak -- overprovision for the peak. 100% SLO at 2,700 replica-steps.")
    print("      This is the baseline every autoscaler must beat, and it is a strong one.")
    print()
    print("  Now the signal comparison, which is File A versus File B:")
    print("    cpu       (File A, 70% CPU target)   74.6% SLO at 2,320 replica-steps")
    print("    queue     (File B, KV/queue-shaped) 100.0% SLO at 4,804 replica-steps")
    print()
    print("  The CPU target costs 25 points of SLO. Two independent reasons, and it is")
    print("  worth separating them because they have different fixes:")
    print()
    print("    (i)  COMPRESSION. CPU tops out around 0.85 even when the fleet is")
    print("         hopelessly behind, because the CPU orchestrates while the accelerator")
    print("         does the work. A 0.70 target is reached only at ~2.5x overload. [D]")
    print("    (ii) THE CONTROL LAW ITSELF. The HPA computes")
    print("             desired = ceil(current * signal / target)")
    print("         which moves GEOMETRICALLY -- it approaches the setpoint rather than")
    print("         jumping to it. A fleet far from its setpoint is slow even with a")
    print("         perfect signal, and this is the part people miss.")
    print()
    print("  To show (ii) is real and not an artifact of (i), the model runs cpu_ideal:")
    print("  CPU that tracks load linearly with no lag and no ceiling.")
    rows2 = a.simulate(sc, "cpu_ideal")
    print(f"    cpu_ideal  {rows2.slo_rate:.1%} SLO at {rows2.replica_steps:,.0f}"
          f" replica-steps -- it reaches the SLO, and costs")
    print(f"    {rows2.replica_steps / base.replica_steps:.2f}x the static baseline with"
          f" {rows2.boot_waste:.1%} of the bill spent on replicas that")
    print("    were never ready. A perfect signal does not fix a slow loop.")

    # -------------------------------------------------------------- the predictive one
    rule("3. Prediction beats reaction, because the boot window cannot be removed")
    print("  queue_predict scales on the SAME queue signal, projected forward across the")
    print("  boot window using the observed arrival ramp. Everything else is identical.")
    print()
    rp = a.simulate(sc, "queue_predict")
    rq = a.simulate(sc, "queue")
    print(f"    queue          {rq.slo_rate:>6.1%} SLO   {rq.replica_steps:>8,.0f} "
          f"replica-steps   {rq.scale_events:>2} scale events")
    print(f"    queue_predict  {rp.slo_rate:>6.1%} SLO   {rp.replica_steps:>8,.0f} "
          f"replica-steps   {rp.scale_events:>2} scale events")
    print()
    print("  The predictor gives up some SLO and costs 53% less -- it is the cheapest")
    print("  policy in the whole table that tracks the workload at all. Reacting to the")
    print("  CURRENT queue is always boot_steps too late: by the time the replicas you")
    print("  started are ready, the queue that justified them has been waiting.")
    print()
    print("  This result does not depend on the CPU compression assumption at all. That")
    print("  is deliberate: it is the one finding here that holds for ANY signal.")

    # ------------------------------------------------ scale-down must exceed cold start
    rule("4. The scale-down cooldown must be longer than the cold start")
    print("  This is the most operationally valuable line in the model, and it is a rule")
    print("  rather than a measurement.")
    print()
    print("  If scale-down is as fast as scale-up, the controller cancels replicas that")
    print("  are STILL BOOTING. They were paid for, they never served, and the capacity")
    print("  the signal asked for never arrives. The controller eats its own tail.")
    print()
    na, de = a.cooldown_comparison(sc, "queue")
    print(a.render_cooldown(na, de, sc.boot_steps))
    print()
    print("  Same policy, same signal, same workload. Only the down-direction hysteresis")
    print("  differs. Deriving it from the boot time instead of hard-coding it converts")
    print(f"  {na.boot_waste:.1%} boot waste into {de.boot_waste:.1%} and takes the SLO from")
    print(f"  {na.slo_rate:.1%} to {de.slo_rate:.1%}.")
    print()
    print("  Note the cost goes UP slightly. That is the honest trade: the derived")
    print("  cooldown holds capacity longer, so it costs more, and it stops burning money")
    print("  on replicas that never serve. Total spend is the wrong lens here -- spend per")
    print("  SERVED request is the right one.")

    # ------------------------------------------------------------ boot sweep / crossover
    rule("5. Past a long enough cold start, autoscaling loses to overprovisioning")
    print("  The cold start is the dominant term, so sweep it rather than asserting it.")
    print("  Each cell is SLO attainment and replica-steps. The static_peak row is the")
    print("  baseline to beat at every delay.")
    print()
    print("  static_peak (5 replicas, always): 100.0% SLO at 2,700 replica-steps, for ANY")
    print("  boot delay -- its replicas were already running when the burst arrived.")
    print()
    print("  boot steps            cpu            queue    queue_predict")
    print("  ----------------------------------------------------------------")
    for boot in (5, 20, 60, 180):
        s2 = a.Scenario(boot_steps=boot)
        cells = []
        for p in ("cpu", "queue", "queue_predict"):
            r = a.simulate(s2, p)
            cells.append(f"{r.slo_rate:>6.1%} {r.replica_steps:>7,.0f}")
        print(f"  {boot:>11} " + " ".join(f"{c:>16}" for c in cells))
    print("                  " + " ".join(f"{'SLO':>6} {'cost':>7}" for _ in range(3)))
    print()
    print("  The crossover is the finding. At a 5-step cold start the predictive loop")
    print("  beats the static baseline on cost (2,015 vs 2,700) and reaches 94.2% SLO.")
    print("  By 60 steps the static overprovision beats every autoscaler on BOTH axes.")
    print("  By 180 the BEST autoscaler still costs three times as much as the static")
    print("  fleet, and gives up 69 points of SLO doing it.")
    print()
    print("  The reason is not subtle and it is not fixable by a better policy: an")
    print("  autoscaler's whole value proposition is that it adds capacity when load")
    print("  rises. If the load rises and falls inside the boot window, the capacity")
    print("  arrives after the need has gone. It then has to be paid for anyway.")
    print()
    print("  Stated as a design rule: AUTOSCALING ONLY PAYS WHEN THE COLD START IS SHORT")
    print("  RELATIVE TO THE BURST DURATION. Which makes cold-start reduction -- the")
    print("  corpus's un-quantized base images and fast weight mounts taking startup from")
    print("  minutes to 15-20 seconds [R] -- a prerequisite for autoscaling rather than a")
    print("  separate optimisation. File A's HPA is not merely on the wrong signal; it is")
    print("  a control loop installed on a plant with a long dead time.")

    # ------------------------------------------------------------------- zero to one
    rule("6. Scale to zero is unsolved, and the corpus says so")
    print("  The corpus contains a practitioner asking exactly the right question and a")
    print("  speaker declining to answer it [T]:")
    print()
    print("    Q: 'How do you handle the zero to one like we do autoscaling right? ...")
    print("        I'm using KA that was an issue like we cannot scale from 0 to 1, only")
    print("        from 1 to n.'")
    print("    A: 'I'm not very much aware of that path, the autoscaling path... I'm not")
    print("        an expert at that so I can't answer.'")
    print()
    print("  The gap is recorded here as a gap. There is no fabricated answer. What the")
    print("  model CAN contribute is why it is hard, as arithmetic anyone can check [D]:")
    print()
    print(f"    cold start C = {sc.boot_steps} steps, and at the corpus's best case of")
    print("    15-20 seconds of wall clock, that is C = 15-20 s.")
    print("    A first request after idle pays C before its first token.")
    print("    Interactive TTFT budgets are sub-second to a few seconds.")
    print("    Therefore scale-to-zero is viable only when C < the TTFT budget,")
    print("    which after 15-20 s of cold start is NEVER for interactive traffic.")
    print()
    print("  So the honest design position is not 'implement scale-to-zero'. It is:")
    print("    - scale to a WARM FLOOR sized for the p99 of the idle-period arrival rate;")
    print("    - treat 'zero' as a batch-tier option only, where no TTFT promise exists;")
    print("    - and if you must offer zero, put a queue in front of it and promise")
    print("      throughput, not latency.")
    print()
    print("  Note what this does NOT need: it needs no measurement, no model and no")
    print("  cluster. It is the boot time compared to the latency budget. The reason")
    print("  scale-to-zero keeps being attempted anyway is that the boot time lives in a")
    print("  different team's backlog from the SLO.")

    # -------------------------------------------------------------------- summary
    rule("Summary")
    print("  The corpus's two guide files disagree about the autoscaling signal, and the")
    print("  disagreement is a first-class edge case rather than a doc bug: the more")
    print("  concrete guide is the worse advice.")
    print("  A 70% CPU target costs 25 points of SLO against a KV/queue-shaped signal, for")
    print("  two separable reasons -- signal compression and the HPA's geometric control")
    print("  law. Fixing only the signal leaves the loop slow.")
    print("  Predicting across the boot window costs 53% less than reacting to the current")
    print("  queue, and that result holds for any signal.")
    print("  Scale-down hysteresis must exceed the cold start, or the controller cancels")
    print("  replicas it has already paid to boot.")
    print("  Past a long enough cold start, overprovisioning beats every autoscaler on")
    print("  BOTH SLO and cost. Cold-start reduction is a prerequisite for autoscaling,")
    print("  not a separate optimisation.")
    print("  Scale-to-zero is unsolved in the corpus and the speaker says so; the")
    print("  arithmetic says it is impossible for interactive traffic at a 15-20 s boot.")
    print()
    print("  No cluster scaled and no replica booted. Every figure is a model output from")
    print("  sim/autoscaler.py. [D]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
