"""T17 -- observability, tracing and evaluation design blueprint: runnable core.

    python run.py

Nine demonstrations, stdlib-only, offline, no GPU and no network. Each is a DECISION rather than a
number: latency attribution, the sampling cost/coverage frontier, head sampling's rarity blind spot,
the three judge-bias corrections, the two scoring modes, eval-set sizing, the gate's choice of
statistic, and the feedback loop that makes tracing pay for itself.

Corpus figures are labelled [T]/[R] in the output; every model number is reproducible from this
package. No fabricated benchmarks -- the judge and the failure taxonomy are synthetic and say so.
"""
from __future__ import annotations

import sys

from sim import experiments as E


def main() -> int:
    print("=" * 78)
    print("T17 -- OBSERVABILITY, TRACING & EVALUATION: three planes and the loop between them")
    print("=" * 78)
    print("\n[corpus] 'A trace is a single request through your app. Inside it, each step is a span.")
    print("          ... Nest those spans, and you can replay exactly what happened in order with")
    print("          the numbers attached to each step. The mystery becomes a timeline.' [T]")
    print("[corpus] 'In high-traffic production, sample the successes, but always keep 100% of the")
    print("          errors. And redact personal data at the boundary before it is ever written.' [T]")
    print("[corpus] 'the judge itself is biased. It tends to favor the first answer it sees. It")
    print("          rewards length even when length adds nothing, and it flatters outputs from its")
    print("          own model family.' [T] -- Jung et al. 2023")
    print("[corpus] 'With 20 examples, one lucky run looks like real progress. ... A mean of 4.2 can")
    print("          still hide the 5% of answers that leak data or invent facts.' [T]")
    print("[corpus] 'Worst of all is the dashboard nobody owns. If no alert fires and no eval reads")
    print("          the traces, you have paid for storage, not for insight.' [T]")

    E.run_all()

    print("\n" + "=" * 78)
    print("THE ONE-PARAGRAPH SUMMARY")
    print("=" * 78)
    print("""
Observability is three planes wearing one name, and every failure in this topic is a team running
one of them. The first is METRICS -- TTFT, ITL, goodput, queue depth, KV occupancy -- which answer
"is it healthy" and cannot answer "why". The second is TRACES: a request as a tree of nested spans,
which is valuable for exactly one reason, that it DECOMPOSES the end-to-end time instead of
measuring it. The demonstration of that is the agentic request, where ranking spans by COUNT and
ranking them by TOTAL give different answers: twelve tool calls beat five LLM turns on count and
lose badly on wall clock. A metrics-only dashboard cannot even ask the question, because a metric
has no name attached. The third plane is EVALUATION, and it is where the topic gets hard, because
unlike the other two it has no ground truth unless you build one. This package therefore does not
measure a judge -- it builds one with a KNOWN latent quality, injects each of the three documented
biases as a parameter, and measures how far the verdicts move. That measurement disagrees with the
corpus's own remedy: randomising the answer order, which the corpus prescribes, buys almost nothing
here (+0.0 to +1.7 points across the sweep), because it decorrelates the bias from the label without
removing the bias. Running BOTH orders and treating a disagreement as undecided is what actually
cancels it -- and the share it cannot decide turns out to be a calibrated measurement of how biased
the judge is, which nothing else in the stack reports. The corrections are also not additive: the
best single one gets most of the gain and the bundle gets little more, so a team that ships "the
three fixes" together pays for a second judge vendor to find out it bought almost nothing.
The same discipline applies to eval-set size, where three different problems
improve at three different rates -- power, false confidence, and tail representation -- and the
third usually binds, because a mean cannot report a problem in a class the set does not contain.
And the gate inherits all of it: over the same release scores, a mean gate passes releases the tail
gates fail, which is the same tail-over-mean pattern this knowledge base finds independently in
T08's goodput, T09's p99 and T10's SNR. The fourth thing is not a plane but an EDGE, and it is the
one the corpus says is the whole point: traces that feed a judge that feeds a dataset that feeds the
gate. Without it the first three are a storage bill -- the open loop stores every trace, prevents
zero defects and has a return of zero per gigabyte, while the closed loop differs in exactly one
edge and its escape rate FALLS over releases instead of staying flat. The value of that edge is not
a constant factor; it compounds, because each failure class is converted from a recurring liability
into a permanent asset.
""".strip())
    print("\nOK -- all nine demonstrations completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
