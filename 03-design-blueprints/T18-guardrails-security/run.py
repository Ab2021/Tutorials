"""T18 -- guardrails and security design blueprint: runnable core.

    python run.py

Six demonstrations, stdlib-only, offline, no GPU and no network. Each is a DECISION rather than a
number: whether four rails are worth their arithmetic, which of two levers still has room, where the
false-refusal optimum sits for two different businesses, what a missing trust tag does to a gate, how
fast a rail decays between red-team runs, and what happens to human approval at volume.

Corpus figures are labelled [T]/[R] in the output; every model number is reproducible from this
package. No fabricated benchmarks -- the attack mix, the detector separation and the severity
weighting are synthetic and say so.
"""
from __future__ import annotations

import sys

from sim import experiments as E


def main() -> int:
    print("=" * 78)
    print("T18 -- GUARDRAILS & SECURITY: a competence matrix, a capability gate, and a schedule")
    print("=" * 78)
    print("\n[corpus] 'A model that is helpful by default is also gullible by default. It will follow")
    print("          instructions hidden in a document and answer questions it should refuse.' [T]")
    print("[corpus] 'Think of rails as filters wrapped around the model at every stage of a request.")
    print("          ... Screen the input, validate the output, and above all, gate the tool calls")
    print("          that can send money, email a customer, or delete a record. That last layer has")
    print("          the biggest blast radius.' [T]")
    print("[corpus] 'Teams reliably add the obvious input and output rails, and reliably skip the two")
    print("          that matter most: filtering the retrieved context and gating the tool calls.' [T]")
    print("[corpus] 'To the model the instructions and the data are just one stream of text and it has")
    print("          no built-in way to tell which is which.' [T]")
    print("[corpus] 'Tune the rails too tight and you frustrate real users with false refusals, which")
    print("          is its own kind of failure. ... a system that blocks everything is safe and")
    print("          useless.' [T]")
    print("[corpus] 'A rail you tested in March may be bypassed by June. So, red teaming has to be a")
    print("          recurring schedule, not a launch checkbox.' [T]")

    E.run_all()

    print("\n" + "=" * 78)
    print("THE ONE-PARAGRAPH SUMMARY")
    print("=" * 78)
    print("""
A guardrail stack is two different kinds of object wearing one name, and every design mistake in
this topic is a team building one of them and believing it built both. The first object DETECTS: it
reduces harm by recognising the attack. The corpus's sentence about it -- "no single rail is enough,
it is the layering that makes the system hard to break" -- is true, and the arithmetic everyone
attaches to that sentence is wrong, because a rail's catch rate is not a scalar. It is conditional on
the attack TECHNIQUE (a regex catches what somebody wrote a rule for and nothing else) and on the PATH
(a payload inside a retrieved document never travels through the user's message, so the input rail has
zero coverage of it, not low competence). Measured against an explicit technique-by-path mix, the
full four-rail stack catches 82.3% where its own scalar arithmetic claims 99.986%; the corpus's
"obvious two rails" that teams actually build catch 15.1% where their scalar arithmetic claims 99.040%.
The sharpest form of the point is a single number: one rail with FULL COVERAGE beats all four with
partial coverage and beats any other single rail, because coverage is not competence. And the
consequence-weighted catch is 5.4 points BELOW the frequency-weighted one, because the classes the
stack cannot see are the highest-severity ones -- the fifth independent topic in this knowledge base
where a frequency-weighted average reads better than a consequence-weighted one, after T08's goodput,
T09's p99, T10's SNR and T17's gate, now with an adversary choosing the tail.

That measurement also produces the topic's real design conclusion, which is not "improve the
classifier". The full stack is already within 1.1 points of its own CEILING, because the ceiling is
set by the residual on techniques nobody has a rule for -- so "train a better detector" buys a
rounding error, permanently. The second object does not detect anything: a capability GATE reduces
harm by removing the thing the attack needs, which makes its strength independent of attack novelty
and gives it a 22x factor that does not erode. The two multiply, so the honest statement is
harm = (1 - catch) x gate_failure, and the build order the corpus gives -- gate the tool calls above
all -- follows from the decomposition rather than from blast radius alone. The measurement makes it
sharp: turning the detectors completely OFF and adding a gate still beats running the detectors at
their theoretical maximum with no gate, by 3.7x.

What follows from the rest of the measurement is less comfortable. The gate's floor is not a
constant, it is a queue: human approval is a finite service, approver accuracy falls continuously
with queue depth, and the gate's strength swings 20x between a quiet day and a backlog with nothing
in the configuration having changed. That makes the approval RATE -- not the request rate -- the
metric that says when the control stopped being real, and it makes "add an undo path" the only
scaling answer, because approver headcount grows linearly with agent volume and irreversibility does
not have to. On top of all of it sits a clock: "a rail you tested in March may be bypassed by June"
is a schedule, and solving for it gives a trough of 45.5% catch at annual red-teaming against 93.4%
for continuous. Here the run DISAGREES with the rest of this knowledge base and says so: it does not
reproduce the average-hides-the-tail pattern, because the decay is fast enough that the published
mean itself falls 53.1 points over two years -- it is not hiding anything, it is simply a level where
a rate of change was needed. What it does show is worse in a quieter way: the novel-class reading
moves 0.0 points across the whole two years, because it starts at the residual and stays there, so
the instrument aimed at the population that is entirely severity-5 is silent by construction. Three
of these six experiments end in the same place -- a missing trust tag, a decaying rail and a busy
approver queue all weaken a control without any control reporting it. That is the topic's real
operational hazard, and it is not addressed by adding another classifier.
""")

    print("=" * 78)
    print("Six decisions, one sentence each:")
    print("=" * 78)
    print("  1. Four rails are not 1-(1-r)^4: a rail's catch is conditional on technique and path, and")
    print("     the scalar arithmetic overstates protection by 17.7 points on the full stack and 83.9")
    print("     points on the two rails teams actually build. One rail with full coverage beats all")
    print("     four with partial coverage.")
    print("  2. The detection approach is within 1.1 points of its own ceiling; the capability gate has")
    print("     22x. Build the gate first -- with the detectors off it still beats detectors at maximum")
    print("     with no gate, by 3.7x.")
    print("  3. The false-refusal optimum is a ratio you must name: 97.1% refusal for a regulated payer,")
    print("     3.6 SD the other way for a marketing bot, and the two disagree about which endpoint is")
    print("     safe. The fix is a cascade, not a better threshold.")
    print("  4. Fail-open on a lost trust tag doubles the gate's failure rate (0.045 -> 0.093) while")
    print("     every detection metric stays flat. Fail closed; that cost is measurable.")
    print("  5. A red-team schedule is the control and its failure mode is the interval: 45.5% trough")
    print("     catch at annual cadence against 93.4% continuous. The mean sees none of it usefully,")
    print("     and the novel-class reading cannot fire at all.")
    print("  6. Approval is a resource. Its failure mode is APPROVAL, its capacity is measured in")
    print("     headcount, and the gate's strength swings 20x with queue depth alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
