"""T19 -- finops and sovereignty design blueprint: runnable core.

    python run.py

Six demonstrations, stdlib-only, offline, no GPU and no network. Each is a DECISION rather than a
number: what a cache discount is actually worth, whether the corpus's "tenth of the cost" headline
survives its own ladder, which of three defensible orderings to spend, which unit to report in, what
a sovereignty posture can evidence, and how attribution behaves as the business grows.

Corpus figures are labelled [T]/[R] in the output; every model number is reproducible from this
package. No fabricated benchmarks -- the bill decomposition, the lever parameters and the team mix
are synthetic, marked [D], and say so.
"""
from __future__ import annotations

import sys

from sim import experiments as E


def main() -> int:
    print("=" * 78)
    print("T19 -- FINOPS & SOVEREIGNTY: what a discount is worth, and what a posture can prove")
    print("=" * 78)
    print("\n[corpus] 'Optimization without measurement is just guessing.'                   -- tokenomics [T]")
    print("[corpus] 'Companies are doing tokens per output as one of the metrics -- you need to")
    print("          change the metrics. You need to start looking at it as a business outcome")
    print("          rather than the token spend.'                                          -- tokenomics [T]")
    print("[corpus] 'Between February and April the cost of a 10,000-token knowledge artifact fell")
    print("          from 30c to 16c, while lines of code drafted fell from 630 to 91 and files")
    print("          touched from 8.2 to 3.6, and thinking tokens rose.'                     -- Token Raj [T]")
    print("[corpus] 'Stack the techniques, do not look for a hero optimization... before you touch")
    print("          the model at all, exhaust the free wins.'                              -- tokenomics [T]")
    print("[corpus] 'People look at sovereignty as near-binary -- sovereign or non-sovereign. I")
    print("          think that is wrong... you need to start looking at it as a system property,")
    print("          and this property has various dimensions.'                              -- sovereign AI [T]")
    print("[corpus] 'You cannot legislate a memory dump.'                                     -- sovereign AI [T]")
    print("[corpus] 'It is a trade-off. It is not going to run as fast as what it was running")
    print("          before. But it is going to run more securely.'                           -- sovereign AI [T]")
    print("[repo]   'Without attribution there is no way to compute unit economics.' Showback")
    print("          before chargeback, and chargeback only once the tags are trustworthy.         [R]")

    E.run_all()

    print("\n" + "=" * 78)
    print("THE ONE-PARAGRAPH SUMMARY")
    print("=" * 78)
    print("""
Cost and sovereignty are the same question asked twice -- who controls the layer that turns my data
into value -- and the corpus states both halves of it in one sentence each: "optimization without
measurement is just guessing" [T], and "you can't legislate a memory dump" [T]. Both halves turn out
to be settled by the same two things, a UNIT and an EVIDENCE TYPE, and both of those are chosen
before any measurement happens, which is why they are the two decisions this blueprint is built
around.

The cost half is a sequence of units that keep getting confused for one another, and the confusion
always runs the same way -- toward the flattering number. A discount is a PRICE and a saving is an
AMOUNT: a 10x cache read cuts the prefix line by 74.6% and the bill by 23.2%, the curve's asymptote
is 26.7% because the uncached share still pays full price, and a prefix that is never read twice is a
bill INCREASE at a 1.25x write premium (break-even 1.39 reads). The ladder's claimed discounts order
the levers differently from the savings they can bank -- distillation is #1 by claim (40x) and #8 by
bankable saving (2.4 points), prompt caching is #3 and #1 -- and eight of ten levers move. Tested
honestly, with every lever at its full claimed discount taking disjoint shares, the ladder's ceiling
is 60.6%, a 2.54x reduction against a headline of 90%: a 29.4-point gap that is a missing ARGUMENT,
not a missing technique. Three defensible orderings of those same levers differ by 4.2 points of the
bill, and the worst of them is the natural one, because a pricing lever applies to what is left and
so there is exactly one pricing slot for ease to get wrong.

The unit itself decides the answer, and the corpus's own figures are the cleanest demonstration in
this knowledge base: 30c to 16c is a 46.7% saving per artifact, a 21.5% INCREASE per file and a
269.2% increase per line of delivered work, because the deliverable shrank faster than the price did.
The denominator is the decision, and the team being measured usually picks it -- which is why "tokens
per output" is not merely uninformative but systematically optimistic, scoring a call with 300 visible
and 2,000 thinking tokens at 1.0x while it costs 7.7x. On top of that sits a mix whose average is the
wrong basis for an order: the expensive population is 1% of calls and 29.2% of the bill today, 29.2x
the mean call, and one growth cycle at the scenario's own 4.3x makes it 63.9% of the bill.

The sovereignty half is a vector with prerequisites, and its two readings disagree in the direction
that matters. A mean says four dimensions are strong; a FLOOR says the system is only as strong as
the one an adversary or a regulator gets to choose, and three of the four continuity examples score a
mean of 0.75 and a floor of zero. Only two numbers here are arithmetic rather than judgement, and
both are usually got wrong: a 20% throughput penalty for confidential computing is a 25% cost uplift,
because capacity is what you buy; and a residency rule has no error tolerance at all -- a score-based
router leaks 15.9% of requests at a modest gap and noise (1,587 per 10,000), a filter leaks zero, and
99.84% correct residency is not 99.84% compliant. Residency is a filter. So is a run cap: drawn from
the tail it catches the runaway, drawn from the mean it terminates legitimate work.

What ties the two halves together is that both end in a measurement that degrades in silence. Cost
coverage is the clearest: 90.5% of spend tagged, and one growth cycle takes it to 82.4% -- not
because anyone tagged worse, but because the untagged spend is concentrated in the team growing 6.5x.
Untagged spend is not a random sample of the bill. The uncovered 17.6% is the part about to matter,
and no dashboard says so unless the coverage figure is printed next to the growth rate.
""")

    print("=" * 78)
    print("Six decisions, one sentence each:")
    print("=" * 78)
    print("  1. A discount is a price and a saving is an amount: a 10x cache read banks 23.2% of the")
    print("     bill against a 74.6% line reduction, the asymptote is 26.7%, and a prefix read once is")
    print("     a bill increase (break-even 1.39 reads at a 1.25x write premium). Compute the")
    print("     read-to-write ratio per prefix and disable caching per prefix, not globally.")
    print("  2. 'Roughly a tenth of the cost' is a 10x claim; the ladder's honest ceiling is 60.6%")
    print("     (2.54x), a 29.4-point gap, and the most aggressive lever in it is worth 2.0 points of")
    print("     that ceiling at its stated reach. Rank levers by `discount x share_of_bill` --")
    print("     distillation is #1 by claim and #8 by saving, prompt caching #3 and #1, eight move.")
    print("  3. The ordering rule is worth 4.2 points of the bill, and the natural rule -- easy things")
    print("     first -- is the worst, because there is one pricing slot and ease puts batch_lane in it")
    print("     (15.8 points of reserved_capacity lost to the conflict). Sort reduction levers by ease;")
    print("     decide the pricing lever once, on value.")
    print("  4. The unit decides the sign. 30c to 16c is -46.7% per artifact, +21.5% per file, +269.2%")
    print("     per line delivered. Pick the denominator in writing before the programme starts, and")
    print("     pick one the team cannot shrink by doing less work.")
    print("  5. Sovereignty is a vector with prerequisites and a FLOOR: 20% throughput penalty = 25%")
    print("     cost uplift, a score-based residency router leaks 15.9% (1,587 per 10,000) where a")
    print("     filter leaks zero -- and residency at 99.84% is non-compliant, not 99.84% compliant.")
    print("  6. Attribution is the precondition and the only metric that degrades as the business")
    print("     succeeds: 90.5% coverage becomes 82.4% under one growth cycle, and untagged spend")
    print("     doubles to 17.6%. Print coverage beside growth rate, and instrument the gateway-")
    print("     invoice residual before enabling chargeback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
