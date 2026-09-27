#!/usr/bin/env python3
"""T03 -- Constrained Generation. Runnable core.

Proves the design's central mechanisms:

  * a schema compiles to an automaton, and the automaton's legal set at each
    step is the mask -- minus infinity before softmax, then renormalise;
  * the legal set is orders of magnitude smaller than the vocabulary, which is
    why a prompt cannot reliably produce it;
  * a state with no legal token is a grammar bug that a unit test must catch,
    never a zero-mass step at runtime;
  * a DFA cannot express arbitrary nesting at any size, which is why a real
    JSON-schema engine is a pushdown automaton and not an FSA;
  * token healing changes the tokens and not the string;
  * a semantic constraint can only reorder the candidates it can see, and the
    top-k truncation is a measured blind spot.

WHAT THIS IS NOT: a benchmark. The vocabulary is a toy of ~2,058 tokens and
every count below is this simulation's own. The corpus's figures (~10 legal
tokens of ~100,000, the top-200 truncation, the 2x contrastive cost) are
quoted and attributed in HLD.md section 9 and are NOT reproduced here.

Stdlib only. Offline. Exits 0.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import experiments as ex  # noqa: E402

RULE = "=" * 78


def header(title):
    print()
    print(RULE)
    print(title)
    print(RULE)


def main():
    print(RULE)
    print("T03 Constrained Generation -- mask, compile, heal")
    print(RULE)
    print("This is a MODEL OF A MECHANISM, not a measurement.")
    print("Toy vocabulary: %d tokens (%d structural, %d filler)." % (
        ex.automata.VOCAB.__len__(), len(ex.automata.STRUCTURAL),
        ex.automata.N_FILLER))
    print("Schema: a flat two-field record.")

    header("1. Schema -> DFA -> mask, walked one token at a time")
    r = ex.exp_compile_and_mask()
    print("Compiled DFA: %d states, accept state %d." % (
        r["n_states"], r["accept"]))
    print()
    print("%-7s %-14s %-34s %-10s" % ("state", "emits", "legal next tokens",
                                      "survivors"))
    for row in r["rows"]:
        legal = ",".join(row["legal"])
        if len(legal) > 32:
            legal = legal[:29] + "..."
        print("%-7d %-14s %-34s %d of %d" % (
            row["state"], row["token"], legal, row["survivors"], row["vocab"]))
    print()
    print("Record accepted by the automaton: %s" % r["accepted"])
    print("Smallest legal set at any step: %d of %d tokens." % (
        r["min_survivors"], r["vocab"]))
    print()
    print("The mask is the legal set. Everything outside it is set to minus")
    print("infinity before softmax and the distribution is renormalised. The")
    print("sparsity is the whole point: a step where a handful of tokens out")
    print("of thousands are legal is not something a prompt reliably")
    print("produces, and a mask makes it structural.")

    header("2. The empty candidate set -- a grammar bug, caught early")
    e = ex.exp_empty_candidate_set()
    a = e["audit"]
    print("Grammar audit over %d reachable states:" % a["n_reachable"])
    print("  dead ends (no legal token) : %s" % (a["dead_ends"] or "none"))
    print("  stranded (cannot reach end): %s" % (a["stranded"] or "none"))
    print("  audit verdict              : %s" % ("PASS" if a["ok"] else "FAIL"))
    print()
    print("The same state at runtime, if the audit had not run:")
    print("  legal tokens    : %s" % (e["dead_state_legal"] or "NONE"))
    print("  survivors       : %d" % e["runtime_survivors"])
    print("  max probability : %.4f" % e["runtime_max_prob"])
    print()
    print("A zero-survivor step must never proceed silently. The runbook's")
    print("per-state assertion -- every reachable state has at least one")
    print("legal token, and every state can still reach an accept state --")
    print("is what turns this from a production incident into a build failure.")

    header("3. The nesting boundary: what a DFA cannot do at any size")
    print("A fixed depth flattens into states -- one state per depth. Beyond")
    print("that the automaton would need infinitely many states.")
    print()
    print("%-8s %-12s %-12s %-14s %-14s %s" % (
        "declared", "FSA states", "PDA states", "accepts <= d",
        "accepts d+1", "PDA accepts"))
    for row in ex.exp_fsa_vs_pda():
        print("%-8d %-12d %-12d %-14s %-14s %s" % (
            row["declared_depth"], row["fsa_states"], row["pda_states"],
            "yes" if row["accepts_within"] else "NO",
            "yes" if row["accepts_beyond"] else "NO",
            "yes" if row["pda_accepts_beyond"] else "NO"))
    print()
    print("The FSA is correct up to the depth it was compiled for and silently")
    print("wrong past it. The one-state counter machine has no depth limit")
    print("because its memory is a stack rather than a state set. This is why")
    print("'anything that supports JSON schemas is actually writing push down")
    print("automa to enforce its constraints, not FSAs' [T].")

    header("4. Flattening optional fields costs combinatorially")
    print("%-18s %-10s %-14s %s" % (
        "optional fields", "states", "vs previous", "note"))
    for row in ex.exp_optional_field_blowup():
        ratio = "-" if row["ratio"] is None else "%.2fx" % row["ratio"]
        print("%-18d %-10d %-14s %s" % (
            row["optional_fields"], row["states"], ratio,
            "approaching 2x per field"))
    print()
    print("Each optional field roughly doubles the state set, because the")
    print("automaton must distinguish which subset of fields remains. The")
    print("ratio is above 2 at these sizes and settles toward it: the state")
    print("count is dominated by the 2^n remaining-subsets term. Optionality")
    print("is not the only multiplier -- nesting is the other -- and both are")
    print("reasons a pushdown automaton is the right target.")

    header("5. Token healing: different tokens, same string")
    h = ex.exp_token_healing()
    print("Grammar requires a ':' next; the model wants to emit '://' as one")
    print("token, because that is how URLs were tokenized in pre-training.")
    print()
    print("  legal set under the mask : %s" % ", ".join(h["legal_set"]))
    print("  healed candidates        : %s" % ", ".join(h["healed_candidates"]))
    print("  mask route first token   : %-6s (p=%.2f)" % (
        h["healed"]["naive_text"], h["healed"]["naive_prob"]))
    print("  healing route first token: %-6s (p=%.2f)" % (
        h["healed"]["chosen_text"], h["healed"]["chosen_prob"]))
    print("  probability gain         : +%.2f" % h["prob_gain"])
    print()
    print("  surface via mask   : %r" % h["naive_surface"])
    print("  surface via healing: %r" % h["healed_surface"])
    print("  surface preserved  : %s" % h["same_surface"])
    print()
    print("Healing changes the tokens of the output, never the output string.")
    print("That invariant is what the test asserts: heal, detokenise, compare.")
    print()
    print("Triggers (versioned heuristics, not universal rules):")
    for name, reasons in h["triggers"].items():
        print("  %-6s -> %s" % (name, ", ".join(reasons) or "no trigger"))

    header("6. What the 'it's just expensive' caveat actually costs")
    print("%-14s %-16s %-14s %s" % (
        "trigger rate", "triggered steps", "extra passes", "overhead"))
    for row in ex.exp_healing_cost():
        print("%-14.2f %-16.1f %-14.1f %.0f%%" % (
            row["trigger_rate"], row["triggered_steps"], row["extra_passes"],
            row["overhead_fraction"] * 100))
    print()
    print("Over a 400-token record. Healing every step is a 100% overhead; at")
    print("a 2% trigger rate it is 2%. That is the number that decides whether")
    print("the caveat applies, and it is why healing is heuristic-gated rather")
    print("than always-on.")

    header("7. Semantic constraints: the top-k trick and its blind spot")
    f = ex.exp_fudge()
    print("Candidate set: %s" % ", ".join(f["vocab"]))
    print()
    print("  unconstrained winner : %s" % f["unconstrained"])
    print("  constrained winner   : %s" % f["constrained"])
    print()
    print("  posterior ranking:")
    for tok, p in f["ranking"]:
        print("    %-10s %.4f" % (tok, p))
    print()
    print("The winner moves because the two candidates were close in language")
    print("model probability and the constraint separated them -- the corpus's")
    print("'want and prefer were relatively even probability but prefer is")
    print("more formal so it gets updated' [T].")
    print()
    print("Truncating to the top 3 instead of the full set:")
    print("  winner        : %s" % f["k3_winner"])
    print("  truncated out : %s" % (f["k3_truncated_out"] or "nothing"))
    print("  constrained mass discarded: %.1f%%" % (
        f["blind"]["discarded_fraction"] * 100))
    print()
    print("Truncation is what makes this affordable and it is also a blind")
    print("spot: a candidate the constraint would love but the language model")
    print("ranked below k is unreachable. The full-set run above truncates at")
    print("nothing and picks %s; the top-3 run cannot see it." % f["constrained"])

    header("8. Contrastive decoding: subtract the amateur")
    c = ex.exp_contrastive()
    print("Expert top token : %s" % c["expert_top"])
    print("After subtraction: %s" % c["adjusted_top"])
    print()
    print("  adjusted ranking:")
    for tok, p in c["ranking"]:
        print("    %-10s %.4f" % (tok, p))
    print()
    print("The amateur model repeats the memorised pair; subtracting its")
    print("logits downweights exactly that and surfaces what the amateur did")
    print("not know. The cost is flat: %.0fx compute [T]." % (
        c["compute_multiplier"]))

    header("Summary")
    print("  * the mask is the automaton's legal set: -inf before softmax")
    print("  * the legal set is tiny, which is why masking beats prompting")
    print("  * an empty legal set is a grammar bug; assert it at build time")
    print("  * fixed nesting flattens into a DFA and explodes; arbitrary")
    print("    nesting needs a pushdown automaton, which is what real")
    print("    schema engines actually are")
    print("  * healing changes tokens, not the string -- assert the surface")
    print("  * the mask guarantees syntax; only the validator guarantees the")
    print("    schema version in force, and it is not redundant with the mask")
    print()
    print("OK -- T03 simulation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
