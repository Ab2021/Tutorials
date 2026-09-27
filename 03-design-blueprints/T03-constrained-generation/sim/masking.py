"""The mask itself: add a large negative before softmax, then renormalise.

The corpus's mechanism statement is exact: "all the things that are allowed as
the next token are zero, all the things that are not allowed to the next token
we add minus infinity and then we take the softmax again to renormalize".

Two properties this module exists to make testable:

  * sparsity -- the legal set at a step is orders of magnitude smaller than
    the vocabulary, which is why a prompt cannot reliably produce it and a
    mask can;
  * the empty candidate set -- a state with no legal token is a grammar bug
    that must be counted, never silently allowed to proceed.
"""

import math

NEG_INF = float("-inf")


def masked_logits(logits, allowed_ids):
    """Set every disallowed logit to -inf. Returns a new list."""
    allowed = set(allowed_ids)
    return [v if i in allowed else NEG_INF for i, v in enumerate(logits)]


def softmax(logits):
    """Softmax that tolerates -inf entries and an all--inf row."""
    finite = [v for v in logits if v != NEG_INF]
    if not finite:
        return [0.0] * len(logits)
    m = max(finite)
    exps = [0.0 if v == NEG_INF else math.exp(v - m) for v in logits]
    total = sum(exps)
    if total == 0.0:
        return [0.0] * len(logits)
    return [e / total for e in exps]


def survivors(probs, threshold=0.0):
    """Indices with non-zero probability -- the step's real candidate set."""
    return [i for i, p in enumerate(probs) if p > threshold]


def renormalisation_loss(raw_probs, masked_probs, allowed_ids):
    """How much probability mass the mask removed, and how it redistributed.

    If the model already placed all its mass on legal tokens the mask is free.
    If it did not, the mask moves mass -- which is the mechanism by which a
    bad grammar produces degenerate output instead of an error.
    """
    kept = sum(raw_probs[i] for i in allowed_ids)
    return {"mass_removed": 1.0 - kept, "mass_kept": kept}


def check_state_has_legal_token(dfa, state, alphabet):
    """Grammar unit test: every reachable state must have >= 1 legal token.

    This is the assertion the runbook requires per state, and the empty
    candidate set is what it prevents.
    """
    legal = dfa.legal_symbols(state, alphabet)
    return {"state": state, "legal": legal, "ok": len(legal) > 0}


def audit_grammar(dfa, alphabet, accept_states):
    """The two properties every compiled grammar must satisfy.

    1. No reachable state is a dead end (>= 1 legal token).
    2. Every reachable state can still reach an accept state.
    """
    reachable = dfa.live_states()

    dead_ends = [
        s for s in sorted(reachable)
        if s not in accept_states and not dfa.legal_symbols(s, alphabet)
    ]

    # Reverse reachability from the accept states.
    back = {s: set() for s in reachable}
    for s in reachable:
        for nxt in dfa.transitions.get(s, {}).values():
            if nxt in back:
                back[nxt].add(s)
    can_accept, stack = set(accept_states), list(accept_states)
    while stack:
        s = stack.pop()
        for p in back.get(s, ()):
            if p not in can_accept:
                can_accept.add(p)
                stack.append(p)
    stranded = [s for s in sorted(reachable) if s not in can_accept]

    return {
        "n_states": dfa.n_states,
        "n_reachable": len(reachable),
        "dead_ends": dead_ends,
        "stranded": stranded,
        "ok": not dead_ends and not stranded,
    }
