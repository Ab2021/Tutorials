"""The experiments run.py executes.

Every count printed here is this simulation's own, over a toy vocabulary of
~2,058 tokens. The corpus's figures -- ~10 legal tokens of ~100,000, the
top-200 FUDGE truncation, the 2x contrastive cost -- are quoted and attributed
in HLD.md section 9 and are NOT reproduced by this model.
"""

import math

from . import automata, masking, healing, fudge
from .automata import VID, VOCAB

FIELDS = [("\"name\"", ["Taylor", "Swift"]), ("\"birth_year\"", ["1989", "2001"])]

# A plausible next-token distribution. Structural tokens get real mass; the
# filler tokens share the rest, which is what makes the legal set sparse.
_STRUCT_MASS = {
    "{": 0.25, "}": 0.10, ":": 0.12, ",": 0.08,
    "\"name\"": 0.06, "\"birth_year\"": 0.05,
    "Taylor": 0.05, "Swift": 0.04, "1989": 0.04, "2001": 0.03,
}


def base_logits():
    """A logit vector over VOCAB: structural tokens favoured, filler flat."""
    logits = [0.0] * len(VOCAB)
    for tok, p in _STRUCT_MASS.items():
        logits[VID[tok]] = math.log(p)
    rest = 1.0 - sum(_STRUCT_MASS.values())
    per = rest / float(automata.N_FILLER)
    for i, tok in enumerate(VOCAB):
        if tok.startswith("f"):
            logits[i] = math.log(per)
    return logits


# ------------------------------------------------------------------ 1

def exp_compile_and_mask():
    """Compile the schema, walk one valid record, and count survivors per step.

    The sparsity claim: at a typical step the legal set is a handful of tokens
    against a vocabulary of thousands.
    """
    dfa, accept = automata.json_schema_fsa(FIELDS)
    logits = base_logits()

    record = ["{", "\"name\"", ":", "Taylor", ",", "\"birth_year\"", ":",
              "1989", "}"]
    state = dfa.start
    rows = []
    for tok in record:
        legal = dfa.legal_symbols(state, VOCAB)
        legal_ids = [VID[t] for t in legal]
        masked = masking.masked_logits(logits, legal_ids)
        probs = masking.softmax(masked)
        surv = masking.survivors(probs)
        rows.append({
            "state": state,
            "token": tok,
            "legal": legal,
            "survivors": len(surv),
            "vocab": len(VOCAB),
        })
        nxt = dfa.step(state, tok)
        if nxt is None:
            raise AssertionError("fixture record is not in the grammar")
        state = nxt

    return {
        "rows": rows,
        "accept": accept,
        "final_state": state,
        "accepted": dfa.accepts(record),
        "vocab": len(VOCAB),
        "min_survivors": min(r["survivors"] for r in rows),
        "n_states": dfa.n_states,
    }


# ------------------------------------------------------------------ 2

def exp_empty_candidate_set():
    """A grammar bug: a state whose legal set is empty.

    This must be caught by a unit test, never by a zero-mass step at runtime.
    The audit finds it before the request does.
    """
    # A deliberately broken schema: the value list for the second field is
    # empty, so the state after its ':' has no legal token at all.
    broken = [("\"name\"", ["Taylor"]), ("\"birth_year\"", [])]
    dfa, accept = automata.json_schema_fsa(broken)

    audit = masking.audit_grammar(dfa, VOCAB, {accept})

    # And the runtime symptom, for contrast: mask everything and softmax.
    logits = base_logits()
    state = dfa.start
    for tok in ["{", "\"name\"", ":", "Taylor", ",", "\"birth_year\"", ":"]:
        state = dfa.step(state, tok)
    legal = dfa.legal_symbols(state, VOCAB)
    probs = masking.softmax(
        masking.masked_logits(logits, [VID[t] for t in legal]))
    runtime_survivors = masking.survivors(probs)

    return {
        "audit": audit,
        "dead_state_legal": legal,
        "runtime_survivors": len(runtime_survivors),
        "runtime_max_prob": max(probs) if probs else 0.0,
    }


# ------------------------------------------------------------------ 3

def exp_fsa_vs_pda():
    """The nesting boundary: what a DFA cannot do at any size.

    A fixed depth can be flattened -- one pair of states per depth. Beyond
    that the automaton would need infinitely many states, which is exactly
    what a pushdown automaton's stack provides instead.
    """
    rows = []
    for depth in (3, 4, 8):
        fsa = automata.bracket_fsa(max_depth=depth)
        pda = automata.CounterPDA()
        deep_ok = "{" * depth + "}" * depth
        too_deep = "{" * (depth + 1) + "}" * (depth + 1)
        rows.append({
            "declared_depth": depth,
            "fsa_states": fsa.n_states,
            "pda_states": pda.n_states,
            "accepts_within": fsa.accepts(deep_ok),
            "accepts_beyond": fsa.accepts(too_deep),
            "pda_accepts_beyond": pda.accepts(too_deep),
        })
    return rows


# ------------------------------------------------------------------ 4

def exp_optional_field_blowup():
    """Flattening optionality into states costs combinatorially.

    Every field after the first doubles the state count, because the automaton
    must distinguish which subset of fields remains. That is the practical
    reason a real JSON-schema engine ships a pushdown automaton.
    """
    rows = []
    prev = None
    for n_optional in (1, 2, 3, 4, 5):
        fields = [("\"name\"", ["Taylor"])] + [
            ("\"f%d\"" % i, ["v%d" % i]) for i in range(n_optional)
        ]
        dfa, _ = automata.json_schema_fsa_optional(fields)
        rows.append({
            "optional_fields": n_optional,
            "states": dfa.n_states,
            "ratio": None if prev is None else dfa.n_states / float(prev),
        })
        prev = dfa.n_states
    return rows


# ------------------------------------------------------------------ 5

def exp_token_healing():
    """Healing changes the tokens, not the string.

    The model wants to emit `://` as one token because that is how URLs were
    tokenized in pre-training, but the grammar is at a state that requires a
    `:` next. Masking leaves `:`; healing finds the multi-character tokens
    that begin with it.
    """
    vocab = ["?", ":", ":/", "://", "/", "abc", "a", "ab"]
    ids = {t: i for i, t in enumerate(vocab)}
    probs = [0.01, 0.02, 0.05, 0.62, 0.20, 0.05, 0.03, 0.02]

    legal = [ids[":"]]                       # the grammar's requirement
    healed = healing.heal(legal, ":", probs, vocab)

    # The surface form the two routes produce, given the rest of the string.
    naive_surface = healing.surface_form([":"]) + "/" + "/"
    healed_surface = healing.surface_form([vocab[healed["chosen"]]])

    triggers = {
        "colon": healing.trigger_fired(":", vocab, [":", " ", "/"]),
        "word": healing.trigger_fired("abc", vocab, [":", " ", "/"]),
        "slash": healing.trigger_fired("/", vocab, [":", " ", "/"]),
    }

    return {
        "legal_set": [vocab[i] for i in legal],
        "healed_candidates": [vocab[i]
                              for i in healing.candidates_starting_with(vocab, ":")],
        "healed": healed,
        "naive_surface": naive_surface,
        "healed_surface": healed_surface,
        "same_surface": naive_surface == healed_surface,
        "prob_gain": healed["chosen_prob"] - probs[ids[":"]],
        "triggers": triggers,
    }


# ------------------------------------------------------------------ 6

def exp_healing_cost():
    """What the "it's just expensive" caveat costs at a real trigger rate."""
    rows = []
    for rate in (0.0, 0.02, 0.10, 1.0):
        row = healing.healing_cost(rate, tokens_per_record=400)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ 7

def exp_fudge():
    """A semantic constraint reorders the candidates -- within the top k.

    The winner changes because the LM probabilities were close and the
    constraint separated them. The blind spot is measured, not assumed.
    """
    vocab = ["do", "you", "want", "prefer", "thus", "please", "kindly"]
    probs = [0.30, 0.25, 0.20, 0.18, 0.05, 0.012, 0.008]
    formality = [0.2, 0.2, 0.35, 0.85, 0.95, 0.9, 0.92]

    out = fudge.reward_augmented(probs, formality, k=len(vocab))
    ranked = sorted(range(len(vocab)), key=lambda i: -out["posterior"][i])

    # The blind spot: a candidate the constraint loves but the LM ranks low.
    blind = fudge.truncated_reward_mass(probs, formality, k=3)
    blind_winner = fudge.reward_augmented(probs, formality, k=3)

    return {
        "vocab": vocab,
        "unconstrained": vocab[out["unconstrained_winner"]],
        "constrained": vocab[out["winner"]],
        "ranking": [(vocab[i], round(out["posterior"][i], 4)) for i in ranked],
        "k3_winner": vocab[blind_winner["winner"]],
        "k3_truncated_out": (
            vocab[blind_winner["truncated_out"]]
            if blind_winner["truncated_out"] is not None else None),
        "blind": blind,
        "k_full": len(vocab),
    }


# ------------------------------------------------------------------ 8

def exp_contrastive():
    """Expert minus amateur, and the flat 2x it costs."""
    vocab = ["Honolulu", "Hawaii", "the", "born", "1961", "Kenya"]
    expert = [0.30, 0.28, 0.15, 0.12, 0.10, 0.05]
    amateur = [0.34, 0.33, 0.16, 0.10, 0.04, 0.03]
    adjusted = fudge.contrastive(
        [math.log(p) for p in expert], [math.log(p) for p in amateur])
    ranked = sorted(range(len(vocab)), key=lambda i: -adjusted[i])
    return {
        "vocab": vocab,
        "expert_top": vocab[max(range(len(vocab)), key=lambda i: expert[i])],
        "adjusted_top": vocab[ranked[0]],
        "ranking": [(vocab[i], round(adjusted[i], 4)) for i in ranked[:4]],
        "compute_multiplier": 2.0,
    }
