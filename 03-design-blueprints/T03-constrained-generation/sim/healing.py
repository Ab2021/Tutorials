"""Token healing: roll back, require the next token to start with the prefix.

The corpus's procedure: "we'll roll back a token or very rarely to [two] of
generation and we'll just require that the next token starts with that token
that we would have predicted before… we'll look at all of our candidates for
the next token and we'll eliminate everything that doesn't start with colon.
So colon is still a valid next token, but so is colon slash, which has a lot
higher probability".

The invariant that makes it safe is that the SURFACE FORM does not change:
"we're not actually changing our output string. We're just changing the tokens
of the output to get there". Everything here is built around asserting that.
"""


def candidates_starting_with(vocab, prefix):
    """The healed candidate set: every token whose text begins with `prefix`."""
    return [i for i, tok in enumerate(vocab) if tok.startswith(prefix)]


def heal(candidates, prefix, probs, vocab):
    """Pick the best continuation among tokens starting with `prefix`.

    Returns the chosen token, whether healing actually changed anything, and
    the surface form produced. `changed` is False when the original predicted
    token was already the argmax of the healed set -- in which case healing
    cost a recompute and bought nothing, which is the reason it is gated on
    heuristics rather than run every step.
    """
    healed_ids = [i for i in candidates_starting_with(vocab, prefix)
                  if probs[i] > 0.0]
    if not healed_ids:
        return {"chosen": None, "changed": False, "error": "no token starts with prefix"}

    best = max(healed_ids, key=lambda i: (probs[i], -i))
    naive = max(candidates, key=lambda i: (probs[i], -i)) if candidates else None
    return {
        "chosen": best,
        "chosen_text": vocab[best],
        "chosen_prob": probs[best],
        "naive": naive,
        "naive_text": vocab[naive] if naive is not None else None,
        "naive_prob": probs[naive] if naive is not None else 0.0,
        "changed": best != naive,
        # Healing is only legitimate if this is True.
        "surface_preserved": vocab[best].startswith(prefix),
        "error": None,
    }


def surface_form(tokens):
    """The string the user sees. Healing must not change this."""
    return "".join(tokens)


def trigger_fired(prev_token_text, vocab, offender_list):
    """The corpus's four trigger heuristics, as one predicate.

    "when the last token is very short" or single-character; "when the last
    token doesn't start or end with whitespace or punctuation"; "when the last
    token predicted was an exact prefix of another likely token"; plus a
    curated list of common offenders. The corpus notes there is no single
    accepted rule, so this list is versioned and evaluated rather than assumed.
    """
    reasons = []
    if prev_token_text in offender_list:
        reasons.append("offender_list")
    if len(prev_token_text) <= 1:
        reasons.append("very_short")
    if prev_token_text and prev_token_text[0].strip() and prev_token_text[-1].strip():
        reasons.append("no_boundary_whitespace")
    if any(tok != prev_token_text and tok.startswith(prev_token_text)
           for tok in vocab):
        reasons.append("prefix_of_another_token")
    return reasons


def healing_cost(trigger_rate, tokens_per_record, added_forward_passes=1):
    """Healing costs one extra forward pass per triggered step.

    The corpus's caveat -- "it's just expensive. you have to go back and
    recompute" -- is a statement about healing EVERY step. This computes what
    it costs at a given trigger rate, which is the number that decides whether
    the caveat applies.
    """
    triggered = tokens_per_record * trigger_rate
    return {
        "trigger_rate": trigger_rate,
        "triggered_steps": triggered,
        "extra_passes": triggered * added_forward_passes,
        "overhead_fraction": (
            triggered * added_forward_passes) / float(tokens_per_record),
    }
