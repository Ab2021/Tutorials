"""Where the draft tokens come from -- and what each source costs.

The speedup formula has two inputs: the acceptance profile (`alpha`) and the draft cost `c` as a
fraction of one target forward pass. Most discussion of speculative decoding is about `alpha`.
This module is about `c`, because `c` is what decides WHICH variant is worth deploying, and it
varies by more than an order of magnitude across the options:

    draft model       c ~ params_draft / params_target       (a real second forward pass)
    MTP heads         c ~ small, but the heads must be trained in
    Medusa/EAGLE      c ~ small; a tree, so a better alpha at the same verify cost
    n-gram / lookup   c ~ 0 -- string matching, no forward pass at all

A drafter with `c = 0` is a different kind of object from one with `c = 0.1`: the first is free
whenever it is right and costs nothing when it is wrong.
"""
from __future__ import annotations

import math

from .acceptance import decay_profile, tokens_per_step


# --------------------------------------------------------------------------------------
# Draft cost models
# --------------------------------------------------------------------------------------

def draft_model_cost(draft_params_b: float, target_params_b: float) -> float:
    """`c` for a separate draft model: roughly the parameter ratio.

    A forward pass of a smaller model is cheaper in proportion to its size, so a 1B draft against
    a 70B target is `c ~ 0.014`. THE CATCH IS MEMORY, not compute: the draft's weights and its
    own KV cache occupy HBM that would otherwise hold target KV, which lowers the concurrency
    ceiling (T07). `c` understates the real cost of this variant -- the cost that bites is
    capacity, not latency.
    """
    if target_params_b <= 0 or draft_params_b <= 0:
        raise ValueError("parameter counts must be positive")
    return draft_params_b / target_params_b


def mtp_cost(n_heads: int, head_params_ratio: float = 0.01) -> float:
    """`c` for multi-token-prediction heads trained into the target.

    The heads are small linear layers on the final hidden state, so the marginal forward cost is
    small and NO second model is resident. This is why MTP is the production path rather than a
    research curiosity -- and it is the corpus's own reported configuration: MTP "enabled more
    interactivity which gained about 2x improvement in throughput" [T] (llm-d).
    """
    if n_heads < 0:
        raise ValueError("n_heads must be >= 0")
    return n_heads * head_params_ratio


def ngram_cost() -> float:
    """`c` for prompt-lookup / n-gram drafting: ZERO forward passes.

    The proposal is produced by string matching against the prompt. There is no model, no weights,
    no KV -- and, crucially, no cost when the match is wrong. A drafter that is free when it fails
    has a fundamentally better cost profile than one that is merely cheap.
    """
    return 0.0


def medusa_tree_cost(n_candidates: int, head_params_ratio: float = 0.01,
                     verify_slack: float = 0.15) -> float:
    """`c` for a tree drafter (Medusa / EAGLE).

    Multiple candidate continuations per position, verified together under a tree attention mask.
    The draft cost stays near-linear in candidates, but the VERIFY pass grows sub-linearly because
    the tree shares a prefix: hence `verify_slack` rather than a full multiple.

    The reason trees exist is not cost, it is `alpha`: a linear draft commits to one branch, so a
    single early rejection discards every later token. A tree keeps `n_candidates` branches alive,
    which raises the expected number of accepted tokens for the same verify pass.
    """
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    return n_candidates * head_params_ratio * (1.0 + verify_slack)


# --------------------------------------------------------------------------------------
# Acceptance profiles by drafter
# --------------------------------------------------------------------------------------

def ngram_acceptance(p_target_of_copied_token: float) -> float:
    """Acceptance for an n-gram drafter, and it has a closed form worth understanding.

    A prompt-lookup drafter proposes a token DETERMINISTICALLY -- it is a copy, so `p_draft = 1`
    on that token and 0 elsewhere. The acceptance rule `min(1, p_target / p_draft)` therefore
    collapses to

        alpha = p_target(copied token)

    THE ACCEPTANCE RATE IS EXACTLY THE MODEL'S OWN PROBABILITY OF THE COPIED TOKEN. There is no
    draft model to be "aligned" with, no distribution to drift, and nothing to retrain. It is a
    direct read of how much the model was going to copy the input anyway.

    That makes the when-to-use decision decidable in advance, from the workload rather than from a
    benchmark:
        summarise / extract / edit / RAG-grounded  ->  high p(copy)  ->  alpha high  ->  use it
        creative / open-ended                       ->  low  p(copy)  ->  alpha low   ->  do not
    """
    if not 0.0 <= p_target_of_copied_token <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return p_target_of_copied_token


def drift_penalty(base_alpha: float, tokenizer_mismatch: float = 0.0,
                  draft_staleness: float = 0.0) -> float:
    """Acceptance after two failure modes that are easy to cause and hard to see.

    `tokenizer_mismatch` -- a draft from a different model family has a different tokenizer, so its
    tokens do not even refer to the same strings. This is NOT a degradation: it is a correctness
    failure, and the acceptance rate collapses toward chance. Check tokenizer identity before
    anything else.

    `draft_staleness` -- the draft was fine-tuned on data that has drifted from the target's
    behaviour, or the target was updated and the draft was not. Acceptance falls silently; the
    only symptom is a speedup that is lower than the benchmark predicted.
    """
    if not 0.0 <= base_alpha <= 1.0:
        raise ValueError("base_alpha must be in [0, 1]")
    if tokenizer_mismatch > 0.5:
        # A different tokenizer is not a partial degradation; treat it as unusable.
        return min(base_alpha, 0.05)
    return max(0.0, base_alpha * (1.0 - tokenizer_mismatch) * (1.0 - draft_staleness))


def tree_expected_accepts(alphas: list[float], n_candidates: int) -> float:
    """Expected accepted tokens for a TREE draft with `n_candidates` branches per position.

    With a single branch the expected accepts are `sum_i prod_{j<=i} alpha_j` -- one early
    rejection truncates everything after it. With `m` candidates at each position, position `i`
    survives if ANY of its candidates is accepted, so the per-position factor becomes
    `1 - (1 - alpha_i)^m`.

    Two things this shows, both of which look wrong at first:
      * the gain from a tree is NOT multiplicative and is largest at SHALLOW depth -- it rescues
        the positions a linear draft was throwing away;
      * at high `alpha` the tree buys almost nothing, because `1 - (1-a)^m -> 1` already. Trees
        pay off exactly where a linear draft hurts: moderate acceptance.
    """
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    total = 1.0
    run = 1.0
    for a in alphas:
        per_pos = 1.0 - (1.0 - a) ** n_candidates
        run *= per_pos
        total += run
    return total


def linear_expected_accepts(alphas: list[float]) -> float:
    """The single-branch case, stated explicitly for contrast with `tree_expected_accepts`."""
    return tokens_per_step(alphas)
