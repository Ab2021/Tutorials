"""Semantic constraints over a truncated candidate set.

The corpus's mechanism: sample proportional to

    P(constraint satisfied | prefix so far) * P(token)

"we're going to softmax everything anyway. So, we don't care about sort of
exact value". The affordability comes from truncation -- "they take the top
200 most likely next tokens um and you run on just those 200 instead of all
100,000" -- plus a small discriminator, "because this sort of zero one choice
is a relatively easy thing to learn".

Two properties this module exists to make testable:

  * the winner can change under the constraint -- the corpus's worked example
    is "do you want" vs "do you prefer", where "prefer" wins because "want and
    prefer were relatively even probability but prefer is more formal so it
    gets updated";
  * truncation hides things. A candidate outside the top-k can never win, no
    matter how strongly the constraint favours it. That is a real limitation,
    not a bug, and it is measured here rather than assumed away.
"""

import math


def top_k_ids(probs, k):
    """The truncation. Returns ids, best first."""
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    return order[:k]


def reward_augmented(probs, discriminator_scores, k):
    """Multiply in the discriminator over the truncated set, then renormalise.

    Both inputs are probabilities, so the product is the joint score the
    corpus describes. Renormalising is what makes the result a distribution
    again -- and is why the exact value of the discriminator does not matter,
    only its ordering.
    """
    ids = top_k_ids(probs, k)
    joint = {i: probs[i] * discriminator_scores[i] for i in ids}
    total = sum(joint.values())
    if total == 0.0:
        return {"ids": ids, "posterior": {i: 0.0 for i in ids},
                "winner": None, "truncated_out": None}
    posterior = {i: v / total for i, v in joint.items()}
    winner = max(ids, key=lambda i: (posterior[i], -i))

    # What the constraint would have picked over the FULL vocabulary. The gap
    # between this and `winner` is the truncation's blind spot.
    full_best = max(range(len(probs)),
                    key=lambda i: (probs[i] * discriminator_scores[i], -i))
    return {
        "ids": ids,
        "posterior": posterior,
        "winner": winner,
        "winner_prob": posterior[winner],
        "unconstrained_winner": max(ids, key=lambda i: (probs[i], -i)),
        "truncated_out": None if full_best in ids else full_best,
    }


def truncated_reward_mass(probs, discriminator_scores, k):
    """How much constrained mass the truncation discards.

    This is the honest cost of the top-k trick: any candidate the constraint
    would have loved but the LM ranked below k is unreachable.
    """
    ids = set(top_k_ids(probs, k))
    total = sum(probs[i] * discriminator_scores[i] for i in range(len(probs)))
    kept = sum(probs[i] * discriminator_scores[i] for i in ids)
    return {
        "k": k,
        "kept": kept,
        "total": total,
        "discarded_fraction": 0.0 if total == 0 else 1.0 - kept / total,
    }


def contrastive(logits_expert, logits_amateur, alpha=1.0):
    """Expert minus amateur, in log space.

    The corpus's prerequisite is stated flatly: "these models need to have the
    same tokenizer for this to work". The mechanism is subtraction -- "we look
    at the output logits and then we subtract them from each other" -- and the
    cost is a flat 2x because "if you're passing these both through the model,
    you need to use twice as much compute to accomplish the same task".
    """
    adjusted = [e - alpha * a for e, a in zip(logits_expert, logits_amateur)]
    m = max(adjusted)
    exps = [math.exp(v - m) for v in adjusted]
    total = sum(exps)
    return [e / total for e in exps]
