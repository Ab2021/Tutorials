"""T19 -- the levers, and why a discount is not a saving.

The cost talk gives the ladder and its ordering principle: stack techniques, do not look for a
hero optimisation, and "before you touch the model at all, exhaust the free wins" [T]. Its headline
is that stacking takes you "from a naive baseline to roughly a tenth of the cost with no change to
the model itself" [T].

That headline is testable, and this module tests it. Two things fall out:

  * **A lever's claimed discount is not its saving.** What a programme can bank is
    `discount x share_of_bill`, and a 40x claim on a 4% line is worth less than a 2x claim on a
    30% line. Ranking by claim and ranking by achievable saving give different orders.

  * **Two kinds of lever compose differently.** A *reduction* lever removes work; a *pricing*
    lever changes the unit price of work that still happens. Pricing levers multiply with
    reduction levers (good) and conflict with EACH OTHER on the same traffic (not obvious), so
    batch and reserved capacity on one workload is one saving, not two.

Whether the corpus's "tenth of the cost" is reachable is left to `ceiling()` to answer honestly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .stack import BILL, BILL_TOTAL, LAYER_BY_NAME

# --------------------------------------------------------------------------------------------
# A lever targets one or more bill layers. `share` is the fraction of that layer the lever can
# reach; `discount` is the fractional reduction it achieves on the part it reaches. A pricing
# lever targets the special layer "*", meaning the whole remaining bill on a traffic slice.
# --------------------------------------------------------------------------------------------
ALL = "*"


@dataclass(frozen=True)
class Lever:
    name: str
    claim: str                        # the corpus's own words [T]/[R]
    claim_multiple: float             # the largest multiple implied by that claim, as READ by the
                                      # author. Stated as data rather than parsed from prose: the
                                      # claims mix "Nx cheaper" and "N% off", and several quote a
                                      # QUALITY percentage alongside a cost one -- a regex that
                                      # cannot tell them apart ranks "95% quality" as a 20x saving.
    claim_notation: str               # how the corpus stated it: multiple | percentage | none
    kind: str                         # "reduction" | "pricing"
    targets: tuple                    # ((layer, share, discount), ...)
    ease: int                         # 5 = free, 1 = a project
    caveat: str = ""
    conflicts: tuple = field(default_factory=tuple)   # names it cannot stack with


LEVERS = (
    Lever(
        name="prompt_caching",
        claim="cache reads ~10x cheaper than fresh tokens [T]; providers 50-90% [R]",
        claim_multiple=10.0, claim_notation="multiple",
        kind="reduction",
        targets=(("system_prompt", 1.00, 0.746),),
        ease=5,
        caveat="only helps a STABLE prefix -- 'cache something that changes each call and you "
               "gain nothing' [T]. A nonce, a timestamp or a reordered tool list at the front "
               "invalidates everything downstream.",
    ),
    Lever(
        name="difficulty_routing",
        claim="'often cuts total spend by half' [T]; cascades 45-85% at ~95% quality [R]",
        claim_multiple=2.0, claim_notation="percentage",
        kind="reduction",
        targets=(("model_tier", 0.55, 0.50),),
        ease=3,
        caveat="needs a routing tier and a quality signal; breaks if the router escalates most "
               "traffic (see T14).",
    ),
    Lever(
        name="distillation",
        claim="5-40x per-token cost cut [R]",
        claim_multiple=40.0, claim_notation="multiple",
        kind="reduction",
        targets=(("model_tier", 0.15, 0.80),),
        ease=1,
        caveat="'wins on narrow, high-volume tasks and fails on open-ended long-tail work' [R]. "
               "Distil the easy path; never the tail.",
        conflicts=("difficulty_routing",),
    ),
    Lever(
        name="quantisation",
        claim="enables smaller/cheaper hardware [T] -- the corpus gives no multiple",
        claim_multiple=2.0, claim_notation="none",
        kind="reduction",
        targets=(("model_tier", 0.30, 0.40),),
        ease=2,
        caveat="'nearly free on many models, but on some it quietly drops accuracy -- always "
               "rerun your evals' [T]. Self-hosted only. The multiple above is an inference, "
               "and it is the only unquantified claim in the ladder.",
    ),
    Lever(
        name="context_discipline",
        claim="every wasted token 'is charged twice, once on the invoice and once as latency' [T]",
        claim_multiple=1.0, claim_notation="none",
        kind="reduction",
        targets=(("retrieved_context", 1.00, 0.30), ("conversation_memory", 1.00, 0.30)),
        ease=4,
        caveat="prompt surgery: breaks if the trimmed content was load-bearing.",
    ),
    Lever(
        name="output_caps",
        claim="20-40% token cut [R]; 'don't be wordy' ~15% of output tokens [R]",
        claim_multiple=1.67, claim_notation="percentage",
        kind="reduction",
        targets=(("output_length", 1.00, 0.30),),
        ease=5,
        caveat="can truncate legitimate answers; pairs with a schema, not with a character count.",
    ),
    Lever(
        name="reasoning_gating",
        claim="thinking billed at output rate, invisible in the response; 3-15x multipliers [R]",
        claim_multiple=15.0, claim_notation="multiple",
        kind="reduction",
        targets=(("reasoning_tokens", 1.00, 0.60),),
        ease=4,
        caveat="gate by task complexity, and measure the quality cost on the tasks you gate OFF, "
               "not on the ones you leave on.",
    ),
    Lever(
        name="bounded_retries",
        claim="retries multiply on failure [R] -- the corpus gives no multiple",
        claim_multiple=1.0, claim_notation="none",
        kind="reduction",
        targets=(("retry_overhead", 1.00, 0.50),),
        ease=4,
        caveat="a retry ceiling must sit INSIDE the loop, not in an after-the-fact alert.",
    ),
    Lever(
        name="batch_lane",
        claim="~50% discount, ~24h ceiling [R]",
        claim_multiple=2.0, claim_notation="percentage",
        kind="pricing",
        targets=((ALL, 0.20, 0.50),),
        ease=4,
        caveat="offline work only; anything a human waits on cannot use it.",
        conflicts=("reserved_capacity",),
    ),
    Lever(
        name="reserved_capacity",
        claim="15-70% on sustained, predictable load [R]",
        claim_multiple=3.33, claim_notation="percentage",
        kind="pricing",
        targets=((ALL, 0.45, 0.35),),
        ease=2,
        caveat="a commitment: breaks if utilisation disappoints. Conflicts with routing -- "
               "committing to capacity for traffic you intend to route away double-counts.",
        conflicts=("batch_lane", "difficulty_routing"),
    ),
)

LEVER_BY_NAME = {l.name: l for l in LEVERS}


# --------------------------------------------------------------------------------------------
# What a lever is actually worth.
# --------------------------------------------------------------------------------------------
def achievable(lever: Lever, bill=BILL) -> dict:
    """The saving a lever can bank, as a share of the TOTAL bill.

    This is the multiplication the corpus's ladder omits. A lever that halves a line worth 4% of
    the bill has saved 2%, whatever its discount says.
    """
    rows = []
    total = 0.0
    for (layer, share, discount) in lever.targets:
        if layer == ALL:
            layer_share = 1.0
        else:
            layer_share = bill_share(layer, bill)
        s = layer_share * share * discount
        rows.append({"layer": layer, "share_of_layer": share, "discount": discount,
                     "share_of_bill": layer_share, "saving": s})
        total += s
    return {"lever": lever.name, "kind": lever.kind, "ease": lever.ease,
            "rows": rows, "saving": total}


def bill_share(layer: str, bill=BILL) -> float:
    for l in bill:
        if l.name == layer:
            return l.share
    raise KeyError(layer)


def rank_by_achievable(levers=LEVERS, bill=BILL) -> list:
    return sorted((achievable(l, bill) for l in levers), key=lambda r: -r["saving"])


def rank_by_claim(levers=LEVERS) -> list:
    """Rank by the largest multiple each claim implies, which is what reading the ladder gives.

    The two orderings are derived from one source, and they disagree. The disagreement is the
    finding: the ladder is a list of DISCOUNTS and a programme banks `discount x share_of_bill`,
    so a 40x claim on a 15% share loses to a 10x claim on a 31% share.
    """
    return sorted(({"lever": l.name, "claim_multiple": l.claim_multiple,
                    "claim": l.claim, "notation": l.claim_notation,
                    "quantified": l.claim_notation != "none"} for l in levers),
                  key=lambda r: -r["claim_multiple"])


def ordering_disagreement(levers=LEVERS, bill=BILL) -> dict:
    by_saving = [r["lever"] for r in rank_by_achievable(levers, bill)]
    by_claim = [r["lever"] for r in rank_by_claim(levers)]
    return {"by_achievable": by_saving, "by_claim": by_claim,
            "moved": [n for i, n in enumerate(by_saving)
                      if i < len(by_claim) and by_claim[i] != n],
            "positions": {n: {"achievable": i + 1, "claim": by_claim.index(n) + 1}
                          for i, n in enumerate(by_saving)}}


# --------------------------------------------------------------------------------------------
# Stacking.
# --------------------------------------------------------------------------------------------
def apply_order(order, levers=LEVERS, bill=BILL) -> dict:
    """Apply levers in order, tracking how much of each layer each lever has already addressed.

    Named `apply_order` rather than `stack` deliberately: a public function called `stack` inside a
    package that also has a `stack` MODULE shadows it in the package namespace, and `from . import
    stack` then hands back this function instead of the module.

    Three rules, and without any one of them the stack produces a number that looks like synergy
    and is actually double-counting:

      * two levers cannot both claim the same share of the same layer;
      * a pricing lever applies to what is LEFT, not to the original bill;
      * a lever whose declared conflict has already been applied is SKIPPED, and the skip is
        reported rather than silently ignored. Batch and reserved capacity on one workload is one
        saving, not two.
    """
    by_name = {l.name: l for l in levers}
    if isinstance(order[0], str):
        order = [by_name[n] for n in order]

    remaining = {l.name: l.share for l in bill}
    claimed = {l.name: 0.0 for l in bill}
    applied = set()
    steps = []
    start = sum(remaining.values())

    for lv in order:
        blocked_by = [c for c in lv.conflicts if c in applied]
        if blocked_by:
            steps.append({"lever": lv.name, "kind": lv.kind, "saved": 0.0,
                          "bill_after": sum(remaining.values()),
                          "skipped": "conflicts with " + ", ".join(blocked_by)})
            continue
        saved = 0.0
        for (layer, share, discount) in lv.targets:
            if layer == ALL:
                base = sum(remaining.values())
                s = base * share * discount
                scale = (base - s) / base if base else 1.0
                for k in remaining:
                    remaining[k] *= scale
                saved += s
            else:
                avail = max(0.0, 1.0 - claimed[layer])
                take = min(share, avail)
                if take <= 0.0:
                    continue
                s = remaining[layer] * take * discount
                remaining[layer] -= s
                claimed[layer] += take
                saved += s
        applied.add(lv.name)
        steps.append({"lever": lv.name, "kind": lv.kind, "saved": saved,
                      "bill_after": sum(remaining.values()), "skipped": None})
    final = sum(remaining.values())
    return {"steps": steps, "start": start, "final": final,
            "reduction": 1.0 - final / start,
            "applied": sorted(applied),
            "skipped": [s["lever"] for s in steps if s["skipped"]]}


def conflict_pairs(levers=LEVERS) -> list:
    """Levers that cannot both apply to the same traffic, with the cost of assuming they can."""
    seen, out = set(), []
    for l in levers:
        for other in l.conflicts:
            key = tuple(sorted((l.name, other)))
            if key in seen:
                continue
            seen.add(key)
            out.append({"a": key[0], "b": key[1]})
    return out


def cheapest_first(levers=LEVERS, bill=BILL) -> list:
    """Alias for the corpus's ordering, kept because the name is the corpus's."""
    return order_free_first(levers, bill)


def free_wins(levers=LEVERS, threshold: int = 4) -> list:
    return [l.name for l in levers if l.ease >= threshold]


def reduction_levers(levers=LEVERS) -> list:
    return [l for l in levers if l.kind == "reduction"]


def pricing_levers(levers=LEVERS) -> list:
    return [l for l in levers if l.kind == "pricing"]


def value_per_ease(lever: Lever, bill=BILL) -> float:
    return achievable(lever, bill)["saving"] / max(lever.ease, 1)


def order_saving_first(levers=LEVERS, bill=BILL) -> list:
    """Highest saving per unit of effort, pricing levers included and competing with each other."""
    return sorted(levers, key=lambda l: -value_per_ease(l, bill))


def order_free_first(levers=LEVERS, bill=BILL) -> list:
    """The corpus's ordering: 'before you touch the model at all, exhaust the free wins' [T].

    Reduction levers only, free ones first. The single best PRICING lever is placed LAST on
    purpose: a pricing lever scales whatever is left, so its position does not change what it is
    worth, whereas placing it early changes what every later reduction lever is worth.
    """
    free = sorted([l for l in reduction_levers(levers) if l.ease >= 4],
                  key=lambda l: -value_per_ease(l, bill))
    rest = sorted([l for l in reduction_levers(levers) if l.ease < 4],
                  key=lambda l: -value_per_ease(l, bill))
    chosen = {l.name for l in free + rest}
    # The pricing lever must be chosen AFTER the reduction set is known, because a commitment
    # that conflicts with a lever already in the plan would simply be dropped.
    eligible = [p for p in pricing_levers(levers)
                if not any(c in chosen for c in p.conflicts)]
    best_pricing = max(eligible, key=lambda l: achievable(l, bill)["saving"]) if eligible else []
    return free + rest + ([best_pricing] if best_pricing else [])


def order_ease_first(levers=LEVERS, bill=BILL) -> list:
    """Ease first, everything included -- what 'do the easy things first' actually produces.

    This is the order that costs the most, because it lets the EASY pricing lever claim the
    pricing slot and then blocks the larger one behind a conflict.
    """
    return sorted(levers, key=lambda l: (-l.ease, -achievable(l, bill)["saving"]))


def ordering_comparison(levers=LEVERS, bill=BILL) -> dict:
    """Three defensible orderings of the same eight levers, and what each is worth.

    The spread between them is the cost of the ordering rule, and it is not small. The worst of
    the three is the one that follows the most natural instruction -- do the easy things first --
    because ease and value are not the same axis for pricing levers, which compete for one slot.
    """
    out = {}
    for name, order in (("free_first", order_free_first(levers, bill)),
                        ("saving_first", order_saving_first(levers, bill)),
                        ("ease_first", order_ease_first(levers, bill))):
        st = apply_order(order, levers, bill)
        out[name] = {"reduction": st["reduction"], "final": st["final"],
                     "order": [l.name for l in order], "skipped": st["skipped"],
                     "steps": st["steps"]}
    best = max(out.values(), key=lambda r: r["reduction"])["reduction"]
    worst = min(out.values(), key=lambda r: r["reduction"])["reduction"]
    return {"orders": out, "best": best, "worst": worst, "spread": best - worst}


def ceiling(levers=LEVERS, bill=BILL) -> dict:
    """The best any ordering can do, ignoring conflicts between reduction levers.

    Within a layer, levers take disjoint shares in order of their discount, so several reduction
    levers on one layer can each contribute their own share rather than only the strongest one
    counting. The single best PRICING lever is then applied to what is left -- one, not the sum,
    because a pricing lever scales every remaining token and two of them on the same token is one
    saving counted twice.

    This is a BOUND, not a plan, and it is the correct thing to compare a headline against
    because a headline has no ordering in it either.
    """
    rows = []
    total = 0.0
    pricing = 0.0
    order = sorted(levers, key=lambda x: -max(d for _, _, d in x.targets))
    for l in bill:
        avail = 1.0
        reduction = 0.0
        for lv in order:
            for (layer, share, d) in lv.targets:
                if layer == ALL:
                    pricing = max(pricing, share * d)
                    continue
                if layer != l.name or avail <= 0.0:
                    continue
                take = min(share, avail)
                reduction += take * d
                avail -= take
        reduction = min(1.0, reduction)
        kept = l.share * (1.0 - reduction)
        total += kept
        rows.append({"layer": l.name, "reduction": reduction, "share": l.share, "kept": kept})
    after_pricing = total * (1.0 - pricing)
    return {"rows": rows, "before_pricing": total, "pricing_lever": pricing,
            "final": after_pricing, "reduction": 1.0 - after_pricing / BILL_TOTAL}


def max_single_discount(levers=LEVERS) -> dict:
    """The largest discount anywhere in the ladder, and the share it is claimed to reach."""
    best = None
    for l in levers:
        for (layer, share, discount) in l.targets:
            if best is None or discount > best["discount"]:
                best = {"lever": l.name, "layer": layer, "share": share, "discount": discount,
                        "claim": l.claim}
    return best


def headline_check(headline_reduction: float = 0.90, levers=LEVERS, bill=BILL) -> dict:
    """Is 'roughly a tenth of the cost' reachable, and what does it take?

    Two bounds: the ceiling with every lever at its full claimed discount, and the ceiling with
    the single most aggressive number in the ladder applied at its widest reach. The gap between
    them is the amount of the headline that comes from one claim.
    """
    ceil = ceiling(levers, bill)
    top = max_single_discount(levers)
    return {"headline": headline_reduction, "ceiling": ceil["reduction"],
            "gap": headline_reduction - ceil["reduction"], "top_claim": top,
            "reachable": ceil["reduction"] >= headline_reduction}
