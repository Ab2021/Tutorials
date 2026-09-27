"""T09 -- seven demonstrations, each one a decision an operator actually has to make."""
from __future__ import annotations

from . import acceptance as A
from . import drafters as D
from . import regimes as R

# An illustrative target/draft pair over an 8-token vocabulary. Fixed rather than random so the
# printed distributions are stable and can be checked by hand.
TARGET = [0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]
DRAFT = [0.20, 0.35, 0.15, 0.12, 0.05, 0.06, 0.04, 0.03]


def exp_exactness() -> None:
    print("\n1. EXACTNESS -- the residual resampling is what makes this an exact sampler")
    print("   rule: accept with min(1, p_target/p_draft); on rejection resample from")
    print("   normalise(max(0, p_target - p_draft)).  Corpus: this is exact, not an")
    print("   approximation [T] -- so a shifted output distribution means a BUG.\n")

    rep = A.exactness_report(TARGET, DRAFT, n=200_000)
    print(f"   vocabulary 8, n = {rep['n']:,}, single-token acceptance = "
          f"{rep['acceptance_rate']:.3f}")
    print(f"\n   {'token':>5}  {'p_target':>9}  {'exact':>9}  {'naive':>9}")
    print("   " + "-" * 38)
    for i in range(len(TARGET)):
        print(f"   {i:>5}  {TARGET[i]:>9.4f}  {rep['exact'][i]:>9.4f}  {rep['broken'][i]:>9.4f}")

    print(f"\n   total variation from p_target:")
    print(f"     residual resampling (correct) : {rep['tv_exact']:>7.5f}")
    print(f"     resample p_target on reject   : {rep['tv_broken']:>7.5f}")
    print(f"     ratio                         : {rep['tv_broken'] / max(rep['tv_exact'], 1e-9):>7.0f}x")

    print("\n   READ: the correct construction converges to p_target to within sampling noise.")
    print(f"   The 'obvious' variant -- on rejection, just ask the target -- is off by")
    print(f"   {rep['tv_broken']:.4f} total variation, i.e. ~{rep['tv_broken'] * 100:.1f} percentage points, in a DIRECTION: it piles")
    print("   extra mass on the tokens the target favours. Fluent, on-topic, and not the model")
    print("   you deployed.")
    print("   WHY IT IS WRONG: the accepted branch has ALREADY contributed min(p_d, p_t) at")
    print("   every token, so resampling the full p_target on rejection double-counts exactly")
    print("   where the target exceeds the draft -- i.e. the tokens the draft declined to")
    print("   propose. This is the bug that no quality eval reliably catches, because the text")
    print("   is good. It is caught by a distribution test, which nobody runs.")
    print("   IF YOU TAKE ONE THING: speculative decoding is EXACT. Any measurable change in")
    print("   your output distribution is a defect, not a tuning effect.")


def exp_acceptance_decay() -> None:
    print("\n2. ACCEPTANCE DECAY -- why 'use more draft tokens' is wrong past a point")
    print("   tokens_per_step = 1 + sum_i prod_{j<=i} alpha_j\n")

    alpha1, decay = 0.80, 0.90
    print(f"   alpha_1 = {alpha1}, decay = {decay}, draft cost c = 0.014 (1B draft vs 70B)")
    print(f"\n   {'gamma':>6}  {'alpha at pos gamma':>19}  {'tokens/step':>12}  {'speedup':>9}")
    print("   " + "-" * 54)
    for r in R.gamma_sweep(alpha1, 0.014, decay, max_gamma=12):
        print(f"   {r['gamma']:>6}  {r['alpha_last']:>19.4f}  "
              f"{r['tokens_per_step']:>12.3f}  {r['speedup']:>9.3f}")

    best = R.optimal_gamma(alpha1, 0.014, decay, max_gamma=12)
    print(f"\n   optimum: gamma = {best['best_gamma']} at {best['best_speedup']:.3f}x")
    print(f"   speedup within 2% of the peak at gamma in {best['within_2pct']}")

    print("\n   READ: the numerator SATURATES while the denominator grows linearly in gamma.")
    print("   Acceptance decays with position because each draft token is conditioned on the")
    print("   previous drafted tokens -- which were themselves guesses. Errors compound.")
    print("   THE COMMON MODELLING ERROR: treating alpha as a single constant for all")
    print("   positions. That makes tokens_per_step rise almost linearly and predicts longer")
    constant_alpha = 1 + sum(alpha1 ** i for i in range(1, 9))
    true_at_8 = A.tokens_per_step(A.decay_profile(alpha1, 8, decay))
    print(f"   drafts are always better. At gamma=8 it predicts {constant_alpha:.3f} tokens/step")
    print(f"   where the decaying profile gives {true_at_8:.3f} -- a {constant_alpha / true_at_8:.2f}x")
    print("   over-prediction, and it is the reason a tuned gamma underperforms its forecast.")
    print("   THE FLATNESS MATTERS MORE THAN THE ARGMAX: pick the smallest gamma within 2% of")
    print("   the peak. Extra draft tokens past that point buy nothing and burn compute on")
    print("   every rejection.")


def exp_draft_cost() -> None:
    print("\n3. DRAFT COST -- acceptance is half the story; 'c' decides which variant wins")
    print("   speedup = tokens_per_step / (1 + gamma*c)\n")

    gamma = 8
    alphas = A.decay_profile(0.80, gamma, 0.90)
    variants = [
        ("n-gram / prompt lookup", D.ngram_cost()),
        ("draft model 1B vs 70B", D.draft_model_cost(1.0, 70.0)),
        ("MTP, 2 heads trained in", D.mtp_cost(2)),
        ("MTP, 4 heads trained in", D.mtp_cost(4)),
        ("Medusa tree, 8 candidates", D.medusa_tree_cost(8)),
        ("Medusa tree, 32 candidates", D.medusa_tree_cost(32)),
    ]
    print(f"   gamma = {gamma}, alpha_1 = 0.80, decay = 0.90 "
          f"-> {A.tokens_per_step(alphas):.3f} tokens/step\n")
    print(f"   {'drafter':>27}  {'c':>8}  {'denominator':>12}  {'speedup':>9}")
    print("   " + "-" * 60)
    for name, c in variants:
        denom = 1 + gamma * c
        print(f"   {name:>27}  {c:>8.4f}  {denom:>12.3f}  {A.tokens_per_step(alphas) / denom:>9.3f}")

    print("\n   READ the c column, not the names. The spread on the DENOMINATOR is 1.00 to")
    print("   3.94x at the same acceptance -- the drafter's cost is a bigger lever than its")
    print("   quality at this gamma, and it is the lever nobody benchmarks.")
    print("   n-gram is CATEGORICALLY different: c = 0 means a WRONG proposal costs nothing.")
    print("   Every other variant pays gamma*c whether or not the tokens are accepted, so a")
    print("   drafter that is merely cheap still loses efficiency on every rejection.")
    print("   c UNDERSTATES THE DRAFT MODEL'S REAL COST. The 1B draft's 0.014 is latency only;")
    print("   its weights and KV cache occupy HBM that would otherwise hold target KV, which")
    print("   lowers the concurrency ceiling (T07). The cost that bites is capacity, not time.")
    print("   AND THE TREES ARE NOT ABOUT c. A tree's cost rises with candidates while its")
    print("   acceptance profile improves (experiment 6) -- buy a tree for the better alpha,")
    print("   never for the cost.")


def exp_ngram_prompt_lookup() -> None:
    print("\n4. N-GRAM / PROMPT LOOKUP -- the drafter with no model, and a closed-form call")
    print("   acceptance = p_target(copied token).  A copy is a DETERMINISTIC proposal,")
    print("   so min(1, p_target/p_draft) collapses to p_target itself.\n")

    gamma = 5
    print(f"   gamma = {gamma}, c = 0 (no forward pass at all)\n")
    print(f"   {'p(copy)':>9}  {'alpha':>7}  {'tokens/step':>12}  {'speedup':>9}   workload")
    print("   " + "-" * 68)
    workloads = {0.99: "verbatim extraction", 0.95: "summarise / quote",
                 0.85: "RAG-grounded answer", 0.70: "code edit with context",
                 0.45: "paraphrase", 0.20: "open-ended writing",
                 0.08: "creative / high temperature"}
    for p_copy in [0.99, 0.95, 0.85, 0.70, 0.45, 0.20, 0.08]:
        alpha = D.ngram_acceptance(p_copy)
        # CONSTANT alpha across positions, unlike a learned draft. The drafter follows a matched
        # SPAN, and the model's copy behaviour along that span is roughly stationary -- a wrong
        # copy is wrong, it does not compound into a worse-conditioned next guess the way a
        # learned draft's error does. Hence no decay_profile() here.
        alphas = [alpha] * gamma
        tps = A.tokens_per_step(alphas)
        print(f"   {p_copy:>9.2f}  {alpha:>7.3f}  {tps:>12.3f}  {tps / 1.0:>9.3f}   "
              f"{workloads[p_copy]}")

    print("\n   READ: the when-to-use decision is DECIDABLE IN ADVANCE from the workload.")
    print("   The acceptance rate is a direct read of how much the model was going to copy")
    print("   the input anyway -- there is no draft model to be aligned, no distribution to")
    print("   drift, nothing to retrain. Ask 'does the output quote the input?' and you have")
    print("   the alpha.")
    print("   THE ASYMMETRY THAT MAKES THIS THE BEST DEFAULT: when the guess is wrong, c = 0")
    print("   means the step cost is 1.0 and you have produced exactly one token -- the same")
    print("   as plain decoding. n-gram speculation is FREE TO BE WRONG. A draft model at the")
    print("   same acceptance always carries its denominator.")
    print("   WHERE IT DOES NOTHING: creative and open-ended generation, where p(copy) is low")
    print("   and the drafter finds no span to follow. It is not harmful there -- it is inert,")
    print("   which is why shipping it by default costs almost nothing and occasionally pays")
    print("   enormously.")


def exp_regime_crossover() -> None:
    print("\n5. THE REGIME CROSSOVER -- the batch size at which speculation stops paying")
    print("   Speculation pays because a verify pass on gamma+1 tokens costs the SAME as a")
    print("   one-token decode -- true only while decode is MEMORY-BOUND.\n")

    params_b = 70.0
    balance = 295.0            # H100-class: 989 TFLOPS fp16 / 3.35 TB/s
    limit = R.memory_bound_batch_limit(params_b, balance)
    print(f"   70B at fp16, GQA KV 0.33 MB/token, machine balance {balance:.0f} flops/byte")
    print(f"   arithmetic intensity at batch 1 : "
          f"{R.arithmetic_intensity(1, params_b):>8.2f} flops/byte")
    print(f"   arithmetic intensity at batch 295: "
          f"{R.arithmetic_intensity(295, params_b):>7.2f} flops/byte")
    print(f"   MEMORY-BOUND LIMIT batch* = {limit:>7.1f} concurrent sequences\n")

    gamma = 4
    alphas = A.decay_profile(0.80, gamma, 0.90)
    c = D.draft_model_cost(1.0, 70.0)
    print(f"   gamma = {gamma}, acceptance = {A.tokens_per_step(alphas) / (gamma + 1):.3f} avg, "
          f"c = {c:.4f}")
    print(f"\n   {'batch':>7}  {'regime':>14}  {'step cost':>10}  {'speedup':>9}")
    print("   " + "-" * 46)
    for r in R.speedup_vs_batch(alphas, c, limit,
                                [1, 8, 32, 128, 256, 512, 1024]):
        print(f"   {r['batch']:>7}  {r['regime']:>14}  {r['cost']:>10.3f}  {r['speedup']:>9.3f}")

    print("\n   READ: the speedup is flat and healthy across the whole memory-bound region")
    print("   and then falls off a cliff at batch*. It does not taper -- a hard switch.")
    print("   WHY batch* MOVES THE REGIME AT ALL: bytes moved per step = W + batch*KV, while")
    print("   FLOPs = 2N*batch. Weights are a CONSTANT and KV GROWS WITH BATCH, so at small")
    print("   batch the weights dominate and extra tokens ride along on bandwidth that was")
    print("   going to be spent anyway. Past batch* the KV traffic dominates and every extra")
    print("   token in the verify pass costs real time.")
    print("   AND NOTE WHERE batch* ACTUALLY LANDS: a few hundred concurrent sequences. That")
    print("   is BELOW the concurrency a throughput-oriented deployment runs at and ABOVE what")
    print("   a latency-oriented one runs at. Most of the disagreement about whether")
    print("   speculative decoding 'works' is two deployments on opposite sides of this line.")
    print("   THE DESIGN CONSEQUENCE: speculation is a LATENCY tool for SMALL batches. It is")
    print("   not a throughput feature, and it must be gated on batch size -- see experiment 7.")


def exp_tree_vs_linear() -> None:
    print("\n6. TREE vs LINEAR DRAFTING -- what Medusa/EAGLE actually buy")
    print("   A linear draft commits to one branch: one early rejection discards every later")
    print("   token. A tree keeps m branches alive at every position.\n")

    gamma = 4
    ceiling = gamma + 1
    print(f"   gamma = {gamma}, so the per-step CEILING is {ceiling} tokens "
          f"(no draft can beat it)\n")
    print(f"   {'alpha_1':>8}  {'linear':>8}  {'tree m=4':>9}  {'tree m=16':>10}  {'gain m=4':>9}")
    print("   " + "-" * 54)
    for alpha1 in [0.95, 0.85, 0.75, 0.60, 0.45, 0.30]:
        alphas = A.decay_profile(alpha1, gamma, 0.90)
        lin = D.linear_expected_accepts(alphas)
        t4 = D.tree_expected_accepts(alphas, 4)
        t16 = D.tree_expected_accepts(alphas, 16)
        print(f"   {alpha1:>8.2f}  {lin:>8.3f}  {t4:>9.3f}  {t16:>10.3f}  {t4 / lin:>8.2f}x")

    print("\n   READ THE COLUMNS RIGHT TO LEFT. The gain from a tree is LARGEST AT MODERATE")
    print("   ACCEPTANCE and collapses at high acceptance, which is the opposite of the")
    print("   intuitive reading.")
    print(f"   AND NOTICE THE CEILING COLUMN: a wide tree saturates {ceiling:.0f} tokens/step at")
    print("   EVERY acceptance above ~0.3. That is what a tree IS -- it converts a")
    print("   per-candidate acceptance rate into near-certainty that the drafted depth is")
    print("   actually used, which is the same thing as saying it removes the truncation")
    print("   loss. Useful, and worth knowing, but it is not free depth.")
    print("   HONEST LIMITATION OF THIS MODEL: `1 - (1-alpha)^m` assumes the m candidates")
    print("   are INDEPENDENT draws from the target's conditional distribution. A real tree")
    print("   is built from a beam or a chain of heads, so its candidates are CORRELATED and")
    print("   drawn from a finite vocabulary -- when the target's token is not in the tree at")
    print("   all, no candidate rescues that position. The true curve is therefore FLATTER")
    print("   than the m=16 column and approaches the ceiling more slowly. Read the DIRECTION")
    print("   (trees pay most at moderate acceptance) and treat the magnitudes as optimistic.")
    print("   WHY: at alpha_1 = 0.95 the linear draft is already keeping nearly every branch,")
    print("   so there is nothing for extra candidates to rescue -- 1 - (1-a)^m is already")
    print("   ~1. At alpha_1 = 0.45 the linear draft is throwing away most of its work, and a")
    print("   tree recovers it.")
    print("   SO TREES PAY WHERE A DRAFT MODEL IS MEDIOCRE, and they pay least where it is")
    print("   already good. That inverts the usual deployment logic: if your draft model is")
    print("   excellent, EAGLE's tree buys you little and a plain draft is the cheaper")
    print("   configuration. The corpus's own note is that runtime support for this family is")
    print("   now native -- SGLang supports spec decoding 'from Eagle, MTP to Deep Flash' with")
    print("   'Spec V2 for better support the native spec decoding speed up' [T].")
    print("   THE COST SIDE: a tree's denominator grows with the candidate count (experiment")
    print("   3 -- 32 candidates gives c = 0.368). The m=16 column is NOT free.")


def exp_deployment_verdict() -> None:
    print("\n7. DEPLOYMENT VERDICT -- judge speculation at the batch size you actually reach")
    print("   A service that is memory-bound at p50 and compute-bound at p99 will show a")
    print("   speedup in every average and a REGRESSION under load.\n")

    params_b, balance = 70.0, 295.0
    limit = R.memory_bound_batch_limit(params_b, balance)
    gamma = 4
    alphas = A.decay_profile(0.80, gamma, 0.90)
    c = D.draft_model_cost(1.0, 70.0)
    print(f"   batch* = {limit:.0f}, gamma = {gamma}, c = {c:.4f}\n")
    print(f"   {'deployment':>28}  {'p50':>6}  {'p99':>6}  {'spd@p50':>8}  {'spd@p99':>8}  "
          f"{'verdict':>10}")
    print("   " + "-" * 76)
    profiles = [
        ("interactive chat, 1 replica", 8, 48),
        ("latency-SLA API, bursty", 32, 400),
        ("batch/offline throughput", 256, 1024),
        ("agentic, wide fan-out", 400, 900),
    ]
    for name, p50, p99 in profiles:
        v = R.deployment_verdict(alphas, c, limit, p50, p99)
        verdict = "SAFE" if v["safe"] else "REGRESSES"
        print(f"   {name:>28}  {p50:>6}  {p99:>6}  {v['speedup_at_p50']:>8.3f}  "
              f"{v['speedup_at_p99']:>8.3f}  {verdict:>10}")

    print("\n   READ: the second row is the trap. p50 32 is comfortably memory-bound and shows")
    print(f"   {R.deployment_verdict(alphas, c, limit, 32, 400)['speedup_at_p50']:.2f}x; p99 400 is past batch* and shows a LOSS. The service reports a")
    print("   speedup in every average and slows down exactly when it is least able to absorb")
    print("   it. Nothing alerts, because the mean improved.")
    print("   THE AGENTIC ROW IS THE OTHER HALF OF THE ANSWER, and it is the corpus's own")
    print("   finding: speculation attacks DECODE, while agentic workloads are prefill")
    print("   dominated -- 'prefill is occupying like 98% of the tokens' [T]. A wide-fan-out")
    print("   agent fleet is BOTH compute-bound (so speculation loses) AND prefill-heavy (so")
    print("   there is little decode to accelerate in the first place). MTP still bought the")
    print("   corpus ~2x throughput [T], but on a different mechanism: it improves")
    print("   INTERACTIVITY, i.e. per-request latency at moderate fan-out, not aggregate")
    print("   throughput at saturation.")
    print("   THE OPERATIONAL RULE: gate speculation on measured batch size, and turn it off")
    print("   above batch*. Or reduce gamma until both rows clear 1.0 -- a shorter draft has a")
    print("   smaller denominator and survives deeper into the compute-bound region.")
