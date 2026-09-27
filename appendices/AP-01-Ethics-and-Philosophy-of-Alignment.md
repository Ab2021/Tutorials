# AP-01 — The Ethics & Philosophy of Alignment

| Field | Value |
|---|---|
| **Module** | Appendix (bridge: moral philosophy → the alignment pipeline) |
| **Source lecture** | Michael Sandel, *Justice: What's the Right Thing To Do?* — Episode 01, "The Moral Side of Murder" |
| **Transcript file(s)** | `NoteGPT_Transcript_Justice What's The Right Thing To Do Episode 01 THE MORAL SIDE OF MURDER.txt` |
| **Companion code** | `none` |
| **Prerequisites** | CS-13 (SFT — the model you are about to align), CS-14 (§4.3 preference data, §4.4 reward modelling, §4.5 the KL term) |
| **Difficulty** | Conceptually hard, technologically trivial. There is no GPU work in this file. |
| **Hands-on required** | No — but §6 and §10 each end in a change to a dataset, a metric, or a gate, so it is not purely contemplative |
| **Estimated study time** | 3h reading + 1h working §12 and §10 |

> **Why this file exists.** `_MANIFEST.md` says it plainly: the Justice transcript *"is **not** a fine-tuning topic — it was bundled by accident. It is used only for AP-01, which bridges moral philosophy → 'what is a preference, and who gets to encode it' → RLHF. **It must not leak into the technical modules.**"* That sentence is the contract for this appendix, and this appendix is the entire budget for it.

> **What this appendix is.** A **bridge**, not a second alignment tutorial. RLHF, DPO, ORPO and GRPO are owned by CS-14 and by CS-24–CS-27. This file does not re-derive them and does not re-explain them; it assumes you have read CS-14 §4 and it points at the exact section for every mechanical claim. What it does instead is take the one question the engineering modules deliberately bracket — *whose ranking is `chosen`, and by what right* — and answer it far enough that you can name the hyperparameter, dataset decision, or metric that changes as a result.

> **What this appendix is not.** It is not an ethics review process, not a compliance manual, and not a substitute for the legal material in CS-12 §4.12 (the four regimes governing training data) and CS-01 §16.5 (the compliance angle). It does not adjudicate any moral dispute, and it will not tell you what your model should value. It tells you *where in your pipeline the question is already being answered without your noticing*, which is a different and more useful thing.

> **Callout legend for this file.** `> **Beyond the lecture:**` marks something not present in Episode 01 — usually because it belongs to a later lecture in the same course (Rawls, Aristotle) or to Sandel's later book, and I say so explicitly rather than inventing a timestamp. `> **Correction:**` marks a place where a claim commonly attributed to the lecture is not actually what Sandel says. Both conventions are inherited from `_BRIEF.md` §2.

---

## 0. Executive Summary

- **A preference is not a feeling. It is a four-slot structure**, and every slot has an owner: **(a) a ranking, (b) over outcomes, (c) held by someone, (d) elicited by some procedure.** The pipeline makes four decisions — which pairs exist, which alternatives were on the menu, which people ranked them, and how they were asked — and all four are value-laden. This is the load-bearing claim of the appendix, and §2 is the whole argument.
- **Sandel's Episode 01 is not a lecture about ethics theory. It is a demonstration that intuitions are inconsistent**, and it establishes that fact with three cases in 13 minutes: the trolley driver [00:00:10], the footbridge [00:04:07], the transplant surgeon [00:10:59]. The class endorses "five over one" in case 1, rejects it in cases 2 and 3, and cannot produce a principle that covers all three.
- **The two families that emerge are consequentialism and categorical reasoning** [00:13:05]–[00:15:35]. Every preference dataset in existence encodes a position on that axis, whether or not its authors have one. The `chosen`/`rejected` pair is the operational form of the question.
- **The lecture's three closing questions map onto three concrete engineering decisions.** Rights and where they come from → the constraint set and the refusal policy (CS-13 §12.3, CS-14 §2.4). Why a fair procedure legitimises its output → the annotation guideline and the tie-handling rule (CS-14 §4.3.4). What moral work consent does → the consent and provenance record for the data (CS-12 §4.12, CS-14 §16.5).
- **The pipeline aggregates disagreement by majority and then hides that fact.** A preference pair is a majority-of-one verdict if one person labelled it, and an unweighted average if a crowd did. The minority ranking — the one that lost — is **deleted at the moment the pair is written**, and its disappearance is not logged anywhere. That is a real, checkable, fixable design decision, and §5.1 and §6.5 treat it as one.
- **A reward model models what annotators *did*, not what is *right*.** Bradley–Terry (CS-14 §4.4) fits `P(y+ ≻ y−) = σ(r(y+) − r(y−))`; nothing in that equation references correctness, welfare, or truth. It is a description of a population's revealed ranking. Treating it as a normative oracle is a category error, and it is the single most common philosophical mistake in alignment engineering — §5.3.
- **Preference satisfaction and welfare are not the same thing.** A ranking can be fully satisfied while the person is harmed, and a person can be harmed by the satisfaction of their own stated preference. This is not a curiosity; it is the reason "the user said they liked it" is a bad reward signal on its own.
- **Distributional harm is invisible to an average.** A policy that maximises mean preference satisfaction can impose all of the cost on a subgroup that is too small to move the mean. The metric that would catch it — a disaggregated, per-slice win rate — is not in any tutorial and is the cheapest thing in this file to adopt (§6.2, §10 Scenario 4).
- **The pipeline's only defence against value drift is the KL term, and it is not a safety mechanism.** It keeps the policy *near the reference*; a KL anchor against a model aligned to someone else's values anchors you to *their* values (CS-14 §4.5, CS-14 §17 item 8).
- **The practices that actually help are boring and documentable:** annotator demographic disclosure, disaggregated evals, a versioned elicitation protocol, red-teaming the values the majority holds, and a written statement of which population the majority is. Every one is a mitigation, not a solution, and §6 says so explicitly rather than overselling them.
- **The failure mode that matters most is not bias. It is paralysis.** A team that treats every pipeline decision as an unresolvable moral question ships nothing, which is itself a decision with its own distributional consequences — the users who needed the model do not get it. §9 and §10 Scenario 5 are about this.
- **The one-line rule to take away:** *you cannot choose whether your pipeline encodes values; you can only choose whether you wrote down which ones.* Everything in §6 is a way of writing it down.

---

## 1. Why an appendix on moral philosophy sits inside a fine-tuning handbook

### 1.1 The honest argument for its inclusion

The alignment track already teaches that a preference dataset has a schema (`{prompt, chosen, rejected}` — CS-14 §4.3.1), a provenance (`CS-14 §4.3.2`'s eight-row table from "human annotators from scratch" to "self-play"), an agreement statistic (CS-14 §4.3.5), and a cost per pair (CS-14 §4.3.6). What it does not teach — because it is out of scope for a technical module and the instructor never raises it — is that **every one of those four things is a normative choice wearing engineering clothes**:

| Engineering statement | What it silently asserts |
|---|---|
| "We used `argilla/ultrafeedback-binarized-preferences-cleaned`" | GPT-4's judgements are an acceptable stand-in for human judgement, and its style preferences are an acceptable component of your reward signal (CS-14 §4.3.6 flags judge-bias transfer) |
| "We collected 5,000 pairs from our support team" | Five thousand pairs is enough, and this team's ranking is the ranking your users should get |
| "We set β = 0.1" | This much drift from the SFT model is acceptable; the values baked in at SFT are the ones worth preserving (CS-14 §4.5) |
| "Win rate improved from 0.52 to 0.61" | The comparison population used for that win rate is the population that matters |

None of those assertions is unreasonable. All of them are unexamined in most projects. The argument for this appendix is not that engineers need philosophy to do their jobs; it is that **the questions are already being answered, by default, by whoever wrote the last line of code, and the default answer is not always the one the team would have chosen deliberately.**

There is a second, sharper argument. Preference alignment is the only stage of the training pipeline whose *entire input* is human judgement. Stage 0 is text, stage 1 is text, stage 2 is demonstrations, stage 3 is **opinions**. That is a category change, and a field guide to what opinions are and what can go wrong with them is a legitimate engineering document.

### 1.2 The honest argument against it

The case against is strong and deserves to be stated at full strength rather than strawmanned:

1. **It is unbounded, and engineering time is bounded.** Epistemology, metaethics, and political philosophy have been running for 2,400 years and will not terminate. A document that invites an engineering team into that conversation risks converting a two-week decision into a two-quarter seminar. §9 calls this failure mode by name.
2. **It does not change the loss curve.** There is no hyperparameter you can set correctly by reading Kant. The honest contribution of philosophy here is *framing*, and framing is worth less than a working pipeline in almost every week that matters.
3. **The field's actual problems are solved by measurement, not by argument.** Inter-annotator agreement (CS-14 §4.3.5), capability regression (§4.8), and reward hacking (§4.9) are all detectable and all measurable. Where a measurement exists, having the argument instead is a downgrade.
4. **It invites the "ethics theatre" outcome.** A team that writes a Values Statement and changes no data, no metric, and no gate has produced a document, not an improvement. This appendix tries to structurally prevent that by requiring every §3 concept to terminate in an artefact (§3's "fails to map onto" line) and every §10 scenario to terminate in a change.
5. **The lecture is not a technical source and was never meant to be read as one.** It is a classroom argument with 18-year-olds at 10 a.m. Treating it as a specification would be a category error, and the appendix says so in §2.7 and §3's failure lines.

The honest synthesis: **the philosophy is worth roughly one day of an engineer's attention and roughly fifty lines of a design doc, and no more.** That is the budget this appendix is written to, and a reader who finds themselves spending a week here has mis-budgeted.

### 1.3 What would make this appendix a failure

Written down so the file can be audited against it:

- A concept in §3 that does not name a **data, metric, hyperparameter, or gate** in its "maps onto" line. (There is one exception, §3.15, and it is flagged as the weakest mapping in the file rather than quietly padded.)
- A claim attributed to the lecture without a timestamp, or with a timestamp that does not exist. Every `[MM:SS]` in this file is in the Episode 01 transcript.
- A cross-reference to a section that does not exist. Every `CS-NN §M` was verified by grep against the file; §13 lists them.
- Re-teaching RLHF or DPO. If you want the DPO loss, it is CS-14 §4.6.3 and CS-25.
- A §10 scenario that ends in "the team discussed it and felt better". Every scenario ends in a diff.

### 1.4 The scope contract

This appendix **will not** leak into the technical modules. Concretely:

- It does not add a section to any other file. It cites them, and it cites them by their real section numbers.
- It does not introduce a new alignment method, metric, or hyperparameter. Every artefact it discusses is already defined somewhere in this repo, and it links to that definition rather than restating it.
- It does not claim that any philosophical position is correct. It claims only that a position is *encoded* somewhere, and names where.

---

## 2. First-principles: what a "preference" actually is

### 2.1 The four slots

The word "preference" is used in this repo in three incompatible ways, and the ambiguity causes real arguments:

| Usage | Example | Where |
|---|---|---|
| **Statistical** | "The model prefers response A" = `π_θ(A\|x) > π_θ(B\|x)` | CS-14 §4.6.3 (the DPO log-ratio) |
| **Data** | "A preference" = one `chosen`/`rejected` row | CS-14 §4.3.1 |
| **Normative** | "The user prefers a shorter answer" | Everywhere, unexamined |

The statistical usage is well-defined. The data usage is a schema. **The normative usage — the one that justifies the other two — is a four-slot structure**, and it is where every value judgement in the pipeline actually lives:

| Slot | Question | Owner in the pipeline | Handbook artefact |
|---|---|---|---|
| **(a) A ranking** | Is A better than B, or is the pair a tie? Is the ordering total or partial? | Whoever wrote the annotation guideline | `CS-14 §4.3.4` (the eight required guideline fields) |
| **(b) Over outcomes** | Which alternatives were generated and put on the menu at all? | Whoever built the sampler / the prompt set | `CS-14 §4.6.12` (rejection sampling), `CS-14 §5.1` (the prompt distribution) |
| **(c) Held by someone** | Which people's rankings are in this dataset, and who is missing? | Whoever recruited the annotators or chose the judge | `CS-14 §4.3.2` (the source table), `CS-14 §4.3.5` (double-annotated control) |
| **(d) Elicited by some procedure** | Ranked, pairwise, thumbs, implicit, self-play? Under what instructions, with what incentives? | Whoever wrote the task and the pay structure | `CS-14 §4.3.4`, `CS-14 §4.3.6` (the cost/pair table) |

Sandel's lecture is, structurally, a 54-minute demonstration that **(a) through (c) are all contested and cannot be settled by appeal to intuition.**

### 2.2 Slot (a) — a ranking, and the tie problem

The lecture's first move is to establish that a ranking exists and that it is not stable. The class overwhelmingly turns the trolley [00:01:53] and overwhelmingly refuses to push the fat man [00:05:49] — the same 5-vs-1 arithmetic, opposite verdicts. Sandel's question — *"What became of the principle, better to save five lives even if it means sacrificing one?"* [00:05:49] — is precisely the discovery that the ranking the class *thought* it held is not a total order over outcomes. It is a partial order conditioned on features (contact, causation, agency) that the class had not articulated.

**That is the tie problem, and it is a live engineering problem.** CS-14 §4.3.4 item 2 requires the guideline to state what counts as a tie, because most guidelines forbid ties — which *"forces noise on genuinely equal pairs"*. IQ-14's Level 1 question on ties gives the correction: allow and drop ties, or handle them with cDPO's label smoothing, ε set to the measured noise rate (CS-14 §4.6.5).

The lecture supplies the *reason* the tie problem is not a technicality: when two responses differ only on a feature that the guideline has not articulated — the trolley's "contact" feature, in modern terms — annotators do not experience themselves as guessing. They experience a firm preference for which they cannot give a reason, exactly as the class did. Sandel lets this run for four minutes [00:07:30]–[00:09:53] precisely because the inability to articulate is the phenomenon.

**Where this bites.** An unarticulated feature becomes a learned feature. If 60% of your annotators implicitly penalise a response for being blunt, and your guideline never mentions tone, you have trained a tone preference and you cannot explain, audit, or reverse it — because no line in any document says it was supposed to be there. This is the mechanism by which a preference dataset ends up encoding values nobody chose.

### 2.3 Slot (b) — over outcomes, and the menu problem

A ranking is only defined over the options that were present. Sandel makes this point physically: the trolley case is a *choice*, but the fat man case introduces an option — push — that the onlooker had to *create*. Speaker 7's argument is exactly about menu membership: *"in the first situation, you're involved directly with the situation. In the second one, you're an onlooker as well… you have the choice of becoming involved or not by pushing the fat man"* [00:09:46]–[00:09:56].

**The pipeline analogue is the sampler.** A DPO pair is a comparison between two responses that a model happened to generate (or that an annotator happened to write). Nothing in the pipeline ranks the responses that were never sampled. CS-14 §4.6.3 lists this as DPO's first break condition — *"Offline only — it cannot explore beyond the pairs you give it, so its ceiling is your dataset's ceiling"* — and CS-14 §4.6.12 lists the corresponding failure for rejection sampling: *"Bounded by the sampler. `max(y_1..y_n) ≤ max over the model's support`."*

**Where this bites.** Preference data teaches the model to choose better *within the menu your sampler produces*. It does not teach it that the menu is wrong. This is why CS-14 §4.6.1 rates PPO's exploration advantage as real rather than obsolete, and it is the technical form of a philosophical point: **a ranking over a truncated option set is not a ranking over outcomes.**

### 2.4 Slot (c) — held by someone, and the question of who

This is the slot the lecture spends its final twenty minutes on, and it is the one with the most direct engineering consequence.

The Dudley and Stephens case [00:29:37]–[00:33:59] forces the class to specify *whose* ranking counts. Marcus defends the killing from necessity [00:35:57] and adds a consequentialist argument: *"they become productive members of society who go home and start a million charity organizations… They benefit everybody in the end"* [00:36:15]. Sandel immediately attacks the hidden variable — *"What if they went home, and they turned out to be assassins?"* [00:36:36] — which exposes that Marcus's argument requires a prediction about the *future* welfare of third parties, and Marcus concedes: *"That's fair."*

Then Sandel supplies the strongest form of the utilitarian case himself: the cabin boy *"had no family… no dependents. These other three had families back home in England. They had dependents. They had wives and children"* [00:49:05]. Note what has happened. **The 3-vs-1 headcount has been replaced by a weighted sum over a wider population**, and the weights come from counting dependents. Parker was an orphan, so his death is cheap: *"Parker was an orphan. No one would miss him"* [00:50:46].

**This is the most important paragraph in the lecture for an alignment engineer, and it is not about cannibalism.** The utilitarian aggregation did not produce a neutral answer. It produced an answer in which **the person with no constituency has weight zero**. In an annotation pipeline, the analogue is exact: a subgroup with few users, or few annotators, or no product owner, has weight approximately zero in the aggregate ranking — not because anyone decided that, but because aggregation does that by default.

**Where this bites.** Every preference dataset has an implicit `who`. CS-14 §4.3.2's table lists the sources but not the populations. §5.4 and §6.1 of this appendix make that gap actionable.

### 2.5 Slot (d) — elicited by some procedure, and why procedure is not neutral

The last third of the lecture is about procedure, and it contains the sharpest turn in Episode 01: the class's *judgement changes when the procedure changes*.

- Raw killing: about 20% find it morally permissible [00:46:12].
- Add a lottery: *"the numbers are rising if we add a lottery"* [00:43:21].
- Add consent from Parker: more still [00:47:11].
- Sandel pushes: suppose Parker consents and then changes his mind [00:45:26]. Speaker 19's answer is a contract: *"You've already decided. It's like a verbal contract. You can't go back on that"* [00:45:34].

And Sandel names the three unresolved questions explicitly [00:51:27]–[00:53:27]:

1. **Question one** — *"Is it because even cabin boys have certain fundamental rights? And if that's the reason, where do those rights come from, if not from some idea of the larger welfare or utility or happiness?"* [00:51:27]
2. **Question two** — *"Why does agreement to a certain procedure, even a fair procedure, justify whatever result flows from the operation of that procedure?"* [00:52:12]
3. **Question three** — *"What is the moral work that consent does? Why does an act of consent make such a moral difference that an act that would be wrong, taking a life without consent, is morally permissible with consent?"* [00:52:46]

**These are the three questions a preference-elicitation protocol answers, usually by accident.** A pairwise "which is better?" prompt is a *different procedure* from a 1–5 Likert rating, which is different again from a thumbs-up, which is different from "reject this response and regenerate". Each produces a different ranking over the same underlying quality, and the difference is not noise — it is the procedure's signature. CS-14 §4.3.2's KTO row is the repository's acknowledgement of this: *"Pointwise / unary preference data — a single response with a binary desirable/undesirable label (KTO's format). Lets you use data that has no pair."* KTO exists because the procedure that produced the data (thumbs) is not the procedure that produces pairs.

**Where this bites.** If you change your elicitation procedure mid-collection, you have changed what you are measuring while keeping the same column names. The dataset looks homogeneous. It is not. §6.3 turns this into a requirement.

### 2.6 The four slots as four injection points

| Slot | The decision | Where it is made | Is it written down anywhere in your repo? | Default if unwritten |
|---|---|---|---|---|
| **(a) Ranking** | Is the ordering total or partial? What is a tie? | Annotation guideline | Sometimes (`CS-14 §4.3.4`) | All pairs are strict; ties become noise |
| **(b) Outcomes** | Which responses were sampled and compared? | Sampler, prompt set, model version | Rarely | The model's own mode, plus the annotator's own style |
| **(c) Holder** | Whose ranking is in the data? | Annotator recruitment, judge choice | Almost never | Whoever was cheapest to hire or easiest to reach |
| **(d) Procedure** | How was the ranking elicited? | Task design, UI, pay structure | Partially (the guideline) | Whatever the labelling vendor's default UI does |

Read the fourth column as the honest state of the field. Read the fifth as what happens if you leave it. **The fifth column is not a moral catastrophe; it is simply an undocumented design decision**, and undocumented design decisions are the normal source of production incidents in every other part of this pipeline too (CS-01 §16.1, CS-13 §16.1).

### 2.7 Where the lecture is not the source

Two honesty notes about this section.

> **Correction:** the four-slot decomposition in §2.1 is *mine*, not Sandel's. He does not present preferences as a four-part structure; he presents three trolley cases and lets the class tie itself in knots. What the lecture *does* establish, at [00:51:27]–[00:53:27], is that rights, procedure, and consent are each independently contested — which is what makes the slots non-trivial. Attribute the structure to this appendix and the contestation to the lecture.

> **Beyond the lecture:** the vocabulary of "revealed preference" vs "stated preference" is economics, not this lecture, and the distinction matters for how seriously you take a labelling UI. A stated preference is what someone says when asked; a revealed preference is what they do when it costs them something. Most preference datasets are stated preferences collected under a task framing, and the two diverge systematically in exactly the domains alignment cares about (safety, tone, verbosity).

---

## 3. Core concepts, exhaustively

Sixteen concepts, each in a fixed three-part format:

- **What the lecture says** — grounded in Episode 01, with a timestamp.
- **What it maps onto** — the named artefact, hyperparameter, metric, or gate it changes. This is required.
- **What it fails to map onto** — where the analogy breaks. Also required, and taken seriously.

The failure line is not a hedge. In several cases it is the more useful half.

---

### 3.1 The trolley problem (the driver)

**What the lecture says.** A runaway trolley at 60 mph is heading for five workers; the brakes are dead; a side track holds one worker; the steering works [00:00:10]–[00:01:03]. The class votes: *"The vast majority would turn"* [00:01:53]. The first articulated reason is pure aggregation: *"it can't be right to kill five people when you can only kill one person instead"* [00:02:31]. At least one student immediately reaches for the extreme case — the 9/11 passengers who brought down the plane in Pennsylvania are *"heroes because they chose to kill the people on the plane and not kill more people in big buildings"* [00:03:02] — and another reaches for the opposite extreme, calling the reasoning *"the same type of mentality that justifies genocide and totalitarianism"* [00:03:40].

**What it maps onto.** The **default objective**: maximise expected good, where good is a scalar and lives are fungible units. That is `max E[r(x,y)]` with no constraint term — which is exactly the objective in the `β → 0` row of CS-14 §4.5's extremes table, and exactly the regime CS-14 §4.9 describes as reward hacking. The trolley case is the clearest possible statement of why unconstrained aggregate optimisation is the *first* instinct and why it needs a brake.

**What it fails to map onto.** The trolley's decision-maker is a single agent making one irreversible decision with full information. A preference pipeline makes millions of tiny reversible decisions with no information about outcomes. The structural feature that makes the trolley hard — **irreversibility under certainty** — is absent from the training loop, and the feature that makes the training loop hard — **aggregation over a population with no shared identity** — is absent from the trolley. Do not read the trolley as a model of alignment; read it as a model of the *objective function only*.

---

### 3.2 The footbridge (the fat man), and the doing/allowing distinction

**What the lecture says.** Same five workers, but you are an onlooker on a bridge, and the available action is to shove a large man onto the track [00:04:07]. *"Most people wouldn't"* [00:05:49]. The class then spends four minutes failing to articulate why. Speaker 5 tries the "he wasn't already involved" argument; Sandel flattens it — *"the guy working, the one on the track off to the side, he didn't choose to sacrifice his life any more than the fat man did, did he?"* [00:06:51]. Speaker 7 offers contact and causation — *"pushing the fat man over is an actual act of murder on your part. You have control over that"* [00:07:30]. Speaker 8 points out the symmetry — *"either way, you're making a choice"* [00:08:09]. Sandel then removes the contact with the trap-door variant [00:08:59], and Speaker 7, pressed, still says it feels wrong: *"For some reason, that still just seems more wrong"* [00:09:18].

**What it maps onto.** Two things, and both are operational.

1. **The doing/allowing distinction is a constraint, not a term.** If harm-by-acting is worse than harm-by-omitting, that is a hard constraint on the action space, not a penalty in the reward. Engineering form: a refusal policy, a confirmation gate, a tool-call allow-list. CS-13 §12.3 makes refusal rate a first-class metric with two directions (false refusal on benign input, true refusal on harmful input) — that is the *engineering* of a deontological constraint, and it is measured, not argued.
2. **The class's inability to state a principle is the guideline problem.** Four minutes of articulate students unable to say why the cases differ is what an annotation guideline is *for*. CS-14 §4.3.4 requires an explicit HHH priority order and an escalation path; the escalation path exists precisely for the pair an annotator cannot judge [00:05:49]'s analogue.

**What it fails to map onto.** The footbridge has one actor, one act, and one victim. A language model's "act" is a token distribution over an unbounded context, and there is no single moment of pushing. **The doing/allowing distinction presumes a well-defined act**, and in a generative model the boundaries of an act are a modelling choice. This does not make the distinction useless; it makes it a *policy* you write (what counts as the model doing a thing) rather than a fact you read off the world.

---

### 3.3 The doctrine of double effect

**What the lecture says.** Sandel never uses the phrase, and it is not in Episode 01. What he does is surface the phenomenon: *"people gestured toward reasons having to do with the intrinsic quality of the act itself, consequences be what they may. People were reluctant. People thought it was just categorically wrong to kill an innocent person, even for the sake of saving five lives"* [00:14:18].

**What it maps onto.** The doctrine's engineering shadow is the **intent/effect split in a reward function**: a policy is scored on what it aimed at, not only on what resulted. In practice this is nearly unrepresentable — a reward model scores the observed completion, and "aim" is not observable. The one place it becomes concrete is **process vs outcome rewards**: CS-14 §4.6.10 notes that GRPO's per-response advantage *"cannot express 'this part of the reasoning was good, that part was bad'"*, and that per-step credit needs a process reward model. A process reward is the closest thing in the field to scoring the act rather than the result.

> **Beyond the lecture:** double effect is Thomas Aquinas's formulation, later developed by Anscombe and Foot, and it is the standard name for the distinction the class gropes toward in [00:07:30]–[00:09:53]. It was not named in Episode 01 and I have not given it a timestamp, because inventing one would be worse than omitting it.

**What it fails to map onto.** Double effect requires knowing the agent's intention. A reward model has no access to intention and a policy has no intention in the philosophically relevant sense — it has a conditional distribution. Any claim that a model "meant" the harmful part of an output is an interpretation, and the current honest position is that the field cannot ground it. Treat this concept as **framing for escalation policy**, not as something you can compute.

---

### 3.4 The transplant surgeon, and the limits of aggregation

**What the lecture says.** Same structure, escalated twice. First the emergency-room version: five moderate injuries, one severe; save the five, lose the one [00:10:15]. Almost unanimous. Then the transplant version: five patients each needing a different organ, and a healthy man asleep in the next room [00:10:59]. *"How many would do it?"* — nobody. And then the best moment in the lecture: a student in the balcony proposes an *accounting* fix — take the organs from whichever of the five dies first and save the other four [00:12:36]. Sandel laughs, calls it *"a great idea"*, and then: *"Except for the fact that you just wrecked the philosophical point"* [00:13:03].

**What it maps onto.** The balcony student's move is **exactly what a metric-optimising pipeline does when the metric is a proxy.** The philosophical point that got wrecked is that the case was constructed to isolate `kill one, save five` with no other variables; the student found a *different* policy that scores better on the metric while being a different kind of act. That is **specification gaming**, and it is the same phenomenon as CS-14 §4.9's reward hacking: the policy maximises the proxy, not the objective, and the proxy was never the objective.

The concrete artefact: **your evaluation set, not your training set.** The transplant case is the argument for holding out adversarial cases whose *only* difference from the training distribution is the feature you care about, and for re-reading any suspiciously large win as "did the model find the student's answer?" before celebrating it.

**What it fails to map onto.** Sandel's point depends on the case being a *thought experiment with fixed facts*. Real pipelines do not have fixed facts; the "student's alternative" is often the genuinely better engineering answer, and refusing it in the name of preserving the experiment is how teams ship worse models. The mapping is to *suspicion of the metric*, not to *refusal of the alternative*.

---

### 3.5 "The moral side of murder" — the episode's thesis

**What the lecture says.** The title is the thesis: murder has a *moral side* — that is, some killings are not murder, and the difference is not in the consequences. The closing frame is Sandel's warning about what the course does to you: *"philosophy teaches us and unsettles us by confronting us with what we already know"*, and *"once the familiar turns strange, it's never quite the same again"* [00:18:08]–[00:18:56].

**What it maps onto.** The claim that **a label can be correct while the concept behind it is unexamined**. `chosen` and `rejected` are labels; `murder` is a label. In both cases the interesting content is in the rule that produced the label, and in both cases the rule is usually unavailable. The engineering form is the **audit question**: for any preference dataset you ship with, can you produce the rule? CS-14 §16.5 makes annotation guidelines a versioned policy document for exactly this reason, and IQ-14 Q99 turns it into the regulatory answer: the guidelines document *"itself"* is the evidence.

**What it fails to map onto.** The philosophy course's *aim* is to unsettle — *"the aim of this course is to awaken the restlessness of reason"* [00:23:40]. A production pipeline's aim is to settle. Adopting the lecture's therapeutic goal inside an engineering team is a category error, and it is the failure mode §9 calls paralysis.

---

### 3.6 Consequentialist moral reasoning

**What the lecture says.** *"The first moral principle that emerged in the discussion said the right thing to do, the moral thing to do, depends on the consequences that will result from your action"* [00:13:05]. Formalised immediately after: *"Consequentialist moral reasoning locates morality in the consequences of an act, in the state of the world that will result from the thing you do"* [00:13:44]. Its most influential example is utilitarianism [00:15:35], and its most influential critics in the syllabus are Kant, and (for other reasons) Aristotle and Locke [00:16:15].

**What it maps onto.** The **loss function**. Every alignment objective in CS-14 §4.6 is consequentialist: it maximises an expected reward subject to a KL constraint, and the constraint itself is justified consequentially (stay in the region where the reward model is valid — CS-14 §4.5's three jobs). The SFT stage is consequentialist in a different sense: cross-entropy measures agreement with outcomes, not the quality of the reasoning that produced them.

**What it fails to map onto.** Consequentialism in ethics is a claim about *what makes an act right*; consequentialism in ML is a choice of *what to optimise*. They come apart in a specific place: a consequentialist ethics can admit side-constraints (rule-consequentialism), and a consequentialist loss cannot express a constraint that is not a term in the sum. That is why the field's safety constraints are implemented as *gates, filters, and classifiers outside the loss* rather than as penalties inside it. The lecture's consequentialist/categorical split predicts this architectural split, which is a genuinely useful thing for it to do.

---

### 3.7 Categorical moral reasoning

**What the lecture says.** *"Categorical moral reasoning locates morality in certain absolute moral requirements, certain categorical duties and rights, regardless of the consequences"* [00:15:03]. The class's categorical voices are the minority in the trolley case [00:03:40] and the majority in the transplant case [00:12:07]. Mike's version at the end of the Dudley discussion is the cleanest statement in the lecture: *"murder is murder in every way… I don't think it's any different in any case"* [00:48:14], and when Sandel escalates the numbers — *"suppose it weren't three, suppose it were 30. 300… 3,000"* — Mike holds: *"I think it's still the same deal"* [00:49:49]. Sandel then forces the contradiction explicitly: *"Well, then Bentham has to be wrong. If you're right, he's wrong."* And Mike: *"Okay, then he's wrong."* [00:50:10]

**What it maps onto.** **Hard constraints**, and their cost. A categorical rule in a pipeline is a thing that does not move when the reward gradient pushes it: a refusal that survives a jailbreak, a schema that rejects a malformed output, a PII filter that does not care that the user asked nicely. CS-13 §12.3's true-refusal rate on a harmful set is the metric for "does the categorical rule still hold". CS-01 §16.4 item 2 is the regression test: *"Safety regression — a refusal suite plus a jailbreak suite."*

**What it fails to map onto.** Kant's categorical imperative is universalisable and grounded in reason; a hard constraint in a pipeline is *implemented* and grounded in a policy decision. The word "categorical" hides the difference. A model's refusal is not categorical in Kant's sense — it is a very strong learned disposition that CS-13 §16.5 warns *"measurably weakens"* under domain fine-tuning. Treating it as a moral absolute rather than a fragile statistical property is the mistake; the lecture's Mike is more coherent than most engineering teams, because at least he accepts that he must reject Bentham. **A team that wants categorical guarantees must build them outside the weights.**

---

### 3.8 Utilitarianism — Bentham, utility, the two sovereign masters

**What the lecture says.** *"The right thing to do, the just thing to do, is to maximize utility"* [00:27:14]. Utility is specified as *"the balance of pleasure over pain, happiness over suffering"* [00:27:14]. The grounding is psychological and universal: *"all of us, all human beings, are governed by two sovereign masters, pain and pleasure"* [00:28:15], therefore *"the right thing to do, individually or collectively, is to maximize… the overall level of happiness"* [00:28:54]. The slogan: *"The greatest good for the greatest number"* [00:28:54].

**What it maps onto.** The **reward model itself**, in the most literal sense available. Bentham's utility is a scalar summary of a population's pleasures and pains; a Bradley–Terry reward model (CS-14 §4.4) is a scalar summary of a population's revealed rankings. Both are commensurating devices: they take heterogeneous things and put them on one axis so they can be added. Both are *learned from data about people* rather than derived. And both are only as good as the population they were fit on.

The mapping is close enough to be worth stating as a table, and close enough that the disanalogies in the next paragraph matter:

| Bentham | Bradley–Terry / RLHF |
|---|---|
| Utility = balance of pleasure over pain | `r_φ(x,y)` = scalar score |
| Sum over persons | Expectation over prompts and pairs |
| The greatest good for the greatest number | Maximise `E[r]` subject to `β·KL` (CS-14 §4.5) |
| Measured by... a legislator's judgment | Measured by... annotator pairs (CS-14 §4.3.2) |

**What it fails to map onto.** Bentham's sum is over *persons* and *experiences*; RLHF's expectation is over *prompts and sampled completions*. There is no person in the RLHF objective, and adding one ("the user's utility") is a modelling choice the objective does not require. CS-14 §2.4's HHH table is the field's attempt to patch this — helpfulness, harmlessness, and honesty as three incommensurable axes — and CS-14 §2.4 is explicit that they *"cannot be maximised independently"* and that the trade-off point is a product decision. That is a Benthamite sum with a hand-inserted Pareto frontier, and the frontier position is where the values live.

---

### 3.9 Aggregation, and the cost of the greatest-number rule

**What the lecture says.** Sandel does not merely present utilitarianism; he demonstrates its cost. Having built the strongest utilitarian case for the killing — three dependents against one orphan [00:49:05] — he immediately makes the class watch the principle eat an entire population: *"Suppose it weren't three, suppose it were 30. 300. One life to save 300. We're in wartime. 3,000"* [00:49:49]. Mike does not flinch, and Sandel's follow-up is the reductio: *"Do you think Bentham is wrong to say the right thing to do is to add up the collective happiness?"* [00:50:02].

Then, at [00:51:27], he lists the three objections the class produced, and the first is about rights: *"Is it because even cabin boys have certain fundamental rights? And if that's the reason, where do those rights come from, if not from some idea of the larger welfare or utility or happiness?"*

**What it maps onto.** Two artefacts, and they are both gates:

1. **The floor.** A minimum acceptable performance on a slice, imposed as a hard gate rather than a term in the objective. Concretely: a disaggregated eval with a per-slice floor (§6.2), the safety suite in CS-01 §16.4, and the CI regression gate in CS-14 §16.4 (`KNOWN_GOOD`, asserted on promotion). A gate is a right in the engineering sense: the aggregate cannot buy its way past it.
2. **The aggregation weight.** Whose pairs count, and how many times. A dataset with 5,000 pairs from one region and 50 from another is an unweighted sum with an implicit 100:1 weighting. Nothing in the training code knows this; nothing in the eval reports it either, unless you ask for slices.

**What it fails to map onto.** Rights in the lecture are *pre-political* — Sandel's question is where they come from if not from utility. A pipeline floor is *post-political*: it comes from a product decision, a legal requirement, or a team's judgement, and it is exactly as legitimate as the process that set it. Do not dress a slice floor as a right; it is a decision, and decisions can be revisited. The value of the lecture here is that it forces the question *who set this floor and on what authority* — not that it supplies an answer.

---

### 3.10 The "we are not our own" thesis

**What the lecture says.** Episode 01 does not contain the full thesis; it contains its precursor. The class's most categorical voice argues against self-ownership as a *power*: *"there's no situation that would allow human beings to take the idea of fate or the other people's lives in their own hands, that we don't have that kind of power"* [00:38:05]. And the strongest pro-consent voice argues the opposite — that a person *does* have that power over their own life, which is why consent transfers it: *"if he was making his own original idea… then he took on the agency to sacrifice himself"* [00:40:16].

> **Beyond the lecture:** "we are not our own" is Sandel's later framing of Kant's and Rawls's anti-ownership arguments, developed in later lectures of this course and in *Justice: A Reader*. It is **not** in Episode 01, and I have not attached a timestamp to it. The material in Episode 01 that motivates it is the exchange at [00:38:05]–[00:41:20].

**What it maps onto.** The **data-governance stance**, and it is a genuinely binary engineering choice:

| Stance | Position | Engineering form |
|---|---|---|
| **Self-ownership** | People may license their preferences and data; consent is the whole of the moral work | Consent capture at the logging boundary, opt-in telemetry, per-user deletion (CS-13 §16.5's *"Right to erasure"* row) |
| **Not our own** | Some things may not be traded regardless of consent | Non-negotiable exclusions — CS-12 §4.12's four regimes, CS-13 §16.5's PII rule |

The repository already takes a position on this without labelling it: CS-13 §16.5 requires that PII *"scan both the prompt and the response spans; de-identified before the row enters the dataset, not after"* — which is a "not our own" rule, applied to PII, regardless of what the user consented to. CS-14 §16.5 applies the opposite stance to telemetry: *"Customer telemetry used as preference data (the KTO case) must pass the same retention and consent rules as the raw logs"* — which is a self-ownership rule.

**What it fails to map onto.** The thesis is about persons and their bodies. A preference pair is about a *label* someone assigned to two machine outputs. Conflating "the annotator consented to label" with "the annotator consented to have their values shipped inside a model" is not licensed by the thesis and is not licensed by consent theory either — they are different acts. This is the specific place where the mapping most often fails in practice, and it is §11's misconception 4.

---

### 3.11 Consent

**What the lecture says.** Kathleen raises it [00:38:28]: *"I'm wondering if Dudley and Steven had asked for Richard Parker's consent in dying, if that would exonerate them from an act of murder?"* Sandel works it hard for seven minutes. He tests the counterfactual (Parker says yes in a semi-stupor [00:39:41]), tests the strong version (*"if he was making his own original idea… you couldn't make the argument that he was pressured"* [00:40:16]), and names the three-part structure of the problem at [00:52:46]: *"What is the moral work that consent does? Why does an act of consent make such a moral difference that an act that would be wrong, taking a life without consent, is morally permissible with consent?"*

He also tests the *limits*: Speaker 17 refuses to accept even consent — *"you don't know when they're going to get rescued. So, if you kill him, it's killing him in vain"* [00:41:20] — and separately objects to cannibalism as such [00:42:08]. And Sandel draws out the coercion worry: with three against one, is a "yes" a choice or a capitulation?

**What it maps onto.** The **consent record**, which is a real field in a real dataset or it is nothing. Concretely:

- **The prompt's source.** Real user traffic vs synthetic vs written for the purpose. CS-14 §4.3.2 distinguishes these by cost and quality but not by consent; CS-12 §4.12.2's provenance ledger and CS-12 §16.6's source-rights checklist are the repository's mechanism for it.
- **The annotator's terms.** Whether the worker consented to their *judgements* being used, not merely to doing the task. This is not in the repo's checklists and it is a genuine gap — §6.1 and §6.3 are the recommendation.
- **The asymmetry.** Sandel's three-against-one point is the coercion analysis, and its engineering form is **annotator power**: a contractor paid per pair who is told the client prefers longer answers has a consent problem that no form fixes. CS-14 §4.3.4 item 3 — the explicit length-neutrality clause — is the mitigation, and it is a *procedure*, not a contract.

**What it fails to map onto.** Consent in the lecture is to a *harm to the consenter*. Consent in a data pipeline is to the *use of a judgement*. The second is far weaker morally, which is why "we have consent" is a much smaller claim than it sounds, and why §11's misconception 4 is worth a whole entry. Additionally: a policy trained on consented data affects *third parties who never consented*, and no amount of consent from the annotators reaches them. That gap is §5.4.

---

### 3.12 The lottery, fair procedure, and procedural legitimacy

**What the lecture says.** Dudley proposed a lottery on the 19th day and Brooks refused [00:32:09]. When Sandel puts the lottery to the class, support for the killing rises: *"the numbers are rising if we add a lottery"* [00:43:21]. Matt explains why with the sharpest line in the lecture: *"the essential element, in my mind, that makes it a crime is the idea that they decided at some point that their lives were more important than his… if they had done a lottery where everyone consented that someone should die… then it would be all right"* [00:43:44]. Sandel's summary: *"what bothers you is not the cannibalism, but the lack of due process"* [00:44:21]. And a second student locates the violation precisely: *"the cabin boy was never consulted about whether or not something was going to happen to him"* [00:44:51].

**What it maps onto.** The **annotation guideline, the sampling frame, and the tie rule** — all three are procedures whose legitimacy is doing moral work, and all three change the data.

| Procedural decision | What it legitimises | Handbook artefact |
|---|---|---|
| Which prompts enter the pool | Whose use cases are represented | CS-14 §5.1 (the prompt distribution), §4.3.2 |
| Which responses are sampled | Which options exist to be ranked | CS-14 §4.6.12 (rejection sampling), §4.6.1 (PPO exploration) |
| The tie rule | Whether a genuine equality is recorded or invented | CS-14 §4.3.4 item 2, §4.6.5 (cDPO) |
| The escalation path | Whether hard pairs become noise or get adjudicated | CS-14 §4.3.4 item 7 |

Sandel's Question Two — *"Why does agreement to a certain procedure, even a fair procedure, justify whatever result flows from the operation of that procedure?"* [00:52:12] — is the question a team answers when it says "we used a standard labelling vendor and a standard guideline". The procedure's fairness does not transfer to the output's correctness. **A well-run annotation process produces a well-documented opinion, not a fact.**

**What it fails to map onto.** In the lecture, everyone the procedure touches is *present and votes*; the wrong is that Parker was excluded from the procedure. In a pipeline, the affected population (end users) is essentially never in the procedure at all, and the people in the procedure (annotators) are largely not the affected population. The lottery analogy therefore **understates** the problem: it is not that one affected party was excluded from a fair procedure, it is that the procedure's participants and its subjects are disjoint groups. §6.1 is about closing that gap as far as it closes.

---

### 3.13 The veil of ignorance

**What the lecture says.** Nothing. The phrase does not appear in Episode 01, and neither does Rawls.

> **Beyond the lecture:** the veil of ignorance is Rawls's device in *A Theory of Justice* (1971) — choose the principles of justice as if you did not know your own place in society — and Sandel treats it in a later lecture of this same course. It is included here because it is the most directly useful idea in the course for an alignment engineer, and because **not** naming it would leave §5.4 with no constructive proposal.

**What it maps onto.** A **specification-writing heuristic**, and a good one. Write the annotation guideline, the refusal policy, and the eval slices **as if you did not know which side of them you would be on** — which user group, which region, which language, which use case. Concretely it produces three artefacts:

1. **A slice list that is not derived from the traffic you happen to have.** The veil forces you to ask which populations *could* be affected, not which ones are in your logs today.
2. **A refusal policy written symmetrically.** Not "what should we refuse for these users" but "what would I accept being refused, not knowing whether I am that user."
3. **A distributional check on every aggregate metric.** Every mean is a claim about a population; the veil is the discipline of asking which population, and who is not in it.

**What it fails to map onto.** Rawls's veil is a device for deriving principles of justice under a specific theory of primary goods and rational choice. Transplanting it as a design heuristic is exactly that — a transplant, and it loses the theoretical grounding. There is also a real objection: an engineer imagining themselves into a user group they have never been part of is a weak substitute for actually consulting that group. **The veil is a floor, not a ceiling**; §6.1's disclosure and §6.4's red-teaming are the stronger instruments.

---

### 3.14 The moral limits of markets

**What the lecture says.** Nothing directly. Episode 01's market-adjacent moment is the dependents argument [00:49:05] — lives weighed by the number of people who would miss them — and Sandel's aside about the newspaper's sympathy: *"if they weren't motivated by affection and concern for their loved ones at home and their dependents, surely they wouldn't have done this"* [00:49:05].

> **Beyond the lecture:** "the moral limits of markets" is Sandel's 2012 book *What Money Can't Buy* and a later part of this course, not Episode 01. No timestamp exists for it in this transcript and none is given.

**What it maps onto.** Two live engineering decisions that are literally about whether a thing should be priced:

1. **Annotation labour markets.** CS-14 §4.3.6 prices pairs at $2–$15 and RLAIF at $0.04. That price difference is the single largest force shaping what preference data exists in the world, and it is a market outcome. When a team chooses RLAIF because it is 50× cheaper (CS-14 §4.3.6's table), it is also choosing a judge's values over a human pool's — a trade the cost table makes look purely financial.
2. **The buy-vs-build line for alignment.** CS-18 and CS-19 cover hosted fine-tuning where the preference data and the alignment recipe are the provider's. That is a market transaction in values, and the price does not include a disclosure of what was bought.

**What it fails to map onto.** Sandel's market critique is about *corruption* — pricing a good changes its meaning (paying for a friend's help makes it not friendship). The corruption argument does not transfer cleanly to annotation, because annotation is already a paid task and no one claims it is a gift. What *does* transfer is the **crowding-out** half: paying per pair shapes annotator behaviour toward fast, low-ambiguity judgements, which is measurable (CS-14 §4.3.5's per-annotator agreement against the majority) and is a data-quality problem before it is a moral one. **Cite this concept for the incentive analysis, not for the corruption claim.**

---

### 3.15 Virtue, character, and the character of the agent

**What the lecture says.** Aristotle appears only in the syllabus list [00:16:15]. But the theme is present in the Dudley discussion, and it is the one moment where the class evaluates the *person* rather than the act. Speaker 21, explaining why they hold the categorical line even granting consent: *"I don't think that there is any remorse. In Dudley's diary, 'We were eating our breakfast,' it seems as though he's just sort of like, 'Oh'… the whole idea of not valuing someone else's life"* [00:47:17]. And Sandel's summary: *"When he lacks remorse or a sense of having done anything wrong"* [00:47:57].

**What it maps onto.** **Tone, calibration, and the honesty axis** — the parts of a model's behaviour that are about disposition rather than correctness. CS-14 §2.4 defines Honest as *"the model states what it believes, does not fabricate, calibrates uncertainty, and does not strategically mislead"*, and measures it with TruthfulQA, hallucination rate, and calibrations. CS-14 §16.4's `sycophancy_probe_pass` is the metric for a character property that no accuracy measure captures: does the model push back on a false premise?

This is the *only* concept in §3 whose primary mapping is a metric rather than a dataset decision, and it is the mapping most at risk of being over-claimed.

**What it fails to map onto.** **Virtue ethics is about a person who persists through time and can be held responsible. A model is a function.** It has no character, no remorse, and no stake in its own actions; "the model is sycophantic" is a statement about a distribution of outputs, not about a disposition anyone can be praised or blamed for. The honest version of the mapping is narrow: **sycophancy and calibration are real, measurable output properties that the virtue frame draws attention to**, and the frame supplies no further engineering content beyond that attention. Anyone who tells you a model has character is telling you something about their own vocabulary, not about the weights. Of the sixteen concepts here, this is the weakest mapping and it is flagged as such rather than padded.

---

### 3.16 The evasion of skepticism, and philosophy as estrangement

**What the lecture says.** Two related ideas close the episode.

First, the **risk**: *"to read these books in this way, as an exercise in self-knowledge, to read them this way carries certain risks. Risks that are both personal and political"* [00:17:21]. And the political one is stated flatly: *"You have to allow for the possibility that political philosophy may make you a worse citizen rather than a better one, or at least a worse citizen before it makes you a better one. And that's because philosophy is a distancing, even debilitating activity"* [00:20:10]. Callicles's advice, quoted approvingly as a serious objection: *"Abandon argument. Learn the accomplishments of active life… Quit philosophizing. Get real. Go to business school"* [00:21:21].

Second, the **evasion**: *"the name of the evasion is skepticism. It's the idea… maybe it's just a matter of each person having his or her own principles, and there's nothing more to be said about it, no way of reasoning"* [00:21:59]. Sandel's reply is the one to keep: *"the very fact that they have recurred and persisted may suggest that though they're impossible in one sense, they're unavoidable in another. And the reason they're unavoidable, the reason they're inescapable is that we live some answer to these questions every day"* [00:22:35]. And the Kant quotation: *"Skepticism is a resting place for human reason, where it can reflect upon its dogmatic wanderings, but it is no dwelling place for permanent settlement"* [00:23:08].

**What it maps onto.** Both halves are engineering content, in opposite directions.

- **The evasion is the paralysis failure mode** (§9.4). "Everyone has their own values, so there's nothing to be said" is the sentence that ends a design discussion without producing a decision. Sandel's reply is the correct engineering reply: *you are already shipping an answer, because the default is an answer.* Not deciding is a decision with a distribution over its consequences. Concretely: not writing a tie rule means ties are recorded as strict preferences; not stating an HHH priority means the annotator's instinct sets it; not choosing an annotator pool means the vendor chooses.
- **The estrangement risk is the over-correction failure mode.** A team that becomes *worse* before it becomes better — that starts auditing every pair, contesting every guideline, and shipping nothing — has taken the lecture's therapeutic aim into a setting that needs it to settle. The `[00:20:10]` quotation is the licence for §9.4, and the mitigation is deliberate time-boxing (§8, §10 Scenario 5).

**What it fails to map onto.** The distinction between "unavoidable in the sense that we live some answer" and "a decision that must be made by Friday" is doing a lot of work in that sentence, and the lecture does not draw it. In an engineering setting, the second is the operative constraint. **The correct reading for this appendix: reflect enough to make the default explicit, then decide, and write down that you decided.** That is the whole of the practical advice, and everything in §6 is its implementation.
---

## 4. The bridge: from "who decides" to "what gets optimised"

This is the section the appendix exists for. Everything in §3 is philosophy; everything from here is a named artefact.

**The claim to be established:** *the preference-alignment pipeline is a values-encoding machine, and it encodes values at seven identifiable places. Each place has an owner, a default, and a document — and in most projects only two of the seven documents exist.*

The pipeline below is deliberately compressed to one line per stage because CS-14 §4 and §5 already own the mechanism. The right-hand column is the only thing this appendix adds.

```
  (1) PROMPT POOL      which questions exist to be asked
        │              → CS-14 §5.1
        ▼
  (2) SAMPLER          which answers exist to be ranked
        │              → CS-14 §4.6.12, §4.6.1
        ▼
  (3) ANNOTATOR POOL   whose ranking enters the data
        │              → CS-14 §4.3.2, §4.3.5
        ▼
  (4) GUIDELINE        what "better" means, in writing
        │              → CS-14 §4.3.4
        ▼
  (5) THE PAIR         one ranking, frozen, with disagreement deleted
        │              → CS-14 §4.3.1
        ▼
  (6) THE OBJECTIVE    the ranking becomes a gradient
        │              → CS-14 §4.4 (RM), §4.6.3 (DPO), §4.5 (β·KL)
        ▼
  (7) THE EVAL         which claims are checkable, on whom
                       → CS-14 §12.1, §16.4
```

### 4.1 Stage 1 — the prompt pool, or who gets to be counted

**What it is.** The distribution of prompts the preference data is conditioned on. CS-14 §5.1 lists it as a pipeline stage; CS-14 §4.3.2's InstructGPT row describes the original recipe — *"Sample k answers per API prompt, humans rank them"*.

**Where the values enter.** A group whose use cases are not in the prompt pool has no `chosen` response at that prompt and therefore no representation in the gradient. This is the Sandel dependents argument [00:49:05] in its modern form: weight is proportional to presence in the pool, and absence is not a decision anyone made.

**The concrete decision.** Whether the prompt pool is (a) the traffic you have, (b) the traffic you want, or (c) a synthetic set written to cover the second. Most teams use (a) because it is free, and (a) systematically over-represents your existing users — which is precisely the population that least needs the model improved for them.

**Where it is documented.** Almost nowhere. CS-14 §16.2's `preference_data_provenance` field is the slot it *should* go in, and CS-14 §16.3's *"Prompt distribution drift"* row (PSI > 0.2) is the monitoring signal that catches a pool going stale — but neither asks the prior question of whether the pool was ever representative.

### 4.2 Stage 2 — the sampler, or which answers were on the ballot

**What it is.** Which responses were generated and handed to the annotator. CS-14 §4.6.12 (rejection sampling) and §4.6.1 (PPO's exploration advantage) are the technical treatments.

**Where the values enter.** The annotator cannot rank a response that was not sampled. This is §2.3's menu problem and §3.4's "you just wrecked the philosophical point" [00:13:03], and the engineering consequence is stated by CS-14 §4.6.3: DPO is *"offline only — it cannot explore beyond the pairs you give it."*

**The concrete decision.** Whether the sampler is your own model (cheap, bounded by its own distribution, and it will not produce the failure you are trying to fix), a stronger model (imports that model's values — CS-14 §4.3.6's judge-bias-transfer warning), or a human (expensive, and the most likely to produce off-distribution examples worth having).

**Where it is documented.** Partially, via the model ID and version in CS-14 §16.2's manifest. The *diversity* of the sampler — how many distinct responses per prompt, from how many sources — is not.

### 4.3 Stage 3 — the annotator pool, or the "held by someone" slot

**What it is.** The people (or the judge model) whose rankings become the data. CS-14 §4.3.2's eight-row source table is the repository's full treatment of *how* to source; CS-14 §4.3.5's IAA protocol is how to *check* the resulting signal.

**Where the values enter.** This is the strongest entry point in the pipeline and the least documented. CS-14 §4.3.2 says the source matters for cost and quality; it does not say the source matters for **which values**. A pool of 40 contractors in one time zone, working from one vendor's UI, with one cultural read on directness, produces a perfectly consistent ranking of a *particular* population's preferences. The consistency is real — that is exactly what IAA measures — and it is the reason IAA cannot detect this problem. **High agreement on a narrow pool is the signature of a coherent bias, not of a universal truth.**

**The concrete decision.** Who is in the pool, whether the pool's demographics are recorded, and whether any group in the affected population is *absent* from it.

**Where it is documented.** Nowhere in this repository. CS-14 §16.2's `preference_data_provenance` names *"annotator IDs"* — that is identity for audit, not composition. §6.1 is the recommendation.

### 4.4 Stage 4 — the guideline, or the constitution that is not called one

**What it is.** The policy document that defines the ordering. CS-14 §4.3.4 lists the eight fields every guideline must specify, and CS-14 §16.5 makes the normative status explicit: *"Annotation guidelines are policy documents. Version them and keep every version, because they encode your refusal policy. A regulator asking 'how does the model decide to refuse?' is asking about your guidelines, not your β."*

**Where the values enter.** Everywhere, and *visibly* — which is the point. A guideline is the one place in the pipeline where the value judgement is written in prose, in English, reviewable by a non-engineer. It is also the only place where a value can be *changed* by editing a document rather than re-running training.

CS-14 §4.6.11 makes the same observation about Constitutional AI: *"values are written down in a document you can audit and edit, rather than being implicit in a crowd of annotators."* **This appendix's position is that every pipeline has a constitution; CAI is the only recipe that names it.**

**The concrete decision.** The HHH priority order (CS-14 §4.3.4 item 1: *"If an answer is more helpful but less safe, choose the safer one. State it; do not assume it."*), the tie rule (item 2), the length-neutrality clause (item 3), and the refusal-calibration clause (item 6). Each of those four is a value judgement in a sentence.

**Where it is documented.** The best case: a versioned file in the repo. The typical case: a slide deck the vendor kept.

### 4.5 Stage 5 — the pair, or the deletion of disagreement

**What it is.** One row: `{prompt, chosen, rejected}` (CS-14 §4.3.1). The ranking, materialised.

**Where the values enter — and this is the appendix's sharpest technical point.** A pair is a **lossy compression of a distribution**. Whatever the annotators thought, by the time the row is written:

- Ties have been resolved into a strict order, or dropped (CS-14 §4.3.4 item 2).
- Disagreement between annotators has been collapsed to one verdict. CS-14 §4.3.5's IAA protocol measures the disagreement as a *quality statistic* — *"if two competent annotators agree on only 65% of pairs, then 35% of your labels are noise"* — but the 35% disagreement is not stored. It is reported and discarded.
- The **minority ranking** is gone. The pair records that A ≻ B; it does not record that 35% of annotators said B ≻ A.

**The concrete decision.** Whether to store the disagreement. This is a schema change, and it is cheap:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "n_annotators": 3,
  "votes_chosen": 2,
  "votes_rejected": 1,
  "agreement": 0.67,
  "guideline_version": "acme_hhh_v3"
}
```

Those six extra fields — the last three are the ones that matter — turn each pair from a verdict into a *distribution*. Nothing in the training code reads them (DPO and ORPO consume `chosen`/`rejected` and nothing else, CS-14 §4.6.3 and §4.6.9), but they enable every practice in §5.1 and §6.5, and they cost nothing to keep. **A pair without a vote count is an unfalsifiable claim about a population.**

**Where it is documented.** Nowhere. The universal schema in the ecosystem is three fields (CS-14 §4.3.1's `hh-rlhf`, `ultrafeedback-binarized-preferences-cleaned`, `math-step-dpo-10k` are all three-column or two-column), and the field that would let you audit the aggregation is the field nobody writes.

### 4.6 Stage 6 — the objective, or the ranking becomes a gradient

**What it is.** The step where a set of pairs becomes parameters. CS-14 §4.4 (Bradley–Terry: `P(y+ ≻ y−) = σ(r(y+) − r(y−))`), CS-14 §4.6.3 (DPO's log-ratio form), CS-14 §4.5 (the `β·KL` term).

**Where the values enter.** Three places, each with a handbook section:

| Mechanism | The value that gets frozen in | Section |
|---|---|---|
| **The BT/RM fit** | The population's *revealed* ranking, averaged. Note it never references correctness. | CS-14 §4.4 |
| **β (the KL coefficient)** | How much of the SFT model's values survive. Low β = the new preference data wins; high β = the old SFT values win. | CS-14 §4.5 |
| **π_ref itself** | The SFT model's values, which become the anchor. *"A KL anchor against a rude SFT model anchors you to rudeness."* | CS-14 §17 item 8 |

The third is the one practitioners forget. **The most consequential value decision in the alignment stage was probably made in the SFT stage**, because π_ref is what β protects. CS-13 owns that stage; CS-13 §4.6.3 (quality filters) and §4.6.6 (licensing) are where the SFT data's values were set.

**The concrete decision.** β, and whether π_ref is the right thing to be anchored to. CS-14 §4.5's typical range is 0.1–0.5 for DPO; the *choice* of where in that range is a statement about how much you trust your preference data relative to your SFT model.

### 4.7 Stage 7 — the eval, or which claims are checkable

**What it is.** The measurement layer. CS-14 §12.1 (*"the metric stack, and how each one lies"*), CS-14 §16.4 (the CI regression gate with six `KNOWN_GOOD` numbers), CS-14 §16.3 (drift monitoring).

**Where the values enter — and this is where the appendix's recommendation is cheapest and largest.** The aggregate win rate answers *"is the model better on the population my comparison set represents?"* It does not answer *"for whom did it get worse?"* CS-14 §16.4's six numbers are win rate, preference accuracy, MMLU delta, mean length, false-refusal rate, sycophancy probe — **none of them is sliced**. CS-14 §16.4 says it well: *"Every one of those six numbers is a different failure mode… A CI gate on win rate alone misses four of the six."* The same argument extends one level: **a CI gate on unsliced numbers misses whichever population is too small to move the mean.**

CS-14 §16.4's `false_refusal_rate: 0.03` is the closest thing in the repo to a slice-aware metric — it is a rate on a *specific subgroup of prompts* (benign ones) rather than on traffic. That is the pattern: **the metric you want is a rate on a defined population, not a mean over all of them.** §6.2 expands this into a concrete requirement.

### 4.8 What the pipeline does with disagreement — the mechanism, stated plainly

Because §5.1 depends on it, stated once and precisely:

| Stage | What happens to disagreement | Where it is visible |
|---|---|---|
| **Raw annotation** | Exists, in full, per annotator | Only in the labelling tool |
| **IAA measurement** | Summarised as κ / α / raw agreement | CS-14 §4.3.5 — a *statistic*, not the data |
| **Pair construction** | Collapsed to one verdict; the minority is dropped | Not visible |
| **Dataset** | One row per prompt; disagreement is no longer representable | CS-14 §4.3.1 |
| **RM training** | Fit to the majority verdict; disagreement becomes label noise | CS-14 §4.4 |
| **DPO/ORPO** | Gradient pushes toward the majority verdict on every pair | CS-14 §4.6.3, §4.6.9 |
| **Eval** | Win rate over a comparison set; disagreement among raters is reported as a CI | CS-14 §12.1 |

Read the last column top to bottom: the disagreement is progressively *converted* from signal into noise, and by the end of the pipeline it is indistinguishable from annotation error. **That is the mechanism. It is nobody's fault and it is a design choice.**

### 4.9 The full bridge table

The one-table summary of the appendix. Slot = §2's four-slot structure. Stage = §4's seven pipeline stages.

| §2 slot | §4 stage | The value decision | Real field / artefact | Handbook definition | Documented by default? |
|---|---|---|---|---|---|
| (b) outcomes | 1 prompt pool | Whose questions get asked | the prompt distribution | CS-14 §5.1 | No |
| (b) outcomes | 2 sampler | Which answers were on the ballot | the generation config + model version | CS-14 §4.6.12 | Partially |
| **(c) holder** | **3 annotator pool** | **Whose ranking is in the data** | annotator IDs, judge model + version | CS-14 §4.3.2, §16.2 | **No** |
| (a) ranking | 4 guideline | What "better" means | `guideline_version`, HHH priority order | CS-14 §4.3.4 | Sometimes |
| (a) ranking | 5 pair | Ties, disagreement, the minority | `chosen` / `rejected` (and nothing else) | CS-14 §4.3.1 | No |
| (d) procedure | 5 pair | How the ranking was elicited | the labelling UI + the task framing | CS-14 §4.3.4, §4.3.6 | Rarely |
| (a) ranking | 6 objective | Which values get frozen | `beta`, `loss_type`, `lora_rank` | CS-14 §4.5, §4.6.3 | Yes — CS-14 §16.2 |
| (c) holder | 6 objective | Whose values are the anchor | π_ref (the SFT checkpoint) | CS-14 §17 item 8 | Yes |
| (c) holder | 7 eval | Which populations are measured | the eval suites and their slices | CS-14 §12.1, §16.4 | Partially |

**The pattern in the last column is the finding.** The two rows that are *reliably documented* — β and π_ref — are the two that are visible in a training config file, and they are also the two where a value decision is hardest to make by accident. The three rows that are *never* documented are the pool, the pair's disagreement, and the elicitation procedure — which are the three where the decision is made by default.

---

## 5. The four hard problems the lecture surfaces for alignment specifically

Each problem here is one the lecture raises in a form that transfers. Each ends in a statement of what it costs to get wrong.

### 5.1 Aggregating conflicting preferences — whose ranking wins?

**The lecture's form.** The class never agrees on the Dudley case. Sandel counts it: *"there are some who think it's morally permissible, but only about 20%, led by Marcus"* [00:46:12], and the rest split between "wrong without a fair procedure" (Matt, [00:43:44]) and "wrong, full stop" (Mike, [00:48:14]). Three groups, three rankings, one decision, and no aggregation rule on the table.

**The alignment form.** Three mechanisms, and the field uses the first almost exclusively:

| Aggregation rule | How it works | Where it appears | Whose ranking wins |
|---|---|---|---|
| **Majority** | The label is the modal annotator verdict | Pair construction by default; CS-14 §4.3.1's schema cannot express anything else | The modal annotator's — and the margin is invisible |
| **Mean / expectation** | The RM fits `E[σ(Δr)]` over all pairs | CS-14 §4.4 (BT is exactly this) | The population mean; intensity of feeling is averaged away |
| **Lexicographic / constrained** | Some dimension is a gate; others are optimised subject to it | CS-13 §12.3 (refusal), CS-14 §16.4 (`KNOWN_GOOD` gates) | The gate-setter's, on the gated dimension only |

**What is lost under majority.** Two things, and both are measurable:

1. **Intensity.** Matt at [00:43:44] holds his position with more conviction than a marginal voter; nothing in a majority rule notices. In annotation, this is the difference between an annotator who finds a response mildly preferable and one who finds it unsafe. CS-14 §4.3.4 item 7's escalation path is the only place the pipeline can express "this pair is not a 51/49 pair" — and it expresses it by *removing* the pair rather than weighting it.
2. **The minority's identity.** A 65/35 split on a pair is stored as a strict preference. It does not record that the 35% were systematically drawn from one region, one language, or one job function. §4.5's six-field extension is the fix.

**What it costs to get wrong.** CS-14 §4.3.5 states the cost as a ceiling: *"If two competent annotators agree on only 65% of pairs, then 35% of your labels are noise, and no amount of training fixes that — you are fitting a coin flip."* That is the *statistical* cost. The *distributional* cost is that the 35% stop being represented in the model's behaviour, and no metric in CS-14 §12.1 will show it.

**The honest position.** There is no correct aggregation rule. Majority is the field's default because it is the simplest to implement and the easiest to defend ("most people preferred it"), not because it is right. What is *not* defensible is using majority without knowing whose majority it is.

### 5.2 Preference vs welfare — a satisfied ranking is not a benefit

**The lecture's form.** Sandel's dependents argument [00:49:05] is the clearest case: the utilitarian case *for* the killing is built entirely out of the preferences and welfare of "everybody" — the three survivors, their wives and children, the wider society that benefits from their productivity. Parker has no dependents, so no welfare attributed to him appears in the sum. His own preference is not consulted at all.

**The alignment form.** Three distinct gaps, and they must be kept separate because they have different fixes:

| Gap | Statement | Measurement that catches it | Where |
|---|---|---|---|
| **Preference ≠ welfare** | A user can prefer a response that harms them (engaging, confident, sycophantic) | Sycophancy probe; calibration; user outcomes over time | CS-14 §16.4 (`sycophancy_probe_pass`) |
| **Stated ≠ revealed** | What users click is not what they would choose with time and information | Regret signals: re-asks, regenerations, abandons | CS-14 §4.3.2 (*"Implicit feedback"* row) |
| **Aggregate welfare hides the tail** | Mean satisfaction can rise while a subgroup is harmed | Disaggregated win rate; per-slice floors | CS-14 §16.3's monitoring table (unsliced) |

The second is the one the field most often gets wrong, and CS-14 §4.3.2 already flags the mechanism: implicit feedback is *"Very noisy, strongly confounded"* and should be used as a *"Supplement only; never sole source."* The confound is exactly the preference/welfare gap — the responses users engage with most are often the ones that flatter them.

**And the sycophancy case is the one where alignment actively creates the harm.** CS-14 §2.4's Honest row: *"Optimising for 'sounding confident' via a preference signal directly attacks honesty."* A preference dataset in which `chosen` is the confident answer and `rejected` is the hedged one teaches sycophancy, and the win rate against a human comparison set will *rise* while it happens, because human raters prefer confidence too. This is the clearest concrete example in the whole handbook of a preference being satisfied while welfare falls.

**The lecture's contribution here is the vocabulary, not a method.** Sandel's dependents argument gives you the question to ask of any aggregate metric: *who is not in this sum?* There is no engineering technique in the lecture. There is a discipline: name the population, then check the metric on it.

### 5.3 The is/ought gap in a reward model

**The lecture's form.** Sandel's sharpest exchange is with Mike at [00:50:02]:

> *"Do you think Bentham is wrong to say the right thing to do is to add up the collective happiness? You think he's wrong about that?"*
> *"I don't think he's wrong, but I think murder is murder in any case."*
> *"Well, then Bentham has to be wrong. If you're right, he's wrong."*

Sandel's move is to force the collision between a *descriptive* claim (people do seek pleasure and avoid pain) and a *normative* one (therefore the right thing is to maximise the sum). Mike's "Okay, then he's wrong" [00:50:10] accepts the normative claim's priority over the descriptive one. **Bentham's inference from the two sovereign masters [00:28:15] to the principle of utility [00:28:54] is the founding instance of the is/ought move.**

**The alignment form, exactly.** A Bradley–Terry reward model (CS-14 §4.4) fits:

```
P(y+ ≻ y− | x) = σ( r_φ(x, y+) − r_φ(x, y−) )
```

There is no term in that equation that references correctness, truth, welfare, intent, or right. It is a **maximum-likelihood fit to a set of observed rankings**, i.e. a descriptive model of what annotators did. Every subsequent step — `max E[r] − β·KL` — treats `r_φ` as the thing to maximise, i.e. as a normative target. **The is/ought transition happens at exactly the point where you write `max` in front of `E[r]`.** Nobody signs it; it is an `argmax` in a training script.

**Why this is not a pedantic point.** Because the two claims come apart *empirically and measurably*:

| The RM is good at | The RM cannot do | Evidence |
|---|---|---|
| Predicting the majority verdict on held-out pairs | Telling you whether the majority verdict is right | CS-14 §4.4's loss is on pairs only |
| Ranking responses the annotators have seen the like of | Ranking off-distribution responses | CS-14 §4.4's distribution-shift warning |
| Being optimised against | Staying valid under that optimisation | CS-14 §4.9 (Goodhart) |

The third row is the is/ought gap's engineering consequence. CS-14 §4.9 documents reward hacking as an empirical phenomenon, and the philosophical reading is precise: **a descriptive model optimised hard enough stops describing.** CS-14 §4.5's three jobs for the KL term include *"Reward-hacking brake — keeps the policy in the region where the reward model is valid."* The KL term is, in this reading, the *epistemic* constraint: it keeps the policy where the description still holds.

**The practical upshot.** Three sentences to put in a design doc:

1. The RM is a description of a population's revealed ranking. Report it that way.
2. Any claim that the model is *better* requires a measurement outside the RM — human comparison (CS-14 §12.1's Arena Elo / win rate), capability suites (CS-14 §16.4's MMLU delta), or task outcomes.
3. Never report RM score as a quality metric. CS-14 §3 says it directly of the `RM score` glossary entry: *"**The metric being gamed.** Never report it as a quality metric."*

### 5.4 Distributional harm — who bears the cost of an average-optimal policy

**The lecture's form.** Parker. He is 17, an orphan, on his first long voyage, and he went *"rather against the advice of his friends"* [00:30:23]. He has no dependents, no constituency, and no vote. Sandel supplies the utilitarian justification himself — *"Parker was an orphan. No one would miss him"* [00:50:46] — and the class does not flinch. That sentence is the entire mechanism of distributional harm in one line: **a person with no constituency has zero weight in the sum, and nobody decided that.**

The 9/11 remark at [00:03:02] is the same structure from the other end: the passengers who brought the plane down are *"heroes"* precisely because the arithmetic that sacrificed them was performed over a larger population that did not include them as beneficiaries.

**The alignment form.** A policy that maximises mean preference satisfaction can impose its cost on a subgroup too small to move the mean. Concretely, four mechanisms, all live in this repo's own material:

| Mechanism | How the harm lands | Handbook instance |
|---|---|---|
| **Aggregate eval** | A win-rate gain on the mean is reported; the losing slice is not measured | CS-14 §16.4's six `KNOWN_GOOD` numbers are unsliced |
| **Length bias** | Verbose answers win on average; users who needed brevity lose | CS-14 §4.9's length bias; CH-14 §5.4's audit |
| **Refusal calibration** | Over-refusal is cheap on average and expensive for the group whose requests are systematically misread as unsafe | CS-14 §16.4's `false_refusal_rate` on benign prompts; CS-13 §12.3 |
| **Judge bias** | A judge model's style preferences become the reward, so the populations whose style differs lose | CS-14 §4.3.6's judge-bias-transfer warning |
| **Language and register** | If the prompt pool and the annotator pool are monolingual, every other language's ranking is unrepresented | §4.1, §4.3 — not covered in CS-14 |

The refusal case deserves one more line because it is the clearest. CS-13 §16.5 warns that fine-tuning *"measurably weakens safety behaviour"*; the opposite error — over-refusal — is the one that lands selectively. A model that refuses more on some phrasings than others has not become safer; it has become safer **for the populations whose phrasings it recognises and less useful for the ones it does not.**

**The mitigation is a floor, not a mean.** Concretely, §6.2: every headline eval reports a per-slice breakdown, and at least one slice is a *floor* asserted in the CI gate rather than a number inspected in a dashboard. CS-14 §16.4's `KNOWN_GOOD` dict is the exact place to put it — it already asserts on six numbers; a seventh, per-slice, is a two-line change.

**What the lecture contributes.** It supplies the *reason* the floor is legitimate. Sandel's Question One [00:51:27] — *"even cabin boys have certain fundamental rights… where do those rights come from, if not from some idea of the larger welfare"* — is the acknowledgement that no aggregation can generate a floor from inside itself. The floor must come from outside the objective. In engineering, "outside the objective" means: a gate, a filter, a policy, or a legal requirement. **A floor expressed as a penalty term is not a floor; it is a price.**

---

## 6. What practitioners actually do about it

Five practices. Each is concrete, checkable, and cites the mechanism already in this repo where one exists. **All five are mitigations.** None of them resolves the underlying question, and §6.6 says so.

### 6.1 Annotator demographic disclosure

**The practice.** Record and disclose the composition of the annotator pool along the axes that plausibly affect the ranking: language(s), region, domain expertise, and whether the annotators are also users of the product. Publish it in the dataset card and the model card next to the data provenance. Where the pool is a judge model instead, disclose the judge — CS-14 §4.3.6 already requires this for a different reason (*"use a judge from a different model family than the one you are aligning"*).

**Where it goes in the repo.** CS-14 §16.2's `preference_data_provenance` field currently reads *"annotator IDs, judge model + version, guidelines version"*. The disclosure adds *composition*, not identity. CS-14 §16.5 already establishes that the guidelines are a policy document; the pool composition belongs beside them.

**What it buys.** It converts an invisible default into a stated fact. The most common discovery when a team does this for the first time is that the annotator pool is not a sample of the user population and was never intended to be — which is fine, and is a different thing from the pool being *unexamined*.

**What it does not buy.** Representativeness. Disclosing that the pool is 92% one country does not make the ranking correct; it makes it *attributable*. §10 Scenario 1 is this practice doing its actual job, which is to let you decide whether the gap matters for this product.

**A caution about the law.** Demographic data about workers is regulated in most jurisdictions, and collecting it can be the wrong call. The disclosure that matters is at the *aggregate* level (composition of the pool), not the individual level, and it can usually be obtained from the vendor without per-person records.

### 6.2 Disaggregated eval

**The practice.** Every headline alignment metric reports a per-slice breakdown, and at least one slice is enforced as a **floor** in the promotion gate rather than inspected in a dashboard.

Concretely, extending CS-14 §16.4's `KNOWN_GOOD`:

```python
# tests/test_alignment_regression.py — the §16.4 block, extended in one dimension
KNOWN_GOOD = {
    "win_rate_vs_sft": 0.58,
    "preference_accuracy_heldout": 0.71,
    "mmlu_delta": -0.008,
    "mean_length_tokens": 212,
    "false_refusal_rate": 0.03,
    "sycophancy_probe_pass": 0.90,
    # ↓ the slice block. Floors, not means.
    "win_rate_slice_short_prompts": 0.52,      # the brevity-needing slice
    "win_rate_slice_non_en": 0.50,             # the non-English slice — a FLOOR
    "win_rate_slice_low_literacy": 0.50,       # plain-language slice — a FLOOR
    "false_refusal_slice_multi_turn": 0.05,    # over-refusal is worse in follow-ups
}
```

**Why floors and not means.** A mean is a claim about a population; a floor is a claim about a *member* of it. §5.4's mechanism is that the mean can rise while a slice falls, so a gate on the mean is structurally incapable of catching the failure. Asserting `>=` per slice is the engineering form of the "no aggregation generates its own floor" point at [00:51:27].

**Where it goes in the repo.** CS-14 §16.4 (the gate) and CS-14 §16.3 (the monitoring table, whose *"Prompt distribution drift"* row is the only slice-aware line today). CS-02 §12.1's five-number protocol already includes an *"OOD slice metric"* — *"A second eval set from a different source, annotator, or time period"* — which is the same idea for a different stage.

**What it does not buy.** Slices you did not think of. A slice has to be named before it can be measured, and naming it is the value judgement. §3.13's veil heuristic is the technique for generating the list; §6.4's red-teaming is the technique for finding the ones you missed.

**The cost.** Real. Each slice needs its own held-out set with enough items for a confidence interval — a 200-item slice gives you roughly ±7 points at 95%, which is too coarse for a floor. Budget 300–500 items per enforced slice, and expect the eval set to grow by 2–3× for three slices.

### 6.3 Documenting the preference-elicitation protocol

**The practice.** Treat the procedure as a versioned artefact, not as a vendor's process. The document states, at minimum:

1. **Framing.** The exact instruction given to the annotator (CS-14 §4.3.4's guideline is the *content*; this is the *prompt*).
2. **UI and options.** Pairwise, k-wise, Likert, thumbs. Whether ties are offered. Whether the annotator may write a rationale.
3. **Incentives.** Paid per pair, per hour, or salaried — because §3.14's crowding-out applies.
4. **Sampling.** How pairs were selected for annotation, and how many annotators saw each one.
5. **The change log.** When any of the above changed, and which slice of the dataset is on which side of the change.

**Where it goes in the repo.** CS-14 §16.2's versioning block is where it belongs (`preference_data_provenance`); CS-14 §16.5's third bullet (*"LLM-judge provenance must be recorded"*) is the same requirement for the judge case; CS-13 §16.1's `MANIFEST.json` is the pattern — *"dataset id, semver version, n_train/n_eval, sha256 of each file, source breakdown, filters applied, template sha256, seed, parent version, notes"*.

**Why item 5 is the one people skip.** Changing the UI mid-collection changes what is measured while keeping the column names. The dataset looks homogeneous and is not, and the resulting model has a behaviour nobody can explain. CS-14 §10's gotchas list has an analogous item (item 18, the chat-template trap: train with one format and serve with another) — same class of bug: a silent format change with no error.

**What it does not buy.** Validity. A perfectly documented procedure can still be measuring the wrong thing. Documentation is what makes the *question* reviewable; it is not an answer.

### 6.4 Red-teaming the values the majority holds

**The practice.** Adversarially search for cases where the majority's ranking is the harmful one, and add them to the eval set — and, where the fix is clear, to the preference data.

**Why this is a distinct activity from safety red-teaming.** Safety red-teaming (CS-01 §16.4 item 4, CS-13 §16.5's safety row, CS-14 §4.9's RM red-teaming) asks *"can the model be made to do something dangerous?"* This asks a different question: *"where is the model confidently doing what most people said, and hurting a few?"* The second is invisible to the first, because the behaviour is popular.

**Three concrete probes.** Each is a generator of eval items, not a one-off test:

| Probe | What it finds | How to build it |
|---|---|---|
| **Style inversion** | Pairs where the majority-preferred answer is verbose, hedging, or listy, and the minority-preferred is direct and short | Take the 200 pairs where `chosen` is longest, invert them, and check the model's win on the inverted set |
| **Register and dialect** | Over-refusal and lower quality on non-standard phrasing | Rewrite a benign eval set in AAVE, in Indian English, in a low-literacy register; measure false-refusal rate per register |
| **Confident-but-wrong** | Sycophancy: agreement preferred over correction | CS-14 §16.4's `sycophancy_probe_pass` is the same instrument; extend it with domain-specific false premises |
| **Preference-versus-outcome** | Cases where the preferred answer leads to a worse user outcome | Requires product telemetry, not annotation — the expensive one, and the one that finds the real harm |

**Where it goes in the repo.** CS-14 §4.9's *"Red-teaming the RM directly"* row is the analogue for the reward model; CS-14 §16.4's CI gate is where the resulting items are enforced; IQ-14 Q93's week-1 step (*"Sample 200 real conversations, hand-label the failure mode of each"*) is the discovery method that surfaces the candidates.

**What it does not buy.** Coverage. You find the values your team thought to question. A team drawn from one population will systematically fail to generate the probes that population does not need. This is the strongest argument in the appendix for §6.1 — the disclosure is what tells you which probes you cannot generate yourself.

### 6.5 Keeping a minority-ranking audit

**The practice.** Retain the per-annotator votes (§4.5's six-field extension), and periodically *inspect the pairs with the highest disagreement* rather than only reporting the aggregate κ.

CS-14 §4.3.5's calibration recipe already double-annotates 10% permanently. The extension is to look at what the 10% disagree about:

```python
# A quarterly audit, ~20 lines of pandas. The output is a meeting agenda, not a metric.
# df: prompt, chosen, rejected, n_annotators, votes_chosen, votes_rejected
contested = df[(df.votes_chosen >= 1) & (df.votes_rejected >= 1)]
by_topic = contested.groupby("topic").agg(
    n=("prompt", "size"),
    mean_split=("votes_chosen", lambda v: (v / contested.loc[v.index, "n_annotators"]).mean()),
).sort_values("n", ascending=False)

# The two questions to ask of the top rows:
#   1. Is the split a *topic* effect (this subject is genuinely contested) → sharpen the guideline.
#   2. Is the split a *population* effect (the pool disagrees with itself) → the pool is not one
#      population, and the majority verdict is a coin-flip dressed as a label. CS-14 §4.3.5.
```

**Where it goes in the repo.** CS-14 §4.3.5 (*"measure agreement by category — you will almost always find that safety pairs agree at 92% and style pairs at 68%"*) is the same audit at the category level; this adds the *split-direction* question (who lost, not just how often).

**Why it is worth doing.** It distinguishes two situations that look identical in an aggregate κ: a topic that is genuinely contested (fix the guideline), and a pool that contains two populations (fix the pool, or accept that you are choosing one). Sandel's class was exactly the second case, and it is why he could not extract a principle from them [00:07:10].

### 6.6 The honest framing: these are mitigations

Four things none of the above does:

- **It does not make the model's values legitimate.** It makes them *attributable*. Attribution is a prerequisite for accountability, and it is not accountability.
- **It does not resolve §5.1.** There is still one ranking per pair and no rule for combining them that everyone would accept.
- **It does not close §5.3.** A better-documented description of what annotators did is still a description.
- **It does not substitute for §5.4's floor.** A disaggregated metric *detects* distributional harm; the gate is what *prevents* the harm from shipping.

The reason to do all five anyway is not that they solve the problem. It is that the alternative is a pipeline whose value decisions are made by whoever wrote the fastest code that week. **Sandel's reply to skepticism is the same reply:** *"the reason they're unavoidable, the reason they're inescapable is that we live some answer to these questions every day"* [00:22:35].

---

## 7. Where this appendix stops

The boundary. Every question below is one this file has deliberately not answered, with the module that owns it. **Every section reference in this table was verified against the file with grep; §13 lists the verification.**

| Question this appendix does NOT answer | Why it is out of scope here | Owning module and section |
|---|---|---|
| How does RLHF/PPO actually work, mechanically? | Re-teaching it would violate the manifest's scope note | CS-14 §4.6.1–§4.6.2 |
| How is the DPO loss derived? | This is a bridge, not a derivation | CS-14 §4.6.3; CS-25 (not yet written) |
| What are the memory costs of each method? | Pure engineering | CS-14 §4.7; CS-14 §11.1 |
| How do I write a preference annotation guideline? | The eight required fields are already specified | CS-14 §4.3.4 |
| How do I measure inter-annotator agreement? | κ, α, Spearman, and the calibration recipe are already specified | CS-14 §4.3.5 |
| What does a preference dataset's schema look like? | The three-field schema and its variants are already specified | CS-14 §4.3.1; CS-14 §4.3.3 |
| Where does preference data come from, and what does it cost? | The eight-source table | CS-14 §4.3.2; CS-14 §4.3.6 |
| How do I stop reward hacking? | A ranked mitigation list exists | CS-14 §4.9.3 |
| What is the alignment tax and how is it mitigated? | The PPO-ptx treatment | CS-14 §4.8 |
| How do I detect a silent alignment failure? | The symptom table | CS-14 §14.2; CS-14 §9.4 |
| What is a model card, and what goes in it? | A worked template exists for domain models; the alignment artefact manifest for adapters | CS-12 §16.7 (the model card template); CS-14 §16.2 (the alignment-artefact manifest) |
| What are the legal regimes governing training data? | The four regimes and the provenance ledger | CS-12 §4.12.1; CS-12 §4.12.2 |
| What is the source-rights checklist for training data? | Owned by the domain-adaptation module | CS-12 §16.6 |
| How do I handle PII and right-to-erasure in SFT data? | Owned by the SFT module | CS-13 §16.5 |
| How is the alignment artefact versioned and monitored in production? | Owned by the alignment module | CS-14 §16.2; CS-14 §16.3 |
| How do I audit a reward model's failure modes? | The RM profile and the over-optimisation evidence | CS-14 §4.6.2; CS-14 §4.9.2 |
| How do I run a win-rate evaluation, and how does it lie? | The metric stack | CS-14 §12.1; CS-14 §12.2 |
| What is the compliance angle for a fine-tuned model generally? | Owned by the foundations module | CS-01 §16.5; CS-01 §16.4 |
| What is the dataset versioning pattern? | `MANIFEST.json` and the three rules | CS-13 §16.1 |
| Is RLHF deprecated? | The misconception and its correction | CS-14 §17 item 3 |

**One thing this appendix would have handed off but cannot.** CS-14 §20 lists *"CS-24 (RL Fundamentals & RLHF with PPO — the deep dive on §4.6.1–4.6.2), CS-25 (DPO …), CS-26 (GRPO …), CS-27 (ORPO …)"* as the deep-dive owners of the method profiles. Those four modules are listed in the README's curriculum table as not yet written. Until they exist, **CS-14 §4.6 is the deepest treatment in the repository** for every method referenced here, and this file's method links deliberately point there rather than at empty targets.
---

## 8. Decision framework — when does this ethical question actually change what you build?

The most useful table in the appendix, because it is the one that stops the discussion. **Most ethical questions in a fine-tuning project do not change the build.** The ones that do are detectable by a specific test: *does the answer change a row of data, a number in a config, a threshold in a gate, or a document in the repo?* If not, it is a conversation, and conversations have a place — but it is not blocking a release.

| Situation | Changes what you build? | What changes | Skip it if |
|---|---|---|---|
| The annotator pool is drawn from one country and the product ships in twelve | **Yes** | Slice the eval set; add a per-language floor; disclose composition | The product is genuinely single-region and the roadmap has no plan for more |
| A pair's two responses are near-identical and annotators split 50/50 | **Yes** | Drop the pair, or use cDPO with ε = the measured noise rate (CS-14 §4.6.5) | Never — this is a data bug before it is an ethics question |
| The guideline forbids ties | **Yes** | Change the guideline; allow ties and drop or smooth them (CS-14 §4.3.4 item 2) | Never |
| Users prefer an answer that measurably harms them | **Yes, and expensive** | Add a sycophancy/calibration probe; weight the eval toward outcomes, not preference | You have no telemetry and no way to measure the outcome — then log the question and revisit |
| The reward model scores a bad response highly | **Yes** | Adversarial pairs; retrain; early-stop on a capability suite | Never — CS-14 §4.9's mitigations apply directly |
| Mean win rate up, one slice down | **Yes** | Per-slice floor in the CI gate (§6.2) | The slice is under 300 items — you cannot assert on it; grow it first |
| "Is it right to build this product at all?" | **Yes — but not here** | Nothing in this file changes; it is a product-ethics decision with legal and business inputs | It is out of scope for the alignment stage entirely |
| "Do models have moral status?" | No | Nothing | A research question with no effect on any artefact in this repo |
| "Which ethical theory is correct?" | No | Nothing | 2,400 years old and not blocking your sprint |
| "Should we use RLAIF instead of human labels?" | **Yes** | Judge provenance, cross-family judge, human-audited holdout (CS-14 §4.3.6) | You already have a human-labelled holdout that the judge is measured against |
| The prompt pool is production traffic only | **Yes, if the users are not the intended users** | Add a synthetic or solicited prompt set for the target population | The product's users *are* the traffic, exactly |
| We changed the labelling UI mid-collection | **Yes** | Version the dataset; mark the boundary; consider training only on the second phase | The change was cosmetic (a layout tweak with identical semantics) |
| Legal flags the dataset's licence | **Yes, and it is a gate** | Do not train. See the licence table | Never — this is a STOP condition, not a discussion |

**The test, stated once.** Ask: *if we answer this question the other way, what file changes?* If the answer is "a config value", "a dataset row", "a threshold", "a versioned document", or "nothing ships", the question is actionable. If the answer is "the team's understanding", it is worth twenty minutes and not a week.

**Two STOP conditions, in the CS-14 §8.3 spirit:**

- **STOP** if the discussion has run for more than one working day and no artefact has changed. §9.4's paralysis failure mode has started; time-box it and default to the documented option.
- **STOP** if the answer depends on a metric you do not have and cannot get. Log the question with the metric that would settle it, ship the documented default, and revisit — do not block on an unmeasurable claim.

---

## 9. Pros · cons · limitations · failure modes

### 9.1 Pros — what the philosophical frame buys you

| Pro | Mechanism | Evidence |
|---|---|---|
| **It names decisions that are otherwise invisible** | Four slots (§2.1) turn "we have a preference dataset" into four questions with owners | §2.6's table; the pattern in §4.9's last column |
| **It supplies vocabulary that survives a reorg** | "Whose majority is this?" is a sentence a non-engineer can act on; "our annotator pool is 92% one region" is a fact | §6.1 |
| **It generates the slices you would not have thought of** | The veil heuristic (§3.13) and the probes in §6.4 are systematic generators, not intuitions | §6.2 |
| **It justifies the floor as a floor** | Sandel's Question One [00:51:27] is the argument that an aggregate cannot generate its own constraint, which is why slice gates are `>=` and not penalties | §5.4, §6.2 |
| **It predicts the architecture** | The consequentialist/categorical split predicts the industry's split between losses and gates (§3.6, §3.7) — a real, checkable prediction | §3.7's failure line |
| **It makes the is/ought transition visible** | §5.3 locates it at the `argmax`, which is a one-line change to how a metric is reported | CS-14 §3, `RM score` entry |
| **It is cheap** | Roughly one day of attention and fifty lines of a design doc (§1.2) | — |

### 9.2 Cons — what it costs

| Con | Mechanism | Mitigation |
|---|---|---|
| **Unbounded** | The literature does not terminate and the sprint does | §8's test; the STOP conditions |
| **Changes no loss curve** | Nothing here sets a hyperparameter correctly | Accept it — the contribution is framing, and framing is bounded |
| **Measurable problems are better solved by measurement** | IAA (CS-14 §4.3.5) and reward hacking (CS-14 §4.9) both want instrumentation, not argument | Where a measurement exists, measure |
| **Vocabulary without mechanism** | "Consent" and "welfare" are easy to say and do not compile | Every §3 concept terminates in a named artefact or is flagged (§3.15) |
| **The lecture is not a technical source** | It is an undergraduate lecture | Take the questions; leave the answers |
| **It can be captured as theatre** | A Values Statement shipped with no data, metric, or gate change | §1.3's audit list |
| **It imports a vocabulary that can be used to launder decisions** | "We used the veil of ignorance" is not a demographic disclosure | §6.1 requires the disclosure, not the phrase |

### 9.3 Hard limitations — things this frame cannot do

1. **It cannot tell you what your model should value.** It has no normative output; it has a normative *inventory*. The decision remains a decision.
2. **It cannot be verified.** There is no test that passes when your values are correctly encoded. Every checkable thing in §6 is a check on *documentation* and *distribution*, not on legitimacy.
3. **It cannot reach the affected population.** The users who bear the cost of a ranking are not in the annotation loop, the eval loop, or the design meeting. The veil (§3.13) is an imaginative substitute for consulting them, and a weak one.
4. **It cannot resolve disagreement, only surface it.** §5.1's aggregation problem is not solved here or anywhere. Majority is a convention, not a derivation.
5. **It cannot survive translation into a single number.** Every attempt to summarise the values position in a metric produces a proxy, and proxies are gamed (CS-14 §4.9).

### 9.4 Silent failure modes — looks fine, is broken

| Failure mode | What it looks like | Why it is silent | Detection |
|---|---|---|---|
| **Paralysis** | Every decision is contested; nothing ships; the roadmap slips | Nobody can point to the decision that was not made, because no artefact is missing that anyone expected | A release date that has moved twice with no gate failing; §8's one-day STOP condition |
| **Theatre** | A Values Statement, an ethics review, an AI principles page | The documents exist and are good | §1.3's audit: which dataset row, metric, or gate changed? If the answer is none, this is it |
| **Documentation without effect** | An `annotator_pool` field that is populated and never read | The field is present, so an audit passes | Check whether any slice, floor, or gate was *derived* from it. If not, it is decoration |
| **The single-population pool** | Excellent IAA; a coherent model; enthusiastic internal reviews | High agreement *looks like* a strong signal — CS-14 §4.3.5's own framing treats agreement as the ceiling on quality, which is true and is also exactly what a coherent bias produces | §6.1's disclosure; §6.5's split-direction audit |
| **The well-meaning override** | An engineer adds a safety pair to `chosen` that the guidelines do not support, "because it's obviously right" | One pair, no version bump, no note | Guidelines version vs dataset hash mismatch (CS-14 §16.2) |
| **The floor that became a price** | A distributional constraint implemented as a penalty term in the loss | It works, mostly, until the reward is large enough to pay the penalty | Anything with a weight is payable. Ask: *what reward would buy this?* If the answer is finite, it is a price |
| **Metric substitution** | RM score reported as quality | CS-14 §3 says not to; a dashboard is easier than an eval suite | Search the reporting code for `reward` used as a headline |
| **The borrowed constitution** | A CAI constitution or a guideline copied from a public example with no local review (CS-14 §4.6.11) | It is a well-written document, so it reads as considered | `git log` on the guideline file; if the first commit is an import, it was never reviewed |

**The two that matter most are paralysis and theatre**, and they are opposites. Paralysis is a team that takes the philosophy seriously and cannot act; theatre is a team that takes the vocabulary and does nothing. §8's decision table is the instrument against both: it forces every question into either "changes an artefact" or "changes nothing", and it time-boxes the second.

---

## 10. Applied scenarios

Five worked scenarios at realistic scale. Each ends in a concrete change to a dataset, a metric, or a gate — stated as a diff.

### Scenario 1 — The tone alignment where the annotator pool was one country

**Situation.** A B2B SaaS company ships a support assistant in 11 languages. The DPO stage (CS-14 §4.6.3) uses 6,200 preference pairs, of which 5,400 are English. The pair set was built by a 12-person contract team, all based in one metropolitan area, working from an English guideline. The model is deployed globally; win rate against the SFT baseline is **0.61** on the English comparison set, and **0.54** on a small German set — a number nobody has looked at because the German set has 180 items and is reported as a footnote.

**The value question.** CS-14 §4.3.2's source table asks *how* the data was sourced and *what it costs*. It does not ask who the annotators are. The pool is not a sample of the user population; it is a sample of whoever the vendor employs. §6.1.

**Why the standard checks do not catch it.** Inter-annotator agreement is **82% raw** (CS-14 §4.3.5's "good" band), the guideline is versioned and detailed, and the reward hacking checks are clean. Every quality signal in CS-14 §12.1 is green. The problem is not quality; it is that a coherent single-population ranking was learned, and coherence is what IAA measures.

**What the lecture contributes.** §2.4's reading of the dependents argument [00:49:05]: the populations absent from the pool have weight zero, and nobody decided that. Sandel's class is not wrong about the trolley — it is *one* class, and its verdict is being applied to everyone.

**The diff.**

1. **Dataset:** add three fields to every pair — `annotator_region`, `elicitation_language`, `guideline_locale` (§4.5's schema extension). Do not backfill; mark existing rows `legacy_unknown`.
2. **Eval:** grow the German set from 180 to 400 items (the CI-width floor from §6.2), and add a non-English slice to `KNOWN_GOOD` with `win_rate_slice_non_en: 0.50` as a **floor**, not a target.
3. **Gate:** promotion requires `win_rate_slice_non_en >= 0.50` **and** the English win rate. A +7-point English gain no longer promotes a −5-point non-English regression.
4. **Disclosure:** add annotator composition to the model card next to the data provenance. This is the change that matters most, because it is the one that survives the next reorg.

**Cost.** Two weeks of one engineer's time plus ~$4,000 for 220 additional German pairs and a re-annotation pass. The gate will block at least one promotion. That is the gate working.

---

### Scenario 2 — The safety pair set where the majority ranking harms a minority of users

**Situation.** A consumer product adds 900 hand-written safety pairs to a 12,000-pair DPO set, because legal asked for "a firmer refusal posture". The pairs are written by the product team from real incidents: each `chosen` is a refusal, each `rejected` is a compliant answer. After training, the true-refusal rate on a harmful set rises from 0.81 to 0.94 — good — and the **false-refusal rate on benign prompts rises from 0.03 to 0.08**. CS-14 §16.4's CI gate asserts `false_refusal_rate: 0.03` and would have caught it, so the team raises the baseline to 0.08 and ships.

**The value question.** CS-14 §2.4 defines Harmless and Helpful as *"three axes in tension"* and states that the trade-off point *"is a product decision, not a training decision"*. The team made that decision by editing a number in a test file, with no record that a decision had been made. §3.9's aggregation problem: a mean false-refusal rate of 0.08 can be 0.04 for the majority's phrasing and 0.20 for everyone else's.

**What the standard checks do not catch.** True-refusal is up, which is the metric legal asked for. The aggregate false-refusal rate is reported and moved by a known amount. Nothing is wrong by any measure in CS-14 §16.4 — because the measure is unsliced.

**What the lecture contributes.** §3.2's doing/allowing distinction as a *policy*, not a term: the safety constraint should have been expressed as a gate on the harmful set, leaving helpfulness to be optimised subject to it, rather than as 900 pairs pushing the whole distribution toward refusal. Adding pairs is a *price*; a gate is a *constraint*. §5.4's "a floor expressed as a penalty term is not a floor" is exactly this bug.

**The diff.**

1. **Dataset:** split the 900 safety pairs by *what they have in common*. If 600 of them are "the user's phrasing is indirect", the model is being taught to read indirect phrasing as a threat signal. Add 300 *benign indirect* pairs with `chosen` = compliance.
2. **Metric:** add `false_refusal_slice_register` — false-refusal rate measured separately for the top three phrasings in your traffic. Assert a floor on the *worst* slice, not the mean.
3. **Gate:** revert `false_refusal_rate` to 0.03 as the **aggregate** threshold and add `false_refusal_slice_worst <= 0.06`. If true-refusal must rise, buy it with targeted pairs, not with a global posture shift.
4. **Process:** any change to a `KNOWN_GOOD` threshold requires a one-line note in the run record stating which decision changed and who made it (CS-14 §16.2's manifest is the place).

**Cost.** 300 pairs (~$900 at CS-14 §4.3.6's $3/pair) and one extra eval slice. The harder cost is cultural: making a test threshold a reviewable decision rather than a number you edit.

---

### Scenario 3 — The reward model that learned confidence instead of correctness

**Situation.** A legal-tech company aligns a 8B model to answer questions about contract clauses. The preference data is 4,500 pairs from three domain experts. After DPO, the win rate against the SFT model is **0.67** on a held-out expert comparison set. Two months later, a customer reports that the model cites clause numbers that do not exist in the attached contract — confidently, in a well-formatted table.

**The value question.** §5.3. The RM was trained on `P(y+ ≻ y−) = σ(r(y+) − r(y−))` (CS-14 §4.4) and it is doing exactly what it was trained to do: predicting which response the three experts ranked higher. The experts, reading pairs quickly, ranked the *confident, well-structured* answer above the *hedged, uncertain* one — not because confidence is correct, but because in a pairwise reading task confidence reads as competence. The model learned confidence, not correctness, and the descriptive model was then maximised.

**Why the standard checks do not catch it.** CS-14 §16.4's six numbers: win rate is up (0.67), preference accuracy on held-out pairs is 0.74, MMLU delta is fine, mean length is up 8% (within the +20% threshold), false refusal is flat, sycophancy probe passes (it was tested on general-knowledge false premises, not on fabricated citations). **Every gate is green.** The failure is invisible because no gate measures the thing that is wrong.

**What the lecture contributes.** Sandel's exchange with Mike at [00:50:02] — *"Well, then Bentham has to be wrong. If you're right, he's wrong."* Mike's "okay, then he's wrong" is the is/ought move in miniature: the descriptive fact (people prefer confidence) does not entail the normative conclusion (confidence is better). The pipeline makes that move silently at the `argmax`. §5.3's practical upshot: **the reward model was never asked whether the citation existed**, and no amount of better RM training changes that — you need a different measurement.

**The diff.**

1. **Eval:** add a **verifiable** slice — 400 questions with answers checkable against a contract fixture. Assert on exact-match accuracy, not win rate. This is CS-14 §8.2's verifiable-vs-subjective split applied *within* a subjective domain.
2. **Gate:** promotion requires `citation_accuracy >= 0.95` on the fixture set, independent of `win_rate_vs_sft`. A gate, not a term (§5.4).
3. **Dataset:** add pairs where the hedged-but-correct answer is `chosen` and the confident-but-unsupported answer is `rejected`. This is CS-14 §4.3.4 item 5 (*"A response containing a fabricated fact loses to a response that admits uncertainty"*) — the guideline clause existed; the pairs did not.
4. **Metric:** report win rate and citation accuracy **side by side, always**. A win rate reported alone will be read as quality.

**Cost.** 400 verifiable eval items (the expensive part — roughly 40 expert-hours at CS-14 §4.3.6's rates, $4,000–$8,000) plus 200 new pairs. The scenario's real lesson: **the win rate was 0.67 and the model was worse.**

---

### Scenario 4 — Average win rate up, worst slice down

**Situation.** A fintech ships a fine-tuned assistant whose users include a large share of non-native English speakers. The DPO run improves the aggregate win rate from 0.53 to **0.64** on a 900-item comparison set. A monthly review of support escalations shows a 22% rise in "the assistant misunderstood me" tickets, concentrated in a segment the team has never sliced.

**The value question.** §5.4. The mean rose because the majority of the comparison set's prompts are phrased the way the annotators phrase things (they were written by the product team, in the product team's register). The minority whose phrasings differ are the ones whose win rate fell. This is Parker at [00:50:46]: the group without a constituency has weight zero, and the arithmetic is not malicious — it is arithmetic.

**Why the standard checks do not catch it.** CS-14 §16.4's gate asserts on six aggregate numbers. CS-14 §16.3's monitoring table watches win rate, length, refusal rate, capability suite, prompt-distribution drift, and complaints — and the *complaint* row is the only one that moved, and CS-14 §16.3 already assigns it the right action (*"Roll back first, diagnose second"*). **The handbook's monitoring was correct and the gate was not.** The escalation review caught it a month late.

**What the lecture contributes.** §3.9's aggregation analysis: the mean is a weighted sum, and the weights come from who is in the comparison set. And §3.13's veil — *write the eval as if you did not know which user you would be* — which generates the slice list before the incident instead of after.

**The diff.**

1. **Eval:** build three slices from *existing* traffic, not from the product team's imagination: (a) prompts under 8 tokens, (b) prompts containing a non-native-English construction from a fixed list, (c) prompts with typos in the key entity. 300–400 items each.
2. **Gate:** add the three slices to `KNOWN_GOOD` as floors. The 0.64 mean does not promote if `win_rate_slice_typo_entity` fell below 0.50.
3. **Monitoring:** add a slice-level drift check to CS-14 §16.3's table — the existing *"Prompt distribution drift (PSI > 0.2)"* row tells you the input distribution moved; a per-slice win rate tells you the *output quality* moved with it.
4. **Rollback:** this scenario is already a rollback case under CS-14 §16.3's complaint row. The gate change is what prevents the next one.

**Cost.** One day of eval construction, no new annotation, no training. **This is the cheapest recommendation in the appendix** and it is the one that catches the failure mode with the highest real-world cost.

---

### Scenario 5 — The team that debated for six weeks and shipped nothing

**Situation.** A six-person team is asked to add preference alignment to an internal knowledge assistant. Two engineers read this appendix and CS-14 §4.3. They convene a working group on annotator ethics. Week 1: a paper is circulated. Week 2: a debate about RLAIF vs human labels becomes a debate about whether AI-generated preferences can be legitimate at all. Week 3: someone raises the veil of ignorance and proposes that the team write their own Rawlsian principles for the assistant. Week 4: the principles draft has nine clauses and no agreement on clause 4. Week 5: the product owner asks for a date. Week 6: a decision is made to "do more research".

Meanwhile the assistant continues to answer questions in the register the SFT stage gave it, which is the register of the 400 scraped Wikipedia articles that CS-13's data pipeline used, and which nobody chose either.

**The value question.** This is §3.16's evasion [00:21:59] in its engineering form, and Sandel supplies the reply verbatim: *"whatever the merits of the debate, the reason they're inescapable is that we live some answer to these questions every day"* [00:22:35]. The team is shipping an answer — the SFT register — and calling the absence of a decision "further research".

**What the lecture contributes, in the opposite direction.** The other half of §3.16: *"You have to allow for the possibility that political philosophy may make you a worse citizen rather than a better one, or at least a worse citizen before it makes you a better one… philosophy is a distancing, even debilitating activity"* [00:20:10]. The team has taken the therapeutic aim of a philosophy course into a delivery setting, which is exactly the over-correction the lecture warns about.

**The diff.**

1. **Process:** adopt §8's test as a standing agenda item. Every question is classified as *changes an artefact* or *changes nothing*, in the meeting, on the spot.
2. **STOP condition:** any question open for more than one working day defaults to the documented option, with the question logged against the metric that would settle it. This is §8's first STOP condition, and it is a rule, not a sentiment.
3. **Budget:** cap the philosophy at **one day and fifty lines** (§1.2) — a design-doc section titled "Values decisions and their owners" containing §4.9's table with your own entries filled in.
4. **Ship:** run the standard pipeline with a written guideline (CS-14 §4.3.4's eight fields), 2,000 pairs, β=0.1, and the six `KNOWN_GOOD` numbers *plus one slice floor* (§6.2). Re-evaluate in a quarter with real data.

**Cost.** The scenario's cost is the six weeks already lost, plus whatever the unexamined SFT register is doing to the product. The fix costs one day and a design-doc section. **The lesson is not that the philosophy is wrong; it is that this appendix is worth one day and no more, and a team that spends six weeks on it has mis-allocated against its own users.**

---

## 11. Common misconceptions

**1. "A preference dataset records what people want."**
It records what a specific set of people ranked, under a specific framing, at a specific moment, among the options that happened to be sampled. `chosen` is not a want; it is a verdict on a pair (§2.1, §4.5). CS-14 §3 says it directly of the `Chosen / rejected` glossary entry: *"Confused with 'correct/incorrect' — preferences are about quality, not truth."* The correction goes further: they are about *rank*, not even quality.

**2. "The reward model learns what is good."**
The reward model learns what the annotators *ranked higher*, expressed as a scalar with a margin (CS-14 §4.4). Nothing in the Bradley–Terry objective references goodness, correctness, or welfare. The transition from description to target happens at the `argmax`, and nobody signs it (§5.3).

**3. "High inter-annotator agreement means our preference signal is good."**
Agreement measures whether the signal *exists*, not whether it is *right*. CS-14 §4.3.5 states it as a ceiling — *"if two competent annotators agree on only 65% of pairs, then 35% of your labels are noise"* — which is correct and is only half the story. **A pool drawn from one population will produce high agreement and a coherent bias**, and no agreement statistic can distinguish those two cases. That is why §6.1 exists and why §6.5 looks at the *direction* of disagreement rather than only its frequency.

**4. "The annotators consented, so the data is ethically clean."**
Three separate things are being conflated: consent to do the labelling task, consent to have one's judgements used as training signal, and the absence of harm to third parties who never consented. Sandel's consent discussion [00:52:46] is about the *first* kind in its strongest form — consent to a harm to oneself — and even there the class cannot agree that it suffices. The third party in an alignment pipeline is **every user of the model**, and no annotator's consent reaches them (§3.11, §5.4).

**5. "RLHF is deprecated, so these questions are historical."**
CS-14 §17 item 3 corrects the "deprecated" reading: *"RLHF/PPO is deprecated as a default for product teams, not as a technique."* The values question does not belong to RLHF anyway — it belongs to the preference data, and **every method in CS-14 §4.6 consumes the same pairs.** DPO, ORPO, KTO, SimPO, SPIN and GRPO-with-an-RM all inherit the same aggregation, the same pool, and the same deleted minority (§4.5, §4.8). Switching method changes the memory bill, not the ethics.

**6. "RLAIF avoids the human-values problem."**
RLAIF moves the problem into the judge. CS-14 §4.3.6 is explicit: *"AI feedback is biased toward the judge's own preferences, which are themselves shaped by its own alignment. A GPT-4 judge systematically prefers verbose, heavily-structured, listy answers — because GPT-4 was trained to produce them."* One population's values have been replaced by one model's values, at 50× lower cost (§4.3.6). The cost improvement is real; the ethical improvement is not established.

**7. "The KL term keeps the model safe."**
CS-14 §17 item 8: *"The KL term keeps the model near the reference, which is not the same as safe. A KL anchor against a rude SFT model anchors you to rudeness. It is an anti-drift term, not a safety mechanism."* In the vocabulary of this appendix: **β is a statement about which values get to persist** (§4.6), and a high β means the SFT stage's values outrank the preference data's. That is a decision, not a safety property.

**8. "If the win rate improved, the model is better."**
Scenario 3 is the counterexample: win rate 0.67 and the model fabricates clause numbers. Win rate is a preference measurement on a comparison set, computed with a judge whose biases are documented in CS-14 §4.3.6 and §12.1. It answers *"is the model preferred by this procedure on this set"* — which is a real and useful question, and is not *"is the model better"* (§5.2, §5.3).

**9. "Ethics is a review step at the end."**
Every decision in §4.9's table is made *before* the first training step, at the moment the pool is recruited, the guideline is written, and the schema is fixed. A review at the end can inspect the artefact; it cannot unmake the aggregation. **The interventions that work are data-schema changes and gate changes, and both are early.**

**10. "This is a problem for big labs, not for my 2,000-pair fine-tune."**
Scale changes the *magnitude* and not the *structure*. A 2,000-pair adapter built by four people from one team encodes four people's ranking of a specific set of options; it will be deployed to users who are not those four people. §10 Scenario 5 is a six-person team. Scenario 1 is a mid-size product. The smallest actionable change in this entire appendix — a single slice floor in a CI gate — costs one day at any scale (§6.2).

**11. "Philosophy gives you answers."**
It gives you questions with names and a vocabulary that survives a reorg (§9.1). The answers are the ones you write down in an annotation guideline and enforce with a gate. A team that expects the philosophy to decide has confused §1.2's one-day budget with a research programme, and is on the way to §9.4's paralysis.

**12. "If we can't measure it, it isn't a real engineering concern."**
The converse of the previous point, and the more common error in practice. §5.4's distributional harm is measurable and is usually unmeasured; §5.2's preference/welfare gap is measurable and requires product telemetry rather than annotation. The cases that are genuinely unmeasurable are a small minority, and the correct response to them is §8's second STOP condition — log the question with the metric that would settle it, ship the documented default, revisit. **"Unmeasurable today" is not "not a concern"; it is "not a blocker".**

---

## 12. Self-check questions

Fifteen questions. Answer them before reading the answers; the second half is harder than the first.

**Q1.** Name the four slots of a preference, and for each, name the pipeline artefact where the corresponding decision is made.

**Q2.** Why does Sandel's trolley sequence (driver → footbridge → transplant surgeon) matter for a preference dataset, rather than being a curiosity about moral psychology?

**Q3.** What does CS-14 §4.3.5's inter-annotator agreement statistic measure, and what does it structurally fail to detect?

**Q4.** State the Bradley–Terry objective and identify precisely which term in the alignment pipeline it lacks. What is the consequence?

**Q5.** What is "welfare" in this appendix, and why can a fully satisfied preference ranking still reduce it? Give one concrete alignment failure that this describes.

**Q6.** Name three distinct things "consent" could mean for a preference dataset, and say which one Sandel's discussion actually covers.

**Q7.** What are the three questions Sandel poses at the end of the lecture [00:51:27]–[00:53:27], and what engineering artefact does each correspond to?

**Q8.** What is the "menu problem", and which CS-14 section documents it as a method limitation?

**Q9.** Explain why CS-14 §3 says the `RM score` is *"the metric being gamed"*, and connect this to the is/ought gap.

**Q10.** Give the veil-of-ignorance heuristic for writing an annotation guideline, and state its main weakness.

**Q11.** What is the difference between a floor and a price in a distributional constraint, and which one does CS-14 §16.4's CI gate implement?

**Q12.** Why does a *high* IAA score not rule out a coherent bias? What measurement would you add?

**Q13.** Scenario 3's model had a win rate of 0.67 and fabricated citations. Name two changes to the eval and one to the dataset that would have caught it.

**Q14.** What is the difference between "changing the labelling UI mid-collection" and "changing the guideline mid-collection", and why does the first matter more than teams expect?

**Q15.** Given §8's test, classify these four questions as *changes an artefact* or *changes nothing*, and say why: (a) "does the model have moral status?", (b) "our annotator pool is 92% one country and we ship in eleven", (c) "which ethical theory is correct?", (d) "we changed the labelling UI in week 3 of a 6-week collection".

<details>
<summary>Answers</summary>

**A1.** (a) **A ranking** — the annotation guideline, CS-14 §4.3.4, where the tie rule and the HHH priority order live. (b) **Over outcomes** — the sampler and prompt pool, CS-14 §4.6.12 and §5.1. (c) **Held by someone** — the annotator pool or judge choice, CS-14 §4.3.2. (d) **Elicited by some procedure** — the task design and elicitation protocol, CS-14 §4.3.4 and §4.3.6. §2.1, §4.9.

**A2.** Because the sequence is a **demonstration that intuitions are not a total order over outcomes**. The class endorses 5-over-1 in case 1 [00:01:53] and rejects it in cases 2 and 3 [00:05:49], [00:12:07], and cannot articulate the feature that distinguishes them [00:07:30]–[00:09:53]. Every preference dataset inherits exactly this structure: annotators hold firm rankings whose distinguishing feature is not in the guideline, so the feature is learned and unauditable. §2.2, §3.2.

**A3.** It measures whether a preference signal **exists** — whether independent annotators produce the same ordering — as raw agreement, Cohen's κ, Krippendorff's α, or Spearman ρ on ranks. It fails to detect **systematic agreement caused by a shared population**: a pool drawn from one region, one language, or one employer will agree strongly and encode a coherent bias, and high agreement is the signature of both cases. The added measurement is §6.5's split-direction audit plus §6.1's composition disclosure. §4.3, §11 item 3.

**A4.** `L_RM(φ) = −E[log σ(r_φ(x,y+) − r_φ(x,y−))]`, i.e. `P(y+ ≻ y−) = σ(r(y+) − r(y−))`. It lacks any term referencing correctness, truth, welfare, or right. Consequence: it is a **descriptive** fit to a population's revealed rankings, and the pipeline then treats it as a normative target by maximising it (`max E[r] − β·KL`). That transition is the is/ought move, and it is performed by an `argmax`, unsigned. §5.3, CS-14 §4.4, CS-14 §4.5.

**A5.** Welfare is what actually benefits or harms the person, as distinct from what they ranked. A ranking can be fully satisfied while welfare falls because *preferences can be for things that harm* — the canonical alignment case is **sycophancy**: the confident, agreeable answer is preferred by annotators and by users, so preference optimisation raises it, and honesty falls. CS-14 §2.4 states it: *"Optimising for 'sounding confident' via a preference signal directly attacks honesty."* Second case: engagement-optimised responses that users regret. §5.2, §11 item 8.

**A6.** (i) Consent to perform the labelling task (an employment question); (ii) consent to have one's *judgements* used as training signal (a data-use question); (iii) the absence of harm to third parties who never consented — every user of the deployed model. Sandel's discussion is (i)-like in its strongest form: consent to a **harm to oneself**, and even there the class does not agree it suffices [00:39:36], [00:41:20]. Category (iii) is the one that matters most and is reached by no one's consent. §3.11, §5.4.

**A7.** (1) *Where do rights come from if not from utility?* [00:51:27] → the **constraint set**: refusal policy, safety gate, slice floor (§3.9, §5.4). (2) *Why does agreement to a fair procedure justify its result?* [00:52:12] → the **annotation guideline, sampling frame, and tie rule** — procedures whose legitimacy does not transfer to the output's correctness (§3.12). (3) *What moral work does consent do?* [00:52:46] → the **consent and provenance record**: what was consented to, by whom, for which use (§3.11, CS-14 §16.5).

**A8.** The menu problem is that a ranking is defined only over the options that were present. In an alignment pipeline, the options are the responses the **sampler** produced. CS-14 §4.6.3 documents the consequence for DPO — *"Offline only — it cannot explore beyond the pairs you give it, so its ceiling is your dataset's ceiling"* — and CS-14 §4.6.12 states the same for rejection sampling: *"Bounded by the sampler."* This is why CS-14 §4.6.1 still rates PPO's exploration advantage as real. §2.3, §4.2.

**A9.** Because the reward model is a learned proxy, and the policy is optimised against the proxy rather than against human judgement. CS-14 §4.9 documents the empirical phenomenon (over-optimisation: RM score rises while true quality falls) and CS-14 §4.4's own glossary entry says RM score is *"**The metric being gamed.**"* The is/ought connection: the RM *describes* rankings; optimisation *targets* the description; a description pushed hard enough stops describing. CS-14 §4.5's KL term is the epistemic constraint that keeps the description valid. §5.3.

**A10.** Write the guideline, the refusal policy, and the eval slices as if you did not know which side of them you would be on — which user group, region, language, or use case. It generates slices not derived from your existing traffic. Its weakness: an engineer imagining themselves into a group they have never been part of is a weak substitute for consulting that group, so the veil is **a floor, not a ceiling**. §3.13, §6.1, §6.4.

**A11.** A **floor** is a constraint outside the objective — a gate asserting `>=` on a defined population — which the aggregate cannot buy its way past. A **price** is a penalty term inside the objective, which any sufficiently large reward can pay. CS-14 §16.4's CI gate implements floors: `KNOWN_GOOD` asserts on six numbers and promotion fails if any regresses. The extension in §6.2 adds per-slice floors (`win_rate_slice_non_en: 0.50`). §5.4, §9.4 ("the floor that became a price").

**A12.** Because IAA measures *whether* annotators agree, not *why*. A pool drawn from one population agrees strongly on a coherent set of values, and the statistic cannot distinguish "the signal is real" from "the pool is homogeneous." The added measurements: (a) aggregate composition disclosure (§6.1); (b) a split-direction audit on the double-annotated 10%, asking whether contested pairs cluster by topic (guideline problem) or by annotator group (pool problem) (§6.5). §4.3, §11 item 3.

**A13.** **Eval:** (1) add a **verifiable slice** — 400 questions answerable against a contract fixture, asserted on exact-match accuracy rather than win rate; (2) report win rate and verification accuracy **side by side**, so the win rate is never read alone. **Dataset:** add pairs where the hedged-but-correct answer is `chosen` and the confident-but-unsupported answer is `rejected` — CS-14 §4.3.4 item 5's guideline clause existed; the pairs did not. **Gate:** assert `citation_accuracy >= 0.95` independently of win rate. §10 Scenario 3, §5.3.

**A14.** A **guideline** change alters what "better" means and is at least visible in the document's version history; a **UI** change alters *how* the ranking is elicited — pairwise to Likert, ties offered or not, rationale required or not — and produces a different measurement while keeping the same column names. Teams expect the guideline change to matter and miss the UI change because the data *looks* homogeneous afterward. This is §2.5's point that procedure is not neutral, and §6.3 item 5's change log is the fix. The analogous silent-format bug in CS-14 §10 item 18 is the chat-template trap.

**A15.** (a) **Changes nothing** — a research question with no artefact consequence in this repo. (b) **Changes an artefact** — slice the eval, add a per-language floor, disclose composition (§10 Scenario 1). (c) **Changes nothing** — 2,400 years unresolved and not blocking a sprint. (d) **Changes an artefact** — version the dataset, mark the phase boundary, consider training only on the post-change data (§6.3 item 5). The test is §8's: *if we answer the other way, what file changes?*

</details>

---

## 13. Cross-references

| Relationship | Module and section |
|---|---|
| **Builds on** | CS-13 (SFT — the π_ref that β protects, and the stage where the anchor's values were set: §4.6 data curation, §16.5 compliance), CS-14 (the entire mechanical substrate: §2.4 HHH, §4.3 preference data, §4.4 reward modelling, §4.5 the KL term, §4.6 method profiles) |
| **Bridges between** | CS-14 §4.6.11 (RLAIF and Constitutional AI — the values question in its written-down form) and CS-12 §4.12 (the legal regimes governing the data) |
| **Pairs with** | CS-14 §16.5 (compliance and the audit trail — the engineering form of §6.3), CS-14 §16.4 (the CI gate this appendix extends with slice floors), CS-14 §16.2 (the versioning manifest that §6.3's protocol belongs in), CS-13 §16.1 (`MANIFEST.json` — the pattern for documenting an elicitation protocol) |
| **Extends** | CS-14 §4.3.2 (sources without populations), CS-14 §4.3.5 (agreement without direction), CS-14 §4.3.4 (the guideline as a policy document — this appendix adds *who wrote it and under what authority*), CS-14 §12.1 and §16.4 (metrics without slices) |
| **Needed by** | CS-14 §20 already declares the bridge: *"Bridges to — AP-01 (The Ethics & Philosophy of Alignment — what is a preference, and who gets to encode it)"* |
| **Contrasts with** | CS-01 §16.5 and CS-12 §16.7 (the compliance and model-card framings — process and disclosure rather than justification), CS-02 §16.5 (drift, guardrails, compliance) |
| **Referred to by** | IQ-14 (Q93's week-1 diagnosis, Q99's regulator answer, Q100's 400-pair budget — all three are §6 practices in interview form) |

**Verification note.** Every `CS-NN §M` above was checked against the file with grep before this appendix was committed, because this repository has been bitten repeatedly by stale and invented section references. The full list with the matching heading is in the report accompanying this file's creation. Two references were adjusted during writing: CS-14's cross-reference table is **§20**, not §10 (CS-14 §10 is *Exceptions, Edge Cases & Gotchas*), and the "how alignment works" material is CS-14 **§4**, with §4.3's subsections carrying the preference-data detail.

**A note on unwritten targets.** CS-14 §20 lists CS-24, CS-25, CS-26 and CS-27 as the deep-dive owners for RLHF, DPO, GRPO and ORPO respectively. Those four modules are marked as not yet written in the README curriculum table. Until they exist, **CS-14 §4.6 is the deepest treatment in this repository** for every method this appendix references, and every method link here deliberately points there rather than at an empty target.

---

## Appendix A — Instructor's verbatim key claims

Every quotation below is from `NoteGPT_Transcript_Justice What's The Right Thing To Do Episode 01 THE MORAL SIDE OF MURDER.txt`, at the timestamp given. Nothing in this table is paraphrased and nothing is invented.

| Timestamp | Claim |
|---|---|
| [00:00:10] | *"Suppose you're the driver of a trolley car, and your trolley car is hurtling down the track at 60 miles an hour, and at the end of the track, you notice five workers working on the track."* |
| [00:02:31] | *"Because it can't be right to kill five people when you can only kill one person instead."* |
| [00:03:40] | *"Well, I think that's the same type of mentality that justifies genocide and totalitarianism. In order to save one type of race, you wipe out the other."* |
| [00:05:49] | *"What became of the principle, better to save five lives even if it means sacrificing one?"* |
| [00:07:30] | *"The trolley car is a runaway thing, and you're making a split-second choice, whereas pushing the fat man over is an actual act of murder on your part."* |
| [00:08:09] | *"Either way, you have to choose who dies, because you either choose to turn and kill the person, which is an active, conscious thought to turn, or you choose to push the fat man over, which is also an active, conscious action."* |
| [00:09:56] | *"So you have the choice of becoming involved or not by pushing the fat man."* |
| [00:13:03] | *"Except for the fact that you just wrecked the philosophical point."* |
| [00:13:05] | *"The first moral principle that emerged in the discussion said the right thing to do, the moral thing to do, depends on the consequences that will result from your action."* |
| [00:13:44] | *"Consequentialist moral reasoning locates morality in the consequences of an act, in the state of the world that will result from the thing you do."* |
| [00:14:18] | *"People gestured toward reasons having to do with the intrinsic quality of the act itself, consequences be what they may."* |
| [00:15:03] | *"Categorical moral reasoning locates morality in certain absolute moral requirements, certain categorical duties and rights, regardless of the consequences."* |
| [00:15:35] | *"The most influential example of consequential moral reasoning is utilitarianism, a doctrine invented by Jeremy Bentham, the 18th century English political philosopher. The most important philosopher of categorical moral reasoning is the 18th century German philosopher Immanuel Kant."* |
| [00:18:08] | *"Philosophy teaches us and unsettles us by confronting us with what we already know."* |
| [00:18:56] | *"Once the familiar turns strange, it's never quite the same again."* |
| [00:20:10] | *"You have to allow for the possibility that political philosophy may make you a worse citizen rather than a better one, or at least a worse citizen before it makes you a better one. And that's because philosophy is a distancing, even debilitating activity."* |
| [00:21:21] | *"So Callicles is really saying to Socrates, 'Quit philosophizing. Get real. Go to business school.'"* |
| [00:21:59] | *"The name of the evasion is skepticism. It's the idea… maybe it's just a matter of each person having his or her own principles, and there's nothing more to be said about it, no way of reasoning."* |
| [00:22:35] | *"The reason they're unavoidable, the reason they're inescapable is that we live some answer to these questions every day."* |
| [00:23:08] | *"Skepticism is a resting place for human reason, where it can reflect upon its dogmatic wanderings, but it is no dwelling place for permanent settlement."* (Kant, quoted by Sandel) |
| [00:23:40] | *"The aim of this course is to awaken the restlessness of reason and to see where it might lead."* |
| [00:25:20] | *"We began with our judgments in particular cases. We tried to articulate the reasons or the principles lying behind our judgments. And then confronted with a new case, we found ourselves re-examining those principles, revising each in the light of the other."* |
| [00:27:14] | *"Bentham's idea is the following: the right thing to do, the just thing to do, is to maximize utility… He meant by utility the balance of pleasure over pain, happiness over suffering."* |
| [00:28:15] | *"He started out by observing that all of us, all human beings, are governed by two sovereign masters, pain and pleasure."* |
| [00:28:54] | *"Bentham's utilitarianism is sometimes summed up with the slogan, 'The greatest good for the greatest number.'"* |
| [00:32:09] | *"So on the 19th day, Dudley, the captain, suggested that they should all have a lottery, that they should draw lots to see who would die to save the rest. Brooks refused."* |
| [00:32:40] | *"Dudley told Brooks to avert his gaze, and he motioned to Stevens that the boy, Parker, had better be killed… he killed him with a penknife, stabbing him in the jugular vein."* |
| [00:35:57] | *"I just feel like in a situation that desperate, you have to do what you have to do to survive."* (Marcus) |
| [00:36:15] | *"Let's say they survive, and then they become productive members of society who go home and start a million charity organizations… They benefit everybody in the end."* (Marcus) |
| [00:38:05] | *"There's no situation that would allow human beings to take the idea of fate or the other people's lives in their own hands, that we don't have that kind of power."* (Britt) |
| [00:38:28] | *"I'm wondering if Dudley and Steven had asked for Richard Parker's consent in dying, if that would exonerate them from an act of murder?"* (Kathleen) |
| [00:40:16] | *"If he was making his own original idea, and it was his idea to start with, then that would be the only situation in which I would see it being appropriate in any way, because that way you couldn't make the argument that he was pressured."* |
| [00:41:20] | *"So, there's no definite reason that he should be killed, because you don't know when they're going to get rescued. So, if you kill him, it's killing him in vain."* |
| [00:43:21] | *"So, the numbers are rising if we add a lottery."* |
| [00:43:44] | *"I think the essential element, in my mind, that makes it a crime is the idea that they decided at some point that their lives were more important than his… It's like my needs, my desires are more important than yours, and mine take precedent."* (Matt) |
| [00:44:21] | *"So Matt, for you, what bothers you is not the cannibalism, but the lack of due process."* |
| [00:44:51] | *"The way I understood it originally was that was the whole issue, is that the cabin boy was never consulted about whether or not something was going to happen to him."* |
| [00:47:17] | *"I don't think that there is any remorse. In Dudley's diary, 'We were eating our breakfast,' it seems as though he's just sort of like, 'Oh,' the whole idea of not valuing someone else's life."* |
| [00:48:14] | *"I think undoubtedly, the way our society's shaped, murder is murder. Murder is murder in every way, and our society looks at murder down on it in the same light, and I don't think it's any different in any case."* (Mike) |
| [00:49:05] | *"The one, the cabin boy, he had no family. He had no dependents. These other three had families back home in England. They had dependents. They had wives and children."* |
| [00:49:49] | *"Suppose it weren't three, suppose it were 30. 300. One life to save 300. We're in wartime. 3,000."* |
| [00:50:02] | *"Do you think Bentham is wrong to say the right thing to do is to add up the collective happiness? You think he's wrong about that?"* / *"I don't think he's wrong, but I think murder is murder in any case."* / *"Well, then Bentham has to be wrong. If you're right, he's wrong."* / *"Okay, then he's wrong."* |
| [00:50:46] | *"Their families back home, their dependents. Parker was an orphan. No one would miss him."* |
| [00:51:27] | *"Is it because even cabin boys have certain fundamental rights? And if that's the reason, where do those rights come from, if not from some idea of the larger welfare or utility or happiness? Question number one."* |
| [00:52:12] | *"Why does agreement to a certain procedure, even a fair procedure, justify whatever result flows from the operation of that procedure? Question number two."* |
| [00:52:46] | *"What is the moral work that consent does? Why does an act of consent make such a moral difference that an act that would be wrong, taking a life without consent, is morally permissible with consent?"* |
| [00:53:27] | *"To investigate those three questions, we're going to have to read some philosophers. And starting next time, we're going to read Bentham and John Stuart Mill, utilitarian philosophers."* |

---

## Appendix B — Reference links

Split into the philosophy sources (what the lecture draws on and names) and the alignment sources (what this appendix maps onto). The alignment side deliberately points at this repository's own modules first, because those are the definitions the reader can check.

### B.1 Philosophy sources

| Source | What it is | Where it is used here |
|---|---|---|
| **Sandel, *Justice: What's the Right Thing To Do?*, Episode 01 — "The Moral Side of Murder"** | The primary source for this appendix. The trolley cases [00:00:10]–[00:13:03], the consequentialist/categorical distinction [00:13:05]–[00:15:35], the Queen v. Dudley and Stephens case [00:29:37]–[00:33:59], the consent and lottery discussion [00:38:28]–[00:47:11], the three closing questions [00:51:27]–[00:53:27], and the warning about philosophy's risks [00:17:21]–[00:23:40]. | Everywhere. The only transcript this appendix draws on. |
| **Bentham, *An Introduction to the Principles of Morals and Legislation* (1789)** | The origin of the utility principle and the "two sovereign masters" formulation. Named in the lecture at [00:15:35] and [00:27:14]. | §3.8 (the mapping to Bradley–Terry), §3.9, §5.3 |
| **Mill, *Utilitarianism* (1861)** | Announced at [00:53:27] as the next reading. Not covered in Episode 01. | Referenced only; no claim attributed. |
| **Kant, *Groundwork of the Metaphysics of Morals* (1785)** | Named at [00:15:35] as the most important categorical philosopher. The skepticism quotation at [00:23:08] is from the *Critique of Pure Reason*. | §3.7, §3.10 |
| **Rawls, *A Theory of Justice* (1971)** | The veil of ignorance. **Not in Episode 01** — the phrase does not appear in the transcript. | §3.13 — flagged as beyond the lecture, no timestamp claimed |
| **Sandel, *What Money Can't Buy: The Moral Limits of Markets* (2012)** | The market-limits argument. **Not in Episode 01.** | §3.14 — flagged as beyond the lecture, no timestamp claimed |
| **Aquinas (13th c.); Anscombe (1958); Foot (1967)** | The doctrine of double effect. **Not named in Episode 01.** The phenomenon the class gropes toward is at [00:07:30]–[00:09:53]. | §3.3 — flagged as beyond the lecture, no timestamp claimed |
| **Aristotle, *Nicomachean Ethics*** | Named in the syllabus list at [00:16:15]. Virtue and character. | §3.15 — flagged as the weakest mapping in the file |
| **Plato, *Gorgias*** | Quoted by Sandel at [00:20:48]–[00:21:21] (Callicles's advice to Socrates). | §3.16 |
| **Locke, *Second Treatise of Government*** | Named in the syllabus at [00:16:15]. Not covered in Episode 01. | Referenced only. |
| **The Queen v. Dudley and Stephens (1884), 14 QBD 273** | The real case behind [00:29:37]–[00:33:59]. The lecture summarises it faithfully. | §2.4, §3.11, §3.12, §5.1, §5.4 |
| **The Trolley Problem — Foot (1967); Thomson (1976, 1985)** | The origin of the case Sandel opens with [00:00:10]. The footbridge variant is Thomson's. Not named in the lecture. | §3.1, §3.2 |

### B.2 Alignment sources in this repository

| Section | What it defines | Used here for |
|---|---|---|
| **CS-14 §2.4** | HHH defined precisely; the three axes in tension | §3.8, §5.2, §11 item 8 |
| **CS-14 §3** | The glossary, including `RM score` — *"The metric being gamed"* | §5.3, §11 item 2 |
| **CS-14 §4.3.1** | The `{prompt, chosen, rejected}` schema | §2.1, §4.5 |
| **CS-14 §4.3.2** | Where preferences come from (the eight-source table) | §2.5, §4.3 |
| **CS-14 §4.3.4** | The eight required annotation-guideline fields | §2.2, §3.2, §3.12, §4.4, §6.3 |
| **CS-14 §4.3.5** | Inter-annotator agreement: κ, α, and the calibration recipe | §3.6, §4.5, §5.1, §6.5, §11 item 3 |
| **CS-14 §4.3.6** | Cost per pair; judge-bias transfer | §3.14, §4.6, §11 item 6 |
| **CS-14 §4.4** | Bradley–Terry; the reward-model loss; the distribution-shift warning | §3.8, §5.3, §11 item 2 |
| **CS-14 §4.5** | The KL term, β, and what β's extremes do | §3.1, §4.6, §7, §11 item 7 |
| **CS-14 §4.6.3 / §4.6.9** | DPO and ORPO profiles, including break conditions | §2.3, §5.2, §8 |
| **CS-14 §4.6.11** | RLAIF and Constitutional AI — values written down | §4.4, §9.4 |
| **CS-14 §4.6.12** | Rejection sampling and the sampler ceiling | §2.3, §4.2 |
| **CS-14 §4.9** | Reward hacking, over-optimisation, the ranked mitigation list | §3.1, §3.4, §5.3 |
| **CS-14 §12.1 / §12.2** | The metric stack and how each one lies; LC win rate | §4.7, §11 item 8 |
| **CS-14 §16.2** | The alignment-artefact versioning manifest | §4.3, §6.1, §6.3 |
| **CS-14 §16.3** | Monitoring and drift thresholds | §4.1, §10 Scenario 4 |
| **CS-14 §16.4** | The CI regression gate (`KNOWN_GOOD`) | §4.7, §6.2, §10 Scenarios 2, 4 |
| **CS-14 §16.5** | Compliance and the audit trail; guidelines as policy documents | §1.1, §3.5, §3.10, §4.4, §6.3 |
| **CS-14 §17 items 3, 8** | "RLHF is deprecated" and "the KL term keeps the model safe" corrections | §11 items 5, 7 |
| **CS-13 §12.3** | Refusal rate in both directions; format compliance | §3.2, §3.7, §5.4 |
| **CS-13 §16.1** | `MANIFEST.json` and dataset versioning | §6.3, §6.1 |
| **CS-13 §16.5** | Compliance table: provenance, licensing, PII, right to erasure, model cards | §3.10, §3.11, §6.4 |
| **CS-13 §4.6.6** | Dataset licensing | §8 (the licence STOP condition) |
| **CS-12 §4.12.1 / §4.12.2** | The four legal regimes; the provenance ledger | §3.10, §3.11 |
| **CS-12 §16.6 / §16.7** | The source-rights checklist; the model-card template | §7, §6.1 |
| **CS-01 §16.4 / §16.5** | Safety regression (refusal + jailbreak suites); the compliance angle | §3.7, §6.4 |
| **CS-01 §4.9** | Data quality: dedup, filtering, contamination, licensing | §7 |
| **CS-02 §12.1** | The five-number protocol, including the OOD slice metric | §6.2 |
| **CH-14 §2.2, §4.1, §5.4, §8** | β in plain terms; preference-dataset sizing; the length-bias audit; symptom→fix | §4.6, §6.2, §10 Scenario 2 |
| **IQ-14 Q93, Q99, Q100** | The 8-week alignment design; the regulator answer; the 400-pair budget | §6.4, §13 |

### B.3 Alignment sources outside this repository

| Source | What it is | Relevance |
|---|---|---|
| **InstructGPT — Ouyang et al., 2022** | The origin of the three-stage pipeline, the alignment tax, and PPO-ptx | The pipeline this appendix reads as a values-encoding machine; CS-14 §4.4, §4.8 |
| **DPO — Rafailov, Sharma, Mitchell et al., 2023** | The closed-form policy and the pairwise supervised loss | CS-14 §4.6.3; the objective whose target this appendix interrogates |
| **Constitutional AI — Bai et al., 2022** | The constitution + self-critique + revision recipe | The only mainstream recipe that writes its values down; §4.4, §9.4 |
| **RLAIF — Lee et al. (Anthropic), 2023/2024** | AI feedback replaces human feedback | §11 item 6 (the judge's values replacing the pool's) |
| **Gao, Schulman & Hilton, 2023 — *Scaling Laws for Reward Model Overoptimization*** | Quantifies the proxy-vs-true-reward gap against KL | §5.3's empirical backing for the is/ought claim |
| **Anthropic HH-RLHF; UltraFeedback/Argilla; `Zilly/math-step-dpo-10k`** | The three public preference datasets CS-14 §4.3.3 walks | §4.5's schema point — all are two or three columns, and none records disagreement |
| **TRL** | `DPOTrainer`, `DPOConfig`, `ORPOTrainer`, `KTOTrainer` | The implementation surface for everything §4 describes; CS-14 §6 |

---

*End of AP-01. This appendix is the repository's entire budget for the Justice material. Per `_MANIFEST.md`, that material does not appear in any technical module; if you find it there, it is a bug.*

