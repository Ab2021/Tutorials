# Interview Bank: Constrained & Structured Generation

> `T03` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T03-constrained-generation.md) · [Case study](../01-case-studies/T03-constrained-generation.md) · [Design blueprint](../03-design-blueprints/T03-constrained-generation/HLD.md)

## How to use this bank

Levels are **L3** (working competence — you have shipped with this), **L4** (senior practitioner — you own the tradeoff), **L5** (staff/architect — you own the decision and its blast radius). Every answer here is a *model* answer, not a script: it shows the shape and the numbers a strong candidate reaches for, and none of it should be recited. Numbers carry provenance — `[T]` for a lecture statement with the talk named, `[R]` for a supporting-repo path, `[D]` for arithmetic derived here with assumptions shown. Where the corpus does not supply a figure, the answer says so rather than inventing one.

Questions are ordered to read as one interview: foundations, then mechanism, then the tradeoffs, then debugging, then design at scale.

---

### Foundations

#### T03-Q1 · What problem does constrained generation actually solve?
**Difficulty:** L3 · **Depth expected:** 2 min
**Question:** Set the libraries aside. A team tells you their extraction endpoint is "mostly valid JSON." What is the problem class, and what kinds of constraint will a real system have to handle?
**Model answer:** The problem class is that a model's output is a *sample from a distribution over the whole vocabulary*, and nothing in that distribution knows your consumer's contract. A prompt moves the distribution's centre of mass; it does not remove the possibility of an invalid sample. So the real question is how to make a class of output **unreachable** rather than merely unlikely. Then the constraints split, and the case study's three surfaces are the canonical set: flat per-customer JSON fields and a closed 900-entry code list are **regular** — a finite automaton checks them token by token; recursive line-item nesting is **context-free** — it needs a pushdown automaton; and "variables must be defined before they are used" is above context-free and is *not* checkable token by token at all, so it goes to post-hoc validation `[T]` (CMU lecture 6). The lecture's own two axes — syntactic vs semantic, token-wise-verifiable vs end-verifiable — are what determine the architecture. Not the customer, not the model, not the library. Interviewers ask this first because the answer reveals whether you think in constraint classes or in vendor names.
**Signal:** Frames it as "unreachable versus unlikely" and immediately produces a taxonomy rather than reaching for a library name.
**Follow-ups:**
- *Which class is "the output must be exactly 10 tokens long"?* End-verifiable only — Q8.
- *Where does a closed code list sit?* Regular, and a hard mask's best case — Q13.
- *What about "don't mention competitors"?* Semantic, and not expressible as a grammar — Q19.
**Red flags:** Names a library ("just use Outlines") as the answer; treats all constraints as one problem; says "the model is usually good at JSON."

#### T03-Q2 · Syntactic versus semantic, and the axis that decides the architecture
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** The lecture gives a two-axis taxonomy. Name both axes, give an example on each side, and tell me which axis actually determines what you build.
**Model answer:** Axis one is **syntactic vs semantic**. Syntactic constraints are "constraints that are very easy to write down as sort of a list of allowable or unallowable tokens"; semantic constraints are "a little bit harder to define in that way" and connect to "a token level notion of alignment" and hence RLHF `[T]` (CMU lecture 6). Both are "verifiable" in the loose sense that you can write a deterministic function to check at the end whether the constraint was satisfied. Axis two is the one that decides the architecture: **token-wise verifiable** — "every token needs to start with a space," "everything I am writing needs to be for instance valid JSON" — versus **end-verifiable only** — "the output must be exactly 10 tokens long" `[T]`. That is the line. Token-wise-verifiable constraints get a **mask**, because at each step you know whether you are still satisfying the constraint, so you can restrict the candidate set. End-verifiable constraints get a **search or a filter**, because no per-token check exists and you are estimating whether the eventual output will satisfy it. The lecture's example is a NeuroLogic-A\*-style constraint, "write a sentence with these concepts car drive and snow" `[T]`. The practical consequence: axis one tells you whether the check can be deterministic or must be learned; axis two tells you *which machine* does the checking.
**Signal:** Names both axes and puts the architectural weight on the second — the discriminator most candidates miss entirely.
**Follow-ups:**
- *Is a semantic constraint ever token-wise verifiable?* Sometimes — a closed code list is both semantic and regular — Q13.
- *What does an end-verifiable constraint cost?* Search, possibly exponential without recombination — Q8.
**Red flags:** Only knows "formatting versus content"; treats semantic as "a harder regex."

#### T03-Q3 · Why not just prompt for JSON?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** The cheapest option is a system prompt that says "return only valid JSON." When is that genuinely fine, and when is it a liability?
**Model answer:** Fine for prototypes and internal tools with tolerant consumers — zero infrastructure, works today, and for a low-stakes surface a retry loop is cheaper than a compiler. A liability wherever validity is contractual, and the reason is structural rather than a matter of prompt quality. The lecture's point is concrete: under load or on an unusual input you may get "I'm sorry, you've exceeded your rate limit" where the JSON should have been `[T]` (CMU lecture 6). A prompt is a request the model may decline; a mask is a constraint it cannot violate. The measured version of this comes from the prompt-management talk: a team added one friendly sentence to a system prompt, the model started wrapping its JSON in prose, and valid replies fell from **98% to 71%** `[T]` (LLMOps prompt-management talk). A 27-point regression from a change nobody thought was semantically relevant is not something you can gate with care. Two secondary arguments. Cost: a prompt-only approach pays a full regeneration on every failure. And the decisive one for a regulated surface: the failure is silent in the metric that matters — 98% validity looks like quality until an invalid record reaches the ERP, at which point it is an audit finding. The repo's house line makes the same point in one sentence: prompt-based requests rely on the model's *willingness*; "JSON mode" relies on the engine's *inability* to do anything else `[R]`.
**Signal:** Distinguishes prototype from contract, and cites a *measured* regression rather than asserting "prompts are unreliable."
**Follow-ups:**
- *What if the model is very good?* The sparsity argument still applies — Q4.
- *So is prompt + validator enough?* It gives you a release guarantee and a poor first-attempt rate — Q18.
**Red flags:** "Prompts work fine if you're careful"; no awareness that validity is a distributional property, not a prompt property.

#### T03-Q4 · The sparsity argument: ~10 of ~100,000
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Give me the single number that justifies building a grammar compiler instead of writing a better prompt.
**Model answer:** "This is a really narrow constraint at any individual decoding step. We have over 100,000 choices but only like maybe 10 of them that will give us valid JSON at the end" `[T]` (CMU lecture 6). Roughly **10 legal continuations out of ~100,000 candidates** — a 1-in-10,000 ratio at a typical step — and that is the whole argument in one line. `[D]` Spelled out with the assumption stated: if the model's mass were spread uniformly across the vocabulary, an unconstrained draw would land on a schema-valid continuation about 0.01% of the time. The mass is not uniform, and real models concentrate probability on plausible tokens, which is exactly why prompt-only JSON works *most* of the time — but "most" degrades with temperature, with unusual documents, and under load. The number also explains why the mask is cheap: you are not computing anything expensive, you are intersecting a 128k-entry logit vector with a set of about ten token ids and writing `-inf` onto the rest `[T]`. And it sets the sampler order: with ten survivors out of 128k, truncating *before* masking spends the budget on tokens the mask is about to discard. One caveat to state out loud — this number is the lecturer's characterisation of the sparsity, not a measurement, and the case study's §9 records that the lecture quotes **no** benchmark figures for constrained generation: no accuracy, no latency, no human-eval results. Anything quantitative you say here should be labelled as a design parameter, not a measured result.
**Signal:** Quotes ~10-of-100k and draws at least two consequences from it — the case for a mask, and the processor order — while refusing to dress it as a measurement.
**Follow-ups:**
- *Is 128k the right denominator?* It is the lecture's Llama-3 vocabulary figure from a different lecture, and the case study uses "~100,000" `[T]`.
- *What does the ratio do for a flat schema?* It loosens — a boolean field has two legal tokens — which is why schema design moves the number `[D]`.
**Red flags:** Does not know the number; claims masking is expensive per token; presents the ratio as a benchmark.

#### T03-Q5 · The mask mechanism, precisely
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Describe exactly what the engine does to the logits at a masked step. When does it happen relative to the softmax, and why does that matter?
**Model answer:** The mechanism in the lecture's own words: "all the things that are allowed as the next token are zero, all the things that are not allowed to the next token we add minus infinity and then we take the softmax again to renormalize" `[T]` (CMU lecture 6). In practice: the schema has been compiled into an automaton with a current state; from that state the engine computes the set of legal token ids; it writes `-inf` — the lecture also describes it as "arbitrarily low" — onto every logit not in that set; then softmax runs over the modified vector. The **order is the point**. Because the mask is applied *before* softmax, illegal tokens are not merely unlikely, they are assigned exactly zero probability, and renormalisation redistributes the surviving mass among legal tokens only. That is what makes it a guarantee rather than a bias. Three consequences worth stating. It costs one lookup per step against a decode step dominated by weight reads, which is why the case study sets its budget at under 3% of a decode step. It composes with any sampler: temperature, top-p and penalties all operate downstream on the masked vector. And it does not change the model — the pre-mask distribution is untouched, so you can log it and still measure the model through the mask. The latency-relevant implementation detail is that the legal set is a compiled token-id list per state, not a regex evaluated per step `[T]`.
**Signal:** Says "before softmax" unprompted and explains that zero probability is a guarantee while renormalisation is a redistribution rather than a filter.
**Follow-ups:**
- *What if the legal set is empty?* You have a grammar bug, and it is the top masking incident — Q23.
- *Does masking break temperature-1 measurement?* The masked surface is no longer the model distribution; measure unmasked — see [T01](T01-sampling-decoding.md).
**Red flags:** Says the model "can't pick" invalid tokens with no mechanism; places the mask after softmax or after truncation.

#### T03-Q6 · What does the mask guarantee, and what does it not?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** A product manager says "with the grammar on, the output is correct." Correct them.
**Model answer:** The mask guarantees **the output is in the language** — that the token sequence is accepted by the automaton compiled from the schema. It does not guarantee the output is in the *schema version currently in force*, and it does not guarantee the output is *correct*. Three gaps. **Staleness:** if a schema is updated and a compiled grammar is cached, the mask happily produces records valid against the old grammar; only a validator reading the current version catches that. **Multi-token values:** a code like "RET-04A" may be emitted as three tokens, and a mask that validates only the first is not validating the code — the same state-machine problem as the FSA, one level down. **Semantics**, the largest gap of the three: the mask never promised that a valid record is a correct record, so a structurally perfect record with the wrong birth year validates cleanly. The case study makes this the whole audit conversation — "we do not promise that a valid record is a correct record" — and its §5.5 decision is that the mask is a *generation* mechanism while the validator is a *release* mechanism, answering to different failure modes. The repo states the same limit for JSON mode: even with the structure guaranteed, "the logic inside the JSON might be wrong" — a missing field, or a date in the wrong format `[R]`.
**Signal:** Separates "in the language" from "in the current schema" from "correct," and volunteers the stale-grammar gap before being asked.
**Follow-ups:**
- *Then why have a mask at all?* Because the validator alone cannot rescue a 92% first-attempt rate — Q18.
- *What else goes wrong inside a valid record?* Omission hallucination on long schemas — the model skips fields or fills them with placeholders `[R]`.
**Red flags:** "The grammar means it's correct"; no awareness that a compiled grammar can be stale.

#### T03-Q7 · The hierarchy: regular, context-free, and the world outside
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** Give me the expressiveness classes, one constraint in each, and the practical rule that follows.
**Model answer:** The lecture's hierarchy, in its own terms. **Regular** — a finite automaton, which "doesn't have any way of keeping track of how many times it's been in a state before" `[T]`; it handles anything needing no bookkeeping beyond the current state, e.g. "any number as 0 to 9 possibly infinite amounts" `[T]` and a JSON object with a fixed key set. **Context-free** — a pushdown automaton, "a stack of prior values" `[T]`; it expresses counted nesting: "match numbers of a's and b's or match numbers of parentheses… or json max numbers of curly braces" `[T]`. **Turing-machine class** — multiple stacks or a tape, "a strictly more expressive thing than a context-free language," which is where "uniqueness of keys," "variables must be defined before they are used," and "you can use numpy as long as you imported numpy further up" live `[T]`. The practical rule: choose your enforcement mechanism by asking which row the constraint is in. Two operational consequences. Fixed nesting depth stays regular, because you can define "the state for brackets nested five times versus six versus seven" — at combinatorial cost, which is why flat automata stop scaling. And anything above context-free is not enforceable token by token at all, so it goes to post-hoc validation and repair. One honest limit to carry: the lecture says "there's a whole broad world outside of these two things" without naming context-sensitive or decidable classes, so **the corpus does not give the full Chomsky hierarchy** — do not present one as if it were quoted.
**Signal:** Places three concrete constraints in three rows, and flags that the corpus stops short of the full hierarchy rather than filling the gap from memory.
**Follow-ups:**
- *Which row is a flat per-customer JSON schema?* Regular.
- *Which row is recursive line-item nesting?* Context-free — Q12.
- *What is not context-free that engineers assume is?* "Valid C," because of define-before-use — Q13.
**Red flags:** Believes a regex library is as expressive as a grammar; cannot place define-before-use.

#### T03-Q8 · End-verifiable constraints: why a mask cannot do it
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** The constraint is "the sentence must mention car, drive and snow." Can you mask for that? What replaces the mask?
**Model answer:** No. It is **end-verifiable only**, and that changes the machine. A mask requires that at each step you know whether the constraint is currently satisfied; for "must contain concepts X, Y and Z" you cannot know that until the sentence ends, so there is no legal-token set to compute. What replaces it is **search**. The lecture's illustration: with the prefix "I drive my car during the," "summer" is the high-probability continuation but makes "snow" hard, while "winter" is slightly lower probability and much more likely to satisfy the constraint `[T]` (CMU lecture 6). That is a lookahead problem — you choose a path by where it can still end up, not by the next token's probability. The machinery is from the search lecture: a weighted finite-state automaton over partial hypotheses, with **recombination** of equivalent states so that hypotheses reaching the same automaton state are merged rather than explored separately `[T]` (CMU lecture 5). Without recombination the cost is exponential, and recombination is the whole reason the search is affordable at all. The case study's §5.2 prices this honestly: search instead of mask "handles constraints that are only checkable at the end" but "breaks when the constraint interacts with every token — you lose the benefit of pruning," and it defers to [T02](../01-case-studies/T02-search-decoding.md) for the family. The division to state: token-wise-verifiable goes to a mask; end-verifiable goes to a search or to generate-and-filter.
**Signal:** States that no legal-token set exists for an end-verifiable constraint, and names recombination as what makes the search tractable.
**Follow-ups:**
- *What is the cheap fallback?* Generate-and-filter, whose cost depends on how common a violating output is — Q20.
- *When is filtering better than searching?* Low violation rates, where full-sequence checking is cheap `[T]`.
**Red flags:** Tries to compile it into a grammar; says "just put the concepts in the prompt."

---

### Mechanism

#### T03-Q9 · Token healing: the problem, the procedure, the invariant
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** What is token healing, what problem does it solve, and what exactly does it guarantee?
**Model answer:** The problem is that template-driven generation produces "token boundaries that are quote unquote unnatural" `[T]` (CMU lecture 6). The lecture's example: an unconstrained model emits "the URL is http slash" as a single token, but the automaton path emits a colon and then needs two slashes — and if pre-training "always tokenized this as this single token dot slash in uh URLs… this could be a relatively difficult token to predict. There could be like sort of not a lot of probability mass on it" `[T]`. The automaton has pushed the model onto a token it has rarely seen in that position. **The procedure:** "we'll roll back a token or very rarely to [two] of generation and we'll just require that the next token starts with that token that we would have predicted before… we'll look at all of our candidates for the next token and we'll eliminate everything that doesn't start with colon. So colon is still a valid next token, but so is colon slash, which has a lot higher probability" `[T]`. **The invariant** is surface-form preservation: "we're not actually changing our output string. We're just changing the tokens of the output to get there" `[T]` — "HTTPS" as two tokens or three, the same string out. Two honest limits. The corpus carries **no measured number** for healing's benefit — no accuracy or latency figure — and the rollback length is the lecturer's procedure description, not a measurement; any overhead figure you quote must be your own arithmetic with the trigger rate shown as an assumption `[D]`. And healing is a heuristic with no guarantee: it raises the probability of the boundary token, it certifies nothing.
**Signal:** States the invariant as surface form rather than probability, and volunteers that no healing measurement exists in the corpus.
**Follow-ups:**
- *When should it fire?* Four heuristics, none of them an accepted standard — Q14.
- *Why does masking create this problem?* The mask forces the automaton's path, which can be an unnatural boundary — Q14.
**Red flags:** Describes healing as "fixing the output"; claims a measured quality gain; thinks it changes the output string.

#### T03-Q10 · Draw it: schema → automaton → mask
**Difficulty:** L4 · **Depth expected:** whiteboard
**Question:** Compile `{"name": "Taylor Swift", "birth year": 1989}` into a state machine on the board, then show me the masking step.
**Model answer:** Four states, drawn as the lecture does. **State 0:** "there's only one valid token to start the JSON, which is the opening curly brace"; invalid branches are crossed out. **State 1:** two options — the `name` key or the `birth year` key. **State 2** (inside `name`): only letters, implemented as "a reax specification of each of these" — a regex per field `[T]`. **State 4** (inside `birth year`): only digits, then a comma, then optionally the other key. The accept state is "a second concentric circle"; the closing brace ends generation `[T]` (CMU lecture 6). Then the mask: at each step take the current state's outgoing transitions, compile them to a set of legal **token ids**, and "set the probabilities to everything that isn't a valid transition in this graph to be like arbitrarily low… by adding like a large negative right before softmax" `[T]`. Formally, allowed tokens keep their logit, everything else goes to `-inf`, and the softmax renormalises. The states compile "down into a pretty like efficient just like check against a list and mask out the logits" `[T]` — the latency-relevant claim being a per-state token-id list rather than a per-step regex. Draw it rather than describe it, because the audience's next question is always "how do you know it's right," and the drawing shows exactly what the compiler must guarantee per state: at least one legal token, and a reachable accept state.
**Signal:** Draws the states, marks the accept state, and compiles transitions to token ids — rather than saying "we filter invalid tokens."
**Follow-ups:**
- *Where does this break?* Key uniqueness, optional-field combinations, and nesting — Q11 and Q12.
- *What if the model wants to explain before the JSON?* It is stuck at state 0; allow an explicit free-text prelude state or a separate thinking channel `[T]`.
- *Is this what production engines actually do?* "I would be surprised if they were doing something different. Um because this is like an exact solution to the problem" `[T]` — with the nesting caveat, and with the note that it is supported in the engine ecosystem. Careful with the vendor name here: the transcript renders it as "BLM or SG link," an **ASR garble** of the vLLM/SGLang family, so do not quote that string as a product name.
**Red flags:** Describes an FSA as "checking the JSON at the end"; cannot say where in the decode step the mask is applied.

#### T03-Q11 · The hand-drawn FSA's defect list
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** The lecture has the class attack its own JSON automaton. What does a hand-drawn FSA get wrong, and which of those defects must a compiler fix?
**Model answer:** Six defects, and they are exactly the bug list a real schema compiler must not have `[T]` (CMU lecture 6): (1) no length limit — "an infinite length name"; (2) repetitive keys — "a thousand birth years in it, which is probably not true for a single person"; (3) `birth year` can be omitted entirely even when the schema requires it; (4) no constraint that the birth year is one number rather than "a thousand"; (5) "a name… can't have any spaces in it" — the regex is too tight; and (6) **no nested JSON — "we cannot do that in this type of construction at all."** The first five are compiler requirements: length bounds, key multiplicity, required-field coverage, single-value constraints, and a character class derived from the field's actual type rather than guessed. The sixth is architectural. The lecture's fix is separate states for "name but no birthday" and "birthday but no name," which works — and is precisely why flat automata do not scale, because the automaton grows with the number of *combinations* of optional fields, not with the number of fields. This list is why "the mask is broken" incidents are almost always grammar bugs, and why the case study's tuning order puts grammar correctness first, with a survivors-per-step instrument, before anything else is touched. Note the direction of the defects: five of the six are *too permissive* — valid-looking records that violate intent — and one is *too restrictive*, and it is the restrictive one that surfaces as an engine error rather than a bad record.
**Signal:** Reproduces at least four of the six defects and separates the permissive ones from the restrictive one.
**Follow-ups:**
- *Which defect needs a pushdown?* Nesting, (6) — Q12.
- *Which needs more than a pushdown?* Key uniqueness, (2), when the key set is unbounded — Q7.
**Red flags:** Believes a compiled grammar is automatically correct; names only the nesting defect.

#### T03-Q12 · The pushdown claim, and what breaks if you ship an FSA
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** You build an FSA-based masker, call it a JSON-schema engine, and it is correct on 200 of your 240 schemas. What happened on the other 40?
**Model answer:** They are the recursive ones. The lecture's claim: "in general, anything that supports uh JSON schemas is actually writing push down automa to enforce its constraints, not FSAs" `[T]` (CMU lecture 6). A finite automaton "doesn't have any way of keeping track of how many times it's been in a state before" — it has no memory of depth — so it cannot express "you can start with an arbitrary number of parentheses as long as you end with exactly the same number" `[T]`. Arbitrary nesting is context-free, and the machine for it is a pushdown automaton, which keeps "a stack of prior values": in JSON, tracking the number of open versus closed braces `[T]`. The failure mode is the dangerous kind. On the 200 flat schemas your engine is exactly right, so the system looks validated. On a recursive line-item structure it is *silently* wrong: it either truncates at the depth the flattened automaton covers, or it rejects a legal deeper nesting and the generator stalls on an empty candidate set. Two design consequences. The compiler's target machine must be chosen from the schema's structure rather than from a global setting — the case study compiles regular schemas to a DFA and recursive fragments to a PDA. And the stack lives on the **engine side, per request**, which makes the PDA branch's real risk a concurrency problem rather than a correctness one: per-request stacks interact with continuous batching and paged attention, and the memory cost per concurrent sequence rises. The escape hatch for bounded depth: a depth limit is a *regular* constraint and belongs in the grammar, not in the stack.
**Signal:** Explains that the flat-schema majority hides the bug, and that the stack's real cost is engine-side per-request state.
**Follow-ups:**
- *What about fixed depth 5/6/7?* Expressible as separate states — but combinatorially — Q11.
- *Where does the stack live?* Per request, per sequence, in the engine — Q27.
**Red flags:** "JSON is regular"; cannot say why the flat-schemas-work case is the dangerous part.

#### T03-Q13 · Classify the surfaces: FSA, PDA, or post-hoc
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Three surfaces: a per-customer JSON schema for invoice fields; a SQL predicate over a fixed table set where variables must be defined before use; and a free-text note that must map to one of 900 codes. Assign each a machine and justify it.
**Model answer:** The case study's own three, and each lands in a different row. **Invoice JSON → FSA or PDA, depending on the fragment.** Flat fields are regular — a regex per field, states for key order, an accept state at the closing brace, exactly the lecture's construction. Nested line items are context-free and need the pushdown. Note the classification is per *schema fragment*, not per surface: with 240 customer schemas some will be flat and some recursive, and a compiler that decides globally will be silently wrong on a minority. **The SQL predicate → neither.** "Variables must be defined before they are used" is the lecture's own example of a constraint that is not context-free; it needs more than one stack and is not enforceable "in this kind of like token by token um, checking way" `[T]` (CMU lecture 6). It goes to post-hoc validation with a repair loop, and the honest framing for the customer is that this surface gets a *validator guarantee*, not a *generation guarantee* — two different promises, and only one of them is 100%. **The 900-code list → regular, and structurally the easiest.** A closed vocabulary is a hard mask's ideal case: each code has exactly one representation, so the synonym failure cannot fire. But note the trap the case study flags — the mask guarantees the model picks a *valid* code, not the semantically *right* one, so validity and correctness come apart exactly where the surface looks easiest.
**Signal:** Assigns three different machines and refuses to give the SQL surface a generation-time guarantee.
**Follow-ups:**
- *Why is the code list not a semantic-constraint problem?* Each code has one representation, so a hard mask does not hit the synonym failure — Q19.
- *What if one schema is mostly flat with a single recursive field?* The fragment decides; compile per fragment.
**Red flags:** Gives all three an FSA; claims a grammar can enforce define-before-use.

#### T03-Q14 · Token healing: the triggers, and why it is not always on
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** You have token healing available. Where do you switch it on, and what makes that decision hard?
**Model answer:** The lecture names four trigger heuristics and says explicitly that there is no single accepted rule: a curated list of "common offenders" — colon, space, slash; "when the last token is very short" or single-character; "when the last token doesn't start or end with whitespace or punctuation"; and "when the last token predicted was an exact prefix of another likely token" `[T]` (CMU lecture 6). The three options are the case study's decision table. **Off** is predictable and free, and defensible when templates are whitespace-clean and schemas are ASCII. **Always on** removes the whole class of boundary failures and "breaks nothing" — but "the reason we don't just like automatically do this at every step is it's just expensive. you have to go back and recompute" `[T]`, and it is pointless when the prefix you would heal was already the most likely next token, in which case "nothing has changed" `[T]`. **Heuristic-triggered** is the choice, and its real cost is ownership: the trigger list becomes a versioned artefact that needs tuning, has no accepted standard, and must be re-derived from the corpus whenever a tokenizer change alters which boundaries are unnatural. Where it matters most here is non-ASCII. The whitespace heuristic fails in many languages, but healing a token while keeping the surface form still works — notably for diacritics and combining marks that tokenizers treat as separate tokens `[T]` — which is why the case study's heavily non-ASCII customs corpus is where the invariant test earns its keep. Note the corpus gives **no measured healing overhead**; a cost figure has to be your own arithmetic with the trigger rate stated as an assumption.
**Signal:** Quotes the "expensive / nothing has changed" reasoning and names the non-ASCII case where the whitespace heuristic fails but surface-form healing still works.
**Follow-ups:**
- *What is the test?* Heal → detokenize → compare against the pre-heal surface form, as an asserted invariant, not a dashboard.
- *What is the cost at a 2% trigger rate?* `[D]` Eight extra forward passes against 400 tokens — about 2% overhead, with the 2% trigger rate an assumption, not a measurement.
- *Why does masking make healing more necessary?* The mask forces unnatural boundaries in the first place — Q9.
**Red flags:** Enables healing everywhere "to be safe"; cannot name a trigger; quotes a measured speedup.

#### T03-Q15 · FUDGE: mechanism, truncation, and the non-guarantee
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** The constraint is "be formal" or "do not suggest climbing." Walk me through FUDGE, then tell me what it does not give you.
**Model answer:** FUDGE targets sampling from `p(next token | history, constraint a)` where the constraint is not regex-expressible `[T]` (CMU lecture 6). The rewrite is proportional to `P(constraint satisfied | prefix so far) × P(token)`, and the lecture's point about the exact values is that "we're going to softmax everything anyway. So, we don't care about sort of exact value" `[T]`. **The discriminator** is trained on *every prefix* of labeled documents: formal documents give their prefixes the label "formal." The lecture's illustration — "starting with I has relatively equal probability of coming from formal or informal. But starting with I would appreciate is almost always only seen in the formal data" `[T]`. At decode time you run the LM for candidate tokens, run the discriminator on history + candidate, and multiply, which is an add in log space. The worked example: "do you want" vs "do you prefer" vs "do you thus" — "thus is a really low probability output in general… but thus is a very formal like phrase and so it gets a high formality score but a low overall score"; the winner is "do you prefer," because "want and prefer were relatively even probability but prefer is more formal so it gets updated" `[T]`. **The number that makes it affordable** is the truncation: "they take the top 200 most likely next tokens um and you run on just those 200 instead of all 100,000," plus a very small model, because "this sort of zero one choice is a relatively easy thing to learn" `[T]`. **What it does not give you:** it is "not guaranteed to satisfy the constraint" `[T]`; it needs a discriminator trained per constraint; it runs on every candidate at every constrained step; and it "requires access to logits which is sort of a fundamental requirement," which rules out third-party APIs `[T]`. The top-200 truncation is also the blind spot — a constraint correlated with a token that never reaches the top 200 is invisible to the discriminator, no matter how good it is.
**Signal:** Names the top-200 truncation as both the affordability mechanism and the blind spot, and states the non-guarantee unprompted.
**Follow-ups:**
- *How do you fix the non-guarantee?* Rejection sampling over full outputs as the backstop — Q20.
- *What is the RLHF connection?* A discriminator can predict end-of-decoding reward instead of a label — reward-augmented decoding `[T]`.
**Red flags:** Calls FUDGE "guaranteed"; cannot say where the 200 comes from; treats it as cheaper than masking.

#### T03-Q16 · Contrastive and adversarial decoding
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Explain contrastive decoding. What is the mechanism, the precondition, the cost, and the name-collision trap?
**Model answer:** The premise is that "smaller models or weaker models or subsets of models make different mistakes than the broader system," so you select tokens the strong model finds likely *and* the weak model finds less likely `[T]` (CMU lecture 6). Mechanism: both models see the same input — **"these models need to have the same tokenizer for this to work"** — and "we look at the output logits and then we subtract them from each other" `[T]`. The lecture's example is "Barack Obama who was born in Honolulu, Hawaii": weak models repeat, so subtracting "downweight[s] things like Hawaii and get[s] surfacing things that maybe the weaker model didn't even know like Barack Obama's birth year" `[T]`. The precondition is what kills it in practice — identical tokenizers means it does not cross model families. The cost is flat: "if you're passing these both through the model, you need to use twice as much compute to accomplish the same task," i.e. **2x** `[T]`. The case study's rule follows directly: use it only where the amateur model is already resident for other reasons, which is the revisit condition on its §5.4 choice. **Adversarial decoding** is the same trick with one model prompted two ways: the user instruction under a safe system prompt, and the same instruction under "you are harmful, you should be as offensive as possible, functionally be evil," then subtract — so "this sort of downweights everything that the model thought was a really likely offensive response" `[T]`. You get safety behaviour without training a separate reward model, still at 2x, and it needs both prompts constructed deliberately. **The name collision:** HuggingFace ships "contrastive search which is not the same method" `[T]` — a decoding heuristic, unrelated. Do not conflate them, and do not cite HF's implementation as this technique.
**Signal:** States the identical-tokenizer precondition as the reason it fails in practice, and flags the HuggingFace name collision unprompted.
**Follow-ups:**
- *When is contrastive the right pick?* Two model sizes already served and a constraint you cannot label — Q20.
- *What does adversarial decoding need that a drop-in guardrail does not?* Both the safe and the unsafe prompt must be constructed `[T]`.
**Red flags:** Believes it needs no second model; conflates it with HuggingFace `contrastive search`.

---

### Tradeoffs

#### T03-Q17 · Fine-tune per customer, or enforce at inference?
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Your ML team wants to fine-tune a model per customer schema. The platform team refuses. Argue both sides, then give me the rule.
**Model answer:** Both sides have a real case, and the lecture's rule resolves it. **For fine-tuning:** no runtime cost, and it can improve *semantics* at the same time as format, which masking never does. It is right for a single-customer deployment with a frozen schema, and it is right for a constraint that applies to every call, because a trained-in format costs nothing per request while a mask costs on every single one `[T]`. **Against:** the case study has 240 schemas, so per-customer fine-tuning is 240 models to version, and every schema change becomes a training run. The lecture's framing is the line to quote: "if you wanted to change hello to bonjour for a week… would you want to retrain your model entirely to do that?" `[T]` (CMU lecture 6) — with weekly schema churn the answer is obviously no, and inference-time enforcement turns a schema change into a compile. **The rule:** training constraints in pays off for *broadly applicable* constraints; inference-time enforcement is attractive for *templatic* constraints, hard limits, and per-token-predictable constraints `[T]`. So the split is not "train versus mask" but "which constraint." The JSON skeleton is templatic and per-customer, so it masks. A generic "answer in the customer's language and register" is broadly applicable, so it is a fine-tuning candidate. And a policy constraint like "never recommend a retirement product to this account" is neither — it is a scorer plus a sampled audit. State the audit consequence explicitly: with a mask the guarantee is a compiled artefact you can show and re-derive per schema version; with 240 fine-tunes the guarantee is a training run you cannot reproduce.
**Signal:** Splits by constraint class rather than picking a side, and uses the hello/bonjour framing as a decision rule rather than a slogan.
**Follow-ups:**
- *What about a schema unchanged for two years?* Fine-tuning is defensible — the case study's exception row.
- *What does training never buy you?* A hard guarantee — a fine-tuned model is still sampling.
**Red flags:** "Fine-tuning always wins because there's no runtime cost"; "masking always wins"; no versioning argument.

#### T03-Q18 · Mask versus validator: why do you need both?
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** You have a hard mask. Why also a validator, and what fails if you drop it?
**Model answer:** Because they answer to different failure modes, and the mask is not a correctness guarantee. The case study's architectural claim is exact: the mask guarantees the output is *in the language*; the validator guarantees it is in the *schema version currently in force* — and the case that separates them is a **stale compiled grammar**. A schema is updated, a cached mask still accepts the old shape, and the model emits a record perfectly valid against grammar v3 while the ERP expects v4. Nothing in the generation path can see that; only a validator reading the current schema catches it. The §5.5 decision table lays out three options: trust the mask and skip validation — fastest, and it "breaks exactly once, at the audit"; validate everything — the auditor's requirement, one parse per record, and it catches staleness; validate a sample — cheaper, and useless for a "must never," because sampling cannot certify an absolute. The choice is validate everything, with the authority split stated as a principle: **the mask is a generation mechanism and the validator is a release mechanism.** Two operational consequences follow from treating them as genuinely different systems. They must not share a schema source — the validator is deployed as an independent service that does not import the compiler, so a compiler bug cannot blind the check. And the validator is the only writer to the ERP: anything failing after the retry budget goes to a dead-letter queue for human review, never downstream. The cost is a parse per record, negligible against generation — and it is the only component whose failure is an audit failure.
**Signal:** Names stale-grammar as the separating case and states the generation/release split as a principle rather than a checklist item.
**Follow-ups:**
- *What is the retry budget and why?* Start at 2; raising it is a signal the schema or prompt is wrong.
- *What does the validator not catch?* Semantically wrong but structurally valid records — Q24.
**Red flags:** "The grammar is the validation"; proposes sample-based validation for a "must never."

#### T03-Q19 · Hard-masking a semantic constraint: the three failures
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Someone proposes adding "climbing" to a banned-token list so the model stops suggesting it. Walk me through what goes wrong.
**Model answer:** The lecture calls this "a hardline approach" — very easy to implement, "we just add a giant negative to the logit and softmax again" — and then enumerates three failures `[T]` (CMU lecture 6). **Synonyms:** "Maybe if I don't want to go climbing, I also don't want to go bouldering" `[T]` — banning the token does not ban the concept. **Senses:** the token has legitimate uses — "maybe I want to um talk about like hiking where you have to sort of go climbing up this mountain to see a good view. That's a valid way of talking about hiking" `[T]` — so you kill correct output. **Presupposition:** earlier tokens can presuppose the banned one, so "if the model says a great activity you could do with your friends is rock and then we are not allowed to produce the term climbing, you could wind up with some like really non-naturalistic or nonfluent generation" `[T]`. The empirical anchor is the lecture's own demo: Gemini 1.5, told the user did not want to go climbing, "got it got sort of a couple a lines into this generation. It suggested some great ideas and then it cycled back around to climbing"; roughly ten months later GPT-5 no longer does `[T]` — a model-capability change, not a masking change. So the failure is not hypothetical and the fix is not a tighter grammar. What replaces it: generate-and-filter, whose cost "could be extremely computationally expensive if the thing we're trying to restrict is a relatively common output," with the compensating advantage that full-sequence checking is easier than subsequence checking precisely because you have given up token-wise verifiability `[T]`; a FUDGE-class discriminator; or rejection sampling as the backstop. The policy line is blunt: **never put a semantic constraint in the hard mask**, except in the one shape where each code has exactly one representation.
**Signal:** Names all three failure modes with the rock→climbing example, and states the single condition under which a hard mask is acceptable for a semantic constraint.
**Follow-ups:**
- *What is the cost of the filter alternative?* It scales with how common the violating output is — Q20.
- *What if the constraint is "no PII"?* Also semantic, and also not a token list.
**Red flags:** "Just ban the word"; no awareness of the synonym or presupposition failure; proposes a longer banned-word list as the fix.

#### T03-Q20 · Choosing the semantic mechanism per surface — and pricing it
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** You have four semantic constraints across your surfaces: a formality register, a topic avoidance, a policy list of forbidden recommendations, and a one-code-per-note mapping. Pick a mechanism for each, price it, and tell me what you would monitor.
**Model answer:** Map each constraint to the mechanism that fits its shape, then price it. **The one-code-per-note mapping → a hard mask**, because it is the one case where a semantic constraint is also regular: each code has exactly one representation, so the synonym failure cannot fire. This is the case study's choice for the closed code list. **Formality register → a FUDGE-class discriminator**, because the constraint is not regex-expressible but *is* labelable — you can label every prefix of formal and informal documents and train a small discriminator on that. Cost: the discriminator runs over the **top 200** candidates at each constrained step `[T]`; with a small discriminator that is one batched forward pass over 200 short sequences per constrained step, so on a 40-token constrained span of a 400-token record that is 40 extra batched passes — roughly **10% overhead** on that surface, at the assumed span `[D]`. **Topic avoidance → FUDGE or rejection sampling**, and the choice is economic: rejection sampling's cost "depends on how common the bad output is" `[T]`, so a frequent violation pushes you to a scorer and a rare one keeps the filter. **The forbidden-recommendation policy → a discriminator plus a sampled human audit**, because the surface's real requirement is a violation *rate* under a budget (the case study targets under 0.5%), not a guarantee. **Contrastive decoding** I would not choose here: it is a flat 2x `[T]` and needs identical tokenizers, so it pays only when the amateur model is already resident. **What I would monitor:** because FUDGE's guarantee is absent, the audit *is* the control — a weekly sampled audit of emitted codes against the policy list, plus the discriminator's own accuracy against its held-out set, since the corpus names discriminator drift after a corpus change as a live failure mode.
**Signal:** Prices each mechanism with a corpus number, and refuses a guarantee where the corpus gives none — with the 0.5% budget and 10% overhead both carrying their assumptions.
**Follow-ups:**
- *Why not one mechanism for all four?* The hard mask fails on three of them and the discriminator is overkill on the fourth.
- *Where does the forbidden-recommendation example come from?* It is the **case study's** illustrative constraint, explicitly not a transcript quote; the lecture's own semantic examples are topic ("not climbing") and register (formal/informal) `[T]` — cite the lecture's ones as quotes and the policy list as an example.
- *What is the truncation's failure?* A constraint correlated with a token outside the top 200 is invisible — Q15.
**Red flags:** Reaches for a hard mask for all semantic constraints; quotes an accuracy number the corpus does not contain.

#### T03-Q21 · The guarantee tiers you can offer an auditor
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** The audit committee wants a written statement of what your system guarantees. Write it, and say what you refuse to sign.
**Model answer:** Three tiers, each with the mechanism that carries it. **Tier 1 — a hard structural guarantee.** For any surface whose constraint is regular or context-free, every emitted record is accepted by the automaton compiled from the schema version in force, because illegal tokens are assigned zero probability before the softmax and are therefore unreachable rather than unlikely `[T]`. This is a *generation* guarantee and it is the strongest claim you make. **Tier 2 — a release guarantee.** Every record is validated against the current schema version by a service independent of the compiler, and anything that fails after a bounded retry budget (2 in the case study) goes to a human queue and never reaches the ERP. This is what makes "must never" true even when the mask is stale, and it is the tier the auditor actually cares about, because validation is a hard gate rather than a metric `[D]`. **Tier 3 — a scored budget, not a guarantee.** Semantic constraints, i.e. the policy list, are enforced by a scorer and audited on a sample, with a stated violation-rate budget (the case study targets under 0.5% with no fine-tuning). I would sign that as a *rate with a monitoring obligation* and never as a guarantee, because FUDGE-class methods are explicitly "not guaranteed to satisfy the constraint" `[T]`. **What I refuse to sign:** that a valid record is a correct record; that arbitrary nesting is guaranteed on any surface still using an FSA mask; that define-before-use is enforced at generation time on the SQL surface, since it sits above context-free and goes to post-hoc validation only `[T]`; and any claim backed by an accuracy figure — the corpus's own §9 records that the lecture quotes **no benchmark numbers** for constrained generation at all, no accuracy, latency or human-eval results, so a vendor offering one should be asked for its conditions. The case study is explicit that this distinction is the entire audit conversation.
**Signal:** Writes the guarantee as three tiers with a named mechanism for each, and refuses "valid means correct" before being pushed.
**Follow-ups:**
- *Which tier does the SQL surface get?* Tier 2 only — validation without a generation-time guarantee.
- *What is the monitoring obligation on Tier 3?* A weekly sampled audit plus discriminator-accuracy tracking.
**Red flags:** Signs "100% accurate extraction"; offers a guarantee for the semantic surface; cannot separate syntax from correctness.

---

### Debugging

#### T03-Q22 · Invalid records reached the ERP. Diagnose.
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** A downstream integrity alert fires: records that fail the customer schema are in the ERP. Walk me through it.
**Model answer:** This is the incident the whole architecture exists to prevent, so the first question is not "what is wrong with the model" but **"was the validator bypassed, or was it reading a different schema version than the compiler?"** That is the case study's incident #1, and it points at version skew rather than at generation. Two branches. **Version skew:** the compiled grammar and the validator disagree about which schema is in force — most often a schema write triggered a compile while the validator's schema source lagged, or the validator is being served a pinned old version. The decisive instrument is the case study's most informative metric: **rejection rate by schema version**, where a step change indicates a compile or a version skew. If the validator rejected nothing while records are invalid, the validator was not on the path — check the ingestion route and whether it shares a code path with the validator or has quietly acquired its own. **Validator bypass:** a new ingestion path, a backfill job, or an emergency manual write skipped the gate entirely. Note what the mask can and cannot explain here. A mask changes *which* output you get; it does not by itself produce schema-invalid output — unless the compiled grammar is itself stale, which is branch one. The fix order matters: halt the ingestion path, re-validate the affected window, then replay from the dead-letter queue against the corrected grammar or schema. Do not blindly re-run generation — the documents were valid inputs and the failure was in enforcement, so regenerating burns tokens to reproduce the same records. The follow-up I would insist on is a release-time assertion that the validator's schema version and the compiler's version are equal, because this is a deployment-ordering bug, not a model bug.
**Signal:** Goes to version skew first and names rejection-rate-by-schema-version as the decisive instrument.
**Follow-ups:**
- *Why is the validator's independence load-bearing?* So a compiler bug cannot blind the check — Q18.
- *What if the mask is stale and the validator is current?* The validator catches it and you see a rejection-rate step change — which is the system working as designed.
**Red flags:** Blames the model; proposes retraining or tightening the grammar; never asks about the validator.

#### T03-Q23 · Generation stalls or errors mid-record
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Generation dies at a specific point in a specific schema. What is happening, and what do you change?
**Model answer:** The most likely cause is an **empty or near-empty candidate set** — the grammar forbids every token the sampler would accept at that step. The case study is precise about this: the mask "leaves an empty or near-empty candidate set; the sampler emits a token with near-zero mass or errors," and "the fix is almost always a schema/grammar bug (an over-tight regex), not an engine bug," with the rule that a zero-survivor step must never proceed silently. The instrument is a **survivors-after-masking counter** with an alert on zero, and this is why the case study's tuning order puts grammar correctness first — before the healing trigger list, before the retry budget — because most "masking is broken" incidents are grammars that forbid a token the model needs. In practice the culprits are the hand-drawn-FSA defects: an over-tight character class for a field (the lecture's "a name… can't have any spaces in it" `[T]`), a length bound that excludes a legitimate value, or a required field the model wants to omit. Two other causes are worth ruling out. **Recursive depth:** a schema whose nesting exceeds the depth the automaton or stack bound covers truncates or loops — the depth counter per generation is the instrument, and the fix is a depth bound in the grammar contract, since a depth limit is a *regular* constraint and belongs there rather than in the stack. **A prelude problem:** the model wants to explain before emitting JSON, the automaton is stuck at state 0, and it looks like a stall at the first step; the fix is an explicit free-text prelude state or a separate thinking channel. After diagnosis: patch the grammar, redeploy, re-run the batch — and add the two per-state unit-test assertions, at least one legal token and a reachable accept state.
**Signal:** Names the empty candidate set first with the survivors counter as its instrument, and cites the runbook's per-state assertions.
**Follow-ups:**
- *Why is the fix in the grammar rather than the engine?* A correct grammar can always emit; the defect is expressiveness, not execution.
- *What if the grammar is right and it still stalls?* Check whether the mask is being applied to the right row — batch mask collision, Q27.
**Red flags:** Raises the temperature or the retry budget; declares it an engine bug; has no diagnostic instrument.

#### T03-Q24 · Field values wrong but structurally valid
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** A customer disputes an extracted field. The record validates perfectly. Where do you look?
**Model answer:** Structurally valid is the expected residual — the mask never promised semantics — so this is a *correctness* investigation, not an enforcement one. Check in the case study's order. **Token healing first.** The case study's incident #3 is exactly this symptom, and the reason it comes first is that healing is the one component that changes *values* while preserving structure: its invariant is surface-form preservation — "we're not actually changing our output string" `[T]` — and the failure is that the invariant was violated, typically on a non-ASCII field where the whitespace trigger fails and a diacritic or combining mark is dropped. The test is the invariant run as a regression: heal → detokenize → compare against the pre-heal surface form, field by field, plus a field-level diff against the source document. The remediation is to disable healing for that field class and re-run the affected documents. **Second, the prompt and template.** The corpus's measured case is a prompt change that cost 27 points of validity (98% → 71%) `[T]`, and the mechanism works in the other direction too: a template that mis-describes a field yields confidently wrong values with perfect structure. **Third, the document itself** — a scanned remittance advice where the value is genuinely unreadable produces the same symptom, and no model change fixes it. **Fourth, and this is the honest part:** this is the class the mask cannot see, and the case study says so — the residual 0.5% is "not a masking failure; it is the class the mask cannot see (semantic errors, stale schemas, upstream document problems)". So the answer to the customer is a domain check, not a grammar change; and if the field matters enough, it needs a field-level verifier.
**Signal:** Puts token healing first with the surface-form invariant as the test, and refuses to fix a semantic error with a structural change.
**Follow-ups:**
- *How would you catch it earlier?* Field-level diff against the source as a production metric, not only a test.
- *What if it is a systematic bias rather than noise?* That is an extraction-eval problem, not a decode-policy one.
**Red flags:** Tightens the grammar; blames the model without checking healing; has no invariant test.

#### T03-Q25 · A semantic policy breach found in an audit
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** An audit finds records that were structurally valid but breached a policy constraint. Reconstruct the diagnosis and the fix.
**Model answer:** The diagnosis is one question with three answers, and the case study's incident #5 poses it exactly: was the constraint in the hard mask, where it fails silently due to synonyms; in the discriminator, where it is not guaranteed; or **nowhere**? **In the hard mask:** the failure is silent by construction. Someone added the forbidden term to a banned-token list, the model routed around it with a synonym — the lecture's bouldering case `[T]` — and the constraint looks enforced in every structural metric while being semantically absent. The tell is the combination: structurally perfect output, and the banned term never literally appears. The fix is to remove the hard mask, because the lecture's three failures are not tunable. **In the discriminator:** the failure is the documented non-guarantee — FUDGE is "not guaranteed to satisfy the constraint" `[T]` — so a breach is expected at some rate and the question becomes whether the observed rate is inside the budget. If it is, the finding is a *monitoring* finding and the sampled human audit was the control that caught it. If the rate drifted upward, suspect **discriminator drift** after a corpus change and check the discriminator's own accuracy against its held-out set before touching anything else. **Nowhere:** the most common case and the worst — a constraint someone assumed the prompt enforced. The fix sequence: move it to a discriminator plus a sampled human review, train on a labeled corpus if one exists, and re-audit the *window* rather than the sample, because you now know the rate was never measured. What not to do: tighten the grammar. The case study's policy is explicit that a semantic constraint is never attempted as a hard mask and that the residual semantic class is handled by scoring or by a human, never by the automaton.
**Signal:** Runs the three-way diagnosis and treats the "nowhere" branch as a measurement gap rather than a model failure.
**Follow-ups:**
- *What is the leading indicator?* Discriminator accuracy against its held-out set, monitored on a schedule.
- *How do you know the budget is right?* It is a product decision — the case study sets under 0.5% with no fine-tuning, and at 10x the residual rate is the thing to watch.
**Red flags:** Proposes banning more tokens; treats a within-budget breach as an incident with no budget defined; does not re-audit the historical window.

---

### Scale and design

#### T03-Q26 · Ten times the volume: what breaks first?
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** The estate grows 10x — 50M documents a month. What breaks, what inverts, and what survives?
**Model answer:** **The first thing that breaks is the dead-letter queue, not the model.** Human review does not scale horizontally at the same rate as GPU capacity, and the case study's §8 arithmetic makes it arithmetic rather than opinion: at 5M documents and a 0.5% residual, exhausted records are 25k a month, and at the §8 assumption of 45 seconds of human time each that is about 312 hours — roughly two reviewers. At 50M the same rate is 250k records, which the case study calls "a division" of people. So the leverage at 10x is in driving the residual rate down, not in throughput, and that is what promotes FUDGE-class or reward-augmented scoring from "nice" to "staffed." **The validator becomes the bottleneck before the GPU does.** A parse per record at 50M records a month is a real service with its own scaling problem, and it is the only component whose failure is an audit failure. **The grammar compiler becomes a platform product.** At 240 schemas it is a library; at 2,400 it needs a registry, a test harness, a canary mechanism and an owner — and the compile step, not the mask, is the thing you regret not building first. **What becomes a memory-planning problem:** pushdown state per sequence. Per-request stacks live in the engine, so they interact with paged attention's block accounting and continuous batching's per-step scheduling, and the memory cost per concurrent sequence rises with the fraction of recursive schemas — a schema-mix shift is therefore a capacity event. **What inverts:** the mask's runtime cost stays negligible, but cache invalidation becomes the risk rather than compile time, because at daily schema churn a stale compiled grammar has a smaller window in which to be wrong and a larger blast radius per hour. **What survives:** the mask/validator split, the FSA/PDA/post-hoc taxonomy, and the refusal to put semantic constraints in a hard mask. Those are structural, and the case study says scale does not change them.
**Signal:** Names the human-review queue as the first breakage and the validator as the second, and refuses to treat this as a GPU-scaling problem.
**Follow-ups:**
- *What is the metric to watch?* The residual rate, not throughput.
- *What changes about schema churn?* Compile time stops mattering and cache invalidation becomes the risk — Q28.
**Red flags:** "Add GPUs"; assumes masking becomes the runtime bottleneck; no view on the human tier.

#### T03-Q27 · Per-request grammar state under continuous batching
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** Two requests with different schemas land in the same batch. What can go wrong, and how do you build against it?
**Model answer:** This is the concurrency hazard of masking in a paged-attention engine: masks are **per-request** but the batch is shared, so the logit processor must be indexed per sequence. If the grammar state is held per batch — or a single processor instance is reused across rows — the masks collide and one request's schema constrains another's tokens. The case study's symptom is exactly right: "cross-contaminated outputs under load," detected by a golden-set regression run *under concurrency* rather than serially, because a single-request test will never see it. The mechanics: the mask is a per-row operation on the logits, and the automaton state is a per-sequence object — for a DFA that is one integer, for a pushdown automaton it is a per-request stack, which is where this stops being a correctness question and becomes a memory-planning one. Note what is *not* shared: grammar state does not live in the KV cache. The sequences it constrains do, and the mask's job is to change their length distribution, so the two interact through occupancy rather than through state. Continuous batching raises the stakes because a step batches sequences at different positions under different grammars — and the batching itself is the thing the cost ladder credits with taking serving from 100 cost units to about 42, so the implementation must not defeat it `[T]` (LLMOps cost talk). Build against it three ways: one processor instance per sequence, keyed by batch index; a concurrency regression test in the golden set; and a development assertion that fails if two rows in a batch ever resolve to the same automaton state object.
**Signal:** Names per-sequence processor state as the requirement and explains that a serial test is why the bug reaches production.
**Follow-ups:**
- *Does grammar state belong in the KV cache?* No — it constrains the sequences that do.
- *What does the PDA cost you here?* A stack per concurrent sequence, which is a memory budget question — Q26.
**Red flags:** "The mask is a global logit processor"; tests only at concurrency one.

#### T03-Q28 · Design the compile-and-release pipeline for 240 schemas
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** 240 customer schemas, changing weekly, with a hard validity requirement. Design the pipeline that turns a schema into a deployed constraint.
**Model answer:** Five stages, in the case study's runbook order. **One — compile on write, not on read.** A schema write triggers the compile, and the compiled artefact is content-addressed and versioned alongside the schema. This is the most important ordering decision in the system: compiling on read is the usual latency bug and the usual source of stale-grammar incidents. Target under 2 seconds per schema, which is generous for an automaton build but forces a dependency-light compiler. **Two — per-fragment machine selection.** The compiler picks the target from the schema's structure rather than from a global setting: flat and fixed-nesting fragments become a DFA; recursive fragments become a pushdown automaton with the stack on the engine side; and anything needing more than one stack — key uniqueness, define-before-use — is refused at compile time with a clear error rather than silently compiled to something weaker. **Three — grammar unit tests as a release gate.** Two assertions per state: at least one legal token, and a reachable accept state. Both come from real defects — the first prevents empty candidate sets, the second prevents a grammar that can never finish. **Four — canary, then promote.** Run a new grammar on 1% of a customer's traffic with the validator in log-only mode and compare reject rates before promoting. **Five — the validator ships independently**, as its own service with its own schema source, and must not import the compiler, so a compiler bug cannot blind the check. **Instrument from day one:** survivors-per-mask-step with an alert on zero, rejection rate by schema version, first-attempt validity rate, retry histogram, dead-letter depth. Build the compile step properly first, because at 10x it is the component that has to become a product.
**Signal:** Puts compile-on-write and validator independence as the two load-bearing ordering decisions, and refuses to compile unsupported constraints silently.
**Follow-ups:**
- *Why must the validator's schema source be separate?* So the mask cannot hide its own bug — Q18.
- *What is the canary comparing?* Reject rates, log-only, on 1% of a customer's traffic.
**Red flags:** Compiles per request; shares one schema source between compiler and validator; ships without the two per-state assertions.

#### T03-Q29 · Design the constraint stack for a new surface — and say what you would not do
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** A new surface arrives: a tool-calling agent that emits JSON arguments against 12 tool schemas, plus a free-text rationale field, and a policy forbidding certain product recommendations. Design it.
**Model answer:** Classify first, choose a machine per class, then decide the guarantee level and say it out loud. **Tool arguments → token-wise verifiable**, regular or context-free depending on the schema, so a compiled grammar mask with `-inf` before softmax. Twelve schemas is the easy regime: compile on write, cache the artefact, per-sequence processor state, survivors counter. **The rationale field → unconstrained text**, and the important decision is to *keep* it unconstrained rather than letting the grammar bleed into it. Two patterns work: emit the rationale in a separate channel that is never masked, or let the grammar allow an explicit free-text prelude state so the model is not stuck at state 0 wanting to explain before the JSON `[T]`. **The product policy → semantic**, so a FUDGE-class scorer over the top-200 candidates `[T]` with a labeled corpus, or a post-hoc check with a retry, plus a sampled human audit — and it never goes in the mask, because the three hard-mask failures will find it `[T]`. **The guarantees I would state:** a hard structural guarantee on the tool arguments; a release guarantee from an independent validator that is the only writer to the tool executor; and a *budgeted rate* on the policy, never a guarantee, because the discriminator is not guaranteed. **What I would not do:** I would not put the policy in the grammar; I would not let the tool schema and the policy share one enforcement path, because the first is exact and the second is statistical, and conflating them makes the exact one look bad and the statistical one look exact; I would not run contrastive decoding here, because it costs 2x and needs identical tokenizers and I have neither a resident amateur model nor a reason; and I would not promise that a schema-valid tool call is a correct one — the mask guarantees the call parses, not that the arguments are right.
**Signal:** Separates three constraint classes into three mechanisms, states three different guarantee levels, and rejects contrastive decoding with a priced reason rather than by omission.
**Follow-ups:**
- *Why not one grammar for everything?* The policy is not expressible and the rationale is not constrained.
- *What would change your mind on contrastive?* A second model size already resident for other reasons — the case study's own revisit condition.
**Red flags:** One mechanism for all three; promises the policy; hides the rationale inside the JSON schema.

---

## Whiteboard exercises

### Exercise 1 — Compile a schema and mask a step
**Prompt.** On the board, take the schema `{"name": "Taylor Swift", "birth year": 1989}` and produce the automaton a masker would compile, the legal-token set at two named states, and the exact logit operation at a masked step. Then say where this automaton is *wrong* for real production schemas.

**What to produce.** The state diagram with the accept state marked, the mask formula, and a ranked list of the automaton's defects with the machine each one needs.

**Expected whiteboard.**

```
State 0  '{'  ->  State 1
State 1  "name" -> State 2      |  "birth year" -> State 4
State 2  [a-zA-Z]+  -> State 3   (regex per field)
State 3  ','  -> State 1        |  '}' -> ACCEPT
State 4  [0-9]+ -> State 5
State 5  ',' -> State 1         |  '}' -> ACCEPT

Mask at State 4:
  allowed(4) = { token ids whose text is a digit }
  z'_i = z_i            if i in allowed(4)
  z'_i = -inf           otherwise
  p = softmax(z')       <-- before softmax, so illegal = probability 0

Defects -> machine that fixes each:
  1. infinite-length name ......... regex quantifier bound   (regular)
  2. a thousand birth years ....... key multiplicity         (regular)
  3. birth year omittable ......... required-field states    (regular)
  4. "one number" not enforced .... scalar cardinality       (regular)
  5. no spaces in a name .......... character class from type (regular)
  6. no nested JSON ............... PUSH DOWN automaton      (context-free)
     (and key uniqueness .......... neither — post-hoc only)
```

**Grading rubric.** Full marks require: the four states plus the accept state drawn as the lecture draws it; the mask written as `-inf` **before** softmax with the words "probability zero," not "unlikely"; at least four of the six defects named with the observation that five are too permissive and one too restrictive; and nesting assigned to a pushdown with key uniqueness refused a token-level answer entirely.
- Missing the accept state, or describing the mask as a post-hoc filter, caps the answer at half.
- Saying "a regex handles JSON" without the nesting caveat fails the exercise.

### Exercise 2 — Invalid records reached the ERP
**Prompt.** `Ledgerline` has shipped 400 invalid records to two customers' ERPs overnight. You have the validator logs, the schema registry, and the compiler's artefact versions. Produce the diagnosis, the containment, and the permanent fix. You may not re-run generation.

**What to produce.** A decision tree that separates the two candidate causes, the metric that identifies each, the containment order, and the release-time control that prevents recurrence.

**Expected whiteboard.**

```
Symptom: schema-invalid records in the ERP (a "must never")

Branch A - version skew
   test: rejection rate BY SCHEMA VERSION  -> step change?
   test: compiler artefact version  vs  validator schema version
   verdict if mismatch: the mask produced v3-shaped records; ERP wants v4

Branch B - validator bypass
   test: did the validator see these request_ids at all?
   verdict if absent: new ingestion path / backfill / manual write skipped the gate

Note: the mask alone does NOT explain invalid output
      (it changes which output you get, not whether it validates)
      unless the compiled grammar itself is stale -> Branch A

Containment order:
  1 halt the ingestion path        2 re-validate the affected window
  3 replay from the dead-letter queue against the corrected grammar

Permanent control:
  release-time assertion:  compiler.schema_version == validator.schema_version
  alert:                   rejection rate by schema version, per-surface floor
```

**Grading rubric.** Full marks require: going to version skew *first* and naming rejection-rate-by-schema-version as the decisive instrument; correctly stating that a correct mask cannot itself emit invalid records, so the failure is enforcement-side; a containment order that halts, re-validates, and replays rather than regenerates; and a release-time version-equality assertion as the permanent control.
- Blaming the model, or proposing to tighten the grammar, fails the exercise outright.
- Proposing to re-run generation for the affected documents loses a mark, since the inputs were valid.

### Exercise 3 — Choose the enforcement mechanism for a three-surface estate
**Prompt.** Three surfaces: (a) an audited invoice extractor with 240 per-customer schemas, some with recursive line items; (b) a policy-sensitive recommendation surface where a forbidden product group must not be suggested; (c) a free-text note that must map to one of 900 codes. Pick the mechanism, the machine, and the guarantee level for each, name one alternative you rejected and why, and give the wrong-if signal you would monitor.

**What to produce.** A decision table with mechanism, machine, guarantee tier, rejected alternative and monitoring signal per surface, plus the processor order for surface (a) and the explicit list of what is *not* guaranteed.

**Expected whiteboard.**

| Surface | Mechanism | Machine | Guarantee | Rejected | Wrong-if signal |
|---|---|---|---|---|---|
| (a) Invoices | logit mask | DFA per flat fragment, **PDA** for recursive | hard structural + validated release | per-customer fine-tuning (240 models to version) | rejection rate by schema version; survivors-per-step = 0 |
| (b) Policy | FUDGE-class scorer + sampled human audit | none — semantic | **budgeted rate**, never a guarantee | hard token mask (synonyms, senses, presupposition) | weekly audit breach rate; discriminator accuracy drift |
| (c) 900 codes | logit mask over the closed list | DFA | hard structural + validated release | free-text generation + lookup | share of emitted codes outside the list |

```
Processor order for (a):
   [ schema mask ] -> [ penalties ] -> [ temperature ] -> [ top-p ] -> draw
        ^ first: ~10 legal tokens out of ~100,000 means truncating
          before masking spends the budget on tokens you discard

NOT guaranteed:   a valid record is a correct record;
                  key uniqueness or define-before-use from any grammar;
                  any semantic constraint from the hard mask;
                  an accuracy or latency figure — the corpus has none
```

**Grading rubric.** Full marks require: the recursive fragment routed to a pushdown while the flat fragment stays a DFA, with the classification stated as per-fragment; the policy surface given a *rate with a monitoring obligation* rather than a guarantee, and the hard-mask rejection justified with the synonym failure; the mask placed first in surface (a)'s processor chain on the ~10-of-100k argument; and an explicit non-guarantee list that includes "valid does not mean correct" and the absence of any benchmark figure in the corpus.
- Giving surface (b) a guarantee of any kind caps the answer at half.
- Choosing fine-tuning for surface (a) without pricing the 240-model versioning cost loses a mark.

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_6_Other_Controlled_Generation_Methods.txt` — the syntactic/semantic and token-wise-vs-end-verifiable taxonomy; the JSON-as-state-machine construction with the accept state; the `-inf`-before-softmax mask; the ~10-of-100k sparsity characterisation; the six hand-drawn-FSA defects; the regular/context-free/Turing-machine hierarchy, the depths 5/6/7 illustration and the "JSON schema engines are pushdown automata" claim; the library survey (llama.cpp grammars, the Willard–Lou Outlines work, OpenAI structured outputs formerly JSON mode, Gemini typed schemas, HuggingFace's missing templatic constraints and its `token_healing` flag, the `contrastive search` name collision); token healing's procedure, surface-form invariant, four triggers and cost; FUDGE with the top-200 truncation and the non-guarantee; contrastive and adversarial decoding at 2x with the identical-tokenizer precondition; the hardline-masking failures with the Gemini 1.5 climbing result; generate-and-filter's cost structure.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_5_A_and_Best_First_Search.txt` — weighted finite-state automata over partial hypotheses and hypothesis recombination, the machinery that end-verifiable constraints require for a tractable search.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_3_Common_Sampling_Methods.txt` — the 128k vocabulary figure used as the denominator behind the sparsity argument.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Prompt_Management_as_Code_Versioning_Injection_DSPy.txt` — the measured 98% → 71% valid-JSON regression from one added system-prompt sentence, and the explicit output-schema-plus-validation guidance.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — the 100 → 42 → 26 → 11 serving cost ladder, used to price what continuous batching is worth and therefore why per-request grammar state must not defeat it.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/05-prompting-and-context/06-structured-generation.md` `[R]` — the engine-masks-the-vocabulary description of JSON mode, the "model's willingness vs engine's inability" framing, the CFG/regex Outlines pattern, the multi-stage extraction pattern, the validation-and-recovery loop, and the schema-complexity-versus-information-integrity tradeoff with omission hallucination.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/16-case-studies/01-enterprise-rag.md` — house style reference.
