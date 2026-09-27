# IQ-12 — Interview Questions: Domain-Adaptive Continued Pretraining (DAPT/CPT)

| Field | Value |
|---|---|
| **Module** | IQ-12 — Domain-adaptive continued pretraining: corpus engineering, forgetting, and the eval that decides |
| **Pairs with** | `case-studies/CS-12-Domain-Adaptive-Continued-Pretraining.md`, `cheat-sheets/CH-12-Domain-Adaptive-Continued-Pretraining.md`, `code/03_continued_pretraining.py` |
| **Total questions** | 54 across Levels 1–5, plus 12 rapid-fire and 2 whiteboard tasks |
| **Levels covered** | L1 screen · L2 working engineer · L3 senior · L4 staff/system design · L5 debugging |
| **Source material** | CS-12 §4–§19 (the derivations, tier tables and numbers), CH-12 §1–§12, and the verified `--help` surface of `code/03_continued_pretraining.py` |

**Ground truth used throughout:** CPT optimises `−log p(x_t | x_{<t})` over **every token of raw text** — `labels = input_ids`, no prompt, no response, no masking (§4.3.2). The scale-free volume unit is `τ_cpt = tokens × epochs / params`, with **T2 (τ = 10⁻³) the minimum viable run** and **T3 (τ = 10⁻²) the recommended target** (§4.10.3). The only trustworthy metric is **held-out domain perplexity on a document-level split of deduplicated data**, reported alongside general PPL and a catastrophe probe (§12.1, §12.2, §12.5). The four forgetting dials rank **replay → learning rate → trainable parameter count → duration** (§4.9.2, §4.9.3).

**Where this file sits.** The CPT interview bank did not exist until now; CS-12 §20, CH-12 §13 and CH-16 §13 already forward-reference it, and those pointers now resolve here. CPT is the stage *before* IQ-13: this bank covers **raw-text training, corpus provenance, deduplication, forgetting and the held-out-PPL gate**; IQ-13 covers instruction masking and chat templates; IQ-11 covers what you do to the artifact afterwards. If a question is about response masking or preference pairs, it belongs in IQ-13, not here.

---

## How To Use This File

- **L1 — phone screen (60–90 s per answer).** Vocabulary: what CPT optimises, why it takes a *base* model, what a held-out split is for. Do not let a candidate say "CPT is SFT on raw text" and move on.
- **L2 — working engineer (3–5 min).** Has built the pipeline. Flags, defaults, dedup thresholds, the collator, the volume verdicts, and the arithmetic that says whether a run is worth scheduling.
- **L3 — senior (5–10 min).** Mechanism: why gradient geometry differs from SFT, why duplicates hurt more at CPT scale, why one epoch, why a lower LR is not free, what the loss curve cannot tell you.
- **L4 — staff / system design (10–15 min).** Decision order against RAG and prompting, the eval harness as a deliverable, corpus licensing, and the rollback pair.
- **L5 — debugging (5–10 min).** Symptoms with no exception: a falling loss, a fluent-but-wrong extractor, a perplexity curve that turns. Name what you look at first, what it rules out, and the fix.
- **The meta-rule.** "It depends" fails unless it immediately says *what* it depends on and then picks a default. "CPT or RAG? Depends on whether the gap is knowledge or register — if the model writes correct facts in the wrong voice, that is CPT; if it lacks the fact, that is RAG, and RAG is 100× cheaper" is a pass.
- **The three answers that get people hired in this module.** (1) *"Training loss is not evidence — it falls by construction, it is depressed by padding, and it is lowered by memorising duplicates. The metric is held-out domain PPL on a document-level split."* (2) *"CPT gives you a better base, not an assistant — you still have to SFT it, and it will continue your prompt rather than answer it."* (3) *"The pipeline is the job; the GPU bill is the rounding error. Budget 10–20× the GPU cost for extraction, dedup, eval and the two re-runs."*

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. What does CPT optimise, and how does it differ from SFT at the loss level?**
- **Answer:** CPT optimises `−log p(x_t | x_{<t})` over **every token of raw text**; SFT optimises `−log p(y | x)` over *response* tokens conditioned on an instruction. There is no prompt and no response in CPT, so no masking. The consequence is gradient geometry: SFT moves the model on a low-dimensional manifold of behaviours, CPT moves it across the whole distribution — which is why CPT adapts more deeply and forgets more.
- **Why asked / trap:** It is the definitional question and it separates people who have run both. The trap is "CPT is SFT on raw text" (CS-12 §17.1) — that framing predicts the wrong hyperparameters, the wrong eval, and the wrong forgetting risk.

**Q2. What is `labels = input_ids`, and why not `labels = input_ids[1:]`?**
- **Answer:** The causal LM shifts internally in the loss computation, so the label tensor must align position-for-position with the logits. Passing `input_ids` is correct; pre-shifting by one makes the model predict the token *after* the one it is being scored on, and the model still learns — badly, and with an off-by-one loss that is hard to spot (CS-12 §4.3.2).
- **Why asked / trap:** A one-line change with a silent, permanent effect on quality. The trap is "shifting is harmless because cross-entropy ignores it" — it does not; it trains the wrong target.

**Q3. Base model or `-Instruct` model for CPT, and why?**
- **Answer:** **Base.** An instruct model's chat-template special tokens are trained on as ordinary text, which teaches it that `<|im_start|>` is never followed by a role — degrading chat behaviour while domain PPL improves. The script's own header says it: "WARNING: use a BASE model, not an -Instruct model, for DAPT." If you must use an instruct base, keep the template tokens in the CPT text (CS-12 §10 item 10).
- **Why asked / trap:** The instruction-following loss is invisible in every metric the CPT pipeline reports. The trap is choosing the instruct model "because it is already good."

**Q4. What is τ_cpt, and what is its unit?**
- **Answer:** `τ_cpt = tokens × epochs / params` — tokens spent *per parameter* during the run. It is dimensionless, which is the point: it is the scale-free way to say how far gradient descent moves the weights, and it makes a 1.1B and an 8B run comparable. T2 (τ = 10⁻³) is the minimum viable run; T3 (τ = 10⁻²) is the recommended enterprise target (CS-12 §4.10.3).
- **Why asked / trap:** It converts "how much data do I need?" from a folk number into arithmetic. The trap is quoting a token count without the model size — the same 8M tokens is T3 for an 8B and two full epochs past T5 for a 1.1B.

> **Correction — the minimum-corpus threshold is stated three incompatible ways in CS-12.** CS-12 §4.2.2 rule 6 (line 380) says *"Is your corpus under ~5M tokens? → Do not do CPT"*, and §4.3.5's fix table repeats it (*"Below ~5M tokens the gradient never generalises"*, line 592). CS-12 §8.2 STOP condition 3 (line 2392) says *"Your total corpus is under ~1M tokens"*, and §4.12.4's provenance checklist (line 1453) repeats *"If exclusion drops you below ~1M tokens"*. CS-12 §4.10.3's tier table gives a third figure: **1.1 M tokens for a 1.1B model** at T2, labelled "the minimum viable CPT run". **The tier table's figure is the correct one, and the other two are the wrong *kind* of number.** The scale-free statement is `τ_cpt = 10⁻³`, which for a 1.1B model is 1.1M tokens and for an 8B model is 8M — so neither "~1M" nor "~5M" can be right as an absolute floor for all sizes. "~1M" is a coincidence that happens to be close for a ~1B model and is far too low for an 8B (which needs 8M); "~5M" is wrong in the other direction — it would rule out the 1.1M-token T2 run that the same document calls the minimum viable one. The script's own printed verdicts are a third absolute scheme (`<1M` warn, `1–10M` "modest style/jargon shift", `≥10M` "meaningful"), which is a defensible coarse screen but is *not* size-aware either. **Practical rule:** compute τ, target T3, and use the script's printed verdict as a smell test, never as a gate.

**Q5. What is catastrophic forgetting, mechanically?**
- **Answer:** Gradient descent on the domain distribution moves shared weights away from the configuration that produced general capability. It is not an overwrite of facts; it is a shift of the whole distribution, and the visible symptoms are general-PPL regression, degraded instruction-following if you trained on an instruct model, and loss of low-resource-language and formatting behaviour (CS-12 §4.9.1).
- **Why asked / trap:** "Forgetting" invites the wrong mental model of deleted data. The trap is treating it as an all-or-nothing event rather than a dial you trade against adaptation.

**Q6. What is replay, and what share should you use?**
- **Answer:** Mixing general-domain text back into the CPT corpus so the model keeps seeing the distribution it must not lose. CS-12 §4.9.3 recommends **~5%** as the default, with 1% as the absolute floor (§8.2 STOP condition 2). It is the cheapest, most effective of the four forgetting dials — one line of config.
- **Why asked / trap:** It is the mitigation the source video omits entirely. The trap is treating replay as all-or-nothing: a corpus with 5% replay and a corpus with 0% replay are different projects, and only one of them is safe. Note the script's `--replay-frac` default is **0.10**, double CS-12's recommended 0.05.

**Q7. What is held-out domain perplexity, and why is it the metric?**
- **Answer:** Perplexity of the CPT'd model on domain documents **withheld before any training**, computed on non-pad tokens. It is the metric because training loss falls by construction, is depressed by padding, and is lowered further by memorising duplicates — so a falling loss tells you nothing. Held-out domain PPL, reported with general PPL and a catastrophe probe, is the only panel that distinguishes adaptation from memorisation (CS-12 §12.1, §12.2).
- **Why asked / trap:** A candidate who quotes training loss as evidence has not shipped a CPT run. The trap is computing PPL on a chunk-level split of a duplicated corpus — the number is then real-looking and meaningless.

**Q8. Why split by document rather than by chunk?**
- **Answer:** Because chunks of the same document are near-duplicates. A chunk-level split puts `Introduction… Scope…` in train and its continuation in eval, so held-out PPL measures memorisation of adjacent text. The script splits **documents**, and it protects the held-out set during dedup — it even counts and drops training documents that are near-duplicates of a held-out one, and reports that count, because those are the leaks you cannot see in a loss curve.
- **Why asked / trap:** It is the single most common way a CPT eval is quietly invalid. The trap is "we used sklearn's train_test_split, so it is held out" — the split unit is the whole question.

**Q9. What does `DataCollatorForLanguageModeling(mlm=False)` do that matters here?**
- **Answer:** It pads **labels with `-100`**, which `CrossEntropyLoss` ignores. That is the entire reason to use it: it makes the pad-as-target bug structurally impossible, where hand-rolling `padding="max_length"` plus `labels = input_ids` trains the model to predict the pad token at every pad position (CS-12 §4.3.4, CH-12 §5.2).
- **Why asked / trap:** It is one line and it recovers ~85% of the wasted compute. The trap is setting `padding="max_length"` in the tokeniser "for speed" and never noticing, because the loss goes down *faster*.

**Q10. Why is a two-column PDF extraction silently wrong?**
- **Answer:** `page.get_text("text")` emits text in the PDF's internal drawing order, which for a two-column layout is typically all of column 1 then all of column 2 — or, worse, interleaved line by line, producing sentences that alternate between columns and read as fluent nonsense. The string is non-empty, so nothing raises, and the model trains on it (CS-12 §4.5, §4.5.1).
- **Why asked / trap:** It is the canonical silent data bug of this module. The trap is trusting a non-empty extraction; the fix is `get_text("blocks")` or `"dict"` and sorting blocks by `(round(y0 / tolerance), x0)`, plus reading 20 pages yourself.

**Q11. Where does CPT sit in the pipeline?**
- **Answer:** **First**: `(base) → CPT → SFT → preference tuning → quantise`. CPT produces a *better base*, not an assistant — it continues your prompt rather than answering it. Every production stack is CPT → SFT, and the CPT checkpoint is an internal artifact, not a shippable one (CS-12 §16.2, §17.3).
- **Why asked / trap:** Tests whether they know what they will be holding at the end. The trap is demoing the CPT checkpoint to a stakeholder, who then asks it a question and gets a continuation.

**Q12. Name the four dials that control forgetting, ranked by cost-effectiveness.**
- **Answer:** **Replay ratio** (1–10% of general-domain tokens — cheapest and most effective, one line of config), **learning rate** (1e-5–5e-5 full FT, 1e-4–2e-4 LoRA; halving it is the second lever), **trainable parameter count** (LoRA freezes the base and structurally protects general capability — a bigger lever than LR but a change of method), **duration** (epochs and corpus size — least precise, but free to shorten). Order: replay → LR → freeze more → shorten (CS-12 §4.9.2, §4.9.3).
- **Why asked / trap:** It is the tuning order under a fixed quality bar, and most teams reach for the LR first. The trap is listing them without the ordering — the ordering *is* the answer.

**Q13. What is a catastrophe probe, and what does it measure?**
- **Answer:** A small fixed battery of general-capability checks — short factual prompts, instruction-following, and a few held-out general-text perplexities — run before and after the run, with a **≤2 point** tolerance for shipping (CS-12 §12.5). It exists because held-out domain PPL and general PPL can both look acceptable while a specific behaviour has collapsed.
- **Why asked / trap:** Domain PPL and general PPL are averages. The trap is shipping on the two PPLs alone and discovering the model no longer follows a simple instruction.

**Q14. Why is a drop in MMLU acceptable while a drop in general PPL is not?**
- **Answer:** MMLU is a 4-option multiple-choice format: coarse, high-variance, and a weak proxy for capability, so a few points is inside what CPT is expected to trade for domain fluency. General PPL on a corpus you did not train on (WikiText-103, C4) is the finer instrument: **≤15% regression** and a catastrophe probe of **≤2 points** is shippable; a **30%** general-PPL regression means you have over-forgotten and must raise the replay ratio (CS-12 §12.2, §12.5, §17.18).
- **Why asked / trap:** It tests whether the candidate has a *threshold*, not just a metric list. The trap is treating any MMLU drop as a failure and aborting a run that was working.

---

## Level 2 — Applied & Implementation

**Q15. State the source-argument grammar of `code/03_continued_pretraining.py` and what breaks if you ignore it.**
- **Answer:** The three sources are a **required, mutually exclusive group**: `--pdf-dir`, `--text-file`, `--jsonl-text-field`. Passing none is an argparse error; passing two is an argparse error. They exist because the ingestion path differs: PDFs go through extraction, cleaning, header stripping and reference stripping; a `.txt` file goes straight to cleaning; JSONL reads one field per row.
- **Why asked / trap:** It is the first thing that breaks for a new user. The trap is assuming `--text-file` behaves like `--pdf-dir` with a single input — it skips the extraction stage entirely, so a two-column PDF passed as `.txt` never gets the layout-aware path.

**Q16. You run `--text-file corpus.txt` without `--doc-separator`. What happens, and why does it matter?**
- **Answer:** The whole file becomes **one document**. The `--doc-separator` help says it exactly: "Without it the whole file is ONE document, which disables the held-out split and dedup." One document means the document-level split has nothing to hold out and MinHash dedup has nothing to compare, so both silently become no-ops and the run reports a meaningless held-out number.
- **Why asked / trap:** A flag that changes the *validity* of the experiment rather than the speed of the run. The trap is reading the missing flag as a formatting preference.

**Q17. `--eval-frac` defaults to 0.05. What does it prevent, and what does it split on?**
- **Answer:** It holds out **5% of documents** — the help says "Fraction of DOCUMENTS held out. Without a held-out set you cannot distinguish adaptation from memorisation." It is the split that makes the final held-out perplexity number mean anything, and it is protected during dedup (training documents that are near-duplicates of a held-out document are dropped and counted).
- **Why asked / trap:** It is the one flag whose default you must not override to 0 for speed. The trap is `--eval-frac 0` "for a quick first run" and then reading the training loss as a result.

**Q18. You pass `--replay-frac 0.10` but no `--replay-file`. What does the script print?**
- **Answer:** It prints that the requested replay is **OFF** and — verbatim — that the script *"used to print 'mixing 10% general text' here while mixing nothing at all — the most dangerous kind of bug, because the mitigation is announced and absent."* Replay only happens when `--replay-file` is supplied. With a replay file, `mix_replay()` actually interleaves general-domain chunks and the script reports how many were mixed in and warns if distinct replay chunks ran out and had to be repeated.
- **Why asked / trap:** It is the difference between an announced mitigation and an applied one. The trap is believing the flag. The deeper lesson: a mitigation you cannot observe in the output is a mitigation you do not have.

**Q19. `--overlap-words` defaults to 0. What does the script guard against?**
- **Answer:** It warns when overlap reaches 5% of `--chunk-words`: `--overlap-words 52` against `--chunk-words 1024` is `52/1024 = 5.1%` and trips the warning. The help says 0 for CPT and to keep any non-zero value under 5%. Overlap exists for retrieval-augmented *chunking*, not for CPT — duplicated tokens are re-weighted training signal.
- **Why asked / trap:** A flag imported from the RAG world where it is correct and here where it is harmful. The trap is setting overlap "to preserve context across boundaries" and buying duplicate training signal instead of context.

**Q20. When would you pass `--no-dedup`?**
- **Answer:** Almost never — the help says "Almost never what you want." The only defensible case is a corpus you have already deduplicated more aggressively offline with template-level dedup, where the in-script MinHash pass is redundant cost. Even then, the in-script pass is cheap and it is what populates the dedup counters you report.
- **Why asked / trap:** Tests whether they understand that dedup at CPT scale is part of the method, not hygiene. The trap is `--no-dedup` to make a first run finish faster, on a corpus whose 55% duplication is exactly the problem.

**Q21. Why is `--lr` default `2e-5`, and what does the help say?**
- **Answer:** `2e-5`, with the help text "10-100x lower than pretraining-from-scratch. Do not raise casually." CPT is a small perturbation of a converged model on a comparatively small corpus: a pretraining LR destroys general capability in a few hundred steps, and the damage is not recoverable by more training.
- **Why asked / trap:** The most expensive dial to get wrong and the least reversible. The trap is raising the LR because the loss is not falling fast enough — the loss falling faster is the *bad* outcome here.

**Q22. `--epochs` defaults to `1.0`. Why not 3?**
- **Answer:** Because the corpus is small enough to memorise, and memorisation is the pathological outcome. The empirical signature is that held-out domain PPL bottoms out after roughly one pass and then *rises* while training loss keeps falling — a divergence you can only see if you have a held-out split. "If your corpus gives you ≥ T3 volume in a single pass, train for one pass and stop" (CS-12 §4.10.3).
- **Why asked / trap:** The default encodes a hard-won rule. The trap is treating epochs as free compute on a corpus that is already past T3 once through.

**Q23. `--chunk-words` is 1024. How does the script turn that into a token count?**
- **Answer:** `est_tokens = sum(len(c.split()) for c in chunks) * 4 // 3` — words × 4/3, the standard ~4-characters-per-token heuristic. So 600,000 words becomes `600000 × 4 // 3 = 800,000` estimated tokens. That number is then compared against the `1M` / `10M` verdict thresholds and printed.
- **Why asked / trap:** It is where the volume verdict comes from, and it is an estimate, not a measurement. The trap is treating it as exact in the 1–3M band where the verdict flips — tokenise a sample properly if the decision is close.

**Q24. What verdicts does the script print at 800K, 5M and 20M estimated tokens?**
- **Answer:** Under `1_000_000`: "⚠ Under ~1M tokens, DAPT will change your model very little. Strongly consider RAG or SFT-only instead." Under `10_000_000`: "ℹ 1-10M tokens: expect a modest style/jargon shift, not new knowledge." At or above: "✅ 10M+ tokens: enough for a meaningful domain adaptation." These are absolute, size-agnostic thresholds and are a **smell test, not a gate** — see the Correction under Q4.
- **Why asked / trap:** Quote the thresholds and then immediately say what they are not: they do not know your model size, so they cannot tell you τ. The trap is treating "✅" as a green light for a 70B.

**Q25. Price the T3 run for an 8B model and show the arithmetic.**
- **Answer:** T3 is `τ = 10⁻²`, so `T = 10⁻² × 8e9 = 80M` tokens. At `6·P·T`: `6 × 8e9 × 80e6 = 3.84e18` FLOPs; at a realistic A100-80 bf16 rate of `312e12 × 0.35 = 1.092e14` FLOP/s that is `3.84e18 / 1.092e14 = 35,164 s = 9.77 h`, or **$24.42** at $2.50/h. CS-12's table prints **1.8 h / $4.50**, which is `3·P·T` at 300 TFLOP/s — see the Correction below.
- **Why asked / trap:** The arithmetic is the answer to "is this project worth scheduling", and the honest reading is that the GPU bill is trivial next to the pipeline. The trap is quoting the table's $4.50 without noticing it disagrees with the table's own formula.

> **Correction — CS-12 §4.11.3 states `6·P·T` and then computes `3·P·T`.** The prose at line 1348 says *"$\text{FLOPs} \approx 6 \cdot P \cdot T$ where the factor 6 = 2 for the forward pass + 4 for the backward"* — that is the correct full-fine-tuning figure. Every row of the table beneath it is then computed at **3·P·T**: for T3/8B, `3 × 8e9 × 80e6 = 1.92e18`, which is the printed `1.9e18`, and `1.92e18 / 300e12 = 1.78 h`, which is the printed `1.8 h`. Using the stated 6·P·T, the same cell would be `6 × 8e9 × 80e6 = 3.84e18` and `3.84e18 / 300e12 = 3.56 h` — exactly double. The one row that *does* use 6·P·T is the notebook-demo row: `6 × 1.1e9 × 1500 = 9.9e12 ≈ 1.0e13`, the printed value, whereas 3·P·T would give `4.95e12`. **The 3·P·T rows are the ones to trust for the runtimes as printed, because they are internally consistent with the hours column; the 6·P·T figure in the prose is the correct one for a full fine-tune and the two cannot both be right for the same rows.** The prose does offer a derivation for 3·P·T — it is the LoRA approximation `3PT + 3P_adapter·T ≈ 3PT` — but that derivation applies to LoRA runs, while the table's rows are not labelled with a method. **Secondly, the same header labels 300 TFLOP/s as "~40% MFU", which is incoherent:** 40% of an A100-80's 312 dense bf16 TFLOP/s is `0.40 × 312 = 124.8`, not 300; 300 TFLOP/s is ~96% MFU. If you keep the 40% figure, the T3/8B cell becomes `3.84e18 / (312e12 × 0.40) = 8.55 h`, and `1.92e18 / 124.8e12 = 4.27 h` — **neither matches the printed 1.8 h.** The printed hours are consistent only with `3·P·T` at an assumed 300 TFLOP/s. **Use one convention and label it:** for a full fine-tune at T3/8B, budget **3.84e18 FLOPs**, and at a defensible 35% MFU on an A100-80 that is **9.8 h ≈ $24**, not $4.50.

**Q26. Why are `--chunk-words` and `--seq-len` both present?**
- **Answer:** Chunking happens on **text** (documents split into ~1024-word passages before tokenisation); `--seq-len` is the **token** length the collator pads and truncates to (2048). A 1024-word chunk is roughly 1365 tokens, which fits inside 2048 — so the two are consistent by default, and a chunk larger than the sequence length would be silently truncated at training time.
- **Why asked / trap:** Two length knobs that interact, and one of them wins. The trap is raising `--chunk-words` to 4096 "for more context" while `--seq-len` stays at 2048 — you have just deleted the second half of every chunk.

**Q27. The script prints an "IMPORTANT" message after the run. What does it say, and why does it exist?**
- **Answer:** It says what you have now is a **BASE model, not an assistant** — the same warning as the header docstring. It exists because the natural next action after a successful CPT run is to chat with the checkpoint, and a base model continues your prompt instead of answering it; without the warning that behaviour reads as a broken run.
- **Why asked / trap:** Tests whether the candidate knows what CPT produces. The trap is shipping or demoing the CPT checkpoint; SFT (IQ-13) is the next required stage.

**Q28. What is `group_by_length=True` doing in the training arguments, and why does it matter more at CPT than at SFT?**
- **Answer:** It batches sequences of similar token length together, cutting pad tokens. At CPT it matters more because sequences are raw chunks of wildly varying length (a title card versus a full appendix), so ungrouped batching wastes a large fraction of every step on padding — and padding is the exact waste the collator's `-100` masking already prevents from *corrupting* the loss.
- **Why asked / trap:** It is the efficiency counterpart to Q9: the collator stops padding from lying, grouping stops it from being paid for. The trap is treating it as a speed flag only — it also reduces gradient noise from pad-heavy steps.

---

## Level 3 — Advanced, Internals & Theory

**Q29. Why does CPT need no labels and yet still apply a loss to every token?**
- **Answer:** The labels are the inputs. Self-supervision means `x_{t+1}` is the target for position `t`, so nothing has to be annotated — which is why CPT is the only stage you can run on a corpus of PDFs you already own. Every token contributes, which is exactly why duplicates and padding matter so much: both are *re-weighted regions of the objective*, not cosmetic issues.
- **Why asked / trap:** It is the mechanism behind both of CPT's advantages (no labels) and its hazards (duplicates, padding). The trap is thinking of the corpus as a bag of documents rather than as a token distribution being fitted.

**Q30. Why do near-duplicates hurt *more* at CPT than at SFT?**
- **Answer:** Because the loss is applied to every token of raw text, whereas SFT applies its loss only to response tokens of a curated example. A near-duplicate pair in SFT is two similar examples in a set of thousands; in CPT it is a re-weighted region of the token distribution the model is fitting, seen repeatedly across epochs — which is precisely the mechanism of memorisation and distribution distortion. Dedup is also cheaper than the alternative, because more epochs on duplicates multiplies the same tokens instead of adding new ones (CS-12 §4.7.2).
- **Why asked / trap:** Tests whether they understand CPT as distribution fitting. The trap is "we deduplicate for hygiene"; here dedup is a hyperparameter with the same status as the learning rate.

**Q31. MinHash/LSH in this script: what do 64, 5 and 0.8 mean?**
- **Answer:** `minhash_signature(text, num_hashes=64, shingle=5)` builds a 64-element signature over 5-word shingles, LSH-buckets the signatures, and `dedup_documents(docs, threshold=0.8)` removes documents whose estimated Jaccard similarity exceeds 0.8. So: 5-word shingles define what "similar" means at the phrase level, 64 hashes bound the estimation error, and 0.8 is the decision threshold.
- **Why asked / trap:** Three parameters, three different jobs, and each one is a defensible interview follow-up. The trap is lowering the threshold "to be safe" — at 0.6 you begin deleting legitimately distinct documents that share boilerplate, which is the §15.4 template problem.

**Q32. Why is a perplexity-scored quality filter controversial for a domain corpus?**
- **Answer:** Because a filter trained on Wikipedia scores *fluency*, not *value*, and it deletes exactly the documents that are most unlike Wikipedia — the clinical notes, the tables, the terse regulatory clauses you collected the corpus for — while keeping fluent boilerplate. A corpus can be perfectly deduplicated and still 100% boilerplate (CS-12 §4.6.3, §17.2 item 8).
- **Why asked / trap:** It is the filter that improves every general-purpose benchmark while destroying a domain project. The trap is adopting a published quality classifier without checking its discard distribution against your own content.

**Q33. Why does `target_modules` decide whether CPT can teach *facts*?**
- **Answer:** The knowledge-bearing capacity is concentrated in the **MLP** blocks — `gate_proj`, `up_proj`, `down_proj` act as key-value memories — and in the embedding matrix, which defines what each token *is*. A q/v-only LoRA can only change *where the model looks*, i.e. routing and style; it has no path to represent a new term or a new association. The correction is `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`, `r=32`, `lora_alpha=64`, `use_rslora=True` (CS-12 §4.3.3, §7.5, §7.6).
- **Why asked / trap:** It is the default everyone copies from a SFT tutorial, where q/v is often enough. The trap is expecting vocabulary acquisition from an attention-only adapter.

**Q34. "A lower learning rate is always safer." Push back.**
- **Answer:** A lower LR reduces forgetting **and** reduces adaptation. At some point the run does nothing but burn GPU-hours, and the symptom — held-out PPL flat — is indistinguishable from "the corpus has no signal". Set the LR from the model-size table (1e-5–5e-5 full FT, 1e-4–2e-4 LoRA) and then tune the **replay ratio**, which has the better forgetting/adaptation trade (CS-12 §17.3 item 10).
- **Why asked / trap:** A false-safety belief that wastes weeks. The trap is halving the LR a third time in response to a flat PPL curve instead of measuring whether the corpus has signal at all.

**Q35. Training loss keeps falling while held-out domain PPL turns upward. Explain.**
- **Answer:** The model has started memorising. Training loss falls because the corpus is being fitted, including its duplicates and its padding; held-out PPL rises because the model is moving away from the distribution that generalises. This divergence is the only reliable overfitting signal and it requires a document-level held-out split to see — which is why §12 calls the split a stop condition rather than a nicety.
- **Why asked / trap:** It is the signature of the classic bad CPT run and the reason to evaluate per checkpoint. The trap is stopping at the last checkpoint instead of at the PPL minimum (CS-12 §12.6).

**Q36. Why does CPT typically *lower* MMLU?**
- **Answer:** Because MMLU is a multiple-choice format dominated by general-world knowledge and a specific answer shape; CPT moves the model's distribution toward your domain, which is a trade against general breadth and against the calibration of the MMLU answer format. A drop is expected; what must not happen is a large regression in general *perplexity* or a catastrophe-probe failure.
- **Why asked / trap:** It prevents a team from aborting a working run. The trap is an acceptance criterion written against a public leaderboard — CS-12 §8.2 STOP condition 10 says plainly that if that is your criterion, CPT is the wrong move.

**Q37. Why is "one epoch, always" the right default for CPT when pretraining tolerates many?**
- **Answer:** Because pretraining corpora are enormous relative to model capacity and each epoch is a different shuffle of a distribution the model cannot memorise. A CPT corpus is small enough to memorise — a 1.1B model has ample capacity for 11M tokens — so additional epochs add memorisation rather than information. One clean pass at T3 beats five passes at T2 with a memorised model at the end (CS-12 §4.10.3).
- **Why asked / trap:** It is the difference between the pretraining and CPT regimes, stated as a capacity argument. The trap is reasoning by analogy from pretraining, where the same rule does not hold.

**Q38. Quantify the pad-as-target bug.**
- **Answer:** With `padding="max_length", max_length=512` and unmasked labels on a corpus of ~75-token chunks, about `512 − 75 = 437` of 512 positions are pad — `437/512 = 85.4%` of the loss. Those positions are trivially predictable (a constant sequence), so they drive the loss down quickly and its *movements* mostly reflect padding dynamics, not learning. The real signal is the remaining ~15% (CS-12 §4.3.4).
- **Why asked / trap:** It is the bug that makes a broken run look like the best run you have ever had. The trap is diagnosing it from the loss curve — you cannot; you have to inspect the label tensor for `-100`.

**Q39. Why can CPT not teach the model a new *tokeniser*?**
- **Answer:** Token identity is fixed at pretraining: the embedding matrix maps a fixed vocabulary to vectors, and continued training reshapes those vectors but cannot invent a subword. New domain terms therefore tokenise into existing subwords — which is exactly why "domain terms tokenise into stable concepts" is listed as a T2 effect and not a T0 one, and why `resize_token_embeddings()` after the fact creates a mismatch with any already-quantised checkpoint (CS-12 §4.3.1, CS-11 §10 item 3).
- **Why asked / trap:** It bounds what CPT can deliver and explains the τ tiers. The trap is promising a stakeholder that the model will "learn our terminology" as a vocabulary change rather than a representation change.

**Q40. Why is CPT not a substitute for RAG?**
- **Answer:** CPT cannot cite, cannot be updated without a re-run, and cannot guarantee recall of a specific fact — it changes the *distribution* the model writes from. RAG does all three and costs a fraction as much. The framing is complementary: CPT makes the model fluent enough that retrieved text is used well; RAG makes it correct. CPT-only fails as confident hallucination with no citations (CS-12 §4.2.1, §9.3, §17.1 item 4).
- **Why asked / trap:** It is the decision the module exists to force. The trap is answering "CPT" to a question that is really "the model does not know this fact."

---

## Level 4 — System Design & Scenario

> These are 15-minute whiteboard questions. Answer in the fixed shape: **requirements → constraints → design → trade-offs → failure modes**.

**Q41. 3M tokens of proprietary text, a 7B model, a 3-week deadline. Walk the decision.**
- **Answer:** Order: **(1) prompting** with 8 few-shot examples — hours, $0, and it establishes the baseline you will be compared against. **(2) RAG** over the 3M tokens — days, near-zero training cost, and it solves factuality, citation and currency in one move; if the user's question is "what does the document say", stop here. **(3) SFT** on 1,000–3,000 pairs — days, ~$20 of GPU, fixes format and behaviour. **(4) CPT** only if 1–3 leave a *fluency and vocabulary* gap nothing else closes — which at 3M tokens on a 7B is `τ = 3e6 / 7e9 = 4.3e-4`, i.e. **T2, on the low end**, with a real risk of under-fitting. With three weeks, ship 1–3 and run a **0.5B CPT probe in parallel** to decide whether a full run is worth scheduling afterwards (CS-12 §8.3, §8.4, §13.2).
- **Why asked / trap:** The order is the answer, and the cheap stages must come first. The trap is jumping to CPT because it is the most interesting work — and burning the deadline on the stage with the poorest cost/benefit at this corpus size.

**Q42. What does the eval harness contain, and when is it built?**
- **Answer:** Built **before** the run, and it contains: a document-level held-out domain set (≥5%, protected during dedup), general-domain held-out PPL on a corpus you did not train on, a catastrophe probe with a ≤2-point tolerance, a task metric, and per-checkpoint logging so you can stop at the PPL minimum rather than at the last step. Six non-negotiables govern the protocol (CS-12 §12.2–§12.6).
- **Why asked / trap:** "We will build the eval after the run" is the failure this section exists to prevent (§8.2 STOP condition 1). The trap is treating the harness as a reporting task rather than the instrument that decides whether the run worked.

**Q43. Compliance asks what you trained on. What is the deliverable?**
- **Answer:** A **provenance ledger** built *before* stage 1: for every source, who owns it, how you obtained it, and what the contract says about training; a fail-closed rule on unknown provenance; the token share of any documents you must exclude; a check for machine-readable TDM reservations; and **a named human's signature**. If exclusion drops you below the viability threshold, the project may not be viable at all (CS-12 §4.12.2, §4.12.4).
- **Why asked / trap:** It is a gate that sits before the pipeline, not a paperwork step after it. The trap is "the team decided" as a sign-off — CS-12 says explicitly that is not one.

**Q44. 15,000 filings yield 400M tokens and MinHash removes 55% of documents. Now what?**
- **Answer:** The removal rate tells you the corpus is **mostly repetition, not 400M tokens of information** — the unique content is closer to 180M, so your τ_cpt was a fiction for a corpus that is heavily templated (filings share a skeleton: Risk Factors, MD&A, financial statements). Do three things: compute the **unique**-token count before choosing a volume tier; add **template/n-gram-profile dedup** on top of MinHash; and re-scope to the *novel* regions — the diff between consecutive filings and the standards they cite — a move that in the analogous case took 400M down to 60M and improved the task metric by 12 points (CS-12 §15.4, §13.5).
- **Why asked / trap:** A high dedup rate is a signal about the corpus, not a successful cleaning step. The trap is reporting "we removed 55% of duplicates" as an achievement and then training on the boilerplate that survived.

**Q45. Design the ingestion pipeline for 12,000 two-column PDFs.**
- **Answer:** Per file: layout-aware extraction (`get_text("blocks")`/`"dict"`, sorted by `(round(y0/tol), x0)` so both columns reconstruct in reading order) → `strip_running_headers(pages)` **before** cleaning, because header removal needs page position → `clean_text` → `strip_references` → `scrub_pii` → `sanity_check_extraction`, which is the gate that catches an extractor that silently returned garbage. Then corpus-level: document split, MinHash dedup with the held-out set protected, chunking, and a volume verdict. Plus a human read of 20 sample pages.
- **Why asked / trap:** The ordering is load-bearing (headers before cleaning; sanity-check per file, not at the end) and the human sample is the only check that catches fluent nonsense. The trap is a pipeline that reports only total character counts — which a two-column extraction failure passes perfectly.

**Q46. What must you be able to hold simultaneously at serving time?**
- **Answer:** Both the pre-CPT base **and** the new model. Rollback requires serving both, and if you can only host one you have no rollback — CS-12 §8.2 STOP condition 7 makes this a precondition, not a post-launch concern. The artifact set that goes with it is the versioned CPT checkpoint, its SFT successor, the corpus hash, the calibration-free but config-complete training record, and the held-out eval numbers (CS-12 §16.3, §16.4).
- **Why asked / trap:** It reframes rollback as a capacity decision made *before* the run. The trap is discovering at incident time that the previous model was overwritten in place.

---

## Level 5 — Debugging & Incident Response

> Answer in a fixed shape: **what I look at first, what each observation rules in or out, and the fix.**

**Q47. "The training loss went from 2.9 to 1.4, so the domain adaptation worked." Name three reasons that is invalid.**
- **Answer:** (i) **Training loss always falls by construction** — gradient descent on an objective reduces that objective; it is arithmetic, not evidence. (ii) **Padding inflates the improvement**: with `padding="max_length"` and unmasked labels, ~85% of the loss at 512 tokens for ~75 real tokens is on pad positions, which are trivially predictable, so the reported number is pulled down and its movements are dominated by padding dynamics. (iii) **Duplicates and memorisation**: a corpus with near-duplicates re-weights those regions, and at 2–3 epochs the model fits them, lowering loss while generalisation falls. Ask instead for **held-out domain perplexity on non-pad tokens over a document-level split of deduplicated data**, with general PPL and a catastrophe probe alongside (CS-12 §12.1, §12.2, §4.3.4).
- **Why asked / trap:** The most common invalid inference in this module, and the answer must name the *mechanism* of each failure. The trap is offering only "you need a validation set" — that is one of the three.

**Q48. Held-out domain PPL improved 40% and MMLU dropped 4 points. Success?**
- **Answer:** Probably yes, and **not yet decided**. A 40% domain-PPL improvement is a real result; a 4-point MMLU drop is within what CPT is expected to trade. The third number you need is **held-out general PPL on a corpus you did not train on** (WikiText-103 or C4), because MMLU is a coarse, high-variance proxy. If general PPL regressed **≤15%** and the catastrophe probe dropped **≤2 points**, ship it; if general PPL regressed **30%**, you have over-forgotten and must raise the replay ratio (CS-12 §12.2, §12.5, §17.18).
- **Why asked / trap:** It tests whether the candidate has thresholds rather than opinions. The trap is approving on the domain number alone, or rejecting on the MMLU number alone.

**Q49. The CPT checkpoint continues your prompt instead of answering it.**
- **Answer:** Nothing is broken — that is what a base model does, and the script says so in its post-run "IMPORTANT" message and its header docstring. It becomes a defect only if you also see degraded instruction-following *after* SFT, which would point at having run CPT on an `-Instruct` base and trained the chat-template tokens as ordinary text. Fix: use the base checkpoint for CPT and run SFT (IQ-13) on top.
- **Why asked / trap:** A "bug" that is a category error about which artifact you are holding. The trap is re-running CPT to fix it, which makes it worse.

**Q50. Held-out domain PPL falls for one epoch, then rises. What do you do?**
- **Answer:** Stop at the minimum, not at the end. That curve is the memorisation signature, and it means the corpus is at or past T4 for this model or the epoch count is too high. Concretely: restore the checkpoint at the PPL minimum, drop to a single pass, raise replay toward the top of the 1–10% band, and only then consider more unique data. Per-checkpoint evaluation exists precisely so this is a choice rather than a discovery (CS-12 §12.6, §4.10.3).
- **Why asked / trap:** It is the one curve that requires action *during* the run. The trap is letting the run finish because it was already launched, and then having nothing but the final checkpoint.

**Q51. Extraction returns fluent text that is nonsense. Where do you look first?**
- **Answer:** The layout. Read three pages of the source alongside the extracted output: if sentences alternate between columns, or if running headers appear mid-paragraph, it is `get_text("text")` order rather than content. Then check the order of operations — `strip_running_headers` needs page position and must run **before** `clean_text`, or the positional signal is gone. Fix with blocks/dict extraction sorted by `(round(y0/tol), x0)`. The `sanity_check_extraction` warnings the script prints per file are what you should have been reading (CS-12 §4.5, §4.5.1).
- **Why asked / trap:** The failure is invisible in every aggregate statistic, so the first move must be reading raw output. The trap is deduplicating and chunking garbage harder, which produces clean, well-chunked garbage.

**Q52. Full fine-tuning a 7B OOMs on one 80 GB card. What are the options, in order?**
- **Answer:** Recompute the budget: full FT at bf16 needs `params × 2 B` weights + `× 2 B` gradients + AdamW `× 8 B` states ≈ **12 bytes/param**, so `7e9 × 12 = 84 GB` before activations — it cannot fit, and no batch-size tuning fixes it. Options in order: (1) **QLoRA**, which is the default for a first CPT run — 4-bit frozen base plus adapters, and CS-12 §13.3's rule is "LoRA first, always"; (2) full FT with FSDP/DeepSpeed across ≥2 GPUs; (3) a smaller base. Full FT is only justified after a LoRA probe shows real ΔPPL and tolerable forgetting (CS-12 §11.1, §11.2, §13.3, §17.3 item 11).
- **Why asked / trap:** The arithmetic answers it in one line, and the candidate should do the arithmetic rather than suggest `--batch 1`. The trap is reaching for gradient checkpointing as the primary fix — it helps activations, not the 84 GB of optimizer state.

**Q53. A 3M-token corpus produces a loss curve that barely moves.**
- **Answer:** At 3M tokens on a 7B, `τ = 3e6 / 7e9 = 4.3e-4` — T2 at the low end and plausibly below the adaptation floor, so a flat curve is the expected result rather than a bug. Before concluding, rule out the cheaper explanations: is the LR at the low end of the range, is the loss actually being computed on `-100`-masked labels, and did the corpus survive cleaning (compare characters in to characters out). Then make the decision the tier table implies: more unique data, a smaller base, or a different intervention — the script's own verdict at this size is "consider RAG or SFT-only instead."
- **Why asked / trap:** A flat curve has three causes and only one of them is "the corpus has no signal." The trap is raising the LR to force movement, which trades a null result for a forgetting incident.

**Q54. The same CPT config six months later produces a different model.**
- **Answer:** Same family of causes as quantization reproducibility: reduction order, kernel versions, library versions and GPU architecture all enter the numerics, and the *corpus* may simply have changed — a re-crawled or re-extracted corpus is a different corpus. The procedural fix is to record the corpus hash, the extraction/cleaning code revision and the full training config with every artifact, and to treat a changed hash as a new experiment requiring its own held-out numbers, not as a rebuild.
- **Why asked / trap:** It is the provenance question at training time rather than at quantisation time. The trap is blaming the library version only — the corpus digest is the field most often missing and the one most likely to have changed.

---

## Rapid Fire — True / False / One-Liner

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | CPT is SFT on raw text. | **False** | CPT's loss is over every token of raw text; SFT's is over response tokens. |
| 2 | CPT needs labelled data. | **False** | The labels are the inputs — that is the whole point. |
| 3 | Use an `-Instruct` model for CPT. | **False** | Template tokens get trained as text and instruction-following degrades. |
| 4 | Training loss is the success metric. | **False** | It falls by construction, is inflated by padding, and is lowered by memorisation. |
| 5 | CPT replaces SFT. | **False** | CPT produces a better *base*; it continues rather than answers. |
| 6 | CPT replaces RAG. | **False** | CPT cannot cite, cannot update, and cannot guarantee recall. |
| 7 | Dedup matters less for CPT than for SFT. | **False** | Inverted — CPT applies loss to every token and re-sees documents. |
| 8 | More epochs is free quality. | **False** | One pass at T3 beats five at T2 with a memorised model. |
| 9 | A lower learning rate is always safer. | **False** | It reduces adaptation as well as forgetting; flat PPL looks like no signal. |
| 10 | `padding="max_length"` + `labels = input_ids` is fine. | **False** | ~85% of the loss lands on pad positions; use the `mlm=False` collator. |
| 11 | `page.get_text("text")` is sufficient for a 2-column PDF. | **False** | It emits drawing order — fluent nonsense, no error. |
| 12 | Held-out PPL may be computed on chunks of trained documents. | **False** | Chunks of one document are near-duplicates; split by document. |

---

## Coding / Whiteboard Tasks

### Task 1 — Spot the four bugs in this CPT tokenisation function

```python
def tokenize_fn(examples):
    out = tok(examples["text"], padding="max_length", max_length=512)
    out["labels"] = out["input_ids"].copy()          # (a)
    return out

tok = AutoTokenizer.from_pretrained(model_id)        # (b)
tok.pad_token = tok.eos_token                        # (c)
trainer = Trainer(model=model, train_dataset=ds,
                  data_collator=DataCollatorForLanguageModeling(tok, mlm=False))  # (d)
```

- **Expected solution sketch:** (a) the labels are unshifted and **unmasked** — pad positions are trained as targets; with ~75-token chunks ~85% of the loss is padding. (b) the tokeniser loads with no `pad_token` for many base models, which is why (c) exists. (c) is correct-but-dangerous: `pad_token = eos_token` means the padding *id* is the EOS id, so any un-masked pad teaches the model to emit EOS in the middle of a document; it is safe *only* because (d) masks pads to `-100`. (d) is right, and it is the reason (c) does not detonate — but (d) also **overrides** the hand-built labels with its own `-100`-padded copy, which means (a) is both wrong and inert, and the author now believes a masking step exists that they did not write.
- **The decisions being graded:** naming the pad-as-target mechanism and quantifying it (~85%); knowing that the `mlm=False` collator is what saves `pad_token = eos_token`; noticing that the collator silently repairs (a) so the *bug is invisible in the output* and will reappear the moment someone swaps the collator.
- **Grading:** the fix is `padding=False` in `tokenize_fn`, mask defensively with `-100`, and keep the collator — plus `group_by_length=True`.
- **Fail condition:** "the labels should be `input_ids[1:]`" (wrong — the causal LM shifts internally), or not noticing that `pad_token = eos_token` is only safe because of the collator.

### Task 2 — Decide whether to run CPT, and price it

A team has 5M unique tokens of clinical notes, wants to use a 7B base, and has one A100-80 for a weekend (48 h). Compute τ, name the tier, price the run at `6·P·T` with 35% MFU, and make the call.

**Expected solution sketch**

```python
P, T = 7e9, 5e6
tau = T / P                                    # 1 epoch
print(f"tau = {tau:.2e}")                      # 7.14e-04 -> T2 ("minimum viable")
flops_full = 6 * P * T                         # 2.1e17
hours_full = flops_full / (312e12 * 0.35) / 3600
print(f"{flops_full:.2e} FLOPs -> {hours_full:.2f} h -> ${hours_full*2.5:.2f}")
```

- **The decisions being graded:** `τ = 5e6 / 7e9 = 7.1e-4` is **T2** — above the 10⁻⁴ register-nudge tier and below T3 (10⁻²), i.e. "style + vocabulary, facts still unreliable"; the run costs `6 × 7e9 × 5e6 = 2.1e17` FLOPs = **0.53 h ≈ $1.34**, so the GPU weekend is wildly oversized for the training and the *real* cost is extraction, dedup and eval; and the recommendation follows from the tier, not the budget — **the corpus is thin for a 7B**, so the defensible plan is RAG or SFT for the fact-shaped need, CPT only if the gap is register, and in that case a **smaller base** (1–3B) moves τ up a tier at the same token count.
- **Grading:** a candidate who computes the cost and then says "so we have plenty of GPU, let us use it" has failed the question; the constraint that binds is corpus volume, not compute.
- **Fail condition:** confusing tokens with `τ`; using the model's total parameter count when only the trainable ones matter for a LoRA variant without saying so; or recommending 3 epochs to "use the weekend" — `3 × 7.1e-4 = 2.1e-3` is still T2 and is now three passes over a memoisable corpus.

---

## Cheat Sheet of Numbers To Memorize

| Quantity | Value | Why it matters |
|---|---|---|
| CPT loss | `−log p(x_t \| x_{<t})` over every token | No prompt, no response, no masking |
| Labels | `labels = input_ids` | The causal LM shifts internally; do not pre-shift |
| τ_cpt | `tokens × epochs / params` | The scale-free volume unit |
| T2 (minimum viable) | `τ = 10⁻³` → 1.1M @1.1B, 8M @8B | "Style + vocabulary" |
| T3 (recommended) | `τ = 10⁻²` → 11M @1.1B, 80M @8B | "Vocabulary + facts"; the enterprise target |
| Replay share | **~5%** recommended, **1%** absolute floor | The cheapest forgetting dial |
| Forgetting dials, ranked | replay → LR → trainable params → duration | The tuning order |
| LR | `2e-5` default; 1e-5–5e-5 full FT, 1e-4–2e-4 LoRA | 10–100× below pretraining |
| Epochs | **1.0**, one pass | PPL bottoms out at ~1 epoch then rises |
| Held-out split | **5% of documents** (`--eval-frac`) | Chunk-level splits are invalid |
| Eval panel | domain PPL + general PPL + catastrophe probe | ≤15% general-PPL regression, ≤2 probe points |
| Catastrophic threshold | general PPL ≤15% regression; 30% = over-forgotten | The ship/abort line |
| Pad-as-target share | `437/512` = **85.4%** of the loss | Why the reported loss is meaningless |
| Token estimate | `words × 4 // 3` | 600,000 words → 800,000 tokens |
| Volume verdicts (script) | `<1M` warn · `1–10M` "modest shift" · `≥10M` "meaningful" | Absolute, size-agnostic — a smell test |
| MinHash defaults | `num_hashes=64`, `shingle=5`, `threshold=0.8` | Three knobs, three jobs |
| Header stripping | `min_frac=0.6` across pages | Positional, must run before cleaning |
| Chunk defaults | `--chunk-words 1024`, `--overlap-words 0`, `--seq-len 2048` | Overlap is a RAG habit, not a CPT one |
| Full-FT VRAM | ≈ **12 bytes/param** bf16 + AdamW | 7B → ~84 GB before activations |
| Compute | `6·P·T` full FT; `≈3·P·T` LoRA | CS-12 §4.11.3's table uses 3·P·T — see the Correction |
| T3/8B honest price | **3.84e18 FLOPs → 9.8 h → ~$24** @35% MFU | Not the table's $4.50 |
| A100-80 bf16 | 312 TFLOP/s dense; 40% MFU = **124.8** | "300 TFLOP/s @40% MFU" is incoherent |
| Demo-run PPL | loss 9.66 → `exp(9.66) = 15,678` | `ln(32000) = 10.37` — worse than uniform over 32k |

---

## Answers To The Self-Check Questions From CS-12

- **Q1 — "loss went 2.9 → 1.4, so it worked."** (i) Training loss always falls by construction. (ii) Padding inflates the improvement: with `padding="max_length"` and unmasked labels, ~85% of the loss at 512 tokens for ~75 real tokens is on trivially predictable pad positions. (iii) Duplicates and memorisation: at 2–3 epochs the model fits them, lowering loss while generalisation falls. Ask for **held-out domain PPL on non-pad tokens over a document-level split of deduplicated data**, with general PPL and a catastrophe probe (CS-12 §12.1, §12.2, §4.3.4). — **trap:** The invalid inference this module exists to kill. The trap is answering only "use a validation set."
- **Q2 — Why dedup matters more in CPT than SFT.** The CPT loss is applied to *every token of raw text*; SFT's only to response tokens of a curated example. A near-duplicate pair in SFT is two similar examples in thousands; in CPT it is a re-weighted region of the token distribution fitted repeatedly across epochs — the mechanism of memorisation and distribution distortion. Dedup is also cheaper than the alternative, since more epochs on duplicates multiplies the same tokens instead of adding new ones (CS-12 §4.7.2, §7.2). — **trap:** Tests whether CPT is understood as distribution fitting. The trap is treating dedup as hygiene.
- **Q3 — 200M tokens, 8B model, 3 epochs.** `τ = 3 × 200e6 / 8e9 = 0.075` — **T4, at the high end**. That is three epochs of the same 200M unique tokens, the classic memorisation mistake, and the risk is held-out PPL falling then rising. Change: **1 epoch on the 200M and stop**, or spend the budget on more unique documents rather than more passes. Add 5% replay, and evaluate at the PPL minimum rather than at the end (CS-12 §4.10.3, §7.2, §12.6). — **trap:** The arithmetic plus the tier plus the decision. The trap is answering "reduce the LR" — the problem is passes, not step size.
- **Q4 — The four dials, ranked.** **Replay ratio** (1–10% general-domain tokens; cheapest, most effective, one line of config) → **learning rate** (1e-5–5e-5 full FT, 1e-4–2e-4 LoRA) → **trainable parameter count** (LoRA freezes the base and structurally protects general capability) → **duration** (least precise, free to shorten) (CS-12 §4.9). — **trap:** The *ordering* is the answer. The trap is a list without it.
- **Q5 — `target_modules=["q_proj","v_proj"]`, `r=8`.** These are *attention* projections: they decide which positions attend to which, so adapting them changes routing and style, not stored associations. Knowledge capacity is concentrated in the **MLP** layers (`gate_proj`, `up_proj`, `down_proj`), which act as key-value memories, and in the embedding matrix, which defines what a token *is*. A q/v LoRA at r=8 has no path to represent a new domain term. Correction: all seven projections, `r=32`, `lora_alpha=64`, `use_rslora=True` (CS-12 §4.3.3, §7.5, §7.6). — **trap:** It is the SFT-tutorial default applied to the wrong stage. The trap is explaining it as "r is too small."
- **Q6 — `padding="max_length", max_length=512` + `labels = input_ids.copy()` with `pad_token == eos_token`.** Every pad position is in `labels`, so the model is trained to predict `pad_token_id` (= `eos_token_id`) there. With a mean chunk of ~75 tokens, ~**437 of 512 positions are padding — about 85% of the loss is on padding**, and because it is a constant sequence it is trivially predictable, so the reported loss is dominated by a signal you do not care about and its movements mostly reflect padding dynamics. Fix: `padding=False`, `DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)` (which pads labels with `-100`), and mask defensively in `tokenize_fn` (CS-12 §4.3.4, §7.8, §6.6). — **trap:** The quantification is the answer. The trap is "it still trains, so it is fine."
- **Q7 — Two-column academic PDF.** `page.get_text("text")` emits text in the PDF's internal drawing order, which for two columns is typically all of column 1 then all of column 2 — or interleaved line by line, giving sentences that alternate between columns and read as fluent nonsense. The failure is **silent**: the string is non-empty and the model trains on it. Fix: layout-aware extraction — `get_text("blocks")` or `"dict"` sorted by `(round(y0 / tolerance), x0)`, or column detection by clustering block x-coordinates — then read 20 pages yourself (CS-12 §4.5, §4.5.1). — **trap:** It is a data bug with no error path. The trap is a fix that improves the average without checking a sample.
- **Q8 — Domain PPL −40%, MMLU −4.** Probably yes, and **not yet decided**. A 40% domain-PPL improvement is real; a 4-point MMLU drop is within the expected trade. The third number is **held-out general PPL on a corpus you did not train on** (WikiText-103 or C4), because MMLU is a 4-option format and a coarse, high-variance proxy. General PPL regressed **≤15%** and probe **≤2 points** → ship; **30%** → over-forgotten, raise replay (CS-12 §12.2, §12.5, §17.18). — **trap:** Thresholds, not opinions. The trap is deciding on the domain number alone.
- **Q9 — 15,000 filings → 400M tokens, MinHash removes 55%.** It tells you the corpus is **mostly repetition, not 400M tokens of information** — unique content is closer to 180M, so τ_cpt was a fiction and the corpus is heavily templated (filings share a skeleton). Do differently: compute the **unique**-token count before choosing a tier; add **template/n-gram-profile dedup**; and re-scope to the *novel* regions — the diff between consecutive filings and the standards they cite — which in the analogous case took 400M to 60M and improved the task metric by 12 points (CS-12 §15.4, §13.5, §10 item 11). — **trap:** A high removal rate is a finding, not an achievement. The trap is reporting it as a win.
- **Q10 — 3M tokens, 7B, 3 weeks.** Order: **(1) prompting** with 8 few-shot examples (hours, $0, and it sets the baseline). **(2) RAG** over the 3M tokens (days, near-zero training cost; solves factuality, citation and currency — if the question is "what does the document say", stop here). **(3) SFT** on 1,000–3,000 pairs (days, ~$20, fixes format and behaviour). **(4) CPT** only if 1–3 leave a *fluency and vocabulary* gap — at 3M tokens on a 7B, `τ = 3e6 / 7e9 = 4.3e-4`, **T2 on the low end**, with a real risk of under-fitting. With three weeks, ship 1–3 and run a **0.5B CPT probe in parallel** to decide whether a full run is worth scheduling (CS-12 §8.3, §8.4, §13.2). — **trap:** The order plus the τ arithmetic plus the probe. The trap is starting at step 4 because it is the interesting one.

---

## Cross-References

| Module | Relationship to IQ-12 |
|---|---|
| `case-studies/CS-12-Domain-Adaptive-Continued-Pretraining.md` | The source of truth. §4.3 is the tokenisation/loss contract, §4.10.3 the tier table, §12 the eval panel, §17 the misconception list this bank's traps come from. |
| `cheat-sheets/CH-12-Domain-Adaptive-Continued-Pretraining.md` | The one-page version: the τ tiers, the corpus-quality rules, the silent-failure list (S1–S12), the nine pre-run checks. |
| `code/03_continued_pretraining.py` | The runnable pipeline; its `--help` is the flag authority for Q15–Q28. Note `--replay-frac` does nothing without `--replay-file`. |
| **IQ-13 — Instruction Fine-Tuning** | The stage that must follow CPT. CPT gives you a better base; IQ-13 turns it into an assistant, and its masking discipline is the mirror image of this module's. |
| **IQ-10 — Quantization I** | Where the bit-width and VRAM arithmetic lives, and where NF4 (the QLoRA base) is defined. |
| **IQ-11 — Quantization II** | What happens to the CPT'd, SFT'd model afterwards: merge, quantise, gate. Both modules end at the same held-out gate. |
| **IQ-16 — Unsloth** | The fast single-GPU path for the SFT step that follows CPT, and where the `target_modules` correction is easiest to apply correctly. |
| `case-studies/CS-13-Instruction-Fine-Tuning.md` | The SFT case study: chat templates, response masking, and the data formats CPT deliberately does not use. |
| `cheat-sheets/CH-13-Instruction-Fine-Tuning.md` | The SFT config card — read alongside this one, since a CPT → SFT stack inherits the LR and replay decisions from here. |
| `case-studies/CS-10-Quantization-Part-1.md` / `cheat-sheets/CH-10-Quantization.md` | The prerequisite quantization material both CPT and SFT assume when the base is loaded in 4-bit. |

*End of IQ-12. Companion artifacts: `case-studies/CS-12-Domain-Adaptive-Continued-Pretraining.md`, `cheat-sheets/CH-12-Domain-Adaptive-Continued-Pretraining.md`, `code/03_continued_pretraining.py`.*
