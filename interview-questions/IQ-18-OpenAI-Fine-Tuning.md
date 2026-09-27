# IQ-18 — Interview Questions: OpenAI Hosted Fine-Tuning (Supervised Fine-Tuning on a Managed API)

| Field | Value |
|---|---|
| **Module** | Managed / Hosted Fine-Tuning — the branch where you buy a behaviour and own an ID |
| **Pairs with** | **CS-18** (case study), **CH-18** (cheat sheet), `code/13_openai_finetune.py` |
| **Total questions** | **50** (12 L1 + 14 L2 + 12 L3 + 5 L4 + 7 L5) + 24 rapid-fire + 4 whiteboard tasks + 10 CS-18 §19 self-check answers |
| **Levels covered** | L1 screening → L5 incident response |
| **Source material** | CS-18 §0–§20 + Appendix A/B, CH-18 §1–§13, `code/13_openai_finetune.py` (flags verified by running `--help`) |
| **Also contains** | Rapid-fire true/false table, 4 coding/whiteboard tasks with grading notes, numbers to memorize, the ten CS-18 §19 self-check answers |
| **Time to work through** | ~5 h at interview pace; ~60 min for a revision pass (L1 + L2 + rapid fire + numbers) |
| **Differentiate from** | **IQ-17** (Axolotl) and **IQ-19** (Vertex). CH-18 §9.4 is the two-provider comparison — this bank assumes it and does not re-ask IQ-19's questions |

**Ground truth used throughout:** the three-layer abstraction **data contract → job → endpoint** (CS-18 §2.3, restated at §20); the break-even inequality `P_f < P_b/2 − 2O` (CS-18 §4.6.5); and the billing identity `billed_tokens × n_epochs × price_per_M` (CS-18 §11.1). Every flag named below was confirmed against `python code/13_openai_finetune.py --help`. **Every price and model id in this file is a dated snapshot copied out of CS-18 or CH-18, not a current fact — CS-18 §16.8 says in its own words: "re-verify every date on this page before you rely on it," and §B.2 is titled "Platform documentation to re-check before relying on anything here."**

---

## How To Use This File

- **L1 — Fundamentals & Vocabulary (12).** What the product *is*: an endpoint ID, a JSONL contract, a per-token bill, three knobs. A candidate who says "it trains a model for you" has not understood what they are buying.
- **L2 — Applied & Implementation (15).** The seven validator checks, the two token counters, the auto-epoch policy, the file lifecycle. This is where "I read the docs" separates from "I have shipped a job."
- **L3 — Advanced, Internals & Theory (12).** The break-even derivation, the four-term cost model, why inference *is* the bill, and what the endpoint-only artefact costs you architecturally.
- **L4 — System Design & Scenario (5).** Requirements → constraints → design → trade-offs → failure modes. 10–15 minutes each, out loud.
- **L5 — Debugging & Incident Response (8).** A symptom and a clock. Answer with a sequence and name the look-alike you are ruling out.

**The meta-rule:** an answer of "it depends" is a failure unless it immediately says what it depends on and then picks a default. Always pick the default.

**The three answers that get people hired in this module:** (1) *you are buying a behaviour, not a model* — the deliverable is an `ft:` string, and §4.1.1's six lost capabilities are the price; (2) *training is billed per 1,000,000 training tokens, and there is no GPU-hour line* — the whole cost model follows from that one sentence; (3) *inference is the bill* — 95–99.9% of lifetime cost, carrying a ~2× markup over the base model, which is why the break-even is an inequality about prompt length and not about training cost.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. In one sentence, what does hosted fine-tuning actually return?**

- **Answer:** An **endpoint ID** — a string of the shape `ft:<base>:<org>:<suffix>:<hash>`, e.g. `ft:gpt-4.1-mini-2025-04-14:org:support-v3:AbCd1234`. The script's own docstring says it in five words: **"You do not get weights. You get an endpoint name."** CS-18 §4.1 puts the same point structurally — *"the difference is not performance, it is the artefact"* — and §17 #11 sharpens it: "You own an **ID string**. Not weights, not a checkpoint, not a portable artefact."
- **Why asked:** It is the module's single load-bearing fact. Every downstream consequence — no merging, no quantising, no self-hosting, no migration, the wind-down exposure — follows from it mechanically.
- **Trap:** "You get a fine-tuned model." You get a *handle* to one, hosted by someone else, that stops working when they say so (CS-18 §16.8).

**Q2. Name four things an open-weight fine-tune gives you that the hosted one does not.**

- **Answer:** From CS-18 §4.1.1's "six capabilities you lose" and §9.3's hard limitations: you cannot **download** the weights, **merge** an adapter, **quantise** to 4-bit, **self-host or serve locally**, **move it to another provider**, or **use it as a teacher** to distil into a model you own without a terms check (§10 #20). The practical test: after the model works, you will want it cheaper, on-prem, in a batch job on your own GPU, or distilled 30× smaller. The endpoint can be none of those (§1.3 item 5).
- **Why asked:** The "what you cannot do" list *is* the buy-side of the decision, and it is the question that reveals whether a candidate has thought past the demo.
- **Trap:** Answering with performance. The difference is not quality; it is the artefact.

**Q3. Write the minimal valid JSONL training row.**

- **Answer:** One JSON object per line, with a top-level `messages` array of message objects, each carrying `role` and `content`:
  `{"messages":[{"role":"user","content":"How long does the warranty last on a new smartphone?"},{"role":"assistant","content":"Most smartphones include a one-year limited warranty that covers manufacturing defects."}]}`
  CS-18 §4.2.1 gives this verbatim, and its §B.3 verdict on the video's schema walk-through is that the schema "is the one part of this module that has not moved."
- **Why asked:** It is the data contract, and CS-18 §2.3 makes it the first of the three layers. Also, this is the only part of the stack that is portable.
- **Trap:** Assuming one *example* per file, or a top-level array. It is **one JSON object per line**, and a single malformed line kills validation for the file (§10 #7).

**Q4. What roles does the current chat API accept, and which legacy role did `tool` replace?**

- **Answer:** `system`, `developer`, `user`, `assistant`, `tool` — with `function` as a legacy opt-in. `tool` replaced `function` when the tool-calling API landed. The relevant per-role allow-list is `system`, `user`, `assistant`, `tool` for the `name` field. CS-18 §4.2.2 states both tuples; §10 #8 warns that misusing `name` is a *silent quality cost*, not an error.
- **Why asked:** The copied validator in circulation allows `("system","user","assistant","function")`, so a candidate who has only ever run that validator will confidently answer "four roles, including function." The interviewer is checking whether you know your allow-list is a **versioned artefact**.
- **Trap:** "`developer` is the same as `system`." It is accepted by newer models as a system-equivalent, but treating the two as interchangeable changes what the model learns about priority (§4.3.2).

**Q5. What are the three training hyperparameters, and which one is "a lie"?**

- **Answer:** `n_epochs`, `batch_size`, `learning_rate_multiplier` (CS-18 §4.7). The lie is **`batch_size`**. CS-18 §7.5's Correction: the value is "accepted, echoed back in `job.hyperparameters`, and is not meaningful" — on a 10-row dataset the automatic rule (~0.2% of examples, floored at 1) resolves to `batch_size=1`, so the instructor's explicit `16` is a request the trainer cannot honour as written. CS-18 §18 #2 names it outright: "Three hyperparameters, one of which is a lie."
- **Why asked:** It is the module's cleanest test of "do you read the resolved job back, or do you read your own request?" That habit generalises to every managed API.
- **Trap:** "It's a lie because it's ignored entirely." It is not ignored — it is *resolved*, and reading the resolved value is how you learn what actually ran (§4.7.3).

**Q6. What does `learning_rate_multiplier: 1.0` mean?**

- **Answer:** "Use the provider's default base learning rate as-is" — **not** zero, and not 1e-5. The relation is `lr = base_lr × learning_rate_multiplier`, where `base_lr` is set by the provider and is not exposed. So a multiplier of 2.0 over a base of 0.05 is 0.1, while a multiplier of 0.1 over a base of 2.0 is 0.2 — **the same number means opposite things across providers and across versions of the same provider's stack.** CS-18 §4.7.2's rule: "treat this knob as *ordinal, not cardinal*."
- **Why asked:** The video gets this wrong on camera ("zero learning rate, to stabilize the training"), so it is a real discriminator. It also tests whether you understand the difference between a knob and a quantity.
- **Trap:** Reasoning about an absolute learning rate. If you need one, that is an argument for open-weight training (CS-13 §7.3) where you set it yourself.

**Q7. How is a supervised fine-tuning job billed?**

- **Answer:** Per **1,000,000 training tokens processed**, computed as `billed_tokens × n_epochs × price_per_M` (CS-18 §11.1, §4.6.4). Wall-clock duration does not enter the bill at all: a 10-token job that queues for 40 minutes and a 500M-token job that runs for six hours are priced by the same formula. CS-18 §11.1's title is the whole answer — "There is no GPU-hour line in a hosted fine-tuning budget."
- **Why asked:** The video reads the price as an hourly GPU rate and produces a figure wrong by **248×**. A candidate who repeats that framing will mis-budget every job for the rest of their career in this stack.
- **Trap:** "But reinforcement fine-tuning is hourly, so the framing is roughly right." RFT is the exception that proves the rule (§11.6) — and conflating the two is exactly the error being tested.

**Q8. What does the auto-epoch policy do when you pass `n_epochs="auto"`?**

- **Answer:** The constants are `TARGET_EPOCHS=3`, `MIN_TARGET_EXAMPLES=100`, `MAX_TARGET_EXAMPLES=25000`, `MIN_DEFAULT_EPOCHS=1`, `MAX_DEFAULT_EPOCHS=25` (CS-18 §4.8). Below ~34 examples, `n_epochs = min(25, 100 // N)`; between 34 and 8,333, `n_epochs = 3`; above 8,333, `max(1, 25000 // N)`. So 10 examples → **10 epochs**; 8,000 examples → **3 epochs**. The policy holds *example observations* at ~100 in the low regime and ~25,000 in the high regime.
- **Why asked:** It explains why the video's first job resolved to 10 epochs on a tiny dataset and had to be cancelled — the most common first-run surprise on this platform.
- **Trap:** Treating the resolved number as a recommendation. CS-18 §18 #4: "The auto-epoch policy is a **cost bound**, not a quality recommendation."

**Q9. What is the API's minimum example count, and what is the *useful* minimum?**

- **Answer:** 10 rows is the API's guard rail; **50–100** is the smallest thing worth a job (CS-18 §10 #1, and §Appendix A.4 records the instructor saying the same: *"minimum you can keep 10 examples… 50 to 100 example are the good one."*). Below ~50, generate synthetic data (CS-09) rather than shipping a template. CS-18 §18 #9 states the consequence flatly: **"On a 10-row dataset you produce a template, not a model."**
- **Why asked:** It separates "the API accepted it" from "it did anything." The whole wind-down narrative rests on teams mistaking acceptance for capability.
- **Trap:** Citing the 10-example minimum as evidence that small datasets work. The API accepts the file; that is a structural check, not a statistical one.

**Q10. What are the seven validation checks, and which one matters least?**

- **Answer:** From CS-18 §4.2.2, with the error keys the reference validator emits: `data_type`, `missing_messages_list`, `message_missing_key`, `message_unrecognized_key`, `unrecognized_role`, `missing_content`, `example_missing_assistant_message`. The one that matters least is **`example_missing_assistant_message`** — CS-18 §10 #3 says so explicitly, because it is the failure that is loudest and the one the platform would have caught anyway. The dangerous one is `missing_content`, whose condition is `(not content and not function_call)`, so `""`, `None`, `0`, `[]` and `{}` are all treated as missing.
- **Why asked:** Knowing which check is cheap tells the interviewer you optimise the *right* gate. §10 #3's framing — the demonstrated error is the least important one — is the insight.
- **Trap:** "Checks 3, 4, 5 and 6 are independent." They sit in one loop and none of them `continue`, so a single bad message can increment up to four counters and one row can be counted four times (§4.2.3). The totals are **violations**, not examples.

**Q11. Decode the model id `ft:gpt-4.1-mini-2025-04-14:org:support-v3:AbCd1234`.**

- **Answer:** Four fields after the `ft:` prefix — the **base snapshot** the adapter is bound to, the **organisation**, the **suffix you chose**, and a provider-generated **hash**. CS-18 §4.1.5 decodes the anatomy and §16.2 builds a registry keyed by the parts you control. The important property: the base snapshot is *inside* the ID, which is why a base deprecation can break inference on a fine-tune that has its own ID (§10 #16).
- **Why asked:** It looks like trivia and is actually the failure model. The ID is the only handle you have, and it is welded to a model revision you do not control.
- **Trap:** "The suffix identifies the model." Two jobs with the same suffix and the same base produce colliding-looking IDs (§10 #13), and the suffix is capped and **silently truncated** (§10 #12).

**Q12. Why does `content: null` appear on a valid assistant turn?**

- **Answer:** Because that is the *correct* modern shape for an assistant turn that only calls a tool: `{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup_order","arguments":"{\"id\":\"88231\"}"}}]}`, followed by `{"role":"tool","tool_call_id":"call_1","content":"…"}` (CS-18 §4.2.1's maximal example). The copied validator's check 6 uses `(not content and not function_call)`, so it flags this **correct** row as `missing_content` (CS-18 §6.3's Correction).
- **Why asked:** It is the cleanest example of a validator being *wronger* than the API, which is the module's most transferable lesson about copied allow-lists.
- **Trap:** Setting `"content": ""` to silence the validator. `""` and `None` are treated identically by the check, so the counter still fires — and you have now changed the data to satisfy a broken tool.

---

## Level 2 — Applied & Implementation

**Q13. Spot the three bugs in this JSONL file's handling: it has a UTF-8 BOM, Windows line endings, and one line with a trailing comma in the `messages` array.**

- **Answer:** Three distinct failures with three distinct symptoms. (1) The **BOM** makes the first `json.loads` see `﻿{` — fix by reading with `encoding="utf-8-sig"` (CS-18 §10 #6). (2) **CRLF** line endings put a stray `\r` in the parsed strings — fix by writing with `newline="\n"` (CS-18 §10 #5). (3) The **trailing comma** raises on that line, and because the parse loop is a plain `for line in f: json.loads(line)`, the loop **stops there** — so your file's apparent row count is wrong and every row after it is silently unreported (CS-18 §10 #7). Fix with a per-line `try` that records the line number.
- **Why asked:** These are the three the platform will *not* catch for you, and they are the ones that make a 4,239-byte file look like a 6-row file. §10 #7's precise wording is worth having: one malformed line kills the whole file's *validation*, but not the whole file's *content*.
- **Trap:** Fixing the BOM and CRLF and declaring the file clean. The truncated line count is the one that silently changes your epoch arithmetic, because your row count feeds the auto-epoch policy.

**Q14. The local validator reports `message_unrecognized_key: 1` on every single row of a tool-using dataset. What is happening?**

- **Answer:** The validator is wrong, not the data. The copied allow-list in the reference validator permits `name`, `function_call` and `weight` and **rejects** `tool_calls`, `tool_call_id` and `refusal`, all of which are valid in the current chat schema (CS-18 §4.2.3's Correction and §6.3). A team that adopts it unmodified and fine-tunes a tool-using model concludes their data is broken. The fix: **drive the allow-list from a constant and update it deliberately**, pinned beside your SDK version.
- **Why asked:** It is the module's best "your tooling is lying to you" question, and it generalises to every validator you will ever copy off a blog.
- **Trap:** Regenerating the data to satisfy the validator — i.e. deleting `tool_calls` from a tool-calling dataset to make a stale check pass.

**Q15. A structurally broken file: does it fail at upload or at job creation, and why does the distinction matter?**

- **Answer:** At **upload**, in the `validating_files` state (CS-18 §10 #2) — the API rejects structural errors before a job runs. It matters twice over: the failure is **cheap** (a round trip, not a training run), and it means the local validator is "a convenience, not a safety net — the server would have caught it anyway." Run it locally because round trips are slow, not because they are dangerous.
- **Why asked:** It tests whether the candidate understands the lifecycle well enough to know where the gates are, and whether they have an accurate model of their own tooling's value.
- **Trap:** "So I can skip the local validator." Two reasons not to: round-trip latency, and the local validator catches things the platform accepts but you did not intend (wrong roles, wrong shape) — the platform only checks structure.

**Q16. You count 562 tokens with a content-only counter. The bill says 672. Where did 110 tokens come from?**

- **Answer:** **Per-message overhead**: 3 tokens per message plus 3 for reply priming, over roughly 21 messages, plus one token per `name` occurrence. That is **+19.6%** (CS-18 §4.5.2's notebook output literally prints `# -> 562 672 +19.6%`). The reference implementation's `num_tokens_from_messages(messages, tokens_per_message, tokens_per_name)` adds `tokens_per_message` per message, `tokens_per_name` per `name` key, and a fixed reply-priming cost — and the content-only counter `sum(len(encoding.encode(m["content"])) for m in messages)` models none of it.
- **Why asked:** It is the arithmetic that turns a "free" training run into a bill, and the video's own job is only 10 rows — so the overhead *dominates* on small datasets and vanishes on large ones. Knowing which regime you are in is the skill.
- **Trap:** Dismissing 19.6% as rounding. On a 6M-token job it is ~$1; on the platform's whole billing model it is the difference between a 16% underestimate and a correct budget — CS-18 §19 A3 computes it as a **16.4% underestimate** ($0.002529 vs $0.003024).

**Q17. Which tokeniser do you count with, and when is it the wrong one?**

- **Answer:** `tiktoken`. `cl100k_base` is what the video uses and what covers `gpt-4`/`gpt-3.5-turbo`-era models; **`o200k_base`** is the vocabulary for the `gpt-4o`/`gpt-4.1` families (CS-18 §4.5.4, §4.5.1). Using `cl100k_base` on a `gpt-4.1-*` model gives "a small systematic error." Two properties make `tiktoken` usable at all: it is **exact** (a deterministic BPE merge table — the same string always gives the same count, so the cost arithmetic is not an estimate) and it is **fast** (millions of tokens per second on one CPU core, which is why the count runs in a Colab CPU runtime while training happens in a datacentre).
- **Why asked:** It tests a claim the video makes on camera and gets wrong — *"this is also a transformer based model"* — and the error is load-bearing: if you believe the tokeniser is a model you will treat its count as approximate.
- **Trap:** "Close enough across families." The error is systematic, not random, so it does not average out — and your bill uses the *provider's* tokeniser, not yours (§4.5.4).

**Q18. `MAX_TOKENS_PER_EXAMPLE = 16385` — what is it for, what happens when you exceed it, and why is it stale?**

- **Answer:** It is the per-example cap the **epoch policy and the pre-flight check** use, not a hard API limit on what the trainer sees. CS-18 §4.5.5: it is the 2024 cookbook constant for a 16k-context family; it is stale for `gpt-4.1`, whose context is far larger. What happens when you exceed it: **truncation, and the truncation is silent** — the notebook's own print says `0 examples may be over the 16,385 token [limit]`, and the `> 0` case produces no error. The script keeps the constant deliberately, with a comment saying so, and prints a histogram instead.
- **Why asked:** It is a two-part question where the second part (silent truncation) is what actually costs money. It also feeds the §4.6.3 reconstruction: the video's $0.702396 is very close to `10 examples × 16,385 max tokens × $1.50/M × 3 epochs = $0.737` — i.e. the cost you get when the **cap**, not the token count, drives the sum, over-counting by **244×**.
- **Trap:** "It's the API's limit, so raise it." It is a local policy constant. Nothing about it is enforced by the platform.

**Q19. You set `batch_size: 16` on a 10-row dataset. What trains, and how do you know?**

- **Answer:** What trains is `batch_size=1` — the automatic rule is ~0.2% of examples (0.02 rows here) floored at 1. To know: **read the resolved hyperparameters back**. CS-18 §4.7.3's auto-resolved echo shows a job with *no* hyperparameters block coming back as `batch_size=1, learning_rate_multiplier=2.0, n_epochs=10`; §15.1 records the video's job as "`batch_size=16` (not honoured)". The gap between your request and the resolved job is the whole lesson.
- **Why asked:** It is the concrete form of "the API is a service, not a library." Every managed platform has fields that are echoed but not honoured, and the engineers who ship are the ones who check.
- **Trap:** "The job failed validation." It succeeded, and echoed your 16 back, so nothing in the response tells you it was ignored.

**Q20. State the three-way interaction between the knobs, and what follows from it.**

- **Answer:** `total_optimiser_steps = n_epochs × (n_examples / batch_size)` and `effective_learning_signal ≈ learning_rate_multiplier × total_optimiser_steps` (CS-18 §7.5). What follows: changing one knob desynchronises the run, because a larger batch wants a larger LR (and vice versa), and the effective signal is a *product*, not a sum. Hence CS-18 §19 A5's conclusion: pass `batch_size="auto"` and `learning_rate_multiplier="auto"` on nearly every job, set `n_epochs` deliberately, and treat manual values as a commitment to tune all three together.
- **Why asked:** It is the only place in the module where the candidate has to reason about the interaction rather than a single knob, so it separates memorisers from practitioners.
- **Trap:** Reasoning about the learning rate independently. A multiplier is meaningless without the step count it is applied over.

**Q21. What do you lose if you never set `validation_file`, and what are the metric names the platform returns?**

- **Answer:** You lose the entire holdout half of the evaluation. CS-18 §4.9.1 states the consequence exactly: **"without a validation file, the hosted API gives you no signal about generalisation at all,"** and it records that `validation_file=None` in **every job in both notebooks**. The returned names are **`train_loss`, `valid_loss`, `full_valid_loss`, `train_mean_token_accuracy`, `valid_mean_token_accuracy`** (CS-18 §4.9.2 — and note §12.1 covers only **four** of these and omits `full_valid_loss`). Without a validation file you get `train_*` only, and no way to see overfitting.
- **Why asked:** It is the same failure as IQ-17's `val_set_size: 0` default, arriving through a different door — and the interviewer is listening for whether the candidate names the *missing metric* rather than the missing flag.
- **Trap:** Reading `valid_loss` as the quality number. §17 #10: loss is a token-level language-modelling metric — "It has no opinion about whether your answers are correct, on-brand, or safe."
- **See:** the naming `> **Correction:**` block below.

**Q22. What is the `identical_outputs` gate, and what threshold makes it fire?**

- **Answer:** Compute the base model's outputs and the fine-tuned model's outputs over the same ~50 prompts at `temperature=0` and measure the fraction that are **identical**. Near 1.0 means the fine-tune changed nothing — CS-18 §12.2 documents the gate firing at **< 0.90**; §19 A7 gives 50 prompts as the sample size. It is the cheapest no-op detector that exists, and it belongs in the harness before a benchmark suite.
- **Why asked:** It is the test that catches the most embarrassing possible outcome — shipping an endpoint that is a no-op — and almost nobody runs it.
- **Trap:** Running it with `temperature > 0` and reading sampling noise as a difference. Set the temperature to 0 or the gate is meaningless.

**Q23. "The model returns badly-shaped JSON." Is that a fine-tuning problem?**

- **Answer:** No, until you have proven otherwise. CS-18 §4.12's Correction-adjacent block and §15.3 are unambiguous: **shape failures are a schema problem; content failures inside a valid shape are a fine-tuning problem.** The right fix for the first is structured outputs — `response_format={"type":"json_schema","strict":true}` — which makes invalid output structurally unreachable at **100%, not 99%**, at zero training cost. §15.3's team "almost spent $3,000 and a 2× markup to fix a shape failure" whose correct fix was "$0 and one afternoon."
- **Why asked:** It is the single most common reason practitioners reach for fine-tuning and get a worse result at a higher price. An interviewer asks it to see whether you reach for the cheap tool first.
- **Trap:** "Fine-tuning will teach it the format more reliably than a schema." A schema is not reliable-but-imperfect; it is *unreachable* when it rejects invalid output. Fine-tuning is the weaker guarantee at the higher price.

**Q24. When is hosted DPO the right method, and what must exist before you run it?**

- **Answer:** Right when the model **can already perform the task** and you need it to *prefer* one of two behaviours — tone, refusal boundaries, verbosity, safety; wrong when the model cannot produce the behaviour at all (CS-18 §19 A8). A **supervised fine-tune must exist first**: DPO runs on top of an SFT checkpoint, never on a base model. It wants `n_epochs=1`, and the signature failure is **likelihood collapse**, where the probabilities of both the chosen and the rejected responses fall together. On the API it is a `method.type = "dpo"` with `chosen`/`rejected` sibling arrays.
- **Why asked:** It tests whether the candidate knows DPO is a *preference* method rather than a better SFT, which is the same confusion IQ-13 and IQ-15 test on the open-weight side.
- **Trap:** Running DPO to teach an absent capability. You cannot prefer what you cannot produce.

**Q25. Reinforcement fine-tuning: what is the billing model, and what does that require of the task?**

- **Answer:** RFT is billed **per hour** of training compute (CS-18 §11.6, §4.10) — the only hourly line in the whole platform — and it optimises directly against a **grader** (a Python function, a unit-test harness, or an LLM judge) rather than a learned reward model. The requirement that follows is hard: the task must have an **automatically verifiable answer**. Math that matches, code that passes tests, JSON that validates. It is useless for "make the tone friendlier," because you cannot write a grader for that.
- **Why asked:** It is the module's cleanest taxonomy question, and the video mislabels the method as "RLHF" on camera — §4.4.2's Correction: "In an interview, saying 'hosted RLHF' when you mean RFT is an immediate tell."
- **Trap:** "It's RLHF with a different name." Classical RLHF trains a separate reward model on human preference pairs and runs PPO against it with a KL anchor. RFT has no reward model, no preference dataset, and no PPO.

**Q26. Two lifecycle calls the video skips. What are they, and what breaks without them?**

- **Answer:** (1) **`files.retrieve` / inspect the upload** — CS-18 §5's nine-call block annotates it "the check the video skips," and it is how you confirm the file the platform accepted is the file you meant to send. (2) **Cleanup** — delete the uploaded files once the job is done, and delete the model when it is retired or superseded. Two reasons it matters beyond housekeeping: uploaded files **persist until explicitly deleted**, so "a fine-tuning file is a permanent copy of your data at a third party" (§4.13), and §17 #17 lists "I should delete the training file once the job succeeds" as a misconception *in the other direction* — people delete too early or not at all, and nobody has a rule.
- **Why asked:** It tests whether the candidate thinks about the job as a lifecycle with data-retention obligations, rather than as a call that returns a string.
- **Trap:** Treating the file as transient. It is permanent until deleted, and the training file is the only asset that survives a migration (§17 #12).

---

## Level 3 — Advanced, Internals & Theory

**Q27. Derive the break-even inequality and say what each term is.**

- **Answer:** CS-18 §4.6.5 derives `P_f < P_b/2 − 2O`, where `P_b` is the **base** prompt length, `P_f` is the **fine-tuned** prompt length, and `O` is the output length. The derivation rests on two assumptions the module states: a **2× markup on both input and output** for the fine-tuned model, and a **4:1 output:input price ratio** in the base model. Rearranged in words: fine-tuning only wins on cost if the prompt you can *delete* saves more than twice what the output markup *costs you*. Because fine-tuning only shrinks the prompt and never the output, every call carries a permanent output-side penalty of `2O` before you have saved anything.
- **Why asked:** It is the module's intellectual centre, and it is on the CS-18 §20 list as the one inequality an interviewer will ask you to write down.
- **Trap:** Treating it as an equality, or as a comparison of *training* costs. Training cost does not appear in the inequality at all — it is a one-off, and the inequality is about the recurring bill.

**Q28. Apply it. Base prompt 1,200 tokens, output 150 tokens, fine-tuned prompt 400. Does the fine-tune win?**

- **Answer:** **No.** Break-even needs `P_f < 600 − 300 = 300`; you have 400. CS-18 §4.6.5's case A: base `= 1,200 × $0.40/M + 150 × $1.60/M = $0.000720`; fine-tuned `= 400 × $0.80/M + 150 × $3.20/M = $0.000800`. The fine-tune costs **more per call**. The lesson in the module's own words: **"You cannot delete enough prompt."** Compare case B — base 3,000 / output 90 / FT prompt 700 → break-even `< 1,320`, and FT wins at $0.000848 vs $0.001344 — and case C — base 450 / output 200 → break-even `< −175`, **impossible**, so the fine-tune loses at any prompt length, because the output-side markup alone exceeds the base total.
- **Why asked:** Deriving the inequality is recall; applying it to three cases and noticing which one is *impossible* is the skill. Case C is the one that changes decisions.
- **Trap:** Assuming a shorter fine-tuned prompt always means a cheaper call. The markup is on **both** directions, and the output side is where the base model was already cheap.

**Q29. Why is inference "the bill"? Give the two figures.**

- **Answer:** Because training is a **one-off** and inference **recurs**, carrying a ~**2× markup over the base model**: CS-18 §11.2 puts it at **95–99.9% of lifetime cost**. §11.3's worked example shows the shape — training `$21.35` one-off, then a monthly comparison in which the fine-tune is **+$46.40/month worse** (`{'train_one_off': 21.35, 'base_per_month': 110.4, 'ft_per_month': 156.8, 'payback_months': None}`). And the flagship tier makes it extreme: a `gpt-4.1` training run is `$5,040` *while the inference is a monthly charge that never stops* (§11.2).
- **Why asked:** It inverts the intuition candidates bring from the open-weight world, where the GPU-hour is the whole cost. Here, a **$0.003** training job can be a bad purchase if the output markup is wrong.
- **Trap:** Optimising training cost. It is a rounding error by construction; the decision lives entirely on the inference side.

**Q30. State the four-term cost model, and name the term people omit.**

- **Answer:** **Training** (`billed_tokens × n_epochs × train_price`) + **Validation** (`val_tokens × n_epochs × train_price`, provider-dependent) + **Inference** (`calls × (P_in·r_in + P_out·r_out + P_cached·r_cached)`) — CS-18 §4.6.4, with the module's own worked values `$0.0030` for training, "— (none used)" for validation, and "~$0.0002/call" → "$25–140/month depending on volume" for inference. The term people omit is **validation**: it is billed at the training rate, it scales with epochs, and because it is a holdout you may run it repeatedly while iterating — and the whole point of §4.9.1 is that most teams set it to `None` and pay for that omission elsewhere.
- **Why asked:** Three of the four terms are obvious once you say them out loud; the fourth is the one that reveals whether you have actually budgeted a job rather than estimated one.
- **Trap:** Forgetting the **cached input** term, which is the only lever that *reduces* the recurring bill and is therefore the one worth engineering (see `--cache-hit-rate` below).

**Q31. Walk CS-18 §11.3's budget. Why does the fine-tune lose, and what single change makes it win?**

- **Answer:** The case is 900 base prompt tokens / 120 output, 200,000 calls/month, on `gpt-4.1-mini`. Training: `7,116,600 ÷ 1e6 × $3.00 = $21.35` one-off. Base: `(900 × $0.40 + 120 × $1.60)/1e6 = $0.000552/call` → **$110.40/month**. Fine-tuned: `(500 × $0.80 + 120 × $3.20)/1e6 = $0.000784/call` → **$156.80/month** — **+$46.40/month worse**, so `payback_months` is `None`. The single change that flips it: shrink the fine-tuned prompt from 500 to **180** tokens, giving `(180 × $0.80 + 120 × $3.20)/1e6 = $0.000528` → **$105.60/month**, a **−$4.80/month** saving and a **4.4-month payback**. CS-18's own verdict: "A $4.80/month saving. … The margins are real and thin."
- **Why asked:** It is the module's honest worked example, and the honest answer — a $4.80/month win — is far more useful than a story about a 10× saving. An interviewer wants to hear whether you would *ship* on that margin.
- **Trap:** Reporting the $4.80 as a win without the payback and the sensitivity. At 480 tokens of prompt instead of 180, the sign flips.

**Q32. At what volume does self-hosting beat the hosted model, and what is the assumption that makes it true?**

- **Answer:** Around **460,000 calls/month** (CS-18 §11.5). The comparison is base `$0.000552/call`, fine-tuned `$0.000784/call`, against a self-hosted 3B LoRA on one 24 GB card at **$0.35/GPU-hour × 730 hours = $252/month**. At 500,000 calls/month: hosted base `$276.00`, hosted FT `$392.00`, self-host `$252` — the first row where self-hosting wins. At 100,000 calls both hosted options win; at 1,000,000 self-hosting wins outright. The assumption that makes it true is the **flat $252**, which assumes the GPU is fully utilised — the same idle-capacity problem that dominates Vertex (CH-18 §9.4).
- **Why asked:** It is the question that connects the two managed modules: OpenAI's cost is usage-shaped and Vertex's is fixed-shaped, so the crossover argument runs in opposite directions.
- **Trap:** Comparing against the *fine-tuned* hosted price rather than the *base* hosted price. You are competing with "prompt the base model harder," not with your own fine-tune.

**Q33. Why is `full_valid_loss` a separate metric from `valid_loss`?**

- **Answer:** Because one is **step-level** and the other is taken at **end-of-epoch boundaries**. CS-18 §4.9.2's own row for `full_valid_loss` reads: *"Valid loss at end-of-epoch boundaries — the one to plot; step-level valid loss is noisy."* So `valid_loss` is the per-step series and `full_valid_loss` is the smoothed, epoch-aligned series. The practical rule: plot `full_valid_loss` for the overfitting decision, because the step-level series is noisy enough that a trend read off it can be all variance.
- **Why asked:** It is a metric-definition question, and it is the kind of "minute aspect" that only shows up if you have actually watched the two curves disagree on a real job. It also tests whether you know that **`full_valid_loss` is not in §12.1's four-metric table at all** — it only appears in §4.9.2's fuller field list.
- **Trap:** Reading `train_loss` against `valid_loss` across *different* step counts and calling the difference overfitting. Compare like with like: `train_loss` against `train_loss`, or `full_valid_loss` against its own previous epoch.

**Q34. Loss looks perfect and the model is useless. What are the three non-loss failure modes, and how do you detect each?**

- **Answer:** CS-18 §19 A7 gives exactly this. (1) **Memorisation** — detect with a paraphrase test on reworded versions of the training questions. (2) **Out-of-scope collapse** — hold out 20 in-scope and 20 out-of-scope prompts and report both rates, because a model that answers everything confidently scores identically to a model that refuses correctly on a loss metric. (3) **A no-op** — `identical_outputs` against the base over 50 prompts; near 1.0 means nothing happened. A fourth: a **leaked split**, detectable by `valid_loss` coming out **below** `train_loss`.
- **Why asked:** It is the module's version of "loss selects between checkpoints; task metrics decide whether to ship" (§18 #8), and it is the reason §12.2 prescribes a five-layer evaluation stack rather than a threshold.
- **Trap:** Adding a fourth layer of loss analysis. The fix is a different *kind* of measurement, not a better reading of the same one.

**Q35. A 10-row dataset at an auto-resolved 10 epochs. How many times does the model see each conversation?**

- **Answer:** With 4 unique conversations, **25 exposures each** (CS-18 §4.9.3, §17 #5's correction). The arithmetic is `epochs × (examples ÷ unique_examples)` — the auto-epoch policy counts *rows*, and rows that are duplicates of each other are separate rows to it. `train_loss → 0` while `valid_loss` rises is the signature (CS-18 §14.4's decision tree). And there is nothing to stop it, because `validation_file=None` means you have no `valid_loss` to look at.
- **Why asked:** It links three separate facts — the epoch policy, the missing validation file, and §18 #9's "you produce a template, not a model" — into one mechanism.
- **Trap:** Fixing it by cutting epochs. On 4 unique conversations, no epoch count gives you a model; §19 A7's first move is to stop early, but the honest fix is more data.

**Q36. Why does the wind-down matter to the *architecture*, not just to the procurement?**

- **Answer:** Because the endpoint is welded to a base snapshot you do not control (Q11), so a base-model deprecation breaks the inference path on a fine-tune that has its own ID (CS-18 §10 #16). That makes an endpoint-only artefact a **time-limited** asset, and it forces you to build the exit **before** you need it. CS-18 §16.8's migration tree and §19 A10 give the shape: freeze the behaviour by sampling the existing model over the production prompt distribution (100k+ completions, `temperature` 0.7–1.0, deduped), then fine-tune an **open-weight student** on those samples, then validate against a frozen test set before cutover. It costs roughly **$340** at 120k samples (§15.4) — and it must begin immediately, "because the teacher is only callable while the base model still serves," and generating the samples is a week of wall-clock that cannot be compressed at the end.
- **Why asked:** It is the module's strategic question, and the answer that impresses is "the distillation corpus is a schedule item, not a contingency."
- **Trap:** Treating the wind-down as a procurement problem. The dates are the *less* important half; the architectural point is that only the **dataset** crosses the boundary (§17 #12 — "There is no migration. There is a **rebuild**").

**Q37. Compare the two shapes of "managed": OpenAI and Vertex.**

- **Answer:** From CH-18 §9.4. **OpenAI:** idle cost **$0**, dominant recurring term the **~2× per-token markup**, dataset role `assistant`, system message allowed at **any position**, hyperparameters `n_epochs` / `batch_size` / `learning_rate_multiplier`, billed **per 1M training tokens**, prerequisite an **API key**, status **winding down**. **Vertex:** idle cost **$3–5 per hour while deployed**, dominant term the **hourly endpoint charge**, role `model` not `assistant`, system message **first message only**, hyperparameters `epochs` / `learning_rate_multiplier` / `adapter_size`, prerequisite a **GCP project + bucket + IAM**, status available. The one-line summary: OpenAI's bill is **usage-shaped**; Vertex's is **fixed-shaped**.
- **Why asked:** It is the cross-module question, and the answer determines the recommendation before any pricing page is opened: low or spiky volume cannot justify an always-on endpoint, regardless of quality.
- **Trap:** Comparing the two on training price. Both are negligible; the difference is entirely in the recurring line, and the recurring lines have different *shapes*, not just different sizes.

**Q38. What does the data contract's portability actually get you?**

- **Answer:** It gets you the **rebuild**. CS-18 §17 #13: "The **data contract** is portable … The **job API**, the **hyperparameters**, the **pricing** and the **model line-up** are not, and all four churn monthly." §17 #12: "There is no migration. There is a **rebuild**, and the only asset that crosses the boundary is your dataset." Concretely: a `{"messages":[…]}` JSONL that trains a `gpt-4.1-mini` will train a Llama-3.1-8B with minor changes — which is exactly CS-17's inline dataset format, and the reason the migration target is an Axolotl config.
- **Why asked:** It reframes the "vendor lock-in" answer from "we would lose everything" to "we would lose everything except the one thing that took the longest to build."
- **Trap:** "So portability is fine." The dataset crosses; the **evaluation harness and the prompt/schema contract** must cross with it or the rebuild is unverifiable — §19 A10's list of what survives includes all three.

---

## Level 4 — System Design & Scenario

> These are 10–15-minute whiteboard questions. Answer in the fixed shape: **requirements → constraints → design → trade-offs → failure modes.**

**Q39. 2M support tickets a month, a 1,800-token few-shot prompt, and a mandate to cut inference cost. Design it.**

- **Answer:** *Requirements:* reduce the largest controllable line; keep the escalation policy; the 12 few-shot examples are the entire prompt and are exactly what a fine-tune encodes. *Constraints:* `gpt-4.1-mini`; volume 2M/month; the prompt must shrink or the project is pointless. *Design:* 8,000 labelled historical tickets, 12% held out, ~400-token average, **deduped by content hash**; `n_epochs=2`; `batch_size="auto"`; `learning_rate_multiplier="auto"`; **`validation_file` set**; `seed=42`; `metadata={"data_version": "…"}`. Training `(7,040 + 960) × 409 × 2 / 1e6 × $3.00 = $19.62`. *Trade-offs:* prompt drops 1,800 → **240** tokens, so base `(1800 × $0.40 + 200 × $1.60)/1e6 × 2M = $2,080/mo` becomes FT `(240 × $0.80 + 200 × $3.20)/1e6 × 2M = $1,664/mo` — **$416/month saved, payback in 1.4 days**. *Failure modes, both recorded in the source:* (1) the first run used **raw ticket text including customer names and order IDs** — a privacy review caught it before upload, and the second run scrubbed with a regex pass plus manual review of 200 samples; (2) round 1 ran with `validation_file=None`, looked fine, shipped, and two weeks later escalation rose because the historical data was **biased toward tickets that got escalated** and the model learned to always escalate — only a **paired eval against the base model** caught it.
- **Grading:** the dedup, the held-out split, and the paired eval are the three non-negotiables. Credit for naming the privacy scrub as a *pre-upload* gate.
- **Fail condition:** proposing the fine-tune before checking whether the 12 few-shot examples can be replaced by structured outputs, or shipping without a base-model comparison.

**Q40. Design the procedure that answers "should we fine-tune at all?" before anyone writes a JSONL.**

- **Answer:** *Requirements:* a decision that can be made in an hour and defended in a review. *Design, in order:* (1) **§8.3's one-question test** — can you state the behaviour in a sentence, and is it a *behaviour* rather than a *fact*? Facts are a retrieval problem (CS-04). (2) **Shape or content?** If the failure is "the JSON is malformed," the answer is structured outputs and you are done (§4.12). (3) **Run the break-even** — compute `P_b`, `P_f`, `O`, and check `P_f < P_b/2 − 2O`. If it fails, the fine-tune needs a non-cost justification, and you must say what it is. (4) **Do you have 50–100 examples minimum, and a held-out set?** (5) **Will you own the endpoint for its lifetime**, including the exit? *Trade-offs:* steps 1–4 cost an afternoon and no data collection; step 5 costs a schedule item. *Failure modes:* the two the source names — fixing a shape failure with $3,000 of annotation and a permanent 2× markup (§15.3), and fine-tuning a *retrieval* problem, which §10 #18 says makes things "worse than the original problem" because the model now confidently fabricates instead of failing visibly.
- **Grading:** the ordering matters, and "we'll measure later" is the fail. §18 #15: "A fine-tuning run without a paired baseline is a purchase, not an experiment."
- **Fail condition:** starting from the dataset. The dataset is the *third* question, not the first.

**Q41. Design the serving, versioning and rollback story for the endpoint.**

- **Answer:** *Requirements:* the endpoint is the only handle, and it is welded to a base snapshot. *Design:* a **model registry keyed by the values you control**, not by the provider's hash — CS-18 §16.2 gives the literal: `"support-v3": {"id": "ft:gpt-4.1-mini-2025-04-14:org:support-v3:Cc33", "sha": "5e6f7a8b"}` alongside `-v1` and `-v2`. Put the base and the data version in `metadata` at job-creation time — §4.4.2's run-record note: "A run whose `metadata` says `{"base": "gpt-4.1-nano-2025-04-14", "data_version": "v7"}` never has this problem." Assert the model id at import time in the serving wrapper (`MODEL_ID = os.environ["FT_MODEL_ID"]`), so a misconfigured deploy fails at boot rather than at the first customer request. Rollback is a registry-pointer flip. *Trade-offs:* the registry costs a file and a convention; without it, "which endpoint is production" is answerable only from a video, a bill, or a memory. *Failure modes:* the ID is not the whole identity — the **base snapshot** is inside it, so a deprecation invalidates a registry entry that looks perfectly valid (§10 #16); and a rollback that has never been executed is documentation, not a capability.
- **Grading:** the registry keyed by *your* names plus the `metadata` provenance line is the core. Credit for the import-time assertion.
- **Fail condition:** versioning by date stamp or by the provider hash, which tells a rollback decision nothing.

**Q42. Design the migration off a deprecated base snapshot, with the clock running.**

- **Answer:** *Requirements:* preserve behaviour, not weights — the weights were never yours. *Constraints:* the teacher is only callable while the base still serves, and distillation-sample generation is a week of wall-clock that cannot be compressed at the end (§19 A10). *Design:* **start now, in parallel with everything else.** (1) Replay the production prompt distribution against the existing endpoint — 100k+ completions at **temperature 0.7–1.0** (§19 A10), deduped — and store the pairs. (2) Freeze a test set and the prompt/schema contract alongside them. (3) Fine-tune an **open-weight student** (§15.4 uses a **7B LoRA on 4×A100 for ~6 GPU-hours**) on the samples. (4) Validate against the frozen set before cutover — §15.4's bar is **91% agreement** with the hosted model. (5) Keep the endpoint live until cutover is proven. *Trade-offs:* generation is ~**$340** at 120k samples, training ~**$15** of GPU time, serving ~**$1,100/month flat** (§15.4) — so the migration *changes the cost shape* from usage-billed to fixed, which is the same trade Vertex makes and must be justified on volume. *Failure modes:* §15.4 records the one that actually happened — **the first run sampled the teacher at temperature 0 and produced 120,000 near-identical completions, so the student learned a lookup table and scored 62% agreement**; raising the temperature to **0.9** and adding a dedupe pass took it to 91%. So the diversity of the sample corpus *is* the quality of the student. Also: the sample distribution is not the production distribution; the ToS question on using a provider's model as a teacher must be checked (§10 #20, and the AP-01 ethics appendix — `appendices/AP-01-Ethics-and-Philosophy-of-Alignment.md`); and starting late means the teacher has already stopped answering.
- **Grading:** parallelism is the graded decision. A plan that sequences "wait for the deprecation notice, then generate samples" fails on the clock alone.
- **Fail condition:** proposing to download, export, or otherwise rescue the fine-tuned weights.

**Q43. Design the evaluation stack that decides whether to ship.**

- **Answer:** *Requirements:* loss selects between checkpoints; task metrics decide whether to ship (§18 #8). *Design, the five layers of CS-18 §12.2, cheapest first:* (0) **load check** — the endpoint answers at all; (1) **structural** — does the output parse and satisfy the schema; (2) **exact / reference** — against a frozen expected-output set; (3) **model-as-judge** — a stronger model scoring open-ended outputs, with every pair judged in **both orders** to cancel position bias; (4) **memorisation and no-op detectors** — a **13-gram verbatim recall** test and `identical_outputs` at `< 0.90` over 50 prompts. Plus the two the platform gives you for free but only if you set `validation_file`: `valid_loss` and `full_valid_loss`. *Trade-offs:* layers 0–1 cost seconds and catch the embarrassing failures; layers 3–4 cost money and catch the ones that ship. *Failure modes:* a judge that agrees with itself, an eval set drawn from the training distribution, and a `valid_loss` that came out *below* `train_loss` — which is a **leaked split**, not excellence.
- **Grading:** the balanced two-order judge and the 13-gram recall test are the differentiators. So is naming the leaked split.
- **Fail condition:** a stack that stops at `valid_loss`. §17 #10: without a validation file you do not even get that.

---

## Level 5 — Debugging & Incident Response

> Answer with a sequence. Say what you check first, second and third, and name the look-alike you are ruling out at each step.

**Q44. The chat call returns 404 against your fine-tuned model id. Sequence.**

- **Answer:** (1) The verbatim error is `NotFoundError: Error code: 404 - {'error': {'message': 'The model \`ft:…\` does not exist or you do not have access to it.', 'type': 'invalid_request_error', 'code': 'model_not_found'}}` (CH-18 §11). Do **not** read it as "wrong key" — that is a different error code. (2) Check the **job's terminal state**: a 404 is what you get for a model whose job was `cancelled` or `failed`, and CS-18 §6.7's first notebook earns exactly this by chatting with a cancelled job's model. (3) Check the **base-model deprecation path** — §10 #16: a base deprecation breaks a fine-tuned model's inference path even though the FT model has its own ID. (4) Check **organisation scope**: the ID embeds the org, and a key from another org cannot see it. (5) Only then check the literal string for a typo — the video's own example is a truncated hash, `ft:gpt-3.5-turbo-0125:personal::9XbBvXXXXXXXXXXXXXXXXX`.
- **Ruling out:** the playground-versus-API mismatch (§10 #17) — the playground and the API are not the same client, so "it works in the playground" is not evidence about your key, your org, or your rate limits.
- **Grading:** step 2 before step 5. The instinct to check the string first is the failure.

**Q45. `valid_loss` comes out *below* `train_loss` and stays there. What is happening?**

- **Answer:** (1) This is the **leaked-split detector** (CS-18 §19 A7). It is not excellence; it is a data-handling bug. (2) The usual cause: the validation rows are a **subset or near-duplicate** of the training rows — the same conversations appearing on both sides, which on a small dataset is exactly what happens when you split after deduplication instead of before, or when you split by row rather than by *conversation*. (3) Check the dedup: §15.2's production case is "deduped by content hash" for this reason. (4) Check the split unit — multi-turn conversations must be split as units, not as turns. (5) Re-split and re-run, then confirm the two curves diverge in the normal direction before believing any other number in the report.
- **Ruling out:** a legitimate `valid_loss` slightly below `train_loss` from regularisation or dropout, which can persist for a short run and does not mean a leak.
- **Grading:** the candidate must name the leak rather than the metric. A `valid_loss` below `train_loss` is a *smell*, and the question is whether you can say what it smells like.

**Q46. Loss goes to ~0, the answers are fluent and on-brand, and the model is useless on anything not in the training set. Sequence.**

- **Answer:** (1) This is **memorisation**, and the arithmetic explains it: CS-18 §4.9.3's "10 epochs on 4 unique conversations = 25 exposures each." (2) Confirm by looking for `valid_loss` *rising* while `train_loss` → 0 (CS-18 §14.4's decision tree row 13 names the video's own `gpt-3.5` job as this case exactly). If `validation_file` was never set, you have no such curve — which is why you are here. (3) The immediate fix is to **cut `n_epochs`** and select on the best intermediate point, not the last. (4) The real fix is more data — §18 #9: on a 10-row dataset you produce a template, not a model. (5) Then run the **paraphrase test** (§19 A7): reword the training questions and see whether the answers survive. Fluent answers to *un*reworded questions prove only that memorisation works — CS-18 §15.1's own evaluation, which asked a question that was in the training set.
- **Ruling out:** a genuinely well-fit model. The test is the paraphrase, not the fluency.
- **Grading:** the paraphrase test and the "question was in the training set" observation are the two things an interviewer wants to hear.

**Q47. The job says `succeeded`, the eval looks fine, and the model is no better than the base. Sequence.**

- **Answer:** (1) Run `identical_outputs` over 50 prompts at `temperature=0` — if it is near 1.0, **nothing happened** and you are done diagnosing (CS-18 §19 A7, gate at `< 0.90`). (2) If the outputs differ but quality does not: check the **system prompt**. §10 #10: the system prompt is part of the *model*, not part of the request — it must be supplied **byte-identically** at inference or quality drops, and "the worst configuration is train *with* and serve *without*." (3) Check the **prompt at serve time**: if your production prompt is the 1,800-token few-shot prompt you were supposed to delete, the fine-tune is being buried under its own training signal. (4) Check the **temperature**: §10 #11 — temperature interacts with a fine-tune more sharply than with a base model, so a value carried over from base-model tuning can flatten the learned behaviour. (5) Then check the data volume: below ~50 rows you have a template.
- **Ruling out:** a training failure. The job succeeded and the loss was fine; the failure is at the *seam* between training and serving, which is why §9.4's silent-failure table exists.
- **Grading:** step 2 is the answer. Train-serve system-prompt drift is the most common cause of "it worked in the notebook."

**Q48. Escalation rate rose two weeks after a successful deployment. Sequence.**

- **Answer:** (1) Treat it as a **data-bias** symptom before a model symptom: CS-18 §15.2's "second thing that went wrong" is exactly this — the historical tickets were biased toward tickets that got escalated, so the model learned to **always escalate**, and the run "looked fine and shipped." (2) Confirm by measuring the predicted escalation rate on a held-out slice against the historical base rate; a model that always escalates will match a biased dataset almost perfectly. (3) Check whether a **paired eval against the base model** was ever run — in the recorded case it was not, and that is the only thing that would have caught it. (4) Check what changed at two weeks: a prompt edit, a routing change, a data refresh, or a drift in the incoming ticket mix. (5) Fix the *data*, not the model: rebalance the training set toward the true decision boundary, and add the paired eval as a merge gate.
- **Ruling out:** drift in the world. §16.3 names four distinct drifts, only one of which is about the model — so "the world changed" is a hypothesis to check, not a default.
- **Grading:** the candidate must reach for the **paired eval** and the **label bias**, in that order of confidence.
- **Fail condition:** retraining on more of the same data, which reinforces the bias.

**Q49. `example_missing_assistant_message: 1` on a file that looks fine in a text editor. Sequence.**

- **Answer:** (1) Trust the counter over your eyes — it means one row's `messages` array has no `assistant` entry at all, and CS-18 §10 #3 warns this is the *loudest* and least important error. (2) Find the row: the reference validator's loop does not report line numbers, so add `for i, line in enumerate(f, 1)` and record which line incremented the counter (CS-18 §10 #7's pattern). (3) Check the **truncated** row first — if the file was generated by a streaming writer, the last row is the usual culprit. (4) Check the **Windows CRLF / BOM** pair, because either can corrupt the *first* row's parse and cascade (§10 #5, #6). (5) Check whether the row is legitimately a tool-only exchange, in which case it is a **false positive** from a validator that has not been told about `tool` turns (§6.3). (6) Fix and re-upload — note that the platform would have caught this too, in `validating_files` (§10 #2).
- **Ruling out:** the `missing_content` false positive, which fires on `content: null` + `tool_calls` and is a *different* check with a *different* remedy.
- **Grading:** step 5 is the discriminator. A candidate who assumes the validator is always right will "fix" valid data.

**Q50. `trained_tokens` came back lower than your estimate, and a job you cancelled still shows a charge. Two incidents.**

- **Answer:** **Lower `trained_tokens`:** not an error (CS-18 §10 #15). It is below your estimate because your estimate counted content tokens and the platform's count reflects the actual billed set — or, more often, because you over-estimated the row count with the `MAX_TOKENS_PER_EXAMPLE` cap driving the arithmetic (the §4.6.3 reconstruction, which over-counts a small dataset by **244×**). Reconcile with `billed_tokens × n_epochs`, not with your pre-flight number. **Cancelled job still billed:** expected (CS-18 §10 #14) — training tokens already processed are billable, and wall-clock does not enter the model at all. Fix the *process*: cancel early, and treat cancellation as a cost control, not a refund.
- **Ruling out:** a billing bug. Both behaviours are documented, and both are explained by the per-token model with no hourly term.
- **Grading:** the candidate must reach for `billed_tokens × n_epochs` as the reconciliation, not for support.
- **Fail condition:** disputing the charge, or concluding that `trained_tokens` being low means the job under-trained. It means your estimate was wrong.

---

## Rapid Fire — True / False / One-Liner

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | You get the fine-tuned weights. | **False** | You get an `ft:` ID string and nothing portable (§17 #11) |
| 2 | SFT is billed per GPU-hour. | **False** | Per 1M **training tokens**; there is no hourly line (§11.1) |
| 3 | Reinforcement fine-tuning is billed per hour. | **True** | The exception that proves the rule (§11.6) |
| 4 | `learning_rate_multiplier: 1.0` means zero LR. | **False** | It means "use the provider's default base rate" (§4.7.2) |
| 5 | `batch_size: 16` on 10 rows trains with batches of 16. | **False** | Resolves to 1; echoed back and not honoured (§7.5) |
| 6 | The auto-epoch policy gave you 10 epochs because 10 is optimal. | **False** | It is a **cost bound**, not a recommendation (§18 #4) |
| 7 | 10 examples is the useful minimum. | **False** | It is the API minimum; 50–100 is the useful one (§10 #1) |
| 8 | The content-token count is the billed count. | **False** | Per-message overhead adds ~19.6% (562 → 672) (§4.5.2) |
| 9 | `tiktoken` is a small transformer. | **False** | A deterministic BPE merge table, no weights (§4.5.1) |
| 10 | `cl100k_base` is right for `gpt-4.1`. | **False** | `o200k_base` is; using cl100k is a systematic error (§4.5.4) |
| 11 | `MAX_TOKENS_PER_EXAMPLE = 16385` is the API's hard limit. | **False** | A stale local policy constant; truncation is silent (§4.5.5) |
| 12 | A malformed line fails at job creation. | **False** | At **upload**, in `validating_files` (§10 #2) |
| 13 | The copied validator's role list is current. | **False** | It rejects `tool`/`tool_calls`; false positives (§6.3) |
| 14 | `content: null` on an assistant turn is invalid. | **False** | Correct for a tool-only turn; the validator is wrong (§4.2.1) |
| 15 | Training is the dominant lifetime cost. | **False** | Inference is 95–99.9%, at a ~2× markup (§11.2) |
| 16 | The fine-tuned markup applies to input only. | **False** | Both directions — which is why `2O` appears in the inequality |
| 17 | Fine-tuning reduces inference cost. | **False** | §17 #15's misconception; it usually *increases* it per call |
| 18 | `P_f < P_b/2 − 2O` decides the cost case. | **True** | The break-even inequality (§4.6.5) |
| 19 | Training cost appears in the break-even inequality. | **False** | It is one-off; the inequality is about the recurring bill |
| 20 | Fine-tuning fixes malformed JSON. | **False** | Structured outputs does, at 100% and $0 (§4.12) |
| 21 | `valid_loss` below `train_loss` means the model is excellent. | **False** | It means a **leaked split** (§19 A7) |
| 22 | Loss is a sufficient ship gate. | **False** | It is a token-level LM metric with no opinion on correctness (§17 #10) |
| 23 | You can migrate the fine-tune to another provider. | **False** | There is a **rebuild**, and only the dataset crosses (§17 #12) |
| 24 | Deleting the fine-tuned model stops the billing. | **True** | But the deployed *endpoint* is the billed thing on Vertex (CH-18 §9.4) |

---

## Coding / Whiteboard Tasks

### Task 1 — Write the validator, correctly (12 minutes)

Write the JSONL pre-flight for a tool-using chat dataset. It must not repeat the two bugs the reference validator has.

- **Expected solution sketch:** Load the role and key allow-lists from **constants** (`VALID_ROLES = ("system","developer","user","assistant","tool")`, `NAME_OK_ROLES = ("system","user","assistant","tool")`), not from a literal inside the function. Iterate `for i, line in enumerate(f, 1)` with a per-line `try/except json.JSONDecodeError` that appends the line number, so a malformed line does not truncate the count. Use a `defaultdict(int)` keyed by **line number**, not a global counter, so one row cannot be counted four times by checks that do not `continue`. Treat `content: null` **plus** `tool_calls` as valid. Read with `encoding="utf-8-sig"` and open for write with `newline="\n"`.
- **The decisions being graded:** (a) do you fix the *allow-list* or the data; (b) do you fix the *counter granularity* (line-keyed, not global); (c) do you treat the validator as a versioned artefact pinned beside the SDK.
- **Grading:** line-numbered errors, a line-keyed histogram, and a tool-aware allow-list.
- **Fail condition:** a global counter and a hard-coded role tuple — i.e. a re-implementation of the bug being tested.

### Task 2 — Compute the two token counts (10 minutes)

Given a 60-row dataset with a 41-token system prompt and ~400-token conversations, give the content-token count, the billed count, the training cost at 3 epochs, and the monthly inference cost at 200,000 calls.

- **Expected solution sketch:** Content ≈ `words × 1.33`; the module's own fixture gives **3,149 tokens/epoch** across 60 rows (CH-18 §7.3). Add per-message overhead of **3 per message + 3 reply priming** — 60 rows × 9 = **+540** → billed ≈ **3,689/epoch** (**+17.1%**). Training = `3,689 × 3 ÷ 1e6 × $3.00 ≈ $0.03`. Inference at `$0.80/M in` and `$3.20/M out` over 200,000 calls lands around **$162.87/month**, i.e. **$1,954.44 in Year 1**, against $977.23 if you used the base-model rates.
- **The decisions being graded:** that you add the overhead *before* multiplying by epochs; that you use the **fine-tuned** inference rates, not the base rates; and that you state the price is a dated snapshot to be re-checked.
- **Grading:** the billed count must differ from the content count by roughly 15–20%, and the recurring line must dominate the one-off by ≥3 orders of magnitude.
- **Fail condition:** multiplying content tokens by epochs and quoting training as the headline cost.

### Task 3 — Decide the case with the inequality (10 minutes)

Base prompt 900 tokens, 120 output tokens, 200,000 calls/month, `gpt-4.1-mini`. Should you fine-tune?

- **Expected solution sketch:** `P_f < 900/2 − 2×120 = 450 − 240 = 210`. So the fine-tuned prompt must be **under 210 tokens** for the cost case to close. Then compute both sides at the honest `P_f`: at 500 tokens, base `(900×$0.40 + 120×$1.60)/1e6 = $0.000552` vs FT `(500×$0.80 + 120×$3.20)/1e6 = $0.000784` → **+$46.40/month, the fine-tune loses**. At `P_f = 180`, FT is `$0.000528` → **−$4.80/month**, payback `$21.35 ÷ $4.80 = 4.4 months`. The honest answer is: **it depends on a prompt-shrinkage you have not yet demonstrated**, and the decision is to prototype `P_f` before collecting any data.
- **The decisions being graded:** deriving the threshold *first* and then checking whether the real prompt can meet it; reporting the thin margin honestly; noting that training cost never enters the inequality.
- **Grading:** the threshold, the sign flip, and the payback must all be present.
- **Fail condition:** answering "yes, fine-tuning is cheaper" from training cost alone, or quoting the $4.80 saving without the payback and the sensitivity.

### Task 4 — Design the pre-flight gate (10 minutes)

Write the assertions a CI job should run before a hosted fine-tune is created.

- **Expected solution sketch:** (1) parse every line with per-line error reporting and a non-zero exit on any failure; (2) assert the role and key allow-lists against **constants**; (3) assert `n_examples >= 50` (warn below 100) and that a `validation_file` is present; (4) compute **both** token counters and fail if the overhead-adjusted figure pushes the job past a budget cap; (5) assert the `--model` is a **dated snapshot**, never a moving alias, and that a price row exists for it — `resolve_price` in `code/13_openai_finetune.py` returns `None` and warns, so treat a `None` as a hard failure; (6) assert `metadata` carries `data_version` and `base`; (7) assert `suffix` is within the length cap so it cannot be silently truncated; (8) assert the training file id is passed **explicitly**, never inferred from "the newest file on the account."
- **The decisions being graded:** the snapshot-not-alias assertion, the explicit training-file id (never "newest file"), and the budget cap computed on **billed** tokens rather than content tokens.
- **Grading:** at least six of the eight, with (4), (5) and (8) present.
- **Fail condition:** a gate that only checks structure. It would pass every job that wastes money.

> **Correction (the flag table vs the live script).** CH-18 §6's table, at **line 515**, describes `--train` as "Creates a job against the newest uploaded fine-tune file" and says it "Picks `files.list()[0]` — see §11." **The script does not do that.** `python code/13_openai_finetune.py --help` documents `--file-id` as: *"Training file id from a prior `--upload`. REQUIRED for `--train`: guessing 'the newest file on the account' trains the wrong data on any account that has uploaded before."* The §6 table is also missing `--file-id`, `--validation-file-id`, `--seed`, `--metadata`, `--ft-prompt-ratio` and `--cache-hit-rate`, all of which exist. **The live `--help` output is the authority; the table is stale.** Treat `--file-id` as mandatory in any CI gate.

> **Correction (the price lookup — fixed in code, still described as broken in the docs).** CH-18 §7.2 carries a `> **Correction:**` block stating, in the present tense, that "The script matches by substring in insertion order: `next(k for k in PRICES if k in model)`. So `--model gpt-4.1-nano-2025-04-14` matches **`gpt-4.1`** at **$25.00/M** — a **16.7× overestimate** of nano's $1.50/M — and it does so **silently**." **That is no longer true of the script on disk.** `code/13_openai_finetune.py:420` defines `resolve_price()`, whose docstring names that exact bug and whose body takes the **longest** match (`max(hits, key=len)`) with a regex fallback; and `PRICES` at line 92 carries its own `gpt-4.1-nano` row at **$1.50/M train**. So `gpt-4.1-nano-2025-04-14` now resolves to nano, and the 16.7× failure cannot occur. The *lesson* in the Correction is right and worth teaching — never trust a prefix match on a model ID — but the claim about the code is stale. **Read `resolve_price` before repeating either version.**
>
> **And a three-way disagreement on the price table itself.** Three artefacts carry fine-tuning prices and **none of them agrees with the other two.** (a) `code/13_openai_finetune.py:92–100`'s `PRICES` dict — `gpt-4.1-nano` `(1.50, 0.10, 0.40, 0.20, 0.80, 0.10)`, `gpt-4.1-mini` `(3.00, 0.40, 1.60, 0.80, 3.20, 0.40)`, `gpt-4.1` `(None, 2.00, 8.00, 4.00, 16.00, 2.00)`, `gpt-3.5-turbo` with `None` in the three FT slots. (b) CS-18 §13.3's tiering table — `gpt-4.1` FT **~$6 in / ~$1.50 cached / ~$24 out**. (c) CH-18 §7.2's table, which is a table of **base** prices (`gpt-4.1` train **$25.00**, cached-in $1.00) with the fine-tuned nano/mini/flagship rates given in the prose beneath it. **They conflict on the cached-input column** (script `ft_cached` 0.40 for mini and 2.00 for `gpt-4.1`, against CS-18's $0.20 and ~$1.50) **and on whether `gpt-4.1` has a training price at all** (script says `None`; CH-18 says $25.00; CS-18 §13.3 says ~$25.00). The dict is what the code will actually print, so **read the dict for the estimator's behaviour and re-check the live pricing page for a budget** — and if you quote a cached-input figure in an interview, say which of the three you are quoting.

> **Correction (the metric names, and which section actually lists them).** This bank uses the *document's* names, because they are the ones CS-18 can be cited for: **`train_loss`**, **`valid_loss`**, **`full_valid_loss`**, **`train_mean_token_accuracy`**, **`valid_mean_token_accuracy`**. Two traps in citing them. First, **§12.1 and §4.9.2 do not list the same set**: §12.1 is titled *"The four metrics the platform gives you"* and its table has exactly `train_loss`, `valid_loss`, `train_mean_token_accuracy`, `valid_mean_token_accuracy` — **`full_valid_loss` is not there** — while §4.9.2's fuller field list does carry `full_valid_loss`, described as *"Valid loss at end-of-epoch boundaries."* Cite §4.9.2 for the five-name set and §12.1 for the four. Second, the API-surface spellings `training_loss` / `validation_loss` / `training_token_accuracy` / `validation_token_accuracy` **do not appear anywhere in CS-18**. If you are reading a dashboard rather than this module, check which set your console actually prints before quoting a name in an interview — the two families are not interchangeable in a metrics query.

> **Correction (CS-18 §1.3 vs the rest of the document — the training cost of the video's job).** CS-18 §1.3 item 4, at line 151, says the cost model "is wrong by two orders of magnitude … **$0.0024** is a rounding error you can spend 400 times for a dollar." Everywhere else in the document computes this job as **$0.003024**, including §4.6.2's worked arithmetic, §6.5, §15.1's scale table and §19 A3. **$0.003024 is right; $0.0024 is a slip.** The correct computation is `672 billed tokens × 3 epochs = 2,016 × $1.50/M = $0.003024`, and A3's comparison against the content-only counter (`562 × 3 = 1,686 → $0.002529`, a **16.4% underestimate**) is what makes the point §1.3 was reaching for. Use $0.003024 and the 16.4% figure.

> **Correction (CS-18 §15.2's token volumes).** §15.2, at line 3574, describes the retail-triage case as "2M support tickets/month. An 1,800-token prompt … **~40M input tokens and 4M output tokens monthly**." That cannot be right: 2,000,000 tickets × 1,800 input tokens ≈ **3.6 billion** input tokens per month, and at `gpt-4.1-mini`'s `$0.40/M` the stated volume would cost about **$16/month**, not the **$2,080/month** the same row computes. **The cost figures are the ones that reconcile** — `(1800 × $0.40 + 200 × $1.60)/1e6 × 2M = $2,080` — so the "~40M / 4M" string is the error, and the monthly-token figures should be read as **~3.6B input and ~400M output**. Anyone budgeting from the prose rather than from the arithmetic will under-provision by roughly two orders of magnitude.

> **Correction (did the video set `seed`?).** CS-18 Appendix A.4 records the video's job as setting "No `seed`, no `metadata`," while §5's nine-call block at line 1933 and §15.2's config at line 3578 both set **`seed=42`** (and the latter sets `metadata={"data_version": …}` as well). The reconciliation: the *video's* original second-notebook call carried neither, and the **production-shaped** call the module recommends — the one in §5 — adds both deliberately. **Treat `seed` and `metadata` as required in your job-creation call**, and read the Appendix claim as a statement about the video, not as advice.

---

## Cheat Sheet of Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| Minimum examples (API) | **10** | The guard rail, not the useful minimum |
| Useful minimum | **50 – 100** | Below ~50 you are shipping a template |
| Auto-epoch, low regime | `min(25, 100 // N)` | 10 rows → **10 epochs** |
| Auto-epoch, mid regime | **3** | For N in [34, 8333] |
| Auto-epoch, high regime | `max(1, 25000 // N)` | 8,000 rows → 3 epochs |
| Auto `batch_size` rule | **~0.2% of rows**, floor 1 | 10 rows → 1 |
| Per-message overhead | **3 tokens/msg + 3 reply priming** | Plus 1 per `name` |
| Overhead as a share | **+19.6%** (562 → 672) | The content-only counter's error: a **16.4%** cost underestimate |
| `MAX_TOKENS_PER_EXAMPLE` | **16,385** | Stale; truncation is **silent** |
| Tokeniser, `gpt-4`/3.5-era | **`cl100k_base`** | ~100k merges |
| Tokeniser, `gpt-4o`/`gpt-4.1` | **`o200k_base`** | Using cl100k is a systematic error |
| Worked small count | `Hello, how are you?` = **6 tokens** | 19 chars, 4 words |
| Files: `data.jsonl` / `data2.jsonl` | **4,239 B / 4,060 B**, 10 rows each | `data2` has **1 missing assistant** |
| Video job training cost | **$0.003024** | `672 × 3 = 2,016 × $1.50/M`. **Not $0.0024, not $0.75** |
| Video job in the video's numbers | **$0.75** training, **$0.702396** total | Wrong on a quantity that does not exist; **248× too high** |
| `MAX_TOKENS_PER_EXAMPLE`-driven estimate | **$0.737** | Over-counts by **244×** |
| Break-even inequality | **`P_f < P_b/2 − 2O`** | Fine-tuned markup 2× both ways |
| Markup over the base model | **~2×** | On train, input, cached and output |
| Flagship inference share | **95 – 99.9%** | "Inference is the bill" |
| §11.3 losing case | **+$46.40/month** | Train $21.35 one-off; payback `None` |
| §11.3 winning case | **−$4.80/month**, **4.4-month** payback | At `P_f = 180` instead of 500 |
| Self-host crossover | **≈ 460,000 calls/month** | 3B LoRA, 1×24 GB, $0.35/GPU-h × 730 h = **$252/mo** |
| Two-order model-as-judge | every pair judged **both ways** | Cancels position bias |
| No-op gate | `identical_outputs` **< 0.90** over **50 prompts** | Near 1.0 = nothing happened |
| Leak detector | `valid_loss` **below** `train_loss` | A leaked split, not excellence |
| `metadata` fields | `data_version`, `base` | Without them a run is unreconstructable |
| DPO | `n_epochs=1`, **SFT first** | Signature failure: likelihood collapse |
| RFT | **per hour**, needs a **grader** | Auto-verifiable answers only |
| Wind-down tiers | **2026-05-07 / 2026-07-02 / 2027-01-06** | FT models keep serving until base deprecation |
| Distillation exit cost | **~$340** at 120k samples | Plus ~$15 of GPU training |
| Prices (dated snapshots — verify before budgeting) | `gpt-4.1-nano` **$1.50** train / 0.20 in / 0.05 cached / 0.80 out; `gpt-4.1-mini` **~$3.00** / 0.80 / 0.20 / 3.20; `gpt-4.1` **~$25** / ~6 / ~1.50 / ~24 | CS-18 §13.3, Sept 2026 |
| The only durable part | `billed_tokens × epochs × $/1M`; `calls × (P·r_in + O·r_out)` | Every price above churns monthly |

---

## Answers To The Self-Check Questions From CS-18

> CS-18 §19 poses ten questions and answers them under `<details>` at §19 A1–A10. These are the interview-grade restatements: the number, the mechanism, and the trap an interviewer will follow up with.

**1. What does hosted fine-tuning return, and name four things an open-weight fine-tune returns that it does not.**
- **Returns:** an endpoint ID of the shape `ft:<base>:<org>:<suffix>:<hash>`. Nothing downloadable.
- **The four:** you cannot **download**, **merge**, **quantise**, or **self-host** it — and by extension you cannot **move it to another provider** or **use it as a teacher** for a competing model without a terms check (§4.1.1, §9.3).
- **Trap:** answering with quality. §4.1's framing — the difference is the *artefact*, not the performance — is the whole point.

**2. Write a JSONL training row with a system, user and assistant message, and name the seven checks.**
- **Row:** one JSON object per line, top-level `messages` array, each element `{role, content}`.
- **The seven:** `data_type`, `missing_messages_list`, `message_missing_key`, `message_unrecognized_key`, `unrecognized_role`, `missing_content`, `example_missing_assistant_message`.
- **Trap:** not knowing that the four middle checks share one loop and none `continue`, so a single bad message increments up to four counters — the totals count **violations**, not rows.

**3. 672 billed tokens per epoch for 3 epochs at $1.50/M — cost? And what would `count_total_tokens` (562) have given?**
- **Cost:** `672 × 3 = 2,016` tokens → `2,016 ÷ 1e6 × $1.50 = $0.003024`.
- **The content-only counter:** `562 × 3 = 1,686` → `$0.002529` — a **16.4% underestimate**, because the **110 tokens** of per-message overhead (3 per message + 3 for the reply, across 21 messages) are real billed tokens.
- **Trap:** treating 16% as rounding. On a 6M-token job it is a real budget line, and it is the *only* error in the counter that is systematic.

**4. `n_epochs="auto"` on 8,000 examples; and on 10.**
- **8,000:** `8,000 × 3 = 24,000`, between `MIN_TARGET_EXAMPLES` (100) and `MAX_TARGET_EXAMPLES` (25,000) → **3 epochs**.
- **10:** `10 × 3 = 30 < 100` → `min(25, 100 // 10)` = **10 epochs**.
- **The principle:** the policy holds *example observations* at ~100 in the low regime and ~25,000 in the high regime. It is a cost bound.

**5. `batch_size` vs `learning_rate_multiplier`.**
- **`batch_size`** sets examples per optimiser step: how many steps a run contains and how noisy each gradient estimate is.
- **`learning_rate_multiplier`** scales a base rate the provider chooses and does not expose, so its absolute value is meaningless across providers and versions: `1.0` means "use the default," not zero.
- **The interaction:** a larger batch usually wants a larger LR, so changing one without the other desynchronises the run — which is why `"auto"` is right for `batch_size` on nearly every job.

**6. Break-even with `P_b = 900` and `O = 200`. What is the largest winning `P_f`?**
- **`P_f < 900/2 − 2×200 = 450 − 400 = 50` tokens** — essentially impossible for a task that needs instructions, so **fine-tuning on cost grounds loses** in this configuration.
- **Trap:** computing `P_f < 450` and stopping. The `2O` term is the whole content of the inequality.

**7. Three things that could still be wrong despite a fluent answer, and the test for each.**
- **Memorisation** — paraphrase test on reworded training questions.
- **Out-of-scope collapse** — hold out 20 in-scope and 20 out-of-scope prompts; report both rates.
- **A no-op** — `identical_outputs` against the base over 50 prompts; near 1.0 means nothing happened.
- **A fourth:** a **leaked split**, detectable by `valid_loss` coming out *below* `train_loss`.

**8. When is DPO right and when is it wrong, and what must exist first?**
- **Right** when the model can already do the task and you need it to *prefer* one behaviour — tone, refusal boundaries, verbosity, safety. **Wrong** when it cannot produce the behaviour at all.
- **First:** a **supervised fine-tune** — DPO runs on top of an SFT checkpoint, never on a base model.
- **Signature failure:** **likelihood collapse**, where the probabilities of both chosen and rejected fall together. It also wants `n_epochs=1`.

**9. What guarantees valid JSON instead?**
- **Structured outputs** — `response_format={"type":"json_schema","strict":true}` — which makes invalid output structurally unreachable (100%, not 99%) at zero training cost.
- **What remains a fine-tuning problem:** wrong *values inside* a valid shape.
- **Trap:** "fine-tuning gives a more reliable 99%." It gives a weaker guarantee at a higher price.

**10. The base snapshot is deprecated. Describe the migration.**
- Freeze the behaviour by **sampling the existing model over the production prompt distribution** (100k+ completions, `temperature` 0.7–1.0, deduped), fine-tune an **open-weight student** on those samples, validate against a **frozen test set**, then cut over.
- **It must start immediately** — the teacher is only callable while the base model still serves, and generating the samples is a week of wall-clock that cannot be compressed at the end.
- **The one asset that survives is the dataset**, with the eval harness and the prompt/schema contract alongside it. The model does not survive: not the weights, not the ID, not the behaviour.

---

## Cross-References

| Module | Relationship to IQ-18 |
|---|---|
| **CS-18** OpenAI GPT Fine-Tuning | The source case study. §4.6.5 is the break-even derivation; §10's twenty numbered items are the gotchas; §19's ten questions are answered above |
| **CH-18** OpenAI Fine-Tuning Cheat Sheet | The compressed reference: §4 the auto-epoch policy, §5 the validator and the `identical_outputs` gate, §7 the calculator, §8 symptom → fix, §9.4 the two-provider comparison |
| **`code/13_openai_finetune.py`** | The runnable estimator. Its `--help` is the authority on flags (see the Correction above); `resolve_price()` is the authority on the price table |
| **IQ-19** Vertex / Gemini | The other managed API. Same three-layer abstraction, opposite cost shape — hourly idle instead of per-token |
| **CS-19 / CH-19** Vertex / Gemini | The comparison material behind Q37; CH-18 §9.4 is the one-page version |
| **IQ-17** Axolotl | The migration destination: the config-driven trainer the distillation student is trained with. Do not re-derive its questions here |
| **CS-17 / CH-17** Axolotl | The open-weight alternative in full, including what "you get the weights" actually buys you |
| **CS-13** Instruction Fine-Tuning | The same SFT loss, trained on your own GPU with full control. The comparison is CS-18's own cross-reference row — *"Every capability lost in **§4.1** is a capability it has"* — where §4.1 means **CS-18 §4.1, "Hosted vs open-weight: the difference is not performance, it is the artefact"**, not CS-13's §4.1 |
| **CS-03** The Fine-Tuning Framework Landscape (Top 10, 2025) | Positions hosted APIs at the extreme "less code" end of the control/complexity axis |
| **CS-04** Fine-Tuning vs RAG vs Agents | The alternative for *knowledge* problems; compose rather than choose — fine-tune the form, retrieve the facts. (**CS-05 is *RNN-LSTM to Attention***, which has nothing to do with retrieval) |
| **CS-08 / CS-09** Knowledge Distillation | The pipeline that recovers an artefact from an endpoint-only model, and the fix for a sub-50-row dataset: CS-08 the foundations, CS-09 the LLM→SLM case study |
| **CS-23** LoRA & QLoRA: the PEFT deep dive | The technique that makes the open-weight migration target cheap enough to be viable. **Planned, not yet written** — README Part V. Until then, CS-13 §6.8 and CS-16 §6.4 carry the config-level form |
| **CS-25** DPO | Hosted DPO is one flag away; the loss and the theory live there. **Planned, not yet written** — until then CS-14 (*The Alignment Map*) is the written treatment |
| **CS-24** RL Fundamentals & RLHF with PPO | The contrast with RFT's grader-based, per-hour-billed model — the taxonomy error in Q25. **Planned, not yet written** — README Part VI |
| **CS-22** Embedding Models and Embedding FT | Note the title: this is **embeddings**, not evaluation (README Part V; not yet written). The evaluation stack this bank assumes has **no home case study** — CH-18 §5.4 (the `identical_outputs` gate) and `code/common/eval_utils.py` are the nearest written material |
| **CS-15 / IQ-15** LLaMA-Factory | The other config-driven trainer; useful when the rebuild needs a registry rather than an inline dataset |
| **AP-01** Ethics & Compliance (`appendices/AP-01-Ethics-and-Philosophy-of-Alignment.md`) | The distillation terms question, PII handling, and automated-decision disclosures — including the "who gets to encode the preference" question behind §10 #20's ToS row |

---

*End of IQ-18. Companion artifacts: `CS-18-OpenAI-GPT-Fine-Tuning.md` (case study), `CH-18-OpenAI-Fine-Tuning.md` (cheat sheet), `code/13_openai_finetune.py` (the estimator whose `--help` output is the authority on flags). Every price quoted here is a dated snapshot — re-verify against the live pricing page before attaching it to a budget.*
