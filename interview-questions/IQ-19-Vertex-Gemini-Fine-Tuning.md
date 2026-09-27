# IQ-19 — Interview Questions: Vertex AI / Gemini Fine-Tuning (The Metered Rental)

| Field | Value |
|---|---|
| **Module** | Managed / Hosted Fine-Tuning — the branch where the *deployment* is the product |
| **Pairs with** | **CS-19** (case study), **CH-19** (cheat sheet), `code/14_vertex_gemini_finetune.py` |
| **Total questions** | **52** (12 L1 + 15 L2 + 12 L3 + 5 L4 + 8 L5) + 25 rapid-fire + 4 whiteboard tasks + 10 CS-19 §19 self-check answers |
| **Levels covered** | L1 screening → L5 incident response |
| **Source material** | CS-19 §0–§20 + Appendix A/B, CH-19 §1–§13, `code/14_vertex_gemini_finetune.py` (flags verified by running `--help`) |
| **Also contains** | Rapid-fire true/false table, 4 coding/whiteboard tasks with grading notes, numbers to memorize, the ten CS-19 §19 self-check answers |
| **Time to work through** | ~5 h at interview pace; ~60 min for a revision pass (L1 + L2 + rapid fire + numbers) |
| **Differentiate from** | **IQ-18** (OpenAI hosted FT — same shape of service, opposite cost shape) and **IQ-17** (Axolotl — what "you get the weights" actually means). This bank assumes CH-18 §9.4's two-provider table and does not re-ask IQ-18's questions |

**Ground truth used throughout:** the module's single load-bearing fact, from CS-19 §18 #1 — **"Vertex SFT is a rental: no weights, no export, no exit" [3:35]** — and the cost identity that follows from it, CS-19 §18 #2: **training is free, deployment is expensive**, at a ratio of roughly **500,000 : 1** on the module's own demo. Every flag named below was confirmed against `python code/14_vertex_gemini_finetune.py --help`. **Every price and model id is a dated snapshot copied out of CS-19 or CH-19, and CS-19 §11 opens with a standing warning worth repeating verbatim: "Vertex pricing changed at least four times between the video and the publication of this module. Re-derive every number against the live pricing page for your region before you commit budget."**

---

## How To Use This File

- **L1 — Fundamentals & Vocabulary (12).** The resource model: a tuned `Model` and a deployed `Endpoint` are two things with two bills. A candidate who treats "trained" and "deployed" as one state has missed the module.
- **L2 — Applied & Implementation (14).** The Vertex JSONL schema, the implicit split, the four hyperparameters the video never names, the setup that is harder than the code.
- **L3 — Advanced, Internals & Theory (12).** The four cost lines, the three deployment regimes, why the hourly charge makes this a *traffic* question, and why the artefact that never crosses the boundary changes what "migration" means.
- **L4 — System Design & Scenario (5).** Requirements → constraints → design → trade-offs → failure modes. 10–15 minutes each, out loud.
- **L5 — Debugging & Incident Response (8).** A symptom and a clock. Sequence first, then name the look-alike you are ruling out.

**The meta-rule:** an answer of "it depends" is a failure unless it immediately says what it depends on and then picks a default. Always pick the default.

**The three answers that get people hired in this module:** (1) *the artefact and the deployment are two resources* — `tuned_model.model` and `tuned_model.endpoint`, two bills, two lifecycles, and confusing them is the module's most expensive mistake; (2) *the cost comparison is never training vs training* — it is **monthly endpoint cost vs monthly self-hosted GPU cost**, and that is a traffic question, not an ML question (CS-19 §11.4); (3) *you must pass an explicit validation file* — Vertex splits for you, and on 10 rows that means every eval number you read is a statistic computed on **2 examples** (CS-19 §4.5).

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. What are the two resources a successful tuning job creates, and which one bills?**

- **Answer:** A tuned **`Model`** resource (`.../models/<id>@1`) and, by default, a deployed **`Endpoint`** (`.../endpoints/<id>`). **The Endpoint bills** — per hour, whether or not it receives traffic. The Model itself carries no published hourly charge. CS-19 §19 A3 gives exactly this answer, and §18 #3 states the rule: **"`tuned_model.model` (the artefact) and `tuned_model.endpoint` (the deployment) are two resources with two bills and two lifecycles. Confusing them is the module's most expensive mistake."**
- **Why asked:** It is the module's whole content in one question. Every cost, lifecycle, rollback and teardown question downstream is a corollary.
- **Trap:** "Training creates a model." Training creates *two* resources, and the second one starts a meter the moment it exists.

**Q2. Can you download a fine-tuned Gemini model? Give the two architectural consequences.**

- **Answer:** **No.** The base weights are closed-source and the tuned artefact is only ever exposed as a Vertex resource (CS-19 §19 A1, [3:35]). Two consequences: **(a) no exit strategy** — migrating means re-doing the fine-tune elsewhere, so the fine-tune is not a portable asset; **(b) no downstream optimisation** — you cannot quantise, merge, batch, or run it outside a Vertex endpoint, so **your serving cost is permanently Google's price**. §18 #1 is the one-liner: *"Vertex SFT is a rental: no weights, no export, no exit."*
- **Why asked:** It is the same fact as IQ-18's Q1/Q2 arriving through a different product, and the follow-up is always the same: *"so what does that cost you?"* The answer is the *serving* economics, not the training bill.
- **Trap:** Answering "you can't, but it doesn't matter because the API is cheap." The API is not cheap — the endpoint is an hourly charge and it dominates.

**Q3. Write the Vertex JSONL schema from memory on the whiteboard, and name the three fields that differ from OpenAI's.**

- **Answer:** The row looks like this (CS-19 §19 A2, §4.2):

```json
{"systemInstruction": {"role": "system", "parts": [{"text": "..."}]},
 "contents": [{"role": "user",  "parts": [{"text": "..."}]},
              {"role": "model", "parts": [{"text": "..."}]}]}
```

  The three differences: **the system prompt is hoisted** out of the message list into a top-level `systemInstruction`; **the assistant role is `"model"`, not `"assistant"`**; and **the text is nested in a list of `parts` objects** rather than being a flat `content` string. CS-19 §4.2 adds a fourth that bites just as often: the top-level conversation key is `contents`, not `messages`, "and there is no aliasing."
- **Why asked:** It is the single most common cause of a first-attempt failure, and CS-19 §4.2 says so: this is "the section that breaks people's first attempt."
- **Trap:** Getting the roles right and the *nesting* wrong. §4.2's difference #3 warns that the failure mode of `parts` done wrong is not a clean error — it is "an empty turn that trains the model to emit nothing."

**Q4. Name the five lifecycle states and say which transition nobody owns.**

- **Answer:** **DRAFT → TRAINED → SHADOW → DEPLOYED → UNDEPLOYED** (CS-19 §16.1, §18 #6). DRAFT is a frozen, split, validated dataset; TRAINED is `Model@N` living in GCS; SHADOW is deployed but not routed; DEPLOYED is serving traffic and billing per hour; UNDEPLOYED is `Model@N` retained with no endpoint. The transition nobody owns is **DEPLOYED → UNDEPLOYED**, "and it is the expensive one." The correct default state for a tuned model is **TRAINED, not DEPLOYED** (CS-19 §17 #20).
- **Why asked:** It converts "we deployed it" from a sentence into a state machine with entry criteria and owners — and it surfaces the one transition that generates the surprise invoice.
- **Trap:** Treating SHADOW as optional. §16.1 gives it 1–14 days with "passing regression + refusal suites" as its entry criterion; without it, DEPLOYED *is* your shadow.

**Q5. What are the four hyperparameters, and which one is a direct multiplier on your bill?**

- **Answer:** `epochCount`, `learningRateMultiplier`, `batchSize`, `adapterSize` (CS-19 §7.0, §7.1, §18 #9). The bill multiplier is **`epochCount`**: `billable_tokens = total_tokens × epochCount` (§7.2). CS-19 §7.0's headline finding is that **"the video names zero hyperparameters. Not one."** — the `CreateTuningJobConfig` it builds contains exactly one field, `tuned_model_display_name`, and "the run proceeds on Vertex's defaults."
- **Why asked:** It tests whether the candidate knows that a *defaulted* hyperparameter is still a decision — and whether they can name the knobs the tutorial never mentioned.
- **Trap:** "`batchSize` changes the bill." §7.3 is explicit: **"`batchSize` does not change your token bill."** Every token is read once per epoch regardless.

**Q6. `learningRateMultiplier: 1.0` — what is it a multiplier *of*, and why can't you set an absolute LR?**

- **Answer:** It multiplies **Vertex's internal default learning rate, which you cannot read** (CS-19 §19 A7). You can only express "more" or "less" relative to whatever Google chose, which means an LR value you know from another framework (say `2e-4`) is **meaningless** here, and a recipe from a paper cannot be reproduced exactly. The knob is **ordinal, not cardinal**.
- **Why asked:** Same trap as IQ-18's Q6, arriving through a different API — an interviewer uses the pair to see whether the candidate learned the *principle* or the *field name*.
- **Trap:** "Vertex's default is 1.0, so 1.0 means no learning." It means "use the default," not "zero." §7.1 lists the safe practitioner range as **0.5–2.0**, with "never touch it on a first run."

**Q7. What is `adapterSize`, what is its default, and what is the safe setting?**

- **Answer:** It is the **LoRA rank of the tuning adapter** — it caps *how much the model can change*, and it determines the artefact you deploy (CH-19 §4.1). CS-19 §7.1 records the Vertex default as **`ADAPTER_SIZE_ONE` (rank 1)** for several families, with a documented set of **1 / 4 / 8 / 16 / 32**. The practitioner default is **4 or 8**: rank 1 for pure style/format, 4 as the safe default, 8 for "a clear behaviour change with a few thousand examples," 16 for a substantial change, and 32 which "overfits unless the dataset is large" (CH-19 §4.1).
- **Why asked:** It is *the* knob people ignore, and the trap has a specific shape: a candidate who never sets it gets rank 1 and reports that "the fine-tune barely changed anything" — when the real cause was a capacity ceiling they did not know existed.
- **Trap:** Reading rank 1 as "conservative and therefore safe." CH-19 §4.1's risk column: rank 1 "may not have capacity to learn the task." A model that learns nothing is not a safe outcome.

**Q8. Where does the training data live, and what is the prerequisite people miss?**

- **Answer:** In **your GCS bucket, in your project, in a region you chose** (CS-19 §17's first row). The prerequisite people miss is that **the bucket and the tuning job must be region-aligned** (§10.5), and the identity granting access matters: CH-19 §11's error table has `403 ... does not have storage.objects.get access` → "Service account lacks bucket read — §5.2 — use `service-NUMBER`". CH-19 §5.2's point is that the principal you grant is the **`service-PROJECT_NUMBER`** account, not something keyed on `PROJECT_ID`.
- **Why asked:** "The setup is harder than the code" is CS-19 §18 #10 — the instructor says so himself at [19:53]. An interviewer asks this to find out whether the candidate has actually run a job, because the code is twelve lines and the setup is an afternoon.
- **Trap:** Using a public or pre-existing bucket. The data-placement question is a compliance question (§17's table), not just an IAM question.

**Q9. What does Vertex do about a validation split if you don't pass one?**

- **Answer:** It **performs a default split of the supplied dataset** — commonly reported as **80/20** for Gemini SFT (CS-19 §4.5). You did not choose the ratio, cannot see which examples were held out, and cannot stratify it. On a 10-row dataset that means **2 validation examples**, so "every 'evaluation' number in that job is a statistic computed on 2 samples." The production rule is to **always pass an explicit, deterministic, stratified, seeded split** via `validationDatasetUri` (SDK: `validation_dataset=`), sized at **≥ 100 examples and ≥ 10% of the dataset, whichever is larger**.
- **Why asked:** It is the module's version of IQ-17's `val_set_size: 0` and IQ-18's `validation_file=None` — the third arrival of the same defect. The follow-up is *"what does the default split cost you at 100,000 rows?"*: it wastes 20,000 rows of your best data and, because the split is not reproducible across jobs, you can never recover a stratified one (§4.5).
- **Trap:** "The metrics say 'evaluation', so there must be a proper holdout." There is a holdout; it is just unseen, unstratified and — at small N — statistically meaningless.

**Q10. What is `JOB_STATE_SUCCEEDED` the *start* of?**

- **Answer:** Your monthly cost. CS-19 §17 #5 is the cleanest statement: **"`JOB_STATE_SUCCEEDED` is the moment your monthly cost starts, not the moment it ends."** The job succeeding *creates new billable resources*, and the default behaviour is to deploy the endpoint automatically. §17 #5's framing — "Once the job succeeds, I'm done" — is listed as a misconception for exactly this reason.
- **Why asked:** It is the module's central inversion. Everyone arrives expecting training to be the expensive part; here training is a rounding error and the thing that starts on success is the entire bill.
- **Trap:** "But I only pay per token." §17 #19: "No. You pay per hour the deployment exists, plus tokens. At hobby traffic the hourly charge is ~100% of the cost (§11.3)."

**Q11. Name three things that differ between Vertex and AI Studio, and say why the confusion is so easy.**

- **Answer:** Different **auth**, different **quotas**, different **model IDs**, and — critically — **different tuning support**: AI Studio lost fine-tuning (CS-19 §17 #2, [7:18]). The confusion is easy because **they share a Python SDK**, which is exactly what makes it hard to notice you are on the wrong product.
- **Why asked:** CS-19 §17 #2 says "The whole reason this module exists is that AI Studio lost fine-tuning." A candidate who does not know the two are distinct products will mis-file every piece of documentation they read.
- **Trap:** Assuming a model available in AI Studio is tunable on Vertex, or vice versa. The supported tuning list is per-product and changes quarterly (§4.1).

**Q12. Two clients, two API versions. Which for what, and what error do you get if you get it wrong?**

- **Answer:** **`v1beta1` for tuning, `v1` for inference and job retrieval** (CS-19 §4.3, §10.6, §18 #11). Tuning ships on `v1beta1` first; if you pin `v1` and call `client.tunings.tune(...)` you may get **`AttributeError: 'Tunings' object has no attribute 'tune'`** or a 404 on the endpoint, depending on how far the feature has graduated. The graduation is **one-directional**: a feature moves `v1beta1 → v1` and never back, so code that pinned `v1beta1` "keeps working for a while and then breaks when the beta surface is retired." The discipline: **one module-level constant per client, a comment naming the feature that forced the version, and a CI smoke test that runs both.**
- **Why asked:** It is a general lesson about consuming a moving API dressed as a Vertex gotcha, and the fix the source prescribes (name the reason, test both) is what an interviewer is listening for.
- **Trap:** "Just use `v1beta1` everywhere to be safe." §10.6 warns of the opposite failure: "beta-only response shapes in your parsing code."

---

## Level 2 — Applied & Implementation

**Q13. Spot the four bugs in this Vertex JSONL row.**

  `{"messages": [{"role":"system","content":"You are a support agent."}, {"role":"user","content":"Refund policy?"}, {"role":"assistant","content":"14 days."}]}`

- **Answer:** Four, and each has its own error. (1) **`messages`** should be **`contents`** — "there is no aliasing" (§4.2 difference #4). (2) The **system turn is inside the list**; Vertex **hoists** it to a sibling top-level `systemInstruction` (§4.2 difference #2) — and a leading `system` turn in `contents` is "either rejected or silently trained on as a user turn." (3) **`assistant`** must be **`model`** — `400 ... role must be one of ['system','user','model']` (CH-19 §11). (4) **`content`** must be a **list of `parts`** with the text at `parts[0].text` — `KeyError: 'text'`, or "an empty turn that trains the model to emit nothing" (§4.2 difference #3).
- **Why asked:** It is the module's canonical spot-the-bug, and it is diagnosable from the code alone. Note that the *fifth* difference — Vertex being "stricter about unknown fields than OpenAI" — means a stray `tools` or `weight` key is its own failure.
- **Trap:** Fixing the role and the key and calling it done. Each of the four produces a *different* symptom, and only two of them produce a clean error message.

**Q14. What single design change makes the schema problem go away permanently?**

- **Answer:** Write a **schema adapter**, not two datasets. CS-19 §13.2: one canonical internal representation — "a list of `(role, text)` tuples" — plus a renderer per target platform, "roughly 60 lines of Python," which "eliminates an entire category of 'the dataset was regenerated and someone forgot the Vertex renderer' incident." The second benefit is the one people miss: it lets you **keep your split and your hashes stable across platforms**, "which is what makes a genuine cross-platform quality comparison possible at all."
- **Why asked:** It tests whether the candidate reaches for an abstraction or a copy. The paired question "so can you compare quality across OpenAI and Vertex?" is answered *yes* only if the hashes are stable — otherwise you are comparing two different datasets and attributing the delta to the platform.
- **Trap:** Solving it with a one-off `sed` in the upload script. That works until the dataset is regenerated by someone else.

**Q15. Your dataset's system instruction is byte-identical on all 10 rows, and it is 26 of the ~56-token average row. So what?**

- **Answer:** **Roughly 46% of your tuning bill is a constant that carries zero gradient information about the task** — `p(response | instruction)` has no variance in the conditioning variable (CS-19 §4.2's defect analysis, citing CS-13 §1.3). And the cost point is general, not specific to 10 rows: §11.2's third observation — if your system instruction is 2,000 tokens and your examples are 300, **you are paying 87% of your tuning bill to re-read the same paragraph.**
- **Why asked:** It is a "minute aspect" that only falls out of reading the arithmetic, and it inverts a common intuition: people assume a big shared preamble is free because it is not new information.
- **Trap:** "So I should delete the system instruction." Then you lose the behaviour. The right move is to **shorten** it to the shortest text that still carries the behaviour, and to spend the savings on more rows — because at these training prices, rows are nearly free.

**Q16. Row 9 of the companion dataset is a refusal. Why does that make it the most valuable row in the file?**

- **Answer:** Because it is a **hand-authored negative**, and in a 10-row dataset it is **10% of the signal** (CS-19 §4.2). `"Can you recommend a good laptop for work?"` → `"I'm here to assist only with smartphone-related questions."` The source's own guidance is quantitative: **in a 10,000-row dataset you would want roughly 5–10% of rows to be exactly this shape.**
- **Why asked:** It is the concrete form of "fine-tuning teaches form, including refusal boundaries" — and it is the row that makes CS-19 §12.4's prompt-mutation tests meaningful, because test 5 ("ask an out-of-scope question — the laptop question, row 9") exists only because a negative was authored.
- **Trap:** Building a dataset of only positive examples and expecting a refusal boundary to emerge. §12.4's verdict on the failure: "**If the model passes 1–4 and fails 5, you have trained a template-follower.**"

**Q17. `epochCount` is 1 by default. You want 3. Walk the decision.**

- **Answer:** Three facts, in the order CS-19 §7.2 gives them. (1) **`epochCount` multiplies the bill linearly** — `billable_tokens = total_tokens × epochCount` — and if Vertex's defaults ever change from 1 to 3 your bill triples with no code change on your side, so **recompute from the job's own metric output** (`totalBillableTokenCount`, if exposed) rather than from your pre-flight estimate. (2) **The useful SFT range is 1–3**, and the video's own demo would be *harmed* by raising it — 10 rows with a constant system instruction means epoch 10 has seen each row ten times, "which is a lookup table, not a behaviour." (3) **You cannot early-stop** — there is no `earlyStoppingCallback`; the job runs to completion and you pay for every epoch. So the practical policy is: **run 3 epochs, evaluate, and if it overfits, run 1** — because training tokens are cheap and engineering time is not.
- **Why asked:** It tests whether the candidate reasons about *relative* cost. The naive answer ("start at 1 to save money") is wrong here for a reason that only appears when you compare the token bill to the wall-clock.
- **Trap:** Reasoning about epoch count as a cost lever at all. §11.2's second observation: the cost argument against more epochs "appears only above ~10⁹ tokens."

**Q18. `batchSize` on 1,000 rows × 3 epochs: what changes between 4 and 16?**

- **Answer:** The **step count** — `steps = ceil(N_examples × epochs / batchSize)` — so a batch of 4 gives **750 steps** and a batch of 16 gives **188** (CS-19 §7.3). What *does not* change: **your token bill.** Every token is still read once per epoch. What else moves: **gradient noise** (smaller batches → noisier, sometimes better generalisation, always less stable), **memory** (larger batches and larger adapters compete for the same memory — "if a job OOMs during tuning, `batchSize` is the first thing to lower"), and the **interaction with LR** (raise `batchSize` 4× and gradient variance drops, so with a fixed multiplier you may want to raise the multiplier slightly — "in practice Vertex's defaults assume its default batch size; change one or the other, not both").
- **Why asked:** It is the IQ-18 Q19 question in a place where the answer is *different* — there `batch_size` was echoed and not honoured; here it is honoured and simply does not touch the bill. An interviewer wants to see you not over-generalise between the two platforms.
- **Trap:** "Halving steps halves training cost." It halves the number of optimizer updates, not the number of tokens read.

**Q19. A 2×2 experiment over `epochCount` and `adapterSize`. How long, and what does it cost?**

- **Answer:** **Four jobs ≈ 80 minutes of wall-clock and pocket change in tokens** (CS-19 §7.3). The wall-clock is dominated by the **fixed 15–20 minute overhead** [50:56], not by the data. The prescribed discipline is **"change exactly one thing per job,"** and the source's verdict is blunt: this experiment is cheap, well-designed, "and **not running it is the most common reason a team ships a tuned model that a rank sweep would have improved.**"
- **Why asked:** It tests whether the candidate has a model of *iteration cost* on this platform. The answer — 80 minutes and a few cents — reframes the whole project from "expensive experiment" to "afternoon."
- **Trap:** "The 15–20 minute runtime means the dataset was small." §17 #16: "It means the *overhead* is fixed. A 564-token job and a 5,000-row job are both dominated by provisioning at the low end."

**Q20. What does "full fine-tuning" mean on Vertex, and what should you do with that claim?**

- **Answer:** Mechanically it means all base weights are updated, versus PEFT's `ΔW = BA` added to a fixed set of projection matrices (CS-19 §7.5). But **you cannot verify it.** §17 #17 and §7.5: you cannot inspect the delta between base and tuned weights, you cannot measure how many parameters moved, and the vendor "has no incentive to update the full weight set if an adapter would do — it is strictly more expensive for them." So **"treat it as a pricing tier and evaluate the output,"** not as a technical claim. And note the export column: **PEFT, full tuning and distillation are all "No. Not exportable."**
- **Why asked:** It is the sharpest test of whether someone reasons about *evidence* rather than *labels* on a closed platform. The follow-up is always the same: "so how do you know?" — and the honest answer is "by the output, over a held-out set, compared against the base."
- **Trap:** "Full tuning is worth the extra cost because more weights move." Unverifiable claims do not get more credible by being expensive.

**Q21. When should you use an adapter (PEFT) and when should you not fine-tune at all here?**

- **Answer:** CS-19 §7.5's decision rule, which is deliberately shaped as a two-way branch with a third exit:

  - **Changing *how* the model says things** → PEFT at rank 4–8. Cheaper and sufficient.
  - **Changing *what* it says, broadly** → full tuning, but **re-read CS-13 §4.1 first: "SFT injects facts weakly and expensively."** RAG or continued pretraining is usually the right answer.
  - **Getting a portable artefact** → **neither.** "Vertex cannot give you one. Go to CS-16 / CS-17 / CS-23."

- **Why asked:** The third branch is the one that matters. A candidate who only knows the first two will recommend Vertex for a project whose actual requirement is ownership.
- **Trap:** "Full tuning is the way to inject domain facts." §17 #3: the fine-tuned model "will know my private business data" only weakly, and only if the behaviour is formatting rather than facts. The demo question ("why do phones have a one-year warranty?") is **a fact Gemini already knew** — "nothing in the video's demo distinguishes a tuned model from a prompted one."

**Q22. What is the prompt-mutation test, and what does failing one axis but passing another mean?**

- **Answer:** Prompt-mutation testing is CS-19 §12.4's **overfitting detector**: mutate the prompt along axes your training data never varied — a different language, the system instruction dropped, leading/trailing whitespace, three rephrasings, an out-of-scope question, a longer answer than any training example, two questions in one turn — and check the behaviour on each. The headline verdict: **"If the model passes 1–4 and fails 5, you have trained a template-follower. That is a real product defect, and it is invisible in every metric Vertex shows you."** The single most informative axis is **dropping the system instruction**: if the house style survives, "the style is in the weights"; if it collapses, "the style was never in the weights."
- **Why asked:** It is the module's cheapest real evaluation, it requires no held-out labels, and it detects the failure the dashboard structurally cannot. The "drop the system instruction" axis is the discriminator between a candidate who has read the list and one who has thought about what each axis *proves*.
- **Trap:** Running all seven and reporting a pass/fail count. The axes are diagnostics with different meanings; the interesting output is *which* one fails.

**Q23. What are the automatic metrics Vertex gives you, and what do they miss?**

- **Answer:** The dashboard shows **evaluation fraction of correct next-step prediction, evaluation number of predictions, evaluation total loss, and training loss** (CS-19 §4.5's transcript quote, [51:36]). CS-19 §18 #7's summary: **"The automatic metrics are log loss and teacher-forced token accuracy. Neither measures task success."** What they miss: correctness, tone, refusal behaviour, format compliance, latency, and cost — the same "*(nothing)*" row as CS-18 §12.1. The build is yours: **`eval_ab.py` and a comparison against the *base* model.**
- **Why asked:** It is the direct parallel to IQ-18's evaluation questions, and the source's own framing — "the four numbers are not an evaluation" (§12.2, and the Correction at CS-19 line 1661) — is what an interviewer wants paraphrased.
- **Trap:** Reading the *fraction of correct next-step prediction* as accuracy. It is teacher-forced: the model is handed the right prefix at every step, so it measures token prediction under ideal conditions, exactly as CS-18 §12.1's `train_mean_token_accuracy` does.

**Q24. A colleague reports `evaluation total loss: 4.2` on a 10-row job and says it looks good. Respond.**

- **Answer:** Two problems, in order. (1) **The number is computed on 2 examples** — Vertex's default split of a 10-row dataset (CS-19 §4.5, §19 A8's "unseen, unstratified and, at 10 rows, two examples"). It "moves by large amounts when one example changes," so you will read a good number and believe it. (2) **Loss measures the wrong thing anyway** — eval loss is next-token cross-entropy on a held-out slice of *your own data*; it measures how well the model predicts your distribution, not how well it does your task. §17 #7: "A model can have excellent eval loss and be worse in production (Case C, §15.3)." Then ask the calibration question from §19 A8: **what does uniform entropy look like for this vocabulary?** `ln(V)` — for a 100k vocabulary that is **≈ 11.51** (CS-19 §19 A8). A loss of 4.2 is *not* uniform, so something was learned — but on two examples, "something" is not a finding.
- **Why asked:** It combines the dataset-size defect, the metric-meaning defect, and the one piece of arithmetic that lets you sanity-check a loss number without any other context. The `ln(V)` bound is the module's most transferable single fact.
- **Trap:** Not having the `ln(V)` baseline. Without it, "4.2" is a number with no scale — and the candidate cannot tell "learned a lot" from "learned nothing."

**Q25. What does `client.tunings.list()[1]` return, and why is it dangerous?**

- **Answer:** It returns **the second-most-recent job in the project**, "which may be a colleague's job, a retried job, or a test job" (CS-19 §19 A4, §10.8). Dangerous because you would then read *its* `tuned_model.endpoint` and **deploy or query the wrong model while believing it is yours.** The fix: **persist the job name returned by `tune()`**, and if you must re-find one, "match on `displayName` plus a creation-time window, not on index." §18 #11's phrasing is the one to remember: the list is newest-first *for your project's visible jobs* — which includes everyone's.
- **Why asked:** It is the same class of bug as IQ-18's "guessing the newest file trains the wrong data," and it is the single most likely way to bill for a colleague's endpoint.
- **Trap:** "But I only have one job." Everyone says that until the first retry.

**Q26. What must you have done *before* the first tuning job runs? Give the four controls.**

- **Answer:** CS-19 §11.5's budget controls, plus §16.1's lifecycle ownership. (1) **A budget with alerts** — created in the console (Billing → Budgets & alerts), **$50 or the local equivalent, with alerts at 50% / 90% / 100%**, optionally scoped to the Vertex AI service. (2) **Verify the budget is actually scoped to the right project** — `gcloud billing budgets list --billing-account=BILLING_ACCOUNT_ID`. (3) **A monthly cost report you actually read** — enable the BigQuery billing export and query by service and SKU, where "the SKU that matters is the **endpoint/online-prediction node-hour line**, not the tuning line." (4) **A written plan to undeploy** — CH-19 §12's Check 4, with the source's own gloss: "Check 4 is not a joke. The most common way this module produces a surprise invoice is a deployed endpoint someone forgot about for three months." Do the arithmetic with the current rate rather than quoting a remembered total: three months is `3 × 730 h × $2.00 =` **$4,380** on the 2.5-generation endpoint rate — and the discipline is to recompute it, because the older revision of CH-19 printed **$6,570** from the 1.5-generation $3.00/hour rate.
- **Why asked:** It is the operational counterpart to the cost questions, and the SKU detail ("the node-hour line, not the tuning line") is the specific thing that separates someone who has written the query from someone who has read about it.
- **Trap:** Relying on the alert alone. An alert tells you after the fact; the sweeper is what stops the meter.

**Q26b. Is there a max-sequence-length knob on Vertex, and what happens to an over-long row?**

- **Answer:** **There is no max-sequence-length knob, and over-long examples are truncated silently.** That is the handbook script's own docstring, and the consequence it names is precise: **"A run can succeed, be billed in full, and have taught the model a torn assistant turn."** Note the three-part damage — the job succeeds, you pay for every token, and the *label* you trained against is a truncated string. The script's mitigation is a two-stage gate: **`--validate` reports the per-example length distribution**, and **`--upload` refuses while any row is over `--max-example-tokens`**. Both flags exist on the CLI (`--max-example-tokens` is in `--help`), and the distribution rather than the mean is the point — CS-18 §4.5.3 makes the same argument for the same reason.
- **Why asked:** It is the direct Vertex analogue of CS-18's stale `MAX_TOKENS_PER_EXAMPLE = 16385` silent truncation, arriving through a *missing* knob rather than a stale constant. An interviewer uses the pair to see whether the candidate understands that "no error" and "no problem" are different claims. **The number to attach to it is the script's own `MAX_EXAMPLE_TOKENS_GUESS = 8192`** — named a *guess* in the source, which is the honest label, and therefore a value you set deliberately and re-derive from your own length histogram rather than inherit. CS-19 §6.5.1 is the source's dedicated treatment of the gap: "**the job succeeds, you are billed for the tokens, and the *target* turn may have been cut off**," and its prescribed sequence is **filter first, then split, then report both counts** — because filtering *after* the split changes the holdout you thought you had.
- **Trap:** "The job succeeded, so the data was fine." A truncated row produces no warning, no error and no non-zero exit code — and it is the one defect that makes the model *confidently* wrong, because the truncated target is what you asked it to predict.

---

## Level 3 — Advanced, Internals & Theory

**Q27. Name the four cost lines and say which one dominates.**

- **Answer:** CS-19 §11.1's table: (1) **Tuning (training tokens)** — `$/1M tokens read × epochs`, **trivial** at `$0.00282` on the demo, $10–$300 per job at production scale. (2) **Endpoint deployment** — **`$/hour the deployment exists`**, **dominant** at **$700–$3,700/month**. (3) **Tuned-model inference** — `$/1M input + $/1M output at a premium over base`, small — `$3–$12/month` at the §4.6 volume, "still an order of magnitude below line 2." (4) **Artefact storage** — **no published hourly charge identified; verify** — negligible. §11.1's verdict: **"The ratio between line 1 and line 2 at demo scale is roughly 500,000 : 1. Every budgeting mistake in this module is a failure to notice that ratio."**
- **Why asked:** It is the module's cost model in one table, and line 4 is the one that tests intellectual honesty — the correct answer is "unpublished, verify," not a number.
- **Trap:** Quoting the 500,000:1 ratio as *the* ratio. It is a line-1-vs-line-2 ratio at demo scale. CH-19 §7.1's **23×** is a *different* ratio — fixed endpoint vs per-token usage at 100k calls. Both are correct; conflating them is not.

**Q28. Where does the "~500,000:1" and the "23×" come from, and why do they differ?**

- **Answer:** They have **different denominators**. CS-19's **~500,000 : 1** compares line 1 to line 2 — the 564-token training bill (`$0.00282`) against a month of idle endpoint (CS-19 §11.1 states it against a `$1,459` endpoint; §18 #2 rounds it to "~$1,400/month"). CH-19 §7.1's **23×** compares the **per-token usage** at 100,000 calls (`$63.04`) against the fixed endpoint (`$1,460.00`) — i.e. line 3 vs line 2, not line 1 vs line 2. Same platform, same point, two different pairs of terms. The one to quote in an interview is whichever pair you are actually comparing, and the honest answer is to **say which two lines you mean.** Note how the pair moved together when the rate changed: at the old 1.5-generation `$3.00/hour`, the same table read `$7.63` usage, a `$2,190` endpoint and a **287×** ratio — so if you memorise a ratio without its two terms you cannot tell whether the platform changed or you did.
- **Why asked:** It is a "do you read the units" question in disguise, and it is the kind of minute distinction that separates a candidate who has internalised the cost model from one who has memorised two numbers.
- **Trap:** Averaging them, or picking the larger one because it is more dramatic. A ratio without its two terms is not a fact.

**Q29. At what traffic does the dedicated endpoint stop being wasteful?**

- **Answer:** CS-19 §11.4's three regimes: **below ~500k requests/month, do not deploy a dedicated endpoint** — "a dedicated endpoint costs 25–100× too much here"; use the base model API, the managed tuned path if it avoids a dedicated deployment, or self-host scale-to-zero. **~500k–20M requests/month, the regimes are comparable on price** — "decide on *control*: do you need the weights, the observability, the ability to quantise?" **Above ~20M requests/month, self-host if the task fits an open model** — "at this volume you are paying for multiple replicas either way, and with open weights you control quantisation, batching and scheduling — which is where the 3–10× cost wins live."
- **Why asked:** It is the crossover question, and CS-19 §11.4's stated decision rule is the answer an interviewer is grading against: *"Managed fine-tuning wins on time-to-first-model and on organisational fit. It loses on unit economics the moment the endpoint is deployed, because you are paying for reserved capacity you may not use. The correct comparison is never 'training cost vs training cost'; it is 'monthly endpoint cost vs monthly self-hosted GPU cost', and that comparison is a traffic question, not a machine-learning question."*
- **Trap:** Comparing against OpenAI's per-token fine-tuned price. The competitor here is **a reserved GPU**, not another token meter — which is the exact inverse of IQ-18's break-even, where the competitor *was* a per-token base model.

**Q30. Which serving path has a "premium," how large is it, and how does it compare with OpenAI's?**

- **Answer:** The **managed tuned path** charges a per-token premium over the base rate; CS-19 §11.3 puts it at **commonly 2–10× the base rate**, with an explicit "**Yes — the multiplier has moved a lot**" in the verify column. The **dedicated-endpoint path** is different in kind: there the **hourly charge dominates at low traffic**, so the per-token premium is not the story. Contrast with OpenAI, where CS-18 §4.6.5 records a clean **2× on both input and output** for the `gpt-4.1` family — a single number you can put in an inequality.
- **Why asked:** It is the cross-module comparison the source sets up, and the practical consequence is large: **`P_f < P_b/2 − 2O` is derivable only because OpenAI's markup is a clean 2×.** On a 2–10× premium you cannot do the algebra in your head, which is *why* Vertex's cost question collapses to the hourly charge instead.
- **Trap:** Porting IQ-18's break-even inequality to Vertex. The algebra assumes a fixed markup ratio and a 4:1 output:input price ratio; §11.3 explicitly declines to give a stable multiplier.

**Q31. What is the real cost driver inside the *training* line, and what does it imply about dataset design?**

- **Answer:** The **system instruction, repeated in every row**. CS-19 §11.2's third observation: **"the token count includes the system instruction in every row"** — in the companion dataset that is **~46% of the bill** (§4.2). The generalisation: "if your system instruction is 2,000 tokens and your examples are 300, you are paying **87%** of your tuning bill to re-read the same paragraph." It implies the system instruction should be as short as it can be while still carrying the behaviour, and that the savings should be spent on **rows**, which are nearly free.
- **Why asked:** It is the *only* place in the module where training-cost optimisation has a real lever, and it is not the lever people reach for (they reach for `epochCount`, which §11.2 says only matters above ~10⁹ tokens). An interviewer uses it to see whether the candidate optimises the term that exists.
- **Trap:** Trimming epochs to save money. §11.2's second observation: epoch count "should be chosen for quality, not for cost, at every realistic dataset size."

**Q32. Why might a 564-token job cost more than the arithmetic predicts?**

- **Answer:** Because the **practical floor on a tuning job is not the token cost** — it is the **fixed ~20-minute overhead plus any minimum-job billing the platform applies** (CS-19 §11.2's "Beyond the video"). "If Vertex enforces a minimum billable token count per job — and several managed tuning services do, in the 10⁵–10⁶ token range — then a 564-token dataset is billed as if it were much larger." The corroborating evidence is the instructor's own **"10 to 15 rupees" [30:38]** — "consistent with a small minimum rather than with the strict token arithmetic, which would predict under one rupee." The action is explicit: **verify whether a minimum applies**, because it changes the "it's basically free to experiment" conclusion at small scale.
- **Why asked:** It is the module's best "your model of the service is incomplete" question. A candidate who computes 564 tokens × $5/1M and stops has produced a number the platform may not charge.
- **Trap:** Dismissing it as a rounding error. The point is not the rupees; it is that the *floor* is a billing construct, not a token count — and the same construct is what makes the 15–20 minute runtime uninformative about job size (§17 #16).

**Q33. What is `ln(V)` for, and what does a loss of 11.4 on a 100k-vocabulary model tell you?**

- **Answer:** It is the **uniform-distribution entropy bound** — the loss a model that has learned nothing would report, because it spreads probability evenly across the vocabulary. For a 100k vocabulary, `ln(100,000) ≈ 11.51`, so **an eval loss of 11.4 is essentially uniform entropy: the model has learned nothing** (CS-19 §19 A8). The prescribed diagnosis order is **"check the schema first (`role: "model"`, populated `parts`), then the data"** — because a schema error that produces empty turns trains a model to emit nothing, which is exactly the uniform-entropy signature.
- **Why asked:** It is the one loss number you can sanity-check without any context, and the *diagnosis order* is the graded part: the source says schema before data, which is counter-intuitive to anyone whose instinct is to blame the data.
- **Trap:** Reporting "11.4 loss, needs more epochs." Uniform entropy is not an underfitting problem, it is a *nothing happened* problem. Note the contrast with IQ-17's Correction on this same constant: the `ln(V)` value must be computed for the actual vocabulary, and reading it as `2.3` is reading `ln(10)`.
- **Cross-check:** CS-18 §19 A7's no-op gate is the same idea in metric form — `identical_outputs` near 1.0. Here you can detect it from the loss alone, *if* you know the bound.

**Q34. Why is the artefact-storage line "negligible/none" rather than a number?**

- **Answer:** Because CS-19 §11.1 line 4 reads **"No published hourly charge identified for the Gemini tuned-model artefact; **verify**"** — with a scale of `$0` and a verdict of "Negligible/none (unpublished)." §11.6 is titled "Storage cost of the adapter — the honest answer," and the honest answer is that you cannot price a line the vendor does not publish. The practical handling is the same as for every other figure in §11: the *structure* is durable (this line is not where your bill lives), the *number* must be re-derived.
- **Why asked:** It tests whether the candidate can say "I don't know, here is how I'd find out" without either inventing a number or hand-waving. On a closed platform this happens constantly, and an interviewer is checking whether the candidate's confidence is calibrated to their evidence.
- **Trap:** Substituting the GCS storage price for the model price and calling it the same line. They are different resources billed by different meters.

**Q35. A tuned Gemini beats the best open-weight alternative by 3% on your eval. Do you deploy the endpoint?**

- **Answer:** Run §11.4's regimes first, then §19 A10's two conditions. A dedicated tuned endpoint beats self-hosting **only when** (a) **traffic is high enough that the fixed hourly charge is a small fraction of the total**, and the per-token comparison favours the self-hosted deployment; **and** (b) **the task cannot be served by an open-weight model of comparable quality** — meaning "you specifically need Gemini's capabilities, and you have **measured** that the tuned Gemini beats the best open alternative by more than the cost delta." A 3% eval delta is almost never more than the cost delta: at 100,000 calls/month you are paying `$1,460` for the endpoint and `$63.04` for tokens — a **96% fixed share**, so you would be paying ~23× the usage charge for 3%. The honest answer is **no unless the traffic regime makes the fixed charge irrelevant.**
- **Why asked:** Both conditions must be present and the candidate must supply the *comparison*, not just the quality number. The measured-delta requirement is the part people skip.
- **Trap:** Answering on quality alone, or on cost alone. §19 A10 pairs them deliberately.

**Q36. Why does the "no export" fact make the compliance position *weaker*, not just the architecture position?**

- **Answer:** Because **provenance of the trained artefact is harder to demonstrate.** CS-19 §16.7's "Beyond the video": with an open-weight fine-tune you can **hash the adapter, sign the model card, and produce the exact training command**. With a managed tune "you have a job record and a metric series. That is a *weaker* evidentiary position" — and it is "the reason Case E (§15.5) went self-hosted despite the otherwise-attractive GCP fit." The module's instruction is to **say it out loud in the design review**, "because it is not obvious until the auditor asks." §16.7's own checklist adds the erasure question: if a data subject asks for erasure, "you can delete the dataset and retrain. **You cannot un-learn.**"
- **Why asked:** It reframes "no weights" from a technical limitation into an audit finding, which is where it actually costs money. The erasure answer is the sharpest sub-question, and it is the one the candidate has to say without flinching.
- **Trap:** "The vendor handles compliance." §17's table makes *you* the owner of data placement, retention, provenance, CMEK, VPC-SC, audit logging and endpoint access control — and it warns that `aiplatform.user` at project scope "means anyone with that role can call **every** endpoint."

**Q37. Why can you not compare two Vertex runs and conclude one dataset is better?**

- **Answer:** Because **there is no seed and no determinism** (CS-19 §10.10, §17 #12). "Two identical jobs produce different models." Before attributing any delta to a change, you must **calibrate your noise floor** — run the same config twice, once — and then treat anything inside that band as noise. The prescribed iteration discipline (change one thing per job, §7.3) is what makes the noise floor measurable rather than mysterious.
- **Why asked:** It is the module's statistical-honesty question, and it is the one most likely to be answered with a confident wrong "yes, run A/B tests." Every other module in this handbook gives you a seed; this one does not.
- **Trap:** Reusing your open-weight evaluation protocol — where seeds are controllable — without re-deriving the noise floor. A protocol that assumes determinism produces false positives here in a way it does not elsewhere.

**Q38. What does "the console is right" mean, and when does it apply?**

- **Answer:** CS-19 §7.4's rule: **"When the docs and the console disagree, the console is right — it is reading the same enum the API validates against."** It applies wherever an enumerable value set is involved — `adapterSize` availability per model, the supported `baseModel` list, `tuningMethod` availability — because the console's dropdowns are **generated from the live API**, while the docs and this module's tables are snapshots. The pattern generalises: on a churning managed platform, **prefer the instrument that reads the live contract over the document that describes it.**
- **Why asked:** It is the module's answer to "how do you know what's current?" — and it is a better answer than "check the pricing page," because it names the specific artefact that cannot be stale.
- **Trap:** Treating the console as a UI convenience. It is a schema-discovery tool, and §7.4 lists it as the fastest way to discover the currently supported value set.

---

## Level 4 — System Design & Scenario

> These are 10–15-minute whiteboard questions. Answer in the fixed shape: **requirements → constraints → design → trade-offs → failure modes.**

**Q39. A team inside a GCP shop wants a support assistant that speaks in house style. Design it, and defend the endpoint.**

- **Answer:** *Requirements:* style and format adaptation — not new facts; the org already runs on GCP, so the organisational fit is real. *Constraints:* no ML engineers; the eval must be defensible; the endpoint must be justified or absent. *Design:* the full §5.1 pipeline — **(0)** a JSONL in **the `systemInstruction`/`contents`/`parts` shape**, with a schema adapter (§13.2) rather than a second dataset; **(1)** a **deterministic, stratified, seeded split** produced by your own splitter, `val_frac=0.15`, with the **sha256[:16] of each file printed and recorded** (§4.5); **(2)** `validationDatasetUri` passed explicitly; **(3)** `CreateTuningJobConfig` with `epochCount=1` then `3`, `learningRateMultiplier=1.0`, `batchSize` default, `adapterSize=ADAPTER_SIZE_FOUR`; **(4)** the **15–20 minute overhead** budgeted and **`tuned_model_display_name` encoding dataset version + date**; **(5)** state **TRAINED**; **(6)** an `eval_ab.py` run **against the base model**, plus §12.4's seven prompt-mutation axes; **(7)** state **SHADOW** with a regression and refusal suite; **(8)** only then **DEPLOYED**. *Trade-offs:* the training is a rounding error and the endpoint is the whole bill, so the deploy decision is a traffic question — under ~500k requests/month, **do not deploy a dedicated endpoint**; serve from the managed tuned path if it avoids one, or self-host scale-to-zero. *Failure modes, all recorded:* the model never gets deployed and someone deploys it manually for "testing" (the §17 #20 default is TRAINED, not DEPLOYED); the system instruction is dropped at serving time, so the style collapses because it was never in the weights (§12.4); the run is compared against another run without a noise floor (§10.10).
- **Grading:** the **explicit split with a printed hash**, the **paired eval against the base**, and the **refusal to deploy by default** are the three non-negotiables.
- **Fail condition:** deploying at `JOB_STATE_SUCCEEDED` because "I need to test it."

**Q40. Design the teardown and cost-governance story *before* the first job runs.**

- **Answer:** *Requirements:* the module's dominant failure is an endpoint someone forgot about; the controls must survive a busy week. *Design:* (1) **The three-way teardown in the correct order — `undeploy_all()` → `endpoint.delete()` → `Model.delete()`** (CS-19 §18 #4, §6.10). (2) **A $50 budget with alerts at 50/90/100%**, scoped to the project and filtered to Vertex AI (§11.5). (3) **`gcloud ai endpoints list`** as the sweeper, on a schedule, with an owner — "a budget alert plus a scheduled sweeper is the only teardown that survives a busy week" (§18 #15). (4) **A BigQuery billing export query keyed on the endpoint/online-prediction node-hour SKU**, not the tuning line, as the monthly report (§11.5). (5) **The five lifecycle states written into the runbook with an explicit owner per transition** (§16.1) — because "the transition nobody owns is DEPLOYED → UNDEPLOYED, and it is the expensive one." *Trade-offs:* the sweeper adds an alert you have to ignore on purpose during a legitimate deployment window; the alternative is a **$4,380** quarter at the current $2.00/hour rate (CH-19 §12). *Failure modes:* deleting the **Model** believing you released the **Endpoint** — CS-19 §10.3 calls this **"gotcha #1 in this module"** and the video does exactly that at [56:39]; and relying on the console to notice, which §17 #13 addresses ("the console tells you the job state… The console has no opinion about whether your model is *good*").
- **Grading:** the order of the three teardown calls, the SKU-level query, and an *owner* for the undeploy transition.
- **Fail condition:** a plan whose teardown is "delete the model," or one that relies on remembering.

**Q41. Design the rollback, and say what makes it possible.**

- **Answer:** *Requirements:* the model resource is versioned; the endpoint's routing is not. *Constraints:* there is no "30% to `@1`" primitive in the naive setup — "you either deploy `@N` or `@N+1`" (§16.3). *Design:* CS-19 §19 A9's four steps — **(1)** `undeploy_all()` on the current endpoint, or deploy `@N-1` to a fresh endpoint; **(2)** deploy the previous `Model@N-1` to an endpoint; **(3)** flip your proxy's model config to the previous endpoint; **(4)** verify with the regression suite. The resources you touch are the **Endpoint** (which endpoint the proxy points at) and the **Model** (`@N-1`). What makes it possible: **pin the endpoint to an explicit `@N`; keep the previous `Model@N-1` undeployed but alive; record the dataset hash with the model version; and never delete the previous model until the new one has served a full traffic cycle — "including a weekend and any batch job that runs monthly"** (§16.3). *Trade-offs:* keeping `@N-1` alive costs ~nothing (a `Model` carries no published hourly charge); deleting it to "stay tidy" costs you the rollback. *Failure modes:* a rollback that has never been executed is documentation, not a capability — §16.3 requires the procedure be "**written down and rehearsed once**"; and a retrain that silently appends `@2` becoming production by accident, which the explicit pin prevents.
- **Grading:** the four steps in order, the explicit `@N` pin, and the "rehearsed once" requirement.
- **Fail condition:** proposing to roll back by retraining.

**Q42. Design the evaluation that decides whether this model ships.**

- **Answer:** *Requirements:* the platform's automatic metrics are log loss and teacher-forced token accuracy, and neither measures task success (§18 #7). *Design, cheapest first:* (0) **A schema/plumbing check** — does the endpoint answer at all, and does a known training question come back in the right shape (this is what §10.9's "a 10-row dataset is a smoke test" is for); (1) **`eval_ab.py` against the base model** on a frozen held-out set — the module's required build (§12.3); (2) **the seven prompt-mutation axes** (§12.4), reported per-axis rather than as a count; (3) **an LLM-as-judge for open-ended outputs** (§12.5), with every pair judged in both orders to cancel position bias; (4) **the refusal suite** that §16.1 requires as SHADOW's entry criterion. Plus the one number the platform gives you *only* if you passed `validationDatasetUri`: the eval loss curve. *Trade-offs:* layers 0–1 cost minutes; layers 3–4 cost money and are the ones that keep you out of an incident. *Failure modes:* reading the dashboard's four numbers as an evaluation (CS-19's Correction at line 1661 — "three specific problems" with treating them as one); a loss compared against nothing, when §19 A8 gives you the `ln(V)` bound for free; and **a model that passes 1–4 and fails 5, which is a template-follower** (§12.4).
- **Grading:** the paired base-model comparison, the per-axis mutation report, and the `ln(V)` sanity check on any loss figure quoted.
- **Fail condition:** a stack that stops at the dashboard.

**Q43. You must choose between tuning on Vertex, tuning on OpenAI, and self-hosting. Design the decision, not the answer.**

- **Answer:** *Requirements:* the decision must be defensible before any pricing page is opened, and it must survive the platform's churn. *Design — four gates in order:* **(1) Artefact.** Do you need to own the weights — to quantise, merge, self-host, or distil? If yes, **stop**: neither managed platform qualifies, and the answer is an open-weight fine-tune (CS-17). This is CS-19 §7.5's third branch and §9.3's "not fixable — plan around them." **(2) Cost shape.** Is your traffic steady (fixed hourly charge amortises) or spiky and low (it does not)? CS-19 §11.1's line 2 dominates below ~500k requests/month; the answer under that threshold is *do not deploy a dedicated endpoint*, which then disqualifies Vertex for the task regardless of quality. **(3) Schema and product lock.** Do you need Gemini specifically, or will any model of comparable quality do? §19 A10's second condition requires you to have **measured** that the tuned Gemini beats the best open alternative by more than the cost delta. **(4) Organisational fit.** Are you already on GCP with billing, IAM and audit in place, or is the setup itself the project? §18 #10 — the setup is harder than the code. *Trade-offs:* gates 1 and 2 are cheap to answer and disqualify most cases; gate 3 is the expensive measurement and should only be run for the cases that survive. *Failure modes:* choosing on training price, which is a rounding error on both platforms; and choosing on quality without the base-model comparison, which makes the answer unmeasurable.
- **Grading:** the ordering — artefact and cost *shape* before quality — plus the requirement for a measured delta. Credit for naming CH-18 §9.4's structural difference: **OpenAI's bill is usage-shaped, Vertex's is fixed-shaped.**
- **Fail condition:** a recommendation that begins with a quality benchmark.

---

## Level 5 — Debugging & Incident Response

> Answer with a sequence. Say what you check first, second and third, and name the look-alike you are ruling out at each step.

**Q44. The job is `JOB_STATE_SUCCEEDED`, the tuning bill is five cents, and last month's Cloud invoice is $1,460. Sequence.**

- **Answer:** (1) Accept the shape of the answer before diagnosing: **the endpoint is still deployed**, and `$1,460` is exactly one month at the `$2.00/hour` endpoint rate on file (CH-19 §7.1) — this is the module's dominant cost line and it is *supposed* to be the surprise, which is why §11.5 and CH-19 §12's Check 4 exist. (2) Confirm with **`gcloud ai endpoints list`** — CH-19 §11's last row is the silent failure "**bill keeps growing → Endpoint still deployed**." (3) Check **whether deleting the model was mistaken for releasing the endpoint**: §10.3's "cleanup that does not clean up," "gotcha #1 in this module," performed by the video itself at [56:39]. (4) Teardown in the correct order: **`undeploy_all()` → `endpoint.delete()` → `Model.delete()`** (§18 #4). (5) Then fix the process — a $50 budget with 50/90/100% alerts, a scheduled `gcloud ai endpoints list` sweeper with an owner, and the five-state runbook (§11.5, §16.1, §18 #15).
- **Ruling out:** a training-cost explosion. Training is `$0.00282` on the demo, `$0.05` on the script's fixture, and `$10–$300` per job in production (§11.1, CH-19 §7.1). If the number is four figures, it is never the tuning line.
- **Grading:** reaching for `undeploy_all()` and the *ownership* fix, not just the immediate teardown.

**Q45. `400 ... role must be one of ['system','user','model']`. Sequence.**

- **Answer:** (1) The message is unambiguous — you used `assistant` and Vertex wants `model` (CH-19 §11). Fix the renderer, not the file. (2) **Check the system prompt's *placement* at the same time**, because the two failures travel together: if the system turn came from an OpenAI-shaped row, it is probably still inside `contents` rather than hoisted to `systemInstruction`, which produces a different error — `400 ... system message must be the first`, or a silent user-turn training (§4.2 difference #2). (3) Check the **nesting**: `content` must have become `parts: [{"text": ...}]`, and §4.2's difference #3 warns the failure mode there is **not an error** — it is "an empty turn that trains the model to emit nothing." (4) Check the **top-level key**: `messages` → `contents`, and §4.2 is explicit that "there is no aliasing." (5) Fix it once by building the schema adapter (§13.2), so the next dataset cannot drift. (6) Note that the script's `_to_vertex()` normaliser already performs the `assistant` → `model` rewrite, so a rejection here means the file bypassed the normaliser — check which path produced it.
- **Ruling out:** a malformed-row error (`400 Invalid JSON at line N`), which is a *different* error and would be reported with a line number.
- **Grading:** fixing the renderer rather than the artefact, and finding the *other three* schema differences while you are in there.

**Q46. `FAILED_PRECONDITION: The service account ... does not exist`. Sequence.**

- **Answer:** (1) The message means you named a principal that does not exist in this project — CH-19 §11's fix is the specific one: **"Wrong project number. Re-derive it."** (2) The trap CH-19 §5.2 flags is that the bucket-access principal is **`service-PROJECT_NUMBER`**, not anything keyed on `PROJECT_ID` — and project *IDs* and project *numbers* are different strings that look similar. (3) If the failure instead reads `403 ... does not have storage.objects.get access`, the account exists and the **bucket** grants are wrong — a different fix on the same IAM surface. (4) If it reads `403 Permission 'aiplatform.tuningJobs.create' denied`, it is the Vertex role (`roles/aiplatform.user`), not the storage role. (5) Only after the principal is right, check **region alignment between bucket and job** (§10.5). (6) Then check that the **Colab identity and the GCP identity match** (§10.4) — the failure that produces a *working* local run and a failing notebook.
- **Ruling out:** a missing-credentials error. `google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials` is the ADC problem (`gcloud auth application-default login`), and it is a different error with a different fix.
- **Grading:** three distinct errors distinguished by their messages, each with its own remedy — and the project-*number* trap named.

**Q47. You run a 10-row job, get `SUCCEEDED`, and the model answers correctly in the console. Is it tuned? Sequence.**

- **Answer:** (1) **Not established.** §17 #4: the demo question is "a fact Gemini already knew," and "**nothing in the video's demo distinguishes a tuned model from a prompted one.**" §18 #13 is blunter: "A 10-row managed fine-tune is a plumbing demo, not a model. **It cannot distinguish a tuned model from a prompted base model.**" (2) The test that decides it is §12.4's **drop the system instruction** axis: if the house style survives, the style is in the weights; if it collapses, **the style was never in the weights** and you have a prompt, not a tune. (3) Run the other six mutation axes — a different language, whitespace, three rephrasings, the out-of-scope laptop question, a longer answer, two questions in one turn — and read them **per-axis**. (4) Run `eval_ab.py` against the base. (5) Check the loss against `ln(V)` for the vocabulary (§19 A8) — if your eval loss is at uniform entropy, nothing was learned regardless of the answers you saw. (6) Then ask the sizing question: 10 rows with an unseen, unstratified 2-example holdout is a smoke test, and §18 #13 says so.
- **Ruling out:** "it works, so it's fine." The console has no opinion about quality (§17 #13), and `SUCCEEDED` means the process exited cleanly.
- **Grading:** the drop-the-system-instruction test and the base-model comparison. Both are required; neither is optional.

**Q48. Two identical jobs give different eval losses. Your colleague wants to conclude dataset B is better. Sequence.**

- **Answer:** (1) **Stop the inference.** There is **no seed and no determinism** on Vertex (§10.10), so two identical jobs produce different models — the delta you are looking at is at least partly noise. (2) **Calibrate the noise floor**: run the *same* config twice and measure the spread (§17 #12). Nothing inside that band is a finding. (3) Only then compare A and B, and only with **one thing changed** between them — §7.3's discipline, which exists precisely so that a delta is attributable. (4) Check whether the two jobs even saw the same split: §4.5 warns that the default split "is not reproducible across jobs," so if either job used the implicit split you are comparing two *datasets* as well as two models. (5) If the delta survives the noise floor and the split is explicit and identical, prefer the comparison on a **task metric**, not on loss — §17 #7's whole point. (6) And if the difference is small, remember the cost: at 100k calls/month the deployment is `$1,460.00` against `$63.04` of tokens, so a small quality delta must be measured against a fixed charge that is still `23×` the usage line.
- **Ruling out:** a genuine dataset effect. It may exist — but it cannot be concluded from a single run on a non-deterministic platform.
- **Grading:** the noise-floor calibration run. A candidate who goes straight to statistical tests on two points has missed the platform's defining property.

**Q49. The tuned model is deployed, in production, and returns `404 ... endpoint not found`. Sequence.**

- **Answer:** (1) CH-19 §11's row for that error is `404 ... endpoint not found` → "**Not deployed, or wrong region**" — so check **deployment state** first, then **`--location`**. (2) Check **which client you are calling with**: §10.6's Correction records that the SDK has **two distinct inference paths**, and conflating them is "a common source of 'works in my notebook, 404s in prod'." (3) Check the **API version** — `v1beta1` for tuning, `v1` for inference (§10.6); a client pinned to the wrong one can miss the resource. (4) Check the **project and region** the client is constructed with against the endpoint you deployed, including the region-alignment issue from §10.5. (5) Check **which endpoint** you are pointing at — if the endpoint was resolved by `jobs[0]` or `list()[1]` rather than a persisted name, you may be querying someone else's resource (§10.8, §19 A4). (6) Only then check whether the endpoint was **undeployed by the sweeper** — a rollback or a scheduled teardown will produce exactly this, and it is a *feature* firing.
- **Ruling out:** a credentials failure, which produces a 403 or a `DefaultCredentialsError`, not a 404.
- **Grading:** the two-inference-paths answer and the API-version answer together. Either alone is a partial diagnosis.

**Q50. The job's loss curve falls to near zero and the model parrots the training rows verbatim. Sequence.**

- **Answer:** (1) This is **memorisation**, and the arithmetic is in the config: `epochCount` is a linear pass count and there is **no early stopping** on Vertex (§7.2), so on a small dataset every extra epoch is another look at the same sentences. (2) Confirm the mechanism: §7.2 — with 10 rows and a constant system instruction, "epoch 10 means the model has seen each row ten times — that is memorisation of 10 sentences, which is a lookup table, not a behaviour." (3) Run **§12.4's rephrase axis**: "surface-form memorisation" is exactly what it detects, and a model that passes 1–4 and fails 5 is a **template-follower** — "a real product defect, and it is invisible in every metric Vertex shows you." (4) Fix it the cheap way: **run 1 epoch instead of 3.** §7.2's policy — "run 3 epochs, evaluate, and if it overfits, run 1" — exists because the token bill is negligible and the wall-clock is the real cost. (5) Then fix the *data*: add rows, and check the system instruction is not a 46%-of-the-bill constant (§4.2). (6) Re-check the eval: with 10 rows the holdout was 2 examples, so the run had no instrument capable of detecting this in the first place (§4.5).
- **Ruling out:** an LR problem. A high `learningRateMultiplier` accelerates memorisation, but the *cause* here is pass count — and §7.3's note that "a bad LR does not fail loudly" means the loss curve is the only warning either way.
- **Grading:** the rephrase axis and the "run 1 epoch" fix. A candidate who proposes more data *before* more epochs has the ordering wrong: epochs are free to change and data is not.

**Q51. The tuned model is worse than the base model. Distinguish the four causes.**

- **Answer:** CS-19 §14.2 is literally titled "Distinguishing the four things that look like 'the fine-tune didn't work'," and the sequence is: (1) **A schema/plumbing defect** — an empty-turn or mis-rendered dataset trains the model to emit nothing, and the signature is an eval loss at `ln(V)` (§19 A8). Check the schema *before* the data. (2) **A capacity defect** — `adapterSize` left at the default rank 1 cannot represent the behaviour change, and §7.1's "too low" column says the result is "the tuned model looks like the base model with a slightly different tone." (3) **A statistical defect** — 10 rows with a 2-example holdout, so the comparison is noise (§4.5, §18 #13). (4) **A task defect** — the task was never a fine-tuning task: if the requirement is *facts*, SFT "injects facts weakly and expensively" and RAG or continued pretraining is the answer (§7.5, §17 #4). Run the four in that order, because each is cheaper to check than the next, and each has a distinct signature rather than a shared "it's worse" symptom.
- **Ruling out:** a serving-side problem. If the model is worse *in production* but fine in the console, that is a different branch — prompt drift on the system instruction (§12.4), or the wrong endpoint (§10.8).
- **Grading:** the ordering by cost-to-check, and the `ln(V)` signature for cause 1.

---

## Rapid Fire — True / False / One-Liner

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | You can download the tuned adapter. | **False** | "No weights, no export, no exit" (§18 #1) |
| 2 | Training is the expensive part. | **False** | `$0.00282` vs a monthly hourly endpoint charge (§11.1) |
| 3 | `JOB_STATE_SUCCEEDED` ends your costs. | **False** | It *creates* the billable endpoint; it starts them (§17 #5) |
| 4 | The tuned model and the endpoint are one resource. | **False** | Two resources, two bills, two lifecycles (§18 #3) |
| 5 | `endpoint.delete()` releases the deployment. | **False** | `undeploy_all()` first, then delete (§18 #4) |
| 6 | Deleting the Model stops the endpoint charge. | **False** | Gotcha #1 in the module (§10.3) |
| 7 | Vertex accepts the OpenAI `messages` JSONL. | **False** | It wants `systemInstruction` + `contents` + `parts` (§4.2) |
| 8 | The assistant role is `assistant`. | **False** | It is `model` (§4.2, CH-19 §11) |
| 9 | The system prompt goes in the message list. | **False** | It is hoisted to a top-level sibling (§4.2) |
| 10 | `content` is a string. | **False** | It is a list of `parts` with `parts[0].text` (§4.2) |
| 11 | If you pass no validation file, nothing is held out. | **False** | Vertex splits automatically — unseen, unstratified (§4.5) |
| 12 | A 10-row job has a 2-example holdout. | **True** | 80/20 default; "a statistic computed on 2 samples" (§4.5) |
| 13 | The video sets four hyperparameters. | **False** | It names **zero**; the config has one field (§7.0) |
| 14 | `epochCount` linearly multiplies the training bill. | **True** | `billable_tokens = total_tokens × epochCount` (§7.2) |
| 15 | `batchSize` changes your token bill. | **False** | Every token is read once per epoch (§7.3) |
| 16 | `learningRateMultiplier: 1.0` means zero LR. | **False** | It means "Vertex's default"; you cannot read the base rate (§19 A7) |
| 17 | The default `adapterSize` is the safe one. | **False** | Default is rank 1; use 4 or 8 (§7.1, CH-19 §4.1) |
| 18 | You can early-stop a Vertex tuning job. | **False** | No `earlyStoppingCallback`; it runs to completion (§7.2) |
| 19 | The 15–20 minute runtime tells you the dataset was small. | **False** | The *overhead* is fixed (§17 #16) |
| 20 | Eval loss measures whether the model does your task. | **False** | It is next-token cross-entropy on your own distribution (§17 #7) |
| 21 | A loss of 11.4 on a 100k vocab means uniform entropy. | **True** | `ln(100,000) ≈ 11.51` — the model learned nothing (§19 A8) |
| 22 | Two identical Vertex jobs produce the same model. | **False** | No seed, no determinism; calibrate a noise floor (§10.10) |
| 23 | `jobs[0]` is your job. | **False** | Newest-first across everyone's jobs; persist the name (§10.8) |
| 24 | `v1` and `v1beta1` are interchangeable here. | **False** | `v1beta1` for tuning, `v1` for inference (§10.6) |
| 25 | An over-long training row raises an error. | **False** | No length knob; **silent** truncation (`--max-example-tokens`) |

---

## Coding / Whiteboard Tasks

### Task 1 — Write the schema adapter (12 minutes)

Write the function that turns one canonical row into both wire formats.

- **Expected solution sketch:** Internally represent a row as `list[tuple[role, text]]` where the roles are the **canonical three** (`system`, `user`, `assistant`) — never `model`, which is a Vertex rendering detail (CS-19 §13.2). Then two renderers. **OpenAI:** `{"messages": [{"role": r, "content": t} for r, t in row]}`. **Vertex:** split the leading `system` turn out into `{"role": "system", "parts": [{"text": t}]}` at the top-level `systemInstruction` key, map `assistant` → `model` in the remainder, and wrap every text as `{"parts": [{"text": t}]}` — emitting `{"systemInstruction": ..., "contents": [...]}`. Assert **exactly one** system turn and that it is **first**, and fail loudly otherwise, because Vertex accepts only one and only in that position (CH-19 §11's `400 ... system message must be the first`).
- **The decisions being graded:** (a) canonical roles internally, platform roles only at render time; (b) the system turn **hoisted** rather than passed through; (c) an assertion that fails on a second system turn or a non-leading one, rather than emitting a file Vertex will silently mis-train.
- **Grading:** both renderers from one source of truth, plus the system-turn assertions.
- **Fail condition:** two hand-maintained datasets, or a renderer that leaves `assistant` in the Vertex output.

### Task 2 — Write the splitter (12 minutes)

Produce train/validation files that are deterministic, stratified and auditable.

- **Expected solution sketch:** Follow CS-19 §4.5's `split_dataset.py` shape: bucket rows by a **stratification key** (there, `systemInstruction`), shuffle each bucket with a **fixed seed**, take `n_val = max(1, round(len(rows) * val_frac))` per bucket when the bucket has more than one row, iterate the buckets in **sorted order** so the result is stable, and write both files with `ensure_ascii=False` and a trailing newline per row. Then **print `sha256[:16]` of each output file**. The default is `val_frac=0.2` with `seed=1337`, and §4.5's sizing rule is **≥ 100 examples and ≥ 10%, whichever is larger**.
- **The decisions being graded:** (a) stratifying at all, so the holdout is not a random slice of the system-instruction distribution; (b) the **hash printed and recorded**, because that hash is what you pin into the model card and what makes two runs comparable; (c) refusing to train when the holdout falls below the sizing rule — the function should warn, and at 10 rows it should say so out loud.
- **Grading:** determinism (re-running gives byte-identical files), stratification, and the printed hash.
- **Fail condition:** `random.shuffle` on the whole file with no seed, or a split that silently gives a 1-example holdout.

### Task 3 — Compute the two-month cost (10 minutes)

A tuned Gemini endpoint is deployed 24/7 for 60 days. Training was 2,000,000 tokens at `$5/1M`, 1 epoch. Your usage is 100,000 calls/month at 500 in / 300 out. Give the two-month total and the fixed share.

- **Expected solution sketch:** Training `= 2,000,000 / 1e6 × $5 = $10.00` **one-off** (the `gemini-2.5-flash` rate on file in CH-19 §7.2 and the script's `PRICES`). Endpoint at the script's and CH-19 §7.1's rate of `$2.00/hour`: `60 days × 24 h = 1,440 hours × $2.00 = $2,880`. Usage at CH-19 §7.1's fixture rate: `$63.04/month × 2 = $126.08` (100k calls/month). **Two-month total ≈ $3,016.08**, of which the endpoint is **~95.5%**. State plainly that the **hourly rate must be re-verified** — CS-19 §11.3/§11.4's `$2.00 × 729.6 h = $1,459/month` and CH-19 §7.1's `$2.00 × 730 h = $1,460` now agree, but both are a dated snapshot that must be re-checked against the live pricing page for your region and your model generation.
- **The decisions being graded:** that you multiply the **hours**, not the calls; that you keep the training line **one-off** and the endpoint line **recurring**; that you quote the endpoint rate **with its generation, its source and its verification warning**; and that you report the fixed share as a percentage, because that percentage *is* the decision.
- **Grading:** the one-off/recurring split correct, the endpoint arithmetic right at 1,440 hours, and the rate flagged as a dated snapshot.
- **Fail condition:** quoting a single endpoint figure as current fact with no source and no generation, or folding the training line into a monthly run-rate.

### Task 4 — Design the pre-flight gate (10 minutes)

Write the checks a CI job should run before a Vertex tuning job is created.

- **Expected solution sketch:** (1) Render the dataset via the **schema adapter** and assert the Vertex invariants — exactly one `systemInstruction`, `contents` non-empty, every element has a non-empty `parts[0].text`, no `assistant` survives, and the last turn is a `model` turn (CH-19 §4.2's last rule). (2) Assert the **split is explicit** — both `trainingDatasetUri` and `validationDatasetUri` present — and that the holdout meets the sizing rule. (3) **Check the per-example length distribution against `--max-example-tokens` and fail on any row over it**, because there is no max-sequence-length knob and truncation is silent — the script's own `--validate`/`--upload` pair is the model to copy. (4) Assert the **base model is a pinned version, not a floating alias** (§7.1: "Pin an explicit version, not a floating alias"). (5) Assert `adapterSize` is **explicitly set** and not left at the rank-1 default. (6) Assert the **cost controls from CH-19 §12's five checks** are satisfied: `--estimate` has been read, the script's prices have been re-verified against the live page, and **a written undeploy plan exists**. (7) Assert `tunedModelDisplayName` encodes **dataset version + date**. (8) Print the training-token estimate and **hard-fail above a budget cap**.
- **The decisions being graded:** (3), (5) and (6) — the three that require having read the module rather than the API docs. A gate that only validates the JSONL's structure would pass every job that produces a surprise invoice or a silently truncated model.
- **Grading:** at least six of the eight, with (2), (3), (5) and (6) present.
- **Fail condition:** a gate that checks structure only, or one that auto-deploys on success.

> **Correction (the idle-endpoint figure — the spread is now CLOSED, and it is a textbook example of why you always name the generation).** This is a correction to a correction, so read it carefully, because the *lesson* outlived the bug. **Earlier revisions of this module had a genuine 1.5× spread on the module's most important number**: CS-19 §11.3/§11.4 priced the dedicated tuned endpoint at **~$2.00/hour → $1,459/month**, while CH-19 §7.1/§7.2 and `code/14_vertex_gemini_finetune.py` used the **1.5-generation** rate of **$3.00/hour → $2,190.00/month** (which is also what made CH-19 §12's "$6,570 at the flash rate" reconcile). **They now agree.** The script's `PRICES` carries the 2.5 generation, `DEFAULT_MODEL` is `gemini-2.5-flash`, and the endpoint rate on file is **$2.00/hour** — so `--estimate` prints **$1,460.00** for a month (730 h × $2.00) against CS-19's **$1,459** ($2.00 × 729.6 h). CH-19 §7.1 now states the agreement explicitly and records the earlier 1.7× training-rate mismatch as historical. **The durable lesson, which is what an interviewer is actually testing:** CS-19 §11's standing warning still applies — "re-derive every number against the live pricing page for your region" — and the *specific* discipline is that **a rate is meaningless without its generation and its source.** A candidate who quotes "the Vertex endpoint rate" without saying *which model generation, which table, and which revision* has quoted nothing checkable. So: quote the generation, quote the source, and re-verify before a budget.

> **Correction (two price *generations*, and a script that now refuses to guess).** CS-19 §11.2 quotes the video's **2.5-generation** rates — *"Gemini 2.5 Pro around $25 for the 1 million token, 2.5 Flash $5, 2.5 Flash Lite $1.5"* [22:29] — and CH-19 §7.2 and `code/14_vertex_gemini_finetune.py` now carry the same generation: **`gemini-2.5-flash` $5.00 train / $0.30 in / $2.50 out / $2.00 endpoint-hour; `gemini-2.5-flash-lite` $1.50 / $0.10 / $0.40 / $2.00.** The **1.5/2.0 rows are retained only for legacy job ids** (`gemini-1.5-flash` $3.00 / $0.075 / $0.30 / $3.00; `gemini-1.5-pro` $8.00 / $1.25 / $5.00 / $5.00; `gemini-2.0-flash` $3.00 / $0.10 / $0.40 / $3.00) and are **not** the current default path. Two things about this are worth more than the numbers. **(a) The Pro tier deliberately has no training rate.** `PRICES` gives `gemini-2.5-pro` **no `train` key at all** — only input $1.25, output $10.00, endpoint $2.00 — and `--estimate` prints **"No tuning rate on file"** and renders its totals as **`≥ $X (training cost unknown)`** rather than folding an invented figure into a total. That is the correct behaviour and it is the direct consequence of CS-19 §4.1's `> **Correction:**` block, which states that the **Pro tier of the 2.5 and 2.0 generations has not generally been offered as a supervised-tuning base model** and that the video's $25/1M Pro claim is "suspect and needing verification" — noting that "$25/1M is 5× the Flash rate he quotes, and if you budget for Pro you have mis-budgeted by 5×." **A tool that refuses to estimate has done you a favour.** **(b) Longest-match-on-token-boundary is now the pricing rule.** `_resolve_price_key` takes the longest key matching on a token boundary rather than the first key contained in the string, so **`gemini-2.5-flash-lite-001` no longer silently prices as `gemini-2.5-flash`** — which was a **3.3× overestimate** on a family where the two models differ by that much. It is the same class of bug as CH-18 §7.2's `next(k for k in PRICES if k in model)`, fixed the same way: never trust a prefix match on a model ID; match the exact snapshot or the longest token-bounded key. **So: pick a generation, verify it on the live page, never mix a training rate from one source with an endpoint rate from another, and never let a prefix match price your job.**

> **Correction (the dataset schema — CH-19 §4.2 shows the variant, not the wire format).** CH-19 §4.2 is headed "The dataset schema" and shows `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "model", "content": "..."}]}`, with the rule table listing key `messages`, roles `system` / `user` / `model`, and "Last turn must be a `model` turn." **That is the *variant* shape, not the canonical one.** CS-19 §4.2, §19 A2 and §13.2 all give the wire format as **`systemInstruction` + `contents[].parts[].text`** with the system prompt **hoisted out** of the message list — and CS-19 §4.2 explicitly describes the `messages`-shaped form as something "Vertex accepted for some tuning flows," warning: **"Do not assume either shape works; validate your file against the schema the console's 'sample dataset' link gives you on the day you train, because the two have coexisted and the accepted one has flipped."** `python code/14_vertex_gemini_finetune.py --help` resolves the ambiguity for the handbook's own tooling, and the resolution is worth quoting: the script's docstring labels **`{"systemInstruction": {"role","parts":[..]}, "contents": [{"role","parts":[..]}]}`** as **"Vertex AI (canonical shape)"** and then says the script *deliberately* writes the other one — *"Vertex also accepts a `messages`-shaped variant — that is what this script writes on upload, because it is the shape the tuning API documents for supervised SFT"* — with `_to_vertex` normalising the common on-disk shapes either way, "including renaming `assistant` to `model`." **So: upload the canonical `systemInstruction`/`contents`/`parts` form if you are writing to GCS yourself; use the `messages` form only if you are going through the script's normaliser — and check which shape your SDK version wants if a job rejects your file.**

> **Correction (the transcript, and the model IDs you might copy from it).** CS-19 §10.1 and the Correction at line 1389: the transcript renders **every** occurrence of "Gemini" as **"Jimny"** (occasionally "Gymni") and **every** occurrence of "Vertex" as **"Vortex"**, "GPD"/"GPT" where the speaker means "Gemini," and "Sunonny" for "Sunny." "If you search the transcript for `gemini-2.5-flash` you will find nothing; if you copy the string `"Jimny 2.5 flash"` into `base_model=` you will get a validation error." §17 #15's rule follows: **model IDs come from the console and the API reference, never from a tutorial transcript.** Two related corrections worth carrying: the notebook as shipped contains a **live project ID and project number** (`gen-lang-client-0486212375`, and `projects/313902242452/...` in printed job names — the Correction at line 715, inside §6.4), so read them from the environment instead; and the video's billing setup (§10.2) describes a **UPI autopay mandate on an India-billed account** — "not how GCP billing works in the US, EU, UK, or most of APAC, where a credit card or bank account is mandatory."

---

## Cheat Sheet of Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| The two resources | `tuned_model.model` @N **and** `tuned_model.endpoint` | Two bills; only the endpoint bills hourly |
| Lifecycle states | **DRAFT → TRAINED → SHADOW → DEPLOYED → UNDEPLOYED** | Owner for each; DEPLOYED→UNDEPLOYED is the unowned one |
| Correct teardown order | **`undeploy_all()` → `endpoint.delete()` → `Model.delete()`** | Deleting the model does nothing to the endpoint |
| Default state for a tuned model | **TRAINED**, not DEPLOYED | "Deploy for evaluation windows measured in hours" |
| Default split if you pass none | **80/20**, held out by Vertex | Unseen, unstratified, not reproducible |
| Holdout on 10 rows | **2 examples** | Every eval number is a statistic on 2 samples |
| Holdout sizing rule | **≥ 100 examples AND ≥ 10%** | Whichever is larger |
| Hyperparameters | **`epochCount`, `learningRateMultiplier`, `batchSize`, `adapterSize`** | The video names **zero** |
| `epochCount` default / safe | **1** / **1–3** | Linear bill multiplier; no early stopping |
| `learningRateMultiplier` default / safe | **1.0** / **0.5–2.0** | Not an absolute rate; never touch on a first run |
| `adapterSize` default / safe | **rank 1** / **4 or 8** | Set is 1/4/8/16/32; 32 overfits small data |
| Step-count formula | `ceil(N_examples × epochs / batchSize)` | 1,000 rows × 3 ep: **750** steps @4, **188** @16 |
| Fixed job overhead | **15–20 minutes** [50:56] | Independent of dataset size — not a proxy for it |
| Demo training cost | **$0.00282** (564 tokens × $5/1M) | CS-19 §11.1, §17 #1, §18 #2 |
| Demo at 3 epochs | **$0.00846** (1,692 tokens) | CS-19 §11.2 |
| Production training band | **$10 – $300 per job** | §11.1's "trivial" column |
| Script fixture tuning cost | **$0.05** (9,447 tokens × 3 ep at $5.00/M) | CH-19 §7.1, on `gemini-2.5-flash` |
| System instruction's share of the bill | **~46%** on the companion set; **87%** if 2,000 tok vs 300 | §4.2, §11.2 |
| Endpoint rate on file | **$2.00/hour** (2.5 generation) | Script `PRICES`; the 1.5-gen row was $3.00 |
| **Endpoint, 1 month** | **$1,460.00** ($2.00 × 730 h) | CH-19 §7.1 — agrees with CS-19's $1,459 ×729.6 h |
| Endpoint band | **$700 – $3,700 / month** | CS-19 §11.1's own range |
| Three months idle | **$4,380** at $2.00/hour | Re-derive; the earlier $6,570 was the 1.5-gen rate |
| Usage at 100k calls | **$63.04/month** | vs $1,460 fixed |
| Fixed share of recurring bill | **96%** at 100k calls | Still the whole story |
| Fixed ÷ usage ratio | **23×** at 100k calls | Line 3 vs line 2 |
| Month 1 / Year 1 total | **$1,523.08 / $18,276.50** | CH-19 §7.1 |
| Training-vs-endpoint ratio | **~500,000 : 1** at demo scale | Line 1 vs line 2 — a *different* pair |
| Crossover (usage catches fixed) | **~2,316,000 calls/month** | CH-19 §7.1 |
| Max-example length guard | **8,192** est. tokens (a documented guess) | `--max-example-tokens`; `--upload` exits non-zero over it |
| Deployment regimes | **<500k do not deploy; 500k–20M comparable; >20M self-host** | CS-19 §11.4 |
| Tuned serving premium | **commonly 2–10× the base rate** | "The multiplier has moved a lot" |
| Uniform-entropy loss | **`ln(V)`** — 100k vocab → **≈ 11.51** | A loss near it means nothing was learned |
| Prompt-mutation axes | **7**, incl. drop the system instruction | Pass 1–4, fail 5 → template-follower |
| Refusal-row share to target | **5–10%** of rows in a 10k set | The hand-authored negative |
| Budget alert | **$50, at 50% / 90% / 100%** | Scoped to the project |
| The SKU that matters | **endpoint / online-prediction node-hour** | Not the tuning line |
| Long-example handling | **No length knob; silent truncation** | `--validate` reports the length distribution; `--upload` refuses over `--max-example-tokens` |
| SDK versions | **`v1beta1`** tuning, **`v1`** inference | Graduation is one-directional |
| Non-determinism | **No seed, no determinism** | Calibrate a noise floor before comparing runs |
| Schema-adapter cost | **~60 lines of Python** | One canonical row, two renderers |
| Prices on file (dated snapshots — VERIFY before budgeting) | **2.5 gen (current default):** `gemini-2.5-flash` **$5.00** train / $0.30 in / $2.50 out / **$2.00**/h; `gemini-2.5-flash-lite` **$1.50** / $0.10 / $0.40 / **$2.00**/h; `gemini-2.5-pro` **no train rate on file** — $1.25 in / $10.00 out / **$2.00**/h. **Legacy, for old job ids only:** 1.5-flash $3.00 / $0.075 / $0.30 / $3.00-h; 1.5-pro $8.00 / $1.25 / $5.00 / $5.00-h; 2.0-flash $3.00 / $0.10 / $0.40 / $3.00-h | CH-19 §7.2 and the script's `PRICES`. **State the generation whenever you quote a rate** |
| On a missing rate | `--estimate` prints **"No tuning rate on file"** and totals as **`≥ $X (training cost unknown)`** | The tool refuses to guess — copy that behaviour |
| Model-id price resolution | **Longest match on a token boundary** | `gemini-2.5-flash-lite-001` must not price as `gemini-2.5-flash` (**3.3×** error) |
| The only durable part | **the hourly endpoint charge dominates** | The structure, not the numbers |

---

## Answers To The Self-Check Questions From CS-19

> CS-19 §19 poses ten questions and answers them under a `<details>` block. These are the interview-grade restatements: the number, the mechanism, and the trap an interviewer will follow up with.

**1. Why can't you download a fine-tuned Gemini model, and what are the two downstream consequences?**
- **Why not:** the base weights are closed-source and the tuned artefact is only ever exposed as a Vertex resource [3:35].
- **Consequences:** (a) **no exit strategy** — migrating means re-doing the fine-tune elsewhere; (b) **no downstream optimisation** — you cannot quantise, merge, batch, or run it outside a Vertex endpoint, so your serving cost is permanently Google's price.
- **Trap:** answering "you can export a checkpoint." You cannot export anything; §7.5's table marks PEFT, full tuning and distillation alike as "No. Not exportable."

**2. Write the Vertex JSONL schema from memory, and name the three fields that differ from OpenAI's.**
- **Schema:** `{"systemInstruction": {"role":"system","parts":[{"text":"..."}]}, "contents":[{"role":"user","parts":[{"text":"..."}]}, {"role":"model","parts":[{"text":"..."}]}]}`.
- **The three differences:** the system prompt is **hoisted** out of the message list; the assistant role is **`"model"`**; the text is nested in a **list of `parts`** objects. (Add the fourth: the top-level key is `contents`, not `messages`, and "there is no aliasing.")
- **Trap:** fixing only the role, because that is the one with a clean error message.

**3. Your colleague's job reports `JOB_STATE_SUCCEEDED`. What resources now exist, and which one bills?**
- **Resources:** a tuned `Model` (`.../models/<id>@1`) and, by default, a deployed `Endpoint` (`.../endpoints/<id>`).
- **Which bills:** the **Endpoint**, per hour, whether or not it receives traffic. The Model carries no published hourly charge.
- **Trap:** assuming the deployment is optional. The default behaviour deploys; the correct resting state is TRAINED.

**4. You forgot to note the job name. What does `client.tunings.list()[1]` return, and why is that dangerous?**
- **Returns:** the second-most-recent job **in the project** — which may be a colleague's, a retry, or a test job.
- **Dangerous because:** you would read *its* `tuned_model.endpoint` and deploy or query the wrong model while believing it is yours.
- **Trap:** "I only have one job." Persist the name returned by `tune()`; if you must re-find one, match on `displayName` plus a creation-time window, never on index.

**5. A tuning job on 10,000 rows × 500 tokens with `epochCount=3` at $5/1M tokens — training cost?**
- `10,000 × 500 × 3 = 15,000,000` tokens. `15 / 1e6 × $5 =` **$75.00**.
- **Trap:** forgetting the epoch multiplier, or applying it to the wrong term. `billable_tokens = total_tokens × epochCount` is the whole identity.

**6. The same job's endpoint runs 45 days at an estimated $2/hour. Endpoint cost, and the ratio to question 5?**
- `45 × 24 = 1,080 hours × $2 =` **$2,160**. Ratio `2,160 / 75 =` **28.8×**.
- **The point:** "the deployment cost nearly thirty times the training cost, and it was entirely avoidable by undeploying."
- **Trap:** quoting the ratio without the hourly rate. The **1.5-generation** rate that earlier revisions of this module carried was `$3.00/hour`, which makes the same 45 days **$3,240** and the ratio **43.2×** — and the 2.5-generation rate on file now is `$2.00/hour`, giving **$2,160** and **28.8×**. The *conclusion* ("the endpoint dwarfs the training bill") is invariant across a 1.5× rate change; the number is not, which is why you quote the generation with the figure.

**7. What is `learningRateMultiplier` a multiplier *of*, and why can't you set an absolute learning rate?**
- **Of:** Vertex's **internal default learning rate**, which you cannot read.
- **Why not absolute:** you can only express "more" or "less" relative to whatever Google chose, so LR values you know from other frameworks (e.g. `2e-4`) are meaningless here and a recipe from a paper cannot be reproduced exactly.
- **Trap:** treating `1.0` as "no change to be made." It means "use the default," and §7.1's safe range is 0.5–2.0.

**8. Your eval loss is 11.4 on a model with a 100k vocabulary. What does that tell you?**
- `ln(100,000) ≈ 11.51`. An eval loss of 11.4 is **essentially the uniform-distribution entropy** — the model has learned **nothing**, spreading probability evenly across the vocabulary.
- **Diagnosis order:** **check the schema first** (`role: "model"`, populated `parts`), **then the data.** An empty-turn schema defect produces exactly this signature.
- **Trap:** reporting it as underfitting and adding epochs. Uniform entropy is a *nothing happened* problem, not a *not enough* problem.

**9. You must roll back a production tuned model. What are the four steps, and which resources do you touch?**
- **(1)** `undeploy_all()` on the current endpoint, or deploy `@N-1` to a fresh endpoint; **(2)** deploy the previous `Model@N-1` to an endpoint; **(3)** flip your proxy's model config to the previous endpoint; **(4)** verify with the regression suite.
- **Resources touched:** the **Endpoint** (which one the proxy points at) and the **Model** (`@N-1`, which must still exist — so never delete the previous model until the new one has served a full traffic cycle).
- **Trap:** not having rehearsed it. §16.3 requires the procedure be written down **and run once** before you need it.

**10. Give the two conditions under which a dedicated tuned endpoint beats self-hosting an open-weight model of comparable quality.**
- **(a)** Traffic is high enough that the **fixed hourly endpoint charge is a small fraction of the total**, and the per-token comparison favours the self-hosted deployment.
- **(b)** The task **cannot be served by an open-weight model of comparable quality** — i.e. you specifically need Gemini's capabilities, and **you have measured** that the tuned Gemini beats the best open alternative **by more than the cost delta**.
- **Trap:** supplying only one. Quality alone ignores the `$1,460`/month endpoint; cost alone ignores the reason you picked Gemini.

---

## Cross-References

| Module | Relationship to IQ-19 |
|---|---|
| **CS-19** Gemini / Vertex Fine-Tuning | The source case study. §4.2 the schema, §4.5 the split, §7 the four hyperparameters, §11 the cost model, §16.1 the five states, §19 the ten questions answered above |
| **CH-19** Vertex / Gemini Cheat Sheet | The compressed reference: §4.1 `adapter_size`, §5.2 the `service-NUMBER` prerequisite, §7 the calculator, §8 symptom → fix, §11 the exact error strings, §12 the five pre-flight checks |
| **`code/14_vertex_gemini_finetune.py`** | The runnable lifecycle and estimator. Its `--help` is the authority on flags; its `PRICES` dict is a third opinion on price (see the Corrections above) |
| **IQ-18** OpenAI Hosted Fine-Tuning | The same shape of service with the opposite cost shape — usage-billed instead of fixed. CH-18 §9.4 is the one-page comparison |
| **CS-18 / CH-18** OpenAI Fine-Tuning | The other managed platform in full, including the break-even algebra that does *not* port to Vertex |
| **IQ-17** Axolotl | The open-weight answer to §7.5's third branch — "Is your goal to get a portable artefact?" |
| **CS-17 / CH-17** Axolotl | The config-driven trainer the migration target is built with |
| **CS-13** Instruction Fine-Tuning | §4.1's argument that "SFT injects facts weakly and expensively," cited three times in CS-19 |
| **CS-04** Fine-Tuning vs RAG vs Agents | CS-19 §7.5's prescribed alternative when the requirement is facts rather than form; Case C (§15.3) is the team that should have used it. (**CS-05 is *RNN-LSTM to Attention***, unrelated to retrieval) |
| **CS-12** Domain-Adaptive Continued Pretraining on Your Own PDFs | The other prescribed alternative for injecting domain knowledge |
| **CS-23** LoRA & QLoRA: the PEFT deep dive | Why `adapterSize` is a rank, and what ranks 4/8/16/32 actually buy on the open-weight side — CS-19 §7.5 and §4.3 cite it by name. **Planned, not yet written** (README Part V); until then CS-11 §4.11 is the written QLoRA treatment |
| **CS-10 / CS-11** Quantisation | What §11.4 means by "with open weights you control quantisation… which is where the 3–10× cost wins live" |
| **CS-22** Embedding Models and Embedding FT | Note the title: **embeddings, not evaluation.** CS-19's only citations of it are the `§16.4` monitoring row *"Data-drift proxy: input embedding centroid shift"* and the standing cross-reference table that points back at that row — a drift detector, not an eval framework. **Planned, not yet written** (README Part V); `code/10_embedding_finetune.py` is the written material. The evaluation stack behind §12 has **no home case study** — CS-19 §12.3's `eval_ab.py` fixture and `code/common/eval_utils.py` are what exists |
| **`code/common/eval_utils.py`** | The written evaluation primitives this bank's §12 questions assume: `exact_match`, `format_compliance`, `refusal_rate`, `win_rate`, `bootstrap_ci`, `decontaminate` |
| **CS-16 / CS-17 / CS-23** | CS-19 §7.5's explicit routing for the portable-artefact branch |
| **AP-01** Ethics & Compliance | Provenance, erasure, CMEK, VPC-SC and endpoint access control — §16.7's checklist. **Planned, not yet written** — README lists appendices as 0 / 1 |

---

*End of IQ-19. Companion artifacts: `CS-19-Gemini-Vertex-AI-Fine-Tuning.md` (case study), `CH-19-Vertex-Gemini.md` (cheat sheet), `code/14_vertex_gemini_finetune.py` (the lifecycle script whose `--help` output is the authority on flags). Every price quoted here is a dated snapshot — CS-19, CH-19 and the script's `PRICES` now agree at $2.00/hour on the 2.5 generation, but that agreement is itself a revision, so re-verify against the live pricing page for your region and your model generation before attaching any of it to a budget.*
