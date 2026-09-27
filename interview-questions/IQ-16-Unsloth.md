# IQ-16 — Unsloth: Interview Question Bank

| | |
|---|---|
| **Module** | IQ-16 — Unsloth (fused-kernel, single-GPU LoRA/QLoRA training) |
| **Source material** | `case-studies/CS-16-Unsloth.md`, `cheat-sheets/CH-16-Unsloth.md`, `code/02_sft_unsloth.py` |
| **Prerequisite banks** | IQ-13 (Instruction Fine-Tuning / SFT), IQ-10 + IQ-11 (Quantization) |
| **Depth tiers** | L1 screening → L2 applied → L3 internals → L4 system design → L5 incident response |
| **Verification rule** | Every flag below was read from `python code/02_sft_unsloth.py --help`; every number is either copied from CS-16/CH-16 or computed in a shell with the arithmetic shown. |

**Ground truth used throughout.** Unsloth is not a training algorithm. It is four mechanical changes to a `transformers + peft + bitsandbytes` QLoRA loop: (1) hand-written backward for the LoRA graph, (2) no attention-matrix materialisation, (3) fused RoPE and fused cross-entropy, and (4) no 4-bit dequantize → compute → requantize round trip on the frozen base. Everything else in this bank — the VRAM arithmetic, the masking traps, the merge bug — follows from those four, and from the fact that the library is *single-GPU* and *architecture-curated*.

**Relationship to IQ-13 — read this first.** IQ-13 asks what a correct SFT *run* is: dataset schema, chat template, assistant-only masking, LR/epochs, eval protocol. IQ-16 asks what changes when that run is executed through Unsloth's kernels — which optimisations are real, which are marketing, which configurations silently drop you back onto the slow path, and where the single-GPU ceiling binds. If a question is about the chat template or the loss mask *as a concept* it belongs to IQ-13; if it is about Unsloth's implementation of it, or about a number in CS-16, it belongs here.

## How To Use This File

- **L1 (Q1–Q14)** — screening. A candidate who cannot answer L1 has not read the library, only the README.
- **L2 (Q15–Q28)** — applied. This is where the real filter is: the `--help` surface, the marker derivation, the mask verification, the save paths.
- **L3 (Q29–Q40)** — internals. The arithmetic behind the kernels and the two-quantization-error merge.
- **L4 (Q41–Q46)** — system design. Sizing, framework choice, cost, and the audit-trail question that decides whether Unsloth is even the right tool.
- **L5 (Q47–Q54)** — debugging. Symptom-first; each answer names the observation that rules the cause in or out.

A correct answer states the *baseline*. "2× faster" is meaningless until you say what it is 2× faster *than*, and the signal is whether the candidate volunteers that unprompted. A wrong answer picks the marketing number; a great answer names the mechanism, the number, and the regime where the mechanism stops helping.

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. In one sentence, what is Unsloth — and what is it not?**
- **Answer:** It is a set of hand-written Triton kernels plus a custom autograd graph that replaces the inner ops of a `transformers + peft` QLoRA training step. It is not a training framework: epochs, grad accumulation, the optimizer step, the LR schedule, logging and checkpointing are TRL's `SFTTrainer` (CS-16 §4.3.3, §17.3). It owns the fused loss and the LoRA backward; it does not own the loop.
- **Why asked / trap:** Candidates who call it "a framework" will blame it for trainer behaviour it does not control — and will not know which of the ten lines in the script is doing what.

**Q2. "2× faster with up to 70% less GPU memory." What is the baseline, and what is the number against a well-tuned FA2 + QLoRA run?**
- **Answer:** The baseline is an unconfigured `transformers + peft + bnb` script: eager attention, no packing, an auto-selected compute dtype that is often FP32, LoRA on `q_proj`/`v_proj` only, and stock `use_gradient_checkpointing=True`. Against a **tuned** baseline — FA2 on, `bnb_4bit_compute_dtype=torch.bfloat16`, `packing=True`, 7 target modules, `use_gradient_checkpointing="unsloth"`, `optim="adamw_8bit"` — Unsloth's own contribution is roughly **1.2–1.4× wall-clock step time and 15–30% peak VRAM** (CS-16 §4.7.1, §4.7.2, §13.1, §18.2). The difference between the two numbers is *configuration*, not kernels.
- **Why asked / trap:** This is the single highest-signal question in the module. A candidate who quotes 2× / 70% without the baseline caveat has memorised a README; one who volunteers that most of the gap is config they could apply themselves is the one to hire.

**Q3. Name the four mechanisms Unsloth actually implements.**
- **Answer:** (1) Manual backprop of the LoRA graph instead of building the full autograd graph. (2) No attention-matrix materialisation — the `[B,H,T,T]` scores are never written to HBM. (3) Fused RoPE and fused cross-entropy. (4) Avoiding the 4-bit dequantize → compute → requantize round trip on the frozen base weights (CS-16 §2.2, §4.2–§4.6; the script's own `--help` intro states the same four).
- **Why asked / trap:** These four map one-to-one onto four separate VRAM/time terms, and every L3 question below is one of them. A candidate who lists "FlashAttention" as a fifth has double-counted.

**Q4. Why must `import unsloth` be the first import in the file?**
- **Answer:** Unsloth rebinds symbols inside `transformers` at import time; if `transformers` classes were already imported and bound, the patches do not take effect and you get a working but unaccelerated run — roughly 1.3–1.5× rather than 2×+ — with no error (CH-16 §1 row 7, CS-16 §4.1, §9.4 #14). This is the same class of silent failure as an unsupported architecture.
- **Why asked / trap:** The failure is invisible in the loss curve. The only detection is asserting that the attention module's `__module__` starts with `unsloth` (CH-16 §5.3).

**Q5. `use_gradient_checkpointing=True` and `use_gradient_checkpointing="unsloth"` — same thing?**
- **Answer:** No. `True` selects stock HF checkpointing: ~25–35% step time for ~60–75% activation savings, and it drops the parts that are cheap to keep. The string selects Unsloth's selective variant — keep the cheap activations, recompute only the expensive ones — which is part of the long-context claim, and is the *only* value that gets you the fused kernels' full activation saving (CS-16 §3 line 211, §7.2 line 1914, §17.13; CH-16 §10 row 16).
- **Why asked / trap:** A string where a bool is expected is a classic silent-default bug: `True` is truthy, so nothing errors, and the run is simply slower than the benchmark you were promised.

**Q6. Why must `lora_dropout` be exactly `0.0`? What happens at `0.05`?**
- **Answer:** Unsloth's fast LoRA path is a hand-written `torch.autograd.Function`; the adapter's input would have to be stochastically masked and rescaled per step, which the fused kernel does not implement (CS-16 §6.4 line 1202, §9.4 #2). At `0.05` the library falls back to peft's generic unfused implementation **silently** — correct model, no fast path, typically costing 15–35% of step time (CS-16 §19 answer 5). There is no warning.
- **Why asked / trap:** It is the archetypal "looks fine, is broken" configuration, and it is why you benchmark after *any* LoRA config change rather than once at project start.

**Q7. `max_seq_length` — is it a truncation window?**
- **Answer:** No, it is a load-time *model-shaping* argument. It configures RoPE scaling and determines which attention kernel compiles. Changing it after `from_pretrained` changes the position encoding, not just the batch shape (CH-16 §1 row 5, CS-16 §6.3, §10). The practical rule: set it once as `ceil(p99.5 / 64) * 64`, and pass the same number to `SFTConfig` — a mismatch between the loader's and the trainer's value is a documented error (CH-16 §11).
- **Why asked / trap:** Candidates who treat it as a runtime knob will raise it "to be safe" and pay 2–4× for a window 99.5% of rows never approach.

**Q8. What does `FastLanguageModel.from_pretrained` return, and what does that one call do?**
- **Answer:** A **tuple** `(model, tokenizer)`. The single call downloads the pre-quantized NF4 checkpoint, loads it without re-quantizing, patches the model class with the fast RoPE / RMSNorm / SwiGLU / attention forward, configures RoPE scaling so `max_seq_length` is valid, sets `bnb_4bit_compute_dtype` from `dtype`, and enables the fast LoRA autograd path for the later `get_peft_model` (CS-16 §6.3).
- **Why asked / trap:** Unpacking it as a single object is the most common first-run error, and the fact that `dtype` silently governs the *compute* dtype is the `bnb_4bit_compute_dtype` trap (Q33).

**Q9. You passed `load_in_4bit=True`. Is the model training in 4-bit?**
- **Answer:** No. The base weights are stored in NF4 and are frozen; the LoRA adapter, its gradients, and the optimizer states are in the compute dtype (bf16/fp16). Backward runs on dequantized activations. The 4-bit storage is what buys the VRAM; the compute dtype is what determines the speed (CS-16 §17.4, §4.6.1).
- **Why asked / trap:** "Training in 4-bit" leads people to expect 4× the throughput, which is Q40's trap. The correct framing is 4× less *weight* memory, not 4× less time.

**Q10. You loaded with `load_in_4bit=True` and call `peft`'s `merge_and_unload()`. What goes wrong?**
- **Answer:** `merge_and_unload()` folds `W + (α/r)·BA` into an FP16 tensor, which requires PEFT to dequantize the NF4 weights first. NF4 uses **double quantization** — the per-block absmax values are themselves quantized — so the merged result carries the reconstruction error of that two-level scheme, and the merge happens in a dtype that differs from the one the adapter was trained against. The model loads, generates fluent text, and is measurably worse on your eval, with no error anywhere (CS-16 §6.10, §19 answer 6). Use `model.save_pretrained_merged(dir, tokenizer, save_method="merged_16bit")` — the script's `--merge` path (line 443) calls exactly that.
- **Why asked / trap:** This is the module's most expensive silent bug. The mandatory guard is the logit-equivalence test: max |Δlogit| < ~0.2 between base+adapter and the merged model, on the same dtype (CS-16 §6.10).

**Q11. What does `train_on_responses_only` actually do?**
- **Answer:** It patches the trainer's data collator so that loss is computed only on the assistant span. Mechanically: Unsloth renders the conversation with the tokenizer's chat template, finds the response marker by string search (`full.find(response_part)`), re-tokenizes the prefix up to that offset, and sets `labels[:prefix_len] = -100` (CS-16 §4.9.2, §19 answer 7). It is string matching on your template, not schema awareness — which is the whole source of its failure modes.
- **Why asked / trap:** The tempting summary is "it masks the prompt". The interviewer wants the mechanism, because every failure mode follows from "string search against a template".

**Q12. Why does the script's `DEFAULTS["model"]` point at an `unsloth/...-bnb-4bit` id rather than the upstream model?**
- **Answer:** Because those uploads are pre-quantized: the download is the NF4 checkpoint (roughly 0.5 bytes/param, ~4.5 bits measured with double quantization) rather than the 2 bytes/param FP16 weights, and it is loaded *without* re-quantizing, which removes a minutes-long startup step and a source of nondeterminism (CS-16 §6.3, §17.11).
- **Why asked / trap:** They are not "just a convenience" — for a 7B they are ~3.9 GB of download instead of ~15 GB, and for a 1.1B tiny model the difference is the difference between a fast startup and a slow one on a free-tier GPU.

**Q13. What happens if you point Unsloth at an architecture it does not support?**
- **Answer:** Nothing errors. `from_pretrained` is written never to fail by design, so the model loads with **stock HF modules** and you run at approximately 1.0× with normal-looking loss. The detection is to assert that the attention module's `__module__` starts with `unsloth` (CH-16 §1 row 3, §5.3; CS-16 §8.2, §17.10).
- **Why asked / trap:** A silent 1.0× is worse than a crash, because it costs you the entire training run and is only discovered at the end. Every "Unsloth didn't help" report starts here.

**Q14. Is Unsloth a multi-GPU tool?**
- **Answer:** No — it is single-GPU. A 70B on 8×A100 needs FSDP or tensor parallelism, which Unsloth does not provide; the recommendation is to use it for the single-GPU *search* phase on a 7B/8B proxy and port the winning configuration to Axolotl for the multi-GPU audited run (CS-16 §8.1, §15.2, §19 answer 10; CH-16 §9.1).
- **Why asked / trap:** The follow-up is always "so where does it still fit in their pipeline?" — and the answer is the configuration search, not the production run. A candidate who says "just use it with accelerate" has not read §8.1.

## Level 2 — Applied & Implementation

**Q15. Recite `code/02_sft_unsloth.py`'s flag surface, and say which flags change what you get.**
- **Answer:** `--data` (required), `--out`, `--model`, `--max-seq-len`, `--r`, `--lora-alpha`, `--lora-dropout`, `--batch-size`, `--grad-accum`, `--epochs`, `--lr`, `--no-4bit`, `--full-finetune`, `--packing`, `--assistant-only-loss`, `--no-assistant-only-loss`, `--max-steps`, `--dry-run`, `--merge`. The four that change the *kind* of run rather than its size: `--no-4bit` (bf16 base instead of NF4 — needs the VRAM, better quality), `--full-finetune` (all weights, "~4x the VRAM" per the help text), `--packing`, and `--merge` (save a merged 16-bit model at the end, "needed for llama.cpp").
- **Why asked / trap:** `--lora-dropout` is exposed with no guard (see the Correction after Q18) and `--assistant-only-loss`/`--no-assistant-only-loss` are a mutually-cancelling pair whose default is on. Candidates who never read `--help` cannot answer the follow-up about defaults.

**Q16. What is the default for assistant-only loss in the script, and how do you turn it off?**
- **Answer:** On. `--assistant-only-loss` and `--no-assistant-only-loss` are both present; the `--dry-run` output prints `assistant-only     True`, and `--no-assistant-only-loss` flips it. This matters because masking is a *correctness* choice, not a speed one — training on the prompt teaches the model to generate questions (IQ-13).
- **Why asked / trap:** The pair exists precisely because the default is a policy decision the operator must be able to reverse; a candidate who assumes the flag turns it on has the polarity backwards.

**Q17. Spot the bug: `FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0.05, ...)` inside an otherwise perfect config. What is the run?**
- **Answer:** It works, produces a correct model, and silently loses the fast path for the affected layers — because Unsloth's fused LoRA kernel asserts a dropout-free configuration and falls back to peft's generic autograd otherwise (CS-16 §6.4, §9.4 #2). Expect 15–35% slower steps and no warning in the log. Fix: `lora_dropout=0.0`, and if you genuinely need regularisation, raise weight decay or shorten training.
- **Why asked / trap:** Note the script itself would have permitted this: `code/02_sft_unsloth.py` line 119 exposes `--lora-dropout` (default 0.0) and line 354 passes it straight into `get_peft_model` with no assert and no warning. The library's silent fallback is one defect; the wrapper inheriting it unguarded is a second.

> **Correction:** `cheat-sheets/CH-16-Unsloth.md` §1 row 4 and §4.2 state that `lora_dropout` "must be exactly 0.0" and that any non-zero value "reverts to peft's generic autograd path" (citing CS-16 §6.4 and §9.4 #2). The library behaviour is as documented — but `code/02_sft_unsloth.py` contradicts the guidance by exposing `--lora-dropout` with **no validation and no warning** at line 354. The doc's rule is the right one; the script is where the rule is not enforced. Anyone running `--lora-dropout 0.05` will get the slow path and will have to discover it by benchmarking.

**Q18. How are `instruction_part` and `response_part` obtained, and why not hard-code them?**
- **Answer:** Derive them from the tokenizer's own template at runtime. `code/02_sft_unsloth.py` lines 416–417 sets `instruction_part="<|im_start|>user\n"` and `response_part="<|im_start|>assistant\n"` for its Qwen default. CH-16 §5.2 shows the robust version: call `apply_chat_template` on a probe conversation with sentinel contents (`"X"` and `"Y"`), take the text between them, and use whatever the template actually emitted. Hard-coded markers are correct only for the family you copied them from — Llama-3.3 uses `<|start_header_id|>assistant<|end_header_id|>\n\n`, Gemma-2 uses `<start_of_turn>model\n`, Phi-3 and TinyLlama-Chat use `<|assistant|>\n` (CH-16 §5.2, CS-16 §6.8).
- **Why asked / trap:** The failure when the marker is wrong is *not* an error — it is a mask that covers nothing or everything, with a healthier-looking loss curve. The derivation is the fix; see Q19 for the proof.

**Q19. After applying `train_on_responses_only`, what do you run before `trainer.train()`?**
- **Answer:** The assertion block from CH-16 §5.2 / CS-16 §4.9.3: run the masking on `dataset[0]` and (1) assert something *is* supervised — if the count is 0 the mask is inverted or the marker never matched; (2) assert *not everything* is supervised — if it is total, the patch is a no-op; (3) assert the supervised fraction is in a plausible band (10–60% for instruction data with short answers; >90% means your prompts are tiny or the boundary is misplaced; <2% means you are training on almost nothing); (4) decode the supervised span and read it — it must be the assistant's reply *including* its trailing terminator, and nothing else.
- **Why asked / trap:** Check (4) is the only one that is actually proof. Fractions can look plausible while the boundary is off by one token, which trains the model on half a word.

**Q20. Why does CH-16 §5.1's dataset cell use `packing=False` while validating the mask?**
- **Answer:** Because packing concatenates multiple examples into one sequence, so a marker-matching bug appears as an *interleaved* mask rather than a cleanly wrong one — much harder to read. Validate unpacked, then turn packing on. With `packing=True` you must run the same check on the *packed* batch and inspect every row in the pack (CH-16 §5.2).
- **Why asked / trap:** Packing is one of the largest throughput levers (2–5× on Alpaca-shaped rows, whose ~150–250 tokens fill a 4,096 window poorly), so people enable it first and debug second — exactly backwards.

**Q21. `optim="adamw_8bit"`. Show the arithmetic for why it is close to free at LoRA ranks but not at full fine-tuning.**
- **Answer:** At `r=32`, 7 modules, TinyLlama-1.1B, the adapter is **25,231,360** parameters = **2.294%** of the 1.100048384e9 base. AdamW in fp32 keeps two moments, so 8-bit states cost `25,231,360 × 2 B = 50,462,720 B ≈ 0.05 GB` versus 4× that in fp32 (`≈ 0.20 GB`) — small either way. At full fine-tuning the same two states apply to 1.1e9 parameters: `1.1e9 × 2 B × 2 = 4.4 GB` of 8-bit state versus `17.6 GB` in fp32, which is the difference between fitting and not fitting (CS-16 §19 answer 8; CH-16 §10 "6 B per trainable param" = 2 adapter + 2 grad + 2 adamw_8bit).
- **Why asked / trap:** The 6 B/trainable-param figure is the number to carry: it converts a parameter count into a VRAM line in one step. The trap is assuming it applies to the *base* parameters in QLoRA, where it does not — the base has no gradients and no optimizer state.

**Q22. `r=16` versus `r=32` versus `r=64` on 7 target modules of a 1.1B. What are the parameter fractions, and what is the ranking rule?**
- **Answer:** Approximately **1.0% / 2.3% / 4.5%** of the base (CH-16 §4.4, §5.3; CS-16 §6.5 band check). The rank sanity rule is **100–1,000 examples per million trainable parameters** (CH-16 §10) — at `r=16` on a 1.1B that is ~11M params, i.e. ~1,100–11,000 examples per epoch-scale pass. Below the band you overfit the adapter; above it, the rank is the binding constraint and you should raise `r` rather than the epoch count.
- **Why asked / trap:** The fractions are a *sanity check on your `target_modules` list*, not just trivia: 0.1% means you covered only `q_proj`/`v_proj` and left four of the seven modules frozen, which is a different (and usually worse) experiment than the one you think you ran.

**Q23. `lora_alpha`: should it equal `r` or be `2r`?**
- **Answer:** Unsloth's own examples use `alpha = r` (so the effective scale `α/r = 1`); CH-13's default is `2r`. Both are defensible, and the script's `DEFAULTS` use `r=16, lora_alpha=16`. What is *not* defensible is changing `r` without re-tuning the LR, since the effective update scale is `α/r` — halving `r` at fixed `α` doubles the adapter's contribution per step (CH-16 §10, §4.5).
- **Why asked / trap:** The question tests whether the candidate knows `α/r` is the quantity that matters, rather than memorising a ratio.

**Q24. Name the four save paths and when each is right.**
- **Answer:** (A) **Adapter only** — `save_pretrained`, the source of truth, ~50–200 MB, needs base + peft at load time; right for iterating, version-controlling, `vLLM --enable-lora`. (B) **Merged FP16** — `save_pretrained_merged(..., save_method="merged_16bit")`, the single deployable artefact, and the *only* safe merge from a 4-bit base. (C) **Merged 4-bit** — lossy twice: the base's quantization error plus the merge's rounding plus re-quantization; usually the wrong choice. (D) **GGUF** — for llama.cpp / Ollama / LM Studio, and the script's `--merge` help text flags llama.cpp as the reason it exists (CS-16 §6.9; CH-16 §5.4; script line 443).
- **Why asked / trap:** The trap is that (B) and (C) look like a memory/size trade-off but (C) is not a *smaller version of the same model* — it is a model with a second, independent quantization error layered on the first (IQ-11).

**Q25. What does `--full-finetune` change in the budget?**
- **Answer:** Everything that scales with parameter count. The base weights become trainable, so you add a bf16 gradient (2 B/param) and fp32 AdamW moments (8 B/param) on top of the weights — the CH-16 figure is **12–14 B/param** for full FT versus **6 B/trainable param** for the adapter and **0.5 B/param** for the frozen NF4 base. The script's own help says "needs ~4x the VRAM". The LR also moves: **~1e-5…2e-5** for full FT against **~1e-4…3e-4** for LoRA (CH-16 §10, §4.3).
- **Why asked / trap:** Candidates routinely carry the LoRA LR (2e-4) into a full fine-tune and destroy the model. The 10–20× LR gap is the load-bearing part of the answer.

**Q26. In the dataset cell, why is `remove_columns=ds.column_names` load-bearing?**
- **Answer:** It leaves only the `messages` column, which is what the trainer expects. Omitting it still trains, but the unused Arrow columns inflate the on-disk dataset and, with some collators, cause a type error (CS-16 §6.6). It is also what makes the schema unambiguous — the `KeyError: 'text'` versus `KeyError: 'messages'` pair in CH-16 §11 is exactly the symptom of having both schemas in play.
- **Why asked / trap:** The deeper answer is that the `messages` column lets the *model's own template* define the format, which is the only portable choice across Llama-3, Qwen2.5, Gemma-2 and Phi-3.

**Q27. Read this `--dry-run` output and say what is right and what is wrong with it.**

```
  4B  |  method=qlora  |  optimizer=adamw  |  seq=2048  batch=2 x accum 4
  trainable params           0.080 B (2.00% of base)
  weights                     1.86 GB
  TOTAL (1 GPU)               3.08 GB
  FITS on 24 GB (4090/3090)                YES
```

- **Answer:** The *shape* is right and the arithmetic is internally consistent — `2.00%` of 4B is 0.080 B trainable, and 4-bit weights for a 4B model are ~1.86 GB. What is wrong is the `4B`: the script's default model is `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, and the plan was computed with 4B memory arithmetic, so every VRAM line is understated for the model that would actually be trained. See the Correction after Q39.
- **Why asked / trap:** This is a "the tool told you a number, do you believe it?" question. The signal is whether the candidate checks the plan against the model they passed rather than reading the ✅.

**Q28. What does `--merge` invoke, and why is that flag the right one to pair with `--no-4bit`?**
- **Answer:** `model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")` (script line 443) — the safe merge from a 4-bit base (Q10). Pairing it with `--no-4bit` matters because if the base was loaded in bf16 there is no NF4 dequantization in the merge at all, so the artifact has exactly one quantization decision in it (none) instead of two; the cost is the VRAM the 4-bit load was buying you.
- **Why asked / trap:** The flags are orthogonal on paper and coupled in practice: which save path is safe depends on which load path you took.

## Level 3 — Advanced, Internals & Theory

**Q29. Why does Unsloth hand-write the LoRA backward, and what does it return as the gradient for the frozen 4-bit weight?**
- **Answer:** Autograd over a 4-bit base would require the dequantized weight to be a leaf with `requires_grad=True`, or a chain of ops from the quantized representation — which either doubles memory (holding bf16 master weights) or forces per-layer custom autograd anyway. Unsloth treats the LoRA path as a closed form `Y = XW_4bit + (α/r)·(X@A@B)` and writes the analytic gradients for `A`, `B` and the input `X` directly in a `torch.autograd.Function`. The gradient with respect to the frozen base is **not computed at all** — `W_4bit` is returned as `None` from `backward`, so torch never allocates a `.grad` for it. At 7B a full weight gradient in bf16 would be `7e9 × 2 B = 14 GB` (CS-16 §4.3.2–§4.3.4, §19 answer 3).
- **Why asked / trap:** "It skips the backward for the base" is the shallow answer. The precise answer — `None` is returned, no `.grad` is allocated, and that is a 14 GB line at 7B — is what separates a reader from a user.

**Q30. At `B=2, T=4096, V=32,000`, how big is the logits tensor in fp16, and what does the naive path hold that the fused kernel does not?**
- **Answer:** `2 × 4096 × 32,000 × 2 B = 524,288,000 B ≈ 524 MB` for the logits alone; the backward also needs the softmax probabilities, so the naive path holds roughly 2× that, ~1.05 GB (CS-16 §19 answer 4). At Llama-3's vocabulary, `2 × 4096 × 152,064 × 2 = 2,491,414,528 B ≈ 2.49 GB` per micro-batch — which is why long-context Llama-3 runs OOM at batch sizes that look absurd. Unsloth's fused cross-entropy **chunks over the vocabulary dimension**, computing the loss and its gradient in tiles that stay in SRAM/registers, never materialising the full `[B,T,V]` tensor (CS-16 §4.5.2).
- **Why asked / trap:** The `V=152,064` number is the point of the question. Candidates who quote the 32k figure and stop have not seen why the vocabulary is the hidden variable in long-context memory.

**Q31. How big is the attention score tensor, and why is "no materialisation" the long-context claim?**
- **Answer:** `B × H × T × T × 2 B = 2 × 32 × 4096 × 4096 × 2 = 2,147,483,648 B ≈ 2.15 GB` **per layer** in fp16 — and FlashAttention-2 writes none of it (CH-16 §10; CS-16 §4.4.1). It is the long-context claim because that term is `O(T²)` while every other activation term is `O(T)`: doubling the sequence length quadruples this one. The limit is that FA2's benefit depends on the head dimension and the kernel actually compiling for your GPU — a pre-Ampere card with no FA2 support removes it entirely (CS-16 §4.4.2).
- **Why asked / trap:** The follow-up is "so what is the real limit of the long-context claim?" — the answer is the hardware/kernel support matrix, not the algorithm.

**Q32. Explain the `bnb_4bit_compute_dtype` trap and the size of the effect.**
- **Answer:** In several `bitsandbytes` versions the default compute dtype for a 4-bit layer is **FP32**, so a naive QLoRA baseline dequantizes to fp32, does the matmul in fp32, and requantizes — 8–32× slower than it should be on tensor cores. Unsloth sets this from `dtype` for you, but *your HF baseline must set it too*, or you are comparing 4-bit-Unsloth against 4-bit-with-FP32-compute-HF, which is a **~1.7× speed difference on its own** (CS-16 §4.6.3, line 604; §6.5).
- **Why asked / trap:** This is the single biggest way to manufacture a fake 2× benchmark. It is also the #1 baseline bug in CH-16 §11 ("THE #1 BASELINE BUG" in the script's own pre-flight comment), and a candidate who has never hit it will not know to check `model.dtype` before timing.

**Q33. What exactly does naive QLoRA do per matmul that Unsloth avoids?**
- **Answer:** Naive: dequantize the NF4 block to the compute dtype → do the matmul → discard the dequantized copy; repeat for every forward, and the dequantized buffer traffic is proportional to the weight size times the number of passes. Unsloth keeps the computation in a form that avoids the round trip on the frozen base, so the 4-bit weights are never expanded into a full-dtype tensor just to be multiplied (CS-16 §4.6.1–§4.6.2; the script's `--help` intro, mechanism 4).
- **Why asked / trap:** The distinction the interviewer wants is *memory traffic*, not FLOPs — the matmul cost is identical; what changes is how many bytes move to do it.

**Q34. Where does the "2×" decompose? Give the attribution.**
- **Answer:** CS-16 §4.7.2's decomposition is that the headline is dominated by configuration the baseline did not have: compute dtype (Q32's ~1.7×), packing (2–5× throughput on short rows), FA2, and the 7-module target set versus `q,v`. Unsloth's own contribution — the kernels — is the 1.2–1.4× residual once those are held fixed, and CS-16 §4.7.3 notes the companion repo's own comparison is not apples-to-apples. CH-16 §1 row 2 and §10 both state the honest figure (CS-16 §4.7.1–§4.7.3).
- **Why asked / trap:** A candidate who can *decompose* the claim has understood it. One who repeats "1.2–1.4×" without knowing where the other 60% went has swapped one memorised number for another.

> **Correction:** `code/02_sft_unsloth.py`'s `--help` intro says "often 1.2-1.6x, and sometimes inside run-to-run noise", while CS-16 §4.7.2, CH-16 §1 row 2 and CH-16 §10 all give **~1.2–1.4×**. The script's upper bound (1.6) is the outlier. Use **1.2–1.4×** for the tuned-baseline delta and treat 1.6 as the favourable end of run-to-run variance, which is what the script's own next clause concedes.

**Q35. `save_pretrained_merged(save_method="merged_16bit")` — what does it do differently from peft's merge, precisely?**
- **Answer:** It dequantizes to bf16/fp16 *first*, through Unsloth's own controlled path, and then merges — so the NF4 double-quantization reconstruction happens once, under the library's own recipe, in the dtype the adapter was trained in. PEFT's generic path dequantizes as a side effect of the merge, in a dtype that may differ from the training compute dtype (CS-16 §6.10, §19 answer 6). CS-16 §18 item 11 and §19 answer 6 also document a `save_method="merged_4bit_forced"` for the genuinely-4-bit case, which is documented as lossy (CH-16 §5.4 path C).
- **Why asked / trap:** The pass/fail test is not "does it load" but the logit-equivalence check: max |Δlogit| < ~0.2 against base+adapter in the same dtype; > 1.0 means wrong scaling, wrong quantization or wrong dtype, and you re-merge. Never ship a merge you have not run this on (CS-16 §6.10).

**Q36. Why can a 100-step Unsloth run and a 100-step manual-backward HF run never agree bit-exactly?**
- **Answer:** Fused kernels reassociate floating-point sums (tiling changes the summation order), Triton autotunes tile shapes per hardware signature, and the hand-written backward computes the same mathematical gradient through a different sequence of operations. The documented tolerance is not bit-exactness but **worst 1−cos < 0.02** after 100 steps (CH-16 §10). The same property is why Unsloth is incompatible with a "bit-exact reproducibility" audit requirement (CS-16 §19 answer 10, §4.2.3).
- **Why asked / trap:** A candidate who expects bit-exactness will spend a week chasing a non-bug. The correct framing is a cosine tolerance on the adapter, not equality of weights.

**Q37. Why is the first benchmark step misleading, and how long does the effect last?**
- **Answer:** Triton compiles and autotunes kernels per signature on first call — roughly **2–20 s per signature** (CH-16 §10), cached in `~/.triton/cache`. A short benchmark is therefore dominated by compilation. Time steps 1–3 separately from steps 10–40 and report steady-state, and expect a stale `~/.triton/cache` after a torch upgrade to produce a traceback ending inside generated Triton code — the fix is `rm -rf ~/.triton/cache` (CH-16 §6, §11).
- **Why asked / trap:** It is one of the four plausible causes when someone's 2.8× reproduces at 1.3×, and the only one that is not a config difference (CS-16 §19 answer 9).

**Q38. Why does the speedup grow with `max_seq_length`?**
- **Answer:** Because the term Unsloth removes — the `O(T²)` attention materialisation — is the one that grows fastest. At short sequences the fixed per-step costs (kernel launch, optimizer, data loading) dominate and the delta is inside noise; at T=4096 the 2.15 GB-per-layer score tensor is the binding constraint, so removing it changes what is possible, not just what is fast (CS-16 §4.4.2, §2.3).
- **Why asked / trap:** This is why "Unsloth didn't help me" is so often a short-sequence report — and why CH-16 §10's long-context table (3k / 21k / 40k / 78k / 340k tokens at 8 / 12 / 16 / 24 / 80 GB) is labelled order-of-magnitude only.

**Q39. How does the script decide the model size used in its VRAM plan?**
- **Answer:** `size = sniff_size(model_id, "7B")` at line 156, from `code/common/memory.py`. `sniff_size` scans the model id for a preset label using `re.search(rf"(?<![0-9.]){label}(?![0-9])", haystack.lower())` over `sorted(MODEL_PRESETS, key=len, reverse=True)` — longest label first — and falls back to the supplied default if nothing matches. The size then drives every line of the plan (weights, optimizer state, activations, the FITS verdict), so a misdetection silently changes the whole budget.

> **Correction:** `sniff_size` misdetects the script's own default. `MODEL_PRESETS` contains a `"4B"` key, and the guard `(?<![0-9.])4b(?![0-9])` matches the `4b` inside the `-bnb-4bit` suffix — the preceding character is `-` and the following one is `i`, so both lookarounds pass. Verified in a shell: `sniff_size("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "7B")` returns **`"4B"`**, and so does `sniff_size("unsloth/Llama-3.1-8B-bnb-4bit", "8B")` — the bug is not specific to one size. `python 02_sft_unsloth.py --dry-run --data data/sample_sft.jsonl` therefore prints `4B | method=qlora | seq=2048 batch=2 x accum 4` and `weights 1.86 GB` / `TOTAL (1 GPU) 3.08 GB` for a model whose 4-bit weights are ~3.9 GB. **The plan is wrong; the `YES` FITS verdicts are unreliable.** Two independent fixes: strip a trailing `-bnb-4bit` / `-bnb-4bit-*` suffix from the haystack before matching, or match against the *basename* only — the size token is never inside a quantization suffix.

**Q40. Why is "4-bit weights, so 4× faster" wrong?**
- **Answer:** Weight *memory* is 4× smaller (0.5 B/param versus 2 B/param in fp16, or 4.5 bits measured with double quantization), but speed is bounded by compute and by memory *traffic*, not by storage. The dequantize-compute-requantize path means a naive 4-bit matmul can be *slower* than an fp16 one; the wins come from fitting a larger model or batch into the same VRAM, and from the kernel work in Q29–Q33. The honest ranges are the ones in Q2: ~1.2–1.4× on time against a tuned baseline.
- **Why asked / trap:** "4-bit means 4× less memory and 4× faster" is the single most common conflation in this area, and it is the same error as "8-bit AdamW means 8× cheaper" (IQ-10, IQ-11).

## Level 4 — System Design & Scenario

Answer in the fixed shape: **requirements → constraints → design → trade-offs → failure modes**.

**Q41. A 7B QLoRA SFT on a single 24 GB card. Sketch the configuration and justify each choice against the arithmetic.**
- **Answer:** `load_in_4bit=True` (weights `7e9 × 4.5/8 ≈ 3.9 GB`), `max_seq_length=2048` (~1.5 GB of KV/activation working set at batch 2), `r=16, lora_alpha=16, lora_dropout=0.0`, 7 target modules, `use_gradient_checkpointing="unsloth"`, `optim="adamw_8bit"`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`. Budget: 3.9 GB weights + ~0.3 GB adapter/grad/optimizer + 4–6 GB activations + <1 GB fused-CE working set + 1–2 GB CUDA context ≈ **9–11 GB peak** (CS-16 §19 answer 8). `r=16` is conservative; raise to 32 only if eval plateaus while train loss keeps falling.
- **Why asked / trap:** The graded part is the *justification*, not the values — every number must trace to a term. A candidate who sets `max_seq_length=4096` "for safety" without checking the token-length p99.5 has just doubled the activation budget for nothing.

**Q42. A team must fine-tune a 70B on 8×A100-80GB with a full audit trail and bit-exact reproducibility. Is Unsloth right? What do you recommend, and where does Unsloth still fit?**
- **Answer:** **No.** Three independent blockers: (a) no multi-GPU — a 70B on 8×A100 needs FSDP or tensor parallelism; (b) no bit-exact reproducibility — fused kernels with hardware autotuning and reassociated float sums cannot reproduce bit-identically (Q36); (c) patch fragility against a long-lived audited environment — the monkey-patches pin you to a narrow `transformers` window that is awkward to freeze for years. Recommend Axolotl (CS-17) with an FSDP config, or plain `accelerate`/`torchrun` + `peft` if the trail must be fully legible. Unsloth still fits the **single-GPU search phase**: find the data mix, sequence length, `r` and LR on a 7B proxy, then port the winning configuration (CS-16 §8.1, §10, §15.2, §16.6, §19 answer 10).
- **Why asked / trap:** The trap is answering only the first half. The design skill being tested is *where the tool belongs in a pipeline*, not whether it is good.

**Q43. A startup has one 4090 and a 7B, and needs a support assistant in two weeks. Walk the plan.**
- **Answer:** QLoRA on a pre-quantized 4-bit upload, all 7 projections, `r=16`, `max_seq_length` = p99.5 rounded to 64, assistant-only loss *verified by decoding the supervised span*, `packing=False` for the first run, a held-out eval set built before training. Ship the **adapter** and serve base + adapter (`vLLM --enable-lora`), not a merge — it keeps multiple adapters against one base and avoids the merge risk entirely. Expect a single-GPU iteration loop measured in minutes, which is the actual product of this setup (CS-16 §15.1, §16.3, §6.9).
- **Why asked / trap:** The interviewer is listening for the eval set and the mask verification, not the hyperparameters. A team that skips either ships a model that rambles or one that answers the wrong span.

**Q44. When is merging the right call over serving an adapter?**
- **Answer:** When the deployment target cannot load PEFT — llama.cpp / Ollama / LM Studio (merge then GGUF), or a serving stack you do not control. Merge to **FP16** (`merged_16bit`) and run the logit-equivalence test; do not merge to 4-bit unless size forces it, because that layers a second, independent quantization error on the base's (Q24, Q35). If you control the server, the adapter is strictly more flexible and cheaper to iterate (CS-16 §6.9, §16.3; script `--merge` help text).
- **Why asked / trap:** "Merge to save inference time" is a documented misconception (CS-16 §17.12) — merging changes the artifact, not the throughput, once the adapter is already fused at load time.

**Q45. A team has no budget and only Kaggle's free GPU. What changes?**
- **Answer:** Everything in the plan is now sized by a T4 (16 GB, no bf16, often no FA2) or a P100 (no bf16, no FA2). Set `fp16=True, bf16=False`, keep `max_seq_length` small, choose a 1B-class model, and expect the fused kernels' contribution to *shrink* because FA2 will not compile — leaving the LoRA backward and fused cross-entropy as the remaining wins. The throughput reference is **2,500–4,500 tok/s** for a 4-bit 1B at T=4096; the video's own free-T4 run measured **3,057 tok/s, 535 s, 1.9 GB peak**, costing 535/3600 = **0.149 GPU-hours** (CS-16 §15.3, §11.5; CH-16 §10).
- **Why asked / trap:** The trap is assuming the headline speedup survives on old hardware. It does not — the biggest single term (attention materialisation) needs FA2, which needs Ampere or newer.

**Q46. Why is Unsloth's real contribution to a project often the *cost of an experiment* rather than the speed of a run?**
- **Answer:** Because the binding constraint in most fine-tuning projects is the number of configurations you can afford to try, not the wall-clock of the one you settled on. A 0.149 GPU-hour run at ~$2.50/h is under 40 cents, so the search loop — data mix, sequence length, rank, LR — becomes cheap enough to actually run, and the winning configuration is then portable to whatever production stack the audit requirements dictate (CS-16 §11.5, §15.1, §19 answer 10; CH-16 §7.4).
- **Why asked / trap:** This is the question that separates "tool knowledge" from "project economics". The correct answer reframes the speedup as a *search-budget* multiplier.

## Level 5 — Debugging & Incident Response

For each: **what I look at first, what each observation rules in or out, and the fix.**

**Q47. Training loss sits at ~0.0 from step 1.**
- **Answer:** Labels equal inputs and the model is copying. Either the masking was inverted, or the collator was handed an unmasked dataset while `dataset_text_field` points at a pre-rendered `text` column. Check by decoding the supervised span (Q19) — if the loss is 0.0, the mask covers everything including the prompt. The expected band for a fresh LoRA on a pretrained base is `ln(V) × [0.7, 1.5]`, and `ln(32000) = 10.3735`, so **≈ 7.3 to 15.6** (CS-16 §6.5 diagnostic table).
- **Why asked / trap:** 0.0 looks like "learning fast" to a novice and is the loudest possible signal of a broken mask. The three-band diagnostic (0.0 = inverted, ~25+ = scrambled template, ~ln V = near-uniform logits) is the thing to memorise.

**Q48. Loss sits around 25 instead of ~10.**
- **Answer:** That is ~2.4× `ln(V)` at V=32k, which points at a scrambled template or the wrong tokenizer for the base — the model is being asked to predict tokens that do not follow from its pretraining distribution. Check: is `model_name` the *instruct* variant whose chat template you are rendering? Is the tokenizer the one that shipped with the checkpoint? (CS-16 §6.5, §4.10.)
- **Why asked / trap:** Novices read a high loss as "needs more training". Past ~1.5× `ln(V)` it is a data or template bug, and more epochs will only overfit the bug.

**Q49. Your A/B says Unsloth is 1.0× on a model you expected to be supported.**
- **Answer:** Two candidates, in order: (a) an **unsupported architecture** — `from_pretrained` never fails by design, so you loaded stock HF modules; assert `type(model.model.layers[0].self_attn).__module__` starts with `unsloth`. (b) `import unsloth` was not the first import, so the patches never bound (Q4). Only after both are ruled out should you look at `lora_dropout` (Q6) and the compute dtype (Q32).
- **Why asked / trap:** The temptation is to blame hardware. All three real causes are configuration and all three are detectable in under a minute.

**Q50. `CUDA out of memory` appears only at validation.**
- **Answer:** The eval batch is too large or eval runs too many steps. Set `per_device_eval_batch_size` to at most the train batch × 2, and evaluate fewer steps (CH-16 §11). The related production variant is an OOM at a step boundary rather than mid-forward, which means the optimizer state or a checkpoint save is doubling the live tensors momentarily.
- **Why asked / trap:** "It trains fine, so the config must be fine" is the wrong inference — eval has no gradient checkpointing and no accumulated-step amortisation.

**Q51. An assertion fires from inside `train_on_responses_only`.**
- **Answer:** The library's own check that the response marker was found in the tokenized inputs failed. This is a **marker fix, never a `force_match` bypass** (CH-16 §11): dump the rendered conversation with sentinel contents, derive the marker from what the template actually emitted (Q18), and re-run. A bypass converts a loud failure into a silent one — the mask matches nothing and you train on nothing.
- **Why asked / trap:** `force_match` is the trap. It exists, it makes the error go away, and it produces a run with no supervised tokens at all.

**Q52. The model answers correctly in training but rambles forever at inference.**
- **Answer:** The EOS token never made it into the supervised span — either the dataset's examples lack the turn terminator, or the response marker is positioned so that the trailing `<|eot_id|>`/`</s>` falls outside the mask. The model learns the answer but never learns to stop. The notebook's own comment states it: "EOS token is mandatory, otherwise generation may never stop." Fix in the data cell, and verify by checking that the decoded supervised span ends with the terminator (CS-16 §6.6, §4.9.3).
- **Why asked / trap:** The failure is invisible in the loss — a missing EOS is one token out of hundreds. It is caught only by reading the supervised span (Q19, check 4).

**Q53. `save_pretrained_merged(save_method="merged_16bit")` succeeded, but the merged model scores worse than base + adapter on your eval.**
- **Answer:** Run the merge-fidelity test before doing anything else: compute the same forward through (a) base + adapter in the merged model's dtype and (b) the merged model, and compare logits. Pass is `max |dlogit| < ~0.2` with the argmax agreeing; `> 1.0` means wrong scaling, wrong quantization or a dtype mismatch. Common causes: a `merged_4bit`-class save path used where 4-bit was not intended, a stale adapter directory, or an `α/r` mismatch between the training config and the merge (CS-16 §6.10).
- **Why asked / trap:** "The merge is lossy, live with it" is wrong — a correct 16-bit merge is within ~0.2 logits. A large drop is a bug, and the test is ten lines.

**Q54. `KeyError: 'instruction'` on a file you know contains instructions.**
- **Answer:** The loader defaulted to the Alpaca schema and your rows use a different one — `messages`, `sharegpt`, or `openai`. CH-16 §11 records this as a defect in `code/02_sft_unsloth.py`: the `--data` help advertises "alpaca/sharegpt/openai/completion", but **there is no CLI flag to change the loader**, so the fix is to convert the file to `instruction`/`input`/`output` upstream, or to pass a `messages`-shaped dataset through the library directly.
- **Why asked / trap:** The candidate should notice the help text is lying about a capability, not just that the key is missing. Read `--help`, then read the code.

## Rapid Fire — True / False / One-Liner

| # | Statement | Verdict |
|---|---|---|
| 1 | Unsloth is 2× faster than a tuned FA2 + bf16 QLoRA baseline. | **False** — ~1.2–1.4× time, 15–30% VRAM (CS-16 §4.7.2). |
| 2 | `use_gradient_checkpointing="unsloth"` is the same as `True`. | **False** — a string selects the selective variant (CS-16 §17.13). |
| 3 | Any non-zero `lora_dropout` silently disables the fast LoRA path. | **True** — and the script does not guard against it. |
| 4 | `merge_and_unload()` is safe on a 4-bit-loaded model. | **False** — use `save_pretrained_merged(..., "merged_16bit")` (CS-16 §6.10). |
| 5 | Unsloth replaces TRL's trainer. | **False** — it owns the fused loss and the LoRA backward only (CS-16 §17.3). |
| 6 | An unsupported architecture raises at load time. | **False** — it loads and runs at ~1.0× (CS-16 §8.2, §17.10). |
| 7 | The attention score tensor is never written to HBM. | **True** — 2.15 GB/layer at B=2, H=32, T=4096 is avoided entirely. |
| 8 | `max_seq_length` only truncates. | **False** — it shapes RoPE scaling and kernel selection (CH-16 §1 row 5). |
| 9 | A stale Triton cache can produce a traceback inside generated code. | **True** — `rm -rf ~/.triton/cache` (CH-16 §6, §11). |
| 10 | Unsloth scales to 8 GPUs with FSDP. | **False** — single-GPU; use Axolotl for multi-GPU (CS-16 §8.1). |
| 11 | A correct 16-bit merge should agree with base+adapter to ~0.2 in logits. | **True** — that is the pass/fail test (CS-16 §6.10). |

## Coding / Whiteboard Tasks

**Task 1 — find every silent-regression bug in this harness.** The candidate is given the block
below and asked to name each defect, its symptom, and the one-line fix.

```python
from unsloth import FastLanguageModel, train_on_responses_only
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=4096, dtype=None, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    use_gradient_checkpointing=True, random_state=3407,
)
trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds,
    args=SFTConfig(packing=True, per_device_train_batch_size=2, learning_rate=2e-5, num_train_epochs=2))
trainer = train_on_responses_only(trainer,
    instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
```

- **Answer — six defects.** (1) `lora_dropout=0.05` — silent fallback to peft's unfused path, 15–35% step-time loss, correct model (CS-16 §6.4). (2) `target_modules=["q_proj","v_proj"]` — two of seven; the parameter fraction collapses to ~0.3% and the sanity band (1.0% at r=16, 2.3% at r=32) never gets checked (CH-16 §4.4, §5.3). (3) `use_gradient_checkpointing=True` — the bool, not the string; stock GC, and the long-context saving is not the one you were sold (CS-16 §17.13). (4) `max_seq_length=4096` with no token-length check against the dataset's p99.5 — 2–4× the activation budget for a window most rows never reach (CS-16 §6.5). (5) `packing=True` during mask validation, so any marker bug appears as an interleaved mask rather than a cleanly wrong one (CH-16 §5.2). (6) `learning_rate=2e-5` — that is a full-FT LR; LoRA wants 1e-4…3e-4, and combined with `alpha=2r` (`α/r = 2`) the effective step is doubly small (CH-16 §4.3, §10).
- **The decisions being graded:** whether `lora_dropout` and `use_gradient_checkpointing` were caught (both are silent), whether the `q,v`-only target set was caught (it looks like a deliberate choice), and whether the candidate asked for the *dataset* before judging the LR.
- **Grading:** 5–6 defects with symptom + fix = strong. 3–4 = has used the library. 1–2 = has read about it. 0, or any answer that starts with "it will crash" = not a user; nothing here errors — that is the point.
- **Fail condition:** saying "it trains fine, these are just slower settings" without naming which ones silently lose the fast path.

**Task 2 — size a run from the tool's own output, then say whether to trust it.** The candidate is shown `python 02_sft_unsloth.py --dry-run --data data/sample_sft.jsonl` printing

```
  4B  |  method=qlora  |  optimizer=adamw  |  seq=2048  batch=2 x accum 4
  weights                     1.86 GB
  TOTAL (1 GPU)               3.08 GB
  FITS on 24 GB (4090/3090)                YES
```

and is told the model is `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, and asked to decide whether `--max-seq-len 4096 --batch-size 4` fits on a 4090.

- **Answer.** First, reject the premise of the printed plan: the header says `4B` for a 7B model, because `sniff_size` matched the `4b` inside `-bnb-4bit` (verified: `sniff_size("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "7B")` returns `"4B"`; the same call returns `"4B"` for an 8B id too). Every line is therefore a 4B budget: `weights 1.86 GB` should be `7e9 × 4.5/8 ≈ 3.9 GB`, so the true total is roughly **5.1 GB** not 3.08 GB — still a comfortable fit at `seq=2048, batch=2`, which is why the bug is survivable and therefore dangerous. Second, answer the actual question with the scaling that matters: `--max-seq-len 4096` doubles the sequence, and `--batch-size 4` doubles the micro-batch, so the activation term scales roughly **4×**; at `seq=2048, batch=2` activations were `0.47 GB` in the plan, so the new run needs ~1.9 GB of activations *plus* the corrected ~3.9 GB of weights and ~0.75 GB of adapter/grad/optimizer — call it **~7 GB**, so yes it fits on 24 GB, but not for the reason the tool printed. Third, note the two lines the plan does not show and that the candidate must supply: the dataset's p99.5 token length (is 4096 even needed?) and the 4×-larger effective batch's effect on LR.
- **The decisions being graded:** whether the candidate verified the plan against the model id instead of reading the ✅; whether both factors in the activation scaling were counted; and whether they asked for the token-length distribution before accepting 4096.
- **Grading:** catches the size bug **and** answers the scaling question with both factors = strong. Catches either alone = workable. Accepts `3.08 GB` and the `YES` = has not yet been burned by a tool that reports confidently.
- **Fail condition:** quoting the printed `TOTAL (1 GPU) 3.08 GB` as the answer for a 7B.

## Cheat Sheet of Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| Unsloth vs unconfigured HF | **2–4× time, 50–70% VRAM** | The tagline's baseline (CS-16 §4.7.1) |
| Unsloth vs tuned FA2+QLoRA+packing | **~1.2–1.4× time, 15–30% VRAM** | The number to quote (CS-16 §4.7.2) |
| `bnb_4bit_compute_dtype` unset | **~1.7× slowdown** | FP32 compute in a 4-bit layer (CS-16 §4.6.3) |
| bytes/trainable param, QLoRA adapter | **6 B** | 2 adapter + 2 grad + 2 adamw_8bit |
| 7B 4-bit weights | `7e9 × 4.5/8 ≈` **3.9 GB** | The dominant VRAM term |
| 7B bf16 full weight gradient | `7e9 × 2 =` **14 GB** | Never allocated — `None` is returned (CS-16 §19 A3) |
| LoRA fraction bands (7 modules, 1.1B) | **~1.0% / 2.3% / 4.5%** | r = 16 / 32 / 64 |
| TinyLlama r=32 trainable params | **25,231,360 = 2.294%** | `25231360/1100048384` |
| Attention score tensor | **2.15 GB/layer** | B=2, H=32, T=4096, fp16 — never materialised |
| Logits tensor, V=32k | **524 MB** | `2×4096×32000×2 B` |
| Logits tensor, Llama-3 V=152,064 | **2.49 GB** | Why long-context Llama-3 OOMs |
| Expected init loss | **`ln(V)` × [0.7, 1.5] ≈ 7.3–15.6** | `ln(32000) = 10.3735`; 0.0 = inverted mask |
| `lora_dropout` | **0.0** exactly | Or the fast path is gone |
| `use_gradient_checkpointing` | **`"unsloth"`** | A string, not `True` |
| Standard GC cost | **~25–35% step time, −60–75% activations** | CH-16 §10; see the §12 note |
| Triton first-call compile | **~2–20 s per signature** | Cache is `~/.triton/cache` |
| Merge fidelity threshold | **max abs Δlogit < ~0.2** | Pass/fail for `save_pretrained_merged` |
| Adapter equivalence threshold | **worst 1−cos < 0.02** after 100 steps | Unsloth vs HF manual backward |
| Packing win, Alpaca-shaped rows | **2–5× throughput** | Rows are ~150–250 tokens in a 4096 window |
| Reference throughput, free T4 | **3,057 tok/s measured** | 1B 4-bit @ T=4096; band 2,500–4,500 |
| Video's measured run | **535 s, 1.9 GB peak, 0.149 GPU-h** | `535/3600`; $0.02–0.06 |

> **Correction:** CS-16 §19 answer 8 states the 7B 4-bit weight figure as `7e9 · 4.5/8 ≈ 3.9 GB`, while CS-16 §11.1 and CH-16 §10 use **0.5 B/param** (which gives `7e9 × 0.5 = 3.5 GB`, and `1,100,048,384 × 0.5 = 0.550 GB` for TinyLlama). The two differ because 4.5 bits is the *measured* NF4-plus-double-quantization rate while 0.5 B/param is the nominal 4-bit rate. **Both are correct at their own precision**; use 4.5 bits when you want the measured number and 0.5 B/param for a one-line estimate, and never mix them in the same budget. Watch the units: CS-16 §11.1 states 0.62 **GB** for TinyLlama's NF4 weights and is right (`1,100,048,384 × 4.5/8 = 618,777,216 B = 0.619 GB`), but CH-16 §2's formula row restates the identical figure as **0.62 GiB** while dividing by `1024³` — and `618,777,216 / 1024³ = 0.576 GiB`. The formula and the value in that row disagree by exactly the GB/GiB factor; quote **0.62 GB or 0.58 GiB** and never 0.62 GiB.

## Answers To The Self-Check Questions From CS-16

**1. What is the baseline behind "2× faster / 70% less memory", and what is it against a tuned run?** An unconfigured `transformers + peft + bnb` script — eager attention, no packing, often FP32 compute dtype, `q,v`-only LoRA, stock checkpointing; against a tuned FA2 + bf16 + packing + 7-module run the honest delta is ~1.2–1.4× time and 15–30% VRAM. — **trap:** quoting the headline without the baseline is the answer the module is designed to catch (see Q2).

**2. Five things Unsloth fuses or eliminates, and the term each removes.** Attention (`O(T²)` scores), cross-entropy (the `[B,T,V]` logits plus softmax intermediate), RMSNorm (fp32 intermediates), RoPE (cos/sin tables and rotated intermediates), SwiGLU MLP (two intermediates, three kernels → one); plus the LoRA backward, which removes the autograd graph over the frozen base. — **trap:** "FlashAttention" alone is one of six, not the answer (CS-16 §19 A2).

**3. Why hand-write the LoRA backward, and what is the gradient for the frozen weight?** Autograd over a 4-bit base would need a `requires_grad` dequantized leaf or a per-layer custom function anyway; the analytic path writes `dA`/`dB`/`dX` directly and returns **`None`** for `W_4bit`, so no `.grad` is allocated — 14 GB of savings at 7B. — **trap:** saying "it skips the backward" without the `None`/no-allocation detail (Q29).

**4. Logits memory at `B=2, T=4096, V=32,000` in fp16, and at `V=152,064`?** `2×4096×32000×2 = 524 MB` (naive path holds ~2× that with the softmax probabilities); at Llama-3's vocabulary `2,491,414,528 ≈ 2.49 GB`. The fused kernel chunks over the vocabulary and never materialises the full tensor. — **trap:** quoting only the 32k number and missing that the vocabulary is the hidden variable (Q30).

**5. Why must `lora_dropout` be `0.0`, and what happens at `0.05`?** The fused LoRA kernel is hand-written for the deterministic path; stochastic masking needs a second kernel or a saved mask, so the library detects non-zero dropout and silently falls back to peft's unfused implementation — correct model, 15–35% slower steps, no warning. `code/02_sft_unsloth.py` exposes `--lora-dropout` without a guard, so the script does not protect you. — **trap:** expecting a warning or an error; the fast path is lost silently (Q6, Q17).

**6. `load_in_4bit=True` plus `merge_and_unload()` — what goes wrong, and what should you call?** PEFT must dequantize NF4 first, and NF4's double quantization makes that reconstruction lossy, in a dtype that may differ from training; the model loads, generates fluently, and is measurably worse with no error. Call `save_pretrained_merged(dir, tokenizer, save_method="merged_16bit")`, then run the `max |Δlogit| < 0.2` test. — **trap:** trusting "it loaded and generated text" as evidence the merge was correct (Q10, Q35).

**7. How does `train_on_responses_only` find the assistant span, and how do you detect a bad template in under a minute?** It renders the chat template, string-searches for `response_part`, re-tokenizes the prefix and sets `labels[:prefix_len] = -100`. Detect by decoding the tokens where `labels != -100` for `dataset[0]`: you must see exactly the assistant's reply and its terminator. — **trap:** a marker that appears *inside* the instruction text, which offsets the boundary silently (Q18, Q19).

**8. Sketch a 7B QLoRA config on 24 GB.** `max_seq_length=2048`, `load_in_4bit=True`, `r=16, lora_alpha=16, lora_dropout=0.0`, 7 target modules, `use_gradient_checkpointing="unsloth"`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, `optim="adamw_8bit"` — ~9–11 GB peak, because 3.9 GB of it is the 4-bit weights. — **trap:** raising `max_seq_length` "for safety" without checking the token-length p99.5 (Q41).

**9. Their benchmark says 2.8×; yours says 1.3×. Four plausible causes.** Different baseline configuration (print both `LoraConfig` and `BitsAndBytesConfig`), different timed scope (compare step counts actually run), packing or FA2 enabled in one arm only (log tokens/sec, not steps/sec), and first-step Triton compilation (time steps 1–3 separately from 10–40). — **trap:** comparing step time between arms whose packing differs, which changes tokens per step by 2–4× (CS-16 §19 A9; Q37).

**10. Is Unsloth right for a 70B on 8×A100-80GB with bit-exact reproducibility?** No — no multi-GPU, no bit-exact fused kernels, and patch fragility against a long-lived audited environment. Use Axolotl with FSDP (or plain `accelerate`/`torchrun` + `peft`), and keep Unsloth for the single-GPU 7B/8B configuration search whose result you port. — **trap:** answering "no" and stopping; the graded half is *where it still fits* (Q42).

## Cross-References

| Relationship | Module | Why |
|---|---|---|
| **Builds on** | **CS-13 / CH-13** — Instruction Fine-Tuning | The dataset schema, chat template, assistant-only masking concept and eval protocol are defined there; Unsloth changes how fast that pipeline runs, not what it does (CS-16 §20). |
| **Builds on** | **CS-06 / CH-06** — Hugging Face Masterclass | Unsloth is an extension of `transformers` + `peft` + `trl`; nothing here makes sense without the `AutoModel`/`Trainer`/`datasets` model. |
| **Builds on** | **CS-01 §4.9** — Data quality (dedup, filtering, contamination, licensing) | Packing, the p99.5 sequence-length decision, and dataset fingerprinting for lineage (CS-16 §16.2) are data-engineering concerns. (**CS-05 / CH-05** are *RNN/LSTM → Attention*, not data preparation) |
| **Builds on** | **CS-10 / CH-10 and CS-11 / CH-11** — Quantization I and II | NF4, double quantization, blockwise absmax, the dequantize → compute → requantize round trip, and GPTQ/AWQ/GGUF export are defined there; this module's §4.6 and §6.9 are their consequences. |
| **Parallel to** | **CS-13 §6.8** / **CS-11 §4.11** — LoRA & QLoRA | The adapter mathematics. Unsloth optimises the LoRA graph; those sections derive it — read them first if `dA`/`dB` in §4.3 look unfamiliar. (**CS-14** is *The Alignment Map*, not PEFT; the **CS-23** PEFT deep dive is planned, not yet written) |
| **Contrasts with** | **CS-17** — Axolotl | The nearest alternative and the natural next step once the single-GPU ceiling binds: FSDP, YAML configs, a broader method menu. CS-16 §8.1, §15.2 and §19 answer 10 pick between them. |
| **Contrasts with** | **CS-15** — LLaMA-Factory | Both wrap a training stack; LLaMA-Factory is config-driven and multi-method/multi-GPU, Unsloth is code-driven, single-GPU and kernel-level. |
| **Contrasts with** | **CS-18 / CS-19** — OpenAI / Vertex fine-tuning | Hosted APIs: zero ops, no kernel access, no adapter artefact, no data residency control. CS-16 §16.6 is the compliance comparison. |
| **Needed by** | **CS-13 §12** / **CS-14 §12** — Evaluation (*How To Know It Worked*) | The benchmarking discipline in CS-16 §12 (warmup, matched step counts, tokens/sec not steps/sec, config diffing) is the general version of what §4.7.2 demands. (**CS-04** is *Fine-Tuning vs RAG vs Agents*, not evaluation) |
| **Sibling banks** | **IQ-10, IQ-11, IQ-13** | IQ-10/IQ-11 own quantization; IQ-13 owns SFT practice. This bank owns the Unsloth-specific layer over both. |

*End of IQ-16. Companion artifacts: `case-studies/CS-16-Unsloth.md`, `cheat-sheets/CH-16-Unsloth.md`, `code/02_sft_unsloth.py`.*
