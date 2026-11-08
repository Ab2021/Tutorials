---
collection: "python-sql-linux-engineering-basics"
file_id: "03"
topic: "LLM fundamentals"
level: 1
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-07"
reviewed: null
source_status: "mapped source title; original body unavailable"
---

# LLM Fundamentals

> **Source mapping:** `Agents_Stack_Infra/Module_00_Prerequisites/03_llm_fundamentals.md` in [`Train/Templates/directory.txt`](../../../../Templates/directory.txt). Original source body unavailable; new mapped draft.

## 1. Core contract

A language model maps a token prefix to a distribution over a tokenizer vocabulary. Generation repeatedly selects a token under a declared decoding policy, appends it, and stops at a declared termination condition. It predicts tokens; it does not retrieve authoritative facts, execute tools, retain durable state or validate outputs unless surrounding systems provide those capabilities.

## 2. Probability and loss

For a target sequence, autoregressive probability factorizes as `P(x_1...x_T)=product_t P(x_t|x_<t)`. Mean negative log-likelihood is `-sum log p(target)/T` over the declared scored tokens. If target probabilities are `.5` and `.25`, mean NLL is `-(ln .5+ln .25)/2=1.03972` nats/token. This checks objective fit under a tokenizer/mask; it does not establish task quality, factuality or safety.

## 3. Runtime boundaries

| Component | Responsibility | Not guaranteed |
|---|---|---|
| Tokenizer | Text/token mapping | Semantic equivalence across tokenizers |
| Model | Next-token distribution | Truth or tool execution |
| Prompt/context | Conditioning evidence/instructions | Authorization or durable memory |
| Decoder | Token selection/stopping | Determinism unless configured |
| Retriever/tool | External facts/actions | Correctness without validation |
| Validator | Schema/policy/business check | Model capability |

Temperature/top-p, stop sequences, tool schemas, context length and prompt/template alter output behavior. Pin them during evaluation. A model version name alone is not a reproducible inference contract.

## 4. Failure modes

| Symptom | Check | Response |
|---|---|---|
| Confident unsupported answer | Evidence/citation and abstention audit | Add retrieval/validation and grounding gate |
| Schema-invalid output | Raw decoding/config and validator trace | Constrain/validate/retry under bounded policy |
| Context overflow | Token count, truncation and position policy | Budget/select context; disclose loss |
| Different output after upgrade | Tokenizer/model/template/decoder diff | Pin versions and rerun evaluation |

## 5. Validation

Test target task, unsupported/no-answer behavior, schema/tool-call validity, prompt injection boundaries, relevant slices and operational cost/latency. A generic benchmark or lower NLL alone cannot promote a product behavior.

## 6. Interview checks

1. **What does an LLM output?** A conditional token distribution.
2. **What makes a response reproducible?** Model/tokenizer/template/decoder/tool/context versions.
3. **Why is NLL insufficient?** It is not task, safety or factuality evidence.
4. **What supplies durable memory?** An explicit state store with ownership policy.
5. **What validates an action?** An external authorization/validator contract.

## 7. Quick reference

| Claim | Required evidence |
|---|---|
| Model answers task | Held-out task/grounding/safety evaluation |
| Output valid | Schema/business validator and failure tests |
| Reproducible | Full inference contract/version manifest |

**Limitations:** no model, benchmark or source lesson was available. Arithmetic is illustrative.
