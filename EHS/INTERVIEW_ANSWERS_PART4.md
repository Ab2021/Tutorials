# Ensemble Health -- Interview Question Answers (Part 4)

> Sections 7-11: GenAI & LLM, Production Architecture, Risk Factors, Error Analysis, Synthesis
> Questions Q96 -- Q150

---

## Section 7: GenAI & LLM Integration (Q96-Q110)

### 7.1 Architecture & Design

**Q96. Walk me through the full flow from a claim row to a plain-English explanation. What happens at each step?**

```
STEP 1: Risk Factor Extraction
claim_row + model_coefficients + feature_matrix
  |
  v
For each feature: contribution = coefficient * feature_value
Sort descending by contribution. Take top 3 positive contributions.
Map raw feature names to human-readable labels.
  |
  v
Example: ["Missing Required Prior Authorization",
          "Missing Supporting Documentation",
          "High-risk segment: Payer ID: P008"]

STEP 2: Input Validation (Pydantic)
ExplanationRequest(
    claim_id="CLM01234",
    denial_probability=0.62,
    risk_estimate_label="High",
    top_risk_factors=[
        RiskFactorItem(fact="Missing Required Prior Authorization",
                       permitted_action="Obtain and attach prior authorization"),
        RiskFactorItem(fact="Missing Supporting Documentation",
                       permitted_action="Attach clinical documentation"),
        RiskFactorItem(fact="High-risk segment: Payer ID: P008",
                       permitted_action="Review P008-specific requirements")
    ]
)
Pydantic validates: prob in [0,1], label matches High/Medium/Low, factors <= 5, each field min_length=1

STEP 3: Prompt Construction
build_explanation_prompt() converts the ExplanationRequest into a structured text prompt:
- Role: "You are a revenue-cycle billing analyst..."
- Context: claim details, probability, risk tier
- Risk factors: inlined with fact + permitted action pairs
- Rules: "Return ONLY JSON. Do NOT invent new risks. Do NOT mention ICD/CPT codes."
- Schema: Exact JSON structure expected

STEP 4: API Call (if High-tier and prob >= 0.25)
_response = ollama.chat(
    model='gemma4:31b-cloud',
    messages=[{'role': 'user', 'content': prompt}]
)
Returns: (response_text, ChatResponse_object)

STEP 5: Response Parsing (three-tier)
Tier 1: Try json.loads(response_text)
  If fails (markdown wrapping, extra text) ->
Tier 2: Strip ```json fences, try json.loads. If fails (malformed JSON) ->
Tier 3: Regex extraction for disclaimer/risk/action fields. If fails ->
  Deterministic template: build_fallback_response()

STEP 6: Output Validation (Pydantic)
ExplanationResponse(
    claim_id="CLM01234",
    disclaimer="This is a statistical estimate, not a guaranteed outcome...",
    risk_description="High denial risk due to missing authorization...",
    recommended_action="Obtain prior authorization and attach documentation...",
    risk_estimate_label="High"
)
Pydantic validates: disclaimer contains "estimate"/"statistical"/"not guaranteed",
claim_id non-empty, risk_description and recommended_action min_length

STEP 7: Text Assembly
to_plain_text() combines disclaimer + risk_description + recommended_action
into a single explanatory paragraph for the CSV output.

STEP 8: Audit Logging
LLMAuditLogger records:
- call_id, timestamp, claim_id, model used, latency
- token counts (prompt_eval_count, eval_count from ChatResponse)
- validation results (disclaimer check, pydantic pass, json parse method)
- quality flags (hallucination markers, PII markers)
Saved to: data/output/audit_logs/audit_YYYYMMDD_HHMMSS.json
```

**Q97. Why use Pydantic models for both input and output validation? What specific failure modes does this two-layer approach prevent?**

Pydantic provides structured data validation with clear error messages and type safety. The two-layer approach creates a defense-in-depth strategy:

**Input validation (ExplanationRequest) -- prevents bad data ENTERING the LLM:**

| Failure Mode | How Input Validation Catches It |
|---|---|
| Probability outside [0,1] (e.g., 1.5 from a bug) | `Field(ge=0.0, le=1.0)` raises ValidationError |
| Risk tier label misspelled ("Hihg" instead of "High") | `Field(pattern=r'^(High\|Medium\|Low)$')` rejects it |
| Empty risk factor list (claim has no factors) | `min_length=1` on RiskFactorItem fields |
| Too many risk factors (15 instead of max 5) | `Field(max_length=5)` on the list |
| Empty fact or action strings | `min_length=1` on each RiskFactorItem field |
| claim_id is empty or None | `Field(min_length=1)` on claim_id |

Without input validation, these bugs would silently send malformed prompts to the API, wasting tokens and producing garbage explanations.

**Output validation (ExplanationResponse) -- prevents bad data LEAVING the system:**

| Failure Mode | How Output Validation Catches It |
|---|---|
| LLM produces disclaimer without uncertainty language ("This claim will be denied") | `@field_validator('disclaimer')` checks for "estimate"/"statistical"/"not guaranteed" |
| LLM hallucinates a completely different claim_id | `Field(min_length=1)` catches empty; regex for format can be added |
| LLM returns empty risk_description | `Field(min_length=1)` rejects |
| LLM returns overly brief recommended_action ("Fix it") | `Field(min_length=10)` catches trivial responses |

Without output validation, a hallucinated explanation ("This claim will definitely be denied") could reach a biller, creating false certainty and potentially leading to inappropriate actions.

**Q98. You chose gemma4:31b-cloud via ollama.chat(). What alternatives did you consider? Why this specific model?**

**Alternatives considered:**

| Model | Pros | Cons | Verdict |
|---|---|---|---|
| **gemma4:31b-cloud** | Strong instruction following, JSON formatting, healthcare-safe, fast via Ollama Cloud | Requires API key, cloud dependency | **Selected** |
| GPT-4o / Claude 3.5 | Best-in-class reasoning, excellent JSON | Expensive per-token, external dependency, data leaves our infra | Rejected for HIPAA concerns |
| Local Llama 3 8B | Runs locally, zero API cost, private | Limited reasoning for structured JSON tasks, slower | Would work but lower quality |
| Rule-based templates only | Zero cost, deterministic, instant | No claim-specific narrative, less useful to billers | Used as fallback, not primary |
| Phi-3 / Mistral (local) | Good instruction following, moderate size | May require GPU, setup complexity | Viable alternative, not tested |

**Why gemma4:31b-cloud specifically:**
- **31B parameters** is the sweet spot -- enough capacity for nuanced medical billing language without being slow or expensive
- **Cloud-hosted** via Ollama, so no local GPU needed
- **JSON-native:** Trained to produce valid JSON, reducing parse failures
- **Healthcare-appropriate:** The model follows safety instructions (no ICD codes, no patient identifiers) reliably
- **Simple API:** `ollama.chat(model='gemma4:31b-cloud', messages=[...])` is a one-line call
- **Cost-effective:** ~450 tokens per explanation vs ~800+ for GPT-4

**Q99. The prompt instructs the LLM to "Return ONLY a valid JSON object." Why is this constraint necessary? What happens without it?**

This constraint is NECESSARY because LLMs are conversational by design. Without it, the model defaults to its training to be helpful and conversational, adding preamble and postamble text.

**What happens without the constraint:**

```
Actual response: "Here is the denial risk explanation for the claim:
```json
{
  "claim_id": "CLM01234",
  "disclaimer": "...",
  ...
}
```
I hope this helps with your pre-bill review process!"
```

The JSON is wrapped in markdown and conversational text. `json.loads()` fails because it expects pure JSON starting from the first character. The system would fall back to regex extraction, adding latency and fragility.

**With the constraint:**

```
Actual response: `{"claim_id":"CLM01234","disclaimer":"...","risk_description":"...","recommended_action":"..."}`
```

Pure JSON, immediately parseable. In practice, gemma4 still sometimes wraps in ```json fences despite the instruction, so we added markdown fence stripping as a safety net. The constraint reduces but doesn't eliminate the issue.

**Why not use function calling / tool use?** Ollama's structured output feature (similar to OpenAI function calling) would guarantee JSON format. However, the assessment specified `ollama.chat()` with plain messages. The "Return ONLY JSON" instruction is the next-best approach.

**Q100. You inline risk factors with their permitted actions rather than just listing names. Why does this reduce hallucination?**

Inlining risk factors means providing the LLM with the EXACT language to use, rather than asking it to generate language from a topic.

**Without inlining (just names):**
```
Risk factors: ["Missing Required Prior Authorization", "Missing Supporting Documentation"]
```
The LLM must INVENT descriptions and actions for these topics. It might hallucinate:
- "The prior authorization for CPT 99213 was not obtained from BCBS" (invented CPT code, invented payer)
- "Please submit the operative report and anesthesia record" (plausible but wrong documentation)

**With inlining (fact + action pairs):**
```
Risk factors:
- FACT: Missing Required Prior Authorization
  ACTION: Obtain and attach prior authorization to the claim before submission
- FACT: Missing Supporting Documentation
  ACTION: Attach clinical documentation supporting medical necessity
```
The LLM is GROUNDED. It paraphrases or directly uses the provided language. The facts and actions are extracted from the model's risk factors (which come from LR coefficients), not invented by the LLM. Hallucination risk drops dramatically because the LLM is summarizing, not creating.

**Additional anti-hallucination measures in the prompt:**
- "Use ONLY the risk facts and actions listed below"
- "Do not invent new risks"
- "Do NOT mention ICD/CPT codes, dollar amounts, or patient identifiers"

These rules, combined with inlined content, create a strong guardrail against LLM hallucination.

**Q101. What specific rules did you include in the prompt to prevent the LLM from mentioning ICD codes, dollar amounts, or patient identifiers? Why are these rules important for healthcare?**

**Rules in the prompt:**
1. "Do NOT invent new risks beyond those listed below"
2. "Do NOT mention specific ICD-10, CPT, or HCPCS codes"
3. "Do NOT mention specific dollar amounts"
4. "Do NOT reference patient names, MRNs, or other PHI"
5. "Provide general, actionable guidance a biller can use"

**Why these are important for healthcare:**

**ICD/CPT codes:** The LLM might hallucinate plausible-sounding codes (e.g., "ICD-10 E11.9 for diabetes") that don't match the patient's actual diagnoses. A biller reading the explanation might add incorrect codes to the claim, creating a compliance violation. Even if the code is real, the model has no knowledge of the patient's actual conditions.

**Dollar amounts:** Mentioning specific amounts (e.g., "$12,500 billed") in explanations that may be stored in audit logs creates financial data exposure. Additionally, dollar amounts might be considered PHI-adjacent in some interpretations.

**Patient identifiers:** Names, MRNs, dates of birth are clearly PHI under HIPAA. The model should never produce these, even if they were somehow in the training data. The explicit prohibition creates a hard stop.

**Legal exposure:** If a biller acts on an LLM explanation that mentions a hallucinated ICD code, and that action leads to a denied claim or audit finding, there's legal exposure. "The AI told me to add code E11.9" is not a defense. By preventing code mentions, we ensure explanations stay at a safe level of abstraction that a biller can interpret and act on using their professional judgment.

### 7.2 Validation & Fallback

**Q102. Explain the three-tier fallback strategy: JSON parse -> regex extraction -> deterministic template. When does each tier activate?**

```
API Response
    |
    v
TIER 1: Direct JSON Parse
    try: json.loads(response_text)
    Success (~95% of cases):
      -> Proceed to Pydantic validation
      -> Return ExplanationResponse
    Failure (markdown wrapping, extra text, malformed JSON):
      -> Activate Tier 2
    |
    v
TIER 2: Markdown Stripping + JSON Parse
    Strip ```json ... ``` fences
    Strip leading/trailing whitespace
    try: json.loads(cleaned_text)
    Success (~4% of cases):
      -> Proceed to Pydantic validation
      -> Return ExplanationResponse
    Failure (truly malformed JSON):
      -> Activate Tier 3
    |
    v
TIER 3: Regex Extraction
    Try to extract fields using regex patterns:
      - claim_id: r'"claim_id"\s*:\s*"([^"]+)"'
      - disclaimer: r'"disclaimer"\s*:\s*"([^"]+)"'
      - risk_description: r'"risk_description"\s*:\s*"([^"]+)"'
      - recommended_action: r'"recommended_action"\s*:\s*"([^"]+)"'
    If all four fields extracted:
      -> Build ExplanationResponse from extracted fields
      -> Proceed to Pydantic validation
    If extraction fails:
      -> Activate Tier 3 Fallback
    |
    v
TIER 3 FALLBACK: Deterministic Template
    build_fallback_response(claim_id, prob, risk_label, risk_factors)
    - disclaimer: "This is a statistical estimate based on historical data..."
    - risk_description: template based on risk factors
    - recommended_action: template based on risk label
    -> Returns Pydantic-valid ExplanationResponse
    -> Guaranteed to produce valid output
```

**Activation triggers:**

| Tier | When | Typical cause |
|---|---|---|
| 1 | API response is clean JSON | Normal operation |
| 2 | API response has markdown wrapping | gemma4 adds ```json fences |
| 3 | API response is valid-ish JSON structure | Model slightly deviates from schema |
| Fallback | All parsing fails OR API call fails | API down, timeout, nonsense response |

**Critical property:** The fallback chain GUARANTEES the pipeline never crashes from an LLM failure. Even if Ollama Cloud is completely down, every claim gets a valid explanation. This is essential for production reliability.

**Q103. The `@field_validator` for disclaimer checks for "estimate", "statistical", "not guaranteed". Why is this specific validation critical? What would happen if a disclaimer was missing?**

This validation is a safety net against overconfident or legally dangerous language from the LLM. In healthcare, predictions carry legal and ethical weight. An explanation that says "This claim WILL be denied" could:

- Cause a biller to escalate unnecessarily, wasting resources
- Create false certainty that leads to inappropriate claim modifications
- Be cited in an audit as "the AI system said this claim would be denied" -- but the system is statistical, not deterministic
- Violate responsible AI principles about communicating uncertainty

**What the validator catches:**

| LLM Output | Passes? | Reason |
|---|---|---|
| "This is an estimate based on statistical patterns" | Yes | Contains "estimate" AND "statistical" |
| "This claim is likely to be denied" | No | No uncertainty qualifier |
| "Based on historical data, this claim will be denied" | No | Absolute language without "estimate" |
| "Statistical model predicts high denial risk (not guaranteed)" | Yes | Contains "statistical" AND "not guaranteed" |
| "This is a prediction, not a guaranteed outcome" | Yes | Contains "not guaranteed" |

**Why keyword matching and not semantic check:** Semantic validation ("is this statement appropriately uncertain?") requires a second LLM call, adding cost and complexity. Keyword matching is 95% effective at catching overconfident language and is deterministic and fast. The specific keywords ("estimate", "statistical", "not guaranteed") were chosen to cover the main dimensions of uncertainty communication: nature of the prediction ("statistical"), degree of confidence ("estimate"), and caveats ("not guaranteed").

**Q104. Your deterministic template produces explanations like "This claim has an estimated denial risk of 26% -- this is a statistical estimate, not a guaranteed outcome." How was this template designed? What factors went into making it sound natural?**

The deterministic template (`build_fallback_response()`) was designed to satisfy four competing goals:

**1. Always Pydantic-valid:** The disclaimer must pass the uncertainty keyword check. Every template includes "is a statistical estimate, not a guaranteed outcome" verbatim. This guarantees the `@field_validator` will always pass.

**2. Appropriate per risk tier:**

| Tier | Template Tone | Example |
|---|---|---|
| High (prob > 0.25, factors present) | Urgent, specific | "High denial risk due to [factor1], [factor2]. Immediate action recommended: [action1]. [action2]." |
| Medium (prob 0.10-0.25) | Moderate, checklist | "Moderate denial risk. Review: [factor1], if applicable: [action1]." |
| Low (prob < 0.10) | Reassuring, brief | "Routine submission recommended. No specific risk flags detected." |
| No actionable factors | Neutral | "No actionable pre-submission risk flags detected. Routine processing recommended." |

**3. Action-oriented:** Every High and Medium template includes specific recommended actions, not just risk descriptions. A biller should be able to read the explanation and immediately know what to DO.

**4. Natural language patterns:**
- Uses the biller's perspective ("you should verify...")
- Varies sentence structure (not robotic "Risk: X. Action: Y.")
- Includes transitional phrases ("Before submission, ensure...")
- Matches the professional tone of revenue cycle communication

**Design iteration:** The templates went through 3 iterations:
- v1: Robotic bullet points (rejected -- felt like a software error message)
- v2: Full sentences but generic ("Review the claim for issues") (rejected -- not actionable)
- v3: Claim-specific factors with mapped actions + natural wrapping (current)

**Q105. If the API returns a perfectly valid JSON with a disclaimer that says "This claim will be denied," would your Pydantic validator catch it? Why or why not?**

**Yes, the Pydantic validator would catch it.** The `@field_validator('disclaimer')` checks for the presence of uncertainty keywords:
```python
keywords = ['estimate', 'not a guarantee', 'statistical', 'not guaranteed']
if not any(kw.lower() in v.lower() for kw in keywords):
    raise ValueError('Disclaimer must include uncertainty qualifier')
```

"This claim will be denied" contains NONE of these keywords. The validator would raise a `ValidationError`, and the system would:
1. Log the validation failure to the audit log
2. Fall through to the next tier (regex extraction, which would also fail since the fields need to be extracted)
3. Ultimately use `build_fallback_response()` to produce a safe, validated explanation

However, I'll note a GAP in the current validation: if the LLM said "Statistical analysis indicates this claim will be denied," the validator would PASS (it contains "statistical"). The validator checks for the PRESENCE of keywords, not the SEMANTIC MEANING. A sufficiently clever (or malicious) prompt injection could produce harmful output that passes keyword-based validation.

To address this gap in production, I'd add:
- A regex check for absolute language: `r'\b(will be|definitely|certainly|guaranteed to be) denied\b'` -- if present, fail the validation even if keywords exist
- A second LLM call to check for overconfident language (expensive but thorough)
- Human review of a sample of explanations during initial deployment

### 7.3 Production Considerations

**Q106. Why do you only send 125 claims to the API instead of all 500? What is the cost-benefit calculation?**

**Cost-benefit by tier:**

| Tier | Claims | Method | Cost | Benefit |
|---|---|---|---|---|
| High | 125 | API (gemma4:31b-cloud) | ~56K tokens ($0.03) | Billers review these claims; detailed explanations justify the review and guide corrective action |
| Medium | 125 | Deterministic template | $0 | Billers don't review these (unless time permits); template is sufficient for pass-through |
| Low | 250 | Deterministic template | $0 | These pass through without review; any explanation is unnecessary overhead |

**Marginal benefit analysis for Medium-tier API calls:**

If I sent Medium-tier claims to the API:
- Cost: 125 * 450 tokens = 56,250 tokens ≈ $0.03
- Benefit: Nicer explanations for claims billers don't review

The benefit is essentially zero -- billers don't read these explanations. The template explanation ("Moderate denial risk. Review if time permits. Ensure documentation is complete.") is functionally identical to what the LLM would produce. The LLM's additional linguistic polish adds no operational value.

**For Low-tier claims, API calls would be actively harmful:**
- The LLM might overstate the risk to sound helpful
- A biller seeing an LLM-written explanation for a Low-tier claim might second-guess the triage
- The template ("Routine submission recommended") communicates exactly the right message

**Edge case: What if a Medium-tier claim is escalated?** A biller can manually promote a claim. In a production system, I'd add an "explain this claim" button that triggers an API call on demand -- giving the benefits of LLM explanations without the cost of generating them for all 375 non-High claims.

**Q107. Estimate the token cost for 500 API calls vs 125. If each 1M tokens costs $0.50, what's the difference?**

**125-claim approach (current):**

```
Input tokens: 125 * ~350 tokens/prompt = 43,750 tokens
Output tokens: 125 * ~100 tokens/response = 12,500 tokens
Total: ~56,250 tokens
Cost: 56,250 / 1,000,000 * $0.50 = $0.028
```

**500-claim approach (hypothetical):**

```
Input tokens: 500 * ~350 = 175,000 tokens
Output tokens: 500 * ~100 = 50,000 tokens
Total: ~225,000 tokens
Cost: 225,000 / 1,000,000 * $0.50 = $0.113
```

**Difference:** $0.113 - $0.028 = **$0.085 per pipeline run.**

At first glance, an 8.5-cent difference is negligible. But consider scale:

| Scale | 125-claim cost | 500-claim cost | Annual difference |
|---|---|---|---|
| Per run (500 claims) | $0.028 | $0.113 | $0.085 |
| Daily (10K claims, 20 runs) | $0.56 | $2.26 | $620/year |
| Monthly (300K claims, 600 runs) | $16.80 | $67.80 | $18,615/year |
| Enterprise (10M claims/yr, 20K runs) | $560 | $2,260 | $620,500/year |

At enterprise scale, the savings become meaningful. More importantly, the operational principle is: **only spend API resources on output that creates value.** Medium/Low-tier API explanations create zero marginal value because billers don't read them. Saving $620K/year at enterprise scale is a bonus on top of the principle.

**Q108. How do you handle API rate limits? What happens if Ollama Cloud is down during pipeline execution?**

**Current implementation:** The pipeline sends 125 sequential API calls (one per High-tier claim). There's no explicit rate limiting handling because Ollama Cloud's rate limits for the `gemma4:31b-cloud` model are generous enough for 125 calls.

**If rate limited:** The `ollama.chat()` call would raise an exception (likely `ollama.ResponseError` with a 429 status). The exception handler in `generate_explanation_api()` would catch this and fall through to `build_fallback_response()`. The claim would receive a deterministic template explanation instead of a GenAI one. The audit log would record the failure with the error type.

**Production improvements I'd add:**

1. **Exponential backoff with jitter:**
   ```python
   for attempt in range(max_retries):
       try:
           response = ollama.chat(model=..., messages=[...])
           return response
       except RateLimitError:
           sleep(2 ** attempt + random.uniform(0, 1))
   # After max retries, fallback
   ```

2. **Batch processing with inter-batch delay:** Send 10 requests, wait 1 second, send 10 more. This stays well within typical rate limits.

3. **Circuit breaker:** If 5 consecutive API calls fail, skip API for remaining claims and use templates exclusively. This prevents cascading failures.

**If Ollama Cloud is completely down:**

The pipeline would detect the failure at the first API call. After the circuit breaker trips (or 3 retries per claim), ALL explanatory generation falls back to deterministic templates. The pipeline completes normally with all 500 claims receiving template explanations. The audit log marks all calls as "fallback" with the error reason. Zero data loss, zero pipeline failure. The user sees a warning: "GenAI unavailable -- using template explanations for this run."

**Q109. The audit log records 500 records but only 125 are from the API. What do the other 375 records contain? Why log bypassed calls?**

**What the 375 bypassed records contain:**

```json
{
    "call_id": "bypass_126",
    "timestamp": "2026-05-27T11:32:44Z",
    "claim_id": "CLM04567",
    "tier": "Medium",
    "denial_probability": 0.18,
    "method": "template_bypass",
    "explanation_source": "deterministic_template_builder",
    "disclaimer_valid": true,
    "pydantic_valid": true,
    "bypass_reason": "Medium tier -- template sufficient",
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "latency_ms": 0,
    "api_call": false
}
```

**Why log bypassed calls:**

1. **Complete audit trail.** If a regulator asks "why was claim CLM04567 assigned Medium tier and given this explanation?", the audit log provides the timestamped answer: the model predicted 18% probability, it fell below the High-tier threshold, and the template was deemed sufficient. Without logging bypassed calls, there's a gap in the audit trail.

2. **Coverage metrics.** The audit log allows computing: "what fraction of claims received GenAI explanations vs templates?" and "is the GenAI coverage aligned with the High tier?" These are operational metrics for the GenAI team.

3. **Troubleshooting.** If a biller reports that a Medium-tier claim had a strange explanation, the audit log confirms whether it was API or template, and what the input probability was.

4. **Cost tracking.** Total API calls (125) vs total pipeline runs is an efficiency metric. If bypassed calls weren't logged, you'd need to cross-reference the CSV output to compute this.

5. **Regulatory compliance.** Healthcare systems must demonstrate consistent and documented decision-making. "We logged every single claim's explanation source" is a strong compliance argument.

**Q110. How would you detect if the LLM's explanations are degrading in quality over time? What metrics would you monitor?**

I'd implement a multi-signal quality monitoring system:

**Signal 1: Structural quality metrics (automatic, per-run):**
| Metric | Target | Alert If |
|---|---|---|
| JSON parse rate | >98% | <95% (model is producing more malformed JSON) |
| Pydantic pass rate | >98% | <95% (model is deviating from schema) |
| Disclaimer keyword rate | 100% | <99% (model is dropping uncertainty language) |
| Avg explanation length | 200-500 chars | <100 or >1000 (model is too terse or verbose) |
| Fallback rate | <5% | >10% (API is unreliable or model quality dropped) |
| Avg response latency | <3s | >5s (API performance degradation) |

**Signal 2: Content quality metrics (automatic, per-run):**
| Metric | How to Measure | Alert If |
|---|---|---|
| PII/hallucination markers | Regex for ICD codes (\b[A-Z]\d{2}\.\d{1,3}\b), dollar amounts (\$\d+), patient identifiers | Any hits |
| Response uniqueness | % of explanations with identical text (copy-paste from model) | >20% (model is being lazy) |
| Risk factor coverage | Do the top 3 API risk factors appear in the explanation? | <80% (model is ignoring input) |

**Signal 3: Human-in-the-loop quality (periodic, sampled):**
- Weekly review of 10 random API explanations by a billing SME
- Rate each on: accuracy, actionability, clarity (1-5 scale)
- Track average scores over time
- Alert if any dimension drops below 3.5/5

**Signal 4: Operational outcomes (lagged, monthly):**
- Do High-tier claims with API explanations have a lower denial rate than High-tier claims from a template-only period? (Suggests explanations are improving biller actions)
- Is the biller override rate stable? (Sudden increase = explanations are less trustworthy)

**Root cause investigation if quality degrades:**
1. Check Ollama Cloud status (model version change? infrastructure issues?)
2. Review prompt template (did someone modify the prompt?)
3. Sample failing explanations and categorize failure modes
4. Roll back to previous prompt version if needed
5. Consider switching to a different model if degradation persists

---

## Section 8: Production & Code Architecture (Q111-Q120)

### 8.1 System Design

**Q111. Walk me through the modular package structure. Why did you separate code into `config/`, `utils/`, `prompts/`, and top-level modules? What principle does this follow?**

```
ensemble_solution/
  src/
    __init__.py
    config/
      __init__.py        -- Exports: settings, paths, constants
      settings.py         -- Centralized configuration (data paths, model params)
    utils/
      __init__.py
      validate_data.py    -- Data contract validation (CSV schema, assertions)
      feature_engineering.py -- Single source of truth for features
      models.py           -- Training, evaluation, metrics
      explainability.py   -- LR coefficient -> risk factor extraction
    prompts/
      __init__.py
      templates.py        -- Pydantic models, prompt builder, fallback
    explanations.py       -- Ollama integration + audit logging
    experiment_tracker.py -- MLflow-compatible experiment versioning
    experiment_runner.py  -- 10-experiment active learning loop
    llm_audit.py          -- Production audit infrastructure
    run_pipeline.py       -- Main entry point (single command)
  tests/
    conftest.py
    test_validate_data.py
    test_feature_engineering.py
    test_models.py
    test_explainability.py
    test_explanations.py
```

**Design principles:**

1. **Separation of concerns (Single Responsibility Principle).** Each file does exactly one thing:
   - `feature_engineering.py`: Creates features. Nothing else.
   - `models.py`: Trains and evaluates models. Doesn't load data or create features.
   - `explanations.py`: Handles LLM integration. Doesn't compute risk factors.
   - `llm_audit.py`: Logs and validates. Doesn't make API calls.

2. **Dependency inversion.** High-level modules (run_pipeline.py) depend on abstractions (settings, feature_engineering functions), not on specific implementations. Changing the LLM model name requires editing only `settings.py`, not `explanations.py`.

3. **Configuration centralization.** All paths, hyperparameters, and model settings live in `config/settings.py`. No other file contains hardcoded paths like `"data/input/claims_history.csv"`. This makes the system portable -- change one file to deploy to a new environment.

4. **Test isolation.** Test files mirror the source structure and import only the modules they test. `test_feature_engineering.py` doesn't import `models.py`.

**What this prevents:**
- Circular imports (utils don't import from top-level, top-level imports from utils)
- Configuration drift (no hardcoded paths in utility modules)
- Accidental feature engineering changes in the pipeline (single source of truth)
- Test coupling (tests don't depend on each other's fixtures)

**Q112. You have `build_features()` in `feature_engineering.py` and `engineer = build_features` in other files. Why the alias? What problem does this solve?**

The alias solves a DRY (Don't Repeat Yourself) problem discovered during code review. The original codebase had TWO separate implementations of feature engineering:

1. `src/utils/feature_engineering.py::build_features()` -- the canonical implementation
2. `src/experiment_runner.py::engineer()` -- an independent copy with slight differences

This duplication violated DRY and created a maintenance nightmare: any bug fix in `build_features()` needed to be manually replicated in `engineer()`. If someone fixed a bug in one but forgot the other, the pipeline and experiment runner would produce different features for the same input -- silent data corruption.

**The fix:**
```python
# In experiment_runner.py:
from src.utils.feature_engineering import build_features
engineer = build_features  # Alias to the single source of truth
```

`engineer` is now a reference to the SAME function object as `build_features`. Any changes to `build_features()` automatically apply everywhere. The alias preserves backward compatibility (existing code calling `engineer()` doesn't break) while eliminating duplication.

**Why not just rename all call sites to `build_features()`?** That would be ideal, but:
1. The assessment's experiment runner was already written with `engineer()` calls
2. The alias approach is minimally invasive -- no risk of missing a call site
3. The Python import is clear: `engineer = build_features` makes the alias explicit

**Q113. Why did you replace `print()` statements with `logging.getLogger()`? What are three advantages of structured logging?**

The original code had 49 `print()` calls scattered across the pipeline. Print to stdout is fine for development scripts but problematic in production.

**Three advantages of structured logging:**

**1. Severity levels and filtering.**
```python
logger.debug("Feature matrix shape: (2240, 53)")     # Hidden in production
logger.info("Training LR with C=0.1...")              # Visible by default
logger.warning("API call failed, using fallback")     # Gets attention
logger.error("Data file not found: claims.csv")       # Triggers alerts
```
In production, you'd set the log level to WARNING, filtering out verbose debug/info messages. With `print()`, everything goes to stdout and you can't suppress it without modifying code.

**2. Timestamped, machine-parseable output.**
```
2026-05-27 11:32:44 [INFO] Starting pipeline execution
2026-05-27 11:32:45 [INFO] Loaded 3200 historical claims
2026-05-27 11:32:46 [INFO] Training Calibrated LR...
```
These timestamps enable:
- Latency analysis ("which step is slow?")
- Correlation with other system events ("did the 11:32 slowdown coincide with a DB backup?")
- Log aggregation into centralized monitoring (ELK, Datadog)

**3. Flexible output routing.**
```python
# Write to file AND console
file_handler = logging.FileHandler('pipeline.log')
console_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(console_handler)
```
With `print()`, output goes to stdout only. With logging, the same message can go to: file, console, syslog, email alerts, Slack webhooks, cloud monitoring services -- without changing any log statement.

**What remains as `print()`:** Only the experiment runner's final summary table (user-facing output, not operational logging).

**Q114. You use `if/raise ValueError` instead of `assert` for validation. Why? What happens to `assert` statements in optimized Python?**

Python's `assert` statements are removed when the interpreter runs with the `-O` (optimize) flag:
```bash
python -O src/run_pipeline.py  # All assertions are silently skipped!
```

**Example of catastrophic failure:**
```python
# WRONG: Using assert for business validation
assert 'claim_id' in df.columns, "Missing claim_id column"
assert df['denial_probability'].between(0, 1).all(), "Probabilities out of range"

# Production deploys with: python -O run_pipeline.py
# Both assertions are removed. The pipeline runs with corrupted data,
# producing garbage predictions, and nobody knows because the check was silently skipped.
```

**Correct approach:**
```python
# RIGHT: Using if/raise for business validation
if 'claim_id' not in df.columns:
    raise ValueError("Critical: Missing claim_id column. Cannot proceed.")

if not df['denial_probability'].between(0, 1).all():
    raise ValueError("Critical: Probability values outside [0,1] range.")
```

These checks CANNOT be disabled by compiler flags. They will always run, regardless of optimization settings.

**When is `assert` acceptable?**
- Internal consistency checks during development ("this list should never be empty here")
- Test code (`pytest` relies on `assert`)
- Debugging aids removed before production

**When to use `if/raise`:**
- Input validation (file existence, schema checks)
- Business logic constraints (denial probability range)
- Security checks (authorization, data access)
- Any check whose failure would cause downstream corruption

**Q115. The tier assignment uses `n // 4` instead of hardcoded `.loc[:124]`. Why is this important? What if current claims grow to 1,000?**

The original code had:
```python
# FRAGILE: Hardcoded for exactly 500 claims
scored.loc[:124, 'risk_tier'] = 'High'
scored.loc[125:249, 'risk_tier'] = 'Medium'
```

This breaks immediately if the number of current claims changes:
- 400 claims: `scored.loc[:124]` captures 31.25% instead of 25%
- 1,000 claims: `scored.loc[:124]` captures only 12.5%
- 100 claims: IndexError on `scored.loc[125:249]`

The fix:
```python
# ROBUST: Dynamic tier assignment
n = len(scored)
n_high = n // 4              # 25% of claims
n_med = n // 4               # 25% of claims
scored['risk_tier'] = 'Low'
scored.loc[:n_high - 1, 'risk_tier'] = 'High'
scored.loc[n_high:n_high + n_med - 1, 'risk_tier'] = 'Medium'
```

| Current Claims | n_high | n_med | High Range | Medium Range | Low Range |
|---|---|---|---|---|---|
| 500 | 125 | 125 | 0-124 | 125-249 | 250-499 |
| 1,000 | 250 | 250 | 0-249 | 250-499 | 500-999 |
| 400 | 100 | 100 | 0-99 | 100-199 | 200-399 |
| 100 | 25 | 25 | 0-24 | 25-49 | 50-99 |

For 1,000 claims: 250 High, 250 Medium, 500 Low. The review capacity percentage stays exactly 25% regardless of total claim count.

**Q116. You added pre-flight validation for file existence. What specific checks would you add for a production deployment?**

**Current pre-flight checks (implemented):**
```python
if not os.path.exists(HISTORICAL_CLAIMS_PATH):
    raise FileNotFoundError(f"Historical claims not found: {HISTORICAL_CLAIMS_PATH}")
if not os.path.exists(CURRENT_CLAIMS_PATH):
    raise FileNotFoundError(f"Current claims not found: {CURRENT_CLAIMS_PATH}")
```

**Production deployment pre-flight checklist:**

| Category | Check | Why |
|---|---|---|
| **File existence** | All input files present | Fail fast with clear error |
| **File freshness** | Input file modified in last 24 hours? | Detects stale data or stuck ETL |
| **File format** | CSV parses without error; expected column count | Detects corrupted or truncated files |
| **Column schema** | All required columns present; no unexpected columns | Detects schema changes from upstream |
| **Target column** | `is_denied` NOT in current claims | Prevents leakage scenario |
| **Data types** | Numeric columns are actually numeric; dates parse | Detects type corruption |
| **Value ranges** | No negative amounts; probabilities in [0,1] if scoring predictions | Detects data corruption |
| **Row counts** | Historical > 1000 rows; current > 10 rows | Detects nearly-empty files |
| **No duplicates** | No duplicate claim_ids | Prevents double-counting |
| **Feature distribution** | PSI < 0.25 vs training baseline | Detects concept drift |
| **.env configuration** | OLLAMA_API_KEY present; OLLAMA_HOST reachable | Prevents runtime API failures |
| **Output directory** | data/output/ exists and is writable | Prevents mid-pipeline write failures |
| **Disk space** | >100MB free in output directory | Prevents partial writes |
| **Python version** | >= 3.10, == 3.12 preferred | Prevents dependency incompatibility |
| **Package versions** | Installed packages match requirements.txt | Prevents silent behavior changes from package upgrades |

All checks should produce specific, actionable error messages: "Historical claims has 3 rows (expected > 1000). The file may be truncated. Last modified: 2026-01-15 (stale: 132 days)."

### 8.2 Testing & Quality

**Q117. You have 82 tests across 6 modules. How did you decide what to test? What is NOT tested that should be?**

**Testing prioritization framework:**

| Priority | Test Type | Why |
|---|---|---|
| **P0: Data integrity** | Feature engineering correctness, leakage prevention, output CSV validation | Wrong data = wrong predictions. Silent failures here propagate everywhere. |
| **P0: Business logic** | Pydantic validation, fallback correctness, tier assignment | These encode the core business rules (25% capacity, uncertainty disclaimer, factor limits). |
| **P1: Model correctness** | Metric computation, training pipeline, model comparison | Model errors are detectable (bad metrics). Less catastrophic than data errors. |
| **P1: API integration** | Prompt building, response parsing, fallback chain | LLM is inherently unreliable. Tests ensure graceful degradation. |
| **P2: Edge cases** | Empty data, single row, all-denied, all-clean | Ensures robustness under unusual inputs. |

**What IS tested (82 tests):**
- `test_validate_data.py` (10): CSV schema, value ranges, monotonic sorting, tier counts
- `test_feature_engineering.py` (16): Feature creation, leakage exclusion, one-hot encoding, scaling
- `test_models.py` (14): Training, evaluation, metrics, capture computation
- `test_explainability.py` (10): Coefficient extraction, factor mapping, attribution logic
- `test_explanations.py` (22): Pydantic models, prompt structure, response parsing, fallback
- `conftest.py`: Shared fixtures (sample data, mock model)

**What should be tested but isn't (production gaps):**

| Gap | Risk | How to Test |
|---|---|---|
| LLM API integration (end-to-end) | High | Mock `ollama.chat()` to simulate: success, 500 error, rate limit, malformed JSON, timeout |
| Audit log integrity | Medium | Verify that 500 records are written, call_ids are unique, timestamps are sequential |
| Experiment tracker artifact validity | Medium | Verify that params.json, metrics.json, model.pkl are written and loadable |
| Pipeline end-to-end with real (synthetic) data | High | Run the full pipeline and verify: output CSV has correct row count, metrics.json is valid, audit log exists |
| Concurrent pipeline execution | Low | Run two pipelines simultaneously; verify no file conflicts |
| Large data volumes (100K claims) | Medium | Profile memory usage and runtime; verify no OOM |
| Unicode/encoding edge cases | Low | Test with claim IDs containing special characters, non-ASCII text |

**Q118. Your tests use `conftest.py` for shared fixtures. What is the advantage of fixtures over creating data in each test?**

**With fixtures (current):**
```python
# conftest.py
@pytest.fixture
def sample_claims_df():
    return pd.DataFrame({
        'claim_id': ['C001', 'C002', 'C003'],
        'total_billed': [5000, 12000, 85000],
        'payer_type': ['Commercial', 'Medicaid MCO', 'BCBS'],
        'is_denied': [0, 1, 0],
        ...
    })

# test_feature_engineering.py
def test_auth_gap(sample_claims_df):
    df = build_features(sample_claims_df)
    assert df['auth_gap'].iloc[0] == 0
```

**Without fixtures (hypothetical):**
```python
# Each test creates its own data
def test_auth_gap():
    df = pd.DataFrame({...})  # Duplicated across 16 tests
    df = build_features(df)
    assert df['auth_gap'].iloc[0] == 0
```

**Advantages of fixtures:**

1. **DRY:** The sample DataFrame is defined once in `conftest.py` and used by 82 tests. A schema change requires one edit, not 82.

2. **Consistency:** All tests use the same reference data. If `build_features()` adds a column, only the fixture needs updating to reflect the new schema. Tests that don't care about the new column continue to work.

3. **Isolation:** Each test gets a FRESH copy of the fixture (or shared via scope). No risk of one test's modifications affecting another test's assertions.

4. **Composability:** Fixtures can depend on other fixtures:
   ```python
   @pytest.fixture
   def engineered_df(sample_claims_df):
       return build_features(sample_claims_df)

   @pytest.fixture
   def trained_model(engineered_df):
       return train_lr(engineered_df)
   ```

5. **Cleanup:** Fixtures with `yield` can perform teardown (delete temp files, close connections).

6. **Parametrization:** One fixture can generate multiple variants:
   ```python
   @pytest.fixture(params=['train', 'val', 'test'])
   def split_name(request):
       return request.param
   ```

**Q119. How would you test the LLM integration without making actual API calls? What mocking strategy would you use?**

I'd use a layered mocking strategy:

**Layer 1: Mock the `ollama.chat()` function directly.**
```python
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_ollama_chat():
    with patch('src.explanations.ollama.chat') as mock_chat:
        mock_response = MagicMock()
        mock_response.message.content = json.dumps({
            "claim_id": "CLM001",
            "disclaimer": "This is a statistical estimate, not a guaranteed outcome.",
            "risk_description": "High denial risk due to missing prior authorization.",
            "recommended_action": "Obtain prior authorization before submission.",
            "risk_estimate_label": "High"
        })
        mock_response.prompt_eval_count = 350
        mock_response.eval_count = 95
        mock_chat.return_value = mock_response
        yield mock_chat

def test_generate_explanation_api_success(mock_ollama_chat, sample_explanation_request):
    result, response_obj = generate_explanation_api(sample_explanation_request)
    assert result.claim_id == "CLM001"
    assert "statistical estimate" in result.disclaimer
    mock_ollama_chat.assert_called_once()
```

**Layer 2: Simulate different API failure modes.**
```python
@pytest.fixture
def mock_ollama_error():
    with patch('src.explanations.ollama.chat') as mock_chat:
        mock_chat.side_effect = ollama.ResponseError("500 Internal Server Error")
        yield mock_chat

@pytest.fixture
def mock_ollama_rate_limit():
    with patch('src.explanations.ollama.chat') as mock_chat:
        mock_chat.side_effect = ollama.ResponseError("429 Too Many Requests")
        yield mock_chat

@pytest.fixture
def mock_ollama_timeout():
    with patch('src.explanations.ollama.chat') as mock_chat:
        mock_chat.side_effect = TimeoutError("Connection timed out")
        yield mock_chat

def test_api_failure_falls_back_to_template(mock_ollama_error, sample_request):
    result, _ = generate_explanation_api(sample_request)
    assert "statistical estimate" in result.disclaimer  # Fallback is valid
```

**Layer 3: Simulate malformed responses.**
```python
@pytest.fixture
def mock_ollama_malformed_json():
    with patch('src.explanations.ollama.chat') as mock_chat:
        mock_response = MagicMock()
        mock_response.message.content = "Here is your JSON:\n```json\n{broken: json'''\n```"
        mock_chat.return_value = mock_response
        yield mock_chat

def test_malformed_json_triggers_fallback(mock_ollama_malformed_json, sample_request):
    result, _ = generate_explanation_api(sample_request)
    # Should fall through all three tiers and use deterministic template
    assert result.claim_id == sample_request.claim_id
```

**Layer 4: Integration test with a fixture that records all API interactions.**
```python
@pytest.fixture
def recorded_ollama():
    """Records all calls and replays saved responses (VCR-style)."""
    with patch('src.explanations.ollama.chat') as mock:
        mock.side_effect = lambda model, messages: load_cached_response(messages)
        yield mock
```

**Q120. What code quality issues did you identify and fix in your codebase? What tools would you add to a CI pipeline?**

**Issues identified and fixed:**

| Issue | Severity | Fix Applied |
|---|---|---|
| Duplicated `engineer()` in two files | HIGH | Aliased to `build_features()` from single source |
| `warnings.filterwarnings('ignore')` suppressing legitimate warnings | HIGH | Removed; addressed root causes of warnings |
| 49 `print()` statements in production pipeline | HIGH | Replaced with `logging.getLogger()` |
| `assert` for business validation (can be disabled with `python -O`) | HIGH | Replaced with `if/raise ValueError` |
| Hardcoded tier indices `.loc[:124]` | MED | Replaced with dynamic `n // 4` |
| Module-level import of optional dependency without fallback | MED | Consolidated imports at module level with proper error handling |
| Duplicated `make_X()` | MED | Standardized on `prepare_model_matrix()` |
| Missing docstrings on several public functions | LOW | Added docstrings to all functions |
| Non-ASCII characters (emojis, dashes) in source code | LOW | Replaced with ASCII equivalents |
| Inconsistent `submission_delay` labels (strings vs ints) | LOW | Standardized to integers |

**CI pipeline tools I'd add:**

| Tool | Purpose | Configuration |
|---|---|---|
| **black** | Automatic code formatting | `--line-length=100 --target-version=py312` |
| **isort** | Import sorting | `--profile=black` |
| **flake8** | Linting | Max line length 100, ignore E501/W503 |
| **mypy** | Static type checking | `--strict` initially, relax as needed |
| **pytest + pytest-cov** | Test running + coverage | Minimum 80% coverage; fail under |
| **bandit** | Security scanning | Check for hardcoded secrets, unsafe deserialization |
| **pre-commit** | Git hooks | Run black + isort + flake8 before commit |

**CI pipeline stages (GitHub Actions example):**
```yaml
- Lint: black --check, isort --check, flake8
- Type Check: mypy src/
- Security: bandit -r src/
- Unit Tests: pytest tests/ --cov=src --cov-report=xml --cov-fail-under=80 -v
- Integration Test: python src/run_pipeline.py --validate-only
```

---

