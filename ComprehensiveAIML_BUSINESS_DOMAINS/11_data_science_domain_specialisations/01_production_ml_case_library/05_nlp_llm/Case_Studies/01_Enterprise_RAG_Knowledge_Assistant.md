---
case_id: CASE-01-05-001
case_title: Authorized policy answers with an enterprise knowledge assistant
case_version: 1.0-draft
created: 2026-11-07
last_reviewed: null
status: proposed
content_status: draft
case_type: llm-agentic
domain: Enterprise knowledge and employee policy support
primary_pattern: authorization-aware-retrieval
secondary_patterns: [evaluation-mismatch, human-in-the-loop]
evidence_level: illustrative
source_basis: illustrative design; public research for general mechanisms
as_of_date: 2026-11-07
owner_role: Knowledge platform product owner
source_index: INDEX.md
review_trigger: Authorization semantics, corpus policy, model, or deployment boundary changes
---
# Authorized policy answers with an enterprise knowledge assistant
## 1. The One-Line Framing
A proposed assistant would answer an authenticated employee's policy question using only currently authorized, versioned evidence and citations, or abstain, instead of relying on the employee's manual policy search.
The decision owner is a knowledge-platform product owner; the output is informational, not an approval or policy exception.
The case is a constructed teaching scenario, not a report about a real enterprise or a deployed assistant.
## 2. Context & Background
Employees need to find internal travel and expense policy while arranging business trips.
The policies change by effective date, jurisdiction, and employee group.
The manual workflow begins with a search in the official policy portal and ends with a human interpreting source passages.
A plausible pain point is ambiguity among near-duplicate pages, not an established measured delay.
The portal remains the source of authority even if an assistant is introduced.
An employee may ask whether a particular meal is reimbursable.
The answer is not a reimbursement approval and cannot modify the expense system.
A policy owner decides the authoritative interpretation; the assistant cannot invent one.
The illustrative corpus includes published travel and expense pages, region-specific appendices, and access metadata.
Draft policies, personal expense claims, individual HR records, and external websites are outside scope.
The product is English-only in this bounded exercise; multilingual coverage requires separate evaluation.
The user must inspect citations and may escalate ambiguity to the policy help desk.
A source page that exists but is not authorized for a user must not be disclosed, even in a refusal.
The problem differs from generic web question answering because access rights and current policy are part of correctness.
No actual company, employee conversation, deployment, or business outcome is represented.
## 3. The Business Question
Illustrative stakeholder question: Should we offer cited answers to routine travel-policy questions, subject to the employee's access rights, rather than requiring manual portal search for every question?
Decision unit: one employee question at a specified time and authorization state.
Decision output: a supported informational answer with source links, an abstention, or a safe service-unavailable response.
Primary comparator: the current manual search portal with the same authenticated user and corpus.
The alternative system is not permitted to approve expenses or advise on an individual's legal entitlements.
The decision owner would authorize a limited pilot only after privacy, policy, and security owners approve the test plan.
A quality improvement is insufficient if access control leaks data or policy answers are materially wrong.
The decision is whether the proposed system merits a controlled pilot, not which foundation model wins universally.
## 4. Assumptions & Constraints
Assumption A1: the portal exposes a stable document identity and authoritative effective date.
Assumption A2: a trustworthy authorization service can evaluate a user against each document at query time.
Assumption A3: source owners can identify a current published version and withdraw superseded content.
Assumption A4: a policy team can adjudicate supported-answer labels and clarify unresolved contradictions.
Assumption A5: a representative set of employee questions can be sampled without leaking personal details.
These are testable assumptions, not facts about an existing organization.
Hard constraint: an answer must not reveal material outside the requester's authorization.
Hard constraint: the application does not execute a transaction or change a policy.
Hard constraint: retrieved content is untrusted data, not system instructions.
Hard constraint: a revoked document is not served merely because an index or cache still contains it.
Illustrative objective: provide a useful cited response when sufficient authorized evidence is available.
Illustrative guardrails: no unauthorized disclosure in adversarial tests and no unsupported policy assertion in the critical-risk test set.
A finite clean test does not establish a zero population risk.
Latency, budget, retention and policy risk limits require owner approval before an actual pilot.
No numeric service-level objective or threshold below is an achieved result.
A deployment must confirm applicable employment, privacy, security and records requirements with qualified owners.
## 5. Data, Tools & Method
### 5a. Data Specification
The source object is a published policy document version, not a page view or expense claim.
Each object has document ID, version, owner, publish time, effective interval, jurisdiction, classification, and source URI.
An ACL reference maps the object to its authoritative access-control decision.
The text extraction records the original section heading and offsets needed to resolve citations.
Chunk identity is derived from the document version and stable chunk ordinal or content hash.
Each chunk inherits document and section-level restrictions; a more restrictive section wins.
A query record contains a pseudonymous request identifier, request time, question, locale, and user authorization context.
The question text may contain personal data and is minimized or redacted under a documented retention policy.
The response record stores version identifiers, decision status, citations and service timings under restricted access.
Evaluation labels include question, applicable time and user role, expected support passages, answerability and adjudication notes.
A source-dependent question is answerable only if sufficient authoritative evidence is accessible to that role at the evaluation time.
A question about an inaccessible policy is evaluated for non-disclosure, not marked answerable on the basis of hidden evidence.
Labels are per question and authorization snapshot; a document relevance label alone does not establish a correct answer.
The illustrative population excludes external contractors until their distinct authorization pathways have been tested.
No data extract, production logs, or actual employee labels were supplied.
### 5b. Tools & Stack
The proposed connector reads the official policy portal through an approved service identity.
A versioned object store holds immutable extracted documents and source metadata.
A parser normalizes text while preserving section hierarchy and links back to authoritative originals.
A keyword index and a vector index are candidate-retrieval mechanisms, not authorities on access.
An embedding model and a language model are unspecified pending security and quality review.
A policy decision point answers live user-document authorization queries.
A stateless application orchestrator applies authorization, evidence selection, generation and citation validation.
A model gateway enforces provider, retention, network, quota and logging controls.
An evaluation store records labeled cases and versioned reports with access limited to reviewers.
The proposed stack does not require a particular cloud, programming language, vector vendor or model release.
No versions, hardware or providers have been selected or tested in this case.
A model cannot be assumed to forget indexed source content after a permission change.
### 5c. Method Summary
Input is an authenticated employee's question and current request context.
The baseline is manual search over the same published corpus and user permissions.
The candidate parses the question, retrieves keyword and embedding candidates, filters authorization, and reranks eligible chunks.
A generation model receives only the eligible selected evidence and a directive to abstain when support is insufficient.
A postprocessor validates citations against the evidence set before displaying an answer.
The user sees source links, effective-date context and an escalation route.
An alternative is keyword search with highlighted excerpts and no generated answer.
Another alternative is a fixed FAQ maintained by policy owners.
A third is a long-context model given all documents, which is considered but rejected for the initial design because authorization and context selection remain necessary.
None of these alternatives has been tested here.
### 5d. Data provenance, quality, exclusions and leakage controls
The source identifier in the directory inventory is DS_Domain/Case_Studies_Production_ML_Systems/NLP_LLM/01_Enterprise_RAG_Knowledge_Assistant.md.
The crosswalk assigns that entry to the local NLP/LLM production-case home and ds-cases-05 brief.
The source filename provides topic provenance only; the actual source contents were not inspected and are not represented as evidence.
The synthetic corpus assumptions in this case are not a reconstruction of that file's implementation.
An eventual ingest run must record extraction timestamp, source revision and ACL revision.
A missing effective date or ambiguous owner blocks publication into the answerable index pending remediation.
The ingestion ledger distinguishes parser failures from documents intentionally excluded by scope.
Duplicate text is not blindly deduplicated across jurisdictions or permission boundaries.
A content hash detects identical bytes but cannot establish equivalent authority or permissions.
The offline evaluation snapshot must precede the evaluation window for time-sensitive questions.
Questions and near-duplicate paraphrases stay in the same partition to reduce leakage.
A later version of a policy must not answer a historical question unless the task explicitly asks about that version.
Test labels and answers must not be fed into prompt tuning or retrieval-rule design before the final holdout evaluation.
Unauthorized examples belong in a restricted evaluation environment and must not be shared with unapproved model providers.
Source deletion or revocation triggers purge of extracted chunks, indexes, derived caches and governed evaluation artifacts.
Where deletion cannot be immediate, request-time checks must fail closed until the purge is complete.
No actual source collection date, freshness distribution, label skew or missingness rate is known.
### 5e. Reproducibility and method-selection ledger
A future run must capture code revision, dependency lock, parser version, chunking settings and index-build ID.
It must also capture the document snapshot, ACL policy revision, embedding and reranker versions, and generation model identifier.
Record prompt template, decoding settings, request timeout, test set ID, adjudication guide, random seed and run ID.
Capture point-in-time applicability and authorization fixtures independently from the model's output.
Record evaluation environment, retrieval scores, selected chunk IDs, citation spans and judge decisions.
Model-provider nondeterminism means a seed alone cannot guarantee bitwise reproduction.
A restricted-data run may be independently audited only by an authorized reviewer.
A synthetic fixture can reproduce arithmetic and processing semantics without exposing actual policy text.
No actual revision, run ID, artifact location or model version exists in this illustrative exercise.
Decision D1 proposes hybrid retrieval to cover exact terms and paraphrases; it is a hypothesis, not a measured gain.
Decision D2 requires live authorization before generation because index-time permissions can become stale.
Decision D3 keeps manual search as a comparator and fallback because it preserves source inspection.
Decision D4 prefers abstention to unsupported completion when authoritative support cannot be established.
Each choice is revisited if controlled tests expose a quality, safety, cost or operational disadvantage.
## 6. Analysis Walkthrough
The supported finding is a design tradeoff, not evidence that RAG improves employee outcomes.
A safe answer requires both evidence sufficiency and authorization; retrieval relevance alone supplies neither guarantee.
Consider an illustrative question: What is the standard meal cap for a domestic trip?
Suppose a fictional current domestic policy says the cap is 60 currency units per day.
A separate fictional policy says the international cap is 90 currency units per day.
The employee is authorized to see the domestic policy but is not authorized to see the international appendix.
A secure request checks effective dates and jurisdiction before using either passage.
If the domestic passage is current and supports the question, the answer can state 60 units per day and cite it.
The international passage must not be sent to the generator or mentioned to the user.
If the domestic passage has expired and no current replacement is available, the correct response is abstention and escalation.
The 60 and 90 values are examples of fictional source content, not actual policy or measured results.
### 6a. Worked retrieval and answerability calculation
Define the evaluation population before computing scores.
Suppose 100 illustrative questions are independently adjudicated against the authorized corpus snapshot.
Of these, 80 are answerable with authorized evidence and 20 are unanswerable or lack sufficient authorized support.
For answerable questions, suppose top-five retrieval contains all evidence needed for 64 questions.
For the 20 others, suppose the system correctly abstains on 18 and produces an unsupported answer on 2.
A further answerable group includes questions with evidence not retrieved; those must not be conflated with the unanswerable group.
This small synthetic table is constructed for arithmetic practice, not a test result.
Retrieval coverage is the proportion of answerable questions with all required evidence among the top five authorized results.
The denominator excludes unanswerable questions because their all-evidence target is undefined.
The numerator requires every necessary passage; merely finding one related passage does not suffice for a multi-source answer.
By the definition of a proportion, divide 64 successful retrievals by 80 eligible questions.

Time validity and authorization must be evaluated before a passage counts as relevant.

```text
FORMULA [R-01]: C5 = n_complete / n_answerable
WHERE: C5 is a dimensionless coverage fraction; n_complete and n_answerable are counts of questions.
CONSTRAINT: n_answerable > 0; all required evidence must be within five authorized results for the correct snapshot.
NUMERICAL EXAMPLE: C5 = 64 / 80 = 0.80 = 80%.
INTERPRETATION: 16 answerable questions lack complete evidence within the selected retrieval set.

FORMULA [R-02]: A = n_correct_abstentions / n_unanswerable
WHERE: A is the dimensionless correct-abstention fraction for unanswerable questions.
CONSTRAINT: n_unanswerable > 0; human labels must reflect the requesting user's rights and the relevant date.
NUMERICAL EXAMPLE: A = 18 / 20 = 0.90 = 90%.
INTERPRETATION: two unanswerable questions receive an unsupported answer in this constructed example.

FORMULA [R-03]: U = n_unsupported / n_unanswerable
WHERE: U is the conditional unsupported-answer rate for the unanswerable group.
CONSTRAINT: every unanswerable case is classified exactly once as abstained or unsupported in this simplified example.
NUMERICAL EXAMPLE: U = 2 / 20 = 0.10 = 10%.
INTERPRETATION: A + U = 1 only under the stated exhaustive binary outcomes.
```

Arithmetic check: 0.8 times 80 is 64; 0.9 times 20 is 18; 0.1 times 20 is 2.
Count check: 64 plus 16 equals 80; 18 plus 2 equals 20.
Units check: questions divided by questions is dimensionless, not a latency or a per-user rate.
The 80% coverage figure does not measure answer correctness after generation.
The 90% abstention figure does not establish performance for answerable cases or for an entire employee population.
If eight additional answerable questions gain complete retrieval, C5 becomes 72/80 = 90%, holding labels fixed.
If one additional unanswerable question produces an unsupported answer, U becomes 3/20 = 15%.
If no questions are answerable, C5 is undefined rather than zero or 100%.
If only one necessary passage is found for a question requiring two, the question is not a complete retrieval success.
The numbers are too small to justify a deployment claim or a precise population estimate.
### 6b. Worked workload and capacity estimate
Assume for planning only 12,000 requests per eight-hour workday.
Assume 80% of those requests arrive uniformly in a four-hour busy interval.
Assume each request triggers one generation and an average of two vector searches, with no retries.
A service's actual peak is not known and must be measured before procurement.

```text
FORMULA [W-01]: lambda_busy = N * p / (h * 3600)
WHERE: lambda_busy is requests per second, N is requests per day, p is busy-interval fraction, h is hours.
CONSTRAINT: traffic uniform within the interval; retries and burstiness are excluded.
NUMERICAL EXAMPLE: 12000 * 0.8 / (4 * 3600) = 9600 / 14400 = 0.667 requests/s.
INTERPRETATION: this is an average over the busy interval, not a peak or a tail-latency requirement.

FORMULA [W-02]: lambda_search = lambda_busy * s
WHERE: lambda_search is vector searches/s and s is vector searches/request.
CONSTRAINT: fixed average s and no retry amplification; keyword search has separate capacity.
NUMERICAL EXAMPLE: 0.667 * 2 = approximately 1.333 searches/s.
INTERPRETATION: provision from measured burst and failure conditions, not this mean alone.
```

Independent check: 1.333 searches/s times 14,400 s is approximately 19,200 searches, which is twice 9,600 requests.
If the same busy traffic arrives over one hour, the request average rises to 9600/3600 = 2.667 requests/s.
At two searches per request that is 5.333 searches/s before retries.
Zero busy hours makes the model undefined; treating the average as an instantaneous peak is invalid.
Token throughput, cache misses, reranker batching, authorization latency and provider limits can dominate despite low average request rate.
Do not multiply separate component p95 latencies to infer end-to-end p95; measure the end-to-end distribution.
### 6c. Evidence selection trace
Candidate A is the active domestic meal policy and passes both jurisdiction and authorization checks.
Candidate B is an older domestic version and fails the effective-date check.
Candidate C is the international appendix and fails authorization for this user.
Candidate D is a generic travel overview that lacks the requested cap.
The reranker may prefer A among the eligible candidates, but relevance score cannot override B's expired status.
C is filtered before any text is included in the model context or user-visible traces.
The system selects the minimum sufficient eligible passage and includes a stable citation identifier.
The response validator checks that the cited identifier belongs to the selected authorized evidence set.
A separate support check asks whether the content actually entails the answer; a valid ID alone is insufficient.
If authorization cannot be determined, the request fails closed and offers manual portal navigation.
If a cited version changes before display, reauthorize and recheck its effective status before sending the answer.
The trace is an illustrative mechanism, not an execution log.
## 7. The Finding
The proposed pilot decision depends on showing that employees receive supported answers without exposure to unauthorized or obsolete policies.
The worked example shows how complete-evidence retrieval coverage and abstention performance expose distinct failure modes.
Neither synthetic number is an observed result, and neither by itself demonstrates a safe or useful product.
A secure, version-aware retrieval path and a manual-search comparator make the hypothesis testable; they do not establish success.
## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
Proposed input-to-output path:
Policy portal → authorized connector → immutable source snapshot → parser and metadata validation → permission-aware chunks → lexical/vector indexes → evaluation gate → signed index manifest → serving registry.
The connector reads only approved published policy material and records source version and access metadata.
The parser rejects corrupt extraction and detects missing section boundaries.
A policy owner resolves missing jurisdiction, owner, effective date, or restriction before the document becomes eligible.
Chunking preserves a section heading and enough surrounding policy text to interpret exceptions.
Chunk boundaries must not separate a numeric limit from its qualifier without preserving a link between them.
Every chunk has its original source URI and an immutable document-version reference.
The embedding job emits vectors bound to the matching chunk and model version.
The lexical index stores exact terms, document identifiers and policy names.
The ACL index is an optimization only; it is not a substitute for the authoritative request-time decision.
The manifest contains index generation, source snapshot, embedding model, configuration and validation results.
An index is not promoted if access-control fixture tests or source-link checks fail.
Index promotion is atomic; the old manifest is retained for controlled rollback only while still safe.
Rollback never bypasses an emergency revocation or restores a removed document's access.
Ingestion events carry monotonically comparable source revisions where the source supports them.
Out-of-order updates are rejected or reconciled against the authoritative latest source revision.
A deletion event marks affected source versions unavailable immediately, then purges secondary representations.
When event delivery is uncertain, periodic reconciliation compares portal inventory to indexed inventory.
An embedding-model change requires a compatible index rebuild and separate evaluation.
No model training or fine-tuning is proposed; inference and indexing suffice for this case.
### 8b. Online / Serving / Real-Time Path
Successful path: employee → authenticated application → query normalization → candidate retrieval → live document authorization → reranking → evidence sufficiency gate → generator → citation and support validation → authorized answer and portal links.
The gateway validates session identity and audience before accepting a question.
The request binds the user's identity, authorization context and relevant time to a unique trace ID.
The application treats the question and all retrieved documents as untrusted input.
Lexical and vector candidate sets are merged without exposing raw candidate text to the user.
Authorization is checked before candidate text reaches the reranker if the reranker is outside the trusted boundary.
Only eligible and current passages may enter generation context.
Retrieval filters use a conservative limit and do not imply that the top result is sufficient evidence.
The evidence gate rejects missing qualifications, ambiguous applicability or contradictory sources.
The prompt instructs the generator to answer only from supplied passages and to label uncertainty.
Instructions inside a policy page to reveal secrets or ignore constraints have no authority.
The validator rejects citations that are missing, malformed, outside the evidence set or no longer authorized.
The user-facing answer includes direct source links and may identify a policy date without exposing inaccessible document names.
The system must avoid giving the impression that a generated statement is an approval.
For audit, a protected trace stores manifest ID, prompt version, selected source IDs and final decision.
The trace itself is access-controlled and retained only for the approved period.
Failed path: authorization service timeout → no evidence forwarded → safe unavailable response → manual portal link.
Failed path: generation timeout → no partial unsupported answer → manual search and retry guidance.
Failed path: source link fails validation → reject that answer, record diagnostic metadata, and escalate or abstain.
Failure messages do not distinguish whether a hidden document exists.
Retries are bounded and must not reuse a stale authorization result after a permission change.
A cold index starts in unavailable mode until a valid manifest is loaded.
At peak load, admission control can shed generation traffic and retain manual search.
Service owners set actual timeout budgets only after a latency budget and dependency behavior are measured.
### 8c. Key Design Decisions
D1: preserve a lexical baseline alongside embeddings because exact policy terms and paraphrases fail differently.
Tradeoff: additional index and ranking operations increase operational cost and debugging complexity.
Boundary: hybrid retrieval is unnecessary if controlled evaluation finds the simpler lexical method adequate.
D2: authoritative authorization occurs at request time, with fail-closed behavior on uncertainty.
Tradeoff: dependency latency and availability directly affect answer availability.
Boundary: a verified entitlement snapshot could be used only if its revocation semantics meet the approved risk budget.
D3: use citations at the claim level, not only a generic list of sources.
Tradeoff: checking support and source links requires additional computation and policy review.
Boundary: an accurate citation identifier does not prove the claim is entailed.
D4: abstain on insufficient evidence rather than asking the model to infer a policy exception.
Tradeoff: more abstentions may reduce utility and should be measured among genuinely answerable questions.
Boundary: a human policy owner may resolve ambiguity, but the system should not manufacture that resolution.
D5: do not allow expense-system actions in this case.
Tradeoff: a separate workflow remains necessary after an employee obtains an answer.
Boundary: adding actions requires a distinct authorization, confirmation and audit decision.
### 8d. Architecture artifacts and operational readiness
Ownership: the product owner owns scope and release decision; policy owners own policy content and interpretations.
Security owns access-control threat modeling; platform operations own service availability and incident response.
Data governance owns retention and deletion rules; evaluation leads own labels and holdout integrity.
Authentication boundary: the front door accepts only verified identities, not a user-supplied role string.
Authorization boundary: every returned chunk is checked against the authoritative user-resource decision.
Model boundary: external inference receives only approved data under reviewed provider terms.
Document boundary: policy text cannot override application instructions, access rules or tool privileges.
Logging boundary: operators receive only minimum necessary sensitive text and have audited access.
Release boundary: promote an index and model configuration as a versioned, reversible bundle.
A release checklist requires owner sign-off on corpus provenance, access tests, latency budget and rollback.
The on-call runbook identifies how to disable generation without disabling the policy portal.
The kill switch must be exercised in a safe preproduction environment before a pilot.
A security incident triggers containment, preservation of authorized audit evidence, and appropriate notification procedures.
An ownership transfer requires updating the escalation route and verification of access to relevant operational tools.
A source revocation requires both synchronous request-time denial and asynchronous derived-data cleanup.
## 9. Evaluation & Validation
Intended protocol only: no offline evaluation, pilot, test, or production deployment has been executed here.
Create a time-separated question set sampled from approved, privacy-reviewed sources and supplemented with adversarial cases.
Stratify by role, jurisdiction, current versus historical policy, exact term versus paraphrase, and answerable versus unanswerable status.
Include conflicting documents, missing qualifications, permission changes, prompt injection and inaccessible-source questions.
Reviewers independently mark answerability and evidence spans, then adjudicate disagreements without seeing candidate model preference.
Freeze the holdout before tuning prompts, models or chunk sizes.
Use the same authorized corpus snapshot for the candidate and the manual-search comparator.
The comparison should preserve realistic task access and allow humans sufficient time to use the portal.
Primary quality metric: question-level supported-answer rate among all eligible evaluation questions.
Its numerator is questions receiving a complete, correct, properly authorized cited answer when one is available, or a correct abstention when not.
Its denominator is all eligible questions in the defined evaluation set and time window.
Report answerable and unanswerable strata separately because the combined score depends on their mixture.
Coverage metric: complete authorized evidence in the first five retrieved chunks among answerable questions, defined in R-01.
Citation precision: supported cited claims divided by all cited claims, with a claim segmentation protocol fixed before scoring.
A correct citation must be authorized, applicable at the relevant time and actually support the attached claim.
Abstention sensitivity is defined in R-02; also report unnecessary abstention among answerable questions.
Authorization guardrail: count exposures of unauthorized source content or revealing metadata across tested requests.
Report the total tested requests and attack classes alongside any zero count; zero observed is not proof of zero risk.
Freshness guardrail: count answers using sources outside their effective interval, excluding explicitly requested historical answers.
Operational guardrails: end-to-end latency distribution, error rate, dependency timeouts and cost per eligible request.
Latency is measured from accepted request to final response, with population, window and percentile specified in the report.
Cost includes provider, retrieval, storage, operations and human review; assumptions and accounting period must be declared.
Human task outcome: independently scored correctness and time to source-supported answer in the manual and assisted arms.
If causal impact on employee work is sought, randomize eligible users or sessions with appropriate contamination controls.
A before-after comparison alone is vulnerable to changes in policy complexity, demand and staffing.
Report sample sizes, uncertainty intervals, missing outcomes and subgroup breakdowns.
A small critical subgroup may be too sparse to rule out material harm even if its aggregate metric looks strong.
No launch threshold is asserted as approved; owners must specify limits and stopping rules before examining results.
A pilot must have a safe escalation path, informed participant handling, staged exposure and an independent stop authority.
Model-based judges may triage but should be calibrated against blinded human adjudication and not be the sole safety arbiter.
Measure the manual baseline's wrong answers and difficulty too; do not assume it is perfect.
Compare candidate failures to the baseline by question and user, not just across different samples.
A policy update after labeling requires a new snapshot or explicit historical evaluation; otherwise labels become stale.
### 9a. Uncertainty illustration
Suppose the 64 successes in R-01 arose from a simple random sample of 80 independent answerable questions.
The observed proportion is 0.8, but it is not a guarantee for future questions.
A rough normal standard error under the Bernoulli approximation is the square root of 0.8 times 0.2 divided by 80.
That is the square root of 0.002, approximately 0.0447, or 4.47 percentage points.
An approximate 95% normal interval is 0.8 plus or minus 1.96 times 0.0447, approximately 0.712 to 0.888.
This interval is illustrative and depends on independent representative sampling and the approximation.
For clustered employees, repeated paraphrases or sparse subgroups, use a suitable cluster-aware or exact/binomial method instead.
The constructed cases are not an actual random sample, so this interval is a teaching calculation only.
The true population definition and sampling frame must be agreed before interpreting any interval.
## 10. Action Taken
No organization action was observed or performed.
The instructional action is to propose an evaluation and architecture for review.
The policy owner would first approve the scope and corpus; security would approve the authorization design.
The evaluation team would create frozen labels and compare the candidate with manual search.
If the gates are not met, the decision is to retain the portal and remediate before another review.
Any pilot would require separate approval and would not be inferred from this document.
## 11. Outcome & Results
No deployment, employee adoption, latency, cost savings, or safety outcome has been measured.
The 80% and 90% values are constructed arithmetic inputs, not a retrospective or a forecast.
No empirical comparison among keyword, hybrid and long-context methods has been executed.
The only supported output is a falsifiable proposed design and evaluation protocol.
A positive offline result would still not establish a production outcome or causal time saving.
## 12. Failure Modes & What Went Wrong
No real incident or failed production test is claimed.
The following are hypothetical diagnostic exercises; competing explanations must be separated before a fix.
### 12.1. Revoked document remains searchable
Trigger: An answer cites a page after the employee loses access.
A revocation can expose a document even if the text has not changed.
Competing causes: A stale ACL cache or a failed deletion event can both explain the symptom.
First discriminating check: Compare live policy-decision output with cached entitlement and event timestamps.
Freeze responses using the affected entitlement and invalidate its cached grants.
Proposed correction: Deny the source immediately and purge derived representations.
Mitigation boundary: Caching a denial too long can hide newly granted access.
Verification plan: Re-run revoke and regrant fixtures across all index replicas.
Security approves recovery after the live decision and every serving replica deny the revoked identity.
### 12.2. Hidden title appears in refusal
Trigger: The refusal names a restricted appendix.
The restricted title itself can reveal that a confidential policy exists.
Competing causes: A logging template or an overly specific abstention generator may leak metadata.
First discriminating check: Inspect the response assembly trace without exposing the hidden title to the requester.
Replace the response with the same wording used when no accessible evidence exists.
Proposed correction: Use a generic refusal and restrict diagnostic traces.
Mitigation boundary: Overly generic operator logs can impede incident diagnosis.
Verification plan: Test both content and metadata leakage using canary titles.
Security reviews user-visible refusals and restricts access to trace metadata.
### 12.3. Expired policy answers current question
Trigger: An old version provides the visible numeric cap.
The employee could follow a cap that was valid only in a previous period.
Competing causes: Missing effective metadata or a wrong timezone conversion are competing causes.
First discriminating check: Compare source timestamps, effective interval and request timezone.
Quarantine the suspect version from current-date answers while preserving authorized historical access.
Proposed correction: Block uncertain versions and correct the timestamp mapping.
Mitigation boundary: Removing a valid historical document can break explicitly historical questions.
Verification plan: Run boundary tests immediately before and after the effective instant.
The policy owner confirms the effective interval before the indexing team restores it.
### 12.4. Future policy applied too early
Trigger: The answer cites a policy effective next month.
Publishing a policy early does not make it effective immediately.
Competing causes: Sorting by latest publish time or incorrectly treating draft as active can explain it.
First discriminating check: Compare publish status with effective-from and current request time.
Exclude the future version from present-date evidence while retaining its scheduled activation.
Proposed correction: Filter on publication state and effective interval.
Mitigation boundary: Suppressing advance notices may harm a separate future-planning task.
Verification plan: Test a future-effective document alongside the current one.
The policy owner checks the activation boundary in the jurisdiction’s time zone.
### 12.5. Regional exception omitted
Trigger: A generic limit is shown without a local exception.
A general cap can be wrong for an employee covered by a local exception.
Competing causes: Chunk boundaries may hide the exception or jurisdiction inference may be wrong.
First discriminating check: Inspect retrieved neighbors and the explicitly supplied user locale.
Suppress the combined answer until the region and applicable appendix are resolved.
Proposed correction: Retrieve related exception sections and ask for missing jurisdiction.
Mitigation boundary: Unverified user location should not silently become an entitlement.
Verification plan: Test matched policy and exception pairs across regions.
The regional policy owner adjudicates whether the exception overrides or supplements the base rule.
### 12.6. Question requires two passages
Trigger: The answer cites only a general rule.
Retrieving the general rule alone gives an incomplete answer when an exception also applies.
Competing causes: A top-k cutoff or single-passage sufficiency check may omit the second source.
First discriminating check: Compare the adjudicated required evidence set to selected chunks.
Return an abstention rather than presenting partial evidence as the complete policy.
Proposed correction: Use a multi-passage completeness gate and abstain if incomplete.
Mitigation boundary: Increasing k can add noise and increase prompt-injection exposure.
Verification plan: Evaluate questions requiring rule plus exception together.
The evaluation lead labels both passages as necessary and reviews the resulting evidence set.
### 12.7. Citation ID is valid but claim unsupported
Trigger: The link opens but does not justify the assertion.
A citation that exists is not evidence that the cited words entail the claim.
Competing causes: The generator may attach a generic source or the citation validator may check only IDs.
First discriminating check: Compare each atomic claim to its cited span with blinded reviewers.
Reject the unsupported claim even if the link resolves and the document is authorized.
Proposed correction: Require claim-level support and reject unsupported claims.
Mitigation boundary: Automated entailment models can themselves misclassify nuances.
Verification plan: Audit citation precision on held-out annotated claims.
A policy reviewer checks the specific cited span against each material answer claim.
### 12.8. Wrong source linked
Trigger: The text is relevant but clicking leads to another policy.
A passage-to-URL mismatch prevents the user from verifying the asserted rule.
Competing causes: A chunk-to-URI mapping defect or a duplicated document ID can explain it.
First discriminating check: Trace chunk ID through manifest to original version and URI.
Disable the faulty citation mapping rather than substituting an unrelated valid link.
Proposed correction: Rebuild the affected mapping from immutable source identifiers.
Mitigation boundary: A blanket index rollback can restore a revoked page.
Verification plan: Test URI resolution for sampled chunks and duplicate-ID fixtures.
The connector owner compares immutable document IDs, source revisions and canonical links.
### 12.9. Retrieved text contains prompt injection
Trigger: A document tells the model to ignore access restrictions.
Untrusted policy text can carry instructions aimed at overriding the assistant’s task.
Competing causes: The source may be compromised or a benign document may quote an attack example.
First discriminating check: Inspect the source revision and contextual provenance under restricted access.
Do not follow embedded commands; remove the compromised source from serving pending review.
Proposed correction: Treat source instructions as data and withhold tools and secrets.
Mitigation boundary: Stripping all imperative language may destroy legitimate policy instructions.
Verification plan: Exercise injection fixtures and verify both answer and tool boundaries.
Security examines whether the source was altered and tests instruction isolation at the model boundary.
### 12.10. User attempts role spoofing
Trigger: A question says the user is an administrator.
An asserted role in question text is not an authenticated entitlement.
Competing causes: The application may trust the prompt or propagate a client-supplied entitlement.
First discriminating check: Compare request identity token and authorization service inputs.
Ignore the claimed role and evaluate the principal from the signed session.
Proposed correction: Derive identity and rights only from trusted session context.
Mitigation boundary: An identity-service outage may reduce availability under fail-closed operation.
Verification plan: Test forged role fields and prompt-based privilege assertions.
Identity owners test forged role claims against both retrieval and final response paths.
### 12.11. Search returns no candidates
Trigger: A known policy is absent from top results.
An empty candidate set can be a genuine corpus gap or a search failure.
Competing causes: The connector may have missed the page or query vocabulary may differ.
First discriminating check: Check source inventory, index manifest and lexical versus vector candidates.
Offer authorized portal search rather than guessing from model memory.
Proposed correction: Repair ingestion if missing; tune retrieval only if indexed.
Mitigation boundary: Query expansion can surface more irrelevant sensitive candidates.
Verification plan: Use a held-out known-item set and reconcile indexed inventory.
The retrieval owner checks indexing coverage and exact-match queries against the known source.
### 12.12. High retrieval score but wrong policy
Trigger: A highly similar obsolete page ranks first.
Semantic similarity does not establish applicability or effective status.
Competing causes: Semantic similarity may ignore effective date or duplicate text may skew ranking.
First discriminating check: Inspect current-status metadata and ranked versions.
Reject the obsolete candidate before ranking eligible documents for the current request.
Proposed correction: Apply eligibility filters before ranking and prefer authoritative active versions.
Mitigation boundary: A hard current-only filter is wrong for historical questions.
Verification plan: Test current and historical questions against identical version families.
The search owner compares rank features with the policy version’s validity metadata.
### 12.13. Important table loses structure
Trigger: A meal-cap table becomes a sequence of unattached numbers.
Flattening a matrix can attach a valid amount to the wrong city or employee group.
Competing causes: Parser extraction or multi-column reading order may corrupt context.
First discriminating check: Compare source rendering with extracted cells and row headers.
Exclude the broken table from answers until row and column associations are restored.
Proposed correction: Use structured table extraction and reject low-confidence parses.
Mitigation boundary: OCR repair may create plausible but incorrect numeric values.
Verification plan: Have reviewers compare table fixtures to source and verify units.
The ingestion owner compares extracted cells and headings with the source rendering.
### 12.14. Scanned page has OCR error
Trigger: A cap of 60 is extracted as 80.
An OCR digit error can change the reimbursement limit while leaving the sentence fluent.
Competing causes: Low-resolution OCR or a font-specific glyph confusion may be responsible.
First discriminating check: Inspect page image and OCR confidence on the affected span.
Quarantine low-confidence numeric extractions and route them for source-image review.
Proposed correction: Route low-confidence numeric fields to human verification.
Mitigation boundary: Raising thresholds can delay indexing of legitimate updates.
Verification plan: Use known numeric OCR fixtures and sample current policy pages.
The document owner verifies the original number before an amended index is promoted.
### 12.15. Two current policies disagree
Trigger: Two authorized sources offer different caps.
Two current authorized sources with incompatible limits cannot both determine one unqualified cap.
Competing causes: A missing jurisdiction qualifier or an unapproved duplicate can create conflict.
First discriminating check: Compare owner, applicability, dates and publication status.
Present the conflict without choosing a winner and direct the user to policy support.
Proposed correction: Abstain and escalate unresolved contradictions to policy owners.
Mitigation boundary: Automatically choosing the newer page may disregard source authority.
Verification plan: Evaluate a conflicting-source fixture and the escalation route.
The policy owner resolves precedence and republishes an authoritative version.
### 12.16. Incorrect no-answer response
Trigger: The assistant abstains despite an available supporting policy.
An unjustified abstention increases manual work even though an authorized answer exists.
Competing causes: Retrieval failure or an excessively strict sufficiency threshold can explain it.
First discriminating check: Check whether complete authorized evidence appears in candidate and selected sets.
Keep the portal fallback available while investigating the failed evidence path.
Proposed correction: Fix the first failing stage rather than forcing an answer.
Mitigation boundary: Relaxing the gate can raise unsupported-answer risk.
Verification plan: Report answerable-case abstention and unsupported rate together.
The evaluation lead distinguishes retrieval misses from an overly strict sufficiency gate.
### 12.17. Correct-looking unauthorized answer
Trigger: The final answer coincides with hidden material.
An answer matching hidden content is still a leak if it was derived from an unauthorized source.
Competing causes: The model may have received unauthorized context or guessed a common value.
First discriminating check: Inspect restricted generation input and candidate authorization trace.
Block the output and preserve a restricted provenance trace for security review.
Proposed correction: Remove unauthorized context and reject unverifiable claims.
Mitigation boundary: The identical public fact may also appear in an authorized source.
Verification plan: Use unique secret canaries to distinguish leakage from coincidental overlap.
Security distinguishes exposure through retrieved evidence, model memory and a coincidental public value.
### 12.18. Stale entitlement cache
Trigger: A recent group change is not reflected.
An entitlement cached before a group removal can continue granting access after revocation.
Competing causes: Cache TTL or delayed identity propagation may be responsible.
First discriminating check: Compare authoritative identity and policy-decision timestamps.
Stop relying on the cached grant until the policy decision point confirms current rights.
Proposed correction: Fail closed when freshness is unknown and honor revocation events.
Mitigation boundary: Disabling all caches may exceed the latency budget.
Verification plan: Test revocation at multiple cache ages and during an outage.
Identity and platform owners agree a revocation propagation contract and exercise it.
### 12.19. Cross-user cache leak
Trigger: One employee sees another employee response.
A cached answer keyed only by question can disclose another user’s authorized evidence.
Competing causes: Cache key may omit identity context or document authorization may change after caching.
First discriminating check: Inspect key dimensions and reauthorization on cache hit.
Disable shared response caching for the affected endpoint and invalidate existing entries.
Proposed correction: Partition by security context and reauthorize every cited source.
Mitigation boundary: Per-user caching increases storage and can still be stale.
Verification plan: Test identical queries from users with different permissions.
The platform owner tests isolation by principal, policy revision, time and relevant context.
### 12.20. Shared model session retains content
Trigger: A later user receives text from a previous request.
A shared conversation state can carry source text between unrelated employee sessions.
Competing causes: Conversation state may be reused or logs may be accidentally replayed.
First discriminating check: Inspect session IDs, gateway isolation and context assembly.
Terminate the affected session pool and prevent reuse across security principals.
Proposed correction: Use tenant-scoped isolated sessions and clear request state.
Mitigation boundary: Provider-side session behavior requires contractual and technical review.
Verification plan: Run cross-session canary tests in an approved environment.
The model gateway owner tests state isolation and provider retention guarantees.
### 12.21. Source deleted but vector persists
Trigger: A deleted policy appears in vector candidates.
A vector can persist after the source is deleted and still produce a candidate ID.
Competing causes: Deletion may have reached the object store but not every vector replica.
First discriminating check: Compare deletion ledger to index replica watermarks.
Deny the deleted document at request time while removing all index and cache derivatives.
Proposed correction: Block the document ID at serving and purge all derived vectors.
Mitigation boundary: A deny list alone is not a complete retention deletion process.
Verification plan: Verify absence after compaction and replica reconciliation.
The index owner reconciles deletion tombstones with every replica and snapshot.
### 12.22. Out-of-order updates restore old content
Trigger: An earlier ingestion event overwrites a newer policy.
An older event arriving late can overwrite a newer publication state without version ordering.
Competing causes: A retry can arrive late or independent connectors may disagree on ordering.
First discriminating check: Inspect source revisions and event sequence.
Pause the affected document pipeline and keep the last verified manifest active.
Proposed correction: Apply conditional writes against authoritative revision.
Mitigation boundary: Source clocks are not reliable revision ordering unless guaranteed.
Verification plan: Replay events in reverse order and verify final state.
The ingestion owner compares monotonic source revisions and rejects stale events.
### 12.23. Failed index promotion
Trigger: Some servers use new chunks with an old embedding model.
An embedding mismatch can make nearest-neighbor scores meaningless without producing an obvious error.
Competing causes: Non-atomic deployment or missing compatibility metadata may be responsible.
First discriminating check: Compare manifest and model IDs across serving replicas.
Stop promotion and route traffic to the last compatible index-model pair.
Proposed correction: Promote a signed compatible bundle atomically.
Mitigation boundary: A rollback to a previous index must preserve revocations.
Verification plan: Run mixed-version rollout and rollback fixtures.
The release owner verifies model fingerprint, vector dimension and manifest compatibility on each replica.
### 12.24. Authorization service outage
Trigger: All policy answers become unavailable.
An authorization outage is not permission to answer using stale or unverified grants.
Competing causes: An upstream outage or local network error can cause indistinguishable timeouts.
First discriminating check: Inspect health signals and scoped dependency metrics.
Fail closed and offer the independent portal if its own access control is available.
Proposed correction: Fail closed and route the user to the existing portal.
Mitigation boundary: Unsafe stale authorization trades availability for potential disclosure.
Verification plan: Inject timeouts and verify no evidence reaches the model.
The identity service owner measures recovery and investigates dependency health without exposing document existence.
### 12.25. Generation provider outage
Trigger: Retrieval succeeds but no answer arrives.
A generation outage need not make authorized search snippets unavailable if that fallback is separately validated.
Competing causes: Provider throttling or a local gateway timeout may be responsible.
First discriminating check: Separate gateway status, quota and provider responses.
Return a safe unavailable response or approved search-only results rather than fabricating an answer.
Proposed correction: Return manual search with a bounded retry policy.
Mitigation boundary: Unbounded retries amplify load and cost.
Verification plan: Exercise rate-limit and timeout fixtures with admission control.
The model gateway owner checks provider errors and retry budgets before restoring generation.
### 12.26. Reranker receives restricted text
Trigger: Final output is safe but an external reranker sees hidden content.
A third-party reranker receiving restricted text is a disclosure even if the user never sees it.
Competing causes: Authorization was applied after reranking or a filter was misconfigured.
First discriminating check: Audit data flow and boundary-specific request payloads.
Disable the external reranker and prevent restricted candidate text leaving the trust boundary.
Proposed correction: Move live authorization before any external reranker call.
Mitigation boundary: A trusted in-boundary reranker still requires least-privilege controls.
Verification plan: Use forbidden canaries and inspect approved test-provider traces.
Security reviews provider data handling and verifies authorization occurs before any external transfer.
### 12.27. Prompt contains excessive source text
Trigger: A relevant detail is lost or latency spikes.
Too much context can bury qualifiers and consume capacity without improving the evidence set.
Competing causes: Too many chunks or duplicate context may exhaust the input budget.
First discriminating check: Measure tokens by component and inspect duplicate passages.
Reduce selected evidence to complete authorized spans rather than truncating arbitrary text.
Proposed correction: Select minimum sufficient distinct evidence and preserve exceptions.
Mitigation boundary: Aggressive truncation can remove a policy qualifier.
Verification plan: Test long documents with critical text near chunk boundaries.
The retrieval owner evaluates relevance, qualifier retention and token use together.
### 12.28. Answer invents currency
Trigger: The policy states a number but no currency.
A number without a currency cannot justify an answer specifying dollars, euros or another unit.
Competing causes: The model may fill a familiar currency or parser may have omitted a header.
First discriminating check: Check the original page and extracted table header.
State the missing unit or abstain; do not fill it in from locale assumptions.
Proposed correction: State units only when supported and ask for clarification otherwise.
Mitigation boundary: Suppressing the amount entirely may be safer than guessing but less useful.
Verification plan: Test missing-unit and inherited-header examples.
The policy owner adds the unit to the source or confirms the authoritative context.
### 12.29. Historical request uses current policy
Trigger: The employee asks about last year but receives today’s cap.
A historical expense question requires the policy effective on the event date, not necessarily today.
Competing causes: Date parsing may ignore the requested interval or the index may retain only current documents.
First discriminating check: Inspect interpreted date and source effective interval.
Ask for the relevant date when absent and search authorized historical versions when available.
Proposed correction: Use explicit as-of selection when authorized historical sources exist.
Mitigation boundary: Retention requirements may prohibit keeping certain old content.
Verification plan: Test historical, current and ambiguous-time questions.
The policy owner checks that the chosen interval covers the event and the user can access it now.
### 12.30. Question includes personal details
Trigger: An employee includes a medical condition in a travel query.
A medical detail in a question introduces sensitive data unrelated to the routine policy lookup.
Competing causes: The application may log raw prompts or send unnecessary details to a provider.
First discriminating check: Inspect governed logs and model-gateway payload under approved access.
Avoid echoing the detail and apply approved redaction before provider transmission and logging.
Proposed correction: Minimize and redact unnecessary fields before external processing.
Mitigation boundary: Redaction can remove context needed for the policy question.
Verification plan: Use synthetic sensitive examples and check utility and leakage.
Privacy owners review retention, access controls and whether the question needs human handling.
### 12.31. Feedback leaks a document
Trigger: A thumbs-down report stores the whole restricted source.
Feedback attachments can copy restricted policy text into a less protected support queue.
Competing causes: Feedback UI may include the full context or an operator export may widen access.
First discriminating check: Trace feedback payload and access permissions.
Disable automatic source attachment and restrict already captured feedback under the deletion policy.
Proposed correction: Store minimal identifiers and gate source reconstruction by authorization.
Mitigation boundary: Removing all context can make error triage difficult.
Verification plan: Review feedback exports and test least-privilege access.
The support and privacy owners inspect the feedback data path and access permissions.
### 12.32. Metric rewards unsafe guessing
Trigger: A score rises because guessed answers match common values.
A guessed common cap can be numerically correct while lacking supporting authorized evidence.
Competing causes: The benchmark may score only string matching or omit unanswerable questions.
First discriminating check: Inspect labels, authorized evidence and adjudication rules.
Do not reward a match unless claim-level support and authorization also pass.
Proposed correction: Score supported answers and abstention separately.
Mitigation boundary: Overly strict evidence matching can reject valid alternative passages.
Verification plan: Use adversarial unsupported-but-plausible answer cases.
The evaluation lead separates answer accuracy, evidence support and unauthorized disclosure metrics.
### 12.33. Unanswerable prevalence changes
Trigger: Aggregate score falls after a new user cohort joins.
A cohort with more unanswerable questions changes aggregate performance without changing conditional behavior.
Competing causes: The cohort may ask harder questions or the system may have regressed.
First discriminating check: Compare stratified rates and mix-adjusted results over time.
Report answerable and unanswerable performance separately before attributing the shift to the model.
Proposed correction: Investigate cohort-specific performance before changing the system.
Mitigation boundary: Mix adjustment cannot replace raw user-impact reporting.
Verification plan: Recompute metrics with fixed and observed population weights.
The analyst standardizes the cohort mix and reviews per-group denominators.
### 12.34. Evaluation labels are stale
Trigger: A correct current answer is judged wrong against an old cap.
A frozen label tied to an old policy can incorrectly penalize a correct new answer.
Competing causes: Policy changed after labels were created or the answer ignored as-of time.
First discriminating check: Compare label snapshot, policy revision and request time.
Suspend comparisons against mismatched policy snapshots until the label is re-adjudicated.
Proposed correction: Version labels by source snapshot and relabel when appropriate.
Mitigation boundary: Relabeling after seeing model output risks biased adjudication.
Verification plan: Test a policy update against frozen historical and new-current sets.
The evaluation owner records the policy revision and effective time used for each label.
### 12.35. Near-duplicate test leakage
Trigger: Holdout results are unusually high.
Paraphrases of the same source question in tuning and holdout sets inflate apparent generalization.
Competing causes: Paraphrases may occur in both tuning and test partitions.
First discriminating check: Cluster questions by semantic and source relation before partitioning.
Quarantine related question families from the final holdout and rerun selection on a clean split.
Proposed correction: Move related variants into a single split and freeze it.
Mitigation boundary: Over-clustering may reduce coverage of legitimate variants.
Verification plan: Audit nearest neighbors across partitions with human review.
The evaluation owner groups by source policy and question intent before partitioning.
### 12.36. Model judge agrees with wrong answer
Trigger: Automated evaluation marks an unsupported answer correct.
A model judge may prefer a fluent answer despite a missing or contradictory citation.
Competing causes: The judge may rely on general knowledge or share generator biases.
First discriminating check: Compare blinded expert decisions and supplied source evidence.
Route disputed and high-risk claims to human adjudication rather than accepting the judge alone.
Proposed correction: Use expert adjudication on risk-critical cases and calibrate judge outputs.
Mitigation boundary: Human reviewers can also disagree and need an adjudication protocol.
Verification plan: Track disagreement by source type and answerability stratum.
The evaluation lead measures judge agreement and audits false approvals against source spans.
### 12.37. Manual baseline is disadvantaged
Trigger: Candidate appears faster because baseline users lack portal training.
An untrained manual-search cohort makes a speed comparison favor the assistant for the wrong reason.
Competing causes: Unequal onboarding or different corpus snapshots may confound comparison.
First discriminating check: Audit assigned training, tasks and portal access.
Pause interpretation of the comparison until both workflows receive equivalent onboarding.
Proposed correction: Standardize context and randomize task order where feasible.
Mitigation boundary: Extra training may not reflect normal operating conditions.
Verification plan: Report both realistic and controlled baseline conditions.
The study owner balances participants and tasks and records whether answers were correct.
### 12.38. User overtrusts the answer
Trigger: The user treats guidance as an expense approval.
An employee can mistake a cited informational response for a binding reimbursement decision.
Competing causes: The UI may omit scope language or make citations hard to inspect.
First discriminating check: Observe task behavior in a privacy-approved usability study.
Display the informational boundary alongside the answer and link to the actual approval workflow.
Proposed correction: Make informational scope and source links salient.
Mitigation boundary: A disclaimer alone does not prevent overreliance.
Verification plan: Test whether users verify consequential claims in realistic tasks.
The product owner tests whether users understand the distinction before exposing the feature broadly.
### 12.39. Escalation channel is unavailable
Trigger: An abstaining user cannot reach a policy owner.
A safe abstention is not operationally useful if no policy support route can receive the question.
Competing causes: The help-desk route may be wrong or the queue may be overloaded.
First discriminating check: Test the link and confirm owner coverage and queue status.
Show available portal guidance and support hours without promising immediate resolution.
Proposed correction: Provide an approved alternate route or clearly state service hours.
Mitigation boundary: Routing every low-risk query to a human can overwhelm the queue.
Verification plan: Monitor escalation delivery and resolution without exposing sensitive queries.
The support owner monitors queue availability and defines an outage contact path.
### 12.40. A source owner changes
Trigger: A new version lacks an accountable approver.
A published policy without an accountable owner may lack a reliable conflict-resolution path.
Competing causes: Ownership metadata may be stale or an organizational transfer may be incomplete.
First discriminating check: Compare portal ownership with approved governance records.
Exclude the unowned version from answerable evidence until responsibility is assigned.
Proposed correction: Quarantine updates until the owner is confirmed.
Mitigation boundary: Excessive quarantine can leave the index outdated.
Verification plan: Exercise ownership transfer in the runbook.
Governance confirms the new owner and publication authority before release.
### 12.41. Question is ambiguous
Trigger: The user asks about meals without specifying location.
A meal question without location may have several mutually exclusive applicable limits.
Competing causes: The policy may depend on jurisdiction or trip type.
First discriminating check: Inspect the required policy qualifiers and the question context.
Ask the employee to clarify the relevant location instead of choosing a default jurisdiction.
Proposed correction: Ask a neutral clarification without revealing hidden policies.
Mitigation boundary: Repeated questions can degrade usability.
Verification plan: Test ambiguous prompts and verify no unsupported assumptions.
The policy team specifies which location attributes determine applicability.
### 12.42. Answer combines different effective dates
Trigger: A single response merges a current cap with a superseded exception.
A current cap paired with an old exception can create a rule that never existed at any time.
Competing causes: Retriever may collect versions independently or context assembly may omit date metadata.
First discriminating check: Trace effective intervals for every selected chunk.
Suppress the composite response and rebuild evidence for one coherent effective instant.
Proposed correction: Require a consistent as-of context and flag incompatible evidence.
Mitigation boundary: A comparison question may legitimately require multiple dates.
Verification plan: Test both point-in-time and explicit change-over-time questions.
The retrieval owner verifies interval overlap and provenance of both cited passages.
### 12.43. System hides uncertainty
Trigger: The response sounds confident despite ambiguous evidence.
Fluent language can hide unresolved policy ambiguity from a reader.
Competing causes: Prompt style or a postprocessing template may remove uncertainty.
First discriminating check: Compare evidence gate output with rendered text.
Use explicit uncertainty and escalation when evidence does not settle the question.
Proposed correction: Preserve calibrated uncertainty and abstain when policy interpretation is unresolved.
Mitigation boundary: Confidence language is not a substitute for evidence support.
Verification plan: Review uncertain cases with policy owners and users.
The evaluation lead checks whether the answer states both the conflict and its practical boundary.
### 12.44. Unknown language request
Trigger: An employee submits a question in an unsupported language.
An unsupported-language question may be misinterpreted even if the model produces fluent text.
Competing causes: Language detection can misclassify a short query or translation can distort policy terms.
First discriminating check: Check explicit language preference and a privacy-safe detection result.
Explain the English-only scope without guessing the meaning of sensitive content.
Proposed correction: Route unsupported cases to the portal or an approved human service.
Mitigation boundary: A blanket language block can exclude valid bilingual queries.
Verification plan: Test short mixed-language prompts and false detections.
The product owner requires separate multilingual evaluation before expanding coverage.
### 12.45. Lexical index misses a synonym
Trigger: A paraphrase returns unrelated results.
A keyword index can miss a concept when the employee uses different vocabulary from the source.
Competing causes: Vocabulary mismatch or a stale index can produce the same symptom.
First discriminating check: Check document presence and compare lexical with embedding retrieval.
Offer manual search and inspect the query terms before changing the retrieval method.
Proposed correction: Improve query expansion or embedding retrieval after diagnosis.
Mitigation boundary: Embedding matches can miss exact version identifiers.
Verification plan: Compare stratified exact-term and paraphrase questions.
The search owner tests controlled synonyms and compares them with authorized vector retrieval.
### 12.46. Embedding index mixes models
Trigger: Relevant pages disappear after a model upgrade.
An index built with one embedding model cannot safely be queried using an unrelated model’s vectors.
Competing causes: Query and document vectors may use incompatible embedding versions.
First discriminating check: Compare embedding version in request with index manifest.
Restore a matching query encoder and index manifest as a single release unit.
Proposed correction: Rebuild and atomically switch matching query and document encoders.
Mitigation boundary: A rollback must not reintroduce revoked content.
Verification plan: Run compatibility checks before index promotion.
The release owner verifies embedding identifiers across builders, stores and serving instances.
### 12.47. Token budget hides qualifier
Trigger: The retrieved passage contains an exception beyond the prompt cutoff.
A prompt cutoff can remove an exception that immediately follows an otherwise complete rule.
Competing causes: Truncation policy or tokenization differences can drop material.
First discriminating check: Inspect the exact model input and compare with source chunk.
Reject evidence bundles that split required qualifiers from the governing clause.
Proposed correction: Preserve qualifiers and reject context that cannot fit safely.
Mitigation boundary: Raising context size can increase cost and dilute attention.
Verification plan: Use boundary fixtures where the qualifier is near truncation.
The context assembly owner tests token-budget boundaries using long source fixtures.
### 12.48. Index growth raises operating cost
Trigger: Storage and query cost exceed an approved budget.
Retaining many revisions and derived vectors can increase storage and index maintenance expense.
Competing causes: Duplicate versions or oversized embeddings may both contribute.
First discriminating check: Attribute cost by version, source, index and traffic.
Pause nonessential backfills while preserving mandatory source retention and access controls.
Proposed correction: Remove expired derived copies under retention rules and right-size after measurement.
Mitigation boundary: Deleting retained historical versions can break required historical answers.
Verification plan: Compare indexed inventory and cost before and after the controlled change.
The platform owner attributes cost to source versions, vector replicas and query volume before tuning.
### 12.49. Burst load overloads dependencies
Trigger: Normal average throughput looks safe but requests time out at opening time.
A busy-start burst can exhaust a dependency even when the day’s mean throughput is low.
Competing causes: A burst or retry storm can exceed per-second capacity.
First discriminating check: Inspect fine-grained arrival, concurrency and dependency saturation.
Apply bounded queues and safe overload responses instead of unbounded retries.
Proposed correction: Use admission control and bounded backoff with a portal fallback.
Mitigation boundary: Overly aggressive shedding can block urgent legitimate requests.
Verification plan: Run a safe load test at a measured representative burst profile.
Operations reviews time-bucketed arrivals, concurrency, dependency limits and retry amplification.
### 12.50. User cannot open cited source
Trigger: A link is generated but the portal rejects access.
A citation that cannot be opened does not let the employee independently inspect the evidence.
Competing causes: Access may have been revoked between generation and click or the link may be malformed.
First discriminating check: Compare link target, source status and current authorization.
Avoid presenting the broken link as actionable and offer the authorized portal route.
Proposed correction: Recheck citations before display and provide a generic access-safe fallback.
Mitigation boundary: A later revocation remains possible after the response is rendered.
Verification plan: Exercise permission changes across request and click boundaries.
The connector owner checks link lifetime, identity propagation and document-level portal permissions.
### 12.51. Incomplete deletion from audit store
Trigger: A removed policy is absent from search but persists in traces.
Deleting a source from search does not remove copies in audit traces or feedback stores.
Competing causes: Trace retention or backup lifecycle may be independent of index purge.
First discriminating check: Map all derived stores and retention obligations.
Restrict the affected records while applying the approved deletion or retention procedure.
Proposed correction: Execute the approved deletion workflow across applicable stores.
Mitigation boundary: Legal holds or audit obligations may constrain deletion.
Verification plan: Verify each store and document exceptions with the responsible owner.
Governance reconciles indexed content, logs and legal retention obligations with store owners.
### 12.52. Model update changes behavior
Trigger: The same evidence yields more confident unsupported claims.
A provider model change can alter support behavior even when retrieval and prompts are unchanged.
Competing causes: Provider model drift or prompt-version mismatch may be responsible.
First discriminating check: Compare pinned model identifiers and replay the frozen evaluation set.
Pin or roll back the approved model version where supported and retain the safe fallback.
Proposed correction: Pin or roll back the configuration and reassess before release.
Mitigation boundary: Some providers cannot guarantee exact replay across backend changes.
Verification plan: Track behavioral regressions on a fixed authorized test suite.
The model owner compares fixed evidence fixtures and claim-support judgments across versions.
### 12.53. Parser changes citation offsets
Trigger: A citation highlights the wrong sentence after reindexing.
Parser normalization can shift text offsets so a link highlights an adjacent sentence.
Competing causes: Normalization or character-encoding changes can shift offsets.
First discriminating check: Compare source-version hash, parser revision and stored offsets.
Disable unreliable deep links while retaining a verified document-level citation if policy permits.
Proposed correction: Bind offsets to exact normalized source version and rebuild links.
Mitigation boundary: Offsets cannot be reused across content revisions safely.
Verification plan: Test Unicode and whitespace transformations against source spans.
The ingestion owner binds offsets to the exact parsed version and tests round-trip highlighting.
### 12.54. Policy update arrives during generation
Trigger: A document is revoked after retrieval but before response.
A permission or policy change after retrieval creates a time-of-check to time-of-use gap.
Competing causes: An access change or a source withdrawal can create a time-of-check race.
First discriminating check: Trace authorization and source-version times at each boundary.
Recheck authorization and applicability immediately before delivery and discard stale evidence.
Proposed correction: Reauthorize and revalidate just before delivery.
Mitigation boundary: No distributed check can make an external click permanently authorized.
Verification plan: Use a deterministic revocation fixture at each pipeline stage.
Security tests revocation during generation and defines the remaining consistency window.
### 12.55. Fallback itself exposes data
Trigger: An error message includes a restricted retrieval snippet.
An exception handler can accidentally echo a restricted snippet into an otherwise safe error page.
Competing causes: A generic exception logger or debug handler may render source text.
First discriminating check: Review error-path templates and restricted logs.
Return a generic error and restrict diagnostic payloads to authorized operators.
Proposed correction: Return generic messages and keep protected diagnostic identifiers.
Mitigation boundary: Removing all diagnostics can prolong recovery.
Verification plan: Inject exceptions after retrieval and inspect user-visible output.
The service owner injects failures at each stage and inspects response bodies and logs.
### 12.56. Ranking favors a popular obsolete page
Trigger: An older frequently viewed policy outranks the current one.
Popularity can keep an obsolete policy highly ranked after a new version takes effect.
Competing causes: Behavioral popularity may dominate authority or freshness features.
First discriminating check: Inspect rank features alongside effective status.
Filter on applicable status before applying popularity-based ranking signals.
Proposed correction: Apply eligibility and authority constraints before ranking.
Mitigation boundary: Popularity may still help rank equally authoritative current pages.
Verification plan: Test popular old pages against current low-traffic policies.
The retrieval owner compares rank traces for old and current revisions with the same query.
### 12.57. Conflicting access granularity
Trigger: A user may see a document but not one restricted section.
Document-level access does not imply permission to see all sections of a mixed-access document.
Competing causes: Document-level ACL inheritance may ignore section overrides.
First discriminating check: Compare section ACL with the inherited document entitlement.
Apply the most restrictive section decision before any chunk reaches a model or reranker.
Proposed correction: Check the most restrictive applicable resource permission.
Mitigation boundary: Overly broad section filtering can hide authorized surrounding text.
Verification plan: Test mixed-permission sections and chunk boundary inheritance.
Security verifies inheritance and deny precedence using mixed-permission section fixtures.
### 12.58. Evaluation misses rare policy groups
Trigger: Overall score is high while a small region receives wrong answers.
A strong aggregate metric can conceal systematically poor performance in a small jurisdiction.
Competing causes: Sampling imbalance or a region-specific parser defect can produce this pattern.
First discriminating check: Inspect subgroup counts, intervals and representative failures.
Report the affected group separately and do not average away critical policy errors.
Proposed correction: Oversample critical groups for diagnosis and report appropriate weights.
Mitigation boundary: Small subgroup estimates remain uncertain even after oversampling.
Verification plan: Track per-group coverage and require policy-owner review.
The evaluation lead collects more appropriately governed cases and reports uncertainty for the slice.
### 12.59. Unsafe rollback
Trigger: A rollback restores a previously removed source.
Rolling back an old index can resurrect a document that has since been revoked or deleted.
Competing causes: The old manifest may precede a revocation or deletion.
First discriminating check: Compare rollback snapshot to the current deny and deletion ledger.
Keep current deny rules and deletion tombstones active independently of the rolled-back index.
Proposed correction: Enforce current revocations independently of index version.
Mitigation boundary: Retaining old snapshots can conflict with deletion obligations.
Verification plan: Rehearse rollback following a synthetic revocation.
The release owner rehearses rollback with post-snapshot revocations and verifies all replicas.
### 12.60. Successful answer lacks actionable source
Trigger: The answer is supported but user cannot locate the actual passage.
A supported answer is hard to audit if its citation lands at the top of a long document.
Competing causes: Citation may link only to a portal homepage or strip the section anchor.
First discriminating check: Check source URI and section mapping against the rendered document.
Provide a verified section anchor or passage locator without bypassing portal authorization.
Proposed correction: Link to a stable version and precise section where available.
Mitigation boundary: Some source systems cannot preserve anchors across revisions.
Verification plan: Test link resolution and user navigation under normal permissions.
The product owner checks that users can navigate from the citation to the supporting text.
## 13. Transferable Lesson
Treat enterprise RAG as an evidence-and-permissions workflow, not a model that happens to see documents.
The transferable principle is to bind each answer to authorized, applicable, sufficient evidence and to preserve an abstention path.
Manual search is a serious comparator because generated fluency is not an outcome.
This principle applies when authoritative source documents and enforceable user-resource permissions exist.
It is weaker when the source corpus is incomplete, policy authority is disputed, or access decisions cannot be evaluated reliably.
In those settings, improve governance or keep human-mediated search rather than claiming the model can repair missing authority.
The lesson is a hypothesis supported by mechanism and the illustrative trace, not by measured deployment outcomes.
## 14. Connections & Cross-References
### 14a. Related cases, theory, and sources
The [NLP/LLM home](../INDEX.md) identifies the applicable subject area and its planned learning scope.
The [source-coverage brief](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-cases-05) records the production-case topic inventory.
The [authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) define the evidence and calculation conventions used here.
The [collection index](INDEX.md) provides local pattern definitions and distinguishes this draft from planned cases.
A related planned content-moderation case would share evaluation concerns but has a different decision and error taxonomy.
A related planned semantic-search case would share retrieval concerns but need not generate policy answers or enforce identical employee entitlements.
Those planned cases are contrasts to investigate, not evidence that their designs have been authored or evaluated.
Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” NeurIPS 2020, https://arxiv.org/abs/2005.11401, describes retrieval-augmented generation as a research approach.
The arXiv abstract, first published 2020-05-22 and accessed 2026-11-07, describes combining parametric sequence generation with a dense non-parametric index.
Scope of that paper: research model architecture and benchmark experiments, not proof of this enterprise design's security or performance.
NIST SP 800-162, “Guide to Attribute Based Access Control (ABAC) Definition and Considerations,” January 2014 with updates through 2019-08-02, https://csrc.nist.gov/pubs/sp/800/162/upd2/final, supplies access-control concepts.
The official abstract, accessed 2026-11-07, defines authorization in terms of subject, object, operation and potentially environment attributes evaluated against policy.
Scope of the guide: general access-control concepts, not a certification of the proposed implementation or a substitute for current organizational policy.
Recheck the research and NIST publication versions when the source, access design or curriculum review changes.
No external research source is claimed to describe a real deployment in this fictional case.
### 14b. What this case leaves open
Open question: Does hybrid retrieval outperform lexical search for this policy corpus?
Evidence required: frozen, permission-aware labeled questions and paired retrieval results.
Decision owner: evaluation lead; trigger: before selection of the retrieval approach.
Open question: Can the authoritative entitlement service meet the freshness and availability requirements?
Evidence required: operational semantics, measured dependency behavior, revocation tests, and approved risk budget.
Decision owner: security and platform owners; trigger: architecture review.
Open question: Which unsupported-answer and disclosure risks are acceptable for a pilot?
Evidence required: policy-owner risk assessment, legal/privacy review and preregistered stop rules.
Decision owner: product and risk owners; trigger: pilot approval.
Open question: Does assistance improve employee task outcomes compared with a realistic portal baseline?
Evidence required: adequately powered, privacy-reviewed randomized task comparison and subgroup analysis.
Decision owner: product research lead; trigger: decision on continued investment.
Open question: How are source updates, legal holds and deletion requirements reconciled?
Evidence required: authoritative retention schedule, data-flow inventory, source-owner policy and tested deletion procedure.
Decision owner: governance lead; trigger: before ingesting production data.
### 14c. Intense interview pack
Candidate prompts are separated from interviewer notes below.
Scores use 0 for incorrect, 1 for partial, 2 for correct with assumptions, and 3 for a defended tradeoff.
#### Q01 / Framing / practitioner
Candidate prompt: What is the bounded decision and who owns it?
Given facts: Assume an internal employee policy portal and no authority to approve expenses.
Strong answer: The product owner decides whether to test cited assistance against manual search; a policy owner owns interpretation and the tool performs no approval.
Reasoning: Define the user, output, comparison, and non-negotiable authorization boundary before choosing a model.
Acceptable alternative: A search-only improvement can be preferable if it satisfies the task without generation.
Follow-up: What if expense approval is requested?
Follow-up answer: Add a separate decision, permissions, confirmation and audit design; do not silently extend this case.
Red flag: Treating an answer as a binding approval conflates guidance and action.
Answer support: Sections 1–4 and 8.
#### Q02 / Evidence / practitioner
Candidate prompt: What does the illustrative evidence establish?
Given facts: The 100-question table was constructed for teaching rather than collected from users.
Strong answer: It establishes only arithmetic and the consequences of specified assumptions, not observed quality or a business outcome.
Reasoning: Separate illustrative inputs, derived calculations, proposed architecture, and unmeasured results.
Acceptable alternative: A published research result can support a general mechanism if the cited scope is accurately described.
Follow-up: What evidence would support a pilot?
Follow-up answer: Frozen representative evaluation, access-control tests, approved thresholds and owner sign-offs are necessary inputs.
Red flag: Calling a synthetic 80% an actual retrieval measurement is fabrication.
Answer support: Sections 6, 9 and 11.
#### Q03 / Data / practitioner
Candidate prompt: What is the correct unit for answerability labels?
Given facts: Users have different access rights and policies have effective dates.
Strong answer: Label each question against a user authorization context and a specific corpus snapshot or effective time.
Reasoning: The same text question can be answerable for one user and unanswerable for another.
Acceptable alternative: A role-level fixture can substitute only if all relevant users have identical verified access.
Follow-up: What if a user loses access during the request?
Follow-up answer: Reauthorize selected sources before delivery and fail closed if the check cannot be completed.
Red flag: Scoring against all documents regardless of access makes an unsafe system appear better.
Answer support: Sections 5a, 5d and 8b.
#### Q04 / Calculation / practitioner
Candidate prompt: Compute complete evidence coverage for 64 successes among 80 answerable questions.
Given facts: All required authorized evidence must occur within the first five candidates.
Strong answer: 64/80 = 0.8 or 80%; 16 answerable questions do not meet this retrieval criterion.
Reasoning: The unit is questions divided by questions and the denominator excludes unanswerable questions.
Acceptable alternative: Report recall at another cutoff if the cutoff is fixed and the definition remains explicit.
Follow-up: What if eight missing cases become complete?
Follow-up answer: 72/80 = 90%, with the same denominator and labeling assumptions.
Red flag: Using 100 total questions as the denominator changes the metric.
Answer support: Section 6a, R-01.
#### Q05 / Calculation / practitioner
Candidate prompt: Compute the unsupported-answer fraction on unanswerable cases.
Given facts: There are 20 unanswerable questions; 18 correct abstentions and 2 unsupported answers.
Strong answer: 2/20 = 10%, while correct abstention is 18/20 = 90%.
Reasoning: The two rates complement each other only because these constructed outcomes are exhaustive and exclusive.
Acceptable alternative: Report raw counts along with fractions, especially for this small sample.
Follow-up: If one more abstention becomes unsupported?
Follow-up answer: The rate becomes 3/20 = 15%; it is not the overall error rate for all 100 questions.
Red flag: Claiming 10% applies to all users or all questions ignores the denominator.
Answer support: Section 6a, R-02 and R-03.
#### Q06 / Capacity / practitioner
Candidate prompt: Estimate average busy-interval request rate.
Given facts: 12,000 requests/day; 80% arrive in four hours uniformly; ignore retries.
Strong answer: 9,600/14,400 = about 0.667 requests/s and, at two searches per request, about 1.333 searches/s.
Reasoning: Convert hours to seconds and multiply the request rate by searches per request.
Acceptable alternative: A trace-based simulator is better when realistic burst and dependency distributions are available.
Follow-up: What if the same busy volume arrives in one hour?
Follow-up answer: It becomes 2.667 requests/s and 5.333 searches/s, still excluding retries and bursts.
Red flag: Calling the four-hour mean a peak or latency guarantee is unjustified.
Answer support: Section 6b, W-01 and W-02.
#### Q07 / Security / practitioner
Candidate prompt: Where should permission checks happen?
Given facts: The reranker or generator might be an external service.
Strong answer: Check authoritative permissions before restricted text crosses each less-trusted boundary and recheck selected evidence before response.
Reasoning: Index filters may accelerate lookup but can be stale after a revocation.
Acceptable alternative: A verified short-lived entitlement snapshot may be acceptable only under explicitly approved revocation semantics.
Follow-up: What if the authorization service times out?
Follow-up answer: Fail closed, emit a generic unavailable response and offer the authorized portal fallback.
Red flag: Sending all documents to the model and filtering the final answer is not access control.
Answer support: Sections 4 and 8b.
#### Q08 / Citation / practitioner
Candidate prompt: Does a valid citation ID prove a claim?
Given facts: A generated sentence includes an ID present in the selected evidence.
Strong answer: No; verify that the source is authorized, applicable and actually supports the atomic claim.
Reasoning: Identifier membership establishes provenance linkage, not entailment or completeness.
Acceptable alternative: Human review can adjudicate difficult policy claims when an automated check is uncertain.
Follow-up: What if the source supports only part of the sentence?
Follow-up answer: Split the claim and cite each supported part or abstain from the unsupported portion.
Red flag: Treating any clickable link as proof is a citation-quality error.
Answer support: Sections 6c and 9.
#### Q09 / Diagnosis / practitioner
Candidate prompt: Why might a known policy not appear in an answer?
Given facts: The symptom alone does not identify the failing component.
Strong answer: Check source ingestion and manifest first, then retrieval candidates, authorization, applicability and evidence selection.
Reasoning: A missing index entry, a vocabulary mismatch and a legitimate access denial require different remedies.
Acceptable alternative: Start from a trace ID or a controlled known-item fixture if a user report lacks detail.
Follow-up: What if the document is indexed but unauthorized?
Follow-up answer: Do not tune ranking to surface it; verify the entitlement decision with authorized security staff.
Red flag: Increasing top-k without checking access or ingestion is an undirected and potentially unsafe fix.
Answer support: Sections 8 and 12.
#### Q10 / Wrong method / practitioner
Candidate prompt: Why not supply every company document to a long-context model?
Given facts: Some documents are restricted and source versions can be stale.
Strong answer: The model would receive data the user may not access and may blend incompatible or obsolete sources.
Reasoning: Context capacity is not an authorization engine, source-of-truth resolver, or evidence validator.
Acceptable alternative: Long context over an already authorized, bounded set can be evaluated if privacy and applicability are preserved.
Follow-up: What if all users have identical access?
Follow-up answer: Version selection, context relevance, injection, cost and citation support still need testing.
Red flag: Assuming a model will reliably ignore forbidden text is an invalid security control.
Answer support: Sections 5c and 8.
#### Q11 / Evaluation / practitioner
Candidate prompt: How should the candidate be compared with manual search?
Given facts: A decision maker wants evidence of task improvement, not only better retrieval scores.
Strong answer: Use matched or randomized eligible tasks, the same authorized corpus, independent outcome scoring and latency/cost/safety guardrails.
Reasoning: Report answerability strata, sample sizes, uncertainty, missing outcomes and subgroup performance.
Acceptable alternative: A paired offline benchmark is useful for development but cannot alone establish employee productivity impact.
Follow-up: What if randomization is impossible?
Follow-up answer: Report an observational comparison with its confounding limits rather than calling it causal.
Red flag: Comparing different users or corpora without qualification can create a false advantage.
Answer support: Section 9.
#### Q12 / Uncertainty / practitioner
Candidate prompt: Interpret the approximate interval for the illustrative coverage fraction.
Given facts: Assume 64 successes in 80 independent representative Bernoulli trials for this teaching calculation.
Strong answer: The standard error is about 0.0447; the approximate normal 95% interval is 0.712 to 0.888.
Reasoning: Compute sqrt(0.8*0.2/80), multiply by 1.96 and state sampling assumptions.
Acceptable alternative: A Wilson or exact binomial interval may be more appropriate; cluster-aware methods handle grouped observations.
Follow-up: What if questions are paraphrases from the same employee?
Follow-up answer: Independence may fail; use a cluster-aware design or resampling scheme and report the effective population.
Red flag: Describing the interval as a guarantee for each future request is wrong.
Answer support: Section 9a.
#### Q13 / Changed constraint / practitioner
Candidate prompt: How does the design change for historical policy questions?
Given facts: The user asks which rule applied on a previous trip date.
Strong answer: Select sources effective at the requested date and authorized for the requester; cite the date and version.
Reasoning: Current-only filtering would answer a different question, while retaining old content is subject to governance.
Acceptable alternative: If authorized history is unavailable, abstain and direct the user to the policy owner.
Follow-up: What if the date is unspecified?
Follow-up answer: Ask a neutral clarification or explicitly bound the answer to the current date if that is appropriate.
Red flag: Blindly selecting the latest published document ignores the requested time.
Answer support: Sections 5a and 12.
#### Q14 / Injection / practitioner
Candidate prompt: How should the system handle a retrieved instruction to ignore access controls?
Given facts: The instruction appears inside a retrieved policy document.
Strong answer: Treat it as untrusted source content; never grant it instruction or tool authority and do not disclose hidden content.
Reasoning: Separate application instructions and authorization enforcement from the source data passed to the model.
Acceptable alternative: Quarantine the document if compromised and investigate its revision with the source owner.
Follow-up: What if a legitimate policy itself contains instructions to employees?
Follow-up answer: The assistant may summarize the policy instruction as evidence without obeying it as a system command.
Red flag: Assuming prompt wording alone enforces the access boundary is insufficient.
Answer support: Sections 4 and 8b.
#### Q15 / Operations / practitioner
Candidate prompt: Describe safe rollback after a source is revoked.
Given facts: An older index manifest exists and a bad model release must be reversed.
Strong answer: Roll back the model or compatible manifest while retaining current revocation enforcement and deletion obligations.
Reasoning: An old snapshot can contain a now-forbidden document, so rollback must not restore its accessibility.
Acceptable alternative: Disable generation and use the portal until a safe bundle is available.
Follow-up: What if the permission service is down during rollback?
Follow-up answer: Fail closed; do not rely on the old manifest to authorize documents.
Red flag: Restoring an old vector index without a current deny mechanism can re-expose revoked data.
Answer support: Sections 8a and 12.
#### Q16 / Tradeoff / practitioner
Candidate prompt: How can excessive abstention be improved without encouraging guesses?
Given facts: Many answerable questions are refused, but current unsupported-answer risk is unknown.
Strong answer: Identify whether evidence is absent from the corpus, retrieval candidates, selected context or answerability gate; fix the earliest cause and remeasure both outcomes.
Reasoning: Abstention and unsupported answers are separate metrics with different denominators.
Acceptable alternative: Add an approved human escalation for ambiguous policies instead of lowering the gate.
Follow-up: What if the evidence is present but applicability is disputed?
Follow-up answer: Escalate to the policy owner rather than tuning a model to choose an interpretation.
Red flag: Optimizing answer rate alone rewards unsupported guesses.
Answer support: Sections 6a, 9 and 12.
## 15. Quick Reference Card
| Field | Value |
|---|---|
| Domain | Enterprise employee travel-policy knowledge |
| Problem type | Authorized question answering versus manual portal search |
| Method | Proposed permission-aware hybrid retrieval, evidence gate, cited answer or abstention |
| Primary result | Illustrative calculations only; empirical quality and outcomes not measured |
| Key lesson | Evidence must be sufficient, applicable and authorized; retrieval relevance is not enough |
| Tags | authorization-aware-retrieval; evaluation-mismatch; human-in-the-loop |
| Supplemental field | Value |
| Case ID / version / as-of | CASE-01-05-001 / 1.0-draft / 2026-11-07 |
| Evidence level | Illustrative design and arithmetic, with general research references |
| Implementation stage | Proposed; no deployment or pilot established |
| Baseline / primary metric | Manual portal search; supported-answer rate on defined eligible question set |
| Key limitation | No real corpus, permissions, labels, workload or results are available |
| Review trigger / owner | Permission, policy, corpus, or model change / knowledge-platform product owner |
