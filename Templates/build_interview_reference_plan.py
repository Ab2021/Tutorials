"""Maintain reference-informed planning blocks; never copy or modify source examples.

Run with --write to refresh, or --check to validate generated blocks and links.
The hand-curated decisions below are proposed writing briefs, not case outcomes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "Train/ComprehensiveAIML_BUSINESS_DOMAINS"
SOURCES = ROOT / "Train/INTERVIEW_CASES"
PLAN = LIB / "00_REFERENCE_INFORMED_CASE_PLAN.md"
REPORT = ROOT / "Train/Templates/INTERVIEW_REFERENCE_PLAN_VALIDATION_2026-11-06.md"
DATE = "2026-11-06"
START = "<!-- BEGIN INTERVIEW REFERENCE PLAN -->"
END = "<!-- END INTERVIEW REFERENCE PLAN -->"

# Each entry: source, proposed title, decision, mechanisms, evaluation,
# transfer boundary, primary pattern, secondary patterns, type, destinations.
# Numeric destinations are modules; named destinations are exact domain leaves.
ROWS = [
    (
        "Agentic_AI_System/AI_engg_challenges/synthetic_corpus_contamination.md",
        "Gate retraining on corpus provenance and long-tail quality",
        "Which documents should be admitted or quarantined before the next training run?",
        "Dataset lineage; synthetic-data feedback; deduplication limits; language and domain coverage",
        "Held-out long-tail factual accuracy, diversity and per-slice retention; compare filtering with provenance controls",
        "A synthetic-text classifier cannot establish authorship; source incident names and costs are unverified",
        "distribution-shift", ["data-leakage", "evaluation-mismatch"], "hybrid", [27, 24, 28],
    ),
    (
        "Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md",
        "Select model routes within hard customer constraints",
        "Which eligible provider and fallback path satisfy residency, capability, deadline and spend requirements?",
        "Eligibility before scoring; capability adapters; health-aware routing; bounded retry budgets",
        "Constraint violations, successful-task cost, deadline misses and fallback correctness on fault replays",
        "Revalidate provider capabilities and economics when authoring; source tables are not current service guarantees",
        "eligibility-before-ranking", ["cost-latency-tradeoff"], "llm-agentic", [25, 29, 52],
    ),
    (
        "Agentic_AI_System/Applied_AI_engineering/grounded_policy_rag.md",
        "Answer support questions from the applicable policy version",
        "When should a support assistant answer, abstain or escalate a policy question?",
        "Authorized retrieval; effective policy versions; claim-level citations; embedding migrations",
        "Unsupported-claim rate, eligible-span recall, abstention quality and policy-version errors against expert labels",
        "Citations and entailment checks require validation; never transfer the example refund window into a new business",
        "evaluation-mismatch", ["authorization-scope", "materialized-view"], "llm-agentic", [20, "enterprise", "omnichannel"],
    ),
    (
        "Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md",
        "Customize tenant assistants without crossing isolation boundaries",
        "How should tenant adapters be trained, selected, versioned and rolled back?",
        "LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints",
        "Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix",
        "Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements",
        "authorization-scope", ["atomic-release-bundle", "rollout-regression"], "llm-agentic", [21, 25, 49, "enterprise"],
    ),
    (
        "Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md",
        "Release and recover an agent as a compatible bundle",
        "Which model, prompt, tools, index and policy versions can be promoted and recovered together?",
        "Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment",
        "Bundle-compatibility tests, rollback drills, guardrail breaches and unresolved side effects after recovery",
        "A prompt-only rollback cannot restore all dependencies; example dwell times and canary percentages are not defaults",
        "atomic-release-bundle", ["rollout-regression", "human-in-the-loop"], "hybrid", [55, 26, 30, "insurance_ops"],
    ),
    (
        "Agentic_AI_System/Langraph/durable_human_approval_graph.md",
        "Recover procurement workflows without replaying approvals",
        "How can a purchase proceed only after approval of its exact version despite retries and restarts?",
        "LangGraph checkpoints and interrupts; authenticated resume; approval binding and expiry; idempotent submission",
        "Crash-window replay, duplicate webhooks, changed purchase payloads, checkpoint migration and reconciliation tests",
        "External APIs may not support exactly-once effects; approval waiting times and run counts are exercise assumptions",
        "idempotent-side-effects", ["human-in-the-loop", "snapshot-consistency"], "llm-agentic", [30, 31, "control_tower"],
    ),
    (
        "Agentic_AI_System/LLM_Inference/continuous_batching_serving.md",
        "Balance interactive inference latency and admitted throughput",
        "Which scheduling and KV-memory policy meets latency and isolation requirements at the admitted load?",
        "Continuous batching; chunked prefill; paged KV storage; prefix authorization; speculative-decoding correctness",
        "TTFT, inter-token latency, offered versus accepted load, KV headroom and target-equivalence tests",
        "GPU geometry, concurrency limits and speedups require workload measurements; speculative decoding can be slower",
        "resource-admission", ["cost-latency-tradeoff", "authorization-scope"], "hybrid", [25, 23, 19],
    ),
    (
        "Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md",
        "Block assistant releases on auditable quality and safety gates",
        "Does a candidate pass the required suites with enough evidence to justify promotion?",
        "Versioned evaluation manifests; contaminated benchmarks; calibrated judges; over-refusal; sandboxed code scoring",
        "Paired quality comparisons, critical-failure review, uncertainty, judge drift and missing-suite failure tests",
        "A judge score is not a deployment guarantee; workload volume and evaluation deadlines remain unspecified",
        "evaluation-mismatch", ["data-leakage", "rollout-regression"], "llm-agentic", [28, 49, "reliability"],
    ),
    (
        "Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md",
        "Allocate a domain-training budget with restart headroom",
        "Should a domain model be trained from scratch or continued from an existing checkpoint within an all-in budget?",
        "Scaling-law proxy sweeps; licensed corpus splits; parallelism; memory accounting; checkpoint recovery",
        "Held-out domain and general quality, full training cost, achieved utilization and restart/reproducibility drills",
        "The reference compute and GPU figures are assumptions; an internal legal checkpoint is not validated legal advice",
        "cost-latency-tradeoff", ["data-leakage", "atomic-release-bundle"], "hybrid", [24, 19, 21],
    ),
    (
        "Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md",
        "Compare specialized models with authorized shared-model baselines",
        "When do repository-aware completion or language adapters justify their added complexity?",
        "Fill-in-the-middle; repository/time splits; authorized code retrieval; multilingual sampling; language adapters",
        "Sandboxed functional correctness and latency for code; native-reviewed grounding and worst-language regressions for support",
        "Split this reference into two independent future narratives; code-test passage does not prove security or language accuracy",
        "data-leakage", ["authorization-scope", "evaluation-mismatch"], "llm-agentic", [33, 18, 21],
    ),
    (
        "Agentic_AI_System/Multimodal/Insurance_Document_VLM.md",
        "Compare document VLMs with OCR for adjuster assistance",
        "Which claim fields can be extracted reliably, and which bundles require adjuster review?",
        "Page and region provenance; thumbnail/tiling tradeoffs; schema validation; claim-level temporal splits",
        "Field accuracy and calibration by template, handwriting and scan quality; reviewer workload, evidence coverage and latency",
        "Neither a similarity match nor a VLM fraud suspicion justifies denial; source scale and patch arithmetic need fresh sizing",
        "human-in-the-loop", ["evaluation-mismatch", "delayed-selective-labels"], "hybrid", [22, 18, "insurance_ops"],
    ),
    (
        "Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md",
        "Preserve answer evidence through ingestion and filtered retrieval",
        "Which chunking and index design retrieves every required authorized span within the service budget?",
        "Structure-aware chunking; actual tokenizer limits; HNSW and exact filtered baselines; CDC; deletion propagation",
        "All-required-span success, filtered recall, duplicate context, ACL violations, freshness lag and per-tenant latency",
        "Reranking cannot recover a missing candidate; the source embedding limit and tenant count are not platform requirements",
        "materialized-view", ["authorization-scope", "evaluation-mismatch"], "llm-agentic", [20, 6, "enterprise"],
    ),
    (
        "Agentic_AI_System/RAG_Infra_Research/Regulatory_RAG_And_Reranking.md",
        "Retrieve regulations applicable to the question date and jurisdiction",
        "Which sources are eligible before ranking, and when is legal-owner review required?",
        "Effective intervals; recorded-time history; jurisdiction rules; hybrid retrieval; reranking and claim verification",
        "Boundary-date and amendment replays, eligible recall, wrong-version citations and expert-reviewed abstentions",
        "This is an architecture exercise, not current legal guidance; eligibility depends on maintained authoritative metadata",
        "eligibility-before-ranking", ["snapshot-consistency", "human-in-the-loop"], "llm-agentic", [20, 50, "insurance_risk"],
    ),
    (
        "Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md",
        "Separate extraction correctness from routing economics",
        "How should extraction contracts and model routes be accepted without confusing valid syntax with business success?",
        "Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing",
        "Gold field accuracy and reconciliation for extraction; full-path cost and grounded quality by tier for routing",
        "Contains two reference problems: author separate narratives; valid JSON and cheap token prices do not establish successful outcomes",
        "evaluation-mismatch", ["idempotent-side-effects", "cost-latency-tradeoff"], "llm-agentic", [31, 25, "insurance_ops"],
    ),
    (
        "Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md",
        "Publish reviewed coaching scripts from current product evidence",
        "Which generated scripts are fit for an approved, versioned coaching registry?",
        "Product/date eligibility; grounded generation; native-language review; source dependencies; supersession triggers",
        "Unsupported product claims, expired evidence, read-aloud time, reviewer agreement and agent usability",
        "Human approval and source maintenance remain necessary; the example word target is not a compliance standard",
        "human-in-the-loop", ["eligibility-before-ranking", "atomic-release-bundle"], "llm-agentic", [34, "enterprise", "insurance_ops"],
    ),
    (
        "Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md",
        "Distinguish speech, speaker and extraction errors in call analytics",
        "Which customer objections can be reported with reliable speaker attribution and evidence?",
        "Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation",
        "Human-gold WER, domain-term WER, role error, objection precision/recall and end-to-end report coverage",
        "Transcript matching does not prove audio fidelity; aggregate coaching themes must not become unvalidated individual scores",
        "evaluation-mismatch", ["authorization-scope", "human-in-the-loop"], "hybrid", [18, 34, "insurance_ops"],
    ),
    (
        "Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md",
        "Adapt domain ASR while preserving supported-domain quality",
        "Should a speech adapter be promoted, and to which evaluated product and language domains?",
        "Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment",
        "Paired WER with uncertainty by domain and code-switch slice, real-time factor and supported-route regression gates",
        "Model-generated transcripts are not independent gold; audio resampling does not restore missing frequencies",
        "rollout-regression", ["distribution-shift", "evaluation-mismatch"], "hybrid", [21, 18, 34],
    ),
    (
        "DSA/Algo_Design_Problems/CASE_STUDIES_01.md",
        "Specify correctness contracts before choosing production algorithms",
        "Which guarantees are needed for road routing, distributed admission and phrase retrieval?",
        "CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics",
        "Reference Dijkstra comparisons, admissible-heuristic checks, concurrent window simulations and brute-force phrase/ranking oracles",
        "Seven overlapping teaching units; source measurements are not independently verified and should not be added together",
        "correctness-contract", ["snapshot-consistency", "bounded-work"], "hybrid", [3, 4, 14, "mobility"],
    ),
    (
        "DSA/Algo_Design_Problems/CASE_STUDIES_02.md",
        "Keep routing correctness separate from ETA quality",
        "How should a navigation service publish legal optimal routes while separately evaluating predicted arrival time?",
        "Coherent traffic snapshots; CH compatibility; fallback Dijkstra; map matching; route versus ETA objectives",
        "Base-graph shortest-path oracle, closure and turn-restriction replays, ETA error and routing tail latency",
        "The OCR filename and ordinal ranges disagree; arbitrary traffic overlays do not preserve ordinary CH correctness",
        "snapshot-consistency", ["correctness-contract", "evaluation-mismatch"], "hybrid", [4, 3, "mobility", "delivery", "transport"],
    ),
    (
        "DSA/Algo_Design_Problems/CASE_STUDIES_03.md",
        "Translate local algorithm guarantees into distributed service contracts",
        "Which correctness and failure guarantees survive sharding, delayed updates and partial outages?",
        "Routing production behavior; coordinated rolling counters; distributed positional inverted indexes",
        "Coherent-snapshot queries, concurrent admission accounting, shard failure tests and search-reference comparisons",
        "Three units continue earlier routing, admission and search topics; this is not a new independent evidence dataset",
        "correctness-contract", ["resource-admission", "materialized-view"], "hybrid", [54, 4, 14],
    ),
    (
        "DSA/Algo_Design_Problems/CASE_STUDIES_04.md",
        "Evolve an authorized near-real-time search index safely",
        "How should search remain correct across indexing, deletion, ranking and shard failures?",
        "WAL and immutable segments; positional phrase queries; field-weighted BM25; exact top-k; ACLs and tombstones",
        "Brute-force Boolean/phrase and global top-k oracles, delete visibility, retrieval quality, index lag and merge capacity",
        "Exact global top-k needs a valid distributed protocol; figures in the source are scenario claims, not search-cluster benchmarks",
        "materialized-view", ["authorization-scope", "snapshot-consistency"], "hybrid", [14, 6, "discovery"],
    ),
    (
        "DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md",
        "Bound global API admission with token leases",
        "Which admission policy remains within its global budget during contention or coordinator failure?",
        "Token-bucket invariants; atomic refill and debit; leased budgets; expiry; failure policy",
        "Aggregate-admission histories, refill boundaries, coordinator outages, lease races and rejected-request latency",
        "Periodic usage reporting alone is not coordinated allocation; token buckets and exact rolling windows have different contracts",
        "resource-admission", ["correctness-contract", "bounded-work"], "hybrid", [3, 29, 4],
    ),
    (
        "DSA/Algo_Design_Problems/Routing/closure_aware_shortest_paths.md",
        "Reject stale shortcuts without losing a shortest-path guarantee",
        "When must a route query fall back to the live base graph after a closure?",
        "Shortcut expansion; closure snapshots; CH compatibility; directed nonnegative Dijkstra fallback",
        "Nested-shortcut and adversarial-closure queries compared with base Dijkstra, including unreachable pairs",
        "Finding a legal route after filtering does not prove completeness or optimality; richer time/turn state needs a separate contract",
        "correctness-contract", ["snapshot-consistency"], "hybrid", [3, "mobility", "delivery", "transport"],
    ),
    (
        "DSA/Algo_Design_Problems/Search/phrase_search_postings.md",
        "Bound phrase-search memory while preserving exact matches",
        "How can common-term phrase queries be evaluated without materializing huge posting lists?",
        "Rarest-term anchor; lazy postings; skips; relative positions; repeated terms; matching analyzers",
        "Brute-force token-position oracle, peak memory and tail latency over frequent and repeated-term phrases",
        "Rarity affects work, not semantic order; deleted documents and analyzer versions must share a consistent snapshot",
        "bounded-work", ["correctness-contract", "snapshot-consistency"], "hybrid", [14, 3, "discovery"],
    ),
    (
        "DSA/index_basics/Greedy_Scheduling/activity_selection.md",
        "Choose scheduling algorithms from objective and constraints",
        "When is earliest-finish greedy optimal, and when must scheduling become a weighted optimization problem?",
        "Half-open intervals; deterministic ties; exchange proof; weighted interval DP boundary",
        "Exhaustive small-instance optimum comparisons and counterexamples for weighted or buffered variants",
        "The reference proves equal-value single-room cardinality only; warehouse jobs and staffing need their own constraints",
        "correctness-contract", [], "analytical", [3, 13],
    ),
    (
        "DSA/index_basics/INTERVIEW_CASE_STUDIES.md",
        "Select data structures from operation mix and failure semantics",
        "Which data structure fits the update/query workload while retaining the required correctness and resource bounds?",
        "Cache policies; ranks/range queries; sketches; prefix/nearest-neighbor search; graphs; scheduling; online windows",
        "Operation traces, exact small-data oracles, measured memory/latency and stated approximation or distributed guarantees",
        "Eight teaching units plus rapid prompts; approximate structures, local proofs and service guarantees must be distinguished",
        "correctness-contract", ["bounded-work", "evaluation-mismatch"], "hybrid", [3, 54, "ml_design"],
    ),
    (
        "DSA/index_basics/Range_Queries/fenwick_inventory_totals.md",
        "Maintain exact inventory range totals under point updates",
        "Which update and snapshot contract makes range totals correct and efficient?",
        "Fenwick tree lowbit traversal; inclusive ranges; point deltas; wide arithmetic; atomic visibility",
        "Randomized update/query comparisons with a simple array, boundary/overflow checks and update-visibility tests",
        "Fixed ordering and one logical writer are reference assumptions; a Fenwick tree alone does not provide distributed stock correctness",
        "snapshot-consistency", ["correctness-contract"], "analytical", [3, "inventory", "quick_inventory"],
    ),
    (
        "DSA/Production_DSA_Incidents/Algorithmic_DoS/hash_and_regex_resource_exhaustion.md",
        "Bound request amplification before expensive parsing and validation",
        "Which per-request limits and algorithm changes stop CPU exhaustion without rejecting legitimate traffic unnecessarily?",
        "Hash collision amplification; catastrophic regex backtracking; pre-parse limits; bounded queues and downstream budgets",
        "Adversarial near-match inputs, collision workload, CPU per request, timeout cascades and legitimate-input acceptance",
        "The hash and regex scenarios are separate; legacy runtime behavior and example input limits require fresh validation",
        "bounded-work", ["resource-admission", "incident-postmortem"], "incident", [55, 47, 3],
    ),
    (
        "DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md",
        "Return useful graph candidates under explicit traversal budgets",
        "How should a recommendation API respond when graph expansion reaches a node, edge, memory or time limit?",
        "Enqueue-time deduplication; paged adjacency; independent work budgets; deterministic traversal; explicit partial results",
        "High-degree and cyclic graph replays, peak memory, examined edges, output completeness and candidate utility",
        "Depth and output-size limits alone do not bound expansion; source product degrees and outage figures are illustrative claims",
        "bounded-work", ["correctness-contract", "incident-postmortem"], "incident", [15, 14, 55, "discovery"],
    ),
    (
        "DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md",
        "Diagnose production failures from violated algorithm invariants",
        "Which invariant failed, how can it be contained, and which replay would prevent recurrence?",
        "Hash flooding; BFS OOM; regex backtracking; recursive DFS depth; ORM N+1; overflow; unstable ties; floating search termination",
        "Reproduce each mechanism separately with bounded adversarial traces; track latency, memory, database calls and deterministic outputs",
        "Ten reconstructed incident units include recurring mechanisms; reported timings are not measured outcomes of this library",
        "incident-postmortem", ["bounded-work", "correctness-contract"], "incident", [55, 3, "reliability"],
    ),
    (
        "DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md",
        "Load authorized hierarchies with bounded database work",
        "Should a subtree API use a recursive CTE or bounded level-order batching for its workload?",
        "N+1 detection; adjacency-list indexing; per-step authorization; cycle checks; stable assembly; snapshot reads",
        "Reference traversal comparisons, round trips, rows scanned, unauthorized descendants, deep/wide trees and truncation behavior",
        "One SQL statement can still do excessive work; an outer LIMIT is not a universal recursion work bound",
        "bounded-work", ["authorization-scope", "snapshot-consistency"], "incident", [6, 3, "insurance_ops"],
    ),
    (
        "DSA/Sys_Des_DSA/Autocomplete/frequency_admission_and_freshness.md",
        "Balance autocomplete freshness, memory and abuse controls",
        "Which terms enter a bounded suggestion index, and how can safe new trends appear promptly?",
        "Trie admission; time-window sketches; bounded candidate stores; immutable snapshots; real-time overlays; CDN versioning",
        "Memory including two generations, suggestion relevance, freshness lag, abusive-query tests and per-locale tail latency",
        "Sketches cannot enumerate terms; approximate frequencies and novelty thresholds must be evaluated rather than copied",
        "materialized-view", ["bounded-work", "rollout-regression"], "hybrid", [14, "discovery"],
    ),
    (
        "DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md",
        "Rebalance caches without overwhelming the authoritative store",
        "How should cache fills and ring migration preserve freshness while limiting database amplification?",
        "Single-flight scope; jitter; fill admission; consistent hashing; old/new ring epochs; version fences and tombstones",
        "Cold-fleet and skewed-key load replays, database concurrency, invalidation lag and stale-fill/delete races",
        "Local single-flight is not fleet-wide coordination; expected key movement is not the same as request miss share",
        "materialized-view", ["resource-admission", "snapshot-consistency"], "hybrid", [4, 27, "commerce_platform"],
    ),
    (
        "DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md",
        "Design stateful serving structures with explicit consistency",
        "Which cache, typeahead, leaderboard and deduplication contracts remain valid during refresh and failure?",
        "Profile caching; safe typeahead publication; ranked score updates; Bloom-filter crawler admission",
        "Failure replays and reference histories; freshness, exact versus approximate results and false-positive consequences",
        "Four teaching units overlap focused examples; Bloom false positives and stale ranks need product-specific tolerance",
        "correctness-contract", ["materialized-view", "snapshot-consistency"], "hybrid", [4, 54],
    ),
    (
        "DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md",
        "Separate live rank approximation from exact competition settlement",
        "How should score events support fast display ranks and a reproducible final prize result?",
        "Score buckets; Fenwick count summaries; deterministic ties; deduplicated events; cross-shard moves; closing watermarks",
        "Reference sorted snapshots, duplicate/late events, negative scores, bucket crossings and exact final-rank replay",
        "A sum of stale counters is not an exact settlement snapshot; tie semantics and late-event policy must be specified",
        "snapshot-consistency", ["idempotent-side-effects", "correctness-contract"], "hybrid", [4, 3, 9],
    ),
    (
        "Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md",
        "Evaluate incremental investigator value from graph fraud signals",
        "Which graph-enriched alerts add confirmed value at a fixed investigation capacity beyond claim-level scoring?",
        "Conservative entity resolution; typed claim-actor graphs; motifs and supernodes; point-in-time features; reversible merges",
        "Temporal baseline comparison, precision at capacity, mature outcomes, sampled low-risk audits and appeal/false-positive slices",
        "Connectivity and shared providers are not proof of fraud; source reference notes are unavailable and labels are delayed/selected",
        "delayed-selective-labels", ["human-in-the-loop", "data-leakage"], "predictive-ml", [15, "graph_ml", "fraud_ml", "insurance_ops", "payments"],
    ),
]

DOMAIN_KEYS = {
    "enterprise": ("14_saas_telecom_and_b2b", "03_enterprise_knowledge_support_and_sales"),
    "omnichannel": ("04_retail_omnichannel_offline_online_stores", "02_omnichannel_customer_loyalty_and_fulfilment"),
    "control_tower": ("07_supply_chain_logistics_warehousing_and_fulfillment", "03_control_towers_events_and_autonomous_operations"),
    "insurance_ops": ("10_insurance_actuarial_and_claims", "02_claims_fraud_customer_and_operations"),
    "insurance_risk": ("10_insurance_actuarial_and_claims", "03_insurance_mlops_model_risk_and_climate"),
    "reliability": ("03_practical_ml_design_and_interviews", "03_evaluation_reliability_and_incidents"),
    "mobility": ("13_mobility_travel_and_hospitality", "01_mobility_dispatch_eta_and_pricing"),
    "delivery": ("06_quick_commerce_food_delivery_and_on_demand", "02_dispatch_eta_and_delivery_operations"),
    "transport": ("07_supply_chain_logistics_warehousing_and_fulfillment", "02_transport_network_warehouse_and_last_mile"),
    "discovery": ("05_ecommerce_marketplaces_and_digital_commerce", "01_catalogue_search_discovery_and_conversion"),
    "ml_design": ("03_practical_ml_design_and_interviews", "01_ml_system_design_production"),
    "inventory": ("07_supply_chain_logistics_warehousing_and_fulfillment", "01_planning_forecasting_inventory_and_sourcing"),
    "quick_inventory": ("06_quick_commerce_food_delivery_and_on_demand", "01_quick_commerce_inventory_and_assortment"),
    "commerce_platform": ("05_ecommerce_marketplaces_and_digital_commerce", "03_digital_commerce_data_platform_and_operations"),
    "graph_ml": ("01_production_ml_case_library", "02_graph_ml"),
    "fraud_ml": ("01_production_ml_case_library", "01_fraud_modelling"),
    "payments": ("11_financial_services_fraud_and_risk", "01_payments_aml_and_financial_crime"),
}

PATTERNS = {
    "distribution-shift": ("The operating or retraining population changes", "Slice regressions and corpus-mix changes", "Do not infer shift from sampling noise alone"),
    "data-leakage": ("Evaluation sees information unavailable at decision time", "Related repositories, calls, documents or future labels cross splits", "A legal pre-cutoff feature is not leakage"),
    "evaluation-mismatch": ("A proxy metric does not establish decision quality", "Valid syntax, fluent text or a judge score masks task failure", "Use when a concrete decision-level comparator exists"),
    "eligibility-before-ranking": ("Apply hard constraints before optimizing a score", "A high-score candidate violates policy, date or capability constraints", "Ranking cannot repair wrong eligibility metadata"),
    "cost-latency-tradeoff": ("Choose resource policy under quality and latency bounds", "Cheaper routes or bigger batches change quality or waiting time", "Compare full-path cost at a stated workload"),
    "authorization-scope": ("Bind data and cached state to verified access rights", "A tenant, repository, adapter or prefix is reused across scopes", "Content similarity does not grant access"),
    "materialized-view": ("Treat an index or cache as rebuildable derived state", "Deletes, source versions or invalidation lag diverge from truth", "An authoritative transaction needs a separate correctness contract"),
    "atomic-release-bundle": ("Promote compatible model and system versions together", "Prompt, tool, adapter or index versions are independently rolled back", "Stateless independent components need not share a release"),
    "rollout-regression": ("A new release harms a supported population or invariant", "Canary or per-domain metrics diverge from baseline", "A passing aggregate metric is insufficient evidence of safety"),
    "human-in-the-loop": ("Qualified review controls uncertain consequential decisions", "Approval, evidence adjudication or uncertainty handoff is required", "Human review must have capacity, authority and a defined timeout"),
    "idempotent-side-effects": ("Retries must not repeat an external action", "Crash windows or duplicate events can replay purchases or updates", "An idempotency key needs durable end-to-end enforcement"),
    "snapshot-consistency": ("Read related state from one declared version", "Graph, ranking, inventory or policy parts can come from different times", "Explicit best-effort views must disclose weaker guarantees"),
    "resource-admission": ("Reserve bounded work before accepting more load", "Queue, token, KV or database capacity can be exhausted", "Utilization alone does not establish a safe admission limit"),
    "correctness-contract": ("Choose algorithms from explicit semantics and assumptions", "A fast heuristic violates the claimed exactness or objective", "A complexity claim cannot replace a correctness proof"),
    "bounded-work": ("Bound processing independently of output size", "Fanout, collisions, recursion or backtracking amplifies work", "A depth or response-size cap may leave intermediate work unbounded"),
    "incident-postmortem": ("Trace a failure mechanism to a preventive replay", "A violated invariant propagates across a service boundary", "These references are exercises, not verified incident records"),
    "delayed-selective-labels": ("Outcomes arrive late and depend on prior selection", "Only investigated claims acquire confirmed labels", "Evaluate mature cohorts and audit beyond the selected queue"),
}


def rel_link(origin: Path, target: Path, title: str, anchor: str = "") -> str:
    path = Path(os.path.relpath(target, origin.parent)).as_posix()
    return f"[{title}]({quote(path, safe='/._-')}{('#' + anchor) if anchor else ''})"


def cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def slug(value: str) -> str:
    value = re.sub(r"[^\w\- ]", "", value.lower())
    return value.replace(" ", "-")


def heading_map(text: str) -> dict[str, str]:
    """GitHub-style heading IDs, including duplicate-heading suffixes."""
    result, seen = {}, defaultdict(int)
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        if in_fence or not re.match(r"^#{1,6} ", line):
            continue
        title = re.sub(r"^#+ ", "", line)
        base = slug(title)
        n = seen[base]
        seen[base] += 1
        result[f"{base}{'-' + str(n) if n else ''}"] = title
    return result


def units(text: str) -> list[str]:
    result = [line[3:] for line in text.splitlines() if re.match(r"^## (?:\d+\. |Case [A-Z0-9])", line)]
    # Numbered sections in the claims-graph case describe one narrative.
    if result and any("Context, objective" in title for title in result):
        result = []
    return result or [next(line[2:] for line in text.splitlines() if line.startswith("# "))]


def resolve_destination(key: int | str) -> Path:
    if isinstance(key, int):
        matches = [p for p in LIB.glob(f"[0-9][0-9]_*/{key:02d}_*")
                   if not p.parent.name.startswith("11_")]
        if len(matches) != 1:
            raise ValueError(f"Module {key}: expected one directory, found {matches}")
        return matches[0]
    family, leaf = DOMAIN_KEYS[key]
    return LIB / "11_data_science_domain_specialisations" / family / leaf


def load_records() -> list[dict]:
    records = []
    for i, row in enumerate(ROWS, 1):
        source, title, decision, topics, evaluation, boundary, primary, secondary, kind, destinations = row
        p = SOURCES / source
        text = p.read_text(encoding="utf-8-sig")
        records.append(dict(
            id=f"PLAN-2026-IR{i:03}", ref=f"REF-IR{i:03}", source=p,
            title=title, decision=decision, topics=topics, evaluation=evaluation,
            boundary=boundary, primary=primary, secondary=secondary, kind=kind,
            destinations=[resolve_destination(key) for key in destinations],
            units=units(text), text=text,
        ))
    actual = set(SOURCES.rglob("*.md"))
    mapped = {r["source"] for r in records}
    if actual != mapped:
        raise ValueError(f"Reference inventory changed: unmapped={actual-mapped}, missing={mapped-actual}")
    assert len(mapped) == len(records), "A source was mapped twice"
    for r in records:
        assert r["primary"] in PATTERNS and len(r["secondary"]) <= 2
        assert all(p in PATTERNS for p in r["secondary"])
        for d in r["destinations"]:
            assert all((d / suffix).is_file() for suffix in ("INDEX.md", "Case_Studies/INDEX.md", "Cheat_Sheets/INDEX.md")), d
    return records


def local_links(text: str) -> list[str]:
    return re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text)


def unresolved_source_links(records: list[dict]) -> list[tuple[dict, str]]:
    bad = []
    for r in records:
        for raw in local_links(r["text"]):
            url = urlsplit(raw.strip("<>"))
            if url.scheme or not url.path:
                continue
            if not (r["source"].parent / unquote(url.path)).resolve().exists():
                bad.append((r, raw))
    return bad


def plan_document(records: list[dict]) -> str:
    file_count = len(records)
    unit_count = sum(len(r["units"]) for r in records)
    bad = unresolved_source_links(records)
    lines = [
        "# Reference-informed case and cheat-sheet plan", "",
        "The interview materials are examples used only as references for curriculum planning. They are not copied, adopted as production evidence, or counted as completed cases. Every brief below remains **not-measured / proposed**; no results have been imported.", "",
        "## 1. Collection metadata", "",
        "| Field | Value |", "|---|---|",
        "| Editorial owner role | Curriculum maintainer; individual authors not yet assigned |",
        f"| Reference inventory reviewed | {DATE}; planning scope and provenance only |",
        "| Review cadence | At authoring, source changes, and before publication |",
        "| Index version | 1.0 |",
        "| Planning ID format | PLAN-2026-IRNNN; replace or crosswalk to canonical case IDs when authoring |",
        "| Reference ID format | REF-IRNNN; identifies an example file, not an authored case |",
        "| Source of truth | This planning catalog; original examples remain in Train/INTERVIEW_CASES |",
        f"| Reference inventory | {file_count} files; {unit_count} reading units, including overlapping anthology material |",
        "| Completed cases created by this plan | 0 |", "",
        rel_link(PLAN, ROOT / "Train/Templates/Case_Study_Template/INDEX_Case_Studies_Template_v2.md", "Case index template")
        + " and " + rel_link(PLAN, ROOT / "Train/Templates/Case_Study_Template/Case_Study_Outline_Template_v2_Integrated.md", "integrated 15-section outline")
        + " govern future authoring. Reference notes have not been certified against the outline.", "",
        "The first destination is the proposed editorial home; others are supporting curricula. A repeated planning ID in a local index is a cross-reference to this one brief. No link to an unwritten case file is fabricated. Multi-case anthologies must be split into independently framed narratives before publication.", "",
        "## 2. Proposed case catalog", "",
        "| Planning ID | Proposed title and reference | Editorial home | Case type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Key result |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        home = r["destinations"][0]
        lines.append("| " + " | ".join([
            r["id"], rel_link(PLAN, r["source"], r["title"] + " (example reference only)"),
            rel_link(PLAN, home / "INDEX.md", home.name), r["kind"], cell(r["decision"]),
            f'`{r["primary"]}`', ", ".join(f"`{p}`" for p in r["secondary"]) or "None",
            "not-measured", "proposed", DATE, "",
        ]) + " |")
    lines += ["", "### Detailed scope and reference inventory", "",
              "Reading-unit totals are inventory aids, not counts of unique cases or independent evidence. Reference figures, incident names, company assumptions and performance targets must not be reused as facts.", ""]
    for r in records:
        lines += [f'#### {r["ref"]}: {r["title"]}', "",
                  f'**Proposed brief:** `{r["id"]}`. **Mechanisms to develop:** {r["topics"]}.', "",
                  f'**Evaluation to design:** {r["evaluation"]}.', "",
                  f'**Transfer boundary:** {r["boundary"]}.', "",
                  "**Destination indexes:** " + "; ".join(rel_link(PLAN, d / "INDEX.md", d.name) for d in r["destinations"]) + ".", "",
                  "**Example reference:** " + rel_link(PLAN, r["source"], r["source"].relative_to(SOURCES).as_posix()) + ". Reading units:", ""]
        for unit in r["units"]:
            lines.append("- " + rel_link(PLAN, r["source"], unit, slug(unit)))
        lines.append("")
    lines += ["## 3. Canonical reusable patterns for these proposed briefs", "",
              "These 17 mechanism tags apply to this planning subset. Existing collections retain their own vocabularies until an explicit consolidation review. No author should silently introduce synonyms.", "",
              "| Tag | Definition | Diagnostic signature | Boundary |", "|---|---|---|---|"]
    for tag, (definition, signature, boundary) in PATTERNS.items():
        lines.append(f"| `{tag}` | {definition} | {signature} | {boundary} |")
    lines += ["", "Pattern preconditions, contrasts, supporting brief IDs and linked curriculum scope are recorded below. Deprecated synonyms: none introduced in this subset. Planned briefs provide comparison questions, not empirical confirmation of a pattern.", "",
              "## 4. Cross-brief similarity and contrast map", "",
              "| Pattern | Supporting proposed briefs | Contrasting reference scope | Curriculum/theory index | Preconditions and boundary to investigate |", "|---|---|---|---|---|"]
    contrasts = {
        "distribution-shift": (17, "Speech-domain regression offers a different population shift from synthetic corpus feedback"),
        "data-leakage": (8, "Safety evaluation also separates benchmark contamination from judge error"),
        "evaluation-mismatch": (27, "A deterministic range-sum oracle contrasts with human-judged language quality"),
        "eligibility-before-ranking": (2, "Provider eligibility concerns capabilities/residency; regulation eligibility concerns date/jurisdiction"),
        "cost-latency-tradeoff": (7, "GPU scheduling trades queueing and memory, unlike model-price routing"),
        "authorization-scope": (12, "Index filtering and tenant adapter selection enforce access at different layers"),
        "materialized-view": (33, "Cache invalidation has different stale-read risks from search index refresh"),
        "atomic-release-bundle": (6, "Workflow checkpoint compatibility also needs external-action reconciliation"),
        "rollout-regression": (17, "ASR uses per-domain human gold; general release gates need task-specific suites"),
        "human-in-the-loop": (16, "A verified text quote can still reflect an acoustic or speaker-attribution error"),
        "idempotent-side-effects": (35, "Replay of score deltas differs from a purchase API's action guarantees"),
        "snapshot-consistency": (23, "Graph closure snapshots and exact final ranks use different state and finalization contracts"),
        "resource-admission": (22, "Global token allocation is different from local GPU memory admission"),
        "correctness-contract": (25, "Earliest-finish greedy proves an optimum only for the equal-value interval objective"),
        "bounded-work": (29, "BFS edge fanout needs different bounds from regex backtracking or SQL recursion"),
        "incident-postmortem": (30, "The anthology contains separate scenarios; do not merge durations into one incident"),
        "delayed-selective-labels": (11, "Document-field gold and mature fraud adjudication have distinct label processes"),
    }
    for tag in PATTERNS:
        support = [r for r in records if tag in [r["primary"], *r["secondary"]]]
        n, contrast = contrasts[tag]
        other = records[n - 1]
        selected = ", ".join(rel_link(PLAN, PLAN, r["id"], slug(r["ref"] + ": " + r["title"])) for r in support[:3])
        lines.append("| " + " | ".join([
            f"`{tag}`", selected,
            rel_link(PLAN, other["source"], other["ref"]) + ": " + contrast,
            rel_link(PLAN, support[0]["destinations"][0] / "INDEX.md", "Scope index; detailed theory still planned"),
            PATTERNS[tag][2],
        ]) + " |")
    lines += ["", "## 5. Questions and future-case backlog", "",
              "| Open question | Origin | Needed evidence | Proposed next step | Priority | Owner role | Review trigger |", "|---|---|---|---|---|---|---|"]
    backlog = [
        ("Which source claims can be independently supported?", "All 36 example files", "Original OCR/images or authoritative replacement sources", "Recover provenance before citing source-reported results", "High", "Research editor", "Before drafting any source-dependent claim"),
        ("How should overlapping anthology units become independent narratives?", "DSA anthologies and paired LLM examples", "Distinct decision, input population, assumptions and evaluation per unit", "Split or merge briefs by decision, recording the crosswalk", "High", "Curriculum editor", "Before assigning canonical case IDs"),
        ("Which domain-specific constraints change the design?", "Routing, inventory, support and insurance references", "Actual business workflow, data contracts and decision owner", "Write domain briefs without transplanting the example's scale or SLO", "High", "Domain author", "At case scoping"),
        ("Which company case claims are documented?", "Retail, quick commerce, ecommerce and streaming request", "Named-company primary sources with dates and stated limits", "Research Walmart, Tesco, Target, Zepto, Swiggy, Flipkart, Disney, Warner and Hotstar separately", "High", "Domain researcher", "Before company-specific writing"),
        ("What remains uncovered in edge robotics and low-level systems?", "Original directory audit and validation report", "Per-source-topic destination crosswalk and new specialist references", "Expand embedded, CUDA, robotics, compiler and streaming topic scope independently", "High", "Technical editor", "Next completeness audit"),
        ("How should planned cheats differ from case narratives?", "All reference-informed scope blocks", "Minimal algorithms/formulas, preconditions, counterexamples and worked checks", "Draft compact reusable references only after their assumptions are explicit", "Medium", "Reference author", "Before cheat-sheet publication"),
    ]
    for row in backlog:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 6. Review and retirement log", "",
              "| Record | Date | What changed | Evidence / trigger | Action | Owner role |", "|---|---|---|---|---|---|"]
    lines.append(f"| All planning briefs | {DATE} | Created reference-only scope; no cases drafted or source outcomes adopted | User instructed that interview materials are examples for reference only | indexed for planning | Curriculum maintainer |")
    lines += ["", "When a brief is split, merged, superseded or retired, preserve its planning ID and record the replacement. A source edit or recovered evidence requires a fresh review; inventory dates are not certification of technical accuracy.", "",
              "## 7. Pre-publication checks", "",
              "- [x] Source inventory mapped without modifying the examples.",
              "- [x] Proposed briefs distinguish reference existence from completed-case status.",
              "- [x] Proposed cases use not-measured evidence, proposed stage and empty result fields.",
              "- [x] Decision questions, mechanisms, evaluation plans and transfer limits are explicit.",
              "- [ ] Independently verify claims, version-sensitive mechanisms and recovered primary sources.",
              "- [ ] Split multi-case references and write each case with metadata and all 15 outline sections, using permitted N/A where appropriate.",
              "- [ ] Specify business owners, actual data availability, baseline, assumptions, operational review triggers and ethical/legal boundaries per case.",
              "- [ ] Assign canonical case IDs, create real case files and update links only when those files exist.",
              "- [ ] Draft and review the separate cheat sheets; their index descriptions are currently plans.",
              "- [ ] Reconcile every visible original-audit topic; this reference pass does not establish full curriculum coverage.", "",
              "### Source availability limitations", "",
              f"At this review, **{len(bad)} relative link occurrences** in the original examples point to unavailable files or directories. The examples themselves exist. The originals were not repaired, rewritten or treated as verified source evidence. See the "
              + rel_link(PLAN, REPORT, "reference-plan validation report") + " for the exact inventory.", "",
              "`Train/INTERVIEW_CASES/DS_Domain` contains no files in this snapshot. No robotics, CUDA-kernel or named-company evidence has been inferred from that empty directory. Road-network routing examples do not establish robot-navigation, SLAM, control or safety coverage.", ""]
    return "\n".join(lines)


def block_for(origin: Path, records: list[dict], kind: str) -> str:
    intro = [START, "", "## Reference-informed additions to develop", "",
             "These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. "
             + rel_link(origin, PLAN, "Planning catalog, source limits and authoring gates") + ".", ""]
    if kind == "subject":
        intro += ["| Planning ID | Detailed topics to develop | Decision to address | Validation to design |", "|---|---|---|---|"]
        for r in records:
            intro.append("| " + " | ".join([r["id"], r["topics"], r["decision"], r["evaluation"]]) + " |")
        intro += ["", "See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics."]
    elif kind == "cases":
        intro += ["| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        home = origin.parent.parent
        for r in records:
            relationship = "Proposed home" if home == r["destinations"][0] else "Supporting curriculum; shared brief"
            intro.append("| " + " | ".join([
                r["id"], r["title"], relationship, r["kind"], r["decision"], f'`{r["primary"]}`',
                ", ".join(f"`{p}`" for p in r["secondary"]) or "None", "not-measured", "proposed", DATE, "",
            ]) + " |")
        intro += ["", "### Reference and transfer notes", "", "| Planning ID | Example reference only | Scope boundary |", "|---|---|---|"]
        for r in records:
            intro.append(f'| {r["id"]} | {rel_link(origin, r["source"], r["ref"])} | {r["boundary"]} |')
        intro += ["", "Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned."]
    else:
        intro += ["| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |", "|---|---|---|---|---|"]
        for r in records:
            intro.append(f'| {r["id"]} | {r["topics"]} | {r["evaluation"]}. Boundary: {r["boundary"]}. | {rel_link(origin, r["source"], r["ref"])} | planned; not written |')
        intro += ["", "Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet."]
    return "\n".join(intro + ["", END])


def replace_block(text: str, block: str) -> str:
    if START in text or END in text:
        assert text.count(START) == text.count(END) == 1, "Unbalanced generated markers"
        return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    return text.rstrip() + "\n\n" + block + "\n"


def generated_outputs(records: list[dict]) -> dict[Path, str]:
    outputs = {PLAN: plan_document(records)}
    by_destination = defaultdict(list)
    for r in records:
        for d in r["destinations"]:
            by_destination[d].append(r)
    for directory, subset in by_destination.items():
        for suffix, kind in [("INDEX.md", "subject"), ("Case_Studies/INDEX.md", "cases"), ("Cheat_Sheets/INDEX.md", "cheats")]:
            target = directory / suffix
            outputs[target] = replace_block(target.read_text(encoding="utf-8-sig"), block_for(target, subset, kind))
    for target in [LIB / "INDEX.md", LIB / "Case_Studies/INDEX.md"]:
        block = "\n".join([START, "", "## Example references for future authoring", "",
                           rel_link(target, PLAN, "Reference-informed case and cheat-sheet plan")
                           + ": 36 example files mapped to proposed technical and domain briefs. Examples are references only; no completed cases or results are added. The plan records detailed scope, evaluation needs, transfer limits and source-availability gaps.", "", END])
        outputs[target] = replace_block(target.read_text(encoding="utf-8-sig"), block)
    return outputs


def validate_outputs(outputs: dict[Path, str]) -> list[str]:
    errors = []
    for p, text in outputs.items():
        # Existing content remains outside the scope of this insertion check.
        inserted = text if p == PLAN else text.split(START, 1)[1].split(END, 1)[0]
        if "\x00" in inserted or "\ufffd" in inserted:
            errors.append(f"Invalid inserted text: {p}")
        for raw in local_links(inserted):
            url = urlsplit(raw.strip("<>"))
            if url.scheme:
                continue
            target = (p.parent / unquote(url.path)).resolve() if url.path else p
            if target not in outputs and target != REPORT and not target.exists():
                errors.append(f"Broken new link: {p}: {raw}")
                continue
            if url.fragment and target.suffix.lower() == ".md":
                content = outputs.get(target)
                if content is None:
                    content = target.read_text(encoding="utf-8-sig")
                if unquote(url.fragment) not in heading_map(content):
                    errors.append(f"Broken new anchor: {p}: {raw}")
    return errors


def report_document(records: list[dict], outputs: dict[Path, str], hashes: dict[Path, str]) -> str:
    bad = unresolved_source_links(records)
    destinations = {d for r in records for d in r["destinations"]}
    lines = ["# Interview-reference planning validation", "", f"**Date:** {DATE}", "",
             "**Scope:** Use interview examples as references only to enrich future case briefs and cheat-sheet scope. This is a planning update, not a certification that the full curriculum is complete.", "",
             rel_link(REPORT, PLAN, "Open the reference-informed plan") + ".", "",
             "| Check | Result |", "|---|---|",
             f"| Example Markdown files inventoried and mapped | {len(records)} of {len(list(SOURCES.rglob('*.md')))} |",
             f"| Reading units, including overlapping anthology sections | {sum(len(r['units']) for r in records)} |",
             f"| Destination subject/domain folders | {len(destinations)} |",
             f"| Local indexes enriched | {len(destinations) * 3}; plus library and root case index |",
             f"| Proposed briefs | {len(records)}; some must split before authoring |",
             "| Actual case or cheat-sheet narratives created | 0 |",
             "| Imported source measurements or claimed outcomes | 0 |",
             "| Reference documents copied or changed | 0; SHA-256 inventory below |",
             "| New relative file and heading links | Validated against existing and generated targets |",
             "| Existing source links to unavailable targets | " + str(len(bad)) + " link occurrences; not silently repaired |",
             "| Full original-audit topic coverage | Still unresolved; this pass does not close the previous audit |", "",
             "Checks validate inventory, destination availability, unique planning IDs, controlled patterns, at most two secondary tags, generated block consistency, new relative links/heading anchors and source immutability. Dates refer to planning review, not verification of source claims. Existing unrelated index content is preserved.", "",
             "## Remaining completeness work", "",
             "- Reconcile all visible original-audit topic records to detailed destination scopes; collapsed source content remains unknown.",
             "- Expand CUDA, embedded compilers, robotics, streaming internals, alignment methods and other gaps recorded in the earlier validation.",
             "- Research named-company domain cases from primary sources before attributing business behavior or impact.",
             "- Complete the existing index-template gaps, missing DS-root collection pair, prerequisite cycle and encoding repairs reported previously.",
             "- Draft and review independent case narratives and cheat sheets; these planned briefs are not substitutes.", "",
             "The new speech, adapter, LangGraph, retrieval and DSA scope addresses parts of the earlier missing detail. It does not establish all-topic preservation. `INTERVIEW_CASES/DS_Domain` is empty. Road-network algorithms have not been relabeled as robot navigation or safety examples.", "",
             "## Unavailable supporting links in original examples", "",
             "This records each occurrence rather than claiming that all paths are unique. The source files are available; their referenced OCR notes, source images or source directories are missing at the stored paths. A repository-wide filename search also did not locate representative missing OCR/graph sources. Restore or replace provenance before treating a claim as sourced.", "",
             "| Reference | Available example | Unavailable stored target |", "|---|---|---|"]
    for r, raw in bad:
        lines.append(f'| {r["ref"]} | {rel_link(REPORT, r["source"], r["source"].name)} | `{cell(raw)}` |')
    lines += ["", "## Source integrity inventory", "", "| Reference | Source | SHA-256 |", "|---|---|---|"]
    for r in records:
        lines.append(f'| {r["ref"]} | {rel_link(REPORT, r["source"], r["source"].relative_to(SOURCES).as_posix())} | `{hashes[r["source"]]}` |')
    lines += ["", "## Reproduction", "",
              "Run `python Train/Templates/build_interview_reference_plan.py --check` from the workspace root. Use `--write` only to refresh this reference-only plan and its marked index sections after editing the curated briefs. The script refuses an unmapped source inventory and preserves content outside its own markers.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = load_records()
    hashes = {r["source"]: hashlib.sha256(r["source"].read_bytes()).hexdigest() for r in records}
    outputs = generated_outputs(records)
    errors = validate_outputs(outputs)
    if errors:
        raise SystemExit("\n".join(errors))
    outputs[REPORT] = report_document(records, outputs, hashes)
    changed = []
    for p, content in outputs.items():
        current = p.read_text(encoding="utf-8-sig") if p.exists() else None
        if current != content:
            changed.append(str(p.relative_to(ROOT)))
            if args.write:
                with p.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
    for p, expected in hashes.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, f"Source changed: {p}"
    if args.check and changed:
        raise SystemExit("Stale generated outputs:\n" + "\n".join(changed))
    print(json.dumps({"mode": "write" if args.write else "check", "source_files": len(records),
                      "reading_units": sum(len(r["units"]) for r in records),
                      "destination_folders": len({d for r in records for d in r["destinations"]}),
                      "changed_files": len(changed), "source_files_modified": 0,
                      "new_link_validation": "passed", "missing_source_link_occurrences": len(unresolved_source_links(records))}, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
