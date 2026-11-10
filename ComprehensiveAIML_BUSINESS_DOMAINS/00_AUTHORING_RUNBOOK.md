# Course authoring runbook

How the 11-part curriculum is completed, verified, and resumed. Everything here is
filesystem-measured: no step trusts an agent's claim of success.

## What "complete" means

A teaching unit is `<module>/<submodule>` and owns two collections:

- `Case_Studies/` — decision cases, 15 numbered sections each
- `Cheat_Sheets/` — reference cards, 12 numbered sections each, plus one final
  `NN_25_Common_Issues.md`

A unit is **complete** when, for every case and sheet file, `Tools/manuscript_lint.py` exits 0,
and a Common Issues artifact exists with at least 25 `CI-` entries. An index is complete when it
has its required section count (7 for cases, 11 for sheets) and every link resolves.

## Acceptance gate

```bash
python Tools/manuscript_lint.py <file.md>     # exit 0 = accepted
```

It enforces, in order of how often each fires:

| Check | Threshold | Why |
|---|---|---|
| Heading contract | exact numbered H2 set, in order | template conformance |
| Physical length | ≥ 950 lines | depth floor |
| Distinct prose | ≥ 680 non-blank lines | depth floor |
| Long-line repeat | any ≥70-char line at most twice | **the anti-padding rule** |
| Duplicate ratio | < 0.15 | catches mechanical inflation |
| Placeholders | none | no `TODO`/`TBD` |
| Case derivations | ≥3 `FORMULA`/`CONSTRAINT`/`NUMERICAL EXAMPLE` blocks | checked arithmetic |
| Common Issues | ≥25 `CI-` entries | real troubleshooting |

The repeat and duplicate-ratio checks are what reject the previous pass's 1,015-line files that
repeated one sentence hundreds of times. Length alone was never the acceptance criterion.

## Pipeline

```text
Tools/build_queue.py      scan disk -> queue.jsonl + per-task briefs (887 tasks)
Tools/plan_batches.py     one manuscript per batch, no file owned twice (887 batches)
Tools/dispatch_next.py    claim N batches, print self-contained prompts
Tools/run_batch.py        shared prompt text + --verify (re-lints everything)
Tools/gen_indexes.py      deterministic 7/11-section collection indexes
Tools/sync_unit_indexes.py deterministic unit topic-map status
Tools/verify_course.py    link integrity across all Markdown
Tools/coverage_report.py  per-part accepted/manuscript counts
```

`gen_indexes` and `sync_unit_indexes` require **no model**: they read frontmatter and H1 titles from
the real files, so an index can never link a file that does not exist.

## Resume after an interruption

State lives in `.authoring_state/` (protected; see contract section 10).

```bash
python Tools/build_queue.py      # recompute gaps from disk
python Tools/plan_batches.py     # rebuild conflict-free batches
python Tools/run_batch.py --verify
python Tools/dispatch_next.py --slots 12      # next 12 prompts
python Tools/coverage_report.py               # where we stand
```

Because the queue is derived from the filesystem, a resumed session re-does nothing that already
passes lint and re-queues everything that does not. Deleting a state file is therefore recoverable;
deleting manuscripts is not.

## Ownership rules that prevent corruption

- One batch owns one manuscript file. Batches never share a path (verified at planning time).
- Workers may not touch root indexes, `00_*` files, `Templates/`, `Tools/`, or any other subtree.
- Indexes are generated centrally, after manuscripts land, from what actually exists.
- Workers build a manuscript in memory and write it in one call, so an interrupted run cannot leave
  a truncated file.

## Known evidence discipline

- Constructed numbers are labelled `ILLUSTRATIVE` and must be re-derived, not quoted.
- No invented company architecture or measured outcome.
- A company name in a catalog is a research context, not a claim about its systems.
- Unavailable sources stay unavailable and are recorded as gaps beside the affected conclusion.

## Verification invariants

These are the checks the integration pass actually runs, and what each can and cannot prove.

| Check | Command | Proves |
|---|---|---|
| Per-file acceptance | `python Tools/manuscript_lint.py <file>` | headings, depth, distinctness, no placeholders, arithmetic blocks present |
| Arithmetic spot-audit | recompute in a kernel | printed numbers follow from stated assumptions |
| Link integrity | `python Tools/verify_course.py --links` | no link points at a missing file |
| Link repair | `python Tools/fix_links.py --check` | wrong-depth links are identified; ambiguous ones are never guessed |
| Index fidelity | `python Tools/gen_indexes.py` | an index lists only files that exist |
| Coverage | `python Tools/coverage_report.py --md` | how many manuscripts actually pass |

Lint passing is necessary, not sufficient. A file can pass and still contain wrong arithmetic,
so the audit recomputes printed figures. Corrections found this way are listed below.

## Corrections applied by the integration owner

| File | Defect | Fix |
|---|---|---|
| `.../15_graph_ml.../Cheat_Sheets/07_bounded_traversal_and_entity_resolution.md` | `2^20.33923` printed as `1,326,322` (inconsistent with its own posterior) | corrected to `1,326,534`; `0.002 x LR = 2,653.07` now matches |
| `.../21_finetuning.../Case_Studies/01_support_domain_adaptation.md` | links overshot with `../../` | repaired to `../` |
| `.../21_finetuning.../Case_Studies/02_Customize_Tenant_Assistants_...md` | same overshoot on three links | repaired |
| `.../01_production_ml_case_library/06_pricing/.../01_Dynamic_Pricing_Ecommerce_Retail.md` | linked an unauthored sibling | unlinked, named as planned |

Independently brute-forced and confirmed correct (no change needed):

| File | Checked |
|---|---|
| `.../13_optimization.../Case_Studies/03_scheduling_objective_and_constraints.md` | 158 compatible subsets of 4,096, closed-interval 77, 1 h buffer 77, max value 29 = (A,D,F,J,L), greedy 26, max cardinality 5 |
| `.../05_math.../Case_Studies/02_Derive_Gradient_Descent_for_Regression.md` | Sxx=10, Sxy=6, slope 0.6, intercept 2.2, residuals sum to 0, SSE 2.4, R^2 0.6, RMSE 0.6928 |
| `.../06_pricing/.../01_Dynamic_Pricing_Ecommerce_Retail.md` | k=0.915, m(18)=7.07, m(16.20)=5.423, -23.3% margin, +30.4% units, e_req=2.517 |
| `.../21_finetuning.../Cheat_Sheets/01_LoRA_Parameter_Efficient_Finetuning.md` | base 5,801,771,008, adapter 36,831,232 (0.6348%), 421.5 MiB at 12 B, 562.0 MiB at 16 B, merge 181.46 GFLOP |

## Known non-goals

- Unit `INDEX.md` topic maps are generated, never authored per topic; they assert coverage only.
- `AUTHORING_PROGRESS.md` rows are worker self-reports; the coverage report is the authority.
- Provider quota can terminate a whole wave at once (HTTP 429). The pipeline is resumable from
  disk; nothing is lost because every manuscript is written atomically.
