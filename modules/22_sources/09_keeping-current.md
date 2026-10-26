# Keeping Current

> Module 22 — Sources and Further Reading
> Topic 09 of 09
> Estimated study time: 2.5 hours
> Prerequisites: 01_how-to-read-this-list
> Version 2026.3

## 1. Why this topic exists

Every other topic in this module teaches how to judge a source at a moment. This topic teaches how to keep that judgement true. Sources decay: guidance is revised, standards are amended, code sets are released, datasets are versioned, papers are retracted, and the sentence that was correct at drafting becomes wrong at submission. The decay is silent, and the cost lands on whoever reuses the sentence next.

The failures are ordinary:

- A regulatory summary is updated every year, and a guidance is revised two months after the update.
- A slide deck cites a standard edition that was withdrawn, because the deck was built from a template.
- A reading list points to a paper that was retracted, and every new analyst inherits the citation.
- A model card describes the dataset version used in development, and the production pipeline moved to a new release.
- A team monitors the FDA but not the MDCG, and the EU launch plan is built on a superseded FAQ.
- A quarterly review checks links but not status, so a page that still resolves is counted as current.

The people affected are the regulatory lead who signs a stale summary, the analyst who reuses an old number, the new hire who trusts the reading list, and the organisation that discovers the problem during an audit rather than before it.

Maintenance is also the part of source work most often skipped, because it has no visible output. A review that finds nothing looks like a waste of an afternoon, and the value only appears in the failure that did not happen.

This topic turns maintenance into a routine with a watch list, a cadence, a triage rule, a quarterly checklist, and a retirement process. It ends the module because it is the practice that keeps the other eight topics from expiring.

> **Note:** Currency is a property of a source at a date, not a property of a document. A summary can be recent and wrong, and an old document can be current. The only reliable check is against the issuer, at the review date.

## 2. Learning outcomes

By the end of this topic you can:

1. **Build** a watch list with sources, owners, cadences, and triggers. Assessed by Exercise 1 and self-check item 1.
2. **Assign** a review cadence from the volatility of the source class rather than from habit. Assessed by Exercise 2.
3. **Apply** a triage rule that separates changes requiring action from changes to watch or ignore. Assessed by Exercise 3 and section 7.
4. **Run** a quarterly review that checks status, not only links. Assessed by Exercise 4 and the worked example.
5. **Propagate** a change to every document and model that depends on the source. Assessed by Exercise 5.
6. **Retire** outdated material without losing the record of what it said. Assessed by Exercise 6.
7. **Estimate** the effort of a maintenance routine and fit it to a realistic monthly budget. Assessed by Exercise 7 and section 7.
8. **Design** a twelve-month maintenance calendar for a team. Assessed by Exercise 8.

## 3. The core mental model

Maintenance is a loop with four stages: watch, triage, act, and retire. The loop runs on a cadence set by the source's volatility, and every action updates a register that the next cycle reads.

```text
                    WATCH LIST
        source | class | owner | cadence | trigger
                              |
                              v
                    DETECT CHANGE
        new release | revision | retraction | deprecation
                              |
                              v
                    TRIAGE
        act now | schedule | watch | ignore
                              |
                              v
                    ACT AND PROPAGATE
        update documents, models, registers
                              |
                              v
                    RETIRE AND ARCHIVE
        supersede, date-stamp, keep the record
                              |
                              v
                    QUARTERLY AUDIT
        check status at the issuer, not the link
```

Read the loop as a cycle, not a project. The output of the audit is the input to the next watch cycle, and the register is the memory that makes the cycle cheap.

**Cadence follows volatility.** A statute changes rarely; guidance changes yearly; a code set changes annually on a fixed date; a dataset can change without notice. A single annual review is either too often for the statute or too rare for the dataset.

**Detection is not the same as reading.** A feed or an alert tells you that something changed. It does not tell you whether the change affects you. The triage step is where most of the effort should go, because reading every update in full is unsustainable.

**Propagation is the hard part.** A change to a guidance can affect a regulatory summary, a model card, a validation plan, a sales deck, and a training module. If the register does not link sources to documents, the change is applied in one place and missed in four.

**Retirement is a source-management act.** An outdated document should be superseded, date-stamped, and archived rather than deleted, because the record of what was believed at a date is itself useful and because an audit may ask for it.

| Source class | Typical cadence | Trigger examples | Effort per cycle |
|---|---|---|---|
| Regulation | Annual, or on legislative event | Amendment, transition deadline | 2 hours |
| Guidance | Quarterly | New draft, revision, withdrawal | 1 hour per source |
| Standard | Semiannual | New edition, amendment | 1 hour |
| Code set | On release date | Annual or periodic release | 2 hours |
| Dataset | On use, and on release | New version, licence change | 1 hour |
| Literature | Monthly alert review | Retraction, contradicting trial | 2 hours |
| Market | Quarterly | New report, revised figures | 1 hour |

## 4. Definitions and vocabulary

### 4.1 Maintenance terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Watch list | The set of sources monitored with owners and cadences. | A folder of bookmarks | Register |
| Cadence | The interval between reviews. | Uniform across all sources | Months |
| Trigger | An event that forces an earlier review. | Replaced by a calendar date | Event |
| Triage | Sorting changes by the action they require. | Reading everything in full | Decision |
| Propagation | Updating every dependent document. | Updating the summary only | Action |
| Retirement | Superseding a source and recording why. | Deleting the old file | Status |
| Archive | A dated copy kept for the record. | A live URL | Snapshot |
| Source register | The record of sources, versions, and uses. | A bibliography | Database |
| Drift | Divergence between a summary and its source. | Treated as a writing issue | Status |
| Change log | The issuer's record of revisions. | Assumed to exist | Document |

### 4.2 Monitoring terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Alert | A notification of a new item. | Treated as an assessment | Notification |
| Feed | A machine-readable list of updates. | Assumed comprehensive | RSS or API |
| Changelog | A dated list of changes. | Ignored in favour of the latest page | Document |
| Release note | A description of a specific release. | Confused with a changelog | Document |
| Deprecation | A notice that a source will be retired. | Treated as immediate removal | Status |
| Sunset date | The date a source stops being supported. | Confused with a retraction | Date |
| Superseded | Replaced by a newer version. | Treated as withdrawn | Status |
| Withdrawn | Removed by the issuer. | Treated as superseded | Status |
| Retracted | Withdrawn for validity reasons. | Confused with a correction | Status |
| Expression of concern | A notice raising a question. | Ignored until retraction | Status |

### 4.3 Archive and record terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Snapshot | A dated copy of a page or document. | A screenshot without a date | Archive copy |
| Permalink | A stable link to a specific version. | A live homepage URL | URL |
| DOI | A persistent identifier. | Assumed immutable content | Identifier |
| Version pin | Recording the exact version used. | Recording "latest" | Version string |
| Retention period | How long a record must be kept. | Set by convenience | Years |
| Audit trail | The record of who changed what and when. | Absent from shared drives | Log |
| Tombstone | A record that a source existed and was retired. | Deleted file | Record |
| Re-verification | Checking the source again at the issuer. | Checking the link only | Action |

### 4.4 Review and governance terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Quarterly review | A scheduled check of the watch list. | A link check | Process |
| Owner | The person accountable for a source. | A shared inbox | Role |
| Escalation | Raising a material change to a decision-maker. | Applied to every change | Process |
| Materiality | Whether a change affects a decision or document. | Assumed for every change | Judgement |
| Register review | A periodic audit of the register itself. | Assumed unnecessary | Process |
| Coverage | The proportion of sources with an owner and cadence. | Counted as the number of sources | Proportion |

### 4.5 Signals that maintenance is failing

| Signal | What it suggests | What to check |
|---|---|---|
| Same sources reviewed every quarter | The list is stale | Coverage and ownership |
| No source ever changes status | The check is a link check | Status at the issuer |
| Documents cite different versions | Propagation is missing | Source-to-document links |
| New hires use old templates | Retirement is missing | Template review |
| Review takes days | Cadence is too coarse | Triage and automation |
| Review takes minutes | Coverage is too thin | Sample the register |

## 5. The mechanics, step by step

The procedure has 20 steps in five phases. It builds the loop in section 3 and leaves a routine that fits a monthly budget.

The routine is designed for a team of one to ten people. A larger organisation can distribute the steps by role, and a smaller one can run them in a single monthly hour.

### Phase A — Build the watch list

**Step 1. List the sources the work depends on.**
Input: active documents, models, submissions, and training material.
Output: every external source cited, with the document that cites it.
Rationale: the watch list should be derived from use, not from interest.
Branch: if a source is cited in no document, it is optional and can be dropped.
Edge case: a source can be cited indirectly through a template, which is still a use.
Check: is every citation mapped to a document.
Record: the source-to-document map.

**Step 2. Classify each source by type.**
Input: the list.
Output: regulation, guidance, standard, code set, dataset, literature, market, community.
Rationale: the type determines the cadence and the trigger.
Branch: a source that mixes types, such as a guidance with a code annex, should be split.
Edge case: a page can host several documents with different statuses.
Check: is every source classified.
Record: the classification.

**Step 3. Assign an owner to each source.**
Input: the team roles.
Output: one named person per source or source group.
Rationale: an unowned source is not monitored, and a shared inbox is not an owner.
Branch: if no one can own a group, reduce the list rather than leave it unowned.
Edge case: an owner can be external, such as a regulatory consultant, with a contract.
Check: does every source have a name.
Record: the owners.

**Step 4. Set the cadence and the triggers.**
Input: the type and the decision horizon.
Output: an interval and a list of events.
Rationale: cadence follows volatility, and triggers catch the changes that matter before the interval expires.
Branch: a source with a known release date should be reviewed on that date, not on a quarterly cycle.
Edge case: an election or a court decision can change a timeline without a release.
Check: does every source have both.
Record: the cadence and triggers.

### Phase B — Detect changes

**Step 5. Choose the detection method for each source.**
Input: the issuer's channels.
Output: a feed, an alert, a mailing list, or a manual check.
Rationale: the method should match how the issuer communicates, not how the team prefers to read.
Branch: if the issuer has no feed, use a saved search or a calendar reminder.
Edge case: some issuers announce changes only in an official journal.
Check: is the method recorded.
Record: the method.

**Step 6. Subscribe and test the method.**
Input: the chosen channels.
Output: a confirmed subscription and a test notification.
Rationale: an unverified subscription is a hope, not a method.
Branch: if a feed is noisy, filter by keyword rather than abandoning it.
Edge case: an alert can silently stop when a page is restructured.
Check: has a test notification arrived.
Record: the subscription.

**Step 7. Collect changes into one queue.**
Input: the notifications.
Output: a dated list of changes with links.
Rationale: a single queue prevents a change from being handled twice or not at all.
Branch: if the queue grows faster than it is processed, tighten the filters.
Edge case: duplicate notifications are common and should be de-duplicated by URL.
Check: is there one queue.
Record: the queue.

**Step 8. Check the issuer's changelog for unannounced changes.**
Input: the issuer's changelog or revision history.
Output: changes that were not announced.
Rationale: not every revision is announced, and the changelog is the authoritative list.
Branch: if there is no changelog, compare the current page with the archived snapshot.
Edge case: a changelog can be incomplete or undated.
Check: is the changelog reviewed.
Record: the unannounced changes.

### Phase C — Triage

**Step 9. Classify each change by impact.**
Input: the change and the documents that cite the source.
Output: high, medium, or low impact.
Rationale: impact is a property of the dependency, not of the change's prominence.
Branch: a change to a definition used in a label is high impact even if the document is short.
Edge case: a change can be high impact for one team and none for another.
Check: is the impact tied to a document.
Record: the impact.

**Step 10. Apply the triage rule.**
Input: the impact and the source type.
Output: act now, schedule, watch, or ignore.
Rationale: the rule prevents both over-reaction and neglect.
Branch: high impact is always act now; low impact on a non-regulatory source is ignore.
Edge case: a medium-impact code change can become high if the code is used in a cohort definition.
Check: is the action one of the four.
Record: the action.

**Step 11. Estimate the effort for each action.**
Input: the change and the affected documents.
Output: hours per action.
Rationale: the estimate makes the load visible and prevents an unsustainable promise.
Branch: if the total exceeds the budget, defer the low-impact items explicitly.
Edge case: an action can require a meeting, which is more than the reading time.
Check: is the effort recorded.
Record: the hours.

**Step 12. Decide what to read in full.**
Input: the action list.
Output: the items to read completely and the items to skim.
Rationale: reading everything is not sustainable, and reading nothing is not maintenance.
Branch: act-now items are read in full; watch items are skimmed for relevance.
Edge case: a summary can be enough for a watch item and never enough for an act-now item.
Check: is the reading depth recorded.
Record: the reading plan.

### Phase D — Act and propagate

**Step 13. Update the source register.**
Input: the change and the new version.
Output: an updated register entry with the date and the change.
Rationale: the register is the record that the loop depends on.
Branch: if the source is superseded, record the successor and the reason.
Edge case: a source can change without a version number, which is why the date is recorded.
Check: is the register current.
Record: the entry.

**Step 14. Propagate to every dependent document.**
Input: the source-to-document map.
Output: an updated list of documents and the changes made.
Rationale: a change applied in one place and missed in four is worse than no update, because it creates inconsistent documents.
Branch: if a document cannot be updated immediately, mark it as stale and date the mark.
Edge case: a template can carry the stale text into future documents.
Check: are all dependents updated or marked.
Record: the propagation list.

**Step 15. Propagate to models and pipelines.**
Input: the model register and the pipeline code.
Output: a statement of whether the change affects features, labels, or thresholds.
Rationale: a definition or code change can silently alter a model's inputs.
Branch: if the change affects a label, re-run the validation.
Edge case: a code release can change a cohort without changing the rule.
Check: is the model impact assessed.
Record: the model impact.

**Step 16. Escalate material changes.**
Input: the act-now items.
Output: a decision-maker informed with the change, the impact, and the recommendation.
Rationale: material changes often require a decision that the source owner cannot make alone.
Branch: a change to a regulatory timeline is material to a launch plan.
Edge case: an escalation can be a single paragraph, not a meeting.
Check: is the escalation recorded.
Record: the escalation.

### Phase E — Retire and audit

**Step 17. Retire superseded sources.**
Input: the register.
Output: a status change, a successor link, and a tombstone entry.
Rationale: retirement keeps the register honest and prevents reuse of the old source.
Branch: if the source is needed for a historical claim, keep it with a date and a "historical" label.
Edge case: a source can be withdrawn without a successor, which leaves a gap to record.
Check: is every superseded source marked.
Record: the retirement.

**Step 18. Archive what was read.**
Input: the page or document.
Output: a dated snapshot and a permalink.
Rationale: the archive is the evidence of what the source said at the review date.
Branch: if the source is paywalled, archive the citation and the passage, not the whole document.
Edge case: an archive can be the only remaining copy after a withdrawal.
Check: is the snapshot dated.
Record: the archive.

**Step 19. Run the quarterly audit.**
Input: the register and a sample of sources.
Output: a coverage statement, a status check, and a list of gaps.
Rationale: the audit checks the register itself, not only the sources.
Branch: if coverage is below the target, reduce the list rather than the checks.
Edge case: an audit should sample sources that have not changed, because they are the ones most likely to be unchecked.
Check: is the audit documented.
Record: the audit.

**Step 20. Review the routine's cost and adjust.**
Input: the hours spent and the changes found.
Output: an adjusted cadence or list.
Rationale: a routine that costs more than the risk it manages will be abandoned.
Branch: if the routine is too expensive, automate detection and keep human triage.
Edge case: a quiet quarter is not evidence that the routine is unnecessary.
Check: is the cost recorded.
Record: the cost.

The procedure produces a routine that is cheap enough to sustain and complete enough to catch the changes that matter. It does not eliminate the risk of a missed change, and it makes the risk visible and bounded.

### The quarterly checklist

- Confirm that every source on the list still exists at the issuer.
- Check status, not only the link.
- Check the changelog for unannounced revisions.
- Review the alert queue and clear it.
- Confirm that every act-now item was propagated.
- Confirm that every superseded source is marked.
- Sample five documents for stale citations.
- Sample five models for version drift.
- Record the hours spent.
- Record what changed and what did not.
- Update the cadence if a source proved quieter or noisier than expected.
- Report coverage to the owner.

The checklist takes about two hours for a list of thirty sources, which is the budget the routine is designed for.

### Archive checklist

- Capture the page or document as it appeared.
- Record the date and the reviewer.
- Record the version or document number.
- Store the snapshot where the team can find it.
- Link the snapshot from the register.
- Note any passage relied on.
- Re-check that the archive opens.

An archive that nobody can open is not an archive, and a snapshot without a date is not evidence.

### What the routine costs

| Activity | Frequency | Hours per cycle | Hours per year |
|---|---|---:|---:|
| Alert triage | Weekly | 0.5 | 26 |
| Quarterly checklist | Quarterly | 2 | 8 |
| Regulatory deep check | Semiannual | 4 | 8 |
| Register maintenance | Monthly | 0.5 | 6 |
| Archive and retirement | Quarterly | 1 | 4 |
| Annual review of the routine | Annual | 3 | 3 |

The total is about 55 hours a year, or roughly one hour a week. That is the realistic cost of keeping a thirty-source register current, and it is small relative to the cost of one stale regulatory claim.

The estimate excludes the first build of the register, which takes about two days for a mature programme and is the largest single cost of the routine.

### Detection methods compared

| Method | Latency | Completeness | Effort | Best for |
|---|---|---|---|---|
| Official feed | Minutes to days | High for that issuer | Low | Regulators with feeds |
| Mailing list | Days | Medium | Low | Guidance updates |
| Saved search | Days | Medium | Low | Literature |
| Changelog review | On demand | High | Medium | Standards and code sets |
| Manual page check | On demand | Depends on the page | High | Issuers without feeds |
| Community mention | Variable | Low | Low | Early signal, not verification |

A feed tells you that a page changed. It does not tell you whether the change matters, and it does not cover issuers without feeds.

### Triage examples

| Change | Impact | Action | Why |
|---|---|---|---|
| Guidance revision affecting a claim | High | Act now | The claim may be unsupported |
| Code release adding a new code | Medium to high | Schedule | The cohort may change |
| New edition of a textbook | Low to medium | Watch | Only therapy chapters matter |
| Benchmark paper with a new result | Low | Watch | No decision depends on it |
| Retraction in the reading list | High | Act now | A document may cite it |
| Journal policy change | Low | Ignore | No local dependency |

The rule is not a substitute for judgement. It is a way to make the judgement consistent across reviewers and to keep the volume manageable.

### A propagation map

| Source change | Document | Model or pipeline | Action |
|---|---|---|---|
| Definition revision | Data dictionary, model card | Label logic | Re-derive labels |
| Threshold change | Validation plan, slide deck | Decision threshold | Re-run evaluation |
| Code release | Cohort specification | Extraction query | Re-extract and compare |
| Guidance revision | Regulatory summary, submission | None | Update and re-review |
| Dataset release | Data register, methods section | Training pipeline | Re-train or re-report |
| Retraction | Any document citing it | None | Replace and annotate |

The map is the artefact that turns a source change into a work list. Without it, propagation depends on someone remembering where the source was used.

### A branch worked in miniature: the quiet quarter

A quarterly review finds no changes. The team concludes that the routine is unnecessary and skips the next quarter.

The next quarter brings two changes: a guidance revision that affects a claim, and a code release that changes a cohort. Neither is caught, and both are discovered in a review.

The branch is not about the quiet quarter; it is about what the quiet quarter means. A quiet quarter is evidence that the issuers were quiet, not that the routine is redundant. The correct response is to record the quiet quarter and keep the cadence, or to lengthen it with a trigger that fires on a release.

The alternative branch is to automate detection and keep human triage. That reduces the cost of a quiet quarter without removing the coverage.

- A quiet quarter is not a reason to stop.
- A busy quarter is not a reason to lengthen the cadence.
- The cadence follows the source, not the workload.

A quiet quarter should be recorded as a finding, because it is the only evidence that the cadence is proportionate.

## 6. Worked example

A regulatory summary on the intranet carries the line: "Last updated 12 September 2026. Current." The team relies on it for a launch decision in November.

### Framing the claim

The sentence "current" is a claim about status, and it is supported by a date. The date is real; the status is not established by the date.

| Sub-claim | Type | What could support it |
|---|---|---|
| The page was edited on 12 September | Fact | The content management system |
| The content reflects the issuer's position | Status | A check at the issuer |
| The position is still in force | Currency | The issuer's current list |
| The summary is adequate for the decision | Judgement | A check of the specific provision |

### Candidate sources

| # | Candidate | Type | Role |
|---|---|---|---|
| C1 | The intranet summary | Secondary | The object under test |
| C2 | The issuer's guidance page | Primary | Current status |
| C3 | The Federal Register notice | Primary | Issue date and availability |
| C4 | An archived snapshot from September | Archive | What the summary read |
| C5 | A colleague's recollection | Anecdote | The origin of "nothing changed" |
| C6 | A vendor newsletter | Secondary | The likely source of a claim that it changed |

### Scoring the candidates

**C1, the summary.**
- Authority: internal, secondary.
- Currency: the edit date is recent; the content may not be.
- Applicability: written for the team's decision.
- Accessibility: internal.
- Provenance: the author and their sources are not recorded.
- Verdict: usable only with a status check.

**C2, the issuer page.**
- Authority: highest for status.
- Currency: current at the access date.
- Applicability: general.
- Accessibility: public.
- Provenance: official.
- Verdict: primary support for status.

**C3, the Federal Register.**
- Authority: high for the issue date and the notice.
- Currency: current.
- Applicability: general.
- Accessibility: public.
- Provenance: official.
- Verdict: primary support for the date.

**C4, the archive.**
- Authority: high for what the page said.
- Currency: historical.
- Applicability: the summary's content.
- Accessibility: the archive service.
- Provenance: dated snapshot.
- Verdict: evidence of the summary's basis.

**C5, the recollection.**
- Authority: low.
- Currency: unknown.
- Applicability: personal.
- Accessibility: a conversation.
- Provenance: none.
- Verdict: not evidence.

**C6, the newsletter.**
- Authority: low to moderate.
- Currency: current.
- Applicability: general.
- Accessibility: public.
- Provenance: secondary.
- Verdict: a prompt to check C2.

### The check

The check is two minutes of work and produces a different answer from the date:

- The issuer's guidance page shows a revision dated 2 October 2026.
- The revision changes the transition provision that the summary describes.
- The summary was edited on 12 September and did not include the revision.
- The archive confirms that the summary matched the pre-revision page.

The summary was current when it was written. It is not current at the review date, and the edit date is not evidence otherwise.

### Verdict

| Sub-claim | Verdict | Supportable sentence |
|---|---|---|
| The page was edited in September | Fact | "The summary was last edited on 12 September 2026." |
| The content is current | Reject | "The issuer revised the guidance on 2 October 2026; the summary predates the revision." |
| Adequate for the November decision | Reject until updated | Requires the revised transition provision |

The corrected line is: "Last checked against the issuer on 26 October 2026; revision of 2 October incorporated." That line is longer and it is the only one that supports a decision.

### Two diagnostics that would change the verdict

**Diagnostic 1: the revision does not affect the cited provision.** If the October revision changed an unrelated annex, the summary's relevant content could still be current, and the verdict would shift to "current for the provision cited, with the revision noted". The diagnostic is a comparison of the revision's scope with the provision relied on.

**Diagnostic 2: the summary's author performed a status check.** If the author recorded a check at the issuer on 12 September, and the issuer's revision came later, the summary was current on its date and the failure is the absence of a trigger, not the author's work. The verdict becomes a process finding: the routine lacked a trigger for the issuer's revision.

A third diagnostic, weaker, is a public announcement of the revision before 12 September. It would show that the summary was stale when written, which is a different failure with a different fix.

### What the team should record

The register records the summary, its edit date, the issuer page, the revision date, and the access date. The summary is updated with the revision and a check line. The trigger is added: any revision to the issuer's guidance forces a review of the summary within five working days.

The case produces a rule: "last updated" is a fact about a file, not a status. A status line must name the issuer and the date of the check.

### Status lines compared

| Status line | What it asserts | What it establishes |
|---|---|---|
| "Updated September 2026" | The file was edited | Nothing about the issuer |
| "Reviewed internally" | Someone read it | Nothing about status |
| "Current" | A claim | Nothing without a source |
| "Checked against the issuer, 26 October 2026" | A source and a date | Status at that date |
| "Version 2.4, issuer revision of 2 October" | A version and a revision | Status and change |

Only the last two lines are auditable. The first three are common and none of them can be checked.

### What a status check requires

- The issuer's current list, not a search result.
- The document number and the revision date.
- The specific provision relied on.
- The access date and the reviewer.
- A note of any revision that affects the provision.

A status check that does not name the provision is a check of the document, not of the claim.

### Triggers that force an early check

| Trigger | Why it matters | Where to watch |
|---|---|---|
| Issuer revision | The summary may be stale | The issuer's guidance list |
| New draft guidance | The position may change | The guidance database |
| Transition deadline | The applicable route changes | The regulation's transitional articles |
| Court or legislative action | The instrument may be vacated | The official journal |
| Code release | The cohort may change | The code set release notes |
| Retraction | A citation may be invalid | The retraction database |

Each trigger should be linked to the sources it affects, so that a notification becomes a work item rather than a headline.

### What the reviewer records

- The source and the provision checked.
- The issuer's page and the access date.
- The revision date and the version.
- The verdict: current, revised, superseded, or unknown.
- The documents affected.
- The next check date.

The record is short, and it is the only thing that distinguishes a status check from a read.

The difference between a check and a read is the output. A read produces an impression; a check produces a dated line that another person can audit.

## 7. Code, query, or calculation
