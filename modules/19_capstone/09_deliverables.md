# Deliverables

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 09 of 11
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, and 08
> Version 2026.3

## 1. Why this topic exists

A capstone produces four deliverables, and they serve four different readers.
The dossier is the evidence record that an employer, a regulator, or a reviewer would request.
The prototype is the working artefact that demonstrates the pipeline runs.
The demo is the ninety-second proof that a stranger can see the output.
The defense is the twenty-minute conversation in which three people test whether you understand
what you built.

Teams usually over-invest in one and neglect the others.
The common pattern is a strong prototype and a weak dossier: a model that works, a repository with
no README, a validation table pasted into slides, and no regulatory memo.
The panel reads the dossier first, finds that the numbers cannot be reproduced, and spends the
defense asking why.

The cost is not only a grade.
In industry, the dossier is the artefact that survives the team.
A model handed over without a data provenance record, a model card, and a monitoring plan is a
liability that the receiving team will either rebuild or retire.
The external validation of a widely used sepsis model by Wong and colleagues in 2021 is a case in
point: the vendor's reported performance could not be reproduced on an academic cohort, and the
gap was discoverable only because an independent team had the documentation and the data to check.

Regulators have made the same point in the language of artefacts.
The FDA's predetermined change control plan guidance, finalised in December 2024, requires a
description of modifications, a modification protocol with acceptance criteria, and an impact
assessment.
TRIPOD+AI requires the intended use, the population, the model specification, calibration, subgroup
performance, and code sharing.
DECIDE-AI requires human factors, human-computer agreement, and safety reporting for early live
evaluation.
None of these is a capstone requirement, and all of them describe the same object: a dossier that
lets a stranger understand, check, and govern the work.

This topic specifies the four deliverables, gives each an acceptance criterion that can be tested,
and provides a self-assessment that a team can run before freeze.
It is the bridge between the evidence produced in weeks 3 to 10 and the defense in week 12.

The deliverables also solve a problem that is specific to team projects: they make individual
contribution visible.
A dossier has named owners per artefact, a defense has questions that any member may be asked, and
a repository has commit history. A team that builds the four deliverables together produces a record
of who did what, which is what the peer adjustment and the individual defense depend on.

None of the four is optional, and the weakest one sets the ceiling. A strong prototype with no
dossier is a demo; a strong dossier with no working prototype is a report; a strong demo on a
training case is a warning sign. The panel scores the combination, not the maximum.

## 2. Learning outcomes

The outcomes below describe what a team can do, not what it can produce. A team can produce a
dossier by copying a template; it can only defend one by understanding it.

By the end of this topic you can:

1. **List** the thirteen dossier artefacts, their weights, and the acceptance test for each.
2. **Write** a model card and a datasheet that a reviewer can check against the repository.
3. **Build** a reproducibility package that a stranger can run with one command, and state the
   tolerance within which the primary metric reproduces.
4. **Design** a demo that runs offline, on held-out or synthetic data, and shows one failure case.
5. **Prepare** a defense that answers questions on the decision, the pipeline, and the governance,
   including questions about parts you did not own.
6. **Apply** an acceptance criterion to your own artefact and record a pass or fail with evidence.
7. **Run** a self-assessment that produces a weighted completeness score and a list of blocking
   gaps.
8. **Distinguish** a prototype from a product in every sentence of the readout and the pitch.

Outcomes 6 and 7 are assessed in Exercises 3 and 6, which together produce the readiness record.
Outcome 8 is assessed in Exercise 5 and in the readout, where every sentence is checked against the
model card.

The outcomes are ordered by the sequence of assembly. A team that cannot list the artefacts cannot
write acceptance criteria for them, and a team that cannot write criteria cannot run an honest
self-assessment.

A note on what is not an outcome: "understand the importance of documentation" is not observable and
cannot be assessed. Every outcome in this list names something a team can do in front of a reviewer.

The same rule applies to the dossier itself. "Describe the limitations" is not an acceptance
criterion; "list three limitations, each tied to a metric or a subgroup, in the model card" is.

## 3. The core mental model

The four deliverables are four views of one project, addressed to four readers.
The dossier is the full record; the prototype is the running system; the demo is the visible
behaviour; the defense is the oral examination.
A weakness in any one is visible in the others, which is why they must be built together.

```text
                 READER
        clinician | engineer | compliance | executive
             |         |           |          |
  DOSSIER    *         *           *          *      full record, 13 artefacts
  PROTOTYPE            *                             running pipeline, versioned
  DEMO       *         *                      *      held-out case, 90 seconds
  DEFENCE    *         *           *          *      questions, 20 minutes

  ACCEPTANCE CRITERION
  "A competent stranger can <do X> and observe <Y> without asking the team."
```

Read the diagram by column.
The clinician cares whether the decision is real and the errors are safe.
The engineer cares whether the pipeline runs and can be maintained.
The compliance reader cares what claim is being made and on what authority.
The executive cares who pays and what changes.

The acceptance criterion is the same shape for every artefact: a competent stranger can do
something and observe something without asking the team.
For the dossier, the stranger can find the intended use and the validation result in under five
minutes.
For the prototype, the stranger can run one command and reproduce the primary metric within a
stated tolerance.
For the demo, the stranger can see the input, the output, and one failure case in ninety seconds.
For the defense, the stranger can ask about any artefact and get an answer that cites the artefact.

Three properties of the model matter.

**The dossier is not a report about the model.** It is the evidence that the claim is supported.
A dossier that describes the architecture and omits the comparator has described the method and
not the evidence.

**The prototype is not the product.** It runs on a held-out or synthetic case, without the
governance, integration, and monitoring a product needs. The readout must say so, and the model
card must repeat it.

**The demo is not the defense.** The demo shows that the artefact works; the defense shows that the
team understands why, when it fails, and what it would take to deploy. A team can pass the first
and fail the second.

A fourth property is about time.
The deliverables are assembled in weeks 9 to 11, and they cannot be assembled from nothing.
The charter written in week 1 becomes the dossier's first section; the analysis plan written in
week 5 becomes its methods; the error analysis from week 6 becomes its error section.
A team that treats the dossier as a week-11 writing task will produce a document that contradicts
its own repository.

### The four deliverables as a chain of custody

Each deliverable hands the next one a claim it must not exceed.

The dossier establishes what is supported.
The prototype demonstrates that the pipeline produces the output the dossier describes.
The demo shows the prototype behaving on a case the team did not tune on.
The defense tests whether the team can explain the chain from decision to evidence to system to
governance.

A break anywhere in the chain is visible. A prototype that produces a different score from the
dossier breaks the second link; a demo on a training case breaks the third; a presenter who cannot
explain the calibration breaks the fourth.

The chain is also the order of preparation. Building the demo before the dossier invites a demo
that shows something the evidence cannot support.

### Why four deliverables and not one

A single document would be simpler to produce and worse for every reader. The clinician would have
to wade through code, the engineer through the value case, and the executive through the
regulatory memo.
The four deliverables are a form of audience segmentation, and each one is short because it omits
what its reader does not need.
The dossier is long because it is the record; the readout is one page because it is the decision;
the demo is ninety seconds because attention is finite; the defense is twenty minutes because
questions are how understanding is tested.

Segmentation also means the deliverables must not contradict each other. The readout's recommendation
must follow from the dossier's evidence, the demo must show the prototype the dossier describes, and
the defense must be able to open any of them. A contradiction between two deliverables is worse than
a weakness in one, because it suggests the team does not know which version is true.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Deliverable | A versioned artefact with an acceptance test | Any output | Artefact and version |
| Dossier | The thirteen-artefact evidence record submitted at freeze | A slide deck | Repository plus index |
| Artefact | One document or package in the dossier | A section of a report | File and version |
| Acceptance criterion | A test a reviewer can apply to decide pass or fail | A description of quality | Observable action |
| Prototype | A working artefact demonstrating the pipeline on held-out data | A product | Maturity label |
| Demo | A timed demonstration of the prototype | A slide walkthrough | Seconds and case |
| Defense | An oral examination on any part of the work | A presentation | Minutes and questions |
| Model card | Structured description of intended use, data, metrics, limits | A paragraph in an appendix | Standard sections |
| Datasheet | Structured description of a dataset's origin, composition, and uses | A data dictionary | Standard sections |
| Readout | One-page decision summary with recommendation and evidence | An executive summary | One page |
| Evidence table | The comparison of comparator and candidate with intervals | A results slide | Table |
| Reproducibility package | Code, environment, seeds, and instructions to rerun | A repository link | Files and command |
| Tolerance | The acceptable difference between reproduced and reported metrics | Exact match | Absolute or relative |
| Freeze | The date after which only factual corrections are allowed | The defense date | Date and tag |
| Version tag | The identifier of the frozen artefact set | A file name | Semantic version |
| Self-assessment | A structured check against acceptance criteria before freeze | A confidence statement | Score and gaps |
| Blocking gap | A missing artefact that fails the defense regardless of score | A weakness | Artefact and weight |
| Audit trail | A record of who changed what and when | A commit history alone | Log and identity |
| Intended use | Population, setting, inputs, output, user, contraindications | A model description | One paragraph |
| Limitations | What the evidence does not support | A disclaimer | Named claims |
| Provenance | Source, version, access date, licence, transformations | The file path | Metadata record |
| Regulatory memo | One page naming the regime, classification, and binding artefact | A legal opinion | One page |
| Risk register | Risks with likelihood, impact, owner, trigger, mitigation | A list of worries | Scored table |
| Value case | Comparator cost, budget impact, adoption assumptions | A revenue projection | Currency and assumptions |
| Maturity level | A stated position from concept to validated product | "Production ready" | Label and evidence |
| Consistency check | Comparison of every quoted number with the source file | A proofread | Count of discrepancies |
| Red-team review | Structured rehearsal by an outsider | A practice run | Questions and artefacts |
| Handover package | Repository, model card, monitoring, owner | A zip file | Files and named owner |
| Claim boundary | The sentence the evidence supports and the one it does not | A disclaimer | Two sentences |

The last definition is the one a panel tests most often. A claim boundary has two parts, and the
second is usually missing: "this model ranks patients for review on held-out data from one site"
and "it does not establish that review improves outcomes". A dossier that states only the first
invites the panel to ask about the second.

The most consequential definition is the acceptance criterion.
It is not a statement about quality in general; it is a test that a specific reader can apply to a
specific artefact.
"Validation report is thorough" is not a criterion. "A reviewer can find the primary metric, its
confidence interval, the split definition, and the calibration plot in under ten minutes" is.

A second definition matters because it is frequently abused: the prototype.
A prototype demonstrates one step on a small sample, without the integration and governance of a
product. A team that calls its prototype a product invites questions about maintenance, security,
and regulation that it cannot answer, and the panel will ask them.

## 5. The mechanics, step by step

The procedure assembles the four deliverables across weeks 9 to 11.
It assumes the artefacts from earlier weeks exist; if they do not, the first step is to recover
them, not to write around them.

### Step 1. Freeze the evidence inputs

**Input:** the results, the analysis plan, and the repository.

**Output:** a frozen results table with the commit hash that produced it, and a list of figures
generated by script.

**Rationale:** Every later artefact quotes a number, and numbers drift when code changes. Freezing
the evidence inputs makes the dossier internally consistent and gives the model card and readout a
single source.

**Branch point:** if a number in the slides cannot be traced to a script output, it is not a result.
Regenerate it or remove it.

### Step 2. Assemble the repository to the required layout

**Input:** the code, data documentation, and governance files.

**Output:** a repository with `README.md`, `charter.md`, `data/`, `src/`, `configs/`, `results/`,
`governance/`, and `docs/`.

**Rationale:** The layout is a contract with the reviewer. A reviewer who knows where to look can
verify an artefact in minutes; a reviewer who has to explore a flat folder will sample instead of
check.

**Branch point:** if the repository contains data that cannot be redistributed, move it to a
documented access statement and keep a synthetic sample. The layout must survive the licence.

### Step 3. Write the data provenance record

**Input:** the data dictionary, the access agreement, and the extraction query.

**Output:** a one-page record with source, version, access date, licence, de-identification method,
row counts, and known limitations.

**Rationale:** Provenance is the artefact that makes the rest checkable. It is also the artefact
most often missing, because it is nobody's modelling task.

**Branch point:** if the source cannot be named because of a confidentiality agreement, name the
tier and the custodian, and say why the source is withheld.

### Step 4. Write the analysis plan and confirm it predates the results

**Input:** the plan from week 5 and the commit history.

**Output:** the plan with a timestamp, plus a change log for any post-hoc change.

**Rationale:** The plan is the defence against choosing the analysis after seeing the result. Its
value depends on the timestamp being earlier than the first result commit.

**Branch point:** if the plan was written after the results, say so and treat the analysis as
exploratory. An honest exploratory analysis is acceptable; a disguised confirmatory one is not.

### Step 5. Build the results and evidence tables

**Input:** the frozen results.

**Output:** a table with comparator and candidate point estimates and intervals, a calibration
plot, a decision curve, and a subgroup table with counts.

**Rationale:** The evidence table is what the panel reads first. It must show the comparator, not
only the candidate, and it must show intervals, not only points.

**Branch point:** if a subgroup has fewer than 30 events, report it with a warning and do not draw
a conclusion from the point estimate.

### Step 6. Write the validation report

**Input:** the evidence tables, the split definition, and the leakage screen.

**Output:** a report with the validation level, what the split proves, the primary and secondary
metrics with intervals, calibration, subgroups, and the leakage screen result.

**Rationale:** The validation report is the heaviest-weighted artefact because it carries the
claim. A report that states the level and the limits is more persuasive than one that reports a
high number without context.

**Branch point:** if the validation level is below the claim, narrow the claim in the report and in
the model card. Do not let the claim and the evidence diverge.

### Step 7. Write the error analysis

**Input:** twenty to thirty reviewed errors and the taxonomy.

**Output:** a table of error categories with counts, two or three examples, and a statement of what
the errors imply for the decision.

**Rationale:** Error analysis is where a team shows that it has looked at individual cases rather
than only at aggregate metrics. It is also where the most useful limitations come from.

**Branch point:** if the errors cluster in a subgroup, that is a finding, and it belongs in the
subgroup section as well as the error section.

### Step 8. Write the model card and datasheet

**Input:** the intended use, the training data, the metrics, and the limits.

**Output:** a model card with intended use, users, training data, evaluation data, metrics by
subgroup, limitations, and ethical considerations; a datasheet with motivation, composition,
collection, preprocessing, uses, and maintenance.

**Rationale:** These two artefacts are the standard way to communicate what a model is and is not.
They also force the team to state the limitations in the same place as the metrics, which prevents
selective quotation.

**Branch point:** if the model card's limitations section is empty, the team has not looked. Return
to the error analysis.

### Step 9. Write the regulatory memo

**Input:** the intended use, the user, the output, and the jurisdiction.

**Output:** one page naming the regime, the preliminary classification, the binding artefact, three
open questions, and the residual risk if the classification is wrong.

**Rationale:** The memo is where the claim meets the law. It is not legal advice, and it must not
pretend to be; it is a structured triage that a regulatory mentor can review in ten minutes.

**Branch point:** if the classification is uncertain, state both readings and the evidence that
would resolve it. An uncertain memo is more useful than a confident wrong one.

### Step 10. Write the risk register and monitoring plan

**Input:** the risks from the project and the deployment design.

**Output:** a scored register with owners, triggers, and mitigations; a monitoring plan with the
signals, thresholds, and response.

**Rationale:** The register shows that the team has thought about failure; the monitoring plan shows
that it has thought about failure after handover. Together they are the difference between a
project and a product.

**Branch point:** if the monitoring plan requires data the team will not have, simplify it to
signals that exist. A plan that cannot be executed is worse than a shorter one that can.

### Step 11. Write the deployment and value case

**Input:** the workflow map, the capacity, and the cost data.

**Output:** a serving diagram with latency and failure behaviour; a value case with comparator
cost, budget impact, and three assumptions with ranges.

**Rationale:** The value case answers the executive's question, and the serving diagram answers the
engineer's. Both are frequently omitted by teams that assume the model speaks for itself.

**Branch point:** if the value case depends on an assumption the team cannot defend, present a
range and a break-even point rather than a single number.

### Step 12. Write the readout and build the demo

**Input:** the evidence, the intended use, and a held-out case.

**Output:** a one-page readout and a ninety-second demo that runs offline.

**Rationale:** The readout is the only artefact some readers will read, and the demo is the only
artefact some will remember. Both must be consistent with the dossier, which is why they come after
it.

**Branch point:** if the demo needs a network, record a backup and say that the recording is the
primary artefact.

The readout is written last and edited hardest. Its first sentence is the recommendation, and the
recommendation must be one of four: deploy, pilot, iterate, or stop. A readout that recommends
"further research" has avoided the decision, and a panel reads avoidance as a missing result.

### Step 13. Run the stranger test

**Input:** the repository and a person who has not worked on the project.

**Output:** a timed record of whether the stranger could find the intended use, reproduce the
primary metric, and explain the regulatory path.

**Rationale:** The stranger test is the only reliable check of the acceptance criteria. The team
cannot test its own artefacts because it knows where everything is.

**Branch point:** if the stranger cannot reproduce the metric, fix the package before adding
anything else. Reproducibility is a gate, not a nice-to-have.

### Step 14. Run the self-assessment

**Input:** the thirteen artefacts and their acceptance tests.

**Output:** a weighted completeness score, a list of blocking gaps, and a remediation plan with
dates.

**Rationale:** The self-assessment converts a vague sense of readiness into a list. It also gives
the team a way to decide what to fix in the last week, which is usually one artefact rather than
five.

**Branch point:** if the score is below the pass threshold, fix the highest-weight blocking gap
first. A missing validation report is worth more than a polished readout.

The self-assessment is not a grade predictor. It measures whether the artefacts exist and pass
their tests, not whether the evidence is convincing. A dossier can score 100 percent and still
contain a weak validation, which is why the panel reads the evidence rather than the checklist.

### Step 15. Freeze and tag

**Input:** the completed artefacts.

**Output:** a version tag, a frozen repository, and a dossier index with links.

**Rationale:** The tag is what makes the defense a description rather than a negotiation. Every
number quoted on stage must correspond to the tag.

**Branch point:** if a factual error is found after the tag, correct it, re-tag, and announce the
correction. A disclosed correction is a strength; a discovered inconsistency is not.

### Step 16. Rehearse the defense twice

**Input:** the frozen artefacts.

**Output:** two timed runs, one with a mentor, and a list of questions the team could not answer.

**Rationale:** The defense tests understanding, not memory. The rehearsal should include questions
about artefacts the presenter did not own, because the panel will ask them.

**Branch point:** if a question cannot be answered from the repository, the repository is
incomplete. Add the missing artefact or record the gap as a limitation.

### Step 17. Assemble the dossier index

**Input:** the thirteen artefacts and their versions.

**Output:** a one-page index with artefact, version, owner, and link, plus a two-line summary of
what the dossier claims.

**Rationale:** The index is what makes the dossier navigable. A reviewer who opens a folder of
files and cannot find the validation report will sample the wrong artefact or give up.

**Branch point:** if the index does not fit on one page, the dossier has too many artefacts or too
many versions. Keep the current version of each and archive the rest.

The index is also the artefact that makes the defense navigable. When a panel member asks about
calibration, the presenter can open the index, click the validation report, and show the plot in
under thirty seconds. That speed is what "the repository is the source of truth" means in practice.

### Step 18. Run a red-team review

**Input:** the frozen dossier and a reviewer who has not worked on the project.

**Output:** ten questions the reviewer would ask at a defense, with the team's answers and the
artefact each answer cites.

**Rationale:** The red team finds the questions the team has stopped seeing. It is the cheapest form
of rehearsal and the closest approximation to the panel.

**Branch point:** if a question has no artefact behind it, either add the artefact or remove the
claim that prompted the question.

A red team of one is enough. The value comes from the questions being asked by someone who has not
already decided what the answers are, and a single colleague from another project supplies that. The
output is a list of ten questions with the artefact that answers each, or a list of gaps.

### Decision table: what to do when an artefact is missing at freeze

| Missing artefact | Weight | Options | Recommended | Who decides |
|---|---:|---|---|---|
| Validation report | 15 | Re-run validation; downgrade claim; delay freeze | Downgrade claim and re-run within 48 hours | Panel at G5 |
| Reproducible code | 12 | Fix environment; ship container; document failure | Ship container with a tested command | Technical mentor |
| Results table | 10 | Regenerate; use baseline; report as incomplete | Regenerate from the frozen commit | Team lead |
| Data provenance | 8 | Reconstruct from emails; state unknown | State what is known and what is not | Data steward |
| Model card | 8 | Write from the repository | Write it; it takes two hours | Modeler |
| Monitoring plan | 8 | Simplify to available signals | Simplify and state the gap | Engineering mentor |
| Regulatory memo | 7 | Triage without mentor review | Write it and mark it unreviewed | Evidence lead |
| Error analysis | 7 | Reduce to ten cases | Ten reviewed cases with a taxonomy | Clinical mentor |
| Risk register | 5 | Write from the decision log | Write it | Lead |
| Value case | 5 | Estimate with ranges | One-page range estimate | Commercial mentor |
| Charter | 5 | Recover from the repository | Recover and note the version | Lead |
| Readout | 2 | Write at freeze | Write it last | Lead |

### Edge cases and forced decisions

**The data licence prevents publishing the extract.** The reproducibility package becomes code plus
a data-access statement plus a synthetic sample. The stranger test then checks that the code runs on
the sample, and the dossier says that the primary metric cannot be reproduced without access.

**The primary metric does not reproduce.** Stop adding artefacts. A non-reproducible result
invalidates the validation report, the model card, and the readout at once. Fix the environment,
the seed, or the preprocessing scope before anything else.

**A team member cannot explain an artefact.** The defense will find this. Assign the artefact to a
backup, require a handoff note, and rehearse a question about it. If the gap cannot be closed,
record it in the limitations and do not claim the artefact as team work.

**The demo fails in rehearsal.** Switch to the recording, test it on the presentation machine, and
make the recording the primary artefact. A recorded demo is legitimate when it is labelled.

**The regulatory memo conflicts with the pitch.** The memo wins. Rewrite the pitch, because the memo
is the artefact that has been reviewed and the pitch is the artefact that has not.

**The freeze date arrives with a known gap.** Freeze anyway, record the gap in the limitations, and
announce it at the start of the defense. An announced gap costs less than a discovered one.

## 6. Worked example

A four-person team is assembling its dossier in weeks 9 to 11 for a capstone on 30-day readmission
risk. The model is built, the temporal validation is done, and the repository exists but has no
README.

### The first self-assessment

The team runs the self-assessment against the thirteen artefacts. The weighted completeness is 57
percent. Four artefacts are blocking:

- the reproducible code, because the environment is not pinned and the one-command run fails;
- the validation report, because the calibration plot exists but the calibration slope is not
  reported;
- the model card and datasheet, which have not been written;
- the deployment and monitoring plan, which is a paragraph in a notebook.

The score is 57 out of 100 against a pass threshold of 70, a shortfall of 13 points. The team has
nine working days.

### The arithmetic of remediation

The team estimates the hours for each gap: reproducible code 8, validation report 6, model card and
datasheet 6, monitoring plan 4. Total 24 hours across four people, or six hours each, which fits in
two days if the work is parallelised.

The panel's time is also a constraint. The reviewer minutes for a full dossier sum to 116, or about
two hours. A reviewer with two hours will read the charter, the validation report, and the results,
and will sample the rest. That is why the validation report carries the highest weight and why the
readout is worth only 2 percent.

### Diagnostic one: the stranger test fails at the first command

The team asks a student from another project to clone the repository and run the documented command.
The command fails because the requirements file lists `pandas>=1.5` and the analysis used a
behaviour that changed in 2.0.

The stranger spends forty minutes and gives up. That is the first diagnostic, and it is fatal to the
reproducibility artefact: a package that fails at the first command is not reproducible, regardless
of how good the code is.

The fix takes three hours: pin the versions, build a container, and test the command on a clean
machine. The team also adds a `make reproduce` target that runs the pipeline and writes the metrics
to `results/metrics.json`.

### Diagnostic two: the numbers do not match

The team compares the validation report's AUC with the value in the readout. The report says 0.71
(95 percent CI 0.67 to 0.75); the readout says 0.73. The difference comes from a re-run after a
preprocessing change, and the readout was written from memory.

The second diagnostic is a consistency check across artefacts. It takes twenty minutes and catches
the error that would have been found on stage. The team regenerates every number from
`results/metrics.json` and adds a check that fails the build if a quoted value differs from the
file.

### The revised plan

The team allocates the nine days: three days to close the four blocking gaps, two days for the
stranger test and fixes, two days for the readout and demo, and two days for rehearsal and buffer.

The self-assessment is re-run on day 6, after the gaps are closed. The score is 88 percent, with two
non-blocking gaps: the value case is a range estimate without a break-even calculation, and the
error analysis has eighteen cases rather than thirty. Both are recorded as limitations.

### Varying one assumption

Suppose the data-use agreement prohibits sharing the extract. The reproducibility package cannot
include the data, and the stranger test cannot reproduce the primary metric.

The team changes the acceptance criterion: the stranger must reproduce the metric on a synthetic
sample with a documented generator, and the dossier states that the real-data result requires
access. The package includes the generator seed, the schema, and a script that regenerates the
sample.

The change costs four hours and reduces the strength of the reproducibility claim. It is recorded in
the model card and the limitations, and it is the honest version of a constraint the team cannot
remove.

### What the panel saw

At the defense, the panel asked four questions.

"Show me the code that computes the calibration slope." The modeler opened the repository and ran
the script; the slope was 0.93, and the report had it.

"Why is the readout's AUC different from the report's?" It was not, after the fix, and the team
explained the consistency check.

"What happens when the model is wrong?" The team showed the error taxonomy and the review workflow,
and named the false-positive cost in minutes.

"Who would maintain this?" The team named the analytics team, the monitoring signal, and the
retraining trigger, and admitted that no maintenance agreement existed.

The dossier scored 88 percent, the defense was strong on evidence and weaker on adoption, and the
panel's main criticism was the value case. That is a typical outcome, and it is visible in the
self-assessment before the defense.

### The demo build

The demo took nine hours across two days, and most of that time was not modelling.

Three hours went to building a fixture: five synthetic patients with medication lists, discharge
summaries, and known discrepancies. The fixture is the demo's foundation, because it makes the
output reproducible and removes any dependency on real data.

Three hours went to the interface: a single page with the input on the left, the flags on the right,
and the evidence for each flag underneath. The interface is deliberately plain, because the demo is
ninety seconds and a complex screen cannot be read in that time.

Two hours went to the failure case: a patient whose reconciliation produces a false flag, shown with
the evidence that lets a reviewer reject it. The failure case is the part the panel remembers,
because it demonstrates that the team knows what the tool cannot do.

One hour went to the recording and the offline test, with the network disabled. The recording is
480 seconds long and is cut to 85 seconds for the pitch.

### The rehearsal

The first rehearsal ran to eleven minutes against a ten-minute slot. The team cut the pipeline
description, which the panel can read in the dossier, and kept the decision, the evidence, the
failure case, and the ask.

The second rehearsal included three questions from the red-team list: why the calibration slope was
0.93 rather than 1.0, how the subgroup gap was estimated, and what would happen if the model were
retrained on next year's data. The team answered the first two from the repository and the third
from the monitoring plan, which is exactly the pattern the defense rewards.

The rehearsal also exposed an inconsistency: the slide said "reduces pharmacist review time by 30
percent", and the value case supported only a range of 12 to 22 percent. The slide was rewritten,
and the discrepancy was recorded in the change log.

### The consistency audit in numbers

The team audited twenty-seven numbers across the readout, the slides, and the dossier. Three
disagreed with `results/metrics.json`, and two were rounded in a way that changed the claim: 0.71
became 0.73, and a 12 to 22 percent range became "up to 30 percent".

The audit took fifty minutes for three people, or 2.5 person-hours, and it removed the two most
likely sources of a public correction. The number of discrepancies is itself a useful metric: a
dossier with more than three is a dossier assembled by hand, and the fix is to generate the
artefacts rather than to check them.

### What the panel did not ask

The panel did not ask about the model architecture, the hyperparameters, or the size of the training
set beyond the cohort definition. It asked about the decision, the comparator, the calibration, the
subgroup gap, the regulatory path, and the maintenance owner.

That distribution is worth recording because it is stable across panels. A team that spends its
rehearsal on architecture is rehearsing the questions the panel will not ask.

### What the team would change with another week

The team's own answer, recorded in the debrief, is that it would spend the week on the value case
and on a second site.

The value case because the panel's main criticism was the break-even calculation, which was a range
without a threshold.
The second site because the intended use named community hospitals and the validation was
single-site, which is a gap the team had documented but not closed.

Neither is a modelling task. That is the pattern: after the evidence is built, the remaining work is
almost always governance, economics, or generalisation, and never another model.

## 7. Code, query, or calculation

The block computes a weighted completeness score, lists blocking gaps, and estimates reviewer time.
It uses only the standard library.

```python
artefacts = [
    ("Charter", 5, True, True),
    ("Data provenance", 8, True, True),
    ("Analysis plan", 8, True, True),
    ("Reproducible code", 12, True, False),
    ("Results", 10, True, True),
    ("Validation report", 15, True, False),
    ("Error analysis", 7, True, True),
    ("Model card and datasheet", 8, False, False),
    ("Regulatory memo", 7, True, True),
    ("Risk register", 5, True, True),
    ("Deployment and monitoring", 8, False, False),
    ("Value case", 5, True, True),
    ("Readout and pitch", 2, True, True),
]
total = sum(w for _, w, _, _ in artefacts)
earned = sum(w for _, w, present, passes in artefacts if present and passes)
print(f"weighted completeness {earned}/{total} = {earned/total:.1%}")
print("blocking gaps:", [(n, w) for n, w, p, t in artefacts if not (p and t)])
need = 0.70 * total
print(f"points needed for 70%: {need:.0f}; shortfall {need - earned:.0f}")

minutes = {"charter": 8, "provenance": 10, "analysis plan": 8, "code": 20,
           "results": 7, "validation": 15, "errors": 10, "model card": 7,
           "reg memo": 8, "risk": 5, "deployment": 8, "value": 6, "readout": 4}
print("reviewer minutes", sum(minutes.values()),
      "=", round(sum(minutes.values()) / 60, 1), "hours")
```

Expected output:

```text
weighted completeness 57/100 = 57.0%
blocking gaps: [('Reproducible code', 12), ('Validation report', 15), ('Model card and datasheet', 8), ('Deployment and monitoring', 8)]
points needed for 70%: 70; shortfall 13
reviewer minutes 116 = 1.9 hours
```

The `passes` flag is the acceptance test, not the presence of a file. A validation report that
exists but omits the calibration slope is present and failing, which is why the two flags are
separate.

A team should run this at the start of week 9, not week 11. The output is a work plan, and a work
plan needs time to execute.

### Extending the checker

Two extensions make the checker more useful in practice.

The first is a deadline column. An artefact that is missing but due in three days is a different
problem from one that is missing with no owner, and the plan should distinguish them.

The second is an evidence field per pass. A pass without a link to the artefact is an opinion, and
the self-assessment should require a path. The team that marks everything as passing without
evidence is the team that discovers the gaps at the defense.

A third extension, for a team that wants a harder test, is to weight by claim proximity: artefacts
that carry the claim, such as the validation report and the analysis plan, count double, and
artefacts that describe it, such as the readout, count once. The scoring changes; the message does
not, which is that the evidence artefacts are the ones that must be right.

## 8. Data, standards, and regulatory context

Each artefact in the dossier answers to a standard or a regulation, and the mapping is not
one-to-one. The table summarises the binding and non-binding instruments; the prose explains where
they conflict.

| Artefact | Standard or instrument | Binding | What it constrains |
|---|---|---|---|
| Charter | TRIPOD+AI items 1-4 | Convention | Intended use, population, outcome |
| Data provenance | HIPAA, GDPR, DUA | Law or contract | Access, de-identification, retention |
| Analysis plan | TRIPOD+AI items 5-12 | Convention | Splits, sample size, analysis |
| Reproducible code | TRIPOD+AI item 18 | Convention | Code and data sharing |
| Results | TRIPOD+AI items 13-16 | Convention | Estimates, intervals, calibration |
| Validation report | TRIPOD+AI, PROBAST | Convention | Performance and risk of bias |
| Error analysis | DECIDE-AI item 13 | Convention | Safety and errors |
| Model card | FDA labelling, model card practice | Partly binding | Intended use and limits |
| Datasheet | Dataset documentation practice | Convention | Composition and uses |
| Regulatory memo | FDA CDS guidance, EU MDR | Binding interpretation | Classification and claim |
| Risk register | ISO 14971 | Binding in a submission | Risk management process |
| Monitoring plan | FDA PCCP, post-market surveillance | Binding in a submission | Change and surveillance |
| Value case | Payer and procurement rules | Contextual | Cost and adoption claims |

### TRIPOD+AI

TRIPOD+AI (2024) has 27 main items and 52 sub-items, covering title, abstract, introduction,
methods, open science, patient and public involvement, results, and discussion. It supersedes the
2015 TRIPOD checklist and applies to regression and machine-learning models alike.
The items most often missing from capstone dossiers are sample size justification, calibration,
subgroup performance, and code sharing. A team that works through the checklist once will find that
half of its missing artefacts are reporting items rather than analyses.

### Model cards and datasheets

A model card describes intended use, users, training and evaluation data, metrics by subgroup,
limitations, and ethical considerations. A datasheet describes a dataset's motivation, composition,
collection process, preprocessing, uses, distribution, and maintenance.
Neither is a regulatory requirement in the United States, and both are expected practice in
industry. A dossier without them looks incomplete to a reviewer who has seen them before.

### FDA and EU instruments

The FDA clinical decision support guidance determines whether the output is a device function. The
predetermined change control plan guidance, finalised in December 2024, describes how a manufacturer
can pre-authorise model updates with a modification protocol, acceptance criteria, and an impact
assessment.
In the European Union, Regulation (EU) 2017/745 governs device classification and Regulation (EU)
2024/1689 adds obligations for high-risk AI systems, with a staged application timeline.
A capstone does not submit, and a regulatory memo that names the regime, the classification, and the
binding artefact is the appropriate level of detail.

### Audit trails and records

21 CFR Part 11 sets requirements for electronic records and signatures in records submitted to the
FDA, including audit trails. A capstone that writes to a system of record, even in a sandbox, should
record who wrote what and when. The audit trail is cheap to design in and expensive to retrofit.

### Where the instruments conflict

The most common conflict is between reproducibility and confidentiality. TRIPOD+AI encourages code
and data sharing; a data-use agreement prohibits redistribution. The resolution is code plus a
data-access statement plus a synthetic sample, which satisfies the spirit of the reporting standard
without breaching the agreement.
A second conflict is between the model card's limitations and the pitch's optimism. The model card
wins, because it is the artefact that has been reviewed against the evidence.

### What the dossier must contain even when the standard is silent

Three artefacts are not required by any of the instruments above and are required by the panel.

An error analysis with reviewed cases. No reporting guideline mandates twenty reviewed errors, and
a panel that asks "what does the model get wrong" expects them.
A subgroup table with counts. TRIPOD+AI requires subgroup performance, and a table with counts is
what makes the intervals interpretable.
A monitoring plan with a trigger. The PCCP guidance covers pre-authorised updates, and a capstone
that never submits still needs to say what signal would prompt a retrain.

The pattern is that the regulatory instruments cover the claim and the evidence, and the panel also
covers the work. A dossier written only to the standard will pass a regulatory reviewer and
disappoint a clinical one.

### A note on what the dossier is not

It is not a thesis. It does not need a literature review of forty pages, and it does not need a
novel method.
It is not a pitch. It does not need a market analysis or a revenue projection, and its value case
can be a range with three assumptions.
It is not a legal opinion. The regulatory memo is a triage, and it should say so.

A dossier that tries to be all three is usually none of them, and the panel reads the confusion as
a lack of judgement about audience.

## 9. Industry practice and case evidence

### What happens when documentation is missing

Wong and colleagues' 2021 external validation of a proprietary sepsis model is the canonical case.
The model had been widely implemented on the strength of vendor-reported performance, and the
independent validation found an AUROC of 0.63 against reported values of 0.76 to 0.83, and failure
to identify 67 percent of sepsis patients. The study is single-site and retrospective, and its
importance is that the gap was detectable only because an independent team could define the outcome
and access the data.

The documentation lesson is not that the vendor lied. It is that a performance claim without a
documented cohort, outcome definition, and validation design cannot be checked, and an unchecked
claim propagates.

The same failure appears at a smaller scale in course projects. A team reports an AUC in a slide,
the repository does not contain the script that produced it, and the number cannot be reproduced.
Nothing dishonest has happened, and the artefact is still unusable. Documentation is the difference
between a result and a claim.

### What regulators ask for

The FDA's PCCP guidance requires three components: a description of modifications, a modification
protocol with data management, retraining, evaluation, and acceptance criteria, and an impact
assessment covering benefits, risks, and unintended bias. Those three components are a dossier in
miniature, and a capstone that writes them for a hypothetical update has done the exercise that a
regulatory reviewer would expect.

The Royal College of Radiologists' 2025 consultation document on AI deployment fundamentals notes
that device marking does not establish clinical or cost-effectiveness and that independent
evaluation is scarce. A dossier that separates marking from evaluation is more credible than one
that treats them as the same.

### What evaluation programmes require

NHS England's guidance on real-world AI evaluations, drawn from the AI in Health and Care Award
portfolio, describes the artefacts that evaluations need: a named owner, a data plan, an evaluation
design, and a plan for generating evidence during deployment. The guidance is practitioner
material, and it matches the dossier structure in this topic closely enough to be a useful external
check.

NICE early value assessment evidence-generation plans add a formal gap table: what is known, what is
not, and what study would fill the gap. A capstone's limitations section is a smaller version of the
same table, and a team that formats it that way makes the reviewer's job easier.

### What public challenges demonstrate about write-ups

Challenge winners publish method write-ups that describe the pipeline, the preprocessing, and the
ensemble. They rarely publish calibration, subgroup performance, or a monitoring plan, because the
competition does not require them. A capstone that copies the write-up format will produce a method
description rather than a dossier, and the panel will notice the missing evidence.

The MedGemma Impact Challenge winners are prototypes with clear users and narrow scope, reported as
prototypes. That is the correct level of claim for a short project, and it is the level a capstone
should adopt when the evidence does not support more.

### Inference, labelled

It is an inference that teams which run a stranger test before freeze produce more reproducible
dossiers. The mechanism is that the test exposes undocumented assumptions, and the repository is the
only artefact that can be tested this way. No controlled study of capstone teams is cited.

### What an employer checks first

Three artefacts are checked before a hiring manager reads the method.

The README, because it says whether the repository can be run. A README with no install step and no
example command signals that the author has not handed over work before.
The model card, because it says whether the author understands limits. A model card with an empty
limitations section is worse than no model card, because it shows the question was not considered.
The validation report, because it says whether the author knows what the numbers mean. A report
that gives an AUC with no interval, no split definition, and no calibration tells the reader that
the author has not been reviewed by a statistician.

The dossier's weights are similar, and for the same reason: the artefacts that survive handover are
the ones that answer a stranger's questions.

### What goes wrong in handover

A model handed over without a monitoring plan is a model that will drift silently. The receiving
team has no signal until someone notices a change in outcomes, and by then the model has been wrong
for months.

A model handed over without a data provenance record cannot be audited. When a regulator asks which
patients were in the training set, the answer is a query that nobody wrote down.

A model handed over without an owner is retired at the next reorganisation, regardless of how well
it performed. The dossier cannot fix that, and it can make the gap visible by naming the absent
owner in the limitations.

### Inference, labelled (continued)

The handover failures above are documented in the evaluation literature rather than in controlled
studies. The NHS England guidance and the Royal College of Radiologists consultation describe the
same pattern from different vantage points. The inference that a complete dossier reduces handover
failure is plausible and not experimentally established.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Dossier assembled last | No artefact files in week 8 | Writing treated as a final task | Contradictions and gaps | Assemble incrementally from week 6 |
| Numbers not traceable | Slides differ from repository | Manual transcription | Credibility loss | Generate figures and tables by script |
| Reproducibility fails | Stranger cannot run the command | Unpinned environment | Invalid evidence | Container, lockfile, tested command |
| Model card empty of limits | No limitations section | Not looking at errors | Overclaim | Write limits from the error analysis |
| Prototype called a product | Pitch says "deploy" | Ambition | Panel asks about maintenance | Label the maturity level |
| Demo depends on network | Live service in the demo | Convenience | Public failure | Offline fixture and recording |
| Regulatory memo contradicts pitch | Memo says research, slides say clinical | Separate authorship | Claim exceeds authority | Memo wins; rewrite the pitch |
| No monitoring plan | "We will monitor it" | Deployment treated as the end | Handover fails | Signals, thresholds, response |
| Value case without assumptions | Single cost number | Precision theatre | Unbelievable economics | Ranges and break-even |
| Missing acceptance tests | "It looks good" | No criteria | Self-assessment meaningless | Test per artefact |
| Self-assessment inflated | All artefacts marked pass | Optimism | Gaps discovered at defense | Stranger test and evidence per pass |
| Freeze ignored | Changes after the tag | Habit | Inconsistent artefacts | Named enforcer and re-tag rule |
| Individual defense gap | One member cannot explain the memo | Siloing | Score multiplier | Handoff notes and rotation |

The table is ordered from assembly failures to handover failures. The first six are visible in week
9; the last four appear at freeze or after.

Two failures are worth reading twice. "Numbers not traceable" is the most common and the most
damaging, because it converts a small discrepancy into a question about the whole dossier.
"Prototype called a product" is the one that changes the regulatory analysis, because a product
claim implies a device pathway that a prototype does not.

The mitigation column is written as an artefact or a routine, not as an attitude. "Be careful with
numbers" is not a mitigation; "generate every figure from results/metrics.json" is.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The repository is the source of truth | Consistency depends on it | Slides edited by hand | Compare every number with the file |
| A stranger can run the code | Acceptance criterion | Undocumented dependencies | Stranger test |
| The data can be described without sharing it | Provenance artefact | Confidentiality agreement | Tier and custodian statement |
| The model card can be written from the artefacts | Two-hour task | Missing metrics | Draft it in week 9 |
| The panel reads the dossier | Weights assume it | Time pressure | Put the evidence table first |
| The regulatory memo is reviewable in ten minutes | Mentor time | Too much detail | One page and three questions |
| The demo can be recorded | Backup plan | No working prototype | Record in week 10 |
| The self-assessment is honest | It is a planning tool | Optimism | Require evidence per pass |
| The reviewer reads in the intended order | Ordering is a design choice | Sampling from the appendix | Put the evidence table first |
| The freeze can be enforced | Consistency depends on it | Continued editing | Named enforcer and re-tag |

The last row is organisational. A freeze that nobody enforces is a suggestion, and the dossier will
drift by a number or a sentence in the final days.

The largest limit is that the dossier cannot exceed the evidence. A well-formatted dossier with a
weak validation is still a weak capstone, and the self-assessment measures completeness rather than
quality.

A second limit is that the weights are the programme's, not a universal standard. A regulator would
weight differently, and a commercial reviewer would weight the value case more heavily. The weights
are a teaching device that encodes what the panel rewards.

A third limit is that the stranger test depends on the stranger. A colleague who has read the
charter is not a stranger, and a colleague who is unfamiliar with the domain may fail for reasons
that do not matter to a reviewer. The test is a lower bound on usability, not a proof of it.

A fourth limit is time. A team with ten working days cannot close four blocking gaps and build a
demo. The plan should have started in week 6, and a team that starts in week 9 should expect to
freeze with documented gaps rather than to finish everything.

A fifth limit is that a dossier is written for a reader who reads. Many reviewers sample, and a
perfect appendix does not help if the evidence table is on page 30. The ordering of artefacts in the
dossier is therefore part of the design: decision, evidence, validation, errors, governance,
adoption, in that order.

## 12. Exercises

### Exercise 1. Map artefacts to readers (recall, 25 minutes)

For each of the thirteen artefacts, name the reader who would check it first and the one question
they would ask.
**Deliverable:** a thirteen-row table.
**Criteria:** every artefact has a reader and a question; at least three artefacts have two readers.
**Time:** 25 minutes.

### Exercise 2. Write an acceptance criterion (application, 30 minutes)

Write an acceptance criterion for the validation report and the demo that a reviewer can apply in
under ten minutes.
**Deliverable:** two criteria.
**Criteria:** each names an observable action and an observable result; neither uses the word
"clear" or "thorough".
**Time:** 30 minutes.

### Exercise 3. Run the self-assessment (calculation, 30 minutes)

Score your own dossier against the thirteen artefacts, marking present and passing separately, and
compute the weighted completeness.
**Deliverable:** the table and the score.
**Criteria:** every fail has evidence; blocking gaps ranked by weight; shortfall computed.
**Time:** 30 minutes.

### Exercise 4. Build the reproducibility package (application, 60 minutes)

Pin the environment, add a one-command run, and test it on a clean machine or container.
**Deliverable:** the package and a transcript of the clean run.
**Criteria:** the command runs without manual steps; the metric matches the reported value within
the stated tolerance; the transcript is included.
**Time:** 60 minutes.

### Exercise 5. Write the model card (design, 45 minutes)

Write a model card for your capstone with intended use, users, training data, evaluation data,
metrics by subgroup, limitations, and ethical considerations.
**Deliverable:** one to two pages.
**Criteria:** every metric has an interval; the limitations section has at least three specific
items; the intended use names a user.
**Time:** 45 minutes.

### Exercise 6. Run the stranger test (critique, 40 minutes)

Ask someone outside the team to find the intended use, reproduce the primary metric, and explain the
regulatory path, and time each task.
**Deliverable:** a timed record with the points of failure.
**Criteria:** three tasks timed; failures specific; fixes listed.
**Time:** 40 minutes.

### Exercise 7. Audit consistency (calculation, 30 minutes)

Compare every number in the readout and slides with `results/metrics.json`, and record any
difference.
**Deliverable:** a discrepancy table with zero rows, or a fix list.
**Criteria:** every number checked; differences explained; fixes applied.
**Time:** 30 minutes.

### Exercise 8. Rehearse the individual defense (design, 30 minutes)

Each member answers one question on the decision, one on the pipeline, and one on governance, chosen
by another member.
**Deliverable:** a recording or notes and a list of gaps.
**Criteria:** answers cite an artefact; gaps are assigned to a backup; at least one answer
acknowledges a limitation.
**Time:** 30 minutes.

**Assessment note.** Exercises 1 to 3 and 7 have checkable outputs. Exercises 4 to 6 and 8 are
assessed on the record: whether the acceptance criterion was tested, whether the failure was
documented, and whether the fix has an owner.

Exercise 3 is the one that changes behaviour. Most teams discover that the artefact they considered
finished does not pass its own acceptance test, and the discovery is cheaper in the exercise than at
freeze.

Exercise 4 is the one that takes longest and matters most. A reproducibility package that a stranger
can run is the difference between a result and an anecdote, and it is the artefact most often
deferred until there is no time left.

## 13. Self-check answers

**1. Why is the validation report weighted most heavily?**
Because it carries the claim. A dossier's other artefacts support the validation; if the validation
is missing or wrong, the claim is unsupported regardless of the quality of the readout.

**2. What is the difference between present and passing?**
Present means the file exists; passing means it meets its acceptance test. A validation report
without a calibration slope is present and failing, and the self-assessment must record both.

**3. Compute the reviewer time for the thirteen artefacts.**
The minutes sum to 116, which is 1.9 hours. A reviewer with two hours will read the charter, the
validation report, and the results, and sample the rest.

**4. Why does the demo come after the dossier?**
Because the demo must be consistent with the dossier's numbers and claims. Building it first invites
a demo that shows something the dossier cannot support.

**5. What should a team do if the data cannot be shared?**
Ship code, a data-access statement, and a synthetic sample, and state that the real-data metric
cannot be reproduced without access. The acceptance criterion changes; the honesty does not.

**6. Why is the stranger test more reliable than the team's own check?**
Because the team knows where everything is and cannot simulate ignorance. A stranger's failure is
the reviewer's failure, and it is cheaper to find in week 10 than at the defense.

**7. What is the correct label for a prototype in a pitch?**
A prototype that demonstrates one step on held-out or synthetic data, with the limits stated. Not a
product, not a deployment, and not a clinical tool.

**8. How should a post-freeze correction be handled?**
Correct it, re-tag, and announce it on stage. A disclosed correction costs less than a discovered
inconsistency, and the panel treats it as evidence of control.

**9. What is the difference between a model card and a datasheet?**
A model card describes the model, its intended use, and its performance; a datasheet describes the
dataset, its composition, and its recommended uses. Both are needed because the model's limits
depend on both.

**10. Rubric for Exercise 4.** A strong package runs from a clean environment with one command,
records the environment, and reproduces the metric within a stated tolerance. A weak package
documents the steps a team member took, which is a recipe rather than a package.

**11. Rubric for Exercise 6.** A strong stranger test records three timed tasks, the exact point of
failure, and a fix with an owner. A weak test records that the stranger "found it difficult" without
saying where.

**12. What should a team do if the stranger test fails at the install step?**
Fix the install step and re-test before anything else. An install failure makes every other
artefact unverifiable, because the reviewer cannot reach them.

**13. Why is the readout weighted only 2 percent?**
Because it is a summary of artefacts that are weighted more heavily. A good readout cannot
compensate for a missing validation report, and a weak readout with strong artefacts is a small
loss.

**14. What makes a demonstration a failure case rather than a failure?**
The demonstration shows a known limitation with the evidence that lets a user reject the output. A
failure is an unexpected error that the team cannot explain.

## 14. Interview questions

**Q1. "Walk me through your dossier."**
Strong answer follows the evidence: decision, data, validation, errors, governance, adoption. Weak
answer lists files.

**Q2. "Can you reproduce your result?"**
Strong answer describes the command, the environment, and the tolerance. Weak answer says the code
is on GitHub.

**Q3. "What does your model card say you should not use this for?"**
Strong answer names specific exclusions. Weak answer says the limitations are in the report.

**Q4. "How did you know the demo would work?"**
Strong answer describes the offline fixture, the recording, and the rehearsal. Weak answer says it
worked in testing.

**Q5. "What would a regulator ask for that you do not have?"**
Strong answer names clinical evidence, a quality system, or a PCCP. Weak answer says nothing.

**Q6. "Which artefact was hardest to write, and why?"**
Strong answer names the validation report or the model card and explains the missing evidence. Weak
answer says the readout.

**Q7. "How did you check consistency across artefacts?"**
Strong answer describes generating numbers from one file and comparing. Weak answer says the team
reviewed the slides.

**Q8. "What is the maturity level of your prototype?"**
Strong answer places it on a scale and names what is missing for the next level. Weak answer calls
it production-ready.

**Q9. "What would you do with two more weeks?"**
Strong answer names the weakest artefact and the evidence it would add. Weak answer says improve
the model.

**Q10. "How would you hand this over?"**
Strong answer describes the repository, the model card, the monitoring plan, and the owner. Weak
answer says send the code.

A strong answer to this set always names an artefact. The candidate who says "the model card says
we should not use it for pediatric patients" is demonstrating that the artefact exists and that it
was read; the candidate who says "we were careful about limitations" is demonstrating neither.

## 15. Key terms

The terms below are the vocabulary a reviewer uses when checking a dossier. A team that can use
them precisely is easier to review, and a team that cannot is asked more questions.

- **Acceptance criterion** — observable test of an artefact.
- **Artefact** — one document or package in the dossier.
- **Audit trail** — record of who changed what and when.
- **Blocking gap** — missing artefact that fails the defense.
- **Datasheet** — structured dataset description.
- **Defense** — oral examination on any part of the work.
- **Deliverable** — versioned artefact with a test.
- **Demo** — timed demonstration on held-out data.
- **Dossier** — thirteen-artefact evidence record.
- **Evidence table** — comparator and candidate with intervals.
- **Freeze** — point after which only corrections are allowed.
- **Intended use** — population, setting, inputs, output, user, limits.
- **Model card** — structured model description.
- **PCCP** — predetermined change control plan.
- **Provenance** — source, version, licence, transformations.
- **Prototype** — working artefact demonstrating one step.
- **Readout** — one-page decision summary.
- **Reproducibility package** — code, environment, seeds, command.
- **Risk register** — scored risks with owners and triggers.
- **Self-assessment** — structured check against criteria.
- **Stranger test** — outsider reproduces and explains the work.
- **Tolerance** — acceptable difference in reproduced metrics.
- **TRIPOD+AI** — reporting guideline for prediction models.
- **Value case** — cost, budget impact, adoption assumptions.
- **Version tag** — identifier of the frozen artefact set.
- **Claim boundary** — the sentence the evidence supports and the one it does not.
- **Consistency check** — comparison of quoted numbers with the source file.
- **Handover package** — repository, model card, monitoring plan, named owner.

The vocabulary is deliberately borrowed from industry handover documents rather than from academic
reports. A team that can use these terms in an interview sounds like someone who has handed work to
another team, which is the impression the dossier is designed to create.

## 16. Sources

The instruments below are cited as they apply at the access date. Reporting guidelines are
conventions; the FDA and EU instruments are binding only within their jurisdictions and only for
the claims they govern.

All URLs accessed 26 October 2026.

- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  https://doi.org/10.1136/bmj-2023-078378
- Vasey B, Nagendran M, Campbell B, et al. DECIDE-AI. *Nature Medicine* 2022;28:924-933.
  https://doi.org/10.1038/s41591-022-01772-9
- FDA, *Marketing Submission Recommendations for a Predetermined Change Control Plan for
  AI-Enabled Device Software Functions*, December 2024. https://www.fda.gov/media/185050/download
- FDA, *Clinical Decision Support Software*, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- Regulation (EU) 2017/745, Medical Devices Regulation.
  https://eur-lex.europa.eu/eli/reg/2017/745/oj
- Regulation (EU) 2024/1689, Artificial Intelligence Act.
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- ISO 14971, risk management for medical devices. https://www.iso.org/standard/72704.html
- 21 CFR Part 11. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
  sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
  https://doi.org/10.1001/jamainternmed.2021.2626
- NHS England, Planning and implementing real-world AI evaluations.
  https://www.england.nhs.uk/long-read/planning-and-implementing-real-world-ai-evaluations-lessons-from-the-ai-in-health-and-care-award/
- Royal College of Radiologists, *AI deployment fundamentals*, 2025.
  https://www.rcr.ac.uk/media/r0th2x0k/ai-deployment-fundamentals_public-consultation.pdf
- NICE, Evidence generation plan, vertebral fragility fractures, 2025.
  https://www.nice.org.uk/consultations/2785/2/approach-to-research
- Mitchell M, Wu S, Zaldivar A, et al. Model cards for model reporting. *FAT* 2019.
  https://doi.org/10.1145/3287560.3287596
- Gebru T, Morgenstern J, Vecchione B, et al. Datasheets for datasets. *Communications of the ACM*
  2021;64(12):86-92. https://doi.org/10.1145/3458723

Source note: model cards and datasheets are documented practice rather than regulatory requirements.
The FDA instruments cited constrain claims and change control; they do not prescribe a dossier
format. The dossier structure in this topic is the programme's, informed by those instruments and
by the reporting guidelines listed.

## 17. Links

The dossier's artefacts are scored in Topic 10, and the readout and demo are delivered in Topic 11.
A team that has completed the self-assessment in this topic has already done half the preparation
for the defense.

The remaining half is judgement about what to leave out, which is the subject of the judging rubric.

A dossier is finished when a stranger can answer the panel's first three questions without the
team's help, and not before.

That test is deliberately behavioural. It does not ask whether the team feels ready, and it does not
ask whether every artefact exists. It asks whether the record works for someone who was not there,
which is the only property that survives the team.

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_scoping-playbook.md`](08_scoping-playbook.md)
- Next topic: [`10_judging.md`](10_judging.md)
- Related topic: [`11_demo-day.md`](11_demo-day.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
